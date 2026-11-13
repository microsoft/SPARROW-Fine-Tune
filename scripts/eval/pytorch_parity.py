"""Parity check: the service's conventional classifier versus a plain PyTorch fine-tuning loop.

Both pipelines train the same timm backbone on the same crops, with the same train/val partition
(the service's camera-grouped split for that seed), the same epoch budget, batch size, learning
rates and, with ``--recipe matched`` (the parity check), the same recipe: crop scale 0.8-1, two
head-only warm-up epochs, early stopping with patience 5. ``--recipe textbook`` uses torchvision
defaults instead (crop scale 0.08-1, no warm-up, no early stopping) as context. Both are tested on the same held-out ``val_cropped`` crops. The service model is scored from
its exported ONNX bundle (what a user deploys); the reference is a textbook torchvision/timm loop.

Equivalence criterion, fixed before the runs: over ``--seeds`` seeds, the mean test accuracy of the
two pipelines differs by at most 2.0 points and the mean macro-F1 by at most 0.03. A same-seed rerun
of the service pipeline must reproduce its test predictions exactly.

    uv run --extra dev python scripts/eval/pytorch_parity.py --side ours --device cuda:0 ...
    uv run --extra dev python scripts/eval/pytorch_parity.py --side reference --device cuda:1 ...
    uv run --extra dev python scripts/eval/pytorch_parity.py --side report ...
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))

from finetune_quality import (  # noqa: E402
    MIN_TRAIN_PER_CLASS,
    conventional_scores,
    experiments,
    load_rows,
    macro_f1,
    run_trainer,
)
from sparrow_finetune.pipelines.conventional_classifier import (  # noqa: E402
    ConventionalClassifierTrainer,
)
from sparrow_finetune.pipelines.dinov3_linear import assign_split  # noqa: E402

EXPERIMENT = "few-shot-200"
ACC_MARGIN, F1_MARGIN = 0.02, 0.03


def dataset(data: Path):
    rows = load_rows(data)
    for name, train, test, _desc in experiments(rows, seed=0):
        if name == EXPERIMENT:
            break
    counts = Counter(r["label"] for r in train)
    labels = sorted(lbl for lbl, n in counts.items() if n >= MIN_TRAIN_PER_CLASS)
    index = {lbl: i for i, lbl in enumerate(labels)}
    train = [r for r in train if r["label"] in index]
    test = [r for r in test if r["label"] in index]
    return train, test, labels, index


def scores(y, pred) -> dict:
    return {"accuracy": float(np.mean(pred == y)), "macro_f1": macro_f1(y, pred)}


# ---------------------------------------------------------------- reference


class _Crops:
    """Map-style dataset at module level so DataLoader workers can pickle it."""

    def __init__(self, rows, tf, index):
        self.rows, self.tf, self.index = rows, tf, index

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        from PIL import Image

        r = self.rows[i]
        with Image.open(r["path"]) as im:
            return self.tf(im.convert("RGB")), self.index[r["label"]]


def reference_run(train, test, index, seed: int, args) -> tuple[np.ndarray, dict]:
    import timm
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms as T

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(args.device)
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    matched = args.recipe == "matched"
    # matched: the service recipe (crop scale 0.8-1, 2 head-only warm-up epochs, patience 5);
    # textbook: torchvision defaults (crop scale 0.08-1, no warm-up, no early stopping).
    scale = (0.8, 1.0) if matched else (0.08, 1.0)
    warmup = 2 if matched else 0
    patience = 5 if matched else 0
    train_tf = T.Compose(
        [
            T.RandomResizedCrop(224, scale=scale),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    eval_tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean, std)])

    def Crops(rows, tf):  # noqa: N802
        return _Crops(rows, tf, index)

    # The same camera-grouped partition the service draws for this seed.
    side = {
        r["camera"]: assign_split(r["camera"], seed=seed, val_fraction=0.2)
        for r in train
    }
    fit = [r for r in train if side[r["camera"]] == "train"]
    val = [r for r in train if side[r["camera"]] == "val"]
    g = torch.Generator().manual_seed(seed)
    fit_dl = DataLoader(
        Crops(fit, train_tf),
        batch_size=64,
        shuffle=True,
        num_workers=16,
        generator=g,
        drop_last=False,
        persistent_workers=True,
    )

    def predict(rows) -> np.ndarray:
        dl = DataLoader(Crops(rows, eval_tf), batch_size=128, num_workers=16)
        out = []
        model.eval()
        with torch.no_grad():
            for x, _ in dl:
                out.append(model(x.to(device)).argmax(1).cpu().numpy())
        return np.concatenate(out)

    model = timm.create_model(
        "resnet50.a1_in1k", pretrained=True, num_classes=len(index)
    ).to(device)
    head = list(model.get_classifier().parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    crit = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda")
    y_val = np.array([index[r["label"]] for r in val])
    best_f1, best_state, curve, since_best = -1.0, None, [], 0
    if warmup:
        for p in backbone:
            p.requires_grad_(False)
        opt = torch.optim.AdamW(head, lr=1e-3, weight_decay=0.05)
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        if epoch == warmup + 1:
            for p in backbone:
                p.requires_grad_(True)
            opt = torch.optim.AdamW(
                [{"params": backbone, "lr": 1e-4}, {"params": head, "lr": 1e-3}],
                weight_decay=0.05,
            )
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=args.epochs - warmup
            )
        if epoch <= warmup:
            model.eval()  # frozen backbone keeps its BatchNorm statistics
        else:
            model.train()
        for x, y in fit_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        if epoch > warmup:
            sched.step()
        f1 = macro_f1(y_val, predict(val))
        curve.append({"epoch": epoch, "val_macro_f1": f1})
        print(f"  ref seed {seed} epoch {epoch} val_f1 {f1:.4f}", flush=True)
        if f1 > best_f1:
            best_f1, since_best = f1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            if epoch > warmup:
                since_best += 1
            if patience and epoch > warmup and since_best >= patience:
                break
    model.load_state_dict(best_state)
    pred = predict(test)
    return pred, {
        "fit_crops": len(fit),
        "val_crops": len(val),
        "best_val_macro_f1": best_f1,
        "curve": curve,
        "seconds": time.time() - t0,
    }


# ---------------------------------------------------------------- service


def ours_run(train, test, labels, seed: int, args, tag: str):
    bundle, val, seconds = run_trainer(
        ConventionalClassifierTrainer(),
        {
            "model": "resnet50.a1_in1k",
            "seed": seed,
            "epochs": args.epochs,
            "batch": 64,
            "workers": 16,
            "device": args.device,
        },
        train,
        labels,
        args.work / f"ours-{tag}",
    )
    pred = conventional_scores(bundle, test).argmax(axis=1)
    return pred, {"inner_val": val, "seconds": seconds}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--side", choices=("ours", "reference", "report"), required=True)
    ap.add_argument(
        "--data", type=Path, default=Path.home() / "datasets/amazon_camera_trap"
    )
    ap.add_argument("--work", type=Path, default=Path("/tmp/ft_parity"))
    ap.add_argument("--out", type=Path, default=ROOT / "docs/review/pytorch-parity")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument(
        "--recipe",
        choices=("matched", "textbook"),
        default="matched",
        help="reference recipe: the service's (parity) or torchvision defaults (context)",
    )
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.side == "report":
        return report(args)

    train, test, labels, index = dataset(args.data)
    y = np.array([index[r["label"]] for r in test])
    print(
        f"{EXPERIMENT}: {len(train)} train / {len(test)} test, {len(labels)} classes",
        flush=True,
    )
    name = args.side if args.side == "ours" else f"reference-{args.recipe}"
    out_file = args.work / f"{name}.json"
    results = json.loads(out_file.read_text()) if out_file.exists() else {}
    jobs = [str(s) for s in range(args.seeds)]
    if args.side == "ours":
        jobs.append("0-rerun")
    for tag in jobs:
        if tag in results:
            continue
        seed = int(tag.split("-")[0])
        if args.side == "ours":
            pred, info = ours_run(train, test, labels, seed, args, tag)
        else:
            pred, info = reference_run(train, test, index, seed, args)
        results[tag] = {"seed": seed, **scores(y, pred), **info, "pred": pred.tolist()}
        out_file.write_text(json.dumps(results, default=float))
        print(f"{args.side} {tag}: {scores(y, pred)}", flush=True)
    return 0


def report(args) -> int:
    ours = json.loads((args.work / "ours.json").read_text())
    summary = {}
    for recipe in ("matched", "textbook"):
        path = args.work / f"reference-{recipe}.json"
        if path.exists():
            summary[recipe] = compare(ours, json.loads(path.read_text()))
    summary["pass"] = summary.get("matched", {}).get("pass", False)
    (args.out / "result.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


def compare(ours: dict, ref: dict) -> dict:
    seeds = sorted({k for k in ours if "-" not in k} & set(ref), key=int)
    rows = []
    for s in seeds:
        rows.append(
            {
                "seed": int(s),
                "ours_accuracy": ours[s]["accuracy"],
                "ref_accuracy": ref[s]["accuracy"],
                "ours_macro_f1": ours[s]["macro_f1"],
                "ref_macro_f1": ref[s]["macro_f1"],
                "ours_seconds": ours[s]["seconds"],
                "ref_seconds": ref[s]["seconds"],
            }
        )

    def stat(key):
        v = np.array([r[key] for r in rows])
        return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0

    d_acc = np.array([r["ours_accuracy"] - r["ref_accuracy"] for r in rows])
    d_f1 = np.array([r["ours_macro_f1"] - r["ref_macro_f1"] for r in rows])

    def ci(d):
        if len(d) < 2:
            return [float(d.mean())] * 2
        half = (
            2.776 * d.std(ddof=1) / math.sqrt(len(d))
            if len(d) == 5
            else 1.96 * d.std(ddof=1) / math.sqrt(len(d))
        )
        return [float(d.mean() - half), float(d.mean() + half)]

    rerun = ours.get("0-rerun")
    same = rerun is not None and rerun["pred"] == ours["0"]["pred"]
    summary = {
        "experiment": EXPERIMENT,
        "seeds": len(rows),
        "per_seed": rows,
        "ours_accuracy_mean_sd": stat("ours_accuracy"),
        "ref_accuracy_mean_sd": stat("ref_accuracy"),
        "ours_macro_f1_mean_sd": stat("ours_macro_f1"),
        "ref_macro_f1_mean_sd": stat("ref_macro_f1"),
        "accuracy_diff_mean": float(d_acc.mean()),
        "accuracy_diff_ci95": ci(d_acc),
        "macro_f1_diff_mean": float(d_f1.mean()),
        "macro_f1_diff_ci95": ci(d_f1),
        "same_seed_rerun_identical_predictions": same,
        "criterion": f"|mean acc diff| <= {ACC_MARGIN}, |mean macro-F1 diff| <= {F1_MARGIN}, rerun identical",
        "pass": bool(
            abs(d_acc.mean()) <= ACC_MARGIN and abs(d_f1.mean()) <= F1_MARGIN and same
        ),
    }
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
