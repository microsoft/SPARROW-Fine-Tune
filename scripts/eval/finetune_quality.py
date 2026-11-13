"""Fine-tune quality versus the stock AI4G-Amazon-V2 classifier on the Amazon camera-trap crops.

The service trainers run in-process (the same ``prepare -> train -> evaluate -> export`` code the
workers run); every model is scored from its exported ONNX bundle, and the stock model from its
Engine ONNX, on the same test crops. Differences carry a paired bootstrap 95% interval that
resamples camera-trap sequences (``site/camera/Seq_ID``), not single crops.

AI4G-Amazon-V2 was trained on this dataset (its 36 labels are this dataset's labels), so only the
``same-data`` experiment compares like with like. ``new-site`` and ``few-shot`` measure the
service pipelines; the AI4G number there is reported for reference and favours AI4G.

    uv run --extra dev python scripts/eval/finetune_quality.py \
        --data ~/datasets/amazon_camera_trap \
        --ai4g /path/to/AI4G-Amazon-V2 --dinov3 /path/to/dinov3_vits16.onnx \
        --work /tmp/ft_quality --out docs/review/finetune-quality
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sparrow_finetune.embedders import Embedder, OnnxEmbedder  # noqa: E402
from sparrow_finetune.pipelines import JobContext, SnapshotItem  # noqa: E402
from sparrow_finetune.pipelines.conventional_classifier import (  # noqa: E402
    ConventionalClassifierTrainer,
)
from sparrow_finetune.pipelines.dinov3_linear import DinoV3LinearTrainer  # noqa: E402

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
BOOTSTRAP = 1000
MIN_TRAIN_PER_CLASS = 10


# ---------------------------------------------------------------- data


def load_rows(data: Path) -> dict[str, list[dict]]:
    out = {}
    for side in ("train", "val"):
        rows = []
        with open(data / "annotations" / f"{side}_cropped.csv") as fh:
            for r in csv.DictReader(fh):
                site, camera, _file = r["Photo_ID"].split("/", 2)
                rows.append(
                    {
                        "path": data / "cropped_resized" / r["Photo_ID"],
                        "photo": r["Photo_ID"],
                        "label": r["New_Label"],
                        "site": site,
                        "camera": f"{site}/{camera}",
                        "seq": f"{site}/{camera}/{r['Seq_ID']}",
                    }
                )
        out[side] = rows
    return out


def pixel_key(im: Image.Image) -> str:
    return hashlib.sha1(np.asarray(im.convert("RGB")).tobytes()).hexdigest()


class CachedEmbedder(Embedder):
    """Looks features up by pixel hash; the cache holds every dataset crop, embedded once."""

    def __init__(self, inner: OnnxEmbedder, cache: dict[str, np.ndarray]) -> None:
        self.inner, self.cache = inner, cache
        self.id, self.dim, self.model_hash = inner.id, inner.dim, inner.model_hash
        self.input_size, self.mean, self.std = inner.input_size, inner.mean, inner.std

    def embed(self, images):
        return np.stack([self.cache[pixel_key(im)] for im in images]).astype(np.float32)


def embed_all(rows: list[dict], embedder: OnnxEmbedder, cache_file: Path):
    if not cache_file.exists():
        keys, chunks, t0 = [], [], time.time()
        paths = sorted({str(r["path"]) for r in rows})
        for start in range(0, len(paths), 256):
            ims = [Image.open(p).convert("RGB") for p in paths[start : start + 256]]
            keys += [pixel_key(im) for im in ims]
            chunks.append(embedder.embed(ims))
            print(
                f"  embedded {start + len(ims)}/{len(paths)} ({time.time() - t0:.0f}s)",
                flush=True,
            )
        np.savez(
            cache_file,
            keys=np.array(keys),
            feats=np.concatenate(chunks),
            paths=np.array(paths),
        )
    z = np.load(cache_file, allow_pickle=False)
    return dict(zip(z["keys"], z["feats"])), dict(zip(z["paths"], z["feats"]))


# ---------------------------------------------------------------- models


def load_image_batch(rows, size: int) -> np.ndarray:
    arr = [
        np.asarray(
            Image.open(r["path"]).convert("RGB").resize((size, size), Image.BILINEAR),
            dtype=np.float32,
        )
        for r in rows
    ]
    return (np.stack(arr) / 255.0).transpose(0, 3, 1, 2)


def onnx_probs(
    session, rows, size: int, *, normalize: bool, batch: int = 128
) -> np.ndarray:
    name = session.get_inputs()[0].name
    out = []
    for start in range(0, len(rows), batch):
        x = load_image_batch(rows[start : start + batch], size)
        if normalize:
            x = (x - IMAGENET_MEAN) / IMAGENET_STD
        out.append(np.asarray(session.run(None, {name: x.astype(np.float32)})[0]))
    return np.concatenate(out)


def ai4g_scores(ai4g: Path, rows) -> tuple[np.ndarray, list[str]]:
    import onnxruntime as ort

    labels = [
        line.split(",")[0]
        for line in (ai4g / "labels.txt").read_text().splitlines()
        if line.strip()
    ]
    session = ort.InferenceSession(
        str(ai4g / "1" / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    return onnx_probs(session, rows, 224, normalize=True), labels


def run_trainer(trainer, params: dict, rows, labels: list[str], workdir: Path):
    workdir.mkdir(parents=True, exist_ok=True)
    items = [
        SnapshotItem(
            image_path=r["path"],
            width=256,
            height=256,
            annotations=[{"label": r["label"], "bbox": None}],
            group=r["camera"],
            name=r["photo"],
            sha256="",
        )
        for r in rows
    ]
    log = []
    ctx = JobContext(
        job_id=0,
        params=trainer.validate_params(params),
        workdir=workdir,
        items=items,
        labels=labels,
        _emit=lambda stage, msg, m: log.append({"stage": stage, "message": msg, **m}),
        _check_cancelled=lambda: None,
    )
    t0 = time.time()
    trainer.prepare(ctx)
    trainer.train(ctx)
    val_metrics = trainer.evaluate(ctx)
    result = trainer.export(ctx)
    (workdir / "events.json").write_text(json.dumps(log, default=float))
    return result.path, val_metrics, time.time() - t0


def conventional_scores(bundle: Path, rows) -> np.ndarray:
    import onnxruntime as ort

    with zipfile.ZipFile(bundle) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        session = ort.InferenceSession(
            zf.read("model.onnx"), providers=["CPUExecutionProvider"]
        )
    # The exported graph normalizes internally; it takes RGB in [0, 1].
    return onnx_probs(session, rows, manifest["input"]["size"], normalize=False)


def dinov3_scores(bundle: Path, rows, feats_by_path) -> np.ndarray:
    import onnxruntime as ort

    with zipfile.ZipFile(bundle) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        session = ort.InferenceSession(
            zf.read("head.onnx"), providers=["CPUExecutionProvider"]
        )
    x = np.stack([feats_by_path[str(r["path"])] for r in rows]).astype(np.float32)
    return np.asarray(
        session.run(
            [manifest["head"]["probabilities"]], {manifest["head"]["input"]: x}
        )[0]
    )


# ---------------------------------------------------------------- metrics


def macro_f1(y, pred) -> float:
    f1s = []
    for c in np.unique(y):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def summarize(
    y, preds: dict[str, np.ndarray], clusters: list[str], seed: int = 0
) -> dict:
    names = list(preds)
    out: dict = {
        name: {"accuracy": float(np.mean(p == y)), "macro_f1": macro_f1(y, p)}
        for name, p in preds.items()
    }
    ids = defaultdict(list)
    for i, c in enumerate(clusters):
        ids[c].append(i)
    groups = [np.array(v) for v in ids.values()]
    rng = np.random.default_rng(seed)
    boot = {name: {"accuracy": [], "macro_f1": []} for name in names}
    for _ in range(BOOTSTRAP):
        idx = np.concatenate(
            [groups[i] for i in rng.integers(0, len(groups), len(groups))]
        )
        for name in names:
            boot[name]["accuracy"].append(np.mean(preds[name][idx] == y[idx]))
            boot[name]["macro_f1"].append(macro_f1(y[idx], preds[name][idx]))
    for name in names:
        for metric in ("accuracy", "macro_f1"):
            lo, hi = np.percentile(boot[name][metric], [2.5, 97.5])
            out[name][f"{metric}_ci95"] = [float(lo), float(hi)]
            if name != "ai4g":
                diff = np.array(boot[name][metric]) - np.array(boot["ai4g"][metric])
                lo, hi = np.percentile(diff, [2.5, 97.5])
                out[name][f"{metric}_minus_ai4g"] = (
                    out[name][metric] - out["ai4g"][metric]
                )
                out[name][f"{metric}_minus_ai4g_ci95"] = [float(lo), float(hi)]
    out["test_crops"], out["test_sequences"] = int(len(y)), len(groups)
    return out


def per_class_recall(y, pred, labels):
    return {
        labels[c]: {
            "support": int(np.sum(y == c)),
            "recall": float(np.mean(pred[y == c] == c)),
        }
        for c in np.unique(y)
    }


# ---------------------------------------------------------------- experiments


def experiments(rows: dict[str, list[dict]], seed: int):
    train = [r for r in rows["train"] if r["label"] != "Empty"]
    test = [r for r in rows["val"] if r["label"] != "Empty"]
    yield (
        "same-data",
        train,
        test,
        "train on train_cropped (all sites), test on val_cropped",
    )
    for site in ("Montenegro", "Desierto_Tatacoa"):
        yield (
            f"new-site-{site}",
            [r for r in train if r["site"] != site],
            [r for r in test if r["site"] == site],
            f"train on train_cropped without {site}; test on {site} val_cropped crops",
        )
    rng = np.random.default_rng(seed)
    by_label = defaultdict(list)
    for r in train:
        by_label[r["label"]].append(r)
    for k in (50, 200):
        sub = []
        for label in sorted(by_label):
            pool = by_label[label]
            if len(pool) >= k:
                sub += [pool[i] for i in rng.choice(len(pool), k, replace=False)]
        yield (
            f"few-shot-{k}",
            sub,
            test,
            f"{k} train_cropped crops per class (classes with >= {k}); test on val_cropped",
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument(
        "--ai4g",
        type=Path,
        required=True,
        help="Engine model dir with labels.txt and 1/model.onnx",
    )
    ap.add_argument(
        "--dinov3", type=Path, required=True, help="DINOv3 ONNX image encoder"
    )
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*", help="experiment names to run")
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.data)
    print("embedding all crops with DINOv3 (cached)", flush=True)
    inner = OnnxEmbedder(str(args.dinov3), batch_size=64)
    cache, feats_by_path = embed_all(
        rows["train"] + rows["val"], inner, args.work / "dinov3_features.npz"
    )
    embedder = CachedEmbedder(inner, cache)

    result_file = args.out / "result.json"
    results = json.loads(result_file.read_text()) if result_file.exists() else {}
    for name, train, test, desc in experiments(rows, args.seed):
        if args.only and name not in args.only:
            continue
        counts = Counter(r["label"] for r in train)
        labels = sorted(lbl for lbl, n in counts.items() if n >= MIN_TRAIN_PER_CLASS)
        index = {lbl: i for i, lbl in enumerate(labels)}
        train = [r for r in train if r["label"] in index]
        dropped = sum(1 for r in test if r["label"] not in index)
        test = [r for r in test if r["label"] in index]
        y = np.array([index[r["label"]] for r in test])
        print(
            f"== {name}: {len(train)} train / {len(test)} test crops, {len(labels)} classes",
            flush=True,
        )

        probs, ai4g_labels = ai4g_scores(args.ai4g, test)
        # Closed set over this experiment's labels, the same label set the fine-tuned models get.
        cols = [ai4g_labels.index(lbl) for lbl in labels]
        preds = {"ai4g": probs[:, cols].argmax(axis=1)}
        runs = {}

        dino_bundle, dino_val, dino_s = run_trainer(
            DinoV3LinearTrainer(lambda: embedder),
            {"seed": args.seed},
            train,
            labels,
            args.work / name / "dinov3_linear",
        )
        preds["dinov3_linear"] = dinov3_scores(dino_bundle, test, feats_by_path).argmax(
            axis=1
        )
        runs["dinov3_linear"] = {"inner_val": dino_val, "seconds": dino_s}

        conv_bundle, conv_val, conv_s = run_trainer(
            ConventionalClassifierTrainer(),
            {
                "seed": args.seed,
                "epochs": args.epochs,
                "batch": 64,
                "workers": 16,
                "device": args.device,
            },
            train,
            labels,
            args.work / name / "conventional_classifier",
        )
        preds["conventional_classifier"] = conventional_scores(
            conv_bundle, test
        ).argmax(axis=1)
        runs["conventional_classifier"] = {"inner_val": conv_val, "seconds": conv_s}

        summary = summarize(y, preds, [r["seq"] for r in test], args.seed)
        results[name] = {
            "description": desc,
            "train_crops": len(train),
            "train_cameras": len({r["camera"] for r in train}),
            "test_dropped_unseen_label": dropped,
            "classes": labels,
            "summary": summary,
            "runs": runs,
            "per_class_recall": {
                k: per_class_recall(y, p, labels) for k, p in preds.items()
            },
        }
        result_file.write_text(json.dumps(results, indent=2, default=float))
        print(
            json.dumps(
                {k: v for k, v in summary.items() if isinstance(v, dict)},
                indent=1,
                default=float,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
