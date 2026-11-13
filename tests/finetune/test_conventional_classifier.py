"""Phase 3: the ``conventional_classifier`` pipeline (timm classifier fine-tuned on crops, ONNX export).

Criteria C1-C9 of ``docs/review/implement-conventional-classifier/spec.md``. The models are tiny and
random-init (``resnet18``, ``pretrained=False``, 32 px) so the suite runs on CPU without downloads.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile

import numpy as np
import pytest
from PIL import Image

from .dinov3_helpers import (
    Recorder,
    SynItem,
    balanced_items,
    bundle_json,
    make_context,
    read_bundle,
    snapshot_items,
    synthetic_items,
    synthetic_png,
)

SMOKE = {
    "model": "resnet18",
    "pretrained": False,
    "imgsz": 32,
    "epochs": 12,
    "warmup_epochs": 1,
    "patience": 0,
    "batch": 16,
    "device": "cpu",
}


def trainer():
    from sparrow_finetune.pipelines.conventional_classifier import (
        ConventionalClassifierTrainer,
    )

    return ConventionalClassifierTrainer()


def run_stages(t, ctx, until="export"):
    t.prepare(ctx)
    if until == "prepare":
        return None
    t.train(ctx)
    if until == "train":
        return None
    metrics = t.evaluate(ctx)
    if until == "evaluate":
        return metrics
    return metrics, t.export(ctx)


def val_side(syn: list[SynItem], seed=0, val_fraction=0.2) -> list[SynItem]:
    from sparrow_finetune.pipelines.dinov3_linear import assign_split

    return [
        it
        for it in syn
        if assign_split(it.group or it.name, seed=seed, val_fraction=val_fraction)
        == "val"
    ]


def onnx_inputs(syn: list[SynItem], imgsz: int) -> np.ndarray:
    """The documented bundle input: RGB crop stretched to imgsz x imgsz, NCHW float32 in [0, 1]."""
    out = []
    for it in syn:
        im = Image.open(io.BytesIO(it.png)).convert("RGB")
        if it.bbox is not None:
            x, y, w, h = it.bbox
            im = im.crop((x, y, x + w, y + h))
        im = im.resize((imgsz, imgsz), Image.BILINEAR)
        out.append(np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0)
    return np.stack(out).astype(np.float32)


def onnx_probabilities(model_bytes: bytes, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
    return np.asarray(sess.run(["probabilities"], {"images": x})[0])


@pytest.fixture(scope="module")
def syn():
    return balanced_items(
        train_groups=4, val_groups=2, images_per_group=2, ungrouped_per_label=1
    )


@pytest.fixture(scope="module")
def trained(syn, tmp_path_factory):
    """One full smoke run shared by the read-only assertions."""
    root = tmp_path_factory.mktemp("conv")
    items = snapshot_items(syn, root / "images")
    t = trainer()
    ctx, rec = make_context(t, items, root / "work", dict(SMOKE))
    metrics, result = run_stages(t, ctx)
    return {
        "trainer": t,
        "ctx": ctx,
        "rec": rec,
        "metrics": metrics,
        "result": result,
        "items": items,
    }


@pytest.fixture
def items(syn, tmp_path):
    return snapshot_items(syn, tmp_path / "images")


# --------------------------------------------------------------------------- C1


def test_c1_identity_attributes():
    t = trainer()
    assert t.id == "conventional_classifier"
    assert t.task == "classification"
    assert set(t.requires) == set()


def test_c1_register_builtin_pipelines_registers_it_idempotently(tmp_path):
    from sparrow_finetune.config import Settings
    from sparrow_finetune.pipelines import Registry, register_builtin_pipelines

    reg = Registry()
    s = Settings(database_url="postgresql://u@h/db", data_dir=tmp_path, embedder="")
    register_builtin_pipelines(reg, s)
    first = reg.get("conventional_classifier")
    register_builtin_pipelines(reg, s)
    assert first is not None and reg.get("conventional_classifier") is first
    listed = {p["id"]: p for p in reg.catalogue()}
    assert listed["conventional_classifier"]["status"] == "available"
    assert listed["conventional_classifier"]["task"] == "classification"


# --------------------------------------------------------------------------- C2


def test_c2_validate_params_defaults():
    assert trainer().validate_params({}) == {
        "model": "resnet50",
        "pretrained": True,
        "epochs": 30,
        "warmup_epochs": 2,
        "batch": 32,
        "lr": 1e-3,
        "backbone_lr_factor": 0.1,
        "weight_decay": 0.05,
        "label_smoothing": 0.1,
        "imgsz": 224,
        "seed": 0,
        "val_fraction": 0.2,
        "device": "auto",
        "amp": True,
        "patience": 5,
        "workers": 0,
        "class_floor": 2,
        "augment": True,
        "near_dup_max_distance": 4,
        "stock_model": None,
        "stock_min_val_crops": 100,
        "base_model_id": None,
        "threshold_target_accuracy": 0.95,
        "threshold_min_val_crops": 30,
        "confidence_threshold": None,
    }


def test_c2_validate_params_keeps_valid_overrides():
    p = trainer().validate_params(dict(SMOKE, lr=3, amp=False, device="cuda:1"))
    assert p["model"] == "resnet18" and p["pretrained"] is False and p["imgsz"] == 32
    assert p["lr"] == 3.0 and isinstance(p["lr"], float)
    assert p["amp"] is False and p["device"] == "cuda:1"


def test_c2_unknown_parameter_named():
    with pytest.raises(ValueError, match="max_epochs"):
        trainer().validate_params({"max_epochs": 3})


@pytest.mark.parametrize(
    "param,value",
    [
        ("model", "not_a_timm_model_xyz"),
        ("model", "/tmp/evil.pth"),
        ("model", "hf-hub:someone/resnet"),
        ("model", "https://example.com/m.pth"),
        ("model", 50),
        ("pretrained", 1),
        ("epochs", 0),
        ("epochs", True),
        ("epochs", 2.0),
        ("warmup_epochs", -1),
        ("batch", 0),
        ("lr", 0),
        ("lr", float("nan")),
        ("lr", float("inf")),
        ("backbone_lr_factor", -0.1),
        ("backbone_lr_factor", 1.5),
        ("weight_decay", -1),
        ("label_smoothing", 1.0),
        ("imgsz", 16),
        ("imgsz", 4096),
        ("seed", -1),
        ("val_fraction", 0),
        ("val_fraction", 0.9),
        ("device", "gpu"),
        ("amp", "yes"),
        ("patience", -1),
        ("workers", -1),
        ("class_floor", 0),
        ("augment", None),
    ],
)
def test_c2_bad_value_rejected_naming_param(param, value):
    with pytest.raises(ValueError, match=param):
        trainer().validate_params({param: value})


def test_c2_warmup_cannot_exceed_epochs():
    t = trainer()
    assert t.validate_params({"epochs": 3, "warmup_epochs": 3})["warmup_epochs"] == 3
    with pytest.raises(ValueError, match="warmup_epochs"):
        t.validate_params({"epochs": 3, "warmup_epochs": 4})
    # FT-44: the default warm-up shrinks to a one-epoch run instead of rejecting it.
    assert t.validate_params({"epochs": 1})["warmup_epochs"] == 1


# --------------------------------------------------------------------------- C3


def test_c3_prepare_keeps_groups_whole_and_counts_crops(trained, syn):
    prepared = json.loads((trained["ctx"].workdir / "prepare.json").read_text())
    split = prepared["split"]
    assert not set(split["train"]["groups"]) & set(split["val"]["groups"])
    assert split["train"]["crops"] + split["val"]["crops"] == len(syn)
    assert split["train"]["items"] + split["val"]["items"] == len(syn)
    assert {it.group for it in val_side(syn) if it.group} == set(split["val"]["groups"])


def test_c3_prepare_fails_when_label_below_class_floor(tmp_path):
    syn = synthetic_items(groups_per_label=4)
    syn.append(
        SynItem(
            "rare-0.png", "rare", "rare-group", None, synthetic_png("cat", 999, None)
        )
    )
    items = snapshot_items(syn, tmp_path / "images")
    t = trainer()
    ctx, _ = make_context(t, items, tmp_path / "work", dict(SMOKE))
    with pytest.raises(ValueError, match="rare"):
        t.prepare(ctx)


def test_c3_prepare_fails_when_validation_split_is_empty(tmp_path):
    from sparrow_finetune.pipelines.dinov3_linear import assign_split

    train_only = [
        it
        for it in synthetic_items(groups_per_label=6)
        if assign_split(it.group or it.name, seed=0, val_fraction=0.2) == "train"
    ]
    items = snapshot_items(train_only, tmp_path / "images")
    t = trainer()
    ctx, _ = make_context(t, items, tmp_path / "work", dict(SMOKE))
    with pytest.raises(ValueError, match="validation"):
        t.prepare(ctx)


# --------------------------------------------------------------------------- C4


def test_c4_one_event_per_epoch_with_phase_losses_f1_and_lrs(trained):
    events = trained["rec"].stage_events("train")
    epoch_events = [e for e in events if "epoch" in e]
    assert [e["epoch"] for e in epoch_events] == list(range(1, SMOKE["epochs"] + 1))
    assert [e["phase"] for e in epoch_events] == ["warmup"] + ["finetune"] * (
        SMOKE["epochs"] - 1
    )
    for e in epoch_events:
        for key in ("train_loss", "val_loss", "val_macro_f1", "lr", "backbone_lr"):
            assert isinstance(e[key], float) and math.isfinite(e[key]), (key, e)
    assert epoch_events[0]["backbone_lr"] == 0.0
    ft = epoch_events[1:]
    assert ft[0]["backbone_lr"] == pytest.approx(ft[0]["lr"] * 0.1)
    assert ft[-1]["lr"] < ft[0]["lr"], "cosine decay over the fine-tune epochs"


def test_c4_keeps_best_epoch_by_val_macro_f1(trained):
    workdir = trained["ctx"].workdir
    info = json.loads((workdir / "train.json").read_text())
    curve = info["epochs"]
    best = max(curve, key=lambda p: p["val_macro_f1"])
    assert info["best_epoch"] == min(
        p["epoch"] for p in curve if p["val_macro_f1"] == best["val_macro_f1"]
    )
    assert (workdir / "best.pt").is_file() and (workdir / "last.pt").is_file()


def test_c4_warmup_trains_only_the_classifier(items, tmp_path):
    import timm
    import torch

    params = dict(SMOKE, epochs=1, warmup_epochs=1, seed=7)
    t = trainer()
    ctx, _ = make_context(t, items, tmp_path / "work", params)
    run_stages(t, ctx, until="train")

    torch.manual_seed(7)
    initial = timm.create_model(
        "resnet18", pretrained=False, num_classes=3
    ).state_dict()
    best = torch.load(
        tmp_path / "work" / "best.pt", map_location="cpu", weights_only=True
    )
    assert set(best) == set(initial)
    head = {k for k in initial if k.startswith("fc.")}
    assert head == {"fc.weight", "fc.bias"}
    for k in initial:
        if k in head:
            assert not torch.equal(best[k], initial[k]), f"classifier {k} did not train"
        else:
            assert torch.equal(best[k], initial[k]), (
                f"backbone {k} changed during warm-up"
            )


def test_c4_early_stopping_after_warmup(items, tmp_path):
    t = trainer()
    ctx, rec = make_context(
        t, items, tmp_path / "work", dict(SMOKE, epochs=40, patience=2)
    )
    run_stages(t, ctx, until="train")
    epochs = [e["epoch"] for e in rec.stage_events("train") if "epoch" in e]
    assert len(epochs) < 40
    curve = json.loads((tmp_path / "work" / "train.json").read_text())["epochs"]
    f1 = [p["val_macro_f1"] for p in curve]
    best_at = f1.index(max(f1))
    assert len(f1) - 1 - best_at == 2, (
        "stops exactly patience epochs after the last improvement"
    )


def test_c4_checks_cancellation_every_epoch(items, tmp_path):
    t = trainer()
    ctx, rec = make_context(t, items, tmp_path / "work", dict(SMOKE, epochs=4))
    t.prepare(ctx)
    before = rec.checks
    t.train(ctx)
    assert rec.checks - before >= 4


def test_c4_cancel_during_train_stops_at_next_epoch(items, tmp_path):
    from sparrow_finetune.pipelines import JobCancelled

    rec = Recorder()
    rec.cancel_on_first_train_event = True
    t = trainer()
    ctx, _ = make_context(
        t, items, tmp_path / "work", dict(SMOKE, epochs=20), recorder=rec
    )
    t.prepare(ctx)
    with pytest.raises(JobCancelled):
        t.train(ctx)
    assert len([e for e in rec.stage_events("train") if "epoch" in e]) <= 1


# --------------------------------------------------------------------------- C5


def test_c5_amp_is_off_on_cpu(trained):
    info = json.loads((trained["ctx"].workdir / "train.json").read_text())
    assert trained["ctx"].params["amp"] is True
    assert info["amp"] is False
    assert info["device"] == "cpu"


# --------------------------------------------------------------------------- C6


def test_c6_same_inputs_give_identical_curves(syn, tmp_path):
    curves = []
    for run in ("a", "b"):
        items = snapshot_items(syn, tmp_path / run / "images")
        t = trainer()
        ctx, _ = make_context(t, items, tmp_path / run / "work", dict(SMOKE, epochs=3))
        run_stages(t, ctx, until="train")
        curves.append(
            json.loads((tmp_path / run / "work" / "train.json").read_text())["epochs"]
        )
    assert curves[0] == curves[1]


def test_c6_train_leaves_global_rng_untouched(items, tmp_path):
    import torch

    t = trainer()
    ctx, _ = make_context(t, items, tmp_path / "work", dict(SMOKE, epochs=2))
    t.prepare(ctx)
    torch.manual_seed(12345)
    state = torch.get_rng_state()
    t.train(ctx)
    assert torch.equal(torch.get_rng_state(), state)


def test_c6_model_built_from_timm_with_pretrained_flag_and_label_count(
    items, tmp_path, monkeypatch
):
    import timm
    import torch

    real = timm.create_model
    calls = []

    def fake(name, pretrained=False, **kw):
        calls.append(
            {
                "name": name,
                "pretrained": pretrained,
                "num_classes": kw.get("num_classes"),
                "first_draw": torch.rand(1).item(),
            }
        )
        torch.manual_seed(
            5
        )  # re-seed so the fake does not consume the trainer's stream
        return real(name, pretrained=False, **kw)

    monkeypatch.setattr(timm, "create_model", fake)
    t = trainer()
    ctx, _ = make_context(
        t, items, tmp_path / "work", dict(SMOKE, pretrained=True, epochs=1, seed=5)
    )
    run_stages(t, ctx, until="train")
    assert calls and calls[0]["name"] == "resnet18"
    assert calls[0]["pretrained"] is True and calls[0]["num_classes"] == 3
    torch.manual_seed(5)
    assert calls[0]["first_draw"] == torch.rand(1).item(), (
        "create_model must be the first RNG use after manual_seed(seed)"
    )


# --------------------------------------------------------------------------- C7


def test_c7_evaluate_metrics_and_report(trained):
    metrics = trained["metrics"]
    for key in (
        "val_macro_f1",
        "val_accuracy",
        "val_log_loss",
        "val_ece",
        "best_epoch",
        "train_crops",
        "val_crops",
    ):
        assert isinstance(metrics[key], float) and math.isfinite(metrics[key]), key
    report = json.loads((trained["ctx"].workdir / "evaluate.json").read_text())
    labels = trained["ctx"].labels
    assert set(report["per_class"]) == set(labels)
    cm = np.asarray(report["confusion_matrix"])
    assert cm.shape == (len(labels), len(labels)) and cm.sum() == metrics["val_crops"]


# --------------------------------------------------------------------------- C8


def test_c8_export_writes_onnx_bundle(trained):
    result = trained["result"]
    assert result.format == "onnx-bundle"
    assert result.path.name == "conventional_classifier_bundle.zip"
    assert result.metadata["labels"] == list(trained["ctx"].labels)
    assert set(read_bundle(result.path)) == {
        "model.onnx",
        "model_card.json",
        "manifest.json",
        "weights.pt",
    }


def test_c8_bundle_carries_torch_weights_for_warm_start(trained):
    """``weights.pt`` is the bare timm state dict, listed with its sha256 in the
    manifest next to the timm architecture name, and loadable with ``weights_only``."""
    import timm
    import torch

    bundle = read_bundle(trained["result"].path)
    manifest = bundle_json(bundle, "manifest.json")
    assert manifest["model_arch"] == SMOKE["model"]
    assert (
        manifest["files"]["weights.pt"]
        == hashlib.sha256(bundle["weights.pt"]).hexdigest()
    )
    state = torch.load(io.BytesIO(bundle["weights.pt"]), weights_only=True)
    net = timm.create_model(SMOKE["model"], pretrained=False, num_classes=3)
    net.load_state_dict(state)  # strict: every key and shape matches


def test_c8_onnx_signature(trained):
    import onnxruntime as ort

    bundle = read_bundle(trained["result"].path)
    sess = ort.InferenceSession(
        bundle["model.onnx"], providers=["CPUExecutionProvider"]
    )
    (inp,) = sess.get_inputs()
    assert inp.name == "images" and inp.type == "tensor(float)"
    assert list(inp.shape[1:]) == [3, 32, 32]
    assert not isinstance(inp.shape[0], int), "batch axis must be dynamic"
    assert [o.name for o in sess.get_outputs()] == ["probabilities"]
    x = np.random.default_rng(0).random((5, 3, 32, 32), dtype=np.float32)
    probs = onnx_probabilities(bundle["model.onnx"], x)
    assert probs.shape == (5, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_c8_manifest_and_card(trained):
    bundle = read_bundle(trained["result"].path)
    manifest = bundle_json(bundle, "manifest.json")
    labels = list(trained["ctx"].labels)
    assert manifest["pipeline"] == "conventional_classifier"
    assert manifest["labels"] == labels
    assert manifest["model"] == "resnet18"
    inp = manifest["input"]
    assert inp["name"] == "images" and inp["size"] == 32 and inp["layout"] == "NCHW"
    assert (
        inp["color"] == "RGB" and inp["scale"] == [0, 1] and inp["resize"] == "stretch"
    )
    assert len(inp["mean"]) == 3 and len(inp["std"]) == 3
    assert manifest["parity"]["max_abs_delta"] <= 1e-4
    assert manifest["parity"]["top1_agreement"] == 1.0
    assert manifest["metrics"]["val_macro_f1"] == trained["metrics"]["val_macro_f1"]
    for name in ("model.onnx", "model_card.json"):
        assert manifest["files"][name] == hashlib.sha256(bundle[name]).hexdigest()
    card = bundle_json(bundle, "model_card.json")
    assert card["pipeline"] == "conventional_classifier" and card["labels"] == labels
    assert (
        card["params"]["model"] == "resnet18" and "split" in card and "epochs" in card
    )


def test_c8_parity_failure_raises_and_writes_no_bundle(items, tmp_path, monkeypatch):
    import sparrow_finetune.pipelines.conventional_classifier as mod

    t = trainer()
    ctx, _ = make_context(
        t, items, tmp_path / "work", dict(SMOKE, epochs=1, warmup_epochs=1)
    )
    run_stages(t, ctx, until="evaluate")
    monkeypatch.setattr(mod, "PARITY_MAX_ABS_DELTA", -1.0)
    with pytest.raises(RuntimeError, match="parity"):
        t.export(ctx)
    assert not (tmp_path / "work" / "conventional_classifier_bundle.zip").exists()


# --------------------------------------------------------------------------- C9


def test_c9_cpu_smoke_learns_and_onnx_reproduces_val_labels(trained, syn):
    assert trained["metrics"]["val_macro_f1"] >= 0.9
    bundle = read_bundle(trained["result"].path)
    labels = bundle_json(bundle, "manifest.json")["labels"]
    val = val_side(syn)
    probs = onnx_probabilities(bundle["model.onnx"], onnx_inputs(val, 32))
    assert [labels[i] for i in probs.argmax(axis=1)] == [it.label for it in val]


def test_c9_bundle_zip_is_valid(trained):
    assert zipfile.is_zipfile(trained["result"].path)
