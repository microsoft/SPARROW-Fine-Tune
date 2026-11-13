"""Phase 1B unit tests for the ``dinov3_linear`` trainer (spec B4-B15).

Strategy: drive ``DinoV3LinearTrainer`` stage by stage with a directly constructed ``JobContext`` and a
deterministic fake ``Embedder`` (mean RGB of each crop, dim 3). The synthetic classes have distinct
colours, so the features are linearly separable: a correct trainer must reach val macro-F1 >= 0.9, and
its exported ``head.onnx`` must reproduce the labels from raw features computed here independently.
Group integrity (B6) is checked against ``assign_split`` itself, so the test pins the contract "split is
decided by group (else image name)" without depending on the hash-to-fraction mapping.
Everything from ``sparrow_finetune`` is imported inside tests/fixtures so collection always succeeds.
"""

from __future__ import annotations

import hashlib
import io
import math
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from .dinov3_helpers import (
    SynItem,
    balanced_items,
    bundle_json,
    head_probabilities,
    make_context,
    make_mean_rgb_embedder,
    mean_rgb,
    read_bundle,
    snapshot_items,
    synthetic_items,
    synthetic_png,
)

MAX_EPOCHS = 20
DEFAULTS = {
    "max_epochs": 50,
    "alpha": 1e-4,
    "seed": 0,
    "val_fraction": 0.2,
    "test_fraction": 0.0,
    "standardize": True,
    "class_floor": 2,
    "feature_noise_std": 0.0,
    "augment_replicas": 0,
    "reference_gate": True,
    "reference_margin": 0.02,
    "reference_min_val_crops": 100,
    "near_dup_max_distance": 4,
    "stock_model": None,
    "stock_min_val_crops": 100,
    "threshold_target_accuracy": 0.95,
    "threshold_min_val_crops": 30,
    "confidence_threshold": None,
    "embedding_cache": True,
}
METRIC_KEYS = {
    "val_macro_f1",
    "val_accuracy",
    "val_log_loss",
    "val_ece",
    "best_epoch",
    "train_crops",
    "val_crops",
}


def new_trainer(embedder=None):
    from sparrow_finetune.pipelines.dinov3_linear import DinoV3LinearTrainer

    embedder = embedder or make_mean_rgb_embedder()
    return DinoV3LinearTrainer(lambda: embedder), embedder


@pytest.fixture
def syn():
    return balanced_items()


@pytest.fixture
def items(syn, tmp_path):
    return snapshot_items(syn, tmp_path / "images")


def run_all(items, workdir: Path, params=None, embedder=None):
    trainer, emb = new_trainer(embedder)
    ctx, rec = make_context(trainer, items, workdir, {"max_epochs": MAX_EPOCHS, **(params or {})})
    trainer.prepare(ctx)
    trainer.train(ctx)
    metrics = trainer.evaluate(ctx)
    result = trainer.export(ctx)
    return {"trainer": trainer, "emb": emb, "ctx": ctx, "rec": rec, "metrics": metrics, "result": result}


@pytest.fixture
def trained(items, tmp_path):
    run = run_all(items, tmp_path / "work")
    run["bundle"] = read_bundle(run["result"].path)
    return run


def split_key(item) -> str:
    return item.group if item.group else item.name


def raw_features(syn_items: list[SynItem]) -> np.ndarray:
    """Mean RGB of each annotation's crop, computed without the code under test."""
    feats = []
    for it in syn_items:
        im = Image.open(io.BytesIO(it.png)).convert("RGB")
        if it.bbox is not None:
            x, y, w, h = it.bbox
            im = im.crop((x, y, x + w, y + h))
        feats.append(mean_rgb(im))
    return np.stack(feats).astype(np.float32)


def train_epoch_events(rec) -> list[dict]:
    return [m for m in rec.stage_events("train") if "epoch" in m]


def find_per_class(obj, labels):
    """Locate per-class metrics: ``{label: {precision, recall, f1, support}}`` or a list with ``label`` keys."""
    want = {"precision", "recall", "f1", "support"}
    if isinstance(obj, dict):
        if set(labels) <= set(obj) and all(isinstance(obj[l], dict) and want <= set(obj[l]) for l in labels):
            return {l: obj[l] for l in labels}
        for v in obj.values():
            found = find_per_class(v, labels)
            if found:
                return found
    if isinstance(obj, list) and obj and all(isinstance(e, dict) and "label" in e for e in obj):
        by = {e["label"]: e for e in obj}
        if set(labels) <= set(by) and all(want <= set(by[l]) for l in labels):
            return {l: by[l] for l in labels}
    return None


# --------------------------------------------------------------------------- entry points


def test_trainer_identity_attributes():
    trainer, _ = new_trainer()
    assert trainer.id == "dinov3_linear"
    assert trainer.task == "classification"
    assert trainer.requires == frozenset()


def test_register_builtin_pipelines_is_idempotent_and_lazy(tmp_path):
    from sparrow_finetune.config import Settings
    from sparrow_finetune.pipelines import Registry, register_builtin_pipelines

    reg = Registry()
    # embedder="" must not fail at registration: the factory is only called when a job runs (B17).
    settings = Settings(database_url="postgresql://unused/db", data_dir=tmp_path, embedder="")
    register_builtin_pipelines(reg, settings)
    register_builtin_pipelines(reg, settings)
    assert reg.ids().count("dinov3_linear") == 1
    entry = {p["id"]: p for p in reg.catalogue()}["dinov3_linear"]
    assert entry["status"] == "available"
    assert entry["task"] == "classification"


# --------------------------------------------------------------------------- B4 crop_box


@pytest.mark.parametrize(
    "bbox, size, expected",
    [
        ([1.2, 2.7, 10.1, 5.0], (20, 10), (1, 2, 12, 8)),  # floor left/top, ceil right/bottom
        ([4, 3, 6, 2], (20, 10), (4, 3, 10, 5)),  # integers unchanged
        ([-5, -3, 10, 10], (8, 8), (0, 0, 5, 7)),  # clamp at the origin
        ([5, 5, 100, 100], (8, 8), (5, 5, 8, 8)),  # clamp at the far edge
        (None, (31, 17), (0, 0, 31, 17)),  # whole image
    ],
)
def test_b4_crop_box_floors_ceils_and_clamps(bbox, size, expected):
    from sparrow_finetune.pipelines.dinov3_linear import crop_box

    box = crop_box(bbox, *size)
    assert tuple(box) == expected
    assert all(type(v) is int for v in box)


@pytest.mark.parametrize("bbox", [[10, 10, 5, 5], [2, 2, 0, 3], [-20, 1, 5, 3], [1, 1, 3, 0]])
def test_b4_crop_box_zero_area_after_clamping_raises(bbox):
    from sparrow_finetune.pipelines.dinov3_linear import crop_box

    with pytest.raises(ValueError):
        crop_box(bbox, 8, 8)


# --------------------------------------------------------------------------- B5 assign_split


def test_b5_assign_split_is_deterministic_and_order_independent():
    from sparrow_finetune.pipelines.dinov3_linear import assign_split

    keys = [f"key-{i}" for i in range(300)]
    forward = {k: assign_split(k, seed=7, val_fraction=0.2) for k in keys}
    backward = {k: assign_split(k, seed=7, val_fraction=0.2) for k in reversed(keys)}
    assert forward == backward
    assert set(forward.values()) == {"train", "val"}
    other_seed = {k: assign_split(k, seed=8, val_fraction=0.2) for k in keys}
    assert other_seed != forward


def test_b5_assign_split_val_share_close_to_fraction():
    from sparrow_finetune.pipelines.dinov3_linear import assign_split

    splits = [assign_split(f"img-{i:04d}.jpg", seed=0, val_fraction=0.2) for i in range(1000)]
    share = splits.count("val") / len(splits)
    assert 0.15 <= share <= 0.25, share


# --------------------------------------------------------------------------- B6 group integrity


def test_b6_model_card_split_keeps_groups_whole(trained, items):
    from sparrow_finetune.pipelines.dinov3_linear import assign_split

    card = bundle_json(trained["bundle"], "model_card.json")
    split = card["split"]
    train_groups, val_groups = set(split["train"]["groups"]), set(split["val"]["groups"])
    assert not train_groups & val_groups

    expected = {"train": Counter(), "val": Counter()}
    for it in items:
        side = assign_split(split_key(it), seed=0, val_fraction=0.2)
        expected[side]["items"] += 1
        expected[side]["crops"] += len(it.annotations)
        if it.group:
            assert it.group in (train_groups if side == "train" else val_groups), it.group
    for side in ("train", "val"):
        assert split[side]["items"] == expected[side]["items"], side
        assert split[side]["crops"] == expected[side]["crops"], side


def test_b6_seed_changes_split_but_groups_stay_whole(items, tmp_path):
    run = run_all(items, tmp_path / "work-seed", {"seed": 11})
    split = bundle_json(read_bundle(run["result"].path), "model_card.json")["split"]
    assert not set(split["train"]["groups"]) & set(split["val"]["groups"])
    assert split["train"]["crops"] + split["val"]["crops"] == len(items)


# --------------------------------------------------------------------------- B7 validate_params


def test_b7_validate_params_defaults():
    trainer, _ = new_trainer()
    assert trainer.validate_params({}) == DEFAULTS
    assert trainer.validate_params(None) == DEFAULTS


def test_b7_validate_params_keeps_valid_overrides():
    trainer, _ = new_trainer()
    out = trainer.validate_params({"max_epochs": 20, "alpha": 1e-3, "standardize": False, "val_fraction": 0.5})
    assert out == {**DEFAULTS, "max_epochs": 20, "alpha": 1e-3, "standardize": False, "val_fraction": 0.5}


def test_b7_validate_params_rejects_unknown_key_naming_it():
    trainer, _ = new_trainer()
    with pytest.raises(ValueError, match="learning_rate"):
        trainer.validate_params({"learning_rate": 0.1})


@pytest.mark.parametrize(
    "param, value",
    [
        ("max_epochs", 0),
        ("max_epochs", 501),
        ("max_epochs", 2.5),
        ("max_epochs", "10"),
        ("alpha", 0),
        ("alpha", -1e-3),
        ("alpha", "small"),
        ("seed", -1),
        ("seed", 1.5),
        ("val_fraction", 0),
        ("val_fraction", 0.51),
        ("val_fraction", "0.2"),
        ("standardize", "yes"),
        ("class_floor", 0),
        ("class_floor", "2"),
        ("feature_noise_std", -0.1),
        ("feature_noise_std", "none"),
        ("augment_replicas", -1),
        ("augment_replicas", 11),
        ("augment_replicas", 1.5),
    ],
)
def test_b7_validate_params_rejects_bad_type_or_range_naming_param(param, value):
    trainer, _ = new_trainer()
    with pytest.raises(ValueError, match=param):
        trainer.validate_params({param: value})


# --------------------------------------------------------------------------- B8 prepare


def test_b8_prepare_embeds_every_crop_and_writes_arrays(items, syn, tmp_path):
    trainer, emb = new_trainer()
    ctx, rec = make_context(trainer, items, tmp_path / "work", {"max_epochs": MAX_EPOCHS})
    trainer.prepare(ctx)

    n = len(items)
    # every annotation embedded once, cropped to its bbox (or the whole image)
    expected_sizes = Counter((it.bbox[2], it.bbox[3]) if it.bbox else (32, 24) for it in syn)
    assert Counter(emb.sizes) == expected_sizes

    features = np.load(ctx.workdir / "features.npy")
    labels = np.load(ctx.workdir / "labels.npy", allow_pickle=True)
    splits = np.load(ctx.workdir / "splits.npy", allow_pickle=True)
    assert features.shape == (n, 3)
    assert len(labels) == n and len(splits) == n
    names = [ctx.labels[int(v)] for v in labels] if labels.dtype.kind in "iu" else [str(v) for v in labels]
    assert Counter(names) == Counter(it.label for it in syn)
    assert len(set(splits.tolist())) == 2
    # features are the embedder's output for the crops (order-insensitive check)
    expected = raw_features(syn)
    np.testing.assert_allclose(np.sort(features, axis=0), np.sort(expected, axis=0), atol=1e-5)

    prepare_events = [m for m in rec.stage_events("prepare") if "crops" in m]
    assert prepare_events, rec.events
    assert prepare_events[-1]["crops"] == n
    assert prepare_events[-1]["dim"] == 3


def test_b8_prepare_checks_cancellation_between_embedding_batches(items, tmp_path):
    trainer, emb = new_trainer()
    ctx, rec = make_context(trainer, items, tmp_path / "work")
    trainer.prepare(ctx)
    assert emb.calls, "embedder never called"
    assert rec.checks >= max(1, len(emb.calls) - 1), (rec.checks, emb.calls)


def test_b8_prepare_stops_when_cancelled_during_embedding(items, tmp_path):
    from sparrow_finetune.pipelines import JobCancelled

    trainer, emb = new_trainer()
    ctx, rec = make_context(trainer, items, tmp_path / "work")

    def cancel_after_first_batch(_n):
        rec.cancel = True

    emb.on_call = cancel_after_first_batch
    with pytest.raises(JobCancelled):
        trainer.prepare(ctx)
    assert len(emb.calls) == 1


# --------------------------------------------------------------------------- B9 prepare failures


def test_b9_prepare_fails_when_label_below_class_floor(syn, tmp_path):
    rare = SynItem("rare-0.png", "rare", "rare-group", None, synthetic_png("cat", 999, None))
    items = snapshot_items([*syn, rare], tmp_path / "images")
    trainer, _ = new_trainer()
    ctx, _ = make_context(trainer, items, tmp_path / "work")
    with pytest.raises(ValueError, match="rare"):
        trainer.prepare(ctx)


def test_b9_prepare_fails_when_validation_split_is_empty(tmp_path):
    from sparrow_finetune.pipelines.dinov3_linear import assign_split

    group = next(
        g for g in (f"one-group-{i}" for i in range(200)) if assign_split(g, seed=0, val_fraction=0.2) == "train"
    )
    syn = [SynItem(it.name, it.label, group, it.bbox, it.png) for it in synthetic_items(groups_per_label=2)]
    items = snapshot_items(syn, tmp_path / "images")
    trainer, _ = new_trainer()
    ctx, _ = make_context(trainer, items, tmp_path / "work")
    with pytest.raises(ValueError) as info:
        trainer.prepare(ctx)
    assert re.search(r"validation split.*empty|val\w* split.*empty", str(info.value), re.I), str(info.value)


# --------------------------------------------------------------------------- B10 train


def test_b10_train_emits_one_event_per_epoch_with_losses_and_f1(trained):
    events = train_epoch_events(trained["rec"])
    assert len(events) == MAX_EPOCHS
    assert len({e["epoch"] for e in events}) == MAX_EPOCHS
    for e in events:
        for key in ("train_loss", "val_loss", "val_macro_f1"):
            assert isinstance(e[key], (int, float)) and math.isfinite(e[key]), (key, e)
        assert 0.0 <= e["val_macro_f1"] <= 1.0


def test_b10_train_keeps_best_epoch_by_val_macro_f1(trained):
    events = train_epoch_events(trained["rec"])
    best_f1 = max(e["val_macro_f1"] for e in events)
    metrics = trained["metrics"]
    best = [e for e in events if e["epoch"] == metrics["best_epoch"]]
    assert len(best) == 1, (metrics["best_epoch"], [e["epoch"] for e in events])
    assert best[0]["val_macro_f1"] == pytest.approx(best_f1, abs=1e-9)
    assert metrics["val_macro_f1"] == pytest.approx(best_f1, abs=1e-6)


def test_b10_train_checks_cancellation_every_epoch(items, tmp_path):
    from sparrow_finetune.pipelines import JobCancelled

    trainer, _ = new_trainer()
    ctx, rec = make_context(trainer, items, tmp_path / "work", {"max_epochs": MAX_EPOCHS})
    trainer.prepare(ctx)
    rec.cancel_on_first_train_event = True
    with pytest.raises(JobCancelled):
        trainer.train(ctx)
    assert len(train_epoch_events(rec)) <= 2


# --------------------------------------------------------------------------- B11 reproducibility


def test_b11_same_snapshot_and_params_give_identical_metrics(items, tmp_path):
    params = {"feature_noise_std": 0.01, "augment_replicas": 2, "seed": 3}
    a = run_all(items, tmp_path / "run-a", params)
    b = run_all(items, tmp_path / "run-b", params)
    assert a["metrics"] == b["metrics"]


# --------------------------------------------------------------------------- B12 evaluate


@pytest.mark.parametrize("standardize", [True, False])
def test_b12_evaluate_returns_float_metrics_and_separates_classes(items, tmp_path, standardize):
    run = run_all(items, tmp_path / "work", {"standardize": standardize})
    metrics = run["metrics"]
    assert set(metrics) >= METRIC_KEYS
    for key in METRIC_KEYS:
        assert isinstance(metrics[key], float), (key, type(metrics[key]))
    assert metrics["val_macro_f1"] >= 0.9
    assert metrics["train_crops"] + metrics["val_crops"] == float(len(items))
    assert metrics["val_crops"] > 0 and metrics["train_crops"] > 0


# --------------------------------------------------------------------------- B13 export bundle


def test_b13_export_writes_onnx_bundle_zip(trained):
    result = trained["result"]
    assert result.format == "onnx-bundle"
    assert zipfile.is_zipfile(result.path)
    assert {"head.onnx", "manifest.json", "model_card.json"} <= set(trained["bundle"])


def test_b13_manifest_contents(trained):
    bundle, ctx, emb = trained["bundle"], trained["ctx"], trained["emb"]
    manifest = bundle_json(bundle, "manifest.json")
    assert manifest["pipeline"] == "dinov3_linear"
    assert sorted(manifest["labels"]) == sorted(ctx.labels)
    assert len(manifest["labels"]) == len(ctx.labels)
    assert manifest["embedder"] == {"id": emb.id, "dim": 3, "model_hash": emb.model_hash}
    assert {"size", "mean", "std"} <= set(manifest["input"])
    for key, value in trained["metrics"].items():
        assert manifest["metrics"][key] == pytest.approx(value), key
    assert {"max_abs_delta", "top1_agreement"} <= set(manifest["parity"])
    assert "head.onnx" in manifest["files"]
    for name, digest in manifest["files"].items():
        assert hashlib.sha256(bundle[name]).hexdigest() == digest, name


def test_b13_head_onnx_maps_raw_features_to_probabilities_in_manifest_label_order(trained, syn):
    bundle = trained["bundle"]
    labels = bundle_json(bundle, "manifest.json")["labels"]
    feats = raw_features(syn)
    probs = head_probabilities(bundle["head.onnx"], feats, len(labels))
    assert probs.shape == (len(syn), len(labels))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-4)
    predicted = [labels[i] for i in probs.argmax(axis=1)]
    accuracy = np.mean([p == it.label for p, it in zip(predicted, syn)])
    assert accuracy >= 0.9, accuracy


# --------------------------------------------------------------------------- B14 parity gate


def test_b14_export_records_parity_within_gate(trained):
    parity = bundle_json(trained["bundle"], "manifest.json")["parity"]
    assert 0.0 <= parity["max_abs_delta"] <= 1e-4
    assert parity["top1_agreement"] == 1.0


# --------------------------------------------------------------------------- B15 model card


def test_b15_model_card_contents(trained):
    ctx, emb, metrics = trained["ctx"], trained["emb"], trained["metrics"]
    card = bundle_json(trained["bundle"], "model_card.json")

    for key, value in ctx.params.items():
        if isinstance(value, float):
            assert card["params"][key] == pytest.approx(value), key
        else:
            assert card["params"][key] == value, key
    for key, value in metrics.items():
        assert card["metrics"][key] == pytest.approx(value), key

    per_class = find_per_class(card, ctx.labels)
    assert per_class is not None, f"no per-class precision/recall/f1/support in {sorted(card)}"
    assert sum(per_class[l]["support"] for l in ctx.labels) == metrics["val_crops"]

    cm = np.asarray(card["confusion_matrix"])
    assert cm.shape == (len(ctx.labels), len(ctx.labels))
    assert cm.sum() == metrics["val_crops"]
    assert np.trace(cm) / cm.sum() == pytest.approx(metrics["val_accuracy"])

    assert {"train", "val"} <= set(card["split"])
    assert len(card["epochs"]) == ctx.params["max_epochs"]
    assert card["embedder"]["id"] == emb.id
    assert card["embedder"]["dim"] == 3
    assert card["embedder"]["model_hash"] == emb.model_hash
