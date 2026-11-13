"""FT-24: calibrated confidence threshold with an abstain outcome (dinov3_linear, conventional_classifier)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from .dinov3_helpers import (
    balanced_items,
    bundle_json,
    make_context,
    make_mean_rgb_embedder,
    read_bundle,
    snapshot_items,
)


def onehot_probs(conf, pred, n_classes=3):
    """Rows whose top-1 is ``pred`` with probability ``conf``; the rest split evenly."""
    conf = np.asarray(conf, dtype=float)
    out = np.tile(((1 - conf) / (n_classes - 1))[:, None], (1, n_classes))
    out[np.arange(len(conf)), pred] = conf
    return out


# --------------------------------------------------------------------------- selection rule


def test_lowest_threshold_meeting_target_accuracy():
    from sparrow_finetune.calibration import select_threshold

    conf = [0.99, 0.95, 0.9, 0.8, 0.7, 0.6]
    pred = np.array([0, 1, 2, 0, 1, 2])
    y = np.array([0, 1, 2, 1, 1, 0])  # 4th and 6th wrong
    rec = select_threshold(
        onehot_probs(conf, pred), y, target_accuracy=0.8, min_val_crops=1
    )
    # Prefix accuracies: 1, 1, 1, .75, .8, .667 -> largest prefix >= .8 ends at conf .7.
    assert rec["calibrated_threshold"] == pytest.approx(0.7)
    assert rec["threshold_coverage"] == pytest.approx(5 / 6)
    assert rec["threshold_accuracy"] == pytest.approx(0.8)
    assert rec["abstain_label"] == "unknown"
    assert rec["reason"] is None


def test_coverage_is_reported_for_lower_targets():
    # FT-30: the record says what a lower target would have given, so the UI can quote it.
    from sparrow_finetune.calibration import COVERAGE_TARGETS, select_threshold

    conf = [0.99, 0.95, 0.9, 0.8, 0.7, 0.6]
    pred = np.array([0, 1, 2, 0, 1, 2])
    y = np.array([0, 1, 2, 1, 1, 0])
    rec = select_threshold(onehot_probs(conf, pred), y, target_accuracy=0.95, min_val_crops=1)
    rows = {r["target"]: r for r in rec["coverage_by_target"]}
    assert set(rows) == set(COVERAGE_TARGETS)
    assert rows[0.95]["threshold"] == pytest.approx(0.9)
    assert rows[0.95]["coverage"] == pytest.approx(0.5)
    assert rows[0.95]["coverage"] == pytest.approx(rec["threshold_coverage"])
    assert rows[0.8]["threshold"] == pytest.approx(0.7)
    assert rows[0.8]["coverage"] == pytest.approx(5 / 6)


def test_coverage_table_is_empty_when_the_split_is_too_small():
    from sparrow_finetune.calibration import select_threshold

    rec = select_threshold(onehot_probs([0.9], np.array([0])), np.array([0]), min_val_crops=5)
    assert rec["coverage_by_target"] == []


def test_all_correct_accepts_everything():
    from sparrow_finetune.calibration import select_threshold

    conf = [0.9, 0.8, 0.5]
    rec = select_threshold(
        onehot_probs(conf, [0, 1, 2]), np.array([0, 1, 2]), min_val_crops=1
    )
    assert rec["calibrated_threshold"] == pytest.approx(0.5)
    assert rec["threshold_coverage"] == 1.0 and rec["threshold_accuracy"] == 1.0


def test_ties_are_never_split():
    from sparrow_finetune.calibration import select_threshold

    # The two 0.8 rows: one right, one wrong. Accepting only one of them is impossible.
    conf = [0.9, 0.8, 0.8]
    y = np.array([0, 1, 0])
    rec = select_threshold(
        onehot_probs(conf, [0, 1, 2]), y, target_accuracy=0.9, min_val_crops=1
    )
    assert rec["calibrated_threshold"] == pytest.approx(0.9)
    assert rec["threshold_coverage"] == pytest.approx(1 / 3)


def test_unreachable_target_gives_null_with_reason():
    from sparrow_finetune.calibration import select_threshold

    rec = select_threshold(
        onehot_probs([0.9, 0.8], [0, 1]),
        np.array([1, 0]),
        target_accuracy=0.5,
        min_val_crops=1,
    )
    assert (
        rec["calibrated_threshold"] is None and "no threshold reaches" in rec["reason"]
    )


@pytest.mark.parametrize("n", [0, 3])
def test_tiny_or_empty_split_gives_null_with_reason(n):
    from sparrow_finetune.calibration import select_threshold, threshold_metrics

    probs = onehot_probs([0.9] * n, [0] * n) if n else np.zeros((0, 3))
    rec = select_threshold(probs, np.zeros(n, dtype=int), min_val_crops=30)
    assert rec["calibrated_threshold"] is None
    assert "threshold_min_val_crops" in rec["reason"]
    assert threshold_metrics(rec) == {}


def test_fixed_threshold_is_measured():
    from sparrow_finetune.calibration import select_threshold

    rec = select_threshold(
        onehot_probs([0.9, 0.6, 0.4], [0, 1, 2]), np.array([0, 2, 2]), fixed=0.5
    )
    assert rec["rule"] == "fixed" and rec["calibrated_threshold"] == 0.5
    assert rec["threshold_coverage"] == pytest.approx(2 / 3)
    assert rec["threshold_accuracy"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- params


@pytest.mark.parametrize("pipeline", ["dinov3", "conventional"])
def test_params_defaults_and_validation(pipeline):
    if pipeline == "dinov3":
        from sparrow_finetune.pipelines.dinov3_linear import DinoV3LinearTrainer

        t = DinoV3LinearTrainer(make_mean_rgb_embedder)
    else:
        from sparrow_finetune.pipelines.conventional_classifier import (
            ConventionalClassifierTrainer,
        )

        t = ConventionalClassifierTrainer()
    p = t.validate_params({})
    assert p["threshold_target_accuracy"] == 0.95
    assert p["confidence_threshold"] is None
    assert p["threshold_min_val_crops"] == 30
    assert t.validate_params({"confidence_threshold": 1})["confidence_threshold"] == 1.0
    for bad in (
        {"confidence_threshold": 1.5},
        {"confidence_threshold": "high"},
        {"threshold_target_accuracy": 0},
        {"threshold_target_accuracy": None},
        {"threshold_min_val_crops": 0},
    ):
        with pytest.raises(ValueError, match=next(iter(bad))):
            t.validate_params(bad)


# --------------------------------------------------------------------------- dinov3_linear wiring


def run_dinov3(tmp_path, params, **item_kw):
    from sparrow_finetune.pipelines.dinov3_linear import DinoV3LinearTrainer

    emb = make_mean_rgb_embedder()
    t = DinoV3LinearTrainer(lambda: emb, cache_dir=tmp_path / "cache")
    items = snapshot_items(balanced_items(**item_kw), tmp_path / "img")
    ctx, _ = make_context(
        t,
        items,
        tmp_path / "work",
        dict({"max_epochs": 10, "reference_gate": False}, **params),
    )
    t.prepare(ctx)
    t.train(ctx)
    metrics = t.evaluate(ctx)
    return metrics, t.export(ctx)


def test_dinov3_records_threshold_in_metrics_manifest_and_metadata(tmp_path):
    metrics, result = run_dinov3(
        tmp_path, {"threshold_min_val_crops": 1, "threshold_target_accuracy": 0.9}
    )
    t = metrics["calibrated_threshold"]
    assert 0.0 <= t <= 1.0
    assert 0.0 < metrics["threshold_coverage"] <= 1.0
    assert metrics["threshold_accuracy"] >= 0.9
    assert all(isinstance(v, float) for v in metrics.values())
    manifest = bundle_json(read_bundle(result.path), "manifest.json")
    assert manifest["calibrated_threshold"] == pytest.approx(t)
    assert manifest["abstain_label"] == "unknown"
    assert manifest["calibration"]["rule"].startswith("min_threshold")
    assert result.metadata["calibrated_threshold"] == pytest.approx(t)
    assert result.metadata["abstain_label"] == "unknown"
    card = bundle_json(read_bundle(result.path), "model_card.json")
    assert card["threshold"]["calibrated_threshold"] == pytest.approx(t)


def test_dinov3_small_validation_split_gives_null_threshold_not_a_crash(tmp_path):
    metrics, result = run_dinov3(tmp_path, {"threshold_min_val_crops": 10_000})
    assert "calibrated_threshold" not in metrics
    assert all(math.isfinite(v) for v in metrics.values())
    manifest = bundle_json(read_bundle(result.path), "manifest.json")
    assert manifest["calibrated_threshold"] is None
    assert manifest["abstain_label"] == "unknown"
    assert "threshold_min_val_crops" in manifest["calibration"]["reason"]
    assert "calibrated_threshold" not in manifest["metrics"]


def test_dinov3_fixed_confidence_threshold(tmp_path):
    metrics, result = run_dinov3(tmp_path, {"confidence_threshold": 0.4})
    assert metrics["calibrated_threshold"] == 0.4
    assert (
        bundle_json(read_bundle(result.path), "manifest.json")["calibration"]["rule"]
        == "fixed"
    )


# --------------------------------------------------------------------------- conventional_classifier wiring


def test_conventional_records_threshold(tmp_path):
    from sparrow_finetune.pipelines.conventional_classifier import (
        ConventionalClassifierTrainer,
    )

    t = ConventionalClassifierTrainer()
    items = snapshot_items(balanced_items(), tmp_path / "img")
    params = {
        "model": "resnet18",
        "pretrained": False,
        "imgsz": 32,
        "epochs": 2,
        "warmup_epochs": 1,
        "patience": 0,
        "batch": 16,
        "device": "cpu",
        "threshold_min_val_crops": 1,
        "threshold_target_accuracy": 0.01,
    }
    ctx, _ = make_context(t, items, tmp_path / "work", params)
    t.prepare(ctx)
    t.train(ctx)
    metrics = t.evaluate(ctx)
    result = t.export(ctx)
    assert 0.0 <= metrics["calibrated_threshold"] <= 1.0
    assert metrics["threshold_coverage"] > 0
    manifest = bundle_json(read_bundle(result.path), "manifest.json")
    assert manifest["calibrated_threshold"] == pytest.approx(
        metrics["calibrated_threshold"]
    )
    assert manifest["abstain_label"] == "unknown"
    assert result.metadata["abstain_label"] == "unknown"
