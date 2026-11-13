"""Pins the Phase 3 review findings (docs/review/implement-conventional-classifier/report.md)."""

from __future__ import annotations

import json

import pytest

from sparrow_finetune.pipelines.conventional_classifier import (
    ConventionalClassifierTrainer,
)

from .dinov3_helpers import balanced_items, make_context, snapshot_items

BASE = {
    "model": "resnet18",
    "pretrained": False,
    "imgsz": 32,
    "epochs": 2,
    "warmup_epochs": 1,
    "patience": 0,
    "batch": 16,
    "device": "cpu",
}


def _prepared(tmp_path, **overrides):
    syn = balanced_items(
        train_groups=4, val_groups=2, images_per_group=2, ungrouped_per_label=1
    )
    items = snapshot_items(syn, tmp_path / "images")
    t = ConventionalClassifierTrainer()
    ctx, rec = make_context(t, items, tmp_path / "work", {**BASE, **overrides})
    t.prepare(ctx)
    index = json.loads((tmp_path / "work" / "crops.json").read_text())
    return t, ctx, rec, sum(r["split"] == "train" for r in index)


@pytest.mark.parametrize("batch_of", ["remainder_one", "one"])
def test_i1_size_one_finetune_batch_does_not_crash_batchnorm(tmp_path, batch_of):
    t, ctx, rec, n_train = _prepared(tmp_path)
    # A train-mode BatchNorm over a 1x1 feature map cannot normalise a single sample.
    ctx.params["batch"] = n_train - 1 if batch_of == "remainder_one" else 1
    assert ctx.params["batch"] == 1 or n_train % ctx.params["batch"] == 1
    t.train(ctx)
    points = json.loads((tmp_path / "work" / "train.json").read_text())["epochs"]
    assert [p["phase"] for p in points] == ["warmup", "finetune"]


def test_i2_model_with_tuple_classifier_trains_and_exports(tmp_path):
    # Distilled heads (deit_*_distilled, levit_*, efficientformer_l*) return (head, head_dist).
    t, ctx, rec, _ = _prepared(tmp_path, model="levit_128s", imgsz=224, batch=32)
    t.train(ctx)
    metrics = t.evaluate(ctx)
    result = t.export(ctx)
    assert result.path.exists()
    assert 0.0 <= metrics["val_macro_f1"] <= 1.0
