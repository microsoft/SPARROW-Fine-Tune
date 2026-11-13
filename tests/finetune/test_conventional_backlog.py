"""Pins the conventional_classifier backlog fixes P3-N2..P3-N6 (docs/review/implement-conventional-classifier/backlog.md)."""

from __future__ import annotations

import hashlib
import json
from collections import Counter

import pytest
from PIL import Image

from sparrow_finetune.pipelines import conventional_classifier as cc
from sparrow_finetune.pipelines.conventional_classifier import (
    ConventionalClassifierTrainer,
)

from .dinov3_helpers import balanced_items, make_context, snapshot_items

BASE = {
    "model": "resnet18",
    "pretrained": False,
    "imgsz": 32,
    "epochs": 3,
    "warmup_epochs": 1,
    "patience": 0,
    "batch": 16,
    "device": "cpu",
}


def _items(tmp_path):
    syn = balanced_items(
        train_groups=4, val_groups=2, images_per_group=2, ungrouped_per_label=1
    )
    return snapshot_items(syn, tmp_path / "images")


def _context(tmp_path, items=None, recorder=None, **overrides):
    items = items if items is not None else _items(tmp_path)
    t = ConventionalClassifierTrainer()
    ctx, rec = make_context(
        t, items, tmp_path / "work", {**BASE, **overrides}, recorder=recorder
    )
    return t, ctx, rec


def _curve(tmp_path):
    return json.loads((tmp_path / "work" / "train.json").read_text())["epochs"]


# --------------------------------------------------------------------------- P3-N2


def _large_item(tmp_path, label):
    from sparrow_finetune.pipelines import SnapshotItem

    path = tmp_path / "large.png"
    im = Image.new("RGB", (1600, 1200), (200, 40, 40))
    im.save(path)
    data = path.read_bytes()
    return SnapshotItem(
        image_path=path,
        width=1600,
        height=1200,
        annotations=[
            {"label": label, "bbox": None},
            {"label": label, "bbox": [100, 100, 900, 600]},
            {"label": label, "bbox": [0, 0, 20, 10]},
        ],
        group=None,
        name="large.png",
        sha256=hashlib.sha256(data).hexdigest(),
    )


def test_p3n2_cached_crops_capped_and_each_source_decoded_once(tmp_path, monkeypatch):
    items = _items(tmp_path)
    large = _large_item(tmp_path, items[0].annotations[0]["label"])
    items.append(large)
    t, ctx, _ = _context(tmp_path, items=items)

    opened: Counter = Counter()
    real_open = Image.open

    def spy(fp, *a, **kw):
        opened[str(fp)] += 1
        return real_open(fp, *a, **kw)

    monkeypatch.setattr(Image, "open", spy)
    t.prepare(ctx)
    monkeypatch.setattr(Image, "open", real_open)

    sources = {str(it.image_path) for it in items}
    assert {p: n for p, n in opened.items() if p in sources} == {p: 1 for p in sources}

    index = json.loads((tmp_path / "work" / "crops.json").read_text())
    assert len(index) == sum(len(it.annotations) for it in items)
    cap = 2 * BASE["imgsz"]
    sizes = []
    for row in index[-3:]:
        with Image.open(tmp_path / "work" / "crops" / row["file"]) as im:
            sizes.append(im.size)
            assert im.format == "JPEG"
    assert max(sizes[0]) <= cap and max(sizes[1]) <= cap
    assert sizes[0] == (64, 48), "aspect ratio kept when downscaling"
    assert sizes[1] == (64, round(600 * 64 / 900))
    assert sizes[2] == (20, 10), "small crops are not upscaled"


def test_p3n2_prepare_is_deterministic(tmp_path):
    outs = []
    for run in ("a", "b"):
        t, ctx, _ = _context(tmp_path / run)
        t.prepare(ctx)
        crops = tmp_path / run / "work" / "crops"
        outs.append({p.name: p.read_bytes() for p in sorted(crops.iterdir())})
    assert outs[0] == outs[1]


# --------------------------------------------------------------------------- P3-N3


def test_p3n3_fixed_input_size_model_rejects_other_imgsz():
    t = ConventionalClassifierTrainer()
    with pytest.raises(ValueError, match="imgsz"):
        t.validate_params({"model": "vit_tiny_patch16_224", "imgsz": 64})
    assert (
        t.validate_params({"model": "vit_tiny_patch16_224", "imgsz": 224})["imgsz"]
        == 224
    )
    assert t.validate_params({"model": "resnet18", "imgsz": 64})["imgsz"] == 64


# --------------------------------------------------------------------------- P3-N4


def test_p3n4_val_pass_runs_in_fp32_even_when_amp_is_on(tmp_path, monkeypatch):
    import contextlib

    import torch

    t, ctx, _ = _context(tmp_path, epochs=2)
    t.prepare(ctx)
    monkeypatch.setattr(cc, "_amp_enabled", lambda params, device: True)
    # CPU stands in for CUDA: record the autocast requests instead of running fp16 on CPU.
    monkeypatch.setattr(torch, "autocast", lambda *a, **kw: contextlib.nullcontext())
    real = ConventionalClassifierTrainer._predict
    amps = []

    def spy(net, dataset, device, batch, *, amp=False):
        amps.append(amp)
        return real(net, dataset, device, batch, amp=False)

    monkeypatch.setattr(ConventionalClassifierTrainer, "_predict", staticmethod(spy))
    real_scaler = torch.amp.GradScaler
    monkeypatch.setattr(
        torch.amp,
        "GradScaler",
        lambda *a, **kw: real_scaler(*a, **{**kw, "enabled": False}),
    )
    t.train(ctx)
    assert amps and not any(amps), "epoch selection must use fp32 predictions"


def test_p3n4_best_epoch_val_f1_equals_evaluate(tmp_path):
    t, ctx, rec = _context(tmp_path, epochs=4)
    t.prepare(ctx)
    t.train(ctx)
    metrics = t.evaluate(ctx)
    best = int(metrics["best_epoch"])
    point = next(p for p in _curve(tmp_path) if p["epoch"] == best)
    assert point["val_macro_f1"] == metrics["val_macro_f1"]


# --------------------------------------------------------------------------- P3-N5


def test_p3n5_loss_read_once_per_epoch(tmp_path, monkeypatch):
    import torch

    t, ctx, _ = _context(tmp_path, batch=2)
    t.prepare(ctx)
    calls = []
    real = torch.Tensor.__float__

    def spy(self):
        calls.append(1)
        return real(self)

    monkeypatch.setattr(torch.Tensor, "__float__", spy)
    t.train(ctx)
    assert len(calls) <= BASE["epochs"]


def test_p3n5_train_loader_built_once(tmp_path, monkeypatch):
    import torch

    t, ctx, _ = _context(tmp_path)
    t.prepare(ctx)
    real = torch.utils.data.DataLoader
    train_loaders = []

    def spy(*a, **kw):
        if kw.get("sampler") is not None:
            train_loaders.append(kw)
        return real(*a, **kw)

    monkeypatch.setattr(torch.utils.data, "DataLoader", spy)
    t.train(ctx)
    assert len(train_loaders) == 1


def test_p3n5_epoch_sampler_order_seeded_per_seed_and_epoch():
    a, b = cc._EpochSampler(20, seed=3), cc._EpochSampler(20, seed=3)
    a.set_epoch(1)
    b.set_epoch(1)
    first = list(a)
    assert first == list(b)
    assert sorted(i for _e, i in first) == list(range(20))
    assert all(e == 1 for e, _i in first)
    a.set_epoch(2)
    assert [i for _e, i in a] != [i for _e, i in first]


def test_p3n5_persistent_workers_match_in_process_curve(tmp_path, monkeypatch):
    import torch

    real = torch.utils.data.DataLoader
    kws = []

    def spy(*a, **kw):
        if kw.get("sampler") is not None:
            kws.append(kw)
        return real(*a, **kw)

    monkeypatch.setattr(torch.utils.data, "DataLoader", spy)
    curves = []
    for workers in (0, 2):
        root = tmp_path / f"w{workers}"
        t, ctx, _ = _context(root, workers=workers)
        t.prepare(ctx)
        t.train(ctx)
        curves.append(_curve(root))
    assert kws[-1]["persistent_workers"] is True and kws[-1]["num_workers"] == 2
    assert curves[0] == curves[1]


# --------------------------------------------------------------------------- P3-N6


def test_p3n6_evaluate_checks_cancellation(tmp_path):
    from sparrow_finetune.pipelines import JobCancelled

    t, ctx, rec = _context(tmp_path, epochs=1)
    t.prepare(ctx)
    t.train(ctx)
    rec.cancel = True
    with pytest.raises(JobCancelled):
        t.evaluate(ctx)
    assert not (tmp_path / "work" / "evaluate.json").exists()


def test_p3n6_export_checks_cancellation(tmp_path):
    from sparrow_finetune.pipelines import JobCancelled

    t, ctx, rec = _context(tmp_path, epochs=1)
    t.prepare(ctx)
    t.train(ctx)
    t.evaluate(ctx)
    rec.cancel = True
    with pytest.raises(JobCancelled):
        t.export(ctx)
    assert not (tmp_path / "work" / "conventional_classifier_bundle.zip").exists()


def test_p3n6_class_floor_counts_training_crops(tmp_path):
    items = _items(tmp_path)
    t, ctx, _ = _context(tmp_path, items=items, class_floor=10_000)
    with pytest.raises(ValueError, match="class_floor requires at least 10000"):
        t.prepare(ctx)
