"""FT-36: ``batch="auto"`` for the conventional classifier.

The batch is sized from a probe of the chosen timm model on the training GPU; on a CPU it is the default 32.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .dinov3_helpers import balanced_items, make_context, snapshot_items
from .test_conventional_classifier import SMOKE, trainer


@pytest.mark.parametrize("value", ["auto", -1])
def test_auto_batch_is_accepted(value):
    assert trainer().validate_params({"batch": value})["batch"] == "auto"


@pytest.mark.parametrize("value", ["big", -2, 1025, True, 2.0, None])
def test_bad_batch_rejected(value):
    with pytest.raises(ValueError, match="batch"):
        trainer().validate_params({"batch": value})


def test_auto_batch_formula():
    from sparrow_finetune.pipelines.conventional_classifier import auto_batch

    # 40 GB free, 600 MB fixed, 50 MB per image: (40960*0.85 - 600) / 50 = 684 -> capped at 128.
    assert auto_batch(40960, 600, 50, n_train=10_000) == 128
    # Few training crops: at least 8 optimiser steps per epoch (400 / 8 = 50 -> 48).
    assert auto_batch(40960, 600, 50, n_train=400) == 48
    assert auto_batch(40960, 600, 50, n_train=20) == 8
    # Memory bound: (4096*0.85 - 600) / 200 = 14.4 -> 14 -> 8.
    assert auto_batch(4096, 600, 200, n_train=10_000) == 8
    # Below 8 the batch is not rounded; it never drops under 1.
    assert auto_batch(2048, 600, 200, n_train=10_000) == 5
    assert auto_batch(512, 600, 200, n_train=10_000) == 1


def _train(tmp_path: Path, **params):
    syn = balanced_items(
        train_groups=4, val_groups=2, images_per_group=2, ungrouped_per_label=1
    )
    items = snapshot_items(syn, tmp_path / "images")
    t = trainer()
    ctx, rec = make_context(
        t, items, tmp_path / "work", dict(SMOKE, epochs=2, **params)
    )
    t.prepare(ctx)
    t.train(ctx)
    return t, ctx, rec, json.loads((tmp_path / "work" / "train.json").read_text())


def test_auto_on_cpu_uses_the_default_and_reports_it(tmp_path):
    _, _, rec, trained = _train(tmp_path, batch="auto")
    hw = trained["hardware"]
    assert hw == {"device": "cpu", "batch": 32, "batch_source": "auto"}
    assert "training on cpu, batch 32 (auto)" in [e[1] for e in rec.events]


def test_fixed_batch_is_recorded_and_carried_to_the_bundle(tmp_path):
    from .dinov3_helpers import bundle_json, read_bundle

    t, ctx, rec, trained = _train(tmp_path, batch=4)
    assert (
        trained["hardware"]["batch"] == 4
        and trained["hardware"]["batch_source"] == "param"
    )
    assert "training on cpu, batch 4 (set by parameter)" in [
        e[1] for e in rec.events
    ]
    t.evaluate(ctx)
    result = t.export(ctx)
    assert result.metadata["hardware"]["batch"] == 4
    assert (
        bundle_json(read_bundle(result.path), "model_card.json")["hardware"]["batch"]
        == 4
    )


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="needs a CUDA GPU"
)
def test_auto_on_gpu_probes_the_model(tmp_path):
    _, _, rec, trained = _train(tmp_path, batch="auto", device="cuda:0", imgsz=64)
    hw = trained["hardware"]
    assert hw["batch_source"] == "auto" and hw["gpu_free_mb"] > 0
    assert hw["probe_mb_per_image"] > 0 and hw["probe_fixed_mb"] >= 0
    # The synthetic set is tiny, so the steps-per-epoch cap wins: 8.
    assert hw["batch"] == 8
    messages = [e[1] for e in rec.events]
    assert "measuring GPU memory for batch=auto" in messages
    assert any(
        m.startswith("training on GPU cuda:0") and m.endswith("batch 8 (auto)")
        for m in messages
    )


def test_verify_shrinks_a_batch_whose_measured_peak_is_over_budget(monkeypatch):
    """The line fit is only a first guess: the chosen batch is measured and shrunk until it fits."""
    import torch

    t = trainer()
    seen = []

    def fake_probe(params, n_labels, device, amp, batches):
        (b,) = batches
        seen.append(b)
        if b >= 112:
            raise torch.cuda.OutOfMemoryError("fake")
        return [100.0 + 90.0 * b]  # budget 0.85 * 10000 = 8500 MB -> fits up to b = 93

    monkeypatch.setattr(t, "_probe_peaks", fake_probe)
    hw = {"gpu_free_mb": 10000}
    assert t._verify_batch({}, 3, None, True, 128, hw) == 88
    assert seen == [128, 96, 88]
    assert hw["probe_verified_mb"] == round(100 + 90 * 88)
    # Within the probe batches nothing more is run.
    seen.clear()
    assert t._verify_batch({}, 3, None, True, 32, {"gpu_free_mb": 10000}) == 32 and seen == []
