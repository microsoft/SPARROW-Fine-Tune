"""FT-4: the Studio reference-model gate ported into ``dinov3_linear``.

Studio (``client/worker/model_finetuning/training.py``, ``ReferenceGateError``) fits a full-batch
``LogisticRegression`` on the same scaled features as the SGD head and fails training when the reference's
val macro-F1 beats the head's by more than a margin (default 0.02). These tests pin that contract.
"""

from __future__ import annotations

import pytest

from .dinov3_helpers import (
    balanced_items,
    bundle_json,
    make_context,
    read_bundle,
    snapshot_items,
)
from .test_dinov3_linear import MAX_EPOCHS, new_trainer, run_all


@pytest.fixture
def items(tmp_path):
    return snapshot_items(balanced_items(), tmp_path / "images")


def test_ft4_gate_params_default_on_with_studio_margin():
    trainer, _ = new_trainer()
    params = trainer.validate_params({})
    assert params["reference_gate"] is True
    assert params["reference_margin"] == pytest.approx(0.02)


@pytest.mark.parametrize("value", [-0.1, 1.5, "x"])
def test_ft4_margin_out_of_range_is_rejected(value):
    trainer, _ = new_trainer()
    with pytest.raises(ValueError, match="reference_margin"):
        trainer.validate_params({"reference_margin": value})


def test_ft4_gate_passes_on_separable_data_and_reports_reference(items, tmp_path):
    run = run_all(items, tmp_path / "work")
    metrics = run["metrics"]
    assert 0.0 <= metrics["reference_val_macro_f1"] <= 1.0
    assert metrics["reference_val_macro_f1"] - metrics["val_macro_f1"] <= 0.02
    card = bundle_json(read_bundle(run["result"].path), "model_card.json")
    gate = card["reference_gate"]
    assert gate["enabled"] is True and gate["passed"] is True
    assert gate["margin"] == pytest.approx(0.02)
    assert gate["reference_val_macro_f1"] == pytest.approx(
        metrics["reference_val_macro_f1"]
    )
    assert gate["head_val_macro_f1"] == pytest.approx(metrics["val_macro_f1"])


def _force_strong_reference(monkeypatch):
    import sparrow_finetune.pipelines.dinov3_linear as mod

    monkeypatch.setattr(mod, "_reference_macro_f1", lambda *a, **k: 1.5)


def test_ft4_train_fails_when_head_trails_reference(items, tmp_path, monkeypatch):
    from sparrow_finetune.pipelines.dinov3_linear import ReferenceGateError

    _force_strong_reference(monkeypatch)
    trainer, _ = new_trainer()
    # FT-19: enforce on any validation size (the default only enforces from 100 crops up).
    ctx, _ = make_context(
        trainer,
        items,
        tmp_path / "work",
        {"max_epochs": MAX_EPOCHS, "reference_min_val_crops": 0},
    )
    trainer.prepare(ctx)
    with pytest.raises(ReferenceGateError, match=r"reference.*1\.5.*margin 0\.02"):
        trainer.train(ctx)
    assert not (tmp_path / "work" / "model.pkl").exists()


def test_ft4_gate_disabled_trains_and_skips_reference(items, tmp_path, monkeypatch):
    _force_strong_reference(monkeypatch)
    run = run_all(items, tmp_path / "work", {"reference_gate": False})
    assert "reference_val_macro_f1" not in run["metrics"]
    card = bundle_json(read_bundle(run["result"].path), "model_card.json")
    assert card["reference_gate"] == {"enabled": False}
