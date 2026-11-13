"""FT-19: the dinov3 reference gate only fails a job on a validation set large enough to trust.

On ~50 validation crops a 0.02 macro-F1 margin is about one crop: on 30 real Amazon subsets of
that size the strict gate refused 5 good heads (docs/review/e2e-backlog-sweep/report.md). Below
``reference_min_val_crops`` (default 100) the job trains and the model carries a visible warning
(``metrics.reference_worse``, the model card, a job event, the API and the model page).
"""

from __future__ import annotations

import numpy as np
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


@pytest.fixture
def strong_reference(monkeypatch):
    import sparrow_finetune.pipelines.dinov3_linear as mod

    monkeypatch.setattr(mod, "_reference_macro_f1", lambda *a, **k: 1.5)


def _val_crops(workdir) -> int:
    return int((np.load(workdir / "splits.npy") == "val").sum())


def test_ft19_min_val_crops_default_and_validation():
    trainer, _ = new_trainer()
    assert trainer.validate_params({})["reference_min_val_crops"] == 100
    assert (
        trainer.validate_params({"reference_min_val_crops": 0})[
            "reference_min_val_crops"
        ]
        == 0
    )
    for bad in (-1, 1.5, "x"):
        with pytest.raises(ValueError, match="reference_min_val_crops"):
            trainer.validate_params({"reference_min_val_crops": bad})


def test_ft19_small_val_set_trains_with_warning(items, tmp_path, strong_reference):
    run = run_all(items, tmp_path / "work")
    n_val = _val_crops(tmp_path / "work")
    assert 0 < n_val < 100
    metrics = run["metrics"]
    assert metrics["reference_worse"] == 1.0
    assert metrics["reference_val_macro_f1"] == pytest.approx(1.5)
    card = bundle_json(read_bundle(run["result"].path), "model_card.json")
    gate = card["reference_gate"]
    assert gate["enabled"] is True
    assert gate["passed"] is False
    assert gate["enforced"] is False
    assert gate["min_val_crops"] == 100 and gate["val_crops"] == n_val
    assert "warning" in gate and str(n_val) in gate["warning"]
    events = [(m, d) for s, m, d in run["rec"].events if s == "train"]
    assert any(m == "reference gate warning" for m, _ in events), events


def test_ft19_gate_passing_reports_no_warning(items, tmp_path):
    run = run_all(items, tmp_path / "work")
    assert run["metrics"]["reference_worse"] == 0.0
    gate = bundle_json(read_bundle(run["result"].path), "model_card.json")[
        "reference_gate"
    ]
    assert gate["passed"] is True and "warning" not in gate


@pytest.mark.parametrize("offset, fails", [(0, True), (1, False)])
def test_ft19_threshold_boundary(items, tmp_path, strong_reference, offset, fails):
    """Enforced when val crops >= reference_min_val_crops, warned below."""
    from sparrow_finetune.pipelines.dinov3_linear import ReferenceGateError

    probe = tmp_path / "probe"
    trainer, _ = new_trainer()
    ctx, _ = make_context(trainer, items, probe, {"max_epochs": 1})
    trainer.prepare(ctx)
    n_val = _val_crops(probe)

    trainer, _ = new_trainer()
    work = tmp_path / "work"
    ctx, _ = make_context(
        trainer,
        items,
        work,
        {"max_epochs": MAX_EPOCHS, "reference_min_val_crops": n_val + offset},
    )
    trainer.prepare(ctx)
    if fails:
        with pytest.raises(ReferenceGateError, match=r"reference.*margin 0\.02"):
            trainer.train(ctx)
        assert not (work / "model.pkl").exists()
    else:
        trainer.train(ctx)
        assert (work / "model.pkl").exists()


def test_ft19_gate_disabled_reports_nothing(items, tmp_path, strong_reference):
    run = run_all(items, tmp_path / "work", {"reference_gate": False})
    assert "reference_worse" not in run["metrics"]
