"""FT-20: dinov3 one-vs-rest probabilities must not underflow to a uniform row.

``SGDClassifier.predict_proba`` keeps float32 input in float32, so ``expit`` returns 0 for every
class once all decision values are below about -88; sklearn then returns a uniform row (argmax =
class 0). The ONNX head computes the same ratio in log space and keeps the right class, so the
export parity gate failed (max_abs_delta 0.8, top1 0.98) on the real-data Engine e2e run.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from .dinov3_helpers import (
    balanced_items,
    bundle_json,
    make_context,
    read_bundle,
    snapshot_items,
)
from .test_dinov3_linear import MAX_EPOCHS, new_trainer

SHIFT = 300.0


def _fitted_model(n_classes=5, dim=16, seed=0):
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    x = rng.normal(size=(200, dim)).astype(np.float32)
    y = rng.integers(0, n_classes, size=200)
    x[np.arange(200), y] += 3.0
    scaler = StandardScaler().fit(x)
    clf = SGDClassifier(loss="log_loss", alpha=1e-4, average=True, random_state=0)
    clf.fit(scaler.transform(x), y)
    return Pipeline([("scaler", scaler), ("clf", clf)]), x


def test_ft20_ovr_proba_matches_sklearn_without_underflow():
    from sparrow_finetune.pipelines.dinov3_linear import _ovr_proba

    model, x = _fitted_model()
    np.testing.assert_allclose(
        _ovr_proba(model, x), model.predict_proba(x.astype(np.float64)), atol=1e-6
    )


def test_ft20_ovr_proba_keeps_argmax_when_every_class_underflows():
    from sparrow_finetune.pipelines.dinov3_linear import _ovr_proba

    model, x = _fitted_model()
    model.steps[-1][1].intercept_ = model.steps[-1][1].intercept_ - SHIFT
    decision = model.decision_function(x.astype(np.float64))
    assert (decision.max(axis=1) < -88).all()
    probs = _ovr_proba(model, x)
    assert np.isfinite(probs).all()
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    assert (probs.argmax(axis=1) == decision.argmax(axis=1)).all()


def test_ft20_underflowing_head_evaluates_and_exports_with_parity(tmp_path):
    items = snapshot_items(balanced_items(), tmp_path / "images")
    trainer, _ = new_trainer()
    workdir = tmp_path / "work"
    ctx, _rec = make_context(trainer, items, workdir, {"max_epochs": MAX_EPOCHS})
    trainer.prepare(ctx)
    trainer.train(ctx)

    model_path = workdir / "model.pkl"
    model = pickle.loads(model_path.read_bytes())
    model.steps[-1][1].intercept_ = model.steps[-1][1].intercept_ - SHIFT
    model_path.write_bytes(pickle.dumps(model))

    _x_tr, _y_tr, x_val, y_val = trainer._load_arrays(workdir)
    decision = model.decision_function(np.asarray(x_val, dtype=np.float64))
    assert (decision.max(axis=1) < -88).all()

    from sklearn.metrics import f1_score

    classes = np.arange(len(ctx.labels))
    expected_f1 = f1_score(
        y_val, decision.argmax(axis=1), labels=classes, average="macro", zero_division=0
    )
    metrics = trainer.evaluate(ctx)
    assert metrics["val_macro_f1"] == pytest.approx(expected_f1)

    result = trainer.export(ctx)
    parity = bundle_json(read_bundle(result.path), "manifest.json")["parity"]
    assert parity["max_abs_delta"] <= 1e-4
    assert parity["top1_agreement"] == 1.0


# --------------------------------------------------------------- FT-20 parity on near-ties


def test_ft20_parity_check_ignores_top1_flips_on_near_ties():
    from sparrow_finetune.pipelines import parity_check

    ref = np.array([[0.5, 0.5 - 1e-6, 1e-6], [0.9, 0.05, 0.05]])
    onnx = np.array([[0.5 - 1e-6, 0.5, 1e-6], [0.9, 0.05, 0.05]])
    parity = parity_check(onnx, ref, max_abs_delta_limit=1e-4)
    assert parity["top1_agreement"] == 1.0
    assert parity["near_ties"] == 1
    assert parity["max_abs_delta"] == pytest.approx(1e-6)


def test_ft20_parity_check_still_fails_on_real_disagreement():
    from sparrow_finetune.pipelines import parity_check

    ref = np.array([[0.6, 0.4], [0.9, 0.1]])
    with pytest.raises(RuntimeError, match="parity gate failed"):
        parity_check(np.array([[0.4, 0.6], [0.9, 0.1]]), ref, max_abs_delta_limit=1e-4)
    with pytest.raises(RuntimeError, match="parity gate failed"):
        parity_check(
            ref + np.array([[1e-3, -1e-3], [0, 0]]), ref, max_abs_delta_limit=1e-4
        )
    assert parity_check(
        np.zeros((0, 2)), np.zeros((0, 2)), max_abs_delta_limit=1e-4
    ) == {
        "max_abs_delta": 0.0,
        "top1_agreement": 1.0,
        "near_ties": 0,
    }


def test_ft20_head_with_exact_float32_tie_exports(tmp_path):
    """Classes 0 and 1 differ only below float32 resolution: float64 picks 1, the ONNX head ties."""
    items = snapshot_items(balanced_items(), tmp_path / "images")
    trainer, _ = new_trainer()
    workdir = tmp_path / "work"
    ctx, _rec = make_context(trainer, items, workdir, {"max_epochs": MAX_EPOCHS})
    trainer.prepare(ctx)
    trainer.train(ctx)

    model_path = workdir / "model.pkl"
    model = pickle.loads(model_path.read_bytes())
    clf = model.steps[-1][1]
    # float64 copies: SGD fitted on float32 stores float32 parameters, which would round the gap away.
    coef, intercept = clf.coef_.astype(np.float64), clf.intercept_.astype(np.float64)
    coef[1] = coef[0]
    intercept -= SHIFT
    intercept[0] += 50.0
    intercept[1] = intercept[0] + 1e-6
    clf.coef_, clf.intercept_ = coef, intercept
    model_path.write_bytes(pickle.dumps(model))

    trainer.evaluate(ctx)
    result = trainer.export(ctx)
    parity = bundle_json(read_bundle(result.path), "manifest.json")["parity"]
    assert parity["max_abs_delta"] <= 1e-4
    assert parity["top1_agreement"] == 1.0
    assert parity["near_ties"] > 0
