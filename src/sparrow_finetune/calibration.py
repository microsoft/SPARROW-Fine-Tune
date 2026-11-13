"""Calibrated confidence threshold with an abstain outcome, shared by the classifier pipelines.

Serving semantics follow the Studio fine-tune module (845dd51
``client/worker/model_finetuning/inference.py``, ``DEFAULT_CALIBRATED_THRESHOLD`` and
``REVIEW_REASON_BELOW_THRESHOLD``; ``training.py`` model card "Uncertainty"): a prediction whose
top-1 probability is BELOW the threshold is not asserted (it gets the abstain label and goes to
review); at or above it the top-1 label stands. A null threshold means "uncalibrated", which Studio
serves as threshold 1.0 (everything routed to review).

Studio declares the threshold but never selects one (its ``TrainingConfig.calibrated_threshold``
defaults to None and "no calibration stage writes one yet"). The selection rule here is ours:
on the validation split, the LOWEST threshold whose accepted predictions reach
``target_accuracy`` (selective accuracy), i.e. the largest coverage meeting the target. Neither
pipeline applies temperature scaling, so the probabilities are the ones the exported model emits.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

ABSTAIN_LABEL = "unknown"
DEFAULT_TARGET_ACCURACY = 0.95
DEFAULT_MIN_VAL_CROPS = 30
# Targets whose coverage the record also reports, so a user whose model abstains a lot can see
# what a lower target would buy before training again (FT-30).
COVERAGE_TARGETS = (0.95, 0.9, 0.85, 0.8)
RULE = "min_threshold_with_selective_accuracy_at_least_target"

# Train params shared by dinov3_linear and conventional_classifier:
# name -> (kind, default, check, requirement shown in the error)
THRESHOLD_PARAMS: dict[str, tuple[str, Any, Any, str]] = {
    "threshold_target_accuracy": (
        "float",
        DEFAULT_TARGET_ACCURACY,
        lambda v: 0 < v <= 1,
        "a number in (0, 1]",
    ),
    "threshold_min_val_crops": (
        "int",
        DEFAULT_MIN_VAL_CROPS,
        lambda v: v >= 1,
        "an integer >= 1",
    ),
    # A fixed threshold replaces the selection rule; coverage/accuracy are still measured.
    "confidence_threshold": (
        "optfloat",
        None,
        lambda v: v is None or 0 <= v <= 1,
        "a number in [0, 1], or null",
    ),
}


def _at(
    confidence: np.ndarray, correct: np.ndarray, threshold: float
) -> tuple[float, float]:
    accepted = confidence >= threshold
    n = int(accepted.sum())
    coverage = n / len(confidence) if len(confidence) else math.nan
    accuracy = float(correct[accepted].mean()) if n else math.nan
    return float(coverage), accuracy


def _min_threshold(
    confidence: np.ndarray, correct: np.ndarray, target_accuracy: float
) -> Optional[float]:
    """The lowest threshold whose accepted predictions reach ``target_accuracy``, or None."""
    n = len(confidence)
    order = np.argsort(-confidence, kind="stable")
    conf_sorted = confidence[order]
    cum_acc = np.cumsum(correct[order]) / np.arange(1, n + 1)
    # A cut is only valid after the last row of a run of equal confidences.
    boundary = np.append(conf_sorted[1:] < conf_sorted[:-1], True)
    ok = np.flatnonzero(boundary & (cum_acc >= target_accuracy))
    return float(conf_sorted[int(ok[-1])]) if len(ok) else None


def _coverage_by_target(confidence: np.ndarray, correct: np.ndarray) -> list[dict]:
    rows = []
    for target in COVERAGE_TARGETS:
        threshold = _min_threshold(confidence, correct, target)
        coverage = _at(confidence, correct, threshold)[0] if threshold is not None else 0.0
        rows.append({"target": target, "threshold": threshold, "coverage": coverage})
    return rows


def select_threshold(
    probs: np.ndarray,
    y: np.ndarray,
    *,
    target_accuracy: float = DEFAULT_TARGET_ACCURACY,
    min_val_crops: int = DEFAULT_MIN_VAL_CROPS,
    fixed: Optional[float] = None,
) -> dict:
    """The threshold record for validation ``probs`` [N, C] and labels ``y`` [N].

    Never raises on degenerate input: an empty or too small split, or an unreachable target, gives
    ``calibrated_threshold: None`` with a ``reason``.
    """
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y)
    n = int(len(y))
    out: dict = {
        "calibrated_threshold": None,
        "abstain_label": ABSTAIN_LABEL,
        "threshold_coverage": None,
        "threshold_accuracy": None,
        "rule": "fixed" if fixed is not None else RULE,
        "target_accuracy": float(target_accuracy),
        "val_crops": n,
        "coverage_by_target": [],
        "reason": None,
    }
    if n and probs.ndim == 2 and probs.shape[0] == n:
        confidence = probs.max(axis=1)
        correct = (probs.argmax(axis=1) == y).astype(np.float64)
    else:
        confidence = np.zeros(0)
        correct = np.zeros(0)
    if len(confidence) >= min_val_crops:
        out["coverage_by_target"] = _coverage_by_target(confidence, correct)
    if fixed is not None:
        out["calibrated_threshold"] = float(fixed)
        if len(confidence):
            out["threshold_coverage"], out["threshold_accuracy"] = _at(
                confidence, correct, float(fixed)
            )
        else:
            out["reason"] = (
                "validation split is empty: coverage and accuracy not measured"
            )
        return out
    if n < min_val_crops or not len(confidence):
        out["reason"] = (
            f"validation split has {n} crop(s) (< threshold_min_val_crops={min_val_crops}): "
            "no threshold calibrated"
        )
        return out
    threshold = _min_threshold(confidence, correct, target_accuracy)
    if threshold is None:
        out["reason"] = (
            f"no threshold reaches validation accuracy {target_accuracy:g} on accepted "
            "predictions: no threshold calibrated"
        )
        return out
    out["calibrated_threshold"] = threshold
    out["threshold_coverage"], out["threshold_accuracy"] = _at(
        confidence, correct, threshold
    )
    return out


def threshold_for_params(params: dict, probs: np.ndarray, y: np.ndarray) -> dict:
    """``select_threshold`` driven by a trainer's validated params (defaults when absent)."""
    return select_threshold(
        probs,
        y,
        target_accuracy=params.get(
            "threshold_target_accuracy", DEFAULT_TARGET_ACCURACY
        ),
        min_val_crops=params.get("threshold_min_val_crops", DEFAULT_MIN_VAL_CROPS),
        fixed=params.get("confidence_threshold"),
    )


def threshold_metrics(record: dict) -> dict[str, float]:
    """The numeric metrics view of ``record``. Model metrics are finite numbers only, so an
    undefined value (no threshold, or nothing accepted) is omitted; the bundle manifest still
    carries ``calibrated_threshold: null`` and the reason."""
    keys = ("calibrated_threshold", "threshold_coverage", "threshold_accuracy")
    return {k: float(record[k]) for k in keys if record[k] is not None}


def threshold_manifest(record: Optional[dict]) -> dict:
    """Bundle-manifest / export-metadata keys: ``calibrated_threshold``, ``abstain_label`` and the
    full ``calibration`` record (rule, target, coverage, accuracy, reason)."""
    if record is None:
        return {}
    return {
        "calibrated_threshold": record["calibrated_threshold"],
        "abstain_label": record["abstain_label"],
        "calibration": dict(record),
    }


def json_metrics(metrics: dict) -> dict:
    """``metrics`` with NaN replaced by None, for JSON files inside the bundle."""
    return {
        k: (None if isinstance(v, float) and not math.isfinite(v) else v)
        for k, v in metrics.items()
    }
