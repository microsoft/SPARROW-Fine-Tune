"""FT-24: per-class detector metrics (parity with the workbench's per-class results).

``evaluate`` keeps the flat float metrics the worker stores, adding ``precision.<label>``,
``recall.<label>``, ``map50.<label>``, ``map50_95.<label>`` and ``instances.<label>`` for every finite value.
The structured table ``[{"label", "precision", "recall", "map50", "map50_95", "instances"}]`` (one row per
job label, nulls for a class absent from val) goes to ``evaluate.json["per_class_table"]`` and to the model
metadata ``per_class``.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from .dinov3_helpers import make_context
from .yolo_helpers import CPU_PARAMS


def fake_box():
    # classes 0 and 2 are present in val; class 1 has no val instance
    return SimpleNamespace(
        map50=0.6,
        map=0.4,
        mp=0.7,
        mr=0.5,
        p=np.array([0.9, 0.5]),
        r=np.array([0.8, 0.2]),
        ap50=np.array([0.85, 0.35]),
        ap=np.array([0.55, 0.25]),
        ap_class_index=np.array([0, 2]),
    )


def test_per_class_table_rows_cover_every_label():
    from sparrow_finetune.pipelines.yolo_detector import per_class_table

    table = per_class_table(fake_box(), ["cat", "dog", "fox"], np.array([4, 0, 3]))
    assert table == [
        {
            "label": "cat",
            "precision": 0.9,
            "recall": 0.8,
            "map50": 0.85,
            "map50_95": 0.55,
            "instances": 4,
        },
        {
            "label": "dog",
            "precision": None,
            "recall": None,
            "map50": None,
            "map50_95": None,
            "instances": 0,
        },
        {
            "label": "fox",
            "precision": 0.5,
            "recall": 0.2,
            "map50": 0.35,
            "map50_95": 0.25,
            "instances": 3,
        },
    ]


def test_per_class_table_without_counts_or_p_r():
    """Older/fake results lack ``p``/``r`` and ``nt_per_class``: those cells are null, nothing raises."""
    from sparrow_finetune.pipelines.yolo_detector import per_class_table

    box = SimpleNamespace(
        ap50=[0.5, float("nan")], ap=[0.3, 0.1], ap_class_index=[1, 0]
    )
    table = per_class_table(box, ["a", "b"], None)
    assert table == [
        {
            "label": "a",
            "precision": None,
            "recall": None,
            "map50": None,
            "map50_95": 0.1,
            "instances": None,
        },
        {
            "label": "b",
            "precision": None,
            "recall": None,
            "map50": 0.5,
            "map50_95": 0.3,
            "instances": None,
        },
    ]


class FakeVal:
    def __init__(self, source=None, *a, **k):
        pass

    def val(self, **kwargs):
        return SimpleNamespace(box=fake_box(), nt_per_class=np.array([4, 0, 3]))


def test_evaluate_stores_per_class_metrics(tmp_path, monkeypatch):
    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", FakeVal)
    trainer = __import__(
        "sparrow_finetune.pipelines.yolo_detector", fromlist=["YoloDetectorTrainer"]
    ).YoloDetectorTrainer()
    work = tmp_path / "work"
    ctx, _ = make_context(trainer, [], work, dict(CPU_PARAMS))
    ctx.labels = ["cat", "dog", "fox"]
    work.mkdir(parents=True, exist_ok=True)
    (work / "train.json").write_text(
        json.dumps({"weights": "w.pt", "device": "cpu", "best_epoch": 1})
    )
    split = {"items": 2, "boxes": 3}
    (work / "prepare.json").write_text(
        json.dumps({"split": {"train": split, "val": split}})
    )

    metrics = trainer.evaluate(ctx)
    assert all(isinstance(v, float) and math.isfinite(v) for v in metrics.values())
    assert metrics["precision.cat"] == 0.9 and metrics["recall.fox"] == 0.2
    assert metrics["map50.cat"] == 0.85 and metrics["map50_95.fox"] == 0.25
    assert metrics["instances.cat"] == 4.0 and metrics["instances.dog"] == 0.0
    assert "precision.dog" not in metrics and "map50.dog" not in metrics
    assert metrics["map50"] == 0.6  # overall mAP unchanged
    report = json.loads((work / "evaluate.json").read_text())
    assert [row["label"] for row in report["per_class_table"]] == ["cat", "dog", "fox"]
    assert report["per_class_table"][1]["map50"] is None
    assert report["missing_classes"] == ["dog"]


def test_real_run_exports_per_class_metadata(tmp_path):
    """Tiny real CPU run: every job label has a per-class row in the model metadata."""
    from .yolo_helpers import det_items, det_snapshot_items

    from sparrow_finetune.pipelines.yolo_detector import YoloDetectorTrainer

    trainer = YoloDetectorTrainer()
    items = det_snapshot_items(det_items(), tmp_path / "images")
    ctx, _ = make_context(trainer, items, tmp_path / "work", dict(CPU_PARAMS))
    trainer.prepare(ctx)
    trainer.train(ctx)
    metrics = trainer.evaluate(ctx)
    result = trainer.export(ctx)
    rows = result.metadata["per_class"]
    assert [r["label"] for r in rows] == list(ctx.labels)
    for row in rows:
        assert set(row) == {
            "label",
            "precision",
            "recall",
            "map50",
            "map50_95",
            "instances",
        }
        assert row["instances"] is None or isinstance(row["instances"], int)
        if row["map50"] is not None:
            assert metrics[f"map50.{row['label']}"] == pytest.approx(row["map50"])
