"""Pins for YOLO backlog items FT-8 (per-class AP in evaluate) and FT-17 (sandbox restores real ultralytics paths)."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tests.finetune.dinov3_helpers import make_context
from tests.finetune.yolo_helpers import CPU_PARAMS

REPO = Path(__file__).resolve().parents[1]


def _new_trainer():
    from sparrow_finetune.pipelines.yolo_detector import YoloDetectorTrainer

    return YoloDetectorTrainer()


def _evaluate_with_box(tmp_path, monkeypatch, box, labels):
    import ultralytics

    model = SimpleNamespace(val=lambda **kw: SimpleNamespace(box=box))
    monkeypatch.setattr(ultralytics, "YOLO", lambda *a, **k: model)
    trainer = _new_trainer()
    ctx, _ = make_context(trainer, [], tmp_path / "work", CPU_PARAMS)
    ctx.labels = list(labels)
    work = tmp_path / "work"
    (work / "train.json").write_text(
        json.dumps({"weights": "w.pt", "device": "cpu", "best_epoch": 1})
    )
    split = {"items": 2, "boxes": 3}
    (work / "prepare.json").write_text(
        json.dumps({"split": {"train": split, "val": split}})
    )
    metrics = trainer.evaluate(ctx)
    return metrics, json.loads((work / "evaluate.json").read_text())


# --------------------------------------------------------------------------- FT-8


def test_evaluate_reports_per_class_ap(tmp_path, monkeypatch):
    """Per-class AP keyed by job label; absent and NaN classes are listed, not reported as numbers."""
    # all_ap rows follow ap_class_index: class 0 ("cat") and class 2 ("fox") are in val; class 1 ("dog") is not.
    all_ap = np.array(
        [
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
            [math.nan] * 10,
        ]
    )
    box = SimpleNamespace(
        map50=0.9,
        map=0.45,
        mp=0.5,
        mr=0.5,
        ap50=all_ap[:, 0],
        ap=all_ap.mean(1),
        ap_class_index=np.array([0, 2]),
    )
    metrics, report = _evaluate_with_box(
        tmp_path, monkeypatch, box, ["cat", "dog", "fox"]
    )
    assert metrics["ap50_95.cat"] == 0.45
    assert "ap50_95.dog" not in metrics and "ap50_95.fox" not in metrics
    assert all(isinstance(v, float) and math.isfinite(v) for v in metrics.values())
    assert report["per_class"] == {"cat": {"ap50": 0.9, "ap50_95": 0.45}}
    assert report["missing_classes"] == ["dog", "fox"]
    assert report["metrics"] == metrics
    assert report["nan_metrics"] == []


def test_evaluate_per_class_empty_without_val_predictions(tmp_path, monkeypatch):
    """ultralytics leaves the per-class arrays empty when nothing was evaluated: every class is missing."""
    box = SimpleNamespace(
        map50=0.0, map=0.0, mp=0.0, mr=0.0, ap50=[], ap=[], ap_class_index=[]
    )
    metrics, report = _evaluate_with_box(tmp_path, monkeypatch, box, ["cat", "dog"])
    assert not any(k.startswith("ap50_95.") for k in metrics)
    assert report["per_class"] == {}
    assert report["missing_classes"] == ["cat", "dog"]


# --------------------------------------------------------------------------- FT-17

_SANDBOX_SCRIPT = textwrap.dedent(
    """
    import json, sys
    from pathlib import Path

    assert "ultralytics" not in sys.modules
    from sparrow_finetune.pipelines.yolo_detector import _ultralytics_sandbox

    work = Path(sys.argv[1])
    with _ultralytics_sandbox(work, lambda name: str(work / name)):
        import ultralytics.utils as ul_utils
        import ultralytics.utils.checks as ul_checks
        import ultralytics.utils.dist as ul_dist

        inside = str(ul_utils.SETTINGS.file)
    print(json.dumps({
        "inside": inside,
        "paths": {
            "utils.USER_CONFIG_DIR": str(ul_utils.USER_CONFIG_DIR),
            "checks.USER_CONFIG_DIR": str(ul_checks.USER_CONFIG_DIR),
            "dist.USER_CONFIG_DIR": str(ul_dist.USER_CONFIG_DIR),
            "SETTINGS_FILE": str(ul_utils.SETTINGS_FILE),
            "SETTINGS.file": str(ul_utils.SETTINGS.file),
            "SETTINGS.file_path": str(ul_utils.SETTINGS.file_path),
            "PERSISTENT_CACHE.file_path": str(ul_utils.PERSISTENT_CACHE.file_path),
            "DATASETS_DIR": str(ul_utils.DATASETS_DIR),
            "WEIGHTS_DIR": str(ul_utils.WEIGHTS_DIR),
            "RUNS_DIR": str(ul_utils.RUNS_DIR),
        },
        "settings": dict(ul_utils.SETTINGS),
    }))
    """
)


def test_sandbox_first_import_restores_real_config_paths(tmp_path):
    """ultralytics first imported inside the sandbox: after exit no ultralytics path points into the workdir."""
    home, xdg, work = tmp_path / "home", tmp_path / "xdg", tmp_path / "work"
    real_dir = xdg / "Ultralytics"
    real_dir.mkdir(parents=True)
    home.mkdir()
    work.mkdir()
    custom_runs = str(tmp_path / "custom-runs")
    # A pre-existing user settings file with a customised runs_dir: after exit SETTINGS must hold the user's values.
    import ultralytics.utils as ul_utils

    user_settings = {**ul_utils.SETTINGS.defaults, "runs_dir": custom_runs}
    (real_dir / "settings.json").write_text(json.dumps(user_settings))

    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"YOLO_CONFIG_DIR", "MPLCONFIGDIR", "XDG_CACHE_HOME"}
    }
    env.update(
        HOME=str(home),
        XDG_CONFIG_HOME=str(xdg),
        YOLO_OFFLINE="true",
        CUDA_VISIBLE_DEVICES="",
        PYTHONPATH=os.pathsep.join([str(REPO / "src"), str(REPO)]),
    )
    proc = subprocess.run(
        [sys.executable, "-c", _SANDBOX_SCRIPT, str(work)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert Path(out["inside"]).is_relative_to(work.resolve())
    leaked = {
        k: v
        for k, v in out["paths"].items()
        if Path(v).resolve().is_relative_to(work.resolve())
    }
    assert leaked == {}
    paths = out["paths"]
    for key in (
        "utils.USER_CONFIG_DIR",
        "checks.USER_CONFIG_DIR",
        "dist.USER_CONFIG_DIR",
    ):
        assert paths[key] == str(real_dir)
    assert paths["SETTINGS_FILE"] == str(real_dir / "settings.json")
    assert paths["SETTINGS.file"] == str(real_dir / "settings.json")
    assert paths["SETTINGS.file_path"] == str(real_dir / "settings.json")
    assert paths["PERSISTENT_CACHE.file_path"] == str(
        real_dir / "persistent_cache.json"
    )
    assert out["settings"]["runs_dir"] == custom_runs
    assert paths["RUNS_DIR"] == custom_runs
    # The user's settings file is untouched.
    assert json.loads((real_dir / "settings.json").read_text()) == user_settings
