"""Pins for the YOLO review-backlog items (docs/review/implement-yolo-detector/backlog.md).

P1C-N1/N6, FT-13  a job writes nothing outside ``ctx.workdir`` (no ~/.config/Ultralytics, no weights in the cwd);
                  ``SPARROW_FT_WEIGHTS_DIR`` maps bare weights names to local files.
P1C-N2            a CUDA index the worker does not have fails before training starts, naming ``device``.
P1C-N3            NaN metrics are left out (listed in ``nan_metrics``), never reported as 0.0.
P1C-N5            an annotation label outside the job labels is a ValueError naming it.
P1B-N7            re-registering the built-ins with different settings replaces the stale trainer.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from .dinov3_helpers import make_context
from .yolo_helpers import CPU_PARAMS, det_items, det_snapshot_items

REPO = Path(__file__).resolve().parents[2]


def new_trainer(**kwargs):
    from sparrow_finetune.pipelines.yolo_detector import YoloDetectorTrainer

    return YoloDetectorTrainer(**kwargs)


def _files(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


# --------------------------------------------------------------------------- P1C-N1 / P1C-N6 / FT-13

_JOB_SCRIPT = textwrap.dedent(
    """
    import json, sys
    from pathlib import Path

    from tests.finetune.dinov3_helpers import make_context
    from tests.finetune.yolo_helpers import CPU_PARAMS, det_items, det_snapshot_items
    from sparrow_finetune.pipelines.yolo_detector import YoloDetectorTrainer

    base = Path(sys.argv[1])

    def tree(root):
        return sorted(str(p.relative_to(root)) for p in root.rglob("*"))

    def run(name):
        trainer = YoloDetectorTrainer()
        items = det_snapshot_items(det_items(prefix=name + "-"), base / name / "images")
        ctx, _ = make_context(trainer, items, base / name / "work", {**CPU_PARAMS, "epochs": 1})
        trainer.prepare(ctx)
        trainer.train(ctx)
        trainer.evaluate(ctx)
        trainer.export(ctx)

    run("a")
    # ultralytics is now imported with its config dir bound to job a: job b must not write there either.
    a_before = tree(base / "a")
    run("b")
    print(json.dumps({"a_unchanged": tree(base / "a") == a_before}))
    """
)


def test_job_writes_nothing_outside_workdir(tmp_path):
    """Two jobs in a fresh process: HOME, the process cwd and the first job's workdir gain no files."""
    home, cwd, base = tmp_path / "home", tmp_path / "cwd", tmp_path / "jobs"
    for d in (home, cwd, base):
        d.mkdir()
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in {"XDG_CONFIG_HOME", "YOLO_CONFIG_DIR", "MPLCONFIGDIR", "XDG_CACHE_HOME"}
    }
    env.update(
        HOME=str(home),
        YOLO_OFFLINE="true",
        CUDA_VISIBLE_DEVICES="",
        PYTHONPATH=os.pathsep.join([str(REPO / "src"), str(REPO)]),
    )
    proc = subprocess.run(
        [sys.executable, "-c", _JOB_SCRIPT, str(base)],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert json.loads(proc.stdout.strip().splitlines()[-1]) == {"a_unchanged": True}
    assert _files(home) == [], _files(home)
    assert _files(cwd) == [], _files(cwd)


def test_amp_check_weights_load_inside_workdir(tmp_path, monkeypatch):
    """The AMP check's bare ``YOLO("yolo11n.pt")`` gets the workdir path, so a download lands there."""
    import ultralytics
    import ultralytics.engine.trainer as ul_trainer
    import ultralytics.utils as ul_utils

    from sparrow_finetune.pipelines.yolo_detector import _ultralytics_sandbox

    calls = []
    monkeypatch.setattr(ultralytics, "YOLO", lambda name, *a, **k: calls.append(name))

    def fake_check_amp(model):
        from ultralytics import YOLO

        YOLO("yolo11n.pt")
        return True

    monkeypatch.setattr(ul_trainer, "check_amp", fake_check_amp)
    trainer = new_trainer()
    config_dir = ul_utils.USER_CONFIG_DIR
    with _ultralytics_sandbox(
        tmp_path, lambda name: trainer._model_source(name, tmp_path)
    ):
        assert (
            ul_utils.USER_CONFIG_DIR
            == tmp_path.resolve() / "ultralytics" / "Ultralytics"
        )
        assert Path(ul_utils.SETTINGS.file).is_relative_to(tmp_path.resolve())
        assert ul_trainer.check_amp(None) is True
    assert calls == [str(tmp_path.resolve() / "yolo11n.pt")]
    assert ul_utils.USER_CONFIG_DIR == config_dir
    assert ul_trainer.check_amp is fake_check_amp


def test_sandbox_points_default_runs_dir_into_workdir(tmp_path, monkeypatch):
    """The GPU AMP check predicts with no ``project``, so ultralytics saves under ``RUNS_DIR``; that default
    is relative to the cwd, which is read-only in the worker image (job 11, PermissionError: 'runs')."""
    import ultralytics.cfg as ul_cfg
    import ultralytics.utils as ul_utils
    from types import SimpleNamespace

    from sparrow_finetune.pipelines.yolo_detector import _ultralytics_sandbox

    monkeypatch.setattr(ul_cfg, "TESTS_RUNNING", False)
    before = (ul_cfg.RUNS_DIR, ul_utils.RUNS_DIR)
    trainer = new_trainer()
    with _ultralytics_sandbox(tmp_path, lambda n: trainer._model_source(n, tmp_path)):
        args = SimpleNamespace(
            task="detect", mode="predict", project=None, name=None, exist_ok=False
        )
        save_dir = ul_cfg.get_save_dir(args)
        assert save_dir.is_relative_to(tmp_path.resolve()), save_dir
        assert Path(ul_utils.RUNS_DIR).is_relative_to(tmp_path.resolve())
    assert (ul_cfg.RUNS_DIR, ul_utils.RUNS_DIR) == before


def test_model_source_keeps_bare_pt_downloads_in_workdir(tmp_path):
    trainer = new_trainer()
    assert trainer._model_source("yolo11n.pt", tmp_path) == str(
        tmp_path.resolve() / "yolo11n.pt"
    )
    assert trainer._model_source("yolo11n.yaml", tmp_path) == "yolo11n.yaml"
    local = new_trainer(base_model_resolver=lambda name: "/weights/" + name)
    assert local._model_source("yolo11n.pt", tmp_path) == "/weights/yolo11n.pt"


def test_resolve_local_weights_maps_into_weights_dir_and_rejects_escapes(tmp_path):
    from sparrow_finetune.pipelines import resolve_local_weights

    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "yolo11n.pt").write_bytes(b"w")
    (tmp_path / "x.pt").write_bytes(b"outside")
    assert resolve_local_weights("yolo11n.pt", weights) == str(
        (weights / "yolo11n.pt").resolve()
    )
    assert (
        resolve_local_weights("yolov8s.pt", weights) == "yolov8s.pt"
    )  # not there: unchanged
    assert resolve_local_weights("yolo11n.pt", None) == "yolo11n.pt"
    for bad in ("../x.pt", str(tmp_path / "x.pt"), "sub/yolo11n.pt", "..", ""):
        with pytest.raises(ValueError, match="bare file name"):
            resolve_local_weights(bad, weights)


def test_resolve_local_weights_rejects_symlink_out_of_weights_dir(tmp_path):
    from sparrow_finetune.pipelines import resolve_local_weights

    weights = tmp_path / "weights"
    weights.mkdir()
    (tmp_path / "x.pt").write_bytes(b"outside")
    (weights / "yolo11n.pt").symlink_to(tmp_path / "x.pt")
    with pytest.raises(ValueError, match="outside weights_dir"):
        resolve_local_weights("yolo11n.pt", weights)


def test_settings_reads_weights_dir_from_env(monkeypatch, tmp_path):
    from sparrow_finetune.config import Settings

    monkeypatch.setenv("SPARROW_FT_DATABASE_URL", "postgresql://u@h/db")
    monkeypatch.setenv("SPARROW_FT_WEIGHTS_DIR", str(tmp_path))
    assert Settings.from_env().weights_dir == tmp_path
    monkeypatch.setenv("SPARROW_FT_WEIGHTS_DIR", "  ")
    assert Settings.from_env().weights_dir is None
    monkeypatch.delenv("SPARROW_FT_WEIGHTS_DIR")
    assert Settings.from_env().weights_dir is None


def test_builtin_yolo_trainer_uses_weights_dir(tmp_path):
    from sparrow_finetune.config import Settings
    from sparrow_finetune.pipelines import Registry, register_builtin_pipelines

    (tmp_path / "yolo11n.pt").write_bytes(b"w")
    settings = Settings(
        database_url="postgresql://u@h/db",
        data_dir=tmp_path,
        embedder="",
        weights_dir=tmp_path,
    )
    reg = Registry()
    register_builtin_pipelines(reg, settings)
    trainer = reg.get("yolo_detector")
    assert trainer._model_source("yolo11n.pt", tmp_path / "work") == str(
        (tmp_path / "yolo11n.pt").resolve()
    )
    assert trainer._model_source("yolov8s.pt", tmp_path / "work") == str(
        (tmp_path / "work").resolve() / "yolov8s.pt"
    )


# --------------------------------------------------------------------------- P1C-N2


@pytest.mark.parametrize(
    ("device", "count"), [("cuda:1", 1), ("7", 1), ("cuda", 0), ("0", 0)]
)
def test_train_rejects_cuda_device_the_worker_lacks(
    tmp_path, monkeypatch, device, count
):
    import torch
    import ultralytics

    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(
        ultralytics,
        "YOLO",
        lambda *a, **k: pytest.fail("YOLO built for a missing device"),
    )
    trainer = new_trainer()
    ctx, _ = make_context(
        trainer, [], tmp_path / "work", {**CPU_PARAMS, "device": device}
    )
    with pytest.raises(ValueError, match="device"):
        trainer.train(ctx)


# --------------------------------------------------------------------------- P1C-N3


def test_split_finite_drops_nan_instead_of_zero():
    from sparrow_finetune.pipelines.yolo_detector import _split_finite

    kept, dropped = _split_finite(
        {"map50": math.nan, "recall": 0.5, "precision": None, "map50_95": math.inf}
    )
    assert kept == {"recall": 0.5}
    assert dropped == ["map50", "map50_95", "precision"]


def test_evaluate_omits_nan_metrics_and_lists_them(tmp_path, monkeypatch):
    import ultralytics

    box = SimpleNamespace(map50=math.nan, map=math.nan, mp=0.25, mr=0.5)
    model = SimpleNamespace(val=lambda **kw: SimpleNamespace(box=box))
    monkeypatch.setattr(ultralytics, "YOLO", lambda *a, **k: model)
    trainer = new_trainer()
    ctx, _ = make_context(trainer, [], tmp_path / "work", CPU_PARAMS)
    work = tmp_path / "work"
    (work / "train.json").write_text(
        json.dumps({"weights": "w.pt", "device": "cpu", "best_epoch": 1})
    )
    split = {"items": 2, "boxes": 3}
    (work / "prepare.json").write_text(
        json.dumps({"split": {"train": split, "val": split}})
    )
    metrics = trainer.evaluate(ctx)
    assert "map50" not in metrics and "map50_95" not in metrics
    assert metrics["precision"] == 0.25 and metrics["recall"] == 0.5
    assert all(math.isfinite(v) for v in metrics.values())
    report = json.loads((work / "evaluate.json").read_text())
    assert report["nan_metrics"] == ["map50", "map50_95"]
    assert report["metrics"] == metrics


# --------------------------------------------------------------------------- P1C-N5


def test_prepare_rejects_unknown_label_with_value_error(tmp_path):
    trainer = new_trainer()
    items = det_snapshot_items(det_items(prefix="u-"), tmp_path / "images")
    ctx, _ = make_context(trainer, items, tmp_path / "work", CPU_PARAMS)
    ctx.labels = [label for label in ctx.labels if label != "dog"]
    with pytest.raises(ValueError, match="'dog'"):
        trainer.prepare(ctx)


# --------------------------------------------------------------------------- P1B-N7


def test_reregistering_builtins_with_new_settings_replaces_them(tmp_path):
    from sparrow_finetune.config import Settings
    from sparrow_finetune.pipelines import Registry, register_builtin_pipelines

    base = dict(database_url="postgresql://u@h/db", data_dir=tmp_path)
    reg = Registry()
    register_builtin_pipelines(reg, Settings(**base, embedder="a"))
    dino, yolo, conv = (
        reg.get(i)
        for i in ("dinov3_linear", "yolo_detector", "conventional_classifier")
    )

    register_builtin_pipelines(reg, Settings(**base, embedder="a"))
    assert reg.get("dinov3_linear") is dino and reg.get("yolo_detector") is yolo

    register_builtin_pipelines(reg, Settings(**base, embedder="b"))
    assert reg.get("dinov3_linear") is not dino
    assert (
        reg.get("yolo_detector") is yolo and reg.get("conventional_classifier") is conv
    )

    register_builtin_pipelines(
        reg, Settings(**base, embedder="b", weights_dir=tmp_path)
    )
    assert reg.get("yolo_detector") is not yolo
    assert reg.ids().count("yolo_detector") == 1


def test_reregistering_builtins_keeps_a_custom_trainer(tmp_path):
    from sparrow_finetune.config import Settings
    from sparrow_finetune.pipelines import Registry, register_builtin_pipelines

    reg = Registry()
    custom = new_trainer()
    reg.register(custom)
    register_builtin_pipelines(
        reg,
        Settings(database_url="postgresql://u@h/db", data_dir=tmp_path, embedder=""),
    )
    assert reg.get("yolo_detector") is custom
