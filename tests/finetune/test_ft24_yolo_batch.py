"""FT-24: ``batch="auto"`` (or -1) and the GPU check for ``yolo_detector``.

``batch`` keeps 16 as its default; ``"auto"`` is opt-in. On CUDA the batch is sized from the free memory of
the training GPU with the workbench heuristic (``sparrow_yolo.hardware.suggest_batch``); on the CPU it is a
fixed 16. The chosen batch and the device facts go to a train event, ``train.json``, the metrics
(``train_batch``) and the model metadata. A job that names a CUDA device on a worker without one fails in
``prepare``, before the dataset is written.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from .dinov3_helpers import make_context
from .yolo_helpers import CPU_PARAMS


def new_trainer():
    from sparrow_finetune.pipelines.yolo_detector import YoloDetectorTrainer

    return YoloDetectorTrainer()


# --------------------------------------------------------------------------- params


def test_batch_default_stays_16_and_auto_is_opt_in():
    t = new_trainer()
    assert t.validate_params({})["batch"] == 16
    assert t.validate_params({"batch": "auto"})["batch"] == "auto"
    assert t.validate_params({"batch": -1})["batch"] == "auto"
    assert t.validate_params({"batch": 32})["batch"] == 32


@pytest.mark.parametrize("bad", [0, 257, -2, True, "big", 8.0, None])
def test_batch_rejects_bad_values(bad):
    with pytest.raises(ValueError, match="batch"):
        new_trainer().validate_params({"batch": bad})


# --------------------------------------------------------------------------- heuristic


def fake_cuda(monkeypatch, free_gib: float, total_gib: float = 48.0, count: int = 1):
    import torch

    gib = 1024**3
    monkeypatch.setattr(torch.cuda, "is_available", lambda: count > 0)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda index=0: (int(free_gib * gib), int(total_gib * gib)),
    )
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index=0: "Fake GPU 48G")


def test_auto_batch_on_cpu_is_fixed():
    from sparrow_finetune.pipelines.yolo_detector import resolve_batch

    batch, hw = resolve_batch("auto", model="yolo11n.pt", imgsz=640, device="cpu")
    assert batch == 16
    assert hw == {"device": "cpu", "batch": 16, "batch_source": "auto"}


def test_explicit_batch_is_kept_and_recorded():
    from sparrow_finetune.pipelines.yolo_detector import resolve_batch

    batch, hw = resolve_batch(4, model="yolo11n.pt", imgsz=640, device="cpu")
    assert batch == 4 and hw["batch_source"] == "param" and hw["device"] == "cpu"


def test_unreadable_gpu_memory_falls_back_without_failing(monkeypatch):
    """A GPU whose memory cannot be probed keeps an explicit batch, and "auto" falls back to 16."""
    import torch

    from sparrow_finetune.pipelines.yolo_detector import resolve_batch

    def boom(index=0):
        raise RuntimeError("no driver")

    monkeypatch.setattr(torch.cuda, "mem_get_info", boom)
    batch, hw = resolve_batch(8, model="yolo11n.pt", imgsz=640, device="0")
    assert batch == 8 and "no driver" in hw["gpu_probe_error"]
    batch, hw = resolve_batch("auto", model="yolo11n.pt", imgsz=640, device="0")
    assert batch == 16 and hw["batch_source"] == "auto"


@pytest.mark.parametrize(
    ("model", "imgsz", "free_gib", "want"),
    [
        # 40 GiB free: (40*0.85 - 6) GiB / (310 MB * (640/1024)^2) -> 236, capped at 64
        ("yolo11n.pt", 640, 40.0, 64),
        # 10 GiB free, yolo11l at 1024: (8.5 - 6) GiB = 2560 MB / 1030 -> 2
        ("yolo11l.pt", 1024, 10.0, 2),
        # 12 GiB free, yolo11s at 1024: (10.2 - 6) GiB = 4300.8 MB / 490 -> 8 (even)
        ("yolo11s.yaml", 1024, 12.0, 8),
        # 13 GiB free, yolo11s at 1024: (11.05 - 6) GiB = 5171 MB / 490 -> 10 -> even 10
        ("yolo11s.pt", 1024, 13.0, 10),
        # 14 GiB free: (11.9-6) GiB = 6041 MB / 490 -> 12
        ("yolo11s.pt", 1024, 14.0, 12),
    ],
)
def test_auto_batch_on_cuda_uses_workbench_heuristic(
    monkeypatch, model, imgsz, free_gib, want
):
    from sparrow_finetune.pipelines.yolo_detector import resolve_batch

    fake_cuda(monkeypatch, free_gib)
    batch, hw = resolve_batch("auto", model=model, imgsz=imgsz, device="0")
    assert batch == want
    assert hw["device"] == "0"
    assert hw["gpu_name"] == "Fake GPU 48G"
    assert hw["gpu_total_mb"] == 48 * 1024
    assert hw["gpu_free_mb"] == int(free_gib * 1024)
    assert hw["batch"] == want and hw["batch_source"] == "auto"


def test_auto_batch_matches_workbench_suggest_batch(monkeypatch):
    """Parity with the workbench function the heuristic was ported from."""
    from sparrow_yolo.hardware import GpuInfo, suggest_batch

    from sparrow_finetune.pipelines.yolo_detector import resolve_batch

    for free in (7.0, 9.5, 16.0, 23.0, 47.0):
        for model in ("yolo11n.pt", "yolov8m.pt", "yolo11x.yaml", "custom.pt"):
            for imgsz in (320, 640, 1024):
                fake_cuda(monkeypatch, free)
                ours, _ = resolve_batch(
                    "auto", model=model, imgsz=imgsz, device="cuda:0"
                )
                gpu = GpuInfo(
                    index=0,
                    name="g",
                    total_mb=48 * 1024,
                    used_mb=int((48 - free) * 1024),
                )
                theirs = suggest_batch(model, imgsz, [0], [gpu])
                assert ours == theirs, (free, model, imgsz)


# --------------------------------------------------------------------------- GPU check


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "0"])
def test_prepare_fails_fast_without_cuda(tmp_path, monkeypatch, device):
    import torch

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    trainer = new_trainer()
    ctx, _ = make_context(
        trainer, [], tmp_path / "work", {**CPU_PARAMS, "device": device}
    )
    with pytest.raises(ValueError, match="no CUDA device"):
        trainer.prepare(ctx)
    assert not (tmp_path / "work" / "dataset").exists()


# --------------------------------------------------------------------------- train / evaluate wiring


class FakeModel:
    """Stands in for ``ultralytics.YOLO``: records ``train``/``val`` kwargs, writes a checkpoint."""

    calls: list = []

    def __init__(self, source=None, *a, **k):
        self.source = source
        self.trainer = None

    def add_callback(self, *a, **k):
        pass

    def train(self, **kwargs):
        FakeModel.calls.append(("train", kwargs))
        save_dir = Path(kwargs["project"]) / kwargs["name"]
        (save_dir / "weights").mkdir(parents=True, exist_ok=True)
        (save_dir / "weights" / "best.pt").write_bytes(b"ckpt")
        self.trainer = SimpleNamespace(save_dir=save_dir)

    def val(self, **kwargs):
        FakeModel.calls.append(("val", kwargs))
        box = SimpleNamespace(map50=0.5, map=0.25, mp=0.5, mr=0.5)
        return SimpleNamespace(box=box)


def test_auto_batch_reaches_ultralytics_and_is_recorded(tmp_path, monkeypatch):
    import ultralytics

    FakeModel.calls = []
    monkeypatch.setattr(ultralytics, "YOLO", FakeModel)
    trainer = new_trainer()
    ctx, rec = make_context(
        trainer, [], tmp_path / "work", {**CPU_PARAMS, "batch": "auto"}
    )
    trainer.train(ctx)
    ((_, kw),) = [c for c in FakeModel.calls if c[0] == "train"]
    assert kw["batch"] == 16
    trained = json.loads((tmp_path / "work" / "train.json").read_text())
    assert trained["batch"] == 16
    assert trained["hardware"] == {"device": "cpu", "batch": 16, "batch_source": "auto"}
    events = [(m, x) for s, m, x in rec.events if s == "train" and "batch" in x]
    assert events and events[0][1]["batch"] == 16
    assert "cpu" in events[0][0] and "16" in events[0][0]

    split = {"items": 2, "boxes": 3}
    (tmp_path / "work" / "prepare.json").write_text(
        json.dumps({"split": {"train": split, "val": split}})
    )
    metrics = trainer.evaluate(ctx)
    ((_, vkw),) = [c for c in FakeModel.calls if c[0] == "val"]
    assert vkw["batch"] == 16
    assert metrics["train_batch"] == 16.0


def test_megadetector_base_defaults_to_its_training_size():
    # FT-31: MegaDetector v6 extra was trained at 1280 px; an explicit imgsz still wins.
    t = new_trainer()
    assert t.validate_params({})["imgsz"] == 640
    assert t.validate_params({"model": "mdv6-yolov10-c"})["imgsz"] == 640
    assert t.validate_params({"model": "mdv6-yolov10-e"})["imgsz"] == 1280
    assert t.validate_params({"model": "mdv6-yolov10-e", "imgsz": 640})["imgsz"] == 640
