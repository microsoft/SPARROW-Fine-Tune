"""Fake trainers registered into a per-test ``Registry``.

They do no ML; they record which stages ran and what the ``JobContext`` held, so
tests can check the worker's orchestration through observable behaviour only.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from sparrow_finetune.pipelines import ExportResult, Trainer

TRAIN_LOSS = 0.5
EVAL_METRICS = {"accuracy": 0.75}
FAIL_MESSAGE = "boom: simulated trainer failure"
BAD_EPOCHS_MESSAGE = "epochs must be a positive integer"


class RecordingTrainer(Trainer):
    id = "fake_cls"
    name = "Fake classifier (tests)"
    task = "classification"
    requires = frozenset()

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.job_ids: list[str] = []
        self.params: list[dict] = []
        self.items: list = []
        self.labels: list[str] = []
        self.exported: bytes = b""

    def validate_params(self, params):
        params = dict(params or {})
        epochs = params.get("epochs", 1)
        if not isinstance(epochs, int) or epochs < 1:
            raise ValueError(f"{BAD_EPOCHS_MESSAGE}, got {epochs!r}")
        params["epochs"] = epochs
        return params

    def prepare(self, ctx):
        self.calls.append("prepare")
        self.job_ids.append(str(ctx.job_id))
        self.params.append(dict(ctx.params))
        self.items = list(ctx.items)
        self.labels = list(ctx.labels)

    def train(self, ctx):
        self.calls.append("train")
        ctx.emit("train", "epoch 1", loss=TRAIN_LOSS)

    def evaluate(self, ctx):
        self.calls.append("evaluate")
        return dict(EVAL_METRICS)

    def export(self, ctx):
        self.calls.append("export")
        workdir = Path(ctx.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        path = workdir / "model.onnx"
        data = f"fake-onnx-model:{ctx.job_id}".encode()
        path.write_bytes(data)
        self.exported = data
        return ExportResult(path=path, format="onnx", metadata={"source": "tests"})


class FailingTrainer(RecordingTrainer):
    id = "fake_fail"
    name = "Fake failing trainer (tests)"

    def train(self, ctx):
        self.calls.append("train")
        raise RuntimeError(FAIL_MESSAGE)


class BlockingTrainer(RecordingTrainer):
    """Blocks in ``train`` until released, then polls ``check_cancelled``."""

    id = "fake_block"
    name = "Fake blocking trainer (tests)"

    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def train(self, ctx):
        self.calls.append("train")
        self.started.set()
        self.release.wait(timeout=15)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            ctx.check_cancelled()
            time.sleep(0.05)


class GpuTrainer(RecordingTrainer):
    id = "fake_gpu"
    name = "Fake GPU trainer (tests)"
    requires = frozenset({"gpu"})


class PlannedOverrideTrainer(RecordingTrainer):
    """Registers under a planned pipeline id to show it flips to available."""

    id = "dinov3_linear"
    name = "DINOv3 linear probe (test stand-in)"


@dataclass
class FakeTrainers:
    basic: RecordingTrainer = field(default_factory=RecordingTrainer)
    failing: FailingTrainer = field(default_factory=FailingTrainer)
    blocking: BlockingTrainer = field(default_factory=BlockingTrainer)
    gpu: GpuTrainer = field(default_factory=GpuTrainer)

    def all(self) -> list:
        return [self.basic, self.failing, self.blocking, self.gpu]
