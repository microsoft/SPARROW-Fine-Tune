"""Trainer plug-in contract, the per-job context handed to trainers, and the trainer registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

TASKS = ("classification", "detection")
STAGES = ("prepare", "train", "evaluate", "export")

# Pipelines the service intends to offer; listed as "planned" until a trainer with the id is registered.
PLANNED_PIPELINES: dict[str, dict[str, str]] = {
    "dinov3_linear": {"name": "DINOv3 linear probe", "task": "classification"},
    "conventional_classifier": {
        "name": "Conventional image classifier",
        "task": "classification",
    },
    "yolo_detector": {"name": "YOLO11 object detector", "task": "detection"},
}


class JobCancelled(Exception):
    """Raised by ``JobContext.check_cancelled`` once cancellation was requested."""


@dataclass(frozen=True)
class ExportResult:
    path: Path
    format: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SnapshotItem:
    image_path: Path
    width: int
    height: int
    annotations: list  # [{"label": str, "bbox": [x, y, w, h] | None}]
    group: Optional[str]
    name: str
    sha256: str


@dataclass
class JobContext:
    job_id: int
    params: dict
    workdir: Path
    items: list[SnapshotItem]
    labels: list[str]
    _emit: Callable[[str, str, dict], None] = field(repr=False)
    _check_cancelled: Callable[[], None] = field(repr=False)
    # Warm start (``base_model_id``): {"id", "model_arch", "labels", "path", "sha256"} of the staged
    # base bundle, set by the worker before ``prepare``; None for a cold start.
    base_model: Optional[dict] = None

    def emit(self, stage: str, message: str = "", **metrics: Any) -> None:
        self._emit(stage, message, metrics)

    def check_cancelled(self) -> None:
        self._check_cancelled()


class Trainer(ABC):
    id: str
    name: str
    task: str
    requires: frozenset[str] = frozenset()
    # True when the trainer accepts ``base_model_id`` (warm start from an imported base model).
    supports_base_model: bool = False

    def validate_params(self, params: dict) -> dict:
        """Return normalised params; raise ``ValueError`` with a user-facing message."""
        return dict(params or {})

    @abstractmethod
    def prepare(self, ctx: JobContext) -> None: ...

    @abstractmethod
    def train(self, ctx: JobContext) -> None: ...

    @abstractmethod
    def evaluate(self, ctx: JobContext) -> dict[str, float]: ...

    @abstractmethod
    def export(self, ctx: JobContext) -> ExportResult: ...


class Registry:
    def __init__(self) -> None:
        self._trainers: dict[str, Trainer] = {}

    def register(self, trainer: Trainer, *, replace: bool = False) -> Trainer:
        if not getattr(trainer, "id", None):
            raise ValueError("trainer.id must be a non-empty string")
        if trainer.task not in TASKS:
            raise ValueError(
                f"trainer {trainer.id!r}: task must be one of {TASKS}, got {trainer.task!r}"
            )
        existing = self._trainers.get(trainer.id)
        if existing is not None and existing is not trainer and not replace:
            raise ValueError(f"a trainer with id {trainer.id!r} is already registered")
        self._trainers[trainer.id] = trainer
        return trainer

    def get(self, trainer_id: str) -> Optional[Trainer]:
        return self._trainers.get(trainer_id)

    def list(self) -> list[Trainer]:
        return list(self._trainers.values())

    def ids(self) -> list[str]:
        return list(self._trainers)

    def catalogue(self) -> list[dict]:
        """Planned pipelines merged with registered trainers (a registered id is ``available``)."""
        out: dict[str, dict] = {
            pid: {
                "id": pid,
                "name": p["name"],
                "task": p["task"],
                "status": "planned",
                "requires": [],
            }
            for pid, p in PLANNED_PIPELINES.items()
        }
        for t in self._trainers.values():
            out[t.id] = {
                "id": t.id,
                "name": getattr(t, "name", t.id),
                "task": t.task,
                "status": "available",
                "requires": sorted(t.requires),
            }
        return list(out.values())


default_registry = Registry()


def resolve_local_weights(name: str, weights_dir: Optional[Path]) -> str:
    """Map a bare weights file name to ``<weights_dir>/<name>`` when that file exists.

    Anything other than a bare file name (absolute paths, ``../x.pt``, ``a/b.pt``) is rejected so a
    job parameter cannot point outside ``weights_dir``. A missing file returns ``name`` unchanged.
    """
    if not isinstance(name, str) or not name or name in (".", ".."):
        raise ValueError(f"weights name must be a bare file name, got {name!r}")
    if Path(name).is_absolute() or Path(name).name != name or "\\" in name:
        raise ValueError(
            f"weights name must be a bare file name inside weights_dir, got {name!r}"
        )
    if weights_dir is None:
        return name
    root = Path(weights_dir).resolve()
    candidate = (root / name).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"weights {name!r} resolves outside weights_dir {str(root)!r}")
    return str(candidate) if candidate.is_file() else name


def parity_check(
    onnx_probs: Any, ref_probs: Any, *, max_abs_delta_limit: float
) -> dict:
    """ONNX export parity gate: ``{max_abs_delta, top1_agreement, near_ties}`` or ``RuntimeError``.

    Top-1 agreement is measured on rows whose reference top-two gap exceeds ``2 *
    max_abs_delta_limit``. Below that gap the two outputs can pick different classes while
    staying within the delta limit (float32 ONNX vs a float64 reference), so those rows are
    counted as ``near_ties`` instead of failing the export (FT-20).
    """
    import numpy as np

    onnx = np.asarray(onnx_probs, dtype=np.float64)
    ref = np.asarray(ref_probs, dtype=np.float64)
    if len(ref) == 0:
        return {"max_abs_delta": 0.0, "top1_agreement": 1.0, "near_ties": 0}
    max_abs_delta = float(np.max(np.abs(onnx - ref)))
    top2 = (
        np.sort(ref, axis=1)[:, -2:]
        if ref.shape[1] > 1
        else np.hstack([np.zeros_like(ref), ref])
    )
    decisive = (top2[:, 1] - top2[:, 0]) > 2 * max_abs_delta_limit
    agree = onnx.argmax(axis=1) == ref.argmax(axis=1)
    top1 = float(np.mean(agree[decisive])) if decisive.any() else 1.0
    near_ties = int((~decisive).sum())
    if not (max_abs_delta <= max_abs_delta_limit and top1 == 1.0):
        raise RuntimeError(
            f"ONNX export parity gate failed: max_abs_delta={max_abs_delta:.3g} "
            f"(limit {max_abs_delta_limit}), top1_agreement={top1:.4f} (required 1.0) "
            f"over {int(decisive.sum())} decisive rows, {near_ties} near-ties"
        )
    return {
        "max_abs_delta": max_abs_delta,
        "top1_agreement": top1,
        "near_ties": near_ties,
    }


def dynamo_onnx_bytes(
    module: Any,
    example: Any,
    *,
    input_name: str,
    output_name: str,
    graph_name: str,
    opset: int,
) -> bytes:
    """Export ``module`` with the dynamo ONNX exporter (dim 0 dynamic) and serialize it (FT-15).

    The graph gets a fixed name so identical runs produce identical bytes.
    """
    import warnings

    import torch

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        program = torch.onnx.export(
            module,
            (example,),
            input_names=[input_name],
            output_names=[output_name],
            dynamic_shapes=({0: torch.export.Dim("batch")},),
            opset_version=opset,
            dynamo=True,
            verbose=False,
        )
    proto = program.model_proto
    proto.graph.name = graph_name
    return proto.SerializeToString()


def register_builtin_pipelines(registry: Registry, settings: Any) -> None:
    """Register the built-in trainers on ``registry``.

    Idempotent for equal settings: the registered trainer object is kept. A built-in trainer
    registered from different settings (another embedder or weights_dir) is replaced.
    The embedder is built when a job runs, so an unset ``SPARROW_FT_EMBEDDER`` fails jobs, not startup.
    """
    from ..embedders import make_embedder
    from .conventional_classifier import ConventionalClassifierTrainer
    from .dinov3_linear import DinoV3LinearTrainer
    from .yolo_detector import YoloDetectorTrainer

    def install(
        trainer_id: str,
        key: tuple,
        build: Callable[[], Trainer],
        unavailable_reason: Optional[str] = None,
    ) -> None:
        existing = registry.get(trainer_id)
        if existing is not None:
            existing_key = getattr(existing, "_builtin_key", None)
            # A trainer registered by someone else (tests, plugins) is left alone.
            if existing_key is None or existing_key == key:
                return
        trainer = build()
        trainer._builtin_key = key  # type: ignore[attr-defined]
        # Workers skip claiming such a pipeline; the API still lists it (P1B-N8).
        trainer.unavailable_reason = unavailable_reason  # type: ignore[attr-defined]
        registry.register(trainer, replace=True)

    spec = settings.embedder
    weights_dir = getattr(settings, "weights_dir", None)
    weights_key = str(weights_dir) if weights_dir is not None else None
    install(
        DinoV3LinearTrainer.id,
        ("embedder", spec, "weights_dir", weights_key),
        lambda: DinoV3LinearTrainer(
            lambda: make_embedder(spec), weights_dir=weights_dir
        ),
        unavailable_reason=None if spec else "SPARROW_FT_EMBEDDER is not set",
    )

    resolver = (
        (lambda name: resolve_local_weights(name, weights_dir))
        if weights_dir is not None
        else None
    )
    install(
        YoloDetectorTrainer.id,
        ("weights_dir", weights_key),
        lambda: YoloDetectorTrainer(base_model_resolver=resolver),
    )
    install(
        ConventionalClassifierTrainer.id,
        ("weights_dir", weights_key),
        lambda: ConventionalClassifierTrainer(weights_dir=weights_dir),
    )


__all__ = [
    "ExportResult",
    "JobCancelled",
    "JobContext",
    "PLANNED_PIPELINES",
    "Registry",
    "STAGES",
    "SnapshotItem",
    "TASKS",
    "Trainer",
    "default_registry",
    "register_builtin_pipelines",
    "resolve_local_weights",
]
