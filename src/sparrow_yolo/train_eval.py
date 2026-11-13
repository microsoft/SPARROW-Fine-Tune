"""In-kernel training, evaluation, and export.

Use this for evaluation, export, and quick experiments on small datasets. For
anything long-running prefer :mod:`sparrow_yolo.process_runner`: training inside
the Jupyter kernel dies with the kernel, and a dropped browser connection is
enough to lose hours of work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .hardware import parse_device
from .process_runner import resolve_batch, resolve_model_weights, resolve_workers


def train_yolo(
    model_weights: str,
    data_yaml: str,
    project: str = "/workspace/runs",
    name: str = "experiment",
    imgsz: int = 1024,
    epochs: int = 100,
    batch: Any = "auto",
    device: Any = "0",
    patience: int = 30,
    workers: Any = "auto",
    exist_ok: bool = True,
    **kwargs: Any,
):
    """Train in-process.

    ``batch="auto"`` measures with Ultralytics AutoBatch on a single GPU and
    falls back to a VRAM-sized estimate under DDP, where AutoBatch is unsupported
    and would silently become a fixed 16.
    """
    from ultralytics import YOLO

    resolved_batch = resolve_batch(batch, model_weights=model_weights, imgsz=imgsz, device=device)
    model = YOLO(resolve_model_weights(model_weights))
    return model.train(
        data=data_yaml,
        imgsz=imgsz,
        epochs=epochs,
        batch=resolved_batch,
        device=device,
        project=project,
        name=name,
        patience=patience,
        workers=resolve_workers(workers, device),
        exist_ok=exist_ok,
        **kwargs,
    )


def results_best_weights(results: Any) -> Optional[Path]:
    """``best.pt`` for a finished in-kernel run, taken from ``results.save_dir``.

    Always prefer this to building the path from the project name: Ultralytics
    increments the directory when one already exists.
    """
    save_dir = getattr(results, "save_dir", None)
    if not save_dir:
        return None
    p = Path(save_dir) / "weights" / "best.pt"
    return p if p.exists() else None


def evaluate_checkpoint(
    checkpoint: str,
    data_yaml: str,
    split: str = "test",
    imgsz: int = 1024,
    device: Any = "0",
    batch: int = 8,
    **kwargs: Any,
):
    """Evaluate a checkpoint on one split. Validation runs on a single GPU."""
    from ultralytics import YOLO

    ckpt = Path(checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    devices = parse_device(device)
    return YOLO(str(ckpt)).val(
        data=data_yaml,
        split=split,
        imgsz=imgsz,
        device=str(devices[0]) if devices else "cpu",
        batch=batch,
        plots=True,
        **kwargs,
    )


def export_onnx(
    checkpoint: str,
    imgsz: int = 1024,
    opset: Optional[int] = None,
    simplify: bool = True,
    dynamic: bool = False,
    half: bool = False,
    **kwargs: Any,
):
    """Export to ONNX. ``opset=None`` lets Ultralytics pick one valid for the model."""
    from ultralytics import YOLO

    ckpt = Path(checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    export_kwargs = dict(format="onnx", imgsz=imgsz, simplify=simplify, dynamic=dynamic, half=half, **kwargs)
    if opset is not None:
        export_kwargs["opset"] = opset
    return YOLO(str(ckpt)).export(**export_kwargs)
