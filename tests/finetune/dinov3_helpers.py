"""Shared helpers for the Phase 1B ``dinov3_linear`` tests (docs/review/implement-dinov3-linear/spec.md).

Nothing here imports ``sparrow_finetune`` at module level, so test collection succeeds before the
feature exists; the code under test is imported inside the functions that need it.

Data design: every class has its own colour, so the mean RGB of a crop separates the classes linearly.
The tiny ONNX "embedder" computes exactly that (GlobalAveragePool over the normalised pixels -> [N, 3]),
and ``make_mean_rgb_embedder`` is the deterministic in-process equivalent used by the unit tests.
Images annotated with a bbox carry the class colour only inside the box (grey elsewhere), so a
trainer that skipped cropping would see diluted features.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image

CLASS_COLOURS = {
    "cat": (210, 50, 50),
    "dog": (50, 210, 50),
    "fox": (50, 50, 210),
}
BACKGROUND = (128, 128, 128)
SYN_SIZE = (32, 24)  # width, height
BBOX = [6, 4, 16, 12]  # x, y, w, h  (inside SYN_SIZE)


# --------------------------------------------------------------------------- ONNX embedder model


def write_mean_rgb_onnx(path: Path, *, flatten: bool = True) -> Path:
    """``pixel_values`` float32 [N,3,H,W] -> GlobalAveragePool -> (Flatten) -> [N,3] (or [N,3,1,1])."""
    import onnx
    from onnx import TensorProto, helper

    nodes = [helper.make_node("GlobalAveragePool", ["pixel_values"], ["pooled"])]
    out_name, out_shape = "pooled", ["N", 3, 1, 1]
    if flatten:
        nodes.append(helper.make_node("Flatten", ["pooled"], ["embedding"], axis=1))
        out_name, out_shape = "embedding", ["N", 3]
    graph = helper.make_graph(
        nodes,
        "mean_rgb",
        [helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, ["N", 3, "H", "W"])],
        [helper.make_tensor_value_info(out_name, TensorProto.FLOAT, out_shape)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- fake embedder


def mean_rgb(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32).mean(axis=(0, 1)) / 255.0


def make_mean_rgb_embedder(model_hash: str = "fake-mean-rgb-v1") -> Any:
    """A deterministic ``Embedder`` subclass (dim 3) that records every call."""
    from sparrow_finetune.embedders import Embedder

    class MeanRGBEmbedder(Embedder):
        id = "fake-mean-rgb"
        dim = 3

        def __init__(self) -> None:
            self.calls: list[int] = []
            self.sizes: list[tuple[int, int]] = []
            self.on_call: Optional[Callable[[int], None]] = None

        def embed(self, images):
            images = list(images)
            self.calls.append(len(images))
            self.sizes.extend(im.size for im in images)
            if self.on_call is not None:
                self.on_call(len(self.calls))
            if not images:
                return np.zeros((0, 3), dtype=np.float32)
            return np.stack([mean_rgb(im) for im in images]).astype(np.float32)

    MeanRGBEmbedder.model_hash = model_hash
    return MeanRGBEmbedder()


# --------------------------------------------------------------------------- synthetic data


@dataclass
class SynItem:
    name: str
    label: str
    group: Optional[str]
    bbox: Optional[list]
    png: bytes


def synthetic_png(label: str, index: int, bbox: Optional[list], size=SYN_SIZE) -> bytes:
    rng = np.random.default_rng(1000 + index)
    jitter = rng.integers(-15, 16, size=3)
    colour = tuple(int(np.clip(c + j, 0, 255)) for c, j in zip(CLASS_COLOURS[label], jitter))
    if bbox is None:
        im = Image.new("RGB", size, colour)
    else:
        im = Image.new("RGB", size, BACKGROUND)
        x, y, w, h = bbox
        im.paste(colour, (x, y, x + w, y + h))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def synthetic_items(
    labels=("cat", "dog", "fox"),
    groups_per_label: int = 8,
    images_per_group: int = 2,
    ungrouped_per_label: int = 1,
    prefix: str = "",
) -> list[SynItem]:
    """Several images per group, distinct groups per label; every third image is bbox-annotated."""
    out: list[SynItem] = []
    k = 0
    for label in labels:
        for g in range(groups_per_label):
            for j in range(images_per_group):
                bbox = list(BBOX) if k % 3 == 0 else None
                name = f"{prefix}{label}-g{g}-{j}.png"
                group = f"{prefix}{label}-group-{g}"
                out.append(SynItem(name, label, group, bbox, synthetic_png(label, k, bbox)))
                k += 1
        for u in range(ungrouped_per_label):
            bbox = list(BBOX) if k % 3 == 0 else None
            name = f"{prefix}{label}-loose-{u}.png"
            out.append(SynItem(name, label, None, bbox, synthetic_png(label, k, bbox)))
            k += 1
    return out


def balanced_items(
    labels=("cat", "dog", "fox"),
    train_groups: int = 4,
    val_groups: int = 2,
    images_per_group: int = 2,
    ungrouped_per_label: int = 1,
    prefix: str = "",
    seed: int = 0,
    val_fraction: float = 0.2,
) -> list[SynItem]:
    """Like ``synthetic_items`` but every label gets groups on BOTH sides of ``assign_split``.

    Group names are picked by asking ``assign_split`` (pinned separately by the B5 tests), so the val
    split holds every class whatever hash-to-fraction mapping the implementation uses.
    """
    from sparrow_finetune.pipelines.dinov3_linear import assign_split

    out: list[SynItem] = []
    k = 0
    for label in labels:
        want = {"train": train_groups, "val": val_groups}
        chosen: list[str] = []
        i = 0
        while want["train"] or want["val"]:
            group = f"{prefix}{label}-group-{i}"
            side = assign_split(group, seed=seed, val_fraction=val_fraction)
            if want[side]:
                want[side] -= 1
                chosen.append(group)
            i += 1
            assert i < 10_000, "assign_split never produced both splits"
        for g, group in enumerate(chosen):
            for j in range(images_per_group):
                bbox = list(BBOX) if k % 3 == 0 else None
                out.append(SynItem(f"{prefix}{label}-g{g}-{j}.png", label, group, bbox, synthetic_png(label, k, bbox)))
                k += 1
        for u in range(ungrouped_per_label):
            bbox = list(BBOX) if k % 3 == 0 else None
            out.append(SynItem(f"{prefix}{label}-loose-{u}.png", label, None, bbox, synthetic_png(label, k, bbox)))
            k += 1
    return out


def snapshot_items(items: list[SynItem], image_dir: Path) -> list:
    """Write the PNGs and return ``SnapshotItem``s as the worker would hand them to a trainer."""
    from sparrow_finetune.pipelines import SnapshotItem

    image_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for pos, it in enumerate(items):
        path = image_dir / f"{pos:06d}.png"
        path.write_bytes(it.png)
        out.append(
            SnapshotItem(
                image_path=path,
                width=SYN_SIZE[0],
                height=SYN_SIZE[1],
                annotations=[{"label": it.label, "bbox": it.bbox}],
                group=it.group,
                name=it.name,
                sha256=hashlib.sha256(it.png).hexdigest(),
            )
        )
    return out


# --------------------------------------------------------------------------- job context


class Recorder:
    """Collects emitted events and cancellation checks for a directly constructed ``JobContext``."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict]] = []
        self.checks = 0
        self.cancel = False
        self.cancel_on_first_train_event = False

    def emit(self, stage: str, message: str, metrics: dict) -> None:
        self.events.append((stage, message, dict(metrics or {})))
        if self.cancel_on_first_train_event and stage == "train" and "epoch" in (metrics or {}):
            self.cancel = True

    def check(self) -> None:
        from sparrow_finetune.pipelines import JobCancelled

        self.checks += 1
        if self.cancel:
            raise JobCancelled("cancelled by test")

    def stage_events(self, stage: str) -> list[dict]:
        return [m for s, _msg, m in self.events if s == stage]


def make_context(trainer, items, workdir: Path, params: Optional[dict] = None, recorder=None):
    from sparrow_finetune.pipelines import JobContext

    recorder = recorder or Recorder()
    workdir.mkdir(parents=True, exist_ok=True)
    ctx = JobContext(
        job_id=1,
        params=trainer.validate_params(params or {}),
        workdir=workdir,
        items=items,
        labels=sorted({a["label"] for it in items for a in it.annotations}),
        _emit=recorder.emit,
        _check_cancelled=recorder.check,
    )
    return ctx, recorder


# --------------------------------------------------------------------------- bundle reading


def read_bundle(path: Path) -> dict:
    """``{basename: bytes}`` for every member of an exported zip bundle."""
    with zipfile.ZipFile(path) as zf:
        return {Path(n).name: zf.read(n) for n in zf.namelist() if not n.endswith("/")}


def read_bundle_bytes(data: bytes) -> dict:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        return {Path(n).name: zf.read(n) for n in zf.namelist() if not n.endswith("/")}


def bundle_json(bundle: dict, name: str) -> dict:
    return json.loads(bundle[name].decode("utf-8"))


def head_probabilities(head_onnx: bytes, features: np.ndarray, n_labels: int) -> np.ndarray:
    """Run ``head.onnx`` and return its [N, n_labels] float probability output."""
    import onnxruntime as ort

    sess = ort.InferenceSession(head_onnx, providers=["CPUExecutionProvider"])
    (inp,) = sess.get_inputs()
    outs = sess.run(None, {inp.name: np.asarray(features, dtype=np.float32)})
    probs = [
        np.asarray(o)
        for o in outs
        if isinstance(o, np.ndarray) and o.ndim == 2 and o.shape[1] == n_labels and o.dtype.kind == "f"
    ]
    assert probs, f"head.onnx has no [N,{n_labels}] float output: {[getattr(o, 'shape', type(o)) for o in outs]}"
    return probs[0]


def crop_image(path: Path, bbox) -> Image.Image:
    from sparrow_finetune.pipelines.dinov3_linear import crop_box

    with Image.open(path) as im:
        im = im.convert("RGB")
        return im.crop(crop_box(bbox, im.width, im.height))
