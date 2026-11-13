"""Shared helpers for the Phase 1C ``yolo_detector`` tests (docs/review/implement-yolo-detector/spec.md).

Nothing here imports ``sparrow_finetune`` or ``ultralytics`` at module level, so test collection succeeds
before the feature exists; code under test is imported inside the functions that need it.

Data design: 64x64 PNGs on a grey background (shade varies per image so every file has a distinct
sha256). Each image carries one coloured square of its own class (left half of the image); every
fourth image also carries a square of the other class (right half), so some label files have two lines.
Groups are chosen by asking ``assign_split`` (pinned by the Phase 1B tests), so the caller controls
which split receives boxes.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

DET_SIZE = 64  # square images, width == height
DET_LABELS = ("cat", "dog")
DET_COLOURS = {"cat": (220, 40, 40), "dog": (40, 40, 220)}
SEED = 0
VAL_FRACTION = 0.2

CPU_PARAMS = {"model": "yolo11n.yaml", "epochs": 2, "imgsz": 64, "batch": 4, "device": "cpu", "workers": 0}

PARAM_DEFAULTS = {
    "model": "yolo11n.pt",
    "epochs": 50,
    "imgsz": 640,
    "batch": 16,
    "seed": 0,
    "val_fraction": 0.2,
    "device": "auto",
    "patience": 10,
    "lr0": 0.01,
    "workers": 0,
    "near_dup_max_distance": 4,
}


@dataclass
class DetItem:
    name: str
    group: Optional[str]
    boxes: list  # [(label, [x, y, w, h])]
    png: bytes
    # extra annotations without a bbox (unit tests only; the HTTP API rejects them on detection datasets)
    bboxless_labels: list = field(default_factory=list)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.png).hexdigest()


def det_png(boxes: list, index: int, size: int = DET_SIZE) -> bytes:
    shade = 90 + (index * 7) % 80
    im = Image.new("RGB", (size, size), (shade, shade, shade))
    for label, (x, y, w, h) in boxes:
        im.paste(DET_COLOURS[label], (int(x), int(y), int(x + w), int(y + h)))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def det_boxes(label: str, index: int) -> list:
    rng = np.random.default_rng(500 + index)
    s = int(rng.integers(12, 22))
    x = int(rng.integers(0, DET_SIZE // 2 - s))
    y = int(rng.integers(0, DET_SIZE - s))
    boxes = [(label, [x, y, s, s])]
    if index % 4 == 0:
        other = DET_LABELS[1 - DET_LABELS.index(label)]
        s2 = int(rng.integers(10, 18))
        x2 = int(rng.integers(DET_SIZE // 2, DET_SIZE - s2))
        y2 = int(rng.integers(0, DET_SIZE - s2))
        boxes.append((other, [x2, y2, s2, s2]))
    return boxes


def pick_groups(prefix: str, label: str, want: dict, seed: int = SEED, val_fraction: float = VAL_FRACTION) -> list:
    """Group names whose ``assign_split`` side matches ``want`` = {"train": n, "val": m}."""
    from sparrow_finetune.pipelines.dinov3_linear import assign_split

    want = dict(want)
    chosen, i = [], 0
    while want.get("train", 0) or want.get("val", 0):
        group = f"{prefix}{label}-group-{i}"
        side = assign_split(group, seed=seed, val_fraction=val_fraction)
        if want.get(side, 0):
            want[side] -= 1
            chosen.append(group)
        i += 1
        assert i < 10_000, "assign_split never produced the wanted split"
    return chosen


def det_items(
    train_groups: int = 3,
    val_groups: int = 1,
    images_per_group: int = 2,
    prefix: str = "",
    backgrounds: int = 1,
) -> list[DetItem]:
    """Grouped detection items for both labels, plus ``backgrounds`` grouped images with no annotation."""
    out: list[DetItem] = []
    k = 0
    for label in DET_LABELS:
        for g, group in enumerate(pick_groups(prefix, label, {"train": train_groups, "val": val_groups})):
            for j in range(images_per_group):
                boxes = det_boxes(label, k)
                out.append(DetItem(f"{prefix}{label}-g{g}-{j}.png", group, boxes, det_png(boxes, k)))
                k += 1
    for b in range(backgrounds):
        out.append(DetItem(f"{prefix}background-{b}.png", f"{prefix}bg-group-{b}", [], det_png([], 900 + b)))
    return out


def det_snapshot_items(items: list[DetItem], image_dir: Path) -> list:
    """Write the PNGs and return ``SnapshotItem``s as the worker would hand them to a trainer."""
    from sparrow_finetune.pipelines import SnapshotItem

    image_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for pos, it in enumerate(items):
        path = image_dir / f"{pos:06d}.png"
        path.write_bytes(it.png)
        anns = [{"label": label, "bbox": [float(v) for v in bbox]} for label, bbox in it.boxes]
        anns += [{"label": label, "bbox": None} for label in it.bboxless_labels]
        out.append(
            SnapshotItem(
                image_path=path,
                width=DET_SIZE,
                height=DET_SIZE,
                annotations=anns,
                group=it.group,
                name=it.name,
                sha256=it.sha256,
            )
        )
    return out


def det_annotation_payload(items: list[DetItem]) -> list[dict]:
    out = []
    for it in items:
        entry: dict = {"image": it.name, "annotations": [{"label": lb, "bbox": list(bb)} for lb, bb in it.boxes]}
        if it.group is not None:
            entry["group"] = it.group
        out.append(entry)
    return out


def expected_side(item, seed: int = SEED, val_fraction: float = VAL_FRACTION) -> str:
    from sparrow_finetune.pipelines.dinov3_linear import assign_split

    return assign_split(item.group or item.name, seed=seed, val_fraction=val_fraction)


def expected_yolo(bbox, width: int, height: int) -> tuple:
    """Independent reference for ``to_yolo_label`` (clamp to the image, then normalised centre format)."""
    x, y, w, h = (float(v) for v in bbox)
    x0, y0 = min(max(x, 0.0), width), min(max(y, 0.0), height)
    x1, y1 = min(max(x + w, 0.0), width), min(max(y + h, 0.0), height)
    return ((x0 + x1) / 2 / width, (y0 + y1) / 2 / height, (x1 - x0) / width, (y1 - y0) / height)


def expected_counts(items) -> dict:
    """Per split: number of items, number of bbox annotations, and the set of groups."""
    out = {s: {"items": 0, "boxes": 0, "groups": set()} for s in ("train", "val")}
    for it in items:
        side = expected_side(it)
        out[side]["items"] += 1
        out[side]["boxes"] += sum(1 for a in it.annotations if a.get("bbox") is not None)
        if it.group:
            out[side]["groups"].add(it.group)
    return out


def read_dataset(dataset_dir: Path) -> dict:
    """``{"train"|"val": {image_sha256: [label lines]}}`` read from a written YOLO dataset."""
    out: dict = {}
    for split in ("train", "val"):
        out[split] = {}
        for img in sorted((dataset_dir / "images" / split).iterdir()):
            label_file = dataset_dir / "labels" / split / f"{img.stem}.txt"
            assert label_file.is_file(), f"missing label file for {img}"
            lines = [ln.split() for ln in label_file.read_text().splitlines() if ln.strip()]
            digest = hashlib.sha256(img.read_bytes()).hexdigest()
            assert digest not in out[split], f"image written twice in {split}: {img}"
            out[split][digest] = lines
    return out
