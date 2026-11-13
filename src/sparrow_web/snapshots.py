"""Persisting an :class:`~sparrow_yolo.dataset.AuditResult` to disk, and the
fingerprints used to tell one dataset state from another.

An audit of 60,000 images takes ~50 seconds and produces frames far too large for
SQLite — 60k image rows, ~80k box rows, and potentially tens of thousands of
issue rows. Writing them once as compressed CSV means the prepare step can
rehydrate them instead of re-auditing, and the database only has to hold
aggregates.

CSV rather than parquet purely to avoid adding pyarrow to the image; these frames
are small enough compressed (a few MB) that the difference does not matter.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from sparrow_yolo.dataset import AuditResult

IMAGES_CSV = "images.csv.gz"
BOXES_CSV = "boxes.csv.gz"
ISSUES_CSV = "issues.csv.gz"
SUMMARY_JSON = "summary.json"


def save_audit(result: AuditResult, out_dir: str | Path) -> Path:
    """Write an audit to ``out_dir``. Returns the directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.images.to_csv(out_dir / IMAGES_CSV, index=False, compression="gzip")
    result.boxes.to_csv(out_dir / BOXES_CSV, index=False, compression="gzip")
    result.issues.to_csv(out_dir / ISSUES_CSV, index=False, compression="gzip")
    (out_dir / SUMMARY_JSON).write_text(
        json.dumps({"summary": result.summary, "root": result.root,
                    "warnings": result.warnings}, indent=2),
        encoding="utf-8",
    )
    return out_dir


def load_audit(in_dir: str | Path) -> AuditResult:
    """Rehydrate an audit written by :func:`save_audit`.

    The ``image`` column holds absolute paths, which are only valid while the
    dataset stays put. They are rebuilt from ``rel_path`` plus the recorded root
    so a moved or remounted dataset still works.
    """
    in_dir = Path(in_dir)
    meta = json.loads((in_dir / SUMMARY_JSON).read_text(encoding="utf-8"))
    root = meta.get("root", "")

    images = pd.read_csv(in_dir / IMAGES_CSV, compression="gzip")
    boxes = pd.read_csv(in_dir / BOXES_CSV, compression="gzip")
    issues = pd.read_csv(in_dir / ISSUES_CSV, compression="gzip")

    if root:
        for frame in (images, boxes):
            if not frame.empty and "rel_path" in frame.columns:
                frame["image"] = frame["rel_path"].map(lambda r: str(Path(root) / r))

    # An all-empty column round-trips as NaN; the split and prepare steps expect
    # real booleans here.
    for col in ("readable", "has_xml"):
        if col in images.columns:
            images[col] = images[col].fillna(False).astype(bool)

    return AuditResult(
        images=images, boxes=boxes, issues=issues, root=root,
        warnings=list(meta.get("warnings", [])),
    )


def _blake(parts) -> str:
    h = hashlib.blake2b(digest_size=16)
    for p in parts:
        h.update(str(p).encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()


def content_hash(images: pd.DataFrame) -> str:
    """Content-true identity of a dataset state.

    Two snapshots with the same value hold the same data, whatever the mtimes
    say. Built from ``(rel_path, content_fp, n_boxes)`` so it moves when an image
    changes *or* when its labels change.
    """
    if images.empty:
        return _blake(["empty"])
    cols = [c for c in ("rel_path", "content_fp", "n_boxes") if c in images.columns]
    frame = images[cols].sort_values("rel_path")
    return _blake(frame.itertuples(index=False, name=None))


def quick_hash(images: pd.DataFrame) -> str:
    """Stat-only hash: a cheap "might have changed" trigger, never the answer.

    Cheap enough to recompute from a directory walk with no file opens, but it
    changes when a dataset is merely re-copied, so it must not be used to decide
    what actually changed.
    """
    if images.empty:
        return _blake(["empty"])
    cols = [c for c in ("rel_path", "size_bytes", "mtime_ns") if c in images.columns]
    frame = images[cols].sort_values("rel_path")
    return _blake(frame.itertuples(index=False, name=None))


def label_fingerprints(boxes: pd.DataFrame) -> Dict[str, int]:
    """``{rel_path: hash of that image's boxes}``.

    Detects re-tagging even when the image bytes are untouched and the XML was
    merely reformatted, because it hashes the canonical sorted box values rather
    than the file.
    """
    if boxes.empty or "rel_path" not in boxes.columns:
        return {}
    cols = ["class_name", "xmin", "ymin", "xmax", "ymax"]
    have = [c for c in cols if c in boxes.columns]
    out: Dict[str, int] = {}
    for rel, group in boxes.groupby("rel_path"):
        rows = sorted(
            tuple(round(float(v), 2) if isinstance(v, (int, float)) else str(v)
                  for v in row)
            for row in group[have].itertuples(index=False, name=None)
        )
        digest = hashlib.blake2b(str(rows).encode(), digest_size=8).digest()
        out[str(rel)] = int.from_bytes(digest, "big") >> 1
    return out


def summarise_classes(boxes: pd.DataFrame) -> Dict[str, Tuple[int, int]]:
    """``{class_name: (n_boxes, n_images)}``."""
    if boxes.empty:
        return {}
    counts = boxes.groupby("class_name").agg(
        n_boxes=("class_name", "size"), n_images=("image", "nunique")
    )
    return {str(k): (int(v.n_boxes), int(v.n_images)) for k, v in counts.iterrows()}


def diff_snapshots(
    before: pd.DataFrame, after: pd.DataFrame
) -> Dict[str, Any]:
    """What changed between two snapshots' image frames.

    Distinguishes *new images* from *re-tagged images*, because they mean very
    different things: new images grow the dataset, while re-tagging means
    existing labels were corrected and any model trained on them is now stale.
    """
    def index(frame: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        if frame.empty:
            return {}
        return {
            str(r.rel_path): {
                "content_fp": getattr(r, "content_fp", None),
                "n_boxes": int(getattr(r, "n_boxes", 0) or 0),
            }
            for r in frame.itertuples(index=False)
        }

    a, b = index(before), index(after)
    added = sorted(set(b) - set(a))
    removed = sorted(set(a) - set(b))
    common = set(a) & set(b)

    modified, relabelled = [], []
    for rel in common:
        fp_a, fp_b = a[rel]["content_fp"], b[rel]["content_fp"]
        if fp_a is not None and fp_b is not None and fp_a != fp_b:
            modified.append(rel)
        elif a[rel]["n_boxes"] != b[rel]["n_boxes"]:
            relabelled.append(rel)

    return {
        "images_added": len(added),
        "images_removed": len(removed),
        "images_modified": len(modified),
        "images_relabelled": len(relabelled),
        "boxes_added": sum(b[r]["n_boxes"] for r in added),
        "boxes_removed": sum(a[r]["n_boxes"] for r in removed),
        "sample_added": added[:50],
        "sample_relabelled": sorted(relabelled)[:50],
    }
