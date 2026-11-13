"""Annotation batches: intake, review bookkeeping, and export.

A batch is a folder the workbench owns::

    <annotations_root>/<project-slug>/<batch-id>/
        images/        copied from wherever you pointed the picker
        annotations/   one Pascal VOC XML per reviewed or predicted image
        batch.json     what was taken in, and from where

Copied rather than referenced, for two reasons. The source is usually a
read-only mount, so the XML has to live somewhere else anyway; and a batch that
owns its images can be exported months later even if the original folder has
moved. The cost is disk, which is the cheap resource here.

Nothing in this module writes to a dataset the pipeline reads. Export is the one
operation that writes outside the workbench, it goes only where
``Settings.is_allowed_export_root`` permits, and it records what it did.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from sparrow_yolo.dataset import IMAGE_EXTS
from sparrow_yolo.xml_utils import read_voc_boxes, write_voc_xml

from .db import Database, dumps, utcnow

#: Review queue states, in the order an image moves through them.
STATES = ("pending", "predicted", "reviewed")

#: Filters offered by the review list. Kept here rather than in the template so
#: the page and the API cannot disagree about what "blank" means.
FILTERS = {
    "all": "All",
    "unreviewed": "Not yet reviewed",
    "reviewed": "Reviewed",
    "predicted": "AI-suggested, unchecked",
    "blank": "No animals found",
    "low": "Low confidence",
    "pending": "Not yet run",
}


class ExportRefused(Exception):
    """The destination is not somewhere this app is allowed to write."""


@dataclass
class Intake:
    n_copied: int
    n_skipped: int
    bytes_copied: int


def find_images(folder: Path) -> List[Path]:
    """Image files directly under ``folder`` and below it, sorted."""
    return sorted(p for p in folder.rglob("*")
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def copy_into_batch(source: Path, batch_dir: Path,
                    on_progress: Optional[Callable[[int, int], None]] = None) -> Intake:
    """Copy the images from ``source`` into ``batch_dir/images``.

    Flattens the tree, because the review UI and the VOC convention both key on
    a filename. A name that would collide gets its parent folders folded into the
    stem rather than being silently dropped — losing an image here would be
    invisible until someone noticed the count was wrong.
    """
    images_dir = batch_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    found = find_images(source)
    used: set = set()
    copied = skipped = total_bytes = 0

    for i, src in enumerate(found, 1):
        rel = src.relative_to(source)
        name = rel.name
        if name.lower() in used:
            name = "__".join(rel.parts)
        if name.lower() in used:                     # still colliding: number it
            name = f"{src.stem}__{i}{src.suffix}"
        used.add(name.lower())

        dst = images_dir / name
        try:
            shutil.copy2(src, dst)
            total_bytes += dst.stat().st_size
            copied += 1
        except OSError:
            skipped += 1
        if on_progress and (i % 50 == 0 or i == len(found)):
            on_progress(i, len(found))

    return Intake(n_copied=copied, n_skipped=skipped, bytes_copied=total_bytes)


def image_size(path: Path) -> Tuple[int, int]:
    """Real pixel dimensions, read from the file.

    The same rule the audit follows: the image is the authority on its own size,
    never the XML.
    """
    from PIL import Image

    with Image.open(path) as im:
        return int(im.width), int(im.height)


# ------------------------------------------------------------------ labels --

#: Species used in a batch, kept beside the images rather than in the database.
#: A label typed once has to be offered again on the next image and after a page
#: reload, and reading it back out of thousands of XML files on every page load
#: would not scale. Same principle as everywhere else here: the artifact on disk
#: is the record, this is just an index of it.
LABELS_FILE = "labels.json"


def known_labels(batch_dir: Path) -> List[str]:
    try:
        data = json.loads((Path(batch_dir) / LABELS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return sorted({str(x) for x in data if str(x).strip()})


def remember_labels(batch_dir: Path, labels: Iterable[str]) -> List[str]:
    """Add any new species to the batch's list. Returns the full list."""
    batch_dir = Path(batch_dir)
    merged = set(known_labels(batch_dir))
    merged.update(str(x).strip() for x in labels if str(x).strip())
    out = sorted(merged)
    try:
        (batch_dir / LABELS_FILE).write_text(json.dumps(out, indent=2), encoding="utf-8")
    except OSError:
        pass          # a lost autocomplete entry must never fail a save
    return out


def xml_for(batch_dir: Path, rel_path: str) -> Path:
    return batch_dir / "annotations" / (Path(rel_path).stem + ".xml")


def box_counts(boxes: Iterable[Dict[str, Any]]) -> Tuple[int, int, int, Optional[float]]:
    """``(total, ai, human, lowest confidence)`` for one image's boxes."""
    boxes = list(boxes)
    ai = sum(1 for b in boxes if str(b.get("source", "model")) == "model")
    scores = [float(b["score"]) for b in boxes if b.get("score") is not None]
    return len(boxes), ai, len(boxes) - ai, (min(scores) if scores else None)


# --------------------------------------------------------------- persistence --

class Batches:
    """Reads and writes for the annotation tables."""

    def __init__(self, db: Database):
        self.db = db

    def create(self, project_id: int, name: str, source_path: str, dir_path: str) -> int:
        now = utcnow()
        with self.db.write() as conn:
            cur = conn.execute(
                "INSERT INTO annotation_batches(project_id,name,source_path,dir_path,"
                "created_at,updated_at) VALUES (?,?,?,?,?,?)",
                (project_id, name, source_path, dir_path, now, now))
            return int(cur.lastrowid)

    def get(self, batch_id: int) -> Optional[Dict[str, Any]]:
        return self.db.one("SELECT * FROM annotation_batches WHERE id=?", (batch_id,))

    def list(self, project_id: int) -> List[Dict[str, Any]]:
        """Batches with their review progress, for the index page."""
        return self.db.query(
            "SELECT b.*,"
            " (SELECT COUNT(*) FROM annotation_images i WHERE i.batch_id=b.id"
            "   AND i.state='reviewed') AS n_reviewed,"
            " (SELECT COUNT(*) FROM annotation_images i WHERE i.batch_id=b.id"
            "   AND i.state='predicted') AS n_predicted,"
            " (SELECT COUNT(*) FROM annotation_publishes p WHERE p.batch_id=b.id)"
            "   AS n_publishes"
            " FROM annotation_batches b WHERE b.project_id=? AND b.state != 'deleted'"
            " ORDER BY b.id DESC", (project_id,))

    def register_images(self, batch_id: int, rel_paths: Iterable[str]) -> int:
        now = utcnow()
        rows = [(batch_id, rel, now) for rel in rel_paths]
        with self.db.write() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO annotation_images(batch_id,rel_path,updated_at)"
                " VALUES (?,?,?)", rows)
            conn.execute(
                "UPDATE annotation_batches SET n_images="
                "(SELECT COUNT(*) FROM annotation_images WHERE batch_id=?),"
                " updated_at=? WHERE id=?", (batch_id, now, batch_id))
        return len(rows)

    def images(self, batch_id: int, filter: str = "all",
               low_confidence_below: float = 0.5) -> List[Dict[str, Any]]:
        where, params = ["batch_id = ?"], [batch_id]
        if filter == "unreviewed":
            where.append("state != 'reviewed'")
        elif filter in ("reviewed", "predicted", "pending"):
            where.append("state = ?")
            params.append(filter)
        elif filter == "blank":
            where.append("state != 'pending' AND n_boxes = 0")
        elif filter == "low":
            where.append("min_confidence IS NOT NULL AND min_confidence < ?")
            params.append(low_confidence_below)
        return self.db.query(
            f"SELECT * FROM annotation_images WHERE {' AND '.join(where)}"
            " ORDER BY rel_path", tuple(params))

    def image(self, batch_id: int, rel_path: str) -> Optional[Dict[str, Any]]:
        return self.db.one(
            "SELECT * FROM annotation_images WHERE batch_id=? AND rel_path=?",
            (batch_id, rel_path))

    def forget_image(self, batch_id: int, rel_path: str) -> None:
        now = utcnow()
        with self.db.write() as conn:
            conn.execute("DELETE FROM annotation_images WHERE batch_id=? AND rel_path=?",
                         (batch_id, rel_path))
            conn.execute(
                "UPDATE annotation_batches SET n_images="
                "(SELECT COUNT(*) FROM annotation_images WHERE batch_id=?),"
                " updated_at=? WHERE id=?", (batch_id, now, batch_id))

    def record_boxes(self, batch_id: int, rel_path: str, boxes: List[Dict[str, Any]], *,
                     state: str, model_version_id: Optional[int] = None,
                     width: Optional[int] = None, height: Optional[int] = None) -> None:
        """Update one image's aggregates after a prediction or a human save."""
        n, ai, human, low = box_counts(boxes)
        now = utcnow()
        with self.db.write() as conn:
            conn.execute(
                "UPDATE annotation_images SET state=?, n_boxes=?, n_ai_boxes=?,"
                " n_human_boxes=?, min_confidence=?, model_version_id=COALESCE(?, model_version_id),"
                " width=COALESCE(?, width), height=COALESCE(?, height),"
                " reviewed_at=CASE WHEN ?='reviewed' THEN ? ELSE reviewed_at END,"
                " updated_at=? WHERE batch_id=? AND rel_path=?",
                (state, n, ai, human, low, model_version_id, width, height,
                 state, now, now, batch_id, rel_path))

    def progress(self, batch_id: int) -> Dict[str, Any]:
        row = self.db.one(
            "SELECT COUNT(*) AS total,"
            " SUM(state='reviewed') AS reviewed,"
            " SUM(state='predicted') AS predicted,"
            " SUM(state='pending') AS pending,"
            " SUM(state!='pending' AND n_boxes=0) AS blank,"
            " SUM(n_boxes) AS boxes"
            " FROM annotation_images WHERE batch_id=?", (batch_id,))
        return {k: (v or 0) for k, v in (row or {}).items()}

    def record_pass(self, batch_id: int, params: Dict[str, Any],
                    result: Dict[str, Any], job_id: Optional[int] = None) -> int:
        with self.db.write() as conn:
            cur = conn.execute(
                "INSERT INTO annotation_passes(batch_id,model_version_id,job_id,conf,iou,"
                "imgsz,max_det,force_class,only_classes_json,only_missing,n_images,"
                "n_boxes,n_blank,n_errors,duration_s,created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (batch_id, params.get("model_version_id"), job_id,
                 params.get("conf"), params.get("iou"), params.get("imgsz"),
                 params.get("max_det"), params.get("force_class", "") or "",
                 dumps(params.get("only_classes", [])),
                 1 if params.get("only_missing") else 0,
                 result.get("n_images", 0), result.get("n_boxes", 0),
                 result.get("n_blank", 0), result.get("n_errors", 0),
                 result.get("duration_s"), utcnow()))
            return int(cur.lastrowid)

    def passes(self, batch_id: int) -> List[Dict[str, Any]]:
        return self.db.query(
            "SELECT p.*, m.version AS model_version FROM annotation_passes p"
            " LEFT JOIN model_versions m ON m.id = p.model_version_id"
            " WHERE p.batch_id=? ORDER BY p.id DESC", (batch_id,))

    def publishes(self, batch_id: int) -> List[Dict[str, Any]]:
        return self.db.query(
            "SELECT * FROM annotation_publishes WHERE batch_id=?"
            " ORDER BY id DESC", (batch_id,))


# -------------------------------------------------------------------- export --

def publish(db: Database, settings, batch: Dict[str, Any], dest: str | Path, *,
            reviewed_only: bool = True, include_images: bool = True) -> Dict[str, Any]:
    """Copy a batch's images and XML to ``dest``. Returns the manifest.

    Refuses any destination outside ``settings.export_roots``. That check is the
    whole reason export takes a separate, differently-mounted allowlist: this is
    the only code path in the app that writes to a folder the user names.
    """
    dest = Path(dest)
    if not settings.is_allowed_export_root(dest):
        raise ExportRefused(
            f"{dest} is outside the folders this app may write to.")

    batches = Batches(db)
    batch_dir = Path(batch["dir_path"])
    # Not "all": an image nobody has run a model over has no annotation, and
    # exporting it would quietly add an untagged file to someone's dataset.
    rows = [r for r in batches.images(batch["id"], "reviewed" if reviewed_only else "all")
            if r["state"] != "pending"]

    images_out = dest / "images"
    xml_out = dest / "annotations"
    xml_out.mkdir(parents=True, exist_ok=True)
    if include_images:
        images_out.mkdir(parents=True, exist_ok=True)

    entries: List[Dict[str, Any]] = []
    n_images = n_xml = 0
    for row in rows:
        rel = row["rel_path"]
        src_img = batch_dir / "images" / rel
        src_xml = xml_for(batch_dir, rel)
        if not src_img.exists():
            continue

        if src_xml.exists():
            shutil.copy2(src_xml, xml_out / src_xml.name)
            n_xml += 1
        elif row["state"] == "reviewed":
            # A reviewed image with no boxes still needs a file: an object-free
            # XML is what marks it a confirmed blank rather than untagged, and
            # that distinction is the difference between a useful negative and a
            # image the pipeline will refuse to train on.
            write_voc_xml(xml_out / (Path(rel).stem + ".xml"), src_img,
                          row["width"] or 0, row["height"] or 0, [], reviewed=True)
            n_xml += 1

        if include_images:
            shutil.copy2(src_img, images_out / src_img.name)
        n_images += 1
        entries.append({"image": rel, "state": row["state"], "boxes": row["n_boxes"],
                        "ai_boxes": row["n_ai_boxes"], "human_boxes": row["n_human_boxes"]})

    manifest = {
        "batch": {"id": batch["id"], "name": batch["name"],
                  "source_path": batch["source_path"]},
        "exported_at": utcnow(),
        "reviewed_only": reviewed_only,
        "include_images": include_images,
        "n_images": n_images,
        "n_xml": n_xml,
        "passes": [
            {k: p[k] for k in ("model_version", "conf", "iou", "imgsz",
                               "force_class", "n_images", "n_boxes", "created_at")}
            for p in batches.passes(batch["id"])
        ],
        "images": entries,
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with db.write() as conn:
        conn.execute(
            "INSERT INTO annotation_publishes(batch_id,dest_path,n_images,n_xml,"
            "include_images,reviewed_only,manifest_json,created_at)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (batch["id"], str(dest), n_images, n_xml,
             1 if include_images else 0, 1 if reviewed_only else 0,
             dumps({k: v for k, v in manifest.items() if k != "images"}), utcnow()))
        conn.execute("UPDATE annotation_batches SET state='published', updated_at=?"
                     " WHERE id=?", (utcnow(), batch["id"]))

    return manifest


def delete_image(batch_dir: Path, rel_path: str) -> Dict[str, bool]:
    """Remove one image and its annotation from a batch.

    Only the batch's own copy. The folder the images came from is never touched —
    that is the point of copying them in, and it is what makes this safe to do on
    a frame that is out of focus or triggered by a branch.
    """
    batch_dir = Path(batch_dir)
    image = batch_dir / "images" / rel_path
    xml = xml_for(batch_dir, rel_path)

    # Same containment check as serving a file: rel_path arrives from a URL.
    root = (batch_dir / "images").resolve()
    try:
        if not image.resolve().is_relative_to(root):
            raise ValueError(f"{rel_path} is outside the batch")
    except OSError as e:
        raise ValueError(str(e)) from e

    removed = {"image": False, "annotation": False}
    for key, path in (("image", image), ("annotation", xml)):
        try:
            path.unlink()
            removed[key] = True
        except FileNotFoundError:
            pass
    return removed


def load_boxes(batch_dir: Path, rel_path: str) -> List[Dict[str, Any]]:
    return read_voc_boxes(xml_for(batch_dir, rel_path))


def save_boxes(batch_dir: Path, rel_path: str, boxes: List[Dict[str, Any]],
               width: int, height: int, *, predicted_by: str = "") -> Path:
    remember_labels(batch_dir, (b.get("label", "") for b in boxes))
    return write_voc_xml(xml_for(batch_dir, rel_path), batch_dir / "images" / rel_path,
                         width, height, boxes, reviewed=True, predicted_by=predicted_by)
