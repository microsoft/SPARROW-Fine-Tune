"""VOC dataset audit, splitting, and YOLO dataset preparation.

Pipeline shape::

    audit_dataset()        -> AuditResult(images, boxes, issues)
    make_class_map()       -> {class_name: id}   (persisted, so IDs stay stable)
    split_images()         -> DataFrame(image, split)
    prepare_yolo_dataset() -> path to data.yaml

Two invariants worth knowing before editing:

1. ``boxes["img_width"]`` / ``["img_height"]`` are the *real* pixel dimensions
   read from the image file, not the values in the XML. VOC files routinely
   disagree with the image after a resize, and normalising against the XML value
   silently scales every box wrong. The XML values are kept alongside as
   ``xml_width``/``xml_height`` purely so the audit can report the mismatch.
2. Class names in ``boxes["class_name"]`` are already canonical (stripped, and
   case-folded if requested) by the time :func:`audit_dataset` returns. Any
   normalisation must happen there, so the class map and the label writer cannot
   disagree about a key.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageFile
from tqdm.auto import tqdm

from .xml_utils import Box, parse_voc_xml, to_yolo_line

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_SKIP_DIRS = {".git", ".ipynb_checkpoints", "__pycache__", ".cache"}

BOX_COLUMNS = [
    "image", "rel_path", "filename", "xml", "class_name",
    "img_width", "img_height", "xml_width", "xml_height",
    "xmin", "ymin", "xmax", "ymax", "box_w", "box_h", "difficult", "truncated",
]
IMAGE_COLUMNS = [
    "image", "rel_path", "filename", "flat_stem",
    "img_width", "img_height", "readable", "has_xml", "n_boxes", "blank_kind",
    # Populated only when audit_dataset(fingerprint=True). size/mtime are the
    # cheap change trigger; content_fp is the content-true answer.
    "size_bytes", "mtime_ns", "content_fp",
]

#: Values of the ``blank_kind`` column, in decreasing order of trust.
#:   labelled      - has at least one box
#:   empty_xml     - XML exists with no <object>: reviewed, confirmed empty
#:   confirmed_dir - no XML, but sits in a directory declared blank by the operator
#:   unknown       - no XML and no declaration: empty, or simply not tagged yet
BLANK_KINDS = ("labelled", "empty_xml", "confirmed_dir", "unknown")
TRUSTED_BLANK_KINDS = ("empty_xml", "confirmed_dir")


class _Progress:
    """Forwards iteration progress to a callback, throttled to ~1% steps.

    Long stages (a 60k-image audit runs for minutes) need determinate progress
    for any caller that isn't a terminal. Calling back per item would be tens of
    thousands of writes, so this coalesces.
    """

    def __init__(self, callback: Optional[Callable[[int, int, str], None]]):
        self._cb = callback

    def _emit(self, done: int, total: int, phase: str) -> None:
        try:
            self._cb(done, total, phase)  # type: ignore[misc]
        except Exception:  # noqa: BLE001 - reporting must never break the job
            pass

    def wrap(self, it: Iterable, total: int, phase: str) -> Iterator:
        if self._cb is None:
            yield from it
            return
        step = max(1, total // 100)
        done = 0
        self._emit(0, total, phase)
        for item in it:
            done += 1
            if done % step == 0 or done == total:
                self._emit(done, total, phase)
            yield item


@dataclass
class AuditResult:
    """Everything the audit learned. ``images`` includes files with no boxes."""

    images: pd.DataFrame
    boxes: pd.DataFrame
    issues: pd.DataFrame
    root: str = ""
    #: Human-readable advisories (class merges applied, excluded blanks, ...).
    #: Also printed, but returned so a UI can render them as actionable notes
    #: instead of losing them to stdout.
    warnings: List[str] = field(default_factory=list)

    @property
    def summary(self) -> Dict:
        return summarize(self)

    def issue_counts(self) -> pd.Series:
        if self.issues.empty:
            return pd.Series(dtype=int)
        return self.issues["type"].value_counts()

    def usable_images(self) -> pd.DataFrame:
        """Readable images only. Includes backgrounds (``n_boxes == 0``)."""
        if self.images.empty:
            return self.images
        return self.images[self.images["readable"]].copy()

    def background_images(self, policy: str = "reviewed") -> pd.DataFrame:
        """Readable images with no boxes, filtered by how trustworthy they are.

        Not all blank frames are equally trustworthy — see ``blank_kind``:

        ``empty_xml``
            An XML exists with no ``<object>`` entries. Someone looked at this
            frame and recorded that it is empty.

        ``confirmed_dir``
            No XML, but the operator declared the containing directory blank via
            ``confirmed_blank_dirs``. Use this when the tagging tool does not
            write a file for empty frames.

        ``unknown``
            No XML and no declaration. Ambiguous: either a genuine blank, or an
            image nobody has tagged yet. Training an untagged animal photo as a
            negative teaches the model that animals are background — worse than
            dropping the image entirely.

        Policies: ``"reviewed"`` (default: ``empty_xml`` + ``confirmed_dir``),
        ``"all"`` (also ``unknown`` — only correct if every image has been
        reviewed), ``"none"``.
        """
        blank = self.usable_images()
        if blank.empty:
            return blank
        blank = blank[blank["n_boxes"] == 0]
        if policy == "none":
            return blank.iloc[0:0]
        if policy == "reviewed":
            return blank[blank["blank_kind"].isin(TRUSTED_BLANK_KINDS)]
        if policy == "all":
            return blank
        raise ValueError(f"background_policy must be reviewed|all|none, got {policy!r}")


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------

def _walk(root: Path, suffixes: Optional[set] = None) -> List[Path]:
    out: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
        for fn in filenames:
            p = Path(dirpath) / fn
            if suffixes is None or p.suffix.lower() in suffixes:
                out.append(p)
    return sorted(out)


def find_images(root: str | Path) -> List[Path]:
    return _walk(Path(root), IMAGE_EXTS)


def find_xmls(root: str | Path) -> List[Path]:
    return _walk(Path(root), {".xml"})


def _under_any(rel_path: str, patterns: Sequence[str]) -> bool:
    """Whether ``rel_path`` sits inside any of ``patterns``.

    Each pattern is a directory relative to the dataset root (``"blanks"``,
    ``"2024/empty_frames"``), or a glob (``"*/blanks"``).
    """
    rel = rel_path.replace("\\", "/").lstrip("./")
    for raw in patterns:
        pattern = str(raw).replace("\\", "/").strip().strip("/")
        if not pattern:
            continue
        if rel == pattern or rel.startswith(pattern + "/"):
            return True
        # Glob form, matched against the directory part so "*/blanks" works.
        parent = rel.rsplit("/", 1)[0] if "/" in rel else ""
        if fnmatch(rel, pattern) or fnmatch(parent, pattern):
            return True
    return False


def flat_stem(path: Path, root: Path) -> str:
    """A collision-free, traceable stem for the flattened prepared dataset.

    ``root/site_a/cam3/0001.jpg`` -> ``site_a__cam3__0001``. Without this, two
    ``0001.jpg`` in different subfolders overwrite each other in
    ``prepared/<project>/images/<split>/``.
    """
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = Path(path.name)
    return rel.with_suffix("").as_posix().replace("/", "__")


# --------------------------------------------------------------------------
# audit
# --------------------------------------------------------------------------

#: End-of-file markers. A truncated file is missing its trailer.
_TRAILERS = {
    "JPEG": b"\xff\xd9",
    "MPO": b"\xff\xd9",
    "PNG": b"\x49\x45\x4e\x44\xae\x42\x60\x82",  # IEND chunk + CRC
    "GIF": b"\x3b",
}


#: Bytes read from each end of a file. Serves two purposes: the window searched
#: for the end-of-image marker, and the content fingerprint. Camera-trap firmware
#: pads files past the marker, so a small window is not safe — a 32-byte window
#: produced a 17% false-positive rate on real data.
_FP_CHUNK = 65536


def _trailer_in(ends: Optional[Tuple[int, bytes, bytes]], fmt: Optional[str]) -> bool:
    """Whether the file's end-of-image marker appears in its tail bytes.

    ``Image.verify()`` does *not* decode JPEG scan data, so a JPEG cut in half
    passes it. Checking for the trailer catches the interrupted-copy case cheaply.

    This is only a *pre-filter*: a missing marker means "suspicious", not
    "corrupt". Trailing padding after the marker is common and legitimate, so
    callers confirm with a real decode before rejecting the file.
    """
    marker = _TRAILERS.get((fmt or "").upper())
    if marker is None:
        return True  # unknown format: nothing to assert
    if ends is None:
        return False  # couldn't read it at all
    return marker in ends[2]


#: The fingerprint needs only enough leading bytes to be distinctive; the trailer
#: search is what needs the large window. Note PIL's verify() does not read the
#: whole JPEG (that is why truncation needed the trailer check), so these bytes
#: are a genuine extra read rather than a cache hit.
#:
#: Measured on the real 60,049-image dataset over a Windows mount:
#:   no fingerprint  ~50s (cold and warm)
#:   fingerprint    ~160s cold, ~70s warm
#: i.e. a one-off cold-read cost, then ~35% steady state. Acceptable for a
#: background job; Phase 2's incremental audit skips unchanged files entirely.
_FP_HEAD = 8192


def _read_ends(
    path: Path, head_chunk: int = _FP_HEAD, tail_chunk: int = _FP_CHUNK
) -> Optional[Tuple[int, bytes, bytes]]:
    """One open, returning ``(size, head, tail)``.

    Serves both the trailer check (needs the tail) and the content fingerprint
    (needs size + both ends) from a single open, instead of two.
    """
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(0)
            head = f.read(head_chunk)
            if size > tail_chunk:
                f.seek(size - tail_chunk)
                tail = f.read(tail_chunk)
            else:
                f.seek(0)
                tail = f.read(tail_chunk)
        return size, head, tail
    except OSError:
        return None


def _fingerprint_from_ends(ends: Tuple[int, bytes, bytes]) -> int:
    """Content-true fingerprint: ``blake2b(size ‖ head ‖ tail)``, 63-bit.

    Answers "did this image actually change?" without hashing 100-300 GB of
    JPEGs. Size plus both ends collides only for a file edited strictly in the
    middle with the byte count preserved, which does not happen to camera-trap
    images.

    Deliberately *not* mtime-based: re-copying a dataset rewrites every mtime and
    would otherwise report all 60,000 images as changed. mtime is only ever a
    cheap "something might have changed" trigger, never the answer.
    """
    size, head, tail = ends
    h = hashlib.blake2b(str(size).encode(), digest_size=8)
    h.update(head)
    h.update(tail)
    # 63-bit so it fits a signed SQLite INTEGER.
    return int.from_bytes(h.digest(), "big") >> 1


def _probe_image(
    path: Path, deep: bool = False, fingerprint: bool = False
) -> Tuple[int, int, Optional[str], Optional[int]]:
    """Return ``(width, height, error, content_fp)``.

    ``Image.open`` alone only parses the header, so a truncated JPEG — the classic
    interrupted-copy failure — used to pass the audit and then blow up during
    training. Three layers, cheapest first:

    1. header parse (catches garbage and wrong extensions)
    2. ``verify()`` (catches structural damage)
    3. trailer marker (catches truncation, which ``verify()`` misses for JPEG)

    A missing trailer is treated as *suspicious*, not corrupt: the file is then
    fully decoded to decide. Trailing padding after the end-of-image marker is
    legitimate and common in camera-trap JPEGs — condemning on the marker alone
    rejected 17% of a real dataset, every one of which decoded perfectly.

    ``deep=True`` decodes every image rather than only the suspicious ones —
    thorough but slower, and opt-in via ``deep_image_check`` in the config.
    """
    prev = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    fp: Optional[int] = None
    try:
        with Image.open(path) as im:
            fmt = im.format
            w, h = im.size
            im.verify()

        # Read both ends once, now that PIL has warmed the page cache. Serves the
        # trailer check and the fingerprint from the same bytes.
        ends = _read_ends(path) if (fingerprint or not deep) else None
        if ends is not None and fingerprint:
            fp = _fingerprint_from_ends(ends)

        if deep or not _trailer_in(ends, fmt):
            # verify() leaves the object unusable, so reopen to decode.
            with Image.open(path) as im:
                im.load()
        return int(w), int(h), None, fp
    except Exception as e:  # noqa: BLE001 - any failure means "do not train on this"
        return 0, 0, f"{type(e).__name__}: {e}", fp
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = prev


def _stat_fields(path: Path, content_fp: Optional[int]) -> Dict[str, Optional[int]]:
    """size/mtime/fingerprint for change detection. Never raises."""
    try:
        st = path.stat()
        return {"size_bytes": st.st_size, "mtime_ns": st.st_mtime_ns, "content_fp": content_fp}
    except OSError:
        return {"size_bytes": None, "mtime_ns": None, "content_fp": content_fp}


def _parse_one_xml(path: Path) -> Tuple[Path, Optional[List[Box]], Optional[str]]:
    try:
        return path, parse_voc_xml(path), None
    except Exception as e:  # noqa: BLE001
        return path, None, f"{type(e).__name__}: {e}"


def audit_dataset(
    root: str | Path,
    *,
    normalise_class_names: bool = True,
    deep_image_check: bool = False,
    size_mismatch_tolerance: int = 2,
    confirmed_blank_dirs: Sequence[str] = (),
    merge_classes: Optional[Dict[str, str]] = None,
    fingerprint: bool = False,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    workers: Optional[int] = None,
    progress: bool = True,
) -> AuditResult:
    """Scan a VOC dataset and return images, boxes, and issues.

    Args:
        normalise_class_names: strip and case-fold class names. Leaving this off
            trains ``Elephant`` and ``elephant`` as two separate classes.
        deep_image_check: additionally decode pixel data. Slower; use once when
            onboarding a new dataset.
        size_mismatch_tolerance: pixels of disagreement between the XML ``<size>``
            and the real image before an issue is raised.
        confirmed_blank_dirs: directories (relative to ``root``, globs allowed)
            whose XML-less images are known-empty frames rather than untagged
            ones. This records operator knowledge without writing anything to the
            dataset, which matters when it is mounted read-only.
        merge_classes: ``{from_name: to_name}`` applied after normalisation.
            Taxonomy typos are common in hand-tagged data and split one species
            across two labels, which guarantees the model confuses them. Merging
            here rather than editing the XML keeps the source dataset untouched.
        fingerprint: also compute a content fingerprint per image, so a later scan
            can tell which files genuinely changed. Nearly free here because the
            file is already being read; prohibitively expensive as a second pass.
        on_progress: called as ``(done, total, phase)`` roughly every 1% so a UI
            can show determinate progress. A 60k-image audit takes minutes.
        workers: thread count for I/O. Defaults to ``min(32, cpu*4)``.

    Warnings are returned on ``AuditResult.warnings`` as well as printed, so a
    caller without a terminal can surface them.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    workers = workers or min(32, (os.cpu_count() or 4) * 4)
    # Normalise the merge map the same way class names are, so the caller can
    # write it in whatever case the XML uses.
    merges: Dict[str, str] = {}
    for src_name, dst_name in (merge_classes or {}).items():
        key = str(src_name).strip()
        val = str(dst_name).strip()
        if normalise_class_names:
            key, val = key.lower(), val.lower()
        if key and val and key != val:
            merges[key] = val
    merge_counts: Dict[str, int] = defaultdict(int)

    images = find_images(root)
    xmls = find_xmls(root)
    issues: List[Dict] = []

    if not images:
        raise FileNotFoundError(f"No images found under {root} (looked for {sorted(IMAGE_EXTS)})")

    # ---- images ----------------------------------------------------------
    image_rows: List[Dict] = []
    by_dir_stem: Dict[Tuple[str, str], Path] = {}
    by_stem: Dict[str, List[Path]] = defaultdict(list)
    flat_seen: Dict[str, Path] = {}

    reporter = _Progress(on_progress)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        probes = list(
            tqdm(
                reporter.wrap(
                    pool.map(
                        lambda p: (p, _probe_image(p, deep_image_check, fingerprint)), images
                    ),
                    total=len(images),
                    phase="Checking images",
                ),
                total=len(images),
                desc="Checking images",
                disable=not progress,
            )
        )

    for img, (w, h, err, content_fp) in probes:
        rel = img.relative_to(root).as_posix() if img.is_relative_to(root) else img.name
        fs = flat_stem(img, root)
        if err:
            issues.append({"type": "corrupt_image", "path": str(img), "detail": err})
        if fs in flat_seen:
            issues.append({
                "type": "duplicate_flat_name",
                "path": str(img),
                "detail": f"collides with {flat_seen[fs]} after flattening to '{fs}'",
            })
        flat_seen.setdefault(fs, img)
        by_dir_stem.setdefault((str(img.parent), img.stem), img)
        by_stem[img.stem].append(img)
        image_rows.append({
            "image": str(img),
            "rel_path": rel,
            "filename": img.name,
            "flat_stem": fs,
            "img_width": w,
            "img_height": h,
            "readable": err is None and w > 0 and h > 0,
            "has_xml": False,
            "n_boxes": 0,
            "blank_kind": "confirmed_dir" if _under_any(rel, confirmed_blank_dirs) else "unknown",
            **(_stat_fields(img, content_fp) if fingerprint else {}),
        })

    images_df = pd.DataFrame(image_rows, columns=IMAGE_COLUMNS)
    size_by_path = dict(zip(images_df["image"], zip(images_df["img_width"], images_df["img_height"])))

    # ---- xml -------------------------------------------------------------
    with ThreadPoolExecutor(max_workers=workers) as pool:
        parsed = list(
            tqdm(
                reporter.wrap(pool.map(_parse_one_xml, xmls), total=len(xmls), phase="Reading XML"),
                total=len(xmls),
                desc="Reading XML",
                disable=not progress,
            )
        )

    box_rows: List[Dict] = []
    matched_images: set = set()

    for xml, boxes, err in parsed:
        if err is not None:
            issues.append({"type": "corrupt_xml", "path": str(xml), "detail": err})
            continue

        # Match XML to image by sibling stem first, then globally by stem. The
        # <filename> element is only a hint: it is frequently a stale absolute
        # path from the tagging machine.
        img_path = by_dir_stem.get((str(xml.parent), xml.stem))
        if img_path is None and boxes:
            hint = Path(boxes[0].image).stem
            img_path = by_dir_stem.get((str(xml.parent), hint))
        if img_path is None:
            candidates = by_stem.get(xml.stem) or []
            if len(candidates) == 1:
                img_path = candidates[0]
            elif len(candidates) > 1:
                issues.append({
                    "type": "ambiguous_image_match",
                    "path": str(xml),
                    "detail": f"{len(candidates)} images share stem '{xml.stem}'; using {candidates[0]}",
                })
                img_path = candidates[0]

        if img_path is None:
            issues.append({
                "type": "missing_image",
                "path": str(xml),
                "detail": f"No image found for stem '{xml.stem}'",
            })
            continue

        matched_images.add(str(img_path))

        if not boxes:
            issues.append({"type": "empty_xml", "path": str(xml), "detail": "No <object> entries"})
            continue

        real_w, real_h = size_by_path.get(str(img_path), (0, 0))
        xml_w, xml_h = boxes[0].width, boxes[0].height
        if real_w and real_h and xml_w and xml_h:
            if abs(real_w - xml_w) > size_mismatch_tolerance or abs(real_h - xml_h) > size_mismatch_tolerance:
                issues.append({
                    "type": "size_mismatch",
                    "path": str(img_path),
                    "detail": f"XML says {xml_w}x{xml_h}, image is {real_w}x{real_h}; using the image",
                })
        # Fall back to the XML dimensions only when the image is unreadable.
        use_w = real_w or xml_w
        use_h = real_h or xml_h

        for b in boxes:
            name = b.class_name.strip()
            if normalise_class_names:
                name = name.lower()
            if name in merges:
                merge_counts[f"{name} -> {merges[name]}"] += 1
                name = merges[name]
            if not name:
                issues.append({"type": "empty_class_name", "path": str(xml), "detail": "<name> was blank"})
                continue
            if not b.is_valid(use_w, use_h):
                issues.append({
                    "type": "invalid_box",
                    "path": str(xml),
                    "detail": f"{name} [{b.xmin},{b.ymin},{b.xmax},{b.ymax}] in {use_w}x{use_h}",
                })
                continue
            c = b.clipped(use_w, use_h)
            box_rows.append({
                "image": str(img_path),
                "rel_path": img_path.relative_to(root).as_posix() if img_path.is_relative_to(root) else img_path.name,
                "filename": img_path.name,
                "xml": str(xml),
                "class_name": name,
                "img_width": use_w,
                "img_height": use_h,
                "xml_width": xml_w,
                "xml_height": xml_h,
                "xmin": c.xmin, "ymin": c.ymin, "xmax": c.xmax, "ymax": c.ymax,
                "box_w": c.xmax - c.xmin,
                "box_h": c.ymax - c.ymin,
                "difficult": b.difficult,
                "truncated": b.truncated,
            })

    boxes_df = pd.DataFrame(box_rows, columns=BOX_COLUMNS)

    # ---- join back -------------------------------------------------------
    if not images_df.empty:
        images_df["has_xml"] = images_df["image"].isin(matched_images)
        if not boxes_df.empty:
            counts = boxes_df.groupby("image").size()
            images_df["n_boxes"] = images_df["image"].map(counts).fillna(0).astype(int)

        labelled = images_df["n_boxes"] > 0
        images_df.loc[labelled, "blank_kind"] = "labelled"
        images_df.loc[~labelled & images_df["has_xml"], "blank_kind"] = "empty_xml"

        for row in images_df.loc[~images_df["has_xml"]].itertuples():
            if row.blank_kind == "confirmed_dir":
                issues.append({
                    "type": "confirmed_blank",
                    "path": str(row.image),
                    "detail": "No XML, but declared blank via confirmed_blank_dirs",
                })
            else:
                issues.append({
                    "type": "missing_xml",
                    "path": str(row.image),
                    "detail": "No matching XML; either an untagged image or an undeclared blank",
                })

    warnings: List[str] = []
    for mapping, count in sorted(merge_counts.items()):
        warnings.append(f"Merged {count} box(es): {mapping}")
    unused = sorted(set(merges) - {m.split(" -> ")[0] for m in merge_counts})
    if unused:
        warnings.append(f"merge_classes entries matched nothing: {unused}")
    for w in warnings:
        print(f"NOTE: {w}")

    issues_df = pd.DataFrame(issues, columns=["type", "path", "detail"])
    return AuditResult(
        images=images_df, boxes=boxes_df, issues=issues_df, root=str(root), warnings=warnings
    )


def summarize(result: AuditResult) -> Dict:
    images, boxes = result.images, result.boxes
    if images.empty:
        readable = labelled = 0
        kinds = {k: 0 for k in BLANK_KINDS}
    else:
        ok = images["readable"]
        readable = int(ok.sum())
        labelled = int((ok & (images["n_boxes"] > 0)).sum())
        blank = ok & (images["n_boxes"] == 0)
        # Blanks are graded by trust, not lumped together. See
        # AuditResult.background_images.
        counts = images.loc[blank, "blank_kind"].value_counts()
        kinds = {k: int(counts.get(k, 0)) for k in BLANK_KINDS}
    return {
        "num_images": int(len(images)),
        "readable_images": readable,
        "unreadable_images": int(len(images)) - readable,
        "labelled_images": labelled,
        "background_images": sum(kinds[k] for k in BLANK_KINDS if k != "labelled"),
        "blank_with_empty_xml": kinds["empty_xml"],
        "blank_confirmed_dir": kinds["confirmed_dir"],
        "blank_unknown": kinds["unknown"],
        "num_boxes": int(len(boxes)),
        "num_classes": int(boxes["class_name"].nunique()) if not boxes.empty else 0,
        "classes": sorted(boxes["class_name"].unique().tolist()) if not boxes.empty else [],
        "num_issues": int(len(result.issues)),
    }


# --------------------------------------------------------------------------
# class map
# --------------------------------------------------------------------------

def make_class_map(
    boxes: pd.DataFrame,
    *,
    drop_classes: Sequence[str] = (),
    min_boxes: int = 0,
) -> Dict[str, int]:
    """Build ``{class_name: id}`` from the classes actually present in ``boxes``.

    Names are taken verbatim from the frame, so this cannot drift from what
    :func:`prepare_yolo_dataset` looks up — normalisation belongs in
    :func:`audit_dataset`.

    IDs are assigned in sorted order, which means adding a class renumbers the
    ones after it. Use :func:`load_or_create_class_map` to pin IDs across
    re-preparations so old checkpoints stay meaningful.
    """
    if boxes.empty:
        return {}
    counts = boxes["class_name"].value_counts()
    drop = {str(d).strip() for d in drop_classes}
    keep = [c for c, n in counts.items() if c not in drop and n >= min_boxes]
    return {name: i for i, name in enumerate(sorted(keep))}


def load_or_create_class_map(
    path: str | Path,
    boxes: pd.DataFrame,
    *,
    drop_classes: Sequence[str] = (),
    min_boxes: int = 0,
    reuse: bool = True,
) -> Tuple[Dict[str, int], List[str]]:
    """Load a pinned class map, extending it with any genuinely new classes.

    Returns ``(class_map, warnings)``. Existing IDs are never reassigned, so a
    model trained against an earlier version of the dataset still lines up.
    """
    path = Path(path)
    warnings: List[str] = []
    fresh = make_class_map(boxes, drop_classes=drop_classes, min_boxes=min_boxes)

    if not (reuse and path.is_file()):
        return fresh, warnings

    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
        existing = {str(k): int(v) for k, v in existing.items()}
    except Exception as e:  # noqa: BLE001
        warnings.append(f"Could not read pinned class map {path} ({e}); rebuilding from scratch")
        return fresh, warnings

    merged = dict(existing)
    next_id = max(merged.values()) + 1 if merged else 0
    for name in sorted(fresh):
        if name not in merged:
            merged[name] = next_id
            warnings.append(f"New class '{name}' appended as id {next_id} (existing ids unchanged)")
            next_id += 1
    for name in sorted(set(merged) - set(fresh)):
        warnings.append(f"Class '{name}' is in the pinned map but has no boxes in this dataset")
    return merged, warnings


# --------------------------------------------------------------------------
# split
# --------------------------------------------------------------------------

def load_pinned_splits(path: str | Path) -> Dict[str, str]:
    """Read ``splits.json`` from a previous preparation. Missing file -> ``{}``."""
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"WARNING: could not read pinned splits from {path} ({e}); splitting afresh")
        return {}
    valid = {"train", "val", "test"}
    return {str(k): str(v) for k, v in data.items() if str(v) in valid}


def _stratum(group: pd.Series, mode: str, class_freq: pd.Series) -> str:
    if mode == "dominant":
        return str(group.value_counts().idxmax())
    # "rarest": label each image by its least common class globally, so rare
    # classes get spread across splits instead of landing entirely in one.
    present = group.unique().tolist()
    return str(min(present, key=lambda c: (class_freq.get(c, 0), c)))


def _allocate(rng: "np.random.Generator", n: int, fracs: Sequence[float]) -> List[int]:
    """Split ``n`` items into buckets sized by ``fracs``, summing exactly to ``n``.

    The fractional remainder is assigned by weighted random draw rather than
    always to the largest remainder. Deterministic rounding sends every small
    stratum to whichever bucket has the biggest share — with an 80/10/10 split
    that means a 3-image class lands entirely in train, every time. Randomised
    rounding is unbiased in expectation, so rare classes spread across splits.
    """
    raw = np.asarray(fracs, dtype=float) * n
    base = np.floor(raw).astype(int)
    remainder = int(n - base.sum())
    weights = raw - base
    for _ in range(max(0, remainder)):
        total = weights.sum()
        if total <= 0:
            # All remainders exhausted; fall back to the largest target.
            idx = int(np.argmax(fracs))
        else:
            idx = int(rng.choice(len(fracs), p=weights / total))
            weights[idx] = 0.0
        base[idx] += 1
    return [int(x) for x in base]


def _stratified_assign(
    rng: "np.random.Generator",
    frame: pd.DataFrame,
    fracs: Sequence[float],
    names: Sequence[str],
) -> pd.Series:
    """Assign each row a split, proportionally within every stratum."""
    out = pd.Series(index=frame.index, dtype=object)
    for _, block in frame.groupby("stratum", sort=True):
        idx = block.index.to_numpy(copy=True)  # pandas indexes are read-only
        rng.shuffle(idx)
        counts = _allocate(rng, len(idx), fracs)
        pos = 0
        for name, count in zip(names, counts):
            out.loc[idx[pos: pos + count]] = name
            pos += count
    return out


def _assign_new_only(
    rng: "np.random.Generator",
    df: pd.DataFrame,
    fracs: Sequence[float],
    names: Sequence[str],
    pinned: Dict[str, str],
    assign,
) -> pd.Series:
    """Keep pinned images in their existing split; place only the new ones.

    Without this, growing a dataset reshuffles everything: images that were in
    the test set of the previous model land in the training set of the next, so
    the two models' test scores are not comparable and the newer one is
    inflated. New images are allocated to whichever splits are furthest below
    their target, so the overall ratio converges back to ``fracs``.
    """
    out = pd.Series(df["rel_path"].map(pinned), index=df.index, dtype=object)
    fresh = df[out.isna()]
    if fresh.empty:
        return out

    total = len(df)
    targets = np.asarray(fracs, dtype=float) * total
    current = np.array([float((out == n).sum()) for n in names])
    deficit = np.maximum(0.0, targets - current)
    # Every split already at or over target: fall back to the plain ratio.
    weights = deficit if deficit.sum() > 0 else np.asarray(fracs, dtype=float)
    weights = weights / weights.sum()

    out.loc[fresh.index] = assign(rng, fresh, weights, names)
    return out


def _grouped_assign(
    rng: "np.random.Generator",
    frame: pd.DataFrame,
    fracs: Sequence[float],
    names: Sequence[str],
) -> pd.Series:
    """Assign whole groups, keeping split sizes close to the requested ratio.

    Groups are shuffled, then each is placed in whichever split is furthest below
    its target item count. Groups vary in size, so proportional allocation has to
    be done by item count rather than group count.
    """
    out = pd.Series(index=frame.index, dtype=object)
    groups = [(key, block.index.to_numpy()) for key, block in frame.groupby("group", sort=True)]
    order = rng.permutation(len(groups))

    total = len(frame)
    targets = np.asarray(fracs, dtype=float) * total
    current = np.zeros(len(fracs), dtype=float)

    for gi in order:
        _, idx = groups[gi]
        deficit = targets - current
        deficit[targets <= 0] = -np.inf  # never fill a split that was asked for 0
        choice = int(np.argmax(deficit))
        out.loc[idx] = names[choice]
        current[choice] += len(idx)
    return out


def split_images(
    images: pd.DataFrame,
    boxes: pd.DataFrame,
    *,
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    seed: int = 42,
    stratify: str = "rarest",
    group_by: Optional[str] = None,
    background_policy: str = "reviewed",
    pinned: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Assign every usable image to train/val/test.

    Args:
        stratify: ``"rarest"`` (default, best rare-class coverage), ``"dominant"``,
            or ``"none"``.
        group_by: keep images sharing a key in the same split. Pass
            ``"parent_dir"`` for camera-trap data, where consecutive frames in a
            burst are near-duplicates — splitting them across train and val
            inflates validation metrics badly.
        background_policy: which blank frames to include as negatives —
            ``"reviewed"`` (default), ``"all"``, or ``"none"``. See
            :meth:`AuditResult.background_images`; the distinction between a
            reviewed-empty frame and an untagged one matters.
        pinned: ``{rel_path: split}`` from a previous preparation. Those images
            keep their split and only new ones are placed. Essential when growing
            a dataset — otherwise the reshuffle moves old test images into train
            and the new model's test score is no longer comparable to the old
            one's. Written to ``prepared/<project>/splits.json``.

    This function decides membership for the whole dataset:
    :func:`prepare_yolo_dataset` writes exactly the images named here, so the
    background policy only needs applying once.

    Deterministic for a given seed. Implemented directly rather than via
    scikit-learn's ``train_test_split``: that raises on any class with fewer
    members than there are splits, which forced the previous version to wrap it
    in a bare ``except`` that silently degraded to an unstratified split.
    """
    total = train + val + test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1.0, got {total} ({train}/{val}/{test})")
    if min(train, val, test) < 0:
        raise ValueError("Split fractions must be non-negative")
    if background_policy not in {"reviewed", "all", "none"}:
        raise ValueError(f"background_policy must be reviewed|all|none, got {background_policy!r}")

    usable = images[images["readable"]] if not images.empty else images
    if usable.empty:
        raise ValueError("No usable images to split (every image failed the readability check)")

    labelled = usable[usable["n_boxes"] > 0]
    blank = usable[usable["n_boxes"] == 0]
    if background_policy == "none":
        blank = blank.iloc[0:0]
    elif background_policy == "reviewed":
        dropped = int((~blank["blank_kind"].isin(TRUSTED_BLANK_KINDS)).sum())
        blank = blank[blank["blank_kind"].isin(TRUSTED_BLANK_KINDS)]
        if dropped:
            print(
                f"NOTE: excluded {dropped} blank image(s) with no XML and no blank-directory "
                f"declaration. These are either genuine empties or simply untagged, and the "
                f"difference is not recoverable from the files. Either list their folder in "
                f"dataset.confirmed_blank_dirs, or set background_policy='all' if every image "
                f"in this dataset has been reviewed."
            )

    usable = pd.concat([labelled, blank])
    if usable.empty:
        raise ValueError(
            "No images left to split: none have boxes and background_policy excluded the rest"
        )

    df = usable[["image", "rel_path", "n_boxes"]].copy()

    # stratum label per image
    if stratify not in {"rarest", "dominant", "none"}:
        raise ValueError(f"stratify must be 'rarest', 'dominant', or 'none'; got {stratify!r}")
    if stratify == "none" or boxes.empty:
        df["stratum"] = "__all__"
    else:
        class_freq = boxes["class_name"].value_counts()
        per_image = boxes.groupby("image")["class_name"].apply(lambda g: _stratum(g, stratify, class_freq))
        df["stratum"] = df["image"].map(per_image).fillna("__background__")

    if group_by == "parent_dir":
        df["group"] = df["rel_path"].map(lambda p: str(Path(p).parent))
    elif group_by in (None, "none", ""):
        df["group"] = df["image"]
    else:
        if group_by not in df.columns:
            raise ValueError(f"group_by={group_by!r} is not a column of the image frame")
        df["group"] = df[group_by]

    rng = np.random.default_rng(seed)
    names = ("train", "val", "test")
    fracs = (train, val, test)

    if pinned:
        known = set(pinned)
        carried = int(df["rel_path"].isin(known).sum())
        assign = _grouped_assign if df["group"].nunique() < len(df) else _stratified_assign
        df["split"] = _assign_new_only(rng, df, fracs, names, dict(pinned), assign)
        print(
            f"Reusing pinned splits: {carried} image(s) kept their existing split, "
            f"{len(df) - carried} new image(s) assigned."
        )
        stale = len(known) - carried
        if stale:
            print(f"  ({stale} pinned image(s) are no longer in the dataset and were ignored)")
        return df[["image", "split"]].reset_index(drop=True)

    n_groups = df["group"].nunique()
    if n_groups < len(df):
        # Grouping and stratification conflict; grouping wins, because leaking a
        # burst across splits corrupts the metric outright while imperfect class
        # balance only makes it noisier.
        wanted = sum(1 for f in fracs if f > 0)
        # A group is indivisible, so one oversized group lands wholly in a single
        # split. A folder of collected blank frames is the usual culprit: it is
        # not a burst sequence, so grouping it gains nothing and costs a lot.
        sizes = df["group"].value_counts()
        smallest_target = min(f for f in fracs if f > 0) * len(df)
        for name, size in sizes[sizes > smallest_target].items():
            print(
                f"WARNING: group {name!r} holds {size} of {len(df)} images, more than the "
                f"smallest split target ({smallest_target:.0f}). It will land entirely in one "
                f"split. If that folder is not a burst sequence (a blanks folder, for example), "
                f"it should not be grouped."
            )
        if n_groups < wanted * 3:
            # Whole groups are indivisible, so with few groups the requested
            # ratios cannot be met and a split can end up empty.
            print(
                f"WARNING: group_by={group_by!r} produced only {n_groups} group(s) for "
                f"{wanted} splits. Split sizes will not match {fracs} and a split may be "
                f"empty. Use a finer grouping key, or set group_by=none."
            )
        df["split"] = _grouped_assign(rng, df, fracs, names)
    else:
        df["split"] = _stratified_assign(rng, df, fracs, names)

    empty = [n for n, f in zip(names, fracs) if f > 0 and (df["split"] == n).sum() == 0]
    if empty:
        print(f"WARNING: split(s) {empty} came out empty despite a non-zero fraction")

    return df[["image", "split"]].reset_index(drop=True)


# --------------------------------------------------------------------------
# prepare
# --------------------------------------------------------------------------

#: Ultralytics' loader checks whether a JPEG's *last two bytes* are the
#: end-of-image marker and, if not, rewrites the file in place to "repair" it.
#: Camera-trap firmware routinely pads past the marker, so a large share of a
#: real dataset trips that check. On a read-only source that rewrite fails with
#: EROFS and the image is dropped from training entirely — silently, and at a
#: scale that matters: 9,349 of 47,829 training images on this project's dataset.
#:
#: The repair below is byte-exact rather than a re-encode. The pixel data is
#: already complete and valid; only the padding after the marker offends. So we
#: copy the file truncated at the marker, which is lossless, needs no decode, and
#: leaves the source untouched.
_JPEG_SUFFIXES = {".jpg", ".jpeg", ".jpe"}
#: An EXIF segment is length-prefixed with 16 bits, so any thumbnail it carries —
#: and that thumbnail's own end-of-image marker — lies within the first 64 KiB.
_EXIF_MAX = 65536
_JPEG_EOI = b"\xff\xd9"


def needs_trailer_repair(path: Path) -> bool:
    """Whether Ultralytics would try to rewrite this file. Reads two bytes."""
    if path.suffix.lower() not in _JPEG_SUFFIXES:
        return False
    try:
        with open(path, "rb") as fh:
            fh.seek(-2, os.SEEK_END)
            return fh.read(2) != _JPEG_EOI
    except OSError:
        return False   # unreadable is the audit's problem, not this function's


def copy_truncated_at_eoi(src: Path, dst: Path) -> bool:
    """Copy ``src`` to ``dst`` with everything after the final EOI removed.

    Returns False when the marker cannot be trusted, in which case the caller
    links the file unchanged and the audit reports it.

    The whole file is searched, not a tail window: measured on this project's
    data, the trailing block runs to ~71 KB, so a 64 KB window missed 2,702
    images and quietly fell back to a link — which is the failure this function
    exists to prevent. The search is free because the bytes are already in hand,
    and ``FF D9`` cannot occur inside entropy-coded scan data (any ``FF`` there
    is byte-stuffed with a following ``00``), so a match is always a real marker.
    """
    try:
        data = src.read_bytes()
    except OSError:
        return False
    cut = data.rfind(_JPEG_EOI)
    if cut == -1:
        return False
    end = cut + 2

    # One marker cannot be trusted on position alone: an EXIF thumbnail is
    # itself a JPEG and carries its own EOI, always inside the first 64 KiB. If
    # that is the last marker in the file then the main image has none — the
    # file is genuinely truncated — and cutting there would yield a small,
    # entirely plausible, wrong image. Rare enough to settle by decoding.
    if end <= _EXIF_MAX and end < len(data) // 2:
        prev = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = False   # or every truncation "decodes"
        try:
            with Image.open(io.BytesIO(data)) as whole:
                expected = whole.size        # from the main image's SOF header
            with Image.open(io.BytesIO(data[:end])) as im:
                im.load()
                if im.size != expected:
                    return False             # we cut back to the thumbnail
        except Exception:  # noqa: BLE001 - any failure means do not cut here
            return False
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = prev

    dst.write_bytes(data[:end])
    shutil.copystat(src, dst, follow_symlinks=True)
    return True


def _materialise(src: Path, dst: Path, mode: str) -> str:
    """Place ``src`` at ``dst`` as cheaply as the filesystem allows.

    Hardlink -> symlink -> copy. Copying a large camera-trap dataset doubles disk
    usage and takes minutes; a hardlink is instant and free. Cross-device or
    read-only mounts fall back automatically.
    """
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    order = {
        "auto": ("hardlink", "symlink", "copy"),
        "hardlink": ("hardlink", "copy"),
        "symlink": ("symlink", "copy"),
        "copy": ("copy",),
    }[mode]
    last: Optional[Exception] = None
    for attempt in order:
        try:
            if attempt == "hardlink":
                os.link(src, dst)
            elif attempt == "symlink":
                os.symlink(src, dst)
            else:
                shutil.copy2(src, dst)
            return attempt
        except (OSError, NotImplementedError) as e:
            last = e
    raise OSError(f"Could not place {src} at {dst}: {last}")


def prepare_yolo_dataset(
    audit: AuditResult,
    out_root: str | Path,
    project_name: str,
    class_map: Dict[str, int],
    split_df: pd.DataFrame,
    *,
    overwrite: bool = False,
    link_mode: str = "auto",
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    workers: Optional[int] = None,
    progress: bool = True,
) -> Path:
    """Write a YOLO dataset and return the path to its ``data.yaml``.

    Membership comes entirely from ``split_df`` — whatever :func:`split_images`
    selected, including its background policy. There is deliberately no second
    background filter here: two filters could disagree.

    Images with no boxes get an empty ``.txt``. YOLO reads that as a negative,
    which materially reduces false positives on camera-trap data; dropping such
    images — the old behaviour — threw away the most useful part of the dataset.

    Args:
        overwrite: required to replace an existing prepared dataset. Guarded
            because this deletes the whole directory, and the previous tree may be
            what an in-flight training run is reading from.
        link_mode: ``auto`` | ``hardlink`` | ``symlink`` | ``copy``.
    """
    out_root = Path(out_root) / project_name
    if out_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"{out_root} already exists. Pass overwrite=True to replace it "
                f"(this deletes the directory)."
            )
        shutil.rmtree(out_root)

    for split in ("train", "val", "test"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    split_map = dict(zip(split_df["image"], split_df["split"]))
    images = audit.usable_images()
    images = images[images["image"].isin(split_map)]
    if images.empty:
        raise ValueError("No images left to prepare after applying the split")

    boxes_by_image: Dict[str, pd.DataFrame] = (
        {k: v for k, v in audit.boxes.groupby("image")} if not audit.boxes.empty else {}
    )

    unknown_classes: set = set()
    dropped_boxes = 0
    counts = {"train": 0, "val": 0, "test": 0}
    label_counts = {"train": 0, "val": 0, "test": 0}
    modes_used: Dict[str, int] = defaultdict(int)

    records = list(images[["image", "flat_stem"]].itertuples(index=False, name=None))

    def _one(rec: Tuple[str, str]) -> Tuple[str, str, int]:
        image, stem = rec
        src = Path(image)
        split = split_map[image]
        dst_img = out_root / "images" / split / f"{stem}{src.suffix}"
        # A padded JPEG must become a real file, not a link: Ultralytics will
        # otherwise try to rewrite it through the link and drop it when the
        # source mount refuses. See needs_trailer_repair.
        if needs_trailer_repair(src):
            if dst_img.exists() or dst_img.is_symlink():
                dst_img.unlink()
            mode_used = ("repaired" if copy_truncated_at_eoi(src, dst_img)
                         else _materialise(src, dst_img, link_mode))
        else:
            mode_used = _materialise(src, dst_img, link_mode)

        lines: List[str] = []
        local_dropped = 0
        group = boxes_by_image.get(image)
        if group is not None:
            for r in group.itertuples(index=False):
                cid = class_map.get(r.class_name)
                if cid is None:
                    unknown_classes.add(r.class_name)
                    continue
                line = to_yolo_line(cid, r.xmin, r.ymin, r.xmax, r.ymax, r.img_width, r.img_height)
                if line is None:
                    local_dropped += 1
                    continue
                lines.append(line)
        (out_root / "labels" / split / f"{stem}.txt").write_text(
            ("\n".join(lines) + "\n") if lines else "", encoding="utf-8"
        )
        return split, mode_used, local_dropped

    workers = workers or min(16, (os.cpu_count() or 4) * 2)
    reporter = _Progress(on_progress)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for split, mode_used, local_dropped in tqdm(
            reporter.wrap(
                pool.map(_one, records), total=len(records), phase="Writing dataset"
            ),
            total=len(records),
            desc=f"Writing YOLO dataset ({link_mode})",
            disable=not progress,
        ):
            counts[split] += 1
            modes_used[mode_used] += 1
            dropped_boxes += local_dropped

    # count labelled images per split for the manifest
    if not audit.boxes.empty:
        labelled = set(audit.boxes["image"].unique())
        for image in images["image"]:
            if image in labelled:
                label_counts[split_map[image]] += 1

    names = {int(i): c for c, i in sorted(class_map.items(), key=lambda kv: kv[1])}
    data = {
        "path": str(out_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
    }
    (out_root / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    (out_root / "classes.json").write_text(json.dumps(class_map, indent=2, sort_keys=True), encoding="utf-8")

    # Persist split membership keyed on the path relative to the dataset root, so
    # a later preparation of a grown dataset can keep every existing image where
    # it was. Absolute paths would break if the dataset moved.
    rel_by_image = dict(zip(images["image"], images["rel_path"]))
    pinned_out = {
        rel_by_image[img]: split_map[img]
        for img in images["image"]
        if img in rel_by_image
    }
    (out_root / "splits.json").write_text(
        json.dumps(pinned_out, indent=2, sort_keys=True), encoding="utf-8"
    )

    manifest = {
        "project_name": project_name,
        "source_root": audit.root,
        "images_per_split": counts,
        "labelled_images_per_split": label_counts,
        "background_images": int(sum(counts.values()) - sum(label_counts.values())),
        "num_classes": len(class_map),
        "classes": names,
        "link_mode_requested": link_mode,
        "link_mode_used": dict(modes_used),
        "dropped_degenerate_boxes": dropped_boxes,
        "classes_not_in_map": sorted(unknown_classes),
        "audit_summary": audit.summary,
    }
    (out_root / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Falling back from symlink/hardlink to copy silently duplicates the entire
    # dataset. On a Windows filesystem (drvfs) symlink creation usually needs
    # Developer Mode, so this fallback is easy to hit and expensive to discover
    # via `df` three hours later.
    if link_mode != "copy" and modes_used.get("copy"):
        copied = modes_used["copy"]
        print(
            f"WARNING: link_mode={link_mode!r} was requested but {copied} of {sum(modes_used.values())} "
            f"image(s) had to be COPIED - the prepared dataset now duplicates that much storage.\n"
            f"         Usual cause: the prepared directory is on a Windows drive (/mnt/c, /mnt/d), "
            f"where hardlinks across filesystems fail and symlinks need Developer Mode.\n"
            f"         Put prepared_root on the Linux filesystem to link instead of copy."
        )

    # Not a warning: this is the fix working, and it is worth stating because
    # without it every one of these images would have been dropped by the
    # trainer rather than trained on.
    if modes_used.get("repaired"):
        n = modes_used["repaired"]
        print(
            f"NOTE: {n} of {sum(modes_used.values())} image(s) were copied with trailing "
            f"padding removed. Their JPEG data ends before the end of the file, which the "
            f"trainer treats as damage and tries to rewrite in place; on a read-only source "
            f"that fails and the image is silently excluded from training.\n"
            f"      The originals are untouched - only the prepared copy is trimmed."
        )

    if unknown_classes:
        print(f"WARNING: {len(unknown_classes)} class(es) had no id and were skipped: {sorted(unknown_classes)}")
    if dropped_boxes:
        print(f"WARNING: dropped {dropped_boxes} degenerate box(es) during conversion")
    if counts["val"] == 0:
        print("WARNING: the val split is empty; Ultralytics needs a non-empty val set")

    return out_root / "data.yaml"


def verify_prepared(data_yaml: str | Path) -> Dict:
    """Post-prepare sanity check: splits populated, label ids in range.

    Cheap insurance against starting a multi-hour run on a broken dataset.
    """
    data_yaml = Path(data_yaml)
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(data["path"])
    n_classes = len(data["names"])
    report: Dict = {"data_yaml": str(data_yaml), "num_classes": n_classes, "splits": {}, "problems": []}

    for split in ("train", "val", "test"):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        imgs = [p for p in img_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS] if img_dir.is_dir() else []
        labels = list(lbl_dir.glob("*.txt")) if lbl_dir.is_dir() else []
        boxes = 0
        bad_id = 0
        oob = 0
        for lp in labels:
            for line in lp.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                boxes += 1
                cid = int(parts[0])
                vals = [float(v) for v in parts[1:]]
                if not (0 <= cid < n_classes):
                    bad_id += 1
                if any(v < 0 or v > 1 for v in vals):
                    oob += 1
        report["splits"][split] = {
            "images": len(imgs),
            "labels": len(labels),
            "boxes": boxes,
            "backgrounds": sum(1 for lp in labels if lp.stat().st_size == 0),
        }
        missing = len(imgs) - len(labels)
        if missing:
            report["problems"].append(f"{split}: {missing} image(s) without a label file")
        if bad_id:
            report["problems"].append(f"{split}: {bad_id} label(s) with a class id outside 0..{n_classes - 1}")
        if oob:
            report["problems"].append(f"{split}: {oob} label(s) with coordinates outside [0,1]")

    if report["splits"]["train"]["images"] == 0:
        report["problems"].append("train split is empty")
    if report["splits"]["val"]["images"] == 0:
        report["problems"].append("val split is empty; Ultralytics requires a val set")

    report["ok"] = not report["problems"]
    return report
