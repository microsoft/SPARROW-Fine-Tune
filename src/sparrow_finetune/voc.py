"""Pascal VOC XML for detection boxes: one ``<annotation>`` file per image.

Writing follows the workbench writer (``sparrow_yolo/xml_utils.py`` ``write_voc_xml``): integer pixel
corners clamped to the image, swapped corners put in order, degenerate boxes and empty names dropped,
``verified="yes"`` on reviewed images, and an object-free file for a reviewed blank frame.

Reading accepts untrusted uploads. ``defusedxml`` is not a dependency, so any document with a DOCTYPE
(and therefore any ENTITY declaration) is refused before the stdlib parser sees it; zip archives are
bounded in member count, per-member size and total size, measured on the bytes actually read rather
than the sizes the archive declares.
"""

from __future__ import annotations

import io
import math
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Iterable, Optional, Sequence
from xml.dom import minidom

MAX_ZIP_BYTES = 64 * 1024 * 1024
MAX_MEMBERS = 20_000
MAX_MEMBER_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024

_DTD = re.compile(rb"<!\s*(DOCTYPE|ENTITY)", re.IGNORECASE)


class VocError(ValueError):
    """The upload as a whole cannot be read (the message says why)."""


# --------------------------------------------------------------------------- writing


def voc_corners(
    bbox: Sequence[float], width: int, height: int
) -> Optional[tuple[int, int, int, int]]:
    """``[x, y, w, h]`` pixels -> clamped integer ``(xmin, ymin, xmax, ymax)``; ``None`` if degenerate."""
    x, y, w, h = (float(v) for v in bbox)
    x0, x1 = sorted((x, x + w))
    y0, y1 = sorted((y, y + h))
    x0 = int(round(max(0.0, min(width, x0))))
    x1 = int(round(max(0.0, min(width, x1))))
    y0 = int(round(max(0.0, min(height, y0))))
    y1 = int(round(max(0.0, min(height, y1))))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def voc_xml(
    image_name: str,
    width: int,
    height: int,
    annotations: Iterable[dict],
    *,
    reviewed: bool = False,
    database: str = "sparrow_finetune",
) -> bytes:
    """One VOC document for ``annotations`` (``{"label", "bbox": [x, y, w, h]}`` in pixels)."""
    width, height = int(width), int(height)
    ann = ET.Element("annotation")
    if reviewed:
        ann.set("verified", "yes")
    ET.SubElement(ann, "folder").text = "images"
    ET.SubElement(ann, "filename").text = image_name
    ET.SubElement(ann, "path").text = image_name
    source = ET.SubElement(ann, "source")
    ET.SubElement(source, "database").text = database
    size = ET.SubElement(ann, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    ET.SubElement(ann, "segmented").text = "0"
    for a in annotations:
        name = str(a.get("label") or "").strip()
        if not name or a.get("bbox") is None:
            continue
        corners = voc_corners(a["bbox"], width, height)
        if corners is None:
            continue
        obj = ET.SubElement(ann, "object")
        ET.SubElement(obj, "name").text = name
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bnd = ET.SubElement(obj, "bndbox")
        for tag, value in zip(("xmin", "ymin", "xmax", "ymax"), corners):
            ET.SubElement(bnd, tag).text = str(value)
    return minidom.parseString(ET.tostring(ann)).toprettyxml(
        indent="  ", encoding="utf-8"
    )


def xml_member_names(image_names: Sequence[str]) -> dict[str, str]:
    """Image name -> zip member name: ``<stem>.xml`` (workbench layout), or ``<name>.xml`` where two
    images share a stem (``a.jpg`` and ``a.png``)."""
    stems: dict[str, int] = {}
    for name in image_names:
        stem = PurePosixPath(name).stem.lower()
        stems[stem] = stems.get(stem, 0) + 1
    out = {}
    for name in image_names:
        stem = PurePosixPath(name).stem
        out[name] = f"{stem}.xml" if stems[stem.lower()] == 1 else f"{name}.xml"
    return out


def voc_zip(rows: Sequence[dict]) -> bytes:
    """Zip of VOC files. Each row: ``image``, ``width``, ``height``, ``annotations``, ``reviewed``."""
    members = xml_member_names([r["image"] for r in rows])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for r in rows:
            zf.writestr(
                f"annotations/{members[r['image']]}",
                voc_xml(
                    r["image"],
                    r["width"],
                    r["height"],
                    r["annotations"],
                    reviewed=bool(r.get("reviewed")),
                ),
            )
    return buf.getvalue()


# --------------------------------------------------------------------------- reading


@dataclass
class VocObject:
    label: str
    corners: tuple[float, float, float, float]  # xmin, ymin, xmax, ymax as written


@dataclass
class VocFile:
    member: str
    filename: str  # leaf of <filename> (or <path>), "" when neither is present
    width: int
    height: int
    objects: list[VocObject] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # per-object reasons


def _text(el: Optional[ET.Element], path: str) -> Optional[str]:
    node = None if el is None else el.find(path)
    if node is None or node.text is None:
        return None
    return node.text.strip()


def _leaf(name: str) -> str:
    return name.replace("\\", "/").rstrip("/").split("/")[-1]


def parse_voc(data: bytes, member: str = "") -> VocFile:
    """Parse one VOC document. Raises ``VocError`` for unsafe or malformed XML."""
    # UTF-16/32 would hide a DOCTYPE from the byte search below; their NUL bytes give them away.
    if b"\x00" in data:
        raise VocError("only UTF-8 (or ASCII) XML is accepted")
    if _DTD.search(data):
        raise VocError("DOCTYPE/ENTITY declarations are not accepted")
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise VocError(f"malformed XML: {exc}") from exc
    if root.tag != "annotation":
        raise VocError(f"root element is <{root.tag}>, not <annotation>")
    filename = _leaf(_text(root, "filename") or "") or _leaf(_text(root, "path") or "")

    def size(tag: str) -> int:
        try:
            value = float(_text(root, f"size/{tag}") or 0)
        except ValueError:
            return 0
        return int(value) if math.isfinite(value) and value > 0 else 0

    out = VocFile(
        member=member, filename=filename, width=size("width"), height=size("height")
    )
    for i, obj in enumerate(root.findall(".//object")):
        name = _text(obj, "name")
        bnd = obj.find("bndbox")
        if not name:
            out.skipped.append(f"object[{i}]: no <name>")
            continue
        if bnd is None:
            out.skipped.append(f"object[{i}] {name!r}: no <bndbox>")
            continue
        try:
            corners = tuple(
                float(_text(bnd, t) or "nan") for t in ("xmin", "ymin", "xmax", "ymax")
            )
        except ValueError:
            corners = (math.nan,) * 4
        if not all(math.isfinite(v) for v in corners):
            out.skipped.append(
                f"object[{i}] {name!r}: <bndbox> coordinates are missing or not numbers"
            )
            continue
        out.objects.append(VocObject(label=name, corners=corners))  # type: ignore[arg-type]
    return out


def to_bbox(corners: Sequence[float], width: int, height: int) -> Optional[list[float]]:
    """VOC corners -> ``[x, y, w, h]`` clamped to the image; ``None`` if nothing is left."""
    x0, y0, x1, y1 = corners
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    x0, x1 = max(0.0, min(width, x0)), max(0.0, min(width, x1))
    y0, y1 = max(0.0, min(height, y0)), max(0.0, min(height, y1))
    if x1 - x0 <= 0 or y1 - y0 <= 0:
        return None
    return [x0, y0, x1 - x0, y1 - y0]


def _unsafe_member(name: str) -> Optional[str]:
    if "\x00" in name or name.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", name):
        return "absolute path"
    if any(part == ".." for part in re.split(r"[\\/]", name)):
        return "path traversal ('..')"
    return None


def read_voc_zip(data: bytes) -> tuple[list[VocFile], list[dict]]:
    """Parse every ``*.xml`` member. Returns the parsed files and ``[{"file", "reason"}]`` for the
    members that were refused. Raises ``VocError`` when the archive itself is unacceptable."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except (zipfile.BadZipFile, OSError) as exc:
        raise VocError(f"body is not a zip archive: {exc}") from exc
    files: list[VocFile] = []
    skipped: list[dict] = []
    total = 0
    with zf:
        infos = zf.infolist()
        if len(infos) > MAX_MEMBERS:
            raise VocError(
                f"archive has {len(infos)} members; the limit is {MAX_MEMBERS}"
            )
        for info in infos:
            name = info.filename
            if (
                info.is_dir()
                or name.startswith("__MACOSX/")
                or _leaf(name).startswith("._")
            ):
                continue
            if not name.lower().endswith(".xml"):
                skipped.append({"file": name, "reason": "not an .xml file"})
                continue
            unsafe = _unsafe_member(name)
            if unsafe:
                skipped.append(
                    {"file": name, "reason": f"unsafe member name: {unsafe}"}
                )
                continue
            if info.file_size > MAX_MEMBER_BYTES:
                skipped.append(
                    {"file": name, "reason": f"larger than {MAX_MEMBER_BYTES} bytes"}
                )
                continue
            try:
                with zf.open(info) as fh:
                    body = fh.read(MAX_MEMBER_BYTES + 1)
            except (
                zipfile.BadZipFile,
                RuntimeError,
                NotImplementedError,
                OSError,
                EOFError,
            ) as exc:
                skipped.append({"file": name, "reason": f"cannot read member: {exc}"})
                continue
            if len(body) > MAX_MEMBER_BYTES:
                skipped.append(
                    {"file": name, "reason": f"larger than {MAX_MEMBER_BYTES} bytes"}
                )
                continue
            total += len(body)
            if total > MAX_TOTAL_BYTES:
                raise VocError(f"archive expands to more than {MAX_TOTAL_BYTES} bytes")
            try:
                files.append(parse_voc(body, member=name))
            except VocError as exc:
                skipped.append({"file": name, "reason": str(exc)})
    return files, skipped


__all__ = [
    "MAX_MEMBERS",
    "MAX_MEMBER_BYTES",
    "MAX_TOTAL_BYTES",
    "MAX_ZIP_BYTES",
    "VocError",
    "VocFile",
    "VocObject",
    "parse_voc",
    "read_voc_zip",
    "to_bbox",
    "voc_corners",
    "voc_xml",
    "voc_zip",
    "xml_member_names",
]
