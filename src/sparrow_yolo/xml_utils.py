"""Pascal VOC XML parsing and writing.

The SPARROW datasets store one ``.xml`` next to each image. This module owns the
single source of truth for reading and writing those files, and for converting a
VOC box into YOLO's normalised ``cls xc yc w h`` form. Nothing else in the
package should re-implement any of that.

Writing exists for the annotation tool, which is the only part of the system that
produces annotations rather than consuming them. It never writes into a dataset
the pipeline reads from — see :mod:`sparrow_web.annotation`.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from xml.dom import minidom


@dataclass(frozen=True)
class Box:
    """One VOC ``<object>``.

    ``width``/``height`` are whatever the XML claimed. They are only a fallback:
    :func:`sparrow_yolo.dataset.audit_dataset` overrides them with the real pixel
    dimensions read from the image, because VOC files routinely disagree with the
    image after a resize.
    """

    image: str
    width: int
    height: int
    class_name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    xml_path: str
    difficult: bool = False
    truncated: bool = False

    def clipped(self, width: Optional[float] = None, height: Optional[float] = None) -> "Box":
        """Return a copy with coordinates clamped into ``[0, width] x [0, height]``."""
        w = float(self.width if width is None else width)
        h = float(self.height if height is None else height)
        return Box(
            image=self.image,
            width=int(w),
            height=int(h),
            class_name=self.class_name,
            xmin=max(0.0, min(w, self.xmin)),
            ymin=max(0.0, min(h, self.ymin)),
            xmax=max(0.0, min(w, self.xmax)),
            ymax=max(0.0, min(h, self.ymax)),
            xml_path=self.xml_path,
            difficult=self.difficult,
            truncated=self.truncated,
        )

    def is_valid(self, width: Optional[float] = None, height: Optional[float] = None) -> bool:
        b = self.clipped(width, height)
        return b.width > 0 and b.height > 0 and b.xmax > b.xmin and b.ymax > b.ymin

    def to_yolo(self, class_id: int, width: Optional[float] = None, height: Optional[float] = None) -> str:
        return to_yolo_line(
            class_id,
            self.xmin,
            self.ymin,
            self.xmax,
            self.ymax,
            self.width if width is None else width,
            self.height if height is None else height,
        )


def to_yolo_line(
    class_id: int,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    width: float,
    height: float,
) -> Optional[str]:
    """Convert one absolute VOC box to a YOLO label line.

    Returns ``None`` when the box is degenerate after clipping, so callers can
    drop it rather than writing a label the trainer will choke on. This is the
    only implementation of the conversion in the package.
    """
    w = float(width)
    h = float(height)
    if w <= 0 or h <= 0:
        return None

    x0 = max(0.0, min(w, float(xmin)))
    y0 = max(0.0, min(h, float(ymin)))
    x1 = max(0.0, min(w, float(xmax)))
    y1 = max(0.0, min(h, float(ymax)))
    if x1 <= x0 or y1 <= y0:
        return None

    xc = ((x0 + x1) / 2.0) / w
    yc = ((y0 + y1) / 2.0) / h
    bw = (x1 - x0) / w
    bh = (y1 - y0) / h

    # Guard against float error pushing a value a hair outside [0, 1]; Ultralytics
    # rejects labels with coordinates > 1.
    xc, yc = min(max(xc, 0.0), 1.0), min(max(yc, 0.0), 1.0)
    bw, bh = min(bw, 1.0), min(bh, 1.0)
    if bw <= 0 or bh <= 0:
        return None
    return f"{int(class_id)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def _txt(root: ET.Element, path: str, default: Optional[str] = None) -> Optional[str]:
    el = root.find(path)
    if el is None or el.text is None:
        return default
    return el.text.strip()


def _num(text: Optional[str], default: float = 0.0) -> float:
    if text is None:
        return default
    try:
        return float(text.strip())
    except (TypeError, ValueError):
        return default


def parse_voc_xml(xml_path: Path) -> List[Box]:
    """Parse one VOC file. Raises on malformed XML so the caller can log it."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = _txt(root, "filename") or _txt(root, "path") or (Path(xml_path).stem + ".jpg")
    # `filename` is sometimes a full Windows path; keep only the leaf.
    leaf = filename.replace("\\", "/").rstrip("/").split("/")[-1] or (Path(xml_path).stem + ".jpg")

    width = int(_num(_txt(root, "size/width"), 0.0))
    height = int(_num(_txt(root, "size/height"), 0.0))

    boxes: List[Box] = []
    for obj in root.findall(".//object"):
        name_el = obj.find("name")
        bnd = obj.find("bndbox")
        if name_el is None or bnd is None or name_el.text is None:
            continue
        boxes.append(
            Box(
                image=leaf,
                width=width,
                height=height,
                class_name=name_el.text.strip(),
                xmin=_num(_txt(bnd, "xmin")),
                ymin=_num(_txt(bnd, "ymin")),
                xmax=_num(_txt(bnd, "xmax")),
                ymax=_num(_txt(bnd, "ymax")),
                xml_path=str(xml_path),
                difficult=_num(_txt(obj, "difficult"), 0.0) >= 1.0,
                truncated=_num(_txt(obj, "truncated"), 0.0) >= 1.0,
            )
        )
    return boxes


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------

#: Where a box came from. Recorded per object so a file that leaves the
#: workbench still says which parts a model guessed and which a person stands
#: behind. Anything else is treated as ``model`` on read.
BOX_SOURCES = ("model", "human")


def read_voc_boxes(xml_path: Path) -> List[Dict[str, Any]]:
    """Parse to plain dicts for the annotation editor.

    :func:`parse_voc_xml` returns frozen :class:`Box` objects keyed for the
    training pipeline. The editor needs mutable records carrying the two fields
    that only exist on annotations we wrote ourselves — ``score`` and ``source``
    — so it gets its own shallow reader rather than distorting ``Box``.
    """
    path = Path(xml_path)
    if not path.exists():
        return []
    root = ET.parse(path).getroot()
    out: List[Dict[str, Any]] = []
    for obj in root.findall(".//object"):
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        source = (_txt(obj, "source") or "model").lower()
        out.append({
            "label": _txt(obj, "name") or "object",
            "xmin": _num(_txt(bnd, "xmin")),
            "ymin": _num(_txt(bnd, "ymin")),
            "xmax": _num(_txt(bnd, "xmax")),
            "ymax": _num(_txt(bnd, "ymax")),
            "score": _num(_txt(obj, "confidence"), 1.0),
            "source": source if source in BOX_SOURCES else "model",
        })
    return out


def write_voc_xml(
    out_path: str | Path,
    image_path: str | Path,
    width: int,
    height: int,
    boxes: Sequence[Dict[str, Any]],
    *,
    reviewed: bool = False,
    predicted_by: str = "",
    database: str = "sparrow_workbench",
) -> Path:
    """Write one VOC file. Returns the path written.

    Degenerate and out-of-bounds boxes are dropped rather than written, for the
    same reason :func:`to_yolo_line` returns ``None`` for them: a label the
    trainer will reject is worse than no label, and the audit reports the gap.

    An empty ``boxes`` is meaningful and produces a valid object-free file — that
    is how a reviewed blank frame is recorded, and blank frames are first-class
    negatives in this pipeline.
    """
    image_path = Path(image_path)
    width, height = int(width), int(height)

    ann = ET.Element("annotation")
    if reviewed:
        # LabelImg's convention, which SPARROW's existing files already use.
        ann.set("verified", "yes")
    ET.SubElement(ann, "folder").text = image_path.parent.name
    ET.SubElement(ann, "filename").text = image_path.name
    ET.SubElement(ann, "path").text = str(image_path)
    source = ET.SubElement(ann, "source")
    ET.SubElement(source, "database").text = database
    if predicted_by:
        ET.SubElement(ann, "predicted_by").text = str(predicted_by)
    size = ET.SubElement(ann, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    ET.SubElement(ann, "segmented").text = "0"

    for b in boxes:
        x0, y0 = float(b.get("xmin", 0)), float(b.get("ymin", 0))
        x1, y1 = float(b.get("xmax", 0)), float(b.get("ymax", 0))
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        x0 = int(round(max(0.0, min(width, x0))))
        x1 = int(round(max(0.0, min(width, x1))))
        y0 = int(round(max(0.0, min(height, y0))))
        y1 = int(round(max(0.0, min(height, y1))))
        name = str(b.get("label", "")).strip()
        if x1 <= x0 or y1 <= y0 or not name:
            continue

        obj = ET.SubElement(ann, "object")
        ET.SubElement(obj, "name").text = name
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        box_source = str(b.get("source", "model")).lower()
        ET.SubElement(obj, "source").text = (
            box_source if box_source in BOX_SOURCES else "model")
        if b.get("score") is not None:
            ET.SubElement(obj, "confidence").text = f"{float(b['score']):.6f}"
        bnd = ET.SubElement(obj, "bndbox")
        for tag, value in (("xmin", x0), ("ymin", y0), ("xmax", x1), ("ymax", y1)):
            ET.SubElement(bnd, tag).text = str(value)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pretty = minidom.parseString(ET.tostring(ann)).toprettyxml(indent="  ")
    out_path.write_text(pretty, encoding="utf-8")
    return out_path
