"""MS COCO object-detection JSON (and the COCO Camera Traps image-label variant).

Writing: ``images`` (``id``, ``file_name``, ``width``, ``height``), ``categories`` (``id``,
``name``) and ``annotations`` (``id``, ``image_id``, ``category_id``, pixel ``bbox`` ``[x, y, w,
h]``, ``area``, ``iscrowd``). An image-level classification label is an annotation without ``bbox``, the way
COCO Camera Traps stores image-level species labels; a boxed classification label keeps its box. Each image also carries the Sparrow review
status as an extra ``review_status`` field, which other COCO tools ignore.

Reading: a ``.json`` body, or a zip holding one ``.json``. An image with no annotations is a
reviewed blank frame, and so is one whose only category is ``empty`` or ``blank`` (COCO Camera
Traps marks empty frames that way).
"""

from __future__ import annotations

import io
import json
import math
import zipfile
from typing import Sequence

from .interchange import Ann, FormatError, Record, leaf, zip_members

MAX_JSON_BYTES = 64 * 1024 * 1024
BLANK_CATEGORIES = {"empty", "blank"}


def coco_dict(rows: Sequence[dict], task: str, description: str = "") -> dict:
    """Rows: ``image``, ``width``, ``height``, ``annotations`` (``label``, ``bbox``), optional
    ``review_status``."""
    labels = sorted(
        {a["label"] for r in rows for a in r["annotations"] if a.get("label")}
    )
    cat_id = {name: i + 1 for i, name in enumerate(labels)}
    images, annotations = [], []
    for image_id, r in enumerate(rows, start=1):
        img = {
            "id": image_id,
            "file_name": r["image"],
            "width": r["width"],
            "height": r["height"],
        }
        if r.get("review_status"):
            img["review_status"] = r["review_status"]
        images.append(img)
        for a in r["annotations"]:
            if not a.get("label"):
                continue
            ann = {
                "id": len(annotations) + 1,
                "image_id": image_id,
                "category_id": cat_id[a["label"]],
                "iscrowd": 0,
            }
            if a.get("bbox"):
                x, y, w, h = (float(v) for v in a["bbox"])
                ann["bbox"] = [round(x, 2), round(y, 2), round(w, 2), round(h, 2)]
                ann["area"] = round(w * h, 2)
            annotations.append(ann)
    return {
        "info": {
            "description": description or "Exported by Sparrow fine-tune",
            "version": "1.0",
        },
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": i, "name": n, "supercategory": ""} for n, i in cat_id.items()
        ],
    }


def coco_json(rows: Sequence[dict], task: str, description: str = "") -> bytes:
    return json.dumps(coco_dict(rows, task, description), indent=1).encode()


def _number(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError
    value = float(value)
    if not math.isfinite(value):
        raise ValueError
    return value


def read_coco(data: bytes) -> tuple[list[Record], list[dict]]:
    """Records for every image in the document, plus ``[{"file", "reason"}]`` for refused
    entries. Raises ``FormatError`` when the document as a whole is unusable."""
    if data[:2] == b"PK":
        _, data = zip_members_json(data)
    try:
        doc = json.loads(data.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FormatError(f"not a JSON document: {exc}") from exc
    if not isinstance(doc, dict) or not isinstance(doc.get("images"), list):
        raise FormatError("COCO JSON needs an 'images' list")
    anns_in = doc.get("annotations") or []
    cats_in = doc.get("categories") or []
    if not isinstance(anns_in, list) or not isinstance(cats_in, list):
        raise FormatError("'annotations' and 'categories' must be lists")

    skipped: list[dict] = []
    categories: dict = {}
    for c in cats_in:
        if isinstance(c, dict) and isinstance(c.get("name"), str) and c["name"].strip():
            categories[c.get("id")] = c["name"].strip()

    records: dict = {}
    for i, img in enumerate(doc["images"]):
        where = f"images[{i}]"
        if not isinstance(img, dict) or not isinstance(img.get("file_name"), str):
            skipped.append({"file": where, "reason": "image has no file_name"})
            continue
        if img.get("id") in records:
            skipped.append(
                {"file": where, "reason": f"duplicate image id {img.get('id')!r}"}
            )
            continue
        try:
            width, height = (
                int(_number(img.get("width"))),
                int(_number(img.get("height"))),
            )
        except ValueError:
            width = height = 0
        records[img.get("id")] = Record(
            source=img["file_name"],
            filename=leaf(img["file_name"]),
            width=max(width, 0),
            height=max(height, 0),
        )

    for j, a in enumerate(anns_in):
        where = f"annotations[{j}]"
        if not isinstance(a, dict):
            skipped.append({"file": where, "reason": "annotation is not an object"})
            continue
        rec = records.get(a.get("image_id"))
        if rec is None:
            skipped.append(
                {"file": where, "reason": f"unknown image_id {a.get('image_id')!r}"}
            )
            continue
        label = categories.get(a.get("category_id"))
        if label is None:
            rec.skipped.append(f"{where}: unknown category_id {a.get('category_id')!r}")
            continue
        if label.lower() in BLANK_CATEGORIES:
            continue
        bbox = a.get("bbox")
        if bbox is None or bbox == []:
            rec.anns.append(Ann(label=label))
            continue
        try:
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError
            x, y, w, h = (_number(v) for v in bbox)
        except ValueError:
            rec.skipped.append(f"{where} {label!r}: bbox {bbox!r} is not four numbers")
            continue
        rec.anns.append(Ann(label=label, corners=(x, y, x + w, y + h)))
    return list(records.values()), skipped


def zip_members_json(data: bytes) -> tuple[str, bytes]:
    try:
        names = [
            i.filename
            for i in zipfile.ZipFile(io.BytesIO(data)).infolist()
            if i.filename.lower().endswith(".json")
            and not i.is_dir()
            and not i.filename.startswith("__MACOSX/")
            and not leaf(i.filename).startswith("._")
        ]
    except (zipfile.BadZipFile, OSError) as exc:
        raise FormatError(f"body is not a zip archive: {exc}") from exc
    if len(names) != 1:
        raise FormatError(
            f"the zip must hold exactly one .json file, found {len(names)}"
        )
    members = zip_members(data, {leaf(names[0]).lower()}, MAX_JSON_BYTES)
    return names[0], next(iter(members.values()))


__all__ = ["MAX_JSON_BYTES", "coco_dict", "coco_json", "read_coco"]
