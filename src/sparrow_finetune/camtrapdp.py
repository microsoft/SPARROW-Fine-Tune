"""Camera Trap Data Package (Camtrap DP 1.0.2, https://camtrap-dp.tdwg.org).

A package is a zip with ``datapackage.json`` and three CSV tables: ``deployments.csv``,
``media.csv`` and ``observations.csv``. Boxes (``bboxX``, ``bboxY``, ``bboxWidth``, ``bboxHeight``)
are fractions of the image with the origin at the top left.

Writing: one media row per item and one media-level observation per label. ``person``/``human``
become ``observationType=human``, ``vehicle`` stays ``vehicle``, ``animal`` is an animal without a
species, and any other label is an animal with ``scientificName`` set to the label. A confirmed
item (or a snapshot item) without labels is a ``blank`` observation; an unreviewed one is
``unclassified``. Sparrow
datasets hold no camera location or capture time, so the single deployment has empty
coordinates and every timestamp is the export time: the package carries the annotations, but a
strict Camtrap DP validator will report the missing location.

Reading: media are matched by ``fileName`` (else the last part of ``filePath``). The label is
``scientificName`` when present, else the observation type. ``blank`` marks a reviewed empty
image; ``unknown`` and ``unclassified`` observations carry no label and are ignored. An
event-level observation (no ``mediaID``) labels every media file of its deployment whose
timestamp lies between ``eventStart`` and ``eventEnd``.
"""

from __future__ import annotations

import csv
import io
import json
import mimetypes
import re
from datetime import datetime, timezone
from typing import Optional, Sequence

from .interchange import Ann, FormatError, Record, leaf, zip_members, zipped

VERSION = "1.0.2"
PROFILE = f"https://raw.githubusercontent.com/tdwg/camtrap-dp/{VERSION}/camtrap-dp-profile.json"
MAX_TABLE_BYTES = 64 * 1024 * 1024
MISSING = {"", "na", "nan"}

DEPLOYMENT_FIELDS = [
    "deploymentID", "locationID", "locationName", "latitude", "longitude",
    "coordinateUncertainty", "deploymentStart", "deploymentEnd", "setupBy", "cameraID",
    "cameraModel", "cameraDelay", "cameraHeight", "cameraDepth", "cameraTilt", "cameraHeading",
    "detectionDistance", "timestampIssues", "baitUse", "featureType", "habitat",
    "deploymentGroups", "deploymentTags", "deploymentComments",
]  # fmt: skip
MEDIA_FIELDS = [
    "mediaID", "deploymentID", "captureMethod", "timestamp", "filePath", "filePublic",
    "fileName", "fileMediatype", "exifData", "favorite", "mediaComments",
]  # fmt: skip
OBSERVATION_FIELDS = [
    "observationID", "deploymentID", "mediaID", "eventID", "eventStart", "eventEnd",
    "observationLevel", "observationType", "cameraSetupType", "scientificName", "count",
    "lifeStage", "sex", "behavior", "individualID", "individualPositionRadius",
    "individualPositionAngle", "individualSpeed", "bboxX", "bboxY", "bboxWidth", "bboxHeight",
    "classificationMethod", "classifiedBy", "classificationTimestamp",
    "classificationProbability", "observationTags", "observationComments",
]  # fmt: skip

HUMAN_LABELS = {"person", "human", "people"}
VEHICLE_LABELS = {"vehicle"}
ANIMAL_LABELS = {"animal"}


# --------------------------------------------------------------------------- writing


def _csv(fields: Sequence[str], rows: Sequence[dict]) -> bytes:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in fields})
    return buf.getvalue().encode()


def _type_and_name(label: str) -> tuple[str, str]:
    low = label.strip().lower()
    if low in HUMAN_LABELS:
        return "human", ""
    if low in VEHICLE_LABELS:
        return "vehicle", ""
    if low in ANIMAL_LABELS:
        return "animal", ""
    return "animal", label.strip()


def _fraction(value: float) -> str:
    return f"{min(max(value, 0.0), 1.0):.6f}".rstrip("0").rstrip(".") or "0"


def write_package(
    rows: Sequence[dict],
    task: str,
    title: str,
    now: Optional[datetime] = None,
) -> bytes:
    """Rows: ``image``, ``width``, ``height``, ``annotations`` (``label``, ``bbox``) and optional
    ``review_status`` and ``blank_if_empty`` (a snapshot row: no labels means a blank frame).
    Returns the zipped package."""
    now = (now or datetime.now(timezone.utc)).replace(microsecond=0)
    stamp = now.strftime("%Y-%m-%dT%H:%M:%S%z")
    deployment = "sparrow-deployment-1"
    media_rows, obs_rows = [], []
    species: set[str] = set()
    for i, r in enumerate(rows, start=1):
        media_id = f"m{i}"
        media_rows.append(
            {
                "mediaID": media_id,
                "deploymentID": deployment,
                "captureMethod": "activityDetection",
                "timestamp": stamp,
                "filePath": f"media/{r['image']}",
                "filePublic": "false",
                "fileName": r["image"],
                "fileMediatype": mimetypes.guess_type(r["image"])[0] or "image/jpeg",
            }
        )
        confirmed = r.get("review_status") == "confirmed"
        base = {
            "deploymentID": deployment,
            "mediaID": media_id,
            "eventStart": stamp,
            "eventEnd": stamp,
            "observationLevel": "media",
            "classificationMethod": "human" if confirmed else "",
            "classificationTimestamp": stamp if confirmed else "",
        }
        anns = [a for a in r["annotations"] if a.get("label")]
        if not anns:
            is_blank = confirmed or bool(r.get("blank_if_empty"))
            obs_rows.append(
                {
                    **base,
                    "observationID": f"o{len(obs_rows) + 1}",
                    "observationType": "blank" if is_blank else "unclassified",
                }
            )
            continue
        for a in anns:
            obs_type, name = _type_and_name(a["label"])
            if name:
                species.add(name)
            row = {
                **base,
                "observationID": f"o{len(obs_rows) + 1}",
                "observationType": obs_type,
                "scientificName": name,
                "count": "1" if task == "detection" or a.get("bbox") else "",
            }
            if a.get("bbox") and r["width"] and r["height"]:
                x, y, w, h = (float(v) for v in a["bbox"])
                row.update(
                    bboxX=_fraction(x / r["width"]),
                    bboxY=_fraction(y / r["height"]),
                    bboxWidth=_fraction(w / r["width"]),
                    bboxHeight=_fraction(h / r["height"]),
                )
            obs_rows.append(row)

    schema = f"https://raw.githubusercontent.com/tdwg/camtrap-dp/{VERSION}/{{}}-table-schema.json"
    package = {
        "profile": PROFILE,
        "name": re.sub(r"[^a-z0-9._-]+", "-", title.lower()).strip("-")
        or "sparrow-dataset",
        "title": title,
        "description": (
            "Annotations exported by Sparrow fine-tune. The dataset holds no camera location "
            "or capture time: the deployment coordinates are empty and every timestamp is the "
            "export time."
        ),
        "created": stamp,
        "contributors": [{"title": "Sparrow fine-tune", "role": "publisher"}],
        "project": {
            "title": title,
            "samplingDesign": "opportunistic",
            "captureMethod": ["activityDetection"],
            "individualAnimals": False,
            "observationLevel": ["media"],
        },
        "temporal": {"start": now.date().isoformat(), "end": now.date().isoformat()},
        "taxonomic": [{"scientificName": s} for s in sorted(species)],
        "resources": [
            {
                "name": name,
                "path": f"{name}.csv",
                "profile": "tabular-data-resource",
                "format": "csv",
                "mediatype": "text/csv",
                "encoding": "UTF-8",
                "schema": schema.format(name),
            }
            for name in ("deployments", "media", "observations")
        ],
    }
    deployments = [
        {
            "deploymentID": deployment,
            "locationName": title,
            "deploymentStart": stamp,
            "deploymentEnd": stamp,
        }
    ]
    return zipped(
        {
            "datapackage.json": json.dumps(package, indent=2).encode(),
            "deployments.csv": _csv(DEPLOYMENT_FIELDS, deployments),
            "media.csv": _csv(MEDIA_FIELDS, media_rows),
            "observations.csv": _csv(OBSERVATION_FIELDS, obs_rows),
        }
    )


# --------------------------------------------------------------------------- reading


def _value(row: dict, key: str) -> str:
    value = (row.get(key) or "").strip()
    return "" if value.lower() in MISSING else value


def parse_time(text: str) -> Optional[datetime]:
    """ISO 8601 with ``Z``, ``+hh:mm`` or ``+hhmm`` offsets (Python 3.10's ``fromisoformat``
    takes only the second); a time without an offset is read as UTC."""
    text = text.strip()
    if not text:
        return None
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    text = re.sub(r"([+-]\d\d)(\d\d)$", r"\1:\2", text)
    try:
        value = datetime.fromisoformat(text)
    except ValueError:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _table(body: bytes, name: str) -> list[dict]:
    try:
        text = body.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise FormatError(f"{name} is not UTF-8: {exc}") from exc
    try:
        return list(csv.DictReader(io.StringIO(text)))
    except csv.Error as exc:
        raise FormatError(f"{name} is not a readable CSV: {exc}") from exc


def _resource_names(data: bytes) -> dict[str, str]:
    """Leaf file names of the media and observations tables, from ``datapackage.json`` when the
    package has one."""
    names = {"media": "media.csv", "observations": "observations.csv"}
    found = zip_members(data, {"datapackage.json"}, MAX_TABLE_BYTES)
    if not found:
        return names
    try:
        package = json.loads(found["datapackage.json"].decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FormatError(f"datapackage.json is not valid JSON: {exc}") from exc
    for res in package.get("resources", []) if isinstance(package, dict) else []:
        if not isinstance(res, dict) or res.get("name") not in names:
            continue
        path = res.get("path")
        if isinstance(path, list) and path:
            path = path[0]
        if isinstance(path, str) and path:
            names[res["name"]] = leaf(path)
    return names


def read_package(data: bytes) -> tuple[list[Record], list[dict]]:
    """One record per media file that has at least one usable observation, plus
    ``[{"file", "reason"}]`` for refused rows."""
    names = _resource_names(data)
    tables = zip_members(data, {n.lower() for n in names.values()}, MAX_TABLE_BYTES)
    for kind, name in names.items():
        if name.lower() not in tables:
            raise FormatError(f"the package has no {name} ({kind} table)")
    media = _table(tables[names["media"].lower()], names["media"])
    observations = _table(tables[names["observations"].lower()], names["observations"])
    skipped: list[dict] = []

    records: dict[str, Record] = {}
    blank: set[str] = set()
    by_deployment: dict[str, list[tuple[Optional[datetime], str]]] = {}
    for i, row in enumerate(media, start=2):
        media_id = _value(row, "mediaID")
        path = _value(row, "filePath")
        filename = leaf(_value(row, "fileName") or path)
        if not media_id or not filename:
            skipped.append(
                {
                    "file": f"{names['media']} line {i}",
                    "reason": "no mediaID or file name",
                }
            )
            continue
        if media_id in records:
            skipped.append(
                {
                    "file": f"{names['media']} line {i}",
                    "reason": f"duplicate mediaID {media_id!r}",
                }
            )
            continue
        records[media_id] = Record(
            source=f"{names['media']} line {i}",
            filename=filename,
            alt_names=[n for n in {leaf(path)} if n and n != filename],
        )
        by_deployment.setdefault(_value(row, "deploymentID"), []).append(
            (parse_time(_value(row, "timestamp")), media_id)
        )

    labelled: set[str] = set()
    ignored = 0
    for i, row in enumerate(observations, start=2):
        where = f"{names['observations']} line {i}"
        obs_type = _value(row, "observationType").lower()
        if obs_type in ("unknown", "unclassified"):
            ignored += 1
            continue
        label = _value(row, "scientificName") or obs_type
        media_id = _value(row, "mediaID")
        level = _value(row, "observationLevel").lower() or (
            "media" if media_id else "event"
        )
        if level == "media" or media_id:
            if media_id not in records:
                skipped.append(
                    {"file": where, "reason": f"unknown mediaID {media_id!r}"}
                )
                continue
            targets = [media_id]
        else:
            start, end = (
                parse_time(_value(row, "eventStart")),
                parse_time(_value(row, "eventEnd")),
            )
            if start is None or end is None:
                skipped.append(
                    {
                        "file": where,
                        "reason": "event observation without a readable eventStart/eventEnd",
                    }
                )
                continue
            targets = [
                m
                for ts, m in by_deployment.get(_value(row, "deploymentID"), [])
                if ts is not None and start <= ts <= end
            ]
            if not targets:
                skipped.append(
                    {
                        "file": where,
                        "reason": "no media file of the deployment falls inside the event",
                    }
                )
                continue
        if obs_type == "blank":
            blank.update(targets)
            continue
        if not label:
            skipped.append(
                {
                    "file": where,
                    "reason": "observation has no scientificName or observationType",
                }
            )
            continue
        corners = None
        box = [_value(row, k) for k in ("bboxX", "bboxY", "bboxWidth", "bboxHeight")]
        if any(box):
            try:
                x, y, w, h = (float(v) for v in box)
                corners = (x, y, x + w, y + h)
            except ValueError:
                skipped.append(
                    {"file": where, "reason": f"bbox {box} is not four numbers"}
                )
                continue
        for m in targets:
            records[m].anns.append(Ann(label=label, corners=corners, normalized=True))
            labelled.add(m)
    if ignored:
        skipped.append(
            {
                "file": names["observations"],
                "reason": f"{ignored} unknown/unclassified observation(s) carry no label and were ignored",
            }
        )
    return [records[m] for m in records if m in labelled or m in blank], skipped


__all__ = ["PROFILE", "VERSION", "parse_time", "read_package", "write_package"]
