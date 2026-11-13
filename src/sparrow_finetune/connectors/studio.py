"""Sparrow Studio connector: import reviewed annotations, push trained models.

Contract (base ``<studio_url>/api/finetune/v1``, bearer ``studio_token``):

- ``GET /projects/{project}/export?reviewed=1|0&limit=&cursor=`` ->
  ``{project, items: [{image_id, name, width, height, reviewed, boxes: [{id, label,
  bbox: [x, y, w, h]}]}], next_cursor}``. ``bbox`` is pixels in the original image;
  ``width``/``height`` may be null (Studio does not store them) and are not used here.
- ``GET /images/{image_id}`` -> image bytes
- ``POST /projects/{project}/models`` multipart ``manifest`` (JSON) + ``bundle`` (tar
  holding the artifact, ``manifest.json`` and, when the model converts, ``engine.onnx``
  described by ``manifest.engine``) -> ``{id}``
- ``POST /projects/{project}/models/{id}/activate`` -> ``{id, install_status}``: Studio
  installs ``engine.onnx`` into Sparrow Engine and switches the project to it.
- ``GET /projects/{project}/models`` -> ``{models: [{id, source_model_id, pipeline, task,
  format, labels, metrics, sha256, size_bytes, created_at, manifest}]}`` (that project's
  models only).
- ``GET /projects/{project}/models/{id}/bundle`` -> the tar.gz bundle pushed above, header
  ``X-Artifact-Sha256``; another project's id answers 404. Used to import a pushed
  warm-start pipeline (``base_models.WARM_START_PIPELINES``) back as a base model.

Every model request is under ``/projects/{project}/``: pushed models stay private to the
project, nothing is published to a shared catalogue.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tarfile
import time
from pathlib import PurePosixPath
from typing import Optional
from urllib.parse import quote

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from PIL import Image
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .. import base_models
from ..api.datasets import (
    ALLOWED_IMAGE_FORMATS,
    ItemIn,
    _image_size,
    _validate_name,
    apply_items,
    image_key,
    register_image,
)
from ..api.security import audit, principal_name, record_audit
from ..db import OBJECTS_LOCK, connect
from ..storage import LocalObjectStore
from .engine_package import (
    EngineConversionError,
    build_engine_package,
    detector_hints,
)

# Studio label vocabulary, after the original Studio fine-tune module (Studio commit
# 845dd51, model_finetuning/intake_contract_v1.py): NEGATIVE_LABELS refuse the item and
# the "uncertain" sentinel means "no label". Matched after strip + whitespace collapse +
# lowercase. An empty label is always unknown.
EXCLUDE_LABELS = frozenset(
    {"empty", "background", "blank", "none", "exclude", "excluded"}
)
UNKNOWN_LABELS = frozenset({"uncertain", "unknown", "unidentified"})

API_PREFIX = "/api/finetune/v1"
TIMEOUT_S = 30.0
PAGE_LIMIT = 100
MAX_PAGES = 1000
MANIFEST_SCHEMA = "sparrow-finetune/model-v1"
MAX_NAME = 255
MAX_SUFFIX = 16
PROJECT_IDS_ENV = "SPARROW_FT_STUDIO_PROJECT_IDS"
_FORBIDDEN = set("/\\") | {chr(c) for c in range(32)}

router = APIRouter()


class StudioError(Exception):
    """Studio was unreachable or answered with something unusable."""


def _settings(request: Request):
    settings = request.app.state.settings
    if not settings.studio_configured:
        raise HTTPException(
            409,
            "the Studio connector is not configured "
            "(set SPARROW_FT_STUDIO_URL and SPARROW_FT_STUDIO_TOKEN)",
        )
    return settings


def _client(request: Request, settings) -> httpx.Client:
    return httpx.Client(
        base_url=settings.studio_url + API_PREFIX,
        headers={"Authorization": f"Bearer {settings.studio_token}"},
        timeout=TIMEOUT_S,
        transport=getattr(request.app.state, "studio_transport", None),
    )


def _call(client: httpx.Client, method: str, path: str, **kwargs) -> httpx.Response:
    try:
        r = client.request(method, path, **kwargs)
    except httpx.HTTPError as exc:
        raise StudioError(f"{method} {path}: {exc}") from exc
    if r.status_code >= 400:
        raise StudioError(
            f"{method} {path}: Studio answered {r.status_code}: {r.text[:200]}"
        )
    return r


def _json(r: httpx.Response) -> dict:
    try:
        body = r.json()
    except ValueError as exc:
        raise StudioError(f"{r.request.url.path}: response is not JSON") from exc
    if not isinstance(body, dict):
        raise StudioError(f"{r.request.url.path}: response is not a JSON object")
    return body


def _clean(text) -> str:
    return "".join("_" if c in _FORBIDDEN else c for c in str(text))


def _safe(name: str) -> str:
    cleaned = _clean(name).strip()
    if cleaned in ("", ".", ".."):
        cleaned = "image"
    if len(cleaned) > MAX_NAME:
        # Cut the stem, not the extension: crops must stay ``.png`` and the image format
        # suffix is what a reader of the dataset relies on.
        suffix = PurePosixPath(cleaned).suffix
        if 0 < len(suffix) <= MAX_SUFFIX:
            return cleaned[: MAX_NAME - len(suffix)] + suffix
        return cleaned[:MAX_NAME]
    return cleaned


def _tagged(stem: str, tag: str, suffix: str = "") -> str:
    """``stem + tag + suffix`` within ``MAX_NAME``, cutting only the stem.

    The tag (``__studio<id>``, ``__box<n>``) is what keeps the name unique, so it must
    survive truncation; ``_safe`` alone would cut it off a long name."""
    stem, tag, suffix = _clean(stem), _clean(tag), _clean(suffix)
    if len(suffix) > MAX_SUFFIX:
        stem, suffix = stem + suffix, ""
    room = MAX_NAME - len(tag) - len(suffix)
    if room < 1:
        return _safe(stem + tag + suffix)
    return _safe(stem[:room].lstrip() + tag + suffix)


def _crop_name(name: str, bid) -> str:
    """Studio crop image name: ``<image stem>__box<bid>.png``."""
    return _tagged(PurePosixPath(name).stem or name, f"__box{bid}", ".png")


def _project_ids_mode() -> str:
    mode = os.environ.get(PROJECT_IDS_ENV, "any").strip().lower() or "any"
    if mode not in ("any", "numeric"):
        raise HTTPException(
            500, f"{PROJECT_IDS_ENV} must be 'any' or 'numeric', got {mode!r}"
        )
    return mode


def _check_project(project: str) -> None:
    """With ``SPARROW_FT_STUDIO_PROJECT_IDS=numeric``, Studio project ids are integers;
    refuse anything else before calling Studio, instead of relaying Studio's error."""
    if _project_ids_mode() == "numeric" and not (
        project.isascii() and project.isdigit()
    ):
        raise HTTPException(
            422,
            f"Studio project must be a numeric project id (for example 12), got {project!r}",
        )


def _boxes(item: dict) -> list[tuple[str, str, list[float]]]:
    boxes = item.get("boxes")
    if not isinstance(boxes, list):
        raise StudioError(f"image {item.get('image_id')}: boxes is not a list")
    out = []
    for box in boxes:
        try:
            bid = str(box["id"])
            label = str(box["label"]).strip()
            bbox = [float(v) for v in box["bbox"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise StudioError(
                f"image {item.get('image_id')}: malformed box {box!r}"
            ) from exc
        if len(bbox) != 4 or not all(math.isfinite(v) for v in bbox):
            raise StudioError(f"image {item.get('image_id')}: malformed box {box!r}")
        out.append((bid, label, bbox))
    return out


def _crop_png(data: bytes, bbox: list[float]) -> bytes:
    x, y, w, h = bbox
    with Image.open(io.BytesIO(data), formats=ALLOWED_IMAGE_FORMATS) as im:
        im.load()
        left, top = max(0, math.floor(x)), max(0, math.floor(y))
        right, bottom = (
            min(im.width, math.ceil(x + w)),
            min(im.height, math.ceil(y + h)),
        )
        if right <= left or bottom <= top:
            raise StudioError(
                f"box {bbox} lies outside the {im.width}x{im.height} image"
            )
        crop = im.crop((left, top, right, bottom))
        if crop.mode not in ("RGB", "RGBA", "L"):
            crop = crop.convert("RGB")
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------- import


def _item_names(fetched: list[tuple]) -> list[str]:
    """Dataset image names for the fetched Studio images.

    Studio's ``name`` is the original filename: it can be null and is not unique within a
    project. A unique name is kept as is, so re-imports update the same items; a missing
    or repeated name gets the Studio image id added, so two images never share a name.
    """
    bases = []
    for item, *_ in fetched:
        raw = item.get("name")
        bases.append(_safe(raw) if isinstance(raw, str) and raw.strip() else None)
    counts: dict[str, int] = {}
    for base in bases:
        if base is not None:
            counts[base] = counts.get(base, 0) + 1
    names = []
    for (item, *_), base in zip(fetched, bases):
        image_id = _safe(str(item["image_id"]))
        if base is None:
            names.append(f"studio_{image_id}")
        elif counts[base] > 1:
            path = PurePosixPath(base)
            names.append(_tagged(path.stem, f"__studio{image_id}", path.suffix))
        else:
            names.append(base)
    return names


class ImportIn(BaseModel):
    project: str = Field(min_length=1, max_length=200)
    dataset_id: int
    reviewed_only: bool = True
    # None = the default vocabulary; [] turns the bucket off.
    exclude_labels: Optional[list[str]] = Field(default=None, max_length=200)
    unknown_labels: Optional[list[str]] = Field(default=None, max_length=200)


def _norm_label(label: str) -> str:
    return " ".join(str(label).split()).lower()


class _LabelRules:
    """Per-import exclude/unknown handling and its counters.

    Exclude-class boxes are dropped. Unknown-class boxes are not imported (the
    original treated ``uncertain`` as "no label", never as a class). Detection: an
    image with any unknown box is skipped, since dropping that box would teach the
    detector the object is background; an image left with no boxes is skipped (the
    original refused negative images). Classification: per-box crops are skipped; an
    image with no crop left is skipped.
    """

    def __init__(self, body: ImportIn):
        self.overridden = body.exclude_labels is not None or body.unknown_labels is not None
        self.exclude = (
            EXCLUDE_LABELS
            if body.exclude_labels is None
            else frozenset(_norm_label(x) for x in body.exclude_labels) - {""}
        )
        self.unknown = (
            UNKNOWN_LABELS
            if body.unknown_labels is None
            else frozenset(_norm_label(x) for x in body.unknown_labels) - {""}
        )
        overlap = self.exclude & self.unknown
        if overlap:
            raise HTTPException(
                422,
                f"labels cannot be both excluded and unknown: {sorted(overlap)}",
            )
        self.counts = dict.fromkeys(
            (
                "excluded_boxes",
                "unknown_boxes",
                "images_skipped_excluded",
                "images_skipped_unknown",
            ),
            0,
        )

    def apply(self, boxes: list, task: str) -> list:
        """Boxes to import for one image; ``[]`` means skip the image."""
        kept, unknown = [], 0
        for box in boxes:
            key = _norm_label(box[1])
            if key in self.exclude:
                self.counts["excluded_boxes"] += 1
            elif not key or key in self.unknown:
                unknown += 1
            else:
                kept.append(box)
        self.counts["unknown_boxes"] += unknown
        if (task == "detection" and unknown) or (not kept and unknown):
            self.counts["images_skipped_unknown"] += 1
            return []
        if not kept:
            self.counts["images_skipped_excluded"] += 1
        return kept

    def summary(self) -> Optional[dict]:
        if not self.overridden and not any(self.counts.values()):
            return None
        return {
            "exclude_labels": sorted(self.exclude),
            "unknown_labels": sorted(self.unknown),
            **self.counts,
        }


def _export(
    client: httpx.Client,
    project: str,
    reviewed_only: bool,
    rules: Optional[_LabelRules] = None,
    task: str = "classification",
    raw: Optional[list] = None,
):
    """Page through the export: ``(items with boxes, skipped, pages)``. Metadata only.

    ``skipped`` counts images Studio sent without boxes; images emptied by ``rules``
    are counted there instead. ``raw``, when given, receives every item Studio sent."""
    listed: list[tuple[dict, list]] = []
    skipped = pages = 0
    cursor: Optional[str] = None
    while True:
        if pages >= MAX_PAGES:
            raise StudioError(f"export exceeded {MAX_PAGES} pages")
        params = {"reviewed": "1" if reviewed_only else "0", "limit": PAGE_LIMIT}
        if cursor:
            params["cursor"] = cursor
        body = _json(
            _call(
                client,
                "GET",
                f"/projects/{quote(project, safe='')}/export",
                params=params,
            )
        )
        pages += 1
        items = body.get("items")
        if not isinstance(items, list):
            raise StudioError("export: items is not a list")
        for item in items:
            if (
                not isinstance(item, dict)
                or "image_id" not in item
                or "name" not in item
            ):
                raise StudioError(f"export: malformed item {item!r}")
            if raw is not None:
                raw.append(item)
            boxes = _boxes(item)
            if not boxes:
                skipped += 1
                continue
            if rules is not None:
                boxes = rules.apply(boxes, task)
                if not boxes:
                    continue
            listed.append((item, boxes))
        cursor = body.get("next_cursor")
        if not cursor:
            break
    return listed, skipped, pages


def _fingerprint(raw: list, reviewed_only: bool) -> str:
    """sha256 of what the export listed: changes when Studio adds, removes, renames,
    re-reviews or re-boxes an image. Image bytes are not part of the export."""
    rows = sorted(
        (
            [
                str(i.get("image_id")),
                str(i.get("name")),
                bool(i.get("reviewed")),
                sorted(
                    [str(b.get("id")), str(b.get("label")), b.get("bbox")]
                    for b in (i.get("boxes") or [])
                    if isinstance(b, dict)
                ),
            ]
            for i in raw
        ),
        key=lambda r: (r[0], r[1]),
    )
    payload = json.dumps([bool(reviewed_only), rows], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _put_image(store: LocalObjectStore, image_id, data: bytes) -> tuple[str, int, int]:
    """Validate ``data`` and write it to the object store: ``(sha256, width, height)``."""
    try:
        width, height = _image_size(data)
    except HTTPException as exc:
        raise HTTPException(
            502, f"Studio image {image_id} is unusable: {exc.detail}"
        ) from exc
    sha256 = hashlib.sha256(data).hexdigest()
    key = image_key(sha256)
    if not store.exists(key):
        store.put_bytes(key, data)
    return sha256, width, height


def _fetch(
    request: Request,
    settings,
    project: str,
    reviewed_only: bool,
    task: str,
    rules: Optional[_LabelRules] = None,
):
    """Download, validate and store every image one at a time, holding only one image's
    bytes in memory. Images go to the content-addressed object store only; the dataset
    itself is untouched until every image has arrived, so a Studio failure part-way
    through leaves the dataset as it was."""
    store = LocalObjectStore(settings.objects_dir)
    # (name, sha256, width, height, item for apply_items, review status)
    staged: list[tuple[str, str, int, int, ItemIn, str]] = []
    raw: list = []
    with _client(request, settings) as client:
        listed, skipped, pages = _export(
            client, project, reviewed_only, rules, task, raw
        )
        names = _item_names(listed)
        for name in names:
            _validate_name(name)
        for (item, boxes), name in zip(listed, names):
            state = "confirmed" if item.get("reviewed") else "pending"
            data = _call(
                client, "GET", f"/images/{quote(str(item['image_id']), safe='')}"
            ).content
            sha, width, height = _put_image(store, item["image_id"], data)
            if task == "detection":
                anns = [{"label": lb, "bbox": bb} for _, lb, bb in boxes]
                staged.append(
                    (
                        name,
                        sha,
                        width,
                        height,
                        ItemIn(image=name, annotations=anns),
                        state,
                    )
                )
                continue
            stem = PurePosixPath(name).stem or name
            for bid, label, bbox in boxes:
                crop_name = _crop_name(name, bid)
                crop_sha, cw, ch = _put_image(
                    store, item["image_id"], _crop_png(data, bbox)
                )
                staged.append(
                    (
                        crop_name,
                        crop_sha,
                        cw,
                        ch,
                        ItemIn(
                            image=crop_name, annotations=[{"label": label}], group=stem
                        ),
                        state,
                    )
                )
            del data
    return staged, len(listed), skipped, pages, _fingerprint(raw, reviewed_only)


def _import(request: Request, body: ImportIn) -> dict:
    settings = _settings(request)
    project = body.project.strip()
    _check_project(project)
    rules = _LabelRules(body)
    with connect(settings.database_url) as conn:
        ds = conn.execute(
            "SELECT id, task FROM datasets WHERE id = %s", (body.dataset_id,)
        ).fetchone()
    if ds is None:
        raise HTTPException(404, f"dataset {body.dataset_id} not found")
    items: list[ItemIn] = []
    status: dict[str, str] = {}
    # Objects are stored long before rows point at them; hold OBJECTS_LOCK shared for the
    # whole span so deleting another dataset cannot remove an object this import reuses.
    # Autocommit keeps the lock-holding session out of an idle transaction.
    with connect(settings.database_url) as lock_conn:
        lock_conn.autocommit = True
        lock_conn.execute("SELECT pg_advisory_lock_shared(%s, %s)", OBJECTS_LOCK)
        try:
            staged, imported, skipped, pages, fingerprint = _fetch(
                request, settings, project, body.reviewed_only, ds["task"], rules
            )
            for name, sha, width, height, item, state in staged:
                # Studio is the source of truth for imported images: a changed image
                # replaces the stored one under the same name.
                register_image(
                    settings, ds["id"], name, sha, width, height,
                    replace=True, objects_locked=True, source="studio",
                )
                items.append(item)
                status[name] = state
        finally:
            lock_conn.execute("SELECT pg_advisory_unlock_shared(%s, %s)", OBJECTS_LOCK)
    with connect(settings.database_url) as conn:
        count = apply_items(conn, ds["id"], items, status) if items else 0
        conn.execute(
            "INSERT INTO studio_imports"
            " (dataset_id, project, reviewed_only, export_sha256, names)"
            " VALUES (%s, %s, %s, %s, %s)"
            " ON CONFLICT (dataset_id) DO UPDATE SET project = EXCLUDED.project,"
            " reviewed_only = EXCLUDED.reviewed_only,"
            " export_sha256 = EXCLUDED.export_sha256, names = EXCLUDED.names,"
            " imported_at = now()",
            (ds["id"], project, body.reviewed_only, fingerprint,
             Jsonb(sorted(s[0] for s in staged))),
        )
        result = {
            "imported_images": imported,
            "imported_items": count,
            "skipped": skipped,
            "pages": pages,
        }
        handling = rules.summary()
        if handling is not None:
            result["label_handling"] = handling
        record_audit(
            conn,
            principal_name(request),
            "studio.import",
            f"dataset:{ds['id']}",
            {"project": project, "reviewed_only": body.reviewed_only, **result},
        )
    return result


# --------------------------------------------------------------------------- push


def _manifest(settings, model_id: int) -> tuple[dict, bytes, str, Optional[bytes]]:
    with connect(settings.database_url) as conn:
        row = conn.execute(
            """
            SELECT m.id, m.name, m.pipeline, m.format, m.metrics, m.metadata, m.sha256,
                   m.object_key, m.created_at, d.task, s.label_counts
            FROM model_versions m
            JOIN snapshots s ON s.id = m.snapshot_id
            JOIN datasets d ON d.id = s.dataset_id
            WHERE m.id = %s
            """,
            (model_id,),
        ).fetchone()
    if row is None:
        raise HTTPException(404, f"model {model_id} not found")
    store = LocalObjectStore(settings.objects_dir)
    if not store.exists(row["object_key"]):
        raise HTTPException(
            500, f"artifact for model {model_id} is missing from the object store"
        )
    path = store.path(row["object_key"])
    data = path.read_bytes()
    metadata = row["metadata"] or {}
    labels = metadata.get("labels")
    if not isinstance(labels, list):
        labels = sorted((row["label_counts"] or {}).keys())
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "model_id": row["id"],
        "name": row["name"],
        "pipeline": row["pipeline"],
        "task": row["task"],
        "format": row["format"],
        "labels": labels,
        "metrics": row["metrics"],
        "sha256": hashlib.sha256(data).hexdigest(),
        "created_at": row["created_at"].isoformat(),
        "source": "sparrow-fine-tune",
    }
    if row["task"] == "detection":
        # Studio pairs detector and classifier from these (generic vs species).
        manifest.update(detector_hints(labels))
    engine_onnx = None
    try:
        pkg = build_engine_package(
            row["pipeline"],
            data,
            labels,
            settings.embedder,
            metrics=row["metrics"],
            params=metadata,
        )
    except EngineConversionError as exc:
        manifest["engine_unavailable"] = str(exc)
    else:
        manifest["engine"] = pkg.meta
        # Top-level keys (calibrated_threshold, abstain_label); Studio rejects
        # unknown keys inside ``engine``.
        for key, value in getattr(pkg, "extra", {}).items():
            manifest.setdefault(key, value)
        engine_onnx = pkg.onnx
    return manifest, data, f"model{path.suffix or '.' + row['format']}", engine_onnx


def _bundle(
    manifest: dict,
    artifact: bytes,
    artifact_name: str,
    engine_onnx: Optional[bytes] = None,
) -> bytes:
    buf = io.BytesIO()
    now = time.time()
    members = [(artifact_name, artifact)]
    if engine_onnx is not None:
        members.append(("engine.onnx", engine_onnx))
    members.append(("manifest.json", json.dumps(manifest, indent=2).encode()))
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, payload in members:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = now
            tar.addfile(info, io.BytesIO(payload))
    return buf.getvalue()


def push_model(
    request: Request,
    model_id: int,
    project: str,
    activate: bool = False,
    detector_pipeline_id: Optional[int] = None,
) -> dict:
    """Upload the model to Studio and, with ``activate``, ask Studio to run it.

    ``detector_pipeline_id`` (classifiers only) names the Studio detector pipeline the
    classifier runs on; without it Studio keeps the project's current detector.

    Returns ``{studio_model_id, install}``; ``install`` is Studio's install status,
    ``"unavailable: <reason>"`` when the model has no Engine package, or ``None`` when
    not activating. 409 unconfigured, 502 on any Studio failure.
    """
    settings = _settings(request)
    _check_project(project)
    manifest, artifact, artifact_name, engine_onnx = _manifest(settings, model_id)
    bundle = _bundle(manifest, artifact, artifact_name, engine_onnx)
    try:
        with _client(request, settings) as client:
            body = _json(
                _call(
                    client,
                    "POST",
                    f"/projects/{quote(project, safe='')}/models",
                    files={
                        "manifest": (
                            "manifest.json",
                            json.dumps(manifest),
                            "application/json",
                        ),
                        "bundle": (
                            f"model-{model_id}.tar.gz",
                            bundle,
                            "application/gzip",
                        ),
                    },
                )
            )
    except StudioError as exc:
        raise HTTPException(502, f"Studio push failed: {exc}") from exc
    studio_id = body.get("id")
    if studio_id in (None, ""):
        raise HTTPException(502, "Studio push failed: the response has no model id")
    studio_id = str(studio_id)
    audit(
        request,
        "studio.push",
        f"model:{model_id}",
        {"project": project, "studio_model_id": studio_id},
    )
    install = None
    if activate:
        if engine_onnx is None:
            install = f"unavailable: {manifest['engine_unavailable']}"
        else:
            install = _activate(
                request, settings, project, studio_id, detector_pipeline_id
            )
    return {"studio_model_id": studio_id, "install": install}


def _activate(
    request: Request,
    settings,
    project: str,
    studio_id: str,
    detector_pipeline_id: Optional[int] = None,
) -> str:
    extra = {}
    if detector_pipeline_id is not None:
        extra["json"] = {"detector_pipeline_id": detector_pipeline_id}
    try:
        with _client(request, settings) as client:
            body = _json(
                _call(
                    client,
                    "POST",
                    f"/projects/{quote(project, safe='')}/models/"
                    f"{quote(studio_id, safe='')}/activate",
                    **extra,
                )
            )
    except StudioError as exc:
        raise HTTPException(502, f"Studio activate failed: {exc}") from exc
    status = body.get("install_status")
    if not isinstance(status, str) or not status:
        raise HTTPException(
            502, "Studio activate failed: the response has no install_status"
        )
    return status


# --------------------------------------------------------------------------- routes


class PushIn(BaseModel):
    model_id: int
    project: str = Field(min_length=1, max_length=200)


@router.get("/connectors/studio")
def studio_status(request: Request):
    settings = request.app.state.settings
    out = {
        "configured": settings.studio_configured,
        "url": settings.studio_url if settings.studio_configured else "",
    }
    if _project_ids_mode() == "numeric":
        out["project_ids"] = "numeric"  # lets the UI validate before calling the API
    return out


@router.post("/connectors/studio/import")
async def studio_import(body: ImportIn, request: Request):
    if not body.project.strip():
        raise HTTPException(422, "project must not be empty")
    try:
        return await run_in_threadpool(_import, request, body)
    except StudioError as exc:
        raise HTTPException(502, f"Studio import failed: {exc}") from exc


def _import_status(
    request: Request, project: str, dataset_id: int, reviewed_only: bool
) -> dict:
    settings = _settings(request)
    _check_project(project)
    with connect(settings.database_url) as conn:
        if conn.execute(
            "SELECT 1 FROM datasets WHERE id = %s", (dataset_id,)
        ).fetchone() is None:
            raise HTTPException(404, f"dataset {dataset_id} not found")
        last = conn.execute(
            "SELECT project, reviewed_only, export_sha256, names, imported_at"
            " FROM studio_imports WHERE dataset_id = %s",
            (dataset_id,),
        ).fetchone()
        missing = 0
        if last is not None:
            missing = conn.execute(
                "SELECT count(*) AS n FROM jsonb_array_elements_text(%s::jsonb) AS n(name)"
                " WHERE NOT EXISTS (SELECT 1 FROM dataset_images i"
                " WHERE i.dataset_id = %s AND i.name = n.name)",
                (Jsonb(last["names"]), dataset_id),
            ).fetchone()["n"]
    raw: list = []
    with _client(request, settings) as client:
        listed, _, _ = _export(client, project, reviewed_only, raw=raw)
    same = (
        last is not None
        and last["project"] == project
        and last["reviewed_only"] == reviewed_only
    )
    return {
        "up_to_date": bool(
            same
            and missing == 0
            and last["export_sha256"] == _fingerprint(raw, reviewed_only)
        ),
        "studio_images": len(listed),
        "last_import_at": last["imported_at"] if same else None,
    }


@router.get("/connectors/studio/import-status")
async def studio_import_status(
    request: Request, project: str, dataset_id: int, reviewed_only: bool = True
):
    """Whether ``dataset_id`` already holds everything an import of ``project`` would
    bring: the last import used the same project and filter, Studio's listing has not
    changed since, and none of the imported images were deleted here."""
    try:
        return await run_in_threadpool(
            _import_status, request, project.strip(), dataset_id, reviewed_only
        )
    except StudioError as exc:
        raise HTTPException(502, f"Studio status check failed: {exc}") from exc


class LoadProjectIn(BaseModel):
    project: str = Field(min_length=1, max_length=200)
    reviewed_only: bool = True


def _load_targets(request: Request, project: str) -> tuple[dict, list[dict]]:
    """The workspace standing for ``project`` and its detection and classification
    datasets, created when missing."""
    settings = _settings(request)
    _check_project(project)
    known = {p["id"]: p["name"] for p in _remote_projects(request)}
    if project not in known:
        raise HTTPException(404, f"Studio project {project} not found")
    name = known[project].strip() or f"Studio project {project}"
    with connect(settings.database_url) as conn:
        conn.execute(
            "INSERT INTO workspaces (name, studio_project) VALUES (%s, %s)"
            " ON CONFLICT (studio_project) WHERE studio_project IS NOT NULL DO NOTHING",
            (name, project),
        )
        ws = conn.execute(
            "SELECT id, name FROM workspaces WHERE studio_project = %s", (project,)
        ).fetchone()
        datasets = []
        for task in ("detection", "classification"):
            ds = conn.execute(
                "SELECT d.id, d.name, d.task FROM datasets d"
                " LEFT JOIN studio_imports s ON s.dataset_id = d.id"
                " WHERE d.workspace_id = %s AND d.task = %s"
                " ORDER BY (s.project = %s) IS TRUE DESC, d.id LIMIT 1",
                (ws["id"], task, project),
            ).fetchone()
            if ds is None:
                ds = conn.execute(
                    "INSERT INTO datasets (workspace_id, name, task) VALUES (%s, %s, %s)"
                    " RETURNING id, name, task",
                    (ws["id"], f"{name} ({task})", task),
                ).fetchone()
            datasets.append(ds)
        record_audit(
            conn, principal_name(request), "studio.load_project",
            f"workspace:{ws['id']}", {"project": project},
        )
    return ws, datasets


def _load_project(request: Request, body: LoadProjectIn) -> dict:
    project = body.project.strip()
    ws, datasets = _load_targets(request, project)
    out = []
    for ds in datasets:
        result = _import(
            request,
            ImportIn(project=project, dataset_id=ds["id"], reviewed_only=body.reviewed_only),
        )
        out.append({**ds, **result})
    return {"workspace": ws, "datasets": out}


@router.post("/connectors/studio/load-project")
async def studio_load_project(body: LoadProjectIn, request: Request):
    """Load a Studio project's images into a workspace named after it: one detection
    dataset (boxes) and one classification dataset (one crop per box). Calling it again
    reuses both datasets and re-imports, so it also brings them up to date."""
    if not body.project.strip():
        raise HTTPException(422, "project must not be empty")
    try:
        return await run_in_threadpool(_load_project, request, body)
    except StudioError as exc:
        raise HTTPException(502, f"Studio import failed: {exc}") from exc


@router.post("/connectors/studio/push")
async def studio_push(body: PushIn, request: Request):
    project = body.project.strip()
    if not project:
        raise HTTPException(422, "project must not be empty")
    pushed = await run_in_threadpool(push_model, request, body.model_id, project)
    return {"studio_model_id": pushed["studio_model_id"]}


# --------------------------------------------------------------------------- base models


class BaseModelImportIn(BaseModel):
    project: str = Field(min_length=1, max_length=200)
    studio_model_id: str = Field(min_length=1, max_length=200)


def _remote_projects(request: Request) -> list[dict]:
    """Studio projects as ``{id, name}`` (id as a string), sorted by name."""
    settings = _settings(request)
    try:
        with _client(request, settings) as client:
            body = _json(_call(client, "GET", "/projects"))
    except StudioError as exc:
        raise HTTPException(502, f"Studio project list failed: {exc}") from exc
    projects = body.get("projects")
    if not isinstance(projects, list) or not all(
        isinstance(p, dict)
        and isinstance(p.get("id"), (int, str))
        and not isinstance(p.get("id"), bool)
        and isinstance(p.get("name"), str)
        for p in projects
    ):
        raise HTTPException(502, "Studio project list failed: no valid 'projects' list")
    out = [{"id": str(p["id"]), "name": p["name"]} for p in projects]
    return sorted(out, key=lambda p: (p["name"].lower(), p["id"]))


def _remote_models(request: Request, project: str) -> list[dict]:
    settings = _settings(request)
    _check_project(project)
    try:
        with _client(request, settings) as client:
            body = _json(
                _call(client, "GET", f"/projects/{quote(project, safe='')}/models")
            )
    except StudioError as exc:
        raise HTTPException(502, f"Studio model list failed: {exc}") from exc
    models = body.get("models")
    if not isinstance(models, list) or not all(isinstance(m, dict) for m in models):
        raise HTTPException(502, "Studio model list failed: no 'models' list")
    warm_start = getattr(
        base_models, "WARM_START_PIPELINES", {base_models.WARM_START_PIPELINE}
    )
    return [
        {**m, "importable": m.get("pipeline") in warm_start}
        for m in models
    ]


def _download_bundle(
    request: Request, settings, project: str, studio_id: str
) -> tuple[bytes, Optional[str]]:
    path = (
        f"/projects/{quote(project, safe='')}/models/{quote(studio_id, safe='')}/bundle"
    )
    try:
        with _client(request, settings) as client:
            with client.stream("GET", path) as r:
                if r.status_code >= 400:
                    r.read()
                    raise StudioError(
                        f"GET {path}: Studio answered {r.status_code}: {r.text[:200]}"
                    )
                chunks, size = [], 0
                for chunk in r.iter_bytes():
                    size += len(chunk)
                    if size > base_models.MAX_BUNDLE_BYTES:
                        raise StudioError(
                            f"GET {path}: bundle is larger than "
                            f"{base_models.MAX_BUNDLE_BYTES} bytes"
                        )
                    chunks.append(chunk)
                return b"".join(chunks), r.headers.get("X-Artifact-Sha256")
    except httpx.HTTPError as exc:
        raise StudioError(f"GET {path}: {exc}") from exc


def _import_base_model(request: Request, project: str, studio_id: str):
    """``(row, created)``: pull a model this service pushed back from Studio."""
    settings = _settings(request)
    _check_project(project)
    existing = base_models.find(settings.database_url, project, studio_id)
    if existing is not None:
        return existing, False
    try:
        bundle, header_sha = _download_bundle(request, settings, project, studio_id)
    except StudioError as exc:
        raise HTTPException(502, f"Studio model download failed: {exc}") from exc
    try:
        facts = base_models.inspect_bundle(bundle, header_sha)
    except base_models.NotImportable as exc:
        raise HTTPException(422, f"cannot import Studio model: {exc}") from exc
    except base_models.BundleError as exc:
        raise HTTPException(502, f"Studio model bundle rejected: {exc}") from exc
    row, created = base_models.store(settings, project, studio_id, facts)
    if created:
        audit(
            request,
            "studio.base_model_import",
            f"base_model:{row['id']}",
            {
                "project": project,
                "studio_model_id": studio_id,
                "sha256": row["sha256"],
                "model_arch": row["model_arch"],
            },
        )
    return row, created


@router.get("/connectors/studio/projects")
async def studio_projects(request: Request):
    """Studio projects the UI offers by name; the id is what the other routes take."""
    return {"projects": await run_in_threadpool(_remote_projects, request)}


_INSTALL_STATES = {"not_installed", "pending", "installing", "installed", "failed"}


def _remote_install(request: Request, project: str, studio_id: str) -> dict:
    settings = _settings(request)
    _check_project(project)
    path = f"/projects/{quote(project, safe='')}/models/{quote(studio_id, safe='')}"
    try:
        with _client(request, settings) as client:
            body = _json(_call(client, "GET", path))
    except StudioError as exc:
        raise HTTPException(502, f"Studio install status failed: {exc}") from exc
    status = body.get("install_status")
    if not isinstance(status, str) or not status:
        raise HTTPException(502, "Studio install status failed: no install_status")
    error = body.get("install_error")
    return {
        "studio_model_id": studio_id,
        "install_status": status,
        "install_error": error if isinstance(error, str) else None,
        "finished": status in {"installed", "failed"} or status not in _INSTALL_STATES,
    }


@router.get("/connectors/studio/install")
async def studio_install(request: Request, project: str, studio_model_id: str):
    """Where Studio is with installing a model this service pushed and activated."""
    project = project.strip()
    studio_model_id = studio_model_id.strip()
    if not project or not studio_model_id:
        raise HTTPException(422, "project and studio_model_id must not be empty")
    return await run_in_threadpool(_remote_install, request, project, studio_model_id)


def _remote_detectors(request: Request, project: str) -> list[dict]:
    settings = _settings(request)
    _check_project(project)
    try:
        with _client(request, settings) as client:
            body = _json(
                _call(client, "GET", f"/projects/{quote(project, safe='')}/detectors")
            )
    except StudioError as exc:
        raise HTTPException(502, f"Studio detector list failed: {exc}") from exc
    detectors = body.get("detectors")
    if not isinstance(detectors, list) or not all(
        isinstance(d, dict) and d.get("pipeline_id") not in (None, "")
        for d in detectors
    ):
        raise HTTPException(502, "Studio detector list failed: no 'detectors' list")
    return detectors


@router.get("/connectors/studio/detectors")
async def studio_detectors(request: Request, project: str):
    """Detectors a fine-tuned classifier can run on in the Studio project.

    Each has ``pipeline_id``, ``name``, ``fine_tuned``, ``usable`` (with a ``reason``
    when not) and ``current`` (what the project runs today).
    """
    project = project.strip()
    if not project:
        raise HTTPException(422, "project must not be empty")
    return {"detectors": await run_in_threadpool(_remote_detectors, request, project)}


@router.get("/connectors/studio/models")
async def studio_models(request: Request, project: str):
    """Models in the Studio project; ``importable`` marks warm-start candidates."""
    project = project.strip()
    if not project:
        raise HTTPException(422, "project must not be empty")
    models = await run_in_threadpool(_remote_models, request, project)
    return {"models": models}


@router.post("/connectors/studio/models/import")
async def studio_model_import(body: BaseModelImportIn, request: Request):
    """Import a Studio model as a base model (201 new, 200 already imported)."""
    project = body.project.strip()
    studio_id = body.studio_model_id.strip()
    if not project or not studio_id:
        raise HTTPException(422, "project and studio_model_id must not be empty")
    row, created = await run_in_threadpool(
        _import_base_model, request, project, studio_id
    )
    return JSONResponse(jsonable_encoder(row), status_code=201 if created else 200)


@router.get("/base-models")
def list_base_models(request: Request):
    return {
        "base_models": base_models.list_all(request.app.state.settings.database_url)
    }
