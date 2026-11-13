"""Workspaces, datasets, image upload, annotation import and immutable snapshots."""

from __future__ import annotations

import hashlib
import io
import json
from collections import Counter
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse
from PIL import Image
from psycopg import errors as pg_errors
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field, FiniteFloat
from starlette.concurrency import run_in_threadpool

from .. import assist, camtrapdp, coco, interchange, voc
from ..db import OBJECTS_LOCK, connect
from ..storage import LocalObjectStore
from .security import principal_name, record_audit

router = APIRouter()

MAX_IMAGE_BYTES = 64 * 1024 * 1024
# Decoding costs width*height*channels bytes; a small compressed file can expand hugely.
MAX_IMAGE_PIXELS = 50_000_000
# Pillow opener names accepted on upload. Multi-picture camera JPEGs (MPO) open via "JPEG".
ALLOWED_IMAGE_FORMATS = ("JPEG", "PNG", "BMP", "TIFF", "WEBP", "GIF")
Image.init()  # register every plugin so ``formats=`` can name WEBP, TIFF, ...
_FORBIDDEN_NAME_CHARS = set("/\\") | {chr(c) for c in range(32)}


class WorkspaceIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class DatasetIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    task: Literal["classification", "detection"]
    # Classification only: train on full images with image-level labels instead of box crops.
    whole_image: bool = False


class DatasetPatch(BaseModel):
    whole_image: bool


class AnnotationIn(BaseModel):
    label: str
    # FiniteFloat: NaN/Infinity pass every bounds comparison and Postgres jsonb rejects them.
    bbox: Optional[list[FiniteFloat]] = Field(default=None, min_length=4, max_length=4)


class ItemIn(BaseModel):
    image: str
    annotations: list[AnnotationIn] = Field(default_factory=list)
    group: Optional[str] = None


class AnnotationsIn(BaseModel):
    items: list[ItemIn]


ReviewStatus = Literal["pending", "confirmed", "excluded"]
REVIEW_STATUSES = ("pending", "confirmed", "excluded")


class ItemPatch(BaseModel):
    review_status: Optional[ReviewStatus] = None
    annotations: Optional[list[AnnotationIn]] = None


def _url(request: Request) -> str:
    return request.app.state.settings.database_url


def _dataset_or_404(conn, dataset_id: int) -> dict:
    row = conn.execute("SELECT * FROM datasets WHERE id = %s", (dataset_id,)).fetchone()
    if row is None:
        raise HTTPException(404, f"dataset {dataset_id} not found")
    return row


# --------------------------------------------------------------------------- workspaces


@router.post("/workspaces", status_code=201)
def create_workspace(body: WorkspaceIn, request: Request):
    with connect(_url(request)) as conn:
        return conn.execute(
            "INSERT INTO workspaces (name) VALUES (%s) RETURNING id, name, created_at",
            (body.name,),
        ).fetchone()


@router.get("/workspaces")
def list_workspaces(request: Request):
    with connect(_url(request)) as conn:
        rows = conn.execute(
            "SELECT id, name, created_at, studio_project FROM workspaces ORDER BY id"
        ).fetchall()
    return {"workspaces": rows}


@router.delete("/workspaces/{workspace_id}")
def delete_workspace(workspace_id: int, request: Request, include_models: bool = False):
    """Permanently remove a workspace and all its datasets. Data that trained a job is only
    removed with ``include_models=true``, which also removes those jobs and models."""
    return _remove(request, "workspace", workspace_id, include_models)


@router.get("/workspaces/{workspace_id}/deletion")
def workspace_deletion(workspace_id: int, request: Request):
    """What deleting the workspace with ``include_models=true`` would remove."""
    return _preview(request, "workspace", workspace_id)


# --------------------------------------------------------------------------- datasets


@router.post("/workspaces/{workspace_id}/datasets", status_code=201)
def create_dataset(workspace_id: int, body: DatasetIn, request: Request):
    with connect(_url(request)) as conn:
        if (
            conn.execute(
                "SELECT 1 FROM workspaces WHERE id = %s", (workspace_id,)
            ).fetchone()
            is None
        ):
            raise HTTPException(404, f"workspace {workspace_id} not found")
        if body.whole_image and body.task != "classification":
            raise HTTPException(422, "whole-image training applies to classification datasets only")
        return conn.execute(
            "INSERT INTO datasets (workspace_id, name, task, whole_image)"
            " VALUES (%s, %s, %s, %s) RETURNING *",
            (workspace_id, body.name, body.task, body.whole_image),
        ).fetchone()


@router.patch("/datasets/{dataset_id}")
def patch_dataset(dataset_id: int, body: DatasetPatch, request: Request):
    """Switch a classification dataset between box-crop and whole-image training. Turning
    whole-image on folds every item's boxes into its image-level labels and deletes the boxes."""
    with connect(_url(request)) as conn:
        dataset = conn.execute(
            "SELECT * FROM datasets WHERE id = %s FOR UPDATE", (dataset_id,)
        ).fetchone()
        if dataset is None:
            raise HTTPException(404, f"dataset {dataset_id} not found")
        if body.whole_image and dataset["task"] != "classification":
            raise HTTPException(422, "whole-image training applies to classification datasets only")
        removed = 0
        if body.whole_image and not dataset["whole_image"]:
            boxed = conn.execute(
                "SELECT DISTINCT a.item_id FROM annotations a"
                " JOIN dataset_items it ON it.id = a.item_id"
                " WHERE it.dataset_id = %s AND a.bbox IS NOT NULL",
                (dataset_id,),
            ).fetchall()
            for row in boxed:
                labels = [
                    r["label"]
                    for r in conn.execute(
                        "SELECT label FROM annotations WHERE item_id = %s ORDER BY id",
                        (row["item_id"],),
                    ).fetchall()
                ]
                removed += conn.execute(
                    "SELECT count(*) AS n FROM annotations WHERE item_id = %s AND bbox IS NOT NULL",
                    (row["item_id"],),
                ).fetchone()["n"]
                conn.execute("DELETE FROM annotations WHERE item_id = %s", (row["item_id"],))
                with conn.cursor() as cur:
                    cur.executemany(
                        "INSERT INTO annotations (item_id, label, bbox) VALUES (%s, %s, NULL)",
                        [(row["item_id"], label) for label in dict.fromkeys(labels)],
                    )
                conn.execute(
                    "UPDATE dataset_items SET updated_at = now() WHERE id = %s", (row["item_id"],)
                )
        updated = conn.execute(
            "UPDATE datasets SET whole_image = %s WHERE id = %s RETURNING *",
            (body.whole_image, dataset_id),
        ).fetchone()
        record_audit(
            conn,
            principal_name(request),
            "dataset.whole_image",
            f"dataset:{dataset_id}",
            {"whole_image": body.whole_image, "boxes_removed": removed},
        )
    return {**updated, "boxes_removed": removed}


@router.get("/workspaces/{workspace_id}/datasets")
def list_datasets(workspace_id: int, request: Request):
    with connect(_url(request)) as conn:
        if (
            conn.execute(
                "SELECT 1 FROM workspaces WHERE id = %s", (workspace_id,)
            ).fetchone()
            is None
        ):
            raise HTTPException(404, f"workspace {workspace_id} not found")
        rows = conn.execute(
            "SELECT * FROM datasets WHERE workspace_id = %s ORDER BY id",
            (workspace_id,),
        ).fetchall()
    return {"datasets": rows}


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: int, request: Request):
    with connect(_url(request)) as conn:
        dataset = _dataset_or_404(conn, dataset_id)
        counts = dict.fromkeys(REVIEW_STATUSES, 0)
        for row in conn.execute(
            "SELECT review_status, count(*) AS n FROM dataset_items"
            " WHERE dataset_id = %s GROUP BY review_status",
            (dataset_id,),
        ).fetchall():
            counts[row["review_status"]] = row["n"]
    return {**dataset, "review_counts": counts}


@router.delete("/datasets/{dataset_id}")
def delete_dataset(dataset_id: int, request: Request, include_models: bool = False):
    """Permanently remove a dataset, its snapshots and its unshared image objects. Once a
    job trained on it, only with ``include_models=true``, which also removes those jobs,
    their models, model files and deployment records."""
    return _remove(request, "dataset", dataset_id, include_models)


@router.get("/datasets/{dataset_id}/deletion")
def dataset_deletion(dataset_id: int, request: Request):
    """What deleting the dataset with ``include_models=true`` would remove."""
    return _preview(request, "dataset", dataset_id)


def _preview(request: Request, kind: str, target_id: int) -> dict:
    with connect(_url(request)) as conn:
        name, datasets = _targets(conn, kind, target_id, lock=False)
        ids = [d["id"] for d in datasets]
        jobs = _jobs_of(conn, ids, lock=False)
        counts = conn.execute(
            "SELECT (SELECT count(*) FROM dataset_images WHERE dataset_id = ANY(%s)) AS images,"
            " (SELECT count(*) FROM dataset_items WHERE dataset_id = ANY(%s)) AS items",
            (ids, ids),
        ).fetchone()
        models = _models_of(conn, [j["id"] for j in jobs], lock=False)
    return {
        "kind": kind,
        "name": name,
        "datasets": len(ids),
        "dataset_names": [d["name"] for d in datasets],
        "images": counts["images"],
        "items": counts["items"],
        "jobs": len(jobs),
        "active_jobs": [j["id"] for j in jobs if j["state"] in ACTIVE_JOB_STATES],
        "models": [
            {"id": m["id"], "name": m["name"], "in_use": m["in_use"]} for m in models
        ],
    }


ACTIVE_JOB_STATES = ("queued", "running")


def _targets(conn, kind: str, target_id: int, lock: bool) -> tuple[str, list]:
    """The workspace or dataset name and the dataset rows deleting it removes."""
    suffix = " FOR UPDATE" if lock else ""
    if kind == "workspace":
        ws = conn.execute(
            "SELECT id, name FROM workspaces WHERE id = %s" + suffix, (target_id,)
        ).fetchone()
        if ws is None:
            raise HTTPException(404, f"workspace {target_id} not found")
        datasets = conn.execute(
            "SELECT id, name FROM datasets WHERE workspace_id = %s ORDER BY id" + suffix,
            (target_id,),
        ).fetchall()
        return ws["name"], datasets
    row = conn.execute(
        "SELECT id, name FROM datasets WHERE id = %s" + suffix, (target_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(404, f"dataset {target_id} not found")
    return row["name"], [row]


def _jobs_of(conn, dataset_ids: list, lock: bool) -> list:
    return conn.execute(
        "SELECT j.id, j.state, s.dataset_id FROM jobs j"
        " JOIN snapshots s ON s.id = j.snapshot_id WHERE s.dataset_id = ANY(%s)"
        " ORDER BY j.id" + (" FOR UPDATE OF j" if lock else ""),
        (dataset_ids,),
    ).fetchall()


def _models_of(conn, job_ids: list, lock: bool) -> list:
    models = conn.execute(
        "SELECT id, name, object_key FROM model_versions WHERE job_id = ANY(%s) ORDER BY id"
        + (" FOR UPDATE" if lock else ""),
        (job_ids,),
    ).fetchall()
    in_use: dict = {}
    for d in conn.execute(
        "SELECT model_id, consumer, project FROM deployments"
        " WHERE model_id = ANY(%s) AND deactivated_at IS NULL ORDER BY consumer, project",
        ([m["id"] for m in models],),
    ).fetchall():
        in_use.setdefault(d["model_id"], []).append(
            {"consumer": d["consumer"], "project": d["project"]}
        )
    return [{**m, "in_use": in_use.get(m["id"], [])} for m in models]


def _remove(request: Request, kind: str, target_id: int, include_models: bool = False) -> dict:
    settings = request.app.state.settings
    with connect(settings.database_url) as conn:
        conn.autocommit = True
        try:
            conn.execute("SET lock_timeout = '15s'")
            conn.execute("SELECT pg_advisory_lock(%s, %s)", OBJECTS_LOCK)
        except (pg_errors.LockNotAvailable, pg_errors.QueryCanceled) as exc:
            raise HTTPException(
                409, "images are being added right now; try the deletion again shortly"
            ) from exc
        try:
            try:
                with conn.transaction():
                    deleted, keys = _delete_rows(conn, kind, target_id, include_models)
                    record_audit(
                        conn,
                        principal_name(request),
                        f"{kind}.delete",
                        f"{kind}:{target_id}",
                        deleted,
                    )
            except pg_errors.ForeignKeyViolation as exc:
                # A job or dataset created after the checks below, before the delete.
                raise HTTPException(
                    409, f"{kind} {target_id} came into use while deleting it; nothing was removed"
                ) from exc
            # Still under the exclusive lock, so no writer can start using these objects.
            store = LocalObjectStore(settings.objects_dir)
            unused = _unreferenced(conn, keys)
            for key in unused:
                store.delete(key)
            deleted["objects"] = len(unused)
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s, %s)", OBJECTS_LOCK)
    return {"deleted": deleted}


def _delete_rows(conn, kind: str, target_id: int, include_models: bool) -> tuple[dict, set]:
    name, datasets = _targets(conn, kind, target_id, lock=True)
    ids = [d["id"] for d in datasets]
    jobs = _jobs_of(conn, ids, lock=True)
    active = [j["id"] for j in jobs if j["state"] in ACTIVE_JOB_STATES]
    if active:
        raise HTTPException(
            409,
            f"job(s) {', '.join(str(j) for j in active)} of this {kind} are queued or "
            "running. Cancel them first, then delete again.",
        )
    if jobs and not include_models:
        names = {d["id"]: d["name"] for d in datasets}
        per: dict = {}
        for j in jobs:
            per.setdefault(j["dataset_id"], []).append(str(j["id"]))
        used = "; ".join(
            f"dataset {names[d]!r} trained job(s) {', '.join(js)}" for d, js in per.items()
        )
        raise HTTPException(
            409,
            f"{used}. Deleting it also deletes those jobs and their models; "
            "repeat the request with include_models=true to do that.",
        )
    job_ids = [j["id"] for j in jobs]
    models = _models_of(conn, job_ids, lock=True)
    model_ids = [m["id"] for m in models]
    counts: dict = {"datasets": len(ids)}
    if job_ids:
        counts["deployments"] = conn.execute(
            "DELETE FROM deployments WHERE model_id = ANY(%s)", (model_ids,)
        ).rowcount
        counts["models"] = conn.execute(
            "DELETE FROM model_versions WHERE id = ANY(%s)", (model_ids,)
        ).rowcount
        counts["model_names"] = [m["name"] for m in models]
        conn.execute("DELETE FROM job_events WHERE job_id = ANY(%s)", (job_ids,))
        counts["jobs"] = conn.execute(
            "DELETE FROM jobs WHERE id = ANY(%s)", (job_ids,)
        ).rowcount
    keys = {
        r["object_key"]
        for r in conn.execute(
            "SELECT object_key FROM dataset_images WHERE dataset_id = ANY(%s)"
            " UNION SELECT si.object_key FROM snapshot_items si"
            " JOIN snapshots s ON s.id = si.snapshot_id WHERE s.dataset_id = ANY(%s)",
            (ids, ids),
        ).fetchall()
    } | {m["object_key"] for m in models}
    for table, sql in (
        (
            "annotations",
            "DELETE FROM annotations a USING dataset_items it"
            " WHERE a.item_id = it.id AND it.dataset_id = ANY(%s)",
        ),
        ("items", "DELETE FROM dataset_items WHERE dataset_id = ANY(%s)"),
        (
            "snapshot_items",
            "DELETE FROM snapshot_items si USING snapshots s"
            " WHERE si.snapshot_id = s.id AND s.dataset_id = ANY(%s)",
        ),
        ("snapshots", "DELETE FROM snapshots WHERE dataset_id = ANY(%s)"),
        ("images", "DELETE FROM dataset_images WHERE dataset_id = ANY(%s)"),
    ):
        counts[table] = conn.execute(sql, (ids,)).rowcount
    conn.execute("DELETE FROM datasets WHERE id = ANY(%s)", (ids,))
    if kind == "workspace":
        conn.execute("DELETE FROM workspaces WHERE id = %s", (target_id,))
    counts[kind] = name
    return counts, keys


def _unreferenced(conn, keys: set) -> list:
    """The keys in ``keys`` no remaining row points at."""
    if not keys:
        return []
    rows = conn.execute(
        "SELECT k FROM unnest(%s::text[]) AS k"
        " EXCEPT SELECT object_key FROM dataset_images"
        " EXCEPT SELECT object_key FROM snapshot_items"
        " EXCEPT SELECT object_key FROM model_versions"
        " EXCEPT SELECT object_key FROM base_models",
        (sorted(keys),),
    ).fetchall()
    return sorted(r["k"] for r in rows)


# --------------------------------------------------------------------------- images


def _validate_name(name: str) -> None:
    if (
        not name
        or len(name) > 255
        or name in (".", "..")
        or set(name) & _FORBIDDEN_NAME_CHARS
    ):
        raise HTTPException(422, f"invalid image name: {name!r}")


def _image_size(data: bytes) -> tuple[int, int]:
    try:
        with Image.open(io.BytesIO(data), formats=ALLOWED_IMAGE_FORMATS) as im:
            im.verify()
            width, height = im.size
    except Exception as exc:  # Pillow raises many types for bad input
        raise HTTPException(
            422,
            "body is not a readable image in a supported format "
            f"({', '.join(ALLOWED_IMAGE_FORMATS)}): {exc}",
        ) from exc
    if width * height > MAX_IMAGE_PIXELS:
        raise HTTPException(
            422,
            f"image has {width * height} pixels; the limit is {MAX_IMAGE_PIXELS} pixels",
        )
    try:
        # verify() checks structure only; load() decodes the pixels, catching truncation.
        with Image.open(io.BytesIO(data), formats=ALLOWED_IMAGE_FORMATS) as im:
            im.load()
            return im.size
    except Exception as exc:  # Pillow raises many types for bad input
        raise HTTPException(
            422,
            "body is not a readable image in a supported format "
            f"({', '.join(ALLOWED_IMAGE_FORMATS)}): {exc}",
        ) from exc


def image_key(sha256: str) -> str:
    return f"images/{sha256[:2]}/{sha256}"


def _store_image(
    settings, dataset_id: int, name: str, data: bytes, replace: bool = False
) -> dict:
    """Store ``data`` as image ``name``. Identical bytes are idempotent; different bytes
    under an existing name are refused with 409 unless ``replace`` is set."""
    width, height = _image_size(data)
    sha256 = hashlib.sha256(data).hexdigest()
    return register_image(
        settings, dataset_id, name, sha256, width, height, replace=replace, data=data
    )


def register_image(
    settings,
    dataset_id: int,
    name: str,
    sha256: str,
    width: int,
    height: int,
    replace: bool = False,
    data: Optional[bytes] = None,
    objects_locked: bool = False,
    source: str = "upload",
) -> dict:
    """Point image ``name`` at the object for ``sha256``. With ``data`` the object is
    written after the checks pass; without it, the caller has already stored it.

    ``objects_locked`` means the caller already holds ``OBJECTS_LOCK`` shared, as it must
    when it stored the object itself, so a dataset deletion cannot remove it in between.

    ``source`` is ``"upload"`` or ``"studio"``. A Studio import marks an existing row as
    ``studio``; a manual re-upload of the same name keeps whatever source the row had."""
    if source not in ("upload", "studio"):
        raise ValueError(f"unknown image source {source!r}")
    key = image_key(sha256)
    with connect(settings.database_url) as conn:
        # Taken before any row lock, the same order as dataset deletion.
        if not objects_locked:
            conn.execute("SELECT pg_advisory_xact_lock_shared(%s, %s)", OBJECTS_LOCK)
        # FOR SHARE conflicts with the FOR UPDATE lock taken by annotation import and
        # snapshots, so the size check below cannot race with new bboxes being validated.
        if (
            conn.execute(
                "SELECT 1 FROM datasets WHERE id = %s FOR SHARE", (dataset_id,)
            ).fetchone()
            is None
        ):
            raise HTTPException(404, f"dataset {dataset_id} not found")
        existing = conn.execute(
            "SELECT id, sha256, width, height FROM dataset_images"
            " WHERE dataset_id = %s AND name = %s FOR UPDATE",
            (dataset_id, name),
        ).fetchone()
        if existing is not None and existing["sha256"] != sha256 and not replace:
            raise HTTPException(
                409,
                f"image {name!r} already exists in dataset {dataset_id} with different "
                "content; re-send with ?replace=true to overwrite it, or use a new name.",
            )
        if existing is not None and (existing["width"], existing["height"]) != (
            width,
            height,
        ):
            has_boxes = conn.execute(
                "SELECT 1 FROM annotations a JOIN dataset_items it ON it.id = a.item_id"
                " WHERE it.image_id = %s AND a.bbox IS NOT NULL LIMIT 1",
                (existing["id"],),
            ).fetchone()
            if has_boxes is not None:
                raise HTTPException(
                    409,
                    f"image {name!r} has bounding-box annotations for a "
                    f"{existing['width']}x{existing['height']} image; re-uploading it as "
                    f"{width}x{height} would invalidate them. Clear its annotations first "
                    "(import the item with an empty annotation list) or use a new name.",
                )
        store = LocalObjectStore(settings.objects_dir)
        if not store.exists(key):
            if data is None:
                raise RuntimeError(f"object {key} for image {name!r} was not stored")
            store.put_bytes(key, data)
        # The WHERE re-checks the 409 rule: a concurrent first upload of this name with
        # other bytes may have inserted the row after the FOR UPDATE above found none.
        row = conn.execute(
            """
            INSERT INTO dataset_images (dataset_id, name, object_key, sha256, width, height, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (dataset_id, name) DO UPDATE
            SET object_key = EXCLUDED.object_key, sha256 = EXCLUDED.sha256,
                width = EXCLUDED.width, height = EXCLUDED.height,
                source = CASE WHEN EXCLUDED.source = 'studio' THEN 'studio'
                              ELSE dataset_images.source END
            WHERE dataset_images.sha256 = EXCLUDED.sha256 OR %s
            RETURNING id, name, sha256, width, height, source
            """,
            (dataset_id, name, key, sha256, width, height, source, replace),
        ).fetchone()
        if row is None:
            raise HTTPException(
                409,
                f"image {name!r} already exists in dataset {dataset_id} with different "
                "content; re-send with ?replace=true to overwrite it, or use a new name.",
            )
        return row


@router.put("/datasets/{dataset_id}/images/{name}")
async def upload_image(
    dataset_id: int, name: str, request: Request, replace: bool = False
):
    _validate_name(name)
    data = await _read_capped_body(request, MAX_IMAGE_BYTES)
    return await run_in_threadpool(
        _store_image, request.app.state.settings, dataset_id, name, data, replace
    )


async def _read_capped_body(request: Request, cap: int, what: str = "image") -> bytes:
    """Read the body without ever buffering more than ``cap`` bytes.

    An oversized Content-Length is refused before any of the body is read. A body without
    one (chunked transfer) is streamed, and any body is cut off with 413 as soon as the
    running total exceeds the cap.
    """
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            length = int(declared)
        except ValueError:
            raise HTTPException(400, f"invalid Content-Length: {declared!r}") from None
        if length < 0:
            raise HTTPException(400, f"invalid Content-Length: {declared!r}")
        if length > cap:
            raise HTTPException(413, f"{what} larger than {cap} bytes")
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > cap:
            raise HTTPException(413, f"{what} larger than {cap} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


@router.get("/datasets/{dataset_id}/images")
def list_images(dataset_id: int, request: Request):
    with connect(_url(request)) as conn:
        _dataset_or_404(conn, dataset_id)
        rows = conn.execute(
            "SELECT id, name, sha256, width, height, source FROM dataset_images"
            " WHERE dataset_id = %s ORDER BY name",
            (dataset_id,),
        ).fetchall()
    return {"images": rows}


class MoveImagesIn(BaseModel):
    to_dataset_id: int
    names: list[str] = Field(min_length=1, max_length=100_000)


def _name_list(names) -> str:
    shown = ", ".join(repr(n) for n in names[:5])
    return shown + (f" and {len(names) - 5} more" if len(names) > 5 else "")


@router.post("/datasets/{dataset_id}/images/move")
def move_images(dataset_id: int, body: MoveImagesIn, request: Request):
    """Move unlabelled images to another dataset of the same workspace, for example images
    uploaded to the detection dataset whose labels are species without boxes. Images that
    already carry labels are refused so no review work is dropped. All or nothing."""
    names = list(dict.fromkeys(body.names))
    target_id = body.to_dataset_id
    if target_id == dataset_id:
        raise HTTPException(422, "the images are already in this dataset")
    with connect(_url(request)) as conn:
        # Same lock order as upload: the objects lock, then the dataset rows (by id).
        conn.execute("SELECT pg_advisory_xact_lock_shared(%s, %s)", OBJECTS_LOCK)
        rows = {
            r["id"]: r
            for r in conn.execute(
                "SELECT id, name, workspace_id FROM datasets WHERE id = ANY(%s)"
                " ORDER BY id FOR UPDATE",
                ([dataset_id, target_id],),
            ).fetchall()
        }
        for wanted in (dataset_id, target_id):
            if wanted not in rows:
                raise HTTPException(404, f"dataset {wanted} not found")
        source, target = rows[dataset_id], rows[target_id]
        if source["workspace_id"] != target["workspace_id"]:
            raise HTTPException(422, "images can only move between datasets of the same workspace")
        found = {
            r["name"]: r
            for r in conn.execute(
                "SELECT id, name, object_key, sha256, width, height, source FROM dataset_images"
                " WHERE dataset_id = %s AND name = ANY(%s)",
                (dataset_id, names),
            ).fetchall()
        }
        missing = [n for n in names if n not in found]
        if missing:
            raise HTTPException(
                422, f"not in dataset {source['name']!r}: {_name_list(missing)}"
            )
        labelled = [
            r["name"]
            for r in conn.execute(
                "SELECT di.name FROM dataset_items it JOIN dataset_images di"
                " ON di.id = it.image_id WHERE it.image_id = ANY(%s) ORDER BY di.name",
                ([r["id"] for r in found.values()],),
            ).fetchall()
        ]
        if labelled:
            raise HTTPException(
                409,
                f"{len(labelled)} image(s) already have labels in {source['name']!r} and were "
                f"not moved: {_name_list(labelled)}. Nothing was moved.",
            )
        existing = {
            r["name"]: r["sha256"]
            for r in conn.execute(
                "SELECT name, sha256 FROM dataset_images WHERE dataset_id = %s AND name = ANY(%s)",
                (target_id, names),
            ).fetchall()
        }
        clash = sorted(n for n, sha in existing.items() if sha != found[n]["sha256"])
        if clash:
            raise HTTPException(
                409,
                f"{target['name']!r} already has different images named {_name_list(clash)}. "
                "Nothing was moved.",
            )
        moved = 0
        for name in names:
            if name in existing:
                continue
            img = found[name]
            conn.execute(
                "INSERT INTO dataset_images"
                " (dataset_id, name, object_key, sha256, width, height, source)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (target_id, name, img["object_key"], img["sha256"], img["width"],
                 img["height"], img["source"]),
            )
            moved += 1
        conn.execute(
            "DELETE FROM dataset_images WHERE id = ANY(%s)",
            ([r["id"] for r in found.values()],),
        )
        record_audit(
            conn,
            principal_name(request),
            "dataset.move_images",
            f"dataset:{dataset_id}",
            {"to_dataset": target_id, "moved": moved, "already_there": len(existing)},
        )
    return {"moved": moved, "already_there": len(existing)}


_MEDIA_TYPES = {
    "JPEG": "image/jpeg",
    "MPO": "image/jpeg",
    "PNG": "image/png",
    "BMP": "image/bmp",
    "TIFF": "image/tiff",
    "WEBP": "image/webp",
    "GIF": "image/gif",
}


@router.get("/datasets/{dataset_id}/images/{name}/content")
def image_content(dataset_id: int, name: str, request: Request):
    with connect(_url(request)) as conn:
        _dataset_or_404(conn, dataset_id)
        row = conn.execute(
            "SELECT object_key, sha256 FROM dataset_images WHERE dataset_id = %s AND name = %s",
            (dataset_id, name),
        ).fetchone()
    if row is None:
        raise HTTPException(404, f"image {name!r} not found in dataset {dataset_id}")
    # The name can be re-pointed at new bytes, so validate by content hash on every use.
    etag = f'"{row["sha256"]}"'
    cache = {"Cache-Control": "private, no-cache", "ETag": etag}
    if etag in _if_none_match(request.headers.get("if-none-match")):
        return Response(status_code=304, headers=cache)
    store = LocalObjectStore(request.app.state.settings.objects_dir)
    if not store.exists(row["object_key"]):
        raise HTTPException(500, f"image {name!r} is missing from the object store")
    path = store.path(row["object_key"])
    with Image.open(path) as im:
        media_type = _MEDIA_TYPES.get(im.format or "", "application/octet-stream")
    return FileResponse(path, media_type=media_type, headers=cache)


def _if_none_match(header: Optional[str]) -> set[str]:
    if not header:
        return set()
    tags = {t.strip() for t in header.split(",")}
    return {t[2:] if t.startswith("W/") else t for t in tags}


# --------------------------------------------------------------------------- annotations


_MAX_ERROR_LINES = 5


def _import_error_detail(
    errors: list[str], items: list["ItemIn"], images: dict, task: str
) -> dict:
    """One readable message plus a few example lines, instead of one line per image."""
    unboxed_images = sorted(
        {
            item.image
            for item in items
            if task == "detection"
            and item.image in images
            and any(a.bbox is None for a in item.annotations)
        }
    )
    unboxed = len(unboxed_images)
    if unboxed:
        message = (
            f"{unboxed} of {len(items)} images have a label without a box. A detection "
            "dataset needs a box for every label. This looks like a species-only label "
            "file: import it into a classification dataset instead, or upload the images "
            "and tick \"Pre-draw boxes with a detector\" to draw boxes first."
        )
        others = len(errors) - unboxed
        if others > 0:
            message += f" {others} other problem(s) were also found."
    else:
        message = f"{len(errors)} problem(s) found; nothing was imported."
        if len(errors) > _MAX_ERROR_LINES:
            message += f" First {_MAX_ERROR_LINES}: " + "; ".join(errors[:_MAX_ERROR_LINES])
        else:
            message += " " + "; ".join(errors)
    detail = {
        "message": message,
        "error_count": len(errors),
        "errors": errors[:_MAX_ERROR_LINES],
    }
    if unboxed:
        # Lets the web app offer to move these images to a classification dataset.
        detail["reason"] = "labels_without_boxes"
        detail["unboxed_images"] = unboxed_images
    return detail


def _item_errors(
    index: int, item: ItemIn, image: Optional[dict], task: str
) -> list[str]:
    prefix = f"items[{index}]"
    if image is None:
        return [f"{prefix}: unknown image {item.image!r}"]
    errors = []
    for j, ann in enumerate(item.annotations):
        where = f"{prefix}.annotations[{j}]"
        if not ann.label.strip():
            errors.append(f"{where}: label must not be empty")
        if ann.bbox is None:
            if task == "detection":
                errors.append(f"{where}: detection datasets require a bbox")
            continue
        x, y, w, h = ann.bbox
        if (
            x < 0
            or y < 0
            or w <= 0
            or h <= 0
            or x + w > image["width"]
            or y + h > image["height"]
        ):
            errors.append(
                f"{where}: bbox {ann.bbox} is outside the {image['width']}x{image['height']} image"
            )
    return errors


def _for_dataset(dataset: dict, anns: list) -> list:
    """In a whole-image dataset, boxes become image-level labels (one per distinct label)."""
    if not dataset.get("whole_image"):
        return anns
    labels = dict.fromkeys(a.label.strip() for a in anns)
    return [AnnotationIn(label=label) for label in labels]


def apply_items(
    conn, dataset_id: int, items: list[ItemIn], review_status: Optional[dict] = None
) -> int:
    """Validate and upsert annotated items in one transaction (422 on any error).

    ``review_status`` maps image name -> status for new items; an existing item keeps its
    status unless it is ``pending`` and the mapping says ``confirmed``.
    """
    dataset = conn.execute(
        "SELECT * FROM datasets WHERE id = %s FOR UPDATE", (dataset_id,)
    ).fetchone()
    if dataset is None:
        raise HTTPException(404, f"dataset {dataset_id} not found")
    images = {
        r["name"]: r
        for r in conn.execute(
            "SELECT id, name, width, height FROM dataset_images WHERE dataset_id = %s",
            (dataset_id,),
        ).fetchall()
    }
    errors: list[str] = []
    seen: set[str] = set()
    for i, item in enumerate(items):
        if item.image in seen:
            errors.append(f"items[{i}]: image {item.image!r} appears more than once")
        seen.add(item.image)
        errors.extend(_item_errors(i, item, images.get(item.image), dataset["task"]))
    if errors:
        raise HTTPException(422, _import_error_detail(errors, items, images, dataset["task"]))

    review_status = review_status or {}
    for item in items:
        status = review_status.get(item.image, "pending")
        item_id = conn.execute(
            """
            INSERT INTO dataset_items (dataset_id, image_id, group_name, review_status)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (image_id) DO UPDATE SET group_name = EXCLUDED.group_name,
                review_status = CASE
                    WHEN dataset_items.review_status = 'pending'
                         AND EXCLUDED.review_status = 'confirmed' THEN 'confirmed'
                    ELSE dataset_items.review_status END,
                updated_at = now()
            RETURNING id
            """,
            (dataset_id, images[item.image]["id"], item.group, status),
        ).fetchone()["id"]
        conn.execute("DELETE FROM annotations WHERE item_id = %s", (item_id,))
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO annotations (item_id, label, bbox) VALUES (%s, %s, %s)",
                [(item_id, a.label.strip(), a.bbox) for a in _for_dataset(dataset, item.annotations)],
            )
    return len(items)


@router.post("/datasets/{dataset_id}/annotations")
def import_annotations(dataset_id: int, body: AnnotationsIn, request: Request):
    with connect(_url(request)) as conn:
        return {"imported": apply_items(conn, dataset_id, body.items)}


# --------------------------------------------------------------------------- review

_ITEM_SQL = """
SELECT i.name AS image, i.width, i.height, it.group_name AS "group", it.review_status,
       coalesce(jsonb_agg(jsonb_build_object('label', a.label, 'bbox', a.bbox) ORDER BY a.id)
                FILTER (WHERE a.id IS NOT NULL), '[]'::jsonb) AS annotations
FROM dataset_items it
JOIN dataset_images i ON i.id = it.image_id
LEFT JOIN annotations a ON a.item_id = it.id
WHERE it.dataset_id = %(dataset)s
  AND (%(status)s::text IS NULL OR it.review_status = %(status)s)
  AND (%(image)s::text IS NULL OR i.name = %(image)s)
GROUP BY it.id, i.id
ORDER BY i.name
"""


@router.get("/datasets/{dataset_id}/items")
def list_items(dataset_id: int, request: Request, status: Optional[str] = None):
    if status is not None and status not in REVIEW_STATUSES:
        raise HTTPException(
            422, f"status must be one of {', '.join(REVIEW_STATUSES)}, not {status!r}"
        )
    with connect(_url(request)) as conn:
        _dataset_or_404(conn, dataset_id)
        rows = conn.execute(
            _ITEM_SQL, {"dataset": dataset_id, "status": status, "image": None}
        ).fetchall()
    return {"items": rows}


@router.patch("/datasets/{dataset_id}/items/{image}")
def patch_item(dataset_id: int, image: str, body: ItemPatch, request: Request):
    if body.review_status is None and body.annotations is None:
        raise HTTPException(422, "provide review_status, annotations, or both")
    with connect(_url(request)) as conn:
        dataset = conn.execute(
            "SELECT * FROM datasets WHERE id = %s FOR UPDATE", (dataset_id,)
        ).fetchone()
        if dataset is None:
            raise HTTPException(404, f"dataset {dataset_id} not found")
        item = conn.execute(
            "SELECT it.id, i.width, i.height, i.name FROM dataset_items it"
            " JOIN dataset_images i ON i.id = it.image_id"
            " WHERE it.dataset_id = %s AND i.name = %s FOR UPDATE OF it",
            (dataset_id, image),
        ).fetchone()
        if item is None:
            raise HTTPException(404, f"item {image!r} not found in dataset {dataset_id}")
        if body.annotations is not None:
            errors = _item_errors(
                0,
                ItemIn(image=image, annotations=body.annotations),
                item,
                dataset["task"],
            )
            if errors:
                raise HTTPException(422, errors)
            conn.execute("DELETE FROM annotations WHERE item_id = %s", (item["id"],))
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO annotations (item_id, label, bbox) VALUES (%s, %s, %s)",
                    [
                        (item["id"], a.label.strip(), a.bbox)
                        for a in _for_dataset(dataset, body.annotations)
                    ],
                )
        if body.review_status is not None:
            conn.execute(
                "UPDATE dataset_items SET review_status = %s WHERE id = %s",
                (body.review_status, item["id"]),
            )
        conn.execute(
            "UPDATE dataset_items SET updated_at = now() WHERE id = %s", (item["id"],)
        )
        return conn.execute(
            _ITEM_SQL, {"dataset": dataset_id, "status": None, "image": image}
        ).fetchone()


# --------------------------------------------------------------------------- snapshots


def _canonical(row: dict) -> dict:
    anns = [
        {
            "label": a["label"],
            "bbox": None if a["bbox"] is None else [float(v) for v in a["bbox"]],
        }
        for a in row["annotations"]
    ]
    anns.sort(key=lambda a: (a["label"], a["bbox"] or []))
    return {
        "sha256": row["sha256"],
        "width": row["width"],
        "height": row["height"],
        "annotations": anns,
        "group": row["group_name"],
    }


def _dumps(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


@router.post("/datasets/{dataset_id}/snapshots", status_code=201)
def create_snapshot(dataset_id: int, request: Request):
    with connect(_url(request)) as conn:
        dataset = conn.execute(
            "SELECT * FROM datasets WHERE id = %s FOR UPDATE", (dataset_id,)
        ).fetchone()
        if dataset is None:
            raise HTTPException(404, f"dataset {dataset_id} not found")
        rows = conn.execute(
            """
            SELECT i.name, i.object_key, i.sha256, i.width, i.height, i.source, it.group_name,
                   coalesce(jsonb_agg(jsonb_build_object('label', a.label, 'bbox', a.bbox) ORDER BY a.id)
                            FILTER (WHERE a.id IS NOT NULL), '[]'::jsonb) AS annotations
            FROM dataset_items it
            JOIN dataset_images i ON i.id = it.image_id
            LEFT JOIN annotations a ON a.item_id = it.id
            WHERE it.dataset_id = %s AND it.review_status <> 'excluded'
            GROUP BY it.id, i.id
            ORDER BY i.name
            """,
            (dataset_id,),
        ).fetchall()
        if not rows:
            raise HTTPException(
                422,
                f"dataset {dataset_id} has no annotated items to snapshot"
                " (excluded items are left out)",
            )
        rows, left_out = _crop_training_rows(dataset, rows)
        if not rows:
            raise HTTPException(
                422,
                f"dataset {dataset_id} trains on box crops but no item has a box: draw boxes,"
                " run assisted labelling, or turn on whole-image training",
            )

        canon = [(_canonical(r), r) for r in rows]
        canon.sort(key=lambda pair: _dumps(pair[0]))
        content_hash = hashlib.sha256(
            _dumps({"task": dataset["task"], "items": [c for c, _ in canon]}).encode()
        ).hexdigest()
        label_counts = dict(
            Counter(a["label"] for c, _ in canon for a in c["annotations"])
        )

        snap = conn.execute(
            "INSERT INTO snapshots (dataset_id, content_hash, item_count, label_counts)"
            " VALUES (%s, %s, %s, %s) RETURNING *",
            (dataset_id, content_hash, len(canon), Jsonb(label_counts)),
        ).fetchone()
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO snapshot_items (snapshot_id, position, image_name, object_key, sha256,"
                " width, height, annotations, group_name) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                [
                    (
                        snap["id"],
                        pos,
                        r["name"],
                        r["object_key"],
                        c["sha256"],
                        c["width"],
                        c["height"],
                        Jsonb(c["annotations"]),
                        c["group"],
                    )
                    for pos, (c, r) in enumerate(canon)
                ],
            )
    return {**snap, "left_out": left_out}


def _crop_training_rows(dataset: dict, rows: list) -> tuple[list, int]:
    """Rows a snapshot keeps, and how many were left out.

    A classification dataset trains on box crops unless ``whole_image`` is on. An item with
    boxes keeps only its boxes; an image-level label on an uploaded image would train on the
    whole frame, so such items are left out. Studio imports are already crops and keep their
    image-level labels."""
    if dataset["task"] != "classification" or dataset["whole_image"]:
        return rows, 0
    kept, left_out = [], 0
    for r in rows:
        boxes = [a for a in r["annotations"] if a.get("bbox") is not None]
        if boxes:
            kept.append({**r, "annotations": boxes})
        elif r["source"] == "studio":
            kept.append(r)
        else:
            left_out += 1
    return kept, left_out


@router.get("/datasets/{dataset_id}/snapshots")
def list_snapshots(dataset_id: int, request: Request):
    with connect(_url(request)) as conn:
        _dataset_or_404(conn, dataset_id)
        rows = conn.execute(
            "SELECT * FROM snapshots WHERE dataset_id = %s ORDER BY id", (dataset_id,)
        ).fetchall()
    return {"snapshots": rows}


@router.get("/snapshots/{snapshot_id}")
def get_snapshot(snapshot_id: int, request: Request):
    with connect(_url(request)) as conn:
        row = conn.execute(
            "SELECT * FROM snapshots WHERE id = %s", (snapshot_id,)
        ).fetchone()
    if row is None:
        raise HTTPException(404, f"snapshot {snapshot_id} not found")
    return row


# --------------------------------------------------------------------------- Pascal VOC


def _zip_response(data: bytes, filename: str) -> Response:
    return Response(
        content=data,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _require_detection(dataset: dict, what: str) -> None:
    if dataset["task"] != "detection":
        raise HTTPException(
            422, f"{what} needs a detection dataset; dataset {dataset['id']} is {dataset['task']}"
        )


@router.get("/datasets/{dataset_id}/export/voc")
def export_voc(dataset_id: int, request: Request, status: Optional[str] = None):
    """Current boxes as one VOC XML per item. Excluded items are left out unless asked for by
    ``status``; confirmed items carry ``verified="yes"``."""
    if status is not None and status not in REVIEW_STATUSES:
        raise HTTPException(
            422, f"status must be one of {', '.join(REVIEW_STATUSES)}, not {status!r}"
        )
    with connect(_url(request)) as conn:
        _require_detection(_dataset_or_404(conn, dataset_id), "VOC export")
        rows = conn.execute(
            _ITEM_SQL, {"dataset": dataset_id, "status": status, "image": None}
        ).fetchall()
    rows = [
        {**r, "reviewed": r["review_status"] == "confirmed"}
        for r in rows
        if status is not None or r["review_status"] != "excluded"
    ]
    return _zip_response(voc.voc_zip(rows), f"dataset-{dataset_id}-voc.zip")


@router.get("/snapshots/{snapshot_id}/export/voc")
def export_snapshot_voc(snapshot_id: int, request: Request):
    """The frozen boxes of a snapshot (the exact set a job trained on) as VOC XML."""
    with connect(_url(request)) as conn:
        snap = conn.execute(
            "SELECT s.id, d.id AS dataset_id, d.task FROM snapshots s"
            " JOIN datasets d ON d.id = s.dataset_id WHERE s.id = %s",
            (snapshot_id,),
        ).fetchone()
        if snap is None:
            raise HTTPException(404, f"snapshot {snapshot_id} not found")
        _require_detection({"id": snap["dataset_id"], "task": snap["task"]}, "VOC export")
        rows = conn.execute(
            "SELECT image_name AS image, width, height, annotations FROM snapshot_items"
            " WHERE snapshot_id = %s ORDER BY position",
            (snapshot_id,),
        ).fetchall()
    return _zip_response(voc.voc_zip(rows), f"snapshot-{snapshot_id}-voc.zip")


def _replace_boxes(
    conn, dataset: dict, image: dict, item: Optional[dict], anns: list
) -> None:
    """Make ``anns`` the item's boxes and put it back to ``pending`` review."""
    dataset_id = dataset["id"]
    anns = _for_dataset(dataset, anns)
    if item is None:
        item_id = conn.execute(
            "INSERT INTO dataset_items (dataset_id, image_id, review_status)"
            " VALUES (%s, %s, 'pending') RETURNING id",
            (dataset_id, image["id"]),
        ).fetchone()["id"]
    else:
        item_id = item["id"]
        conn.execute("DELETE FROM annotations WHERE item_id = %s", (item_id,))
        conn.execute(
            "UPDATE dataset_items SET review_status = 'pending', updated_at = now()"
            " WHERE id = %s",
            (item_id,),
        )
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO annotations (item_id, label, bbox) VALUES (%s, %s, %s)",
            [(item_id, a.label.strip(), a.bbox) for a in anns],
        )


def _protected_reason(item: Optional[dict], overwrite: bool) -> Optional[str]:
    """Why an item's boxes must not be replaced, or ``None``."""
    if item is None:
        return None
    if item["review_status"] == "excluded":
        return "item is excluded from training; include it before replacing its boxes"
    if overwrite:
        return None
    if item["review_status"] == "confirmed":
        return "item is confirmed by a reviewer; pass overwrite=true to replace its boxes"
    return None


def _dataset_images_and_items(conn, dataset_id: int) -> tuple[dict, dict]:
    images = {
        r["name"]: r
        for r in conn.execute(
            "SELECT id, name, object_key, sha256, width, height, source FROM dataset_images"
            " WHERE dataset_id = %s",
            (dataset_id,),
        ).fetchall()
    }
    items = {
        r["image_id"]: r
        for r in conn.execute(
            "SELECT it.id, it.image_id, it.review_status,"
            " (SELECT count(*) FROM annotations a WHERE a.item_id = it.id) AS boxes,"
            " (SELECT count(*) FROM annotations a WHERE a.item_id = it.id AND a.bbox IS NOT NULL)"
            " AS bbox_boxes,"
            " (SELECT array_agg(DISTINCT a.label) FROM annotations a WHERE a.item_id = it.id)"
            " AS labels"
            " FROM dataset_items it WHERE it.dataset_id = %s",
            (dataset_id,),
        ).fetchall()
    }
    return images, items


def _has_boxes(item: Optional[dict], task: str, whole_image: bool = False) -> bool:
    """Whether assist counts the item as already labelled. A classification item with only
    image-level labels has no box yet: assist adds boxes and keeps its label (FT-35). In
    whole-image mode there are no boxes, so any image-level label counts."""
    if item is None:
        return False
    if task == "classification" and not whole_image:
        return bool(item["bbox_boxes"])
    return bool(item["boxes"])


def _classification_label(item: Optional[dict], animal_label: Optional[str]) -> str:
    """Label for an assisted box on a classification item: ``animal_label`` when given, else the
    item's single existing label, else ``animal`` for the reviewer to correct."""
    if animal_label:
        return animal_label.strip()
    labels = [name for name in ((item or {}).get("labels") or []) if name]
    return labels[0] if len(labels) == 1 else "animal"


def _dataset_labels(conn, dataset_id: int) -> set[str]:
    return {
        r["label"]
        for r in conn.execute(
            "SELECT DISTINCT a.label FROM annotations a JOIN dataset_items it ON it.id = a.item_id"
            " WHERE it.dataset_id = %s",
            (dataset_id,),
        ).fetchall()
    }


@router.post("/datasets/{dataset_id}/import/voc")
async def import_voc(dataset_id: int, request: Request, overwrite: bool = False):
    """Import boxes from a zip of VOC XML files (raw ``application/zip`` body)."""
    data = await _read_capped_body(request, voc.MAX_ZIP_BYTES, "VOC zip")
    return await run_in_threadpool(_import_voc, request, dataset_id, data, overwrite)


def _match_image(rec: interchange.Record, images: dict, stems: dict) -> Optional[str]:
    """The file name exactly, else another name the format gives (a VOC member, a Camtrap DP
    ``filePath``) exactly, else a unique image with the same stem (a ``.jpg`` named in the
    annotation for a ``.png`` upload, or a VOC file without ``<filename>``)."""
    for name in (rec.filename, *rec.alt_names):
        if name and name in images:
            return name
    for name in (rec.filename, *rec.alt_names):
        if not name:
            continue
        candidates = stems.get(name.rsplit(".", 1)[0].lower(), [])
        if len(candidates) == 1:
            return candidates[0]
    return None


def _voc_records(files: list) -> list[interchange.Record]:
    return [
        interchange.Record(
            source=f.member,
            filename=f.filename,
            alt_names=[interchange.leaf(f.member)],
            width=f.width,
            height=f.height,
            anns=[interchange.Ann(label=o.label, corners=o.corners) for o in f.objects],
            skipped=list(f.skipped),
        )
        for f in files
    ]


def _record_annotations(
    rec: interchange.Record, image: dict, task: str, skipped: list
) -> Optional[list[AnnotationIn]]:
    """The record's annotations for ``image``, or ``None`` when nothing usable is left of a
    record that had labels (a detection record whose labels all lack boxes)."""
    w, h = image["width"], image["height"]
    classification = task == "classification"
    anns = []
    for a in rec.anns:
        if a.corners is None:
            if classification:
                continue
            skipped.append(
                {"file": rec.source, "reason": f"label {a.label!r} has no box; detection datasets need one"}
            )
            continue
        corners = a.corners
        if a.normalized:
            x0, y0, x1, y1 = corners
            corners = (x0 * w, y0 * h, x1 * w, y1 * h)
        bbox = voc.to_bbox(corners, w, h)
        if bbox is None:
            skipped.append(
                {
                    "file": rec.source,
                    "reason": f"object {a.label!r} {list(a.corners)} is empty inside the {w}x{h} image",
                }
            )
            continue
        anns.append(AnnotationIn(label=a.label, bbox=bbox))
    if classification:
        # A boxed label trains on its crop, as Studio imports do; a whole-image label is kept only
        # for a species with no box on this image, so a box and an image-level tag do not double count.
        boxed = {a.label.strip() for a in anns}
        whole = [
            label
            for label in dict.fromkeys(a.label.strip() for a in rec.anns if a.corners is None)
            if label and label not in boxed
        ]
        anns = [AnnotationIn(label=a.label.strip(), bbox=a.bbox) for a in anns if a.label.strip()]
        return anns + [AnnotationIn(label=label) for label in whole]
    if rec.anns and not anns:
        return None
    return anns


def _apply_records(
    request: Request,
    dataset_id: int,
    records: list[interchange.Record],
    skipped: list[dict],
    overwrite: bool,
    fmt: str,
    detection_only: bool = False,
) -> dict:
    """Replace the annotations of every matched item; the summary says what happened to the rest."""
    with connect(_url(request)) as conn:
        dataset = conn.execute(
            "SELECT * FROM datasets WHERE id = %s FOR UPDATE", (dataset_id,)
        ).fetchone()
        if dataset is None:
            raise HTTPException(404, f"dataset {dataset_id} not found")
        if detection_only:
            _require_detection(dataset, f"{FORMAT_NAMES[fmt]} import")
        task = dataset["task"]
        images, items = _dataset_images_and_items(conn, dataset_id)
        known = _dataset_labels(conn, dataset_id)
        stems: dict[str, list[str]] = {}
        for name in images:
            stems.setdefault(name.rsplit(".", 1)[0].lower(), []).append(name)
        # Camtrap DP says "human"; MegaDetector and most detector datasets say "person".
        alias = {"human": "person"} if "person" in known and "human" not in known else {}

        unmatched: list[str] = []
        # Unlabelled images whose labels all lack boxes: candidates to move to a
        # classification dataset.
        unboxed: list[str] = []
        done: dict[str, str] = {}
        matched = boxes = 0
        new_labels: set[str] = set()
        for rec in records:
            name = _match_image(rec, images, stems)
            if name is None:
                unmatched.append(rec.source)
                continue
            if name in done:
                skipped.append(
                    {"file": rec.source, "reason": f"image {name!r} already imported from {done[name]}"}
                )
                continue
            image = images[name]
            item = items.get(image["id"])
            reason = _protected_reason(item, overwrite)
            if reason is None and rec.width and rec.height and (rec.width, rec.height) != (
                image["width"],
                image["height"],
            ):
                reason = (
                    f"{FORMAT_SIZE_WORD[fmt]} size {rec.width}x{rec.height} differs from the stored "
                    f"{image['width']}x{image['height']} image {name!r}"
                )
            if reason is not None:
                skipped.append({"file": rec.source, "reason": reason})
                continue
            for a in rec.anns:
                a.label = alias.get(a.label.strip().lower(), a.label)
            anns = _record_annotations(rec, image, task, skipped)
            skipped.extend({"file": rec.source, "reason": r} for r in rec.skipped)
            if anns is None:
                skipped.append({"file": rec.source, "reason": "nothing usable left; item left unchanged"})
                if item is None and all(a.corners is None for a in rec.anns):
                    unboxed.append(name)
                continue
            errors = _item_errors(0, ItemIn(image=name, annotations=anns), image, task)
            if errors:
                skipped.append({"file": rec.source, "reason": "; ".join(errors)})
                continue
            _replace_boxes(conn, dataset, image, item, anns)
            done[name] = rec.source
            matched += 1
            boxes += sum(1 for a in anns if a.bbox is not None)
            new_labels.update(a.label.strip() for a in anns if a.label.strip() not in known)
        summary = {
            "matched": matched,
            "boxes": boxes,
            "unmatched_files": unmatched,
            "new_labels": sorted(new_labels),
            "skipped": skipped,
            "unboxed_images": sorted(unboxed),
        }
        record_audit(
            conn,
            principal_name(request),
            f"dataset.import_{fmt}",
            f"dataset:{dataset_id}",
            {
                "matched": matched,
                "boxes": boxes,
                "unmatched": len(unmatched),
                "skipped": len(skipped),
                "overwrite": overwrite,
            },
        )
    return summary


def _import_voc(request: Request, dataset_id: int, data: bytes, overwrite: bool) -> dict:
    try:
        files, skipped = voc.read_voc_zip(data)
    except voc.VocError as exc:
        raise HTTPException(422, str(exc)) from exc
    return _apply_records(
        request, dataset_id, _voc_records(files), skipped, overwrite, "voc", detection_only=True
    )


# --------------------------------------------------------------------------- COCO and Camera Trap DP

FORMAT_NAMES = {"voc": "VOC", "coco": "COCO", "camtrapdp": "Camtrap DP"}
FORMAT_SIZE_WORD = {"voc": "XML", "coco": "COCO", "camtrapdp": "Camtrap DP"}


def _export_rows(conn, dataset_id: int, status: Optional[str]) -> list[dict]:
    if status is not None and status not in REVIEW_STATUSES:
        raise HTTPException(
            422, f"status must be one of {', '.join(REVIEW_STATUSES)}, not {status!r}"
        )
    rows = conn.execute(
        _ITEM_SQL, {"dataset": dataset_id, "status": status, "image": None}
    ).fetchall()
    return [r for r in rows if status is not None or r["review_status"] != "excluded"]


def _snapshot_rows(conn, snapshot_id: int) -> tuple[dict, list[dict]]:
    snap = conn.execute(
        "SELECT s.id, d.id AS dataset_id, d.name, d.task FROM snapshots s"
        " JOIN datasets d ON d.id = s.dataset_id WHERE s.id = %s",
        (snapshot_id,),
    ).fetchone()
    if snap is None:
        raise HTTPException(404, f"snapshot {snapshot_id} not found")
    rows = conn.execute(
        "SELECT image_name AS image, width, height, annotations FROM snapshot_items"
        " WHERE snapshot_id = %s ORDER BY position",
        (snapshot_id,),
    ).fetchall()
    # Training reads an item without labels in a snapshot as a blank frame.
    return snap, [{**r, "blank_if_empty": True} for r in rows]


def _json_response(data: bytes, filename: str) -> Response:
    return Response(
        content=data,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/datasets/{dataset_id}/export/coco")
def export_coco(dataset_id: int, request: Request, status: Optional[str] = None):
    """Current annotations as one COCO JSON. Excluded items are left out unless asked for."""
    with connect(_url(request)) as conn:
        dataset = _dataset_or_404(conn, dataset_id)
        rows = _export_rows(conn, dataset_id, status)
    return _json_response(
        coco.coco_json(rows, dataset["task"], dataset["name"]), f"dataset-{dataset_id}-coco.json"
    )


@router.get("/snapshots/{snapshot_id}/export/coco")
def export_snapshot_coco(snapshot_id: int, request: Request):
    with connect(_url(request)) as conn:
        snap, rows = _snapshot_rows(conn, snapshot_id)
    return _json_response(
        coco.coco_json(rows, snap["task"], snap["name"]), f"snapshot-{snapshot_id}-coco.json"
    )


@router.get("/datasets/{dataset_id}/export/camtrapdp")
def export_camtrapdp(dataset_id: int, request: Request, status: Optional[str] = None):
    """Current annotations as a Camera Trap Data Package zip."""
    with connect(_url(request)) as conn:
        dataset = _dataset_or_404(conn, dataset_id)
        rows = _export_rows(conn, dataset_id, status)
    return _zip_response(
        camtrapdp.write_package(rows, dataset["task"], dataset["name"]),
        f"dataset-{dataset_id}-camtrapdp.zip",
    )


@router.get("/snapshots/{snapshot_id}/export/camtrapdp")
def export_snapshot_camtrapdp(snapshot_id: int, request: Request):
    with connect(_url(request)) as conn:
        snap, rows = _snapshot_rows(conn, snapshot_id)
    return _zip_response(
        camtrapdp.write_package(rows, snap["task"], snap["name"]),
        f"snapshot-{snapshot_id}-camtrapdp.zip",
    )


def _import_parsed(request, dataset_id, data, overwrite, fmt, reader) -> dict:
    try:
        records, skipped = reader(data)
    except interchange.FormatError as exc:
        raise HTTPException(422, str(exc)) from exc
    return _apply_records(request, dataset_id, records, skipped, overwrite, fmt)


@router.post("/datasets/{dataset_id}/import/coco")
async def import_coco(dataset_id: int, request: Request, overwrite: bool = False):
    """Import annotations from a COCO JSON document (raw body: the ``.json``, or a zip of it)."""
    data = await _read_capped_body(request, coco.MAX_JSON_BYTES, "COCO file")
    return await run_in_threadpool(
        _import_parsed, request, dataset_id, data, overwrite, "coco", coco.read_coco
    )


@router.post("/datasets/{dataset_id}/import/camtrapdp")
async def import_camtrapdp(dataset_id: int, request: Request, overwrite: bool = False):
    """Import annotations from a Camera Trap Data Package zip (raw ``application/zip`` body)."""
    data = await _read_capped_body(request, voc.MAX_ZIP_BYTES, "Camtrap DP zip")
    return await run_in_threadpool(
        _import_parsed, request, dataset_id, data, overwrite, "camtrapdp", camtrapdp.read_package
    )


# --------------------------------------------------------------------------- assisted labelling

# Inference runs inside the request on CPU; past this many items a call is refused, and the
# caller sends the images in batches. Adding a queued job kind would need the job queue changed.
ASSIST_MAX_ITEMS = 500


class AssistIn(BaseModel):
    # A trained yolo_detector model id, or "megadetector" (the built-in default).
    model_id: int | Literal["megadetector"] = assist.MEGADETECTOR_ID
    # MegaDetector only: write its "animal" boxes under this label (e.g. a species) instead.
    animal_label: Optional[str] = Field(default=None, min_length=1, max_length=200)
    # MegaDetector on a classification dataset only: label each animal box with this species
    # classifier, run on the box crop.
    classifier: Optional[Literal["speciesnet"]] = None
    images: Optional[list[str]] = Field(default=None, min_length=1)
    conf: FiniteFloat = Field(default=assist.DEFAULT_CONF, ge=0.0, le=1.0)
    overwrite: bool = False
    # Only label images from these sources ("upload": added in this app, "studio": imported
    # from Sparrow Studio). None means every image.
    sources: Optional[list[Literal["upload", "studio"]]] = Field(default=None, min_length=1)


@router.get("/assist/models")
def assist_models(request: Request):
    """Detectors assisted labelling can run: MegaDetector (when installed) then trained detectors."""
    settings = request.app.state.settings
    md_labels = assist.megadetector_labels(settings.weights_dir)
    out = [
        {
            "id": assist.MEGADETECTOR_ID,
            "name": assist.MEGADETECTOR_NAME,
            "labels": md_labels,
            "available": bool(md_labels),
            "default": True,
        }
    ]
    with connect(settings.database_url) as conn:
        rows = conn.execute(
            "SELECT id FROM model_versions WHERE pipeline = 'yolo_detector' ORDER BY id DESC"
        ).fetchall()
    out.extend({"id": r["id"], "available": True, "default": False} for r in rows)
    classifiers = [
        {
            "id": assist.SPECIESNET_ID,
            "name": assist.SPECIESNET_NAME,
            "available": assist.speciesnet_available(settings.weights_dir),
        }
    ]
    return {"models": out, "classifiers": classifiers}


IMAGE_SOURCES = ("upload", "studio")


@router.get("/datasets/{dataset_id}/assist/summary")
def assist_summary(dataset_id: int, request: Request):
    """Per-source image counts the UI shows before pre-drawing boxes.

    Each image falls in exactly one bucket: ``excluded``; ``confirmed``; ``with_boxes``
    (pending, already has boxes); ``empty`` (no boxes yet). Without overwrite assist labels
    only ``empty`` images; with overwrite it labels everything that is not excluded."""
    with connect(_url(request)) as conn:
        dataset = _dataset_or_404(conn, dataset_id)
        task = dataset["task"]
        images, items = _dataset_images_and_items(conn, dataset_id)
    out = {
        src: {"images": 0, "empty": 0, "with_boxes": 0, "confirmed": 0, "excluded": 0}
        for src in IMAGE_SOURCES
    }
    for image in images.values():
        counts = out[image["source"]]
        counts["images"] += 1
        item = items.get(image["id"])
        if item is not None and item["review_status"] == "excluded":
            counts["excluded"] += 1
        elif item is not None and item["review_status"] == "confirmed":
            counts["confirmed"] += 1
        elif _has_boxes(item, task, dataset["whole_image"]):
            counts["with_boxes"] += 1
        else:
            counts["empty"] += 1
    return {"sources": out}


@router.post("/datasets/{dataset_id}/assist")
def assist_labels(dataset_id: int, body: AssistIn, request: Request):
    """Pre-draw boxes with MegaDetector (default) or a trained ``yolo_detector`` model.

    Boxes land as ``pending`` review. MegaDetector boxes keep its labels (animal, person,
    vehicle) unless ``animal_label`` renames the animal boxes.

    On a classification dataset (FT-35) the boxes become the crops the classifier trains on.
    MegaDetector keeps its ``animal`` boxes only; each takes ``animal_label``, else the item's
    single existing image-level label, else ``animal``. An item with image-level labels but no
    box counts as not yet boxed. A ``yolo_detector`` model's boxes keep the model's labels.
    With ``classifier="speciesnet"`` (MegaDetector only) each animal box is cropped and
    labelled by SpeciesNet, rolled up through its taxonomy to the dataset's existing label
    spelling (genus, family, ...); SpeciesNet's label replaces the item's existing label.

    On a whole-image dataset (FT-38) the boxes are folded into image-level labels on write.

    Targets are ``images`` (by name) or every image in the dataset, narrowed to ``sources``
    when given (images from other sources are skipped). Excluded items are never
    touched; confirmed items, and items that already have boxes, are skipped unless
    ``overwrite`` is set. An image with no detections above ``conf`` is left unchanged."""
    settings = request.app.state.settings
    store = LocalObjectStore(settings.objects_dir)
    with connect(settings.database_url) as conn:
        dataset = _dataset_or_404(conn, dataset_id)
        task = dataset["task"]
        megadetector = body.model_id == assist.MEGADETECTOR_ID
        model = None
        if megadetector:
            if body.animal_label is not None and not body.animal_label.strip():
                raise HTTPException(422, "animal_label must not be blank")
            if body.classifier is not None:
                if task != "classification":
                    raise HTTPException(422, "classifier applies to classification datasets only")
                if body.animal_label is not None:
                    raise HTTPException(
                        422, "pass either animal_label or classifier, not both"
                    )
        else:
            if body.animal_label is not None:
                raise HTTPException(422, "animal_label applies to MegaDetector only")
            if body.classifier is not None:
                raise HTTPException(422, "classifier applies to MegaDetector only")
            model = conn.execute(
                "SELECT id, pipeline, object_key FROM model_versions WHERE id = %s",
                (body.model_id,),
            ).fetchone()
            if model is None:
                raise HTTPException(404, f"model {body.model_id} not found")
            if model["pipeline"] != "yolo_detector":
                raise HTTPException(
                    422,
                    f"model {body.model_id} is a {model['pipeline']} model; assisted labelling "
                    "needs a yolo_detector model or MegaDetector",
                )
        images, items = _dataset_images_and_items(conn, dataset_id)
        known = _dataset_labels(conn, dataset_id)
    if body.images is not None:
        missing = sorted({n for n in body.images if n not in images})
        if missing:
            raise HTTPException(422, f"unknown images in dataset {dataset_id}: {missing}")
        names = list(dict.fromkeys(body.images))
    else:
        names = sorted(images)
    allowed = set(body.sources) if body.sources is not None else set(IMAGE_SOURCES)
    if body.images is None:
        names = [n for n in names if images[n]["source"] in allowed]

    skipped: list[dict] = []
    targets = []
    for name in names:
        item = items.get(images[name]["id"])
        if images[name]["source"] not in allowed:
            skipped.append(
                {"image": name, "reason": f"image source is {images[name]['source']}, not in sources"}
            )
            continue
        reason = _protected_reason(item, body.overwrite)
        if reason is None and _has_boxes(item, task, dataset["whole_image"]) and not body.overwrite:
            reason = "item already has boxes; pass overwrite=true to replace them"
        if reason is None:
            targets.append(images[name])
        else:
            skipped.append({"image": name, "reason": reason})
    if len(targets) > ASSIST_MAX_ITEMS:
        raise HTTPException(
            422,
            f"{len(targets)} images to label; one call handles at most {ASSIST_MAX_ITEMS}. "
            "Pass `images` in batches.",
        )

    if megadetector:
        try:
            predictor = assist.load_megadetector(settings.weights_dir)
        except assist.AssistError as exc:
            raise HTTPException(503, str(exc)) from exc
        # MegaDetector's classes are generic by design: keep them all and let reviewers relabel.
        rename = {"animal": body.animal_label.strip()} if body.animal_label else {}
        unknown = []
        classifier = None
        if body.classifier is not None:
            try:
                classifier = assist.load_speciesnet(settings.weights_dir)
            except assist.AssistError as exc:
                raise HTTPException(503, str(exc)) from exc
        known_labels = sorted(known)
    else:
        if not store.exists(model["object_key"]):
            raise HTTPException(500, f"artifact for model {body.model_id} is missing from the object store")
        try:
            predictor = assist.load_predictor(store.path(model["object_key"]).read_bytes())
        except assist.AssistError as exc:
            raise HTTPException(422, str(exc)) from exc
        rename = {}
        unknown = sorted(set(predictor.labels) - known) if known else []

    found: dict[str, list] = {}
    # Best score among images that drew nothing, so the UI can say how far below `conf` they were.
    best_below_conf: Optional[float] = None
    classify_md = megadetector and task == "classification"
    for image in targets:
        if not store.exists(image["object_key"]):
            skipped.append({"image": image["name"], "reason": "image is missing from the object store"})
            continue
        stats: dict = {}
        rgb = assist.read_rgb(store.path(image["object_key"]))
        detections = [
            assist.Detection(label=rename.get(d.label, d.label), bbox=d.bbox, score=d.score)
            for d in predictor.predict(rgb, body.conf, stats=stats)
            if d.label not in unknown and (not classify_md or d.label == "animal")
        ]
        if classify_md and classifier is not None:
            detections = [
                assist.Detection(
                    label=classifier.label_for(rgb, d.bbox, known_labels)[0],
                    bbox=d.bbox,
                    score=d.score,
                )
                for d in detections
            ]
        elif classify_md:
            label = _classification_label(items.get(image["id"]), body.animal_label)
            detections = [assist.Detection(label=label, bbox=d.bbox, score=d.score) for d in detections]
        if not detections:
            top = stats.get("top_score")
            if top is not None and top <= body.conf:
                best_below_conf = top if best_below_conf is None else max(best_below_conf, top)
            skipped.append({"image": image["name"], "reason": f"no detections above conf {body.conf}"})
            continue
        found[image["name"]] = detections

    written = []
    with connect(settings.database_url) as conn:
        if (
            conn.execute(
                "SELECT 1 FROM datasets WHERE id = %s FOR UPDATE", (dataset_id,)
            ).fetchone()
            is None
        ):
            raise HTTPException(404, f"dataset {dataset_id} not found")
        # Inference ran without locks: re-check each item against what a reviewer did meanwhile.
        now_images, now_items = _dataset_images_and_items(conn, dataset_id)
        for name, detections in found.items():
            image = now_images.get(name)
            if image is None or image["sha256"] != images[name]["sha256"]:
                skipped.append({"image": name, "reason": "image changed while the model ran"})
                continue
            item = now_items.get(image["id"])
            before = items.get(image["id"])
            changed = (item is None) != (before is None) or (
                item is not None
                and (item["review_status"], item["boxes"]) != (before["review_status"], before["boxes"])
            )
            if changed and not body.overwrite:
                skipped.append({"image": name, "reason": "item changed while the model ran"})
                continue
            reason = _protected_reason(item, body.overwrite)
            if reason is not None:
                skipped.append({"image": name, "reason": reason})
                continue
            anns = [AnnotationIn(label=d.label, bbox=d.bbox) for d in detections]
            errors = _item_errors(0, ItemIn(image=name, annotations=anns), image, task)
            if errors:
                skipped.append({"image": name, "reason": "; ".join(errors)})
                continue
            _replace_boxes(conn, dataset, image, item, anns)
            written.append(
                {
                    "image": name,
                    "boxes": [
                        {"label": d.label, "bbox": d.bbox, "score": round(d.score, 6)}
                        for d in detections
                    ],
                }
            )
        n_boxes = sum(len(w["boxes"]) for w in written)
        record_audit(
            conn,
            principal_name(request),
            "dataset.assist",
            f"dataset:{dataset_id}",
            {
                "model_id": body.model_id,
                "conf": body.conf,
                "overwrite": body.overwrite,
                "processed": len(targets),
                "items": len(written),
                "boxes": n_boxes,
                **({"animal_label": body.animal_label} if body.animal_label else {}),
                **({"classifier": body.classifier} if body.classifier else {}),
            },
        )
    return {
        "model_id": body.model_id,
        "classifier": body.classifier,
        "conf": body.conf,
        "processed": len(targets),
        "items_written": len(written),
        "boxes": n_boxes,
        "unknown_labels": unknown,
        "best_below_conf": None if best_below_conf is None else round(best_below_conf, 6),
        "skipped": skipped,
        "items": written,
    }
