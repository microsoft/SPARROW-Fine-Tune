"""Local base models: ``POST /base-models/local`` imports a timm training package.

The package is the ZIP that ``sparrow-uploader prepare-finetune`` writes; ``base_models``
validates it in full (stored members only, manifest, sha256, safetensors framing, exact timm
``resnet50`` or ``efficientnet_b0`` tensors for the declared ``model_arch``, bounded labels)
before anything is stored. No Sparrow Studio is involved,
so this works on a standalone service. ``GET /base-models`` lists local and Studio rows alike.
"""

from __future__ import annotations

import hashlib
import threading
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from .. import base_models
from .security import audit

router = APIRouter()

FIELD = "file"
# Multipart framing (boundaries, part headers) around the package itself.
_ENVELOPE_BYTES = 64 << 10
MAX_BODY_BYTES = base_models.MAX_LOCAL_PACKAGE_BYTES + _ENVELOPE_BYTES
# Each upload buffers up to MAX_LOCAL_PACKAGE_BYTES in memory.
MAX_CONCURRENT_UPLOADS = 2
_UPLOADS = threading.BoundedSemaphore(MAX_CONCURRENT_UPLOADS)


async def _capped(request: Request, cap: int) -> AsyncIterator[bytes]:
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > cap:
            raise HTTPException(
                413, f"package larger than {base_models.MAX_LOCAL_PACKAGE_BYTES} bytes"
            )
        yield chunk


async def _read_package(request: Request) -> bytes:
    """The bytes of the single multipart file field ``file``, never more than the cap."""
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            length = int(declared)
        except ValueError:
            raise HTTPException(400, f"invalid Content-Length: {declared!r}") from None
        if length < 0:
            raise HTTPException(400, f"invalid Content-Length: {declared!r}")
        if length > MAX_BODY_BYTES:
            raise HTTPException(
                413, f"package larger than {base_models.MAX_LOCAL_PACKAGE_BYTES} bytes"
            )
    content_type = request.headers.get("content-type", "")
    if not content_type.lower().startswith("multipart/form-data"):
        raise HTTPException(
            422, f"upload the package as multipart/form-data in the field {FIELD!r}"
        )
    parser = MultiPartParser(
        request.headers,
        _capped(request, MAX_BODY_BYTES),
        max_files=1,
        max_fields=4,
        max_part_size=64 << 10,
    )
    try:
        form = await parser.parse()
    except MultiPartException as exc:
        raise HTTPException(422, f"invalid multipart upload: {exc.message}") from exc
    try:
        upload = form.get(FIELD)
        if not isinstance(upload, UploadFile):
            raise HTTPException(
                422, f"the package must be sent as the multipart file field {FIELD!r}"
            )
        data = await upload.read(base_models.MAX_LOCAL_PACKAGE_BYTES + 1)
    finally:
        await form.close()
    if len(data) > base_models.MAX_LOCAL_PACKAGE_BYTES:
        raise HTTPException(
            413, f"package larger than {base_models.MAX_LOCAL_PACKAGE_BYTES} bytes"
        )
    return data


def _import(request: Request, data: bytes) -> tuple[dict, bool]:
    settings = request.app.state.settings
    existing = base_models.find_local(
        settings.database_url, hashlib.sha256(data).hexdigest()
    )
    if existing is not None:
        return existing, False
    try:
        facts = base_models.inspect_local_package(data)
    except base_models.NotImportable as exc:
        raise HTTPException(422, f"cannot import the local base model: {exc}") from exc
    row, created = base_models.store_local(settings, facts)
    if created:
        audit(
            request,
            "base_model.local_upload",
            f"base_model:{row['id']}",
            {
                "sha256": row["sha256"],
                "weights_sha256": facts["weights_sha256"],
                "model_arch": row["model_arch"],
                "labels": len(row["labels"]),
                "size_bytes": row["size_bytes"],
            },
        )
    return row, created


@router.post("/base-models/local")
async def upload_local_base_model(request: Request):
    """Import a ``sparrow-finetune/base-v1`` package; 201 when new, 200 for the same bytes again.

    At most ``MAX_CONCURRENT_UPLOADS`` packages are buffered at once; more answer 429."""
    if not _UPLOADS.acquire(blocking=False):
        raise HTTPException(
            429, "too many base-model uploads in progress; retry when one finishes"
        )
    try:
        data = await _read_package(request)
        row, created = await run_in_threadpool(_import, request, data)
    finally:
        _UPLOADS.release()
    return JSONResponse(jsonable_encoder(row), status_code=201 if created else 200)
