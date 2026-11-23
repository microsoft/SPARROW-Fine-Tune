"""Model catalogue: Sparrow Engine Model Zoo (standalone) or Studio project models (connected).

- ``GET /catalog`` -> ``{mode, models, catalog_error}``. Standalone: every Engine Zoo
  entry (``sparrow_engine.zoo_models()``) plus anything already installed in
  ``<data_dir>/engine-models``; offline, the installed models are still listed and
  ``catalog_error`` says why the Zoo is missing. Connected (Studio configured): the models
  of each Studio project, nothing is downloaded from the Zoo and no Studio bundle is fetched:
  only a model already imported as a base model is tagged fine-tunable.
- ``POST /catalog/models/{id}/download`` installs exactly one selected, hosted Zoo entry
  (201; 200 when it is already installed, without calling the Engine). Link-only and
  pending-rights entries answer 422 with their source URL.

Every row carries ``finetune_tag``/``finetune_reason``. A Zoo entry is never tagged
fine-tunable: the catalogue ships runtime packages and declares no validated training
checkpoint, and an architecture name alone does not make a model trainable.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from .. import base_models
from ..connectors import studio

router = APIRouter()

MODEL_DIR_NAME = "engine-models"
INSTALL_MARKERS = ("manifest.toml", "pipeline.toml", "ensemble.toml")
ZOO_REASON = (
    "the Engine catalogue declares no training weights for this model "
    "(it ships a runtime package only)"
)
MAX_STUDIO_PROJECTS = 20
MAX_STUDIO_MODELS = 200
MAX_INSTALLED_LISTED = 1000
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")
_DOWNLOAD_LOCK = threading.Lock()
STUDIO_UNCHECKED = (
    "not checked yet: importing it as a base model checks its training weights "
    "(the list never downloads Studio bundles)"
)


def _model_dir(settings) -> Path:
    return Path(settings.data_dir) / MODEL_DIR_NAME


def _safe_id(model_id: str) -> str:
    if not _ID.match(model_id or "") or ".." in model_id:
        raise HTTPException(422, f"invalid model id {model_id!r}")
    return model_id


def _installed(model_dir: Path, model_id: str) -> bool:
    d = model_dir / model_id
    return any((d / name).is_file() for name in INSTALL_MARKERS)


def _installed_ids(model_dir: Path) -> list[str]:
    if not model_dir.is_dir():
        return []
    out = []
    for d in sorted(model_dir.iterdir()):
        if len(out) >= MAX_INSTALLED_LISTED:
            break
        if d.is_dir() and _ID.match(d.name) and _installed(model_dir, d.name):
            out.append(d.name)
    return out


def _engine():
    try:
        import sparrow_engine
    except ImportError as exc:
        raise RuntimeError(
            "Sparrow Engine is not installed (pip install 'sparrow-finetune[catalog]')"
        ) from exc
    return sparrow_engine


def _zoo() -> list:
    return list(_engine().zoo_models())


def _zoo_row(m, installed: bool) -> dict:
    row = {
        "id": m.id,
        "name": m.id,
        "domain": m.domain,
        "task": m.task,
        "format": m.format,
        "license": m.license,
        "status": m.status,
        "hosting_status": m.hosting_status,
        "downloadable": bool(m.downloadable),
        "source_url": m.source_url,
        "installed": installed,
        "source": "engine",
    }
    return base_models.tag(row, False, ZOO_REASON)


def _installed_row(model_id: str) -> dict:
    row = {
        "id": model_id,
        "name": model_id,
        "domain": None,
        "task": None,
        "format": None,
        "license": None,
        "downloadable": False,
        "source_url": None,
        "installed": True,
        "source": "engine",
    }
    return base_models.tag(row, False, ZOO_REASON)


def _standalone(settings) -> dict:
    model_dir = _model_dir(settings)
    installed = set(_installed_ids(model_dir))
    error = None
    try:
        zoo = _zoo()
    except Exception as exc:  # noqa: BLE001 - ZooError, network, missing Engine
        zoo, error = [], f"the Engine Model Zoo is unavailable: {exc}"
    rows = [_zoo_row(m, m.id in installed) for m in zoo]
    listed = {m.id for m in zoo}
    rows += [_installed_row(i) for i in sorted(installed - listed)]
    return {"mode": "standalone", "models": rows, "catalog_error": error}


def _studio_tag(settings, project: str, m: dict) -> tuple:
    """Tag a Studio model from local evidence only; listing never fetches a bundle.

    Only a model already imported as a base model has passed the training-weights check; any
    other model is listed as not checked, and importing it runs the full check."""
    studio_id = str(m.get("id"))
    if m.get("pipeline") not in base_models.WARM_START_PIPELINES:
        return False, f"pipeline {m.get('pipeline')!r} cannot be a warm-start base"
    if base_models.find(settings.database_url, project, studio_id) is not None:
        return True, None
    return False, STUDIO_UNCHECKED


def _connected(request: Request, settings) -> dict:
    rows, errors = [], []
    try:
        projects = studio._remote_projects(request)
    except HTTPException as exc:
        return {"mode": "connected", "models": [], "catalog_error": str(exc.detail)}
    if len(projects) > MAX_STUDIO_PROJECTS:
        errors.append(
            f"only the first {MAX_STUDIO_PROJECTS} Studio projects are listed"
        )
    for p in projects[:MAX_STUDIO_PROJECTS]:
        try:
            models = studio._remote_models(request, p["id"])
        except HTTPException as exc:
            errors.append(f"project {p['name']}: {exc.detail}")
            continue
        for m in models:
            if len(rows) >= MAX_STUDIO_MODELS:
                errors.append(
                    f"only the first {MAX_STUDIO_MODELS} Studio models are listed"
                )
                break
            ok, reason = _studio_tag(settings, p["id"], m)
            row = {
                "id": str(m.get("id")),
                "name": m.get("source_model_id") or str(m.get("id")),
                "project": p["id"],
                "project_name": p["name"],
                "pipeline": m.get("pipeline"),
                "domain": None,
                "task": m.get("task"),
                "format": m.get("format"),
                "license": None,
                "labels": m.get("labels"),
                "sha256": m.get("sha256"),
                "created_at": m.get("created_at"),
                "downloadable": False,
                "installed": False,
                "source_url": None,
                "source": "studio",
            }
            rows.append(base_models.tag(row, ok, reason))
        if len(rows) >= MAX_STUDIO_MODELS:
            break
    return {
        "mode": "connected",
        "models": rows,
        "catalog_error": "; ".join(dict.fromkeys(errors)) or None,
    }


@router.get("/catalog")
async def get_catalog(request: Request):
    """Models offered for download (standalone) or in Studio projects (connected)."""
    settings = request.app.state.settings
    if settings.studio_configured:
        return await run_in_threadpool(_connected, request, settings)
    return await run_in_threadpool(_standalone, settings)


def _download(settings, model_id: str) -> tuple[dict, bool]:
    model_dir = _model_dir(settings)
    with _DOWNLOAD_LOCK:
        if _installed(model_dir, model_id):
            return _installed_row(model_id), False
        try:
            zoo = _zoo()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                502, f"the Engine Model Zoo is unavailable: {exc}"
            ) from exc
        m = next((m for m in zoo if m.id == model_id), None)
        if m is None:
            raise HTTPException(404, f"model {model_id} is not in the Engine catalogue")
        if not m.downloadable:
            raise HTTPException(
                422,
                f"model {model_id} is {m.hosting_status} ({m.status}); Sparrow does not "
                f"distribute it. Get it from its source: {m.source_url or 'not listed'}",
            )
        model_dir.mkdir(parents=True, exist_ok=True)
        try:
            _engine().download_models([model_id], dest=str(model_dir))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(502, f"download of {model_id} failed: {exc}") from exc
        if not _installed(model_dir, model_id):
            raise HTTPException(
                502, f"download of {model_id} did not install a runtime package"
            )
        return _zoo_row(m, True), True


@router.post("/catalog/models/{model_id}/download")
async def download_model(model_id: str, request: Request):
    """Install one selected, hosted Zoo model (201 new, 200 already installed)."""
    settings = request.app.state.settings
    if settings.studio_configured:
        raise HTTPException(
            409,
            "connected to Sparrow Studio: models come from Studio projects, "
            "not from the Engine Model Zoo",
        )
    model_id = _safe_id(model_id)
    row, created = await run_in_threadpool(_download, settings, model_id)
    return JSONResponse(row, status_code=201 if created else 200)
