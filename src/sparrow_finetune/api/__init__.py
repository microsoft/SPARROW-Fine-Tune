"""HTTP API (FastAPI). ``create_app`` wires settings and the trainer registry into routers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import psycopg
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..config import Settings
from ..connectors import studio
from ..jsonsafe import sanitise_non_finite
from ..pipelines import Registry, default_registry
from . import catalog, datasets, jobs, local_models
from .security import audit_router, auth_router, require_auth

log = logging.getLogger(__name__)

UI_DIR = Path(__file__).resolve().parent.parent / "ui"


class _RevalidatedStaticFiles(StaticFiles):
    """UI assets revalidate on every load (ETag 304s), so new HTML never runs with old CSS/JS."""

    def file_response(self, *args: Any, **kwargs: Any):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response

health_router = APIRouter()


@health_router.get("/health")
def health(request: Request):
    try:
        with psycopg.connect(
            request.app.state.settings.database_url, connect_timeout=5
        ) as conn:
            conn.execute("SELECT 1")
    except psycopg.Error:
        return JSONResponse(
            {"status": "degraded", "database": "error"}, status_code=503
        )
    return {"status": "ok", "database": "ok"}


def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    # FastAPI echoes the offending input; a NaN/Infinity input would make the 422 body
    # itself unserialisable (500), so render non-finite floats as null.
    detail = sanitise_non_finite(jsonable_encoder(exc.errors()))
    return JSONResponse({"detail": detail}, status_code=422)


def create_app(
    settings: Settings,
    registry: Optional[Registry] = None,
    studio_transport: Optional[Any] = None,
) -> FastAPI:
    """``studio_transport`` is an ``httpx`` transport used for Studio calls (tests)."""
    app = FastAPI(
        title="Sparrow fine-tuning service",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.settings = settings
    if settings.auth_disabled:
        log.warning(
            "sign-in is disabled (SPARROW_FT_AUTH_DISABLED): anyone who can reach this"
            " service can use it; keep it behind Sparrow Studio's network"
        )
    app.state.registry = default_registry if registry is None else registry
    app.state.studio_transport = studio_transport
    app.add_exception_handler(RequestValidationError, _validation_error)
    protected = [Depends(require_auth)]
    app.include_router(health_router, prefix="/v1")
    app.include_router(auth_router, prefix="/v1")
    app.include_router(audit_router, prefix="/v1", dependencies=protected)
    app.include_router(datasets.router, prefix="/v1", dependencies=protected)
    app.include_router(jobs.router, prefix="/v1", dependencies=protected)
    app.include_router(local_models.router, prefix="/v1", dependencies=protected)
    app.include_router(catalog.router, prefix="/v1", dependencies=protected)
    app.include_router(studio.router, prefix="/v1", dependencies=protected)

    # API docs carry the full route map, so they sit behind the same auth as the API.
    @app.get("/openapi.json", include_in_schema=False, dependencies=protected)
    def openapi_schema():
        return JSONResponse(app.openapi())

    @app.get("/docs", include_in_schema=False, dependencies=protected)
    def swagger_ui():
        return get_swagger_ui_html(openapi_url="/openapi.json", title=app.title)

    @app.get("/redoc", include_in_schema=False, dependencies=protected)
    def redoc_ui():
        return get_redoc_html(openapi_url="/openapi.json", title=app.title)

    @app.get("/", include_in_schema=False)
    def ui_index():
        return FileResponse(
            UI_DIR / "index.html", headers={"Cache-Control": "no-cache"}
        )

    app.mount("/ui", _RevalidatedStaticFiles(directory=UI_DIR), name="ui")
    return app


__all__ = ["create_app"]
