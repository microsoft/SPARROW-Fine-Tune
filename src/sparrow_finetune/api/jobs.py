"""Pipelines catalogue, jobs (submit, inspect, stream, cancel, retry), models and deployments."""

from __future__ import annotations

import asyncio
import json
import math
import threading
from dataclasses import asdict
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from ..base_models import check_base_model, check_parent_model, reuse_preview, tag_model
from .. import features as features_mod
from ..db import connect
from ..jobs import TERMINAL_STATES, AlreadyRetried, JobQueue
from ..jsonsafe import find_non_finite, sanitise_non_finite
from ..pipelines import PLANNED_PIPELINES
from ..pipelines.stock_model import list_stock_packages
from ..storage import LocalObjectStore
from .security import audit, principal_name, record_audit

router = APIRouter()

_MODEL_COLUMNS = (
    "id, name, job_id, pipeline, snapshot_id, metrics, format, sha256, metadata, created_at"
)

MODEL_NAME_MAX = 100

# Jobs and model versions reach their workspace through snapshot -> dataset.
_DATASET_JOIN = (
    "JOIN snapshots s ON s.id = j.snapshot_id JOIN datasets d ON d.id = s.dataset_id"
)


class JobIn(BaseModel):
    pipeline: str
    snapshot_id: int
    params: dict = Field(default_factory=dict)


def _queue(request: Request) -> JobQueue:
    return JobQueue(request.app.state.settings.database_url)


def _requires(request: Request, pipeline: str, declared) -> set[str]:
    """The trainer's own requirements, plus ``gpu`` for ``SPARROW_FT_GPU_PIPELINES``."""
    requires = set(declared)
    if pipeline in request.app.state.settings.gpu_pipelines:
        requires.add("gpu")
    return requires


def _with_queue(request: Request, jobs: list[dict]) -> list[dict]:
    """Attach ``queue`` (why it is waiting, see ``JobQueue.queue_status``) to queued jobs."""
    status = _queue(request).queue_status(
        [j["id"] for j in jobs if j["state"] == "queued"],
        request.app.state.settings.max_running_per_user,
    )
    for j in jobs:
        j["queue"] = status.get(j["id"])
    return jobs


@router.get("/pipelines")
def list_pipelines(request: Request):
    registry = request.app.state.registry
    pipelines = registry.catalogue()
    for entry in pipelines:
        trainer = registry.get(entry["id"])
        entry["unavailable_reason"] = getattr(trainer, "unavailable_reason", None)
    return {"pipelines": pipelines}


@router.get("/stock-models")
def list_stock_models(request: Request):
    """Stock classifier packages a classifier job can compare against (``stock_model``)."""
    return {"models": list_stock_packages(request.app.state.settings.weights_dir)}


# --------------------------------------------------------------------------- jobs


@router.post("/jobs", status_code=201)
def submit_job(body: JobIn, request: Request):
    trainer = request.app.state.registry.get(body.pipeline)
    if trainer is not None and getattr(trainer, "internal", False):
        trainer = None  # Data Preparation is queued by snapshots, never submitted directly
    if trainer is None:
        if body.pipeline in PLANNED_PIPELINES:
            raise HTTPException(
                422, f"pipeline {body.pipeline!r} is planned but not available yet"
            )
        raise HTTPException(422, f"unknown pipeline {body.pipeline!r}")
    with connect(request.app.state.settings.database_url) as conn:
        snap = conn.execute(
            "SELECT s.id, d.task FROM snapshots s JOIN datasets d ON d.id = s.dataset_id WHERE s.id = %s",
            (body.snapshot_id,),
        ).fetchone()
    if snap is None:
        raise HTTPException(422, f"snapshot {body.snapshot_id} not found")
    if snap["task"] != trainer.task:
        raise HTTPException(
            422,
            f"pipeline {trainer.id!r} trains {trainer.task} models but snapshot "
            f"{body.snapshot_id} is a {snap['task']} dataset",
        )
    if trainer.id == "dinov3_linear":
        with connect(request.app.state.settings.database_url) as conn:
            row = conn.execute(
                "SELECT features FROM snapshots WHERE id = %s", (body.snapshot_id,)
            ).fetchone()
            prepared = features_mod.snapshot_state(
                conn, request.app.state.settings.data_dir, body.snapshot_id, row["features"]
            )
        if prepared["state"] != "ready":
            raise HTTPException(
                422,
                f"dinov3_linear trains on prepared DINO features, but snapshot "
                f"{body.snapshot_id} is {prepared['state']}: create the snapshot with Data "
                "Preparation (prepare_features) and wait until it is ready",
            )
    bad = find_non_finite(body.params)
    if bad is not None:
        raise HTTPException(
            422, f"invalid params: {bad} is not a finite number (NaN/Infinity)"
        )
    try:
        params = trainer.validate_params(dict(body.params))
    except (ValueError, OverflowError) as exc:
        raise HTTPException(422, f"invalid params: {exc}") from exc
    bad = find_non_finite(params)
    if bad is not None:
        raise HTTPException(
            422, f"invalid params: {bad} is not a finite number (NaN/Infinity)"
        )
    # An omitted ``model`` follows the base/parent architecture instead of the pipeline default;
    # an explicit one must match it.
    adopt_arch = trainer.id == "conventional_classifier" and "model" not in body.params
    start_arch = None
    if params.get("base_model_id") is not None:
        if not trainer.supports_base_model:
            raise HTTPException(
                422,
                f"invalid params: base_model_id is not supported by {trainer.id} "
                "(this pipeline cannot warm-start from a base model)",
            )
        try:
            row = check_base_model(
                request.app.state.settings.database_url,
                params["base_model_id"],
                trainer.id,
                None if adopt_arch else params.get("model"),
            )
        except ValueError as exc:
            raise HTTPException(422, f"invalid params: {exc}") from exc
        start_arch = row["model_arch"]
    if params.get("parent_model_id") is not None:
        if not trainer.supports_base_model:
            raise HTTPException(
                422,
                f"invalid params: parent_model_id is not supported by {trainer.id} "
                "(this pipeline cannot start from a finished model)",
            )
        try:
            row = check_parent_model(
                request.app.state.settings,
                params["parent_model_id"],
                trainer.id,
                {k: v for k, v in params.items() if k != "model"} if adopt_arch else params,
                body.snapshot_id,
            )
        except ValueError as exc:
            raise HTTPException(422, f"invalid params: {exc}") from exc
        start_arch = row.get("model_arch")
    if adopt_arch and start_arch and start_arch != params.get("model"):
        try:
            params = trainer.validate_params({**body.params, "model": start_arch})
        except (ValueError, OverflowError) as exc:
            raise HTTPException(422, f"invalid params: {exc}") from exc
    job = _queue(request).enqueue(
        trainer.id,
        body.snapshot_id,
        params,
        _requires(request, trainer.id, trainer.requires),
        submitted_by=principal_name(request),
    )
    audit(request, "job.submit", f"job:{job.id}", {"pipeline": trainer.id})
    return asdict(job)


@router.get("/jobs")
def list_jobs(
    request: Request,
    state: Optional[str] = None,
    limit: int = 100,
    workspace_id: Optional[int] = None,
):
    limit = max(1, min(limit, 1000))
    with connect(request.app.state.settings.database_url) as conn:
        rows = conn.execute(
            "SELECT j.*, d.id AS dataset_id, d.name AS dataset_name, d.workspace_id"
            f" FROM jobs j {_DATASET_JOIN}"
            " WHERE (%s::text IS NULL OR j.state = %s)"
            " AND (%s::bigint IS NULL OR d.workspace_id = %s)"
            " ORDER BY j.id DESC LIMIT %s",
            (state, state, workspace_id, workspace_id, limit),
        ).fetchall()
    return {"jobs": _with_queue(request, rows)}


@router.get("/jobs/{job_id}")
def get_job(job_id: int, request: Request):
    job = _queue(request).get(job_id)
    if job is None:
        raise HTTPException(404, f"job {job_id} not found")
    return _with_queue(request, [asdict(job)])[0]


@router.get("/jobs/{job_id}/events")
def job_events(job_id: int, request: Request):
    with connect(request.app.state.settings.database_url) as conn:
        if (
            conn.execute("SELECT 1 FROM jobs WHERE id = %s", (job_id,)).fetchone()
            is None
        ):
            raise HTTPException(404, f"job {job_id} not found")
        rows = conn.execute(
            "SELECT seq, stage, message, metrics, created_at FROM job_events"
            " WHERE job_id = %s ORDER BY seq",
            (job_id,),
        ).fetchall()
    return {"events": rows}


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: int, request: Request):
    queue = _queue(request)
    job = queue.request_cancel(job_id)
    if job is None:
        existing = queue.get(job_id)
        if existing is None:
            raise HTTPException(404, f"job {job_id} not found")
        raise HTTPException(409, f"job {job_id} is already {existing.state}")
    audit(request, "job.cancel", f"job:{job_id}")
    return asdict(job)


@router.post("/jobs/{job_id}/retry", status_code=201)
def retry_job(job_id: int, request: Request):
    queue = _queue(request)
    job = queue.get(job_id)
    if job is None:
        raise HTTPException(404, f"job {job_id} not found")
    if job.state not in ("failed", "cancelled"):
        raise HTTPException(
            409,
            f"job {job_id} is {job.state}; only failed or cancelled jobs can be retried",
        )
    try:
        # Requirements are recomputed so a retry follows the current SPARROW_FT_GPU_PIPELINES.
        trainer = request.app.state.registry.get(job.pipeline)
        declared = trainer.requires if trainer is not None else job.requires
        new = queue.enqueue(
            job.pipeline,
            job.snapshot_id,
            job.params,
            _requires(request, job.pipeline, declared),
            retry_of=job_id,
            submitted_by=principal_name(request),
        )
    except AlreadyRetried as exc:
        raise HTTPException(409, str(exc)) from None
    audit(request, "job.retry", f"job:{new.id}", {"retry_of": job_id})
    return asdict(new)


def _sse(event: str, data, event_id: Optional[int] = None) -> str:
    payload = json.dumps(sanitise_non_finite(jsonable_encoder(data)))
    head = f"id: {event_id}\n" if event_id is not None else ""
    return f"{head}event: {event}\ndata: {payload}\n\n"


def _poll_events(conn, job_id: int, after: int) -> tuple[list, Optional[str]]:
    job = conn.execute("SELECT state FROM jobs WHERE id = %s", (job_id,)).fetchone()
    rows = conn.execute(
        "SELECT seq, stage, message, metrics, created_at FROM job_events"
        " WHERE job_id = %s AND seq > %s ORDER BY seq",
        (job_id, after),
    ).fetchall()
    return rows, (job["state"] if job else None)


class _StreamConn:
    """One autocommit connection per SSE stream, safe to close while a poll is running.

    ``poll`` runs in a worker thread while ``close`` may be called from the event loop (client
    disconnect, response teardown). Closing a connection under a running query is unsafe, so a
    close requested mid-poll is deferred until that poll returns; a poll after close raises.
    """

    def __init__(self, database_url: str):
        self.conn = connect(database_url)
        # Autocommit: each poll sees fresh data and no transaction stays open between polls.
        self.conn.autocommit = True
        self._lock = threading.Lock()
        self._busy = False
        self._closed = False

    def poll(self, job_id: int, after: int) -> tuple[list, Optional[str]]:
        with self._lock:
            if self._closed:
                raise RuntimeError("SSE stream connection is closed")
            self._busy = True
        try:
            return _poll_events(self.conn, job_id, after)
        finally:
            with self._lock:
                self._busy = False
                close_now = self._closed
            if close_now:
                self.conn.close()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            close_now = not self._busy
        if close_now:
            self.conn.close()


@router.get("/jobs/{job_id}/stream")
async def stream_job(
    job_id: int, request: Request, after: Optional[int] = None, poll: float = 0.5
):
    url = request.app.state.settings.database_url
    last = request.headers.get("last-event-id")
    cursor = after if after is not None else 0
    if last is not None:
        try:
            cursor = int(last)
        except ValueError:
            raise HTTPException(400, f"invalid Last-Event-ID: {last!r}") from None
    if not math.isfinite(poll):
        raise HTTPException(422, f"poll must be a finite number of seconds, got {poll}")
    interval = min(max(poll, 0.05), 5.0)
    # One connection for the whole stream instead of a new one per poll.
    stream = await run_in_threadpool(_StreamConn, url)
    try:
        rows, state = await run_in_threadpool(stream.poll, job_id, cursor)
    except BaseException:
        stream.close()
        raise
    if state is None:
        stream.close()
        raise HTTPException(404, f"job {job_id} not found")

    async def gen():
        nonlocal rows, state, cursor
        sent_state = None
        try:
            while True:
                if state != sent_state:
                    # No id: a state change must not move the Last-Event-ID resume cursor.
                    sent_state = state
                    yield _sse("job-state", {"state": state})
                for row in rows:
                    cursor = row["seq"]
                    yield _sse("job-event", row, row["seq"])
                if state in TERMINAL_STATES:
                    # A terminal state is written after the last event; one more read drains it.
                    rows, _ = await run_in_threadpool(stream.poll, job_id, cursor)
                    if not rows:
                        yield _sse("end", {"state": state})
                        return
                    continue
                if await request.is_disconnected():
                    return
                await asyncio.sleep(interval)
                rows, state = await run_in_threadpool(stream.poll, job_id, cursor)
        finally:
            stream.close()

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        # Closes the connection even if the generator never started; close() is idempotent
        # and waits for an in-flight poll instead of closing under it.
        background=BackgroundTask(stream.close),
    )


# --------------------------------------------------------------------------- models


@router.get("/models")
def list_models(
    request: Request, job_id: Optional[int] = None, workspace_id: Optional[int] = None
):
    columns = ", ".join(f"j.{c.strip()}" for c in _MODEL_COLUMNS.split(","))
    with connect(request.app.state.settings.database_url) as conn:
        rows = conn.execute(
            f"SELECT {columns}, j.object_key, d.id AS dataset_id, d.name AS dataset_name, d.workspace_id"
            f" FROM model_versions j {_DATASET_JOIN}"
            " WHERE (%s::bigint IS NULL OR j.job_id = %s)"
            " AND (%s::bigint IS NULL OR d.workspace_id = %s) ORDER BY j.id",
            (job_id, job_id, workspace_id, workspace_id),
        ).fetchall()
        active = _active_deployments(conn)
    objects_dir = request.app.state.settings.objects_dir
    for row in rows:
        tag_model(objects_dir, row)
        row.pop("object_key", None)
        row["active_deployments"] = active.get(row["id"], [])
        row["stock_comparison"] = stock_comparison(row["metrics"], row["metadata"])
        row["reference_comparison"] = reference_comparison(row["metrics"])
    return {"models": rows}


def _active_deployments(conn, model_id: Optional[int] = None) -> dict[int, list]:
    active: dict[int, list] = {}
    for d in conn.execute(
        "SELECT model_id, consumer, project, external_id FROM deployments"
        " WHERE deactivated_at IS NULL AND (%s::bigint IS NULL OR model_id = %s)"
        " ORDER BY consumer, project",
        (model_id, model_id),
    ).fetchall():
        active.setdefault(d["model_id"], []).append(
            {
                "consumer": d["consumer"],
                "project": d["project"],
                "external_id": d["external_id"],
            }
        )
    return active


def _model_or_404(request: Request, model_id: int, columns: str) -> dict:
    with connect(request.app.state.settings.database_url) as conn:
        row = conn.execute(
            f"SELECT {columns} FROM model_versions WHERE id = %s", (model_id,)
        ).fetchone()
    if row is None:
        raise HTTPException(404, f"model {model_id} not found")
    return row


@router.get("/models/{model_id}")
def get_model(model_id: int, request: Request):
    row = _model_or_404(request, model_id, f"{_MODEL_COLUMNS}, object_key")
    tag_model(request.app.state.settings.objects_dir, row)
    row.pop("object_key", None)
    with connect(request.app.state.settings.database_url) as conn:
        row["active_deployments"] = _active_deployments(conn, model_id).get(
            model_id, []
        )
    row["stock_comparison"] = stock_comparison(row["metrics"], row["metadata"])
    row["reference_comparison"] = reference_comparison(row["metrics"])
    return row


@router.get("/models/{model_id}/reuse")
def preview_reuse(model_id: int, snapshot_id: int, request: Request):
    """What a job on ``snapshot_id`` with ``parent_model_id=model_id`` would reuse."""
    try:
        preview = reuse_preview(request.app.state.settings, model_id, snapshot_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    if preview is None:
        raise HTTPException(404, f"model {model_id} not found")
    return preview


class ModelPatch(BaseModel):
    # Required key; null or blank resets to the default "Model #<id>".
    name: Optional[str] = Field(..., max_length=MODEL_NAME_MAX)


@router.patch("/models/{model_id}")
def rename_model(model_id: int, body: ModelPatch, request: Request):
    name = (body.name or "").strip() or None
    with connect(request.app.state.settings.database_url) as conn:
        updated = conn.execute(
            "UPDATE model_versions SET name = %s WHERE id = %s RETURNING id",
            (name, model_id),
        ).fetchone()
        if updated is None:
            raise HTTPException(404, f"model {model_id} not found")
        record_audit(
            conn,
            principal_name(request),
            "model.rename",
            str(model_id),
            {"model_id": model_id, "name": name},
        )
    return get_model(model_id, request)


@router.get("/models/{model_id}/artifact")
def download_artifact(model_id: int, request: Request):
    row = _model_or_404(request, model_id, "id, object_key, format")
    store = LocalObjectStore(request.app.state.settings.objects_dir)
    if not store.exists(row["object_key"]):
        raise HTTPException(
            500, f"artifact for model {model_id} is missing from the object store"
        )
    if_none_match = request.headers.get("if-none-match")
    if if_none_match is not None and "*" in (
        tag.strip() for tag in if_none_match.split(",")
    ):
        # RFC 9110 13.1.2: "*" matches any current representation.
        return Response(status_code=304)
    path = store.path(row["object_key"])
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=f"model-{model_id}-{path.name}",
    )


# --------------------------------------------------------------------------- deployments


class ActivateIn(BaseModel):
    consumer: Literal["local", "studio"]
    project: str = Field(min_length=1, max_length=200)
    # Activate even though the evaluation found the stock model better (FT-16).
    confirm: bool = False
    # Studio only: the detector pipeline a classifier runs on (FT-43). None keeps the
    # project's current detector.
    detector_pipeline_id: Optional[int] = Field(None, ge=1)


def _number(value) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def stock_comparison(metrics, metadata) -> Optional[dict]:
    """The stock-vs-fine-tuned comparison of a classifier version, or None if it was not run."""
    metrics = metrics if isinstance(metrics, dict) else {}
    metadata = metadata if isinstance(metadata, dict) else {}
    worse = _number(metrics.get("stock_worse"))
    if worse is None:
        return None
    crops = _number(metrics.get("stock_compared_crops"))
    model = metadata.get("stock_model")
    # Models trained before stock_min_val_crops existed keep blocking at any size (minimum 0).
    min_val = _number(metrics.get("stock_min_val_crops")) or 0.0
    return {
        "model": model if isinstance(model, str) else None,
        "stock_macro_f1": _number(metrics.get("stock_val_macro_f1")),
        "finetuned_macro_f1": _number(metrics.get("stock_finetuned_val_macro_f1")),
        "crops": int(crops) if crops is not None else None,
        "worse": worse >= 0.5,
        "min_val_crops": int(min_val),
        # Activation is blocked on ``worse`` only when the val split reached the minimum.
        "enforced": crops is None or crops >= min_val,
    }


def reference_comparison(metrics) -> Optional[dict]:
    """The dinov3 head vs its logistic-regression reference (FT-19), or None if not run.

    ``worse`` is set when the head trailed the reference by more than the margin on a validation
    set too small for the gate to fail the job; it is a warning, activation is not gated.
    """
    metrics = metrics if isinstance(metrics, dict) else {}
    worse = _number(metrics.get("reference_worse"))
    if worse is None:
        return None
    crops = _number(metrics.get("val_crops"))
    return {
        "reference_macro_f1": _number(metrics.get("reference_val_macro_f1")),
        "finetuned_macro_f1": _number(metrics.get("val_macro_f1")),
        "crops": int(crops) if crops is not None else None,
        "worse": worse >= 0.5,
    }


def _f1(value: Optional[float]) -> str:
    return "unknown" if value is None else f"{value:.2f}"


def stock_warning(comparison: dict) -> str:
    crops = comparison["crops"]
    on = (
        f" on {crops} validation crops"
        if crops is not None
        else " on the validation crops"
    )
    return (
        f"The stock model {comparison['model'] or '(unnamed)'} scored macro-F1 "
        f"{_f1(comparison['stock_macro_f1'])}{on}, better than this fine-tuned model "
        f"({_f1(comparison['finetuned_macro_f1'])}). Activate anyway?"
    )


_DEPLOYMENT_COLUMNS = (
    "d.id, d.model_id, m.pipeline, d.consumer, d.project, d.activated_at, d.activated_by,"
    " d.deactivated_at, d.external_id"
)


@router.post("/models/{model_id}/activate")
def activate_model(model_id: int, body: ActivateIn, request: Request):
    project = body.project.strip()
    if not project:
        raise HTTPException(422, "project must not be empty")
    if body.detector_pipeline_id is not None and body.consumer != "studio":
        raise HTTPException(422, "detector_pipeline_id is only for Studio activations")
    model = _model_or_404(request, model_id, "id, snapshot_id, metrics, metadata")
    if body.detector_pipeline_id is not None:
        with connect(request.app.state.settings.database_url) as conn:
            task = conn.execute(
                "SELECT d.task FROM snapshots s JOIN datasets d ON d.id = s.dataset_id "
                "WHERE s.id = %s",
                (model["snapshot_id"],),
            ).fetchone()
        if task is not None and task["task"] == "detection":
            raise HTTPException(
                422, "a detector model is not installed on another detector"
            )
    comparison = stock_comparison(model["metrics"], model["metadata"])
    override = comparison is not None and comparison["worse"] and comparison["enforced"]
    if override and not body.confirm:
        raise HTTPException(
            409,
            {
                "code": "stock_model_better",
                "message": stock_warning(comparison),
                "stock_comparison": comparison,
            },
        )
    external_id = None
    studio_install = None
    if body.consumer == "studio":
        from ..connectors.studio import push_model

        pushed = push_model(
            request,
            model_id,
            project,
            activate=True,
            detector_pipeline_id=body.detector_pipeline_id,
        )
        external_id = pushed["studio_model_id"]
        studio_install = pushed["install"]
    principal = principal_name(request)
    with connect(request.app.state.settings.database_url) as conn:
        # Serialise activations of the same (consumer, project) so the partial unique
        # index never sees two active rows from concurrent requests.
        conn.execute(
            "SELECT pg_advisory_xact_lock(hashtext(%s))",
            (f"{body.consumer}/{project}",),
        )
        conn.execute(
            "UPDATE deployments SET deactivated_at = now()"
            " WHERE consumer = %s AND project = %s AND deactivated_at IS NULL",
            (body.consumer, project),
        )
        dep_id = conn.execute(
            "INSERT INTO deployments (model_id, consumer, project, activated_by, external_id)"
            " VALUES (%s, %s, %s, %s, %s) RETURNING id",
            (model_id, body.consumer, project, principal, external_id),
        ).fetchone()["id"]
        record_audit(
            conn,
            principal,
            "model.activate",
            f"model:{model_id}",
            {
                "consumer": body.consumer,
                "project": project,
                "external_id": external_id,
                "studio_install": studio_install,
                **(
                    {"stock_override": True, "stock_comparison": comparison}
                    if override
                    else {}
                ),
            },
        )
        row = conn.execute(
            f"SELECT {_DEPLOYMENT_COLUMNS} FROM deployments d"
            " JOIN model_versions m ON m.id = d.model_id WHERE d.id = %s",
            (dep_id,),
        ).fetchone()
    if body.consumer == "studio":
        row = {**row, "studio_install": studio_install}
    return row


@router.get("/deployments")
def list_deployments(
    request: Request,
    consumer: Optional[str] = None,
    project: Optional[str] = None,
    active: Optional[bool] = None,
):
    with connect(request.app.state.settings.database_url) as conn:
        rows = conn.execute(
            f"SELECT {_DEPLOYMENT_COLUMNS} FROM deployments d"
            " JOIN model_versions m ON m.id = d.model_id"
            " WHERE (%(consumer)s::text IS NULL OR d.consumer = %(consumer)s)"
            "   AND (%(project)s::text IS NULL OR d.project = %(project)s)"
            "   AND (%(active)s::boolean IS NULL OR (d.deactivated_at IS NULL) = %(active)s)"
            " ORDER BY d.id DESC",
            {"consumer": consumer, "project": project, "active": active},
        ).fetchall()
    return {"deployments": rows}
