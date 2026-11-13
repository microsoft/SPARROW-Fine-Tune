"""Worker: claim one job, run its trainer stage by stage, store the exported model."""

from __future__ import annotations

import logging
import shutil
import tarfile
import threading
import time
from pathlib import Path, PurePosixPath
from typing import Iterable, Optional

from . import base_models
from .config import Settings
from .db import connect
from .jobs import TERMINAL_STATES, CancelRequested, Job, JobQueue, LeaseLost
from .jsonsafe import sanitise_non_finite
from .pipelines import (
    ExportResult,
    JobCancelled,
    JobContext,
    Registry,
    SnapshotItem,
    Trainer,
    default_registry,
)
from .storage import LocalObjectStore

log = logging.getLogger(__name__)


class _Heartbeat(threading.Thread):
    """Extends the lease in the background and records cancel requests / lease loss."""

    def __init__(
        self,
        queue: JobQueue,
        job_id: int,
        worker_id: str,
        lease_seconds: float,
        attempt: Optional[int] = None,
        claimed_at: Optional[float] = None,
    ) -> None:
        super().__init__(name=f"heartbeat-job-{job_id}", daemon=True)
        # Fenced by the claim's attempt: a heartbeat left over from an earlier attempt of
        # the same worker id must not extend the lease of a newer one.
        self.queue = queue if attempt is None else queue.fenced(attempt)
        self.job_id, self.worker_id = job_id, worker_id
        self.lease_seconds = lease_seconds
        self.interval = max(0.2, lease_seconds / 3)
        self.cancel_requested = threading.Event()
        self.lease_lost = threading.Event()
        self._halt = threading.Event()
        # Local deadline: the lease runs from the claim, so count from just before it was
        # requested (``claimed_at``), not from whenever this thread object is built.
        self._last_ok = time.monotonic() if claimed_at is None else claimed_at

    def beat(self) -> None:
        if self.lease_lost.is_set():
            return
        started = time.monotonic()
        try:
            if self.queue.heartbeat(self.job_id, self.worker_id, self.lease_seconds):
                self.cancel_requested.set()
            self._last_ok = started
        except LeaseLost:
            self.lease_lost.set()
        except Exception:  # transient database error: retry on the next beat
            log.warning("heartbeat for job %s failed", self.job_id, exc_info=True)
            if time.monotonic() - self._last_ok > self.lease_seconds:
                # The lease has expired by now whatever the database says: another worker
                # may already hold the job, so stop writing to it.
                log.warning(
                    "job %s: no successful heartbeat within the lease", self.job_id
                )
                self.lease_lost.set()

    def run(self) -> None:
        while not self._halt.wait(self.interval) and not self.lease_lost.is_set():
            self.beat()

    def check(self) -> None:
        if (
            not self.lease_lost.is_set()
            and time.monotonic() - self._last_ok > self.lease_seconds
        ):
            self.lease_lost.set()
        if self.lease_lost.is_set():
            raise LeaseLost(f"lease on job {self.job_id} lost")
        if self.cancel_requested.is_set():
            raise JobCancelled(f"job {self.job_id} cancelled")

    def stop(self) -> None:
        self._halt.set()
        if self.is_alive():
            self.join(timeout=self.interval + 5)


def _staged_name(sha256: str, image_name: str) -> str:
    """``<sha256><ext>``: content-addressed, keeping the extension trainers dispatch on."""
    suffix = PurePosixPath(image_name).suffix.lower()
    if not (2 <= len(suffix) <= 11 and suffix[1:].isascii() and suffix[1:].isalnum()):
        suffix = ""
    return f"{sha256}{suffix}"


def _attempt_dir(settings: Settings, job: Job) -> Path:
    # Per-attempt: a stale worker whose lease was reaped never shares files with the new owner.
    return settings.jobs_dir / str(job.id) / f"attempt-{job.attempts}"


def _staged_job_dir(settings: Settings, job_id: int) -> Path:
    # Per job, shared by its attempts; removed once the job is terminal.
    return settings.staged_dir / f"job-{job_id}"


# How long after a job finished its leftover ``attempt-N`` dirs are kept (see below).
STALE_ATTEMPT_GRACE_S = 3600


def _prune_finished_job_dirs(settings: Settings) -> None:
    """Remove dirs of jobs that are terminal (or gone): ``staged/job-<id>`` trees and
    ``jobs/<id>`` dirs.

    Runs before a claim, not in the job's own cleanup: by then no attempt of a terminal job
    can start, so nothing will create an attempt dir under ``jobs/<id>`` again. A
    ``jobs/<id>`` still holding an attempt dir is kept for ``STALE_ATTEMPT_GRACE_S`` after the
    job finished (a superseded worker may still be shutting down in it), then removed: a
    killed worker never runs its own cleanup. Dirs of jobs that no longer exist go at once.
    """
    jobs = {}
    if settings.jobs_dir.is_dir():
        for path in settings.jobs_dir.iterdir():
            if path.name.isdigit() and path.is_dir():
                jobs.setdefault(int(path.name), []).append((path, False))
    if settings.staged_dir.is_dir():
        for path in settings.staged_dir.iterdir():
            name = path.name
            if name.startswith("job-") and name[4:].isdigit() and path.is_dir():
                jobs.setdefault(int(name[4:]), []).append((path, True))
    if not jobs:
        return
    with connect(settings.database_url) as conn:
        rows = conn.execute(
            "SELECT id, state <> ALL(%s) AS active,"
            " (finished_at IS NULL OR finished_at < now() - make_interval(secs => %s))"
            " AS settled FROM jobs WHERE id = ANY(%s)",
            (list(TERMINAL_STATES), float(STALE_ATTEMPT_GRACE_S), sorted(jobs)),
        ).fetchall()
    known = {row["id"]: row for row in rows}
    for job_id, paths in jobs.items():
        row = known.get(job_id)
        if row is not None and row["active"]:
            continue
        settled = row is None or row["settled"]
        for path, staged in paths:
            if staged or settled:
                shutil.rmtree(path, ignore_errors=True)
            else:
                try:
                    path.rmdir()
                except (
                    OSError
                ):  # not empty (a stale attempt still runs) or already gone
                    pass


def _prune_legacy_staged(settings: Settings) -> None:
    """Remove ``staged/<sha[:2]>`` dirs of the old shared layout: nothing reads them now."""
    if not settings.staged_dir.is_dir():
        return
    for path in settings.staged_dir.iterdir():
        name = path.name
        if (
            len(name) == 2
            and all(c in "0123456789abcdef" for c in name)
            and path.is_dir()
        ):
            shutil.rmtree(path, ignore_errors=True)


def _remove_older_attempts(settings: Settings, job: Job) -> None:
    """Delete dirs of earlier attempts: a killed worker never ran its own cleanup.

    Earlier attempts have lost their lease, so their workers cannot record a result.
    """
    for k in range(1, job.attempts):
        shutil.rmtree(
            settings.jobs_dir / str(job.id) / f"attempt-{k}", ignore_errors=True
        )


def _load_items(settings: Settings, job: Job) -> list[SnapshotItem]:
    """Snapshot items with ``image_path`` under ``staged/job-<id>``.

    Staged views are read-only hard links (or copies) of the immutable, content-addressed
    snapshot images, shared by the job's attempts; they are removed once the job is terminal.
    """
    store = LocalObjectStore(settings.objects_dir)
    staged = _staged_job_dir(settings, job.id)
    with connect(settings.database_url) as conn:
        rows = conn.execute(
            "SELECT * FROM snapshot_items WHERE snapshot_id = %s ORDER BY position",
            (job.snapshot_id,),
        ).fetchall()
    items = []
    for row in rows:
        sha = row["sha256"]
        dest = staged / sha[:2] / _staged_name(sha, row["image_name"])
        store.link_to(row["object_key"], dest)
        items.append(
            SnapshotItem(
                image_path=dest,
                width=row["width"],
                height=row["height"],
                annotations=[dict(a) for a in row["annotations"]],
                group=row["group_name"],
                name=row["image_name"],
                sha256=row["sha256"],
            )
        )
    return items


def _store_artifact(
    settings: Settings, job: Job, result: ExportResult
) -> tuple[str, str]:
    path = Path(result.path)
    if path.is_dir():
        archive = _attempt_dir(settings, job) / f"{path.name or 'model'}.tar"
        with tarfile.open(archive, "w") as tar:
            tar.add(path, arcname=path.name or "model")
        path = archive
    elif not path.is_file():
        raise FileNotFoundError(f"exported artifact not found: {path}")
    key = f"models/job-{job.id}/attempt-{job.attempts}/{path.name}"
    return key, LocalObjectStore(settings.objects_dir).put_file(key, path)


def _run_job(
    settings: Settings,
    queue: JobQueue,
    job: Job,
    trainer: Trainer,
    worker_id: str,
    lease_seconds: float,
    claimed_at: Optional[float] = None,
) -> None:
    heart = _Heartbeat(
        queue, job.id, worker_id, lease_seconds, job.attempts, claimed_at
    )
    heart.start()

    def event(
        stage_name: str, message: str = "", metrics: Optional[dict] = None
    ) -> None:
        queue.add_event(
            job.id,
            stage_name,
            message,
            metrics,
            worker_id=worker_id,
            attempt=job.attempts,
        )

    def stage(name: str) -> None:
        heart.beat()
        heart.check()
        event(name, "started")

    try:
        try:
            stage("prepare")
            job_dir = _attempt_dir(settings, job)
            _remove_older_attempts(settings, job)
            items = _load_items(settings, job)
            labels = sorted({a["label"] for item in items for a in item.annotations})
            ctx = JobContext(
                job_id=job.id,
                params=trainer.validate_params(job.params),
                workdir=job_dir / "work",
                items=items,
                labels=labels,
                _emit=event,
                _check_cancelled=heart.check,
            )
            ctx.workdir.mkdir(parents=True, exist_ok=True)
            if ctx.params.get("base_model_id") is not None:
                ctx.base_model = base_models.stage(
                    settings, ctx.params["base_model_id"], ctx.workdir
                )
            trainer.prepare(ctx)

            stage("train")
            trainer.train(ctx)

            stage("evaluate")
            # NaN/Infinity are legal floats but not JSON: record them as null.
            metrics = sanitise_non_finite(
                {k: float(v) for k, v in (trainer.evaluate(ctx) or {}).items()}
            )
            event("evaluate", "metrics", metrics)

            stage("export")
            result = trainer.export(ctx)
            heart.check()
            # The heartbeat thread may be failing transiently while the lease expires:
            # confirm with the database that this attempt still owns the job.
            queue.verify_lease(job.id, worker_id, job.attempts)
            object_key, sha256 = _store_artifact(settings, job, result)
            try:
                heart.beat()
                heart.check()
                model_id = queue.complete(
                    job.id,
                    worker_id,
                    {
                        "metrics": metrics,
                        "format": result.format,
                        "object_key": object_key,
                        "sha256": sha256,
                        "metadata": dict(result.metadata or {}),
                    },
                    attempt=job.attempts,
                )
            except (LeaseLost, CancelRequested, JobCancelled):
                # The key is unique to this attempt, so no other owner references it.
                LocalObjectStore(settings.objects_dir).delete(object_key)
                raise
            event("succeeded", f"model version {model_id} stored")
        except (JobCancelled, CancelRequested):
            queue.mark_cancelled(job.id, worker_id, attempt=job.attempts)
            event("cancelled", "job cancelled")
        except LeaseLost:
            raise
        except Exception as exc:
            log.exception("job %s failed", job.id)
            error = str(exc) or type(exc).__name__
            queue.fail(job.id, worker_id, error, attempt=job.attempts)
            event("failed", error)
    except LeaseLost:
        log.warning("job %s: lease lost; abandoning without writing a result", job.id)
    finally:
        heart.stop()
        # Only this attempt's directory: another attempt's worker may still be using its own.
        # The parent jobs/<id> dir is pruned by a later sweep (_prune_finished_job_dirs).
        # staged/job-<id> is pruned by the same sweep once the job is terminal.
        shutil.rmtree(_attempt_dir(settings, job), ignore_errors=True)


def claimable_pipelines(registry: Registry) -> list[str]:
    """Registered pipeline ids this worker can run; built-ins missing a setting are left out."""
    return [
        pid
        for pid in registry.ids()
        if not getattr(registry.get(pid), "unavailable_reason", None)
    ]


def run_once(
    settings: Settings,
    worker_id: str,
    resources: Iterable[str],
    registry: Optional[Registry] = None,
    lease_seconds: float = 60,
) -> bool:
    """Claim and run at most one job; return False when nothing was claimable."""
    registry = default_registry if registry is None else registry
    queue = JobQueue(settings.database_url)
    resources = set(resources)
    pipelines = claimable_pipelines(registry)
    queue.register_worker(worker_id, resources, pipelines)
    queue.reap_expired()
    try:
        _prune_finished_job_dirs(settings)
    except Exception:  # housekeeping only: never stop the worker from claiming
        log.warning("could not prune dirs of finished jobs", exc_info=True)
    # Before the claim: the lease the database sets runs from a moment after this.
    claimed_at = time.monotonic()
    job = queue.claim(
        worker_id,
        resources,
        lease_seconds,
        pipelines=pipelines,
        max_running_per_user=settings.max_running_per_user,
    )
    if job is None:
        return False
    trainer = registry.get(job.pipeline)
    if trainer is None:  # registry changed between claim and lookup
        queue.fail(
            job.id,
            worker_id,
            f"pipeline {job.pipeline!r} is not available on this worker",
            attempt=job.attempts,
        )
        return True
    _run_job(settings, queue, job, trainer, worker_id, lease_seconds, claimed_at)
    return True


def run_forever(
    settings: Settings,
    worker_id: str,
    resources: Iterable[str],
    registry: Optional[Registry] = None,
    poll_seconds: float = 2.0,
    lease_seconds: float = 60,
    stop: Optional[threading.Event] = None,
) -> None:
    stop = stop or threading.Event()
    resources = set(resources)
    reg = default_registry if registry is None else registry
    for pid in reg.ids():
        reason = getattr(reg.get(pid), "unavailable_reason", None)
        if reason:
            log.warning("not claiming %s jobs: %s", pid, reason)
    for root in (settings.objects_dir, settings.staged_dir):
        try:
            LocalObjectStore(root).remove_stale_tmp()
        except OSError:
            log.warning(
                "could not sweep stale temporary files under %s", root, exc_info=True
            )
    try:
        _prune_legacy_staged(settings)
        _prune_finished_job_dirs(settings)
    except Exception:
        log.warning("could not prune staged/job dirs at start", exc_info=True)
    while not stop.is_set():
        try:
            claimed = run_once(settings, worker_id, resources, registry, lease_seconds)
        except Exception:
            log.exception("worker loop error")
            claimed = False
        if not claimed:
            stop.wait(poll_seconds)
