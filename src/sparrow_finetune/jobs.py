"""PostgreSQL job queue: leased claims with ``FOR UPDATE SKIP LOCKED``, heartbeats and reaping.

Every state change made by a worker is guarded by ``lease_owner`` so a worker whose
lease was reaped (and possibly re-claimed elsewhere) cannot overwrite the job.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime
from typing import Any, Iterable, Optional

from psycopg.types.json import Jsonb

from .db import _LOCK_NAMESPACE, connect
from .jsonsafe import find_non_finite, sanitise_non_finite

TERMINAL_STATES = ("succeeded", "failed", "cancelled")


class CancelRequested(Exception):
    """``complete`` refused: cancellation was requested while the attempt still held the lease."""


class LeaseLost(Exception):
    """The worker no longer holds the lease on the job (reaped, re-claimed or finished)."""


@dataclass(frozen=True)
class Job:
    id: int
    pipeline: str
    snapshot_id: int
    params: dict
    requires: list
    state: str
    attempts: int
    max_attempts: int
    lease_owner: Optional[str]
    lease_expires_at: Optional[datetime]
    cancel_requested_at: Optional[datetime]
    error: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    submitted_by: Optional[str] = None

    @classmethod
    def from_row(cls, row: dict) -> "Job":
        return cls(**{f.name: row[f.name] for f in fields(cls)})


class AlreadyRetried(Exception):
    def __init__(self, job_id: int, retry_id: int):
        super().__init__(f"job {job_id} was already retried as job {retry_id}")
        self.job_id, self.retry_id = job_id, retry_id


# Among claimable jobs, the submitter with the fewest running jobs goes first, then the oldest
# job: when capacity frees up it rotates between users instead of draining one user's backlog.
# A submitter at the per-user cap (``max_per_user`` > 0) is skipped; the caller then holds
# ``_CLAIM_LOCK`` so two workers cannot both see the same running count.
_CLAIM_SQL = """
UPDATE jobs
SET state = 'running',
    lease_owner = %(worker)s,
    lease_expires_at = clock_timestamp() + make_interval(secs => %(lease)s),
    attempts = attempts + 1,
    started_at = coalesce(started_at, clock_timestamp())
WHERE id = (
    SELECT q.id FROM jobs q
    LEFT JOIN LATERAL (
        SELECT count(*) AS n FROM jobs r
        WHERE r.state = 'running' AND r.submitted_by IS NOT DISTINCT FROM q.submitted_by
    ) running ON true
    WHERE q.state = 'queued'
      AND q.cancel_requested_at IS NULL
      AND q.requires <@ %(resources)s::text[]
      AND (%(pipelines)s::text[] IS NULL OR q.pipeline = ANY(%(pipelines)s::text[]))
      AND (%(max_per_user)s = 0 OR q.submitted_by IS NULL OR running.n < %(max_per_user)s)
    ORDER BY running.n, q.id
    FOR UPDATE OF q SKIP LOCKED
    LIMIT 1
)
RETURNING *
"""

# Transaction-scoped advisory lock serialising capped claims (key 1 is the migration lock).
_CLAIM_LOCK = f"SELECT pg_advisory_xact_lock({_LOCK_NAMESPACE}, 2)"

# A worker counts as online when it polled within this many seconds or holds a live lease.
WORKER_ONLINE_SECONDS = 30

_REAP_SQL = """
UPDATE jobs
SET state = CASE
        WHEN cancel_requested_at IS NOT NULL THEN 'cancelled'
        WHEN attempts >= max_attempts THEN 'failed'
        ELSE 'queued' END,
    error = CASE
        WHEN cancel_requested_at IS NULL AND attempts >= max_attempts
        THEN 'lease expired after ' || attempts || ' attempts'
        ELSE error END,
    finished_at = CASE
        WHEN cancel_requested_at IS NOT NULL OR attempts >= max_attempts THEN clock_timestamp()
        ELSE NULL END,
    lease_owner = NULL,
    lease_expires_at = NULL
WHERE id IN (
    SELECT id FROM jobs
    WHERE state = 'running' AND lease_expires_at < clock_timestamp()
    FOR UPDATE SKIP LOCKED
)
RETURNING id
"""


# One row per queued job in %(ids)s: online workers able to run it (polled recently or holding a
# live lease), how many of those are busy, queued jobs ahead of it that such a worker could take,
# and its submitter's running count.
_QUEUE_STATUS_SQL = """
WITH online AS (
    SELECT w.id, w.resources, w.pipelines,
           EXISTS (SELECT 1 FROM jobs r WHERE r.lease_owner = w.id AND r.state = 'running'
                   AND r.lease_expires_at > clock_timestamp()) AS busy,
           w.last_seen > clock_timestamp() - make_interval(secs => %(online)s) AS polled
    FROM workers w
),
able AS (  -- (queued job, online worker able to run it)
    SELECT j.id AS job_id, o.id AS worker_id, o.busy
    FROM jobs j JOIN online o
      ON (o.busy OR o.polled) AND j.requires <@ o.resources
     AND (o.pipelines IS NULL OR j.pipeline = ANY(o.pipelines))
    WHERE j.state = 'queued' AND j.cancel_requested_at IS NULL
)
SELECT q.id, q.requires, q.submitted_by,
       (SELECT count(*) FROM able b WHERE b.job_id = q.id)::int AS workers,
       (SELECT count(*) FROM able b WHERE b.job_id = q.id AND b.busy)::int AS busy,
       (SELECT count(DISTINCT a.job_id) FROM able a JOIN able b
          ON b.worker_id = a.worker_id AND b.job_id = q.id
        WHERE a.job_id < q.id)::int AS ahead,
       (SELECT count(*) FROM jobs r WHERE r.state = 'running'
          AND r.submitted_by IS NOT DISTINCT FROM q.submitted_by)::int AS user_running
FROM jobs q
WHERE q.id = ANY(%(ids)s::bigint[]) AND q.state = 'queued'
"""


class JobQueue:
    def __init__(self, database_url: str, attempt: Optional[int] = None) -> None:
        self.database_url = database_url
        # Default fencing attempt for ``heartbeat`` (see ``fenced``).
        self.attempt = attempt

    def fenced(self, attempt: int) -> "JobQueue":
        """A queue whose ``heartbeat`` calls are fenced by ``attempt`` unless given one."""
        return type(self)(self.database_url, attempt=attempt)

    # ---------------------------------------------------------------- submission (API side)

    def enqueue(
        self,
        pipeline: str,
        snapshot_id: int,
        params: dict,
        requires: Iterable[str],
        retry_of: Optional[int] = None,
        submitted_by: Optional[str] = None,
    ) -> Job:
        """Insert a queued job. With ``retry_of``, raises ``AlreadyRetried`` when that job
        already has a retry (``jobs.retry_of`` is unique)."""
        bad = find_non_finite(params)
        if bad is not None:
            raise ValueError(f"params: {bad} is not a finite number")
        with connect(self.database_url) as conn:
            if retry_of is not None:
                # Serialise concurrent retries of the same job on its row.
                conn.execute("SELECT 1 FROM jobs WHERE id = %s FOR UPDATE", (retry_of,))
                prior = conn.execute(
                    "SELECT id FROM jobs WHERE retry_of = %s", (retry_of,)
                ).fetchone()
                if prior is not None:
                    raise AlreadyRetried(retry_of, prior["id"])
            row = conn.execute(
                "INSERT INTO jobs (pipeline, snapshot_id, params, requires, retry_of, submitted_by)"
                " VALUES (%s, %s, %s, %s, %s, %s) RETURNING *",
                (
                    pipeline,
                    snapshot_id,
                    Jsonb(params),
                    sorted(set(requires)),
                    retry_of,
                    submitted_by,
                ),
            ).fetchone()
        return Job.from_row(row)

    def get(self, job_id: int) -> Optional[Job]:
        with connect(self.database_url) as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = %s", (job_id,)).fetchone()
        return Job.from_row(row) if row else None

    def request_cancel(self, job_id: int) -> Optional[Job]:
        """Queued jobs become ``cancelled`` at once; running jobs get ``cancel_requested_at``.

        Returns the updated job, or ``None`` if the job is not queued/running.
        """
        with connect(self.database_url) as conn:
            row = conn.execute(
                """
                UPDATE jobs
                SET state = CASE WHEN state = 'queued' THEN 'cancelled' ELSE state END,
                    finished_at = CASE WHEN state = 'queued' THEN clock_timestamp() ELSE finished_at END,
                    cancel_requested_at = coalesce(cancel_requested_at, clock_timestamp())
                WHERE id = %s AND state IN ('queued', 'running')
                RETURNING *
                """,
                (job_id,),
            ).fetchone()
        return Job.from_row(row) if row else None

    # ---------------------------------------------------------------- worker side

    def claim(
        self,
        worker_id: str,
        resources: Iterable[str],
        lease_seconds: float = 60,
        pipelines: Optional[Iterable[str]] = None,
        max_running_per_user: int = 0,
    ) -> Optional[Job]:
        """Lease a queued job whose ``requires`` is a subset of ``resources``.

        The submitter with the fewest running jobs goes first, then the oldest job. With
        ``max_running_per_user`` > 0, a submitter already running that many jobs is skipped.
        """
        params = {
            "worker": worker_id,
            "lease": float(lease_seconds),
            "resources": sorted(resources),
            "pipelines": None if pipelines is None else sorted(pipelines),
            "max_per_user": max(0, int(max_running_per_user)),
        }
        with connect(self.database_url) as conn:
            if params["max_per_user"]:
                conn.execute(_CLAIM_LOCK)
            row = conn.execute(_CLAIM_SQL, params).fetchone()
        return Job.from_row(row) if row else None

    def heartbeat(
        self,
        job_id: int,
        worker_id: str,
        lease_seconds: float = 60,
        attempt: Optional[int] = None,
    ) -> bool:
        """Extend the lease; return whether cancellation was requested. Raise ``LeaseLost``.

        ``attempt`` (default: the queue's ``attempt``) fences the update to that claim, so a
        stale heartbeat of an earlier attempt by the same worker id cannot extend a new lease.
        """
        if attempt is None:
            attempt = self.attempt
        with connect(self.database_url) as conn:
            row = conn.execute(
                "UPDATE jobs SET lease_expires_at = clock_timestamp() + make_interval(secs => %s)"
                " WHERE id = %s AND lease_owner = %s AND state = 'running'"
                " AND (%s::int IS NULL OR attempts = %s::int)"
                " RETURNING cancel_requested_at IS NOT NULL AS cancel_requested",
                (float(lease_seconds), job_id, worker_id, attempt, attempt),
            ).fetchone()
        if row is None:
            raise LeaseLost(
                f"worker {worker_id!r} does not hold the lease on job {job_id}"
            )
        return bool(row["cancel_requested"])

    def reap_expired(self) -> int:
        """Requeue (or fail after ``max_attempts``) running jobs whose lease expired."""
        with connect(self.database_url) as conn:
            return len(conn.execute(_REAP_SQL).fetchall())

    def register_worker(
        self,
        worker_id: str,
        resources: Iterable[str],
        pipelines: Optional[Iterable[str]] = None,
    ) -> None:
        """Record that ``worker_id`` is polling with ``resources`` for ``pipelines`` (None:
        any pipeline); drop rows unseen for a day."""
        with connect(self.database_url) as conn:
            conn.execute(
                "INSERT INTO workers (id, resources, pipelines, last_seen)"
                " VALUES (%s, %s, %s, now())"
                " ON CONFLICT (id) DO UPDATE SET resources = EXCLUDED.resources,"
                " pipelines = EXCLUDED.pipelines, last_seen = now()",
                (
                    worker_id,
                    sorted(set(resources)),
                    None if pipelines is None else sorted(set(pipelines)),
                ),
            )
            conn.execute(
                "DELETE FROM workers w WHERE last_seen < now() - interval '1 day'"
                " AND NOT EXISTS (SELECT 1 FROM jobs j WHERE j.lease_owner = w.id"
                " AND j.state = 'running')"
            )

    def queue_status(
        self, job_ids: Iterable[int], max_running_per_user: int = 0
    ) -> dict[int, dict]:
        """Why each queued job in ``job_ids`` is waiting; other ids are left out.

        Per job: ``ahead`` (queued jobs submitted earlier that an online worker able to run
        this one could also take; approximate, as claims rotate between users), ``workers``
        and ``busy`` (online workers offering everything the job requires, and how many are
        running a job), ``user_running`` / ``user_limit``, and ``reason``: ``no_worker``,
        ``user_limit``, ``busy`` or ``starting``.
        """
        ids = sorted({int(i) for i in job_ids})
        if not ids:
            return {}
        limit = max(0, int(max_running_per_user))
        with connect(self.database_url) as conn:
            rows = conn.execute(
                _QUEUE_STATUS_SQL,
                {"ids": ids, "online": WORKER_ONLINE_SECONDS},
            ).fetchall()
        out: dict[int, dict] = {}
        for row in rows:
            if row["workers"] == 0:
                reason = "no_worker"
            elif (
                limit
                and row["submitted_by"] is not None
                and row["user_running"] >= limit
            ):
                reason = "user_limit"
            elif row["busy"] >= row["workers"]:
                reason = "busy"
            else:
                reason = "starting"
            out[row["id"]] = {
                "ahead": row["ahead"],
                "workers": row["workers"],
                "busy": row["busy"],
                "requires": row["requires"],
                "user_running": row["user_running"],
                "user_limit": limit,
                "reason": reason,
            }
        return out

    def add_event(
        self,
        job_id: int,
        stage: str,
        message: str = "",
        metrics: Optional[dict] = None,
        *,
        worker_id: Optional[str] = None,
        attempt: Optional[int] = None,
    ) -> None:
        """Append an event; ``seq`` is assigned under a lock on the job row.

        With ``worker_id`` and ``attempt`` the write is fenced inside the INSERT: it raises
        ``LeaseLost`` and inserts nothing unless that attempt still owns the running job, or
        that worker itself finished the job on that attempt and ``stage`` names the terminal
        state (the worker logs its own terminal event after finishing). A job the reaper or
        a cancel ended (``finished_by`` NULL) takes no more fenced events.
        """
        metrics_json = Jsonb(sanitise_non_finite(metrics or {}))
        with connect(self.database_url) as conn:
            conn.execute(
                "SELECT 1 FROM jobs WHERE id = %s FOR NO KEY UPDATE", (job_id,)
            )
            if worker_id is None or attempt is None:
                conn.execute(
                    "INSERT INTO job_events (job_id, seq, stage, message, metrics)"
                    " SELECT %s, coalesce(max(seq), 0) + 1, %s, %s, %s FROM job_events"
                    " WHERE job_id = %s",
                    (job_id, stage, message, metrics_json, job_id),
                )
                return
            cur = conn.execute(
                "INSERT INTO job_events (job_id, seq, stage, message, metrics)"
                " SELECT j.id,"
                "  coalesce((SELECT max(e.seq) FROM job_events e WHERE e.job_id = j.id), 0) + 1,"
                "  %(stage)s, %(message)s, %(metrics)s"
                " FROM jobs j"
                " WHERE j.id = %(job)s AND j.attempts = %(attempt)s"
                "  AND ((j.state = 'running' AND j.lease_owner = %(worker)s)"
                "   OR (j.state = ANY(%(terminal)s) AND j.finished_by = %(worker)s"
                "       AND j.state = %(stage)s))",
                {
                    "job": job_id,
                    "stage": stage,
                    "message": message,
                    "metrics": metrics_json,
                    "attempt": attempt,
                    "worker": worker_id,
                    "terminal": list(TERMINAL_STATES),
                },
            )
            if cur.rowcount != 1:
                raise LeaseLost(
                    f"worker {worker_id!r} attempt {attempt} no longer owns job {job_id}"
                )

    def complete(
        self,
        job_id: int,
        worker_id: str,
        model: dict[str, Any],
        attempt: Optional[int] = None,
    ) -> int:
        """Mark the job succeeded and record its model version atomically; return the model id.

        Raises ``CancelRequested`` (nothing written) if cancellation was requested after the
        worker's last check, and ``LeaseLost`` if the attempt no longer owns the job.
        """
        with connect(self.database_url) as conn:
            job = self._finish(
                conn,
                job_id,
                worker_id,
                "succeeded",
                None,
                attempt,
                refuse_if_cancelled=True,
            )
            row = conn.execute(
                "INSERT INTO model_versions"
                " (job_id, pipeline, snapshot_id, metrics, format, object_key, sha256, metadata)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
                (
                    job_id,
                    job["pipeline"],
                    job["snapshot_id"],
                    Jsonb(sanitise_non_finite(model["metrics"])),
                    model["format"],
                    model["object_key"],
                    model["sha256"],
                    Jsonb(sanitise_non_finite(model.get("metadata") or {})),
                ),
            ).fetchone()
        return row["id"]

    def complete_without_model(
        self, job_id: int, worker_id: str, attempt: Optional[int] = None
    ) -> None:
        """Mark a job that produces no model (Data Preparation) succeeded."""
        with connect(self.database_url) as conn:
            self._finish(
                conn,
                job_id,
                worker_id,
                "succeeded",
                None,
                attempt,
                refuse_if_cancelled=True,
            )

    def fail(
        self, job_id: int, worker_id: str, error: str, attempt: Optional[int] = None
    ) -> None:
        with connect(self.database_url) as conn:
            self._finish(conn, job_id, worker_id, "failed", error, attempt)

    def mark_cancelled(
        self, job_id: int, worker_id: str, attempt: Optional[int] = None
    ) -> None:
        with connect(self.database_url) as conn:
            self._finish(conn, job_id, worker_id, "cancelled", None, attempt)

    def verify_lease(
        self, job_id: int, worker_id: str, attempt: Optional[int] = None
    ) -> None:
        """Raise ``LeaseLost`` unless ``worker_id`` holds an unexpired lease on this attempt."""
        with connect(self.database_url) as conn:
            row = conn.execute(
                "SELECT 1 FROM jobs WHERE id = %s AND lease_owner = %s AND state = 'running'"
                " AND lease_expires_at > clock_timestamp()"
                " AND (%s::int IS NULL OR attempts = %s::int)",
                (job_id, worker_id, attempt, attempt),
            ).fetchone()
        if row is None:
            raise LeaseLost(
                f"worker {worker_id!r} does not hold the lease on job {job_id}"
            )

    @staticmethod
    def _finish(
        conn,
        job_id: int,
        worker_id: str,
        state: str,
        error: Optional[str],
        attempt: Optional[int] = None,
        refuse_if_cancelled: bool = False,
    ) -> dict:
        # ``attempt`` (the claim's attempt number) is the per-claim fencing token: it stops a
        # worker that lost its lease from writing after the same worker id re-claims the job.
        row = conn.execute(
            "UPDATE jobs SET state = %s, error = %s, finished_at = clock_timestamp(),"
            " lease_owner = NULL, lease_expires_at = NULL, finished_by = %s"
            " WHERE id = %s AND lease_owner = %s AND state = 'running'"
            " AND (%s::int IS NULL OR attempts = %s::int)"
            " AND (NOT %s OR cancel_requested_at IS NULL) RETURNING *",
            (
                state,
                error,
                worker_id,
                job_id,
                worker_id,
                attempt,
                attempt,
                refuse_if_cancelled,
            ),
        ).fetchone()
        if row is None and refuse_if_cancelled:
            pending = conn.execute(
                "SELECT 1 FROM jobs WHERE id = %s AND lease_owner = %s AND state = 'running'"
                " AND (%s::int IS NULL OR attempts = %s::int)"
                " AND cancel_requested_at IS NOT NULL",
                (job_id, worker_id, attempt, attempt),
            ).fetchone()
            if pending is not None:
                raise CancelRequested(f"job {job_id} was cancelled before it completed")
        if row is None:
            raise LeaseLost(
                f"worker {worker_id!r} does not hold the lease on job {job_id}"
            )
        return row
