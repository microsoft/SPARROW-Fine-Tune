"""Job queue: creation, resource leasing, launching, and reconciliation.

The whole queue is the ``resource_leases`` table. Its primary key is the mutex —
at most one row can hold ``gpu:1``, so at most one job can be training. There is
no Redis, no Celery, and no broker: the pattern the repo already uses for
training (a detached process plus a status file) generalises to every other long
operation, and it survives the web process restarting, which a broker-backed
worker pool would not do for free.

Resource names:
  ``gpu:<n>``        a specific card. Training takes the display-free card;
                     evaluation and export take the other, so a two-day run never
                     blocks a five-minute evaluation.
  ``project:<id>``   serialises audits within one project.
  ``prepared:<id>``  held while a prepared tree is being written, because
                     ``prepare_yolo_dataset(overwrite=True)`` deletes a directory
                     that a running job may be reading from.
"""

from __future__ import annotations

import json
import os
import signal
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from sparrow_yolo.process_runner import launch_detached, process_is_running

from .db import Database, dumps, loads, utcnow

#: Job states that will never change again without us doing something.
TERMINAL = {"succeeded", "failed", "cancelled", "died"}
ACTIVE = {"queued", "leased", "running"}


def host_epoch() -> str:
    """An identifier for this boot of the container.

    After a container restart, PIDs begin again from low numbers, so a stale pid
    file can match a completely unrelated process. Comparing the init process's
    start time detects a restart without consulting pids at all.
    """
    try:
        return str(os.stat("/proc/1").st_mtime_ns)
    except OSError:
        # Not Linux, or /proc unavailable — fall back to boot-agnostic behaviour.
        return "unknown"


class JobQueue:
    def __init__(self, db: Database, settings):
        self.db = db
        self.settings = settings

    # -- creation ----------------------------------------------------------
    def create(
        self,
        kind: str,
        *,
        project_id: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
        resources: Optional[List[str]] = None,
        priority: int = 0,
        **links: Any,
    ) -> int:
        """Enqueue a job. Returns its id.

        ``links`` may carry ``run_id`` / ``snapshot_id`` / ``prepared_id`` /
        ``model_version_id`` so the result can be attached on completion.
        """
        now = utcnow()
        with self.db.write() as conn:
            cur = conn.execute(
                "INSERT INTO jobs(project_id, kind, state, priority, params_json, work_dir,"
                " needs_resources_json, run_id, snapshot_id, prepared_id, model_version_id,"
                " created_at, updated_at)"
                " VALUES (?,?,'queued',?,?,'',?,?,?,?,?,?,?)",
                (project_id, kind, priority, dumps(params or {}),
                 dumps(resources or []), links.get("run_id"), links.get("snapshot_id"),
                 links.get("prepared_id"), links.get("model_version_id"), now, now),
            )
            job_id = int(cur.lastrowid)
            work_dir = self.settings.job_dir(job_id)
            conn.execute("UPDATE jobs SET work_dir=? WHERE id=?", (str(work_dir), job_id))
        return job_id

    # -- queries -----------------------------------------------------------
    def get(self, job_id: int) -> Optional[Dict[str, Any]]:
        return self.db.one("SELECT * FROM jobs WHERE id=?", (job_id,))

    def list(self, project_id: Optional[int] = None, limit: int = 50) -> List[Dict[str, Any]]:
        if project_id is None:
            return self.db.query(
                "SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,))
        return self.db.query(
            "SELECT * FROM jobs WHERE project_id=? ORDER BY id DESC LIMIT ?",
            (project_id, limit))

    def active(self) -> List[Dict[str, Any]]:
        marks = ",".join("?" * len(ACTIVE))
        return self.db.query(
            f"SELECT * FROM jobs WHERE state IN ({marks}) ORDER BY priority DESC, id",
            tuple(ACTIVE))

    def leases(self) -> Dict[str, int]:
        return {r["resource"]: r["job_id"]
                for r in self.db.query("SELECT resource, job_id FROM resource_leases")}

    # -- lifecycle ---------------------------------------------------------
    def _release(self, conn: sqlite3.Connection, job_id: int) -> None:
        conn.execute("DELETE FROM resource_leases WHERE job_id=?", (job_id,))

    def finish(self, job_id: int, state: str, *, error: Optional[str] = None,
               traceback_text: Optional[str] = None,
               result: Optional[Dict[str, Any]] = None) -> None:
        """Mark a job terminal and free whatever it held."""
        with self.db.write() as conn:
            conn.execute(
                "UPDATE jobs SET state=?, error=?, traceback=?, result_json=?,"
                " finished_at=?, updated_at=? WHERE id=?",
                (state, error, traceback_text,
                 dumps(result) if result is not None else None,
                 utcnow(), utcnow(), job_id),
            )
            self._release(conn, job_id)

    def cancel(self, job_id: int) -> Dict[str, Any]:
        """Cancel a queued job, or signal a running one."""
        job = self.get(job_id)
        if job is None:
            return {"cancelled": False, "detail": "No such job"}
        if job["state"] in TERMINAL:
            return {"cancelled": False, "detail": f"Job is already {job['state']}"}

        if job["state"] == "queued":
            self.finish(job_id, "cancelled")
            return {"cancelled": True, "detail": "Removed from the queue"}

        pid = job.get("pid")
        if pid and process_is_running(pid):
            try:
                # Signal the group: the child owns its own session, and a
                # training job has ranks/dataloader workers under it.
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except OSError:
                try:
                    os.kill(pid, signal.SIGTERM)
                except OSError:
                    pass
        self.finish(job_id, "cancelled")
        return {"cancelled": True, "detail": f"Signalled pid {pid}"}

    # -- dequeue -----------------------------------------------------------
    def try_lease(self, job_id: int, resources: List[str]) -> bool:
        """Atomically take every resource a job needs, or none of them.

        Partial acquisition is the classic way to deadlock a queue, so the whole
        set goes in one transaction and a single primary-key conflict aborts it.
        """
        now = utcnow()
        try:
            with self.db.write() as conn:
                for resource in resources:
                    conn.execute(
                        "INSERT INTO resource_leases(resource, job_id, acquired_at)"
                        " VALUES (?,?,?)", (resource, job_id, now))
                changed = conn.execute(
                    "UPDATE jobs SET state='leased', updated_at=?"
                    " WHERE id=? AND state='queued'", (now, job_id)).rowcount
                if not changed:
                    raise sqlite3.IntegrityError("job is no longer queued")
            return True
        except sqlite3.IntegrityError:
            return False

    def launch(self, job_id: int) -> Dict[str, Any]:
        """Write the job file and start the detached worker."""
        job = self.get(job_id)
        if job is None:
            raise ValueError(f"No such job: {job_id}")

        work_dir = Path(job["work_dir"])
        work_dir.mkdir(parents=True, exist_ok=True)
        job_file = work_dir / "job.json"
        job_file.write_text(
            json.dumps({"id": job_id, "kind": job["kind"],
                        "params": loads(job["params_json"], {})}, indent=2, default=str),
            encoding="utf-8",
        )

        handles = launch_detached(
            "sparrow_web.worker",
            ["--job", str(job_file), "--status", str(work_dir / "status.json")],
            work_dir=work_dir,
            log_name="job.log",
            pid_name="job.pid",
            banner=f"job {job_id} ({job['kind']})",
        )

        with self.db.write() as conn:
            conn.execute(
                "UPDATE jobs SET state='running', pid=?, host_epoch=?, started_at=?,"
                " updated_at=? WHERE id=?",
                (handles["pid"], host_epoch(), utcnow(), utcnow(), job_id),
            )
        return handles

    def fail_launch(self, job_id: int, error: str) -> None:
        self.finish(job_id, "failed", error=error)

    # -- reconcile ---------------------------------------------------------
    def read_worker_status(self, job: Dict[str, Any]) -> Dict[str, Any]:
        path = Path(job["work_dir"]) / "status.json"
        if not path.is_file():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def read_result(self, job: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        path = Path(job["work_dir"]) / "result.json"
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def update_progress(self, job_id: int, status: Dict[str, Any]) -> None:
        with self.db.write() as conn:
            conn.execute(
                "UPDATE jobs SET progress_phase=?, progress_current=?, progress_total=?,"
                " updated_at=? WHERE id=?",
                (status.get("progress_phase"), status.get("progress_current"),
                 status.get("progress_total"), utcnow(), job_id),
            )

    def reconcile_boot(self) -> List[int]:
        """Reap jobs orphaned by a container restart.

        Their processes died with the container, but their rows still say
        running. Detected by the host epoch rather than the pid, because pids are
        reused after a restart and would give false positives.
        """
        current = host_epoch()
        orphaned: List[int] = []
        for job in self.active():
            if job["state"] == "queued":
                continue  # never started; the scheduler will pick it up
            if job.get("host_epoch") and job["host_epoch"] != current:
                orphaned.append(job["id"])
                self.finish(
                    job["id"], "died",
                    error="The container restarted while this job was running.",
                )
        return orphaned
