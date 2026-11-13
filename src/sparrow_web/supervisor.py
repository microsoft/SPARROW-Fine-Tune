"""Background threads: the scheduler, the GPU poller, and the run watcher.

Everything here assumes **one process**. The scheduler, the resource leases and
the SSE subscriber list are all in-memory singletons, so running more than one
gunicorn worker would give you two schedulers racing for the same GPU lease and
half your browser clients missing half their events. Run with ``-w 1`` and
threads, and never with Flask's reloader (it double-starts every thread).

The supervisor observes training by reading files, because it did not start the
trainer and cannot wait on it: training is a detached process that outlives the
web application. That is the property that lets a two-day run survive a
``docker compose restart`` of the web layer.
"""

from __future__ import annotations

import threading
import traceback
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List

import yaml

from sparrow_yolo.hardware import query_gpus
from sparrow_yolo.process_runner import (best_weights, launch_training_process,
                                         resolve_save_dir, write_train_config)

from .annotation import Batches
from .db import Database, dumps, loads, utcnow
from .events import EventBus
from .ingest import HEADLINE_KEYS, RunIngestor
from .jobs import JobQueue, host_epoch
from .repo import Models, Projects, Runs, ingest_audit_result, ingest_prepare_result

#: Run states from which no further epoch will arrive. A model is registered
#: on reaching any of them, provided a checkpoint was actually written.
#: `failed` and `died` are included deliberately: an OOM at epoch 60 still
#: leaves 60 epochs of training in best.pt.
TERMINAL_RUN_STATES = ("completed", "stopped", "failed", "died")

#: Kept in memory only. A GPU sample every two seconds has no long-term value and
#: writing it forever would bloat the database for nothing.
GPU_HISTORY = 300


class Supervisor:
    def __init__(self, db: Database, settings, bus: EventBus):
        self.db = db
        self.settings = settings
        self.bus = bus
        self.jobs = JobQueue(db, settings)
        self.ingestor = RunIngestor(db)
        self.projects = Projects(db)
        self.runs = Runs(db)
        self.models = Models(db)

        self.gpu_history: Deque[Dict[str, Any]] = deque(maxlen=GPU_HISTORY)
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self._started = False

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if self._started:
            return
        self._started = True

        # Anything still marked active belongs to a previous boot of the
        # container; its process died with it.
        try:
            orphans = self.jobs.reconcile_boot()
            if orphans:
                print(f"[supervisor] reaped {len(orphans)} job(s) from a previous boot")
        except Exception:  # noqa: BLE001
            traceback.print_exc()

        try:
            registered = self.backfill_models()
            if registered:
                print(f"[supervisor] registered {registered} model(s) from finished runs")
        except Exception:  # noqa: BLE001
            traceback.print_exc()

        for name, target, interval in (
            ("scheduler", self._scheduler_tick, self.settings.scheduler_interval_s),
            ("gpu", self._gpu_tick, self.settings.gpu_poll_interval_s),
            ("runs", self._runs_tick, self.settings.run_poll_interval_s),
        ):
            thread = threading.Thread(target=self._loop, args=(name, target, interval),
                                      name=f"sparrow-{name}", daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=5)

    def _loop(self, name: str, tick, interval: float) -> None:
        while not self._stop.is_set():
            try:
                tick()
            except Exception:  # noqa: BLE001 - a bad tick must not kill the thread
                print(f"[supervisor:{name}] tick failed")
                traceback.print_exc()
            self._stop.wait(interval)

    # -- GPU ---------------------------------------------------------------
    def _gpu_tick(self) -> None:
        gpus = [g.as_dict() for g in query_gpus()]
        sample = {"at": utcnow(), "gpus": gpus}
        self.gpu_history.append(sample)
        self.bus.publish("gpu", "gpu.sample", sample)

    # -- runs --------------------------------------------------------------
    def _runs_tick(self) -> None:
        """Pull every active run's state and any new epochs off disk."""
        for run in self.runs.active():
            if run["state"] == "queued":
                continue
            try:
                self._watch_run(run)
            except Exception:  # noqa: BLE001
                traceback.print_exc()

    def _watch_run(self, run: Dict[str, Any]) -> None:
        status = self.ingestor.ingest_status(run)
        if not status:
            return

        save_dir = resolve_save_dir(run["run_dir"]) or Path(run["run_dir"])
        new_rows = self.ingestor.ingest_metrics(run["id"], Path(save_dir) / "results.csv")

        updated = self.runs.get(run["id"]) or run
        topic = f"run:{run['id']}"

        if new_rows:
            self.bus.publish(topic, "run.metrics", {
                "run_id": run["id"],
                "rows": new_rows,
                "eta": self.ingestor.eta(run["id"], updated.get("epochs")),
            })

        if updated["state"] != run["state"] or new_rows:
            self.bus.publish(topic, "run.status", self.run_snapshot(updated))

        # A run that has just reached a terminal state becomes a model version,
        # so the registry is never a manual step the user can forget.
        #
        # Not only `completed`: stopping early is a normal, encouraged action —
        # the training page says so — and a run killed at epoch 24 with a good
        # best.pt is a usable model. Ultralytics keeps the best checkpoint
        # regardless of how the run ended, so the only real question is whether
        # a checkpoint exists, which _register_model already decides.
        if updated["state"] in TERMINAL_RUN_STATES and run["state"] not in TERMINAL_RUN_STATES:
            self._register_model(updated)

    def backfill_models(self) -> int:
        """Register any finished run that has a checkpoint but no model version.

        The live path in :meth:`_watch_run` only sees a run cross into a terminal
        state, so a run that ended while the app was down — or before this
        registered anything but `completed` — would leave a perfectly good
        checkpoint stranded and invisible. Runs at boot, and is idempotent.
        """
        registered = 0
        for run in self.db.query(
            "SELECT r.* FROM runs r LEFT JOIN model_versions m ON m.run_id = r.id"
            f" WHERE m.id IS NULL AND r.state IN ({','.join('?' * len(TERMINAL_RUN_STATES))})"
            " ORDER BY r.id", TERMINAL_RUN_STATES
        ):
            before = self.db.one("SELECT id FROM model_versions WHERE run_id=?", (run["id"],))
            self._register_model(run)
            after = self.db.one("SELECT id FROM model_versions WHERE run_id=?", (run["id"],))
            if after and not before:
                registered += 1
        return registered

    def _register_model(self, run: Dict[str, Any]) -> None:
        weights = best_weights(run["run_dir"])
        if not weights:
            return
        existing = self.db.one("SELECT id FROM model_versions WHERE run_id=?", (run["id"],))
        if existing:
            return
        prepared = self.db.one("SELECT * FROM prepared_datasets WHERE id=?",
                               (run.get("prepared_id"),)) if run.get("prepared_id") else None
        class_map = loads(prepared["class_map_json"], {}) if prepared else {}
        model_id = self.models.create_from_run(
            run["project_id"], run, weights_path=str(weights), class_map=class_map)
        self.bus.publish(f"project:{run['project_id']}", "model.created",
                         {"model_version_id": model_id, "run_id": run["id"]})

    def run_snapshot(self, run: Dict[str, Any]) -> Dict[str, Any]:
        """The full current state of a run, as sent on every event.

        Complete rather than a delta so a client that reconnected, or that
        dropped events while lagging, is correct again immediately.
        """
        metrics = loads(run.get("metrics_json"), {}) or {}
        return {
            "run_id": run["id"],
            "project_id": run["project_id"],
            "name": run["name"],
            "state": run["state"],
            "epoch": run.get("epoch"),
            "epochs": run.get("epochs"),
            "best_epoch": run.get("best_epoch"),
            "best_fitness": run.get("best_fitness"),
            "stopped_early": bool(run.get("stopped_early")),
            "headline": {k: metrics.get(k) for k in HEADLINE_KEYS},
            "eta": self.ingestor.eta(run["id"], run.get("epochs")),
            "error": run.get("error"),
            "updated_at": run.get("updated_at"),
        }

    # -- scheduler ---------------------------------------------------------
    def _scheduler_tick(self) -> None:
        self._reconcile_jobs()
        self._dequeue()

    def _reconcile_jobs(self) -> None:
        from sparrow_yolo.process_runner import process_is_running

        for job in self.jobs.active():
            if job["state"] == "queued":
                continue

            status = self.jobs.read_worker_status(job)
            if status:
                self.jobs.update_progress(job["id"], status)
                self.bus.publish("jobs", "job.progress", {
                    "job_id": job["id"], "kind": job["kind"],
                    "phase": status.get("progress_phase"),
                    "current": status.get("progress_current"),
                    "total": status.get("progress_total"),
                })

            pid = job.get("pid")
            if job["state"] == "leased" or (pid and process_is_running(pid)):
                continue
            if not pid:
                continue

            # The process is gone. Its result file decides whether that was
            # success, a handled failure, or a kill that never ran the handler.
            result = self.jobs.read_result(job)
            worker_state = (status or {}).get("state")
            if result is not None and worker_state == "succeeded":
                self._complete_job(job, result)
            elif worker_state == "failed":
                self.jobs.finish(job["id"], "failed",
                                 error=(status or {}).get("error"),
                                 traceback_text=(status or {}).get("traceback"))
            else:
                self.jobs.finish(
                    job["id"], "died",
                    error="The process exited without reporting a result — usually an "
                          "out-of-memory kill. Check the job log.")
            self.bus.publish("jobs", "job.updated", {"job_id": job["id"],
                                                     "state": self.jobs.get(job["id"])["state"]})

    def _complete_job(self, job: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Fold a finished job's output into the domain tables."""
        project = self.projects.get(job["project_id"]) if job.get("project_id") else None
        try:
            if job["kind"] == "audit" and project:
                snapshot_id = ingest_audit_result(self.db, self.settings, project, job, result)
                with self.db.write() as conn:
                    conn.execute("UPDATE jobs SET snapshot_id=? WHERE id=?",
                                 (snapshot_id, job["id"]))
                self.bus.publish(f"project:{project['id']}", "snapshot.created",
                                 {"snapshot_id": snapshot_id, "project_id": project["id"]})
            elif job["kind"] == "prepare" and project:
                prepared_id = ingest_prepare_result(self.db, project, job, result)
                with self.db.write() as conn:
                    conn.execute("UPDATE jobs SET prepared_id=? WHERE id=?",
                                 (prepared_id, job["id"]))
                self.bus.publish(f"project:{project['id']}", "prepared.created",
                                 {"prepared_id": prepared_id, "project_id": project["id"]})
            elif job["kind"] == "evaluate":
                self._ingest_evaluation(job, result)
            elif job["kind"] == "predict":
                self._ingest_prediction(job, result)
        except Exception as e:  # noqa: BLE001 - a bad ingest must not lose the job
            self.jobs.finish(job["id"], "failed",
                             error=f"Job succeeded but its result could not be recorded: {e!r}",
                             traceback_text=traceback.format_exc())
            return
        self.jobs.finish(job["id"], "succeeded", result=result)

    def _ingest_prediction(self, job: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Fold an annotation pass back in.

        A prediction never overwrites review: an image someone has already
        checked keeps its ``reviewed`` state and its human box counts, even
        though the pass rewrote the XML underneath. Re-running a model at a lower
        threshold should not quietly discard an afternoon of review.
        """
        params = loads(job["params_json"], {})
        batch_id = int(params["batch_id"])
        batches = Batches(self.db)

        now = utcnow()
        with self.db.write() as conn:
            for row in result.get("images", []):
                if row.get("error"):
                    continue
                conn.execute(
                    "UPDATE annotation_images SET"
                    " state=CASE WHEN state='reviewed' THEN 'reviewed' ELSE 'predicted' END,"
                    " n_boxes=?, n_ai_boxes=?, n_human_boxes=0, min_confidence=?,"
                    " width=?, height=?, model_version_id=?, updated_at=?"
                    " WHERE batch_id=? AND rel_path=? AND state != 'reviewed'",
                    (row["n_boxes"], row["n_boxes"], row.get("min_confidence"),
                     row.get("width"), row.get("height"),
                     params.get("model_version_id"), now, batch_id, row["rel_path"]))

        pass_id = batches.record_pass(batch_id, params, result, job_id=job["id"])
        if job.get("project_id"):
            self.bus.publish(f"project:{job['project_id']}", "annotation.pass",
                             {"batch_id": batch_id, "pass_id": pass_id,
                              "n_boxes": result.get("n_boxes", 0)})

    def _ingest_evaluation(self, job: Dict[str, Any], result: Dict[str, Any]) -> None:
        metrics = result.get("metrics", {})
        with self.db.write() as conn:
            cur = conn.execute(
                "INSERT INTO model_evaluations(model_version_id,prepared_id,split,job_id,"
                "metrics_json,map50,map5095,precision,recall,fitness,created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (job.get("model_version_id"), job.get("prepared_id"),
                 result.get("split", "test"), job["id"], dumps(metrics),
                 metrics.get("metrics/mAP50(B)"), metrics.get("metrics/mAP50-95(B)"),
                 metrics.get("metrics/precision(B)"), metrics.get("metrics/recall(B)"),
                 metrics.get("fitness"), utcnow()),
            )
            eval_id = int(cur.lastrowid)
            conn.executemany(
                "INSERT OR REPLACE INTO model_class_metrics(evaluation_id,class_id,class_name,"
                "ap50,ap5095,precision,recall) VALUES (?,?,?,?,?,?,?)",
                [(eval_id, c["class_id"], c["class_name"], c.get("ap50"), c.get("ap5095"),
                  c.get("precision"), c.get("recall"))
                 for c in result.get("per_class", [])],
            )

    def _dequeue(self) -> None:
        for job in self.db.query(
                "SELECT * FROM jobs WHERE state='queued' ORDER BY priority DESC, id"):
            resources = loads(job["needs_resources_json"], []) or []
            if not self.jobs.try_lease(job["id"], resources):
                continue
            try:
                if job["kind"] == "train":
                    self._launch_training(job)
                else:
                    self.jobs.launch(job["id"])
                self.bus.publish("jobs", "job.updated",
                                 {"job_id": job["id"], "state": "running"})
            except Exception as e:  # noqa: BLE001
                self.jobs.fail_launch(job["id"], f"Could not start: {e!r}")
                traceback.print_exc()

    def _launch_training(self, job: Dict[str, Any]) -> None:
        """Training uses its own purpose-built launcher rather than the generic
        worker: it already has one, with the DDP-aware signal handling and the
        status contract the rest of this module reads."""
        params = loads(job["params_json"], {})
        run = self.runs.get(job["run_id"])
        if run is None:
            raise ValueError(f"Job {job['id']} has no run row")

        run_dir = Path(run["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)

        config_path = write_train_config(
            run_dir / "train_config.yaml",
            model_weights=params["model_weights"],
            data_yaml=params["data_yaml"],
            project=str(self.settings.runs_root),
            name=run_dir.name,
            imgsz=int(params.get("imgsz", 1024)),
            epochs=int(params.get("epochs", 100)),
            batch=params.get("batch", "auto"),
            device=str(params.get("device", self.settings.train_gpu)),
            workers=params.get("workers", "auto"),
            patience=int(params.get("patience", 50)),
            amp=bool(params.get("amp", True)),
            **(params.get("extra") or {}),
        )
        handles = launch_training_process(config_path, run_dir=run_dir)

        # What the user asked for is in runs.config_json; this is what it became.
        # They differ wherever something was 'auto', and the resolved value is the
        # one you need to reproduce or explain a run.
        try:
            resolved = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            resolved = {}

        with self.db.write() as conn:
            conn.execute(
                "UPDATE runs SET state='starting', pid=?, train_args_json=?, batch=?,"
                " started_at=?, updated_at=? WHERE id=?",
                (handles["pid"], dumps(resolved), str(resolved.get("batch", "")),
                 utcnow(), utcnow(), run["id"]))
            conn.execute(
                "UPDATE jobs SET state='running', pid=?, host_epoch=?, started_at=?,"
                " updated_at=? WHERE id=?",
                (handles["pid"], host_epoch(), utcnow(), utcnow(), job["id"]))

    # -- for the SSE hello frame ------------------------------------------
    def hello(self, topics: List[str]) -> Dict[str, Any]:
        """Full current state for everything a client subscribed to."""
        payload: Dict[str, Any] = {
            "gpus": self.gpu_history[-1]["gpus"] if self.gpu_history else [],
            "gpu_history": list(self.gpu_history)[-60:],
            "jobs": self.jobs.active(),
        }
        for topic in topics:
            if topic.startswith("run:"):
                run = self.runs.get(int(topic.split(":", 1)[1]))
                if run:
                    payload.setdefault("runs", []).append({
                        **self.run_snapshot(run),
                        "series": self.ingestor.series(run["id"]),
                    })
        return payload
