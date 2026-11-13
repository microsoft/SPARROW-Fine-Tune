"""Job queue: leasing, launching, cancellation and restart recovery.

The lease logic is the single-GPU queue, and the restart recovery is what stops
a two-day run's row from claiming to be alive forever after the container
bounced. Both are worth testing properly.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from sparrow_web.db import Database
from sparrow_web.jobs import JobQueue, host_epoch
from sparrow_web.settings import Settings


@pytest.fixture
def queue(tmp_path) -> JobQueue:
    db = Database(tmp_path / "t.db")
    db.migrate()
    settings = Settings(state_dir=tmp_path / "state")
    return JobQueue(db, settings)


class TestCreate:
    def test_assigns_work_dir_under_state(self, queue, tmp_path):
        job_id = queue.create("audit", params={"a": 1}, resources=["project:1"])
        job = queue.get(job_id)
        assert job["state"] == "queued"
        assert job["work_dir"] == str(tmp_path / "state" / "jobs" / str(job_id))
        assert json.loads(job["params_json"]) == {"a": 1}

    def test_links_are_stored(self, queue):
        job_id = queue.create("prepare", snapshot_id=42, prepared_id=7)
        job = queue.get(job_id)
        assert job["snapshot_id"] == 42 and job["prepared_id"] == 7

    def test_listing_is_newest_first(self, queue):
        ids = [queue.create("audit") for _ in range(3)]
        assert [j["id"] for j in queue.list()] == sorted(ids, reverse=True)


class TestLeasing:
    def test_lease_succeeds_and_marks_leased(self, queue):
        job_id = queue.create("train", resources=["gpu:1"])
        assert queue.try_lease(job_id, ["gpu:1"]) is True
        assert queue.get(job_id)["state"] == "leased"
        assert queue.leases() == {"gpu:1": job_id}

    def test_second_job_cannot_take_a_held_gpu(self, queue):
        a = queue.create("train", resources=["gpu:1"])
        b = queue.create("train", resources=["gpu:1"])
        assert queue.try_lease(a, ["gpu:1"]) is True
        assert queue.try_lease(b, ["gpu:1"]) is False
        assert queue.get(b)["state"] == "queued", "a failed lease must not change state"

    def test_training_and_evaluation_use_different_cards(self, queue):
        """A two-day run must not block a five-minute evaluation."""
        train = queue.create("train", resources=["gpu:1"])
        evaluate = queue.create("evaluate", resources=["gpu:0"])
        assert queue.try_lease(train, ["gpu:1"]) is True
        assert queue.try_lease(evaluate, ["gpu:0"]) is True

    def test_partial_acquisition_is_rolled_back(self, queue):
        """All-or-nothing: a half-held set is how a queue deadlocks."""
        holder = queue.create("prepare", resources=["prepared:1"])
        queue.try_lease(holder, ["prepared:1"])

        other = queue.create("train", resources=["gpu:1", "prepared:1"])
        assert queue.try_lease(other, ["gpu:1", "prepared:1"]) is False
        assert "gpu:1" not in queue.leases(), "gpu:1 was leaked by a failed multi-lease"
        assert queue.get(other)["state"] == "queued"

    def test_finish_releases_every_lease(self, queue):
        job_id = queue.create("train", resources=["gpu:1", "prepared:2"])
        queue.try_lease(job_id, ["gpu:1", "prepared:2"])
        queue.finish(job_id, "succeeded", result={"ok": True})
        assert queue.leases() == {}
        assert queue.get(job_id)["state"] == "succeeded"

    def test_a_job_leased_twice_is_rejected(self, queue):
        job_id = queue.create("train", resources=["gpu:1"])
        assert queue.try_lease(job_id, ["gpu:1"]) is True
        assert queue.try_lease(job_id, ["gpu:1"]) is False


class TestLaunch:
    def test_launch_writes_job_file_and_records_pid(self, queue):
        job_id = queue.create("audit", params={"x": 1})
        queue.try_lease(job_id, [])
        handles = queue.launch(job_id)

        job = queue.get(job_id)
        assert job["state"] == "running"
        assert job["pid"] == handles["pid"]
        assert job["host_epoch"] == host_epoch()

        written = json.loads((Path(job["work_dir"]) / "job.json").read_text())
        assert written["kind"] == "audit" and written["params"] == {"x": 1}


class TestCancel:
    def test_queued_job_is_removed(self, queue):
        job_id = queue.create("audit")
        assert queue.cancel(job_id)["cancelled"] is True
        assert queue.get(job_id)["state"] == "cancelled"

    def test_already_finished_job_is_not_cancelled(self, queue):
        job_id = queue.create("audit")
        queue.finish(job_id, "succeeded")
        result = queue.cancel(job_id)
        assert result["cancelled"] is False and "already" in result["detail"]

    def test_unknown_job(self, queue):
        assert queue.cancel(9999)["cancelled"] is False


class TestRestartRecovery:
    def test_job_from_a_previous_boot_is_marked_died(self, queue):
        """The container restarted; the process is gone but the row says running.

        Detected via the host epoch, not the pid — pids restart from low numbers
        after a reboot, so a stale pid can match an unrelated live process.
        """
        job_id = queue.create("train")
        with queue.db.write() as c:
            c.execute("UPDATE jobs SET state='running', pid=1234, host_epoch='old-boot'"
                      " WHERE id=?", (job_id,))
            c.execute("INSERT INTO resource_leases(resource, job_id, acquired_at)"
                      " VALUES ('gpu:1', ?, '2020-01-01')", (job_id,))

        assert queue.reconcile_boot() == [job_id]
        job = queue.get(job_id)
        assert job["state"] == "died"
        assert "container restarted" in job["error"]
        assert queue.leases() == {}, "a dead job must not keep holding the GPU"

    def test_job_from_this_boot_is_left_alone(self, queue):
        job_id = queue.create("train")
        with queue.db.write() as c:
            c.execute("UPDATE jobs SET state='running', pid=1234, host_epoch=? WHERE id=?",
                      (host_epoch(), job_id))
        assert queue.reconcile_boot() == []
        assert queue.get(job_id)["state"] == "running"

    def test_queued_jobs_survive_a_restart(self, queue):
        """Nothing was started, so there is nothing to reap — it just runs later."""
        job_id = queue.create("audit")
        assert queue.reconcile_boot() == []
        assert queue.get(job_id)["state"] == "queued"


class TestWorkerFiles:
    def test_progress_is_read_from_status_file(self, queue):
        job_id = queue.create("audit")
        work = Path(queue.get(job_id)["work_dir"])
        work.mkdir(parents=True, exist_ok=True)
        (work / "status.json").write_text(json.dumps(
            {"state": "running", "progress_phase": "Checking images",
             "progress_current": 500, "progress_total": 60049}))

        status = queue.read_worker_status(queue.get(job_id))
        queue.update_progress(job_id, status)
        job = queue.get(job_id)
        assert job["progress_phase"] == "Checking images"
        assert job["progress_current"] == 500 and job["progress_total"] == 60049

    def test_missing_or_corrupt_files_degrade_quietly(self, queue):
        job_id = queue.create("audit")
        job = queue.get(job_id)
        assert queue.read_worker_status(job) == {}
        assert queue.read_result(job) is None

        work = Path(job["work_dir"])
        work.mkdir(parents=True, exist_ok=True)
        (work / "status.json").write_text("{not json")
        assert queue.read_worker_status(job) == {}


class TestWorkerEndToEnd:
    def test_audit_job_runs_detached_and_writes_a_result(self, queue, tmp_path):
        """The real worker, launched detached, against a real little dataset."""
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from tests.conftest import write_pair

        root = tmp_path / "ds"
        for i in range(6):
            write_pair(root, f"i{i}", objects=[("lion", 10, 10, 60, 60)])

        artifact_dir = tmp_path / "snap"
        job_id = queue.create("audit", params={
            "dataset_root": str(root), "artifact_dir": str(artifact_dir)})
        queue.try_lease(job_id, [])
        queue.launch(job_id)

        job = queue.get(job_id)
        for _ in range(600):
            status = queue.read_worker_status(job)
            if status.get("state") in {"succeeded", "failed"}:
                break
            time.sleep(0.1)

        status = queue.read_worker_status(job)
        log = (Path(job["work_dir"]) / "job.log").read_text(errors="replace")
        assert status.get("state") == "succeeded", f"worker failed:\n{log}"

        result = queue.read_result(job)
        assert result["summary"]["num_images"] == 6
        assert result["summary"]["num_boxes"] == 6
        assert result["content_hash"] and result["quick_hash"]
        assert result["class_counts"]["lion"]["n_boxes"] == 6
        assert (artifact_dir / "images.csv.gz").is_file()
