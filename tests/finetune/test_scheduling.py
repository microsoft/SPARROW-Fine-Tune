"""Multi-user GPU scheduling (2026-11-03): GPU-only pipelines wait for a GPU worker, freed
capacity rotates between users, an optional per-user running cap, and queue status that says
why a job is waiting."""

from __future__ import annotations

import dataclasses
import threading

import pytest

from sparrow_finetune.jobs import JobQueue

from .helpers import build_snapshot, get_job, make_client, queued_job, same_id


@pytest.fixture
def snapshot_id(client) -> str:
    return build_snapshot(client)["snapshot"]["id"]


def _enqueue(queue: JobQueue, snapshot_id, user, requires=()):
    return queue.enqueue("fake_cls", int(snapshot_id), {}, requires, submitted_by=user)


def test_submit_records_the_submitter(client, snapshot_id):
    job = get_job(client, queued_job(client, "fake_cls", snapshot_id))
    assert job["submitted_by"] == "anonymous"


def test_gpu_pipelines_wait_for_a_gpu_worker(settings, registry, snapshot_id):
    gpu_settings = dataclasses.replace(settings, gpu_pipelines="fake_cls")
    queue = JobQueue(settings.database_url)
    with make_client(gpu_settings, registry) as client:
        job_id = queued_job(client, "fake_cls", snapshot_id)
        assert get_job(client, job_id)["requires"] == ["gpu"]
        assert queue.claim("cpu-1", {"cpu"}) is None
        claimed = queue.claim("gpu-1", {"cpu", "gpu"})
        assert claimed is not None and same_id(claimed.id, job_id)
        queue.fail(claimed.id, "gpu-1", "boom", attempt=claimed.attempts)
        r = client.post(f"/v1/jobs/{job_id}/retry")
        assert r.status_code == 201, r.text
        assert r.json()["requires"] == ["gpu"]


def test_retry_follows_the_current_gpu_pipelines_setting(client, queue, snapshot_id):
    job_id = queued_job(client, "fake_cls", snapshot_id)
    claimed = queue.claim("cpu-1", {"cpu"})
    queue.fail(claimed.id, "cpu-1", "boom", attempt=claimed.attempts)
    assert client.post(f"/v1/jobs/{job_id}/retry").json()["requires"] == []


def test_freed_capacity_rotates_between_users(queue, snapshot_id):
    a1 = _enqueue(queue, snapshot_id, "alice")
    a2 = _enqueue(queue, snapshot_id, "alice")
    a3 = _enqueue(queue, snapshot_id, "alice")
    b1 = _enqueue(queue, snapshot_id, "bob")

    first = queue.claim("w-1", {"cpu"})
    assert first.id == a1.id  # nobody running: oldest first
    second = queue.claim("w-2", {"cpu"})
    assert second.id == b1.id  # bob has nothing running, alice has one
    third = queue.claim("w-3", {"cpu"})
    assert third.id == a2.id
    assert queue.claim("w-4", {"cpu"}).id == a3.id


def test_per_user_cap_keeps_extra_jobs_queued(queue, snapshot_id):
    a1 = _enqueue(queue, snapshot_id, "alice")
    a2 = _enqueue(queue, snapshot_id, "alice")
    b1 = _enqueue(queue, snapshot_id, "bob")

    assert queue.claim("w-1", {"cpu"}, max_running_per_user=1).id == a1.id
    assert queue.claim("w-2", {"cpu"}, max_running_per_user=1).id == b1.id
    queue.register_worker("w-3", {"cpu"})
    assert queue.claim("w-3", {"cpu"}, max_running_per_user=1) is None
    status = queue.queue_status([a2.id], max_running_per_user=1)
    assert status[a2.id]["reason"] == "user_limit"

    queue.fail(a1.id, "w-1", "done", attempt=1)
    assert queue.claim("w-3", {"cpu"}, max_running_per_user=1).id == a2.id


def test_per_user_cap_holds_under_concurrent_claims(queue, snapshot_id):
    for _ in range(6):
        _enqueue(queue, snapshot_id, "alice")
    barrier = threading.Barrier(6)
    claimed: list = []

    def worker(n: int) -> None:
        barrier.wait()
        claimed.append(queue.claim(f"w-{n}", {"cpu"}, max_running_per_user=2))

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(job is not None for job in claimed) == 2


def test_queue_status_explains_the_wait(settings, registry, queue, snapshot_id):
    gpu_settings = dataclasses.replace(settings, gpu_pipelines="fake_cls")
    with make_client(gpu_settings, registry) as client:
        first = queued_job(client, "fake_cls", snapshot_id)
        second = queued_job(client, "fake_cls", snapshot_id)

        queue.register_worker("cpu-1", {"cpu"})
        status = get_job(client, second)["queue"]
        assert status["reason"] == "no_worker"
        assert status["workers"] == 0

        queue.register_worker("gpu-1", {"cpu", "gpu"})
        status = get_job(client, second)["queue"]
        assert (
            status["reason"],
            status["workers"],
            status["busy"],
            status["ahead"],
        ) == (
            "starting",
            1,
            0,
            1,
        )

        assert same_id(queue.claim("gpu-1", {"cpu", "gpu"}).id, first)
        status = get_job(client, second)["queue"]
        assert (status["reason"], status["busy"], status["ahead"]) == ("busy", 1, 0)

        listed = {j["id"]: j for j in client.get("/v1/jobs").json()["jobs"]}
        assert listed[int(second)]["queue"]["reason"] == "busy"
        assert listed[int(first)]["queue"] is None  # running jobs carry no queue status


def test_queue_status_reports_user_limit(settings, registry, queue, snapshot_id):
    capped = dataclasses.replace(settings, max_running_per_user=1)
    with make_client(capped, registry) as client:
        first = queued_job(client, "fake_cls", snapshot_id)
        second = queued_job(client, "fake_cls", snapshot_id)
        queue.register_worker("w-1", {"cpu"})
        queue.register_worker("w-2", {"cpu"})
        assert same_id(queue.claim("w-1", {"cpu"}, max_running_per_user=1).id, first)
        status = get_job(client, second)["queue"]
        assert status["reason"] == "user_limit"
        assert (status["user_running"], status["user_limit"]) == (1, 1)


def test_settings_parse_scheduling_env(monkeypatch, database_url):
    from sparrow_finetune.config import Settings

    monkeypatch.setenv("SPARROW_FT_DATABASE_URL", database_url)
    monkeypatch.setenv(
        "SPARROW_FT_GPU_PIPELINES", " yolo_detector, conventional_classifier,"
    )
    monkeypatch.setenv("SPARROW_FT_MAX_RUNNING_PER_USER", "2")
    s = Settings.from_env()
    assert s.gpu_pipelines == ("conventional_classifier", "yolo_detector")
    assert s.max_running_per_user == 2
    monkeypatch.setenv("SPARROW_FT_MAX_RUNNING_PER_USER", "-1")
    with pytest.raises(ValueError):
        Settings.from_env()


def test_queue_status_counts_only_workers_serving_the_pipeline(queue, snapshot_id):
    job = _enqueue(queue, snapshot_id, "alice")
    queue.register_worker("other", {"cpu", "gpu"}, ["yolo_detector"])
    assert queue.queue_status([job.id])[job.id]["reason"] == "no_worker"
    queue.register_worker("cls", {"cpu"}, ["fake_cls"])
    status = queue.queue_status([job.id])[job.id]
    assert (status["reason"], status["workers"]) == ("starting", 1)
