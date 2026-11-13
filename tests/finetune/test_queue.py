"""AC15-AC17: the PostgreSQL job queue -- leases, resource matching, concurrency.

Jobs are created over HTTP (the only public way to enqueue) and then driven
directly through ``JobQueue``. Lease expiry uses ``lease_seconds=1`` plus a short
sleep rather than editing rows, so only the public API is exercised.
"""

from __future__ import annotations

import threading
import time

import pytest

from sparrow_finetune.jobs import LeaseLost
from sparrow_finetune.workers import run_once

from .helpers import build_snapshot, get_job, queued_job, same_id

EXPIRE_WAIT_S = 1.5


@pytest.fixture
def snapshot_id(client) -> str:
    return build_snapshot(client)["snapshot"]["id"]


# --------------------------------------------------------------------------- AC15


def test_ac15_claim_marks_job_running(client, queue, snapshot_id):
    job_id = queued_job(client, "fake_cls", snapshot_id)
    job = queue.claim("w-1", {"cpu"}, lease_seconds=30)
    assert job is not None and same_id(job.id, job_id)
    assert get_job(client, job_id)["state"] == "running"
    assert queue.claim("w-2", {"cpu"}, lease_seconds=30) is None


def test_ac15_live_lease_is_not_reaped(client, queue, snapshot_id):
    queued_job(client, "fake_cls", snapshot_id)
    assert queue.claim("w-1", {"cpu"}, lease_seconds=30) is not None
    assert queue.reap_expired() == 0
    assert queue.claim("w-2", {"cpu"}, lease_seconds=30) is None


def test_ac15_expired_lease_requeued_and_claimable_again(client, queue, snapshot_id):
    job_id = queued_job(client, "fake_cls", snapshot_id)
    first = queue.claim("w-1", {"cpu"}, lease_seconds=1)
    assert first is not None
    time.sleep(EXPIRE_WAIT_S)

    assert queue.reap_expired() == 1

    assert get_job(client, job_id)["state"] == "queued"
    again = queue.claim("w-2", {"cpu"}, lease_seconds=30)
    assert again is not None and same_id(again.id, job_id)


def test_ac15_former_owner_heartbeat_after_reap_raises(client, queue, snapshot_id):
    queued_job(client, "fake_cls", snapshot_id)
    job = queue.claim("w-1", {"cpu"}, lease_seconds=1)
    time.sleep(EXPIRE_WAIT_S)
    queue.reap_expired()
    assert queue.claim("w-2", {"cpu"}, lease_seconds=30) is not None
    with pytest.raises(LeaseLost):
        queue.heartbeat(job.id, "w-1", 30)


def test_ac15_fails_after_max_attempts(client, queue, snapshot_id):
    job_id = queued_job(client, "fake_cls", snapshot_id)
    for attempt in range(1, 4):
        job = queue.claim("w-1", {"cpu"}, lease_seconds=1)
        assert job is not None and same_id(job.id, job_id), f"attempt {attempt} not claimable"
        time.sleep(EXPIRE_WAIT_S)
        queue.reap_expired()
        if attempt < 3:
            assert get_job(client, job_id)["state"] == "queued"

    assert get_job(client, job_id)["state"] == "failed"
    assert queue.claim("w-1", {"cpu"}, lease_seconds=30) is None


def test_ac15_heartbeat_by_non_owner_raises(client, queue, snapshot_id):
    queued_job(client, "fake_cls", snapshot_id)
    job = queue.claim("w-1", {"cpu"}, lease_seconds=30)
    with pytest.raises(LeaseLost):
        queue.heartbeat(job.id, "w-intruder", 30)
    assert queue.heartbeat(job.id, "w-1", 30) is False


# --------------------------------------------------------------------------- AC16


def test_ac16_gpu_job_not_claimed_by_cpu_worker(client, queue, snapshot_id):
    gpu_job = queued_job(client, "fake_gpu", snapshot_id)
    assert queue.claim("cpu-worker", {"cpu"}, lease_seconds=30) is None
    assert get_job(client, gpu_job)["state"] == "queued"

    job = queue.claim("gpu-worker", {"cpu", "gpu"}, lease_seconds=30)
    assert job is not None and same_id(job.id, gpu_job)


def test_ac16_cpu_worker_claims_next_eligible_job(client, queue, snapshot_id):
    gpu_job = queued_job(client, "fake_gpu", snapshot_id)
    cpu_job = queued_job(client, "fake_cls", snapshot_id)

    job = queue.claim("cpu-worker", {"cpu"}, lease_seconds=30)

    assert job is not None and same_id(job.id, cpu_job)
    assert get_job(client, gpu_job)["state"] == "queued"


def test_ac16_run_once_cpu_only_leaves_gpu_job(client, settings, registry, trainers, snapshot_id):
    gpu_job = queued_job(client, "fake_gpu", snapshot_id)
    assert run_once(settings, "cpu-worker", {"cpu"}, registry=registry) is False
    assert trainers.gpu.calls == []
    assert get_job(client, gpu_job)["state"] == "queued"


# --------------------------------------------------------------------------- AC17


def test_ac17_parallel_claims_never_return_same_job(client, settings, snapshot_id):
    from sparrow_finetune.jobs import JobQueue

    job_ids = {str(queued_job(client, "fake_cls", snapshot_id)) for _ in range(10)}
    barrier = threading.Barrier(2)
    claimed: dict[str, list[str]] = {"a": [], "b": []}
    errors: list[BaseException] = []

    def worker(name: str) -> None:
        q = JobQueue(settings.database_url)
        try:
            barrier.wait(timeout=10)
            while (job := q.claim(f"w-{name}", {"cpu"}, lease_seconds=60)) is not None:
                claimed[name].append(str(job.id))
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,), daemon=True) for n in claimed]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert not errors, errors
    a, b = claimed["a"], claimed["b"]
    assert len(a) == len(set(a)) and len(b) == len(set(b))
    assert set(a).isdisjoint(b), f"job claimed twice: {set(a) & set(b)}"
    assert set(a) | set(b) == job_ids
