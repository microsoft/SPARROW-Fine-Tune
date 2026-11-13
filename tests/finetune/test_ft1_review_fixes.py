"""Regression tests for the FT-1 review findings (docs/review/ft1-minor-fixes/report.md)."""

import io
import os
import time

import psycopg
from PIL import Image

from sparrow_finetune import storage, workers
from sparrow_finetune.jobs import JobQueue
from sparrow_finetune.storage import LocalObjectStore
from sparrow_finetune.workers import run_once

from .helpers import (
    build_snapshot,
    create_dataset,
    create_workspace,
    get_job,
    queued_job,
)


def test_r_i1_cancel_seen_by_final_beat_leaves_no_orphan_model(
    client, trainers, registry, settings, monkeypatch
):
    snap = build_snapshot(client)["snapshot"]
    job_id = queued_job(client, trainers.basic.id, snap["id"])
    real_store = workers._store_artifact

    def store_then_cancel(s, job, result):
        out = real_store(s, job, result)
        JobQueue(settings.database_url).request_cancel(job.id)
        return out

    monkeypatch.setattr(workers, "_store_artifact", store_then_cancel)
    assert run_once(settings, "w", {"cpu"}, registry, lease_seconds=600)
    assert get_job(client, job_id)["state"] == "cancelled"
    models = settings.objects_dir / "models"
    stored = [p for p in models.rglob("*") if p.is_file()] if models.exists() else []
    assert not stored, f"orphaned model artifact(s): {stored}"


def test_r_i2_stale_tmp_sweep_does_not_break_concurrent_link_to(tmp_path, monkeypatch):
    objects = LocalObjectStore(tmp_path / "objects")
    objects.put_bytes("images/ab/abc", b"x" * 10)
    old = time.time() - 7200
    os.utime(objects.path("images/ab/abc"), (old, old))
    staged_root = tmp_path / "staged"
    real_replace = os.replace

    def sweep_then_replace(src, dst):
        LocalObjectStore(staged_root).remove_stale_tmp()
        return real_replace(src, dst)

    monkeypatch.setattr(storage.os, "replace", sweep_then_replace)
    objects.link_to("images/ab/abc", staged_root / "ab" / "abc.png")
    assert (staged_root / "ab" / "abc.png").is_file()


def test_r_i2_sweep_still_removes_old_abandoned_tmp(tmp_path):
    store = LocalObjectStore(tmp_path)
    tmp = tmp_path / "ab" / ".abc.png.dead.tmp"
    tmp.parent.mkdir()
    tmp.write_bytes(b"x")
    time.sleep(0.05)  # ctime cannot be backdated; use a zero age limit instead
    assert store.remove_stale_tmp(max_age_seconds=0) == 1
    assert not tmp.exists()


def test_r_m1_cleanup_keeps_the_job_dir_for_a_concurrent_attempt(
    client, trainers, registry, settings
):
    """A stale attempt's cleanup must not rmdir ``jobs/<id>``: the new owner may be mid-mkdir."""
    snap = build_snapshot(client)["snapshot"]
    job_id = queued_job(client, trainers.basic.id, snap["id"])
    assert run_once(settings, "w", {"cpu"}, registry)
    assert get_job(client, job_id)["state"] == "succeeded"
    job_dir = settings.jobs_dir / str(job_id)
    assert job_dir.is_dir()
    assert not (job_dir / "attempt-1").exists()


def test_r_m3_new_attempt_removes_dirs_of_older_attempts(
    client, trainers, registry, settings, queue
):
    snap = build_snapshot(client)["snapshot"]
    job_id = queued_job(client, trainers.basic.id, snap["id"])
    job = queue.claim("dead", {"cpu"})
    assert job is not None and job.attempts == 1
    leftover = settings.jobs_dir / str(job_id) / "attempt-1" / "work" / "big.bin"
    leftover.parent.mkdir(parents=True)
    leftover.write_bytes(b"x" * 100)
    with psycopg.connect(settings.database_url) as conn:
        conn.execute(
            "UPDATE jobs SET lease_expires_at = clock_timestamp() - interval '1 second'"
            " WHERE id = %s",
            (job_id,),
        )
    assert queue.reap_expired() == 1
    assert run_once(settings, "w", {"cpu"}, registry)
    assert get_job(client, job_id)["state"] == "succeeded"
    assert not (settings.jobs_dir / str(job_id) / "attempt-1").exists()


def test_r_m2_huge_pixel_count_rejected_before_decode(client):
    ds = create_dataset(client, create_workspace(client)["id"])
    buf = io.BytesIO()
    Image.new("L", (9000, 9000)).save(buf, format="PNG")
    r = client.put(f"/v1/datasets/{ds['id']}/images/huge.png", content=buf.getvalue())
    assert r.status_code == 422, (r.status_code, r.text)
    assert "pixels" in r.text
