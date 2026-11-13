"""FT-1: pins for the Phase 1A minor review items P1A-N1..N9 and the seed-range nit (P1C review).

See docs/review/implement-foundation/backlog.md and docs/review/ft1-minor-fixes/.
"""

from __future__ import annotations

import io
import os
import threading
import time

import psycopg
import pytest
from PIL import Image

from sparrow_finetune import db as db_module
from sparrow_finetune.jobs import CancelRequested, JobQueue, LeaseLost
from sparrow_finetune.pipelines import Registry
from sparrow_finetune.storage import LocalObjectStore
from sparrow_finetune.workers import run_forever, run_once

from .fakes import RecordingTrainer
from .helpers import (
    build_snapshot,
    create_dataset,
    create_workspace,
    get_job,
    job_events,
    make_client,
    png_bytes,
    queued_job,
)


def _image_bytes(fmt: str, size=(32, 24), **save) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (10, 120, 200)).save(buf, format=fmt, **save)
    return buf.getvalue()


def _model_rows(settings, job_id) -> int:
    with psycopg.connect(settings.database_url) as conn:
        return conn.execute(
            "SELECT count(*) FROM model_versions WHERE job_id = %s", (job_id,)
        ).fetchone()[0]


def _job_row(settings, job_id) -> dict:
    with psycopg.connect(settings.database_url) as conn:
        cur = conn.execute(
            "SELECT state, attempts, lease_owner FROM jobs WHERE id = %s", (job_id,)
        )
        return dict(zip(("state", "attempts", "lease_owner"), cur.fetchone()))


def _expire_lease(settings, job_id) -> None:
    with psycopg.connect(settings.database_url) as conn:
        conn.execute(
            "UPDATE jobs SET lease_expires_at = clock_timestamp() - interval '1 second'"
            " WHERE id = %s",
            (job_id,),
        )


def _event_messages(settings, job_id) -> list[str]:
    with psycopg.connect(settings.database_url) as conn:
        return [
            r[0]
            for r in conn.execute(
                "SELECT message FROM job_events WHERE job_id = %s ORDER BY seq",
                (job_id,),
            ).fetchall()
        ]


# --------------------------------------------------------------------------- N1 cancel vs complete


def test_n1_complete_refuses_job_with_pending_cancel(client, trainers, settings, queue):
    snap = build_snapshot(client)["snapshot"]
    job_id = queued_job(client, trainers.basic.id, snap["id"])
    job = queue.claim("w", {"cpu"})
    assert job is not None and str(job.id) == str(job_id)
    queue.request_cancel(job.id)
    model = {"metrics": {}, "format": "onnx", "object_key": "k", "sha256": "0" * 64}
    with pytest.raises(CancelRequested):
        queue.complete(job.id, "w", model, attempt=job.attempts)
    assert _job_row(settings, job.id)["state"] != "succeeded"
    assert _model_rows(settings, job.id) == 0


def test_n1_worker_cancel_requested_during_final_step_is_cancelled(
    client, trainers, registry, settings, monkeypatch
):
    snap = build_snapshot(client)["snapshot"]
    job_id = queued_job(client, trainers.basic.id, snap["id"])
    real_complete = JobQueue.complete

    def cancel_then_complete(self, jid, *args, **kwargs):
        JobQueue(settings.database_url).request_cancel(
            jid
        )  # lands after the last heartbeat
        return real_complete(self, jid, *args, **kwargs)

    monkeypatch.setattr(JobQueue, "complete", cancel_then_complete)
    assert run_once(settings, "w", {"cpu"}, registry)
    assert get_job(client, job_id)["state"] == "cancelled"
    assert _model_rows(settings, job_id) == 0
    stored = (
        list((settings.objects_dir / "models").rglob("*"))
        if (settings.objects_dir / "models").exists()
        else []
    )
    assert not [p for p in stored if p.is_file()], stored


# --------------------------------------------------------------------------- N2 attempt dir cleanup


def test_n2_attempt_dir_removed_after_success(client, trainers, registry, settings):
    snap = build_snapshot(client)["snapshot"]
    job_id = queued_job(client, trainers.basic.id, snap["id"])
    assert run_once(settings, "w", {"cpu"}, registry)
    assert get_job(client, job_id)["state"] == "succeeded"
    assert not (settings.jobs_dir / str(job_id) / "attempt-1").exists()


def test_n2_attempt_dir_removed_after_failure(client, trainers, registry, settings):
    snap = build_snapshot(client)["snapshot"]
    job_id = queued_job(client, trainers.failing.id, snap["id"])
    assert run_once(settings, "w", {"cpu"}, registry)
    assert get_job(client, job_id)["state"] == "failed"
    assert not (settings.jobs_dir / str(job_id) / "attempt-1").exists()


# --------------------------------------------------------------------------- N3 timeouts, local deadline


def test_n3_connect_sets_a_connect_timeout(monkeypatch):
    seen = {}

    def fake_connect(*args, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop")

    monkeypatch.setattr(db_module.psycopg, "connect", fake_connect)
    with pytest.raises(RuntimeError):
        db_module.connect("postgresql://x@127.0.0.1:1/db")
    assert "connect_timeout" in seen, seen
    assert 0 < int(seen["connect_timeout"]) <= 30


class _CheckLoopTrainer(RecordingTrainer):
    id = "check_loop"

    def __init__(self):
        super().__init__()
        self.raised: list = []
        self.elapsed = None

    def train(self, ctx):
        start = time.monotonic()
        try:
            while time.monotonic() - start < 10:
                ctx.check_cancelled()
                time.sleep(0.05)
        except BaseException as exc:
            self.raised.append(type(exc))
            self.elapsed = time.monotonic() - start
            raise


def test_n3_worker_gives_up_when_heartbeats_fail_past_the_lease(settings, monkeypatch):
    trainer = _CheckLoopTrainer()
    reg = Registry()
    reg.register(trainer)
    real = JobQueue.heartbeat
    broken = threading.Event()

    def flaky(self, job_id, worker_id, lease_seconds=60):
        if broken.is_set():
            raise psycopg.OperationalError("simulated database outage")
        return real(self, job_id, worker_id, lease_seconds)

    orig_train = trainer.train

    def train(ctx):
        broken.set()
        return orig_train(ctx)

    trainer.train = train
    monkeypatch.setattr(JobQueue, "heartbeat", flaky)
    with make_client(settings, reg) as client:
        snap = build_snapshot(client)["snapshot"]
        job_id = queued_job(client, trainer.id, snap["id"])
        assert run_once(settings, "w", {"cpu"}, reg, 1.0)
        assert trainer.raised == [LeaseLost], trainer.raised
        assert trainer.elapsed < 5, trainer.elapsed
        # The worker abandoned without writing a result.
        assert get_job(client, job_id)["state"] == "running"


# --------------------------------------------------------------------------- N4 fenced events


def test_n4_stale_attempt_cannot_append_events(client, trainers, settings, queue):
    snap = build_snapshot(client)["snapshot"]
    job_id = queued_job(client, trainers.basic.id, snap["id"])
    job = queue.claim("w", {"cpu"})
    queue.add_event(job.id, "train", "owned", worker_id="w", attempt=job.attempts)
    _expire_lease(settings, job.id)
    assert queue.reap_expired() == 1
    with pytest.raises(LeaseLost):
        queue.add_event(job.id, "train", "stale-after-reap", worker_id="w", attempt=1)
    job2 = queue.claim("w", {"cpu"})
    assert job2.attempts == 2
    with pytest.raises(LeaseLost):
        queue.add_event(
            job.id, "train", "stale-after-reclaim", worker_id="w", attempt=1
        )
    with pytest.raises(LeaseLost):
        queue.add_event(job.id, "train", "wrong-owner", worker_id="other", attempt=2)
    queue.add_event(job.id, "train", "new-owner", worker_id="w", attempt=2)
    msgs = _event_messages(settings, job_id)
    assert "owned" in msgs and "new-owner" in msgs
    assert not {"stale-after-reap", "stale-after-reclaim", "wrong-owner"} & set(msgs)


def test_n4_worker_terminal_events_still_recorded(client, trainers, registry, settings):
    snap = build_snapshot(client)["snapshot"]
    job_id = queued_job(client, trainers.basic.id, snap["id"])
    assert run_once(settings, "w", {"cpu"}, registry)
    stages = [e["stage"] for e in job_events(client, job_id)]
    assert stages[0] == "prepare" and stages[-1] == "succeeded", stages


# --------------------------------------------------------------------------- N5 storage durability


def _tmp_files(root):
    return [p for p in root.rglob("*") if p.name.endswith(".tmp")]


def test_n5_put_fsyncs_before_rename(tmp_path, monkeypatch):
    calls = []
    real = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (calls.append(fd), real(fd))[1])
    store = LocalObjectStore(tmp_path / "objects")
    store.put_bytes("a/b", b"data")
    n_bytes = len(calls)
    src = tmp_path / "src.bin"
    src.write_bytes(b"x" * 10)
    store.put_file("a/c", src)
    assert n_bytes >= 1 and len(calls) > n_bytes, calls


@pytest.mark.parametrize("method", ["put_bytes", "put_file"])
def test_n5_failed_put_leaves_no_tmp_file(tmp_path, monkeypatch, method):
    store = LocalObjectStore(tmp_path / "objects")
    src = tmp_path / "src.bin"
    src.write_bytes(b"payload")

    def boom(*a, **k):
        raise OSError("simulated rename failure")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        if method == "put_bytes":
            store.put_bytes("x/y", b"payload")
        else:
            store.put_file("x/y", src)
    assert _tmp_files(tmp_path / "objects") == []


def test_n5_stale_tmp_swept_fresh_kept(tmp_path):
    root = tmp_path / "objects"
    store = LocalObjectStore(root)
    store.put_bytes("images/ab/real", b"keep me")
    old = root / "images" / "ab" / ".real.deadbeef.tmp"
    old.write_bytes(b"partial")
    past = time.time() - 7200
    os.utime(old, (past, past))
    fresh = root / "images" / "ab" / ".other.cafe.tmp"
    fresh.write_bytes(b"in flight")
    store.remove_stale_tmp(max_age_seconds=3600)
    assert not old.exists()
    assert fresh.exists()
    assert (root / "images" / "ab" / "real").read_bytes() == b"keep me"


def test_n5_worker_startup_sweeps_stale_tmp(settings):
    stale = settings.objects_dir / "models" / ".x.0.tmp"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"partial")
    past = time.time() - 7 * 24 * 3600
    os.utime(stale, (past, past))
    stop = threading.Event()
    stop.set()
    run_forever(settings, "w", {"cpu"}, Registry(), stop=stop)
    assert not stale.exists()


# --------------------------------------------------------------------------- N6 list datasets 404


def test_n6_list_datasets_unknown_workspace_404(client):
    r = client.get("/v1/workspaces/987654/datasets")
    assert r.status_code == 404, (r.status_code, r.text)


def test_n6_list_datasets_known_empty_workspace_200(client):
    ws = create_workspace(client)
    r = client.get(f"/v1/workspaces/{ws['id']}/datasets")
    assert r.status_code == 200 and r.json()["datasets"] == []


# --------------------------------------------------------------------------- N7 image format checks


def _put(client, ds_id, name, data):
    return client.put(f"/v1/datasets/{ds_id}/images/{name}", content=data)


@pytest.mark.parametrize(
    "fmt,name",
    [
        ("JPEG", "a.jpg"),
        ("PNG", "a.png"),
        ("BMP", "a.bmp"),
        ("TIFF", "a.tif"),
        ("GIF", "a.gif"),
        ("WEBP", "a.webp"),
    ],
)
def test_n7_common_formats_accepted(client, fmt, name):
    ds = create_dataset(client, create_workspace(client)["id"])
    r = _put(client, ds["id"], name, _image_bytes(fmt))
    assert r.is_success, (fmt, r.status_code, r.text)
    assert (r.json()["width"], r.json()["height"]) == (32, 24)


@pytest.mark.parametrize("fmt", ["PPM", "ICO", "PCX"])
def test_n7_unlisted_formats_rejected(client, fmt):
    ds = create_dataset(client, create_workspace(client)["id"])
    r = _put(client, ds["id"], "a.img", _image_bytes(fmt))
    assert r.status_code == 422, (fmt, r.status_code, r.text)


def test_n7_truncated_jpeg_rejected(client):
    ds = create_dataset(client, create_workspace(client)["id"])
    data = _image_bytes("JPEG", size=(256, 256), quality=95)
    r = _put(client, ds["id"], "t.jpg", data[: len(data) // 2])
    assert r.status_code == 422, (r.status_code, r.text)


# --------------------------------------------------------------------------- N8 advisory lock space


def test_n8_add_event_not_blocked_by_foreign_advisory_lock(
    client, trainers, settings, queue
):
    snap = build_snapshot(client)["snapshot"]
    job_id = int(queued_job(client, trainers.basic.id, snap["id"]))
    other = psycopg.connect(settings.database_url)
    try:
        other.execute("SELECT pg_advisory_lock(%s)", (job_id,))  # another app's key
        done = threading.Event()
        errors: list = []

        def add():
            try:
                queue.add_event(job_id, "note", "hello")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
            done.set()

        t = threading.Thread(target=add, daemon=True)
        t.start()
        assert done.wait(5), "add_event blocked on an unrelated advisory lock"
        assert not errors, errors
    finally:
        other.close()


# --------------------------------------------------------------------------- N9 chunked upload


def test_n9_chunked_upload_within_cap_accepted(client):
    ds = create_dataset(client, create_workspace(client)["id"])
    data = png_bytes(3)

    def body():
        for i in range(0, len(data), 7):
            yield data[i : i + 7]

    r = client.put(f"/v1/datasets/{ds['id']}/images/c.png", content=body())
    assert r.is_success, (r.status_code, r.text)
    assert r.json()["width"] == 64


# --------------------------------------------------------------------------- seed range (P1C nit)


def _pipelines():
    from sparrow_finetune.pipelines.dinov3_linear import DinoV3LinearTrainer
    from sparrow_finetune.pipelines.yolo_detector import YoloDetectorTrainer

    return [DinoV3LinearTrainer(lambda: None), YoloDetectorTrainer()]


@pytest.mark.parametrize("idx", [0, 1])
def test_seed_at_or_above_2_32_rejected(idx):
    trainer = _pipelines()[idx]
    with pytest.raises(ValueError):
        trainer.validate_params({"seed": 2**32})
    assert trainer.validate_params({"seed": 2**32 - 1})["seed"] == 2**32 - 1


def test_seed_out_of_range_rejected_at_submit(settings):
    from sparrow_finetune.pipelines.dinov3_linear import DinoV3LinearTrainer

    reg = Registry()
    reg.register(DinoV3LinearTrainer(lambda: None))
    with make_client(settings, reg) as c:
        snap = build_snapshot(c)["snapshot"]
        r = c.post(
            "/v1/jobs",
            json={
                "pipeline": "dinov3_linear",
                "snapshot_id": snap["id"],
                "params": {"seed": 2**32},
            },
        )
    assert r.status_code == 422, (r.status_code, r.text)
