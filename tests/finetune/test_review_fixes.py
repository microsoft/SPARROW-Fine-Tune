"""Regression tests for the verified Important findings of the Phase 1A review.

See docs/review/implement-foundation/review.md (I1-I7). Each test pins observable behaviour.
"""

import hashlib
import threading
import time

import psycopg
from fastapi.testclient import TestClient

from sparrow_finetune.api import create_app
from sparrow_finetune.config import Settings
from sparrow_finetune.db import migrate
from sparrow_finetune.jobs import JobQueue
from sparrow_finetune.pipelines import ExportResult, Registry
from sparrow_finetune.workers import run_once
from .conftest import create_database, drop_database
from .fakes import RecordingTrainer
from .helpers import (
    annotation_item,
    build_snapshot,
    create_dataset,
    create_snapshot,
    create_workspace,
    get_job,
    import_annotations,
    make_client,
    png_bytes,
    queued_job,
    upload_image,
)

NAN_BBOX = '{"items":[{"image":"img0.png","annotations":[{"label":"cat","bbox":[NaN,0,1,1]}]}]}'


def _detection_ds_with_image(client):
    ws = create_workspace(client)
    ds = create_dataset(client, ws["id"], task="detection")
    upload_image(client, ds["id"], "img0.png", seed=0)  # 64x48
    return ds


def test_i2_nan_bbox_is_rejected_422(client):
    ds = _detection_ds_with_image(client)
    r = client.post(
        f"/v1/datasets/{ds['id']}/annotations",
        content=NAN_BBOX,
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 422, (r.status_code, r.text)


def test_i2_snapshot_after_nan_bbox_import_is_not_500(settings, registry):
    with TestClient(create_app(settings, registry), raise_server_exceptions=False) as client:
        ds = _detection_ds_with_image(client)
        client.post(
            f"/v1/datasets/{ds['id']}/annotations",
            content=NAN_BBOX,
            headers={"content-type": "application/json"},
        )
        r = create_snapshot(client, ds["id"])
        assert r.status_code in (201, 422), (r.status_code, r.text)


def test_i2_nan_param_submit_is_not_500(settings, registry, trainers):
    with TestClient(create_app(settings, registry), raise_server_exceptions=False) as client:
        snap = build_snapshot(client)["snapshot"]
        body = '{"pipeline":"%s","snapshot_id":%s,"params":{"epochs":1,"lr":NaN}}' % (
            trainers.basic.id,
            snap["id"],
        )
        r = client.post("/v1/jobs", content=body, headers={"content-type": "application/json"})
        assert r.status_code in (201, 422), (r.status_code, r.text)


class NanMetricTrainer(RecordingTrainer):
    id = "nan_metric"

    def evaluate(self, ctx):
        return {"accuracy": 0.9, "map50": float("nan")}  # legal dict[str, float]


def test_i2_nan_metric_does_not_fail_trained_job(settings):
    reg = Registry()
    reg.register(NanMetricTrainer())
    with make_client(settings, reg) as client:
        snap = build_snapshot(client)["snapshot"]
        job_id = queued_job(client, "nan_metric", snap["id"])
        assert run_once(settings, "w", {"cpu"}, reg)
        job = get_job(client, job_id)
        assert job["state"] == "succeeded", job["error"]


def test_i5_max_length_image_name_is_trainable(client, settings, registry, trainers):
    ws = create_workspace(client)
    ds = create_dataset(client, ws["id"], whole_image=True)
    name = "x" * 251 + ".png"  # 255 chars: _validate_name allows <= 255
    upload_image(client, ds["id"], name, seed=1)
    assert import_annotations(client, ds["id"], [annotation_item(name, "cat")]).status_code == 200
    snap = create_snapshot(client, ds["id"])
    assert snap.status_code == 201, snap.text
    job_id = queued_job(client, trainers.basic.id, snap.json()["id"])
    assert run_once(settings, "w", {"cpu"}, registry)
    job = get_job(client, job_id)
    assert job["state"] == "succeeded", job["error"]


def test_i6_reupload_smaller_image_never_freezes_bbox_outside(client, settings):
    ws = create_workspace(client)
    ds = create_dataset(client, ws["id"], task="detection")
    upload_image(client, ds["id"], "a.png", seed=0)  # 64x48
    r = import_annotations(client, ds["id"], [annotation_item("a.png", "cat", [30, 20, 30, 25])])
    assert r.status_code == 200, r.text
    r = client.put(f"/v1/datasets/{ds['id']}/images/a.png", content=png_bytes(1, size=(16, 16)))
    if r.status_code in (409, 422):
        return  # rejecting a re-upload that invalidates existing annotations is acceptable
    assert r.status_code == 200 and r.json()["width"] == 16, r.text
    snap = create_snapshot(client, ds["id"])
    if snap.status_code == 422:
        return  # annotations were invalidated by the re-upload; nothing left to snapshot
    assert snap.status_code == 201, snap.text
    with psycopg.connect(settings.database_url) as conn:
        w, h, anns = conn.execute(
            "SELECT width, height, annotations FROM snapshot_items WHERE snapshot_id = %s",
            (snap.json()["id"],),
        ).fetchone()
    for ann in anns:
        if ann.get("bbox") is None:
            continue
        x, y, bw, bh = ann["bbox"]
        assert x + bw <= w and y + bh <= h, f"snapshot item {w}x{h} carries bbox {ann['bbox']}"
class StallingExportTrainer(RecordingTrainer):
    id = "stall_export"

    def __init__(self):
        super().__init__()
        self.exports = 0
        self.in_export = threading.Event()
        self.release = threading.Event()

    def export(self, ctx):
        self.exports += 1
        call = self.exports
        if call == 1:  # worker A: slow export (e.g. ONNX conversion)
            self.in_export.set()
            assert self.release.wait(30)
        path = ctx.workdir / "model.onnx"
        path.write_bytes(f"weights-from-export-call-{call}".encode())
        return ExportResult(path=path, format="onnx")


def test_i1_stale_worker_does_not_clobber_new_owners_artifact(settings, monkeypatch):
    trainer = StallingExportTrainer()
    reg = Registry()
    reg.register(trainer)
    real_heartbeat = JobQueue.heartbeat
    blip = threading.Event()

    def flaky_heartbeat(self, job_id, worker_id, lease_seconds=60):
        # Worker A's DB link drops; _Heartbeat.beat logs it as transient and retries.
        if worker_id == "worker-A" and blip.is_set():
            raise psycopg.OperationalError("simulated connection loss")
        return real_heartbeat(self, job_id, worker_id, lease_seconds)

    monkeypatch.setattr(JobQueue, "heartbeat", flaky_heartbeat)
    with make_client(settings, reg) as client:
        snap = build_snapshot(client)["snapshot"]
        job_id = queued_job(client, trainer.id, snap["id"])
        a = threading.Thread(
            target=run_once, args=(settings, "worker-A", {"cpu"}, reg, 1.0)
        )
        a.start()
        assert trainer.in_export.wait(15)
        blip.set()
        time.sleep(1.6)  # A's 1 s lease expires
        assert run_once(settings, "worker-B", {"cpu"}, reg, 60)  # reap, re-claim, finish
        assert get_job(client, job_id)["state"] == "succeeded"
        trainer.release.set()
        a.join(30)
        models = client.get("/v1/models", params={"job_id": job_id}).json()["models"]
        assert len(models) == 1  # no duplicate model version (guarded)
        blob = client.get(f"/v1/models/{models[0]['id']}/artifact").content
        print("stored bytes:", blob, "recorded sha256:", models[0]["sha256"][:12])
        assert hashlib.sha256(blob).hexdigest() == models[0]["sha256"], blob


# --------------------------------------------------------------------------- I3 upload cap


def test_i3_declared_oversize_upload_rejected_413(client):
    ws = create_workspace(client)
    ds = create_dataset(client, ws["id"])
    r = client.put(
        f"/v1/datasets/{ds['id']}/images/big.png",
        content=b"\0" * 1024,
        headers={"content-length": str(65 * 1024 * 1024)},
    )
    assert r.status_code == 413, (r.status_code, r.text)


def test_i3_streamed_oversize_upload_stops_reading_early(client, settings, registry):
    """A chunked 200 MiB upload is cut off with 413 soon after the 64 MiB cap.

    Runs against a real uvicorn server: Starlette's TestClient reads the whole request body
    before the app sees any of it, so it cannot show how much the server actually consumed.
    """
    import asyncio
    import select
    import socket

    import uvicorn

    ws = create_workspace(client)
    ds = create_dataset(client, ws["id"])

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(create_app(settings, registry=registry), log_level="warning")
    )
    thread = threading.Thread(
        target=lambda: asyncio.run(server.serve(sockets=[sock])), daemon=True
    )
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started:
            assert time.monotonic() < deadline, "uvicorn did not start"
            time.sleep(0.02)

        chunk = b"\0" * (1024 * 1024)
        framed = f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n"
        sent = 0
        response = b""
        with socket.create_connection(("127.0.0.1", port), timeout=10) as conn:
            conn.sendall(
                (
                    f"PUT /v1/datasets/{ds['id']}/images/big.png HTTP/1.1\r\n"
                    f"Host: 127.0.0.1:{port}\r\n"
                    "Content-Type: application/octet-stream\r\n"
                    "Transfer-Encoding: chunked\r\n\r\n"
                ).encode()
            )
            try:
                for _ in range(200):  # up to 200 MiB, well past the 64 MiB cap
                    if select.select([conn], [], [], 0)[0]:
                        break  # the server has answered; stop sending
                    conn.sendall(framed)
                    sent += len(chunk)
                else:
                    conn.sendall(b"0\r\n\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass  # the server closed the connection after refusing the body
            try:
                while b"\r\n" not in response:
                    data = conn.recv(65536)
                    if not data:
                        break
                    response += data
            except ConnectionResetError:
                pass
        status_line = response.split(b"\r\n", 1)[0]
        assert status_line.split()[1:2] == [b"413"], response[:200]
        assert sent <= 80 * 1024 * 1024, f"server consumed {sent} bytes past the cap"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


# --------------------------------------------------------------------------- I4 concurrent migrate


def test_i4_concurrent_first_migrate_does_not_fail(pg_admin_url):
    for _ in range(5):
        dbname, url = create_database(pg_admin_url)
        try:
            errors: list = []
            barrier = threading.Barrier(4)

            def run():
                try:
                    barrier.wait()
                    migrate(url)
                except Exception as exc:  # noqa: BLE001
                    errors.append(repr(exc))

            threads = [threading.Thread(target=run) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(60)
            assert errors == []
            assert migrate(url) == []
        finally:
            drop_database(pg_admin_url, dbname)


# --------------------------------------------------------------------------- I7 token normalisation


def test_i7_string_api_tokens_is_not_split_into_characters(settings, registry):
    s = Settings(database_url=settings.database_url, data_dir=settings.data_dir, api_tokens="s3cret-token")
    with make_client(s, registry) as client:
        assert client.get("/v1/pipelines", headers={"Authorization": "Bearer s"}).status_code == 401
        assert client.get("/v1/pipelines", headers={"Authorization": "Bearer 3"}).status_code == 401
        ok = client.get("/v1/pipelines", headers={"Authorization": "Bearer s3cret-token"})
        assert ok.status_code == 200, ok.text


def test_species_only_labels_on_detection_dataset_get_one_clear_message(client):
    """Manual test 2026-10-27: a classification label file uploaded to a detection
    dataset produced one identical error line per image. The 422 must instead
    explain the problem once and keep at most a few example lines."""
    ws = create_workspace(client)
    ds = create_dataset(client, ws["id"], task="detection")
    names = [f"img{i}.png" for i in range(8)]
    for i, name in enumerate(names):
        upload_image(client, ds["id"], name, seed=i)
    items = [{"image": n, "annotations": [{"label": "Bos"}]} for n in names]
    r = client.post(f"/v1/datasets/{ds['id']}/annotations", json={"items": items})
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    message = detail["message"]
    assert "8 of 8 images" in message
    assert "without a box" in message
    assert "classification dataset" in message
    assert message.count("require a bbox") == 0
    assert len(detail["errors"]) <= 5
    assert detail["error_count"] == 8
