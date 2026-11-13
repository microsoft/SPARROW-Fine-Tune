"""Phase 1C end to end: the ``yolo_detector`` pipeline through HTTP + ``run_once`` on real PostgreSQL.

Covers C11 (full client journey on the CPU: detection dataset of synthetic coloured squares with groups ->
snapshot -> job -> succeeded, one model version, downloaded artifact sha256 matches, bundle holds
model.onnx / manifest.json / model_card.json, model.onnx loads in onnxruntime with input [1, 3, 64, 64]),
C5 through the worker (all boxes in one split -> job ``failed`` naming the empty split), cancellation
during train (-> ``cancelled``, no model version) and C12 (``yolo_detector`` on a classification
snapshot -> 422 with the task-mismatch message). The registry is built by ``register_builtin_pipelines``
exactly as the CLI does and is shared by the API app and the worker. ``settings`` comes from conftest.py.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np
import pytest

from .dinov3_helpers import bundle_json, read_bundle_bytes
from .helpers import (
    STAGES,
    as_list,
    build_snapshot,
    create_dataset,
    create_job,
    create_workspace,
    get_job,
    import_annotations,
    job_events,
    make_client,
    models_for_job,
    stage_sequence,
)
from .yolo_helpers import CPU_PARAMS, DET_SIZE, DetItem, det_annotation_payload, det_items


@pytest.fixture(scope="module", autouse=True)
def yolo_offline():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("YOLO_OFFLINE", "1")
        yield


def builtin_registry(settings):
    from sparrow_finetune.pipelines import Registry, register_builtin_pipelines

    reg = Registry()
    register_builtin_pipelines(reg, settings)
    assert reg.get("yolo_detector") is not None, f"yolo_detector not registered: {reg.ids()}"
    return reg


def upload_detection_snapshot(client, syn: list[DetItem]) -> int:
    ws = create_workspace(client)
    ds = create_dataset(client, ws["id"], task="detection")
    for it in syn:
        r = client.put(f"/v1/datasets/{ds['id']}/images/{it.name}", content=it.png)
        assert r.is_success, r.text
    r = import_annotations(client, ds["id"], det_annotation_payload(syn))
    assert r.status_code == 200, r.text
    snap = client.post(f"/v1/datasets/{ds['id']}/snapshots")
    assert snap.status_code == 201, snap.text
    return snap.json()["id"]


def submit(client, snapshot_id, params) -> int:
    r = client.post("/v1/jobs", json={"pipeline": "yolo_detector", "snapshot_id": snapshot_id, "params": params})
    assert r.status_code == 201, r.text
    return r.json()["id"]


# --------------------------------------------------------------------------- C11


def test_c11_end_to_end_yolo_detector_job_on_cpu(settings):
    import onnxruntime as ort

    from sparrow_finetune.workers import run_once

    registry = builtin_registry(settings)
    syn = det_items(prefix="e2e-")
    boxed = [it for it in syn if it.boxes]
    assert len(boxed) >= 16 and {lb for it in boxed for lb, _ in it.boxes} == {"cat", "dog"}

    with make_client(settings, registry) as client:
        pipelines = {p["id"]: p for p in as_list(client.get("/v1/pipelines").json(), "pipelines")}
        assert pipelines["yolo_detector"]["status"] == "available"
        assert pipelines["yolo_detector"]["task"] == "detection"

        job_id = submit(client, upload_detection_snapshot(client, syn), dict(CPU_PARAMS))
        assert run_once(settings, "yolo-e2e-worker", {"cpu"}, registry=registry) is True
        job = get_job(client, job_id)
        assert job["state"] == "succeeded", job
        assert stage_sequence(job_events(client, job_id)) == STAGES

        (model,) = models_for_job(client, job_id)
        assert model["format"] == "onnx-bundle"
        art = client.get(f"/v1/models/{model['id']}/artifact")
        assert art.status_code == 200
        assert hashlib.sha256(art.content).hexdigest() == model["sha256"]

    bundle = read_bundle_bytes(art.content)
    assert {"model.onnx", "manifest.json", "model_card.json"} <= set(bundle)
    manifest = bundle_json(bundle, "manifest.json")
    assert manifest["pipeline"] == "yolo_detector"
    assert manifest["labels"] == ["cat", "dog"]

    sess = ort.InferenceSession(bundle["model.onnx"], providers=["CPUExecutionProvider"])
    (inp,) = sess.get_inputs()
    assert list(inp.shape) == [1, 3, DET_SIZE, DET_SIZE]
    outs = sess.run(None, {inp.name: np.zeros((1, 3, DET_SIZE, DET_SIZE), dtype=np.float32)})
    assert outs and all(np.isfinite(np.asarray(o)).all() for o in outs)


# --------------------------------------------------------------------------- C5 through the worker


def test_c5_job_fails_when_every_box_lands_in_train(settings):
    from sparrow_finetune.workers import run_once

    registry = builtin_registry(settings)
    syn = det_items(train_groups=3, val_groups=0, backgrounds=0, prefix="c5w-")
    with make_client(settings, registry) as client:
        job_id = submit(client, upload_detection_snapshot(client, syn), dict(CPU_PARAMS))
        assert run_once(settings, "yolo-c5-worker", {"cpu"}, registry=registry) is True
        job = get_job(client, job_id)
        assert job["state"] == "failed", job
        assert "val" in str(job["error"]).lower(), job["error"]
        assert models_for_job(client, job_id) == []


# --------------------------------------------------------------------------- cancellation through the worker


def test_c6_cancel_during_train_cancels_job_without_model(settings):
    from sparrow_finetune.jobs import JobQueue
    from sparrow_finetune.pipelines import JobCancelled, Registry
    from sparrow_finetune.pipelines.yolo_detector import YoloDetectorTrainer
    from sparrow_finetune.workers import run_once

    class CancelRequestedAtTrainStart(YoloDetectorTrainer):
        """Requests cancellation, waits until the worker has observed it, then trains as usual."""

        def train(self, ctx):
            JobQueue(settings.database_url).request_cancel(ctx.job_id)
            deadline = time.monotonic() + 15
            while True:
                try:
                    ctx.check_cancelled()
                except JobCancelled:
                    break
                assert time.monotonic() < deadline, "worker never observed the cancel request"
                time.sleep(0.05)
            return super().train(ctx)

    registry = Registry()
    registry.register(CancelRequestedAtTrainStart())
    syn = det_items(prefix="cxl-")
    with make_client(settings, registry) as client:
        job_id = submit(client, upload_detection_snapshot(client, syn), {**CPU_PARAMS, "epochs": 5})
        assert run_once(settings, "yolo-cancel-worker", {"cpu"}, registry=registry, lease_seconds=0.9) is True
        assert get_job(client, job_id)["state"] == "cancelled"
        assert models_for_job(client, job_id) == []


# --------------------------------------------------------------------------- C12


def test_c12_yolo_detector_rejects_classification_snapshot(settings):
    registry = builtin_registry(settings)
    with make_client(settings, registry) as client:
        snap = build_snapshot(client, task="classification")["snapshot"]
        r = create_job(client, "yolo_detector", snap["id"], dict(CPU_PARAMS))
        assert 400 <= r.status_code < 500, r.text
        detail = r.text
        assert "yolo_detector" in detail
        assert "trains detection models" in detail
        assert "classification dataset" in detail
