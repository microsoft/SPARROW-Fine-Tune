"""AC11-AC14: the worker's ``run_once`` orchestration, results and failure modes.

Fake trainers record the stages they ran; the job's state, events and model
version are then read back over HTTP, i.e. the way a client would see them.
"""

from __future__ import annotations

import hashlib
import threading

from sparrow_finetune.workers import run_once

from .fakes import EVAL_METRICS, FAIL_MESSAGE, TRAIN_LOSS
from .helpers import (
    STAGES,
    as_list,
    build_snapshot,
    create_workspace,
    dumps,
    find_key,
    get_job,
    job_events,
    models_for_job,
    queued_job,
    same_id,
    stage_sequence,
)


def _snapshot_id(client) -> str:
    return build_snapshot(client)["snapshot"]["id"]


# --------------------------------------------------------------------------- AC11


def test_ac11_run_once_returns_false_when_queue_empty(settings, registry):
    assert run_once(settings, "w-1", {"cpu"}, registry=registry) is False


def test_ac11_run_once_runs_stages_in_order_and_succeeds(client, settings, registry, trainers):
    job_id = queued_job(client, "fake_cls", _snapshot_id(client), {"epochs": 2})

    assert run_once(settings, "w-1", {"cpu"}, registry=registry) is True

    assert trainers.basic.calls == STAGES
    assert trainers.basic.job_ids == [str(job_id)]
    assert trainers.basic.params[0]["epochs"] == 2
    assert get_job(client, job_id)["state"] == "succeeded"
    assert run_once(settings, "w-1", {"cpu"}, registry=registry) is False


def test_ac11_events_cover_stages_in_order_with_emitted_metrics(client, settings, registry):
    job_id = queued_job(client, "fake_cls", _snapshot_id(client))
    assert run_once(settings, "w-1", {"cpu"}, registry=registry) is True

    events = job_events(client, job_id)

    assert stage_sequence(events) == STAGES
    train_events = [e for e in events if e.get("stage") == "train"]
    emitted = [e for e in train_events if TRAIN_LOSS in find_key(e, "loss")]
    assert emitted, f"trainer emit(loss={TRAIN_LOSS}) missing from train events: {dumps(train_events)}"
    assert "epoch 1" in dumps(emitted)


def test_ac11_success_creates_model_version(client, settings, registry):
    job_id = queued_job(client, "fake_cls", _snapshot_id(client))
    assert run_once(settings, "w-1", {"cpu"}, registry=registry) is True
    assert len(models_for_job(client, job_id)) == 1


# --------------------------------------------------------------------------- AC12


def test_ac12_model_fields_and_artifact_hash(client, settings, registry, trainers):
    snapshot_id = _snapshot_id(client)
    job_id = queued_job(client, "fake_cls", snapshot_id)
    assert run_once(settings, "w-1", {"cpu"}, registry=registry) is True

    (listed,) = models_for_job(client, job_id)
    r = client.get(f"/v1/models/{listed['id']}")
    assert r.status_code == 200, r.text
    model = r.json()

    for key in ("id", "job_id", "pipeline", "snapshot_id", "metrics", "format", "sha256"):
        assert key in model, f"model is missing {key!r}: {model}"
        assert model[key] == listed[key]
    assert same_id(model["job_id"], job_id)
    assert model["pipeline"] == "fake_cls"
    assert same_id(model["snapshot_id"], snapshot_id)
    assert model["metrics"] == EVAL_METRICS
    assert model["format"] == "onnx"

    art = client.get(f"/v1/models/{model['id']}/artifact")
    assert art.status_code == 200
    assert hashlib.sha256(art.content).hexdigest() == model["sha256"]
    assert art.content == trainers.basic.exported


# --------------------------------------------------------------------------- AC13


def test_ac13_trainer_exception_fails_job_with_error(client, settings, registry, trainers):
    job_id = queued_job(client, "fake_fail", _snapshot_id(client))

    assert run_once(settings, "w-1", {"cpu"}, registry=registry) is True

    job = get_job(client, job_id)
    assert job["state"] == "failed"
    assert FAIL_MESSAGE in str(job["error"])
    assert trainers.failing.calls == ["prepare", "train"]
    assert models_for_job(client, job_id) == []


# --------------------------------------------------------------------------- AC14


def test_ac14_cancel_running_job_stops_at_check_cancelled(client, settings, registry, trainers):
    job_id = queued_job(client, "fake_block", _snapshot_id(client))
    blocking = trainers.blocking
    outcome: dict = {}

    def work():
        try:
            outcome["claimed"] = run_once(
                settings, "w-1", {"cpu"}, registry=registry, lease_seconds=3
            )
        except BaseException as exc:  # surfaced in the main thread below
            outcome["error"] = exc

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    try:
        assert blocking.started.wait(timeout=20), "trainer.train never started"
        assert get_job(client, job_id)["state"] == "running"
        r = client.post(f"/v1/jobs/{job_id}/cancel")
        assert r.is_success, r.text
    finally:
        blocking.release.set()
        worker.join(timeout=40)

    assert not worker.is_alive(), "run_once did not return after cancel"
    assert "error" not in outcome, f"run_once raised: {outcome.get('error')!r}"
    assert outcome["claimed"] is True
    assert get_job(client, job_id)["state"] == "cancelled"
    assert "export" not in blocking.calls
    assert models_for_job(client, job_id) == []


def test_ac14_heartbeat_reports_cancel_request(client, queue):
    job_id = queued_job(client, "fake_cls", _snapshot_id(client))
    job = queue.claim("w-1", {"cpu"}, lease_seconds=30)
    assert job is not None and same_id(job.id, job_id)

    assert queue.heartbeat(job.id, "w-1", 30) is False
    assert client.post(f"/v1/jobs/{job_id}/cancel").is_success
    assert queue.heartbeat(job.id, "w-1", 30) is True


def test_ac12_models_list_empty_before_any_job(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    assert as_list(r.json(), "models") == []


def test_jobs_and_models_filter_by_workspace(client, settings, registry):
    """2026-10-28: the Jobs and Models pages listed every workspace's jobs and models."""
    a = build_snapshot(client)
    b = build_snapshot(client)
    job_a = queued_job(client, "fake_cls", a["snapshot"]["id"])
    job_b = queued_job(client, "fake_cls", b["snapshot"]["id"])
    assert run_once(settings, "w-1", {"cpu"}, registry=registry) is True
    assert run_once(settings, "w-1", {"cpu"}, registry=registry) is True

    for ws, job_id in ((a["workspace"], job_a), (b["workspace"], job_b)):
        jobs = client.get(f"/v1/jobs?workspace_id={ws['id']}").json()["jobs"]
        assert [j["id"] for j in jobs] == [job_id]
        models = client.get(f"/v1/models?workspace_id={ws['id']}").json()["models"]
        assert [m["job_id"] for m in models] == [job_id]
        assert models[0]["dataset_name"] == jobs[0]["dataset_name"] == "ds"
    # Without the filter every workspace is listed, as before.
    assert len(client.get("/v1/jobs").json()["jobs"]) == 2
    assert len(client.get("/v1/models").json()["models"]) == 2
    empty = create_workspace(client)
    assert client.get(f"/v1/jobs?workspace_id={empty['id']}").json()["jobs"] == []
    assert client.get(f"/v1/models?workspace_id={empty['id']}").json()["models"] == []
