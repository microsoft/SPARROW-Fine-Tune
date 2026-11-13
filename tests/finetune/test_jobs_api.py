"""AC10 (job submission validation) and AC14 (cancel while still queued)."""

from __future__ import annotations

import pytest

from sparrow_finetune.workers import run_once

from .fakes import BAD_EPOCHS_MESSAGE
from .helpers import build_snapshot, create_job, get_job, queued_job


@pytest.fixture
def cls_snapshot(client) -> dict:
    return build_snapshot(client, task="classification")["snapshot"]


def test_ac10_submit_job_returns_queued(client, cls_snapshot):
    r = create_job(client, "fake_cls", cls_snapshot["id"], {"epochs": 2})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["state"] == "queued"
    assert get_job(client, body["id"])["state"] == "queued"


@pytest.mark.parametrize("pipeline", ["no_such_pipeline", "dinov3_linear", "yolo_detector"])
def test_ac10_unknown_or_planned_pipeline_422(client, cls_snapshot, pipeline):
    assert create_job(client, pipeline, cls_snapshot["id"]).status_code == 422


def test_ac10_pipeline_task_mismatch_422(client):
    det_snapshot = build_snapshot(client, task="detection", labels=("deer", "fox"))["snapshot"]
    assert create_job(client, "fake_cls", det_snapshot["id"]).status_code == 422


def test_ac10_validate_params_error_422_with_message(client, cls_snapshot):
    r = create_job(client, "fake_cls", cls_snapshot["id"], {"epochs": -1})
    assert r.status_code == 422
    assert BAD_EPOCHS_MESSAGE in r.text


def test_ac14_cancel_queued_job_is_immediate_and_never_claimed(
    client, settings, registry, trainers, cls_snapshot
):
    job_id = queued_job(client, "fake_cls", cls_snapshot["id"])

    r = client.post(f"/v1/jobs/{job_id}/cancel")

    assert r.is_success, r.text
    assert get_job(client, job_id)["state"] == "cancelled"
    assert run_once(settings, "w-1", {"cpu", "gpu"}, registry=registry) is False
    assert trainers.basic.calls == []
    assert get_job(client, job_id)["state"] == "cancelled"
