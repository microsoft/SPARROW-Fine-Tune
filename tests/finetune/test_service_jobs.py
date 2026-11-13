"""Phase 2A acceptance tests A8 (job event stream) and A9 (job retry).

Strategy: the SSE stream is what the UI shows during training, so the tests check
ordering, ids that equal ``seq`` (resume depends on them), resume via
``Last-Event-ID`` and ``?after=``, and that the stream *ends* with an ``end`` event
once the job is terminal -- each read runs in a thread with a timeout, so an
implementation that never closes the stream fails instead of hanging the suite.
Jobs are driven to a terminal state first with ``run_once`` and the fake trainers.
Retry is checked for every source state the spec names.

Spec: docs/review/implement-phase2-service/spec.md, section 2A "Jobs".
"""

from __future__ import annotations

import pytest

from sparrow_finetune.workers import run_once

from .helpers import build_snapshot, get_job, job_events, queued_job
from .service_helpers import audit_rows, make_service_client, read_sse


@pytest.fixture
def svc(settings, registry):
    with make_service_client(settings, registry) as client:
        yield client


def finished_job(client, settings, registry, pipeline="fake_cls", params=None) -> str:
    built = build_snapshot(client)
    job_id = queued_job(client, pipeline, built["snapshot"]["id"], params)
    assert run_once(settings, "w-test", {"cpu", "gpu"}, registry=registry) is True
    return job_id


# --------------------------------------------------------------------------- A8


def assert_end(events: list, state: str) -> None:
    """The last block is ``event: end`` / ``data: {"state": ...}`` (an ``id:`` is allowed)."""
    assert events, "empty stream"
    assert events[-1].get("event") == "end", events[-1]
    assert events[-1]["data"] == {"state": state}


def test_a8_stream_sends_all_events_in_order_then_end(svc, settings, registry):
    job_id = finished_job(svc, settings, registry)
    expected = job_events(svc, job_id)
    assert len(expected) >= 4

    out = read_sse(svc, f"/v1/jobs/{job_id}/stream?poll=0.05")
    assert out["status"] == 200
    assert out["content_type"].startswith("text/event-stream")
    events = out["events"]
    job_evs = [e for e in events if e.get("event") == "job-event"]
    assert [int(e["id"]) for e in job_evs] == [ev["seq"] for ev in expected]
    for got, want in zip(job_evs, expected):
        data = got["data"]
        assert set(data) >= {"seq", "stage", "message", "metrics", "created_at"}
        assert (data["seq"], data["stage"], data["message"], data["metrics"]) == (
            want["seq"], want["stage"], want["message"], want["metrics"]
        )
    assert_end(events, "succeeded")
    assert [e.get("event") for e in events].count("end") == 1


def test_a8_failed_job_stream_ends_with_failed_state(svc, settings, registry):
    job_id = finished_job(svc, settings, registry, pipeline="fake_fail")
    out = read_sse(svc, f"/v1/jobs/{job_id}/stream?poll=0.05")
    assert out["status"] == 200
    assert_end(out["events"], "failed")


def test_a8_last_event_id_resumes_after_that_seq(svc, settings, registry):
    job_id = finished_job(svc, settings, registry)
    seqs = [ev["seq"] for ev in job_events(svc, job_id)]
    pivot = seqs[1]
    out = read_sse(
        svc, f"/v1/jobs/{job_id}/stream?poll=0.05", headers={"Last-Event-ID": str(pivot)}
    )
    got = [int(e["id"]) for e in out["events"] if e.get("event") == "job-event"]
    assert got == [s for s in seqs if s > pivot]
    assert out["events"][-1]["event"] == "end"


def test_a8_after_query_resumes(svc, settings, registry):
    job_id = finished_job(svc, settings, registry)
    seqs = [ev["seq"] for ev in job_events(svc, job_id)]
    out = read_sse(svc, f"/v1/jobs/{job_id}/stream?poll=0.05&after={seqs[-1]}")
    assert out["status"] == 200
    assert [e for e in out["events"] if e.get("event") == "job-event"] == []
    assert_end(out["events"], "succeeded")


def test_a8_cancelled_job_stream_ends_with_cancelled_state(svc, settings, registry):
    built = build_snapshot(svc)
    job_id = queued_job(svc, "fake_cls", built["snapshot"]["id"])
    assert svc.post(f"/v1/jobs/{job_id}/cancel").status_code == 200
    out = read_sse(svc, f"/v1/jobs/{job_id}/stream?poll=0.05")
    assert out["status"] == 200
    assert_end(out["events"], "cancelled")


def test_a8_unknown_job_is_404(svc, settings, registry):
    # a known job streams (proves the route exists, so the 404 below is about the job)
    known = finished_job(svc, settings, registry)
    assert read_sse(svc, f"/v1/jobs/{known}/stream?poll=0.05")["status"] == 200
    out = read_sse(svc, "/v1/jobs/999999/stream?poll=0.05")
    assert out["status"] == 404


# --------------------------------------------------------------------------- A9


def assert_same_fields(new: dict, old: dict) -> None:
    for key in ("pipeline", "snapshot_id", "params", "requires"):
        assert new[key] == old[key], key


def test_a9_retry_failed_job_creates_new_queued_copy(svc, settings, registry):
    job_id = finished_job(svc, settings, registry, pipeline="fake_fail", params={"epochs": 2})
    old = get_job(svc, job_id)
    assert old["state"] == "failed"
    r = svc.post(f"/v1/jobs/{job_id}/retry")
    assert r.status_code == 201, r.text
    new = r.json()
    assert str(new["id"]) != str(job_id)
    assert new["state"] == "queued"
    assert_same_fields(new, old)
    assert get_job(svc, job_id)["state"] == "failed", "the original job is unchanged"
    assert_same_fields(get_job(svc, new["id"]), old)


def test_a9_retry_cancelled_job(svc):
    built = build_snapshot(svc)
    job_id = queued_job(svc, "fake_gpu", built["snapshot"]["id"], {"epochs": 3})
    assert svc.post(f"/v1/jobs/{job_id}/cancel").json()["state"] == "cancelled"
    old = get_job(svc, job_id)
    r = svc.post(f"/v1/jobs/{job_id}/retry")
    assert r.status_code == 201, r.text
    assert r.json()["state"] == "queued"
    assert_same_fields(r.json(), old)
    assert r.json()["requires"] == old["requires"]


def test_a9_retry_of_queued_or_succeeded_is_409(svc, settings, registry):
    done = finished_job(svc, settings, registry)
    assert get_job(svc, done)["state"] == "succeeded"
    built = build_snapshot(svc)
    waiting = queued_job(svc, "fake_cls", built["snapshot"]["id"])
    assert svc.post(f"/v1/jobs/{done}/retry").status_code == 409
    assert svc.post(f"/v1/jobs/{waiting}/retry").status_code == 409
    assert len(svc.get("/v1/jobs").json()["jobs"]) == 2, "a refused retry creates no job"


def test_a9_retry_unknown_job_is_404(svc):
    # a known queued job answers 409 (proves the route exists)
    built = build_snapshot(svc)
    waiting = queued_job(svc, "fake_cls", built["snapshot"]["id"])
    assert svc.post(f"/v1/jobs/{waiting}/retry").status_code == 409
    assert svc.post("/v1/jobs/999999/retry").status_code == 404


def test_a9_retry_is_audited(svc, settings, registry):
    job_id = finished_job(svc, settings, registry, pipeline="fake_fail")
    new_id = svc.post(f"/v1/jobs/{job_id}/retry").json()["id"]
    rows = [r for r in audit_rows(svc) if r["action"] == "job.retry"]
    assert len(rows) == 1
    assert rows[0]["principal"] == "anonymous"
    target = str(rows[0]["target"]) + str(rows[0]["detail"])
    assert str(job_id) in target and str(new_id) in target
