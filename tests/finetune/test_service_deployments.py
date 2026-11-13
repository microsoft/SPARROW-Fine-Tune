"""Phase 2A acceptance test A10: model activation and deployments (local consumer).

Strategy: the invariant that matters is "at most one active deployment per
(consumer, project)" -- consumers load whatever is active, so two active rows would
make the served model ambiguous. It is checked through the API (switching
deactivates the previous row, other projects are independent) and at the schema
level (the partial unique index refuses a second active row even when written
directly). Models come from real ``run_once`` runs of the ``fake_cls`` trainer.

Spec: docs/review/implement-phase2-service/spec.md, section 2A "Models and deployments".
"""

from __future__ import annotations

import pytest

from sparrow_finetune.workers import run_once

from .helpers import build_snapshot, models_for_job, queued_job
from .service_helpers import audit_rows, db, make_service_client


@pytest.fixture
def svc(settings, registry):
    with make_service_client(settings, registry) as client:
        yield client


def trained_model(client, settings, registry) -> dict:
    built = build_snapshot(client)
    job_id = queued_job(client, "fake_cls", built["snapshot"]["id"])
    assert run_once(settings, "w-deploy", {"cpu"}, registry=registry) is True
    (model,) = models_for_job(client, job_id)
    return model


def activate(client, model_id, consumer="local", project="p1"):
    return client.post(
        f"/v1/models/{model_id}/activate", json={"consumer": consumer, "project": project}
    )


def deployments(client, **params) -> list:
    r = client.get("/v1/deployments", params=params)
    assert r.status_code == 200, r.text
    return r.json()["deployments"]


def active_of(client, model_id) -> list:
    rows = client.get("/v1/models").json()["models"]
    model = next(m for m in rows if str(m["id"]) == str(model_id))
    return sorted((d["consumer"], d["project"]) for d in model["active_deployments"])


def test_a10_activate_local_returns_deployment(svc, settings, registry):
    m = trained_model(svc, settings, registry)
    r = activate(svc, m["id"])
    assert r.status_code == 200, r.text
    dep = r.json()
    assert str(dep["model_id"]) == str(m["id"])
    assert dep["consumer"] == "local"
    assert dep["project"] == "p1"
    assert dep["deactivated_at"] is None
    assert dep["activated_at"]
    assert dep["activated_by"] == "anonymous"
    (listed,) = deployments(svc)
    assert str(listed["model_id"]) == str(m["id"])
    assert listed["pipeline"] == "fake_cls"


def test_a10_switching_deactivates_previous(svc, settings, registry):
    m1 = trained_model(svc, settings, registry)
    m2 = trained_model(svc, settings, registry)
    assert activate(svc, m1["id"]).status_code == 200
    assert activate(svc, m2["id"]).status_code == 200

    active = deployments(svc, consumer="local", project="p1", active="true")
    assert [str(d["model_id"]) for d in active] == [str(m2["id"])]
    everything = deployments(svc, consumer="local", project="p1")
    assert len(everything) == 2
    old = next(d for d in everything if str(d["model_id"]) == str(m1["id"]))
    assert old["deactivated_at"] is not None

    assert active_of(svc, m2["id"]) == [("local", "p1")]
    assert active_of(svc, m1["id"]) == []


def test_a10_projects_are_independent(svc, settings, registry):
    m1 = trained_model(svc, settings, registry)
    m2 = trained_model(svc, settings, registry)
    assert activate(svc, m1["id"], project="p1").status_code == 200
    assert activate(svc, m2["id"], project="p2").status_code == 200
    assert activate(svc, m1["id"], project="p3").status_code == 200
    assert len(deployments(svc, active="true")) == 3
    assert active_of(svc, m1["id"]) == [("local", "p1"), ("local", "p3")]
    assert active_of(svc, m2["id"]) == [("local", "p2")]
    assert [d["project"] for d in deployments(svc, project="p2")] == ["p2"]


def test_a10_one_active_row_enforced_by_schema(svc, settings, registry):
    import psycopg

    m = trained_model(svc, settings, registry)
    assert activate(svc, m["id"]).status_code == 200
    with db(settings) as conn, pytest.raises(psycopg.errors.UniqueViolation):
        conn.execute(
            "INSERT INTO deployments (model_id, consumer, project, activated_at, activated_by)"
            " VALUES (%s, 'local', 'p1', now(), 'sql')",
            (m["id"],),
        )


def test_a10_activation_validation(svc, settings, registry):
    m = trained_model(svc, settings, registry)
    assert activate(svc, m["id"], consumer="cloud").status_code == 422
    assert activate(svc, m["id"], project="").status_code == 422
    assert deployments(svc) == []


def test_a10_activation_is_audited(svc, settings, registry):
    m = trained_model(svc, settings, registry)
    activate(svc, m["id"])
    rows = [r for r in audit_rows(svc) if r["action"] == "model.activate"]
    assert len(rows) == 1
    assert rows[0]["principal"] == "anonymous"
    assert str(m["id"]) in str(rows[0]["target"]) + str(rows[0]["detail"])


def test_a10_model_detail_lists_active_deployments(svc, settings, registry):
    """GET /models/{id} carries the same active_deployments field as GET /models."""
    m1 = trained_model(svc, settings, registry)
    m2 = trained_model(svc, settings, registry)
    assert activate(svc, m1["id"], project="p1").status_code == 200
    assert activate(svc, m1["id"], project="p2").status_code == 200
    assert activate(svc, m2["id"], project="p1").status_code == 200

    detail = svc.get(f"/v1/models/{m1['id']}").json()
    assert sorted(
        (d["consumer"], d["project"]) for d in detail["active_deployments"]
    ) == [("local", "p2")]
    assert active_of(svc, m1["id"]) == [("local", "p2")]
    detail2 = svc.get(f"/v1/models/{m2['id']}").json()
    assert detail2["active_deployments"] == [
        {"consumer": "local", "project": "p1", "external_id": None}
    ]
