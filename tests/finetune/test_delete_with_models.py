"""Deleting a workspace or dataset together with the jobs and models it trained.

Policy (user decision 2026-10-28): after a clear confirmation, deleting a workspace or
dataset may remove everything it produced: jobs, their events, models, model files and
deployment records, including models in use (Studio keeps its own installed copy). The
API only does this when asked (``include_models=true``) and never while a job of the
data is queued or running, because a worker is using it. ``GET .../deletion`` lists
what would go, so the confirmation can name it.
"""

from __future__ import annotations

import pytest

from sparrow_finetune.storage import LocalObjectStore
from sparrow_finetune.workers import run_once

from .helpers import build_snapshot, create_dataset, models_for_job, queued_job
from .service_helpers import audit_rows, db, make_service_client


@pytest.fixture
def svc(settings, registry):
    with make_service_client(settings, registry) as client:
        yield client


def _trained(client, settings, registry, name: str) -> dict:
    built = build_snapshot(client)
    job_id = queued_job(client, "fake_cls", built["snapshot"]["id"])
    assert run_once(settings, "w-del", {"cpu"}, registry=registry) is True
    (model,) = models_for_job(client, job_id)
    assert (
        client.patch(f"/v1/models/{model['id']}", json={"name": name}).status_code
        == 200
    )
    return {**built, "job_id": job_id, "model": model}


def _in_use(settings, model_id, consumer="studio", project="1"):
    with db(settings) as conn:
        conn.execute(
            "INSERT INTO deployments (model_id, consumer, project, activated_by)"
            " VALUES (%s, %s, %s, 'test')",
            (model_id, consumer, project),
        )


def _rows(settings, sql, params=()):
    with db(settings) as conn:
        return conn.execute(sql, params).fetchall()


def test_deletion_preview_lists_jobs_models_and_where_they_are_used(
    svc, settings, registry
):
    t = _trained(svc, settings, registry, "Amazon v1")
    _in_use(settings, t["model"]["id"])
    ws_id = t["workspace"]["id"]

    r = svc.get(f"/v1/workspaces/{ws_id}/deletion")

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["datasets"] == 1
    assert body["images"] == 4
    assert body["jobs"] == 1
    assert body["active_jobs"] == []
    (model,) = body["models"]
    assert model["id"] == t["model"]["id"]
    assert model["name"] == "Amazon v1"
    assert model["in_use"] == [{"consumer": "studio", "project": "1"}]
    ds = svc.get(f"/v1/datasets/{t['dataset']['id']}/deletion").json()
    assert [m["id"] for m in ds["models"]] == [t["model"]["id"]]
    assert svc.get("/v1/workspaces/999999/deletion").status_code == 404


def test_delete_workspace_with_models_removes_jobs_models_files_and_deployments(
    svc, settings, registry
):
    t = _trained(svc, settings, registry, "Amazon v1")
    model_id, job_id = t["model"]["id"], t["job_id"]
    _in_use(settings, model_id)
    _in_use(settings, model_id, consumer="local", project="default")
    (key,) = _rows(
        settings, "SELECT object_key FROM model_versions WHERE id = %s", (model_id,)
    )
    store = LocalObjectStore(settings.objects_dir)
    assert store.exists(key["object_key"])
    other = _trained(svc, settings, registry, "Keep me")

    r = svc.delete(
        f"/v1/workspaces/{t['workspace']['id']}", params={"include_models": "true"}
    )

    assert r.status_code == 200, r.text
    deleted = r.json()["deleted"]
    assert (
        deleted["jobs"] == 1 and deleted["models"] == 1 and deleted["deployments"] == 2
    )
    assert _rows(settings, "SELECT id FROM jobs WHERE id = %s", (job_id,)) == []
    assert (
        _rows(settings, "SELECT id FROM job_events WHERE job_id = %s", (job_id,)) == []
    )
    assert (
        _rows(settings, "SELECT id FROM model_versions WHERE id = %s", (model_id,))
        == []
    )
    assert (
        _rows(settings, "SELECT id FROM deployments WHERE model_id = %s", (model_id,))
        == []
    )
    assert not store.exists(key["object_key"])
    assert svc.get(f"/v1/models/{model_id}/download").status_code == 404
    # Another workspace's model is untouched.
    assert [m["id"] for m in models_for_job(svc, other["job_id"])] == [
        other["model"]["id"]
    ]
    (audit,) = [
        a for a in audit_rows(svc) if a["target"] == f"workspace:{t['workspace']['id']}"
    ]
    assert audit["action"] == "workspace.delete"


def test_delete_dataset_with_models(svc, settings, registry):
    t = _trained(svc, settings, registry, "Amazon v1")

    r = svc.delete(
        f"/v1/datasets/{t['dataset']['id']}", params={"include_models": "true"}
    )

    assert r.status_code == 200, r.text
    assert r.json()["deleted"]["models"] == 1
    assert (
        _rows(
            settings, "SELECT id FROM model_versions WHERE id = %s", (t["model"]["id"],)
        )
        == []
    )
    assert (
        svc.get(f"/v1/workspaces/{t['workspace']['id']}/datasets").json()["datasets"]
        == []
    )


def test_delete_with_models_refused_while_a_job_is_queued_or_running(
    svc, settings, registry
):
    t = _trained(svc, settings, registry, "Amazon v1")
    queued = queued_job(svc, "fake_cls", t["snapshot"]["id"])
    extra = create_dataset(svc, t["workspace"]["id"], name="extra")

    r = svc.delete(
        f"/v1/workspaces/{t['workspace']['id']}", params={"include_models": "true"}
    )

    assert r.status_code == 409, r.text
    assert str(queued) in r.json()["detail"]
    assert "cancel" in r.json()["detail"].lower()
    assert svc.get(f"/v1/datasets/{extra['id']}").status_code == 200
    assert _rows(
        settings, "SELECT id FROM model_versions WHERE id = %s", (t["model"]["id"],)
    )
    preview = svc.get(f"/v1/workspaces/{t['workspace']['id']}/deletion").json()
    assert preview["active_jobs"] == [queued]


def test_delete_without_include_models_still_refuses_trained_data(
    svc, settings, registry
):
    t = _trained(svc, settings, registry, "Amazon v1")

    r = svc.delete(f"/v1/workspaces/{t['workspace']['id']}")

    assert r.status_code == 409, r.text
    assert "include_models" in r.json()["detail"]
    assert _rows(
        settings, "SELECT id FROM model_versions WHERE id = %s", (t["model"]["id"],)
    )
