"""Phase 2A acceptance test A13: one operator journey end to end over HTTP.

Strategy: the units are covered by A1-A12; this test proves they compose the way
an operator uses them -- a password user (not an API token) logs in with a session
cookie and the CSRF header, imports reviewed boxes from (fake) Studio into a
classification dataset, freezes a snapshot, submits a job, a worker runs it with
``run_once``, the model is activated for Studio (which pushes the bundle), and the
audit log names ``alice`` for every step. Real PostgreSQL; Studio is the recording
``FakeStudio`` behind ``create_app(..., studio_transport=...)``.

Spec: docs/review/implement-phase2-service/spec.md, section 2A, criterion A13.
"""

from __future__ import annotations

from sparrow_finetune.workers import run_once

from .helpers import get_job, job_events, models_for_job
from .service_helpers import (
    CSRF,
    add_user,
    audit_rows,
    login,
    make_service_client,
    studio_settings,
)
from .studio_fake import FakeStudio, manifest_from, multipart_parts


def test_a13_login_import_snapshot_train_activate_studio_audit(settings, registry, trainers):
    studio = FakeStudio(project="p1", page_size=2)
    cfg = studio_settings(settings)
    add_user(settings, "alice")
    with make_service_client(cfg, registry, studio.transport) as client:
        # unauthenticated -> refused; then a cookie session
        assert client.get("/v1/workspaces").status_code == 401
        login(client)
        me = client.get("/v1/auth/me").json()
        assert me == {"username": "alice", "method": "session", "auth_required": True}

        r = client.post("/v1/workspaces", json={"name": "e2e"}, headers=CSRF)
        assert r.status_code == 201, r.text
        r = client.post(
            f"/v1/workspaces/{r.json()['id']}/datasets",
            json={"name": "studio-crops", "task": "classification"},
            headers=CSRF,
        )
        assert r.status_code == 201, r.text
        ds = r.json()["id"]

        # Studio import (reviewed only): 4 boxes on 3 images, one zero-box image skipped
        r = client.post(
            "/v1/connectors/studio/import",
            json={"project": "p1", "dataset_id": ds},
            headers=CSRF,
        )
        assert r.status_code in (200, 201), r.text
        assert r.json() == {"imported_images": 3, "imported_items": 4, "skipped": 1, "pages": 2}
        detail = client.get(f"/v1/datasets/{ds}").json()
        assert detail["review_counts"] == {"pending": 0, "confirmed": 4, "excluded": 0}

        # snapshot -> job
        r = client.post(f"/v1/datasets/{ds}/snapshots", headers=CSRF)
        assert r.status_code == 201, r.text
        snap = r.json()
        assert snap["item_count"] == 4
        assert snap["label_counts"] == {"cat": 2, "dog": 1, "fox": 1}
        r = client.post(
            "/v1/jobs",
            json={"pipeline": "fake_cls", "snapshot_id": snap["id"], "params": {}},
            headers=CSRF,
        )
        assert r.status_code == 201, r.text
        job_id = r.json()["id"]

        # worker
        assert run_once(settings, "e2e-worker", {"cpu"}, registry=registry) is True
        assert get_job(client, job_id)["state"] == "succeeded"
        assert job_events(client, job_id)
        assert trainers.basic.labels == ["cat", "dog", "fox"]
        (model,) = models_for_job(client, job_id)

        # activate for Studio -> pushes the bundle, records the deployment
        r = client.post(
            f"/v1/models/{model['id']}/activate",
            json={"consumer": "studio", "project": "p1"},
            headers=CSRF,
        )
        assert r.status_code == 200, r.text
        dep = r.json()
        assert dep["activated_by"] == "alice"
        assert dep["external_id"] == "studio-model-1"
        assert len(studio.uploads) == 1
        manifest = manifest_from(multipart_parts(studio.uploads[0]))
        assert str(manifest["model_id"]) == str(model["id"])
        assert manifest["labels"] == ["cat", "dog", "fox"]
        models = client.get("/v1/models").json()["models"]
        mine = next(m for m in models if str(m["id"]) == str(model["id"]))
        assert mine["active_deployments"] == [
            {"consumer": "studio", "project": "p1", "external_id": "studio-model-1"}
        ]

        # logout, log back in, read the audit trail
        assert client.post("/v1/auth/logout", headers=CSRF).status_code == 204
        assert client.get("/v1/workspaces").status_code == 401
        login(client)
        rows = list(reversed(audit_rows(client)))  # oldest first
        actions = [row["action"] for row in rows]
        for step in ("login", "studio.import", "job.submit", "model.activate", "logout"):
            assert step in actions, f"audit is missing {step!r}: {actions}"
        order = [actions.index(s) for s in ("login", "studio.import", "job.submit",
                                            "model.activate", "logout")]
        assert order == sorted(order), actions
        assert {row["principal"] for row in rows} == {"alice"}
