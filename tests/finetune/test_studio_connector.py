"""Phase 2A acceptance tests A11 (Studio import) and A12 (Studio push / activate studio).

Strategy: Studio is an external HTTP dependency, so it is replaced by a recording
fake (``studio_fake.FakeStudio``) injected through ``create_app(...,
studio_transport=...)``. The tests pin the contract both ways: what the service
sends (bearer token, ``reviewed`` filter, pagination cursor, multipart manifest +
tar bundle whose artifact equals the model artifact byte for byte) and what it does
with Studio's answers (detection boxes -> annotations, classification boxes -> one
crop per box, reviewed -> confirmed, zero-box images skipped, idempotent re-import,
Studio failure -> 502 with nothing recorded, unconfigured -> 409).

Spec: docs/review/implement-phase2-service/spec.md, section 2A "Studio connector".
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile

import pytest

from sparrow_finetune.workers import run_once

from .helpers import build_snapshot, create_dataset, create_workspace, models_for_job, queued_job
from .service_helpers import (
    STUDIO_TOKEN,
    STUDIO_URL,
    audit_rows,
    make_service_client,
    studio_settings,
)
from .studio_fake import API, FakeStudio, manifest_from, multipart_parts


@pytest.fixture
def studio():
    return FakeStudio(project="p1", page_size=2)


@pytest.fixture
def svc(settings, registry, studio):
    with make_service_client(studio_settings(settings), registry, studio.transport) as client:
        yield client


@pytest.fixture
def plain(settings, registry):
    """Studio not configured (Phase 1 settings)."""
    with make_service_client(settings, registry) as client:
        yield client


def new_dataset(client, task) -> int:
    ws = create_workspace(client)
    return create_dataset(client, ws["id"], task=task)["id"]


def studio_import(client, dataset_id, **extra):
    return client.post(
        "/v1/connectors/studio/import", json={"project": "p1", "dataset_id": dataset_id, **extra}
    )


def items(client, dataset_id) -> list:
    r = client.get(f"/v1/datasets/{dataset_id}/items")
    assert r.status_code == 200, r.text
    return r.json()["items"]


def trained_model(client, settings, registry) -> dict:
    built = build_snapshot(client)
    job_id = queued_job(client, "fake_cls", built["snapshot"]["id"])
    assert run_once(settings, "w-studio", {"cpu"}, registry=registry) is True
    (model,) = models_for_job(client, job_id)
    return model


# --------------------------------------------------------------------------- status


def test_a11_connector_status_configured_hides_token(svc):
    r = svc.get("/v1/connectors/studio")
    assert r.status_code == 200
    assert r.json() == {"configured": True, "url": STUDIO_URL}
    assert STUDIO_TOKEN not in r.text


def test_a11_connector_status_unconfigured(plain):
    r = plain.get("/v1/connectors/studio")
    assert r.status_code == 200
    assert r.json() == {"configured": False, "url": ""}


def test_a11_a12_unconfigured_endpoints_are_409(plain, settings, registry):
    ds = new_dataset(plain, "detection")
    assert studio_import(plain, ds).status_code == 409
    m = trained_model(plain, settings, registry)
    r = plain.post("/v1/connectors/studio/push", json={"model_id": m["id"], "project": "p1"})
    assert r.status_code == 409
    r = plain.post(f"/v1/models/{m['id']}/activate", json={"consumer": "studio", "project": "p1"})
    assert r.status_code == 409
    assert plain.get("/v1/deployments").json()["deployments"] == []


# --------------------------------------------------------------------------- A11 import


def test_a11_detection_import_boxes_pagination_and_bearer(svc, studio):
    ds = new_dataset(svc, "detection")
    r = studio_import(svc, ds)
    assert r.status_code in (200, 201), r.text
    assert r.json() == {"imported_images": 3, "imported_items": 3, "skipped": 1, "pages": 2}

    got = {i["image"]: i for i in items(svc, ds)}
    assert sorted(got) == ["a.jpg", "b.png", "c.png"]
    for src in studio.items[:3]:
        item = got[src["name"]]
        assert (item["width"], item["height"]) == (src["width"], src["height"])
        assert item["review_status"] == "confirmed"
        want = sorted((b["label"], [float(v) for v in b["bbox"]]) for b in src["boxes"])
        have = sorted((a["label"], [float(v) for v in a["bbox"]]) for a in item["annotations"])
        assert have == want
    content = svc.get(f"/v1/datasets/{ds}/images/a.jpg/content")
    assert content.status_code == 200
    assert content.content == studio.image_bytes(101)

    # what was sent to Studio
    assert studio.requests, "the fake Studio was never called"
    for req in studio.requests:
        assert req.headers["authorization"] == f"Bearer {STUDIO_TOKEN}"
        assert req.url.host == "studio.test"
        assert req.url.path.startswith(API + "/")
    exports = studio.export_requests()
    assert len(exports) == 2
    assert {e.url.params.get("reviewed") for e in exports} == {"1"}
    assert exports[0].url.params.get("cursor") in (None, "")
    assert exports[1].url.params.get("cursor") == "2"


def test_a11_classification_import_one_crop_per_box(svc, studio):
    ds = new_dataset(svc, "classification")
    r = studio_import(svc, ds)
    assert r.status_code in (200, 201), r.text
    assert r.json() == {"imported_images": 3, "imported_items": 4, "skipped": 1, "pages": 2}

    got = {i["image"]: i for i in items(svc, ds)}
    expected = {}
    for src in studio.items[:3]:
        stem = src["name"].rsplit(".", 1)[0]
        for box in src["boxes"]:
            expected[f"{stem}__box{box['id']}.png"] = (box["label"], box["bbox"])
    assert sorted(got) == sorted(expected)
    assert len(got) == sum(len(s["boxes"]) for s in studio.items[:3])
    for name, (label, bbox) in expected.items():
        item = got[name]
        assert [a["label"] for a in item["annotations"]] == [label]
        assert all(a.get("bbox") is None for a in item["annotations"])
        assert (item["width"], item["height"]) == (bbox[2], bbox[3])
        assert item["review_status"] == "confirmed"
    crop = svc.get(f"/v1/datasets/{ds}/images/a__box1.png/content")
    assert crop.status_code == 200
    assert crop.headers["content-type"].split(";")[0] == "image/png"


def test_a11_unreviewed_items_become_pending(svc, studio):
    ds = new_dataset(svc, "detection")
    r = studio_import(svc, ds, reviewed_only=False)
    assert r.status_code in (200, 201), r.text
    assert r.json() == {"imported_images": 4, "imported_items": 4, "skipped": 1, "pages": 3}
    status = {i["image"]: i["review_status"] for i in items(svc, ds)}
    assert status == {"a.jpg": "confirmed", "b.png": "confirmed", "c.png": "confirmed",
                      "d.png": "pending"}
    assert {e.url.params.get("reviewed") for e in studio.export_requests()} == {"0"}


def test_a11_reimport_is_idempotent(svc):
    ds = new_dataset(svc, "classification")
    assert studio_import(svc, ds).status_code in (200, 201)
    first_items = items(svc, ds)
    first_images = svc.get(f"/v1/datasets/{ds}/images").json()["images"]
    r = studio_import(svc, ds)
    assert r.status_code in (200, 201), r.text
    assert items(svc, ds) == first_items
    assert len(svc.get(f"/v1/datasets/{ds}/images").json()["images"]) == len(first_images)


def test_a11_studio_500_is_502_and_imports_nothing(svc, studio):
    ds = new_dataset(svc, "detection")
    studio.fail = True
    r = studio_import(svc, ds)
    assert r.status_code == 502, r.text
    assert items(svc, ds) == []


def test_a11_studio_bad_json_is_502(svc, studio):
    ds = new_dataset(svc, "detection")
    studio.bad_json = True
    assert studio_import(svc, ds).status_code == 502


def test_a11_import_is_audited(svc):
    ds = new_dataset(svc, "detection")
    studio_import(svc, ds)
    rows = [r for r in audit_rows(svc) if r["action"] == "studio.import"]
    assert len(rows) == 1
    assert rows[0]["principal"] == "anonymous"


# --------------------------------------------------------------------------- A12 push


def check_upload(client, studio, model, upload) -> dict:
    """Assert the multipart upload carries the documented manifest and a tar bundle
    holding the exact artifact plus ``manifest.json``; return the manifest."""
    assert upload.method == "POST"
    assert upload.url.path == f"{API}/projects/p1/models"
    assert upload.headers["authorization"] == f"Bearer {STUDIO_TOKEN}"
    parts = multipart_parts(upload)
    assert {"manifest", "bundle"} <= set(parts)
    manifest = manifest_from(parts)

    artifact = client.get(f"/v1/models/{model['id']}/artifact")
    assert artifact.status_code == 200
    assert manifest["schema"] == "sparrow-finetune/model-v1"
    assert str(manifest["model_id"]) == str(model["id"])
    assert manifest["pipeline"] == "fake_cls"
    assert manifest["task"] == "classification"
    assert manifest["format"] == "onnx"
    assert manifest["labels"] == ["cat", "dog"]
    assert manifest["metrics"] == model["metrics"]
    assert manifest["sha256"] == hashlib.sha256(artifact.content).hexdigest()
    assert manifest["source"] == "sparrow-fine-tune"
    assert manifest["created_at"]
    assert manifest["name"] == model.get("name")

    with tarfile.open(fileobj=io.BytesIO(parts["bundle"])) as tar:
        files = {m.name.split("/")[-1]: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
    assert "manifest.json" in files
    assert json.loads(files["manifest.json"])["sha256"] == manifest["sha256"]
    others = [data for name, data in files.items() if name != "manifest.json"]
    assert artifact.content in others, "bundle must hold the model artifact unchanged"
    return manifest


def test_a12_push_sends_manifest_and_bundle(svc, studio, settings, registry):
    m = trained_model(svc, settings, registry)
    r = svc.post("/v1/connectors/studio/push", json={"model_id": m["id"], "project": "p1"})
    assert r.status_code in (200, 201), r.text
    assert r.json() == {"studio_model_id": "studio-model-1"}
    assert len(studio.uploads) == 1
    check_upload(svc, studio, m, studio.uploads[0])
    assert svc.get("/v1/deployments").json()["deployments"] == [], "push records no deployment"
    rows = [row for row in audit_rows(svc) if row["action"] == "studio.push"]
    assert len(rows) == 1


def test_a12_push_carries_the_model_name(svc, studio, settings, registry):
    m = trained_model(svc, settings, registry)
    r = svc.patch(f"/v1/models/{m['id']}", json={"name": "Amazon mammals v1"})
    assert r.status_code == 200, r.text
    r = svc.post("/v1/connectors/studio/push", json={"model_id": m["id"], "project": "p1"})
    assert r.status_code in (200, 201), r.text
    named = svc.get(f"/v1/models/{m['id']}").json()
    manifest = check_upload(svc, studio, named, studio.uploads[0])
    assert manifest["name"] == "Amazon mammals v1"


def test_a12_push_studio_500_is_502(svc, studio, settings, registry):
    m = trained_model(svc, settings, registry)
    studio.fail = True
    r = svc.post("/v1/connectors/studio/push", json={"model_id": m["id"], "project": "p1"})
    assert r.status_code == 502, r.text


def test_a12_activate_studio_pushes_then_records(svc, studio, settings, registry):
    m = trained_model(svc, settings, registry)
    r = svc.post(f"/v1/models/{m['id']}/activate", json={"consumer": "studio", "project": "p1"})
    assert r.status_code == 200, r.text
    dep = r.json()
    assert (dep["consumer"], dep["project"]) == ("studio", "p1")
    assert str(dep["model_id"]) == str(m["id"])
    assert dep["external_id"] == "studio-model-1"
    assert len(studio.uploads) == 1
    check_upload(svc, studio, m, studio.uploads[0])
    (listed,) = svc.get("/v1/deployments", params={"consumer": "studio", "active": "true"}).json()["deployments"]
    assert str(listed["model_id"]) == str(m["id"])


def test_a12_activate_studio_failure_is_502_and_keeps_previous(svc, studio, settings, registry):
    m1 = trained_model(svc, settings, registry)
    m2 = trained_model(svc, settings, registry)
    act = {"consumer": "studio", "project": "p1"}
    assert svc.post(f"/v1/models/{m1['id']}/activate", json=act).status_code == 200
    studio.fail = True
    r = svc.post(f"/v1/models/{m2['id']}/activate", json=act)
    assert r.status_code == 502, r.text
    deps = svc.get("/v1/deployments").json()["deployments"]
    assert len(deps) == 1, "a failed push records nothing"
    assert str(deps[0]["model_id"]) == str(m1["id"])
    assert deps[0]["deactivated_at"] is None


def test_a12_activate_studio_500_records_nothing(svc, studio, settings, registry):
    m = trained_model(svc, settings, registry)
    studio.fail = True
    r = svc.post(f"/v1/models/{m['id']}/activate", json={"consumer": "studio", "project": "p1"})
    assert r.status_code == 502
    assert svc.get("/v1/deployments").json()["deployments"] == []
    assert active_deployments_of(svc, m["id"]) == []


def active_deployments_of(client, model_id) -> list:
    rows = client.get("/v1/models").json()["models"]
    return next(row for row in rows if str(row["id"]) == str(model_id))["active_deployments"]
