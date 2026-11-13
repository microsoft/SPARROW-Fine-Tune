"""Regression tests for the Phase 2 review findings (I1, I2, M1, M3)."""

from fastapi.testclient import TestClient

from sparrow_finetune.api import create_app
from tests.finetune.helpers import (
    build_snapshot,
    create_dataset,
    create_workspace,
    queued_job,
)
from tests.finetune.service_helpers import make_service_client, studio_settings
from tests.finetune.studio_fake import FakeStudio, studio_item


def _dataset(client, task):
    ws = create_workspace(client)
    return create_dataset(client, ws["id"], task=task)["id"]


def _import(client, ds):
    return client.post(
        "/v1/connectors/studio/import", json={"project": "p1", "dataset_id": ds}
    )


def _items(client, ds):
    return {
        it["image"]: it for it in client.get(f"/v1/datasets/{ds}/items").json()["items"]
    }


# I1: two Studio images with the same original filename must not overwrite each other.
def test_duplicate_studio_names_import_as_distinct_images(settings, registry):
    a = studio_item(201, "IMG_0001.JPG", [(11, "cat", [2, 2, 10, 10])])
    b = studio_item(202, "IMG_0001.JPG", [(12, "dog", [5, 5, 10, 10])])
    studio = FakeStudio(items=[a, b], page_size=10)
    with make_service_client(
        studio_settings(settings), registry, studio.transport
    ) as c:
        ds = _dataset(c, "detection")
        r = _import(c, ds)
        assert r.status_code in (200, 201), r.text
        assert r.json()["imported_images"] == 2
        items = _items(c, ds)
        assert len(items) == 2
        labels = sorted(it["annotations"][0]["label"] for it in items.values())
        assert labels == ["cat", "dog"]
        for name, item in items.items():
            content = c.get(f"/v1/datasets/{ds}/images/{name}/content").content
            want = 201 if item["annotations"][0]["label"] == "cat" else 202
            assert content == studio.image_bytes(want)
        # A re-import is stable: same names, nothing new, no error.
        r2 = _import(c, ds)
        assert r2.status_code in (200, 201), r2.text
        assert set(_items(c, ds)) == set(items)


def test_null_studio_name_gets_a_usable_name(settings, registry):
    a = studio_item(301, None, [(21, "cat", [1, 1, 5, 5])])
    b = studio_item(302, None, [(22, "dog", [1, 1, 5, 5])])
    studio = FakeStudio(items=[a, b], page_size=10)
    with make_service_client(
        studio_settings(settings), registry, studio.transport
    ) as c:
        ds = _dataset(c, "detection")
        r = _import(c, ds)
        assert r.status_code in (200, 201), r.text
        assert len(_items(c, ds)) == 2


class _BadBytesStudio(FakeStudio):
    def image_bytes(self, image_id):
        return b"definitely not an image"


# I2: an unreadable Studio image is a 502 for classification too, and nothing is stored.
def test_unreadable_studio_image_is_502_for_classification(settings, registry):
    studio = _BadBytesStudio(
        items=[studio_item(401, "x.png", [(31, "cat", [1, 1, 5, 5])])], page_size=10
    )
    app = create_app(
        studio_settings(settings), registry=registry, studio_transport=studio.transport
    )
    with TestClient(app, raise_server_exceptions=False) as c:
        for task in ("classification", "detection"):
            ds = _dataset(c, task)
            r = _import(c, ds)
            assert r.status_code == 502, (task, r.status_code, r.text)
            assert _items(c, ds) == {}


class _MixedStudio(FakeStudio):
    def image_bytes(self, image_id):
        if image_id == 502:
            return b"definitely not an image"
        return super().image_bytes(image_id)


def test_bad_image_late_in_export_stores_nothing(settings, registry):
    good = studio_item(501, "good.png", [(41, "cat", [1, 1, 5, 5])])
    bad = studio_item(502, "bad.png", [(42, "dog", [1, 1, 5, 5])])
    studio = _MixedStudio(items=[good, bad], page_size=10)
    with make_service_client(
        studio_settings(settings), registry, studio.transport
    ) as c:
        ds = _dataset(c, "classification")
        assert _import(c, ds).status_code == 502
        images = c.get(f"/v1/datasets/{ds}/images/good__box41.png/content")
        assert images.status_code == 404


# M1: a non-finite poll interval is rejected, not turned into a busy loop.
def test_stream_rejects_non_finite_poll(settings, registry):
    with make_service_client(settings, registry) as c:
        job = queued_job(c, "fake_cls", build_snapshot(c)["snapshot"]["id"])
        for poll in ("nan", "inf", "-inf"):
            r = c.get(f"/v1/jobs/{job}/stream?poll={poll}")
            assert r.status_code == 422, (poll, r.status_code)


# M3: retrying the same job twice queues one retry, not two.
def test_retry_is_idempotent(settings, registry):
    with make_service_client(settings, registry) as c:
        job = queued_job(c, "fake_cls", build_snapshot(c)["snapshot"]["id"])
        assert c.post(f"/v1/jobs/{job}/cancel").status_code == 200
        first = c.post(f"/v1/jobs/{job}/retry")
        assert first.status_code == 201
        second = c.post(f"/v1/jobs/{job}/retry")
        assert second.status_code == 409
        assert str(first.json()["id"]) in second.json()["detail"]
        queued = [j for j in c.get("/v1/jobs").json()["jobs"] if j["state"] == "queued"]
        assert len(queued) == 1
