"""Phase 2A acceptance test A7: review state on dataset items.

Strategy: review status decides what trains, so the critical behaviour is that an
``excluded`` item never reaches a snapshot (items *and* label counts), and that a
re-import of annotations does not silently reset a reviewer's decision. The list,
filter, patch and image-content routes are checked for the exact shapes and codes
the spec states. Open API (no tokens, no users) throughout.

Spec: docs/review/implement-phase2-service/spec.md, section 2A "Review".
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from .helpers import (
    IMAGE_SIZE,
    annotation_item,
    build_snapshot,
    create_snapshot,
    import_annotations,
    png_bytes,
)
from .service_helpers import db, make_service_client


@pytest.fixture
def svc(settings, registry):
    with make_service_client(settings, registry) as client:
        yield client


@pytest.fixture
def built(svc):
    # img0 cat, img1 dog, img2 cat, img3 dog
    return build_snapshot(svc)


def items(client, dataset_id, **params) -> list:
    r = client.get(f"/v1/datasets/{dataset_id}/items", params=params)
    assert r.status_code == 200, r.text
    return r.json()["items"]


def patch(client, dataset_id, image, body):
    return client.patch(f"/v1/datasets/{dataset_id}/items/{image}", json=body)


def test_a7_list_items_shape_order_default_pending(svc, built):
    got = items(svc, built["dataset"]["id"])
    assert [i["image"] for i in got] == ["img0.png", "img1.png", "img2.png", "img3.png"]
    first = got[0]
    assert set(first) >= {"image", "width", "height", "annotations", "group", "review_status"}
    assert (first["width"], first["height"]) == IMAGE_SIZE
    assert [a["label"] for a in first["annotations"]] == ["cat"]
    assert first["group"] is None
    assert {i["review_status"] for i in got} == {"pending"}


def test_a7_patch_status_and_filter(svc, built):
    ds = built["dataset"]["id"]
    r = patch(svc, ds, "img1.png", {"review_status": "excluded"})
    assert r.status_code == 200, r.text
    assert r.json()["image"] == "img1.png"
    assert r.json()["review_status"] == "excluded"
    assert patch(svc, ds, "img0.png", {"review_status": "confirmed"}).status_code == 200
    assert [i["image"] for i in items(svc, ds, status="excluded")] == ["img1.png"]
    assert [i["image"] for i in items(svc, ds, status="confirmed")] == ["img0.png"]
    assert [i["image"] for i in items(svc, ds, status="pending")] == ["img2.png", "img3.png"]


def test_a7_review_counts_on_dataset(svc, built):
    ds = built["dataset"]["id"]
    patch(svc, ds, "img1.png", {"review_status": "excluded"})
    patch(svc, ds, "img0.png", {"review_status": "confirmed"})
    r = svc.get(f"/v1/datasets/{ds}")
    assert r.status_code == 200
    assert r.json()["review_counts"] == {"pending": 2, "confirmed": 1, "excluded": 1}


def test_a7_patch_annotations_replaces_and_validates(svc, built):
    ds = built["dataset"]["id"]
    r = patch(svc, ds, "img2.png", {"annotations": [{"label": "bird"}]})
    assert r.status_code == 200, r.text
    assert [a["label"] for a in r.json()["annotations"]] == ["bird"]
    assert r.json()["review_status"] == "pending"
    stored = next(i for i in items(svc, ds) if i["image"] == "img2.png")
    assert [a["label"] for a in stored["annotations"]] == ["bird"]
    # validated like import: empty label and out-of-image bbox are refused, unchanged
    assert patch(svc, ds, "img2.png", {"annotations": [{"label": "  "}]}).status_code == 422
    bad_box = {"annotations": [{"label": "cat", "bbox": [60, 40, 20, 20]}]}
    assert patch(svc, ds, "img2.png", bad_box).status_code == 422
    stored = next(i for i in items(svc, ds) if i["image"] == "img2.png")
    assert [a["label"] for a in stored["annotations"]] == ["bird"]


def test_a7_patch_errors(svc, built):
    ds = built["dataset"]["id"]
    assert patch(svc, ds, "img0.png", {}).status_code == 422
    assert patch(svc, ds, "img0.png", {"review_status": "approved"}).status_code == 422
    assert patch(svc, ds, "nope.png", {"review_status": "confirmed"}).status_code == 404
    assert patch(svc, 999999, "img0.png", {"review_status": "confirmed"}).status_code == 404


def test_a7_list_errors(svc, built):
    assert svc.get(f"/v1/datasets/{built['dataset']['id']}/items", params={"status": "bogus"}).status_code == 422
    assert svc.get("/v1/datasets/999999/items").status_code == 404


def test_a7_excluded_items_absent_from_snapshot_and_label_counts(svc, built, settings):
    ds = built["dataset"]["id"]
    patch(svc, ds, "img1.png", {"review_status": "excluded"})
    r = create_snapshot(svc, ds)
    assert r.status_code == 201, r.text
    snap = r.json()
    assert snap["item_count"] == 3
    assert snap["label_counts"] == {"cat": 2, "dog": 1}
    with db(settings) as conn:
        names = [
            row["image_name"]
            for row in conn.execute(
                "SELECT image_name FROM snapshot_items WHERE snapshot_id = %s ORDER BY position",
                (snap["id"],),
            )
        ]
    assert "img1.png" not in names and len(names) == 3


def test_a7_all_excluded_snapshot_is_422(svc, built):
    ds = built["dataset"]["id"]
    for name in built["images"]:
        assert patch(svc, ds, name, {"review_status": "excluded"}).status_code == 200
    assert create_snapshot(svc, ds).status_code == 422


def test_a7_reimport_keeps_review_status(svc, built):
    ds = built["dataset"]["id"]
    patch(svc, ds, "img1.png", {"review_status": "excluded"})
    patch(svc, ds, "img0.png", {"review_status": "confirmed"})
    r = import_annotations(
        svc, ds, [annotation_item("img1.png", "fox"), annotation_item("img0.png", "owl")]
    )
    assert r.status_code == 200, r.text
    by_name = {i["image"]: i for i in items(svc, ds)}
    assert by_name["img1.png"]["review_status"] == "excluded"
    assert by_name["img0.png"]["review_status"] == "confirmed"
    assert [a["label"] for a in by_name["img1.png"]["annotations"]] == ["fox"]


def test_a7_image_content_bytes_and_type(svc, built):
    ds = built["dataset"]["id"]
    r = svc.get(f"/v1/datasets/{ds}/images/img0.png/content")
    assert r.status_code == 200
    assert r.content == png_bytes(0)
    assert r.headers["content-type"].split(";")[0] == "image/png"

    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (10, 20, 30)).save(buf, format="JPEG")
    up = svc.put(f"/v1/datasets/{ds}/images/photo.jpg", content=buf.getvalue())
    assert up.is_success, up.text
    r = svc.get(f"/v1/datasets/{ds}/images/photo.jpg/content")
    assert r.status_code == 200
    assert r.content == buf.getvalue()
    assert r.headers["content-type"].split(";")[0] == "image/jpeg"

    assert svc.get(f"/v1/datasets/{ds}/images/missing.png/content").status_code == 404
