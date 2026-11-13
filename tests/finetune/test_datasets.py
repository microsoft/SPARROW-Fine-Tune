"""AC5-AC9: workspaces, datasets, image upload, annotation import, snapshots.

The validation tests (AC6-AC8) pin the 422 boundary for each invalid input the
spec names; the snapshot tests (AC9) pin immutability and content hashing, which
make training runs reproducible.
"""

from __future__ import annotations

import hashlib

import pytest

from sparrow_finetune.storage import LocalObjectStore
from sparrow_finetune.workers import run_once

from .helpers import (
    IMAGE_SIZE,
    annotation_item,
    as_list,
    build_snapshot,
    create_dataset,
    create_snapshot,
    create_workspace,
    field,
    import_annotations,
    png_bytes,
    queued_job,
    upload_image,
)

W, H = IMAGE_SIZE


# --------------------------------------------------------------------------- AC5


def test_ac5_create_and_list_workspace(client):
    r = client.post("/v1/workspaces", json={"name": "camera-traps"})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["name"] == "camera-traps"
    assert "id" in body

    listed = as_list(client.get("/v1/workspaces").json(), "workspaces")
    assert any(str(w["id"]) == str(body["id"]) and w["name"] == "camera-traps" for w in listed)


# --------------------------------------------------------------------------- AC6


@pytest.mark.parametrize("task", ["classification", "detection"])
def test_ac6_create_dataset_valid_task(client, task):
    ws = create_workspace(client)
    r = client.post(f"/v1/workspaces/{ws['id']}/datasets", json={"name": "d", "task": task})
    assert r.status_code == 201, r.text
    assert "id" in r.json()


@pytest.mark.parametrize("task", ["segmentation", ""])
def test_ac6_create_dataset_invalid_task_422(client, task):
    ws = create_workspace(client)
    r = client.post(f"/v1/workspaces/{ws['id']}/datasets", json={"name": "d", "task": task})
    assert r.status_code == 422


# --------------------------------------------------------------------------- AC7


def test_ac7_upload_image_returns_metadata_and_stores_bytes(client, settings):
    ds = create_dataset(client, create_workspace(client)["id"])
    data = png_bytes(7)

    r = client.put(f"/v1/datasets/{ds['id']}/images/a.png", content=data)

    assert r.is_success, r.text
    body = r.json()
    expected_sha = hashlib.sha256(data).hexdigest()
    assert (body["name"], body["sha256"], body["width"], body["height"]) == (
        "a.png", expected_sha, W, H,
    )
    stored = [
        p for p in settings.data_dir.rglob("*")
        if p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == expected_sha
    ]
    assert stored, "uploaded image bytes not found under data_dir (object store)"


def test_ac7_non_image_bytes_rejected_422(client):
    ds = create_dataset(client, create_workspace(client)["id"])
    r = client.put(f"/v1/datasets/{ds['id']}/images/b.png", content=b"this is not an image")
    assert r.status_code == 422


def test_ac7_local_object_store_roundtrip(tmp_path):
    store = LocalObjectStore(tmp_path / "objects")
    data = b"hello object store"
    assert store.put_bytes("a/b.bin", data) == hashlib.sha256(data).hexdigest()
    assert store.exists("a/b.bin")
    assert not store.exists("a/missing.bin")
    with store.open("a/b.bin") as fh:
        assert fh.read() == data

    src = tmp_path / "src.bin"
    src.write_bytes(b"file payload")
    assert store.put_file("c.bin", src) == hashlib.sha256(b"file payload").hexdigest()
    with store.open("c.bin") as fh:
        assert fh.read() == b"file payload"


# --------------------------------------------------------------------------- AC8


def _dataset_with_images(client, task="classification", n=3) -> str:
    ds = create_dataset(client, create_workspace(client)["id"], task=task)
    for i in range(n):
        upload_image(client, ds["id"], f"img{i}.png", seed=i)
    return ds["id"]


def test_ac8_import_annotations_returns_count(client):
    ds = _dataset_with_images(client)
    items = [annotation_item(f"img{i}.png", "cat", group="train") for i in range(3)]
    r = import_annotations(client, ds, items)
    assert r.status_code == 200, r.text
    assert r.json() == {"imported": 3}


def test_ac8_detection_import_with_bboxes(client):
    ds = _dataset_with_images(client, task="detection", n=2)
    items = [annotation_item(f"img{i}.png", "deer", [1, 2, 30, 20]) for i in range(2)]
    r = import_annotations(client, ds, items)
    assert r.status_code == 200, r.text
    assert r.json() == {"imported": 2}


BAD_ITEMS = {
    "unknown_image": annotation_item("nope.png", "cat"),
    "empty_label": annotation_item("img2.png", ""),
    "bbox_past_right_edge": annotation_item("img2.png", "cat", [50, 0, 20, 10]),
    "bbox_past_bottom_edge": annotation_item("img2.png", "cat", [0, 40, 10, 20]),
    "bbox_negative_origin": annotation_item("img2.png", "cat", [-5, 0, 10, 10]),
}


@pytest.mark.parametrize("case", sorted(BAD_ITEMS))
def test_ac8_invalid_item_rejected_422_naming_index(client, case):
    ds = _dataset_with_images(client)
    items = [annotation_item("img0.png", "cat"), annotation_item("img1.png", "dog"), BAD_ITEMS[case]]
    r = import_annotations(client, ds, items)
    assert r.status_code == 422
    assert "2" in r.text, f"error should name item index 2: {r.text}"


def test_ac8_detection_requires_bbox_on_every_annotation(client):
    ds = _dataset_with_images(client, task="detection")
    items = [
        annotation_item("img0.png", "deer", [1, 1, 10, 10]),
        annotation_item("img1.png", "deer", [1, 1, 10, 10]),
        annotation_item("img2.png", "deer", None),
    ]
    r = import_annotations(client, ds, items)
    assert r.status_code == 422
    assert "2" in r.text


# --------------------------------------------------------------------------- AC9


def test_ac9_snapshot_reports_counts_and_hash(client):
    built = build_snapshot(client, labels=("cat", "dog", "cat", "dog"))
    snap = built["snapshot"]
    assert snap["item_count"] == 4
    assert snap["label_counts"] == {"cat": 2, "dog": 2}
    assert snap["content_hash"]
    assert "id" in snap


def test_ac9_same_content_same_hash(client):
    built = build_snapshot(client, labels=("cat", "dog"))
    again = create_snapshot(client, built["dataset"]["id"])
    assert again.status_code == 201, again.text
    assert again.json()["content_hash"] == built["snapshot"]["content_hash"]


def test_ac9_empty_dataset_snapshot_422(client):
    ds = create_dataset(client, create_workspace(client)["id"])
    assert create_snapshot(client, ds["id"]).status_code == 422


def test_ac9_snapshot_immutable_after_later_import(client, settings, registry, trainers):
    built = build_snapshot(client, labels=("cat", "dog"))
    ds = built["dataset"]["id"]
    snap1 = built["snapshot"]

    upload_image(client, ds, "late.png", seed=9)
    r = import_annotations(client, ds, [annotation_item("late.png", "bird")])
    assert r.status_code == 200, r.text
    snap2 = create_snapshot(client, ds)
    assert snap2.status_code == 201, snap2.text

    assert snap2.json()["item_count"] == 3
    assert snap2.json()["label_counts"] == {"cat": 1, "dog": 1, "bird": 1}
    assert snap2.json()["content_hash"] != snap1["content_hash"]

    # The first snapshot, consumed by a job after the import, still has its original content.
    queued_job(client, "fake_cls", snap1["id"])
    assert run_once(settings, "w-snap", {"cpu"}, registry=registry) is True
    assert len(trainers.basic.items) == 2
    assert set(trainers.basic.labels) == {"cat", "dog"}
    seen = sorted(
        field(a, "label") for item in trainers.basic.items for a in field(item, "annotations")
    )
    assert seen == ["cat", "dog"]
