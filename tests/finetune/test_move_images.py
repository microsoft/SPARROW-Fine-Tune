"""Moving uploaded images between the datasets of one workspace.

Manual test 2026-10-27: after opening a Studio project the app selected the detection
dataset, so 80 images with species-only labels were uploaded there and their label file
was refused. The fix names the box-less images and lets the user move them into the
workspace's classification dataset, where the same labels import cleanly."""

import json

from .helpers import create_dataset, create_workspace, upload_image


def _pair(client):
    ws = create_workspace(client)
    det = create_dataset(client, ws["id"], task="detection")
    cls = create_dataset(client, ws["id"], name="species", task="classification")
    return det, cls


def _names(client, dataset_id):
    return [
        i["name"]
        for i in client.get(f"/v1/datasets/{dataset_id}/images").json()["images"]
    ]


def _move(client, src, dst, names):
    return client.post(
        f"/v1/datasets/{src}/images/move", json={"to_dataset_id": dst, "names": names}
    )


def test_box_less_labels_name_the_images_to_move(client):
    det, _ = _pair(client)
    for i, name in enumerate(["b.png", "a.png"]):
        upload_image(client, det["id"], name, seed=i)
    items = [
        {"image": n, "annotations": [{"label": "Bos"}]} for n in ("a.png", "b.png")
    ]
    r = client.post(f"/v1/datasets/{det['id']}/annotations", json={"items": items})
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert detail["reason"] == "labels_without_boxes"
    assert detail["unboxed_images"] == ["a.png", "b.png"]


def test_move_then_species_labels_import_into_classification(client):
    det, cls = _pair(client)
    for i, name in enumerate(["a.png", "b.png", "keep.png"]):
        upload_image(client, det["id"], name, seed=i)
    r = _move(client, det["id"], cls["id"], ["a.png", "b.png"])
    assert r.status_code == 200, r.text
    assert r.json() == {"moved": 2, "already_there": 0}
    assert _names(client, det["id"]) == ["keep.png"]
    assert _names(client, cls["id"]) == ["a.png", "b.png"]
    assert (
        client.get(f"/v1/datasets/{cls['id']}/images/a.png/content").status_code == 200
    )
    items = [
        {"image": n, "annotations": [{"label": "Bos"}]} for n in ("a.png", "b.png")
    ]
    r = client.post(f"/v1/datasets/{cls['id']}/annotations", json={"items": items})
    assert r.status_code == 200, r.text
    assert r.json()["imported"] == 2


def test_move_refuses_labelled_images_and_moves_nothing(client):
    det, cls = _pair(client)
    upload_image(client, det["id"], "free.png", seed=0)
    upload_image(client, det["id"], "boxed.png", seed=1)  # 64x48
    items = [
        {
            "image": "boxed.png",
            "annotations": [{"label": "Bos", "bbox": [1, 1, 10, 10]}],
        }
    ]
    assert (
        client.post(
            f"/v1/datasets/{det['id']}/annotations", json={"items": items}
        ).status_code
        == 200
    )
    r = _move(client, det["id"], cls["id"], ["free.png", "boxed.png"])
    assert r.status_code == 409, r.text
    assert "boxed.png" in r.text
    assert _names(client, det["id"]) == ["boxed.png", "free.png"]
    assert _names(client, cls["id"]) == []


def test_move_rejects_unknown_names_other_workspace_and_same_dataset(client):
    det, cls = _pair(client)
    upload_image(client, det["id"], "a.png", seed=0)
    assert _move(client, det["id"], cls["id"], ["nope.png"]).status_code == 422
    assert _move(client, det["id"], det["id"], ["a.png"]).status_code == 422
    other_ws = create_workspace(client, name="other")
    other = create_dataset(client, other_ws["id"], task="classification")
    assert _move(client, det["id"], other["id"], ["a.png"]).status_code == 422
    assert _move(client, det["id"], 999999, ["a.png"]).status_code == 404
    assert _names(client, det["id"]) == ["a.png"]


def test_move_with_name_clash(client):
    """Different bytes under the same name in the target: refused. Identical bytes:
    the target already has it, so the source copy is simply removed."""
    det, cls = _pair(client)
    upload_image(client, det["id"], "same.png", seed=0)
    upload_image(client, cls["id"], "same.png", seed=0)
    upload_image(client, det["id"], "clash.png", seed=1)
    upload_image(client, cls["id"], "clash.png", seed=2)
    r = _move(client, det["id"], cls["id"], ["same.png", "clash.png"])
    assert r.status_code == 409, r.text
    assert "clash.png" in r.text
    assert _names(client, det["id"]) == ["clash.png", "same.png"]
    r = _move(client, det["id"], cls["id"], ["same.png"])
    assert r.status_code == 200, r.text
    assert r.json() == {"moved": 0, "already_there": 1}
    assert _names(client, det["id"]) == ["clash.png"]


def test_coco_box_less_labels_on_detection_report_unboxed_images(client):
    det, _ = _pair(client)
    upload_image(client, det["id"], "a.png", seed=0)  # 64x48
    doc = {
        "images": [{"id": 1, "file_name": "a.png", "width": 64, "height": 48}],
        "categories": [{"id": 1, "name": "Bos"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1}],
    }
    r = client.post(
        f"/v1/datasets/{det['id']}/import/coco",
        content=json.dumps(doc),
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["matched"] == 0
    assert body["unboxed_images"] == ["a.png"]
