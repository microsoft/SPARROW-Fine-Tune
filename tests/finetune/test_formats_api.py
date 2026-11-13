"""COCO and Camera Trap DP export/import endpoints on real PostgreSQL."""

from __future__ import annotations

import csv
import io
import json
import zipfile

from .helpers import create_dataset, create_workspace, import_annotations, upload_image


def _dataset(client, task="detection", names=("a.png", "b.png", "c.png")):
    ws = create_workspace(client)
    ds = create_dataset(client, ws["id"], task=task)
    for i, n in enumerate(names):
        upload_image(client, ds["id"], n, seed=i)  # 64x48
    return ds


def _items(client, ds_id):
    r = client.get(f"/v1/datasets/{ds_id}/items")
    assert r.status_code == 200, r.text
    return {it["image"]: it for it in r.json()["items"]}


def _seed_detection(client, ds_id):
    r = import_annotations(
        client,
        ds_id,
        [
            {
                "image": "a.png",
                "annotations": [
                    {"label": "Vulpes vulpes", "bbox": [4, 4, 20, 20]},
                    {"label": "person", "bbox": [32, 8, 16, 24]},
                ],
            },
            {"image": "b.png", "annotations": []},
        ],
    )
    assert r.status_code == 200, r.text
    r = client.patch(
        f"/v1/datasets/{ds_id}/items/b.png", json={"review_status": "confirmed"}
    )
    assert r.status_code == 200


def _post(client, ds_id, fmt, body, **params):
    return client.post(
        f"/v1/datasets/{ds_id}/import/{fmt}",
        params=params,
        content=body,
        headers={"Content-Type": "application/octet-stream"},
    )


def _table(zf, name):
    return list(csv.DictReader(io.StringIO(zf.read(name).decode())))


def test_camtrapdp_detection_round_trip(client):
    ds = _dataset(client)
    _seed_detection(client, ds["id"])
    exp = client.get(f"/v1/datasets/{ds['id']}/export/camtrapdp")
    assert exp.status_code == 200 and exp.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(exp.content)) as zf:
        assert sorted(zf.namelist()) == [
            "datapackage.json",
            "deployments.csv",
            "media.csv",
            "observations.csv",
        ]
        package = json.loads(zf.read("datapackage.json"))
        obs = _table(zf, "observations.csv")
        media = _table(zf, "media.csv")
    assert package["profile"].endswith("/1.0.2/camtrap-dp-profile.json")
    assert package["taxonomic"] == [{"scientificName": "Vulpes vulpes"}]
    assert [m["fileName"] for m in media] == ["a.png", "b.png"]
    by_type = {(o["observationType"], o["scientificName"]): o for o in obs}
    fox = by_type[("animal", "Vulpes vulpes")]
    assert (fox["bboxX"], fox["bboxY"], fox["bboxWidth"], fox["bboxHeight"]) == (
        "0.0625",
        "0.083333",
        "0.3125",
        "0.416667",
    )
    assert ("human", "") in by_type
    blank = by_type[("blank", "")]
    assert (
        blank["mediaID"] == media[1]["mediaID"]
        and blank["classificationMethod"] == "human"
    )

    ds2 = _dataset(client)
    # The receiving dataset says "person", so Camtrap DP's "human" maps back to it.
    import_annotations(
        client,
        ds2["id"],
        [
            {
                "image": "c.png",
                "annotations": [{"label": "person", "bbox": [1, 1, 5, 5]}],
            }
        ],
    )
    r = _post(client, ds2["id"], "camtrapdp", exp.content)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["matched"] == 2 and body["boxes"] == 2, body
    assert body["unmatched_files"] == [] and body["new_labels"] == ["Vulpes vulpes"]
    items = _items(client, ds2["id"])
    got = {a["label"]: a["bbox"] for a in items["a.png"]["annotations"]}
    assert set(got) == {"Vulpes vulpes", "person"}
    assert [round(v) for v in got["Vulpes vulpes"]] == [4, 4, 20, 20]
    assert items["b.png"]["annotations"] == []
    assert items["a.png"]["review_status"] == "pending"


def test_camtrapdp_classification_event_and_refusals(client):
    ds = _dataset(client, task="classification", names=("x1.jpg", "x2.jpg", "x3.jpg"))
    package = {
        "media.csv": (
            "mediaID,deploymentID,timestamp,filePath,filePublic,fileMediatype\n"
            "m1,d1,2024-05-01T10:00:00Z,photos/x1.jpg,false,image/jpeg\n"
            "m2,d1,2024-05-01T10:00:05+0000,photos/x2.jpg,false,image/jpeg\n"
            "m3,d2,2024-05-01T10:00:02Z,photos/x3.jpg,false,image/jpeg\n"
            "m4,d1,2024-05-01T10:00:03Z,photos/missing.jpg,false,image/jpeg\n"
        ),
        "observations.csv": (
            "observationID,deploymentID,mediaID,eventStart,eventEnd,observationLevel,observationType,scientificName\n"
            "o1,d1,NA,2024-05-01T09:59:59Z,2024-05-01T10:00:06Z,event,animal,Sus scrofa\n"
            "o2,d2,m3,2024-05-01T10:00:02Z,2024-05-01T10:00:02Z,media,blank,\n"
            "o3,d2,m3,2024-05-01T10:00:02Z,2024-05-01T10:00:02Z,media,unknown,\n"
            "o4,d1,m9,2024-05-01T10:00:02Z,2024-05-01T10:00:02Z,media,animal,Bos taurus\n"
        ),
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, text in package.items():
            zf.writestr(f"pkg/{name}", text)
    r = _post(client, ds["id"], "camtrapdp", buf.getvalue())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["matched"] == 3 and body["boxes"] == 0
    assert body["unmatched_files"] == ["media.csv line 5"]
    reasons = " | ".join(s["reason"] for s in body["skipped"])
    assert "unknown mediaID 'm9'" in reasons and "1 unknown/unclassified" in reasons
    items = _items(client, ds["id"])
    assert items["x1.jpg"]["annotations"] == [{"label": "Sus scrofa", "bbox": None}]
    assert items["x2.jpg"]["annotations"] == [{"label": "Sus scrofa", "bbox": None}]
    assert items["x3.jpg"]["annotations"] == []

    # Labels without boxes cannot replace a detection item's boxes.
    det = _dataset(client, names=("x1.jpg",))
    import_annotations(
        client,
        det["id"],
        [{"image": "x1.jpg", "annotations": [{"label": "cat", "bbox": [1, 1, 9, 9]}]}],
    )
    r = _post(client, det["id"], "camtrapdp", buf.getvalue())
    assert r.status_code == 200 and r.json()["matched"] == 0
    assert _items(client, det["id"])["x1.jpg"]["annotations"][0]["label"] == "cat"

    assert _post(client, ds["id"], "camtrapdp", b"not a zip").status_code == 422
    only_media = io.BytesIO()
    with zipfile.ZipFile(only_media, "w") as zf:
        zf.writestr("media.csv", package["media.csv"])
    r = _post(client, ds["id"], "camtrapdp", only_media.getvalue())
    assert r.status_code == 422 and "observations.csv" in r.text


def test_coco_round_trip_overwrite_and_snapshot(client):
    ds = _dataset(client)
    _seed_detection(client, ds["id"])
    exp = client.get(f"/v1/datasets/{ds['id']}/export/coco")
    assert exp.status_code == 200 and exp.headers["content-type"] == "application/json"
    doc = exp.json()
    assert [i["file_name"] for i in doc["images"]] == ["a.png", "b.png"]
    assert doc["images"][1]["review_status"] == "confirmed"
    assert {c["name"] for c in doc["categories"]} == {"Vulpes vulpes", "person"}
    assert sorted(a["bbox"] for a in doc["annotations"]) == [
        [4, 4, 20, 20],
        [32, 8, 16, 24],
    ]

    ds2 = _dataset(client)
    import_annotations(
        client,
        ds2["id"],
        [{"image": "b.png", "annotations": [{"label": "cat", "bbox": [1, 1, 9, 9]}]}],
    )
    client.patch(
        f"/v1/datasets/{ds2['id']}/items/b.png", json={"review_status": "confirmed"}
    )
    r = _post(client, ds2["id"], "coco", exp.content)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["matched"] == 1 and "confirmed" in body["skipped"][0]["reason"]
    # The same document zipped, with overwrite: the confirmed box is replaced by the blank.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("export/instances.json", exp.content)
    r = _post(client, ds2["id"], "coco", buf.getvalue(), overwrite="true")
    assert r.status_code == 200 and r.json()["matched"] == 2, r.text
    assert _items(client, ds2["id"])["b.png"]["annotations"] == []

    snap = client.post(f"/v1/datasets/{ds['id']}/snapshots").json()
    r = client.get(f"/v1/snapshots/{snap['id']}/export/coco")
    assert r.status_code == 200 and len(r.json()["annotations"]) == 2
    r = client.get(f"/v1/snapshots/{snap['id']}/export/camtrapdp")
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        types = sorted(o["observationType"] for o in _table(zf, "observations.csv"))
    assert types == ["animal", "blank", "human"]
    assert client.get("/v1/snapshots/999999/export/coco").status_code == 404

    assert _post(client, ds["id"], "coco", b"{}").status_code == 422
    assert (
        client.get(f"/v1/datasets/{ds['id']}/export/coco?status=bogus").status_code
        == 422
    )


def test_coco_classification_labels_and_blank_category(client):
    ds = _dataset(client, task="classification", names=("a.png", "b.png", "c.png", "d.png", "e.png"))
    doc = {
        "images": [
            {"id": 1, "file_name": "imgs/a.png", "width": 64, "height": 48},
            {"id": 2, "file_name": "b.png", "width": 64, "height": 48},
            {"id": 3, "file_name": "c.png", "width": 100, "height": 48},
            {"id": 4, "file_name": "d.png", "width": 64, "height": 48},
            {"id": 5, "file_name": "e.png", "width": 64, "height": 48},
        ],
        "categories": [{"id": 0, "name": "empty"}, {"id": 5, "name": "deer"}, {"id": 6, "name": "fox"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 5},
            {"id": 2, "image_id": 1, "category_id": 5, "bbox": [1, 1, 5, 5]},
            {"id": 3, "image_id": 2, "category_id": 0},
            {"id": 4, "image_id": 3, "category_id": 5},
            {"id": 5, "image_id": 4, "category_id": 5, "bbox": [0, 0, 10, 10]},
            {"id": 6, "image_id": 4, "category_id": 5, "bbox": [20, 10, 8, 8]},
            {"id": 7, "image_id": 4, "category_id": 6},
            {"id": 8, "image_id": 5, "category_id": 6},
        ],
    }
    r = _post(client, ds["id"], "coco", json.dumps(doc).encode())
    assert r.status_code == 200, r.text
    body = r.json()
    assert (
        body["matched"] == 4
        and body["boxes"] == 3
        and "differs from the stored 64x48" in body["skipped"][0]["reason"]
    )
    items = _items(client, ds["id"])
    # A boxed label is kept as a crop the classifier trains on; the duplicate image-level tag is dropped.
    assert items["a.png"]["annotations"] == [{"label": "deer", "bbox": [1, 1, 5, 5]}]
    assert items["b.png"]["annotations"] == []
    # Each box is its own crop; an image-level label for another species stays whole-image.
    assert items["d.png"]["annotations"] == [
        {"label": "deer", "bbox": [0, 0, 10, 10]},
        {"label": "deer", "bbox": [20, 10, 8, 8]},
        {"label": "fox", "bbox": None},
    ]
    assert items["e.png"]["annotations"] == [{"label": "fox", "bbox": None}]
    exp = client.get(f"/v1/datasets/{ds['id']}/export/coco").json()
    boxes = sorted(tuple(a["bbox"]) for a in exp["annotations"] if "bbox" in a)
    assert boxes == [(0, 0, 10, 10), (1, 1, 5, 5), (20, 10, 8, 8)]
