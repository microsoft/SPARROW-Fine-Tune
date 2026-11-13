"""FT-24: VOC export/import endpoints on real PostgreSQL (``client``/``settings`` from conftest)."""

from __future__ import annotations

import io
import zipfile

from sparrow_finetune import voc

from .helpers import create_dataset, create_workspace, import_annotations, upload_image


def _zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _det_dataset(client, names=("a.png", "b.png", "c.png")):
    ws = create_workspace(client)
    ds = create_dataset(client, ws["id"], task="detection")
    for i, n in enumerate(names):
        upload_image(client, ds["id"], n, seed=i)  # 64x48
    return ds


def _items(client, ds_id):
    r = client.get(f"/v1/datasets/{ds_id}/items")
    assert r.status_code == 200, r.text
    return {it["image"]: it for it in r.json()["items"]}


def _audit(settings, action):
    from sparrow_finetune.db import connect

    with connect(settings.database_url) as conn:
        return [
            r["detail"]
            for r in conn.execute(
                "SELECT detail FROM audit_events WHERE action = %s ORDER BY id",
                (action,),
            ).fetchall()
        ]


def test_export_round_trips_through_import(client, settings):
    ds = _det_dataset(client)
    r = import_annotations(
        client,
        ds["id"],
        [
            {
                "image": "a.png",
                "annotations": [
                    {"label": "cat", "bbox": [4, 4, 20, 20]},
                    {"label": "dog", "bbox": [30, 10, 30, 30]},
                ],
            },
            {"image": "b.png", "annotations": []},
        ],
    )
    assert r.status_code == 200, r.text
    assert (
        client.patch(
            f"/v1/datasets/{ds['id']}/items/b.png", json={"review_status": "confirmed"}
        ).status_code
        == 200
    )

    exp = client.get(f"/v1/datasets/{ds['id']}/export/voc")
    assert exp.status_code == 200
    assert exp.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(exp.content)) as zf:
        assert sorted(zf.namelist()) == ["annotations/a.xml", "annotations/b.xml"]
        b = voc.parse_voc(zf.read("annotations/b.xml"))
        assert b.objects == [] and b'verified="yes"' in zf.read("annotations/b.xml")

    # Import the exported boxes into a fresh dataset with the same images.
    ds2 = _det_dataset(client)
    r = client.post(
        f"/v1/datasets/{ds2['id']}/import/voc",
        content=exp.content,
        headers={"Content-Type": "application/zip"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["matched"] == 2 and body["boxes"] == 2
    assert body["unmatched_files"] == [] and body["skipped"] == []
    assert body["new_labels"] == ["cat", "dog"]
    items = _items(client, ds2["id"])
    assert items["a.png"]["review_status"] == "pending"
    assert [(a["label"], a["bbox"]) for a in items["a.png"]["annotations"]] == [
        ("cat", [4, 4, 20, 20]),
        ("dog", [30, 10, 30, 30]),
    ]
    assert (
        items["b.png"]["annotations"] == []
        and items["b.png"]["review_status"] == "pending"
    )
    assert "c.png" not in items
    assert _audit(settings, "dataset.import_voc")[-1]["matched"] == 2


def test_import_matching_overwrite_and_reporting(client, settings):
    ds = _det_dataset(client, ("a.png", "b.png", "c.png", "d.png"))
    import_annotations(
        client,
        ds["id"],
        [
            {"image": "b.png", "annotations": [{"label": "cat", "bbox": [1, 1, 5, 5]}]},
            {"image": "d.png", "annotations": [{"label": "cat", "bbox": [1, 1, 5, 5]}]},
        ],
    )
    client.patch(
        f"/v1/datasets/{ds['id']}/items/b.png", json={"review_status": "confirmed"}
    )
    client.patch(
        f"/v1/datasets/{ds['id']}/items/d.png", json={"review_status": "excluded"}
    )

    def doc(name, w=64, h=48, objs=(("bird", 2, 2, 10, 10),)):
        return voc.voc_xml(
            name,
            w,
            h,
            [
                {"label": lab, "bbox": [x0, y0, x1 - x0, y1 - y0]}
                for lab, x0, y0, x1, y1 in objs
            ],
        )

    data = _zip(
        {
            "x/a.xml": doc(
                "a.jpg", objs=(("bird", -5, 2, 80, 10),)
            ),  # stem match; box clamped
            "b.xml": doc("b.png"),  # confirmed: protected
            "c.xml": doc("c.png", w=100, h=100),  # size mismatch
            "d.xml": doc("d.png"),  # excluded
            "zzz.xml": doc("zzz.png"),  # no such image
            "dup/a.xml": doc("a.png"),  # second file for a.png
        }
    )
    r = client.post(f"/v1/datasets/{ds['id']}/import/voc", content=data)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["matched"] == 1 and body["boxes"] == 1
    assert body["unmatched_files"] == ["zzz.xml"]
    reasons = {s["file"]: s["reason"] for s in body["skipped"]}
    assert "overwrite=true" in reasons["b.xml"]
    assert "differs" in reasons["c.xml"]
    assert "excluded" in reasons["d.xml"]
    assert "already imported" in reasons["dup/a.xml"]
    items = _items(client, ds["id"])
    assert items["a.png"]["annotations"] == [{"label": "bird", "bbox": [0, 2, 64, 8]}]
    assert items["b.png"]["review_status"] == "confirmed"
    assert items["b.png"]["annotations"][0]["label"] == "cat"

    r = client.post(
        f"/v1/datasets/{ds['id']}/import/voc?overwrite=true",
        content=_zip({"b.xml": doc("b.png")}),
    )
    assert r.json()["matched"] == 1
    items = _items(client, ds["id"])
    assert items["b.png"]["review_status"] == "pending"
    assert [a["label"] for a in items["b.png"]["annotations"]] == ["bird"]
    assert _audit(settings, "dataset.import_voc")[-1]["overwrite"] is True


def test_import_and_export_refusals(client):
    ws = create_workspace(client)
    cls = create_dataset(client, ws["id"], task="classification")
    assert client.get(f"/v1/datasets/{cls['id']}/export/voc").status_code == 422
    assert (
        client.post(
            f"/v1/datasets/{cls['id']}/import/voc", content=_zip({})
        ).status_code
        == 422
    )
    ds = _det_dataset(client, ("a.png",))
    assert (
        client.get(f"/v1/datasets/{ds['id']}/export/voc?status=bogus").status_code
        == 422
    )
    assert client.get("/v1/datasets/999999/export/voc").status_code == 404
    r = client.post(f"/v1/datasets/{ds['id']}/import/voc", content=b"not a zip")
    assert r.status_code == 422 and "zip" in r.json()["detail"]
    r = client.post(
        f"/v1/datasets/{ds['id']}/import/voc",
        content=_zip(
            {"a.xml": b'<!DOCTYPE a [<!ENTITY x "y">]><annotation>&x;</annotation>'}
        ),
    )
    assert r.status_code == 200 and r.json()["matched"] == 0
    assert "DOCTYPE" in r.json()["skipped"][0]["reason"]


def test_snapshot_export(client):
    ds = _det_dataset(client, ("a.png",))
    import_annotations(
        client,
        ds["id"],
        [{"image": "a.png", "annotations": [{"label": "cat", "bbox": [4, 4, 20, 20]}]}],
    )
    snap = client.post(f"/v1/datasets/{ds['id']}/snapshots").json()
    # Later edits don't change the frozen export.
    client.patch(f"/v1/datasets/{ds['id']}/items/a.png", json={"annotations": []})
    r = client.get(f"/v1/snapshots/{snap['id']}/export/voc")
    assert r.status_code == 200
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        (f,) = [voc.parse_voc(zf.read(n)) for n in zf.namelist()]
    assert [o.label for o in f.objects] == ["cat"]
    assert client.get("/v1/snapshots/999999/export/voc").status_code == 404
