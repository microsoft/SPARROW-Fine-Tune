"""FT-24: ``POST /v1/datasets/{id}/assist`` on real PostgreSQL.

Most cases swap ``assist.load_predictor`` for a fake predictor (fixed detections) so the review-status,
overwrite, label-mapping, cap and audit rules are exact. One case trains a real ``yolo_detector`` on the
CPU (the C11 recipe from test_yolo_end_to_end.py) and runs the stored ONNX bundle through the endpoint,
checking the suggestions against ultralytics' own ``predict`` on the same ONNX file.
"""

from __future__ import annotations

import io
import json
import zipfile

import pytest

from sparrow_finetune import assist
from sparrow_finetune.api import datasets as datasets_api

from .helpers import (
    create_dataset,
    create_workspace,
    import_annotations,
    make_client,
    models_for_job,
    upload_image,
)


class FakePredictor:
    labels = ["cat", "bird"]

    def __init__(self):
        self.calls = 0

    def predict(self, rgb, conf, stats=None):
        self.calls += 1
        dets = [
            assist.Detection("cat", [4.0, 4.0, 20.0, 20.0], 0.9),
            assist.Detection("bird", [1.0, 1.0, 5.0, 5.0], 0.8),
        ]
        if stats is not None:
            stats["top_score"] = 0.9
        return [d for d in dets if d.score > conf]


@pytest.fixture
def fake(monkeypatch):
    pred = FakePredictor()
    monkeypatch.setattr(assist, "load_predictor", lambda bundle: pred)
    return pred


def _insert_model(settings, snapshot_id: int, pipeline: str = "yolo_detector") -> int:
    from sparrow_finetune.db import connect
    from sparrow_finetune.storage import LocalObjectStore

    key = f"models/fake-{pipeline}-{snapshot_id}.zip"
    sha = LocalObjectStore(settings.objects_dir).put_bytes(key, b"fake bundle")
    with connect(settings.database_url) as conn:
        job = conn.execute(
            "INSERT INTO jobs (pipeline, snapshot_id, state) VALUES (%s, %s, 'succeeded') RETURNING id",
            (pipeline, snapshot_id),
        ).fetchone()["id"]
        return conn.execute(
            "INSERT INTO model_versions (job_id, pipeline, snapshot_id, format, object_key, sha256)"
            " VALUES (%s, %s, %s, 'onnx-bundle', %s, %s) RETURNING id",
            (job, pipeline, snapshot_id, key, sha),
        ).fetchone()["id"]


def _items(client, ds_id):
    return {
        it["image"]: it
        for it in client.get(f"/v1/datasets/{ds_id}/items").json()["items"]
    }


def _setup(client, settings):
    ws = create_workspace(client)
    ds = create_dataset(client, ws["id"], task="detection")
    for i, n in enumerate(("a.png", "b.png", "c.png", "d.png", "e.png")):
        upload_image(client, ds["id"], n, seed=i)
    import_annotations(
        client,
        ds["id"],
        [
            {"image": n, "annotations": [{"label": "cat", "bbox": [10, 10, 5, 5]}]}
            for n in ("a.png", "b.png", "d.png")
        ],
    )
    client.patch(
        f"/v1/datasets/{ds['id']}/items/a.png", json={"review_status": "confirmed"}
    )
    client.patch(
        f"/v1/datasets/{ds['id']}/items/d.png", json={"review_status": "excluded"}
    )
    snap = client.post(f"/v1/datasets/{ds['id']}/snapshots").json()
    return ds["id"], _insert_model(settings, snap["id"]), snap["id"]


def test_assist_fills_only_unlabelled_items_by_default(client, settings, fake):
    ds, model, _ = _setup(client, settings)
    r = client.post(f"/v1/datasets/{ds}/assist", json={"model_id": model})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["processed"] == 2 and body["items_written"] == 2 and body["boxes"] == 2
    assert body["unknown_labels"] == ["bird"]
    assert body["conf"] == assist.DEFAULT_CONF
    assert [w["image"] for w in body["items"]] == ["c.png", "e.png"]
    assert body["items"][0]["boxes"] == [
        {"label": "cat", "bbox": [4.0, 4.0, 20.0, 20.0], "score": 0.9}
    ]
    reasons = {s["image"]: s["reason"] for s in body["skipped"]}
    assert set(reasons) == {"a.png", "b.png", "d.png"}
    assert (
        "confirmed" in reasons["a.png"]
        and "already has boxes" in reasons["b.png"]
        and "excluded" in reasons["d.png"]
    )

    items = _items(client, ds)
    assert items["c.png"]["review_status"] == "pending"
    assert items["c.png"]["annotations"] == [{"label": "cat", "bbox": [4, 4, 20, 20]}]
    assert items["a.png"]["review_status"] == "confirmed"
    assert items["b.png"]["annotations"] == [{"label": "cat", "bbox": [10, 10, 5, 5]}]

    from sparrow_finetune.db import connect

    with connect(settings.database_url) as conn:
        (row,) = conn.execute(
            "SELECT target, detail FROM audit_events WHERE action = 'dataset.assist'"
        ).fetchall()
    assert row["target"] == f"dataset:{ds}"
    assert row["detail"] == {
        "model_id": model,
        "conf": 0.25,
        "overwrite": False,
        "processed": 2,
        "items": 2,
        "boxes": 2,
    }


def test_assist_overwrite_and_conf(client, settings, fake):
    ds, model, _ = _setup(client, settings)
    r = client.post(
        f"/v1/datasets/{ds}/assist",
        json={
            "model_id": model,
            "images": ["a.png", "b.png", "d.png"],
            "overwrite": True,
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["items_written"] == 2
    items = _items(client, ds)
    for n in ("a.png", "b.png"):
        assert items[n]["review_status"] == "pending"
        assert items[n]["annotations"] == [{"label": "cat", "bbox": [4, 4, 20, 20]}]
    assert items["d.png"]["review_status"] == "excluded"

    # Nothing above conf: the image is reported and no item is created.
    r = client.post(
        f"/v1/datasets/{ds}/assist",
        json={"model_id": model, "images": ["c.png"], "conf": 0.95},
    )
    assert r.json()["items_written"] == 0
    assert "no detections" in r.json()["skipped"][0]["reason"]
    assert r.json()["best_below_conf"] == 0.9  # FT-32: tells the user how far below conf it was
    assert "c.png" not in _items(client, ds)


def test_assist_refusals(client, settings, fake, monkeypatch):
    ds, model, snap = _setup(client, settings)
    url = f"/v1/datasets/{ds}/assist"
    assert client.post(url, json={"model_id": 999999}).status_code == 404
    assert (
        client.post("/v1/datasets/999999/assist", json={"model_id": model}).status_code
        == 404
    )
    clf = _insert_model(settings, snap, pipeline="conventional_classifier")
    r = client.post(url, json={"model_id": clf})
    assert r.status_code == 422 and "yolo_detector" in r.json()["detail"]
    r = client.post(url, json={"model_id": model, "images": ["nope.png"]})
    assert r.status_code == 422 and "nope.png" in r.json()["detail"]
    for bad in ({"conf": 1.5}, {"conf": -0.1}, {"images": []}):
        assert client.post(url, json={"model_id": model, **bad}).status_code == 422
    monkeypatch.setattr(datasets_api, "ASSIST_MAX_ITEMS", 1)
    r = client.post(url, json={"model_id": model})
    assert r.status_code == 422 and "batches" in r.json()["detail"]
    assert fake.calls == 0
    # Classification datasets are supported since FT-35 (test_ft35_classification_assist.py).


# --------------------------------------------------------------------------- real model on the CPU


@pytest.fixture
def yolo_offline(monkeypatch):
    monkeypatch.setenv("YOLO_OFFLINE", "1")


def test_assist_with_a_trained_yolo_detector_matches_ultralytics(
    settings, tmp_path, yolo_offline
):
    import numpy as np

    from sparrow_finetune.pipelines import Registry, register_builtin_pipelines
    from sparrow_finetune.workers import run_once

    from .yolo_helpers import CPU_PARAMS, DET_SIZE, det_annotation_payload, det_items

    registry = Registry()
    register_builtin_pipelines(registry, settings)
    syn = det_items(prefix="train-")
    fresh = [it for it in det_items(prefix="new-") if it.boxes][:4]
    with make_client(settings, registry) as client:
        ws = create_workspace(client)
        ds = create_dataset(client, ws["id"], task="detection")["id"]
        for it in syn + fresh:
            assert client.put(
                f"/v1/datasets/{ds}/images/{it.name}", content=it.png
            ).is_success
        assert (
            import_annotations(client, ds, det_annotation_payload(syn)).status_code
            == 200
        )
        snap = client.post(f"/v1/datasets/{ds}/snapshots").json()["id"]
        job = client.post(
            "/v1/jobs",
            json={
                "pipeline": "yolo_detector",
                "snapshot_id": snap,
                "params": dict(CPU_PARAMS),
            },
        )
        assert job.status_code == 201, job.text
        assert run_once(settings, "assist-worker", {"cpu"}, registry=registry) is True
        (model,) = models_for_job(client, job.json()["id"])

        conf = 0.001
        r = client.post(
            f"/v1/datasets/{ds}/assist",
            json={
                "model_id": model["id"],
                "images": [it.name for it in fresh],
                "conf": conf,
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["unknown_labels"] == []
        assert body["processed"] == len(fresh)
        items = _items(client, ds)

    for w in body["items"]:
        assert items[w["image"]]["review_status"] == "pending"
        assert len(items[w["image"]]["annotations"]) == len(w["boxes"])
        for b in w["boxes"]:
            x, y, bw, bh = b["bbox"]
            assert b["label"] in {"cat", "dog"} and 0 < b["score"] <= 1
            assert (
                x >= 0
                and y >= 0
                and bw > 0
                and bh > 0
                and x + bw <= DET_SIZE
                and y + bh <= DET_SIZE
            )

    # Parity: ultralytics' own predict on the stored ONNX gives the same boxes.
    from ultralytics import YOLO

    with zipfile.ZipFile(io.BytesIO(_artifact_bytes(settings, model["id"]))) as zf:
        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_bytes(zf.read("model.onnx"))
        labels = json.loads(zf.read("manifest.json"))["labels"]
    yolo = YOLO(str(onnx_path), task="detect")
    ours = {w["image"]: w["boxes"] for w in body["items"]}
    for it in fresh:
        img_path = tmp_path / it.name
        img_path.write_bytes(it.png)
        (res,) = yolo.predict(
            str(img_path), conf=conf, imgsz=DET_SIZE, device="cpu", verbose=False
        )
        theirs = sorted(
            (labels[int(c)], [round(float(v), 2) for v in xyxy], round(float(s), 4))
            for xyxy, c, s in zip(
                res.boxes.xyxy.numpy(), res.boxes.cls.numpy(), res.boxes.conf.numpy()
            )
        )
        mine = sorted(
            (
                b["label"],
                [
                    round(b["bbox"][0], 2),
                    round(b["bbox"][1], 2),
                    round(b["bbox"][0] + b["bbox"][2], 2),
                    round(b["bbox"][1] + b["bbox"][3], 2),
                ],
                round(b["score"], 4),
            )
            for b in ours.get(it.name, [])
        )
        assert len(mine) == len(theirs), (it.name, mine, theirs)
        for (la, ba, sa), (lb, bb, sb) in zip(mine, theirs):
            assert la == lb and abs(sa - sb) < 1e-3
            np.testing.assert_allclose(ba, bb, atol=0.05)


def _artifact_bytes(settings, model_id: int) -> bytes:
    from sparrow_finetune.db import connect
    from sparrow_finetune.storage import LocalObjectStore

    with connect(settings.database_url) as conn:
        key = conn.execute(
            "SELECT object_key FROM model_versions WHERE id = %s", (model_id,)
        ).fetchone()["object_key"]
    return LocalObjectStore(settings.objects_dir).path(key).read_bytes()


# --------------------------------------------------------------------------- MegaDetector (default)


class FakeMegaDetector:
    labels = ["animal", "person", "vehicle"]

    def __init__(self):
        self.calls = 0

    def predict(self, rgb, conf, stats=None):
        self.calls += 1
        return [
            assist.Detection("animal", [4.0, 4.0, 20.0, 20.0], 0.9),
            assist.Detection("person", [1.0, 1.0, 5.0, 5.0], 0.8),
        ]


def _with_megadetector(client, tmp_path, onnx: bytes = b"onnx"):
    import dataclasses

    pkg = tmp_path / "weights" / assist.MEGADETECTOR_DIR
    (pkg / "1").mkdir(parents=True)
    (pkg / "1" / "model.onnx").write_bytes(onnx)
    (pkg / "labels.txt").write_text("animal,0\nperson,1\ncar,2")
    app = client.app
    app.state.settings = dataclasses.replace(app.state.settings, weights_dir=pkg.parent)
    return pkg.parent


def test_megadetector_is_the_default_and_keeps_its_labels(client, settings, tmp_path, monkeypatch):
    weights = _with_megadetector(client, tmp_path)
    md = FakeMegaDetector()
    seen = []
    monkeypatch.setattr(assist, "load_megadetector", lambda d: seen.append(d) or md)
    ds, _, _ = _setup(client, settings)

    listed = client.get("/v1/assist/models").json()["models"]
    assert listed[0] == {
        "id": "megadetector",
        "name": assist.MEGADETECTOR_NAME,
        "labels": ["animal", "person", "vehicle"],
        "available": True,
        "default": True,
    }

    # No model_id: MegaDetector. Its labels are not in the dataset (only "cat") but are kept.
    r = client.post(f"/v1/datasets/{ds}/assist", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model_id"] == "megadetector" and seen == [weights]
    assert body["unknown_labels"] == [] and body["items_written"] == 2  # c.png, e.png
    items = _items(client, ds)
    assert sorted(a["label"] for a in items["c.png"]["annotations"]) == ["animal", "person"]

    # animal_label renames the animal boxes only.
    r = client.post(
        f"/v1/datasets/{ds}/assist",
        json={"model_id": "megadetector", "images": ["c.png"], "overwrite": True, "animal_label": " deer "},
    )
    assert r.status_code == 200, r.text
    assert sorted(a["label"] for a in _items(client, ds)["c.png"]["annotations"]) == ["deer", "person"]


def test_megadetector_refusals(client, settings, tmp_path, fake):
    ds, model, _ = _setup(client, settings)
    url = f"/v1/datasets/{ds}/assist"
    # Not installed (no weights dir): listed as unavailable, and a run is a 503 that says why.
    listed = client.get("/v1/assist/models").json()["models"]
    assert listed[0]["available"] is False and listed[1:] == [
        {"id": model, "available": True, "default": False}
    ]
    r = client.post(url, json={"model_id": "megadetector"})
    assert r.status_code == 503 and assist.MEGADETECTOR_DIR in r.json()["detail"]
    # animal_label is MegaDetector-only; a blank one is refused; unknown string ids are refused.
    assert client.post(url, json={"model_id": model, "animal_label": "deer"}).status_code == 422
    _with_megadetector(client, tmp_path)
    assert client.post(url, json={"animal_label": "  "}).status_code == 422
    assert client.post(url, json={"model_id": "yolo11n"}).status_code == 422
    assert fake.calls == 0


_MD_PACKAGE = __import__("pathlib").Path(
    __import__("os").environ.get(
        "SPARROW_FT_TEST_MEGADETECTOR",
        str(__import__("pathlib").Path.home() / "repos/SparrowOPS/models/MDV6-yolov10-c"),
    )
)


@pytest.mark.skipif(
    not (_MD_PACKAGE / "1" / "model.onnx").is_file(),
    reason="MegaDetector engine package not present (set SPARROW_FT_TEST_MEGADETECTOR)",
)
def test_real_megadetector_package_loads_and_decodes():
    import numpy as np

    pred = assist.load_megadetector(_MD_PACKAGE.parent)
    assert pred.labels == ["animal", "person", "vehicle"] and pred.end2end
    # A blank frame yields nothing above the default confidence; output layout is accepted.
    assert pred.predict(np.full((480, 640, 3), 114, np.uint8)) == []


def _mark_studio(settings, ds_id, *names):
    from sparrow_finetune.db import connect

    with connect(settings.database_url) as conn:
        conn.execute(
            "UPDATE dataset_images SET source = 'studio' WHERE dataset_id = %s AND name = ANY(%s)",
            (ds_id, list(names)),
        )


def test_image_source_summary_and_sources_filter(client, settings, fake):
    ds, model, _ = _setup(client, settings)
    images = {i["name"]: i for i in client.get(f"/v1/datasets/{ds}/images").json()["images"]}
    assert {i["source"] for i in images.values()} == {"upload"}
    # b (pending with boxes) and e (empty) came from Studio.
    _mark_studio(settings, ds, "b.png", "e.png")
    # A manual re-upload of the same bytes keeps the Studio source.
    upload_image(client, ds, "e.png", seed=4)
    assert {
        i["name"]: i["source"] for i in client.get(f"/v1/datasets/{ds}/images").json()["images"]
    }["e.png"] == "studio"

    r = client.get(f"/v1/datasets/{ds}/assist/summary")
    assert r.status_code == 200, r.text
    assert r.json() == {
        "sources": {
            "upload": {"images": 3, "empty": 1, "with_boxes": 0, "confirmed": 1, "excluded": 1},
            "studio": {"images": 2, "empty": 1, "with_boxes": 1, "confirmed": 0, "excluded": 0},
        }
    }

    r = client.post(f"/v1/datasets/{ds}/assist", json={"model_id": model, "sources": ["upload"]})
    assert r.status_code == 200, r.text
    assert [w["image"] for w in r.json()["items"]] == ["c.png"]
    assert {s["image"] for s in r.json()["skipped"]} == {"a.png", "d.png"}

    # Named images from an excluded source are skipped with a reason, not labelled.
    r = client.post(
        f"/v1/datasets/{ds}/assist",
        json={"model_id": model, "sources": ["upload"], "images": ["e.png"]},
    )
    assert r.status_code == 200, r.text
    assert r.json()["items"] == [] and "studio" in r.json()["skipped"][0]["reason"]

    r = client.post(
        f"/v1/datasets/{ds}/assist", json={"model_id": model, "sources": ["upload", "studio"]}
    )
    assert [w["image"] for w in r.json()["items"]] == ["e.png"]
    assert client.post(
        f"/v1/datasets/{ds}/assist", json={"model_id": model, "sources": ["elsewhere"]}
    ).status_code == 422
    assert client.post(
        f"/v1/datasets/{ds}/assist", json={"model_id": model, "sources": []}
    ).status_code == 422
