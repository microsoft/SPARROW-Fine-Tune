"""FT-24 connector: detector push hints, YOLO warm-start listing, unknown/exclude labels
on Studio import.

Label vocabulary follows the original Studio fine-tune module (Studio commit 845dd51,
``client/worker/model_finetuning/intake_contract_v1.py``): ``NEGATIVE_LABELS`` refuse
the item, the ``uncertain`` sentinel is "no label" and never trains.
"""

# ruff: noqa: F811
from __future__ import annotations

import json

import pytest
from psycopg.types.json import Jsonb

from sparrow_finetune import base_models
from sparrow_finetune.connectors import engine_package as ep
from sparrow_finetune.connectors import studio as studio_mod
from sparrow_finetune.db import connect

from .studio_fake import FakeStudio, studio_item
from .test_base_models import seed_model
from .test_engine_package import FAKE_ENGINE, _bundle_files
from .test_studio_connector import items, new_dataset, studio_import, trained_model
from .test_studio_connector import svc  # noqa: F401
from .service_helpers import make_service_client, studio_settings


@pytest.fixture
def studio():
    return FakeStudio(project="p1", page_size=2)


@pytest.fixture
def captured(monkeypatch):
    calls = []

    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        return FAKE_ENGINE

    monkeypatch.setattr(studio_mod, "build_engine_package", fake)
    return calls


def _make_detector(settings, model_id, labels):
    with connect(settings.database_url) as conn:
        conn.execute(
            "UPDATE model_versions SET pipeline = 'yolo_detector', metadata = %s "
            "WHERE id = %s",
            (Jsonb({"labels": labels}), model_id),
        )
        conn.execute(
            "UPDATE datasets SET task = 'detection', whole_image = false WHERE id = ("
            " SELECT s.dataset_id FROM model_versions m"
            " JOIN snapshots s ON s.id = m.snapshot_id WHERE m.id = %s)",
            (model_id,),
        )


def _push(client, model_id):
    r = client.post(
        "/v1/connectors/studio/push", json={"model_id": model_id, "project": "p1"}
    )
    assert r.status_code == 200, r.text


# --------------------------------------------------------------------------- push hints


@pytest.mark.parametrize(
    "labels, kind, gate",
    [
        (["animal", "person", "vehicle"], "generic", "animal"),
        (["deer", "fox"], "species", None),
    ],
)
def test_push_detector_manifest_carries_pairing_hints(
    captured, svc, studio, settings, registry, labels, kind, gate
):
    m = trained_model(svc, settings, registry)
    _make_detector(settings, m["id"], labels)
    _push(svc, m["id"])
    manifest, files = _bundle_files(studio.uploads[0])
    assert manifest["labels"] == labels
    assert manifest["detector_labels"] == labels
    assert manifest["detector_kind"] == kind
    assert manifest["detector_gate_class"] == gate
    assert manifest["engine"] == FAKE_ENGINE.meta
    assert json.loads(files["manifest.json"]) == manifest


def test_push_classifier_manifest_has_no_detector_hints_and_passes_metrics(
    captured, svc, studio, settings, registry
):
    m = trained_model(svc, settings, registry)
    _push(svc, m["id"])
    manifest, _ = _bundle_files(studio.uploads[0])
    assert not any(k.startswith("detector_") for k in manifest)
    (args, kwargs) = captured[0]
    assert kwargs["metrics"] == manifest["metrics"]
    assert isinstance(kwargs["params"], dict)


def test_push_merges_engine_extra_at_top_level(
    monkeypatch, svc, studio, settings, registry
):
    pkg = ep.EnginePackage(
        onnx=FAKE_ENGINE.onnx,
        meta={**FAKE_ENGINE.meta, "confidence_threshold": 0.6},
        extra={"calibrated_threshold": 0.6, "abstain_label": "unknown"},
    )
    monkeypatch.setattr(studio_mod, "build_engine_package", lambda *a, **k: pkg)
    m = trained_model(svc, settings, registry)
    _push(svc, m["id"])
    manifest, _ = _bundle_files(studio.uploads[0])
    assert manifest["engine"]["confidence_threshold"] == 0.6
    assert manifest["calibrated_threshold"] == 0.6
    assert manifest["abstain_label"] == "unknown"
    assert "abstain_label" not in manifest["engine"]


# --------------------------------------------------------------------------- importable


def _listed(client):
    r = client.get("/v1/connectors/studio/models", params={"project": "p1"})
    assert r.status_code == 200, r.text
    return {m["pipeline"]: m["importable"] for m in r.json()["models"]}


def test_yolo_detector_listed_importable_with_warm_start_set(monkeypatch, svc, studio):
    monkeypatch.setattr(
        base_models,
        "WARM_START_PIPELINES",
        frozenset({"conventional_classifier", "yolo_detector"}),
        raising=False,
    )
    seed_model(studio)
    seed_model(studio, pipeline="yolo_detector", task="detection")
    seed_model(studio, pipeline="dinov3_linear")
    assert _listed(svc) == {
        "conventional_classifier": True,
        "yolo_detector": True,
        "dinov3_linear": False,
    }


def test_importable_falls_back_to_single_pipeline(monkeypatch, svc, studio):
    monkeypatch.delattr(base_models, "WARM_START_PIPELINES", raising=False)
    seed_model(studio)
    seed_model(studio, pipeline="yolo_detector", task="detection")
    assert _listed(svc) == {"conventional_classifier": True, "yolo_detector": False}


# --------------------------------------------------------------------------- import labels


def _label_items():
    return [
        # keeps cat, drops the excluded box
        studio_item(
            201, "a.png", [(1, "cat", [2, 3, 20, 10]), (2, " Empty ", [30, 10, 16, 20])]
        ),
        # only excluded boxes -> image refused (original: negative_image)
        studio_item(
            202, "b.png", [(3, "blank", [0, 0, 30, 30]), (4, "EXCLUDE", [1, 1, 9, 9])]
        ),
        # an unknown box next to a known one
        studio_item(
            203, "c.png", [(5, "fox", [10, 8, 12, 12]), (6, "Unknown", [30, 8, 12, 12])]
        ),
        # only unknown / unlabeled boxes
        studio_item(
            204, "d.png", [(7, "unidentified", [5, 5, 10, 10]), (8, "", [6, 6, 10, 10])]
        ),
        studio_item(205, "e.png", [(9, "dog", [5, 5, 10, 10])]),
    ]


@pytest.fixture
def label_svc(settings, registry):
    fake = FakeStudio(project="p1", page_size=10, items=_label_items())
    with make_service_client(studio_settings(settings), registry, fake.transport) as c:
        yield c


def test_detection_import_handles_exclude_and_unknown(label_svc):
    ds = new_dataset(label_svc, "detection")
    r = studio_import(label_svc, ds)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["imported_images"] == 2
    assert body["skipped"] == 0
    assert body["label_handling"]["excluded_boxes"] == 3
    assert body["label_handling"]["unknown_boxes"] == 3
    assert body["label_handling"]["images_skipped_excluded"] == 1
    # c.png (unknown next to fox) and d.png: a dropped unknown box would teach the
    # detector that the object is background, so the image is skipped.
    assert body["label_handling"]["images_skipped_unknown"] == 2
    got = {
        i["image"]: sorted(a["label"] for a in i["annotations"])
        for i in items(label_svc, ds)
    }
    assert got == {"a.png": ["cat"], "e.png": ["dog"]}
    # Imported images are tagged as coming from Studio (assisted labelling can leave them alone).
    images = label_svc.get(f"/v1/datasets/{ds}/images").json()["images"]
    assert {i["name"]: i["source"] for i in images} == {"a.png": "studio", "e.png": "studio"}


def test_classification_import_handles_exclude_and_unknown(label_svc):
    ds = new_dataset(label_svc, "classification")
    r = studio_import(label_svc, ds)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["imported_images"] == 3
    assert body["imported_items"] == 3
    handling = body["label_handling"]
    assert (handling["excluded_boxes"], handling["unknown_boxes"]) == (3, 3)
    assert handling["images_skipped_excluded"] == 1
    assert handling["images_skipped_unknown"] == 1
    got = {
        i["image"]: [a["label"] for a in i["annotations"]] for i in items(label_svc, ds)
    }
    assert got == {
        "a__box1.png": ["cat"],
        "c__box5.png": ["fox"],
        "e__box9.png": ["dog"],
    }


def test_import_label_overrides(label_svc):
    ds = new_dataset(label_svc, "classification")
    r = studio_import(label_svc, ds, exclude_labels=["Cat"], unknown_labels=[])
    assert r.status_code == 200, r.text
    handling = r.json()["label_handling"]
    assert handling["exclude_labels"] == ["cat"]
    assert handling["unknown_labels"] == []
    got = {i["image"] for i in items(label_svc, ds)}
    # "empty"/"blank"/"unknown" now import as classes; "cat" is dropped; the
    # empty-string label is always unlabelled.
    assert "a__box1.png" not in got
    assert {"a__box2.png", "b__box3.png", "c__box6.png", "d__box7.png"} <= got
    assert "d__box8.png" not in got
    assert handling["unknown_boxes"] == 1


def test_import_label_overrides_must_not_overlap(label_svc):
    ds = new_dataset(label_svc, "classification")
    r = studio_import(label_svc, ds, exclude_labels=["x"], unknown_labels=[" X "])
    assert r.status_code == 422, r.text


def test_import_without_special_labels_has_plain_summary(svc):
    ds = new_dataset(svc, "detection")
    r = studio_import(svc, ds)
    assert r.json() == {
        "imported_images": 3,
        "imported_items": 3,
        "skipped": 1,
        "pages": 2,
    }
