"""FT-38: per-dataset whole-image training and SpeciesNet-named classification assist.

A classification dataset trains on box crops by default: snapshots keep only boxed annotations
and leave out uploaded images that only have an image-level label (Studio crops keep theirs).
``whole_image`` turns that off: boxes fold into image-level labels and assist writes labels, not
boxes. With ``classifier: "speciesnet"`` MegaDetector boxes are named by SpeciesNet (faked here).
"""

from __future__ import annotations

import numpy as np
import psycopg

from .helpers import create_dataset, create_workspace, import_annotations, upload_image
from .test_ft24_assist_api import FakeMegaDetector, _items, _with_megadetector
from sparrow_finetune import assist


def _dataset(client, **kw):
    ws = create_workspace(client)
    return create_dataset(client, ws["id"], task="classification", **kw)


def test_whole_image_only_on_classification(client):
    ws = create_workspace(client)
    r = client.post(
        f"/v1/workspaces/{ws['id']}/datasets",
        json={"name": "d", "task": "detection", "whole_image": True},
    )
    assert r.status_code == 422
    det = create_dataset(client, ws["id"], task="detection", name="d2")
    assert (
        client.patch(
            f"/v1/datasets/{det['id']}", json={"whole_image": True}
        ).status_code
        == 422
    )
    assert _dataset(client, whole_image=True)["whole_image"] is True


def test_turning_whole_image_on_folds_boxes_into_labels(client):
    ds = _dataset(client)["id"]
    upload_image(client, ds, "a.png", seed=1)
    upload_image(client, ds, "b.png", seed=2)
    import_annotations(
        client,
        ds,
        [
            {
                "image": "a.png",
                "annotations": [
                    {"label": "deer", "bbox": [1, 1, 5, 5]},
                    {"label": "deer", "bbox": [6, 6, 5, 5]},
                    {"label": "fox", "bbox": [2, 2, 3, 3]},
                ],
            },
            {"image": "b.png", "annotations": [{"label": "fox"}]},
        ],
    )
    r = client.patch(f"/v1/datasets/{ds}", json={"whole_image": True})
    assert r.status_code == 200, r.text
    assert r.json()["whole_image"] is True and r.json()["boxes_removed"] == 3
    items = _items(client, ds)
    assert sorted((a["label"], a["bbox"]) for a in items["a.png"]["annotations"]) == [
        ("deer", None),
        ("fox", None),
    ]
    assert [(a["label"], a["bbox"]) for a in items["b.png"]["annotations"]] == [
        ("fox", None)
    ]

    # New boxes are folded too.
    r = import_annotations(
        client,
        ds,
        [{"image": "b.png", "annotations": [{"label": "elk", "bbox": [1, 1, 4, 4]}]}],
    )
    assert r.is_success, r.text
    assert [
        (a["label"], a["bbox"]) for a in _items(client, ds)["b.png"]["annotations"]
    ] == [("elk", None)]
    snap = client.post(f"/v1/datasets/{ds}/snapshots")
    assert snap.status_code == 201, snap.text
    assert snap.json()["left_out"] == 0


def test_crop_snapshot_keeps_boxes_and_studio_crops(client, settings):
    ds = _dataset(client)["id"]
    for i, n in enumerate(("boxed.png", "plain.png", "studio.png")):
        upload_image(client, ds, n, seed=i)
    import_annotations(
        client,
        ds,
        [
            {
                "image": "boxed.png",
                "annotations": [
                    {"label": "deer", "bbox": [1, 1, 5, 5]},
                    {"label": "fox"},
                ],
            },
            {"image": "plain.png", "annotations": [{"label": "deer"}]},
            {"image": "studio.png", "annotations": [{"label": "fox"}]},
        ],
    )
    with psycopg.connect(settings.database_url) as conn:
        conn.execute(
            "UPDATE dataset_images SET source = 'studio' WHERE dataset_id = %s AND name = 'studio.png'",
            (ds,),
        )
    r = client.post(f"/v1/datasets/{ds}/snapshots")
    assert r.status_code == 409, r.text
    r = client.post(f"/v1/datasets/{ds}/snapshots", json={"missing_boxes": "exclude"})
    assert r.status_code == 201, r.text
    snap = r.json()
    assert snap["left_out"] == 1
    with psycopg.connect(settings.database_url) as conn:
        rows = dict(
            conn.execute(
                "SELECT image_name, annotations FROM snapshot_items WHERE snapshot_id = %s",
                (snap["id"],),
            ).fetchall()
        )
    assert set(rows) == {"boxed.png", "studio.png"}
    assert [a["label"] for a in rows["boxed.png"]] == ["deer"]
    assert rows["boxed.png"][0]["bbox"] is not None
    assert [(a["label"], a.get("bbox")) for a in rows["studio.png"]] == [("fox", None)]


def test_crop_snapshot_with_nothing_boxed_is_refused(client):
    ds = _dataset(client)["id"]
    upload_image(client, ds, "a.png", seed=1)
    import_annotations(
        client, ds, [{"image": "a.png", "annotations": [{"label": "deer"}]}]
    )
    r = client.post(f"/v1/datasets/{ds}/snapshots")
    assert r.status_code == 422
    assert "whole-image" in r.json()["detail"]


class FakeSpeciesNet:
    def __init__(self):
        self.calls = []

    def label_for(self, rgb, bbox, known=()):
        self.calls.append(list(known))
        return "Dasyprocta", 0.9


def _assist_setup(client, tmp_path, monkeypatch, sn=True, **kw):
    _with_megadetector(client, tmp_path)
    monkeypatch.setattr(assist, "load_megadetector", lambda d: FakeMegaDetector())
    fake = FakeSpeciesNet()
    if sn:
        monkeypatch.setattr(assist, "load_speciesnet", lambda d: fake)
    ds = _dataset(client, **kw)["id"]
    upload_image(client, ds, "a.png", seed=1)
    import_annotations(
        client,
        ds,
        [
            {
                "image": "a.png",
                "annotations": [{"label": "Pecari"}, {"label": "Dasyprocta"}],
            }
        ],
    )
    upload_image(client, ds, "b.png", seed=2)
    return ds, fake


def test_speciesnet_names_each_megadetector_box(client, tmp_path, monkeypatch):
    ds, fake = _assist_setup(client, tmp_path, monkeypatch)
    r = client.post(f"/v1/datasets/{ds}/assist", json={"classifier": "speciesnet"})
    assert r.status_code == 200, r.text
    assert r.json()["classifier"] == "speciesnet"
    items = _items(client, ds)
    for name in ("a.png", "b.png"):
        # Only the animal box is kept and SpeciesNet's name replaces the item's label.
        assert [(a["label"], a["bbox"]) for a in items[name]["annotations"]] == [
            ("Dasyprocta", [4.0, 4.0, 20.0, 20.0])
        ], name
    assert fake.calls and fake.calls[0] == ["Dasyprocta", "Pecari"]


def test_whole_image_assist_writes_image_labels(client, tmp_path, monkeypatch):
    ds, _ = _assist_setup(client, tmp_path, monkeypatch, whole_image=True)
    summary = client.get(f"/v1/datasets/{ds}/assist/summary").json()["sources"][
        "upload"
    ]
    assert (
        summary["with_boxes"] == 1 and summary["empty"] == 1
    )  # a has labels, b has none
    r = client.post(f"/v1/datasets/{ds}/assist", json={"classifier": "speciesnet"})
    assert r.status_code == 200, r.text
    items = _items(client, ds)
    assert [(a["label"], a["bbox"]) for a in items["b.png"]["annotations"]] == [
        ("Dasyprocta", None)
    ]
    # a already had labels: left alone without overwrite.
    assert sorted(a["label"] for a in items["a.png"]["annotations"]) == [
        "Dasyprocta",
        "Pecari",
    ]


def test_classifier_validation(client, tmp_path, monkeypatch):
    ds, _ = _assist_setup(client, tmp_path, monkeypatch, sn=False)
    both = {"classifier": "speciesnet", "animal_label": "deer"}
    assert client.post(f"/v1/datasets/{ds}/assist", json=both).status_code == 422
    other = {"classifier": "speciesnet", "model_id": 999}
    assert client.post(f"/v1/datasets/{ds}/assist", json=other).status_code == 422
    # Not installed (tmp weights dir has no SpeciesNet package).
    r = client.post(f"/v1/datasets/{ds}/assist", json={"classifier": "speciesnet"})
    assert r.status_code == 503 and "SpeciesNet" in r.json()["detail"]
    listed = client.get("/v1/assist/models").json()["classifiers"]
    assert listed == [
        {"id": "speciesnet", "name": assist.SPECIESNET_NAME, "available": False}
    ]
    ws = create_workspace(client)
    det = create_dataset(client, ws["id"], task="detection", name="det")["id"]
    r = client.post(f"/v1/datasets/{det}/assist", json={"classifier": "speciesnet"})
    assert r.status_code == 422


def _classifier(labels, probs):
    clf = object.__new__(assist.SpeciesNetCropClassifier)
    clf.labels = labels
    clf.taxonomy = assist.speciesnet_taxonomy(len(labels)) or [
        {"name": x, "keys": {assist.label_key(x)}} for x in labels
    ]
    clf.scores = lambda rgb, bbox: np.asarray(probs, dtype=float)
    return clf


def test_label_for_rolls_up_the_taxonomy():
    tax = assist.speciesnet_taxonomy(2498)
    assert tax is not None and len(tax) == 2498
    agoutis = [i for i, t in enumerate(tax) if "dasyprocta" in t["keys"]]
    peccary = next(i for i, t in enumerate(tax) if "pecaritajacu" in t["keys"])
    assert len(agoutis) >= 2
    probs = np.zeros(2498)
    for i in agoutis:  # spread across agouti species, none above the peccary on its own
        probs[i] = 0.5 / len(agoutis)
    probs[peccary] = 0.3
    probs[0] += 1.0 - probs.sum()
    labels = [t["name"] for t in tax]
    clf = _classifier(labels, probs)
    rgb = np.zeros((8, 8, 3), np.uint8)
    label, p = clf.label_for(rgb, [0, 0, 4, 4], ["Pecari", "Dasyprocta"])
    assert label == "Dasyprocta" and abs(p - 0.5) < 1e-9
    # No dataset label matches: SpeciesNet's own best class, by common name.
    label, _ = clf.label_for(rgb, [0, 0, 4, 4], ["Panthera"])
    assert label == tax[int(np.argmax(probs))]["name"]


def test_label_for_skips_blank_without_known_labels():
    clf = _classifier(["blank", "deer"], [0.7, 0.3])
    assert clf.label_for(np.zeros((4, 4, 3), np.uint8), [0, 0, 2, 2]) == ("deer", 0.3)
