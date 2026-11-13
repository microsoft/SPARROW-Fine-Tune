"""FT-35: assisted labelling on a classification dataset, on real PostgreSQL.

MegaDetector is a fake (fixed animal + person boxes). The rules pinned here: only animal boxes are
kept; each box takes ``animal_label``, else the item's single image-level label, else ``animal``; an
item with image-level labels but no box is eligible without overwrite; confirmed and excluded items
stay protected; the summary counts only real boxes for classification.
"""

from __future__ import annotations

from .helpers import create_dataset, create_workspace, import_annotations, upload_image
from .test_ft24_assist_api import (  # noqa: F401  (fixture re-export)
    FakeMegaDetector,
    _insert_model,
    _items,
    _with_megadetector,
    fake,
)
from sparrow_finetune import assist


def _cls_setup(client):
    ws = create_workspace(client)
    ds = create_dataset(client, ws["id"], task="classification")["id"]
    for i, n in enumerate(("a.png", "b.png", "c.png", "d.png", "e.png", "f.png")):
        upload_image(client, ds, n, seed=i)
    r = import_annotations(
        client,
        ds,
        [
            {"image": "a.png", "annotations": [{"label": "deer"}]},  # image-level, pending
            {"image": "b.png", "annotations": [{"label": "deer"}, {"label": "fox"}]},  # two labels
            {"image": "c.png", "annotations": [{"label": "fox", "bbox": [1, 1, 8, 8]}]},  # boxed
            {"image": "d.png", "annotations": [{"label": "deer"}]},  # confirmed below
            {"image": "e.png", "annotations": [{"label": "fox"}]},  # excluded below
        ],
    )
    assert r.is_success, r.text
    client.patch(f"/v1/datasets/{ds}/items/d.png", json={"review_status": "confirmed"})
    client.patch(f"/v1/datasets/{ds}/items/e.png", json={"review_status": "excluded"})
    return ds


def test_megadetector_on_classification_keeps_animals_and_inherits_labels(
    client, tmp_path, monkeypatch
):
    _with_megadetector(client, tmp_path)
    md = FakeMegaDetector()
    monkeypatch.setattr(assist, "load_megadetector", lambda d: md)
    ds = _cls_setup(client)

    summary = client.get(f"/v1/datasets/{ds}/assist/summary").json()["sources"]["upload"]
    # a, b (image-level only) and f (no item) are empty; c has a box.
    assert summary == {"images": 6, "empty": 3, "with_boxes": 1, "confirmed": 1, "excluded": 1}

    r = client.post(f"/v1/datasets/{ds}/assist", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["items_written"] == 3 and body["boxes"] == 3
    reasons = {s["image"]: s["reason"] for s in body["skipped"]}
    assert set(reasons) == {"c.png", "d.png", "e.png"}

    items = _items(client, ds)
    want = {"a.png": "deer", "b.png": "animal", "f.png": "animal"}
    for name, label in want.items():
        anns = items[name]["annotations"]
        assert [(a["label"], a["bbox"]) for a in anns] == [(label, [4.0, 4.0, 20.0, 20.0])], name
        assert items[name]["review_status"] == "pending"
    assert items["c.png"]["annotations"][0]["label"] == "fox"  # boxed item left alone
    assert items["d.png"]["review_status"] == "confirmed"
    assert items["d.png"]["annotations"][0]["bbox"] is None

    # animal_label wins over the inherited label; overwrite reaches confirmed items, never excluded.
    r = client.post(
        f"/v1/datasets/{ds}/assist",
        json={"overwrite": True, "animal_label": "elk"},
    )
    assert r.status_code == 200, r.text
    items = _items(client, ds)
    for name in ("a.png", "b.png", "c.png", "d.png", "f.png"):
        assert [a["label"] for a in items[name]["annotations"]] == ["elk"], name
    assert items["e.png"]["review_status"] == "excluded"
    assert items["e.png"]["annotations"][0]["label"] == "fox"


def test_trained_detector_on_classification_keeps_its_labels(client, settings, fake):  # noqa: F811 (pytest fixture)
    ds = _cls_setup(client)
    import_annotations(client, ds, [{"image": "f.png", "annotations": [{"label": "cat"}, ]}])
    snap = client.post(f"/v1/datasets/{ds}/snapshots").json()
    model = _insert_model(settings, snap["id"])
    r = client.post(f"/v1/datasets/{ds}/assist", json={"model_id": model, "images": ["f.png"]})
    assert r.status_code == 200, r.text
    # "bird" is not a dataset label and is left out, as on detection datasets.
    assert r.json()["unknown_labels"] == ["bird"]
    anns = _items(client, ds)["f.png"]["annotations"]
    assert [(a["label"], a["bbox"]) for a in anns] == [("cat", [4.0, 4.0, 20.0, 20.0])]
