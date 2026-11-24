"""Release model flows (companion ``docs/plan.md`` items 1-4): executable specification.

Strategy: a few named real journeys through the public HTTP API, the worker and the browser,
not a combinatorial sweep. Each block pins one user-visible contract; unit internals are free.

1. Reuse a finished local run as the parent of a new job (``params.parent_model_id``).
   ``GET /v1/models/{id}/reuse?snapshot_id=`` previews what transfers:
   ``{"usable", "transfers": {"backbone", "head"}, "reason"}``. Real v1 -> v2 jobs on new data:
   conventional resnet18 with the same labels (backbone + head) and with changed labels
   (backbone only); DINOv3 with the same encoder (head), and a different encoder refused (422).
   The parent model stays listed with the same sha256; the child records ``parent_model_id``.
2. Crop preparation before a snapshot. ``GET /v1/datasets/{id}/crop-prep`` ->
   ``{"mode": "crop"|"whole_image", "boxed", "studio_crops", "needs_boxes": [names]}``.
   A crop snapshot with raw images that need boxes is refused (409, names listed) unless the
   user chooses: ``{"missing_boxes": "exclude"}``, drawing boxes, or detector assist. Assist that
   finds nothing leaves the image in ``needs_boxes`` (never dropped); reviewed boxes and Studio
   crops are untouched; whole-image datasets need no choice.
3. Model catalogue: ``GET /v1/catalog`` -> ``{"mode", "models": [...], "catalog_error"}``; the
   list is whatever ``sparrow_engine.zoo_models()`` returns (no second list in this service),
   ``POST /v1/catalog/models/{id}/download`` installs ONE id via ``sparrow_engine.download_model``
   / ``download_models`` and reuses an installed copy offline. Link-only / pending-rights entries
   are never downloaded and show their source link. Connected (Studio) mode lists Studio models
   and refuses downloads (409). EVERY model row (catalogue, base models, finished runs) carries
   ``finetune_tag`` exactly ``"Can be fine-tuned"`` or ``"Fine-tune not supported"``, plus a
   ``finetune_reason`` for the latter. A positive tag needs validated training weights; it is
   never inferred from a timm-like name, an ONNX/runtime file or the format. First release: the
   Zoo is not a training starting point (every Zoo row says so; a downloaded Zoo model is never a
   base model and job params cannot name one).
4. First-run architectures (supersedes the local-package upload journey, which the first release
   removes): ``efficientnet_b0`` is offered and trains with no package; resnet50 stays the
   default. Browser wording is not ResNet-specific, there is no upload control, and the tags are
   visible before any download. DINOv3 snapshots in block 1 are prepared in Data Preparation
   first, as ``test_release_first_run`` requires.

Known conflict (not weakened): ``test_ft38_whole_image_speciesnet.py::
test_crop_snapshot_keeps_boxes_and_studio_crops`` expects a crop snapshot with an unboxed raw
image to succeed silently (``left_out == 1``); block 2 requires a 409 until the user chooses.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import types
from pathlib import Path
from typing import Optional

import pytest

from sparrow_finetune.workers import run_once

from .dinov3_helpers import balanced_items, write_mean_rgb_onnx
from .helpers import (
    create_dataset,
    create_workspace,
    import_annotations,
    job_events,
    make_client,
    mark_studio,
    models_for_job,
    upload_image,
)
from .service_helpers import PASSWORD, add_user, make_service_client, studio_settings
from .studio_fake import FakeStudio
from .test_base_models import conv_zip, seed_model
from .test_conventional_classifier import SMOKE
from .test_dinov3_end_to_end import (
    builtin_registry,
    settings_with_embedder,
    upload_snapshot,
)
from .test_ft24_assist_api import _with_megadetector
from .test_local_base_models import standalone_server  # noqa: F401 - a fixture

YES = "Can be fine-tuned"
NO = "Fine-tune not supported"
TAGS = {YES, NO}
CONV = {**SMOKE, "epochs": 2, "warmup_epochs": 1}
DINO = {"max_epochs": 3, "reference_gate": False}


# =========================================================================== shared


def run_job(
    client, settings, registry, pipeline: str, snapshot_id: int, params: dict
) -> dict:
    r = client.post(
        "/v1/jobs",
        json={"pipeline": pipeline, "snapshot_id": snapshot_id, "params": params},
    )
    assert r.status_code == 201, r.text
    job_id = r.json()["id"]
    assert run_once(settings, "w-release", {"cpu"}, registry=registry) is True
    job = client.get(f"/v1/jobs/{job_id}").json()
    assert job["state"] == "succeeded", job
    (model,) = models_for_job(client, job_id)
    return {
        "job": job,
        "model": model,
        "events": [e.get("message") or "" for e in job_events(client, job_id)],
    }


def snap(client, labels=("cat", "dog", "fox"), prefix="v") -> int:
    return upload_snapshot(
        client,
        balanced_items(labels=labels, train_groups=2, val_groups=1, prefix=prefix),
    )


def dino_snap(client, settings, registry, prefix: str) -> int:
    """A classification snapshot whose DINO features were prepared (opt-in) by the worker."""
    from .test_release_first_run import prepared_snapshot, syn, upload_dataset

    ds = upload_dataset(client, syn(prefix))
    return prepared_snapshot(client, settings, registry, ds)["id"]


def model_row(client, model_id) -> dict:
    r = client.get(f"/v1/models/{model_id}")
    assert r.status_code == 200, r.text
    return r.json()


def assert_tagged(row: dict) -> None:
    assert row.get("finetune_tag") in TAGS, row
    if row["finetune_tag"] == NO:
        assert (row.get("finetune_reason") or "").strip(), row


# =========================================================================== 1. parent reuse


@pytest.fixture
def svc(settings):
    registry = builtin_registry(settings)
    with make_service_client(settings, registry) as client:
        yield client, settings, registry


def test_conventional_v1_to_v2_same_labels_keeps_parent_and_transfers_backbone_and_head(
    svc,
):
    client, settings, registry = svc
    v1 = run_job(
        client,
        settings,
        registry,
        "conventional_classifier",
        snap(client, prefix="a1-"),
        CONV,
    )
    parent = v1["model"]
    assert_tagged(parent)
    assert parent["starting_weights"] == f"{CONV['model']} · random initialization"
    assert parent["finetune_tag"] == YES

    new_data = snap(client, prefix="a2-")
    r = client.get(f"/v1/models/{parent['id']}/reuse", params={"snapshot_id": new_data})
    assert r.status_code == 200, r.text
    preview = r.json()
    assert preview["usable"] is True and preview["transfers"] == {
        "backbone": True,
        "head": True,
    }, preview

    v2 = run_job(
        client,
        settings,
        registry,
        "conventional_classifier",
        new_data,
        {**CONV, "parent_model_id": parent["id"]},
    )
    assert v2["job"]["snapshot_id"] == new_data
    assert v2["model"]["starting_weights"] == f"Finished model #{parent['id']}"
    assert v2["model"]["id"] != parent["id"]
    assert v2["model"]["metadata"].get("parent_model_id") == parent["id"]
    warm = [m for m in v2["events"] if "warm start" in m]
    assert len(warm) == 1 and "head loaded" in warm[0], v2["events"]
    # The parent is preserved, byte for byte.
    assert model_row(client, parent["id"])["sha256"] == parent["sha256"]


def test_conventional_v1_to_v2_changed_labels_transfers_backbone_only(svc):
    client, settings, registry = svc
    parent = run_job(
        client,
        settings,
        registry,
        "conventional_classifier",
        snap(client, prefix="b1-"),
        CONV,
    )["model"]
    # Label set changes (fox dropped); synthetic data only has cat/dog/fox colours.
    changed = snap(client, labels=("cat", "dog"), prefix="b2-")
    preview = client.get(
        f"/v1/models/{parent['id']}/reuse", params={"snapshot_id": changed}
    ).json()
    assert preview["usable"] is True and preview["transfers"] == {
        "backbone": True,
        "head": False,
    }, preview
    assert "label" in preview["reason"].lower()

    v2 = run_job(
        client,
        settings,
        registry,
        "conventional_classifier",
        changed,
        {**CONV, "parent_model_id": parent["id"]},
    )
    warm = [m for m in v2["events"] if "warm start" in m]
    assert len(warm) == 1 and "head loaded" not in warm[0], v2["events"]
    assert v2["model"]["metadata"].get("parent_model_id") == parent["id"]


def test_parent_of_another_architecture_or_unknown_id_is_refused(svc):
    client, settings, registry = svc
    parent = run_job(
        client,
        settings,
        registry,
        "conventional_classifier",
        snap(client, prefix="c1-"),
        CONV,
    )["model"]
    s2 = snap(client, prefix="c2-")
    for params, needle in (
        ({**CONV, "model": "resnet34", "parent_model_id": parent["id"]}, "resnet18"),
        ({**CONV, "parent_model_id": 999_999}, "999999"),
    ):
        r = client.post(
            "/v1/jobs",
            json={
                "pipeline": "conventional_classifier",
                "snapshot_id": s2,
                "params": params,
            },
        )
        assert r.status_code == 422 and needle in r.json()["detail"], r.text


def test_dinov3_v1_to_v2_needs_the_identical_encoder(settings, tmp_path):
    enc_a = write_mean_rgb_onnx(tmp_path / "a" / "enc.onnx")
    enc_b = write_mean_rgb_onnx(tmp_path / "b" / "enc.onnx", flatten=False)
    assert enc_a.read_bytes() != enc_b.read_bytes()

    # First release: DINO trains only on snapshots prepared in Data Preparation
    # (``test_release_first_run``), so every DINO snapshot here is a prepared one.
    sa = settings_with_embedder(settings, f"onnx:{enc_a}")
    reg_a = builtin_registry(sa)
    with make_client(sa, reg_a) as client:
        parent = run_job(
            client,
            sa,
            reg_a,
            "dinov3_linear",
            dino_snap(client, sa, reg_a, "d1-"),
            DINO,
        )["model"]
        assert parent["finetune_tag"] == YES
        same = dino_snap(client, sa, reg_a, "d2-")
        preview = client.get(
            f"/v1/models/{parent['id']}/reuse", params={"snapshot_id": same}
        ).json()
        assert preview["usable"] is True and preview["transfers"]["head"] is True, (
            preview
        )
        v2 = run_job(
            client,
            sa,
            reg_a,
            "dinov3_linear",
            same,
            {**DINO, "parent_model_id": parent["id"]},
        )
        assert v2["model"]["metadata"].get("parent_model_id") == parent["id"]
        assert any("warm start" in m for m in v2["events"]), v2["events"]
        assert model_row(client, parent["id"])["sha256"] == parent["sha256"]

    sb = settings_with_embedder(settings, f"onnx:{enc_b}")
    reg_b = builtin_registry(sb)
    with make_client(sb, reg_b) as client:
        other = dino_snap(client, sb, reg_b, "d3-")
        preview = client.get(
            f"/v1/models/{parent['id']}/reuse", params={"snapshot_id": other}
        ).json()
        assert preview["usable"] is False and "encoder" in preview["reason"].lower(), (
            preview
        )
        r = client.post(
            "/v1/jobs",
            json={
                "pipeline": "dinov3_linear",
                "snapshot_id": other,
                "params": {**DINO, "parent_model_id": parent["id"]},
            },
        )
        assert r.status_code == 422 and "encoder" in r.json()["detail"].lower(), r.text


# =========================================================================== 2. crop preparation


def _mixed_crop_dataset(client, settings) -> int:
    """boxed.png (reviewed box), studio.png (Studio crop), raw1/raw2.png (label only, raw)."""
    ds = create_dataset(client, create_workspace(client)["id"], task="classification")[
        "id"
    ]
    for i, n in enumerate(("boxed.png", "studio.png", "raw1.png", "raw2.png")):
        upload_image(client, ds, n, seed=i)
    r = import_annotations(
        client,
        ds,
        [
            {
                "image": "boxed.png",
                "annotations": [{"label": "deer", "bbox": [1, 1, 5, 5]}],
            },
            {"image": "studio.png", "annotations": [{"label": "fox"}]},
            {"image": "raw1.png", "annotations": [{"label": "deer"}]},
            {"image": "raw2.png", "annotations": [{"label": "fox"}]},
        ],
    )
    assert r.status_code == 200, r.text
    mark_studio(client, ds, ["studio.png"])
    return ds


def _prep(client, ds) -> dict:
    r = client.get(f"/v1/datasets/{ds}/crop-prep")
    assert r.status_code == 200, r.text
    return r.json()


def _snapshot_rows(settings, snapshot_id) -> dict:
    import psycopg

    with psycopg.connect(settings.database_url) as conn:
        return dict(
            conn.execute(
                "SELECT image_name, annotations FROM snapshot_items WHERE snapshot_id = %s",
                (snapshot_id,),
            ).fetchall()
        )


def _items(client, ds) -> dict:
    r = client.get(f"/v1/datasets/{ds}/items")
    assert r.status_code == 200, r.text
    body = r.json()
    rows = body["items"] if isinstance(body, dict) else body
    return {i["image"] if "image" in i else i["name"]: i for i in rows}


def test_crop_prep_summary_counts_a_mixed_batch(client, settings):
    ds = _mixed_crop_dataset(client, settings)
    prep = _prep(client, ds)
    assert prep["mode"] == "crop"
    assert prep["boxed"] == 1 and prep["studio_crops"] == 1
    assert sorted(prep["needs_boxes"]) == ["raw1.png", "raw2.png"]


def test_crop_snapshot_never_silently_drops_raw_images(client, settings):
    ds = _mixed_crop_dataset(client, settings)
    before = client.get(f"/v1/datasets/{ds}/snapshots").json()
    r = client.post(f"/v1/datasets/{ds}/snapshots")
    assert r.status_code == 409, r.text
    assert "raw1.png" in r.text and "raw2.png" in r.text
    assert client.get(f"/v1/datasets/{ds}/snapshots").json() == before
    assert (
        client.post(
            f"/v1/datasets/{ds}/snapshots", json={"missing_boxes": "guess"}
        ).status_code
        == 422
    )


def test_explicit_exclude_keeps_boxes_and_studio_crops(client, settings):
    ds = _mixed_crop_dataset(client, settings)
    r = client.post(f"/v1/datasets/{ds}/snapshots", json={"missing_boxes": "exclude"})
    assert r.status_code == 201, r.text
    rows = _snapshot_rows(settings, r.json()["id"])
    assert set(rows) == {"boxed.png", "studio.png"}
    assert rows["boxed.png"][0]["bbox"] is not None
    assert [(a["label"], a.get("bbox")) for a in rows["studio.png"]] == [("fox", None)]


def test_drawn_boxes_resolve_raw_images(client, settings):
    ds = _mixed_crop_dataset(client, settings)
    r = import_annotations(
        client,
        ds,
        [
            {
                "image": "raw1.png",
                "annotations": [{"label": "deer", "bbox": [2, 2, 6, 6]}],
            },
            {
                "image": "raw2.png",
                "annotations": [{"label": "fox", "bbox": [3, 3, 6, 6]}],
            },
        ],
    )
    assert r.status_code == 200, r.text
    assert _prep(client, ds)["needs_boxes"] == []
    r = client.post(f"/v1/datasets/{ds}/snapshots")
    assert r.status_code == 201, r.text
    assert set(_snapshot_rows(settings, r.json()["id"])) == {
        "boxed.png",
        "studio.png",
        "raw1.png",
        "raw2.png",
    }


def test_assist_boxes_only_raw_images_and_no_detection_is_not_dropped(
    client, settings, tmp_path, monkeypatch
):
    from sparrow_finetune import assist

    class OneHit:
        """First image gets an animal box, the next one gets nothing."""

        def __init__(self):
            self.calls = 0

        def predict(self, rgb, conf, stats=None):
            self.calls += 1
            return (
                [assist.Detection("animal", [4.0, 4.0, 20.0, 20.0], 0.9)]
                if self.calls == 1
                else []
            )

    ds = _mixed_crop_dataset(client, settings)
    boxed_before = _items(client, ds)["boxed.png"]["annotations"]
    _with_megadetector(client, tmp_path)
    det = OneHit()
    monkeypatch.setattr(assist, "load_megadetector", lambda d: det)
    r = client.post(
        f"/v1/datasets/{ds}/assist", json={"images": _prep(client, ds)["needs_boxes"]}
    )
    assert r.status_code == 200, r.text
    assert (
        det.calls == 2
    )  # raw images only: reviewed boxes and Studio crops are not re-detected
    left = _prep(client, ds)["needs_boxes"]
    assert len(left) == 1 and left[0] in {"raw1.png", "raw2.png"}
    assert _items(client, ds)["boxed.png"]["annotations"] == boxed_before
    assert client.post(f"/v1/datasets/{ds}/snapshots").status_code == 409


def test_no_detector_installed_still_offers_draw_or_exclude(client, settings):
    ds = _mixed_crop_dataset(client, settings)
    listed = client.get("/v1/assist/models").json()["models"]
    assert not any(m.get("available") for m in listed)
    assert len(_prep(client, ds)["needs_boxes"]) == 2
    assert (
        client.post(
            f"/v1/datasets/{ds}/snapshots", json={"missing_boxes": "exclude"}
        ).status_code
        == 201
    )


def test_whole_image_dataset_needs_no_box_choice(client, settings):
    ds = create_dataset(
        client, create_workspace(client)["id"], task="classification", whole_image=True
    )["id"]
    for i, n in enumerate(("a.png", "b.png")):
        upload_image(client, ds, n, seed=i)
    import_annotations(
        client,
        ds,
        [
            {"image": "a.png", "annotations": [{"label": "deer"}]},
            {"image": "b.png", "annotations": [{"label": "fox"}]},
        ],
    )
    prep = _prep(client, ds)
    assert prep["mode"] == "whole_image" and prep["needs_boxes"] == []
    r = client.post(f"/v1/datasets/{ds}/snapshots")
    assert r.status_code == 201, r.text
    assert set(_snapshot_rows(settings, r.json()["id"])) == {"a.png", "b.png"}


# =========================================================================== 3. catalogue


class ZooError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class ZooModel:
    """Same fields as ``sparrow_engine.zoo.ZooModel`` (SPARROW-Engine ``zoo.py``)."""

    id: str
    domain: str
    task: str
    format: str
    license: str
    status: str
    hosting_status: str
    zip: str
    family: tuple = ()
    alias: tuple = ()
    flavor: Optional[str] = None
    source_url: Optional[str] = None
    raw: dict = dataclasses.field(default_factory=dict, compare=False)

    @property
    def downloadable(self) -> bool:
        return self.status == "active" and self.hosting_status in {
            "hosted",
            "hosted_restricted",
        }


LINK = "https://example.org/original-weights"
ZOO = [
    ZooModel(
        "MDV6-yolov10-e",
        "camera_trap",
        "detection",
        "onnx",
        "AGPL-3.0",
        "active",
        "hosted",
        "a.zip",
    ),
    # timm-like name and a runtime file: still NOT trainable (no validated training weights).
    ZooModel(
        "AI4G-Amazon-resnet50",
        "camera_trap",
        "classification",
        "onnx",
        "MIT",
        "active",
        "hosted",
        "b.zip",
    ),
    ZooModel(
        "Partner-link-only",
        "camera_trap",
        "classification",
        "pt",
        "CC-BY-NC",
        "active",
        "link_only",
        "",
        source_url=LINK,
    ),
    ZooModel(
        "Pending-rights",
        "overhead",
        "detection",
        "onnx",
        "",
        "active",
        "pending_rights",
        "",
        source_url=LINK,
    ),
]


class FakeEngine:
    def __init__(self, models=ZOO):
        self.models = list(models)
        self.offline = False
        self.downloads: list[list[str]] = []

    def zoo_models(self, **_kw):
        if self.offline:
            raise ZooError("network unreachable (test)")
        return list(self.models)

    def download_models(self, models=None, *, dest=None, **_kw):
        ids = [models] if isinstance(models, str) else list(models or [])
        self.downloads.append(ids)
        if self.offline:
            raise ZooError("network unreachable (test)")
        assert dest is not None, "the service must choose its model directory"
        out = []
        for i in ids:
            m = next(m for m in self.models if m.id == i)
            if not m.downloadable:
                raise ZooError(f"{i} is {m.hosting_status}")
            d = Path(dest) / i
            d.mkdir(parents=True, exist_ok=True)
            (d / "manifest.toml").write_text(f'id = "{i}"\n')
            out.append(d)
        return out

    def download_model(self, model, **kw):
        return self.download_models([model], **kw)[0]

    def install(self, monkeypatch):
        mod = types.ModuleType("sparrow_engine")
        zoo = types.ModuleType("sparrow_engine.zoo")
        for m in (mod, zoo):
            m.ZooModel, m.ZooError = ZooModel, ZooError
            m.zoo_models, m.download_models, m.download_model = (
                self.zoo_models,
                self.download_models,
                self.download_model,
            )
        mod.zoo = zoo
        monkeypatch.setitem(sys.modules, "sparrow_engine", mod)
        monkeypatch.setitem(sys.modules, "sparrow_engine.zoo", zoo)
        return self


@pytest.fixture
def engine(monkeypatch):
    return FakeEngine().install(monkeypatch)


def _catalog(client) -> dict:
    r = client.get("/v1/catalog")
    assert r.status_code == 200, r.text
    return r.json()


def _download(client, model_id):
    return client.post(f"/v1/catalog/models/{model_id}/download")


def test_standalone_catalogue_is_the_engine_list_with_a_tag_on_every_model(
    client, engine
):
    body = _catalog(client)
    assert body["mode"] == "standalone"
    rows = {m["id"]: m for m in body["models"]}
    assert set(rows) == {m.id for m in ZOO}  # the Engine list, nothing added or dropped
    for row in rows.values():
        assert_tagged(row)
        assert row["installed"] is False
    # No fraudulent positive: nothing in this catalogue declares validated training weights.
    assert {r["finetune_tag"] for r in rows.values()} == {NO}
    assert rows["Partner-link-only"]["downloadable"] is False
    assert rows["Partner-link-only"]["source_url"] == LINK
    assert (
        rows["MDV6-yolov10-e"]["downloadable"] is True
        and rows["MDV6-yolov10-e"]["license"] == "AGPL-3.0"
    )
    assert engine.downloads == []  # listing never downloads


def test_download_installs_exactly_one_selected_model(client, engine):
    r = _download(client, "MDV6-yolov10-e")
    assert r.status_code in (200, 201), r.text
    assert engine.downloads == [["MDV6-yolov10-e"]]
    rows = {m["id"]: m for m in _catalog(client)["models"]}
    assert rows["MDV6-yolov10-e"]["installed"] is True
    assert rows["AI4G-Amazon-resnet50"]["installed"] is False
    # No bulk / default-set download route.
    for path in ("/v1/catalog/download", "/v1/catalog/models/download"):
        assert client.post(path, json={"ids": [m.id for m in ZOO]}).status_code in (
            404,
            405,
        )
    assert engine.downloads == [["MDV6-yolov10-e"]]


def test_zoo_model_is_never_a_training_starting_point(client, engine):
    """First release: the Zoo is for browsing / download only; training starts from an ImageNet
    architecture, a finished run or a Studio import (``test_release_first_run``)."""
    for row in _catalog(client)["models"]:
        assert row["finetune_tag"] == NO, row
        assert "starting point" in (row.get("finetune_reason") or "").lower(), row
    r = _download(client, "AI4G-Amazon-resnet50")
    assert r.status_code in (200, 201), r.text
    listed = client.get("/v1/base-models").json()["base_models"]
    assert [b for b in listed if "AI4G-Amazon-resnet50" in json.dumps(b)] == [], listed
    from .helpers import build_snapshot

    s = build_snapshot(client)["snapshot"]["id"]
    for params in (
        {**SMOKE, "zoo_model_id": "AI4G-Amazon-resnet50"},
        {**SMOKE, "base_model_id": "AI4G-Amazon-resnet50"},
    ):
        r = client.post(
            "/v1/jobs",
            json={
                "pipeline": "conventional_classifier",
                "snapshot_id": s,
                "params": params,
            },
        )
        assert r.status_code == 422, r.text


@pytest.mark.parametrize("model_id", ["Partner-link-only", "Pending-rights"])
def test_link_only_and_pending_rights_are_never_downloaded(client, engine, model_id):
    r = _download(client, model_id)
    assert r.status_code == 422, r.text
    assert LINK in r.json()["detail"]
    assert engine.downloads == []
    assert _download(client, "no-such-model").status_code == 404


def test_installed_model_is_reused_offline(client, engine):
    assert _download(client, "MDV6-yolov10-e").status_code in (200, 201)
    engine.offline = True
    body = _catalog(client)
    assert (body.get("catalog_error") or "").strip()
    rows = {m["id"]: m for m in body["models"]}
    assert rows["MDV6-yolov10-e"]["installed"] is True
    assert_tagged(rows["MDV6-yolov10-e"])
    r = _download(client, "MDV6-yolov10-e")
    assert r.status_code == 200, r.text
    assert engine.downloads == [["MDV6-yolov10-e"]]  # no second Engine call


@pytest.mark.skipif(
    not os.environ.get("SPARROW_CATALOG"),
    reason="set SPARROW_CATALOG to the Engine catalog.toml",
)
def test_real_engine_catalogue_has_no_positive_tag_without_training_weights(
    client, monkeypatch
):
    import tomllib

    data = tomllib.loads(
        Path(os.environ["SPARROW_CATALOG"]).read_text(encoding="utf-8")
    )
    entries = [
        ZooModel(
            m["id"],
            m.get("domain", ""),
            m.get("task", ""),
            m.get("format", ""),
            m.get("license", ""),
            m.get("status", "unpublished"),
            m.get("hosting_status", "pending_rights"),
            m.get("zip", ""),
            source_url=m.get("original_source_url"),
            raw=dict(m),
        )
        for m in data["model"]
    ]
    assert entries, "published Engine catalogue is empty"
    FakeEngine(entries).install(monkeypatch)
    rows = _catalog(client)["models"]
    assert len(rows) == len(entries)
    for row in rows:
        assert_tagged(row)
    assert [r["id"] for r in rows if r["finetune_tag"] == YES] == []


def test_connected_mode_lists_studio_models_without_downloads(
    settings, registry, engine
):
    studio = FakeStudio(project="p1")
    with_weights = seed_model(studio)
    runtime_only = seed_model(studio, artifact=conv_zip(weights=False))
    with make_service_client(
        studio_settings(settings), registry, studio.transport
    ) as client:
        body = _catalog(client)
        assert body["mode"] == "connected"
        rows = {str(m["id"]): m for m in body["models"]}
        assert {with_weights, runtime_only} <= set(rows)
        for row in rows.values():
            assert_tagged(row)
        assert rows[runtime_only]["finetune_tag"] == NO
        r = _download(client, "MDV6-yolov10-e")
        assert r.status_code == 409 and "Studio" in r.json()["detail"], r.text
    assert engine.downloads == []


# =========================================================================== 4. first-run architectures

# Superseded (first release): the local package upload journey (efficientnet_b0 ``.zip`` import,
# warm start from it, and the ``sparrow-uploader`` CLI cross-check) is removed together with
# ``POST /v1/base-models/local``. The independent checks kept: efficientnet_b0 trains for real,
# resnet50 stays the default, the browser wording is not ResNet-specific and tags show before any
# download. Upload removal itself is pinned in ``test_release_first_run``.

EFF = "efficientnet_b0"
EFF_PARAMS = {**SMOKE, "model": EFF, "epochs": 1, "warmup_epochs": 1}


def test_efficientnet_b0_is_offered_and_trains_while_resnet50_stays_default(svc):
    client, settings, registry = svc
    (conv,) = [
        p
        for p in client.get("/v1/pipelines").json()["pipelines"]
        if p["id"] == "conventional_classifier"
    ]
    archs = {a["id"]: a for a in conv["architectures"]}
    assert EFF in archs and archs["resnet50"]["default"] is True, archs
    out = run_job(
        client,
        settings,
        registry,
        "conventional_classifier",
        snap(client, prefix="e-"),
        EFF_PARAMS,
    )
    assert out["job"]["params"]["model"] == EFF
    assert "base_model_id" not in out["model"]["metadata"], out["model"]
    assert_tagged(out["model"])
    assert out["model"]["finetune_tag"] == YES


def test_ui_wording_is_not_resnet_specific_and_tags_show_before_download(
    standalone_server, settings, registry, client, engine
):
    sync_api = pytest.importorskip("playwright.sync_api")
    from .helpers import build_snapshot
    from .test_backlog_service_ui import _browser, _login
    from .test_studio_picker_ui import _register_classifier

    expect = sync_api.expect
    _register_classifier(registry)
    build_snapshot(client)
    add_user(settings, "alice", PASSWORD)
    problems: list[str] = []
    with sync_api.sync_playwright() as pw:
        browser, page = _browser(pw, problems)
        try:
            _login(page, standalone_server)
            page.select_option("#pipeline-select", "conventional_classifier")
            arch = page.locator("#model-arch")
            expect(arch).to_be_visible()
            expect(arch.locator(f"option[value='{EFF}']")).to_have_count(1)
            expect(page.locator("label[for='model-arch']")).not_to_contain_text(
                "ResNet-50"
            )
            expect(page.locator("#local-base-model-file")).to_have_count(0)
            expect(page.locator("#local-base-model-upload")).to_have_count(0)
            expect(page.locator("#error-banner")).to_be_hidden()

            page.click("[data-page='models']")
            for m in ZOO:
                row = page.locator(f"#catalog-list [data-catalog-id='{m.id}']")
                expect(row).to_be_visible()
                expect(row.locator(".finetune-tag")).to_have_text(NO)
                expect(row.get_by_role("button", name="Use as base")).to_have_count(0)
            link = page.locator("#catalog-list [data-catalog-id='Partner-link-only'] a")
            expect(link).to_have_attribute("href", LINK)
            assert (
                engine.downloads == []
            )  # the tags were visible before anything was downloaded
        finally:
            browser.close()
    assert problems == [], problems
