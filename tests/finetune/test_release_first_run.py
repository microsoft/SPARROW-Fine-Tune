"""First release: model choice for a first run and opt-in DINO feature preparation.

Executable specification, written before the implementation. Each test pins one user-visible
contract through the public HTTP API, the real worker (``run_once``) or the browser; internals
(table layout, file names, job kinds) stay free except where stated.

Contract selected here (the implementer and the UI coordinate on exactly this):

A. Model choice
   * No user model upload: ``POST /v1/base-models/local`` is gone (404/405) and the training form
     has no package file input. Starting points are: ImageNet architectures, finished compatible
     runs (``params.parent_model_id``, unchanged) and Studio imports (unchanged). The Engine Model
     Zoo is a browsing/download source only (pinned in ``test_release_model_flows`` block 3).
   * ``GET /v1/pipelines``: the ``conventional_classifier`` entry carries ``architectures``, 2-4
     rows ``{"id", "weights": "imagenet", "default": bool}`` including ``resnet50`` (the only
     default) and ``efficientnet_b0``. The ``dinov3_linear`` entry is the only one with
     ``"recommended": true`` and describes itself as ``"encoder": "frozen"``, ``"head": "random"``,
     ``"device": "cpu"``.
   * Browser: ``#model-arch`` offers exactly those ids; the DINO option of ``#pipeline-select``
     says "Recommended".

B. DINO feature preparation (Data Preparation, classification datasets only)
   * ``GET /v1/datasets/{id}/feature-prep[?input_size=N]`` ->
     ``{"items": [{"image", "crop", "status"}], "counts": {"total", "valid", "missing", "stale",
     "corrupt"}, "to_compute", "message"}``. One item per training crop. ``status`` is
     ``valid`` | ``missing`` (never computed) | ``stale`` (computed for another crop box, encoder
     or preprocessing) | ``corrupt`` (stored vector unreadable). Labels and training
     hyperparameters never change a status. Detection datasets -> 422.
   * ``POST /v1/datasets/{id}/snapshots`` with ``{"prepare_features": true[,
     "feature_input_size": N]}`` (opt-in; default off) returns the immutable snapshot with
     ``features = {"state", "total", "to_compute", "reused", "done", "job_id", "message"}``.
     ``state`` is ``queued``/``running`` while work remains, ``ready`` at once when
     ``to_compute == 0`` (then ``job_id`` is null and ``message`` says "nothing to compute").
     Without opt-in ``features`` is null. Detection + opt-in -> 422.
   * ``GET /v1/snapshots/{id}/features`` -> the same object; ``{"state": "not_prepared"}`` for a
     snapshot created without opt-in; not ``ready`` once a stored vector is corrupt.
   * The worker (``run_once``) does the preparation; a worker without an encoder never claims it.
     Vectors are persisted as files under ``data_dir`` and survive a service restart.
   * ``dinov3_linear`` jobs are accepted only on a snapshot whose features are ``ready`` (else 422
     naming "Data Preparation"); training reads the prepared vectors, so a worker with no encoder
     configured trains it and the model records the preparing encoder's hash.
   * ``conventional_classifier`` and ``yolo_detector`` bypass all of this.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from sparrow_finetune.workers import run_once

from .dinov3_helpers import (
    balanced_items,
    bundle_json,
    read_bundle_bytes,
    write_mean_rgb_onnx,
)
from .helpers import (
    annotation_item,
    create_dataset,
    create_workspace,
    get_job,
    import_annotations,
    make_client,
    mark_studio,
    models_for_job,
)
from .service_helpers import PASSWORD, add_user
from .test_conventional_classifier import SMOKE
from .test_dinov3_end_to_end import builtin_registry, settings_with_embedder
from .test_local_base_models import standalone_server  # noqa: F401  (browser fixture)

DINO = {"max_epochs": 3, "reference_gate": False}
CONV = {**SMOKE, "epochs": 1, "warmup_epochs": 1}
STATUSES = {"valid", "missing", "stale", "corrupt"}


# =========================================================================== helpers


def upload_dataset(client, syn, prefix_labels=None) -> int:
    """A crop-trained classification dataset: boxed items crop, unboxed ones are Studio crops."""
    ws = create_workspace(client)
    ds = create_dataset(client, ws["id"], task="classification")
    for it in syn:
        r = client.put(f"/v1/datasets/{ds['id']}/images/{it.name}", content=it.png)
        assert r.is_success, r.text
    r = import_annotations(
        client,
        ds["id"],
        [annotation_item(it.name, it.label, it.bbox, it.group) for it in syn],
    )
    assert r.status_code == 200, r.text
    mark_studio(client, ds["id"], [it.name for it in syn if it.bbox is None])
    return ds["id"]


def new_snapshot(client, ds: int, **body) -> dict:
    r = client.post(f"/v1/datasets/{ds}/snapshots", json=body)
    assert r.status_code == 201, r.text
    return r.json()


def feature_prep(client, ds: int, **query) -> dict:
    r = client.get(f"/v1/datasets/{ds}/feature-prep", params=query)
    assert r.status_code == 200, r.text
    body = r.json()
    assert {i["status"] for i in body["items"]} <= STATUSES, body
    counts = body["counts"]
    assert counts["total"] == len(body["items"]), body
    assert sum(counts[s] for s in STATUSES) == counts["total"], body
    assert body["to_compute"] == counts["total"] - counts["valid"], body
    return body


def tags(prep: dict) -> dict:
    return {i["image"]: i["status"] for i in prep["items"]}


def snapshot_features(client, snapshot_id: int) -> dict:
    r = client.get(f"/v1/snapshots/{snapshot_id}/features")
    assert r.status_code == 200, r.text
    return r.json()


def prepared_snapshot(client, settings, registry, ds: int, **body) -> dict:
    """Opt-in snapshot, prepared by the worker when there is work; asserts it ends ``ready``."""
    snap = new_snapshot(client, ds, prepare_features=True, **body)
    if snap["features"]["to_compute"]:
        assert run_once(settings, "w-prep", {"cpu"}, registry=registry) is True
    features = snapshot_features(client, snap["id"])
    assert features["state"] == "ready", features
    return snap


def submit(client, pipeline: str, snapshot_id: int, params: dict):
    return client.post(
        "/v1/jobs",
        json={"pipeline": pipeline, "snapshot_id": snapshot_id, "params": params},
    )


def train(client, settings, registry, pipeline, snapshot_id, params) -> dict:
    r = submit(client, pipeline, snapshot_id, params)
    assert r.status_code == 201, r.text
    assert run_once(settings, "w-train", {"cpu"}, registry=registry) is True
    job = get_job(client, r.json()["id"])
    assert job["state"] == "succeeded", job
    return job


def data_files(settings) -> set:
    root = Path(settings.data_dir)
    return {p for p in root.rglob("*") if p.is_file()} if root.exists() else set()


def syn(prefix: str):
    return balanced_items(train_groups=2, val_groups=1, prefix=prefix)


@pytest.fixture
def encoder(tmp_path) -> Path:
    return write_mean_rgb_onnx(tmp_path / "enc-a" / "encoder.onnx")


@pytest.fixture
def dino(settings, encoder):
    """A standalone service whose worker has the ONNX encoder; yields (client, settings, registry)."""
    s = settings_with_embedder(settings, f"onnx:{encoder}")
    reg = builtin_registry(s)
    with make_client(s, reg) as client:
        yield client, s, reg


def no_encoder(settings):
    s = settings_with_embedder(settings, "")
    return s, builtin_registry(s)


# =========================================================================== A. model choice


def test_first_run_offers_imagenet_architectures_and_recommends_frozen_dino_on_cpu(
    dino,
):
    client, _s, _r = dino
    pipelines = {p["id"]: p for p in client.get("/v1/pipelines").json()["pipelines"]}
    archs = pipelines["conventional_classifier"]["architectures"]
    ids = [a["id"] for a in archs]
    assert 2 <= len(archs) <= 4, archs
    assert {"resnet50", "efficientnet_b0"} <= set(ids), archs
    assert len(set(ids)) == len(ids), archs
    assert all(a["weights"] == "imagenet" for a in archs), archs
    assert [a["id"] for a in archs if a["default"]] == ["resnet50"], archs

    recommended = [pid for pid, p in pipelines.items() if p.get("recommended")]
    assert recommended == ["dinov3_linear"], pipelines
    d = pipelines["dinov3_linear"]
    assert (d["encoder"], d["head"], d["device"]) == ("frozen", "random", "cpu"), d


def test_local_model_upload_route_is_removed(dino):
    from .test_local_base_models import package, resnet_state, st_bytes

    client, _s, _r = dino
    pkg = package(st_bytes(resnet_state("resnet50", classes=3)))
    r = client.post(
        "/v1/base-models/local",
        files={"file": ("resnet50-package.zip", pkg, "application/zip")},
    )
    assert r.status_code in (404, 405), r.text
    listed = client.get("/v1/base-models").json()["base_models"]
    assert [b for b in listed if b.get("origin") == "local"] == []


def test_conventional_first_run_needs_no_preparation_and_defaults_to_resnet50(dino):
    client, s, reg = dino
    ds = upload_dataset(client, syn("cv-"))
    plain = new_snapshot(client, ds)
    assert plain.get("features") is None, plain
    assert snapshot_features(client, plain["id"])["state"] == "not_prepared"
    # An omitted model is the default architecture.
    params = {k: v for k, v in CONV.items() if k != "model"}
    job = train(client, s, reg, "conventional_classifier", plain["id"], params)
    assert job["params"]["model"] == "resnet50", job
    # The other offered architecture trains end to end with no base package at all.
    job = train(
        client,
        s,
        reg,
        "conventional_classifier",
        plain["id"],
        {**CONV, "model": "efficientnet_b0"},
    )
    (model,) = models_for_job(client, job["id"])
    assert model["sha256"]
    # Training a conventional model never prepared DINO features.
    assert snapshot_features(client, plain["id"])["state"] == "not_prepared"


# =========================================================================== B. preparation


def test_feature_prep_tags_every_crop_missing_before_any_preparation(dino):
    client, _s, _r = dino
    items = syn("m-")
    ds = upload_dataset(client, items)
    prep = feature_prep(client, ds)
    assert set(tags(prep)) == {it.name for it in items}
    assert set(tags(prep).values()) == {"missing"}, prep
    assert prep["counts"]["missing"] == len(items) == prep["to_compute"]
    boxed = {it.name: it.bbox for it in items if it.bbox}
    for item in prep["items"]:
        if item["image"] in boxed:
            assert [round(v) for v in item["crop"]] == boxed[item["image"]], item


def test_snapshot_without_opt_in_prepares_nothing(dino):
    client, s, reg = dino
    ds = upload_dataset(client, syn("n-"))
    plain = new_snapshot(client, ds)
    assert plain.get("features") is None, plain
    assert snapshot_features(client, plain["id"])["state"] == "not_prepared"
    assert run_once(s, "w-idle", {"cpu"}, registry=reg) is False  # nothing was queued
    assert feature_prep(client, ds)["counts"]["valid"] == 0


def test_opt_in_dino_snapshot_returns_progress_and_the_worker_prepares_it(dino):
    client, s, reg = dino
    items = syn("p-")
    ds = upload_dataset(client, items)
    snap = new_snapshot(client, ds, prepare_features=True)
    f = snap["features"]
    assert f["state"] in ("queued", "running"), f
    assert f["total"] == len(items) == f["to_compute"], f
    assert (f["done"], f["reused"]) == (0, 0), f
    assert isinstance(f["job_id"], int), f
    assert (f.get("message") or "").strip(), f

    assert run_once(s, "w-prep", {"cpu"}, registry=reg) is True
    assert get_job(client, f["job_id"])["state"] == "succeeded"
    done = snapshot_features(client, snap["id"])
    assert done["state"] == "ready", done
    assert done["done"] == done["to_compute"] == len(items), done
    prep = feature_prep(client, ds)
    assert set(tags(prep).values()) == {"valid"} and prep["to_compute"] == 0, prep


def test_fully_prepared_snapshot_shows_zero_work_and_queues_nothing(dino):
    client, s, reg = dino
    ds = upload_dataset(client, syn("z-"))
    prepared_snapshot(client, s, reg, ds)
    prep = feature_prep(client, ds)
    assert (
        prep["to_compute"] == 0 and "nothing to compute" in prep["message"].lower()
    ), prep

    again = new_snapshot(client, ds, prepare_features=True)
    f = again["features"]
    assert f["state"] == "ready" and f["to_compute"] == 0, f
    assert f["reused"] == f["total"] > 0 and f["job_id"] is None, f
    assert "nothing to compute" in f["message"].lower(), f
    assert run_once(s, "w-idle", {"cpu"}, registry=reg) is False


def test_label_and_hyperparameter_changes_keep_features_valid(dino, settings, encoder):
    client, s, reg = dino
    items = syn("l-")
    ds = upload_dataset(client, items)
    prepared_snapshot(client, s, reg, ds)

    it = next(i for i in items if i.bbox and i.label == "fox")
    r = client.patch(
        f"/v1/datasets/{ds}/items/{it.name}",
        json={"annotations": [{"label": "cat", "bbox": it.bbox}]},
    )
    assert r.status_code == 200, r.text
    assert set(tags(feature_prep(client, ds)).values()) == {"valid"}
    relabelled = new_snapshot(client, ds, prepare_features=True)
    assert relabelled["features"]["to_compute"] == 0, relabelled

    # Different hyperparameters reuse the same vectors: no encoder is needed to train.
    s0, reg0 = no_encoder(settings)
    hashes = set()
    for params in ({**DINO, "alpha": 1e-4}, {**DINO, "alpha": 1e-2, "max_epochs": 2}):
        job = train(client, s0, reg0, "dinov3_linear", relabelled["id"], params)
        (model,) = models_for_job(client, job["id"])
        art = client.get(f"/v1/models/{model['id']}/artifact")
        manifest = bundle_json(read_bundle_bytes(art.content), "manifest.json")
        hashes.add(manifest["embedder"]["model_hash"])
    assert hashes == {hashlib.sha256(encoder.read_bytes()).hexdigest()}
    assert set(tags(feature_prep(client, ds)).values()) == {"valid"}


def test_changed_crop_invalidates_only_that_crop(dino):
    client, s, reg = dino
    items = syn("c-")
    ds = upload_dataset(client, items)
    prepared_snapshot(client, s, reg, ds)
    it = next(i for i in items if i.bbox)
    x, y, w, h = it.bbox
    r = client.patch(
        f"/v1/datasets/{ds}/items/{it.name}",
        json={
            "annotations": [{"label": it.label, "bbox": [x + 1, y + 1, w - 2, h - 2]}]
        },
    )
    assert r.status_code == 200, r.text
    prep = feature_prep(client, ds)
    t = tags(prep)
    assert t.pop(it.name) == "stale", prep
    assert set(t.values()) == {"valid"}, prep
    assert prep["to_compute"] == 1
    snap = new_snapshot(client, ds, prepare_features=True)
    assert snap["features"]["to_compute"] == 1, snap
    assert snap["features"]["reused"] == len(items) - 1, snap


def test_changed_encoder_invalidates_every_crop(dino, settings, tmp_path):
    client, s, reg = dino
    items = syn("e-")
    ds = upload_dataset(client, items)
    prepared_snapshot(client, s, reg, ds)
    other = write_mean_rgb_onnx(tmp_path / "enc-b" / "encoder.onnx", flatten=False)
    sb = settings_with_embedder(settings, f"onnx:{other}")
    with make_client(sb, builtin_registry(sb)) as client_b:
        prep = feature_prep(client_b, ds)
        assert set(tags(prep).values()) == {"stale"}, prep
        assert prep["to_compute"] == len(items)


def test_changed_preprocessing_invalidates_every_crop(dino):
    client, s, reg = dino
    items = syn("r-")
    ds = upload_dataset(client, items)
    prepared_snapshot(client, s, reg, ds)  # default preprocessing (224 px)
    resized = feature_prep(client, ds, input_size=112)
    assert set(tags(resized).values()) == {"stale"}, resized
    assert set(tags(feature_prep(client, ds)).values()) == {"valid"}
    snap = new_snapshot(client, ds, prepare_features=True, feature_input_size=112)
    assert snap["features"]["to_compute"] == len(items), snap


def test_corrupt_stored_features_are_detected_and_recomputed(dino):
    client, s, reg = dino
    items = syn("x-")
    ds = upload_dataset(client, items)
    before = data_files(s)
    first = prepared_snapshot(client, s, reg, ds)
    written = sorted(p for p in data_files(s) - before if "jobs" not in p.parts)
    assert written, "prepared features must be persisted as files under data_dir"
    for p in written:
        p.write_bytes(b"\x00not a feature vector")

    prep = feature_prep(client, ds)
    assert prep["counts"]["corrupt"] >= 1, prep
    assert prep["counts"]["valid"] + prep["counts"]["corrupt"] == len(items), prep
    # The old snapshot can no longer train silently on damaged vectors.
    assert snapshot_features(client, first["id"])["state"] != "ready"
    r = submit(client, "dinov3_linear", first["id"], DINO)
    assert r.status_code == 422, r.text

    fixed = prepared_snapshot(client, s, reg, ds)
    assert fixed["features"]["to_compute"] == prep["counts"]["corrupt"], fixed
    assert set(tags(feature_prep(client, ds)).values()) == {"valid"}


def test_prepared_features_persist_across_a_service_restart(settings, encoder):
    s = settings_with_embedder(settings, f"onnx:{encoder}")
    with make_client(s, builtin_registry(s)) as client:
        ds = upload_dataset(client, syn("s-"))
        snap = prepared_snapshot(client, s, builtin_registry(s), ds)
    with make_client(s, builtin_registry(s)) as client:  # a fresh app process state
        assert snapshot_features(client, snap["id"])["state"] == "ready"
        assert set(tags(feature_prep(client, ds)).values()) == {"valid"}


def test_worker_without_encoder_never_claims_feature_preparation(dino, settings):
    """Supersedes ``test_b17_worker_without_embedder_leaves_dinov3_job_queued``: the encoder is
    needed where features are computed, which is now preparation, not training."""
    client, s, reg = dino
    ds = upload_dataset(client, syn("w-"))
    snap = new_snapshot(client, ds, prepare_features=True)
    s0, reg0 = no_encoder(settings)
    assert run_once(s0, "w-none", {"cpu"}, registry=reg0) is False
    assert snapshot_features(client, snap["id"])["state"] == "queued"
    assert run_once(s, "w-enc", {"cpu"}, registry=reg) is True
    assert snapshot_features(client, snap["id"])["state"] == "ready"


# =========================================================================== C. training gate


def test_dino_training_refuses_unprepared_and_unfinished_snapshots(dino):
    client, _s, _r = dino
    ds = upload_dataset(client, syn("g-"))
    plain = new_snapshot(client, ds)
    r = submit(client, "dinov3_linear", plain["id"], DINO)
    assert r.status_code == 422 and "Data Preparation" in r.json()["detail"], r.text
    pending = new_snapshot(client, ds, prepare_features=True)
    assert pending["features"]["state"] != "ready"
    r = submit(client, "dinov3_linear", pending["id"], DINO)
    assert r.status_code == 422 and "Data Preparation" in r.json()["detail"], r.text


# =========================================================================== D. detection bypass


def test_detection_datasets_bypass_feature_preparation(dino):
    client, _s, _r = dino
    ws = create_workspace(client)
    ds = create_dataset(client, ws["id"], task="detection")
    r = client.put(f"/v1/datasets/{ds['id']}/images/a.png", content=syn("d-")[0].png)
    assert r.is_success, r.text
    r = import_annotations(
        client, ds["id"], [annotation_item("a.png", "deer", [2, 2, 10, 10])]
    )
    assert r.status_code == 200, r.text
    assert client.get(f"/v1/datasets/{ds['id']}/feature-prep").status_code == 422
    r = client.post(
        f"/v1/datasets/{ds['id']}/snapshots", json={"prepare_features": True}
    )
    assert r.status_code == 422, r.text
    plain = new_snapshot(client, ds["id"])
    assert plain.get("features") is None, plain
    r = submit(
        client, "yolo_detector", plain["id"], {"model": "yolo11n.pt", "epochs": 1}
    )
    assert r.status_code == 201, r.text


# =========================================================================== E. browser


class TestBrowserFirstRun:
    """The real page against a real server with the built-in pipelines and an encoder."""

    @pytest.fixture
    def settings(self, settings, tmp_path):
        enc = write_mean_rgb_onnx(tmp_path / "enc-ui" / "encoder.onnx")
        from dataclasses import replace

        return replace(settings, embedder=f"onnx:{enc}")

    @pytest.fixture
    def registry(self, settings):
        return builtin_registry(settings)

    def test_finished_model_picker_fits_snapshot_panel(self, standalone_server, client, settings):
        pw = pytest.importorskip("playwright.sync_api")
        from .test_backlog_service_ui import _browser, _login

        upload_dataset(client, syn("picker-"))
        add_user(settings, "alice", PASSWORD)
        with pw.sync_playwright() as playwright:
            browser, page = _browser(playwright, [])
            try:
                _login(page, standalone_server)
                picker = page.get_by_label("Retrain a finished model on this snapshot")
                pw.expect(picker).to_be_visible()
                picker.evaluate("(s) => s.add(new Option('Model #123 · DINOv3 linear probe · job 123 · snapshot 123'))")
                for width in (1440, 1024, 390, 320):
                    page.set_viewport_size({"width": width, "height": 900})
                    panel = page.locator(".readiness").bounding_box()
                    for child in ("label", ".help-text", "select"):
                        box = page.locator(f"#parent-model-block {child}").first.bounding_box()
                        assert panel["x"] <= box["x"]
                        assert box["x"] + box["width"] <= panel["x"] + panel["width"]
            finally:
                browser.close()

    def test_detector_snapshot_never_requests_features(self, standalone_server, client, settings):
        pw = pytest.importorskip("playwright.sync_api")
        from .test_backlog_service_ui import _browser, _login

        ds = create_dataset(client, create_workspace(client)["id"], task="detection")["id"]
        image = syn("detector-")[0]
        assert client.put(f"/v1/datasets/{ds}/images/{image.name}", content=image.png).is_success
        assert import_annotations(client, ds, [annotation_item(image.name, "cat", [0, 0, 5, 5])]).is_success
        add_user(settings, "alice", PASSWORD)
        with pw.sync_playwright() as playwright:
            browser, page = _browser(playwright, [])
            try:
                reads = []
                page.on("request", lambda r: reads.append(r.url)
                        if r.url.endswith(("/feature-prep", "/features")) else None)
                _login(page, standalone_server)
                pw.expect(page.locator("#pipeline-select")).to_have_value("yolo_detector")
                with page.expect_response(lambda r: r.request.method == "POST" and r.url.endswith("/snapshots")) as snap:
                    page.click("#create-snapshot")
                assert snap.value.status == 201
                assert not (snap.value.request.post_data_json or {}).get("prepare_features")
                pw.expect(page.locator("#snapshot-info")).to_contain_text(f"Snapshot {snap.value.json()['id']}:")
                pw.expect(page.locator("#feature-prep")).to_be_hidden()
                pw.expect(page.locator("#feature-prep-status")).to_be_hidden()
                assert not reads
            finally:
                browser.close()

    def test_choices_upload_removed_and_pipeline_specific_preparation(
        self, standalone_server, settings, registry, client
    ):
        sync_api = pytest.importorskip("playwright.sync_api")
        from .test_backlog_service_ui import _browser, _login

        expect = sync_api.expect
        upload_dataset(client, syn("ui-"))
        add_user(settings, "alice", PASSWORD)
        problems: list[str] = []
        with sync_api.sync_playwright() as pw:
            browser, page = _browser(pw, problems)
            try:
                _login(page, standalone_server)
                dino_opt = page.locator(
                    "#pipeline-select option[value='dinov3_linear']"
                )
                expect(dino_opt).to_contain_text("Recommended")

                expect(page.locator("#feature-prep")).to_be_visible()
                expect(page.locator("#feature-prep input[type=checkbox]")).to_have_count(0)
                with page.expect_request(
                    lambda r: r.method == "POST" and r.url.endswith("/snapshots")
                ) as req:
                    page.click("#create-snapshot")
                assert req.value.post_data_json["prepare_features"] is True
                expect(page.locator("#feature-prep-status")).to_contain_text("queued")

                # Conventional first run: the offered ImageNet architectures, no upload.
                page.select_option("#pipeline-select", "conventional_classifier")
                expect(page.locator("#feature-prep")).to_be_hidden()
                expect(page.locator("#feature-prep-status")).to_be_hidden()
                feature_reads = []
                page.on("request", lambda r: feature_reads.append(r.url)
                        if r.url.endswith(("/feature-prep", "/features")) else None)
                with page.expect_response(
                    lambda r: r.request.method == "POST" and r.url.endswith("/snapshots")
                ) as ordinary:
                    page.click("#create-snapshot")
                assert not (ordinary.value.request.post_data_json or {}).get("prepare_features")
                expect(page.locator("#snapshot-info")).to_contain_text(f"Snapshot {ordinary.value.json()['id']}:")
                assert not feature_reads
                arch = page.locator("#model-arch")
                expect(arch).to_be_visible()
                expect(arch).to_have_value("resnet50")
                listing = page.request.get(f"{standalone_server}/v1/pipelines")
                assert listing.status == 200
                offered = sorted(
                    a["id"]
                    for p in listing.json()["pipelines"]
                    if p["id"] == "conventional_classifier"
                    for a in p["architectures"]
                )
                values = sorted(
                    arch.locator("option").evaluate_all("os => os.map(o => o.value)")
                )
                assert values == offered
                expect(page.locator("#local-base-model-file")).to_have_count(0)
                expect(page.locator("input[type=file][accept*='zip']:not(#annotations-file)")).to_have_count(0)
                page.select_option("#model-arch", "efficientnet_b0")
                with page.expect_request(
                    lambda r: r.method == "POST" and r.url.endswith("/v1/jobs")
                ) as req:
                    page.click("#start-training")
                assert req.value.post_data_json["params"]["model"] == "efficientnet_b0"
                expect(page.locator("#error-banner")).to_be_hidden()
            finally:
                browser.close()
        assert problems == [], problems

    def test_retraining_older_parent_without_arch_metadata_omits_model(
        self, standalone_server, settings, registry, client
    ):
        """F2: the parent's record predates ``model_arch``; the hidden architecture picker
        (resnet50 by default) must not be sent, so the server adopts the parent's network."""
        sync_api = pytest.importorskip("playwright.sync_api")
        from .test_backlog_service_ui import _browser, _login

        expect = sync_api.expect
        ds = upload_dataset(client, syn("old-parent-"))
        parent_snap = new_snapshot(client, ds)
        job = train(
            client, settings, registry, "conventional_classifier", parent_snap["id"],
            {**CONV, "model": "efficientnet_b0"},
        )
        (parent,) = models_for_job(client, job["id"])
        _sql(
            settings,
            "UPDATE model_versions SET metadata = metadata - 'model_arch' - 'model'"
            " WHERE id = %s",
            (parent["id"],),
        )
        listed = client.get(f"/v1/models/{parent['id']}").json()
        assert not {"model_arch", "model"} & set(listed.get("metadata") or {}), listed

        add_user(settings, "alice", PASSWORD)
        problems: list[str] = []
        with sync_api.sync_playwright() as pw:
            browser, page = _browser(pw, problems)
            try:
                _login(page, standalone_server)
                page.select_option("#pipeline-select", "conventional_classifier")
                with page.expect_response(
                    lambda r: r.request.method == "POST" and r.url.endswith("/snapshots")
                ) as made:
                    page.click("#create-snapshot")
                assert made.value.status == 201
                expect(page.locator("#snapshot-info")).to_contain_text(
                    f"Snapshot {made.value.json()['id']}:"
                )
                picker = page.locator("#parent-model-select")
                expect(picker.locator(f"option[value='{parent['id']}']")).to_have_count(1)
                picker.select_option(str(parent["id"]))
                expect(page.locator("#model-arch-field")).to_be_hidden()
                try:
                    with page.expect_response(
                        lambda r: r.request.method == "POST" and r.url.endswith("/v1/jobs")
                    ) as started:
                        page.click("#start-training")
                except sync_api.TimeoutError:
                    banner = page.locator("#error-banner")
                    pytest.fail(f"no POST /v1/jobs; page shows: {banner.inner_text()!r}")
                sent = started.value.request.post_data_json["params"]
                assert sent["parent_model_id"] == parent["id"], sent
                assert "model" not in sent, sent
                assert started.value.status == 201, started.value.text()
                assert started.value.json()["params"]["model"] == "efficientnet_b0"
                expect(page.locator("#error-banner")).to_be_hidden()
            finally:
                browser.close()
        assert problems == [], problems


# =========================================================================== F. regressions
#
# Release-audit regressions. F1: a finished Data Preparation run is not a "training job": it
# must not force the include_models confirmation, nor be reported as one in the deletion
# preview, while queued/running preparation and real training keep their protections.
# F2 (browser, in TestBrowserFirstRun): retraining an older conventional model whose record
# lacks ``model_arch`` must not send the hidden default architecture; the server adopts the
# parent's own. F3: preparation failure, cancel, retry and worker-crash recovery through the
# public API and the real worker.


def _sql(settings, query, params=()):
    import psycopg

    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        cur = conn.execute(query, params)
        return cur.fetchall() if cur.description else None


def _deletion_target(client, settings, kind: str, ds: int) -> str:
    if kind == "dataset":
        return f"/v1/datasets/{ds}"
    ((ws,),) = _sql(settings, "SELECT workspace_id FROM datasets WHERE id = %s", (ds,))
    return f"/v1/workspaces/{ws}"


def _claim_prep(settings, worker="w-crashed"):
    """A worker leases the preparation job and then dies without finishing it."""
    from sparrow_finetune.features import PREP_PIPELINE
    from sparrow_finetune.jobs import JobQueue

    job = JobQueue(settings.database_url).claim(
        worker, {"cpu"}, 60, pipelines=[PREP_PIPELINE]
    )
    assert job is not None and job.pipeline == PREP_PIPELINE, job
    return job


@pytest.mark.parametrize("kind", ["dataset", "workspace"])
def test_completed_preparation_only_deletes_without_include_models(dino, kind):
    client, s, reg = dino
    ds = upload_dataset(client, syn(f"pd-{kind}-"))
    snap = prepared_snapshot(client, s, reg, ds)
    assert get_job(client, snap["features"]["job_id"])["state"] == "succeeded"
    path = _deletion_target(client, s, kind, ds)

    preview = client.get(f"{path}/deletion").json()
    r = client.delete(path)
    assert r.status_code == 200, r.text
    assert client.get(f"/v1/datasets/{ds}").status_code == 404
    # The preview lists "training jobs" and models; a preparation run is neither.
    assert (preview["jobs"], preview["models"], preview["active_jobs"]) == (0, [], []), preview


@pytest.mark.parametrize("phase", ["queued", "running"])
def test_active_preparation_still_blocks_deletion(dino, phase):
    client, s, _reg = dino
    ds = upload_dataset(client, syn(f"pa-{phase}-"))
    job_id = new_snapshot(client, ds, prepare_features=True)["features"]["job_id"]
    if phase == "running":
        _claim_prep(s)
    assert get_job(client, job_id)["state"] == phase

    assert client.get(f"/v1/datasets/{ds}/deletion").json()["active_jobs"] == [job_id]
    r = client.delete(f"/v1/datasets/{ds}", params={"include_models": "true"})
    assert r.status_code == 409 and str(job_id) in r.json()["detail"], r.text
    assert client.get(f"/v1/datasets/{ds}").status_code == 200


def test_prepared_dataset_with_real_training_still_needs_include_models(dino):
    client, s, reg = dino
    ds = upload_dataset(client, syn("pt-"))
    snap = prepared_snapshot(client, s, reg, ds)
    job = train(client, s, reg, "conventional_classifier", snap["id"], CONV)
    (model,) = models_for_job(client, job["id"])

    preview = client.get(f"/v1/datasets/{ds}/deletion").json()
    assert preview["jobs"] == 1, preview  # the training job only, not the preparation
    assert [m["id"] for m in preview["models"]] == [model["id"]], preview
    r = client.delete(f"/v1/datasets/{ds}")
    assert r.status_code == 409 and "include_models" in r.json()["detail"], r.text
    assert client.get(f"/v1/datasets/{ds}").status_code == 200

    r = client.delete(f"/v1/datasets/{ds}", params={"include_models": "true"})
    assert r.status_code == 200, r.text
    assert r.json()["deleted"]["models"] == 1, r.json()


def test_failed_preparation_is_reported_blocks_training_and_a_retry_recovers(
    dino, settings, tmp_path
):
    client, s, reg = dino
    ds = upload_dataset(client, syn("pf-"))
    snap = new_snapshot(client, ds, prepare_features=True)
    first = snap["features"]["job_id"]

    # A worker configured with a different encoder claims the job and must fail it.
    other = write_mean_rgb_onnx(tmp_path / "enc-other" / "encoder.onnx", flatten=False)
    sb = settings_with_embedder(settings, f"onnx:{other}")
    assert run_once(sb, "w-wrong", {"cpu"}, registry=builtin_registry(sb)) is True
    failed = get_job(client, first)
    assert failed["state"] == "failed" and failed["error"], failed
    f = snapshot_features(client, snap["id"])
    assert f["state"] == "failed" and f["job_id"] == first, f
    assert "retry" in f["message"].lower(), f
    r = submit(client, "dinov3_linear", snap["id"], DINO)
    assert r.status_code == 422 and "Data Preparation" in r.json()["detail"], r.text

    r = client.post(f"/v1/jobs/{first}/retry")
    assert r.status_code == 201, r.text
    second = r.json()["id"]
    f = snapshot_features(client, snap["id"])
    assert (f["state"], f["job_id"]) == ("queued", second), f
    assert run_once(s, "w-right", {"cpu"}, registry=reg) is True
    f = snapshot_features(client, snap["id"])
    assert (f["state"], f["done"]) == ("ready", f["to_compute"]), f
    train(client, s, reg, "dinov3_linear", snap["id"], DINO)


def test_cancelled_preparation_can_be_retried_to_ready(dino):
    client, s, reg = dino
    ds = upload_dataset(client, syn("pc-"))
    snap = new_snapshot(client, ds, prepare_features=True)
    job_id = snap["features"]["job_id"]
    assert client.post(f"/v1/jobs/{job_id}/cancel").status_code == 200
    f = snapshot_features(client, snap["id"])
    assert f["state"] == "cancelled" and "retry" in f["message"].lower(), f
    assert run_once(s, "w-idle", {"cpu"}, registry=reg) is False

    assert client.post(f"/v1/jobs/{job_id}/retry").status_code == 201
    assert run_once(s, "w-prep", {"cpu"}, registry=reg) is True
    assert snapshot_features(client, snap["id"])["state"] == "ready"


def test_preparation_abandoned_by_a_crashed_worker_is_recovered(dino):
    client, s, reg = dino
    ds = upload_dataset(client, syn("pr-"))
    snap = new_snapshot(client, ds, prepare_features=True)
    job = _claim_prep(s)
    assert snapshot_features(client, snap["id"])["state"] == "running"
    _sql(
        s,
        "UPDATE jobs SET lease_expires_at = clock_timestamp() - interval '1 second'"
        " WHERE id = %s",
        (job.id,),
    )
    # The next worker poll reaps the expired lease and finishes the same job.
    assert run_once(s, "w-next", {"cpu"}, registry=reg) is True
    done = get_job(client, job.id)
    assert done["state"] == "succeeded" and done["attempts"] == 2, done
    f = snapshot_features(client, snap["id"])
    assert (f["state"], f["job_id"]) == ("ready", job.id), f
    assert set(tags(feature_prep(client, ds)).values()) == {"valid"}
