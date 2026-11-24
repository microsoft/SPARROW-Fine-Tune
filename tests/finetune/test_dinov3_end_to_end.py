"""Phase 1B end to end: the ``dinov3_linear`` pipeline through HTTP + ``run_once`` on real PostgreSQL.

Covers B9 (prepare failure surfaces as a failed job), B10 (cancel during train -> ``cancelled``, no model
version), B16 (full client journey with the tiny ONNX embedder; the downloaded bundle's ``head.onnx``
reproduces a val image's label from the embedder's features) and B17 (``SPARROW_FT_EMBEDDER``).
The registry is built by ``register_builtin_pipelines`` exactly as the CLI does, and shared between
the API app and the worker. Fixtures ``settings`` (migrated fresh database) come from conftest.py.
"""

from __future__ import annotations

import hashlib
import io
import time

import pytest
from PIL import Image

from .dinov3_helpers import (
    SynItem,
    balanced_items,
    bundle_json,
    head_probabilities,
    read_bundle_bytes,
    synthetic_items,
    synthetic_png,
    write_mean_rgb_onnx,
)
from .helpers import (
    annotation_item,
    as_list,
    create_dataset,
    mark_studio,
    create_workspace,
    find_key,
    get_job,
    import_annotations,
    job_events,
    make_client,
    models_for_job,
)


@pytest.fixture
def onnx_model(tmp_path):
    return write_mean_rgb_onnx(tmp_path / "embedder" / "mean_rgb.onnx")


def settings_with_embedder(settings, embedder: str):
    from sparrow_finetune.config import Settings

    return Settings(database_url=settings.database_url, data_dir=settings.data_dir, embedder=embedder)


def builtin_registry(settings):
    from sparrow_finetune.pipelines import Registry, register_builtin_pipelines

    reg = Registry()
    register_builtin_pipelines(reg, settings)
    return reg


def upload_snapshot(client, syn: list[SynItem], settings=None, registry=None) -> int:
    """Create a plain snapshot, or prepare DINO features when given worker settings."""
    from sparrow_finetune.workers import run_once

    ws = create_workspace(client)
    ds = create_dataset(client, ws["id"], task="classification")
    for it in syn:
        r = client.put(f"/v1/datasets/{ds['id']}/images/{it.name}", content=it.png)
        assert r.is_success, r.text
    r = import_annotations(client, ds["id"], [annotation_item(it.name, it.label, it.bbox, it.group) for it in syn])
    assert r.status_code == 200, r.text
    # Unboxed items stand for Studio crops: a crop-trained dataset keeps them whole-frame.
    mark_studio(client, ds["id"], [it.name for it in syn if it.bbox is None])
    snap = client.post(f"/v1/datasets/{ds['id']}/snapshots", json={"prepare_features": settings is not None})
    assert snap.status_code == 201, snap.text
    snapshot = snap.json()
    if settings is None:
        return snapshot["id"]
    if snapshot["features"]["to_compute"]:
        assert run_once(settings, "prep-worker", {"cpu"}, registry=registry) is True
    features = client.get(f"/v1/snapshots/{snapshot['id']}/features").json()
    assert features["state"] == "ready", features
    return snapshot["id"]


def submit(client, snapshot_id, params=None) -> int:
    r = client.post("/v1/jobs", json={"pipeline": "dinov3_linear", "snapshot_id": snapshot_id, "params": params or {}})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def train_epoch_events(client, job_id) -> list:
    return [e for e in job_events(client, job_id) if e.get("stage") == "train" and find_key(e, "epoch")]


# --------------------------------------------------------------------------- B16


def test_b16_end_to_end_dinov3_linear_job(settings, onnx_model):
    from sparrow_finetune.embedders import make_embedder
    from sparrow_finetune.workers import run_once

    spec = f"onnx:{onnx_model}"
    s = settings_with_embedder(settings, spec)
    registry = builtin_registry(s)
    syn = balanced_items(train_groups=4, val_groups=2, images_per_group=2, ungrouped_per_label=1, prefix="e2e-")
    assert len(syn) >= 24 and any(it.bbox for it in syn) and any(it.bbox is None for it in syn)

    with make_client(s, registry) as client:
        pipelines = {p["id"]: p for p in as_list(client.get("/v1/pipelines").json(), "pipelines")}
        assert pipelines["dinov3_linear"]["status"] == "available"

        snapshot_id = upload_snapshot(client, syn, s, registry)
        job_id = submit(client, snapshot_id, {"max_epochs": 20})
        assert run_once(s, "e2e-worker", {"cpu"}, registry=registry) is True
        job = get_job(client, job_id)
        assert job["state"] == "succeeded", job

        (model,) = models_for_job(client, job_id)
        origin = model["starting_weights"]
        assert "randomly initialized linear head" in origin
        assert model["metadata"]["embedder"]["id"] in origin
        assert client.get(f"/v1/models/{model['id']}").json()["starting_weights"] == origin
        art = client.get(f"/v1/models/{model['id']}/artifact")
        assert art.status_code == 200
        assert hashlib.sha256(art.content).hexdigest() == model["sha256"]
        assert model["format"] == "onnx-bundle"
        assert model["metrics"]["val_macro_f1"] >= 0.9

    bundle = read_bundle_bytes(art.content)
    manifest = bundle_json(bundle, "manifest.json")
    card = bundle_json(bundle, "model_card.json")
    assert card["starting_weights"] == origin
    labels = manifest["labels"]
    assert manifest["embedder"]["model_hash"] == hashlib.sha256(onnx_model.read_bytes()).hexdigest()

    val_groups = set(card["split"]["val"]["groups"])
    val_items = [it for it in syn if it.group in val_groups]
    assert val_items, f"no grouped item in the val split: {card['split']}"
    embedder = make_embedder(spec)
    crops = []
    for it in val_items:
        im = Image.open(io.BytesIO(it.png)).convert("RGB")
        if it.bbox is not None:
            x, y, w, h = it.bbox
            im = im.crop((x, y, x + w, y + h))
        crops.append(im)
    probs = head_probabilities(bundle["head.onnx"], embedder.embed(crops), len(labels))
    predicted = [labels[i] for i in probs.argmax(axis=1)]
    assert predicted == [it.label for it in val_items]


def test_b16_cli_worker_registers_builtin_pipelines_on_default_registry(settings, monkeypatch):
    import sparrow_finetune.cli as cli
    import sparrow_finetune.pipelines as pipelines
    import sparrow_finetune.workers as workers
    from sparrow_finetune.pipelines import Registry

    fresh = Registry()
    for mod in (pipelines, workers, cli):
        if hasattr(mod, "default_registry"):
            monkeypatch.setattr(mod, "default_registry", fresh)
    monkeypatch.setenv("SPARROW_FT_DATABASE_URL", settings.database_url)
    monkeypatch.setenv("SPARROW_FT_DATA_DIR", str(settings.data_dir))
    monkeypatch.setenv("SPARROW_FT_EMBEDDER", "")

    assert cli.main(["worker", "--once"]) == 0
    assert "dinov3_linear" in fresh.ids()


def test_b16_cli_api_registers_builtin_pipelines_on_default_registry(settings, monkeypatch):
    import uvicorn

    import sparrow_finetune.api as api
    import sparrow_finetune.cli as cli
    import sparrow_finetune.pipelines as pipelines
    from sparrow_finetune.pipelines import Registry

    fresh = Registry()
    for mod in (pipelines, api, cli):
        if hasattr(mod, "default_registry"):
            monkeypatch.setattr(mod, "default_registry", fresh)
    served = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: served.append(app))
    monkeypatch.setenv("SPARROW_FT_DATABASE_URL", settings.database_url)
    monkeypatch.setenv("SPARROW_FT_DATA_DIR", str(settings.data_dir))
    monkeypatch.setenv("SPARROW_FT_EMBEDDER", "")

    assert cli.main(["api"]) == 0
    assert served, "uvicorn.run was not called"
    assert "dinov3_linear" in served[0].state.registry.ids()


# --------------------------------------------------------------------------- B9


def test_b9_job_fails_when_label_below_class_floor(settings, onnx_model):
    from sparrow_finetune.workers import run_once

    s = settings_with_embedder(settings, f"onnx:{onnx_model}")
    registry = builtin_registry(s)
    syn = synthetic_items(groups_per_label=4, prefix="b9-")
    syn.append(SynItem("b9-rare-0.png", "rare", "b9-rare-group", None, synthetic_png("cat", 777, None)))

    with make_client(s, registry) as client:
        job_id = submit(client, upload_snapshot(client, syn, s, registry), {"max_epochs": 5})
        assert run_once(s, "b9-worker", {"cpu"}, registry=registry) is True
        job = get_job(client, job_id)
        assert job["state"] == "failed", job
        assert "rare" in str(job["error"])
        assert models_for_job(client, job_id) == []


# --------------------------------------------------------------------------- B10 cancel


def test_b10_cancel_during_train_cancels_job_without_model(settings, onnx_model):
    from sparrow_finetune.embedders import make_embedder
    from sparrow_finetune.jobs import JobQueue
    from sparrow_finetune.pipelines import JobCancelled, Registry
    from sparrow_finetune.pipelines.dinov3_linear import DinoV3LinearTrainer
    from sparrow_finetune.workers import run_once

    spec = f"onnx:{onnx_model}"
    s = settings_with_embedder(settings, spec)

    class CancelRequestedAtTrainStart(DinoV3LinearTrainer):
        """Requests cancellation, waits until the worker has observed it, then trains as usual."""

        def train(self, ctx):
            JobQueue(s.database_url).request_cancel(ctx.job_id)
            deadline = time.monotonic() + 15
            while True:
                try:
                    ctx.check_cancelled()
                except JobCancelled:
                    break
                assert time.monotonic() < deadline, "worker never observed the cancel request"
                time.sleep(0.05)
            return super().train(ctx)

    registry = Registry()
    registry.register(CancelRequestedAtTrainStart(lambda: make_embedder(spec)))
    syn = synthetic_items(groups_per_label=4, prefix="b10-")

    with make_client(s, registry) as client:
        job_id = submit(client, upload_snapshot(client, syn, s, builtin_registry(s)), {"max_epochs": 50})
        assert run_once(s, "b10-worker", {"cpu"}, registry=registry, lease_seconds=0.9) is True
        assert get_job(client, job_id)["state"] == "cancelled"
        assert models_for_job(client, job_id) == []
        # train() must stop at its own per-epoch check, not run all 50 epochs first
        assert len(train_epoch_events(client, job_id)) <= 1


# --------------------------------------------------------------------------- B17


def test_b17_settings_from_env_reads_embedder(monkeypatch, tmp_path):
    from sparrow_finetune.config import Settings

    monkeypatch.setenv("SPARROW_FT_DATABASE_URL", "postgresql://u@h/db")
    monkeypatch.setenv("SPARROW_FT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SPARROW_FT_EMBEDDER", "onnx:/models/dinov3.onnx")
    assert Settings.from_env().embedder == "onnx:/models/dinov3.onnx"
    monkeypatch.delenv("SPARROW_FT_EMBEDDER")
    assert Settings.from_env().embedder == ""


def test_b17_worker_without_embedder_trains_a_prepared_dinov3_job(settings, onnx_model):
    """P1B-N8, migrated: the encoder is needed where features are computed, which is now Data
    Preparation (pinned by ``test_release_first_run``). Training only reads prepared vectors, so
    a worker with no SPARROW_FT_EMBEDDER claims and trains the job."""
    from sparrow_finetune.workers import run_once

    s2 = settings_with_embedder(settings, f"onnx:{onnx_model}")
    s = settings_with_embedder(settings, "")
    registry = builtin_registry(s)
    with make_client(s2, builtin_registry(s2)) as client:
        snapshot_id = upload_snapshot(client, synthetic_items(groups_per_label=3, prefix="b17-"), s2, builtin_registry(s2))
        job_id = submit(client, snapshot_id)
        assert run_once(s, "b17-worker", {"cpu"}, registry=registry) is True
        assert get_job(client, job_id)["state"] == "succeeded", get_job(client, job_id)


# --------------------------------------------------------------------------- FT-4 reference gate


def test_ft4_job_fails_when_reference_gate_trips(settings, onnx_model, monkeypatch):
    import sparrow_finetune.pipelines.dinov3_linear as mod
    from sparrow_finetune.workers import run_once

    monkeypatch.setattr(mod, "_reference_macro_f1", lambda *a, **k: 1.5)
    s = settings_with_embedder(settings, f"onnx:{onnx_model}")
    registry = builtin_registry(s)
    with make_client(s, registry) as client:
        # FT-19: enforced on any validation size only when reference_min_val_crops allows it.
        job_id = submit(
            client,
            upload_snapshot(client, balanced_items(prefix="ft4-"), s, registry),
            {"max_epochs": 5, "reference_min_val_crops": 0},
        )
        assert run_once(s, "ft4-worker", {"cpu"}, registry=registry) is True
        job = get_job(client, job_id)
        assert job["state"] == "failed", job
        assert "reference" in str(job["error"]).lower()
        assert models_for_job(client, job_id) == []


def test_ft19_small_val_job_succeeds_with_reference_warning(settings, onnx_model, monkeypatch):
    """Default params, a small validation set: the job trains and the model carries the warning."""
    import sparrow_finetune.pipelines.dinov3_linear as mod
    from sparrow_finetune.workers import run_once

    monkeypatch.setattr(mod, "_reference_macro_f1", lambda *a, **k: 1.5)
    s = settings_with_embedder(settings, f"onnx:{onnx_model}")
    registry = builtin_registry(s)
    with make_client(s, registry) as client:
        job_id = submit(client, upload_snapshot(client, balanced_items(prefix="ft19-"), s, registry), {"max_epochs": 5})
        assert run_once(s, "ft19-worker", {"cpu"}, registry=registry) is True
        job = get_job(client, job_id)
        assert job["state"] == "succeeded", job
        (model,) = models_for_job(client, job_id)
        detail = client.get(f"/v1/models/{model['id']}").json()
        ref = detail["reference_comparison"]
        assert ref["worse"] is True and ref["reference_macro_f1"] == 1.5
        assert 0 < ref["crops"] < 100
        events = job_events(client, job_id)
        assert any("reference gate warning" in str(e.get("message")) for e in events), events
