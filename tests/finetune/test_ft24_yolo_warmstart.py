"""FT-24: a ``yolo_detector`` model pushed to Studio and imported back starts a new detector job.

Pinned here:

- the export carries the trained checkpoint as ``weights.pt`` (sha256 in the inner manifest ``files``),
  ``model_arch`` (the job's ``model`` param) and ``labels``;
- ``base_models.inspect_bundle`` accepts that artifact, refuses an older export without ``weights.pt``
  (re-train hint) and a ``weights.pt`` that is not an ultralytics checkpoint or pickles other classes;
- ``validate_params`` accepts ``base_model_id``; ``check_base_model`` compares architectures by name stem;
- training starts from the staged ``weights.pt`` and re-heads it for a different label set;
- the whole flow through the HTTP API and the worker (import -> submit -> run -> succeeded).
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from .dinov3_helpers import bundle_json, make_context, read_bundle
from .helpers import get_job, job_events, models_for_job
from .service_helpers import make_service_client, studio_settings
from .studio_fake import FakeStudio
from .test_base_models import import_model, seed_model, studio_manifest
from .test_yolo_end_to_end import builtin_registry, submit, upload_detection_snapshot
from .yolo_helpers import CPU_PARAMS, DetItem, det_items, det_snapshot_items


@pytest.fixture(scope="module", autouse=True)
def yolo_offline():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("YOLO_OFFLINE", "1")
        yield


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def new_trainer():
    from sparrow_finetune.pipelines.yolo_detector import YoloDetectorTrainer

    return YoloDetectorTrainer()


def relabel(items: list[DetItem], label: str) -> list[DetItem]:
    """The same images with every box renamed to ``label`` (a 1-class job from a 2-class base)."""
    return [
        DetItem(it.name, it.group, [(label, box) for _, box in it.boxes], it.png)
        for it in items
    ]


@pytest.fixture(scope="module")
def base_run(tmp_path_factory):
    """One real CPU training of a cat/dog detector, exported."""
    root = tmp_path_factory.mktemp("ft24-base")
    trainer = new_trainer()
    items = det_snapshot_items(det_items(prefix="b-"), root / "images")
    ctx, rec = make_context(trainer, items, root / "work", CPU_PARAMS)
    trainer.prepare(ctx)
    trainer.train(ctx)
    metrics = trainer.evaluate(ctx)
    result = trainer.export(ctx)
    artifact = Path(result.path).read_bytes()
    return {
        "ctx": ctx,
        "metrics": metrics,
        "result": result,
        "artifact": artifact,
        "bundle": read_bundle(result.path),
    }


def rezip(artifact: bytes, *, drop=(), replace=None, manifest_over=None) -> bytes:
    """A copy of an exported artifact with members dropped/replaced (manifest ``files`` kept in sync)."""
    replace = dict(replace or {})
    with zipfile.ZipFile(io.BytesIO(artifact)) as zf:
        members = {n: zf.read(n) for n in zf.namelist()}
    manifest = json.loads(members.pop("manifest.json"))
    for name in drop:
        members.pop(name, None)
        manifest["files"].pop(name, None)
    for name, data in replace.items():
        members[name] = data
        manifest["files"][name] = sha(data)
    manifest.update(manifest_over or {})
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
        zf.writestr("manifest.json", json.dumps(manifest))
    return buf.getvalue()


def studio_bundle(artifact: bytes, **over) -> bytes:
    from .test_base_models import tar_bundle

    manifest = studio_manifest(
        artifact,
        pipeline="yolo_detector",
        task="detection",
        labels=["cat", "dog"],
        **over,
    )
    return tar_bundle(
        [("model.zip", artifact), ("manifest.json", json.dumps(manifest).encode())]
    )


# --------------------------------------------------------------------------- export


def test_export_includes_checkpoint_as_weights_pt(base_run):
    bundle = base_run["bundle"]
    manifest = bundle_json(bundle, "manifest.json")
    assert "weights.pt" in bundle
    assert manifest["files"]["weights.pt"] == sha(bundle["weights.pt"])
    assert manifest["model_arch"] == CPU_PARAMS["model"]
    assert manifest["labels"] == ["cat", "dog"]
    trained = json.loads((base_run["ctx"].workdir / "train.json").read_text())
    assert bundle["weights.pt"] == Path(trained["weights"]).read_bytes()


# --------------------------------------------------------------------------- inspect_bundle


def test_inspect_bundle_accepts_yolo_export(base_run):
    from sparrow_finetune import base_models

    facts = base_models.inspect_bundle(studio_bundle(base_run["artifact"]), None)
    assert facts["pipeline"] == "yolo_detector"
    assert facts["model_arch"] == CPU_PARAMS["model"]
    assert facts["labels"] == ["cat", "dog"]
    assert facts["sha256"] == sha(base_run["artifact"])
    assert base_models.WARM_START_PIPELINES == frozenset(
        {"conventional_classifier", "yolo_detector", "dinov3_linear"}
    )
    assert base_models.WARM_START_PIPELINE == "conventional_classifier"


def test_inspect_bundle_refuses_old_yolo_export_without_weights(base_run):
    from sparrow_finetune import base_models

    old = rezip(base_run["artifact"], drop=["weights.pt"])
    with pytest.raises(base_models.NotImportable, match="re-train"):
        base_models.inspect_bundle(studio_bundle(old), None)


def test_inspect_bundle_refuses_state_dict_as_yolo_weights(base_run):
    import torch

    from sparrow_finetune import base_models

    buf = io.BytesIO()
    torch.save({"w": torch.zeros(2)}, buf)
    bad = rezip(base_run["artifact"], replace={"weights.pt": buf.getvalue()})
    with pytest.raises(base_models.BundleError, match="weights.pt"):
        base_models.inspect_bundle(studio_bundle(bad), None)


class Payload:
    """A class outside torch.nn / ultralytics: its presence in the pickle must refuse the bundle."""

    def __reduce__(self):
        return (print, ("pwned",))


def _foreign_module_checkpoint() -> bytes:
    import torch

    class Sneaky(torch.nn.Module):
        pass

    Sneaky.__module__ = __name__
    Sneaky.__qualname__ = "Sneaky"
    globals()["Sneaky"] = Sneaky
    buf = io.BytesIO()
    torch.save({"model": Sneaky()}, buf)
    return buf.getvalue()


@pytest.mark.parametrize("kind", ["raw_pickle", "torch_zip_foreign_module"])
def test_inspect_bundle_refuses_pickle_with_foreign_globals(base_run, kind):
    import pickle

    from sparrow_finetune import base_models

    if kind == "raw_pickle":
        evil = pickle.dumps({"model": Payload()})
    else:
        evil = _foreign_module_checkpoint()
    bad = rezip(base_run["artifact"], replace={"weights.pt": evil})
    with pytest.raises(base_models.BundleError, match="weights.pt"):
        base_models.inspect_bundle(studio_bundle(bad), None)


def test_inspect_bundle_checks_weights_sha_before_loading(base_run, monkeypatch):
    import torch

    from sparrow_finetune import base_models

    with zipfile.ZipFile(io.BytesIO(base_run["artifact"])) as zf:
        members = {n: zf.read(n) for n in zf.namelist()}
    members["weights.pt"] = members["weights.pt"] + b"tampered"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    monkeypatch.setattr(
        torch, "load", lambda *a, **k: pytest.fail("torch.load before the sha check")
    )
    with pytest.raises(base_models.BundleError, match="sha256"):
        base_models.inspect_bundle(studio_bundle(buf.getvalue()), None)


# --------------------------------------------------------------------------- params / checks


def test_validate_params_accepts_base_model_id():
    t = new_trainer()
    assert t.supports_base_model is True
    assert "base_model_id" not in t.validate_params({})
    assert t.validate_params({"base_model_id": 3})["base_model_id"] == 3
    for bad in ("3", 0, -1, True, 1.5):
        with pytest.raises(ValueError, match="base_model_id"):
            t.validate_params({"base_model_id": bad})


def test_check_base_model_compares_yolo_arch_by_stem(settings, base_run):
    from sparrow_finetune import base_models

    s = settings
    facts = base_models.inspect_bundle(studio_bundle(base_run["artifact"]), None)
    row, created = base_models.store(s, "p1", "ft24-stem", facts)
    assert created
    assert row["pipeline"] == "yolo_detector"
    base_models.check_base_model(
        s.database_url, row["id"], "yolo_detector", "yolo11n.yaml"
    )
    base_models.check_base_model(
        s.database_url, row["id"], "yolo_detector", "yolo11n.pt"
    )
    with pytest.raises(ValueError, match="yolov8s"):
        base_models.check_base_model(
            s.database_url, row["id"], "yolo_detector", "yolov8s.pt"
        )
    with pytest.raises(ValueError, match="conventional_classifier"):
        base_models.check_base_model(
            s.database_url, row["id"], "conventional_classifier", "yolo11n.yaml"
        )
    staged = base_models.stage(s, row["id"], s.data_dir / "stage-ft24")
    assert Path(staged["path"]).name == "yolo_detector_bundle.zip"
    assert Path(staged["path"]).read_bytes() == base_run["artifact"]


# --------------------------------------------------------------------------- warm-start training


def test_warm_start_trains_from_base_weights_with_new_labels(
    tmp_path, base_run, monkeypatch
):
    import ultralytics

    real_yolo = ultralytics.YOLO
    built: list[str] = []

    def spy(model=None, *args, **kwargs):
        built.append(str(model))
        return real_yolo(model, *args, **kwargs)

    monkeypatch.setattr(ultralytics, "YOLO", spy)
    trainer = new_trainer()
    items = det_snapshot_items(
        relabel(det_items(prefix="w-"), "animal"), tmp_path / "images"
    )
    ctx, rec = make_context(
        trainer, items, tmp_path / "work", {**CPU_PARAMS, "base_model_id": 5}
    )
    assert ctx.labels == ["animal"]
    ctx.base_model = {
        "id": 5,
        "model_arch": CPU_PARAMS["model"],
        "labels": ["cat", "dog"],
        "path": base_run["result"].path,
        "sha256": sha(base_run["artifact"]),
    }
    trainer.prepare(ctx)
    staged = ctx.workdir / "base_weights.pt"
    assert staged.read_bytes() == base_run["bundle"]["weights.pt"]
    trainer.train(ctx)
    assert built[0] == str(staged)
    assert any(s == "train" and "warm start" in m for s, m, _ in rec.events)
    trainer.evaluate(ctx)
    result = trainer.export(ctx)
    manifest = bundle_json(read_bundle(result.path), "manifest.json")
    assert manifest["labels"] == ["animal"]
    assert manifest["base_model_id"] == 5
    assert result.metadata["base_model_id"] == 5


def test_warm_start_refuses_tampered_staged_weights(tmp_path, base_run):
    trainer = new_trainer()
    items = det_snapshot_items(det_items(prefix="t-"), tmp_path / "images")
    ctx, _ = make_context(
        trainer, items, tmp_path / "work", {**CPU_PARAMS, "base_model_id": 5}
    )
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(io.BytesIO(base_run["artifact"])) as zf:
        members = {n: zf.read(n) for n in zf.namelist()}
    members["weights.pt"] += b"x"
    with zipfile.ZipFile(bad, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    ctx.base_model = {
        "id": 5,
        "model_arch": CPU_PARAMS["model"],
        "labels": ["cat", "dog"],
        "path": bad,
        "sha256": sha(bad.read_bytes()),
    }
    with pytest.raises(ValueError, match="sha256"):
        trainer.prepare(ctx)


# --------------------------------------------------------------------------- end to end


def test_yolo_base_model_import_submit_and_train_end_to_end(settings, base_run):
    from sparrow_finetune.workers import run_once

    s = studio_settings(settings)
    registry = builtin_registry(s)
    studio = FakeStudio(project="p1", page_size=2)
    sid = seed_model(
        studio,
        base_run["artifact"],
        pipeline="yolo_detector",
        task="detection",
        labels=["cat", "dog"],
    )
    with make_service_client(s, registry, studio.transport) as client:
        r = import_model(client, sid)
        assert r.status_code in (200, 201), r.text
        base = r.json()
        assert base["pipeline"] == "yolo_detector"
        snap = upload_detection_snapshot(
            client, relabel(det_items(prefix="e2e24-"), "animal")
        )
        job_id = submit(client, snap, {**CPU_PARAMS, "base_model_id": base["id"]})
        assert run_once(s, "ft24-worker", {"cpu"}, registry=registry) is True
        job = get_job(client, job_id)
        assert job["state"] == "succeeded", job
        messages = [e["message"] for e in job_events(client, job_id)]
        assert any("warm start" in m for m in messages), messages
        (model,) = models_for_job(client, job_id)
        assert model["metadata"]["base_model_id"] == base["id"]
        assert model["metadata"]["labels"] == ["animal"]


def test_submit_yolo_with_mismatched_arch_is_422(settings, base_run):
    s = studio_settings(settings)
    registry = builtin_registry(s)
    studio = FakeStudio(project="p1", page_size=2)
    sid = seed_model(
        studio,
        base_run["artifact"],
        pipeline="yolo_detector",
        task="detection",
        labels=["cat", "dog"],
    )
    with make_service_client(s, registry, studio.transport) as client:
        r = import_model(client, sid)
        assert r.status_code in (200, 201), r.text
        base = r.json()
        snap = upload_detection_snapshot(client, det_items(prefix="mm24-"))
        r = client.post(
            "/v1/jobs",
            json={
                "pipeline": "yolo_detector",
                "snapshot_id": snap,
                "params": {
                    **CPU_PARAMS,
                    "model": "yolov8s.pt",
                    "base_model_id": base["id"],
                },
            },
        )
        assert r.status_code == 422, r.text
        assert "yolov8s" in r.json()["detail"]
