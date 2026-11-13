"""Warm start from a model this service pushed to Sparrow Studio (base models).

The flow: a ``conventional_classifier`` model is pushed to Studio, listed back from
``GET /projects/{project}/models``, imported from its bundle into the local
``base_models`` table, and named as ``base_model_id`` in a new training job, which
then starts from its weights.

Pinned here:

- the Studio list and import routes, the bundle checks (safe tar members, manifest
  schema and source, sha256 against manifest and header, pipeline, ``weights.pt``),
  idempotent re-import and the local list;
- ``base_model_id`` validation at submit time (unknown id, architecture mismatch,
  pipelines that cannot warm start);
- the warm start itself: backbone loaded, head kept only when the labels are the same
  list in the same order, otherwise re-initialised with an event saying so;
- push stays project-scoped;
- one end-to-end run through the HTTP API.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest

from sparrow_finetune.workers import run_once

from .dinov3_helpers import balanced_items, make_context, snapshot_items
from .helpers import job_events, models_for_job
from .service_helpers import make_service_client, studio_settings
from .studio_fake import API, FakeStudio
from .test_conventional_classifier import SMOKE, run_stages, trainer
from .test_conventional_end_to_end import submit as submit_conventional
from .test_dinov3_end_to_end import builtin_registry, upload_snapshot
from .test_studio_connector import trained_model

REPO = Path(__file__).resolve().parents[2]
LABELS = ["cat", "dog", "fox"]
WARM = {**SMOKE, "epochs": 2, "warmup_epochs": 1}


# --------------------------------------------------------------------------- builders


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tiny_weights() -> bytes:
    import torch

    buf = io.BytesIO()
    torch.save({"fc.weight": torch.zeros(3, 4), "fc.bias": torch.zeros(3)}, buf)
    return buf.getvalue()


def conv_zip(
    labels=LABELS, arch="resnet18", weights=True, pipeline="conventional_classifier"
) -> bytes:
    """A ``conventional_classifier_bundle.zip`` look-alike (no real ONNX inside)."""
    files = {"model.onnx": b"onnx", "model_card.json": b"{}"}
    if weights:
        files["weights.pt"] = tiny_weights()
    manifest = {
        "pipeline": pipeline,
        "model": arch,
        "model_arch": arch,
        "labels": list(labels),
        "files": {name: sha(data) for name, data in files.items()},
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
        zf.writestr("manifest.json", json.dumps(manifest))
    return buf.getvalue()


def studio_manifest(artifact: bytes, **over) -> dict:
    manifest = {
        "schema": "sparrow-finetune/model-v1",
        "model_id": 7,
        "pipeline": "conventional_classifier",
        "task": "classification",
        "format": "onnx-bundle",
        "labels": list(LABELS),
        "metrics": {"val_macro_f1": 0.9},
        "sha256": sha(artifact),
        "created_at": "2026-10-01T00:00:00+00:00",
        "source": "sparrow-fine-tune",
    }
    manifest.update(over)
    return manifest


def tar_bundle(members: list[tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def seed_model(studio: FakeStudio, artifact: bytes = None, members=None, **over) -> str:
    artifact = conv_zip() if artifact is None else artifact
    manifest = studio_manifest(artifact, **over)
    if members is None:
        members = [
            ("model.zip", artifact),
            ("manifest.json", json.dumps(manifest).encode()),
        ]
    return studio.add_model(manifest, tar_bundle(members))


def import_model(client, studio_model_id, project="p1"):
    return client.post(
        "/v1/connectors/studio/models/import",
        json={"project": project, "studio_model_id": studio_model_id},
    )


def local_base_models(client) -> list:
    r = client.get("/v1/base-models")
    assert r.status_code == 200, r.text
    return r.json()["base_models"]


# --------------------------------------------------------------------------- fixtures


@pytest.fixture
def studio():
    return FakeStudio(project="p1", page_size=2)


@pytest.fixture
def svc(settings, registry, studio):
    with make_service_client(
        studio_settings(settings), registry, studio.transport
    ) as client:
        yield client


@pytest.fixture
def conv_svc(settings, studio):
    """Built-in pipelines (conventional_classifier, ...) with Studio configured."""
    s = studio_settings(settings)
    registry = builtin_registry(s)
    with make_service_client(s, registry, studio.transport) as client:
        yield client, s, registry


# --------------------------------------------------------------------------- list


def test_list_remote_studio_models(svc, studio):
    sid = seed_model(studio)
    studio.add_model(studio_manifest(b"x"), b"", project="other")
    r = svc.get("/v1/connectors/studio/models", params={"project": "p1"})
    assert r.status_code == 200, r.text
    (model,) = r.json()["models"]
    assert model["id"] == sid
    assert model["pipeline"] == "conventional_classifier"
    assert model["labels"] == LABELS
    assert model["source_model_id"] == 7
    assert model["importable"] is True
    listed = [q for q in studio.requests if q.url.path == f"{API}/projects/p1/models"]
    assert len(listed) == 1 and listed[0].method == "GET"


def test_list_marks_other_pipelines_not_importable(svc, studio):
    seed_model(studio, pipeline="unknown_pipeline")
    (model,) = svc.get("/v1/connectors/studio/models", params={"project": "p1"}).json()[
        "models"
    ]
    assert model["importable"] is False


def test_list_marks_yolo_detector_importable(svc, studio):
    seed_model(studio, pipeline="yolo_detector", task="detection")
    (model,) = svc.get("/v1/connectors/studio/models", params={"project": "p1"}).json()[
        "models"
    ]
    assert model["pipeline"] == "yolo_detector"
    assert model["importable"] is True


def test_list_requires_configured_studio(settings, registry):
    with make_service_client(settings, registry) as client:
        r = client.get("/v1/connectors/studio/models", params={"project": "p1"})
    assert r.status_code == 409


def test_list_studio_failure_is_502(svc, studio):
    studio.fail = True
    r = svc.get("/v1/connectors/studio/models", params={"project": "p1"})
    assert r.status_code == 502


def test_list_studio_projects_by_name(svc, studio):
    """Users pick a Studio project by name; the numeric id stays the value sent back."""
    studio.extra_projects = [{"id": "2", "name": "Amazon review test"}]
    r = svc.get("/v1/connectors/studio/projects")
    assert r.status_code == 200, r.text
    assert r.json()["projects"] == [
        {"id": "2", "name": "Amazon review test"},
        {"id": "p1", "name": "Project One"},
    ]


def test_projects_requires_configured_studio(settings, registry):
    with make_service_client(settings, registry) as client:
        assert client.get("/v1/connectors/studio/projects").status_code == 409


def test_projects_studio_failure_is_502(svc, studio):
    studio.fail = True
    assert svc.get("/v1/connectors/studio/projects").status_code == 502


def test_projects_malformed_list_is_502(svc, studio):
    studio.extra_projects = [{"name": "no id"}]
    assert svc.get("/v1/connectors/studio/projects").status_code == 502


# --------------------------------------------------------------------------- import


def test_import_stores_base_model_and_is_idempotent(svc, studio, settings):
    from sparrow_finetune.storage import LocalObjectStore

    artifact = conv_zip()
    sid = seed_model(studio, artifact)
    r = import_model(svc, sid)
    assert r.status_code == 201, r.text
    row = r.json()
    assert row["studio_project_id"] == "p1"
    assert row["studio_import_id"] == sid
    assert row["source_model_id"] == "7"
    assert row["pipeline"] == "conventional_classifier"
    assert row["model_arch"] == "resnet18"
    assert row["labels"] == LABELS
    assert row["sha256"] == sha(artifact)
    assert row["size_bytes"] == len(artifact)
    assert "object_key" not in row
    bundle_get = [q for q in studio.requests if q.url.path.endswith("/bundle")]
    assert [q.url.path for q in bundle_get] == [
        f"{API}/projects/p1/models/{sid}/bundle"
    ]

    again = import_model(svc, sid)
    assert again.status_code == 200, again.text
    assert again.json()["id"] == row["id"]
    (listed,) = local_base_models(svc)
    assert listed["id"] == row["id"] and listed["model_arch"] == "resnet18"

    from .service_helpers import db

    with db(settings) as conn:
        key = conn.execute(
            "SELECT object_key FROM base_models WHERE id = %s", (row["id"],)
        ).fetchone()["object_key"]
    assert LocalObjectStore(settings.objects_dir).path(key).read_bytes() == artifact


def test_import_is_audited(svc, studio):
    sid = seed_model(studio)
    assert import_model(svc, sid).status_code == 201
    rows = svc.get("/v1/audit").json()
    rows = rows if isinstance(rows, list) else rows.get("events") or rows.get("audit")
    assert [r for r in rows if r["action"] == "studio.base_model_import"]


@pytest.mark.parametrize(
    "case, status, needle",
    [
        ("other_pipeline", 422, "conventional_classifier"),
        ("no_weights", 422, "re-train"),
        ("wrong_schema", 422, "schema"),
        ("wrong_source", 422, "sparrow-fine-tune"),
        ("manifest_sha_mismatch", 502, "sha256"),
        ("header_sha_mismatch", 502, "sha256"),
        ("absolute_member", 502, "unsafe"),
        ("dotdot_member", 502, "unsafe"),
        ("unexpected_member", 502, "unexpected"),
        ("symlink_member", 502, "unsafe"),
        ("not_a_tar", 502, "bundle"),
        ("zip_weights_sha_mismatch", 502, "weights.pt"),
    ],
)
def test_import_rejects_bad_bundles(svc, studio, settings, case, status, needle):
    artifact = conv_zip()
    good_manifest = studio_manifest(artifact)
    man_bytes = json.dumps(good_manifest).encode()
    if case == "other_pipeline":
        sid = seed_model(
            studio, conv_zip(pipeline="unknown_pipeline"), pipeline="unknown_pipeline"
        )
    elif case == "no_weights":
        sid = seed_model(studio, conv_zip(weights=False))
    elif case == "wrong_schema":
        sid = seed_model(studio, schema="something-else/v9")
    elif case == "wrong_source":
        sid = seed_model(studio, source="another-tool")
    elif case == "manifest_sha_mismatch":
        bad = studio_manifest(artifact, sha256="0" * 64)
        sid = studio.add_model(
            bad,
            tar_bundle(
                [("model.zip", artifact), ("manifest.json", json.dumps(bad).encode())]
            ),
            sha256=sha(artifact),
        )
    elif case == "header_sha_mismatch":
        sid = studio.add_model(
            good_manifest,
            tar_bundle([("model.zip", artifact), ("manifest.json", man_bytes)]),
            sha256="f" * 64,
        )
    elif case == "absolute_member":
        sid = seed_model(
            studio,
            artifact,
            members=[("/etc/model.zip", artifact), ("manifest.json", man_bytes)],
        )
    elif case == "dotdot_member":
        sid = seed_model(
            studio,
            artifact,
            members=[("../model.zip", artifact), ("manifest.json", man_bytes)],
        )
    elif case == "unexpected_member":
        sid = seed_model(
            studio,
            artifact,
            members=[
                ("model.zip", artifact),
                ("manifest.json", man_bytes),
                ("run.sh", b"#!/bin/sh"),
            ],
        )
    elif case == "symlink_member":
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            link = tarfile.TarInfo("model.zip")
            link.type = tarfile.SYMTYPE
            link.linkname = "/etc/passwd"
            tar.addfile(link)
            info = tarfile.TarInfo("manifest.json")
            info.size = len(man_bytes)
            tar.addfile(info, io.BytesIO(man_bytes))
        sid = studio.add_model(good_manifest, buf.getvalue())
    elif case == "not_a_tar":
        sid = studio.add_model(good_manifest, b"definitely not a tar.gz")
    elif case == "zip_weights_sha_mismatch":
        buf = io.BytesIO()
        with (
            zipfile.ZipFile(io.BytesIO(artifact)) as src,
            zipfile.ZipFile(buf, "w") as dst,
        ):
            for name in src.namelist():
                data = src.read(name)
                if name == "weights.pt":
                    data = data + b"tampered"
                dst.writestr(name, data)
        sid = seed_model(studio, buf.getvalue())
    else:  # pragma: no cover
        raise AssertionError(case)
    r = import_model(svc, sid)
    assert r.status_code == status, r.text
    assert needle in r.json()["detail"], r.text
    assert local_base_models(svc) == []


def test_import_of_another_projects_model_fails(svc, studio):
    sid = studio.add_model(
        studio_manifest(conv_zip()), tar_bundle([]), project="other-project"
    )
    r = import_model(svc, sid)
    assert r.status_code == 502, r.text
    assert local_base_models(svc) == []


def test_import_requires_configured_studio(settings, registry):
    with make_service_client(settings, registry) as client:
        assert import_model(client, "studio-model-1").status_code == 409


# --------------------------------------------------------------------------- submit validation


def _conv_snapshot(client) -> int:
    return upload_snapshot(
        client, balanced_items(train_groups=2, val_groups=1, prefix="bm-")
    )


def test_submit_rejects_unknown_base_model(conv_svc):
    client, _s, _r = conv_svc
    snap = _conv_snapshot(client)
    r = client.post(
        "/v1/jobs",
        json={
            "pipeline": "conventional_classifier",
            "snapshot_id": snap,
            "params": {**SMOKE, "base_model_id": 999999},
        },
    )
    assert r.status_code == 422, r.text
    assert "base model 999999" in r.json()["detail"]


def test_submit_rejects_architecture_mismatch(conv_svc, studio):
    client, _s, _r = conv_svc
    base = import_model(client, seed_model(studio)).json()
    snap = _conv_snapshot(client)
    r = client.post(
        "/v1/jobs",
        json={
            "pipeline": "conventional_classifier",
            "snapshot_id": snap,
            "params": {**SMOKE, "model": "resnet34", "base_model_id": base["id"]},
        },
    )
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert "resnet18" in detail and "resnet34" in detail


def test_submit_accepts_matching_base_model(conv_svc, studio):
    client, _s, _r = conv_svc
    base = import_model(client, seed_model(studio)).json()
    job_id = submit_conventional(
        client, _conv_snapshot(client), {**SMOKE, "base_model_id": base["id"]}
    )
    assert (
        client.get(f"/v1/jobs/{job_id}").json()["params"]["base_model_id"] == base["id"]
    )


def test_submit_rejects_base_model_for_trainer_without_warm_start(svc, studio):
    from .helpers import build_snapshot

    snap = build_snapshot(svc)["snapshot"]
    r = svc.post(
        "/v1/jobs",
        json={
            "pipeline": "fake_cls",
            "snapshot_id": snap["id"],
            "params": {"base_model_id": 1},
        },
    )
    assert r.status_code == 422, r.text
    assert "base_model_id" in r.json()["detail"]


def test_conventional_base_model_id_type_is_checked():
    t = trainer()
    assert t.validate_params({})["base_model_id"] is None
    assert t.validate_params({"base_model_id": 3})["base_model_id"] == 3
    for bad in ("3", 0, -1, True, 1.5):
        with pytest.raises(ValueError, match="base_model_id"):
            t.validate_params({"base_model_id": bad})


def test_dinov3_base_model_id_type_is_checked():
    from sparrow_finetune.pipelines.dinov3_linear import DinoV3LinearTrainer

    t = DinoV3LinearTrainer.__new__(DinoV3LinearTrainer)  # validate_params needs no constructor state
    assert "base_model_id" not in t.validate_params({})
    assert t.validate_params({"base_model_id": 3})["base_model_id"] == 3
    for bad in ("3", 0, -1, True, 1.5):
        with pytest.raises(ValueError, match="base_model_id"):
            t.validate_params({"base_model_id": bad})


# --------------------------------------------------------------------------- push scope


def test_push_targets_only_the_project(svc, studio, settings, registry):
    """Pushed models stay private to the Studio project: every request of a push is
    under ``/projects/<project>/``; nothing goes to a shared catalogue."""
    m = trained_model(svc, settings, registry)
    r = svc.post(
        "/v1/connectors/studio/push", json={"model_id": m["id"], "project": "p1"}
    )
    assert r.status_code in (200, 201), r.text
    assert studio.requests
    for q in studio.requests:
        assert q.url.path.startswith(f"{API}/projects/p1/"), q.url.path


# --------------------------------------------------------------------------- warm start (trainer)


@pytest.fixture(scope="module")
def base_bundle(tmp_path_factory):
    """A real exported conventional bundle to warm start from."""
    root = tmp_path_factory.mktemp("base")
    syn = balanced_items(train_groups=2, val_groups=1, prefix="base-")
    items = snapshot_items(syn, root / "images")
    t = trainer()
    ctx, _rec = make_context(t, items, root / "work", dict(WARM))
    _metrics, result = run_stages(t, ctx)
    return {"path": Path(result.path), "labels": list(ctx.labels), "items": items}


def _warm_ctx(base_bundle, tmp_path, labels=None):
    t = trainer()
    ctx, rec = make_context(
        t,
        base_bundle["items"],
        tmp_path / "work",
        {**SMOKE, "epochs": 1, "warmup_epochs": 1, "base_model_id": 5},
    )
    ctx.base_model = {
        "id": 5,
        "model_arch": SMOKE["model"],
        "labels": list(base_bundle["labels"] if labels is None else labels),
        "path": base_bundle["path"],
        "sha256": sha(base_bundle["path"].read_bytes()),
    }
    return t, ctx, rec


def _base_state(base_bundle):
    import torch

    with zipfile.ZipFile(base_bundle["path"]) as zf:
        return torch.load(io.BytesIO(zf.read("weights.pt")), weights_only=True)


def _warm_messages(rec) -> list[str]:
    return [msg for _stage, msg, _m in rec.events if "warm start" in msg]


def test_warm_start_same_labels_loads_backbone_and_head(base_bundle, tmp_path):
    import torch

    t, ctx, rec = _warm_ctx(base_bundle, tmp_path)
    t.prepare(ctx)
    t.train(ctx)
    (msg,) = _warm_messages(rec)
    assert "base model 5" in msg and "head loaded" in msg
    # One warmup epoch freezes the backbone (eval-mode BatchNorm too): it must still
    # equal the base weights exactly.
    base = _base_state(base_bundle)
    best = torch.load(Path(ctx.workdir) / "best.pt", weights_only=True)
    for key, value in base.items():
        if not key.startswith("fc."):
            assert torch.equal(best[key], value), key


def test_warm_start_different_labels_reinitialises_head(base_bundle, tmp_path):
    import torch

    t, ctx, rec = _warm_ctx(
        base_bundle, tmp_path, labels=list(reversed(base_bundle["labels"]))
    )
    t.prepare(ctx)
    t.train(ctx)
    (msg,) = _warm_messages(rec)
    assert "re-initialised" in msg
    base = _base_state(base_bundle)
    best = torch.load(Path(ctx.workdir) / "best.pt", weights_only=True)
    for key, value in base.items():
        if not key.startswith("fc."):
            assert torch.equal(best[key], value), key


def test_warm_start_rejects_bundle_without_weights(tmp_path):
    t = trainer()
    path = tmp_path / "old.zip"
    path.write_bytes(conv_zip(weights=False))
    syn = balanced_items(train_groups=2, val_groups=1, prefix="old-")
    ctx, _rec = make_context(
        t,
        snapshot_items(syn, tmp_path / "images"),
        tmp_path / "work",
        {**SMOKE, "base_model_id": 9},
    )
    ctx.base_model = {
        "id": 9,
        "model_arch": "resnet18",
        "labels": LABELS,
        "path": path,
        "sha256": sha(path.read_bytes()),
    }
    with pytest.raises(ValueError, match="re-train"):
        t.prepare(ctx)


# --------------------------------------------------------------------------- end to end


def test_end_to_end_push_list_import_warm_start(conv_svc, studio):
    """Train -> push to Studio -> list -> import -> new job warm-started from it."""
    client, s, registry = conv_svc
    syn = balanced_items(train_groups=2, val_groups=1, prefix="e2e-")
    snap = upload_snapshot(client, syn)

    first = submit_conventional(client, snap, dict(WARM))
    assert run_once(s, "w1", {"cpu"}, registry=registry) is True
    assert client.get(f"/v1/jobs/{first}").json()["state"] == "succeeded"
    (model,) = models_for_job(client, first)

    pushed = client.post(
        "/v1/connectors/studio/push", json={"model_id": model["id"], "project": "p1"}
    )
    assert pushed.status_code == 200, pushed.text
    studio_id = pushed.json()["studio_model_id"]

    listed = client.get("/v1/connectors/studio/models", params={"project": "p1"})
    assert listed.status_code == 200, listed.text
    (remote,) = listed.json()["models"]
    assert remote["id"] == studio_id
    assert str(remote["source_model_id"]) == str(model["id"])
    assert remote["importable"] is True

    imported = import_model(client, studio_id)
    assert imported.status_code == 201, imported.text
    base = imported.json()
    assert base["sha256"] == model["sha256"]
    assert base["labels"] == LABELS
    assert [b["id"] for b in local_base_models(client)] == [base["id"]]

    second = submit_conventional(client, snap, {**WARM, "base_model_id": base["id"]})
    assert run_once(s, "w2", {"cpu"}, registry=registry) is True
    job = client.get(f"/v1/jobs/{second}").json()
    assert job["state"] == "succeeded", job
    messages = [e.get("message") or "" for e in job_events(client, second)]
    warm = [m for m in messages if "warm start" in m]
    assert len(warm) == 1 and "head loaded" in warm[0], messages
    (second_model,) = models_for_job(client, second)
    assert second_model["metadata"].get("base_model_id") == base["id"]


# --------------------------------------------------------------------------- UI


def test_ui_has_base_model_picker_and_studio_import():
    html = (REPO / "src/sparrow_finetune/ui/index.html").read_text()
    js = (REPO / "src/sparrow_finetune/ui/app.js").read_text()
    assert 'id="base-model-info"' in html
    assert 'id="studio-model-select"' in html
    assert 'id="studio-model-clear"' in html
    assert "/v1/connectors/studio/models" in js
    assert "params.base_model_id = state.baseModel.row.id" in js
    # The import route takes studio_model_id as a string; a Number() cast got a 422.
    assert "studio_model_id: studioModelId," in js
