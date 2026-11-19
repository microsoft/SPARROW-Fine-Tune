"""Local base models: a ResNet-50 package from ``sparrow-uploader prepare-finetune``.

The flow: a user (or an agent) runs ``sparrow-uploader prepare-finetune`` on a timm
``resnet50`` state dict, uploads the ZIP with ``POST /v1/base-models/local`` (multipart
field ``file``), picks it in the training form, and a ``conventional_classifier`` job
warm-starts from it. No Sparrow Studio is involved, so this works on a standalone service.

Package contract (``schema=sparrow-finetune/base-v1``), validated here independently of the
uploader: exactly ``manifest.json`` + ``weights.safetensors``, ZIP_STORED; manifest
``pipeline=conventional_classifier``, ``model_arch=resnet50``, ``labels`` (distinct strings,
one per row of the classifier head), ``files["weights.safetensors"]`` = sha256.

Pinned here:

- upload stores a row whose origin is ``local`` with no Studio project / import / source
  ids (nothing fabricated), listed by ``GET /v1/base-models``; re-upload is idempotent;
- malformed, tampered, unsupported (another pipeline or architecture) or mismatched
  packages are refused with 422 and nothing is stored;
- ``weights.safetensors`` is never unpickled: a pickle payload in it must not run;
- a job of another architecture is refused; a resnet50 job warm-starts with the uploaded
  backbone tensors exactly (trainer level) and runs through the worker (end to end);
- the training UI of a service without Studio can upload and select a local base model;
- opt-in release proof: set ``SPARROW_UPLOADER_CMD`` to an uploader invocation (for example
  ``uv run --project ../sparrow-model-uploader sparrow-uploader``) to run the real CLI ->
  import -> trainer chain; skipped when unset.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import pickle
import shlex
import socket
import subprocess
import threading
import time
import zipfile
from pathlib import Path

import pytest

from sparrow_finetune.workers import run_once

from .dinov3_helpers import balanced_items, make_context, snapshot_items
from .helpers import job_events, models_for_job
from .service_helpers import PASSWORD, add_user, make_service_client
from .test_conventional_classifier import SMOKE, trainer
from .test_conventional_end_to_end import submit as submit_conventional
from .test_dinov3_end_to_end import builtin_registry, upload_snapshot

SCHEMA = "sparrow-finetune/base-v1"
# The labels balanced_items produces, in the order a job sees them: a warm start keeps the head.
LABELS = ["cat", "dog", "fox"]
R50 = {**SMOKE, "model": "resnet50", "epochs": 1, "warmup_epochs": 1}
ROUTE = "/v1/base-models/local"


# --------------------------------------------------------------------------- builders


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def resnet_state(arch: str = "resnet50", classes: int = 3, seed: int = 0) -> dict:
    import timm
    import torch

    torch.manual_seed(seed)
    model = timm.create_model(arch, pretrained=False, num_classes=classes)
    return {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}


def st_bytes(state: dict) -> bytes:
    from safetensors.torch import save

    return save(state)


def package(
    weights: bytes,
    labels=LABELS,
    *,
    manifest_bytes: bytes | None = None,
    drop: tuple = (),
    extra: tuple = (),
    compression: int = zipfile.ZIP_STORED,
    **over,
) -> bytes:
    manifest = {
        "schema": SCHEMA,
        "pipeline": "conventional_classifier",
        "model_arch": "resnet50",
        "labels": list(labels),
        "files": {"weights.safetensors": sha(weights)},
    }
    manifest.update(over)
    members = [
        (
            "manifest.json",
            json.dumps(manifest).encode() if manifest_bytes is None else manifest_bytes,
        ),
        ("weights.safetensors", weights),
        *extra,
    ]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=compression) as zf:
        for name, data in members:
            if name not in drop:
                zf.writestr(name, data)
    return buf.getvalue()


def upload(client, data: bytes, filename: str = "resnet50-package.zip"):
    return client.post(ROUTE, files={"file": (filename, data, "application/zip")})


def uploaded(client, data: bytes) -> dict:
    r = upload(client, data)
    assert r.status_code == 201, r.text
    return r.json()


def base_models(client) -> list:
    r = client.get("/v1/base-models")
    assert r.status_code == 200, r.text
    return r.json()["base_models"]


def stored_objects(settings) -> list:
    root = Path(settings.objects_dir) / "base_models"
    return sorted(p for p in root.rglob("*") if p.is_file()) if root.exists() else []


class _Plant:
    """Unpickling this creates ``marker``: proof that the service executed pickle code."""

    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        return (open, (str(self.marker), "w"))


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def r50() -> dict:
    return resnet_state("resnet50", classes=len(LABELS))


@pytest.fixture(scope="module")
def r50_weights(r50) -> bytes:
    return st_bytes(r50)


@pytest.fixture(scope="module")
def r50_package(r50_weights) -> bytes:
    return package(r50_weights)


@pytest.fixture(scope="module")
def r18_weights() -> bytes:
    return st_bytes(resnet_state("resnet18", classes=len(LABELS)))


@pytest.fixture
def local_svc(settings):
    """A standalone service (no Studio configured) with the built-in pipelines."""
    registry = builtin_registry(settings)
    with make_service_client(settings, registry) as client:
        yield client, settings, registry


def _snapshot(client) -> int:
    return upload_snapshot(
        client, balanced_items(train_groups=2, val_groups=1, prefix="lb-")
    )


# --------------------------------------------------------------------------- upload


def test_local_upload_stores_a_resnet50_base_model_without_studio(
    local_svc, r50_package
):
    client, settings, _r = local_svc
    r = upload(client, r50_package)
    assert r.status_code == 201, r.text
    row = r.json()
    assert row["origin"] == "local"
    # A local model has no Studio provenance; none may be invented.
    assert row.get("studio_project_id") is None
    assert row.get("studio_import_id") is None
    assert row.get("source_model_id") is None
    assert row["pipeline"] == "conventional_classifier"
    assert row["model_arch"] == "resnet50"
    assert row["labels"] == LABELS
    assert row["sha256"] == sha(r50_package)
    assert row["size_bytes"] == len(r50_package)
    assert "object_key" not in row
    (listed,) = base_models(client)
    assert listed["id"] == row["id"] and listed["origin"] == "local"
    assert len(stored_objects(settings)) == 1


def test_local_upload_is_idempotent(local_svc, r50_package):
    client, settings, _r = local_svc
    first = upload(client, r50_package)
    assert first.status_code == 201, first.text
    again = upload(client, r50_package)
    assert again.status_code in (200, 201), again.text
    assert again.json()["id"] == first.json()["id"]
    assert [b["id"] for b in base_models(client)] == [first.json()["id"]]
    assert len(stored_objects(settings)) == 1


def test_studio_import_reports_studio_origin(settings, registry):
    """The origin tells the two kinds apart; a Studio row keeps its Studio ids."""
    from .service_helpers import studio_settings
    from .studio_fake import FakeStudio
    from .test_base_models import import_model, seed_model

    studio = FakeStudio(project="p1")
    with make_service_client(
        studio_settings(settings), registry, studio.transport
    ) as client:
        sid = seed_model(studio)
        row = import_model(client, sid).json()
    assert row["origin"] == "studio"
    assert row["studio_project_id"] == "p1" and row["studio_import_id"] == sid


def _bad_package(case: str, w: bytes, request) -> bytes:
    if case == "empty_file":
        return b""
    if case == "not_a_zip":
        return b"PK not really a zip"
    if case == "deflated":
        return package(w, compression=zipfile.ZIP_DEFLATED)
    if case == "extra_member":
        return package(w, extra=(("model.onnx", b"onnx"),))
    if case == "unsafe_member":
        return package(w, extra=(("../escape.txt", b"x"),))
    if case == "duplicate_member":
        return package(w, extra=(("manifest.json", b"{}"),))
    if case == "no_manifest":
        return package(w, drop=("manifest.json",))
    if case == "no_weights":
        return package(w, drop=("weights.safetensors",))
    if case == "manifest_not_json":
        return package(w, manifest_bytes=b"{not json")
    if case == "wrong_schema":
        return package(w, schema="sparrow-finetune/model-v1")
    if case == "wrong_pipeline":
        return package(w, pipeline="yolo_detector")
    if case == "sha_mismatch":
        return package(w, files={"weights.safetensors": "0" * 64})
    if case == "labels_do_not_fit_head":
        return package(w, labels=[*LABELS, "okapi"])
    if case == "duplicate_labels":
        return package(w, labels=["cat", "cat", "fox"])
    if case == "non_string_labels":
        return package(w, labels=["cat", 1, "fox"])
    if case == "empty_labels":
        return package(w, labels=[])
    if case == "not_safetensors":
        junk = b"this is not a safetensors file"
        return package(junk)
    r18 = request.getfixturevalue("r18_weights")
    if case == "unsupported_arch":
        return package(r18, model_arch="resnet18")
    if case == "weights_not_resnet50":
        return package(r18)  # manifest claims resnet50
    raise AssertionError(case)  # pragma: no cover - parametrize typo


@pytest.mark.parametrize(
    "case",
    [
        "empty_file",
        "not_a_zip",
        "deflated",
        "extra_member",
        "unsafe_member",
        "duplicate_member",
        "no_manifest",
        "no_weights",
        "manifest_not_json",
        "wrong_schema",
        "wrong_pipeline",
        "sha_mismatch",
        "labels_do_not_fit_head",
        "duplicate_labels",
        "non_string_labels",
        "empty_labels",
        "not_safetensors",
        "unsupported_arch",
        "weights_not_resnet50",
    ],
)
def test_local_upload_rejects_bad_packages_and_stores_nothing(
    case, local_svc, r50_weights, request
):
    client, settings, _r = local_svc
    r = upload(client, _bad_package(case, r50_weights, request))
    assert r.status_code == 422, (case, r.status_code, r.text)
    detail = r.json().get("detail")
    assert detail, r.text
    if case == "sha_mismatch":
        assert "sha256" in str(detail)
    assert base_models(client) == []
    assert stored_objects(settings) == []


def test_local_upload_requires_the_file_field(local_svc, r50_package):
    client, _s, _r = local_svc
    r = client.post(ROUTE, files={"package": ("p.zip", r50_package, "application/zip")})
    assert r.status_code == 422, r.text
    assert base_models(client) == []


@pytest.mark.parametrize("container", ["raw_pickle", "torch_checkpoint"])
def test_local_upload_never_unpickles_weights(container, local_svc, tmp_path):
    import torch

    client, settings, _r = local_svc
    marker = tmp_path / "PWNED"
    payload = {"fc.weight": _Plant(marker)}
    if container == "raw_pickle":
        weights = pickle.dumps(payload)
    else:
        buf = io.BytesIO()
        torch.save(payload, buf)
        weights = buf.getvalue()
    r = upload(client, package(weights))  # the manifest sha256 matches the payload
    assert not marker.exists(), "the service unpickled an uploaded weights file"
    assert r.status_code == 422, r.text
    assert base_models(client) == []
    assert stored_objects(settings) == []


# --------------------------------------------------------------------------- jobs


def test_submit_accepts_resnet50_job_with_local_base_model(local_svc, r50_package):
    client, _s, _r = local_svc
    base = uploaded(client, r50_package)
    job_id = submit_conventional(
        client, _snapshot(client), {**R50, "base_model_id": base["id"]}
    )
    assert (
        client.get(f"/v1/jobs/{job_id}").json()["params"]["base_model_id"] == base["id"]
    )


def test_submit_rejects_other_architecture_for_local_base_model(local_svc, r50_package):
    client, _s, _r = local_svc
    base = uploaded(client, r50_package)
    r = client.post(
        "/v1/jobs",
        json={
            "pipeline": "conventional_classifier",
            "snapshot_id": _snapshot(client),
            "params": {**SMOKE, "model": "resnet18", "base_model_id": base["id"]},
        },
    )
    assert r.status_code == 422, r.text
    assert "resnet50" in r.json()["detail"] and "resnet18" in r.json()["detail"]


def test_local_base_model_warm_starts_with_the_uploaded_backbone(
    local_svc, r50, r50_package, tmp_path
):
    """Trainer level: the staged local package loads; one frozen warmup epoch keeps the
    backbone equal to the uploaded tensors, and the same labels keep the head."""
    import torch

    from sparrow_finetune import base_models as bm

    client, settings, _r = local_svc
    base = uploaded(client, r50_package)
    t = trainer()
    items = snapshot_items(
        balanced_items(train_groups=2, val_groups=1, prefix="ws-"), tmp_path / "images"
    )
    ctx, rec = make_context(
        t, items, tmp_path / "work", {**R50, "base_model_id": base["id"]}
    )
    assert list(ctx.labels) == LABELS
    ctx.base_model = bm.stage(settings, base["id"], tmp_path / "work")
    t.prepare(ctx)
    t.train(ctx)
    warm = [msg for _stage, msg, _m in rec.events if "warm start" in msg]
    assert len(warm) == 1 and "head loaded" in warm[0], warm
    best = torch.load(Path(ctx.workdir) / "best.pt", weights_only=True)
    differ = [
        k
        for k, v in r50.items()
        if not k.startswith("fc.") and not torch.equal(best[k], v)
    ]
    assert differ == [], differ[:5]


def _run_warm_job(client, settings, registry, base_id: int) -> dict:
    job_id = submit_conventional(
        client, _snapshot(client), {**R50, "base_model_id": base_id}
    )
    assert run_once(settings, "w-local", {"cpu"}, registry=registry) is True
    job = client.get(f"/v1/jobs/{job_id}").json()
    assert job["state"] == "succeeded", job
    messages = [e.get("message") or "" for e in job_events(client, job_id)]
    warm = [m for m in messages if "warm start" in m]
    assert len(warm) == 1 and "head loaded" in warm[0], messages
    (model,) = models_for_job(client, job_id)
    assert model["metadata"].get("base_model_id") == base_id
    return job


def test_end_to_end_local_upload_then_warm_started_job(local_svc, r50_package):
    client, settings, registry = local_svc
    _run_warm_job(client, settings, registry, uploaded(client, r50_package)["id"])


# --------------------------------------------------------------------------- UI (standalone)


@pytest.fixture
def standalone_server(settings, registry):
    import uvicorn

    from sparrow_finetune.api import create_app

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(create_app(settings, registry=registry), log_level="warning")
    )
    web = threading.Thread(
        target=lambda: asyncio.run(server.serve(sockets=[sock])), daemon=True
    )
    web.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline, "uvicorn did not start"
        time.sleep(0.02)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        web.join(10)


def test_ui_uploads_and_selects_local_base_model_without_studio(
    standalone_server, settings, registry, client, r50_package, tmp_path
):
    """Without Studio the Studio panel is hidden, yet a local package can be uploaded
    (``#local-base-model-file`` + ``#local-base-model-upload``), is shown as the base model
    and is sent as ``base_model_id`` with ``model=resnet50`` when training starts."""
    sync_api = pytest.importorskip("playwright.sync_api")
    from .helpers import build_snapshot
    from .test_backlog_service_ui import _browser, _login
    from .test_studio_picker_ui import _register_classifier

    expect = sync_api.expect
    _register_classifier(registry)
    build_snapshot(client)
    add_user(settings, "alice", PASSWORD)
    pkg = tmp_path / "resnet50-package.zip"
    pkg.write_bytes(r50_package)
    problems: list[str] = []
    with sync_api.sync_playwright() as pw:
        browser, page = _browser(pw, problems)
        try:
            _login(page, standalone_server)
            page.select_option("#pipeline-select", "conventional_classifier")
            expect(page.locator("#studio-panel")).to_be_hidden()
            page.set_input_files("#local-base-model-file", str(pkg))
            page.click("#local-base-model-upload")
            info = page.locator("#base-model-info")
            expect(info).to_contain_text("resnet50", timeout=60_000)
            expect(info).to_contain_text("3 classes")
            expect(info).not_to_contain_text("Studio model")
            (row,) = base_models(client)
            with page.expect_request(
                lambda r: r.method == "POST" and r.url.endswith("/v1/jobs")
            ) as req:
                page.click("#start-training")
            params = req.value.post_data_json["params"]
            assert params["base_model_id"] == row["id"]
            assert params["model"] == "resnet50"
            expect(page.locator("#error-banner")).to_be_hidden()
        finally:
            browser.close()
    assert problems == [], problems


# --------------------------------------------------------------------------- opt-in release proof


@pytest.mark.skipif(
    not os.environ.get("SPARROW_UPLOADER_CMD"),
    reason="release cross-check: set SPARROW_UPLOADER_CMD to a sparrow-uploader invocation",
)
def test_release_cross_check_uploader_cli_to_import_to_trainer(
    local_svc, r50, tmp_path
):
    from safetensors.torch import save_file

    client, settings, registry = local_svc
    weights = tmp_path / "resnet50.safetensors"
    save_file(r50, str(weights))
    labels = tmp_path / "labels.txt"
    labels.write_text("\n".join(LABELS) + "\n", encoding="utf-8")
    out = tmp_path / "package.zip"
    cmd = shlex.split(os.environ["SPARROW_UPLOADER_CMD"]) + [
        "prepare-finetune",
        "--weights",
        str(weights),
        "--labels",
        str(labels),
        "--output",
        str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = out.read_bytes()
    r = upload(client, data)
    assert r.status_code == 201, r.text
    assert r.json()["sha256"] == sha(data) and r.json()["origin"] == "local"
    _run_warm_job(client, settings, registry, r.json()["id"])
