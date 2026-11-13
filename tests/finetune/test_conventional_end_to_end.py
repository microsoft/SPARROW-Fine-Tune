"""Phase 3 end to end (C10): ``conventional_classifier`` through HTTP + ``run_once`` on real PostgreSQL."""

from __future__ import annotations

import hashlib
import time

import numpy as np

from .dinov3_helpers import (
    SynItem,
    balanced_items,
    bundle_json,
    read_bundle_bytes,
    synthetic_items,
    synthetic_png,
)
from .helpers import as_list, find_key, get_job, job_events, make_client, models_for_job
from .test_conventional_classifier import (
    SMOKE,
    onnx_inputs,
    onnx_probabilities,
    val_side,
)
from .test_dinov3_end_to_end import builtin_registry, upload_snapshot


def submit(client, snapshot_id, params) -> int:
    r = client.post(
        "/v1/jobs",
        json={
            "pipeline": "conventional_classifier",
            "snapshot_id": snapshot_id,
            "params": params,
        },
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


def test_c10_end_to_end_conventional_classifier_job(settings):
    from sparrow_finetune.workers import run_once

    registry = builtin_registry(settings)
    syn = balanced_items(
        train_groups=4,
        val_groups=2,
        images_per_group=2,
        ungrouped_per_label=1,
        prefix="cc-",
    )

    with make_client(settings, registry) as client:
        pipelines = {
            p["id"]: p for p in as_list(client.get("/v1/pipelines").json(), "pipelines")
        }
        assert pipelines["conventional_classifier"]["status"] == "available"

        job_id = submit(client, upload_snapshot(client, syn), dict(SMOKE))
        assert run_once(settings, "cc-worker", {"cpu"}, registry=registry) is True
        job = get_job(client, job_id)
        assert job["state"] == "succeeded", job

        (model,) = models_for_job(client, job_id)
        art = client.get(f"/v1/models/{model['id']}/artifact")
        assert art.status_code == 200
        assert hashlib.sha256(art.content).hexdigest() == model["sha256"]
        assert model["format"] == "onnx-bundle"
        assert model["metrics"]["val_macro_f1"] >= 0.9

    bundle = read_bundle_bytes(art.content)
    labels = bundle_json(bundle, "manifest.json")["labels"]
    val = val_side(syn)
    assert val
    probs = onnx_probabilities(bundle["model.onnx"], onnx_inputs(val, SMOKE["imgsz"]))
    assert [labels[i] for i in np.asarray(probs).argmax(axis=1)] == [
        it.label for it in val
    ]


def test_c10_invalid_params_rejected_at_submit(settings):
    registry = builtin_registry(settings)
    with make_client(settings, registry) as client:
        snap = upload_snapshot(
            client, synthetic_items(groups_per_label=2, prefix="cc-bad-")
        )
        r = client.post(
            "/v1/jobs",
            json={
                "pipeline": "conventional_classifier",
                "snapshot_id": snap,
                "params": {"model": "/etc/passwd"},
            },
        )
        assert r.status_code == 422, r.text
        assert "model" in r.text


def test_c10_job_fails_when_label_below_class_floor(settings):
    from sparrow_finetune.workers import run_once

    registry = builtin_registry(settings)
    syn = synthetic_items(groups_per_label=4, prefix="cc-floor-")
    syn.append(
        SynItem(
            "cc-floor-rare-0.png",
            "rare",
            "cc-floor-rare",
            None,
            synthetic_png("cat", 777, None),
        )
    )
    with make_client(settings, registry) as client:
        job_id = submit(client, upload_snapshot(client, syn), dict(SMOKE))
        assert run_once(settings, "cc-worker", {"cpu"}, registry=registry) is True
        job = get_job(client, job_id)
        assert job["state"] == "failed", job
        assert "rare" in str(job["error"])
        assert models_for_job(client, job_id) == []


def test_c10_cancel_during_train_cancels_job_without_model(settings):
    from sparrow_finetune.jobs import JobQueue
    from sparrow_finetune.pipelines import JobCancelled, Registry
    from sparrow_finetune.pipelines.conventional_classifier import (
        ConventionalClassifierTrainer,
    )
    from sparrow_finetune.workers import run_once

    class CancelRequestedAtTrainStart(ConventionalClassifierTrainer):
        def train(self, ctx):
            JobQueue(settings.database_url).request_cancel(ctx.job_id)
            deadline = time.monotonic() + 15
            while True:
                try:
                    ctx.check_cancelled()
                except JobCancelled:
                    break
                assert time.monotonic() < deadline, (
                    "worker never observed the cancel request"
                )
                time.sleep(0.05)
            return super().train(ctx)

    registry = Registry()
    registry.register(CancelRequestedAtTrainStart())
    syn = synthetic_items(groups_per_label=4, prefix="cc-cancel-")
    with make_client(settings, registry) as client:
        job_id = submit(client, upload_snapshot(client, syn), dict(SMOKE, epochs=50))
        assert (
            run_once(
                settings, "cc-worker", {"cpu"}, registry=registry, lease_seconds=0.9
            )
            is True
        )
        assert get_job(client, job_id)["state"] == "cancelled"
        assert models_for_job(client, job_id) == []
        epochs = [
            e
            for e in job_events(client, job_id)
            if e.get("stage") == "train" and find_key(e, "epoch")
        ]
        assert len(epochs) <= 1
