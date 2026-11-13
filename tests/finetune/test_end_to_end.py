"""AC18: one client journey end to end -- HTTP for every user step, ``run_once``
for the worker, sharing one ``Registry`` object between the API and the worker.

This is the test that proves the pieces agree with each other (dataset content
reaches the trainer as local files; the trainer's export comes back byte-exact).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image

from sparrow_finetune.workers import run_once

from .fakes import EVAL_METRICS
from .helpers import (
    IMAGE_SIZE,
    STAGES,
    annotation_item,
    as_list,
    field,
    get_job,
    png_bytes,
    same_id,
    stage_sequence,
)

LABELS = ["cat", "dog", "cat", "fox"]
GROUPS = ["train", "train", "val", "val"]


def test_ac18_end_to_end_http_and_worker(client, settings, registry, trainers):
    # workspace -> dataset
    r = client.post("/v1/workspaces", json={"name": "e2e"})
    assert r.status_code == 201, r.text
    ws_id = r.json()["id"]
    r = client.post(f"/v1/workspaces/{ws_id}/datasets", json={"name": "e2e-ds", "task": "classification", "whole_image": True})
    assert r.status_code == 201, r.text
    ds_id = r.json()["id"]

    # 4 images
    uploaded = {}
    for i in range(4):
        name = f"e2e{i}.png"
        r = client.put(f"/v1/datasets/{ds_id}/images/{name}", content=png_bytes(100 + i))
        assert r.is_success, r.text
        uploaded[name] = r.json()
        assert uploaded[name]["sha256"] == hashlib.sha256(png_bytes(100 + i)).hexdigest()

    # annotations
    items = [annotation_item(f"e2e{i}.png", LABELS[i], group=GROUPS[i]) for i in range(4)]
    r = client.post(f"/v1/datasets/{ds_id}/annotations", json={"items": items})
    assert r.status_code == 200, r.text
    assert r.json() == {"imported": 4}

    # snapshot
    r = client.post(f"/v1/datasets/{ds_id}/snapshots")
    assert r.status_code == 201, r.text
    snap = r.json()
    assert snap["item_count"] == 4
    assert snap["label_counts"] == {"cat": 2, "dog": 1, "fox": 1}

    # job with the test-registered trainer
    r = client.post("/v1/jobs", json={"pipeline": "fake_cls", "snapshot_id": snap["id"], "params": {}})
    assert r.status_code == 201, r.text
    assert r.json()["state"] == "queued"
    job_id = r.json()["id"]

    # worker, same Registry object as the API
    assert run_once(settings, "e2e-worker", {"cpu"}, registry=registry) is True
    assert get_job(client, job_id)["state"] == "succeeded"
    assert stage_sequence(as_list(client.get(f"/v1/jobs/{job_id}/events").json(), "events")) == STAGES

    # the trainer received the snapshot as local image files with their labels/groups
    assert set(trainers.basic.labels) == {"cat", "dog", "fox"}
    assert len(trainers.basic.items) == 4
    seen = []
    for item in trainers.basic.items:
        path = Path(field(item, "image_path"))
        assert path.is_file()
        with Image.open(path) as im:
            assert im.size == IMAGE_SIZE
        assert (field(item, "width"), field(item, "height")) == IMAGE_SIZE
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        labels = [field(a, "label") for a in field(item, "annotations")]
        seen.append((digest, tuple(labels), field(item, "group")))
    expected = sorted(
        (uploaded[f"e2e{i}.png"]["sha256"], (LABELS[i],), GROUPS[i]) for i in range(4)
    )
    assert sorted(seen) == expected

    # model version -> artifact download, hash-verified
    r = client.get("/v1/models")
    assert r.status_code == 200, r.text
    (model,) = [m for m in as_list(r.json(), "models") if same_id(m["job_id"], job_id)]
    assert model["metrics"] == EVAL_METRICS
    art = client.get(f"/v1/models/{model['id']}/artifact")
    assert art.status_code == 200
    assert hashlib.sha256(art.content).hexdigest() == model["sha256"]
    assert art.content == trainers.basic.exported
