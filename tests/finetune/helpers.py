"""Small helpers shared by the ``sparrow_finetune`` acceptance tests.

Nothing here imports ``sparrow_finetune`` at module level, so ``conftest.py`` can
import this module even before the package exists. Response-shape details that
docs/review/implement-foundation/spec.md leaves open (list wrapping, item objects
vs dicts, where emitted metrics live) are read tolerantly here; everything the spec
does state is asserted exactly in the tests.
"""

from __future__ import annotations

import io
import json
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any, Iterator, Optional, Sequence

from PIL import Image

IMAGE_SIZE = (64, 48)  # width, height
STAGES = ["prepare", "train", "evaluate", "export"]


def png_bytes(seed: int = 0, size: tuple[int, int] = IMAGE_SIZE) -> bytes:
    """A small deterministic PNG; different seeds give different bytes."""
    colour = ((seed * 53) % 256, (seed * 97 + 40) % 256, (seed * 29 + 120) % 256)
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, format="PNG")
    return buf.getvalue()


def as_list(body: Any, *keys: str) -> list:
    """Return a JSON list, accepting either a bare list or an object wrapping one."""
    if isinstance(body, list):
        return body
    if isinstance(body, dict):
        for key in (*keys, "items", "data", "results"):
            value = body.get(key)
            if isinstance(value, list):
                return value
    raise AssertionError(f"expected a JSON list (or an object wrapping one), got {body!r}")


def field(obj: Any, key: str) -> Any:
    """Read ``key`` from a mapping or an attribute-style object."""
    if isinstance(obj, Mapping):
        return obj[key]
    return getattr(obj, key)


def find_key(obj: Any, key: str) -> list:
    """All values stored under ``key`` anywhere inside a JSON-like structure."""
    found = []
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if k == key:
                found.append(v)
            found.extend(find_key(v, key))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            found.extend(find_key(v, key))
    return found


def same_id(a: Any, b: Any) -> bool:
    return str(a) == str(b)


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@contextmanager
def make_client(settings, registry) -> Iterator[Any]:
    from fastapi.testclient import TestClient

    from sparrow_finetune.api import create_app

    with TestClient(create_app(settings, registry=registry)) as client:
        yield client


# --------------------------------------------------------------------------- HTTP flows


def create_workspace(client, name: Optional[str] = None) -> dict:
    r = client.post("/v1/workspaces", json={"name": name or f"ws-{uuid.uuid4().hex[:8]}"})
    assert r.status_code == 201, r.text
    return r.json()


def create_dataset(
    client, workspace_id, task: str = "classification", name: str = "ds", whole_image: bool = False
) -> dict:
    r = client.post(
        f"/v1/workspaces/{workspace_id}/datasets",
        json={"name": name, "task": task, "whole_image": whole_image},
    )
    assert r.status_code == 201, r.text
    return r.json()


def mark_studio(client, dataset_id, names: Sequence[str]) -> None:
    """Mark images as Studio imports (crops that train whole-frame in a crop-trained dataset)."""
    import psycopg

    with psycopg.connect(client.app.state.settings.database_url) as conn:
        conn.execute(
            "UPDATE dataset_images SET source = 'studio' WHERE dataset_id = %s AND name = ANY(%s)",
            (dataset_id, list(names)),
        )


def upload_image(client, dataset_id, name: str, seed: int) -> dict:
    r = client.put(f"/v1/datasets/{dataset_id}/images/{name}", content=png_bytes(seed))
    assert r.is_success, r.text
    return r.json()


def import_annotations(client, dataset_id, items: list) -> Any:
    return client.post(f"/v1/datasets/{dataset_id}/annotations", json={"items": items})


def annotation_item(image: str, label: str, bbox=None, group: Optional[str] = None) -> dict:
    item: dict = {"image": image, "annotations": [{"label": label, "bbox": bbox}]}
    if group is not None:
        item["group"] = group
    return item


def create_snapshot(client, dataset_id) -> Any:
    return client.post(f"/v1/datasets/{dataset_id}/snapshots")


def build_snapshot(
    client,
    task: str = "classification",
    labels: Sequence[str] = ("cat", "dog", "cat", "dog"),
    groups: Optional[Sequence[str]] = None,
) -> dict:
    """Workspace -> dataset -> one image per label -> annotations -> snapshot."""
    ws = create_workspace(client)
    # Whole-frame labels: classification fixtures train on the whole image.
    ds = create_dataset(client, ws["id"], task=task, whole_image=task == "classification")
    bbox = [4, 4, 20, 20] if task == "detection" else None
    names = []
    items = []
    for i, label in enumerate(labels):
        name = f"img{i}.png"
        upload_image(client, ds["id"], name, seed=i)
        names.append(name)
        items.append(annotation_item(name, label, bbox, groups[i] if groups else None))
    r = import_annotations(client, ds["id"], items)
    assert r.status_code == 200, r.text
    snap = create_snapshot(client, ds["id"])
    assert snap.status_code == 201, snap.text
    return {"workspace": ws, "dataset": ds, "images": names, "snapshot": snap.json()}


def create_job(client, pipeline: str, snapshot_id, params: Optional[dict] = None) -> Any:
    return client.post(
        "/v1/jobs", json={"pipeline": pipeline, "snapshot_id": snapshot_id, "params": params or {}}
    )


def queued_job(client, pipeline: str, snapshot_id, params: Optional[dict] = None) -> str:
    r = create_job(client, pipeline, snapshot_id, params)
    assert r.status_code == 201, r.text
    assert r.json()["state"] == "queued"
    return r.json()["id"]


def get_job(client, job_id) -> dict:
    r = client.get(f"/v1/jobs/{job_id}")
    assert r.status_code == 200, r.text
    return r.json()


def job_events(client, job_id) -> list:
    r = client.get(f"/v1/jobs/{job_id}/events")
    assert r.status_code == 200, r.text
    return as_list(r.json(), "events")


def models_for_job(client, job_id) -> list:
    r = client.get("/v1/models")
    assert r.status_code == 200, r.text
    return [m for m in as_list(r.json(), "models") if same_id(m.get("job_id"), job_id)]


def stage_sequence(events: list) -> list:
    """Pipeline stages in event order, consecutive duplicates collapsed."""
    seq: list = []
    for ev in events:
        stage = ev.get("stage")
        if stage in STAGES and (not seq or seq[-1] != stage):
            seq.append(stage)
    return seq


def dumps(obj: Any) -> str:
    return json.dumps(obj, default=str)
