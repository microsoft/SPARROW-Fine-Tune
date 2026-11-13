"""Removing datasets and workspaces.

Policy (user decisions 2026-10-21, revised 2026-10-28): deletion is permanent and frees
stored images. Data that trained a job is refused unless the caller asks to remove the
jobs and models too (``include_models=true``, see test_delete_with_models.py), and always
refused while one of its jobs is queued or running. Image objects are content-addressed
and may be shared, so only objects nothing else references go.
"""

from __future__ import annotations

import hashlib

from sparrow_finetune.api.datasets import image_key
from sparrow_finetune.db import connect
from sparrow_finetune.storage import LocalObjectStore

from .helpers import (
    annotation_item,
    as_list,
    create_dataset,
    create_snapshot,
    create_workspace,
    import_annotations,
    png_bytes,
    queued_job,
    upload_image,
)


def _labelled_dataset(client, ws_id, seeds=(1, 2), name="ds"):
    ds = create_dataset(client, ws_id, name=name, whole_image=True)
    for i, seed in enumerate(seeds):
        upload_image(client, ds["id"], f"img{i}.png", seed)
    import_annotations(
        client,
        ds["id"],
        [
            annotation_item(f"img{i}.png", "cat" if i % 2 else "dog")
            for i in range(len(seeds))
        ],
    )
    return ds


def _key(seed: int) -> str:
    return image_key(hashlib.sha256(png_bytes(seed)).hexdigest())


def _count(settings, sql, params=()):
    with connect(settings.database_url) as conn:
        return conn.execute(sql, params).fetchone()["n"]


def test_delete_dataset_removes_rows_objects_and_audits(client, settings):
    ws = create_workspace(client)
    ds = _labelled_dataset(client, ws["id"], seeds=(11, 12))
    create_snapshot(client, ds["id"])
    store = LocalObjectStore(settings.objects_dir)
    assert store.exists(_key(11)) and store.exists(_key(12))

    r = client.delete(f"/v1/datasets/{ds['id']}")

    assert r.status_code == 200, r.text
    assert r.json()["deleted"]["images"] == 2
    assert r.json()["deleted"]["snapshots"] == 1
    assert client.get(f"/v1/datasets/{ds['id']}").status_code == 404
    listed = as_list(
        client.get(f"/v1/workspaces/{ws['id']}/datasets").json(), "datasets"
    )
    assert listed == []
    assert not store.exists(_key(11)) and not store.exists(_key(12))
    for table in ("dataset_images", "dataset_items", "snapshots"):
        assert (
            _count(
                settings,
                f"SELECT count(*) AS n FROM {table} WHERE dataset_id = %s",
                (ds["id"],),
            )
            == 0
        )
    assert (
        _count(
            settings,
            "SELECT count(*) AS n FROM audit_events WHERE action = 'dataset.delete' AND target = %s",
            (f"dataset:{ds['id']}",),
        )
        == 1
    )


def test_delete_dataset_keeps_objects_shared_with_another_dataset(client, settings):
    ws = create_workspace(client)
    gone = _labelled_dataset(client, ws["id"], seeds=(21, 22), name="gone")
    kept = create_dataset(client, ws["id"], name="kept")
    upload_image(client, kept["id"], "same.png", 21)
    store = LocalObjectStore(settings.objects_dir)

    assert client.delete(f"/v1/datasets/{gone['id']}").status_code == 200

    assert store.exists(_key(21)), "object still used by another dataset was removed"
    assert not store.exists(_key(22))
    r = client.get(f"/v1/datasets/{kept['id']}/images/same.png/content")
    assert r.status_code == 200 and r.content == png_bytes(21)


def test_delete_dataset_keeps_objects_frozen_in_another_snapshot(client, settings):
    """An image replaced in dataset B still lives on in B's older snapshot."""
    ws = create_workspace(client)
    a = _labelled_dataset(client, ws["id"], seeds=(31,), name="a")
    b = _labelled_dataset(client, ws["id"], seeds=(31,), name="b")
    create_snapshot(client, b["id"])
    r = client.put(
        f"/v1/datasets/{b['id']}/images/img0.png?replace=true", content=png_bytes(32)
    )
    assert r.is_success, r.text

    assert client.delete(f"/v1/datasets/{a['id']}").status_code == 200

    assert LocalObjectStore(settings.objects_dir).exists(_key(31))


def test_delete_dataset_refused_while_a_job_used_it(client, settings):
    ws = create_workspace(client)
    ds = _labelled_dataset(client, ws["id"], seeds=(41, 42))
    snap = create_snapshot(client, ds["id"]).json()
    job_id = queued_job(client, "fake_cls", snap["id"])

    r = client.delete(f"/v1/datasets/{ds['id']}")

    assert r.status_code == 409, r.text
    assert str(job_id) in r.json()["detail"]
    assert client.get(f"/v1/datasets/{ds['id']}").status_code == 200
    assert LocalObjectStore(settings.objects_dir).exists(_key(41))


def test_delete_missing_dataset_and_workspace_404(client):
    assert client.delete("/v1/datasets/999999").status_code == 404
    assert client.delete("/v1/workspaces/999999").status_code == 404


def test_delete_workspace_removes_its_datasets(client, settings):
    ws = create_workspace(client)
    _labelled_dataset(client, ws["id"], seeds=(51,), name="one")
    _labelled_dataset(client, ws["id"], seeds=(52,), name="two")
    other = create_workspace(client)

    r = client.delete(f"/v1/workspaces/{ws['id']}")

    assert r.status_code == 200, r.text
    assert r.json()["deleted"]["datasets"] == 2
    ids = [w["id"] for w in as_list(client.get("/v1/workspaces").json(), "workspaces")]
    assert ws["id"] not in ids and other["id"] in ids
    store = LocalObjectStore(settings.objects_dir)
    assert not store.exists(_key(51)) and not store.exists(_key(52))
    assert (
        _count(
            settings,
            "SELECT count(*) AS n FROM audit_events WHERE action = 'workspace.delete' AND target = %s",
            (f"workspace:{ws['id']}",),
        )
        == 1
    )


def test_delete_workspace_refused_atomically_when_any_dataset_trained(client, settings):
    ws = create_workspace(client)
    plain = _labelled_dataset(client, ws["id"], seeds=(61,), name="plain")
    trained = _labelled_dataset(client, ws["id"], seeds=(62,), name="trained")
    snap = create_snapshot(client, trained["id"]).json()
    queued_job(client, "fake_cls", snap["id"])

    r = client.delete(f"/v1/workspaces/{ws['id']}")

    assert r.status_code == 409, r.text
    assert "cancel" in r.json()["detail"].lower()
    assert client.get(f"/v1/datasets/{plain['id']}").status_code == 200
    assert LocalObjectStore(settings.objects_dir).exists(_key(61))
