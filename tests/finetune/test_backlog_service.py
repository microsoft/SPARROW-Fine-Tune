"""FT-16 (stock-model gate, service side) and FT-17 (service nits) pins.

FT-16: a classifier version whose evaluation found the stock model better on the validation
crops (``metrics.stock_worse == 1``) is not activated unless the request confirms it.
FT-17: docs/ideas.md FT-17 (limiter, trusted proxies, SSE close, If-None-Match: *,
``unavailable_reason``, Studio names, attempt-dir pruning, database host parsing).
"""

from __future__ import annotations

import dataclasses
import threading
from collections import OrderedDict

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict
from psycopg.types.json import Jsonb

from sparrow_finetune import db as db_module
from sparrow_finetune import workers as workers_module
from sparrow_finetune.workers import run_once

from .helpers import build_snapshot, queued_job
from .service_helpers import PASSWORD, add_user, make_service_client, studio_settings
from .studio_fake import FakeStudio

STOCK_WORSE = {
    "stock_val_macro_f1": 0.81,
    "stock_finetuned_val_macro_f1": 0.62,
    "stock_compared_crops": 40.0,
    "stock_worse": 1.0,
}
STOCK_BETTER = {
    "stock_val_macro_f1": 0.55,
    "stock_finetuned_val_macro_f1": 0.9,
    "stock_compared_crops": 40.0,
    "stock_worse": 0.0,
}


def _sql(settings, query, params=()):
    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        cur = conn.execute(query, params)
        return cur.fetchall() if cur.description else None


def _model(client, trainers, registry, settings, metrics=None, metadata=None) -> int:
    snap = build_snapshot(client)["snapshot"]
    job_id = queued_job(client, trainers.basic.id, snap["id"])
    assert run_once(settings, "w", {"cpu"}, registry)
    models = client.get("/v1/models", params={"job_id": job_id}).json()["models"]
    assert len(models) == 1, models
    model_id = models[0]["id"]
    _sql(
        settings,
        "UPDATE model_versions SET metrics = metrics || %s, metadata = metadata || %s"
        " WHERE id = %s",
        (Jsonb(metrics or {}), Jsonb(metadata or {}), model_id),
    )
    return model_id


def _deployments(settings, model_id) -> list:
    return _sql(
        settings, "SELECT consumer FROM deployments WHERE model_id = %s", (model_id,)
    )


def _audit(settings, action) -> list:
    return [
        r[0]
        for r in _sql(
            settings,
            "SELECT detail FROM audit_events WHERE action = %s ORDER BY id",
            (action,),
        )
    ]


# --------------------------------------------------------------------------- FT-16


def test_ft16_activate_worse_than_stock_needs_confirm(
    client, trainers, registry, settings
):
    mid = _model(
        client,
        trainers,
        registry,
        settings,
        # Threshold 0 = always enforce, so the 40-crop comparison still blocks (FT-18).
        {**STOCK_WORSE, "stock_min_val_crops": 0.0},
        {"stock_model": "speciesnet-4"},
    )
    r = client.post(
        f"/v1/models/{mid}/activate", json={"consumer": "local", "project": "p"}
    )
    assert r.status_code == 409, r.text
    detail = r.json()["detail"]
    assert detail["code"] == "stock_model_better"
    assert "speciesnet-4" in detail["message"]
    assert "0.81" in detail["message"] and "0.62" in detail["message"]
    assert _deployments(settings, mid) == []
    assert _audit(settings, "model.activate") == []

    r = client.post(
        f"/v1/models/{mid}/activate",
        json={"consumer": "local", "project": "p", "confirm": True},
    )
    assert r.status_code == 200, r.text
    assert _deployments(settings, mid) == [("local",)]
    (detail,) = _audit(settings, "model.activate")
    assert detail["stock_override"] is True
    assert detail["stock_comparison"]["model"] == "speciesnet-4"


def test_ft16_activate_not_gated_when_finetuned_better_or_unknown(
    client, trainers, registry, settings
):
    better = _model(
        client, trainers, registry, settings, STOCK_BETTER, {"stock_model": "s"}
    )
    plain = _model(client, trainers, registry, settings)
    for mid, project in ((better, "a"), (plain, "b")):
        r = client.post(
            f"/v1/models/{mid}/activate", json={"consumer": "local", "project": project}
        )
        assert r.status_code == 200, r.text
        (detail,) = [
            d for d in _audit(settings, "model.activate") if d["project"] == project
        ]
        assert "stock_override" not in detail


def test_ft16_models_expose_stock_comparison(client, trainers, registry, settings):
    # Legacy model: metrics predate ``stock_min_val_crops``, so the API falls back to a
    # minimum of 0 and keeps enforcing the gate it was registered under (FT-18).
    worse = _model(
        client,
        trainers,
        registry,
        settings,
        STOCK_WORSE,
        {"stock_model": "speciesnet-4"},
    )
    plain = _model(client, trainers, registry, settings)
    expected = {
        "model": "speciesnet-4",
        "stock_macro_f1": 0.81,
        "finetuned_macro_f1": 0.62,
        "crops": 40,
        "worse": True,
        "min_val_crops": 0,
        "enforced": True,
    }
    one = client.get(f"/v1/models/{worse}").json()
    assert one["stock_comparison"] == expected
    listed = {m["id"]: m for m in client.get("/v1/models").json()["models"]}
    assert listed[worse]["stock_comparison"] == expected
    assert listed[plain]["stock_comparison"] is None
    assert client.get(f"/v1/models/{plain}").json()["stock_comparison"] is None


def test_ft18_legacy_stock_worse_without_minimum_still_blocks(
    client, trainers, registry, settings
):
    mid = _model(
        client, trainers, registry, settings, STOCK_WORSE, {"stock_model": "s"}
    )
    r = client.post(
        f"/v1/models/{mid}/activate", json={"consumer": "local", "project": "p"}
    )
    assert r.status_code == 409, r.text
    assert r.json()["detail"]["code"] == "stock_model_better"
    assert _deployments(settings, mid) == []


# --------------------------------------------------------------------------- FT-19

REFERENCE_WORSE = {
    "reference_val_macro_f1": 0.7636,
    "val_macro_f1": 0.74,
    "val_crops": 48.0,
    "reference_worse": 1.0,
}


def test_ft19_models_expose_reference_comparison(client, trainers, registry, settings):
    worse = _model(client, trainers, registry, settings, REFERENCE_WORSE)
    plain = _model(client, trainers, registry, settings)
    expected = {
        "reference_macro_f1": 0.7636,
        "finetuned_macro_f1": 0.74,
        "crops": 48,
        "worse": True,
    }
    assert client.get(f"/v1/models/{worse}").json()["reference_comparison"] == expected
    listed = {m["id"]: m for m in client.get("/v1/models").json()["models"]}
    assert listed[worse]["reference_comparison"] == expected
    assert listed[plain]["reference_comparison"] is None
    # a warning, not a block: activation needs no confirm
    r = client.post(
        f"/v1/models/{worse}/activate", json={"consumer": "local", "project": "p"}
    )
    assert r.status_code == 200, r.text


@pytest.fixture
def studio():
    return FakeStudio(project="p1", page_size=2)


def test_ft16_studio_activate_gated_before_push(settings, registry, trainers, studio):
    with make_service_client(
        studio_settings(settings), registry, studio.transport
    ) as c:
        mid = _model(
            c,
            trainers,
            registry,
            settings,
            {**STOCK_WORSE, "stock_min_val_crops": 0.0},
            {"stock_model": "s"},
        )
        r = c.post(
            f"/v1/models/{mid}/activate", json={"consumer": "studio", "project": "p1"}
        )
    assert r.status_code == 409, r.text
    assert r.json()["detail"]["code"] == "stock_model_better"
    assert studio.requests == []


# --------------------------------------------------------------------------- FT-17 limiter


class _CountingDict(OrderedDict):
    """Counts the entries any full or partial scan of the limiter's table visits."""

    visited = 0

    def __iter__(self):
        for k in super().__iter__():
            type(self).visited += 1
            yield k

    def items(self):
        for kv in super().items():
            type(self).visited += 1
            yield kv

    def keys(self):
        return iter(self)

    def values(self):
        for _, v in self.items():
            yield v


def test_ft17_limiter_eviction_does_not_scan_every_key():
    from sparrow_finetune.api.security import LoginLimiter

    limiter = LoginLimiter()
    for i in range(10_001):
        limiter.failure(f"10.0.{i // 256}.{i % 256}")
    limiter._failures = _CountingDict(limiter._failures)
    _CountingDict.visited = 0
    for i in range(200):
        limiter.failure(f"192.168.0.{i}")
    # A full scan per call would visit ~2 million entries.
    assert _CountingDict.visited < 2_000, _CountingDict.visited


def test_ft17_limiter_still_evicts_expired_keys(monkeypatch):
    from sparrow_finetune.api import security

    now = [1000.0]
    monkeypatch.setattr(security.time, "monotonic", lambda: now[0])
    limiter = security.LoginLimiter(window=60)
    for i in range(10_001):
        limiter.failure(f"k{i}")
    now[0] += 61
    limiter.failure("fresh")
    assert set(limiter._failures) == {"fresh"}
    for _ in range(10):
        limiter.failure("fresh")
    assert limiter.retry_after("fresh") is not None


# --------------------------------------------------------------------------- FT-17 trusted proxies


def _login(c, password, xff=None):
    headers = {"X-Forwarded-For": xff} if xff else {}
    return c.post(
        "/v1/auth/login",
        json={"username": "alice", "password": password},
        headers=headers,
    ).status_code


def _proxied_client(settings, registry, peer):
    from fastapi.testclient import TestClient

    from sparrow_finetune.api import create_app

    return TestClient(create_app(settings, registry=registry), client=(peer, 40000))


def test_ft17_trusted_proxy_keys_limiter_by_forwarded_client(settings, registry):
    add_user(settings)
    s = dataclasses.replace(settings, trusted_proxies="10.0.0.0/8, 192.168.1.1")
    with _proxied_client(s, registry, "10.1.2.3") as c:
        for _ in range(10):
            assert _login(c, "wrong", "203.0.113.7") == 401
        assert _login(c, PASSWORD, "203.0.113.7") == 429
        # Another client behind the same proxy is not locked out.
        assert _login(c, PASSWORD, "198.51.100.9") == 200
        # The right-most untrusted hop counts; a spoofed left-most entry does not.
        assert _login(c, PASSWORD, "198.51.100.9, 203.0.113.7") == 429
        assert _login(c, PASSWORD, "203.0.113.7, 192.168.1.1") == 429


def test_ft17_forwarded_for_ignored_by_default_and_from_untrusted_peers(
    settings, registry
):
    add_user(settings)
    assert settings.trusted_proxies == ()
    for s in (settings, dataclasses.replace(settings, trusted_proxies="10.0.0.0/8")):
        with _proxied_client(s, registry, "172.16.0.5") as c:
            for i in range(10):
                assert _login(c, "wrong", f"203.0.113.{i}") == 401
            assert _login(c, PASSWORD, "198.51.100.9") == 429


def test_ft17_trusted_proxies_from_env(monkeypatch):
    from sparrow_finetune.config import Settings

    monkeypatch.setenv("SPARROW_FT_DATABASE_URL", "postgresql://x/y")
    monkeypatch.setenv("SPARROW_FT_TRUSTED_PROXIES", "10.0.0.0/8,::1")
    s = Settings.from_env()
    assert [str(n) for n in s.trusted_proxies] == ["10.0.0.0/8", "::1/128"]
    with pytest.raises(ValueError):
        dataclasses.replace(s, trusted_proxies="not-a-network")


# --------------------------------------------------------------------------- FT-17 SSE close race


def test_ft17_sse_close_during_poll_waits_for_the_poll(settings, monkeypatch):
    import sparrow_finetune.api.jobs as api_jobs

    started, release = threading.Event(), threading.Event()
    seen = {}

    def blocking(conn, job_id, after):
        started.set()
        assert release.wait(10)
        seen["closed_during_poll"] = conn.closed
        return [], "running"

    monkeypatch.setattr(api_jobs, "_poll_events", blocking)
    stream = api_jobs._StreamConn(settings.database_url)
    t = threading.Thread(target=stream.poll, args=(1, 0))
    t.start()
    assert started.wait(10)
    stream.close()  # e.g. the client went away mid-poll
    assert not stream.conn.closed
    release.set()
    t.join(10)
    assert seen["closed_during_poll"] is False
    assert stream.conn.closed
    stream.close()  # idempotent
    with pytest.raises(RuntimeError):
        stream.poll(1, 0)


def test_ft17_sse_close_when_idle_closes_now(settings):
    import sparrow_finetune.api.jobs as api_jobs

    stream = api_jobs._StreamConn(settings.database_url)
    stream.close()
    assert stream.conn.closed


# --------------------------------------------------------------------------- FT-17 If-None-Match


def test_ft17_artifact_if_none_match_star(client, trainers, registry, settings):
    mid = _model(client, trainers, registry, settings)
    url = f"/v1/models/{mid}/artifact"
    r = client.get(url, headers={"If-None-Match": "*"})
    assert r.status_code == 304, r.status_code
    assert r.content == b""
    r = client.get(url, headers={"If-None-Match": '"nope"'})
    assert r.status_code == 200 and r.content


# --------------------------------------------------------------------------- FT-17 unavailable_reason


def test_ft17_pipelines_expose_unavailable_reason(client, trainers):
    trainers.basic.unavailable_reason = "SPARROW_FT_EMBEDDER is not set"
    pipelines = {p["id"]: p for p in client.get("/v1/pipelines").json()["pipelines"]}
    assert pipelines[trainers.basic.id]["unavailable_reason"] == (
        "SPARROW_FT_EMBEDDER is not set"
    )
    assert all("unavailable_reason" in p for p in pipelines.values())
    others = [p for pid, p in pipelines.items() if pid != trainers.basic.id]
    assert others and all(p["unavailable_reason"] is None for p in others)


# --------------------------------------------------------------------------- FT-17 Studio names


def test_ft17_studio_long_duplicate_names_keep_unique_tag():
    from sparrow_finetune.connectors.studio import MAX_NAME, _item_names

    long = "a" * 300 + ".jpg"
    names = _item_names(
        [({"name": long, "image_id": 1},), ({"name": long, "image_id": 2},)]
    )
    assert len(set(names)) == 2, names
    assert all(len(n) <= MAX_NAME and n.endswith(".jpg") for n in names)
    assert "__studio1" in names[0] and "__studio2" in names[1]


def test_ft17_studio_long_crop_names_keep_box_tag():
    from sparrow_finetune.connectors.studio import MAX_NAME, _crop_name

    a, b = _crop_name("x" * 300 + ".jpg", 7), _crop_name("x" * 300 + ".jpg", 8)
    assert a != b
    assert len(a) <= MAX_NAME and a.endswith("__box7.png") and b.endswith("__box8.png")
    assert _crop_name("cat.jpg", 3) == "cat__box3.png"


# --------------------------------------------------------------------------- FT-17 attempt dirs


def test_ft17_stale_attempt_dirs_of_finished_jobs_pruned(
    client, trainers, registry, settings, monkeypatch
):
    monkeypatch.setattr(workers_module, "STALE_ATTEMPT_GRACE_S", 3600, raising=False)
    snap = build_snapshot(client)["snapshot"]
    old = queued_job(client, trainers.basic.id, snap["id"])
    assert run_once(settings, "w", {"cpu"}, registry)
    recent = queued_job(client, trainers.basic.id, snap["id"])
    assert run_once(settings, "w", {"cpu"}, registry)
    # A killed worker's attempt dir: its own cleanup never ran.
    for job_id in (old, recent):
        stale = settings.jobs_dir / str(job_id) / "attempt-1"
        stale.mkdir(parents=True, exist_ok=True)
        (stale / "weights.pt").write_bytes(b"x")
    _sql(
        settings,
        "UPDATE jobs SET finished_at = now() - interval '2 hours' WHERE id = %s",
        (int(old),),
    )
    assert run_once(settings, "w", {"cpu"}, registry) is False
    assert not (settings.jobs_dir / str(old)).exists()
    # Within the grace period a stale worker may still be shutting down.
    assert (settings.jobs_dir / str(recent) / "attempt-1" / "weights.pt").exists()


def test_ft17_attempt_dirs_of_deleted_jobs_pruned(registry, settings):
    stale = settings.jobs_dir / "987654" / "attempt-2"
    stale.mkdir(parents=True)
    (stale / "x").write_bytes(b"x")
    assert run_once(settings, "w", {"cpu"}, registry) is False
    assert not stale.parent.exists()


def test_ft17_running_jobs_attempt_dirs_kept(
    client, trainers, registry, settings, queue
):
    snap = build_snapshot(client)["snapshot"]
    queued_job(client, trainers.basic.id, snap["id"])
    job = queue.claim("other", {"cpu"})
    live = settings.jobs_dir / str(job.id) / f"attempt-{job.attempts}"
    live.mkdir(parents=True)
    (live / "x").write_bytes(b"x")
    assert run_once(settings, "w", {"cpu"}, registry) is False
    assert (live / "x").exists()


# --------------------------------------------------------------------------- FT-17 database host


def _url(**env):
    return conninfo_to_dict(
        db_module.build_database_url({f"SPARROW_FT_DB_{k}": v for k, v in env.items()})
    )


@pytest.mark.parametrize(
    "host, port, want_host, want_port",
    [
        ("::1", None, "::1", "5432"),
        ("[::1]", None, "::1", "5432"),
        ("fe80::1", "6000", "fe80::1", "6000"),
        ("[::1]:6543", None, "::1", "6543"),
        ("db.example:6543", None, "db.example", "6543"),
        ("10.0.0.1:5433", None, "10.0.0.1", "5433"),
        ("db.example:6543", "6543", "db.example", "6543"),
        ("postgres", None, "postgres", "5432"),
    ],
)
def test_ft17_database_host_forms(host, port, want_host, want_port):
    env = {"HOST": host}
    if port is not None:
        env["PORT"] = port
    info = _url(**env)
    assert (info["host"], info["port"]) == (want_host, want_port)


@pytest.mark.parametrize(
    "host, port",
    [
        ("db.example:6543", "5432"),
        ("a:b:c", None),
        ("db:notaport", None),
        ("[db]", None),
    ],
)
def test_ft17_database_host_bad_forms(host, port):
    env = {"SPARROW_FT_DB_HOST": host}
    if port is not None:
        env["SPARROW_FT_DB_PORT"] = port
    with pytest.raises(RuntimeError):
        db_module.build_database_url(env)


# --------------------------------------------------------------------------- model names


def test_model_rename_sets_trims_and_audits(client, trainers, registry, settings):
    mid = _model(client, trainers, registry, settings)
    assert client.get(f"/v1/models/{mid}").json()["name"] is None
    r = client.patch(f"/v1/models/{mid}", json={"name": "  Amazon mammals v1  "})
    assert r.status_code == 200, r.text
    assert r.json()["name"] == "Amazon mammals v1"
    assert client.get(f"/v1/models/{mid}").json()["name"] == "Amazon mammals v1"
    (listed,) = [m for m in client.get("/v1/models").json()["models"] if m["id"] == mid]
    assert listed["name"] == "Amazon mammals v1"
    assert _audit(settings, "model.rename") == [{"model_id": mid, "name": "Amazon mammals v1"}]


def test_model_rename_blank_resets_to_default(client, trainers, registry, settings):
    mid = _model(client, trainers, registry, settings)
    assert client.patch(f"/v1/models/{mid}", json={"name": "first"}).status_code == 200
    r = client.patch(f"/v1/models/{mid}", json={"name": "   "})
    assert r.status_code == 200, r.text
    assert r.json()["name"] is None
    assert client.patch(f"/v1/models/{mid}", json={"name": None}).json()["name"] is None


def test_model_rename_rejects_long_names_and_unknown_models(client, trainers, registry, settings):
    mid = _model(client, trainers, registry, settings)
    assert client.patch(f"/v1/models/{mid}", json={"name": "x" * 101}).status_code == 422
    assert client.patch(f"/v1/models/{mid}", json={}).status_code == 422
    assert client.patch("/v1/models/999999", json={"name": "x"}).status_code == 404
    assert client.get(f"/v1/models/{mid}").json()["name"] is None
