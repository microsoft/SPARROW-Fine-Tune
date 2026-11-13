"""FT-9 / FT-10: pins for the service review-backlog items.

FT-9: P-FT1-N1..N9 (docs/review/ft1-minor-fixes/backlog.md).
FT-10: P2-N4..N10, P2-P1, P2-P2 (docs/review/implement-phase2-service/backlog.md).
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import stat
import threading
import time
from pathlib import Path

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict

from sparrow_finetune import db as db_module
from sparrow_finetune import storage as storage_module
from sparrow_finetune.jobs import JobQueue, LeaseLost
from sparrow_finetune.pipelines import Registry
from sparrow_finetune.storage import LocalObjectStore
from sparrow_finetune.workers import _Heartbeat, run_forever, run_once

from .fakes import RecordingTrainer
from .helpers import (
    build_snapshot,
    create_dataset,
    create_workspace,
    get_job,
    make_client,
    png_bytes,
    queued_job,
)
from .service_helpers import add_user, make_service_client, read_sse, studio_settings
from .studio_fake import FakeStudio

REPO = Path(__file__).resolve().parents[2]


def _sql(settings, query, params=()):
    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        cur = conn.execute(query, params)
        return cur.fetchall() if cur.description else None


def _expire_lease(settings, job_id) -> None:
    _sql(
        settings,
        "UPDATE jobs SET lease_expires_at = clock_timestamp() - interval '1 second'"
        " WHERE id = %s",
        (job_id,),
    )


def _messages(settings, job_id) -> list:
    return [
        r[0]
        for r in _sql(
            settings,
            "SELECT message FROM job_events WHERE job_id = %s ORDER BY seq",
            (job_id,),
        )
    ]


# --------------------------------------------------------------------------- P-FT1-N1


def test_p_ft1_n1_heartbeat_fenced_by_attempt(client, trainers, settings, queue):
    snap = build_snapshot(client)["snapshot"]
    queued_job(client, trainers.basic.id, snap["id"])
    job = queue.claim("w", {"cpu"})
    assert job.attempts == 1
    # The same worker id re-claimed the job: attempt 2 now owns it.
    _sql(settings, "UPDATE jobs SET attempts = 2 WHERE id = %s", (job.id,))
    with pytest.raises(LeaseLost):
        queue.fenced(1).heartbeat(job.id, "w", 60)
    with pytest.raises(LeaseLost):
        queue.heartbeat(job.id, "w", 60, attempt=1)
    heart = _Heartbeat(queue, job.id, "w", 60, attempt=1)
    heart.beat()
    assert heart.lease_lost.is_set()
    assert queue.fenced(2).heartbeat(job.id, "w", 60) is False


# --------------------------------------------------------------------------- P-FT1-N2


def test_p_ft1_n2_heartbeat_deadline_counts_from_claim(settings, monkeypatch):
    trainer = RecordingTrainer()
    reg = Registry()
    reg.register(trainer)
    real_claim = JobQueue.claim

    def slow_claim(self, *args, **kwargs):
        job = real_claim(self, *args, **kwargs)
        if job is not None:
            time.sleep(1.5)  # the claim returns late: the 1 s lease has already run out
        return job

    def broken(self, *args, **kwargs):
        raise psycopg.OperationalError("simulated database outage")

    monkeypatch.setattr(JobQueue, "claim", slow_claim)
    monkeypatch.setattr(JobQueue, "heartbeat", broken)
    with make_client(settings, reg) as client:
        snap = build_snapshot(client)["snapshot"]
        job_id = queued_job(client, trainer.id, snap["id"])
        assert run_once(settings, "w", {"cpu"}, reg, 1.0)
        assert trainer.calls == []
        assert "started" not in _messages(settings, job_id)
        assert get_job(client, job_id)["state"] == "running"


def test_p_ft1_n2_heartbeat_seeded_with_claim_time(queue):
    heart = _Heartbeat(queue, 1, "w", 1.0, attempt=1, claimed_at=time.monotonic() - 5)
    with pytest.raises(LeaseLost):
        heart.check()


# --------------------------------------------------------------------------- P-FT1-N3


def test_p_ft1_n3_statement_timeout_default(settings, monkeypatch):
    monkeypatch.delenv("SPARROW_FT_STATEMENT_TIMEOUT_MS", raising=False)
    with db_module.connect(settings.database_url) as conn:
        assert (
            conn.execute("SHOW statement_timeout").fetchone()["statement_timeout"]
            == "1min"
        )


def test_p_ft1_n3_statement_timeout_env_override(settings, monkeypatch):
    monkeypatch.setenv("SPARROW_FT_STATEMENT_TIMEOUT_MS", "5000")
    with db_module.connect(settings.database_url) as conn:
        assert (
            conn.execute("SHOW statement_timeout").fetchone()["statement_timeout"]
            == "5s"
        )
    monkeypatch.setenv("SPARROW_FT_STATEMENT_TIMEOUT_MS", "soon")
    with pytest.raises(RuntimeError):
        db_module.connect(settings.database_url)


# --------------------------------------------------------------------------- P-FT1-N4


def test_p_ft1_n4_stale_worker_cannot_append_to_terminal_job(
    client, trainers, settings, queue
):
    snap = build_snapshot(client)["snapshot"]
    job_id = queued_job(client, trainers.basic.id, snap["id"])
    job = queue.claim("w", {"cpu"})
    queue.request_cancel(job.id)
    _expire_lease(settings, job.id)
    assert queue.reap_expired() == 1
    assert get_job(client, job_id)["state"] == "cancelled"
    # Same worker, same attempt, but the reaper ended the job: no more events from it.
    with pytest.raises(LeaseLost):
        queue.add_event(job.id, "train", "stale-on-terminal", worker_id="w", attempt=1)
    with pytest.raises(LeaseLost):
        queue.add_event(job.id, "cancelled", "stale-terminal", worker_id="w", attempt=1)
    assert not {"stale-on-terminal", "stale-terminal"} & set(
        _messages(settings, job_id)
    )


def test_p_ft1_n4_finishing_worker_logs_only_its_terminal_event(
    client, trainers, settings, queue
):
    snap = build_snapshot(client)["snapshot"]
    job_id = queued_job(client, trainers.basic.id, snap["id"])
    job = queue.claim("w", {"cpu"})
    queue.fail(job.id, "w", "boom", attempt=1)
    queue.add_event(job.id, "failed", "boom", worker_id="w", attempt=1)
    with pytest.raises(LeaseLost):
        queue.add_event(job.id, "train", "late", worker_id="w", attempt=1)
    with pytest.raises(LeaseLost):
        queue.add_event(job.id, "failed", "other", worker_id="other", attempt=1)
    msgs = _messages(settings, job_id)
    assert "boom" in msgs and "late" not in msgs and "other" not in msgs


# --------------------------------------------------------------------------- P-FT1-N5


class _ModeTrainer(RecordingTrainer):
    def __init__(self) -> None:
        super().__init__()
        self.modes: list = []
        self.write_error = None

    def prepare(self, ctx):
        super().prepare(ctx)
        self.modes = [stat.S_IMODE(os.stat(i.image_path).st_mode) for i in ctx.items]
        if os.geteuid() != 0:  # root ignores permission bits
            try:
                with open(ctx.items[0].image_path, "ab"):
                    pass
            except PermissionError as exc:
                self.write_error = exc


def test_p_ft1_n5_stored_objects_read_only(tmp_path):
    store = LocalObjectStore(tmp_path / "objects")
    store.put_bytes("a/b", b"data")
    src = tmp_path / "src.bin"
    src.write_bytes(b"x")
    store.put_file("a/c", src)
    for key in ("a/b", "a/c"):
        assert stat.S_IMODE(store.path(key).stat().st_mode) == 0o444


def test_p_ft1_n5_staged_images_read_only(settings):
    trainer = _ModeTrainer()
    reg = Registry()
    reg.register(trainer)
    with make_client(settings, reg) as client:
        snap = build_snapshot(client)["snapshot"]
        job_id = queued_job(client, trainer.id, snap["id"])
        assert run_once(settings, "w", {"cpu"}, reg)
        assert get_job(client, job_id)["state"] == "succeeded"
    assert trainer.modes and set(trainer.modes) == {0o444}, trainer.modes
    if os.geteuid() != 0:
        assert trainer.write_error is not None


# --------------------------------------------------------------------------- P-FT1-N6 / N9


def _files(root: Path) -> list:
    return [p for p in root.rglob("*") if not p.is_dir()] if root.exists() else []


def test_p_ft1_n6_staged_dir_pruned_when_job_terminal(
    client, trainers, registry, settings
):
    snap = build_snapshot(client)["snapshot"]
    job_id = queued_job(client, trainers.basic.id, snap["id"])
    assert run_once(settings, "w", {"cpu"}, registry)
    assert get_job(client, job_id)["state"] == "succeeded"
    assert trainers.basic.items, "the trainer saw no staged items"
    # The next poll's sweep prunes it (existing tests read staged files right after run_once).
    assert run_once(settings, "w", {"cpu"}, registry) is False
    assert _files(settings.staged_dir) == []


def test_p_ft1_n6_orphan_and_legacy_staged_dirs_pruned(
    client, trainers, registry, settings
):
    orphan = settings.staged_dir / "job-999999" / "ab"
    orphan.mkdir(parents=True)
    (orphan / "x.png").write_bytes(b"x")
    assert run_once(settings, "w", {"cpu"}, registry) is False
    assert not orphan.parent.exists()

    legacy = settings.staged_dir / "ab"
    legacy.mkdir(parents=True)
    (legacy / "ab12.png").write_bytes(b"x")
    stop = threading.Event()
    stop.set()
    run_forever(settings, "w", {"cpu"}, registry, stop=stop)
    assert not legacy.exists()


def test_p_ft1_n6_running_jobs_staged_dir_kept(
    client, trainers, settings, queue, registry
):
    snap = build_snapshot(client)["snapshot"]
    queued_job(client, trainers.basic.id, snap["id"])
    job = queue.claim("other", {"cpu"})
    live = settings.staged_dir / f"job-{job.id}"
    live.mkdir(parents=True)
    (live / "x.png").write_bytes(b"x")
    assert run_once(settings, "w", {"cpu"}, registry) is False
    assert (live / "x.png").exists()


def test_p_ft1_n9_empty_job_dir_pruned_when_terminal(
    client, trainers, registry, settings
):
    snap = build_snapshot(client)["snapshot"]
    job_id = queued_job(client, trainers.basic.id, snap["id"])
    assert run_once(settings, "w", {"cpu"}, registry)
    assert get_job(client, job_id)["state"] == "succeeded"
    assert run_once(settings, "w", {"cpu"}, registry) is False
    assert not (settings.jobs_dir / str(job_id)).exists()


# --------------------------------------------------------------------------- P-FT1-N7


def test_p_ft1_n7_new_parent_dirs_fsynced(tmp_path, monkeypatch):
    synced: list = []
    real = storage_module._fsync_dir
    monkeypatch.setattr(
        storage_module, "_fsync_dir", lambda p: (synced.append(Path(p)), real(p))[1]
    )
    root = tmp_path / "objects"
    LocalObjectStore(root).put_bytes("a/b/c", b"data")
    # Each new directory's entry lives in its parent: root's parent, root, root/a, root/a/b.
    for parent in (tmp_path, root, root / "a", root / "a" / "b"):
        assert parent in synced, (parent, synced)


# --------------------------------------------------------------------------- P2-N4


def test_p2_n4_safe_keeps_extension():
    from sparrow_finetune.connectors.studio import MAX_NAME, _safe

    out = _safe("x" * 300 + ".png")
    assert out.endswith(".png") and len(out) == MAX_NAME
    crop = _safe("y" * 300 + ".jpg#box-12.png")
    assert crop.endswith(".png") and len(crop) == MAX_NAME
    assert _safe("short.png") == "short.png"


# --------------------------------------------------------------------------- P2-N5


@pytest.fixture
def studio():
    return FakeStudio(project="p1", page_size=2)


@pytest.fixture
def svc(settings, registry, studio):
    with make_service_client(
        studio_settings(settings), registry, studio.transport
    ) as c:
        yield c


def test_p2_n5_numeric_project_ids_rejected_before_studio(svc, studio, monkeypatch):
    monkeypatch.setenv("SPARROW_FT_STUDIO_PROJECT_IDS", "numeric")
    ws = create_workspace(svc)
    ds = create_dataset(svc, ws["id"], task="detection")["id"]
    r = svc.post(
        "/v1/connectors/studio/import", json={"project": "p1", "dataset_id": ds}
    )
    assert r.status_code == 422, r.text
    assert "numeric project id" in r.text
    r = svc.post(
        "/v1/connectors/studio/push", json={"model_id": 999999, "project": "p1"}
    )
    assert r.status_code == 422, r.text
    assert studio.requests == []
    assert svc.get("/v1/connectors/studio").json()["project_ids"] == "numeric"


def test_p2_n5_ui_validates_project_id():
    js = (REPO / "src/sparrow_finetune/ui/app.js").read_text()
    assert "checkStudioProject" in js and "/^[0-9]+$/" in js
    # Import and Studio activation both read the project through the validating helper.
    helper = js[js.index("function selectedStudioProject("):]
    assert "checkStudioProject(project)" in helper[: helper.index("\n}")]
    for fn in ("async function studioImport(", "async function activate("):
        body = js[js.index(fn):]
        assert "selectedStudioProject(" in body[: body.index("\n}")], fn


# --------------------------------------------------------------------------- P2-N6


TOKEN = "docs-token"


@pytest.fixture
def tokened(settings, registry):
    s = dataclasses.replace(settings, api_tokens=(TOKEN,))
    with make_service_client(s, registry) as c:
        yield c


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_p2_n6_docs_require_auth(tokened, path):
    assert tokened.get(path).status_code == 401
    r = tokened.get(path, headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 200, r.text
    if path == "/openapi.json":
        assert "/v1/jobs" in r.json()["paths"]


# --------------------------------------------------------------------------- P2-N7


def test_p2_n7_login_rate_limited(settings, registry):
    add_user(settings)
    with make_service_client(settings, registry) as c:
        for _ in range(10):
            r = c.post(
                "/v1/auth/login", json={"username": "alice", "password": "wrong"}
            )
            assert r.status_code == 401, r.text
        r = c.post(
            "/v1/auth/login",
            json={"username": "alice", "password": "correct-horse-battery"},
        )
        assert r.status_code == 429, r.text
        assert int(r.headers["Retry-After"]) > 0


def test_p2_n7_successful_logins_not_counted(settings, registry):
    add_user(settings)
    with make_service_client(settings, registry) as c:
        for _ in range(12):
            r = c.post(
                "/v1/auth/login",
                json={"username": "alice", "password": "correct-horse-battery"},
            )
            assert r.status_code == 200, r.text


# --------------------------------------------------------------------------- P2-N8


def test_p2_n8_database_url_quotes_password():
    password = "p@:/%x?#"
    url = db_module.build_database_url(
        {
            "SPARROW_FT_DB_HOST": "postgres",
            "SPARROW_FT_DB_USER": "sparrow",
            "SPARROW_FT_DB_PASSWORD": password,
        }
    )
    info = conninfo_to_dict(url)
    assert info["password"] == password
    assert info["host"] == "postgres" and info["user"] == "sparrow"
    assert info["port"] == "5432" and info["dbname"] == "sparrow_finetune"
    assert db_module.build_database_url({}) is None


def test_p2_n8_compose_passes_db_parts():
    text = (REPO / "compose.service.yml").read_text()
    assert "SPARROW_FT_DATABASE_URL" not in text
    assert "SPARROW_FT_DB_PASSWORD: ${POSTGRES_PASSWORD" in text


def test_compose_forwards_weights_dir():
    """FT-13 offline weights and the FT-16 stock_model gate read SPARROW_FT_WEIGHTS_DIR on the worker."""
    text = (REPO / "compose.service.yml").read_text()
    assert "SPARROW_FT_WEIGHTS_DIR: ${SPARROW_FT_WEIGHTS_DIR:-}" in text
    example = (REPO / ".env.service.example").read_text()
    assert "SPARROW_FT_WEIGHTS_DIR=" in example


def test_p2_n8_cli_builds_url_from_parts(monkeypatch):
    from sparrow_finetune import cli

    seen = {}
    monkeypatch.setenv("SPARROW_FT_DATABASE_URL", "")
    monkeypatch.setenv("SPARROW_FT_DB_HOST", "db")
    monkeypatch.setenv("SPARROW_FT_DB_PASSWORD", "a@b")

    def fake_migrate(url):
        seen["url"] = url
        return []

    monkeypatch.setattr(db_module, "migrate", fake_migrate)
    assert cli.main(["migrate"]) in (0, None)
    assert conninfo_to_dict(seen["url"])["password"] == "a@b"


# --------------------------------------------------------------------------- P2-N9 / P2-P1


def _image_ds(client) -> int:
    ws = create_workspace(client)
    return create_dataset(client, ws["id"], task="classification")["id"]


def test_p2_n9_image_content_etag_revalidates(client):
    ds = _image_ds(client)
    r = client.put(f"/v1/datasets/{ds}/images/a.png", content=png_bytes(1))
    assert r.is_success, r.text
    url = f"/v1/datasets/{ds}/images/a.png/content"
    r = client.get(url)
    assert r.status_code == 200
    etag = r.headers["etag"]
    assert etag == f'"{hashlib.sha256(png_bytes(1)).hexdigest()}"'
    assert "no-cache" in r.headers["cache-control"]
    assert "max-age" not in r.headers["cache-control"]
    r = client.get(url, headers={"If-None-Match": etag})
    assert r.status_code == 304
    r = client.put(f"/v1/datasets/{ds}/images/a.png?replace=true", content=png_bytes(2))
    assert r.is_success, r.text
    r = client.get(url, headers={"If-None-Match": etag})
    assert r.status_code == 200 and r.content == png_bytes(2)
    assert r.headers["etag"] != etag


def test_p2_p1_image_replace_needs_flag(client):
    ds = _image_ds(client)
    put = f"/v1/datasets/{ds}/images/a.png"
    assert client.put(put, content=png_bytes(1)).is_success
    assert client.put(put, content=png_bytes(1)).is_success  # identical: idempotent
    r = client.put(put, content=png_bytes(2))
    assert r.status_code == 409, r.text
    assert "replace=true" in r.text
    assert client.get(put + "/content").content == png_bytes(1)
    r = client.put(put + "?replace=true", content=png_bytes(2))
    assert r.is_success, r.text
    assert client.get(put + "/content").content == png_bytes(2)


# --------------------------------------------------------------------------- P2-N10


def test_p2_n10_sse_stream_uses_one_connection(client, trainers, settings, monkeypatch):
    import sparrow_finetune.api.jobs as api_jobs

    snap = build_snapshot(client)["snapshot"]
    job_id = queued_job(client, trainers.basic.id, snap["id"])
    opened = []
    real = api_jobs.connect

    def counting(url):
        opened.append(threading.get_ident())
        return real(url)

    monkeypatch.setattr(api_jobs, "connect", counting)

    def cancel_later():
        time.sleep(0.6)
        JobQueue(settings.database_url).request_cancel(int(job_id))

    t = threading.Thread(target=cancel_later)
    t.start()
    out = read_sse(client, f"/v1/jobs/{job_id}/stream?poll=0.05")
    t.join()
    assert out["status"] == 200
    assert out["events"][-1]["event"] == "end"
    assert len(opened) == 1, len(opened)


# --------------------------------------------------------------------------- P2-P2


def test_p2_p2_studio_images_stored_as_fetched(svc, studio, monkeypatch):
    order: list = []
    real_put = LocalObjectStore.put_bytes

    def recording_put(self, key, data):
        fetched = [r for r in studio.requests if "/images/" in r.url.path]
        order.append(len(fetched))
        return real_put(self, key, data)

    monkeypatch.setattr(LocalObjectStore, "put_bytes", recording_put)
    ws = create_workspace(svc)
    ds = create_dataset(svc, ws["id"], task="detection")["id"]
    r = svc.post(
        "/v1/connectors/studio/import", json={"project": "p1", "dataset_id": ds}
    )
    assert r.status_code == 200, r.text
    images = [r for r in studio.requests if "/images/" in r.url.path]
    assert len(images) >= 3
    # Each image is stored right after it is fetched, not after all are in memory.
    assert order[0] == 1, order
    assert order == sorted(order) and order[-1] == len(images), order


# --------------------------------------------------------------------------- review of the backlog merge


def test_review_i1_concurrent_bad_logins_respect_limit(settings, registry):
    from concurrent.futures import ThreadPoolExecutor

    add_user(settings)
    with make_service_client(settings, registry) as c:
        barrier = threading.Barrier(40)

        def attempt(_):
            barrier.wait()
            return c.post(
                "/v1/auth/login", json={"username": "alice", "password": "wrong"}
            ).status_code

        with ThreadPoolExecutor(40) as ex:
            codes = list(ex.map(attempt, range(40)))
    assert codes.count(401) <= 10, codes
    assert codes.count(429) >= 30, codes


def test_review_i2_concurrent_first_uploads_need_replace(client, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    from .helpers import create_dataset, create_workspace, png_bytes

    ws = create_workspace(client)
    ds = create_dataset(client, ws["id"], task="classification")["id"]
    real_exists = LocalObjectStore.exists

    def slow_exists(self, key):  # widen the window between the 409 check and the upsert
        time.sleep(0.5)
        return real_exists(self, key)

    monkeypatch.setattr(LocalObjectStore, "exists", slow_exists)
    barrier = threading.Barrier(2)

    def put(i):
        barrier.wait()
        return client.put(
            f"/v1/datasets/{ds}/images/a.png", content=png_bytes(i)
        ).status_code

    with ThreadPoolExecutor(2) as ex:
        codes = sorted(ex.map(put, [1, 2]))
    assert codes == [200, 409], codes


def test_review_i3_connect_keeps_dsn_options(settings, monkeypatch):
    from psycopg.conninfo import make_conninfo

    monkeypatch.delenv("SPARROW_FT_STATEMENT_TIMEOUT_MS", raising=False)
    url = make_conninfo(settings.database_url, options="-c application_name=ft_i3")
    with db_module.connect(url) as conn:
        assert conn.execute("SHOW application_name").fetchone()["application_name"] == "ft_i3"
        assert conn.execute("SHOW statement_timeout").fetchone()["statement_timeout"] == "1min"


def test_review_i3_connect_keeps_pgoptions(settings, monkeypatch):
    monkeypatch.delenv("SPARROW_FT_STATEMENT_TIMEOUT_MS", raising=False)
    monkeypatch.setenv("PGOPTIONS", "-c application_name=ft_pgopt")
    with db_module.connect(settings.database_url) as conn:
        assert conn.execute("SHOW application_name").fetchone()["application_name"] == "ft_pgopt"
        assert conn.execute("SHOW statement_timeout").fetchone()["statement_timeout"] == "1min"


def test_ui_assets_revalidate_so_a_new_page_never_runs_with_old_css(client):
    # index.html is no-cache; its CSS/JS must be too, or a browser pairs new HTML with a
    # heuristically cached old stylesheet (an unstyled SVG chevron drew at 300x150).
    for path in ("/ui/styles.css", "/ui/app.js"):
        r = client.get(path)
        assert r.status_code == 200, path
        assert "no-cache" in r.headers.get("cache-control", ""), path
        r = client.get(path, headers={"If-None-Match": r.headers["etag"]})
        assert r.status_code == 304, path


def test_ui_svg_icons_carry_their_own_size():
    from sparrow_finetune.api import UI_DIR
    import re

    html = (UI_DIR / "index.html").read_text()
    unsized = [s for s in re.findall(r"<svg[^>]*>", html) if 'width="' not in s or 'height="' not in s]
    assert unsized == []
