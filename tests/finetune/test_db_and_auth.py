"""AC1 (migrations), AC2 (health), AC3 (bearer-token auth).

Migrations are checked against a database nothing else has touched; auth is
checked on several route families so one forgotten dependency is caught.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sparrow_finetune.config import Settings
from sparrow_finetune.db import migrate

from .helpers import bearer, make_client

TOKEN = "test-token-123"


def _migration_files() -> list[Path]:
    import sparrow_finetune.db as db_pkg

    return sorted((Path(db_pkg.__file__).parent / "migrations").glob("*.sql"))


def test_ac1_migrate_applies_all_migrations_once(database_url):
    files = _migration_files()
    assert files, "expected at least one NNN_*.sql migration"

    applied = migrate(database_url)

    assert [Path(n).stem for n in applied] == [f.stem for f in files]
    assert migrate(database_url) == []


def test_ac1_migrate_creates_schema_and_records_versions(database_url):
    import psycopg

    applied = migrate(database_url)

    with psycopg.connect(database_url) as conn:
        recorded = conn.execute("SELECT count(*) FROM schema_migrations").fetchone()[0]
        tables = conn.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name <> 'schema_migrations'"
        ).fetchone()[0]
    assert recorded == len(applied)
    assert tables > 0


def test_ac2_health_reports_ok(client):
    r = client.get("/v1/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "database": "ok"}


def _token_settings(settings) -> Settings:
    return Settings(database_url=settings.database_url, data_dir=settings.data_dir, api_tokens=(TOKEN,))


def test_ac2_health_needs_no_token_when_auth_enabled(settings, registry):
    with make_client(_token_settings(settings), registry) as c:
        r = c.get("/v1/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "database": "ok"}


PROTECTED_GETS = ["/v1/pipelines", "/v1/workspaces", "/v1/models"]


@pytest.mark.parametrize("path", PROTECTED_GETS)
def test_ac3_routes_reject_missing_or_wrong_token(settings, registry, path):
    with make_client(_token_settings(settings), registry) as c:
        assert c.get(path).status_code == 401
        assert c.get(path, headers=bearer("wrong-token")).status_code == 401


@pytest.mark.parametrize("path", PROTECTED_GETS)
def test_ac3_routes_accept_valid_token(settings, registry, path):
    with make_client(_token_settings(settings), registry) as c:
        assert c.get(path, headers=bearer(TOKEN)).status_code == 200


def test_ac3_post_requires_token(settings, registry):
    with make_client(_token_settings(settings), registry) as c:
        assert c.post("/v1/workspaces", json={"name": "a"}).status_code == 401
        r = c.post("/v1/workspaces", json={"name": "a"}, headers=bearer(TOKEN))
        assert r.status_code == 201, r.text


@pytest.mark.parametrize("path", PROTECTED_GETS)
def test_ac3_no_tokens_configured_means_no_auth(client, path):
    assert client.get(path).status_code == 200
