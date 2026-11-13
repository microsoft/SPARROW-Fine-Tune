"""Fixtures for the ``sparrow_finetune`` Phase 1A acceptance tests.

Strategy: every test talks to a real PostgreSQL (the queue relies on
``FOR UPDATE SKIP LOCKED`` and leases, which a fake would not prove). One server
per session -- ``SPARROW_FT_TEST_DATABASE_URL`` if set, otherwise a throwaway
``postgres:16-alpine`` container -- and one freshly created database per test, so
tests never see each other's rows. Missing docker is a hard failure, not a skip:
a skipped database suite would read as green while proving nothing.

``sparrow_finetune``, ``psycopg`` and ``fastapi`` are imported inside fixtures so
this module itself always imports.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pytest

PG_IMAGE = "postgres:16-alpine"
PG_PASSWORD = "sparrow-ft-test"
READY_TIMEOUT_S = 60.0


# --------------------------------------------------------------------------- plain helpers


def with_dbname(url: str, dbname: str) -> str:
    return urlunsplit(urlsplit(url)._replace(path="/" + dbname))


def wait_for_postgres(url: str, timeout: float = READY_TIMEOUT_S) -> None:
    import psycopg

    deadline = time.monotonic() + timeout
    last: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with psycopg.connect(url, connect_timeout=2) as conn:
                conn.execute("SELECT 1")
            return
        except psycopg.OperationalError as exc:
            last = exc
            time.sleep(0.5)
    raise TimeoutError(f"PostgreSQL at {url} not ready after {timeout:.0f}s: {last}")


def remove_container(name: str) -> None:
    subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True, timeout=60)


def start_postgres_container() -> tuple[str, str]:
    """Start a throwaway server; return (container name, admin URL to db ``postgres``)."""
    if shutil.which("docker") is None:
        raise RuntimeError("the `docker` CLI is not on PATH")
    name = f"sparrow-ft-test-{uuid.uuid4().hex[:12]}"
    subprocess.run(
        [
            "docker", "run", "-d", "--rm", "--name", name,
            "-e", f"POSTGRES_PASSWORD={PG_PASSWORD}",
            "-p", "127.0.0.1::5432",  # docker picks a free host port
            PG_IMAGE,
        ],
        check=True, capture_output=True, text=True, timeout=120,
    )
    try:
        out = subprocess.run(
            ["docker", "port", name, "5432/tcp"],
            check=True, capture_output=True, text=True, timeout=30,
        ).stdout
        port = out.strip().splitlines()[0].rsplit(":", 1)[1]
        url = f"postgresql://postgres:{PG_PASSWORD}@127.0.0.1:{port}/postgres"
        wait_for_postgres(url)
    except BaseException:
        remove_container(name)
        raise
    return name, url


def create_database(admin_url: str) -> tuple[str, str]:
    import psycopg
    from psycopg import sql

    dbname = f"ft_test_{uuid.uuid4().hex[:16]}"
    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname)))
    return dbname, with_dbname(admin_url, dbname)


def drop_database(admin_url: str, dbname: str) -> None:
    import psycopg
    from psycopg import sql

    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute(
            sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(dbname))
        )


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="session")
def pg_admin_url():
    env_url = os.environ.get("SPARROW_FT_TEST_DATABASE_URL")
    if env_url:
        admin = with_dbname(env_url, "postgres")
        wait_for_postgres(admin, timeout=15)
        yield admin
        return
    try:
        name, url = start_postgres_container()
    except subprocess.CalledProcessError as exc:
        pytest.fail(
            f"could not start {PG_IMAGE} via docker ({exc.stderr.strip()}); "
            "set SPARROW_FT_TEST_DATABASE_URL to an existing server instead",
            pytrace=False,
        )
    except Exception as exc:  # docker missing, timeout, ...
        pytest.fail(
            f"no PostgreSQL for the sparrow_finetune tests: {exc}; "
            "install/start docker or set SPARROW_FT_TEST_DATABASE_URL",
            pytrace=False,
        )
    try:
        yield url
    finally:
        remove_container(name)


@pytest.fixture
def database_url(pg_admin_url):
    """URL of a fresh, empty database; dropped after the test."""
    dbname, url = create_database(pg_admin_url)
    try:
        yield url
    finally:
        drop_database(pg_admin_url, dbname)


@pytest.fixture
def data_dir(tmp_path) -> Path:
    path = tmp_path / "data"
    path.mkdir()
    return path


@pytest.fixture
def settings(database_url, data_dir):
    from sparrow_finetune.config import Settings
    from sparrow_finetune.db import migrate

    migrate(database_url)
    return Settings(database_url=database_url, data_dir=data_dir)


@pytest.fixture
def trainers():
    from .fakes import FakeTrainers

    return FakeTrainers()


@pytest.fixture
def registry(trainers):
    from sparrow_finetune.pipelines import Registry

    reg = Registry()
    for trainer in trainers.all():
        reg.register(trainer)
    return reg


@pytest.fixture
def client(settings, registry):
    from .helpers import make_client

    with make_client(settings, registry) as c:
        yield c


@pytest.fixture
def queue(settings):
    from sparrow_finetune.jobs import JobQueue

    return JobQueue(settings.database_url)
