"""Helpers for the Phase 2A service tests (auth, review, deployments, Studio connector).

Like ``helpers.py``, nothing here imports ``sparrow_finetune`` at module level: the
Phase 2A modules (``sparrow_finetune.auth``, ``sparrow_finetune.connectors.studio``)
do not exist before implementation, and a missing module must make the individual
test FAIL, not break collection of the whole suite.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import threading
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from .helpers import as_list

CSRF = {"X-Sparrow-CSRF": "1"}
STUDIO_URL = "http://studio.test"
STUDIO_TOKEN = "stok"
PASSWORD = "correct-horse-battery"
SESSION_COOKIE = "sft_session"


@contextmanager
def make_service_client(settings, registry, studio_transport=None) -> Iterator[Any]:
    """TestClient over ``create_app``; ``studio_transport`` is passed only when given,
    so tests that do not need Studio also exercise the Phase 1 call signature."""
    from fastapi.testclient import TestClient

    from sparrow_finetune.api import create_app

    kwargs: dict = {"registry": registry}
    if studio_transport is not None:
        kwargs["studio_transport"] = studio_transport
    with TestClient(create_app(settings, **kwargs)) as client:
        yield client


def studio_settings(settings, **extra):
    """Settings with the Studio connector configured (TypeError until 2A adds the fields)."""
    return dataclasses.replace(
        settings, studio_url=STUDIO_URL, studio_token=STUDIO_TOKEN, **extra
    )


def db(settings):
    import psycopg
    from psycopg.rows import dict_row

    return psycopg.connect(settings.database_url, autocommit=True, row_factory=dict_row)


def add_user(settings, username: str = "alice", password: str = PASSWORD) -> None:
    """Insert a user directly (hash via ``sparrow_finetune.auth.hash_password``)."""
    from sparrow_finetune.auth import hash_password

    with db(settings) as conn:
        conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
            (username, hash_password(password)),
        )


def login(client, username: str = "alice", password: str = PASSWORD):
    r = client.post("/v1/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    assert r.json()["username"] == username
    assert client.cookies.get(SESSION_COOKIE), "login did not set the sft_session cookie"
    return r


def audit_rows(client, **params) -> list:
    r = client.get("/v1/audit", params=params)
    assert r.status_code == 200, r.text
    return as_list(r.json(), "events", "audit")


def audit_actions(rows: list) -> list:
    return [row["action"] for row in rows]


def token_principal(token: str) -> str:
    return "token:" + hashlib.sha256(token.encode()).hexdigest()[:8]


def sha256_matches(stored: Any, value: str) -> bool:
    """``sessions.token_sha256`` may be stored as hex text or raw bytes."""
    digest = hashlib.sha256(value.encode())
    if isinstance(stored, (bytes, bytearray, memoryview)):
        return bytes(stored) == digest.digest()
    return str(stored).lower() == digest.hexdigest()


# --------------------------------------------------------------------------- SSE


def read_sse(
    client,
    url: str,
    headers: Optional[dict] = None,
    timeout: float = 15.0,
    max_lines: int = 5000,
) -> dict:
    """Read a server-sent-event stream to its end in a worker thread.

    Returns ``{"status", "content_type", "events": [{id, event, data}]}``. A stream
    that does not close within ``timeout`` fails the test instead of hanging it.
    """
    out: dict = {"status": None, "content_type": None, "lines": [], "error": None}

    def run() -> None:
        try:
            with client.stream("GET", url, headers=headers or {}) as r:
                out["status"] = r.status_code
                out["content_type"] = r.headers.get("content-type", "")
                if r.status_code != 200:
                    r.read()
                    return
                for line in r.iter_lines():
                    out["lines"].append(line)
                    if len(out["lines"]) >= max_lines:
                        return
        except BaseException as exc:  # surfaced to the test below
            out["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise AssertionError(
            f"SSE stream {url} did not close within {timeout}s; "
            f"first lines: {out['lines'][:50]!r}"
        )
    if out["error"] is not None:
        raise out["error"]
    assert len(out["lines"]) < max_lines, "SSE stream produced too many lines"
    out["events"] = parse_sse(out["lines"])
    return out


def parse_sse(lines: list) -> list:
    """Blank-line separated blocks -> ``[{"id"?, "event"?, "data": parsed JSON}]``."""
    events: list = []
    current: dict = {}
    data: list = []
    for raw in [*lines, ""]:
        line = raw.rstrip("\r")
        if line == "":
            if current or data:
                current["data"] = json.loads("\n".join(data)) if data else None
                events.append(current)
            current, data = {}, []
            continue
        if line.startswith(":"):
            continue  # comment / keep-alive
        key, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if key == "data":
            data.append(value)
        elif key in ("id", "event", "retry"):
            current[key] = value
    return events
