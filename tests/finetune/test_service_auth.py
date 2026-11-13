"""Phase 2A acceptance tests A1-A6: passwords, create-user CLI, sessions, CSRF, audit.

Strategy: auth is a security boundary, so every rule the spec states is asserted
over HTTP against a real PostgreSQL (``settings`` fixture): when auth switches on
(tokens or >=1 user), each way in (bearer, session cookie) and each way it must be
refused (no credential, missing CSRF header, logged-out or expired session). Users
are inserted with ``sparrow_finetune.auth.hash_password`` so the HTTP tests do not
depend on the CLI; the CLI has its own tests (A2). Phase 1 open-API behaviour is
kept by explicit guard tests (they pass before and after 2A).

Spec: docs/review/implement-phase2-service/spec.md, section 2A ("Auth", "Audit").
"""

from __future__ import annotations

import dataclasses
import io
import sys
from datetime import timedelta

import pytest

from .helpers import bearer, create_workspace
from .service_helpers import (
    CSRF,
    PASSWORD,
    SESSION_COOKIE,
    add_user,
    audit_rows,
    db,
    login,
    make_service_client,
    sha256_matches,
    token_principal,
)

TOKEN = "tok-phase2a-0123456789"


# --------------------------------------------------------------------------- A1


def test_a1_hash_verify_roundtrip_and_format():
    from sparrow_finetune.auth import hash_password, verify_password

    stored = hash_password("s3cret-pass")
    scheme, n, r, p, salt_hex, hash_hex = stored.split("$")
    assert (scheme, n, r, p) == ("scrypt", str(2**14), "8", "1")
    assert len(bytes.fromhex(salt_hex)) == 16
    assert bytes.fromhex(hash_hex)
    assert verify_password("s3cret-pass", stored) is True
    assert hash_password("s3cret-pass") != stored, "salt must be random per hash"


def test_a1_wrong_password_is_rejected():
    from sparrow_finetune.auth import hash_password, verify_password

    stored = hash_password("s3cret-pass")
    assert verify_password("s3cret-pasS", stored) is False
    assert verify_password("", stored) is False


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s[:-2] + ("00" if s[-2:] != "00" else "11"),  # tampered hash
        lambda s: s.replace(s.split("$")[4], "00" * 16),  # tampered salt
        lambda s: "",
        lambda s: "not-a-hash",
        lambda s: "scrypt$16384$8$1$zz$zz",  # non-hex
        lambda s: "scrypt$abc$8$1$" + "00" * 16 + "$" + "00" * 64,  # bad n
        lambda s: "bcrypt" + s[len("scrypt"):],  # unknown scheme
        lambda s: "$".join(s.split("$")[:5]),  # missing field
    ],
    ids=["tampered-hash", "tampered-salt", "empty", "garbage", "non-hex", "bad-n",
         "other-scheme", "truncated"],
)
def test_a1_tampered_or_malformed_hash_is_false_not_error(mutate):
    from sparrow_finetune.auth import hash_password, verify_password

    assert verify_password("s3cret-pass", mutate(hash_password("s3cret-pass"))) is False


# --------------------------------------------------------------------------- A2


def run_cli(argv) -> int:
    from sparrow_finetune.cli import main

    try:
        return main(argv)
    except SystemExit as exc:  # argparse or sys.exit inside the command
        return exc.code if isinstance(exc.code, int) else 1


@pytest.fixture
def cli_env(settings, monkeypatch):
    monkeypatch.setenv("SPARROW_FT_DATABASE_URL", settings.database_url)
    monkeypatch.setenv("SPARROW_FT_DATA_DIR", str(settings.data_dir))
    monkeypatch.delenv("SPARROW_FT_PASSWORD", raising=False)
    return settings


def user_hash(settings, username):
    with db(settings) as conn:
        row = conn.execute(
            "SELECT password_hash FROM users WHERE username = %s", (username,)
        ).fetchone()
    return None if row is None else row["password_hash"]


def test_a2_create_user_from_env_password(cli_env, monkeypatch):
    from sparrow_finetune.auth import verify_password

    monkeypatch.setenv("SPARROW_FT_PASSWORD", PASSWORD)
    assert run_cli(["create-user", "alice"]) == 0
    stored = user_hash(cli_env, "alice")
    assert stored is not None and stored.startswith("scrypt$")
    assert verify_password(PASSWORD, stored) is True


def test_a2_create_user_from_stdin_first_line(cli_env, monkeypatch):
    from sparrow_finetune.auth import verify_password

    monkeypatch.setattr(sys, "stdin", io.StringIO("stdin-password-1\nignored\n"))
    assert run_cli(["create-user", "bob"]) == 0
    assert verify_password("stdin-password-1", user_hash(cli_env, "bob")) is True


def test_a2_duplicate_username_exits_1_and_keeps_original(cli_env, monkeypatch):
    from sparrow_finetune.auth import verify_password

    monkeypatch.setenv("SPARROW_FT_PASSWORD", PASSWORD)
    assert run_cli(["create-user", "alice"]) == 0
    monkeypatch.setenv("SPARROW_FT_PASSWORD", "another-password")
    assert run_cli(["create-user", "alice"]) == 1
    assert verify_password(PASSWORD, user_hash(cli_env, "alice")) is True


def test_a2_short_password_exits_2_and_creates_nothing(cli_env, monkeypatch):
    # First prove the command exists (an argparse error also exits 2).
    monkeypatch.setenv("SPARROW_FT_PASSWORD", PASSWORD)
    assert run_cli(["create-user", "alice"]) == 0
    monkeypatch.setenv("SPARROW_FT_PASSWORD", "1234567")  # 7 chars
    assert run_cli(["create-user", "carol"]) == 2
    assert user_hash(cli_env, "carol") is None


# --------------------------------------------------------------------------- A3


def test_a3_guard_open_api_without_tokens_or_users(settings, registry):
    """GUARD (passes before 2A): Phase 1 behaviour -- no tokens, no users -> open API,
    including writes without any CSRF header."""
    with make_service_client(settings, registry) as client:
        assert client.get("/v1/workspaces").status_code == 200
        create_workspace(client)
        assert client.get("/v1/health").status_code == 200


def test_a3_auth_me_reports_open(settings, registry):
    with make_service_client(settings, registry) as client:
        r = client.get("/v1/auth/me")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["method"] == "open"
        assert body["auth_required"] is False
        assert body["username"] is None


def test_a3_tokens_only_require_auth_and_me_never_401(settings, registry):
    tokened = dataclasses.replace(settings, api_tokens=(TOKEN,))
    with make_service_client(tokened, registry) as client:
        assert client.get("/v1/workspaces").status_code == 401
        r = client.get("/v1/auth/me")
        assert r.status_code == 200, r.text
        assert r.json() == {"username": None, "method": None, "auth_required": True}


# --------------------------------------------------------------------------- A4


@pytest.fixture
def user_client(settings, registry):
    add_user(settings, "alice")
    with make_service_client(settings, registry) as client:
        yield client


def test_a4_user_present_requires_auth_but_health_stays_open(user_client):
    assert user_client.get("/v1/workspaces").status_code == 401
    assert user_client.post("/v1/workspaces", json={"name": "w"}).status_code == 401
    assert user_client.get("/v1/health").status_code == 200
    me = user_client.get("/v1/auth/me")
    assert me.status_code == 200
    assert me.json() == {"username": None, "method": None, "auth_required": True}


def test_a4_login_sets_session_cookie_with_flags(user_client, settings):
    r = login(user_client)
    cookie = r.headers["set-cookie"]
    assert cookie.startswith(f"{SESSION_COOKIE}=")
    flags = [f.strip().lower() for f in cookie.split(";")]
    assert "httponly" in flags
    assert "samesite=strict" in flags
    assert "path=/" in flags
    assert "secure" not in flags  # SPARROW_FT_COOKIE_SECURE defaults off
    value = user_client.cookies.get(SESSION_COOKIE)
    assert len(value) >= 43  # 32 random bytes, urlsafe base64
    with db(settings) as conn:
        rows = conn.execute(
            "SELECT s.token_sha256, s.created_at, s.expires_at FROM sessions s"
            " JOIN users u ON u.id = s.user_id WHERE u.username = 'alice'"
        ).fetchall()
    assert len(rows) == 1
    assert sha256_matches(rows[0]["token_sha256"], value), "store sha256 of the cookie, not the value"
    ttl = rows[0]["expires_at"] - rows[0]["created_at"]
    assert timedelta(hours=11, minutes=55) < ttl < timedelta(hours=12, minutes=5)


def test_a4_cookie_secure_and_session_hours_settings(settings, registry):
    add_user(settings, "alice")
    tuned = dataclasses.replace(settings, cookie_secure=True, session_hours=1.5)
    with make_service_client(tuned, registry) as client:
        r = client.post("/v1/auth/login", json={"username": "alice", "password": PASSWORD})
        assert r.status_code == 200, r.text
        flags = [f.strip().lower() for f in r.headers["set-cookie"].split(";")]
        assert "secure" in flags
    with db(settings) as conn:
        row = conn.execute("SELECT created_at, expires_at FROM sessions").fetchone()
    ttl = row["expires_at"] - row["created_at"]
    assert timedelta(minutes=85) < ttl < timedelta(minutes=95)


def test_a4_session_cookie_authenticates_get_and_me(user_client):
    login(user_client)
    assert user_client.get("/v1/workspaces").status_code == 200
    me = user_client.get("/v1/auth/me")
    assert me.json() == {"username": "alice", "method": "session", "auth_required": True}


def test_a4_cookie_write_needs_csrf_header(user_client):
    login(user_client)
    r = user_client.post("/v1/workspaces", json={"name": "no-csrf"})
    assert r.status_code == 403, r.text
    r = user_client.post("/v1/workspaces", json={"name": "csrf"}, headers=CSRF)
    assert r.status_code == 201, r.text
    names = [w["name"] for w in user_client.get("/v1/workspaces").json()["workspaces"]]
    assert names == ["csrf"], "the refused write must not have been applied"


def test_a4_logout_deletes_session_and_rejects_cookie(user_client, settings):
    login(user_client)
    value = user_client.cookies.get(SESSION_COOKIE)
    r = user_client.post("/v1/auth/logout", headers=CSRF)
    assert r.status_code == 204, r.text
    assert SESSION_COOKIE in r.headers.get("set-cookie", ""), "logout must clear the cookie"
    with db(settings) as conn:
        assert conn.execute("SELECT count(*) AS n FROM sessions").fetchone()["n"] == 0
    user_client.cookies.set(SESSION_COOKIE, value)  # replay the old cookie
    assert user_client.get("/v1/workspaces").status_code == 401


def test_a4_expired_session_is_rejected(user_client, settings):
    login(user_client)
    assert user_client.get("/v1/workspaces").status_code == 200
    with db(settings) as conn:
        conn.execute("UPDATE sessions SET expires_at = now() - interval '1 minute'")
    assert user_client.get("/v1/workspaces").status_code == 401
    assert user_client.get("/v1/auth/me").json()["method"] is None


def test_a4_forged_cookie_is_rejected(user_client):
    user_client.cookies.set(SESSION_COOKIE, "x" * 43)
    assert user_client.get("/v1/workspaces").status_code == 401


# --------------------------------------------------------------------------- A5


def test_a5_login_failures_are_401_with_identical_message(user_client):
    unknown = user_client.post("/v1/auth/login", json={"username": "mallory", "password": PASSWORD})
    wrong = user_client.post("/v1/auth/login", json={"username": "alice", "password": "wrong-password"})
    assert unknown.status_code == 401, unknown.text
    assert wrong.status_code == 401, wrong.text
    assert unknown.json() == wrong.json()
    assert SESSION_COOKIE not in user_client.cookies


def test_a5_login_attempts_are_audited(user_client):
    user_client.post("/v1/auth/login", json={"username": "mallory", "password": PASSWORD})
    user_client.post("/v1/auth/login", json={"username": "alice", "password": "wrong-password"})
    login(user_client)
    logins = [row for row in audit_rows(user_client) if row["action"] == "login"]
    got = [(row["principal"], row["detail"]) for row in logins]
    # newest first
    assert got == [
        ("alice", {"ok": True}),
        ("alice", {"ok": False}),
        ("mallory", {"ok": False}),
    ]


# --------------------------------------------------------------------------- A6


def test_a6_bearer_works_with_users_and_needs_no_csrf(settings, registry):
    add_user(settings, "alice")
    tokened = dataclasses.replace(settings, api_tokens=(TOKEN,))
    with make_service_client(tokened, registry) as client:
        assert client.get("/v1/workspaces").status_code == 401
        h = bearer(TOKEN)
        assert client.get("/v1/workspaces", headers=h).status_code == 200
        r = client.post("/v1/workspaces", json={"name": "w"}, headers=h)
        assert r.status_code == 201, r.text
        me = client.get("/v1/auth/me", headers=h).json()
        assert me["method"] == "token"
        assert me["auth_required"] is True


def test_a6_token_principal_in_audit(settings, registry):
    from .helpers import build_snapshot, queued_job

    tokened = dataclasses.replace(settings, api_tokens=(TOKEN,))
    with make_service_client(tokened, registry) as client:
        client.headers.update(bearer(TOKEN))
        built = build_snapshot(client)
        job_id = queued_job(client, "fake_cls", built["snapshot"]["id"])
        rows = [r for r in audit_rows(client) if r["action"] == "job.submit"]
    assert len(rows) == 1
    assert rows[0]["principal"] == token_principal(TOKEN)
    assert str(job_id) in str(rows[0]["target"])


# --------------------------------------------------------------------------- audit endpoint


def test_audit_anonymous_principal_newest_first_and_limit(settings, registry):
    from .helpers import build_snapshot, queued_job

    with make_service_client(settings, registry) as client:
        built = build_snapshot(client)
        first = queued_job(client, "fake_cls", built["snapshot"]["id"])
        second = queued_job(client, "fake_cls", built["snapshot"]["id"])
        assert client.post(f"/v1/jobs/{second}/cancel").status_code == 200
        rows = audit_rows(client)
        assert [r["action"] for r in rows] == ["job.cancel", "job.submit", "job.submit"]
        assert {r["principal"] for r in rows} == {"anonymous"}
        assert str(second) in str(rows[0]["target"])
        assert str(first) in str(rows[2]["target"])
        ids = [r["id"] for r in rows]
        assert ids == sorted(ids, reverse=True)
        assert len(audit_rows(client, limit=1)) == 1
        assert audit_rows(client, limit=1)[0]["action"] == "job.cancel"
        assert len(audit_rows(client, limit=1000)) == 3
        assert client.get("/v1/audit", params={"limit": 0}).status_code == 422
        assert client.get("/v1/audit", params={"limit": 1001}).status_code == 422


def test_audit_requires_auth_when_users_exist(user_client):
    assert user_client.get("/v1/audit").status_code == 401
