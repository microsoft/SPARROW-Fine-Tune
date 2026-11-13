"""PostgreSQL access (psycopg 3) and the SQL migration runner."""

from __future__ import annotations

import ipaddress
import os
from pathlib import Path
from typing import Mapping, Optional
from urllib.parse import quote

import psycopg
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import dict_row

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
# Two-key advisory lock (namespace, id): serialises concurrent `migrate` calls. Postgres keeps
# two-int4 keys apart from single-bigint keys, so this cannot collide with an app's bigint locks.
_LOCK_NAMESPACE = 0x5350_4654  # "SPFT"
_MIGRATE_LOCK_KEY = 1
# Stored images are content-addressed, so datasets and snapshots can share one object. Code
# that finds an object present and then commits a row pointing at it holds this lock shared;
# deleting a dataset holds it exclusively while it drops rows and the objects left unused.
OBJECTS_LOCK = (_LOCK_NAMESPACE, 2)
CONNECT_TIMEOUT_S = 10
# Server-side cap on any one statement, so a lock wait or runaway query cannot hang a worker
# (or an API thread) forever. Override with SPARROW_FT_STATEMENT_TIMEOUT_MS (0 disables).
STATEMENT_TIMEOUT_ENV = "SPARROW_FT_STATEMENT_TIMEOUT_MS"
DEFAULT_STATEMENT_TIMEOUT_MS = 60_000


def statement_timeout_ms() -> int:
    raw = os.environ.get(STATEMENT_TIMEOUT_ENV, "").strip()
    if not raw:
        return DEFAULT_STATEMENT_TIMEOUT_MS
    try:
        value = int(raw)
    except ValueError:
        raise RuntimeError(
            f"{STATEMENT_TIMEOUT_ENV} must be an integer number of milliseconds, got {raw!r}"
        ) from None
    if value < 0:
        raise RuntimeError(f"{STATEMENT_TIMEOUT_ENV} must be >= 0, got {value}")
    return value


def connect(database_url: str) -> psycopg.Connection:
    """A new connection returning dict rows; use as a context manager (commit on success)."""
    # An explicit ``options`` replaces the DSN's and makes libpq ignore PGOPTIONS, so
    # keep whichever of those applies and append the timeout to it.
    base = conninfo_to_dict(database_url).get("options")
    if base is None:
        base = os.environ.get("PGOPTIONS", "")
    timeout = f"-c statement_timeout={statement_timeout_ms()}"
    return psycopg.connect(
        database_url,
        row_factory=dict_row,
        connect_timeout=CONNECT_TIMEOUT_S,
        options=f"{base} {timeout}".strip(),
    )


def _is_ipv6(text: str) -> bool:
    try:
        ipaddress.IPv6Address(text)
    except ValueError:
        return False
    return True


def _split_host(host: str) -> tuple[str, Optional[str]]:
    """``SPARROW_FT_DB_HOST`` -> (URL host, port or None).

    Accepts ``name``, ``name:port``, ``1.2.3.4:port``, a bare IPv6 literal (``::1``) and a
    bracketed one with an optional port (``[::1]:5433``). Only real IPv6 literals are
    bracketed; anything else with more than one colon is rejected."""

    def check_port(port: str) -> str:
        if not port.isdigit():
            raise RuntimeError(f"SPARROW_FT_DB_HOST has an invalid port {port!r}")
        return port

    if host.startswith("["):
        end = host.find("]")
        literal = host[1:end] if end > 0 else ""
        if not _is_ipv6(literal):
            raise RuntimeError(
                f"SPARROW_FT_DB_HOST {host!r} is not a bracketed IPv6 literal"
            )
        rest = host[end + 1 :]
        if rest and not rest.startswith(":"):
            raise RuntimeError(f"SPARROW_FT_DB_HOST {host!r} is malformed")
        port = check_port(rest[1:]) if rest else None
        return f"[{quote(literal, safe=':')}]", port
    if _is_ipv6(host):
        return f"[{quote(host, safe=':')}]", None  # quote a zone id's "%"
    if host.count(":") > 1:
        raise RuntimeError(
            f"SPARROW_FT_DB_HOST {host!r} is neither a host name nor an IPv6 literal"
        )
    if ":" in host:
        name, port = host.split(":")
        if not name:
            raise RuntimeError(f"SPARROW_FT_DB_HOST {host!r} has no host name")
        return name, check_port(port)
    return host, None


def build_database_url(env: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """``postgresql://`` URL from ``SPARROW_FT_DB_*`` parts, or ``None`` if they are unset.

    User, password and database name are percent-encoded, so a password may contain
    ``@:/%`` and the like. Parts: ``SPARROW_FT_DB_HOST`` (required), ``SPARROW_FT_DB_PORT``
    (5432), ``SPARROW_FT_DB_NAME`` (sparrow_finetune), ``SPARROW_FT_DB_USER`` (sparrow),
    ``SPARROW_FT_DB_PASSWORD`` (empty: no password). The host may be a name, an IPv4
    address or an IPv6 literal (bare or ``[bracketed]``), optionally with a ``:port``
    (``db:5433``, ``[::1]:5433``); an embedded port must agree with ``SPARROW_FT_DB_PORT``.
    """
    env = os.environ if env is None else env
    host = env.get("SPARROW_FT_DB_HOST", "").strip()
    if not host:
        return None
    host, host_port = _split_host(host)
    explicit = env.get("SPARROW_FT_DB_PORT", "").strip()
    if explicit and not explicit.isdigit():
        raise RuntimeError(
            f"SPARROW_FT_DB_PORT must be a port number, got {explicit!r}"
        )
    if host_port and explicit and int(host_port) != int(explicit):
        raise RuntimeError(
            f"SPARROW_FT_DB_HOST carries port {host_port} but SPARROW_FT_DB_PORT is {explicit}"
        )
    port = host_port or explicit or "5432"
    name = env.get("SPARROW_FT_DB_NAME", "").strip() or "sparrow_finetune"
    user = env.get("SPARROW_FT_DB_USER", "").strip() or "sparrow"
    password = env.get("SPARROW_FT_DB_PASSWORD", "")
    auth = quote(user, safe="")
    if password:
        auth += ":" + quote(password, safe="")
    return f"postgresql://{auth}@{host}:{port}/{quote(name, safe='')}"


def migration_files() -> list[Path]:
    return sorted(MIGRATIONS_DIR.glob("[0-9][0-9][0-9]_*.sql"))


def migrate(database_url: str) -> list[str]:
    """Apply pending migrations in order; return the names (file stems) newly applied."""
    applied: list[str] = []
    with psycopg.connect(
        database_url, autocommit=True, connect_timeout=CONNECT_TIMEOUT_S
    ) as conn:
        # Lock before any DDL: concurrent CREATE TABLE IF NOT EXISTS races on pg_type.
        conn.execute(
            "SELECT pg_advisory_lock(%s, %s)", (_LOCK_NAMESPACE, _MIGRATE_LOCK_KEY)
        )
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                " name text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
            )
            done = {
                r[0]
                for r in conn.execute("SELECT name FROM schema_migrations").fetchall()
            }
            for path in migration_files():
                name = path.stem
                if name in done:
                    continue
                with conn.transaction():
                    conn.execute(path.read_text(encoding="utf-8"))
                    conn.execute(
                        "INSERT INTO schema_migrations (name) VALUES (%s)", (name,)
                    )
                applied.append(name)
        finally:
            try:
                conn.execute(
                    "SELECT pg_advisory_unlock(%s, %s)",
                    (_LOCK_NAMESPACE, _MIGRATE_LOCK_KEY),
                )
            except psycopg.Error:
                # Never mask the original error; the session lock ends with the connection.
                pass
    return applied
