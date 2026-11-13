"""Authentication (bearer tokens, session cookies + CSRF header), auth routes and the audit trail."""

from __future__ import annotations

import hmac
import ipaddress
import math
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from ..auth import (
    new_session_token,
    token_principal,
    token_sha256,
    verify_password,
)
from ..db import connect

SESSION_COOKIE = "sft_session"
CSRF_HEADER = "X-Sparrow-CSRF"
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_LOGIN_FAILED = "invalid username or password"
LOGIN_MAX_FAILURES = 10
LOGIN_WINDOW_SECONDS = 300.0
# Verified against when the username is unknown, so both failure paths cost one scrypt.
_DUMMY_HASH = (
    "scrypt$16384$8$1$00000000000000000000000000000000$"
    "0000000000000000000000000000000000000000000000000000000000000000"
)

_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class Principal:
    name: str
    method: str  # "token" | "session" | "open"
    session_id: Optional[int] = None


def _url(request: Request) -> str:
    return request.app.state.settings.database_url


def _valid_token(
    request: Request, creds: Optional[HTTPAuthorizationCredentials]
) -> Optional[str]:
    tokens = request.app.state.settings.api_tokens
    if creds is None or not tokens:
        return None
    given = creds.credentials.encode()
    ok = False
    for token in tokens:  # no short-circuit: compare against every token
        ok |= hmac.compare_digest(given, token.encode())
    return creds.credentials if ok else None


def _session_user(conn, cookie: Optional[str]) -> Optional[dict]:
    if not cookie:
        return None
    return conn.execute(
        "SELECT s.id AS session_id, u.username FROM sessions s JOIN users u ON u.id = s.user_id"
        " WHERE s.token_sha256 = %s AND s.expires_at > now()",
        (token_sha256(cookie),),
    ).fetchone()


def _users_exist(conn) -> bool:
    return conn.execute("SELECT EXISTS (SELECT 1 FROM users) AS e").fetchone()["e"]


def resolve_principal(
    request: Request, creds: Optional[HTTPAuthorizationCredentials]
) -> tuple[Optional[Principal], bool]:
    """(principal or None, auth_required). Never raises for missing credentials."""
    token = _valid_token(request, creds)
    if token is not None:
        return Principal(token_principal(token), "token"), True
    with connect(_url(request)) as conn:
        row = _session_user(conn, request.cookies.get(SESSION_COOKIE))
        if row is not None:
            return Principal(row["username"], "session", row["session_id"]), True
        settings = request.app.state.settings
        required = not settings.auth_disabled and (
            bool(settings.api_tokens) or _users_exist(conn)
        )
    if not required:
        return Principal("anonymous", "open"), False
    return None, True


def require_auth(
    request: Request, creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer)
) -> Principal:
    principal, _ = resolve_principal(request, creds)
    if principal is None:
        raise HTTPException(
            401,
            "authentication required: bearer token or login session",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if (
        principal.method == "session"
        and request.method not in _SAFE_METHODS
        and request.headers.get(CSRF_HEADER) != "1"
    ):
        raise HTTPException(403, f"missing {CSRF_HEADER}: 1 header")
    request.state.principal = principal
    return principal


def principal_name(request: Request) -> str:
    principal = getattr(request.state, "principal", None)
    return principal.name if principal is not None else "anonymous"


def record_audit(
    conn, principal: str, action: str, target: str = "", detail: Optional[dict] = None
) -> None:
    conn.execute(
        "INSERT INTO audit_events (principal, action, target, detail) VALUES (%s, %s, %s, %s)",
        (principal, action, target, Jsonb(detail or {})),
    )


def audit(
    request: Request, action: str, target: str = "", detail: Optional[dict] = None
) -> None:
    with connect(_url(request)) as conn:
        record_audit(conn, principal_name(request), action, target, detail)


# --------------------------------------------------------------------------- routes

auth_router = APIRouter()


class LoginIn(BaseModel):
    username: str = Field(max_length=200)
    password: str = Field(max_length=1024)


class LoginLimiter:
    """In-process per-client limit on failed logins (sliding window)."""

    def __init__(
        self,
        max_failures: int = LOGIN_MAX_FAILURES,
        window: float = LOGIN_WINDOW_SECONDS,
    ):
        self.max_failures = max_failures
        self.window = window
        # Ordered by last recorded attempt (oldest first), so expired keys sit at the front.
        self._failures: OrderedDict[str, deque] = OrderedDict()
        self._lock = threading.Lock()

    def _trim(self, key: str, now: float) -> deque:
        q = self._failures.setdefault(key, deque())
        while q and q[0] <= now - self.window:
            q.popleft()
        return q

    def retry_after(self, key: str) -> Optional[int]:
        """Seconds until the client may try again, or None when not limited."""
        now = time.monotonic()
        with self._lock:
            q = self._trim(key, now)
            if len(q) < self.max_failures:
                if not q:
                    self._failures.pop(key, None)
                return None
            return max(1, math.ceil(q[0] + self.window - now))

    def begin(self, key: str) -> tuple[Optional[int], Optional[float]]:
        """Check and count one attempt atomically: ``(retry_after, None)`` when limited,
        else ``(None, stamp)``. The attempt counts as a failure until ``succeeded`` drops
        it, so concurrent attempts cannot all pass the check before any is recorded."""
        now = time.monotonic()
        with self._lock:
            q = self._trim(key, now)
            if len(q) >= self.max_failures:
                return max(1, math.ceil(q[0] + self.window - now)), None
            q.append(now)
            self._failures.move_to_end(key)
            self._bound(now)
            return None, now

    def succeeded(self, key: str, stamp: float) -> None:
        with self._lock:
            q = self._failures.get(key)
            if q is not None:
                try:
                    q.remove(stamp)
                except ValueError:
                    pass
                if not q:
                    self._failures.pop(key, None)

    def failure(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            self._trim(key, now).append(now)
            self._failures.move_to_end(key)
            self._bound(now)

    def _bound(self, now: float) -> None:
        """Bound memory under address spraying: drop expired keys from the oldest end.

        Every append moves its key to the end, so the front holds the least recently
        failed keys; eviction stops at the first live one (amortised O(1) per call)."""
        failures = self._failures
        if len(failures) <= 10_000:
            return
        while failures:
            oldest = next(iter(failures))
            q = failures[oldest]
            if q and q[-1] > now - self.window:
                break
            failures.popitem(last=False)


def _ip(value: str):
    """An address from a peer or ``X-Forwarded-For`` hop: ``a.b.c.d``, ``a.b.c.d:port``, an IPv6
    address or ``[v6]:port``; IPv4-mapped IPv6 becomes IPv4. None when malformed."""
    text = value.strip()
    port = ""
    if text.startswith("["):
        host, sep, rest = text[1:].partition("]")
        if not sep or (rest and not rest.startswith(":")):
            return None
        text, port = host, rest[1:]
        if rest and not port:
            return None
    elif text.count(":") == 1:
        text, _, port = text.partition(":")
        if not port:
            return None
    if ":" in text:
        text = text.partition("%")[0]  # an IPv6 zone id names a local interface, not a client
    # Bound the digit count before int(): a huge numeric port would hit the int-conversion limit.
    if port and not (
        len(port) <= 5 and port.isascii() and port.isdigit() and int(port) <= 65535
    ):
        return None
    try:
        ip = ipaddress.ip_address(text)
    except ValueError:
        return None
    mapped = getattr(ip, "ipv4_mapped", None)
    return mapped if mapped is not None else ip


def _unmap(net):
    """``::ffff:a.b.c.d/n`` (n >= 96) as the IPv4 network it covers; other networks unchanged."""
    mapped = getattr(net.network_address, "ipv4_mapped", None)
    if net.version == 6 and mapped is not None and net.prefixlen >= 96:
        return ipaddress.ip_network(f"{mapped}/{net.prefixlen - 96}")
    return net


def _bucket(ip) -> str:
    """The limiter key for a client address: IPv4 as is, IPv6 as its /64, since one host
    commonly holds a whole /64 and could otherwise rotate addresses to reset its count."""
    if ip.version == 6:
        return str(ipaddress.ip_network(f"{ip}/64", strict=False))
    return str(ip)


def _trusted(ip, networks) -> bool:
    return ip is not None and any(ip in _unmap(net) for net in networks)


def client_key(request: Request) -> str:
    """The client address the login limiter keys on.

    The socket peer, unless the peer is a trusted reverse proxy
    (``SPARROW_FT_TRUSTED_PROXIES``): then ``X-Forwarded-For`` is walked from the right,
    skipping trusted hops, and the first untrusted address is the client. Entries left of it
    are client-controlled and never used. The key is canonical (``_bucket``), so a client
    has one key whether it is reached directly or through the proxy."""
    peer = request.client.host if request.client else "unknown"
    peer_ip = _ip(peer)
    networks = getattr(request.app.state.settings, "trusted_proxies", ())
    if not networks or not _trusted(peer_ip, networks):
        return _bucket(peer_ip) if peer_ip is not None else peer
    hops = [
        h.strip()
        for header in request.headers.getlist("x-forwarded-for")
        for h in header.split(",")
        if h.strip()
    ]
    client = peer_ip
    for hop in reversed(hops):
        ip = _ip(hop)
        if ip is None:
            break  # malformed: do not trust anything to its left
        client = ip
        if not _trusted(ip, networks):
            break
    return _bucket(client)


def _login_limiter(request: Request) -> LoginLimiter:
    limiter = getattr(request.app.state, "login_limiter", None)
    if limiter is None:
        limiter = request.app.state.login_limiter = LoginLimiter()
    return limiter


@auth_router.post("/auth/login")
def login(body: LoginIn, request: Request, response: Response):
    settings = request.app.state.settings
    username = body.username.strip()
    limiter = _login_limiter(request)
    client = client_key(request)
    wait, stamp = limiter.begin(client)
    if wait is not None:
        raise HTTPException(
            429,
            "too many failed login attempts; try again later",
            headers={"Retry-After": str(wait)},
        )
    with connect(_url(request)) as conn:
        user = conn.execute(
            "SELECT id, password_hash FROM users WHERE username = %s", (username,)
        ).fetchone()
        ok = verify_password(
            body.password, user["password_hash"] if user else _DUMMY_HASH
        )
        ok = ok and user is not None
        record_audit(conn, username or "anonymous", "login", "", {"ok": ok})
        if not ok:
            conn.commit()
            raise HTTPException(401, _LOGIN_FAILED)
        limiter.succeeded(client, stamp)
        token = new_session_token()
        expires = datetime.now(timezone.utc) + timedelta(hours=settings.session_hours)
        conn.execute("DELETE FROM sessions WHERE expires_at <= now()")
        conn.execute(
            "INSERT INTO sessions (user_id, token_sha256, expires_at) VALUES (%s, %s, %s)",
            (user["id"], token_sha256(token), expires),
        )
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=int(settings.session_hours * 3600),
        httponly=True,
        samesite="strict",
        secure=settings.cookie_secure,
        path="/",
    )
    return {"username": username}


@auth_router.post("/auth/logout", status_code=204)
def logout(request: Request, principal: Principal = Depends(require_auth)):
    with connect(_url(request)) as conn:
        if principal.session_id is not None:
            conn.execute("DELETE FROM sessions WHERE id = %s", (principal.session_id,))
        record_audit(conn, principal.name, "logout")
    response = Response(status_code=204)
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response


@auth_router.get("/auth/me")
def me(
    request: Request, creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer)
):
    principal, required = resolve_principal(request, creds)
    if principal is None:
        return {"username": None, "method": None, "auth_required": required}
    return {
        "username": principal.name if principal.method == "session" else None,
        "method": principal.method,
        "auth_required": required,
    }


audit_router = APIRouter()


@audit_router.get("/audit")
def list_audit(request: Request, limit: int = 100):
    if not 1 <= limit <= 1000:
        raise HTTPException(422, "limit must be between 1 and 1000")
    with connect(_url(request)) as conn:
        rows = conn.execute(
            "SELECT id, principal, action, target, detail, created_at FROM audit_events"
            " ORDER BY id DESC LIMIT %s",
            (limit,),
        ).fetchall()
    return {"events": rows}
