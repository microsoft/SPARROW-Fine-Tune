"""Password hashing (scrypt) and session tokens."""

from __future__ import annotations

import hashlib
import hmac
import secrets

_N, _R, _P = 2**14, 8, 1
_SALT_BYTES = 16
_DKLEN = 32
MIN_PASSWORD_LENGTH = 8


def _scrypt(password: str, salt: bytes, n: int, r: int, p: int, dklen: int) -> bytes:
    return hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=n,
        r=r,
        p=p,
        dklen=dklen,
        maxmem=256 * 1024 * 1024,
    )


def hash_password(password: str) -> str:
    """``scrypt$<n>$<r>$<p>$<salt_hex>$<hash_hex>``."""
    salt = secrets.token_bytes(_SALT_BYTES)
    digest = _scrypt(password, salt, _N, _R, _P, _DKLEN)
    return f"scrypt${_N}${_R}${_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Constant-time check; a malformed stored value is a mismatch, never an exception."""
    try:
        scheme, n, r, p, salt_hex, hash_hex = stored.split("$")
        if scheme != "scrypt":
            return False
        n_i, r_i, p_i = int(n), int(r), int(p)
        # Bound the parameters so a tampered row cannot make verification hang or OOM.
        if not (
            2 <= n_i <= 2**20
            and n_i & (n_i - 1) == 0
            and 1 <= r_i <= 32
            and 1 <= p_i <= 16
        ):
            return False
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
        if not salt or not expected or len(expected) > 128:
            return False
        actual = _scrypt(password, salt, n_i, r_i, p_i, len(expected))
    except (ValueError, TypeError, AttributeError, MemoryError):
        return False
    return hmac.compare_digest(actual, expected)


def new_session_token() -> str:
    return secrets.token_urlsafe(32)


def token_sha256(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def token_principal(token: str) -> str:
    return f"token:{token_sha256(token)[:8]}"
