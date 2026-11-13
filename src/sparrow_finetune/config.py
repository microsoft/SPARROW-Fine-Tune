"""Service settings, read from the environment by the CLI."""

from __future__ import annotations

import ipaddress
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _normalise_tokens(value: object) -> tuple[str, ...]:
    """A comma-separated string or an iterable of strings -> stripped, non-empty tokens."""
    if value is None:
        return ()
    if isinstance(value, str):
        parts: Iterable[object] = value.split(",")
    elif isinstance(value, Iterable) and not isinstance(
        value, (bytes, bytearray, Mapping)
    ):
        parts = value
    else:
        raise TypeError(
            f"api_tokens must be a str or an iterable of str, not {type(value).__name__}"
        )
    tokens = []
    for part in parts:
        if not isinstance(part, str):
            raise TypeError(
                f"api_tokens entries must be str, not {type(part).__name__}"
            )
        if part.strip():
            tokens.append(part.strip())
    return tuple(tokens)


def _normalise_networks(value: object) -> tuple:
    """Trusted proxy addresses or CIDRs (comma-separated str or iterable) -> ip networks."""
    if value is None:
        return ()
    parts = value.split(",") if isinstance(value, str) else value
    if not isinstance(parts, Iterable) or isinstance(parts, (bytes, bytearray, Mapping)):
        raise TypeError(
            f"trusted_proxies must be a str or an iterable, not {type(value).__name__}"
        )
    networks = []
    for part in parts:
        if isinstance(part, (ipaddress.IPv4Network, ipaddress.IPv6Network)):
            networks.append(part)
            continue
        text = str(part).strip()
        if not text:
            continue
        try:
            networks.append(ipaddress.ip_network(text, strict=False))
        except ValueError:
            raise ValueError(
                f"trusted_proxies entry {text!r} is not an IP address or CIDR"
            ) from None
    return tuple(networks)


@dataclass(frozen=True)
class Settings:
    database_url: str
    data_dir: Path
    api_tokens: tuple[str, ...] = field(default=())
    embedder: str = ""
    studio_url: str = ""
    studio_token: str = ""
    cookie_secure: bool = False
    session_hours: float = 12.0
    # Local pretrained weights (``yolo11n.pt``, ...) that trainers use instead of downloading; None = unset.
    weights_dir: Optional[Path] = None
    # Reverse proxies (addresses or CIDRs) whose X-Forwarded-For is trusted for the login
    # limiter's client key; empty = key on the socket peer.
    trusted_proxies: tuple = field(default=())
    # Sign-in off: requests without a token or session act as "anonymous" even when
    # user accounts exist (the app is reached through Sparrow Studio).
    auth_disabled: bool = False
    # Pipelines whose jobs wait for a worker offering ``gpu`` instead of training on a CPU
    # worker (``SPARROW_FT_GPU_PIPELINES``, comma-separated ids; empty = any worker).
    gpu_pipelines: tuple = field(default=())
    # Most jobs one user may have running at once; further jobs stay queued (0 = no limit).
    max_running_per_user: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_dir", Path(self.data_dir))
        weights_dir = self.weights_dir
        if isinstance(weights_dir, str):
            weights_dir = weights_dir.strip()
        object.__setattr__(
            self, "weights_dir", Path(weights_dir) if weights_dir else None
        )
        object.__setattr__(self, "api_tokens", _normalise_tokens(self.api_tokens))
        object.__setattr__(self, "embedder", (self.embedder or "").strip())
        object.__setattr__(
            self, "studio_url", (self.studio_url or "").strip().rstrip("/")
        )
        object.__setattr__(self, "studio_token", (self.studio_token or "").strip())
        hours = float(self.session_hours)
        if not (0 < hours <= 24 * 365):
            raise ValueError(f"session_hours must be in (0, 8760], got {hours}")
        object.__setattr__(self, "session_hours", hours)
        object.__setattr__(
            self, "trusted_proxies", _normalise_networks(self.trusted_proxies)
        )
        pipelines = self.gpu_pipelines
        if isinstance(pipelines, str):
            pipelines = pipelines.split(",")
        object.__setattr__(
            self,
            "gpu_pipelines",
            tuple(sorted({p.strip() for p in pipelines if p and p.strip()})),
        )
        cap = int(self.max_running_per_user)
        if cap < 0:
            raise ValueError(f"max_running_per_user must be >= 0, got {cap}")
        object.__setattr__(self, "max_running_per_user", cap)

    @property
    def studio_configured(self) -> bool:
        return bool(self.studio_url and self.studio_token)

    @property
    def objects_dir(self) -> Path:
        return self.data_dir / "objects"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def staged_dir(self) -> Path:
        """Read-only trainer inputs: ``<sha[:2]>/<sha><ext>`` views of stored images."""
        return self.data_dir / "staged"

    @classmethod
    def from_env(cls) -> "Settings":
        database_url = os.environ.get("SPARROW_FT_DATABASE_URL", "").strip()
        if not database_url:
            raise RuntimeError("SPARROW_FT_DATABASE_URL is not set")
        data_dir = Path(os.environ.get("SPARROW_FT_DATA_DIR", "data"))
        tokens = os.environ.get("SPARROW_FT_API_TOKENS", "")
        embedder = os.environ.get("SPARROW_FT_EMBEDDER", "")
        return cls(
            database_url=database_url,
            data_dir=data_dir,
            api_tokens=tokens,
            embedder=embedder,
            studio_url=os.environ.get("SPARROW_FT_STUDIO_URL", ""),
            studio_token=os.environ.get("SPARROW_FT_STUDIO_TOKEN", ""),
            cookie_secure=os.environ.get("SPARROW_FT_COOKIE_SECURE", "").strip().lower()
            in {"1", "true", "yes", "on"},
            session_hours=float(os.environ.get("SPARROW_FT_SESSION_HOURS", "") or 12),
            weights_dir=os.environ.get("SPARROW_FT_WEIGHTS_DIR", ""),
            trusted_proxies=os.environ.get("SPARROW_FT_TRUSTED_PROXIES", ""),
            auth_disabled=os.environ.get("SPARROW_FT_AUTH_DISABLED", "").strip().lower()
            in {"1", "true", "yes", "on"},
            gpu_pipelines=os.environ.get("SPARROW_FT_GPU_PIPELINES", ""),
            max_running_per_user=int(
                os.environ.get("SPARROW_FT_MAX_RUNNING_PER_USER", "").strip() or 0
            ),
        )
