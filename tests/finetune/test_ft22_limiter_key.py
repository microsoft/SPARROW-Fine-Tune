"""FT-22: one login-limiter key per client, however its address is spelled or reached."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from starlette.requests import Request

from sparrow_finetune.api.security import client_key
from sparrow_finetune.config import _normalise_networks


def _key(peer: str, xff: str | None = None, trusted: str = "10.0.0.0/8") -> str:
    app = SimpleNamespace(
        state=SimpleNamespace(settings=SimpleNamespace(trusted_proxies=_normalise_networks(trusted)))
    )
    headers = [(b"x-forwarded-for", xff.encode())] if xff is not None else []
    return client_key(
        Request({"type": "http", "client": (peer, 1234), "headers": headers, "app": app})
    )


@pytest.mark.parametrize(
    "a, b",
    [
        # Direct peers: mapped IPv4, zone ids and IPv6 spellings collapse to one key.
        (("::ffff:203.0.113.7",), ("203.0.113.7",)),
        (("fe80::1%eth0",), ("fe80::1",)),
        (("2001:DB8:0:0::1",), ("2001:db8::1",)),
        # A client reached directly and through the trusted proxy shares its key.
        (("2001:db8::1",), ("10.1.2.3", "2001:db8::1")),
        (("203.0.113.7",), ("10.1.2.3", "::ffff:203.0.113.7")),
        (("fe80::1",), ("10.1.2.3", "fe80::1%eth0")),
        # Rotating addresses inside one IPv6 /64 does not give new buckets.
        (("2001:db8::1",), ("2001:db8::ffff:1",)),
        (("10.1.2.3", "2001:db8::1"), ("10.1.2.3", "[2001:db8::abcd]:443")),
    ],
)
def test_same_client_same_key(a, b):
    assert _key(*a) == _key(*b)


@pytest.mark.parametrize(
    "a, b",
    [
        (("2001:db8::1",), ("2001:db8:0:1::1",)),
        (("203.0.113.7",), ("203.0.113.8",)),
        (("10.1.2.3", "2001:db8::1"), ("10.1.2.3", "2001:db8:1::1")),
    ],
)
def test_different_clients_different_keys(a, b):
    assert _key(*a) != _key(*b)


def test_trusted_peer_ipv6_proxy_not_bucketed_with_its_neighbours():
    """The trust check still uses the full address: a /64 key never makes a neighbour trusted."""
    assert _key("2001:db8::2", "203.0.113.7", trusted="2001:db8::1/128") == _key("2001:db8::2")
