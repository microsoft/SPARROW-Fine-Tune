"""FT-18 review follow-up pins (two defects found in the FT-18 fix review).

Strategy: drive the real public surface, not the source text.

- Login limiter: an ``X-Forwarded-For`` hop with an absurd port (thousands of digits, plain or
  bracketed IPv6) must be treated as malformed like any other bad hop, so the login route
  answers 401/429 rather than 500, and the address left of it is never the limiter key. The
  65535/65536 boundary pins where "valid port" ends.
- Compose: ``docker compose config`` (real interpolation, profiles and anchors) must never give
  one image name to services built with different ``TORCH_VARIANT`` args, or ``up --build``
  makes the last build overwrite the other. Defaults stay one shared CUDA ``latest`` image and
  explicit tag overrides are honoured. No particular new tag name is required.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from .service_helpers import PASSWORD, add_user
from .test_backlog_service import _login, _proxied_client

REPO = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------- forwarded port

_HUGE = (
    "9" * 5000
)  # above CPython's int() string-digit limit: int(_HUGE) raises ValueError
_LEFT = "198.51.100.66"  # client-controlled entry left of the hop under test
_PEER = "10.1.2.3"


@pytest.mark.parametrize(
    "hop, key",
    [
        ("203.0.113.7:65535", "203.0.113.7"),
        ("203.0.113.7:65536", None),
        (f"203.0.113.7:{_HUGE}", None),
        (f"[2001:db8::1]:{_HUGE}", None),
    ],
    ids=[
        "port-65535",
        "port-65536",
        "malformed_long-ipv4-port",
        "malformed_long-ipv6-port",
    ],
)
def test_ft18_review_forwarded_port_bounds_and_malformed_long_port(
    settings, registry, hop, key
):
    """``key`` is the expected limiter key: the hop's address, or None for the socket peer."""
    add_user(settings)
    s = dataclasses.replace(settings, trusted_proxies="10.0.0.0/8")
    with _proxied_client(s, registry, _PEER) as c:
        statuses = [_login(c, "wrong", f"{_LEFT}, {hop}") for _ in range(10)]
        assert statuses == [401] * 10
        # The spoofable entry left of the hop never shares the failing key.
        assert _login(c, PASSWORD, _LEFT) == 200
        # The failures were counted against the expected key (locked out now).
        assert _login(c, PASSWORD, key) == 429


# --------------------------------------------------------------------------- compose images

_COMPOSE_VARS = ("TORCH_VARIANT", "SPARROW_FT_IMAGE_TAG", "SPARROW_FT_GPU_IMAGE_TAG")
_APP_SERVICES = ("migrate", "api", "worker-cpu", "worker-gpu")


def _compose_services(**overrides: str) -> dict:
    if shutil.which("docker") is None:
        pytest.skip("docker CLI not installed")
    env = {k: v for k, v in os.environ.items() if k not in _COMPOSE_VARS}
    env.update(POSTGRES_PASSWORD="x", **overrides)
    out = subprocess.run(
        [
            "docker",
            "compose",
            "--env-file",
            os.devnull,
            "-f",
            str(REPO / "compose.service.yml"),
            "--profile",
            "gpu",
            "config",
            "--format",
            "json",
        ],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)["services"]


def _variant(svc: dict) -> str:
    return svc["build"]["args"]["TORCH_VARIANT"]


def _assert_one_variant_per_image(services: dict) -> None:
    by_image: dict[str, set[str]] = {}
    for name in _APP_SERVICES:
        by_image.setdefault(services[name]["image"], set()).add(
            _variant(services[name])
        )
    clashes = {img: v for img, v in by_image.items() if len(v) > 1}
    assert not clashes, f"one image name built from different TORCH_VARIANTs: {clashes}"


def test_ft18_review_compose_cpu_variant_without_tags_keeps_gpu_image_separate():
    services = _compose_services(TORCH_VARIANT="cpu")
    assert [_variant(services[n]) for n in _APP_SERVICES] == [
        "cpu",
        "cpu",
        "cpu",
        "cu128",
    ]
    cpu_images = {services[n]["image"] for n in ("migrate", "api", "worker-cpu")}
    assert len(cpu_images) == 1, cpu_images
    assert services["worker-gpu"]["image"] not in cpu_images
    _assert_one_variant_per_image(services)


def test_ft18_review_compose_default_is_one_shared_cuda_latest_image():
    services = _compose_services()
    assert {_variant(services[n]) for n in _APP_SERVICES} == {"cu128"}
    assert {services[n]["image"] for n in _APP_SERVICES} == {"sparrow-finetune:latest"}


@pytest.mark.parametrize(
    "overrides, cpu_image, gpu_image",
    [
        (
            {
                "TORCH_VARIANT": "cpu",
                "SPARROW_FT_IMAGE_TAG": "mycpu",
                "SPARROW_FT_GPU_IMAGE_TAG": "mygpu",
            },
            "sparrow-finetune:mycpu",
            "sparrow-finetune:mygpu",
        ),
        (
            {"SPARROW_FT_IMAGE_TAG": "v2", "SPARROW_FT_GPU_IMAGE_TAG": "v2"},
            "sparrow-finetune:v2",
            "sparrow-finetune:v2",
        ),
    ],
    ids=["cpu-with-both-tags", "cuda-shared-explicit-tag"],
)
def test_ft18_review_compose_explicit_image_tags_are_honoured(
    overrides, cpu_image, gpu_image
):
    services = _compose_services(**overrides)
    assert {services[n]["image"] for n in ("migrate", "api", "worker-cpu")} == {
        cpu_image
    }
    assert services["worker-gpu"]["image"] == gpu_image
    _assert_one_variant_per_image(services)
