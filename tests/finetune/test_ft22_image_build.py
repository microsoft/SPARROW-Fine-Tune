"""FT-22: no image variant fetches an unhashed build backend for the project install."""

from __future__ import annotations

import re
from pathlib import Path

DOCKERFILE = Path(__file__).resolve().parents[2] / "Dockerfile.service"


def _run_lines() -> list[str]:
    text = DOCKERFILE.read_text().replace("\\\n", " ")
    return [line for line in text.splitlines() if line.lstrip().startswith("RUN ")]


def test_every_project_install_skips_build_isolation():
    runs = " ".join(_run_lines())
    # `uv sync` without --no-install-project builds the project in an isolated env whose
    # setuptools/wheel come from the index unhashed.
    for sync in re.findall(r"uv sync[^;&|]*", runs):
        assert "--no-install-project" in sync, sync
    installs = re.findall(r"uv pip install[^;&|]*\s\.(?=\s|;|$)", runs)
    assert installs, "the project is never installed"
    for cmd in installs:
        assert "--no-build-isolation" in cmd and "--no-deps" in cmd, cmd
