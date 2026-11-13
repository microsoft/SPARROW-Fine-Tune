#!/usr/bin/env bash
# Regenerate requirements-cpu.txt, the lock for the CPU-only image (FT-12).
# Versions are pinned to uv.lock (the +cu128 local tags stripped), so the CPU image runs the
# same package versions as the tested CUDA environment; only torch/torchvision come from the
# PyTorch CPU index. The [build-system] requirements (setuptools, wheel) are locked with
# hashes too: the image installs the project with --no-build-isolation, so no unhashed build
# backend is fetched at build time. Constraints and build requirements are passed through
# process substitution, so the output carries no temporary paths and is reproducible.
# Run after every change to uv.lock or [build-system].
set -euo pipefail
cd "$(dirname "$0")/.."
uv pip compile pyproject.toml \
    <(uv run --no-project --python 3.12 python -c \
        'import tomllib; print("\n".join(tomllib.load(open("pyproject.toml", "rb"))["build-system"]["requires"]))') \
    --extra export --no-sources --torch-backend cpu \
    --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes \
    --no-header --no-annotate -q -o requirements-cpu.txt \
    -c <(uv export --frozen --no-dev --extra export --no-hashes --no-emit-project --no-header -q \
        | grep -v '^ *#' | sed -E 's/\+cu128//' | grep '==')
