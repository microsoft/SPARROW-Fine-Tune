"""Emit a pip constraints file pinning the currently installed torch stack.

The CUDA 12.8 torch build is installed first and must survive installing
everything else. Filtering ``torch`` out of requirements.txt is not enough:
``ultralytics`` depends on torch itself, so pip is free to resolve a different
(PyPI, non-Blackwell) build as a transitive dependency. A constraints file binds
transitive resolution too.

Usage: ``python3 scripts/pin_torch_constraints.py /tmp/constraints.txt``
"""

from __future__ import annotations

import importlib.metadata as md
import sys

PACKAGES = ("torch", "torchvision", "torchaudio")


def main(argv: list[str]) -> int:
    out_path = argv[1] if len(argv) > 1 else "/tmp/constraints.txt"

    lines = []
    for pkg in PACKAGES:
        try:
            lines.append(f"{pkg}=={md.version(pkg)}")
        except md.PackageNotFoundError:
            continue

    if not lines:
        print("FATAL: no torch packages installed to pin", file=sys.stderr)
        return 1

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"pinned to {out_path}:")
    for line in lines:
        print(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
