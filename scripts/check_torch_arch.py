"""Fail the image build if torch cannot target the installed GPUs.

A torch wheel built for an older CUDA imports cleanly, reports
``cuda.is_available() == True``, and then dies at the first kernel launch with
"no kernel image is available for execution on the device". On Blackwell
(RTX PRO 4500 / RTX 50-series, ``sm_120``) that is a very easy mistake to make,
and it surfaces minutes into training rather than at build time.

Usage: ``python3 scripts/check_torch_arch.py sm_120 [sm_89 ...]``
"""

from __future__ import annotations

import sys


def main(argv: list[str]) -> int:
    required = argv[1:] or ["sm_120"]

    try:
        import torch
    except ImportError as e:
        print(f"FATAL: torch is not importable: {e}", file=sys.stderr)
        return 1

    # get_arch_list() is [] when no GPU is visible, as in `docker build`; the compiled flags are not.
    archs = list(torch.cuda.get_arch_list()) or torch._C._cuda_getArchFlags().split()
    print(f"torch {torch.__version__}  (CUDA {torch.version.cuda})")
    print(f"compiled arch list: {archs}")

    missing = [a for a in required if a not in archs]
    if missing:
        print(
            f"\nFATAL: this torch build lacks {', '.join(missing)} kernels.\n"
            f"Blackwell GPUs need sm_120, which requires a CUDA 12.8+ wheel.\n"
            f"Rebuild with a newer wheel, e.g.:\n"
            f"  docker build --build-arg TORCH_SPEC='torch==2.8.0 torchvision==0.23.0' .",
            file=sys.stderr,
        )
        return 1

    print(f"OK: {', '.join(required)} present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
