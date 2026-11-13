"""scripts/check_torch_arch.py must work in a GPU-less ``docker build``.

``torch.cuda.get_arch_list()`` returns ``[]`` whenever CUDA is unavailable, so the build-time check
has to read the compiled arch flags instead.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_torch_arch.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_torch_arch", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_torch(monkeypatch, flags: str):
    torch = types.ModuleType("torch")
    torch.__version__ = "0.0+cu128"
    torch.version = types.SimpleNamespace(cuda="12.8")
    torch.cuda = types.SimpleNamespace(get_arch_list=lambda: [])  # no GPU visible
    torch._C = types.SimpleNamespace(_cuda_getArchFlags=lambda: flags)
    monkeypatch.setitem(sys.modules, "torch", torch)


def test_passes_without_a_gpu_when_the_wheel_has_sm_120(monkeypatch):
    _fake_torch(monkeypatch, "sm_75 sm_80 sm_86 sm_90 sm_100 sm_120")
    assert _load().main(["check", "sm_120"]) == 0


def test_fails_without_a_gpu_when_the_wheel_lacks_sm_120(monkeypatch):
    _fake_torch(monkeypatch, "sm_75 sm_80 sm_86 sm_90")
    assert _load().main(["check", "sm_120"]) == 1
