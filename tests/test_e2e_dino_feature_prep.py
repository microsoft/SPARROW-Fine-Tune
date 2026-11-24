"""Deployment runners: DINO features are prepared (opt-in, progress polled) before training.

Strategy: run the real helpers of ``scripts/e2e`` (the shared ``prepare_dino_snapshot`` used by the
CPU image, Studio/GPU and feature-matrix runners, and the CPU runner's post-training checks)
against a small fake of the service's Data Preparation API, which follows the contract in
``tests/finetune/test_release_first_run.py``. Each helper must record a passing check for a
service that honours the contract and a failing check for one that breaks it (work reported
wrongly, zero work that still queues a job, progress going backwards, an encoder changed during
training, features left stale by training). The containers themselves are exercised only by the
runners; this pins what the runners assert.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import pytest

E2E = Path(__file__).resolve().parents[1] / "scripts" / "e2e"


@pytest.fixture(scope="module")
def cpu():
    sys.path.insert(0, str(E2E))
    spec = importlib.util.spec_from_file_location(
        "dino_prep_cpu_e2e", E2E / "cpu_image_e2e.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    yield mod
    sys.modules.pop(spec.name, None)
    sys.path.remove(str(E2E))


@pytest.fixture
def checks(cpu):
    cpu.CHECKS.clear()
    yield cpu.CHECKS
    cpu.CHECKS.clear()


def _api_class(cpu):
    class FakeService(cpu.Api):
        """Data Preparation of one classification dataset (id 1); ``prepared`` maps crop -> encoder."""

        def __init__(self, crops, encoder_sha="enc-a"):
            super().__init__("http://service.invalid", None)
            self.crops = list(crops)
            self.prepared: dict[str, str] = {}
            self.encoder_sha = encoder_sha
            self.manifest_hash = encoder_sha
            self.snaps: dict[int, dict] = {}
            self.zero_work_job = False  # break the zero-work contract
            self.done_backwards = False  # break monotonic progress

        def _missing(self):
            return [c for c in self.crops if self.prepared.get(c) != self.encoder_sha]

        def call(self, method, path, body=None, raw=None, timeout=900):
            parts = path.split("?")[0].strip("/").split("/")
            if method == "POST" and parts == ["datasets", "1", "snapshots"]:
                assert (body or {}).get("prepare_features") is True, body
                sid = len(self.snaps) + 1
                todo = self._missing()
                zero = not todo
                f = {
                    "state": "ready" if zero else "queued",
                    "total": len(self.crops),
                    "to_compute": len(todo),
                    "reused": len(self.crops) - len(todo),
                    "done": 0,
                    "job_id": 7 if (not zero or self.zero_work_job) else None,
                    "message": "Nothing to compute"
                    if zero
                    else "Preparing DINO features",
                }
                self.snaps[sid] = {"f": f, "todo": todo, "polls": 0}
                return 201, {"id": sid, "features": dict(f)}
            if method == "GET" and parts[0] == "snapshots" and parts[2] == "features":
                s = self.snaps[int(parts[1])]
                f = s["f"]
                if f["state"] != "ready":
                    s["polls"] += 1
                    if s["polls"] == 1:
                        f.update(state="running", done=len(s["todo"]) // 2 + 1)
                    else:
                        f.update(state="ready", done=f["to_compute"])
                        for c in s["todo"]:
                            self.prepared[c] = self.encoder_sha
                        if (
                            self.done_backwards
                        ):  # reports ready while progress went 2 -> 0
                            return 200, {**f, "done": 0}
                return 200, dict(f)
            if method == "GET" and parts == ["datasets", "1", "feature-prep"]:
                items = [
                    {
                        "image": c,
                        "crop": None,
                        "status": "valid" if c not in self._missing() else "stale",
                    }
                    for c in self.crops
                ]
                valid = sum(i["status"] == "valid" for i in items)
                counts = {
                    "total": len(items),
                    "valid": valid,
                    "missing": 0,
                    "stale": len(items) - valid,
                    "corrupt": 0,
                }
                return 200, {
                    "items": items,
                    "counts": counts,
                    "to_compute": len(items) - valid,
                    "message": "",
                }
            if method == "GET" and parts[0] == "models" and parts[-1] == "artifact":
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w") as zf:
                    zf.writestr(
                        "inner/manifest.json",
                        json.dumps({"embedder": {"model_hash": self.manifest_hash}}),
                    )
                return 200, buf.getvalue()
            raise AssertionError(f"unexpected call {method} {path}")

    return FakeService


def _named(checks, fragment):
    found = [(n, ok) for n, ok, _ in checks if fragment in n]
    assert found, [n for n, _, _ in checks]
    return found


def test_split_holds_back_one_new_crop_per_species(cpu):
    items = [
        {"image": f"{lab}{i}", "annotations": [{"label": lab, "bbox": None}]}
        for lab in cpu.SPECIES
        for i in range(3)
    ]
    first, held = cpu.split_new_crops(items)
    assert sorted(it["annotations"][0]["label"] for it in held) == sorted(cpu.SPECIES)
    assert {it["image"] for it in first} | {it["image"] for it in held} == {
        it["image"] for it in items
    }
    assert not {it["image"] for it in first} & {it["image"] for it in held}


def test_new_then_mixed_then_zero_work_preparation_passes(cpu, checks):
    api = _api_class(cpu)(["a", "b", "c"])
    s1 = cpu.prepare_dino_snapshot(api, 1, 30, "new", expect_new=3, expect_reused=0)
    api.crops += ["d", "e"]
    s2 = cpu.prepare_dino_snapshot(api, 1, 30, "mixed", expect_new=2, expect_reused=3)
    s3 = cpu.prepare_dino_snapshot(api, 1, 30, "zero", expect_new=0, expect_reused=5)
    assert all(ok for _, ok, _ in checks), [(n, ok, d) for n, ok, d in checks if not ok]
    assert [s["features"]["state"] for s in (s1, s2, s3)] == ["ready"] * 3
    assert _named(checks, "zero: zero work is ready at once") == [
        ("zero: zero work is ready at once with no preparation job", True)
    ]
    # The poll waited for the worker: work-bearing snapshots were polled past ``running``.
    assert [api.snaps[i]["polls"] for i in (1, 2)] == [2, 2]


@pytest.mark.parametrize(
    ("breakage", "fragment"),
    [
        ("wrong_counts", "preparation computes"),
        ("zero_work_job", "zero work is ready at once"),
        ("done_backwards", "DINO features ready before training"),
    ],
)
def test_preparation_contract_breaks_fail_the_runner(cpu, checks, breakage, fragment):
    api = _api_class(cpu)(["a", "b", "c"])
    if breakage == "wrong_counts":
        cpu.prepare_dino_snapshot(api, 1, 30, "x", expect_new=2, expect_reused=1)
    elif breakage == "zero_work_job":
        api.prepared = dict.fromkeys(api.crops, api.encoder_sha)
        api.zero_work_job = True
        cpu.prepare_dino_snapshot(api, 1, 30, "x", expect_new=0, expect_reused=3)
    else:
        api.done_backwards = True
        cpu.prepare_dino_snapshot(api, 1, 30, "x", expect_new=3, expect_reused=0)
    assert _named(checks, fragment) == [(n, False) for n, _ in _named(checks, fragment)]


def test_training_that_left_encoder_and_features_alone_passes(cpu, checks, tmp_path):
    enc = tmp_path / "dinov3_vits16.onnx"
    enc.write_bytes(b"real onnx bytes")
    sha = hashlib.sha256(enc.read_bytes()).hexdigest()
    api = _api_class(cpu)(["a", "b"], encoder_sha=sha)
    api.prepared = dict.fromkeys(api.crops, sha)
    cpu.dino_training_left_features_alone(api, 1, {"id": 4}, enc, sha, 30)
    assert checks and all(ok for _, ok, _ in checks), [c for c in checks if not c[1]]
    assert _named(checks, "after training: zero work")


@pytest.mark.parametrize(
    ("breakage", "fragment"),
    [
        ("encoder_rewritten", "left the ONNX encoder unchanged"),
        ("manifest_other_encoder", "records the encoder that prepared"),
        ("features_invalidated", "still valid after training"),
    ],
)
def test_training_that_touched_encoder_or_features_fails(
    cpu, checks, tmp_path, breakage, fragment
):
    enc = tmp_path / "dinov3_vits16.onnx"
    enc.write_bytes(b"real onnx bytes")
    sha = hashlib.sha256(enc.read_bytes()).hexdigest()
    api = _api_class(cpu)(["a", "b"], encoder_sha=sha)
    api.prepared = dict.fromkeys(api.crops, sha)
    if breakage == "encoder_rewritten":
        enc.write_bytes(b"fine-tuned encoder")
    elif breakage == "manifest_other_encoder":
        api.manifest_hash = "0" * 64
    else:
        api.prepared["a"] = "another-encoder"
    cpu.dino_training_left_features_alone(api, 1, {"id": 4}, enc, sha, 30)
    assert _named(checks, fragment) == [(n, False) for n, _ in _named(checks, fragment)]
