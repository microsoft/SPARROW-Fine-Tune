"""FT-21: the Studio + Engine e2e scenario must survive a pipeline that produced no model.

Strategy: run the real ``scenario()`` from ``scripts/e2e/studio_engine_e2e.py`` against
lightweight fakes of its two network boundaries (the service ``Api.call`` and the ``Studio``
helper). One training job is made to fail, so ``/models?job_id=`` returns no model. The scenario
must record the failure as a failed check and keep running the sections that do not depend on the
missing model, instead of raising ``KeyError`` on ``model[pipe]["id"]``. The combination section
(fine-tuned classifier on the fine-tuned detector) needs both ``conventional_classifier`` and
``yolo_detector``; with either missing it must not report a pass. The ``None`` case is a control
proving the fakes carry a fully successful run through every section.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "e2e" / "studio_engine_e2e.py"
)
PIPES = ("conventional_classifier", "dinov3_linear", "yolo_detector")
# Substrings of check names emitted once each pipeline's own Studio section has run.
SECTION = {
    "conventional_classifier": "conventional_classifier: service activation",
    "dinov3_linear": "dinov3_linear runs in Studio",
    "yolo_detector": "yolo_detector: installed as",
}
COMBINATION = (
    "reuses the fine-tuned detector",
    "fine-tuned detector + fine-tuned classifier",
)


@pytest.fixture(scope="module")
def e2e():
    spec = importlib.util.spec_from_file_location("ft21_studio_engine_e2e", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    yield mod
    sys.modules.pop(spec.name, None)


def _fake_api(e2e, failed: str | None):
    class FakeApi(e2e.Api):
        """Service API: every route the scenario uses; ``failed``'s job ends without a model."""

        def __init__(self):
            super().__init__("http://service.invalid", None)
            self.jobs: dict[int, str] = {}
            self.activations: list[tuple[str, str]] = []
            self.params: dict[int, dict] = {}
            self.models: list[int] = []
            self.prep: dict[int, int] = {}  # prepared snapshot id -> polls so far
            self.dino_snapshots: list[int] = []

        def _features(self, sid: int) -> dict:
            if sid not in self.prep:
                return {"state": "not_prepared"}
            polls = self.prep[sid]
            done = min(polls, 3)
            state = "ready" if polls >= 2 else ("running" if polls else "queued")
            done = 3 if state == "ready" else min(done, 1)
            return {
                "state": state,
                "total": 3,
                "to_compute": 3,
                "reused": 0,
                "done": done,
                "job_id": 900 + sid,
                "message": "Preparing DINO features",
            }

        def call(self, method, path, body=None, raw=None, timeout=900):
            url = urlsplit(path)
            parts = url.path.strip("/").split("/")
            if method == "POST" and parts == ["workspaces"]:
                return 201, {"id": 1}
            if method == "POST" and parts[-1] == "datasets":
                if body["name"] == "animal boxes":
                    return 201, {"id": 3}
                return 201, {"id": 2 if body["task"] == "detection" else 1}
            if method == "GET" and parts[0] == "datasets" and parts[-1] == "items":
                box = {"label": "Bos", "bbox": [0, 0, 1, 1]}
                return 200, {"items": [{"image": "a/b.jpg", "annotations": [box]}]}
            if method == "PATCH" and parts[0] == "datasets" and parts[2] == "items":
                return 200, {}
            if method == "GET" and parts == ["connectors", "studio", "models"]:
                return 200, {
                    "models": [
                        {"id": str(100 + m), "importable": True} for m in self.models
                    ]
                }
            if method == "POST" and parts == [
                "connectors",
                "studio",
                "models",
                "import",
            ]:
                return 201, {"id": 50}
            if method == "POST" and parts == ["connectors", "studio", "import"]:
                return 200, {"imported_images": 3, "imported_items": 3, "skipped": 0}
            if method == "POST" and parts[-1] == "snapshots":
                snap = {
                    "id": int(parts[1]),
                    "label_counts": {"animal": 1}
                    if parts[1] == "3"
                    else {s: 1 for s in e2e.SPECIES},
                }
                if (body or {}).get("prepare_features"):
                    # DINO Data Preparation: queued work the worker finishes over two polls.
                    snap["id"] = 100 + int(parts[1])
                    self.prep[snap["id"]] = 0
                    snap["features"] = self._features(snap["id"])
                return 201, snap
            if method == "GET" and parts[0] == "snapshots" and parts[-1] == "features":
                sid = int(parts[1])
                if sid not in self.prep:
                    return 200, {"state": "not_prepared"}
                self.prep[sid] += 1
                return 200, self._features(sid)
            if method == "POST" and parts == ["jobs"]:
                sid = body["snapshot_id"]
                if body["pipeline"] == "dinov3_linear" and (
                    self._features(sid).get("state") != "ready"
                ):
                    return 422, {"detail": "Prepare DINO features in Data Preparation"}
                if body["pipeline"] == "dinov3_linear":
                    self.dino_snapshots.append(sid)
                jid = len(self.jobs) + 1
                self.jobs[jid] = body["pipeline"]
                self.params[jid] = body.get("params") or {}
                return 201, {"id": jid}
            if method == "GET" and parts[0] == "jobs":
                pipe = self.jobs[int(parts[1])]
                state = "failed" if pipe == failed else "succeeded"
                return 200, {
                    "id": int(parts[1]),
                    "state": state,
                    "error": None,
                    "params": self.params[int(parts[1])],
                }
            if method == "GET" and parts == ["models"]:
                jid = int(parse_qs(url.query)["job_id"][0])
                pipe = self.jobs[jid]
                return 200, {"models": [self._model(10 + jid, pipe)]}
            if method == "GET" and parts[0] == "models" and parts[-1] == "artifact":
                return 404, None
            if method == "GET" and parts[0] == "models":
                mid = int(parts[1])
                return 200, self._model(mid, self.jobs[mid - 10])
            if method == "POST" and parts[0] == "models" and parts[-1] == "activate":
                self.models.append(int(parts[1]))
                self.activations.append(
                    (self.jobs[int(parts[1]) - 10], body["project"])
                )
                return 200, {
                    "studio_install": "pending",
                    "external_id": str(100 + int(parts[1])),
                }
            if method == "POST" and parts == ["connectors", "studio", "push"]:
                return 200, {"studio_model_id": 200 + int(body["model_id"])}
            if method == "GET" and parts == ["deployments"]:
                return 200, {"deployments": []}
            raise AssertionError(f"unexpected service call {method} {path}")

        @staticmethod
        def _model(mid: int, pipe: str) -> dict:
            return {
                "id": mid,
                "pipeline": pipe,
                "metrics": {},
                "stock_comparison": {
                    "model": e2e.CLASSIFIER,
                    "crops": 4,
                    "worse": False,
                },
                "reference_comparison": {"reference_macro_f1": 0.5, "crops": 4},
                "active_deployments": [],
            }

    return FakeApi()


class FakeStudio:
    """Studio helper: installs succeed, uploads produce no predictions."""

    def wait_install(self, pid, import_id, timeout_s=600):
        return {
            "install_status": "installed",
            "install_error": None,
            "engine_model_id": f"m{import_id}",
            "pipeline_id": 1,
        }

    def primary(self, pid):
        return {
            "id": 1,
            "pipeline_key": "k",
            "pipeline_type": "detector_classifier",
            "detector_server_folder": "d",
            "classifier_server_folder": None,
            "apply_when_detector_class_equals": "animal",
        }

    def upload(self, pid, photos):
        return {}

    def wait_processed(self, ids, timeout_s=900):
        return {}

    def predictions(self, ids):
        return {}

    def all_boxes(self, ids):
        return []

    def label_ids(self, model_key):
        return {}

    def import_row(self, import_id):
        return {"bundle_key": "bundle.zip"}

    def status(self, pid, import_id):
        return {"install_status": "not_requested"}

    def activate(self, pid, import_id):
        return 202


@pytest.mark.parametrize("failed", [None, *PIPES])
def test_ft21_scenario_continues_past_a_pipeline_without_a_model(e2e, tmp_path, failed):
    (tmp_path / "models").mkdir()
    (tmp_path / "studio_models").mkdir()
    (tmp_path / "studio_models" / "bundle.zip").write_bytes(b"0123456789")
    args = SimpleNamespace(
        epochs_cls=1,
        epochs_det=1,
        train_timeout=5,
        val_main=2,
        val_small=1,
        min_acc=0.0,
        min_det_acc=0.0,
    )
    info = {
        "projects": {"train": 1, "base": 2, "stock_gate": 3, "base_pipeline": 1},
        "reviewed_train_images": 3,
        "baseline": {"accuracy": 0.0, "detected": 0.0, "per_class": {}},
    }
    e2e.CHECKS.clear()
    api = _fake_api(e2e, failed)
    try:
        e2e.scenario(
            args,
            FakeStudio(),
            api,
            info,
            tmp_path,
            [("img", "Bos")] * 20,
            {},
        )
        checks = list(e2e.CHECKS)
    finally:
        e2e.CHECKS.clear()

    names = [name for name, _, _ in checks]
    gate_pipes = [p for p in PIPES[:2] if p != failed]
    assert api.activations[: len(gate_pipes)] == [
        (p, str(info["projects"]["stock_gate"])) for p in gate_pipes
    ], (
        "Stock checks must not install models in the later install-failure test's project"
    )
    if failed is not None:
        assert any(failed in name and not ok for name, ok, _ in checks), names
    for pipe in PIPES:
        if pipe != failed:
            assert any(SECTION[pipe] in n for n in names), (pipe, names)
    combination = [(n, ok) for n, ok, _ in checks if any(c in n for c in COMBINATION)]
    if failed in ("conventional_classifier", "yolo_detector"):
        assert not any(ok for _, ok in combination), combination
    else:
        assert combination, names


def test_gpu_scenario_prepares_dino_features_before_submitting_dinov3(e2e, tmp_path):
    """The Studio/GPU runner opts a snapshot into DINO Data Preparation and polls its progress to
    ``ready`` before submitting ``dinov3_linear``; the fake refuses an unprepared snapshot with 422
    exactly like the service, so a runner that skips preparation loses its DINO model."""
    (tmp_path / "models").mkdir()
    (tmp_path / "studio_models").mkdir()
    (tmp_path / "studio_models" / "bundle.zip").write_bytes(b"0123456789")
    args = SimpleNamespace(
        epochs_cls=1,
        epochs_det=1,
        train_timeout=5,
        val_main=2,
        val_small=1,
        min_acc=0.0,
        min_det_acc=0.0,
    )
    info = {
        "projects": {"train": 1, "base": 2, "stock_gate": 3, "base_pipeline": 1},
        "reviewed_train_images": 3,
        "baseline": {"accuracy": 0.0, "detected": 0.0, "per_class": {}},
    }
    e2e.CHECKS.clear()
    api = _fake_api(e2e, None)
    try:
        e2e.scenario(args, FakeStudio(), api, info, tmp_path, [("img", "Bos")] * 20, {})
        checks = list(e2e.CHECKS)
    finally:
        e2e.CHECKS.clear()

    assert api.dino_snapshots, "dinov3_linear was never submitted"
    for sid in api.dino_snapshots:
        assert sid in api.prep, f"dinov3_linear submitted on unprepared snapshot {sid}"
        assert api.prep[sid] >= 2, "submitted before progress reached ready"
    prep = [(n, ok) for n, ok, _ in checks if "DINO" in n or "features" in n]
    assert prep and all(ok for _, ok in prep), prep
    assert any(
        "dinov3_linear trains on real data" in n and ok for n, ok, _ in checks
    ), [n for n, _, _ in checks]
