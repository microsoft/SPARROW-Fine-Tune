# ruff: noqa: F811  (pytest fixtures imported from test_studio_connector)
"""Sparrow Engine packaging of trained models and the Studio activate call.

Spec: docs/decisions.md "2026-10-13 — Studio runs activated fine-tuned models
through Sparrow Engine". The Engine applies softmax to every classifier output, so
classifier graphs must end in log-probabilities; ``yolo_e2e`` expects ``[K, 6]``
rows (x1, y1, x2, y2, score, class) in input pixels with NMS already applied.
"""

from __future__ import annotations

import io
import json
import tarfile
import zipfile

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from sparrow_finetune.connectors import engine_package as ep  # noqa: E402
from sparrow_finetune.connectors import studio as studio_mod  # noqa: E402

from .dinov3_helpers import sha256_file, write_mean_rgb_onnx  # noqa: E402
from .studio_fake import manifest_from, multipart_parts  # noqa: E402
from .test_studio_connector import studio, svc, trained_model  # noqa: E402,F401


def _zip(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data if isinstance(data, bytes) else json.dumps(data))
    return buf.getvalue()


def _run(model_bytes: bytes, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
    (out,) = sess.run(None, {sess.get_inputs()[0].name: x})
    return np.asarray(out)


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- conventional


def conventional_bundle(size: int = 8, n_cls: int = 3) -> tuple[bytes, bytes]:
    w = np.random.default_rng(0).normal(size=(3, n_cls)).astype(np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("GlobalAveragePool", ["images"], ["pool"]),
            helper.make_node("Flatten", ["pool"], ["flat"]),
            helper.make_node("MatMul", ["flat", "w"], ["logits"]),
            helper.make_node("Softmax", ["logits"], ["probabilities"], axis=1),
        ],
        "conv",
        [
            helper.make_tensor_value_info(
                "images", TensorProto.FLOAT, ["N", 3, size, size]
            )
        ],
        [
            helper.make_tensor_value_info(
                "probabilities", TensorProto.FLOAT, ["N", n_cls]
            )
        ],
        initializer=[numpy_helper.from_array(w, "w")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    data = model.SerializeToString()
    manifest = {"input": {"name": "images", "size": size}, "output": "probabilities"}
    return _zip({"model.onnx": data, "manifest.json": manifest}), data


def test_conventional_package_softmax_recovers_probabilities():
    bundle, original = conventional_bundle()
    pkg = ep.build_engine_package("conventional_classifier", bundle, ["a", "b", "c"])
    assert pkg.meta == {
        "kind": "classifier",
        "input_size": 8,
        "preprocessing": "resize",
        "normalization": "unit",
        "postprocessing": "softmax",
        "sha256": pkg.meta["sha256"],
    }
    import hashlib

    assert pkg.meta["sha256"] == hashlib.sha256(pkg.onnx).hexdigest()
    x = np.random.default_rng(1).random((2, 3, 8, 8), dtype=np.float32)
    want = _run(original, x)
    got = _softmax(_run(pkg.onnx, x))
    np.testing.assert_allclose(got, want, atol=1e-6)
    sess = ort.InferenceSession(pkg.onnx, providers=["CPUExecutionProvider"])
    assert [o.name for o in sess.get_outputs()] == ["log_probabilities"]


# --------------------------------------------------------------------------- YOLO


def yolo_bundle(raw: np.ndarray, size: int = 32, output: str | None = None) -> bytes:
    """A stand-in YOLO11 graph: ``images`` [1,3,S,S] -> the constant ``raw`` [1,4+nc,N]."""
    graph = helper.make_graph(
        [
            helper.make_node("ReduceSum", ["images"], ["s"], keepdims=0),
            helper.make_node("Mul", ["s", "zero"], ["z"]),
            helper.make_node("Add", ["raw", "z"], ["output0"]),
        ],
        "yolo",
        [
            helper.make_tensor_value_info(
                "images", TensorProto.FLOAT, [1, 3, size, size]
            )
        ],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, list(raw.shape))],
        initializer=[
            numpy_helper.from_array(raw.astype(np.float32), "raw"),
            numpy_helper.from_array(np.zeros((), np.float32), "zero"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    manifest = {"input": {"name": "images", "size": size, "letterbox": True}}
    if output is not None:
        manifest["output"] = output
    return _zip({"model.onnx": model.SerializeToString(), "manifest.json": manifest})


def test_yolo_package_decodes_and_suppresses():
    # columns: cx, cy, w, h, score(cls0), score(cls1)
    anchors = np.array(
        [
            [10, 10, 8, 8, 0.90, 0.00],  # kept: best cls0
            [
                10.5,
                10,
                8,
                8,
                0.80,
                0.00,
            ],  # suppressed by the first (same class, IoU ~0.9)
            [10, 10, 8, 8, 0.00, 0.60],  # kept: other class, NMS is per class
            [24, 24, 4, 4, 0.01, 0.02],  # dropped: below the score threshold
        ],
        dtype=np.float32,
    )
    raw = anchors.T[None]  # (1, 6, 4)
    pkg = ep.build_engine_package("yolo_detector", yolo_bundle(raw), ["cat", "dog"])
    assert pkg.meta["kind"] == "detector"
    assert pkg.meta["postprocessing"] == "yolo_e2e"
    assert pkg.meta["preprocessing"] == "letterbox"
    assert pkg.meta["normalization"] == "unit"
    assert pkg.meta["input_size"] == 32
    assert pkg.meta["pad_value"] == pytest.approx(114 / 255, abs=1e-6)
    assert 0 < pkg.meta["confidence_threshold"] < 1
    det = _run(pkg.onnx, np.zeros((1, 3, 32, 32), np.float32))
    assert det.ndim == 2 and det.shape[1] == 6
    rows = sorted(det.tolist(), key=lambda r: -r[4])
    assert len(rows) == 2
    np.testing.assert_allclose(rows[0], [6, 6, 14, 14, 0.9, 0], atol=1e-5)
    np.testing.assert_allclose(rows[1], [6, 6, 14, 14, 0.6, 1], atol=1e-5)


def test_yolo_package_with_no_detections_returns_empty_rows():
    raw = np.zeros((1, 6, 5), np.float32)
    pkg = ep.build_engine_package("yolo_detector", yolo_bundle(raw), ["cat", "dog"])
    det = _run(pkg.onnx, np.zeros((1, 3, 32, 32), np.float32))
    assert det.shape == (0, 6)


def test_yolo_package_rejects_label_count_mismatch():
    raw = np.zeros((1, 6, 5), np.float32)
    with pytest.raises(ep.EngineConversionError, match="classes"):
        ep.build_engine_package("yolo_detector", yolo_bundle(raw), ["a", "b", "c"])


def test_yolo_end2end_package_filters_and_suppresses():
    # YOLOv10 rows: x1, y1, x2, y2, score, class (already top-k, no NMS in the graph).
    rows = np.array(
        [
            [6, 6, 14, 14, 0.90, 0],  # kept
            [6.5, 6, 14.5, 14, 0.80, 0],  # suppressed: same class, IoU ~0.9
            [6, 6, 14, 14, 0.60, 1],  # kept: other class
            [20, 20, 24, 24, 0.01, 1],  # dropped: below threshold
            [0, 0, 0, 0, 0.0, 0],  # padding row
        ],
        dtype=np.float32,
    )[None]
    pkg = ep.build_engine_package(
        "yolo_detector", yolo_bundle(rows, output="end2end"), ["cat", "dog"]
    )
    assert pkg.meta["postprocessing"] == "yolo_e2e"
    det = _run(pkg.onnx, np.zeros((1, 3, 32, 32), np.float32))
    got = sorted(det.tolist(), key=lambda r: -r[4])
    assert len(got) == 2
    np.testing.assert_allclose(got[0], [6, 6, 14, 14, 0.9, 0], atol=1e-5)
    np.testing.assert_allclose(got[1], [6, 6, 14, 14, 0.6, 1], atol=1e-5)

    empty = np.zeros((1, 4, 6), np.float32)
    pkg = ep.build_engine_package(
        "yolo_detector", yolo_bundle(empty, output="end2end"), ["cat", "dog"]
    )
    assert _run(pkg.onnx, np.zeros((1, 3, 32, 32), np.float32)).shape == (0, 6)


def test_yolo_end2end_package_rejects_a_raw_head_shape():
    raw = np.zeros((1, 6, 5), np.float32)
    with pytest.raises(ep.EngineConversionError, match=r"\(1, N, 6\)"):
        ep.build_engine_package(
            "yolo_detector", yolo_bundle(raw, output="end2end"), ["cat", "dog"]
        )


# --------------------------------------------------------------------------- DINOv3


def dinov3_bundle(tmp_path, embedder_path) -> tuple[bytes, object]:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    feats = rng.normal(size=(30, 3)).astype(np.float32)
    y = np.arange(30) % 3
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200)).fit(
        feats, y
    )
    head = convert_sklearn(
        clf,
        initial_types=[("features", FloatTensorType([None, 3]))],
        options={id(clf.steps[-1][1]): {"zipmap": False}},
    ).SerializeToString()
    manifest = {
        "embedder": {
            "id": "onnx:mean",
            "dim": 3,
            "model_hash": sha256_file(embedder_path),
        },
        "input": {
            "size": 16,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "head": {"input": "features", "probabilities": "probabilities"},
    }
    return _zip({"head.onnx": head, "manifest.json": manifest}), clf


def test_dinov3_package_merges_embedder_and_head(tmp_path):
    emb = write_mean_rgb_onnx(tmp_path / "emb.onnx")
    bundle, clf = dinov3_bundle(tmp_path, emb)
    pkg = ep.build_engine_package(
        "dinov3_linear", bundle, ["a", "b", "c"], f"onnx:{emb}"
    )
    assert pkg.meta["kind"] == "classifier"
    assert pkg.meta["input_size"] == 16
    assert pkg.meta["normalization"] == "unit", "normalisation is baked into the graph"
    x = np.random.default_rng(2).random((2, 3, 16, 16), dtype=np.float32)
    mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)
    feats = ((x - mean) / std).mean(axis=(2, 3))
    want = clf.predict_proba(feats)
    got = _softmax(_run(pkg.onnx, x))
    np.testing.assert_allclose(got, want, atol=1e-4)


def test_dinov3_package_refuses_other_embedder(tmp_path):
    emb = write_mean_rgb_onnx(tmp_path / "emb.onnx")
    bundle, _ = dinov3_bundle(tmp_path, emb)
    other = write_mean_rgb_onnx(tmp_path / "other.onnx", flatten=False)
    with pytest.raises(ep.EngineConversionError, match="differs"):
        ep.build_engine_package(
            "dinov3_linear", bundle, ["a", "b", "c"], f"onnx:{other}"
        )


def test_dinov3_package_refuses_remote_embedder(tmp_path):
    emb = write_mean_rgb_onnx(tmp_path / "emb.onnx")
    bundle, _ = dinov3_bundle(tmp_path, emb)
    with pytest.raises(ep.EngineConversionError, match="local"):
        ep.build_engine_package(
            "dinov3_linear", bundle, ["a", "b", "c"], "engine:http://e#m"
        )


def test_unknown_pipeline_is_a_conversion_error():
    with pytest.raises(ep.EngineConversionError):
        ep.build_engine_package("fake_cls", b"", ["a"])


# --------------------------------------------------------------------------- push / activate


def _bundle_files(upload) -> tuple[dict, dict]:
    parts = multipart_parts(upload)
    with tarfile.open(fileobj=io.BytesIO(parts["bundle"])) as tar:
        files = {
            m.name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()
        }
    return manifest_from(parts), files


FAKE_ENGINE = ep.EnginePackage(
    onnx=b"engine-graph",
    meta={
        "kind": "classifier",
        "input_size": 8,
        "preprocessing": "resize",
        "normalization": "unit",
        "postprocessing": "softmax",
        "sha256": __import__("hashlib").sha256(b"engine-graph").hexdigest(),
    },
)


@pytest.fixture
def engine_ok(monkeypatch):
    monkeypatch.setattr(studio_mod, "build_engine_package", lambda *a, **k: FAKE_ENGINE)


def test_push_adds_engine_graph_and_manifest_block(
    engine_ok, svc, studio, settings, registry
):
    m = trained_model(svc, settings, registry)
    r = svc.post(
        "/v1/connectors/studio/push", json={"model_id": m["id"], "project": "p1"}
    )
    assert r.status_code == 200, r.text
    manifest, files = _bundle_files(studio.uploads[0])
    assert manifest["engine"] == FAKE_ENGINE.meta
    assert files["engine.onnx"] == b"engine-graph"
    assert json.loads(files["manifest.json"]) == manifest
    assert studio.activations == [], "push alone does not activate"


def test_push_without_engine_package_says_why(svc, studio, settings, registry):
    m = trained_model(svc, settings, registry)  # fake_cls has no Engine packaging
    r = svc.post(
        "/v1/connectors/studio/push", json={"model_id": m["id"], "project": "p1"}
    )
    assert r.status_code == 200, r.text
    manifest, files = _bundle_files(studio.uploads[0])
    assert "engine" not in manifest
    assert "fake_cls" in manifest["engine_unavailable"]
    assert "engine.onnx" not in files


def test_activate_in_studio_requests_install(
    engine_ok, svc, studio, settings, registry
):
    m = trained_model(svc, settings, registry)
    r = svc.post(
        f"/v1/models/{m['id']}/activate", json={"consumer": "studio", "project": "p1"}
    )
    assert r.status_code == 200, r.text
    assert studio.activations == ["studio-model-1"]
    assert r.json()["studio_install"] == "pending"


def test_activate_in_studio_sends_the_chosen_detector(
    engine_ok, svc, studio, settings, registry
):
    m = trained_model(svc, settings, registry)
    act = {"consumer": "studio", "project": "p1"}
    assert svc.post(f"/v1/models/{m['id']}/activate", json=act).status_code == 200
    r = svc.post(
        f"/v1/models/{m['id']}/activate", json={**act, "detector_pipeline_id": 9}
    )
    assert r.status_code == 200, r.text
    # Without a choice Studio gets no body and keeps the project's current detector.
    assert studio.activation_bodies == [None, {"detector_pipeline_id": 9}]


@pytest.mark.parametrize(
    "extra, code",
    [
        ({"consumer": "local", "detector_pipeline_id": 1}, 422),
        ({"consumer": "studio", "detector_pipeline_id": 0}, 422),
        ({"consumer": "studio", "detector_pipeline_id": "x"}, 422),
    ],
)
def test_activate_rejects_a_bad_detector_choice(
    engine_ok, svc, studio, settings, registry, extra, code
):
    m = trained_model(svc, settings, registry)
    r = svc.post(f"/v1/models/{m['id']}/activate", json={"project": "p1", **extra})
    assert r.status_code == code, r.text
    assert studio.uploads == [] and studio.activations == []


def test_activate_refuses_a_detector_choice_for_a_detector_model(
    engine_ok, svc, studio, settings, registry
):
    from sparrow_finetune.db import connect

    m = trained_model(svc, settings, registry)
    with connect(settings.database_url) as conn:
        conn.execute(
            "UPDATE datasets SET task = 'detection', whole_image = false WHERE id = (SELECT s.dataset_id "
            "FROM snapshots s JOIN model_versions m ON m.snapshot_id = s.id "
            "WHERE m.id = %s)",
            (m["id"],),
        )
    r = svc.post(
        f"/v1/models/{m['id']}/activate",
        json={"consumer": "studio", "project": "p1", "detector_pipeline_id": 1},
    )
    assert r.status_code == 422 and "detector model" in r.text
    assert studio.uploads == []


def test_studio_detectors_proxies_the_project_list(svc, studio):
    r = svc.get("/v1/connectors/studio/detectors", params={"project": "p1"})
    assert r.status_code == 200, r.text
    assert r.json() == {"detectors": studio.detectors}
    assert svc.get(
        "/v1/connectors/studio/detectors", params={"project": "nope"}
    ).status_code == 502
    assert svc.get(
        "/v1/connectors/studio/detectors", params={"project": " "}
    ).status_code == 422
    studio.detectors = [{"name": "no id"}]
    assert svc.get(
        "/v1/connectors/studio/detectors", params={"project": "p1"}
    ).status_code == 502


def test_activate_in_studio_without_engine_package_is_import_only(
    svc, studio, settings, registry
):
    m = trained_model(svc, settings, registry)
    r = svc.post(
        f"/v1/models/{m['id']}/activate", json={"consumer": "studio", "project": "p1"}
    )
    assert r.status_code == 200, r.text
    assert studio.activations == []
    assert r.json()["studio_install"].startswith("unavailable:")


def test_activate_refused_by_studio_is_502_and_records_nothing(
    engine_ok, svc, studio, settings, registry
):
    m = trained_model(svc, settings, registry)
    studio.fail_activate = True
    r = svc.post(
        f"/v1/models/{m['id']}/activate", json={"consumer": "studio", "project": "p1"}
    )
    assert r.status_code == 502, r.text
    assert svc.get("/v1/deployments").json()["deployments"] == []
