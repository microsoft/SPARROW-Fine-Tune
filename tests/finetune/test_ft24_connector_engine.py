"""FT-24 connector: classifier calibrated threshold + abstain label, detector hints.

The Studio dashboard rejects unknown keys inside ``manifest.engine``
(``finetune_api._ENGINE_KEYS``), so only ``confidence_threshold`` (an allowed key) goes
there; the abstain label and detector hints travel as top-level manifest keys
(``EnginePackage.extra`` and ``engine_package.detector_hints``).
"""

from __future__ import annotations

import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from sparrow_finetune.connectors import engine_package as ep  # noqa: E402

from .test_engine_package import _zip, conventional_bundle  # noqa: E402

STUDIO_ENGINE_KEYS = {
    "kind",
    "sha256",
    "input_size",
    "preprocessing",
    "normalization",
    "postprocessing",
    "pad_value",
    "confidence_threshold",
}


def test_classifier_without_calibration_keeps_defaults():
    bundle, _ = conventional_bundle()
    pkg = ep.build_engine_package(
        "conventional_classifier", bundle, ["a", "b", "c"], metrics={"accuracy": 0.9}
    )
    assert "confidence_threshold" not in pkg.meta
    assert pkg.extra == {}


def test_classifier_calibrated_threshold_from_metrics():
    bundle, _ = conventional_bundle()
    pkg = ep.build_engine_package(
        "conventional_classifier",
        bundle,
        ["a", "b", "c"],
        metrics={"accuracy": 0.9, "calibrated_threshold": 0.62},
    )
    assert pkg.meta["confidence_threshold"] == pytest.approx(0.62)
    assert set(pkg.meta) <= STUDIO_ENGINE_KEYS
    assert pkg.extra == {"calibrated_threshold": 0.62, "abstain_label": "unknown"}


def test_classifier_threshold_and_abstain_from_params_or_bundle():
    bundle, _ = conventional_bundle()
    pkg = ep.build_engine_package(
        "conventional_classifier",
        bundle,
        ["a", "b", "c"],
        metrics={},
        params={"calibrated_threshold": 0.4, "abstain_label": "unsure"},
    )
    assert pkg.meta["confidence_threshold"] == pytest.approx(0.4)
    assert pkg.extra["abstain_label"] == "unsure"

    # The bundle's own manifest.json is the last place looked.
    import io
    import json
    import zipfile

    zf = zipfile.ZipFile(io.BytesIO(bundle))
    manifest = json.loads(zf.read("manifest.json"))
    manifest["calibrated_threshold"] = 0.3
    rebundled = _zip({"model.onnx": zf.read("model.onnx"), "manifest.json": manifest})
    pkg = ep.build_engine_package("conventional_classifier", rebundled, ["a", "b", "c"])
    assert pkg.meta["confidence_threshold"] == pytest.approx(0.3)
    assert pkg.extra["abstain_label"] == "unknown"


@pytest.mark.parametrize("bad", [1.5, -0.1, float("nan"), True, "0.5", None])
def test_invalid_calibrated_threshold_falls_back(bad):
    bundle, _ = conventional_bundle()
    pkg = ep.build_engine_package(
        "conventional_classifier",
        bundle,
        ["a", "b", "c"],
        metrics={"calibrated_threshold": bad},
    )
    assert "confidence_threshold" not in pkg.meta
    assert pkg.extra == {}


def test_invalid_abstain_label_uses_default():
    bundle, _ = conventional_bundle()
    pkg = ep.build_engine_package(
        "conventional_classifier",
        bundle,
        ["a", "b", "c"],
        metrics={"calibrated_threshold": 0.5, "abstain_label": "a,b"},
    )
    assert pkg.extra["abstain_label"] == "unknown"


@pytest.mark.parametrize(
    "labels, kind, gate",
    [
        (["animal", "person", "vehicle"], "generic", "animal"),
        (["Animal"], "generic", "Animal"),
        (["person", "vehicle"], "generic", None),
        (["deer", "fox"], "species", None),
        (["animal", "fox"], "species", None),
        ([], "species", None),
    ],
)
def test_detector_hints(labels, kind, gate):
    hints = ep.detector_hints(labels)
    assert hints == {
        "detector_labels": labels,
        "detector_kind": kind,
        "detector_gate_class": gate,
    }
