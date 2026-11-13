"""Turn a trained model bundle into one ONNX graph Sparrow Engine can run as is.

Studio runs models only through Sparrow Engine, which supports a fixed set of
pre/post-processing methods (SPARROW-Engine ``manifest.rs``):

* classifiers: ``resize`` + ``softmax``. The Engine always applies softmax, so the
  graph ends in ``log(probabilities)`` (``softmax(log p) == p``).
* detectors: ``letterbox`` + ``yolo_e2e``, which expects ``[K, 6]`` rows
  ``x1, y1, x2, y2, score, class`` in model-input pixels with NMS already done.

``build_engine_package`` returns the graph bytes and the ``engine`` block of the
push manifest. Studio writes the Engine ``manifest.toml`` itself from that block.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

YOLO_SCORE_THRESHOLD = 0.05
YOLO_IOU_THRESHOLD = 0.7
YOLO_MAX_PER_CLASS = 100
# End-to-end heads: one score column, so the NMS cap is the head's own max_det; class offset as ultralytics.
YOLO_E2E_MAX_OUT = 300
YOLO_CLASS_OFFSET = 7680.0
DETECTOR_CONFIDENCE = 0.25
PAD_VALUE = 114.0 / 255.0
_LOG_FLOOR = 1e-30
DEFAULT_ABSTAIN_LABEL = "unknown"
# MegaDetector-style classes: a detector whose labels all fall in this set is generic.
GENERIC_DETECTOR_LABELS = frozenset({"animal", "person", "vehicle"})
GATE_CLASS = "animal"


class EngineConversionError(Exception):
    """The model cannot be packaged for Sparrow Engine (reason in the message)."""


@dataclass(frozen=True)
class EnginePackage:
    onnx: bytes
    meta: dict
    # Top-level push-manifest keys; Studio rejects unknown keys inside ``engine``.
    extra: dict = field(default_factory=dict)


def _read_zip(artifact: bytes) -> tuple[zipfile.ZipFile, dict]:
    try:
        zf = zipfile.ZipFile(io.BytesIO(artifact))
        manifest = json.loads(zf.read("manifest.json"))
    except (zipfile.BadZipFile, KeyError, ValueError) as exc:
        raise EngineConversionError(f"model bundle is unreadable: {exc}") from exc
    return zf, manifest


def _const(name: str, value, dtype=np.float32):
    from onnx import numpy_helper

    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name)


def _opset(model, domain: str = "") -> int:
    for imp in model.opset_import:
        if imp.domain in (domain, "ai.onnx" if domain == "" else domain):
            return int(imp.version)
    return 0


def _append_log_probabilities(
    graph, probs: str, out: str = "log_probabilities"
) -> None:
    """Replace the graph outputs with ``log(max(probs, floor))``."""
    from onnx import TensorProto, helper

    floor = f"{out}__floor"
    graph.initializer.append(_const(floor, _LOG_FLOOR))
    graph.node.extend(
        [
            helper.make_node("Max", [probs, floor], [f"{out}__clamped"]),
            helper.make_node("Log", [f"{out}__clamped"], [out]),
        ]
    )
    del graph.output[:]
    graph.output.append(
        helper.make_tensor_value_info(out, TensorProto.FLOAT, ["batch", "classes"])
    )


def _finish(model, meta: dict) -> EnginePackage:
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(model)
    data = model.SerializeToString()
    try:
        ort.InferenceSession(data, providers=["CPUExecutionProvider"])
    except Exception as exc:  # pragma: no cover - defensive, reported to the caller
        raise EngineConversionError(f"converted graph does not load: {exc}") from exc
    meta = {**meta, "sha256": hashlib.sha256(data).hexdigest()}
    return EnginePackage(onnx=data, meta=meta)


def _input_size(manifest: dict) -> int:
    size = (manifest.get("input") or {}).get("size")
    if not isinstance(size, int) or size <= 0:
        raise EngineConversionError("model manifest has no positive input.size")
    return size


# --------------------------------------------------------------------------- classifiers


def _conventional(artifact: bytes) -> EnginePackage:
    import onnx

    zf, manifest = _read_zip(artifact)
    model = onnx.load_from_string(zf.read("model.onnx"))
    outputs = [o.name for o in model.graph.output]
    probs = manifest.get("output") or "probabilities"
    if probs not in outputs:
        raise EngineConversionError(f"classifier graph has no output {probs!r}")
    _append_log_probabilities(model.graph, probs)
    return _finish(
        model,
        {
            "kind": "classifier",
            "input_size": _input_size(manifest),
            "preprocessing": "resize",
            "normalization": "unit",
            "postprocessing": "softmax",
        },
    )


def _dinov3(artifact: bytes, embedder_spec: str) -> EnginePackage:
    import onnx
    from onnx import TensorProto, compose, helper

    zf, manifest = _read_zip(artifact)
    emb_info = manifest.get("embedder") or {}
    kind, _, rest = (embedder_spec or "").strip().partition(":")
    if kind != "onnx" or not rest.strip():
        raise EngineConversionError(
            "the DINOv3 head needs the embedder graph to run in Sparrow Engine; "
            f"SPARROW_FT_EMBEDDER={embedder_spec!r} is not a local 'onnx:<path>' embedder"
        )
    path = Path(rest.strip())
    if not path.is_file():
        raise EngineConversionError(f"embedder model not found: {path}")
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    want = emb_info.get("model_hash")
    if want and digest.hexdigest() != want:
        raise EngineConversionError(
            "the configured embedder differs from the one the head was trained on "
            f"(sha256 {digest.hexdigest()[:12]} != {str(want)[:12]})"
        )

    head = onnx.load_from_string(zf.read("head.onnx"))
    head_meta = manifest.get("head") or {}
    head_in = head_meta.get("input") or head.graph.input[0].name
    head_probs = head_meta.get("probabilities") or "probabilities"
    if head_probs not in [o.name for o in head.graph.output]:
        raise EngineConversionError(f"head graph has no output {head_probs!r}")

    emb = onnx.load(str(path), load_external_data=True)
    size = int((manifest.get("input") or {}).get("size") or 224)
    mean = (manifest.get("input") or {}).get("mean") or [0.485, 0.456, 0.406]
    std = (manifest.get("input") or {}).get("std") or [0.229, 0.224, 0.225]

    emb = compose.add_prefix(emb, "emb/")
    head = compose.add_prefix(head, "head/")
    emb_in = emb.graph.input[0].name
    emb_out = emb.graph.output[0].name

    nodes = [
        helper.make_node("Sub", ["images", "pre/mean"], ["pre/centered"]),
        helper.make_node("Div", ["pre/centered", "pre/std"], [emb_in]),
        *emb.graph.node,
        helper.make_node("Reshape", [emb_out, "pre/flat"], [f"head/{head_in}"]),
        *head.graph.node,
    ]
    inits = [
        _const("pre/mean", np.asarray(mean, np.float32).reshape(1, 3, 1, 1)),
        _const("pre/std", np.asarray(std, np.float32).reshape(1, 3, 1, 1)),
        _const("pre/flat", [0, -1], np.int64),
        *emb.graph.initializer,
        *head.graph.initializer,
    ]
    graph = helper.make_graph(
        nodes,
        "dinov3_linear_engine",
        [
            helper.make_tensor_value_info(
                "images", TensorProto.FLOAT, ["batch", 3, size, size]
            )
        ],
        [],
        initializer=inits,
    )
    _append_log_probabilities(graph, f"head/{head_probs}")
    opsets: dict[str, int] = {}
    for m in (emb, head):
        for imp in m.opset_import:
            dom = "" if imp.domain == "ai.onnx" else imp.domain
            opsets[dom] = max(opsets.get(dom, 0), int(imp.version))
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid(d, v) for d, v in opsets.items()],
        producer_name="sparrow-finetune",
    )
    model.ir_version = max(emb.ir_version, head.ir_version)
    return _finish(
        model,
        {
            "kind": "classifier",
            "input_size": size,
            "preprocessing": "resize",
            "normalization": "unit",
            "postprocessing": "softmax",
        },
    )


# --------------------------------------------------------------------------- detectors


def _yolo(artifact: bytes, labels: list) -> EnginePackage:
    import onnx
    from onnx import TensorProto, helper

    zf, manifest = _read_zip(artifact)
    model = onnx.load_from_string(zf.read("model.onnx"))
    if _opset(model) < 13:
        raise EngineConversionError("YOLO graph opset < 13 is not supported")
    if len(model.graph.output) != 1:
        raise EngineConversionError("YOLO graph must have exactly one output")
    raw = model.graph.output[0].name
    nc = len(labels)
    dims = [d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim]
    if manifest.get("output") == "end2end":
        if len(dims) != 3 or dims[2] != 6:
            raise EngineConversionError(
                f"end-to-end YOLO output shape {dims} does not match (1, N, 6)"
            )
        _yolo_end2end_tail(model.graph, raw)
        return _finish(model, _yolo_meta(manifest))
    if len(dims) != 3 or dims[1] not in (0, 4 + nc):
        raise EngineConversionError(
            f"YOLO output shape {dims} does not match (1, 4 + {nc} classes, N)"
        )
    g = model.graph
    p = "engine/"
    g.initializer.extend(
        [
            _const(p + "ax0", [0], np.int64),
            _const(p + "ax1", [1], np.int64),
            _const(p + "c0", [0], np.int64),
            _const(p + "c4", [4], np.int64),
            _const(p + "cend", [4 + nc], np.int64),
            _const(p + "i1", 1, np.int64),
            _const(p + "i2", 2, np.int64),
            _const(
                p + "xywh2xyxy",
                [[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
            ),
            _const(p + "max_out", [YOLO_MAX_PER_CLASS], np.int64),
            _const(p + "iou", [YOLO_IOU_THRESHOLD]),
            _const(p + "score", [YOLO_SCORE_THRESHOLD]),
        ]
    )
    n = helper.make_node
    g.node.extend(
        [
            n("Squeeze", [raw, p + "ax0"], [p + "sq"]),
            n("Transpose", [p + "sq"], [p + "rows"], perm=[1, 0]),
            n("Slice", [p + "rows", p + "c0", p + "c4", p + "ax1"], [p + "xywh"]),
            n("Slice", [p + "rows", p + "c4", p + "cend", p + "ax1"], [p + "cls"]),
            n("MatMul", [p + "xywh", p + "xywh2xyxy"], [p + "xyxy"]),
            n("Unsqueeze", [p + "xyxy", p + "ax0"], [p + "boxes3"]),
            n("Transpose", [p + "cls"], [p + "cls_t"], perm=[1, 0]),
            n("Unsqueeze", [p + "cls_t", p + "ax0"], [p + "scores3"]),
            n(
                "NonMaxSuppression",
                [p + "boxes3", p + "scores3", p + "max_out", p + "iou", p + "score"],
                [p + "sel"],
            ),
            n("Gather", [p + "sel", p + "i1"], [p + "cls_idx"], axis=1),
            n("Gather", [p + "sel", p + "i2"], [p + "box_idx"], axis=1),
            n("Gather", [p + "xyxy", p + "box_idx"], [p + "sel_boxes"], axis=0),
            n("Unsqueeze", [p + "box_idx", p + "ax1"], [p + "box_col"]),
            n("Unsqueeze", [p + "cls_idx", p + "ax1"], [p + "cls_col"]),
            n("Concat", [p + "box_col", p + "cls_col"], [p + "pairs"], axis=1),
            n("GatherND", [p + "cls", p + "pairs"], [p + "sel_score"]),
            n("Unsqueeze", [p + "sel_score", p + "ax1"], [p + "score_col"]),
            n("Cast", [p + "cls_col"], [p + "cls_f"], to=TensorProto.FLOAT),
            n(
                "Concat",
                [p + "sel_boxes", p + "score_col", p + "cls_f"],
                ["detections"],
                axis=1,
            ),
        ]
    )
    del g.output[:]
    g.output.append(
        helper.make_tensor_value_info(
            "detections", TensorProto.FLOAT, ["num_detections", 6]
        )
    )
    return _finish(model, _yolo_meta(manifest))


def _yolo_meta(manifest: dict) -> dict:
    return {
        "kind": "detector",
        "input_size": _input_size(manifest),
        "preprocessing": "letterbox",
        "normalization": "unit",
        "pad_value": round(PAD_VALUE, 6),
        "postprocessing": "yolo_e2e",
        "confidence_threshold": DETECTOR_CONFIDENCE,
    }


def _yolo_end2end_tail(g, raw: str) -> None:
    """``(1, N, 6)`` end-to-end rows -> ``detections`` ``[num_detections, 6]``.

    The NMS-free head still emits the odd near-duplicate box, so a same-class NMS runs over it (boxes
    offset by class so classes never suppress each other), the same pass ``assist.decode_end2end`` makes.
    """
    from onnx import TensorProto, helper

    p = "engine/"
    g.initializer.extend(
        [
            _const(p + "ax0", [0], np.int64),
            _const(p + "ax1", [1], np.int64),
            _const(p + "c0", [0], np.int64),
            _const(p + "c4", [4], np.int64),
            _const(p + "c5", [5], np.int64),
            _const(p + "c6", [6], np.int64),
            _const(p + "i2", 2, np.int64),
            _const(p + "class_offset", [YOLO_CLASS_OFFSET]),
            _const(p + "max_out", [YOLO_E2E_MAX_OUT], np.int64),
            _const(p + "iou", [YOLO_IOU_THRESHOLD]),
            _const(p + "score", [YOLO_SCORE_THRESHOLD]),
        ]
    )
    n = helper.make_node
    g.node.extend(
        [
            n("Squeeze", [raw, p + "ax0"], [p + "rows"]),
            n("Slice", [p + "rows", p + "c0", p + "c4", p + "ax1"], [p + "xyxy"]),
            n("Slice", [p + "rows", p + "c4", p + "c5", p + "ax1"], [p + "score_col"]),
            n("Slice", [p + "rows", p + "c5", p + "c6", p + "ax1"], [p + "cls_col"]),
            n("Mul", [p + "cls_col", p + "class_offset"], [p + "offset"]),
            n("Add", [p + "xyxy", p + "offset"], [p + "xyxy_off"]),
            n("Unsqueeze", [p + "xyxy_off", p + "ax0"], [p + "boxes3"]),
            n("Transpose", [p + "score_col"], [p + "score_row"], perm=[1, 0]),
            n("Unsqueeze", [p + "score_row", p + "ax0"], [p + "scores3"]),
            n(
                "NonMaxSuppression",
                [p + "boxes3", p + "scores3", p + "max_out", p + "iou", p + "score"],
                [p + "sel"],
            ),
            n("Gather", [p + "sel", p + "i2"], [p + "idx"], axis=1),
            n("Gather", [p + "rows", p + "idx"], ["detections"], axis=0),
        ]
    )
    del g.output[:]
    g.output.append(
        helper.make_tensor_value_info(
            "detections", TensorProto.FLOAT, ["num_detections", 6]
        )
    )


def detector_hints(labels: list) -> dict:
    """Pairing hints for Studio: generic (MegaDetector-like) vs species detector."""
    names = [str(x) for x in labels]
    lowered = {n.strip().lower() for n in names}
    generic = bool(lowered) and lowered <= GENERIC_DETECTOR_LABELS
    gate = None
    if generic:
        gate = next((n for n in names if n.strip().lower() == GATE_CLASS), None)
    return {
        "detector_labels": names,
        "detector_kind": "generic" if generic else "species",
        "detector_gate_class": gate,
    }


def _unit_float(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        return None
    return value


def _valid_label(value) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= 255
        and value.strip() == value
        and not any(c in value for c in ",\n\r")
    )


def _calibration_sources(artifact: bytes, metrics, params) -> list[dict]:
    sources = [d for d in (metrics, params) if isinstance(d, dict)]
    try:
        _, manifest = _read_zip(artifact)
    except EngineConversionError:
        return sources
    if isinstance(manifest, dict):
        sources.append(manifest)
        sources += [
            manifest[k]
            for k in ("params", "metrics")
            if isinstance(manifest.get(k), dict)
        ]
    return sources


def _with_calibration(
    pkg: EnginePackage, artifact: bytes, metrics, params
) -> EnginePackage:
    sources = _calibration_sources(artifact, metrics, params)
    thresholds = (_unit_float(s.get("calibrated_threshold")) for s in sources)
    threshold = next((t for t in thresholds if t is not None), None)
    if threshold is None:
        return pkg
    abstain = next(
        (s["abstain_label"] for s in sources if _valid_label(s.get("abstain_label"))),
        DEFAULT_ABSTAIN_LABEL,
    )
    return EnginePackage(
        onnx=pkg.onnx,
        meta={**pkg.meta, "confidence_threshold": threshold},
        extra={
            **pkg.extra,
            "calibrated_threshold": threshold,
            "abstain_label": abstain,
        },
    )


def build_engine_package(
    pipeline: str,
    artifact: bytes,
    labels: list,
    embedder_spec: str = "",
    metrics: dict | None = None,
    params: dict | None = None,
) -> EnginePackage:
    """Engine-ready graph for a model bundle; ``EngineConversionError`` when impossible.

    For classifiers, a ``calibrated_threshold`` (unit float) from ``metrics``,
    ``params`` or the bundle manifest (first valid wins) becomes the Engine
    ``confidence_threshold``; it and the abstain label (default ``"unknown"``) are
    also returned in ``extra`` for the top level of the push manifest.
    """
    if pipeline == "conventional_classifier":
        pkg = _conventional(artifact)
    elif pipeline == "yolo_detector":
        return _yolo(artifact, labels)
    elif pipeline == "dinov3_linear":
        pkg = _dinov3(artifact, embedder_spec)
    else:
        raise EngineConversionError(
            f"pipeline {pipeline!r} has no Sparrow Engine packaging"
        )
    return _with_calibration(pkg, artifact, metrics, params)
