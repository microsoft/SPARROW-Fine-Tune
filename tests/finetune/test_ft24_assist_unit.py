"""FT-24: assisted-labelling inference (``sparrow_finetune.assist``) without the API.

A hand-built ONNX graph returns a fixed raw YOLO head output, so decode, NMS, letterbox inversion
and bundle loading are checked against hand-computed boxes; letterbox is compared with ultralytics.
"""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest

from sparrow_finetune import assist

IMGSZ = 64


def _raw() -> np.ndarray:
    """``[1, 4 + 2, anchors]`` centre-xywh in 64x64 input pixels; classes cat, dog."""
    anchors = [
        (16, 16, 8, 8, 0.9, 0.1),  # cat
        (16.5, 16, 8, 8, 0.8, 0.0),  # duplicate cat: suppressed by NMS
        (16.5, 16, 8, 8, 0.0, 0.7),  # same place, other class: kept (per-class NMS)
        (40, 40, 10, 10, 0.2, 0.5),  # dog
        (50, 10, 4, 4, 0.1, 0.05),  # below conf
    ]
    return np.asarray(anchors, dtype=np.float32).T[None]


def _onnx_bytes(raw: np.ndarray) -> bytes:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    nodes = [
        helper.make_node("ReduceMean", ["images"], ["m"], keepdims=0),
        helper.make_node(
            "Constant", [], ["zero"], value=numpy_helper.from_array(np.float32(0.0))
        ),
        helper.make_node("Mul", ["m", "zero"], ["z"]),
        helper.make_node("Constant", [], ["head"], value=numpy_helper.from_array(raw)),
        helper.make_node("Add", ["head", "z"], ["output0"]),
    ]
    graph = helper.make_graph(
        nodes,
        "fixed_head",
        [
            helper.make_tensor_value_info(
                "images", TensorProto.FLOAT, [1, 3, IMGSZ, IMGSZ]
            )
        ],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, list(raw.shape))],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model.SerializeToString()


def _bundle(manifest: dict, onnx: bytes | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest))
        zf.writestr("model.onnx", onnx if onnx is not None else _onnx_bytes(_raw()))
    return buf.getvalue()


MANIFEST = {
    "pipeline": "yolo_detector",
    "task": "detection",
    "labels": ["cat", "dog"],
    "input": {"size": IMGSZ},
}


def test_predict_decodes_nms_and_unletterboxes():
    pred = assist.load_predictor(_bundle(MANIFEST))
    # 128x96 image: gain 0.5, resized to 64x48, padded 8 px top and bottom.
    dets = pred.predict(np.zeros((96, 128, 3), dtype=np.uint8), conf=0.25)
    got = [(d.label, [round(v, 3) for v in d.bbox], round(d.score, 3)) for d in dets]
    assert got == [
        ("cat", [24.0, 8.0, 16.0, 16.0], 0.9),
        ("dog", [25.0, 8.0, 16.0, 16.0], 0.7),
        ("dog", [70.0, 54.0, 20.0, 20.0], 0.5),
    ]
    assert [
        d.label for d in pred.predict(np.zeros((96, 128, 3), dtype=np.uint8), conf=0.75)
    ] == ["cat"]


def test_boxes_are_clipped_to_the_image():
    raw = np.asarray([[2, 2, 10, 10, 0.9]], dtype=np.float32).T[None]
    (d,) = assist.decode(raw, ["cat"], 0.25, 1.0, (0, 0), 64, 64)
    assert d.bbox == [0.0, 0.0, 7.0, 7.0]


def test_decoders_report_the_top_score_even_below_conf():
    stats = {}
    raw = np.asarray([[2, 2, 10, 10, 0.08], [5, 5, 10, 10, 0.03]], dtype=np.float32).T[None]
    assert assist.decode(raw, ["cat"], 0.25, 1.0, (0, 0), 64, 64, stats) == []
    assert stats["top_score"] == pytest.approx(0.08)
    e2e = np.array([[[0, 0, 9, 9, 0.12, 0], [0, 0, 9, 9, 0.5, 4]]], dtype=np.float32)
    assert assist.decode_end2end(e2e, ["cat"], 0.25, 1.0, (0, 0), 64, 64, stats) == []
    assert stats["top_score"] == pytest.approx(0.12)  # out-of-range class 4 is ignored


def test_decode_rejects_a_head_that_does_not_match_the_labels():
    with pytest.raises(assist.AssistError, match="does not match"):
        assist.decode(_raw(), ["cat"], 0.25, 1.0, (0, 0), 64, 64)


@pytest.mark.parametrize(
    "bundle, match",
    [
        (b"not a zip", "not a yolo_detector bundle"),
        (
            _bundle(
                {
                    **MANIFEST,
                    "pipeline": "conventional_classifier",
                    "task": "classification",
                }
            ),
            "needs a yolo_detector",
        ),
        (_bundle({**MANIFEST, "labels": []}), "lacks labels"),
        (_bundle({**MANIFEST, "input": {}}), "lacks labels or input size"),
    ],
)
def test_load_predictor_refuses_non_detector_bundles(bundle, match):
    with pytest.raises(assist.AssistError, match=match):
        assist.load_predictor(bundle)


@pytest.mark.parametrize("shape", [(48, 64), (64, 48), (37, 91), (64, 64), (200, 120)])
def test_letterbox_matches_ultralytics(shape):
    augment = pytest.importorskip("ultralytics.data.augment")
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (*shape, 3), dtype=np.uint8)
    ours, gain, pad = assist.letterbox(img, IMGSZ)
    theirs = augment.LetterBox(new_shape=(IMGSZ, IMGSZ), auto=False)(image=img)
    assert ours.shape == (IMGSZ, IMGSZ, 3)
    np.testing.assert_array_equal(ours, theirs)
    assert gain == min(IMGSZ / shape[0], IMGSZ / shape[1])


def test_decode_end2end_maps_letterboxed_xyxy_back_to_the_image():
    import numpy as np

    raw = np.array(
        [[
            [10, 90, 50, 130, 0.9, 0],   # letterbox pad top 80, gain 0.5
            [0, 0, 1, 1, 0.1, 1],        # below conf
            [5, 85, 5, 100, 0.8, 1],     # zero width
            [0, 80, 20, 100, 0.7, 7],    # class out of range
            [11, 91, 50, 130, 0.85, 0],  # near-duplicate of row 0: dropped
            [11, 91, 50, 130, 0.6, 1],   # same box, other class: kept
        ]],
        dtype=np.float32,
    )
    dets = assist.decode_end2end(raw, ["animal", "person"], 0.25, 0.5, (0, 80), 1280, 960)
    assert [(d.label, d.bbox, round(d.score, 3)) for d in dets] == [
        ("animal", [20.0, 20.0, 80.0, 80.0], 0.9),
        ("person", [22.0, 22.0, 78.0, 78.0], 0.6),
    ]
    with pytest.raises(assist.AssistError):
        assist.decode_end2end(np.zeros((1, 300, 7)), ["a"], 0.25, 1, (0, 0), 10, 10)


def test_megadetector_package_lookup_and_labels(tmp_path):
    assert assist.megadetector_dir(None) is None
    assert assist.megadetector_labels(tmp_path) == []
    pkg = tmp_path / assist.MEGADETECTOR_DIR
    (pkg / "1").mkdir(parents=True)
    (pkg / "1" / "model.onnx").write_bytes(b"x")
    assert assist.megadetector_dir(tmp_path) is None  # labels.txt missing
    (pkg / "labels.txt").write_text("person,1\nanimal,0\ncar,2\n")
    assert assist.megadetector_dir(tmp_path) == pkg
    assert assist.megadetector_labels(tmp_path) == ["animal", "person", "vehicle"]
    (pkg / "labels.txt").write_text("animal,0\ncar,5\n")
    with pytest.raises(assist.AssistError):
        assist.megadetector_labels(tmp_path)
    with pytest.raises(assist.AssistError, match="not installed"):
        assist.load_megadetector(tmp_path / "nowhere")


def test_load_predictor_decodes_an_end2end_bundle():
    # (1, N, 6) xyxy, score, class in 64x64 input pixels.
    raw = np.asarray(
        [[[12, 12, 20, 20, 0.9, 1], [40, 40, 50, 50, 0.1, 0], [0, 0, 0, 0, 0, 0]]],
        dtype=np.float32,
    )
    pred = assist.load_predictor(_bundle({**MANIFEST, "output": "end2end"}, _onnx_bytes(raw)))
    assert pred.end2end
    dets = pred.predict(np.zeros((64, 64, 3), dtype=np.uint8), conf=0.25)
    assert [(d.label, [round(v, 3) for v in d.bbox], round(d.score, 3)) for d in dets] == [
        ("dog", [12.0, 12.0, 8.0, 8.0], 0.9)
    ]
    assert not assist.load_predictor(_bundle(MANIFEST)).end2end
