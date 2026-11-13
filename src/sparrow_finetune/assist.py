"""Assisted labelling: run a detector over dataset images to pre-draw boxes.

The default detector is MegaDetector v6 (compact, YOLOv10-c), read from the Sparrow engine model
package ``<SPARROW_FT_WEIGHTS_DIR>/MDV6-yolov10-c/`` (``1/model.onnx`` + ``labels.txt``). It finds
animals, people and vehicles; reviewers relabel ``animal`` boxes to species. Its ONNX is end-to-end
(``[1, 300, 6]`` rows of letterboxed ``x0, y0, x1, y1, score, class``; a same-class IoU 0.7 pass
removes the occasional near-duplicate box the NMS-free head emits).

A trained detector is the stored ``onnx-bundle`` (``model.onnx`` + ``manifest.json``) a completed
``yolo_detector`` job leaves in the object store; the job workdir and ``best.pt`` are not kept, so
inference uses onnxruntime on CPU. Pre- and post-processing mirror ultralytics ``predict`` so the
suggestions match what the workbench pre-annotation (``sparrow_web``) produced with the same weights:
``LetterBox(auto=False)`` with grey 114 padding, RGB/255 NCHW input, class = arg-max score, per-class
NMS at IoU 0.7, at most 300 boxes, boxes scaled back to the image and clipped.
"""

from __future__ import annotations

import functools
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

DEFAULT_CONF = 0.25
IOU_THRESHOLD = 0.7
MAX_DET = 300
PAD_VALUE = 114
_MAX_WH = 7680  # ultralytics: per-class box offset so one NMS pass keeps classes apart


MEGADETECTOR_ID = "megadetector"
MEGADETECTOR_NAME = "MegaDetector v6 (compact)"
MEGADETECTOR_DIR = "MDV6-yolov10-c"
MEGADETECTOR_IMGSZ = 640
# The engine package names MegaDetector's third class "car"; MegaDetector calls it "vehicle".
_MD_LABEL_NAMES = {"car": "vehicle"}


class AssistError(ValueError):
    """The model cannot be used for assisted labelling (the message says why)."""


@dataclass(frozen=True)
class Detection:
    label: str
    bbox: list  # [x, y, w, h] in image pixels
    score: float


def letterbox(rgb: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize keeping aspect ratio and pad to ``size`` x ``size`` like ultralytics ``LetterBox``.

    Returns the padded image, the gain and the ``(left, top)`` padding."""
    import cv2

    h, w = rgb.shape[:2]
    r = min(size / h, size / w)
    new_w, new_h = round(w * r), round(h * r)
    dw, dh = (size - new_w) / 2, (size - new_h) / 2
    if (w, h) != (new_w, new_h):
        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    rgb = cv2.copyMakeBorder(
        rgb, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3
    )
    return rgb, r, (left, top)


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> list[int]:
    """Greedy NMS over ``xyxy`` boxes; indices kept, highest score first."""
    order = np.argsort(-scores, kind="stable")
    x0, y0, x1, y1 = boxes.T
    areas = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        ix0 = np.maximum(x0[i], x0[rest])
        iy0 = np.maximum(y0[i], y0[rest])
        ix1 = np.minimum(x1[i], x1[rest])
        iy1 = np.minimum(y1[i], y1[rest])
        inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
        union = areas[i] + areas[rest] - inter
        overlap = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = rest[overlap <= iou]
    return keep


def _record_top(stats: Optional[dict], scores: np.ndarray) -> None:
    """Store the best raw score in ``stats["top_score"]``. NMS always keeps the best box, so
    this is the highest score a lower ``conf`` would let through."""
    if stats is not None:
        stats["top_score"] = float(scores.max()) if len(scores) else 0.0


def decode(
    raw: np.ndarray,
    labels: Sequence[str],
    conf: float,
    gain: float,
    pad: tuple[int, int],
    width: int,
    height: int,
    stats: Optional[dict] = None,
) -> list[Detection]:
    """Raw YOLO output ``[1, 4 + classes, anchors]`` (centre xywh in input pixels) -> detections."""
    rows = np.asarray(raw, dtype=np.float32)
    if rows.ndim == 3:
        rows = rows[0]
    if rows.ndim != 2 or rows.shape[0] != 4 + len(labels):
        raise AssistError(
            f"model output shape {np.asarray(raw).shape} does not match {len(labels)} labels"
        )
    rows = rows.T
    cls_scores = rows[:, 4:]
    cls = cls_scores.argmax(axis=1)
    score = cls_scores[np.arange(len(rows)), cls]
    _record_top(stats, score)
    keep = score > conf
    rows, cls, score = rows[keep], cls[keep], score[keep]
    if not len(rows):
        return []
    cx, cy, w, h = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    kept = nms(xyxy + cls[:, None] * _MAX_WH, score, IOU_THRESHOLD)[:MAX_DET]
    left, top = pad
    out = []
    for i in kept:
        x0 = float(np.clip((xyxy[i, 0] - left) / gain, 0, width))
        y0 = float(np.clip((xyxy[i, 1] - top) / gain, 0, height))
        x1 = float(np.clip((xyxy[i, 2] - left) / gain, 0, width))
        y1 = float(np.clip((xyxy[i, 3] - top) / gain, 0, height))
        if x1 - x0 <= 0 or y1 - y0 <= 0:
            continue
        out.append(
            Detection(
                label=str(labels[int(cls[i])]),
                bbox=[x0, y0, x1 - x0, y1 - y0],
                score=float(score[i]),
            )
        )
    return out


def decode_end2end(
    raw: np.ndarray,
    labels: Sequence[str],
    conf: float,
    gain: float,
    pad: tuple[int, int],
    width: int,
    height: int,
    stats: Optional[dict] = None,
) -> list[Detection]:
    """End-to-end YOLO output ``[1, N, 6]`` (letterboxed ``xyxy``, score, class) -> detections."""
    rows = np.asarray(raw, dtype=np.float32)
    if rows.ndim == 3:
        rows = rows[0]
    if rows.ndim != 2 or rows.shape[1] != 6:
        raise AssistError(f"model output shape {np.asarray(raw).shape} is not [1, N, 6]")
    valid = (rows[:, 5] >= 0) & (rows[:, 5] < len(labels))
    _record_top(stats, rows[valid, 4])
    rows = rows[rows[:, 4] > conf]
    rows = rows[np.argsort(-rows[:, 4], kind="stable")][:MAX_DET]
    if len(rows):
        # The NMS-free head still emits the odd near-duplicate box (seen live on an ENA24 frame,
        # IoU ~0.98); a same-class pass at the usual 0.7 keeps reviewers from deleting them by hand.
        rows = rows[nms(rows[:, :4] + rows[:, 5:6] * _MAX_WH, rows[:, 4], IOU_THRESHOLD)]
    left, top = pad
    out = []
    for x0, y0, x1, y1, score, cls in rows:
        c = int(cls)
        if not 0 <= c < len(labels):
            continue
        x0 = float(np.clip((x0 - left) / gain, 0, width))
        y0 = float(np.clip((y0 - top) / gain, 0, height))
        x1 = float(np.clip((x1 - left) / gain, 0, width))
        y1 = float(np.clip((y1 - top) / gain, 0, height))
        if x1 - x0 <= 0 or y1 - y0 <= 0:
            continue
        out.append(Detection(label=str(labels[c]), bbox=[x0, y0, x1 - x0, y1 - y0], score=float(score)))
    return out


def read_rgb(path: Path | str) -> np.ndarray:
    """Raw pixels as stored (no EXIF rotation), the frame dataset boxes and training use."""
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


class YoloOnnxPredictor:
    """One onnxruntime session over a ``yolo_detector`` bundle."""

    def __init__(
        self, onnx: bytes, labels: Sequence[str], imgsz: int, end2end: bool = False
    ) -> None:
        import onnxruntime as ort

        self.labels = [str(x) for x in labels]
        self.imgsz = int(imgsz)
        self.end2end = bool(end2end)
        self._session = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
        self._input = self._session.get_inputs()[0].name

    def predict(
        self, rgb: np.ndarray, conf: float = DEFAULT_CONF, stats: Optional[dict] = None
    ) -> list[Detection]:
        """Detections scoring above ``conf``; with ``stats``, also ``stats["top_score"]``."""
        height, width = rgb.shape[:2]
        img, gain, pad = letterbox(rgb, self.imgsz)
        x = np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32)[None] / 255.0
        raw = self._session.run(None, {self._input: x})[0]
        fn = decode_end2end if self.end2end else decode
        return fn(raw, self.labels, conf, gain, pad, width, height, stats)


def load_predictor(bundle: bytes) -> YoloOnnxPredictor:
    """Predictor for a stored ``onnx-bundle``; ``AssistError`` when it is not a detector bundle."""
    try:
        with zipfile.ZipFile(io.BytesIO(bundle)) as zf:
            manifest = json.loads(zf.read("manifest.json"))
            onnx = zf.read("model.onnx")
    except (zipfile.BadZipFile, KeyError, ValueError) as exc:
        raise AssistError(
            f"model artifact is not a yolo_detector bundle: {exc}"
        ) from exc
    if (
        manifest.get("pipeline") != "yolo_detector"
        or manifest.get("task") != "detection"
    ):
        raise AssistError(
            f"model is {manifest.get('pipeline')!r} ({manifest.get('task')!r}); "
            "assisted labelling needs a yolo_detector detection model"
        )
    labels = manifest.get("labels") or []
    size = (manifest.get("input") or {}).get("size")
    if not labels or not isinstance(size, int) or size <= 0:
        raise AssistError("model manifest lacks labels or input size")
    return YoloOnnxPredictor(
        onnx, labels, size, end2end=manifest.get("output") == "end2end"
    )


def megadetector_dir(weights_dir) -> Path | None:
    """The MegaDetector package directory under ``weights_dir``; None when it is not installed."""
    if weights_dir is None:
        return None
    root = Path(weights_dir) / MEGADETECTOR_DIR
    return root if _md_onnx(root) is not None and (root / "labels.txt").is_file() else None


def _md_onnx(root: Path) -> Path | None:
    for candidate in (root / "1" / "model.onnx", root / "model.onnx"):
        if candidate.is_file():
            return candidate
    return None


def _md_labels(text: str) -> list[str]:
    """``labels.txt`` as ``name,index`` lines (the engine's ``name_index_csv``) or one name per line."""
    pairs = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        name, sep, index = line.rpartition(",")
        if sep and index.strip().isdigit():
            pairs.append((int(index), name.strip()))
        else:
            pairs.append((len(pairs), line))
    pairs.sort()
    if [i for i, _ in pairs] != list(range(len(pairs))):
        raise AssistError("MegaDetector labels.txt indices are not 0..n-1")
    return [_MD_LABEL_NAMES.get(n, n) for _, n in pairs]


def megadetector_labels(weights_dir) -> list[str]:
    root = megadetector_dir(weights_dir)
    if root is None:
        return []
    return _md_labels((root / "labels.txt").read_text())


def load_megadetector(weights_dir) -> YoloOnnxPredictor:
    root = megadetector_dir(weights_dir)
    if root is None:
        raise AssistError(
            f"MegaDetector is not installed: put the {MEGADETECTOR_DIR} model package "
            "(1/model.onnx and labels.txt) in the directory SPARROW_FT_WEIGHTS_DIR names"
        )
    labels = _md_labels((root / "labels.txt").read_text())
    return YoloOnnxPredictor(
        _md_onnx(root).read_bytes(), labels, MEGADETECTOR_IMGSZ, end2end=True
    )


SPECIESNET_ID = "speciesnet"
SPECIESNET_NAME = "SpeciesNet (crop classifier)"
SPECIESNET_DIR = "SpeciesNet-Crop"
SPECIESNET_IMGSZ = 480
# Classes that never name what is in a MegaDetector animal box.
_SPECIESNET_SKIP = {"blank", "novehicle"}


def label_key(name: str) -> str:
    """Compare labels the way SpeciesNet spells them: lowercase letters and digits only."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def speciesnet_dir(weights_dir) -> Path | None:
    """The SpeciesNet crop-classifier package under ``weights_dir``; None when not installed."""
    if weights_dir is None:
        return None
    root = Path(weights_dir) / SPECIESNET_DIR
    return root if _md_onnx(root) is not None and (root / "labels.txt").is_file() else None


TAXONOMY_FILE = Path(__file__).parent / "data" / "speciesnet_taxonomy.txt"
# A dataset label must collect at least this much SpeciesNet probability to be chosen.
SPECIESNET_MIN_MATCH = 0.2


def speciesnet_taxonomy(n_labels: int) -> list[dict] | None:
    """Per output index: the common name and the label keys (common name, binomial, genus,
    family, order, class) it counts towards. None when the vendored file does not fit."""
    if not TAXONOMY_FILE.is_file():
        return None
    rows = []
    for line in TAXONOMY_FILE.read_text().splitlines():
        parts = line.split(";")
        if len(parts) != 7:
            return None
        _, cls, order, family, genus, species, common = (x.strip() for x in parts)
        binomial = f"{genus} {species}".strip() if genus and species else ""
        name = common or binomial or genus or family or order or cls
        keys = {label_key(x) for x in (common, binomial, genus, family, order, cls) if x}
        rows.append({"name": name, "keys": keys - {""}})
    return rows if len(rows) == n_labels else None


class SpeciesNetCropClassifier:
    """SpeciesNet's crop classifier: one box crop resized to 480x480, NHWC RGB in [0, 1],
    softmax over its species and taxon classes."""

    def __init__(self, onnx: bytes | Path, labels: Sequence[str], imgsz: int = SPECIESNET_IMGSZ):
        import onnxruntime as ort

        self.labels = [str(x) for x in labels]
        self.imgsz = int(imgsz)
        self.taxonomy = speciesnet_taxonomy(len(self.labels)) or [
            {"name": x, "keys": {label_key(x)}} for x in self.labels
        ]
        self._session = ort.InferenceSession(
            str(onnx) if isinstance(onnx, Path) else onnx, providers=["CPUExecutionProvider"]
        )
        self._input = self._session.get_inputs()[0].name

    def scores(self, rgb: np.ndarray, bbox: Sequence[float]) -> np.ndarray:
        """Class probabilities for the ``[x, y, w, h]`` box of ``rgb``."""
        import cv2

        height, width = rgb.shape[:2]
        x, y, w, h = bbox
        x0, y0 = max(0, int(np.floor(x))), max(0, int(np.floor(y)))
        x1, y1 = min(width, int(np.ceil(x + w))), min(height, int(np.ceil(y + h)))
        crop = rgb[y0:y1, x0:x1] if x1 > x0 and y1 > y0 else rgb
        crop = cv2.resize(crop, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        logits = self._session.run(
            None, {self._input: crop.astype(np.float32)[None] / 255.0}
        )[0][0].astype(np.float64)
        exp = np.exp(logits - logits.max())
        return exp / exp.sum()

    def label_for(
        self, rgb: np.ndarray, bbox: Sequence[float], known: Sequence[str] = ()
    ) -> tuple[str, float]:
        """The box's label and its probability.

        Each dataset label collects the probability of every SpeciesNet class it names by common
        name, binomial, genus, family, order or class (a dataset labelled ``Dasyprocta`` gets
        all agouti species). The label with the most probability wins and keeps the dataset's
        spelling, provided it reaches ``SPECIESNET_MIN_MATCH``; otherwise the box takes
        SpeciesNet's best class by common name."""
        probs = self.scores(rgb, bbox)
        best_label, best_p = None, 0.0
        for label in dict.fromkeys(known):
            key = label_key(label)
            if not key:
                continue
            p = float(sum(probs[i] for i, t in enumerate(self.taxonomy) if key in t["keys"]))
            if p > best_p:
                best_label, best_p = label, p
        if best_label is not None and best_p >= SPECIESNET_MIN_MATCH:
            return best_label, best_p
        order = [i for i in np.argsort(-probs) if self.labels[i] not in _SPECIESNET_SKIP]
        return self.taxonomy[order[0]]["name"], float(probs[order[0]])


def speciesnet_available(weights_dir) -> bool:
    return speciesnet_dir(weights_dir) is not None


def load_speciesnet(weights_dir) -> SpeciesNetCropClassifier:
    """The SpeciesNet classifier, loaded once per process (the model is about 220 MB)."""
    root = speciesnet_dir(weights_dir)
    if root is None:
        raise AssistError(
            f"SpeciesNet is not installed: put the {SPECIESNET_DIR} model package "
            "(model.onnx and labels.txt) in the directory SPARROW_FT_WEIGHTS_DIR names"
        )
    return _speciesnet_cached(str(root))


@functools.lru_cache(maxsize=1)
def _speciesnet_cached(root: str) -> SpeciesNetCropClassifier:
    path = Path(root)
    return SpeciesNetCropClassifier(_md_onnx(path), _md_labels((path / "labels.txt").read_text()))


__all__ = [
    "AssistError",
    "DEFAULT_CONF",
    "Detection",
    "YoloOnnxPredictor",
    "MEGADETECTOR_ID",
    "MEGADETECTOR_NAME",
    "SPECIESNET_ID",
    "SPECIESNET_NAME",
    "SpeciesNetCropClassifier",
    "label_key",
    "load_speciesnet",
    "speciesnet_available",
    "speciesnet_dir",
    "decode",
    "decode_end2end",
    "load_megadetector",
    "megadetector_dir",
    "megadetector_labels",
    "letterbox",
    "load_predictor",
    "nms",
    "read_rgb",
]
