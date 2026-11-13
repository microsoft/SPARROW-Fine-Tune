"""Score a sparrow-engine stock classifier package on validation crops (FT-16).

A package directory holds ``manifest.toml``, the ONNX model and a labels file. Only the manifest
values the classifier comparison needs are supported; anything else is rejected with a ValueError
rather than silently mis-preprocessed. ``[inference]`` sections are engine runtime settings and are
ignored here (onnxruntime CPU is used).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from PIL import Image

PREDICT_BATCH = 32
# Activation is blocked on ``stock_worse`` only with at least this many compared val crops.
DEFAULT_STOCK_MIN_VAL_CROPS = 100
_IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

_SUPPORTED = {
    ("model", "format"): ("onnx",),
    ("preprocessing", "method"): ("resize",),
    ("preprocessing", "layout"): ("nchw",),
    ("preprocessing", "normalization"): ("unit", "imagenet", "none"),
    ("preprocessing", "channel_order"): ("rgb", "bgr"),
    ("postprocessing", "method"): ("softmax", "sigmoid"),
    ("labels", "format"): ("name_index_csv",),
}
_DEFAULTS = {("preprocessing", "channel_order"): "rgb"}


def is_bare_name(name: Any) -> bool:
    """A single path component: no separators, not ``.``/``..``, not empty or absolute."""
    return (
        isinstance(name, str)
        and bool(name)
        and name not in (".", "..")
        and "\\" not in name
        and not Path(name).is_absolute()
        and Path(name).name == name
    )


def resolve_stock_package(name: str, weights_dir: Optional[Path]) -> Path:
    """``<weights_dir>/<name>``: a package directory containing ``manifest.toml``."""
    if not is_bare_name(name):
        raise ValueError(
            f"stock_model must be a bare package name inside weights_dir, got {name!r}"
        )
    if weights_dir is None:
        raise ValueError(
            "stock_model needs a weights directory: set SPARROW_FT_WEIGHTS_DIR on the worker"
        )
    root = Path(weights_dir).resolve()
    pkg = (root / name).resolve()
    if not pkg.is_relative_to(root):
        raise ValueError(
            f"stock_model {name!r} resolves outside weights_dir {str(root)!r}"
        )
    if not pkg.is_dir():
        raise ValueError(f"stock model package {name!r} not found in {str(root)!r}")
    if not (pkg / "manifest.toml").is_file():
        raise ValueError(f"stock model package {str(pkg)!r} has no manifest.toml")
    return pkg


def _value(manifest: dict, section: str, key: str) -> str:
    raw = manifest.get(section, {}).get(key, _DEFAULTS.get((section, key)))
    allowed = _SUPPORTED[(section, key)]
    if raw not in allowed:
        raise ValueError(
            f"unsupported stock model manifest value [{section}].{key} = {raw!r}; "
            f"supported: {', '.join(allowed)}"
        )
    return raw


def _package_file(pkg: Path, rel: Any, what: str) -> Path:
    if not isinstance(rel, str) or not rel:
        raise ValueError(f"stock model manifest is missing the {what} file")
    path = (pkg / rel).resolve()
    if Path(rel).is_absolute() or not path.is_relative_to(pkg.resolve()):
        raise ValueError(
            f"stock model {what} file {rel!r} would escape the package directory"
        )
    if not path.is_file():
        raise ValueError(f"stock model {what} file {str(path)!r} not found")
    return path


def _read_labels(path: Path) -> list[str]:
    by_index: dict[int, str] = {}
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        name, sep, idx = line.rpartition(",")
        try:
            index = int(idx.strip())
        except ValueError:
            index = -1
        if not sep or not name.strip() or index < 0 or index in by_index:
            raise ValueError(
                f"{path}:{n}: expected a unique 'Name,index' line, got {line!r}"
            )
        by_index[index] = name.strip()
    if not by_index or sorted(by_index) != list(range(len(by_index))):
        raise ValueError(f"{path}: label indices must be 0..N-1")
    return [by_index[i] for i in range(len(by_index))]


def _read_package(pkg: Path) -> dict:
    """Validate a package's manifest, model file and labels without loading the model."""
    with open(pkg / "manifest.toml", "rb") as fh:
        manifest = tomllib.load(fh)
    model = manifest.get("model", {})
    _value(manifest, "model", "format")
    _value(manifest, "preprocessing", "method")
    _value(manifest, "preprocessing", "layout")
    normalization = _value(manifest, "preprocessing", "normalization")
    channel_order = _value(manifest, "preprocessing", "channel_order")
    postprocessing = _value(manifest, "postprocessing", "method")
    _value(manifest, "labels", "format")
    size = manifest.get("preprocessing", {}).get("input_size")
    if (
        not isinstance(size, list)
        or len(size) != 2
        or not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in size)
    ):
        raise ValueError(
            f"stock model [preprocessing].input_size must be [width, height], got {size!r}"
        )
    return {
        "id": str(model.get("id") or pkg.name),
        "model_path": _package_file(pkg, model.get("file"), "model"),
        "labels": _read_labels(
            _package_file(pkg, manifest.get("labels", {}).get("file"), "labels")
        ),
        "input_size": (size[0], size[1]),
        "normalization": normalization,
        "channel_order": channel_order,
        "postprocessing": postprocessing,
    }


def list_stock_packages(weights_dir: Optional[Path]) -> list[dict]:
    """Classifier packages under ``weights_dir`` a job can name as ``stock_model``.

    Directories without a supported classifier manifest (detectors, assist models) are skipped."""
    if weights_dir is None or not Path(weights_dir).is_dir():
        return []
    out = []
    for pkg in sorted(Path(weights_dir).iterdir()):
        if not pkg.is_dir() or not (pkg / "manifest.toml").is_file():
            continue
        try:
            info = _read_package(pkg)
        except (ValueError, OSError, tomllib.TOMLDecodeError):
            continue
        out.append({"id": pkg.name, "name": info["id"], "classes": len(info["labels"])})
    return out


@dataclass
class StockModel:
    id: str
    labels: list[str]
    input_size: tuple[int, int]  # width, height
    normalization: str
    channel_order: str
    postprocessing: str
    session: Any

    @classmethod
    def load(cls, pkg: Path) -> "StockModel":
        info = _read_package(Path(pkg))

        import onnxruntime as ort

        session = ort.InferenceSession(
            str(info["model_path"]), providers=["CPUExecutionProvider"]
        )
        return cls(
            id=info["id"],
            labels=info["labels"],
            input_size=info["input_size"],
            normalization=info["normalization"],
            channel_order=info["channel_order"],
            postprocessing=info["postprocessing"],
            session=session,
        )

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        arr = np.asarray(
            image.convert("RGB").resize(self.input_size, Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
        if self.normalization in ("unit", "imagenet"):
            arr = arr / 255.0
        if self.normalization == "imagenet":
            arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
        if self.channel_order == "bgr":
            arr = arr[..., ::-1]
        return arr.transpose(2, 0, 1)

    def predict_proba(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Class scores ``[N, len(labels)]`` after the manifest's postprocessing."""
        n = len(self.labels)
        if not images:
            return np.zeros((0, n), dtype=np.float64)
        name = self.session.get_inputs()[0].name
        out = []
        for start in range(0, len(images), PREDICT_BATCH):
            batch = np.stack(
                [self._preprocess(im) for im in images[start : start + PREDICT_BATCH]]
            )
            raw = np.asarray(
                self.session.run(None, {name: batch.astype(np.float32)})[0],
                dtype=np.float64,
            ).reshape(len(batch), -1)
            if raw.shape[1] != n:
                raise ValueError(
                    f"stock model {self.id!r} outputs {raw.shape[1]} classes but lists {n} labels"
                )
            out.append(raw)
        logits = np.concatenate(out)
        if self.postprocessing == "softmax":
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)
        return 1.0 / (1.0 + np.exp(-logits))


def load_stock_model(name: str, weights_dir: Optional[Path]) -> "StockModel":
    return StockModel.load(resolve_stock_package(name, weights_dir))


def stock_report(
    ctx: Any,
    stock: "StockModel",
    mapping: dict[int, int],
    y_true: np.ndarray,
    finetuned_pred: np.ndarray,
    crops: Sequence[Image.Image],
    metrics: dict,
) -> dict:
    """Add the stock comparison to ``metrics`` and return the model-card block.

    ``y_true``/``finetuned_pred`` are the val crops whose label the stock model knows, in the same
    order as ``crops``. With no such crop only ``stock_compared_crops = 0`` is reported, with a warning.
    """
    if len(crops) != len(y_true):
        raise RuntimeError(
            f"stock comparison: {len(crops)} crops for {len(y_true)} labels"
        )
    min_val = int(ctx.params.get("stock_min_val_crops", DEFAULT_STOCK_MIN_VAL_CROPS))
    enforced = len(y_true) >= min_val
    metrics["stock_min_val_crops"] = float(min_val)
    card = {
        "name": ctx.params.get("stock_model"),
        "id": stock.id,
        "matched_labels": sorted(ctx.labels[i] for i in mapping),
        "min_val_crops": min_val,
        "enforced": enforced,
    }
    if not len(y_true):
        metrics["stock_compared_crops"] = 0.0
        ctx.emit(
            "evaluate",
            f"warning: no val crop has a label the stock model {stock.id!r} knows; "
            "stock comparison skipped",
        )
        card["compared_crops"] = 0
        return card
    result = comparison_metrics(
        y_true, finetuned_pred, stock.predict_proba(crops), mapping
    )
    metrics.update(result)
    if result["stock_worse"]:
        ctx.emit(
            "evaluate",
            f"warning: the fine-tuned model's val macro F1 "
            f"{result['stock_finetuned_val_macro_f1']:.4f} is below stock model {stock.id!r}'s "
            f"{result['stock_val_macro_f1']:.4f} on {len(y_true)} shared-label crop(s)"
            + (
                ""
                if enforced
                else f"; activation is not blocked below stock_min_val_crops={min_val}"
            ),
        )
    card.update(result)
    card["compared_crops"] = len(y_true)
    return card


def matched_label_map(
    labels: Sequence[str], stock_labels: Sequence[str]
) -> dict[int, int]:
    """``{fine-tuned label index: stock label index}`` for labels equal ignoring case."""
    stock = {}
    for j, name in enumerate(stock_labels):
        stock.setdefault(name.casefold(), j)
    return {
        i: stock[lab.casefold()]
        for i, lab in enumerate(labels)
        if lab.casefold() in stock
    }


def comparison_metrics(
    y_true: np.ndarray,
    finetuned_pred: np.ndarray,
    stock_probs: np.ndarray,
    mapping: dict[int, int],
) -> dict[str, float]:
    """Macro F1 of the stock and the fine-tuned model on the same crops (all labelled with a matched class).

    The stock argmax is over all stock classes; a winning stock class the job does not train maps
    to -1, which is outside the scored labels and so counts as wrong.
    """
    from sklearn.metrics import f1_score

    ft_idx = sorted(mapping)
    stock_probs = np.asarray(stock_probs)
    to_ft = np.full(stock_probs.shape[1], -1, dtype=np.int64)
    for i in ft_idx:
        to_ft[mapping[i]] = i
    stock_pred = to_ft[stock_probs.argmax(axis=1)]
    stock_f1 = float(
        f1_score(y_true, stock_pred, labels=ft_idx, average="macro", zero_division=0)
    )
    ft_f1 = float(
        f1_score(
            y_true, finetuned_pred, labels=ft_idx, average="macro", zero_division=0
        )
    )
    return {
        "stock_val_macro_f1": stock_f1,
        "stock_finetuned_val_macro_f1": ft_f1,
        "stock_compared_crops": float(len(y_true)),
        "stock_worse": float(ft_f1 < stock_f1),
    }
