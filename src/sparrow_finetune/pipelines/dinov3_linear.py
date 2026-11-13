"""``dinov3_linear``: a linear classification head trained on frozen image-encoder features.

Ported from the Sparrow Studio private-head trainer: every annotation is cropped and embedded once
(``prepare``), an ``SGDClassifier`` (one-vs-rest logistic regression, L2, averaged weights) is
fitted with one ``partial_fit`` pass per epoch and the best epoch by validation macro-F1 is kept
(``train``), and the head is exported as ONNX (torch dynamo exporter) behind a parity gate against
scikit-learn (``export``).
State between stages lives in ``ctx.workdir``: one trainer instance serves every job.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import pickle
import zipfile
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image

from ..calibration import (
    THRESHOLD_PARAMS,
    json_metrics,
    threshold_for_params,
    threshold_manifest,
    threshold_metrics,
)
from ..embedders import Embedder, EmbeddingCache, embedder_fingerprint
from . import ExportResult, JobContext, Trainer, dynamo_onnx_bytes, parity_check
from .near_dup import DEFAULT_MAX_DISTANCE, dhash, near_duplicate_report
from .stock_model import (
    DEFAULT_STOCK_MIN_VAL_CROPS,
    is_bare_name,
    load_stock_model,
    matched_label_map,
    stock_report,
)

EMBED_BATCH = 16
ONNX_OPSET = 18
ECE_BINS = 10
PARITY_MAX_ABS_DELTA = 1e-4
# Fixed zip entry timestamp so identical runs produce byte-identical bundles.
ZIP_DATE_TIME = (1980, 1, 1, 0, 0, 0)

# name -> (kind, default, check, requirement shown in the error)
_PARAMS: dict[str, tuple[str, Any, Callable[[Any], bool], str]] = {
    "max_epochs": ("int", 50, lambda v: 1 <= v <= 500, "an integer in 1..500"),
    "alpha": ("float", 1e-4, lambda v: v > 0, "a number > 0"),
    "seed": ("int", 0, lambda v: 0 <= v < 2**32, "an integer in [0, 2**32)"),
    "val_fraction": ("float", 0.2, lambda v: 0 < v <= 0.5, "a number in (0, 0.5]"),
    "test_fraction": ("float", 0.0, lambda v: 0 <= v < 0.5, "a number in [0, 0.5)"),
    "standardize": ("bool", True, lambda v: True, "a boolean"),
    "class_floor": ("int", 2, lambda v: v >= 1, "an integer >= 1"),
    "feature_noise_std": ("float", 0.0, lambda v: v >= 0, "a number >= 0"),
    "augment_replicas": ("int", 0, lambda v: 0 <= v <= 10, "an integer in 0..10"),
    "reference_gate": ("bool", True, lambda v: True, "a boolean"),
    "reference_margin": ("float", 0.02, lambda v: 0 <= v <= 1, "a number in [0, 1]"),
    # FT-19: below this many validation crops the gate warns instead of failing the job.
    "reference_min_val_crops": ("int", 100, lambda v: v >= 0, "an integer >= 0"),
    "near_dup_max_distance": (
        "int",
        DEFAULT_MAX_DISTANCE,
        lambda v: 0 <= v <= 32,
        "an integer in 0..32",
    ),
    "stock_model": (
        "str",
        None,
        lambda v: v is None or is_bare_name(v),
        "a bare package name inside SPARROW_FT_WEIGHTS_DIR, or null",
    ),
    # FT-18: a worse-than-stock result blocks activation only with this many compared crops.
    "stock_min_val_crops": (
        "int",
        DEFAULT_STOCK_MIN_VAL_CROPS,
        lambda v: v >= 0,
        "an integer >= 0",
    ),
    # FT-24: calibrated confidence threshold (see ``calibration``) and the embedding cache.
    **THRESHOLD_PARAMS,
    "embedding_cache": ("bool", True, lambda v: True, "a boolean"),
}


class ReferenceGateError(RuntimeError):
    """The SGD head trails the full-batch logistic-regression reference by more than the margin (FT-4)."""


def _ovr_proba(estimator, x) -> np.ndarray:
    """sklearn's one-vs-rest probabilities, computed in float64 log space (FT-20).

    ``predict_proba`` keeps float32 input in float32, so ``expit`` returns 0 for every class once
    all decision values are below about -88 and sklearn substitutes a uniform row. The ratio
    ``sigmoid(z_i) / sum_j sigmoid(z_j)`` equals ``softmax(log_sigmoid(z))``, which the ONNX head
    also computes; binary heads give ``softmax([0, z])``.
    """
    z = np.asarray(
        estimator.decision_function(np.asarray(x, dtype=np.float64)), dtype=np.float64
    )
    if z.ndim == 1:
        z = np.stack([np.zeros_like(z), z], axis=1)
    else:
        z = np.minimum(z, 0.0) - np.log1p(np.exp(-np.abs(z)))
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _reference_macro_f1(
    xs_train, y_train, xs_val, y_val, classes, *, alpha: float, seed: int
) -> float:
    """Val macro-F1 of a full-batch ``LogisticRegression`` with the head's L2 strength (Studio's reference)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    ref = LogisticRegression(
        solver="lbfgs", max_iter=1000, C=1.0 / (alpha * len(y_train)), random_state=seed
    ).fit(xs_train, y_train)
    return float(
        f1_score(
            y_val, ref.predict(xs_val), labels=classes, average="macro", zero_division=0
        )
    )


def crop_box(bbox, width: int, height: int) -> tuple[int, int, int, int]:
    """Pixel ``[x, y, w, h]`` (or ``None`` for the whole image) -> clamped ``(left, top, right, bottom)``."""
    if bbox is None:
        return (0, 0, int(width), int(height))
    x, y, w, h = (float(v) for v in bbox)
    left = min(max(math.floor(x), 0), width)
    top = min(max(math.floor(y), 0), height)
    right = min(max(math.ceil(x + w), 0), width)
    bottom = min(max(math.ceil(y + h), 0), height)
    if right <= left or bottom <= top:
        raise ValueError(
            f"bbox {list(bbox)} has zero area inside a {width}x{height} image"
        )
    return (int(left), int(top), int(right), int(bottom))


def assign_split(
    key: str, *, seed: int, val_fraction: float, test_fraction: float = 0.0
) -> str:
    """Deterministic train/val/test assignment from a hash of ``seed`` and ``key``."""
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    unit = int.from_bytes(digest[:8], "big") / 2**64
    if unit < val_fraction:
        return "val"
    if unit < val_fraction + test_fraction:
        return "test"
    return "train"


def _split_key(item) -> str:
    return item.group if item.group else item.name


def source_crops(items, keep: Callable[[Any, dict], bool]) -> list[Image.Image]:
    """RGB crops of the annotations ``keep(item, annotation)`` selects, cut from the original images.

    In ``items`` and annotation order; both classifier pipelines score the stock model on these
    (FT-18), not on any cached or re-encoded copy.
    """
    crops: list[Image.Image] = []
    for item in items:
        wanted = [a for a in item.annotations if keep(item, a)]
        if not wanted:
            continue
        with Image.open(item.image_path) as im:
            im = im.convert("RGB")
            crops.extend(
                im.crop(crop_box(a.get("bbox"), im.width, im.height)) for a in wanted
            )
    return crops


def _expected_calibration_error(
    probs: np.ndarray, y: np.ndarray, bins: int = ECE_BINS
) -> float:
    if len(y) == 0:
        return 0.0
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def _export_head_onnx(model, dim: int) -> bytes:
    """The fitted scaler + SGD head as an ONNX graph ``features [N, dim] -> probabilities [N, C]``.

    A torch module reproduces ``_ovr_proba`` (sklearn's ``Pipeline.predict_proba`` without underflow): ``(x - mean) / scale``, the linear
    decision, then sklearn's one-vs-rest probability (binary: ``[1 - p, p]``; multiclass: sigmoids
    normalised per row, computed as a softmax of log-sigmoids so it does not underflow). Exported
    with the dynamo exporter; the parity gate in ``export`` checks it against scikit-learn.
    """
    import torch

    steps = dict(model.steps)
    clf = steps["clf"]
    scaler = steps.get("scaler")

    def buf(a) -> torch.Tensor:
        return torch.as_tensor(np.asarray(a, dtype=np.float32))

    class Head(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            n_features = clf.coef_.shape[1]
            mean = (
                scaler.mean_
                if scaler is not None and scaler.mean_ is not None
                else np.zeros(n_features)
            )
            scale = (
                scaler.scale_
                if scaler is not None and scaler.scale_ is not None
                else np.ones(n_features)
            )
            self.register_buffer("mean", buf(mean))
            self.register_buffer("scale", buf(scale))
            self.register_buffer("coef_t", buf(np.asarray(clf.coef_).T))
            self.register_buffer("intercept", buf(clf.intercept_))
            self.binary = clf.coef_.shape[0] == 1

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            z = ((features - self.mean) / self.scale) @ self.coef_t + self.intercept
            if self.binary:
                # softmax([0, z]) == [1 - sigmoid(z), sigmoid(z)] without underflow.
                return torch.softmax(torch.cat([torch.zeros_like(z), z], dim=1), dim=1)
            # sklearn's OvR ratio sigmoid(z_i) / sum_j sigmoid(z_j) equals
            # softmax(log_sigmoid(z)). onnxruntime's Sigmoid returns 0 below
            # about -18, so the ratio is computed in log space instead.
            log_sig = torch.minimum(z, torch.zeros_like(z)) - torch.log(
                1.0 + torch.exp(-torch.abs(z))
            )
            return torch.softmax(log_sig, dim=1)

    return dynamo_onnx_bytes(
        Head().eval(),
        torch.zeros(2, dim, dtype=torch.float32),
        input_name="features",
        output_name="probabilities",
        graph_name="dinov3_linear_head",
        opset=ONNX_OPSET,
    )


def _head_arrays(model) -> dict[str, np.ndarray]:
    """The fitted head as plain arrays: ``mean``/``scale`` of the scaler (0/1 without one), ``coef`` and ``intercept``."""
    steps = dict(model.steps)
    clf, scaler = steps["clf"], steps.get("scaler")
    coef = np.asarray(clf.coef_, dtype=np.float64)
    n = coef.shape[1]
    has = scaler is not None and scaler.mean_ is not None
    return {
        "mean": np.asarray(scaler.mean_ if has else np.zeros(n), dtype=np.float64),
        "scale": np.asarray(
            scaler.scale_ if scaler is not None and scaler.scale_ is not None else np.ones(n),
            dtype=np.float64,
        ),
        "coef": coef,
        "intercept": np.asarray(clf.intercept_, dtype=np.float64),
    }


def warm_start_head(
    base: dict[str, np.ndarray],
    base_labels: list[str],
    labels: list[str],
    mean: np.ndarray,
    scale: np.ndarray,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], list[str]]:
    """Initial ``(coef, intercept)`` for a head on ``labels`` whose inputs are ``(x - mean) / scale``.

    The base head computes ``z = ((x - m0) / s0) @ W0.T + b0``. Re-expressed for the new scaler it is
    ``W = W0 * scale / s0`` and ``b = b0 + ((mean - m0) / s0) @ W0.T``, so the starting head scores raw
    features exactly as the base did. Rows are matched by label name; a label the base did not have
    starts at zero. Binary heads keep one row (the second label's score): reused when the label pair
    matches, negated when it is swapped. Returns ``(None, None, [])`` when nothing can be reused.
    """
    w0, b0 = base["coef"], base["intercept"]
    m0, s0 = base["mean"], base["scale"]
    w = w0 * (scale / s0)[None, :]
    b = b0 + ((mean - m0) / s0) @ w0.T
    if len(base_labels) == 2 or len(labels) == 2:
        if len(base_labels) == len(labels) == 2:
            if list(base_labels) == list(labels):
                return w.reshape(1, -1), b.reshape(1), list(labels)
            if list(base_labels) == list(labels)[::-1]:
                return -w.reshape(1, -1), -b.reshape(1), list(labels)
        return None, None, []
    index = {label: i for i, label in enumerate(base_labels)}
    shared = [label for label in labels if label in index]
    if not shared:
        return None, None, []
    coef = np.zeros((len(labels), w.shape[1]))
    intercept = np.zeros(len(labels))
    for row, label in enumerate(labels):
        if label in index:
            coef[row], intercept[row] = w[index[label]], b[index[label]]
    return coef, intercept, shared


class DinoV3LinearTrainer(Trainer):
    id = "dinov3_linear"
    name = "DINOv3 linear probe"
    task = "classification"
    requires: frozenset[str] = frozenset()
    # FT-37: the head starts from an imported dinov3_linear model on the same embedder.
    supports_base_model = True

    def __init__(
        self,
        embedder_factory: Callable[[], Embedder],
        *,
        weights_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self._embedder_factory = embedder_factory
        # Root of the stock-model packages (``stock_model`` param, FT-16).
        self._weights_dir = weights_dir
        # Embedding cache root; None derives ``<data_dir>/cache/embeddings`` from the job workdir.
        self._cache_dir = cache_dir

    # ------------------------------------------------------------------ params

    def validate_params(self, params: Optional[dict]) -> dict:
        params = dict(params or {})
        base_model_id = params.pop("base_model_id", None)
        if base_model_id is not None and (
            not isinstance(base_model_id, int)
            or isinstance(base_model_id, bool)
            or base_model_id < 1
        ):
            raise ValueError(
                f"parameter base_model_id must be a positive integer (an imported base model id), "
                f"got {base_model_id!r}"
            )
        unknown = sorted(set(params) - set(_PARAMS))
        if unknown:
            raise ValueError(f"unknown parameter(s): {', '.join(unknown)}")
        out = {}
        for name, (kind, default, check, requirement) in _PARAMS.items():
            value = params.get(name, default)
            if kind == "bool":
                ok_type = isinstance(value, bool)
            elif kind == "int":
                ok_type = isinstance(value, int) and not isinstance(value, bool)
            elif kind == "str":
                ok_type = value is None or isinstance(value, str)
            elif kind == "optfloat" and value is None:
                ok_type = True
            else:
                ok_type = isinstance(value, (int, float)) and not isinstance(
                    value, bool
                )
                if ok_type:
                    try:
                        value = float(value)
                    except OverflowError:
                        ok_type = False
                    else:
                        ok_type = math.isfinite(value)
            if not ok_type or not check(value):
                raise ValueError(
                    f"parameter {name} must be {requirement}, got {value!r}"
                )
            out[name] = value
        if out["feature_noise_std"] > 0 and out["augment_replicas"] == 0:
            raise ValueError(
                "parameter feature_noise_std > 0 has no effect while augment_replicas is 0: "
                "set augment_replicas >= 1 or feature_noise_std to 0"
            )
        if base_model_id is not None:
            out["base_model_id"] = base_model_id
        return out

    # ------------------------------------------------------------------ prepare

    def prepare(self, ctx: JobContext) -> None:
        # Built before any crop is read, so a malformed SPARROW_FT_EMBEDDER or a missing embedder
        # model fails the job without wasted work. A worker whose SPARROW_FT_EMBEDDER is unset never
        # claims this pipeline (register_builtin_pipelines sets ``unavailable_reason``).
        embedder = self._embedder_factory()
        try:
            self._prepare(ctx, embedder)
        finally:
            embedder.close()

    def _prepare(self, ctx: JobContext, embedder: Embedder) -> None:
        params = ctx.params
        if ctx.base_model is not None:
            # Before the embedding pass: the base head must fit this worker's embedder.
            self._stage_base_head(ctx, embedder)
        if params.get("stock_model"):
            # Fail before the embedding pass when the stock package is unusable.
            load_stock_model(params["stock_model"], self._weights_dir)
        if len(ctx.labels) < 2:
            raise ValueError(
                f"dinov3_linear needs at least 2 labels, the snapshot has {len(ctx.labels)}"
                + (f" ({ctx.labels[0]!r})" if ctx.labels else "")
            )
        label_index = {label: i for i, label in enumerate(ctx.labels)}
        sides = (
            ("train", "val", "test")
            if params["test_fraction"] > 0
            else ("train", "val")
        )
        labels: list[int] = []
        splits: list[str] = []
        summary = {side: {"items": 0, "crops": 0, "groups": set()} for side in sides}
        for item in ctx.items:
            side = assign_split(
                _split_key(item),
                seed=params["seed"],
                val_fraction=params["val_fraction"],
                test_fraction=params["test_fraction"],
            )
            summary[side]["items"] += 1
            if item.group:
                summary[side]["groups"].add(item.group)
            for ann in item.annotations:
                labels.append(label_index[ann["label"]])
                splits.append(side)
                summary[side]["crops"] += 1

        labels_arr = np.asarray(labels, dtype=np.int64)
        splits_arr = np.asarray(splits, dtype=str)
        floor = params["class_floor"]
        for i, label in enumerate(ctx.labels):
            n_train = int(((labels_arr == i) & (splits_arr == "train")).sum())
            if n_train < floor:
                raise ValueError(
                    f"label {label!r} has {n_train} training crop(s); class_floor requires at least {floor}"
                )
        if not (splits_arr == "val").any():
            raise ValueError(
                "the validation split is empty: add more images or groups, or raise val_fraction"
            )
        if params["test_fraction"] > 0 and not (splits_arr == "test").any():
            raise ValueError(
                "the test split is empty: add more images or groups, or raise test_fraction"
            )

        cache = (
            self._open_cache(ctx, embedder)
            if params.get("embedding_cache", True)
            else None
        )
        features, hashes = self._extract(ctx, embedder, cache)
        if cache is not None and features is None:
            # Cached and freshly computed vectors disagree on the dimension (an embedder whose
            # output changed without a new id or model hash): recompute everything uncached.
            ctx.emit(
                "prepare",
                "embedding cache dimension mismatch: recomputing without the cache",
                fingerprint=cache.fingerprint,
            )
            stats = cache.stats()
            cache = None
            features, hashes = self._extract(ctx, embedder, None)
            cache_stats = dict(stats, enabled=False, reason="dimension mismatch")
        else:
            cache_stats = cache.stats() if cache is not None else {"enabled": False}
        n_crops = len(labels)
        if features.shape[0] != n_crops:
            raise RuntimeError(
                f"embedder returned {features.shape[0]} features for {n_crops} crops"
            )

        workdir = Path(ctx.workdir)
        np.save(workdir / "features.npy", features)
        np.save(workdir / "labels.npy", labels_arr)
        np.save(workdir / "splits.npy", splits_arr)
        np.save(workdir / "dhash.npy", np.asarray(hashes, dtype=np.uint64))
        dim = int(features.shape[1])
        info = {
            "embedder": {
                "id": embedder.id,
                "dim": dim,
                "model_hash": embedder.model_hash,
            },
            "input": {
                "size": getattr(embedder, "input_size", None),
                "mean": [float(v) for v in np.ravel(getattr(embedder, "mean", ()))],
                "std": [float(v) for v in np.ravel(getattr(embedder, "std", ()))],
            },
            "split": {
                side: {
                    "items": s["items"],
                    "crops": s["crops"],
                    "groups": sorted(s["groups"]),
                }
                for side, s in summary.items()
            },
            "embedding_cache": cache_stats,
        }
        (workdir / "prepare.json").write_text(json.dumps(info, indent=2))
        ctx.emit(
            "prepare",
            "embedding cache",
            **{
                k: v
                for k, v in cache_stats.items()
                if k
                in ("enabled", "hits", "misses", "corrupt", "writes", "write_errors")
            },
        )
        ctx.emit("prepare", "features extracted", crops=n_crops, dim=dim)

    @staticmethod
    def _stage_base_head(ctx: JobContext, embedder: Embedder) -> None:
        """Check the base bundle's ``weights.pt`` and embedder; copy it to ``workdir/base_head.pt``."""
        base = ctx.base_model
        with zipfile.ZipFile(Path(base["path"])) as zf:
            names = set(zf.namelist())
            manifest = (
                json.loads(zf.read("manifest.json")) if "manifest.json" in names else {}
            )
            if "weights.pt" not in names:
                raise ValueError(
                    f"base model {base['id']} has no head weights (weights.pt): it was exported by an "
                    "older version; re-train with this version to enable warm start"
                )
            data = zf.read("weights.pt")
        if (manifest.get("files") or {}).get("weights.pt") != hashlib.sha256(data).hexdigest():
            raise ValueError(
                f"base model {base['id']}: weights.pt does not match the sha256 in its manifest"
            )
        base_embedder = manifest.get("embedder") or {}
        if base_embedder.get("model_hash") != embedder.model_hash:
            raise ValueError(
                f"base model {base['id']} was trained on embedder {base_embedder.get('id')!r} "
                f"(model hash {base_embedder.get('model_hash')!r}), but this worker embeds with "
                f"{embedder.id!r} (model hash {embedder.model_hash!r}); its head cannot be reused"
            )
        workdir = Path(ctx.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / "base_head.pt").write_bytes(data)

    def _warm_start(self, ctx: JobContext, clf, mean, scale, classes, dtype) -> dict:
        """Seed ``clf`` with the base head (see ``warm_start_head``); return the card's warm-start block."""
        import torch

        base = ctx.base_model
        state = torch.load(
            Path(ctx.workdir) / "base_head.pt", map_location="cpu", weights_only=True
        )
        arrays = {k: state[k].double().numpy() for k in ("mean", "scale", "coef", "intercept")}
        if arrays["coef"].shape[1] != mean.shape[0]:
            raise ValueError(
                f"base model {base['id']} head takes {arrays['coef'].shape[1]} features, "
                f"the embedder returns {mean.shape[0]}"
            )
        coef, intercept, shared = warm_start_head(
            arrays, list(base["labels"]), list(ctx.labels), mean, scale
        )
        if coef is not None:
            # SGDClassifier.partial_fit has no coef_init: allocate the parameters it would
            # allocate on its first call, starting from the base head (sklearn 1.9 API).
            clf._allocate_parameter_mem(
                n_classes=len(classes),
                n_features=coef.shape[1],
                input_dtype=dtype,
                coef_init=coef.astype(dtype),
                intercept_init=intercept.astype(dtype),
            )
        block = {
            "base_model_id": base["id"],
            "base_labels": list(base["labels"]),
            "reused_labels": shared,
        }
        detail = (
            f"head initialised for {len(shared)} of {len(ctx.labels)} labels"
            if shared
            else f"head re-initialised (base labels {list(base['labels'])} share nothing reusable "
            f"with {list(ctx.labels)})"
        )
        ctx.emit(
            "train",
            f"warm start from base model {base['id']}: {detail}",
            base_model_id=base["id"],
            reused_labels=len(shared),
        )
        return block

    def _cache_root(self, ctx: JobContext) -> Optional[Path]:
        """``<data_dir>/cache/embeddings``, derived from ``<data_dir>/jobs/<id>/attempt-N/work``."""
        if self._cache_dir is not None:
            return Path(self._cache_dir)
        work = Path(ctx.workdir)
        parents = work.parents
        if (
            work.name == "work"
            and len(parents) >= 4
            and parents[0].name.startswith("attempt-")
            and parents[2].name == "jobs"
        ):
            return parents[3] / "cache" / "embeddings"
        return None

    def _open_cache(
        self, ctx: JobContext, embedder: Embedder
    ) -> Optional[EmbeddingCache]:
        root = self._cache_root(ctx)
        if root is None:
            ctx.emit(
                "prepare",
                "embedding cache disabled: no service data directory for this job",
            )
            return None
        return EmbeddingCache(root, embedder_fingerprint(embedder))

    @staticmethod
    def _extract(
        ctx: JobContext, embedder: Embedder, cache: Optional[EmbeddingCache]
    ) -> tuple[Optional[np.ndarray], list[int]]:
        """Crop embeddings in snapshot order and their dHashes.

        Each image is decoded once; its crops are shrunk by the embedder's preprocessing before
        they wait for a batch, so at most EMBED_BATCH small crops are held. With a cache, a crop
        whose (embedder fingerprint, image sha256, crop box) entry exists is not embedded; computed
        vectors are stored. Returns ``(None, hashes)`` when cached and computed vectors disagree on
        the dimension.
        """
        rows: list[Optional[np.ndarray]] = []
        hashes: list[int] = []
        pending: list[Image.Image] = []
        pending_rows: list[int] = []
        pending_keys: list[Optional[str]] = []
        dim_hint = getattr(embedder, "dim", None)
        served = 0

        def flush() -> None:
            out = np.asarray(embedder.embed(pending), dtype=np.float32)
            got = out.shape[0] if out.ndim >= 1 else 0
            if out.ndim != 2 or got != len(pending):
                raise RuntimeError(
                    f"embedder returned {got} features for {len(pending)} crops"
                )
            for row, key, vector in zip(pending_rows, pending_keys, out):
                rows[row] = vector
                if cache is not None and key is not None:
                    cache.put(key, vector)
            pending.clear()
            pending_rows.clear()
            pending_keys.clear()
            ctx.check_cancelled()

        for item in ctx.items:
            if not item.annotations:
                continue
            data = Path(item.image_path).read_bytes()
            image_sha = hashlib.sha256(data).hexdigest() if cache is not None else None
            with Image.open(io.BytesIO(data)) as im:
                im = im.convert("RGB")
                for ann in item.annotations:
                    box = crop_box(ann.get("bbox"), im.width, im.height)
                    crop = im.crop(box)
                    hashes.append(dhash(crop))
                    rows.append(None)
                    key = None
                    if cache is not None:
                        key = cache.key(image_sha, box)
                        hit = cache.get(key, dim_hint)
                        if hit is not None:
                            rows[-1] = hit
                            served += 1
                            if served % EMBED_BATCH == 0:
                                ctx.check_cancelled()
                            continue
                    pending.append(embedder.preprocess_crop(crop))
                    pending_rows.append(len(rows) - 1)
                    pending_keys.append(key)
                    if len(pending) >= EMBED_BATCH:
                        flush()
        if pending:
            flush()
        if not rows:
            return np.zeros((0, 0), dtype=np.float32), hashes
        if len({len(r) for r in rows}) != 1:
            if cache is not None:
                return None, hashes
            raise RuntimeError("embedder returned vectors of different dimensions")
        return np.stack(rows).astype(np.float32, copy=False), hashes

    # ------------------------------------------------------------------ train

    @staticmethod
    def _load_arrays(workdir: Path):
        features = np.load(workdir / "features.npy")
        labels = np.load(workdir / "labels.npy")
        splits = np.load(workdir / "splits.npy")
        train, val = splits == "train", splits == "val"
        return features[train], labels[train], features[val], labels[val]

    @staticmethod
    def _load_test(workdir: Path):
        splits = np.load(workdir / "splits.npy")
        test = splits == "test"
        return np.load(workdir / "features.npy")[test], np.load(workdir / "labels.npy")[
            test
        ]

    def train(self, ctx: JobContext) -> None:
        from sklearn.linear_model import SGDClassifier
        from sklearn.metrics import f1_score, log_loss
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        params = ctx.params
        workdir = Path(ctx.workdir)
        x_train, y_train, x_val, y_val = self._load_arrays(workdir)
        classes = np.arange(len(ctx.labels))
        rng = np.random.default_rng(params["seed"])

        # Train-only feature augmentation: noisy replicas of the clean training rows.
        x_fit, y_fit = [x_train], [y_train]
        for _ in range(params["augment_replicas"]):
            noise = rng.normal(0.0, params["feature_noise_std"], size=x_train.shape)
            x_fit.append((x_train + noise).astype(np.float32))
            y_fit.append(y_train)
        x_fit_arr, y_fit_arr = np.concatenate(x_fit), np.concatenate(y_fit)

        steps = []
        if params["standardize"]:
            steps.append(("scaler", StandardScaler().fit(x_train)))
        clf = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=params["alpha"],
            average=True,
            random_state=params["seed"],
        )
        steps.append(("clf", clf))
        model = Pipeline(steps)
        scale = (
            (lambda x: steps[0][1].transform(x))
            if params["standardize"]
            else (lambda x: x)
        )
        xs_fit, xs_train, xs_val = scale(x_fit_arr), scale(x_train), scale(x_val)
        warm_start = None
        if ctx.base_model is not None:
            dim = x_train.shape[1]
            scaler = steps[0][1] if params["standardize"] else None
            warm_start = self._warm_start(
                ctx,
                clf,
                np.asarray(scaler.mean_ if scaler else np.zeros(dim), dtype=np.float64),
                np.asarray(scaler.scale_ if scaler else np.ones(dim), dtype=np.float64),
                classes,
                xs_fit.dtype,
            )

        best_f1, best_epoch, best_clf = -1.0, 0, None
        curve = []
        for epoch in range(1, params["max_epochs"] + 1):
            ctx.check_cancelled()
            order = rng.permutation(len(y_fit_arr))
            clf.partial_fit(xs_fit[order], y_fit_arr[order], classes=classes)
            train_loss = float(
                log_loss(y_train, _ovr_proba(clf, xs_train), labels=classes)
            )
            val_probs = _ovr_proba(clf, xs_val)
            val_loss = float(log_loss(y_val, val_probs, labels=classes))
            val_f1 = float(
                f1_score(
                    y_val,
                    val_probs.argmax(axis=1),
                    labels=classes,
                    average="macro",
                    zero_division=0,
                )
            )
            point = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_f1": val_f1,
            }
            curve.append(point)
            ctx.emit("train", f"epoch {epoch}", **point)
            if val_f1 > best_f1:
                best_f1, best_epoch, best_clf = val_f1, epoch, copy.deepcopy(clf)

        reference = {"enabled": False}
        if params["reference_gate"]:
            ref_f1 = _reference_macro_f1(
                xs_train,
                y_train,
                xs_val,
                y_val,
                classes,
                alpha=params["alpha"],
                seed=params["seed"],
            )
            margin = params["reference_margin"]
            min_val = params["reference_min_val_crops"]
            n_val = int(len(y_val))
            reference = {
                "enabled": True,
                "passed": ref_f1 - best_f1 <= margin,
                "margin": margin,
                "reference_val_macro_f1": ref_f1,
                "head_val_macro_f1": best_f1,
                "val_crops": n_val,
                "min_val_crops": min_val,
                "enforced": n_val >= min_val,
            }
            ctx.emit("train", "reference gate", **reference)
            if not reference["passed"] and not reference["enforced"]:
                # FT-19: on a small validation set the margin is about one crop, so a trailing
                # head is reported, not refused (docs/decisions.md).
                reference["warning"] = (
                    f"the logistic-regression reference scored val macro-F1 {ref_f1:.4f}, "
                    f"the head {best_f1:.4f} (margin {margin:g}); with only {n_val} "
                    f"validation crops (< reference_min_val_crops={min_val}) the gate warns "
                    f"instead of failing"
                )
                ctx.emit(
                    "train",
                    "reference gate warning",
                    reference_val_macro_f1=ref_f1,
                    head_val_macro_f1=best_f1,
                    val_crops=n_val,
                    min_val_crops=min_val,
                )
            elif not reference["passed"]:
                raise ReferenceGateError(
                    f"reference-model gate failed: logistic-regression reference val macro-F1 "
                    f"{ref_f1:.4f} beats the head's {best_f1:.4f} by more than margin {margin:g}; "
                    f"try more max_epochs or a different alpha, or set reference_gate=false"
                )

        model.steps[-1] = ("clf", best_clf)
        with open(workdir / "model.pkl", "wb") as fh:
            pickle.dump(model, fh)
        (workdir / "train.json").write_text(
            json.dumps(
                {
                    "best_epoch": best_epoch,
                    "epochs": curve,
                    "reference": reference,
                    "warm_start": warm_start,
                },
                indent=2,
            )
        )

    def _load_model(self, workdir: Path):
        with open(workdir / "model.pkl", "rb") as fh:
            return pickle.load(fh)

    # ------------------------------------------------------------------ evaluate

    def evaluate(self, ctx: JobContext) -> dict[str, float]:
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
            log_loss,
            precision_recall_fscore_support,
        )

        workdir = Path(ctx.workdir)
        x_train, _y_train, x_val, y_val = self._load_arrays(workdir)
        model = self._load_model(workdir)
        trained = json.loads((workdir / "train.json").read_text())
        classes = np.arange(len(ctx.labels))

        probs = _ovr_proba(model, x_val)
        pred = probs.argmax(axis=1)
        metrics = {
            "val_macro_f1": float(
                f1_score(y_val, pred, labels=classes, average="macro", zero_division=0)
            ),
            "val_accuracy": float(accuracy_score(y_val, pred)),
            "val_log_loss": float(log_loss(y_val, probs, labels=classes)),
            "val_ece": _expected_calibration_error(probs, y_val),
            "best_epoch": float(trained["best_epoch"]),
            "train_crops": float(len(x_train)),
            "val_crops": float(len(x_val)),
        }
        if ctx.params.get("test_fraction", 0.0) > 0:
            x_test, y_test = self._load_test(workdir)
            test_pred = _ovr_proba(model, x_test).argmax(axis=1)
            metrics["test_macro_f1"] = float(
                f1_score(
                    y_test, test_pred, labels=classes, average="macro", zero_division=0
                )
            )
            metrics["test_accuracy"] = float(accuracy_score(y_test, test_pred))
            metrics["test_crops"] = float(len(x_test))
        reference = trained.get("reference", {"enabled": False})
        if reference["enabled"]:
            metrics["reference_val_macro_f1"] = float(
                reference["reference_val_macro_f1"]
            )
            metrics["reference_worse"] = 0.0 if reference["passed"] else 1.0
        threshold = self._threshold(ctx.params, probs, y_val)
        metrics.update(threshold_metrics(threshold))
        cache_stats = (
            json.loads((workdir / "prepare.json").read_text()).get("embedding_cache")
            if (workdir / "prepare.json").exists()
            else None
        )
        if cache_stats and "hits" in cache_stats:
            metrics["embedding_cache_hits"] = float(cache_stats["hits"])
            metrics["embedding_cache_misses"] = float(cache_stats["misses"])
        precision, recall, f1, support = precision_recall_fscore_support(
            y_val, pred, labels=classes, zero_division=0
        )
        per_class = {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(ctx.labels)
        }
        report = {
            "metrics": metrics,
            "per_class": per_class,
            "confusion_matrix": confusion_matrix(y_val, pred, labels=classes).tolist(),
            "threshold": threshold,
        }
        splits = np.load(workdir / "splits.npy")
        report["near_duplicates"] = near_duplicate_report(
            ctx,
            np.load(workdir / "dhash.npy"),
            splits == "train",
            splits == "val",
            metrics,
        )
        if ctx.params.get("stock_model"):
            report["stock_model"] = self._stock_comparison(ctx, y_val, pred, metrics)
        (workdir / "evaluate.json").write_text(
            json.dumps(dict(report, metrics=json_metrics(metrics)), indent=2)
        )
        return metrics

    @staticmethod
    def _threshold(params: dict, probs, y_val) -> dict:
        return threshold_for_params(params, probs, y_val)

    def _stock_comparison(self, ctx: JobContext, y_val, pred, metrics: dict) -> dict:
        """Score the stock model on the val crops whose label it knows (FT-16)."""
        params = ctx.params
        stock = load_stock_model(params["stock_model"], self._weights_dir)
        mapping = matched_label_map(ctx.labels, stock.labels)
        label_index = {label: i for i, label in enumerate(ctx.labels)}
        matched = np.isin(y_val, list(mapping))

        def keep(item, ann) -> bool:
            side = assign_split(
                _split_key(item),
                seed=params["seed"],
                val_fraction=params["val_fraction"],
                test_fraction=params["test_fraction"],
            )
            return side == "val" and label_index[ann["label"]] in mapping

        crops = source_crops(ctx.items, keep) if matched.any() else []
        return stock_report(
            ctx, stock, mapping, y_val[matched], pred[matched], crops, metrics
        )

    # ------------------------------------------------------------------ export

    @staticmethod
    def _head_weights(model) -> bytes:
        """``weights.pt``: the head as a ``state_dict`` of float64 tensors (base-model import, FT-37)."""
        import torch

        buf = io.BytesIO()
        torch.save(
            {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in _head_arrays(model).items()},
            buf,
        )
        return buf.getvalue()

    def export(self, ctx: JobContext) -> ExportResult:
        import onnxruntime as ort

        workdir = Path(ctx.workdir)
        _x_train, _y_train, x_val, _y_val = self._load_arrays(workdir)
        model = self._load_model(workdir)
        prepared = json.loads((workdir / "prepare.json").read_text())
        trained = json.loads((workdir / "train.json").read_text())
        report = json.loads((workdir / "evaluate.json").read_text())
        dim = int(prepared["embedder"]["dim"])

        head_bytes = _export_head_onnx(model, dim)
        weights_bytes = self._head_weights(model)

        session = ort.InferenceSession(head_bytes, providers=["CPUExecutionProvider"])
        outputs = [o.name for o in session.get_outputs()]
        prob_name = "probabilities" if "probabilities" in outputs else outputs[-1]
        x_val32 = np.asarray(x_val, dtype=np.float32)
        onnx_probs = np.asarray(
            session.run([prob_name], {session.get_inputs()[0].name: x_val32})[0]
        )
        sk_probs = _ovr_proba(model, x_val32)
        parity = parity_check(
            onnx_probs, sk_probs, max_abs_delta_limit=PARITY_MAX_ABS_DELTA
        )

        metrics = report["metrics"]
        card = {
            "pipeline": self.id,
            "labels": list(ctx.labels),
            "params": dict(ctx.params),
            "metrics": metrics,
            "per_class": report["per_class"],
            "confusion_matrix": report["confusion_matrix"],
            "split": prepared["split"],
            "epochs": trained["epochs"],
            "embedder": prepared["embedder"],
            "reference_gate": trained.get("reference", {"enabled": False}),
            "near_duplicates": report.get("near_duplicates"),
            "threshold": report.get("threshold"),
        }
        if trained.get("warm_start"):
            card["warm_start"] = trained["warm_start"]
        stock = report.get("stock_model")
        if stock:
            card["stock_model"] = stock
        card_bytes = json.dumps(card, indent=2).encode("utf-8")
        files = {
            "head.onnx": hashlib.sha256(head_bytes).hexdigest(),
            "model_card.json": hashlib.sha256(card_bytes).hexdigest(),
            "weights.pt": hashlib.sha256(weights_bytes).hexdigest(),
        }
        manifest = {
            "pipeline": self.id,
            "labels": list(ctx.labels),
            # Base-model import (FT-37) names the network by model_arch: here the embedder.
            "model_arch": prepared["embedder"]["id"],
            "embedder": prepared["embedder"],
            "input": prepared["input"],
            "metrics": metrics,
            "parity": parity,
            "head": {"input": session.get_inputs()[0].name, "probabilities": prob_name},
            "files": files,
            **threshold_manifest(report.get("threshold")),
        }

        path = workdir / "dinov3_linear_bundle.zip"
        entries = {
            "head.onnx": head_bytes,
            "model_card.json": card_bytes,
            "weights.pt": weights_bytes,
            "manifest.json": json.dumps(manifest, indent=2).encode("utf-8"),
        }
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name, data in entries.items():
                info = zipfile.ZipInfo(name, date_time=ZIP_DATE_TIME)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                zf.writestr(info, data)
        ctx.emit("export", "bundle written", **parity)
        metadata = {
            "labels": list(ctx.labels),
            "embedder": prepared["embedder"],
            **threshold_manifest(report.get("threshold")),
        }
        if stock:
            metadata["stock_model"] = stock["id"]
        if ctx.params.get("base_model_id") is not None:
            metadata["base_model_id"] = ctx.params["base_model_id"]
        return ExportResult(path=path, format="onnx-bundle", metadata=metadata)
