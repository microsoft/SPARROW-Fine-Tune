"""``conventional_classifier``: a timm image classifier fine-tuned end to end on annotation crops.

Every annotation is cropped once (``prepare``); the classifier head is warmed up on a frozen backbone, then
the whole network is fine-tuned with AdamW (backbone at a lower learning rate, cosine decay) and the best
epoch by validation macro-F1 is kept (``train``); the model is exported with the dynamo ONNX exporter as one
graph that takes the RGB crop and returns class probabilities, behind a parity gate against PyTorch (``export``).
State between stages lives in ``ctx.workdir``: one trainer instance serves every job.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import zipfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import numpy as np
from PIL import Image

from ..calibration import (
    THRESHOLD_PARAMS,
    json_metrics,
    threshold_for_params,
    threshold_manifest,
    threshold_metrics,
)
from . import ExportResult, JobContext, Trainer, dynamo_onnx_bytes, parity_check
from .dinov3_linear import (
    _expected_calibration_error,
    _split_key,
    assign_split,
    crop_box,
    source_crops,
)
from .near_dup import DEFAULT_MAX_DISTANCE, dhash, near_duplicate_report
from .yolo_detector import _hardware_message
from .stock_model import (
    DEFAULT_STOCK_MIN_VAL_CROPS,
    is_bare_name,
    load_stock_model,
    matched_label_map,
    stock_report,
)

PARITY_MAX_ABS_DELTA = 1e-4
# The torch.export-based (dynamo) ONNX exporter targets opset 18.
ONNX_OPSET = 18
EVAL_BATCH = 64
PREPARE_CHECK_EVERY = 64
# Cached crops are capped at this multiple of imgsz on their longer side: random-resized-crop keeps at least
# 80% of the area, so 2x leaves headroom over the training resolution without caching full-resolution pixels.
CROP_CACHE_FACTOR = 2
CROP_JPEG_QUALITY = 95
_DEVICE_RE = re.compile(r"auto|cpu|cuda|cuda:[0-9]+|[0-9]+")
# A timm architecture name with an optional pretrained tag (resnet50, resnet50.a1_in1k). Paths, URLs and
# ``hf-hub:`` ids are refused: they would load caller-chosen files or repositories.
_MODEL_RE = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)?")
_RRC_SCALE = (0.8, 1.0)
_RRC_RATIO = (3 / 4, 4 / 3)


def _is_timm_model(name: str) -> bool:
    if _MODEL_RE.fullmatch(name) is None:
        return False
    import timm

    if not timm.is_model(name):
        return False
    return "." not in name or name in timm.list_pretrained(name)


def _fixed_input_size(name: str) -> Optional[tuple[int, int]]:
    """``(height, width)`` a timm model requires, or ``None`` when it accepts any input size.

    Reads the registry's pretrained config only: no weights are downloaded and no model is built.
    """
    import timm

    cfg = timm.models.get_pretrained_cfg(name)
    if cfg is None or not getattr(cfg, "fixed_input_size", False):
        return None
    _channels, height, width = cfg.input_size
    return int(height), int(width)


# FT-36 batch="auto". Any timm model is accepted, so no per-backbone table: a probe runs two fine-tune steps
# of a random-init copy at batch 8 and 32 and fits peak reserved memory = fixed + per_image x batch. On the
# RTX 6000 Ada the 8->32 slope matched the 32->128 slope within -2..+14 % (resnet50, convnext_tiny/base,
# efficientnet_b0, vit_base at 224 px, resnet50 at 448 px); smaller probe batches were too noisy. For small
# activations (resnet18 at 128-224 px) the line under-predicted batch 128 by 10-48 %, so the chosen batch is
# then run once and shrunk until its measured peak fits (_verify_batch).
_PROBE_BATCHES = (8, 32)
_PROBE_FALLBACK_BATCHES = (2, 8)  # when batch 32 does not fit
_AUTO_HEADROOM = 0.85
_MAX_AUTO_BATCH = 128
# A batch larger than this share of the training crops leaves too few optimiser steps per epoch.
_MIN_STEPS_PER_EPOCH = 8
_VERIFY_ATTEMPTS = 3
CPU_AUTO_BATCH = 32


def auto_batch(free_mb: float, fixed_mb: float, per_image_mb: float, n_train: int) -> int:
    """Largest batch whose fitted peak stays within 85 % of free memory, capped at 128 and at
    ``n_train / 8`` (at least 8), rounded down to a multiple of 8 when >= 8."""
    fits = int((free_mb * _AUTO_HEADROOM - fixed_mb) // max(1.0, per_image_mb))
    cap = min(_MAX_AUTO_BATCH, max(8, n_train // _MIN_STEPS_PER_EPOCH))
    chosen = max(1, min(cap, fits))
    return chosen - chosen % 8 if chosen >= 8 else chosen


def _amp_enabled(params: dict, device) -> bool:
    return bool(params["amp"] and device.type == "cuda")


class _EpochSampler:
    """Yields ``(epoch, index)`` keys in an order seeded by ``(seed, epoch)``.

    The loader is built once (its workers persist across epochs), so the epoch travels with every key:
    a worker's copy of the dataset never needs the main process to update it.
    """

    def __init__(self, n: int, seed: int) -> None:
        self.n, self.seed, self.epoch = n, seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        order = np.random.default_rng([self.seed, self.epoch]).permutation(self.n)
        return iter([(self.epoch, int(i)) for i in order])


# name -> (kind, default, check, requirement shown in the error)
_PARAMS: dict[str, tuple[str, Any, Callable[[Any], bool], str]] = {
    "model": (
        "str",
        "resnet50",
        _is_timm_model,
        "a timm model name such as resnet50 or convnext_tiny.fb_in22k",
    ),
    "pretrained": ("bool", True, lambda v: True, "a boolean"),
    "epochs": ("int", 30, lambda v: 1 <= v <= 1000, "an integer in 1..1000"),
    "warmup_epochs": (
        "int",
        2,
        lambda v: 0 <= v <= 1000,
        "an integer in 0..1000 and <= epochs",
    ),
    "batch": (
        "batch",
        32,
        lambda v: v == "auto" or 1 <= v <= 1024,
        'an integer in 1..1024, or "auto" (or -1) to size it from the free GPU memory',
    ),
    "lr": ("float", 1e-3, lambda v: v > 0, "a finite number > 0"),
    "backbone_lr_factor": ("float", 0.1, lambda v: 0 <= v <= 1, "a number in [0, 1]"),
    "weight_decay": ("float", 0.05, lambda v: v >= 0, "a finite number >= 0"),
    "label_smoothing": ("float", 0.1, lambda v: 0 <= v < 1, "a number in [0, 1)"),
    "imgsz": ("int", 224, lambda v: 32 <= v <= 1024, "an integer in 32..1024"),
    "seed": ("int", 0, lambda v: 0 <= v < 2**32, "an integer in [0, 2**32)"),
    "val_fraction": ("float", 0.2, lambda v: 0 < v <= 0.5, "a number in (0, 0.5]"),
    "device": (
        "str",
        "auto",
        lambda v: _DEVICE_RE.fullmatch(v) is not None,
        'one of "auto", "cpu", "cuda", "cuda:N" or "N"',
    ),
    "amp": ("bool", True, lambda v: True, "a boolean"),
    "patience": (
        "int",
        5,
        lambda v: 0 <= v <= 1000,
        "an integer in 0..1000 (0 disables early stopping)",
    ),
    "workers": ("int", 0, lambda v: 0 <= v <= 32, "an integer in 0..32"),
    "class_floor": ("int", 2, lambda v: v >= 1, "an integer >= 1"),
    "augment": ("bool", True, lambda v: True, "a boolean"),
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
    # Warm start: id of a row in base_models (a bundle this service pushed to Studio and imported back).
    "base_model_id": (
        "int",
        None,
        lambda v: v is None or v >= 1,
        "a base model id (an integer >= 1), or null",
    ),
    # Reuse a finished model of this service (a model_versions id) as the start of a new version.
    "parent_model_id": (
        "int",
        None,
        lambda v: v is None or v >= 1,
        "a finished model id (an integer >= 1), or null",
    ),
    # FT-24: calibrated confidence threshold with an abstain outcome (see ``calibration``).
    **THRESHOLD_PARAMS,
}
_OFFLINE_VALUES = {"1", "true", "yes", "on"}
_TIMM_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


def _hf_offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in _OFFLINE_VALUES


def _resolve_device(device: str):
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.isdigit():
        return torch.device(f"cuda:{device}")
    return torch.device(device)


@contextmanager
def _private_rng(device) -> Iterator[None]:
    """Run with torch's global RNG forked so a job never changes the worker's RNG state."""
    import torch

    # manual_seed reseeds every CUDA device; fork them all, but never initialise CUDA for a CPU job.
    use_cuda = device.type == "cuda" or torch.cuda.is_initialized()
    devices = list(range(torch.cuda.device_count())) if use_cuda else []
    with torch.random.fork_rng(devices=devices):
        yield


def _to_tensor(im: Image.Image):
    import torch

    arr = np.asarray(im.convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.0
    return torch.from_numpy(np.ascontiguousarray(arr))


def _cap_long_side(im: Image.Image, max_side: int) -> Image.Image:
    longest = max(im.size)
    if longest <= max_side:
        return im
    scale = max_side / longest
    size = (max(1, round(im.width * scale)), max(1, round(im.height * scale)))
    return im.resize(size, Image.LANCZOS)


def _random_resized_crop(im: Image.Image, rng: np.random.Generator) -> Image.Image:
    width, height = im.size
    area = width * height * rng.uniform(*_RRC_SCALE)
    ratio = math.exp(rng.uniform(math.log(_RRC_RATIO[0]), math.log(_RRC_RATIO[1])))
    w = min(max(int(round(math.sqrt(area * ratio))), 1), width)
    h = min(max(int(round(math.sqrt(area / ratio))), 1), height)
    x0 = int(rng.integers(0, width - w + 1))
    y0 = int(rng.integers(0, height - h + 1))
    return im.crop((x0, y0, x0 + w, y0 + h))


class _CropDataset:
    """Crops on disk -> ``(CHW float tensor in [0, 1], label)``, stretched to ``imgsz``.

    Augmentation draws from a generator keyed on ``(seed, epoch, index)``, so a run is reproducible whatever
    the number of loader workers.
    """

    def __init__(
        self,
        files: list[Path],
        labels: list[int],
        imgsz: int,
        *,
        augment: bool,
        seed: int,
    ) -> None:
        self.files, self.labels, self.imgsz = files, labels, imgsz
        self.augment, self.seed, self.epoch = augment, seed, 0

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, key):
        epoch, i = key if isinstance(key, tuple) else (self.epoch, key)
        with Image.open(self.files[i]) as im:
            im = im.convert("RGB")
            if self.augment:
                rng = np.random.default_rng([self.seed, epoch, i])
                im = _random_resized_crop(im, rng)
                if rng.random() < 0.5:
                    im = im.transpose(Image.FLIP_LEFT_RIGHT)
            im = im.resize((self.imgsz, self.imgsz), Image.BILINEAR)
            return _to_tensor(im), self.labels[i]


def _normalized(model, mean, std):
    import torch

    class Normalized(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = model
            self.register_buffer(
                "mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
            )
            self.register_buffer(
                "std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
            )

        def forward(self, x):
            return self.model((x - self.mean) / self.std)

    return Normalized()


def _with_softmax(module):
    import torch

    class Probabilities(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner = module

        def forward(self, images):
            return torch.softmax(self.inner(images), dim=1)

    return Probabilities()


class ConventionalClassifierTrainer(Trainer):
    id = "conventional_classifier"
    name = "Conventional classifier (timm fine-tune)"
    task = "classification"
    requires: frozenset[str] = frozenset()
    supports_base_model = True

    def __init__(self, *, weights_dir: Optional[Path] = None) -> None:
        # Offline timm weights live in <weights_dir>/timm/<model>/ (FT-13); stock-model packages
        # (``stock_model`` param, FT-16) live directly under it.
        self._weights_dir = Path(weights_dir) if weights_dir is not None else None

    def _local_weight_candidates(self, model: str) -> list[Path]:
        if self._weights_dir is None:
            return []
        return [self._weights_dir / "timm" / model / f for f in _TIMM_WEIGHT_FILES]

    def _local_weights(self, model: str) -> Optional[Path]:
        return next(
            (p for p in self._local_weight_candidates(model) if p.is_file()), None
        )

    @staticmethod
    def _hf_cache_weights(model: str) -> tuple[Optional[Path], str]:
        """Pretrained weights for ``model`` already in the Hugging Face cache, and where they were looked for.

        Reads the cache only (``try_to_load_from_cache``); never contacts the Hub.
        """
        import timm
        from huggingface_hub import constants, try_to_load_from_cache

        cfg = timm.models.get_pretrained_cfg(model)
        hf_id = getattr(cfg, "hf_hub_id", None) if cfg is not None else None
        if not hf_id:
            return None, f"(no Hugging Face Hub id for {model!r})"
        repo, _, revision = hf_id.partition("@")
        names = [getattr(cfg, "hf_hub_filename", None), *_TIMM_WEIGHT_FILES]
        for name in dict.fromkeys(n for n in names if n):
            hit = try_to_load_from_cache(repo, name, revision=revision or None)
            if isinstance(hit, str):
                return Path(hit), ""
        where = Path(constants.HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}"
        return None, f"the Hugging Face cache ({where})"

    # ------------------------------------------------------------------ params

    def validate_params(self, params: Optional[dict]) -> dict:
        params = dict(params or {})
        unknown = sorted(set(params) - set(_PARAMS))
        if unknown:
            raise ValueError(f"unknown parameter(s): {', '.join(unknown)}")
        out = {}
        for name, (kind, default, check, requirement) in _PARAMS.items():
            value = params.get(name, default)
            if kind == "batch":
                if value == "auto" or (type(value) is int and value == -1):
                    value = "auto"
                ok_type = value == "auto" or (
                    isinstance(value, int) and not isinstance(value, bool)
                )
            elif kind == "str":
                ok_type = isinstance(value, str) or (value is None and default is None)
            elif kind == "bool":
                ok_type = isinstance(value, bool)
            elif kind == "int":
                ok_type = (isinstance(value, int) and not isinstance(value, bool)) or (
                    value is None and default is None
                )
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
        if out["base_model_id"] is not None and out["parent_model_id"] is not None:
            raise ValueError("set base_model_id or parent_model_id, not both")
        if out["parent_model_id"] is None:
            # Absent unless set, so jobs without a parent keep their stored params.
            del out["parent_model_id"]
        if "warmup_epochs" not in params:
            # The default warm-up must not make a short run (e.g. epochs=1) invalid.
            out["warmup_epochs"] = min(out["warmup_epochs"], out["epochs"])
        if out["warmup_epochs"] > out["epochs"]:
            raise ValueError(
                f"parameter warmup_epochs must be <= epochs ({out['epochs']}), got {out['warmup_epochs']}"
            )
        fixed = _fixed_input_size(out["model"])
        if fixed is not None and fixed != (out["imgsz"], out["imgsz"]):
            raise ValueError(
                f"parameter imgsz must be {fixed[0]} for model {out['model']} "
                f"(fixed input size {fixed[0]}x{fixed[1]}), got {out['imgsz']}"
            )
        return out

    # ------------------------------------------------------------------ prepare

    def prepare(self, ctx: JobContext) -> None:
        params = ctx.params
        if ctx.base_model is not None:
            # Before any crop is written: the base bundle must carry PyTorch weights.
            self._stage_base_weights(ctx)
        cached, cache = None, ""
        if (
            params["pretrained"]
            and ctx.base_model is None
            and _hf_offline()
            and self._local_weights(params["model"]) is None
        ):
            cached, cache = self._hf_cache_weights(params["model"])
        if cache and cached is None:
            candidates = self._local_weight_candidates(params["model"])
            where = (
                " or ".join(str(p) for p in candidates)
                if candidates
                else "<SPARROW_FT_WEIGHTS_DIR>/timm/"
                f"{params['model']}/{{{','.join(_TIMM_WEIGHT_FILES)}}} "
                "(SPARROW_FT_WEIGHTS_DIR is not set)"
            )
            raise ValueError(
                f"pretrained weights for {params['model']!r} cannot be downloaded with HF_HUB_OFFLINE set "
                f"and are not in {cache}; "
                f"place them at {where}, or set pretrained to false"
            )
        if params.get("stock_model"):
            # Fail before any crop is written when the stock package is unusable.
            load_stock_model(params["stock_model"], self._weights_dir)
        workdir = Path(ctx.workdir)
        crop_dir = workdir / "crops"
        crop_dir.mkdir(parents=True, exist_ok=True)
        label_index = {label: i for i, label in enumerate(ctx.labels)}

        plan: list[tuple[Any, str]] = []
        train_counts: Counter[int] = Counter()
        n_val = 0
        summary = {
            side: {"items": 0, "crops": 0, "groups": set()} for side in ("train", "val")
        }
        for item in ctx.items:
            side = assign_split(
                _split_key(item),
                seed=params["seed"],
                val_fraction=params["val_fraction"],
            )
            plan.append((item, side))
            summary[side]["items"] += 1
            if item.group:
                summary[side]["groups"].add(item.group)
            summary[side]["crops"] += len(item.annotations)
            if side == "train":
                train_counts.update(label_index[a["label"]] for a in item.annotations)
            else:
                n_val += len(item.annotations)

        floor = params["class_floor"]
        for i, label in enumerate(ctx.labels):
            if train_counts[i] < floor:
                raise ValueError(
                    f"label {label!r} has {train_counts[i]} training crop(s); class_floor requires at least {floor}"
                )
        if not n_val:
            raise ValueError(
                "the validation split is empty: add more images or groups, or raise val_fraction"
            )

        max_side = CROP_CACHE_FACTOR * params["imgsz"]
        index = []
        hashes: list[int] = []
        for item, side in plan:
            if not item.annotations:
                continue
            # One decode per source image, however many annotations it carries.
            with Image.open(item.image_path) as im:
                im = im.convert("RGB")
                for ann in item.annotations:
                    n = len(index)
                    if n % PREPARE_CHECK_EVERY == 0:
                        ctx.check_cancelled()
                    raw = im.crop(crop_box(ann.get("bbox"), im.width, im.height))
                    hashes.append(dhash(raw))
                    crop = _cap_long_side(raw, max_side)
                    name = f"{n:06d}.jpg"
                    crop.save(crop_dir / name, format="JPEG", quality=CROP_JPEG_QUALITY)
                    index.append(
                        {
                            "file": name,
                            "label": label_index[ann["label"]],
                            "split": side,
                        }
                    )
        (workdir / "crops.json").write_text(json.dumps(index))
        # dHash of each raw crop, aligned with crops.json (FT-5 near-duplicate report).
        np.save(workdir / "dhash.npy", np.asarray(hashes, dtype=np.uint64))
        info = {
            "split": {
                side: {
                    "items": s["items"],
                    "crops": s["crops"],
                    "groups": sorted(s["groups"]),
                }
                for side, s in summary.items()
            }
        }
        (workdir / "prepare.json").write_text(json.dumps(info, indent=2))
        ctx.emit("prepare", "crops written", crops=len(index))

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _split(workdir: Path, side: str) -> tuple[list[Path], list[int]]:
        index = json.loads((workdir / "crops.json").read_text())
        rows = [r for r in index if r["split"] == side]
        return [workdir / "crops" / r["file"] for r in rows], [
            int(r["label"]) for r in rows
        ]

    @staticmethod
    def _stage_base_weights(ctx: JobContext) -> None:
        """Extract ``weights.pt`` of the base bundle to ``workdir/base_weights.pt`` after checking it."""
        base = ctx.base_model
        path = Path(base["path"])
        if base.get("origin") == "local":
            ConventionalClassifierTrainer._stage_local_weights(ctx, path)
            return
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            manifest = (
                json.loads(zf.read("manifest.json")) if "manifest.json" in names else {}
            )
            if "weights.pt" not in names:
                raise ValueError(
                    f"base model {base['id']} has no PyTorch weights (weights.pt): it was exported by an "
                    "older version; re-train with this version to enable warm start"
                )
            data = zf.read("weights.pt")
        expected = (manifest.get("files") or {}).get("weights.pt")
        if expected != hashlib.sha256(data).hexdigest():
            raise ValueError(
                f"base model {base['id']}: weights.pt does not match the sha256 in its manifest"
            )
        arch = manifest.get("model_arch") or base.get("model_arch")
        if arch != ctx.params["model"]:
            raise ValueError(
                f"base model {base['id']} is a {arch} model, but this job trains {ctx.params['model']}"
            )
        workdir = Path(ctx.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / "base_weights.pt").write_bytes(data)

    @staticmethod
    def _stage_local_weights(ctx: JobContext, path: Path) -> None:
        """Re-validate a local ``sparrow-finetune/base-v1`` package and save its tensors for warm start.

        The package is checked again in full (sha256, safetensors framing, exact timm keys,
        shapes and dtypes of the declared ``model_arch``, finite values); nothing in it is unpickled or executed."""
        import torch

        from .. import base_models

        base = ctx.base_model
        if path.stat().st_size > base_models.MAX_LOCAL_PACKAGE_BYTES:
            raise ValueError(f"base model {base['id']}: the stored package is too large")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != base.get("sha256"):
            raise ValueError(
                f"base model {base['id']}: the stored package does not match its recorded sha256"
            )
        try:
            facts, state = base_models.load_local_package(data)
        except base_models.NotImportable as exc:
            raise ValueError(f"base model {base['id']}: {exc}") from exc
        if facts["model_arch"] != ctx.params["model"]:
            raise ValueError(
                f"base model {base['id']} is a {facts['model_arch']} model, "
                f"but this job trains {ctx.params['model']}"
            )
        if list(facts["labels"]) != list(base["labels"]):
            raise ValueError(
                f"base model {base['id']}: the package labels differ from the recorded labels"
            )
        workdir = Path(ctx.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        torch.save(state, workdir / "base_weights.pt")

    @staticmethod
    def _warm_start(ctx: JobContext, model, heads) -> None:
        """Load the base model's weights into ``model``; the head only when the label list is identical."""
        import torch

        base = ctx.base_model
        state = torch.load(
            Path(ctx.workdir) / "base_weights.pt", map_location="cpu", weights_only=True
        )
        same_labels = list(base["labels"]) == list(ctx.labels)
        if same_labels:
            model.load_state_dict(state)
            detail = "classifier head loaded (same labels in the same order)"
        else:
            head_ids = {id(p) for h in heads for p in h.parameters()}
            head_ids |= {id(b) for h in heads for b in h.buffers()}
            current = {**dict(model.named_parameters()), **dict(model.named_buffers())}
            head_keys = {k for k, v in current.items() if id(v) in head_ids}
            missing, unexpected = model.load_state_dict(
                {k: v for k, v in state.items() if k not in head_keys}, strict=False
            )
            if unexpected or set(missing) - head_keys:
                raise ValueError(
                    f"base model {base['id']} weights do not fit {ctx.params['model']}: "
                    f"missing {sorted(set(missing) - head_keys)}, unexpected {sorted(unexpected)}"
                )
            detail = (
                f"classifier head re-initialised (base labels {list(base['labels'])} "
                f"differ from job labels {list(ctx.labels)})"
            )
        ctx.emit(
            "train",
            f"warm start: loaded backbone weights from {base.get('kind', 'base')} model {base['id']} "
            f"({base['model_arch']}); {detail}",
            base_model_id=base["id"],
            head_reused=int(same_labels),
        )

    def _create(self, params: dict, n_labels: int, *, pretrained: bool):
        import timm

        local = self._local_weights(params["model"]) if pretrained else None
        if local is not None:
            # Load the pretrained weights from the local file: timm never reaches the Hub.
            return timm.create_model(
                params["model"],
                pretrained=True,
                pretrained_cfg_overlay={"file": str(local)},
                num_classes=n_labels,
            )
        return timm.create_model(
            params["model"], pretrained=pretrained, num_classes=n_labels
        )

    def _load_best(self, ctx: JobContext):
        import torch

        workdir = Path(ctx.workdir)
        model = self._create(ctx.params, len(ctx.labels), pretrained=False)
        model.load_state_dict(
            torch.load(workdir / "best.pt", map_location="cpu", weights_only=True)
        )
        trained = json.loads((workdir / "train.json").read_text())
        return _normalized(model, trained["mean"], trained["std"]).eval(), trained

    def _probe_peaks(self, params: dict, n_labels: int, device, amp: bool, batches) -> list[float]:
        """Peak reserved MB of two fine-tune steps (AdamW, AMP as configured) at each batch size."""
        import torch

        torch.cuda.init()  # reset_peak_memory_stats rejects the device before CUDA is initialised
        peaks = []
        for b in batches:
            model = opt = x = y = loss = None
            try:
                torch.cuda.empty_cache()
                before = torch.cuda.memory_reserved(device)
                torch.cuda.reset_peak_memory_stats(device)
                model = self._create(params, n_labels, pretrained=False).to(device).train()
                opt = torch.optim.AdamW(model.parameters(), lr=1e-6)
                scaler = torch.amp.GradScaler("cuda", enabled=amp)
                imgsz = params["imgsz"]
                for _ in range(2):
                    x = torch.rand(b, 3, imgsz, imgsz, device=device)
                    y = torch.randint(0, n_labels, (b,), device=device)
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                        loss = torch.nn.functional.cross_entropy(model(x), y)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                torch.cuda.synchronize(device)
                peaks.append((torch.cuda.max_memory_reserved(device) - before) / 2**20)
            finally:
                del model, opt, x, y, loss
                torch.cuda.empty_cache()
        return peaks

    def _resolve_batch(self, ctx: JobContext, device, amp: bool, n_train: int) -> tuple[int, dict]:
        """``(batch, hardware record)``; ``batch="auto"`` probes the GPU (see ``auto_batch``)."""
        import torch

        params = ctx.params
        source = "auto" if params["batch"] == "auto" else "param"
        hardware: dict[str, Any] = {"device": "cpu" if device.type == "cpu" else str(device)}
        if device.type == "cuda":
            try:
                index = device.index if device.index is not None else torch.cuda.current_device()
                torch.cuda.empty_cache()
                free, total = torch.cuda.mem_get_info(index)
                hardware.update(
                    gpu_name=str(torch.cuda.get_device_name(index)),
                    gpu_total_mb=int(total // 2**20),
                    gpu_free_mb=int(free // 2**20),
                )
            except Exception as exc:  # noqa: BLE001 - a failed probe must not stop a job with a fixed batch
                hardware["gpu_probe_error"] = f"{type(exc).__name__}: {exc}"
        if source == "param":
            chosen = int(params["batch"])
        elif "gpu_free_mb" not in hardware:
            chosen = CPU_AUTO_BATCH
        else:
            ctx.emit("train", "measuring GPU memory for batch=auto")
            batches = _PROBE_BATCHES
            try:
                peaks = self._probe_peaks(params, len(ctx.labels), device, amp, batches)
            except torch.cuda.OutOfMemoryError:
                batches = _PROBE_FALLBACK_BATCHES
                try:
                    peaks = self._probe_peaks(params, len(ctx.labels), device, amp, batches)
                except torch.cuda.OutOfMemoryError:
                    raise RuntimeError(
                        f"model {params['model']} at imgsz {params['imgsz']} does not fit in the "
                        f"{hardware['gpu_free_mb']} MB free on GPU {hardware['device']} even at batch "
                        f"{batches[1]}; use a smaller model or imgsz"
                    ) from None
            per_image = max(1.0, (peaks[1] - peaks[0]) / (batches[1] - batches[0]))
            fixed = max(0.0, peaks[0] - per_image * batches[0])
            chosen = auto_batch(hardware["gpu_free_mb"], fixed, per_image, n_train)
            hardware.update(probe_fixed_mb=round(fixed), probe_mb_per_image=round(per_image, 1))
            chosen = self._verify_batch(params, len(ctx.labels), device, amp, chosen, hardware)
        hardware.update(batch=chosen, batch_source=source)
        return chosen, hardware

    def _verify_batch(self, params, n_labels, device, amp, chosen: int, hardware: dict) -> int:
        """Run the chosen batch once; shrink it while its peak exceeds the headroom budget.

        The line through the batch-8 and batch-32 peaks under-predicts small models by up to
        ~50 % (allocator and cuDNN workspace noise dominate there), so it is only a first guess.
        """
        import torch

        budget = hardware["gpu_free_mb"] * _AUTO_HEADROOM
        for _ in range(_VERIFY_ATTEMPTS):
            if chosen <= _PROBE_BATCHES[1]:
                return chosen  # already measured within the probe
            try:
                (peak,) = self._probe_peaks(params, n_labels, device, amp, (chosen,))
            except torch.cuda.OutOfMemoryError:
                peak = float("inf")
            if peak <= budget:
                hardware["probe_verified_mb"] = round(peak)
                return chosen
            ratio = 0.75 if peak == float("inf") else budget / peak
            chosen = max(_PROBE_BATCHES[1], min(chosen - 8, int(chosen * ratio) // 8 * 8))
        return _PROBE_BATCHES[1]  # the largest batch actually measured to fit

    @staticmethod
    def _predict(net, dataset, device, batch: int, *, amp: bool = False) -> np.ndarray:
        import torch

        loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=False)
        out = []
        with torch.no_grad():
            for x, _y in loader:
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=amp
                ):
                    logits = net(x.to(device))
                out.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        probs = np.concatenate(out).astype(np.float64)
        return probs / probs.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------ train

    def train(self, ctx: JobContext) -> None:
        import torch
        from sklearn.metrics import f1_score, log_loss

        params = ctx.params
        workdir = Path(ctx.workdir)
        device = _resolve_device(params["device"])
        amp = _amp_enabled(params, device)
        classes = np.arange(len(ctx.labels))
        train_files, train_labels = self._split(workdir, "train")
        val_files, val_labels = self._split(workdir, "val")
        y_val = np.asarray(val_labels)
        train_ds = _CropDataset(
            train_files,
            train_labels,
            params["imgsz"],
            augment=params["augment"],
            seed=params["seed"],
        )
        val_ds = _CropDataset(
            val_files, val_labels, params["imgsz"], augment=False, seed=params["seed"]
        )
        sampler = _EpochSampler(len(train_ds), params["seed"])
        with _private_rng(device):
            batch, hardware = self._resolve_batch(ctx, device, amp, len(train_ds))
        ctx.emit("train", _hardware_message(hardware), batch=batch)
        loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch,
            sampler=sampler,
            num_workers=params["workers"],
            persistent_workers=params["workers"] > 0,
        )

        with _private_rng(device):
            torch.manual_seed(params["seed"])
            warm = ctx.base_model is not None
            # A warm start replaces the backbone weights, so the pretrained ones are not loaded.
            model = self._create(
                params, len(ctx.labels), pretrained=params["pretrained"] and not warm
            )
            cfg = getattr(model, "pretrained_cfg", None) or {}
            mean = [float(v) for v in cfg.get("mean", (0.485, 0.456, 0.406))]
            std = [float(v) for v in cfg.get("std", (0.229, 0.224, 0.225))]
            # Distilled heads (deit_*_distilled, levit_*, efficientformer_l*) return (head, head_dist).
            heads = model.get_classifier()
            heads = heads if isinstance(heads, (tuple, list)) else (heads,)
            if warm:
                self._warm_start(ctx, model, heads)
            net = _normalized(model, mean, std).to(device)
            head_params = [p for h in heads for p in h.parameters()]
            head_ids = {id(p) for p in head_params}
            backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
            criterion = torch.nn.CrossEntropyLoss(
                label_smoothing=params["label_smoothing"]
            )
            scaler = torch.amp.GradScaler("cuda", enabled=amp)

            warmup, total = params["warmup_epochs"], params["epochs"]
            finetune_epochs = total - warmup
            optimizer = None
            best_f1, best_epoch, since_best = -1.0, 0, 0
            curve: list[dict] = []
            stopped_early = False
            for epoch in range(1, total + 1):
                ctx.check_cancelled()
                phase = "warmup" if epoch <= warmup else "finetune"
                if epoch == 1 and phase == "warmup":
                    for p in backbone_params:
                        p.requires_grad_(False)
                    optimizer = torch.optim.AdamW(
                        head_params,
                        lr=params["lr"],
                        weight_decay=params["weight_decay"],
                    )
                if epoch == warmup + 1:
                    for p in model.parameters():
                        p.requires_grad_(True)
                    optimizer = torch.optim.AdamW(
                        [
                            {
                                "params": backbone_params,
                                "lr": params["lr"] * params["backbone_lr_factor"],
                            },
                            {"params": head_params, "lr": params["lr"]},
                        ],
                        weight_decay=params["weight_decay"],
                    )
                if phase == "finetune":
                    k = epoch - warmup - 1
                    scale = 0.5 * (1.0 + math.cos(math.pi * k / finetune_epochs))
                    head_lr = params["lr"] * scale
                    backbone_lr = head_lr * params["backbone_lr_factor"]
                    optimizer.param_groups[0]["lr"] = backbone_lr
                    optimizer.param_groups[1]["lr"] = head_lr
                    net.train()
                else:
                    head_lr, backbone_lr = params["lr"], 0.0
                    net.eval()  # frozen backbone: BatchNorm statistics stay as loaded

                sampler.set_epoch(epoch)
                # Summed on the device and read once per epoch: a per-step float() would sync the GPU.
                losses = torch.zeros((), dtype=torch.float64, device=device)
                seen = 0
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    # Train-mode BatchNorm cannot normalise one sample over a 1x1 map; a size-1 step uses
                    # the running statistics instead.
                    single = phase == "finetune" and len(y) == 1
                    if single:
                        net.eval()
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(
                        device_type=device.type, dtype=torch.float16, enabled=amp
                    ):
                        loss = criterion(net(x), y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    if single:
                        net.train()
                    losses += loss.detach().double() * len(y)
                    seen += len(y)

                net.eval()
                # fp32 like evaluate, so the epoch kept is the one evaluate reports on.
                probs = self._predict(net, val_ds, device, batch, amp=False)
                val_f1 = float(
                    f1_score(
                        y_val,
                        probs.argmax(axis=1),
                        labels=classes,
                        average="macro",
                        zero_division=0,
                    )
                )
                point = {
                    "epoch": epoch,
                    "phase": phase,
                    "train_loss": float(losses) / max(seen, 1),
                    "val_loss": float(log_loss(y_val, probs, labels=classes)),
                    "val_macro_f1": val_f1,
                    "lr": float(head_lr),
                    "backbone_lr": float(backbone_lr),
                }
                curve.append(point)
                ctx.emit("train", f"epoch {epoch}", **point)
                state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                torch.save(state, workdir / "last.pt")
                if val_f1 > best_f1:
                    best_f1, best_epoch, since_best = val_f1, epoch, 0
                    torch.save(state, workdir / "best.pt")
                elif phase == "finetune":
                    since_best += 1
                if (
                    params["patience"]
                    and phase == "finetune"
                    and since_best >= params["patience"]
                ):
                    stopped_early = epoch < total
                    break

        (workdir / "train.json").write_text(
            json.dumps(
                {
                    "best_epoch": best_epoch,
                    "epochs": curve,
                    "amp": amp,
                    "device": str(device),
                    "hardware": hardware,
                    "stopped_early": stopped_early,
                    "mean": mean,
                    "std": std,
                },
                indent=2,
            )
        )

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
        params = ctx.params
        device = _resolve_device(params["device"])
        classes = np.arange(len(ctx.labels))
        train_files, _ = self._split(workdir, "train")
        val_files, val_labels = self._split(workdir, "val")
        y_val = np.asarray(val_labels)
        ctx.check_cancelled()
        with _private_rng(device):
            net, trained = self._load_best(ctx)
            probs = self._predict(
                net.to(device),
                _CropDataset(
                    val_files,
                    val_labels,
                    params["imgsz"],
                    augment=False,
                    seed=params["seed"],
                ),
                device,
                EVAL_BATCH,
            )
        ctx.check_cancelled()
        pred = probs.argmax(axis=1)
        metrics = {
            "val_macro_f1": float(
                f1_score(y_val, pred, labels=classes, average="macro", zero_division=0)
            ),
            "val_accuracy": float(accuracy_score(y_val, pred)),
            "val_log_loss": float(log_loss(y_val, probs, labels=classes)),
            "val_ece": _expected_calibration_error(probs, y_val),
            "best_epoch": float(trained["best_epoch"]),
            "train_crops": float(len(train_files)),
            "val_crops": float(len(val_files)),
        }
        threshold = threshold_for_params(params, probs, y_val)
        metrics.update(threshold_metrics(threshold))
        precision, recall, f1, support = precision_recall_fscore_support(
            y_val, pred, labels=classes, zero_division=0
        )
        report = {
            "metrics": metrics,
            "per_class": {
                label: {
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1[i]),
                    "support": int(support[i]),
                }
                for i, label in enumerate(ctx.labels)
            },
            "confusion_matrix": confusion_matrix(y_val, pred, labels=classes).tolist(),
            "threshold": threshold,
        }
        splits = np.asarray(
            [r["split"] for r in json.loads((workdir / "crops.json").read_text())]
        )
        report["near_duplicates"] = near_duplicate_report(
            ctx,
            np.load(workdir / "dhash.npy"),
            splits == "train",
            splits == "val",
            metrics,
        )
        if params.get("stock_model"):
            report["stock_model"] = self._stock_comparison(ctx, y_val, pred, metrics)
        (workdir / "evaluate.json").write_text(
            json.dumps(dict(report, metrics=json_metrics(metrics)), indent=2)
        )
        return metrics

    def _stock_comparison(self, ctx: JobContext, y_val, pred, metrics: dict) -> dict:
        """Score the stock model on the source-image val crops whose label it knows (FT-16, FT-18)."""
        params = ctx.params
        stock = load_stock_model(params["stock_model"], self._weights_dir)
        mapping = matched_label_map(ctx.labels, stock.labels)
        label_index = {label: i for i, label in enumerate(ctx.labels)}
        matched = np.isin(y_val, list(mapping))

        # Same split and annotation order as prepare's crops.json, so crops align with y_val.
        def keep(item, ann) -> bool:
            side = assign_split(
                _split_key(item),
                seed=params["seed"],
                val_fraction=params["val_fraction"],
            )
            return side == "val" and label_index[ann["label"]] in mapping

        crops = source_crops(ctx.items, keep) if matched.any() else []
        return stock_report(
            ctx, stock, mapping, y_val[matched], pred[matched], crops, metrics
        )

    # ------------------------------------------------------------------ export

    def export(self, ctx: JobContext) -> ExportResult:
        import onnxruntime as ort
        import torch

        workdir = Path(ctx.workdir)
        params = ctx.params
        imgsz = params["imgsz"]
        prepared = json.loads((workdir / "prepare.json").read_text())
        report = json.loads((workdir / "evaluate.json").read_text())
        val_files, val_labels = self._split(workdir, "val")
        val_ds = _CropDataset(
            val_files, val_labels, imgsz, augment=False, seed=params["seed"]
        )

        ctx.check_cancelled()
        cpu = torch.device("cpu")
        with _private_rng(cpu):
            net, trained = self._load_best(ctx)
            model_bytes = dynamo_onnx_bytes(
                _with_softmax(net).eval(),
                torch.zeros(2, 3, imgsz, imgsz),
                input_name="images",
                output_name="probabilities",
                graph_name="conventional_classifier",
                opset=ONNX_OPSET,
            )
            ctx.check_cancelled()
            torch_probs = self._predict(net, val_ds, cpu, EVAL_BATCH)

        session = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
        onnx_chunks = []
        for start in range(0, len(val_ds), EVAL_BATCH):
            x = np.stack(
                [
                    val_ds[i][0].numpy()
                    for i in range(start, min(start + EVAL_BATCH, len(val_ds)))
                ]
            )
            onnx_chunks.append(
                np.asarray(
                    session.run(["probabilities"], {"images": x.astype(np.float32)})[0]
                )
            )
        ctx.check_cancelled()
        onnx_probs = np.concatenate(onnx_chunks).astype(np.float64)
        parity = parity_check(
            onnx_probs, torch_probs, max_abs_delta_limit=PARITY_MAX_ABS_DELTA
        )

        metrics = report["metrics"]
        card = {
            "pipeline": self.id,
            "model": params["model"],
            "labels": list(ctx.labels),
            "params": dict(params),
            "metrics": metrics,
            "per_class": report["per_class"],
            "confusion_matrix": report["confusion_matrix"],
            "split": prepared["split"],
            "epochs": trained["epochs"],
            "amp": trained["amp"],
            "device": trained["device"],
            "hardware": trained.get("hardware"),
            "stopped_early": trained["stopped_early"],
            "near_duplicates": report.get("near_duplicates"),
            "threshold": report.get("threshold"),
        }
        stock = report.get("stock_model")
        if stock:
            card["stock_model"] = stock
        card_bytes = json.dumps(card, indent=2).encode("utf-8")
        # The bare timm state dict of the kept epoch: what a later job warm starts from.
        weights_bytes = (workdir / "best.pt").read_bytes()
        manifest = {
            "pipeline": self.id,
            "model": params["model"],
            "model_arch": params["model"],
            "labels": list(ctx.labels),
            "input": {
                "name": "images",
                "size": imgsz,
                "layout": "NCHW",
                "color": "RGB",
                "scale": [0, 1],
                "resize": "stretch",
                "mean": trained["mean"],
                "std": trained["std"],
            },
            "output": "probabilities",
            "metrics": metrics,
            "parity": parity,
            "files": {
                "model.onnx": hashlib.sha256(model_bytes).hexdigest(),
                "model_card.json": hashlib.sha256(card_bytes).hexdigest(),
                "weights.pt": hashlib.sha256(weights_bytes).hexdigest(),
            },
            **threshold_manifest(report.get("threshold")),
        }
        path = workdir / "conventional_classifier_bundle.zip"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("model.onnx", model_bytes)
            zf.writestr("model_card.json", card_bytes)
            zf.writestr("weights.pt", weights_bytes)
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        ctx.emit("export", "bundle written", **parity)
        metadata = {
            "labels": list(ctx.labels),
            "hardware": trained.get("hardware"),
            **threshold_manifest(report.get("threshold")),
        }
        if stock:
            metadata["stock_model"] = stock["id"]
        if params.get("base_model_id") is not None:
            metadata["base_model_id"] = params["base_model_id"]
        if params.get("parent_model_id") is not None:
            metadata["parent_model_id"] = params["parent_model_id"]
        return ExportResult(path=path, format="onnx-bundle", metadata=metadata)
