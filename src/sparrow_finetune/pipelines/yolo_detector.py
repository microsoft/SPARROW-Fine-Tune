"""``yolo_detector``: an ultralytics YOLO object detector fine-tuned on a detection snapshot.

``prepare`` writes a YOLO dataset (``images/{train,val}``, ``labels/{train,val}``, ``data.yaml``) with the
group-aware split of ``dinov3_linear``; ``train`` runs ``YOLO(...).train`` and reports one event per epoch;
``evaluate`` validates the best checkpoint on the val split; ``export`` writes a static-shape ONNX model
behind a parity gate against the PyTorch model. State between stages lives in ``ctx.workdir``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
import re
import shutil
import zipfile
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional

import numpy as np

from . import ExportResult, JobContext, Trainer
from .dinov3_linear import assign_split
from .near_dup import DEFAULT_MAX_DISTANCE, dhash, near_duplicate_report

PARITY_MAX_ABS_DELTA = 1e-3
_DEVICE_RE = re.compile(r"auto|cpu|cuda|cuda:[0-9]+|[0-9]+")
# An ultralytics model name (yolo11n.pt, yolov8s.pt, yolo11n.yaml). Paths and URLs are refused: YOLO() unpickles
# a .pt with weights_only=False, so a caller-chosen file could run arbitrary code. Local weights come in only
# through base_model_resolver.
_MODEL_RE = re.compile(r"yolo(?:v?[0-9]{1,2})[nsmlx]\.(?:pt|yaml)")
# MegaDetector v6 checkpoints (Microsoft AI for Good, Zenodo record 15398270, CC-BY-4.0) offered as named
# starting weights. Only these pinned files are ever loaded: the sha256 is checked before ultralytics
# unpickles them. "arch" is the ultralytics network (yolov10n / yolov10x); "imgsz" is what it was trained at.
MEGADETECTOR_URL = "https://zenodo.org/records/15398270/files/{file}?download=1"
MEGADETECTOR_BASES: dict[str, dict[str, Any]] = {
    "mdv6-yolov10-c": {
        "name": "MegaDetector v6 compact",
        "file": "MDV6-yolov10-c.pt",
        "sha256": "21ee78a2d4887128e2a4920937d3295b493f44d788d13e1a635378c16dd74ef7",
        "size": 5752051,
        "arch": "yolov10n",
        "imgsz": 640,
    },
    "mdv6-yolov10-e": {
        "name": "MegaDetector v6 extra",
        "file": "MDV6-yolov10-e-1280.pt",
        "sha256": "4a3a3d380ce7e151b2a8b991ab5d86f329ccd7b0b33e5d3ba0593a1166d55109",
        "size": 127821713,
        "arch": "yolov10x",
        "imgsz": 1280,
    },
}
_DOWNLOAD_TIMEOUT_S = 120
# A dropped connection mid-transfer is common on a 128 MB file; each attempt is size- and hash-checked.
_DOWNLOAD_ATTEMPTS = 3
# Formats cv2 (the ultralytics loader) reads; anything else is re-encoded to PNG in prepare.
_COPY_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
_EXIF_ORIENTATION = 0x0112
# Process-wide state ultralytics changes in select_device / init_seeds and never restores.
_GLOBAL_ENV = ("CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED")
# Weights ultralytics' AMP check loads by bare name (resolved against the cwd unless redirected).
_AMP_CHECK_MODEL = "yolo11n.pt"

# name -> (kind, default, check, requirement shown in the error)
_PARAMS: dict[str, tuple[str, Any, Callable[[Any], bool], str]] = {
    "model": (
        "str",
        "yolo11n.pt",
        lambda v: _MODEL_RE.fullmatch(v) is not None or v in MEGADETECTOR_BASES,
        "an ultralytics model name such as yolo11n.pt, yolov8s.pt or yolo11n.yaml, "
        "or a MegaDetector v6 base: " + ", ".join(MEGADETECTOR_BASES),
    ),
    "epochs": ("int", 50, lambda v: 1 <= v <= 1000, "an integer in 1..1000"),
    "imgsz": (
        "int",
        640,
        lambda v: 32 <= v <= 2048 and v % 32 == 0,
        "an integer in 32..2048 and a multiple of 32",
    ),
    "batch": (
        "batch",
        16,
        lambda v: v == "auto" or 1 <= v <= 256,
        'an integer in 1..256, or "auto" (or -1) to size it from the free GPU memory',
    ),
    "seed": ("int", 0, lambda v: 0 <= v < 2**32, "an integer in [0, 2**32)"),
    "val_fraction": ("float", 0.2, lambda v: 0 < v <= 0.5, "a number in (0, 0.5]"),
    "device": (
        "str",
        "auto",
        lambda v: _DEVICE_RE.fullmatch(v) is not None,
        'one of "auto", "cpu", "cuda", "cuda:N" or "N"',
    ),
    # 10 so early stopping can trigger within the default 50 epochs.
    "patience": ("int", 10, lambda v: 0 <= v <= 1000, "an integer in 0..1000 (0 disables early stopping)"),
    "lr0": ("float", 0.01, lambda v: v > 0, "a finite number > 0"),
    "workers": ("int", 0, lambda v: 0 <= v <= 32, "an integer in 0..32"),
    "near_dup_max_distance": (
        "int",
        DEFAULT_MAX_DISTANCE,
        lambda v: 0 <= v <= 32,
        "an integer in 0..32",
    ),
}

_METRIC_KEYS = {
    "map50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
}


def to_yolo_label(bbox, width: int, height: int) -> tuple[float, float, float, float]:
    """Pixel ``[x, y, w, h]`` -> normalised ``(cx, cy, w, h)``, clamped to the image first."""
    x, y, w, h = (float(v) for v in bbox)
    width, height = float(width), float(height)
    x0, y0 = min(max(x, 0.0), width), min(max(y, 0.0), height)
    x1, y1 = min(max(x + w, 0.0), width), min(max(y + h, 0.0), height)
    if not (x1 > x0 and y1 > y0):
        raise ValueError(
            f"bbox {list(bbox)} has zero area inside a {width:g}x{height:g} image"
        )
    clip = lambda v: min(max(v, 0.0), 1.0)  # noqa: E731
    return (
        clip((x0 + x1) / 2 / width),
        clip((y0 + y1) / 2 / height),
        clip((x1 - x0) / width),
        clip((y1 - y0) / height),
    )


def _resolve_device(device: str) -> str:
    if device == "auto":
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    return device


def _check_cuda_device(device: str) -> None:
    """Fail fast on a CUDA device this worker does not have (the API host may differ from the worker)."""
    if device == "cpu":
        return
    import torch

    count = torch.cuda.device_count()
    if count == 0:
        raise ValueError(
            f"parameter device={device!r} asks for a GPU but this worker has no CUDA device; "
            "use device='cpu' or 'auto'"
        )
    index = 0 if device == "cuda" else int(device.removeprefix("cuda:"))
    if index >= count:
        raise ValueError(
            f"parameter device={device!r} names a CUDA device this worker does not have "
            f"({count} CUDA device(s) visible)"
        )


def _clear_font_cache(check_font: Any) -> None:
    """Drop cached font paths (they point into the previous config dir); the lru_cache sits under a lock wrapper."""
    fn = check_font
    while fn is not None:
        if hasattr(fn, "cache_clear"):
            fn.cache_clear()
            return
        fn = getattr(fn, "__wrapped__", None)


@contextmanager
def _ultralytics_sandbox(
    workdir: Path, model_source: Callable[[str], str]
) -> Iterator[None]:
    """Keep ultralytics' files (settings, caches, fonts, downloaded weights) inside ``workdir``.

    ultralytics binds its config dir when first imported, so the env var covers a first import here and the
    module attributes later writes go through are redirected for an already-imported ultralytics. The AMP
    check loads ``yolo11n.pt`` by bare name, which would download into the process cwd; it gets
    ``model_source(name)`` instead. ``XDG_CACHE_HOME`` covers onnxruntime, which writes a device-id cache
    when first imported.

    When the first import happens here, the values recorded for restoration are the sandbox paths. On exit
    they are rebound to the config dir ultralytics picks under the restored environment, and the settings and
    persistent cache are reloaded from there, so no ultralytics global keeps pointing into ``workdir``.
    """
    config_root = Path(workdir).resolve() / "ultralytics"
    user_dir = config_root / "Ultralytics"
    user_dir.mkdir(parents=True, exist_ok=True)
    env = {
        "YOLO_CONFIG_DIR": str(config_root),
        "MPLCONFIGDIR": str(config_root / "matplotlib"),
        "XDG_CACHE_HOME": str(config_root / "cache"),
    }
    saved_env = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    patched: list[tuple[Any, str, Any]] = []
    try:
        import ultralytics
        import ultralytics.cfg as ul_cfg
        import ultralytics.engine.trainer as ul_trainer
        import ultralytics.utils as ul_utils
        import ultralytics.utils.checks as ul_checks
        import ultralytics.utils.dist as ul_dist

        check_amp = ul_trainer.check_amp

        def sandboxed_check_amp(model):
            real_yolo = ultralytics.YOLO

            def local_yolo(name: str = _AMP_CHECK_MODEL, *args, **kwargs):
                return real_yolo(model_source(str(name)), *args, **kwargs)

            ultralytics.YOLO = local_yolo
            try:
                return check_amp(model)
            finally:
                ultralytics.YOLO = real_yolo

        targets = [
            (ul_utils, "USER_CONFIG_DIR", user_dir),
            (ul_utils, "SETTINGS_FILE", user_dir / "settings.json"),
            (ul_checks, "USER_CONFIG_DIR", user_dir),
            (ul_dist, "USER_CONFIG_DIR", user_dir),
            (ul_utils.SETTINGS, "file", user_dir / "settings.json"),
            (ul_utils.SETTINGS, "file_path", user_dir / "settings.json"),
            (
                ul_utils.PERSISTENT_CACHE,
                "file_path",
                user_dir / "persistent_cache.json",
            ),
            (ul_trainer, "check_amp", sandboxed_check_amp),
            # The GPU AMP check predicts with no project, saving under RUNS_DIR (cwd-relative by default).
            (ul_utils, "RUNS_DIR", config_root / "runs"),
            (ul_cfg, "RUNS_DIR", config_root / "runs"),
        ]
        for obj, attr, value in targets:
            patched.append((obj, attr, getattr(obj, attr)))
            setattr(obj, attr, value)
        _clear_font_cache(ul_checks.check_font)
        yield
    finally:
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if patched:
            _restore_ultralytics(patched, user_dir)
            _clear_font_cache(ul_checks.check_font)


def _default_config_dir(ul_utils: Any) -> Path:
    """The dir ``ultralytics.utils.get_user_config_dir`` picks under the current env, without creating it."""
    if env_dir := os.getenv("YOLO_CONFIG_DIR"):
        return Path(env_dir).expanduser() / "Ultralytics"
    if ul_utils.WINDOWS:
        return Path.home() / "AppData" / "Roaming" / "Ultralytics"
    if ul_utils.MACOS:
        return Path.home() / "Library" / "Application Support" / "Ultralytics"
    return Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config")) / "Ultralytics"


def _restore_ultralytics(patched: list[tuple[Any, str, Any]], user_dir: Path) -> None:
    """Undo the sandbox patches; originals inside ``user_dir`` (a first import in the sandbox) go to the real dir."""
    import ultralytics.utils as ul_utils

    def inside(value: Any) -> bool:
        return isinstance(value, Path) and value.is_relative_to(user_dir)

    leaked = any(inside(value) for _, _, value in patched)
    real_dir = _default_config_dir(ul_utils) if leaked else user_dir
    for obj, attr, value in reversed(patched):
        if inside(value):
            value = real_dir / value.relative_to(user_dir)
        setattr(obj, attr, value)
    if not leaked:
        return
    # SETTINGS and PERSISTENT_CACHE were loaded from the sandbox files; reload them from the real ones. Without a
    # real settings file the defaults ultralytics created in the sandbox stay (what a fresh import would hold).
    for store in (ul_utils.SETTINGS, ul_utils.PERSISTENT_CACHE):
        if store.file_path.is_file():
            with store.lock:
                dict.clear(store)
                store._load()
    if ul_utils.SETTINGS.keys() >= {"datasets_dir", "weights_dir", "runs_dir"}:
        ul_utils.DATASETS_DIR = Path(ul_utils.SETTINGS["datasets_dir"])
        ul_utils.WEIGHTS_DIR = Path(ul_utils.SETTINGS["weights_dir"])
        ul_utils.RUNS_DIR = Path(ul_utils.SETTINGS["runs_dir"])


@contextmanager
def _restore_global_state() -> Iterator[None]:
    """Undo ultralytics' process-wide changes so later jobs in the same worker see the original devices."""
    import torch

    env = {k: os.environ.get(k) for k in _GLOBAL_ENV}
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_deterministic = torch.backends.cudnn.deterministic
    try:
        yield
    finally:
        for key, value in env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cudnn.deterministic = cudnn_deterministic


def _write_image(source: Path, dest_stem: Path) -> int:
    """Copy the image byte for byte when ultralytics reads it in the stored (raw) frame; else re-encode to PNG.

    Labels are normalised against the raw stored size, so an EXIF-rotated image is written without its
    orientation tag (raw pixels), and formats cv2 cannot read (GIF, ...) are converted. Returns the
    raw image's dHash for the near-duplicate report (FT-38).
    """
    from PIL import Image

    suffix = source.suffix.lower()
    with Image.open(source) as im:
        rotated = im.getexif().get(_EXIF_ORIENTATION, 1) != 1
        rgb = im.convert("RGB")
    image_hash = dhash(rgb)
    if suffix in _COPY_SUFFIXES and not rotated:
        shutil.copyfile(source, dest_stem.with_suffix(suffix))
        return image_hash
    rgb.info.clear()
    rgb.save(dest_stem.with_suffix(".png"), format="PNG", compress_level=1)
    return image_hash


def _finite(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if math.isfinite(out) else 0.0


def _split_finite(values: dict) -> tuple[dict[str, float], list[str]]:
    """``(finite metrics, sorted names of missing/non-finite ones)``: "no signal" must not read as a real 0.0."""
    kept: dict[str, float] = {}
    dropped: list[str] = []
    for key, value in values.items():
        try:
            out = float(value)
        except (TypeError, ValueError):
            out = math.nan
        if math.isfinite(out):
            kept[key] = out
        else:
            dropped.append(key)
    return kept, sorted(dropped)


CONFUSION_CONF = 0.25  # ultralytics' confusion-matrix confidence when val runs at its 0.001 default
CONFUSION_IOU = 0.45


def _confusion_validator() -> type:
    """A ``DetectionValidator`` that fills the confusion matrix with ``plots=False``.

    Ultralytics only updates its matrix when it also draws plots; this subclass updates it on every
    batch so evaluation gets the matrix without writing PNGs.
    """
    from ultralytics.models.yolo.detect import DetectionValidator

    class ConfusionValidator(DetectionValidator):
        def update_metrics(self, preds, batch):  # type: ignore[override]
            super().update_metrics(preds, batch)
            if self.args.plots:
                return  # ultralytics already updated it
            for si, pred in enumerate(preds):
                self.confusion_matrix.process_batch(
                    self._prepare_pred(pred), self._prepare_batch(si, batch), conf=self.args.conf
                )

    return ConfusionValidator


def confusion_report(matrix: Any, labels: list[str]) -> Optional[dict[str, Any]]:
    """Ultralytics' ``(nc+1)x(nc+1)`` matrix (rows predicted, columns true, last index background) as
    ``{"labels", "matrix", ...}`` with rows true and columns predicted, the classifier layout.

    The ``background`` row counts false detections (no ground-truth box); the ``background`` column
    counts missed boxes. None when the matrix has the wrong shape.
    """
    if matrix is None:
        return None
    m = np.asarray(matrix)
    n = len(labels) + 1
    if m.shape != (n, n):
        return None
    return {
        "labels": [*labels, "background"],
        "matrix": [[int(round(v)) for v in row] for row in m.T],
        "conf": CONFUSION_CONF,
        "iou": CONFUSION_IOU,
        "rows": "true",
        "columns": "predicted",
    }


def _per_class_ap(
    box: Any, labels: list[str]
) -> tuple[dict[str, dict[str, float]], list[str]]:
    """``({label: {"ap50", "ap50_95"}}, labels absent from val or with a non-finite AP)``.

    ultralytics' per-class arrays (``ap50``, ``ap``) hold one row per class present in val, in the order of
    ``ap_class_index``; class ids are positions in the job labels (``prepare`` writes ``names`` that way).
    """
    ap50 = getattr(box, "ap50", [])
    ap = getattr(box, "ap", [])
    index = getattr(box, "ap_class_index", [])
    per_class: dict[str, dict[str, float]] = {}
    for row, cls in enumerate(index):
        cls = int(cls)
        if not 0 <= cls < len(labels) or row >= len(ap50) or row >= len(ap):
            continue
        kept, dropped = _split_finite({"ap50": ap50[row], "ap50_95": ap[row]})
        if not dropped:
            per_class[labels[cls]] = kept
    return per_class, [label for label in labels if label not in per_class]


def _cell(values: Any, row: int) -> Optional[float]:
    """``float(values[row])`` when present and finite, else None."""
    try:
        out = float(values[row])
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    return out if math.isfinite(out) else None


def per_class_table(
    box: Any, labels: list[str], instances: Any = None
) -> list[dict[str, Any]]:
    """One row per job label: ``{"label", "precision", "recall", "map50", "map50_95", "instances"}``.

    ultralytics' ``p``, ``r``, ``ap50`` and ``ap`` hold one row per class present in val, in the order of
    ``ap_class_index``; a label absent from val keeps null metrics. ``instances`` is the val box count per
    class id (``results.nt_per_class``), or None when the results do not carry it.
    """
    rows = {
        label: {
            "label": label,
            "precision": None,
            "recall": None,
            "map50": None,
            "map50_95": None,
            "instances": None,
        }
        for label in labels
    }
    if instances is not None:
        for cls, label in enumerate(labels):
            try:
                rows[label]["instances"] = int(instances[cls])
            except (TypeError, ValueError, IndexError):
                pass
    columns = {"precision": "p", "recall": "r", "map50": "ap50", "map50_95": "ap"}
    index = getattr(box, "ap_class_index", None)
    for row, cls in enumerate([] if index is None else index):
        cls = int(cls)
        if not 0 <= cls < len(labels):
            continue
        for key, attr in columns.items():
            rows[labels[cls]][key] = _cell(getattr(box, attr, None), row)
    return [rows[label] for label in labels]


# Workbench auto-batch heuristic (``sparrow_yolo.hardware.suggest_batch``), ported for one training GPU:
# peak activation MB per image at 1024 px, calibrated on yolo11l (the rest keep their ratios to it).
_MB_PER_IMAGE_AT_1024 = {
    "yolo11n": 310,
    "yolo11s": 490,
    "yolo11m": 810,
    "yolo11l": 1030,
    "yolo11x": 1390,
    "yolo12n": 340,
    "yolo12s": 540,
    "yolo12m": 850,
    "yolo12l": 1080,
    "yolo12x": 1430,
    "yolov8n": 290,
    "yolov8s": 450,
    "yolov8m": 760,
    "yolov8l": 990,
    "yolov8x": 1340,
}
_MB_BY_SCALE = {"n": 310, "s": 490, "m": 810, "l": 1030, "x": 1390}
_SCALE_RE = re.compile(r"^yolo[a-z]*\d+([nsmlx])$")
_DEFAULT_MB_PER_IMAGE = 1030
_RESERVE_GB = 6.0  # fixed cost per GPU: weights, grads, optimiser, EMA, cuDNN workspace, CUDA context
_HEADROOM = 0.85
_MAX_AUTO_BATCH = 64
CPU_AUTO_BATCH = 16


def _mb_per_image(model: str) -> int:
    if model in MEGADETECTOR_BASES:
        model = MEGADETECTOR_BASES[model]["arch"]
    key = Path(str(model)).stem.lower().replace("-", "")
    if key in _MB_PER_IMAGE_AT_1024:
        return _MB_PER_IMAGE_AT_1024[key]
    match = _SCALE_RE.match(key)
    return _MB_BY_SCALE[match.group(1)] if match else _DEFAULT_MB_PER_IMAGE


def _gpu_index(device: str) -> int:
    return 0 if device == "cuda" else int(device.removeprefix("cuda:"))


def resolve_batch(
    batch: Any, *, model: str, imgsz: int, device: str
) -> tuple[int, dict[str, Any]]:
    """``(batch passed to ultralytics, hardware record)`` for a resolved ``device`` ("cpu", "0", "cuda:N").

    ``batch="auto"`` sizes the batch from the free memory of the training GPU with the workbench heuristic
    (``fixed reserve + per-image cost x batch`` within 85 % of free memory, at most 64, rounded down to even);
    chosen over ultralytics' AutoBatch (``batch=-1``) because it is deterministic, known before training
    starts, and AutoBatch profiles wrong numbers under WSL2. On the CPU ``"auto"`` is a fixed 16.
    """
    source = "auto" if batch == "auto" else "param"
    hardware: dict[str, Any] = {"device": device}
    if device != "cpu":
        import torch

        try:
            index = _gpu_index(device)
            free, total = torch.cuda.mem_get_info(index)
            hardware.update(
                gpu_name=str(torch.cuda.get_device_name(index)),
                gpu_total_mb=int(total // (1024 * 1024)),
                gpu_free_mb=int(free // (1024 * 1024)),
            )
        except Exception as exc:  # noqa: BLE001 - a failed probe must not stop a job with a fixed batch
            hardware["gpu_probe_error"] = f"{type(exc).__name__}: {exc}"
    if source == "param":
        chosen = int(batch)
    elif "gpu_free_mb" not in hardware:
        # CPU, or a GPU whose free memory could not be read: the former fixed default.
        chosen = CPU_AUTO_BATCH
    else:
        free_gb = hardware["gpu_free_mb"] / 1024.0
        usable_gb = max(1.0, free_gb * _HEADROOM - _RESERVE_GB)
        mb = _mb_per_image(model) * (float(imgsz) / 1024.0) ** 2
        chosen = int((usable_gb * 1024.0) // max(1.0, mb))
        chosen = max(1, min(_MAX_AUTO_BATCH, chosen))
        if chosen > 1 and chosen % 2 == 1:
            chosen -= 1
    hardware.update(batch=chosen, batch_source=source)
    return chosen, hardware


def _hardware_message(hw: dict[str, Any]) -> str:
    if hw["device"] == "cpu":
        where = "cpu"
    elif "gpu_free_mb" not in hw:
        where = f"GPU {hw['device']} (memory not readable: {hw.get('gpu_probe_error')})"
    else:
        where = (
            f"GPU {hw['device']} {hw['gpu_name']} "
            f"({hw['gpu_free_mb']} MB free of {hw['gpu_total_mb']} MB)"
        )
    how = "auto" if hw["batch_source"] == "auto" else "set by parameter"
    return f"training on {where}, batch {hw['batch']} ({how})"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path, size: int) -> None:
    """Fetch ``url`` into ``dest``; ``OSError`` unless exactly ``size`` bytes (and Content-Length) arrived.

    http.client returns a short body without raising when the server drops the connection mid-transfer.
    """
    import urllib.request

    with (
        urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT_S) as resp,
        open(dest, "wb") as fh,
    ):  # noqa: S310 - pinned https URL
        shutil.copyfileobj(resp, fh, 1 << 20)
        header = resp.headers.get("Content-Length")
    got = dest.stat().st_size
    expected = int(header) if header and header.isdigit() else size
    if got != expected or got != size:
        raise OSError(f"incomplete download: {got} of {size} bytes")


def verify_megadetector(name: str, path: Path) -> None:
    """Raise ``ValueError`` unless ``path`` is byte-identical to the pinned MegaDetector release file."""
    spec = MEGADETECTOR_BASES[name]
    digest = _sha256_file(path)
    if digest != spec["sha256"]:
        raise ValueError(
            f"{spec['file']} at {path} does not match the published MegaDetector release "
            f"(sha256 {digest}, expected {spec['sha256']}); delete it and retry"
        )


def fetch_megadetector(
    name: str,
    *,
    local: Callable[[str], str],
    cache_dir: Path,
    emit: Optional[Callable[[str], None]] = None,
) -> Path:
    """Path to the verified MegaDetector v6 checkpoint ``name``.

    Order: ``<weights_dir>/<file>`` (via ``local``, for offline workers), then ``cache_dir/<file>``, then a
    download from Zenodo into ``cache_dir``. Every candidate is sha256-checked against the pinned digest;
    a local file that does not match is an error, a bad download is deleted.
    """
    spec = MEGADETECTOR_BASES[name]
    found = local(spec["file"])
    if found != spec["file"]:
        path = Path(found)
        verify_megadetector(name, path)
        return path
    cache_dir = Path(cache_dir)
    path = cache_dir / spec["file"]
    if path.is_file():
        verify_megadetector(name, path)
        return path
    cache_dir.mkdir(parents=True, exist_ok=True)
    if emit is not None:
        emit(
            f"downloading {spec['name']} weights ({spec['size'] / 1e6:.0f} MB) from Zenodo"
        )
    tmp = path.with_name(f".{spec['file']}.{uuid.uuid4().hex}.part")
    url = MEGADETECTOR_URL.format(file=spec["file"])
    for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
        try:
            _download(url, tmp, spec["size"])
            verify_megadetector(name, tmp)
            os.replace(tmp, path)
            return path
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            if attempt < _DOWNLOAD_ATTEMPTS:
                if emit is not None:
                    emit(
                        f"download of {spec['file']} failed ({exc}); retrying "
                        f"({attempt + 1}/{_DOWNLOAD_ATTEMPTS})"
                    )
                continue
            if isinstance(exc, ValueError):
                raise ValueError(
                    f"{spec['file']} downloaded from Zenodo does not match the published MegaDetector "
                    f"release after {_DOWNLOAD_ATTEMPTS} attempts (expected sha256 {spec['sha256']}); "
                    "retry the job later, or put the file in SPARROW_FT_WEIGHTS_DIR"
                ) from exc
            raise RuntimeError(
                f"could not download {spec['file']} from Zenodo after {_DOWNLOAD_ATTEMPTS} attempts "
                f"({exc}); on an offline worker put the file in SPARROW_FT_WEIGHTS_DIR"
            ) from exc
    raise AssertionError("unreachable")


def _weights_cache(workdir: Path) -> Path:
    """``<data_dir>/cache/weights`` for a ``<data_dir>/jobs/<id>/attempt-N/work`` workdir, else inside it."""
    work = Path(workdir).resolve()
    parents = work.parents
    if (
        work.name == "work"
        and len(parents) >= 4
        and parents[0].name.startswith("attempt-")
        and parents[2].name == "jobs"
    ):
        return parents[3] / "cache" / "weights"
    return work / "weights"


def is_end2end(head_end2end: bool, out: np.ndarray) -> bool:
    """A YOLOv10-style end-to-end head: ``(1, max_det, 6)`` rows of x1, y1, x2, y2, score, class.

    The head's own ``end2end`` flag decides; the output shape must agree with it."""
    if not head_end2end:
        return False
    if not (out.ndim == 3 and out.shape[0] == 1 and out.shape[2] == 6):
        raise RuntimeError(
            f"end-to-end detection head produced output shape {out.shape}, expected (1, N, 6)"
        )
    return True


def parity_delta_end2end(
    onnx_out: np.ndarray, torch_out: np.ndarray, imgsz: int, *, min_score: float = 0.1
) -> float:
    """Worst ONNX-vs-PyTorch mismatch between end-to-end detection sets ``(1, max_det, 6)``.

    The rows are top-k sorted by score, so near-equal scores can swap places between runtimes. Each row
    above ``min_score`` on either side is matched to the closest same-class row on the other side (box
    divided by ``imgsz``, score as is); the result is the largest such distance. A row with no same-class
    partner counts as 1.0.
    """

    def worst(a: np.ndarray, b: np.ndarray) -> float:
        out = 0.0
        for row in a[a[:, 4] > min_score]:
            same = b[b[:, 5] == row[5]]
            if not len(same):
                return 1.0
            d = np.abs(same[:, :5] - row[:5])
            d[:, :4] /= float(imgsz)
            out = max(out, float(d.max(axis=1).min()))
        return out

    a = onnx_out.astype(np.float64)[0]
    b = torch_out.astype(np.float64)[0]
    return max(worst(a, b), worst(b, a))


def parity_delta(onnx_out: np.ndarray, torch_out: np.ndarray, imgsz: int) -> float:
    """Max abs ONNX-vs-PyTorch difference on the raw head output ``(1, 4 + nc, anchors)``.

    The first four rows are box coordinates in pixels (0..imgsz), so they are divided by ``imgsz`` first:
    float32 rounding there reaches ~1e-3 px, which an absolute limit sized for [0, 1] scores would reject.
    """
    delta = np.abs(onnx_out.astype(np.float64) - torch_out.astype(np.float64))
    delta[:, :4] /= float(imgsz)
    return float(delta.max())


class YoloDetectorTrainer(Trainer):
    id = "yolo_detector"
    name = "YOLO11 object detector"
    task = "detection"
    requires: frozenset[str] = frozenset()
    supports_base_model = True

    def __init__(
        self, *, base_model_resolver: Optional[Callable[[str], str]] = None
    ) -> None:
        self._resolve_model = base_model_resolver or (lambda name: name)

    def _model_source(self, name: str, workdir: Path) -> str:
        """The resolver's answer; a bare ``.pt`` name it left alone downloads into ``workdir``, not the cwd."""
        source = self._resolve_model(name)
        if source == name and name.endswith(".pt") and Path(name).name == name:
            return str(Path(workdir).resolve() / name)
        return source

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
        # FT-31: a MegaDetector base trains at the size it was trained at unless imgsz is given.
        if (
            "imgsz" not in params
            and base_model_id is None
            and params.get("model") in MEGADETECTOR_BASES
        ):
            params["imgsz"] = MEGADETECTOR_BASES[params["model"]]["imgsz"]
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
                ok_type = isinstance(value, str)
            elif kind == "int":
                ok_type = isinstance(value, int) and not isinstance(value, bool)
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
        if base_model_id is not None:
            out["base_model_id"] = base_model_id
        return out

    # ------------------------------------------------------------------ prepare

    def prepare(self, ctx: JobContext) -> None:
        import yaml

        params = ctx.params
        if params["device"] != "auto":
            # Fail before writing anything when the job asks for a GPU this worker does not have.
            _check_cuda_device(_resolve_device(params["device"]))
        if ctx.base_model is not None:
            self._stage_base_weights(ctx)
        start_weights = None
        if ctx.base_model is None and params["model"] in MEGADETECTOR_BASES:
            workdir = Path(ctx.workdir)
            start_weights = str(
                fetch_megadetector(
                    params["model"],
                    local=self._resolve_model,
                    cache_dir=_weights_cache(workdir),
                    emit=lambda msg: ctx.emit("prepare", msg),
                )
            )
        label_index = {label: i for i, label in enumerate(ctx.labels)}
        dataset = Path(ctx.workdir) / "dataset"
        if dataset.exists():
            shutil.rmtree(dataset)
        for sub in ("images", "labels"):
            for side in ("train", "val"):
                (dataset / sub / side).mkdir(parents=True, exist_ok=True)

        summary = {
            side: {"items": 0, "boxes": 0, "groups": set()} for side in ("train", "val")
        }
        label_boxes = {side: {label: 0 for label in ctx.labels} for side in summary}
        hashes: list[int] = []
        sides: list[str] = []
        for position, item in enumerate(ctx.items):
            side = assign_split(
                item.group if item.group else item.name,
                seed=params["seed"],
                val_fraction=params["val_fraction"],
            )
            lines = []
            for ann in item.annotations:
                bbox = ann.get("bbox")
                if bbox is None:
                    continue
                try:
                    cx, cy, w, h = to_yolo_label(bbox, item.width, item.height)
                except ValueError as exc:
                    raise ValueError(f"image {item.name!r}: {exc}") from exc
                if ann.get("label") not in label_index:
                    raise ValueError(
                        f"image {item.name!r}: annotation label {ann.get('label')!r} is not one of the "
                        f"job labels {list(ctx.labels)}"
                    )
                lines.append(
                    f"{label_index[ann['label']]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                )
                label_boxes[side][ann["label"]] += 1
            stem = f"{position:06d}"
            hashes.append(
                _write_image(Path(item.image_path), dataset / "images" / side / stem)
            )
            sides.append(side)
            (dataset / "labels" / side / f"{stem}.txt").write_text(
                "".join(line + "\n" for line in lines)
            )
            summary[side]["items"] += 1
            summary[side]["boxes"] += len(lines)
            if item.group:
                summary[side]["groups"].add(item.group)
            ctx.check_cancelled()

        if summary["train"]["boxes"] + summary["val"]["boxes"] == 0:
            raise ValueError(
                "the snapshot has no bounding boxes: no label has any box to train on"
            )
        if summary["train"]["boxes"] == 0:
            raise ValueError(
                "the train split has no boxes: add more images or groups, or lower val_fraction"
            )
        if summary["val"]["boxes"] == 0:
            raise ValueError(
                "the val (validation) split has no boxes: add more images or groups, or raise val_fraction"
            )
        # mAP/P/R average over the classes present in val only, so every label must be in both splits.
        for side in ("train", "val"):
            missing = [label for label, n in label_boxes[side].items() if n == 0]
            if missing:
                raise ValueError(
                    f"label(s) {', '.join(missing)} have no boxes in the {side} split: add images or groups "
                    "for them, change val_fraction or seed, or remove the label"
                )

        data = {
            "path": str(dataset.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": dict(enumerate(ctx.labels)),
        }
        (dataset / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
        np.save(Path(ctx.workdir) / "dhash.npy", np.asarray(hashes, dtype=np.uint64))
        (Path(ctx.workdir) / "image_sides.json").write_text(json.dumps(sides))
        split = {
            side: {
                "items": s["items"],
                "boxes": s["boxes"],
                "groups": sorted(s["groups"]),
            }
            for side, s in summary.items()
        }
        (Path(ctx.workdir) / "prepare.json").write_text(
            json.dumps({"split": split, "start_weights": start_weights}, indent=2)
        )
        ctx.emit(
            "prepare",
            "YOLO dataset written",
            train_images=split["train"]["items"],
            val_images=split["val"]["items"],
            train_boxes=split["train"]["boxes"],
            val_boxes=split["val"]["boxes"],
        )

    # ------------------------------------------------------------------ train

    @staticmethod
    def _stage_base_weights(ctx: JobContext) -> None:
        """Copy the base bundle's ``weights.pt`` to ``workdir/base_weights.pt`` after re-checking it.

        Trust boundary: ultralytics fully unpickles a checkpoint when ``YOLO(path)`` loads it. The staged
        bundle was accepted at import only from a sparrow-fine-tune manifest whose sha256 values match, with
        a pickle that names nothing but torch.nn / ultralytics.nn modules; the same sha and pickle checks run
        again here, so a file swapped on disk after import is refused before ultralytics sees it.
        """
        from sparrow_finetune import base_models

        base = ctx.base_model
        with zipfile.ZipFile(Path(base["path"])) as zf:
            names = set(zf.namelist())
            manifest = (
                json.loads(zf.read("manifest.json")) if "manifest.json" in names else {}
            )
            if "weights.pt" not in names:
                raise ValueError(
                    f"base model {base['id']} has no PyTorch weights (weights.pt): it was exported by an "
                    f"older version; {base_models.RETRAIN_HINT}"
                )
            data = zf.read("weights.pt")
        expected = (manifest.get("files") or {}).get("weights.pt")
        if expected != hashlib.sha256(data).hexdigest():
            raise ValueError(
                f"base model {base['id']}: weights.pt does not match the sha256 in its manifest"
            )
        arch = manifest.get("model_arch") or base.get("model_arch") or ""
        if not base_models._same_arch(
            YoloDetectorTrainer.id, arch, ctx.params["model"]
        ):
            raise ValueError(
                f"base model {base['id']} is a {arch} model, but this job trains {ctx.params['model']}"
            )
        try:
            base_models._check_yolo_checkpoint(data)
        except base_models.BundleError as exc:
            raise ValueError(f"base model {base['id']}: {exc}") from exc
        workdir = Path(ctx.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / "base_weights.pt").write_bytes(data)

    def train(self, ctx: JobContext) -> None:
        params = ctx.params
        workdir = Path(ctx.workdir)
        device = _resolve_device(params["device"])
        if params["device"] != "auto":
            _check_cuda_device(device)

        curve: list[dict] = []
        seen: set[int] = set()

        def on_fit_epoch_end(trainer) -> None:
            epoch = int(trainer.epoch) + 1
            # final_eval fires this callback a second time for the last epoch: report each epoch once.
            if epoch not in seen:
                seen.add(epoch)
                metrics = trainer.metrics or {}
                finite, _ = _split_finite(
                    {k: metrics.get(src) for k, src in _METRIC_KEYS.items()}
                )
                point = {"epoch": epoch, **finite}
                curve.append({**point, "fitness": _finite(trainer.fitness)})
                ctx.emit("train", f"epoch {epoch}", **point)
            ctx.check_cancelled()

        batch, hardware = resolve_batch(
            params["batch"], model=params["model"], imgsz=params["imgsz"], device=device
        )
        ctx.emit(
            "train",
            _hardware_message(hardware),
            batch=batch,
            **(
                {"gpu_free_mb": hardware["gpu_free_mb"]}
                if "gpu_free_mb" in hardware
                else {}
            ),
        )
        kwargs: dict[str, Any] = {
            "data": str(workdir / "dataset" / "data.yaml"),
            "epochs": params["epochs"],
            "imgsz": params["imgsz"],
            "batch": batch,
            "seed": params["seed"],
            "device": device,
            "patience": params["patience"],
            "lr0": params["lr0"],
            "workers": params["workers"],
            "project": str(workdir / "runs"),
            "name": "train",
            "exist_ok": True,
            "plots": False,
            "deterministic": True,
            "verbose": False,
        }
        if device == "cpu":
            kwargs["amp"] = False
        source = lambda name: self._model_source(name, workdir)  # noqa: E731
        start = source(params["model"])
        if ctx.base_model is None and params["model"] in MEGADETECTOR_BASES:
            start = json.loads((workdir / "prepare.json").read_text())["start_weights"]
            # Re-checked here: ultralytics unpickles the file in full, so it must still be the pinned release.
            verify_megadetector(params["model"], Path(start))
            ctx.emit(
                "train",
                f"starting from {MEGADETECTOR_BASES[params['model']]['name']} weights",
            )
        if ctx.base_model is not None:
            # prepare() re-checked these bytes against the base bundle (see _stage_base_weights).
            start = str(workdir / "base_weights.pt")
            ctx.emit(
                "train",
                f"warm start from base model {ctx.base_model['id']} "
                f"({len(ctx.base_model.get('labels') or [])} base labels -> {len(ctx.labels)} job labels)",
            )
        with _ultralytics_sandbox(workdir, source), _restore_global_state():
            from ultralytics import YOLO

            model = YOLO(start)
            model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
            model.train(**kwargs)
        # Defensive: a cancellation raised inside a callback must never be lost.
        ctx.check_cancelled()

        trainer = getattr(model, "trainer", None)
        save_dir = (
            Path(trainer.save_dir)
            if trainer is not None
            else workdir / "runs" / "train"
        )
        weights = None
        for name in ("best.pt", "last.pt"):
            if (save_dir / "weights" / name).is_file():
                weights = save_dir / "weights" / name
                break
        if weights is None:
            raise RuntimeError(
                f"YOLO training wrote no checkpoint under {save_dir / 'weights'}"
            )

        best_epoch = 0
        best_fitness = -math.inf
        for point in curve:
            # ultralytics overwrites best.pt on fitness ties, so the last tied epoch is the saved one.
            if point["fitness"] >= best_fitness:
                best_fitness, best_epoch = point["fitness"], point["epoch"]
        (workdir / "train.json").write_text(
            json.dumps(
                {
                    "weights": str(weights),
                    "device": device,
                    "best_epoch": best_epoch,
                    "batch": batch,
                    "hardware": hardware,
                    "epochs": [
                        {k: v for k, v in p.items() if k != "fitness"} for p in curve
                    ],
                },
                indent=2,
            )
        )

    # ------------------------------------------------------------------ evaluate

    def evaluate(self, ctx: JobContext) -> dict[str, float]:
        params = ctx.params
        workdir = Path(ctx.workdir)
        trained = json.loads((workdir / "train.json").read_text())
        prepared = json.loads((workdir / "prepare.json").read_text())
        batch = trained.get("batch")
        if not isinstance(batch, int):
            batch = (
                params["batch"] if isinstance(params["batch"], int) else CPU_AUTO_BATCH
            )
        source = lambda name: self._model_source(name, workdir)  # noqa: E731
        with _ultralytics_sandbox(workdir, source), _restore_global_state():
            from ultralytics import YOLO

            results = YOLO(trained["weights"]).val(
                validator=_confusion_validator(),
                data=str(workdir / "dataset" / "data.yaml"),
                imgsz=params["imgsz"],
                batch=batch,
                device=trained["device"],
                plots=False,
                split="val",
                project=str(workdir / "runs"),
                name="val",
                exist_ok=True,
                verbose=False,
            )
        box = results.box
        split = prepared["split"]
        metrics, nan_metrics = _split_finite(
            {
                "map50": box.map50,
                "map50_95": box.map,
                "precision": box.mp,
                "recall": box.mr,
            }
        )
        metrics = {
            **metrics,
            "best_epoch": float(trained["best_epoch"]),
            "train_images": float(split["train"]["items"]),
            "val_images": float(split["val"]["items"]),
            "val_boxes": float(split["val"]["boxes"]),
            "train_batch": float(batch),
        }
        per_class, missing_classes = _per_class_ap(box, list(ctx.labels))
        metrics.update(
            {f"ap50_95.{label}": ap["ap50_95"] for label, ap in per_class.items()}
        )
        table = per_class_table(
            box, list(ctx.labels), getattr(results, "nt_per_class", None)
        )
        for row in table:
            for key in ("precision", "recall", "map50", "map50_95", "instances"):
                if row[key] is not None:
                    metrics[f"{key}.{row['label']}"] = float(row[key])
        cm = getattr(results, "confusion_matrix", None)
        confusion = confusion_report(getattr(cm, "matrix", None), list(ctx.labels))
        near_dups = None
        if (workdir / "image_sides.json").exists():  # absent in a workdir prepared before FT-38
            sides = np.asarray(json.loads((workdir / "image_sides.json").read_text()))
            near_dups = near_duplicate_report(
                ctx,
                np.load(workdir / "dhash.npy"),
                sides == "train",
                sides == "val",
                metrics,
                unit="image",
            )
        (workdir / "evaluate.json").write_text(
            json.dumps(
                {
                    "metrics": metrics,
                    "nan_metrics": nan_metrics,
                    "per_class": per_class,
                    "per_class_table": table,
                    "missing_classes": missing_classes,
                    "confusion_matrix": confusion,
                    "near_duplicates": near_dups,
                },
                indent=2,
            )
        )
        return metrics

    # ------------------------------------------------------------------ export

    @staticmethod
    def _parity_input(dataset: Path, imgsz: int) -> np.ndarray:
        """One val image, letterboxed to ``imgsz`` and normalised as ultralytics predicts: RGB, NCHW, /255."""
        import cv2
        from ultralytics.data.augment import LetterBox

        images = sorted((dataset / "images" / "val").iterdir())
        img = cv2.imread(str(images[0]))
        if img is None:
            raise RuntimeError(f"cannot read parity image {images[0]}")
        img = LetterBox(new_shape=(imgsz, imgsz), auto=False)(image=img)
        img = img[..., ::-1].transpose(2, 0, 1)
        return (np.ascontiguousarray(img, dtype=np.float32) / 255.0)[None]

    def export(self, ctx: JobContext) -> ExportResult:
        import torch

        params = ctx.params
        workdir = Path(ctx.workdir)
        trained = json.loads((workdir / "train.json").read_text())
        prepared = json.loads((workdir / "prepare.json").read_text())
        report = json.loads((workdir / "evaluate.json").read_text())
        weights = Path(trained["weights"])
        imgsz = params["imgsz"]

        source = lambda name: self._model_source(name, workdir)  # noqa: E731
        with _ultralytics_sandbox(workdir, source), _restore_global_state():
            import onnxruntime as ort
            from ultralytics import YOLO

            onnx_path = Path(
                YOLO(str(weights)).export(
                    format="onnx",
                    imgsz=imgsz,
                    simplify=False,
                    dynamic=False,
                    device="cpu",
                )
            )
        onnx_bytes = onnx_path.read_bytes()
        ctx.check_cancelled()

        x = self._parity_input(workdir / "dataset", imgsz)
        with _ultralytics_sandbox(workdir, source), _restore_global_state():
            from ultralytics import YOLO

            torch_model = YOLO(str(weights)).model.float().eval().to("cpu")
        with torch.no_grad():
            out = torch_model(torch.from_numpy(x))
        if isinstance(out, (list, tuple)):
            out = out[0]
        torch_out = out.detach().cpu().numpy().astype(np.float32)
        session = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
        onnx_out = np.asarray(
            session.run(None, {session.get_inputs()[0].name: x})[0], dtype=np.float32
        )
        if onnx_out.shape != torch_out.shape:
            raise RuntimeError(
                f"ONNX export parity gate failed: output shape {onnx_out.shape} != PyTorch {torch_out.shape}"
            )
        end2end = is_end2end(
            bool(getattr(torch_model.model[-1], "end2end", False)), torch_out
        )
        max_abs_delta = (
            parity_delta_end2end(onnx_out, torch_out, imgsz)
            if end2end
            else parity_delta(onnx_out, torch_out, imgsz)
        )
        if not max_abs_delta <= PARITY_MAX_ABS_DELTA:
            raise RuntimeError(
                f"ONNX export parity gate failed: max_abs_delta={max_abs_delta:.3g} "
                f"(limit {PARITY_MAX_ABS_DELTA})"
            )

        metrics = report["metrics"]
        nan_metrics = report.get("nan_metrics", [])
        per_class = report.get("per_class_table", [])
        hardware = trained.get("hardware")
        base_model_id = params.get("base_model_id")
        weights_bytes = weights.read_bytes()
        card = {
            "pipeline": self.id,
            "task": self.task,
            "labels": list(ctx.labels),
            "base_model": params["model"],
            "params": dict(ctx.params),
            "metrics": metrics,
            "per_class": per_class,
            "confusion_matrix": report.get("confusion_matrix"),
            "near_duplicates": report.get("near_duplicates"),
            "nan_metrics": nan_metrics,
            "split": prepared["split"],
            "best_epoch": trained["best_epoch"],
            "epochs": trained["epochs"],
            "device": trained["device"],
            "hardware": hardware,
        }
        card_bytes = json.dumps(card, indent=2).encode("utf-8")
        files = {
            "model.onnx": hashlib.sha256(onnx_bytes).hexdigest(),
            "model_card.json": hashlib.sha256(card_bytes).hexdigest(),
            # The ultralytics checkpoint, so an imported copy of this model can warm-start a new job.
            "weights.pt": hashlib.sha256(weights_bytes).hexdigest(),
        }
        parity = {
            "max_abs_delta": max_abs_delta,
            "limit": PARITY_MAX_ABS_DELTA,
            "box_rows": "divided by imgsz",
        }
        if end2end:
            parity["matching"] = "closest same-class detection, both directions"
        manifest = {
            "pipeline": self.id,
            "task": self.task,
            "labels": list(ctx.labels),
            "input": {
                "name": session.get_inputs()[0].name,
                "size": imgsz,
                "layout": "NCHW",
                "scale": "1/255",
                "letterbox": True,
            },
            "base_model": params["model"],
            "model_arch": params["model"],
            # "end2end": (1, max_det, 6) rows of x1, y1, x2, y2, score, class (YOLOv10 / MegaDetector v6);
            # "raw": (1, 4 + nc, anchors) that needs NMS (YOLO11, YOLOv8).
            "output": "end2end" if end2end else "raw",
            "metrics": metrics,
            "per_class": per_class,
            "nan_metrics": nan_metrics,
            "parity": parity,
            "files": files,
        }
        if base_model_id is not None:
            manifest["base_model_id"] = base_model_id

        path = workdir / "yolo_detector_bundle.zip"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("model.onnx", onnx_bytes)
            zf.writestr("model_card.json", card_bytes)
            zf.writestr("weights.pt", weights_bytes)
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        ctx.emit("export", "bundle written", max_abs_delta=max_abs_delta)
        metadata: dict[str, Any] = {
            "labels": list(ctx.labels),
            "base_model": params["model"],
            "per_class": per_class,
            "hardware": hardware,
        }
        if base_model_id is not None:
            metadata["base_model_id"] = base_model_id
        return ExportResult(path=path, format="onnx-bundle", metadata=metadata)


__all__ = [
    "CPU_AUTO_BATCH",
    "YoloDetectorTrainer",
    "confusion_report",
    "per_class_table",
    "resolve_batch",
    "to_yolo_label",
]
