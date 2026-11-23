"""Base models: models pulled back from Sparrow Studio for warm start.

A base model is a model this service trained and pushed to a Studio project
(``connectors/studio.py``), imported back from that project. Its artifact is the
``<pipeline>_bundle.zip`` the trainer exported, and its ``weights.pt`` member is the starting
point of a new job of the same pipeline with ``base_model_id``:

- ``conventional_classifier``: the timm network's ``state_dict`` (loaded with ``weights_only=True``);
- ``dinov3_linear``: the linear head (scaler mean/scale, coef, intercept) as a ``state_dict``; its
  ``model_arch`` is the embedder id, and the trainer also requires the same embedder model hash;
- ``yolo_detector``: the ultralytics checkpoint (``best.pt``). An ultralytics checkpoint pickles the
  network classes, so ``YOLO(path)`` unpickles it with ``weights_only=False``. Trust boundary: a
  checkpoint is accepted only from a bundle whose manifest ``source`` is this service, after the outer
  and inner sha256 checks, and after a static scan of the pickle's globals (only ``torch.nn`` /
  ``ultralytics.nn`` module classes) plus a ``weights_only=True`` load restricted to those classes.
  The trainer re-checks the sha256 of the staged copy before ``YOLO`` loads it in full.

A base model can also be a local package (``origin='local'``), uploaded with
``POST /v1/base-models/local`` and built by ``sparrow-uploader prepare-finetune``: a ZIP_STORED
archive (``schema=sparrow-finetune/base-v1``) with exactly ``manifest.json`` and
``weights.safetensors`` holding a timm ``resnet50`` or ``efficientnet_b0`` state dict (the
manifest's ``model_arch``) for ``conventional_classifier``.
The weights are parsed as safetensors only (never unpickled) and must match the timm network's
tensor names, shapes and dtypes exactly, with one head row per label. A local row has no Studio ids.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Optional

from .db import connect
from .storage import LocalObjectStore

MANIFEST_SCHEMA = "sparrow-finetune/model-v1"
SOURCE = "sparrow-fine-tune"
# Backwards-compatible alias (the conventional pipeline was the only warm-start pipeline before FT-24).
WARM_START_PIPELINE = "conventional_classifier"
WARM_START_PIPELINES = frozenset(
    {"conventional_classifier", "yolo_detector", "dinov3_linear"}
)
ARTIFACT_MEMBER = "model.zip"
BUNDLE_MEMBERS = frozenset({ARTIFACT_MEMBER, "engine.onnx", "manifest.json"})
MAX_BUNDLE_BYTES = 1 << 30
MAX_MEMBER_BYTES = 1 << 30
MAX_MANIFEST_BYTES = 1 << 20
ARTIFACT_NAMES = {
    pipeline: f"{pipeline}_bundle.zip" for pipeline in WARM_START_PIPELINES
}
ARTIFACT_NAME = ARTIFACT_NAMES[WARM_START_PIPELINE]
RETRAIN_HINT = "re-train with this version to enable warm start"
COLUMNS = (
    "id, origin, studio_project_id, studio_import_id, source_model_id, pipeline, model_arch, "
    "labels, sha256, size_bytes, created_at"
)

# Local packages (``sparrow-uploader prepare-finetune``). The limits match the uploader's, so any
# package it writes is accepted; each is checked before the allocation it bounds.
LOCAL_SCHEMA = "sparrow-finetune/base-v1"
LOCAL_PIPELINE = "conventional_classifier"
LOCAL_ARCH = "resnet50"  # the default architecture of a local package
# timm architectures a local package may declare; the tensors must match the declared one exactly.
LOCAL_ARCHS = ("resnet50", "efficientnet_b0")
LOCAL_MANIFEST = "manifest.json"
LOCAL_WEIGHTS = "weights.safetensors"
LOCAL_ARTIFACT = "local_base.zip"
LOCAL_MANIFEST_KEYS = frozenset({"schema", "pipeline", "model_arch", "labels", "files"})
MAX_LOCAL_WEIGHTS_BYTES = 1 << 30
MAX_LOCAL_MANIFEST_BYTES = 128 << 20
MAX_LOCAL_PACKAGE_BYTES = MAX_LOCAL_WEIGHTS_BYTES + MAX_LOCAL_MANIFEST_BYTES + (1 << 20)
MAX_SAFETENSORS_HEADER_BYTES = 16 << 20
MAX_LOCAL_LABELS = 100_000
MAX_LABEL_CHARS = 256
# safetensors dtype -> element size in bytes.
_ST_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E5M2": 1,
    "F8_E4M3": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


class BundleError(ValueError):
    """The Studio bundle is malformed or tampered with (Studio-side problem: 502)."""


class NotImportable(ValueError):
    """The bundle is well formed but cannot be a base model (client-side problem: 422)."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def public(row: dict) -> dict:
    """A ``base_models`` row as the API returns it (no object-store key).

    Every stored base model passed the full import check (validated training weights for a
    warm-start pipeline), so each one is tagged as fine-tunable."""
    out = {k: v for k, v in row.items() if k != "object_key"}
    if out.get("created_at") is not None and hasattr(out["created_at"], "isoformat"):
        out["created_at"] = out["created_at"].isoformat()
    return tag(out, True)


# --------------------------------------------------------------------------- fine-tune tags

FINETUNE_YES = "Can be fine-tuned"
FINETUNE_NO = "Fine-tune not supported"


def tag(row: dict, ok: bool, reason: Optional[str] = None) -> dict:
    """Set ``finetune_tag`` / ``finetune_reason`` on ``row`` (a NO always carries a reason)."""
    row["finetune_tag"] = FINETUNE_YES if ok else FINETUNE_NO
    row["finetune_reason"] = None if ok else (reason or "no validated training weights")
    return row


# --------------------------------------------------------------------------- import


def _safe_member(info: tarfile.TarInfo) -> str:
    name = info.name
    path = PurePosixPath(name)
    if (
        not info.isfile()
        or path.is_absolute()
        or name.startswith("\\")
        or ".." in path.parts
    ):
        raise BundleError(f"bundle has an unsafe member {name!r}")
    if name not in BUNDLE_MEMBERS:
        raise BundleError(f"bundle has an unexpected member {name!r}")
    cap = MAX_MANIFEST_BYTES if name == "manifest.json" else MAX_MEMBER_BYTES
    if info.size > cap:
        raise BundleError(f"bundle member {name!r} is larger than {cap} bytes")
    return name


def _read_tar(bundle: bytes) -> dict[str, bytes]:
    if len(bundle) > MAX_BUNDLE_BYTES:
        raise BundleError(f"bundle is larger than {MAX_BUNDLE_BYTES} bytes")
    try:
        tar = tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz")
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise BundleError(f"bundle is not a tar.gz archive: {exc}") from exc
    members: dict[str, bytes] = {}
    try:
        with tar:
            for info in tar:
                name = _safe_member(info)
                if name in members:
                    raise BundleError(f"bundle has a duplicate member {name!r}")
                fh = tar.extractfile(info)
                if fh is None:  # pragma: no cover - isfile() was checked
                    raise BundleError(f"bundle has an unsafe member {name!r}")
                members[name] = fh.read()
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise BundleError(f"bundle is not a readable tar.gz archive: {exc}") from exc
    return members


def _json_member(data: bytes, what: str) -> dict:
    try:
        value = json.loads(data)
    except (ValueError, UnicodeDecodeError) as exc:
        raise BundleError(f"{what} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise BundleError(f"{what} is not a JSON object")
    return value


def inspect_bundle(bundle: bytes, header_sha256: Optional[str]) -> dict:
    """Validate a Studio bundle this service pushed and return the base-model facts.

    Returns ``{artifact, sha256, source_model_id, pipeline, model_arch, labels}``.
    Raises ``BundleError`` (malformed / tampered: Studio's fault) or ``NotImportable``
    (a valid bundle that cannot be a warm-start base).
    """
    members = _read_tar(bundle)
    if "manifest.json" not in members:
        raise BundleError("bundle has no manifest.json")
    if ARTIFACT_MEMBER not in members:
        raise BundleError(f"bundle has no {ARTIFACT_MEMBER}")
    manifest = _json_member(members["manifest.json"], "bundle manifest.json")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise NotImportable(
            f"bundle manifest schema is {manifest.get('schema')!r}, "
            f"expected {MANIFEST_SCHEMA!r}"
        )
    if manifest.get("source") != SOURCE:
        raise NotImportable(
            f"only models trained by {SOURCE} can be imported, "
            f"this one comes from {manifest.get('source')!r}"
        )
    pipeline = manifest.get("pipeline")
    if pipeline not in WARM_START_PIPELINES:
        raise NotImportable(
            f"only {' and '.join(sorted(WARM_START_PIPELINES))} models can be used as a base "
            f"model, this one is {pipeline!r}"
        )
    artifact = members[ARTIFACT_MEMBER]
    digest = _sha(artifact)
    if manifest.get("sha256") != digest:
        raise BundleError(
            f"artifact sha256 {digest} does not match the manifest's "
            f"{manifest.get('sha256')!r}"
        )
    if header_sha256 is not None and header_sha256.strip().lower() != digest:
        raise BundleError(
            f"artifact sha256 {digest} does not match Studio's X-Artifact-Sha256 "
            f"{header_sha256!r}"
        )
    arch, labels = _inspect_artifact(artifact, pipeline)
    return {
        "artifact": artifact,
        "sha256": digest,
        "source_model_id": manifest.get("model_id"),
        "pipeline": pipeline,
        "model_arch": arch,
        "labels": labels,
    }


def _inspect_artifact(artifact: bytes, pipeline: str) -> tuple[str, list[str]]:
    try:
        zf = zipfile.ZipFile(io.BytesIO(artifact))
    except (zipfile.BadZipFile, OSError) as exc:
        raise BundleError(f"{ARTIFACT_MEMBER} is not a zip archive") from exc
    with zf:
        names = set(zf.namelist())
        if "manifest.json" not in names:
            raise BundleError(f"{ARTIFACT_MEMBER} has no manifest.json")
        inner = _json_member(zf.read("manifest.json"), f"{ARTIFACT_MEMBER} manifest")
        if "weights.pt" not in names:
            raise NotImportable(
                f"this model has no weights.pt (it was trained by an older version); "
                f"{RETRAIN_HINT}"
            )
        info = zf.getinfo("weights.pt")
        if info.file_size > MAX_MEMBER_BYTES:
            raise BundleError("weights.pt is too large")
        weights = zf.read("weights.pt")
    expected = (inner.get("files") or {}).get("weights.pt")
    if expected != _sha(weights):
        raise BundleError(
            "weights.pt sha256 does not match the bundle manifest (tampered or corrupt)"
        )
    arch = inner.get("model_arch") or inner.get("model")
    if not isinstance(arch, str) or not arch:
        raise NotImportable(f"the bundle manifest names no model_arch; {RETRAIN_HINT}")
    labels = inner.get("labels")
    if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
        raise BundleError("the bundle manifest has no label list")
    if pipeline == "yolo_detector":
        _check_yolo_checkpoint(weights)
    else:
        _check_state_dict(weights)
    return arch, labels


def _check_state_dict(weights: bytes) -> None:
    import torch

    try:
        state = torch.load(io.BytesIO(weights), map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - any unpickling failure is a bad bundle
        raise BundleError(f"weights.pt is not a loadable state_dict: {exc}") from exc
    if not isinstance(state, dict) or not all(
        isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()
    ):
        raise BundleError("weights.pt is not a state_dict of tensors")


# Module prefixes whose nn.Module subclasses an ultralytics checkpoint may pickle.
_YOLO_GLOBAL_PREFIXES = ("torch.nn.modules.", "ultralytics.nn.")


def _yolo_checkpoint_globals(weights: bytes) -> list:
    """The classes the checkpoint pickles beyond torch's defaults; refused unless all are nn modules.

    ``get_unsafe_globals_in_checkpoint`` reads the pickle opcodes without executing them.
    """
    import importlib

    import torch

    try:
        names = torch.serialization.get_unsafe_globals_in_checkpoint(
            io.BytesIO(weights)
        )
    except Exception as exc:  # noqa: BLE001 - not a torch zip checkpoint
        raise BundleError(f"weights.pt is not a torch checkpoint: {exc}") from exc
    classes = []
    for name in names:
        module, _, attr = name.rpartition(".")
        if not (module + ".").startswith(_YOLO_GLOBAL_PREFIXES):
            raise BundleError(
                f"weights.pt pickles {name!r}, which is not a torch.nn / ultralytics.nn module"
            )
        try:
            obj = getattr(importlib.import_module(module), attr)
        except (ImportError, AttributeError) as exc:
            raise BundleError(f"weights.pt pickles an unknown class {name!r}") from exc
        if not (isinstance(obj, type) and issubclass(obj, torch.nn.Module)):
            raise BundleError(
                f"weights.pt pickles {name!r}, which is not an nn.Module class"
            )
        classes.append(obj)
    return classes


def _check_yolo_checkpoint(weights: bytes) -> None:
    """``weights.pt`` must be an ultralytics checkpoint: ``{"model": <nn.Module>, ...}``.

    Loaded with ``weights_only=True`` and only the scanned nn.Module classes allow-listed, so no
    arbitrary callable can run here (see the module docstring for the trust boundary).
    """
    import torch

    classes = _yolo_checkpoint_globals(weights)
    try:
        with torch.serialization.safe_globals(classes):
            ckpt = torch.load(
                io.BytesIO(weights), map_location="cpu", weights_only=True
            )
    except Exception as exc:  # noqa: BLE001 - any unpickling failure is a bad bundle
        raise BundleError(
            f"weights.pt is not a loadable ultralytics checkpoint: {exc}"
        ) from exc
    model = ckpt.get("model") if isinstance(ckpt, dict) else None
    if not isinstance(model, torch.nn.Module):
        raise BundleError(
            "weights.pt is not an ultralytics checkpoint (no model network)"
        )


# --------------------------------------------------------------------------- local packages

_TORCH_TO_ST = {
    "torch.float64": "F64",
    "torch.float32": "F32",
    "torch.float16": "F16",
    "torch.bfloat16": "BF16",
    "torch.int64": "I64",
    "torch.int32": "I32",
    "torch.int16": "I16",
    "torch.int8": "I8",
    "torch.uint8": "U8",
    "torch.bool": "BOOL",
}


def _few(items, limit: int = 8) -> str:
    items = sorted(items)
    more = f" and {len(items) - limit} more" if len(items) > limit else ""
    return ", ".join(items[:limit]) + more


def _local_members(data: bytes) -> tuple[bytes, bytes]:
    """``(manifest, weights)`` bytes of a package: exactly the two members, stored, bounded."""
    if not data:
        raise NotImportable(
            "the package is empty; upload the ZIP written by sparrow-uploader prepare-finetune"
        )
    if len(data) > MAX_LOCAL_PACKAGE_BYTES:
        raise NotImportable(
            f"the package is larger than {MAX_LOCAL_PACKAGE_BYTES} bytes"
        )
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except (
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
        OSError,
        EOFError,
        ValueError,
    ) as exc:
        raise NotImportable(f"the package is not a ZIP archive ({exc})") from exc
    with zf:
        infos = zf.infolist()
        names = [info.filename for info in infos]
        allowed = {LOCAL_MANIFEST, LOCAL_WEIGHTS}
        stray = {n for n in names if n not in allowed}
        if stray:
            raise NotImportable(
                f"the package holds {_few(repr(n)[:80] for n in stray)}; a package holds exactly "
                f"{LOCAL_MANIFEST} and {LOCAL_WEIGHTS}"
            )
        if len(names) != len(set(names)):
            raise NotImportable(
                "the package repeats a member; each member must appear once"
            )
        missing = allowed - set(names)
        if missing:
            raise NotImportable(f"the package has no {_few(missing)}")
        by_name = {info.filename: info for info in infos}
        caps = {
            LOCAL_MANIFEST: MAX_LOCAL_MANIFEST_BYTES,
            LOCAL_WEIGHTS: MAX_LOCAL_WEIGHTS_BYTES,
        }
        for name, info in by_name.items():
            if info.flag_bits & 0x1:
                raise NotImportable(f"package member {name} is encrypted")
            if (
                info.compress_type != zipfile.ZIP_STORED
                or info.compress_size != info.file_size
            ):
                raise NotImportable(
                    f"package member {name} is compressed; a package is stored without compression "
                    "(ZIP_STORED), as sparrow-uploader prepare-finetune writes it"
                )
            if info.file_size > caps[name]:
                raise NotImportable(
                    f"package member {name} is larger than {caps[name]} bytes"
                )
        if sum(info.file_size for info in infos) > len(data):
            raise NotImportable("the package member sizes do not fit the archive")
        try:
            manifest = zf.read(by_name[LOCAL_MANIFEST])
            weights = zf.read(by_name[LOCAL_WEIGHTS])
        except (
            zipfile.BadZipFile,
            OSError,
            EOFError,
            RuntimeError,
            NotImplementedError,
            ValueError,
        ) as exc:
            raise NotImportable(f"the package is damaged ({exc})") from exc
    return manifest, weights


def _local_labels(labels) -> list[str]:
    if not isinstance(labels, list) or not labels:
        raise NotImportable(
            "manifest labels must be a non-empty list of strings, one per classifier head row"
        )
    if len(labels) > MAX_LOCAL_LABELS:
        raise NotImportable(
            f"manifest lists {len(labels)} labels; the limit is {MAX_LOCAL_LABELS}"
        )
    seen: set[str] = set()
    for n, label in enumerate(labels, 1):
        if not isinstance(label, str):
            raise NotImportable(
                f"manifest label {n} is a {type(label).__name__}, not a string"
            )
        if not label.strip():
            raise NotImportable(f"manifest label {n} is blank")
        if label != label.strip():
            raise NotImportable(
                f"manifest label {n} has leading or trailing whitespace"
            )
        if len(label) > MAX_LABEL_CHARS:
            raise NotImportable(
                f"manifest label {n} is longer than {MAX_LABEL_CHARS} characters"
            )
        if any(ord(c) < 32 or ord(c) == 127 for c in label):
            raise NotImportable(f"manifest label {n} holds a control character")
        if label in seen:
            raise NotImportable(
                f"manifest repeats label {label!r}; labels must be distinct"
            )
        seen.add(label)
    return list(labels)


def _local_manifest(raw: bytes) -> dict:
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise NotImportable(f"{LOCAL_MANIFEST} is not UTF-8 JSON ({exc})") from exc
    if not isinstance(manifest, dict):
        raise NotImportable(f"{LOCAL_MANIFEST} is not a JSON object")
    if manifest.get("schema") != LOCAL_SCHEMA:
        raise NotImportable(
            f"{LOCAL_MANIFEST} schema is {str(manifest.get('schema'))[:80]!r}, expected {LOCAL_SCHEMA!r}"
        )
    keys = set(manifest)
    if keys != LOCAL_MANIFEST_KEYS:
        parts = []
        if LOCAL_MANIFEST_KEYS - keys:
            parts.append(f"missing {_few(LOCAL_MANIFEST_KEYS - keys)}")
        if keys - LOCAL_MANIFEST_KEYS:
            parts.append(
                f"unexpected {_few(repr(k)[:80] for k in keys - LOCAL_MANIFEST_KEYS)}"
            )
        raise NotImportable(f"{LOCAL_MANIFEST} fields are wrong: " + "; ".join(parts))
    if manifest["pipeline"] != LOCAL_PIPELINE:
        raise NotImportable(
            f"a local package must be a {LOCAL_PIPELINE} model, not "
            f"{str(manifest['pipeline'])[:80]!r}"
        )
    if manifest["model_arch"] not in LOCAL_ARCHS:
        raise NotImportable(
            f"local packages must declare model_arch {' or '.join(LOCAL_ARCHS)}, "
            f"not {str(manifest['model_arch'])[:80]!r}"
        )
    files = manifest["files"]
    digest = files.get(LOCAL_WEIGHTS) if isinstance(files, dict) else None
    if (
        not isinstance(files, dict)
        or set(files) != {LOCAL_WEIGHTS}
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise NotImportable(
            f"{LOCAL_MANIFEST} files must be {{{LOCAL_WEIGHTS!r}: <lowercase hex sha256>}}"
        )
    manifest["labels"] = _local_labels(manifest["labels"])
    return manifest


def _safetensors_header(data: bytes) -> dict[str, dict]:
    """The tensor entries of a safetensors file, framing checked without any loader.

    A pickle or a torch checkpoint fails on its first bytes, so it is never deserialised."""
    hint = "(pickle and .pt/.pth checkpoints are not accepted; save the state dict with safetensors)"
    if len(data) < 8:
        raise NotImportable(
            f"{LOCAL_WEIGHTS} is not safetensors: shorter than its header {hint}"
        )
    n = int.from_bytes(data[:8], "little")
    if n < 2 or n > MAX_SAFETENSORS_HEADER_BYTES or 8 + n > len(data):
        raise NotImportable(
            f"{LOCAL_WEIGHTS} is not safetensors: header length {n} does not fit a "
            f"{len(data)}-byte file {hint}"
        )
    try:
        header = json.loads(data[8 : 8 + n].decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise NotImportable(
            f"{LOCAL_WEIGHTS} is not safetensors: header is not JSON {hint}"
        ) from exc
    if not isinstance(header, dict):
        raise NotImportable(
            f"{LOCAL_WEIGHTS} is not safetensors: header is not a JSON object"
        )
    meta = header.get("__metadata__", {})
    if not isinstance(meta, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in meta.items()
    ):
        raise NotImportable(
            f"{LOCAL_WEIGHTS} is not valid safetensors: bad __metadata__"
        )
    buffer = len(data) - 8 - n
    tensors: dict[str, dict] = {}
    spans = []
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        ok = (
            isinstance(entry, dict)
            and set(entry) == {"dtype", "shape", "data_offsets"}
            and entry["dtype"] in _ST_DTYPE_BYTES
            and isinstance(entry["shape"], list)
            and len(entry["shape"]) <= 8
            and all(
                isinstance(d, int) and not isinstance(d, bool) and d >= 0
                for d in entry["shape"]
            )
            and isinstance(entry["data_offsets"], list)
            and len(entry["data_offsets"]) == 2
            and all(
                isinstance(o, int) and not isinstance(o, bool)
                for o in entry["data_offsets"]
            )
        )
        if not ok:
            raise NotImportable(
                f"{LOCAL_WEIGHTS} is not valid safetensors: bad entry {name[:80]!r}"
            )
        begin, end = entry["data_offsets"]
        size = math.prod(entry["shape"]) * _ST_DTYPE_BYTES[entry["dtype"]]
        if not 0 <= begin <= end <= buffer or end - begin != size:
            raise NotImportable(
                f"{LOCAL_WEIGHTS} is not valid safetensors: tensor {name[:80]!r} lies outside the data"
            )
        spans.append((begin, end))
        tensors[name] = entry
    pos = 0
    for begin, end in sorted(spans):
        if begin != pos:
            raise NotImportable(
                f"{LOCAL_WEIGHTS} is not valid safetensors: tensor data overlaps or has gaps"
            )
        pos = end
    if pos != buffer:
        raise NotImportable(
            f"{LOCAL_WEIGHTS} is not valid safetensors: bytes after the tensor data"
        )
    if not tensors:
        raise NotImportable(f"{LOCAL_WEIGHTS} holds no tensors")
    return tensors


def _check_timm_state(header: dict[str, dict], arch: str, n_labels: int) -> None:
    """Names, shapes and dtypes exactly as timm ``arch`` with ``n_labels`` head rows.

    The reference network is built on the meta device: no memory for its weights."""
    import timm
    import torch

    with torch.device("meta"):
        model = timm.create_model(arch, pretrained=False, num_classes=n_labels)
    classifier = model.get_classifier()
    head_name = next(n for n, m in model.named_modules() if m is classifier)
    head_key = f"{head_name}.weight"
    ref = {
        k: (list(v.shape), _TORCH_TO_ST.get(str(v.dtype), str(v.dtype)))
        for k, v in model.state_dict().items()
    }
    missing, extra = set(ref) - set(header), set(header) - set(ref)
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing tensors {_few(missing)}")
        if extra:
            parts.append(f"unexpected tensors {_few(n[:80] for n in extra)}")
        raise NotImportable(
            f"the weights are not a timm {arch} state dict: " + "; ".join(parts)
        )
    head = header[head_key]["shape"]
    if len(head) == 2 and head[1] == ref[head_key][0][1] and head[0] != n_labels:
        raise NotImportable(
            f"the classifier head ({head_name}) has {head[0]} outputs but the manifest lists {n_labels} "
            "labels; there must be one label per head row, in head order"
        )
    shapes = [
        f"{k} {tuple(header[k]['shape'])} (expected {tuple(shape)})"
        for k, (shape, _dt) in ref.items()
        if header[k]["shape"] != shape
    ]
    if shapes:
        raise NotImportable(
            f"the weights do not have timm {arch} shapes: "
            + "; ".join(sorted(shapes)[:8])
        )
    dtypes = [
        f"{k} {header[k]['dtype']} (expected {dt})"
        for k, (_shape, dt) in ref.items()
        if header[k]["dtype"] != dt
    ]
    if dtypes:
        raise NotImportable(
            f"the weights do not have timm {arch} dtypes: "
            + "; ".join(sorted(dtypes)[:8])
        )


def load_local_package(data: bytes) -> tuple[dict, dict]:
    """``(facts, state)`` of a local package, or ``NotImportable`` for any defect.

    ``state`` is the timm state dict of the declared architecture (``torch`` tensors), loaded
    with safetensors only."""
    manifest_raw, weights = _local_members(data)
    manifest = _local_manifest(manifest_raw)
    expected = manifest["files"][LOCAL_WEIGHTS]
    actual = _sha(weights)
    if actual != expected:
        raise NotImportable(
            f"{LOCAL_WEIGHTS} does not match the sha256 in {LOCAL_MANIFEST} "
            f"(file {actual}, manifest {expected})"
        )
    header = _safetensors_header(weights)
    _check_timm_state(header, manifest["model_arch"], len(manifest["labels"]))
    import torch
    from safetensors.torch import load as st_load

    try:
        state = st_load(weights)
    except Exception as exc:  # noqa: BLE001 - the safetensors loader raises its own error types
        raise NotImportable(
            f"{LOCAL_WEIGHTS} is not valid safetensors ({exc})"
        ) from exc
    if set(state) != set(header):
        raise NotImportable(f"{LOCAL_WEIGHTS} tensors differ from its header")
    bad = [
        k
        for k, v in state.items()
        if v.is_floating_point() and not bool(torch.isfinite(v).all())
    ]
    if bad:
        raise NotImportable(f"the weights hold NaN or infinite values in {_few(bad)}")
    facts = {
        "pipeline": LOCAL_PIPELINE,
        "model_arch": manifest["model_arch"],
        "labels": manifest["labels"],
        "sha256": _sha(data),
        "weights_sha256": actual,
        "artifact": data,
    }
    return facts, state


def inspect_local_package(data: bytes) -> dict:
    """Facts of a valid local package (``NotImportable`` otherwise); the tensors are dropped."""
    facts, _state = load_local_package(data)
    return facts


def find_local(database_url: str, sha256: str) -> Optional[dict]:
    with connect(database_url) as conn:
        row = conn.execute(
            f"SELECT {COLUMNS} FROM base_models WHERE origin = 'local' AND sha256 = %s",
            (sha256,),
        ).fetchone()
    return public(row) if row else None


def store_local(settings, facts: dict) -> tuple[dict, bool]:
    """Save a checked local package and its row; return ``(row, created)`` (idempotent by sha256)."""
    existing = find_local(settings.database_url, facts["sha256"])
    if existing is not None:
        return existing, False
    key = f"base_models/{facts['sha256']}/{LOCAL_ARTIFACT}"
    objects = LocalObjectStore(settings.objects_dir)
    objects.put_bytes(key, facts["artifact"])
    try:
        with connect(settings.database_url) as conn:
            row = conn.execute(
                f"""
                INSERT INTO base_models (origin, pipeline, model_arch, labels, sha256,
                    object_key, size_bytes)
                VALUES ('local', %s, %s, %s, %s, %s, %s)
                ON CONFLICT (sha256) WHERE origin = 'local' DO NOTHING
                RETURNING {COLUMNS}
                """,
                (
                    facts["pipeline"],
                    facts["model_arch"],
                    json.dumps(facts["labels"]),
                    facts["sha256"],
                    key,
                    len(facts["artifact"]),
                ),
            ).fetchone()
    except Exception:
        if find_local(settings.database_url, facts["sha256"]) is None:
            objects.delete(key)
        raise
    if row is not None:
        return public(row), True
    return find_local(settings.database_url, facts["sha256"]), False


def find(database_url: str, project: str, studio_id: str) -> Optional[dict]:
    with connect(database_url) as conn:
        row = conn.execute(
            f"SELECT {COLUMNS} FROM base_models "
            "WHERE studio_project_id = %s AND studio_import_id = %s",
            (project, studio_id),
        ).fetchone()
    return public(row) if row else None


def store(settings, project: str, studio_id: str, facts: dict) -> tuple[dict, bool]:
    """Save the artifact and the row; return ``(row, created)`` (idempotent)."""
    key = f"base_models/{facts['sha256']}/{ARTIFACT_NAMES[facts['pipeline']]}"
    LocalObjectStore(settings.objects_dir).put_bytes(key, facts["artifact"])
    source = facts["source_model_id"]
    with connect(settings.database_url) as conn:
        row = conn.execute(
            f"""
            INSERT INTO base_models (studio_project_id, studio_import_id, source_model_id,
                pipeline, model_arch, labels, sha256, object_key, size_bytes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (studio_project_id, studio_import_id) DO NOTHING
            RETURNING {COLUMNS}
            """,
            (
                project,
                studio_id,
                None if source is None else str(source),
                facts["pipeline"],
                facts["model_arch"],
                json.dumps(facts["labels"]),
                facts["sha256"],
                key,
                len(facts["artifact"]),
            ),
        ).fetchone()
    if row is not None:
        return public(row), True
    return find(settings.database_url, project, studio_id), False


def list_all(database_url: str) -> list[dict]:
    with connect(database_url) as conn:
        rows = conn.execute(f"SELECT {COLUMNS} FROM base_models ORDER BY id").fetchall()
    return [public(r) for r in rows]


# --------------------------------------------------------------------------- jobs


def _row(database_url: str, base_model_id: int) -> Optional[dict]:
    with connect(database_url) as conn:
        return conn.execute(
            f"SELECT {COLUMNS}, object_key FROM base_models WHERE id = %s",
            (base_model_id,),
        ).fetchone()


def check_base_model(
    database_url: str, base_model_id: int, pipeline: str, model: Optional[str]
) -> dict:
    """Raise ``ValueError`` unless the base model exists and fits the job."""
    row = _row(database_url, base_model_id)
    if row is None:
        raise ValueError(f"base model {base_model_id} not found")
    if row["pipeline"] != pipeline:
        raise ValueError(
            f"base model {base_model_id} is a {row['pipeline']} model, "
            f"it cannot start a {pipeline} job"
        )
    if model is not None and not _same_arch(row["pipeline"], row["model_arch"], model):
        raise ValueError(
            f"base model {base_model_id} is a {row['model_arch']} network but the job "
            f"trains {model}; pick model={row['model_arch']!r} or another base model"
        )
    return row


def _same_arch(pipeline: str, base_arch: str, model: str) -> bool:
    """YOLO names differ only by source format (``yolo11n.pt`` vs ``yolo11n.yaml``): compare stems."""
    if pipeline == "yolo_detector":
        return (
            PurePosixPath(base_arch).stem.lower() == PurePosixPath(model).stem.lower()
        )
    return base_arch == model


def stage(settings, base_model_id: int, workdir: Path) -> dict:
    """Copy the base artifact into the job workdir; return ``JobContext.base_model``."""
    row = _row(settings.database_url, base_model_id)
    if row is None:
        raise ValueError(f"base model {base_model_id} not found")
    if row["origin"] == "local":
        name = LOCAL_ARTIFACT
    else:
        name = ARTIFACT_NAMES.get(row["pipeline"], f"{row['pipeline']}_bundle.zip")
    dest = Path(workdir) / "base_model" / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    LocalObjectStore(settings.objects_dir).copy_to(row["object_key"], dest)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    if digest != row["sha256"]:
        raise ValueError(
            f"base model {base_model_id} artifact is corrupt (sha256 {digest}, "
            f"expected {row['sha256']})"
        )
    return {
        "id": row["id"],
        "origin": row["origin"],
        "model_arch": row["model_arch"],
        "labels": list(row["labels"]),
        "path": dest,
        "sha256": digest,
    }


# --------------------------------------------------------------------------- finished models as parents

# A finished run of these pipelines carries its validated training weights (``weights.pt`` in the
# bundle, sha256 in the manifest) and can start a new job with ``parent_model_id``. The stored
# artifact is read in place: the parent is never copied, changed or re-imported.
PARENT_PIPELINES = frozenset({"conventional_classifier", "dinov3_linear"})
_PARENT_CACHE: dict[tuple, tuple[Optional[dict], Optional[str]]] = {}
_PARENT_CACHE_MAX = 512


def _read_parent_manifest(path: Path) -> tuple[Optional[dict], Optional[str]]:
    try:
        with zipfile.ZipFile(path) as zf:
            infos = {i.filename: i for i in zf.infolist()}
            m_info, w_info = infos.get("manifest.json"), infos.get("weights.pt")
            if m_info is None:
                return None, "the stored model has no manifest"
            if w_info is None:
                return None, (
                    "the model was exported without training weights (weights.pt); "
                    f"{RETRAIN_HINT}"
                )
            if m_info.file_size > MAX_MANIFEST_BYTES:
                return None, "the stored model manifest is too large"
            manifest = json.loads(zf.read(m_info))
            digest = hashlib.sha256()
            with zf.open(w_info) as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    digest.update(chunk)
    except (zipfile.BadZipFile, OSError, ValueError, RuntimeError) as exc:
        return None, f"the stored model is not a readable bundle ({str(exc)[:200]})"
    if not isinstance(manifest, dict):
        return None, "the stored model manifest is not a JSON object"
    files = manifest.get("files")
    if not isinstance(files, dict) or files.get("weights.pt") != digest.hexdigest():
        return (
            None,
            "the training weights do not match the sha256 in the model manifest",
        )
    return manifest, None


def _parent_manifest(path: Path) -> tuple[Optional[dict], Optional[str]]:
    """``(manifest, None)`` for validated training weights, else ``(None, reason)``.

    Cached by path, size and mtime: listing models hashes each stored ``weights.pt`` once."""
    try:
        st = path.stat()
    except OSError:
        return None, "the stored model file is missing"
    key = (str(path), st.st_size, st.st_mtime_ns)
    hit = _PARENT_CACHE.get(key)
    if hit is None:
        hit = _read_parent_manifest(path)
        if len(_PARENT_CACHE) >= _PARENT_CACHE_MAX:
            _PARENT_CACHE.clear()
        _PARENT_CACHE[key] = hit
    return hit


def _parent_state(objects_dir, row: dict) -> tuple[Optional[dict], Optional[str]]:
    """The manifest of a finished model that can be a parent, else ``(None, reason)``."""
    pipeline = row.get("pipeline")
    if pipeline not in PARENT_PIPELINES:
        if pipeline == "yolo_detector":
            return None, (
                "a YOLO run is fine-tuned through Sparrow Studio: push it to a Studio project "
                "and import it as a base model"
            )
        return None, f"{pipeline} models cannot start a new fine-tune job"
    key = row.get("object_key")
    if not key:
        return None, "the stored model file is missing"
    try:
        path = LocalObjectStore(objects_dir).path(key)
    except ValueError:
        return None, "the stored model file is missing"
    manifest, reason = _parent_manifest(path)
    if manifest is not None and manifest.get("pipeline") != pipeline:
        return None, "the stored model manifest names another pipeline"
    return manifest, reason


def tag_model(objects_dir, row: dict) -> dict:
    """Tag a ``model_versions`` row (needs ``pipeline`` and ``object_key``) in place."""
    manifest, reason = _parent_state(objects_dir, row)
    return tag(row, manifest is not None, reason)


def _model_row(database_url: str, model_id: int) -> Optional[dict]:
    with connect(database_url) as conn:
        return conn.execute(
            "SELECT id, pipeline, object_key, sha256 FROM model_versions WHERE id = %s",
            (model_id,),
        ).fetchone()


def _encoder_mismatch(settings, model_id: int, manifest: dict) -> Optional[str]:
    """Why a DINOv3 parent's head cannot be reused with this service's encoder (or ``None``).

    An unset or unreadable encoder spec is not judged here; the worker checks it again."""
    from .embedders import embedder_identity

    current = embedder_identity(getattr(settings, "embedder", "") or "")
    trained = manifest.get("embedder") or {}
    if current is None or current == trained.get("model_hash"):
        return None
    return (
        f"model {model_id} was trained on encoder {trained.get('id')!r}, but this service "
        "embeds with a different encoder; a different encoder means its head cannot be reused"
    )


def check_parent_model(settings, model_id: int, pipeline: str, params: dict) -> dict:
    """Raise ``ValueError`` unless finished model ``model_id`` can be the parent of the job."""
    row = _model_row(settings.database_url, model_id)
    if row is None:
        raise ValueError(f"model {model_id} not found")
    if row["pipeline"] != pipeline:
        raise ValueError(
            f"model {model_id} is a {row['pipeline']} model, it cannot start a {pipeline} job"
        )
    manifest, reason = _parent_state(settings.objects_dir, row)
    if manifest is None:
        raise ValueError(f"model {model_id} cannot be fine-tuned: {reason}")
    if pipeline == "conventional_classifier":
        arch, model = manifest.get("model_arch"), params.get("model")
        if model is not None and arch != model:
            raise ValueError(
                f"model {model_id} is a {arch} network but the job trains {model}; "
                f"pick model={arch!r} or another parent model"
            )
    else:
        mismatch = _encoder_mismatch(settings, model_id, manifest)
        if mismatch:
            raise ValueError(mismatch)
    return row


def stage_parent(settings, model_id: int, workdir: Path) -> dict:
    """``JobContext.base_model`` for a parent: its stored bundle, read in place after a sha check."""
    row = _model_row(settings.database_url, model_id)
    if row is None:
        raise ValueError(f"model {model_id} not found")
    path = LocalObjectStore(settings.objects_dir).path(row["object_key"])
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != row["sha256"]:
        raise ValueError(
            f"model {model_id} artifact is corrupt (sha256 {digest.hexdigest()}, "
            f"expected {row['sha256']})"
        )
    manifest, reason = _parent_state(settings.objects_dir, row)
    if manifest is None:
        raise ValueError(f"model {model_id} cannot be fine-tuned: {reason}")
    return {
        "id": row["id"],
        "origin": "parent",
        "kind": "parent",
        "model_arch": manifest.get("model_arch"),
        "labels": list(manifest.get("labels") or []),
        "path": path,
        "sha256": row["sha256"],
    }


def _snapshot_labels(database_url: str, snapshot_id: int) -> Optional[list[str]]:
    with connect(database_url) as conn:
        if (
            conn.execute(
                "SELECT 1 FROM snapshots WHERE id = %s", (snapshot_id,)
            ).fetchone()
            is None
        ):
            return None
        rows = conn.execute(
            "SELECT DISTINCT a->>'label' AS label FROM snapshot_items si, "
            "jsonb_array_elements(si.annotations) a WHERE si.snapshot_id = %s",
            (snapshot_id,),
        ).fetchall()
    return sorted({r["label"] for r in rows if r["label"] is not None})


def reuse_preview(settings, model_id: int, snapshot_id: int) -> Optional[dict]:
    """What a new job on ``snapshot_id`` would reuse from finished model ``model_id``.

    ``None`` for an unknown model; ``LookupError`` for an unknown snapshot."""
    row = _model_row(settings.database_url, model_id)
    if row is None:
        return None
    labels = _snapshot_labels(settings.database_url, snapshot_id)
    if labels is None:
        raise LookupError(f"snapshot {snapshot_id} not found")

    def out(usable: bool, backbone: bool, head: bool, reason: str) -> dict:
        return {
            "usable": usable,
            "transfers": {"backbone": backbone, "head": head},
            "reason": reason,
        }

    manifest, reason = _parent_state(settings.objects_dir, row)
    if manifest is None:
        return out(False, False, False, reason or "no validated training weights")
    parent_labels = list(manifest.get("labels") or [])
    if row["pipeline"] == "conventional_classifier":
        if parent_labels == labels:
            return out(
                True,
                True,
                True,
                "same labels: the backbone and the classifier head are both reused",
            )
        return out(
            True,
            True,
            False,
            f"the labels differ (model: {_few(parent_labels)}; snapshot: {_few(labels)}): "
            "the backbone is reused and the classifier head is re-initialised",
        )
    mismatch = _encoder_mismatch(settings, model_id, manifest)
    if mismatch:
        return out(False, False, False, mismatch)
    shared = set(parent_labels) & set(labels)
    if not shared:
        return out(
            True,
            False,
            False,
            "same encoder, but no labels in common: the linear head is re-initialised",
        )
    return out(
        True,
        False,
        True,
        f"same encoder: the linear head is reused for {len(shared)} shared label(s); "
        "the encoder itself is frozen and not trained",
    )
