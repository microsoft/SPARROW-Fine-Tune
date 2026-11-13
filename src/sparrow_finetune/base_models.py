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
"""

from __future__ import annotations

import hashlib
import io
import json
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
    "id, studio_project_id, studio_import_id, source_model_id, pipeline, model_arch, "
    "labels, sha256, size_bytes, created_at"
)


class BundleError(ValueError):
    """The Studio bundle is malformed or tampered with (Studio-side problem: 502)."""


class NotImportable(ValueError):
    """The bundle is well formed but cannot be a base model (client-side problem: 422)."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def public(row: dict) -> dict:
    """A ``base_models`` row as the API returns it (no object-store key)."""
    out = {k: v for k, v in row.items() if k != "object_key"}
    if out.get("created_at") is not None and hasattr(out["created_at"], "isoformat"):
        out["created_at"] = out["created_at"].isoformat()
    return out


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
        "model_arch": row["model_arch"],
        "labels": list(row["labels"]),
        "path": dest,
        "sha256": digest,
    }
