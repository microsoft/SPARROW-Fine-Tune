"""DINO feature preparation (Data Preparation).

A classification snapshot can opt in to having its DINO crop features computed by a worker that
has the encoder. Each vector is stored once as a file under ``<data_dir>/features`` and indexed in
``feature_vectors`` by a key over the crop's pixels (image sha256 + crop box), the encoder (its
model hash) and the preprocessing (input size, mean, std). Labels and training hyperparameters are
not part of the key, so relabelling or retraining reuses the vectors. ``dinov3_linear`` training
reads the prepared vectors and never runs the encoder.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np
from PIL import Image

from .db import connect
from .embedders import IMAGENET_MEAN, IMAGENET_STD

PREP_PIPELINE = "dinov3_features"
DEFAULT_INPUT_SIZE = 224
MIN_INPUT_SIZE = 32
MAX_INPUT_SIZE = 1024
STATUSES = ("valid", "missing", "stale", "corrupt")
BATCH = 16


def vector_key(
    image_sha: str, box: Iterable[int], encoder_hash: str, input_size: int
) -> str:
    payload = {
        "image": image_sha,
        "box": [int(v) for v in box],
        "encoder": encoder_hash,
        "input_size": int(input_size),
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def vector_relpath(key: str, encoder_hash: str, input_size: int) -> str:
    enc = hashlib.sha256(encoder_hash.encode()).hexdigest()[:16]
    return f"features/{enc}/{int(input_size)}/{key[:2]}/{key}.npy"


def crops(rows: Iterable[dict]) -> list[dict]:
    """Crops in training order: every annotation of every row, skipping unannotated rows.

    A row needs ``name``, ``sha256``, ``width``, ``height`` and ``annotations``; extra keys
    (``object_key``) are carried along."""
    from .pipelines.dinov3_linear import crop_box

    out = []
    for row in rows:
        for ann in row.get("annotations") or []:
            bbox = ann.get("bbox")
            box = crop_box(bbox, row["width"], row["height"])
            out.append(
                {
                    "image": row["name"],
                    "sha256": row["sha256"],
                    "object_key": row.get("object_key"),
                    "box": list(box),
                    "crop": [float(v) for v in bbox]
                    if bbox is not None
                    else [0, 0, int(row["width"]), int(row["height"])],
                }
            )
    return out


def _load_vector(path: Path, dim: int) -> Optional[np.ndarray]:
    try:
        vec = np.load(path, allow_pickle=False)
    except Exception:
        return None
    if not isinstance(vec, np.ndarray) or vec.ndim != 1 or vec.shape[0] != dim:
        return None
    if not np.issubdtype(vec.dtype, np.floating) or not np.isfinite(vec).all():
        return None
    return vec.astype(np.float32, copy=False)


def statuses(
    conn,
    data_dir: Path,
    crop_list: list[dict],
    encoder_hash: Optional[str],
    input_size: int,
    *,
    load: bool = False,
) -> list[dict]:
    """Annotate each crop with ``status`` (and ``key``; ``vector`` when ``load``)."""
    shas = sorted({c["sha256"] for c in crop_list})
    keys = (
        [vector_key(c["sha256"], c["box"], encoder_hash, input_size) for c in crop_list]
        if encoder_hash
        else [None] * len(crop_list)
    )
    rows = {}
    known: set[str] = set()
    if shas:
        for r in conn.execute(
            "SELECT key, image_sha256, dim, path, encoder_id, encoder_hash FROM feature_vectors"
            " WHERE image_sha256 = ANY(%s)",
            (shas,),
        ).fetchall():
            rows[r["key"]] = r
            known.add(r["image_sha256"])
    out = []
    for crop, key in zip(crop_list, keys):
        entry = dict(crop, key=key)
        row = rows.get(key) if key else None
        if row is not None:
            vec = _load_vector(Path(data_dir) / row["path"], row["dim"])
            if vec is None:
                entry["status"] = "corrupt"
            else:
                entry["status"] = "valid"
                entry["row"] = row
                if load:
                    entry["vector"] = vec
        elif crop["sha256"] in known:
            entry["status"] = "stale"
        else:
            entry["status"] = "missing"
        out.append(entry)
    return out


def summary(checked: list[dict]) -> dict:
    counts = {s: 0 for s in STATUSES}
    for c in checked:
        counts[c["status"]] += 1
    counts["total"] = len(checked)
    to_compute = counts["total"] - counts["valid"]
    if to_compute:
        message = (
            f"{to_compute} of {counts['total']} crop(s) need DINO features computed;"
            f" {counts['valid']} prepared vector(s) can be reused."
        )
    else:
        message = f"Nothing to compute: all {counts['total']} crop(s) have prepared DINO features."
    return {
        "items": [
            {"image": c["image"], "crop": c["crop"], "status": c["status"]}
            for c in checked
        ],
        "counts": counts,
        "to_compute": to_compute,
        "message": message,
    }


def snapshot_crops(conn, snapshot_id: int) -> list[dict]:
    rows = conn.execute(
        "SELECT image_name AS name, object_key, sha256, width, height, annotations"
        " FROM snapshot_items WHERE snapshot_id = %s ORDER BY position",
        (snapshot_id,),
    ).fetchall()
    return crops(rows)


def snapshot_state(
    conn, data_dir: Path, snapshot_id: int, features: Optional[dict]
) -> dict:
    """The ``GET /v1/snapshots/{id}/features`` object."""
    if not features:
        return {"state": "not_prepared"}
    job = conn.execute(
        "SELECT id, state, error FROM jobs WHERE snapshot_id = %s AND pipeline = %s"
        " ORDER BY id DESC LIMIT 1",
        (snapshot_id, PREP_PIPELINE),
    ).fetchone()
    checked = statuses(
        conn,
        data_dir,
        snapshot_crops(conn, snapshot_id),
        features.get("encoder_hash"),
        features.get("input_size", DEFAULT_INPUT_SIZE),
    )
    total = len(checked)
    remaining = sum(1 for c in checked if c["status"] != "valid")
    to_compute = int(features.get("to_compute", 0))
    done = max(0, min(to_compute, to_compute - remaining))
    job_state = job["state"] if job else None
    if job_state in ("queued", "running", "failed", "cancelled"):
        state = job_state
    elif remaining == 0:
        state = "ready"
        done = to_compute
    elif any(c["status"] == "corrupt" for c in checked):
        state = "corrupt"
    else:
        state = "stale"
    messages = {
        "queued": "DINO feature preparation is waiting for a worker with the encoder.",
        "running": f"Preparing DINO features: {done} of {to_compute} computed.",
        "failed": "DINO feature preparation failed"
        + (f": {job['error']}" if job and job.get("error") else "")
        + ". Retry the preparation job.",
        "cancelled": "DINO feature preparation was cancelled. Retry the preparation job.",
        "ready": "DINO features are prepared; the snapshot is ready for training.",
        "corrupt": "Some prepared DINO features are damaged: create a new prepared snapshot.",
        "stale": "Some prepared DINO features are missing: create a new prepared snapshot.",
    }
    return {
        "state": state,
        "total": total,
        "to_compute": to_compute,
        "reused": int(features.get("reused", total - to_compute)),
        "done": done,
        "job_id": job["id"] if job else features.get("job_id"),
        "message": messages[state],
    }


def _write_vector(data_dir: Path, relpath: str, vec: np.ndarray) -> None:
    dest = Path(data_dir) / relpath
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("xb") as fh:
            np.save(fh, np.asarray(vec, dtype=np.float32))
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def build_embedder(spec: str, input_size: int):
    """The embedder named by ``spec`` with the snapshot's input size."""
    from .embedders import EngineEmbedder, OnnxEmbedder, make_embedder

    spec = (spec or "").strip()
    kind, sep, rest = spec.partition(":")
    rest = rest.strip()
    if sep and kind == "onnx" and rest:
        return OnnxEmbedder(rest, input_size=int(input_size))
    if sep and kind == "engine":
        url, hash_sep, model_id = rest.rpartition("#")
        if hash_sep and url.strip() and model_id.strip():
            return EngineEmbedder(
                url.strip(), model_id.strip(), input_size=int(input_size)
            )
    return make_embedder(spec)  # raises the standard error


def run_preparation(
    settings: Any,
    snapshot_id: int,
    emit: Callable[[str, dict], None],
    check_cancelled: Callable[[], None],
) -> dict:
    """Compute and store every crop of the snapshot that has no valid prepared vector."""
    from .storage import LocalObjectStore

    data_dir = Path(settings.data_dir)
    with connect(settings.database_url) as conn:
        snap = conn.execute(
            "SELECT features FROM snapshots WHERE id = %s", (snapshot_id,)
        ).fetchone()
        if snap is None or not snap["features"]:
            raise ValueError(
                f"snapshot {snapshot_id} did not opt in to DINO feature preparation"
            )
        features = snap["features"]
        crop_list = snapshot_crops(conn, snapshot_id)
        encoder_hash = features["encoder_hash"]
        input_size = int(features.get("input_size", DEFAULT_INPUT_SIZE))
        checked = statuses(conn, data_dir, crop_list, encoder_hash, input_size)
    todo = [c for c in checked if c["status"] != "valid"]
    total = len(todo)
    emit(f"computing {total} of {len(checked)} crop(s)", {"done": 0, "total": total})
    if not todo:
        return {"computed": 0, "total": 0}
    embedder = build_embedder(settings.embedder, input_size)
    try:
        if embedder.model_hash != encoder_hash:
            raise ValueError(
                f"this worker's encoder (model hash {embedder.model_hash!r}) is not the one the"
                f" snapshot was prepared for ({encoder_hash!r})"
            )
        store = LocalObjectStore(settings.objects_dir)
        done = 0
        pending: list[tuple[dict, Image.Image]] = []

        def flush() -> None:
            nonlocal done
            out = np.asarray(embedder.embed([p for _, p in pending]), dtype=np.float32)
            if out.ndim != 2 or out.shape[0] != len(pending):
                raise RuntimeError(
                    f"encoder returned {out.shape[0] if out.ndim else 0} features"
                    f" for {len(pending)} crops"
                )
            rows = []
            for (crop, _), vec in zip(pending, out):
                if not np.isfinite(vec).all():
                    raise RuntimeError(
                        f"encoder returned a non-finite vector for {crop['image']}"
                    )
                rel = vector_relpath(crop["key"], encoder_hash, input_size)
                _write_vector(data_dir, rel, vec)
                rows.append((crop, rel, int(vec.shape[0])))
            with connect(settings.database_url) as conn:
                for crop, rel, dim in rows:
                    conn.execute(
                        "INSERT INTO feature_vectors (key, image_sha256, box, encoder_hash,"
                        " encoder_id, input_size, dim, path) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"
                        " ON CONFLICT (key) DO UPDATE SET dim = EXCLUDED.dim,"
                        " path = EXCLUDED.path, encoder_id = EXCLUDED.encoder_id,"
                        " created_at = now()",
                        (
                            crop["key"],
                            crop["sha256"],
                            crop["box"],
                            encoder_hash,
                            embedder.id,
                            input_size,
                            dim,
                            rel,
                        ),
                    )
            done += len(pending)
            pending.clear()
            emit(f"computed {done} of {total} crop(s)", {"done": done, "total": total})
            check_cancelled()

        by_image: dict[str, list[dict]] = {}
        for crop in todo:
            by_image.setdefault(crop["object_key"], []).append(crop)
        for object_key, group in by_image.items():
            check_cancelled()
            data = store.path(object_key).read_bytes()
            with Image.open(io.BytesIO(data)) as im:
                im = im.convert("RGB")
                for crop in group:
                    pending.append(
                        (crop, embedder.preprocess_crop(im.crop(tuple(crop["box"]))))
                    )
                    if len(pending) >= BATCH:
                        flush()
        if pending:
            flush()
    finally:
        embedder.close()
    return {"computed": done, "total": total}


def load_prepared(settings: Any, snapshot_id: int) -> dict:
    """Prepared vectors of a ready snapshot in training crop order, for ``dinov3_linear``."""
    with connect(settings.database_url) as conn:
        snap = conn.execute(
            "SELECT features FROM snapshots WHERE id = %s", (snapshot_id,)
        ).fetchone()
        features = snap["features"] if snap else None
        if not features:
            raise ValueError(
                f"snapshot {snapshot_id} has no prepared DINO features: create the snapshot with"
                " Data Preparation (prepare_features) first"
            )
        input_size = int(features.get("input_size", DEFAULT_INPUT_SIZE))
        checked = statuses(
            conn,
            Path(settings.data_dir),
            snapshot_crops(conn, snapshot_id),
            features["encoder_hash"],
            input_size,
            load=True,
        )
    bad = [c for c in checked if c["status"] != "valid"]
    if bad:
        raise ValueError(
            f"snapshot {snapshot_id}: {len(bad)} crop(s) have no valid prepared DINO features"
            f" ({bad[0]['status']}); run Data Preparation again"
        )
    if not checked:
        raise ValueError(f"snapshot {snapshot_id} has no crops")
    dims = {c["row"]["dim"] for c in checked}
    if len(dims) != 1:
        raise ValueError(
            f"snapshot {snapshot_id}: prepared vectors disagree on dimension"
        )
    first = checked[0]["row"]
    return {
        "vectors": np.stack([c["vector"] for c in checked]).astype(np.float32),
        "embedder": {
            "id": first["encoder_id"],
            "dim": int(first["dim"]),
            "model_hash": first["encoder_hash"],
        },
        "input": {
            "size": input_size,
            "mean": [float(v) for v in IMAGENET_MEAN],
            "std": [float(v) for v in IMAGENET_STD],
        },
    }
