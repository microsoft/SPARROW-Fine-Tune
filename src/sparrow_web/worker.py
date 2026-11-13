"""Detached job entry point.

    python -u -m sparrow_web.worker --job <work_dir>/job.json --status <work_dir>/status.json

Mirrors :mod:`sparrow_yolo.process_train`: the process is launched detached, owns
its own working directory, and reports exclusively through ``status.json`` and
``result.json``. The web process never talks to it directly — it reads those
files. That is what makes a job survive the web app restarting, and what makes
ingestion replayable if the web app was down when the job finished.

Two rules that must hold:

* **Torch is imported lazily, inside the handlers that need it.** Audit and
  prepare must never pull CUDA into this process — they are CPU/IO work and are
  deliberately allowed to run while a training job holds the GPU.
* **Every exit path writes a terminal state.** A job that dies without one is
  indistinguishable from an OOM kill, and the scheduler will (correctly) mark it
  ``died``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict

from sparrow_yolo.process_train import StatusWriter


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------- handlers --

def _handle_audit(params: Dict[str, Any], status: StatusWriter) -> Dict[str, Any]:
    """Scan a dataset and persist the result for the prepare step to reuse."""
    from sparrow_yolo.dataset import audit_dataset

    from .snapshots import (content_hash, label_fingerprints, quick_hash,
                            save_audit, summarise_classes)

    out_dir = Path(params["artifact_dir"])
    started = time.time()

    def on_progress(done: int, total: int, phase: str) -> None:
        status.update(state="running", progress_phase=phase,
                      progress_current=done, progress_total=total)

    result = audit_dataset(
        params["dataset_root"],
        normalise_class_names=params.get("normalise_class_names", True),
        deep_image_check=params.get("deep_image_check", False),
        confirmed_blank_dirs=params.get("confirmed_blank_dirs", ()),
        merge_classes=params.get("merge_classes") or None,
        fingerprint=True,          # always: this is what makes change detection work
        on_progress=on_progress,
        progress=False,            # no tqdm; the log is read by a machine
    )
    save_audit(result, out_dir)

    labels = label_fingerprints(result.boxes)
    classes = summarise_classes(result.boxes)
    issue_counts = (
        {str(k): int(v) for k, v in result.issue_counts().items()}
        if not result.issues.empty else {}
    )

    return {
        "artifact_dir": str(out_dir),
        "summary": result.summary,
        "warnings": result.warnings,
        "content_hash": content_hash(result.images),
        "quick_hash": quick_hash(result.images),
        "class_counts": {k: {"n_boxes": v[0], "n_images": v[1]} for k, v in classes.items()},
        "issue_counts": issue_counts,
        "label_fingerprints": labels,
        "duration_s": round(time.time() - started, 2),
    }


def _handle_prepare(params: Dict[str, Any], status: StatusWriter) -> Dict[str, Any]:
    """Build a trainable YOLO tree from a stored audit.

    Reuses the snapshot's saved audit rather than re-scanning: a re-audit would
    cost another ~50 seconds and, worse, could see different data than the
    snapshot the user chose.
    """
    from sparrow_yolo.dataset import (load_or_create_class_map, load_pinned_splits,
                                      prepare_yolo_dataset, split_images, verify_prepared)

    from .snapshots import load_audit

    status.update(state="running", progress_phase="Loading audit",
                  progress_current=0, progress_total=1)
    audit = load_audit(params["artifact_dir"])

    pinned_classes = Path(params["pinned_dir"]) / "classes.json"
    pinned_splits = Path(params["pinned_dir"]) / "splits.json"

    class_map, warnings = load_or_create_class_map(
        pinned_classes,
        audit.boxes,
        drop_classes=params.get("drop_classes", ()),
        min_boxes=int(params.get("min_boxes_per_class", 0)),
        reuse=bool(params.get("reuse_class_map", True)),
    )
    if not class_map:
        raise ValueError("No classes found in this dataset — nothing to prepare.")

    group_by = params.get("group_by") or None
    if group_by in ("none", ""):
        group_by = None

    splits = split_images(
        audit.images, audit.boxes,
        train=float(params.get("train", 0.8)),
        val=float(params.get("val", 0.1)),
        test=float(params.get("test", 0.1)),
        seed=int(params.get("seed", 42)),
        stratify=str(params.get("stratify", "rarest")),
        group_by=group_by,
        background_policy=str(params.get("background_policy", "reviewed")),
        # Keeps every existing image in the split it was in last time, so a model
        # trained after new data arrives is still comparable to the one before.
        pinned=load_pinned_splits(pinned_splits) if params.get("reuse_split", True) else None,
    )

    def on_progress(done: int, total: int, phase: str) -> None:
        status.update(state="running", progress_phase=phase,
                      progress_current=done, progress_total=total)

    data_yaml = prepare_yolo_dataset(
        audit,
        out_root=params["out_root"],
        project_name=str(params["prepared_name"]),
        class_map=class_map,
        split_df=splits,
        overwrite=True,          # writes a fresh, versioned directory of its own
        link_mode=str(params.get("link_mode", "auto")),
        on_progress=on_progress,
        progress=False,
    )

    prepared_dir = Path(data_yaml).parent
    # Promote this tree's pinned files to the project level so the next prepare
    # inherits the same class ids and split membership.
    pinned_classes.parent.mkdir(parents=True, exist_ok=True)
    for name in ("classes.json", "splits.json"):
        src = prepared_dir / name
        if src.is_file():
            (pinned_classes.parent / name).write_text(
                src.read_text(encoding="utf-8"), encoding="utf-8")

    status.update(state="running", progress_phase="Verifying",
                  progress_current=0, progress_total=1)
    verify = verify_prepared(data_yaml)
    manifest = json.loads((prepared_dir / "dataset_manifest.json").read_text(encoding="utf-8"))

    counts = splits["split"].value_counts().to_dict()
    return {
        "dir_path": str(prepared_dir),
        "data_yaml": str(data_yaml),
        "class_map": class_map,
        "warnings": warnings,
        "manifest": manifest,
        "verify": verify,
        "verify_ok": bool(verify.get("ok")),
        "n_train": int(counts.get("train", 0)),
        "n_val": int(counts.get("val", 0)),
        "n_test": int(counts.get("test", 0)),
    }


def _handle_evaluate(params: Dict[str, Any], status: StatusWriter) -> Dict[str, Any]:
    """Score a checkpoint on a split. Runs on the inference GPU, not the training one."""
    from sparrow_yolo.train_eval import evaluate_checkpoint

    status.update(state="running", progress_phase="Evaluating",
                  progress_current=0, progress_total=1)
    metrics = evaluate_checkpoint(
        params["weights"],
        params["data_yaml"],
        split=params.get("split", "test"),
        imgsz=int(params.get("imgsz", 1024)),
        device=str(params.get("device", "0")),
    )

    results = {k: float(v) for k, v in getattr(metrics, "results_dict", {}).items()
               if isinstance(v, (int, float))}

    # Per-class metrics are the point on a camera-trap dataset: an overall mAP50
    # can look healthy while a rare species sits at zero.
    per_class = []
    try:
        box = metrics.box
        names = getattr(metrics, "names", {}) or {}
        for i, class_id in enumerate(getattr(box, "ap_class_index", []) or []):
            per_class.append({
                "class_id": int(class_id),
                "class_name": str(names.get(int(class_id), class_id)),
                "precision": float(box.p[i]), "recall": float(box.r[i]),
                "ap50": float(box.ap50[i]), "ap5095": float(box.ap[i]),
            })
    except Exception:  # noqa: BLE001 - shape varies by Ultralytics version
        pass

    return {"metrics": results, "per_class": per_class,
            "split": params.get("split", "test")}


def _handle_export(params: Dict[str, Any], status: StatusWriter) -> Dict[str, Any]:
    from sparrow_yolo.train_eval import export_onnx

    status.update(state="running", progress_phase="Exporting",
                  progress_current=0, progress_total=1)
    path = export_onnx(
        params["weights"],
        imgsz=int(params.get("imgsz", 1024)),
        opset=params.get("opset"),
        simplify=bool(params.get("simplify", True)),
        dynamic=bool(params.get("dynamic", False)),
        half=bool(params.get("half", False)),
    )
    return {"path": str(path)}


def _handle_predict(params: Dict[str, Any], status: StatusWriter) -> Dict[str, Any]:
    """Pre-annotate a batch with one of the project's own trained models.

    Runs on the inference GPU, so it can proceed while a training run holds the
    other card — which is the point of having the annotation tool inside the
    workbench rather than beside it.

    Uses the registered ``.pt`` through Ultralytics rather than an ONNX export.
    That gets the class names, letterboxing and NMS the model was trained with,
    instead of re-deriving them from a tensor whose layout varies by exporter.
    """
    from ultralytics import YOLO

    from sparrow_yolo.xml_utils import write_voc_xml

    batch_dir = Path(params["batch_dir"])
    images_dir = batch_dir / "images"
    force_class = (params.get("force_class") or "").strip()
    only_classes = set(params.get("only_classes") or [])
    only_missing = bool(params.get("only_missing"))
    predicted_by = str(params.get("predicted_by") or "")

    targets = [p for p in sorted(images_dir.iterdir()) if p.is_file()]
    if only_missing:
        targets = [p for p in targets
                   if not (batch_dir / "annotations" / (p.stem + ".xml")).exists()]

    total = len(targets)
    status.update(state="running", progress_phase="Loading model",
                  progress_current=0, progress_total=max(1, total))
    model = YOLO(params["weights"])
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))

    started = time.time()
    images: list = []
    n_boxes = n_blank = n_errors = 0

    for i, image_path in enumerate(targets, 1):
        try:
            result = model.predict(
                source=str(image_path),
                conf=float(params.get("conf", 0.35)),
                iou=float(params.get("iou", 0.45)),
                imgsz=int(params.get("imgsz", 1024)),
                max_det=int(params.get("max_det", 300)),
                device=str(params.get("device", "0")),
                verbose=False,
            )[0]
            height, width = result.orig_shape

            boxes = []
            for b in result.boxes:
                label = force_class or str(names.get(int(b.cls), int(b.cls)))
                if only_classes and label not in only_classes:
                    continue
                x0, y0, x1, y1 = (float(v) for v in b.xyxy[0].tolist())
                boxes.append({"label": label, "xmin": x0, "ymin": y0,
                              "xmax": x1, "ymax": y1, "score": float(b.conf),
                              "source": "model"})

            write_voc_xml(batch_dir / "annotations" / (image_path.stem + ".xml"),
                          image_path, width, height, boxes,
                          reviewed=False, predicted_by=predicted_by)

            scores = [b["score"] for b in boxes]
            images.append({
                "rel_path": image_path.name, "width": int(width), "height": int(height),
                "n_boxes": len(boxes), "min_confidence": min(scores) if scores else None,
            })
            n_boxes += len(boxes)
            n_blank += 0 if boxes else 1
        except Exception as e:  # noqa: BLE001 - one bad image must not lose the pass
            n_errors += 1
            images.append({"rel_path": image_path.name, "error": str(e)})

        if i % 10 == 0 or i == total:
            status.update(progress_phase="Finding animals",
                          progress_current=i, progress_total=max(1, total))

    return {"n_images": total, "n_boxes": n_boxes, "n_blank": n_blank,
            "n_errors": n_errors, "duration_s": round(time.time() - started, 2),
            "images": images}


HANDLERS: Dict[str, Callable[[Dict[str, Any], StatusWriter], Dict[str, Any]]] = {
    "audit": _handle_audit,
    "prepare": _handle_prepare,
    "evaluate": _handle_evaluate,
    "export": _handle_export,
    "predict": _handle_predict,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a SPARROW job outside the web process.")
    parser.add_argument("--job", required=True, help="Path to job.json")
    parser.add_argument("--status", required=True, help="Path to write status JSON")
    args = parser.parse_args()

    status = StatusWriter(args.status)
    job_path = Path(args.job)
    started = time.strftime("%Y-%m-%d %H:%M:%S")

    try:
        job = json.loads(job_path.read_text(encoding="utf-8"))
        kind = job["kind"]
        params = job.get("params", {})
    except Exception as e:  # noqa: BLE001
        status.update(state="failed", started=started, error=f"Unreadable job file: {e!r}")
        return 1

    handler = HANDLERS.get(kind)
    if handler is None:
        status.update(state="failed", started=started,
                      error=f"Unknown job kind {kind!r}. Known: {sorted(HANDLERS)}")
        return 1

    status.update(state="running", kind=kind, started=started, job=str(job_path))
    print(f"=== job {job.get('id', '?')} ({kind}) starting ===", flush=True)

    try:
        result = handler(params, status)
    except Exception as e:  # noqa: BLE001 - must be recorded, not just raised
        tb = traceback.format_exc()
        status.update(state="failed", finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                      error=repr(e), traceback=tb)
        print(tb, file=sys.stderr, flush=True)
        return 1

    _write_json(job_path.parent / "result.json", result)
    status.update(state="succeeded", finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"=== job {job.get('id', '?')} ({kind}) succeeded ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
