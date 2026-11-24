"""Full end-to-end test of Studio data sync and model sync, through the service REST API.

Builds on ``studio_branch_e2e.py`` (same compose project shape, same synthetic scenes) and covers
what that UI run did not:

* Data sync: classification and detection imports, paging (exactly 100 and more than 100 images),
  ``reviewed_only`` on and off, re-import (upsert, no duplicates, pending -> confirmed promotion,
  excluded stays excluded, annotation replacement), duplicate and missing filenames, deleted
  boxes, blocked images, images without a gallery copy, images without boxes, the label
  precedence chain, manual uploads mixed with imported items, snapshot hashes.
* Training every pipeline on imported data: ``conventional_classifier``, ``dinov3_linear`` (real
  DINOv3 ONNX embedder) and ``yolo_detector``; task-mismatch submissions are refused.
* Model sync: push via the connector and via activation, re-push upsert, model versions and
  deployment switching, local and Studio activations side by side, one model pushed to two
  projects, inference with every pushed artifact.
* Studio on Azure Blob storage (Azurite): import from blob, push into blob.
* Failure paths: wrong Studio token, Studio down, missing image, unknown project, dataset and
  model, connector not configured, empty project, no API token. Every failed import must leave
  the dataset unchanged.

Usage (from the fine-tune repo root; ``sparrow-finetune:latest`` built; ONNX assets exported with
``scripts/e2e/export_dinov3_onnx.py`` into ``--assets`` together with ``yolo11n.pt``):

    uv run --extra dev --extra e2e --with azure-storage-blob \
        python scripts/e2e/studio_full_e2e.py --studio-root <Studio checkout> --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from datetime import datetime
import os
import random
import secrets
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import psycopg
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from studio_branch_e2e import (  # noqa: E402
    CHECKS,
    REPO,
    check,
    compose,
    free_port,
    run,
    scene,
    write_override,
)

AZURITE_KEY = "Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw=="
BLOB_CONTAINER = "studio"


def azurite_conn(endpoint: str) -> str:
    return (
        "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;"
        f"AccountKey={AZURITE_KEY};BlobEndpoint={endpoint}/devstoreaccount1"
    )


# --------------------------------------------------------------------------- seeding
class Seeder:
    """Writes Studio rows the way Studio's inference worker does, and remembers what an
    export of each project must return."""

    def __init__(self, conn, gallery: Path, blob=None):
        self.conn = conn
        self.gallery = gallery
        self.blob = blob
        self.uid = conn.execute(
            "INSERT INTO users (username, email) VALUES ('e2e', 'e2e@example.org') RETURNING id"
        ).fetchone()[0]
        self.images: dict[int, list[dict]] = {}

    def project(self, name: str) -> int:
        pid = conn_one(
            self.conn,
            "INSERT INTO projects (owner_user_id, project_type, name) "
            "VALUES (%s, 'batch', %s) RETURNING id",
            (self.uid, name),
        )
        self.images[pid] = []
        return pid

    def add(
        self,
        pid: int,
        name,
        im: Image.Image,
        boxes: list,
        *,
        status="reviewed",
        gallery=True,
        blocked=False,
        detection=True,
        deleted: list = (),
        meta_boxes: list = (),
        to_blob=False,
    ) -> dict:
        """boxes: [(label, (x0,y0,x1,y1))] stored as classifier_class.
        meta_boxes: [(classifier_class, box_meta, xyxy, expected_label)]."""
        n = len(self.images[pid])
        iid = conn_one(
            self.conn,
            "INSERT INTO images (project_id, ingestion_type, image_key, original_filename, "
            "review_status, processing_status, last_error) "
            "VALUES (%s, 'upload', %s, %s, %s, %s, %s) RETURNING id",
            (
                pid,
                f"staging/{n}_{name}",
                name,
                status,
                "failed" if blocked else "done",
                "blocked: e2e" if blocked else None,
            ),
        )
        rec = {
            "id": iid,
            "name": name,
            "reviewed": status == "reviewed",
            "exported": gallery and not blocked,
            "boxes": [],
            "size": im.size,
        }
        if gallery:
            key = f"projects/{pid}/2026-10-11/{iid}__{name or 'noname.jpg'}"
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=92)
            if to_blob:
                self.blob.get_blob_client(f"gallery/{key}").upload_blob(
                    buf.getvalue(), overwrite=True
                )
            else:
                path = self.gallery / key
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(buf.getvalue())
            rec["key"] = key
            self.conn.execute(
                "UPDATE images SET gallery_image_key=%s WHERE id=%s", (key, iid)
            )
        if detection:
            rows = [(lb, None, b, False, lb) for lb, b in boxes]
            rows += [(lb, None, b, True, None) for lb, b in deleted]
            rows += [(cc, meta, b, False, exp) for cc, meta, b, exp in meta_boxes]
            det = conn_one(
                self.conn,
                "INSERT INTO detections (image_id, project_id, detection_count) "
                "VALUES (%s, %s, %s) RETURNING id",
                (iid, pid, len(rows)),
            )
            for cc, meta, (x0, y0, x1, y1), is_del, expected in rows:
                bid = conn_one(
                    self.conn,
                    "INSERT INTO detection_boxes (detection_id, confidence, x_min, y_min, "
                    "x_max, y_max, classifier_class, box_meta, is_deleted) "
                    "VALUES (%s, 0.9, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
                    (
                        det,
                        x0,
                        y0,
                        x1,
                        y1,
                        cc,
                        json.dumps(meta) if meta is not None else None,
                        is_del,
                    ),
                )
                if not is_del:
                    rec["boxes"].append(
                        {"id": bid, "label": expected, "xyxy": (x0, y0, x1, y1)}
                    )
        self.images[pid].append(rec)
        return rec

    # ---- what the service must produce
    def exported(self, pid: int, reviewed_only: bool) -> list[dict]:
        return [
            r
            for r in self.images[pid]
            if r["exported"] and (r["reviewed"] or not reviewed_only)
        ]

    def expect(self, pid: int, reviewed_only: bool, task: str) -> dict:
        """Expected import result and item map {item name: (label(s), group, status)}."""
        rows = self.exported(pid, reviewed_only)
        with_boxes = [r for r in rows if r["boxes"]]
        bases = [r["name"] if r["name"] else None for r in with_boxes]
        counts: dict = {}
        for b in bases:
            if b:
                counts[b] = counts.get(b, 0) + 1
        items: dict[str, dict] = {}
        for r, base in zip(with_boxes, bases):
            if base is None:
                name = f"studio_{r['id']}"
            elif counts[base] > 1:
                stem, suf = os.path.splitext(base)
                name = f"{stem}__studio{r['id']}{suf}"
            else:
                name = base
            status = "confirmed" if r["reviewed"] else "pending"
            if task == "detection":
                items[name] = {
                    "status": status,
                    "group": None,
                    "annotations": sorted(
                        (
                            b["label"],
                            (
                                b["xyxy"][0],
                                b["xyxy"][1],
                                b["xyxy"][2] - b["xyxy"][0],
                                b["xyxy"][3] - b["xyxy"][1],
                            ),
                        )
                        for b in r["boxes"]
                    ),
                }
            else:
                stem = os.path.splitext(name)[0] or name
                for b in r["boxes"]:
                    items[f"{stem}__box{b['id']}.png"] = {
                        "status": status,
                        "group": stem,
                        "annotations": [(b["label"], None)],
                    }
        pages = max(1, -(-len(rows) // 100))
        return {
            "imported_images": len(with_boxes),
            "imported_items": len(items),
            "skipped": len(rows) - len(with_boxes),
            "pages": pages,
            "items": items,
        }


def conn_one(conn, sql, params):
    return conn.execute(sql, params).fetchone()[0]


def seed_all(dsn: str, gallery: Path, schema: Path, blob) -> tuple[Seeder, dict]:
    rng = random.Random(11)
    conn = psycopg.connect(dsn, autocommit=True)
    conn.execute(schema.read_text())
    s = Seeder(conn, gallery, blob)
    p: dict = {}

    # A: classification, > 100 exported images, every edge case of the export.
    a = p["A"] = s.project("E2E A classification")
    for i in range(120):
        im, boxes = scene(rng, 1 + (i % 2))
        s.add(
            a,
            f"cam{i:03d}.jpg",
            im,
            boxes,
            deleted=[("lion", (5, 5, 60, 60))] if i == 0 else [],
        )
    for i in range(8):
        im, boxes = scene(rng, 2)
        s.add(a, f"unrev{i}.jpg", im, boxes, status="unreviewed")
    im, _ = scene(rng, 0)
    s.add(a, "empty_det.jpg", im, [])  # detection row, no boxes
    im, _ = scene(rng, 0)
    s.add(a, "no_det.jpg", im, [], detection=False)  # no detection row at all
    im, boxes = scene(rng, 1)
    s.add(a, "no_gallery.jpg", im, boxes, gallery=False)
    im, boxes = scene(rng, 1)
    s.add(a, "blocked.jpg", im, boxes, blocked=True)
    for _ in range(2):
        im, boxes = scene(rng, 1)
        s.add(a, "dup.jpg", im, boxes)
    im, boxes = scene(rng, 1)
    s.add(a, None, im, boxes)

    # B: detection.
    b = p["B"] = s.project("E2E B detection")
    for i in range(80):
        im, boxes = scene(rng, 1 + (i % 2))
        s.add(b, f"det{i:03d}.jpg", im, boxes)
    for i in range(6):
        im, boxes = scene(rng, 2)
        s.add(b, f"detunrev{i}.jpg", im, boxes, status="unreviewed")

    # C: images only in Azure Blob storage.
    c = p["C"] = s.project("E2E C blob")
    for i in range(20):
        im, boxes = scene(rng, 1 + (i % 2))
        s.add(c, f"blob{i:02d}.jpg", im, boxes, to_blob=True)

    # D: label precedence (classifier_class -> box_meta classifier_class_name ->
    # detector_class_name -> label -> 'unknown'; '' and 'null' count as missing).
    d = p["D"] = s.project("E2E D labels")
    cases = [
        ("lion", {"classifier_class_name": "zebra"}, "lion"),
        (
            None,
            {"classifier_class_name": "zebra", "detector_class_name": "animal"},
            "zebra",
        ),
        (
            "null",
            {"classifier_class_name": "", "detector_class_name": "animal"},
            "animal",
        ),
        (
            "",
            {
                "classifier_class_name": "null",
                "detector_class_name": "null",
                "label": "giraffe",
            },
            "giraffe",
        ),
        (None, {}, "unknown"),
        (None, None, "unknown"),
    ]
    for i, (cc, meta, exp) in enumerate(cases):
        im, boxes = scene(rng, 1)
        s.add(d, f"label{i}.jpg", im, [], meta_boxes=[(cc, meta, boxes[0][1], exp)])

    # F: exactly one page (100 images).
    f = p["F"] = s.project("E2E F page boundary")
    for i in range(100):
        im, boxes = scene(rng, 1, size=(320, 240))
        s.add(f, f"f{i:03d}.jpg", im, boxes)
    return s, p


# --------------------------------------------------------------------------- service client
class Api:
    def __init__(self, base: str, token: str | None):
        self.base = base.rstrip("/") + "/v1"
        self.token = token

    def call(self, method, path, body=None, raw: bytes | None = None, timeout=900):
        headers = {}
        data = None
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if raw is not None:
            data = raw
            headers["Content-Type"] = "application/octet-stream"
        elif body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(
            self.base + path, data=data, method=method, headers=headers
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                payload = r.read()
                status = r.status
        except urllib.error.HTTPError as e:
            payload, status = e.read(), e.code
        try:
            return status, json.loads(payload) if payload else None
        except ValueError:
            return status, payload

    def ok(self, method, path, body=None, raw=None, expect=(200, 201)):
        status, out = self.call(method, path, body, raw)
        if status not in expect:
            raise RuntimeError(f"{method} {path} -> {status}: {str(out)[:400]}")
        return out


def items_of(api: Api, ds: int) -> dict[str, dict]:
    return {it["image"]: it for it in api.ok("GET", f"/datasets/{ds}/items")["items"]}


def actual_items(api: Api, ds: int, task: str) -> dict:
    out = {}
    for name, it in items_of(api, ds).items():
        anns = it["annotations"] or []
        if task == "detection":
            ann = sorted((a["label"], tuple(round(v) for v in a["bbox"])) for a in anns)
        else:
            ann = [(a["label"], None) for a in anns]
        out[name] = {
            "status": it["review_status"],
            "group": it["group"],
            "annotations": ann,
        }
    return out


def diff_items(expected: dict, actual: dict) -> str:
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    wrong = sorted(n for n in set(expected) & set(actual) if expected[n] != actual[n])
    if not (missing or extra or wrong):
        return ""
    parts = []
    if missing:
        parts.append(f"missing {missing[:3]} (+{len(missing)})")
    if extra:
        parts.append(f"extra {extra[:3]} (+{len(extra)})")
    if wrong:
        n = wrong[0]
        parts.append(f"wrong {len(wrong)}: {n} exp={expected[n]} got={actual[n]}")
    return "; ".join(parts)


def import_counts(res) -> dict:
    return {
        k: res.get(k) for k in ("imported_images", "imported_items", "skipped", "pages")
    }


def job_seconds(job: dict) -> str:
    if not (job.get("started_at") and job.get("finished_at")):
        return "?s"
    start = datetime.fromisoformat(job["started_at"])
    end = datetime.fromisoformat(job["finished_at"])
    return f"{(end - start).total_seconds():.0f}s"


def wait_job(api: Api, job_id: int, timeout_s: int) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        job = api.ok("GET", f"/jobs/{job_id}")
        if job["state"] in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(5)
    raise TimeoutError(f"job {job_id} still {job['state']} after {timeout_s}s")


FEATURE_KEYS = {"state", "total", "to_compute", "reused", "done", "job_id", "message"}


def wait_features(api: Api, snapshot_id: int, timeout_s: int) -> tuple[dict, list]:
    """Poll ``GET /snapshots/{id}/features`` until it leaves queued/running.

    Returns the final features object and every progress object seen, so a caller can check that
    ``done`` never decreased and never passed ``to_compute``.
    """
    deadline = time.time() + timeout_s
    seen: list[dict] = []
    while True:
        f = api.ok("GET", f"/snapshots/{snapshot_id}/features")
        seen.append(f)
        if f.get("state") not in ("queued", "running"):
            return f, seen
        if time.time() >= deadline:
            raise TimeoutError(
                f"snapshot {snapshot_id} features still {f.get('state')} after {timeout_s}s"
            )
        time.sleep(2)


def prepare_dino_snapshot(
    api: Api,
    ds: int,
    timeout_s: int,
    label: str,
    expect_new: int | None = None,
    expect_reused: int | None = None,
) -> dict:
    """Data Preparation for ``dinov3_linear``: opt-in snapshot, progress polled to ``ready``.

    The DINO embeddings are computed here, during preparation, never during training. A
    ``dinov3_linear`` job is only accepted on the returned snapshot. ``expect_new`` and
    ``expect_reused`` pin how much work the snapshot reported (new crops to compute, prepared
    crops reused); ``expect_new == 0`` additionally requires the zero-work shape: ``ready`` at
    once, no preparation job and a "nothing to compute" message.
    """
    snap = api.ok("POST", f"/datasets/{ds}/snapshots", {"prepare_features": True})
    f0 = snap.get("features") or {}
    check(
        f"{label}: opt-in snapshot reports DINO preparation progress",
        FEATURE_KEYS <= set(f0)
        and f0.get("total") == f0.get("to_compute", -1) + f0.get("reused", -1) > 0,
        json.dumps(f0),
    )
    if expect_new is not None or expect_reused is not None:
        check(
            f"{label}: preparation computes {expect_new} new and reuses {expect_reused}",
            (expect_new is None or f0.get("to_compute") == expect_new)
            and (expect_reused is None or f0.get("reused") == expect_reused),
            json.dumps(f0),
        )
    if expect_new == 0:
        check(
            f"{label}: zero work is ready at once with no preparation job",
            f0.get("state") == "ready"
            and f0.get("job_id") is None
            and "nothing to compute" in (f0.get("message") or "").lower(),
            json.dumps(f0),
        )
    elif f0.get("to_compute"):
        check(
            f"{label}: work remaining is queued for the worker",
            f0.get("state") in ("queued", "running")
            and isinstance(f0.get("job_id"), int),
            json.dumps(f0),
        )
    final, seen = wait_features(api, snap["id"], timeout_s)
    done = [s.get("done", 0) for s in seen]
    check(
        f"{label}: DINO features ready before training",
        final.get("state") == "ready"
        and final.get("done") == final.get("to_compute")
        and done == sorted(done)
        and all(d <= (final.get("to_compute") or 0) for d in done),
        f"{len(seen)} polls, done={done[:3]}..{done[-1:]} final={json.dumps(final)}",
    )
    snap["features"] = final
    return snap


# --------------------------------------------------------------------------- Studio side
def studio_rows(dsn: str, project: int) -> list[dict]:
    with psycopg.connect(dsn) as conn:
        cur = conn.execute(
            "SELECT id, source_model_id, pipeline, task, format, labels, metrics, sha256, "
            "bundle_key, size_bytes FROM finetune_model_import WHERE project_id=%s ORDER BY id",
            (project,),
        )
        cols = [c.name for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def open_bundle(data: bytes, row: dict) -> tuple[dict, dict, zipfile.ZipFile, str]:
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        names = tar.getnames()
        manifest = json.load(tar.extractfile("manifest.json"))
        artifact_name = next(n for n in names if n.startswith("model."))
        artifact = tar.extractfile(artifact_name).read()
        # Since 85b3f41 the bundle also carries the Engine-ready model when it converts.
        engine = (
            tar.extractfile("engine.onnx").read() if "engine.onnx" in names else None
        )
    digest = hashlib.sha256(artifact).hexdigest()
    problems = []
    extra = set(names) - {"manifest.json", artifact_name, "engine.onnx"}
    if extra or len(names) != len(set(names)):
        problems.append(f"tar members {names}")
    if engine is not None:
        import onnxruntime as ort

        try:
            ort.InferenceSession(engine, providers=["CPUExecutionProvider"])
        except Exception as exc:  # noqa: BLE001
            problems.append(f"engine.onnx does not load: {exc}")
    if not (digest == row["sha256"].strip() == manifest["sha256"]):
        problems.append(f"sha {digest[:12]} row {row['sha256'][:12]}")
    if len(data) != row["size_bytes"]:
        problems.append(f"size {len(data)} != {row['size_bytes']}")
    zf = zipfile.ZipFile(io.BytesIO(artifact))
    inner = json.loads(zf.read("manifest.json"))
    for n, h in inner["files"].items():
        if hashlib.sha256(zf.read(n)).hexdigest() != h:
            problems.append(f"inner file {n} hash mismatch")
    if inner["labels"] != manifest["labels"]:
        problems.append("labels differ between Studio manifest and service bundle")
    return manifest, inner, zf, "; ".join(problems)


def classify_accuracy(
    inner: dict, zf: zipfile.ZipFile, pipeline: str, embedder_path: Path
) -> tuple[int, int]:
    import onnxruntime as ort

    labels = inner["labels"]
    rng = random.Random(99)
    crops, truth = [], []
    for _ in range(40):
        im, boxes = scene(rng, 1)
        label, (x0, y0, x1, y1) = boxes[0]
        crops.append(im.crop((x0, y0, x1, y1)))
        truth.append(label)
    if pipeline == "dinov3_linear":
        from sparrow_finetune.embedders import OnnxEmbedder

        feats = OnnxEmbedder(embedder_path).embed(crops)
        sess = ort.InferenceSession(
            zf.read("head.onnx"), providers=["CPUExecutionProvider"]
        )
        head = inner["head"]
        probs = sess.run(
            [head["probabilities"]], {head["input"]: feats.astype(np.float32)}
        )[0]
    else:
        sess = ort.InferenceSession(
            zf.read("model.onnx"), providers=["CPUExecutionProvider"]
        )
        size = int((inner.get("input") or {}).get("size", 224))
        x = np.stack(
            [
                (
                    np.asarray(c.resize((size, size), Image.BILINEAR), np.float32)
                    / 255.0
                ).transpose(2, 0, 1)
                for c in crops
            ]
        )
        probs = np.concatenate(
            [
                sess.run(None, {sess.get_inputs()[0].name: x[i : i + 1]})[0]
                for i in range(len(x))
            ]
        )
    pred = [labels[int(i)] for i in np.argmax(probs, axis=1)]
    return sum(p == t for p, t in zip(pred, truth)), len(truth)


def letterbox(im: Image.Image, size: int) -> tuple[np.ndarray, float, int, int]:
    w, h = im.size
    r = min(size / w, size / h)
    nw, nh = round(w * r), round(h * r)
    left, top = round((size - nw) / 2 - 0.1), round((size - nh) / 2 - 0.1)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(im.resize((nw, nh), Image.BILINEAR), (left, top))
    x = (np.asarray(canvas, np.float32) / 255.0).transpose(2, 0, 1)[None]
    return x, r, left, top


def iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def detect_recall(inner: dict, zf: zipfile.ZipFile) -> tuple[int, int, str]:
    import onnxruntime as ort

    sess = ort.InferenceSession(
        zf.read("model.onnx"), providers=["CPUExecutionProvider"]
    )
    spec = inner["input"]
    size = int(spec["size"])
    labels = inner["labels"]
    rng = random.Random(1234)
    hit = total = 0
    shape = None
    for _ in range(20):
        im, boxes = scene(rng, 2)
        x, r, left, top = letterbox(im, size)
        out = sess.run(None, {spec["name"]: x})[0]
        shape = out.shape
        pred = out[0].T  # (anchors, 4 + nc)
        scores = pred[:, 4:]
        cls = scores.argmax(1)
        conf = scores.max(1)
        keep = conf > 0.25
        cxcywh = pred[keep, :4]
        cls, conf = cls[keep], conf[keep]
        xyxy = (
            np.stack(
                [
                    (cxcywh[:, 0] - cxcywh[:, 2] / 2 - left) / r,
                    (cxcywh[:, 1] - cxcywh[:, 3] / 2 - top) / r,
                    (cxcywh[:, 0] + cxcywh[:, 2] / 2 - left) / r,
                    (cxcywh[:, 1] + cxcywh[:, 3] / 2 - top) / r,
                ],
                1,
            )
            if len(cxcywh)
            else np.zeros((0, 4))
        )
        for label, gt in boxes:
            total += 1
            if any(
                labels[int(c)] == label and iou(gt, tuple(bx)) > 0.5
                for bx, c in zip(xyxy, cls)
            ):
                hit += 1
    return hit, total, f"output shape {shape}, nc={len(labels)}"


# --------------------------------------------------------------------------- compose additions
def extend_override(
    path: Path, work: Path, assets: Path, ports: dict, token: str, pg_password: str
) -> None:
    doc = json.loads(path.read_text())
    svc = doc["services"]
    in_net = azurite_conn("http://azurite:10000")
    svc["azurite"] = {
        "image": "mcr.microsoft.com/azure-storage/azurite:latest",
        "command": [
            "azurite-blob",
            "--blobHost",
            "0.0.0.0",
            "--skipApiVersionCheck",
            "--loose",
        ],
        "ports": [f"127.0.0.1:{ports['azurite']}:10000"],
    }
    blob = json.loads(json.dumps(svc["studio"]))
    blob["environment"].update(
        {
            "STORAGE_BACKEND": "azure_blob",
            "AZURE_STORAGE_CONNECTION_STRING": in_net,
            "AZURE_STORAGE_CONTAINER": BLOB_CONTAINER,
        }
    )
    blob["ports"] = [f"127.0.0.1:{ports['studio_blob']}:5000"]
    blob["depends_on"] = {
        "studio-db": {"condition": "service_healthy"},
        "azurite": {"condition": "service_started"},
    }
    svc["studio-blob"] = blob

    yolo_cwd = work / "yolo_cwd"
    worker_extra = {
        "volumes": ["ft-data:/data", f"{assets}:/assets:ro", f"{yolo_cwd}:/yolo"],
        "working_dir": "/yolo",
    }
    svc["worker-cpu"] = dict(worker_extra)
    svc["worker-gpu"] = dict(worker_extra)
    svc["api"] = {"volumes": ["ft-data:/data", f"{assets}:/assets:ro"]}

    def env(studio_url: str, studio_token: str) -> dict:
        return {
            "SPARROW_FT_DATABASE_URL": "postgresql://sparrow:${POSTGRES_PASSWORD}@postgres:5432/sparrow_finetune",
            "SPARROW_FT_DATA_DIR": "/data",
            "SPARROW_FT_API_TOKENS": "${SPARROW_FT_API_TOKENS}",
            "SPARROW_FT_AUTH_DISABLED": "0",
            "SPARROW_FT_EMBEDDER": "${SPARROW_FT_EMBEDDER}",
            "SPARROW_FT_STUDIO_URL": studio_url,
            "SPARROW_FT_STUDIO_TOKEN": studio_token,
            "SPARROW_FT_COOKIE_SECURE": "0",
        }

    health = {
        "test": [
            "CMD",
            "python",
            "-c",
            "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen("
            "'http://127.0.0.1:8000/v1/health', timeout=5).status == 200 else 1)",
        ],
        "interval": "5s",
        "retries": 30,
    }
    variants = {
        "api-blob": ("http://studio-blob:5000", token, ports["api_blob"]),
        "api-noconf": ("", "", ports["api_noconf"]),
        "api-badtoken": (
            "http://studio:5000",
            "wrong-" + secrets.token_hex(8),
            ports["api_bad"],
        ),
    }
    for name, (url, tok, port) in variants.items():
        svc[name] = {
            "image": "sparrow-finetune:${SPARROW_FT_IMAGE_TAG:-latest}",
            "command": ["api", "--host", "0.0.0.0", "--port", "8000"],
            "environment": env(url, tok),
            "volumes": ["ft-data:/data", f"{assets}:/assets:ro"],
            "ports": [f"127.0.0.1:{port}:8000"],
            "healthcheck": health,
            "depends_on": {"migrate": {"condition": "service_completed_successfully"}},
        }
    path.write_text(json.dumps(doc, indent=2))


# --------------------------------------------------------------------------- scenarios
class Ctx:
    pass


def scenario_classification_sync(c: Ctx) -> None:
    api, s, p = c.api, c.seed, c.p
    ws = c.ws
    ds = api.ok(
        "POST",
        f"/workspaces/{ws}/datasets",
        {"name": "A crops", "task": "classification", "whole_image": True},
    )["id"]
    c.ds_a = ds

    # reviewed_only=false: every exported image, reviewed -> confirmed, others pending.
    res = api.ok(
        "POST",
        "/connectors/studio/import",
        {"project": str(p["A"]), "dataset_id": ds, "reviewed_only": False},
    )
    exp = s.expect(p["A"], False, "classification")
    check(
        "A import reviewed_only=false: counts (2 pages, empty images skipped)",
        import_counts(res)
        == {
            k: exp[k] for k in ("imported_images", "imported_items", "skipped", "pages")
        },
        f"got {import_counts(res)} expected {[exp[k] for k in ('imported_images', 'imported_items', 'skipped', 'pages')]}",
    )
    got = actual_items(api, ds, "classification")
    d = diff_items(exp["items"], got)
    check(
        "A items: crop names, groups, labels and statuses match the Studio rows "
        "(dup filenames, null filename, deleted box, blocked and gallery-less images excluded)",
        not d,
        d or f"{len(got)} items",
    )

    # Studio changes: two unreviewed images get reviewed, one reviewed box is relabelled.
    unrev = [
        r for r in s.images[p["A"]] if r["name"] and r["name"].startswith("unrev")
    ][:2]
    for r in unrev:
        s.conn.execute(
            "UPDATE images SET review_status='reviewed' WHERE id=%s", (r["id"],)
        )
        r["reviewed"] = True
    target = next(r for r in s.images[p["A"]] if r["name"] == "cam005.jpg")
    box = target["boxes"][0]
    new_label = "zebra" if box["label"] == "lion" else "lion"
    s.conn.execute(
        "UPDATE detection_boxes SET classifier_class=%s WHERE id=%s",
        (new_label, box["id"]),
    )
    box["label"] = new_label

    res = api.ok(
        "POST",
        "/connectors/studio/import",
        {"project": str(p["A"]), "dataset_id": ds, "reviewed_only": True},
    )
    exp_r = s.expect(p["A"], True, "classification")
    check(
        "A re-import reviewed_only=true: counts",
        import_counts(res)
        == {
            k: exp_r[k]
            for k in ("imported_images", "imported_items", "skipped", "pages")
        },
        str(import_counts(res)),
    )
    exp_all = s.expect(p["A"], False, "classification")
    got = actual_items(api, ds, "classification")
    d = diff_items(exp_all["items"], got)
    check(
        "A after re-import: no duplicates, newly reviewed items promoted pending->confirmed, "
        "relabelled box updated, still-unreviewed items stay pending",
        not d,
        d
        or f"{len(got)} items, pending={sum(v['status'] == 'pending' for v in got.values())}",
    )

    snap1 = api.ok("POST", f"/datasets/{ds}/snapshots", {})
    res = api.ok(
        "POST",
        "/connectors/studio/import",
        {"project": str(p["A"]), "dataset_id": ds, "reviewed_only": True},
    )
    snap2 = api.ok("POST", f"/datasets/{ds}/snapshots", {})
    check(
        "unchanged re-import is idempotent (same snapshot content_hash and item count)",
        snap1["content_hash"] == snap2["content_hash"]
        and snap1["item_count"] == snap2["item_count"],
        f"{snap1['item_count']} items, hash {snap1['content_hash'][:12]}",
    )

    # Exclude an item; a re-import must not bring it back.
    victim = next(n for n, v in got.items() if v["status"] == "confirmed")
    api.ok("PATCH", f"/datasets/{ds}/items/{victim}", {"review_status": "excluded"})
    api.ok(
        "POST",
        "/connectors/studio/import",
        {"project": str(p["A"]), "dataset_id": ds, "reviewed_only": True},
    )
    status = items_of(api, ds)[victim]["review_status"]
    snap3 = api.ok("POST", f"/datasets/{ds}/snapshots", {})
    check(
        "excluded item stays excluded after re-import and leaves the snapshot",
        status == "excluded" and snap3["item_count"] == snap2["item_count"] - 1,
        f"{victim}: {status}; snapshot {snap2['item_count']} -> {snap3['item_count']}",
    )

    # Manual uploads next to imported items.
    rng = random.Random(5)
    manual = []
    for i in range(6):
        im, boxes = scene(rng, 1)
        label, bb = boxes[0]
        buf = io.BytesIO()
        im.crop(bb).save(buf, format="PNG")
        name = f"manual_{i}.png"
        api.ok("PUT", f"/datasets/{ds}/images/{name}", raw=buf.getvalue())
        manual.append(
            {"image": name, "annotations": [{"label": label}], "group": "manual"}
        )
    api.ok("POST", f"/datasets/{ds}/annotations", {"items": manual})
    snap4 = api.ok("POST", f"/datasets/{ds}/snapshots", {})
    api.ok(
        "POST",
        "/connectors/studio/import",
        {"project": str(p["A"]), "dataset_id": ds, "reviewed_only": True},
    )
    names = items_of(api, ds)
    check(
        "manual uploads coexist with imported items and survive a re-import",
        snap4["item_count"] == snap3["item_count"] + 6
        and all(m["image"] in names for m in manual),
        f"snapshot {snap3['item_count']} -> {snap4['item_count']}",
    )
    c.snap_a = snap4
    c.ds_a = ds
    c.label_counts_a = snap4["label_counts"]


def scenario_failed_imports(c: Ctx) -> None:
    """Every failing import leaves the dataset as it was."""
    api, s, p = c.api, c.seed, c.p
    before = items_of(api, c.ds_a)

    # A missing image file part-way through the export.
    rec = next(r for r in s.images[p["A"]] if r["name"] == "cam110.jpg")
    path = c.gallery / rec["key"]
    hidden = path.with_suffix(".hidden")
    path.rename(hidden)
    try:
        st, body = api.call(
            "POST",
            "/connectors/studio/import",
            {"project": str(p["A"]), "dataset_id": c.ds_a, "reviewed_only": True},
        )
        fresh = api.ok(
            "POST",
            f"/workspaces/{c.ws}/datasets",
            {"name": "A fresh", "task": "classification"},
        )["id"]
        st2, _ = api.call(
            "POST",
            "/connectors/studio/import",
            {"project": str(p["A"]), "dataset_id": fresh, "reviewed_only": True},
        )
        n_fresh = len(items_of(api, fresh))
    finally:
        hidden.rename(path)
    check(
        "an image missing in Studio (page 2) fails the import with 502 and writes nothing",
        st == 502 and st2 == 502 and items_of(api, c.ds_a) == before and n_fresh == 0,
        f"{st}/{st2}: {str(body)[:160]}; fresh dataset items={n_fresh}",
    )

    # Unknown Studio project: Studio refuses it (502). Malformed id: compose sets
    # SPARROW_FT_STUDIO_PROJECT_IDS=numeric, so the service refuses it first (422, FT-10).
    for proj, label, want in (
        ("999999", "unknown project id", 502),
        ("not-a-number", "non-numeric project", 422),
    ):
        st, body = api.call(
            "POST", "/connectors/studio/import", {"project": proj, "dataset_id": c.ds_a}
        )
        check(
            f"import from {label} -> {want}, dataset unchanged",
            st == want and items_of(api, c.ds_a) == before,
            f"{st}: {str(body)[:160]}",
        )
    st, _ = api.call(
        "POST", "/connectors/studio/import", {"project": "   ", "dataset_id": c.ds_a}
    )
    check("import with a blank project -> 422", st == 422, str(st))
    st, _ = api.call(
        "POST",
        "/connectors/studio/import",
        {"project": str(p["A"]), "dataset_id": 987654},
    )
    check("import into an unknown dataset -> 404", st == 404, str(st))

    # Blob-only images through the local-storage Studio: every image 404s.
    blob_ds = api.ok(
        "POST",
        f"/workspaces/{c.ws}/datasets",
        {"name": "C via local Studio", "task": "classification"},
    )["id"]
    st, body = api.call(
        "POST",
        "/connectors/studio/import",
        {"project": str(p["C"]), "dataset_id": blob_ds},
    )
    check(
        "project whose images live only in blob storage, through the local-disk Studio -> 502, nothing written",
        st == 502 and not items_of(api, blob_ds),
        f"{st}: {str(body)[:160]}",
    )


def scenario_detection_and_labels(c: Ctx) -> None:
    api, s, p = c.api, c.seed, c.p
    ds = api.ok(
        "POST", f"/workspaces/{c.ws}/datasets", {"name": "B boxes", "task": "detection"}
    )["id"]
    c.ds_b = ds
    res = api.ok(
        "POST",
        "/connectors/studio/import",
        {"project": str(p["B"]), "dataset_id": ds, "reviewed_only": True},
    )
    exp = s.expect(p["B"], True, "detection")
    check(
        "B detection import: counts",
        import_counts(res)
        == {
            k: exp[k] for k in ("imported_images", "imported_items", "skipped", "pages")
        },
        str(import_counts(res)),
    )
    got = actual_items(api, ds, "detection")
    d = diff_items(exp["items"], got)
    check(
        "B detection items: whole images, xywh boxes and labels equal the Studio boxes",
        not d,
        d or f"{len(got)} images",
    )
    dims = {
        it["image"]: (it["width"], it["height"])
        for it in api.ok("GET", f"/datasets/{ds}/items")["items"]
    }
    check(
        "B detection items carry the image size",
        set(dims.values()) == {(640, 480)},
        str(set(dims.values())),
    )

    res = api.ok(
        "POST",
        "/connectors/studio/import",
        {"project": str(p["B"]), "dataset_id": ds, "reviewed_only": False},
    )
    exp = s.expect(p["B"], False, "detection")
    got = actual_items(api, ds, "detection")
    d = diff_items(exp["items"], got)
    check(
        "B re-import reviewed_only=false adds the unreviewed images as pending",
        not d,
        d or str(import_counts(res)),
    )
    c.snap_b = api.ok("POST", f"/datasets/{ds}/snapshots", {})

    # D: label precedence, into a detection dataset (one box per image).
    dd = api.ok(
        "POST",
        f"/workspaces/{c.ws}/datasets",
        {"name": "D labels", "task": "detection"},
    )["id"]
    res = api.ok(
        "POST", "/connectors/studio/import", {"project": str(p["D"]), "dataset_id": dd}
    )
    exp = s.expect(p["D"], True, "detection")
    # A detection image with a box that resolves to 'unknown' is skipped, not imported with an
    # 'unknown' class: dropping only that box would teach the detector the animal is background.
    unknown = {n for n, v in exp["items"].items() if v["annotations"][0][0] == "unknown"}
    want = {n: v for n, v in exp["items"].items() if n not in unknown}
    got = actual_items(api, dd, "detection")
    labels = {n: v["annotations"][0][0] for n, v in got.items()}
    d = diff_items(want, got)
    check(
        "D label precedence: classifier_class > classifier_class_name > detector_class_name > label",
        not d and len(want) == 4,
        d or json.dumps(labels),
    )
    skipped = (res.get("label_handling") or {}).get("images_skipped_unknown")
    check(
        "D images whose box resolves to 'unknown' are skipped and counted",
        len(unknown) == 2 and skipped == 2 and not unknown & set(got),
        json.dumps(res.get("label_handling")),
    )

    # F: exactly 100 exported images is one page.
    fd = api.ok(
        "POST",
        f"/workspaces/{c.ws}/datasets",
        {"name": "F page", "task": "classification"},
    )["id"]
    res = api.ok(
        "POST", "/connectors/studio/import", {"project": str(p["F"]), "dataset_id": fd}
    )
    check(
        "F: exactly 100 images import in one page",
        import_counts(res)
        == {"imported_images": 100, "imported_items": 100, "skipped": 0, "pages": 1},
        str(import_counts(res)),
    )


def scenario_train(c: Ctx) -> None:
    api = c.api
    # Task mismatches are refused before queueing.
    for pipeline, snap, what in (
        ("yolo_detector", c.snap_a["id"], "yolo_detector on a classification snapshot"),
        ("dinov3_linear", c.snap_b["id"], "dinov3_linear on a detection snapshot"),
        (
            "conventional_classifier",
            c.snap_b["id"],
            "conventional_classifier on a detection snapshot",
        ),
    ):
        st, body = api.call(
            "POST", "/jobs", {"pipeline": pipeline, "snapshot_id": snap, "params": {}}
        )
        check(f"job refused: {what} -> 422", st == 422, f"{st}: {str(body)[:120]}")
    # dinov3_linear trains only on prepared features: a plain snapshot is refused, naming Data
    # Preparation; an opt-in snapshot is prepared (progress polled) before the job is submitted.
    st, body = api.call(
        "POST",
        "/jobs",
        {"pipeline": "dinov3_linear", "snapshot_id": c.snap_a["id"], "params": {}},
    )
    check(
        "job refused: dinov3_linear on an unprepared snapshot -> 422 (Data Preparation)",
        st == 422 and "Data Preparation" in str(body),
        f"{st}: {str(body)[:160]}",
    )
    c.snap_dino = prepare_dino_snapshot(
        api, c.ds_a, c.train_timeout, "dinov3_linear Studio data"
    )

    jobs = {
        "conventional_classifier": api.ok(
            "POST",
            "/jobs",
            {
                "pipeline": "conventional_classifier",
                "snapshot_id": c.snap_a["id"],
                "params": {"epochs": 8, "warmup_epochs": 1},
            },
        )["id"],
        "dinov3_linear": api.ok(
            "POST",
            "/jobs",
            {
                "pipeline": "dinov3_linear",
                "snapshot_id": c.snap_dino["id"],
                "params": {},
            },
        )["id"],
        "yolo_detector": api.ok(
            "POST",
            "/jobs",
            {
                "pipeline": "yolo_detector",
                "snapshot_id": c.snap_b["id"],
                "params": {
                    "model": "yolo11n.pt",
                    "epochs": 50,
                    "imgsz": 320,
                    "batch": 8,
                },
            },
        )["id"],
    }
    c.models = {}
    for pipeline, jid in jobs.items():
        job = wait_job(api, jid, c.train_timeout)
        models = api.ok("GET", f"/models?job_id={jid}")["models"]
        ok = job["state"] == "succeeded" and len(models) == 1
        check(
            f"{pipeline} trains on imported Studio data",
            ok,
            f"job {jid} {job['state']}, ran {job_seconds(job)}"
            + (
                f" metrics={json.dumps(models[0]['metrics'])[:200]}"
                if models
                else f" error={str(job.get('error'))[:300]}"
            ),
        )
        if ok:
            c.models[pipeline] = models[0]


def verify_pushed(
    c: Ctx, project: int, studio_id: str, model: dict, task: str, *, blob=False
) -> dict | None:
    rows = [r for r in studio_rows(c.dsn, project) if str(r["id"]) == str(studio_id)]
    if not rows:
        check(f"Studio row {studio_id} exists in project {project}", False, "")
        return None
    row = rows[0]
    if blob:
        key = row["bundle_key"]
        data = c.blob.get_blob_client(key).download_blob().readall()
        check(
            "blob Studio stored the bundle in the container under finetune_models/",
            key.startswith("finetune_models/"),
            key,
        )
    else:
        data = (c.models_dir / row["bundle_key"]).read_bytes()
    manifest, inner, zf, problems = open_bundle(data, row)
    check(
        f"project {project}: row {studio_id} for model {model['id']} ({model['pipeline']}) matches the bundle",
        not problems
        and row["source_model_id"] == str(model["id"])
        and row["pipeline"] == model["pipeline"]
        and row["task"] == task
        and row["sha256"].strip() == model["sha256"]
        and sorted(row["labels"]) == sorted(inner["labels"]),
        problems
        or f"{row['pipeline']}/{row['task']} sha {row['sha256'][:12]} {row['size_bytes']} bytes",
    )
    return {"row": row, "manifest": manifest, "inner": inner, "zf": zf}


def scenario_model_sync(c: Ctx) -> None:
    api, p = c.api, c.p
    conv = c.models.get("conventional_classifier")
    dino = c.models.get("dinov3_linear")
    yolo = c.models.get("yolo_detector")
    a = str(p["A"])

    if conv:
        out = api.ok(
            "POST", "/connectors/studio/push", {"model_id": conv["id"], "project": a}
        )
        sid = out["studio_model_id"]
        got = verify_pushed(c, p["A"], sid, conv, "classification")
        if got:
            hit, n = classify_accuracy(
                got["inner"], got["zf"], "conventional_classifier", c.embedder
            )
            check(
                "pushed conventional_classifier ONNX classifies fresh crops (>= 90%)",
                hit / n >= 0.9,
                f"{hit}/{n}",
            )
        again = api.ok(
            "POST", "/connectors/studio/push", {"model_id": conv["id"], "project": a}
        )["studio_model_id"]
        check(
            "re-pushing the same model is an upsert (same Studio id, one row)",
            again == sid and len(studio_rows(c.dsn, p["A"])) == 1,
            f"{sid} / {again}",
        )
        dep = api.ok(
            "POST",
            f"/models/{conv['id']}/activate",
            {"consumer": "studio", "project": a},
        )
        check(
            "activate in Studio pushes and records the Studio id on the deployment",
            dep["external_id"] == sid and dep["deactivated_at"] is None,
            json.dumps(dep, default=str)[:200],
        )
        c.dep_conv = dep

    if dino:
        dep = api.ok(
            "POST",
            f"/models/{dino['id']}/activate",
            {"consumer": "studio", "project": a},
        )
        rows = studio_rows(c.dsn, p["A"])
        got = verify_pushed(c, p["A"], dep["external_id"], dino, "classification")
        check(
            "a second model version adds a second Studio row",
            len(rows) == (2 if conv else 1),
            f"{len(rows)} rows",
        )
        if got:
            hit, n = classify_accuracy(
                got["inner"], got["zf"], "dinov3_linear", c.embedder
            )
            check(
                "pushed dinov3_linear head + DINOv3 ONNX embedder classifies fresh crops (>= 90%)",
                hit / n >= 0.9,
                f"{hit}/{n}",
            )
        active = api.ok("GET", f"/deployments?consumer=studio&project={a}&active=true")
        active = active["deployments"] if isinstance(active, dict) else active
        check(
            "activating the second version deactivates the first for the same Studio project",
            [d["model_id"] for d in active] == [dino["id"]],
            f"active={[d['model_id'] for d in active]}",
        )
        if conv:
            m1 = api.ok("GET", f"/models/{conv['id']}")
            local = api.ok(
                "POST",
                f"/models/{conv['id']}/activate",
                {"consumer": "local", "project": a},
            )
            all_active = api.ok("GET", "/deployments?active=true")
            all_active = (
                all_active["deployments"]
                if isinstance(all_active, dict)
                else all_active
            )
            pairs = sorted(
                (d["consumer"], d["project"], d["model_id"])
                for d in all_active
                if d["project"] == a
            )
            check(
                "local and Studio activations are independent",
                pairs == sorted([("local", a, conv["id"]), ("studio", a, dino["id"])])
                and local["external_id"] is None,
                f"{pairs}; model {conv['id']} active_deployments before={m1.get('active_deployments')}",
            )
            m2 = api.ok("GET", f"/models/{dino['id']}")
            check(
                "model detail lists its active deployment",
                [
                    (d["consumer"], d["project"])
                    for d in m2.get("active_deployments", [])
                ]
                == [("studio", a)],
                str(m2.get("active_deployments")),
            )
        # One model, two Studio projects.
        f_id = api.ok(
            "POST",
            "/connectors/studio/push",
            {"model_id": dino["id"], "project": str(p["F"])},
        )["studio_model_id"]
        verify_pushed(c, p["F"], f_id, dino, "classification")
        check(
            "one model pushed to two projects gives one row in each",
            len(studio_rows(c.dsn, p["F"])) == 1 and f_id != dep["external_id"],
            f"A:{dep['external_id']} F:{f_id}",
        )

    if yolo:
        dep = api.ok(
            "POST",
            f"/models/{yolo['id']}/activate",
            {"consumer": "studio", "project": str(p["B"])},
        )
        got = verify_pushed(c, p["B"], dep["external_id"], yolo, "detection")
        if got:
            hit, n, info = detect_recall(got["inner"], got["zf"])
            check(
                "pushed yolo_detector ONNX finds fresh boxes (>= 70% matched at IoU > 0.5 with the right label)",
                hit / n >= 0.7,
                f"{hit}/{n}; {info}",
            )

    # Unknown model.
    st, _ = api.call(
        "POST", "/connectors/studio/push", {"model_id": 987654, "project": a}
    )
    check("push of an unknown model -> 404", st == 404, str(st))
    st, _ = api.call(
        "POST", "/models/987654/activate", {"consumer": "studio", "project": a}
    )
    check("activate of an unknown model -> 404", st == 404, str(st))
    if conv:
        st, body = api.call(
            "POST",
            "/connectors/studio/push",
            {"model_id": conv["id"], "project": "999999"},
        )
        check(
            "push into an unknown Studio project -> 502",
            st == 502,
            f"{st}: {str(body)[:120]}",
        )


def scenario_blob(c: Ctx) -> None:
    api, p, s = c.api_blob, c.p, c.seed
    ds = api.ok(
        "POST",
        f"/workspaces/{c.ws}/datasets",
        {"name": "C blob crops", "task": "classification"},
    )["id"]
    res = api.ok(
        "POST", "/connectors/studio/import", {"project": str(p["C"]), "dataset_id": ds}
    )
    exp = s.expect(p["C"], True, "classification")
    got = actual_items(api, ds, "classification")
    d = diff_items(exp["items"], got)
    check(
        "Studio on Azure Blob (Azurite): import reads every image from the container",
        not d and import_counts(res)["imported_items"] == exp["imported_items"],
        d or str(import_counts(res)),
    )
    model = c.models.get("conventional_classifier") or next(
        iter(c.models.values()), None
    )
    if model:
        dep = api.ok(
            "POST",
            f"/models/{model['id']}/activate",
            {"consumer": "studio", "project": str(p["C"])},
        )
        verify_pushed(
            c,
            p["C"],
            dep["external_id"],
            model,
            model["metadata"].get("task")
            or (
                "detection"
                if model["pipeline"] == "yolo_detector"
                else "classification"
            ),
            blob=True,
        )


def scenario_connector_failures(c: Ctx) -> None:
    p = c.p
    a = str(p["A"])
    model = next(iter(c.models.values()), None)
    before = items_of(c.api, c.ds_a)

    st, body = c.api_noconf.call("GET", "/connectors/studio")
    check(
        "connector not configured: status says so",
        st == 200 and body == {"configured": False, "url": ""},
        str(body),
    )
    st, _ = c.api_noconf.call(
        "POST", "/connectors/studio/import", {"project": a, "dataset_id": c.ds_a}
    )
    st2 = (
        c.api_noconf.call(
            "POST", "/connectors/studio/push", {"model_id": model["id"], "project": a}
        )[0]
        if model
        else 409
    )
    st3 = (
        c.api_noconf.call(
            "POST",
            f"/models/{model['id']}/activate",
            {"consumer": "studio", "project": a},
        )[0]
        if model
        else 409
    )
    check(
        "connector not configured: import, push and Studio activation -> 409",
        (st, st2, st3) == (409, 409, 409),
        f"{st}/{st2}/{st3}",
    )
    if model:
        dep = c.api_noconf.ok(
            "POST",
            f"/models/{model['id']}/activate",
            {"consumer": "local", "project": "noconf"},
        )
        check(
            "connector not configured: local activation still works",
            dep["consumer"] == "local",
            "",
        )

    st, body = c.api_bad.call(
        "POST", "/connectors/studio/import", {"project": a, "dataset_id": c.ds_a}
    )
    st2, body2 = (
        c.api_bad.call(
            "POST", "/connectors/studio/push", {"model_id": model["id"], "project": a}
        )
        if model
        else (502, "")
    )
    check(
        "wrong Studio token: import and push -> 502, dataset unchanged",
        st == 502 and st2 == 502 and items_of(c.api, c.ds_a) == before,
        f"{st} {str(body)[:100]} / {st2} {str(body2)[:100]}",
    )

    unauth = Api(c.api.base[:-3], None)
    st, _ = unauth.call(
        "POST", "/connectors/studio/import", {"project": a, "dataset_id": c.ds_a}
    )
    st2, _ = unauth.call("GET", "/connectors/studio")
    check("service API without a token -> 401", (st, st2) == (401, 401), f"{st}/{st2}")

    # Studio down.
    active_before = c.api.ok(
        "GET", f"/deployments?consumer=studio&project={a}&active=true"
    )
    compose(c.project, c.files, c.env_file, "stop", "studio")
    try:
        st, body = c.api.call(
            "POST", "/connectors/studio/import", {"project": a, "dataset_id": c.ds_a}
        )
        st2 = (
            c.api.call(
                "POST",
                "/connectors/studio/push",
                {"model_id": model["id"], "project": a},
            )[0]
            if model
            else 502
        )
        st3 = (
            c.api.call(
                "POST",
                f"/models/{model['id']}/activate",
                {"consumer": "studio", "project": a},
            )[0]
            if model
            else 502
        )
        active_after = c.api.ok(
            "GET", f"/deployments?consumer=studio&project={a}&active=true"
        )
        check(
            "Studio down: import, push and activation -> 502; dataset and active deployment unchanged",
            (st, st2, st3) == (502, 502, 502)
            and items_of(c.api, c.ds_a) == before
            and active_after == active_before,
            f"{st}/{st2}/{st3}: {str(body)[:120]}",
        )
    finally:
        r = compose(
            c.project,
            c.files,
            c.env_file,
            "up",
            "-d",
            "--wait",
            "--no-build",
            "studio",
            check_rc=False,
        )
    res = c.api.call(
        "POST", "/connectors/studio/import", {"project": a, "dataset_id": c.ds_a}
    )
    check(
        "Studio back up: import works again",
        r.returncode == 0 and res[0] == 200,
        f"{res[0]}",
    )


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--studio-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--assets", type=Path, default=Path("/tmp/ft_e2e_full_assets"))
    ap.add_argument("--train-timeout", type=int, default=2400)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    from azure.storage.blob import BlobServiceClient

    studio_root = args.studio_root.resolve()
    out = args.out.resolve()
    assets = args.assets.resolve()
    for f in ("dinov3_vits16.onnx", "dinov3_vits16.onnx.data", "yolo11n.pt"):
        if not (assets / f).is_file():
            raise SystemExit(f"missing asset {assets / f}")
    work = out / "work"
    gallery, models_dir = work / "gallery", work / "studio_models"
    for d in (
        gallery,
        models_dir,
        work / "storage",
        work / "staging",
        work / "yolo_cwd",
    ):
        d.mkdir(parents=True, exist_ok=True)
    shutil.copy2(assets / "yolo11n.pt", work / "yolo_cwd" / "yolo11n.pt")
    os.chmod(work / "yolo_cwd", 0o777)
    os.chmod(work / "yolo_cwd" / "yolo11n.pt", 0o644)

    token = secrets.token_urlsafe(32)
    api_token = secrets.token_urlsafe(32)
    pg_password = secrets.token_urlsafe(16)
    project = f"ft-e2e-full-{secrets.token_hex(3)}"
    env_file = work / "service.env"
    override = work / "compose.e2e.yml"
    files = [REPO / "compose.service.yml", override]
    ports = {
        k: free_port()
        for k in (
            "pg",
            "studio",
            "studio_blob",
            "azurite",
            "ft",
            "api_blob",
            "api_noconf",
            "api_bad",
        )
    }

    info: dict = {
        "studio_branch": run(
            ["git", "-C", str(studio_root), "rev-parse", "--abbrev-ref", "HEAD"]
        ).stdout.strip(),
        "studio_commit": run(
            ["git", "-C", str(studio_root), "rev-parse", "HEAD"]
        ).stdout.strip(),
        "finetune_commit": run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"]
        ).stdout.strip(),
    }
    print(
        f"Studio {info['studio_branch']} @ {info['studio_commit'][:9]}; fine-tune @ {info['finetune_commit'][:9]}",
        flush=True,
    )

    env_file.write_text(
        "\n".join(
            [
                f"POSTGRES_PASSWORD={secrets.token_urlsafe(16)}",
                f"SPARROW_FT_PORT={ports['ft']}",
                "SPARROW_FT_STUDIO_URL=http://studio:5000",
                f"SPARROW_FT_STUDIO_TOKEN={token}",
                f"SPARROW_FT_API_TOKENS={api_token}",
                "SPARROW_FT_AUTH_DISABLED=0",
                "SPARROW_FT_EMBEDDER=onnx:/assets/dinov3_vits16.onnx",
            ]
        )
        + "\n"
    )
    os.chmod(env_file, 0o600)
    write_override(
        override, studio_root, work, ports["pg"], ports["studio"], pg_password, token
    )
    extend_override(override, work, assets, ports, token, pg_password)
    os.chmod(override, 0o600)
    dsn = f"host=127.0.0.1 port={ports['pg']} user=postgres password={pg_password} dbname=postgres"

    c = Ctx()
    c.dsn, c.gallery, c.models_dir, c.embedder = (
        dsn,
        gallery,
        models_dir,
        assets / "dinov3_vits16.onnx",
    )
    c.project, c.files, c.env_file, c.train_timeout = (
        project,
        files,
        env_file,
        args.train_timeout,
    )
    t_start = time.time()
    try:
        r = compose(
            project,
            files,
            env_file,
            "up",
            "-d",
            "--wait",
            "studio-db",
            "azurite",
            check_rc=False,
        )
        if r.returncode:
            raise RuntimeError(f"studio-db/azurite did not start: {r.stderr[-800:]}")
        svc = BlobServiceClient.from_connection_string(
            azurite_conn(f"http://127.0.0.1:{ports['azurite']}")
        )
        container = svc.get_container_client(BLOB_CONTAINER)
        for _ in range(30):
            try:
                container.create_container()
                break
            except Exception as exc:  # noqa: BLE001
                if "ContainerAlreadyExists" in str(exc):
                    break
                time.sleep(1)
        c.blob = container
        c.seed, c.p = seed_all(
            dsn, gallery, studio_root / "db/schema/01_schema.sql", container
        )
        info["projects"] = c.p
        check(
            "Studio schema loaded; projects A-D, F seeded (C into Azurite)",
            True,
            json.dumps(c.p),
        )

        r = compose(
            project, files, env_file, "up", "-d", "--wait", "--no-build", check_rc=False
        )
        (out / "compose_up.log").write_text(r.stdout + r.stderr)
        check(
            "compose up: 2 Studios (disk, blob), Azurite, service stack + 3 extra API variants healthy",
            r.returncode == 0,
            project,
        )
        if r.returncode:
            raise RuntimeError(
                "compose up failed; see compose_up.log and compose_logs.txt"
            )

        c.api = Api(f"http://127.0.0.1:{ports['ft']}", api_token)
        c.api_blob = Api(f"http://127.0.0.1:{ports['api_blob']}", api_token)
        c.api_noconf = Api(f"http://127.0.0.1:{ports['api_noconf']}", api_token)
        c.api_bad = Api(f"http://127.0.0.1:{ports['api_bad']}", api_token)
        check(
            "connector status: configured on api and api-blob",
            c.api.ok("GET", "/connectors/studio")["configured"]
            and c.api_blob.ok("GET", "/connectors/studio")["configured"],
            "",
        )
        c.ws = c.api.ok("POST", "/workspaces", {"name": "Studio full e2e"})["id"]

        for name, fn in (
            ("classification sync", scenario_classification_sync),
            ("failed imports", scenario_failed_imports),
            ("detection, labels and paging", scenario_detection_and_labels),
            ("training", scenario_train),
            ("model sync", scenario_model_sync),
            ("blob storage", scenario_blob),
            ("connector failures", scenario_connector_failures),
        ):
            print(f"== {name}", flush=True)
            try:
                fn(c)
            except Exception as exc:  # noqa: BLE001
                check(
                    f"scenario '{name}' completed without an exception",
                    False,
                    f"{type(exc).__name__}: {exc}",
                )
    except Exception as exc:  # noqa: BLE001
        check(
            "harness completed without an exception",
            False,
            f"{type(exc).__name__}: {exc}",
        )
    finally:
        logs = compose(project, files, env_file, "logs", "--no-color", check_rc=False)
        (out / "compose_logs.txt").write_text(logs.stdout + logs.stderr)
        if not args.keep:
            compose(project, files, env_file, "down", "-v", check_rc=False)
            env_file.unlink(missing_ok=True)
            override.unlink(missing_ok=True)

    info["duration_s"] = round(time.time() - t_start)
    info["checks"] = [{"name": n, "pass": ok, "detail": d} for n, ok, d in CHECKS]
    (out / "result.json").write_text(json.dumps(info, indent=2, default=str))
    failed = [n for n, ok, _ in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
