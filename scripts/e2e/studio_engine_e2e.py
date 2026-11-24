"""End-to-end test: real camera-trap data -> GPU fine-tuning -> Studio runs the fine-tuned model in Sparrow Engine.

One throwaway compose project on this machine:

* Studio: ``studio-db`` (Postgres with the branch schema), ``studio`` (the dashboard, which receives
  bundles), ``model-download-worker`` (the branch's fine-tune installer), ``client-worker`` (the
  branch's inference worker) and ``sparrow-engine`` (Sparrow Engine GPU, the image Studio pins).
  The two workers run the branch's live ``client/worker`` code mounted over the client image.
* The fine-tune service from ``compose.service.yml`` (API + GPU worker; the CPU worker is stopped
  so every job trains on the GPU).

Flow, with real images from the Amazon camera-trap dataset (5 species):

1. A Studio project with the stock pipeline (MDV6 detector + AI4G-Amazon classifier) runs its
   inference worker over the training images. The harness then plays reviewer: the top box of
   each image gets the ground-truth species, other boxes are deleted, the image is "reviewed".
   A second project runs the stock pipeline over the validation images as the baseline.
2. The service imports the reviewed project (crops and boxes), snapshots it and trains
   ``conventional_classifier``, ``dinov3_linear`` and ``yolo_detector`` on the GPU.
3. Each model is activated in Studio. The installer stages it into ``/models``, load-probes it in
   the Engine and switches the project's pipeline. New validation images uploaded to the project
   are then classified (or detected) by the fine-tuned model through Studio's own worker.
4. Error path: a corrupted bundle fails the install and leaves the live pipeline in place;
   restoring it and activating again installs it.
5. Combinations: the installed species detector imports back as a base model and warm-starts a
   new detector (batch=auto); a classifier is refused on top of the species detector; a generic
   "animal" detector trained on the same boxes gates the classifier; re-activating an
   already-installed model.

Usage (from the fine-tune repo root; ``sparrow-finetune:latest`` and the Studio client image
must be built):

    uv run --extra dev --extra e2e python scripts/e2e/studio_engine_e2e.py \
        --studio-root <Studio checkout> --out <artifact dir>
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import secrets
import shutil
import stat
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import quote

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parent))
from studio_branch_e2e import (  # noqa: E402
    CHECKS,
    REPO,
    check,
    compose,
    free_port,
    run,
    write_override,
)
from studio_full_e2e import (  # noqa: E402
    Api,
    job_seconds,
    prepare_dino_snapshot,
    wait_job,
)

SPECIES = ("Dasyprocta", "Bos", "Pecari", "Mazama", "Cuniculus")
DETECTOR, CLASSIFIER = "MDV6-yolov10-c", "AI4G-Amazon-V2"
ENGINE_IMAGE = "zhongqimiao/sparrow-engine-server-gpu:v0.1.24"
CLIENT_IMAGE = "ft-e2e-studio-client:local"
ABSTAIN = "unknown"  # Studio's default abstain label (client/worker/abstain.py)
WEIGHTS = Path.home() / ".local/share/Trash/files/sparrow-engine-models-v0.25.0"
FT_PREFIX = "/api/finetune/v1"


# --------------------------------------------------------------------------- data
def sample(
    dataset: Path, split: str, per_class: int, seed: int
) -> list[tuple[str, str]]:
    """(Photo_ID, label) pairs: per_class images per species, at most one per sequence."""
    by_label: dict[str, dict[tuple, str]] = {s: {} for s in SPECIES}
    with open(dataset / "annotations" / f"{split}_raw.csv", newline="") as f:
        for row in csv.DictReader(f):
            lb = row["Label"]
            if lb not in by_label:
                continue
            key = (os.path.dirname(row["Photo_ID"]), row["Seq_ID"])
            if (
                key not in by_label[lb]
                and (dataset / "resized" / row["Photo_ID"]).is_file()
            ):
                by_label[lb][key] = row["Photo_ID"]
    rng = random.Random(seed)
    out = []
    for lb in SPECIES:
        photos = sorted(by_label[lb].values())
        out += [(p, lb) for p in rng.sample(photos, min(per_class, len(photos)))]
    rng.shuffle(out)
    return out


def labels_of(path: Path) -> list[str]:
    rows = [ln.split(",") for ln in path.read_text().splitlines() if ln.strip()]
    return [name for name, _ in sorted(rows, key=lambda r: int(r[1]))]


def build_models_dir(studio_root: Path, models: Path) -> None:
    for name in (DETECTOR, CLASSIFIER):
        src, dst = studio_root / "models" / name, models / name
        (dst / "1").mkdir(parents=True, exist_ok=True)
        for f in ("manifest.toml", "labels.txt"):
            shutil.copy2(src / f, dst / f)
        weights = WEIGHTS / name / "1" / "model.onnx"
        if not weights.is_file():
            raise SystemExit(f"missing Engine weights {weights}")
        if not (dst / "1" / "model.onnx").is_file():
            shutil.copy2(weights, dst / "1" / "model.onnx")
        for p in (dst, dst / "1"):
            os.chmod(p, 0o755)
        for p in dst.rglob("*"):
            if p.is_file():
                os.chmod(p, 0o644)
    os.chmod(models, 0o755)


# --------------------------------------------------------------------------- compose
def extend(
    path: Path,
    studio_root: Path,
    work: Path,
    assets: Path,
    pg_password: str,
    engine_token: str,
    engine_port: int,
) -> None:
    doc = json.loads(path.read_text())
    svc = doc["services"]
    uid_gid = f"{os.getuid()}:{os.getgid()}"
    gpu = {
        "resources": {
            "reservations": {
                "devices": [
                    {"driver": "nvidia", "count": "all", "capabilities": ["gpu"]}
                ]
            }
        }
    }
    svc["sparrow-engine"] = {
        "image": ENGINE_IMAGE,
        "user": "1000:1000",
        "init": True,
        "environment": {
            "SPARROW_ENGINE_DEVICE": "cuda:0",
            "SPARROW_ENGINE_PRELOAD": "",
            "SPARROW_ENGINE_TRT_WARMUP": "",
            "SPARROW_ENGINE_REQUEST_TIMEOUT": "600",
            "SPARROW_ENGINE_MANAGEMENT_TOKEN": engine_token,
        },
        "volumes": [f"{work / 'models'}:/models:ro"],
        "ports": [f"127.0.0.1:{engine_port}:8080"],
        "deploy": gpu,
        "healthcheck": {
            "test": [
                "CMD-SHELL",
                "/usr/local/bin/sparrow-engine-server healthcheck || exit 1",
            ],
            "interval": "3s",
            "retries": 60,
        },
    }
    worker_env = {
        "PYTHONUNBUFFERED": "1",
        "HOME": "/tmp",
        "DB_DRIVER": "postgres",
        "DB_HOST": "studio-db",
        "DB_PORT": "5432",
        "DB_USER": "postgres",
        "DB_PASSWORD": pg_password,
        "DB_NAME": "postgres",
        "APP_SECRET_KEY": secrets.token_hex(16),
        "STORAGE_BACKEND": "local",
        "SPARROW_ENGINE_URL": "http://sparrow-engine:8080",
        "SPARROW_ENGINE_MANAGEMENT_TOKEN": engine_token,
        "MODEL_REPO_ROOT": "/models",
        "MODEL_ZOO_CATALOG": "/model-zoo/catalog.toml",
        "LOG_DIR": "/tmp/logs",
    }
    code = f"{studio_root / 'client' / 'worker'}:/app/worker:ro"
    zoo = f"{studio_root / 'model-zoo'}:/model-zoo:ro"
    deps = {
        "studio-db": {"condition": "service_healthy"},
        "sparrow-engine": {"condition": "service_healthy"},
    }
    svc["client-worker"] = {
        "image": CLIENT_IMAGE,
        "user": uid_gid,
        "command": ["python", "-m", "worker.infer_worker"],
        "environment": {
            **worker_env,
            "STAGING_DIR": "/app/staging",
            "GALLERY_DIR": "/data/gallery",
            "INFER_POLL_SECONDS": "1",
        },
        "volumes": [
            code,
            zoo,
            f"{work / 'staging'}:/app/staging",
            f"{work}:/data",
            f"{work / 'models'}:/models:ro",
        ],
        "depends_on": deps,
    }
    svc["model-download-worker"] = {
        "image": CLIENT_IMAGE,
        "user": uid_gid,
        "command": ["python", "-m", "worker.model_download_worker"],
        "environment": {
            **worker_env,
            "MODEL_DOWNLOAD_POLL_SECONDS": "2",
            "FINETUNE_INSTALL_ENABLED": "true",
            "FINETUNE_MODELS_DIR": "/finetune_models",
        },
        "volumes": [
            code,
            zoo,
            f"{work / 'models'}:/models",
            f"{work / 'studio_models'}:/finetune_models:ro",
        ],
        "depends_on": deps,
    }
    yolo_cwd = work / "yolo_cwd"
    extra = {
        "volumes": [
            "ft-data:/data",
            f"{assets}:/assets:ro",
            f"{yolo_cwd}:/yolo",
            # The Engine model packages double as the stock_model weights dir (FT-16).
            f"{work / 'models'}:/weights:ro",
        ],
        "working_dir": "/yolo",
    }
    svc["worker-cpu"] = dict(extra)
    svc["worker-gpu"] = dict(extra)
    svc["api"] = {"volumes": ["ft-data:/data", f"{assets}:/assets:ro"]}
    path.write_text(json.dumps(doc, indent=2))


def http(method: str, url: str, token: str | None = None, body=None, timeout=120):
    headers = {"Content-Type": "application/json"} if body is not None else {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload, status = r.read(), r.status
    except urllib.error.HTTPError as e:
        payload, status = e.read(), e.code
    try:
        return status, json.loads(payload) if payload else None
    except ValueError:
        return status, payload.decode(errors="replace")


# --------------------------------------------------------------------------- Studio DB
class Studio:
    def __init__(
        self, dsn: str, staging: Path, dataset: Path, base_url: str, token: str
    ):
        self.dsn, self.staging, self.dataset = dsn, staging, dataset
        self.base_url, self.token = base_url, token
        self.n = 0

    def conn(self):
        return psycopg.connect(self.dsn, autocommit=True)

    def seed(self, schema: Path, studio_root: Path) -> dict:
        with self.conn() as c:
            c.execute(schema.read_text())
            uid = c.execute(
                "INSERT INTO users (username, email) VALUES ('e2e', 'e2e@example.org') RETURNING id"
            ).fetchone()[0]
            pids = {}
            for tag, name in (
                ("train", "E2E Amazon fine-tune"),
                ("base", "E2E Amazon baseline"),
                ("stock_gate", "E2E stock activation checks"),
            ):
                pids[tag] = c.execute(
                    "INSERT INTO projects (owner_user_id, project_type, name) "
                    "VALUES (%s, 'batch', %s) RETURNING id",
                    (uid, name),
                ).fetchone()[0]
            pipe = c.execute(
                "INSERT INTO inference_pipelines (pipeline_key, pipeline_type, display_name, "
                "detector_key, detector_display_name, detector_server_folder, "
                "detector_class_mapping_file, classifier_key, classifier_display_name, "
                "classifier_server_folder, classifier_class_mapping_file, "
                "apply_when_detector_class_equals, pipeline_kind) VALUES "
                "('e2e-mdv6-ai4g', 'detector_classifier', 'MDV6 + AI4G Amazon', %s, 'MDV6', %s, "
                "'labels.txt', %s, 'AI4G Amazon', %s, 'labels.txt', 'animal', 'camera_trap') "
                "RETURNING id",
                (DETECTOR, DETECTOR, CLASSIFIER, CLASSIFIER),
            ).fetchone()[0]
            for kind, model in (("detector", DETECTOR), ("classifier", CLASSIFIER)):
                for i, name in enumerate(
                    labels_of(studio_root / "models" / model / "labels.txt")
                ):
                    c.execute(
                        "INSERT INTO model_labels (pipeline_id, model_kind, model_key, class_id, "
                        "label_key, display_name) VALUES (%s, %s, %s, %s, %s, %s)",
                        (pipe, kind, model, i, name.lower(), name),
                    )
            for pid in pids.values():
                c.execute(
                    "INSERT INTO project_pipelines (project_id, pipeline_id, is_primary) "
                    "VALUES (%s, %s, TRUE)",
                    (pid, pipe),
                )
        return {**pids, "base_pipeline": pipe}

    def upload(self, pid: int, photos: list[tuple[str, str]]) -> dict[int, str]:
        """Insert photos as pending uploads (the way Studio's upload route queues them)."""
        out = {}
        with self.conn() as c:
            for photo, label in photos:
                self.n += 1
                name = f"img{self.n:05d}.jpg"
                rel = f"staging/p{pid}/{name}"
                dst = self.staging / f"p{pid}" / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(self.dataset / "resized" / photo, dst)
                iid = c.execute(
                    "INSERT INTO images (project_id, ingestion_type, image_key, original_filename, "
                    "processing_status) VALUES (%s, 'upload', %s, %s, 'pending') RETURNING id",
                    (pid, rel, name),
                ).fetchone()[0]
                out[iid] = label
        return out

    def wait_processed(self, ids, timeout_s=900) -> dict[str, int]:
        deadline = time.time() + timeout_s
        with self.conn() as c:
            while True:
                rows = c.execute(
                    "SELECT processing_status, COUNT(*) FROM images WHERE id = ANY(%s) "
                    "GROUP BY processing_status",
                    (list(ids),),
                ).fetchall()
                counts = {s: n for s, n in rows}
                if not counts.get("pending") and not counts.get("processing"):
                    return counts
                if time.time() > deadline:
                    raise TimeoutError(f"inference still running: {counts}")
                time.sleep(2)

    def predictions(self, ids) -> dict[int, dict | None]:
        """Top-confidence box per image with its detection's raw_json; None without a box."""
        with self.conn() as c:
            cur = c.execute(
                "SELECT DISTINCT ON (d.image_id) d.image_id, d.raw_json, b.id AS box_id, "
                "b.classifier_class, b.confidence, b.model_label_id, b.box_meta "
                "FROM detections d JOIN detection_boxes b ON b.detection_id = d.id "
                "WHERE d.image_id = ANY(%s) AND COALESCE(b.is_deleted, FALSE) = FALSE "
                "ORDER BY d.image_id, b.confidence DESC, b.id",
                (list(ids),),
            )
            cols = [x.name for x in cur.description]
            top = {r[0]: dict(zip(cols, r)) for r in cur.fetchall()}
            raws = dict(
                c.execute(
                    "SELECT image_id, raw_json FROM detections WHERE image_id = ANY(%s)",
                    (list(ids),),
                ).fetchall()
            )
        return {
            i: (top.get(i) or ({"raw_json": raws[i]} if i in raws else None))
            for i in ids
        }

    def all_boxes(self, ids) -> list[dict]:
        with self.conn() as c:
            cur = c.execute(
                "SELECT d.image_id, d.raw_json, b.classifier_class, b.model_label_id, b.box_meta "
                "FROM detections d JOIN detection_boxes b ON b.detection_id = d.id "
                "WHERE d.image_id = ANY(%s)",
                (list(ids),),
            )
            cols = [x.name for x in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

    def review(self, truth: dict[int, str]) -> int:
        """Reviewer pass: the top box gets the true species, the rest are deleted."""
        top = self.predictions(truth)
        n = 0
        with self.conn() as c:
            for iid, label in truth.items():
                p = top[iid]
                if not p or "box_id" not in p:
                    continue
                c.execute(
                    "UPDATE detection_boxes SET classifier_class = %s WHERE id = %s",
                    (label, p["box_id"]),
                )
                c.execute(
                    "UPDATE detection_boxes b SET is_deleted = TRUE FROM detections d "
                    "WHERE b.detection_id = d.id AND d.image_id = %s AND b.id <> %s",
                    (iid, p["box_id"]),
                )
                c.execute(
                    "UPDATE images SET review_status = 'reviewed' WHERE id = %s", (iid,)
                )
                n += 1
        return n

    def primary(self, pid: int) -> dict:
        with self.conn() as c:
            cur = c.execute(
                "SELECT ip.* FROM project_pipelines pp JOIN inference_pipelines ip "
                "ON ip.id = pp.pipeline_id WHERE pp.project_id = %s AND pp.is_primary",
                (pid,),
            )
            cols = [x.name for x in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        if len(rows) != 1:
            raise RuntimeError(f"project {pid} has {len(rows)} primary pipelines")
        return rows[0]

    def label_ids(self, model_key: str) -> dict[int, str]:
        with self.conn() as c:
            return dict(
                c.execute(
                    "SELECT id, display_name FROM model_labels WHERE model_key = %s",
                    (model_key,),
                ).fetchall()
            )

    def import_row(self, import_id: int) -> dict:
        with self.conn() as c:
            cur = c.execute(
                "SELECT * FROM finetune_model_import WHERE id = %s", (import_id,)
            )
            cols = [x.name for x in cur.description]
            return dict(zip(cols, cur.fetchone()))

    # Studio fine-tune API, as the service calls it.
    def status(self, pid: int, import_id: int) -> dict:
        s, body = http(
            "GET",
            f"{self.base_url}{FT_PREFIX}/projects/{pid}/models/{import_id}",
            self.token,
        )
        if s != 200:
            raise RuntimeError(f"status -> {s}: {body}")
        return body

    def activate(self, pid: int, import_id: int) -> int:
        s, _ = http(
            "POST",
            f"{self.base_url}{FT_PREFIX}/projects/{pid}/models/{import_id}/activate",
            self.token,
            {},
        )
        return s

    def wait_install(self, pid: int, import_id: int, timeout_s=600) -> dict:
        deadline = time.time() + timeout_s
        while True:
            st = self.status(pid, import_id)
            if st.get("install_status") in ("installed", "failed"):
                return st
            if time.time() > deadline:
                raise TimeoutError(
                    f"install {import_id} still {st.get('install_status')}"
                )
            time.sleep(2)


def accuracy(preds: dict, truth: dict, field: str) -> tuple[float, float, dict]:
    """(accuracy over all images, detection rate, per-class correct/total)."""
    per: dict[str, list[int]] = {s: [0, 0] for s in SPECIES}
    hit = boxed = 0
    for iid, label in truth.items():
        p = preds.get(iid)
        per[label][1] += 1
        if not p or "box_id" not in p:
            continue
        boxed += 1
        got = (
            p["classifier_class"]
            if field == "classifier"
            else (p["box_meta"] or {}).get("detector_class_name")
        )
        if got == label:
            hit += 1
            per[label][0] += 1
    n = max(1, len(truth))
    return hit / n, boxed / n, {k: f"{a}/{b}" for k, (a, b) in per.items()}


def selective(preds: dict, truth: dict) -> tuple[float, float]:
    """(accuracy over images whose top box the classifier did not abstain on, their share of all
    images). Below the calibrated threshold Studio records the abstain label instead of a species."""
    hit = accepted = 0
    for iid, label in truth.items():
        p = preds.get(iid)
        if not p or "box_id" not in p or p["classifier_class"] in (None, ABSTAIN):
            continue
        accepted += 1
        hit += p["classifier_class"] == label
    return hit / max(1, accepted), accepted / max(1, len(truth))


def model_files_ok(models: Path, model_id: str) -> tuple[bool, str]:
    d = models / model_id
    if not d.is_dir():
        return False, f"{d} missing"
    names = sorted(p.name for p in d.iterdir())
    bad = [
        p.name
        for p in d.iterdir()
        if p.is_file() and not (p.stat().st_mode & stat.S_IROTH)
    ]
    world_dir = bool(d.stat().st_mode & stat.S_IXOTH)
    ok = {"manifest.toml", "labels.txt"} <= set(names) and not bad and world_dir
    return ok, f"files={names} unreadable={bad}"


# --------------------------------------------------------------------------- scenario
def model_card(api: Api, model_id: int) -> dict:
    """The model_card.json inside the stored bundle: the resolved training device lives there."""
    status, blob = api.call("GET", f"/models/{model_id}/artifact")
    if status != 200 or not isinstance(blob, bytes):
        return {}
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        return json.loads(zf.read("model_card.json"))


def stock_worse(m: dict) -> bool:
    return bool((m.get("metrics") or {}).get("stock_worse"))


def service_activate(api: Api, m: dict, project: int) -> dict:
    """Activate for Studio through the service; the install must have been requested.

    A model the stock classifier beat needs ``confirm`` (FT-16); ``stock_gate`` checks that path.
    """
    dep = api.ok(
        "POST",
        f"/models/{m['id']}/activate",
        {"consumer": "studio", "project": str(project), "confirm": stock_worse(m)},
    )
    install = dep.get("studio_install")
    check(
        f"{m['pipeline']}: service activation requests a Studio install",
        install in ("pending", "installing", "installed"),
        f"studio_install={install!r}",
    )
    if install not in ("pending", "installing", "installed"):
        raise RuntimeError(
            f"activation of model {m['id']} gave studio_install={install!r}"
        )
    return dep


def stock_gate(api: Api, model: dict, project: int) -> None:
    """FT-16: both classifiers were scored against the stock AI4G package on their val split.

    A model the stock classifier beat is refused with 409 stock_model_better until confirmed;
    a model that matched or beat it activates without confirm. The 409 must leave no deployment.
    """
    refused = 0
    for pipe in ("conventional_classifier", "dinov3_linear"):
        if not model.get(pipe, {}).get("id"):
            check(f"{pipe}: stock comparison (no model: training failed)", False)
            continue
        m = api.ok("GET", f"/models/{model[pipe]['id']}")
        model[pipe]["metrics"] = m.get("metrics") or {}
        if pipe == "dinov3_linear":
            ref = m.get("reference_comparison") or {}
            check(
                "dinov3_linear: reference comparison reported (FT-19; "
                f"warning={'yes' if ref.get('worse') else 'no'})",
                isinstance(ref.get("reference_macro_f1"), float)
                and (ref.get("crops") or 0) > 0,
                json.dumps(ref),
            )
        cmp = m.get("stock_comparison") or {}
        check(
            f"{pipe}: compared against stock {CLASSIFIER} on the val split",
            cmp.get("model") == CLASSIFIER and (cmp.get("crops") or 0) > 0,
            json.dumps(cmp),
        )
        before = m.get("active_deployments")
        status, body = api.call(
            "POST",
            f"/models/{m['id']}/activate",
            {"consumer": "studio", "project": str(project)},
        )
        if cmp.get("worse") and cmp.get("enforced", True):
            refused += 1
            detail = (body or {}).get("detail") if isinstance(body, dict) else None
            after = api.ok("GET", f"/models/{m['id']}").get("active_deployments")
            check(
                f"{pipe}: stock model scored higher, activation without confirm -> 409 stock_model_better",
                status == 409
                and isinstance(detail, dict)
                and detail.get("code") == "stock_model_better"
                and after == before,
                f"{status} {str(body)[:300]}",
            )
        else:
            check(
                f"{pipe}: not worse than stock (or below stock_min_val_crops="
                f"{cmp.get('min_val_crops')}), activation without confirm accepted",
                status == 200,
                f"{status} {str(body)[:300]}",
            )
    info_line = (
        f"{refused} of 2 classifiers required confirmation against stock {CLASSIFIER}"
    )
    print(info_line, flush=True)


def skipped(what: str, needs: str) -> None:
    """Record checks that cannot run because a model they need failed to train."""
    check(f"{what} (skipped: no trained model for {needs})", False)


def scenario(
    a, studio: Studio, api: Api, info: dict, work: Path, val: list, train_truth: dict
) -> None:
    P, B = info["projects"]["train"], info["projects"]["base"]
    models_root, bundles = work / "models", work / "studio_models"
    val_iter = iter(val)

    def take(n):
        return [next(val_iter) for _ in range(n)]

    # ---- service: import, snapshot, train on GPU
    ws = api.ok("POST", "/workspaces", {"name": "Amazon e2e"})["id"]
    ds_c = api.ok(
        "POST",
        f"/workspaces/{ws}/datasets",
        {"name": "crops", "task": "classification"},
    )["id"]
    ds_d = api.ok(
        "POST", f"/workspaces/{ws}/datasets", {"name": "boxes", "task": "detection"}
    )["id"]
    reviewed = info["reviewed_train_images"]
    for ds, task in ((ds_c, "classification"), (ds_d, "detection")):
        res = api.ok(
            "POST",
            "/connectors/studio/import",
            {"project": str(P), "dataset_id": ds, "reviewed_only": True},
        )
        check(
            f"service imports the reviewed Studio project as {task}",
            res.get("imported_images") == reviewed,
            json.dumps(
                {
                    k: res.get(k)
                    for k in ("imported_images", "imported_items", "skipped")
                }
            ),
        )
    snaps = {}
    for ds, key in ((ds_c, "cls"), (ds_d, "det")):
        snap = api.ok("POST", f"/datasets/{ds}/snapshots", {})
        snaps[key] = snap["id"]
        counts = snap.get("label_counts") or {}
        check(
            f"{key} snapshot holds the 5 reviewed species",
            set(counts) == set(SPECIES),
            json.dumps(counts),
        )
    # dinov3_linear: DINO features are prepared (opt-in, progress polled) before training.
    snaps["dino"] = prepare_dino_snapshot(
        api, ds_c, a.train_timeout, "dinov3_linear reviewed Studio crops"
    )["id"]
    info["snapshots"] = snaps

    jobs = {
        "conventional_classifier": (
            snaps["cls"],
            {
                "epochs": a.epochs_cls,
                "warmup_epochs": 1,
                "device": "cuda",
                "stock_model": CLASSIFIER,
            },
        ),
        "dinov3_linear": (snaps["dino"], {"stock_model": CLASSIFIER}),
        "yolo_detector": (
            snaps["det"],
            {
                "model": "yolo11n.pt",
                "epochs": a.epochs_det,
                "imgsz": 512,
                "batch": 16,
                "device": "0",
            },
        ),
    }
    model = {}
    for pipe, (snap, params) in jobs.items():
        j = api.ok(
            "POST", "/jobs", {"pipeline": pipe, "snapshot_id": snap, "params": params}
        )
        job = wait_job(api, j["id"], a.train_timeout)
        ok = job["state"] == "succeeded"
        models = api.ok("GET", f"/models?job_id={j['id']}")["models"] if ok else []
        m = models[0] if models else {}
        metrics = m.get("metrics") or {}
        device = model_card(api, m["id"]).get("device") if m else None
        info.setdefault("training", {})[pipe] = {
            "job": j["id"],
            "seconds": job_seconds(job),
            "metrics": metrics,
            "device": device,
            "error": job.get("error"),
        }
        check(
            f"{pipe} trains on real data ({job_seconds(job)})",
            ok and bool(m),
            f"state={job['state']} device={device} error={str(job.get('error'))[:300]}",
        )
        model[pipe] = m
    gpu_ok = all(
        str(info["training"][p]["device"]).startswith(("cuda", "0"))
        for p in ("conventional_classifier", "yolo_detector")
    )
    check(
        "conventional_classifier and yolo_detector trained on the GPU",
        gpu_ok,
        f"conventional={info['training']['conventional_classifier']['device']} "
        f"yolo={info['training']['yolo_detector']['device']}; "
        "dinov3_linear embeds in Data Preparation and trains its head on CPU by design",
    )

    stock_gate(api, model, info["projects"]["stock_gate"])

    # ---- 1. conventional classifier: activate from the service, Studio installs and runs it
    conv = model["conventional_classifier"]
    if not conv.get("id"):
        skipped(
            "conventional_classifier: Studio install and inference",
            "conventional_classifier",
        )
    else:
        dep = service_activate(api, conv, P)
        conv_import = int(dep["external_id"])
        st = studio.wait_install(P, conv_import)
        conv_mid = st.get("engine_model_id")
        check(
            "conventional_classifier: service activation -> Studio install_status=installed",
            st["install_status"] == "installed",
            json.dumps(
                {
                    k: st.get(k)
                    for k in (
                        "install_status",
                        "install_error",
                        "engine_model_id",
                        "pipeline_id",
                    )
                }
            ),
        )
        ok, detail = model_files_ok(models_root, conv_mid or "-")
        check(
            "installed model dir in /models is readable by the Engine user", ok, detail
        )
        prim = studio.primary(P)
        check(
            "project's primary pipeline = MDV6 detector + fine-tuned classifier",
            str(prim["id"]) == str(st.get("pipeline_id"))
            and prim["pipeline_type"] == "detector_classifier"
            and prim["detector_server_folder"] == DETECTOR
            and prim["classifier_server_folder"] == conv_mid
            and prim["apply_when_detector_class_equals"] == "animal",
            json.dumps(
                {
                    k: prim[k]
                    for k in (
                        "pipeline_key",
                        "pipeline_type",
                        "detector_server_folder",
                        "classifier_server_folder",
                        "apply_when_detector_class_equals",
                    )
                }
            ),
        )
        check(
            "baseline project keeps the stock pipeline",
            studio.primary(B)["id"] == info["projects"]["base_pipeline"],
        )

        truth = studio.upload(P, take(a.val_main))
        counts = studio.wait_processed(truth)
        preds = studio.predictions(truth)
        raws = [p["raw_json"] for p in preds.values() if p]
        check(
            f"Studio's inference worker processed {len(truth)} new uploads with the fine-tuned classifier",
            raws
            and all(
                r["classifier_model"] == conv_mid and r["pipeline_key"] == conv_mid
                for r in raws
            ),
            f"status={counts} models={sorted({r['classifier_model'] for r in raws})}",
        )
        labels = studio.label_ids(conv_mid)
        boxes = [
            b for b in studio.all_boxes(truth) if b["classifier_class"] is not None
        ]
        sure = [b for b in boxes if b["classifier_class"] != ABSTAIN]
        unsure = [b for b in boxes if b["classifier_class"] == ABSTAIN]
        check(
            "every classified box carries a trained species and its model_labels id, "
            "or the abstain label with the detector's model_labels id",
            sure
            and all(b["classifier_class"] in SPECIES for b in sure)
            and all(
                labels.get(b["model_label_id"]) == b["classifier_class"] for b in sure
            )
            # an abstained box keeps the detector's label (client/worker/infer_worker.py)
            and all(
                b["model_label_id"]
                == (b["box_meta"] or {}).get("detector_model_label_id")
                for b in unsure
            ),
            f"classified boxes={len(boxes)} abstained={len(unsure)} label rows={len(labels)}",
        )
        acc, det_rate, per = accuracy(preds, truth, "classifier")
        sel_acc, coverage = selective(preds, truth)
        base_acc, base_det, base_per = (
            info["baseline"]["accuracy"],
            info["baseline"]["detected"],
            info["baseline"]["per_class"],
        )
        info.setdefault("results", {})["conventional_classifier"] = {
            "accuracy": acc,
            "detected": det_rate,
            "per_class": per,
            "accepted_accuracy": sel_acc,
            "coverage": coverage,
            "n": len(truth),
        }
        check(
            f"fine-tuned classifier accuracy on accepted (not abstained) predictions in Studio >= {a.min_acc:.0%}",
            coverage > 0 and sel_acc >= a.min_acc,
            f"accepted {sel_acc:.1%} at coverage {coverage:.0%}; all images {acc:.1%} {per}; stock AI4G baseline {base_acc:.1%} {base_per} (detected {det_rate:.0%}/{base_det:.0%})",
        )

    # ---- 2. dinov3_linear: corrupted bundle fails and keeps the live pipeline; restore installs
    dino = model["dinov3_linear"]
    if not dino.get("id"):
        skipped(
            "dinov3_linear: Studio push, corrupted-bundle and restore", "dinov3_linear"
        )
    else:
        pushed = api.ok(
            "POST",
            "/connectors/studio/push",
            {"model_id": dino["id"], "project": str(P)},
        )
        dino_import = int(pushed["studio_model_id"])
        row = studio.import_row(dino_import)
        check(
            "dinov3_linear pushed without activation stays not_requested",
            studio.status(P, dino_import).get("install_status")
            in (None, "not_requested", "none", ""),
            str(studio.status(P, dino_import).get("install_status")),
        )
        bundle = bundles / row["bundle_key"]
        good = bundle.read_bytes()
        bundle.write_bytes(good[: len(good) // 2])
        before = studio.primary(P)
        s = studio.activate(P, dino_import)
        st = studio.wait_install(P, dino_import)
        leftovers = [
            p.name for p in models_root.iterdir() if f"-m{dino_import}-" in p.name
        ]
        check(
            "corrupted bundle: install fails with a reason, the live pipeline is untouched, no model dir left",
            s == 202
            and st["install_status"] == "failed"
            and st.get("install_error")
            and studio.primary(P)["id"] == before["id"]
            and not leftovers,
            f"activate={s} error={st.get('install_error')!r} leftovers={leftovers}",
        )
        bundle.write_bytes(good)
        s = studio.activate(P, dino_import)
        st = studio.wait_install(P, dino_import)
        dino_mid = st.get("engine_model_id")
        check(
            "restored bundle: activating again installs it and switches the project",
            s == 202
            and st["install_status"] == "installed"
            and studio.primary(P)["classifier_server_folder"] == dino_mid,
            json.dumps(
                {
                    k: st.get(k)
                    for k in ("install_status", "install_error", "engine_model_id")
                }
            ),
        )
        truth = studio.upload(P, take(a.val_small))
        studio.wait_processed(truth)
        preds = studio.predictions(truth)
        raws = [p["raw_json"] for p in preds.values() if p]
        acc, det_rate, per = accuracy(preds, truth, "classifier")
        info.setdefault("results", {})["dinov3_linear"] = {
            "accuracy": acc,
            "detected": det_rate,
            "per_class": per,
            "n": len(truth),
        }
        check(
            "dinov3_linear runs in Studio's worker via the Engine",
            raws
            and all(r["classifier_model"] == dino_mid for r in raws)
            and acc >= a.min_acc,
            f"accuracy {acc:.1%} {per}",
        )

    # ---- 3. yolo_detector: detector-only pipeline
    yolo = model["yolo_detector"]
    if not yolo.get("id"):
        skipped("yolo_detector: Studio install and inference", "yolo_detector")
    else:
        dep = service_activate(api, yolo, P)
        yolo_import = int(dep["external_id"])
        st = studio.wait_install(P, yolo_import)
        yolo_mid = st.get("engine_model_id")
        prim = studio.primary(P)
        check(
            "yolo_detector: installed as a detector_only pipeline",
            st["install_status"] == "installed"
            and prim["pipeline_type"] == "detector_only"
            and prim["detector_server_folder"] == yolo_mid
            and prim["classifier_server_folder"] is None,
            json.dumps(
                {
                    "status": st.get("install_status"),
                    "error": st.get("install_error"),
                    "type": prim["pipeline_type"],
                    "detector": prim["detector_server_folder"],
                }
            ),
        )
        truth = studio.upload(P, take(a.val_small))
        studio.wait_processed(truth)
        preds = studio.predictions(truth)
        raws = [p["raw_json"] for p in preds.values() if p]
        labels = studio.label_ids(yolo_mid)
        boxes = studio.all_boxes(truth)
        acc, det_rate, per = accuracy(preds, truth, "detector")
        info.setdefault("results", {})["yolo_detector"] = {
            "accuracy": acc,
            "detected": det_rate,
            "per_class": per,
            "n": len(truth),
        }
        check(
            "fine-tuned detector runs in Studio: species boxes with its model_labels ids",
            raws
            and all(
                r["detector_model"] == yolo_mid
                and r["pipeline_type"] == "detector_only"
                for r in raws
            )
            and boxes
            and all(
                labels.get(b["model_label_id"])
                == (b["box_meta"] or {}).get("detector_class_name")
                for b in boxes
            ),
            f"boxes={len(boxes)} detected {det_rate:.0%}",
        )
        check(
            f"fine-tuned detector top-box species accuracy >= {a.min_det_acc:.0%}",
            acc >= a.min_det_acc,
            f"{acc:.1%} {per}",
        )

    def train(pipe: str, snap: int, params: dict) -> dict:
        j = api.ok(
            "POST", "/jobs", {"pipeline": pipe, "snapshot_id": snap, "params": params}
        )
        job = wait_job(api, j["id"], a.train_timeout)
        models = (
            api.ok("GET", f"/models?job_id={j['id']}")["models"]
            if job["state"] == "succeeded"
            else []
        )
        return {"job": job, "model": models[0] if models else {}}

    # ---- 4a. warm start: the installed species detector comes back as a base model
    if not yolo.get("id"):
        skipped("yolo_detector warm start from the Studio install", "yolo_detector")
    else:
        listed = api.ok("GET", f"/connectors/studio/models?project={P}")["models"]
        cand = [m for m in listed if str(m.get("id")) == str(yolo_import)]
        s, base = api.call(
            "POST",
            "/connectors/studio/models/import",
            {"project": str(P), "studio_model_id": str(yolo_import)},
        )
        base_id = (base or {}).get("id") if isinstance(base, dict) else None
        check(
            "the installed species detector imports back as a warm-start base model",
            bool(cand)
            and cand[0].get("importable") is True
            and s in (200, 201)
            and isinstance(base_id, int),
            json.dumps({"listed": cand, "status": s, "body": base}, default=str)[:400],
        )
        if isinstance(base_id, int):
            warm = train(
                "yolo_detector",
                snaps["det"],
                {
                    "model": "yolo11n.pt",
                    "epochs": 5,
                    "imgsz": 512,
                    "batch": "auto",
                    "device": "0",
                    "base_model_id": base_id,
                },
            )
            wm = warm["model"]
            card = model_card(api, wm["id"]) if wm else {}
            info.setdefault("training", {})["yolo_warm_start"] = {
                "job": warm["job"]["id"],
                "seconds": job_seconds(warm["job"]),
                "metrics": wm.get("metrics"),
                "params": warm["job"].get("params"),
                "error": warm["job"].get("error"),
            }
            check(
                "yolo_detector warm-starts from the base model with batch=auto on the GPU",
                warm["job"]["state"] == "succeeded"
                and bool(wm)
                and (warm["job"].get("params") or {}).get("base_model_id") == base_id
                and str(card.get("device")).startswith(("cuda", "0")),
                json.dumps(
                    {
                        "state": warm["job"]["state"],
                        "device": card.get("device"),
                        "metrics": wm.get("metrics"),
                        "error": str(warm["job"].get("error"))[:300],
                    },
                    default=str,
                ),
            )
            # per_class_table rows: {label, precision, recall, map50, map50_95, instances}
            per_class = (wm.get("metadata") or {}).get("per_class") or []
            check(
                "warm-started detector reports per-class metrics for the trained species",
                bool(per_class)
                and {r.get("label") for r in per_class} <= set(SPECIES)
                and any(r.get("map50") is not None for r in per_class),
                json.dumps(per_class, default=str)[:400],
            )

    # ---- 4b. a species detector runs alone: Studio refuses a classifier on top of it
    if not (conv.get("id") and yolo.get("id")):
        skipped(
            "combinations: species detector refuses a classifier; generic detector gates one",
            "conventional_classifier and yolo_detector",
        )
        return
    s = studio.activate(P, conv_import)
    st = studio.wait_install(P, conv_import) if s == 202 else {}
    prim = studio.primary(P)
    check(
        "a classifier is refused on top of the fine-tuned species detector",
        (s != 202 or st.get("install_status") == "failed")
        and prim["pipeline_type"] == "detector_only"
        and prim["detector_server_folder"] == yolo_mid,
        json.dumps(
            {
                "status": s,
                "install": st.get("install_status"),
                "error": st.get("install_error"),
                "type": prim["pipeline_type"],
            }
        ),
    )

    # ---- 4c. generic detector: every box relabelled "animal"; it gates the classifier
    ds_g = api.ok(
        "POST",
        f"/workspaces/{ws}/datasets",
        {"name": "animal boxes", "task": "detection"},
    )["id"]
    api.ok(
        "POST",
        "/connectors/studio/import",
        {"project": str(P), "dataset_id": ds_g, "reviewed_only": True},
    )
    for item in api.ok("GET", f"/datasets/{ds_g}/items")["items"]:
        api.ok(
            "PATCH",
            f"/datasets/{ds_g}/items/{quote(item['image'], safe='')}",
            {
                "annotations": [
                    {"label": "animal", "bbox": x["bbox"]} for x in item["annotations"]
                ]
            },
        )
    snap_g = api.ok("POST", f"/datasets/{ds_g}/snapshots", {})
    gen = train(
        "yolo_detector",
        snap_g["id"],
        {"model": "yolo11n.pt", "epochs": 15, "imgsz": 512, "batch": 16, "device": "0"},
    )
    gm = gen["model"]
    info.setdefault("training", {})["yolo_generic"] = {
        "job": gen["job"]["id"],
        "seconds": job_seconds(gen["job"]),
        "metrics": gm.get("metrics"),
        "error": gen["job"].get("error"),
    }
    check(
        "generic 'animal' detector trains",
        set(snap_g.get("label_counts") or {}) == {"animal"} and bool(gm),
        json.dumps(
            {
                "labels": snap_g.get("label_counts"),
                "state": gen["job"]["state"],
                "error": str(gen["job"].get("error"))[:300],
            }
        ),
    )
    if not gm:
        return
    dep = service_activate(api, gm, P)
    gen_import = int(dep["external_id"])
    st = studio.wait_install(P, gen_import)
    gen_mid = st.get("engine_model_id")
    prim = studio.primary(P)
    caps = prim.get("detector_capabilities") or {}
    check(
        "generic detector installs as detector_only with generic capabilities and gate 'animal'",
        st["install_status"] == "installed"
        and prim["pipeline_type"] == "detector_only"
        and prim["detector_server_folder"] == gen_mid
        and caps.get("detector_kind") == "generic"
        and caps.get("gate_class") == "animal",
        json.dumps(
            {
                "status": st.get("install_status"),
                "error": st.get("install_error"),
                "caps": caps,
            },
            default=str,
        ),
    )
    s = studio.activate(P, conv_import)
    st = studio.wait_install(P, conv_import)
    prim = studio.primary(P)
    check(
        "re-activating the classifier after the generic detector installs again",
        s == 202
        and st["install_status"] == "installed"
        and st.get("engine_model_id") == conv_mid,
        json.dumps(
            {
                k: st.get(k)
                for k in ("install_status", "install_error", "engine_model_id")
            }
        ),
    )
    check(
        "the classifier pipeline gates on the generic detector's 'animal' class",
        prim["pipeline_type"] == "detector_classifier"
        and prim["detector_server_folder"] == gen_mid
        and prim["classifier_server_folder"] == conv_mid
        and prim["apply_when_detector_class_equals"] == "animal",
        json.dumps(
            {
                k: prim[k]
                for k in (
                    "detector_server_folder",
                    "classifier_server_folder",
                    "apply_when_detector_class_equals",
                )
            }
        ),
    )
    truth = studio.upload(P, take(a.val_small))
    studio.wait_processed(truth)
    boxes = studio.all_boxes(truth)
    classified = [b for b in boxes if b["classifier_class"] is not None]
    acc, det_rate, per = accuracy(studio.predictions(truth), truth, "classifier")
    info.setdefault("results", {})["yolo_then_conventional"] = {
        "accuracy": acc,
        "detected": det_rate,
        "per_class": per,
        "n": len(truth),
    }
    check(
        "fine-tuned detector + fine-tuned classifier: the classifier runs on the detector's boxes",
        boxes and len(classified) == len(boxes),
        f"boxes={len(boxes)} classified={len(classified)} accuracy {acc:.1%}",
    )

    # ---- 5. re-activate the classifier through the service: Studio upserts the import on
    # (project, source model, sha256), so the re-push reuses it; the service records it
    dep = service_activate(api, conv, P)
    new_import = int(dep["external_id"])
    st = studio.wait_install(P, new_import)
    new_mid = st.get("engine_model_id") or ""
    prim = studio.primary(P)
    check(
        "service re-activation re-uses the Studio import and makes it the project's classifier",
        new_import == conv_import
        and st["install_status"] == "installed"
        and new_mid == conv_mid
        and prim["classifier_server_folder"] == conv_mid
        and prim["detector_server_folder"] == gen_mid,
        json.dumps(
            {
                "import": new_import,
                "engine_model_id": new_mid,
                "detector": prim["detector_server_folder"],
                "classifier": prim["classifier_server_folder"],
            }
        ),
    )
    conv_deps = api.ok("GET", f"/models/{conv['id']}").get("active_deployments") or []
    yolo_deps = [
        d
        for m in (yolo, gm)
        for d in api.ok("GET", f"/models/{m['id']}").get("active_deployments") or []
    ]
    active = [
        d
        for d in api.ok("GET", "/deployments")["deployments"]
        if d["consumer"] == "studio"
        and d["project"] == str(P)
        and d["deactivated_at"] is None
    ]
    check(
        "service records one active Studio deployment for the project: the re-activated classifier",
        conv_deps
        == [{"consumer": "studio", "project": str(P), "external_id": str(new_import)}]
        and yolo_deps == []
        and len(active) == 1
        and active[0]["model_id"] == conv["id"]
        and str(active[0]["external_id"]) == str(new_import),
        json.dumps(
            {"conv": conv_deps, "yolo": yolo_deps, "active": active}, default=str
        )[:400],
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--studio-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--dataset", type=Path, default=Path.home() / "datasets/amazon_camera_trap"
    )
    ap.add_argument("--assets", type=Path, default=Path("/tmp/ft_e2e_full_assets"))
    ap.add_argument("--train-per-class", type=int, default=60)
    ap.add_argument("--val-per-class", type=int, default=30)
    ap.add_argument("--val-main", type=int, default=75)
    ap.add_argument("--val-small", type=int, default=25)
    ap.add_argument("--epochs-cls", type=int, default=10)
    ap.add_argument("--epochs-det", type=int, default=40)
    ap.add_argument("--min-acc", type=float, default=0.5)
    ap.add_argument("--min-det-acc", type=float, default=0.3)
    ap.add_argument("--train-timeout", type=int, default=3600)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()

    studio_root, out, assets = (
        a.studio_root.resolve(),
        a.out.resolve(),
        a.assets.resolve(),
    )
    for f in ("dinov3_vits16.onnx", "dinov3_vits16.onnx.data", "yolo11n.pt"):
        if not (assets / f).is_file():
            raise SystemExit(f"missing asset {assets / f}")
    work = out / "work"
    for d in ("gallery", "studio_models", "storage", "staging", "yolo_cwd", "models"):
        (work / d).mkdir(parents=True, exist_ok=True)
    os.chmod(work / "staging", 0o777)
    shutil.copy2(assets / "yolo11n.pt", work / "yolo_cwd" / "yolo11n.pt")
    os.chmod(work / "yolo_cwd", 0o777)
    os.chmod(work / "yolo_cwd" / "yolo11n.pt", 0o644)
    build_models_dir(studio_root, work / "models")

    train = sample(a.dataset, "train", a.train_per_class, seed=1)
    val = sample(a.dataset, "val", a.val_per_class, seed=2)
    need = a.val_main + 3 * a.val_small
    if len(val) < need:
        raise SystemExit(f"need {need} validation images, sampled {len(val)}")
    base_val, val = val[: a.val_main], val[a.val_main :] + val[: a.val_main]

    token, api_token, engine_token = (secrets.token_urlsafe(32) for _ in range(3))
    pg_password = secrets.token_urlsafe(16)
    project = f"ft-e2e-engine-{secrets.token_hex(3)}"
    env_file, override = work / "service.env", work / "compose.e2e.yml"
    files = [REPO / "compose.service.yml", override]
    ports = {k: free_port() for k in ("pg", "studio", "ft", "engine")}
    info: dict = {
        "studio_branch": run(
            ["git", "-C", str(studio_root), "rev-parse", "--abbrev-ref", "HEAD"]
        ).stdout.strip(),
        "studio_commit": run(
            ["git", "-C", str(studio_root), "rev-parse", "HEAD"]
        ).stdout.strip(),
        "studio_dirty": run(
            [
                "git",
                "-C",
                str(studio_root),
                "status",
                "--porcelain",
                "client",
                "dashboard",
                "db",
            ]
        ).stdout.strip(),
        "finetune_commit": run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"]
        ).stdout.strip(),
        "engine_image": ENGINE_IMAGE,
        "dataset": str(a.dataset),
        "species": SPECIES,
        "train_images": len(train),
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
                "SPARROW_FT_WEIGHTS_DIR=/weights",
            ]
        )
        + "\n"
    )
    os.chmod(env_file, 0o600)
    write_override(
        override, studio_root, work, ports["pg"], ports["studio"], pg_password, token
    )
    extend(
        override, studio_root, work, assets, pg_password, engine_token, ports["engine"]
    )
    os.chmod(override, 0o600)
    dsn = f"host=127.0.0.1 port={ports['pg']} user=postgres password={pg_password} dbname=postgres"
    studio = Studio(
        dsn, work / "staging", a.dataset, f"http://127.0.0.1:{ports['studio']}", token
    )

    t0 = time.time()
    try:
        r = compose(
            project, files, env_file, "up", "-d", "--wait", "studio-db", check_rc=False
        )
        if r.returncode:
            raise RuntimeError(f"studio-db did not start: {r.stderr[-800:]}")
        info["projects"] = studio.seed(
            studio_root / "db/schema/01_schema.sql", studio_root
        )
        r = compose(
            project, files, env_file, "up", "-d", "--wait", "--no-build", check_rc=False
        )
        (out / "compose_up.log").write_text(r.stdout + r.stderr)
        check(
            "compose up: Studio, Engine (GPU), client + model-download workers, service stack",
            r.returncode == 0,
            r.stderr[-400:] if r.returncode else "",
        )
        if r.returncode:
            raise RuntimeError("compose up failed")
        compose(project, files, env_file, "stop", "worker-cpu")

        for mid in (DETECTOR, CLASSIFIER):
            s, body = http(
                "POST",
                f"http://127.0.0.1:{ports['engine']}/v1/models/load",
                engine_token,
                {"model_id": mid},
                timeout=600,
            )
            check(f"Engine loads stock model {mid}", s == 200, f"{s} {str(body)[:200]}")

        # Stock pipeline over training and baseline images.
        P, B = info["projects"]["train"], info["projects"]["base"]
        train_truth = studio.upload(P, train)
        base_truth = studio.upload(B, base_val)
        c1 = studio.wait_processed(train_truth)
        c2 = studio.wait_processed(base_truth)
        preds = studio.predictions(train_truth)
        boxed = sum(1 for p in preds.values() if p and "box_id" in p)
        check(
            f"stock pipeline (MDV6 + AI4G) runs over {len(train_truth)} training + {len(base_truth)} baseline uploads",
            boxed > 0.6 * len(train_truth)
            and not c1.get("failed", 0) > 0.2 * len(train_truth),
            f"train status={c1} base status={c2} images with a box={boxed}",
        )
        bacc, bdet, bper = accuracy(
            studio.predictions(base_truth), base_truth, "classifier"
        )
        info["baseline"] = {
            "accuracy": bacc,
            "detected": bdet,
            "per_class": bper,
            "n": len(base_truth),
        }
        info["reviewed_train_images"] = studio.review(train_truth)
        check(
            "reviewer pass labels the top box of every detected training image",
            info["reviewed_train_images"] == boxed,
            f"reviewed={info['reviewed_train_images']}",
        )

        api = Api(f"http://127.0.0.1:{ports['ft']}", api_token)
        scenario(a, studio, api, info, work, val, train_truth)
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        check(
            "harness completed without an exception",
            False,
            f"{type(exc).__name__}: {exc}",
        )
    finally:
        logs = out / "logs"
        logs.mkdir(exist_ok=True)
        for svc in (
            "studio",
            "client-worker",
            "model-download-worker",
            "sparrow-engine",
            "api",
            "worker-gpu",
        ):
            r = compose(
                project, files, env_file, "logs", "--no-color", svc, check_rc=False
            )
            (logs / f"{svc}.log").write_text(r.stdout + r.stderr)
        if not a.keep:
            compose(project, files, env_file, "down", "-v", check_rc=False)
        info["seconds"] = round(time.time() - t0)
        info["checks"] = [{"name": n, "ok": ok, "detail": d} for n, ok, d in CHECKS]
        (out / "result.json").write_text(json.dumps(info, indent=2, default=str))
    failed = [n for n, ok, _ in CHECKS if not ok]
    print(
        f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed in {info['seconds']}s",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
