"""Host a real Sparrow Studio on this machine, wired to a running fine-tune compose stack.

For hands-on manual testing of the Studio round trip: review in Studio, import reviewed data into
the fine-tune service, activate a trained model in Studio, and pull a Studio model back as a
training base. Unlike ``studio_branch_e2e.py`` the stack stays up until ``--down``.

What it starts (compose project ``--project``):

- ``studio-db``: Postgres 16 with the Studio checkout's ``db/schema/01_schema.sql``.
- ``studio``: the real dashboard (Flask, ``--no-reload``; restart the container after editing the
  checkout). It joins the fine-tune stack's network with the alias ``studio``, so the fine-tune
  service reaches it at ``http://studio:5000``. The browser reaches it at
  ``http://127.0.0.1:<--port>``.

What it seeds (only on first start): one active Studio login without email 2FA, and one batch
project of real camera-trap crops from the Amazon dataset (one full-frame box per image). Part of
the project is already reviewed; the rest is unreviewed with model guesses, some of them wrong, for
the tester to review in the Studio UI.

Credentials go to ``<work>/studio_live.json`` (mode 0600).

    uv run --extra dev --extra e2e python scripts/e2e/studio_live_host.py \
        --studio-root <Studio checkout> --work <dir> [--down]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import secrets
import subprocess
import sys
import time
from pathlib import Path

import psycopg
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
CLASSES = ("Dasyprocta", "Bos", "Pecari", "Mazama")


def werkzeug_hash(password: str, iterations: int = 600_000) -> str:
    """Same format and derivation as ``werkzeug.security.generate_password_hash`` (pbkdf2)."""
    salt = secrets.token_urlsafe(12)[:16]
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), salt.encode(), iterations
    ).hex()
    return f"pbkdf2:sha256:{iterations}${salt}${digest}"


def compose_doc(
    studio_root: Path,
    work: Path,
    network: str,
    port: int,
    pg_port: int,
    creds: dict,
) -> dict:
    dash = studio_root / "dashboard"
    studio_cmd = (
        "pip install -q --no-cache-dir --disable-pip-version-check --target /tmp/deps "
        "-r /src/dashboard/build/requirements.txt && cd /src/dashboard/sparrow_dashboard && "
        "exec python -m flask --app wsgi:app run --host 0.0.0.0 --port 5000 --no-reload"
    )
    health = (
        "import urllib.request,urllib.error\n"
        "try: urllib.request.urlopen('http://127.0.0.1:5000/api/finetune/v1/projects/1/export')\n"
        "except urllib.error.HTTPError as e: raise SystemExit(0 if e.code == 401 else 1)\n"
        "raise SystemExit(1)"
    )
    return {
        "services": {
            "studio-db": {
                "image": "postgres:16-alpine",
                "environment": {"POSTGRES_PASSWORD": creds["pg_password"]},
                "ports": [f"127.0.0.1:{pg_port}:5432"],
                "volumes": ["studio-db:/var/lib/postgresql/data"],
                "healthcheck": {
                    "test": ["CMD-SHELL", "pg_isready -U postgres"],
                    "interval": "2s",
                    "retries": 30,
                },
            },
            "studio": {
                "image": "python:3.12-slim",
                "user": f"{os.getuid()}:{os.getgid()}",
                "depends_on": {"studio-db": {"condition": "service_healthy"}},
                "environment": {
                    "HOME": "/tmp",
                    "PYTHONPATH": "/src/dashboard/sparrow_dashboard:/tmp/deps",
                    "PYTHONUNBUFFERED": "1",
                    "SECRET_KEY": creds["secret_key"],
                    "DB_DRIVER": "postgres",
                    "DB_HOST": "studio-db",
                    "DB_PORT": "5432",
                    "DB_USER": "postgres",
                    "DB_PASSWORD": creds["pg_password"],
                    "DB_NAME": "postgres",
                    "STORAGE_BACKEND": "local",
                    "GALLERY_DIR": "/data/gallery",
                    "LOCAL_STORAGE_ROOT": "/data/storage",
                    "STAGING_DIR": "/data/staging",
                    "FINETUNE_MODELS_DIR": "/data/studio_models",
                    "SPARROW_FINETUNE_SERVICE_TOKEN": creds["service_token"],
                },
                "volumes": [f"{dash}:/src/dashboard:ro", f"{work}/data:/data"],
                "ports": [f"127.0.0.1:{port}:5000"],
                "command": ["sh", "-c", studio_cmd],
                "networks": {"default": {}, "finetune": {"aliases": ["studio"]}},
                "healthcheck": {
                    "test": ["CMD", "python", "-c", health],
                    "interval": "3s",
                    "retries": 100,
                    "start_period": "240s",
                },
            },
        },
        "networks": {"finetune": {"external": True, "name": network}},
        "volumes": {"studio-db": {}},
    }


def pick_images(
    data: Path, per_class: int, rng: random.Random
) -> list[tuple[str, Path]]:
    by_class: dict[str, list[Path]] = {c: [] for c in CLASSES}
    with (data / "annotations" / "val_cropped.csv").open() as f:
        for row in csv.DictReader(f):
            label = row["Label"]
            path = data / "cropped_resized" / row["Photo_ID"]
            if label in by_class and path.is_file():
                by_class[label].append(path)
    picked = []
    for label, paths in by_class.items():
        if len(paths) < per_class:
            raise SystemExit(f"only {len(paths)} {label} crops available")
        picked += [(label, p) for p in rng.sample(paths, per_class)]
    rng.shuffle(picked)
    return picked


def write_thumbnail_pair(src: Path, gallery: Path, key: str) -> str:
    """Write the gallery image and a 512px thumb the way Studio's local storage does.

    Studio's image grid only lists rows with ``gallery_thumb_key`` set.
    """
    dst = gallery / key
    dst.parent.mkdir(parents=True, exist_ok=True)
    thumb = dst.with_name(f"thumb_{dst.stem}.jpg")
    with Image.open(src) as im:
        rgb = im.convert("RGB")
        rgb.save(dst, format="JPEG", quality=92)
        rgb.thumbnail((512, 512))
        rgb.save(thumb, format="JPEG", quality=85)
    return str(thumb.relative_to(gallery)).replace("\\", "/")


def seed_labels(conn, pid: int) -> None:
    """Give the project a primary pipeline whose classifier labels are CLASSES.

    Studio's review label dropdown lists model_labels resolved through the
    project's primary pipeline; without one it offers only "Unlabelled".
    Class ids follow AI4G-Amazon-V2's labels.txt. Idempotent.
    """
    det, clf = "MDV6-yolov10-e", "AI4G-Amazon-V2"
    pipe = conn.execute(
        "INSERT INTO inference_pipelines (pipeline_key, pipeline_type, display_name, "
        "pipeline_kind, detector_key, detector_server_folder, classifier_key, "
        "classifier_server_folder, apply_when_detector_class_equals) "
        "VALUES (%s, 'detector_classifier', %s, 'camera_trap', %s, %s, %s, %s, 'animal') "
        "ON CONFLICT (pipeline_key) DO UPDATE SET display_name = EXCLUDED.display_name "
        "RETURNING id",
        (f"camera_trap:{det}__{clf}", f"{det} + {clf}", det, det, clf, clf),
    ).fetchone()[0]
    for class_id, name in enumerate(CLASSES):
        conn.execute(
            "INSERT INTO model_labels (pipeline_id, model_kind, model_key, class_id, "
            "label_key, display_name) VALUES (%s, 'classifier', %s, %s, %s, %s) "
            "ON CONFLICT (model_kind, model_key, class_id) DO NOTHING",
            (pipe, clf, class_id, name.lower(), name),
        )
    if not conn.execute(
        "SELECT 1 FROM project_pipelines WHERE project_id=%s AND is_primary", (pid,)
    ).fetchone():
        conn.execute(
            "INSERT INTO project_pipelines (project_id, pipeline_id, is_primary) "
            "VALUES (%s, %s, TRUE)",
            (pid, pipe),
        )


def seed(
    dsn: str, schema: Path, gallery: Path, data: Path, creds: dict, per_class: int
) -> dict:
    rng = random.Random(11)
    with psycopg.connect(dsn, autocommit=True) as conn:
        if conn.execute("SELECT to_regclass('public.users')").fetchone()[0]:
            pid = conn.execute("SELECT min(id) FROM projects").fetchone()[0]
            seed_labels(conn, pid)
            return {"project": pid, "seeded": False}
        conn.execute(schema.read_text())
        uid = conn.execute(
            "INSERT INTO users (username, email, password_hash, role, status, is_active, "
            "requires_2fa) VALUES (%s, %s, %s, 'admin', 'active', TRUE, FALSE) RETURNING id",
            (creds["username"], creds["email"], werkzeug_hash(creds["password"])),
        ).fetchone()[0]
        pid = conn.execute(
            "INSERT INTO projects (owner_user_id, project_type, name, description) "
            "VALUES (%s, 'batch', 'Amazon review test', "
            "'Real camera-trap crops for the fine-tune round trip') RETURNING id",
            (uid,),
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO project_members (project_id, user_id, member_role) "
            "VALUES (%s, %s, 'owner') ON CONFLICT DO NOTHING",
            (pid, uid),
        )
        seed_labels(conn, pid)
        counts = {"reviewed": 0, "unreviewed": 0, "wrong_guesses": 0}
        for n, (label, src) in enumerate(pick_images(data, per_class, rng)):
            reviewed = n % 3 != 0
            guess = label
            if not reviewed and rng.random() < 0.4:
                guess = rng.choice([c for c in CLASSES if c != label])
                counts["wrong_guesses"] += 1
            name = f"{n:03d}_{src.name}"
            iid = conn.execute(
                "INSERT INTO images (project_id, ingestion_type, image_key, original_filename, "
                "review_status, processing_status) VALUES (%s, 'upload', %s, %s, %s, 'done') "
                "RETURNING id",
                (
                    pid,
                    f"staging/{name}",
                    name,
                    "reviewed" if reviewed else "unreviewed",
                ),
            ).fetchone()[0]
            key = f"projects/{pid}/2026-10-19/{iid}__{name}"
            thumb_key = write_thumbnail_pair(src, gallery, key)
            with Image.open(src) as im:
                w, h = im.size
            conn.execute(
                "UPDATE images SET gallery_image_key=%s, gallery_thumb_key=%s WHERE id=%s",
                (key, thumb_key, iid),
            )
            det = conn.execute(
                "INSERT INTO detections (image_id, project_id, detection_count) "
                "VALUES (%s, %s, 1) RETURNING id",
                (iid, pid),
            ).fetchone()[0]
            conn.execute(
                "INSERT INTO detection_boxes (detection_id, confidence, x_min, y_min, x_max, "
                "y_max, classifier_class, classifier_confidence, is_human_reviewed) "
                "VALUES (%s, 0.9, 0, 0, %s, %s, %s, %s, %s)",
                (det, w - 1, h - 1, guess, round(rng.uniform(0.5, 0.95), 2), reviewed),
            )
            counts["reviewed" if reviewed else "unreviewed"] += 1
    return {"project": pid, "seeded": True, **counts}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--studio-root", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--project", default="sparrow-studio-live")
    ap.add_argument("--network", default="sparrow-ft-human-review_default")
    ap.add_argument("--port", type=int, default=18090)
    ap.add_argument("--pg-port", type=int, default=18091)
    ap.add_argument(
        "--data", type=Path, default=Path.home() / "datasets/amazon_camera_trap"
    )
    ap.add_argument("--per-class", type=int, default=18)
    ap.add_argument(
        "--down", action="store_true", help="stop and remove the stack and its DB"
    )
    args = ap.parse_args()

    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    (work / "data" / "gallery").mkdir(parents=True, exist_ok=True)
    (work / "data" / "studio_models").mkdir(parents=True, exist_ok=True)
    override = work / "studio_live.compose.json"
    base = ["docker", "compose", "-p", args.project, "-f", str(override)]
    if args.down:
        subprocess.run([*base, "down", "-v"], check=True)
        return 0

    creds_path = work / "studio_live.json"
    if creds_path.exists():
        creds = json.loads(creds_path.read_text())
    else:
        creds = {
            "username": "studio-tester",
            "email": "studio-tester@example.org",
            "password": secrets.token_urlsafe(18),
            "pg_password": secrets.token_urlsafe(18),
            "secret_key": secrets.token_hex(32),
            "service_token": secrets.token_urlsafe(32),
        }
        creds_path.write_text(json.dumps(creds, indent=2))
        creds_path.chmod(0o600)

    override.write_text(
        json.dumps(
            compose_doc(
                args.studio_root.resolve(),
                work,
                args.network,
                args.port,
                args.pg_port,
                creds,
            ),
            indent=2,
        )
    )
    t0 = time.time()
    subprocess.run([*base, "up", "-d", "--wait"], check=True)
    dsn = f"postgresql://postgres:{creds['pg_password']}@127.0.0.1:{args.pg_port}/postgres"
    info = seed(
        dsn,
        args.studio_root / "db" / "schema" / "01_schema.sql",
        work / "data" / "gallery",
        args.data,
        creds,
        args.per_class,
    )
    creds.update(
        studio_url=f"http://127.0.0.1:{args.port}",
        studio_url_from_finetune="http://studio:5000",
        studio_project_id=info["project"],
    )
    creds_path.write_text(json.dumps(creds, indent=2))
    print(
        json.dumps(
            {
                "ready_after_s": round(time.time() - t0),
                **info,
                "studio_url": creds["studio_url"],
                "credentials": str(creds_path),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
