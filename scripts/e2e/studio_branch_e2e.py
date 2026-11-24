"""End-to-end test: the fine-tune service (compose stack) against a real Sparrow Studio checkout.

Everything runs on this machine, in one throwaway compose project:

1. Studio: ``studio-db`` (Postgres 16 loaded with the Studio branch's ``db/schema/01_schema.sql``)
   and ``studio`` (the real dashboard, ``create_app`` from ``dashboard/sparrow_dashboard`` with the
   branch's ``dashboard/build/requirements.txt``) served by Flask. The project is seeded with
   synthetic two-class camera-trap images: JPEG originals in a local gallery and integer pixel
   boxes, stored the way Studio's inference worker stores them.
2. The fine-tune service from ``compose.service.yml`` with the GPU profile. It reaches Studio at
   ``http://studio:5000`` on the shared compose network. Studio runs as a container, not a host
   process, because this host's firewall drops container-to-host traffic.
3. A headless Chromium session through the fine-tune UI: sign in, create a workspace and a
   classification dataset, import the Studio project (crops), snapshot, train
   ``conventional_classifier``, then "Activate in Studio", which pushes the model bundle.
4. Checks on the Studio side: the ``finetune_model_import`` row, the stored bundle, its sha256,
   labels, and the pushed ONNX model classifying fresh synthetic crops.

Usage (from the fine-tune repo root; the ``sparrow-finetune:latest`` image must be built):

    uv run --extra dev --extra e2e python scripts/e2e/studio_branch_e2e.py \
        --studio-root <path to the Studio checkout> --out <artifact dir>

Exit code 0 only when every check passes. The compose project is removed at the end.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import secrets
import socket
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import psycopg
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]
CLASSES = ("lion", "zebra")
CHECKS: list[tuple[str, bool, str]] = []
STUDIO_IN_NETWORK = "http://studio:5000"


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))
    print(
        f"[{'PASS' if ok else 'FAIL'}] {name}{' — ' + detail if detail else ''}",
        flush=True,
    )


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run(cmd, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kw)


# --------------------------------------------------------------------------- synthetic data
def draw_animal(
    d: ImageDraw.ImageDraw, label: str, box: tuple[int, int, int, int], rng
) -> None:
    x0, y0, x1, y1 = box
    if label == "zebra":
        d.rectangle(box, fill=(235, 235, 235))
        step = max(4, (x1 - x0) // 8)
        for x in range(x0, x1, step * 2):
            d.rectangle((x, y0, min(x + step, x1), y1), fill=(20, 20, 20))
    else:
        tone = rng.randint(-15, 15)
        d.ellipse(box, fill=(196 + tone, 150 + tone, 80 + tone))


def scene(rng, n_animals: int, size=(640, 480)) -> tuple[Image.Image, list]:
    w, h = size
    base = np.array(
        [rng.randint(60, 110), rng.randint(90, 140), rng.randint(40, 80)], np.int16
    )
    noise = np.random.default_rng(rng.randint(0, 2**31)).integers(-25, 26, (h, w, 3))
    im = Image.fromarray(np.clip(base + noise, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    boxes = []
    for i in range(n_animals):
        label = rng.choice(CLASSES)
        bw, bh = rng.randint(90, 160), rng.randint(70, 130)
        x0 = rng.randint(0, w // 2 - bw) + (w // 2 if i % 2 else 0)
        y0 = rng.randint(0, h - bh)
        box = (x0, y0, x0 + bw, y0 + bh)
        draw_animal(d, label, box, rng)
        boxes.append((label, box))
    return im, boxes


def seed_studio(dsn: str, gallery: Path, schema: Path, n_images: int) -> dict:
    rng = random.Random(7)
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(schema.read_text())
        uid = conn.execute(
            "INSERT INTO users (username, email) VALUES ('e2e', 'e2e@example.org') RETURNING id"
        ).fetchone()[0]
        pid = conn.execute(
            "INSERT INTO projects (owner_user_id, project_type, name) "
            "VALUES (%s, 'batch', 'E2E Serengeti') RETURNING id",
            (uid,),
        ).fetchone()[0]
        counts = {"images": 0, "crops": 0}
        labels_seen: dict[str, int] = {}

        def add(name, status, im, boxes, deleted_extra=False):
            iid = conn.execute(
                "INSERT INTO images (project_id, ingestion_type, image_key, original_filename, "
                "review_status, processing_status) VALUES (%s, 'upload', %s, %s, %s, 'done') "
                "RETURNING id",
                (pid, f"staging/{name}", name, status),
            ).fetchone()[0]
            key = f"projects/{pid}/2026-10-11/{iid}__{name}"
            path = gallery / key
            path.parent.mkdir(parents=True, exist_ok=True)
            im.save(path, format="JPEG", quality=92)
            conn.execute(
                "UPDATE images SET gallery_image_key=%s WHERE id=%s", (key, iid)
            )
            rows = [(lb, b, False) for lb, b in boxes]
            if deleted_extra:  # a reviewer-deleted box: must not be imported
                rows.append(("lion", (5, 5, 60, 60), True))
            det = conn.execute(
                "INSERT INTO detections (image_id, project_id, detection_count) "
                "VALUES (%s, %s, %s) RETURNING id",
                (iid, pid, len(rows)),
            ).fetchone()[0]
            for lb, (x0, y0, x1, y1), deleted in rows:
                conn.execute(
                    "INSERT INTO detection_boxes (detection_id, confidence, x_min, y_min, x_max, "
                    "y_max, classifier_class, is_deleted) VALUES (%s, 0.9, %s, %s, %s, %s, %s, %s)",
                    (det, x0, y0, x1, y1, lb, deleted),
                )

        for i in range(n_images):
            im, boxes = scene(rng, 1 + (i % 2))
            add(f"cam{i:03d}.jpg", "reviewed", im, boxes, deleted_extra=(i == 0))
            counts["images"] += 1
            counts["crops"] += len(boxes)
            for lb, _ in boxes:
                labels_seen[lb] = labels_seen.get(lb, 0) + 1
        # Not reviewed: a reviewed-only import must skip it.
        im, boxes = scene(rng, 2)
        add("unreviewed.jpg", "unreviewed", im, boxes)
    return {
        "project": pid,
        "expected_images": counts["images"],
        "expected_crops": counts["crops"],
        "label_counts": labels_seen,
    }


# --------------------------------------------------------------------------- compose
def write_override(
    path: Path,
    studio_root: Path,
    work: Path,
    pg_port: int,
    studio_port: int,
    pg_password: str,
    token: str,
) -> None:
    dash = studio_root / "dashboard"
    uid_gid = f"{os.getuid()}:{os.getgid()}"
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
    doc = {
        "services": {
            "studio-db": {
                "image": "postgres:16-alpine",
                "environment": {"POSTGRES_PASSWORD": pg_password},
                "ports": [f"127.0.0.1:{pg_port}:5432"],
                "healthcheck": {
                    "test": ["CMD-SHELL", "pg_isready -U postgres"],
                    "interval": "2s",
                    "retries": 30,
                },
            },
            "studio": {
                "image": "python:3.12-slim",
                "user": uid_gid,
                "depends_on": {"studio-db": {"condition": "service_healthy"}},
                "environment": {
                    "HOME": "/tmp",
                    "PYTHONPATH": "/src/dashboard/sparrow_dashboard:/tmp/deps",
                    "PYTHONUNBUFFERED": "1",
                    "SECRET_KEY": secrets.token_hex(16),
                    "DB_DRIVER": "postgres",
                    "DB_HOST": "studio-db",
                    "DB_PORT": "5432",
                    "DB_USER": "postgres",
                    "DB_PASSWORD": pg_password,
                    "DB_NAME": "postgres",
                    "STORAGE_BACKEND": "local",
                    "GALLERY_DIR": "/data/gallery",
                    "LOCAL_STORAGE_ROOT": "/data/storage",
                    "STAGING_DIR": "/data/staging",
                    "FINETUNE_MODELS_DIR": "/data/studio_models",
                    "SPARROW_FINETUNE_SERVICE_TOKEN": token,
                },
                "volumes": [f"{dash}:/src/dashboard:ro", f"{work}:/data"],
                "ports": [f"127.0.0.1:{studio_port}:5000"],
                "command": ["sh", "-c", studio_cmd],
                "healthcheck": {
                    "test": ["CMD", "python", "-c", health],
                    "interval": "3s",
                    "retries": 100,
                    "start_period": "240s",
                },
            },
        }
    }
    path.write_text(json.dumps(doc, indent=2))  # JSON is valid YAML


def compose(project: str, files: list[Path], env_file: Path, *args, check_rc=True):
    cmd = ["docker", "compose", "-p", project]
    for f in files:
        cmd += ["-f", str(f)]
    cmd += ["--env-file", str(env_file), "--profile", "gpu", *args]
    return subprocess.run(cmd, cwd=REPO, check=check_rc, text=True, capture_output=True)


# --------------------------------------------------------------------------- browser
def drive_ui(
    base: str,
    user: str,
    password: str,
    project: int,
    seed: dict,
    shots: Path,
    train_timeout_s: int,
) -> dict:
    from playwright.sync_api import expect, sync_playwright

    problems: list[str] = []
    out: dict = {}
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 1000})
        page.set_default_timeout(30_000)
        page.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
        page.on(
            "console",
            lambda m: (
                problems.append(f"console: {m.text}") if m.type == "error" else None
            ),
        )
        try:
            page.goto(base + "/")
            page.fill("#login-username", user)
            page.fill("#login-password", password)
            page.click("#login-submit")
            expect(page.locator("#view-data")).to_be_visible()
            page.click("#show-new-workspace")  # the name field opens behind "+ New workspace"
            page.fill("#new-workspace", "Studio e2e")
            page.click("#create-workspace")
            expect(page.locator("#toast")).to_contain_text(
                'Workspace "Studio e2e" created'
            )
            page.click("#show-new-dataset")  # the name field opens behind "+ New dataset"
            page.fill("#new-dataset", "Serengeti crops")
            page.select_option("#new-dataset-task", "classification")
            page.click("#create-dataset")
            expect(page.locator("#topbar-dataset")).to_contain_text("Serengeti crops")
            expect(page.locator("#studio-panel")).to_be_visible()
            expect(page.locator("#studio-url")).to_contain_text(STUDIO_IN_NETWORK)
            check("UI shows the Studio connection", True, STUDIO_IN_NETWORK)

            # the project is a picker at the top; choosing one unfolds the import section
            page.select_option("#studio-project", str(project))
            expect(page.locator("#studio-body")).to_be_visible()
            page.check("#studio-reviewed-only")
            page.click("#studio-import-btn")
            toast = (
                f"Imported {seed['expected_crops']} item(s) from "
                f"{seed['expected_images']} Studio image(s)"
            )
            expect(page.locator("#toast")).to_contain_text(toast, timeout=120_000)
            check(
                "Studio import: crop and image counts (reviewed only, deleted box dropped)",
                True,
                toast,
            )
            expect(page.locator("#example-grid .example-card").first).to_be_visible()
            expect(page.locator("#example-grid .example-card")).to_have_count(
                seed["expected_crops"]
            )
            # Thumbnails are loading="lazy" inside a scrolling pane, so only some
            # load. Wait for decoded ones, reject any broken one, and fetch every
            # URL directly through the authenticated context.
            page.wait_for_function(
                "() => [...document.querySelectorAll('#example-grid img')]"
                ".some((i) => i.complete && i.naturalWidth > 0)"
            )
            page.wait_for_timeout(1500)
            broken = page.eval_on_selector_all(
                "#example-grid img",
                "els => els.filter((i) => i.complete && i.naturalWidth === 0).length",
            )
            srcs = page.eval_on_selector_all(
                "#example-grid img", "els => [...new Set(els.map((e) => e.src))]"
            )
            bad = [
                s
                for s in srcs
                if not (
                    (r := page.request.get(s)).ok
                    and r.headers.get("content-type", "").startswith("image/")
                )
            ]
            check(
                "imported crop thumbnails render",
                not bad and not broken,
                f"{len(srcs)} image URL(s), {len(bad)} failed fetch, "
                f"{broken} broken in page",
            )
            page.screenshot(path=str(shots / "01-studio-import.png"), full_page=True)

            page.select_option("#pipeline-select", "conventional_classifier")
            expect(page.locator("#create-snapshot")).to_be_enabled()
            page.click("#create-snapshot")
            expect(page.locator("#snapshot-info")).to_contain_text(
                f"{seed['expected_crops']} item(s)"
            )
            balance = page.locator("#class-balance").inner_text()
            check(
                "snapshot holds every crop and both classes",
                all(c in balance for c in CLASSES),
                balance.replace("\n", " "),
            )
            expect(page.locator("#start-training")).to_be_enabled()
            with page.expect_response(
                lambda r: r.url.endswith("/v1/jobs") and r.request.method == "POST",
                timeout=60_000,
            ) as job_resp:
                page.click("#start-training")
            check(
                "training job submitted",
                job_resp.value.status == 201,
                f"POST /v1/jobs -> {job_resp.value.status}",
            )
            expect(page.locator("#view-jobs")).to_be_visible(timeout=15_000)
            t0 = time.monotonic()
            status = page.locator("#job-status")
            state = None
            while time.monotonic() < t0 + train_timeout_s:
                state = status.get_attribute("data-state")
                if state in ("succeeded", "failed", "cancelled"):
                    break
                time.sleep(3)
            out["train_seconds"] = round(time.monotonic() - t0)
            out["events"] = page.locator("#event-list").inner_text()[-3000:]
            page.screenshot(path=str(shots / "02-job.png"), full_page=True)
            check(
                "training job succeeded",
                state == "succeeded",
                f"state={state} after {out['train_seconds']} s",
            )
            if state != "succeeded":
                return out

            page.click('.sidebar-nav [data-page="models"]')
            expect(page.locator("#model-list [data-model-id]")).to_have_count(1)
            expect(page.locator("#activate-studio")).to_be_visible()
            page.fill("#deploy-project", str(project))
            page.click("#activate-studio")
            badge = page.locator("#model-active-badge")
            # the badge names the Studio project, not its numeric id
            expect(badge).to_have_text("Active: studio/E2E Serengeti", timeout=120_000)
            check("Activate in Studio pushed the model", True, badge.inner_text())
            out["model_detail"] = page.locator("#model-detail").inner_text()
            page.screenshot(path=str(shots / "03-model-in-studio.png"), full_page=True)
            expect(page.locator("#error-banner")).to_be_hidden()
        except Exception:
            page.screenshot(path=str(shots / "99-failure.png"), full_page=True)
            raise
        finally:
            browser.close()
    check(
        "no browser page errors or console errors",
        not problems,
        "; ".join(problems)[:500],
    )
    return out


# --------------------------------------------------------------------------- studio checks
def verify_studio(dsn: str, models_dir: Path, project: int, out_dir: Path) -> None:
    with psycopg.connect(dsn) as conn:
        rows = conn.execute(
            "SELECT source_model_id, pipeline, task, format, labels, metrics, sha256, "
            "bundle_key, size_bytes FROM finetune_model_import WHERE project_id=%s",
            (project,),
        ).fetchall()
    check(
        "Studio recorded exactly one imported model",
        len(rows) == 1,
        f"{len(rows)} row(s)",
    )
    if not rows:
        return
    src_id, pipeline, task, fmt, labels, metrics, sha, key, size = rows[0]
    check(
        "import row: pipeline/task/format",
        (pipeline, task, fmt)
        == ("conventional_classifier", "classification", "onnx-bundle"),
        f"{pipeline}/{task}/{fmt} source_model_id={src_id}",
    )
    check("import row: labels", sorted(labels) == sorted(CLASSES), str(labels))
    bundle = models_dir / key
    check(
        "bundle stored under FINETUNE_MODELS_DIR",
        bundle.is_file() and bundle.stat().st_size == size,
        f"{key} ({size} bytes)",
    )
    if not bundle.is_file():
        return
    with tarfile.open(bundle, "r:gz") as tar:
        manifest = json.load(tar.extractfile("manifest.json"))
        artifact_name = next(n for n in tar.getnames() if n != "manifest.json")
        artifact = tar.extractfile(artifact_name).read()
    digest = hashlib.sha256(artifact).hexdigest()
    check(
        "bundle sha256 matches row and manifest",
        digest == sha.strip() == manifest["sha256"],
        f"{artifact_name} {digest[:16]}",
    )
    with zipfile.ZipFile(io.BytesIO(artifact)) as zf:
        inner = json.loads(zf.read("manifest.json"))
        onnx_bytes = zf.read("model.onnx")
        file_ok = all(
            hashlib.sha256(zf.read(n)).hexdigest() == h
            for n, h in inner["files"].items()
        )
    check(
        "service bundle inside Studio artifact is intact",
        file_ok and inner["labels"] == manifest["labels"],
        f"files={sorted(inner['files'])} parity={inner.get('parity')}",
    )
    (out_dir / "studio_manifest.json").write_text(json.dumps(manifest, indent=2))
    check(
        "manifest metrics present",
        isinstance(metrics, dict) and bool(metrics),
        json.dumps(metrics)[:300],
    )

    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    rng = random.Random(99)
    correct = total = 0
    for _ in range(40):
        im, boxes = scene(rng, 1)
        label, (x0, y0, x1, y1) = boxes[0]
        crop = im.crop((x0, y0, x1, y1)).resize((224, 224), Image.BILINEAR)
        x = (np.asarray(crop, np.float32) / 255.0).transpose(2, 0, 1)[None]
        probs = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
        correct += int(manifest["labels"][int(np.argmax(probs))] == label)
        total += 1
    check(
        "pushed ONNX model classifies 40 fresh synthetic crops (>= 90% correct)",
        correct / total >= 0.9,
        f"{correct}/{total} correct",
    )


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--studio-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--images", type=int, default=40)
    ap.add_argument("--train-timeout", type=int, default=1200)
    ap.add_argument(
        "--keep", action="store_true", help="leave the compose project running"
    )
    args = ap.parse_args()

    studio_root = args.studio_root.resolve()
    out = args.out.resolve()
    shots = out / "screenshots"
    shots.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    gallery, models = work / "gallery", work / "studio_models"
    for d in (gallery, models, work / "storage", work / "staging"):
        d.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    pg_password = secrets.token_urlsafe(16)
    ft_user, ft_password = "e2e-admin", secrets.token_urlsafe(16)
    project_name = f"ft-e2e-{secrets.token_hex(3)}"
    env_file = work / "service.env"
    override = work / "compose.e2e.yml"
    files = [REPO / "compose.service.yml", override]
    pg_port, studio_port, ft_port = free_port(), free_port(), free_port()

    commit = run(["git", "-C", str(studio_root), "rev-parse", "HEAD"]).stdout.strip()
    branch = run(
        ["git", "-C", str(studio_root), "rev-parse", "--abbrev-ref", "HEAD"]
    ).stdout.strip()
    ft_commit = run(["git", "-C", str(REPO), "rev-parse", "HEAD"]).stdout.strip()
    print(f"Studio {branch} @ {commit[:9]}; fine-tune @ {ft_commit[:9]}", flush=True)
    info: dict = {
        "studio_branch": branch,
        "studio_commit": commit,
        "finetune_commit": ft_commit,
    }

    env_file.write_text(
        "\n".join(
            [
                f"POSTGRES_PASSWORD={secrets.token_urlsafe(16)}",
                f"SPARROW_FT_PORT={ft_port}",
                f"SPARROW_FT_STUDIO_URL={STUDIO_IN_NETWORK}",
                f"SPARROW_FT_STUDIO_TOKEN={token}",
                "SPARROW_FT_AUTH_DISABLED=0",
            ]
        )
        + "\n"
    )
    os.chmod(env_file, 0o600)
    write_override(
        override, studio_root, work, pg_port, studio_port, pg_password, token
    )
    os.chmod(override, 0o600)
    dsn = (
        f"host=127.0.0.1 port={pg_port} user=postgres password={pg_password} "
        "dbname=postgres"
    )
    try:
        r = compose(
            project_name,
            files,
            env_file,
            "up",
            "-d",
            "--wait",
            "studio-db",
            check_rc=False,
        )
        if r.returncode:
            raise RuntimeError(f"studio-db did not start: {r.stderr[-800:]}")
        seed = seed_studio(
            dsn, gallery, studio_root / "db/schema/01_schema.sql", args.images
        )
        info["seed"] = seed
        check("Studio schema loaded and project seeded", True, json.dumps(seed))

        r = compose(
            project_name,
            files,
            env_file,
            "up",
            "-d",
            "--wait",
            "--no-build",
            check_rc=False,
        )
        (out / "compose_up.log").write_text(r.stdout + r.stderr)
        check(
            "compose up: Studio dashboard and fine-tune stack (postgres, migrate, api, "
            "workers) healthy",
            r.returncode == 0,
            f"project {project_name}",
        )
        if r.returncode:
            raise RuntimeError(
                "compose up failed; see compose_up.log and compose_logs.txt"
            )

        studio_host = f"http://127.0.0.1:{studio_port}/api/finetune/v1"
        req = urllib.request.Request(
            f"{studio_host}/projects/{seed['project']}/export",
            headers={"Authorization": "Bearer wrong-token"},
        )
        try:
            urllib.request.urlopen(req, timeout=5)
            check("Studio rejects a wrong service token", False, "200")
        except urllib.error.HTTPError as e:
            check("Studio rejects a wrong service token", e.code == 401, str(e.code))

        compose(
            project_name,
            files,
            env_file,
            "exec",
            "-T",
            "-e",
            f"SPARROW_FT_PASSWORD={ft_password}",
            "api",
            "sparrow-finetune",
            "create-user",
            ft_user,
        )
        info["ui"] = drive_ui(
            f"http://127.0.0.1:{ft_port}",
            ft_user,
            ft_password,
            seed["project"],
            seed,
            shots,
            args.train_timeout,
        )
        verify_studio(dsn, models, seed["project"], out)
    except Exception as exc:  # recorded as a failed check
        check(
            "harness completed without an exception",
            False,
            f"{type(exc).__name__}: {exc}",
        )
    finally:
        logs = compose(
            project_name, files, env_file, "logs", "--no-color", check_rc=False
        )
        (out / "compose_logs.txt").write_text(logs.stdout + logs.stderr)
        if not args.keep:
            compose(project_name, files, env_file, "down", "-v", check_rc=False)
            env_file.unlink(missing_ok=True)
            override.unlink(missing_ok=True)

    info["checks"] = [{"name": n, "pass": ok, "detail": d} for n, ok, d in CHECKS]
    (out / "result.json").write_text(json.dumps(info, indent=2, default=str))
    failed = [n for n, ok, _ in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
