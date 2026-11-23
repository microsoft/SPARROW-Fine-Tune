"""End-to-end test: training jobs in the CPU-only worker image (FT-12), offline.

One throwaway compose project from ``compose.service.yml`` with ``SPARROW_FT_IMAGE_TAG=cpu`` and
no ``gpu`` profile, so the only worker is ``worker-cpu`` running ``sparrow-finetune:cpu``. The
worker runs with ``HF_HUB_OFFLINE=1`` and ``SPARROW_FT_WEIGHTS_DIR=/weights``, where ``/weights``
holds the timm ResNet-50 weights (FT-13) and the AI4G-Amazon-V2 stock package (FT-16).

Data: real Amazon camera-trap crops for classification, synthetic box images for detection.
Jobs: ``conventional_classifier`` (device auto, stock_model AI4G-Amazon-V2), ``dinov3_linear``
(ONNX embedder) and ``yolo_detector`` (yolo11n, device cpu). Each must succeed, record a CPU
device, and ship an ONNX artifact.

Usage (from the repo root; build the image first with
``docker build -f Dockerfile.service --build-arg TORCH_VARIANT=cpu -t sparrow-finetune:cpu .``):

    uv run --extra dev --extra e2e python scripts/e2e/cpu_image_e2e.py \
        --amazon ~/datasets/amazon_camera_trap \
        --assets /tmp/ft_e2e_full_assets --stock <AI4G-Amazon-V2 package dir> \
        --timm-weights <resnet50 model.safetensors> --out <artifact dir>

Exit code 0 only when every check passes. The compose project is removed at the end.
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
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from studio_branch_e2e import CHECKS, REPO, check, free_port, scene  # noqa: E402
from studio_full_e2e import Api, job_seconds, wait_job  # noqa: E402

SPECIES = ("Dasyprocta", "Bos", "Pecari", "Mazama", "Cuniculus")
STOCK = "AI4G-Amazon-V2"


def compose(project: str, files: list[Path], env: dict, *args, check_rc=True):
    cmd = ["docker", "compose", "-p", project]
    for f in files:
        cmd += ["-f", str(f)]
    return subprocess.run(
        [*cmd, *args],
        cwd=REPO,
        check=check_rc,
        text=True,
        capture_output=True,
        env={**os.environ, **env},
    )


def pick_crops(amazon: Path, per_class: int) -> list[tuple[str, str, str]]:
    """``(photo_id, label, seq_group)`` rows, ``per_class`` per species, several sequences each."""
    by_label: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    with open(amazon / "annotations/train_cropped.csv", newline="") as f:
        for row in csv.DictReader(f):
            if (
                row["Label"] in SPECIES
                and (amazon / "cropped_resized" / row["Photo_ID"]).is_file()
            ):
                by_label[row["Label"]][row["Seq_ID"]].append(row["Photo_ID"])
    rng = random.Random(11)
    out = []
    for label in SPECIES:
        seqs = sorted(by_label[label])
        rng.shuffle(seqs)
        taken = 0
        for seq in seqs:
            for pid in by_label[label][seq][:2]:
                out.append((pid, label, f"{label}-{seq}"))
                taken += 1
                if taken == per_class:
                    break
            if taken == per_class:
                break
    return out


def write_override(path: Path, weights: Path, assets: Path, yolo_cwd: Path, image: str) -> None:
    worker = {
        "image": image,
        "environment": {"HF_HUB_OFFLINE": "1"},
        "volumes": [
            "ft-data:/data",
            f"{weights}:/weights:ro",
            f"{assets}:/assets:ro",
            f"{yolo_cwd}:/yolo",
        ],
        "working_dir": "/yolo",
    }
    doc = {"services": {"worker-cpu": worker}}
    path.write_text(json.dumps(doc, indent=2))


def artifact_check(api: Api, model: dict, out: Path) -> tuple[list[str], list[str]]:
    """Download the model artifact; return its ONNX members and every recorded ``device``."""
    status, data = api.call("GET", f"/models/{model['id']}/artifact")
    if status != 200 or not isinstance(data, bytes):
        raise RuntimeError(f"artifact download for model {model['id']} -> {status}")
    (out / f"model-{model['id']}.zip").write_bytes(data)
    zf = zipfile.ZipFile(io.BytesIO(data))
    onnx = [n for n in zf.namelist() if n.endswith(".onnx")]
    devices = []
    for n in zf.namelist():
        if n.endswith(".json"):
            try:
                doc = json.loads(zf.read(n))
            except ValueError:
                continue
            if isinstance(doc, dict) and "device" in doc:
                devices.append(str(doc["device"]))
    meta = model.get("metadata") or {}
    if isinstance(meta, dict) and "device" in meta:
        devices.append(str(meta["device"]))
    return onnx, devices


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--amazon", type=Path, required=True)
    ap.add_argument("--assets", type=Path, required=True)
    ap.add_argument("--stock", type=Path, required=True)
    ap.add_argument("--timm-weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--image-tag", default="cpu", help="tag of the CPU-only image under test")
    ap.add_argument("--per-class", type=int, default=12)
    ap.add_argument("--train-timeout", type=int, default=3600)
    ap.add_argument("--keep", action="store_true", help="leave the stack running")
    args = ap.parse_args()

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    weights = work / "weights"
    (weights / "timm/resnet50").mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.timm_weights, weights / "timm/resnet50/model.safetensors")
    shutil.copytree(args.stock, weights / STOCK, dirs_exist_ok=True)
    yolo_cwd = work / "yolo_cwd"
    yolo_cwd.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.assets / "yolo11n.pt", yolo_cwd / "yolo11n.pt")
    os.chmod(yolo_cwd, 0o777)

    project = "ftcpu" + secrets.token_hex(3)
    port = free_port()
    token = secrets.token_urlsafe(24)
    env = {
        "POSTGRES_PASSWORD": secrets.token_urlsafe(16),
        "SPARROW_FT_IMAGE_TAG": args.image_tag,
        "TORCH_VARIANT": "cpu",
        "SPARROW_FT_PORT": str(port),
        "SPARROW_FT_API_TOKENS": token,
        "SPARROW_FT_AUTH_DISABLED": "0",
        "SPARROW_FT_EMBEDDER": "onnx:/assets/dinov3_vits16.onnx",
        "SPARROW_FT_WEIGHTS_DIR": "/weights",
    }
    override = work / "override.json"
    image = f"sparrow-finetune:{args.image_tag}"
    write_override(override, weights.resolve(), args.assets.resolve(), yolo_cwd, image)
    files = [REPO / "compose.service.yml", override]
    info: dict = {"project": project}
    try:
        r = compose(
            project, files, env, "up", "-d", "--wait", "--no-build", check_rc=False
        )
        (out / "compose_up.log").write_text(r.stdout + r.stderr)
        check(
            "compose up (CPU image, no gpu profile) healthy", r.returncode == 0, project
        )
        if r.returncode:
            raise RuntimeError("compose up failed; see compose_up.log")

        ps = compose(project, files, env, "ps", "--format", "{{.Service}} {{.Image}}")
        services = dict(line.split(" ", 1) for line in ps.stdout.split("\n") if line)
        check(
            f"only worker-cpu runs, on {image}",
            "worker-gpu" not in services
            and services.get("worker-cpu") == image,
            json.dumps(services),
        )
        probe = compose(
            project,
            files,
            env,
            "exec",
            "-T",
            "worker-cpu",
            "python",
            "-c",
            "import torch, onnxruntime as o; print(torch.__version__, "
            "torch.cuda.is_available(), torch.version.cuda, o.get_available_providers())",
            check_rc=False,
        )
        torch_line = probe.stdout.strip()
        check(
            "worker torch is the CPU build (no CUDA runtime)",
            probe.returncode == 0
            and "+cpu" in torch_line
            and " False None " in torch_line,
            torch_line or probe.stderr[-300:],
        )

        api = Api(f"http://127.0.0.1:{port}", token)
        ws = api.ok("POST", "/workspaces", {"name": "CPU e2e"})["id"]

        # Classification: real Amazon crops, sequences as groups.
        cls = api.ok(
            "POST",
            f"/workspaces/{ws}/datasets",
            {"name": "Amazon crops", "task": "classification", "whole_image": True},
        )["id"]
        crops = pick_crops(args.amazon, args.per_class)
        items = []
        for pid, label, group in crops:
            name = pid.replace("/", "__")
            api.ok(
                "PUT",
                f"/datasets/{cls}/images/{name}",
                raw=(args.amazon / "cropped_resized" / pid).read_bytes(),
            )
            items.append(
                {
                    "image": name,
                    "annotations": [{"label": label, "bbox": None}],
                    "group": group,
                }
            )
        api.ok("POST", f"/datasets/{cls}/annotations", {"items": items})
        snap_cls = api.ok("POST", f"/datasets/{cls}/snapshots", {})
        check(
            "classification snapshot from real Amazon crops",
            snap_cls.get("item_count", len(items)) == len(items),
            f"{len(items)} crops, {len(SPECIES)} species",
        )

        # Detection: synthetic box scenes.
        det = api.ok(
            "POST",
            f"/workspaces/{ws}/datasets",
            {"name": "Synthetic boxes", "task": "detection"},
        )["id"]
        rng = random.Random(5)
        det_items = []
        for i in range(24):
            im, boxes = scene(rng, 1 + i % 2, size=(320, 240))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=90)
            name = f"scene{i:03d}.jpg"
            api.ok("PUT", f"/datasets/{det}/images/{name}", raw=buf.getvalue())
            det_items.append(
                {
                    "image": name,
                    "annotations": [
                        {"label": lab, "bbox": [x0, y0, x1 - x0, y1 - y0]}
                        for lab, (x0, y0, x1, y1) in boxes
                    ],
                    "group": f"g{i % 8}",
                }
            )
        api.ok("POST", f"/datasets/{det}/annotations", {"items": det_items})
        snap_det = api.ok("POST", f"/datasets/{det}/snapshots", {})

        jobs = {
            "conventional_classifier": (
                snap_cls["id"],
                {
                    "epochs": 2,
                    "warmup_epochs": 1,
                    "batch": 8,
                    "device": "auto",
                    "stock_model": STOCK,
                },
            ),
            "dinov3_linear": (snap_cls["id"], {}),
            "yolo_detector": (
                snap_det["id"],
                {
                    "model": "yolo11n.pt",
                    "epochs": 2,
                    "imgsz": 160,
                    "batch": 8,
                    "device": "cpu",
                },
            ),
        }
        ids = {}
        for pipeline, (snap, params) in jobs.items():
            st, body = api.call(
                "POST",
                "/jobs",
                {"pipeline": pipeline, "snapshot_id": snap, "params": params},
            )
            check(f"{pipeline} job accepted", st == 201, f"{st}: {str(body)[:300]}")
            if st == 201:
                ids[pipeline] = body["id"]

        info["jobs"] = {}
        for pipeline, jid in ids.items():
            job = wait_job(api, jid, args.train_timeout)
            models = api.ok("GET", f"/models?job_id={jid}")["models"]
            ok = job["state"] == "succeeded" and len(models) == 1
            check(
                f"{pipeline} trains in the CPU image",
                ok,
                f"job {jid} {job['state']}, ran {job_seconds(job)}"
                + (
                    f" metrics={json.dumps(models[0]['metrics'])[:200]}"
                    if models
                    else f" error={str(job.get('error'))[:400]}"
                ),
            )
            info["jobs"][pipeline] = {
                "job": jid,
                "state": job["state"],
                "ran": job_seconds(job),
            }
            if not ok:
                continue
            model = api.ok("GET", f"/models/{models[0]['id']}")
            onnx, devices = artifact_check(api, model, out)
            check(f"{pipeline} artifact ships ONNX", bool(onnx), ", ".join(onnx))
            if pipeline != "dinov3_linear":
                check(
                    f"{pipeline} records a CPU device",
                    bool(devices) and all(d.startswith("cpu") for d in devices),
                    ", ".join(devices) or "no device recorded",
                )
            if pipeline == "conventional_classifier":
                sc = model.get("stock_comparison") or {}
                check(
                    "conventional_classifier scored the stock AI4G package offline (FT-13 + FT-16)",
                    sc.get("model") == STOCK and (sc.get("crops") or 0) > 0,
                    json.dumps(sc),
                )
                info["stock_comparison"] = sc

        logs = compose(
            project, files, env, "logs", "--no-color", "worker-cpu", check_rc=False
        )
        (out / "worker_cpu.log").write_text(logs.stdout + logs.stderr)
        check(
            "worker made no Hugging Face hub request (offline)",
            "huggingface.co" not in logs.stdout
            and "OfflineModeIsEnabled" not in logs.stdout,
        )
    except Exception as exc:  # noqa: BLE001
        check("harness ran to completion", False, f"{type(exc).__name__}: {exc}")
    finally:
        logs = compose(project, files, env, "logs", "--no-color", check_rc=False)
        (out / "compose_logs.txt").write_text(logs.stdout + logs.stderr)
        if not args.keep:
            compose(
                project, files, env, "down", "-v", "--remove-orphans", check_rc=False
            )

    passed = sum(ok for _, ok, _ in CHECKS)
    info["checks"] = [{"name": n, "ok": ok, "detail": d} for n, ok, d in CHECKS]
    info["passed"], info["total"] = passed, len(CHECKS)
    (out / "result.json").write_text(json.dumps(info, indent=2))
    print(f"\n{passed}/{len(CHECKS)} checks passed; artifacts in {out}")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
