"""End-to-end feature matrix: every fine-tune feature, and the combinations users chain together.

One throwaway compose project on this machine: a seeded Sparrow Studio (database and dashboard)
and the fine-tune service with the GPU profile, the DINOv3 ONNX embedder, the MegaDetector v6
compact package (assisted labelling) and checkpoint (YOLO starting weights), and offline timm
resnet50 weights. Scenarios, all through the HTTP API unless named otherwise:

1. Annotation formats: Camera Trap DP, COCO and Pascal VOC export from a dataset and from a
   snapshot, round-trip import into a fresh dataset, confirmed/excluded protection, overwrite,
   VOC refused on classification, excluded items left out of exports.
2. Studio import into a detection dataset plus uploaded images; per-source assist summary.
3. YOLO detector jobs: YOLO11n with an automatic batch on Studio + uploaded data, MegaDetector v6
   compact with a set batch, MegaDetector v6 extra (downloaded from Zenodo); per-class metrics;
   parameter validation.
4. Assisted labelling: MegaDetector on real camera-trap images with an animal label; the trained
   YOLO11n model by source (upload / studio), with and without overwrite; error cases.
5. Classifiers: DINOv3 linear probe twice on one snapshot (embedding cache hits) and the
   conventional classifier; calibrated threshold in each bundle.
6. Job and model lifecycle: cancel, retry, rename, local activation, deployments.
7. The combined chain: Studio import -> assist -> export -> snapshot -> train.
8. A headless browser pass over the web app controls for all of the above.

Usage (from the repo root; ``sparrow-finetune:latest`` must be built):

    uv run --extra dev --extra e2e python scripts/e2e/features_matrix_e2e.py \
        --studio-root <Studio checkout> --out <artifact dir>

Exit code 0 only when every check passes.
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
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from studio_branch_e2e import (  # noqa: E402
    CHECKS,
    REPO,
    check,
    compose,
    free_port,
    run,
    scene,
    seed_studio,
    write_override,
)
from studio_full_e2e import Api, items_of, wait_job  # noqa: E402

# A yolo11n trained 40 epochs on about 20 synthetic images scores its detections at only
# 0.03-0.12 (ONNX and torch agree; offline probe 2026-10-27), so the assist plumbing checks
# use a low cutoff. They test that boxes flow through, not how good the model is.
TRAINED_CONF = 0.01

MD_PACKAGE = Path(
    os.environ.get("SPARROW_FT_TEST_MEGADETECTOR", Path.home() / "repos/SparrowOPS/models/MDV6-yolov10-c")
)
MD_C_PT = "MDV6-yolov10-c.pt"
TIMM_RESNET50 = Path.home() / ".cache/huggingface/hub/models--timm--resnet50.a1_in1k"
AMAZON = Path.home() / "datasets/amazon_camera_trap"


class Ctx:
    pass


def jpeg(im) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def xywh(box) -> list[float]:
    x0, y0, x1, y1 = box
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]


def boxes_of(api: Api, ds: int) -> dict[str, list]:
    return {
        name: sorted(
            (a["label"], tuple(round(v) for v in a["bbox"]) if a["bbox"] else None)
            for a in it["annotations"] or []
        )
        for name, it in items_of(api, ds).items()
    }


def events_text(api: Api, job_id: int) -> str:
    return "\n".join(
        e["message"] for e in api.ok("GET", f"/jobs/{job_id}/events")["events"]
    )


def wait_state(api: Api, job_id: int, states: tuple, timeout_s: int) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        job = api.ok("GET", f"/jobs/{job_id}")
        if job["state"] in states:
            return job
        time.sleep(2)
    raise TimeoutError(f"job {job_id} never reached {states}")


def bundle_manifest(api: Api, model_id: int) -> dict:
    status, data = api.call("GET", f"/models/{model_id}/artifact")
    if status != 200 or not isinstance(data, bytes):
        raise RuntimeError(f"artifact of model {model_id}: {status}")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        return json.loads(zf.read("manifest.json"))


def export_names(fmt: str, data) -> set[str]:
    """Image names an export carries."""
    if fmt == "coco":
        return {im["file_name"] for im in data["images"]}
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        if fmt == "voc":
            return {
                n.rsplit("/", 1)[-1].rsplit(".", 1)[0]
                for n in zf.namelist()
                if n.endswith(".xml")
            }
        media = next(n for n in zf.namelist() if n.endswith("media.csv"))
        rows = csv.DictReader(io.StringIO(zf.read(media).decode()))
        return {r.get("fileName") or r["filePath"].rsplit("/", 1)[-1] for r in rows}


def raw_body(data) -> bytes:
    return json.dumps(data).encode() if isinstance(data, (dict, list)) else data


# --------------------------------------------------------------------------- 1. formats
def upload_scenes(api: Api, ds: int, n: int, seed: int, prefix: str) -> dict[str, list]:
    rng = random.Random(seed)
    truth = {}
    for i in range(n):
        im, boxes = scene(rng, 1 + i % 2)
        name = f"{prefix}{i:03d}.jpg"
        data = jpeg(im)
        api.ok("PUT", f"/datasets/{ds}/images/{name}", raw=data)
        truth[name] = (boxes, data)
    return truth


def scenario_formats(c: Ctx) -> None:
    api = c.api
    det = api.ok(
        "POST", f"/workspaces/{c.ws}/datasets", {"name": "fmt det", "task": "detection"}
    )["id"]
    uploaded = upload_scenes(api, det, 8, 11, "f")
    truth = {n: b for n, (b, _) in uploaded.items()}
    c.fmt_bytes = {n: d for n, (_, d) in uploaded.items()}
    api.ok(
        "POST",
        f"/datasets/{det}/annotations",
        {
            "items": [
                {
                    "image": n,
                    "annotations": [{"label": lb, "bbox": xywh(b)} for lb, b in bx],
                }
                for n, bx in truth.items()
            ]
        },
    )
    api.ok("PATCH", f"/datasets/{det}/items/f000.jpg", {"review_status": "confirmed"})
    api.ok("PATCH", f"/datasets/{det}/items/f001.jpg", {"review_status": "excluded"})
    expected = boxes_of(api, det)
    kept = sorted(n for n in expected if n != "f001.jpg")
    snap = api.ok("POST", f"/datasets/{det}/snapshots", {})["id"]

    exports = {}
    for fmt in ("camtrapdp", "coco", "voc"):
        st, data = api.call("GET", f"/datasets/{det}/export/{fmt}")
        names = export_names(fmt, data) if st == 200 else set()
        stems = {n.rsplit(".", 1)[0] for n in kept}
        got = {n.rsplit(".", 1)[0] for n in names}
        check(
            f"detection export {fmt}: excluded image left out, all others present",
            st == 200 and got == stems,
            f"{st} {sorted(got)[:4]}… ({len(got)})",
        )
        exports[fmt] = data
        st2, sdata = api.call("GET", f"/snapshots/{snap}/export/{fmt}")
        snames = (
            {n.rsplit(".", 1)[0] for n in export_names(fmt, sdata)}
            if st2 == 200
            else set()
        )
        check(
            f"detection snapshot export {fmt}",
            st2 == 200 and snames == stems,
            f"{st2} {len(snames)} images",
        )

        # Round trip into a fresh dataset holding the same image files.
        fresh = api.ok(
            "POST",
            f"/workspaces/{c.ws}/datasets",
            {"name": f"rt det {fmt}", "task": "detection"},
        )["id"]
        for n in truth:
            api.ok(
                "PUT",
                f"/datasets/{fresh}/images/{n}",
                raw=c.fmt_bytes[n],
            )
        res = api.ok("POST", f"/datasets/{fresh}/import/{fmt}", raw=raw_body(data))
        got_boxes = {n: b for n, b in boxes_of(api, fresh).items() if b}
        want = {n: expected[n] for n in kept}
        check(
            f"detection round trip {fmt}: boxes and labels equal the source",
            got_boxes == want and res["matched"] == len(kept),
            f"matched={res['matched']} boxes={res['boxes']}"
            + (
                ""
                if got_boxes == want
                else f" diff on {sorted(set(got_boxes) ^ set(want)) or [n for n in want if got_boxes.get(n) != want[n]][:2]}"
            ),
        )

    # Protection: a moved box on a pending image is restored; a confirmed one only with overwrite.
    api.ok(
        "PATCH",
        f"/datasets/{det}/items/f002.jpg",
        {"annotations": [{"label": "lion", "bbox": [1, 1, 30, 30]}]},
    )
    api.ok(
        "PATCH",
        f"/datasets/{det}/items/f000.jpg",
        {"annotations": [{"label": "zebra", "bbox": [2, 2, 40, 40]}]},
    )
    api.ok("PATCH", f"/datasets/{det}/items/f000.jpg", {"review_status": "confirmed"})
    res = api.ok("POST", f"/datasets/{det}/import/coco", raw=raw_body(exports["coco"]))
    now = boxes_of(api, det)
    skipped = {s["file"] for s in res["skipped"]}
    check(
        "import without overwrite: pending image restored, confirmed image kept and reported",
        now["f002.jpg"] == expected["f002.jpg"]
        and now["f000.jpg"] == [("zebra", (2, 2, 40, 40))]
        and "f000.jpg" in skipped,
        json.dumps(res["skipped"])[:200],
    )
    res = api.ok(
        "POST",
        f"/datasets/{det}/import/coco?overwrite=true",
        raw=raw_body(exports["coco"]),
    )
    now = boxes_of(api, det)
    check(
        "import with overwrite replaces the confirmed image's boxes",
        now["f000.jpg"] == expected["f000.jpg"],
        f"skipped={res['skipped']}",
    )
    st, exc_only = api.call("GET", f"/datasets/{det}/export/coco?status=excluded")
    check(
        "export ?status=excluded carries only the excluded image",
        st == 200 and export_names("coco", exc_only) == {"f001.jpg"},
        str(st),
    )
    api.ok("PATCH", f"/datasets/{det}/items/f001.jpg", {"annotations": []})
    res = api.ok(
        "POST", f"/datasets/{det}/import/coco?overwrite=true", raw=raw_body(exc_only)
    )
    check(
        "an excluded image is never overwritten by an import, even with overwrite",
        boxes_of(api, det)["f001.jpg"] == []
        and any(s["file"] == "f001.jpg" for s in res["skipped"]),
        json.dumps(res["skipped"])[:200],
    )
    c.fmt_det = det

    # Classification: COCO and Camera Trap DP round trip; VOC refused both ways.
    cls = api.ok(
        "POST",
        f"/workspaces/{c.ws}/datasets",
        {"name": "fmt cls", "task": "classification"},
    )["id"]
    rng = random.Random(5)
    for i in range(6):
        im, boxes = scene(rng, 1)
        crop = im.crop(boxes[0][1])
        api.ok("PUT", f"/datasets/{cls}/images/k{i:02d}.jpg", raw=jpeg(crop))
        api.ok(
            "POST",
            f"/datasets/{cls}/annotations",
            {
                "items": [
                    {
                        "image": f"k{i:02d}.jpg",
                        "annotations": [{"label": boxes[0][0], "bbox": None}],
                    }
                ]
            },
        )
        c.cls_bytes = getattr(c, "cls_bytes", {})
        c.cls_bytes[f"k{i:02d}.jpg"] = jpeg(crop)
    want = boxes_of(api, cls)
    for fmt in ("camtrapdp", "coco"):
        st, data = api.call("GET", f"/datasets/{cls}/export/{fmt}")
        fresh = api.ok(
            "POST",
            f"/workspaces/{c.ws}/datasets",
            {"name": f"rt cls {fmt}", "task": "classification"},
        )["id"]
        for n in want:
            api.ok("PUT", f"/datasets/{fresh}/images/{n}", raw=c.cls_bytes[n])
        res = api.ok("POST", f"/datasets/{fresh}/import/{fmt}", raw=raw_body(data))
        check(
            f"classification round trip {fmt}: labels equal",
            st == 200 and boxes_of(api, fresh) == want,
            f"matched={res['matched']}",
        )
    st, _ = api.call("GET", f"/datasets/{cls}/export/voc")
    st2, _ = api.call(
        "POST", f"/datasets/{cls}/import/voc", raw=raw_body(exports["voc"])
    )
    check(
        "VOC export and import refused on a classification dataset (422)",
        st == 422 and st2 == 422,
        f"{st} {st2}",
    )
    # Camera Trap DP "human" maps to the "person" label.
    buf = io.BytesIO()
    with (
        zipfile.ZipFile(io.BytesIO(exports["camtrapdp"])) as src,
        zipfile.ZipFile(buf, "w") as dst,
    ):
        for n in src.namelist():
            body = src.read(n)
            if n.endswith("observations.csv"):
                rows = list(csv.DictReader(io.StringIO(body.decode())))
                for r in rows:
                    if r.get("scientificName"):
                        r["scientificName"] = ""
                        r["observationType"] = "human"
                out = io.StringIO()
                w = csv.DictWriter(out, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
                body = out.getvalue().encode()
            dst.writestr(n, body)
    fresh = api.ok(
        "POST",
        f"/workspaces/{c.ws}/datasets",
        {"name": "rt human", "task": "detection"},
    )["id"]
    for n in truth:
        api.ok("PUT", f"/datasets/{fresh}/images/{n}", raw=c.fmt_bytes[n])
    # The alias applies when the dataset already uses "person" (as MegaDetector boxes do).
    api.ok(
        "POST",
        f"/datasets/{fresh}/annotations",
        {
            "items": [
                {
                    "image": "f000.jpg",
                    "annotations": [{"label": "person", "bbox": [1, 1, 9, 9]}],
                }
            ]
        },
    )
    res = api.ok("POST", f"/datasets/{fresh}/import/camtrapdp", raw=buf.getvalue())
    labels = {lb for bx in boxes_of(api, fresh).values() for lb, _ in bx}
    check(
        "Camera Trap DP human observations import as 'person'",
        labels == {"person"},
        f"labels={labels} new={res.get('new_labels')}",
    )


# --------------------------------------------------------------------------- 2-4. detection chain
def scenario_detection_chain(c: Ctx) -> None:
    api = c.api
    ds = api.ok(
        "POST",
        f"/workspaces/{c.ws}/datasets",
        {"name": "combo det", "task": "detection"},
    )["id"]
    c.combo = ds
    res = api.ok(
        "POST",
        "/connectors/studio/import",
        {"project": str(c.seed["project"]), "dataset_id": ds, "reviewed_only": False},
    )
    n_studio = c.seed["expected_images"] + 1
    check(
        "Studio import into a detection dataset",
        res.get("imported_images") == n_studio,
        json.dumps(res)[:200],
    )
    rng = random.Random(23)
    c.uploads = {}
    for i in range(6):
        im, _ = scene(rng, 1 + i % 2)
        name = f"up{i:02d}.jpg"
        api.ok("PUT", f"/datasets/{ds}/images/{name}", raw=jpeg(im))
        c.uploads[name] = im
    summ = api.ok("GET", f"/datasets/{ds}/assist/summary")["sources"]
    check(
        "assist summary splits images by source",
        summ["upload"]["images"] == 6
        and summ["upload"]["empty"] == 6
        and summ["studio"]["images"] == n_studio
        and summ["studio"]["empty"] == 0,
        json.dumps(summ),
    )
    st, out = api.call(
        "POST",
        f"/datasets/{ds}/assist",
        {"model_id": "megadetector", "sources": ["upload"], "conf": 0.9},
    )
    check(
        "MegaDetector on synthetic uploads runs and reports only upload images",
        st == 200
        and out["processed"] == 6
        and all(s["image"].startswith("up") for s in out["skipped"]),
        f"{st} processed={out.get('processed') if isinstance(out, dict) else out}",
    )
    # Bring the uploads back to empty for the trained-model assist below.
    existing = items_of(api, ds)
    for name in c.uploads:
        if name in existing:
            api.ok("PATCH", f"/datasets/{ds}/items/{name}", {"annotations": []})

    # Exclude one Studio image: it must stay out of assist, exports and training.
    api.ok("PATCH", f"/datasets/{ds}/items/cam001.jpg", {"review_status": "excluded"})
    snap = api.ok("POST", f"/datasets/{ds}/snapshots", {})
    c.combo_snap = snap["id"]
    st, data = api.call("GET", f"/snapshots/{snap['id']}/export/camtrapdp")
    names = export_names("camtrapdp", data) if st == 200 else set()
    check(
        "combined chain: snapshot export after Studio import + exclusion",
        st == 200
        and "cam001.jpg" not in names
        and len(names) >= c.seed["expected_images"] - 1,
        f"{st} {len(names)} images",
    )

    # Parameter validation before queueing.
    for params, what in (
        ({"imgsz": 330}, "imgsz not a multiple of 32"),
        ({"model": "mdv6-yolov10-x"}, "unknown MegaDetector base"),
        ({"batch": "big"}, "batch not an int or auto"),
        ({"epochs": 0}, "epochs 0"),
    ):
        st, body = api.call(
            "POST",
            "/jobs",
            {"pipeline": "yolo_detector", "snapshot_id": snap["id"], "params": params},
        )
        check(f"yolo job refused: {what} -> 422", st == 422, f"{st}: {str(body)[:120]}")

    jobs = {
        "yolo11n auto batch": {
            "model": "yolo11n.pt",
            "epochs": 40,
            "imgsz": 320,
            "batch": "auto",
        },
        "MDV6 compact batch 8": {
            "model": "mdv6-yolov10-c",
            "epochs": 3,
            "imgsz": 320,
            "batch": 8,
        },
        "MDV6 extra (Zenodo)": {
            "model": "mdv6-yolov10-e",
            "epochs": 1,
            "imgsz": 320,
            "batch": 4,
        },
    }
    ids = {
        k: api.ok(
            "POST",
            "/jobs",
            {"pipeline": "yolo_detector", "snapshot_id": snap["id"], "params": p},
        )["id"]
        for k, p in jobs.items()
    }
    c.det_models = {}
    for what, jid in ids.items():
        job = wait_job(api, jid, c.train_timeout)
        models = api.ok("GET", f"/models?job_id={jid}")["models"]
        ev = events_text(api, jid)
        ok = job["state"] == "succeeded" and len(models) == 1
        check(
            f"yolo job {what} succeeds",
            ok,
            f"job {jid} {job['state']} {str(job.get('error'))[:300] if not ok else ''}",
        )
        if not ok:
            continue
        m = models[0]
        c.det_models[what] = m
        per_class = (m.get("metadata") or {}).get("per_class") or []
        check(
            f"{what}: per-class metrics for lion and zebra",
            {"ap50_95.lion", "ap50_95.zebra"} <= set(m["metrics"])
            and {r["label"] for r in per_class} >= {"lion", "zebra"},
            str(sorted(k for k in m["metrics"] if "." in k))[:200],
        )
        man = bundle_manifest(api, m["id"])
        check(
            f"{what}: bundle records the starting weights",
            man["base_model"] == jobs[what]["model"]
            and man["output"] == ("raw" if what.startswith("yolo11n") else "end2end"),
            f"{man['base_model']} {man['output']}",
        )
        if what.startswith("yolo11n"):
            check(
                "yolo11n: automatic batch chosen and reported",
                "(auto)" in ev and "batch" in ev,
                next(
                    (ln for ln in ev.splitlines() if "training on" in ln),
                    "no 'training on' event",
                ),
            )
        elif "compact" in what:
            check(
                "MDV6 compact: starts from local MegaDetector weights, batch set by parameter",
                "starting from MegaDetector v6 compact weights" in ev
                and "batch 8 (set by parameter)" in ev
                and "downloading" not in ev,
                "; ".join(
                    [
                        ln
                        for ln in ev.splitlines()
                        if "MegaDetector" in ln or "training on" in ln
                    ][:2]
                ),
            )
        else:
            check(
                "MDV6 extra: weights downloaded from Zenodo and verified, then trained",
                "downloading" in ev and "MegaDetector v6 extra" in ev,
                "; ".join(
                    [
                        ln
                        for ln in ev.splitlines()
                        if "MegaDetector" in ln or "download" in ln
                    ][:3]
                ),
            )

    trained = c.det_models.get("yolo11n auto batch")
    if trained is None:
        check("assist with the trained detector", False, "yolo11n job failed")
        return
    # Assist with the trained model, by source.
    out = api.ok(
        "POST",
        f"/datasets/{ds}/assist",
        {"model_id": trained["id"], "sources": ["upload"], "conf": TRAINED_CONF},
    )
    written = {w["image"] for w in out["items"]}
    check(
        "trained detector assist, uploads only: writes boxes on uploads, no Studio image touched",
        out["processed"] == 6
        and written
        and written <= set(c.uploads)
        and not out["unknown_labels"],
        f"processed={out['processed']} written={len(written)} boxes={out['boxes']}",
    )
    after = items_of(api, ds)
    check(
        "assisted items are pending for review",
        all(after[n]["review_status"] == "pending" for n in written),
        "",
    )
    out = api.ok(
        "POST",
        f"/datasets/{ds}/assist",
        {"model_id": trained["id"], "sources": ["upload"], "conf": TRAINED_CONF},
    )
    check(
        "assist again without overwrite skips images that now have boxes",
        out["processed"] == 6 - len(written)
        and out["items_written"] <= 6 - len(written),
        f"processed={out['processed']}",
    )
    out = api.ok(
        "POST",
        f"/datasets/{ds}/assist",
        {"model_id": trained["id"], "sources": ["studio"], "conf": TRAINED_CONF},
    )
    check(
        "assist on Studio images without overwrite: none processed (they already have boxes)",
        out["processed"] == 0,
        f"processed={out['processed']}",
    )
    before = boxes_of(api, ds)
    out = api.ok(
        "POST",
        f"/datasets/{ds}/assist",
        {
            "model_id": trained["id"],
            "sources": ["studio"],
            "conf": TRAINED_CONF,
            "overwrite": True,
        },
    )
    after_boxes = boxes_of(api, ds)
    check(
        "assist on Studio images with overwrite: every non-excluded Studio image processed, excluded untouched",
        out["processed"] == c.seed["expected_images"]
        and after_boxes["cam001.jpg"] == before["cam001.jpg"]
        and all(after_boxes.get(n) == before.get(n) for n in c.uploads),
        f"processed={out['processed']} written={out['items_written']}",
    )
    # Combination: assisted boxes flow into a new snapshot and a new export.
    snap2 = api.ok("POST", f"/datasets/{ds}/snapshots", {})
    st, data = api.call("GET", f"/datasets/{ds}/export/voc")
    check(
        "combined chain: export and snapshot after assist",
        st == 200
        and snap2["id"] != snap["id"]
        and "cam001" not in export_names("voc", data),
        f"{st}",
    )

    # FT-35: MegaDetector assist on a classification dataset; boxes take the image's label.
    # Assist replaces an item's annotations, so the expected label comes from before the call.
    cls_ds = c.cls_ds
    cls_before = boxes_of(api, cls_ds)
    st, b = api.call("POST", f"/datasets/{cls_ds}/assist", {"model_id": "megadetector"})
    cls_boxes = boxes_of(api, cls_ds)

    def _expected_box_label(name: str) -> str:
        labels = {lab for lab, _ in cls_before.get(name, [])}
        return next(iter(labels)) if len(labels) == 1 else "animal"

    mislabelled = [
        n
        for n, anns in cls_boxes.items()
        for label, bbox in anns
        if bbox is not None and label != _expected_box_label(n)
    ]
    check(
        "assist on a classification dataset: MegaDetector boxes inherit the image label",
        st == 200 and b.get("items_written", 0) > 0 and not mislabelled,
        f"{st}: written={b.get('items_written') if isinstance(b, dict) else b} mislabelled={mislabelled[:3]}",
    )

    # Error cases.
    for body, path_ds, code, what in (
        (
            {"model_id": "megadetector", "animal_label": "   "},
            ds,
            422,
            "blank animal label",
        ),
        (
            {"model_id": trained["id"], "animal_label": "Bos"},
            ds,
            422,
            "animal label with a trained model",
        ),
        (
            {"model_id": "megadetector", "images": ["nope.jpg"]},
            ds,
            422,
            "unknown image",
        ),
        ({"model_id": 999999}, ds, 404, "missing model"),
    ):
        st, b = api.call("POST", f"/datasets/{path_ds}/assist", body)
        check(f"assist refused: {what} -> {code}", st == code, f"{st}: {str(b)[:120]}")


def scenario_megadetector_real(c: Ctx) -> None:
    api = c.api
    models = api.ok("GET", "/assist/models")
    first = (models.get("models") or models)[0]
    check(
        "assist model list: MegaDetector first and installed",
        first.get("id") == "megadetector" and first.get("available") is True,
        json.dumps(first)[:200],
    )
    rows = list(csv.DictReader(open(AMAZON / "annotations/train_raw.csv")))
    picks = [
        r
        for r in rows
        if r["Label"] == "Bos" and (AMAZON / "resized" / r["Photo_ID"]).is_file()
    ][:6]
    ds = api.ok(
        "POST",
        f"/workspaces/{c.ws}/datasets",
        {"name": "amazon md", "task": "detection"},
    )["id"]
    for r in picks:
        api.ok(
            "PUT",
            f"/datasets/{ds}/images/{Path(r['Photo_ID']).name}",
            raw=(AMAZON / "resized" / r["Photo_ID"]).read_bytes(),
        )
    out = api.ok(
        "POST",
        f"/datasets/{ds}/assist",
        {"model_id": "megadetector", "animal_label": "Bos"},
    )
    labels = {b["label"] for w in out["items"] for b in w["boxes"]}
    check(
        "MegaDetector on real Amazon cattle images with animal label 'Bos'",
        out["items_written"] >= len(picks) - 1
        and "Bos" in labels
        and "animal" not in labels,
        f"written={out['items_written']}/{len(picks)} labels={labels} boxes={out['boxes']}",
    )
    out2 = api.ok(
        "POST",
        f"/datasets/{ds}/assist",
        {"model_id": "megadetector", "animal_label": "Bos"},
    )
    check(
        "MegaDetector re-run without overwrite skips labelled images",
        out2["processed"] == len(picks) - out["items_written"],
        f"processed={out2['processed']}",
    )
    c.amazon_ds = ds


# --------------------------------------------------------------------------- 5. classifiers
def scenario_classifiers(c: Ctx) -> None:
    api = c.api
    ds = api.ok(
        "POST",
        f"/workspaces/{c.ws}/datasets",
        {"name": "cls crops", "task": "classification"},
    )["id"]
    c.cls_ds = ds
    rng = random.Random(31)
    items = []
    for i in range(40):
        im, boxes = scene(rng, 1)
        lb, box = boxes[0]
        name = f"c{i:03d}.jpg"
        api.ok("PUT", f"/datasets/{ds}/images/{name}", raw=jpeg(im.crop(box)))
        items.append({"image": name, "annotations": [{"label": lb, "bbox": None}]})
    api.ok("POST", f"/datasets/{ds}/annotations", {"items": items})
    snap = api.ok("POST", f"/datasets/{ds}/snapshots", {})["id"]
    c.cls_snap = snap
    first = api.ok(
        "POST",
        "/jobs",
        {"pipeline": "dinov3_linear", "snapshot_id": snap, "params": {}},
    )["id"]
    conv = api.ok(
        "POST",
        "/jobs",
        {
            "pipeline": "conventional_classifier",
            "snapshot_id": snap,
            "params": {"epochs": 2, "warmup_epochs": 1, "batch": 8},
        },
    )["id"]
    job = wait_job(api, first, c.train_timeout)
    second = api.ok(
        "POST",
        "/jobs",
        {"pipeline": "dinov3_linear", "snapshot_id": snap, "params": {}},
    )["id"]
    job2 = wait_job(api, second, c.train_timeout)
    c.cls_models = {}
    for what, j, jid in (
        ("dinov3 first run", job, first),
        ("dinov3 second run", job2, second),
    ):
        models = api.ok("GET", f"/models?job_id={jid}")["models"]
        ok = j["state"] == "succeeded" and len(models) == 1
        check(
            f"{what} succeeds",
            ok,
            f"{j['state']} {str(j.get('error'))[:300] if not ok else ''}",
        )
        if ok:
            c.cls_models[what] = models[0]
    if len(c.cls_models) == 2:
        m1, m2 = (
            c.cls_models["dinov3 first run"]["metrics"],
            c.cls_models["dinov3 second run"]["metrics"],
        )
        check(
            "DINOv3 retrain on the same snapshot reuses cached embeddings",
            m1.get("embedding_cache_misses", 0) > 0
            and m2.get("embedding_cache_hits", 0) >= 40
            and m2.get("embedding_cache_misses", 1) == 0,
            f"first hits/misses={m1.get('embedding_cache_hits')}/{m1.get('embedding_cache_misses')} "
            f"second={m2.get('embedding_cache_hits')}/{m2.get('embedding_cache_misses')}",
        )
        man = bundle_manifest(api, c.cls_models["dinov3 second run"]["id"])
        check(
            "DINOv3 bundle carries a calibrated threshold and abstain label",
            "calibrated_threshold" in man and "abstain_label" in man,
            json.dumps(
                {k: man.get(k) for k in ("calibrated_threshold", "abstain_label")}
            ),
        )
    job = wait_job(api, conv, c.train_timeout)
    models = api.ok("GET", f"/models?job_id={conv}")["models"]
    ok = job["state"] == "succeeded" and len(models) == 1
    check(
        "conventional classifier (offline resnet50 weights) succeeds",
        ok,
        f"{job['state']} {str(job.get('error'))[:300] if not ok else ''}",
    )
    if ok:
        c.cls_models["conventional"] = models[0]
        man = bundle_manifest(api, models[0]["id"])
        check(
            "conventional classifier bundle carries a calibrated threshold",
            "calibrated_threshold" in man and "abstain_label" in man,
            json.dumps(
                {k: man.get(k) for k in ("calibrated_threshold", "abstain_label")}
            ),
        )
    st, body = api.call(
        "POST",
        "/jobs",
        {"pipeline": "yolo_detector", "snapshot_id": snap, "params": {}},
    )
    check("yolo_detector on a classification snapshot -> 422", st == 422, str(st))


# --------------------------------------------------------------------------- 6. lifecycle
def scenario_lifecycle(c: Ctx) -> None:
    api = c.api
    jid = api.ok(
        "POST",
        "/jobs",
        {
            "pipeline": "yolo_detector",
            "snapshot_id": c.combo_snap,
            "params": {"model": "yolo11n.pt", "epochs": 300, "imgsz": 320, "batch": 4},
        },
    )["id"]
    wait_state(api, jid, ("running",), 600)
    api.ok("POST", f"/jobs/{jid}/cancel")
    job = wait_state(api, jid, ("cancelled", "failed", "succeeded"), 600)
    check("cancel a running job", job["state"] == "cancelled", job["state"])
    st, _ = api.call("POST", f"/jobs/{jid}/cancel")
    check("cancel a finished job -> 409", st == 409, str(st))
    retry = api.ok("POST", f"/jobs/{jid}/retry")
    check(
        "retry a cancelled job creates a new queued job with the same params",
        retry["id"] != jid and retry["params"].get("epochs") == 300,
        json.dumps({k: retry.get(k) for k in ("id", "state")}),
    )
    api.call("POST", f"/jobs/{retry['id']}/cancel")
    wait_state(api, retry["id"], ("cancelled", "failed", "succeeded"), 600)
    ok_job = next(iter(c.det_models.values()), {}).get("job_id")
    st, _ = api.call("POST", f"/jobs/{ok_job}/retry")
    check("retry a succeeded job -> 409", ok_job is not None and st == 409, str(st))

    m = next(iter(c.det_models.values()), None)
    if m is None:
        check("model lifecycle", False, "no detector model")
        return
    api.ok("PATCH", f"/models/{m['id']}", {"name": "E2E detector"})
    check(
        "rename a model",
        api.ok("GET", f"/models/{m['id']}").get("name") == "E2E detector",
        "",
    )
    api.ok(
        "POST",
        f"/models/{m['id']}/activate",
        {"consumer": "local", "project": "e2e-site"},
    )
    deps = api.ok("GET", "/deployments")
    deps = deps.get("deployments", deps)
    check(
        "local activation listed in deployments",
        any(
            d.get("model_id") == m["id"] and d.get("project") == "e2e-site"
            for d in deps
        ),
        json.dumps(deps)[:200],
    )


# --------------------------------------------------------------------------- 8. browser
def scenario_ui(c: Ctx) -> None:
    from playwright.sync_api import expect, sync_playwright

    shots = c.out / "ui"
    shots.mkdir(exist_ok=True)
    errors: list[str] = []
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1400, "height": 1000})
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: m.type == "error" and errors.append(m.text))
        page.goto(c.base)
        page.fill("#login-username", c.ui_user)
        page.fill("#login-password", c.ui_password)
        page.click("#login-submit")
        expect(page.locator("#dataset-select")).to_be_visible(timeout=20000)
        page.select_option("#dataset-select", str(c.combo))
        page.wait_for_timeout(1500)

        fmt = page.locator("#annotation-format")
        check(
            "UI: label format defaults to Camera Trap DP with VOC and COCO options",
            fmt.input_value() == "camtrapdp"
            and {"coco", "voc"}
            <= set(fmt.locator("option").evaluate_all("o => o.map(x => x.value)")),
            "",
        )
        check(
            "UI: Export button labelled 'Export'",
            page.locator("#export-annotations").inner_text().strip() == "Export",
            "",
        )
        check(
            "UI: Camera Trap DP import warning shown",
            page.locator("#annotation-format-warning").is_visible(),
            "",
        )

        check(
            "UI: assist body hidden until the box is ticked",
            not page.locator("#assist-body").is_visible(),
            "",
        )
        page.check("#assist-enable")
        expect(page.locator("#assist-body")).to_be_visible()
        check(
            "UI: assist defaults to MegaDetector",
            page.locator("#assist-model").input_value() == "megadetector",
            "",
        )
        # The source labels are filled in only once the summary has loaded.
        expect(page.locator("#assist-source-all-text")).to_contain_text("from Studio")
        check(
            "UI: source choice offers Studio images on a dataset that has them",
            page.locator("#assist-source-all-row").is_visible(),
            "",
        )
        # Earlier scenarios may have pre-drawn boxes on the uploads, so the expected warnings follow the
        # dataset's state rather than assuming fresh uploads.
        summ = c.api.ok("GET", f"/datasets/{c.combo}/assist/summary")["sources"]
        uploads_boxed = summ["upload"]["images"] - summ["upload"]["empty"]
        warns = page.locator("#assist-warnings p.assist-warning").count()
        check(
            "UI: uploads only: Studio images noted as left out; conflict warning iff uploads have boxes",
            page.locator("#assist-warnings p.assist-note").count() == 1
            and (warns >= 1 if uploads_boxed else warns == 0),
            f"uploads with boxes={uploads_boxed}; "
            + page.locator("#assist-warnings").inner_text()[:200],
        )
        page.check("input[name=assist-source][value=all]")
        check(
            "UI: Studio source chosen: conflict warnings for confirmed and Studio images",
            page.locator("#assist-warnings p.assist-warning").count() >= 1,
            page.locator("#assist-warnings").inner_text()[:200],
        )
        if summ["upload"]["empty"] + summ["studio"]["empty"] == 0:
            # Every image already has boxes: the app refuses unless replacing is ticked.
            page.check("#assist-overwrite")
        page.click("#assist-btn")
        expect(page.locator("#assist-confirm")).to_be_visible()
        check(
            "UI: assist asks for confirmation before running",
            page.locator("#assist-confirm-list li").count() >= 1,
            page.locator("#assist-confirm").inner_text()[:200],
        )
        page.screenshot(path=str(shots / "01-data-assist.png"), full_page=True)
        page.click("#assist-cancel-btn")

        check(
            "UI: Studio box shows the hint before a project is chosen",
            page.locator("#studio-hint").is_visible(),
            "",
        )
        page.select_option("#studio-project", str(c.seed["project"]))
        expect(page.locator("#studio-body")).to_be_visible()
        check(
            "UI: choosing a Studio project unfolds the import section",
            page.locator("#studio-import-btn").is_visible(),
            "",
        )

        page.select_option("#pipeline-select", "yolo_detector")
        expect(page.locator("#start-weights-field")).to_be_visible()
        opts = page.locator("#start-weights option").evaluate_all(
            "o => o.map(x => x.value)"
        )
        check(
            "UI: YOLO starting weights offer YOLO11n, MDV6 compact and extra",
            opts == ["yolo11n.pt", "mdv6-yolov10-c", "mdv6-yolov10-e"],
            str(opts),
        )
        page.select_option("#start-weights", "mdv6-yolov10-e")
        check(
            "UI: starting-weights help text updates",
            "extra" in page.locator("#start-weights-help").inner_text().lower()
            or "download" in page.locator("#start-weights-help").inner_text().lower(),
            page.locator("#start-weights-help").inner_text()[:120],
        )
        check(
            "UI: pipeline guide explains the pipeline",
            len(page.locator("#pipeline-guide").inner_text()) > 80,
            "",
        )
        check(
            "UI: batch automatic option present",
            page.locator("#batch-auto").is_visible(),
            "",
        )

        page.locator("button:has-text('Edit boxes')").first.click()
        expect(page.locator("#box-editor")).to_be_visible()
        labels = page.locator("#box-label option").all_inner_texts()
        check(
            "UI: box editor label list has existing labels and '+ New label…'",
            any("New label" in t for t in labels) and "lion" in labels,
            str(labels),
        )
        page.screenshot(path=str(shots / "02-box-editor.png"))
        page.click("#box-cancel")

        page.click("button[data-page=jobs]")
        page.locator("[data-job-id]").first.click()
        page.wait_for_timeout(1500)
        style = page.locator("#event-list").evaluate(
            "e => [getComputedStyle(e).overflowY, e.clientHeight, e.scrollHeight]"
        )
        check(
            "UI: job events list has a fixed height and scrolls",
            style[0] in ("auto", "scroll") and style[1] < 600,
            str(style),
        )
        page.screenshot(path=str(shots / "03-jobs.png"))

        page.click("button[data-page=models]")
        det = next(iter(c.det_models.values()), None)
        if det:
            page.locator(f"[data-model-id='{det['id']}']").click()
            expect(page.locator("#per-class-section")).to_be_visible()
            check(
                "UI: detector model shows the per-class table",
                page.locator("#per-class-section tr").count() >= 3,
                "",
            )
        cls = c.cls_models.get("dinov3 second run")
        if cls:
            page.locator(f"[data-model-id='{cls['id']}']").click()
            expect(page.locator("#threshold-info")).to_be_visible()
            check(
                "UI: classifier model shows the calibrated threshold",
                True,
                page.locator("#threshold-info").inner_text()[:120],
            )
        page.screenshot(path=str(shots / "04-models.png"), full_page=True)
        b.close()
    check("UI: no page errors or console errors", not errors, "; ".join(errors)[:300])


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--studio-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--assets", type=Path, default=Path("/tmp/ft_e2e_full_assets"))
    ap.add_argument("--train-timeout", type=int, default=2400)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    studio_root, out, assets = (
        args.studio_root.resolve(),
        args.out.resolve(),
        args.assets.resolve(),
    )
    for f in ("dinov3_vits16.onnx", "dinov3_vits16.onnx.data", "yolo11n.pt", MD_C_PT):
        if not (assets / f).is_file():
            raise SystemExit(f"missing asset {assets / f}")
    resnet = next(TIMM_RESNET50.glob("snapshots/*/model.safetensors"), None)
    if resnet is None or not (MD_PACKAGE / "1/model.onnx").is_file():
        raise SystemExit("missing resnet50 safetensors or the MDV6-yolov10-c package")
    work = out / "work"
    for d in (
        "gallery",
        "studio_models",
        "storage",
        "staging",
        "yolo_cwd",
        "weights/timm/resnet50",
    ):
        (work / d).mkdir(parents=True, exist_ok=True)
    shutil.copy2(assets / "yolo11n.pt", work / "yolo_cwd/yolo11n.pt")
    os.chmod(work / "yolo_cwd", 0o777)
    os.chmod(work / "yolo_cwd/yolo11n.pt", 0o644)
    weights = work / "weights"
    shutil.copytree(MD_PACKAGE, weights / MD_PACKAGE.name, dirs_exist_ok=True)
    shutil.copy2(assets / MD_C_PT, weights / MD_C_PT)
    shutil.copy2(resnet, weights / "timm/resnet50/model.safetensors")
    for p in [weights, *weights.rglob("*")]:
        os.chmod(p, 0o755 if p.is_dir() else 0o644)

    token, api_token = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    pg_password = secrets.token_urlsafe(16)
    project = f"ft-e2e-matrix-{secrets.token_hex(3)}"
    env_file, override = work / "service.env", work / "compose.e2e.yml"
    files = [REPO / "compose.service.yml", override]
    ports = {k: free_port() for k in ("pg", "studio", "ft")}
    info: dict = {
        "studio_commit": run(
            ["git", "-C", str(studio_root), "rev-parse", "HEAD"]
        ).stdout.strip(),
        "finetune_commit": run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"]
        ).stdout.strip(),
    }
    env_file.write_text(
        "\n".join(
            [
                f"POSTGRES_PASSWORD={secrets.token_urlsafe(16)}",
                f"SPARROW_FT_PORT={ports['ft']}",
                "SPARROW_FT_STUDIO_URL=http://studio:5000",
                f"SPARROW_FT_STUDIO_TOKEN={token}",
                "SPARROW_FT_AUTH_DISABLED=0",
                f"SPARROW_FT_API_TOKENS={api_token}",
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
    doc = json.loads(override.read_text())
    mounts = ["ft-data:/data", f"{assets}:/assets:ro", f"{weights}:/weights:ro"]
    for name in ("worker-cpu", "worker-gpu"):
        doc["services"][name] = {
            "volumes": [*mounts, f"{work / 'yolo_cwd'}:/yolo"],
            "working_dir": "/yolo",
        }
    doc["services"]["api"] = {"volumes": mounts}
    override.write_text(json.dumps(doc, indent=2))
    os.chmod(override, 0o600)
    dsn = f"host=127.0.0.1 port={ports['pg']} user=postgres password={pg_password} dbname=postgres"

    c = Ctx()
    c.out, c.train_timeout = out, args.train_timeout
    c.base = f"http://127.0.0.1:{ports['ft']}"
    c.ui_user, c.ui_password = "matrix", secrets.token_urlsafe(12)
    c.det_models, c.cls_models = {}, {}
    t0 = time.time()
    try:
        r = compose(
            project, files, env_file, "up", "-d", "--wait", "studio-db", check_rc=False
        )
        if r.returncode:
            raise RuntimeError(f"studio-db did not start: {r.stderr[-800:]}")
        c.seed = seed_studio(
            dsn, work / "gallery", studio_root / "db/schema/01_schema.sql", 24
        )
        r = compose(
            project, files, env_file, "up", "-d", "--wait", "--no-build", check_rc=False
        )
        (out / "compose_up.log").write_text(r.stdout + r.stderr)
        check(
            "compose up: Studio + fine-tune stack (GPU profile) healthy",
            r.returncode == 0,
            project,
        )
        if r.returncode:
            raise RuntimeError("compose up failed")
        compose(
            project,
            files,
            env_file,
            "exec",
            "-T",
            "-e",
            f"SPARROW_FT_PASSWORD={c.ui_password}",
            "api",
            "sparrow-finetune",
            "create-user",
            c.ui_user,
        )
        c.api = Api(c.base, api_token)
        c.ws = c.api.ok("POST", "/workspaces", {"name": "Feature matrix"})["id"]
        for name, fn in (
            ("formats", scenario_formats),
            ("MegaDetector on real images", scenario_megadetector_real),
            ("classifiers", scenario_classifiers),
            ("detection chain", scenario_detection_chain),
            ("job and model lifecycle", scenario_lifecycle),
            ("browser", scenario_ui),
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
    info["duration_s"] = round(time.time() - t0)
    info["checks"] = [{"name": n, "pass": ok, "detail": d} for n, ok, d in CHECKS]
    (out / "result.json").write_text(json.dumps(info, indent=2, default=str))
    failed = [n for n, ok, _ in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
