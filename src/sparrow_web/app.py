"""Flask application factory.

Server-rendered pages with HTMX for partial updates, plus a small JSON surface
for the things charts need and one multiplexed SSE stream.

Run with exactly one worker and threads::

    gunicorn -k gthread -w 1 --threads 16 --timeout 0 \\
             -b 0.0.0.0:8080 "sparrow_web.app:create_app()"

More than one worker would give you two schedulers competing for the same GPU
lease and SSE clients receiving a random subset of events. ``--timeout 0``
because a worker holding an SSE generator open for hours must not be reaped.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict

from flask import (Flask, Response, abort, jsonify, redirect, render_template,
                   request, send_file, url_for)

from sparrow_yolo.hardware import query_gpus
from sparrow_yolo.process_runner import resolve_batch, unique_run_dir

from . import annotation, removal, training
from .annotation import Batches
from .db import Database, dumps, loads, utcnow
from .events import EventBus, sse_stream
from .ingest import RunIngestor, tail_bytes
from .jobs import JobQueue
from .repo import Models, Projects, Runs, Snapshots
from .settings import load_settings
from .steps import compute_steps, current_step
from .supervisor import Supervisor


def create_app(settings=None, start_supervisor: bool = True) -> Flask:
    settings = settings or load_settings()
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["SETTINGS"] = settings

    db = Database(settings.db_path)
    db.migrate()
    bus = EventBus(max_clients=settings.max_sse_clients)
    supervisor = Supervisor(db, settings, bus)

    app.config.update(DB=db, BUS=bus, SUPERVISOR=supervisor)
    projects, snapshots, runs, models = Projects(db), Snapshots(db), Runs(db), Models(db)
    jobs, ingestor = JobQueue(db, settings), RunIngestor(db)

    if start_supervisor:
        supervisor.start()

    # -- helpers ----------------------------------------------------------
    def get_project_or_404(slug: str) -> Dict[str, Any]:
        project = projects.by_slug(slug)
        if project is None:
            abort(404, f"No project named {slug!r}")
        return project

    def project_context(project: Dict[str, Any]) -> Dict[str, Any]:
        steps = compute_steps(db, project)
        snapshot = snapshots.latest(project["id"])
        return {
            "project": project,
            "config": loads(project["config_json"], {}),
            "steps": steps,
            "current": current_step(steps),
            "snapshot": snapshot,
            "latest_model": models.latest(project["id"]),
            "runs": runs.list(project["id"], limit=10),
            "jobs": jobs.list(project["id"], limit=10),
        }

    @app.template_filter("thousands")
    def thousands(value: Any) -> str:
        try:
            return f"{int(value):,}"
        except (TypeError, ValueError):
            return "-"

    @app.template_filter("filesize")
    def filesize(value: Any) -> str:
        try:
            n = float(value)
        except (TypeError, ValueError):
            return "-"
        for unit in ("B", "KB", "MB", "GB"):
            if n < 1024 or unit == "GB":
                return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.1f} {unit}"
            n /= 1024
        return f"{n:,.1f} GB"

    @app.template_filter("pct")
    def pct(value: Any, digits: int = 1) -> str:
        try:
            return f"{float(value) * 100:.{digits}f}%"
        except (TypeError, ValueError):
            return "-"

    @app.template_filter("metric")
    def metric(value: Any, digits: int = 3) -> str:
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return "-"

    # -- pages ------------------------------------------------------------
    @app.get("/")
    def index():
        return render_template(
            "index.html",
            projects=[{**p, "steps": compute_steps(db, p),
                       "model": models.latest(p["id"]),
                       "snapshot": snapshots.latest(p["id"])}
                      for p in projects.list()],
            gpus=[g.as_dict() for g in query_gpus()],
        )

    def discover_candidates(max_depth: int = 5, budget_s: float = 6.0) -> Dict[str, Any]:
        """Find plausible dataset folders under the allowed roots.

        Bounded in three directions — depth, entries per directory, and wall
        clock. The library root can be a whole Windows drive reached over the
        WSL filesystem bridge, where an unbounded walk would take minutes and
        hang the page. ``find_images()`` deliberately is not used here: it walks
        the entire tree, which is right for an audit and wrong for a form.
        """
        found: list = []
        deadline = time.monotonic() + budget_s
        truncated = False

        def scan(directory: Path, depth: int) -> None:
            nonlocal truncated
            if depth > max_depth or len(found) >= 40:
                return
            if time.monotonic() > deadline:
                truncated = True
                return
            try:
                with os.scandir(directory) as it:
                    entries = []
                    for i, entry in enumerate(it):
                        if i >= 400:
                            break
                        entries.append(entry)
            except OSError:
                return

            images = xmls = 0
            subdirs = []
            for entry in entries:
                try:
                    if entry.is_dir():
                        if not entry.name.startswith("."):
                            subdirs.append(Path(entry.path))
                        continue
                    suffix = Path(entry.name).suffix.lower()
                    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
                        images += 1
                    elif suffix == ".xml":
                        xmls += 1
                except OSError:
                    continue

            if images:
                found.append({"path": str(directory), "images": images, "xmls": xmls,
                              "truncated": len(entries) >= 400})
                return   # a folder holding images is itself the dataset; stop here
            for sub in sorted(subdirs):
                scan(sub, depth + 1)

        for root in settings.dataset_search_roots:
            if root.is_dir():
                scan(root, 0)
        return {"candidates": found, "truncated": truncated,
                "roots": [str(r) for r in settings.dataset_search_roots]}

    @app.get("/projects/new")
    def new_project_form():
        # No eager discovery scan here: the browsable roots can include whole
        # drives, and crawling them on every page load would cost seconds for a
        # result the picker does not use. The picker lists one folder at a time.
        return render_template("project_new.html", roots=settings.dataset_search_roots)

    @app.get("/api/discover")
    def api_discover():
        return jsonify(discover_candidates())


    IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    def quick_count(directory: Path, cap: int = 400) -> Dict[str, Any]:
        """Cheap look inside a folder: how many images, annotations and subfolders.

        Stops after `cap` entries. The library root can be a whole Windows drive
        reached over the WSL filesystem bridge, where counting everything would
        make the picker feel broken.
        """
        images = xmls = subdirs = 0
        seen = 0
        try:
            with os.scandir(directory) as it:
                for entry in it:
                    seen += 1
                    if seen > cap:
                        break
                    try:
                        if entry.is_dir():
                            if not entry.name.startswith("."):
                                subdirs += 1
                            continue
                        suffix = Path(entry.name).suffix.lower()
                        if suffix in IMAGE_SUFFIXES:
                            images += 1
                        elif suffix == ".xml":
                            xmls += 1
                    except OSError:
                        continue
        except OSError:
            return {"images": 0, "xmls": 0, "subdirs": 0, "partial": False, "readable": False}
        return {"images": images, "xmls": xmls, "subdirs": subdirs,
                "partial": seen > cap, "readable": True}

    @app.get("/api/browse")
    def api_browse():
        """Directory listing for the folder picker.

        Every path is checked against the allowlist, including the one the client
        asks for: without that check a crafted `?path=/` would let anyone reachable
        walk the whole filesystem, and this process runs as root.
        """
        # Two allowlists, never mixed: browsing for a dataset must not offer the
        # writable roots, and choosing an export destination must not offer the
        # read-only ones — a path picked in one mode would fail in the other.
        writable = request.args.get("writable") == "1"
        roots = settings.export_roots if writable else settings.dataset_search_roots
        allowed = (settings.is_allowed_export_root if writable
                   else settings.is_allowed_dataset_root)
        if writable and not roots:
            return jsonify({"error": "No writable folders are configured. Set "
                                     "SPARROW_EXPORT_ROOTS and mount them."}), 409

        raw = request.args.get("path") or (str(roots[0]) if roots else "/")
        target = Path(raw)

        if not allowed(target):
            return jsonify({"error": "That folder is outside the allowed locations."}), 403
        if not target.is_dir():
            return jsonify({"error": f"{target} is not a folder."}), 404

        target = target.resolve()
        entries = []
        try:
            with os.scandir(target) as it:
                for entry in it:
                    if not entry.is_dir() or entry.name.startswith("."):
                        continue
                    counts = quick_count(Path(entry.path))
                    entries.append({
                        "name": entry.name,
                        "path": entry.path,
                        **counts,
                        # A folder holding images is selectable as a dataset; one
                        # holding only folders is somewhere to navigate through.
                        "is_dataset": counts["images"] > 0,
                    })
        except OSError as e:
            return jsonify({"error": str(e)}), 400
        entries.sort(key=lambda e: (not e["is_dataset"], e["name"].lower()))

        # Breadcrumbs stop at the allowed root - there is nothing above it the
        # user is permitted to see.
        root = next((r.resolve() for r in roots
                     if target == r.resolve() or target.is_relative_to(r.resolve())), None)
        crumbs, parent = [], None
        if root is not None:
            crumbs.append({"name": root.name or str(root), "path": str(root)})
            if target != root:
                for part in target.relative_to(root).parts:
                    crumbs.append({"name": part,
                                   "path": str(Path(crumbs[-1]["path"]) / part)})
                parent = str(target.parent)

        return jsonify({
            "path": str(target),
            "parent": parent,
            "breadcrumbs": crumbs,
            "entries": entries,
            "here": quick_count(target),
        })

    @app.post("/projects")
    def create_project():
        def form_error(message: str, status: int = 400):
            return render_template("project_new.html", error=message,
                                   roots=settings.dataset_search_roots), status

        name = (request.form.get("name") or "").strip()
        root = (request.form.get("dataset_root") or "").strip()
        if not name or not root:
            return form_error("A name and a dataset folder are both required.")
        if not settings.is_allowed_dataset_root(root):
            allowed = ", ".join(str(r) for r in settings.dataset_search_roots)
            return form_error(
                f"That folder is outside the allowed locations ({allowed}). "
                f"Set DATASETS_HOST_ROOT in .env to mount a different library.")
        if not Path(root).is_dir():
            return form_error(f"{root} does not exist or is not a folder.")
        try:
            project_id = projects.create(name, root)
        except ValueError as e:
            return form_error(str(e))
        return redirect(url_for("project", slug=projects.get(project_id)["slug"]))

    @app.get("/p/<slug>")
    def project(slug: str):
        return render_template("project.html", **project_context(get_project_or_404(slug)))

    @app.get("/p/<slug>/delete")
    def confirm_delete(slug: str):
        project = get_project_or_404(slug)
        return render_template("project_delete.html", project=project,
                               plan=removal.plan(db, settings, project))

    @app.post("/p/<slug>/delete")
    def delete_project(slug: str):
        """Delete the project. The dataset folder is never touched — see
        ``removal._removable``, which is what actually enforces that."""
        project = get_project_or_404(slug)
        # Typing the name is the guard here: this is irreversible, and the
        # projects list puts the control one click from a project you meant to open.
        if request.form.get("confirm", "").strip() != project["name"]:
            return render_template("project_delete.html", project=project,
                                   plan=removal.plan(db, settings, project),
                                   error="Type the project name exactly to confirm."), 400
        try:
            removal.delete(db, settings, project,
                           remove_files=request.form.get("remove_files") == "on")
        except RuntimeError as e:
            return render_template("project_delete.html", project=project,
                                   plan=removal.plan(db, settings, project),
                                   error=str(e)), 409
        bus.publish("projects", "project.deleted", {"id": project["id"], "slug": slug})
        return redirect(url_for("index"))

    @app.get("/p/<slug>/review")
    def review(slug: str):
        project = get_project_or_404(slug)
        snapshot = snapshots.latest(project["id"])
        return render_template(
            "review.html",
            **project_context(project),
            classes=snapshots.classes(snapshot["id"]) if snapshot else [],
            issue_counts=snapshots.issue_counts(snapshot["id"]) if snapshot else {},
            summary=loads(snapshot["summary_json"], {}) if snapshot else {},
        )


    # ------------------------------------------------------------ train --
    def latest_prepared(project_id: int):
        return db.one("SELECT * FROM prepared_datasets WHERE project_id=? AND state='ready'"
                      " ORDER BY id DESC LIMIT 1", (project_id,))

    def train_context(project, form=None, errors=None):
        """Everything the training form needs, with values carried over from the
        last run so a second attempt starts from what you last chose rather than
        from the defaults."""
        prepared = latest_prepared(project["id"])
        previous = db.one(
            "SELECT * FROM runs WHERE project_id=? AND config_json != '{}'"
            " ORDER BY id DESC LIMIT 1", (project["id"],))

        values = training.defaults()
        values["model"] = training.DEFAULT_MODEL
        values["batch"] = "auto"
        values["device"] = str(settings.train_gpu)
        if previous and not request.args.get("defaults"):
            values.update({k: v for k, v in loads(previous["config_json"], {}).items()
                           if k in values})
        if form:
            values.update({k: v for k, v in form.items() if k in values})

        return {
            **project_context(project),
            "prepared": prepared,
            "models": models.list(project["id"]),
            "groups": training.groups(),
            "base_models": training.BASE_MODELS,
            "values": values,
            "errors": errors or {},
            "gpus": [g.as_dict() for g in query_gpus()],
            "train_gpu": settings.train_gpu,
            "previous_run": previous,
        }

    @app.get("/p/<slug>/train")
    def train_page(slug: str):
        project = get_project_or_404(slug)
        return render_template("train.html", **train_context(project))

    @app.get("/api/projects/<int:project_id>/batch")
    def batch_preview(project_id: int):
        """What 'auto' would actually resolve to, for the form to show live.

        Worth its own endpoint: 'Auto' on its own tells you nothing, and the
        number depends on the model, the image size and how much memory the
        display is using right now.
        """
        try:
            # Through the same parser the submission uses, so the preview cannot
            # promise something the launch would reject.
            requested = training.parse_batch(request.args.get("batch", "auto"))
            value = resolve_batch(
                requested,
                model_weights=request.args.get("model", training.DEFAULT_MODEL),
                imgsz=int(request.args.get("imgsz", 1024)),
                device=request.args.get("device", "0"),
            )
        except Exception as e:  # noqa: BLE001 - a bad guess must not break the form
            return jsonify({"error": str(e)}), 200
        if value == -1:
            label = "a size measured on the GPU when the run starts"
        elif isinstance(value, float):
            label = f"whatever fills {value:.0%} of the GPU's memory"
        else:
            label = f"{value} images per step"
        return jsonify({"batch": value, "label": label})

    @app.post("/p/<slug>/train")
    def start_training(slug: str):
        project = get_project_or_404(slug)

        running = db.one(
            "SELECT * FROM runs WHERE project_id=? AND state IN"
            " ('queued','starting','running','stopping') ORDER BY id DESC LIMIT 1",
            (project["id"],))
        if running:
            return redirect(url_for("run_detail", slug=slug, run_id=running["id"]))

        prepared = latest_prepared(project["id"])
        if prepared is None:
            return redirect(url_for("dataset_page", slug=slug))

        form = request.form.to_dict()
        try:
            values = training.parse(form)
        except training.ValidationError as e:
            return render_template(
                "train.html", **train_context(project, form=form, errors=e.errors)), 400

        # Start from COCO, or fine-tune from one of this project's own models.
        # Two structurally similar options with completely different meanings, so
        # they are one explicit choice rather than an inferred default.
        base_kind, base_model_id, weights = "coco", None, values["model"]
        chosen = form.get("start_from", "coco")
        if chosen.startswith("model:"):
            model = models.get(int(chosen.split(":", 1)[1]))
            if model is None or model["project_id"] != project["id"]:
                return render_template("train.html", **train_context(
                    project, form=form,
                    errors={"start_from": "that model is not in this project"})), 400
            base_kind, base_model_id, weights = "model_version", model["id"], model["weights_path"]

        run_dir = unique_run_dir(settings.runs_root, slug)
        now = utcnow()
        with db.write() as conn:
            cur = conn.execute(
                "INSERT INTO runs(project_id,prepared_id,name,run_dir,kind,base_kind,"
                "base_model_version_id,base_weights,config_json,device,imgsz,batch,"
                "state,epochs,queued_at,updated_at)"
                " VALUES (?,?,?,?,'train',?,?,?,?,?,?,?,'queued',?,?,?)",
                (project["id"], prepared["id"], run_dir.name, str(run_dir), base_kind,
                 base_model_id, weights,
                 # The full record of what was chosen, exactly as submitted and
                 # validated. This is what makes a run reproducible and what a
                 # comparison between two versions is read from.
                 dumps(values), str(values["device"]), int(values["imgsz"]),
                 str(values["batch"]), int(values["epochs"]), now, now))
            run_id = int(cur.lastrowid)

        params = training.split_params(values)
        params.update({"model_weights": weights, "data_yaml": prepared["data_yaml"]})
        jobs.create("train", project_id=project["id"], run_id=run_id, priority=10,
                    # The GPU lease is the queue: one training run per card, and
                    # evaluation on the other card never waits behind it.
                    resources=[f"gpu:{values['device']}", f"project:{project['id']}"],
                    params=params)
        bus.publish(f"project:{project['id']}", "run.queued", {"run_id": run_id})
        return redirect(url_for("run_detail", slug=slug, run_id=run_id))

    @app.get("/p/<slug>/runs/<int:run_id>")
    def run_detail(slug: str, run_id: int):
        project = get_project_or_404(slug)
        run = runs.get(run_id)
        if run is None or run["project_id"] != project["id"]:
            abort(404)
        chosen = loads(run["config_json"], {})
        resolved = loads(run["train_args_json"], {})
        # Show the label a human picked the setting by, not the Ultralytics key.
        rows = [(training.BY_NAME[k].label if k in training.BY_NAME else k, v)
                for k, v in chosen.items() if k not in ("model", "device")]
        note = ""
        if resolved.get("batch") not in (None, "") and str(chosen.get("batch")) == "auto":
            note = (f"Batch resolved to {resolved['batch']}"
                    + (" — measured on the GPU at start-up."
                       if resolved["batch"] == -1 else "."))
        return render_template(
            "run.html", **project_context(project), run=run,
            series=ingestor.series(run_id),
            eta=ingestor.eta(run_id, run.get("epochs")),
            chosen=rows, resolved=resolved, resolved_note=note,
        )

    @app.get("/system")
    def system():
        from sparrow_yolo.hardware import is_wsl, link_support, shm_bytes
        return render_template(
            "system.html",
            gpus=[g.as_dict() for g in query_gpus()],
            is_wsl=is_wsl(),
            shm_gb=round((shm_bytes() or 0) / 1024 ** 3, 1),
            links=link_support(settings.prepared_root),
            settings=settings,
        )

    # -- actions ----------------------------------------------------------
    def unique_prepared_name(project_id: int, slug: str, snapshot_id: int) -> str:
        """``<snapshot_id>``, or ``<snapshot_id>-2``, ``-3``... if taken.

        Named after the snapshot because that is what a person recognises, but
        made unique because preparing the same snapshot twice is normal — a
        different split, or a rebuild after a fix — and the second must not
        delete the tree the first model was evaluated against.
        """
        root = Path(settings.prepared_root) / slug
        taken = {Path(r["dir_path"]).name for r in db.query(
            "SELECT dir_path FROM prepared_datasets WHERE project_id=?", (project_id,))}
        base, name, n = str(snapshot_id), str(snapshot_id), 2
        while name in taken or (root / name).exists():
            name, n = f"{base}-{n}", n + 1
        return name

    def active_job(project_id: int, kind: str):
        return db.one(
            "SELECT * FROM jobs WHERE project_id=? AND kind=? AND state IN"
            " ('queued','leased','running') ORDER BY id DESC LIMIT 1", (project_id, kind))

    @app.post("/p/<slug>/scan")
    def start_scan(slug: str):
        """Queue a scan. Answers JSON to fetch() and redirects a plain form post."""
        project = get_project_or_404(slug)
        wants_json = request.headers.get("Accept", "").startswith("application/json")

        # Guard against a double click: the button is disabled client-side too,
        # but a refresh or a second tab would otherwise queue a duplicate scan of
        # 60,000 images.
        running = active_job(project["id"], "audit")
        if running:
            if wants_json:
                return jsonify({"job_id": running["id"], "already_running": True}), 200
            return redirect(url_for("review", slug=slug))

        cfg = loads(project["config_json"], {})
        dataset = cfg.get("dataset", {})
        job_id = jobs.create(
            "audit", project_id=project["id"],
            resources=[f"project:{project['id']}"],
            params={
                "dataset_root": project["dataset_root"],
                "artifact_dir": str(settings.project_dir(slug) / "snapshots" / "pending"),
                "normalise_class_names": dataset.get("normalise_class_names", True),
                "deep_image_check": dataset.get("deep_image_check", False),
                "confirmed_blank_dirs": dataset.get("confirmed_blank_dirs", []),
                "merge_classes": dataset.get("merge_classes", {}),
            },
        )
        if wants_json:
            return jsonify({"job_id": job_id, "already_running": False}), 202
        return redirect(url_for("review", slug=slug))

    @app.get("/p/<slug>/dataset")
    def dataset_page(slug: str):
        project = get_project_or_404(slug)
        ctx = project_context(project)
        prepared = db.query(
            "SELECT * FROM prepared_datasets WHERE project_id=? ORDER BY id DESC LIMIT 10",
            (project["id"],))
        return render_template("dataset.html", **ctx, prepared=prepared,
                               latest_prepared=prepared[0] if prepared else None)

    @app.post("/p/<slug>/prepare")
    def start_prepare(slug: str):
        """Queue a prepare. Writes a fresh, versioned tree — never overwrites the
        one an existing model was trained and tested on."""
        project = get_project_or_404(slug)
        wants_json = request.headers.get("Accept", "").startswith("application/json")

        running = active_job(project["id"], "prepare")
        if running:
            if wants_json:
                return jsonify({"job_id": running["id"], "already_running": True}), 200
            return redirect(url_for("dataset_page", slug=slug))

        snapshot = snapshots.latest(project["id"])
        if snapshot is None:
            if wants_json:
                return jsonify({"error": "Scan the dataset first."}), 409
            return redirect(url_for("review", slug=slug))

        cfg = loads(project["config_json"], {})
        dataset_cfg, split_cfg, prep_cfg = (cfg.get("dataset", {}), cfg.get("split", {}),
                                            cfg.get("prepare", {}))

        def field(name, default, cast=float):
            raw = request.form.get(name)
            if raw in (None, ""):
                return default
            try:
                return cast(raw)
            except (TypeError, ValueError):
                return default

        job_id = jobs.create(
            "prepare", project_id=project["id"],
            # Serialised against other work on this project, and against the
            # prepared tree, because prepare deletes and rewrites a directory a
            # running job could be reading.
            resources=[f"project:{project['id']}", f"prepared:{project['id']}"],
            snapshot_id=snapshot["id"],
            params={
                "artifact_dir": snapshot["artifact_dir"],
                "pinned_dir": str(Path(settings.prepared_root) / slug / "pinned"),
                "out_root": str(Path(settings.prepared_root) / slug),
                "prepared_name": unique_prepared_name(
                    project["id"], slug, snapshot["id"]),
                "train": field("train", split_cfg.get("train", 0.8)),
                "val": field("val", split_cfg.get("val", 0.1)),
                "test": field("test", split_cfg.get("test", 0.1)),
                "seed": field("seed", split_cfg.get("seed", 42), int),
                "stratify": request.form.get("stratify") or split_cfg.get("stratify", "rarest"),
                "group_by": request.form.get("group_by") or split_cfg.get("group_by", "none"),
                "background_policy": (request.form.get("background_policy")
                                      or dataset_cfg.get("background_policy", "reviewed")),
                "drop_classes": dataset_cfg.get("drop_classes", []),
                "min_boxes_per_class": field(
                    "min_boxes_per_class", dataset_cfg.get("min_boxes_per_class", 0), int),
                "reuse_class_map": prep_cfg.get("reuse_class_map", True),
                "reuse_split": prep_cfg.get("reuse_split", True),
                "link_mode": prep_cfg.get("link_mode", "auto"),
            },
        )
        if wants_json:
            return jsonify({"job_id": job_id, "already_running": False}), 202
        return redirect(url_for("dataset_page", slug=slug))

    # -- JSON -------------------------------------------------------------

    # --------------------------------------------------------- annotate --
    batches = Batches(db)

    def get_batch_or_404(project, batch_id: int):
        batch = batches.get(batch_id)
        if batch is None or batch["project_id"] != project["id"]:
            abort(404)
        return batch

    def selectable_models():
        """Every registered model, newest first, labelled by project.

        Not restricted to this project: a detector trained elsewhere is often the
        best way to bootstrap a new one, and the class names come from the model
        rather than from the project, so nothing is assumed about the match.
        """
        return db.query(
            "SELECT m.*, p.name AS project_name, p.slug AS project_slug"
            " FROM model_versions m JOIN projects p ON p.id = m.project_id"
            " WHERE m.status != 'archived' ORDER BY m.id DESC")

    @app.get("/p/<slug>/annotate")
    def annotate(slug: str):
        project = get_project_or_404(slug)
        return render_template("annotate.html", **project_context(project),
                               batches=batches.list(project["id"]),
                               models=selectable_models(),
                               export_configured=bool(settings.export_roots))

    @app.post("/p/<slug>/annotate")
    def create_batch(slug: str):
        """Take in a folder of images. Copying is synchronous on purpose: the
        picker only offers folders it has already counted, so the size is known
        before you press the button, and a batch that half-exists is worse than
        a page that took a moment."""
        project = get_project_or_404(slug)
        source = (request.form.get("source_path") or "").strip()
        name = (request.form.get("name") or "").strip() or Path(source).name

        def fail(message: str):
            return render_template(
                "annotate.html", **project_context(project),
                batches=batches.list(project["id"]), models=selectable_models(),
                export_configured=bool(settings.export_roots), error=message), 400

        if not source:
            return fail("Choose a folder of images.")
        if not settings.is_allowed_dataset_root(source):
            return fail("That folder is outside the locations this app can read.")
        if not Path(source).is_dir():
            return fail(f"{source} is not a folder.")

        batch_id = batches.create(project["id"], name, source, "")
        batch_dir = settings.batch_dir(slug, batch_id)
        with db.write() as conn:
            conn.execute("UPDATE annotation_batches SET dir_path=? WHERE id=?",
                         (str(batch_dir), batch_id))

        intake = annotation.copy_into_batch(Path(source), batch_dir)
        if not intake.n_copied:
            with db.write() as conn:
                conn.execute("DELETE FROM annotation_batches WHERE id=?", (batch_id,))
            shutil.rmtree(batch_dir, ignore_errors=True)
            return fail("No images were found in that folder.")

        batches.register_images(
            batch_id, sorted(p.name for p in (batch_dir / "images").iterdir()))
        (batch_dir / "batch.json").write_text(json.dumps({
            "source_path": source, "created_at": utcnow(),
            "n_copied": intake.n_copied, "n_skipped": intake.n_skipped,
            "bytes": intake.bytes_copied,
        }, indent=2), encoding="utf-8")
        return redirect(url_for("annotate_batch", slug=slug, batch_id=batch_id))

    @app.get("/p/<slug>/annotate/<int:batch_id>")
    def annotate_batch(slug: str, batch_id: int):
        project = get_project_or_404(slug)
        batch = get_batch_or_404(project, batch_id)
        chosen = request.args.get("filter", "all")
        images = batches.images(batch_id, chosen)
        model_rows = selectable_models()
        # Suggestions, not a whitelist. Three sources, because each covers a gap
        # the others leave: what the models can predict, what the project's
        # dataset already contains, and what has been typed into this batch. A
        # species in none of them is exactly the case worth supporting — a new
        # animal turns up and the model has never seen it.
        snapshot = snapshots.latest(project["id"])
        classes = sorted(
            {name for m in model_rows for name in loads(m["class_map_json"], {})}
            | {c["class_name"] for c in
               (snapshots.classes(snapshot["id"]) if snapshot else [])}
            | set(annotation.known_labels(Path(batch["dir_path"]))))
        return render_template(
            "annotate_batch.html", **project_context(project),
            batch=batch, images=images, filters=annotation.FILTERS,
            chosen_filter=chosen, models=model_rows, class_names=classes,
            progress=batches.progress(batch_id), passes=batches.passes(batch_id),
            publishes=batches.publishes(batch_id),
            running=active_job(project["id"], "predict"),
            infer_gpu=settings.infer_gpu,
            export_configured=bool(settings.export_roots))

    @app.post("/p/<slug>/annotate/<int:batch_id>/run")
    def run_prediction(slug: str, batch_id: int):
        project = get_project_or_404(slug)
        batch = get_batch_or_404(project, batch_id)
        wants_json = request.headers.get("Accept", "").startswith("application/json")

        running = active_job(project["id"], "predict")
        if running:
            if wants_json:
                return jsonify({"job_id": running["id"], "already_running": True}), 200
            return redirect(url_for("annotate_batch", slug=slug, batch_id=batch_id))

        model = models.get(int(request.form.get("model_version_id") or 0))
        if model is None:
            if wants_json:
                return jsonify({"error": "Choose a model."}), 400
            return redirect(url_for("annotate_batch", slug=slug, batch_id=batch_id))

        owner = projects.get(model["project_id"])
        predicted_by = f"{owner['slug'] if owner else 'model'} v{model['version']}"

        def number(name, default, cast=float):
            raw = request.form.get(name)
            try:
                return cast(raw) if raw not in (None, "") else default
            except (TypeError, ValueError):
                return default

        job_id = jobs.create(
            "predict", project_id=project["id"], model_version_id=model["id"],
            # The inference GPU, not the training one: an annotation pass must
            # never queue behind a run that has two days left.
            resources=[f"gpu:{settings.infer_gpu}", f"annotation:{batch_id}"],
            params={
                "batch_id": batch_id,
                "batch_dir": batch["dir_path"],
                "weights": model["weights_path"],
                "model_version_id": model["id"],
                "predicted_by": predicted_by,
                "device": str(settings.infer_gpu),
                "conf": number("conf", 0.35),
                "iou": number("iou", 0.45),
                "imgsz": number("imgsz", 1024, int),
                "max_det": number("max_det", 300, int),
                # Free text: forcing a species the model cannot predict is the
                # normal way to tag a new animal, since the model is only being
                # asked to find the boxes.
                "force_class": (request.form.get("force_class") or "").strip(),
                "only_classes": request.form.getlist("only_classes"),
                "only_missing": request.form.get("only_missing") == "on",
            })
        if wants_json:
            return jsonify({"job_id": job_id, "already_running": False}), 202
        return redirect(url_for("annotate_batch", slug=slug, batch_id=batch_id))

    # -- the review editor talks JSON ------------------------------------
    @app.get("/api/batches/<int:batch_id>/images/<path:rel_path>")
    def batch_annotation(batch_id: int, rel_path: str):
        batch = batches.get(batch_id) or abort(404)
        row = batches.image(batch_id, rel_path)
        if row is None:
            abort(404)
        batch_dir = Path(batch["dir_path"])
        width, height = row["width"], row["height"]
        if not width or not height:
            try:
                width, height = annotation.image_size(batch_dir / "images" / rel_path)
            except (OSError, ValueError):
                width = height = 0
        return jsonify({"image": rel_path, "width": width, "height": height,
                        "state": row["state"],
                        "boxes": annotation.load_boxes(batch_dir, rel_path)})

    @app.get("/api/batches/<int:batch_id>/file/<path:rel_path>")
    def batch_image_file(batch_id: int, rel_path: str):
        """Serve one image out of a batch.

        Same containment check as the run-artifact route: this process runs as
        root with no authentication, so a path assembled from a URL is only ever
        served after proving it stayed inside the directory it belongs to.
        """
        batch = batches.get(batch_id) or abort(404)
        root = (Path(batch["dir_path"]) / "images").resolve()
        try:
            target = (root / rel_path).resolve()
        except OSError:
            abort(404)
        if not target.is_relative_to(root) or not target.is_file():
            abort(404)
        return send_file(target)

    @app.post("/api/batches/<int:batch_id>/images/<path:rel_path>")
    def save_annotation(batch_id: int, rel_path: str):
        batch = batches.get(batch_id) or abort(404)
        row = batches.image(batch_id, rel_path)
        if row is None:
            abort(404)
        payload = request.get_json(silent=True) or {}
        boxes = payload.get("boxes") or []
        batch_dir = Path(batch["dir_path"])

        width = int(payload.get("width") or row["width"] or 0)
        height = int(payload.get("height") or row["height"] or 0)
        if not width or not height:
            width, height = annotation.image_size(batch_dir / "images" / rel_path)

        annotation.save_boxes(batch_dir, rel_path, boxes, width, height,
                              predicted_by=str(payload.get("predicted_by") or ""))
        batches.record_boxes(batch_id, rel_path, boxes, state="reviewed",
                             width=width, height=height)
        return jsonify({"ok": True, **batches.progress(batch_id)})

    @app.delete("/api/batches/<int:batch_id>/images/<path:rel_path>")
    def delete_batch_image(batch_id: int, rel_path: str):
        """Drop one image from a batch, with its annotation if it has one.

        The batch's copy only — the folder the images came from is untouched, so
        this is safe on a frame that is out of focus or fired by a branch.
        """
        batch = batches.get(batch_id) or abort(404)
        if batches.image(batch_id, rel_path) is None:
            abort(404)
        try:
            removed = annotation.delete_image(Path(batch["dir_path"]), rel_path)
        except ValueError:
            abort(404)
        batches.forget_image(batch_id, rel_path)
        return jsonify({"ok": True, "removed": removed, **batches.progress(batch_id)})

    # -- publish ----------------------------------------------------------
    @app.get("/p/<slug>/annotate/<int:batch_id>/publish")
    def publish_page(slug: str, batch_id: int):
        project = get_project_or_404(slug)
        batch = get_batch_or_404(project, batch_id)
        return render_template("annotate_publish.html", **project_context(project),
                               batch=batch, progress=batches.progress(batch_id),
                               publishes=batches.publishes(batch_id),
                               export_roots=[str(r) for r in settings.export_roots])

    @app.post("/p/<slug>/annotate/<int:batch_id>/publish")
    def publish_batch(slug: str, batch_id: int):
        project = get_project_or_404(slug)
        batch = get_batch_or_404(project, batch_id)
        dest = (request.form.get("dest_path") or "").strip()

        def fail(message: str, code: int = 400):
            return render_template(
                "annotate_publish.html", **project_context(project), batch=batch,
                progress=batches.progress(batch_id),
                publishes=batches.publishes(batch_id),
                export_roots=[str(r) for r in settings.export_roots],
                error=message), code

        if not dest:
            return fail("Choose where to write the images and annotations.")
        try:
            manifest = annotation.publish(
                db, settings, batch, dest,
                reviewed_only=request.form.get("reviewed_only") == "on",
                include_images=request.form.get("include_images") == "on")
        except annotation.ExportRefused as e:
            return fail(str(e), 403)
        except OSError as e:
            return fail(f"Could not write there: {e}")
        if not manifest["n_images"]:
            return fail("Nothing to export yet — review at least one image, or "
                        "untick 'reviewed images only'.")
        return redirect(url_for("annotate_batch", slug=slug, batch_id=batch_id))

    @app.get("/api/gpus")
    def api_gpus():
        return jsonify({"gpus": [g.as_dict() for g in query_gpus()]})

    @app.get("/api/runs/<int:run_id>/metrics")
    def api_metrics(run_id: int):
        run = runs.get(run_id) or abort(404)
        return jsonify({
            "run": supervisor.run_snapshot(run),
            "series": ingestor.series(run_id, request.args.getlist("key") or None),
        })

    @app.get("/api/runs/<int:run_id>/log")
    def api_log(run_id: int):
        run = runs.get(run_id) or abort(404)
        offset = request.args.get("offset", type=int, default=0)
        return jsonify(tail_bytes(Path(run["run_dir"]) / "train.log", offset))

    @app.get("/api/projects/<int:project_id>/steps")
    def api_steps(project_id: int):
        project = projects.get(project_id) or abort(404)
        return jsonify({"steps": compute_steps(db, project)})

    @app.get("/api/jobs/<int:job_id>")
    def api_job(job_id: int):
        return jsonify(jobs.get(job_id) or abort(404))

    @app.get("/runs/<int:run_id>/artifacts/<path:name>")
    def artifact(run_id: int, name: str):
        """Serve a plot or sample image from a run directory.

        Path-validated against the run directory: without this,
        ``../../../../etc/shadow`` would be served happily, and this process runs
        as root with no authentication.
        """
        run = runs.get(run_id) or abort(404)
        base = Path(run.get("save_dir") or run["run_dir"]).resolve()
        target = (base / name).resolve()
        if not target.is_relative_to(base) or not target.is_file():
            abort(404)
        return send_file(target)

    # -- SSE --------------------------------------------------------------
    @app.get("/stream")
    def stream():
        """One multiplexed stream. ``?topics=gpu,jobs,run:12``

        Deliberately a single endpoint: browsers allow only a handful of
        concurrent connections per origin, so a stream per topic would stall
        once a couple of tabs were open.
        """
        topics = {t for t in (request.args.get("topics") or "gpu,jobs").split(",") if t}
        sub = bus.subscribe(topics)
        if sub is None:
            return jsonify({"error": "Too many live connections. Close a tab and retry."}), 503

        def generate():
            try:
                yield from sse_stream(sub, hello=supervisor.hello(sorted(topics)))
            finally:
                bus.unsubscribe(sub)

        return Response(generate(), mimetype="text/event-stream", headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",     # stop nginx buffering the stream
            "Connection": "keep-alive",
        })

    @app.get("/healthz")
    def healthz():
        return jsonify({"ok": True, "projects": len(projects.list())})

    @app.errorhandler(404)
    def not_found(e):
        if request.path.startswith("/api/"):
            return jsonify({"error": str(e)}), 404
        return render_template("error.html", code=404, message=str(e)), 404

    return app


app = None  # gunicorn uses the factory: "sparrow_web.app:create_app()"
