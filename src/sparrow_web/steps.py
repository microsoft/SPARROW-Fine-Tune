"""The step machine.

One place decides what a project's next action is, and why anything is blocked.
Both the page and the API read from here, so a disabled button and a rejected
request always agree — and, more importantly, a blocked step can always explain
itself in a sentence rather than presenting a dead control.

States:
  ``locked``    a prerequisite is missing. ``reason`` is always populated.
  ``ready``     can be started now.
  ``running``   a job holds it.
  ``stale``     done, but its inputs have moved on since.
  ``done``      complete and current.
  ``failed``    the last attempt failed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .db import Database

STEP_IDS = ("data", "review", "prepare", "train", "evaluate", "export")

# Shown before a scan starts. A 60k-image folder takes minutes, and a user who
# was not warned assumes the app has hung.
SCAN_CONFIRM = (
    "Scan the dataset?\n\n"
    "Every image and annotation is read, so this takes a few minutes on a large "
    "folder. It runs in the background — you can carry on using the app.")


@dataclass
class Step:
    id: str
    label: str
    state: str
    summary: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    action: Optional[Dict[str, str]] = None
    job_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _running_job(db: Database, project_id: int, kind: str) -> Optional[Dict[str, Any]]:
    return db.one(
        "SELECT * FROM jobs WHERE project_id=? AND kind=? AND state IN "
        "('queued','leased','running') ORDER BY id DESC LIMIT 1", (project_id, kind))


def _last_failed(db: Database, project_id: int, kind: str) -> Optional[Dict[str, Any]]:
    return db.one(
        "SELECT * FROM jobs WHERE project_id=? AND kind=? ORDER BY id DESC LIMIT 1",
        (project_id, kind))


def compute_steps(db: Database, project: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Derive every step's state. Cheap enough to call on each page load."""
    pid = project["id"]

    snapshot = db.one(
        "SELECT * FROM dataset_snapshots WHERE project_id=? ORDER BY id DESC LIMIT 1", (pid,))
    prepared = db.one(
        "SELECT * FROM prepared_datasets WHERE project_id=? AND state='ready'"
        " ORDER BY id DESC LIMIT 1", (pid,))
    last_run = db.one("SELECT * FROM runs WHERE project_id=? ORDER BY id DESC LIMIT 1", (pid,))
    completed_run = db.one(
        "SELECT * FROM runs WHERE project_id=? AND state='completed' ORDER BY id DESC LIMIT 1",
        (pid,))
    model = db.one(
        "SELECT * FROM model_versions WHERE project_id=? ORDER BY version DESC LIMIT 1", (pid,))
    evaluation = db.one(
        "SELECT e.* FROM model_evaluations e JOIN model_versions m ON m.id=e.model_version_id"
        " WHERE m.project_id=? AND e.split='test' ORDER BY e.id DESC LIMIT 1", (pid,)) \
        if model else None

    steps: List[Step] = []

    # 1. Data -------------------------------------------------------------
    steps.append(Step(
        id="data", label="Data",
        state="done" if project["dataset_root"] else "ready",
        summary={"dataset_root": project["dataset_root"]},
        action={"label": "Change folder", "href": f"/p/{project['slug']}/data"},
    ))

    # 2. Review (audit) ---------------------------------------------------
    audit_job = _running_job(db, pid, "audit")
    if audit_job:
        review = Step("review", "Review", "running",
                      summary={"phase": audit_job.get("progress_phase"),
                               "current": audit_job.get("progress_current"),
                               "total": audit_job.get("progress_total")},
                      job_id=audit_job["id"])
    elif snapshot is None:
        last = _last_failed(db, pid, "audit")
        if last and last["state"] in ("failed", "died"):
            review = Step("review", "Review", "failed", reason=last.get("error") or "Audit failed",
                          action={"label": "Retry scan", "href": f"/p/{project['slug']}/review",
                                  "post": f"/p/{project['slug']}/scan",
                                  "confirm": SCAN_CONFIRM})
        else:
            review = Step("review", "Review", "ready",
                          reason="Scan the folder to see what is in it.",
                          action={"label": "Scan dataset", "href": f"/p/{project['slug']}/review",
                                  "post": f"/p/{project['slug']}/scan",
                                  "confirm": SCAN_CONFIRM})
    else:
        review = Step("review", "Review", "done", summary={
            "snapshot_id": snapshot["id"],
            "n_images": snapshot["n_images"], "n_boxes": snapshot["n_boxes"],
            "n_classes": snapshot["n_classes"], "n_issues": snapshot["n_issues"],
            "blank_unknown": snapshot["blank_unknown"],
        }, action={"label": "View data health", "href": f"/p/{project['slug']}/review"})
    steps.append(review)

    # 3. Prepare ----------------------------------------------------------
    prepare_job = _running_job(db, pid, "prepare")
    if prepare_job:
        prep = Step("prepare", "Prepare", "running",
                    summary={"phase": prepare_job.get("progress_phase"),
                             "current": prepare_job.get("progress_current"),
                             "total": prepare_job.get("progress_total")},
                    job_id=prepare_job["id"])
    elif snapshot is None:
        prep = Step("prepare", "Prepare", "locked",
                    reason="Scan the dataset first — there is nothing to prepare yet.")
    elif prepared is None:
        prep = Step("prepare", "Prepare", "ready",
                    reason="Build the training/validation/test split.",
                    action={"label": "Build dataset", "href": f"/p/{project['slug']}/dataset"})
    elif prepared["snapshot_id"] != snapshot["id"]:
        # New data has arrived since this tree was built.
        prep = Step("prepare", "Prepare", "stale",
                    reason="The dataset has changed since this training set was built.",
                    summary=_prepared_summary(prepared),
                    action={"label": "Rebuild", "href": f"/p/{project['slug']}/dataset"})
    else:
        prep = Step("prepare", "Prepare", "done", summary=_prepared_summary(prepared),
                    action={"label": "View", "href": f"/p/{project['slug']}/dataset"})
    steps.append(prep)

    # 4. Train ------------------------------------------------------------
    active_run = db.one(
        "SELECT * FROM runs WHERE project_id=? AND state IN"
        " ('queued','starting','running','stopping') ORDER BY id DESC LIMIT 1", (pid,))
    if active_run:
        train = Step("train", "Train", "running", summary={
            "run_id": active_run["id"], "epoch": active_run.get("epoch"),
            "epochs": active_run.get("epochs"), "state": active_run["state"],
        }, action={"label": "Watch", "href": f"/p/{project['slug']}/runs/{active_run['id']}"})
    elif prepared is None:
        train = Step("train", "Train", "locked",
                     reason="Build a training set before training.")
    elif last_run and last_run["state"] in ("failed", "died"):
        train = Step("train", "Train", "failed",
                     reason=last_run.get("error") or "The last run did not finish.",
                     summary={"run_id": last_run["id"]},
                     action={"label": "Set up a new run",
                             "href": f"/p/{project['slug']}/train"})
    elif completed_run is None:
        train = Step("train", "Train", "ready",
                     reason="No model has been trained yet. Choose the settings, then "
                            "start the run.",
                     action={"label": "Set up training",
                             "href": f"/p/{project['slug']}/train"})
    else:
        train = Step("train", "Train", "done", summary={
            "run_id": completed_run["id"],
            "best_fitness": completed_run.get("best_fitness"),
            "best_epoch": completed_run.get("best_epoch"),
            "stopped_early": bool(completed_run.get("stopped_early")),
        }, action={"label": "Set up another run",
                   "href": f"/p/{project['slug']}/train"})
    steps.append(train)

    # 5. Evaluate ---------------------------------------------------------
    if model is None:
        evaluate = Step("evaluate", "Evaluate", "locked",
                        reason="Train a model first.")
    elif evaluation is None:
        evaluate = Step("evaluate", "Evaluate", "ready",
                        reason="Score the model on images it has never seen.",
                        action={"label": "Evaluate",
                                "href": f"/p/{project['slug']}/models/{model['id']}"})
    else:
        evaluate = Step("evaluate", "Evaluate", "done", summary={
            "map50": evaluation.get("map50"), "map5095": evaluation.get("map5095"),
            "model_version": model["version"],
        }, action={"label": "View results",
                   "href": f"/p/{project['slug']}/models/{model['id']}"})
    steps.append(evaluate)

    # 6. Export -----------------------------------------------------------
    if model is None:
        export = Step("export", "Export", "locked", reason="Train a model first.")
    elif model.get("onnx_path"):
        export = Step("export", "Export", "done", summary={"onnx_path": model["onnx_path"]})
    else:
        export = Step("export", "Export", "ready", reason="Export to ONNX for deployment.",
                      action={"label": "Export",
                              "href": f"/p/{project['slug']}/models/{model['id']}"})
    steps.append(export)

    return [s.to_dict() for s in steps]


def _prepared_summary(prepared: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prepared_id": prepared["id"],
        "n_train": prepared["n_train"], "n_val": prepared["n_val"],
        "n_test": prepared["n_test"], "n_classes": prepared["n_classes"],
        "verify_ok": bool(prepared["verify_ok"]),
    }


def current_step(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The step the user should look at: whatever is running, else the first
    thing that is actionable, else the last one."""
    for state in ("running", "failed", "stale", "ready"):
        for step in steps:
            if step["state"] == state:
                return step
    return steps[-1]
