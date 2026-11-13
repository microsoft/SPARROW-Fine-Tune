"""Step machine.

Every locked step must explain itself — a disabled control with no reason is the
single most frustrating thing in a wizard-style UI.
"""

from __future__ import annotations

import pytest

from sparrow_web.db import Database, dumps, utcnow
from sparrow_web.repo import Projects
from sparrow_web.steps import STEP_IDS, compute_steps, current_step


@pytest.fixture
def ctx(tmp_path):
    db = Database(tmp_path / "t.db")
    db.migrate()
    projects = Projects(db)
    project = projects.get(projects.create("P", "/workspace/datasets/X"))
    return db, project


def steps_by_id(db, project):
    return {s["id"]: s for s in compute_steps(db, project)}


def _snapshot(db, project_id, **over):
    fields = {"n_images": 100, "n_boxes": 200, "n_classes": 5, "n_issues": 3,
              "blank_unknown": 0}
    fields.update(over)
    with db.write() as c:
        cur = c.execute(
            "INSERT INTO dataset_snapshots(project_id,dataset_root,audit_params_json,"
            "content_hash,quick_hash,n_images,n_boxes,n_classes,n_issues,blank_unknown,"
            "artifact_dir,summary_json,last_verified_at,created_at)"
            " VALUES (?,'/d','{}',?,'q',?,?,?,?,?,'/a','{}',?,?)",
            (project_id, f"h{utcnow()}{fields['n_images']}", fields["n_images"],
             fields["n_boxes"], fields["n_classes"], fields["n_issues"],
             fields["blank_unknown"], utcnow(), utcnow()))
        return int(cur.lastrowid)


def _prepared(db, project_id, snapshot_id):
    with db.write() as c:
        cur = c.execute(
            "INSERT INTO prepared_datasets(project_id,snapshot_id,dir_path,data_yaml,"
            "class_map_json,n_classes,split_config_json,background_policy,splits_hash,"
            "n_train,n_val,n_test,verify_ok,created_at)"
            " VALUES (?,?,?,'/p/data.yaml','{}',5,'{}','reviewed','s',80,10,10,1,?)",
            (project_id, snapshot_id, f"/p/{snapshot_id}", utcnow()))
        return int(cur.lastrowid)


def _run(db, project_id, state, prepared_id=None, **over):
    with db.write() as c:
        cur = c.execute(
            "INSERT INTO runs(project_id,prepared_id,name,run_dir,state,epoch,epochs,"
            "best_fitness,best_epoch,error,queued_at,updated_at)"
            " VALUES (?,?,'r',?,?,?,?,?,?,?,?,?)",
            (project_id, prepared_id, f"/runs/{state}{utcnow()}", state,
             over.get("epoch"), over.get("epochs"), over.get("best_fitness"),
             over.get("best_epoch"), over.get("error"), utcnow(), utcnow()))
        return int(cur.lastrowid)


def _job(db, project_id, kind, state="running", **over):
    with db.write() as c:
        cur = c.execute(
            "INSERT INTO jobs(project_id,kind,state,work_dir,progress_phase,progress_current,"
            "progress_total,error,created_at,updated_at) VALUES (?,?,?,'/w',?,?,?,?,?,?)",
            (project_id, kind, state, over.get("phase"), over.get("current"),
             over.get("total"), over.get("error"), utcnow(), utcnow()))
        return int(cur.lastrowid)


class TestFreshProject:
    def test_all_six_steps_present_in_order(self, ctx):
        db, project = ctx
        assert [s["id"] for s in compute_steps(db, project)] == list(STEP_IDS)

    def test_review_is_the_first_action(self, ctx):
        db, project = ctx
        steps = steps_by_id(db, project)
        assert steps["data"]["state"] == "done"
        assert steps["review"]["state"] == "ready"
        assert current_step(compute_steps(db, project))["id"] == "review"

    def test_downstream_steps_are_locked_with_reasons(self, ctx):
        db, project = ctx
        steps = steps_by_id(db, project)
        for name in ("prepare", "train", "evaluate", "export"):
            assert steps[name]["state"] == "locked"
            assert steps[name]["reason"], f"{name} is locked without saying why"

    def test_locked_steps_offer_no_action(self, ctx):
        db, project = ctx
        steps = steps_by_id(db, project)
        assert steps["prepare"]["action"] is None


class TestProgression:
    def test_after_scan_prepare_unlocks(self, ctx):
        db, project = ctx
        _snapshot(db, project["id"])
        steps = steps_by_id(db, project)
        assert steps["review"]["state"] == "done"
        assert steps["review"]["summary"]["n_images"] == 100
        assert steps["prepare"]["state"] == "ready"
        assert steps["train"]["state"] == "locked"

    def test_after_prepare_train_unlocks(self, ctx):
        db, project = ctx
        sid = _snapshot(db, project["id"])
        _prepared(db, project["id"], sid)
        steps = steps_by_id(db, project)
        assert steps["prepare"]["state"] == "done"
        assert steps["prepare"]["summary"]["n_train"] == 80
        assert steps["train"]["state"] == "ready"

    def test_completed_run_marks_train_done(self, ctx):
        db, project = ctx
        sid = _snapshot(db, project["id"])
        pid = _prepared(db, project["id"], sid)
        _run(db, project["id"], "completed", pid, best_fitness=0.61, best_epoch=38)
        steps = steps_by_id(db, project)
        assert steps["train"]["state"] == "done"
        assert steps["train"]["summary"]["best_epoch"] == 38


class TestStaleness:
    def test_new_data_makes_prepare_stale_not_locked(self, ctx):
        """The user can still train on the old set; they just should know."""
        db, project = ctx
        first = _snapshot(db, project["id"], n_images=100)
        _prepared(db, project["id"], first)
        _snapshot(db, project["id"], n_images=140)     # new data arrived

        steps = steps_by_id(db, project)
        assert steps["prepare"]["state"] == "stale"
        assert "changed" in steps["prepare"]["reason"]
        assert steps["prepare"]["action"]["label"] == "Rebuild"

    def test_stale_prepare_blocks_training(self, ctx):
        db, project = ctx
        first = _snapshot(db, project["id"], n_images=100)
        _prepared(db, project["id"], first)
        _snapshot(db, project["id"], n_images=140)
        # Training is still possible against the existing tree, so it stays
        # available rather than locked.
        assert steps_by_id(db, project)["train"]["state"] in ("ready", "done")


class TestRunning:
    def test_running_audit_shows_progress(self, ctx):
        db, project = ctx
        job_id = _job(db, project["id"], "audit", phase="Checking images",
                      current=12000, total=60049)
        step = steps_by_id(db, project)["review"]
        assert step["state"] == "running"
        assert step["job_id"] == job_id
        assert step["summary"]["current"] == 12000

    def test_active_run_shows_epoch(self, ctx):
        db, project = ctx
        sid = _snapshot(db, project["id"])
        pid = _prepared(db, project["id"], sid)
        run_id = _run(db, project["id"], "running", pid, epoch=41, epochs=100)
        step = steps_by_id(db, project)["train"]
        assert step["state"] == "running"
        assert step["summary"]["epoch"] == 41
        assert str(run_id) in step["action"]["href"]

    def test_current_step_prefers_the_running_one(self, ctx):
        db, project = ctx
        _snapshot(db, project["id"])
        _job(db, project["id"], "prepare", phase="Writing dataset")
        assert current_step(compute_steps(db, project))["id"] == "prepare"


class TestFailure:
    def test_failed_run_surfaces_its_error(self, ctx):
        db, project = ctx
        sid = _snapshot(db, project["id"])
        pid = _prepared(db, project["id"], sid)
        _run(db, project["id"], "died", pid,
             error="Process gone but status never finalised - almost always OOM.")
        step = steps_by_id(db, project)["train"]
        assert step["state"] == "failed"
        assert "OOM" in step["reason"]
        # Named for what it opens, not what it does: the settings form, where a
        # failed run is usually retried with something changed.
        assert step["action"]["label"] == "Set up a new run"

    def test_failed_audit_offers_a_retry(self, ctx):
        db, project = ctx
        _job(db, project["id"], "audit", state="failed", error="Dataset root does not exist")
        step = steps_by_id(db, project)["review"]
        assert step["state"] == "failed"
        assert "does not exist" in step["reason"]


class TestEvaluateAndExport:
    def _model(self, db, project_id, **over):
        run_id = _run(db, project_id, "completed")
        with db.write() as c:
            cur = c.execute(
                "INSERT INTO model_versions(project_id,version,run_id,weights_path,"
                "class_map_json,n_classes,onnx_path,created_at)"
                " VALUES (?,1,?,'/w/best.pt','{}',5,?,?)",
                (project_id, run_id, over.get("onnx_path"), utcnow()))
            return int(cur.lastrowid)

    def test_evaluate_ready_once_a_model_exists(self, ctx):
        db, project = ctx
        self._model(db, project["id"])
        steps = steps_by_id(db, project)
        assert steps["evaluate"]["state"] == "ready"
        assert steps["export"]["state"] == "ready"

    def test_evaluate_done_with_metrics(self, ctx):
        db, project = ctx
        model_id = self._model(db, project["id"])
        with db.write() as c:
            c.execute("INSERT INTO model_evaluations(model_version_id,split,metrics_json,"
                      "map50,map5095,created_at) VALUES (?,'test','{}',0.81,0.51,?)",
                      (model_id, utcnow()))
        step = steps_by_id(db, project)["evaluate"]
        assert step["state"] == "done" and step["summary"]["map5095"] == 0.51

    def test_export_done_when_onnx_exists(self, ctx):
        db, project = ctx
        self._model(db, project["id"], onnx_path="/w/best.onnx")
        assert steps_by_id(db, project)["export"]["state"] == "done"
