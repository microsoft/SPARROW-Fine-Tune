"""Projects, snapshot ingestion, diffing and model lineage.

The diff is the "what's new since the last model" feature, so its distinction
between *new images* and *re-tagged images* is tested explicitly — they demand
different responses from the user.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from sparrow_web.db import Database, loads, utcnow
from sparrow_web.repo import (Models, Projects, Runs, Snapshots, ingest_audit_result,
                              slugify)
from sparrow_web.settings import Settings


@pytest.fixture
def ctx(tmp_path):
    db = Database(tmp_path / "t.db")
    db.migrate()
    settings = Settings(state_dir=tmp_path / "state")
    return db, settings, tmp_path


class TestSlugify:
    @pytest.mark.parametrize("name,expected", [
        ("Sub Saharan v8", "sub_saharan_v8"),
        ("  Spaces  ", "spaces"),
        ("Weird!!Chars@@", "weird_chars"),
        ("", "project"),
        ("///", "project"),
    ])
    def test_cases(self, name, expected):
        assert slugify(name) == expected


class TestProjects:
    def test_create_validates_config_through_the_library(self, ctx):
        """The UI must not be able to store settings the CLI would reject."""
        db, _, _ = ctx
        projects = Projects(db)
        with pytest.raises(ValueError, match="sum to 1.0"):
            projects.create("Bad", "/data", config={"split": {"train": 0.5, "val": 0.1,
                                                              "test": 0.1}})

    def test_create_seeds_config_from_defaults(self, ctx):
        db, _, _ = ctx
        projects = Projects(db)
        pid = projects.create("Sub Saharan", "/workspace/datasets/Sub_Saharan")
        project = projects.get(pid)
        cfg = loads(project["config_json"])
        assert project["slug"] == "sub_saharan"
        assert cfg["dataset"]["root"] == "/workspace/datasets/Sub_Saharan"
        assert cfg["training"]["imgsz"] == 1024      # from library DEFAULTS

    def test_duplicate_names_get_distinct_slugs(self, ctx):
        db, _, _ = ctx
        projects = Projects(db)
        projects.create("Same", "/a")
        projects.create("Same", "/b")
        assert {p["slug"] for p in projects.list()} == {"same", "same_2"}

    def test_update_config_is_validated(self, ctx):
        db, _, _ = ctx
        projects = Projects(db)
        pid = projects.create("P", "/a")
        with pytest.raises(ValueError, match="multiple of 32"):
            projects.update_config(pid, {"training": {"imgsz": 1000}})

    def test_archived_projects_are_hidden_by_default(self, ctx):
        db, _, _ = ctx
        projects = Projects(db)
        pid = projects.create("P", "/a")
        with db.write() as c:
            c.execute("UPDATE projects SET archived=1 WHERE id=?", (pid,))
        assert projects.list() == []
        assert len(projects.list(include_archived=True)) == 1


def _audit_job(db, settings, project, root, artifact_dir):
    """Run a real audit and hand its result to the ingester."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from sparrow_web.jobs import JobQueue
    from sparrow_web.worker import _handle_audit
    from sparrow_yolo.process_train import StatusWriter

    queue = JobQueue(db, settings)
    job_id = queue.create("audit", project_id=project["id"], params={
        "dataset_root": str(root), "artifact_dir": str(artifact_dir)})
    job = queue.get(job_id)
    status = StatusWriter(Path(job["work_dir"]) / "status.json")
    result = _handle_audit(loads(job["params_json"]), status)
    return ingest_audit_result(db, settings, project, job, result)


class TestSnapshotIngestion:
    def test_ingests_counts_and_per_file_rows(self, ctx):
        db, settings, tmp_path = ctx
        from tests.conftest import write_pair

        root = tmp_path / "ds"
        for i in range(6):
            write_pair(root, f"i{i}", objects=[("lion", 10, 10, 60, 60)])

        projects = Projects(db)
        project = projects.get(projects.create("P", str(root)))
        sid = _audit_job(db, settings, project, root, tmp_path / "snap1")

        snap = Snapshots(db).get(sid)
        assert snap["n_images"] == 6 and snap["n_boxes"] == 6 and snap["n_classes"] == 1
        assert db.scalar("SELECT COUNT(*) FROM snapshot_files WHERE snapshot_id=?", (sid,)) == 6
        assert Snapshots(db).classes(sid)[0]["class_name"] == "lion"

    def test_reaudit_of_unchanged_data_does_not_duplicate(self, ctx):
        """Six snapshots should mean the data changed six times."""
        db, settings, tmp_path = ctx
        from tests.conftest import write_pair

        root = tmp_path / "ds"
        for i in range(4):
            write_pair(root, f"i{i}", objects=[("lion", 10, 10, 60, 60)])
        projects = Projects(db)
        project = projects.get(projects.create("P", str(root)))

        first = _audit_job(db, settings, project, root, tmp_path / "s1")
        second = _audit_job(db, settings, project, root, tmp_path / "s2")
        assert first == second
        assert db.scalar("SELECT COUNT(*) FROM dataset_snapshots") == 1
        assert Snapshots(db).get(first)["n_audits"] == 2

    def test_issue_sample_is_capped_but_counts_are_exact(self, ctx):
        """A partly-tagged dataset emits one issue per untagged image."""
        db, tmp_path = ctx[0], ctx[2]
        settings = Settings(state_dir=tmp_path / "state", issue_sample_cap=5)
        from tests.conftest import write_image, write_pair

        root = tmp_path / "ds"
        for i in range(3):
            write_pair(root, f"tagged{i}", objects=[("lion", 10, 10, 60, 60)])
        for i in range(25):
            write_image(root / f"untagged{i}.jpg")

        projects = Projects(db)
        project = projects.get(projects.create("P", str(root)))
        sid = _audit_job(db, settings, project, root, tmp_path / "snap")

        counts = Snapshots(db).issue_counts(sid)
        assert counts["missing_xml"] == 25, "the exact count must be preserved"
        stored = db.scalar(
            "SELECT COUNT(*) FROM audit_issues WHERE snapshot_id=? AND issue_type='missing_xml'",
            (sid,))
        assert stored == 5, "the stored sample must be capped"


class TestDiff:
    def _project_with_two_snapshots(self, db, settings, tmp_path, mutate):
        from tests.conftest import write_pair

        root = tmp_path / "ds"
        for i in range(6):
            write_pair(root, f"i{i}", objects=[("lion", 10, 10, 60, 60)])
        projects = Projects(db)
        project = projects.get(projects.create("P", str(root)))
        first = _audit_job(db, settings, project, root, tmp_path / "s1")
        mutate(root)
        second = _audit_job(db, settings, project, root, tmp_path / "s2")
        return first, second

    def test_new_images_are_counted(self, ctx):
        db, settings, tmp_path = ctx
        from tests.conftest import write_pair

        def add(root):
            for i in range(3):
                write_pair(root, f"new{i}", objects=[("elephant", 5, 5, 50, 50)])

        a, b = self._project_with_two_snapshots(db, settings, tmp_path, add)
        diff = Snapshots(db).diff(a, b)
        assert diff["images_added"] == 3
        assert diff["boxes_added"] == 3
        assert diff["new_classes"] == ["elephant"]
        assert diff["changed"] is True
        assert len(diff["sample_added"]) == 3

    def test_relabelling_is_distinguished_from_new_images(self, ctx):
        """Same image bytes, different boxes: existing labels were corrected, so
        any model trained on them is stale. That is a different message to the
        user than 'new data arrived'."""
        db, settings, tmp_path = ctx
        from tests.conftest import write_xml

        def relabel(root):
            write_xml(root / "i0.xml", filename="i0.jpg", width=200, height=100,
                      objects=[("lion", 10, 10, 60, 60), ("lion", 80, 20, 150, 90)])

        a, b = self._project_with_two_snapshots(db, settings, tmp_path, relabel)
        diff = Snapshots(db).diff(a, b)
        assert diff["images_added"] == 0
        assert diff["images_modified"] == 0, "image bytes did not change"
        assert diff["images_relabelled"] == 1
        assert diff["changed"] is True

    def test_removed_images_are_counted(self, ctx):
        db, settings, tmp_path = ctx

        def remove(root):
            (root / "i0.jpg").unlink()
            (root / "i0.xml").unlink()

        a, b = self._project_with_two_snapshots(db, settings, tmp_path, remove)
        diff = Snapshots(db).diff(a, b)
        assert diff["images_removed"] == 1 and diff["boxes_removed"] == 1

    def test_identical_snapshots_report_no_change(self, ctx):
        db, settings, tmp_path = ctx
        a, b = self._project_with_two_snapshots(db, settings, tmp_path, lambda root: None)
        diff = Snapshots(db).diff(a, b)
        assert diff["changed"] is False and diff["images_added"] == 0


class TestModelLineage:
    def _run(self, db, project_id, base_model_version_id=None):
        with db.write() as c:
            cur = c.execute(
                "INSERT INTO runs(project_id,name,run_dir,state,base_model_version_id,"
                "best_fitness,queued_at,updated_at) VALUES (?,?,?,'completed',?,0.5,?,?)",
                (project_id, "r", f"/runs/r{utcnow()}{base_model_version_id}",
                 base_model_version_id, utcnow(), utcnow()))
            return int(cur.lastrowid)

    def test_versions_increment_and_chain(self, ctx):
        db, _, _ = ctx
        projects = Projects(db)
        pid = projects.create("P", "/a")
        models = Models(db)

        v1 = models.create_from_run(pid, Runs(db).get(self._run(db, pid)),
                                    weights_path="/w/v1.pt", class_map={"lion": 0})
        v2 = models.create_from_run(pid, Runs(db).get(self._run(db, pid, v1)),
                                    weights_path="/w/v2.pt", class_map={"lion": 0})
        v3 = models.create_from_run(pid, Runs(db).get(self._run(db, pid, v2)),
                                    weights_path="/w/v3.pt", class_map={"lion": 0})

        assert [m["version"] for m in models.list(pid)] == [3, 2, 1]
        chain = models.lineage(v3)
        assert [m["id"] for m in chain] == [v1, v2, v3], "lineage must read oldest-first"

    def test_first_version_has_no_parent(self, ctx):
        db, _, _ = ctx
        pid = Projects(db).create("P", "/a")
        models = Models(db)
        v1 = models.create_from_run(pid, Runs(db).get(self._run(db, pid)),
                                    weights_path="/w/v1.pt", class_map={})
        assert models.get(v1)["parent_model_version_id"] is None
        assert len(models.lineage(v1)) == 1
