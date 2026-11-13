"""Deleting a project.

The point of this suite is one guarantee: whatever the database says, the
dataset folder survives. Everything else about deletion is recoverable by
scanning again; the source images are not.
"""

from __future__ import annotations

import shutil

import pytest

flask = pytest.importorskip("flask", reason="Flask not installed")

from sparrow_web import removal
from sparrow_web.db import Database, utcnow
from sparrow_web.repo import Projects
from sparrow_web.settings import Settings


@pytest.fixture
def env(tmp_path):
    datasets = tmp_path / "datasets"
    (datasets / "Sub_Saharan").mkdir(parents=True)
    (datasets / "Sub_Saharan" / "a.jpg").write_bytes(b"jpeg")
    (datasets / "Sub_Saharan" / "a.xml").write_text("<annotation/>")

    settings = Settings(
        state_dir=tmp_path / "state",
        prepared_root=tmp_path / "prepared",
        runs_root=tmp_path / "runs",
        dataset_search_roots=[datasets],
    )
    settings.state_dir.mkdir(parents=True)
    settings.prepared_root.mkdir()
    settings.runs_root.mkdir()

    db = Database(settings.db_path)
    db.migrate()
    projects = Projects(db)
    project = projects.get(projects.create("Sub Saharan", str(datasets / "Sub_Saharan")))
    return {"db": db, "settings": settings, "project": project, "datasets": datasets}


def _try_symlink(link, target, directory=False):
    """True if a symlink was created. Windows refuses them without developer
    mode, and this suite is expected to run on the host as well as in the
    container."""
    try:
        link.symlink_to(target, target_is_directory=directory)
        return True
    except OSError:
        return False


def add_snapshot(env):
    with env["db"].write() as c:
        cur = c.execute(
            "INSERT INTO dataset_snapshots(project_id,dataset_root,audit_params_json,"
            "content_hash,quick_hash,artifact_dir,summary_json,last_verified_at,created_at)"
            " VALUES (?,?,'{}','h','q','','{}',?,?)",
            (env["project"]["id"], env["project"]["dataset_root"], utcnow(), utcnow()))
        return int(cur.lastrowid)


def add_prepared(env, name="1"):
    d = env["settings"].prepared_root / env["project"]["slug"] / name
    (d / "images" / "train").mkdir(parents=True)
    link = d / "images" / "train" / "a.jpg"
    # prepare_yolo_dataset writes symlinks here; where the platform will not
    # allow one, a small stub is enough for what these tests assert.
    if not _try_symlink(link, env["datasets"] / "Sub_Saharan" / "a.jpg"):
        link.write_bytes(b"stub")
    (d / "data.yaml").write_text("names: [a]")
    snapshot_id = add_snapshot(env)
    with env["db"].write() as c:
        c.execute(
            "INSERT INTO prepared_datasets(project_id,snapshot_id,dir_path,data_yaml,"
            "class_map_json,n_classes,split_config_json,background_policy,splits_hash,"
            "created_at) VALUES (?,?,?,?,'{}',1,'{}','reviewed','h',?)",
            (env["project"]["id"], snapshot_id, str(d), str(d / "data.yaml"), utcnow()))
    return d


def add_run(env, state="completed", name="run1"):
    d = env["settings"].runs_root / name
    (d / "weights").mkdir(parents=True)
    (d / "weights" / "best.pt").write_bytes(b"weights")
    with env["db"].write() as c:
        c.execute("INSERT INTO runs(project_id,name,run_dir,save_dir,state,queued_at,"
                  "updated_at) VALUES (?,?,?,?,?,?,?)",
                  (env["project"]["id"], name, str(d), str(d), state, utcnow(), utcnow()))
    return d


class TestTheDatasetSurvives:
    """The whole reason this code is separated out and tested."""

    def test_deleting_leaves_the_source_folder_untouched(self, env):
        add_prepared(env)
        add_run(env)
        source = env["datasets"] / "Sub_Saharan"
        removal.delete(env["db"], env["settings"], env["project"], remove_files=True)
        assert source.is_dir()
        assert (source / "a.jpg").read_bytes() == b"jpeg"
        assert (source / "a.xml").exists()

    def test_the_dataset_root_is_never_a_removal_candidate(self, env):
        """Even if the database claims a prepared tree lives inside the dataset."""
        rogue = env["datasets"] / "Sub_Saharan" / "prepared_by_mistake"
        rogue.mkdir()
        snapshot_id = add_snapshot(env)
        with env["db"].write() as c:
            c.execute(
                "INSERT INTO prepared_datasets(project_id,snapshot_id,dir_path,data_yaml,"
                "class_map_json,n_classes,split_config_json,background_policy,splits_hash,"
                "created_at) VALUES (?,?,?,'','{}',1,'{}','reviewed','h',?)",
                (env["project"]["id"], snapshot_id, str(rogue), utcnow()))
        plan = removal.plan(env["db"], env["settings"], env["project"])
        assert rogue not in plan.dirs
        removal.delete(env["db"], env["settings"], env["project"], remove_files=True)
        assert rogue.is_dir()

    def test_a_symlink_out_of_a_managed_root_is_refused(self, env, tmp_path):
        """prepared/ is full of symlinks into the dataset; resolving before the
        containment check is what stops one being followed back out."""
        escape = env["settings"].prepared_root / "escape"
        if not _try_symlink(escape, env["datasets"] / "Sub_Saharan", directory=True):
            pytest.skip("symlinks unavailable on this platform")
        assert removal._removable(escape, env["settings"]) is None

    def test_a_managed_root_itself_is_never_removed(self, env):
        assert removal._removable(env["settings"].prepared_root, env["settings"]) is None
        assert removal._removable(env["settings"].runs_root, env["settings"]) is None


class TestPlan:
    def test_counts_what_would_go(self, env):
        add_prepared(env)
        add_run(env)
        plan = removal.plan(env["db"], env["settings"], env["project"])
        assert plan.n_prepared == 1 and plan.n_runs == 1
        assert plan.dataset_root == str(env["datasets"] / "Sub_Saharan")

    def test_size_does_not_follow_symlinks_into_the_dataset(self, env):
        """Otherwise a prepared tree reports the size of the whole dataset."""
        add_prepared(env)
        plan = removal.plan(env["db"], env["settings"], env["project"])
        assert plan.bytes_on_disk < 1000

    def test_a_running_job_blocks_deletion(self, env):
        with env["db"].write() as c:
            c.execute("INSERT INTO jobs(project_id,kind,state,work_dir,created_at,updated_at)"
                      " VALUES (?,'audit','running','/tmp',?,?)",
                      (env["project"]["id"], utcnow(), utcnow()))
        plan = removal.plan(env["db"], env["settings"], env["project"])
        assert not plan.can_delete
        with pytest.raises(RuntimeError):
            removal.delete(env["db"], env["settings"], env["project"], remove_files=True)

    def test_a_running_run_blocks_deletion(self, env):
        add_run(env, state="running")
        assert not removal.plan(env["db"], env["settings"], env["project"]).can_delete

    def test_a_finished_run_does_not_block(self, env):
        add_run(env, state="completed")
        assert removal.plan(env["db"], env["settings"], env["project"]).can_delete

    def test_nested_save_dir_is_not_counted_twice(self, env):
        d = add_run(env)
        with env["db"].write() as c:
            c.execute("UPDATE runs SET save_dir=? WHERE project_id=?",
                      (str(d / "weights"), env["project"]["id"]))
        plan = removal.plan(env["db"], env["settings"], env["project"])
        assert plan.dirs.count(d.resolve()) == 1
        assert (d / "weights").resolve() not in plan.dirs


class TestDelete:
    def test_removes_the_generated_trees(self, env):
        prepared, run = add_prepared(env), add_run(env)
        removal.delete(env["db"], env["settings"], env["project"], remove_files=True)
        assert not prepared.exists() and not run.exists()

    def test_keeps_the_generated_trees_when_asked(self, env):
        prepared, run = add_prepared(env), add_run(env)
        removal.delete(env["db"], env["settings"], env["project"], remove_files=False)
        assert prepared.is_dir() and (run / "weights" / "best.pt").exists()

    def test_rows_cascade(self, env):
        add_prepared(env)
        add_run(env)
        pid = env["project"]["id"]
        removal.delete(env["db"], env["settings"], env["project"], remove_files=True)
        for table in ("projects", "prepared_datasets", "runs", "dataset_snapshots"):
            column = "id" if table == "projects" else "project_id"
            n = env["db"].one(
                f"SELECT COUNT(*) c FROM {table} WHERE {column}=?", (pid,))["c"]
            assert n == 0, table

    def test_a_missing_directory_is_not_an_error(self, env):
        prepared = add_prepared(env)
        shutil.rmtree(prepared)
        removal.delete(env["db"], env["settings"], env["project"], remove_files=True)
