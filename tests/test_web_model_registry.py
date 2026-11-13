"""A finished run's checkpoint must become a usable model.

Stopping a run early is normal and the training page invites it — "you can stop
it at any time and keep the best checkpoint". A run that reached epoch 24 with a
good best.pt and produced nothing selectable would make that a lie.
"""

from __future__ import annotations

import pytest

flask = pytest.importorskip("flask", reason="Flask not installed")

from sparrow_web.db import Database, utcnow
from sparrow_web.events import EventBus
from sparrow_web.repo import Projects
from sparrow_web.settings import Settings
from sparrow_web.supervisor import TERMINAL_RUN_STATES, Supervisor


@pytest.fixture
def env(tmp_path):
    settings = Settings(state_dir=tmp_path / "state", runs_root=tmp_path / "runs",
                        prepared_root=tmp_path / "prepared",
                        dataset_search_roots=[tmp_path])
    db = Database(settings.db_path)
    db.migrate()
    projects = Projects(db)
    project = projects.get(projects.create("P", str(tmp_path)))
    return {"db": db, "settings": settings, "project": project, "tmp": tmp_path}


def add_run(env, state, *, with_weights=True, name="r1"):
    run_dir = env["settings"].runs_root / name
    if with_weights:
        (run_dir / "weights").mkdir(parents=True)
        (run_dir / "weights" / "best.pt").write_bytes(b"weights")
    else:
        run_dir.mkdir(parents=True)
    with env["db"].write() as c:
        cur = c.execute(
            "INSERT INTO runs(project_id,name,run_dir,save_dir,state,epoch,epochs,"
            "best_fitness,queued_at,updated_at) VALUES (?,?,?,?,?,24,100,0.861,?,?)",
            (env["project"]["id"], name, str(run_dir), str(run_dir), state,
             utcnow(), utcnow()))
        return int(cur.lastrowid)


def supervisor(env):
    return Supervisor(env["db"], env["settings"], EventBus())


class TestBackfill:
    @pytest.mark.parametrize("state", TERMINAL_RUN_STATES)
    def test_any_finished_run_with_a_checkpoint_becomes_a_model(self, env, state):
        """Including `died`: an OOM at epoch 60 still leaves 60 epochs in best.pt."""
        run_id = add_run(env, state)
        assert supervisor(env).backfill_models() == 1
        model = env["db"].one("SELECT * FROM model_versions WHERE run_id=?", (run_id,))
        assert model and model["version"] == 1
        assert model["weights_path"].endswith("best.pt")

    def test_a_run_still_going_is_left_alone(self, env):
        add_run(env, "running")
        assert supervisor(env).backfill_models() == 0

    def test_a_run_that_never_wrote_a_checkpoint_is_skipped(self, env):
        """Run 2 on the real machine died during the label scan. There is
        nothing to register and pretending otherwise would put a broken model
        in the picker."""
        add_run(env, "failed", with_weights=False)
        assert supervisor(env).backfill_models() == 0
        assert env["db"].one("SELECT COUNT(*) c FROM model_versions")["c"] == 0

    def test_running_it_twice_registers_nothing_new(self, env):
        add_run(env, "stopped")
        s = supervisor(env)
        assert s.backfill_models() == 1
        assert s.backfill_models() == 0
        assert env["db"].one("SELECT COUNT(*) c FROM model_versions")["c"] == 1

    def test_versions_follow_run_order(self, env):
        add_run(env, "completed", name="r1")
        add_run(env, "stopped", name="r2")
        assert supervisor(env).backfill_models() == 2
        versions = [m["version"] for m in env["db"].query(
            "SELECT version FROM model_versions ORDER BY run_id")]
        assert versions == [1, 2]
