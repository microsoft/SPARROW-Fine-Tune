"""Schema, connection and transaction behaviour.

These are cheap but load-bearing: the resource-lease mutex is the entire
single-GPU queue, and getting its concurrency wrong would let two training runs
start on the same card.
"""

from __future__ import annotations

import sqlite3
import threading

import pytest

from sparrow_web.db import SCHEMA_VERSION, Database, dumps, loads, utcnow


@pytest.fixture
def db(tmp_path) -> Database:
    d = Database(tmp_path / "test.db")
    d.migrate()
    return d


class TestMigrate:
    def test_creates_schema_and_records_version(self, db):
        assert db.scalar("SELECT MAX(version) FROM schema_migrations") == SCHEMA_VERSION

    def test_is_idempotent(self, db):
        db.migrate()
        db.migrate()
        assert db.scalar("SELECT COUNT(*) FROM schema_migrations") == 1

    def test_expected_tables_exist(self, db):
        names = {
            r["name"] for r in db.query("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert {
            "projects", "dataset_files", "dataset_snapshots", "snapshot_files",
            "snapshot_class_counts", "snapshot_issue_counts", "audit_issues",
            "prepared_datasets", "runs", "metric_samples", "run_ingest_state",
            "model_versions", "model_evaluations", "model_class_metrics",
            "jobs", "resource_leases", "settings",
        } <= names

    def test_creates_parent_directory(self, tmp_path):
        d = Database(tmp_path / "nested" / "deeper" / "x.db")
        d.migrate()
        assert (tmp_path / "nested" / "deeper" / "x.db").exists()

    def test_foreign_keys_are_enforced(self, db):
        with pytest.raises(sqlite3.IntegrityError):
            with db.write() as c:
                c.execute(
                    "INSERT INTO dataset_files(project_id, rel_path, first_seen_at) "
                    "VALUES (999, 'x.jpg', ?)",
                    (utcnow(),),
                )

    def test_cascade_delete_removes_children(self, db):
        with db.write() as c:
            c.execute(
                "INSERT INTO projects(slug,name,dataset_root,config_json,created_at,updated_at)"
                " VALUES ('p','P','/d','{}',?,?)", (utcnow(), utcnow()))
            c.execute(
                "INSERT INTO dataset_files(project_id, rel_path, first_seen_at)"
                " VALUES (1,'a.jpg',?)", (utcnow(),))
        with db.write() as c:
            c.execute("DELETE FROM projects WHERE id=1")
        assert db.scalar("SELECT COUNT(*) FROM dataset_files") == 0


class TestWriteTransaction:
    def test_rolls_back_on_error(self, db):
        with pytest.raises(RuntimeError):
            with db.write() as c:
                c.execute(
                    "INSERT INTO projects(slug,name,dataset_root,config_json,created_at,updated_at)"
                    " VALUES ('a','A','/d','{}',?,?)", (utcnow(), utcnow()))
                raise RuntimeError("boom")
        assert db.scalar("SELECT COUNT(*) FROM projects") == 0

    def test_commits_on_success(self, db):
        with db.write() as c:
            c.execute(
                "INSERT INTO projects(slug,name,dataset_root,config_json,created_at,updated_at)"
                " VALUES ('a','A','/d','{}',?,?)", (utcnow(), utcnow()))
        assert db.scalar("SELECT COUNT(*) FROM projects") == 1


class TestResourceLease:
    """The lease table's primary key IS the GPU mutex."""

    def _job(self, db, n=1):
        with db.write() as c:
            for _ in range(n):
                c.execute(
                    "INSERT INTO jobs(kind,state,work_dir,created_at,updated_at)"
                    " VALUES ('train','queued','/tmp',?,?)", (utcnow(), utcnow()))

    def test_second_claim_on_same_gpu_fails(self, db):
        self._job(db, 2)
        with db.write() as c:
            c.execute("INSERT INTO resource_leases(resource,job_id,acquired_at)"
                      " VALUES ('gpu:1',1,?)", (utcnow(),))
        with pytest.raises(sqlite3.IntegrityError):
            with db.write() as c:
                c.execute("INSERT INTO resource_leases(resource,job_id,acquired_at)"
                          " VALUES ('gpu:1',2,?)", (utcnow(),))

    def test_different_gpus_coexist(self, db):
        self._job(db, 2)
        with db.write() as c:
            c.execute("INSERT INTO resource_leases(resource,job_id,acquired_at)"
                      " VALUES ('gpu:0',1,?)", (utcnow(),))
            c.execute("INSERT INTO resource_leases(resource,job_id,acquired_at)"
                      " VALUES ('gpu:1',2,?)", (utcnow(),))
        assert db.scalar("SELECT COUNT(*) FROM resource_leases") == 2

    def test_releasing_the_job_releases_the_lease(self, db):
        self._job(db)
        with db.write() as c:
            c.execute("INSERT INTO resource_leases(resource,job_id,acquired_at)"
                      " VALUES ('gpu:1',1,?)", (utcnow(),))
        with db.write() as c:
            c.execute("DELETE FROM jobs WHERE id=1")
        assert db.scalar("SELECT COUNT(*) FROM resource_leases") == 0

    def test_only_one_thread_wins_a_contended_lease(self, db):
        """Two schedulers must never both start a run on the same GPU."""
        self._job(db, 8)
        won, lost = [], []
        barrier = threading.Barrier(8)

        def claim(job_id: int) -> None:
            barrier.wait()
            try:
                with db.write() as c:
                    c.execute("INSERT INTO resource_leases(resource,job_id,acquired_at)"
                              " VALUES ('gpu:1',?,?)", (job_id, utcnow()))
                won.append(job_id)
            except sqlite3.IntegrityError:
                lost.append(job_id)

        threads = [threading.Thread(target=claim, args=(i,)) for i in range(1, 9)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(won) == 1, f"expected exactly one winner, got {won}"
        assert len(lost) == 7


class TestSnapshotIdentity:
    def test_same_content_and_params_is_one_snapshot(self, db):
        """Re-auditing unchanged data must not create a duplicate snapshot."""
        with db.write() as c:
            c.execute(
                "INSERT INTO projects(slug,name,dataset_root,config_json,created_at,updated_at)"
                " VALUES ('p','P','/d','{}',?,?)", (utcnow(), utcnow()))
        args = ("h1", '{"deep":false}')
        for _ in range(2):
            try:
                with db.write() as c:
                    c.execute(
                        "INSERT INTO dataset_snapshots(project_id,dataset_root,audit_params_json,"
                        "content_hash,quick_hash,artifact_dir,summary_json,last_verified_at,created_at)"
                        " VALUES (1,'/d',?,?,'q','/a','{}',?,?)",
                        (args[1], args[0], utcnow(), utcnow()))
            except sqlite3.IntegrityError:
                pass
        assert db.scalar("SELECT COUNT(*) FROM dataset_snapshots") == 1


class TestJsonHelpers:
    def test_round_trip(self):
        assert loads(dumps({"a": [1, 2]})) == {"a": [1, 2]}

    def test_corrupt_json_returns_fallback(self):
        # A bad value must not 500 a page.
        assert loads("{not json", fallback={}) == {}
        assert loads(None, fallback=[]) == []

    def test_non_serialisable_falls_back_to_str(self, tmp_path):
        # Paths appear in configs and manifests; they must not raise. Compare
        # after a round trip rather than by substring, since JSON escapes the
        # backslashes in a Windows path.
        assert loads(dumps({"p": tmp_path}))["p"] == str(tmp_path)
