"""SQLite schema and connection handling.

Design rules, in priority order:

1. **Disk is authoritative; the database is an index over it.**
   ``runs/<run>/status.json``, ``results.csv`` and ``train.pid`` remain the truth
   about a run — they are written by a detached process that knows nothing about
   this database. Delete the database and a rescan of ``runs/`` and ``prepared/``
   rebuilds most of it. Never let the two disagree in a way that matters.

2. **Never store what is expensive and never queried row-wise.** Per-box rows
   (300k+ per snapshot) and the full issue list live on disk as CSV; the database
   holds aggregates and a capped sample. Weights are paths, never blobs.

3. **One writer.** Only the web process writes here. Job subprocesses write their
   own ``status.json`` / ``result.json``; the web process ingests. That avoids
   cross-process lock contention entirely and makes ingestion replayable if the
   web process was down when a job finished.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

#: 2 adds the annotation tables. Every statement in _SCHEMA is CREATE ... IF NOT
#: EXISTS, so adding tables needs no migration step — an existing database picks
#: them up on the next migrate().
SCHEMA_VERSION = 2

#: Terminal job/run states — nothing more will happen to these.
TERMINAL_STATES = {"succeeded", "failed", "cancelled", "died", "completed", "stopped"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

-- ---------------------------------------------------------------- projects --
CREATE TABLE IF NOT EXISTS projects (
    id           INTEGER PRIMARY KEY,
    slug         TEXT NOT NULL UNIQUE,      -- filesystem-safe; names prepared/<slug>/
    name         TEXT NOT NULL,
    description  TEXT NOT NULL DEFAULT '',
    dataset_root TEXT NOT NULL,             -- container path, mounted read-only
    -- The whole ExperimentConfig, validated through ExperimentConfig.from_dict()
    -- on every write so the API cannot store something the CLI would reject.
    config_json  TEXT NOT NULL,
    archived     INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_projects_archived ON projects(archived, updated_at DESC);

-- ------------------------------------------------------- dataset snapshots --
-- rel_path interned once per project: 60k rows total, not 60k per snapshot.
CREATE TABLE IF NOT EXISTS dataset_files (
    id            INTEGER PRIMARY KEY,
    project_id    INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    rel_path      TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    UNIQUE(project_id, rel_path)
);

CREATE TABLE IF NOT EXISTS dataset_snapshots (
    id                 INTEGER PRIMARY KEY,
    project_id         INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    parent_snapshot_id INTEGER REFERENCES dataset_snapshots(id),
    job_id             INTEGER,
    dataset_root       TEXT NOT NULL,       -- captured: the project's root may move later
    audit_params_json  TEXT NOT NULL,       -- normalise/deep/merge/blank-dirs
    -- content_hash is the content-true identity of the dataset. Two snapshots
    -- with the same hash are the same data, whatever the mtimes say.
    content_hash       TEXT NOT NULL,
    quick_hash         TEXT NOT NULL,       -- stat-only; a cheap "might have changed" trigger
    n_images INTEGER NOT NULL DEFAULT 0, n_readable INTEGER NOT NULL DEFAULT 0,
    n_labelled INTEGER NOT NULL DEFAULT 0, n_background INTEGER NOT NULL DEFAULT 0,
    n_boxes INTEGER NOT NULL DEFAULT 0, n_classes INTEGER NOT NULL DEFAULT 0,
    n_issues INTEGER NOT NULL DEFAULT 0,
    blank_empty_xml INTEGER NOT NULL DEFAULT 0,
    blank_confirmed_dir INTEGER NOT NULL DEFAULT 0,
    blank_unknown INTEGER NOT NULL DEFAULT 0,
    artifact_dir       TEXT NOT NULL,       -- images.csv.gz / boxes.csv.gz / issues.csv.gz
    summary_json       TEXT NOT NULL,
    warnings_json      TEXT NOT NULL DEFAULT '[]',
    duration_s         REAL,
    n_audits           INTEGER NOT NULL DEFAULT 1,
    last_verified_at   TEXT NOT NULL,
    created_at         TEXT NOT NULL,
    -- Re-auditing unchanged data bumps n_audits instead of making a duplicate,
    -- so "6 snapshots" means "the data changed 6 times", not "you clicked 40 times".
    UNIQUE(project_id, content_hash, audit_params_json)
);
CREATE INDEX IF NOT EXISTS ix_snapshots_project
    ON dataset_snapshots(project_id, created_at DESC);

CREATE TABLE IF NOT EXISTS snapshot_files (
    snapshot_id INTEGER NOT NULL REFERENCES dataset_snapshots(id) ON DELETE CASCADE,
    file_id     INTEGER NOT NULL REFERENCES dataset_files(id),
    size_bytes  INTEGER,
    mtime_ns    INTEGER,
    content_fp  INTEGER,                    -- blake2b(size|head|tail); survives re-copy
    label_fp    INTEGER,                    -- hash of the sorted boxes; detects re-tagging
    img_w INTEGER, img_h INTEGER,
    n_boxes     INTEGER NOT NULL DEFAULT 0,
    readable    INTEGER NOT NULL DEFAULT 1,
    has_xml     INTEGER NOT NULL DEFAULT 0,
    blank_kind  TEXT NOT NULL DEFAULT 'unknown',
    PRIMARY KEY (snapshot_id, file_id)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS ix_snapshot_files_file ON snapshot_files(file_id, snapshot_id);

CREATE TABLE IF NOT EXISTS snapshot_class_counts (
    snapshot_id INTEGER NOT NULL REFERENCES dataset_snapshots(id) ON DELETE CASCADE,
    class_name  TEXT NOT NULL,
    n_boxes     INTEGER NOT NULL,
    n_images    INTEGER NOT NULL,
    PRIMARY KEY (snapshot_id, class_name)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS snapshot_issue_counts (
    snapshot_id INTEGER NOT NULL REFERENCES dataset_snapshots(id) ON DELETE CASCADE,
    issue_type  TEXT NOT NULL,
    n           INTEGER NOT NULL,
    PRIMARY KEY (snapshot_id, issue_type)
) WITHOUT ROWID;

-- A capped SAMPLE only. A partly-tagged 60k dataset emits one missing_xml issue
-- per untagged image; unbounded that is tens of thousands of rows of noise per
-- snapshot. The exact count always lives in snapshot_issue_counts, and the full
-- list is served from the artifact CSV.
CREATE TABLE IF NOT EXISTS audit_issues (
    id          INTEGER PRIMARY KEY,
    snapshot_id INTEGER NOT NULL REFERENCES dataset_snapshots(id) ON DELETE CASCADE,
    issue_type  TEXT NOT NULL,
    path        TEXT NOT NULL,
    detail      TEXT
);
CREATE INDEX IF NOT EXISTS ix_audit_issues ON audit_issues(snapshot_id, issue_type);

-- ------------------------------------------------------ prepared datasets --
CREATE TABLE IF NOT EXISTS prepared_datasets (
    id                  INTEGER PRIMARY KEY,
    project_id          INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    snapshot_id         INTEGER NOT NULL REFERENCES dataset_snapshots(id),
    job_id              INTEGER,
    -- Each prepare writes its own directory, so preparing v2 never destroys the
    -- tree v1 was trained and tested on. It is symlinks plus small .txt labels,
    -- so keeping several is nearly free.
    dir_path            TEXT NOT NULL UNIQUE,
    data_yaml           TEXT NOT NULL,
    class_map_json      TEXT NOT NULL,
    n_classes           INTEGER NOT NULL,
    split_config_json   TEXT NOT NULL,
    background_policy   TEXT NOT NULL,
    splits_hash         TEXT NOT NULL,
    link_mode_used_json TEXT NOT NULL DEFAULT '{}',
    n_train INTEGER, n_val INTEGER, n_test INTEGER,
    manifest_json       TEXT NOT NULL DEFAULT '{}',
    verify_json         TEXT,
    verify_ok           INTEGER NOT NULL DEFAULT 0,
    warnings_json       TEXT NOT NULL DEFAULT '[]',
    state               TEXT NOT NULL DEFAULT 'ready',   -- ready|superseded|deleted
    created_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_prepared_project
    ON prepared_datasets(project_id, created_at DESC);

-- -------------------------------------------------------------------- runs --
CREATE TABLE IF NOT EXISTS runs (
    id                    INTEGER PRIMARY KEY,
    project_id            INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    prepared_id           INTEGER REFERENCES prepared_datasets(id),
    job_id                INTEGER,
    name                  TEXT NOT NULL,
    run_dir               TEXT NOT NULL UNIQUE,
    save_dir              TEXT,
    kind                  TEXT NOT NULL DEFAULT 'train',   -- train | resume
    resumed_from_run_id   INTEGER REFERENCES runs(id),
    -- coco | model_version | weights. Lineage for "v2 was fine-tuned from v1".
    base_kind             TEXT NOT NULL DEFAULT 'coco',
    base_model_version_id INTEGER,
    base_weights          TEXT NOT NULL DEFAULT '',
    config_json           TEXT NOT NULL DEFAULT '{}',
    train_args_json       TEXT NOT NULL DEFAULT '{}',
    device TEXT, imgsz INTEGER, batch TEXT,
    state                 TEXT NOT NULL,   -- queued|starting|running|stopping|
                                           -- completed|stopped|failed|died|cancelled
    pid                   INTEGER,
    epoch INTEGER, epochs INTEGER,
    epochs_completed INTEGER, best_epoch INTEGER, best_fitness REAL,
    stopped_early         INTEGER NOT NULL DEFAULT 0,
    best_weights TEXT, last_weights TEXT,
    metrics_json          TEXT,            -- latest epoch, for list views
    error                 TEXT,
    queued_at TEXT NOT NULL, started_at TEXT, finished_at TEXT, updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_runs_project ON runs(project_id, queued_at DESC);
CREATE INDEX IF NOT EXISTS ix_runs_state   ON runs(state);

-- Long format, not wide: Ultralytics' results.csv columns vary by task and
-- version, and a wide table would need a migration every time one is added.
-- 100 epochs x ~14 keys = ~1,400 rows per run, which is nothing.
CREATE TABLE IF NOT EXISTS metric_samples (
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    epoch  INTEGER NOT NULL,
    key    TEXT NOT NULL,        -- verbatim CSV header, e.g. 'metrics/mAP50(B)'
    value  REAL,
    PRIMARY KEY (run_id, epoch, key)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS ix_metric_series ON metric_samples(run_id, key, epoch);

-- Byte offsets so results.csv is tailed incrementally, and so a web-app restart
-- resumes exactly where it left off without re-reading or double-counting.
CREATE TABLE IF NOT EXISTS run_ingest_state (
    run_id             INTEGER PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
    results_csv_offset INTEGER NOT NULL DEFAULT 0,
    header_json        TEXT,
    log_offset         INTEGER NOT NULL DEFAULT 0,
    status_mtime_ns    INTEGER NOT NULL DEFAULT 0
);

-- ---------------------------------------------------------- model versions --
CREATE TABLE IF NOT EXISTS model_versions (
    id                      INTEGER PRIMARY KEY,
    project_id              INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    version                 INTEGER NOT NULL,
    parent_model_version_id INTEGER REFERENCES model_versions(id),
    run_id                  INTEGER NOT NULL REFERENCES runs(id),
    prepared_id             INTEGER REFERENCES prepared_datasets(id),
    snapshot_id             INTEGER REFERENCES dataset_snapshots(id),
    name TEXT, notes TEXT NOT NULL DEFAULT '',
    weights_path            TEXT NOT NULL,
    weights_bytes           INTEGER,
    onnx_path               TEXT,
    class_map_json          TEXT NOT NULL,
    n_classes               INTEGER NOT NULL,
    fitness                 REAL,
    status                  TEXT NOT NULL DEFAULT 'ready',  -- ready|promoted|archived
    created_at              TEXT NOT NULL,
    UNIQUE(project_id, version)
);
CREATE INDEX IF NOT EXISTS ix_models_project ON model_versions(project_id, version DESC);

CREATE TABLE IF NOT EXISTS model_evaluations (
    id               INTEGER PRIMARY KEY,
    model_version_id INTEGER NOT NULL REFERENCES model_versions(id) ON DELETE CASCADE,
    prepared_id      INTEGER REFERENCES prepared_datasets(id),
    split            TEXT NOT NULL,
    job_id           INTEGER,
    metrics_json     TEXT NOT NULL,
    map50 REAL, map5095 REAL, precision REAL, recall REAL, fitness REAL,
    created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_evals
    ON model_evaluations(model_version_id, split, created_at DESC);

-- Per-species metrics. The whole point on a camera-trap dataset: overall mAP50
-- can look healthy while a rare species sits at zero.
CREATE TABLE IF NOT EXISTS model_class_metrics (
    evaluation_id INTEGER NOT NULL REFERENCES model_evaluations(id) ON DELETE CASCADE,
    class_id      INTEGER NOT NULL,
    class_name    TEXT NOT NULL,
    ap50 REAL, ap5095 REAL, precision REAL, recall REAL, n_instances INTEGER,
    PRIMARY KEY (evaluation_id, class_id)
) WITHOUT ROWID;

-- -------------------------------------------------------------------- jobs --
CREATE TABLE IF NOT EXISTS jobs (
    id                   INTEGER PRIMARY KEY,
    project_id           INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    kind                 TEXT NOT NULL,     -- audit|prepare|train|evaluate|export|doctor
    state                TEXT NOT NULL,     -- queued|leased|running|succeeded|failed|
                                            -- cancelled|died
    priority             INTEGER NOT NULL DEFAULT 0,
    params_json          TEXT NOT NULL DEFAULT '{}',
    work_dir             TEXT NOT NULL,
    needs_resources_json TEXT NOT NULL DEFAULT '[]',
    pid                  INTEGER,
    -- mtime of /proc/1. After a container restart PIDs begin again from low
    -- numbers, so a stale pid file can match an unrelated process; comparing the
    -- init epoch detects that without consulting pids at all.
    host_epoch           TEXT,
    progress_phase TEXT, progress_current INTEGER, progress_total INTEGER,
    result_json TEXT, error TEXT, traceback TEXT,
    run_id INTEGER, snapshot_id INTEGER, prepared_id INTEGER, model_version_id INTEGER,
    created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT, updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_jobs_queue   ON jobs(state, priority DESC, id);
CREATE INDEX IF NOT EXISTS ix_jobs_project ON jobs(project_id, created_at DESC);

-- The PRIMARY KEY is the mutex. One row per held resource means at most one job
-- can hold 'gpu:1', which is the entire single-GPU queue implementation.
CREATE TABLE IF NOT EXISTS resource_leases (
    resource    TEXT PRIMARY KEY,           -- 'gpu:1' | 'project:3' | 'prepared:7'
    job_id      INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    acquired_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    scope      TEXT NOT NULL,
    key        TEXT NOT NULL,
    value_json TEXT NOT NULL,
    PRIMARY KEY (scope, key)
) WITHOUT ROWID;

-- ------------------------------------------------------------- annotation --
-- AI-assisted tagging. Same principle as everywhere else here: the boxes live
-- in Pascal VOC XML on disk, where they are portable and where the pipeline
-- already knows how to read them. These tables hold identity, state, aggregates
-- and — the reason they exist — the record of what produced what.
CREATE TABLE IF NOT EXISTS annotation_batches (
    id          INTEGER PRIMARY KEY,
    project_id  INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    -- Captured, not a reference: the folder the images came from may move or be
    -- deleted, and the batch must still be able to say where it came from.
    source_path TEXT NOT NULL,
    dir_path    TEXT NOT NULL UNIQUE,
    state       TEXT NOT NULL DEFAULT 'ready',   -- ready|published|deleted
    n_images    INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_batches_project
    ON annotation_batches(project_id, created_at DESC);

-- One row per inference pass. Never updated, so re-running with a lower
-- threshold leaves the earlier attempt legible instead of overwriting it.
CREATE TABLE IF NOT EXISTS annotation_passes (
    id               INTEGER PRIMARY KEY,
    batch_id         INTEGER NOT NULL REFERENCES annotation_batches(id) ON DELETE CASCADE,
    model_version_id INTEGER REFERENCES model_versions(id),
    job_id           INTEGER,
    conf REAL, iou REAL, imgsz INTEGER, max_det INTEGER,
    force_class      TEXT NOT NULL DEFAULT '',
    only_classes_json TEXT NOT NULL DEFAULT '[]',
    only_missing     INTEGER NOT NULL DEFAULT 0,
    n_images INTEGER NOT NULL DEFAULT 0, n_boxes INTEGER NOT NULL DEFAULT 0,
    n_blank  INTEGER NOT NULL DEFAULT 0, n_errors INTEGER NOT NULL DEFAULT 0,
    duration_s       REAL,
    created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_passes_batch ON annotation_passes(batch_id, id DESC);

CREATE TABLE IF NOT EXISTS annotation_images (
    id               INTEGER PRIMARY KEY,
    batch_id         INTEGER NOT NULL REFERENCES annotation_batches(id) ON DELETE CASCADE,
    rel_path         TEXT NOT NULL,
    width INTEGER, height INTEGER,
    state            TEXT NOT NULL DEFAULT 'pending',  -- pending|predicted|reviewed
    n_boxes          INTEGER NOT NULL DEFAULT 0,
    -- Split so "how much of this did a person actually check?" is a query, not
    -- an archaeology exercise across XML files.
    n_ai_boxes       INTEGER NOT NULL DEFAULT 0,
    n_human_boxes    INTEGER NOT NULL DEFAULT 0,
    min_confidence   REAL,
    model_version_id INTEGER REFERENCES model_versions(id),
    reviewed_at      TEXT,
    updated_at       TEXT NOT NULL,
    UNIQUE(batch_id, rel_path)
);
CREATE INDEX IF NOT EXISTS ix_ann_images_state ON annotation_images(batch_id, state);

-- Append-only. Exporting the same batch twice records two rows, because the
-- question later is "what did I hand over, and when", not "where is it now".
CREATE TABLE IF NOT EXISTS annotation_publishes (
    id             INTEGER PRIMARY KEY,
    batch_id       INTEGER NOT NULL REFERENCES annotation_batches(id) ON DELETE CASCADE,
    dest_path      TEXT NOT NULL,
    n_images       INTEGER NOT NULL DEFAULT 0,
    n_xml          INTEGER NOT NULL DEFAULT 0,
    include_images INTEGER NOT NULL DEFAULT 1,
    reviewed_only  INTEGER NOT NULL DEFAULT 1,
    manifest_json  TEXT NOT NULL DEFAULT '{}',
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_publishes_batch
    ON annotation_publishes(batch_id, created_at DESC);
"""


def utcnow() -> str:
    """ISO-8601 UTC. Stored as TEXT, which sorts chronologically as it sorts
    lexicographically — so ORDER BY on a timestamp column is correct."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    """Thread-local SQLite connections with WAL enabled.

    One connection per thread because the web process has several: request
    threads plus the scheduler, the GPU poller and the run watcher. SQLite
    connections are not safe to share across threads.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._write_lock = threading.Lock()

    # -- connections -------------------------------------------------------
    @property
    def conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode = WAL")      # readers never block the writer
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA busy_timeout = 30000")
            self._local.conn = conn
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    @contextmanager
    def write(self) -> Iterator[sqlite3.Connection]:
        """A serialised, immediate write transaction.

        ``BEGIN IMMEDIATE`` takes the write lock up front, so two threads racing
        to claim the same resource lease fail fast on the primary-key conflict
        rather than deadlocking at COMMIT.
        """
        with self._write_lock:
            conn = self.conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except Exception:
                conn.execute("ROLLBACK")
                raise
            else:
                conn.execute("COMMIT")

    # -- schema ------------------------------------------------------------
    def migrate(self) -> int:
        """Create or upgrade the schema. Idempotent.

        ``executescript`` cannot run inside our own transaction — sqlite3 issues
        an implicit COMMIT before executing it, which would leave the enclosing
        ``BEGIN IMMEDIATE`` with nothing to commit. So the DDL runs on its own,
        then the version row is written transactionally.
        """
        with self._write_lock:
            self.conn.executescript(_SCHEMA)

        with self.write() as conn:
            row = conn.execute("SELECT MAX(version) AS v FROM schema_migrations").fetchone()
            current = row["v"] if row and row["v"] is not None else 0
            if current < SCHEMA_VERSION:
                conn.execute(
                    "INSERT OR REPLACE INTO schema_migrations(version, applied_at) VALUES (?,?)",
                    (SCHEMA_VERSION, utcnow()),
                )
        return SCHEMA_VERSION

    # -- helpers -----------------------------------------------------------
    def query(self, sql: str, params: Any = ()) -> List[Dict[str, Any]]:
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def one(self, sql: str, params: Any = ()) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    def scalar(self, sql: str, params: Any = ()) -> Any:
        row = self.conn.execute(sql, params).fetchone()
        return row[0] if row else None


def dumps(value: Any) -> str:
    """JSON for a *_json column. `default=str` so Paths and datetimes survive."""
    return json.dumps(value, default=str)


def loads(text: Optional[str], fallback: Any = None) -> Any:
    """Tolerant read of a *_json column — a corrupt value must not 500 a page."""
    if not text:
        return fallback
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return fallback
