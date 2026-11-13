"""Domain reads and writes.

Plain SQL rather than an ORM: the interesting queries here are anti-joins and
aggregates over snapshot files, which an ORM makes harder to read, not easier.

Ingestion functions (``ingest_audit_result`` and friends) are called by the
supervisor when a job finishes. They must be **idempotent** — the supervisor may
see the same completed job twice if the web process restarted between the job
finishing and its result being recorded.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from sparrow_yolo.config import ExperimentConfig

from .db import Database, dumps, loads, utcnow


def slugify(name: str) -> str:
    """Filesystem-safe project identifier; also used as a directory name."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return slug or "project"


# ------------------------------------------------------------- projects --

class Projects:
    def __init__(self, db: Database):
        self.db = db

    def list(self, include_archived: bool = False) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM projects"
        if not include_archived:
            sql += " WHERE archived=0"
        return self.db.query(sql + " ORDER BY updated_at DESC")

    def get(self, project_id: int) -> Optional[Dict[str, Any]]:
        return self.db.one("SELECT * FROM projects WHERE id=?", (project_id,))

    def by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        return self.db.one("SELECT * FROM projects WHERE slug=?", (slug,))

    def create(self, name: str, dataset_root: str, *, description: str = "",
               config: Optional[Dict[str, Any]] = None) -> int:
        """Create a project, validating its config through the same class the CLI
        uses — so the UI cannot store settings the CLI would reject."""
        base = slugify(name)
        slug, n = base, 2
        while self.by_slug(slug) is not None:
            slug, n = f"{base}_{n}", n + 1

        raw = dict(config or {})
        raw.setdefault("dataset", {})
        raw["dataset"]["root"] = dataset_root
        raw["dataset"]["name"] = slug
        cfg = ExperimentConfig.from_dict(raw)   # raises ValueError on bad input

        now = utcnow()
        with self.db.write() as conn:
            cur = conn.execute(
                "INSERT INTO projects(slug,name,description,dataset_root,config_json,"
                "created_at,updated_at) VALUES (?,?,?,?,?,?,?)",
                (slug, name, description, dataset_root, dumps(cfg.to_dict()), now, now),
            )
            return int(cur.lastrowid)

    def update_config(self, project_id: int, config: Dict[str, Any]) -> Dict[str, Any]:
        cfg = ExperimentConfig.from_dict(config)
        with self.db.write() as conn:
            conn.execute("UPDATE projects SET config_json=?, updated_at=? WHERE id=?",
                         (dumps(cfg.to_dict()), utcnow(), project_id))
        return cfg.to_dict()

    def config(self, project: Dict[str, Any]) -> ExperimentConfig:
        return ExperimentConfig.from_dict(loads(project["config_json"], {}))

    def delete(self, project_id: int) -> None:
        with self.db.write() as conn:
            conn.execute("DELETE FROM projects WHERE id=?", (project_id,))


# ------------------------------------------------------------ snapshots --

class Snapshots:
    def __init__(self, db: Database):
        self.db = db

    def latest(self, project_id: int) -> Optional[Dict[str, Any]]:
        return self.db.one(
            "SELECT * FROM dataset_snapshots WHERE project_id=?"
            " ORDER BY created_at DESC, id DESC LIMIT 1", (project_id,))

    def get(self, snapshot_id: int) -> Optional[Dict[str, Any]]:
        return self.db.one("SELECT * FROM dataset_snapshots WHERE id=?", (snapshot_id,))

    def list(self, project_id: int, limit: int = 50) -> List[Dict[str, Any]]:
        return self.db.query(
            "SELECT * FROM dataset_snapshots WHERE project_id=?"
            " ORDER BY created_at DESC, id DESC LIMIT ?", (project_id, limit))

    def classes(self, snapshot_id: int) -> List[Dict[str, Any]]:
        return self.db.query(
            "SELECT class_name, n_boxes, n_images FROM snapshot_class_counts"
            " WHERE snapshot_id=? ORDER BY n_boxes DESC", (snapshot_id,))

    def issue_counts(self, snapshot_id: int) -> Dict[str, int]:
        return {r["issue_type"]: r["n"] for r in self.db.query(
            "SELECT issue_type, n FROM snapshot_issue_counts WHERE snapshot_id=?"
            " ORDER BY n DESC", (snapshot_id,))}

    def issues(self, snapshot_id: int, issue_type: Optional[str] = None,
               limit: int = 200, offset: int = 0) -> List[Dict[str, Any]]:
        if issue_type:
            return self.db.query(
                "SELECT * FROM audit_issues WHERE snapshot_id=? AND issue_type=?"
                " ORDER BY id LIMIT ? OFFSET ?", (snapshot_id, issue_type, limit, offset))
        return self.db.query(
            "SELECT * FROM audit_issues WHERE snapshot_id=? ORDER BY id LIMIT ? OFFSET ?",
            (snapshot_id, limit, offset))

    def diff(self, from_id: int, to_id: int) -> Dict[str, Any]:
        """What changed between two snapshots.

        Anti-joins over ``(snapshot_id, file_id)``, which is the primary key, so
        this stays fast at 60,000 files per side.

        ``images_modified`` (the image bytes changed) is kept separate from
        ``images_relabelled`` (same bytes, different boxes) because they mean
        very different things: the second says existing labels were corrected,
        which makes any model trained on them stale.
        """
        added = self.db.one(
            "SELECT COUNT(*) AS n, COALESCE(SUM(b.n_boxes),0) AS boxes"
            " FROM snapshot_files b"
            " LEFT JOIN snapshot_files a ON a.snapshot_id=? AND a.file_id=b.file_id"
            " WHERE b.snapshot_id=? AND a.file_id IS NULL", (from_id, to_id)) or {}
        removed = self.db.one(
            "SELECT COUNT(*) AS n, COALESCE(SUM(a.n_boxes),0) AS boxes"
            " FROM snapshot_files a"
            " LEFT JOIN snapshot_files b ON b.snapshot_id=? AND b.file_id=a.file_id"
            " WHERE a.snapshot_id=? AND b.file_id IS NULL", (to_id, from_id)) or {}
        changed = self.db.one(
            "SELECT"
            "  SUM(CASE WHEN a.content_fp IS NOT NULL AND b.content_fp IS NOT NULL"
            "           AND a.content_fp <> b.content_fp THEN 1 ELSE 0 END) AS modified,"
            "  SUM(CASE WHEN (a.content_fp IS NULL OR b.content_fp IS NULL"
            "                 OR a.content_fp = b.content_fp)"
            "           AND a.label_fp IS NOT NULL AND b.label_fp IS NOT NULL"
            "           AND a.label_fp <> b.label_fp THEN 1 ELSE 0 END) AS relabelled"
            " FROM snapshot_files a JOIN snapshot_files b USING (file_id)"
            " WHERE a.snapshot_id=? AND b.snapshot_id=?", (from_id, to_id)) or {}

        new_classes = self.db.query(
            "SELECT b.class_name, b.n_boxes FROM snapshot_class_counts b"
            " LEFT JOIN snapshot_class_counts a"
            "   ON a.snapshot_id=? AND a.class_name=b.class_name"
            " WHERE b.snapshot_id=? AND a.class_name IS NULL", (from_id, to_id))

        sample = self.db.query(
            "SELECT f.rel_path FROM snapshot_files b"
            " JOIN dataset_files f ON f.id=b.file_id"
            " LEFT JOIN snapshot_files a ON a.snapshot_id=? AND a.file_id=b.file_id"
            " WHERE b.snapshot_id=? AND a.file_id IS NULL LIMIT 20", (from_id, to_id))

        return {
            "from_snapshot_id": from_id,
            "to_snapshot_id": to_id,
            "images_added": int(added.get("n") or 0),
            "images_removed": int(removed.get("n") or 0),
            "images_modified": int(changed.get("modified") or 0),
            "images_relabelled": int(changed.get("relabelled") or 0),
            "boxes_added": int(added.get("boxes") or 0),
            "boxes_removed": int(removed.get("boxes") or 0),
            "new_classes": [r["class_name"] for r in new_classes],
            "sample_added": [r["rel_path"] for r in sample],
            "changed": bool(
                (added.get("n") or 0) or (removed.get("n") or 0)
                or (changed.get("modified") or 0) or (changed.get("relabelled") or 0)
            ),
        }


def ingest_audit_result(db: Database, settings, project: Dict[str, Any],
                        job: Dict[str, Any], result: Dict[str, Any]) -> int:
    """Turn a finished audit job into a snapshot row. Idempotent.

    Re-auditing unchanged data bumps ``n_audits`` rather than creating another
    snapshot, so the snapshot list reads as "the data changed N times" instead of
    "you pressed the button N times".
    """
    import pandas as pd

    from .snapshots import IMAGES_CSV

    project_id = project["id"]
    summary = result.get("summary", {})
    params = loads(job["params_json"], {})
    audit_params = dumps({
        "normalise_class_names": params.get("normalise_class_names", True),
        "deep_image_check": params.get("deep_image_check", False),
        "confirmed_blank_dirs": sorted(params.get("confirmed_blank_dirs") or []),
        "merge_classes": params.get("merge_classes") or {},
    })
    content_hash = result.get("content_hash", "")

    existing = db.one(
        "SELECT id FROM dataset_snapshots WHERE project_id=? AND content_hash=?"
        " AND audit_params_json=?", (project_id, content_hash, audit_params))
    if existing:
        with db.write() as conn:
            conn.execute(
                "UPDATE dataset_snapshots SET n_audits=n_audits+1, last_verified_at=?"
                " WHERE id=?", (utcnow(), existing["id"]))
        return int(existing["id"])

    parent = db.one(
        "SELECT id FROM dataset_snapshots WHERE project_id=?"
        " ORDER BY created_at DESC, id DESC LIMIT 1", (project_id,))
    now = utcnow()

    with db.write() as conn:
        cur = conn.execute(
            "INSERT INTO dataset_snapshots(project_id,parent_snapshot_id,job_id,dataset_root,"
            "audit_params_json,content_hash,quick_hash,n_images,n_readable,n_labelled,"
            "n_background,n_boxes,n_classes,n_issues,blank_empty_xml,blank_confirmed_dir,"
            "blank_unknown,artifact_dir,summary_json,warnings_json,duration_s,"
            "last_verified_at,created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (project_id, parent["id"] if parent else None, job["id"], project["dataset_root"],
             audit_params, content_hash, result.get("quick_hash", ""),
             summary.get("num_images", 0), summary.get("readable_images", 0),
             summary.get("labelled_images", 0), summary.get("background_images", 0),
             summary.get("num_boxes", 0), summary.get("num_classes", 0),
             summary.get("num_issues", 0), summary.get("blank_with_empty_xml", 0),
             summary.get("blank_confirmed_dir", 0), summary.get("blank_unknown", 0),
             result.get("artifact_dir", ""), dumps(summary),
             dumps(result.get("warnings", [])), result.get("duration_s"), now, now),
        )
        snapshot_id = int(cur.lastrowid)

        for name, counts in (result.get("class_counts") or {}).items():
            conn.execute(
                "INSERT OR REPLACE INTO snapshot_class_counts(snapshot_id,class_name,"
                "n_boxes,n_images) VALUES (?,?,?,?)",
                (snapshot_id, name, counts.get("n_boxes", 0), counts.get("n_images", 0)))

        for issue_type, n in (result.get("issue_counts") or {}).items():
            conn.execute(
                "INSERT OR REPLACE INTO snapshot_issue_counts(snapshot_id,issue_type,n)"
                " VALUES (?,?,?)", (snapshot_id, issue_type, int(n)))

    _ingest_snapshot_files(db, settings, project, snapshot_id, result, IMAGES_CSV, pd)
    return snapshot_id


def _ingest_snapshot_files(db, settings, project, snapshot_id, result, images_csv, pd) -> None:
    """Per-file rows, which is what makes the diff a SQL anti-join.

    Also stores a capped sample of issues: a partly-tagged 60,000-image dataset
    emits one ``missing_xml`` per untagged image, and storing all of them would
    be tens of thousands of rows of noise per snapshot. The exact counts live in
    ``snapshot_issue_counts``, and the full list stays in the artifact CSV.
    """
    artifact_dir = Path(result.get("artifact_dir", ""))
    images_path = artifact_dir / images_csv
    if not images_path.is_file():
        return

    images = pd.read_csv(images_path, compression="gzip")
    labels = result.get("label_fingerprints") or {}
    project_id = project["id"]
    now = utcnow()

    with db.write() as conn:
        for row in images.itertuples(index=False):
            rel = str(getattr(row, "rel_path", "") or "")
            if not rel:
                continue
            conn.execute(
                "INSERT OR IGNORE INTO dataset_files(project_id, rel_path, first_seen_at)"
                " VALUES (?,?,?)", (project_id, rel, now))

        file_ids = {r["rel_path"]: r["id"] for r in db.query(
            "SELECT id, rel_path FROM dataset_files WHERE project_id=?", (project_id,))}

        def val(row, name, default=None):
            v = getattr(row, name, default)
            return None if (v is None or (isinstance(v, float) and v != v)) else v

        conn.executemany(
            "INSERT OR REPLACE INTO snapshot_files(snapshot_id,file_id,size_bytes,mtime_ns,"
            "content_fp,label_fp,img_w,img_h,n_boxes,readable,has_xml,blank_kind)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (snapshot_id, file_ids[str(row.rel_path)],
                 val(row, "size_bytes"), val(row, "mtime_ns"), val(row, "content_fp"),
                 labels.get(str(row.rel_path)),
                 val(row, "img_width"), val(row, "img_height"),
                 int(val(row, "n_boxes", 0) or 0),
                 1 if val(row, "readable", True) else 0,
                 1 if val(row, "has_xml", False) else 0,
                 str(val(row, "blank_kind", "unknown") or "unknown"))
                for row in images.itertuples(index=False)
                if str(getattr(row, "rel_path", "")) in file_ids
            ],
        )

    issues_path = artifact_dir / "issues.csv.gz"
    if issues_path.is_file():
        issues = pd.read_csv(issues_path, compression="gzip")
        if not issues.empty:
            cap = settings.issue_sample_cap
            sample = issues.groupby("type", group_keys=False).head(cap)
            with db.write() as conn:
                conn.executemany(
                    "INSERT INTO audit_issues(snapshot_id,issue_type,path,detail)"
                    " VALUES (?,?,?,?)",
                    [(snapshot_id, str(r.type), str(r.path), str(getattr(r, "detail", "")))
                     for r in sample.itertuples(index=False)],
                )


def ingest_prepare_result(db: Database, project: Dict[str, Any], job: Dict[str, Any],
                          result: Dict[str, Any]) -> int:
    """Record a prepared dataset and retire the previous one."""
    params = loads(job["params_json"], {})
    class_map = result.get("class_map", {})
    now = utcnow()

    with db.write() as conn:
        conn.execute(
            "UPDATE prepared_datasets SET state='superseded'"
            " WHERE project_id=? AND state='ready'", (project["id"],))
        # The tree on disk is authoritative and has just been rewritten, so an
        # existing row for this path describes something that no longer exists.
        conn.execute("DELETE FROM prepared_datasets WHERE dir_path=?",
                     (result["dir_path"],))
        cur = conn.execute(
            "INSERT INTO prepared_datasets(project_id,snapshot_id,job_id,dir_path,data_yaml,"
            "class_map_json,n_classes,split_config_json,background_policy,splits_hash,"
            "link_mode_used_json,n_train,n_val,n_test,manifest_json,verify_json,verify_ok,"
            "warnings_json,state,created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'ready',?)",
            (project["id"], job["snapshot_id"], job["id"], result["dir_path"],
             result["data_yaml"], dumps(class_map), len(class_map),
             dumps({k: params.get(k) for k in
                    ("train", "val", "test", "seed", "stratify", "group_by")}),
             params.get("background_policy", "reviewed"),
             result.get("manifest", {}).get("splits_hash", ""),
             dumps(result.get("manifest", {}).get("link_mode_used", {})),
             result.get("n_train", 0), result.get("n_val", 0), result.get("n_test", 0),
             dumps(result.get("manifest", {})), dumps(result.get("verify", {})),
             1 if result.get("verify_ok") else 0, dumps(result.get("warnings", [])), now),
        )
        return int(cur.lastrowid)


class Runs:
    def __init__(self, db: Database):
        self.db = db

    def get(self, run_id: int) -> Optional[Dict[str, Any]]:
        return self.db.one("SELECT * FROM runs WHERE id=?", (run_id,))

    def list(self, project_id: int, limit: int = 50) -> List[Dict[str, Any]]:
        return self.db.query(
            "SELECT * FROM runs WHERE project_id=? ORDER BY id DESC LIMIT ?",
            (project_id, limit))

    def active(self) -> List[Dict[str, Any]]:
        return self.db.query(
            "SELECT * FROM runs WHERE state IN ('queued','starting','running','stopping')")

    def latest_completed(self, project_id: int) -> Optional[Dict[str, Any]]:
        return self.db.one(
            "SELECT * FROM runs WHERE project_id=? AND state='completed'"
            " ORDER BY id DESC LIMIT 1", (project_id,))


class Models:
    def __init__(self, db: Database):
        self.db = db

    def list(self, project_id: int) -> List[Dict[str, Any]]:
        return self.db.query(
            "SELECT * FROM model_versions WHERE project_id=? ORDER BY version DESC",
            (project_id,))

    def get(self, model_id: int) -> Optional[Dict[str, Any]]:
        return self.db.one("SELECT * FROM model_versions WHERE id=?", (model_id,))

    def latest(self, project_id: int) -> Optional[Dict[str, Any]]:
        return self.db.one(
            "SELECT * FROM model_versions WHERE project_id=? ORDER BY version DESC LIMIT 1",
            (project_id,))

    def create_from_run(self, project_id: int, run: Dict[str, Any], *,
                        weights_path: str, class_map: Dict[str, int],
                        notes: str = "") -> int:
        """Register a trained checkpoint as the next version.

        ``parent_model_version_id`` records the fine-tuning chain, so the UI can
        show ``v3 <- v2 <- v1 <- COCO`` rather than a flat list of runs.
        """
        previous = self.latest(project_id)
        version = (previous["version"] + 1) if previous else 1
        parent = run.get("base_model_version_id") or (previous["id"] if previous else None)

        size = None
        try:
            size = Path(weights_path).stat().st_size
        except OSError:
            pass

        with self.db.write() as conn:
            cur = conn.execute(
                "INSERT INTO model_versions(project_id,version,parent_model_version_id,run_id,"
                "prepared_id,snapshot_id,weights_path,weights_bytes,class_map_json,n_classes,"
                "fitness,notes,created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (project_id, version, parent, run["id"], run.get("prepared_id"),
                 None, weights_path, size, dumps(class_map), len(class_map),
                 run.get("best_fitness"), notes, utcnow()),
            )
            return int(cur.lastrowid)

    def lineage(self, model_id: int) -> List[Dict[str, Any]]:
        """The fine-tuning chain, oldest first, ending at this version."""
        chain: List[Dict[str, Any]] = []
        seen = set()
        current = self.get(model_id)
        while current and current["id"] not in seen:
            seen.add(current["id"])
            chain.append(current)
            parent_id = current.get("parent_model_version_id")
            current = self.get(parent_id) if parent_id else None
        return list(reversed(chain))
