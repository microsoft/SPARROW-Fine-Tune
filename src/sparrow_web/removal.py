"""Deleting a project.

The dataset folder is never touched. Everything this app generates lives under
``prepared_root``, ``runs_root`` and ``state_dir``; the dataset root is somewhere
else entirely, and is usually a read-only Windows mount besides.

That separation is enforced here rather than assumed. ``_removable()`` refuses
any path that is not inside one of the three generated roots, so a corrupted
``dir_path`` in the database — or a project whose prepared tree was configured to
sit inside the dataset folder — cannot turn a project deletion into data loss.
The check is on the resolved path, so a symlink cannot be used to escape either.
"""

from __future__ import annotations

import shutil
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from .db import Database
from .settings import Settings

#: A run or job in one of these states owns a live process. Deleting the project
#: out from under it would leave an orphan writing into a directory we just
#: removed, so deletion is refused until it is stopped.
ACTIVE_RUN_STATES = ("queued", "starting", "running", "stopping")
ACTIVE_JOB_STATES = ("queued", "leased", "running")


@dataclass
class RemovalPlan:
    """What deleting this project would remove, and what it would leave alone."""

    project: Dict[str, Any]
    dataset_root: str
    n_snapshots: int = 0
    n_prepared: int = 0
    n_runs: int = 0
    n_models: int = 0
    #: Generated directories, each already verified to sit inside a managed root.
    dirs: List[Path] = field(default_factory=list)
    bytes_on_disk: int = 0
    #: Non-empty when something is still running; deletion is blocked until then.
    blockers: List[str] = field(default_factory=list)

    @property
    def can_delete(self) -> bool:
        return not self.blockers


def _managed_roots(settings: Settings) -> List[Path]:
    roots = []
    for root in (settings.prepared_root, settings.runs_root, settings.state_dir):
        try:
            roots.append(Path(root).resolve())
        except OSError:
            continue
    return roots


def _removable(path: Any, settings: Settings) -> Path | None:
    """The path, if it is a real directory inside a generated root — else None.

    Resolved before comparing: ``prepared/<slug>`` is full of symlinks into the
    dataset, and an unresolved check would be satisfied by a link that points
    straight back out into the source images.
    """
    if not path:
        return None
    try:
        candidate = Path(str(path)).resolve()
    except OSError:
        return None
    if not candidate.is_dir():
        return None
    for root in _managed_roots(settings):
        if candidate == root:
            return None          # never delete the root itself, only trees inside it
        if candidate.is_relative_to(root):
            return candidate
    return None


def _dir_size(path: Path) -> int:
    """Bytes used, counting a symlink as its own size rather than its target's.

    ``prepared/`` mirrors the dataset with symlinks, so following them would
    report tens of gigabytes for a tree that occupies a few hundred megabytes.
    """
    total = 0
    for entry in path.rglob("*"):
        try:
            info = entry.lstat()
        except OSError:
            continue
        # Directories are skipped: their inode is 4096 bytes on ext4 and 0 on
        # Windows, so counting them would report mostly filesystem bookkeeping
        # and would differ by platform for the same tree.
        if stat.S_ISDIR(info.st_mode):
            continue
        total += info.st_size
    return total


def plan(db: Database, settings: Settings, project: Dict[str, Any]) -> RemovalPlan:
    """Describe the deletion without performing any of it."""
    pid = project["id"]
    result = RemovalPlan(project=project, dataset_root=project["dataset_root"])

    def count(sql: str) -> int:
        row = db.one(sql, (pid,))
        return int(row["c"]) if row else 0

    result.n_snapshots = count("SELECT COUNT(*) c FROM dataset_snapshots WHERE project_id=?")
    result.n_prepared = count("SELECT COUNT(*) c FROM prepared_datasets WHERE project_id=?")
    result.n_runs = count("SELECT COUNT(*) c FROM runs WHERE project_id=?")
    result.n_models = count("SELECT COUNT(*) c FROM model_versions WHERE project_id=?")

    for run in db.query(
        "SELECT name, state FROM runs WHERE project_id=? AND state IN "
        f"({','.join('?' * len(ACTIVE_RUN_STATES))})", (pid, *ACTIVE_RUN_STATES)
    ):
        result.blockers.append(f"Training run {run['name']} is {run['state']}.")
    for job in db.query(
        "SELECT kind, state FROM jobs WHERE project_id=? AND state IN "
        f"({','.join('?' * len(ACTIVE_JOB_STATES))})", (pid, *ACTIVE_JOB_STATES)
    ):
        result.blockers.append(f"A {job['kind']} job is {job['state']}.")

    seen = set()
    candidates: List[Any] = [settings.project_dir(project["slug"])]
    candidates += [r["dir_path"] for r in
                   db.query("SELECT dir_path FROM prepared_datasets WHERE project_id=?", (pid,))]
    for run in db.query("SELECT run_dir, save_dir FROM runs WHERE project_id=?", (pid,)):
        candidates += [run["run_dir"], run["save_dir"]]

    for candidate in candidates:
        resolved = _removable(candidate, settings)
        if resolved is None or resolved in seen:
            continue
        # A run's save_dir usually sits inside its run_dir; counting both would
        # double the reported size and delete a path that no longer exists.
        if any(resolved.is_relative_to(other) for other in seen):
            continue
        seen.add(resolved)
        result.dirs.append(resolved)
        result.bytes_on_disk += _dir_size(resolved)

    return result


def delete(db: Database, settings: Settings, project: Dict[str, Any], *,
           remove_files: bool) -> RemovalPlan:
    """Delete the project. Raises ``RuntimeError`` if something is still running.

    The database rows go regardless; ``remove_files`` decides whether the
    generated trees on disk go with them. The dataset folder is untouched either
    way — it is not in ``plan.dirs`` and could not be, given ``_removable()``.
    """
    outcome = plan(db, settings, project)
    if not outcome.can_delete:
        raise RuntimeError("; ".join(outcome.blockers))

    if remove_files:
        for path in outcome.dirs:
            shutil.rmtree(path, ignore_errors=True)

    # After the files, so a failure mid-delete leaves the project visible and
    # retryable rather than orphaning directories nothing references any more.
    with db.write() as conn:
        conn.execute("DELETE FROM projects WHERE id=?", (project["id"],))
    return outcome
