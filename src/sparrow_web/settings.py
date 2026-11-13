"""Environment-driven settings.

Paths default to the container layout. Every one is overridable so the app can
run outside Docker for development and tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


def _path(env: str, default: str) -> Path:
    return Path(os.environ.get(env, default))


def _int(env: str, default: int) -> int:
    try:
        return int(os.environ.get(env, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    #: SQLite plus per-project artifacts (snapshot CSVs, job working dirs).
    #: Must NOT live on a Windows drive: SQLite locking over drvfs is unreliable.
    state_dir: Path = field(default_factory=lambda: _path("SPARROW_STATE_DIR", "/workspace/state"))
    prepared_root: Path = field(
        default_factory=lambda: _path("SPARROW_PREPARED_ROOT", "/workspace/prepared"))
    runs_root: Path = field(default_factory=lambda: _path("SPARROW_RUNS_ROOT", "/workspace/runs"))

    #: Annotation batches: copied images plus the XML written for them. Owned by
    #: the workbench, like prepared/ and runs/, and never a dataset root.
    annotations_root: Path = field(
        default_factory=lambda: _path("SPARROW_ANNOTATIONS_ROOT", "/workspace/annotations"))

    #: Dataset roots a project may point at. An allowlist because this app runs
    #: as root with no auth and will happily audit whatever path it is given.
    dataset_search_roots: List[Path] = field(
        default_factory=lambda: [
            Path(p) for p in os.environ.get(
                "SPARROW_DATASET_SEARCH_ROOTS", "/workspace/datasets"
            ).split(os.pathsep) if p
        ]
    )

    #: Where a finished annotation batch may be exported to. Mounted read-write
    #: under their own prefix, so the capability is visible in docker-compose.yml
    #: rather than implied. Empty by default: with no writable mount configured,
    #: export is simply unavailable rather than silently writing somewhere odd.
    export_roots: List[Path] = field(
        default_factory=lambda: [
            Path(p) for p in os.environ.get("SPARROW_EXPORT_ROOTS", "").split(os.pathsep) if p
        ]
    )

    #: GPU policy. Multi-GPU DDP does not work under WSL2, so these are separate
    #: single-GPU slots rather than one pooled resource: training gets the card
    #: with no display attached, and evaluation/export get the other one so they
    #: never queue behind a two-day run.
    train_gpu: int = field(default_factory=lambda: _int("SPARROW_TRAIN_GPU", 1))
    infer_gpu: int = field(default_factory=lambda: _int("SPARROW_INFER_GPU", 0))

    scheduler_interval_s: float = 2.0
    gpu_poll_interval_s: float = 2.0
    run_poll_interval_s: float = 2.0

    #: A partly-tagged dataset emits one issue per untagged image. Store a
    #: sample; the exact count lives in snapshot_issue_counts and the full list
    #: in the snapshot's artifact CSV.
    issue_sample_cap: int = field(default_factory=lambda: _int("SPARROW_ISSUE_SAMPLE_CAP", 500))

    #: Guards the gthread pool against a pile of abandoned browser tabs.
    max_sse_clients: int = field(default_factory=lambda: _int("SPARROW_MAX_SSE_CLIENTS", 16))

    @property
    def db_path(self) -> Path:
        return self.state_dir / "sparrow.db"

    def project_dir(self, slug: str) -> Path:
        return self.state_dir / "projects" / slug

    def snapshot_dir(self, slug: str, snapshot_id: int) -> Path:
        return self.project_dir(slug) / "snapshots" / str(snapshot_id)

    def job_dir(self, job_id: int) -> Path:
        return self.state_dir / "jobs" / str(job_id)

    def batch_dir(self, slug: str, batch_id: int) -> Path:
        return self.annotations_root / slug / str(batch_id)

    @staticmethod
    def _within(path: str | Path, roots: List[Path]) -> bool:
        try:
            resolved = Path(path).resolve()
        except OSError:
            return False
        for root in roots:
            try:
                root = root.resolve()
            except OSError:
                continue
            if resolved == root or resolved.is_relative_to(root):
                return True
        return False

    def is_allowed_dataset_root(self, path: str | Path) -> bool:
        """Whether a dataset root is inside the read-only allowlist.

        No auth means anyone reaching the port can create a project; without this
        they could point one at ``/`` and have the app walk the filesystem.
        """
        return self._within(path, self.dataset_search_roots)

    def is_allowed_export_root(self, path: str | Path) -> bool:
        """Whether annotations may be written here.

        Deliberately a *separate* allowlist from the browsable dataset roots,
        mounted read-write under its own prefix. Scanning a dataset must never be
        able to write to it; exporting an annotation batch is the one operation
        that writes outside the workbench's own directories, and it should take
        an explicit, differently-mounted path to do so.
        """
        return self._within(path, self.export_roots)


def load_settings() -> Settings:
    return Settings()
