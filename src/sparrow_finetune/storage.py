"""Local filesystem object store. Callers choose keys; images use ``images/<sha256>``."""

from __future__ import annotations

import hashlib
import os
import shutil
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import BinaryIO

_CHUNK = 1024 * 1024
# Stored objects are immutable: content-addressed images or per-attempt artifacts. They and
# their staged hard links (which share the inode) are read-only so no reader can alter them.
READ_ONLY = 0o444


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return  # directory fsync is not supported everywhere
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def make_dirs(path: Path) -> None:
    """``mkdir -p`` that fsyncs the parent of every directory it creates.

    Without it a crash can lose a new directory entry (and the file renamed into it)
    even though the file's data and its own directory were fsynced.
    """
    missing = []
    probe = path
    while not probe.is_dir():
        missing.append(probe)
        if probe.parent == probe:
            break
        probe = probe.parent
    if not missing:
        return
    path.mkdir(parents=True, exist_ok=True)
    for created in reversed(missing):  # top-down: each new entry, then its children
        _fsync_dir(created.parent)


class LocalObjectStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def path(self, key: str) -> Path:
        parts = PurePosixPath(key).parts
        if not parts or key.startswith("/") or any(p in ("", ".", "..") for p in parts):
            raise ValueError(f"invalid object key: {key!r}")
        return self.root.joinpath(*parts)

    def _tmp_for(self, target: Path) -> Path:
        make_dirs(target.parent)
        return target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")

    @staticmethod
    def _commit(tmp: Path, target: Path) -> None:
        """Rename a fully written, fsynced temp file into place and fsync the directory."""
        os.chmod(tmp, READ_ONLY)
        os.replace(tmp, target)
        _fsync_dir(target.parent)

    def put_bytes(self, key: str, data: bytes) -> str:
        target = self.path(key)
        tmp = self._tmp_for(target)
        try:
            with open(tmp, "wb") as fout:
                fout.write(data)
                fout.flush()
                os.fsync(fout.fileno())
            self._commit(tmp, target)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return hashlib.sha256(data).hexdigest()

    def put_file(self, key: str, src: Path | str) -> str:
        target = self.path(key)
        tmp = self._tmp_for(target)
        digest = hashlib.sha256()
        try:
            with open(src, "rb") as fin, open(tmp, "wb") as fout:
                while chunk := fin.read(_CHUNK):
                    digest.update(chunk)
                    fout.write(chunk)
                fout.flush()
                os.fsync(fout.fileno())
            self._commit(tmp, target)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return digest.hexdigest()

    def remove_stale_tmp(self, max_age_seconds: float = 3600) -> int:
        """Delete ``.*.tmp`` files left by interrupted writes; return how many were removed.

        Files younger than ``max_age_seconds`` are kept: another process may be writing them.
        """
        if not self.root.is_dir():
            return 0
        cutoff = time.time() - max_age_seconds
        removed = 0
        for path in self.root.rglob(".*.tmp"):
            try:
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except FileNotFoundError:
                pass
        return removed

    def open(self, key: str) -> BinaryIO:
        return open(self.path(key), "rb")

    def exists(self, key: str) -> bool:
        return self.path(key).is_file()

    def delete(self, key: str) -> None:
        """Remove the object if present."""
        self.path(key).unlink(missing_ok=True)

    def link_to(self, key: str, dest: Path) -> None:
        """Expose the object at ``dest``: hard link when possible, else copy; atomic and idempotent.

        ``dest`` is read-only (mode 0o444) like the object: a hard link shares its bytes.
        """
        for attempt in range(3):
            if dest.is_file():
                return
            tmp = self._tmp_for(dest)
            try:
                try:
                    os.link(self.path(key), tmp)
                    # The link shares the object's (possibly old) mtime; refresh it so a
                    # concurrent remove_stale_tmp sweep treats the tmp as in flight.
                    os.utime(tmp)
                except OSError:
                    shutil.copyfile(self.path(key), tmp)
                # Also covers objects stored before they were made read-only, and copies.
                if os.stat(tmp).st_mode & 0o222:
                    os.chmod(tmp, READ_ONLY)
                os.replace(tmp, dest)
                return
            except FileNotFoundError:
                tmp.unlink(missing_ok=True)
                # A sweep removed the tmp before the utime: retry with a fresh one.
                if attempt == 2 or not self.path(key).is_file():
                    raise
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise

    def copy_to(self, key: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.path(key), dest)
