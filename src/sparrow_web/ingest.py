"""Reading the state of a training process we did not start.

Training runs detached and knows nothing about this application. Everything we
show about it is read from three files it writes:

``results.csv``
    One row per epoch, written by Ultralytics. **This is the metrics source**,
    not ``status.json`` — it carries the full history (so a browser reconnecting
    mid-run can be served from the database), more keys, and a cumulative
    ``time`` column, which is the only honest basis for an ETA.

``status.json``
    Written by ``process_train``'s callbacks. The authority on run *state*,
    including the reconciliation that turns a vanished process into ``died`` —
    which is what an out-of-memory kill looks like, since it never runs the
    exception handler.

``train.log``
    Read by byte offset for the log panel. Never streamed over SSE by default:
    Ultralytics is chatty enough to swamp the metric events.

Ingestion is incremental (byte offsets are persisted) and idempotent
(``INSERT OR REPLACE`` on ``(run_id, epoch, key)``), so a web-app restart resumes
exactly where it stopped without re-reading or double-counting.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .db import Database, dumps, loads, utcnow

#: Ultralytics emits keys like ``metrics/mAP50(B)`` and ``train/box_loss``.
#: Those must never reach a screen, so they are normalised on the way in.
METRIC_ALIASES = {
    "metrics/precision(b)": "precision",
    "metrics/recall(b)": "recall",
    "metrics/map50(b)": "map50",
    "metrics/map50-95(b)": "map50_95",
    "train/box_loss": "train_box_loss",
    "train/cls_loss": "train_cls_loss",
    "train/dfl_loss": "train_dfl_loss",
    "val/box_loss": "val_box_loss",
    "val/cls_loss": "val_cls_loss",
    "val/dfl_loss": "val_dfl_loss",
    "lr/pg0": "lr",
    "time": "time",
    "epoch": "epoch",
}

#: What the training screen leads with.
HEADLINE_KEYS = ("map50_95", "map50", "precision", "recall")


def normalise_key(raw: str) -> str:
    """``metrics/mAP50-95(B)`` -> ``map50_95``. Unknown keys pass through cleaned."""
    key = raw.strip()
    alias = METRIC_ALIASES.get(key.lower())
    if alias:
        return alias
    return (key.lower().replace("/", "_").replace("(", "").replace(")", "")
            .replace("-", "_").replace(" ", "_"))


def parse_results_csv(text: str, header: Optional[List[str]] = None
                      ) -> Tuple[List[str], List[Dict[str, float]]]:
    """Parse whole lines of ``results.csv``.

    ``header`` is supplied when resuming from a byte offset, because the header
    line was consumed by an earlier read.
    """
    rows: List[Dict[str, float]] = []
    reader = csv.reader(io.StringIO(text))
    for parts in reader:
        if not parts or not any(p.strip() for p in parts):
            continue
        # A header line reappears if Ultralytics rewrote the file.
        if parts[0].strip().lower() == "epoch":
            header = [p.strip() for p in parts]
            continue
        if header is None:
            header = [p.strip() for p in parts]
            continue
        row: Dict[str, float] = {}
        for name, value in zip(header, parts):
            try:
                row[normalise_key(name)] = float(value)
            except (TypeError, ValueError):
                continue
        if "epoch" in row:
            rows.append(row)
    return header or [], rows


class RunIngestor:
    """Pulls a run's on-disk state into the database."""

    def __init__(self, db: Database):
        self.db = db

    # -- offsets -----------------------------------------------------------
    def _state(self, run_id: int) -> Dict[str, Any]:
        row = self.db.one("SELECT * FROM run_ingest_state WHERE run_id=?", (run_id,))
        if row is None:
            with self.db.write() as conn:
                conn.execute("INSERT OR IGNORE INTO run_ingest_state(run_id) VALUES (?)",
                             (run_id,))
            row = self.db.one("SELECT * FROM run_ingest_state WHERE run_id=?", (run_id,))
        return row or {}

    # -- metrics -----------------------------------------------------------
    def ingest_metrics(self, run_id: int, results_csv: str | Path) -> List[Dict[str, float]]:
        """Read any new epochs from ``results.csv``. Returns just the new rows."""
        path = Path(results_csv)
        if not path.is_file():
            return []

        state = self._state(run_id)
        offset = int(state.get("results_csv_offset") or 0)
        header = loads(state.get("header_json"), None)

        size = path.stat().st_size
        if size < offset:
            # The file shrank: Ultralytics rewrote it, which a resume can do.
            # Re-read from the top; the upsert makes that harmless.
            offset, header = 0, None

        if size == offset:
            return []

        # Binary mode deliberately: the stored offset is a byte offset, and text
        # mode would translate line endings on some platforms, so bytes read and
        # characters returned would drift apart and the offset would slide.
        with open(path, "rb") as f:
            f.seek(offset)
            raw = f.read()

        # Only consume whole lines; the trainer is appending as we read, so a
        # partial trailing line must be left for the next tick.
        cut = raw.rfind(b"\n")
        if cut == -1:
            return []
        consumed = cut + 1
        chunk = raw[:consumed].decode("utf-8", errors="replace")

        header, rows = parse_results_csv(chunk, header)
        if rows:
            with self.db.write() as conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO metric_samples(run_id, epoch, key, value)"
                    " VALUES (?,?,?,?)",
                    [(run_id, int(r["epoch"]), k, v)
                     for r in rows for k, v in r.items() if k != "epoch"],
                )
        with self.db.write() as conn:
            conn.execute(
                "UPDATE run_ingest_state SET results_csv_offset=?, header_json=? WHERE run_id=?",
                (offset + consumed, dumps(header), run_id),
            )
        return rows

    def series(self, run_id: int, keys: Optional[List[str]] = None) -> Dict[str, List[List[float]]]:
        """``{key: [[epoch, value], ...]}`` for charting."""
        if keys:
            marks = ",".join("?" * len(keys))
            rows = self.db.query(
                f"SELECT key, epoch, value FROM metric_samples WHERE run_id=? AND key IN ({marks})"
                " ORDER BY key, epoch", (run_id, *keys))
        else:
            rows = self.db.query(
                "SELECT key, epoch, value FROM metric_samples WHERE run_id=?"
                " ORDER BY key, epoch", (run_id,))
        out: Dict[str, List[List[float]]] = {}
        for r in rows:
            out.setdefault(r["key"], []).append([r["epoch"], r["value"]])
        return out

    # -- state -------------------------------------------------------------
    def ingest_status(self, run: Dict[str, Any]) -> Dict[str, Any]:
        """Reconcile a run row against ``status.json`` and the live process."""
        from sparrow_yolo.process_runner import best_weights, read_status

        run_dir = Path(run["run_dir"])
        status = read_status(run_dir)          # already turns a dead process into 'died'
        state = status.get("state", "unknown")
        if state == "unknown":
            return status

        metrics = status.get("metrics") or {}
        normalised = {normalise_key(k): v for k, v in metrics.items()
                      if isinstance(v, (int, float))}
        best = best_weights(run_dir)

        finished = status.get("finished") if state in {"completed", "stopped", "failed",
                                                       "died"} else None
        with self.db.write() as conn:
            conn.execute(
                "UPDATE runs SET state=?, pid=?, save_dir=?, epoch=?, epochs=?,"
                " epochs_completed=?, best_epoch=?, best_fitness=?, stopped_early=?,"
                " best_weights=?, last_weights=?, metrics_json=?, error=?,"
                " started_at=COALESCE(started_at, ?), finished_at=COALESCE(finished_at, ?),"
                " updated_at=? WHERE id=?",
                (state, status.get("pid"), status.get("save_dir"),
                 status.get("epoch"), status.get("epochs"),
                 status.get("epochs_completed"), status.get("best_epoch"),
                 status.get("best_fitness"), 1 if status.get("stopped_early") else 0,
                 str(best) if best else status.get("best_weights") or None,
                 status.get("last_weights") or None,
                 dumps(normalised) if normalised else None,
                 status.get("error") or status.get("detail"),
                 status.get("started"), finished, utcnow(), run["id"]),
            )
        return status

    # -- progress ----------------------------------------------------------
    def eta(self, run_id: int, epochs_total: Optional[int]) -> Optional[Dict[str, Any]]:
        """Seconds remaining, from the median of recent epoch durations.

        Ultralytics' cumulative ``time`` column makes this a measurement rather
        than a guess. Withheld until three epochs are in, because the first is
        unrepresentative (warmup, caching, AutoBatch).
        """
        rows = self.db.query(
            "SELECT epoch, value FROM metric_samples WHERE run_id=? AND key='time'"
            " ORDER BY epoch", (run_id,))
        if not epochs_total or len(rows) < 3:
            return None

        durations = [rows[i]["value"] - rows[i - 1]["value"] for i in range(1, len(rows))]
        recent = sorted(durations[-10:])
        if not recent:
            return None
        median = recent[len(recent) // 2]
        done = rows[-1]["epoch"]
        remaining = max(0, epochs_total - done)
        return {
            "epoch": done,
            "epochs": epochs_total,
            "seconds_per_epoch": round(median, 1),
            "seconds_remaining": int(median * remaining),
            "elapsed_s": int(rows[-1]["value"]),
        }


def tail_bytes(path: str | Path, offset: int = 0, limit: int = 65536) -> Dict[str, Any]:
    """Byte-range read of a log file, for a polling log panel.

    Binary mode so ``offset`` is a true byte offset the caller can hand straight
    back next poll. Reading in text mode would translate line endings on some
    platforms, and the returned character count would no longer match the bytes
    consumed — the offset would drift a little further every poll.
    """
    p = Path(path)
    if not p.is_file():
        return {"text": "", "offset": 0, "next_offset": 0, "size": 0, "eof": True}
    size = p.stat().st_size
    if offset < 0 or offset > size:
        # Past the end (the file was rotated or truncated): show the tail.
        offset = max(0, size - limit)
    with open(p, "rb") as f:
        f.seek(offset)
        raw = f.read(limit)
    next_offset = offset + len(raw)
    return {
        "text": raw.decode("utf-8", errors="replace"),
        "offset": offset,
        "next_offset": next_offset,
        "size": size,
        "eof": next_offset >= size,
    }
