"""Metric ingestion from a run we didn't start.

Built against the real header from a completed run on the training machine:

    epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,
    metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),
    val/box_loss,val/cls_loss,val/dfl_loss,lr/pg0,lr/pg1,lr/pg2
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from sparrow_web.db import Database, utcnow
from sparrow_web.ingest import RunIngestor, normalise_key, parse_results_csv, tail_bytes

HEADER = ("epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,metrics/precision(B),"
          "metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,"
          "val/dfl_loss,lr/pg0,lr/pg1,lr/pg2")
ROW1 = ("1,87.467,0.65835,3.00039,1.1045,0.86553,0.09762,0.1227,0.09197,0.94421,5.06349,"
        "1.42447,8.09442e-05,8.09442e-05,8.09442e-05")
ROW2 = ("2,147.966,0.62576,1.46613,1.09414,0.71176,0.14069,0.11853,0.08924,0.88001,4.64926,"
        "1.37982,8.19501e-05,8.19501e-05,8.19501e-05")


@pytest.fixture
def ingestor(tmp_path):
    db = Database(tmp_path / "t.db")
    db.migrate()
    with db.write() as c:
        c.execute("INSERT INTO projects(slug,name,dataset_root,config_json,created_at,updated_at)"
                  " VALUES ('p','P','/d','{}',?,?)", (utcnow(), utcnow()))
        c.execute("INSERT INTO runs(project_id,name,run_dir,state,queued_at,updated_at)"
                  " VALUES (1,'r',?,'running',?,?)", (str(tmp_path / "run"), utcnow(), utcnow()))
    return RunIngestor(db)


class TestNormaliseKey:
    @pytest.mark.parametrize("raw,expected", [
        ("metrics/mAP50-95(B)", "map50_95"),
        ("metrics/mAP50(B)", "map50"),
        ("metrics/precision(B)", "precision"),
        ("metrics/recall(B)", "recall"),
        ("train/box_loss", "train_box_loss"),
        ("val/cls_loss", "val_cls_loss"),
        ("lr/pg0", "lr"),
        ("time", "time"),
    ])
    def test_known_keys(self, raw, expected):
        assert normalise_key(raw) == expected

    def test_unknown_key_is_cleaned_not_dropped(self):
        """A new Ultralytics column must survive, just tidied."""
        assert normalise_key("metrics/newThing(X)") == "metrics_newthingx"


class TestParseResultsCsv:
    def test_parses_header_and_rows(self):
        header, rows = parse_results_csv(f"{HEADER}\n{ROW1}\n{ROW2}\n")
        assert len(rows) == 2
        assert rows[0]["epoch"] == 1
        assert rows[0]["map50"] == pytest.approx(0.1227)
        assert rows[1]["map50_95"] == pytest.approx(0.08924)
        assert rows[0]["time"] == pytest.approx(87.467)

    def test_continues_without_a_header_when_resuming(self):
        header, _ = parse_results_csv(f"{HEADER}\n")
        _, rows = parse_results_csv(f"{ROW2}\n", header)
        assert rows[0]["epoch"] == 2

    def test_repeated_header_is_ignored(self):
        _, rows = parse_results_csv(f"{HEADER}\n{ROW1}\n{HEADER}\n{ROW2}\n")
        assert [r["epoch"] for r in rows] == [1, 2]

    def test_blank_lines_are_skipped(self):
        _, rows = parse_results_csv(f"{HEADER}\n\n{ROW1}\n\n")
        assert len(rows) == 1


class TestIngestMetrics:
    def _csv(self, tmp_path) -> Path:
        p = tmp_path / "results.csv"
        p.write_text(f"{HEADER}\n{ROW1}\n", encoding="utf-8")
        return p

    def test_first_read_ingests_everything(self, ingestor, tmp_path):
        rows = ingestor.ingest_metrics(1, self._csv(tmp_path))
        assert len(rows) == 1
        series = ingestor.series(1, ["map50"])
        assert series["map50"] == [[1, pytest.approx(0.1227)]]

    def test_second_read_returns_only_new_epochs(self, ingestor, tmp_path):
        csv_path = self._csv(tmp_path)
        ingestor.ingest_metrics(1, csv_path)

        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(ROW2 + "\n")
        rows = ingestor.ingest_metrics(1, csv_path)
        assert [r["epoch"] for r in rows] == [2], "re-read an epoch it had already seen"
        assert len(ingestor.series(1, ["map50"])["map50"]) == 2

    def test_no_change_is_a_no_op(self, ingestor, tmp_path):
        csv_path = self._csv(tmp_path)
        ingestor.ingest_metrics(1, csv_path)
        assert ingestor.ingest_metrics(1, csv_path) == []

    def test_partial_trailing_line_is_not_consumed(self, ingestor, tmp_path):
        """The trainer appends while we read; half a row must not be parsed."""
        csv_path = tmp_path / "results.csv"
        csv_path.write_text(f"{HEADER}\n{ROW1}\n2,147.9,0.62", encoding="utf-8")
        rows = ingestor.ingest_metrics(1, csv_path)
        assert [r["epoch"] for r in rows] == [1]

        with open(csv_path, "a", encoding="utf-8") as f:
            f.write("576,1.09414,0.71176,0.14069,0.11853,0.08924,0.88,4.64,1.37,1e-4,1e-4,1e-4\n")
        rows = ingestor.ingest_metrics(1, csv_path)
        assert [r["epoch"] for r in rows] == [2]

    def test_rewritten_file_is_re_ingested_not_corrupted(self, ingestor, tmp_path):
        """A resume can rewrite results.csv; the upsert makes that harmless."""
        csv_path = self._csv(tmp_path)
        ingestor.ingest_metrics(1, csv_path)
        csv_path.write_text(f"{HEADER}\n{ROW1}\n", encoding="utf-8")  # same size, reset
        with ingestor.db.write() as c:
            c.execute("UPDATE run_ingest_state SET results_csv_offset=999999 WHERE run_id=1")
        rows = ingestor.ingest_metrics(1, csv_path)
        assert [r["epoch"] for r in rows] == [1]
        assert len(ingestor.series(1, ["map50"])["map50"]) == 1, "duplicated an epoch"

    def test_missing_file_is_quiet(self, ingestor, tmp_path):
        assert ingestor.ingest_metrics(1, tmp_path / "nope.csv") == []


class TestEta:
    def test_withheld_until_three_epochs(self, ingestor, tmp_path):
        p = tmp_path / "results.csv"
        p.write_text(f"{HEADER}\n{ROW1}\n{ROW2}\n", encoding="utf-8")
        ingestor.ingest_metrics(1, p)
        assert ingestor.eta(1, 100) is None, "an ETA from 2 epochs is a guess, not a measurement"

    def test_computed_from_epoch_durations(self, ingestor, tmp_path):
        p = tmp_path / "results.csv"
        rows = "\n".join(
            f"{i},{i * 60.0}," + ",".join(["0.5"] * 13) for i in range(1, 6))
        p.write_text(f"{HEADER}\n{rows}\n", encoding="utf-8")
        ingestor.ingest_metrics(1, p)

        eta = ingestor.eta(1, 100)
        assert eta["seconds_per_epoch"] == pytest.approx(60.0)
        assert eta["epoch"] == 5
        assert eta["seconds_remaining"] == 95 * 60

    def test_no_total_means_no_eta(self, ingestor, tmp_path):
        assert ingestor.eta(1, None) is None


class TestIngestStatus:
    def _status(self, run_dir: Path, *, alive: bool = False, **fields) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "status.json").write_text(json.dumps(fields), encoding="utf-8")
        if alive:
            # read_status reconciles against the pid file; without a live pid it
            # would correctly conclude the process is gone.
            (run_dir / "train.pid").write_text(str(os.getpid()), encoding="utf-8")

    def test_running_state_and_metrics_are_stored(self, ingestor, tmp_path):
        run_dir = tmp_path / "run"
        self._status(run_dir, alive=True, state="running", epoch=41, epochs=100,
                     metrics={"metrics/mAP50(B)": 0.81, "metrics/mAP50-95(B)": 0.51},
                     best_fitness=0.55, save_dir=str(run_dir))
        run = ingestor.db.one("SELECT * FROM runs WHERE id=1")
        ingestor.ingest_status(run)

        row = ingestor.db.one("SELECT * FROM runs WHERE id=1")
        assert row["state"] == "running" and row["epoch"] == 41
        assert json.loads(row["metrics_json"])["map50"] == pytest.approx(0.81)

    def test_dead_process_becomes_died(self, ingestor, tmp_path):
        """An OOM kill never runs the exception handler, so the file still says
        'running'. read_status reconciles against the pid; we must persist that."""
        run_dir = tmp_path / "run"
        self._status(run_dir, state="running", epoch=5, epochs=100)
        (run_dir / "train.pid").write_text("999999", encoding="utf-8")

        run = ingestor.db.one("SELECT * FROM runs WHERE id=1")
        ingestor.ingest_status(run)
        row = ingestor.db.one("SELECT * FROM runs WHERE id=1")
        assert row["state"] == "died"
        assert "memory" in (row["error"] or "").lower()

    def test_early_stop_fields_persist(self, ingestor, tmp_path):
        run_dir = tmp_path / "run"
        self._status(run_dir, state="completed", epochs_completed=43, epochs_requested=100,
                     best_epoch=13, stopped_early=True, finished="2026-01-01 00:00:00")
        run = ingestor.db.one("SELECT * FROM runs WHERE id=1")
        ingestor.ingest_status(run)
        row = ingestor.db.one("SELECT * FROM runs WHERE id=1")
        assert row["state"] == "completed" and row["stopped_early"] == 1
        assert row["best_epoch"] == 13 and row["finished_at"] == "2026-01-01 00:00:00"

    def test_missing_status_leaves_the_row_alone(self, ingestor, tmp_path):
        run = ingestor.db.one("SELECT * FROM runs WHERE id=1")
        ingestor.ingest_status(run)
        assert ingestor.db.one("SELECT * FROM runs WHERE id=1")["state"] == "running"


class TestTailBytes:
    def test_reads_from_offset(self, tmp_path):
        p = tmp_path / "train.log"
        p.write_text("line one\nline two\n", encoding="utf-8")
        first = tail_bytes(p, 0)
        assert first["text"].startswith("line one")
        assert first["eof"] is True

        p.write_text("line one\nline two\nline three\n", encoding="utf-8")
        more = tail_bytes(p, first["next_offset"])
        assert "line three" in more["text"]

    def test_missing_file(self, tmp_path):
        assert tail_bytes(tmp_path / "nope.log")["text"] == ""

    def test_offset_past_end_rewinds(self, tmp_path):
        p = tmp_path / "train.log"
        p.write_text("hello\n", encoding="utf-8")
        assert "hello" in tail_bytes(p, 99999)["text"]
