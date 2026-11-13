"""Batch/worker resolution, config validation, and run supervision.

None of these need a GPU: they cover the argument plumbing that decides whether a
run starts correctly, which is where the previous version failed.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest
import yaml

from sparrow_yolo.config import ExperimentConfig
from sparrow_yolo.hardware import GpuInfo, parse_device, suggest_batch, suggest_workers
from sparrow_yolo.process_runner import (
    best_weights,
    list_runs,
    read_status,
    resolve_batch,
    resolve_model_weights,
    resolve_resume,
    resolve_workers,
    unique_run_dir,
    write_train_config,
)

TWO_32GB = [
    GpuInfo(0, "NVIDIA RTX PRO 4500 Blackwell", 32623, 1197, display_attached=True),
    GpuInfo(1, "NVIDIA RTX PRO 4500 Blackwell", 32623, 0),
]


class TestParseDevice:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("0", [0]), ("0,1", [0, 1]), ("0, 1", [0, 1]), (0, [0]),
            ([0, 1], [0, 1]), ("cuda:1", [1]), ("cpu", []), ("", []), (None, []),
        ],
    )
    def test_parses(self, value, expected):
        assert parse_device(value) == expected


class TestSuggestBatch:
    def test_scales_with_gpu_count(self):
        one = suggest_batch("yolo11l.pt", 1024, [0], TWO_32GB)
        two = suggest_batch("yolo11l.pt", 1024, [0, 1], TWO_32GB)
        assert two == 2 * one

    def test_larger_model_gets_smaller_batch(self):
        assert (suggest_batch("yolo11x.pt", 1024, [0], TWO_32GB)
                < suggest_batch("yolo11n.pt", 1024, [0], TWO_32GB))

    def test_larger_imgsz_gets_smaller_batch(self):
        assert (suggest_batch("yolo11l.pt", 1536, [0], TWO_32GB)
                < suggest_batch("yolo11l.pt", 640, [0], TWO_32GB))

    def test_sized_from_tightest_gpu(self):
        """GPU 0 drives the display, so it has less free memory than GPU 1.

        Sizing from the average would OOM rank 0 under DDP.
        """
        busy = [GpuInfo(0, "gpu", 32623, 20000), GpuInfo(1, "gpu", 32623, 0)]
        assert (suggest_batch("yolo11l.pt", 1024, [0, 1], busy)
                < suggest_batch("yolo11l.pt", 1024, [0, 1], TWO_32GB))

    def test_always_at_least_one_per_gpu(self):
        tiny = [GpuInfo(0, "gpu", 4096, 3900)]
        assert suggest_batch("yolo11x.pt", 1536, [0], tiny) >= 1

    def test_result_is_even_for_multi_gpu(self):
        b = suggest_batch("yolo11l.pt", 1024, [0, 1], TWO_32GB)
        assert b % 2 == 0

    def test_scale_parsed_from_unknown_release(self):
        """A release newer than the lookup table must still size by its scale letter.

        Otherwise yolo26m falls back to the 'l' default and wastes a third of the
        batch.
        """
        from sparrow_yolo.hardware import per_image_mb

        assert per_image_mb("yolo26m.pt")[0] == per_image_mb("yolo11m.pt")[0]
        assert per_image_mb("yolo26n.pt")[0] < per_image_mb("yolo26x.pt")[0]
        assert "parsed" in per_image_mb("yolo26m.pt")[1]

    def test_custom_checkpoint_falls_back_with_explanation(self):
        from sparrow_yolo.hardware import per_image_mb

        mb, why = per_image_mb("sparrow_africa_v1.pt")
        # Pinned to the 'l' entry rather than a literal, so recalibrating the
        # table does not fail a test that is really about the fallback path.
        assert mb == per_image_mb("yolo11l.pt")[0]
        assert "unknown architecture" in why

    def test_matches_what_the_hardware_actually_did(self):
        """The calibration points, from real runs of yolo11l at 1024 on a 32 GiB
        card: 9.74 GiB reserved at batch 4, 15.9 GiB at batch 10. So ~5.6 GiB
        fixed plus ~1.03 GiB per image, and the suggestion must fit inside the
        card with the headroom it claims."""
        free_gb = TWO_32GB[1].free_gb
        per_gpu = suggest_batch("yolo11l.pt", 1024, [1], TWO_32GB)
        predicted_gb = 5.6 + 1.03 * per_gpu
        assert predicted_gb <= free_gb * 0.85, (
            f"batch {per_gpu} would want {predicted_gb:.1f} GiB of {free_gb:.1f}")
        # And it must not be so timid that it wastes the card. Ultralytics'
        # AutoBatch picked 4 here, which is what prompted the recalibration.
        assert predicted_gb >= free_gb * 0.60

    def test_plausible_for_target_hardware(self):
        """Sanity band for 2x32 GiB at 1024px - not a fit, just a guard."""
        b = suggest_batch("yolo11l.pt", 1024, [0, 1], TWO_32GB)
        assert 16 <= b <= 48


class TestResolveBatch:
    """These pin the platform explicitly rather than inheriting it: the suite is
    run both on a Windows host and inside the WSL2 container, and the answer
    legitimately differs between them."""

    @pytest.fixture
    def native(self, monkeypatch):
        monkeypatch.setattr("sparrow_yolo.process_runner.is_wsl", lambda: False)

    @pytest.fixture
    def wsl(self, monkeypatch):
        monkeypatch.setattr("sparrow_yolo.process_runner.is_wsl", lambda: True)

    def test_auto_single_gpu_uses_ultralytics_autobatch(self, native):
        assert resolve_batch("auto", model_weights="yolo11l.pt", imgsz=1024, device="0") == -1

    def test_auto_single_gpu_on_wsl_does_not(self, wsl):
        """WDDM hides per-process device memory, so AutoBatch profiles against
        numbers that are not real. Measured on the target machine: a 32 GiB card
        reported 0.25G in use and AutoBatch chose batch-size 4 — roughly a
        quarter of what fits, for four times the steps per epoch."""
        b = resolve_batch("auto", model_weights="yolo11l.pt", imgsz=1024, device="1")
        assert isinstance(b, int) and b > 0
        assert b != -1

    def test_explicit_minus_one_on_wsl_is_replaced(self, wsl):
        b = resolve_batch(-1, model_weights="yolo11l.pt", imgsz=1024, device="1")
        assert isinstance(b, int) and b > 0

    def test_an_explicit_batch_is_never_second_guessed(self, wsl):
        """Overriding the heuristic is the point of typing a number."""
        assert resolve_batch(24, model_weights="yolo11l.pt", imgsz=1024, device="1") == 24

    def test_a_fraction_still_survives_wsl(self, wsl):
        b = resolve_batch(0.7, model_weights="yolo11l.pt", imgsz=1024, device="1")
        assert isinstance(b, float) and b == 0.7

    def test_auto_multi_gpu_returns_concrete_batch(self):
        """AutoBatch is single-GPU only.

        Left as -1 under DDP, Ultralytics silently substitutes a fixed 16 - which
        is exactly the batch recorded in this repo's failed `device: 0,1` run.
        """
        b = resolve_batch("auto", model_weights="yolo11l.pt", imgsz=1024, device="0,1")
        assert isinstance(b, int) and b > 0

    def test_explicit_minus_one_multi_gpu_is_replaced(self):
        b = resolve_batch(-1, model_weights="yolo11l.pt", imgsz=1024, device="0,1")
        assert isinstance(b, int) and b > 0

    def test_explicit_minus_one_single_gpu_preserved(self, native):
        assert resolve_batch(-1, model_weights="yolo11l.pt", imgsz=1024, device="0") == -1

    def test_fraction_preserved_as_float(self):
        """Ultralytics reads 0 < batch < 1 as a GPU-utilisation fraction.

        The old int() coercion turned 0.7 into 0 and broke the run.
        """
        b = resolve_batch(0.7, model_weights="yolo11l.pt", imgsz=1024, device="0")
        assert isinstance(b, float) and b == 0.7

    def test_integer_passthrough(self):
        assert resolve_batch(12, model_weights="yolo11l.pt", imgsz=1024, device="0,1") == 12


class TestResolveModelWeights:
    """Bare model names should prefer the checkpoint baked into the image.

    Otherwise Ultralytics downloads them into the working directory, which under
    `docker run --rm` means a fresh download per run and no offline capability.
    """

    def test_bare_name_uses_cache_when_present(self, tmp_path):
        (tmp_path / "yolo11l.pt").write_bytes(b"weights")
        assert resolve_model_weights("yolo11l.pt", cache=tmp_path) == str(tmp_path / "yolo11l.pt")

    def test_bare_name_passes_through_when_absent(self, tmp_path):
        assert resolve_model_weights("yolo11l.pt", cache=tmp_path) == "yolo11l.pt"

    def test_explicit_path_is_respected(self, tmp_path):
        (tmp_path / "yolo11l.pt").write_bytes(b"weights")
        explicit = "/runs/prev/weights/best.pt"
        assert resolve_model_weights(explicit, cache=tmp_path) == explicit

    def test_yaml_architecture_passes_through(self, tmp_path):
        """A .yaml means train from scratch - never silently swap in weights."""
        (tmp_path / "yolo11l.yaml").write_bytes(b"arch")
        assert resolve_model_weights("yolo11l.yaml", cache=tmp_path) == str(tmp_path / "yolo11l.yaml")
        assert resolve_model_weights("yolo11m.yaml", cache=tmp_path) == "yolo11m.yaml"

    def test_write_train_config_records_resolved_path(self, tmp_path):
        cache = tmp_path / "weights"
        cache.mkdir()
        (cache / "yolo11l.pt").write_bytes(b"w")
        import sparrow_yolo.process_runner as pr

        original = pr.WEIGHTS_CACHE
        pr.WEIGHTS_CACHE = cache
        try:
            p = write_train_config(tmp_path / "c.yaml", model_weights="yolo11l.pt",
                                   data_yaml="d.yaml", device="0")
            assert yaml.safe_load(p.read_text(encoding="utf-8"))["model_weights"] == str(cache / "yolo11l.pt")
        finally:
            pr.WEIGHTS_CACHE = original


class TestResolveWorkers:
    def test_auto_is_positive(self):
        assert resolve_workers("auto", "0") >= 1

    def test_multi_gpu_gets_fewer_per_rank(self):
        """workers is per rank, so the total is workers x GPUs."""
        assert suggest_workers([0, 1]) <= suggest_workers([0])

    def test_explicit_passthrough(self):
        assert resolve_workers(3, "0,1") == 3


class TestWriteTrainConfig:
    def test_round_trips(self, tmp_path):
        p = write_train_config(
            tmp_path / "c.yaml",
            model_weights="yolo11l.pt",
            data_yaml="/prepared/p/data.yaml",
            name="exp",
            imgsz=1024,
            epochs=50,
            batch=8,
            device="0",
            workers=4,
        )
        cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
        assert cfg["batch"] == 8
        assert cfg["workers"] == 4
        assert cfg["exist_ok"] is True   # runner owns the directory, not Ultralytics

    def test_extra_args_passed_through(self, tmp_path):
        p = write_train_config(
            tmp_path / "c.yaml",
            model_weights="yolo11l.pt",
            data_yaml="d.yaml",
            cos_lr=True,
            lr0=0.005,
        )
        cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
        assert cfg["cos_lr"] is True and cfg["lr0"] == 0.005

    def test_fraction_batch_not_truncated(self, tmp_path):
        p = write_train_config(tmp_path / "c.yaml", model_weights="yolo11l.pt",
                               data_yaml="d.yaml", batch=0.8, device="0")
        assert yaml.safe_load(p.read_text(encoding="utf-8"))["batch"] == 0.8


class TestLaunchDetached:
    """The generic launcher shared by training and the audit/prepare/eval jobs.

    Detachment is what lets a two-day run survive the web app restarting, so the
    session and pid-file behaviour is load-bearing, not incidental.
    """

    def test_runs_module_and_writes_handles(self, tmp_path):
        from sparrow_yolo.process_runner import launch_detached, process_is_running

        h = launch_detached(
            "sys", ["-c", "pass"], work_dir=tmp_path, log_name="j.log", pid_name="j.pid",
            banner="job",
        )
        assert h["pid"] > 0
        assert Path(h["pid_path"]).read_text().strip() == str(h["pid"])
        assert "j.log" in h["log_path"]
        for _ in range(50):
            if not process_is_running(h["pid"]):
                break
            time.sleep(0.1)

    def test_log_is_appended_with_a_banner(self, tmp_path):
        """A retry must not destroy the failing attempt's log."""
        from sparrow_yolo.process_runner import launch_detached

        log = tmp_path / "j.log"
        log.write_text("previous attempt output\n", encoding="utf-8")
        launch_detached("sys", ["-c", "pass"], work_dir=tmp_path, log_name="j.log",
                        pid_name="j.pid", banner="job")
        text = log.read_text(encoding="utf-8")
        assert "previous attempt output" in text
        assert "=== job started" in text

    def test_training_wrapper_keeps_its_contract(self, tmp_path):
        """launch_training_process still returns run_dir/status_path keys."""
        from sparrow_yolo.process_runner import launch_training_process

        cfg = tmp_path / "train_config.yaml"
        cfg.write_text("model_weights: x\ndata_yaml: y\n", encoding="utf-8")
        h = launch_training_process(cfg, run_dir=tmp_path, python_exe=sys.executable)
        assert h["run_dir"] == str(tmp_path)
        assert h["status_path"] == str(tmp_path / "status.json")
        assert "work_dir" not in h


class TestUniqueRunDir:
    def test_first_run_uses_plain_name(self, tmp_path):
        assert unique_run_dir(tmp_path, "exp").name == "exp"

    def test_increments_past_existing(self, tmp_path):
        (tmp_path / "exp").mkdir()
        assert unique_run_dir(tmp_path, "exp").name == "exp-2"
        (tmp_path / "exp-2").mkdir()
        assert unique_run_dir(tmp_path, "exp").name == "exp-3"


class TestReadStatus:
    def test_missing_status(self, tmp_path):
        assert read_status(tmp_path)["state"] == "unknown"

    def test_running_without_process_becomes_died(self, tmp_path):
        """An OOM kill never runs the exception handler.

        Without reconciling against the pid, status.json says 'running' forever.
        """
        (tmp_path / "status.json").write_text(json.dumps({"state": "running"}), encoding="utf-8")
        (tmp_path / "train.pid").write_text("999999", encoding="utf-8")
        status = read_status(tmp_path)
        assert status["state"] == "died"
        assert "memory" in status["detail"].lower()

    def test_completed_is_left_alone(self, tmp_path):
        (tmp_path / "status.json").write_text(json.dumps({"state": "completed"}), encoding="utf-8")
        (tmp_path / "train.pid").write_text("999999", encoding="utf-8")
        assert read_status(tmp_path)["state"] == "completed"

    def test_corrupt_status_file(self, tmp_path):
        (tmp_path / "status.json").write_text("{not json", encoding="utf-8")
        assert read_status(tmp_path)["state"] == "unknown"


class TestAllocatorConfig:
    """expandable_segments must never be set on WSL2.

    It needs CUDA virtual memory APIs WSL2 does not implement. With it set,
    nvidia-smi, cuInit, cuCtxCreate, cuMemAlloc and torch's device queries all
    succeed, and only the first tensor allocation fails - which reads as a broken
    driver rather than a config mistake. This cost hours to find once already.
    """

    def test_not_set_in_launch_env(self):
        from sparrow_yolo.process_runner import WSL_SAFE_ENV

        assert "expandable_segments" not in WSL_SAFE_ENV.get("PYTORCH_CUDA_ALLOC_CONF", "")

    def test_not_baked_into_container_config(self):
        import yaml as _yaml

        root = Path(__file__).resolve().parents[1]
        env = _yaml.safe_load((root / "docker-compose.yml").read_text(encoding="utf-8"))
        env = env["services"]["sparrow-yolo"].get("environment", {}) or {}
        assert "expandable_segments" not in str(env.get("PYTORCH_CUDA_ALLOC_CONF", ""))
        assert "expandable_segments" not in (root / "Dockerfile").read_text(encoding="utf-8").split("#")[0]

    def test_doctor_flags_it_on_wsl(self, monkeypatch):
        import sparrow_yolo.hardware as hw

        monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        monkeypatch.setattr(hw, "is_wsl", lambda: True)
        report = hw.doctor([0], run_nccl=False)
        check = next(c for c in report.checks if c.name == "alloc_conf")
        assert not check.ok and check.fatal
        assert "unsupported on WSL2" in check.detail

    def test_doctor_accepts_it_off_wsl(self, monkeypatch):
        import sparrow_yolo.hardware as hw

        monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        monkeypatch.setattr(hw, "is_wsl", lambda: False)
        report = hw.doctor([0], run_nccl=False)
        assert next(c for c in report.checks if c.name == "alloc_conf").ok


class TestLinkSupport:
    """prepare links images rather than copying; doctor must detect when it can't."""

    def test_detects_working_filesystem(self, tmp_path):
        from sparrow_yolo.hardware import link_support

        r = link_support(tmp_path)
        assert r["symlink"] or r["hardlink"]

    def test_cleans_up_its_probes(self, tmp_path):
        from sparrow_yolo.hardware import link_support

        link_support(tmp_path)
        assert list(tmp_path.iterdir()) == []

    def test_reports_failure_without_raising(self, tmp_path, monkeypatch):
        import os as _os

        from sparrow_yolo.hardware import link_support

        def boom(*a, **k):
            raise OSError("operation not permitted")

        monkeypatch.setattr(_os, "symlink", boom)
        monkeypatch.setattr(_os, "link", boom)
        r = link_support(tmp_path)
        assert r["symlink"] is False and r["hardlink"] is False
        assert "not permitted" in r["error"]
        assert list(tmp_path.iterdir()) == []


class TestResolveResume:
    """Resuming must name the checkpoint, not let Ultralytics guess.

    `resume=True` with no path calls get_latest_run(), which picks the most
    recently modified last.pt anywhere under runs/ - not necessarily this run.
    """

    def test_finds_last_in_run_dir(self, tmp_path):
        (tmp_path / "weights").mkdir()
        (tmp_path / "weights" / "last.pt").write_bytes(b"x")
        assert resolve_resume(tmp_path) == tmp_path / "weights" / "last.pt"

    def test_follows_save_dir_from_status(self, tmp_path):
        actual = tmp_path / "exp-2"
        (actual / "weights").mkdir(parents=True)
        (actual / "weights" / "last.pt").write_bytes(b"x")
        run_dir = tmp_path / "exp"
        run_dir.mkdir()
        (run_dir / "status.json").write_text(
            json.dumps({"state": "died", "save_dir": str(actual)}), encoding="utf-8"
        )
        assert resolve_resume(run_dir) == actual / "weights" / "last.pt"

    def test_raises_with_actionable_message(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no weights/last.pt"):
            resolve_resume(tmp_path)


class TestEarlyStopReporting:
    """status.json must distinguish 'ran the full schedule' from 'stopped on patience'."""

    def _write(self, run_dir, **fields):
        (run_dir / "status.json").write_text(json.dumps(fields), encoding="utf-8")

    def test_early_stop_fields_surface(self, tmp_path):
        self._write(tmp_path, state="completed", epochs_completed=43,
                    epochs_requested=100, best_epoch=13, patience=30, stopped_early=True)
        s = read_status(tmp_path)
        assert s["stopped_early"] is True
        assert s["epochs_completed"] == 43 and s["best_epoch"] == 13

    def test_full_run_is_not_flagged_early(self, tmp_path):
        self._write(tmp_path, state="completed", epochs_completed=100,
                    epochs_requested=100, best_epoch=97, stopped_early=False)
        assert read_status(tmp_path)["stopped_early"] is False

    def test_listed_in_run_summary(self, tmp_path):
        run = tmp_path / "exp"
        run.mkdir()
        self._write(run, state="completed", epoch=43, epochs=100,
                    best_epoch=13, stopped_early=True, updated="2026-01-01 00:00:00")
        row = list_runs(tmp_path)[0]
        assert row["stopped_early"] is True and row["best_epoch"] == 13


class TestBestWeights:
    def test_resolves_from_status_save_dir(self, tmp_path):
        """The run directory and Ultralytics' save_dir can differ.

        Building the path from the project name evaluates a previous run's
        checkpoint whenever the directory was auto-incremented.
        """
        actual = tmp_path / "exp-2"
        (actual / "weights").mkdir(parents=True)
        (actual / "weights" / "best.pt").write_bytes(b"x")

        run_dir = tmp_path / "exp"
        run_dir.mkdir()
        (run_dir / "status.json").write_text(
            json.dumps({"state": "completed", "save_dir": str(actual)}), encoding="utf-8"
        )
        assert best_weights(run_dir) == actual / "weights" / "best.pt"

    def test_falls_back_to_run_dir(self, tmp_path):
        (tmp_path / "weights").mkdir()
        (tmp_path / "weights" / "best.pt").write_bytes(b"x")
        (tmp_path / "status.json").write_text(json.dumps({"state": "completed"}), encoding="utf-8")
        assert best_weights(tmp_path) == tmp_path / "weights" / "best.pt"

    def test_none_when_absent(self, tmp_path):
        assert best_weights(tmp_path) is None


class TestExperimentConfig:
    def test_example_config_loads(self):
        cfg = ExperimentConfig.load(Path(__file__).resolve().parents[1] / "configs" / "example_experiment.yaml")
        assert cfg.project_name
        assert cfg.data_yaml.name == "data.yaml"
        assert cfg.devices == [0, 1]

    def test_defaults_fill_missing_sections(self):
        cfg = ExperimentConfig.from_dict({"dataset": {"name": "x"}})
        assert cfg.training["imgsz"] == 1024
        assert cfg.split["train"] == 0.8

    def test_rejects_bad_split(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            ExperimentConfig.from_dict({"split": {"train": 0.5, "val": 0.1, "test": 0.1}})

    def test_rejects_non_multiple_of_32_imgsz(self):
        with pytest.raises(ValueError, match="multiple of 32"):
            ExperimentConfig.from_dict({"training": {"imgsz": 1000}})

    def test_rejects_unknown_section(self, tmp_path):
        p = tmp_path / "c.yaml"
        p.write_text("trainingg:\n  epochs: 1\n", encoding="utf-8")
        with pytest.raises(ValueError, match="unknown top-level"):
            ExperimentConfig.load(p)

    def test_rejects_bad_stratify(self):
        with pytest.raises(ValueError, match="stratify"):
            ExperimentConfig.from_dict({"split": {"stratify": "nope"}})

    def test_derived_paths(self):
        cfg = ExperimentConfig.from_dict({
            "dataset": {"name": "proj"},
            "paths": {"prepared_root": "/p", "runs_root": "/r"},
        })
        assert cfg.prepared_dir == Path("/p/proj")
        assert cfg.class_map_path == Path("/p/proj/classes.json")
        assert cfg.run_dir() == Path("/r/proj")

    def test_save_round_trip(self, tmp_path):
        cfg = ExperimentConfig.from_dict({"dataset": {"name": "x"}})
        cfg.save(tmp_path / "out.yaml")
        assert ExperimentConfig.load(tmp_path / "out.yaml").project_name == "x"


class TestPreparedLocality:
    """Reading training images over the WSL filesystem bridge measured 277
    images/s against 10,043/s on the Linux filesystem — 36x. Invisible unless
    something says so, because the run still works, just slowly."""

    def _tree(self, tmp_path, remote, local):
        from sparrow_yolo.hardware import _remote_image_share
        prepared = tmp_path / "prepared" / "proj" / "images" / "train"
        prepared.mkdir(parents=True)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        for i in range(remote + local):
            # /mnt is what the check looks for, so the "remote" ones are made to
            # resolve there by name rather than by actually mounting anything.
            target = Path("/mnt/d/dataset") / f"img{i}.jpg" if i < remote \
                else elsewhere / f"img{i}.jpg"
            if i >= remote:
                target.write_bytes(b"x")
            try:
                (prepared / f"img{i}.jpg").symlink_to(target)
            except OSError:
                pytest.skip("symlinks unavailable on this platform")
        return _remote_image_share(tmp_path / "prepared")

    def test_reports_the_share_that_lives_on_the_bridge(self, tmp_path):
        share, sampled = self._tree(tmp_path, remote=8, local=2)
        assert sampled == 10
        assert share == pytest.approx(0.8)

    def test_a_fully_local_tree_reports_nothing_remote(self, tmp_path):
        share, sampled = self._tree(tmp_path, remote=0, local=5)
        assert sampled == 5 and share == 0.0

    def test_a_missing_tree_is_not_an_error(self, tmp_path):
        from sparrow_yolo.hardware import _remote_image_share

        assert _remote_image_share(tmp_path / "nope") == (0.0, 0)

    def test_the_check_warns_but_never_blocks_a_run(self, tmp_path):
        """Slow is not broken. A fatal check here would stop a run that works."""
        from sparrow_yolo.hardware import doctor

        self._tree(tmp_path, remote=10, local=0)
        report = doctor((), run_nccl=False, prepared_root=str(tmp_path / "prepared"))
        check = next(c for c in report.checks if c.name == "prepared_locality")
        assert check.fatal is False
        assert check.ok is False and "slower" in check.detail
