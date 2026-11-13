"""Training entry point, executed as a detached subprocess.

Run directly for a headless job::

    python -u -m sparrow_yolo.process_train --config runs/<exp>/train_config.yaml \
                                            --status runs/<exp>/status.json

Design constraints that must survive any refactor:

* ``torch`` and ``ultralytics`` are imported inside :func:`main`, never at module
  scope, so importing this module from a notebook does not pull a CUDA context
  into the kernel.
* ``status.json`` is updated from an Ultralytics callback every epoch. Without
  that, a job killed by the OOM reaper leaves the status stuck at ``running``
  forever, because the process never gets to run its exception handler.
* SIGTERM asks the trainer to stop at the end of the current epoch so the
  checkpoint is written; a second SIGTERM exits immediately.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

_ULTRALYTICS_PASSTHROUGH_BLOCKLIST = {"model_weights", "data_yaml", "run_dir"}


class StatusWriter:
    """Atomic-ish status file. Every write replaces the whole document."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._state: Dict[str, Any] = {}

    def update(self, **fields: Any) -> Dict[str, Any]:
        self._state.update(fields)
        self._state["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._state, indent=2, default=str), encoding="utf-8")

        # Write-then-rename so a reader never sees a half-written document.
        # The rename is atomic on POSIX, but on Windows it fails outright while
        # any process has the destination open — and something polling this file
        # every couple of seconds is exactly the intended usage. Retry briefly
        # rather than losing the update.
        for attempt in range(10):
            try:
                os.replace(tmp, self.path)
                break
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.05)
        return dict(self._state)


def _load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _raise_fd_limit() -> Optional[str]:
    """Raise RLIMIT_NOFILE to the hard limit.

    With ``file_system`` tensor sharing, each dataloader worker holds open file
    handles; the default soft limit is a common cause of "Too many open files"
    several epochs in.
    """
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < hard:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        return f"{soft} -> {resource.getrlimit(resource.RLIMIT_NOFILE)[0]}"
    except Exception:  # noqa: BLE001 - non-POSIX or not permitted
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run YOLO training outside the Jupyter kernel.")
    parser.add_argument("--config", required=True, help="Path to training YAML config")
    parser.add_argument("--status", required=True, help="Path to write status JSON")
    args = parser.parse_args()

    status = StatusWriter(args.status)
    cfg_path = Path(args.config)
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    status.update(state="starting", started=started, config=str(cfg_path), pid=os.getpid())

    try:
        # expandable_segments cuts allocator fragmentation over a long run, but
        # it depends on CUDA's virtual memory APIs (cuMemAddressReserve/
        # cuMemCreate/cuMemMap), which WSL2's GPU virtualisation does not
        # implement. Enabling it there makes *every* tensor allocation fail with
        # "CUDA driver error: unknown error" — while nvidia-smi, cuInit,
        # cuCtxCreate, cuMemAlloc and torch's device queries all still succeed,
        # which makes it extremely hard to attribute. Enable only off WSL.
        from .hardware import is_wsl

        if not is_wsl():
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        fd_note = _raise_fd_limit()

        import torch
        import torch.multiprocessing as mp
        from ultralytics import YOLO

        # Relieves /dev/shm pressure under Docker/WSL2, where the default
        # file_descriptor strategy exhausts the segment during epoch 1.
        try:
            mp.set_sharing_strategy("file_system")
        except Exception:  # noqa: BLE001
            pass

        # TF32 on Blackwell/Ampere+: large matmul speedup, no meaningful accuracy
        # cost for detection training, and orthogonal to AMP.
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:  # noqa: BLE001
            pass

        cfg = _load_config(cfg_path)
        model_weights = cfg.pop("model_weights")
        data_yaml = cfg.pop("data_yaml")
        cfg.pop("run_dir", None)

        train_args: Dict[str, Any] = {
            "data": data_yaml,
            "project": cfg.pop("project", "/workspace/runs"),
            "name": cfg.pop("name", "experiment"),
            "imgsz": int(cfg.pop("imgsz", 1024)),
            "epochs": int(cfg.pop("epochs", 100)),
            "batch": cfg.pop("batch", 8),
            "device": cfg.pop("device", "0"),
            "workers": int(cfg.pop("workers", 4)),
            "patience": int(cfg.pop("patience", 30)),
            "amp": bool(cfg.pop("amp", True)),
            # The runner already chose a unique directory. Letting Ultralytics
            # auto-increment would put the weights somewhere other than where the
            # log, status, and pid files live.
            "exist_ok": bool(cfg.pop("exist_ok", True)),
        }
        # Everything else in the YAML goes straight to Ultralytics: cos_lr, lr0,
        # optimizer, mosaic, close_mosaic, cache, freeze, ...
        train_args.update({k: v for k, v in cfg.items() if k not in _ULTRALYTICS_PASSTHROUGH_BLOCKLIST})

        gpus = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpus.append({
                    "index": i,
                    "name": props.name,
                    "total_gb": round(props.total_memory / 1024**3, 1),
                    "capability": f"sm_{props.major}{props.minor}",
                })

        status.update(
            state="running",
            torch=torch.__version__,
            cuda=torch.version.cuda,
            cuda_available=torch.cuda.is_available(),
            gpus=gpus,
            arch_list=list(torch.cuda.get_arch_list()),
            fd_limit=fd_note,
            train_args=train_args,
            epoch=0,
            epochs=train_args["epochs"],
        )

        print("=" * 80, flush=True)
        print("SPARROW YOLO subprocess training", flush=True)
        print(f"  config : {cfg_path}", flush=True)
        print(f"  model  : {model_weights}", flush=True)
        print(f"  data   : {data_yaml}", flush=True)
        print(f"  torch  : {torch.__version__} (CUDA {torch.version.cuda})", flush=True)
        for g in gpus:
            print(f"  gpu {g['index']}  : {g['name']} {g['total_gb']} GiB {g['capability']}", flush=True)
        print(f"  args   : {train_args}", flush=True)
        print("=" * 80, flush=True)

        model = YOLO(model_weights)

        state: Dict[str, Any] = {"trainer": None, "stopping": False}

        def _on_train_start(trainer):
            # Publish save_dir immediately so watchers can find the weights
            # without guessing at Ultralytics' directory naming.
            state["trainer"] = trainer
            status.update(save_dir=str(trainer.save_dir), state="running")

        def _on_epoch_end(trainer):
            metrics = {}
            try:
                metrics = {k: float(v) for k, v in (trainer.metrics or {}).items()}
            except Exception:  # noqa: BLE001
                pass
            status.update(
                state="running",
                epoch=int(getattr(trainer, "epoch", 0)) + 1,
                epochs=int(getattr(trainer, "epochs", train_args["epochs"])),
                best_fitness=float(trainer.best_fitness) if getattr(trainer, "best_fitness", None) else None,
                metrics=metrics,
                save_dir=str(trainer.save_dir),
                last=str(getattr(trainer, "last", "")),
                best=str(getattr(trainer, "best", "")),
            )

        def _on_train_end(trainer):
            # Ultralytics' EarlyStopping lives on trainer.stopper and knows which
            # epoch was best. Without recording this, a run that stopped at epoch
            # 43 of 100 on patience is indistinguishable in status.json from one
            # that ran the full schedule.
            stopper = getattr(trainer, "stopper", None)
            done = int(getattr(trainer, "epoch", 0)) + 1
            requested = int(getattr(trainer, "epochs", train_args["epochs"]))
            best_epoch = getattr(stopper, "best_epoch", None)
            state["summary"] = {
                "epochs_completed": done,
                "epochs_requested": requested,
                "best_epoch": int(best_epoch) + 1 if best_epoch is not None else None,
                "patience": int(getattr(stopper, "patience", 0)) or None,
                "stopped_early": done < requested and not state["stopping"],
            }

        model.add_callback("on_train_start", _on_train_start)
        model.add_callback("on_fit_epoch_end", _on_epoch_end)
        model.add_callback("on_train_end", _on_train_end)

        def _handle_term(signum, _frame):
            trainer = state["trainer"]
            if state["stopping"] or trainer is None:
                print(f"\nSignal {signum} again - exiting now.", flush=True)
                status.update(state="stopped", detail=f"hard stop on signal {signum}")
                os._exit(143)
            state["stopping"] = True
            trainer.stop = True  # Ultralytics checks this at the epoch boundary
            status.update(state="stopping", detail=f"signal {signum}; finishing current epoch")
            print(f"\nSignal {signum} received - stopping after this epoch. Send again to force.", flush=True)

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, _handle_term)
            except (ValueError, OSError):
                pass

        results = model.train(**train_args)

        save_dir = str(getattr(results, "save_dir", "") or getattr(state["trainer"], "save_dir", ""))
        best = Path(save_dir) / "weights" / "best.pt" if save_dir else None
        summary = state.get("summary") or {}
        status.update(
            state="stopped" if state["stopping"] else "completed",
            finished=time.strftime("%Y-%m-%d %H:%M:%S"),
            save_dir=save_dir,
            best_weights=str(best) if best and best.exists() else "",
            last_weights=str(Path(save_dir) / "weights" / "last.pt") if save_dir else "",
            **summary,
        )

        if state["stopping"]:
            reason = "stopped on signal"
        elif summary.get("stopped_early"):
            reason = (
                f"early stopping: no improvement for {summary.get('patience')} epochs; "
                f"ran {summary.get('epochs_completed')}/{summary.get('epochs_requested')} epochs, "
                f"best was epoch {summary.get('best_epoch')}"
            )
        else:
            reason = f"completed all {summary.get('epochs_requested', train_args['epochs'])} epochs"
        print(f"Training finished - {reason}. save_dir={save_dir}", flush=True)
        return 0

    except Exception as e:  # noqa: BLE001 - must be recorded, not just raised
        tb = traceback.format_exc()
        status.update(
            state="failed",
            finished=time.strftime("%Y-%m-%d %H:%M:%S"),
            error=repr(e),
            traceback=tb,
        )
        print(tb, file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
