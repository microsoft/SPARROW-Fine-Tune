"""Launch and supervise detached training processes.

The notebook only ever *watches*; the training process is in its own session, so
interrupting a cell (or losing the kernel entirely) does not kill the run.

The run directory is chosen here and passed to Ultralytics with ``exist_ok=True``
so that ``train.log``, ``status.json``, ``train.pid`` and ``weights/`` all end up
in the same place. Letting Ultralytics auto-increment the directory is what
produced this repo's split ``Sub_Saharan_v8`` / ``Sub_Saharan_v8-2`` pair, where
the logs and the weights lived in different folders.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from .hardware import (is_wsl, parse_device, suggest_batch,
                       suggest_workers)

# Defaults applied to every launch. Overridable via extra_env.
WSL_SAFE_ENV: Dict[str, str] = {
    "PYTHONUNBUFFERED": "1",
    "TORCH_SHOW_CPP_STACKTRACES": "1",
    # WSL2 exposes GPUs through WDDM, which has no peer-to-peer support. Leaving
    # P2P enabled is the usual reason a two-GPU run hangs at NCCL init and then
    # dies without writing anything.
    "NCCL_P2P_DISABLE": "1",
    "NCCL_IB_DISABLE": "1",
    "NCCL_DEBUG": "WARN",
    # Surface a hung collective as an error instead of blocking forever.
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    # PYTORCH_CUDA_ALLOC_CONF is deliberately absent. `expandable_segments:True`
    # cuts fragmentation on native Linux but depends on CUDA virtual memory APIs
    # that WSL2 does not implement, and every allocation then fails with
    # "CUDA driver error: unknown error". process_train.py enables it only when
    # not running under WSL.
    # Each dataloader worker otherwise spawns as many OpenMP threads as there are
    # cores, which thrashes badly once workers * ranks is large.
    "OMP_NUM_THREADS": "4",
}

TERMINAL_STATES = {"completed", "failed", "stopped"}


def unique_run_dir(runs_root: str | Path, name: str) -> Path:
    """``runs/<name>``, or ``runs/<name>-2``, ``-3``... if already used.

    Mirrors Ultralytics' increment scheme, but decides it up-front so the
    directory can be passed back in with ``exist_ok=True``.
    """
    runs_root = Path(runs_root)
    candidate = runs_root / name
    if not candidate.exists():
        return candidate
    i = 2
    while (runs_root / f"{name}-{i}").exists():
        i += 1
    return runs_root / f"{name}-{i}"


#: Where the image bakes COCO-pretrained checkpoints. See scripts/fetch_weights.sh.
WEIGHTS_CACHE = Path(os.environ.get("SPARROW_WEIGHTS_DIR", "/workspace/weights"))


def resolve_model_weights(model: str, cache: Optional[Path] = None) -> str:
    """Point a bare model name at the pre-downloaded checkpoint, if there is one.

    ``"yolo11l.pt"`` is COCO-pretrained — loading it is what makes this transfer
    learning rather than training from scratch (a ``.yaml`` model would be random
    init). Ultralytics fetches such a name over the network into the working
    directory, so under ``docker run --rm`` it re-downloads every run and cannot
    start at all offline. Preferring the baked-in copy avoids both.

    Explicit paths and ``.yaml`` architectures are passed through untouched.
    """
    cache = WEIGHTS_CACHE if cache is None else cache
    p = Path(str(model))
    if p.is_absolute() or len(p.parts) > 1:
        return str(model)  # caller gave a path; respect it
    cached = cache / p.name
    return str(cached) if cached.is_file() else str(model)


def resolve_resume(run_dir: str | Path) -> Path:
    """Return the ``last.pt`` to resume a specific run from.

    Ultralytics' ``resume=True`` with no explicit checkpoint calls
    ``get_latest_run()``, which scans for the most recently modified ``last.pt``
    anywhere under the runs directory — not necessarily the run you meant. Naming
    the checkpoint removes that ambiguity.
    """
    run_dir = Path(run_dir)
    last = run_dir / "weights" / "last.pt"
    if last.is_file():
        return last

    save_dir = resolve_save_dir(run_dir)
    if save_dir:
        alt = save_dir / "weights" / "last.pt"
        if alt.is_file():
            return alt

    raise FileNotFoundError(
        f"Cannot resume: no weights/last.pt under {run_dir}. Resuming continues an "
        f"existing run; to start a new one, set training.resume: false."
    )


def resolve_batch(
    batch: Any,
    *,
    model_weights: str,
    imgsz: int,
    device: Any,
) -> Any:
    """Turn ``"auto"`` into a concrete batch and refuse the multi-GPU AutoBatch trap.

    Ultralytics' ``batch=-1`` AutoBatch measures actual VRAM use, but only works
    on a single GPU. Under DDP it silently falls back to a fixed 16 — which at
    imgsz 1024 on a large model is an immediate OOM. ``"auto"`` therefore maps to
    real AutoBatch on one GPU and to a sized estimate on several.
    """
    devices = parse_device(device)

    # AutoBatch (-1) measures real VRAM use and is the best answer where it
    # works, but it is unusable in two situations here.
    #
    # Multiple GPUs: it is single-GPU only, and under DDP Ultralytics silently
    # substitutes a fixed 16 — an immediate OOM at imgsz 1024 on a large model.
    #
    # WSL2: WDDM does not report per-process device memory to CUDA, so the
    # measurements AutoBatch fits its curve to are wrong. Observed on this
    # project's machine: a 32 GiB card reported "31.86G total, 0.25G reserved"
    # and AutoBatch chose batch-size 4, roughly a quarter of what fits — four
    # times the steps per epoch, and noisier gradients with it.
    unmeasurable = len(devices) > 1 or is_wsl()
    if batch == "auto":
        if not unmeasurable:
            return -1  # let Ultralytics measure it properly
        return suggest_batch(model_weights, imgsz, devices)
    if isinstance(batch, (int, float)) and batch == -1 and unmeasurable:
        return suggest_batch(model_weights, imgsz, devices)
    if isinstance(batch, float) and 0.0 < batch < 1.0:
        return float(batch)  # Ultralytics reads this as a GPU-utilisation fraction
    return int(batch)


def resolve_workers(workers: Any, device: Any) -> int:
    if workers == "auto":
        return suggest_workers(parse_device(device))
    return int(workers)


def write_train_config(
    path: str | Path,
    *,
    model_weights: str,
    data_yaml: str,
    project: str = "/workspace/runs",
    name: str = "experiment",
    imgsz: int = 1024,
    epochs: int = 100,
    batch: Any = "auto",
    device: Any = "0",
    workers: Any = "auto",
    patience: int = 30,
    amp: bool = True,
    exist_ok: bool = True,
    **extra: Any,
) -> Path:
    """Write the standalone YAML consumed by :mod:`sparrow_yolo.process_train`.

    ``batch`` accepts ``"auto"``, an int, ``-1``, or a float in ``(0, 1)``. The
    float form is Ultralytics' GPU-utilisation target and is preserved as a float
    — coercing it to int, as this function used to, turns ``0.7`` into ``0`` and
    breaks the run.
    """
    resolved_batch = resolve_batch(batch, model_weights=model_weights, imgsz=imgsz, device=device)
    cfg: Dict[str, Any] = {
        "model_weights": resolve_model_weights(model_weights),
        "data_yaml": str(data_yaml),
        "project": str(project),
        "name": str(name),
        "imgsz": int(imgsz),
        "epochs": int(epochs),
        "batch": resolved_batch,
        "device": str(device),
        "workers": resolve_workers(workers, device),
        "patience": int(patience),
        "amp": bool(amp),
        "exist_ok": bool(exist_ok),
    }
    cfg.update({k: v for k, v in extra.items() if v is not None})

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def preflight(config_path: str | Path, *, run_nccl: bool = True) -> Dict[str, Any]:
    """Validate the environment before committing to a long run.

    Returns ``{"ok": bool, "report": str, "checks": [...]}``. Cheap, and catches
    the two failure modes this project has actually hit: an undersized
    ``/dev/shm`` and a DDP job that cannot initialise NCCL.
    """
    from .hardware import doctor

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    devices = parse_device(cfg.get("device", "0"))
    report = doctor(devices or [0], run_nccl=run_nccl, data_yaml=cfg.get("data_yaml"))
    return {
        "ok": report.ok,
        "report": report.render(),
        "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail, "fatal": c.fatal} for c in report.checks],
    }


def launch_detached(
    module: str,
    args: Sequence[str],
    *,
    work_dir: str | Path,
    log_name: str = "train.log",
    pid_name: str = "train.pid",
    banner: str = "run",
    python_exe: Optional[str] = None,
    extra_env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
) -> Dict[str, Any]:
    """Start ``python -m <module> <args>`` detached, and return its handles.

    The child gets its own session (``start_new_session=True``), so it outlives
    the launching process entirely — a web request, a notebook kernel, or a CLI
    invocation can all start work that keeps running after they exit — and
    :func:`stop_training` can signal the whole process group at once, including
    ranks Ultralytics spawns for DDP.

    Shared by training and by the audit/prepare/evaluate/export jobs so they all
    inherit the same session handling, WSL-safe environment, append-with-banner
    logging and pid file. Do not reimplement this per job type.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    log_path = work_dir / log_name
    pid_path = work_dir / pid_name

    env = os.environ.copy()
    for k, v in WSL_SAFE_ENV.items():
        env.setdefault(k, v)
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})

    # Default to the interpreter already running us. Using a bare "python3" would
    # resolve differently inside a virtualenv (or not at all on Windows), and the
    # child needs the same packages we have.
    interpreter = python_exe or os.environ.get("PYTHON") or sys.executable or "python3"
    cmd = [interpreter, "-u", "-m", module, *map(str, args)]

    # Append rather than truncate so a retry keeps the failing attempt's log, but
    # mark the boundary — concatenated logs with no separator are unreadable.
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            f"\n{'=' * 80}\n=== {banner} started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n{'=' * 80}\n"
        )

    # Opened for the lifetime of the child and closed here; a long-lived parent
    # must not accumulate a file handle per launch.
    with open(log_path, "ab", buffering=0) as log_f:
        proc = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=cwd or str(Path(__file__).resolve().parents[2]),
            env=env,
            start_new_session=True,
        )

    pid_path.write_text(str(proc.pid), encoding="utf-8")
    return {
        "pid": proc.pid,
        "cmd": cmd,
        "work_dir": str(work_dir),
        "log_path": str(log_path),
        "pid_path": str(pid_path),
    }


def launch_training_process(
    config_path: str | Path,
    *,
    run_dir: str | Path,
    python_exe: Optional[str] = None,
    extra_env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
) -> Dict[str, Any]:
    """Start training detached and return its handles.

    Thin wrapper over :func:`launch_detached` that knows the training module and
    the ``status.json`` convention.
    """
    run_dir = Path(run_dir)
    if is_running(run_dir):
        raise RuntimeError(
            f"A training process is already running in {run_dir} "
            f"(pid {read_pid(run_dir)}). Call stop_training() first."
        )

    status_path = Path(run_dir) / "status.json"
    handles = launch_detached(
        "sparrow_yolo.process_train",
        ["--config", str(config_path), "--status", str(status_path)],
        work_dir=run_dir,
        log_name="train.log",
        pid_name="train.pid",
        banner="run",
        python_exe=python_exe,
        extra_env=extra_env,
        cwd=cwd,
    )
    handles["run_dir"] = handles.pop("work_dir")
    handles["status_path"] = str(status_path)
    return handles


# --------------------------------------------------------------------------
# supervision
# --------------------------------------------------------------------------

def read_pid(run_dir: str | Path) -> Optional[int]:
    pid_path = Path(run_dir) / "train.pid"
    if not pid_path.exists():
        return None
    try:
        return int(pid_path.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return None


def process_is_running(pid: Optional[int]) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, TypeError):
        return False


def is_running(run_dir: str | Path) -> bool:
    return process_is_running(read_pid(run_dir))


def read_status(run_dir: str | Path) -> Dict[str, Any]:
    """Read ``status.json``, cross-checked against the actual process.

    A job killed by the kernel OOM reaper never runs its exception handler, so
    the file is left saying ``running`` indefinitely. Reconciling against the pid
    is the only way a reader can tell the difference.
    """
    run_dir = Path(run_dir)
    status_path = run_dir / "status.json"
    if not status_path.exists():
        return {"state": "unknown", "message": "No status.json yet"}
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return {"state": "unknown", "error": repr(e)}

    pid = read_pid(run_dir)
    alive = process_is_running(pid)
    status["pid"] = pid
    status["process_alive"] = alive
    if status.get("state") in {"starting", "running", "stopping"} and not alive:
        status["state"] = "died"
        status["detail"] = (
            "Process is gone but status was never finalised - almost always an "
            "out-of-memory kill. Check the tail of train.log, then lower batch or workers."
        )
    return status


def tail_log(run_dir: str | Path, n: int = 80) -> str:
    log_path = Path(run_dir) / "train.log"
    if not log_path.exists():
        return "No train.log yet"
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


def stream_log(run_dir: str | Path, poll_seconds: float = 2.0, n_start_lines: int = 20) -> Dict[str, Any]:
    """Follow the log until the run reaches a terminal state. Returns the status.

    Interrupting this is safe: it stops watching, not training.
    """
    run_dir = Path(run_dir)
    log_path = run_dir / "train.log"

    pos = 0
    if log_path.exists() and n_start_lines > 0:
        print(tail_log(run_dir, n_start_lines), flush=True)
        pos = log_path.stat().st_size

    try:
        while True:
            if log_path.exists():
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    f.seek(pos)
                    new = f.read()
                    pos = f.tell()
                    if new:
                        print(new, end="", flush=True)

            status = read_status(run_dir)
            state = status.get("state")
            if state in TERMINAL_STATES or state == "died":
                print(f"\n=== state: {state} ===", flush=True)
                if state in {"failed", "died"} and status.get("detail"):
                    print(status["detail"], flush=True)
                return status
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        print("\nStopped watching. Training continues in the background.", flush=True)
        return read_status(run_dir)


def stop_training(run_dir: str | Path, *, timeout: float = 120.0, force: bool = False) -> Dict[str, Any]:
    """Ask the run to stop, waiting for it to checkpoint.

    Sends SIGTERM to the process group so DDP ranks go too. The trainer finishes
    its current epoch and saves, then exits; after ``timeout`` it is killed.
    """
    run_dir = Path(run_dir)
    pid = read_pid(run_dir)
    if pid is None:
        return {"stopped": False, "detail": "No train.pid found"}
    if not process_is_running(pid):
        return {"stopped": True, "detail": f"Process {pid} was already gone"}

    sig = signal.SIGKILL if force else signal.SIGTERM
    for send in (lambda: os.killpg(os.getpgid(pid), sig), lambda: os.kill(pid, sig)):
        try:
            send()
            break
        except OSError:
            continue
    else:
        return {"stopped": False, "detail": f"Could not signal process {pid}"}

    deadline = time.time() + timeout
    while time.time() < deadline:
        if not process_is_running(pid):
            return {"stopped": True, "detail": f"Process {pid} exited after {sig.name}"}
        time.sleep(1.0)

    if not force:
        return stop_training(run_dir, timeout=30.0, force=True)
    return {"stopped": False, "detail": f"Process {pid} still alive after SIGKILL"}


def resolve_save_dir(run_dir: str | Path) -> Optional[Path]:
    """Where Ultralytics actually wrote this run, per ``status.json``."""
    status = read_status(run_dir)
    save_dir = status.get("save_dir")
    return Path(save_dir) if save_dir else None


def best_weights(run_dir: str | Path) -> Optional[Path]:
    """Path to ``best.pt`` for a run, resolved rather than guessed.

    Guessing ``runs/<project>/weights/best.pt`` silently evaluates a *previous*
    run's checkpoint whenever the directory was incremented.
    """
    status = read_status(run_dir)
    for key in ("best_weights", "best"):
        p = status.get(key)
        if p and Path(p).exists():
            return Path(p)
    save_dir = resolve_save_dir(run_dir)
    for candidate in ([save_dir] if save_dir else []) + [Path(run_dir)]:
        p = candidate / "weights" / "best.pt"
        if p.exists():
            return p
    return None


def list_runs(runs_root: str | Path) -> List[Dict[str, Any]]:
    """Summarise every run directory, newest first."""
    runs_root = Path(runs_root)
    if not runs_root.is_dir():
        return []
    out: List[Dict[str, Any]] = []
    for d in sorted(runs_root.iterdir()):
        if not d.is_dir():
            continue
        status = read_status(d)
        best = best_weights(d)
        out.append({
            "run": d.name,
            "path": str(d),
            "state": status.get("state", "unknown"),
            "epoch": status.get("epoch"),
            "epochs": status.get("epochs"),
            "best_epoch": status.get("best_epoch"),
            "stopped_early": status.get("stopped_early"),
            "best_fitness": status.get("best_fitness"),
            "best_weights": str(best) if best else "",
            "updated": status.get("updated", ""),
        })
    return sorted(out, key=lambda r: r["updated"], reverse=True)
