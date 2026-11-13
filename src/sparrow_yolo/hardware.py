"""GPU / WSL2 preflight checks and batch-size heuristics.

Two of the failures visible in this repo's ``runs/`` history are environment
problems, not model problems: a dataloader that died mid-epoch (shared memory)
and a two-GPU run that produced nothing at all (DDP/NCCL init). Everything here
exists to surface those in seconds instead of after dataset preparation.
"""

from __future__ import annotations

import multiprocessing as std_mp
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# MARGINAL activation memory per additional image at imgsz=1024 with AMP, in MB.
# Marginal, not total: the fixed cost of a run is _RESERVE_GB below, and counting
# it once per image is what made this table size a batch at roughly half what
# fits. Scaled by (imgsz/1024)**2.
#
# Calibrated against two real runs of yolo11l at 1024 on a 32 GiB card:
#   batch 4  -> 9.74 GiB reserved
#   batch 10 -> 15.9 GiB reserved
# i.e. ~1.03 GiB per image on top of ~5.6 GiB fixed. Only yolo11l is measured;
# the rest keep their previous ratios to it, so they are consistent but not
# verified. Set training.batch explicitly if you need the last few images.
_MB_PER_IMAGE_AT_1024 = {
    "yolo11n": 310,
    "yolo11s": 490,
    "yolo11m": 810,
    "yolo11l": 1030,   # measured
    "yolo11x": 1390,
    "yolo12n": 340,
    "yolo12s": 540,
    "yolo12m": 850,
    "yolo12l": 1080,
    "yolo12x": 1430,
    "yolov8n": 290,
    "yolov8s": 450,
    "yolov8m": 760,
    "yolov8l": 990,
    "yolov8x": 1340,
}
_DEFAULT_MB_PER_IMAGE = 1030  # unrecognisable name -> assume "l"-sized

# Ultralytics names encode the scale in the trailing letter, so a release the
# table predates (yolo26m, yolo13x, ...) still sizes correctly instead of falling
# back to the "l" default and wasting half the batch.
_MB_BY_SCALE = {"n": 310, "s": 490, "m": 810, "l": 1030, "x": 1390}
_SCALE_RE = re.compile(r"^yolo[a-z]*\d+([nsmlx])$")

# Fixed cost per GPU, independent of batch: weights, gradients, optimiser state,
# the EMA copy, cuDNN workspace, NCCL buffers, and the CUDA context itself.
# Measured at 5.6 GiB for yolo11l; rounded up, since underestimating this is an
# OOM and overestimating it costs one image of batch.
_RESERVE_GB = 6.0


@dataclass
class GpuInfo:
    index: int
    name: str
    total_mb: int
    used_mb: int
    display_attached: bool = False
    #: Live telemetry. None when nvidia-smi did not report the field — older
    #: drivers and some virtualised GPUs omit power, and WSL2 omits nothing but
    #: has been known to return "[N/A]".
    utilization_pct: Optional[int] = None
    temperature_c: Optional[int] = None
    power_w: Optional[float] = None
    power_limit_w: Optional[float] = None

    @property
    def free_mb(self) -> int:
        return max(0, self.total_mb - self.used_mb)

    @property
    def free_gb(self) -> float:
        return self.free_mb / 1024.0

    @property
    def used_pct(self) -> float:
        return (self.used_mb / self.total_mb * 100.0) if self.total_mb else 0.0

    def as_dict(self) -> Dict[str, Any]:
        """Flat JSON-safe view for the API / SSE stream."""
        return {
            "index": self.index,
            "name": self.name,
            "total_mb": self.total_mb,
            "used_mb": self.used_mb,
            "free_mb": self.free_mb,
            "used_pct": round(self.used_pct, 1),
            "display_attached": self.display_attached,
            "utilization_pct": self.utilization_pct,
            "temperature_c": self.temperature_c,
            "power_w": self.power_w,
            "power_limit_w": self.power_limit_w,
        }


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fatal: bool = False


@dataclass
class DoctorReport:
    checks: List[Check] = field(default_factory=list)
    gpus: List[GpuInfo] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks if c.fatal)

    def add(self, name: str, ok: bool, detail: str, fatal: bool = False) -> None:
        self.checks.append(Check(name, ok, detail, fatal))

    def render(self) -> str:
        lines = []
        for c in self.checks:
            mark = "PASS" if c.ok else ("FAIL" if c.fatal else "WARN")
            lines.append(f"[{mark}] {c.name}: {c.detail}")
        return "\n".join(lines)


def parse_device(device: Any) -> List[int]:
    """Normalise Ultralytics-style device strings to a list of GPU indices.

    ``"0,1"`` -> ``[0, 1]``; ``"cpu"`` / ``""`` / ``None`` -> ``[]``.
    """
    if device is None:
        return []
    if isinstance(device, int):
        return [device]
    if isinstance(device, (list, tuple)):
        out: List[int] = []
        for d in device:
            out.extend(parse_device(d))
        return out
    text = str(device).strip().lower()
    if text in {"", "cpu", "mps", "none"}:
        return []
    parts = [p.strip() for p in text.replace(" ", ",").split(",") if p.strip() != ""]
    idx: List[int] = []
    for p in parts:
        p = p.removeprefix("cuda:")
        try:
            idx.append(int(p))
        except ValueError:
            continue
    return idx


#: Fields requested from nvidia-smi, in order. Keep in sync with _parse_gpu_row.
_GPU_QUERY_FIELDS = (
    "index",
    "name",
    "memory.total",
    "memory.used",
    "display_active",
    "utilization.gpu",
    "temperature.gpu",
    "power.draw",
    "power.limit",
)


def _num(text: str, cast):
    """nvidia-smi writes '[N/A]' or '[Not Supported]' for unavailable fields."""
    try:
        return cast(float(text))
    except (TypeError, ValueError):
        return None


def _parse_gpu_row(parts: List[str]) -> Optional[GpuInfo]:
    if len(parts) < 4:
        return None
    try:
        info = GpuInfo(
            index=int(parts[0]),
            name=parts[1],
            total_mb=int(float(parts[2])),
            used_mb=int(float(parts[3])),
        )
    except ValueError:
        return None

    def at(i: int) -> str:
        return parts[i] if len(parts) > i else ""

    info.display_attached = at(4).lower().startswith("enab")
    info.utilization_pct = _num(at(5), int)
    info.temperature_c = _num(at(6), int)
    info.power_w = _num(at(7), float)
    info.power_limit_w = _num(at(8), float)
    return info


def query_gpus() -> List[GpuInfo]:
    """Read GPU inventory and live telemetry via nvidia-smi.

    Uses nvidia-smi rather than torch so this works before torch is importable
    and without allocating a CUDA context (which would itself consume VRAM) —
    which matters because this is polled once a second while training runs.

    Telemetry fields degrade to ``None`` rather than failing: some drivers and
    virtualised GPUs report ``[N/A]`` for power, and the whole call returns ``[]``
    if nvidia-smi is absent.
    """
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run(
            [exe, f"--query-gpu={','.join(_GPU_QUERY_FIELDS)}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return []

    gpus: List[GpuInfo] = []
    for line in out.strip().splitlines():
        info = _parse_gpu_row([p.strip() for p in line.split(",")])
        if info is not None:
            gpus.append(info)
    return gpus


def model_key(model_weights: str) -> str:
    return Path(str(model_weights)).stem.lower().replace("-", "")


def per_image_mb(model_weights: str) -> Tuple[int, str]:
    """Peak activation MB per image at 1024px, and how it was determined.

    Falls back through: exact table entry -> scale letter parsed from the name ->
    an "l"-sized guess. A custom checkpoint such as ``sparrow_africa_v1.pt``
    carries no scale in its name, so the guess applies; set ``training.batch``
    explicitly if that is wrong for your architecture.
    """
    key = model_key(model_weights)
    if key in _MB_PER_IMAGE_AT_1024:
        return _MB_PER_IMAGE_AT_1024[key], "known model"
    match = _SCALE_RE.match(key)
    if match:
        return _MB_BY_SCALE[match.group(1)], f"scale '{match.group(1)}' parsed from name"
    return _DEFAULT_MB_PER_IMAGE, "unknown architecture, assuming 'l'-sized"


def suggest_batch(
    model_weights: str,
    imgsz: int,
    devices: Sequence[int],
    gpus: Optional[Sequence[GpuInfo]] = None,
    *,
    headroom: float = 0.85,
    max_per_gpu: int = 64,
) -> int:
    """Pick a total batch size that fits the tightest selected GPU.

    Returns the TOTAL batch (what Ultralytics expects); under DDP it is split
    across ranks. Sized from the *free* memory of the most-loaded selected GPU,
    which matters here because GPU 0 drives the display.

    Memory is modelled as ``fixed + per_image x batch``, both calibrated against
    real runs. The earlier form applied ``headroom`` to what was left *after* the
    fixed cost and then charged the fixed cost again through an inflated
    per-image figure — discounting twice, which halved the answer. ``headroom``
    now means what it says: the fraction of free memory the run may occupy.

    Still a heuristic, and still deliberately short of the edge. Ultralytics' own
    AutoBatch (``batch=-1``) measures properly, but only on a single GPU and only
    where CUDA can see real memory use — under DDP it substitutes a fixed 16, and
    on WSL2 the numbers it profiles are not real. ``resolve_batch`` sends both
    cases here.
    """
    n = max(1, len(devices))
    gpus = list(gpus if gpus is not None else query_gpus())
    selected = [g for g in gpus if g.index in set(devices)] or gpus

    if selected:
        free_gb = min(g.free_gb for g in selected)
    else:
        free_gb = 24.0  # no nvidia-smi: assume a mid-range card and stay conservative

    usable_gb = max(1.0, free_gb * headroom - _RESERVE_GB)

    mb, _ = per_image_mb(model_weights)
    mb *= (float(imgsz) / 1024.0) ** 2
    per_gpu = int((usable_gb * 1024.0) // max(1.0, mb))

    # Round down to an even number; odd batches interact badly with the final
    # partial batch under DDP.
    per_gpu = max(1, min(max_per_gpu, per_gpu))
    if per_gpu > 1 and per_gpu % 2 == 1:
        per_gpu -= 1
    return per_gpu * n


def suggest_workers(devices: Sequence[int], *, cap: int = 8) -> int:
    """Dataloader workers PER RANK.

    Ultralytics spawns ``workers`` processes per GPU, so the total is
    ``workers * n_gpus``. Oversubscribing is the classic cause of a run that dies
    partway through the first epoch under WSL2.
    """
    n = max(1, len(devices))
    try:
        cpus = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except AttributeError:
        cpus = os.cpu_count() or 4
    # Leave half the cores for the training processes themselves.
    per_rank = max(2, (cpus // 2) // n)
    return int(min(cap, per_rank))


def shm_bytes(path: str = "/dev/shm") -> Optional[int]:
    try:
        st = os.statvfs(path)
    except (OSError, AttributeError):
        return None
    return st.f_blocks * st.f_frsize


def link_support(directory: str | Path) -> Dict[str, Any]:
    """Whether ``directory`` can hold symlinks and hardlinks.

    ``prepare`` links images into the prepared tree instead of copying them. On a
    Windows filesystem reached through drvfs (a checkout on the Desktop, C:, D:)
    symlink creation usually needs Developer Mode, and the fallback is to copy —
    silently duplicating the whole dataset. Better to find out here than via
    ``df`` three hours later.
    """
    directory = Path(directory)
    result: Dict[str, Any] = {"path": str(directory), "symlink": False, "hardlink": False, "error": None}
    target = directory / ".sparrow_probe_target"
    sym = directory / ".sparrow_probe_symlink"
    hard = directory / ".sparrow_probe_hardlink"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        target.write_text("probe", encoding="utf-8")
        try:
            os.symlink(target, sym)
            result["symlink"] = sym.is_symlink()
        except (OSError, NotImplementedError) as e:
            result["error"] = f"{type(e).__name__}: {e}"
        try:
            os.link(target, hard)
            result["hardlink"] = hard.exists()
        except (OSError, NotImplementedError):
            pass
    except OSError as e:
        result["error"] = f"{type(e).__name__}: {e}"
    finally:
        for p in (sym, hard, target):
            try:
                if p.is_symlink() or p.exists():
                    p.unlink()
            except OSError:
                pass
    return result


def is_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text(errors="replace").lower()
    except OSError:
        return False


def _cuda_smoke_worker(devices: Sequence[int], queue) -> None:  # pragma: no cover - subprocess
    try:
        import torch

        notes = []
        for d in devices:
            torch.cuda.set_device(d)
            x = torch.ones(1024, 1024, device=f"cuda:{d}")
            notes.append(f"cuda:{d} matmul -> {float((x @ x)[0, 0].item()):.0f}")
        queue.put((True, "; ".join(notes)))
    except Exception as e:  # noqa: BLE001
        queue.put((False, f"{type(e).__name__}: {e}"))


def cuda_smoke_test(devices: Sequence[int], timeout: float = 120.0) -> Check:
    """Run a real kernel on each selected GPU, in a clean subprocess.

    Separates "CUDA does not work in this container at all" from "NCCL
    specifically cannot initialise" — very different fixes, and the NCCL error
    message alone does not distinguish them.
    """
    ctx = std_mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_cuda_smoke_worker, args=(list(devices), queue))
    proc.start()
    proc.join(timeout=timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=10)
        return Check("cuda_compute", False, "kernel launch hung", fatal=True)
    if queue.empty():
        return Check("cuda_compute", False, "worker died without reporting", fatal=True)
    ok, detail = queue.get()
    return Check("cuda_compute", ok, detail, fatal=True)


def _nccl_worker(rank: int, world_size: int, port: int, queue) -> None:  # pragma: no cover - subprocess
    try:
        import torch
        import torch.distributed as dist

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ["MASTER_PORT"] = str(port)
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        t = torch.ones(1024, device=f"cuda:{rank}")
        dist.all_reduce(t)
        value = float(t[0].item())
        dist.destroy_process_group()
        queue.put((rank, True, f"all_reduce -> {value:.0f} (expected {world_size})"))
    except Exception as e:  # noqa: BLE001 - report anything, this is a diagnostic
        queue.put((rank, False, f"{type(e).__name__}: {e}"))


def nccl_selftest(devices: Sequence[int], timeout: float = 120.0) -> Check:
    """Run a real two-process NCCL all-reduce before committing to a long run.

    A DDP job that cannot initialise NCCL fails *silently* under Ultralytics: the
    run directory gets an ``args.yaml`` and nothing else. Ten seconds here is
    worth it.
    """
    if len(devices) < 2:
        return Check("nccl", True, "single GPU, DDP not used", fatal=False)

    try:
        import torch  # noqa: F401
    except ImportError:
        return Check("nccl", False, "torch not importable", fatal=True)

    ctx = std_mp.get_context("spawn")
    queue = ctx.Queue()
    port = 29555
    procs = [
        ctx.Process(target=_nccl_worker, args=(r, len(devices), port, queue))
        for r in range(len(devices))
    ]
    for p in procs:
        p.start()

    results: List[tuple] = []
    for p in procs:
        p.join(timeout=timeout)
    while not queue.empty():
        results.append(queue.get())
    for p in procs:
        if p.is_alive():
            p.terminate()
            p.join(timeout=10)

    hint = (
        "\n         WSL2 note: CUDA IPC is not supported under WDDM, and NCCL's shared-memory"
        "\n         transport relies on it. Try NCCL_SHM_DISABLE=1 in addition to"
        "\n         NCCL_P2P_DISABLE=1, which forces NCCL onto the socket path:"
        "\n           docker compose exec -e NCCL_SHM_DISABLE=1 -e NCCL_DEBUG=INFO \\"
        "\n               sparrow-yolo python3 -m sparrow_yolo.cli doctor -c <config>"
        "\n         If that still fails, train on one GPU (device: \"1\" has no display"
        "\n         attached) - a single RTX PRO 4500 is perfectly capable here."
    )

    if len(results) < len(devices):
        return Check("nccl", False, "NCCL all_reduce hung or crashed." + hint, fatal=True)
    failed = [r for r in results if not r[1]]
    if failed:
        return Check(
            "nccl",
            False,
            "; ".join(f"rank{r[0]}: {r[2]}" for r in failed) + hint,
            fatal=True,
        )
    return Check("nccl", True, f"{len(devices)}-way all_reduce OK", fatal=False)


def _remote_image_share(prepared_root: str | Path, cap: int = 200) -> Tuple[float, int]:
    """Fraction of a sample of prepared images that resolve onto a /mnt path.

    Sampled rather than exhaustive: the answer is the same at 200 files as at
    60,000, and walking the whole tree over the bridge would itself take longer
    than the check is worth.
    """
    root = Path(prepared_root)
    remote = seen = 0
    try:
        for split_dir in sorted(root.glob("*/images/*")):
            if not split_dir.is_dir():
                continue
            with os.scandir(split_dir) as it:
                for entry in it:
                    if seen >= cap:
                        break
                    try:
                        target = Path(entry.path).resolve()
                    except OSError:
                        continue
                    seen += 1
                    if str(target).startswith("/mnt/"):
                        remote += 1
            if seen >= cap:
                break
    except OSError:
        return 0.0, 0
    return (remote / seen if seen else 0.0), seen


def doctor(
    devices: Sequence[int] = (0,),
    *,
    run_nccl: bool = True,
    data_yaml: Optional[str] = None,
    prepared_root: Optional[str] = None,
) -> DoctorReport:
    """Full environment preflight. Fatal checks block training."""
    report = DoctorReport()
    report.gpus = query_gpus()

    report.add("platform", True, f"{'WSL2' if is_wsl() else 'native Linux/other'}, pid {os.getpid()}")

    # --- torch / CUDA -------------------------------------------------------
    try:
        import torch

        arch_list = torch.cuda.get_arch_list()
        report.add("torch", True, f"{torch.__version__} (CUDA {torch.version.cuda})")
        report.add(
            "cuda_available",
            torch.cuda.is_available(),
            f"{torch.cuda.device_count()} device(s) visible to torch",
            fatal=True,
        )
        # Blackwell (RTX PRO / RTX 50xx) is sm_120. A cu126-or-older wheel imports
        # fine and then fails at the first kernel launch with "no kernel image".
        if torch.cuda.is_available():
            caps = {torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())}
            needed = {f"sm_{major}{minor}" for major, minor in caps}
            missing = sorted(s for s in needed if s not in arch_list)
            report.add(
                "gpu_arch_supported",
                not missing,
                (
                    f"torch built for {', '.join(a for a in arch_list if a.startswith('sm_'))}; "
                    + (f"MISSING {', '.join(missing)} - rebuild the image against a newer CUDA wheel"
                       if missing else "matches installed GPUs")
                ),
                fatal=True,
            )
    except ImportError as e:
        report.add("torch", False, f"not importable: {e}", fatal=True)

    # --- GPU inventory ------------------------------------------------------
    if report.gpus:
        for g in report.gpus:
            report.add(
                f"gpu{g.index}",
                True,
                f"{g.name}, {g.total_mb / 1024:.0f} GiB total, {g.free_gb:.1f} GiB free"
                + (" (display attached)" if g.display_attached else ""),
            )
        known = {g.index for g in report.gpus}
        unknown = [d for d in devices if d not in known]
        report.add(
            "devices_present",
            not unknown,
            f"requested {list(devices)}" + (f", missing {unknown}" if unknown else ""),
            fatal=True,
        )
    else:
        report.add("nvidia-smi", False, "not available; cannot inventory GPUs", fatal=False)

    # --- shared memory ------------------------------------------------------
    shm = shm_bytes()
    if shm is not None:
        gb = shm / 1024**3
        # Docker's 64 MiB default kills dataloader workers partway through epoch 1.
        report.add(
            "shm",
            gb >= 8,
            f"/dev/shm is {gb:.1f} GiB"
            + ("" if gb >= 8 else " - too small. Run with --ipc=host --shm-size=32g"),
            fatal=False,
        )

    # --- NCCL environment ---------------------------------------------------
    if len(devices) > 1:
        p2p = os.environ.get("NCCL_P2P_DISABLE")
        report.add(
            "nccl_env",
            p2p == "1",
            f"NCCL_P2P_DISABLE={p2p!r}, NCCL_IB_DISABLE={os.environ.get('NCCL_IB_DISABLE')!r}"
            + ("" if p2p == "1" else " - WSL2 does not support GPU P2P; set NCCL_P2P_DISABLE=1"),
            fatal=False,
        )

    # --- allocator config ---------------------------------------------------
    # expandable_segments needs CUDA virtual memory APIs that WSL2 does not
    # implement. With it set, every torch allocation fails with "CUDA driver
    # error: unknown error" while nvidia-smi, cuInit, cuCtxCreate and torch's
    # device queries all still succeed - so it looks like a broken GPU or driver.
    alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments" in alloc_conf and is_wsl():
        report.add(
            "alloc_conf",
            False,
            f"PYTORCH_CUDA_ALLOC_CONF={alloc_conf!r} contains expandable_segments, which is "
            f"unsupported on WSL2 and makes every tensor allocation fail. Unset it.",
            fatal=True,
        )
    else:
        report.add("alloc_conf", True, f"PYTORCH_CUDA_ALLOC_CONF={alloc_conf or '(unset)'}")

    # --- prepared tree ------------------------------------------------------
    if prepared_root:
        links = link_support(prepared_root)
        can_link = links["symlink"] or links["hardlink"]
        report.add(
            "prepared_links",
            can_link,
            (
                f"{prepared_root}: symlink={links['symlink']} hardlink={links['hardlink']}"
                + ("" if can_link else
                   " - neither works here, so prepare will COPY every image and duplicate "
                   "the dataset. Point PREPARED_PATH at the Linux filesystem.")
            ),
            fatal=False,
        )

        # Linking is only half the question. A prepared tree full of symlinks
        # into /mnt reads over the WSL filesystem bridge on every epoch, which
        # is 36x slower than the Linux filesystem (measured: 277 vs 10,043
        # images/s on the target machine). Not fatal, and often not even the
        # bottleneck while the GPU is the limit - but it becomes one the moment
        # anything else is fixed, and it is invisible without being told.
        remote, sampled = _remote_image_share(prepared_root)
        if sampled:
            report.add(
                "prepared_locality",
                remote < 0.5,
                (f"{remote:.0%} of a {sampled}-image sample resolves onto /mnt "
                 "(the Windows filesystem bridge). Reads there are roughly 36x "
                 "slower than local; copy the dataset to the Linux filesystem if "
                 "the GPU is waiting on the dataloader."
                 if remote >= 0.5 else
                 f"{remote:.0%} of a {sampled}-image sample is on /mnt"),
                fatal=False,
            )

    # --- dataset ------------------------------------------------------------
    if data_yaml:
        p = Path(data_yaml)
        report.add("data_yaml", p.is_file(), str(p) + ("" if p.is_file() else " (missing)"), fatal=True)

    # --- live GPU tests -----------------------------------------------------
    # Plain compute first: if a kernel will not launch, the NCCL failure that
    # follows is a symptom rather than the cause.
    if run_nccl and devices:
        smoke = cuda_smoke_test(devices)
        report.checks.append(smoke)
        if smoke.ok and len(devices) > 1:
            report.checks.append(nccl_selftest(devices))
        elif not smoke.ok:
            report.add("nccl", False, "skipped - plain CUDA compute failed first", fatal=False)

    return report
