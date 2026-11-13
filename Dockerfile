# CUDA 12.8 is required for Blackwell (RTX PRO / RTX 50-series, compute capability
# sm_120). An older CUDA wheel imports fine and then fails at the first kernel
# launch with "no kernel image is available for execution on the device", so this
# build asserts the arch is present rather than letting that surface mid-training.
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # DO NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here.
    # It reduces fragmentation on native Linux, but it relies on CUDA's virtual
    # memory management APIs (cuMemAddressReserve/cuMemCreate/cuMemMap), which
    # WSL2's GPU virtualisation does not support. The symptom is brutal to
    # diagnose: nvidia-smi works, cuInit/cuCtxCreate/cuMemAlloc all succeed,
    # torch reports the right device count and name, and then the first tensor
    # allocation dies with "CUDA driver error: unknown error".
    #
    # WSL2 exposes GPUs through WDDM, which has no peer-to-peer support. Leaving
    # P2P on is the usual reason a two-GPU run hangs at NCCL init.
    NCCL_P2P_DISABLE=1 \
    NCCL_IB_DISABLE=1 \
    NCCL_DEBUG=WARN \
    # Turn a hung collective into an error instead of an indefinite block.
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    # Each dataloader worker would otherwise spawn one OpenMP thread per core.
    OMP_NUM_THREADS=4 \
    YOLO_CONFIG_DIR=/workspace/.ultralytics

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev git curl wget unzip \
        libgl1 libglib2.0-0 ffmpeg tini \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

COPY scripts/check_torch_arch.py scripts/pin_torch_constraints.py /tmp/

# PyTorch first, from the CUDA 12.8 index, so the CUDA build is the one that
# sticks. TORCH_SPEC is an ARG so a verified pair can be pinned at build time:
#   docker build --build-arg TORCH_SPEC="torch==2.8.0 torchvision==0.23.0" .
ARG TORCH_SPEC="torch torchvision torchaudio"
RUN pip install --no-cache-dir ${TORCH_SPEC} --index-url https://download.pytorch.org/whl/cu128 \
    && python3 /tmp/check_torch_arch.py sm_120

COPY requirements.txt /tmp/requirements.txt

# Nothing in requirements.txt may replace the CUDA 12.8 torch build. Grepping the
# requirements file is not sufficient on its own, because ultralytics depends on
# torch and pip may resolve a PyPI (non-Blackwell) build transitively - hence the
# constraints file, and the re-check afterwards.
RUN python3 /tmp/pin_torch_constraints.py /tmp/constraints.txt \
    && grep -vE '^(torch|torchvision|torchaudio)([=<>!~ ]|$)' /tmp/requirements.txt > /tmp/req.txt \
    && pip install --no-cache-dir -r /tmp/req.txt -c /tmp/constraints.txt \
    && python3 /tmp/check_torch_arch.py sm_120

WORKDIR /workspace

# Source last, and installed with --no-deps, so editing it does not re-resolve
# the whole dependency tree on rebuild.
COPY pyproject.toml README.md /workspace/
COPY src /workspace/src
# Verify the console script landed on PATH. Without this check a stale or
# partial install only shows up much later as
# "exec: sparrow-yolo: executable file not found in $PATH".
RUN pip install --no-cache-dir -e . --no-deps \
    && sparrow-yolo --version \
    && python3 -m sparrow_yolo.cli --version

COPY configs /workspace/configs
COPY tests /workspace/tests
COPY scripts /workspace/scripts
RUN chmod +x /workspace/scripts/*.sh

# Exact resolved versions, so this build can be reproduced later:
#   docker cp sparrow-yolo-workbench:/workspace/requirements.lock.txt .
RUN pip freeze > /workspace/requirements.lock.txt

# YOLO_CONFIG_DIR holds Ultralytics' settings.json, NOT downloaded weights -
# those go to the working directory, so with `docker run --rm` every run would
# re-download them and training could not start without internet access.
# Baking the COCO-pretrained checkpoints in fixes both. Non-fatal if it fails.
ARG PRETRAINED_WEIGHTS="yolo11l.pt"
RUN mkdir -p /workspace/.ultralytics /workspace/runs /workspace/prepared /workspace/datasets \
    && /workspace/scripts/fetch_weights.sh /workspace/weights ${PRETRAINED_WEIGHTS}

# resolve_model_weights() prefers this directory for bare model names.
ENV SPARROW_WEIGHTS_DIR=/workspace/weights

EXPOSE 8080

HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/healthz || exit 1

# tini reaps the DDP ranks and dataloader workers Ultralytics forks; without a
# real init, stopping the container can leave zombies holding GPU memory.
ENTRYPOINT ["/usr/bin/tini", "--"]

# The workbench binds 0.0.0.0 *inside* the container; publish it as
# `-p 127.0.0.1:8080:8080` so it is not reachable from the network. There is no
# authentication and this process launches subprocesses as root, so that loopback
# binding is the only thing between it and anyone who can reach the host.
CMD ["python3", "-m", "sparrow_web"]
