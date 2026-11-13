#!/usr/bin/env bash
# Run a one-off container on WSL2 with the GPUs attached.
#
#   ./scripts/run.sh doctor                # environment check, then exit
#   ./scripts/run.sh shell                 # interactive shell
#   ./scripts/run.sh sparrow-yolo audit -c configs/example_experiment.yaml
#
# For the workbench itself use `docker compose up -d` — that publishes the port,
# mounts everything, and restarts the app if it dies. This script is for the
# throwaway cases where you want a container, no port, and no supervision.
#
# Reads the same .env as docker compose. Override the dataset library with
#   DATASETS_HOST_ROOT=/path/to/datasets ./scripts/run.sh shell
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-sparrow-yolo-workbench}"

# Same variable docker-compose.yml uses, so one .env drives both. The old
# DATASET_HOST_PATH/DATASET_NAME pair pointed at a single dataset; this is the
# directory holding all of them, matching how projects are created in the UI.
[[ -f "$REPO_ROOT/.env" ]] && set -a && . "$REPO_ROOT/.env" && set +a
DATASETS_HOST_ROOT="${DATASETS_HOST_ROOT:-/home/$USER/datasets}"

if [[ ! -d "$DATASETS_HOST_ROOT" ]]; then
    echo "Dataset library not found: $DATASETS_HOST_ROOT" >&2
    echo "Set DATASETS_HOST_ROOT in .env, or pass it inline:" >&2
    echo "  DATASETS_HOST_ROOT=/path/to/datasets ./scripts/run.sh shell" >&2
    exit 1
fi

# Keep training data on the Linux filesystem. Reading from /mnt/* goes over the
# WSL bridge: measured at 277 images/s against 10,043/s locally, which is the
# usual reason the GPUs sit idle waiting on the dataloader.
case "$DATASETS_HOST_ROOT" in
    /mnt/*) echo "WARNING: $DATASETS_HOST_ROOT is on a Windows drive. Expect slow data" >&2
            echo "         loading; copying into the WSL filesystem is 3-5x faster." >&2 ;;
esac

mkdir -p "$REPO_ROOT/runs" "$REPO_ROOT/prepared" "$REPO_ROOT/annotations"

COMMON_ARGS=(
    --gpus all
    --ipc=host                      # DDP + dataloader workers need real shared memory
    --shm-size=32g                  # Docker's 64 MiB default kills workers in epoch 1
    --ulimit memlock=-1
    --ulimit stack=67108864
    --ulimit nofile=65536:65536
    --init                          # reap forked DDP ranks
    -e NCCL_P2P_DISABLE=1           # WDDM/WSL2 has no GPU peer-to-peer
    -e NCCL_IB_DISABLE=1
    -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    # NOT setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True - it needs
    # CUDA virtual memory APIs that WSL2 does not support, and every tensor
    # allocation then fails with "CUDA driver error: unknown error".
    -e OMP_NUM_THREADS=4
    -e PYTHONPATH=/workspace/src
    -v "$DATASETS_HOST_ROOT:/workspace/datasets:ro"
    -v "$REPO_ROOT/runs:/workspace/runs"
    -v "$REPO_ROOT/prepared:/workspace/prepared"   # must be mounted, or --rm destroys it
    -v "$REPO_ROOT/annotations:/workspace/annotations"
    -v "$REPO_ROOT/configs:/workspace/configs"
    -v "$REPO_ROOT/src:/workspace/src"             # live source; no rebuild to edit code
    -v sparrow-ultralytics-cache:/workspace/.ultralytics
)

case "${1:-doctor}" in
    doctor)
        exec docker run --rm "${COMMON_ARGS[@]}" "$IMAGE" sparrow-yolo doctor
        ;;
    shell)
        exec docker run -it --rm "${COMMON_ARGS[@]}" "$IMAGE" bash
        ;;
    *)
        exec docker run -it --rm "${COMMON_ARGS[@]}" "$IMAGE" "$@"
        ;;
esac
