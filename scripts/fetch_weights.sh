#!/usr/bin/env bash
# Pre-download pretrained YOLO weights into the image.
#
# Transfer learning starts from COCO-pretrained weights. Left to itself
# Ultralytics fetches them at the start of the run, into the working directory —
# which with `docker run --rm` means a fresh download every single time, and a
# training job that cannot start at all without internet access.
#
# Baking them in makes the first epoch start immediately and the container work
# offline. Failure here is non-fatal: the build still succeeds and Ultralytics
# falls back to downloading at runtime.
#
#   ./scripts/fetch_weights.sh /workspace/weights yolo11l.pt yolo11x.pt
set -uo pipefail

DEST="${1:-/workspace/weights}"
shift || true
MODELS=("$@")
[[ ${#MODELS[@]} -eq 0 ]] && MODELS=("yolo11l.pt")

# Ultralytics publishes pretrained checkpoints as GitHub release assets.
BASE_URL="${WEIGHTS_BASE_URL:-https://github.com/ultralytics/assets/releases/download/v8.3.0}"

mkdir -p "$DEST"
failed=0

for model in "${MODELS[@]}"; do
    target="$DEST/$model"
    if [[ -s "$target" ]]; then
        echo "already present: $target"
        continue
    fi
    echo "fetching $model ..."
    if curl -fsSL --retry 3 --retry-delay 2 -o "$target.part" "$BASE_URL/$model"; then
        mv "$target.part" "$target"
        echo "  -> $target ($(du -h "$target" | cut -f1))"
    else
        rm -f "$target.part"
        echo "  WARNING: could not fetch $model from $BASE_URL"
        echo "           Ultralytics will download it at runtime instead."
        failed=1
    fi
done

if [[ -d "$DEST" ]]; then
    echo "weights cache: $DEST"
    ls -1 "$DEST" 2>/dev/null || true
fi

# Never fail the build over this.
exit 0
