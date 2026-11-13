"""Near-duplicate detection between train and val crops (FT-5).

A 64-bit difference hash (dHash) per crop; a val crop counts as a near-duplicate when its hash is
within ``max_distance`` bits (Hamming distance) of any train crop's hash. Near-duplicates across the
split inflate validation metrics, so the classifier pipelines report how many there are.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

DEFAULT_MAX_DISTANCE = 4
# Val rows x train columns of the XOR matrix processed at once: 4M uint64 (32 MiB) per tile, so
# memory stays fixed however many train hashes there are.
_VAL_TILE = 128
_TRAIN_TILE = 32768


def dhash(image: Image.Image) -> int:
    """64-bit dHash: grayscale, resize to 9x8, bit set where a pixel is brighter than its right neighbour."""
    small = image.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
    px = np.asarray(small, dtype=np.int16)
    bits = (px[:, :-1] > px[:, 1:]).ravel()
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def _popcount_swar(x: np.ndarray) -> np.ndarray:
    """Bit count per uint64 (SWAR) for numpy < 2.0, which lacks ``np.bitwise_count``; overwrites ``x``."""
    x -= (x >> np.uint64(1)) & np.uint64(0x5555555555555555)
    x[...] = (x & np.uint64(0x3333333333333333)) + (
        (x >> np.uint64(2)) & np.uint64(0x3333333333333333)
    )
    x += x >> np.uint64(4)
    x &= np.uint64(0x0F0F0F0F0F0F0F0F)
    x *= np.uint64(0x0101010101010101)
    x >>= np.uint64(56)
    return x


_popcount = getattr(np, "bitwise_count", None) or _popcount_swar


def count_near_duplicates(val: np.ndarray, train: np.ndarray, max_distance: int) -> int:
    """Number of ``val`` hashes within Hamming distance ``max_distance`` of any ``train`` hash."""
    val = np.ascontiguousarray(val, dtype=np.uint64)
    train = np.ascontiguousarray(train, dtype=np.uint64)
    if val.size == 0 or train.size == 0:
        return 0
    count = 0
    for v0 in range(0, len(val), _VAL_TILE):
        rows = val[v0 : v0 + _VAL_TILE, None]
        near = np.zeros(len(rows), dtype=bool)
        for t0 in range(0, len(train), _TRAIN_TILE):
            xor = np.bitwise_xor(rows, train[None, t0 : t0 + _TRAIN_TILE])
            near |= (_popcount(xor) <= max_distance).any(axis=1)
        count += int(near.sum())
    return count


def near_duplicate_report(
    ctx: Any,
    hashes: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    metrics: dict,
    unit: str = "crop",
) -> dict:
    """Add ``near_dup_val_<unit>s`` / ``near_dup_val_fraction`` to ``metrics``; warn when any exist.

    ``unit`` is ``"crop"`` for the classifiers and ``"image"`` for the detector (FT-38). Returns the
    model-card block ``{"max_distance", "val_<unit>s", "val_fraction"}``.
    """
    max_distance = int(ctx.params.get("near_dup_max_distance", DEFAULT_MAX_DISTANCE))
    n_val = int(np.count_nonzero(val_mask))
    count = count_near_duplicates(hashes[val_mask], hashes[train_mask], max_distance)
    fraction = count / n_val if n_val else 0.0
    metrics[f"near_dup_val_{unit}s"] = float(count)
    metrics["near_dup_val_fraction"] = float(fraction)
    if count:
        ctx.emit(
            "evaluate",
            f"warning: {count} of {n_val} val {unit}(s) are near-duplicates of a train {unit} "
            f"(dHash Hamming distance <= {max_distance}); validation metrics may be optimistic",
            **{f"near_dup_val_{unit}s": count, "near_dup_val_fraction": fraction},
        )
    return {"max_distance": max_distance, f"val_{unit}s": count, "val_fraction": fraction}
