"""Regression tests pinning the Phase 1B review findings I1 and I2."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.mark.parametrize("name", ["alpha", "val_fraction", "feature_noise_std"])
def test_i1_huge_integer_float_param_is_a_value_error(name):
    from sparrow_finetune.pipelines.dinov3_linear import DinoV3LinearTrainer

    with pytest.raises(ValueError, match=name):
        DinoV3LinearTrainer(lambda: None).validate_params({name: 10**400})


def _mean_rgb_embedder():
    from sparrow_finetune.embedders import Embedder

    class MeanRGB(Embedder):
        id = "mean-rgb"
        dim = 3
        model_hash = "mean-rgb"

        def embed(self, images):
            return np.asarray(
                [
                    np.asarray(im.convert("RGB"), dtype=np.float32)
                    .reshape(-1, 3)
                    .mean(0)
                    / 255.0
                    for im in images
                ],
                dtype=np.float32,
            )

    return MeanRGB()


def _group_on(side: str, start: int) -> tuple[str, int]:
    from sparrow_finetune.pipelines.dinov3_linear import assign_split

    i = start
    while assign_split(f"g{i}", seed=0, val_fraction=0.2) != side:
        i += 1
    return f"g{i}", i + 1


def test_i2_val_macro_f1_averages_over_every_class(tmp_path: Path):
    from sparrow_finetune.pipelines import JobContext, SnapshotItem
    from sparrow_finetune.pipelines.dinov3_linear import DinoV3LinearTrainer

    # "fox" has training groups only, so it never appears in validation.
    spec = [
        ("cat", (250, 10, 10), "train", 4),
        ("cat", (240, 20, 20), "val", 2),
        ("dog", (10, 250, 10), "train", 4),
        ("dog", (20, 240, 20), "val", 2),
        ("fox", (10, 10, 250), "train", 4),
    ]
    root = tmp_path / "img"
    root.mkdir()
    items, nxt, pos = [], 0, 0
    for label, rgb, side, n in spec:
        group, nxt = _group_on(side, nxt)
        for j in range(n):
            path = root / f"{pos:04d}.png"
            Image.new("RGB", (32, 32), rgb).save(path)
            pos += 1
            items.append(
                SnapshotItem(
                    image_path=path,
                    width=32,
                    height=32,
                    annotations=[{"label": label, "bbox": None}],
                    group=group,
                    name=f"{group}-{j}.png",
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            )

    trainer = DinoV3LinearTrainer(_mean_rgb_embedder)
    events: list = []
    workdir = tmp_path / "work"
    workdir.mkdir()
    ctx = JobContext(
        job_id=1,
        params=trainer.validate_params({"max_epochs": 20}),
        workdir=workdir,
        items=items,
        labels=["cat", "dog", "fox"],
        _emit=lambda s, m, d: events.append((s, m, d)),
        _check_cancelled=lambda: None,
    )
    trainer.prepare(ctx)
    trainer.train(ctx)
    metrics = trainer.evaluate(ctx)
    result = trainer.export(ctx)

    card = json.loads(zipfile.ZipFile(result.path).read("model_card.json"))
    per_class = [v["f1"] for v in card["per_class"].values()]
    expected = sum(per_class) / len(per_class)
    assert metrics["val_macro_f1"] == pytest.approx(expected)
    assert metrics["val_macro_f1"] < 1.0

    epoch_f1 = [
        d["val_macro_f1"]
        for s, _, d in events
        if s == "train" and d and "val_macro_f1" in d
    ]
    assert epoch_f1, "expected per-epoch val_macro_f1 in train events"
    assert max(epoch_f1) <= 2 / 3 + 1e-9
