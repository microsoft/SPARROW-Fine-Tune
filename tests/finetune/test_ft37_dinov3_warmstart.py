"""FT-37: a DINOv3 linear head can warm start from an earlier dinov3_linear model."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from .dinov3_helpers import (
    balanced_items,
    make_context,
    make_mean_rgb_embedder,
    snapshot_items,
)

FAST = {"max_epochs": 3, "reference_gate": False}


def new_trainer(embedder=None):
    from sparrow_finetune.pipelines.dinov3_linear import DinoV3LinearTrainer

    embedder = embedder or make_mean_rgb_embedder()
    return DinoV3LinearTrainer(lambda: embedder), embedder


def train_base(tmp_path: Path, labels=("cat", "dog", "fox")):
    trainer, _ = new_trainer()
    items = snapshot_items(balanced_items(labels=labels), tmp_path / "img")
    ctx, _ = make_context(trainer, items, tmp_path / "base", FAST)
    trainer.prepare(ctx)
    trainer.train(ctx)
    trainer.evaluate(ctx)
    return ctx, trainer.export(ctx), items


def base_block(result, ctx, model_id=5):
    return {
        "id": model_id,
        "model_arch": "x",
        "labels": list(ctx.labels),
        "path": Path(result.path),
        "sha256": hashlib.sha256(Path(result.path).read_bytes()).hexdigest(),
    }


def scores(head: dict, x: np.ndarray) -> np.ndarray:
    return ((x - head["mean"]) / head["scale"]) @ head["coef"].T + head["intercept"]


# --------------------------------------------------------------------------- transfer maths


def test_transfer_reproduces_base_scores_under_a_new_scaler():
    from sparrow_finetune.pipelines.dinov3_linear import warm_start_head

    rng = np.random.default_rng(0)
    dim = 6
    base = {
        "mean": rng.normal(size=dim),
        "scale": rng.uniform(0.5, 2, size=dim),
        "coef": rng.normal(size=(3, dim)),
        "intercept": rng.normal(size=3),
    }
    mean, scale = rng.normal(size=dim), rng.uniform(0.5, 2, size=dim)
    # Reordered labels plus one the base never saw.
    coef, intercept, shared = warm_start_head(
        base, ["cat", "dog", "fox"], ["fox", "cat", "owl", "dog"], mean, scale
    )
    assert shared == ["fox", "cat", "dog"]
    x = rng.normal(size=(10, dim))
    new = {"mean": mean, "scale": scale, "coef": coef, "intercept": intercept}
    got, want = scores(new, x), scores(base, x)
    np.testing.assert_allclose(got[:, [0, 1, 3]], want[:, [2, 0, 1]], atol=1e-9)
    np.testing.assert_allclose(got[:, 2], 0.0, atol=1e-12)


def test_transfer_binary_keeps_or_negates_and_refuses_mixed_shapes():
    from sparrow_finetune.pipelines.dinov3_linear import warm_start_head

    rng = np.random.default_rng(1)
    dim = 4
    base = {
        "mean": np.zeros(dim),
        "scale": np.ones(dim),
        "coef": rng.normal(size=(1, dim)),
        "intercept": rng.normal(size=1),
    }
    same, b_same, _ = warm_start_head(
        base, ["a", "b"], ["a", "b"], np.zeros(dim), np.ones(dim)
    )
    swap, b_swap, _ = warm_start_head(
        base, ["a", "b"], ["b", "a"], np.zeros(dim), np.ones(dim)
    )
    np.testing.assert_allclose(swap, -same)
    np.testing.assert_allclose(b_swap, -b_same)
    assert (
        warm_start_head(base, ["a", "b"], ["a", "c"], np.zeros(dim), np.ones(dim))[0]
        is None
    )
    assert (
        warm_start_head(base, ["a", "b"], ["a", "b", "c"], np.zeros(dim), np.ones(dim))[
            0
        ]
        is None
    )


# --------------------------------------------------------------------------- export


def test_export_carries_head_weights_and_model_arch(tmp_path):
    ctx, result, _ = train_base(tmp_path)
    with zipfile.ZipFile(result.path) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        data = zf.read("weights.pt")
    assert manifest["files"]["weights.pt"] == hashlib.sha256(data).hexdigest()
    assert manifest["model_arch"] == manifest["embedder"]["id"]
    state = torch.load(io.BytesIO(data), weights_only=True)
    assert set(state) == {"mean", "scale", "coef", "intercept"}
    assert tuple(state["coef"].shape) == (3, manifest["embedder"]["dim"])


def test_exported_bundle_passes_base_model_import_checks(tmp_path):
    from sparrow_finetune import base_models

    _ctx, result, _ = train_base(tmp_path)
    assert "dinov3_linear" in base_models.WARM_START_PIPELINES
    artifact = Path(result.path).read_bytes()
    with zipfile.ZipFile(io.BytesIO(artifact)) as zf:
        inner = json.loads(zf.read("manifest.json"))
    manifest = {
        "schema": "sparrow-finetune/model-v1",
        "source": "sparrow-fine-tune",
        "pipeline": "dinov3_linear",
        "model_arch": inner["model_arch"],
        "labels": inner["labels"],
        "sha256": hashlib.sha256(artifact).hexdigest(),
    }
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in (
            ("model.zip", artifact),
            ("manifest.json", json.dumps(manifest).encode()),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    facts = base_models.inspect_bundle(buf.getvalue(), None)
    assert facts["pipeline"] == "dinov3_linear"
    assert facts["model_arch"] == inner["model_arch"]


# --------------------------------------------------------------------------- warm start


def test_warm_start_seeds_the_head_with_the_base_scores(tmp_path, monkeypatch):
    from sklearn.linear_model import SGDClassifier

    base_ctx, base_result, items = train_base(tmp_path)
    with zipfile.ZipFile(base_result.path) as zf:
        base = {
            k: v.double().numpy()
            for k, v in torch.load(
                io.BytesIO(zf.read("weights.pt")), weights_only=True
            ).items()
        }

    seen = {}
    original = SGDClassifier.partial_fit

    def spy(self, X, y, classes=None, sample_weight=None):
        if "coef" not in seen:
            seen["coef"] = (
                None if getattr(self, "coef_", None) is None else self.coef_.copy()
            )
            seen["intercept"] = None if seen["coef"] is None else self.intercept_.copy()
        return original(self, X, y, classes=classes, sample_weight=sample_weight)

    monkeypatch.setattr(SGDClassifier, "partial_fit", spy)
    trainer, _ = new_trainer()
    params = dict(FAST, base_model_id=5)
    ctx, rec = make_context(trainer, items, tmp_path / "warm", params)
    ctx.base_model = base_block(base_result, base_ctx)
    trainer.prepare(ctx)
    trainer.train(ctx)

    assert seen["coef"] is not None, (
        "the first partial_fit must start from the base head"
    )
    features = np.load(Path(ctx.workdir) / "features.npy").astype(np.float64)
    trained = json.loads((Path(ctx.workdir) / "train.json").read_text())
    assert trained["warm_start"]["base_model_id"] == 5
    assert trained["warm_start"]["reused_labels"] == ["cat", "dog", "fox"]
    # The seeded head scores raw features exactly as the base head does.
    steps = dict(trainer._load_model(Path(ctx.workdir)).steps)
    sc = steps["scaler"]
    seeded = {
        "mean": sc.mean_,
        "scale": sc.scale_,
        "coef": seen["coef"].astype(np.float64),
        "intercept": seen["intercept"].astype(np.float64),
    }
    np.testing.assert_allclose(
        scores(seeded, features), scores(base, features), atol=1e-3
    )
    assert any("warm start from base model 5" in e[1] for e in rec.events)


def test_cold_start_without_base_model_does_not_seed(tmp_path, monkeypatch):
    from sklearn.linear_model import SGDClassifier

    seen = []
    original = SGDClassifier.partial_fit

    def spy(self, X, y, classes=None, sample_weight=None):
        seen.append(getattr(self, "coef_", None) is None)
        return original(self, X, y, classes=classes, sample_weight=sample_weight)

    monkeypatch.setattr(SGDClassifier, "partial_fit", spy)
    trainer, _ = new_trainer()
    items = snapshot_items(balanced_items(), tmp_path / "img")
    ctx, _ = make_context(trainer, items, tmp_path / "w", FAST)
    trainer.prepare(ctx)
    trainer.train(ctx)
    assert seen[0] is True
    assert (
        json.loads((Path(ctx.workdir) / "train.json").read_text())["warm_start"] is None
    )


def test_base_from_another_embedder_is_rejected_before_embedding(tmp_path):
    base_ctx, base_result, items = train_base(tmp_path)
    trainer, emb = new_trainer(make_mean_rgb_embedder(model_hash="another-embedder"))
    ctx, _ = make_context(
        trainer, items, tmp_path / "warm", dict(FAST, base_model_id=5)
    )
    ctx.base_model = base_block(base_result, base_ctx)
    with pytest.raises(ValueError, match="cannot be reused"):
        trainer.prepare(ctx)
    assert not (Path(ctx.workdir) / "features.npy").exists()


def test_base_without_head_weights_is_rejected(tmp_path):
    base_ctx, base_result, items = train_base(tmp_path)
    old = tmp_path / "old.zip"
    with zipfile.ZipFile(base_result.path) as src, zipfile.ZipFile(old, "w") as dst:
        for name in src.namelist():
            if name != "weights.pt":
                dst.writestr(name, src.read(name))
    trainer, _ = new_trainer()
    ctx, _ = make_context(
        trainer, items, tmp_path / "warm", dict(FAST, base_model_id=5)
    )
    ctx.base_model = dict(base_block(base_result, base_ctx), path=old)
    with pytest.raises(ValueError, match="no head weights"):
        trainer.prepare(ctx)
