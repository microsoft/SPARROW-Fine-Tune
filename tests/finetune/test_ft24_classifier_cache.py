"""FT-24: content-addressed DINOv3 embedding cache, so retraining skips feature extraction."""

from __future__ import annotations

import json

import numpy as np

from .dinov3_helpers import (
    Recorder,
    balanced_items,
    make_context,
    make_mean_rgb_embedder,
    snapshot_items,
    synthetic_png,
)


def prepare_once(tmp_path, run, emb, items_dir, *, cache_dir=None, params=None):
    from sparrow_finetune.pipelines.dinov3_linear import DinoV3LinearTrainer

    t = DinoV3LinearTrainer(lambda: emb, cache_dir=cache_dir or tmp_path / "cache")
    items = snapshot_items(balanced_items(), items_dir)
    rec = Recorder()
    ctx, _ = make_context(t, items, tmp_path / f"work{run}", params, recorder=rec)
    t.prepare(ctx)
    return ctx, rec, t


def n_embedded(emb) -> int:
    return sum(emb.calls)


def test_second_run_on_same_snapshot_embeds_nothing(tmp_path):
    emb1 = make_mean_rgb_embedder()
    ctx1, rec1, _ = prepare_once(tmp_path, 1, emb1, tmp_path / "img")
    n = n_embedded(emb1)
    assert n > 0
    emb2 = make_mean_rgb_embedder()
    ctx2, rec2, t = prepare_once(tmp_path, 2, emb2, tmp_path / "img")
    assert n_embedded(emb2) == 0
    f1 = np.load(ctx1.workdir / "features.npy")
    f2 = np.load(ctx2.workdir / "features.npy")
    np.testing.assert_array_equal(f1, f2)
    stats = json.loads((ctx2.workdir / "prepare.json").read_text())["embedding_cache"]
    assert stats["hits"] == n and stats["misses"] == 0
    (event,) = [m for s, msg, m in rec2.events if msg == "embedding cache"]
    assert event["hits"] == n and event["misses"] == 0
    t.train(ctx2)
    metrics = t.evaluate(ctx2)
    assert metrics["embedding_cache_hits"] == float(n)
    assert metrics["embedding_cache_misses"] == 0.0


def test_changed_image_bytes_miss(tmp_path):
    emb1 = make_mean_rgb_embedder()
    prepare_once(tmp_path, 1, emb1, tmp_path / "img")
    # Same paths, one image rewritten with different pixels.
    items = snapshot_items(balanced_items(), tmp_path / "img")
    target = items[0]
    target.image_path.write_bytes(synthetic_png("dog", 999, None))
    from sparrow_finetune.pipelines.dinov3_linear import DinoV3LinearTrainer

    emb2 = make_mean_rgb_embedder()
    t = DinoV3LinearTrainer(lambda: emb2, cache_dir=tmp_path / "cache")
    ctx, _ = make_context(t, items, tmp_path / "work2")
    t.prepare(ctx)
    assert n_embedded(emb2) == len(target.annotations)
    stats = json.loads((ctx.workdir / "prepare.json").read_text())["embedding_cache"]
    assert stats["misses"] == len(target.annotations)


def test_changed_embedder_id_or_weights_miss(tmp_path):
    emb1 = make_mean_rgb_embedder()
    prepare_once(tmp_path, 1, emb1, tmp_path / "img")
    n = n_embedded(emb1)
    emb2 = make_mean_rgb_embedder(model_hash="fake-mean-rgb-v2")
    prepare_once(tmp_path, 2, emb2, tmp_path / "img")
    assert n_embedded(emb2) == n
    emb3 = make_mean_rgb_embedder()
    emb3.id = "another-embedder"
    prepare_once(tmp_path, 3, emb3, tmp_path / "img")
    assert n_embedded(emb3) == n


def test_corrupt_entry_is_recomputed(tmp_path):
    emb1 = make_mean_rgb_embedder()
    prepare_once(tmp_path, 1, emb1, tmp_path / "img")
    entries = sorted((tmp_path / "cache").rglob("*.npy"))
    assert entries and not list((tmp_path / "cache").rglob("*.tmp"))
    entries[0].write_bytes(b"not a numpy file")
    entries[1].write_bytes(b"")
    emb2 = make_mean_rgb_embedder()
    ctx, _, _ = prepare_once(tmp_path, 2, emb2, tmp_path / "img")
    assert n_embedded(emb2) == 2
    stats = json.loads((ctx.workdir / "prepare.json").read_text())["embedding_cache"]
    assert stats["corrupt"] == 2 and stats["writes"] == 2
    # Rewritten: a third run is all hits.
    emb3 = make_mean_rgb_embedder()
    prepare_once(tmp_path, 3, emb3, tmp_path / "img")
    assert n_embedded(emb3) == 0


def test_cache_can_be_disabled(tmp_path):
    emb1 = make_mean_rgb_embedder()
    ctx, _, _ = prepare_once(
        tmp_path, 1, emb1, tmp_path / "img", params={"embedding_cache": False}
    )
    assert not (tmp_path / "cache").exists()
    stats = json.loads((ctx.workdir / "prepare.json").read_text())["embedding_cache"]
    assert stats == {"enabled": False}


def test_cache_root_derived_from_service_data_dir(tmp_path):
    from sparrow_finetune.pipelines.dinov3_linear import DinoV3LinearTrainer

    emb = make_mean_rgb_embedder()
    t = DinoV3LinearTrainer(lambda: emb)
    items = snapshot_items(balanced_items(), tmp_path / "img")
    ctx, _ = make_context(
        t, items, tmp_path / "data" / "jobs" / "7" / "attempt-1" / "work"
    )
    t.prepare(ctx)
    assert list((tmp_path / "data" / "cache" / "embeddings").rglob("*.npy"))


def test_entries_are_float32_npy_loadable_without_pickle(tmp_path):
    emb = make_mean_rgb_embedder()
    prepare_once(tmp_path, 1, emb, tmp_path / "img")
    for path in (tmp_path / "cache").rglob("*.npy"):
        v = np.load(path, allow_pickle=False)
        assert v.dtype == np.dtype("<f4") and v.shape == (3,)
