"""Regression tests for the Phase 1B review backlog items P1B-N1..N10 (FT-6)."""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from .dinov3_helpers import (
    balanced_items,
    make_context,
    make_mean_rgb_embedder,
    snapshot_items,
    synthetic_png,
    write_mean_rgb_onnx,
)

FAST = {"max_epochs": 5, "reference_gate": False}


def new_trainer(embedder=None):
    from sparrow_finetune.pipelines.dinov3_linear import DinoV3LinearTrainer

    embedder = embedder or make_mean_rgb_embedder()
    return DinoV3LinearTrainer(lambda: embedder), embedder


def run_all(trainer, items, workdir: Path, params=None):
    ctx, rec = make_context(trainer, items, workdir, params)
    trainer.prepare(ctx)
    trainer.train(ctx)
    metrics = trainer.evaluate(ctx)
    result = trainer.export(ctx)
    return ctx, metrics, result


def solid(rgb, size=(20, 20)) -> Image.Image:
    return Image.new("RGB", size, rgb)


# --------------------------------------------------------------------------- P1B-N1


def test_n1_noise_without_replicas_is_rejected_naming_both_params():
    trainer, _ = new_trainer()
    with pytest.raises(ValueError) as exc:
        trainer.validate_params({"feature_noise_std": 0.1})
    assert "feature_noise_std" in str(exc.value)
    assert "augment_replicas" in str(exc.value)
    out = trainer.validate_params({"feature_noise_std": 0.1, "augment_replicas": 1})
    assert out["feature_noise_std"] == 0.1


# --------------------------------------------------------------------------- P1B-N2


def test_n2_single_label_snapshot_fails_in_prepare(tmp_path):
    items = snapshot_items(balanced_items(labels=("cat",)), tmp_path / "img")
    trainer, emb = new_trainer()
    ctx, _ = make_context(trainer, items, tmp_path / "work", FAST)
    with pytest.raises(ValueError, match="needs at least 2 labels"):
        trainer.prepare(ctx)
    assert emb.calls == []


# --------------------------------------------------------------------------- P1B-N3


def _closing_embedder():
    emb = make_mean_rgb_embedder()
    emb.closed = 0

    def close():
        emb.closed += 1

    emb.close = close
    return emb


def test_n3_embedder_interface_has_noop_close_and_context_manager():
    emb = make_mean_rgb_embedder()
    emb.close()
    with emb as inner:
        assert inner is emb


def test_n3_trainer_closes_embedder_after_prepare(tmp_path):
    items = snapshot_items(balanced_items(), tmp_path / "img")
    emb = _closing_embedder()
    trainer, _ = new_trainer(emb)
    ctx, _ = make_context(trainer, items, tmp_path / "work", FAST)
    trainer.prepare(ctx)
    assert emb.closed == 1


def test_n3_trainer_closes_embedder_when_prepare_fails(tmp_path):
    items = snapshot_items(balanced_items(labels=("cat",)), tmp_path / "img")
    emb = _closing_embedder()
    trainer, _ = new_trainer(emb)
    ctx, _ = make_context(trainer, items, tmp_path / "work", FAST)
    with pytest.raises(ValueError):
        trainer.prepare(ctx)
    assert emb.closed == 1


def test_n3_engine_embedder_closes_only_its_own_client():
    import httpx

    from sparrow_finetune.embedders import EngineEmbedder

    own = EngineEmbedder("http://engine.test", "m")
    client = own._client
    own.close()
    assert client.is_closed

    passed = httpx.Client()
    try:
        with EngineEmbedder("http://engine.test", "m", client=passed):
            pass
        assert not passed.is_closed
    finally:
        passed.close()


def _fake_engine_handler(request):
    import io

    import httpx

    results = []
    body = request.read()
    boundary = request.headers["content-type"].split("boundary=")[1].encode()
    for i, part in enumerate(
        p for p in body.split(b"--" + boundary) if b"\r\n\r\n" in p
    ):
        payload = part.split(b"\r\n\r\n", 1)[1].rsplit(b"\r\n", 1)[0]
        with Image.open(io.BytesIO(payload)) as im:
            vec = (
                np.asarray(im.convert("RGB"), dtype=np.float64).mean(axis=(0, 1))
                / 255.0
            )
        results.append({"index": i, "embedding": [float(v) for v in vec]})
    return httpx.Response(200, json={"results": results})


def test_n3_engine_embedder_reopens_its_client_after_close():
    import httpx

    from sparrow_finetune.embedders import EngineEmbedder

    emb = EngineEmbedder("http://engine.test", "m")
    emb.close()
    # After close a later job (the same instance served by ``lambda: embedder``) still works.
    emb._new_client = lambda: httpx.Client(
        transport=httpx.MockTransport(_fake_engine_handler)
    )
    out = emb.embed([solid((255, 0, 0))])
    assert out.shape == (1, 3)
    emb.close()


# --------------------------------------------------------------------------- P1B-N4


def _token_model(path: Path, *, with_pooler: bool) -> Path:
    """``pixel_values`` [N,3,H,W] -> ``last_hidden_state`` [N,H*W,3] (+ ``pooler_output`` [N,3])."""
    import onnx
    from onnx import TensorProto, helper

    shape = helper.make_tensor("shape", TensorProto.INT64, [3], [0, 3, -1])
    nodes = [
        helper.make_node("Reshape", ["pixel_values", "shape"], ["flat"]),
        helper.make_node("Transpose", ["flat"], ["last_hidden_state"], perm=[0, 2, 1]),
    ]
    outputs = [
        helper.make_tensor_value_info(
            "last_hidden_state", TensorProto.FLOAT, ["N", "T", 3]
        )
    ]
    if with_pooler:
        nodes += [
            helper.make_node("GlobalAveragePool", ["pixel_values"], ["gap"]),
            helper.make_node("Flatten", ["gap"], ["pooler_output"], axis=1),
        ]
        outputs.append(
            helper.make_tensor_value_info("pooler_output", TensorProto.FLOAT, ["N", 3])
        )
    graph = helper.make_graph(
        nodes,
        "tokens",
        [
            helper.make_tensor_value_info(
                "pixel_values", TensorProto.FLOAT, ["N", 3, "H", "W"]
            )
        ],
        outputs,
        initializer=[shape],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _two_tone(left, right, size=16) -> Image.Image:
    im = Image.new("RGB", (size, size), right)
    im.paste(left, (0, 0, size // 2, size))
    return im


def test_n4_onnx_embedder_prefers_pooled_output_over_last_hidden_state(tmp_path):
    from sparrow_finetune.embedders import OnnxEmbedder

    emb = OnnxEmbedder(
        _token_model(tmp_path / "m.onnx", with_pooler=True), input_size=16
    )
    out = emb.embed([_two_tone((255, 0, 0), (0, 0, 255))])
    assert out.shape == (1, 3)
    assert emb.dim == 3
    expected = (np.array([0.5, 0.0, 0.5], dtype=np.float32) - MEAN) / STD
    np.testing.assert_allclose(out[0], expected, atol=2e-2)


def test_n4_onnx_embedder_takes_cls_token_of_rank3_output(tmp_path):
    from sparrow_finetune.embedders import OnnxEmbedder

    emb = OnnxEmbedder(
        _token_model(tmp_path / "m.onnx", with_pooler=False), input_size=16
    )
    out = emb.embed([_two_tone((255, 0, 0), (0, 0, 255)), solid((0, 255, 0), (16, 16))])
    assert out.shape == (2, 3)
    assert emb.dim == 3
    np.testing.assert_allclose(
        out[0], (np.array([1, 0, 0], np.float32) - MEAN) / STD, atol=1e-3
    )
    np.testing.assert_allclose(
        out[1], (np.array([0, 1, 0], np.float32) - MEAN) / STD, atol=1e-3
    )


# --------------------------------------------------------------------------- P1B-N5


def _multi_annotation_items(tmp_path: Path):
    """Each image carries three annotations (two boxes + whole image) of one label."""
    from sparrow_finetune.pipelines import SnapshotItem

    out = []
    root = tmp_path / "img"
    root.mkdir(parents=True, exist_ok=True)
    for pos, syn in enumerate(balanced_items()):
        path = root / f"{pos:04d}.png"
        path.write_bytes(syn.png)
        out.append(
            SnapshotItem(
                image_path=path,
                width=32,
                height=24,
                annotations=[
                    {"label": syn.label, "bbox": [6, 4, 16, 12]},
                    {"label": syn.label, "bbox": [0, 0, 10, 10]},
                    {"label": syn.label, "bbox": None},
                ],
                group=syn.group,
                name=syn.name,
                sha256=hashlib.sha256(syn.png).hexdigest(),
            )
        )
    return out


def test_n5_each_image_is_decoded_once(tmp_path, monkeypatch):
    from sparrow_finetune.pipelines import dinov3_linear

    items = _multi_annotation_items(tmp_path)
    opened: Counter = Counter()
    real_open = Image.open

    def counting_open(fp, *a, **kw):
        # prepare decodes from the bytes it hashed for the embedding cache; key by content,
        # since freed BytesIO objects reuse addresses and would collide on str(fp).
        key = hashlib.sha256(fp.getvalue()).hexdigest() if hasattr(fp, "getvalue") else str(fp)
        opened[key] += 1
        return real_open(fp, *a, **kw)

    monkeypatch.setattr(dinov3_linear.Image, "open", counting_open)
    trainer, emb = new_trainer()
    ctx, _ = make_context(trainer, items, tmp_path / "work", FAST)
    trainer.prepare(ctx)
    assert sum(emb.calls) == 3 * len(items)
    assert set(opened.values()) == {1}, opened
    assert len(opened) == len(items)
    assert np.load(ctx.workdir / "features.npy").shape == (3 * len(items), 3)


def test_n5_crops_are_resized_to_embedder_input_size_before_batching(tmp_path):
    from sparrow_finetune.embedders import OnnxEmbedder

    class SpyOnnx(OnnxEmbedder):
        def embed(self, images):
            images = list(images)
            self.seen = getattr(self, "seen", []) + [im.size for im in images]
            return super().embed(images)

    emb = SpyOnnx(write_mean_rgb_onnx(tmp_path / "e.onnx"), input_size=8)
    items = _multi_annotation_items(tmp_path)
    trainer, _ = new_trainer(emb)
    ctx, _ = make_context(trainer, items, tmp_path / "work", FAST)
    trainer.prepare(ctx)
    assert set(emb.seen) == {(8, 8)}


def test_n5_early_resize_gives_identical_onnx_features(tmp_path):
    from sparrow_finetune.embedders import OnnxEmbedder

    emb = OnnxEmbedder(write_mean_rgb_onnx(tmp_path / "e.onnx"), input_size=16)
    crop = Image.open(_png(tmp_path)).convert("RGB").crop((3, 2, 29, 21))
    np.testing.assert_array_equal(
        emb.embed([crop]), emb.embed([emb.preprocess_crop(crop)])
    )


def _png(tmp_path: Path) -> Path:
    path = tmp_path / "x.png"
    path.write_bytes(synthetic_png("cat", 3, [6, 4, 16, 12]))
    return path


# --------------------------------------------------------------------------- P1B-N6


def test_n6_engine_embedder_carries_input_preprocessing():
    from sparrow_finetune.embedders import EngineEmbedder

    emb = EngineEmbedder("http://engine.test", "m")
    assert emb.input_size == 224
    np.testing.assert_allclose(emb.mean, MEAN, atol=1e-6)
    np.testing.assert_allclose(emb.std, STD, atol=1e-6)
    custom = EngineEmbedder(
        "http://engine.test", "m", input_size=256, mean=(0.5,) * 3, std=(0.25,) * 3
    )
    assert (custom.input_size, list(custom.mean), list(custom.std)) == (
        256,
        [0.5] * 3,
        [0.25] * 3,
    )
    emb.close()
    custom.close()


def test_n6_manifest_input_block_filled_with_engine_embedder(tmp_path):
    import httpx

    from sparrow_finetune.embedders import EngineEmbedder

    client = httpx.Client(transport=httpx.MockTransport(_fake_engine_handler))
    emb = EngineEmbedder("http://engine.test", "dinov3-test", client=client)
    items = snapshot_items(balanced_items(), tmp_path / "img")
    trainer, _ = new_trainer(emb)
    _ctx, _m, result = run_all(trainer, items, tmp_path / "work", FAST)
    with zipfile.ZipFile(result.path) as zf:
        manifest = json.loads(zf.read("manifest.json"))
    assert manifest["input"]["size"] == 224
    np.testing.assert_allclose(manifest["input"]["mean"], MEAN, atol=1e-6)
    np.testing.assert_allclose(manifest["input"]["std"], STD, atol=1e-6)
    assert not client.is_closed
    client.close()


# --------------------------------------------------------------------------- P1B-N8


def test_n8_missing_embedder_fails_prepare_first_naming_env_var(tmp_path):
    from sparrow_finetune.embedders import make_embedder
    from sparrow_finetune.pipelines.dinov3_linear import DinoV3LinearTrainer

    # The data would also fail (class_floor); the embedder error must come first.
    items = snapshot_items(balanced_items(), tmp_path / "img")
    trainer = DinoV3LinearTrainer(lambda: make_embedder(""))
    ctx, _ = make_context(trainer, items, tmp_path / "work", {**FAST, "class_floor": 1000})
    with pytest.raises(ValueError, match="SPARROW_FT_EMBEDDER"):
        trainer.prepare(ctx)


# --------------------------------------------------------------------------- P1B-N9


@pytest.mark.parametrize("value", [-0.1, 0.5, 0.7, "x"])
def test_n9_test_fraction_range_is_validated(value):
    trainer, _ = new_trainer()
    with pytest.raises(ValueError, match="test_fraction"):
        trainer.validate_params({"test_fraction": value})


def test_n9_test_fraction_default_keeps_two_way_split():
    from sparrow_finetune.pipelines.dinov3_linear import assign_split

    keys = [f"k{i}" for i in range(200)]
    default = [assign_split(k, seed=0, val_fraction=0.2) for k in keys]
    explicit = [
        assign_split(k, seed=0, val_fraction=0.2, test_fraction=0.0) for k in keys
    ]
    assert default == explicit
    assert set(default) == {"train", "val"}
    three = [assign_split(k, seed=0, val_fraction=0.2, test_fraction=0.2) for k in keys]
    # val assignment is unchanged; test is carved from the former train side only
    assert [s == "val" for s in three] == [s == "val" for s in default]
    assert "test" in three


def test_n9_test_split_is_group_disjoint_and_reported(tmp_path):
    items = snapshot_items(
        balanced_items(train_groups=8, val_groups=3), tmp_path / "img"
    )
    trainer, _ = new_trainer()
    params = {**FAST, "test_fraction": 0.3}
    ctx, metrics, _ = run_all(trainer, items, tmp_path / "work", params)
    prepared = json.loads((ctx.workdir / "prepare.json").read_text())
    split = prepared["split"]
    assert set(split) == {"train", "val", "test"}
    assert split["test"]["crops"] > 0
    groups = [set(split[s]["groups"]) for s in ("train", "val", "test")]
    assert (
        not (groups[0] & groups[2])
        and not (groups[1] & groups[2])
        and not (groups[0] & groups[1])
    )
    assert {"test_macro_f1", "test_accuracy"} <= set(metrics)
    assert 0.0 <= metrics["test_macro_f1"] <= 1.0
    assert metrics["test_crops"] == split["test"]["crops"]
    assert metrics["train_crops"] + metrics["val_crops"] + metrics["test_crops"] == len(
        items
    )


def test_n9_default_reports_no_test_metrics(tmp_path):
    items = snapshot_items(balanced_items(), tmp_path / "img")
    trainer, _ = new_trainer()
    ctx, metrics, _ = run_all(trainer, items, tmp_path / "work", FAST)
    assert not any(k.startswith("test_") for k in metrics)
    assert set(json.loads((ctx.workdir / "prepare.json").read_text())["split"]) == {
        "train",
        "val",
    }


# --------------------------------------------------------------------------- P1B-N10


def test_n10_identical_runs_give_byte_identical_bundles(tmp_path, monkeypatch):
    import time as time_mod

    syn = balanced_items()
    items = snapshot_items(syn, tmp_path / "img")
    bundles = []
    for run, fake_now in enumerate((1_000_000_000.0, 1_700_000_000.0)):
        # Pretend the runs are years apart: wall-clock zip timestamps would differ.
        monkeypatch.setattr(time_mod, "time", lambda now=fake_now: now)
        trainer, _ = new_trainer()
        _ctx, _m, result = run_all(trainer, items, tmp_path / f"work{run}", FAST)
        bundles.append(Path(result.path).read_bytes())
    assert bundles[0] == bundles[1]
    with zipfile.ZipFile(tmp_path / "work0" / "dinov3_linear_bundle.zip") as zf:
        assert {i.date_time for i in zf.infolist()} == {(1980, 1, 1, 0, 0, 0)}


@pytest.mark.parametrize(
    ("embedder", "claims_dinov3"), [("", False), ("onnx:/models/dinov3.onnx", True)]
)
def test_n8_worker_without_embedder_claims_training_but_not_preparation(
    tmp_path, monkeypatch, embedder, claims_dinov3
):
    from sparrow_finetune import workers
    from sparrow_finetune.config import Settings
    from sparrow_finetune.pipelines import Registry, register_builtin_pipelines

    seen: dict = {}

    class FakeQueue:
        def __init__(self, url):
            pass

        def register_worker(self, worker_id, resources, pipelines):
            seen["registered"] = set(pipelines)

        def reap_expired(self):
            return 0

        def claim(
            self, worker_id, resources, lease_seconds, pipelines=None,
            max_running_per_user=None,
        ):
            seen["pipelines"] = set(pipelines)
            return None

    monkeypatch.setattr(workers, "JobQueue", FakeQueue)
    settings = Settings(
        database_url="postgresql://u@h/db", data_dir=tmp_path, embedder=embedder
    )
    reg = Registry()
    register_builtin_pipelines(reg, settings)
    assert workers.run_once(settings, "w", {"cpu"}, registry=reg) is False
    assert "dinov3_linear" in seen["pipelines"]
    assert ("dinov3_features" in seen["pipelines"]) is claims_dinov3
    # The worker advertises the same pipelines it claims, so the queue reason is accurate.
    assert seen["registered"] == seen["pipelines"]
    assert {"yolo_detector", "conventional_classifier"} <= seen["pipelines"]
    # The API still lists the pipeline so jobs can be queued for a worker that has one.
    assert reg.get("dinov3_linear") is not None
