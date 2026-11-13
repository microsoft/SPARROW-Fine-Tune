"""Classifier-pipeline backlog items FT-5, FT-13, FT-15, FT-16 and FT-17 (docs/review/implement-backlog-sweep/state.md).

- FT-5: a 64-bit dHash per training crop; evaluate reports val crops that nearly duplicate a train crop.
- FT-13: the conventional classifier loads timm weights from ``<weights_dir>/timm/<model>/`` with no network.
- FT-15: both classifier exports use the dynamo ONNX exporter and keep a dynamic batch dimension.
- FT-16: an optional sparrow-engine stock model is scored on the matching val crops.
- FT-17: the dinov3 embedder is built (and a malformed spec fails) before any crop is read.
"""

from __future__ import annotations

import io
import json
import socket
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from .dinov3_helpers import (
    CLASS_COLOURS,
    SYN_SIZE,
    SynItem,
    balanced_items,
    bundle_json,
    make_context,
    make_mean_rgb_embedder,
    read_bundle,
    snapshot_items,
)

DINO_FAST = {"max_epochs": 5, "reference_gate": False}
CONV_BASE = {
    "model": "resnet18",
    "pretrained": False,
    "imgsz": 32,
    "epochs": 2,
    "warmup_epochs": 1,
    "patience": 0,
    "batch": 16,
    "device": "cpu",
}
STOCK_NAME = "tiny-stock"
STOCK_ID = "tiny-stock-v1"


# --------------------------------------------------------------------------- data helpers


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def textured_items(*, duplicate: bool) -> list[SynItem]:
    """Noisy class-coloured images (distinct dHashes), every item grouped so its split is fixed.

    With ``duplicate`` the first val image becomes a byte copy of the first train image of its label.
    """
    from sparrow_finetune.pipelines.dinov3_linear import assign_split

    syn = balanced_items(
        train_groups=3, val_groups=2, images_per_group=2, ungrouped_per_label=0
    )
    out = []
    for k, it in enumerate(syn):
        rng = np.random.default_rng(5000 + k)
        base = np.asarray(CLASS_COLOURS[it.label], dtype=np.int64)
        arr = np.clip(
            base + rng.integers(-40, 41, size=(SYN_SIZE[1], SYN_SIZE[0], 3)), 0, 255
        )
        out.append(
            SynItem(it.name, it.label, it.group, None, _png(arr.astype(np.uint8)))
        )
    if duplicate:
        side = [assign_split(it.group, seed=0, val_fraction=0.2) for it in out]
        v = side.index("val")
        t = next(
            i
            for i, s in enumerate(side)
            if s == "train" and out[i].label == out[v].label
        )
        out[v] = SynItem(out[v].name, out[v].label, out[v].group, None, out[t].png)
    return out


def write_stock_package(
    root: Path,
    name: str = STOCK_NAME,
    labels=("Cat", "Dog", "Bird"),
    replace: dict | None = None,
) -> Path:
    """A sparrow-engine classifier package: mean RGB (after ImageNet normalisation) -> class logits.

    The graph input is fixed at 16x20 (HxW), so a loader that ignored ``input_size = [20, 16]``
    (width, height) fails in onnxruntime.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    pkg = root / name
    (pkg / "1").mkdir(parents=True, exist_ok=True)
    n = len(labels)
    w = np.zeros((3, n), dtype=np.float32)
    for i in range(min(3, n)):
        w[i, i] = 5.0
    graph = helper.make_graph(
        [
            helper.make_node("GlobalAveragePool", ["images"], ["pooled"]),
            helper.make_node("Flatten", ["pooled"], ["flat"], axis=1),
            helper.make_node("MatMul", ["flat", "w"], ["mm"]),
            helper.make_node("Add", ["mm", "b"], ["logits"]),
        ],
        "tiny_stock",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, ["N", 3, 16, 20])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["N", n])],
        initializer=[
            numpy_helper.from_array(w, "w"),
            numpy_helper.from_array(np.zeros(n, dtype=np.float32), "b"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(pkg / "1" / "model.onnx"))
    (pkg / "labels.txt").write_text(
        "".join(f"{lab},{i}\n" for i, lab in enumerate(labels))
    )
    manifest = f"""[model]
id = "{STOCK_ID}"
format = "onnx"
file = "1/model.onnx"

[preprocessing]
method = "resize"
input_size = [20, 16]
layout = "nchw"
normalization = "imagenet"

[postprocessing]
method = "softmax"

[labels]
file = "labels.txt"
format = "name_index_csv"

[inference]
strategy = "single"

[inference.trt]
mode = "on_demand"
"""
    for old, new in (replace or {}).items():
        assert old in manifest, old
        manifest = manifest.replace(old, new)
    (pkg / "manifest.toml").write_text(manifest)
    return pkg


class _ExportSpy:
    def __init__(self, real):
        self.real, self.calls = real, []

    def __call__(self, *args, **kwargs):
        self.calls.append(kwargs)
        return self.real(*args, **kwargs)


def _dino_run(tmp: Path, syn, params=None, weights_dir=None, spy=False):
    import torch

    from sparrow_finetune.pipelines.dinov3_linear import DinoV3LinearTrainer

    items = snapshot_items(syn, tmp / "images")
    emb = make_mean_rgb_embedder()
    trainer = DinoV3LinearTrainer(lambda: emb, weights_dir=weights_dir)
    ctx, rec = make_context(
        trainer, items, tmp / "work", {**DINO_FAST, **(params or {})}
    )
    export_spy = _ExportSpy(torch.onnx.export)
    with pytest.MonkeyPatch.context() as mp:
        if spy:
            mp.setattr(torch.onnx, "export", export_spy)
        trainer.prepare(ctx)
        trainer.train(ctx)
        metrics = trainer.evaluate(ctx)
        result = trainer.export(ctx)
    return {
        "metrics": metrics,
        "result": result,
        "rec": rec,
        "ctx": ctx,
        "spy": export_spy,
    }


def _conv_run(tmp: Path, syn, params=None, weights_dir=None, spy=False):
    import torch

    from sparrow_finetune.pipelines.conventional_classifier import (
        ConventionalClassifierTrainer,
    )

    items = snapshot_items(syn, tmp / "images")
    trainer = ConventionalClassifierTrainer(weights_dir=weights_dir)
    ctx, rec = make_context(
        trainer, items, tmp / "work", {**CONV_BASE, **(params or {})}
    )
    export_spy = _ExportSpy(torch.onnx.export)
    with pytest.MonkeyPatch.context() as mp:
        if spy:
            mp.setattr(torch.onnx, "export", export_spy)
        trainer.prepare(ctx)
        trainer.train(ctx)
        metrics = trainer.evaluate(ctx)
        result = trainer.export(ctx)
    return {
        "metrics": metrics,
        "result": result,
        "rec": rec,
        "ctx": ctx,
        "spy": export_spy,
    }


@pytest.fixture(scope="module")
def dino_dup_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("dino_dup")
    weights = tmp / "weights"
    write_stock_package(weights)
    return _dino_run(
        tmp,
        textured_items(duplicate=True),
        {"stock_model": STOCK_NAME},
        weights_dir=weights,
        spy=True,
    )


@pytest.fixture(scope="module")
def conv_dup_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("conv_dup")
    weights = tmp / "weights"
    write_stock_package(weights)
    return _conv_run(
        tmp,
        textured_items(duplicate=True),
        {"stock_model": STOCK_NAME},
        weights_dir=weights,
        spy=True,
    )


def _warnings(rec, needle: str) -> list[str]:
    return [m for s, m, _ in rec.events if s == "evaluate" and needle in m.lower()]


def _n_val(ctx) -> int:
    from sparrow_finetune.pipelines.dinov3_linear import _split_key, assign_split

    return sum(
        len(it.annotations)
        for it in ctx.items
        if assign_split(_split_key(it), seed=0, val_fraction=0.2) == "val"
    )


# --------------------------------------------------------------------------- FT-5 near-duplicates


def test_ft5_dhash_is_64_bit_and_stable():
    from sparrow_finetune.pipelines.near_dup import dhash

    rng = np.random.default_rng(0)
    im = Image.fromarray(rng.integers(0, 256, size=(24, 32, 3), dtype=np.uint8))
    h = dhash(im)
    assert isinstance(h, int) and 0 <= h < 2**64
    assert dhash(im.copy()) == h
    assert dhash(im.convert("L")) == h
    # A left-to-right brightness ramp: no pixel is brighter than its right neighbour.
    ramp = Image.fromarray(np.tile(np.arange(0, 256, 8, dtype=np.uint8), (24, 1)))
    assert dhash(ramp) == 0
    assert dhash(ramp.transpose(Image.FLIP_LEFT_RIGHT)) == 2**64 - 1


def test_ft5_count_near_duplicates_uses_hamming_distance():
    from sparrow_finetune.pipelines.near_dup import count_near_duplicates

    train = np.asarray([0, 2**64 - 1], dtype=np.uint64)
    val = np.asarray([0b1111, 0b11111, 2**63, 2**64 - 1 - 0b111], dtype=np.uint64)
    assert count_near_duplicates(val, train, 4) == 3
    assert count_near_duplicates(val, train, 0) == 0
    assert count_near_duplicates(val, train, 5) == 4
    assert count_near_duplicates(np.zeros(0, np.uint64), train, 4) == 0
    assert count_near_duplicates(val, np.zeros(0, np.uint64), 4) == 0


@pytest.mark.parametrize("pipeline", ["dinov3", "conventional"])
def test_ft5_near_dup_max_distance_param(pipeline):
    trainer = _trainer(pipeline)
    base = CONV_BASE if pipeline == "conventional" else {}
    assert trainer.validate_params(dict(base))["near_dup_max_distance"] == 4
    assert (
        trainer.validate_params({**base, "near_dup_max_distance": 32})[
            "near_dup_max_distance"
        ]
        == 32
    )
    for bad in (-1, 33, 2.5, True, "4"):
        with pytest.raises(ValueError, match="near_dup_max_distance"):
            trainer.validate_params({**base, "near_dup_max_distance": bad})


def _trainer(pipeline):
    if pipeline == "dinov3":
        from sparrow_finetune.pipelines.dinov3_linear import DinoV3LinearTrainer

        emb = make_mean_rgb_embedder()
        return DinoV3LinearTrainer(lambda: emb)
    from sparrow_finetune.pipelines.conventional_classifier import (
        ConventionalClassifierTrainer,
    )

    return ConventionalClassifierTrainer()


@pytest.mark.parametrize("run", ["dino_dup_run", "conv_dup_run"])
def test_ft5_duplicate_val_crop_is_reported_warned_and_carded(run, request):
    out = request.getfixturevalue(run)
    m, ctx = out["metrics"], out["ctx"]
    n_val = _n_val(ctx)
    assert m["near_dup_val_crops"] == 1
    assert m["near_dup_val_fraction"] == pytest.approx(1 / n_val)
    assert all(isinstance(v, (int, float)) for v in m.values())
    assert _warnings(out["rec"], "near-duplicate"), out["rec"].events
    bundle = read_bundle(out["result"].path)
    card = bundle_json(bundle, "model_card.json")
    assert card["near_duplicates"] == {
        "max_distance": 4,
        "val_crops": 1,
        "val_fraction": pytest.approx(1 / n_val),
    }
    assert card["metrics"]["near_dup_val_crops"] == 1


def test_ft5_no_duplicate_reports_zero_without_warning(tmp_path):
    out = _dino_run(tmp_path, textured_items(duplicate=False))
    assert out["metrics"]["near_dup_val_crops"] == 0
    assert out["metrics"]["near_dup_val_fraction"] == 0.0
    assert not _warnings(out["rec"], "near-duplicate")


def test_ft5_hashes_the_bbox_crop_not_the_whole_image(tmp_path):
    from sparrow_finetune.pipelines.near_dup import dhash

    out = _dino_run(
        tmp_path, textured_items(duplicate=False), {"near_dup_max_distance": 0}
    )
    hashes = np.load(tmp_path / "work" / "dhash.npy")
    assert hashes.dtype == np.uint64 and len(hashes) == len(
        np.load(tmp_path / "work" / "labels.npy")
    )
    first = out["ctx"].items[0]
    with Image.open(first.image_path) as im:
        assert int(hashes[0]) == dhash(im.convert("RGB"))
    # Solid-colour bbox crops hash to 0 whatever the background: the crop is hashed.
    syn = balanced_items(train_groups=3, val_groups=2, ungrouped_per_label=0)
    boxed = [SynItem(s.name, s.label, s.group, [6, 4, 16, 12], s.png) for s in syn]
    for i, s in enumerate(boxed):
        arr = np.asarray(Image.open(io.BytesIO(s.png)).convert("RGB")).copy()
        arr[:, :6] = np.random.default_rng(i).integers(0, 256, size=arr[:, :6].shape)
        boxed[i] = SynItem(s.name, s.label, s.group, s.bbox, _png(arr.astype(np.uint8)))
    _dino_run(tmp_path / "boxed", boxed)
    assert not np.load(tmp_path / "boxed" / "work" / "dhash.npy").any()


# --------------------------------------------------------------------------- FT-13 offline timm weights


def _save_timm_weights(weights_dir: Path, name: str, fmt: str) -> dict:
    import timm
    import torch

    torch.manual_seed(1234)
    sd = timm.create_model(name, pretrained=False).state_dict()
    d = weights_dir / "timm" / name
    d.mkdir(parents=True, exist_ok=True)
    if fmt == "safetensors":
        from safetensors.torch import save_file

        save_file(
            {k: v.contiguous() for k, v in sd.items()}, str(d / "model.safetensors")
        )
    else:
        torch.save(sd, d / "pytorch_model.bin")
    return sd


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*_a, **_k):
        raise OSError("network access is disabled in this test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


@pytest.mark.parametrize("fmt", ["safetensors", "bin"])
def test_ft13_local_timm_weights_load_without_network(tmp_path, no_network, fmt):
    import torch

    from sparrow_finetune.pipelines.conventional_classifier import (
        ConventionalClassifierTrainer,
    )

    sd = _save_timm_weights(tmp_path / "w", "resnet18", fmt)
    trainer = ConventionalClassifierTrainer(weights_dir=tmp_path / "w")
    params = trainer.validate_params({**CONV_BASE, "pretrained": True})
    model = trainer._create(params, 3, pretrained=True)
    assert torch.equal(model.state_dict()["conv1.weight"], sd["conv1.weight"])
    assert model.get_classifier().out_features == 3


def test_ft13_offline_pretrained_run_uses_local_weights(tmp_path, no_network):
    sd = _save_timm_weights(tmp_path / "w", "resnet18", "safetensors")
    out = _conv_run(
        tmp_path,
        balanced_items(),
        {"pretrained": True, "epochs": 1, "warmup_epochs": 1},
        weights_dir=tmp_path / "w",
    )
    assert out["metrics"]["val_crops"] > 0
    assert sd  # weights were present; the run made no network call


def test_ft13_offline_without_local_weights_fails_prepare_naming_paths(
    tmp_path, monkeypatch
):
    from huggingface_hub import constants

    from sparrow_finetune.pipelines.conventional_classifier import (
        ConventionalClassifierTrainer,
    )

    # Empty, test-owned Hugging Face cache: weights in the host cache must not change the result.
    hub = tmp_path / "hf" / "hub"
    hub.mkdir(parents=True)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(hub))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    items = snapshot_items(balanced_items(), tmp_path / "images")
    (tmp_path / "w").mkdir()
    trainer = ConventionalClassifierTrainer(weights_dir=tmp_path / "w")
    ctx, _ = make_context(
        trainer, items, tmp_path / "work", {**CONV_BASE, "pretrained": True}
    )
    with pytest.raises(ValueError) as err:
        trainer.prepare(ctx)
    msg = str(err.value)
    assert str(tmp_path / "w" / "timm" / "resnet18" / "model.safetensors") in msg
    assert "pytorch_model.bin" in msg and "HF_HUB_OFFLINE" in msg

    no_dir = ConventionalClassifierTrainer()
    ctx, _ = make_context(
        no_dir, items, tmp_path / "work2", {**CONV_BASE, "pretrained": True}
    )
    with pytest.raises(ValueError, match="SPARROW_FT_WEIGHTS_DIR"):
        no_dir.prepare(ctx)


def test_ft13_offline_without_pretrained_needs_no_weights(tmp_path, monkeypatch):
    from sparrow_finetune.pipelines.conventional_classifier import (
        ConventionalClassifierTrainer,
    )

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    items = snapshot_items(balanced_items(), tmp_path / "images")
    trainer = ConventionalClassifierTrainer()
    ctx, _ = make_context(
        trainer, items, tmp_path / "work", {**CONV_BASE, "pretrained": False}
    )
    trainer.prepare(ctx)


def test_ft13_register_builtin_pipelines_wires_weights_dir(tmp_path):
    from sparrow_finetune.config import Settings
    from sparrow_finetune.pipelines import Registry, register_builtin_pipelines

    base = dict(
        database_url="postgresql://u@h/db", data_dir=tmp_path, embedder="onnx:x"
    )
    reg = Registry()
    register_builtin_pipelines(reg, Settings(**base))
    conv, dino = reg.get("conventional_classifier"), reg.get("dinov3_linear")
    assert conv._weights_dir is None and dino._weights_dir is None
    register_builtin_pipelines(reg, Settings(**base))
    assert (
        reg.get("conventional_classifier") is conv and reg.get("dinov3_linear") is dino
    )
    register_builtin_pipelines(reg, Settings(**base, weights_dir=tmp_path))
    assert reg.get("conventional_classifier") is not conv
    assert reg.get("conventional_classifier")._weights_dir == tmp_path
    assert reg.get("dinov3_linear")._weights_dir == tmp_path


# --------------------------------------------------------------------------- FT-15 dynamo export


def _run_onnx(data: bytes, inputs: np.ndarray) -> tuple[list, list, np.ndarray]:
    import onnxruntime as ort

    sess = ort.InferenceSession(data, providers=["CPUExecutionProvider"])
    ins = [i.name for i in sess.get_inputs()]
    outs = [o.name for o in sess.get_outputs()]
    return ins, outs, np.asarray(sess.run(None, {ins[0]: inputs.astype(np.float32)})[0])


def test_ft15_conventional_exports_with_dynamo_and_dynamic_batch(conv_dup_run):
    spy = conv_dup_run["spy"]
    assert spy.calls and all(c.get("dynamo") is True for c in spy.calls), spy.calls
    bundle = read_bundle(conv_dup_run["result"].path)
    manifest = bundle_json(bundle, "manifest.json")
    assert manifest["parity"]["top1_agreement"] == 1.0
    for n in (1, 3):
        x = np.random.default_rng(n).random((n, 3, 32, 32))
        ins, outs, probs = _run_onnx(bundle["model.onnx"], x)
        assert ins == [manifest["input"]["name"]] and manifest["output"] in outs
        assert probs.shape == (n, 3) and np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_ft15_dinov3_head_exports_with_dynamo_and_matches_sklearn(dino_dup_run):
    import pickle

    spy = dino_dup_run["spy"]
    assert spy.calls and all(c.get("dynamo") is True for c in spy.calls), spy.calls
    bundle = read_bundle(dino_dup_run["result"].path)
    manifest = bundle_json(bundle, "manifest.json")
    workdir = Path(dino_dup_run["ctx"].workdir)
    with open(workdir / "model.pkl", "rb") as fh:
        model = pickle.load(fh)
    for n in (1, 3):
        x = np.random.default_rng(n).random((n, 3)).astype(np.float32)
        ins, outs, probs = _run_onnx(bundle["head.onnx"], x)
        assert (
            ins == [manifest["head"]["input"]]
            and manifest["head"]["probabilities"] in outs
        )
        assert np.max(np.abs(probs - model.predict_proba(x))) <= 1e-4


def test_ft15_dinov3_binary_head_matches_sklearn(tmp_path):
    import pickle

    syn = balanced_items(labels=("cat", "dog"))
    out = _dino_run(tmp_path, syn, spy=True)
    bundle = read_bundle(out["result"].path)
    with open(tmp_path / "work" / "model.pkl", "rb") as fh:
        model = pickle.load(fh)
    x = np.random.default_rng(0).random((4, 3)).astype(np.float32)
    _ins, _outs, probs = _run_onnx(bundle["head.onnx"], x)
    assert probs.shape == (4, 2)
    assert np.max(np.abs(probs - model.predict_proba(x))) <= 1e-4


@pytest.mark.parametrize("n_classes", [2, 3])
def test_ft15_dinov3_head_matches_sklearn_for_strongly_negative_logits(n_classes):
    # onnxruntime's Sigmoid returns 0 for logits <= -18, which broke the
    # multiclass ratio p_i / sum(p) (review I1, 2026-10-16).
    import onnxruntime as ort
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from sparrow_finetune.pipelines.dinov3_linear import _export_head_onnx

    rng = np.random.default_rng(0)
    x = rng.normal(size=(90, 16)).astype(np.float32)
    y = np.arange(90) % n_classes
    scaler = StandardScaler().fit(x)
    clf = SGDClassifier(loss="log_loss", average=True, random_state=0).fit(
        scaler.transform(x), y
    )
    clf.intercept_ -= 30
    model = Pipeline([("scaler", scaler), ("clf", clf)])
    session = ort.InferenceSession(_export_head_onnx(model, 16))
    probs = session.run(None, {"features": x})[0]
    expected = model.predict_proba(x)
    assert np.isfinite(probs).all()
    assert np.max(np.abs(probs - expected)) <= 1e-4
    assert (probs.argmax(1) == expected.argmax(1)).all()


def test_ft15_onnxscript_is_a_locked_dependency():
    root = Path(__file__).resolve().parents[2]
    assert "onnxscript" in (root / "pyproject.toml").read_text()
    assert 'name = "onnxscript"' in (root / "uv.lock").read_text()


# --------------------------------------------------------------------------- FT-16 stock model


def test_ft16_stock_model_loads_and_scores(tmp_path):
    from sparrow_finetune.pipelines.stock_model import StockModel, resolve_stock_package

    write_stock_package(tmp_path)
    stock = StockModel.load(resolve_stock_package(STOCK_NAME, tmp_path))
    assert stock.id == STOCK_ID and stock.labels == ["Cat", "Dog", "Bird"]
    crops = [
        Image.new("RGB", (40, 30), c)
        for c in ((210, 50, 50), (50, 210, 50), (50, 50, 210))
    ]
    probs = stock.predict_proba(crops)
    assert probs.shape == (3, 3)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert probs.argmax(axis=1).tolist() == [0, 1, 2]
    # imagenet normalisation then softmax: check one value exactly.
    rgb = np.asarray([210, 50, 50], np.float32) / 255
    feat = (rgb - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    logits = 5.0 * feat
    want = np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()
    assert np.allclose(probs[0], want, atol=1e-4)


@pytest.mark.parametrize(
    "old,new,needle",
    [
        ('method = "resize"', 'method = "letterbox"', "letterbox"),
        ('format = "onnx"', 'format = "tflite"', "tflite"),
        ('layout = "nchw"', 'layout = "nhwc"', "nhwc"),
        ('normalization = "imagenet"', 'normalization = "zscore"', "zscore"),
        (
            '[postprocessing]\nmethod = "softmax"',
            '[postprocessing]\nmethod = "yolo"',
            "yolo",
        ),
        ('format = "name_index_csv"', 'format = "json"', "json"),
        ('file = "1/model.onnx"', 'file = "../escape.onnx"', "escape"),
    ],
)
def test_ft16_unsupported_manifest_values_are_rejected(tmp_path, old, new, needle):
    from sparrow_finetune.pipelines.stock_model import StockModel

    pkg = write_stock_package(tmp_path, replace={old: new})
    with pytest.raises(ValueError, match=needle):
        StockModel.load(pkg)


def test_ft16_resolve_stock_package_rejects_escapes_and_missing(tmp_path):
    from sparrow_finetune.pipelines.stock_model import resolve_stock_package

    write_stock_package(tmp_path)
    assert (
        resolve_stock_package(STOCK_NAME, tmp_path) == (tmp_path / STOCK_NAME).resolve()
    )
    for bad in ("../x", "/abs", "a/b", "..", ""):
        with pytest.raises(ValueError):
            resolve_stock_package(bad, tmp_path)
    with pytest.raises(ValueError, match="SPARROW_FT_WEIGHTS_DIR"):
        resolve_stock_package(STOCK_NAME, None)
    with pytest.raises(ValueError, match="not found"):
        resolve_stock_package("missing-model", tmp_path)
    (tmp_path / "nomanifest").mkdir()
    with pytest.raises(ValueError, match="manifest.toml"):
        resolve_stock_package("nomanifest", tmp_path)


def test_ft16_stock_models_endpoint_lists_only_classifier_packages(client, tmp_path):
    import dataclasses

    assert client.get("/v1/stock-models").json() == {"models": []}
    write_stock_package(tmp_path)
    (tmp_path / "MDV6-yolov10-c" / "1").mkdir(parents=True)  # assist package: no manifest
    bad = write_stock_package(tmp_path, name="Detector")
    text = (bad / "manifest.toml").read_text().replace('method = "softmax"', 'method = "yolo"')
    (bad / "manifest.toml").write_text(text)
    app = client.app
    app.state.settings = dataclasses.replace(app.state.settings, weights_dir=tmp_path)
    r = client.get("/v1/stock-models")
    assert r.status_code == 200
    assert r.json() == {"models": [{"id": STOCK_NAME, "name": STOCK_ID, "classes": 3}]}


def test_ft16_comparison_metrics_score_stock_over_full_label_space():
    from sparrow_finetune.pipelines.stock_model import (
        comparison_metrics,
        matched_label_map,
    )

    mapping = matched_label_map(["cat", "dog", "fox"], ["Bird", "CAT", "Dog"])
    assert mapping == {0: 1, 1: 2}
    y = np.asarray([0, 0, 1, 1])
    # FT-18: Bird (unmatched) scores highest everywhere. The stock argmax covers every stock
    # class, so Bird wins and every stock prediction is wrong; restricting the argmax to
    # Cat/Dog would wrongly score the stock model as perfect.
    bird_wins = np.asarray(
        [[0.9, 0.08, 0.02], [0.9, 0.06, 0.04], [0.9, 0.01, 0.09], [0.9, 0.03, 0.07]]
    )
    out = comparison_metrics(y, np.asarray([0, 2, 1, 1]), bird_wins, mapping)
    assert out["stock_val_macro_f1"] == 0.0
    assert out["stock_finetuned_val_macro_f1"] == pytest.approx(5 / 6)
    assert out["stock_compared_crops"] == 4.0 and out["stock_worse"] == 0.0
    # Matched classes on top: the stock model is perfect and beats a worse fine-tune.
    matched_wins = np.asarray(
        [[0.02, 0.9, 0.08], [0.04, 0.9, 0.06], [0.09, 0.01, 0.9], [0.07, 0.03, 0.9]]
    )
    worse = comparison_metrics(y, np.asarray([0, 2, 1, 1]), matched_wins, mapping)
    assert worse["stock_val_macro_f1"] == 1.0
    assert worse["stock_finetuned_val_macro_f1"] < 1.0
    assert worse["stock_compared_crops"] == 4.0 and worse["stock_worse"] == 1.0
    equal = comparison_metrics(y, y.copy(), matched_wins, mapping)
    assert equal["stock_finetuned_val_macro_f1"] == 1.0 and equal["stock_worse"] == 0.0


@pytest.mark.parametrize("pipeline", ["dinov3", "conventional"])
def test_ft16_stock_model_param_validation(pipeline):
    trainer = _trainer(pipeline)
    base = CONV_BASE if pipeline == "conventional" else {}
    assert trainer.validate_params(dict(base))["stock_model"] is None
    assert (
        trainer.validate_params({**base, "stock_model": "AI4G-Amazon-V2"})[
            "stock_model"
        ]
        == "AI4G-Amazon-V2"
    )
    for bad in ("../x", "a/b", "", 3):
        with pytest.raises(ValueError, match="stock_model"):
            trainer.validate_params({**base, "stock_model": bad})


@pytest.mark.parametrize("run", ["dino_dup_run", "conv_dup_run"])
def test_ft16_stock_comparison_metrics_and_card(run, request):
    from sparrow_finetune.pipelines.dinov3_linear import _split_key, assign_split

    out = request.getfixturevalue(run)
    m, ctx = out["metrics"], out["ctx"]
    compared = sum(
        1
        for it in ctx.items
        for a in it.annotations
        if a["label"] in ("cat", "dog")
        and assign_split(_split_key(it), seed=0, val_fraction=0.2) == "val"
    )
    assert m["stock_compared_crops"] == compared > 0
    assert m["stock_val_macro_f1"] == 1.0
    assert 0.0 <= m["stock_finetuned_val_macro_f1"] <= 1.0
    assert m["stock_worse"] == float(
        m["stock_finetuned_val_macro_f1"] < m["stock_val_macro_f1"]
    )
    card = bundle_json(read_bundle(out["result"].path), "model_card.json")
    assert (
        card["stock_model"]["id"] == STOCK_ID
        and card["stock_model"]["name"] == STOCK_NAME
    )
    assert out["result"].metadata["stock_model"] == STOCK_ID


def test_ft16_no_matching_label_reports_zero_and_warns(tmp_path):
    weights = tmp_path / "weights"
    write_stock_package(weights, labels=("Zebra", "Lion", "Okapi"))
    out = _dino_run(
        tmp_path, balanced_items(), {"stock_model": STOCK_NAME}, weights_dir=weights
    )
    m = out["metrics"]
    assert m["stock_compared_crops"] == 0
    for key in ("stock_val_macro_f1", "stock_finetuned_val_macro_f1", "stock_worse"):
        assert key not in m
    assert _warnings(out["rec"], "stock")


@pytest.mark.parametrize("pipeline", ["dinov3", "conventional"])
def test_ft16_prepare_fails_without_weights_dir_or_package(tmp_path, pipeline):
    from sparrow_finetune.pipelines.conventional_classifier import (
        ConventionalClassifierTrainer,
    )
    from sparrow_finetune.pipelines.dinov3_linear import DinoV3LinearTrainer

    items = snapshot_items(balanced_items(), tmp_path / "images")
    base = CONV_BASE if pipeline == "conventional" else DINO_FAST

    def build(wd):
        if pipeline == "conventional":
            return ConventionalClassifierTrainer(weights_dir=wd)
        emb = make_mean_rgb_embedder()
        return DinoV3LinearTrainer(lambda: emb, weights_dir=wd)

    t = build(None)
    ctx, _ = make_context(t, items, tmp_path / "a", {**base, "stock_model": STOCK_NAME})
    with pytest.raises(ValueError, match="SPARROW_FT_WEIGHTS_DIR"):
        t.prepare(ctx)
    (tmp_path / "w").mkdir()
    t = build(tmp_path / "w")
    ctx, _ = make_context(t, items, tmp_path / "b", {**base, "stock_model": STOCK_NAME})
    with pytest.raises(ValueError, match="not found"):
        t.prepare(ctx)


# --------------------------------------------------------------------------- FT-17 stale comment


def test_ft17_malformed_embedder_spec_fails_before_any_crop_is_read(
    tmp_path, monkeypatch
):
    from sparrow_finetune.embedders import make_embedder
    from sparrow_finetune.pipelines.dinov3_linear import DinoV3LinearTrainer

    items = snapshot_items(balanced_items(), tmp_path / "img")
    trainer = DinoV3LinearTrainer(lambda: make_embedder("bogus"))
    ctx, _ = make_context(
        trainer, items, tmp_path / "work", {**DINO_FAST, "class_floor": 1000}
    )
    opened = []
    monkeypatch.setattr(
        Image, "open", lambda *a, **k: opened.append(a) or pytest.fail("crop read")
    )
    with pytest.raises(ValueError, match="invalid SPARROW_FT_EMBEDDER"):
        trainer.prepare(ctx)
    assert not opened


def test_ft17_dinov3_prepare_comment_is_accurate():
    src = (
        Path(__file__).resolve().parents[2]
        / "src/sparrow_finetune/pipelines/dinov3_linear.py"
    ).read_text()
    assert "fails the job at once" not in src
    assert "unavailable_reason" in src


def test_card_values_are_json_numbers(dino_dup_run):
    report = json.loads(
        (Path(dino_dup_run["ctx"].workdir) / "evaluate.json").read_text()
    )
    assert all(isinstance(v, (int, float)) for v in report["metrics"].values())
