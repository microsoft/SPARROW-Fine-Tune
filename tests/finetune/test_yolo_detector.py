"""Phase 1C unit/trainer-level specification tests for the ``yolo_detector`` pipeline.

Spec: docs/review/implement-yolo-detector/spec.md. Only the public entry points are used:
``to_yolo_label``, ``YoloDetectorTrainer`` (validate_params / prepare / train / evaluate / export on a
directly built ``JobContext``) and ``register_builtin_pipelines``.

Strategy:
- C1/C2 are pure functions: checked against an independent reference formula and an exhaustive list of
  rejection classes (each must raise ValueError naming the param, never OverflowError).
- C3/C4/C5/C10 run ``prepare`` on synthetic PNGs whose groups are chosen through ``assign_split``, so the
  expected split of every image is known without the code under test. Written images are matched back to
  items by sha256, which holds for copies and links alike.
- C6-C9 run one real 2-epoch CPU training of ``yolo11n.yaml`` (random init, no download) shared by a
  module fixture; ``ultralytics.engine.model.Model.train`` is wrapped by a pass-through spy to check the
  arguments C6 prescribes. ``YOLO_OFFLINE=1`` keeps ultralytics off the network.
- Cancellation: ``check_cancelled`` raises after the first epoch event; ``train`` must raise JobCancelled.
"""

from __future__ import annotations

import hashlib
import io
import math
import zipfile

import numpy as np
import pytest
import yaml

from .dinov3_helpers import bundle_json, make_context, read_bundle
from .yolo_helpers import (
    CPU_PARAMS,
    DET_LABELS,
    DET_SIZE,
    PARAM_DEFAULTS,
    DetItem,
    det_items,
    det_png,
    det_snapshot_items,
    expected_counts,
    expected_side,
    expected_yolo,
    read_dataset,
)


@pytest.fixture(scope="module", autouse=True)
def yolo_offline():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("YOLO_OFFLINE", "1")
        yield


def new_trainer(resolver=None):
    from sparrow_finetune.pipelines.yolo_detector import YoloDetectorTrainer

    if resolver is None:
        return YoloDetectorTrainer()
    return YoloDetectorTrainer(base_model_resolver=resolver)


def unit_items(prefix: str = "u-") -> list[DetItem]:
    """Two labels, groups on both sides, one image without annotations and one with a bbox-less annotation."""
    syn = det_items(prefix=prefix)
    syn.append(DetItem(f"{prefix}bboxless.png", f"{prefix}bboxless-group", [], det_png([], 950), ["cat"]))
    return syn


def prepared(tmp_path, syn=None, params=None, workdir_name="work"):
    trainer = new_trainer()
    items = det_snapshot_items(syn if syn is not None else unit_items(), tmp_path / "images")
    ctx, rec = make_context(trainer, items, tmp_path / workdir_name, {**CPU_PARAMS, **(params or {})})
    trainer.prepare(ctx)
    return trainer, ctx, rec, items


def train_epoch_events(rec) -> list[dict]:
    return [m for m in rec.stage_events("train") if "epoch" in m]


# --------------------------------------------------------------------------- entry points


def test_trainer_identity_attributes():
    trainer = new_trainer()
    assert trainer.id == "yolo_detector"
    assert trainer.task == "detection"
    assert trainer.requires == frozenset()
    assert isinstance(trainer.name, str) and trainer.name


def test_register_builtin_pipelines_registers_yolo_detector_idempotently(tmp_path):
    from sparrow_finetune.config import Settings
    from sparrow_finetune.pipelines import Registry, register_builtin_pipelines

    settings = Settings(database_url="postgresql://unused@localhost/unused", data_dir=tmp_path, embedder="")
    reg = Registry()
    register_builtin_pipelines(reg, settings)
    first = reg.get("yolo_detector")
    register_builtin_pipelines(reg, settings)
    assert first is not None
    assert reg.get("yolo_detector") is first
    assert reg.ids().count("yolo_detector") == 1
    status = {p["id"]: p["status"] for p in reg.catalogue()}
    assert status["yolo_detector"] == "available"


# --------------------------------------------------------------------------- C1 to_yolo_label


def test_c1_spec_example():
    from sparrow_finetune.pipelines.yolo_detector import to_yolo_label

    assert to_yolo_label([10, 20, 30, 40], 100, 200) == pytest.approx((0.25, 0.2, 0.3, 0.2))


@pytest.mark.parametrize(
    "bbox, size, expected",
    [
        ([-10, -10, 30, 30], (100, 100), (0.1, 0.1, 0.2, 0.2)),  # clamped at the top-left
        ([90, 90, 20, 20], (100, 100), (0.95, 0.95, 0.1, 0.1)),  # clamped at the bottom-right
        ([-5, 10, 200, 20], (100, 50), (0.5, 0.4, 1.0, 0.4)),  # wider than the image
        ([0, 0, 64, 64], (64, 64), (0.5, 0.5, 1.0, 1.0)),
    ],
)
def test_c1_clamps_box_to_image_first(bbox, size, expected):
    from sparrow_finetune.pipelines.yolo_detector import to_yolo_label

    assert to_yolo_label(bbox, *size) == pytest.approx(expected)


@pytest.mark.parametrize(
    "bbox",
    [
        [100, 10, 5, 5],  # starts at the right edge -> zero width after clamping
        [10, 100, 5, 5],  # starts at the bottom edge
        [-20, 0, 10, 10],  # entirely left of the image
        [10, 10, 0, 5],  # zero width
        [10, 10, 5, 0],  # zero height
    ],
)
def test_c1_zero_area_after_clamping_raises_value_error(bbox):
    from sparrow_finetune.pipelines.yolo_detector import to_yolo_label

    with pytest.raises(ValueError):
        to_yolo_label(bbox, 100, 100)


def test_c1_outputs_lie_in_unit_interval_and_match_reference():
    from sparrow_finetune.pipelines.yolo_detector import to_yolo_label

    rng = np.random.default_rng(7)
    checked = 0
    for _ in range(300):
        width, height = int(rng.integers(8, 400)), int(rng.integers(8, 400))
        x, y = rng.uniform(-0.3, 1.1) * width, rng.uniform(-0.3, 1.1) * height
        w, h = rng.uniform(0.01, 1.2) * width, rng.uniform(0.01, 1.2) * height
        ref = expected_yolo([x, y, w, h], width, height)
        if ref[2] <= 0 or ref[3] <= 0:
            continue
        out = to_yolo_label([x, y, w, h], width, height)
        assert len(out) == 4
        assert all(0.0 <= v <= 1.0 for v in out), out
        assert out == pytest.approx(ref, abs=1e-9)
        checked += 1
    assert checked > 100


# --------------------------------------------------------------------------- C2 validate_params


def test_c2_validate_params_fills_defaults():
    assert new_trainer().validate_params({}) == PARAM_DEFAULTS


def test_c2_validate_params_keeps_valid_overrides():
    params = {
        "model": "yolo11s.pt",
        "epochs": 1000,
        "imgsz": 32,
        "batch": 256,
        "seed": 123,
        "val_fraction": 0.5,
        "device": "cuda:1",
        "patience": 0,
        "lr0": 0.5,
        "workers": 32,
        "near_dup_max_distance": 0,
    }
    assert new_trainer().validate_params(dict(params)) == params


@pytest.mark.parametrize("device", ["auto", "cpu", "cuda", "cuda:0", "cuda:3", "0", "1"])
def test_c2_accepts_every_device_form(device):
    assert new_trainer().validate_params({"device": device})["device"] == device


@pytest.mark.parametrize("imgsz", [32, 64, 640, 2048])
def test_c2_accepts_imgsz_multiples_of_32_in_range(imgsz):
    assert new_trainer().validate_params({"imgsz": imgsz})["imgsz"] == imgsz


def test_c2_rejects_unknown_key_naming_it():
    with pytest.raises(ValueError, match="bogus_key"):
        new_trainer().validate_params({"bogus_key": 1})


HUGE = 10**400


@pytest.mark.parametrize(
    "param, value",
    [
        ("model", ""),
        ("model", 3),
        ("model", None),
        ("epochs", 0),
        ("epochs", 1001),
        ("epochs", 2.5),
        ("epochs", "2"),
        ("epochs", True),
        ("imgsz", 0),
        ("imgsz", 31),
        ("imgsz", 100),  # not a multiple of 32
        ("imgsz", 2080),
        ("imgsz", 64.0),
        ("imgsz", True),
        ("batch", 0),
        ("batch", 257),
        ("batch", True),
        ("seed", -1),
        ("seed", 1.5),
        ("seed", False),
        ("val_fraction", 0),
        ("val_fraction", 0.0),
        ("val_fraction", -0.1),
        ("val_fraction", 0.51),
        ("val_fraction", "0.2"),
        ("val_fraction", True),
        ("val_fraction", float("nan")),
        ("val_fraction", HUGE),
        ("device", "gpu"),
        ("device", "cuda:x"),
        ("device", "cuda:"),
        ("device", "cuda:-1"),
        ("device", "cpu0"),
        ("device", ""),
        ("device", "-1"),
        ("device", 0),
        ("device", None),
        ("patience", -1),
        ("patience", 1001),
        ("patience", True),
        ("lr0", 0),
        ("lr0", 0.0),
        ("lr0", -0.01),
        ("lr0", float("inf")),
        ("lr0", float("nan")),
        ("lr0", HUGE),
        ("lr0", "0.01"),
        ("lr0", True),
        ("workers", -1),
        ("workers", 33),
        ("workers", True),
    ],
)
def test_c2_rejects_bad_type_or_range_with_value_error_naming_param(param, value):
    try:
        new_trainer().validate_params({param: value})
    except OverflowError as exc:  # pragma: no cover - the failure branch
        pytest.fail(f"{param}={value!r} raised OverflowError, expected ValueError: {exc}")
    except ValueError as exc:
        assert param in str(exc), f"message does not name {param!r}: {exc}"
    else:
        pytest.fail(f"{param}={value!r} was accepted")


# --------------------------------------------------------------------------- C3 prepare


def test_c3_prepare_writes_yolo_dataset_layout_and_data_yaml(tmp_path):
    _, ctx, _, _ = prepared(tmp_path)
    dataset = ctx.workdir / "dataset"
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        assert (dataset / sub).is_dir(), sub

    data = yaml.safe_load((dataset / "data.yaml").read_text())
    from pathlib import Path

    assert Path(data["path"]).is_absolute()
    assert Path(data["path"]).resolve() == dataset.resolve()
    assert data["train"] == "images/train"
    assert data["val"] == "images/val"
    assert data["names"] == dict(enumerate(ctx.labels))


def test_c3_every_item_written_once_with_matching_label_lines_in_ctx_label_order(tmp_path):
    _, ctx, _, items = prepared(tmp_path)
    by_sha = {it.sha256: it for it in items}
    assert len(by_sha) == len(items)
    written = read_dataset(ctx.workdir / "dataset")

    seen = {sha for split in written.values() for sha in split}
    assert seen == set(by_sha), "images missing from or foreign to the dataset"
    assert sum(len(split) for split in written.values()) == len(items)

    stems = [p.stem for s in ("train", "val") for p in (ctx.workdir / "dataset" / "images" / s).iterdir()]
    assert len(stems) == len(set(stems)), "image file names are not unique"

    for split, files in written.items():
        for sha, lines in files.items():
            item = by_sha[sha]
            boxed = [a for a in item.annotations if a.get("bbox") is not None]
            assert len(lines) == len(boxed), item.name
            got = sorted((int(ln[0]), *map(float, ln[1:])) for ln in lines)
            want = sorted(
                (ctx.labels.index(a["label"]), *expected_yolo(a["bbox"], item.width, item.height)) for a in boxed
            )
            for g, w in zip(got, want):
                assert len(g) == 5
                assert g[0] == w[0], item.name
                assert g[1:] == pytest.approx(w[1:], abs=1e-4), item.name
                assert all(0.0 <= v <= 1.0 for v in g[1:])


def test_c3_items_without_boxes_become_background_images_with_empty_label_files(tmp_path):
    _, ctx, _, items = prepared(tmp_path)
    written = read_dataset(ctx.workdir / "dataset")
    backgrounds = [it for it in items if not any(a.get("bbox") is not None for a in it.annotations)]
    assert any(not it.annotations for it in backgrounds) and any(it.annotations for it in backgrounds)
    for it in backgrounds:
        side = expected_side(it)
        assert it.sha256 in written[side], it.name
        assert written[side][it.sha256] == [], it.name


def test_c3_prepare_emits_event_with_image_and_box_counts(tmp_path):
    _, _, rec, items = prepared(tmp_path)
    expected = expected_counts(items)
    events = [m for m in rec.stage_events("prepare") if "train_images" in m]
    assert len(events) == 1, rec.events
    ev = events[0]
    assert ev["train_images"] == expected["train"]["items"]
    assert ev["val_images"] == expected["val"]["items"]
    assert ev["train_boxes"] == expected["train"]["boxes"]
    assert ev["val_boxes"] == expected["val"]["boxes"]


# --------------------------------------------------------------------------- C4 split integrity


@pytest.mark.parametrize("seed", [0, 7])
def test_c4_split_follows_assign_split_and_keeps_groups_whole(tmp_path, seed):
    from sparrow_finetune.pipelines.dinov3_linear import assign_split

    syn = unit_items()
    if seed:  # extra ungrouped items exercise the item-name key
        syn += [DetItem(f"u-loose-{i}.png", None, [("cat", [4, 4, 16, 16])], det_png([("cat", [4, 4, 16, 16])], 700 + i))
                for i in range(4)]
    _, ctx, _, items = prepared(tmp_path, syn, {"seed": seed})
    written = read_dataset(ctx.workdir / "dataset")
    side_of = {sha: s for s, files in written.items() for sha in files}
    groups = {"train": set(), "val": set()}
    for it in items:
        side = side_of[it.sha256]
        assert side == assign_split(it.group or it.name, seed=seed, val_fraction=0.2), it.name
        if it.group:
            groups[side].add(it.group)
    assert not groups["train"] & groups["val"]


# --------------------------------------------------------------------------- C5 empty split


def test_c5_prepare_rejects_snapshot_whose_boxes_all_land_in_train(tmp_path):
    syn = det_items(train_groups=3, val_groups=0, backgrounds=0, prefix="c5t-")
    with pytest.raises(ValueError) as exc:
        prepared(tmp_path, syn)
    assert "val" in str(exc.value).lower(), exc.value


def test_c5_prepare_rejects_snapshot_whose_boxes_all_land_in_val(tmp_path):
    syn = det_items(train_groups=0, val_groups=2, backgrounds=0, prefix="c5v-")
    with pytest.raises(ValueError) as exc:
        prepared(tmp_path, syn)
    assert "train" in str(exc.value).lower(), exc.value


def test_c5_prepare_rejects_snapshot_without_any_box(tmp_path):
    syn = det_items(train_groups=0, val_groups=0, backgrounds=8, prefix="c5n-")
    syn.append(DetItem("c5n-bboxless.png", "c5n-x", [], det_png([], 960), ["cat"]))
    assert {expected_side(it) for it in syn} == {"train", "val"}  # both splits hold images, neither holds a box
    with pytest.raises(ValueError):
        prepared(tmp_path, syn)


# --------------------------------------------------------------------------- C10 reproducibility


def test_c10_same_snapshot_and_params_give_same_split_and_data_yaml(tmp_path):
    syn = unit_items()
    _, ctx_a, _, _ = prepared(tmp_path / "a", syn, workdir_name="work-a")
    _, ctx_b, _, _ = prepared(tmp_path / "b", syn, workdir_name="work-b")
    assert read_dataset(ctx_a.workdir / "dataset") == read_dataset(ctx_b.workdir / "dataset")
    yaml_a = yaml.safe_load((ctx_a.workdir / "dataset" / "data.yaml").read_text())
    yaml_b = yaml.safe_load((ctx_b.workdir / "dataset" / "data.yaml").read_text())
    yaml_a.pop("path")
    yaml_b.pop("path")
    assert yaml_a == yaml_b


# --------------------------------------------------------------------------- C6-C9 one real CPU training


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    from ultralytics.engine.model import Model

    base = tmp_path_factory.mktemp("yolo-trained")
    resolved: list[str] = []

    def resolver(name: str) -> str:
        resolved.append(name)
        return name

    trainer = new_trainer(resolver)
    items = det_snapshot_items(unit_items(prefix="t-"), base / "images")
    train_calls: list[dict] = []
    original = Model.train

    def spy(self, *args, **kwargs):
        train_calls.append(dict(kwargs))
        return original(self, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Model, "train", spy)
        ctx, rec = make_context(trainer, items, base / "work", CPU_PARAMS)
        trainer.prepare(ctx)
        trainer.train(ctx)
        metrics = trainer.evaluate(ctx)
        result = trainer.export(ctx)
    return {
        "ctx": ctx,
        "rec": rec,
        "items": items,
        "metrics": metrics,
        "result": result,
        "bundle": read_bundle(result.path),
        "resolved": resolved,
        "train_calls": train_calls,
    }


def test_c6_train_passes_prescribed_arguments_to_ultralytics(trained):
    from pathlib import Path

    ctx = trained["ctx"]
    assert trained["resolved"] and set(trained["resolved"]) == {"yolo11n.yaml"}
    assert len(trained["train_calls"]) == 1
    kw = trained["train_calls"][0]
    assert Path(kw["data"]).resolve() == (ctx.workdir / "dataset" / "data.yaml").resolve()
    expected = {
        "epochs": 2,
        "imgsz": 64,
        "batch": 4,
        "seed": 0,
        "device": "cpu",
        "patience": 10,
        "lr0": pytest.approx(0.01),
        "workers": 0,
        "name": "train",
        "exist_ok": True,
        "plots": False,
        "deterministic": True,
        "verbose": False,
        "amp": False,
    }
    for key, value in expected.items():
        assert key in kw, key
        assert kw[key] == value, (key, kw[key])
    assert Path(kw["project"]).resolve() == (ctx.workdir / "runs").resolve()


def test_c6_train_emits_one_event_per_epoch_with_float_metrics(trained):
    events = train_epoch_events(trained["rec"])
    assert [e["epoch"] for e in events] == [1, 2]
    for e in events:
        assert isinstance(e["epoch"], int) and not isinstance(e["epoch"], bool)
        for key in ("map50", "map50_95", "precision", "recall"):
            assert isinstance(e[key], float), (key, e[key])
            assert math.isfinite(e[key]) and 0.0 <= e[key] <= 1.0, (key, e[key])
    assert trained["rec"].checks >= 2


@pytest.mark.parametrize("cuda, expected_device", [(False, "cpu"), (True, "0")])
def test_c6_auto_device_resolves_from_cuda_availability(tmp_path, monkeypatch, cuda, expected_device):
    import torch
    from ultralytics.engine.model import Model

    class Captured(Exception):
        pass

    calls: list[dict] = []

    def stub(self, *args, **kwargs):
        calls.append(dict(kwargs))
        raise Captured("stop before training")

    trainer, ctx, _, _ = prepared(tmp_path, params={"device": "auto"})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(Model, "train", stub)
    with pytest.raises(Exception):
        trainer.train(ctx)
    assert calls, "YOLO.train was never called"
    assert calls[0]["device"] == expected_device
    if not cuda:
        assert calls[0].get("amp") is False


def test_c6_cancellation_after_first_epoch_raises_job_cancelled(tmp_path):
    from sparrow_finetune.pipelines import JobCancelled

    trainer, ctx, rec, _ = prepared(tmp_path, params={"epochs": 3})
    rec.cancel_on_first_train_event = True
    with pytest.raises(JobCancelled):
        trainer.train(ctx)
    assert train_epoch_events(rec), "no train epoch event before cancellation"


def test_c7_evaluate_returns_finite_float_metrics(trained):
    metrics, items = trained["metrics"], trained["items"]
    want = {"map50", "map50_95", "precision", "recall", "best_epoch", "train_images", "val_images", "val_boxes"}
    assert want <= set(metrics), sorted(metrics)
    for key, value in metrics.items():
        assert isinstance(value, float), (key, value)
        assert math.isfinite(value), (key, value)
    counts = expected_counts(items)
    assert metrics["train_images"] == counts["train"]["items"]
    assert metrics["val_images"] == counts["val"]["items"]
    assert metrics["val_boxes"] == counts["val"]["boxes"]
    assert metrics["best_epoch"] in (1.0, 2.0)
    for key in ("map50", "map50_95", "precision", "recall"):
        assert 0.0 <= metrics[key] <= 1.0, key


def test_c8_export_writes_onnx_bundle_zip(trained):
    result = trained["result"]
    assert result.format == "onnx-bundle"
    assert zipfile.is_zipfile(result.path)
    assert {"model.onnx", "manifest.json", "model_card.json"} <= set(trained["bundle"])


def test_c8_manifest_contents(trained):
    bundle, ctx, metrics = trained["bundle"], trained["ctx"], trained["metrics"]
    manifest = bundle_json(bundle, "manifest.json")
    assert manifest["pipeline"] == "yolo_detector"
    assert manifest["task"] == "detection"
    assert manifest["labels"] == ctx.labels == list(DET_LABELS)
    inp = manifest["input"]
    assert inp["size"] == 64
    assert inp["layout"] == "NCHW"
    assert inp["scale"] == "1/255"
    assert inp["letterbox"] is True
    assert "base_model" in manifest and manifest["base_model"]
    for key, value in metrics.items():
        assert manifest["metrics"][key] == pytest.approx(value), key
    assert "parity" in manifest
    assert "model.onnx" in manifest["files"]
    for name, digest in manifest["files"].items():
        assert hashlib.sha256(bundle[name]).hexdigest() == digest, name


def test_c8_model_card_contents(trained):
    ctx, metrics, items = trained["ctx"], trained["metrics"], trained["items"]
    card = bundle_json(trained["bundle"], "model_card.json")
    assert ctx.params["model"] in card["starting_weights"]
    for key, value in ctx.params.items():
        if isinstance(value, float):
            assert card["params"][key] == pytest.approx(value), key
        else:
            assert card["params"][key] == value, key
    for key, value in metrics.items():
        assert card["metrics"][key] == pytest.approx(value), key
    expected = expected_counts(items)
    split = card["split"]
    for side in ("train", "val"):
        assert split[side]["items"] == expected[side]["items"], side
        assert split[side]["boxes"] == expected[side]["boxes"], side
        assert set(split[side]["groups"]) == expected[side]["groups"], side
    assert not set(split["train"]["groups"]) & set(split["val"]["groups"])
    assert len(card["epochs"]) == ctx.params["epochs"]
    assert card["device"] == "cpu"  # the resolved training device, as the conventional card records it
    cm = card["confusion_matrix"]
    labels = list(ctx.labels)
    assert cm["labels"] == [*labels, "background"] and cm["rows"] == "true"
    assert len(cm["matrix"]) == len(labels) + 1
    # Every validation box lands exactly once in its true-class row (hit, wrong class, or missed).
    assert sum(sum(row) for row in cm["matrix"][: len(labels)]) == split["val"]["boxes"]


def test_confusion_report_transposes_to_true_rows():
    from sparrow_finetune.pipelines.yolo_detector import confusion_report

    # ultralytics layout: rows predicted, columns true, last index background.
    raw = [[5, 1, 2], [0, 3, 0], [4, 0, 0]]
    report = confusion_report(raw, ["deer", "fox"])
    assert report["labels"] == ["deer", "fox", "background"]
    assert report["matrix"] == [[5, 0, 4], [1, 3, 0], [2, 0, 0]]
    assert confusion_report([[1]], ["deer"]) is None and confusion_report(None, ["deer"]) is None


def test_c9_export_records_parity_within_gate(trained):
    parity = bundle_json(trained["bundle"], "manifest.json")["parity"]
    assert isinstance(parity["max_abs_delta"], float)
    assert 0.0 <= parity["max_abs_delta"] <= 1e-3


def test_c9_onnx_input_is_static_at_training_imgsz(trained):
    import onnxruntime as ort

    sess = ort.InferenceSession(trained["bundle"]["model.onnx"], providers=["CPUExecutionProvider"])
    (inp,) = sess.get_inputs()
    assert list(inp.shape) == [1, 3, DET_SIZE, DET_SIZE]
    outs = sess.run(None, {inp.name: np.zeros((1, 3, DET_SIZE, DET_SIZE), dtype=np.float32)})
    assert outs and all(np.isfinite(np.asarray(o)).all() for o in outs)


def test_parity_delta_scales_box_rows_by_imgsz():
    """Box rows are in pixels (0..imgsz); float32 noise there is ~1e-3 px and must not fail the gate,
    while a class-score row drifting by 2e-3 must still count in full."""
    from sparrow_finetune.pipelines.yolo_detector import PARITY_MAX_ABS_DELTA, parity_delta

    ref = np.zeros((1, 4 + 2, 8), dtype=np.float32)
    ref[0, :4] = 600.0
    noisy_boxes = ref.copy()
    noisy_boxes[0, :4] += 0.0012
    assert parity_delta(noisy_boxes, ref, imgsz=640) <= PARITY_MAX_ABS_DELTA

    drifted_scores = ref.copy()
    drifted_scores[0, 5, 3] += 0.002
    assert parity_delta(drifted_scores, ref, imgsz=640) == pytest.approx(0.002, abs=1e-6)


# --------------------------------------------------------------------------- MegaDetector v6 starting weights


@pytest.mark.parametrize("name", ["mdv6-yolov10-c", "mdv6-yolov10-e"])
def test_megadetector_names_are_valid_models(name):
    assert new_trainer().validate_params({"model": name})["model"] == name


def test_megadetector_batch_estimate_uses_its_arch():
    from sparrow_finetune.pipelines.yolo_detector import _mb_per_image

    assert _mb_per_image("mdv6-yolov10-c") == _mb_per_image("yolov10n.pt")
    assert _mb_per_image("mdv6-yolov10-e") == _mb_per_image("yolov10x.pt")
    assert _mb_per_image("mdv6-yolov10-e") > _mb_per_image("mdv6-yolov10-c")


@pytest.fixture
def fake_md(monkeypatch, tmp_path):
    """Pin ``mdv6-yolov10-c`` to a small payload; ``downloads`` records each Zenodo fetch."""
    from sparrow_finetune.pipelines import yolo_detector as yd

    payload = b"md weights"
    spec = {**yd.MEGADETECTOR_BASES["mdv6-yolov10-c"], "sha256": hashlib.sha256(payload).hexdigest()}
    monkeypatch.setitem(yd.MEGADETECTOR_BASES, "mdv6-yolov10-c", spec)
    downloads: list[str] = []
    # "bodies", when set, is consumed one per download before falling back to "body".
    state = {"body": payload, "error": None, "bodies": []}

    def download(url, dest, size):
        downloads.append(url)
        if state["error"] is not None:
            raise state["error"]
        dest.write_bytes(state["bodies"].pop(0) if state["bodies"] else state["body"])

    monkeypatch.setattr(yd, "_download", download)
    return yd, spec, payload, downloads, state


def test_fetch_megadetector_prefers_weights_dir(fake_md, tmp_path):
    yd, spec, payload, downloads, _ = fake_md
    local = tmp_path / "weights" / spec["file"]
    local.parent.mkdir()
    local.write_bytes(payload)
    got = yd.fetch_megadetector(
        "mdv6-yolov10-c", local=lambda f: str(local), cache_dir=tmp_path / "cache"
    )
    assert got == local and downloads == []

    local.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="does not match"):
        yd.fetch_megadetector(
            "mdv6-yolov10-c", local=lambda f: str(local), cache_dir=tmp_path / "cache"
        )


def test_fetch_megadetector_downloads_once_then_uses_the_cache(fake_md, tmp_path):
    yd, spec, payload, downloads, _ = fake_md
    msgs: list[str] = []
    cache = tmp_path / "cache"
    first = yd.fetch_megadetector(
        "mdv6-yolov10-c", local=lambda f: f, cache_dir=cache, emit=msgs.append
    )
    assert first == cache / spec["file"] and first.read_bytes() == payload
    assert len(downloads) == 1 and spec["file"] in downloads[0]
    assert "zenodo.org" in downloads[0]
    assert msgs and "downloading MegaDetector v6 compact" in msgs[0]
    again = yd.fetch_megadetector("mdv6-yolov10-c", local=lambda f: f, cache_dir=cache)
    assert again == first and len(downloads) == 1
    assert [p.name for p in cache.iterdir()] == [spec["file"]]


def test_fetch_megadetector_rejects_a_bad_download(fake_md, tmp_path):
    yd, spec, _, downloads, state = fake_md
    state["body"] = b"corrupt"
    cache = tmp_path / "cache"
    with pytest.raises(ValueError, match="does not match") as err:
        yd.fetch_megadetector("mdv6-yolov10-c", local=lambda f: f, cache_dir=cache)
    assert len(downloads) == yd._DOWNLOAD_ATTEMPTS
    # The partial file is already gone, so the message must not ask the user to delete it.
    assert "delete it" not in str(err.value)
    assert f"{yd._DOWNLOAD_ATTEMPTS} attempts" in str(err.value)
    assert list(cache.iterdir()) == []


def test_fetch_megadetector_retries_a_corrupt_download(fake_md, tmp_path):
    """Run 4 of the feature matrix: one truncated 128 MB Zenodo download failed the whole job."""
    yd, spec, payload, downloads, state = fake_md
    state["bodies"] = [b"truncated"]
    msgs: list[str] = []
    cache = tmp_path / "cache"
    got = yd.fetch_megadetector("mdv6-yolov10-c", local=lambda f: f, cache_dir=cache, emit=msgs.append)
    assert got.read_bytes() == payload and len(downloads) == 2
    assert any("retrying" in m for m in msgs)
    assert [p.name for p in cache.iterdir()] == [spec["file"]]


def test_fetch_megadetector_offline_error_names_the_weights_dir(fake_md, tmp_path):
    yd, _, _, downloads, state = fake_md
    state["error"] = OSError("network unreachable")
    cache = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="SPARROW_FT_WEIGHTS_DIR"):
        yd.fetch_megadetector("mdv6-yolov10-c", local=lambda f: f, cache_dir=cache)
    assert len(downloads) == yd._DOWNLOAD_ATTEMPTS
    assert list(cache.iterdir()) == []


class _ShortResponse(io.BytesIO):
    """An HTTP body that ends early, as http.client returns it when the connection drops."""

    def __init__(self, body: bytes, length: int | None):
        super().__init__(body)
        self.headers = {} if length is None else {"Content-Length": str(length)}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.mark.parametrize("length", [10, None])
def test_download_rejects_a_short_body(monkeypatch, tmp_path, length):
    import urllib.request

    from sparrow_finetune.pipelines import yolo_detector as yd

    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout: _ShortResponse(b"12345", length))
    with pytest.raises(OSError, match="5 of 10 bytes"):
        yd._download("https://example.org/w.pt", tmp_path / "w.part", 10)


def test_download_accepts_a_complete_body(monkeypatch, tmp_path):
    import urllib.request

    from sparrow_finetune.pipelines import yolo_detector as yd

    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout: _ShortResponse(b"12345", 5))
    dest = tmp_path / "w.part"
    yd._download("https://example.org/w.pt", dest, 5)
    assert dest.read_bytes() == b"12345"


def test_weights_cache_is_shared_across_jobs(tmp_path):
    from sparrow_finetune.pipelines.yolo_detector import _weights_cache

    work = tmp_path / "data" / "jobs" / "12" / "attempt-1" / "work"
    work.mkdir(parents=True)
    assert _weights_cache(work) == (tmp_path / "data" / "cache" / "weights").resolve()
    other = tmp_path / "scratch"
    other.mkdir()
    assert _weights_cache(other) == other.resolve() / "weights"


def test_prepare_records_the_fetched_megadetector_weights(fake_md, tmp_path):
    yd, spec, payload, _, _ = fake_md
    src = tmp_path / "w" / spec["file"]
    src.parent.mkdir()
    src.write_bytes(payload)
    trainer = new_trainer(lambda *a, **k: None)
    trainer._resolve_model = lambda name: str(src) if name == spec["file"] else name
    items = det_snapshot_items(unit_items(), tmp_path / "images")
    ctx, _ = make_context(
        trainer, items, tmp_path / "work", {**CPU_PARAMS, "model": "mdv6-yolov10-c"}
    )
    trainer.prepare(ctx)
    import json

    assert json.loads((tmp_path / "work" / "prepare.json").read_text())["start_weights"] == str(src)


def test_is_end2end_follows_the_head_flag_and_checks_the_shape():
    from sparrow_finetune.pipelines.yolo_detector import is_end2end

    assert is_end2end(False, np.zeros((1, 6, 84))) is False
    assert is_end2end(True, np.zeros((1, 84, 6))) is True
    with pytest.raises(RuntimeError, match="expected"):
        is_end2end(True, np.zeros((1, 6, 84)))


def test_parity_delta_end2end_matches_rows_by_class_not_position():
    from sparrow_finetune.pipelines.yolo_detector import parity_delta_end2end

    ref = np.array(
        [[[10, 10, 50, 50, 0.80, 0], [12, 12, 52, 52, 0.79, 1], [0, 0, 0, 0, 0.0, 0]]],
        dtype=np.float32,
    )
    swapped = ref[:, [1, 0, 2]]
    assert parity_delta_end2end(swapped, ref, imgsz=640) == pytest.approx(0.0)
    moved = ref.copy()
    moved[0, 0, 0] += 6.4
    assert parity_delta_end2end(moved, ref, imgsz=640) == pytest.approx(0.01, abs=1e-6)
    lost = ref.copy()
    lost[0, 1, 5] = 0
    assert parity_delta_end2end(lost, ref, imgsz=640) == 1.0


# --------------------------------------------------------------------------- FT-38 near-duplicates


def test_ft38_prepare_records_one_dhash_and_side_per_image(tmp_path):
    import json

    _, ctx, _, items = prepared(tmp_path)
    hashes = np.load(ctx.workdir / "dhash.npy")
    sides = json.loads((ctx.workdir / "image_sides.json").read_text())
    assert hashes.dtype == np.uint64 and len(hashes) == len(sides) == len(items)
    assert sides == [expected_side(it) for it in items]


def test_ft38_near_duplicate_report_counts_val_images_near_a_train_image():
    from types import SimpleNamespace

    from sparrow_finetune.pipelines.near_dup import near_duplicate_report

    events: list = []
    ctx = SimpleNamespace(
        params={"near_dup_max_distance": 4},
        emit=lambda stage, msg, **kw: events.append((stage, msg, kw)),
    )
    hashes = np.asarray([0, 0b1, 2**64 - 1], dtype=np.uint64)
    metrics: dict = {}
    block = near_duplicate_report(
        ctx, hashes, np.array([True, False, False]), np.array([False, True, True]), metrics, unit="image"
    )
    assert block == {"max_distance": 4, "val_images": 1, "val_fraction": 0.5}
    assert metrics["near_dup_val_images"] == 1.0 and "near_dup_val_crops" not in metrics
    assert len(events) == 1 and "val image(s)" in events[0][1]


def test_ft38_evaluate_and_card_carry_the_near_duplicate_block(trained):
    import json

    metrics = trained["metrics"]
    assert "near_dup_val_images" in metrics and "near_dup_val_fraction" in metrics
    report = json.loads((trained["ctx"].workdir / "evaluate.json").read_text())
    card = bundle_json(trained["bundle"], "model_card.json")
    assert card["near_duplicates"] == report["near_duplicates"]
    assert set(card["near_duplicates"]) == {"max_distance", "val_images", "val_fraction"}
    assert card["params"]["near_dup_max_distance"] == 4
