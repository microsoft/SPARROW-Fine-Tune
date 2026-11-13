"""Pins for the Phase 1C review findings I1-I6 (docs/review/implement-yolo-detector/report.md).

I1 a train/evaluate/export run leaves process-wide state (CUDA_VISIBLE_DEVICES, deterministic mode) as it found it.
I2 EXIF-rotated images: written dataset pixels match the raw-frame labels ultralytics reads with them.
I3 every accepted upload format (GIF, ...) reaches ultralytics; prepare counts equal what ultralytics loads.
I4 ``model`` is restricted to an ultralytics model name: no paths, no URLs, no arbitrary pickles.
I5 every label needs boxes in both the train and the val split.
I6 ``best_epoch`` names the epoch whose weights ``best.pt`` holds (ultralytics keeps the LAST tied epoch).
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pytest
from PIL import Image

from .dinov3_helpers import make_context
from .yolo_helpers import (
    CPU_PARAMS,
    DET_SIZE,
    DetItem,
    det_boxes,
    det_items,
    det_png,
    det_snapshot_items,
    pick_groups,
)


@pytest.fixture(scope="module", autouse=True)
def yolo_offline():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("YOLO_OFFLINE", "1")
        yield


def new_trainer():
    from sparrow_finetune.pipelines.yolo_detector import YoloDetectorTrainer

    return YoloDetectorTrainer()


_ENV_KEYS = ("CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED")
_SENTINELS = {
    "CUDA_VISIBLE_DEVICES": "0,1",
    "CUBLAS_WORKSPACE_CONFIG": ":16:8",
    "PYTHONHASHSEED": "1234",
}


def _global_state() -> dict:
    import torch

    return {
        **{k: os.environ.get(k) for k in _ENV_KEYS},
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }


@pytest.fixture(scope="module")
def three_epoch_run(tmp_path_factory):
    """One 3-epoch CPU run of train/evaluate/export; global state captured before and after each stage."""
    base = tmp_path_factory.mktemp("yolo-pins")
    trainer = new_trainer()
    items = det_snapshot_items(det_items(prefix="p-"), base / "images")
    with pytest.MonkeyPatch.context() as mp:
        for k, v in _SENTINELS.items():
            mp.setenv(k, v)
        ctx, rec = make_context(
            trainer, items, base / "work", {**CPU_PARAMS, "epochs": 3}
        )
        trainer.prepare(ctx)
        before = _global_state()
        after = {}
        trainer.train(ctx)
        after["train"] = _global_state()
        trainer.evaluate(ctx)
        after["evaluate"] = _global_state()
        trainer.export(ctx)
        after["export"] = _global_state()
    return {"before": before, "after": after, "workdir": base / "work"}


# --------------------------------------------------------------------------- I1


@pytest.mark.parametrize("stage", ["train", "evaluate", "export"])
def test_i1_stage_restores_process_global_state(three_epoch_run, stage):
    before = three_epoch_run["before"]
    assert before["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert three_epoch_run["after"][stage] == before


# --------------------------------------------------------------------------- I6


def test_i6_best_epoch_is_the_epoch_saved_in_best_pt(three_epoch_run):
    import torch

    train = json.loads((three_epoch_run["workdir"] / "train.json").read_text())
    weights = Path(train["weights"]).parent

    def state(name):
        return (
            torch.load(weights / name, weights_only=False)["model"].float().state_dict()
        )

    best, last = state("best.pt"), state("last.pt")
    if all(torch.equal(best[k], last[k]) for k in best):
        assert train["best_epoch"] == 3
    else:
        assert train["best_epoch"] < 3


# --------------------------------------------------------------------------- I2


def _exif_rotated_jpeg() -> tuple[bytes, list]:
    """64x32 raw frame, a red square at raw x=44..60, y=8..24, EXIF Orientation=6 (display rotated 90 CW)."""
    im = Image.new("RGB", (64, 32), (128, 128, 128))
    im.paste((220, 40, 40), (44, 8, 60, 24))
    exif = Image.Exif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    im.save(buf, format="JPEG", exif=exif, quality=95)
    return buf.getvalue(), [44.0, 8.0, 16.0, 16.0]


def _written_image(dataset: Path, position: int) -> Path:
    found = [p for p in (dataset / "images").glob(f"*/{position:06d}.*")]
    assert len(found) == 1, found
    return found[0]


def test_i2_exif_rotated_image_pixels_match_raw_frame_label(tmp_path):
    from sparrow_finetune.pipelines import SnapshotItem
    from ultralytics.utils.patches import imread

    trainer = new_trainer()
    items = det_snapshot_items(det_items(prefix="e-"), tmp_path / "images")
    data, bbox = _exif_rotated_jpeg()
    path = tmp_path / "images" / "exif.jpg"
    path.write_bytes(data)
    items.append(
        SnapshotItem(
            path, 64, 32, [{"label": "cat", "bbox": bbox}], "e-exif", "exif.jpg", "x"
        )
    )
    ctx, _ = make_context(trainer, items, tmp_path / "work", CPU_PARAMS)
    trainer.prepare(ctx)

    position = len(items) - 1
    image = _written_image(tmp_path / "work" / "dataset", position)
    side = image.parent.name
    label = (
        (tmp_path / "work" / "dataset" / "labels" / side / f"{position:06d}.txt")
        .read_text()
        .split()
    )
    cx, cy, bw, bh = (float(v) for v in label[1:])
    arr = imread(str(image))  # BGR, exactly as the ultralytics dataset loader reads it
    assert arr is not None
    h, w = arr.shape[:2]
    assert (w, h) == (64, 32)
    box = arr[
        int((cy - bh / 2) * h) : int((cy + bh / 2) * h),
        int((cx - bw / 2) * w) : int((cx + bw / 2) * w),
    ]
    b, g, r = box.reshape(-1, 3).mean(0)
    assert r > 180 and g < 90 and b < 90


# --------------------------------------------------------------------------- I3


def test_i3_gif_uploads_reach_ultralytics_and_counts_agree(tmp_path):
    import cv2
    from sparrow_finetune.pipelines import SnapshotItem
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.data.utils import check_det_dataset

    items = []
    for pos, it in enumerate(det_items(prefix="g-")):
        gif = pos % 2 == 0
        path = tmp_path / "images" / f"{pos:06d}{'.gif' if gif else '.png'}"
        path.parent.mkdir(parents=True, exist_ok=True)
        if gif:
            buf = io.BytesIO()
            Image.open(io.BytesIO(it.png)).save(buf, format="GIF")
            path.write_bytes(buf.getvalue())
        else:
            path.write_bytes(it.png)
        anns = [{"label": lb, "bbox": [float(v) for v in bb]} for lb, bb in it.boxes]
        items.append(
            SnapshotItem(path, DET_SIZE, DET_SIZE, anns, it.group, path.name, it.sha256)
        )

    trainer = new_trainer()
    ctx, rec = make_context(trainer, items, tmp_path / "work", CPU_PARAMS)
    trainer.prepare(ctx)
    event = rec.stage_events("prepare")[-1]

    dataset = tmp_path / "work" / "dataset"
    for image in (dataset / "images").glob("*/*"):
        assert cv2.imread(str(image)) is not None, image

    data = check_det_dataset(str(dataset / "data.yaml"))
    for side in ("train", "val"):
        ds = YOLODataset(
            img_path=data[side], data=data, imgsz=DET_SIZE, augment=False, batch_size=4
        )
        assert len(ds.im_files) == event[f"{side}_images"]
        assert sum(len(lb["cls"]) for lb in ds.labels) == event[f"{side}_boxes"]


# --------------------------------------------------------------------------- I4


@pytest.mark.parametrize(
    "model",
    [
        "/tmp/evil.pt",
        "../yolo11n.pt",
        "weights/yolo11n.pt",
        "https://example.com/yolo11n.pt",
        "evil.pt",
        "yolo11n.pkl",
        "yolo11n.pt\n",
        "YOLO11N.pt/../x",
    ],
)
def test_i4_model_rejects_paths_urls_and_unknown_files(model):
    with pytest.raises(ValueError, match="model"):
        new_trainer().validate_params({"model": model})


@pytest.mark.parametrize(
    "model", ["yolo11n.pt", "yolo11s.pt", "yolo11n.yaml", "yolov8m.pt", "yolo12x.pt"]
)
def test_i4_model_accepts_ultralytics_names(model):
    assert new_trainer().validate_params({"model": model})["model"] == model


# --------------------------------------------------------------------------- I5


def _coverage_items(want_dog: dict) -> list[DetItem]:
    syn, k = [], 0
    for label, want in (("cat", {"train": 3, "val": 1}), ("dog", want_dog)):
        for g, group in enumerate(pick_groups("cov-", label, want)):
            for j in range(2):
                boxes = [b for b in det_boxes(label, k) if b[0] == label]
                syn.append(
                    DetItem(
                        f"cov-{label}-g{g}-{j}.png", group, boxes, det_png(boxes, k)
                    )
                )
                k += 1
    return syn


@pytest.mark.parametrize(
    "want_dog,side",
    [({"train": 3, "val": 0}, "val"), ({"train": 0, "val": 2}, "train")],
)
def test_i5_label_missing_from_a_split_is_rejected(tmp_path, want_dog, side):
    trainer = new_trainer()
    items = det_snapshot_items(_coverage_items(want_dog), tmp_path / "images")
    ctx, _ = make_context(trainer, items, tmp_path / "work", CPU_PARAMS)
    with pytest.raises(ValueError) as exc:
        trainer.prepare(ctx)
    message = str(exc.value)
    assert "dog" in message and side in message

