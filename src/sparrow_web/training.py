"""The tunable training settings, declared once.

One list drives three things that would otherwise drift apart: the form, the
validation of what comes back from it, and the record written to the database.
A setting added here appears in the UI, is range-checked on submit, and is
stored against the run — with no other edits.

Everything named here is a real Ultralytics ``model.train()`` argument. The
seven the launcher takes directly (imgsz, epochs, batch, device, workers,
patience, amp) are passed as such; the rest ride through ``training.extra``,
which the pipeline already forwards verbatim. That is why a new knob costs
nothing but a line in this file.

Defaults match Ultralytics' own unless a comment says why they differ, because a
value that silently disagrees with the documentation is worse than no default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Baked into the image at /workspace/weights, so a run starts offline and
#: immediately. All are COCO-pretrained: this is transfer learning, and a
#: ``.yaml`` here would train from scratch instead.
BASE_MODELS: Sequence[Tuple[str, str]] = (
    ("yolo11n.pt", "Nano — fastest, lowest accuracy"),
    ("yolo11s.pt", "Small"),
    ("yolo11m.pt", "Medium"),
    ("yolo11l.pt", "Large — best fit for 32 GiB at 1024px"),
    ("yolo11x.pt", "Extra large — roughly half the batch and throughput of Large"),
)

DEFAULT_MODEL = "yolo11l.pt"

#: Keys the launcher takes as named arguments. Everything else goes to `extra`.
LAUNCHER_KEYS = ("imgsz", "epochs", "batch", "device", "workers", "patience", "amp")


@dataclass(frozen=True)
class Setting:
    name: str
    label: str
    kind: str                      # int | float | bool | choice | text
    default: Any
    help: str = ""
    group: str = "Schedule"
    advanced: bool = False
    choices: Sequence[Tuple[str, str]] = ()
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    step: Optional[float] = None
    unit: str = ""


# --------------------------------------------------------------- schedule --

SETTINGS: List[Setting] = [
    Setting("epochs", "Epochs", "int", 100, group="Schedule",
            minimum=1, maximum=2000,
            help="Passes over the training set. Early stopping usually ends a run "
                 "before this, so a generous number costs nothing."),
    Setting("patience", "Early stopping patience", "int", 30, group="Schedule",
            minimum=0, maximum=1000,
            help="Stop when accuracy has not improved for this many epochs. 0 disables "
                 "it. The best checkpoint is kept either way. Note that mosaic is "
                 "switched off for the last few epochs and usually gives a step up in "
                 "accuracy — stopping too eagerly means never seeing it."),
    Setting("close_mosaic", "Turn mosaic off for the final", "int", 10, group="Schedule",
            minimum=0, maximum=200, unit="epochs",
            help="Mosaic augmentation helps early and hurts at the end, because the "
                 "model finishes on images that look like the ones it will be given "
                 "in the field."),
    Setting("time", "Stop after", "float", 0.0, group="Schedule", advanced=True,
            minimum=0, maximum=336, step=0.5, unit="hours",
            help="A wall-clock limit that overrides epochs. 0 means no limit."),

    # ----------------------------------------------------------- image ----
    Setting("imgsz", "Image size", "choice", 1024, group="Image",
            choices=(("640", "640 — fastest"), ("768", "768"), ("896", "896"),
                     ("1024", "1024 — recommended"), ("1280", "1280 — small animals"),
                     ("1536", "1536 — slow, needs a small batch")),
            help="Images are resized to this before training. Larger sees smaller "
                 "animals but costs memory quadratically, so the batch must shrink. "
                 "Must be a multiple of 32."),
    Setting("rect", "Rectangular batches", "bool", False, group="Image", advanced=True,
            help="Pad to the batch's aspect ratio instead of a square. Faster, but "
                 "interacts badly with shuffling; leave off unless throughput matters "
                 "more than accuracy."),
    Setting("multi_scale", "Vary image size during training", "bool", False,
            group="Image", advanced=True,
            help="Randomly scales each batch by ±50%. Helps the model cope with "
                 "animals at unfamiliar distances, at roughly 30% more time per epoch."),

    # -------------------------------------------------------- optimiser ---
    Setting("optimizer", "Optimiser", "choice", "auto", group="Optimiser",
            choices=(("auto", "Auto — chooses AdamW and a learning rate for you"),
                     ("SGD", "SGD"), ("Adam", "Adam"), ("AdamW", "AdamW"),
                     ("NAdam", "NAdam"), ("RAdam", "RAdam"), ("RMSProp", "RMSProp")),
            help="Auto is the right answer unless you are deliberately comparing. It "
                 "also overrides the learning rate and momentum below."),
    Setting("lr0", "Initial learning rate", "float", 0.01, group="Optimiser",
            minimum=0.00001, maximum=1.0, step=0.0001,
            help="How large a step each update takes. Lower it (0.002 or so) when "
                 "fine-tuning from one of your own models rather than from COCO — the "
                 "weights are already close, and a big step throws that away."),
    Setting("lrf", "Final learning rate fraction", "float", 0.01, group="Optimiser",
            advanced=True, minimum=0.0001, maximum=1.0, step=0.001,
            help="The schedule decays to initial × this by the last epoch."),
    Setting("cos_lr", "Cosine schedule", "bool", True, group="Optimiser",
            help="Decay the learning rate along a cosine curve rather than a straight "
                 "line. Usually converges a little better over a long run."),
    Setting("momentum", "Momentum", "float", 0.937, group="Optimiser", advanced=True,
            minimum=0.0, maximum=0.999, step=0.001),
    Setting("weight_decay", "Weight decay", "float", 0.0005, group="Optimiser",
            advanced=True, minimum=0.0, maximum=0.1, step=0.0001,
            help="Penalises large weights. Raising it can help a small dataset overfit "
                 "less."),
    Setting("warmup_epochs", "Warm-up epochs", "float", 3.0, group="Optimiser",
            advanced=True, minimum=0.0, maximum=50.0, step=0.5,
            help="Ramp the learning rate up over the first few epochs so early "
                 "batches cannot wreck the pretrained weights."),
    Setting("dropout", "Dropout", "float", 0.0, group="Optimiser", advanced=True,
            minimum=0.0, maximum=0.9, step=0.05),
    Setting("freeze", "Freeze the first", "int", 0, group="Optimiser",
            minimum=0, maximum=24, unit="layers",
            help="0 trains everything, which is right when you have thousands of "
                 "tagged images. Freezing 10 (the backbone) is faster and overfits "
                 "less on a few hundred, but caps accuracy — camera-trap imagery is "
                 "infrared, dark and motion-blurred, quite unlike the COCO photos the "
                 "backbone learned from, so it usually benefits from adapting."),

    # ------------------------------------------------------- loss gains ---
    Setting("box", "Box loss weight", "float", 7.5, group="Loss", advanced=True,
            minimum=0.0, maximum=50.0, step=0.1,
            help="How much box position matters relative to classification."),
    Setting("cls", "Class loss weight", "float", 0.5, group="Loss", advanced=True,
            minimum=0.0, maximum=20.0, step=0.05,
            help="Raise it when telling species apart matters more than exact boxes — "
                 "which for camera-trap work it usually does."),
    Setting("dfl", "Distribution focal loss weight", "float", 1.5, group="Loss",
            advanced=True, minimum=0.0, maximum=20.0, step=0.05),

    # ----------------------------------------------------- augmentation ---
    Setting("mosaic", "Mosaic", "float", 1.0, group="Augmentation",
            minimum=0.0, maximum=1.0, step=0.1,
            help="Stitches four images into one. Strong augmentation that teaches "
                 "scale and context; switched off for the final epochs above."),
    Setting("fliplr", "Horizontal flip", "float", 0.5, group="Augmentation",
            minimum=0.0, maximum=1.0, step=0.1,
            help="Safe for animals — a mirrored elephant is still an elephant."),
    Setting("flipud", "Vertical flip", "float", 0.0, group="Augmentation",
            minimum=0.0, maximum=1.0, step=0.1,
            help="Leave at 0. Camera traps are never upside down, so this only "
                 "teaches the model something it will never see."),
    Setting("scale", "Scale jitter", "float", 0.5, group="Augmentation",
            minimum=0.0, maximum=0.95, step=0.05,
            help="Random zoom. Helps with animals at varying distances from the trap."),
    Setting("degrees", "Rotation", "float", 0.0, group="Augmentation",
            minimum=0.0, maximum=45.0, step=1.0, unit="degrees",
            help="A few degrees can help with tilted mountings; large values are "
                 "unrealistic for a fixed camera."),
    Setting("translate", "Translation", "float", 0.1, group="Augmentation",
            minimum=0.0, maximum=0.9, step=0.05),
    Setting("shear", "Shear", "float", 0.0, group="Augmentation", advanced=True,
            minimum=0.0, maximum=20.0, step=0.5),
    Setting("perspective", "Perspective", "float", 0.0, group="Augmentation",
            advanced=True, minimum=0.0, maximum=0.001, step=0.0001),
    Setting("hsv_h", "Hue jitter", "float", 0.015, group="Augmentation", advanced=True,
            minimum=0.0, maximum=1.0, step=0.005),
    Setting("hsv_s", "Saturation jitter", "float", 0.7, group="Augmentation",
            advanced=True, minimum=0.0, maximum=1.0, step=0.05,
            help="Worth keeping high: infrared night frames are effectively greyscale "
                 "while daytime frames are not."),
    Setting("hsv_v", "Brightness jitter", "float", 0.4, group="Augmentation",
            advanced=True, minimum=0.0, maximum=1.0, step=0.05),
    Setting("mixup", "Mixup", "float", 0.0, group="Augmentation", advanced=True,
            minimum=0.0, maximum=1.0, step=0.05,
            help="Blends two images together. Rarely helps detection."),
    Setting("copy_paste", "Copy-paste", "float", 0.0, group="Augmentation",
            advanced=True, minimum=0.0, maximum=1.0, step=0.05,
            help="Segmentation-oriented; has no effect on a detection dataset."),
    Setting("erasing", "Random erasing", "float", 0.4, group="Augmentation",
            advanced=True, minimum=0.0, maximum=0.9, step=0.05),

    # ---------------------------------------------------------- hardware --
    Setting("workers", "Data loader workers", "text", "auto", group="Hardware",
            help="'auto' derives from the CPU count. Lower it if the dataloader runs "
                 "out of shared memory."),
    Setting("amp", "Mixed precision", "bool", True, group="Hardware",
            help="Half precision where it is safe. Large speedup, no measurable "
                 "accuracy cost. Turn it off only to rule it out as a cause of NaNs."),
    Setting("cache", "Cache images", "choice", "false", group="Hardware",
            choices=(("false", "No caching"), ("true", "In RAM — needs the whole "
                     "dataset resident"), ("disk", "On disk as .npy")),
            help="RAM caching is a large speedup but unrealistic for tens of thousands "
                 "of camera-trap images at 1024px."),
    Setting("seed", "Random seed", "int", 0, group="Hardware", advanced=True,
            minimum=0, maximum=2**31 - 1),
    Setting("deterministic", "Deterministic", "bool", False, group="Hardware",
            advanced=True,
            help="Reproducible at roughly 10% less throughput, because cuDNN cannot "
                 "autotune. Worth it only when comparing runs exactly."),
    Setting("single_cls", "Treat all species as one class", "bool", False,
            group="Hardware", advanced=True,
            help="Trains an animal/no-animal detector. Useful as a first pass over a "
                 "dataset whose species labels you do not trust yet."),
]

GROUP_ORDER = ("Schedule", "Image", "Optimiser", "Loss", "Augmentation", "Hardware")

BY_NAME: Dict[str, Setting] = {s.name: s for s in SETTINGS}


def groups(advanced: bool = True) -> List[Tuple[str, List[Setting]]]:
    """Settings bucketed for display, in a deliberate order."""
    out = []
    for name in GROUP_ORDER:
        members = [s for s in SETTINGS if s.group == name and (advanced or not s.advanced)]
        if members:
            out.append((name, members))
    return out


def defaults() -> Dict[str, Any]:
    return {s.name: s.default for s in SETTINGS}


# ----------------------------------------------------------------- parse ---

class ValidationError(ValueError):
    """Carries per-field messages so the form can mark the offending input."""

    def __init__(self, errors: Dict[str, str]):
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))
        self.errors = errors


def _coerce(setting: Setting, raw: Any) -> Any:
    if setting.kind == "bool":
        # An unchecked HTML checkbox submits nothing at all, so absence is False
        # and any present value is True.
        return raw not in (None, "", "0", "false", "off", False)
    if raw in (None, ""):
        return setting.default
    if setting.kind == "int":
        return int(float(str(raw)))          # tolerate "10.0" from a number input
    if setting.kind == "float":
        return float(str(raw))
    if setting.kind == "choice":
        value = str(raw)
        allowed = {c[0] for c in setting.choices}
        if value not in allowed:
            raise ValueError(f"must be one of {', '.join(sorted(allowed))}")
        return int(value) if setting.name == "imgsz" else value
    return str(raw).strip()


def parse_batch(raw: Any) -> Any:
    """``auto``, an int, ``-1``, or a float in (0, 1).

    The float form is Ultralytics' target GPU-utilisation fraction and must stay
    a float: coercing 0.7 to an int gives 0, which fails deep inside the trainer
    with an error that says nothing about the batch size.
    """
    if raw in (None, "", "auto"):
        return "auto"
    text = str(raw).strip().lower()
    if text == "auto":
        return "auto"
    try:
        value = float(text)
    except ValueError:
        raise ValueError("must be 'auto', a whole number, or a fraction like 0.8")
    if value == -1:
        return -1
    if 0 < value < 1:
        return value
    if value >= 1 and value == int(value):
        return int(value)
    raise ValueError("must be 'auto', a whole number, or a fraction like 0.8")


def parse_workers(raw: Any) -> Any:
    if raw in (None, "", "auto"):
        return "auto"
    try:
        n = int(float(str(raw)))
    except ValueError:
        raise ValueError("must be 'auto' or a whole number")
    if n < 0:
        raise ValueError("cannot be negative")
    return n


def parse(form: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a submitted form into the settings to store and train with.

    Raises :class:`ValidationError` with one message per bad field, so the form
    can show all of them at once rather than one per round trip.
    """
    errors: Dict[str, str] = {}
    values: Dict[str, Any] = {}

    for setting in SETTINGS:
        try:
            if setting.name == "workers":
                value = parse_workers(form.get("workers"))
            else:
                value = _coerce(setting, form.get(setting.name))
                if setting.minimum is not None and float(value) < setting.minimum:
                    raise ValueError(f"cannot be below {setting.minimum:g}")
                if setting.maximum is not None and float(value) > setting.maximum:
                    raise ValueError(f"cannot be above {setting.maximum:g}")
        except ValidationError:
            raise
        except ValueError as e:
            errors[setting.name] = str(e)
            continue
        values[setting.name] = value

    try:
        values["batch"] = parse_batch(form.get("batch"))
    except ValueError as e:
        errors["batch"] = str(e)

    if "imgsz" in values and int(values["imgsz"]) % 32:
        errors["imgsz"] = "must be a multiple of 32"

    model = str(form.get("model") or DEFAULT_MODEL)
    if model not in {m[0] for m in BASE_MODELS}:
        errors["model"] = "unknown model"
    values["model"] = model

    values["device"] = str(form.get("device") or "").strip()
    if not values["device"]:
        errors["device"] = "choose a GPU"

    # Ultralytics reads `time` as "no limit" only when absent, not when zero.
    if values.get("time") in (0, 0.0):
        values.pop("time", None)

    if errors:
        raise ValidationError(errors)
    return values


def split_params(values: Dict[str, Any]) -> Dict[str, Any]:
    """Split validated settings into the launcher's named arguments plus `extra`.

    ``extra`` is forwarded to ``model.train()`` untouched, which is why adding a
    setting to this module needs no change in the pipeline.
    """
    params = {k: values[k] for k in LAUNCHER_KEYS if k in values}
    params["extra"] = {k: v for k, v in values.items()
                       if k not in LAUNCHER_KEYS and k != "model"}
    return params
