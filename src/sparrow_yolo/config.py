"""Experiment configuration.

``configs/*.yaml`` is the single place experiments are defined. Previously that
file existed but nothing read it — all real settings lived as constants inside
the notebooks, which made runs impossible to reproduce or diff. This module makes
the config authoritative; the notebooks now load it too.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

DEFAULTS: Dict[str, Any] = {
    "dataset": {
        "root": "/workspace/datasets/Sub_Saharan",
        "name": "Sub_Saharan_v8",
        "normalise_class_names": True,
        "deep_image_check": False,
        "drop_classes": [],
        "min_boxes_per_class": 0,
        # reviewed | all | none - which blank frames to train on as negatives.
        "background_policy": "reviewed",
        # Directories (relative to root, globs allowed) whose XML-less images are
        # known-empty frames rather than untagged ones.
        "confirmed_blank_dirs": [],
        # {from: to} - fold misspelt or synonymous class names together.
        "merge_classes": {},
    },
    "paths": {
        "prepared_root": "/workspace/prepared",
        "runs_root": "/workspace/runs",
    },
    "split": {
        "train": 0.8,
        "val": 0.1,
        "test": 0.1,
        "seed": 42,
        "stratify": "rarest",
        "group_by": "none",
    },
    "prepare": {
        "link_mode": "auto",
        "overwrite": False,
        "reuse_class_map": True,
        # Keep existing images in the split they were assigned last time, so
        # metrics stay comparable as the dataset grows.
        "reuse_split": True,
        "workers": None,
    },
    "training": {
        "model": "yolo11l.pt",
        "imgsz": 1024,
        "epochs": 100,
        "batch": "auto",
        "device": "0",
        "workers": "auto",
        "patience": 30,
        "amp": True,
        "seed": 0,
        "deterministic": False,
        "cache": False,
        "resume": False,
        "extra": {},
    },
    "export": {
        "onnx": True,
        "opset": None,
        "dynamic": False,
        "half": False,
        "simplify": True,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class ExperimentConfig:
    dataset: Dict[str, Any] = field(default_factory=dict)
    paths: Dict[str, Any] = field(default_factory=dict)
    split: Dict[str, Any] = field(default_factory=dict)
    prepare: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    export: Dict[str, Any] = field(default_factory=dict)
    source_path: Optional[str] = None

    # -- construction ------------------------------------------------------
    @classmethod
    def from_dict(cls, raw: Dict[str, Any], source_path: Optional[str] = None) -> "ExperimentConfig":
        merged = _deep_merge(DEFAULTS, raw or {})
        cfg = cls(
            dataset=merged["dataset"],
            paths=merged["paths"],
            split=merged["split"],
            prepare=merged["prepare"],
            training=merged["training"],
            export=merged["export"],
            source_path=source_path,
        )
        cfg.validate()
        return cfg

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        path = Path(path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{path} must contain a YAML mapping")
        unknown = set(raw) - set(DEFAULTS)
        if unknown:
            raise ValueError(
                f"{path}: unknown top-level section(s) {sorted(unknown)}. "
                f"Expected any of {sorted(DEFAULTS)}"
            )
        return cls.from_dict(raw, source_path=str(path))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("source_path", None)
        return d

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
        return path

    # -- validation --------------------------------------------------------
    def validate(self) -> None:
        s = self.split
        total = float(s["train"]) + float(s["val"]) + float(s["test"])
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split fractions must sum to 1.0, got {total}")
        if s["stratify"] not in {"rarest", "dominant", "none"}:
            raise ValueError(f"split.stratify must be rarest|dominant|none, got {s['stratify']!r}")
        if self.dataset["background_policy"] not in {"reviewed", "all", "none"}:
            raise ValueError(
                f"dataset.background_policy must be reviewed|all|none, "
                f"got {self.dataset['background_policy']!r}"
            )
        if isinstance(self.dataset["confirmed_blank_dirs"], str):
            raise ValueError("dataset.confirmed_blank_dirs must be a list, not a single string")
        if not isinstance(self.dataset["merge_classes"], dict):
            raise ValueError("dataset.merge_classes must be a mapping of {from: to}")
        if self.prepare["link_mode"] not in {"auto", "hardlink", "symlink", "copy"}:
            raise ValueError(f"prepare.link_mode must be auto|hardlink|symlink|copy, got {self.prepare['link_mode']!r}")
        if int(self.training["imgsz"]) % 32 != 0:
            raise ValueError(f"training.imgsz must be a multiple of 32, got {self.training['imgsz']}")
        if int(self.training["epochs"]) < 1:
            raise ValueError("training.epochs must be >= 1")
        b = self.training["batch"]
        if not (b == "auto" or isinstance(b, (int, float))):
            raise ValueError(f"training.batch must be 'auto', an int, -1, or a float in (0,1); got {b!r}")
        if not isinstance(self.training.get("extra", {}), dict):
            raise ValueError("training.extra must be a mapping of Ultralytics arguments")

    # -- derived paths -----------------------------------------------------
    @property
    def project_name(self) -> str:
        return str(self.dataset["name"])

    @property
    def prepared_dir(self) -> Path:
        return Path(self.paths["prepared_root"]) / self.project_name

    @property
    def data_yaml(self) -> Path:
        return self.prepared_dir / "data.yaml"

    @property
    def class_map_path(self) -> Path:
        return self.prepared_dir / "classes.json"

    @property
    def splits_path(self) -> Path:
        return self.prepared_dir / "splits.json"

    @property
    def runs_root(self) -> Path:
        return Path(self.paths["runs_root"])

    def run_dir(self, run_name: Optional[str] = None) -> Path:
        return self.runs_root / (run_name or self.project_name)

    @property
    def devices(self) -> List[int]:
        from .hardware import parse_device

        return parse_device(self.training["device"])
