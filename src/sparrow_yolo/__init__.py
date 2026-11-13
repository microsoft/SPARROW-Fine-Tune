"""SPARROW YOLO Training Workbench.

Pascal VOC -> YOLO dataset preparation and Ultralytics training, built to run in
the project's CUDA container.

Typical headless use::

    sparrow-yolo doctor  --config configs/example_experiment.yaml
    sparrow-yolo prepare --config configs/example_experiment.yaml
    sparrow-yolo train   --config configs/example_experiment.yaml --watch

Submodules are imported lazily so that ``import sparrow_yolo`` stays cheap and,
critically, does not pull torch into a notebook kernel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.2.0"

__all__ = [
    "__version__",
    # config
    "ExperimentConfig",
    # dataset
    "AuditResult",
    "audit_dataset",
    "summarize",
    "make_class_map",
    "load_or_create_class_map",
    "split_images",
    "prepare_yolo_dataset",
    "verify_prepared",
    # hardware
    "doctor",
    "query_gpus",
    "suggest_batch",
    "suggest_workers",
    # runner
    "write_train_config",
    "launch_training_process",
    "preflight",
    "read_status",
    "stream_log",
    "tail_log",
    "stop_training",
    "best_weights",
    "list_runs",
    "unique_run_dir",
]

_LAZY = {
    "ExperimentConfig": "config",
    "AuditResult": "dataset",
    "audit_dataset": "dataset",
    "summarize": "dataset",
    "make_class_map": "dataset",
    "load_or_create_class_map": "dataset",
    "split_images": "dataset",
    "prepare_yolo_dataset": "dataset",
    "verify_prepared": "dataset",
    "doctor": "hardware",
    "query_gpus": "hardware",
    "suggest_batch": "hardware",
    "suggest_workers": "hardware",
    "write_train_config": "process_runner",
    "launch_training_process": "process_runner",
    "preflight": "process_runner",
    "read_status": "process_runner",
    "stream_log": "process_runner",
    "tail_log": "process_runner",
    "stop_training": "process_runner",
    "best_weights": "process_runner",
    "list_runs": "process_runner",
    "unique_run_dir": "process_runner",
}

if TYPE_CHECKING:  # pragma: no cover
    from .config import ExperimentConfig
    from .dataset import (
        AuditResult,
        audit_dataset,
        load_or_create_class_map,
        make_class_map,
        prepare_yolo_dataset,
        split_images,
        summarize,
        verify_prepared,
    )
    from .hardware import doctor, query_gpus, suggest_batch, suggest_workers
    from .process_runner import (
        best_weights,
        launch_training_process,
        list_runs,
        preflight,
        read_status,
        stop_training,
        stream_log,
        tail_log,
        unique_run_dir,
        write_train_config,
    )


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(f".{module}", __name__), name)


def __dir__() -> list:
    return sorted(__all__)
