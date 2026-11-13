"""Command line interface.

Every stage is runnable without a notebook, which is what makes a run
reproducible: the YAML config is the record of what was done.

    sparrow-yolo doctor  --config configs/example_experiment.yaml
    sparrow-yolo audit   --config configs/example_experiment.yaml
    sparrow-yolo prepare --config configs/example_experiment.yaml
    sparrow-yolo train   --config configs/example_experiment.yaml --watch
    sparrow-yolo status  --run runs/Sub_Saharan_v8
    sparrow-yolo stop    --run runs/Sub_Saharan_v8
    sparrow-yolo eval    --run runs/Sub_Saharan_v8 --config ...
    sparrow-yolo export  --run runs/Sub_Saharan_v8 --config ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .config import ExperimentConfig


def _load(args) -> ExperimentConfig:
    if not args.config:
        raise SystemExit("--config is required for this command")
    path = Path(args.config)
    if not path.is_file():
        raise SystemExit(f"Config not found: {path}")
    cfg = ExperimentConfig.load(path)
    for override in args.set or []:
        if "=" not in override:
            raise SystemExit(f"--set expects section.key=value, got {override!r}")
        key, value = override.split("=", 1)
        parts = key.split(".")
        if len(parts) != 2 or parts[0] not in cfg.to_dict():
            raise SystemExit(f"--set key must be section.key, got {key!r}")
        section = getattr(cfg, parts[0])
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
        section[parts[1]] = parsed
    cfg.validate()
    return cfg


def _print_header(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# --------------------------------------------------------------------------

def cmd_doctor(args) -> int:
    from .hardware import doctor

    devices: List[int] = [0]
    data_yaml: Optional[str] = None
    prepared_root: Optional[str] = None
    if args.config:
        cfg = _load(args)
        devices = cfg.devices or [0]
        data_yaml = str(cfg.data_yaml) if cfg.data_yaml.exists() else None
        prepared_root = str(cfg.paths["prepared_root"])

    _print_header(f"Environment check (devices: {devices})")
    report = doctor(devices, run_nccl=not args.skip_nccl, data_yaml=data_yaml,
                    prepared_root=prepared_root)
    print(report.render())
    if report.ok:
        print("\nAll fatal checks passed.")
    else:
        print("\nFATAL checks failed - fix these before training.")
    return 0 if report.ok else 1


def cmd_audit(args) -> int:
    from .dataset import audit_dataset

    cfg = _load(args)
    _print_header(f"Auditing {cfg.dataset['root']}")
    result = audit_dataset(
        cfg.dataset["root"],
        normalise_class_names=bool(cfg.dataset["normalise_class_names"]),
        deep_image_check=bool(cfg.dataset["deep_image_check"]),
        confirmed_blank_dirs=cfg.dataset["confirmed_blank_dirs"],
        merge_classes=cfg.dataset["merge_classes"],
    )
    summary = result.summary
    for k, v in summary.items():
        if k != "classes":
            print(f"  {k}: {v}")
    print(f"  classes ({len(summary['classes'])}): {summary['classes']}")

    # Blank frames are the most commonly misunderstood part of the audit, so
    # spell out what the current policy will do with them.
    policy = cfg.dataset["background_policy"]
    with_xml = summary["blank_with_empty_xml"]
    confirmed = summary["blank_confirmed_dir"]
    unknown = summary["blank_unknown"]
    if with_xml or confirmed or unknown:
        print(f"\nBlank frames (background_policy: {policy}):")
        print(f"  {with_xml} with an empty XML       - reviewed, confirmed empty")
        print(f"  {confirmed} in a confirmed blank dir - declared via confirmed_blank_dirs")
        print(f"  {unknown} with no XML at all      - genuinely empty, or simply untagged")
        used = {
            "reviewed": with_xml + confirmed,
            "all": with_xml + confirmed + unknown,
            "none": 0,
        }[policy]
        print(f"  -> {used} will be trained on as negatives")
        if policy == "reviewed" and unknown:
            print(
                f"     To use the {unknown} without XML, either list their folder(s) in "
                f"dataset.confirmed_blank_dirs (preferred - it records which ones you "
                f"vouch for), or set background_policy: all if every image has been reviewed."
            )
        # Ultralytics recommends 0-10% background images; COCO is ~1%. Past that
        # the extra negatives mostly cost epoch time and start suppressing recall.
        labelled = summary["labelled_images"]
        if used and labelled:
            frac = used / (used + labelled)
            target = int(round(labelled / 9))   # 10% of the final training set
            if frac > 0.30:
                print(
                    f"     WARNING: {frac:.0%} of the dataset would be negatives. The usual "
                    f"guidance is 0-10% (COCO is ~1%). This lengthens every epoch and tends to "
                    f"suppress recall. Consider using ~{target} blanks instead of {used}."
                )
            elif frac > 0.12:
                print(
                    f"     NOTE: {frac:.0%} of the dataset would be negatives, above the usual "
                    f"0-10%. Fine if they are hard negatives; wasteful if they are near-duplicates."
                )

    if not result.issues.empty:
        print("\nIssues by type:")
        for issue_type, count in result.issue_counts().items():
            print(f"  {issue_type}: {count}")
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            result.issues.to_csv(args.out, index=False)
            print(f"\nFull issue list written to {args.out}")
    else:
        print("\nNo issues found.")
    return 0


def cmd_prepare(args) -> int:
    from .dataset import (
        audit_dataset,
        load_or_create_class_map,
        load_pinned_splits,
        prepare_yolo_dataset,
        split_images,
        verify_prepared,
    )

    cfg = _load(args)
    _print_header(f"Preparing {cfg.project_name}")

    result = audit_dataset(
        cfg.dataset["root"],
        normalise_class_names=bool(cfg.dataset["normalise_class_names"]),
        deep_image_check=bool(cfg.dataset["deep_image_check"]),
        confirmed_blank_dirs=cfg.dataset["confirmed_blank_dirs"],
        merge_classes=cfg.dataset["merge_classes"],
    )
    print(json.dumps(result.summary, indent=2))

    class_map, warnings = load_or_create_class_map(
        cfg.class_map_path,
        result.boxes,
        drop_classes=cfg.dataset["drop_classes"],
        min_boxes=int(cfg.dataset["min_boxes_per_class"]),
        reuse=bool(cfg.prepare["reuse_class_map"]),
    )
    for w in warnings:
        print(f"  NOTE: {w}")
    if not class_map:
        print("No classes found - nothing to prepare.")
        return 1
    print(f"\n{len(class_map)} classes: {class_map}")

    group_by = cfg.split["group_by"]
    splits = split_images(
        result.images,
        result.boxes,
        train=float(cfg.split["train"]),
        val=float(cfg.split["val"]),
        test=float(cfg.split["test"]),
        seed=int(cfg.split["seed"]),
        stratify=str(cfg.split["stratify"]),
        group_by=None if group_by in ("none", None, "") else str(group_by),
        background_policy=str(cfg.dataset["background_policy"]),
        pinned=load_pinned_splits(cfg.splits_path) if cfg.prepare["reuse_split"] else None,
    )
    print("\nSplit sizes:")
    for split, count in splits["split"].value_counts().items():
        print(f"  {split}: {count}")

    data_yaml = prepare_yolo_dataset(
        result,
        cfg.paths["prepared_root"],
        cfg.project_name,
        class_map,
        splits,
        overwrite=bool(args.overwrite or cfg.prepare["overwrite"]),
        link_mode=str(cfg.prepare["link_mode"]),
        workers=cfg.prepare["workers"],
    )
    print(f"\ndata.yaml: {data_yaml}")

    check = verify_prepared(data_yaml)
    print("\nVerification:")
    for split, stats in check["splits"].items():
        print(f"  {split}: {stats['images']} images, {stats['boxes']} boxes, {stats['backgrounds']} backgrounds")
    if check["problems"]:
        for p in check["problems"]:
            print(f"  PROBLEM: {p}")
        return 1
    print("  OK")
    return 0


def cmd_train(args) -> int:
    from .process_runner import (
        launch_training_process,
        preflight,
        resolve_resume,
        stream_log,
        unique_run_dir,
        write_train_config,
    )

    cfg = _load(args)
    if not cfg.data_yaml.exists():
        raise SystemExit(f"{cfg.data_yaml} not found. Run 'sparrow-yolo prepare' first.")

    t = cfg.training
    resuming = bool(t["resume"])

    if resuming:
        # Resuming continues an existing run. Allocating a fresh directory would
        # leave Ultralytics to guess which checkpoint you meant.
        if not args.run:
            raise SystemExit(
                "training.resume is true, so --run <existing run dir> is required.\n"
                "See 'sparrow-yolo status --all' for the available runs."
            )
        run_dir = Path(args.run)
        if not run_dir.is_dir():
            raise SystemExit(f"Cannot resume: {run_dir} does not exist")
        model_weights = str(resolve_resume(run_dir))
        print(f"Resuming from {model_weights}")
    else:
        run_dir = Path(args.run) if args.run else unique_run_dir(cfg.runs_root, cfg.project_name)
        model_weights = t["model"]

    run_dir.mkdir(parents=True, exist_ok=True)
    _print_header(f"{'Resuming' if resuming else 'Training'} {cfg.project_name} -> {run_dir}")
    config_path = write_train_config(
        run_dir / "train_config.yaml",
        model_weights=model_weights,
        data_yaml=str(cfg.data_yaml),
        project=str(cfg.runs_root),
        name=run_dir.name,
        imgsz=int(t["imgsz"]),
        epochs=int(t["epochs"]),
        batch=t["batch"],
        device=t["device"],
        workers=t["workers"],
        patience=int(t["patience"]),
        amp=bool(t["amp"]),
        seed=int(t["seed"]),
        deterministic=bool(t["deterministic"]),
        cache=t["cache"],
        resume=bool(t["resume"]),
        **dict(t.get("extra") or {}),
    )
    print(config_path.read_text(encoding="utf-8"))
    # Keep the full experiment definition beside the run so it is reproducible.
    cfg.save(run_dir / "experiment.yaml")

    if not args.skip_preflight:
        _print_header("Preflight")
        result = preflight(config_path, run_nccl=not args.skip_nccl)
        print(result["report"])
        if not result["ok"]:
            print("\nPreflight failed. Fix the FAIL items, or re-run with --skip-preflight to override.")
            return 1

    proc = launch_training_process(config_path, run_dir=run_dir)
    print(f"\nStarted pid {proc['pid']}")
    print(f"  log    : {proc['log_path']}")
    print(f"  status : {proc['status_path']}")
    print(f"\nWatch:  sparrow-yolo status --run {run_dir} --follow")
    print(f"Stop :  sparrow-yolo stop --run {run_dir}")

    if args.watch:
        status = stream_log(run_dir)
        return 0 if status.get("state") in {"completed", "stopped"} else 1
    return 0


def cmd_status(args) -> int:
    from .process_runner import best_weights, list_runs, read_status, stream_log, tail_log

    if args.all:
        runs = list_runs(args.runs_root or "/workspace/runs")
        if not runs:
            print("No runs found")
            return 0
        print(f"{'run':<30} {'state':<12} {'epoch':<12} {'updated'}")
        for r in runs:
            epoch = f"{r['epoch']}/{r['epochs']}" if r["epoch"] is not None else "-"
            print(f"{r['run']:<30} {r['state']:<12} {epoch:<12} {r['updated']}")
        return 0

    if not args.run:
        raise SystemExit("--run or --all is required")
    run_dir = Path(args.run)
    if args.follow:
        stream_log(run_dir)
        return 0

    status = read_status(run_dir)
    print(json.dumps(status, indent=2, default=str))

    if status.get("stopped_early"):
        print(
            f"\nStopped early: no improvement for {status.get('patience')} epochs. "
            f"Ran {status.get('epochs_completed')}/{status.get('epochs_requested')}, "
            f"best was epoch {status.get('best_epoch')}."
        )
    elif status.get("epochs_completed"):
        print(f"\nRan all {status.get('epochs_completed')} epochs "
              f"(best was epoch {status.get('best_epoch')}).")

    best = best_weights(run_dir)
    if best:
        print(f"\nbest weights: {best}")
    if args.tail:
        _print_header(f"Last {args.tail} log lines")
        print(tail_log(run_dir, args.tail))
    return 0


def cmd_stop(args) -> int:
    from .process_runner import stop_training

    result = stop_training(Path(args.run), timeout=float(args.timeout), force=bool(args.force))
    print(result["detail"])
    return 0 if result["stopped"] else 1


def _resolve_checkpoint(args) -> Path:
    from .process_runner import best_weights

    if args.weights:
        ckpt = Path(args.weights)
        if not ckpt.exists():
            raise SystemExit(f"Checkpoint not found: {ckpt}")
        return ckpt
    if not args.run:
        raise SystemExit("Pass --run (to take best.pt from a run) or --weights")
    ckpt = best_weights(Path(args.run))
    if not ckpt:
        raise SystemExit(
            f"No checkpoint found for {args.run}. Check 'sparrow-yolo status --run {args.run}', "
            f"or pass --weights explicitly."
        )
    return ckpt


def cmd_eval(args) -> int:
    from .train_eval import evaluate_checkpoint

    cfg = _load(args)
    ckpt = _resolve_checkpoint(args)
    _print_header(f"Evaluating {ckpt} on split '{args.split}'")
    metrics = evaluate_checkpoint(
        str(ckpt),
        str(cfg.data_yaml),
        split=args.split,
        imgsz=int(cfg.training["imgsz"]),
        device=cfg.training["device"],
    )
    print(metrics)
    return 0


def cmd_export(args) -> int:
    from .train_eval import export_onnx

    cfg = _load(args)
    ckpt = _resolve_checkpoint(args)
    e = cfg.export
    _print_header(f"Exporting {ckpt} to ONNX")
    path = export_onnx(
        str(ckpt),
        imgsz=int(cfg.training["imgsz"]),
        opset=e["opset"],
        simplify=bool(e["simplify"]),
        dynamic=bool(e["dynamic"]),
        half=bool(e["half"]),
    )
    print(f"Exported: {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sparrow-yolo", description="SPARROW YOLO training workbench")
    p.add_argument("--version", action="store_true", help="print version and exit")
    sub = p.add_subparsers(dest="command")

    def add_config(sp, required: bool = True):
        sp.add_argument("--config", "-c", required=required, help="experiment YAML")
        sp.add_argument("--set", action="append", metavar="SECTION.KEY=VALUE",
                        help="override a config value, e.g. --set training.batch=8")

    d = sub.add_parser("doctor", help="check GPUs, shared memory, NCCL, and torch build")
    add_config(d, required=False)
    d.add_argument("--skip-nccl", action="store_true", help="skip the live multi-GPU all_reduce test")
    d.set_defaults(func=cmd_doctor)

    a = sub.add_parser("audit", help="scan the VOC dataset and report issues")
    add_config(a)
    a.add_argument("--out", help="write the full issue list to this CSV")
    a.set_defaults(func=cmd_audit)

    pr = sub.add_parser("prepare", help="build the YOLO dataset and data.yaml")
    add_config(pr)
    pr.add_argument("--overwrite", action="store_true", help="replace an existing prepared dataset")
    pr.set_defaults(func=cmd_prepare)

    t = sub.add_parser("train", help="launch detached training")
    add_config(t)
    t.add_argument("--run", help="run directory (default: runs/<project>, auto-incremented)")
    t.add_argument("--watch", action="store_true", help="stream the log until the run finishes")
    t.add_argument("--skip-preflight", action="store_true", help="skip environment checks")
    t.add_argument("--skip-nccl", action="store_true", help="skip only the NCCL self-test")
    t.set_defaults(func=cmd_train)

    s = sub.add_parser("status", help="inspect a run")
    s.add_argument("--run", help="run directory")
    s.add_argument("--all", action="store_true", help="summarise every run")
    s.add_argument("--runs-root", help="runs root for --all (default /workspace/runs)")
    s.add_argument("--follow", action="store_true", help="stream the log")
    s.add_argument("--tail", type=int, default=0, help="print the last N log lines")
    s.set_defaults(func=cmd_status)

    st = sub.add_parser("stop", help="stop a run after its current epoch")
    st.add_argument("--run", required=True)
    st.add_argument("--timeout", default=120, help="seconds to wait for a clean stop")
    st.add_argument("--force", action="store_true", help="SIGKILL immediately (loses the current epoch)")
    st.set_defaults(func=cmd_stop)

    e = sub.add_parser("eval", help="evaluate a checkpoint")
    add_config(e)
    e.add_argument("--run", help="run directory to take best.pt from")
    e.add_argument("--weights", help="explicit checkpoint path")
    e.add_argument("--split", default="test", choices=["train", "val", "test"])
    e.set_defaults(func=cmd_eval)

    x = sub.add_parser("export", help="export a checkpoint to ONNX")
    add_config(x)
    x.add_argument("--run", help="run directory to take best.pt from")
    x.add_argument("--weights", help="explicit checkpoint path")
    x.set_defaults(func=cmd_export)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "version", False):
        from . import __version__

        print(f"sparrow-yolo {__version__}")
        return 0
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        return 130
    except (FileNotFoundError, FileExistsError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
