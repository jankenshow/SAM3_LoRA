"""Argument schemas only; importing this module performs no work."""

import argparse


def boolean(value):
    if value.lower() not in ("true", "false", "yes", "no", "1", "0"):
        raise argparse.ArgumentTypeError("Expected true/false")
    return value.lower() in ("true", "yes", "1")


def build_parser():
    parser = argparse.ArgumentParser(prog="sam3-lora")
    parser.add_argument("--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train adapters; --dry-run inspects config only")
    train.add_argument("--config", required=True)
    train.add_argument("--device", nargs="+", default=["auto"])
    train.add_argument("--resume")
    train.add_argument("--dry-run", action="store_true")
    train.add_argument("--master-port", "--master_port", type=int, default=29500)
    train.add_argument("--local_rank", type=int, default=-1, help=argparse.SUPPRESS)
    train.add_argument("--_launched_by_torchrun", action="store_true", help=argparse.SUPPRESS)
    for name in ["infer", "evaluate", "merge", "compare"]:
        sub = commands.add_parser(name)
        sub.add_argument("--config", required=name != "evaluate")
        sub.add_argument("--weights")
        sub.add_argument("--device", default="auto")
        if name == "merge":
            sub.add_argument("--output", required=True)
            continue
        sub.add_argument("--threshold", "--prob-threshold", type=float, default=0.3)
        sub.add_argument("--nms-iou", type=float, default=0.7)
        if name == "evaluate":
            sub.add_argument("--data-dir", "--val_data_dir", required=True, dest="data_dir")
            sub.add_argument("--output", default="outputs/evaluation")
            sub.add_argument("--use-base-model", action="store_true")
            sub.add_argument("--num-samples", type=int)
            sub.add_argument("--include-negatives", action="store_true")
            sub.add_argument("--pixel-metrics", type=boolean, default=True)
            sub.add_argument("--no-presence", action="store_true")
            sub.add_argument("--merge", action="store_true")
            sub.add_argument("--merge-iou", type=float, default=0.15)
            sub.add_argument("--merge-dilate", type=int, default=0)
        else:
            sub.add_argument("--image", required=True)
            sub.add_argument("--prompt", nargs="+", default=["object"])
            sub.add_argument(
                "--output",
                default="outputs/comparison" if name == "compare" else "outputs/inference.png",
            )
            sub.add_argument("--resolution", type=int, default=1008)
            sub.add_argument("--boundingbox", type=boolean, default=False)
            sub.add_argument("--score", type=boolean, default=True)
            sub.add_argument("--no-masks", action="store_true")
    prepare = commands.add_parser("prepare", help="Inspect, split or convert datasets")
    tools = prepare.add_subparsers(dest="tool", required=True)
    inspect = tools.add_parser("inspect")
    inspect.add_argument("--split-dir", required=True)
    inspect.add_argument("--check-images", action="store_true")
    split = tools.add_parser("split")
    split.add_argument("--split-dir", required=True)
    split.add_argument("--output", required=True)
    split.add_argument("--valid-fraction", type=float, default=0.1)
    split.add_argument("--test-fraction", type=float, default=0.1)
    split.add_argument("--seed", type=int, default=42)
    odvg = tools.add_parser("coco-to-odvg")
    group = odvg.add_mutually_exclusive_group(required=True)
    group.add_argument("--data-dir")
    group.add_argument("--split-dir")
    odvg.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    odvg.add_argument("--no-masks", action="store_true")
    odvg.add_argument("--skip-empty", action="store_true")
    odvg.add_argument("--output-name", default="_annotations.odvg.jsonl")
    roboflow = tools.add_parser("roboflow")
    roboflow.add_argument("--split-dir", required=True)
    ref = tools.add_parser("refcoco")
    ref.add_argument("--refcoco-dir", required=True)
    ref.add_argument("--images-dir", required=True)
    ref.add_argument("--output-dir", required=True)
    ref.add_argument("--mode", choices=["ref", "category"], default="ref")
    ref.add_argument("--sentences", choices=["first", "all"], default="first")
    ref.add_argument("--refs-file")
    ref.add_argument("--symlink", action="store_true")
    return parser
