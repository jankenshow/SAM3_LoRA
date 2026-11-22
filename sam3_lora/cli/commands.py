"""Command orchestration; heavy runtime dependencies are loaded on demand."""

import os
import subprocess
import sys
from pathlib import Path


def prepare_command(args):
    from sam3_lora.data.preparation import inspect_split, split_coco

    if args.tool == "inspect":
        return inspect_split(args.split_dir, args.check_images)
    if args.tool == "split":
        return split_coco(
            args.split_dir, args.output, args.valid_fraction, args.test_fraction, args.seed
        )
    if args.tool == "coco-to-odvg":
        from sam3_lora.data.conversion.coco_to_odvg import convert_split

        directories = (
            [Path(args.split_dir)]
            if args.split_dir
            else [Path(args.data_dir) / s for s in args.splits]
        )
        count = 0
        for directory in directories:
            if not (directory / "_annotations.coco.json").exists():
                continue
            if (directory / args.output_name).exists():
                raise FileExistsError(directory / args.output_name)
            convert_split(
                directory,
                keep_masks=not args.no_masks,
                output_name=args.output_name,
                skip_empty=args.skip_empty,
            )
            count += 1
        if not count:
            raise ValueError("No COCO annotation files found")
        return {"converted": count}
    if args.tool == "roboflow":
        from sam3_lora.data.conversion.roboflow import convert_roboflow_to_coco

        return str(convert_roboflow_to_coco(args.split_dir))
    from sam3_lora.data.conversion.refcoco import convert

    if Path(args.output_dir).exists():
        raise FileExistsError(args.output_dir)
    convert(
        args.refcoco_dir,
        args.images_dir,
        args.output_dir,
        mode=args.mode,
        sentences=args.sentences,
        refs_file=args.refs_file,
        symlink=args.symlink,
    )
    return {"output": args.output_dir}


def train_command(args, config):
    from sam3_lora.config import validate_training_config

    validate_training_config(config)
    if args.dry_run:
        return {"status": "dry-run", "model_loaded": False, "config": config}
    if all(s.isdigit() for s in args.device):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.device)
        if len(args.device) > 1 and "LOCAL_RANK" not in os.environ:
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nproc_per_node={len(args.device)}",
                f"--master_port={args.master_port}",
                "-m",
                "sam3_lora",
                "train",
                "--config",
                args.config,
                "--device",
                "cuda",
            ]
            if args.resume:
                command += ["--resume", args.resume]
            subprocess.run(command, check=True)
            return {"status": "distributed-completed"}
        device = "cuda"
    elif len(args.device) == 1:
        device = args.device[0]
    else:
        raise ValueError("--device accepts one backend or a list of CUDA indices")
    from sam3_lora.training.runner import TrainingApplication

    return TrainingApplication(config).run(device, args.resume)


def merge_command(args, config):
    from sam3_lora.adapters import merge_lora_linears, restore_multihead_attention
    from sam3_lora.adapters.checkpoint import atomic_torch_save
    from sam3_lora.modeling import build_adapted_model

    if Path(args.output).exists():
        raise FileExistsError(args.output)
    weights = args.weights or str(Path(config["output"]["output_dir"]) / "best_lora_weights.pt")
    if not Path(weights).is_file():
        raise FileNotFoundError(weights)
    model = build_adapted_model(config, "cpu", inference=True, weights=weights)
    linears = merge_lora_linears(model)
    attentions = restore_multihead_attention(model)
    atomic_torch_save(model.state_dict(), args.output)
    return {"linear_layers": linears, "attention_modules": attentions, "output": args.output}


def dispatch(args):
    if args.command == "prepare":
        return prepare_command(args)
    from sam3_lora.config import load_config

    config = load_config(args.config) if args.config else {"model": {}, "training": {}, "lora": {}}
    if args.command == "train":
        return train_command(args, config)
    if args.command == "merge":
        return merge_command(args, config)
    policy = {"threshold": args.threshold, "nms_iou": args.nms_iou}
    if args.command == "evaluate":
        if not args.use_base_model and (not args.config or not args.weights):
            raise ValueError("Evaluation needs --config/--weights or --use-base-model")
        if args.merge_dilate and not args.merge:
            raise ValueError("--merge-dilate requires --merge")
        from sam3_lora.evaluation.runner import evaluate

        return evaluate(
            config,
            args.data_dir,
            args.output,
            weights=args.weights,
            base=args.use_base_model,
            device=args.device,
            num_samples=args.num_samples,
            include_negatives=args.include_negatives,
            pixel_metrics=args.pixel_metrics,
            use_presence=not args.no_presence,
            merge=args.merge,
            merge_iou=args.merge_iou,
            merge_dilate=args.merge_dilate,
            **policy,
        )
    if args.command == "compare":
        from sam3_lora.inference.comparison import compare

        return str(
            compare(
                config,
                args.image,
                args.prompt,
                args.output,
                weights=args.weights,
                device=args.device,
                resolution=args.resolution,
                show_boxes=args.boundingbox,
                show_masks=not args.no_masks,
                show_scores=args.score,
                **policy,
            )
        )
    from sam3_lora.inference.session import InferenceSession
    from sam3_lora.inference.visualization import save_results

    session = InferenceSession(config, args.weights, device=args.device, resolution=args.resolution)
    results = session.predict(args.image, args.prompt, **policy)
    save_results(
        results,
        args.output,
        show_boxes=args.boundingbox,
        show_masks=not args.no_masks,
        show_scores=args.score,
    )
    return {
        "output": args.output,
        "detections": {
            r["prompt"]: r["num_detections"] for k, r in results.items() if isinstance(k, int)
        },
    }
