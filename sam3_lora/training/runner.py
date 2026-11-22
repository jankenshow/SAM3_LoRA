"""High-level training workflow; model construction is injectable for integration tests."""

import json
import logging
import math
import os
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from sam3_lora.adapters import save_lora_weights
from sam3_lora.config import validate_training_config
from sam3_lora.data.bridge import collate_samples
from sam3_lora.data.dataset import build_dataset
from sam3_lora.modeling import build_adapted_model
from sam3_lora.utils.tensors import resolve_device

from .checkpoint import restore_training_state, save_training_state
from .distributed import distributed_runtime, is_main_process
from .engine import EpochEngine
from .losses import SAM3LossStep

LOG = logging.getLogger(__name__)


class EvaluationSampler(Sampler):
    """Partition validation without DistributedSampler's duplicate padding."""

    def __init__(self, dataset, rank, world_size):
        self.indices = range(rank, len(dataset), world_size)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def make_scheduler(optimizer, config, steps):
    policy = config.get("lr_scheduler", "cosine")
    warmup = config.get("warmup_steps", 0)
    if policy not in ("cosine", "linear", "constant"):
        raise ValueError(f"Unknown lr_scheduler: {policy}")

    def factor(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = min(1.0, (step - warmup) / max(1, steps - warmup))
        return (
            0.5 * (1 + math.cos(math.pi * progress))
            if policy == "cosine"
            else (1 - progress if policy == "linear" else 1.0)
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class TrainingApplication:
    def __init__(
        self,
        config,
        *,
        model_factory=build_adapted_model,
        loss_factory=SAM3LossStep,
        dataset_factory=build_dataset,
        collate=collate_samples,
    ):
        self.config = validate_training_config(config)
        self.model_factory, self.loss_factory = model_factory, loss_factory
        self.dataset_factory, self.collate = dataset_factory, collate

    def run(self, device="auto", resume=None):
        with distributed_runtime() as distributed:
            return self._run(device, resume, distributed)

    def _run(self, requested, resume, distributed):
        config, t = self.config, self.config["training"]
        output = Path(config["output"]["output_dir"])
        if not resume and any(
            (output / name).exists() for name in ("training_state.pt", "history.jsonl")
        ):
            raise FileExistsError(
                f"Training artifacts already exist in {output}; use --resume or a new output_dir"
            )
        seed = t.get("seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        device = (
            torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
            if distributed
            else resolve_device(requested)
        )
        if t.get("mixed_precision", "no") == "fp16" and device.type != "cuda":
            raise ValueError("fp16 training is supported only on CUDA")
        train = self.dataset_factory(config, t["data_dir"], "train")
        valid = self.dataset_factory(
            config, t["data_dir"], "valid"
        )  # Invalid/missing validation must not be hidden.
        if not len(train) or not len(valid):
            raise ValueError("train and valid must both contain samples")
        train_sampler = DistributedSampler(train) if distributed else None
        valid_sampler = (
            EvaluationSampler(valid, dist.get_rank(), dist.get_world_size())
            if distributed
            else None
        )
        options = dict(
            batch_size=t.get("batch_size", 1),
            num_workers=t.get("num_workers", 0),
            collate_fn=self.collate,
            pin_memory=device.type == "cuda",
        )
        train_loader = DataLoader(
            train, sampler=train_sampler, shuffle=train_sampler is None, **options
        )
        valid_loader = DataLoader(valid, sampler=valid_sampler, shuffle=False, **options)
        model = self.model_factory(config, "cpu").to(device)
        if distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[device.index],
                find_unused_parameters=config.get("hardware", {}).get(
                    "find_unused_parameters", True
                ),
            )
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(t.get("learning_rate", 1e-4)),
            weight_decay=float(t.get("weight_decay", 0.01)),
            betas=(float(t.get("adam_beta1", 0.9)), float(t.get("adam_beta2", 0.999))),
            eps=float(t.get("adam_epsilon", 1e-8)),
        )
        accumulation = t.get("gradient_accumulation_steps", 1)
        scheduler = make_scheduler(
            optimizer, t, math.ceil(len(train_loader) / accumulation) * t.get("num_epochs", 1)
        )
        loss = self.loss_factory(device)
        engine = EpochEngine(
            model,
            optimizer,
            loss,
            device,
            accumulation=accumulation,
            precision=t.get("mixed_precision", "no"),
            max_grad_norm=float(t.get("max_grad_norm", 1.0)),
            scheduler=scheduler,
        )
        # Unwrapped validation avoids DDP buffer broadcasts with unequal validation partitions.
        evaluation = EpochEngine(
            model.module if distributed else model,
            optimizer,
            loss,
            device,
            precision=t.get("mixed_precision", "no"),
        )
        start, best, bad_epochs = 0, float("inf"), 0
        if resume:
            state = restore_training_state(resume, model, engine)
            start, best, bad_epochs = state["epoch"], state["best_loss"], state["bad_epochs"]
        if is_main_process():
            output.mkdir(parents=True, exist_ok=True)
            (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
            original = model.module if distributed else model
            if hasattr(original, "sam3_lora_report"):
                (output / "injection.json").write_text(
                    json.dumps(asdict(original.sam3_lora_report), indent=2)
                )
        for epoch in range(start, t.get("num_epochs", 1)):
            if train_sampler:
                train_sampler.set_epoch(epoch)
            train_loss = engine.run(train_loader, training=True)
            val_loss = evaluation.run(valid_loader, training=False)
            improved = val_loss < best - float(t.get("early_stopping_min_delta", 0.0))
            if improved:
                best = val_loss
            bad_epochs = 0 if improved else bad_epochs + 1
            if is_main_process():
                original = model.module if distributed else model
                save_lora_weights(original, output / "last_lora_weights.pt")
                if improved:
                    save_lora_weights(original, output / "best_lora_weights.pt")
                save_training_state(
                    output / "training_state.pt", model, engine, epoch + 1, best, bad_epochs
                )
                with (output / "history.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "epoch": epoch + 1,
                                "train_loss": train_loss,
                                "val_loss": val_loss,
                                "optimizer_steps": engine.optimizer_steps,
                            }
                        )
                        + "\n"
                    )
                LOG.info("Epoch %d: train=%.6f valid=%.6f", epoch + 1, train_loss, val_loss)
            patience = t.get("early_stopping_patience")
            if patience is not None and bad_epochs >= patience:
                break
        return {
            "best_val_loss": best,
            "optimizer_steps": engine.optimizer_steps,
            "output_dir": str(output),
        }
