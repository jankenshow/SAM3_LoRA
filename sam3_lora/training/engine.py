"""A model/loss-agnostic epoch engine, testable with tiny CPU networks."""

from contextlib import nullcontext

import torch

from .distributed import aggregate_mean


class EpochEngine:
    def __init__(
        self,
        model,
        optimizer,
        loss_step,
        device,
        *,
        accumulation=1,
        precision="no",
        max_grad_norm=1.0,
        scheduler=None,
    ):
        if accumulation < 1:
            raise ValueError("accumulation must be >= 1")
        self.model, self.optimizer, self.loss_step, self.device = (
            model,
            optimizer,
            loss_step,
            torch.device(device),
        )
        self.accumulation, self.precision, self.max_grad_norm = (
            accumulation,
            precision,
            max_grad_norm,
        )
        self.scheduler, self.optimizer_steps = scheduler, 0
        if precision == "fp16" and self.device.type != "cuda":
            raise ValueError("fp16 training is supported only on CUDA")
        self.scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")

    def autocast(self):
        if self.precision in ("no", "fp32"):
            return nullcontext()
        return torch.autocast(
            self.device.type, dtype=torch.float16 if self.precision == "fp16" else torch.bfloat16
        )

    def run(self, loader, *, training):
        if training and not len(loader):
            raise ValueError("Cannot run an empty training DataLoader")
        self.model.train(training)
        total, count = 0.0, 0
        if training:
            self.optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            for index, batch in enumerate(loader):
                group_start = (index // self.accumulation) * self.accumulation
                group_size = min(self.accumulation, len(loader) - group_start)
                boundary = (index + 1) % self.accumulation == 0 or index + 1 == len(loader)
                sync = (
                    self.model.no_sync()
                    if training and not boundary and hasattr(self.model, "no_sync")
                    else nullcontext()
                )
                with sync:
                    with self.autocast():
                        loss = self.loss_step(self.model, batch)
                        if loss.ndim or not torch.isfinite(loss):
                            raise FloatingPointError(
                                f"Invalid loss at batch {index}: {loss.detach()}"
                            )
                    if training:
                        self.scaler.scale(loss / group_size).backward()
                total += float(loss.detach())
                count += 1
                if training and boundary:
                    self.scaler.unscale_(self.optimizer)
                    if self.max_grad_norm:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm, error_if_nonfinite=True
                        )
                    previous_scale = self.scaler.get_scale()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    if self.scaler.get_scale() >= previous_scale:
                        self.optimizer_steps += 1
                        if self.scheduler:
                            self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
        return aggregate_mean(total, count, self.device)
