"""Resumable training state is distinct from deployment adapter weights."""

import random

import numpy as np
import torch

from sam3_lora.adapters.checkpoint import adapter_state_dict, atomic_torch_save, load_adapter_state


def save_training_state(path, model, engine, epoch, best_loss, bad_epochs=0):
    original = model.module if hasattr(model, "module") else model
    numpy_rng = np.random.get_state()
    atomic_torch_save(
        {
            "schema": 1,
            "adapter": adapter_state_dict(original),
            "optimizer": engine.optimizer.state_dict(),
            "scaler": engine.scaler.state_dict(),
            "scheduler": engine.scheduler.state_dict() if engine.scheduler else None,
            "epoch": epoch,
            "best_loss": best_loss,
            "bad_epochs": bad_epochs,
            "optimizer_steps": engine.optimizer_steps,
            "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
            "numpy_rng": (
                numpy_rng[0],
                torch.tensor(numpy_rng[1].astype(np.int64)),
                numpy_rng[2],
                numpy_rng[3],
                numpy_rng[4],
            ),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
        },
        path,
    )


def restore_training_state(path, model, engine):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("schema") != 1:
        raise ValueError("Not a resumable training checkpoint (expected schema=1)")
    load_adapter_state(model.module if hasattr(model, "module") else model, state["adapter"])
    engine.optimizer.load_state_dict(state["optimizer"])
    engine.scaler.load_state_dict(state["scaler"])
    if engine.scheduler and state["scheduler"] is not None:
        engine.scheduler.load_state_dict(state["scheduler"])
    engine.optimizer_steps = state["optimizer_steps"]
    torch.set_rng_state(state["torch_rng"])
    random.setstate(state["python_rng"])
    if "numpy_rng" in state:
        name, values, position, has_gauss, cached = state["numpy_rng"]
        np.random.set_state((name, values.numpy().astype(np.uint32), position, has_gauss, cached))
    if state.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state
