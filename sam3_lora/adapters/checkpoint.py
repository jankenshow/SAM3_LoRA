"""Adapter-only checkpoints with preflight validation before changing a model."""

import os
import tempfile
from pathlib import Path

import torch

from .layers import LoRALinear


def adapter_state_dict(model):
    return {
        f"{name}.lora.{key}": value.detach().cpu().clone()
        for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
        for key, value in module.lora.state_dict().items()
    }


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_lora_weights(model, path):
    atomic_torch_save(adapter_state_dict(model), path)


def load_adapter_state(model, state):
    expected = adapter_state_dict(model)
    if set(state) != set(expected):
        raise ValueError(
            f"Adapter keys differ: missing={sorted(set(expected) - set(state))}, unexpected={sorted(set(state) - set(expected))}"
        )
    for key, value in state.items():
        if not isinstance(value, torch.Tensor) or value.shape != expected[key].shape:
            raise ValueError(f"Adapter shape mismatch at {key}")
    model.load_state_dict(state, strict=False)


def load_lora_weights(model, path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    load_adapter_state(model, state)
