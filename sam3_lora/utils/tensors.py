"""Device transfer keeps dataclasses intact and is shared by every workflow."""

import dataclasses

import torch


def move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.replace(
            value,
            **{
                f.name: move_to_device(getattr(value, f.name), device)
                for f in dataclasses.fields(value)
                if f.init
            },
        )
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def resolve_device(requested="auto"):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; choose --device cpu explicitly")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return device
