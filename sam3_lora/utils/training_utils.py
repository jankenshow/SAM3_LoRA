"""Compatibility names for old prototypes."""

from sam3_lora.adapters import count_parameters

from .tensors import resolve_device as get_device


def print_trainable_parameters(model):
    print(count_parameters(model))


__all__ = ["get_device", "print_trainable_parameters"]
