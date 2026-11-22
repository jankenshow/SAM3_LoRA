"""The single supported LoRA implementation, imported lazily."""

from importlib import import_module

_EXPORTS = {
    "LoRAConfig": "config",
    "LoRALayer": "layers",
    "LoRALinear": "layers",
    "MultiheadAttentionLoRA": "attention",
    "AdapterInjector": "injection",
    "apply_lora_to_model": "injection",
    "get_lora_parameters": "injection",
    "count_parameters": "injection",
    "save_lora_weights": "checkpoint",
    "load_lora_weights": "checkpoint",
    "adapter_state_dict": "checkpoint",
    "load_adapter_state": "checkpoint",
    "merge_lora_linears": "merge",
    "restore_multihead_attention": "merge",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    return getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
