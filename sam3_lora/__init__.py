"""SAM3 LoRA workflows. Importing the package never builds a model."""

__version__ = "0.4.0"


def __getattr__(name):
    if name in {"LoRAConfig", "LoRALayer", "LoRALinear", "apply_lora_to_model"}:
        from importlib import import_module

        return getattr(import_module(".adapters", __name__), name)
    raise AttributeError(name)
