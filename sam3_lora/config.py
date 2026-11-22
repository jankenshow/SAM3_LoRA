"""Configuration loading and validation without importing torch or SAM3."""

import math
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    with Path(path).open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    if "lora" not in config:
        raise ValueError("Missing lora section")
    return config


def validate_training_config(config):
    from .adapters.config import LoRAConfig

    LoRAConfig.from_mapping(config.get("lora", {}))
    t = config.get("training", {})
    if not t.get("data_dir"):
        raise ValueError("training.data_dir is required")
    if not config.get("output", {}).get("output_dir"):
        raise ValueError("output.output_dir is required")
    for key, default in [("batch_size", 1), ("num_epochs", 1), ("gradient_accumulation_steps", 1)]:
        value = t.get(key, default)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"training.{key} must be a positive integer")
    for key, default in [
        ("num_workers", 0),
        ("warmup_steps", 0),
        ("num_negatives", 2),
        ("num_cross_negatives", 2),
    ]:
        if not isinstance(t.get(key, default), int) or t.get(key, default) < 0:
            raise ValueError(f"training.{key} must be a non-negative integer")
    lr = float(t.get("learning_rate", 1e-4))
    if lr <= 0 or not math.isfinite(lr):
        raise ValueError("training.learning_rate must be positive and finite")
    if t.get("mixed_precision", "no") not in ("no", "fp32", "fp16", "bf16"):
        raise ValueError("mixed_precision must be no/fp32/fp16/bf16")
    if t.get("data_format", "coco") not in ("coco", "odvg"):
        raise ValueError("data_format must be coco or odvg")
    if t.get("lr_scheduler", "cosine") not in ("cosine", "linear", "constant"):
        raise ValueError("lr_scheduler must be cosine/linear/constant")
    for key, default in [
        ("adam_epsilon", 1e-8),
        ("weight_decay", 0.01),
        ("max_grad_norm", 1.0),
        ("early_stopping_min_delta", 0.0),
    ]:
        value = float(t.get(key, default))
        if not math.isfinite(value) or value < 0 or (key == "adam_epsilon" and value == 0):
            raise ValueError(f"training.{key} is out of range")
    for key, default in [("adam_beta1", 0.9), ("adam_beta2", 0.999)]:
        if not 0 <= float(t.get(key, default)) < 1:
            raise ValueError(f"training.{key} must be in [0, 1)")
    patience = t.get("early_stopping_patience")
    if patience is not None and (not isinstance(patience, int) or patience < 1):
        raise ValueError("early_stopping_patience must be a positive integer")
    return config
