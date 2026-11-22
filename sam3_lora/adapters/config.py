"""Validated adapter configuration, shared by training, inference and merge."""

import math
from dataclasses import asdict, dataclass


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: float = 16
    dropout: float = 0.0
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "qkv",
        "proj",
        "fc1",
        "fc2",
        "c_fc",
        "c_proj",
        "linear1",
        "linear2",
    )
    apply_to_vision_encoder: bool = True
    apply_to_text_encoder: bool = False
    apply_to_geometry_encoder: bool = False
    apply_to_detr_encoder: bool = True
    apply_to_detr_decoder: bool = True
    apply_to_mask_decoder: bool = False

    def __post_init__(self):
        if not isinstance(self.rank, int) or isinstance(self.rank, bool) or self.rank <= 0:
            raise ValueError("lora.rank must be a positive integer")
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("lora.alpha must be positive and finite")
        if not 0 <= self.dropout < 1:
            raise ValueError("lora.dropout must be in [0, 1)")
        self.target_modules = tuple(self.target_modules)
        if not self.target_modules or any(
            not isinstance(x, str) or not x for x in self.target_modules
        ):
            raise ValueError("lora.target_modules must contain non-empty names")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_mapping(cls, values, *, inference=False):
        values = dict(values)
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown LoRA configuration keys: {sorted(unknown)}")
        if inference:
            values["dropout"] = 0.0
        return cls(**values)
