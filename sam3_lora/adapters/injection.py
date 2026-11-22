"""Select components, expose MHA projections, then inject linear adapters."""

import logging
from dataclasses import dataclass

from torch import nn

from .attention import MultiheadAttentionLoRA
from .config import LoRAConfig
from .layers import LoRALinear

LOG = logging.getLogger(__name__)
COMPONENTS = {
    "apply_to_vision_encoder": ("vision_encoder", "vision_backbone"),
    "apply_to_text_encoder": ("text_encoder", "language_backbone"),
    "apply_to_geometry_encoder": ("geometry_encoder",),
    "apply_to_detr_encoder": ("detr_encoder", "transformer.encoder"),
    "apply_to_detr_decoder": ("detr_decoder", "transformer.decoder"),
    "apply_to_mask_decoder": ("segmentation_head", "mask_decoder"),
}


@dataclass(frozen=True)
class InjectionReport:
    attention_modules: tuple[str, ...]
    linear_modules: tuple[str, ...]


class AdapterInjector:
    def __init__(self, config: LoRAConfig):
        self.config = config

    def includes_component(self, name):
        return all(
            getattr(self.config, flag) or not any(part in name for part in names)
            for flag, names in COMPONENTS.items()
        )

    def includes_linear(self, name):
        leaf = name.rsplit(".", 1)[-1]
        return self.includes_component(name) and any(
            target == "all" or target in leaf for target in self.config.target_modules
        )

    @staticmethod
    def replace(model, name, module):
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, module)

    def apply(self, model):
        if any(isinstance(m, LoRALinear) for m in model.modules()):
            raise ValueError("Model already contains LoRA; do not inject twice")
        model.requires_grad_(False)
        attentions = []
        for name, module in list(model.named_modules()):
            if not isinstance(module, nn.MultiheadAttention) or not self.includes_component(name):
                continue
            if not any(
                self.includes_linear(f"{name}.{p}")
                for p in ("q_proj", "k_proj", "v_proj", "out_proj")
            ):
                continue
            if not module._qkv_same_embed_dim or module.bias_k is not None or module.add_zero_attn:
                raise ValueError(
                    f"Unsupported attention variant at {name}; separate kdim/vdim, bias_kv and zero_attn require an adapter implementation"
                )
            replacement = MultiheadAttentionLoRA(
                module.embed_dim,
                module.num_heads,
                dropout=module.dropout,
                bias=module.in_proj_bias is not None,
                batch_first=module.batch_first,
                in_proj_weight=module.in_proj_weight,
                in_proj_bias=module.in_proj_bias,
                out_proj_weight=module.out_proj.weight,
                out_proj_bias=module.out_proj.bias,
            ).to(device=module.out_proj.weight.device, dtype=module.out_proj.weight.dtype)
            replacement.requires_grad_(False).train(module.training)
            self.replace(model, name, replacement)
            attentions.append(name)
        linears = []
        for name, module in list(model.named_modules()):
            if name and isinstance(module, nn.Linear) and self.includes_linear(name):
                replacement = LoRALinear(
                    module, self.config.rank, self.config.alpha, self.config.dropout
                )
                replacement.train(module.training)
                self.replace(model, name, replacement)
                linears.append(name)
        if not linears:
            raise ValueError("No LoRA targets matched; inspect target_modules and component flags")
        model.sam3_lora_report = InjectionReport(tuple(attentions), tuple(linears))
        LOG.info(
            "Replaced %d attention modules; adapted %d linear layers", len(attentions), len(linears)
        )
        return model


def apply_lora_to_model(model, config):
    return AdapterInjector(config).apply(model)


def get_lora_parameters(model):
    return [
        p
        for module in model.modules()
        if isinstance(module, LoRALinear)
        for p in module.lora.parameters()
    ]


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percentage": 100 * trainable / total if total else 0.0,
    }
