"""Merge adapters and restore standard torch attention, without SAM3 imports."""

import torch
from torch import nn

from .attention import MultiheadAttentionLoRA
from .layers import LoRALinear


def _get_parent(model: nn.Module, module_name: str):
    """Return (parent_module, attribute_name) for a dotted module path."""
    *parent_path, attr_name = module_name.split(".")
    parent = model
    for p in parent_path:
        parent = getattr(parent, p)
    return parent, attr_name


def merge_lora_linears(model: nn.Module) -> int:
    """Fold every LoRALinear's low-rank delta into its frozen base Linear and
    put the plain nn.Linear back in place. Returns the number of merged layers."""
    to_merge = [
        (name, module) for name, module in model.named_modules() if isinstance(module, LoRALinear)
    ]

    with torch.no_grad():
        for name, module in to_merge:
            # forward is: original(x) + (x @ A @ B) * scaling, so the weight
            # delta in nn.Linear layout (out, in) is (A @ B)^T * scaling
            delta = (module.lora.lora_A @ module.lora.lora_B).T * module.lora.scaling
            base = module.original_layer
            base.weight += delta.to(dtype=base.weight.dtype, device=base.weight.device)

            parent, attr_name = _get_parent(model, name)
            setattr(parent, attr_name, base)

    return len(to_merge)


def restore_multihead_attention(model: nn.Module) -> int:
    """Convert MultiheadAttentionLoRA modules (separate q/k/v/out projections)
    back to stock nn.MultiheadAttention (fused in_proj). Must run AFTER
    merge_lora_linears so the projections are plain nn.Linear with merged
    weights. Returns the number of restored modules."""
    to_restore = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, MultiheadAttentionLoRA)
    ]

    with torch.no_grad():
        for name, module in to_restore:
            for proj in (module.q_proj, module.k_proj, module.v_proj, module.out_proj):
                if not isinstance(proj, nn.Linear):
                    raise RuntimeError(
                        f"{name} still contains un-merged LoRA projections; "
                        "merge_lora_linears must run first"
                    )

            has_bias = module.q_proj.bias is not None
            mha = nn.MultiheadAttention(
                embed_dim=module.embed_dim,
                num_heads=module.num_heads,
                dropout=module.dropout,
                bias=has_bias,
                batch_first=module.batch_first,
            ).to(device=module.q_proj.weight.device, dtype=module.q_proj.weight.dtype)
            mha.train(module.training)
            mha.requires_grad_(False)
            mha.in_proj_weight.copy_(
                torch.cat(
                    [module.q_proj.weight, module.k_proj.weight, module.v_proj.weight],
                    dim=0,
                )
            )
            mha.out_proj.weight.copy_(module.out_proj.weight)
            if has_bias:
                mha.in_proj_bias.copy_(
                    torch.cat(
                        [module.q_proj.bias, module.k_proj.bias, module.v_proj.bias],
                        dim=0,
                    )
                )
                mha.out_proj.bias.copy_(module.out_proj.bias)

            parent, attr_name = _get_parent(model, name)
            setattr(parent, attr_name, mha)

    return len(to_restore)
