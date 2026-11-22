"""Explicit attention projections that execute the LoRA forward path."""

import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class MultiheadAttentionLoRA(nn.Module):
    """
    Custom MultiheadAttention that doesn't use F.multi_head_attention_forward,
    allowing LoRA to be properly applied to Q, K, V, and output projections.

    This replaces nn.MultiheadAttention to enable LoRA on all projection layers.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = False,
        # Copy weights from existing MHA
        in_proj_weight: Optional[torch.Tensor] = None,
        in_proj_bias: Optional[torch.Tensor] = None,
        out_proj_weight: Optional[torch.Tensor] = None,
        out_proj_bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first
        self.dropout = dropout

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Separate Q, K, V projections (instead of fused in_proj)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Initialize from existing MHA weights if provided
        if in_proj_weight is not None:
            # Split in_proj_weight into q, k, v
            self.q_proj.weight.data = in_proj_weight[:embed_dim, :].clone()
            self.k_proj.weight.data = in_proj_weight[embed_dim : 2 * embed_dim, :].clone()
            self.v_proj.weight.data = in_proj_weight[2 * embed_dim :, :].clone()

        if in_proj_bias is not None:
            self.q_proj.bias.data = in_proj_bias[:embed_dim].clone()
            self.k_proj.bias.data = in_proj_bias[embed_dim : 2 * embed_dim].clone()
            self.v_proj.bias.data = in_proj_bias[2 * embed_dim :].clone()

        if out_proj_weight is not None:
            self.out_proj.weight.data = out_proj_weight.clone()

        if out_proj_bias is not None:
            self.out_proj.bias.data = out_proj_bias.clone()

        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass using separate Q, K, V projections so LoRA works.
        """
        # Handle batch_first
        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            raise ValueError("LoRA attention requires batched 3-D query, key and value tensors")
        if self.batch_first:
            # Input: (batch, seq, embed_dim)
            batch_size, tgt_len, _ = query.shape
            src_len = key.shape[1]
        else:
            # Input: (seq, batch, embed_dim)
            tgt_len, batch_size, _ = query.shape
            src_len = key.shape[0]
            # Convert to batch_first for easier processing
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        # Project Q, K, V - LoRA is applied here through the Linear layers
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape for multi-head attention
        # (batch, seq, embed_dim) -> (batch, num_heads, seq, head_dim)
        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        if is_causal:
            causal = torch.ones(tgt_len, src_len, dtype=torch.bool, device=q.device).triu(1)
            attn_weights = attn_weights.masked_fill(causal, float("-inf"))

        # Apply attention mask - handle various input formats
        if attn_mask is not None:
            # attn_weights shape: (batch, num_heads, tgt_len, src_len)
            if attn_mask.dim() == 2:
                # (tgt_len, src_len) -> (1, 1, tgt_len, src_len)
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                # Could be (batch, tgt_len, src_len) or (batch*num_heads, tgt_len, src_len)
                if attn_mask.shape[0] == batch_size:
                    # (batch, tgt_len, src_len) -> (batch, 1, tgt_len, src_len)
                    attn_mask = attn_mask.unsqueeze(1)
                elif attn_mask.shape[0] == batch_size * self.num_heads:
                    # (batch*num_heads, tgt_len, src_len) -> (batch, num_heads, tgt_len, src_len)
                    attn_mask = attn_mask.view(batch_size, self.num_heads, tgt_len, src_len)
                else:
                    # Unknown format, try to broadcast
                    attn_mask = attn_mask.unsqueeze(1)
            elif attn_mask.dim() == 4:
                # Already (batch, num_heads, tgt_len, src_len) or similar
                pass

            # Expand to match attn_weights if needed
            if attn_mask.shape != attn_weights.shape:
                attn_mask = attn_mask.expand_as(attn_weights)

            if attn_mask.dtype == torch.bool:
                attn_weights = attn_weights.masked_fill(attn_mask, float("-inf"))
            else:
                attn_weights = attn_weights + attn_mask

        # Apply key padding mask
        if key_padding_mask is not None:
            padding = key_padding_mask[:, None, None, :]
            if padding.dtype == torch.bool:
                attn_weights = attn_weights.masked_fill(padding, float("-inf"))
            else:
                attn_weights = attn_weights + padding

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back: (batch, num_heads, seq, head_dim) -> (batch, seq, embed_dim)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.embed_dim)
        )

        # Output projection - LoRA is applied here
        attn_output = self.out_proj(attn_output)

        # Convert back if not batch_first
        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        if need_weights:
            if average_attn_weights:
                attn_weights = attn_weights.mean(dim=1)
            return attn_output, attn_weights
        else:
            return attn_output, None
