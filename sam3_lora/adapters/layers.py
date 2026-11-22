"""Linear adapters. Tensor names preserve upstream checkpoint compatibility."""

import math
from typing import Optional

import torch
from torch import nn


class LoRALayer(nn.Module):
    """
    LoRA layer that replaces a linear layer with low-rank adaptation.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: Rank of the low-rank matrices (r in the paper)
        alpha: Scaling factor (typically set to rank)
        dropout: Dropout probability for LoRA weights
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        if rank <= 0 or not 0 <= dropout < 1:
            raise ValueError("rank must be positive and dropout must be in [0, 1)")
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # Dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply LoRA transformation: x @ (A @ B) * scaling
        """
        # x shape: (..., in_features)
        lora_out = self.dropout(x) @ self.lora_A @ self.lora_B
        return lora_out * self.scaling


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.
    Combines the original frozen linear layer with a LoRA layer.

    Exposes weight/bias properties to maintain compatibility with modules
    that access these attributes directly (e.g., nn.MultiheadAttention).
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Freeze the original layer
        self.original_layer = original_layer
        for param in self.original_layer.parameters():
            param.requires_grad = False

        # Store original layer attributes for compatibility
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features

        # Create LoRA layer
        self.lora = LoRALayer(
            in_features=original_layer.in_features,
            out_features=original_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        ).to(device=original_layer.weight.device, dtype=original_layer.weight.dtype)

    @property
    def weight(self) -> torch.Tensor:
        """Proxy to original layer's weight for compatibility with nn.MultiheadAttention."""
        return self.original_layer.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        """Proxy to original layer's bias for compatibility with nn.MultiheadAttention."""
        return self.original_layer.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: original output + LoRA output
        """
        return self.original_layer(x) + self.lora(x)
