"""Model-independent data contracts; source IDs and positional indices are distinct."""

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class Instance:
    object_id: int
    bbox: torch.Tensor  # normalized cx, cy, width, height
    segment: torch.Tensor | None


@dataclass(frozen=True)
class PromptQuery:
    text: str
    object_ids: tuple[int, ...] = ()
    negative: bool = False


@dataclass(frozen=True)
class Sample:
    image: torch.Tensor
    objects: tuple[Instance, ...]
    queries: tuple[PromptQuery, ...]
    source_index: int
    source_id: int
    original_size: tuple[int, int]  # height, width
    image_path: Path
