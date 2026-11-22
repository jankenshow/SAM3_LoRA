import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from sam3_lora.adapters import LoRAConfig, apply_lora_to_model


def tiny_model(attention=False, dtype=torch.float32):
    torch.manual_seed(3)
    model = nn.Module()
    if attention:
        model.attn = nn.MultiheadAttention(8, 2, batch_first=True, dtype=dtype)
    else:
        model.proj = nn.Linear(2, 1, dtype=dtype)
        model.forward = lambda x: model.proj(x)
    return model


def adapted(attention=False, dtype=torch.float32):
    return apply_lora_to_model(
        tiny_model(attention, dtype), LoRAConfig(rank=2, target_modules=("all",))
    )


def make_coco(directory, count=3):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    images = []
    for i in range(count):
        Image.new("RGB", (8, 6), (0, 128, 255)).save(directory / f"{i}.png")
        images.append({"id": 10 + 32 * i, "file_name": f"{i}.png", "width": 8, "height": 6})
    data = {
        "images": images,
        "categories": [{"id": 3, "name": "Dog"}, {"id": 9, "name": "Cat"}],
        "annotations": [
            {
                "id": 100 + i,
                "image_id": images[min(i, count - 1)]["id"],
                "category_id": c,
                "bbox": [2, 1, 4, 3],
                "segmentation": [[2, 1, 6, 1, 6, 4, 2, 4]],
                "area": 12,
                "iscrowd": 0,
            }
            for i, c in enumerate([3, 9])
        ],
    }
    (directory / "_annotations.coco.json").write_text(json.dumps(data))
    return data
