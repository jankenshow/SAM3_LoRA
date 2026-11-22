"""One resize/normalization and coordinate convention for every workflow."""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def image_tensor(image, resolution=1008):
    image = image.convert("RGB").resize((resolution, resolution), Image.Resampling.BILINEAR)
    return torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float().div(127.5).sub(1)


def xywh_to_normalized(box, width, height):
    x, y, w, h = map(float, box)
    if width <= 0 or height <= 0 or w <= 0 or h <= 0 or not np.isfinite([x, y, w, h]).all():
        raise ValueError(f"Invalid image dimensions or box: {box}")
    return torch.tensor([(x + w / 2) / width, (y + h / 2) / height, w / width, h / height])


def cxcywh_to_xyxy(boxes):
    center, size = boxes[..., :2], boxes[..., 2:]
    return torch.cat((center - size / 2, center + size / 2), dim=-1)


def decode_segmentation(segmentation, height, width, resolution):
    if not segmentation:
        return None
    from pycocotools import mask as api

    if isinstance(segmentation, list):
        encoded = api.merge(api.frPyObjects(segmentation, height, width))
    elif isinstance(segmentation, dict):
        encoded = dict(segmentation)
        if isinstance(encoded["counts"], list):
            encoded = api.frPyObjects(encoded, height, width)
        elif isinstance(encoded["counts"], str):
            encoded["counts"] = encoded["counts"].encode()
    else:
        raise ValueError("Mask must be COCO polygons or RLE")
    mask = api.decode(encoded)
    if mask.ndim == 3:
        mask = mask.any(axis=-1)
    if mask.shape != (height, width):
        raise ValueError(f"Mask dimensions {mask.shape} do not match image {(height, width)}")
    return F.interpolate(
        torch.as_tensor(mask.copy()).float()[None, None],
        size=(resolution, resolution),
        mode="nearest",
    )[0, 0].bool()
