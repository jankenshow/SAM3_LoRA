"""COCO ground truth is built per source-image/prompt pair, not per raw image ID."""

import numpy as np
import torch.nn.functional as F
from pycocotools import mask as mask_api


def encode_mask(mask):
    encoded = mask_api.encode(np.asfortranarray(np.asarray(mask, dtype=np.uint8)))
    encoded["counts"] = encoded["counts"].decode()
    return encoded


class PromptCOCOBuilder:
    def __init__(self, resolution=288):
        self.resolution = resolution
        self.ground_truth = {
            "info": {"description": "SAM3 per-prompt evaluation"},
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "object"}],
        }
        self.predictions = []

    def add(self, sample, query, prediction):
        unit = len(self.ground_truth["images"])
        size = self.resolution
        self.ground_truth["images"].append(
            {
                "id": unit,
                "height": size,
                "width": size,
                "source_id": sample.source_id,
                "source_index": sample.source_index,
                "prompt": query.text,
                "is_instance_exhaustive": True,
                "negative": query.negative,
            }
        )
        for obj in sample.objects:
            if obj.object_id not in query.object_ids:
                continue
            if obj.segment is None:
                raise ValueError(
                    "Segmentation evaluation requires GT masks; found a box-only annotation"
                )
            mask = (
                F.interpolate(obj.segment.float()[None, None], size=(size, size), mode="nearest")[
                    0, 0
                ]
                .bool()
                .numpy()
            )
            cx, cy, w, h = obj.bbox.tolist()
            self.ground_truth["annotations"].append(
                {
                    "id": len(self.ground_truth["annotations"]) + 1,
                    "image_id": unit,
                    "category_id": 1,
                    "bbox": [(cx - w / 2) * size, (cy - h / 2) * size, w * size, h * size],
                    "area": int(mask.sum()),
                    "segmentation": encode_mask(mask),
                    "iscrowd": 0,
                    "ignore": 0,
                }
            )
        for box, score, mask in zip(prediction.boxes, prediction.scores, prediction.masks):
            mask = (
                F.interpolate(mask.float()[None, None], size=(size, size), mode="nearest")[0, 0]
                .bool()
                .numpy()
            )
            cx, cy, w, h = box.tolist()
            self.predictions.append(
                {
                    "id": len(self.predictions) + 1,
                    "image_id": unit,
                    "category_id": 1,
                    "bbox": [(cx - w / 2) * size, (cy - h / 2) * size, w * size, h * size],
                    "score": float(score),
                    "segmentation": encode_mask(mask),
                }
            )
        return unit


def compute_coco_metrics(ground_truth, predictions):
    """Compute segmentation AP, including a valid zero-detection result set."""
    import contextlib
    import io

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = ground_truth
        gt.createIndex()
        if predictions:
            predicted = gt.loadRes(predictions)
        else:
            predicted = COCO()
            predicted.dataset = {**ground_truth, "annotations": []}
            predicted.createIndex()
        metric = COCOeval(gt, predicted, "segm")
        metric.params.useCats = False
        metric.evaluate()
        metric.accumulate()
        metric.summarize()
    return {key: float(value) for key, value in zip(("mAP", "mAP50", "mAP75"), metric.stats[:3])}
