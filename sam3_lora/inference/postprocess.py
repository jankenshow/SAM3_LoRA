"""Shared presence scoring, mask NMS, and geometry for inference/evaluation."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from sam3_lora.data.transforms import cxcywh_to_xyxy


@dataclass(frozen=True)
class Prediction:
    boxes: torch.Tensor  # normalized cxcywh, never silently treated as xyxy
    scores: torch.Tensor
    masks: torch.Tensor  # boolean, model-output resolution


def joint_scores(logits, presence=None):
    scores = logits.float().sigmoid()
    if scores.ndim > 1:
        scores = scores.amax(dim=-1)
    if presence is not None:
        scores = scores * presence.float().sigmoid().reshape(-1)[0]
    return scores


def mask_nms(masks, scores, threshold):
    if not len(scores):
        return torch.empty(0, dtype=torch.long, device=scores.device)
    flat = masks.flatten(1).float()
    intersection = flat @ flat.T
    area = flat.sum(1)
    iou = intersection / (area[:, None] + area[None, :] - intersection).clamp_min(1)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel():
        index = order[0]
        keep.append(index)
        order = order[1:][iou[index, order[1:]] <= threshold]
    return torch.stack(keep)


def postprocess(
    logits,
    boxes,
    masks,
    *,
    presence=None,
    threshold=0.3,
    nms_iou=0.7,
    max_detections=100,
    merge=False,
    merge_iou=0.15,
    merge_dilate=0,
):
    if not 0 <= threshold <= 1 or not 0 <= nms_iou <= 1:
        raise ValueError("Score and NMS thresholds must be in [0, 1]")
    scores = joint_scores(logits, presence)
    keep = scores > threshold
    scores, boxes, masks = scores[keep], boxes[keep], masks[keep] > 0
    if merge and len(scores):
        from sam3_lora.evaluation.masks import merge_overlapping_masks

        masks, scores, boxes = merge_overlapping_masks(
            masks.cpu(), scores.cpu(), boxes.cpu(), iou_threshold=merge_iou, dilate=merge_dilate
        )
    elif len(scores):
        selected = mask_nms(masks, scores, nms_iou)
        masks, scores, boxes = masks[selected], scores[selected], boxes[selected]
    selected = scores.argsort(descending=True)
    if max_detections > 0:
        selected = selected[:max_detections]
    return Prediction(
        boxes[selected].detach().cpu(),
        scores[selected].detach().cpu(),
        masks[selected].detach().cpu(),
    )


def pixel_output(prediction, original_size):
    height, width = original_size
    boxes = cxcywh_to_xyxy(prediction.boxes) * torch.tensor([width, height, width, height])
    boxes[:, 0::2].clamp_(0, width)
    boxes[:, 1::2].clamp_(0, height)
    masks = (
        F.interpolate(prediction.masks[:, None].float(), size=(height, width), mode="nearest")[
            :, 0
        ].bool()
        if len(boxes)
        else torch.empty((0, height, width), dtype=torch.bool)
    )
    return boxes, masks
