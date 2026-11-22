"""Mask merging and COCO-format prediction conversion, independent of training."""

import numpy as np
import pycocotools.mask as mask_utils
import torch


def _dilate_masks(binary_masks, radius):
    """Dilate binary masks [N, H, W] by `radius` pixels (max-pool dilation)."""
    if radius <= 0:
        return binary_masks
    kernel = 2 * radius + 1
    dilated = torch.nn.functional.max_pool2d(
        binary_masks.float().unsqueeze(1), kernel_size=kernel, stride=1, padding=radius
    ).squeeze(1)
    return dilated > 0.5


def merge_overlapping_masks(binary_masks, scores, boxes, iou_threshold=0.15, dilate=0):
    """
    Merge overlapping masks that likely represent the same object (e.g., crack segments).

    This is more aggressive than NMS - it MERGES masks instead of suppressing them.
    Useful for cracks where model splits one crack into many segments.

    Args:
        binary_masks: Binary masks [N, H, W]
        scores: Confidence scores [N]
        boxes: Bounding boxes [N, 4]
        iou_threshold: IoU threshold for merging (default: 0.15, lower = more aggressive)
        dilate: Dilation radius in pixels. When > 0, masks are dilated by this
            many pixels and any two fragments whose dilated masks TOUCH are
            merged. This joins disjoint portions of one crack (which have
            mask IoU 0 and are never merged by the IoU test alone). The
            merged output keeps the ORIGINAL (undilated) masks.

    Returns:
        Tuple of (merged_masks, merged_scores, merged_boxes)
    """
    if len(binary_masks) == 0:
        return binary_masks, scores, boxes

    # Sort by score (highest first)
    sorted_indices = torch.argsort(scores, descending=True)
    binary_masks = binary_masks[sorted_indices]
    scores = scores[sorted_indices]
    boxes = boxes[sorted_indices]

    dilated_masks = _dilate_masks(binary_masks, dilate) if dilate > 0 else None

    merged_masks = []
    merged_scores = []
    merged_boxes = []
    used = torch.zeros(len(binary_masks), dtype=torch.bool)

    for i in range(len(binary_masks)):
        if used[i]:
            continue

        current_mask = binary_masks[i].clone()
        current_dilated = dilated_masks[i].clone() if dilated_masks is not None else None
        current_score = scores[i].item()
        current_box = boxes[i]
        used[i] = True

        # Find overlapping masks and merge them
        for j in range(i + 1, len(binary_masks)):
            if used[j]:
                continue

            if current_dilated is not None:
                # Proximity criterion: merge if the dilated masks touch at all.
                # (IoU of thin dilated fragments is tiny even when they belong
                # to the same crack, so a threshold test would never fire.)
                should_merge = (current_dilated & dilated_masks[j]).any().item()
            else:
                # Overlap criterion: IoU of the raw masks
                intersection = (current_mask & binary_masks[j]).sum().item()
                union = (current_mask | binary_masks[j]).sum().item()
                iou = intersection / union if union > 0 else 0
                should_merge = iou > iou_threshold

            if should_merge:
                current_mask = current_mask | binary_masks[j]
                if current_dilated is not None:
                    current_dilated = current_dilated | dilated_masks[j]
                current_score = max(current_score, scores[j].item())
                used[j] = True

        merged_masks.append(current_mask)
        merged_scores.append(current_score)
        merged_boxes.append(current_box)

    if len(merged_masks) > 0:
        merged_masks = torch.stack(merged_masks)
        merged_scores = torch.tensor(merged_scores, device=scores.device)
        merged_boxes = torch.stack(merged_boxes)
    else:
        merged_masks = binary_masks[:0]
        merged_scores = scores[:0]
        merged_boxes = boxes[:0]

    return merged_masks, merged_scores, merged_boxes


def compute_pixel_metrics(coco_gt_dict, coco_predictions):
    """
    Pixel-level (semantic) segmentation metrics per (image, prompt) unit.

    Instance matching (mAP/cgF1) requires one prediction to cover one GT
    instance at IoU >= threshold, which triple-penalizes a crack predicted as
    several disjoint fragments (GT missed + each fragment a false positive).
    Here we instead union all predicted masks and all GT masks per unit and
    compare pixels, so fragmentation does not matter — only coverage does.
    This is the standard metric in the crack-segmentation literature.

    Predictions are the post-NMS/post-merge set (already score-filtered), so
    the prob_threshold acts as the binarization threshold.

    Returns dict with:
        pixel_iou:       micro-averaged IoU (all TP/FP/FN pixels pooled)
        pixel_precision: micro-averaged precision
        pixel_recall:    micro-averaged recall
        mean_unit_iou:   mean per-unit IoU over units where GT or prediction
                         is non-empty (empty-vs-empty units are skipped)
    """
    from collections import defaultdict

    def _decode_rle(rle):
        rle = dict(rle)
        if isinstance(rle.get("counts"), str):
            rle["counts"] = rle["counts"].encode("utf-8")
        return mask_utils.decode(rle).astype(bool)

    preds_by_img = defaultdict(list)
    for p in coco_predictions:
        preds_by_img[p["image_id"]].append(p["segmentation"])

    gts_by_img = defaultdict(list)
    for a in coco_gt_dict["annotations"]:
        if "segmentation" in a:
            gts_by_img[a["image_id"]].append(a["segmentation"])

    tp = fp = fn = 0
    unit_ious = []

    for img in coco_gt_dict["images"]:
        img_id = img["id"]
        h, w = img["height"], img["width"]

        gt_union = np.zeros((h, w), dtype=bool)
        for rle in gts_by_img.get(img_id, []):
            gt_union |= _decode_rle(rle)

        pred_union = np.zeros((h, w), dtype=bool)
        for rle in preds_by_img.get(img_id, []):
            pred_union |= _decode_rle(rle)

        inter = int(np.logical_and(gt_union, pred_union).sum())
        union = int(np.logical_or(gt_union, pred_union).sum())
        tp += inter
        fp += int(pred_union.sum()) - inter
        fn += int(gt_union.sum()) - inter
        if union > 0:
            unit_ious.append(inter / union)

    return {
        "pixel_iou": tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0,
        "pixel_precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "pixel_recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "mean_unit_iou": float(np.mean(unit_ious)) if unit_ious else 0.0,
    }
