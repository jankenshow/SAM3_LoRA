"""Streaming evaluation reuses inference scoring and the training data loader."""

import json
from pathlib import Path

from sam3_lora.data.dataset import build_dataset
from sam3_lora.inference.postprocess import postprocess
from sam3_lora.inference.session import InferenceSession

from .coco import PromptCOCOBuilder, compute_coco_metrics
from .masks import compute_pixel_metrics


def evaluate(
    config,
    data_dir,
    output_dir,
    *,
    weights=None,
    base=False,
    device="auto",
    num_samples=None,
    include_negatives=False,
    pixel_metrics=True,
    use_presence=True,
    session_factory=InferenceSession,
    **policy,
):
    dataset_config = {**config, "training": {**config.get("training", {}), "seed": 42}}
    dataset = build_dataset(dataset_config, data_dir, "")
    dataset.include_negatives = include_negatives
    builder = PromptCOCOBuilder()
    limit = len(dataset) if num_samples is None else min(num_samples, len(dataset))
    if limit <= 0:
        raise ValueError("Evaluation requires at least one image")
    session = session_factory(config, weights, device=device, base=base)
    for index in range(limit):
        sample = dataset[index]
        outputs = session.raw_batch([sample])
        if len(outputs["pred_logits"]) != len(sample.queries):
            raise ValueError("SAM3 output prompt count disagrees with the data queries")
        for row, query in enumerate(sample.queries):
            presence = outputs.get("presence_logit_dec") if use_presence else None
            prediction = postprocess(
                outputs["pred_logits"][row],
                outputs["pred_boxes"][row],
                outputs["pred_masks"][row],
                presence=presence[row] if presence is not None else None,
                **policy,
            )
            builder.add(sample, query, prediction)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    gt_path, pred_path = output / "ground_truth.json", output / "predictions.json"
    gt_path.write_text(json.dumps(builder.ground_truth))
    pred_path.write_text(json.dumps(builder.predictions))
    results = compute_coco_metrics(builder.ground_truth, builder.predictions)
    if pixel_metrics:
        results.update(compute_pixel_metrics(builder.ground_truth, builder.predictions))
    results.update(optional_cgf1(gt_path, pred_path))
    negatives = [image["id"] for image in builder.ground_truth["images"] if image["negative"]]
    detected = {p["image_id"] for p in builder.predictions}
    results["negative_prompt_fpr"] = (
        sum(i in detected for i in negatives) / len(negatives) if negatives else None
    )
    results.update(
        images=limit,
        prompt_units=len(builder.ground_truth["images"]),
        negative_units=len(negatives),
    )
    (output / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    return results


def optional_cgf1(gt_path, pred_path):
    """Report missing optional vendor evaluation dependencies explicitly."""
    try:
        from sam3.eval.cgf1_eval import CGF1Evaluator
    except ImportError as error:
        return {"cgF1_unavailable": str(error)}
    return {
        "cgF1": CGF1Evaluator(gt_path=str(gt_path), iou_type="segm", verbose=False).evaluate(
            str(pred_path)
        )
    }
