"""Reusable image-inference session; model construction is a separate dependency."""

from pathlib import Path

import torch
from PIL import Image

from sam3_lora.config import load_config
from sam3_lora.data.bridge import collate_samples, final_output
from sam3_lora.data.records import PromptQuery, Sample
from sam3_lora.data.transforms import image_tensor
from sam3_lora.modeling import build_adapted_model, build_model
from sam3_lora.utils.tensors import move_to_device, resolve_device

from .postprocess import pixel_output, postprocess


class InferenceSession:
    def __init__(
        self, config, weights=None, *, device="auto", base=False, resolution=1008, model=None
    ):
        self.config, self.device, self.resolution = config, resolve_device(device), resolution
        if model is None:
            if base:
                model = build_model(config, "cpu", inference=True)
            else:
                weights = weights or str(
                    Path(config["output"]["output_dir"]) / "best_lora_weights.pt"
                )
                if not Path(weights).is_file():
                    raise FileNotFoundError(weights)
                model = build_adapted_model(config, "cpu", inference=True, weights=weights)
        self.model = model.to(self.device).eval()

    @torch.inference_mode()
    def raw_batch(self, samples):
        batch = move_to_device(collate_samples(samples)["input"], self.device)
        return final_output(self.model(batch))

    def predict(self, image_path, text_prompts, **policy):
        if not text_prompts or any(not t.strip() for t in text_prompts):
            raise ValueError("At least one non-empty prompt is required")
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        tensor = image_tensor(image, self.resolution)
        result = {"_image": image}
        for index, prompt in enumerate(text_prompts):
            sample = Sample(
                tensor,
                (),
                (PromptQuery(prompt),),
                0,
                0,
                (image.height, image.width),
                Path(image_path),
            )
            outputs = self.raw_batch([sample])
            presence = outputs.get("presence_logit_dec")
            prediction = postprocess(
                outputs["pred_logits"][0],
                outputs["pred_boxes"][0],
                outputs["pred_masks"][0],
                presence=presence[0] if presence is not None else None,
                **policy,
            )
            boxes, masks = pixel_output(prediction, sample.original_size)
            result[index] = {
                "prompt": prompt,
                "boxes": boxes.numpy(),
                "scores": prediction.scores.numpy(),
                "masks": masks.numpy(),
                "num_detections": len(prediction.scores),
            }
        return result


class SAM3LoRAInference(InferenceSession):
    """Compatibility facade for the previous infer_sam.py Python API."""

    def __init__(
        self,
        config_path,
        weights_path=None,
        resolution=1008,
        detection_threshold=0.5,
        nms_iou_threshold=0.5,
        device="auto",
    ):
        super().__init__(
            load_config(config_path), weights_path, device=device, resolution=resolution
        )
        self.policy = {"threshold": detection_threshold, "nms_iou": nms_iou_threshold}

    def predict(self, image_path, text_prompts):
        return super().predict(
            image_path,
            [text_prompts] if isinstance(text_prompts, str) else text_prompts,
            **self.policy,
        )

    def visualize(self, results, output_path, show_boxes=True, show_masks=True, show_scores=True):
        from .visualization import save_results

        return save_results(
            results,
            output_path,
            show_boxes=show_boxes,
            show_masks=show_masks,
            show_scores=show_scores,
        )
