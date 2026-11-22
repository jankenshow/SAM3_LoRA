import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from sam3_lora.data.dataset import COCOSegmentDataset
from sam3_lora.evaluation.coco import PromptCOCOBuilder, compute_coco_metrics
from sam3_lora.evaluation.runner import evaluate
from sam3_lora.inference.postprocess import Prediction
from tests.helpers import make_coco


class EvaluationTests(unittest.TestCase):
    def test_ap_handles_perfect_empty_and_negative_detections(self):
        with tempfile.TemporaryDirectory() as temp:
            make_coco(temp)
            sample = COCOSegmentDataset(temp, "", resolution=8, include_negatives=False)[0]
            obj = sample.objects[0]
            builder = PromptCOCOBuilder(resolution=8)
            builder.add(
                sample,
                sample.queries[0],
                Prediction(obj.bbox[None], torch.tensor([0.9]), obj.segment[None]),
            )
            self.assertAlmostEqual(
                compute_coco_metrics(builder.ground_truth, copy.deepcopy(builder.predictions))[
                    "mAP"
                ],
                1.0,
            )
            self.assertEqual(compute_coco_metrics(builder.ground_truth, [])["mAP"], 0.0)

    def test_streaming_evaluation_with_tiny_outputs_and_negatives(self):
        class EmptySession:
            def __init__(self, *args, **kwargs):
                pass

            def raw_batch(self, samples):
                count = len(samples[0].queries)
                return {
                    "pred_logits": torch.empty(count, 0, 1),
                    "pred_boxes": torch.empty(count, 0, 4),
                    "pred_masks": torch.empty(count, 0, 4, 4),
                }

        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp) / "data"
            make_coco(data)
            out = Path(temp) / "output"
            with patch(
                "sam3_lora.evaluation.runner.optional_cgf1",
                return_value={"cgF1_unavailable": "test fixture"},
            ):
                result = evaluate(
                    {"training": {"resolution": 8, "num_negatives": 0}},
                    data,
                    out,
                    base=True,
                    device="cpu",
                    include_negatives=True,
                    session_factory=EmptySession,
                )
            self.assertEqual(result["mAP"], 0.0)
            self.assertEqual(result["negative_prompt_fpr"], 0.0)
            self.assertGreater(result["negative_units"], 0)
            self.assertEqual(json.loads((out / "metrics.json").read_text())["images"], 3)
            self.assertEqual(json.loads((out / "predictions.json").read_text()), [])
