import tempfile
import unittest
from pathlib import Path

import torch

from sam3_lora.data.records import Instance, PromptQuery, Sample
from sam3_lora.evaluation.coco import PromptCOCOBuilder
from sam3_lora.evaluation.masks import compute_pixel_metrics
from sam3_lora.inference.postprocess import Prediction, joint_scores, pixel_output, postprocess
from sam3_lora.inference.visualization import save_results


class PostprocessTests(unittest.TestCase):
    def prediction(self):
        return Prediction(
            torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
            torch.tensor([0.8]),
            torch.ones(1, 4, 4, dtype=torch.bool),
        )

    def test_presence_changes_threshold_consistently(self):
        logits = torch.tensor([[2.0], [1.0]])
        boxes = torch.ones(2, 4)
        masks = torch.ones(2, 4, 4)
        torch.testing.assert_close(
            joint_scores(logits, torch.tensor([0.0])), logits.sigmoid().flatten() * 0.5
        )
        positive = postprocess(logits, boxes, masks, threshold=0.5)
        negative = postprocess(logits, boxes, masks, presence=torch.tensor([-10.0]), threshold=0.5)
        self.assertEqual(len(positive.scores), 1)
        self.assertEqual(len(negative.scores), 0)

    def test_nms_preserves_disjoint_masks(self):
        masks = torch.full((3, 4, 4), -1.0)
        masks[:2, :, :2] = 1
        masks[2, :, 2:] = 1
        result = postprocess(torch.tensor([[3.0], [2.0], [1.0]]), torch.zeros(3, 4), masks)
        self.assertEqual(len(result.scores), 2)
        self.assertGreater(result.scores[0], result.scores[1])

    def test_empty_and_pixel_coordinate_output(self):
        pred = self.prediction()
        boxes, masks = pixel_output(pred, (8, 12))
        torch.testing.assert_close(boxes, torch.tensor([[3.0, 2.0, 9.0, 6.0]]))
        self.assertEqual(tuple(masks.shape), (1, 8, 12))
        empty = postprocess(torch.zeros(0, 1), torch.zeros(0, 4), torch.zeros(0, 4, 4))
        self.assertEqual(tuple(pixel_output(empty, (8, 12))[1].shape), (0, 8, 12))

    def test_prompt_coco_ids_boxes_and_nearest_masks(self):
        obj = Instance(0, torch.tensor([0.5, 0.5, 0.5, 0.5]), torch.ones(4, 4, dtype=torch.bool))
        sample = Sample(torch.zeros(3, 4, 4), (obj,), (), 7, 420, (8, 12), Path("unused"))
        builder = PromptCOCOBuilder(resolution=4)
        builder.add(sample, PromptQuery("dog", (0,)), self.prediction())
        builder.add(
            sample,
            PromptQuery("cat", (), True),
            Prediction(torch.empty(0, 4), torch.empty(0), torch.empty(0, 4, 4, dtype=torch.bool)),
        )
        self.assertEqual([i["id"] for i in builder.ground_truth["images"]], [0, 1])
        self.assertEqual(builder.ground_truth["images"][0]["source_id"], 420)
        self.assertEqual(builder.ground_truth["images"][0]["source_index"], 7)
        self.assertEqual(builder.ground_truth["annotations"][0]["bbox"], [1.0, 1.0, 2.0, 2.0])
        self.assertEqual(builder.ground_truth["annotations"][0]["area"], 16)
        self.assertEqual(
            compute_pixel_metrics(builder.ground_truth, builder.predictions)["pixel_iou"], 1.0
        )
        self.assertEqual(compute_pixel_metrics(builder.ground_truth, [])["pixel_recall"], 0.0)

    def test_box_only_segmentation_evaluation_rejected(self):
        obj = Instance(0, torch.ones(4), None)
        sample = Sample(torch.zeros(3, 4, 4), (obj,), (), 0, 10, (4, 4), Path("x"))
        with self.assertRaisesRegex(ValueError, "GT masks"):
            PromptCOCOBuilder().add(sample, PromptQuery("dog", (0,)), self.prediction())

    def test_invalid_thresholds(self):
        with self.assertRaises(ValueError):
            postprocess(torch.zeros(0, 1), torch.zeros(0, 4), torch.zeros(0, 4, 4), threshold=2)

    def test_empty_rendering_and_json(self):
        import numpy as np
        from PIL import Image

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "result.png"
            save_results(
                {
                    "_image": Image.new("RGB", (8, 8)),
                    0: {
                        "prompt": "dog",
                        "boxes": np.empty((0, 4)),
                        "scores": np.empty(0),
                        "masks": np.empty((0, 8, 8), dtype=bool),
                        "num_detections": 0,
                    },
                },
                path,
            )
            self.assertTrue(path.exists())
            self.assertTrue(path.with_suffix(".json").exists())
