import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from sam3_lora.data.bridge import collate_samples, final_output
from sam3_lora.data.dataset import COCOSegmentDataset
from sam3_lora.inference.session import InferenceSession
from sam3_lora.modeling import build_model
from sam3_lora.utils.tensors import move_to_device
from tests.helpers import make_coco
from tests.runtime_fixtures import lightweight_vendor


class RuntimeBoundaryTests(unittest.TestCase):
    def test_real_vendor_collator_and_ids_without_model(self):
        with tempfile.TemporaryDirectory() as temp, lightweight_vendor():
            make_coco(temp)
            dataset = COCOSegmentDataset(temp, "", resolution=8, num_negatives=0)
            batch = collate_samples([dataset[0], dataset[1]])["input"]
            self.assertEqual(tuple(batch.img_batch.shape), (2, 3, 8, 8))
            self.assertEqual(batch.find_targets[0].num_boxes.tolist(), [1, 0, 1, 0])
            self.assertEqual(batch.find_metadatas[0].coco_image_id.tolist(), [10, 10, 42, 42])
            self.assertEqual(batch.find_metadatas[0].original_image_id.tolist(), [0, 0, 1, 1])
            torch.testing.assert_close(batch.find_targets[0].boxes[0], dataset[0].objects[0].bbox)
            moved = move_to_device(batch, "cpu")
            self.assertIs(type(moved), type(batch))
            self.assertIs(type(moved.find_targets[0]), type(batch.find_targets[0]))

    def test_background_only_vendor_collation(self):
        with tempfile.TemporaryDirectory() as temp, lightweight_vendor():
            make_coco(temp)
            dataset = COCOSegmentDataset(temp, "", resolution=8, include_negatives=False)
            batch = collate_samples([dataset[2]])["input"]
            self.assertEqual(batch.find_targets[0].num_boxes.tolist(), [0])
            self.assertEqual(batch.find_targets[0].boxes.numel(), 0)

    def test_output_container_restores_iteration_mode(self):
        with lightweight_vendor() as output_type:
            output = output_type(
                [[{"step": 0}, {"step": 1}]], iter_mode=output_type.IterMode.LAST_STEP_PER_STAGE
            )
            self.assertEqual(final_output(output), {"step": 1})
            self.assertIs(output.iter_mode, output_type.IterMode.LAST_STEP_PER_STAGE)

    def test_session_uses_real_collator_with_tiny_fake_outputs(self):
        class Tiny(torch.nn.Module):
            def forward(self, batch):
                from sam3.model.model_misc import SAM3Output

                count = len(batch.find_inputs[0].img_ids)
                return SAM3Output(
                    [
                        [
                            {
                                "pred_logits": torch.ones(count, 1, 1) * 5,
                                "pred_boxes": torch.tensor([0.5, 0.5, 0.5, 0.5]).expand(
                                    count, 1, 4
                                ),
                                "pred_masks": torch.ones(count, 1, 4, 4),
                                "presence_logit_dec": torch.ones(count, 1) * 5,
                            }
                        ]
                    ]
                )

        with tempfile.TemporaryDirectory() as temp, lightweight_vendor():
            make_coco(temp)
            session = InferenceSession({}, device="cpu", resolution=8, model=Tiny())
            result = session.predict(Path(temp) / "0.png", ["dog", "cat"])
            self.assertEqual(result[0]["num_detections"], 1)
            torch.testing.assert_close(
                torch.from_numpy(result[0]["boxes"]), torch.tensor([[2.0, 1.5, 6.0, 4.5]])
            )
            self.assertEqual(result[1]["masks"].shape, (1, 6, 8))

    def test_missing_local_weights_fail_before_vendor_import(self):
        with patch(
            "sam3_lora.modeling.ensure_sam3_path",
            side_effect=AssertionError("Should not import vendor"),
        ):
            with self.assertRaises(ValueError):
                build_model({})
            with self.assertRaises(FileNotFoundError):
                build_model({"model": {"checkpoint_path": "/no/such/sam3.pt"}})


class SourceLayoutTests(unittest.TestCase):
    def test_submodule_source_path_and_missing_checkout(self):
        import sys

        from sam3_lora.modeling import ensure_sam3_path

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            package = root / "vendor/sam3/sam3"
            package.mkdir(parents=True)
            entry = package / "__init__.py"
            entry.touch()
            before = sys.path[:]
            try:
                with patch("sam3_lora.modeling.PROJECT_ROOT", root):
                    ensure_sam3_path()
                    self.assertEqual(sys.path[0], str(package.parent))
                    entry.unlink()
                    with self.assertRaisesRegex(FileNotFoundError, "submodule update"):
                        ensure_sam3_path()
            finally:
                sys.path[:] = before

    def test_bpe_lookup_for_official_checkout_and_wheel(self):
        from sam3_lora.modeling import find_bpe_path

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            package = root / "sam3"
            package.mkdir()
            entry = package / "__init__.py"
            entry.touch()
            asset = root / "assets/bpe_simple_vocab_16e6.txt.gz"
            asset.parent.mkdir()
            asset.write_bytes(b"fixture")
            self.assertEqual(find_bpe_path(entry), str(asset))
            nested = package / "assets/bpe_simple_vocab_16e6.txt.gz"
            nested.parent.mkdir()
            asset.rename(nested)
            self.assertEqual(find_bpe_path(entry), str(nested))
            nested.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "vocabulary"):
                find_bpe_path(entry)
