import contextlib
import io
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mask_api

from sam3_lora.data.conversion.coco_to_odvg import convert_split
from sam3_lora.data.conversion.roboflow import convert_roboflow_to_coco
from sam3_lora.data.dataset import COCOSegmentDataset, ODVGDataset
from sam3_lora.data.preparation import inspect_split, split_coco
from sam3_lora.data.prompts import NegativePromptPolicy, PromptResolver
from sam3_lora.data.transforms import cxcywh_to_xyxy, decode_segmentation, xywh_to_normalized
from tests.helpers import make_coco


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.document = make_coco(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def dataset(self, **kwargs):
        return COCOSegmentDataset(self.root, "", resolution=8, **kwargs)

    def test_noncontiguous_ids_and_object_coordinates(self):
        dataset = self.dataset(include_negatives=False)
        first = dataset[0]
        second = dataset[1]
        self.assertEqual((first.source_index, first.source_id), (0, 10))
        self.assertEqual((second.source_index, second.source_id), (1, 42))
        self.assertEqual(first.queries[0].text, "dog")
        self.assertEqual(first.queries[0].object_ids, (0,))
        torch.testing.assert_close(first.objects[0].bbox, torch.tensor([0.5, 2.5 / 6, 0.5, 0.5]))
        self.assertEqual(tuple(first.objects[0].segment.shape), (8, 8))
        self.assertEqual(tuple(first.image.shape), (3, 8, 8))
        self.assertEqual(first.image[0, 0, 0], -1)

    def test_empty_image_has_background_query(self):
        sample = self.dataset(include_negatives=False)[2]
        self.assertFalse(sample.objects)
        self.assertTrue(sample.queries[0].negative)

    def test_negative_taxonomy_generic_and_reproducibility(self):
        dataset = self.dataset(num_negatives=2, generic_negatives=["dog", "bird", "bird", "tree"])
        query = dataset[0].queries
        self.assertEqual(query, dataset[0].queries)
        negatives = [q.text for q in query if q.negative]
        self.assertIn("cat", negatives)
        self.assertNotIn("dog", negatives)
        self.assertEqual(len(negatives), len(set(negatives)))
        chosen = NegativePromptPolicy(0, 1, max_always_negatives=1).queries(
            ["a"], ["a", "b", "c"], [], random.Random(3)
        )
        self.assertEqual(len(chosen), 1)

    def test_sidecar_priority_and_class_mix(self):
        (self.root / "_descriptions.json").write_text(
            json.dumps({"by_annotation": {"100": "a Shiba Inu"}, "by_category": {"dog": "animal"}})
        )
        sample = self.dataset(mix_class_prompt=True, include_negatives=False)[0]
        self.assertEqual([q.text for q in sample.queries], ["a shiba inu", "dog"])
        self.assertEqual(sample.queries[0].object_ids, sample.queries[1].object_ids)
        self.assertEqual(
            PromptResolver("category_name").resolve(
                {}, {"id": 1, "file_name": "x"}, {"name": "Dog"}
            ),
            "dog",
        )

    def test_polygon_and_rle_agree(self):
        polygon = self.document["annotations"][0]["segmentation"]
        decoded = decode_segmentation(polygon, 6, 8, 8)
        rle = mask_api.encode(
            np.asfortranarray(mask_api.decode(mask_api.merge(mask_api.frPyObjects(polygon, 6, 8))))
        )
        rle["counts"] = rle["counts"].decode()
        torch.testing.assert_close(decoded, decode_segmentation(rle, 6, 8, 8))
        self.assertIsNone(decode_segmentation(None, 6, 8, 8))

    def test_invalid_masks_and_boxes_fail(self):
        with self.assertRaises(ValueError):
            decode_segmentation("wrong", 6, 8, 8)
        with self.assertRaises(ValueError):
            xywh_to_normalized([0, 0, -1, 1], 8, 6)
        rle = mask_api.encode(np.ones((2, 2), dtype=np.uint8, order="F"))
        with self.assertRaisesRegex(ValueError, "dimensions"):
            decode_segmentation(rle, 6, 8, 8)

    def test_bbox_roundtrip(self):
        box = xywh_to_normalized([2, 1, 4, 3], 8, 6)
        torch.testing.assert_close(
            cxcywh_to_xyxy(box) * torch.tensor([8, 6, 8, 6]), torch.tensor([2.0, 1.0, 6.0, 4.0])
        )

    def test_unknown_reference_and_duplicate_ids(self):
        self.document["images"].append(self.document["images"][0])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.dataset(document=self.document)
        self.document["images"].pop()
        self.document["annotations"][0]["category_id"] = 555
        with self.assertRaisesRegex(ValueError, "unknown"):
            self.dataset(document=self.document)

    def test_path_escape_and_dimension_mismatch(self):
        self.document["images"][0]["file_name"] = "../outside.png"
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.dataset(document=self.document)[0]
        self.document["images"][0]["file_name"] = "0.png"
        self.document["images"][0]["width"] = 999
        with self.assertRaisesRegex(ValueError, "dimensions"):
            self.dataset(document=self.document)[0]

    def test_coco_odvg_same_objects_and_phrases(self):
        with contextlib.redirect_stdout(io.StringIO()):
            convert_split(self.root)
        coco = self.dataset(include_negatives=False)
        odvg = ODVGDataset(
            self.root, "", resolution=8, use_full_caption=False, include_negatives=False
        )
        for index in range(len(coco)):
            left, right = coco[index], odvg[index]
            self.assertEqual([q.text for q in left.queries], [q.text for q in right.queries])
            for a, b in zip(left.objects, right.objects):
                torch.testing.assert_close(a.bbox, b.bbox)
                torch.testing.assert_close(a.segment, b.segment)

    def test_split_is_disjoint_preserves_source_and_refuses_overwrite(self):
        before = (self.root / "_annotations.coco.json").read_bytes()
        output = self.root / "split"
        counts = split_coco(self.root, output, 0.34, 0.34, seed=3)
        self.assertEqual(counts, {"valid": 1, "test": 1, "train": 1})
        sets = [
            {
                i["id"]
                for i in json.loads((output / s / "_annotations.coco.json").read_text())["images"]
            }
            for s in counts
        ]
        self.assertEqual(set.union(*sets), {10, 42, 74})
        self.assertTrue(all(not a & b for i, a in enumerate(sets) for b in sets[i + 1 :]))
        self.assertEqual(before, (self.root / "_annotations.coco.json").read_bytes())
        with self.assertRaises(FileExistsError):
            split_coco(self.root, output)
        self.assertEqual(inspect_split(self.root, True)["annotations"], 2)

    def test_roboflow_preserves_distinct_classes(self):
        folder = self.root / "roboflow"
        folder.mkdir()
        (folder / "image.json").write_text(
            json.dumps(
                {
                    "image": self.document["images"][0],
                    "categories": self.document["categories"],
                    "annotations": self.document["annotations"],
                }
            )
        )
        output = convert_roboflow_to_coco(folder)
        converted = json.loads(output.read_text())
        self.assertEqual({c["name"] for c in converted["categories"]}, {"Dog", "Cat"})
        with self.assertRaises(FileExistsError):
            convert_roboflow_to_coco(folder)


class RefCOCOTests(unittest.TestCase):
    def test_refcoco_symlink_conversion_is_readable(self):
        import pickle

        from sam3_lora.data.conversion.refcoco import convert

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            document = make_coco(source)
            (source / "instances.json").write_text(json.dumps(document))
            with (source / "refs.p").open("wb") as stream:
                pickle.dump(
                    [
                        {
                            "ann_id": 100,
                            "image_id": 10,
                            "split": "train",
                            "sentences": [{"sent": "a Shiba dog"}],
                        }
                    ],
                    stream,
                )
            output = root / "converted"
            with contextlib.redirect_stdout(io.StringIO()):
                convert(source, source, output, symlink=True)
            self.assertTrue((output / "train/0.png").is_symlink())
            self.assertEqual(inspect_split(output / "train", True)["images"], 1)
            sample = COCOSegmentDataset(output, "train", resolution=8, include_negatives=False)[0]
            self.assertEqual(sample.queries[0].text, "a shiba dog")
            self.assertEqual(len(sample.objects), 1)
            with self.assertRaises(FileExistsError):
                convert(source, source, output)
