"""COCO and ODVG loaders share prompt grouping, geometry and negative policy."""

import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from .paths import image_path
from .prompts import (
    DEFAULT_GENERIC_NEGATIVES,
    NegativePromptPolicy,
    PromptResolver,
    first_text,
    load_descriptions_sidecar,
)
from .records import Instance, PromptQuery, Sample
from .transforms import decode_segmentation, image_tensor, xywh_to_normalized


class COCOSegmentDataset(Dataset):
    def __init__(
        self,
        data_dir,
        split="train",
        num_negatives=2,
        generic_negatives=None,
        prompt_source="auto",
        mix_class_prompt=False,
        num_cross_negatives=2,
        resolution=1008,
        seed=42,
        include_negatives=True,
        *,
        document=None,
    ):
        self.split_dir = Path(data_dir) / split if split else Path(data_dir)
        self.coco_data = (
            document
            if document is not None
            else json.loads((self.split_dir / "_annotations.coco.json").read_text())
        )
        self.images = {x["id"]: x for x in self.coco_data["images"]}
        if len(self.images) != len(self.coco_data["images"]):
            raise ValueError("Duplicate COCO image IDs")
        self.image_ids = sorted(self.images)
        self.category_defs = {x["id"]: x for x in self.coco_data["categories"]}
        self.categories = {k: v["name"] for k, v in self.category_defs.items()}
        self.img_to_anns = defaultdict(list)
        self.resolver = PromptResolver(prompt_source, load_descriptions_sidecar(self.split_dir))
        self.mix_class_prompt, self.resolution, self.seed = mix_class_prompt, resolution, seed
        self.include_negatives = include_negatives
        self.policy = NegativePromptPolicy(num_negatives, num_cross_negatives)
        self.full_captions = {}
        corpus = set()
        for ann in self.coco_data["annotations"]:
            if ann["image_id"] not in self.images or ann["category_id"] not in self.categories:
                raise ValueError(f"Annotation {ann.get('id')} references an unknown image/category")
            self.img_to_anns[ann["image_id"]].append(ann)
            corpus.add(
                self.resolver.resolve(
                    ann, self.images[ann["image_id"]], self.category_defs[ann["category_id"]]
                )
            )
            if mix_class_prompt:
                corpus.add(self.categories[ann["category_id"]].lower())
        self.prompt_corpus = sorted(corpus)
        self.generic_negatives = tuple(
            DEFAULT_GENERIC_NEGATIVES if generic_negatives is None else generic_negatives
        )

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        source_id = self.image_ids[index]
        info = self.images[source_id]
        path = image_path(self.split_dir, info["file_name"])
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        width, height = image.size
        if (info.get("width", width), info.get("height", height)) != (width, height):
            raise ValueError(f"Annotation dimensions disagree with image: {path}")
        objects, grouped = [], defaultdict(list)
        for ann in self.img_to_anns[source_id]:
            # Reject invalid masks/boxes instead of silently shifting label/object alignment.
            bbox = xywh_to_normalized(ann["bbox"], width, height)
            mask = decode_segmentation(ann.get("segmentation"), height, width, self.resolution)
            obj = Instance(len(objects), bbox, mask)
            objects.append(obj)
            category = self.category_defs[ann["category_id"]]
            text = self.resolver.resolve(ann, info, category)
            grouped[text].append(obj.object_id)
            if self.mix_class_prompt and category["name"].lower() != text:
                grouped[category["name"].lower()].append(obj.object_id)
        if self.full_captions.get(source_id) and objects:
            grouped.setdefault(self.full_captions[source_id].lower(), list(range(len(objects))))
        queries = [PromptQuery(text, tuple(ids)) for text, ids in grouped.items()]
        if not queries:
            queries = [PromptQuery("object", (), True)]
        if self.include_negatives:
            queries += self.policy.queries(
                grouped,
                self.prompt_corpus,
                self.generic_negatives,
                rng=random.Random(self.seed + index) if self.seed is not None else random,
            )
        return Sample(
            image_tensor(image, self.resolution),
            tuple(objects),
            tuple(queries),
            index,
            source_id,
            (height, width),
            path,
        )


class ODVGDataset(COCOSegmentDataset):
    def __init__(
        self,
        data_dir,
        split="train",
        use_full_caption=True,
        annotations_file="_annotations.odvg.jsonl",
        **kwargs,
    ):
        directory = Path(data_dir) / split if split else Path(data_dir)
        records = [
            json.loads(line)
            for line in (directory / annotations_file).read_text().splitlines()
            if line.strip()
        ]
        document = {"images": [], "annotations": [], "categories": []}
        categories, captions = {}, {}
        for index, record in enumerate(records):
            document["images"].append(
                {
                    "id": index,
                    "file_name": record["filename"],
                    "height": record["height"],
                    "width": record["width"],
                }
            )
            grounding = record.get("grounding") or {}
            regions = grounding.get("regions") or (record.get("detection") or {}).get(
                "instances", []
            )
            for region in regions:
                phrase = first_text(region, ("phrase", "caption", "category", "text"))
                if not phrase:
                    raise ValueError(f"ODVG record {index}: region has no phrase/category")
                category = categories.setdefault(phrase, len(categories))
                x0, y0, x1, y1 = region["bbox"]
                document["annotations"].append(
                    {
                        "id": len(document["annotations"]),
                        "image_id": index,
                        "category_id": category,
                        "caption": phrase,
                        "bbox": [x0, y0, x1 - x0, y1 - y0],
                        "segmentation": region.get("segmentation") or region.get("mask"),
                    }
                )
            if use_full_caption and grounding.get("caption"):
                captions[index] = grounding["caption"]
        document["categories"] = [{"id": i, "name": name} for name, i in categories.items()]
        super().__init__(data_dir, split, document=document, **kwargs)
        self.full_captions = captions
        self.prompt_corpus = sorted(
            set(self.prompt_corpus) | {s.lower() for s in captions.values()}
        )


def build_dataset(config, data_dir, split):
    t = config["training"]
    options = {
        key: t[key]
        for key in (
            "num_negatives",
            "generic_negatives",
            "prompt_source",
            "mix_class_prompt",
            "num_cross_negatives",
            "resolution",
            "seed",
        )
        if key in t
    }
    # Training samples fresh negatives; validation remains reproducible.
    if split == "train":
        options["seed"] = None
    kind = t.get("data_format", "coco").lower()
    if kind == "coco":
        return COCOSegmentDataset(data_dir, split, **options)
    if kind == "odvg":
        return ODVGDataset(
            data_dir,
            split,
            use_full_caption=t.get("use_full_caption", True),
            annotations_file=t.get("odvg_annotations_file", "_annotations.odvg.jsonl"),
            **options,
        )
    raise ValueError(f"Unknown data_format: {kind}")
