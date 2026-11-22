"""Non-destructive COCO metadata inspection and deterministic image-level splits."""

import json
import random
import shutil
from collections import Counter
from pathlib import Path

from .paths import image_path


def inspect_split(directory, check_images=False):
    directory = Path(directory)
    data = json.loads((directory / "_annotations.coco.json").read_text())
    ids = [i["id"] for i in data["images"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate image IDs")
    categories = {c["id"]: c["name"] for c in data["categories"]}
    annotation_ids = set()
    for ann in data["annotations"]:
        if ann["id"] in annotation_ids:
            raise ValueError(f"Duplicate annotation ID: {ann['id']}")
        annotation_ids.add(ann["id"])
        if ann["image_id"] not in set(ids) or ann["category_id"] not in categories:
            raise ValueError(f"Broken annotation reference: {ann['id']}")
        if len(ann["bbox"]) != 4 or min(ann["bbox"][2:]) <= 0:
            raise ValueError(f"Invalid bbox: {ann['id']}")
    if check_images:
        for image in data["images"]:
            path = image_path(directory, image["file_name"])
            if not path.is_file():
                raise FileNotFoundError(path)
    counts = Counter(a["category_id"] for a in data["annotations"])
    return {
        "images": len(ids),
        "annotations": len(annotation_ids),
        "categories": {name: counts[key] for key, name in categories.items()},
    }


def split_coco(source_dir, output_dir, valid_fraction=0.1, test_fraction=0.1, seed=42):
    source, output = Path(source_dir), Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if min(valid_fraction, test_fraction) < 0 or valid_fraction + test_fraction >= 1:
        raise ValueError("valid/test fractions must be >= 0 and sum to < 1")
    inspect_split(source, check_images=True)
    data = json.loads((source / "_annotations.coco.json").read_text())
    images = list(data["images"])
    random.Random(seed).shuffle(images)
    n_valid, n_test = int(len(images) * valid_fraction), int(len(images) * test_fraction)
    groups = {
        "valid": images[:n_valid],
        "test": images[n_valid : n_valid + n_test],
        "train": images[n_valid + n_test :],
    }
    for split, subset in groups.items():
        directory = output / split
        directory.mkdir(parents=True)
        selected = {i["id"] for i in subset}
        annotations = [a for a in data["annotations"] if a["image_id"] in selected]
        for image in subset:
            destination = directory / image["file_name"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / image["file_name"], destination)
        (directory / "_annotations.coco.json").write_text(
            json.dumps({**data, "images": subset, "annotations": annotations})
        )
        sidecar = source / "_descriptions.json"
        if sidecar.exists():
            descriptions = json.loads(sidecar.read_text())
            descriptions["by_annotation"] = {
                k: v
                for k, v in descriptions.get("by_annotation", {}).items()
                if k in {str(a["id"]) for a in annotations}
            }
            descriptions["by_image"] = {
                k: v
                for k, v in descriptions.get("by_image", {}).items()
                if k in {str(i["id"]) for i in subset} | {i["file_name"] for i in subset}
            }
            (directory / sidecar.name).write_text(json.dumps(descriptions))
    return {name: len(subset) for name, subset in groups.items()}
