"""Convert per-image JSON, preserving multi-class names."""

import json
from pathlib import Path


def convert_roboflow_to_coco(data_dir, output_file="_annotations.coco.json"):
    directory = Path(data_dir)
    output = directory / output_file
    if output.exists():
        raise FileExistsError(output)
    result = {"images": [], "annotations": [], "categories": []}
    names = {}
    for file in sorted(directory.glob("*.json")):
        record = json.loads(file.read_text())
        if "image" not in record:
            continue
        image_id = len(result["images"]) + 1
        result["images"].append({**record["image"], "id": image_id})
        categories = {c["id"]: c["name"] for c in record.get("categories", [])}
        for annotation in record.get("annotations", []):
            name = (
                annotation.get("category_name")
                or annotation.get("class")
                or categories.get(annotation.get("category_id"))
            )
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    f"{file}: annotation needs category_name/class or a categories table"
                )
            category_id = names.setdefault(name, len(names) + 1)
            result["annotations"].append(
                {
                    **annotation,
                    "id": len(result["annotations"]) + 1,
                    "image_id": image_id,
                    "category_id": category_id,
                    "iscrowd": annotation.get("iscrowd", 0),
                }
            )
    if not result["images"]:
        raise ValueError("No per-image records found")
    result["categories"] = [{"id": key, "name": name} for name, key in names.items()]
    output.write_text(json.dumps(result, indent=2) + "\n")
    return output
