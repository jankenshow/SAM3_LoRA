"""Pillow-based rendering/output; no model, pyplot backend, or GPU dependency."""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

COLORS = [(248, 84, 84), (53, 166, 241), (73, 198, 136), (240, 178, 63), (190, 97, 240)]


def save_results(results, output_path, *, show_boxes=True, show_masks=True, show_scores=True):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas = np.array(results["_image"].convert("RGB"), copy=True)
    report = []
    for index in sorted(k for k in results if isinstance(k, int)):
        result = results[index]
        color = np.array(COLORS[index % len(COLORS)])
        row = {
            "prompt": result["prompt"],
            "boxes": result["boxes"].tolist(),
            "scores": result["scores"].tolist(),
            "masks": [],
        }
        for number, mask in enumerate(result["masks"]):
            name = f"{output.stem}_prompt{index}_mask{number}.png"
            Image.fromarray(mask.astype(np.uint8) * 255).save(output.parent / name)
            row["masks"].append(name)
            if show_masks:
                canvas[mask] = (canvas[mask] * 0.6 + color * 0.4).astype(np.uint8)
        report.append(row)
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    for index in sorted(k for k in results if isinstance(k, int)):
        result = results[index]
        color = COLORS[index % len(COLORS)]
        for box, score in zip(result["boxes"], result["scores"]):
            if show_boxes:
                draw.rectangle(tuple(box), outline=color, width=2)
            if show_scores:
                draw.text(
                    (float(box[0]), max(0, float(box[1]) - 12)),
                    f"{result['prompt']} {score:.2f}",
                    fill=color,
                )
    image.save(output)
    output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return output
