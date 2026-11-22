"""Compare base and LoRA using the same inference and rendering pipeline."""

from pathlib import Path

from PIL import Image

from .session import InferenceSession
from .visualization import save_results


def compare(
    config,
    image,
    prompts,
    output,
    *,
    weights=None,
    device="auto",
    resolution=1008,
    show_boxes=False,
    show_masks=True,
    show_scores=True,
    **policy,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, base in [("base", True), ("lora", False)]:
        session = InferenceSession(config, weights, device=device, base=base, resolution=resolution)
        result = session.predict(image, prompts, **policy)
        save_results(
            result,
            output / f"{name}.png",
            show_boxes=show_boxes,
            show_masks=show_masks,
            show_scores=show_scores,
        )
        del result, session
    with Image.open(output / "base.png") as left, Image.open(output / "lora.png") as right:
        combined = Image.new("RGB", (left.width + right.width, max(left.height, right.height)))
        combined.paste(left, (0, 0))
        combined.paste(right, (left.width, 0))
        combined.save(output / "comparison.png")
    return output / "comparison.png"
