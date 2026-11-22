"""Text resolution and negative supervision, without any SAM3 imports."""

import json
import random
from dataclasses import dataclass
from pathlib import Path

from .records import PromptQuery

ANNOTATION_TEXT_KEYS = ("caption", "description", "text", "phrase", "expression")
IMAGE_TEXT_KEYS = ("caption", "description", "text")
CATEGORY_TEXT_KEYS = ("description", "caption", "instructions", "prompt")
PROMPT_SOURCES = ("auto", "caption", "image_caption", "category_description", "category_name")
DEFAULT_GENERIC_NEGATIVES = (
    "car",
    "person",
    "dog",
    "cat",
    "tree",
    "building",
    "sky",
    "road",
    "chair",
    "table",
    "bottle",
    "phone",
    "bicycle",
    "bird",
    "flower",
    "water",
    "grass",
    "cloud",
    "window",
    "door",
)
MAX_ALWAYS_NEGATIVES = 50


def first_text(container, keys):
    if isinstance(container, dict):
        for key in keys:
            value = container.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def load_descriptions_sidecar(split_dir):
    path = Path(split_dir) / "_descriptions.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    return tuple(data.get(key, {}) for key in ("by_annotation", "by_image", "by_category"))


class PromptResolver:
    def __init__(self, source="auto", sidecar=({}, {}, {})):
        if source not in PROMPT_SOURCES:
            raise ValueError(f"Unknown prompt_source: {source}")
        self.source, self.sidecar = source, sidecar

    def resolve(self, annotation, image, category):
        by_ann, by_img, by_cat = self.sidecar
        name = category["name"]
        choices = {
            "caption": by_ann.get(str(annotation.get("id")))
            or first_text(annotation, ANNOTATION_TEXT_KEYS),
            "image_caption": by_img.get(image["file_name"])
            or by_img.get(str(image["id"]))
            or first_text(image, IMAGE_TEXT_KEYS),
            "category_description": by_cat.get(name.lower())
            or first_text(category, CATEGORY_TEXT_KEYS),
            "category_name": name,
        }
        if self.source == "auto":
            result = next(value for value in choices.values() if value)
        else:
            result = choices[self.source] or name
        if not isinstance(result, str) or not result.strip():
            raise ValueError("Prompts must be non-empty strings")
        return result.strip().lower()


def filter_generic_negatives(generic_negatives, prompt_texts):
    words = {word for text in prompt_texts for word in text.lower().split()}
    return [text for text in generic_negatives if not set(text.lower().split()) & words]


@dataclass(frozen=True)
class NegativePromptPolicy:
    num_negatives: int = 2
    num_cross_negatives: int = 2
    max_always_negatives: int = MAX_ALWAYS_NEGATIVES

    def __post_init__(self):
        if min(self.num_negatives, self.num_cross_negatives, self.max_always_negatives) < 0:
            raise ValueError("Negative-prompt counts cannot be negative")

    def queries(self, present_texts, prompt_corpus, generic_negatives, rng=random):
        present = {s.lower() for s in present_texts}
        corpus = {s.lower() for s in prompt_corpus}
        absent = sorted(corpus - present)
        chosen = (
            absent
            if len(corpus) <= self.max_always_negatives
            else rng.sample(absent, min(len(absent), self.num_cross_negatives))
        )
        generic = sorted(
            set(filter_generic_negatives(generic_negatives, corpus)) - present - set(chosen)
        )
        chosen += rng.sample(generic, min(len(generic), self.num_negatives))
        return [PromptQuery(text, (), True) for text in chosen]
