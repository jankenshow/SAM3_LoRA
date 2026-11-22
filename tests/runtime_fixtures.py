"""Load exact vendored data contracts without importing its model-builder entrypoint.

Only CPU dataclasses/collation and output-container code execute. This fixture
is deliberately not evidence that the complete SAM3 runtime works on this Mac.
"""

import ast
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

VENDOR = Path(__file__).resolve().parents[1] / "vendor/sam3/sam3"


@contextmanager
def lightweight_vendor():
    modules = {}
    for name in ["sam3", "sam3.model", "sam3.train", "sam3.train.data"]:
        module = ModuleType(name)
        module.__path__ = []
        modules[name] = module
    original = {
        name: module
        for name, module in sys.modules.items()
        if name == "sam3" or name.startswith("sam3.")
    }
    original_path = sys.path[:]
    sys.modules.update(modules)
    try:

        def load(name, relative, selected_classes=None):
            path = VENDOR / relative
            tree = ast.parse(path.read_text())
            if selected_classes:
                selected = [
                    n
                    for n in tree.body
                    if isinstance(n, ast.ClassDef) and n.name in selected_classes
                ]
                imports = ast.parse("""from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from contextlib import AbstractContextManager
from collections.abc import Iterator
from enum import Enum, auto
import weakref
import torch
from PIL import Image as PILImage
from typing_extensions import override
""").body
                tree = ast.Module(body=imports + selected, type_ignores=[])
            module = ModuleType(name)
            module.__file__ = str(path)
            module.__package__ = name.rpartition(".")[0]
            sys.modules[name] = module
            exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), module.__dict__)
            return module

        load("sam3.model.data_misc", "model/data_misc.py")
        load(
            "sam3.train.data.sam3_image_dataset",
            "train/data/sam3_image_dataset.py",
            {"InferenceMetadata", "FindQuery", "FindQueryLoaded", "Object", "Image", "Datapoint"},
        )
        load("sam3.train.data.collator", "train/data/collator.py")
        output = load("sam3.model.model_misc", "model/model_misc.py", {"SAM3Output"})
        yield output.SAM3Output
    finally:
        for name in list(sys.modules):
            if name == "sam3" or name.startswith("sam3."):
                del sys.modules[name]
        sys.modules.update(original)
        sys.path[:] = original_path
