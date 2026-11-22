"""Resolve the source package for direct script execution without installation."""

import sys
from collections.abc import Sequence
from pathlib import Path


def run(command: Sequence[str], argv: Sequence[str] | None = None) -> int:
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from sam3_lora.cli.main import main

    return main([*command, *(sys.argv[1:] if argv is None else argv)])
