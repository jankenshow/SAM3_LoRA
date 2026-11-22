"""Dataset-relative names, allowing intentional image symlinks."""

from pathlib import Path


def image_path(directory, filename):
    relative = Path(filename)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Image path escapes dataset: {filename}")
    return Path(directory) / relative
