"""Command wrapper for python -m sam3_lora infer."""

if __package__:
    from ._bootstrap import run
else:
    from _bootstrap import run

if __name__ == "__main__":
    raise SystemExit(run(("infer",)))

# Existing Python integrations may use the canonical session module directly.
from sam3_lora.inference.session import SAM3LoRAInference

__all__ = ["SAM3LoRAInference"]
