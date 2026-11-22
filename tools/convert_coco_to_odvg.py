"""Command wrapper for python -m sam3_lora prepare coco-to-odvg."""

if __package__:
    from ._bootstrap import run
else:
    from _bootstrap import run

if __name__ == "__main__":
    raise SystemExit(run(("prepare", "coco-to-odvg")))
