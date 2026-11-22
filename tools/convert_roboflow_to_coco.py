"""Command wrapper for python -m sam3_lora prepare roboflow."""

if __package__:
    from ._bootstrap import run
else:
    from _bootstrap import run

if __name__ == "__main__":
    raise SystemExit(run(("prepare", "roboflow")))
