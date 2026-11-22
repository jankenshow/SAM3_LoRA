"""Command wrapper for python -m sam3_lora evaluate."""

if __package__:
    from ._bootstrap import run
else:
    from _bootstrap import run

if __name__ == "__main__":
    raise SystemExit(run(("evaluate",)))
