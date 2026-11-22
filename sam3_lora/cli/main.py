"""Lightweight CLI entry point and user-facing errors."""

import json
import logging

from .commands import dispatch
from .parser import build_parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        result = dispatch(args)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        if args.verbose:
            raise
        parser.exit(2, f"error: {error}\n")
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0
