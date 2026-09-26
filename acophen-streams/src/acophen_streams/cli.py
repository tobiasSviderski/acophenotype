"""CLI: list streams / inspect schemas / extract one recording's stream features."""
from __future__ import annotations

import argparse
import sys

from ._version import __version__
from .registry import available, get_stream, schema_for


def _cmd_list(args) -> int:
    for name in available():
        s = schema_for(name)
        print(f"{name:8}  dim={s.dim:<6} schema v{s.version}")
    return 0


def _cmd_schema(args) -> int:
    s = schema_for(args.stream)
    print(f"# {args.stream} schema v{s.version} ({s.dim} features)")
    for c in s.columns:
        print(c)
    return 0


def _cmd_extract(args) -> int:
    stream = get_stream(args.stream)
    vec = stream.extract(args.wav)
    if args.output:
        vec.to_frame().T.to_csv(args.output, index=False)
        print(f"Wrote {len(vec)} features -> {args.output}")
    else:
        vec.to_csv(sys.stdout)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="acophen-streams",
        description="Deep acoustic feature streams (research use only).",
    )
    p.add_argument("--version", action="version", version=f"acophen-streams {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list available streams and their dimensions").set_defaults(func=_cmd_list)

    sc = sub.add_parser("schema", help="print a stream's frozen column list")
    sc.add_argument("stream", choices=available())
    sc.set_defaults(func=_cmd_schema)

    ex = sub.add_parser("extract", help="extract one stream's features from a WAV (needs the extra)")
    ex.add_argument("stream", choices=available())
    ex.add_argument("wav")
    ex.add_argument("-o", "--output", help="output CSV (default: stdout)")
    ex.set_defaults(func=_cmd_extract)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
