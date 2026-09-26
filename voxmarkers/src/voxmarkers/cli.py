"""Command-line interface: ``voxmarkers extract`` and ``voxmarkers describe``."""
from __future__ import annotations

import argparse
import sys

from ._version import __version__
from .schema import DEFAULT_TIERS


def _cmd_extract(args) -> int:
    from .extract import extract_batch

    df = extract_batch(args.wav, tiers=tuple(args.tiers), progress=not args.quiet)
    if args.output:
        df.to_csv(args.output)
        if not args.quiet:
            print(f"Wrote {df.shape[0]} rows x {df.shape[1]} features -> {args.output}")
    else:
        df.to_csv(sys.stdout)
    return 0


def _cmd_describe(args) -> int:
    from .provenance import describe, provenance_frame

    if args.feature:
        info = describe(args.feature)
        for k, v in info.items():
            print(f"{k:16}: {v}")
    else:
        df = provenance_frame()
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"Wrote provenance table ({len(df)} features) -> {args.output}")
        else:
            print(df.to_string(index=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="voxmarkers",
        description="Interpretable acoustic biomarkers of speech (research use only).",
    )
    p.add_argument("--version", action="version", version=f"voxmarkers {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    ex = sub.add_parser("extract", help="extract features from one or more WAV files")
    ex.add_argument("wav", nargs="+", help="WAV file path(s)")
    ex.add_argument("-o", "--output", help="output CSV path (default: stdout)")
    ex.add_argument(
        "-t", "--tiers", nargs="+", default=list(DEFAULT_TIERS),
        choices=["core", "extension", "aerodynamics"],
        help="feature tiers to compute (default: all)",
    )
    ex.add_argument("-q", "--quiet", action="store_true", help="suppress progress output")
    ex.set_defaults(func=_cmd_extract)

    de = sub.add_parser("describe", help="show provenance for a feature (or the whole table)")
    de.add_argument("feature", nargs="?", help="feature name; omit to dump the full table")
    de.add_argument("-o", "--output", help="write the full table to CSV")
    de.set_defaults(func=_cmd_describe)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
