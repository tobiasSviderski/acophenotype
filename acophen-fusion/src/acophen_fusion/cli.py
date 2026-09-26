"""Command-line interface for acophen-fusion.

Mostly useful for the binary path from precomputed per-stream probabilities:

    acophen-fusion score-binary --biomarkers 0.2 --embeddings 0.8 \\
        --emotion 0.5 --vision 0.4
"""
from __future__ import annotations

import argparse
import json
import sys

from ._version import __version__
from .scorer import Scorer


def _cmd_score_binary(args) -> int:
    scorer = Scorer.load(task="binary", variant="stream_only", weights_dir=args.weights_dir)
    probas = {
        "biomarkers": args.biomarkers,
        "embeddings": args.embeddings,
        "emotion": args.emotion,
        "vision": args.vision,
    }
    result = scorer.score(stream_probas=probas)
    json.dump(result.as_dict(), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def _cmd_info(args) -> int:
    from .weights import resolve_weights_dir, TASKS, VARIANTS
    try:
        wdir = resolve_weights_dir(args.weights_dir)
        print(f"weights dir : {wdir}")
    except FileNotFoundError as e:
        print(f"weights dir : NOT FOUND\n  {e}")
    print(f"tasks       : {', '.join(TASKS)}")
    print(f"variants    : {', '.join(VARIANTS)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="acophen-fusion",
        description="Fuse acoustic-phenotyping streams into a prediction (research use only).",
    )
    p.add_argument("--version", action="version", version=f"acophen-fusion {__version__}")
    p.add_argument("--weights-dir", help="directory containing the fusion model folders")
    sub = p.add_subparsers(dest="command", required=True)

    sb = sub.add_parser("score-binary", help="fuse precomputed per-stream P(AD)")
    for s in ("biomarkers", "embeddings", "emotion", "vision"):
        sb.add_argument(f"--{s}", type=float, required=True, help=f"{s} stream P(AD)")
    sb.set_defaults(func=_cmd_score_binary)

    inf = sub.add_parser("info", help="show resolved weights dir and available tasks")
    inf.set_defaults(func=_cmd_info)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # --weights-dir is defined on the top parser; sub-commands read args.weights_dir
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
