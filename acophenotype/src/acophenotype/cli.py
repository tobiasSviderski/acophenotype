"""CLI: analyze one recording end-to-end and write an HTML (and optional JSON) report."""
from __future__ import annotations

import argparse

from ._version import __version__
from .profile import AcousticProfile


def _cmd_analyze(args) -> int:
    demographics = None
    if args.sex is not None and args.age is not None and args.educ is not None:
        demographics = {"sex": args.sex, "age": args.age, "educ": args.educ}

    profile = AcousticProfile.from_audio(
        args.wav, task=args.task, denoise=args.denoise,
        streams="all" if args.streams is None else args.streams,
        demographics=demographics,
        weights_dir=args.weights_dir, reference_dir=args.reference_dir,
    )
    profile.to_html(args.output, spectrogram=args.spectrogram)
    print(f"Report written to {args.output}  ({profile!r})")
    if args.json:
        profile.to_json(args.json)
        print(f"JSON written to {args.json}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="acophenotype",
        description="End-to-end acoustic phenotyping report (research use only).",
    )
    p.add_argument("--version", action="version", version=f"acophenotype {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="analyze a WAV and write an HTML report")
    a.add_argument("wav")
    a.add_argument("-o", "--output", default="report.html", help="output HTML path")
    a.add_argument("--json", help="also write the profile JSON here")
    a.add_argument("--task", default="binary", choices=["binary", "multiclass", "regression"])
    a.add_argument("--streams", nargs="+", help="subset of streams (default: all)")
    a.add_argument("--denoise", action="store_true", help="denoise with MetricGAN+ first")
    a.add_argument("--spectrogram", action="store_true", help="embed a log-mel spectrogram")
    a.add_argument("--weights-dir", help="fusion model weights directory")
    a.add_argument("--reference-dir", help="reference-cohort directory")
    a.add_argument("--sex", type=int, help="demographics: sex (for with_demos)")
    a.add_argument("--age", type=int, help="demographics: age")
    a.add_argument("--educ", type=int, help="demographics: years of education")
    a.set_defaults(func=_cmd_analyze)

    b = sub.add_parser("build-reference",
                       help="build reference-cohort stats from your own control data")
    b.add_argument("--features", required=True,
                   help="parquet/CSV of features (rows=recordings, cols=features)")
    b.add_argument("--index-column", default="recording_name")
    b.add_argument("--labels", help="parquet/CSV containing the label column")
    b.add_argument("--label-column", default="target")
    b.add_argument("--control-value", default=None,
                   help="label value identifying controls (e.g. 0 or CN)")
    b.add_argument("--space", default="raw", choices=["raw", "standardized"])
    b.add_argument("--emit-scaler", action="store_true",
                   help="also derive a raw->standardized scaler from these features")
    b.add_argument("-o", "--output", default="biomarker_cn_stats.json")
    b.set_defaults(func=_cmd_build_reference)

    return p


def _read_table(path, index_column=None):
    import pandas as pd
    path = str(path)
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    if index_column and index_column in df.columns:
        df = df.set_index(index_column)
    return df


def _cmd_build_reference(args) -> int:
    from .build_reference import build_reference, save_reference, scaler_from_features

    features = _read_table(args.features, args.index_column)
    labels = None
    control_value = args.control_value
    if args.labels:
        ldf = _read_table(args.labels, args.index_column)
        if args.label_column not in ldf.columns:
            raise SystemExit(f"Label column {args.label_column!r} not in {args.labels}")
        labels = ldf[args.label_column]
        if control_value is not None:
            # coerce "0"/"1" to numeric when the labels are numeric
            try:
                control_value = type(labels.dropna().iloc[0])(control_value)
            except (ValueError, TypeError, IndexError):
                pass

    scaler = scaler_from_features(features) if args.emit_scaler else None
    stats = build_reference(features, labels=labels, control_value=control_value,
                            space=args.space, scaler=scaler)
    save_reference(stats, args.output)
    meta = stats["_meta"]
    print(f"Wrote {args.output}: {meta['n_features']} features from "
          f"{meta['n_controls']} controls (space={meta['space']})")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
