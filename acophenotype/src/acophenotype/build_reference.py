"""Build reference-cohort statistics from your own control data.

This is the proper fix for the feature-space mismatch: regenerate the reference in the
same space you actually extract in (raw ``voxmarkers`` output), instead of comparing raw
values against standardized statistics.

Usage (Python)::

    import pandas as pd
    from acophenotype.build_reference import build_reference, save_reference

    features = pd.read_parquet("streams/biomarkers.parquet").set_index("recording_name")
    labels   = pd.read_parquet("metadata.parquet").set_index("recording_name")["target"]
    stats = build_reference(features, labels=labels, control_value=0, space="raw")
    save_reference(stats, "reference/biomarker_cn_stats.json")

Usage (CLI)::

    acophenotype build-reference \\
        --features streams/biomarkers.parquet \\
        --labels metadata.parquet --label-column target --control-value 0 \\
        --space raw -o biomarker_cn_stats.json

If your features are standardized but you also have the raw scaler, pass
``scaler={"feat": {"mean": …, "std": …}}`` so the packages can align raw patient vectors
automatically at scoring time.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .reference import META_KEY

_STAT_KEYS = ("median", "q1", "q3", "mean", "std", "min", "max")


def build_reference(features, labels=None, control_value=None, space: str = "raw",
                    scaler: dict | None = None, min_samples: int = 10) -> dict:
    """Compute per-feature control statistics.

    Parameters
    ----------
    features : pandas.DataFrame
        One row per recording, one column per feature (numeric).
    labels : pandas.Series, optional
        Aligned labels. If given with ``control_value``, only control rows are used.
    control_value : optional
        The label value identifying controls (e.g. ``0`` or ``"CN"``).
    space : {"raw", "standardized"}
        Declares the space these statistics live in. Recorded in ``_meta``.
    scaler : dict, optional
        ``{feature: {"mean": …, "std": …}}`` mapping *raw* values into this space.
        Only meaningful when ``space="standardized"``.
    min_samples : int
        Warn-threshold: features with fewer finite samples than this are skipped.

    Returns
    -------
    dict
        ``{feature: {median,q1,q3,mean,std,min,max}, "_meta": {...}}``
    """
    import pandas as pd

    if not isinstance(features, pd.DataFrame):
        raise TypeError("`features` must be a pandas DataFrame (rows=recordings).")
    if space not in ("raw", "standardized"):
        raise ValueError('space must be "raw" or "standardized"')

    df = features.select_dtypes(include=[np.number]).copy()

    if labels is not None and control_value is not None:
        labels = labels.reindex(df.index)
        mask = labels == control_value
        if mask.sum() == 0:
            raise ValueError(
                f"No rows matched control_value={control_value!r}. "
                f"Observed labels: {sorted(pd.unique(labels.dropna()))[:10]}"
            )
        df = df[mask.fillna(False)]

    stats: dict = {}
    skipped = []
    for col in df.columns:
        vals = df[col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) < min_samples:
            skipped.append(col)
            continue
        stats[col] = {
            "median": float(np.median(vals)),
            "q1": float(np.percentile(vals, 25)),
            "q3": float(np.percentile(vals, 75)),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    meta = {
        "space": space,
        "n_controls": int(len(df)),
        "n_features": len(stats),
        "built_by": "acophenotype.build_reference",
    }
    if skipped:
        meta["skipped_features"] = skipped
    if scaler:
        meta["scaler"] = {k: {"mean": float(v["mean"]), "std": float(v["std"])}
                          for k, v in scaler.items() if v.get("std")}
    stats[META_KEY] = meta
    return stats


def scaler_from_features(features) -> dict:
    """Derive a raw→standardized scaler (per-feature mean/std) from a raw feature frame.

    Use this when your models were trained on standardized features: build the reference in
    standardized space and attach this scaler so raw patient vectors can be aligned.
    """
    import pandas as pd

    if not isinstance(features, pd.DataFrame):
        raise TypeError("`features` must be a pandas DataFrame.")
    df = features.select_dtypes(include=[np.number])
    return {
        col: {"mean": float(df[col].mean()), "std": float(df[col].std(ddof=0))}
        for col in df.columns
        if np.isfinite(df[col].std(ddof=0)) and df[col].std(ddof=0) > 0
    }


def save_reference(stats: dict, path, indent: int = 2):
    Path(path).write_text(json.dumps(stats, indent=indent), encoding="utf-8")
    return path
