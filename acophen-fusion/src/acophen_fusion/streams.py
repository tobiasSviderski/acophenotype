"""Stream identity and the feature-name contract shared across the suite.

The fusion models consume a *named* feature matrix whose columns are
``"{stream}__{feature}"``. ``voxmarkers`` supplies the ``biomarkers`` stream; the deep
streams (``acophen-streams``) supply the others. Fusion never recomputes features -- it
only aligns incoming named vectors to the frozen column order each model was trained on.
"""
from __future__ import annotations

import pandas as pd

#: Canonical stream order.
STREAMS = ["biomarkers", "embeddings", "emotion", "vision"]

#: Demographic columns used by the ``with_demos`` model variants, in order.
DEMOGRAPHIC_COLUMNS = ["sex", "age", "educ"]

COLUMN_SEP = "__"


def prefixed(stream: str, feature: str) -> str:
    return f"{stream}{COLUMN_SEP}{feature}"


def stream_of(column: str) -> str | None:
    """Return the stream prefix of a ``'{stream}__{feature}'`` column, or None."""
    if COLUMN_SEP in column:
        return column.split(COLUMN_SEP, 1)[0]
    return None


def build_feature_row(streams: dict, demographics: dict | None = None) -> pd.DataFrame:
    """Assemble a single-row DataFrame of ``'{stream}__{feature}'`` columns.

    Parameters
    ----------
    streams : dict
        Maps stream name -> a named feature vector (pandas Series, dict, or 1-D array
        with a matching ``schema``). The vector's index/keys are the unprefixed feature
        names for that stream.
    demographics : dict, optional
        ``{"sex": ..., "age": ..., "educ": ...}`` for the ``with_demos`` variants.
    """
    data: dict[str, float] = {}
    for stream, vec in streams.items():
        if isinstance(vec, pd.Series):
            items = vec.items()
        elif isinstance(vec, dict):
            items = vec.items()
        else:
            raise TypeError(
                f"Stream {stream!r} must be a pandas Series or dict of named features, "
                f"got {type(vec).__name__}. (Fusion aligns by feature name.)"
            )
        for feat, val in items:
            data[prefixed(stream, str(feat))] = val

    if demographics:
        for k, v in demographics.items():
            data[k] = v

    return pd.DataFrame([data])
