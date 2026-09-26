"""Fixtures. Model-dependent tests are skipped automatically when the fusion weights
are not resolvable (e.g. on CI without the model bundle)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from acophen_fusion.streams import STREAMS
from acophen_fusion.weights import resolve_weights_dir


@pytest.fixture(scope="session")
def weights_dir():
    try:
        return resolve_weights_dir()
    except FileNotFoundError:
        pytest.skip("fusion weights not available (set ACOPHEN_FUSION_WEIGHTS to run)")


def synthetic_streams_from_columns(columns, seed=0):
    """Build a per-stream dict of named Series covering the given prefixed columns.

    ``columns`` are ``'{stream}__{feature}'`` names; demographic columns (no prefix) are
    ignored here and supplied separately.
    """
    rng = np.random.default_rng(seed)
    per_stream: dict[str, dict] = {s: {} for s in STREAMS}
    for col in columns:
        if "__" not in col:
            continue
        s, feat = col.split("__", 1)
        if s in per_stream:
            per_stream[s][feat] = float(rng.normal())
    return {s: pd.Series(d) for s, d in per_stream.items() if d}
