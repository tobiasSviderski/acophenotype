"""Regression FC2FS fusion against the real pipeline (skipped if weights unavailable).

Also exercises the unpickle compat shim: loading the FC2FS pipeline requires
``run_3d_fusion_fc2fs.CorrelationSelector`` to resolve.
"""
from __future__ import annotations

import numpy as np

from acophen_fusion import Scorer
from conftest import synthetic_streams_from_columns


def test_regression_predicts_scalar(weights_dir):
    scorer = Scorer.load("regression", "stream_only", weights_dir=weights_dir)
    streams = synthetic_streams_from_columns(scorer.expected_columns)
    r = scorer.score(streams=streams)

    assert r.task == "regression"
    assert r.probability is None
    assert isinstance(r.prediction, float)
    assert np.isfinite(r.prediction)
    # per-stream importance shares sum to ~1 (some streams may be 0)
    assert abs(sum(r.per_stream.values()) - 1.0) < 1e-6


def test_regression_reference_columns_recovered(weights_dir):
    scorer = Scorer.load("regression", "stream_only", weights_dir=weights_dir)
    cols = scorer.expected_columns
    # the FC2FS pipeline retained its full training column order
    assert len(cols) > 20000
    assert any(c.startswith("biomarkers__") for c in cols)
