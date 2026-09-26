"""Multiclass SGL fusion against the real bundle (skipped if weights unavailable)."""
from __future__ import annotations

from acophen_fusion import Scorer
from conftest import synthetic_streams_from_columns


def test_multiclass_scores_and_normalizes(weights_dir):
    scorer = Scorer.load("multiclass", "stream_only", weights_dir=weights_dir)
    streams = synthetic_streams_from_columns(scorer.expected_columns)
    r = scorer.score(streams=streams)

    assert r.task == "multiclass"
    assert r.classes and r.prediction in r.classes
    # class probabilities sum to ~1
    assert abs(sum(r.probabilities.values()) - 1.0) < 1e-6
    # predicted class is the argmax
    assert r.probabilities[r.prediction] == max(r.probabilities.values())
    # per-stream shares sum to ~1
    assert abs(sum(r.per_stream.values()) - 1.0) < 1e-6


def test_multiclass_expected_columns_are_prefixed(weights_dir):
    scorer = Scorer.load("multiclass", "stream_only", weights_dir=weights_dir)
    cols = scorer.expected_columns
    assert any(c.startswith("biomarkers__") for c in cols)
    assert any(c.startswith("vision__") for c in cols)
    assert len(cols) > 20000  # concatenated deep-stream feature space
