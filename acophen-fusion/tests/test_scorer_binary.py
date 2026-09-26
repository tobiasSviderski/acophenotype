"""Binary fusion against the real stacking model (skipped if weights unavailable)."""
from __future__ import annotations

import pytest

from acophen_fusion import Scorer, STREAMS


def test_binary_from_stream_probas(weights_dir):
    scorer = Scorer.load("binary", "stream_only", weights_dir=weights_dir)
    probas = {"biomarkers": 0.2, "embeddings": 0.8, "emotion": 0.5, "vision": 0.4}
    r = scorer.score(stream_probas=probas)

    assert r.task == "binary"
    assert r.prediction in ("AD", "CN")
    assert 0.0 <= r.probability <= 1.0
    # per-stream shares are non-negative and sum to ~1
    assert set(r.per_stream) >= set(STREAMS)
    assert abs(sum(r.per_stream.values()) - 1.0) < 1e-6
    assert all(v >= 0 for v in r.per_stream.values())


def test_binary_monotonic_in_probas(weights_dir):
    scorer = Scorer.load("binary", "stream_only", weights_dir=weights_dir)
    low = scorer.score(stream_probas={s: 0.05 for s in STREAMS}).probability
    high = scorer.score(stream_probas={s: 0.95 for s in STREAMS}).probability
    # All-high stream probabilities should not fuse to a lower P(AD) than all-low.
    assert high >= low


def test_expected_columns_are_proba_features(weights_dir):
    scorer = Scorer.load("binary", "stream_only", weights_dir=weights_dir)
    assert scorer.expected_columns == [f"{s}_proba" for s in STREAMS]


def test_with_demos_requires_demographics(weights_dir):
    scorer = Scorer.load("binary", "with_demos", weights_dir=weights_dir)
    with pytest.raises(ValueError):
        scorer.score(stream_probas={s: 0.5 for s in STREAMS})  # no demographics
    r = scorer.score(
        stream_probas={s: 0.5 for s in STREAMS},
        demographics={"sex": 1, "age": 70, "educ": 12},
    )
    assert 0.0 <= r.probability <= 1.0
    assert "demographics" in r.per_stream
