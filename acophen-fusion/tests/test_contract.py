"""Tests that need no model weights: the API contract and input validation."""
from __future__ import annotations

import pandas as pd
import pytest

from acophen_fusion import Scorer, STREAMS, DEMOGRAPHIC_COLUMNS, TASKS, VARIANTS
from acophen_fusion.streams import build_feature_row, prefixed, stream_of


def test_streams_and_variants_defined():
    assert STREAMS == ["biomarkers", "embeddings", "emotion", "vision"]
    assert DEMOGRAPHIC_COLUMNS == ["sex", "age", "educ"]
    assert set(TASKS) == {"binary", "multiclass", "regression"}
    assert set(VARIANTS) == {"stream_only", "with_demos"}


def test_prefix_helpers():
    assert prefixed("biomarkers", "art_fcr") == "biomarkers__art_fcr"
    assert stream_of("vision__feat_10") == "vision"
    assert stream_of("age") is None


def test_build_feature_row_prefixes_and_merges_demographics():
    streams = {
        "biomarkers": pd.Series({"art_fcr": 1.0, "phon_hnr": 2.0}),
        "emotion": {"e0": 0.5},
    }
    row = build_feature_row(streams, demographics={"sex": 1, "age": 70, "educ": 12})
    assert row.shape[0] == 1
    assert "biomarkers__art_fcr" in row.columns
    assert "emotion__e0" in row.columns
    assert row["age"].iloc[0] == 70


def test_build_feature_row_rejects_unnamed_vector():
    with pytest.raises(TypeError):
        build_feature_row({"biomarkers": [1, 2, 3]})


def test_load_rejects_unknown_task_and_variant():
    with pytest.raises(ValueError):
        Scorer.load(task="nope")
    with pytest.raises(ValueError):
        Scorer.load(task="binary", variant="nope")


def test_resolve_error_is_actionable(monkeypatch, tmp_path):
    # Point discovery at an empty dir with no models and no env var.
    monkeypatch.delenv("ACOPHEN_FUSION_WEIGHTS", raising=False)
    monkeypatch.chdir(tmp_path)
    from acophen_fusion.weights import resolve_weights_dir
    with pytest.raises(FileNotFoundError) as ei:
        resolve_weights_dir()
    assert "ACOPHEN_FUSION_WEIGHTS" in str(ei.value)
