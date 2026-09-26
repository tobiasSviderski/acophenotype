"""The frozen output schemas must match the fusion contract exactly."""
from __future__ import annotations

from acophen_streams import SCHEMAS, SCHEMA_VERSION, EMOTION_LABELS
from acophen_streams.pooling import STAT_NAMES


def test_schema_version():
    assert SCHEMA_VERSION == "1.0"
    for s in SCHEMAS.values():
        assert s.version == "1.0"


def test_dims_match_fusion_contract():
    assert SCHEMAS["ssl"].dim == 12288
    assert SCHEMAS["vision"].dim == 14336
    assert SCHEMAS["emotion"].dim == 120


def test_feat_naming_is_zero_padded():
    ssl_cols = SCHEMAS["ssl"].columns
    assert ssl_cols[0] == "feat_0000"
    assert ssl_cols[-1] == "feat_12287"
    assert SCHEMAS["vision"].columns[-1] == "feat_14335"


def test_emotion_columns_are_named_and_complete():
    cols = SCHEMAS["emotion"].columns
    assert len(cols) == len(EMOTION_LABELS) * len(STAT_NAMES) == 120
    # exact set matches {emotion}_{stat}
    expected = {f"{e}_{s}" for e in EMOTION_LABELS for s in STAT_NAMES}
    assert set(cols) == expected
    assert "angry_mean" in cols and "neutral_cv" in cols


def test_no_duplicate_columns():
    for name, s in SCHEMAS.items():
        assert len(s.columns) == len(set(s.columns)), f"{name} has duplicate columns"
