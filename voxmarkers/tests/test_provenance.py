"""Every feature in the schema must carry provenance, and the metadata must be sane."""
from __future__ import annotations

import pytest

from voxmarkers import describe, provenance_frame, FEATURE_SCHEMA
from voxmarkers.provenance import missing_provenance

VALID_DIRECTIONS = {"increases", "decreases", "altered"}


def test_no_feature_missing_provenance():
    assert missing_provenance() == []


def test_every_schema_column_describable():
    for col in FEATURE_SCHEMA.columns:
        info = describe(col)
        assert info["name"] == col
        assert info["family"]
        assert info["tier"] in {"core", "extension", "aerodynamics"}
        assert info["direction_in_ad"] in VALID_DIRECTIONS
        assert info["description"]
        assert isinstance(info["references"], list) and info["references"]


def test_describe_unknown_feature_raises():
    with pytest.raises(KeyError):
        describe("not_a_real_feature")


def test_provenance_frame_matches_schema():
    df = provenance_frame()
    assert list(df["name"]) == FEATURE_SCHEMA.columns
    assert not df["references"].eq("").any()
