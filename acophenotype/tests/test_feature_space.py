"""The feature-space fix: detection, alignment, and reference rebuilding."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from acophenotype.reference import ReferenceData, META_KEY
from acophenotype.build_reference import (
    build_reference, save_reference, scaler_from_features,
)


# --- fixtures ---------------------------------------------------------------
def _raw_frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "temp_syllable_rate": rng.normal(3.8, 0.5, n),
        "pros_f0_variability": rng.normal(14.0, 2.0, n),
        "spec_centroid": rng.normal(1800.0, 200.0, n),
    })


def _standardized_reference_from(raw: pd.DataFrame):
    """Mimic the shipped reference: stats computed on standardized features."""
    scaler = scaler_from_features(raw)
    std = raw.copy()
    for c in std.columns:
        std[c] = (std[c] - scaler[c]["mean"]) / scaler[c]["std"]
    return build_reference(std, space="standardized", scaler=scaler), scaler


# --- build_reference --------------------------------------------------------
def test_build_reference_shape_and_meta():
    raw = _raw_frame()
    stats = build_reference(raw, space="raw")
    assert set(stats) == {"temp_syllable_rate", "pros_f0_variability", "spec_centroid", META_KEY}
    for feat in ("temp_syllable_rate", "pros_f0_variability", "spec_centroid"):
        s = stats[feat]
        assert s["q1"] < s["median"] < s["q3"]
        assert s["min"] <= s["q1"] and s["q3"] <= s["max"]
    assert stats[META_KEY]["space"] == "raw"
    assert stats[META_KEY]["n_controls"] == len(raw)


def test_build_reference_filters_to_controls():
    raw = _raw_frame(n=100)
    labels = pd.Series([0] * 60 + [1] * 40, index=raw.index)
    stats = build_reference(raw, labels=labels, control_value=0, space="raw")
    assert stats[META_KEY]["n_controls"] == 60


def test_build_reference_rejects_unknown_control_value():
    raw = _raw_frame(n=50)
    labels = pd.Series([0] * 50, index=raw.index)
    with pytest.raises(ValueError, match="No rows matched"):
        build_reference(raw, labels=labels, control_value=99)


def test_save_reference_roundtrip(tmp_path):
    stats = build_reference(_raw_frame(), space="raw")
    p = tmp_path / "ref.json"
    save_reference(stats, p)
    assert json.loads(p.read_text())[META_KEY]["space"] == "raw"


# --- mismatch detection -----------------------------------------------------
def test_raw_values_vs_standardized_reference_is_detected():
    raw = _raw_frame()
    std_stats, _ = _standardized_reference_from(raw)
    ref = ReferenceData(biomarker_stats=std_stats)
    assert ref.space == "standardized"
    # a genuine raw patient vector compared to standardized stats -> mismatch
    patient = {"temp_syllable_rate": 3.9, "pros_f0_variability": 13.5, "spec_centroid": 1750.0}
    mismatch, frac = ref.space_mismatch(patient)
    assert mismatch is True
    assert frac > 0.25


def test_matching_spaces_are_not_flagged_as_mismatch():
    raw = _raw_frame()
    ref = ReferenceData(biomarker_stats=build_reference(raw, space="raw"))
    assert ref.space == "raw"
    patient = {"temp_syllable_rate": 3.9, "pros_f0_variability": 13.5, "spec_centroid": 1750.0}
    mismatch, frac = ref.space_mismatch(patient)
    assert mismatch is False
    assert frac == 0.0


# --- alignment (the automatic fix) ------------------------------------------
def test_align_standardizes_raw_vector_when_scaler_present():
    raw = _raw_frame()
    std_stats, scaler = _standardized_reference_from(raw)
    ref = ReferenceData(biomarker_stats=std_stats)
    assert ref.scaler  # scaler travelled with the reference

    patient = {"temp_syllable_rate": 3.8, "pros_f0_variability": 14.0, "spec_centroid": 1800.0}
    aligned, note = ref.align(patient)
    assert note and "standardized" in note
    # values near the cohort mean should map to roughly zero in standardized space
    for v in aligned.values():
        assert abs(v) < 1.0
    # and now the spaces agree
    mismatch, _ = ref.space_mismatch(aligned)
    assert mismatch is False


def test_align_is_passthrough_without_scaler():
    raw = _raw_frame()
    ref = ReferenceData(biomarker_stats=build_reference(raw, space="raw"))
    patient = {"temp_syllable_rate": 3.9}
    aligned, note = ref.align(patient)
    assert aligned == patient and note is None


def test_alignment_makes_flags_sane():
    """The whole point: after alignment a typical speaker shouldn't flag everything."""
    raw = _raw_frame()
    std_stats, _ = _standardized_reference_from(raw)
    ref = ReferenceData(biomarker_stats=std_stats)
    typical = {c: float(raw[c].median()) for c in raw.columns}

    before = ref.flags(typical)              # raw vs standardized -> everything flags
    aligned, _ = ref.align(typical)
    after = ref.flags(aligned)               # aligned -> a median speaker flags nothing

    assert len(before) == len(typical)
    assert after == []
