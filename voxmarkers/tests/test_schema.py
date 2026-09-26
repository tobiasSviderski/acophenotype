"""The most important tests: the frozen feature-name contract must not drift.

If any of these fail, downstream zero-shot scoring against previously-trained
models is invalid. A deliberate schema change must bump SCHEMA_VERSION *and*
update the expected values here in the same commit.
"""
from __future__ import annotations

from voxmarkers import FEATURE_SCHEMA
from voxmarkers.schema import TIER_COLUMNS, columns_for

# The exact, ordered contract at schema v1.0. Frozen on purpose.
EXPECTED_CORE = [
    "temp_syllable_rate", "temp_speaking_rate", "temp_pause_rate",
    "temp_phonation_ratio", "temp_speech_pause_ratio",
    "temp_mean_pause_dur", "temp_pause_variability",
    "pros_f0_variability", "pros_f0_range", "pros_f0_slope",
    "pros_loudness_variability", "pros_energy_slope",
    "art_fcr", "art_vai", "art_f1_bw", "art_f2_bw", "art_f2_slope",
    "phon_jitter_local", "phon_jitter_rap",
    "phon_shimmer_local_db", "phon_shimmer_apq3",
    "phon_hnr", "phon_cpp",
    "spec_centroid", "spec_flux", "spec_entropy", "spec_rpde",
] + [f"spec_mfcc{i}_{s}" for i in range(1, 14) for s in ("mean", "std")]

EXPECTED_EXTENSION = [
    "spec_hammarberg", "spec_alpha_ratio",
    "spec_skewness", "spec_kurtosis", "spec_rolloff",
    "comp_dfa", "comp_lzc", "comp_ppe", "comp_gne",
    "rhythm_npvi",
    "qual_vti", "qual_spi", "qual_ftri",
    "trans_wavelet_energy",
]

EXPECTED_AERO = [
    "aero_h1_h2_diff",
    "dyn_mfcc_velocity_mean", "dyn_mfcc_accel_mean",
    "dyn_teager_energy_mean", "dyn_teager_energy_std",
    "art_formant_dispersion",
    "phon_v_uv_ratio", "phon_voice_break_factor",
    "pros_articulation_rate_variability",
]


def test_schema_version_pinned():
    assert FEATURE_SCHEMA.version == "1.0"


def test_tier_columns_exact_and_ordered():
    assert TIER_COLUMNS["core"] == EXPECTED_CORE
    assert TIER_COLUMNS["extension"] == EXPECTED_EXTENSION
    assert TIER_COLUMNS["aerodynamics"] == EXPECTED_AERO


def test_full_contract_is_concatenation_in_canonical_order():
    assert FEATURE_SCHEMA.columns == EXPECTED_CORE + EXPECTED_EXTENSION + EXPECTED_AERO


def test_feature_counts():
    assert len(EXPECTED_CORE) == 53
    assert len(EXPECTED_EXTENSION) == 14
    assert len(EXPECTED_AERO) == 9
    assert len(FEATURE_SCHEMA) == 76


def test_no_duplicate_feature_names():
    cols = FEATURE_SCHEMA.columns
    assert len(cols) == len(set(cols))


def test_columns_for_is_canonical_regardless_of_request_order():
    # request reversed; output must still be canonical
    got = columns_for(["aerodynamics", "core"])
    assert got == EXPECTED_CORE + EXPECTED_AERO


def test_unknown_tier_raises():
    import pytest
    with pytest.raises(ValueError):
        columns_for(["not_a_tier"])
