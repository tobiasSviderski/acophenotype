"""End-to-end extraction tests on a synthetic waveform (no external data needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from voxmarkers import extract, FEATURE_SCHEMA
from voxmarkers.extract import extract_batch

SR = 16000


def test_extract_from_array_returns_full_contract(synthetic_wave):
    s = extract(synthetic_wave, sr=SR)
    assert isinstance(s, pd.Series)
    assert list(s.index) == FEATURE_SCHEMA.columns
    assert len(s) == 76


def test_output_is_finite_and_numeric(synthetic_wave):
    s = extract(synthetic_wave, sr=SR)
    assert s.dtype == float
    assert np.isfinite(s.to_numpy()).all()  # NaN/inf are cleaned to 0.0


PHONATION = ["phon_jitter_local", "phon_jitter_rap", "phon_shimmer_local_db",
             "phon_shimmer_apq3", "phon_hnr", "phon_cpp"]


def test_phonation_features_are_actually_computed(synthetic_wave):
    """Regression: in the thesis script one invalid Praat call zeroed ALL phonation
    features for every recording. On a voiced signal they must be non-zero."""
    s = extract(synthetic_wave, sr=SR, tiers="core")
    zeros = [k for k in PHONATION if s[k] == 0.0]
    assert not zeros, f"phonation features silently zero: {zeros}"


def test_hnr_excludes_undefined_frames(synthetic_wave):
    """HNR must not average in Praat's -200 dB 'undefined' markers."""
    s = extract(synthetic_wave, sr=SR, tiers="core")
    assert s["phon_hnr"] > -50, f"HNR looks polluted by -200 markers: {s['phon_hnr']}"


def test_tier_subset_returns_only_that_tier(synthetic_wave):
    s = extract(synthetic_wave, sr=SR, tiers="core")
    assert list(s.index) == FEATURE_SCHEMA.columns_for(["core"])
    assert len(s) == 53


def test_multiple_tiers_are_canonically_ordered(synthetic_wave):
    s = extract(synthetic_wave, sr=SR, tiers=["aerodynamics", "core"])
    assert list(s.index) == FEATURE_SCHEMA.columns_for(["core", "aerodynamics"])


def test_return_quality(synthetic_wave):
    s, report = extract(synthetic_wave, sr=SR, return_quality=True)
    assert report.duration_s > 5.0
    assert report.ok is True
    assert report.mean_rms > 0


def test_silent_input_flagged():
    silent = np.zeros(SR * 3, dtype=np.float32)
    s, report = extract(silent, sr=SR, return_quality=True)
    assert report.ok is False
    assert any("silent" in w for w in report.warnings)


def test_resampling_path(synthetic_wave):
    # Feed at a different sample rate; should resample internally and still work.
    import librosa
    up = librosa.resample(synthetic_wave.astype(float), orig_sr=SR, target_sr=22050)
    s = extract(up, sr=22050)
    assert len(s) == 76


def test_extract_from_wav_path(synthetic_wav_path):
    s = extract(synthetic_wav_path)
    assert len(s) == 76
    assert s.name == "synthetic"


def test_batch(synthetic_wav_path):
    df = extract_batch([synthetic_wav_path, synthetic_wav_path], tiers="core")
    assert df.shape == (2, 53)
    assert df.index.name == "recording_name"
    assert list(df.columns) == FEATURE_SCHEMA.columns_for(["core"])
