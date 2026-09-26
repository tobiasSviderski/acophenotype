"""AcousticProfile behaviour: percentiles, JSON, and biomarker-only from_audio."""
from __future__ import annotations

import json

from acophenotype import AcousticProfile
from acophenotype.reference import _percentile_from_stats


def test_percentile_interpolation_knots():
    s = {"min": 0.0, "q1": 1.0, "median": 2.0, "q3": 3.0, "max": 4.0}
    assert _percentile_from_stats(2.0, s) == 50.0
    assert _percentile_from_stats(1.0, s) == 25.0
    assert _percentile_from_stats(3.0, s) == 75.0
    # clamped
    assert _percentile_from_stats(-5.0, s) == 0.0
    assert _percentile_from_stats(99.0, s) == 100.0


def test_percentile_falls_back_to_normal_cdf_when_non_monotonic():
    s = {"min": 5, "q1": 5, "median": 5, "q3": 5, "max": 5, "mean": 0.0, "std": 1.0}
    # non-monotonic knots -> normal CDF from mean/std; value at mean -> ~50th pct
    assert abs(_percentile_from_stats(0.0, s) - 50.0) < 1e-6


def test_to_dict_excludes_vectors(profile_data):
    prof = AcousticProfile.from_dict(profile_data)
    d = prof.to_dict()
    assert "_vectors" not in d
    assert d["fusion"]["prediction"] == "AD"


def test_to_json_roundtrips(profile_data, tmp_path):
    prof = AcousticProfile.from_dict(profile_data)
    p = tmp_path / "p.json"
    prof.to_json(p)
    loaded = json.loads(p.read_text())
    assert loaded["metadata"]["task"] == "binary"
    assert loaded["biomarkers"]["flags"] == ["phon_jitter_local"]


def test_from_audio_biomarker_only(synthetic_wave):
    # No torch/streams/weights here -> must produce a biomarker-only profile, not error.
    prof = AcousticProfile.from_audio(synthetic_wave, sr=16000, task="binary",
                                      reference_dir=None)
    assert prof.prediction is None                 # no fusion without deep streams
    assert len(prof.biomarkers) == 76              # full voxmarkers vector present
    assert prof.quality["duration_seconds"] > 5
    # renders a valid biomarker-only report
    html = prof.to_html()
    assert html.startswith("<!DOCTYPE html>")
    assert "Fusion result" not in html
