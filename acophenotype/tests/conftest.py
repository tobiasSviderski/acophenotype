"""Fixtures: a synthetic wav and a synthetic profile data dict (no torch needed)."""
from __future__ import annotations

import numpy as np
import pytest

SR = 16000


@pytest.fixture(scope="session")
def synthetic_wave():
    t = np.linspace(0, 6, 6 * SR, endpoint=False)
    f0 = 120 + 10 * np.sin(2 * np.pi * 0.5 * t)
    ph = 2 * np.pi * np.cumsum(f0) / SR
    y = (np.sin(ph) + 0.4 * np.sin(2 * ph)) * (0.5 * (1 + np.sin(2 * np.pi * 3 * t)))
    for a, b in ((1.5, 2.0), (4.0, 4.4)):
        y[int(a * SR):int(b * SR)] = 0.0
    return (y / (np.max(np.abs(y)) + 1e-9) * 0.9).astype(np.float32)


@pytest.fixture(scope="session")
def synthetic_wav_path(tmp_path_factory, synthetic_wave):
    sf = pytest.importorskip("soundfile")
    p = tmp_path_factory.mktemp("audio") / "syn.wav"
    sf.write(str(p), synthetic_wave, SR)
    return str(p)


@pytest.fixture
def profile_data():
    """A full profile dict with a fusion block, to exercise the report end to end."""
    return {
        "metadata": {"audio_file": "demo.wav", "task": "binary",
                     "duration_seconds": 42.0, "snr_db": 18.3, "quality_flag": "high"},
        "fusion": {
            "task": "binary", "variant": "stream_only", "prediction": "AD",
            "probability": 0.78,
            "per_stream": {"biomarkers": 0.41, "embeddings": 0.24, "emotion": 0.12, "vision": 0.23},
            "attribution_method": "linear stacker coefficients (exact)",
        },
        "biomarkers": {
            "values": {"temp_syllable_rate": 3.9, "phon_hnr": 12.0, "art_fcr": 1.1,
                       "temp_pause_rate": 0.6, "phon_jitter_local": 0.02,
                       "spec_centroid": 1800.0, "comp_dfa": 1.2, "rhythm_npvi": 44.0,
                       "dyn_teager_energy_mean": 0.003},
            "percentiles": {"temp_syllable_rate": 22.0, "phon_hnr": 40.0, "art_fcr": 88.0,
                            "temp_pause_rate": 71.0, "phon_jitter_local": 95.0,
                            "spec_centroid": 55.0},
            "bands": {"temp_syllable_rate": {"median": 4.2, "q1": 3.7, "q3": 4.8},
                      "phon_jitter_local": {"median": 0.01, "q1": 0.007, "q3": 0.015},
                      "spec_centroid": None},           # None band must not crash
            "flags": ["phon_jitter_local"],
            "reference_available": True,
            "space_warning": False,
            "reference_space": "raw",
        },
        "emotion": {
            "table": {
                "neutral": {"mean": 0.31, "mean_z": -1.2, "std": 0.05, "range": 0.2, "cv": 0.16},
                "sad": {"mean": 0.08, "mean_z": 0.6, "std": 0.03, "range": 0.1, "cv": 0.38},
                "angry": {"mean": 0.04, "std": 0.02, "range": 0.08, "cv": 0.5},
            },
            "stats": ["mean", "std", "range", "cv"],
            "trajectory": {
                "times": [1.75 + 0.5 * i for i in range(20)],
                "series": {"neutral": [0.3 + 0.01 * i for i in range(20)],
                           "sad": [0.1] * 20,
                           "angry": [0.05] * 20},
            },
            "aggregates": {"neutral_mean": 0.31, "sad_mean": 0.08, "angry_mean": 0.04},
        },
        "umap": {
            "coordinates": [1.0, 0.5],
            "cn": [[0.1, 0.2], [0.3, -0.1], [-0.2, 0.4]],
            "ad": [[2.0, 1.1], [2.4, 0.9], [1.8, 1.4]],
        },
        "provenance": {
            "generated_utc": "2026-07-20 10:00:00 UTC",
            "python": "3.11.5",
            "platform": "Linux-6.1",
            "packages": {"acophenotype": "0.1.0", "voxmarkers": "0.1.0",
                         "scikit-learn": "1.7.2", "lightgbm": "4.6.0"},
            "schemas": {"voxmarkers_feature_schema": "1.0"},
            "task": "binary",
            "training_dataset": "Pitt Corpus (DementiaBank)",
            "fusion_model": {"task": "binary", "variant": "stream_only"},
            "stream_models": {"biomarkers": "binary_biomarkers_LGBMClassifier"},
            "reference": {"source": "/x/reference", "space": "raw", "n_features": 48,
                          "has_scaler": False, "n_controls": 247},
            "limitations": ["Research use only — not a medical device.",
                            "Trained on English picture-description speech."],
        },
        "streams_present": ["biomarkers", "embeddings", "emotion", "vision"],
        "reference_source": "/x/feature_fusion/reference",
        "notes": ["denoised with MetricGAN+"],
    }
