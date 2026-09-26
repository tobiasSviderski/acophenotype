"""Raw voxmarkers biomarkers must be z-scored with the Pitt scaler before scoring.

Every fusion model was trained on biomarkers z-scored over the Pitt cohort
(biomarkers_stream/2_pre_processing.py). Feeding raw values (e.g. spectral centroid
~1000 Hz into a model trained on values in roughly [-2, 4]) silently corrupts predictions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from acophen_fusion import Scorer
from acophen_fusion import models as M
from conftest import synthetic_streams_from_columns


def test_scaler_is_loaded_with_the_models(weights_dir):
    scorer = Scorer.load("multiclass", "stream_only", weights_dir=weights_dir)
    scaler = scorer._artifacts["biomarker_scaler"]
    assert scaler, "biomarker_scaler.json should sit next to the model folders"
    assert "temp_syllable_rate" in scaler and scaler["temp_syllable_rate"]["std"] > 0


def test_standardize_maps_cohort_mean_to_zero(weights_dir):
    scaler = M.load_biomarker_scaler(weights_dir)
    raw = pd.Series({k: v["mean"] for k, v in scaler.items()})
    z = M.standardize_biomarkers(raw, scaler)
    np.testing.assert_allclose(z.to_numpy(), 0.0, atol=1e-12)


def test_standardize_leaves_unknown_features_alone(weights_dir):
    scaler = M.load_biomarker_scaler(weights_dir)
    z = M.standardize_biomarkers(pd.Series({"not_a_feature": 7.0}), scaler)
    assert z["not_a_feature"] == 7.0


def test_raw_and_prestandardized_inputs_give_same_prediction(weights_dir):
    """Scoring raw biomarkers == scoring the same biomarkers already z-scored."""
    scorer = Scorer.load("multiclass", "stream_only", weights_dir=weights_dir)
    scaler = scorer._artifacts["biomarker_scaler"]
    streams = synthetic_streams_from_columns(scorer.expected_columns, seed=5)

    # build a plausible RAW biomarker vector: mean + noise * std
    rng = np.random.default_rng(6)
    raw = pd.Series({k: v["mean"] + rng.normal() * v["std"] for k, v in scaler.items()})
    z = M.standardize_biomarkers(raw, scaler)

    a = scorer.score(streams={**streams, "biomarkers": raw})
    b = scorer.score(streams={**streams, "biomarkers": z}, biomarkers_standardized=True)
    assert a.prediction == b.prediction
    for cls in a.probabilities:
        assert abs(a.probabilities[cls] - b.probabilities[cls]) < 1e-9


def test_missing_scaler_warns(weights_dir):
    scorer = Scorer.load("multiclass", "stream_only", weights_dir=weights_dir)
    scorer._artifacts["biomarker_scaler"] = None
    streams = synthetic_streams_from_columns(scorer.expected_columns, seed=7)
    with pytest.warns(RuntimeWarning, match="biomarker_scaler"):
        scorer.score(streams=streams)
