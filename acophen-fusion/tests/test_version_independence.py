"""The version-independent transforms must reproduce scikit-learn exactly.

These compare our direct-from-fitted-arrays maths against sklearn's own methods on the
REAL frozen models. They are the guarantee that removing the dependency on sklearn
internals did not change a single prediction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from acophen_fusion import models as M
from acophen_fusion.weights import artifact_paths


def test_standard_scale_matches_sklearn_on_stacking_scaler(weights_dir):
    _, scaler, order = M.load_stacking(artifact_paths(weights_dir, "binary", "stream_only"))
    X = np.random.default_rng(0).uniform(size=(50, len(order)))
    np.testing.assert_allclose(M.standard_scale(scaler, X), scaler.transform(X), rtol=1e-12)


def test_logistic_proba_matches_sklearn(weights_dir):
    stacker, scaler, order = M.load_stacking(artifact_paths(weights_dir, "binary", "stream_only"))
    Xs = scaler.transform(np.random.default_rng(1).uniform(size=(50, len(order))))
    np.testing.assert_allclose(M.logistic_proba(stacker, Xs),
                               stacker.predict_proba(Xs)[:, 1], rtol=1e-10)


@pytest.mark.parametrize("variant", ["stream_only", "with_demos"])
def test_sgl_imputer_and_scaler_match_sklearn(weights_dir, variant):
    bundle = M.load_sgl(artifact_paths(weights_dir, "multiclass", variant))
    p = len(bundle["feature_columns"])
    rng = np.random.default_rng(2)
    X = rng.normal(size=(5, p))
    X[rng.uniform(size=X.shape) < 0.3] = np.nan          # plenty of missing values
    ours = M.standard_scale(bundle["scaler"], M.mean_impute(bundle["imputer"], X))
    ref = bundle["scaler"].transform(bundle["imputer"].transform(X))
    np.testing.assert_allclose(ours, ref, rtol=1e-10, atol=1e-10)


def test_predict_sgl_unchanged(weights_dir):
    """End to end: the SGL probabilities equal the original sklearn-based computation."""
    bundle = M.load_sgl(artifact_paths(weights_dir, "multiclass", "stream_only"))
    cols = bundle["feature_columns"]
    X = pd.DataFrame(np.random.default_rng(3).normal(size=(3, len(cols))), columns=cols)
    X.iloc[:, ::7] = np.nan

    proba, _, _ = M.predict_sgl(bundle, X)

    Xs = bundle["scaler"].transform(bundle["imputer"].transform(X.values))
    z = Xs @ np.asarray(bundle["weights"]).T + np.asarray(bundle["intercepts"])
    ref = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
    ref = ref / ref.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(proba, ref, rtol=1e-10)
