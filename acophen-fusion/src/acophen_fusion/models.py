"""Low-level model loading and prediction.

The prediction routines are faithful ports of ``evaluation/common/zeroshot.py`` from the
thesis: name-based column alignment, then the frozen model. Column alignment is the
correctness invariant -- an incoming vector is reindexed onto the exact training column
order; extra features are dropped, missing ones imputed.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import compat  # noqa: F401  (registers unpickle shims on import)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _load(path):
    import joblib
    return joblib.load(path)


# --------------------------------------------------------------------------------------
# Version-independent transforms.
#
# The frozen models were pickled with scikit-learn 1.7.2. Calling ``.transform()`` /
# ``.predict_proba()`` on those objects under a newer scikit-learn runs *new* code against
# *old* pickled state, which can raise (missing private attributes) or silently change
# results. The fitted maths of these estimators is trivial and stable, so we apply it
# directly from their public fitted arrays. Output is identical to sklearn's own methods
# (verified in tests), but no longer depends on sklearn internals.
# --------------------------------------------------------------------------------------
def standard_scale(scaler, X):
    """``StandardScaler.transform`` from the fitted ``mean_`` / ``scale_`` arrays."""
    X = np.asarray(X, dtype=float)
    mean = getattr(scaler, "mean_", None)
    scale = getattr(scaler, "scale_", None)
    if getattr(scaler, "with_mean", True) and mean is not None:
        X = X - np.asarray(mean, dtype=float)
    if getattr(scaler, "with_std", True) and scale is not None:
        X = X / np.asarray(scale, dtype=float)
    return X


def mean_impute(imputer, X):
    """``SimpleImputer(strategy='mean')`` from the fitted ``statistics_`` array."""
    X = np.array(X, dtype=float, copy=True)
    strategy = getattr(imputer, "strategy", "mean")
    if strategy not in ("mean", "median", "most_frequent", "constant"):
        return imputer.transform(X)  # unknown strategy: fall back to sklearn
    stats = np.asarray(imputer.statistics_, dtype=float)
    mask = np.isnan(X)
    if mask.any():
        X[mask] = np.broadcast_to(stats, X.shape)[mask]
    return X


def logistic_proba(model, Xs):
    """Positive-class probability of a fitted binary ``LogisticRegression``."""
    coef = np.asarray(model.coef_, dtype=float).ravel()
    intercept = float(np.ravel(model.intercept_)[0])
    return _sigmoid(np.asarray(Xs, dtype=float) @ coef + intercept)


# ------------------------------------------------------------- biomarker standardization
BIOMARKER_SCALER_FILE = "biomarker_scaler.json"


def load_biomarker_scaler(weights_dir):
    """Load the Pitt z-score parameters for the biomarker stream, or None if absent.

    Every fusion model was trained on biomarkers z-scored over the Pitt cohort, while
    ``voxmarkers`` emits raw values. This scaler bridges the two. Looked up next to the
    model folders, then fetched from the Hub.
    """
    from pathlib import Path
    candidates = [Path(weights_dir) / BIOMARKER_SCALER_FILE] if weights_dir else []
    for p in candidates:
        if p.exists():
            with open(p) as f:
                return json.load(f)["features"]
    try:
        from .hub import fetch_file
        p = fetch_file(BIOMARKER_SCALER_FILE)
        if p is not None and Path(p).exists():
            with open(p) as f:
                return json.load(f)["features"]
    except Exception:
        pass
    return None


def standardize_biomarkers(vec, scaler: dict):
    """Apply ``(x - mean) / std`` by feature name. Unknown features pass through."""
    if isinstance(vec, dict):
        vec = pd.Series(vec, dtype=float)
    out = vec.astype(float).copy()
    for name in out.index:
        s = scaler.get(name)
        if s and s.get("std"):
            out[name] = (out[name] - s["mean"]) / s["std"]
    return out


# --------------------------------------------------------------------------- binary
def load_stacking(paths: dict):
    """Load the binary stacking meta-model, its scaler, and feature order."""
    stacker = _load(paths["model"])
    scaler = _load(paths["scaler"])
    with open(paths["config"]) as f:
        config = json.load(f)
    return stacker, scaler, config["features"]


def predict_stacking(stacker, scaler, feature_order, stream_probas: dict, demographics=None):
    """Fuse per-stream P(AD) into one probability. Port of zeroshot.predict_stacking.

    ``stream_probas`` maps stream -> scalar (or length-1 array) probability.
    ``feature_order`` is the config ``features`` list (proba columns [+ demo columns]).
    """
    row = {f"{s}_proba": float(np.ravel(v)[0]) for s, v in stream_probas.items()}
    if demographics:
        row.update({k: float(v) for k, v in demographics.items()})
    X = pd.DataFrame([row]).reindex(columns=feature_order)
    Xs = standard_scale(scaler, X.values)
    if hasattr(stacker, "coef_") and hasattr(stacker, "intercept_") and \
            len(getattr(stacker, "classes_", [0, 1])) == 2:
        proba = float(logistic_proba(stacker, Xs)[0])
    elif hasattr(stacker, "predict_proba"):
        proba = float(stacker.predict_proba(Xs)[:, 1][0])
    else:
        proba = float(stacker.predict(Xs)[0])
    return proba, Xs[0]


def stacking_contributions(stacker, feature_order, Xs_row, streams):
    """Signed per-stream contribution to the logit = coef * scaled input.

    The stacker is a linear model, so contributions are exact. Demographic columns are
    grouped under ``"demographics"``.
    """
    coef = np.ravel(stacker.coef_)
    contrib = coef * np.asarray(Xs_row, dtype=float)
    out: dict[str, float] = {}
    for name, c in zip(feature_order, contrib):
        if name.endswith("_proba"):
            key = name[: -len("_proba")]
        elif name in ("sex", "age", "educ"):
            key = "demographics"
        else:
            key = name
        out[key] = out.get(key, 0.0) + float(c)
    return out


# ----------------------------------------------------------------------- multiclass
def load_sgl(paths: dict):
    """Load the SGL one-vs-rest multiclass bundle (dict of arrays + preprocessors)."""
    return _load(paths["bundle"])


def predict_sgl(bundle, X: pd.DataFrame):
    """Return (proba [n,K], classes, scaled_X). Port of zeroshot.predict_sgl."""
    cols = bundle["feature_columns"]
    Xr = X.reindex(columns=cols)
    Xi = mean_impute(bundle["imputer"], Xr.values)
    Xs = standard_scale(bundle["scaler"], Xi)
    W = np.asarray(bundle["weights"], dtype=float)     # (K, p)
    b = np.asarray(bundle["intercepts"], dtype=float)  # (K,)
    proba = _sigmoid(Xs @ W.T + b)
    proba = proba / proba.sum(axis=1, keepdims=True)
    return proba, bundle["classes"], Xs


def sgl_contributions(bundle, Xs, class_index, streams):
    """Per-stream contribution to the predicted class's logit.

    For class k, logit_k = sum_j Xs[j] * W[k,j] (+b). We sum the per-column terms by
    the column's stream prefix -> an exact linear attribution for that class.
    """
    cols = bundle["feature_columns"]
    W = np.asarray(bundle["weights"], dtype=float)
    terms = np.asarray(Xs, dtype=float).ravel() * W[class_index]
    out: dict[str, float] = {}
    for name, t in zip(cols, terms):
        key = name.split("__", 1)[0] if "__" in name else "demographics"
        out[key] = out.get(key, 0.0) + float(t)
    return out


# ----------------------------------------------------------------------- regression
def load_fc2fs(paths: dict):
    """Load the FC2FS regression pipeline and recover its training column order."""
    pipe = _load(paths["pipeline"])
    pre = pipe.named_steps.get("preprocessor")
    reference_columns = list(getattr(pre, "feature_names_in_", []))
    return pipe, reference_columns


def predict_fc2fs(pipe, X: pd.DataFrame, reference_columns):
    """Return a scalar prediction. Port of zeroshot.predict_fc2fs."""
    Xr = X.reindex(columns=reference_columns)
    return float(np.asarray(pipe.predict(Xr), dtype=float)[0])


def fc2fs_contributions(pipe, reference_columns, streams):
    """Per-stream attribution from the LGBM's grouped feature importances.

    FC2FS is a non-linear tree model, so there is no exact linear attribution. We group
    the fitted ``feature_importances_`` by stream prefix over the columns that survived
    the correlation filter -- a global (not per-sample) importance share.
    """
    model = pipe.steps[-1][1]
    importances = np.asarray(getattr(model, "feature_importances_", []), dtype=float)
    # Columns surviving FC2FS, in order: apply the selector's drop mask to the
    # preprocessor's output columns (same length/order as reference_columns).
    kept_cols = list(reference_columns)
    fc2fs = pipe.named_steps.get("fc2fs")
    drop_idx = list(getattr(fc2fs, "to_drop_indices_", []))
    if drop_idx:
        keep_mask = np.ones(len(kept_cols), dtype=bool)
        valid = [i for i in drop_idx if i < len(keep_mask)]
        keep_mask[valid] = False
        kept_cols = [c for c, k in zip(kept_cols, keep_mask) if k]

    out: dict[str, float] = {}
    if len(importances) == len(kept_cols):
        for name, imp in zip(kept_cols, importances):
            key = name.split("__", 1)[0] if "__" in name else "demographics"
            out[key] = out.get(key, 0.0) + float(imp)
    return out
