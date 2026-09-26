"""Unpickling compatibility shims.

The frozen regression pipeline (``final_regression_fc2fs_*.pkl``) was pickled with a
reference to ``run_3d_fusion_fc2fs.CorrelationSelector`` -- a custom transformer that
lived in a thesis training script. joblib/pickle resolves classes by their *original*
module path, so we vendor the class here and register a stand-in module under that
name **before any model is loaded**. Importing this module (which the package does at
import time) is enough.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CorrelationSelector(BaseEstimator, TransformerMixin):
    """FC2FS: drop features whose absolute correlation exceeds ``threshold``.

    Vendored verbatim from ``feature_fusion/run_3d_fusion_fc2fs.py`` so that fitted
    pipelines unpickle. ``to_drop_indices_`` is restored from the pickle; ``fit`` is
    only needed if someone re-fits.
    """

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self.to_drop_indices_ = []

    def fit(self, X, y=None):
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        X_arr = X_arr.astype(np.float32)
        try:
            with np.errstate(divide="ignore", invalid="ignore"):
                corr_matrix = np.abs(np.corrcoef(X_arr, rowvar=False))
        except MemoryError:
            self.to_drop_indices_ = []
            return self
        upper = np.triu(corr_matrix, k=1)
        to_drop_mask = np.any(upper > self.threshold, axis=0)
        self.to_drop_indices_ = np.where(to_drop_mask)[0]
        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            return X.drop(X.columns[self.to_drop_indices_], axis=1)
        return np.delete(X, self.to_drop_indices_, axis=1)


def install_unpickle_shims() -> None:
    """Register legacy module names so old joblib pickles resolve.

    Idempotent: safe to call more than once.
    """
    if "run_3d_fusion_fc2fs" not in sys.modules:
        shim = types.ModuleType("run_3d_fusion_fc2fs")
        shim.CorrelationSelector = CorrelationSelector
        sys.modules["run_3d_fusion_fc2fs"] = shim


# Register on import.
install_unpickle_shims()
