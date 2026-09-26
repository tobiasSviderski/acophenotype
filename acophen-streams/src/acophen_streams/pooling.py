"""Temporal pooling functions -- pure NumPy, no deep-learning deps.

Ported verbatim (behaviour-preserving) from the thesis flattening scripts so that
packaged output matches the vectors the fusion models were trained on:

- :func:`vlad_pool`             <- ``embedding_stream/src/flattening_vlad.py``
- :func:`temporal_pyramid_pool` <- ``vision_stream/src/flattening_temproal_pyramid.py``
- :func:`statistical_pool`      <- ``emotion_stream/src/flattening_statistical.py``
"""
from __future__ import annotations

import numpy as np

# The 15 statistical functions, in the exact order the emotion stream emits them.
STAT_NAMES = [
    "mean", "std", "max", "min", "mode", "skew", "kurt",
    "median", "q1", "q3", "iqr", "range", "var", "sem", "cv",
]


def vlad_pool(embs: np.ndarray, kmeans) -> np.ndarray:
    """VLAD pooling: residuals to K cluster centroids, flattened and L2-normalized.

    Returns a vector of length ``D * K``. ``kmeans`` is a fitted (MiniBatch)KMeans.
    """
    if embs.ndim != 2 or embs.shape[0] == 0:
        D = embs.shape[1] if embs.ndim == 2 and embs.shape[1] > 0 else 1
        K = getattr(kmeans, "n_clusters", 16)
        return np.zeros((D * K,), dtype=np.float32)

    T, D = embs.shape
    centroids = np.asarray(kmeans.cluster_centers_, dtype=np.float64)
    K = centroids.shape[0]
    # Nearest-centroid assignment computed directly (== KMeans.predict), so a codebook
    # pickled under one scikit-learn version works under any other.
    d2 = ((np.asarray(embs, dtype=np.float64)[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
    cluster_ids = d2.argmin(axis=1)

    vlad = np.zeros((K, D), dtype=np.float32)
    for t in range(T):
        vlad[cluster_ids[t]] += embs[t] - centroids[cluster_ids[t]]

    vlad_flat = vlad.flatten()
    norm = np.linalg.norm(vlad_flat)
    if norm > 1e-6:
        vlad_flat = vlad_flat / norm
    return vlad_flat.astype(np.float32)


def temporal_pyramid_pool(embs: np.ndarray, levels: int = 3) -> np.ndarray:
    """Mean-pool over a temporal pyramid. Vector length ``D * (2**levels - 1)``."""
    if embs.ndim != 2 or embs.shape[0] == 0:
        D = embs.shape[1] if embs.ndim == 2 and embs.shape[1] > 0 else 1
        total_bins = (2 ** levels) - 1
        return np.zeros((D * total_bins,), dtype=np.float32)

    T, D = embs.shape
    pooled = [embs.mean(axis=0)]  # level 0: full sequence
    for level in range(1, levels):
        num_bins = 2 ** level
        bin_size = max(1, T // num_bins)
        for bin_idx in range(num_bins):
            start = bin_idx * bin_size
            end = min((bin_idx + 1) * bin_size, T)
            if start < end:
                pooled.append(embs[start:end].mean(axis=0))
            else:
                pooled.append(np.zeros(D, dtype=np.float32))
    return np.concatenate(pooled, axis=0).astype(np.float32)


def statistical_pool(embs: np.ndarray) -> np.ndarray:
    """Compute the 15 statistics in :data:`STAT_NAMES` per dimension, concatenated.

    Vector length ``D * 15``, ordered dimension-major (all 15 stats of dim 0, then dim 1).
    """
    from scipy import stats

    if embs.ndim != 2 or embs.shape[0] == 0:
        return np.array([], dtype=np.float32)

    T, D = embs.shape
    features = []
    for d in range(D):
        col = embs[:, d]
        mean_val = np.mean(col)
        std_val = np.std(col)
        max_val = np.max(col)
        min_val = np.min(col)
        mode_res = stats.mode(col, keepdims=True)
        mode_val = mode_res.mode[0] if len(mode_res.mode) > 0 else 0
        skew_val = stats.skew(col)
        kurt_val = stats.kurtosis(col)
        median_val = np.median(col)
        q1 = np.percentile(col, 25)
        q3 = np.percentile(col, 75)
        iqr = q3 - q1
        rnge = max_val - min_val
        var_val = np.var(col)
        sem_val = stats.sem(col)
        cv_val = std_val / mean_val if abs(mean_val) > 1e-8 else 0.0
        features.extend([
            mean_val, std_val, max_val, min_val, mode_val,
            skew_val, kurt_val, median_val, q1, q3,
            iqr, rnge, var_val, sem_val, cv_val,
        ])
    return np.array(features, dtype=np.float32)
