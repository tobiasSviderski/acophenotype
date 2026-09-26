"""Pooling functions are pure NumPy and fully testable without any backbone."""
from __future__ import annotations

import numpy as np

from acophen_streams.pooling import (
    vlad_pool, temporal_pyramid_pool, statistical_pool, STAT_NAMES,
)


class _FakeKMeans:
    """Minimal stand-in for a fitted KMeans (predict + centroids)."""
    def __init__(self, centroids):
        self.cluster_centers_ = np.asarray(centroids, dtype=np.float32)
        self.n_clusters = len(self.cluster_centers_)

    def predict(self, X):
        d = ((X[:, None, :] - self.cluster_centers_[None]) ** 2).sum(-1)
        return d.argmin(1)


def test_vlad_pool_dim_and_l2_norm():
    D, K, T = 8, 4, 20
    km = _FakeKMeans(np.random.default_rng(0).normal(size=(K, D)))
    embs = np.random.default_rng(1).normal(size=(T, D)).astype(np.float32)
    v = vlad_pool(embs, km)
    assert v.shape == (D * K,)
    assert abs(np.linalg.norm(v) - 1.0) < 1e-5  # L2 normalized


def test_vlad_pool_empty_returns_zeros():
    km = _FakeKMeans(np.zeros((4, 8)))
    v = vlad_pool(np.zeros((0, 8), dtype=np.float32), km)
    assert v.shape == (32,) and not v.any()


def test_vlad_assignment_matches_real_kmeans():
    """Direct nearest-centroid assignment must equal sklearn's KMeans.predict."""
    from sklearn.cluster import MiniBatchKMeans
    rng = np.random.default_rng(4)
    data = rng.normal(size=(300, 12)).astype(np.float32)
    km = MiniBatchKMeans(n_clusters=5, random_state=0, n_init=3).fit(data)

    embs = rng.normal(size=(40, 12)).astype(np.float32)
    ours = vlad_pool(embs, km)

    # reference: the original implementation using km.predict
    ids = km.predict(embs)
    ref = np.zeros((5, 12), dtype=np.float32)
    for t in range(len(embs)):
        ref[ids[t]] += embs[t] - km.cluster_centers_[ids[t]]
    ref = ref.flatten()
    ref /= np.linalg.norm(ref)
    np.testing.assert_allclose(ours, ref, rtol=1e-5, atol=1e-6)


def test_temporal_pyramid_dim_matches_formula():
    D, T, levels = 2048, 50, 3
    embs = np.random.default_rng(0).normal(size=(T, D)).astype(np.float32)
    v = temporal_pyramid_pool(embs, levels=levels)
    assert v.shape == (D * (2 ** levels - 1),)      # 2048 * 7 = 14336
    assert v.shape[0] == 14336


def test_temporal_pyramid_level0_is_full_mean():
    D, T = 5, 40
    embs = np.random.default_rng(2).normal(size=(T, D)).astype(np.float32)
    v = temporal_pyramid_pool(embs, levels=3)
    np.testing.assert_allclose(v[:D], embs.mean(axis=0), rtol=1e-5)


def test_statistical_pool_dim():
    D, T = 8, 30
    embs = np.random.default_rng(3).uniform(size=(T, D)).astype(np.float32)
    v = statistical_pool(embs)
    assert v.shape == (D * len(STAT_NAMES),)        # 8 * 15 = 120
    assert v.shape[0] == 120
