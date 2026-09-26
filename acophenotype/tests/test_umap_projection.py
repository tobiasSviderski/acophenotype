"""The portable UMAP placement: numpy-only, no umap-learn, works on any Python version."""
from __future__ import annotations

import numpy as np

from acophenotype.reference import ReferenceData


def _ref_with_layout(tmp_path, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(200, 16)).astype(np.float32)
    E = X[:, :2] * 2.0                       # a layout that preserves the first two axes
    np.savez_compressed(tmp_path / "umap_reference.npz", raw_data=X, embedding=E,
                        n_neighbors=np.int64(15), metric=np.array("euclidean"))
    return ReferenceData(biomarker_stats={"a": {}}, source=str(tmp_path)), X, E


def test_training_point_lands_on_itself(tmp_path):
    ref, X, E = _ref_with_layout(tmp_path)
    xy = ref.project(X[7])
    # a training point's nearest neighbour is itself (weight 1), so it maps to its own spot
    assert np.linalg.norm(np.array(xy) - E[7]) < 0.25


def test_projection_follows_the_layout(tmp_path):
    ref, X, E = _ref_with_layout(tmp_path)
    rng = np.random.default_rng(1)
    new = rng.normal(size=(40, 16))
    out = np.array([ref.project(v) for v in new])
    # the layout keeps axes 0/1, so projected coordinates should track them
    assert np.corrcoef(out[:, 0], new[:, 0])[0, 1] > 0.7
    assert np.isfinite(out).all()


def test_dimension_mismatch_is_reported(tmp_path):
    ref, X, E = _ref_with_layout(tmp_path)
    assert ref.project(np.zeros(5)) is None
    assert "dims" in ref.umap_error


def test_no_umap_learn_needed(tmp_path, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "umap", None)   # simulate umap-learn absent
    ref, X, E = _ref_with_layout(tmp_path)
    assert ref.project(X[0]) is not None
