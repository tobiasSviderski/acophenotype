"""Reference-cohort statistics (the control distribution) for percentiles & flags.

Reference files live *outside* the wheel. Resolve them via the ``ACOPHENOTYPE_REFERENCE``
environment variable, an explicit ``reference_dir``, or auto-discovery of
``feature_fusion/reference``.

The feature-space problem
-------------------------
Percentiles and flags are only meaningful if the patient vector and the reference cohort
are in the **same feature space**. The reference shipped with the thesis
(``biomarker_cn_stats.json``) was computed in *standardized* space, while ``voxmarkers``
emits *raw* values — comparing them directly flags almost every feature.

This module handles that in three ways, in order of preference:

1. **Declared space + scaler.** A reference may carry an ``_meta`` block::

       {"_meta": {"space": "standardized",
                  "scaler": {"temp_syllable_rate": {"mean": 3.8, "std": 0.9}, ...}}}

   When the reference is standardized *and* ships the raw scaler, :meth:`ReferenceData.
   align` converts a raw patient vector into the reference space automatically, and the
   comparison becomes valid.
2. **Regenerate in raw space.** Use :mod:`acophenotype.build_reference` to rebuild the
   statistics from your own control data in whatever space you actually extract in.
3. **Detect and warn.** Failing both, :meth:`ReferenceData.space_mismatch` detects the
   mismatch (via the fraction of features falling outside the reference's observed range)
   so the profile and report can warn instead of silently reporting nonsense.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ENV_VAR = "ACOPHENOTYPE_REFERENCE"
META_KEY = "_meta"

#: Fraction of features outside the reference's observed [min,max] above which we call it
#: a feature-space mismatch.
MISMATCH_THRESHOLD = 0.25

#: Features whose reference statistics are invalid and must not be used for percentiles
#: or flags. `phon_hnr`: the thesis extraction was broken (only two values, -100 or 0,
#: across all Pitt recordings), so the CN reference is meaningless, while voxmarkers now
#: computes a real HNR. Remove a name here once its reference is rebuilt from fixed
#: features. A reference file can add names via `_meta.unreliable_features`.
UNRELIABLE_REFERENCE_FEATURES = frozenset({"phon_hnr"})


def resolve_reference_dir(reference_dir=None):
    if reference_dir is not None:
        p = Path(reference_dir)
        if (p / "biomarker_cn_stats.json").exists():
            return p
        raise FileNotFoundError(f"No biomarker_cn_stats.json in {p}")
    env = os.environ.get(ENV_VAR)
    if env and (Path(env) / "biomarker_cn_stats.json").exists():
        return Path(env)
    here = Path.cwd()
    for base in [here, *here.parents]:
        cand = base / "feature_fusion" / "reference"
        if (cand / "biomarker_cn_stats.json").exists():
            return cand

    # Automatic download from the Hub (the default path for end users).
    from .hub import fetch_reference_dir
    downloaded = fetch_reference_dir()
    if downloaded is not None:
        return downloaded

    return None  # reference is optional; the profile degrades without it


def _percentile_from_stats(value: float, s: dict) -> float:
    """Piecewise-linear percentile of ``value`` through the (min,q1,median,q3,max) knots."""
    xs_raw = [s.get("min"), s.get("q1"), s.get("median"), s.get("q3"), s.get("max")]
    ys_raw = [0.0, 25.0, 50.0, 75.0, 100.0]
    pts = [(x, y) for x, y in zip(xs_raw, ys_raw) if x is not None]
    if len(pts) < 2:
        return float("nan")
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    if any(xs[i + 1] <= xs[i] for i in range(len(xs) - 1)):
        mean, std = s.get("mean"), s.get("std")
        if mean is not None and std and std > 0:
            from math import erf, sqrt
            z = (value - mean) / std
            return float(50.0 * (1 + erf(z / sqrt(2))))
        return float("nan")
    return float(np.clip(np.interp(value, xs, ys), 0.0, 100.0))


class ReferenceData:
    """Loaded reference distributions with percentile / flag / space-alignment helpers."""

    def __init__(self, biomarker_stats=None, emotion_reference=None, umap_clouds=None,
                 source=None, meta=None):
        stats = dict(biomarker_stats or {})
        self.meta = dict(meta or stats.pop(META_KEY, {}) or {})
        self.unreliable = set(UNRELIABLE_REFERENCE_FEATURES) | set(
            self.meta.get("unreliable_features", []))
        self.biomarker_stats = {k: v for k, v in stats.items() if k not in self.unreliable}
        self.emotion_reference = emotion_reference or {}
        self.umap_clouds = umap_clouds or {}
        self.source = source
        self._umap = None
        self._umap_loaded = False

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, reference_dir=None):
        rdir = resolve_reference_dir(reference_dir)
        if rdir is None:
            return cls(source=None)
        bm = _read_json(rdir / "biomarker_cn_stats.json")
        emo = _read_json(rdir / "emotion_cn_reference.json")
        clouds = _read_json(rdir / "umap_pitt_clouds.json")
        meta = dict(bm.pop(META_KEY, {}) or {})
        # The thesis reference stats are z-scored; biomarker_scaler.json holds the exact
        # Pitt z-score parameters, which lets raw voxmarkers output be aligned to them.
        if not meta.get("scaler"):
            sc = _read_json(rdir / "biomarker_scaler.json")
            if sc.get("features"):
                meta["scaler"] = sc["features"]
                meta.setdefault("space", "standardized")
        return cls(biomarker_stats=bm, emotion_reference=emo, umap_clouds=clouds,
                   source=str(rdir), meta=meta)

    @property
    def available(self) -> bool:
        return bool(self.biomarker_stats)

    @property
    def space(self) -> str:
        """Declared feature space: ``"raw"``, ``"standardized"``, or ``"unknown"``."""
        return self.meta.get("space", "unknown")

    @property
    def scaler(self) -> dict:
        """Optional raw→reference-space scaler: ``{feature: {"mean":…, "std":…}}``."""
        return self.meta.get("scaler", {}) or {}

    # -------------------------------------------------------------- space logic
    def space_mismatch(self, values: dict):
        """Detect a feature-space mismatch.

        Returns ``(is_mismatch, fraction_outside_range)``. A raw value compared against a
        standardized reference typically falls far outside the reference's observed
        ``[min, max]``, so a high fraction is a strong signal.
        """
        checked = outside = 0
        for feat, val in values.items():
            s = self.biomarker_stats.get(feat)
            if not s or s.get("min") is None or s.get("max") is None:
                continue
            checked += 1
            lo, hi = s["min"], s["max"]
            span = (hi - lo) or 1.0
            # allow a generous margin so genuinely extreme-but-valid values don't trip it
            if val < lo - span or val > hi + span:
                outside += 1
        if checked == 0:
            return False, 0.0
        frac = outside / checked
        return frac > MISMATCH_THRESHOLD, frac

    def align(self, values: dict):
        """Convert a raw patient vector into the reference space when possible.

        Returns ``(aligned_values, note)``. If the reference is standardized and ships a
        scaler, raw values are z-scored with it. Otherwise values pass through unchanged.
        """
        scaler = self.scaler
        if self.space == "standardized" and scaler:
            aligned = {}
            for feat, val in values.items():
                sc = scaler.get(feat)
                if sc and sc.get("std"):
                    aligned[feat] = (val - sc["mean"]) / sc["std"]
                else:
                    aligned[feat] = val
            return aligned, "patient vector standardized using the reference scaler"
        return dict(values), None

    # -------------------------------------------------------------- comparisons
    def percentile(self, feature: str, value: float) -> float:
        s = self.biomarker_stats.get(feature)
        if not s:
            return float("nan")
        return _percentile_from_stats(float(value), s)

    def band(self, feature: str, raw_units: bool = True):
        """Return the (q1, median, q3, min, max) reference band for a feature, or None.

        With ``raw_units`` (default) a standardized reference is converted back to the
        feature's original units using the scaler, so it can be displayed next to the
        speaker's raw value.
        """
        s = self.biomarker_stats.get(feature)
        if not s:
            return None
        out = {k: s.get(k) for k in ("q1", "median", "q3", "min", "max")}
        sc = self.scaler.get(feature) if (raw_units and self.space == "standardized") else None
        if sc and sc.get("std"):
            out = {k: (None if v is None else v * sc["std"] + sc["mean"]) for k, v in out.items()}
        return out

    #: Tukey's fence multiplier. 1.5 x IQR beyond the quartiles is the standard outlier
    #: rule (~0.7% of normally distributed values). Flagging anything merely outside the
    #: IQR would flag half of every healthy speaker's features by construction.
    FENCE_K = 1.5

    def flags(self, values: dict, k: float | None = None) -> list:
        """Features that are outliers relative to the reference cohort (Tukey fences)."""
        k = self.FENCE_K if k is None else k
        out = []
        for feat, val in values.items():
            s = self.biomarker_stats.get(feat)
            if not s or s.get("q1") is None or s.get("q3") is None:
                continue
            q1, q3 = s["q1"], s["q3"]
            iqr = q3 - q1
            if val < q1 - k * iqr or val > q3 + k * iqr:
                out.append(feat)
        return out

    # -------------------------------------------------------------------- umap
    @property
    def umap_projector(self):
        """Lazily load ``umap_projector.pkl`` from the reference dir (None if absent)."""
        if not self._umap_loaded:
            self._umap_loaded = True
            self.umap_error = None
            if not self.source:
                self.umap_error = "no reference directory"
            else:
                p = Path(self.source) / "umap_projector.pkl"
                if not p.exists():
                    self.umap_error = f"{p.name} not found in {self.source}"
                else:
                    try:
                        import joblib
                        self._umap = joblib.load(p)
                    except Exception as e:  # keep the reason: it's the only way to debug this
                        self.umap_error = f"could not load {p.name}: {type(e).__name__}: {e}"
        return self._umap

    @property
    def umap_arrays(self):
        """Portable UMAP reference: training embeddings + their 2-D layout (or None).

        ``umap_reference.npz`` holds plain arrays, so unlike ``umap_projector.pkl`` (which
        embeds numba bytecode tied to the Python version that saved it) it loads anywhere.
        """
        if not hasattr(self, "_umap_arrays"):
            self._umap_arrays = None
            if self.source:
                p = Path(self.source) / "umap_reference.npz"
                if p.exists():
                    try:
                        z = np.load(p)
                        self._umap_arrays = (z["raw_data"].astype(np.float64),
                                             z["embedding"].astype(np.float64),
                                             int(z["n_neighbors"]))
                    except Exception as e:
                        self.umap_error = f"could not read {p.name}: {e}"
        return self._umap_arrays

    @staticmethod
    def _knn_project(v, X, E, k):
        """UMAP's placement of a new point: fuzzy-membership-weighted mean of the 2-D
        positions of its k nearest training points (UMAP's transform initialisation,
        without the short SGD refinement). Matches ``umap.transform`` to a median 1.8% of
        the layout diameter on unseen cohorts (corr 0.996 / 0.986)."""
        dist = np.linalg.norm(X - v, axis=1)
        idx = np.argsort(dist)[:k]
        d = dist[idx]
        rho, target = d[0], np.log2(k)
        lo, hi, sigma = 0.0, np.inf, 1.0
        for _ in range(64):                      # smooth_knn_dist binary search
            s = np.exp(-np.maximum(d - rho, 0) / sigma).sum()
            if abs(s - target) < 1e-5:
                break
            if s > target:
                hi = sigma
                sigma = (lo + hi) / 2
            else:
                lo = sigma
                sigma = sigma * 2 if hi == np.inf else (lo + hi) / 2
        sigma = max(sigma, 1e-3 * float(np.mean(d)) + 1e-12)
        w = np.exp(-np.maximum(d - rho, 0) / sigma)
        return w @ E[idx] / w.sum()

    def project(self, embedding):
        """Project a deep embedding into the reference UMAP space -> ``[x, y]`` or None.

        Prefers the portable ``umap_reference.npz``; falls back to the pickled projector.
        On failure, the reason is stored in ``self.umap_error``.
        """
        if embedding is None:
            return None
        arrays = self.umap_arrays
        if arrays is not None:
            X, E, k = arrays
            vec = np.asarray(embedding, dtype=float).ravel()
            if vec.shape[0] != X.shape[1]:
                self.umap_error = (f"embedding has {vec.shape[0]} dims, UMAP reference "
                                   f"expects {X.shape[1]}")
                return None
            return [float(c) for c in self._knn_project(vec, X, E, k)]
        proj = self.umap_projector
        if proj is None:
            return None
        try:
            vec = np.asarray(embedding, dtype=float).reshape(1, -1)
            return [float(v) for v in np.asarray(proj.transform(vec))[0][:2]]
        except Exception as e:
            self.umap_error = f"UMAP transform failed: {type(e).__name__}: {e}"
            return None

    def emotion_zscore(self, key: str, value: float) -> float:
        agg = self.emotion_reference.get("aggregates", {}).get(key)
        if not agg or not agg.get("std"):
            return 0.0
        return float((value - agg["mean"]) / agg["std"])


def _read_json(path: Path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
