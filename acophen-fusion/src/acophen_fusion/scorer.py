"""Public API: load a frozen fusion model and score a set of stream feature vectors.

    from acophen_fusion import Scorer

    scorer = Scorer.load(task="binary", variant="stream_only")
    result = scorer.score({
        "biomarkers": bvec, "embeddings": evec, "emotion": mvec, "vision": vvec,
    })
    result.prediction      # "AD"
    result.probability     # 0.78
    result.per_stream      # {"biomarkers": 0.41, "embeddings": 0.24, ...} shares
"""
from __future__ import annotations

import glob
from dataclasses import dataclass, field

import numpy as np

from . import models as M
from .streams import STREAMS, DEMOGRAPHIC_COLUMNS, build_feature_row
from .weights import (
    TASKS, VARIANTS, artifact_paths, resolve_weights_dir, stream_model_path,
    stream_model_relpath,
)

BINARY_LABELS = {0: "CN", 1: "AD"}


@dataclass
class FusionResult:
    task: str
    variant: str
    prediction: object                     # str/int label, or float for regression
    probability: float | None              # P(AD) for binary; None for regression
    per_stream: dict = field(default_factory=dict)        # normalized |contribution| shares
    per_stream_signed: dict = field(default_factory=dict)  # raw signed contributions
    probabilities: dict | None = None      # full class -> prob (multiclass)
    classes: list | None = None
    attribution_method: str = ""

    def as_dict(self) -> dict:
        return {
            "task": self.task,
            "variant": self.variant,
            "prediction": self.prediction,
            "probability": self.probability,
            "per_stream": self.per_stream,
            "per_stream_signed": self.per_stream_signed,
            "probabilities": self.probabilities,
            "classes": self.classes,
            "attribution_method": self.attribution_method,
        }

    def __repr__(self) -> str:
        p = f", p={self.probability:.3f}" if self.probability is not None else ""
        return f"FusionResult(task={self.task!r}, prediction={self.prediction!r}{p})"


def _normalize_shares(signed: dict) -> dict:
    total = sum(abs(v) for v in signed.values())
    if total == 0:
        return {k: 0.0 for k in signed}
    return {k: abs(v) / total for k, v in signed.items()}


class Scorer:
    """A loaded, frozen fusion model for one (task, variant)."""

    def __init__(self, task, variant, weights_dir, artifacts):
        self.task = task
        self.variant = variant
        self.weights_dir = weights_dir
        self._artifacts = artifacts  # task-specific loaded objects

    # ---------------------------------------------------------------- construction
    @classmethod
    def load(cls, task: str = "binary", variant: str = "stream_only", weights_dir=None):
        """Load a frozen fusion model.

        Parameters
        ----------
        task : {"binary", "multiclass", "regression"}
        variant : {"stream_only", "with_demos"}
        weights_dir : path, optional
            Directory containing the model folders. If omitted, resolved from the
            ``ACOPHEN_FUSION_WEIGHTS`` env var or by auto-discovery (see :mod:`weights`).
        """
        if task not in TASKS:
            raise ValueError(f"Unknown task {task!r}. Valid: {TASKS}")
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}. Valid: {VARIANTS}")

        wdir = resolve_weights_dir(weights_dir, task=task, variant=variant)
        paths = artifact_paths(wdir, task, variant)
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing artifact(s) for task={task} variant={variant}:\n  "
                + "\n  ".join(missing)
            )

        if task == "binary":
            stacker, scaler, feature_order = M.load_stacking(paths)
            artifacts = {"stacker": stacker, "scaler": scaler, "feature_order": feature_order}
        elif task == "multiclass":
            bundle = M.load_sgl(paths)
            artifacts = {"bundle": bundle}
        else:  # regression
            pipe, reference_columns = M.load_fc2fs(paths)
            artifacts = {"pipe": pipe, "reference_columns": reference_columns}

        artifacts["biomarker_scaler"] = M.load_biomarker_scaler(wdir)
        return cls(task, variant, wdir, artifacts)

    @property
    def expected_columns(self) -> list:
        """The exact feature columns this model was trained on (frozen order).

        - binary: the stacker's proba/demographic feature order.
        - multiclass: the SGL bundle's ``{stream}__{feature}`` columns.
        - regression: the FC2FS pipeline's training column order.
        """
        if self.task == "binary":
            return list(self._artifacts["feature_order"])
        if self.task == "multiclass":
            return list(self._artifacts["bundle"]["feature_columns"])
        return list(self._artifacts["reference_columns"])

    # --------------------------------------------------------------------- scoring
    def score(self, streams: dict | None = None, demographics: dict | None = None,
              stream_probas: dict | None = None,
              biomarkers_standardized: bool = False) -> FusionResult:
        """Score one recording's stream feature vectors.

        For **binary**, either pass ``streams`` (named feature vectors -> run per-stream
        base models -> stack) or ``stream_probas`` (already-computed per-stream P(AD)).
        For **multiclass/regression**, pass ``streams``.

        ``demographics`` (``{"sex","age","educ"}``) is required for ``with_demos`` variants.

        The models were trained on biomarkers z-scored over the Pitt cohort. Raw
        ``voxmarkers`` output is standardized automatically with the shipped Pitt scaler.
        Pass ``biomarkers_standardized=True`` only if your biomarker vector is *already*
        in that space (e.g. rows of the thesis ``biomarkers.parquet``).
        """
        self._check_demographics(demographics)
        if streams is not None and "biomarkers" in streams and not biomarkers_standardized:
            scaler = self._artifacts.get("biomarker_scaler")
            if scaler:
                streams = dict(streams)
                streams["biomarkers"] = M.standardize_biomarkers(streams["biomarkers"], scaler)
            else:
                import warnings
                warnings.warn(
                    "No biomarker_scaler.json found: raw biomarkers are being scored by models "
                    "trained on z-scored features, so biomarker contributions will be wrong. "
                    "Place biomarker_scaler.json next to the model folders.",
                    RuntimeWarning, stacklevel=2,
                )
        if self.task == "binary":
            return self._score_binary(streams, demographics, stream_probas)
        if self.task == "multiclass":
            return self._score_multiclass(streams, demographics)
        return self._score_regression(streams, demographics)

    # ------------------------------------------------------------------- internals
    def _check_demographics(self, demographics):
        if self.variant == "with_demos":
            if not demographics or any(k not in demographics for k in DEMOGRAPHIC_COLUMNS):
                raise ValueError(
                    f"variant='with_demos' requires demographics with keys "
                    f"{DEMOGRAPHIC_COLUMNS}; got {sorted((demographics or {}).keys())}."
                )

    def _score_binary(self, streams, demographics, stream_probas):
        if stream_probas is None:
            if streams is None:
                raise ValueError("Provide either `streams` or `stream_probas` for binary scoring.")
            stream_probas = self._per_stream_probas(streams)

        stacker = self._artifacts["stacker"]
        scaler = self._artifacts["scaler"]
        feature_order = self._artifacts["feature_order"]

        proba, Xs_row = M.predict_stacking(stacker, scaler, feature_order,
                                            stream_probas, demographics)
        signed = M.stacking_contributions(stacker, feature_order, Xs_row, STREAMS)
        pred_label = int(proba >= 0.5)
        return FusionResult(
            task=self.task, variant=self.variant,
            prediction=BINARY_LABELS.get(pred_label, pred_label),
            probability=proba,
            per_stream=_normalize_shares(signed),
            per_stream_signed=signed,
            classes=list(BINARY_LABELS.values()),
            attribution_method="linear stacker coefficients (exact)",
        )

    def _per_stream_probas(self, streams: dict) -> dict:
        """Run each stream's frozen base model to get its P(AD)."""
        import joblib
        out = {}
        for s in STREAMS:
            if s not in streams:
                raise ValueError(f"Missing stream {s!r}. Binary fusion needs all of {STREAMS}.")
            path = stream_model_path(self.weights_dir, s)
            if not path.exists():
                from .hub import fetch_file
                fetched = fetch_file(stream_model_relpath(s))
                path = fetched if fetched is not None else path
            if not path.exists():
                raise FileNotFoundError(
                    f"Per-stream binary model for {s!r} not found ({path}). "
                    f"Pass precomputed `stream_probas` instead, or install the stream models."
                )
            try:
                model = joblib.load(path)
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    f"Loading the {s!r} base model needs an extra dependency ({e.name}). "
                    f"Reinstall acophen-fusion to pull its dependencies "
                    f"(`pip install -e acophen-fusion`) or pass precomputed `stream_probas`."
                ) from e
            out[s] = self._model_proba(model, streams[s])
        return out

    @staticmethod
    def _model_proba(model, vec):
        import pandas as pd
        names = getattr(model, "feature_names_in_", None)
        if names is None:
            try:
                names = model.booster_.feature_name()
            except Exception:
                names = None
        if isinstance(vec, dict):
            vec = pd.Series(vec)
        if names is not None:
            X = vec.reindex(list(names)).to_frame().T
        else:
            X = vec.to_frame().T
        # pass a named frame when the model knows its feature names (avoids sklearn's
        # "X does not have valid feature names" warning; order is already aligned above)
        X = X.astype(float)
        payload = X if names is not None else X.values
        return float(model.predict_proba(payload)[:, 1][0])

    def _score_multiclass(self, streams, demographics):
        if streams is None:
            raise ValueError("Provide `streams` (named feature vectors) for multiclass scoring.")
        bundle = self._artifacts["bundle"]
        X = build_feature_row(streams, demographics)
        proba, classes, Xs = M.predict_sgl(bundle, X)
        proba = proba[0]
        k = int(np.argmax(proba))
        signed = M.sgl_contributions(bundle, Xs, k, STREAMS)
        classes = [float(c) for c in classes]
        return FusionResult(
            task=self.task, variant=self.variant,
            prediction=classes[k],
            probability=float(proba[k]),
            per_stream=_normalize_shares(signed),
            per_stream_signed=signed,
            probabilities={c: float(p) for c, p in zip(classes, proba)},
            classes=classes,
            attribution_method="SGL logit contribution to predicted class (exact, per-sample)",
        )

    def _score_regression(self, streams, demographics):
        if streams is None:
            raise ValueError("Provide `streams` (named feature vectors) for regression scoring.")
        pipe = self._artifacts["pipe"]
        reference_columns = self._artifacts["reference_columns"]
        X = build_feature_row(streams, demographics)
        yhat = M.predict_fc2fs(pipe, X, reference_columns)
        signed = M.fc2fs_contributions(pipe, reference_columns, STREAMS)
        return FusionResult(
            task=self.task, variant=self.variant,
            prediction=yhat,
            probability=None,
            per_stream=_normalize_shares(signed),
            per_stream_signed=signed,
            attribution_method="FC2FS grouped LGBM feature importance (global)",
        )
