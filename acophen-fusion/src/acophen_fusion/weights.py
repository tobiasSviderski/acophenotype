"""Resolve model weights, which live *outside* the wheel.

Frozen fusion weights are ~9 MB and (for the deep streams) far larger, so they are not
shipped inside the package. They are resolved at load time from, in priority order:

1. an explicit ``weights_dir`` argument,
2. the ``ACOPHEN_FUSION_WEIGHTS`` environment variable,
3. auto-discovery: a ``models`` directory next to the current working directory or any
   parent, or a sibling ``feature_fusion/models`` (the thesis layout).

A future release will lazy-download a versioned bundle from Zenodo/HuggingFace using the
DOIs recorded in :data:`WEIGHTS_BUNDLE`. For now, point the resolver at your local
``feature_fusion/models`` directory (or set the env var).
"""
from __future__ import annotations

import os
from pathlib import Path

TASKS = ("binary", "multiclass", "regression")
VARIANTS = ("stream_only", "with_demos")

#: Placeholder for the eventual hosted bundle (fill in on first Zenodo release).
WEIGHTS_BUNDLE = {
    "doi": None,           # e.g. "10.5281/zenodo.XXXXXXX"
    "url": None,
    "sha256": None,
    "voxmarkers_schema": "1.0",   # the biomarker schema these weights expect
}

ENV_VAR = "ACOPHEN_FUSION_WEIGHTS"


def artifact_paths(models_dir: Path, task: str, variant: str) -> dict[str, Path]:
    """Return the artifact file paths for a (task, variant) under ``models_dir``."""
    models_dir = Path(models_dir)
    if task == "binary":
        base = models_dir / "stacking" / f"stacking_binary_{variant}"
        return {
            "model": base.with_name(base.name + ".pkl"),
            "scaler": base.with_name(base.name + "_scaler.pkl"),
            "config": base.with_name(base.name + "_config.json"),
        }
    if task == "multiclass":
        return {
            "bundle": models_dir / "final_multiclass_sgl" / f"final_multiclass_sgl_{variant}.pkl",
        }
    if task == "regression":
        return {
            "pipeline": models_dir / "final_regression_fc2fs" / f"final_regression_fc2fs_{variant}.pkl",
        }
    raise ValueError(f"Unknown task {task!r}. Valid: {TASKS}")


#: The per-stream base models the binary stacker was trained on
#: (feature_fusion/run_6b_final_binary.py, SELECTION_STREAM_MAP). Named explicitly because
#: the models folder holds alternatives (e.g. binary_biomarkers_XGBClassifier,
#: binary_vision_LogisticRegression) and a wildcard would pick one arbitrarily.
BINARY_STREAM_MODELS = {
    "biomarkers": "LGBMClassifier",
    "embeddings": "XGBClassifier",
    "emotion": "LogisticRegression",
    "vision": "LGBMClassifier",
}


def stream_model_relpath(stream: str, task: str = "binary") -> str:
    """Path of a per-stream base model relative to the models root."""
    if task != "binary":
        raise ValueError("Per-stream base models are only used by the binary stacker.")
    return f"streams/{stream}/{task}_{stream}_{BINARY_STREAM_MODELS[stream]}.pkl"


def stream_model_path(models_dir: Path, stream: str, task: str = "binary") -> Path:
    return Path(models_dir) / stream_model_relpath(stream, task)


def _looks_like_models_dir(p: Path) -> bool:
    p = Path(p)
    return (p / "stacking").is_dir() or (p / "final_multiclass_sgl").is_dir() \
        or (p / "final_regression_fc2fs").is_dir()


def resolve_weights_dir(weights_dir=None, task=None, variant=None) -> Path:
    """Locate the directory that holds the fusion model artifacts.

    Order: explicit arg -> env var -> local discovery -> **automatic Hub download**.
    Raises a clear error (with instructions) only if all of those fail.
    """
    # 1. explicit
    if weights_dir is not None:
        p = Path(weights_dir)
        if _looks_like_models_dir(p):
            return p
        raise FileNotFoundError(
            f"weights_dir={p} does not contain the expected model folders "
            f"(stacking/ , final_multiclass_sgl/ , final_regression_fc2fs/)."
        )

    # 2. environment variable
    env = os.environ.get(ENV_VAR)
    if env:
        p = Path(env)
        if _looks_like_models_dir(p):
            return p
        raise FileNotFoundError(
            f"${ENV_VAR}={p} does not contain the expected model folders."
        )

    # 3. auto-discovery: walk up from cwd looking for models/ or feature_fusion/models/
    here = Path.cwd()
    for base in [here, *here.parents]:
        for cand in (base / "models", base / "feature_fusion" / "models", base / "weights"):
            if _looks_like_models_dir(cand):
                return cand

    # 4. automatic download from the Hugging Face Hub (the default for end users)
    if task is not None and variant is not None:
        from .hub import ensure_task_weights, hub_hint
        downloaded = ensure_task_weights(task, variant)
        if downloaded is not None:
            return Path(downloaded)
        hint = hub_hint()
    else:
        from .hub import hub_hint
        hint = hub_hint()

    raise FileNotFoundError(
        "Could not locate fusion model weights.\n"
        f"{hint}\n"
        f"Alternatively set ${ENV_VAR} to your local models directory (the one containing "
        "stacking/, final_multiclass_sgl/, final_regression_fc2fs/), or pass "
        "weights_dir=... to Scorer.load()."
    )
