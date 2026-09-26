"""Automatic download of fusion model weights from the Hugging Face Hub.

Users should never have to fetch weights by hand. Resolution order:

1. an explicit ``weights_dir`` argument,
2. ``$ACOPHEN_FUSION_WEIGHTS``,
3. a local ``models`` / ``feature_fusion/models`` directory (developer convenience),
4. **automatic download from the Hub** (the default path for end users).

Downloads are cached by ``huggingface_hub`` (``~/.cache/huggingface``), so the first call
pays the cost and every later call is instant and offline-friendly.
"""
from __future__ import annotations

import os
from pathlib import Path

#: Hub repo holding the frozen model bundle. Override with $ACOPHEN_HUB_REPO.
DEFAULT_REPO_ID = "sivdma/acophenotype-pitt-models"
#: Pin a revision (tag/commit) for reproducibility; None = latest on main.
DEFAULT_REVISION = None
#: Prefix inside the repo where the fusion artefacts live.
REPO_PREFIX = "fusion"

REPO_ENV = "ACOPHEN_HUB_REPO"
REVISION_ENV = "ACOPHEN_HUB_REVISION"
DISABLE_ENV = "ACOPHEN_NO_DOWNLOAD"

# Files needed per (task, variant), relative to REPO_PREFIX.
_TASK_FILES = {
    "binary": [
        "stacking/stacking_binary_{variant}.pkl",
        "stacking/stacking_binary_{variant}_scaler.pkl",
        "stacking/stacking_binary_{variant}_config.json",
    ],
    "multiclass": [
        "final_multiclass_sgl/final_multiclass_sgl_{variant}.pkl",
    ],
    "regression": [
        "final_regression_fc2fs/final_regression_fc2fs_{variant}.pkl",
    ],
}

#: Per-stream base models (needed only for the full binary path from raw vectors).
_STREAM_FILES = [
    "streams/{stream}/{task}_{stream}_{model}.pkl",
]


def repo_id() -> str:
    return os.environ.get(REPO_ENV, DEFAULT_REPO_ID)


def revision():
    return os.environ.get(REVISION_ENV, DEFAULT_REVISION)


def downloads_enabled() -> bool:
    return os.environ.get(DISABLE_ENV, "").lower() not in ("1", "true", "yes")


def _require_hub():
    try:
        from huggingface_hub import hf_hub_download  # noqa: F401
        return True
    except ImportError:
        return False


def fetch_file(path_in_repo: str, prefix: str = REPO_PREFIX):
    """Download one file from the Hub and return its local path (or None)."""
    if not downloads_enabled() or not _require_hub():
        return None
    from huggingface_hub import hf_hub_download
    full = f"{prefix}/{path_in_repo}" if prefix else path_in_repo
    try:
        return Path(hf_hub_download(repo_id=repo_id(), filename=full, revision=revision()))
    except Exception:
        return None


def ensure_task_weights(task: str, variant: str):
    """Ensure the artefacts for one (task, variant) are present locally.

    Returns the directory that can be handed to the normal path resolver, or None if
    the download is unavailable (offline, no ``huggingface_hub``, or repo not published).
    """
    files = _TASK_FILES.get(task)
    if not files:
        return None
    local_paths = []
    for tmpl in files:
        p = fetch_file(tmpl.format(variant=variant))
        if p is None:
            return None
        local_paths.append(p)
    # hf_hub_download mirrors the repo layout inside the cache, so the models root is
    # the parent of the artefact's own sub-directory.
    return local_paths[0].parent.parent


def hub_hint() -> str:
    """A helpful message for error paths."""
    return (
        f"Weights are downloaded automatically from '{repo_id()}' on first use. "
        f"If you are offline, set ${REPO_ENV}/$ACOPHEN_FUSION_WEIGHTS to a local "
        f"models directory, or install `huggingface_hub`."
    )
