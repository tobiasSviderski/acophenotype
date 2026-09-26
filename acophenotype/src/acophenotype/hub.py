"""Automatic download of the reference-cohort files from the Hugging Face Hub.

Percentiles, flags and the UMAP position all need the reference cohort. Rather than make
users fetch it manually, it is downloaded on first use and cached by ``huggingface_hub``.
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_REPO_ID = "sivdma/acophenotype-pitt-models"
DEFAULT_REVISION = None
REPO_PREFIX = "reference"

REPO_ENV = "ACOPHEN_HUB_REPO"
REVISION_ENV = "ACOPHEN_HUB_REVISION"
DISABLE_ENV = "ACOPHEN_NO_DOWNLOAD"

#: The reference bundle. The first is required; the rest are best-effort.
REQUIRED_FILES = ["biomarker_cn_stats.json"]
OPTIONAL_FILES = ["biomarker_scaler.json", "emotion_cn_reference.json",
                  "umap_pitt_clouds.json", "umap_reference.npz"]


def repo_id() -> str:
    return os.environ.get(REPO_ENV, DEFAULT_REPO_ID)


def revision():
    return os.environ.get(REVISION_ENV, DEFAULT_REVISION)


def downloads_enabled() -> bool:
    return os.environ.get(DISABLE_ENV, "").lower() not in ("1", "true", "yes")


def _download(filename: str):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None
    try:
        return Path(hf_hub_download(repo_id=repo_id(),
                                    filename=f"{REPO_PREFIX}/{filename}",
                                    revision=revision()))
    except Exception:
        return None


def fetch_reference_dir():
    """Download the reference bundle and return the directory holding it, or None."""
    if not downloads_enabled():
        return None
    first = _download(REQUIRED_FILES[0])
    if first is None:
        return None
    for name in OPTIONAL_FILES:
        _download(name)  # best-effort; the report degrades without them
    return first.parent


def hub_hint() -> str:
    return (f"Reference data is downloaded automatically from '{repo_id()}' on first use. "
            f"Set $ACOPHENOTYPE_REFERENCE to use a local directory instead.")
