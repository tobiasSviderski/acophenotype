"""Automatic download of stream assets (the VLAD codebook) from the Hugging Face Hub.

The deep backbones (Whisper, ResNet50, WavLM) already download themselves via
``transformers`` / ``torchvision``. The only extra asset this package needs is the fitted
VLAD codebook used by the SSL stream, and it is fetched automatically on first use.
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_REPO_ID = "msvidersky/acophenotype-pitt-models"
DEFAULT_REVISION = None
CODEBOOK_PATH_IN_REPO = "reference/vlad_codebook.pkl"

REPO_ENV = "ACOPHEN_HUB_REPO"
REVISION_ENV = "ACOPHEN_HUB_REVISION"
DISABLE_ENV = "ACOPHEN_NO_DOWNLOAD"


def repo_id() -> str:
    return os.environ.get(REPO_ENV, DEFAULT_REPO_ID)


def revision():
    return os.environ.get(REVISION_ENV, DEFAULT_REVISION)


def downloads_enabled() -> bool:
    return os.environ.get(DISABLE_ENV, "").lower() not in ("1", "true", "yes")


def fetch_vlad_codebook():
    """Download the VLAD codebook and return its local path, or None if unavailable."""
    if not downloads_enabled():
        return None
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None
    try:
        return Path(hf_hub_download(repo_id=repo_id(), filename=CODEBOOK_PATH_IN_REPO,
                                    revision=revision()))
    except Exception:
        return None


def hub_hint() -> str:
    return (
        f"The VLAD codebook is downloaded automatically from '{repo_id()}' on first use. "
        f"If you are offline, set $ACOPHEN_STREAMS_VLAD_CODEBOOK to a local "
        f"vlad_codebook.pkl, or install `huggingface_hub`."
    )
