"""Lazy loaders for the heavy optional backbones, with actionable error messages.

Nothing here imports torch at module import time -- the deep-learning dependencies are
only touched when a stream actually extracts, so importing ``acophen_streams`` (and using
the registry / schemas) stays lightweight.
"""
from __future__ import annotations

from .base import MissingBackendError


def _require(module: str, extra: str):
    try:
        return __import__(module)
    except ImportError as e:
        raise MissingBackendError(
            f"This stream needs '{module}', which is an optional dependency. "
            f"Install it with:  pip install \"acophen-streams[{extra}]\""
        ) from e


def require_torch(extra: str):
    return _require("torch", extra)


def require_transformers(extra: str):
    return _require("transformers", extra)


def require_torchvision(extra: str):
    return _require("torchvision", extra)


def require_torchaudio(extra: str):
    return _require("torchaudio", extra)


def torch_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"
