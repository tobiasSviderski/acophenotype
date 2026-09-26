"""SSL / embeddings stream: Whisper-small high layer -> VLAD pooling (12288-d).

Faithful port of ``embedding_stream/src/inference_whisper.py`` (high layer =
last_hidden_state, 30 s chunks, mean-pool over time) + ``flattening_vlad.py`` (VLAD with
a 16-cluster codebook). Requires the ``ssl`` extra (torch + transformers) and a fitted
VLAD codebook.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .base import BaseStream, MissingBackendError
from .schemas import SSL_SCHEMA
from .pooling import vlad_pool
from ._backends import require_torch, require_transformers, torch_device

WHISPER_MODEL = "openai/whisper-small"   # 768-d embeddings
HIGH_LAYER = 12                          # "high" -> encoder last_hidden_state
CHUNK_SEC = 30.0
STRIDE_SEC = 30.0
BATCH_SIZE = 16
N_CLUSTERS = 16                          # 16 x 768 = 12288
CODEBOOK_ENV = "ACOPHEN_STREAMS_VLAD_CODEBOOK"


def _resolve_codebook(path=None):
    """Locate the fitted VLAD codebook (kmeans .pkl)."""
    if path is not None:
        p = Path(path)
        if p.exists():
            return p
        raise FileNotFoundError(f"VLAD codebook not found at {p}")
    env = os.environ.get(CODEBOOK_ENV)
    if env and Path(env).exists():
        return Path(env)
    here = Path.cwd()
    for base in [here, *here.parents]:
        cand = base / "feature_fusion" / "reference" / "vlad_codebook.pkl"
        if cand.exists():
            return cand

    # Automatic download from the Hub (the default path for end users).
    from .hub import fetch_vlad_codebook, hub_hint
    downloaded = fetch_vlad_codebook()
    if downloaded is not None:
        return downloaded

    raise FileNotFoundError(
        "VLAD codebook not found.\n"
        f"{hub_hint()}\n"
        f"Alternatively set ${CODEBOOK_ENV} to your vlad_codebook.pkl, or pass "
        "codebook_path=... to get_stream('ssl', ...)."
    )


class SSLStream(BaseStream):
    name = "ssl"
    schema = SSL_SCHEMA
    extra = "ssl"

    def __init__(self, codebook_path=None, device=None):
        self._codebook_path = codebook_path
        self._device = device
        self._model = None
        self._processor = None
        self._kmeans = None

    # -- lazy resource loading -------------------------------------------------
    def _ensure_backbone(self):
        if self._model is not None:
            return
        require_torch(self.extra)
        require_transformers(self.extra)
        import torch
        from transformers import WhisperProcessor, WhisperModel

        self._device = self._device or torch_device()
        self._processor = WhisperProcessor.from_pretrained(WHISPER_MODEL)
        base = WhisperModel.from_pretrained(WHISPER_MODEL).to(self._device).eval()
        self._encoder = base.encoder
        self._torch = torch

    def _ensure_codebook(self):
        if self._kmeans is None:
            import joblib
            self._kmeans = joblib.load(_resolve_codebook(self._codebook_path))
            if getattr(self._kmeans, "n_clusters", N_CLUSTERS) != N_CLUSTERS:
                raise ValueError(
                    f"VLAD codebook has {self._kmeans.n_clusters} clusters; "
                    f"schema expects {N_CLUSTERS}."
                )

    # -- hooks -----------------------------------------------------------------
    def _extract_frames(self, y: np.ndarray, sr: int) -> np.ndarray:
        self._ensure_backbone()
        torch = self._torch
        chunk = int(CHUNK_SEC * sr)
        stride = int(STRIDE_SEC * sr)

        # 30 s chunks, zero-padded last chunk (matches make_audio_chunks).
        chunks = []
        if len(y) <= chunk:
            chunks.append(np.concatenate([y, np.zeros(chunk - len(y), dtype=y.dtype)]))
        else:
            for start in range(0, max(len(y) - chunk, 0) + 1, stride):
                end = start + chunk
                sl = y[start:end] if end <= len(y) else np.concatenate(
                    [y[start:len(y)], np.zeros(end - len(y), dtype=y.dtype)])
                chunks.append(sl)

        feats = []
        with torch.inference_mode():
            for i in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[i:i + BATCH_SIZE]
                inputs = self._processor(batch, sampling_rate=sr, return_tensors="pt")
                x = inputs.input_features.to(self._device)
                out = self._encoder(x, output_hidden_states=True)
                hs = out.last_hidden_state  # N x T x D  (high layer)
                feats.append(hs.mean(dim=1).cpu().numpy())  # N x D (mean-pool time)
        return np.concatenate(feats, axis=0).astype(np.float32) if feats \
            else np.zeros((0, 768), dtype=np.float32)

    def _pool(self, frames: np.ndarray) -> np.ndarray:
        self._ensure_codebook()
        return vlad_pool(frames, self._kmeans)
