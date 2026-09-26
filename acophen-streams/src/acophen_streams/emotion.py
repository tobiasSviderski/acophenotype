"""Emotion stream: WavLM 3LOI categorical SER -> 15-statistic pooling (120-d).

Faithful port of ``emotion_stream/src/inference_3loi_categorical.py``
(model ``3loi/SER-Odyssey-Baseline-WavLM-Categorical``, 3.5 s chunks, 0.5 s stride,
config mean/std normalization, softmax over 8 classes) + ``flattening_statistical.py``
(15 statistics per class). Output columns are ``{emotion}_{stat}``. Requires the
``emotion`` extra (torch + transformers).
"""
from __future__ import annotations

import numpy as np

from .base import BaseStream
from .schemas import EMOTION_SCHEMA
from .pooling import statistical_pool, STAT_NAMES
from ._backends import require_torch, require_transformers, torch_device

MODEL_ID = "3loi/SER-Odyssey-Baseline-WavLM-Categorical"
# Optuna-tuned values used to produce the thesis emotion stream
# (emotion_stream/results/3loi_categorical/1_optimization/best_params.json). The script
# defaults (3.5 s / 0.5 s) were NOT what the fusion models were trained on.
CHUNK_SEC = 1.5
STRIDE_SEC = 0.75
BATCH_SIZE = 32


class EmotionStream(BaseStream):
    name = "emotion"
    schema = EMOTION_SCHEMA
    extra = "emotion"

    def __init__(self, device=None):
        self._device = device
        self._model = None
        self._labels = None  # runtime id2label order

    def _ensure_backbone(self):
        if self._model is not None:
            return
        require_torch(self.extra)
        require_transformers(self.extra)
        import torch
        from transformers import AutoModelForAudioClassification

        self._device = self._device or torch_device()
        self._torch = torch
        self._model = AutoModelForAudioClassification.from_pretrained(
            MODEL_ID, trust_remote_code=True
        ).to(self._device)
        if self._device == "cuda":
            self._model = self._model.half()   # as in the thesis script
        self._model.eval()
        id2label = self._model.config.id2label
        self._labels = [id2label[i].lower() for i in range(len(id2label))]

    def frame_times(self, n_frames: int):
        """Centre time of each 3.5 s chunk taken at a 0.5 s stride."""
        if n_frames <= 0:
            return np.zeros((0,), dtype=float)
        return np.arange(n_frames, dtype=float) * STRIDE_SEC + CHUNK_SEC / 2.0

    @property
    def frame_labels(self):
        """The 8 emotion labels, in the model's own output order (after a backbone load)."""
        return list(self._labels) if self._labels else None

    def _extract_frames(self, y: np.ndarray, sr: int) -> np.ndarray:
        self._ensure_backbone()
        torch = self._torch
        chunk = int(CHUNK_SEC * sr)
        stride = int(STRIDE_SEC * sr)

        wav = torch.from_numpy(np.asarray(y, dtype=np.float32))
        if len(wav) <= chunk:
            chunks = [torch.cat([wav, torch.zeros(chunk - len(wav))])]
        else:
            chunks = [wav[s:s + chunk] for s in range(0, len(wav) - chunk + 1, stride)]

        cfg = self._model.config
        mean_val = getattr(cfg, "mean", None)
        std_val = getattr(cfg, "std", None)
        has_norm = (mean_val is not None) and (std_val is not None)

        probs = []
        with torch.inference_mode():
            for i in range(0, len(chunks), BATCH_SIZE):
                x = torch.stack(chunks[i:i + BATCH_SIZE])
                if has_norm:
                    x = (x - mean_val) / (std_val + 1e-6)
                x = x.to(self._device)
                mask = torch.ones_like(x)
                # The 3LOI custom SERModel takes (input, mask) POSITIONALLY and returns raw
                # logits (not a ModelOutput). Mirrors inference_3loi_categorical.py.
                with torch.amp.autocast("cuda", enabled=(self._device == "cuda")):
                    out = self._model(x, mask)
                logits = getattr(out, "logits", out)
                probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        return np.concatenate(probs, axis=0).astype(np.float32) if probs \
            else np.zeros((0, len(self._labels)), dtype=np.float32)

    def _pool(self, frames: np.ndarray) -> np.ndarray:
        # frames: T x 8 probabilities (columns in runtime id2label order)
        flat = statistical_pool(frames)  # dim-major: [d0 x15, d1 x15, ...]
        named: dict[str, float] = {}
        n_stats = len(STAT_NAMES)
        if frames.ndim == 2 and len(flat) == frames.shape[1] * n_stats:
            for d, label in enumerate(self._labels):
                for j, stat in enumerate(STAT_NAMES):
                    named[f"{label}_{stat}"] = float(flat[d * n_stats + j])
        # reindex to the frozen schema column order (0.0 for anything unseen)
        return np.array([named.get(c, 0.0) for c in self.schema.columns], dtype=float)
