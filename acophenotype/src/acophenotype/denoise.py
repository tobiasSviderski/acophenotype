"""Optional speech enhancement (MetricGAN+), matching the thesis ``denoising/`` stage.

Denoising is opt-in. When ``denoise=False`` (default) or the ``[denoise]`` extra is not
installed, audio passes through unchanged. This keeps the umbrella light for users who
only want a biomarker profile.
"""
from __future__ import annotations

import numpy as np

TARGET_SR = 16000
_ENHANCER = None


def _load_enhancer():
    global _ENHANCER
    if _ENHANCER is None:
        try:
            from speechbrain.inference.enhancement import SpectralMaskEnhancement
        except ImportError as e:
            raise ImportError(
                "Denoising needs the '[denoise]' extra: pip install \"acophenotype[denoise]\""
            ) from e
        _ENHANCER = SpectralMaskEnhancement.from_hparams(
            source="speechbrain/metricgan-plus-voicebank",
            savedir="pretrained_models/metricgan-plus-voicebank",
        )
    return _ENHANCER


def denoise(y: np.ndarray, sr: int) -> np.ndarray:
    """Enhance a mono waveform with MetricGAN+ (requires the ``[denoise]`` extra)."""
    import torch

    enhancer = _load_enhancer()
    wav = torch.tensor(np.asarray(y, dtype=np.float32)).unsqueeze(0)
    lengths = torch.tensor([1.0])
    enhanced = enhancer.enhance_batch(wav, lengths=lengths)
    return enhanced.squeeze(0).cpu().numpy().astype(np.float32)
