"""Shared pytest fixtures: a synthetic but speech-like WAV so tests need no data."""
from __future__ import annotations

import numpy as np
import pytest

SR = 16000


def _synthetic_speech(duration_s: float = 6.0, sr: int = SR) -> np.ndarray:
    """A crude voiced/unvoiced signal: an amplitude- and frequency-modulated tone
    with inserted pauses. Not real speech, but rich enough to exercise every
    extractor (pitch, formants, pauses, spectral shape) without crashing.
    """
    rng = np.random.default_rng(0)
    t = np.linspace(0, duration_s, int(duration_s * sr), endpoint=False)

    # Wandering F0 around 120 Hz + a couple of harmonics.
    f0 = 120 + 10 * np.sin(2 * np.pi * 0.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    voiced = (
        np.sin(phase)
        + 0.5 * np.sin(2 * phase)
        + 0.25 * np.sin(3 * phase)
    )

    # Amplitude envelope with syllable-like modulation + a bit of noise.
    env = 0.5 * (1 + np.sin(2 * np.pi * 3.0 * t)) + 0.05 * rng.standard_normal(len(t))
    sig = voiced * np.clip(env, 0, None)
    sig += 0.02 * rng.standard_normal(len(t))

    # Insert two silent pauses.
    for start_s, end_s in ((1.5, 2.0), (4.0, 4.4)):
        sig[int(start_s * sr) : int(end_s * sr)] = 0.0

    sig = sig / (np.max(np.abs(sig)) + 1e-9) * 0.9
    return sig.astype(np.float32)


@pytest.fixture(scope="session")
def synthetic_wave() -> np.ndarray:
    return _synthetic_speech()


@pytest.fixture(scope="session")
def synthetic_wav_path(tmp_path_factory, synthetic_wave) -> str:
    soundfile = pytest.importorskip("soundfile")
    path = tmp_path_factory.mktemp("audio") / "synthetic.wav"
    soundfile.write(str(path), synthetic_wave, SR)
    return str(path)
