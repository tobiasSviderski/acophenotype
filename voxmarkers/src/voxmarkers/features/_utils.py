"""Shared low-level helpers for the feature extractors.

These are lifted verbatim (behaviour-preserving) from the thesis extraction
scripts ``1a_core_features.py``, ``1b_extension.py`` and ``1c_aerodynamics.py``
so that packaged output matches the values the fusion models were trained on.
"""
from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000
COMPLEXITY_SR = 8000  # complexity metrics are computed on a downsampled signal


def calculate_rpde(y: np.ndarray, tau: int = 1, dim: int = 3) -> float:
    """Recurrence-period-density-entropy proxy (histogram entropy of the signal).

    Ported from ``1a_core_features.calculate_rpde``.
    """
    try:
        n = len(y)
        if n < (dim - 1) * tau + 1:
            return 0.0
        hist_counts, _ = np.histogram(y, bins=50, density=True)
        hist_counts = hist_counts[hist_counts > 0]
        if len(hist_counts) == 0:
            return 0.0
        return float(-np.sum(hist_counts * np.log(hist_counts)))
    except Exception:
        return 0.0


def calculate_dfa(x: np.ndarray, scale_lim=(4, 8)) -> float:
    """Detrended Fluctuation Analysis (native NumPy implementation).

    Ported from ``1b_extension.calculate_dfa``. Avoids the ``nolds`` dependency,
    which breaks on Python 3.10+.
    """
    try:
        x = x - np.mean(x)
        y = np.cumsum(x)
        N = len(y)

        min_scale = 2 ** scale_lim[0]
        max_scale = min(N // 5, 2 ** scale_lim[1])

        if max_scale <= min_scale:
            scales = np.array([min_scale])
        else:
            scales = np.unique(
                np.logspace(np.log10(min_scale), np.log10(max_scale), num=10).astype(int)
            )

        scales = scales[scales > 3]
        if len(scales) < 2:
            return np.nan

        fluctuations = []
        for scale in scales:
            n_segments = N // scale
            rms = []
            for i in range(n_segments):
                seg = y[i * scale : (i + 1) * scale]
                x_idx = np.arange(len(seg))
                coef = np.polyfit(x_idx, seg, 1)
                trend = np.polyval(coef, x_idx)
                rms.append(np.sqrt(np.mean((seg - trend) ** 2)))
            fluctuations.append(np.mean(rms))

        valid_idx = np.where(np.array(fluctuations) > 0)
        log_scales = np.log2(scales[valid_idx])
        log_flucts = np.log2(np.array(fluctuations)[valid_idx])

        if len(log_scales) > 1:
            return float(np.polyfit(log_scales, log_flucts, 1)[0])
        return np.nan
    except Exception:
        return np.nan


def calculate_wavelet_energy(y: np.ndarray) -> float:
    """Energy of DWT detail coefficients (db4), normalised by length.

    Ported from ``1b_extension.calculate_wavelet_energy``.
    """
    import pywt

    try:
        max_level = pywt.dwt_max_level(len(y), pywt.Wavelet("db4").dec_len)
        level = min(5, max_level)
        if level < 1:
            return np.nan
        coeffs = pywt.wavedec(y, "db4", level=level)
        energy = sum(np.sum(np.square(c)) for c in coeffs[1:])
        return float(energy / len(y))
    except Exception:
        return np.nan


def teager_energy_operator(signal: np.ndarray) -> np.ndarray:
    """Teager energy: psi[n] = x[n]^2 - x[n-1]*x[n+1].

    Ported from ``1c_aerodynamics.teager_energy_operator``.
    """
    sq = signal[1:-1] ** 2
    odd = signal[:-2] * signal[2:]
    return sq - odd


def calculate_articulation_var(y: np.ndarray, sr: int) -> float:
    """Std-dev of windowed syllable rate (articulation-rate variability).

    Ported from ``1c_aerodynamics.calculate_articulation_var``.
    """
    import librosa

    try:
        win_len = 4 * sr
        if len(y) < win_len:
            return 0.0
        rates = []
        for start in range(0, len(y) - win_len, win_len):
            chunk = y[start : start + win_len]
            rms = librosa.feature.rms(y=chunk)[0]
            peaks = librosa.util.peak_pick(
                rms, pre_max=5, post_max=5, pre_avg=5, post_avg=5, delta=0.1, wait=10
            )
            rates.append(len(peaks) / 4.0)
        return float(np.std(rates)) if rates else 0.0
    except Exception:
        return 0.0


def praat_report_value(report: str, label: str) -> float:
    """Parse a labelled numeric value out of a Praat 'Voice report' string.

    Ported from the inline ``get_val`` closures in tiers 2 and 3.
    """
    for line in report.split("\n"):
        if label in line:
            try:
                return float(line.split(":")[-1].split("(")[0].strip())
            except (ValueError, IndexError):
                return np.nan
    return np.nan
