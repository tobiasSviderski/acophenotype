"""Tier 2 -- extension biomarkers (spectral tilt/shape, complexity, voice quality, rhythm).

Refactored from ``1b_extension.py`` (``process_tier2_segment``). Behaviour-preserving.
"""
from __future__ import annotations

import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
from scipy.stats import skew, kurtosis

from ._utils import (
    COMPLEXITY_SR,
    calculate_dfa,
    calculate_wavelet_energy,
    praat_report_value,
)

TIER = "extension"

COLUMNS = [
    "spec_hammarberg", "spec_alpha_ratio",
    "spec_skewness", "spec_kurtosis", "spec_rolloff",
    "comp_dfa", "comp_lzc", "comp_ppe", "comp_gne",
    "rhythm_npvi",
    "qual_vti", "qual_spi", "qual_ftri",
    "trans_wavelet_energy",
]


def process_segment(y: np.ndarray, sr: int) -> dict | None:
    try:
        import antropy

        feat: dict = {}
        sound = parselmouth.Sound(y, sampling_frequency=sr)

        # 1. SPECTRAL TILT
        try:
            ltas = call(sound, "To Ltas", 100)
            max_low = call(ltas, "Get maximum", 0, 2000, "None")
            max_high = call(ltas, "Get maximum", 2000, 5000, "None")
            feat["spec_hammarberg"] = max_low - max_high
            mean_low = call(ltas, "Get mean", 0, 1000, "dB")
            mean_high = call(ltas, "Get mean", 1000, 5000, "dB")
            feat["spec_alpha_ratio"] = mean_high - mean_low
        except Exception:
            feat["spec_hammarberg"] = np.nan
            feat["spec_alpha_ratio"] = np.nan

        # 2. SPECTRAL SHAPE
        try:
            spec = np.abs(librosa.stft(y))
            spec_mag = librosa.amplitude_to_db(spec, ref=np.max)
            feat["spec_skewness"] = skew(spec_mag.flatten())
            feat["spec_kurtosis"] = kurtosis(spec_mag.flatten())
            feat["spec_rolloff"] = np.mean(
                librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
            )
        except Exception:
            feat["spec_skewness"] = np.nan
            feat["spec_kurtosis"] = np.nan
            feat["spec_rolloff"] = np.nan

        # 3. COMPLEXITY (downsampled)
        y_subs = librosa.resample(y, orig_sr=sr, target_sr=COMPLEXITY_SR)
        feat["comp_dfa"] = calculate_dfa(y_subs)

        try:
            y_bin = np.where(y_subs > np.mean(y_subs), 1, 0)
            feat["comp_lzc"] = antropy.lziv_complexity(y_bin, normalize=True)
        except Exception:
            feat["comp_lzc"] = np.nan

        try:
            pitch = sound.to_pitch()
            f0 = pitch.selected_array["frequency"]
            f0 = f0[f0 > 0]
            if len(f0) > 10:
                semitones = 12 * np.log2(f0 / 440.0)
                feat["comp_ppe"] = antropy.spectral_entropy(semitones, sf=100, method="welch")
            else:
                feat["comp_ppe"] = np.nan
        except Exception:
            feat["comp_ppe"] = np.nan

        try:
            harmonicity = call(sound, "To Harmonicity (GNE)", 500, 4500, 1000, 80)
            feat["comp_gne"] = call(harmonicity, "Get mean", 0, 0)
        except Exception:
            feat["comp_gne"] = np.nan

        # 4. VOICE QUALITY & TRANSFORMS
        try:
            pitch_floor, pitch_ceil = 75, 600
            pulses = call(sound, "To PointProcess (periodic, cc)", pitch_floor, pitch_ceil)
            report = call(
                [sound, pitch, pulses], "Voice report", 0, 0,
                pitch_floor, pitch_ceil, 1.3, 1.6, 0.03, 0.45,
            )
            feat["qual_vti"] = praat_report_value(report, "Voice turbulence index")
            feat["qual_spi"] = praat_report_value(report, "Soft phonation index")
            feat["qual_ftri"] = praat_report_value(report, "FTRI")
        except Exception:
            feat["qual_vti"] = np.nan
            feat["qual_spi"] = np.nan
            feat["qual_ftri"] = np.nan

        feat["trans_wavelet_energy"] = calculate_wavelet_energy(y)

        # 5. RHYTHM
        try:
            intervals = librosa.effects.split(y, top_db=25)
            durs = [(end - start) / sr for start, end in intervals]
            if len(durs) > 1:
                diffs = [
                    abs(durs[k] - durs[k + 1]) / ((durs[k] + durs[k + 1]) / 2)
                    for k in range(len(durs) - 1)
                ]
                feat["rhythm_npvi"] = 100 * np.mean(diffs)
            else:
                feat["rhythm_npvi"] = 0.0
        except Exception:
            feat["rhythm_npvi"] = np.nan

        return feat
    except Exception:
        return None


def aggregate(segments: list[dict], total_duration: float) -> dict:
    """Mean of finite per-segment values for each column (matches the original)."""
    out: dict = {}
    for k in COLUMNS:
        vals = [s[k] for s in segments if k in s and not np.isnan(s[k])]
        out[k] = float(np.mean(vals)) if vals else 0.0
    return out
