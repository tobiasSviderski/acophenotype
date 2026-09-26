"""Tier 3 -- aerodynamic / dynamic biomarkers (H1-H2, MFCC dynamics, Teager energy,
formant dispersion, voicing).

Refactored from ``1c_aerodynamics.py`` (``process_tier3_segment``). Behaviour-preserving.
"""
from __future__ import annotations

import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call

from ._utils import (
    teager_energy_operator,
    calculate_articulation_var,
    praat_report_value,
)

TIER = "aerodynamics"

COLUMNS = [
    "aero_h1_h2_diff",
    "dyn_mfcc_velocity_mean", "dyn_mfcc_accel_mean",
    "dyn_teager_energy_mean", "dyn_teager_energy_std",
    "art_formant_dispersion",
    "phon_v_uv_ratio", "phon_voice_break_factor",
    "pros_articulation_rate_variability",
]


def process_segment(y: np.ndarray, sr: int) -> dict | None:
    try:
        sound = parselmouth.Sound(y, sampling_frequency=sr)
        feat: dict = {}

        # 1. DYNAMICS
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        delta1 = librosa.feature.delta(mfcc, order=1)
        delta2 = librosa.feature.delta(mfcc, order=2)
        feat["dyn_mfcc_velocity_mean"] = np.mean(np.abs(delta1))
        feat["dyn_mfcc_accel_mean"] = np.mean(np.abs(delta2))

        teo_sig = teager_energy_operator(y)
        feat["dyn_teager_energy_mean"] = np.mean(teo_sig)
        feat["dyn_teager_energy_std"] = np.std(teo_sig)

        # 2. AERODYNAMICS (H1-H2)
        try:
            pitch = sound.to_pitch()
            f0_mean = call(pitch, "Get mean", 0, 0, "Hertz")
            if not np.isnan(f0_mean) and f0_mean > 0:
                spectrum = sound.to_spectrum()
                h1_freq = f0_mean
                h2_freq = f0_mean * 2
                window = f0_mean * 0.1
                h1_amp = call(spectrum, "Get band density", h1_freq - window, h1_freq + window)
                h2_amp = call(spectrum, "Get band density", h2_freq - window, h2_freq + window)
                h1_db = 10 * np.log10(h1_amp) if h1_amp > 0 else -100
                h2_db = 10 * np.log10(h2_amp) if h2_amp > 0 else -100
                feat["aero_h1_h2_diff"] = h1_db - h2_db
            else:
                feat["aero_h1_h2_diff"] = np.nan
        except Exception:
            feat["aero_h1_h2_diff"] = np.nan

        # 3. PHONATION & ARTICULATION
        try:
            pitch_floor, pitch_ceil = 75, 600
            pulses = call(sound, "To PointProcess (periodic, cc)", pitch_floor, pitch_ceil)
            report = call(
                [sound, pitch, pulses], "Voice report", 0, 0,
                pitch_floor, pitch_ceil, 1.3, 1.6, 0.03, 0.45,
            )
            feat["phon_voice_break_factor"] = praat_report_value(report, "Voice break factor")

            frames = pitch.frames
            voiced_frames = [f for f in frames if f.candidates[0].frequency > 0]
            feat["phon_v_uv_ratio"] = len(voiced_frames) / (len(frames) - len(voiced_frames) + 1e-6)
        except Exception:
            feat["phon_voice_break_factor"] = np.nan
            feat["phon_v_uv_ratio"] = np.nan

        # Formant dispersion (subsampled frames)
        try:
            formant = sound.to_formant_burg(time_step=None)
            dispersion_list = []
            n_frames = call(formant, "Get number of frames")
            for i in range(1, n_frames + 1, 10):
                t = call(formant, "Get time from frame number", i)
                f1 = call(formant, "Get value at time", 1, t, "Hertz", "Linear")
                f2 = call(formant, "Get value at time", 2, t, "Hertz", "Linear")
                f3 = call(formant, "Get value at time", 3, t, "Hertz", "Linear")
                if not np.isnan(f1) and not np.isnan(f2) and not np.isnan(f3):
                    d = (f2 - f1) + (f3 - f2)
                    dispersion_list.append(d / 2)
            feat["art_formant_dispersion"] = np.mean(dispersion_list) if dispersion_list else np.nan
        except Exception:
            feat["art_formant_dispersion"] = np.nan

        # 4. PROSODY
        feat["pros_articulation_rate_variability"] = calculate_articulation_var(y, sr)

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
