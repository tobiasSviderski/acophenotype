"""Tier 1 -- core interpretable biomarkers.

Refactored from ``1a_core_features.py``. The per-segment computation is a
behaviour-preserving lift of ``process_audio_segment``; temporal *rates* are
finalised in :func:`aggregate`, mirroring the original chunked aggregation so
that single-file and long-file (chunked) paths agree.
"""
from __future__ import annotations

import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
from scipy.stats import entropy

from ._utils import calculate_rpde

TIER = "core"

# Feature columns produced by this tier, in canonical order (excludes recording id).
COLUMNS = [
    "temp_syllable_rate", "temp_speaking_rate", "temp_pause_rate",
    "temp_phonation_ratio", "temp_speech_pause_ratio",
    "temp_mean_pause_dur", "temp_pause_variability",
    "pros_f0_variability", "pros_f0_range", "pros_f0_slope",
    "pros_loudness_variability", "pros_energy_slope",
    "art_fcr", "art_vai", "art_f1_bw", "art_f2_bw", "art_f2_slope",
    "phon_jitter_local", "phon_jitter_rap",
    "phon_shimmer_local_db", "phon_shimmer_apq3",
    "phon_hnr", "phon_cpp",
    "spec_centroid", "spec_flux", "spec_entropy", "spec_rpde",
]
for _i in range(1, 14):
    COLUMNS += [f"spec_mfcc{_i}_mean", f"spec_mfcc{_i}_std"]

# Non-temporal columns are aggregated across chunks by averaging.
_TEMPORAL = COLUMNS[:7]
_AVERAGED = COLUMNS[7:]


def _safe(fn):
    """Run one Praat measure; NaN on failure or a non-finite result."""
    try:
        v = float(fn())
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def _phonation(sound) -> dict:
    """Jitter, shimmer, HNR and CPPS, each isolated so one failure can't erase the rest."""
    out = {k: np.nan for k in ("phon_jitter_local", "phon_jitter_rap", "phon_shimmer_local_db",
                               "phon_shimmer_apq3", "phon_hnr", "phon_cpp")}
    try:
        pp = call(sound, "To PointProcess (periodic, cc)", 75, 500)
    except Exception:
        pp = None
    if pp is not None:
        out["phon_jitter_local"] = _safe(lambda: call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
        out["phon_jitter_rap"] = _safe(lambda: call(pp, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3))
        out["phon_shimmer_local_db"] = _safe(
            lambda: call([sound, pp], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
        out["phon_shimmer_apq3"] = _safe(
            lambda: call([sound, pp], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6))

    # HNR: Praat's "Get mean" averages voiced frames only (unvoiced = undefined, -200).
    out["phon_hnr"] = _safe(
        lambda: call(call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0), "Get mean", 0, 0))

    # Smoothed cepstral peak prominence (CPPS; Hillenbrand et al. 1994/1996 recipe).
    def _cpps():
        cg = call(sound, "To PowerCepstrogram", 60, 0.002, 5000, 50)
        return call(cg, "Get CPPS", "no", 0.02, 0.0005, 60, 330, 0.05,
                    "Parabolic", 0.001, 0.05, "Straight", "Robust")
    out["phon_cpp"] = _safe(_cpps)
    return out


def process_segment(y: np.ndarray, sr: int) -> dict | None:
    """Compute raw tier-1 features for one audio segment.

    Returns a dict that includes intermediate counts (``n_syllables`` etc.) used
    by :func:`aggregate`. Returns ``None`` on failure, matching the original.
    """
    try:
        sound = parselmouth.Sound(y, sampling_frequency=sr)
        duration = len(y) / sr
        feat: dict = {}

        # 1. TEMPORAL (counts; rates finalised in aggregate)
        rms = librosa.feature.rms(y=y)[0]
        peaks = librosa.util.peak_pick(
            rms, pre_max=5, post_max=5, pre_avg=5, post_avg=5, delta=0.1, wait=10
        )
        n_syllables = len(peaks)

        intervals = librosa.effects.split(y, top_db=25)
        phonation_time = sum([(end - start) / sr for start, end in intervals])

        pauses = []
        for i in range(len(intervals) - 1):
            pause_dur = (intervals[i + 1][0] - intervals[i][1]) / sr
            if pause_dur > 0.150:
                pauses.append(pause_dur)

        feat["n_syllables"] = n_syllables
        feat["phonation_time"] = phonation_time
        feat["duration"] = duration
        feat["n_pauses"] = len(pauses)
        feat["pause_durs"] = pauses

        # 2. PROSODY
        pitch = sound.to_pitch()
        f0 = pitch.selected_array["frequency"]
        f0 = f0[f0 > 10]
        if len(f0) > 5:
            feat["pros_f0_variability"] = np.std(f0)
            feat["pros_f0_range"] = np.percentile(f0, 90) - np.percentile(f0, 10)
            times = np.arange(len(f0)) * pitch.dt
            slope, _ = np.polyfit(times, f0, 1)
            feat["pros_f0_slope"] = np.abs(slope)
        else:
            feat["pros_f0_variability"] = np.nan
            feat["pros_f0_range"] = np.nan
            feat["pros_f0_slope"] = np.nan

        intensity = sound.to_intensity()
        int_vals = intensity.values.T
        if int_vals.size > 0:
            feat["pros_loudness_variability"] = np.std(int_vals)
            int_times = np.arange(len(int_vals)) * intensity.dt
            slope_int, _ = np.polyfit(int_times.flatten(), int_vals.flatten(), 1)
            feat["pros_energy_slope"] = slope_int
        else:
            feat["pros_loudness_variability"] = np.nan
            feat["pros_energy_slope"] = np.nan

        # 3. ARTICULATION
        formant = sound.to_formant_burg(time_step=None)
        f1_vals, f2_vals, bw1_vals, bw2_vals = [], [], [], []
        n_frames = call(formant, "Get number of frames")
        for i in range(1, n_frames + 1):
            t = call(formant, "Get time from frame number", i)
            f1 = call(formant, "Get value at time", 1, t, "Hertz", "Linear")
            f2 = call(formant, "Get value at time", 2, t, "Hertz", "Linear")
            bw1 = call(formant, "Get bandwidth at time", 1, t, "Hertz", "Linear")
            bw2 = call(formant, "Get bandwidth at time", 2, t, "Hertz", "Linear")
            if not np.isnan(f1):
                f1_vals.append(f1)
            if not np.isnan(f2):
                f2_vals.append(f2)
            if not np.isnan(bw1):
                bw1_vals.append(bw1)
            if not np.isnan(bw2):
                bw2_vals.append(bw2)

        if len(f1_vals) > 5 and len(f2_vals) > 5:
            f1_i = np.percentile(f1_vals, 5)
            f2_i = np.percentile(f2_vals, 95)
            f1_u = np.percentile(f1_vals, 5)
            f2_u = np.percentile(f2_vals, 5)
            f1_a = np.percentile(f1_vals, 95)
            f2_a = np.percentile(f2_vals, 50)

            fcr_den = f2_i + f1_a
            feat["art_fcr"] = (f2_u + f2_a + f1_i + f1_u) / fcr_den if fcr_den != 0 else np.nan
            feat["art_vai"] = 1 / feat["art_fcr"] if feat["art_fcr"] and feat["art_fcr"] != 0 else np.nan
            feat["art_f1_bw"] = np.nanmean(bw1_vals)
            feat["art_f2_bw"] = np.nanmean(bw2_vals)
            if len(f2_vals) > 1:
                dt = 0.00625
                feat["art_f2_slope"] = np.mean(np.abs(np.diff(f2_vals))) / dt
            else:
                feat["art_f2_slope"] = np.nan
        else:
            feat["art_fcr"] = np.nan
            feat["art_vai"] = np.nan
            feat["art_f1_bw"] = np.nan
            feat["art_f2_bw"] = np.nan
            feat["art_f2_slope"] = np.nan

        # 4. PHONATION
        # Each measure is computed independently. In the thesis script all six lived in one
        # try-block whose last call ("To PowerCepstrum" on a Sound) is not a valid Praat
        # command, so the block ALWAYS raised and every phonation feature became missing
        # for all 1,284 Pitt recordings. HNR also averaged Praat's -200 "undefined" frame
        # markers into the mean. Both are fixed here.
        feat.update(_phonation(sound))

        # 5. SPECTRAL
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            feat[f"spec_mfcc{i + 1}_mean"] = np.mean(mfcc[i])
            feat[f"spec_mfcc{i + 1}_std"] = np.std(mfcc[i])

        feat["spec_centroid"] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        feat["spec_flux"] = np.mean(librosa.onset.onset_strength(y=y, sr=sr))
        feat["spec_entropy"] = entropy(np.abs(np.fft.fft(y)))
        feat["spec_rpde"] = calculate_rpde(y)

        return feat
    except Exception:
        return None


def aggregate(segments: list[dict], total_duration: float) -> dict:
    """Combine per-segment dicts into the final tier-1 feature vector.

    Temporal rates are recomputed from summed counts; every other feature is the
    mean of its finite per-segment values. This reproduces both the single-file
    and chunked code paths of the original script.
    """
    n_syll = sum(s["n_syllables"] for s in segments)
    phon = sum(s["phonation_time"] for s in segments)
    n_pauses = sum(s["n_pauses"] for s in segments)
    pause_durs = [d for s in segments for d in s["pause_durs"]]

    out: dict = {}
    out["temp_syllable_rate"] = n_syll / phon if phon > 0 else 0
    out["temp_speaking_rate"] = n_syll / total_duration if total_duration > 0 else 0
    out["temp_pause_rate"] = n_pauses / total_duration if total_duration > 0 else 0
    out["temp_phonation_ratio"] = phon / total_duration if total_duration > 0 else 0
    total_pause = sum(pause_durs)
    out["temp_speech_pause_ratio"] = phon / total_pause if total_pause > 0 else 0
    out["temp_mean_pause_dur"] = np.mean(pause_durs) if pause_durs else 0
    out["temp_pause_variability"] = np.std(pause_durs) if pause_durs else 0

    for k in _AVERAGED:
        vals = [s[k] for s in segments if k in s and not np.isnan(s[k])]
        out[k] = float(np.mean(vals)) if vals else 0.0

    return out
