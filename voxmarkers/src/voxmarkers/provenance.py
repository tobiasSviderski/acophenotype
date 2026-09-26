"""Per-feature provenance: what each biomarker is, its unit, its expected direction
of change in Alzheimer's disease (AD), a one-line clinical rationale, and the
reference(s) that motivate it.

This provenance is the thing that distinguishes ``voxmarkers`` from a generic
feature dumper: every number carries its meaning and its source.

IMPORTANT -- reference reconciliation
-------------------------------------
The ``references`` fields below are **canonical seed references**: the origin
papers for each measure and the best-known clinical-speech sources. They are a
correct, honest starting point, but they were **not** auto-extracted from the
thesis PRISMA-ScR scoping review (that companion review was not present in the
research folder used to build this package). Before publishing, replace/augment
these with the exact rows from your scoping review. Where the AD-specific
direction of change was uncertain, ``direction_in_ad`` is set to ``"altered"``
rather than overclaiming a sign.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .schema import TIER_COLUMNS


@dataclass(frozen=True)
class FeatureInfo:
    name: str
    family: str
    tier: str
    unit: str
    direction_in_ad: str  # "increases" | "decreases" | "altered"
    description: str
    references: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "family": self.family,
            "tier": self.tier,
            "unit": self.unit,
            "direction_in_ad": self.direction_in_ad,
            "description": self.description,
            "references": list(self.references),
        }


# --- seed reference strings (canonical origins; reconcile with scoping review) ---
R_KONIG = "König et al. (2015), Alzheimer's & Dementia: DADM"
R_PISTONO = "Pistono et al. (2019), J Alzheimers Dis"
R_YUAN = "Yuan et al. (2020), Interspeech"
R_MARTINEZ = "Martínez-Sánchez et al. (2017), Clin Linguist Phon"
R_SAPIR = "Sapir et al. (2010), J Speech Lang Hear Res (FCR)"
R_ROY = "Roy et al. (2009), J Voice (VAI)"
R_TEIXEIRA = "Teixeira et al. (2013), Procedia Technology (jitter/shimmer/HNR)"
R_HILLENBRAND = "Hillenbrand et al. (1994), J Speech Hear Res (CPP)"
R_LITTLE07 = "Little et al. (2007), BioMed Eng OnLine (RPDE/DFA dysphonia)"
R_LITTLE09 = "Little et al. (2009), IEEE Trans Biomed Eng (PPE)"
R_MICHAELIS = "Michaelis et al. (1997), Acustica (GNE)"
R_GRABE = "Grabe & Low (2002), Papers in Laboratory Phonology (nPVI)"
R_HAMMARBERG = "Hammarberg et al. (1980), Acta Otolaryngol"
R_EYBEN = "Eyben et al. (2016), IEEE Trans Affect Comput (GeMAPS)"
R_PENG = "Peng et al. (1994), Phys Rev E (DFA)"
R_LZ = "Lempel & Ziv (1976), IEEE Trans Inf Theory"
R_KAISER = "Kaiser (1990), ICASSP (Teager energy operator)"
R_HANSON = "Hanson (1997), J Acoust Soc Am (H1–H2 / glottal source)"
R_FITCH = "Fitch (1997), J Acoust Soc Am (formant dispersion)"
R_DAUBECHIES = "Daubechies (1992), Ten Lectures on Wavelets"
R_DELIYSKI = "Deliyski (1993) / Praat voice report (MDVP-style quality indices)"

_NOTE = "SEED reference — reconcile with PRISMA-ScR scoping review."


# --- named-feature provenance (MFCCs are generated separately, below) ---
_ENTRIES: list[FeatureInfo] = [
    # ---- TIER 1: temporal ----
    FeatureInfo("temp_syllable_rate", "temporal", "core", "syllables/s", "decreases",
                "Syllable nuclei per second of phonated speech; slows in AD.", [R_KONIG]),
    FeatureInfo("temp_speaking_rate", "temporal", "core", "syllables/s", "decreases",
                "Syllable nuclei per second of total recording duration.", [R_KONIG]),
    FeatureInfo("temp_pause_rate", "temporal", "core", "pauses/s", "increases",
                "Number of silent pauses (>150 ms) per second; rises with word-finding difficulty.", [R_PISTONO]),
    FeatureInfo("temp_phonation_ratio", "temporal", "core", "ratio", "decreases",
                "Fraction of the recording that is phonated speech.", [R_KONIG]),
    FeatureInfo("temp_speech_pause_ratio", "temporal", "core", "ratio", "decreases",
                "Phonation time divided by total pause time.", [R_YUAN]),
    FeatureInfo("temp_mean_pause_dur", "temporal", "core", "s", "increases",
                "Mean duration of silent pauses (>150 ms).", [R_PISTONO]),
    FeatureInfo("temp_pause_variability", "temporal", "core", "s", "increases",
                "Std-dev of silent-pause durations; pausing becomes more erratic.", [R_PISTONO]),
    # ---- TIER 1: prosodic ----
    FeatureInfo("pros_f0_variability", "prosodic", "core", "Hz", "decreases",
                "Std-dev of fundamental frequency; reduced pitch variation (monopitch).", [R_MARTINEZ]),
    FeatureInfo("pros_f0_range", "prosodic", "core", "Hz", "decreases",
                "10th–90th percentile F0 range.", [R_MARTINEZ]),
    FeatureInfo("pros_f0_slope", "prosodic", "core", "Hz/s", "altered",
                "Absolute linear slope of the F0 contour.", [R_MARTINEZ]),
    FeatureInfo("pros_loudness_variability", "prosodic", "core", "dB", "decreases",
                "Std-dev of intensity; reduced loudness variation (monoloudness).", [R_EYBEN]),
    FeatureInfo("pros_energy_slope", "prosodic", "core", "dB/s", "altered",
                "Linear slope of the intensity contour.", [R_EYBEN]),
    # ---- TIER 1: articulatory ----
    FeatureInfo("art_fcr", "articulatory", "core", "ratio", "increases",
                "Formant Centralization Ratio; higher = more centralized vowel space.", [R_SAPIR]),
    FeatureInfo("art_vai", "articulatory", "core", "ratio", "decreases",
                "Vowel Articulation Index (reciprocal of FCR); lower = reduced articulation.", [R_ROY, R_SAPIR]),
    FeatureInfo("art_f1_bw", "articulatory", "core", "Hz", "increases",
                "Mean first-formant bandwidth.", [R_EYBEN]),
    FeatureInfo("art_f2_bw", "articulatory", "core", "Hz", "increases",
                "Mean second-formant bandwidth.", [R_EYBEN]),
    FeatureInfo("art_f2_slope", "articulatory", "core", "Hz/s", "decreases",
                "Mean absolute F2 transition rate; reduced formant dynamics.", [R_SAPIR]),
    # ---- TIER 1: phonatory ----
    FeatureInfo("phon_jitter_local", "phonatory", "core", "ratio", "increases",
                "Local jitter (cycle-to-cycle F0 perturbation).", [R_TEIXEIRA]),
    FeatureInfo("phon_jitter_rap", "phonatory", "core", "ratio", "increases",
                "Relative Average Perturbation jitter.", [R_TEIXEIRA]),
    FeatureInfo("phon_shimmer_local_db", "phonatory", "core", "dB", "increases",
                "Local shimmer (cycle-to-cycle amplitude perturbation), in dB.", [R_TEIXEIRA]),
    FeatureInfo("phon_shimmer_apq3", "phonatory", "core", "ratio", "increases",
                "3-point Amplitude Perturbation Quotient shimmer.", [R_TEIXEIRA]),
    FeatureInfo("phon_hnr", "phonatory", "core", "dB", "decreases",
                "Harmonics-to-Noise Ratio, mean over voiced frames; lower = noisier phonation.",
                [R_TEIXEIRA]),
    FeatureInfo("phon_cpp", "phonatory", "core", "dB", "decreases",
                "Smoothed Cepstral Peak Prominence (CPPS); robust marker of dysphonia, "
                "lower = breathier.", [R_HILLENBRAND]),
    # ---- TIER 1: spectral ----
    FeatureInfo("spec_centroid", "spectral", "core", "Hz", "altered",
                "Spectral centroid (centre of mass of the spectrum).", [R_EYBEN]),
    FeatureInfo("spec_flux", "spectral", "core", "a.u.", "altered",
                "Mean onset strength / spectral flux.", [R_EYBEN]),
    FeatureInfo("spec_entropy", "spectral", "core", "nats", "altered",
                "Entropy of the magnitude spectrum.", [R_LITTLE07]),
    FeatureInfo("spec_rpde", "spectral", "core", "nats", "increases",
                "Recurrence-period-density-entropy proxy; higher = less periodic signal.", [R_LITTLE07]),
    # ---- TIER 2: spectral tilt / shape ----
    FeatureInfo("spec_hammarberg", "spectral", "extension", "dB", "altered",
                "Hammarberg index: energy difference between low and high spectral bands.", [R_HAMMARBERG, R_EYBEN]),
    FeatureInfo("spec_alpha_ratio", "spectral", "extension", "dB", "altered",
                "Alpha ratio: high-band minus low-band mean energy (spectral tilt).", [R_EYBEN]),
    FeatureInfo("spec_skewness", "spectral", "extension", "a.u.", "altered",
                "Skewness of the log-magnitude spectrum.", [R_EYBEN]),
    FeatureInfo("spec_kurtosis", "spectral", "extension", "a.u.", "altered",
                "Kurtosis of the log-magnitude spectrum.", [R_EYBEN]),
    FeatureInfo("spec_rolloff", "spectral", "extension", "Hz", "altered",
                "85% spectral roll-off frequency.", [R_EYBEN]),
    # ---- TIER 2: complexity ----
    FeatureInfo("comp_dfa", "complexity", "extension", "exponent", "altered",
                "Detrended Fluctuation Analysis scaling exponent of the waveform.", [R_PENG, R_LITTLE07]),
    FeatureInfo("comp_lzc", "complexity", "extension", "ratio", "altered",
                "Normalized Lempel–Ziv complexity of the binarized signal.", [R_LZ]),
    FeatureInfo("comp_ppe", "complexity", "extension", "nats", "increases",
                "Pitch Period Entropy; higher = less stable pitch control.", [R_LITTLE09]),
    FeatureInfo("comp_gne", "complexity", "extension", "ratio", "decreases",
                "Glottal-to-Noise Excitation ratio; lower = more turbulent noise.", [R_MICHAELIS]),
    # ---- TIER 2: rhythm ----
    FeatureInfo("rhythm_npvi", "rhythm", "extension", "index", "altered",
                "Normalized Pairwise Variability Index of successive voiced-interval durations.", [R_GRABE]),
    # ---- TIER 2: voice quality ----
    FeatureInfo("qual_vti", "voice_quality", "extension", "ratio", "increases",
                "Voice Turbulence Index (Praat voice report).", [R_DELIYSKI]),
    FeatureInfo("qual_spi", "voice_quality", "extension", "ratio", "increases",
                "Soft Phonation Index (Praat voice report).", [R_DELIYSKI]),
    FeatureInfo("qual_ftri", "voice_quality", "extension", "ratio", "increases",
                "Frequency Tremor Intensity Index (Praat voice report).", [R_DELIYSKI]),
    # ---- TIER 2: transform ----
    FeatureInfo("trans_wavelet_energy", "transform", "extension", "a.u.", "altered",
                "Normalized energy of db4 DWT detail coefficients.", [R_DAUBECHIES]),
    # ---- TIER 3: aerodynamic / dynamic ----
    FeatureInfo("aero_h1_h2_diff", "aerodynamic", "aerodynamics", "dB", "altered",
                "H1–H2: amplitude difference of the first two harmonics (glottal source / breathiness).", [R_HANSON]),
    FeatureInfo("dyn_mfcc_velocity_mean", "dynamic", "aerodynamics", "a.u.", "decreases",
                "Mean absolute MFCC delta (articulatory velocity).", [R_EYBEN]),
    FeatureInfo("dyn_mfcc_accel_mean", "dynamic", "aerodynamics", "a.u.", "decreases",
                "Mean absolute MFCC delta-delta (articulatory acceleration).", [R_EYBEN]),
    FeatureInfo("dyn_teager_energy_mean", "dynamic", "aerodynamics", "a.u.", "altered",
                "Mean Teager–Kaiser energy of the waveform.", [R_KAISER]),
    FeatureInfo("dyn_teager_energy_std", "dynamic", "aerodynamics", "a.u.", "altered",
                "Std-dev of Teager–Kaiser energy.", [R_KAISER]),
    FeatureInfo("art_formant_dispersion", "articulatory", "aerodynamics", "Hz", "altered",
                "Mean spacing between successive formants (F1–F3).", [R_FITCH]),
    FeatureInfo("phon_v_uv_ratio", "phonatory", "aerodynamics", "ratio", "altered",
                "Ratio of voiced to unvoiced frames.", [R_TEIXEIRA]),
    FeatureInfo("phon_voice_break_factor", "phonatory", "aerodynamics", "%", "increases",
                "Voice break factor from the Praat voice report.", [R_DELIYSKI]),
    FeatureInfo("pros_articulation_rate_variability", "prosodic", "aerodynamics", "syllables/s", "increases",
                "Std-dev of syllable rate across 4-second windows.", [R_KONIG]),
]

# Build the lookup, then add MFCC entries programmatically.
PROVENANCE: dict[str, FeatureInfo] = {e.name: e for e in _ENTRIES}

for _i in range(1, 14):
    for _stat, _statname in (("mean", "mean"), ("std", "std-dev")):
        _name = f"spec_mfcc{_i}_{_stat}"
        PROVENANCE[_name] = FeatureInfo(
            name=_name,
            family="spectral",
            tier="core",
            unit="a.u.",
            direction_in_ad="altered",
            description=f"{_statname.capitalize()} of MFCC coefficient {_i} "
                        f"(mel-cepstral spectral envelope).",
            references=[R_EYBEN],
        )


def describe(feature: str) -> dict:
    """Return the provenance metadata for a single feature as a plain dict."""
    try:
        return PROVENANCE[feature].as_dict()
    except KeyError:
        raise KeyError(
            f"{feature!r} has no provenance entry. "
            f"Known features: {len(PROVENANCE)} total."
        ) from None


def provenance_frame():
    """Return the full provenance table as a pandas DataFrame (one row per feature)."""
    import pandas as pd

    rows = [PROVENANCE[c].as_dict() for tier in TIER_COLUMNS for c in TIER_COLUMNS[tier]]
    df = pd.DataFrame(rows)
    df["references"] = df["references"].apply(lambda refs: "; ".join(refs))
    return df


def missing_provenance() -> list[str]:
    """Feature columns in the schema that lack a provenance entry (should be empty)."""
    return [c for tier in TIER_COLUMNS for c in TIER_COLUMNS[tier] if c not in PROVENANCE]
