"""acophenotype -- end-to-end acoustic phenotyping (the umbrella).

Research-use software (NOT a medical device). See the README disclaimer.

    from acophenotype import AcousticProfile

    profile = AcousticProfile.from_audio("recording.wav", task="binary")
    profile.prediction         # "AD" / "CN" / None (biomarker-only)
    profile.per_stream         # each stream's contribution
    profile.percentile("temp_syllable_rate")
    profile.to_html("report.html")

Ties together voxmarkers (biomarkers), acophen-streams (deep streams), and
acophen-fusion (scoring). Degrades gracefully to a biomarker-only profile.
"""
from ._version import __version__
from .profile import AcousticProfile
from .report import build_html
from .reference import ReferenceData
from .build_reference import build_reference, save_reference, scaler_from_features

__all__ = [
    "__version__",
    "AcousticProfile",
    "build_html",
    "ReferenceData",
    "build_reference",
    "save_reference",
    "scaler_from_features",
]
