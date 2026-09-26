"""voxmarkers -- interpretable acoustic biomarkers of speech.

Research-use software (NOT a medical device). See the README disclaimer.

Quick start
-----------
    from voxmarkers import extract, describe, FEATURE_SCHEMA

    feats = extract("recording.wav")            # pandas Series, schema-ordered
    feats["temp_syllable_rate"]

    describe("art_fcr")                          # provenance for one feature
    FEATURE_SCHEMA.version                       # frozen column-order version
    FEATURE_SCHEMA.columns                       # canonical ordered feature list
"""
from ._version import __version__
from .schema import FEATURE_SCHEMA, SCHEMA_VERSION
from .extract import extract, extract_batch
from .provenance import describe, provenance_frame, PROVENANCE

__all__ = [
    "__version__",
    "extract",
    "extract_batch",
    "describe",
    "provenance_frame",
    "PROVENANCE",
    "FEATURE_SCHEMA",
    "SCHEMA_VERSION",
]
