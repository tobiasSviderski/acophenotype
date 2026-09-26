"""Reproducibility provenance: versions, model identities, environment, limitations.

A report is only citable if you can tell, later, exactly what produced it. This module
collects the version of every suite package, the frozen feature-schema version, the
identity of the models that actually ran, and the pinned versions of the libraries whose
pickles are version-sensitive (scikit-learn, LightGBM).
"""
from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone

#: Constraints a reader must know before interpreting any number in the report.
LIMITATIONS = [
    "Research use only — not a medical device and not a diagnostic instrument.",
    "Models were trained on the Pitt Corpus (DementiaBank): English, predominantly "
    "picture-description speech, from one recording setup.",
    "Performance is not established for other languages, elicitation tasks, microphones, "
    "or clinical populations.",
    "Reference percentiles describe an acoustic distribution, not a clinical norm.",
    "A single recording is a noisy observation; do not interpret one profile in isolation.",
]


def _version(module_name: str):
    try:
        mod = __import__(module_name)
        return getattr(mod, "__version__", None)
    except Exception:
        return None


def package_versions() -> dict:
    """Versions of the suite packages and the pickle-sensitive dependencies."""
    names = {
        "acophenotype": "acophenotype",
        "voxmarkers": "voxmarkers",
        "acophen_fusion": "acophen_fusion",
        "acophen_streams": "acophen_streams",
        "scikit-learn": "sklearn",
        "lightgbm": "lightgbm",
        "numpy": "numpy",
        "pandas": "pandas",
        "librosa": "librosa",
        "torch": "torch",
    }
    out = {}
    for label, mod in names.items():
        v = _version(mod)
        if v:
            out[label] = v
    return out


def schema_versions() -> dict:
    """Frozen feature-schema versions -- the contract the models were trained against."""
    out = {}
    try:
        import voxmarkers
        out["voxmarkers_feature_schema"] = voxmarkers.FEATURE_SCHEMA.version
    except Exception:
        pass
    try:
        import acophen_streams
        out["acophen_streams_schema"] = acophen_streams.SCHEMA_VERSION
    except Exception:
        pass
    return out


def stream_model_identities(weights_dir=None, task: str = "binary") -> dict:
    """Which per-stream model backs each stream (binary stacker only; name, no load)."""
    if task != "binary":
        return {}
    try:
        from acophen_fusion.weights import BINARY_STREAM_MODELS
    except Exception:
        return {}
    return {s: f"binary_{s}_{m}" for s, m in BINARY_STREAM_MODELS.items()}


def build_provenance(task: str = "binary", weights_dir=None, reference=None,
                     fusion: dict | None = None) -> dict:
    """Assemble the provenance block stored on the profile and rendered in the report."""
    prov = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
        "schemas": schema_versions(),
        "task": task,
        "training_dataset": "Pitt Corpus (DementiaBank)",
        "limitations": list(LIMITATIONS),
    }
    if fusion:
        prov["fusion_model"] = {
            "task": fusion.get("task"),
            "variant": fusion.get("variant"),
            "attribution_method": fusion.get("attribution_method"),
        }
    models = stream_model_identities(weights_dir, task)
    if models:
        prov["stream_models"] = models
    if reference is not None:
        prov["reference"] = {
            "source": reference.source,
            "space": reference.space,
            "n_features": len(reference.biomarker_stats),
            "has_scaler": bool(reference.scaler),
            **({k: v for k, v in reference.meta.items()
                if k in ("n_controls", "built_by")}),
        }
    return prov
