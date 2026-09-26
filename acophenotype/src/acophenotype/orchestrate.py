"""Orchestration: denoise -> streams -> fuse. Ties the suite packages together.

Everything degrades gracefully. If the deep streams (`acophen-streams`) or the fusion
weights are not available, you still get a biomarker-only profile (no fused prediction)
rather than an error.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .quality import assess
from .reference import ReferenceData

TARGET_SR = 16000

# fusion stream key <- acophen_streams stream name
_STREAM_TO_FUSION = {"ssl": "embeddings", "vision": "vision", "emotion": "emotion"}


def _load_audio(audio, sr):
    if isinstance(audio, (str, os.PathLike)):
        import librosa
        y, sr = librosa.load(os.fspath(audio), sr=TARGET_SR, mono=True)
        return y.astype(np.float32), TARGET_SR, Path(os.fspath(audio)).name
    y = np.asarray(audio, dtype=np.float32)
    if sr is None:
        raise ValueError("Pass `sr` when giving a raw array.")
    if sr != TARGET_SR:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
    return y, TARGET_SR, "array"


def build_profile(audio, sr=None, task="binary", denoise=False, streams="all",
                  demographics=None, weights_dir=None, reference_dir=None):
    """Run the pipeline and return a plain data dict (consumed by AcousticProfile)."""
    import voxmarkers

    notes: list[str] = []
    y, sr, audio_name = _load_audio(audio, sr)

    # 1. optional denoise
    if denoise:
        try:
            from .denoise import denoise as _dn
            y = _dn(y, sr)
            notes.append("denoised with MetricGAN+")
        except Exception as e:  # noqa: BLE001
            notes.append(f"denoise skipped: {e}")

    # 2. quality
    quality = assess(y, sr)

    # 3. biomarkers (always available -- the light core)
    biomarkers = voxmarkers.extract(y, sr=sr, name=audio_name)

    # 4. deep streams (optional)
    stream_vectors = {"biomarkers": biomarkers}
    emotion_vector = None
    trajectory = None
    vision_embedding = None
    requested = _requested_streams(streams)
    try:
        import acophen_streams  # noqa: F401
        from acophen_streams import get_stream, available
        for s in available():
            if s not in requested:
                continue
            try:
                stream = get_stream(s)
                # Per-chunk frames give us the time-resolved views (emotion trajectory)
                # and raw embeddings (vision -> UMAP) that pooling would discard.
                frames, times = stream.extract_frames(y, sr=sr)
                vec = stream.extract(y, sr=sr)
                stream_vectors[_STREAM_TO_FUSION[s]] = vec
                if s == "emotion":
                    emotion_vector = vec
                    trajectory = _trajectory(frames, times, stream.frame_labels)
                elif s == "vision" and len(frames):
                    vision_embedding = np.asarray(frames, dtype=float).mean(axis=0)
            except Exception as e:  # noqa: BLE001
                notes.append(f"{s} stream unavailable: {type(e).__name__}")
    except ImportError:
        notes.append("acophen-streams not installed; biomarker-only profile")

    # 5. fusion (needs all four streams + resolvable weights)
    fusion = None
    have_all = all(k in stream_vectors for k in ("biomarkers", "embeddings", "emotion", "vision"))
    if have_all:
        try:
            from acophen_fusion import Scorer
            variant = "with_demos" if demographics else "stream_only"
            scorer = Scorer.load(task=task, variant=variant, weights_dir=weights_dir)
            result = scorer.score(streams=stream_vectors, demographics=demographics)
            fusion = result.as_dict()
        except Exception as e:  # noqa: BLE001
            notes.append(f"fusion skipped: {type(e).__name__}: {e}")
    else:
        missing = [k for k in ("embeddings", "emotion", "vision") if k not in stream_vectors]
        if missing:
            notes.append(f"no fusion prediction (missing streams: {', '.join(missing)})")

    # 6. reference-based percentiles & flags
    reference = ReferenceData.load(reference_dir)
    raw_values = {k: float(v) for k, v in biomarkers.items()}

    # Align the patient vector into the reference space when the reference tells us how
    # (declared space + scaler). Otherwise compare as-is and detect a mismatch.
    compare_values = raw_values
    space_warning = False
    if reference.available:
        compare_values, align_note = reference.align(raw_values)
        if align_note:
            notes.append(align_note)
        space_warning, frac_out = reference.space_mismatch(compare_values)
        if space_warning:
            notes.append(
                f"Feature-space mismatch: {frac_out:.0%} of biomarkers fall far outside the "
                f"reference's observed range (reference space: {reference.space}). "
                "Percentiles and flags are unreliable. Fix by regenerating the reference in "
                "your extraction space (`acophenotype build-reference`), or by attaching a "
                "raw->standardized scaler to the reference '_meta'."
            )

    flags = reference.flags(compare_values) if reference.available else []
    percentiles = ({k: reference.percentile(k, v) for k, v in compare_values.items()
                    if k in reference.biomarker_stats} if reference.available else {})
    bands = ({k: reference.band(k) for k in percentiles} if reference.available else {})

    emotion_block = _emotion_block(emotion_vector, reference, trajectory)

    # UMAP position of this speaker among the reference clouds
    umap_block = None
    if reference.available and vision_embedding is not None:
        coords = reference.project(vision_embedding)
        if coords:
            umap_block = {
                "coordinates": coords,
                "cn": reference.umap_clouds.get("cn", []),
                "ad": reference.umap_clouds.get("ad", []),
            }
        else:
            notes.append(f"UMAP projection unavailable: "
                         f"{getattr(reference, 'umap_error', None) or 'unknown reason'}")

    from .provenance import build_provenance
    provenance = build_provenance(task=task, weights_dir=weights_dir,
                                  reference=reference, fusion=fusion)

    return {
        "metadata": {
            "audio_file": audio_name,
            "task": task,
            **quality.as_dict(),
        },
        "fusion": fusion,
        "umap": umap_block,
        "provenance": provenance,
        "biomarkers": {
            "values": raw_values,
            "percentiles": percentiles,
            "bands": bands,
            "flags": flags,
            "reference_available": reference.available,
            "space_warning": space_warning,
            "reference_space": reference.space if reference.available else None,
        },
        "emotion": emotion_block,
        "streams_present": sorted(stream_vectors.keys()),
        "reference_source": reference.source,
        "notes": notes,
        "_vectors": stream_vectors,  # kept in-memory (not serialized to JSON)
    }


def _requested_streams(streams) -> set:
    if streams in (None, "all"):
        return {"ssl", "vision", "emotion"}
    if isinstance(streams, str):
        streams = [streams]
    # accept fusion-style "embeddings" as an alias for "ssl"
    alias = {"embeddings": "ssl"}
    return {alias.get(s, s) for s in streams}


def _trajectory(frames, times, labels):
    """Per-chunk emotion probabilities as a serializable time series."""
    frames = np.asarray(frames, dtype=float)
    if frames.ndim != 2 or frames.shape[0] == 0:
        return None
    if not labels or len(labels) != frames.shape[1]:
        labels = [f"dim{i}" for i in range(frames.shape[1])]
    if times is None:
        times = np.arange(frames.shape[0], dtype=float)
    return {
        "times": [round(float(t), 3) for t in np.asarray(times).ravel()[:len(frames)]],
        "series": {lab: [round(float(v), 5) for v in frames[:, i]]
                   for i, lab in enumerate(labels)},
    }


#: statistics surfaced per emotion (mean plus the variability measures)
_EMOTION_STATS = ("mean", "std", "range", "cv")


def _emotion_block(emotion_vector, reference, trajectory=None):
    if emotion_vector is None:
        return None

    # every emotion the schema knows about, not a hardcoded four
    emotions = []
    try:
        from acophen_streams import EMOTION_LABELS
        emotions = list(EMOTION_LABELS)
    except Exception:
        emotions = sorted({c.rsplit("_", 1)[0] for c in emotion_vector.index})

    table = {}
    for emo in emotions:
        row = {}
        for stat in _EMOTION_STATS:
            key = f"{emo}_{stat}"
            if key in emotion_vector.index:
                val = float(emotion_vector[key])
                row[stat] = val
                if reference.available:
                    z = reference.emotion_zscore(key, val)
                    if z:
                        row[f"{stat}_z"] = round(z, 2)
        if row:
            table[emo] = row

    block = {"table": table, "stats": list(_EMOTION_STATS)}
    if trajectory:
        block["trajectory"] = trajectory
    # keep the flat aggregates for backwards compatibility with existing reports/JSON
    block["aggregates"] = {f"{e}_mean": v["mean"] for e, v in table.items() if "mean" in v}
    return block
