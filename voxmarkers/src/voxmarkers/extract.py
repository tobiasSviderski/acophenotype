"""Public extraction API: turn a recording into a named, schema-ordered feature vector.

``extract`` accepts a WAV path *or* a ``(numpy array, sample_rate)`` pair, runs the
requested feature tiers (chunking long recordings exactly as the thesis scripts did),
and returns a pandas ``Series`` whose index is the frozen feature-name contract.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

from .features import TIERS
from .schema import DEFAULT_TIERS, ID_COLUMN, TIER_COLUMNS, columns_for
from .quality import QualityReport, assess

SAMPLE_RATE = 16000
# Whole-file processing by default. Verified against the thesis CSV: recordings longer than
# 60 s (40% of Pitt) match all 47 fusion features only when processed as ONE segment, so
# the thesis features were not chunked despite the script's CHUNK_DURATION = 60.
# Set a finite value (e.g. voxmarkers.extract.CHUNK_DURATION = 60.0) to bound memory on
# very long recordings, at the cost of no longer matching the trained models exactly.
CHUNK_DURATION = float("inf")

AudioLike = "str | os.PathLike | np.ndarray"


def _normalize_tiers(tiers) -> tuple[str, ...]:
    if tiers is None:
        return tuple(DEFAULT_TIERS)
    if isinstance(tiers, str):
        tiers = (tiers,)
    tiers = tuple(tiers)
    unknown = set(tiers) - set(TIER_COLUMNS)
    if unknown:
        raise ValueError(f"Unknown tier(s): {sorted(unknown)}. Valid: {sorted(TIER_COLUMNS)}")
    # canonical order
    return tuple(t for t in DEFAULT_TIERS if t in set(tiers))


def _iter_chunks(audio, sr) -> tuple[Iterator[np.ndarray], float]:
    """Yield 16 kHz mono chunks (<=60 s) and report total duration.

    Path inputs are streamed from disk with offset loads (low memory) exactly as
    the original scripts did; array inputs are sliced in memory.
    """
    import librosa

    if isinstance(audio, (str, os.PathLike)):
        path = os.fspath(audio)
        total_duration = librosa.get_duration(path=path)

        def _gen_path() -> Iterator[np.ndarray]:
            if total_duration <= CHUNK_DURATION:
                y, _ = librosa.load(path, sr=SAMPLE_RATE)
                yield y
            else:
                for offset in np.arange(0, total_duration, CHUNK_DURATION):
                    yc, _ = librosa.load(path, sr=SAMPLE_RATE, offset=offset, duration=CHUNK_DURATION)
                    if len(yc) < SAMPLE_RATE:  # skip tiny tails (<1 s)
                        continue
                    yield yc

        return _gen_path(), total_duration

    # array-like input
    y = np.asarray(audio, dtype=float)
    if y.ndim > 1:
        y = librosa.to_mono(y)
    if sr is None:
        raise ValueError("When passing an array you must also pass its sample_rate `sr`.")
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    total_duration = len(y) / SAMPLE_RATE

    def _gen_array() -> Iterator[np.ndarray]:
        if total_duration <= CHUNK_DURATION:
            yield y
        else:
            step = int(CHUNK_DURATION * SAMPLE_RATE)
            for start in range(0, len(y), step):
                yc = y[start : start + step]
                if len(yc) < SAMPLE_RATE:
                    continue
                yield yc

    return _gen_array(), total_duration


def _clean(value) -> float:
    """NaN/inf -> 0.0, matching the originals' final CSV cleaning."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(v):
        return 0.0
    return v


def extract(
    audio,
    sr: int | None = None,
    tiers: Sequence[str] | str | None = DEFAULT_TIERS,
    *,
    return_quality: bool = False,
    name: str | None = None,
):
    """Extract interpretable acoustic biomarkers from one recording.

    Parameters
    ----------
    audio : str | PathLike | numpy.ndarray
        Path to a WAV file, or a raw mono waveform array.
    sr : int, optional
        Sample rate of ``audio`` when it is an array. Ignored for path inputs.
    tiers : sequence of str | str, optional
        Any of ``"core"``, ``"extension"``, ``"aerodynamics"``. Defaults to all
        three. Output is always in canonical schema order regardless of request order.
    return_quality : bool
        If True, return ``(series, QualityReport)`` instead of just the series.
    name : str, optional
        Value for the returned Series' ``.name`` (defaults to the file stem).

    Returns
    -------
    pandas.Series
        Named feature vector indexed by the frozen feature-name contract.
    """
    import pandas as pd

    tiers = _normalize_tiers(tiers)
    cols = columns_for(tiers)

    chunks, total_duration = _iter_chunks(audio, sr)

    per_tier_segments: dict[str, list[dict]] = {t: [] for t in tiers}

    # quality accumulators
    n_samples = 0
    sum_sq = 0.0
    n_clipped = 0

    for chunk in chunks:
        n_samples += len(chunk)
        sum_sq += float(np.sum(np.square(chunk)))
        n_clipped += int(np.sum(np.abs(chunk) >= 0.999))
        for t in tiers:
            seg = TIERS[t].process_segment(chunk, SAMPLE_RATE)
            if seg is not None:
                per_tier_segments[t].append(seg)

    values: dict[str, float] = {}
    for t in tiers:
        segs = per_tier_segments[t]
        if segs:
            values.update(TIERS[t].aggregate(segs, total_duration))
        else:
            values.update({c: 0.0 for c in TIER_COLUMNS[t]})

    series = pd.Series({c: _clean(values.get(c, 0.0)) for c in cols}, dtype=float)
    if name is None and isinstance(audio, (str, os.PathLike)):
        name = Path(os.fspath(audio)).stem
    series.name = name

    if return_quality:
        mean_rms = float(np.sqrt(sum_sq / n_samples)) if n_samples else 0.0
        clip_fraction = float(n_clipped / n_samples) if n_samples else 1.0
        report = QualityReport(duration_s=total_duration, mean_rms=mean_rms, clip_fraction=clip_fraction)
        if total_duration < 1.0:
            report.ok = False
            report.warnings.append(f"recording is very short ({total_duration:.2f}s)")
        if mean_rms < 1e-4:
            report.ok = False
            report.warnings.append("recording appears silent (very low RMS)")
        if clip_fraction > 0.01:
            report.warnings.append(f"possible clipping ({clip_fraction:.1%} of samples at full scale)")
        return series, report

    return series


def extract_batch(
    paths: Iterable,
    tiers: Sequence[str] | str | None = DEFAULT_TIERS,
    *,
    progress: bool = False,
):
    """Extract features for many recordings, returning a DataFrame.

    The DataFrame is indexed by recording name (file stem) and columns follow the
    frozen schema order. Failures produce an all-zero row rather than aborting the batch.
    """
    import pandas as pd

    paths = list(paths)
    if progress:
        try:
            from tqdm import tqdm
            paths = tqdm(paths, unit="file")
        except ImportError:
            pass

    tiers = _normalize_tiers(tiers)
    cols = columns_for(tiers)

    rows = []
    index = []
    for p in paths:
        stem = Path(os.fspath(p)).stem
        try:
            s = extract(p, tiers=tiers)
        except Exception:
            s = pd.Series({c: 0.0 for c in cols}, dtype=float)
        rows.append(s)
        index.append(stem)

    df = pd.DataFrame(rows, index=index)
    df = df.reindex(columns=cols)
    df.index.name = ID_COLUMN
    return df
