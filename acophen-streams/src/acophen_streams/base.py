"""The common stream interface.

Every stream implements ``extract(wav) -> named feature vector`` and exposes a frozen
``schema``. The fusion layer never cares which streams exist; it consumes named vectors
through this one interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .schemas import StreamSchema

TARGET_SR = 16000


class MissingBackendError(ImportError):
    """Raised when a stream's heavy optional dependency (torch, etc.) is not installed."""


class BaseStream(ABC):
    """Abstract base for a feature stream.

    Subclasses set :attr:`name` and :attr:`schema`, implement :meth:`_extract_frames`
    (backbone forward -> ``T x D`` array) and :meth:`_pool` (``T x D`` -> flat vector).
    The base handles audio loading, naming, and schema validation.
    """

    name: str
    schema: StreamSchema
    #: pip extra that provides this stream's backbone, e.g. ``"ssl"``.
    extra: str = ""

    # -- subclass hooks --------------------------------------------------------
    @abstractmethod
    def _extract_frames(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Backbone forward pass -> ``T x D`` per-chunk representation."""

    @abstractmethod
    def _pool(self, frames: np.ndarray) -> np.ndarray:
        """Pool ``T x D`` frames into a flat feature vector matching ``self.schema``."""

    # -- public API ------------------------------------------------------------
    def extract(self, audio, sr: int | None = None, name: str | None = None):
        """Extract this stream's features from one recording.

        Parameters
        ----------
        audio : str | PathLike | numpy.ndarray
            WAV path or a raw mono waveform.
        sr : int, optional
            Sample rate when ``audio`` is an array.
        name : str, optional
            Name for the returned Series.

        Returns
        -------
        pandas.Series
            Feature vector indexed by ``self.schema.columns`` (frozen order).
        """
        import pandas as pd

        y, sr = self._load_audio(audio, sr)
        frames = self._extract_frames(y, sr)
        vec = np.asarray(self._pool(frames), dtype=float).ravel()
        cols = self.schema.columns
        if len(vec) != len(cols):
            raise ValueError(
                f"{self.name} stream produced {len(vec)} values but schema "
                f"v{self.schema.version} expects {len(cols)}."
            )
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        return pd.Series(vec, index=list(cols), name=name, dtype=float)

    def extract_frames(self, audio, sr: int | None = None):
        """Return the *per-chunk* representation before pooling.

        This is what makes time-resolved views possible (e.g. the emotion trajectory) and
        gives access to raw embeddings (e.g. the mean vision embedding for a UMAP
        projection). Pooling collapses the time axis; this does not.

        Returns
        -------
        frames : numpy.ndarray
            ``T x D`` array, one row per analysis chunk.
        times : numpy.ndarray | None
            Chunk centre times in seconds (``T``,), or ``None`` if the stream doesn't
            define a meaningful time axis.
        """
        y, sr = self._load_audio(audio, sr)
        frames = np.asarray(self._extract_frames(y, sr))
        return frames, self.frame_times(len(frames))

    def frame_times(self, n_frames: int):
        """Chunk centre times in seconds, or None if the stream has no time axis."""
        return None

    @property
    def frame_labels(self):
        """Names for the ``D`` columns of :meth:`extract_frames`, or None."""
        return None

    # -- helpers ---------------------------------------------------------------
    @staticmethod
    def _load_audio(audio, sr):
        import os

        if isinstance(audio, (str, os.PathLike)):
            import librosa
            y, sr = librosa.load(os.fspath(audio), sr=TARGET_SR, mono=True)
            return y, TARGET_SR
        y = np.asarray(audio, dtype=np.float32)
        if y.ndim > 1:
            import librosa
            y = librosa.to_mono(y)
        if sr is None:
            raise ValueError("Pass `sr` when giving a raw array.")
        if sr != TARGET_SR:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        return y, TARGET_SR

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} dim={self.schema.dim}>"
