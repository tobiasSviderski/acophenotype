"""The public object: an ``AcousticProfile`` for one recording."""
from __future__ import annotations

import json
from pathlib import Path


class AcousticProfile:
    """A per-recording acoustic profile: biomarkers, (optional) fused prediction,
    per-stream contributions, reference percentiles, and an HTML report.

    Build one with :meth:`from_audio`. It degrades gracefully: with only ``voxmarkers``
    installed you get a biomarker-only profile (``prediction`` is ``None``).
    """

    def __init__(self, data: dict):
        self._data = data

    # ------------------------------------------------------------------ builders
    @classmethod
    def from_audio(cls, audio, sr=None, task="binary", denoise=False, streams="all",
                   demographics=None, weights_dir=None, reference_dir=None):
        from .orchestrate import build_profile
        data = build_profile(
            audio, sr=sr, task=task, denoise=denoise, streams=streams,
            demographics=demographics, weights_dir=weights_dir, reference_dir=reference_dir,
        )
        return cls(data)

    @classmethod
    def from_dict(cls, data: dict):
        """Wrap an existing profile data dict (e.g. loaded from JSON)."""
        return cls(dict(data))

    # ---------------------------------------------------------------- accessors
    @property
    def task(self) -> str:
        return self._data["metadata"]["task"]

    @property
    def prediction(self):
        f = self._data.get("fusion")
        return f["prediction"] if f else None

    @property
    def probability(self):
        f = self._data.get("fusion")
        return f["probability"] if f else None

    @property
    def per_stream(self) -> dict:
        f = self._data.get("fusion")
        return f["per_stream"] if f else {}

    @property
    def streams(self) -> dict:
        """Raw named feature vectors per stream (in-memory only)."""
        return self._data.get("_vectors", {})

    @property
    def biomarkers(self):
        return self._data["biomarkers"]["values"]

    @property
    def flags(self) -> list:
        return self._data["biomarkers"].get("flags", [])

    @property
    def quality(self) -> dict:
        m = self._data["metadata"]
        return {k: m[k] for k in ("duration_seconds", "snr_db", "quality_flag") if k in m}

    @property
    def notes(self) -> list:
        return self._data.get("notes", [])

    def percentile(self, feature: str):
        """Reference-cohort percentile (0–100) for a biomarker, or None if unavailable."""
        p = self._data["biomarkers"].get("percentiles", {})
        return p.get(feature)

    # ------------------------------------------------------------------- output
    def to_dict(self, include_vectors: bool = False) -> dict:
        d = {k: v for k, v in self._data.items() if k != "_vectors"}
        if include_vectors:
            d["_vectors"] = {k: dict(v.items()) for k, v in self.streams.items()}
        return d

    def to_json(self, path=None, indent: int = 2):
        payload = self.to_dict()
        text = json.dumps(payload, indent=indent, default=float)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
            return path
        return text

    def to_html(self, path=None, *, spectrogram: bool = False) -> str:
        """Render the self-contained HTML report. Returns the HTML string; also writes
        it to ``path`` if given. Set ``spectrogram=True`` to embed a log-mel spectrogram
        (requires ``[report]`` extra + the original audio vectors)."""
        from .report import build_html
        html = build_html(self._data, include_spectrogram=spectrogram)
        if path is not None:
            Path(path).write_text(html, encoding="utf-8")
            return html
        return html

    def __repr__(self) -> str:
        if self.prediction is not None:
            p = f", prediction={self.prediction!r}"
            if self.probability is not None:
                p += f", p={self.probability:.3f}"
        else:
            p = " (biomarker-only)"
        return f"<AcousticProfile task={self.task!r}{p}>"
