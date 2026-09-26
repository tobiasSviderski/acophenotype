"""Lightweight audio quality checks so callers know when a feature vector is
untrustworthy (too short, silent, or heavily clipped) rather than silently
getting zeros.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MIN_DURATION_S = 1.0        # below this, most windowed features are unreliable
SILENCE_RMS_THRESHOLD = 1e-4  # mean RMS below this ≈ silence
CLIP_FRACTION_THRESHOLD = 0.01  # >1% samples at full scale ≈ clipping


@dataclass
class QualityReport:
    duration_s: float
    mean_rms: float
    clip_fraction: float
    ok: bool = True
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "duration_s": self.duration_s,
            "mean_rms": self.mean_rms,
            "clip_fraction": self.clip_fraction,
            "warnings": list(self.warnings),
        }


def assess(y: np.ndarray, sr: int) -> QualityReport:
    """Assess a loaded mono waveform and flag common problems."""
    duration = len(y) / sr if sr else 0.0
    rms = float(np.sqrt(np.mean(np.square(y)))) if len(y) else 0.0
    clip_fraction = float(np.mean(np.abs(y) >= 0.999)) if len(y) else 1.0

    report = QualityReport(duration_s=duration, mean_rms=rms, clip_fraction=clip_fraction)

    if duration < MIN_DURATION_S:
        report.ok = False
        report.warnings.append(f"recording is very short ({duration:.2f}s < {MIN_DURATION_S}s)")
    if rms < SILENCE_RMS_THRESHOLD:
        report.ok = False
        report.warnings.append("recording appears silent (very low RMS)")
    if clip_fraction > CLIP_FRACTION_THRESHOLD:
        report.warnings.append(f"possible clipping ({clip_fraction:.1%} of samples at full scale)")

    return report
