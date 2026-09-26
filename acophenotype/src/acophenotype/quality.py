"""Recording quality assessment (duration, crude SNR)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Quality:
    duration_s: float
    snr_db: float
    flag: str          # "high" | "low"
    warnings: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "duration_seconds": round(self.duration_s, 2),
            "snr_db": round(self.snr_db, 2),
            "quality_flag": self.flag,
            "warnings": list(self.warnings),
        }


def assess(y: np.ndarray, sr: int) -> Quality:
    """Crude quality check matching the thesis report generator.

    SNR is estimated from the first 100 ms (assumed near-silence) vs the whole signal.
    """
    duration = len(y) / sr if sr else 0.0
    n_noise = min(sr // 10, len(y)) if sr else 0
    noise_std = float(np.std(y[:n_noise])) if n_noise > 0 else 0.0
    signal_std = float(np.std(y)) if len(y) else 0.0
    snr = 20.0 * np.log10(signal_std / (noise_std + 1e-8)) if signal_std > 0 else 0.0

    warnings = []
    if duration < 30:
        warnings.append(f"short recording ({duration:.1f}s < 30s)")
    if snr < 10:
        warnings.append(f"low SNR ({snr:.1f} dB)")
    flag = "high" if (duration > 30 and snr > 10) else "low"
    return Quality(duration_s=duration, snr_db=snr, flag=flag, warnings=warnings)
