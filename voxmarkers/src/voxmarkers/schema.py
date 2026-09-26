"""The feature-name contract.

Zero-shot / cross-corpus scoring is only valid if features come out in exactly
the order the downstream fusion models were trained on. This module is the single
source of truth for that order. **Do not reorder or rename columns** without
bumping :data:`SCHEMA_VERSION` -- a change here is a breaking change for every
model trained against a previous version.
"""
from __future__ import annotations

from dataclasses import dataclass

from .features import core, extension, aerodynamics

#: Bump this (and CHANGELOG) whenever the columns below change in any way.
SCHEMA_VERSION = "1.0"

#: Column identifier for the recording name in batch output.
ID_COLUMN = "recording_name"

#: Canonical, ordered feature columns for each tier.
TIER_COLUMNS: dict[str, list[str]] = {
    core.TIER: list(core.COLUMNS),
    extension.TIER: list(extension.COLUMNS),
    aerodynamics.TIER: list(aerodynamics.COLUMNS),
}

#: Default tier selection (everything), in canonical order.
DEFAULT_TIERS: tuple[str, ...] = (core.TIER, extension.TIER, aerodynamics.TIER)


def columns_for(tiers) -> list[str]:
    """Ordered feature columns for the requested tiers.

    Tiers are always emitted in canonical order (core, extension, aerodynamics)
    regardless of the order requested, so the contract is stable.
    """
    requested = set(tiers)
    unknown = requested - set(TIER_COLUMNS)
    if unknown:
        raise ValueError(
            f"Unknown tier(s): {sorted(unknown)}. Valid tiers: {sorted(TIER_COLUMNS)}"
        )
    cols: list[str] = []
    for tier in DEFAULT_TIERS:
        if tier in requested:
            cols.extend(TIER_COLUMNS[tier])
    return cols


@dataclass(frozen=True)
class FeatureSchema:
    """Immutable description of the frozen feature-name contract."""

    version: str
    tiers: tuple[str, ...]

    @property
    def columns(self) -> list[str]:
        """The full canonical ordered column list across all tiers."""
        return columns_for(self.tiers)

    def columns_for(self, tiers) -> list[str]:
        return columns_for(tiers)

    def tier_of(self, feature: str) -> str:
        for tier, cols in TIER_COLUMNS.items():
            if feature in cols:
                return tier
        raise KeyError(f"{feature!r} is not part of schema v{self.version}")

    def __len__(self) -> int:
        return len(self.columns)

    def __contains__(self, feature: str) -> bool:
        return any(feature in cols for cols in TIER_COLUMNS.values())


#: The public, importable schema object.
FEATURE_SCHEMA = FeatureSchema(version=SCHEMA_VERSION, tiers=DEFAULT_TIERS)
