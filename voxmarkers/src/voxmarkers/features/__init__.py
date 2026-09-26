"""Feature-tier modules. Each exposes ``TIER``, ``COLUMNS``, ``process_segment`` and
``aggregate`` behind a common interface so the extractor can iterate over tiers
without special-casing any of them.
"""
from . import core, extension, aerodynamics

# Registry: tier name -> module. Order here defines the canonical tier order.
TIERS = {
    core.TIER: core,
    extension.TIER: extension,
    aerodynamics.TIER: aerodynamics,
}

__all__ = ["core", "extension", "aerodynamics", "TIERS"]
