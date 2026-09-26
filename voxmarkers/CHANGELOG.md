# Changelog

All notable changes to `voxmarkers` are documented here. This project follows
[Semantic Versioning](https://semver.org/). The **feature schema** carries its own
independent version (`FEATURE_SCHEMA.version`) because zero-shot scoring depends on a
frozen column order — see the README.

## [0.1.1] — 2026-09-26

### Fixed
- Author name spelling (Sviderski) and project links in the package metadata.

## [0.1.0] — unreleased

### Added
- Initial public API: `extract`, `describe`, `FEATURE_SCHEMA`.
- Three feature tiers refactored from the thesis extraction scripts:
  - **core** (`1a_core_features.py`) — temporal, prosodic, articulatory, phonatory, spectral, MFCC.
  - **extension** (`1b_extension.py`) — spectral tilt/shape, signal complexity, voice quality, rhythm, wavelet energy.
  - **aerodynamics** (`1c_aerodynamics.py`) — H1–H2, MFCC dynamics, Teager energy, formant dispersion, voicing.
- Frozen, versioned feature schema (`schema.py`) — feature-name contract at **schema version 1.0**.
- Provenance table (`provenance.py`): family, unit, direction-in-AD, rationale, and seed references for every feature.
- Per-recording `QualityReport` (too-short / silent / clipped detection).
- Command-line interface: `voxmarkers extract <wav...> -o features.csv`.
- Test suite covering the schema contract, extraction shape, and provenance completeness.

### Notes
- Reference citations in `provenance.py` are **canonical seed references** and must be
  reconciled against the PRISMA-ScR scoping review before publication.
