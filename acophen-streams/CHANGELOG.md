# Changelog

Follows [Semantic Versioning](https://semver.org/). Stream output schemas carry their
own version (`SCHEMA_VERSION`) because fusion scoring depends on frozen column order.

## [0.1.1] — 2026-09-26

### Fixed
- Author name spelling (Sviderski) and project links in the package metadata.

## [0.1.0] — unreleased

### Added
- Common stream interface (`BaseStream`) with `extract(wav) -> named Series`, plus a
  registry (`get_stream`, `available`, `schema_for`).
- Three streams refactored from the thesis source, each behind the common interface:
  - **ssl** — Whisper-small high layer → VLAD (16 clusters) = 12288 features.
  - **vision** — linear spectrogram → ResNet50 → temporal pyramid (levels=3) = 14336.
  - **emotion** — WavLM 3LOI categorical SER → 15-statistic pooling = 120 (`{emotion}_{stat}`).
- Frozen output schemas (`SCHEMA_VERSION = "1.0"`) verified against the trained fusion
  models (dims and emotion column names match exactly).
- Pure-NumPy pooling functions (`vlad_pool`, `temporal_pyramid_pool`, `statistical_pool`)
  ported verbatim; usable and testable without torch.
- Lazy backbone loading: importing the package does not import torch; each stream loads
  its backbone on first `extract`. Backbones are optional extras (`[ssl]`, `[vision]`,
  `[emotion]`, `[all]`).
- CLI: `acophen-streams list | schema | extract`.

### Notes
- Extraction requires the relevant extra (torch etc.) and, for `ssl`, a fitted VLAD
  codebook (resolved via `ACOPHEN_STREAMS_VLAD_CODEBOOK` or auto-discovery of
  `feature_fusion/reference/vlad_codebook.pkl`). The torch extraction paths were ported
  faithfully from the source but require a GPU/torch environment to run.
