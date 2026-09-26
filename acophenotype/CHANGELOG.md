# Changelog

Follows [Semantic Versioning](https://semver.org/).

## [0.1.0] — unreleased

### Added
- `AcousticProfile.from_audio(...)` — end-to-end orchestration: (optional) denoise →
  biomarkers (`voxmarkers`) → deep streams (`acophen-streams`) → fusion
  (`acophen-fusion`) → profile.
- Graceful degradation: with only `voxmarkers` installed you get a biomarker-only
  profile (`prediction` is `None`) instead of an error; missing streams/weights are
  recorded in `profile.notes`.
- `AcousticProfile` API: `prediction`, `probability`, `per_stream`, `streams`,
  `biomarkers`, `flags`, `quality`, `percentile(feature)`, `to_dict`, `to_json`, `to_html`.
- Reference-cohort support (`ReferenceData`): biomarker percentiles and IQR flags,
  emotion z-scores; resolved outside the wheel via `ACOPHENOTYPE_REFERENCE` / auto-discovery.
- **Improved self-contained HTML report** — a single dependency-free file with inline
  CSS and hand-drawn SVG charts (semicircular P(AD) gauge, per-stream contribution bars,
  biomarker percentile bars vs the CN interquartile band, emotion aggregates with
  z-scores, quality strip, provenance, prominent research-use disclaimer). No matplotlib
  needed; an optional log-mel spectrogram embeds if `[report]` is installed.
- Optional MetricGAN+ denoising stage (`[denoise]` extra); passthrough by default.
- CLI: `acophenotype analyze recording.wav -o report.html`.

- **Feature-space handling** for percentiles/flags:
  - reference files may declare their space and carry a raw→reference scaler in `_meta`;
  - `ReferenceData.align()` standardizes a raw patient vector automatically when a scaler
    is present, making the comparison valid;
  - `ReferenceData.space_mismatch()` detects a mismatch via the fraction of features
    falling far outside the reference's observed range, and the report shows a warning
    banner instead of misleading percentiles;
  - `build_reference()` / `scaler_from_features()` / `save_reference()` and the
    `acophenotype build-reference` CLI regenerate reference statistics from your own
    control data, in whichever space you extract in.

### Notes
- Replaces the old three-script report pipeline (`run_7a/7b/7c`): the monolithic
  extraction + `report.json` + matplotlib-base64 HTML is now the composition of the four
  suite packages, and the report is rendered as crisp inline SVG.
- The thesis-shipped `biomarker_cn_stats.json` is in *standardized* space while
  `voxmarkers` emits *raw* values. Regenerate it with `acophenotype build-reference`, or
  attach a scaler, so percentiles and flags are meaningful.
