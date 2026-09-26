# Changelog

Follows [Semantic Versioning](https://semver.org/).

## [0.1.0] — unreleased

### Added
- `Scorer.load(task, variant)` and `Scorer.score(...)` returning a `FusionResult`
  (`prediction`, `probability`, `per_stream` contributions, `probabilities`, `classes`).
- Three frozen fusion models, faithful to `evaluation/common/zeroshot.py`:
  - **binary** — late-fusion stacking (per-stream P(AD) → logistic meta-model).
  - **multiclass** — SGL one-vs-rest over the concatenated stream feature matrix.
  - **regression** — FC2FS (correlation filter) → LGBM over the concatenated matrix.
- `stream_only` and `with_demos` variants for every task.
- Name-based feature-contract alignment: incoming vectors are reindexed to each model's
  frozen training columns; extras dropped, missing imputed.
- Per-stream contributions: exact linear attribution for binary and multiclass; grouped
  LGBM importance (global) for regression.
- Weights resolver — model artifacts live **outside** the wheel (env var / arg /
  auto-discovery), with a placeholder Zenodo/HF bundle manifest.
- Unpickle compatibility shim (`compat.py`) so the FC2FS pipeline, pickled against a
  thesis training module, loads cleanly.
- CLI: `acophen-fusion score-binary` and `acophen-fusion info`.

### Notes
- The shipped weights were trained on a **48-feature** biomarker set; current
  `voxmarkers` emits 76. Alignment is by name and degrades gracefully — record the
  `voxmarkers` schema version paired with these weights in release notes.
