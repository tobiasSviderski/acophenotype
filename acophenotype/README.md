# acophenotype

**End-to-end acoustic phenotyping: one recording in, a profile and a polished HTML report out.**

`acophenotype` is the umbrella of the suite. It composes the other three packages —
[`voxmarkers`](../voxmarkers) (interpretable biomarkers),
[`acophen-streams`](../acophen-streams) (deep streams), and
[`acophen-fusion`](../acophen-fusion) (scoring) — into a single `AcousticProfile` and a
self-contained HTML report.

> ⚠️ **Research use only. Not a medical device. Not for diagnosis.**

## What you get

```
recording.wav
   │  (optional denoise: MetricGAN+)
   ▼
biomarkers (voxmarkers) ─┐
ssl · vision · emotion   ├─► acophen-fusion ─► prediction + per-stream contribution
(acophen-streams)        ┘
   ▼
AcousticProfile  ─►  .to_html("report.html")   ·   .to_json(...)
```

It **degrades gracefully**: with only the core installed you still get a biomarker-only
profile (no fused prediction) — no errors, and the report adapts.

## Install

```bash
pip install acophenotype
```

That's it. This installs the **entire suite** (`voxmarkers`, `acophen-fusion`,
`acophen-streams` and the deep backbones). The model weights, VLAD codebook and reference
cohort are **downloaded automatically on first use** and cached, so there is nothing else
to configure.

Two genuinely optional add-ons:

```bash
pip install "acophenotype[denoise]"   # MetricGAN+ speech enhancement
pip install "acophenotype[report]"    # embedded log-mel spectrogram in the report
```

## Quick start

```python
from acophenotype import AcousticProfile

profile = AcousticProfile.from_audio(
    "recording.wav",
    task="binary",              # or "multiclass" / "regression"
    denoise=False,
    streams="all",              # or ["ssl", "vision"] etc.
    demographics=None,          # or {"sex":.., "age":.., "educ":..} for with_demos
)

profile.prediction              # "AD" / "CN" / None (biomarker-only)
profile.probability             # 0.78
profile.per_stream              # {"biomarkers":0.41, "embeddings":0.24, ...}
profile.percentile("temp_syllable_rate")   # vs the Pitt reference cohort
profile.to_html("report.html")  # the report
profile.to_json("report.json")  # machine-readable profile
```

CLI:

```bash
acophenotype analyze recording.wav -o report.html --json report.json
acophenotype analyze recording.wav --task regression --spectrogram
```

## The report

A **single self-contained HTML file** — inline CSS and hand-drawn SVG, no external assets,
no JavaScript, no matplotlib required. It renders:

- a colour-coded verdict + a semicircular **P(AD) gauge**,
- **per-stream contribution** bars (how each stream voted),
- **biomarker percentile** bars against the CN interquartile band, with out-of-range
  features flagged (each labelled with its clinical direction from `voxmarkers`),
- **emotion** aggregates with z-scores vs the reference cohort,
- a quality strip, provenance table, pipeline notes, and a prominent disclaimer.

This replaces the old `run_7a/7b/7c` scripts (monolithic extraction → `report.json` →
matplotlib-base64 HTML). The extraction now comes from the suite packages, and the charts
are crisp inline SVG that scale to any width.

Embed a log-mel spectrogram with `profile.to_html("report.html", spectrogram=True)`
(needs the `[report]` extra).

## Feature space (percentiles & flags)

Percentiles and flags are only meaningful when the patient vector and the reference cohort
are in the **same feature space**. The reference shipped with the thesis
(`biomarker_cn_stats.json`) is in *standardized* space while `voxmarkers` emits *raw*
values — compared directly, nearly every feature flags. There are three paths, and the
package handles all of them:

**1. Regenerate the reference in your extraction space (recommended).**

```bash
acophenotype build-reference \
    --features streams/biomarkers.parquet \
    --labels metadata.parquet --label-column target --control-value 0 \
    --space raw -o biomarker_cn_stats.json
```

**2. Keep a standardized reference, but attach the scaler** so raw vectors are converted
automatically at scoring time:

```python
from acophenotype import build_reference, scaler_from_features, save_reference
scaler = scaler_from_features(raw_features)          # per-feature mean/std
stats  = build_reference(standardized_features, space="standardized", scaler=scaler)
save_reference(stats, "biomarker_cn_stats.json")
```

With a scaler present, `ReferenceData.align()` z-scores the incoming raw vector into the
reference space before comparing — no further action needed.

**3. Neither of the above** — the package detects the mismatch (via the fraction of
features falling far outside the reference's observed range), records it in
`profile.notes`, and shows a warning banner in the report instead of reporting misleading
percentiles.

## Weights & reference data (downloaded automatically)

Model weights live outside the wheel and are fetched from the Hugging Face Hub the first
time you need them, then cached in `~/.cache/huggingface`. Nothing to set up.

Resolution order for each asset: explicit argument → environment variable → local
directory → **automatic download**.

| Asset | Escape hatch (optional) |
|---|---|
| Fusion weights | `ACOPHEN_FUSION_WEIGHTS` |
| VLAD codebook (ssl stream) | `ACOPHEN_STREAMS_VLAD_CODEBOOK` |
| Reference cohort | `ACOPHENOTYPE_REFERENCE` |
| All downloads | `ACOPHEN_NO_DOWNLOAD=1` to work fully offline |
| Different Hub repo / pinned version | `ACOPHEN_HUB_REPO`, `ACOPHEN_HUB_REVISION` |

If an asset genuinely can't be obtained, the profile still builds — it omits the parts that
need it and records why in `profile.notes`.

## License

MIT for the code. Pretrained backbones and model weights carry their own licenses.
