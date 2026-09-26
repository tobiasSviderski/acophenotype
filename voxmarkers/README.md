# voxmarkers

**Interpretable acoustic biomarkers of speech, in one line — no deep-learning stack required.**

`voxmarkers` computes ~76 documented, clinically-motivated acoustic features from a
single speech recording. Every feature knows its family, its unit, its expected
direction of change in Alzheimer's disease, and the reference(s) that motivate it.
It is the software form of a scoping review of interpretable acoustic biomarkers for
neurodegenerative disease, extracted from spontaneous speech.

> ⚠️ **Research use only. This is not a medical device and must not be used for
> diagnosis.** Outputs are acoustic measurements for research, not clinical decisions.

---

## Why this exists

Most speech-based dementia work bolts NLP/transcript features on top of the acoustics.
This package deliberately keeps the spotlight on the **acoustic channel alone**, which
is what makes it usable in low-resource and cross-lingual settings where transcripts
don't exist. `voxmarkers` is the lightweight, interpretable core of a larger acoustic
phenotyping suite (see [The package family](#the-package-family)).

## Install

```bash
pip install voxmarkers
```

From source (development):

```bash
git clone https://github.com/msvidersky/voxmarkers
cd voxmarkers
pip install -e ".[dev]"
```

Depends only on `numpy`, `pandas`, `scipy`, `librosa`, `praat-parselmouth`, `antropy`,
and `PyWavelets`. **No torch.**

## Quick start

```python
from voxmarkers import extract, describe, FEATURE_SCHEMA

# 1. One call: WAV -> named pandas Series (schema-ordered)
feats = extract("recording.wav")
feats["temp_syllable_rate"]        # e.g. 3.9

# 2. Provenance for any feature
describe("art_fcr")
# {'name': 'art_fcr', 'family': 'articulatory', 'tier': 'core', 'unit': 'ratio',
#  'direction_in_ad': 'increases', 'description': 'Formant Centralization Ratio ...',
#  'references': ['Sapir et al. (2010), ...']}

# 3. The frozen feature-name contract
FEATURE_SCHEMA.version             # '1.0'
FEATURE_SCHEMA.columns             # canonical ordered feature list (len 76)
```

Pick specific tiers, get a quality report, or process a folder:

```python
feats = extract("recording.wav", tiers=("core", "extension"))

feats, quality = extract("recording.wav", return_quality=True)
quality.ok, quality.warnings       # e.g. (False, ['recording appears silent ...'])

from voxmarkers import extract_batch
df = extract_batch(["a.wav", "b.wav"], tiers="core")   # DataFrame, one row per file
```

Or from the command line:

```bash
voxmarkers extract *.wav -o features.csv
voxmarkers describe art_fcr
voxmarkers describe -o provenance_table.csv   # dump the whole provenance table
```

You can also pass a raw waveform: `extract(y, sr=16000)`.

## The feature tiers

Every recording is resampled to mono 16 kHz; long recordings are processed in 60-second
chunks and aggregated (rates from summed counts, other features averaged).

| Tier | Import name | Count | Families |
|------|-------------|-------|----------|
| **core** | `"core"` | 53 | temporal, prosodic, articulatory, phonatory, spectral, MFCC |
| **extension** | `"extension"` | 14 | spectral tilt/shape, signal complexity, voice quality, rhythm, wavelet energy |
| **aerodynamics** | `"aerodynamics"` | 9 | H1–H2, MFCC dynamics, Teager energy, formant dispersion, voicing |

See `voxmarkers describe` or `provenance_frame()` for the full annotated list.

## The frozen feature-name contract

Zero-shot / cross-corpus scoring against a previously-trained model is only valid if
features come out in **exactly** the order the model was trained on. `voxmarkers` treats
that column order as a versioned contract:

- `FEATURE_SCHEMA.version` is bumped on **any** change to columns or their order.
- `tests/test_schema.py` pins the exact ordered contract and fails CI if it drifts.
- Requesting tiers in any order still yields output in canonical order.

If you retrain models against this feature set, record `FEATURE_SCHEMA.version`
alongside your weights.

## A note on references

The `references` in the provenance table are **canonical seed references** — the origin
paper for each measure and the best-known clinical-speech sources. They are a correct
starting point but were **not** auto-extracted from the PRISMA-ScR scoping review.
Before citing this package in a publication, reconcile the reference list in
`src/voxmarkers/provenance.py` against the exact rows of that review. Where the
AD-specific direction of change was uncertain, `direction_in_ad` is `"altered"` rather
than a possibly-wrong sign.

## The package family

`voxmarkers` is the standalone, interpretable core of a four-package acoustic
phenotyping suite:

- **`voxmarkers`** (this package) — interpretable biomarkers, light deps.
- `acophen-fusion` — turn stream vectors into a prediction + per-stream contributions.
- `acophen-streams` — deep streams (SSL embeddings, vision, emotion).
- `acophenotype` — end-to-end umbrella: denoise → streams → fuse → profile/report.

Each package implements the same `extract(wav) -> named vector` interface so the fusion
layer can consume them through a registry.

## Citation

A Zenodo DOI will be minted on first release. Until then, please cite the thesis and
the companion scoping review. See `CITATION.cff`.

## License

MIT for the code (see `LICENSE`). Pretrained backbones used by sibling packages carry
their own licenses.
