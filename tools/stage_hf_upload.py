#!/usr/bin/env python3
"""Stage the Hugging Face model repository in the layout the packages download from.

Copies (never moves) the frozen Pitt artefacts into ``hf_upload/``:

    hf_upload/
    ├── README.md                      model card (library versions pinned from this env)
    ├── fusion/
    │   ├── biomarker_scaler.json
    │   ├── stacking/...               binary stacker (+ scaler, config), both variants
    │   ├── final_multiclass_sgl/...   both variants
    │   ├── final_regression_fc2fs/... both variants
    │   └── streams/<stream>/...       the four per-stream models the stacker uses
    └── reference/
        ├── vlad_codebook.pkl
        ├── biomarker_cn_stats.json
        ├── biomarker_scaler.json
        ├── emotion_cn_reference.json
        ├── umap_pitt_clouds.json
        └── umap_reference.npz

The XGBoost stream model is re-saved under the installed xgboost so it loads without the
"older version" warning. The source files are left untouched.

Usage:
    python tools/stage_hf_upload.py --models /mnt/d/research/feature_fusion/models \
        --reference /mnt/d/research/feature_fusion/reference --out hf_upload
"""
from __future__ import annotations

import argparse
import platform
import shutil
import sys
from pathlib import Path

VARIANTS = ("stream_only", "with_demos")
BINARY_STREAM_MODELS = {
    "biomarkers": "LGBMClassifier",
    "embeddings": "XGBClassifier",
    "emotion": "LogisticRegression",
    "vision": "LGBMClassifier",
}
REFERENCE_FILES = [
    "vlad_codebook.pkl", "biomarker_cn_stats.json", "biomarker_scaler.json",
    "emotion_cn_reference.json", "umap_pitt_clouds.json", "umap_reference.npz",
]


def fusion_files():
    for v in VARIANTS:
        yield f"stacking/stacking_binary_{v}.pkl"
        yield f"stacking/stacking_binary_{v}_scaler.pkl"
        yield f"stacking/stacking_binary_{v}_config.json"
        yield f"final_multiclass_sgl/final_multiclass_sgl_{v}.pkl"
        yield f"final_regression_fc2fs/final_regression_fc2fs_{v}.pkl"
    for s, m in BINARY_STREAM_MODELS.items():
        yield f"streams/{s}/binary_{s}_{m}.pkl"
    yield "biomarker_scaler.json"


def _version(mod):
    try:
        return __import__(mod).__version__
    except Exception:
        return "not installed"


MODEL_CARD = """---
license: cc-by-nc-sa-4.0
tags:
  - speech
  - audio
  - acoustic-biomarkers
  - dementia
  - alzheimers
library_name: acophenotype
---

# acophenotype: Pitt-trained models

Frozen models and reference data released with the PhD thesis *Acoustic Phenotyping in Low-Resource Settings: A Multi-Representational Fusion Framework for Alzheimer's Detection*
(Marek Sviderski, University of Sunderland, 2026). They are downloaded automatically by the
[`acophenotype`](https://github.com/tobiasSviderski/acophenotype) Python packages; you do not
need to fetch them by hand.

```bash
pip install acophenotype
```

```python
from acophenotype import AcousticProfile
profile = AcousticProfile.from_audio("recording.wav", task="binary")
```

## Intended use and limitations

**Research use only. Not a medical device and not for diagnosis or any clinical decision.**

- Trained on the Pitt Corpus (DementiaBank): English, predominantly picture-description
  speech from a single study's recording setup.
- Performance has not been established for other languages, elicitation tasks, recording
  conditions or clinical populations.
- A single recording is a noisy observation; outputs should not be interpreted in isolation.

## Contents

| Path | What it is |
|---|---|
| `fusion/stacking/` | Late-fusion stacking (binary): logistic meta-model over per-stream P(AD) |
| `fusion/final_multiclass_sgl/` | Sparse-group-lasso one-vs-rest model (multiclass) |
| `fusion/final_regression_fc2fs/` | Correlation-filtered LightGBM regressor (regression) |
| `fusion/streams/` | The per-stream base models used by the binary stacker |
| `fusion/biomarker_scaler.json` | Pitt z-score parameters applied to biomarkers before scoring |
| `reference/vlad_codebook.pkl` | 16-cluster VLAD codebook for the Whisper embedding stream |
| `reference/*.json`, `umap_reference.npz` | Control-cohort statistics and UMAP layout for the report |

Each model has `stream_only` and `with_demos` (sex, age, education) variants.

## Licence

The models and reference files are released under **CC BY-NC-SA 4.0**
(attribution, non-commercial, share-alike), consistent with the CC BY-NC-SA 3.0 terms
governing the DementiaBank data they were trained on. They contain no audio or
transcripts. The accompanying software is MIT-licensed.

## Training data

Pitt Corpus, DementiaBank (TalkBank). Access to the corpus is governed by DementiaBank's
terms; these files contain trained models and summary statistics, not audio or transcripts.

## Environment

The models are serialized with pickle/joblib and are sensitive to library versions. They
were saved with scikit-learn 1.7.2; the packages pin compatible versions. Staged with:

{versions}

Pickle files execute code when loaded; load them only from this repository.

## Citation

Please cite the thesis and the software repository (see `CITATION.cff` there).
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", required=True, type=Path, help="feature_fusion/models")
    ap.add_argument("--reference", required=True, type=Path, help="feature_fusion/reference")
    ap.add_argument("--out", default=Path("hf_upload"), type=Path)
    args = ap.parse_args()

    out = args.out
    if out.exists():
        sys.exit(f"{out} already exists; remove it first so stale files can't slip in.")
    missing = []

    for rel in fusion_files():
        src = args.models / rel
        dst = out / "fusion" / rel
        if not src.exists():
            missing.append(str(src))
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith("_XGBClassifier.pkl"):
            try:
                import joblib
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = joblib.load(src)
                joblib.dump(model, dst)        # re-serialized under the installed xgboost
                print(f"  re-saved {rel} under xgboost {_version('xgboost')}")
            except Exception as e:
                shutil.copy2(src, dst)
                print(f"  WARNING: could not re-save {rel} ({type(e).__name__}); copied as-is. "
                      f"It still loads, with an 'older XGBoost' warning.")
        else:
            shutil.copy2(src, dst)

    for name in REFERENCE_FILES:
        src = args.reference / name
        if not src.exists():
            missing.append(str(src))
            continue
        (out / "reference").mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out / "reference" / name)

    if missing:
        shutil.rmtree(out, ignore_errors=True)
        sys.exit("Missing files, nothing staged:\n  " + "\n  ".join(missing))

    versions = "\n".join(f"- {k}: {_version(m)}" for k, m in [
        ("python", "platform"), ("scikit-learn", "sklearn"), ("lightgbm", "lightgbm"),
        ("xgboost", "xgboost"), ("numpy", "numpy")]).replace(
        "- python: not installed", f"- python: {platform.python_version()}")
    (out / "README.md").write_text(MODEL_CARD.replace("{versions}", versions), encoding="utf-8")
    (out / "LICENSE").write_text(
        "The model weights and reference files in this repository are licensed under the\n"
        "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License\n"
        "(CC BY-NC-SA 4.0): https://creativecommons.org/licenses/by-nc-sa/4.0/\n\n"
        "They were trained on the Pitt Corpus (DementiaBank, TalkBank), whose data are governed\n"
        "by CC BY-NC-SA 3.0 and the TalkBank Ground Rules "
        "(https://talkbank.org/0share/rules.html).\nNo audio or transcripts are included.\n\n"
        "Copyright (c) 2026 Marek Sviderski\n", encoding="utf-8")

    files = [p for p in out.rglob("*") if p.is_file()]
    size = sum(p.stat().st_size for p in files) / 1e6
    print(f"\nStaged {len(files)} files ({size:.1f} MB) in {out}/")
    print("Next:\n  hf auth login\n"
          "  hf repos create acophenotype-pitt-models --repo-type model --private\n"
          f"  hf upload <your-hf-username>/acophenotype-pitt-models {out} . --repo-type model")


if __name__ == "__main__":
    main()
