# acophen-fusion

**Fuse acoustic-phenotyping stream vectors into a prediction — with per-stream contributions.**

`acophen-fusion` takes the feature vectors produced by the acoustic streams
(`biomarkers` from [`voxmarkers`](https://github.com/tobiasSviderski/acophenotype/tree/main/voxmarkers), plus the
deep `embeddings`, `emotion`, and `vision` streams) and runs the **frozen** models from
the thesis to produce a prediction and a breakdown of how much each stream contributed.
It's built for zero-shot cross-corpus scoring: align an incoming recording to the exact
columns each model was trained on, then score.

> ⚠️ **Research use only. This is not a medical device and must not be used for
> diagnosis.**

---

## What it does

| Task | Model | Input | Output |
|------|-------|-------|--------|
| `binary` | late-fusion **stacking** | per-stream P(AD) → logistic meta-model | P(AD), AD/CN |
| `multiclass` | **SGL** one-vs-rest | concatenated stream feature matrix | class + probabilities |
| `regression` | **FC2FS** → LGBM | concatenated stream feature matrix | scalar (e.g. severity score) |

Each task ships a `stream_only` and a `with_demos` variant (the latter also uses
`sex`, `age`, `educ`). Prediction routines are faithful ports of the thesis
`evaluation/common/zeroshot.py`.

## Install

```bash
pip install acophen-fusion
# optional, only for the full binary path from raw embeddings vectors:
pip install "acophen-fusion[xgboost]"
```

Depends on `scikit-learn`, `lightgbm`, `numpy`, `pandas`, `scipy`, `joblib`. **No torch.**

## Model weights live outside the wheel

The frozen weights are resolved at load time (they are not shipped in the package). Point
the library at your models directory in any of these ways:

```bash
export ACOPHEN_FUSION_WEIGHTS=/path/to/feature_fusion/models
```
```python
Scorer.load("binary", weights_dir="/path/to/feature_fusion/models")
```

Auto-discovery also looks for a `models/` or `feature_fusion/models/` directory near the
current working directory. A future release will lazy-download a versioned bundle from
Zenodo/HuggingFace (see `weights.WEIGHTS_BUNDLE`).

## Quick start

The simplest path — fuse per-stream probabilities you already have:

```python
from acophen_fusion import Scorer

scorer = Scorer.load(task="binary", variant="stream_only")
result = scorer.score(stream_probas={
    "biomarkers": 0.20, "embeddings": 0.80, "emotion": 0.50, "vision": 0.40,
})
result.prediction      # "AD"
result.probability     # 0.78
result.per_stream      # {'biomarkers': 0.31, 'embeddings': 0.44, ...} contribution shares
```

From raw **named** stream feature vectors (the fusion layer aligns by feature name):

```python
import voxmarkers
bvec = voxmarkers.extract("recording.wav")     # biomarkers stream, a named Series

result = scorer.score(streams={
    "biomarkers": bvec,          # pandas Series / dict of named features
    "embeddings": evec,
    "emotion":    mvec,
    "vision":     vvec,
})
```

Multiclass and regression consume the concatenated matrix directly:

```python
mc = Scorer.load("multiclass").score(streams={...})
mc.probabilities        # {0.0: .., 1.0: .., 2.0: ..}

rg = Scorer.load("regression").score(streams={...})
rg.prediction           # a float; rg.probability is None
```

CLI:

```bash
acophen-fusion info
acophen-fusion score-binary --biomarkers 0.2 --embeddings 0.8 --emotion 0.5 --vision 0.4
```

## The feature-name contract

Zero-shot scoring is only valid if features arrive in exactly the order each model was
trained on. `acophen-fusion` never trusts positional order — it reindexes every incoming
vector onto the model's frozen columns (`scorer.expected_columns`), dropping extras and
imputing anything missing. Pair these weights with the `voxmarkers` schema version they
were trained against.

> **Version note:** the shipped weights were trained on a **48-feature** biomarker set,
> while current `voxmarkers` emits 76. Alignment is by name, so it degrades gracefully
> (uses the overlapping biomarker features), but the biomarker contribution reflects only
> the overlapping columns. Retrain the fusion layer to use all 76.

## Per-stream contributions

- **binary** — exact: the stacker is linear, so each stream's contribution is
  `coefficient × scaled P(AD)`.
- **multiclass** — exact, per-sample: each stream's summed contribution to the predicted
  class's logit.
- **regression** — FC2FS/LGBM is non-linear, so `per_stream` is a **global** attribution
  from grouped LGBM feature importances, not a per-sample decomposition.

`result.attribution_method` records which was used.

## Where this sits

`acophen-fusion` is the scoring layer of the acoustic phenotyping suite:
`voxmarkers` + `acophen-streams` → **`acophen-fusion`** → `acophenotype` (umbrella).

## License

MIT for the code. See `LICENSE`. Model weights carry the terms of their training data.
