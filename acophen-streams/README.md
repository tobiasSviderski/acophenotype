# acophen-streams

**Deep acoustic feature streams behind one interface — install only the ones you need.**

`acophen-streams` provides the three learned streams of the acoustic phenotyping suite,
each producing a **named, schema-versioned** feature vector that plugs straight into
[`acophen-fusion`](https://github.com/tobiasSviderski/acophenotype/tree/main/acophen-fusion):

| Stream | Import name | Backbone → pooling | Dim |
|--------|-------------|--------------------|-----|
| SSL / embeddings | `ssl` | Whisper-small high layer → VLAD (16 clusters) | 12288 |
| Vision | `vision` | linear spectrogram → ResNet50 → temporal pyramid (levels 3) | 14336 |
| Emotion | `emotion` | WavLM 3LOI categorical SER → 15-statistic pooling | 120 |

> ⚠️ **Research use only. Not a medical device.**

## Install

The core (registry, schemas, pooling) is light and torch-free. Backbones are optional
extras — install only the streams you need:

```bash
pip install acophen-streams              # core: registry + schemas + pooling
pip install "acophen-streams[ssl]"       # + Whisper (torch, transformers)
pip install "acophen-streams[vision]"    # + ResNet50 (torch, torchvision)
pip install "acophen-streams[emotion]"   # + WavLM SER (torch, torchaudio, transformers)
pip install "acophen-streams[all]"       # everything
```

Importing the package does **not** import torch; each backbone loads lazily on first
`extract`.

## Quick start

```python
from acophen_streams import get_stream, available

available()                        # ['ssl', 'vision', 'emotion']

ssl = get_stream("ssl")            # backbone loads on first use
vec = ssl.extract("recording.wav") # named pandas Series, len 12288, schema-versioned
ssl.schema.version                 # '1.0'

emo = get_stream("emotion")
emo.extract("recording.wav")       # 120 named features: angry_mean, ..., neutral_cv
```

Every stream implements the same interface — `extract(wav_or_array) -> named Series` and
a frozen `schema`. That is what lets the fusion layer consume them by name.

Feed straight into fusion:

```python
import acophen_fusion, acophen_streams as A
streams = {name.replace("ssl", "embeddings"): A.get_stream(name).extract("rec.wav")
           for name in ("ssl", "vision", "emotion")}
# add the biomarkers stream from voxmarkers, then score with acophen_fusion.Scorer
```

CLI:

```bash
acophen-streams list                 # streams and their dimensions
acophen-streams schema emotion       # print the 120 frozen column names
acophen-streams extract ssl rec.wav  # needs the [ssl] extra
```

## The SSL VLAD codebook

The `ssl` stream needs a fitted VLAD codebook (`kmeans .pkl`, 16 clusters × 768). Point to
it via the `ACOPHEN_STREAMS_VLAD_CODEBOOK` environment variable, pass
`get_stream("ssl", codebook_path=...)`, or let auto-discovery find
`feature_fusion/reference/vlad_codebook.pkl`. Ship the codebook with the model bundle
(Zenodo/HF), not inside the wheel.

## Frozen schemas

Stream output columns and order are a versioned contract (`SCHEMA_VERSION`), verified
against the trained fusion models (dims 12288 / 14336 / 120, and the emotion column names
`{emotion}_{stat}`, match exactly). Changing them is a breaking change for fusion — bump
the version.

## Notes

- The pooling functions (`vlad_pool`, `temporal_pyramid_pool`, `statistical_pool`) are
  pure NumPy ports of the thesis flattening scripts and are tested directly.
- **Licensing:** Whisper (MIT) and torchvision ResNet (BSD) are fine to build on; verify
  the WavLM SER model's license before redistributing its weights.
- The torch extraction paths are faithful ports of the source pipelines but require a
  torch environment (and, ideally, a GPU) to run.

## License

MIT for the code. Pretrained backbones carry their own licenses.
