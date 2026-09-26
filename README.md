# acophenotype

**Multi-representation acoustic phenotyping of spontaneous speech, as installable Python
packages.**

This repository releases the software developed for the PhD thesis *[thesis title]*
(Marek Svidersky, [University], 2026). The thesis studies whether the acoustic channel of
spontaneous speech, analysed through several complementary representations, carries
signal relevant to Alzheimer's disease. The packages let others compute the same
representations on their own recordings and apply the models trained in the thesis.

> **Research use only.** This is not a medical device and must not be used for diagnosis
> or any clinical decision.

## Install

```bash
pip install acophenotype
```

This installs the whole suite. The pretrained models, VLAD codebook and reference-cohort
statistics are downloaded automatically from the
[Hugging Face Hub](https://huggingface.co/msvidersky/acophenotype-pitt-models) the first
time they are needed, then cached.

To use only the interpretable biomarkers (no deep-learning dependencies):

```bash
pip install voxmarkers
```

## Quick start

```python
from acophenotype import AcousticProfile

profile = AcousticProfile.from_audio("recording.wav", task="binary")
profile.prediction      # "AD" or "CN"
profile.probability     # P(AD) from the fused model
profile.per_stream      # contribution of each representation
profile.to_html("report.html")
```

Or from the command line:

```bash
acophenotype analyze recording.wav -o report.html
```

## The packages

| Package | What it does | Heavy dependencies |
|---|---|---|
| [`voxmarkers`](voxmarkers/) | 76 interpretable acoustic biomarkers (temporal, prosodic, articulatory, phonatory, spectral, complexity), each documented with its clinical rationale | none |
| [`acophen-streams`](acophen-streams/) | Learned representations: Whisper-VLAD embeddings, spectrogram-ResNet50 vision features, WavLM emotion statistics | PyTorch |
| [`acophen-fusion`](acophen-fusion/) | The frozen fusion models from the thesis (stacking, SGL, FC2FS) with per-stream contributions | scikit-learn, LightGBM, XGBoost |
| [`acophenotype`](acophenotype/) | End-to-end pipeline: extraction, fusion, reference-cohort comparison and a self-contained HTML report | all of the above |

```
recording.wav
    │
    ├── voxmarkers ───────────── biomarkers ──┐
    └── acophen-streams ── ssl / vision / emotion ─┤
                                                   ▼
                                           acophen-fusion ──► prediction + per-stream contributions
                                                   │
                                           acophenotype ──► AcousticProfile + HTML report
```

## Models and training data

The models were trained on the **Pitt Corpus** (DementiaBank): English, predominantly
picture-description speech. They are released on the Hugging Face Hub, not in this
repository. Performance has not been established for other languages, elicitation tasks,
recording setups or clinical populations. See the model card for details.

## Development

```bash
git clone https://github.com/msvidersky/acophenotype
cd acophenotype
pip install -e ./voxmarkers -e ./acophen-fusion -e "./acophen-streams[all]" -e ./acophenotype
pip install pytest soundfile
cd voxmarkers && pytest      # and likewise for each package
```

## Citation

If you use this software or the models, please cite the thesis and this repository
(see [`CITATION.cff`](CITATION.cff); the archived release has a Zenodo DOI).

## License

MIT for the code. Pretrained backbones (Whisper, ResNet50, WavLM) and the released models
are subject to their own licences and to the terms of the training data.
