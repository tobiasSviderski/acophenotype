"""Frozen output schemas for each stream.

These column names and dimensions ARE the feature-name contract that ``acophen-fusion``
consumes (as ``"{stream}__{column}"``). They were verified against the trained fusion
models: embeddings=12288, vision=14336, emotion=120. **Do not reorder or resize without
bumping the schema version** -- doing so invalidates zero-shot scoring.
"""
from __future__ import annotations

from dataclasses import dataclass

from .pooling import STAT_NAMES

SCHEMA_VERSION = "1.0"

# Emotion labels emitted by the WavLM 3LOI categorical model, matched to the fusion
# contract's 8 classes (alphabetical). Runtime output is aligned to these by name.
EMOTION_LABELS = [
    "angry", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise",
]


def _feat_columns(n: int) -> list[str]:
    """Zero-padded ``feat_0000 .. feat_{n-1}`` names (4 digits, matching the source)."""
    return [f"feat_{i:04d}" for i in range(n)]


def _emotion_columns() -> list[str]:
    return [f"{emo}_{stat}" for emo in EMOTION_LABELS for stat in STAT_NAMES]


@dataclass(frozen=True)
class StreamSchema:
    name: str
    version: str
    columns: tuple

    @property
    def dim(self) -> int:
        return len(self.columns)

    def __len__(self) -> int:
        return len(self.columns)


# SSL / embeddings: Whisper-small high layer (768-d) -> VLAD (16 clusters) = 12288.
SSL_SCHEMA = StreamSchema("ssl", SCHEMA_VERSION, tuple(_feat_columns(12288)))

# Vision: linear-spectrogram -> ResNet50 (2048-d) -> temporal pyramid (levels=3, 7 bins) = 14336.
VISION_SCHEMA = StreamSchema("vision", SCHEMA_VERSION, tuple(_feat_columns(14336)))

# Emotion: WavLM 3LOI categorical (8 classes) -> 15 statistics = 120, named {emotion}_{stat}.
EMOTION_SCHEMA = StreamSchema("emotion", SCHEMA_VERSION, tuple(_emotion_columns()))

SCHEMAS = {
    "ssl": SSL_SCHEMA,
    "vision": VISION_SCHEMA,
    "emotion": EMOTION_SCHEMA,
}
