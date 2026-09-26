"""acophen-streams -- deep acoustic feature streams behind one interface.

Research-use software (NOT a medical device). See the README disclaimer.

    from acophen_streams import get_stream, available

    available()                       # ['ssl', 'vision', 'emotion']
    ssl = get_stream("ssl")           # loads the backbone lazily on first extract
    vec = ssl.extract("recording.wav")  # named, schema-versioned pandas Series
    ssl.schema.version

Backbones are optional extras: install only the streams you need, e.g.
``pip install "acophen-streams[ssl]"``. Importing this package does not import torch.
"""
from ._version import __version__
from .registry import available, get_stream, schema_for
from .schemas import SCHEMAS, SCHEMA_VERSION, EMOTION_LABELS
from .base import BaseStream, MissingBackendError

__all__ = [
    "__version__",
    "available",
    "get_stream",
    "schema_for",
    "SCHEMAS",
    "SCHEMA_VERSION",
    "EMOTION_LABELS",
    "BaseStream",
    "MissingBackendError",
]
