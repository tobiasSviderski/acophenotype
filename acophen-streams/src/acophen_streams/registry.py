"""Stream registry: one place to discover and construct streams by name."""
from __future__ import annotations

from .ssl import SSLStream
from .vision import VisionStream
from .emotion import EmotionStream

# name -> stream class. The classes are lightweight; backbones load lazily on first use.
_STREAMS = {
    "ssl": SSLStream,
    "vision": VisionStream,
    "emotion": EmotionStream,
}


def available() -> list[str]:
    """Names of all registered streams (installed or not)."""
    return list(_STREAMS)


def get_stream(name: str, **kwargs):
    """Construct a stream by name.

    Extra keyword args are passed to the stream constructor, e.g.
    ``get_stream("ssl", codebook_path="vlad_codebook.pkl")`` or
    ``get_stream("vision", device="cpu")``.
    """
    if name not in _STREAMS:
        raise KeyError(f"Unknown stream {name!r}. Available: {available()}")
    return _STREAMS[name](**kwargs)


def schema_for(name: str):
    """Return a stream's frozen output schema without constructing/loading it."""
    if name not in _STREAMS:
        raise KeyError(f"Unknown stream {name!r}. Available: {available()}")
    return _STREAMS[name].schema
