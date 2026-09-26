"""Registry behaviour and lazy loading (no torch required)."""
from __future__ import annotations

import sys

import pytest

from acophen_streams import available, get_stream, schema_for
from acophen_streams.base import BaseStream


def test_available_lists_three_streams():
    assert available() == ["ssl", "vision", "emotion"]


def test_get_stream_returns_basestream_instances():
    for name in available():
        s = get_stream(name)
        assert isinstance(s, BaseStream)
        assert s.name == name
        assert s.schema is schema_for(name)


def test_get_unknown_stream_raises():
    with pytest.raises(KeyError):
        get_stream("nope")
    with pytest.raises(KeyError):
        schema_for("nope")


def test_importing_package_does_not_import_torch():
    # Constructing streams must not pull in torch (backbones are lazy).
    for name in available():
        get_stream(name)
    assert "torch" not in sys.modules, "torch was imported too eagerly"


def test_extract_without_backend_raises_actionable_error():
    """When the backbone dep is absent, extract() raises a clear, install-pointing error."""
    import importlib.util
    import numpy as np
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch is installed; backend-missing path not exercised here")

    from acophen_streams.base import MissingBackendError
    ssl = get_stream("ssl")
    # valid array so audio loading succeeds; failure must come from the missing backbone
    y = np.zeros(16000, dtype="float32")
    with pytest.raises(MissingBackendError) as ei:
        ssl.extract(y, sr=16000)
    assert "acophen-streams[" in str(ei.value)
