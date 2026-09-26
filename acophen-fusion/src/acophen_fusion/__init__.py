"""acophen-fusion -- fuse acoustic-phenotyping stream vectors into a prediction.

Research-use software (NOT a medical device). See the README disclaimer.

    from acophen_fusion import Scorer

    scorer = Scorer.load(task="binary", variant="stream_only")
    result = scorer.score(stream_probas={"biomarkers": 0.2, "embeddings": 0.8,
                                         "emotion": 0.5, "vision": 0.4})
    result.prediction, result.probability, result.per_stream
"""
from ._version import __version__
from .scorer import Scorer, FusionResult
from .streams import STREAMS, DEMOGRAPHIC_COLUMNS
from .weights import TASKS, VARIANTS

# Registering the unpickle shims happens on importing compat (pulled in by models).
from . import compat  # noqa: F401,E402

__all__ = [
    "__version__",
    "Scorer",
    "FusionResult",
    "STREAMS",
    "DEMOGRAPHIC_COLUMNS",
    "TASKS",
    "VARIANTS",
]
