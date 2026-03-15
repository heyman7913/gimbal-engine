"""cuda-motion-flow: classical-vs-learned camera-motion estimation for video stabilization."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "2.1.0"

__all__ = [
    "ClassicalEstimator",
    "Estimator",
    "LearnedEstimator",
    "MotionField",
    "StabilizeResult",
    "__version__",
    "stabilize",
]

# public name -> (submodule, attribute), resolved lazily so `import gimbal` (and the
# CLI's --help) stays cheap and does not pull in torch/cupy until an estimator is actually used.
_LAZY = {
    "Estimator": ("estimators.base", "Estimator"),
    "ClassicalEstimator": ("estimators.classical", "ClassicalEstimator"),
    "LearnedEstimator": ("estimators.learned", "LearnedEstimator"),
    "MotionField": ("motion", "MotionField"),
    "StabilizeResult": ("pipeline", "StabilizeResult"),
    "stabilize": ("pipeline", "stabilize"),
}

if TYPE_CHECKING:
    from .estimators.base import Estimator
    from .estimators.classical import ClassicalEstimator
    from .estimators.learned import LearnedEstimator
    from .motion import MotionField
    from .pipeline import StabilizeResult, stabilize


def __getattr__(name: str) -> Any:
    try:
        module, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(f".{module}", __name__), attr)
    globals()[name] = value  # cache, so the next lookup is a plain global
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
