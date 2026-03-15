"""cuda-motion-flow: classical-vs-learned camera-motion estimation for video stabilization."""

from .estimators.base import Estimator
from .estimators.classical import ClassicalEstimator
from .estimators.learned import LearnedEstimator
from .motion import MotionField
from .pipeline import StabilizeResult, stabilize

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
