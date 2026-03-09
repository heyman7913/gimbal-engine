"""The keystone interface. The pipeline and benchmark treat every estimator identically:
hand it two consecutive grayscale frames, get back the homography between them.
"""

from __future__ import annotations

import abc

import numpy as np


class Estimator(abc.ABC):
    """Estimates the camera motion between two consecutive frames as a 3x3 homography.

    Contract:
      - input: two single-channel grayscale frames, same shape (H, W), as host numpy
        arrays. Dtype is uint8 or float32; the estimator handles conversion internally.
      - output: a (3, 3) float64 homography H that maps frame-A pixel coordinates to
        frame-B pixel coordinates (homogeneous), normalized so H[2, 2] == 1.
      - all compute happens on the GPU; there is no CPU fallback.
    """

    name: str

    @abc.abstractmethod
    def estimate(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
        """Return the 3x3 homography mapping prev_gray coords to curr_gray coords."""

    def warmup(self, height: int, width: int) -> None:
        """Optional one-time setup (kernel JIT, weight load) so it is excluded from timing.

        Default is a no-op. Estimators with first-call compilation cost should override.
        """
        return None

    @staticmethod
    def _validate_pair(prev_gray: np.ndarray, curr_gray: np.ndarray) -> None:
        if prev_gray.ndim != 2 or curr_gray.ndim != 2:
            raise ValueError("estimators expect single-channel (H, W) grayscale frames")
        if prev_gray.shape != curr_gray.shape:
            raise ValueError(f"frame shapes differ: {prev_gray.shape} vs {curr_gray.shape}")
