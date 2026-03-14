"""The classical estimator: find corners, track them with LK, fit a homography with RANSAC."""

from __future__ import annotations

import numpy as np

from .._gpu import require_cuda
from ..cuda.optical_flow import compute_flow
from ..cuda.ransac import ransac_homography
from ..motion import MotionField
from .base import Estimator


class ClassicalEstimator(Estimator):
    name = "classical"

    def __init__(self, n_iter: int = 500, ransac_threshold: float = 3.0) -> None:
        require_cuda()
        self.n_iter = n_iter
        self.ransac_threshold = ransac_threshold

    def warmup(self, height: int, width: int) -> None:
        # run once up front so the kernels compile here and not during timing
        dummy = np.zeros((height, width), dtype=np.uint8)
        dummy[:: max(1, height // 16), :] = 255  # draw a grid so there are corners to find
        dummy[:, :: max(1, width // 16)] = 255
        self.estimate(dummy, dummy)

    def estimate(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> MotionField:
        import cupy as cp

        self._validate_pair(prev_gray, curr_gray)
        prev = cp.asarray(prev_gray, dtype=cp.float32)
        curr = cp.asarray(curr_gray, dtype=cp.float32)

        src, dst = compute_flow(prev, curr)
        if int(src.shape[0]) < 4:
            return MotionField.global_(np.eye(3))

        h, mask = ransac_homography(src, dst, n_iter=self.n_iter, threshold=self.ransac_threshold)
        if int(mask.sum()) < 4:
            return MotionField.global_(np.eye(3))
        return MotionField.global_(cp.asnumpy(h).astype(np.float64))
