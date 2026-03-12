"""Learned estimator: the trained IHN wrapped behind the shared Estimator interface.

Frames are resized to the network's square patch size; the homography recovered at that scale
is mapped back to full resolution by the (anisotropic) resize transform S, H_full = S^-1 H S.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .._gpu import require_cuda
from ..learned.model import IHN
from .base import Estimator


class LearnedEstimator(Estimator):
    name = "ihn"

    def __init__(
        self,
        weights_path: str | Path | None = None,
        iters: int = 6,
        size: int = 128,
        use_fused: bool = False,
    ) -> None:
        import torch

        require_cuda()
        self.size = size
        self.model = IHN(size=size, iters=iters).cuda().eval()
        self.model.use_fused = use_fused
        if weights_path is not None:
            state = torch.load(str(weights_path), map_location="cuda")
            self.model.load_state_dict(state)

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.model.load_state_dict(state)

    def warmup(self, height: int, width: int) -> None:
        dummy = np.zeros((height, width), dtype=np.uint8)
        self.estimate(dummy, dummy)

    def estimate(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
        import torch

        self._validate_pair(prev_gray, curr_gray)
        h0, w0 = prev_gray.shape
        # The network was trained on undistorted square patches, so feed it an undistorted
        # centre square crop rather than an anisotropic full-frame resize. The estimated
        # camera motion is global, so it applies to the whole frame.
        side = min(h0, w0)
        cx0, cy0 = (w0 - side) // 2, (h0 - side) // 2
        a_crop = prev_gray[cy0 : cy0 + side, cx0 : cx0 + side]
        b_crop = curr_gray[cy0 : cy0 + side, cx0 : cx0 + side]
        a = cv2.resize(a_crop, (self.size, self.size), interpolation=cv2.INTER_AREA)
        b = cv2.resize(b_crop, (self.size, self.size), interpolation=cv2.INTER_AREA)

        with torch.no_grad():
            ta = torch.from_numpy(a).cuda().float().div_(255.0)[None, None]
            tb = torch.from_numpy(b).cuda().float().div_(255.0)[None, None]
            h_sq = self.model.predict(ta, tb)[0].double().cpu().numpy()

        # T maps full-frame pixels into the square-crop grid: translate to the crop, then scale
        scale = self.size / side
        t = np.array([[scale, 0, -scale * cx0], [0, scale, -scale * cy0], [0, 0, 1.0]])
        h_full = np.linalg.inv(t) @ h_sq @ t
        h_full = h_full / h_full[2, 2]

        # real inter-frame camera motion is near-rigid; reject a degenerate prediction and fall
        # back to identity for that frame. Checks: affine anisotropy, area scale, and the
        # perspective magnitude across the frame (h31, h32) which otherwise accumulates into
        # severe shear over a clip.
        sv = np.linalg.svd(h_full[:2, :2], compute_uv=False)
        perspective = abs(h_full[2, 0]) * w0 + abs(h_full[2, 1]) * h0
        if (
            sv[1] < 1e-6
            or sv[0] / sv[1] > 1.5
            or not (0.5 < sv[0] * sv[1] < 2.0)
            or perspective > 0.05
        ):
            return np.eye(3, dtype=np.float64)
        result: np.ndarray = h_full
        return result
