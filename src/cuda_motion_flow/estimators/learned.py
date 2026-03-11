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
        a = cv2.resize(prev_gray, (self.size, self.size), interpolation=cv2.INTER_AREA)
        b = cv2.resize(curr_gray, (self.size, self.size), interpolation=cv2.INTER_AREA)

        with torch.no_grad():
            ta = torch.from_numpy(a).cuda().float().div_(255.0)[None, None]
            tb = torch.from_numpy(b).cuda().float().div_(255.0)[None, None]
            h_patch = self.model.predict(ta, tb)[0].double().cpu().numpy()

        # S maps full-resolution pixels to the square patch grid (anisotropic)
        s = np.diag([self.size / w0, self.size / h0, 1.0])
        h_full = np.linalg.inv(s) @ h_patch @ s
        result: np.ndarray = h_full / h_full[2, 2]
        return result
