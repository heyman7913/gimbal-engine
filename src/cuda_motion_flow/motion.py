"""MotionField: the motion an estimator hands back.

A single global homography is just the 1x1 case of a grid of local homographies. The classical
and global-IHN estimators return a 1x1 field; the mesh estimator returns a GxG grid. The
back-end is written so the 1x1 case reduces exactly to the original single-homography path, so
the leak-free comparison still holds (there is a bit-exact test for this).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MotionField:
    homographies: np.ndarray  # (gh, gw, 3, 3), one local homography per cell

    @property
    def grid_shape(self) -> tuple[int, int]:
        return (int(self.homographies.shape[0]), int(self.homographies.shape[1]))

    @property
    def is_global(self) -> bool:
        return self.grid_shape == (1, 1)

    @classmethod
    def global_(cls, h: np.ndarray) -> MotionField:
        """Wrap a single 3x3 homography as a 1x1 field."""
        return cls(np.asarray(h, dtype=np.float64).reshape(1, 1, 3, 3))

    def as_global(self) -> np.ndarray:
        """The single 3x3 homography of a 1x1 field."""
        if not self.is_global:
            raise ValueError(f"as_global called on a {self.grid_shape} mesh field")
        out: np.ndarray = self.homographies[0, 0]
        return out
