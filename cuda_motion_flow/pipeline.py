"""The stabilization pipeline.

It only ever talks to the Estimator interface, so it has no idea whether it's running the
classical or the learned one - which is the whole point of the project.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from . import metrics, trajectory
from .estimators.base import Estimator
from .motion import MotionField
from .trajectory import Smoother
from .video_io import to_gray
from .warp import compute_crop_box, warp_frame

ProgressFn = Callable[[str, int, int], None]


@dataclass
class StabilizeResult:
    frames: list[np.ndarray]
    pairwise: np.ndarray  # (N-1, 3, 3) inter-frame homographies from the estimator
    cumulative: np.ndarray  # (N, 3, 3) original absolute path
    smoothed: np.ndarray  # (N, 3, 3) smoothed absolute path (the stabilized camera path)
    transforms: np.ndarray  # (N, 3, 3) per-frame stabilizing warps
    crop_box: tuple[int, int, int, int]
    metrics: dict[str, float] = field(default_factory=dict)


def stabilize(
    frames: list[np.ndarray],
    estimator: Estimator,
    smoother: Smoother = "kalman_rts",
    strength: float = 0.6,
    auto_crop: bool = True,
    preserve_resolution: bool = True,
    progress: ProgressFn | None = None,
) -> StabilizeResult:
    """Stabilize a decoded clip with the given estimator and return frames plus diagnostics."""
    import cupy as cp

    if len(frames) < 2:
        raise ValueError("need at least two frames to stabilize")
    h, w = frames[0].shape[:2]

    grays = [to_gray(f) for f in frames]
    estimator.warmup(h, w)

    fields = []
    for i in range(1, len(frames)):
        fields.append(estimator.estimate(grays[i - 1], grays[i]))
        if progress is not None:
            progress("estimate", i, len(frames) - 1)

    if not all(f.is_global for f in fields):
        return _stabilize_mesh(frames, fields, smoother, strength, auto_crop, preserve_resolution)

    # global path: identical to the original single-homography pipeline (bit-exact)
    pairwise = np.stack([f.as_global() for f in fields])
    cumulative = trajectory.cumulative_path(pairwise)
    smoothed = trajectory.smooth_path(cumulative, smoother, strength)
    transforms = trajectory.stabilizing_transforms(cumulative, smoothed)

    crop_box = compute_crop_box(transforms, w, h) if auto_crop else (0, 0, w, h)
    li, ti, ri, bi = crop_box

    out: list[np.ndarray] = []
    for i, (frame, b) in enumerate(zip(frames, transforms, strict=True)):
        dev = cp.asarray(frame)
        warped = warp_frame(dev, b)
        host = cp.asnumpy(warped)
        cropped = host[ti:bi, li:ri]
        if preserve_resolution and cropped.shape[:2] != (h, w):
            import cv2

            cropped = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        out.append(cropped)
        if progress is not None:
            progress("warp", i + 1, len(frames))

    result = StabilizeResult(
        frames=out,
        pairwise=pairwise,
        cumulative=cumulative,
        smoothed=smoothed,
        transforms=transforms,
        crop_box=crop_box,
    )
    result.metrics = {
        "cropping_ratio": metrics.cropping_ratio(ri - li, bi - ti, w, h),
        "distortion_value": metrics.distortion_value(transforms),
        "stability_score": metrics.stability_score(smoothed),
        "stability_input": metrics.stability_score(cumulative),
    }
    return result


def _stabilize_mesh(
    frames: list[np.ndarray],
    fields: list[MotionField],
    smoother: Smoother,
    strength: float,
    auto_crop: bool,
    preserve_resolution: bool,
) -> StabilizeResult:
    raise NotImplementedError("mesh MotionField stabilization is added with the mesh estimator")
