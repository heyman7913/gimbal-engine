"""GPU frame warping and auto-crop.

The stabilizing transform B maps an original frame onto the smoothed path. Rendering the
output samples the source through B^{-1} (dst -> src), so black borders appear wherever the
correction pushes content out of frame. The auto-crop finds the largest axis-aligned window
that stays inside every warped frame, which is also the cropping-ratio numerator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .cuda.kernels import WARP_BILINEAR_U8, get_kernel

if TYPE_CHECKING:
    import cupy as cp


def warp_frame(src: "cp.ndarray", b: np.ndarray) -> "cp.ndarray":
    """Warp a uint8 (H, W, C) device frame by homography B (source -> destination).

    Returns a same-size device frame; out-of-frame pixels are black.
    """
    import cupy as cp

    if src.dtype != cp.uint8 or src.ndim != 3:
        raise ValueError("warp_frame expects a uint8 (H, W, C) device array")
    h, w, ch = src.shape
    h_inv = cp.asarray(np.linalg.inv(b).ravel(), dtype=cp.float64)
    dst = cp.empty_like(src)
    kernel = get_kernel("warp_bilinear_u8", WARP_BILINEAR_U8)
    block = (16, 16)
    grid = ((w + block[0] - 1) // block[0], (h + block[1] - 1) // block[1])
    kernel(
        grid,
        block,
        (src, dst, h_inv, np.int32(h), np.int32(w), np.int32(h), np.int32(w), np.int32(ch)),
    )
    return dst


def _frame_quad(b: np.ndarray, w: int, h: int) -> np.ndarray:
    """Map the four source corners through B into destination coordinates."""
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    pts = np.concatenate([corners, np.ones((4, 1))], axis=1) @ b.T
    return pts[:, :2] / pts[:, 2:3]


def compute_crop_box(transforms: np.ndarray, w: int, h: int) -> tuple[int, int, int, int]:
    """Largest axis-aligned (left, top, right, bottom) window inside every warped frame.

    Per frame, the inner rectangle of the warped quad is bounded by the inner corners; the
    crop is the intersection of those rectangles across all frames.
    """
    left = top = -np.inf
    right = bottom = np.inf
    for b in transforms:
        q = _frame_quad(b, w, h)  # tl, tr, br, bl
        left = max(left, q[0, 0], q[3, 0])
        right = min(right, q[1, 0], q[2, 0])
        top = max(top, q[0, 1], q[1, 1])
        bottom = min(bottom, q[2, 1], q[3, 1])
    li = int(np.ceil(max(0.0, left)))
    ti = int(np.ceil(max(0.0, top)))
    ri = int(np.floor(min(float(w), right)))
    bi = int(np.floor(min(float(h), bottom)))
    if ri - li < 8 or bi - ti < 8:
        # corrections too large for a usable crop; keep the full frame
        return 0, 0, w, h
    return li, ti, ri, bi
