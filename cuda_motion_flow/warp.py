"""Warping frames on the GPU and figuring out the crop.

To render a stabilized frame I sample the source through the inverse transform, so wherever the
correction pushes content off-screen I get black borders. The auto-crop then finds the biggest
rectangle that stays inside every frame - that same rectangle is the cropping-ratio metric.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .cuda._bridge import ext, to_cupy, to_torch

if TYPE_CHECKING:
    import cupy as cp


def warp_frame(src: cp.ndarray, b: np.ndarray) -> cp.ndarray:
    """Warp a uint8 (H, W, C) device frame by homography B (source -> destination).

    Returns a same-size device frame; out-of-frame pixels are black.
    """
    import cupy as cp
    import torch

    if src.dtype != cp.uint8 or src.ndim != 3:
        raise ValueError("warp_frame expects a uint8 (H, W, C) device array")
    h, w, _ = src.shape
    h_inv = torch.from_numpy(np.ascontiguousarray(np.linalg.inv(b))).cuda()
    dst_t = ext().warp_bilinear(to_torch(src), h_inv, h, w)
    return to_cupy(dst_t)


def mesh_warp_frame(src: np.ndarray, field_inv: np.ndarray) -> np.ndarray:
    """Warp a uint8 (H, W, C) frame by a per-cell field of inverse transforms (dst -> src).

    Uses grid_sample over the bilinearly-blended per-cell sampling grid. With a 1x1 field this
    is the single-homography warp.
    """
    import torch

    from .learned.model import mesh_sampling_grid

    h, w = src.shape[:2]
    t = torch.as_tensor(src, device="cuda").permute(2, 0, 1).unsqueeze(0).float()
    f = torch.as_tensor(field_inv, device="cuda", dtype=torch.float32).unsqueeze(0)
    grid = mesh_sampling_grid(f, h, w)
    out = torch.nn.functional.grid_sample(
        t, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    arr: np.ndarray = out[0].permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
    return arr


def _frame_quad(b: np.ndarray, w: int, h: int) -> np.ndarray:
    """Map the four source corners through B into destination coordinates."""
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    pts = np.concatenate([corners, np.ones((4, 1))], axis=1) @ b.T
    quad: np.ndarray = pts[:, :2] / pts[:, 2:3]
    return quad


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
