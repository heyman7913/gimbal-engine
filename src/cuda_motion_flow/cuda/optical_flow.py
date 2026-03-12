"""Corner detection and pyramidal Lucas-Kanade, all on the GPU.

The heavy per-pixel stuff (gradients, corner scores, downsampling) is done by the CUDA kernels;
the LK solve itself I vectorise over all the points with CuPy so there's no python loop over
features. Points are (x, y) float32.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .kernels import (
    GAUSSIAN_DOWNSAMPLE_F32,
    SCHARR_GRADIENT_F32,
    SHI_TOMASI_RESPONSE_F32,
    get_kernel,
)

if TYPE_CHECKING:
    import cupy as cp


def _grid(w: int, h: int, block: tuple[int, int] = (16, 16)) -> tuple[int, int]:
    return ((w + block[0] - 1) // block[0], (h + block[1] - 1) // block[1])


def scharr_gradients(img: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
    """First derivatives (ix, iy) of a float32 image, normalized like cv2.Scharr / 32."""
    import cupy as cp

    h, w = img.shape
    ix = cp.empty_like(img)
    iy = cp.empty_like(img)
    kernel = get_kernel("scharr_gradient_f32", SCHARR_GRADIENT_F32)
    block = (16, 16)
    kernel(_grid(w, h, block), block, (img, ix, iy, np.int32(h), np.int32(w)))
    return ix, iy


def shi_tomasi_response(ix: cp.ndarray, iy: cp.ndarray, radius: int = 3) -> cp.ndarray:
    """Min-eigenvalue corner response over a (2*radius+1) box window."""
    import cupy as cp

    h, w = ix.shape
    resp = cp.empty_like(ix)
    kernel = get_kernel("shi_tomasi_response_f32", SHI_TOMASI_RESPONSE_F32)
    block = (16, 16)
    kernel(_grid(w, h, block), block, (ix, iy, resp, np.int32(h), np.int32(w), np.int32(radius)))
    return resp


def downsample(img: cp.ndarray) -> cp.ndarray:
    """2x Gaussian-pyramid downsample (cv2.pyrDown convention)."""
    import cupy as cp

    sh, sw = img.shape
    dh, dw = (sh + 1) // 2, (sw + 1) // 2
    dst = cp.empty((dh, dw), dtype=cp.float32)
    kernel = get_kernel("gaussian_downsample_f32", GAUSSIAN_DOWNSAMPLE_F32)
    block = (16, 16)
    args = (img, dst, np.int32(sh), np.int32(sw), np.int32(dh), np.int32(dw))
    kernel(_grid(dw, dh, block), block, args)
    return dst


def build_pyramid(img: cp.ndarray, levels: int) -> list[cp.ndarray]:
    pyr = [img]
    for _ in range(levels):
        pyr.append(downsample(pyr[-1]))
    return pyr


def detect_corners(
    gray: cp.ndarray,
    max_corners: int = 600,
    quality: float = 0.01,
    min_distance: int = 8,
    radius: int = 3,
) -> cp.ndarray:
    """Shi-Tomasi corners as an (N, 2) float32 array of (x, y) device coordinates."""
    import cupy as cp
    from cupyx.scipy.ndimage import maximum_filter

    ix, iy = scharr_gradients(gray)
    resp = shi_tomasi_response(ix, iy, radius)
    peak = maximum_filter(resp, size=2 * min_distance + 1)
    keep = (resp >= peak) & (resp > quality * float(resp.max()))
    ys, xs = cp.where(keep)
    if xs.size == 0:
        return cp.empty((0, 2), dtype=cp.float32)
    vals = resp[ys, xs]
    order = cp.argsort(-vals)[:max_corners]
    pts = cp.stack([xs[order].astype(cp.float32), ys[order].astype(cp.float32)], axis=1)
    return pts


def _sample(img: cp.ndarray, x: cp.ndarray, y: cp.ndarray) -> cp.ndarray:
    """Bilinear sample img at (x, y); out-of-range clamps to the edge."""
    from cupyx.scipy.ndimage import map_coordinates

    return map_coordinates(img, cp_stack(y, x), order=1, mode="nearest")


def cp_stack(a: cp.ndarray, b: cp.ndarray) -> cp.ndarray:
    import cupy as cp

    return cp.stack([a.ravel(), b.ravel()], axis=0)


def track_points(
    prev: cp.ndarray,
    curr: cp.ndarray,
    pts: cp.ndarray,
    window: int = 15,
    levels: int = 3,
    iters: int = 12,
    eps: float = 0.01,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Pyramidal LK (Bouguet). Returns tracked (N, 2) points and a boolean status array."""
    import cupy as cp

    n = pts.shape[0]
    if n == 0:
        return pts.copy(), cp.zeros((0,), dtype=bool)

    r = window // 2
    off = cp.arange(-r, r + 1, dtype=cp.float32)
    oy, ox = cp.meshgrid(off, off, indexing="ij")
    ox = ox.ravel()[None, :]  # (1, win*win)
    oy = oy.ravel()[None, :]

    prev_pyr = build_pyramid(prev, levels)
    curr_pyr = build_pyramid(curr, levels)

    g = cp.zeros((n, 2), dtype=cp.float32)  # flow guess in full-res pixels
    status = cp.ones((n,), dtype=bool)

    for lvl in range(levels, -1, -1):
        scale = 1.0 / (2**lvl)
        p = pts * scale  # (N, 2) point at this level
        ix, iy = scharr_gradients(prev_pyr[lvl])
        i_prev = prev_pyr[lvl]
        j_curr = curr_pyr[lvl]
        hh, ww = i_prev.shape

        cx = p[:, 0:1]
        cy = p[:, 1:2]
        wx = (cx + ox).clip(0, ww - 1)
        wy = (cy + oy).clip(0, hh - 1)

        i_patch = _sample(i_prev, wx, wy).reshape(n, -1)
        ix_patch = _sample(ix, wx, wy).reshape(n, -1)
        iy_patch = _sample(iy, wx, wy).reshape(n, -1)

        gxx = (ix_patch * ix_patch).sum(axis=1)
        gyy = (iy_patch * iy_patch).sum(axis=1)
        gxy = (ix_patch * iy_patch).sum(axis=1)
        det = gxx * gyy - gxy * gxy
        good = det > 1e-6

        nu = cp.zeros((n, 2), dtype=cp.float32)
        for _ in range(iters):
            qx = (cx + g[:, 0:1] * scale + nu[:, 0:1] + ox).clip(0, ww - 1)
            qy = (cy + g[:, 1:2] * scale + nu[:, 1:2] + oy).clip(0, hh - 1)
            j_patch = _sample(j_curr, qx, qy).reshape(n, -1)
            it = j_patch - i_patch
            bx = (ix_patch * it).sum(axis=1)
            by = (iy_patch * it).sum(axis=1)
            # solve G d = -b per point (2x2 closed form)
            safe_det = cp.where(good, det, 1.0)
            dx = cp.where(good, -(gyy * bx - gxy * by) / safe_det, 0.0)
            dy = cp.where(good, -(gxx * by - gxy * bx) / safe_det, 0.0)
            nu[:, 0] += dx
            nu[:, 1] += dy
            if float((dx * dx + dy * dy).max()) < eps * eps:
                break

        # nu is the residual flow at this level; lift it into full-res guess
        g = g + nu / scale
        status &= good

    tracked = pts + g
    inside = (
        (tracked[:, 0] >= 0)
        & (tracked[:, 0] < prev.shape[1])
        & (tracked[:, 1] >= 0)
        & (tracked[:, 1] < prev.shape[0])
    )
    status &= inside
    return tracked, status


def compute_flow(prev_gray: cp.ndarray, curr_gray: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
    """Detect corners in prev and track them into curr. Returns matched (src, dst) points."""
    pts = detect_corners(prev_gray)
    tracked, status = track_points(prev_gray, curr_gray, pts)
    return pts[status], tracked[status]
