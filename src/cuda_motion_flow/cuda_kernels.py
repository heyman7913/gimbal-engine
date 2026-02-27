"""
CUDA-accelerated kernels for the video stabilization pipeline.

Implements the full GPU pipeline:
  1. Shi-Tomasi corner detection   — raw CUDA kernel via raw_kernels.py
  2. Pyramidal Lucas-Kanade flow   — vectorized CuPy, pyramid built with raw kernel
  3. Vectorized GPU RANSAC         — all iterations in parallel on GPU
  4. Frame warping                 — raw CUDA bilinear kernel via raw_kernels.py

All public functions accept and return NumPy arrays at the boundary;
CuPy arrays are used internally and freed before returning.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import map_coordinates

from .raw_kernels import (
    affine_warp_u8,
    gaussian_downsample,
    scharr_gradients,
    shi_tomasi_response,
)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

_cuda_verbose = False


def set_cuda_verbose(enabled: bool) -> None:
    global _cuda_verbose
    _cuda_verbose = enabled


def check_cuda_available() -> bool:
    """Return True if CUDA is available and fully operational."""
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            return False
        _ = cp.sum(cp.array([1.0]))
        return True
    except Exception:
        return False


def get_device_info() -> dict:
    """Return basic properties of the current CUDA device."""
    if not check_cuda_available():
        return {"available": False}
    dev = cp.cuda.Device()
    try:
        name = cp.cuda.runtime.getDeviceProperties(dev.id)["name"].decode()
    except Exception:
        name = f"Device {dev.id}"
    return {
        "available": True,
        "device_id": dev.id,
        "device_name": name,
        "compute_capability": dev.compute_capability,
        "total_memory_gb": dev.mem_info[1] / 1e9,
        "free_memory_gb":  dev.mem_info[0] / 1e9,
    }


class _Timer:
    """Context manager that logs a CUDA-synchronised wall-clock interval."""

    def __init__(self, label: str, enabled: bool):
        self.label   = label
        self.enabled = enabled

    def __enter__(self):
        if self.enabled:
            cp.cuda.Stream.null.synchronize()
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        if self.enabled:
            cp.cuda.Stream.null.synchronize()
            ms = (time.perf_counter() - self._t0) * 1e3
            print(f"    [CUDA] {self.label}: {ms:.2f}ms")


# ---------------------------------------------------------------------------
# Trajectory smoothing (delegates to trajectory module)
# ---------------------------------------------------------------------------

def smooth_trajectory_gpu(
    dx: np.ndarray,
    dy: np.ndarray,
    da: np.ndarray,
    smoothing_factor: float,
) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """GPU Gaussian trajectory smoother (legacy entry point)."""
    from .trajectory import smooth_trajectory_gaussian
    return smooth_trajectory_gaussian(dx, dy, da, smoothing_factor)


def build_correction_transforms_gpu(
    corr_x: cp.ndarray,
    corr_y: cp.ndarray,
    corr_a: cp.ndarray,
) -> np.ndarray:
    """Build (N, 3, 3) correction transforms from GPU correction arrays."""
    n = len(corr_x)
    cos_a = cp.cos(corr_a)
    sin_a = cp.sin(corr_a)

    T = cp.zeros((n, 3, 3), dtype=cp.float32)
    T[:, 0, 0] =  cos_a
    T[:, 0, 1] = -sin_a
    T[:, 0, 2] =  corr_x
    T[:, 1, 0] =  sin_a
    T[:, 1, 1] =  cos_a
    T[:, 1, 2] =  corr_y
    T[:, 2, 2] =  1.0
    return cp.asnumpy(T)


def compute_max_displacement_gpu(transforms: np.ndarray) -> Tuple[float, float, float]:
    """Compute max absolute displacement from an array of 3×3 transforms."""
    T  = cp.asarray(transforms, dtype=cp.float32)
    dx = T[:, 0, 2]
    dy = T[:, 1, 2]
    da = cp.arctan2(T[:, 1, 0], T[:, 0, 0])
    return float(cp.max(cp.abs(dx))), float(cp.max(cp.abs(dy))), float(cp.max(cp.abs(da)))


# ---------------------------------------------------------------------------
# Corner detection — raw CUDA Scharr + Shi-Tomasi kernels
# ---------------------------------------------------------------------------

def detect_corners_gpu(
    img: cp.ndarray,
    max_corners: int = 500,
    quality_level: float = 0.01,
    min_distance: int = 10,
) -> cp.ndarray:
    """
    Shi-Tomasi corner detection on GPU using raw CUDA kernels.

    Uses the compiled Scharr gradient and Shi-Tomasi response kernels,
    then applies non-maximum suppression via a dilation filter.

    Returns:
        (N, 2) float32 array of corner (x, y) coordinates.
    """
    from cupyx.scipy.ndimage import maximum_filter

    img_f = img.astype(cp.float32) if img.dtype != cp.float32 else img

    with _Timer("Scharr gradients", _cuda_verbose):
        Gx, Gy = scharr_gradients(img_f)

    with _Timer("Shi-Tomasi response", _cuda_verbose):
        resp = shi_tomasi_response(Gx, Gy, half_window=3)

    max_score = float(cp.max(resp))
    if max_score <= 0:
        return cp.zeros((0, 2), dtype=cp.float32)

    threshold = quality_level * max_score
    local_max = maximum_filter(resp, size=min_distance * 2 + 1)
    is_peak   = (resp == local_max) & (resp > threshold)

    ys, xs   = cp.where(is_peak)
    scores   = resp[is_peak]

    top_k   = cp.argsort(scores)[::-1][:max_corners]
    corners = cp.stack([xs[top_k], ys[top_k]], axis=1).astype(cp.float32)
    return corners


# ---------------------------------------------------------------------------
# Lucas-Kanade optical flow — vectorised
# ---------------------------------------------------------------------------

def _extract_patches(img: cp.ndarray, pts: cp.ndarray, half_win: int) -> cp.ndarray:
    """Vectorised patch extraction via map_coordinates (all N patches in one call)."""
    win = 2 * half_win + 1
    dy, dx = cp.mgrid[-half_win:half_win + 1, -half_win:half_win + 1]
    dy = dy.astype(cp.float32).ravel()
    dx = dx.astype(cp.float32).ravel()

    px = pts[:, 0:1] + dx[None, :]   # (N, win^2)
    py = pts[:, 1:2] + dy[None, :]

    coords  = cp.stack([py.ravel(), px.ravel()], axis=0)
    sampled = map_coordinates(img, coords, order=1, mode='constant', cval=0.0)
    return sampled.reshape(len(pts), win, win)


def track_points_gpu(
    prev_img: cp.ndarray,
    curr_img: cp.ndarray,
    prev_pts: cp.ndarray,
    window_size: int = 21,
    max_level: int = 3,
    iterations: int = 20,
    epsilon: float = 0.03,
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Pyramidal Lucas-Kanade optical flow — fully vectorised on GPU.

    All N points are processed in parallel at every pyramid level and every
    iteration. Pyramid is built with the raw anti-aliased downsampling kernel.

    Returns:
        (tracked_pts, status) where status=1 indicates a successfully tracked point.
    """
    n_pts = len(prev_pts)
    if n_pts == 0:
        return cp.zeros((0, 2), dtype=cp.float32), cp.zeros(0, dtype=cp.int32)

    half_win = window_size // 2

    # Build pyramids using the raw Gaussian downsampling kernel
    prev_pyr: List[cp.ndarray] = [prev_img.astype(cp.float32)]
    curr_pyr: List[cp.ndarray] = [curr_img.astype(cp.float32)]
    for _ in range(max_level):
        prev_pyr.append(gaussian_downsample(prev_pyr[-1]))
        curr_pyr.append(gaussian_downsample(curr_pyr[-1]))

    flow   = cp.zeros((n_pts, 2), dtype=cp.float32)
    status = cp.ones(n_pts, dtype=cp.int32)

    for level in range(max_level, -1, -1):
        prev_l = prev_pyr[level]
        curr_l = curr_pyr[level]
        h, w   = prev_l.shape
        scale  = 2.0 ** level
        margin = half_win + 2

        pts_s  = prev_pts.astype(cp.float32) / scale
        flow_s = flow / scale

        Ix, Iy = scharr_gradients(prev_l)

        px  = pts_s[:, 0]
        py  = pts_s[:, 1]
        oob = (px < margin) | (px >= w - margin) | (py < margin) | (py >= h - margin)
        status = cp.where(oob, 0, status)

        active = cp.where(status == 1)[0]
        if len(active) == 0:
            flow = flow_s * scale
            continue

        a_pts  = pts_s[active]
        a_flow = flow_s[active].copy()

        templates  = _extract_patches(prev_l, a_pts, half_win)
        Ix_patches = _extract_patches(Ix,     a_pts, half_win)
        Iy_patches = _extract_patches(Iy,     a_pts, half_win)

        Ixx     = cp.sum(Ix_patches * Ix_patches, axis=(1, 2))
        Iyy     = cp.sum(Iy_patches * Iy_patches, axis=(1, 2))
        Ixy     = cp.sum(Ix_patches * Iy_patches, axis=(1, 2))
        det     = Ixx * Iyy - Ixy * Ixy
        bad_det = det < 1e-4
        inv_det = 1.0 / cp.where(bad_det, 1.0, det)

        converged = cp.zeros(len(active), dtype=cp.bool_)
        failed    = bad_det.copy()

        for _ in range(iterations):
            qx  = a_pts[:, 0] + a_flow[:, 0]
            qy  = a_pts[:, 1] + a_flow[:, 1]
            oob = (qx < margin) | (qx >= w - margin) | (qy < margin) | (qy >= h - margin)
            failed = failed | oob

            iterating = ~converged & ~failed
            if not cp.any(iterating):
                break

            curr_patches = _extract_patches(curr_l, cp.stack([qx, qy], axis=1), half_win)
            It = curr_patches - templates

            bx  = -cp.sum(Ix_patches * It, axis=(1, 2))
            by  = -cp.sum(Iy_patches * It, axis=(1, 2))
            dvx = (Iyy * bx - Ixy * by) * inv_det
            dvy = (-Ixy * bx + Ixx * by) * inv_det

            mask         = iterating.astype(cp.float32)
            a_flow[:, 0] += dvx * mask
            a_flow[:, 1] += dvy * mask

            newly     = (cp.abs(dvx) < epsilon) & (cp.abs(dvy) < epsilon)
            converged = converged | (newly & iterating)

        status[active[failed]] = 0
        flow_s[active] = a_flow
        flow = flow_s * scale

    return prev_pts.astype(cp.float32) + flow, status


# ---------------------------------------------------------------------------
# RANSAC affine estimation — fully vectorised
# ---------------------------------------------------------------------------

def ransac_affine_gpu(
    src_pts: cp.ndarray,
    dst_pts: cp.ndarray,
    n_iterations: int = 500,
    threshold: float = 3.0,
    min_inliers: int = 10,
) -> Tuple[Optional[np.ndarray], cp.ndarray]:
    """
    Vectorised GPU RANSAC for 4-DOF similarity (rotation + scale + translation).

    All `n_iterations` hypotheses are generated and scored in parallel.
    The best hypothesis is refined with least-squares over all its inliers.

    Returns:
        (2×3 affine matrix, boolean inlier mask) or (None, empty mask).
    """
    n_pts = len(src_pts)
    if n_pts < 2:
        return None, cp.zeros(n_pts, dtype=cp.bool_)

    idx0    = cp.random.randint(0, n_pts, size=n_iterations)
    idx1    = (idx0 + cp.random.randint(1, n_pts, size=n_iterations)) % n_pts

    s0 = src_pts[idx0];  s1 = src_pts[idx1]
    d0 = dst_pts[idx0];  d1 = dst_pts[idx1]

    sv = s1 - s0;  dv = d1 - d0
    sa = cp.arctan2(sv[:, 1], sv[:, 0])
    da = cp.arctan2(dv[:, 1], dv[:, 0])

    scales = cp.clip(
        cp.sqrt(cp.sum(dv ** 2, axis=1)) / (cp.sqrt(cp.sum(sv ** 2, axis=1)) + 1e-6),
        0.9, 1.1,
    )
    cos_a = cp.cos(da - sa) * scales
    sin_a = cp.sin(da - sa) * scales

    sc = (s0 + s1) / 2.0;  dc = (d0 + d1) / 2.0
    tx = dc[:, 0] - cos_a * sc[:, 0] + sin_a * sc[:, 1]
    ty = dc[:, 1] - sin_a * sc[:, 0] - cos_a * sc[:, 1]

    # Score all hypotheses against all points — shape (n_iter, n_pts)
    px = cos_a[:, None] * src_pts[None, :, 0] - sin_a[:, None] * src_pts[None, :, 1] + tx[:, None]
    py = sin_a[:, None] * src_pts[None, :, 0] + cos_a[:, None] * src_pts[None, :, 1] + ty[:, None]
    errs  = cp.sqrt((px - dst_pts[None, :, 0]) ** 2 + (py - dst_pts[None, :, 1]) ** 2)
    masks = errs < threshold

    best_iter  = int(cp.argmax(cp.sum(masks, axis=1)))
    best_count = int(cp.sum(masks[best_iter]))
    best_mask  = masks[best_iter]

    if best_count < min_inliers:
        return None, cp.zeros(n_pts, dtype=cp.bool_)

    best_M = np.array([
        [float(cos_a[best_iter]), -float(sin_a[best_iter]), float(tx[best_iter])],
        [float(sin_a[best_iter]),  float(cos_a[best_iter]), float(ty[best_iter])],
    ], dtype=np.float32)

    # Least-squares refinement over all inliers
    si = src_pts[best_mask];  di = dst_pts[best_mask]
    ni = len(si)
    if ni >= 2:
        A = cp.zeros((2 * ni, 4), dtype=cp.float32)
        A[0::2, 0] =  si[:, 0];  A[0::2, 1] = -si[:, 1];  A[0::2, 2] = 1
        A[1::2, 0] =  si[:, 1];  A[1::2, 1] =  si[:, 0];                A[1::2, 3] = 1
        b = cp.zeros(2 * ni, dtype=cp.float32)
        b[0::2] = di[:, 0];  b[1::2] = di[:, 1]
        try:
            x = cp.linalg.lstsq(A, b, rcond=None)[0].get()
            best_M = np.array([[x[0], -x[1], x[2]], [x[1], x[0], x[3]]], dtype=np.float32)
        except Exception:
            pass

    return best_M, best_mask


# ---------------------------------------------------------------------------
# Public pipeline entry points
# ---------------------------------------------------------------------------

def compute_optical_flow_gpu(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    max_corners: int = 200,
    quality_level: float = 0.01,
    min_distance: int = 30,
    window_size: int = 15,
    max_level: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full sparse optical flow via GPU Shi-Tomasi detection + pyramidal LK tracking.

    Returns:
        (prev_pts, curr_pts) matched point pairs as (N, 2) float32 arrays.
    """
    with _Timer("Upload + corners", _cuda_verbose):
        prev_g  = cp.asarray(prev_gray, dtype=cp.float32)
        curr_g  = cp.asarray(curr_gray, dtype=cp.float32)
        corners = detect_corners_gpu(prev_g, max_corners, quality_level, min_distance)

    if len(corners) < 3:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    with _Timer("LK tracking", _cuda_verbose):
        tracked, status = track_points_gpu(prev_g, curr_g, corners,
                                           window_size=window_size,
                                           max_level=max_level)

    valid     = status == 1
    prev_good = corners[valid]
    curr_good = tracked[valid]

    if _cuda_verbose:
        print(f"    [CUDA] Tracked {int(cp.sum(valid))}/{len(corners)} corners")

    return cp.asnumpy(prev_good), cp.asnumpy(curr_good)


def estimate_transform_from_flow_gpu(
    prev_pts: np.ndarray,
    curr_pts: np.ndarray,
    ransac_iterations: int = 500,
    ransac_threshold: float = 5.0,
) -> Tuple[Optional[np.ndarray], float, float, float]:
    """
    4-DOF affine transform estimation via vectorised GPU RANSAC.

    Returns:
        (H_3x3, dx, dy, da) or (None, 0, 0, 0) on failure.
    """
    if len(prev_pts) < 3:
        return None, 0.0, 0.0, 0.0

    with _Timer("RANSAC", _cuda_verbose):
        src_g = cp.asarray(prev_pts, dtype=cp.float32)
        dst_g = cp.asarray(curr_pts, dtype=cp.float32)
        M2x3, inliers = ransac_affine_gpu(src_g, dst_g,
                                           n_iterations=ransac_iterations,
                                           threshold=ransac_threshold)

    if M2x3 is None:
        return None, 0.0, 0.0, 0.0

    H  = np.vstack([M2x3, [0.0, 0.0, 1.0]]).astype(np.float32)
    dx = float(M2x3[0, 2])
    dy = float(M2x3[1, 2])
    da = float(np.arctan2(M2x3[1, 0], M2x3[0, 0]))

    if _cuda_verbose:
        n_in = int(cp.sum(inliers)) if inliers is not None else 0
        print(f"    [CUDA] RANSAC inliers={n_in} dx={dx:.2f} dy={dy:.2f} da={np.degrees(da):.3f}°")

    return H, dx, dy, da


# ---------------------------------------------------------------------------
# Frame warping — raw CUDA kernel
# ---------------------------------------------------------------------------

def warp_frame_gpu(
    frame: np.ndarray,
    transform_2x3: np.ndarray,
    width: int,
    height: int,
    stream: Optional["cp.cuda.Stream"] = None,
) -> np.ndarray:
    """
    Warp a BGR frame using the raw CUDA bilinear affine kernel.

    The kernel handles all 3 channels with `#pragma unroll` and uses `__ldg()`
    for L1-cached reads — significantly faster than map_coordinates.

    Args:
        frame:          (H, W, 3) uint8 BGR frame.
        transform_2x3:  2×3 forward affine (src → dst). Inverted internally.
        width, height:  Output dimensions.
        stream:         Optional CUDA stream.

    Returns:
        (height, width, 3) uint8 warped frame.
    """
    with _Timer("Upload", _cuda_verbose):
        frame_gpu = cp.asarray(frame, dtype=cp.uint8)

    T_inv = np.linalg.inv(np.vstack([transform_2x3, [0, 0, 1]]))[:2, :].astype(np.float32)

    with _Timer("Warp kernel", _cuda_verbose):
        warped_gpu = affine_warp_u8(frame_gpu, T_inv, width, height, stream=stream)

    return cp.asnumpy(warped_gpu)


def warp_frames_batch_gpu(
    frames: List[np.ndarray],
    transforms: np.ndarray,
    width: int,
    height: int,
) -> List[np.ndarray]:
    """
    Warp a list of frames using two alternating CUDA streams.

    Overlapping H2D transfers and GPU compute keeps the GPU continuously busy.
    """
    stream_a = cp.cuda.Stream(non_blocking=True)
    stream_b = cp.cuda.Stream(non_blocking=True)
    streams  = [stream_a, stream_b]

    results: List[Optional[np.ndarray]] = [None] * len(frames)

    for i, (frame, T) in enumerate(zip(frames, transforms)):
        s     = streams[i % 2]
        T_inv = np.linalg.inv(T)[:2, :].astype(np.float32)
        with s:
            frame_gpu  = cp.asarray(frame, dtype=cp.uint8)
            warped_gpu = affine_warp_u8(frame_gpu, T_inv, width, height, stream=s)
        results[i] = cp.asnumpy(warped_gpu)

    stream_a.synchronize()
    stream_b.synchronize()
    return results  # type: ignore[return-value]
