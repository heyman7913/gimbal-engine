"""
CUDA-accelerated kernels for video stabilization using CuPy.

This module provides GPU-accelerated implementations of computationally
intensive operations used in the video stabilization pipeline.
"""

import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import map_coordinates, convolve, gaussian_filter
from typing import Tuple, Optional, Callable, List
import time


# Global verbose flag for CUDA profiling
_cuda_verbose = False


def set_cuda_verbose(enabled: bool) -> None:
    """Enable or disable verbose CUDA profiling output."""
    global _cuda_verbose
    _cuda_verbose = enabled


def _cuda_profile(operation_name: str):
    """Decorator to profile CUDA operations with timing and memory."""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            if not _cuda_verbose:
                return func(*args, **kwargs)

            # Sync GPU before timing
            cp.cuda.Stream.null.synchronize()
            mem_before = cp.cuda.Device().mem_info[0]
            start = time.perf_counter()

            result = func(*args, **kwargs)

            # Sync and measure
            cp.cuda.Stream.null.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            mem_after = cp.cuda.Device().mem_info[0]
            mem_used = (mem_before - mem_after) / 1024 / 1024

            print(f"    [CUDA] {operation_name}: {elapsed:.2f}ms, GPU mem delta: {mem_used:+.2f}MB")
            return result
        return wrapper
    return decorator


def check_cuda_available() -> bool:
    """Check if CUDA is available and fully functional."""
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            return False
        # Actually test GPU operations to catch missing DLLs (nvrtc, etc.)
        test_arr = cp.array([1.0, 2.0, 3.0])
        _ = cp.sum(test_arr)
        return True
    except Exception:
        return False


def get_device_info() -> dict:
    """Get information about the current CUDA device."""
    if not check_cuda_available():
        return {"available": False}

    device = cp.cuda.Device()
    return {
        "available": True,
        "device_id": device.id,
        "compute_capability": device.compute_capability,
        "total_memory_gb": device.mem_info[1] / 1e9,
        "free_memory_gb": device.mem_info[0] / 1e9,
    }


def gaussian_kernel_1d_gpu(size: int, sigma: float) -> cp.ndarray:
    """
    Generate a 1D Gaussian kernel on the GPU.

    Args:
        size: Kernel size (must be odd)
        sigma: Standard deviation of the Gaussian

    Returns:
        Normalized 1D Gaussian kernel as CuPy array
    """
    if size % 2 == 0:
        size += 1

    x = cp.arange(size) - size // 2
    kernel = cp.exp(-x**2 / (2 * sigma**2))
    kernel = kernel / cp.sum(kernel)  # Normalize
    return kernel.astype(cp.float32)


def smooth_trajectory_gpu(
    dx: np.ndarray,
    dy: np.ndarray,
    da: np.ndarray,
    smoothing_factor: float
) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """
    Smooth camera trajectory using GPU-accelerated Gaussian convolution.

    Args:
        dx: X translation per frame
        dy: Y translation per frame
        da: Rotation angle per frame
        smoothing_factor: Smoothing strength (0.0-1.0)

    Returns:
        Tuple of (corr_x, corr_y, corr_a) correction arrays on GPU
    """
    global _cuda_verbose

    def _log(msg: str):
        if _cuda_verbose:
            print(f"    [CUDA] {msg}")

    def _timed(name: str, func: Callable):
        if not _cuda_verbose:
            return func()
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        result = func()
        cp.cuda.Stream.null.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        print(f"    [CUDA] {name}: {elapsed:.2f}ms")
        return result

    n_frames = len(dx)
    _log(f"Processing {n_frames} frames on GPU")

    # Show GPU memory status
    if _cuda_verbose:
        free, total = cp.cuda.Device().mem_info
        print(f"    [CUDA] GPU memory: {free/1e9:.2f}GB free / {total/1e9:.2f}GB total")

    # Transfer to GPU
    def _transfer():
        return (
            cp.asarray(dx, dtype=cp.float32),
            cp.asarray(dy, dtype=cp.float32),
            cp.asarray(da, dtype=cp.float32),
        )
    dx_gpu, dy_gpu, da_gpu = _timed("CPU->GPU transfer", _transfer)

    # Build cumulative trajectory on GPU
    def _cumsum():
        return cp.cumsum(dx_gpu), cp.cumsum(dy_gpu), cp.cumsum(da_gpu)
    traj_x, traj_y, traj_a = _timed("Cumulative sum (trajectory)", _cumsum)

    # Compute window size and sigma for Gaussian
    smoothing_factor = max(0.0, min(1.0, smoothing_factor))
    window = int(5 + smoothing_factor * 96)
    if window % 2 == 0:
        window += 1
    sigma = window / 6.0
    _log(f"Gaussian kernel: window={window}, sigma={sigma:.2f}")

    # Generate Gaussian kernel on GPU
    kernel = _timed("Gaussian kernel generation", lambda: gaussian_kernel_1d_gpu(window, sigma))

    # Pad trajectories (edge mode)
    pad = window // 2
    def _pad():
        return (
            cp.pad(traj_x, (pad, pad), mode='edge'),
            cp.pad(traj_y, (pad, pad), mode='edge'),
            cp.pad(traj_a, (pad, pad), mode='edge'),
        )
    traj_x_padded, traj_y_padded, traj_a_padded = _timed("Padding trajectories", _pad)

    # Convolve on GPU (Gaussian smoothing) - this is the heavy operation
    def _convolve():
        return (
            cp.convolve(traj_x_padded, kernel, mode='valid'),
            cp.convolve(traj_y_padded, kernel, mode='valid'),
            cp.convolve(traj_a_padded, kernel, mode='valid'),
        )
    traj_x_smoothed, traj_y_smoothed, traj_a_smoothed = _timed("Gaussian convolution (3x)", _convolve)

    # Compute corrections
    def _corrections():
        return traj_x_smoothed - traj_x, traj_y_smoothed - traj_y, traj_a_smoothed - traj_a
    corr_x, corr_y, corr_a = _timed("Compute corrections", _corrections)

    if _cuda_verbose:
        free, _ = cp.cuda.Device().mem_info
        print(f"    [CUDA] GPU memory after smoothing: {free/1e9:.2f}GB free")

    return corr_x, corr_y, corr_a


def build_correction_transforms_gpu(
    corr_x: cp.ndarray,
    corr_y: cp.ndarray,
    corr_a: cp.ndarray
) -> np.ndarray:
    """
    Build correction transformation matrices on GPU.

    Args:
        corr_x: X correction array (GPU)
        corr_y: Y correction array (GPU)
        corr_a: Rotation correction array (GPU)

    Returns:
        Array of 3x3 transformation matrices (CPU, for OpenCV)
    """
    n = len(corr_x)

    # Compute cos and sin on GPU
    cos_a = cp.cos(corr_a)
    sin_a = cp.sin(corr_a)

    # Build transformation matrices on GPU
    # Shape: (n, 3, 3)
    transforms = cp.zeros((n, 3, 3), dtype=cp.float32)
    transforms[:, 0, 0] = cos_a
    transforms[:, 0, 1] = -sin_a
    transforms[:, 0, 2] = corr_x
    transforms[:, 1, 0] = sin_a
    transforms[:, 1, 1] = cos_a
    transforms[:, 1, 2] = corr_y
    transforms[:, 2, 2] = 1.0

    # Transfer back to CPU for OpenCV
    return cp.asnumpy(transforms)


def compute_max_displacement_gpu(transforms: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute maximum displacement from transformation matrices on GPU.

    Args:
        transforms: Array of 3x3 transformation matrices

    Returns:
        Tuple of (max_dx, max_dy, max_da) maximum displacements
    """
    transforms_gpu = cp.asarray(transforms, dtype=cp.float32)

    # Extract translation components
    dx = transforms_gpu[:, 0, 2]
    dy = transforms_gpu[:, 1, 2]

    # Extract rotation angles
    da = cp.arctan2(transforms_gpu[:, 1, 0], transforms_gpu[:, 0, 0])

    # Compute maximum absolute displacements
    max_dx = float(cp.max(cp.abs(dx)))
    max_dy = float(cp.max(cp.abs(dy)))
    max_da = float(cp.max(cp.abs(da)))

    return max_dx, max_dy, max_da


# ============================================================================
# GPU Frame Warping (CuPy)
# ============================================================================

def warp_frame_gpu(
    frame: np.ndarray,
    transform_2x3: np.ndarray,
    width: int,
    height: int
) -> np.ndarray:
    """
    Warp a frame using CuPy GPU-accelerated affine transformation.

    Pure CUDA implementation using CuPy - no OpenCV CUDA needed.

    Args:
        frame: Input BGR frame (H, W, 3)
        transform_2x3: 2x3 affine transformation matrix
        width: Output width
        height: Output height

    Returns:
        Warped frame
    """
    global _cuda_verbose

    if _cuda_verbose:
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()

    # Upload frame to GPU
    frame_gpu = cp.asarray(frame, dtype=cp.float32)

    # Build inverse transform for mapping
    # We need inverse because map_coordinates maps output -> input
    T = np.vstack([transform_2x3, [0, 0, 1]])
    T_inv = np.linalg.inv(T)[:2, :]
    T_inv_gpu = cp.asarray(T_inv, dtype=cp.float32)

    # Create output coordinate grid
    y_coords, x_coords = cp.mgrid[0:height, 0:width]

    # Apply inverse transform to get source coordinates
    # [x_src, y_src] = T_inv @ [x_dst, y_dst, 1]
    x_src = T_inv_gpu[0, 0] * x_coords + T_inv_gpu[0, 1] * y_coords + T_inv_gpu[0, 2]
    y_src = T_inv_gpu[1, 0] * x_coords + T_inv_gpu[1, 1] * y_coords + T_inv_gpu[1, 2]

    # Stack coordinates for map_coordinates
    coords = cp.array([y_src, x_src])

    # Warp each channel
    warped_channels = []
    for c in range(frame_gpu.shape[2]):
        warped_c = map_coordinates(
            frame_gpu[:, :, c],
            coords,
            order=1,  # Bilinear interpolation
            mode='nearest',  # Replicate border
        )
        warped_channels.append(warped_c)

    # Stack channels and convert back
    warped_gpu = cp.stack(warped_channels, axis=-1)
    result = cp.asnumpy(warped_gpu.astype(cp.uint8))

    if _cuda_verbose:
        cp.cuda.Stream.null.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        print(f"    [CUDA] Frame warp: {elapsed:.2f}ms")

    return result


def warp_frames_batch_gpu(
    frames: List[np.ndarray],
    transforms: np.ndarray,
    width: int,
    height: int,
    verbose: bool = False,
    batch_size: int = 8
) -> List[np.ndarray]:
    """
    Warp multiple frames using batched GPU processing.

    Processes frames in batches to maximize GPU throughput while
    managing memory usage.

    Args:
        frames: List of BGR frames
        transforms: Array of 3x3 transforms (uses first 2 rows)
        width: Output width
        height: Output height
        verbose: Print timing
        batch_size: Number of frames to process per GPU batch

    Returns:
        List of warped frames
    """
    global _cuda_verbose

    if verbose:
        start = time.perf_counter()
        print(f"    [CUDA WARP] Processing {len(frames)} frames (batch_size={batch_size})...")

    warped = []
    n_frames = len(frames)

    # Process in batches
    for batch_start in range(0, n_frames, batch_size):
        batch_end = min(batch_start + batch_size, n_frames)
        batch_frames = frames[batch_start:batch_end]
        batch_transforms = transforms[batch_start:batch_end]

        # Upload all frames in batch to GPU at once
        batch_gpu = cp.asarray(np.stack(batch_frames), dtype=cp.float32)

        # Pre-compute coordinate grid (same for all frames)
        y_coords, x_coords = cp.mgrid[0:height, 0:width]

        batch_warped = []
        for i, frame_gpu in enumerate(batch_gpu):
            T = batch_transforms[i]
            T_2x3 = T[:2, :]

            # Inverse transform
            T_full = np.vstack([T_2x3, [0, 0, 1]])
            T_inv = np.linalg.inv(T_full)[:2, :]
            T_inv_gpu = cp.asarray(T_inv, dtype=cp.float32)

            # Apply inverse transform
            x_src = T_inv_gpu[0, 0] * x_coords + T_inv_gpu[0, 1] * y_coords + T_inv_gpu[0, 2]
            y_src = T_inv_gpu[1, 0] * x_coords + T_inv_gpu[1, 1] * y_coords + T_inv_gpu[1, 2]
            coords = cp.array([y_src, x_src])

            # Warp each channel
            warped_channels = []
            for c in range(3):
                warped_c = map_coordinates(frame_gpu[:, :, c], coords, order=1, mode='nearest')
                warped_channels.append(warped_c)

            warped_frame = cp.stack(warped_channels, axis=-1).astype(cp.uint8)
            batch_warped.append(cp.asnumpy(warped_frame))

        warped.extend(batch_warped)

        # Free GPU memory for this batch
        del batch_gpu
        cp.get_default_memory_pool().free_all_blocks()

    if verbose:
        cp.cuda.Stream.null.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        fps = len(frames) / (elapsed / 1000)
        print(f"    [CUDA WARP] Completed in {elapsed:.1f}ms ({fps:.1f} fps)")

    return warped


# ============================================================================
# GPU Dense Optical Flow (CuPy Implementation) - Improved
# ============================================================================

def compute_gradients_gpu(img: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray]:
    """Compute image gradients using Scharr operators on GPU (more accurate than Sobel)."""
    # Scharr kernels - flipped for proper gradient direction with convolve
    # convolve() does true convolution (flips kernel), so we pre-flip to get correct sign
    # Result: Ix > 0 when intensity increases rightward, Iy > 0 when intensity increases downward
    scharr_x = cp.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=cp.float32) / 32.0
    scharr_y = cp.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=cp.float32) / 32.0

    Ix = convolve(img, scharr_x, mode='reflect')
    Iy = convolve(img, scharr_y, mode='reflect')

    return Ix, Iy


def detect_corners_gpu(img: cp.ndarray, max_corners: int = 500, quality_level: float = 0.01, min_distance: int = 10) -> cp.ndarray:
    """
    GPU-accelerated Shi-Tomasi corner detection using CuPy.

    Returns array of corner coordinates (N, 2) as (x, y).
    """
    # Compute gradients
    Ix, Iy = compute_gradients_gpu(img)

    # Compute structure tensor components
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    IxIy = Ix * Iy

    # Apply Gaussian window
    window_size = 5
    sigma = window_size / 6.0
    Ix2 = gaussian_filter(Ix2, sigma=sigma)
    Iy2 = gaussian_filter(Iy2, sigma=sigma)
    IxIy = gaussian_filter(IxIy, sigma=sigma)

    # Shi-Tomasi score: min eigenvalue of structure tensor
    # lambda_min = (Ix2 + Iy2) / 2 - sqrt(((Ix2 - Iy2) / 2)^2 + IxIy^2)
    trace = Ix2 + Iy2
    det = Ix2 * Iy2 - IxIy * IxIy
    discriminant = cp.sqrt(cp.maximum((trace / 2) ** 2 - det, 0))
    min_eigenvalue = trace / 2 - discriminant

    # Threshold
    max_score = float(cp.max(min_eigenvalue))
    threshold = quality_level * max_score

    # Non-maximum suppression using dilation
    from cupyx.scipy.ndimage import maximum_filter
    local_max = maximum_filter(min_eigenvalue, size=min_distance * 2 + 1)
    is_peak = (min_eigenvalue == local_max) & (min_eigenvalue > threshold)

    # Get coordinates
    y_coords, x_coords = cp.where(is_peak)
    scores = min_eigenvalue[is_peak]

    # Sort by score and take top corners
    sorted_idx = cp.argsort(scores)[::-1][:max_corners]
    x_coords = x_coords[sorted_idx]
    y_coords = y_coords[sorted_idx]

    corners = cp.stack([x_coords, y_coords], axis=1).astype(cp.float32)

    return corners


def _extract_patches_vectorized(img: cp.ndarray, points: cp.ndarray, half_win: int) -> cp.ndarray:
    """
    Extract patches around all points simultaneously using map_coordinates.

    Args:
        img: Image (H, W)
        points: Points (N, 2) as (x, y)
        half_win: Half window size

    Returns:
        Patches array (N, win_size, win_size)
    """
    n_pts = len(points)
    win_size = 2 * half_win + 1

    # Create local coordinate grid for a single patch
    local_y, local_x = cp.mgrid[-half_win:half_win+1, -half_win:half_win+1]
    local_y = local_y.astype(cp.float32).ravel()  # (win_size^2,)
    local_x = local_x.astype(cp.float32).ravel()

    # Broadcast to all points: (N, win_size^2)
    px = points[:, 0:1] + local_x[None, :]  # (N, win_size^2)
    py = points[:, 1:2] + local_y[None, :]

    # Flatten for map_coordinates
    coords = cp.stack([py.ravel(), px.ravel()], axis=0)  # (2, N*win_size^2)

    # Sample all at once
    sampled = map_coordinates(img, coords, order=1, mode='constant', cval=0.0)

    # Reshape to (N, win_size, win_size)
    return sampled.reshape(n_pts, win_size, win_size)


def track_points_gpu(
    prev_img: cp.ndarray,
    curr_img: cp.ndarray,
    prev_pts: cp.ndarray,
    window_size: int = 21,
    max_level: int = 3,
    iterations: int = 20,
    epsilon: float = 0.03
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    GPU-accelerated sparse Lucas-Kanade optical flow tracking (VECTORIZED).

    Processes ALL points in parallel for maximum GPU utilization.

    Args:
        prev_img: Previous frame (grayscale, float32)
        curr_img: Current frame (grayscale, float32)
        prev_pts: Points to track (N, 2) as (x, y)
        window_size: Size of search window
        max_level: Number of pyramid levels
        iterations: Max iterations for convergence
        epsilon: Convergence threshold

    Returns:
        Tuple of (tracked_points, status) where status is 1 for success
    """
    n_pts = len(prev_pts)
    if n_pts == 0:
        return cp.array([]).reshape(0, 2), cp.array([])

    # Build image pyramids
    prev_pyr = [prev_img]
    curr_pyr = [curr_img]

    for level in range(1, max_level + 1):
        prev_blurred = gaussian_filter(prev_pyr[-1], sigma=1.0)
        curr_blurred = gaussian_filter(curr_pyr[-1], sigma=1.0)
        prev_pyr.append(prev_blurred[::2, ::2])
        curr_pyr.append(curr_blurred[::2, ::2])

    half_win = window_size // 2
    win_size = window_size

    # Initialize flow and status for all points
    flow = cp.zeros((n_pts, 2), dtype=cp.float32)
    status = cp.ones(n_pts, dtype=cp.int32)

    # Process from coarse to fine
    for level in range(max_level, -1, -1):
        prev_level = prev_pyr[level]
        curr_level = curr_pyr[level]
        h, w = prev_level.shape

        scale = 2.0 ** level
        margin = half_win + 2

        # Scale points and flow to this level
        pts_scaled = prev_pts.astype(cp.float32) / scale
        flow_scaled = flow / scale

        # Compute gradients
        Ix, Iy = compute_gradients_gpu(prev_level)

        # ===== VECTORIZED: Check bounds for all points =====
        px = pts_scaled[:, 0]
        py = pts_scaled[:, 1]
        out_of_bounds = (px < margin) | (px >= w - margin) | (py < margin) | (py >= h - margin)
        status = cp.where(out_of_bounds, 0, status)

        # Get indices of still-active points
        active_mask = status == 1
        active_idx = cp.where(active_mask)[0]

        if len(active_idx) == 0:
            flow = flow_scaled * scale
            continue

        # Active points only
        active_pts = pts_scaled[active_idx]
        active_flow = flow_scaled[active_idx].copy()

        # ===== VECTORIZED: Extract template windows for all active points =====
        templates = _extract_patches_vectorized(prev_level, active_pts, half_win)
        Ix_patches = _extract_patches_vectorized(Ix, active_pts, half_win)
        Iy_patches = _extract_patches_vectorized(Iy, active_pts, half_win)

        # ===== VECTORIZED: Compute structure tensors for all points =====
        # Shape: (n_active,)
        Ixx = cp.sum(Ix_patches * Ix_patches, axis=(1, 2))
        Iyy = cp.sum(Iy_patches * Iy_patches, axis=(1, 2))
        Ixy = cp.sum(Ix_patches * Iy_patches, axis=(1, 2))

        det = Ixx * Iyy - Ixy * Ixy

        # Mark points with bad structure tensor
        bad_det = det < 1e-4

        # For computation, use safe values (will be masked out)
        safe_det = cp.where(bad_det, 1.0, det)
        inv_det = 1.0 / safe_det

        # ===== VECTORIZED: Iterative Lucas-Kanade =====
        converged = cp.zeros(len(active_idx), dtype=cp.bool_)
        failed = bad_det.copy()

        for _ in range(iterations):
            # Current target positions
            qx = active_pts[:, 0] + active_flow[:, 0]
            qy = active_pts[:, 1] + active_flow[:, 1]

            # Check bounds
            oob = (qx < margin) | (qx >= w - margin) | (qy < margin) | (qy >= h - margin)
            failed = failed | oob

            # Points still iterating
            iterating = ~converged & ~failed

            if not cp.any(iterating):
                break

            # Extract current windows for ALL active points at target locations
            target_pts = cp.stack([qx, qy], axis=1)
            curr_patches = _extract_patches_vectorized(curr_level, target_pts, half_win)

            # Temporal gradient
            It = curr_patches - templates

            # Compute b vector for all points: b = -[sum(Ix*It), sum(Iy*It)]
            bx = -cp.sum(Ix_patches * It, axis=(1, 2))
            by = -cp.sum(Iy_patches * It, axis=(1, 2))

            # Solve 2x2 system: dv = G^-1 * b
            dvx = (Iyy * bx - Ixy * by) * inv_det
            dvy = (-Ixy * bx + Ixx * by) * inv_det

            # Update flow (only for non-converged, non-failed points)
            update_mask = iterating.astype(cp.float32)
            active_flow[:, 0] += dvx * update_mask
            active_flow[:, 1] += dvy * update_mask

            # Check convergence
            newly_converged = (cp.abs(dvx) < epsilon) & (cp.abs(dvy) < epsilon)
            converged = converged | (newly_converged & iterating)

        # ===== Write results back =====
        # Update status for failed points
        failed_global_idx = active_idx[failed]
        status[failed_global_idx] = 0

        # Update flow for all active points (including failed, will be masked by status)
        flow_scaled[active_idx] = active_flow

        # Scale flow back
        flow = flow_scaled * scale

    # Final tracked points
    tracked_pts = prev_pts.astype(cp.float32) + flow

    return tracked_pts, status


def ransac_affine_gpu(
    src_pts: cp.ndarray,
    dst_pts: cp.ndarray,
    n_iterations: int = 500,
    threshold: float = 3.0,
    min_inliers: int = 10
) -> Tuple[Optional[np.ndarray], cp.ndarray]:
    """
    RANSAC-based affine transform estimation (4 DOF: rotation, scale, translation).

    VECTORIZED: Processes all iterations in parallel on GPU.

    Returns:
        Tuple of (2x3 affine transform matrix, inlier mask)
    """
    n_pts = len(src_pts)
    if n_pts < 2:
        return None, cp.zeros(n_pts, dtype=cp.bool_)

    # ===== VECTORIZED: Generate all random samples at once =====
    # Shape: (n_iterations, 2) - two point indices per iteration
    all_idx = cp.zeros((n_iterations, 2), dtype=cp.int32)
    all_idx[:, 0] = cp.random.randint(0, n_pts, size=n_iterations)
    # Second index different from first
    offsets = cp.random.randint(1, n_pts, size=n_iterations)
    all_idx[:, 1] = (all_idx[:, 0] + offsets) % n_pts

    # Gather sample points: (n_iterations, 2, 2) - [iter, point_idx, xy]
    src_samples = src_pts[all_idx]  # (n_iter, 2, 2)
    dst_samples = dst_pts[all_idx]  # (n_iter, 2, 2)

    # ===== VECTORIZED: Compute transforms for all iterations =====
    # Centers: (n_iter, 2)
    src_centers = cp.mean(src_samples, axis=1)
    dst_centers = cp.mean(dst_samples, axis=1)

    # Relative positions: (n_iter, 2, 2)
    src_rel = src_samples - src_centers[:, None, :]
    dst_rel = dst_samples - dst_centers[:, None, :]

    # Compute angles: atan2(dy, dx) for the vector between the two points
    # src_rel[:, 1] - src_rel[:, 0] gives vector from point 0 to point 1
    src_vec = src_rel[:, 1] - src_rel[:, 0]  # (n_iter, 2)
    dst_vec = dst_rel[:, 1] - dst_rel[:, 0]  # (n_iter, 2)

    src_angles = cp.arctan2(src_vec[:, 1], src_vec[:, 0])  # (n_iter,)
    dst_angles = cp.arctan2(dst_vec[:, 1], dst_vec[:, 0])  # (n_iter,)
    angles = dst_angles - src_angles

    # Compute scales: |dst_vec| / |src_vec|
    src_dists = cp.sqrt(cp.sum(src_vec ** 2, axis=1)) + 1e-6
    dst_dists = cp.sqrt(cp.sum(dst_vec ** 2, axis=1))
    scales = dst_dists / src_dists

    # Clamp scales to [0.9, 1.1]
    scales = cp.clip(scales, 0.9, 1.1)

    # Build transform parameters: (n_iter,)
    cos_a = cp.cos(angles) * scales
    sin_a = cp.sin(angles) * scales

    # Translation: dst_center = R @ src_center + t
    # t = dst_center - R @ src_center
    tx = dst_centers[:, 0] - cos_a * src_centers[:, 0] + sin_a * src_centers[:, 1]
    ty = dst_centers[:, 1] - sin_a * src_centers[:, 0] - cos_a * src_centers[:, 1]

    # ===== VECTORIZED: Apply all transforms to all points =====
    # src_pts: (n_pts, 2), cos_a/sin_a/tx/ty: (n_iter,)
    # Result: (n_iter, n_pts, 2)
    transformed_x = cos_a[:, None] * src_pts[None, :, 0] - sin_a[:, None] * src_pts[None, :, 1] + tx[:, None]
    transformed_y = sin_a[:, None] * src_pts[None, :, 0] + cos_a[:, None] * src_pts[None, :, 1] + ty[:, None]

    # Errors: (n_iter, n_pts)
    errors = cp.sqrt((transformed_x - dst_pts[None, :, 0]) ** 2 +
                     (transformed_y - dst_pts[None, :, 1]) ** 2)

    # Inlier counts: (n_iter,)
    inlier_masks = errors < threshold  # (n_iter, n_pts)
    inlier_counts = cp.sum(inlier_masks, axis=1)

    # Find best iteration
    best_iter = int(cp.argmax(inlier_counts))
    best_inliers = int(inlier_counts[best_iter])
    best_mask = inlier_masks[best_iter]

    if best_inliers < min_inliers:
        return None, cp.zeros(n_pts, dtype=cp.bool_)

    # Extract best transform parameters (single GPU->CPU transfer)
    best_cos_a = float(cos_a[best_iter])
    best_sin_a = float(sin_a[best_iter])
    best_tx = float(tx[best_iter])
    best_ty = float(ty[best_iter])

    best_transform = np.array([
        [best_cos_a, -best_sin_a, best_tx],
        [best_sin_a, best_cos_a, best_ty]
    ], dtype=np.float32)

    # Refine with all inliers using least squares
    src_inliers = src_pts[best_mask]
    dst_inliers = dst_pts[best_mask]

    n = len(src_inliers)
    if n >= 2:
        A = cp.zeros((2 * n, 4), dtype=cp.float32)
        A[0::2, 0] = src_inliers[:, 0]
        A[0::2, 1] = -src_inliers[:, 1]
        A[0::2, 2] = 1
        A[1::2, 0] = src_inliers[:, 1]
        A[1::2, 1] = src_inliers[:, 0]
        A[1::2, 3] = 1

        b = cp.zeros(2 * n, dtype=cp.float32)
        b[0::2] = dst_inliers[:, 0]
        b[1::2] = dst_inliers[:, 1]

        try:
            x, _, _, _ = cp.linalg.lstsq(A, b, rcond=None)
            cos_a_refined = float(x[0])
            sin_a_refined = float(x[1])
            tx_refined = float(x[2])
            ty_refined = float(x[3])
            best_transform = np.array([
                [cos_a_refined, -sin_a_refined, tx_refined],
                [sin_a_refined, cos_a_refined, ty_refined]
            ], dtype=np.float32)
        except Exception:
            pass

    return best_transform, best_mask


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
    Compute sparse optical flow using pure CuPy GPU implementation.

    Uses GPU-accelerated Shi-Tomasi corner detection + Lucas-Kanade tracking.

    Args:
        prev_gray: Previous frame (grayscale, uint8)
        curr_gray: Current frame (grayscale, uint8)
        max_corners: Maximum corners to detect
        quality_level: Corner quality threshold
        min_distance: Minimum distance between corners
        window_size: LK window size
        max_level: Pyramid levels

    Returns:
        Tuple of (prev_points, curr_points) matched point pairs
    """
    global _cuda_verbose

    if _cuda_verbose:
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()

    # Upload to GPU and convert to float32
    prev_gpu = cp.asarray(prev_gray, dtype=cp.float32)
    curr_gpu = cp.asarray(curr_gray, dtype=cp.float32)

    # Detect corners using GPU Shi-Tomasi
    corners = detect_corners_gpu(prev_gpu, max_corners, quality_level, min_distance)

    if len(corners) < 3:
        if _cuda_verbose:
            print(f"    [CUDA] Not enough corners detected: {len(corners)}")
        return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2)

    # Track points using GPU Lucas-Kanade
    tracked_pts, status = track_points_gpu(
        prev_gpu, curr_gpu, corners,
        window_size=window_size,
        max_level=max_level
    )

    # Filter by tracking status
    valid_mask = status == 1
    prev_good = corners[valid_mask]
    curr_good = tracked_pts[valid_mask]

    if _cuda_verbose:
        cp.cuda.Stream.null.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        print(f"    [CUDA] Sparse flow: {len(corners)} corners -> {int(cp.sum(valid_mask))} tracked ({elapsed:.2f}ms)")

    return cp.asnumpy(prev_good), cp.asnumpy(curr_good)


def estimate_transform_from_flow_gpu(
    prev_pts: np.ndarray,
    curr_pts: np.ndarray,
    ransac_iterations: int = 500,
    ransac_threshold: float = 5.0
) -> Tuple[Optional[np.ndarray], float, float, float]:
    """
    Estimate affine transform from point correspondences using GPU RANSAC.

    Args:
        prev_pts: Source points (N, 2)
        curr_pts: Destination points (N, 2)
        ransac_iterations: RANSAC iterations
        ransac_threshold: Inlier threshold in pixels

    Returns:
        Tuple of (transform_3x3, dx, dy, da) or (None, 0, 0, 0) if failed
    """
    global _cuda_verbose

    if len(prev_pts) < 3 or len(curr_pts) < 3:
        return None, 0.0, 0.0, 0.0

    if _cuda_verbose:
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()

    # Upload to GPU
    src_gpu = cp.asarray(prev_pts, dtype=cp.float32)
    dst_gpu = cp.asarray(curr_pts, dtype=cp.float32)

    # Run GPU RANSAC
    transform_2x3, inlier_mask = ransac_affine_gpu(
        src_gpu, dst_gpu,
        n_iterations=ransac_iterations,
        threshold=ransac_threshold
    )

    if transform_2x3 is None:
        if _cuda_verbose:
            print(f"    [CUDA] RANSAC failed - not enough inliers")
        return None, 0.0, 0.0, 0.0

    # Convert to 3x3
    transform_3x3 = np.vstack([transform_2x3, [0, 0, 1]]).astype(np.float32)

    # Extract translation and rotation
    dx = float(transform_2x3[0, 2])
    dy = float(transform_2x3[1, 2])
    da = float(np.arctan2(transform_2x3[1, 0], transform_2x3[0, 0]))

    if _cuda_verbose:
        cp.cuda.Stream.null.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        n_inliers = int(cp.sum(inlier_mask)) if inlier_mask is not None else 0
        print(f"    [CUDA] Transform: {n_inliers} inliers, dx={dx:.2f}, dy={dy:.2f}, da={np.degrees(da):.3f}° ({elapsed:.2f}ms)")

    return transform_3x3, dx, dy, da
