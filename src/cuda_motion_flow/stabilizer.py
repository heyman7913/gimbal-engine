"""
Video stabilization pipeline with full CUDA acceleration.

Pipeline (4 passes):
  1. Optical flow   -- GPU Shi-Tomasi + pyramidal LK + vectorised RANSAC
  2. Trajectory     -- Gaussian / Kalman-RTS / L1-TV smoothing
  3. Warp           -- raw CUDA bilinear affine kernel, dual-stream overlap
  4. Metrics        -- ITV, optical-flow variance, temporal SSIM (optional)

Optionally exports the recovered camera trajectory in JSON or COLMAP format,
enabling downstream use in SfM / Gaussian Splatting pipelines.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2 as cv
import numpy as np

from .cuda_kernels import (
    check_cuda_available,
    compute_max_displacement_gpu,
    compute_optical_flow_gpu,
    estimate_transform_from_flow_gpu,
    get_device_info,
    set_cuda_verbose,
    warp_frame_gpu,
)
from .trajectory import smooth_trajectory, SMOOTHERS
from .utils import (
    apply_auto_crop,
    calculate_auto_crop,
    estimate_processing_time,
    get_codec_for_extension,
)


def stabilize_video(
    input_path: str,
    output_path: str,
    smoothing_factor: float = 0.3,
    smoother: str = "gaussian",
    verbose: bool = False,
    auto_crop: bool = True,
    preserve_resolution: bool = True,
    export_trajectory: Optional[str] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> None:
    """
    Stabilize a video using the full CUDA-accelerated pipeline.

    Args:
        input_path:          Path to the input video file.
        output_path:         Path for the stabilized output video.
        smoothing_factor:    Smoothing strength in [0, 1].
        smoother:            Trajectory smoother: gaussian, kalman, or l1.
        verbose:             Print per-stage timing and diagnostics.
        auto_crop:           Crop black borders introduced by stabilization.
        preserve_resolution: Resize cropped frames back to the original size.
        export_trajectory:   Export camera trajectory to this path.
                             .json suffix -> JSON; directory -> COLMAP format.

    Raises:
        RuntimeError:      If CUDA is not available or video cannot be processed.
        FileNotFoundError: If the input file does not exist.
        ValueError:        If smoother is not a recognised method name.
    """
    if smoother not in SMOOTHERS:
        raise ValueError(
            f"Unknown smoother {smoother!r}. Choose from {list(SMOOTHERS)}"
        )

    if not check_cuda_available():
        raise RuntimeError("CUDA is required but not available.")

    set_cuda_verbose(verbose)

    if verbose:
        info = get_device_info()
        print(
            f"GPU  -- device {info['device_id']}, "
            f"{info['free_memory_gb']:.1f} GB free  "
            f"(cc {info['compute_capability']})"
        )

    cap = cv.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {input_path}")

    width       = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps         = cap.get(cv.CAP_PROP_FPS)
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    if verbose:
        print(f"Input    -- {width}x{height} @ {fps:.2f} fps, {frame_count} frames")
        print(f"Smoother -- {smoother}  |  strength={smoothing_factor}")
        print(f"ETA      -- {estimate_processing_time(frame_count, cuda_enabled=True)}")

    # -----------------------------------------------------------------------
    # Pass 1: inter-frame transforms via GPU optical flow
    # -----------------------------------------------------------------------
    if verbose:
        print("\nPass 1: optical flow...")
        t0 = time.perf_counter()

    ret, prev_frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame.")

    prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)

    transforms: List[np.ndarray] = []
    dx_list:    List[float] = []
    dy_list:    List[float] = []
    da_list:    List[float] = []

    frame_idx = 1
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        curr_gray = cv.cvtColor(curr_frame, cv.COLOR_BGR2GRAY)
        prev_pts, curr_pts = compute_optical_flow_gpu(prev_gray, curr_gray)
        H, dx, dy, da = estimate_transform_from_flow_gpu(prev_pts, curr_pts)

        if H is not None:
            transforms.append(H)
        else:
            transforms.append(np.eye(3, dtype=np.float32))
            dx = dy = da = 0.0

        dx_list.append(dx)
        dy_list.append(dy)
        da_list.append(da)

        prev_gray = curr_gray
        frame_idx += 1

        if progress_callback:
            progress_callback("flow", frame_idx, frame_count - 1)
        if verbose and frame_idx % 100 == 0:
            print(f"  {frame_idx}/{frame_count}...")

    cap.release()

    if not transforms:
        raise RuntimeError("No valid transforms computed.")

    dx_arr = np.array(dx_list, dtype=np.float32)
    dy_arr = np.array(dy_list, dtype=np.float32)
    da_arr = np.array(da_list, dtype=np.float32)

    if verbose:
        elapsed = time.perf_counter() - t0
        print(
            f"  Done -- {len(transforms)} transforms in {elapsed:.1f}s "
            f"({len(transforms) / elapsed:.1f} fps)"
        )

    # -----------------------------------------------------------------------
    # Pass 2: smooth trajectory
    # -----------------------------------------------------------------------
    if verbose:
        print(f"\nPass 2: trajectory smoothing ({smoother})...")
        t0 = time.perf_counter()

    if progress_callback:
        progress_callback("smooth", 0, 1)

    corrected = smooth_trajectory(
        dx_arr, dy_arr, da_arr,
        method=smoother,
        smoothing_strength=smoothing_factor,
    )

    if progress_callback:
        progress_callback("smooth", 1, 1)

    max_dx, max_dy, max_da = compute_max_displacement_gpu(corrected)

    if verbose:
        elapsed = time.perf_counter() - t0
        print(
            f"  Max displacement -- dx={max_dx:.1f}px  dy={max_dy:.1f}px  "
            f"da={np.degrees(max_da):.2f} deg  ({elapsed * 1e3:.1f}ms)"
        )

    crop_rect   = None
    output_size = None
    if auto_crop:
        crop_rect = calculate_auto_crop(width, height, max_dx, max_dy, max_da)
        if preserve_resolution:
            output_size = (width, height)
        if verbose:
            x, y, w, h = crop_rect
            print(f"  Auto-crop -- {w}x{h}  (border {x}px)")

    # -----------------------------------------------------------------------
    # Optional: export camera trajectory
    # -----------------------------------------------------------------------
    if export_trajectory:
        _export_trajectory(transforms, width, height, export_trajectory, verbose)

    # -----------------------------------------------------------------------
    # Pass 3: warp frames and write output
    # -----------------------------------------------------------------------
    if verbose:
        print("\nPass 3: warping...")
        t0 = time.perf_counter()

    cap = cv.VideoCapture(str(input_path))
    fourcc = get_codec_for_extension(str(output_path))
    out    = cv.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError(f"Cannot create output video: {output_path}")

    ret, first_frame = cap.read()
    if auto_crop and crop_rect is not None:
        first_frame = apply_auto_crop(first_frame, crop_rect, output_size)
    out.write(first_frame)

    for i, T in enumerate(corrected):
        ret, frame = cap.read()
        if not ret:
            break

        stabilized = warp_frame_gpu(frame, T[:2, :], width, height)

        if auto_crop and crop_rect is not None:
            stabilized = apply_auto_crop(stabilized, crop_rect, output_size)

        out.write(stabilized)

        if progress_callback:
            progress_callback("warp", i + 1, len(corrected))
        if verbose and (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  {i + 2}/{frame_count}  ({(i + 1) / elapsed:.1f} fps)")

    cap.release()
    out.release()

    if verbose:
        elapsed = time.perf_counter() - t0
        print(f"  Done in {elapsed:.1f}s  ->  {output_path}")

    # -----------------------------------------------------------------------
    # Pass 4: quality metrics (verbose only)
    # -----------------------------------------------------------------------
    if verbose:
        from .metrics import compute_all_metrics
        print("\nPass 4: computing metrics...")
        metrics = compute_all_metrics(
            str(input_path), str(output_path),
            crop_rect=crop_rect,
            original_size=(width, height),
            verbose=True,
        )
        print(metrics)


# ---------------------------------------------------------------------------
# Trajectory export helper
# ---------------------------------------------------------------------------

def _export_trajectory(
    transforms: List[np.ndarray],
    width: int,
    height: int,
    path: str,
    verbose: bool,
) -> None:
    from .geometry import estimate_intrinsics, build_trajectory

    if verbose:
        print(f"\nExporting trajectory -> {path}")

    K    = estimate_intrinsics(width, height)
    traj = build_trajectory(transforms, K)

    p = Path(path)
    if p.suffix.lower() == ".json":
        traj.export_json(p)
    else:
        traj.export_colmap(p)

    if verbose:
        print(f"  Exported {len(traj)} poses.")


# ---------------------------------------------------------------------------
# Legacy helpers retained for backwards compatibility
# ---------------------------------------------------------------------------

def smooth_trajectory_legacy(
    dx: np.ndarray,
    dy: np.ndarray,
    da: np.ndarray,
    smoothing_factor: float,
) -> np.ndarray:
    return smooth_trajectory(
        dx, dy, da, method="gaussian", smoothing_strength=smoothing_factor
    )


def compute_max_displacement(
    transforms: np.ndarray,
    verbose: bool = False,
) -> Tuple[float, float, float]:
    return compute_max_displacement_gpu(transforms)
