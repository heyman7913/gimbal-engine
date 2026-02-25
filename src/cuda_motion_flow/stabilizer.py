"""
Video stabilization module with CUDA acceleration.

This module implements robust video stabilization using:
- GPU-accelerated sparse optical flow (CuPy Lucas-Kanade)
- GPU-accelerated RANSAC transform estimation (CuPy)
- Gaussian smoothing for natural motion preservation
- CUDA-accelerated trajectory smoothing via CuPy
- GPU-accelerated frame warping via CuPy
- Auto-crop to remove black borders
"""

import numpy as np
import cv2 as cv
from typing import Tuple, Optional, List
import time
from .cuda_kernels import (
    check_cuda_available,
    get_device_info,
    smooth_trajectory_gpu,
    build_correction_transforms_gpu,
    compute_max_displacement_gpu,
    set_cuda_verbose,
    warp_frame_gpu,
    compute_optical_flow_gpu,
    estimate_transform_from_flow_gpu,
)
from .utils import (
    calculate_auto_crop,
    apply_auto_crop,
    get_codec_for_extension,
    estimate_processing_time,
)
from .metrics import compute_all_metrics


def stabilize_video(
    input_path: str,
    output_path: str,
    smoothing_factor: float = 0.3,
    verbose: bool = False,
    auto_crop: bool = True,
    preserve_resolution: bool = True,
) -> None:
    """
    Stabilize a video using GPU-accelerated feature tracking and trajectory smoothing.

    Args:
        input_path: Path to input video file
        output_path: Path for stabilized output video
        smoothing_factor: Smoothing strength (0.0-1.0)
        verbose: Print detailed progress information
        auto_crop: Whether to crop black borders
        preserve_resolution: If True, resize cropped frame back to original size

    Raises:
        FileNotFoundError: If input file doesn't exist
        RuntimeError: If video cannot be opened or processed
    """
    # Check for CUDA
    cuda_available = check_cuda_available()

    # Enable CUDA profiling if verbose
    set_cuda_verbose(verbose)

    if not cuda_available:
        raise RuntimeError("CUDA is required but not available. Check your CUDA installation.")

    if verbose:
        info = get_device_info()
        print(f"CUDA enabled - Device {info['device_id']}, "
              f"{info['free_memory_gb']:.1f}GB free")

    # Open video
    cap = cv.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open input video: {input_path}")

    # Get video properties
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS)
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    if verbose:
        print(f"Input: {width}x{height} @ {fps:.2f}fps, {frame_count} frames")
        print(f"Estimated time: {estimate_processing_time(frame_count, cuda_available)}")
        print("Optical flow: CuPy GPU (CUDA)")

    # === PASS 1: Compute inter-frame transforms ===
    if verbose:
        print("Pass 1: Computing optical flow...")
        flow_start = time.perf_counter()

    ret, prev_frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame")

    prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)

    # Store transforms and individual motion components
    transforms: List[np.ndarray] = []
    dx_list: List[float] = []
    dy_list: List[float] = []
    da_list: List[float] = []

    frame_idx = 1
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        curr_gray = cv.cvtColor(curr_frame, cv.COLOR_BGR2GRAY)

        # Compute transform between frames using GPU optical flow
        prev_pts, curr_pts = compute_optical_flow_gpu(prev_gray, curr_gray)
        H, dx, dy, da = estimate_transform_from_flow_gpu(prev_pts, curr_pts)

        if H is not None:
            transforms.append(H)
            dx_list.append(dx)
            dy_list.append(dy)
            da_list.append(da)
        else:
            # No transform found - assume no motion
            transforms.append(np.eye(3, dtype=np.float32))
            dx_list.append(0.0)
            dy_list.append(0.0)
            da_list.append(0.0)

        prev_gray = curr_gray
        frame_idx += 1

        if verbose and frame_idx % 100 == 0:
            print(f"  Processed {frame_idx}/{frame_count} frames...")

    cap.release()

    if len(transforms) == 0:
        raise RuntimeError("No valid transforms computed")

    # Convert to arrays
    dx = np.array(dx_list, dtype=np.float32)
    dy = np.array(dy_list, dtype=np.float32)
    da = np.array(da_list, dtype=np.float32)

    if verbose:
        flow_elapsed = time.perf_counter() - flow_start
        flow_fps = len(transforms) / flow_elapsed if flow_elapsed > 0 else 0
        print(f"Completed optical flow: {len(transforms)} frames in {flow_elapsed:.1f}s ({flow_fps:.1f} fps)")

    # === PASS 2: Smooth trajectory ===
    if verbose:
        print("Pass 2: Smoothing trajectory...")

    corrected_transforms = smooth_trajectory(dx, dy, da, smoothing_factor)

    # Compute max displacements for auto-crop
    max_dx, max_dy, max_da = compute_max_displacement(corrected_transforms, verbose)

    if verbose:
        print(f"  Max displacement: dx={max_dx:.1f}px, dy={max_dy:.1f}px, da={np.degrees(max_da):.2f}deg")

    # Calculate crop rectangle
    crop_rect = None
    output_size = None
    if auto_crop:
        crop_rect = calculate_auto_crop(width, height, max_dx, max_dy, max_da)
        if verbose:
            x, y, w, h = crop_rect
            print(f"  Auto-crop: {w}x{h} (removed {x}px borders)")

        if preserve_resolution:
            output_size = (width, height)

    # === PASS 3: Apply transforms and write output ===
    if verbose:
        print("Pass 3: Applying stabilization...")
        print(f"  Warp mode: GPU (CuPy CUDA)")

    cap = cv.VideoCapture(str(input_path))

    # Determine output codec and create writer
    fourcc = get_codec_for_extension(str(output_path))
    out = cv.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not out.isOpened():
        raise RuntimeError(f"Cannot create output video: {output_path}")

    # Write first frame (no transform applied)
    ret, frame = cap.read()
    if auto_crop and crop_rect is not None:
        frame = apply_auto_crop(frame, crop_rect, output_size)
    out.write(frame)

    # Apply transforms to remaining frames using GPU warping
    warp_start = time.perf_counter()
    for i, T in enumerate(corrected_transforms):
        ret, frame = cap.read()
        if not ret:
            break

        # GPU-accelerated warp (falls back to CPU LANCZOS4 if no OpenCV CUDA)
        stabilized = warp_frame_gpu(frame, T[:2, :], width, height)

        # Apply auto-crop if enabled
        if auto_crop and crop_rect is not None:
            stabilized = apply_auto_crop(stabilized, crop_rect, output_size)

        out.write(stabilized)

        if verbose and (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - warp_start
            fps_actual = (i + 1) / elapsed
            print(f"  Written {i + 2}/{frame_count} frames... ({fps_actual:.1f} fps)")

    cap.release()
    out.release()

    if verbose:
        total_warp_time = time.perf_counter() - warp_start
        print(f"  Warping completed in {total_warp_time:.1f}s")
        print(f"Done! Output saved to: {output_path}")

    # === PASS 4: Compute and display metrics ===
    if verbose:
        print("\n" + "="*60)
        metrics = compute_all_metrics(
            str(input_path),
            str(output_path),
            crop_rect=crop_rect,
            original_size=(width, height),
            verbose=True
        )
        print(metrics)


def smooth_trajectory(
    dx: np.ndarray,
    dy: np.ndarray,
    da: np.ndarray,
    smoothing_factor: float,
) -> np.ndarray:
    """
    Smooth the camera trajectory and compute correction transforms using CUDA.

    Args:
        dx: X translation per frame
        dy: Y translation per frame
        da: Rotation angle per frame
        smoothing_factor: Smoothing strength (0.0-1.0)

    Returns:
        Array of 3x3 correction transformation matrices

    Raises:
        RuntimeError: If CUDA is not available
    """
    if not check_cuda_available():
        raise RuntimeError("CUDA required but not available. Check your CUDA installation.")
    corr_x, corr_y, corr_a = smooth_trajectory_gpu(dx, dy, da, smoothing_factor)
    return build_correction_transforms_gpu(corr_x, corr_y, corr_a)


def compute_max_displacement(transforms: np.ndarray, verbose: bool = False) -> Tuple[float, float, float]:
    """
    Compute maximum displacement from correction transforms using CUDA.

    Args:
        transforms: Array of 3x3 transformation matrices
        verbose: Unused, kept for API compatibility

    Returns:
        Tuple of (max_dx, max_dy, max_da)
    """
    if not check_cuda_available():
        raise RuntimeError("CUDA required but not available.")
    return compute_max_displacement_gpu(transforms)

