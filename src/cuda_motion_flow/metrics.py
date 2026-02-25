"""
Video stabilization quality metrics.

This module provides metrics to evaluate stabilization quality by comparing
the input and output videos.
"""

import numpy as np
import cv2 as cv
import cupy as cp
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class StabilizationMetrics:
    """Container for stabilization quality metrics."""
    # Jitter metrics (lower = better)
    input_itv: float          # Inter-frame Transform Variance (input)
    output_itv: float         # Inter-frame Transform Variance (output)
    itv_reduction: float      # % reduction in jitter

    # Flow metrics (lower = smoother)
    input_flow_mean: float    # Mean optical flow magnitude (input)
    output_flow_mean: float   # Mean optical flow magnitude (output)
    input_flow_std: float     # Flow std dev (input)
    output_flow_std: float    # Flow std dev (output)

    # Quality metrics
    cropping_ratio: float     # % of original frame preserved
    mean_ssim: float          # Mean SSIM between consecutive frames

    def __str__(self) -> str:
        return f"""
╔══════════════════════════════════════════════════════════════╗
║                   STABILIZATION METRICS                       ║
╠══════════════════════════════════════════════════════════════╣
║  JITTER (Inter-frame Transform Variance)                      ║
║    Input:  {self.input_itv:>10.4f}                                       ║
║    Output: {self.output_itv:>10.4f}                                       ║
║    Reduction: {self.itv_reduction:>6.1f}%  {'✓ IMPROVED' if self.itv_reduction > 0 else '✗ WORSE':<20}    ║
╠══════════════════════════════════════════════════════════════╣
║  MOTION (Mean Optical Flow)                                   ║
║    Input:  {self.input_flow_mean:>10.4f} px/frame (σ={self.input_flow_std:.4f})           ║
║    Output: {self.output_flow_mean:>10.4f} px/frame (σ={self.output_flow_std:.4f})           ║
╠══════════════════════════════════════════════════════════════╣
║  QUALITY                                                      ║
║    Cropping Ratio: {self.cropping_ratio:>5.1f}% of original frame            ║
║    Temporal SSIM:  {self.mean_ssim:>5.3f} (1.0 = identical frames)         ║
╚══════════════════════════════════════════════════════════════╝"""


def compute_itv_gpu(transforms: np.ndarray) -> float:
    """
    Compute Inter-frame Transform Variance on GPU.

    ITV measures the variance of frame-to-frame motion - lower = less jitter.

    Args:
        transforms: Array of 3x3 transformation matrices

    Returns:
        ITV value (lower is better)
    """
    if len(transforms) < 2:
        return 0.0

    # Transfer to GPU
    T = cp.asarray(transforms, dtype=cp.float32)

    # Extract motion components
    dx = T[:, 0, 2]
    dy = T[:, 1, 2]
    da = cp.arctan2(T[:, 1, 0], T[:, 0, 0])

    # Compute variance of motion derivatives (acceleration = jitter)
    ddx = cp.diff(dx)
    ddy = cp.diff(dy)
    dda = cp.diff(da)

    # Combined variance (weighted: rotation has more visual impact)
    itv = float(cp.var(ddx) + cp.var(ddy) + 100 * cp.var(dda))

    return itv


def compute_flow_metrics_gpu(
    video_path: str,
    max_frames: int = 300,
    verbose: bool = False
) -> Tuple[float, float]:
    """
    Compute mean optical flow magnitude and std dev using GPU-accelerated Farneback.

    Args:
        video_path: Path to video file
        max_frames: Maximum frames to sample (for speed)
        verbose: Print progress

    Returns:
        Tuple of (mean_flow_magnitude, flow_std_dev)
    """
    cap = cv.VideoCapture(video_path)
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    # Sample frames evenly
    step = max(1, frame_count // max_frames)

    ret, prev_frame = cap.read()
    if not ret:
        return 0.0, 0.0

    prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)

    flow_magnitudes = []
    frame_idx = 0

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % step != 0:
            continue

        curr_gray = cv.cvtColor(curr_frame, cv.COLOR_BGR2GRAY)

        # Compute dense optical flow (Farneback)
        flow = cv.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        # Compute magnitude
        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        flow_magnitudes.append(np.mean(mag))

        prev_gray = curr_gray

    cap.release()

    if len(flow_magnitudes) == 0:
        return 0.0, 0.0

    return float(np.mean(flow_magnitudes)), float(np.std(flow_magnitudes))


def compute_ssim_consecutive(
    video_path: str,
    max_frames: int = 100,
) -> float:
    """
    Compute mean SSIM between consecutive frames.

    Higher SSIM = more temporal consistency (smoother video).

    Args:
        video_path: Path to video file
        max_frames: Maximum frames to sample

    Returns:
        Mean SSIM value (0-1, higher is better)
    """
    cap = cv.VideoCapture(video_path)
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    step = max(1, frame_count // max_frames)

    ret, prev_frame = cap.read()
    if not ret:
        return 0.0

    prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)

    ssim_values = []
    frame_idx = 0

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % step != 0:
            continue

        curr_gray = cv.cvtColor(curr_frame, cv.COLOR_BGR2GRAY)

        # Compute SSIM
        ssim = _compute_ssim(prev_gray, curr_gray)
        ssim_values.append(ssim)

        prev_gray = curr_gray

    cap.release()

    if len(ssim_values) == 0:
        return 0.0

    return float(np.mean(ssim_values))


def _compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute SSIM between two grayscale images."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    kernel = cv.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv.filter2D(img2, -1, window)[5:-5, 5:-5]

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return float(ssim_map.mean())


def compute_transforms_from_video(video_path: str, verbose: bool = False) -> np.ndarray:
    """
    Extract frame-to-frame transforms from a video for ITV calculation.

    Args:
        video_path: Path to video
        verbose: Print progress

    Returns:
        Array of 3x3 transformation matrices
    """
    cap = cv.VideoCapture(video_path)
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    ret, prev_frame = cap.read()
    if not ret:
        return np.array([])

    prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)

    transforms = []
    frame_idx = 0

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        curr_gray = cv.cvtColor(curr_frame, cv.COLOR_BGR2GRAY)

        # Detect features
        prev_pts = cv.goodFeaturesToTrack(
            prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30
        )

        if prev_pts is not None and len(prev_pts) >= 3:
            curr_pts, status, _ = cv.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, prev_pts, None
            )

            status = status.flatten()
            prev_good = prev_pts[status == 1]
            curr_good = curr_pts[status == 1]

            if len(prev_good) >= 3:
                T, _ = cv.estimateAffinePartial2D(prev_good, curr_good)
                if T is not None:
                    T3x3 = np.vstack([T, [0, 0, 1]]).astype(np.float32)
                    transforms.append(T3x3)
                else:
                    transforms.append(np.eye(3, dtype=np.float32))
            else:
                transforms.append(np.eye(3, dtype=np.float32))
        else:
            transforms.append(np.eye(3, dtype=np.float32))

        prev_gray = curr_gray
        frame_idx += 1

        if verbose and frame_idx % 100 == 0:
            print(f"    Analyzing frame {frame_idx}/{frame_count}...")

    cap.release()
    return np.array(transforms, dtype=np.float32)


def compute_all_metrics(
    input_path: str,
    output_path: str,
    crop_rect: Optional[Tuple[int, int, int, int]] = None,
    original_size: Optional[Tuple[int, int]] = None,
    verbose: bool = False
) -> StabilizationMetrics:
    """
    Compute all stabilization quality metrics.

    Args:
        input_path: Path to original video
        output_path: Path to stabilized video
        crop_rect: Crop rectangle used (x, y, w, h)
        original_size: Original frame size (w, h)
        verbose: Print progress

    Returns:
        StabilizationMetrics object
    """
    if verbose:
        print("Computing stabilization metrics...")

    # Compute ITV for both videos
    if verbose:
        print("  Analyzing input video jitter...")
    input_transforms = compute_transforms_from_video(input_path, verbose=False)
    input_itv = compute_itv_gpu(input_transforms)

    if verbose:
        print("  Analyzing output video jitter...")
    output_transforms = compute_transforms_from_video(output_path, verbose=False)
    output_itv = compute_itv_gpu(output_transforms)

    itv_reduction = ((input_itv - output_itv) / max(input_itv, 0.0001)) * 100

    # Compute flow metrics
    if verbose:
        print("  Computing optical flow metrics...")
    input_flow_mean, input_flow_std = compute_flow_metrics_gpu(input_path, verbose=False)
    output_flow_mean, output_flow_std = compute_flow_metrics_gpu(output_path, verbose=False)

    # Compute cropping ratio
    if crop_rect and original_size:
        _, _, crop_w, crop_h = crop_rect
        orig_w, orig_h = original_size
        cropping_ratio = (crop_w * crop_h) / (orig_w * orig_h) * 100
    else:
        cropping_ratio = 100.0

    # Compute SSIM for output
    if verbose:
        print("  Computing temporal consistency (SSIM)...")
    mean_ssim = compute_ssim_consecutive(output_path)

    return StabilizationMetrics(
        input_itv=input_itv,
        output_itv=output_itv,
        itv_reduction=itv_reduction,
        input_flow_mean=input_flow_mean,
        output_flow_mean=output_flow_mean,
        input_flow_std=input_flow_std,
        output_flow_std=output_flow_std,
        cropping_ratio=cropping_ratio,
        mean_ssim=mean_ssim,
    )
