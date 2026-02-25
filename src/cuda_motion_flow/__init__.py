"""
cuda-motion-flow: GPU-accelerated video stabilization.

This package provides high-performance video stabilization using:
- GPU-accelerated sparse optical flow (CuPy Shi-Tomasi + Lucas-Kanade)
- GPU-accelerated RANSAC transform estimation (CuPy)
- GPU-accelerated trajectory smoothing (CuPy CUDA)
- GPU-accelerated frame warping (CuPy CUDA)
- Auto-crop to remove black borders
- Quality metrics for before/after comparison

Full GPU Pipeline (pure CuPy, no OpenCV CUDA needed):
1. Feature Detection: CuPy Shi-Tomasi corner detection
2. Feature Tracking: CuPy pyramidal Lucas-Kanade optical flow
3. Transform Estimation: CuPy RANSAC affine estimation
4. Trajectory Smoothing: CuPy Gaussian 1D convolution
5. Frame Warping: CuPy affine warp with scipy.ndimage.map_coordinates

Usage:
    CLI: cuda-motion-flow input.mp4 output.mp4 --smoothing 0.5 --verbose

    Python API:
        from cuda_motion_flow import stabilize_video
        stabilize_video("input.mp4", "output.mp4", smoothing_factor=0.5)
"""

from .stabilizer import stabilize_video
from .cuda_kernels import (
    check_cuda_available,
    get_device_info,
    compute_optical_flow_gpu,
    estimate_transform_from_flow_gpu,
    detect_corners_gpu,
    track_points_gpu,
    ransac_affine_gpu,
)
from .utils import get_video_info, validate_video_file
from .metrics import compute_all_metrics, StabilizationMetrics

__version__ = "0.4.0"
__all__ = [
    "stabilize_video",
    "check_cuda_available",
    "compute_optical_flow_gpu",
    "estimate_transform_from_flow_gpu",
    "detect_corners_gpu",
    "track_points_gpu",
    "ransac_affine_gpu",
    "get_device_info",
    "get_video_info",
    "validate_video_file",
    "compute_all_metrics",
    "StabilizationMetrics",
]
