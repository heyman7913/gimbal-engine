"""
cuda-motion-flow: GPU-accelerated video stabilization.

Full CUDA pipeline:

  1. Corner detection  : raw CUDA Scharr gradient + Shi-Tomasi response kernel
  2. Feature tracking  : vectorised pyramidal Lucas-Kanade (all points parallel)
  3. Transform est.    : vectorised GPU RANSAC (all iterations parallel)
  4. Trajectory smooth : Gaussian (GPU) | Kalman-RTS (optimal) | L1-TV
  5. Frame warp        : raw CUDA bilinear affine kernel, dual-stream overlap
  6. Pose export       : JSON or COLMAP format for SfM / Gaussian Splatting

Python API
----------
    from cuda_motion_flow import stabilize_video

    stabilize_video(
        "shaky.mp4", "stable.mp4",
        smoother="kalman",          # or "gaussian" / "l1"
        smoothing_factor=0.5,
        export_trajectory="poses/", # COLMAP output directory
    )

CLI
---
    cuda-motion-flow input.mp4 output.mp4 --smoother kalman --smoothing 0.5
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
from .trajectory import smooth_trajectory, SMOOTHERS
from .geometry import (
    CameraIntrinsics,
    CameraTrajectory,
    FramePose,
    estimate_intrinsics,
    build_trajectory,
    decompose_homography,
)
from .utils import get_video_info, validate_video_file
from .metrics import compute_all_metrics, StabilizationMetrics

__version__ = "0.5.0"

__all__ = [
    # Primary API
    "stabilize_video",
    # CUDA pipeline primitives
    "check_cuda_available",
    "get_device_info",
    "compute_optical_flow_gpu",
    "estimate_transform_from_flow_gpu",
    "detect_corners_gpu",
    "track_points_gpu",
    "ransac_affine_gpu",
    # Trajectory smoothing
    "smooth_trajectory",
    "SMOOTHERS",
    # Camera geometry
    "CameraIntrinsics",
    "CameraTrajectory",
    "FramePose",
    "estimate_intrinsics",
    "build_trajectory",
    "decompose_homography",
    # Utilities
    "get_video_info",
    "validate_video_file",
    # Metrics
    "compute_all_metrics",
    "StabilizationMetrics",
]
