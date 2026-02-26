# cuda-motion-flow

GPU-accelerated video stabilization built entirely on CUDA.

Every stage of the pipeline runs on the GPU: feature detection, optical flow tracking, transform estimation, trajectory smoothing, and frame warping. The recovered camera motion can also be exported as COLMAP-compatible poses, making the tool useful as a preprocessing step for 3D reconstruction pipelines (Structure from Motion, Gaussian Splatting, VPS).

---

## Pipeline

```
Input frames
    │
    ▼  Stage 1 — Corner detection
    │  Raw CUDA kernel: Scharr gradient + Shi-Tomasi response (shared-memory tiled)
    │
    ▼  Stage 2 — Feature tracking
    │  Vectorised pyramidal Lucas-Kanade: all N points processed in parallel
    │  Pyramid built with raw CUDA anti-aliased 2x downsampling kernel
    │
    ▼  Stage 3 — Transform estimation
    │  GPU RANSAC: all 500 hypotheses scored simultaneously (n_iter x n_pts grid)
    │  Least-squares refinement over inliers via cupy.linalg.lstsq
    │
    ▼  Stage 4 — Trajectory smoothing  (choose one)
    │  gaussian: GPU Gaussian convolution (fast, symmetric)
    │  kalman:   Rauch-Tung-Striebel optimal smoother (adaptive, best quality)
    │  l1:       Total-Variation / Chambolle-Pock (preserves intentional pans)
    │
    ▼  Stage 5 — Frame warping
    │  Raw CUDA bilinear affine kernel: #pragma unroll channels, __ldg() L1 cache
    │  Two non-blocking CUDA streams overlap H->D transfers with compute
    │
    ▼  Stage 6 — Camera pose export (optional)
       Homography decomposition -> R, t per frame
       JSON export or COLMAP cameras.txt / images.txt
```

---

## Installation

Requires an NVIDIA GPU with CUDA 12.x and the matching CuPy build.

```bash
pip install cupy-cuda12x          # or cupy-cuda11x for CUDA 11
pip install cuda-motion-flow
```

---

## Quick start

**CLI**

```bash
# Basic stabilization
cuda-motion-flow shaky.mp4 stable.mp4

# Kalman smoother — better quality on mixed motion
cuda-motion-flow input.mp4 output.mp4 --smoother kalman --smoothing 0.6

# L1 / Total-Variation smoother — preserves intentional pans
cuda-motion-flow vlog.mp4 vlog_stable.mp4 --smoother l1 --smoothing 0.4

# Export camera trajectory for Gaussian Splatting / SfM
cuda-motion-flow input.mp4 output.mp4 --export-trajectory ./colmap_poses/

# Full diagnostics
cuda-motion-flow input.mp4 output.mp4 --verbose

# GPU info
cuda-motion-flow --device-info
```

**Python API**

```python
from cuda_motion_flow import stabilize_video

# Gaussian smoother (default, fast)
stabilize_video("shaky.mp4", "stable.mp4", smoothing_factor=0.4)

# Kalman-RTS smoother
stabilize_video(
    "shaky.mp4", "stable.mp4",
    smoother="kalman",
    smoothing_factor=0.6,
)

# Export COLMAP trajectory for downstream 3D reconstruction
stabilize_video(
    "shaky.mp4", "stable.mp4",
    smoother="kalman",
    export_trajectory="./colmap/",   # writes cameras.txt + images.txt
)

# Export JSON trajectory
stabilize_video(
    "shaky.mp4", "stable.mp4",
    export_trajectory="trajectory.json",
)
```

---

## CLI reference

```
Usage: cuda-motion-flow [OPTIONS] INPUT_VIDEO OUTPUT_VIDEO

Options:
  --smoothing FLOAT               Smoothing strength 0.0-1.0  [default: 0.3]
  --smoother [gaussian|kalman|l1] Trajectory smoothing algorithm  [default: gaussian]
  --no-crop                       Disable auto-crop of black borders
  --no-resize                     Keep cropped resolution (do not upscale)
  --export-trajectory PATH        Export camera poses (.json or COLMAP directory)
  -v, --verbose                   Per-stage timing and diagnostics
  --device-info                   Print GPU info and exit
```

---

## Trajectory export and 3D reconstruction

The stabilizer recovers a per-frame camera trajectory as a by-product of stabilization.
This trajectory can be exported and fed into 3D reconstruction pipelines directly.

**COLMAP format** (for Gaussian Splatting, Mip-NeRF 360, etc.)

```bash
cuda-motion-flow input.mp4 stable.mp4 --export-trajectory ./colmap/
```

Writes:
```
colmap/
  cameras.txt    # PINHOLE model with estimated intrinsics
  images.txt     # Per-frame qvec + tvec (COLMAP convention)
  points3D.txt   # Empty (no point cloud)
```

**JSON format**

```json
{
  "intrinsics": {"fx": 1280.0, "fy": 1280.0, "cx": 640.0, "cy": 360.0, ...},
  "frames": [
    {
      "id": 0,
      "R": [[1,0,0],[0,1,0],[0,0,1]],
      "t": [0.0, 0.0, 0.0],
      "qvec": [1.0, 0.0, 0.0, 0.0],
      "camera_center": [0.0, 0.0, 0.0]
    }
  ]
}
```

**Direct geometry access**

```python
from cuda_motion_flow.geometry import (
    estimate_intrinsics,
    decompose_homography,
    build_trajectory,
)

K = estimate_intrinsics(width=1280, height=720)

# Decompose a homography into rotation/translation candidates
candidates = decompose_homography(H, K)
R, t, n = candidates[0]   # rotation, translation, plane normal

# Build a full trajectory from per-frame relative homographies
traj = build_trajectory(homographies, K)
traj.export_colmap("./colmap/")
traj.export_json("trajectory.json")
```

---

## Trajectory smoothers

| Smoother   | Algorithm                              | Best for                                   |
|------------|----------------------------------------|--------------------------------------------|
| `gaussian` | GPU Gaussian convolution (symmetric)   | Fast preview, short clips                  |
| `kalman`   | Rauch-Tung-Striebel optimal smoother   | General use, mixed slow/fast motion        |
| `l1`       | Total-Variation (Chambolle-Pock)       | Vlogs with intentional pans to preserve    |

The **Kalman-RTS smoother** is the globally optimal (minimum-variance) batch smoother
for a constant-velocity linear Gaussian model. Unlike Gaussian convolution, it is
adaptive — the effective smoothing window adjusts automatically to local motion magnitude.
`smoothing_strength` controls the process-to-measurement noise ratio (Q/R).

---

## Raw CUDA kernels

Performance-critical operations are implemented as raw CUDA C++ kernels compiled
at runtime via `cupy.RawKernel`, bypassing Python dispatch overhead:

| Kernel                      | Details                                                  |
|-----------------------------|----------------------------------------------------------|
| `affine_warp_bilinear_u8`   | 32x8 thread block; `__ldg()` L1 reads; `#pragma unroll` |
| `gaussian_downsample_f32`   | 16x16 tile; 36x36 shared-memory halo; separable 5-tap   |
| `scharr_gradient_f32`       | 18x18 shared-memory tile; Gx and Gy in a single pass    |
| `shi_tomasi_response_f32`   | 22x22 compile-time tile; min-eigenvalue corner response  |

All kernels support an optional `stream` argument for asynchronous execution.
Frame warping uses two non-blocking CUDA streams to overlap memory transfers with compute.

---

## Python API reference

```python
# Stabilization
stabilize_video(input_path, output_path, smoothing_factor=0.3, smoother="gaussian",
                verbose=False, auto_crop=True, preserve_resolution=True,
                export_trajectory=None)

# CUDA pipeline primitives
check_cuda_available() -> bool
get_device_info() -> dict
compute_optical_flow_gpu(prev_gray, curr_gray) -> (prev_pts, curr_pts)
estimate_transform_from_flow_gpu(prev_pts, curr_pts) -> (H, dx, dy, da)
detect_corners_gpu(img, max_corners, quality_level, min_distance) -> corners
track_points_gpu(prev, curr, pts, window_size, max_level) -> (tracked, status)
ransac_affine_gpu(src, dst, n_iterations, threshold) -> (M2x3, inlier_mask)

# Trajectory smoothing
smooth_trajectory(dx, dy, da, method, smoothing_strength) -> (N, 3, 3) ndarray

# Camera geometry
estimate_intrinsics(width, height) -> CameraIntrinsics
decompose_homography(H, K) -> List[(R, t, n)]
build_trajectory(homographies, K) -> CameraTrajectory

# Metrics
compute_all_metrics(input_path, output_path) -> StabilizationMetrics
```

---

## Requirements

- Python 3.9+
- CUDA 12.x (or 11.x with the matching CuPy build)
- `cupy-cuda12x >= 13.0`
- `opencv-python >= 4.8`
- `numpy >= 1.22`
- `click >= 8.0`

---

## License

MIT
