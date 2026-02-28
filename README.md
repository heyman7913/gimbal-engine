# cuda-motion-flow

GPU-accelerated video stabilization built entirely on CUDA.

Every stage of the pipeline runs on the GPU — feature detection, optical flow,
transform estimation, trajectory smoothing, and frame warping. The recovered
camera motion can be exported as COLMAP-compatible poses for direct use in
Structure from Motion, Gaussian Splatting, and VPS pipelines.

---

## Pipeline

```
Input frames
    │
    ▼  Stage 1 — Corner detection
    │  Scharr gradient + Shi-Tomasi response via shared-memory-tiled raw CUDA kernels
    │  22×22 compile-time tile; __ldg() L1 cache reads
    │
    ▼  Stage 2 — Feature tracking
    │  Pyramidal Lucas-Kanade: all N points tracked in parallel on the GPU
    │  Pyramid built with raw CUDA 2× anti-aliased Gaussian downsampling kernel
    │
    ▼  Stage 3 — Transform estimation
    │  Vectorised GPU RANSAC: all 500 hypotheses scored simultaneously (n_iter × n_pts grid)
    │  Least-squares affine refinement over inliers via cupy.linalg.lstsq
    │
    ▼  Stage 4 — Trajectory smoothing  (choose one)
    │  gaussian  GPU Gaussian convolution — fast, symmetric, good for previews
    │  kalman    Rauch-Tung-Striebel optimal smoother — globally minimum-variance
    │  l1        Total-Variation / Chambolle-Pock — preserves intentional pans
    │
    ▼  Stage 5 — Frame warping
    │  Raw CUDA bilinear affine kernel: 32×8 blocks, #pragma unroll channels, __ldg()
    │  Two non-blocking CUDA streams overlap H→D transfers with GPU compute
    │
    ▼  Stage 6 — Camera pose export  (optional)
       Homography → R, t via Malis-Vargas decomposition (cv2.decomposeHomographyMat)
       Hamilton quaternion conversion via Shepperd's method
       JSON export or COLMAP cameras.txt / images.txt
```

---

## Installation

Check your CUDA version first:

```bash
nvcc --version
```

Then install the matching CuPy build:

```bash
# CUDA 13.x
pip install cupy-cuda13x

# CUDA 12.x
pip install cupy-cuda12x
```

Install the package:

```bash
pip install cuda-motion-flow
```

**Recommended — use a virtual environment:**

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux / macOS

pip install cupy-cuda13x
pip install cuda-motion-flow
```

---

## Quick start

**CLI**

```bash
# Basic stabilization (Gaussian smoother, default)
cuda-motion-flow shaky.mp4 stable.mp4

# Kalman-RTS smoother — best general-purpose quality
cuda-motion-flow input.mp4 output.mp4 --smoother kalman --smoothing 0.6

# L1 / Total-Variation — preserves intentional pans and cuts
cuda-motion-flow vlog.mp4 vlog_stable.mp4 --smoother l1 --smoothing 0.4

# Export camera trajectory for Gaussian Splatting / SfM
cuda-motion-flow input.mp4 output.mp4 --export-trajectory ./colmap_poses/

# GPU device info
cuda-motion-flow --device-info
```

**Python API**

```python
from cuda_motion_flow import stabilize_video

# Gaussian smoother (default)
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
```

---

## Trajectory smoothers

| Smoother   | Algorithm                             | When to use                                  |
|------------|---------------------------------------|----------------------------------------------|
| `gaussian` | Gaussian convolution                  | Fast previews, short clips                   |
| `kalman`   | Rauch-Tung-Striebel optimal smoother  | General use — best quality on mixed motion   |
| `l1`       | Total-Variation (Chambolle-Pock ADMM) | Content with intentional pans to preserve    |

**Kalman-RTS** is the globally optimal (minimum-variance) batch smoother for a
constant-velocity linear Gaussian trajectory model. Unlike Gaussian convolution,
it adapts — the effective smoothing window automatically adjusts to local motion
magnitude. `smoothing_strength` controls the process-to-measurement noise ratio (Q/R).

**L1 / Total-Variation** produces piecewise-constant trajectories. It removes
high-frequency jitter while leaving deliberate camera moves completely intact.
Solved via Chambolle-Pock primal-dual ADMM.

---

## Trajectory export and 3D reconstruction

The stabilizer recovers a per-frame camera trajectory as a by-product of
stabilization. This can be exported directly into 3D reconstruction pipelines.

**COLMAP format** (Gaussian Splatting, Mip-NeRF 360, COLMAP SfM)

```bash
cuda-motion-flow input.mp4 stable.mp4 --export-trajectory ./colmap/
```

Output:

```
colmap/
  cameras.txt    # PINHOLE model, estimated intrinsics (f = max(W, H), cx = W/2, cy = H/2)
  images.txt     # Per-frame qvec + tvec in COLMAP convention
  points3D.txt   # Empty — no point cloud from video alone
```

**JSON format**

```json
{
  "intrinsics": {"fx": 1280.0, "fy": 1280.0, "cx": 640.0, "cy": 360.0},
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

**Direct geometry API**

```python
from cuda_motion_flow.geometry import (
    estimate_intrinsics,
    decompose_homography,
    build_trajectory,
)

K = estimate_intrinsics(width=1280, height=720)

# Decompose a homography into rotation/translation candidates
candidates = decompose_homography(H, K)
R, t, n = candidates[0]    # rotation matrix, translation, plane normal

# Build a full trajectory from per-frame relative homographies
traj = build_trajectory(homographies, K)
traj.export_colmap("./colmap/")
traj.export_json("trajectory.json")
```

---

## Quality analysis

Compare an original video against stabilized outputs across five metric
categories drawn from published stabilization literature
(Grundmann 2011, Liu 2013, Wang 2013):

```bash
python compare_videos.py test.mp4 out_gaussian.mp4 out_kalman.mp4 out_l1.mp4
```

Metrics reported:

| Category     | Metrics                                                             |
|--------------|---------------------------------------------------------------------|
| Stability    | Mean / std / P95 / max motion, stability score = 1/(1+σ)           |
| Smoothness   | Velocity std (Δm), jerk std (Δ²m)                                  |
| Frequency    | High-freq power ratio, low-freq ratio, spectral centroid (fps/4 threshold) |
| Visual       | Temporal SSIM, Laplacian sharpness                                  |
| Fidelity     | SSIM vs original, PSNR vs original                                  |

---

## Raw CUDA kernels

Performance-critical operations are raw CUDA C++ kernels compiled at runtime
via `cupy.RawKernel` — no Python dispatch overhead in the hot path:

| Kernel                     | Configuration                                               |
|----------------------------|-------------------------------------------------------------|
| `affine_warp_bilinear_u8`  | 32×8 thread block, `__ldg()` L1 reads, `#pragma unroll` 3× |
| `gaussian_downsample_f32`  | 16×16 tile, 36×36 shared-memory halo, separable 5-tap       |
| `scharr_gradient_f32`      | 18×18 shared-memory tile, Gx and Gy computed in one pass    |
| `shi_tomasi_response_f32`  | 22×22 compile-time tile, min-eigenvalue corner response      |

All kernels accept an optional `stream` argument. Frame warping uses two
non-blocking CUDA streams to overlap host→device transfers with compute.

---

## CLI reference

```
Usage: cuda-motion-flow [OPTIONS] INPUT_VIDEO OUTPUT_VIDEO

Options:
  Smoothing:
    --smoother [gaussian|kalman|l1]   Trajectory smoothing algorithm  [default: gaussian]
    --smoothing FLOAT                 Smoothing strength 0.0–1.0      [default: 0.3]

  Output:
    --no-crop                         Disable auto-crop of black borders
    --no-resize                       Keep cropped resolution (do not upscale)
    --export-trajectory PATH          Export camera poses (.json or COLMAP directory)

  Diagnostics:
    -v, --verbose                     Per-stage timing
    --device-info                     Print GPU info and exit
    --help                            Show this message and exit
```

---

## Python API reference

```python
# Stabilization
stabilize_video(
    input_path, output_path,
    smoothing_factor=0.3,
    smoother="gaussian",        # "gaussian" | "kalman" | "l1"
    verbose=False,
    auto_crop=True,
    preserve_resolution=True,
    export_trajectory=None,     # path.json or colmap/directory/
)

# CUDA device
check_cuda_available() -> bool
get_device_info()      -> dict  # device_name, compute_capability, memory

# Pipeline primitives
compute_optical_flow_gpu(prev_gray, curr_gray)      -> (prev_pts, curr_pts)
estimate_transform_from_flow_gpu(prev_pts, curr_pts) -> (H, dx, dy, da)
detect_corners_gpu(img, max_corners, quality, min_dist) -> corners
track_points_gpu(prev, curr, pts, window_size, max_level) -> (tracked, status)
ransac_affine_gpu(src, dst, n_iterations, threshold)    -> (M_2x3, inlier_mask)

# Trajectory smoothing
smooth_trajectory(dx, dy, da, method, smoothing_strength) -> (N, 3, 3) ndarray

# Camera geometry
estimate_intrinsics(width, height) -> CameraIntrinsics
decompose_homography(H, K)         -> List[(R, t, n)]
build_trajectory(homographies, K)  -> CameraTrajectory
```

---

## Requirements

- Python 3.9+
- NVIDIA GPU (CUDA 12.x or 13.x)
- `cupy-cuda12x` or `cupy-cuda13x` — install separately to match `nvcc --version`
- `opencv-python >= 4.8`
- `numpy >= 1.22`
- `rich >= 13.0`
- `rich-click >= 1.7`
- `click >= 8.0`

---

## License

MIT
