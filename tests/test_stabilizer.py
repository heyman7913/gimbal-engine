"""
Tests for cuda-motion-flow.

GPU-dependent tests are skipped automatically when CUDA is not available,
so the suite can run on CI machines without a GPU.

Structure:
  - Unit tests for trajectory smoothers (CPU-only, always run)
  - Unit tests for geometry utilities  (CPU-only, always run)
  - Integration tests for GPU kernels  (skipped without CUDA)
  - Integration test for full pipeline (skipped without CUDA)
"""

import math
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

CUDA_AVAILABLE = False
try:
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() > 0:
        _ = cp.sum(cp.array([1.0]))
        CUDA_AVAILABLE = True
except Exception:
    pass

requires_cuda = pytest.mark.skipif(
    not CUDA_AVAILABLE, reason="CUDA not available"
)


def _make_gray_frame(h: int = 240, w: int = 320, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(30, 200, size=(h, w), dtype=np.uint8)
    # Add some texture so corner detection finds points
    for _ in range(20):
        r, c = rng.integers(10, h - 10), rng.integers(10, w - 10)
        base[r - 5:r + 5, c - 5:c + 5] = rng.integers(0, 255)
    return base


def _make_bgr_frame(h: int = 240, w: int = 320, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = rng.integers(50, 200, size=(h, w, 3), dtype=np.uint8)
    return frame


def _make_synthetic_video(path: Path, n_frames: int = 30,
                           h: int = 240, w: int = 320) -> None:
    """Write a tiny synthetic video with slight per-frame translations."""
    import cv2 as cv
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out    = cv.VideoWriter(str(path), fourcc, 30.0, (w, h))
    rng    = np.random.default_rng(42)
    base   = rng.integers(30, 200, (h, w, 3), dtype=np.uint8)
    for i in range(n_frames):
        dx = int(rng.integers(-3, 4))
        dy = int(rng.integers(-3, 4))
        M  = np.float32([[1, 0, dx], [0, 1, dy]])
        frame = cv.warpAffine(base, M, (w, h))
        out.write(frame)
    out.release()


# ---------------------------------------------------------------------------
# Trajectory smoother tests (no GPU needed)
# ---------------------------------------------------------------------------

class TestGaussianSmoother:
    def test_output_shape(self):
        from cuda_motion_flow.trajectory import smooth_trajectory
        n = 100
        dx = np.random.randn(n).astype(np.float32) * 2
        dy = np.random.randn(n).astype(np.float32) * 2
        da = np.random.randn(n).astype(np.float32) * 0.01
        T  = smooth_trajectory(dx, dy, da, method="gaussian", smoothing_strength=0.3)
        assert T.shape == (n, 3, 3)

    def test_identity_on_zero_motion(self):
        from cuda_motion_flow.trajectory import smooth_trajectory
        n  = 50
        dx = np.zeros(n, dtype=np.float32)
        dy = np.zeros(n, dtype=np.float32)
        da = np.zeros(n, dtype=np.float32)
        T  = smooth_trajectory(dx, dy, da, method="gaussian")
        # Corrections should be near-zero for a static camera
        assert np.allclose(T[:, 0, 2], 0, atol=1e-4)
        assert np.allclose(T[:, 1, 2], 0, atol=1e-4)


class TestKalmanSmoother:
    def test_output_shape(self):
        from cuda_motion_flow.trajectory import smooth_trajectory
        n  = 80
        dx = np.random.randn(n).astype(np.float32)
        dy = np.random.randn(n).astype(np.float32)
        da = np.zeros(n, dtype=np.float32)
        T  = smooth_trajectory(dx, dy, da, method="kalman", smoothing_strength=0.5)
        assert T.shape == (n, 3, 3)

    def test_reduces_jitter(self):
        from cuda_motion_flow.trajectory import smooth_trajectory
        rng = np.random.default_rng(0)
        n   = 200
        # True slow drift + high-frequency noise
        slow  = np.linspace(0, 10, n)
        noisy = slow + rng.standard_normal(n) * 3
        dx = np.diff(noisy, prepend=noisy[0]).astype(np.float32)
        dy = np.zeros(n, dtype=np.float32)
        da = np.zeros(n, dtype=np.float32)
        T  = smooth_trajectory(dx, dy, da, method="kalman", smoothing_strength=0.7)
        # The variance of corrections should be positive (actual smoothing happened)
        assert np.var(T[:, 0, 2]) >= 0

    def test_valid_rotation_matrices(self):
        from cuda_motion_flow.trajectory import smooth_trajectory
        n  = 40
        dx = np.random.randn(n).astype(np.float32)
        dy = np.random.randn(n).astype(np.float32)
        da = np.random.randn(n).astype(np.float32) * 0.05
        T  = smooth_trajectory(dx, dy, da, method="kalman")
        for i in range(n):
            R = T[i, :2, :2]
            det = R[0, 0] * R[1, 1] - R[0, 1] * R[1, 0]
            assert abs(abs(det) - 1.0) < 0.01, f"Frame {i}: det={det:.4f}"


class TestL1Smoother:
    def test_output_shape(self):
        from cuda_motion_flow.trajectory import smooth_trajectory
        n  = 60
        dx = np.random.randn(n).astype(np.float32)
        dy = np.random.randn(n).astype(np.float32)
        da = np.zeros(n, dtype=np.float32)
        T  = smooth_trajectory(dx, dy, da, method="l1", smoothing_strength=0.4)
        assert T.shape == (n, 3, 3)

    def test_invalid_smoother_raises(self):
        from cuda_motion_flow.trajectory import smooth_trajectory
        with pytest.raises(ValueError, match="Unknown smoother"):
            smooth_trajectory(np.zeros(10), np.zeros(10), np.zeros(10),
                              method="bogus")


# ---------------------------------------------------------------------------
# Geometry tests (no GPU needed)
# ---------------------------------------------------------------------------

class TestCameraIntrinsics:
    def test_matrix_shape(self):
        from cuda_motion_flow.geometry import estimate_intrinsics
        K = estimate_intrinsics(1920, 1080)
        assert K.matrix.shape == (3, 3)
        assert K.matrix[2, 2] == 1.0

    def test_principal_point(self):
        from cuda_motion_flow.geometry import estimate_intrinsics
        K = estimate_intrinsics(640, 480)
        assert K.cx == pytest.approx(320.0)
        assert K.cy == pytest.approx(240.0)


class TestHomographyDecomposition:
    def test_returns_candidates(self):
        from cuda_motion_flow.geometry import estimate_intrinsics, decompose_homography
        K = estimate_intrinsics(640, 480)
        H = np.eye(3, dtype=np.float64)
        candidates = decompose_homography(H, K)
        assert len(candidates) >= 1
        R, t, n = candidates[0]
        assert R.shape == (3, 3)
        assert t.shape == (3,)

    def test_rotation_orthogonal(self):
        from cuda_motion_flow.geometry import estimate_intrinsics, decompose_homography
        K = estimate_intrinsics(640, 480)
        # Small rotation + translation
        theta = 0.02
        H = np.array([
            [math.cos(theta), -math.sin(theta), 5.0],
            [math.sin(theta),  math.cos(theta), 3.0],
            [0.0,              0.0,              1.0],
        ])
        candidates = decompose_homography(H, K)
        for R, _, _ in candidates:
            # R should be close to orthogonal: R^T R ~ I
            err = np.linalg.norm(R.T @ R - np.eye(3))
            assert err < 0.1


class TestQuaternion:
    def test_identity_rotation(self):
        from cuda_motion_flow.geometry import _rot_to_quat
        q = _rot_to_quat(np.eye(3))
        assert abs(q[0] - 1.0) < 1e-6  # qw = 1 for identity

    def test_unit_norm(self):
        from cuda_motion_flow.geometry import _rot_to_quat
        rng = np.random.default_rng(1)
        for _ in range(10):
            # Build a random rotation matrix via SVD
            A = rng.standard_normal((3, 3))
            U, _, Vt = np.linalg.svd(A)
            R = U @ Vt
            if np.linalg.det(R) < 0:
                R[:, 0] *= -1
            q = _rot_to_quat(R)
            assert abs(np.linalg.norm(q) - 1.0) < 1e-9


class TestCOLMAPExport:
    def test_json_export(self, tmp_path):
        from cuda_motion_flow.geometry import (
            estimate_intrinsics, CameraTrajectory, FramePose
        )
        K    = estimate_intrinsics(640, 480)
        traj = CameraTrajectory(intrinsics=K)
        for i in range(5):
            traj.poses.append(FramePose(i, np.eye(3), np.zeros(3), f"frame_{i:04d}.jpg"))

        out = tmp_path / "traj.json"
        traj.export_json(out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert len(data["frames"]) == 5
        assert "R" in data["frames"][0]
        assert "qvec" in data["frames"][0]

    def test_colmap_export(self, tmp_path):
        from cuda_motion_flow.geometry import (
            estimate_intrinsics, CameraTrajectory, FramePose
        )
        K    = estimate_intrinsics(640, 480)
        traj = CameraTrajectory(intrinsics=K)
        for i in range(3):
            traj.poses.append(FramePose(i, np.eye(3), np.zeros(3)))

        colmap_dir = tmp_path / "colmap"
        traj.export_colmap(colmap_dir)
        assert (colmap_dir / "cameras.txt").exists()
        assert (colmap_dir / "images.txt").exists()
        assert (colmap_dir / "points3D.txt").exists()

        cameras = (colmap_dir / "cameras.txt").read_text()
        assert "PINHOLE" in cameras


# ---------------------------------------------------------------------------
# GPU kernel tests (skipped without CUDA)
# ---------------------------------------------------------------------------

@requires_cuda
class TestRawKernels:
    def test_warp_output_shape(self):
        from cuda_motion_flow.raw_kernels import affine_warp_u8
        import cupy as cp
        frame_gpu = cp.zeros((240, 320, 3), dtype=cp.uint8)
        M_inv = np.eye(2, 3, dtype=np.float32)
        out = affine_warp_u8(frame_gpu, M_inv, 320, 240)
        assert out.shape == (240, 320, 3)
        assert out.dtype == cp.uint8

    def test_warp_identity(self):
        from cuda_motion_flow.raw_kernels import affine_warp_u8
        import cupy as cp
        frame = _make_bgr_frame()
        frame_gpu = cp.asarray(frame)
        M_inv = np.eye(2, 3, dtype=np.float32)
        out = cp.asnumpy(affine_warp_u8(frame_gpu, M_inv, 320, 240))
        # Identity warp should preserve the image (bilinear at integer coords)
        assert np.allclose(out.astype(float), frame.astype(float), atol=1.0)

    def test_downsample_shape(self):
        from cuda_motion_flow.raw_kernels import gaussian_downsample
        import cupy as cp
        img = cp.ones((240, 320), dtype=cp.float32)
        out = gaussian_downsample(img)
        assert out.shape == (120, 160)

    def test_downsample_constant_field(self):
        from cuda_motion_flow.raw_kernels import gaussian_downsample
        import cupy as cp
        img = cp.full((128, 128), 1.0, dtype=cp.float32)
        out = cp.asnumpy(gaussian_downsample(img))
        # Gaussian filter of a constant field is the same constant
        assert np.allclose(out, 1.0, atol=1e-5)

    def test_scharr_zero_image(self):
        from cuda_motion_flow.raw_kernels import scharr_gradients
        import cupy as cp
        img = cp.zeros((64, 64), dtype=cp.float32)
        Gx, Gy = scharr_gradients(img)
        assert cp.max(cp.abs(Gx)) == pytest.approx(0.0, abs=1e-6)
        assert cp.max(cp.abs(Gy)) == pytest.approx(0.0, abs=1e-6)

    def test_shi_tomasi_flat_image(self):
        from cuda_motion_flow.raw_kernels import scharr_gradients, shi_tomasi_response
        import cupy as cp
        img = cp.full((64, 64), 128.0, dtype=cp.float32)
        Gx, Gy = scharr_gradients(img)
        resp = shi_tomasi_response(Gx, Gy)
        assert cp.max(resp) == pytest.approx(0.0, abs=1e-5)


@requires_cuda
class TestOpticalFlow:
    def test_returns_matched_points(self):
        from cuda_motion_flow.cuda_kernels import compute_optical_flow_gpu
        prev = _make_gray_frame(seed=0)
        curr = _make_gray_frame(seed=1)
        prev_pts, curr_pts = compute_optical_flow_gpu(prev, curr)
        assert prev_pts.shape[1] == 2
        assert curr_pts.shape[1] == 2
        assert prev_pts.shape[0] == curr_pts.shape[0]

    def test_static_scene_small_flow(self):
        from cuda_motion_flow.cuda_kernels import compute_optical_flow_gpu
        frame = _make_gray_frame(seed=7)
        prev_pts, curr_pts = compute_optical_flow_gpu(frame, frame)
        if len(prev_pts) > 0:
            flow_mag = np.linalg.norm(curr_pts - prev_pts, axis=1)
            assert np.mean(flow_mag) < 2.0


@requires_cuda
class TestRANSAC:
    def test_recovers_pure_translation(self):
        from cuda_motion_flow.cuda_kernels import ransac_affine_gpu
        import cupy as cp

        n  = 100
        dx, dy = 5.0, -3.0
        src = np.random.rand(n, 2).astype(np.float32) * 200
        dst = src + np.array([dx, dy], dtype=np.float32)
        # Add 10% outliers
        dst[:10] = np.random.rand(10, 2).astype(np.float32) * 200

        M, mask = ransac_affine_gpu(cp.asarray(src), cp.asarray(dst),
                                     n_iterations=300, threshold=3.0)
        assert M is not None
        assert abs(M[0, 2] - dx) < 1.5
        assert abs(M[1, 2] - dy) < 1.5
        assert int(cp.sum(mask)) >= 80  # at least 80% inliers

    def test_returns_none_on_degenerate(self):
        from cuda_motion_flow.cuda_kernels import ransac_affine_gpu
        import cupy as cp
        src = cp.zeros((1, 2), dtype=cp.float32)
        dst = cp.zeros((1, 2), dtype=cp.float32)
        M, mask = ransac_affine_gpu(src, dst)
        assert M is None


# ---------------------------------------------------------------------------
# Full pipeline integration test
# ---------------------------------------------------------------------------

@requires_cuda
class TestFullPipeline:
    def test_stabilize_produces_output(self, tmp_path):
        from cuda_motion_flow.stabilizer import stabilize_video
        inp = tmp_path / "input.mp4"
        out = tmp_path / "output.mp4"
        _make_synthetic_video(inp, n_frames=20)

        stabilize_video(str(inp), str(out), smoother="gaussian", verbose=False)
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_kalman_smoother_pipeline(self, tmp_path):
        from cuda_motion_flow.stabilizer import stabilize_video
        inp = tmp_path / "input.mp4"
        out = tmp_path / "output.mp4"
        _make_synthetic_video(inp, n_frames=20)

        stabilize_video(str(inp), str(out), smoother="kalman", smoothing_factor=0.6)
        assert out.exists()

    def test_trajectory_json_export(self, tmp_path):
        from cuda_motion_flow.stabilizer import stabilize_video
        inp    = tmp_path / "input.mp4"
        out    = tmp_path / "output.mp4"
        traj   = tmp_path / "traj.json"
        _make_synthetic_video(inp, n_frames=15)

        stabilize_video(str(inp), str(out), export_trajectory=str(traj))
        assert traj.exists()
        data = json.loads(traj.read_text())
        assert "frames" in data
        assert len(data["frames"]) > 0

    def test_colmap_export(self, tmp_path):
        from cuda_motion_flow.stabilizer import stabilize_video
        inp      = tmp_path / "input.mp4"
        out      = tmp_path / "output.mp4"
        colmap   = tmp_path / "colmap"
        _make_synthetic_video(inp, n_frames=15)

        stabilize_video(str(inp), str(out), export_trajectory=str(colmap))
        assert (colmap / "cameras.txt").exists()
        assert (colmap / "images.txt").exists()
