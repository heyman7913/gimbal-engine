"""
Camera geometry utilities for pose extraction and trajectory export.

This module bridges video stabilization with 3D reconstruction pipelines
(Structure from Motion, Visual Positioning Systems, Gaussian Splatting)
by exposing the recovered camera motion as explicit geometric quantities.

The stabilizer accumulates per-frame homographies H_k (relative transforms
between consecutive frames). This module converts those homographies into:

  - Absolute camera poses R_k, t_k (in a common reference frame)
  - Approximate camera intrinsics K from video dimensions
  - COLMAP-compatible text exports (cameras.txt / images.txt)
  - JSON trajectory export

Homography decomposition follows:
  Malis & Vargas, "Deeper understanding of the homography decomposition for
  vision-based control," INRIA Technical Report, 2007.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2 as cv
import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsics (no distortion)."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            [self.fx,  0.0,    self.cx],
            [0.0,      self.fy, self.cy],
            [0.0,      0.0,    1.0],
        ], dtype=np.float64)


@dataclass
class FramePose:
    """Camera pose for a single frame in world coordinates.

    Convention: p_world = R @ p_cam + t
    (OpenCV / COLMAP sign convention)
    """
    frame_id: int
    R: np.ndarray          # (3, 3) rotation matrix
    t: np.ndarray          # (3,)   translation vector
    filename: str = ""

    @property
    def quaternion(self) -> np.ndarray:
        """Hamilton quaternion [qw, qx, qy, qz] from rotation matrix."""
        return _rot_to_quat(self.R)

    @property
    def camera_center(self) -> np.ndarray:
        """Camera center in world coordinates: C = -R^T @ t."""
        return -self.R.T @ self.t


@dataclass
class CameraTrajectory:
    """Sequence of camera poses forming a trajectory."""
    intrinsics: CameraIntrinsics
    poses: List[FramePose] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.poses)

    def export_json(self, path: str | Path) -> None:
        """Write trajectory to a JSON file."""
        data = {
            "intrinsics": {
                "fx": self.intrinsics.fx,
                "fy": self.intrinsics.fy,
                "cx": self.intrinsics.cx,
                "cy": self.intrinsics.cy,
                "width":  self.intrinsics.width,
                "height": self.intrinsics.height,
            },
            "frames": [
                {
                    "id":  p.frame_id,
                    "name": p.filename or f"frame_{p.frame_id:06d}.jpg",
                    "R":  p.R.tolist(),
                    "t":  p.t.tolist(),
                    "qvec": p.quaternion.tolist(),
                    "camera_center": p.camera_center.tolist(),
                }
                for p in self.poses
            ],
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def export_colmap(self, directory: str | Path) -> None:
        """
        Write COLMAP-format cameras.txt and images.txt to `directory`.

        This allows feeding the recovered trajectory into Gaussian Splatting
        pipelines (e.g. 3D-GS, Mip-NeRF 360) or any COLMAP-compatible tool.
        """
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        self._write_colmap_cameras(out / "cameras.txt")
        self._write_colmap_images(out / "images.txt")
        (out / "points3D.txt").write_text("")  # Empty; no reconstruction

    def _write_colmap_cameras(self, path: Path) -> None:
        K = self.intrinsics
        lines = [
            "# Camera list with one line of data per camera:",
            "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
            f"1 PINHOLE {K.width} {K.height} {K.fx:.6f} {K.fy:.6f} {K.cx:.6f} {K.cy:.6f}",
        ]
        path.write_text("\n".join(lines) + "\n")

    def _write_colmap_images(self, path: Path) -> None:
        lines = [
            "# Image list with two lines of data per image:",
            "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
            "# POINTS2D[] as (X, Y, POINT3D_ID)",
        ]
        for p in self.poses:
            q  = p.quaternion
            name = p.filename or f"frame_{p.frame_id:06d}.jpg"
            lines.append(
                f"{p.frame_id} "
                f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f} "
                f"{p.t[0]:.9f} {p.t[1]:.9f} {p.t[2]:.9f} "
                f"1 {name}"
            )
            lines.append("")  # Empty POINTS2D line
        path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Intrinsics estimation
# ---------------------------------------------------------------------------

def estimate_intrinsics(width: int, height: int) -> CameraIntrinsics:
    """
    Estimate camera intrinsics from frame dimensions.

    Uses the common heuristic: focal length ≈ max(w, h) in pixels
    (approximately correct for a 70° horizontal FOV, common in smartphones).
    """
    f = float(max(width, height))
    return CameraIntrinsics(
        fx=f, fy=f,
        cx=width  / 2.0,
        cy=height / 2.0,
        width=width,
        height=height,
    )


# ---------------------------------------------------------------------------
# Homography decomposition
# ---------------------------------------------------------------------------

def decompose_homography(
    H: np.ndarray,
    K: CameraIntrinsics,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Decompose a planar homography into rotation/translation candidates.

    Wraps cv2.decomposeHomographyMat which implements the SVD-based
    decomposition from Malis & Vargas (2007).

    Args:
        H: (3, 3) homography matrix in pixel coordinates.
        K: Camera intrinsics.

    Returns:
        List of (R, t, n) tuples — up to 4 candidates.
        R: (3, 3) rotation matrix
        t: (3,)   translation (scale ambiguous)
        n: (3,)   scene plane normal
    """
    Km = K.matrix
    retval, Rs, ts, ns = cv.decomposeHomographyMat(H, Km)
    return [(R, t.ravel(), n.ravel()) for R, t, n in zip(Rs, ts, ns)]


def select_best_decomposition(
    candidates: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    prev_R: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select the physically consistent decomposition.

    Rules (in order):
    1. Points must be in front of both cameras (positive depth):
       n · t > 0 and n · (R^T t) > 0  (Malis & Vargas §4.1)
    2. Among remaining, choose the one closest to the previous rotation
       (temporal continuity).

    Args:
        candidates: Output of decompose_homography().
        prev_R:     Previous frame's rotation matrix (for continuity).

    Returns:
        (R, t) best rotation and translation.
    """
    valid = []
    for R, t, n in candidates:
        # Cheirality: plane normal must face both cameras
        if np.dot(n, t) > 0 and np.dot(n, R.T @ t) > 0:
            valid.append((R, t))

    if not valid:
        valid = [(R, t) for R, t, _ in candidates]

    if prev_R is None or len(valid) == 1:
        return valid[0]

    # Pick closest to previous rotation (minimal angular distance)
    best, best_angle = valid[0], float("inf")
    for R, t in valid:
        dR = prev_R.T @ R
        angle = abs(math.acos(np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)))
        if angle < best_angle:
            best, best_angle = (R, t), angle
    return best


# ---------------------------------------------------------------------------
# Trajectory construction from per-frame homographies
# ---------------------------------------------------------------------------

def build_trajectory(
    homographies: Sequence[np.ndarray],
    intrinsics: CameraIntrinsics,
    frame_names: Optional[Sequence[str]] = None,
) -> CameraTrajectory:
    """
    Convert a sequence of relative homographies to an absolute trajectory.

    Each H_k is the homography from frame k-1 to frame k. We integrate
    these into world-frame poses by chaining rotations and translations.

    This yields a trajectory amenable to downstream 3D reconstruction —
    the same motion estimates that drive stabilization can seed SfM/VPS.

    Args:
        homographies: Per-frame relative homographies, len N (for N+1 frames).
        intrinsics:   Camera intrinsics.
        frame_names:  Optional list of filenames for COLMAP export.

    Returns:
        CameraTrajectory with N+1 poses.
    """
    traj = CameraTrajectory(intrinsics=intrinsics)

    # First pose is the identity (world origin)
    R_world = np.eye(3, dtype=np.float64)
    t_world = np.zeros(3, dtype=np.float64)
    name0   = frame_names[0] if frame_names else "frame_000000.jpg"
    traj.poses.append(FramePose(0, R_world.copy(), t_world.copy(), name0))

    prev_R = R_world
    for k, H in enumerate(homographies):
        candidates = decompose_homography(H, intrinsics)
        R_rel, t_rel = select_best_decomposition(candidates, prev_R)

        # Chain: R_world_new = R_world_old @ R_rel^T  (world ← camera)
        R_world = R_world @ R_rel.T
        t_world = R_world @ (-t_rel) + t_world

        name = frame_names[k + 1] if frame_names else f"frame_{k + 1:06d}.jpg"
        traj.poses.append(FramePose(k + 1, R_world.copy(), t_world.copy(), name))
        prev_R = R_rel

    return traj


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3×3 rotation matrix to Hamilton quaternion [qw, qx, qy, qz]."""
    m = R.astype(np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]

    if trace > 0:
        s  = 0.5 / math.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (m[2, 1] - m[1, 2]) * s
        qy = (m[0, 2] - m[2, 0]) * s
        qz = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s  = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s  = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s  = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    return q / np.linalg.norm(q)
