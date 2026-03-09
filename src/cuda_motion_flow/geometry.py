"""Camera trajectory from inter-frame homographies, exported to COLMAP or JSON.

This is the only 3D-adjacent feature and it is a front-end export, not reconstruction. Pose
recovery from a homography assumes a dominant plane or near-rotational motion, so the export
is an approximation suitable for seeding an SfM / Gaussian-Splatting pipeline, not a metric
reconstruction. It is gated to the homography estimators (both tracks qualify).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])


def estimate_intrinsics(width: int, height: int) -> CameraIntrinsics:
    """PINHOLE guess with focal = max(W, H) and the principal point at the image center."""
    f = float(max(width, height))
    return CameraIntrinsics(f, f, width / 2.0, height / 2.0, width, height)


def rotation_to_quaternion(r: np.ndarray) -> np.ndarray:
    """3x3 rotation -> unit quaternion [w, x, y, z] (Hamilton), via Shepperd's method."""
    m = np.asarray(r, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    unit: np.ndarray = q / np.linalg.norm(q)
    return unit


def _select_decomposition(
    rs: list[np.ndarray], ts: list[np.ndarray], ns: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Pick the physically plausible solution: plane facing the camera, smallest rotation."""
    best = None
    best_angle = np.inf
    for r, t, n in zip(rs, ts, ns, strict=True):
        if float(n[2]) < 0:  # plane normal should face the camera (+z)
            continue
        angle = np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))
        if angle < best_angle:
            best_angle = angle
            best = (np.asarray(r), np.asarray(t).ravel())
    if best is None:
        return np.eye(3), np.zeros(3)
    return best


@dataclass
class CameraTrajectory:
    intrinsics: CameraIntrinsics
    frames: list[dict[str, Any]] = field(default_factory=list)

    def export_json(self, path: str | Path) -> None:
        k = self.intrinsics
        payload = {
            "intrinsics": {"fx": k.fx, "fy": k.fy, "cx": k.cx, "cy": k.cy},
            "frames": self.frames,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def export_colmap(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        k = self.intrinsics
        with (d / "cameras.txt").open("w", encoding="utf-8") as f:
            f.write("# Camera list\n# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n")
            f.write(f"1 PINHOLE {k.width} {k.height} {k.fx} {k.fy} {k.cx} {k.cy}\n")
        with (d / "images.txt").open("w", encoding="utf-8") as f:
            f.write("# Image list\n# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n")
            for fr in self.frames:
                q = fr["qvec"]
                t = fr["t"]
                f.write(
                    f"{fr['id'] + 1} {q[0]} {q[1]} {q[2]} {q[3]} "
                    f"{t[0]} {t[1]} {t[2]} 1 frame_{fr['id']:05d}.png\n\n"
                )
        (d / "points3D.txt").write_text("# 3D point list (empty)\n", encoding="utf-8")


def build_trajectory(pairwise: np.ndarray, intrinsics: CameraIntrinsics) -> CameraTrajectory:
    """Accumulate per-frame homographies into absolute world-to-camera poses."""
    k = intrinsics.matrix()
    r_abs = np.eye(3)
    t_abs = np.zeros(3)
    traj = CameraTrajectory(intrinsics=intrinsics)
    traj.frames.append(_frame_entry(0, r_abs, t_abs))
    for i, h in enumerate(pairwise, start=1):
        _, rs, ts, ns = cv2.decomposeHomographyMat(np.asarray(h, dtype=np.float64), k)
        r_rel, t_rel = _select_decomposition(list(rs), list(ts), list(ns))
        t_abs = r_rel @ t_abs + t_rel
        r_abs = r_rel @ r_abs
        traj.frames.append(_frame_entry(i, r_abs, t_abs))
    return traj


def _frame_entry(idx: int, r: np.ndarray, t: np.ndarray) -> dict[str, Any]:
    center = (-r.T @ t).tolist()
    return {
        "id": idx,
        "R": r.tolist(),
        "t": t.tolist(),
        "qvec": rotation_to_quaternion(r).tolist(),
        "camera_center": center,
    }
