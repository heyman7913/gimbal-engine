"""Video decode/encode and grayscale conversion (OpenCV).

Decoding and encoding are host-side I/O, not the GPU compute path. Frames are returned as
uint8 BGR (H, W, 3); the pipeline uploads them to the device for warping.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def video_info(path: str | Path) -> dict[str, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    info = {
        "fps": float(cap.get(cv2.CAP_PROP_FPS)) or 30.0,
        "frame_count": float(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": float(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def read_video(path: str | Path) -> tuple[list[np.ndarray], float]:
    """Read all frames as uint8 BGR plus the source fps."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    return frames, fps


def write_video(path: str | Path, frames: list[np.ndarray], fps: float) -> None:
    if not frames:
        raise ValueError("no frames to write")
    h, w = frames[0].shape[:2]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer for {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def to_gray(frame: np.ndarray) -> np.ndarray:
    """uint8 BGR (H, W, 3) -> uint8 grayscale (H, W)."""
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
