"""
Utility functions for video stabilization.

This module provides helper functions for video processing,
including auto-crop calculations and codec detection.
"""

import numpy as np
import cv2 as cv
from typing import Tuple, Optional


def calculate_auto_crop(
    width: int,
    height: int,
    max_dx: float,
    max_dy: float,
    max_da: float,
    padding: float = 1.05
) -> Tuple[int, int, int, int]:
    """
    Calculate crop rectangle to remove black borders after stabilization.

    Args:
        width: Original frame width
        height: Original frame height
        max_dx: Maximum X displacement in pixels
        max_dy: Maximum Y displacement in pixels
        max_da: Maximum rotation angle in radians
        padding: Safety padding multiplier (>1.0 for extra margin)

    Returns:
        Tuple of (x, y, crop_width, crop_height) for cropping
    """
    # Calculate border from translation
    border_x = int(abs(max_dx) * padding)
    border_y = int(abs(max_dy) * padding)

    # Calculate additional border from rotation
    # When rotating, corners move outward
    if max_da > 0:
        diagonal = np.sqrt(width**2 + height**2) / 2
        rotation_border = int(diagonal * abs(np.sin(max_da)) * padding)
        border_x = max(border_x, rotation_border)
        border_y = max(border_y, rotation_border)

    # Minimum border - just enough to hide minor artifacts (was 20, too aggressive)
    border_x = max(border_x, 5)
    border_y = max(border_y, 5)

    # Ensure crop doesn't exceed frame dimensions (leave at least 85% of frame)
    max_border_x = int(width * 0.075)   # Max 7.5% crop per side
    max_border_y = int(height * 0.075)
    border_x = min(border_x, max_border_x)
    border_y = min(border_y, max_border_y)

    crop_x = border_x
    crop_y = border_y
    crop_width = width - 2 * border_x
    crop_height = height - 2 * border_y

    return crop_x, crop_y, crop_width, crop_height


def apply_auto_crop(
    frame: np.ndarray,
    crop_rect: Tuple[int, int, int, int],
    output_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Apply crop to frame and optionally resize to target size.

    Args:
        frame: Input frame
        crop_rect: Tuple of (x, y, width, height)
        output_size: Optional (width, height) to resize after crop

    Returns:
        Cropped (and optionally resized) frame
    """
    x, y, w, h = crop_rect
    cropped = frame[y:y+h, x:x+w]

    if output_size is not None:
        cropped = cv.resize(cropped, output_size, interpolation=cv.INTER_LANCZOS4)

    return cropped


def get_codec_for_extension(filename: str) -> int:
    """
    Get appropriate video codec based on file extension.

    Args:
        filename: Output filename

    Returns:
        OpenCV fourcc code for the codec
    """
    ext = filename.lower().split('.')[-1]

    codec_map = {
        'mp4': 'mp4v',   # MPEG-4, widely compatible
        'avi': 'XVID',   # XVID for AVI
        'mkv': 'X264',   # H.264 for MKV
        'mov': 'avc1',   # H.264 for MOV
        'webm': 'VP80',  # VP8 for WebM
    }

    codec_name = codec_map.get(ext, 'mp4v')
    return cv.VideoWriter_fourcc(*codec_name)


def validate_video_file(filepath: str) -> Tuple[bool, str]:
    """
    Validate that a video file exists and is readable.

    Args:
        filepath: Path to video file

    Returns:
        Tuple of (is_valid, error_message)
    """
    cap = cv.VideoCapture(filepath)

    if not cap.isOpened():
        return False, f"Cannot open video file: {filepath}"

    ret, _ = cap.read()
    if not ret:
        cap.release()
        return False, f"Cannot read frames from: {filepath}"

    cap.release()
    return True, ""


def get_video_info(filepath: str) -> dict:
    """
    Get video metadata.

    Args:
        filepath: Path to video file

    Returns:
        Dictionary with video properties
    """
    cap = cv.VideoCapture(filepath)

    info = {
        "width": int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv.CAP_PROP_FRAME_COUNT)),
        "codec": int(cap.get(cv.CAP_PROP_FOURCC)),
    }

    cap.release()
    return info


def estimate_processing_time(frame_count: int, cuda_enabled: bool) -> str:
    """
    Estimate processing time based on frame count and hardware.

    Args:
        frame_count: Number of frames to process
        cuda_enabled: Whether CUDA acceleration is enabled

    Returns:
        Human-readable time estimate string
    """
    # Rough estimates: 30fps processing on CUDA, 5fps on CPU
    fps_estimate = 30 if cuda_enabled else 5
    seconds = frame_count / fps_estimate

    if seconds < 60:
        return f"~{int(seconds)} seconds"
    elif seconds < 3600:
        return f"~{int(seconds / 60)} minutes"
    else:
        return f"~{seconds / 3600:.1f} hours"


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max."""
    return max(min_val, min(max_val, value))
