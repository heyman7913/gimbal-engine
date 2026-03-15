"""The three stabilization metrics from the Liu et al. 2013 paper.

There's no ground truth on real footage, so we judge the output by:
  - cropping ratio: how much of the frame is left after cropping
  - distortion value: how much the warp stretches the frame (1 = none)
  - stability score: how much of the camera path's energy is low-frequency

All three are in (0, 1] and higher is better. MACE (corner error) is a separate, synthetic-only
metric and lives in the training code, not here.
"""

from __future__ import annotations

import numpy as np


def affine_singular_values(h: np.ndarray) -> tuple[float, float]:
    """Singular values (s1 >= s2 >= 0) of the 2x2 affine part of a homography.

    I use singular values rather than eigenvalues on purpose - a rotation has complex
    eigenvalues, but its singular values are the real per-axis scales, which is what distortion
    actually is. For a plain rotation+scale they're the same anyway.
    """
    a = np.asarray(h, dtype=np.float64)[:2, :2]
    s = np.linalg.svd(a, compute_uv=False)
    return float(s[0]), float(s[1])


def distortion_value(transforms: np.ndarray) -> float:
    """Worst-case anisotropy s2/s1 of the per-frame warps. 1 means no shape distortion."""
    ratios = []
    for h in transforms:
        s1, s2 = affine_singular_values(h)
        ratios.append(s2 / s1 if s1 > 0 else 0.0)
    return float(min(ratios)) if ratios else 1.0


def cropping_ratio(crop_w: float, crop_h: float, full_w: float, full_h: float) -> float:
    """Linear fraction of the frame retained by the auto-crop (sqrt of the area ratio)."""
    if full_w <= 0 or full_h <= 0:
        return 0.0
    area = (crop_w * crop_h) / (full_w * full_h)
    return float(np.sqrt(max(0.0, min(1.0, area))))


def _translation_rotation(path: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tx = path[:, 0, 2]
    ty = path[:, 1, 2]
    theta = np.arctan2(path[:, 1, 0], path[:, 0, 0])
    return tx, ty, theta


def _low_freq_fraction(signal: np.ndarray, n_low: int = 6) -> float:
    """Fraction of AC energy in the lowest `n_low` non-DC frequencies."""
    sig = signal - signal.mean()
    spectrum = np.abs(np.fft.rfft(sig))
    ac = spectrum[1:]
    total = float((ac**2).sum())
    if total <= 0:
        return 1.0
    low = float((ac[:n_low] ** 2).sum())
    return low / total


def stability_score(path: np.ndarray, n_low: int = 6) -> float:
    """Low-frequency energy fraction of the stabilized path, min over translation and rotation.

    `path` is the absolute (cumulative) homography sequence of the stabilized output. A steady
    camera concentrates its motion energy in a few low frequencies, so a higher score means a
    smoother trajectory.
    """
    if path.shape[0] < 4:
        return 1.0
    tx, ty, theta = _translation_rotation(path)
    trans = 0.5 * (_low_freq_fraction(tx, n_low) + _low_freq_fraction(ty, n_low))
    rot = _low_freq_fraction(theta, n_low)
    return float(min(trans, rot))
