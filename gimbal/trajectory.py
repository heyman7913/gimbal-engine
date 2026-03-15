"""Building the camera path and smoothing it.

The estimator gives one homography per frame pair. I chain those into an absolute path (where
each frame sits relative to frame 0), smooth that path, and then work out the warp that pulls
each frame onto the smooth version.

This all runs on a few numbers per frame, not on pixels, so plain numpy is fine. I standardise
each of the 8 homography entries first, otherwise `strength` would behave totally differently
for translations (~pixels) vs the perspective terms (~1e-4).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

Smoother = Literal["gaussian", "kalman_rts", "l1_tv"]


def normalize_h(h: np.ndarray) -> np.ndarray:
    """Scale a 3x3 (or stack of) homography so the bottom-right entry is 1."""
    h = np.asarray(h, dtype=np.float64)
    out: np.ndarray = h / h[..., 2:3, 2:3]
    return out


def cumulative_path(pairwise: np.ndarray) -> np.ndarray:
    """Accumulate inter-frame homographies into absolute poses.

    pairwise[k] maps frame k -> frame k+1. Returns C of shape (M+1, 3, 3) with C[0] = I and
    C[i] mapping frame-0 coordinates to frame-i coordinates.
    """
    pairwise = normalize_h(pairwise)
    n = pairwise.shape[0] + 1
    out = np.empty((n, 3, 3), dtype=np.float64)
    out[0] = np.eye(3)
    for i in range(1, n):
        out[i] = normalize_h(pairwise[i - 1] @ out[i - 1])
    return out


def stabilizing_transforms(cumulative: np.ndarray, smoothed: np.ndarray) -> np.ndarray:
    """Per-frame warp B[i] = S[i] @ inv(C[i]) mapping original frame i to its smoothed pose.

    When the smoothed path equals the original path, every B[i] is identity.
    """
    out = np.empty_like(cumulative)
    for i in range(cumulative.shape[0]):
        out[i] = normalize_h(smoothed[i] @ np.linalg.inv(cumulative[i]))
    return out


def smooth_path(cumulative: np.ndarray, method: Smoother, strength: float) -> np.ndarray:
    """Smooth the absolute path. strength in [0, 1]; higher is smoother."""
    strength = float(np.clip(strength, 0.0, 1.0))
    n = cumulative.shape[0]
    params = cumulative.reshape(n, 9)[:, :8].copy()  # 8 free entries, h33 fixed to 1
    smoothed = np.empty_like(params)
    for c in range(8):
        col = params[:, c]
        mu = col.mean()
        sd = col.std() + 1e-9
        z = (col - mu) / sd
        smoothed[:, c] = _smooth_1d(z, method, strength) * sd + mu
    out = np.empty((n, 3, 3), dtype=np.float64)
    out.reshape(n, 9)[:, :8] = smoothed
    out[:, 2, 2] = 1.0
    return out


def _smooth_1d(z: np.ndarray, method: Smoother, strength: float) -> np.ndarray:
    if method == "gaussian":
        return _gaussian_1d(z, sigma=0.5 + strength * 24.0)
    if method == "kalman_rts":
        # smoother path -> smaller process noise relative to a unit measurement noise
        process_var = 10.0 ** (-1.0 - 4.0 * strength)
        return _rts_1d(z, process_var=process_var, meas_var=1.0)
    if method == "l1_tv":
        return _tv_1d(z, lam=1e-3 + strength * 6.0)
    raise ValueError(f"unknown smoother '{method}'")


def _gaussian_1d(z: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(1, int(round(3.0 * sigma)))
    t = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(t**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()
    padded = np.pad(z, radius, mode="reflect")
    return np.convolve(padded, kernel, mode="valid")


def _rts_1d(z: np.ndarray, process_var: float, meas_var: float) -> np.ndarray:
    """Rauch-Tung-Striebel smoother, constant-velocity model.

    The state is [position, velocity] and I return the position. It runs a Kalman filter
    forward then a correction pass backward, which is the optimal smoother for this kind of
    linear model.
    """
    n = len(z)
    if n == 1:
        return z.copy()
    f = np.array([[1.0, 1.0], [0.0, 1.0]])
    q = process_var * np.array([[0.25, 0.5], [0.5, 1.0]])
    h = np.array([[1.0, 0.0]])
    r = np.array([[meas_var]])

    xf = np.zeros((n, 2))
    pf = np.zeros((n, 2, 2))
    xp = np.zeros((n, 2))
    pp = np.zeros((n, 2, 2))

    x = np.array([z[0], 0.0])
    p = np.eye(2) * 1.0
    for k in range(n):
        if k == 0:
            xp[k], pp[k] = x, p
        else:
            xp[k] = f @ xf[k - 1]
            pp[k] = f @ pf[k - 1] @ f.T + q
        s = h @ pp[k] @ h.T + r
        gain = pp[k] @ h.T @ np.linalg.inv(s)
        xf[k] = xp[k] + (gain @ (z[k] - h @ xp[k])).ravel()
        pf[k] = (np.eye(2) - gain @ h) @ pp[k]

    xs = xf.copy()
    for k in range(n - 2, -1, -1):
        c = pf[k] @ f.T @ np.linalg.inv(pp[k + 1])
        xs[k] = xf[k] + c @ (xs[k + 1] - xp[k + 1])
    return xs[:, 0]


def _tv_1d(z: np.ndarray, lam: float, iters: int = 400) -> np.ndarray:
    """1D total-variation denoising, solved with Chambolle's dual method.

    This gives a piecewise-constant path, so a real pan stays as one smooth move while the
    little jitters get flattened out.
    """
    n = len(z)
    if n < 2:
        return z.copy()

    def diff(x: np.ndarray) -> np.ndarray:
        d: np.ndarray = x[1:] - x[:-1]
        return d

    def divt(p: np.ndarray) -> np.ndarray:
        out = np.zeros(n)
        out[:-1] -= p
        out[1:] += p
        return out

    p = np.zeros(n - 1)
    step = 1.0 / (4.0 * lam)
    for _ in range(iters):
        x = z - lam * divt(p)
        p = np.clip(p + step * diff(x), -1.0, 1.0)
    out: np.ndarray = z - lam * divt(p)
    return out
