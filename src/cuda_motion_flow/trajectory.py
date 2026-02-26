"""
Camera trajectory smoothing strategies.

Three algorithms are provided, each with different trade-offs:

  gaussian  — Gaussian convolution on GPU. Fast, simple, symmetric.
  kalman    — Rauch-Tung-Striebel (RTS) optimal smoother. Adaptive,
              handles both slow drifts and fast pans correctly.
  l1        — Total-variation (Chambolle-Pock) smoothing. Piecewise-
              constant trajectory; preserves intentional camera cuts.

All strategies take per-frame deltas (dx, dy, da), compute the cumulative
trajectory, smooth it, and return (N, 3, 3) correction transform matrices.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_correction_transforms(
    corr_x: np.ndarray,
    corr_y: np.ndarray,
    corr_a: np.ndarray,
) -> np.ndarray:
    """Assemble (N, 3, 3) rotation-translation correction matrices."""
    n = len(corr_x)
    cos_a = np.cos(corr_a)
    sin_a = np.sin(corr_a)

    T = np.zeros((n, 3, 3), dtype=np.float32)
    T[:, 0, 0] =  cos_a
    T[:, 0, 1] = -sin_a
    T[:, 0, 2] =  corr_x
    T[:, 1, 0] =  sin_a
    T[:, 1, 1] =  cos_a
    T[:, 1, 2] =  corr_y
    T[:, 2, 2] =  1.0
    return T


# ---------------------------------------------------------------------------
# Strategy 1: Gaussian (GPU)
# ---------------------------------------------------------------------------

def smooth_trajectory_gaussian(
    dx: np.ndarray,
    dy: np.ndarray,
    da: np.ndarray,
    smoothing_strength: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Smooth cumulative trajectory with Gaussian convolution.

    Returns (corr_x, corr_y, corr_a) correction arrays.
    """
    smoothing_strength = float(np.clip(smoothing_strength, 0.0, 1.0))
    window = int(5 + smoothing_strength * 96)
    if window % 2 == 0:
        window += 1
    sigma = window / 6.0

    # Trajectory arrays are tiny (frame count) — no benefit from GPU FFT,
    # and it requires cuFFT DLLs from the full CUDA toolkit install.
    traj_x_np = np.cumsum(np.asarray(dx, dtype=np.float32))
    traj_y_np = np.cumsum(np.asarray(dy, dtype=np.float32))
    traj_a_np = np.cumsum(np.asarray(da, dtype=np.float32))

    pad = window // 2
    tx_pad = np.pad(traj_x_np, (pad, pad), mode='edge')
    ty_pad = np.pad(traj_y_np, (pad, pad), mode='edge')
    ta_pad = np.pad(traj_a_np, (pad, pad), mode='edge')

    kernel = np.exp(-np.arange(-(window // 2), window // 2 + 1,
                               dtype=np.float32) ** 2 / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()

    sx = np.convolve(tx_pad, kernel, mode='valid')
    sy = np.convolve(ty_pad, kernel, mode='valid')
    sa = np.convolve(ta_pad, kernel, mode='valid')

    return sx - traj_x_np, sy - traj_y_np, sa - traj_a_np


# ---------------------------------------------------------------------------
# Strategy 2: Kalman RTS smoother (CPU, optimal)
# ---------------------------------------------------------------------------

def smooth_trajectory_kalman(
    dx: np.ndarray,
    dy: np.ndarray,
    da: np.ndarray,
    smoothing_strength: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Smooth cumulative trajectory using the Rauch-Tung-Striebel (RTS) smoother.

    The RTS smoother is the globally optimal (minimum-variance) batch smoother
    for linear Gaussian systems — superior to causal Kalman filtering or
    windowed Gaussian convolution.

    State:   s = [position, velocity]^T
    Process: s_{k+1} = F s_k + w,  w ~ N(0, Q)
    Measure: z_k = H s_k + v,      v ~ N(0, R)

    `smoothing_strength` controls Q/R ratio:
        → 0  (trust measurements): less temporal smoothing
        → 1  (trust process model): stronger smoothing

    Returns:
        (corr_x, corr_y, corr_a) correction arrays as float32 NumPy arrays.
    """
    smoothing_strength = float(np.clip(smoothing_strength, 0.01, 0.99))

    # Tune process/measurement noise from the smoothing parameter
    q_var = ((1.0 - smoothing_strength) ** 2) * 1e-2
    r_var = (smoothing_strength ** 2) * 10.0

    traj_x = np.cumsum(dx).astype(np.float64)
    traj_y = np.cumsum(dy).astype(np.float64)
    traj_a = np.cumsum(da).astype(np.float64)

    smoothed_x = _rts_smooth_1d(traj_x, q_var, r_var)
    smoothed_y = _rts_smooth_1d(traj_y, q_var, r_var)
    smoothed_a = _rts_smooth_1d(traj_a, q_var, r_var)

    corr_x = (smoothed_x - traj_x).astype(np.float32)
    corr_y = (smoothed_y - traj_y).astype(np.float32)
    corr_a = (smoothed_a - traj_a).astype(np.float32)
    return corr_x, corr_y, corr_a


def _rts_smooth_1d(z: np.ndarray, q_var: float, r_var: float) -> np.ndarray:
    """
    Rauch-Tung-Striebel smoother for a 1D constant-velocity trajectory.

    State dimension: 2 ([position, velocity])
    Measurement dimension: 1 ([position])
    """
    n = len(z)

    # System matrices
    F = np.array([[1.0, 1.0], [0.0, 1.0]])      # State transition
    H = np.array([[1.0, 0.0]])                   # Measurement extraction

    # Noise covariances — Q uses the discrete-time Wiener process model
    Q = q_var * np.array([[0.25, 0.5], [0.5, 1.0]])
    R = np.array([[r_var]])

    # Storage
    x_pred  = np.zeros((n, 2))
    P_pred  = np.zeros((n, 2, 2))
    x_filt  = np.zeros((n, 2))
    P_filt  = np.zeros((n, 2, 2))

    # Forward pass (Kalman filter)
    x = np.array([z[0], 0.0])
    P = np.eye(2)

    for k in range(n):
        # Predict
        xp = F @ x
        Pp = F @ P @ F.T + Q
        x_pred[k] = xp
        P_pred[k] = Pp

        # Update
        S  = H @ Pp @ H.T + R
        K  = Pp @ H.T @ np.linalg.inv(S)
        x  = xp + K @ (z[k:k+1] - H @ xp)
        P  = (np.eye(2) - K @ H) @ Pp
        x_filt[k] = x
        P_filt[k] = P

    # Backward pass (RTS smoother)
    x_smooth    = np.zeros((n, 2))
    x_smooth[-1] = x_filt[-1]
    P_smooth    = np.zeros((n, 2, 2))
    P_smooth[-1] = P_filt[-1]

    for k in range(n - 2, -1, -1):
        G = P_filt[k] @ F.T @ np.linalg.inv(P_pred[k + 1])
        x_smooth[k] = x_filt[k] + G @ (x_smooth[k + 1] - x_pred[k + 1])
        P_smooth[k] = P_filt[k] + G @ (P_smooth[k + 1] - P_pred[k + 1]) @ G.T

    return x_smooth[:, 0]  # Return smoothed positions


# ---------------------------------------------------------------------------
# Strategy 3: L1 / Total-Variation (Chambolle-Pock, CPU)
# ---------------------------------------------------------------------------

def smooth_trajectory_l1(
    dx: np.ndarray,
    dy: np.ndarray,
    da: np.ndarray,
    smoothing_strength: float = 0.5,
    iterations: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Smooth trajectory via Total-Variation minimisation (Chambolle-Pock).

    Solves:  min_p  (1/2)||p - z||^2  +  lambda * TV(p)
    where    TV(p) = sum_k |p_{k+1} - p_k|

    The TV prior promotes piecewise-constant trajectories, making this the
    best choice when intentional camera pans should be preserved.

    Returns:
        (corr_x, corr_y, corr_a) correction arrays as float32 NumPy arrays.
    """
    smoothing_strength = float(np.clip(smoothing_strength, 0.0, 1.0))
    lam = smoothing_strength * 50.0

    traj_x = np.cumsum(dx).astype(np.float64)
    traj_y = np.cumsum(dy).astype(np.float64)
    traj_a = np.cumsum(da).astype(np.float64)

    sx = _tv_smooth_1d(traj_x, lam, iterations)
    sy = _tv_smooth_1d(traj_y, lam, iterations)
    sa = _tv_smooth_1d(traj_a, lam, iterations)

    corr_x = (sx - traj_x).astype(np.float32)
    corr_y = (sy - traj_y).astype(np.float32)
    corr_a = (sa - traj_a).astype(np.float32)
    return corr_x, corr_y, corr_a


def _tv_smooth_1d(z: np.ndarray, lam: float, iterations: int) -> np.ndarray:
    """
    1D TV denoising via Chambolle-Pock primal-dual algorithm.

    Operator norm of finite-difference matrix: L = 2, so step sizes:
        tau = sigma = 1 / L = 0.5 for guaranteed convergence.
    """
    n   = len(z)
    p   = np.zeros(n - 1)   # Dual variable (finite differences)
    u   = z.copy()           # Primal variable

    tau   = 0.45            # Slightly conservative for stability
    sigma = 0.45
    theta = 1.0             # Over-relaxation (PDHG variant)

    u_bar = u.copy()

    for _ in range(iterations):
        # Dual update: p = prox_{sigma * || . ||_inf}(p + sigma * D u_bar)
        diff  = u_bar[1:] - u_bar[:-1]
        p_new = p + sigma * diff
        # Project onto |p| <= lam
        p     = np.clip(p_new, -lam, lam)

        # Primal update: u = prox_{tau * (1/2)||.||^2}(u - tau * D^T p)
        div_p          = np.zeros(n)
        div_p[:-1]    -= p
        div_p[1:]     += p

        u_old = u.copy()
        u     = (u - tau * div_p + tau * z) / (1.0 + tau)

        # Over-relaxation
        u_bar = u + theta * (u - u_old)

    return u


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

SMOOTHERS = {
    "gaussian": smooth_trajectory_gaussian,
    "kalman":   smooth_trajectory_kalman,
    "l1":       smooth_trajectory_l1,
}


def smooth_trajectory(
    dx: np.ndarray,
    dy: np.ndarray,
    da: np.ndarray,
    method: str = "gaussian",
    smoothing_strength: float = 0.3,
) -> np.ndarray:
    """
    Smooth the camera trajectory and return (N, 3, 3) correction transforms.

    Args:
        dx, dy, da:        Per-frame delta arrays (inter-frame motion).
        method:            'gaussian', 'kalman', or 'l1'.
        smoothing_strength: 0.0 = minimal smoothing, 1.0 = maximum.

    Returns:
        (N, 3, 3) float32 array of correction affine matrices.

    Raises:
        ValueError: If `method` is not recognised.
    """
    if method not in SMOOTHERS:
        raise ValueError(f"Unknown smoother '{method}'. Choose from {list(SMOOTHERS)}")

    fn = SMOOTHERS[method]
    result = fn(dx, dy, da, smoothing_strength)

    # Normalise to CPU numpy regardless of which smoother ran
    corr_x, corr_y, corr_a = result
    if _CUPY_AVAILABLE:
        if isinstance(corr_x, cp.ndarray):
            corr_x = cp.asnumpy(corr_x)
        if isinstance(corr_y, cp.ndarray):
            corr_y = cp.asnumpy(corr_y)
        if isinstance(corr_a, cp.ndarray):
            corr_a = cp.asnumpy(corr_a)

    return _build_correction_transforms(
        corr_x.astype(np.float32),
        corr_y.astype(np.float32),
        corr_a.astype(np.float32),
    )
