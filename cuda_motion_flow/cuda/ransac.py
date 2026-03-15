"""RANSAC homography fit, vectorised over all the hypotheses at once.

Instead of a python loop over RANSAC iterations, I build and score every hypothesis on the GPU
in one go. The minimal 4-point fits use regularised normal equations so a bad sample doesn't
crash the solve, and the final fit on the inliers uses Hartley-normalised DLT for stability. H
goes from frame A to frame B.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import cupy as cp


def _dlt_rows(src: cp.ndarray, dst: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
    """Build the DLT system rows for correspondences. src,dst: (..., K, 2). h33 fixed to 1.

    Returns A (..., 2K, 8) and b (..., 2K).
    """
    import cupy as cp

    x = src[..., 0]
    y = src[..., 1]
    u = dst[..., 0]
    v = dst[..., 1]
    zeros = cp.zeros_like(x)
    ones = cp.ones_like(x)
    row_u = cp.stack([x, y, ones, zeros, zeros, zeros, -u * x, -u * y], axis=-1)
    row_v = cp.stack([zeros, zeros, zeros, x, y, ones, -v * x, -v * y], axis=-1)
    a = cp.concatenate([row_u, row_v], axis=-2)
    b = cp.concatenate([u, v], axis=-1)
    return a, b


def _solve_reg(a: cp.ndarray, b: cp.ndarray, ridge: float = 1e-6) -> cp.ndarray:
    """Solve least squares via regularized normal equations, batched over leading dims."""
    import cupy as cp

    at = cp.swapaxes(a, -1, -2)
    ata = at @ a
    atb = at @ b[..., None]  # (..., 8, 1); batched solve needs a 2D rhs per matrix
    eye = cp.eye(ata.shape[-1], dtype=ata.dtype)
    h = cp.linalg.solve(ata + ridge * eye, atb)[..., 0]
    return h


def _to_h33(h8: cp.ndarray) -> cp.ndarray:
    import cupy as cp

    ones = cp.ones((*h8.shape[:-1], 1), dtype=h8.dtype)
    return cp.concatenate([h8, ones], axis=-1).reshape(*h8.shape[:-1], 3, 3)


def _normalize_points(pts: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
    """Hartley normalization: translate to centroid, scale to mean distance sqrt(2)."""
    import cupy as cp

    c = pts.mean(axis=0)
    d = pts - c
    mean_dist = cp.sqrt((d**2).sum(axis=1)).mean()
    s = cp.sqrt(cp.asarray(2.0, dtype=pts.dtype)) / (mean_dist + 1e-12)
    # build T with device ops; a python list holding cupy scalars is rejected by cupy
    t = cp.eye(3, dtype=pts.dtype)
    t[0, 0] = s
    t[1, 1] = s
    t[0, 2] = -s * c[0]
    t[1, 2] = -s * c[1]
    normed = (pts - c) * s
    return normed, t


def fit_homography(src: cp.ndarray, dst: cp.ndarray) -> cp.ndarray:
    """Normalized-DLT homography over all given correspondences (>= 4)."""
    import cupy as cp

    src_n, t_src = _normalize_points(src)
    dst_n, t_dst = _normalize_points(dst)
    a, b = _dlt_rows(src_n, dst_n)
    h = _to_h33(_solve_reg(a, b))
    # denormalize: H = T_dst^{-1} @ H_n @ T_src
    h = cp.linalg.inv(t_dst) @ h @ t_src
    return h / h[2, 2]


def ransac_homography(
    src: cp.ndarray,
    dst: cp.ndarray,
    n_iter: int = 500,
    threshold: float = 3.0,
    seed: int = 0,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Return (H 3x3, inlier mask). Falls back to identity when correspondences are too few."""
    import cupy as cp

    n = src.shape[0]
    if n < 4:
        return cp.eye(3, dtype=cp.float64), cp.zeros((n,), dtype=bool)

    rng = cp.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_iter, 4))
    sample_src = src[idx]  # (n_iter, 4, 2)
    sample_dst = dst[idx]
    a, b = _dlt_rows(sample_src.astype(cp.float64), sample_dst.astype(cp.float64))
    hyp = _to_h33(_solve_reg(a, b))  # (n_iter, 3, 3)

    src_h = cp.concatenate([src, cp.ones((n, 1), dtype=src.dtype)], axis=1).astype(cp.float64)
    proj = cp.einsum("kij,nj->kni", hyp, src_h)  # (n_iter, n, 3)
    w = proj[..., 2:3]
    w = cp.where(cp.abs(w) < 1e-12, 1e-12, w)
    proj_xy = proj[..., :2] / w
    err = cp.sqrt(((proj_xy - dst[None].astype(cp.float64)) ** 2).sum(axis=2))  # (n_iter, n)
    inliers = err < threshold
    counts = inliers.sum(axis=1)
    best = int(cp.argmax(counts))

    best_mask = inliers[best]
    if int(best_mask.sum()) < 4:
        return cp.eye(3, dtype=cp.float64), best_mask
    h = fit_homography(src[best_mask].astype(cp.float64), dst[best_mask].astype(cp.float64))
    return h, best_mask
