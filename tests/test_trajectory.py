import numpy as np

from cuda_motion_flow import trajectory


def _translation_h(tx: float, ty: float) -> np.ndarray:
    h = np.eye(3)
    h[0, 2] = tx
    h[1, 2] = ty
    return h


def test_cumulative_path_of_identities_is_identity():
    pairwise = np.stack([np.eye(3)] * 5)
    cum = trajectory.cumulative_path(pairwise)
    assert cum.shape == (6, 3, 3)
    for c in cum:
        np.testing.assert_allclose(c, np.eye(3), atol=1e-9)


def test_cumulative_path_accumulates_translation():
    pairwise = np.stack([_translation_h(2.0, -1.0)] * 3)
    cum = trajectory.cumulative_path(pairwise)
    # frame 3 sits at 3 * (2, -1) from the origin
    np.testing.assert_allclose(cum[3, :2, 2], [6.0, -3.0], atol=1e-9)


def test_stabilizing_transform_is_identity_when_path_unchanged():
    pairwise = np.stack([_translation_h(1.5, 0.5)] * 4)
    cum = trajectory.cumulative_path(pairwise)
    b = trajectory.stabilizing_transforms(cum, cum)
    for t in b:
        np.testing.assert_allclose(t, np.eye(3), atol=1e-9)


def _shaky_cumulative(n: int = 120) -> np.ndarray:
    t = np.arange(n)
    drift = 0.4 * t  # intentional slow pan
    jitter = 3.0 * np.sin(2.0 * np.pi * t / 4.0)  # high-frequency shake
    cum = np.stack([np.eye(3)] * n)
    cum[:, 0, 2] = drift + jitter
    cum[:, 1, 2] = 0.5 * t + 2.0 * np.cos(2.0 * np.pi * t / 3.0)
    return cum


def _hf_energy(signal: np.ndarray) -> float:
    spec = np.abs(np.fft.rfft(signal - signal.mean()))
    return float((spec[len(spec) // 2 :] ** 2).sum())


def test_each_smoother_reduces_high_frequency_energy():
    cum = _shaky_cumulative()
    before = _hf_energy(cum[:, 0, 2])
    for method in ("gaussian", "kalman_rts", "l1_tv"):
        smoothed = trajectory.smooth_path(cum, method, strength=0.9)
        after = _hf_energy(smoothed[:, 0, 2])
        assert after < 0.5 * before, f"{method} did not cut HF energy ({after} vs {before})"
