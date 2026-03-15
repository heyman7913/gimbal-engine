import cv2
import numpy as np
import pytest

pytestmark = pytest.mark.cuda


def _blocky_image(h: int, w: int, seed: int) -> np.ndarray:
    """Blocky texture with many trackable corners at the cell boundaries."""
    rng = np.random.default_rng(seed)
    small = (rng.integers(0, 256, size=(h // 8, w // 8))).astype(np.uint8)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def _apply(h: np.ndarray, x: float, y: float) -> tuple[float, float]:
    p = h @ np.array([x, y, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def test_classical_recovers_translation():
    from gimbal.estimators.classical import ClassicalEstimator

    h, w = 240, 320
    a = _blocky_image(h, w, seed=7)
    m = np.array([[1.0, 0.0, 4.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]])
    b = cv2.warpPerspective(a, m, (w, h))

    est = ClassicalEstimator()
    recovered = est.estimate(a, b).as_global()

    ux, uy = _apply(recovered, w / 2, h / 2)
    assert abs(ux - (w / 2 + 4.0)) < 1.5
    assert abs(uy - (h / 2 + 2.0)) < 1.5


def test_classical_recovers_small_rotation():
    from gimbal.estimators.classical import ClassicalEstimator

    h, w = 240, 320
    a = _blocky_image(h, w, seed=11)
    angle = np.deg2rad(2.0)
    cx, cy = w / 2, h / 2
    rot = np.array(
        [
            [np.cos(angle), -np.sin(angle), cx - cx * np.cos(angle) + cy * np.sin(angle)],
            [np.sin(angle), np.cos(angle), cy - cx * np.sin(angle) - cy * np.cos(angle)],
            [0.0, 0.0, 1.0],
        ]
    )
    b = cv2.warpPerspective(a, rot, (w, h))

    est = ClassicalEstimator()
    recovered = est.estimate(a, b).as_global()

    # a corner point should map close to where the true rotation sends it
    px, py = 80.0, 60.0
    rx, ry = _apply(rot, px, py)
    ux, uy = _apply(recovered, px, py)
    assert np.hypot(ux - rx, uy - ry) < 2.0
