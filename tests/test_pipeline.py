import cv2
import numpy as np
import pytest

pytestmark = pytest.mark.cuda


def _shaky_clip(n: int = 40, h: int = 240, w: int = 320) -> list[np.ndarray]:
    """A static blocky scene viewed through a panning, jittering virtual camera."""
    rng = np.random.default_rng(3)
    base = cv2.resize(
        rng.integers(0, 256, size=(h // 8, w // 8)).astype(np.uint8),
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    )
    frames = []
    for i in range(n):
        dx = 0.3 * i + 4.0 * np.sin(2 * np.pi * i / 4.0)  # slow pan + fast shake
        dy = 0.2 * i + 3.0 * np.cos(2 * np.pi * i / 3.0)
        m = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])
        warped = cv2.warpPerspective(base, m, (w, h))
        frames.append(cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR))
    return frames


def test_pipeline_stabilizes_with_classical():
    from gimbal.estimators.classical import ClassicalEstimator
    from gimbal.pipeline import stabilize

    frames = _shaky_clip()
    result = stabilize(frames, ClassicalEstimator(), smoother="gaussian", strength=0.85)

    assert len(result.frames) == len(frames)
    # the stabilized path must carry more of its energy at low frequencies than the input
    assert result.metrics["stability_score"] > result.metrics["stability_input"]
