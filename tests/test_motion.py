import numpy as np
from cuda_motion_flow import trajectory
from cuda_motion_flow.motion import MotionField


def test_global_field_roundtrip():
    h = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]])
    f = MotionField.global_(h)
    assert f.is_global
    assert f.grid_shape == (1, 1)
    np.testing.assert_array_equal(f.as_global(), h)


def test_global_field_path_is_bit_exact_with_raw_homographies():
    rng = np.random.default_rng(0)
    n = 8
    pairwise = np.stack([np.eye(3) for _ in range(n)])
    for k in range(n):
        pairwise[k, :2, 2] = rng.normal(0, 2.0, 2)
        pairwise[k, 0, 0] = 1.0 + rng.normal(0, 0.01)

    # the original raw-homography back-end
    cum = trajectory.cumulative_path(pairwise)
    sm = trajectory.smooth_path(cum, "gaussian", 0.5)
    raw = trajectory.stabilizing_transforms(cum, sm)

    # the same homographies routed through 1x1 MotionFields
    fields = [MotionField.global_(h) for h in pairwise]
    pairwise2 = np.stack([f.as_global() for f in fields])
    cum2 = trajectory.cumulative_path(pairwise2)
    sm2 = trajectory.smooth_path(cum2, "gaussian", 0.5)
    via_field = trajectory.stabilizing_transforms(cum2, sm2)

    np.testing.assert_array_equal(raw, via_field)  # bit for bit
