import numpy as np
from gimbal import metrics


def test_distortion_identity_is_one():
    transforms = np.stack([np.eye(3)] * 4)
    assert metrics.distortion_value(transforms) == 1.0


def test_distortion_pure_rotation_is_one():
    a = np.deg2rad(20.0)
    r = np.eye(3)
    r[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    assert abs(metrics.distortion_value(r[None]) - 1.0) < 1e-9


def test_distortion_anisotropic_scale():
    h = np.eye(3)
    h[0, 0] = 2.0  # stretch x only -> singular values 2 and 1
    assert abs(metrics.distortion_value(h[None]) - 0.5) < 1e-9


def test_cropping_ratio_full_frame_is_one():
    assert metrics.cropping_ratio(100, 100, 100, 100) == 1.0


def test_cropping_ratio_half_area():
    assert abs(metrics.cropping_ratio(50, 100, 100, 100) - np.sqrt(0.5)) < 1e-9


def test_stability_higher_for_smooth_path():
    n = 128
    t = np.arange(n)
    smooth = np.stack([np.eye(3)] * n)
    smooth[:, 0, 2] = 0.2 * t
    shaky = smooth.copy()
    shaky[:, 0, 2] = 0.2 * t + 5.0 * np.sin(2 * np.pi * t / 3.0)
    assert metrics.stability_score(smooth) > metrics.stability_score(shaky)
