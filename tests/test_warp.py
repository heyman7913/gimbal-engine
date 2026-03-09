import numpy as np
import pytest

pytestmark = pytest.mark.cuda


def test_identity_warp_is_lossless():
    import cupy as cp

    from cuda_motion_flow.warp import warp_frame

    img = (np.random.default_rng(0).random((64, 80, 3)) * 255).astype(np.uint8)
    out = cp.asnumpy(warp_frame(cp.asarray(img), np.eye(3)))
    assert np.array_equal(out, img)


def test_integer_translation_shifts_pixels():
    import cupy as cp

    from cuda_motion_flow.warp import warp_frame

    img = (np.random.default_rng(1).random((64, 80, 3)) * 255).astype(np.uint8)
    t = np.eye(3)
    t[0, 2], t[1, 2] = 5.0, 3.0  # maps src -> dst by (+5, +3)
    out = cp.asnumpy(warp_frame(cp.asarray(img), t))
    # dst(x, y) samples src(x-5, y-3), so the shifted interior matches
    assert np.array_equal(out[3:, 5:], img[:-3, :-5])


def test_crop_box_identity_is_full_frame():
    from cuda_motion_flow.warp import compute_crop_box

    box = compute_crop_box(np.stack([np.eye(3)] * 4), 80, 64)
    assert box == (0, 0, 80, 64)
