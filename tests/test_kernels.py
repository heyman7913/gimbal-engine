import cv2
import numpy as np
import pytest

pytestmark = pytest.mark.cuda


def _interior(a: np.ndarray, m: int = 3) -> np.ndarray:
    return a[m:-m, m:-m]


def test_scharr_matches_opencv():
    import cupy as cp
    from cuda_motion_flow.cuda.optical_flow import scharr_gradients

    img = (np.random.default_rng(0).random((96, 128)) * 255).astype(np.float32)
    ix_ref = cv2.Scharr(img, cv2.CV_32F, 1, 0) / 32.0
    iy_ref = cv2.Scharr(img, cv2.CV_32F, 0, 1) / 32.0
    ix, iy = scharr_gradients(cp.asarray(img))
    np.testing.assert_allclose(_interior(cp.asnumpy(ix)), _interior(ix_ref), atol=1e-3)
    np.testing.assert_allclose(_interior(cp.asnumpy(iy)), _interior(iy_ref), atol=1e-3)


def test_downsample_matches_pyrdown():
    import cupy as cp
    from cuda_motion_flow.cuda.optical_flow import downsample

    img = (np.random.default_rng(1).random((96, 128)) * 255).astype(np.float32)
    ref = cv2.pyrDown(img)
    out = cp.asnumpy(downsample(cp.asarray(img)))
    assert out.shape == ref.shape
    np.testing.assert_allclose(_interior(out), _interior(ref), atol=1e-2)


def test_shi_tomasi_matches_numpy_reference():
    import cupy as cp
    from cuda_motion_flow.cuda.optical_flow import scharr_gradients, shi_tomasi_response

    img = (np.random.default_rng(2).random((80, 100)) * 255).astype(np.float32)
    ix, iy = scharr_gradients(cp.asarray(img))
    ix_h, iy_h = cp.asnumpy(ix), cp.asnumpy(iy)

    radius = 3
    k = 2 * radius + 1
    sxx = cv2.boxFilter(ix_h * ix_h, -1, (k, k), normalize=False)
    syy = cv2.boxFilter(iy_h * iy_h, -1, (k, k), normalize=False)
    sxy = cv2.boxFilter(ix_h * iy_h, -1, (k, k), normalize=False)
    t = 0.5 * (sxx + syy)
    d = 0.5 * (sxx - syy)
    ref = t - np.sqrt(d * d + sxy * sxy)

    out = cp.asnumpy(shi_tomasi_response(ix, iy, radius))
    # float32 accumulation order differs from the separable box filter; compare loosely
    np.testing.assert_allclose(
        _interior(out, radius + 1), _interior(ref, radius + 1), rtol=2e-2, atol=1.0
    )
