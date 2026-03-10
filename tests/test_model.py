import numpy as np
import pytest

pytestmark = pytest.mark.cuda


def test_dlt_recovers_identity_from_zero_offset():
    import torch

    from cuda_motion_flow.learned.model import dlt_solve

    src = torch.tensor([[[0.0, 0.0], [127.0, 0.0], [127.0, 127.0], [0.0, 127.0]]], device="cuda")
    h = dlt_solve(src, src)
    assert torch.allclose(h, torch.eye(3, device="cuda").unsqueeze(0), atol=1e-4)


def test_dlt_recovers_known_homography():
    import torch

    from cuda_motion_flow.learned.model import dlt_solve

    rng = np.random.default_rng(0)
    h_true = np.eye(3)
    h_true[:2, :] += rng.normal(0, 0.05, size=(2, 3))
    h_true[2, :2] += rng.normal(0, 1e-4, size=2)
    h_true /= h_true[2, 2]

    corners = np.array([[10.0, 12.0], [110.0, 8.0], [120.0, 118.0], [6.0, 116.0]])
    homog = np.concatenate([corners, np.ones((4, 1))], axis=1) @ h_true.T
    dst = homog[:, :2] / homog[:, 2:3]

    src_t = torch.tensor(corners, device="cuda").unsqueeze(0)
    dst_t = torch.tensor(dst, device="cuda").unsqueeze(0)
    h = dlt_solve(src_t, dst_t).squeeze(0).cpu().numpy()
    np.testing.assert_allclose(h, h_true, atol=1e-3)


def test_dlt_gradcheck():
    import torch

    from cuda_motion_flow.learned.model import dlt_solve

    src = torch.tensor(
        [[[0.0, 0.0], [120.0, 0.0], [120.0, 120.0], [0.0, 120.0]]],
        device="cuda",
        dtype=torch.float64,
    )
    dst = src + torch.randn(1, 4, 2, device="cuda", dtype=torch.float64, generator=None) * 2.0
    dst = dst.detach().requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda d: dlt_solve(src, d, ridge=0.0), (dst,), eps=1e-6, atol=1e-4
    )


def test_ihn_forward_shapes():
    import torch

    from cuda_motion_flow.learned.model import IHN, RegressionHomographyNet

    a = torch.rand(2, 1, 128, 128, device="cuda")
    b = torch.rand(2, 1, 128, 128, device="cuda")

    ihn = IHN(iters=4).cuda()
    offsets = ihn(a, b)
    assert len(offsets) == 4
    assert offsets[-1].shape == (2, 4, 2)
    assert ihn.predict(a, b).shape == (2, 3, 3)

    reg = RegressionHomographyNet().cuda()
    assert len(reg(a, b)) == 1
    assert reg.predict(a, b).shape == (2, 3, 3)
