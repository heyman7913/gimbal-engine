import pytest

pytestmark = pytest.mark.cuda


def test_extension_imports_and_runs():
    import gimbal._cuda as ext
    import torch

    fa = torch.randn(2, 8, 12, 16, device="cuda")
    fb = torch.randn(2, 8, 12, 16, device="cuda")
    out = ext.correlation_forward(fa, fb, 3)
    assert out.shape == (2, 49, 12, 16)
    gfa, gfb = ext.correlation_backward(torch.randn_like(out), fa, fb, 3)
    assert gfa.shape == fa.shape and gfb.shape == fb.shape

    img = torch.rand(32, 40, device="cuda")
    ix, iy = ext.scharr_gradient(img)
    assert ix.shape == img.shape
    assert ext.shi_tomasi_response(ix, iy, 3).shape == img.shape
    assert ext.gaussian_downsample(img).shape == (16, 20)

    src = (torch.rand(24, 32, 3, device="cuda") * 255).to(torch.uint8)
    warped = ext.warp_bilinear(src, torch.eye(3, device="cuda", dtype=torch.float64), 24, 32)
    assert torch.equal(warped, src)  # identity warp is lossless
