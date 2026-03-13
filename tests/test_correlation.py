import pytest

pytestmark = pytest.mark.cuda


def test_fused_forward_matches_reference():
    import torch

    from cuda_motion_flow.learned.correlation import (
        FusedLocalCorrelation,
        local_correlation_reference,
    )

    torch.manual_seed(0)
    fa = torch.randn(2, 8, 12, 16, device="cuda")
    fb = torch.randn(2, 8, 12, 16, device="cuda")
    ref = local_correlation_reference(fa, fb, 3)
    out = FusedLocalCorrelation.apply(fa, fb, 3)
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)


def test_fused_backward_matches_reference():
    import torch

    from cuda_motion_flow.learned.correlation import (
        FusedLocalCorrelation,
        local_correlation_reference,
    )

    torch.manual_seed(1)
    fa0 = torch.randn(2, 8, 10, 12, device="cuda")
    fb0 = torch.randn(2, 8, 10, 12, device="cuda")
    grad = torch.randn(2, 49, 10, 12, device="cuda")

    fa1 = fa0.clone().requires_grad_(True)
    fb1 = fb0.clone().requires_grad_(True)
    local_correlation_reference(fa1, fb1, 3).backward(grad)

    fa2 = fa0.clone().requires_grad_(True)
    fb2 = fb0.clone().requires_grad_(True)
    FusedLocalCorrelation.apply(fa2, fb2, 3).backward(grad)

    assert torch.allclose(fa1.grad, fa2.grad, atol=1e-4, rtol=1e-4)
    assert torch.allclose(fb1.grad, fb2.grad, atol=1e-4, rtol=1e-4)


def test_correlation_variants_match_v0():
    import cuda_motion_flow._cuda as ext
    import torch

    torch.manual_seed(3)
    fa = torch.randn(4, 96, 12, 14, device="cuda")
    fb = torch.randn(4, 96, 12, 14, device="cuda")
    v0 = ext.correlation_forward(fa, fb, 4)
    assert torch.equal(ext.correlation_forward_v1(fa, fb, 4), v0)  # identical math, fewer loads
    fan = fa.permute(0, 2, 3, 1).contiguous()
    fbn = fb.permute(0, 2, 3, 1).contiguous()
    assert torch.allclose(ext.correlation_forward_v2(fan, fbn, 4), v0, atol=1e-4)


def test_fused_runs_under_bfloat16():
    import torch

    from cuda_motion_flow.learned.correlation import FusedLocalCorrelation

    fa = torch.randn(2, 8, 10, 12, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    fb = torch.randn(2, 8, 10, 12, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = FusedLocalCorrelation.apply(fa, fb, 3)
    assert out.dtype == torch.bfloat16
    out.sum().backward()
    assert fa.grad is not None and torch.isfinite(fa.grad.float()).all()


def test_fused_gradcheck_double():
    import torch

    from cuda_motion_flow.learned.correlation import FusedLocalCorrelation

    torch.manual_seed(2)
    fa = torch.randn(2, 4, 5, 6, device="cuda", dtype=torch.float64, requires_grad=True)
    fb = torch.randn(2, 4, 5, 6, device="cuda", dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda a, b: FusedLocalCorrelation.apply(a, b, 2), (fa, fb), eps=1e-6, atol=1e-4
    )
