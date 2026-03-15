"""Local correlation (the cost volume) for the homography network.

The correlation at offset (dy, dx) is just the channel dot product of feature map A at (y, x)
with B at (y+dy, x+dx), divided by sqrt(C). I have two versions: a plain PyTorch one that I
trust as the reference, and the fused CUDA one (compiled extension under csrc/). The fused one
is only switched on after it passes gradcheck against the reference, so if I got the kernel
wrong training still works.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


def _ext() -> Any:
    import cuda_motion_flow._cuda as _c

    return _c


def local_correlation_reference(fa: torch.Tensor, fb: torch.Tensor, radius: int) -> torch.Tensor:
    """Pure-PyTorch local correlation. Output (B, (2r+1)^2, H, W)."""
    b, c, h, w = fa.shape
    fb_pad = F.pad(fb, (radius, radius, radius, radius))
    outs = []
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            shifted = fb_pad[:, :, dy : dy + h, dx : dx + w]
            outs.append((fa * shifted).sum(dim=1, keepdim=True))
    return torch.cat(outs, dim=1) / math.sqrt(c)


class FusedLocalCorrelation(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, fa: torch.Tensor, fb: torch.Tensor, radius: int) -> torch.Tensor:
        # the kernel only does fp32/fp64, so lift bf16/half to fp32 and cast the result back.
        # gradients still reach the original tensors (cast again in backward).
        orig_dtype = fa.dtype
        if orig_dtype not in (torch.float32, torch.float64):
            fa = fa.float()
            fb = fb.float()
        fa_c = fa.contiguous()
        fb_c = fb.contiguous()
        out = _ext().correlation_forward(fa_c, fb_c, radius)
        ctx.save_for_backward(fa_c, fb_c)
        ctx.radius = radius
        ctx.orig_dtype = orig_dtype
        result: torch.Tensor = out.to(orig_dtype)
        return result

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        fa, fb = ctx.saved_tensors  # contiguous fp32/fp64 from forward
        grad_c = grad.to(fa.dtype).contiguous()
        gfa, gfb = _ext().correlation_backward(grad_c, fa, fb, ctx.radius)
        return gfa.to(ctx.orig_dtype), gfb.to(ctx.orig_dtype), None


def local_correlation(
    fa: torch.Tensor, fb: torch.Tensor, radius: int, fused: bool = False
) -> torch.Tensor:
    if fused:
        out: torch.Tensor = FusedLocalCorrelation.apply(fa, fb, radius)  # type: ignore[no-untyped-call]
        return out
    return local_correlation_reference(fa, fb, radius)


def fused_passes_gradcheck(radius: int = 2) -> bool:
    """Gate: the fused kernel may be used in training only if it matches the reference here."""
    torch.manual_seed(0)
    fa = torch.randn(2, 4, 5, 6, device="cuda", dtype=torch.float64, requires_grad=True)
    fb = torch.randn(2, 4, 5, 6, device="cuda", dtype=torch.float64, requires_grad=True)

    def fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return FusedLocalCorrelation.apply(a, b, radius)  # type: ignore[no-any-return,no-untyped-call]

    try:
        return bool(torch.autograd.gradcheck(fn, (fa, fb), eps=1e-6, atol=1e-4))
    except Exception:
        return False
