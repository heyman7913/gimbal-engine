"""Local correlation (cost volume) for the iterative homography network.

Two implementations:
  - `local_correlation_reference`: pure PyTorch (shift-multiply-sum). This is the correctness
    oracle and the default used in training. It runs on the GPU.
  - `FusedLocalCorrelation`: a custom autograd.Function backed by hand-written CuPy RawKernels
    (forward and both backward passes). It computes the same volume in a single pass with no
    per-shift temporaries. Both backward passes are written as gathers, so no atomics are
    needed. It is gated: it must match the reference under torch.autograd.gradcheck before any
    training run uses it; otherwise training falls back to the reference.

For a feature map of C channels the correlation at offset (dy, dx) is the channel dot product
of A at (y, x) with B at (y+dy, x+dx), divided by sqrt(C). After the iterative warp the
residual motion is small, so a local window of radius r (giving (2r+1)^2 channels) suffices,
which is exactly why the fused kernel beats materializing a full volume.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    import cupy as cp

_KERNEL_CACHE: dict[tuple[str, str], Any] = {}

_SOURCE = r"""
extern "C" __global__
void corr_fwd_{suf}(const {S}* fa, const {S}* fb, {S}* out,
                    const int B, const int C, const int H, const int W,
                    const int R, const {S} inv_sqrt_c) {{
    const int D = 2 * R + 1;
    const int K = D * D;
    const long total = (long)B * K * H * W;
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int x = idx % W; long t = idx / W;
    const int y = t % H; t /= H;
    const int k = t % K; t /= K;
    const int b = t;
    const int dy = k / D - R;
    const int dx = k % D - R;
    const int yy = y + dy, xx = x + dx;
    {S} acc = 0;
    if (yy >= 0 && yy < H && xx >= 0 && xx < W) {{
        for (int c = 0; c < C; ++c) {{
            const long base = ((long)b * C + c) * H;
            acc += fa[(base + y) * W + x] * fb[(base + yy) * W + xx];
        }}
    }}
    out[idx] = acc * inv_sqrt_c;
}}

extern "C" __global__
void corr_bwd_fa_{suf}(const {S}* grad, const {S}* fb, {S}* gfa,
                       const int B, const int C, const int H, const int W,
                       const int R, const {S} inv_sqrt_c) {{
    const int D = 2 * R + 1;
    const int K = D * D;
    const long total = (long)B * C * H * W;
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int x = idx % W; long t = idx / W;
    const int y = t % H; t /= H;
    const int c = t % C; t /= C;
    const int b = t;
    {S} acc = 0;
    for (int k = 0; k < K; ++k) {{
        const int dy = k / D - R;
        const int dx = k % D - R;
        const int yy = y + dy, xx = x + dx;
        if (yy >= 0 && yy < H && xx >= 0 && xx < W) {{
            const long g = ((long)b * K + k) * H + y;
            const long f = ((long)b * C + c) * H + yy;
            acc += grad[g * W + x] * fb[f * W + xx];
        }}
    }}
    gfa[idx] = acc * inv_sqrt_c;
}}

extern "C" __global__
void corr_bwd_fb_{suf}(const {S}* grad, const {S}* fa, {S}* gfb,
                       const int B, const int C, const int H, const int W,
                       const int R, const {S} inv_sqrt_c) {{
    const int D = 2 * R + 1;
    const int K = D * D;
    const long total = (long)B * C * H * W;
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int xx = idx % W; long t = idx / W;
    const int yy = t % H; t /= H;
    const int c = t % C; t /= C;
    const int b = t;
    {S} acc = 0;
    for (int k = 0; k < K; ++k) {{
        const int dy = k / D - R;
        const int dx = k % D - R;
        const int y = yy - dy, x = xx - dx;  // source position that used (yy, xx)
        if (y >= 0 && y < H && x >= 0 && x < W) {{
            const long g = ((long)b * K + k) * H + y;
            const long f = ((long)b * C + c) * H + y;
            acc += grad[g * W + x] * fa[f * W + x];
        }}
    }}
    gfb[idx] = acc * inv_sqrt_c;
}}
"""


def _kernels(dtype: torch.dtype) -> tuple[Any, Any, Any]:
    import cupy as cp

    suf, s = ("f64", "double") if dtype == torch.float64 else ("f32", "float")
    if (suf, "fwd") not in _KERNEL_CACHE:
        src = _SOURCE.format(suf=suf, S=s)
        module = cp.RawModule(code=src, options=("--std=c++14",))
        _KERNEL_CACHE[(suf, "fwd")] = module.get_function(f"corr_fwd_{suf}")
        _KERNEL_CACHE[(suf, "fa")] = module.get_function(f"corr_bwd_fa_{suf}")
        _KERNEL_CACHE[(suf, "fb")] = module.get_function(f"corr_bwd_fb_{suf}")
    return (
        _KERNEL_CACHE[(suf, "fwd")],
        _KERNEL_CACHE[(suf, "fa")],
        _KERNEL_CACHE[(suf, "fb")],
    )


def _cp_view(t: torch.Tensor) -> cp.ndarray:
    """Zero-copy CuPy view of a contiguous CUDA tensor. Caller keeps `t` alive."""
    import cupy as cp

    return cp.from_dlpack(t.detach())


_STREAM_CACHE: dict[int, Any] = {}


def _torch_stream() -> Any:
    """A CuPy stream wrapping torch's current stream, cached per stream pointer."""
    import cupy as cp

    ptr = torch.cuda.current_stream().cuda_stream
    if ptr not in _STREAM_CACHE:
        _STREAM_CACHE[ptr] = cp.cuda.ExternalStream(ptr)
    return _STREAM_CACHE[ptr]


def _launch(kernel: Any, n: int, args: tuple[Any, ...]) -> None:
    """Launch on torch's current stream so ordering and the caching allocator stay correct."""
    threads = 256
    blocks = (n + threads - 1) // threads
    with _torch_stream():
        kernel((blocks,), (threads,), args)


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
        import numpy as np

        # CuPy has no bfloat16/half; run the kernel in fp32 and cast back. Gradients still
        # flow to the original-dtype inputs (cast in backward).
        orig_dtype = fa.dtype
        if orig_dtype not in (torch.float32, torch.float64):
            fa = fa.float()
            fb = fb.float()
        b, c, h, w = fa.shape
        k = (2 * radius + 1) ** 2
        fa_c = fa.contiguous()
        fb_c = fb.contiguous()
        out = torch.empty((b, k, h, w), device=fa.device, dtype=fa.dtype)
        fwd, _, _ = _kernels(fa.dtype)
        npdt = np.float64 if fa.dtype == torch.float64 else np.float32
        inv = npdt(1.0 / math.sqrt(c))
        ints = (np.int32(b), np.int32(c), np.int32(h), np.int32(w), np.int32(radius))
        _launch(fwd, b * k * h * w, (_cp_view(fa_c), _cp_view(fb_c), _cp_view(out), *ints, inv))
        ctx.save_for_backward(fa_c, fb_c)
        ctx.radius = radius
        ctx.orig_dtype = orig_dtype
        return out.to(orig_dtype)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        import numpy as np

        fa, fb = ctx.saved_tensors  # contiguous fp32/fp64 from forward
        radius = ctx.radius
        b, c, h, w = fa.shape
        gfa = torch.empty_like(fa)
        gfb = torch.empty_like(fb)
        _, bwd_fa, bwd_fb = _kernels(fa.dtype)
        npdt = np.float64 if fa.dtype == torch.float64 else np.float32
        inv = npdt(1.0 / math.sqrt(c))
        ints = (np.int32(b), np.int32(c), np.int32(h), np.int32(w), np.int32(radius))
        grad_c = grad.to(fa.dtype).contiguous()
        _launch(bwd_fa, b * c * h * w, (_cp_view(grad_c), _cp_view(fb), _cp_view(gfa), *ints, inv))
        _launch(bwd_fb, b * c * h * w, (_cp_view(grad_c), _cp_view(fa), _cp_view(gfb), *ints, inv))
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
