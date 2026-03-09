"""Raw CUDA kernels compiled at runtime via CuPy, with a process-wide compile cache.

Kernel sources are kept as strings here; the python wrappers in the sibling modules and in
warp.py launch them. Everything is float32/uint8 on the device; there is no CPU path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import cupy as cp

_CACHE: dict[str, cp.RawKernel] = {}


def get_kernel(name: str, source: str) -> cp.RawKernel:
    """Compile (once) and return the named kernel from the given source."""
    import cupy as cp

    if name not in _CACHE:
        _CACHE[name] = cp.RawKernel(source, name, options=("--std=c++14",))
    return _CACHE[name]


# Output warp: for each destination pixel, map back through the inverse homography and
# bilinearly sample the source. Out-of-bounds samples are black so the auto-crop can remove
# the borders afterwards. h_inv maps destination coords to source coords (dst -> src).
WARP_BILINEAR_U8 = r"""
extern "C" __global__
void warp_bilinear_u8(
        const unsigned char* __restrict__ src, unsigned char* __restrict__ dst,
        const double* __restrict__ h_inv,
        const int sh, const int sw, const int dh, const int dw, const int ch) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;

    const double X = h_inv[0] * x + h_inv[1] * y + h_inv[2];
    const double Y = h_inv[3] * x + h_inv[4] * y + h_inv[5];
    const double W = h_inv[6] * x + h_inv[7] * y + h_inv[8];

    unsigned char* out = dst + (y * dw + x) * ch;
    if (W == 0.0) {
        for (int c = 0; c < ch; ++c) out[c] = 0;
        return;
    }
    const double sx = X / W;
    const double sy = Y / W;
    const int x0 = (int)floor(sx);
    const int y0 = (int)floor(sy);
    const double ax = sx - x0;
    const double ay = sy - y0;

    for (int c = 0; c < ch; ++c) {
        double acc = 0.0;
        // 4 neighbours; contributions from out-of-bounds taps are zero
        #pragma unroll
        for (int dy = 0; dy < 2; ++dy) {
            const int yy = y0 + dy;
            if (yy < 0 || yy >= sh) continue;
            const double wy = dy ? ay : (1.0 - ay);
            #pragma unroll
            for (int dx = 0; dx < 2; ++dx) {
                const int xx = x0 + dx;
                if (xx < 0 || xx >= sw) continue;
                const double wx = dx ? ax : (1.0 - ax);
                acc += wx * wy * (double)src[(yy * sw + xx) * ch + c];
            }
        }
        const int v = (int)(acc + 0.5);
        out[c] = (unsigned char)(v < 0 ? 0 : (v > 255 ? 255 : v));
    }
}
"""
