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


# Scharr first derivatives, normalized by 1/32 so the values match cv2.Scharr * (1/32).
# Borders replicate the edge pixel. ix and iy are written in one pass.
SCHARR_GRADIENT_F32 = r"""
extern "C" __global__
void scharr_gradient_f32(
        const float* __restrict__ img, float* __restrict__ ix, float* __restrict__ iy,
        const int h, const int w) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    const int xm = max(x - 1, 0), xp = min(x + 1, w - 1);
    const int ym = max(y - 1, 0), yp = min(y + 1, h - 1);
    #define P(xx, yy) img[(yy) * w + (xx)]
    const float tl = P(xm, ym), tc = P(x, ym), tr = P(xp, ym);
    const float ml = P(xm, y),                 mr = P(xp, y);
    const float bl = P(xm, yp), bc = P(x, yp), br = P(xp, yp);
    #undef P
    ix[y * w + x] = (-3.f * tl + 3.f * tr - 10.f * ml + 10.f * mr - 3.f * bl + 3.f * br) / 32.f;
    iy[y * w + x] = (-3.f * tl - 10.f * tc - 3.f * tr + 3.f * bl + 10.f * bc + 3.f * br) / 32.f;
}
"""


# Shi-Tomasi corner response: min eigenvalue of the windowed structure tensor built from
# ix, iy. The window is a (2R+1) box; borders clamp. Matches the numpy reference of the same
# formula.
SHI_TOMASI_RESPONSE_F32 = r"""
extern "C" __global__
void shi_tomasi_response_f32(
        const float* __restrict__ ix, const float* __restrict__ iy,
        float* __restrict__ resp, const int h, const int w, const int radius) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    float sxx = 0.f, syy = 0.f, sxy = 0.f;
    for (int dy = -radius; dy <= radius; ++dy) {
        const int yy = min(max(y + dy, 0), h - 1);
        for (int dx = -radius; dx <= radius; ++dx) {
            const int xx = min(max(x + dx, 0), w - 1);
            const float gx = ix[yy * w + xx];
            const float gy = iy[yy * w + xx];
            sxx += gx * gx;
            syy += gy * gy;
            sxy += gx * gy;
        }
    }
    const float t = 0.5f * (sxx + syy);
    const float d = 0.5f * (sxx - syy);
    const float lambda_min = t - sqrtf(d * d + sxy * sxy);
    resp[y * w + x] = lambda_min;
}
"""


# 2x Gaussian-pyramid downsample with the separable 5-tap [1,4,6,4,1]/16 kernel, matching
# cv2.pyrDown. Output pixel (ox, oy) is centered on input (2*ox, 2*oy); borders replicate.
GAUSSIAN_DOWNSAMPLE_F32 = r"""
extern "C" __global__
void gaussian_downsample_f32(
        const float* __restrict__ src, float* __restrict__ dst,
        const int sh, const int sw, const int dh, const int dw) {
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= dw || oy >= dh) return;

    const float k[5] = {1.f, 4.f, 6.f, 4.f, 1.f};
    const int cx = 2 * ox, cy = 2 * oy;
    float acc = 0.f;
    for (int j = -2; j <= 2; ++j) {
        const int yy = min(max(cy + j, 0), sh - 1);
        for (int i = -2; i <= 2; ++i) {
            const int xx = min(max(cx + i, 0), sw - 1);
            acc += k[j + 2] * k[i + 2] * src[yy * sw + xx];
        }
    }
    dst[oy * dw + ox] = acc / 256.f;
}
"""
