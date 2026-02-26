"""
CUDA C++ kernels compiled at runtime via CuPy RawKernel.

Provides high-throughput replacements for Python-dispatch paths:
  - affine_warp_bilinear_u8: bilinear affine warp for uint8 BGR frames
  - gaussian_downsample_f32: 2x downsample with 5-tap Gaussian anti-aliasing
  - scharr_gradient_f32: simultaneous Scharr Gx/Gy computation
  - shi_tomasi_response_f32: min-eigenvalue corner response from gradient images
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Kernel source strings
# ---------------------------------------------------------------------------

_AFFINE_WARP_SRC = r"""
extern "C" __global__
void affine_warp_bilinear_u8(
    const unsigned char* __restrict__ src,
    unsigned char*       __restrict__ dst,
    int src_w, int src_h,
    int dst_w, int dst_h,
    float m00, float m01, float m02,
    float m10, float m11, float m12)
{
    const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (dst_x >= dst_w || dst_y >= dst_h) return;

    // Inverse affine: map destination pixel to source coordinates
    float sx = m00 * (float)dst_x + m01 * (float)dst_y + m02;
    float sy = m10 * (float)dst_x + m11 * (float)dst_y + m12;

    int x0 = __float2int_rd(sx);
    int y0 = __float2int_rd(sy);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float fx = sx - (float)x0;
    float fy = sy - (float)y0;

    // Clamp to border
    x0 = max(0, min(x0, src_w - 1));
    x1 = max(0, min(x1, src_w - 1));
    y0 = max(0, min(y0, src_h - 1));
    y1 = max(0, min(y1, src_h - 1));

    float w00 = (1.0f - fx) * (1.0f - fy);
    float w01 = fx * (1.0f - fy);
    float w10 = (1.0f - fx) * fy;
    float w11 = fx * fy;

    // Source is HWC layout, 3 channels
    int src_stride = src_w * 3;
    int dst_stride = dst_w * 3;

    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        float v = w00 * (float)__ldg(&src[y0 * src_stride + x0 * 3 + c])
                + w01 * (float)__ldg(&src[y0 * src_stride + x1 * 3 + c])
                + w10 * (float)__ldg(&src[y1 * src_stride + x0 * 3 + c])
                + w11 * (float)__ldg(&src[y1 * src_stride + x1 * 3 + c]);
        dst[dst_y * dst_stride + dst_x * 3 + c] = (unsigned char)fminf(fmaxf(v + 0.5f, 0.0f), 255.0f);
    }
}
"""

_GAUSSIAN_DOWNSAMPLE_SRC = r"""
#define BLOCK_W 16
#define BLOCK_H 16
#define HALO    2
#define TILE_W  (BLOCK_W * 2 + 2 * HALO)
#define TILE_H  (BLOCK_H * 2 + 2 * HALO)

extern "C" __global__
void gaussian_downsample_f32(
    const float* __restrict__ src,
    float*       __restrict__ dst,
    int src_w, int src_h,
    int dst_w, int dst_h)
{
    __shared__ float tile[TILE_H][TILE_W];

    const int out_x = blockIdx.x * BLOCK_W + threadIdx.x;
    const int out_y = blockIdx.y * BLOCK_H + threadIdx.y;

    // Top-left corner of the source tile this block covers
    const int tile_x0 = blockIdx.x * BLOCK_W * 2 - HALO;
    const int tile_y0 = blockIdx.y * BLOCK_H * 2 - HALO;

    // Cooperatively load shared memory tile
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int n_threads = BLOCK_W * BLOCK_H;  // 256
    const int tile_elems = TILE_W * TILE_H;

    for (int i = tid; i < tile_elems; i += n_threads) {
        int ty = i / TILE_W;
        int tx = i % TILE_W;
        int gx = tile_x0 + tx;
        int gy = tile_y0 + ty;
        // Clamp to border
        gx = max(0, min(gx, src_w - 1));
        gy = max(0, min(gy, src_h - 1));
        tile[ty][tx] = __ldg(&src[gy * src_w + gx]);
    }
    __syncthreads();

    if (out_x >= dst_w || out_y >= dst_h) return;

    // Center in shared memory for this output pixel
    int cx = threadIdx.x * 2 + HALO;
    int cy = threadIdx.y * 2 + HALO;

    // 5-tap Gaussian [1,4,6,4,1]/16 applied as separable 2D (outer product / 256)
    const float k[5] = {1.0f / 16.0f, 4.0f / 16.0f, 6.0f / 16.0f, 4.0f / 16.0f, 1.0f / 16.0f};
    float sum = 0.0f;

    #pragma unroll
    for (int dy = -2; dy <= 2; ++dy) {
        #pragma unroll
        for (int dx = -2; dx <= 2; ++dx) {
            sum += k[dy + 2] * k[dx + 2] * tile[cy + dy][cx + dx];
        }
    }

    dst[out_y * dst_w + out_x] = sum;
}
"""

_SCHARR_GRADIENT_SRC = r"""
#define BLOCK 16
#define TILE  (BLOCK + 2)

extern "C" __global__
void scharr_gradient_f32(
    const float* __restrict__ src,
    float*       __restrict__ Gx,
    float*       __restrict__ Gy,
    int width, int height)
{
    __shared__ float tile[TILE][TILE];

    const int gx = blockIdx.x * BLOCK + threadIdx.x;
    const int gy = blockIdx.y * BLOCK + threadIdx.y;

    // Load shared memory tile with 1-pixel halo
    const int tile_x0 = blockIdx.x * BLOCK - 1;
    const int tile_y0 = blockIdx.y * BLOCK - 1;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int n_threads = BLOCK * BLOCK;
    const int tile_elems = TILE * TILE;

    for (int i = tid; i < tile_elems; i += n_threads) {
        int ty = i / TILE;
        int tx = i % TILE;
        int sx = tile_x0 + tx;
        int sy = tile_y0 + ty;
        sx = max(0, min(sx, width - 1));
        sy = max(0, min(sy, height - 1));
        tile[ty][tx] = __ldg(&src[sy * width + sx]);
    }
    __syncthreads();

    if (gx >= width || gy >= height) return;

    // Position in shared memory (offset by halo=1)
    int lx = threadIdx.x + 1;
    int ly = threadIdx.y + 1;

    // Scharr X: [[-3,0,3],[-10,0,10],[-3,0,3]] / 32
    float dx = (-3.0f * tile[ly - 1][lx - 1] + 3.0f * tile[ly - 1][lx + 1]
              - 10.0f * tile[ly    ][lx - 1] + 10.0f * tile[ly    ][lx + 1]
              -  3.0f * tile[ly + 1][lx - 1] + 3.0f * tile[ly + 1][lx + 1]) / 32.0f;

    // Scharr Y: [[-3,-10,-3],[0,0,0],[3,10,3]] / 32
    float dy = (-3.0f  * tile[ly - 1][lx - 1] - 10.0f * tile[ly - 1][lx] - 3.0f * tile[ly - 1][lx + 1]
              +  3.0f  * tile[ly + 1][lx - 1] + 10.0f * tile[ly + 1][lx] + 3.0f * tile[ly + 1][lx + 1]) / 32.0f;

    int idx = gy * width + gx;
    Gx[idx] = dx;
    Gy[idx] = dy;
}
"""

_SHI_TOMASI_SRC = r"""
// half_window is fixed at MAX_HALF_WIN=3 to avoid runtime-vs-compile-time
// stride mismatch in the 2D shared memory arrays.
#define ST_BLOCK      16
#define MAX_HALF_WIN  3
#define TILE_SIDE     (ST_BLOCK + 2 * MAX_HALF_WIN)  // 22

extern "C" __global__
void shi_tomasi_response_f32(
    const float* __restrict__ Ix,
    const float* __restrict__ Iy,
    float*       __restrict__ response,
    int width, int height, int half_window)
{
    __shared__ float sIx[TILE_SIDE][TILE_SIDE];
    __shared__ float sIy[TILE_SIDE][TILE_SIDE];

    const int gx = blockIdx.x * ST_BLOCK + threadIdx.x;
    const int gy = blockIdx.y * ST_BLOCK + threadIdx.y;

    const int tile_x0 = blockIdx.x * ST_BLOCK - MAX_HALF_WIN;
    const int tile_y0 = blockIdx.y * ST_BLOCK - MAX_HALF_WIN;

    const int tid       = threadIdx.y * blockDim.x + threadIdx.x;
    const int n_threads = ST_BLOCK * ST_BLOCK;

    // Always load TILE_SIDE x TILE_SIDE using the compile-time stride.
    for (int i = tid; i < TILE_SIDE * TILE_SIDE; i += n_threads) {
        int ty = i / TILE_SIDE;
        int tx = i % TILE_SIDE;
        int sx = max(0, min(tile_x0 + tx, width  - 1));
        int sy = max(0, min(tile_y0 + ty, height - 1));
        int idx = sy * width + sx;
        sIx[ty][tx] = __ldg(&Ix[idx]);
        sIy[ty][tx] = __ldg(&Iy[idx]);
    }
    __syncthreads();

    if (gx >= width || gy >= height) return;

    // Shared-memory center for this thread, offset by MAX_HALF_WIN.
    int lx = threadIdx.x + MAX_HALF_WIN;
    int ly = threadIdx.y + MAX_HALF_WIN;

    float Sxx = 0.0f, Syy = 0.0f, Sxy = 0.0f;
    for (int dy = -half_window; dy <= half_window; ++dy) {
        for (int dx = -half_window; dx <= half_window; ++dx) {
            float ix = sIx[ly + dy][lx + dx];
            float iy = sIy[ly + dy][lx + dx];
            Sxx += ix * ix;
            Syy += iy * iy;
            Sxy += ix * iy;
        }
    }

    float trace = Sxx + Syy;
    float det   = Sxx * Syy - Sxy * Sxy;
    float disc  = fmaxf(trace * trace * 0.25f - det, 0.0f);
    response[gy * width + gx] = fmaxf(trace * 0.5f - sqrtf(disc), 0.0f);
}
"""


# ---------------------------------------------------------------------------
# Kernel cache
# ---------------------------------------------------------------------------

class CudaKernelCache:
    """Lazily compiles and caches CUDA kernels. Thread-safe via module-level singleton."""

    _instance: Optional["CudaKernelCache"] = None
    _kernels: dict = {}

    @classmethod
    def get(cls) -> "CudaKernelCache":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _compile(self, name: str, src: str, func_name: str) -> "cp.RawKernel":
        if name not in self._kernels:
            self._kernels[name] = cp.RawKernel(
                src, func_name, options=("--std=c++14",)
            )
        return self._kernels[name]

    def warp_kernel(self) -> "cp.RawKernel":
        return self._compile("warp", _AFFINE_WARP_SRC, "affine_warp_bilinear_u8")

    def downsample_kernel(self) -> "cp.RawKernel":
        return self._compile("downsample", _GAUSSIAN_DOWNSAMPLE_SRC, "gaussian_downsample_f32")

    def shi_tomasi_kernel(self) -> "cp.RawKernel":
        return self._compile("shi_tomasi", _SHI_TOMASI_SRC, "shi_tomasi_response_f32")

    def scharr_kernel(self) -> "cp.RawKernel":
        return self._compile("scharr", _SCHARR_GRADIENT_SRC, "scharr_gradient_f32")


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------

def affine_warp_u8(
    frame_gpu: "cp.ndarray",
    transform_2x3: np.ndarray,
    out_w: int,
    out_h: int,
    stream: Optional["cp.cuda.Stream"] = None,
) -> "cp.ndarray":
    """GPU affine warp via raw CUDA kernel. Returns (out_h, out_w, 3) uint8.

    *transform_2x3* maps dst -> src (already inverted).
    """
    assert frame_gpu.dtype == np.uint8 and frame_gpu.ndim == 3
    src_h, src_w = frame_gpu.shape[:2]

    dst = cp.zeros((out_h, out_w, 3), dtype=cp.uint8)
    kernel = CudaKernelCache.get().warp_kernel()

    block = (32, 8)
    grid = ((out_w + block[0] - 1) // block[0],
            (out_h + block[1] - 1) // block[1])

    m = transform_2x3.astype(np.float32)
    args = (
        frame_gpu, dst,
        np.int32(src_w), np.int32(src_h),
        np.int32(out_w), np.int32(out_h),
        np.float32(m[0, 0]), np.float32(m[0, 1]), np.float32(m[0, 2]),
        np.float32(m[1, 0]), np.float32(m[1, 1]), np.float32(m[1, 2]),
    )

    if stream is not None:
        kernel(grid, block, args, stream=stream)
    else:
        kernel(grid, block, args)

    return dst


def gaussian_downsample(
    img_gpu: "cp.ndarray",
    stream: Optional["cp.cuda.Stream"] = None,
) -> "cp.ndarray":
    """Downsample float32 image by 2x with Gaussian anti-aliasing."""
    assert img_gpu.dtype == np.float32 and img_gpu.ndim == 2
    src_h, src_w = img_gpu.shape
    dst_w = (src_w + 1) // 2
    dst_h = (src_h + 1) // 2

    dst = cp.empty((dst_h, dst_w), dtype=cp.float32)
    kernel = CudaKernelCache.get().downsample_kernel()

    block = (16, 16)
    grid = ((dst_w + block[0] - 1) // block[0],
            (dst_h + block[1] - 1) // block[1])

    args = (
        img_gpu, dst,
        np.int32(src_w), np.int32(src_h),
        np.int32(dst_w), np.int32(dst_h),
    )

    if stream is not None:
        kernel(grid, block, args, stream=stream)
    else:
        kernel(grid, block, args)

    return dst


def scharr_gradients(
    img_gpu: "cp.ndarray",
    stream: Optional["cp.cuda.Stream"] = None,
) -> Tuple["cp.ndarray", "cp.ndarray"]:
    """Compute Scharr gradients. Returns (Gx, Gy) float32."""
    assert img_gpu.dtype == np.float32 and img_gpu.ndim == 2
    h, w = img_gpu.shape

    Gx = cp.empty_like(img_gpu)
    Gy = cp.empty_like(img_gpu)
    kernel = CudaKernelCache.get().scharr_kernel()

    block = (16, 16)
    grid = ((w + block[0] - 1) // block[0],
            (h + block[1] - 1) // block[1])

    args = (img_gpu, Gx, Gy, np.int32(w), np.int32(h))

    if stream is not None:
        kernel(grid, block, args, stream=stream)
    else:
        kernel(grid, block, args)

    return Gx, Gy


def shi_tomasi_response(
    Gx_gpu: "cp.ndarray",
    Gy_gpu: "cp.ndarray",
    half_window: int = 3,
    stream: Optional["cp.cuda.Stream"] = None,
) -> "cp.ndarray":
    """Compute Shi-Tomasi corner response. Returns float32 response map."""
    assert Gx_gpu.dtype == np.float32 and Gx_gpu.ndim == 2
    assert half_window <= 3, "Maximum supported half_window is 3"
    h, w = Gx_gpu.shape

    response = cp.empty((h, w), dtype=cp.float32)
    kernel = CudaKernelCache.get().shi_tomasi_kernel()

    block = (16, 16)
    grid = ((w + block[0] - 1) // block[0],
            (h + block[1] - 1) // block[1])

    args = (Gx_gpu, Gy_gpu, response, np.int32(w), np.int32(h), np.int32(half_window))

    if stream is not None:
        kernel(grid, block, args, stream=stream)
    else:
        kernel(grid, block, args)

    return response
