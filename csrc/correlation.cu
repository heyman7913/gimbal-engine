#include "kernels.h"

#include <ATen/cuda/CUDAContext.h>
#include <cmath>

// local correlation: out[b,k,y,x] = (1/sqrt(C)) * sum_c fa[b,c,y,x] * fb[b,c,y+dy,x+dx],
// where k indexes the (2R+1)^2 window and taps off the edge contribute 0. both backward
// kernels are written as gathers so there are no atomics.

template <typename scalar_t>
__global__ void corr_fwd_kernel(
    const scalar_t* __restrict__ fa, const scalar_t* __restrict__ fb,
    scalar_t* __restrict__ out, int B, int C, int H, int W, int R, scalar_t inv) {
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
  scalar_t acc = 0;
  if (yy >= 0 && yy < H && xx >= 0 && xx < W) {
    for (int c = 0; c < C; ++c) {
      const long base = ((long)b * C + c) * H;
      acc += fa[(base + y) * W + x] * fb[(base + yy) * W + xx];
    }
  }
  out[idx] = acc * inv;
}

template <typename scalar_t>
__global__ void corr_bwd_fa_kernel(
    const scalar_t* __restrict__ grad, const scalar_t* __restrict__ fb,
    scalar_t* __restrict__ gfa, int B, int C, int H, int W, int R, scalar_t inv) {
  const int D = 2 * R + 1;
  const int K = D * D;
  const long total = (long)B * C * H * W;
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int x = idx % W; long t = idx / W;
  const int y = t % H; t /= H;
  const int c = t % C; t /= C;
  const int b = t;
  scalar_t acc = 0;
  for (int k = 0; k < K; ++k) {
    const int dy = k / D - R;
    const int dx = k % D - R;
    const int yy = y + dy, xx = x + dx;
    if (yy >= 0 && yy < H && xx >= 0 && xx < W) {
      const long g = ((long)b * K + k) * H + y;
      const long f = ((long)b * C + c) * H + yy;
      acc += grad[g * W + x] * fb[f * W + xx];
    }
  }
  gfa[idx] = acc * inv;
}

template <typename scalar_t>
__global__ void corr_bwd_fb_kernel(
    const scalar_t* __restrict__ grad, const scalar_t* __restrict__ fa,
    scalar_t* __restrict__ gfb, int B, int C, int H, int W, int R, scalar_t inv) {
  const int D = 2 * R + 1;
  const int K = D * D;
  const long total = (long)B * C * H * W;
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int xx = idx % W; long t = idx / W;
  const int yy = t % H; t /= H;
  const int c = t % C; t /= C;
  const int b = t;
  scalar_t acc = 0;
  for (int k = 0; k < K; ++k) {
    const int dy = k / D - R;
    const int dx = k % D - R;
    const int y = yy - dy, x = xx - dx;  // which source pixel used (yy, xx)
    if (y >= 0 && y < H && x >= 0 && x < W) {
      const long g = ((long)b * K + k) * H + y;
      const long f = ((long)b * C + c) * H + y;
      acc += grad[g * W + x] * fa[f * W + x];
    }
  }
  gfb[idx] = acc * inv;
}

torch::Tensor correlation_forward(torch::Tensor fa, torch::Tensor fb, int64_t radius) {
  const int B = fa.size(0), C = fa.size(1), H = fa.size(2), W = fa.size(3);
  const int K = (2 * radius + 1) * (2 * radius + 1);
  auto out = torch::empty({B, K, H, W}, fa.options());
  const long total = (long)B * K * H * W;
  const int threads = 256;
  const long blocks = (total + threads - 1) / threads;
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(fa.scalar_type(), "correlation_forward", [&] {
    const scalar_t inv = (scalar_t)(1.0 / std::sqrt((double)C));
    corr_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        fa.data_ptr<scalar_t>(), fb.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
        B, C, H, W, radius, inv);
  });
  return out;
}

std::vector<torch::Tensor> correlation_backward(
    torch::Tensor grad, torch::Tensor fa, torch::Tensor fb, int64_t radius) {
  const int B = fa.size(0), C = fa.size(1), H = fa.size(2), W = fa.size(3);
  auto gfa = torch::empty_like(fa);
  auto gfb = torch::empty_like(fb);
  const long total = (long)B * C * H * W;
  const int threads = 256;
  const long blocks = (total + threads - 1) / threads;
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(fa.scalar_type(), "correlation_backward", [&] {
    const scalar_t inv = (scalar_t)(1.0 / std::sqrt((double)C));
    corr_bwd_fa_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        grad.data_ptr<scalar_t>(), fb.data_ptr<scalar_t>(), gfa.data_ptr<scalar_t>(),
        B, C, H, W, radius, inv);
    corr_bwd_fb_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        grad.data_ptr<scalar_t>(), fa.data_ptr<scalar_t>(), gfb.data_ptr<scalar_t>(),
        B, C, H, W, radius, inv);
  });
  return {gfa, gfb};
}
