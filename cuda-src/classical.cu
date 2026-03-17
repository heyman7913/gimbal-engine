#include "kernels.h"

#include <ATen/cuda/CUDAContext.h>

// scharr gradients, divided by 32 to match cv2.Scharr. borders repeat the edge pixel.
__global__ void scharr_kernel(
    const float* __restrict__ img, float* __restrict__ ix, float* __restrict__ iy,
    int h, int w) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  const int xm = max(x - 1, 0), xp = min(x + 1, w - 1);
  const int ym = max(y - 1, 0), yp = min(y + 1, h - 1);
#define P(xx, yy) img[(yy) * w + (xx)]
  const float tl = P(xm, ym), tc = P(x, ym), tr = P(xp, ym);
  const float ml = P(xm, y), mr = P(xp, y);
  const float bl = P(xm, yp), bc = P(x, yp), br = P(xp, yp);
#undef P
  ix[y * w + x] = (-3.f * tl + 3.f * tr - 10.f * ml + 10.f * mr - 3.f * bl + 3.f * br) / 32.f;
  iy[y * w + x] = (-3.f * tl - 10.f * tc - 3.f * tr + 3.f * bl + 10.f * bc + 3.f * br) / 32.f;
}

// shi-tomasi corner score: smaller eigenvalue of the structure tensor over a (2R+1) box.
__global__ void shi_tomasi_kernel(
    const float* __restrict__ ix, const float* __restrict__ iy, float* __restrict__ resp,
    int h, int w, int radius) {
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
  resp[y * w + x] = t - sqrtf(d * d + sxy * sxy);
}

// halve the image with the [1,4,6,4,1] gaussian, like cv2.pyrDown.
__global__ void downsample_kernel(
    const float* __restrict__ src, float* __restrict__ dst, int sh, int sw, int dh, int dw) {
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

// for each output pixel, go back through h_inv and bilinearly sample the source. off-frame
// taps are black so the auto-crop can trim the borders later.
__global__ void warp_kernel(
    const unsigned char* __restrict__ src, unsigned char* __restrict__ dst,
    const double* __restrict__ h_inv, int sh, int sw, int dh, int dw, int ch) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dw || y >= dh) return;
  const double X = h_inv[0] * x + h_inv[1] * y + h_inv[2];
  const double Y = h_inv[3] * x + h_inv[4] * y + h_inv[5];
  const double Wd = h_inv[6] * x + h_inv[7] * y + h_inv[8];
  unsigned char* out = dst + (y * dw + x) * ch;
  if (Wd == 0.0) {
    for (int c = 0; c < ch; ++c) out[c] = 0;
    return;
  }
  double sx = X / Wd, sy = Y / Wd;
  // a near-singular stabilizing homography can map a pixel to a non-finite or astronomically
  // distant source coordinate; reject non-finite and clamp the rest to a small margin so the
  // integer neighbours below stay in representable range. Anything past the frame still samples
  // to black through the per-neighbour bounds checks.
  if (!isfinite(sx) || !isfinite(sy)) {
    for (int c = 0; c < ch; ++c) out[c] = 0;
    return;
  }
  sx = fmin(fmax(sx, -2.0), (double)(sw + 1));
  sy = fmin(fmax(sy, -2.0), (double)(sh + 1));
  const int x0 = (int)floor(sx), y0 = (int)floor(sy);
  const double ax = sx - x0, ay = sy - y0;
  for (int c = 0; c < ch; ++c) {
    double acc = 0.0;
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

static dim3 grid2d(int w, int h, dim3 block) {
  return dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
}

std::vector<torch::Tensor> scharr_gradient(torch::Tensor img) {
  const int h = img.size(0), w = img.size(1);
  auto ix = torch::empty_like(img);
  auto iy = torch::empty_like(img);
  const dim3 block(16, 16);
  auto stream = at::cuda::getCurrentCUDAStream();
  scharr_kernel<<<grid2d(w, h, block), block, 0, stream>>>(
      img.data_ptr<float>(), ix.data_ptr<float>(), iy.data_ptr<float>(), h, w);
  return {ix, iy};
}

torch::Tensor shi_tomasi_response(torch::Tensor ix, torch::Tensor iy, int64_t radius) {
  const int h = ix.size(0), w = ix.size(1);
  auto resp = torch::empty_like(ix);
  const dim3 block(16, 16);
  auto stream = at::cuda::getCurrentCUDAStream();
  shi_tomasi_kernel<<<grid2d(w, h, block), block, 0, stream>>>(
      ix.data_ptr<float>(), iy.data_ptr<float>(), resp.data_ptr<float>(), h, w, radius);
  return resp;
}

torch::Tensor gaussian_downsample(torch::Tensor img) {
  const int sh = img.size(0), sw = img.size(1);
  const int dh = (sh + 1) / 2, dw = (sw + 1) / 2;
  auto dst = torch::empty({dh, dw}, img.options());
  const dim3 block(16, 16);
  auto stream = at::cuda::getCurrentCUDAStream();
  downsample_kernel<<<grid2d(dw, dh, block), block, 0, stream>>>(
      img.data_ptr<float>(), dst.data_ptr<float>(), sh, sw, dh, dw);
  return dst;
}

torch::Tensor warp_bilinear(torch::Tensor src, torch::Tensor h_inv, int64_t out_h, int64_t out_w) {
  const int sh = src.size(0), sw = src.size(1), ch = src.size(2);
  auto dst = torch::empty({out_h, out_w, ch}, src.options());
  const dim3 block(16, 16);
  auto stream = at::cuda::getCurrentCUDAStream();
  warp_kernel<<<grid2d(out_w, out_h, block), block, 0, stream>>>(
      src.data_ptr<unsigned char>(), dst.data_ptr<unsigned char>(), h_inv.data_ptr<double>(),
      sh, sw, out_h, out_w, ch);
  return dst;
}
