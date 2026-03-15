#pragma once
#include <torch/extension.h>
#include <vector>

// fused local correlation (the cost volume). fa, fb are (B, C, H, W).
torch::Tensor correlation_forward(torch::Tensor fa, torch::Tensor fb, int64_t radius);
std::vector<torch::Tensor> correlation_backward(
    torch::Tensor grad, torch::Tensor fa, torch::Tensor fb, int64_t radius);
// optimization variants used by the systems study (forward only).
torch::Tensor correlation_forward_v1(torch::Tensor fa, torch::Tensor fb, int64_t radius);
torch::Tensor correlation_forward_v2(torch::Tensor fa_nhwc, torch::Tensor fb_nhwc, int64_t radius);

// classical pipeline kernels. images are float32 (H, W) unless noted.
std::vector<torch::Tensor> scharr_gradient(torch::Tensor img);
torch::Tensor shi_tomasi_response(torch::Tensor ix, torch::Tensor iy, int64_t radius);
torch::Tensor gaussian_downsample(torch::Tensor img);
// src is (H, W, C) uint8, h_inv is a 3x3 float64 (dst -> src). returns (out_h, out_w, C) uint8.
torch::Tensor warp_bilinear(torch::Tensor src, torch::Tensor h_inv, int64_t out_h, int64_t out_w);
