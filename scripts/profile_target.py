"""Tiny target for profilers: run the fused correlation forward+backward at a fixed shape.

Shapes match the IHN at 16x16 features (batch 32, 96 channels, radius 4). Used by ncu.
"""

import gimbal._cuda as ext
import torch

torch.manual_seed(0)
B, C, H, W, R = 32, 96, 16, 16, 4
fa = torch.randn(B, C, H, W, device="cuda", requires_grad=True)
fb = torch.randn(B, C, H, W, device="cuda", requires_grad=True)

for _ in range(3):  # warmup so the profiled launch is steady state
    out = ext.correlation_forward(fa, fb, R)
torch.cuda.synchronize()

out = ext.correlation_forward(fa, fb, R)
grad = torch.ones_like(out)
ext.correlation_backward(grad, fa, fb, R)
torch.cuda.synchronize()
