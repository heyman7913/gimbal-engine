"""Iterative homography network (IHN lineage) and a single-shot regression baseline.

The network regresses the 4-point parameterization (eight corner offsets), never the raw
homography entries. A differentiable Tensor-DLT layer turns the four corner correspondences
into a 3x3 homography, so gradients flow from H back to the offsets. The iterative model warps
the second feature map by the current estimate, correlates it with the first, and predicts a
residual to the offsets; the regression baseline predicts all eight offsets in a single shot.

All coordinates are in patch pixels (default 128). H maps frame-A coordinates to frame-B
coordinates.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from .correlation import local_correlation


def dlt_solve(src: torch.Tensor, dst: torch.Tensor, ridge: float = 0.0) -> torch.Tensor:
    """Differentiable Tensor-DLT. src, dst: (B, 4, 2) corner correspondences -> H (B, 3, 3).

    Convention: rows [x y 1 0 0 0 -ux -uy].h = u and [0 0 0 x y 1 -vx -vy].h = v, h33 fixed to
    1. The solve is forced to fp32 even under autocast, since the 8x8 system is ill-conditioned
    in bf16. ridge > 0 switches to regularized normal equations.
    """
    ctx = (
        torch.autocast(device_type="cuda", enabled=False)
        if src.is_cuda
        else contextlib.nullcontext()
    )
    with ctx:
        # keep float64 (gradcheck) and float32 as-is; only lift bf16/half out of autocast
        if src.dtype not in (torch.float32, torch.float64):
            src = src.float()
            dst = dst.float()
        b = src.shape[0]
        x, y = src[..., 0], src[..., 1]
        u, v = dst[..., 0], dst[..., 1]
        o, i = torch.zeros_like(x), torch.ones_like(x)
        row_u = torch.stack([x, y, i, o, o, o, -u * x, -u * y], dim=-1)  # (B, 4, 8)
        row_v = torch.stack([o, o, o, x, y, i, -v * x, -v * y], dim=-1)
        a = torch.cat([row_u, row_v], dim=1)  # (B, 8, 8)
        rhs = torch.cat([u, v], dim=1).unsqueeze(-1)  # (B, 8, 1)
        if ridge > 0.0:
            at = a.transpose(1, 2)
            eye = ridge * torch.eye(8, device=a.device, dtype=a.dtype)
            h8 = torch.linalg.solve(at @ a + eye, at @ rhs)
        else:
            h8 = torch.linalg.solve(a, rhs)
        h8 = h8.squeeze(-1)
        ones = torch.ones(b, 1, device=a.device, dtype=a.dtype)
        return torch.cat([h8, ones], dim=1).reshape(b, 3, 3)


def homography_warp(fb: torch.Tensor, h_feat: torch.Tensor) -> torch.Tensor:
    """Sample fb at h_feat applied to each output grid point (aligns fb into fa's frame)."""
    b, _, hh, ww = fb.shape
    ys, xs = torch.meshgrid(
        torch.arange(hh, device=fb.device, dtype=fb.dtype),
        torch.arange(ww, device=fb.device, dtype=fb.dtype),
        indexing="ij",
    )
    grid = torch.stack([xs.reshape(-1), ys.reshape(-1), torch.ones(hh * ww, device=fb.device)], 0)
    pts = h_feat @ grid  # (B, 3, hh*ww)
    w = pts[:, 2:3]
    w = torch.where(w.abs() < 1e-8, torch.full_like(w, 1e-8), w)
    xp = (pts[:, 0:1] / w).reshape(b, hh, ww)
    yp = (pts[:, 1:2] / w).reshape(b, hh, ww)
    xn = 2.0 * xp / (ww - 1) - 1.0
    yn = 2.0 * yp / (hh - 1) - 1.0
    sample_grid = torch.stack([xn, yn], dim=-1)
    return F.grid_sample(fb, sample_grid, mode="bilinear", padding_mode="zeros", align_corners=True)


class _ResBlock(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        self.norm1 = nn.InstanceNorm2d(c, affine=True)
        self.norm2 = nn.InstanceNorm2d(c, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.relu(x + y)


class FeatureEncoder(nn.Module):
    """Grayscale (B,1,S,S) -> (B, dim, S/8, S/8). Shared weights across both frames."""

    def __init__(self, dim: int = 96) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 7, stride=2, padding=3),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, dim, 3, stride=2, padding=1),
            nn.InstanceNorm2d(dim, affine=True),
            nn.ReLU(inplace=True),
        )
        self.res1 = _ResBlock(dim)
        self.res2 = _ResBlock(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.res2(self.res1(self.stem(x)))
        return out


class _OffsetHead(nn.Module):
    """Map a feature map to eight corner offsets via conv + global pool + MLP."""

    def __init__(self, in_ch: int, hidden: int = 128) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 8)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.mlp(self.body(x).flatten(1))
        return out


def _corner_template(size: int, device: torch.device | None = None) -> torch.Tensor:
    return torch.tensor(
        [[0.0, 0.0], [size - 1, 0.0], [size - 1, size - 1], [0.0, size - 1]], device=device
    )


class IHN(nn.Module):
    """Iterative homography estimator: correlation-driven refinement of the 4-point offsets."""

    corners: torch.Tensor
    s_to_feat: torch.Tensor
    s_from_feat: torch.Tensor

    def __init__(self, size: int = 128, dim: int = 96, radius: int = 4, iters: int = 6) -> None:
        super().__init__()
        self.size = size
        self.stride = 8
        self.radius = radius
        self.iters = iters
        self.use_fused = False  # flipped on only after the gradcheck gate passes
        self.encoder = FeatureEncoder(dim)
        self.head = _OffsetHead(dim + (2 * radius + 1) ** 2)
        # content mask for unsupervised photometric loss (Phase B only)
        self.mask_head = nn.Sequential(
            nn.Conv2d(dim, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )
        self.register_buffer("corners", _corner_template(size))
        s = 1.0 / self.stride
        self.register_buffer("s_to_feat", torch.tensor([[s, 0, 0], [0, s, 0], [0, 0, 1.0]]))
        self.register_buffer(
            "s_from_feat", torch.tensor([[1 / s, 0, 0], [0, 1 / s, 0], [0, 0, 1.0]])
        )

    def _h_feat(self, h: torch.Tensor) -> torch.Tensor:
        return self.s_to_feat @ h @ self.s_from_feat

    def features_and_mask(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encoder features and content mask for the unsupervised photometric loss."""
        feat = self.encoder(img)
        return feat, self.mask_head(feat)

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor) -> list[torch.Tensor]:
        """Return the list of 4-point offsets (B, 4, 2), one per refinement iteration."""
        fa = self.encoder(img_a)
        fb = self.encoder(img_b)
        b = img_a.shape[0]
        src = self.corners.unsqueeze(0).expand(b, 4, 2)
        offset = torch.zeros(b, 4, 2, device=img_a.device)
        outputs = []
        for _ in range(self.iters):
            h = dlt_solve(src, src + offset, ridge=1e-4)
            fb_w = homography_warp(fb, self._h_feat(h))
            corr = local_correlation(fa, fb_w, self.radius, fused=self.use_fused)
            delta = self.head(torch.cat([corr, fa], dim=1)).view(b, 4, 2)
            offset = offset + delta
            outputs.append(offset)
        return outputs

    def predict(self, img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
        """Final homography (B, 3, 3) at patch scale."""
        offset = self.forward(img_a, img_b)[-1]
        src = self.corners.unsqueeze(0).expand(img_a.shape[0], 4, 2)
        return dlt_solve(src, src + offset, ridge=1e-4)


class RegressionHomographyNet(nn.Module):
    """DeTone-style single-shot baseline: predict all eight offsets at once (the ablation)."""

    corners: torch.Tensor

    def __init__(self, size: int = 128, dim: int = 96) -> None:
        super().__init__()
        self.size = size
        self.encoder = FeatureEncoder(dim)
        self.head = _OffsetHead(dim)
        self.register_buffer("corners", _corner_template(size))

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor) -> list[torch.Tensor]:
        # the two frames share the encoder; their features are concatenated by addition so the
        # head sees both. Returns a one-element list to match the IHN interface.
        fa = self.encoder(img_a)
        fb = self.encoder(img_b)
        offset = self.head(fa - fb).view(img_a.shape[0], 4, 2)
        return [offset]

    def predict(self, img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
        offset = self.forward(img_a, img_b)[-1]
        src = self.corners.unsqueeze(0).expand(img_a.shape[0], 4, 2)
        return dlt_solve(src, src + offset, ridge=1e-4)
