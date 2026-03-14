"""The homography network (IHN style) plus a single-shot regression baseline to compare against.

The net predicts how the four patch corners move (8 numbers), not the raw 3x3 entries - that
is much better behaved. The Tensor-DLT layer then turns those four corners into a homography,
and since it is differentiable the gradients flow back to the corners. The iterative model
keeps refining the corners; the baseline just predicts them once.

Everything is in patch pixels (128 by default), and H maps frame A to frame B.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from .correlation import local_correlation


def _solve_nopivot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batched solve of A x = b by gaussian elimination, no pivoting, written functionally.

    torch.linalg.solve calls cuSOLVER which does a host sync and so breaks CUDA-graph capture.
    This stays on the device and is differentiable, at the cost of needing A well-conditioned
    (it is only used on the ridge-regularised normal equations, which are SPD). n is small (8).
    """
    n = a.shape[-1]
    m = torch.cat([a, b], dim=-1)  # augmented (B, n, n+1)
    for c in range(n):
        factor = (m[:, c + 1 :, c] / m[:, c, c].unsqueeze(-1)).unsqueeze(-1)
        new_below = m[:, c + 1 :, :] - factor * m[:, c : c + 1, :]
        m = torch.cat([m[:, : c + 1, :], new_below], dim=1)
    cols: list[torch.Tensor] = [a.new_zeros(a.shape[0])] * n
    for r in range(n - 1, -1, -1):
        acc = m[:, r, n]
        for j in range(r + 1, n):
            acc = acc - m[:, r, j] * cols[j]
        cols[r] = acc / m[:, r, r]
    return torch.stack(cols, dim=1).unsqueeze(-1)


def dlt_solve(
    src: torch.Tensor, dst: torch.Tensor, ridge: float = 0.0, graph_safe: bool = False
) -> torch.Tensor:
    """Differentiable DLT. src, dst are (B, 4, 2) corner pairs, returns H (B, 3, 3).

    Each point gives two rows [x y 1 0 0 0 -ux -uy].h = u and [0 0 0 x y 1 -vx -vy].h = v, with
    h33 pinned to 1. ridge > 0 uses the regularised normal equations instead.
    """
    # force the solve to fp32 - the 8x8 system falls apart in bf16
    ctx = (
        torch.autocast(device_type="cuda", enabled=False)
        if src.is_cuda
        else contextlib.nullcontext()
    )
    with ctx:
        # leave float32/float64 alone (gradcheck needs the doubles), only fix bf16/half
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
            ata, atb = at @ a + eye, at @ rhs
            h8 = _solve_nopivot(ata, atb) if graph_safe else torch.linalg.solve(ata, atb)
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


def mesh_sampling_grid(field: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """grid_sample grid (B, H, W, 2) from a (B, gh, gw, 3, 3) field of per-cell homographies.

    Each pixel takes the four homographies of the cells around its continuous grid position,
    applies each to the pixel, and bilinearly blends the resulting coordinates. If every cell
    holds the same homography this reduces exactly to that single homography's grid.
    """
    b, gh, gw = field.shape[0], field.shape[1], field.shape[2]
    dev = field.device
    ys, xs = torch.meshgrid(
        torch.arange(h, device=dev, dtype=field.dtype),
        torch.arange(w, device=dev, dtype=field.dtype),
        indexing="ij",
    )
    fy = (ys + 0.5) / h * gh - 0.5
    fx = (xs + 0.5) / w * gw - 0.5
    c0y = fy.floor().clamp(0, gh - 1).long()
    c0x = fx.floor().clamp(0, gw - 1).long()
    c1y = (c0y + 1).clamp(0, gh - 1)
    c1x = (c0x + 1).clamp(0, gw - 1)
    wy = (fy - c0y.to(field.dtype)).clamp(0, 1)[None, :, :, None]
    wx = (fx - c0x.to(field.dtype)).clamp(0, 1)[None, :, :, None]
    flat = field.reshape(b, gh * gw, 3, 3)
    p = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1)  # (H, W, 3)

    def apply(cy: torch.Tensor, cx: torch.Tensor) -> torch.Tensor:
        hm = flat[:, (cy * gw + cx).reshape(-1)].reshape(b, h, w, 3, 3)
        m = (hm * p[None, :, :, None, :]).sum(dim=-1)  # (B, H, W, 3)
        denom = m[..., 2:3]
        denom = torch.where(denom.abs() < 1e-8, torch.full_like(denom, 1e-8), denom)
        return m[..., :2] / denom

    top = apply(c0y, c0x) * (1 - wx) + apply(c0y, c1x) * wx
    bot = apply(c1y, c0x) * (1 - wx) + apply(c1y, c1x) * wx
    mapped = top * (1 - wy) + bot * wy
    xn = 2.0 * mapped[..., 0] / (w - 1) - 1.0
    yn = 2.0 * mapped[..., 1] / (h - 1) - 1.0
    return torch.stack([xn, yn], dim=-1)


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
        self.use_fused = False  # only turned on once the fused kernel passes gradcheck
        self.graph_safe = False  # use the capturable DLT solve (for CUDA-graph inference)
        self.encoder = FeatureEncoder(dim)
        self.head = _OffsetHead(dim + (2 * radius + 1) ** 2)
        # only used by the unsupervised phase, predicts which pixels to trust
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
            h = dlt_solve(src, src + offset, ridge=1e-4, graph_safe=self.graph_safe)
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
        return dlt_solve(src, src + offset, ridge=1e-4, graph_safe=self.graph_safe)


class RegressionHomographyNet(nn.Module):
    """The baseline from the DeTone paper: predict all 8 offsets in one shot, no iteration."""

    corners: torch.Tensor

    def __init__(self, size: int = 128, dim: int = 96) -> None:
        super().__init__()
        self.size = size
        self.encoder = FeatureEncoder(dim)
        self.head = _OffsetHead(dim)
        self.register_buffer("corners", _corner_template(size))

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor) -> list[torch.Tensor]:
        # same encoder for both frames, the difference of features feeds the head. return a
        # one-item list so this matches the IHN's interface.
        fa = self.encoder(img_a)
        fb = self.encoder(img_b)
        offset = self.head(fa - fb).view(img_a.shape[0], 4, 2)
        return [offset]

    def predict(self, img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
        offset = self.forward(img_a, img_b)[-1]
        src = self.corners.unsqueeze(0).expand(img_a.shape[0], 4, 2)
        return dlt_solve(src, src + offset, ridge=1e-4)


class MeshIHN(nn.Module):
    """A grid of local homographies. It runs the global IHN for the dominant motion, then a head
    predicts a per-cell residual on top, so a 1x1 grid behaves like the global model and a real
    grid can bend to local motion (parallax). corners are shared across cells.
    """

    corners: torch.Tensor

    def __init__(
        self,
        size: int = 128,
        grid: tuple[int, int] = (8, 8),
        dim: int = 96,
        radius: int = 4,
        iters: int = 6,
    ) -> None:
        super().__init__()
        self.size = size
        self.grid = grid
        self.radius = radius
        self.global_ihn = IHN(size, dim, radius, iters)
        self.mesh_head = nn.Sequential(
            nn.Conv2d(dim + (2 * radius + 1) ** 2, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 8, 1),
        )
        self.register_buffer("corners", _corner_template(size))

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
        """Per-cell 4-point offsets, shape (B, gh, gw, 4, 2)."""
        g = self.global_ihn(img_a, img_b)[-1]  # (B, 4, 2) dominant motion
        ihn = self.global_ihn
        fa = ihn.encoder(img_a)
        fb = ihn.encoder(img_b)
        b = img_a.shape[0]
        src = self.corners.unsqueeze(0).expand(b, 4, 2)
        h = dlt_solve(src, src + g, ridge=1e-4, graph_safe=ihn.graph_safe)
        fb_w = homography_warp(fb, ihn._h_feat(h))
        corr = local_correlation(fa, fb_w, self.radius, fused=ihn.use_fused)
        cell = self.mesh_head(torch.cat([corr, fa], dim=1))
        cell = F.adaptive_avg_pool2d(cell, self.grid)  # (B, 8, gh, gw)
        gh, gw = self.grid
        residual = cell.permute(0, 2, 3, 1).reshape(b, gh, gw, 4, 2)
        offsets: torch.Tensor = g[:, None, None, :, :] + residual
        return offsets

    def predict_field(self, img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
        """Per-cell homographies, shape (B, gh, gw, 3, 3)."""
        offsets = self.forward(img_a, img_b)
        b, gh, gw = offsets.shape[0], offsets.shape[1], offsets.shape[2]
        src = self.corners.reshape(1, 4, 2).expand(b * gh * gw, 4, 2)
        dst = offsets.reshape(b * gh * gw, 4, 2) + src
        h = dlt_solve(src, dst, ridge=1e-4, graph_safe=self.global_ihn.graph_safe)
        out: torch.Tensor = h.reshape(b, gh, gw, 3, 3)
        return out
