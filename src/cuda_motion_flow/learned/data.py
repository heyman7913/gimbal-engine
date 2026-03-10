"""Training data: synthetic homography pairs (Phase A) and real video pairs (Phase B).

Synthetic pairs are generated entirely on the GPU with grid_sample, so the data path never
starves the device. A pool of grayscale images is held on the host as uint8; each step a random
batch is moved to the GPU, a random 4-corner perturbation defines the homography, and both
patches are sampled from the (larger) source image so the warped patch is filled with real
content rather than borders. The corner offsets are the free supervision labels.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .model import _corner_template, dlt_solve


def load_image_pool(
    directory: str | Path, size: tuple[int, int] = (240, 320), limit: int = 20000
) -> torch.Tensor:
    """Load grayscale images from a directory into a single host uint8 tensor (N, H, W)."""
    paths = sorted(
        p for p in Path(directory).iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"}
    )
    paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"no images found in {directory}")
    h, w = size
    out = np.empty((len(paths), h, w), dtype=np.uint8)
    for i, p in enumerate(paths):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"failed to decode image: {p}")
        out[i] = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(out)


def _sample_patch(
    src: torch.Tensor, px: torch.Tensor, py: torch.Tensor, h_inv: torch.Tensor, patch: int
) -> torch.Tensor:
    """Sample a (patch x patch) crop from src at (px, py), warped by h_inv (local dst -> src)."""
    b, _, hh, ww = src.shape
    device = src.device
    qy, qx = torch.meshgrid(
        torch.arange(patch, device=device, dtype=torch.float32),
        torch.arange(patch, device=device, dtype=torch.float32),
        indexing="ij",
    )
    q = torch.stack([qx.reshape(-1), qy.reshape(-1), torch.ones(patch * patch, device=device)], 0)
    pts = h_inv @ q  # (B, 3, P*P)
    w = pts[:, 2:3]
    w = torch.where(w.abs() < 1e-8, torch.full_like(w, 1e-8), w)
    gx = px[:, None] + pts[:, 0] / w[:, 0]  # (B, P*P)
    gy = py[:, None] + pts[:, 1] / w[:, 0]
    xn = (2.0 * gx / (ww - 1) - 1.0).reshape(b, patch, patch)
    yn = (2.0 * gy / (hh - 1) - 1.0).reshape(b, patch, patch)
    grid = torch.stack([xn, yn], dim=-1)
    return torch.nn.functional.grid_sample(
        src, grid, mode="bilinear", padding_mode="border", align_corners=True
    )


class SyntheticPairGenerator:
    """Yields (patch_a, patch_b, corner_offsets) batches generated on the GPU."""

    def __init__(
        self,
        pool: torch.Tensor,
        patch: int = 128,
        rho: int = 32,
        margin: int = 16,
        device: str = "cuda",
        seed: int = 0,
    ) -> None:
        self.pool = pool  # host uint8 (N, H, W)
        self.patch = patch
        self.rho = rho
        self.margin = margin
        self.device = device
        self.gen = torch.Generator().manual_seed(seed)
        self.src_corners = _corner_template(patch, torch.device(device))

    def sample(self, batch: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n, h, w = self.pool.shape
        idx = torch.randint(0, n, (batch,), generator=self.gen)
        imgs = self.pool[idx].to(self.device).float().div_(255.0).unsqueeze(1)  # (B,1,H,W)

        lo_x = self.margin + self.rho
        hi_x = w - self.patch - self.margin - self.rho
        lo_y = self.margin + self.rho
        hi_y = h - self.patch - self.margin - self.rho
        px = torch.randint(lo_x, hi_x, (batch,), generator=self.gen).to(self.device).float()
        py = torch.randint(lo_y, hi_y, (batch,), generator=self.gen).to(self.device).float()

        delta = (torch.rand(batch, 4, 2, generator=self.gen) * 2 - 1) * self.rho
        delta = delta.to(self.device)
        src = self.src_corners.unsqueeze(0).expand(batch, 4, 2)
        h_local = dlt_solve(src, src + delta)  # patch-scale A -> B
        h_inv = torch.linalg.inv(h_local)
        eye = torch.eye(3, device=self.device).unsqueeze(0).expand(batch, 3, 3)

        patch_a = _sample_patch(imgs, px, py, eye, self.patch)
        patch_b = _sample_patch(imgs, px, py, h_inv, self.patch)
        return patch_a, patch_b, delta


def video_frame_pairs(
    clip_paths: list[Path], patch: int = 128, stride: int = 2, limit_per_clip: int = 200
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Consecutive grayscale frame pairs from clips, resized to (patch, patch) for Phase B."""
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for path in clip_paths:
        cap = cv2.VideoCapture(str(path))
        prev = None
        count = 0
        while count < limit_per_clip:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.resize(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                (patch, patch),
                interpolation=cv2.INTER_AREA,
            )
            if prev is not None:
                pairs.append((prev, gray))
                count += 1
            prev = gray if count % stride == 0 else prev
        cap.release()
    return pairs
