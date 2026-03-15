"""The training losses.

Phase A is supervised: L1 on the corner offsets (weighting the later iterations more), and I
report MACE, the average corner error in pixels.

Phase B is unsupervised: a photometric loss in feature space, weighted by a learned mask. The
trick is to divide by the mask's own sum, otherwise the network would just push the mask to
zero to make the loss vanish.
"""

from __future__ import annotations

import torch

from .model import IHN, homography_warp


def supervised_corner_loss(
    offsets: list[torch.Tensor], target: torch.Tensor, gamma: float = 0.85
) -> torch.Tensor:
    """Sum over refinement iterations of gamma-weighted L1, later iterations weighted more."""
    n = len(offsets)
    loss = offsets[0].new_zeros(())
    for i, off in enumerate(offsets):
        weight = gamma ** (n - 1 - i)
        loss = loss + weight * (off - target).abs().mean()
    return loss


def mace(offset: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean average corner error in pixels between predicted and target corner offsets."""
    return ((offset - target) ** 2).sum(dim=-1).sqrt().mean()


def mesh_corner_loss(offsets: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 between per-cell offsets (B, gh, gw, 4, 2) and a target. A (B, 4, 2) target is the
    global motion broadcast to every cell (used for the global pretrain)."""
    if target.dim() == 3:
        target = target[:, None, None]
    return (offsets - target).abs().mean()


def mesh_smoothness(offsets: torch.Tensor) -> torch.Tensor:
    """Penalty on the difference between neighbouring cells' offsets (the parallax inductive
    bias). offsets is (B, gh, gw, 4, 2)."""
    dh = (offsets[:, 1:] - offsets[:, :-1]).abs().mean()
    dw = (offsets[:, :, 1:] - offsets[:, :, :-1]).abs().mean()
    return dh + dw


def photometric_feature_loss(
    model: IHN, patch_a: torch.Tensor, patch_b: torch.Tensor, h: torch.Tensor, eps: float = 1e-3
) -> torch.Tensor:
    """Masked feature-space photometric loss between A and B warped into A's frame."""
    fa, ma = model.features_and_mask(patch_a)
    fb, mb = model.features_and_mask(patch_b)
    h_feat = model._h_feat(h)
    fb_w = homography_warp(fb, h_feat)
    mb_w = homography_warp(mb, h_feat)

    weight = ma * mb_w  # only trust pixels both frames are confident about
    diff = (fa - fb_w).abs().mean(dim=1, keepdim=True)
    # dividing by the mask sum is the important bit - it stops the net cheating by zeroing the mask
    return (weight * diff).sum() / (weight.sum() + eps)
