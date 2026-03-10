"""Training losses.

Supervised (Phase A): iteration-weighted L1 on the 4-point corner offsets, plus MACE (mean
average corner error in pixels) as the reported metric.

Unsupervised (Phase B): photometric loss in the learned feature space, weighted by a content
mask. The mask weighting is normalized by its own sum, so scaling the mask down does not lower
the loss; this is what stops the degenerate all-zero (M -> 0) solution.
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


def photometric_feature_loss(
    model: IHN, patch_a: torch.Tensor, patch_b: torch.Tensor, h: torch.Tensor, eps: float = 1e-3
) -> torch.Tensor:
    """Masked feature-space photometric loss between A and B warped into A's frame."""
    fa, ma = model.features_and_mask(patch_a)
    fb, mb = model.features_and_mask(patch_b)
    h_feat = model._h_feat(h)
    fb_w = homography_warp(fb, h_feat)
    mb_w = homography_warp(mb, h_feat)

    weight = ma * mb_w  # (B, 1, h, w); high only where both views are confident
    diff = (fa - fb_w).abs().mean(dim=1, keepdim=True)
    # normalization by the mask sum makes the loss invariant to the mask scale, so the mask
    # cannot collapse to zero to cheat the photometric term
    return (weight * diff).sum() / (weight.sum() + eps)
