"""Quick global-vs-mesh sanity check on synthetic data with ground truth.

Trains MeshIHN briefly on the synthetic parallax task, then checks two things:
  1. on parallax motion the mesh beats the single global homography (lower MACE),
  2. on global motion the mesh reduces to the global model (cells agree, similar MACE).

This is the foundation demonstration; the full training + smoothness sweep comes later. Writes
perf_study/mesh_sanity.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from gimbal.learned.data import (
    SyntheticPairGenerator,
    SyntheticParallaxGenerator,
    load_image_pool,
)
from gimbal.learned.losses import mace, mesh_corner_loss, mesh_smoothness
from gimbal.learned.model import MeshIHN

OUT = Path("perf_study")


def _bcast(off_global: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return off_global[:, None, None].expand_as(like)


def main(steps: int = 600, smooth_w: float = 0.1) -> None:
    torch.manual_seed(0)
    pool = load_image_pool("data/coco", size=(240, 320), limit=500)
    par = SyntheticParallaxGenerator(pool, patch=128, grid=(8, 8), seed=0)
    par_val = SyntheticParallaxGenerator(pool, patch=128, grid=(8, 8), seed=999).sample(64)
    glob_val = SyntheticPairGenerator(pool, patch=128, rho=24, seed=1).sample(64)

    model = MeshIHN(size=128, grid=(8, 8), iters=6).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for step in range(1, steps + 1):
        pa, pb, gt = par.sample(32)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            offsets = model(pa, pb)
            loss = mesh_corner_loss(offsets, gt) + smooth_w * mesh_smoothness(offsets)
        loss.backward()
        opt.step()
        if step % 150 == 0:
            print(f"step {step}  loss {float(loss):.4f}")

    model.eval()
    with torch.no_grad():
        pa, pb, gt = par_val
        mesh_off = model(pa, pb)
        glob = model.global_ihn(pa, pb)[-1]
        par_mesh = float(mace(mesh_off, gt))
        par_global = float(mace(_bcast(glob, gt), gt))

        pa, pb, delta = glob_val
        gt_g = delta[:, None, None].expand(delta.shape[0], 8, 8, 4, 2)
        mesh_off_g = model(pa, pb)
        glob_g = model.global_ihn(pa, pb)[-1]
        glob_mesh = float(mace(mesh_off_g, gt_g))
        glob_global = float(mace(_bcast(glob_g, gt_g), gt_g))
        cell_std = float(mesh_off_g.std(dim=(1, 2)).mean())

    result = {
        "steps": steps,
        "smoothness_weight": smooth_w,
        "parallax_set": {
            "mace_global_px": round(par_global, 3),
            "mace_mesh_px": round(par_mesh, 3),
            "mesh_better_by_px": round(par_global - par_mesh, 3),
        },
        "global_motion_set": {
            "mace_global_px": round(glob_global, 3),
            "mace_mesh_px": round(glob_mesh, 3),
            "cell_offset_std_px": round(cell_std, 3),
        },
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "mesh_sanity.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
