"""Ties the whole training run together: check the fused kernel, train both the IHN and the
regression baseline on COCO (Phase A), fine-tune on NUS with the guard (Phase B), and save the
weights it decides to keep.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..bench.datasets import Clip
from ..pipeline import stabilize
from ..video_io import read_video
from .correlation import fused_passes_gradcheck
from .data import SyntheticPairGenerator, load_image_pool, video_frame_pairs
from .model import IHN, RegressionHomographyNet
from .train import PhaseAConfig, PhaseBConfig, evaluate_mace, phase_b_with_guard, train_supervised


def build_pairs(clips: list[Clip], patch: int) -> torch.Tensor:
    raw = video_frame_pairs([c.path for c in clips], patch=patch)
    if not raw:
        return torch.empty(0, 2, patch, patch)
    arr = np.stack([np.stack([a, b]) for a, b in raw]).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


def stability_metric(model: IHN, clips: list[Clip], log: Callable[[str], None]) -> float:
    """Mean stabilized-path stability over held-out clips, using the in-training model."""
    from ..estimators.learned import LearnedEstimator

    # wrap the model we're training without going through __init__ (it would reload weights)
    est = LearnedEstimator.__new__(LearnedEstimator)
    est.name = "ihn"
    est.size = model.size
    est.model = model
    was_training = model.training
    model.eval()
    scores = []
    for clip in clips:
        frames, _ = read_video(clip.path)
        if len(frames) < 8:
            continue
        result = stabilize(frames, est, smoother="kalman_rts", strength=0.6)
        scores.append(result.metrics["stability_score"])
    if was_training:
        model.train()
    return float(np.mean(scores)) if scores else 0.0


def run_two_phase(
    coco_dir: str | Path,
    phaseb_train: list[Clip],
    guard_val: list[Clip],
    out_weights: str | Path,
    phase_a: PhaseAConfig,
    phase_a_reg: PhaseAConfig,
    phase_b: PhaseBConfig,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    use_fused = fused_passes_gradcheck()
    log("fused correlation gate: " + ("PASS (using fused)" if use_fused else "FAIL (reference)"))

    pool = load_image_pool(coco_dir, size=(240, 320), limit=5000)
    n_val = max(1, pool.shape[0] // 10)
    train_pool, val_pool = pool[:-n_val], pool[-n_val:]
    gen = SyntheticPairGenerator(train_pool, patch=128, rho=32, seed=0)
    val_gen = SyntheticPairGenerator(val_pool, patch=128, rho=32, seed=999)
    val = val_gen.sample(256)

    log("Phase A: iterative IHN")
    ihn = IHN(size=128, iters=6).cuda()
    ihn.use_fused = use_fused
    hist_ihn = train_supervised(ihn, gen, val, phase_a)

    log("Phase A: regression baseline")
    reg = RegressionHomographyNet(size=128).cuda()
    hist_reg = train_supervised(reg, gen, val, phase_a_reg)
    reg_mace = evaluate_mace(reg, val)

    log("Phase B: unsupervised photometric fine-tuning (guarded)")
    pairs = build_pairs(phaseb_train, patch=128)
    result_b: dict[str, Any]
    if pairs.shape[0] > 0 and guard_val:
        result_b = phase_b_with_guard(
            ihn, pairs, lambda m: stability_metric(m, guard_val, log), phase_b
        )
    else:
        result_b = {"adopted": False, "note": "no phase-b data"}

    Path(out_weights).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ihn.state_dict(), str(out_weights))
    log(f"saved weights to {out_weights}")

    return {
        "use_fused": use_fused,
        "ihn_best_mace": hist_ihn.best_mace,
        "regression_best_mace": min(hist_reg.val_mace) if hist_reg.val_mace else reg_mace,
        "ihn_history": {"steps": hist_ihn.steps, "val_mace": hist_ihn.val_mace},
        "regression_history": {"steps": hist_reg.steps, "val_mace": hist_reg.val_mace},
        "phase_b": result_b,
        "weights": str(out_weights),
    }
