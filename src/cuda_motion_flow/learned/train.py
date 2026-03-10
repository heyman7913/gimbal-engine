"""Two-phase training.

Phase A (supervised): drive synthetic-validation MACE to a low plateau; early-stop on the
metric rather than a fixed step count. Phase B (unsupervised): photometric fine-tuning on real
video, adopted only if it measurably improves a held-out stabilization metric, otherwise the
Phase-A weights are kept. Both outcomes are returned so the value of Phase B is a measurement.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from .data import SyntheticPairGenerator
from .losses import mace, photometric_feature_loss, supervised_corner_loss
from .model import IHN


@dataclass
class PhaseAConfig:
    steps: int = 60000
    batch: int = 48
    lr: float = 2.5e-4
    weight_decay: float = 1e-4
    eval_every: int = 500
    patience: int = 12  # evals without MACE improvement before stopping
    target_mace: float = 3.0  # stop early once below this
    time_budget_s: float = 4.0 * 3600.0
    log: Callable[[str], None] = print


@dataclass
class PhaseBConfig:
    steps: int = 1500
    batch: int = 24
    lr: float = 5e-5
    log: Callable[[str], None] = print


@dataclass
class History:
    steps: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    val_mace: list[float] = field(default_factory=list)
    best_mace: float = float("inf")


def _autocast() -> torch.autocast:
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


@torch.no_grad()
def evaluate_mace(
    model: torch.nn.Module, val: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
) -> float:
    model.eval()
    pa, pb, delta = val
    with _autocast():
        offsets = model(pa, pb)
    return float(mace(offsets[-1].float(), delta))


def train_supervised(
    model: torch.nn.Module,
    gen: SyntheticPairGenerator,
    val: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    cfg: PhaseAConfig,
) -> History:
    """Supervised corner-loss training with early stop on validation MACE."""
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg.lr, total_steps=cfg.steps, pct_start=0.05)
    history = History()
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    start = time.perf_counter()

    for step in range(1, cfg.steps + 1):
        model.train()
        pa, pb, delta = gen.sample(cfg.batch)
        opt.zero_grad(set_to_none=True)
        with _autocast():
            offsets = model(pa, pb)
            loss = supervised_corner_loss(offsets, delta)
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % cfg.eval_every == 0 or step == cfg.steps:
            loss_val = float(loss.item())
            val_mace = evaluate_mace(model, val)
            history.steps.append(step)
            history.train_loss.append(loss_val)
            history.val_mace.append(val_mace)
            elapsed = time.perf_counter() - start
            cfg.log(
                f"step {step:6d}  loss {loss_val:.4f}  val_mace {val_mace:.3f}px  {elapsed:.0f}s"
            )
            if val_mace < history.best_mace:
                history.best_mace = val_mace
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            if history.best_mace < cfg.target_mace and stale >= cfg.patience:
                cfg.log(f"early stop: MACE plateaued at {history.best_mace:.3f}px")
                break
            if stale >= cfg.patience * 2 or elapsed > cfg.time_budget_s:
                cfg.log(f"stopping (stale={stale}, elapsed={elapsed:.0f}s)")
                break

    model.load_state_dict(best_state)
    return history


def train_unsupervised(
    model: IHN, pairs: torch.Tensor, cfg: PhaseBConfig, seed: int = 0
) -> list[float]:
    """Photometric fine-tuning on real consecutive-frame pairs. pairs: (N, 2, P, P) float."""
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    gen = torch.Generator().manual_seed(seed)
    n = pairs.shape[0]
    losses: list[float] = []
    for step in range(1, cfg.steps + 1):
        idx = torch.randint(0, n, (cfg.batch,), generator=gen)
        batch = pairs[idx].to("cuda", non_blocking=True)
        pa, pb = batch[:, 0:1], batch[:, 1:2]
        model.train()
        opt.zero_grad(set_to_none=True)
        with _autocast():
            h = model.predict(pa, pb)
            loss = photometric_feature_loss(model, pa, pb, h)
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        loss_val = float(loss.item())
        losses.append(loss_val)
        if step % 100 == 0 or step == cfg.steps:
            cfg.log(f"phaseB step {step:5d}  photo_loss {loss_val:.4f}")
    return losses


def phase_b_with_guard(
    model: IHN,
    pairs: torch.Tensor,
    eval_metric: Callable[[IHN], float],
    cfg: PhaseBConfig,
) -> dict[str, float | bool]:
    """Run Phase B but keep it only if a held-out metric (higher is better) improves."""
    baseline = eval_metric(model)
    before_state = copy.deepcopy(model.state_dict())
    train_unsupervised(model, pairs, cfg)
    after = eval_metric(model)
    adopted = after > baseline
    if not adopted:
        model.load_state_dict(before_state)
        cfg.log(f"Phase B not adopted (held-out {after:.4f} <= {baseline:.4f}); kept Phase A")
    else:
        cfg.log(f"Phase B adopted (held-out {baseline:.4f} -> {after:.4f})")
    return {"adopted": adopted, "baseline": baseline, "after": after}
