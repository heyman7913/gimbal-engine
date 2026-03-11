"""Shared train / phase-B / benchmark orchestration used by both the CLI and the dev script."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from .bench.datasets import Clip, coco_root, nus_clips
from .bench.harness import BenchmarkReport, plot_report, run_benchmark, write_report
from .estimators.classical import ClassicalEstimator
from .estimators.learned import LearnedEstimator
from .learned.driver import build_pairs, run_two_phase, stability_metric
from .learned.model import IHN
from .learned.train import PhaseAConfig, PhaseBConfig, phase_b_with_guard

WEIGHTS = Path("weights/ihn.pt")
OUT = Path("outputs")


def splits() -> tuple[list[Clip], list[Clip], list[Clip]]:
    """Benchmark / Phase-B-train / Phase-B-guard clips, all disjoint."""
    clips = nus_clips(per_category=6)
    benchmark = [c for c in clips if c.split == "benchmark"]
    phaseb_all = [c for c in clips if c.split == "phaseb"]
    by_cat: dict[str, list[Clip]] = defaultdict(list)
    for c in phaseb_all:
        by_cat[c.category].append(c)
    guard, phaseb_train = [], []
    for cat_clips in by_cat.values():
        guard.append(cat_clips[0])
        phaseb_train.extend(cat_clips[1:])
    return benchmark, phaseb_train, guard


def configs(quick: bool) -> tuple[PhaseAConfig, PhaseAConfig, PhaseBConfig]:
    if quick:
        a = PhaseAConfig(
            steps=300, batch=24, eval_every=100, patience=100, target_mace=0.0, time_budget_s=600
        )
        return a, a, PhaseBConfig(steps=60, batch=12)
    a = PhaseAConfig(
        steps=40000, batch=128, lr=4e-4, eval_every=500, patience=10, time_budget_s=9000
    )
    reg = PhaseAConfig(
        steps=40000, batch=128, lr=4e-4, eval_every=500, patience=10, time_budget_s=5400
    )
    return a, reg, PhaseBConfig(steps=1200, batch=24)


def train(quick: bool = False, with_phase_b: bool = False) -> dict[str, Any]:
    benchmark, phaseb_train, guard = splits()
    if not with_phase_b:
        phaseb_train, guard = [], []
    a, reg, b = configs(quick)
    summary = run_two_phase(coco_root(), phaseb_train, guard, WEIGHTS, a, reg, b)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "training_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def phaseb(quick: bool = False) -> dict[str, Any]:
    _, phaseb_train, guard = splits()
    if not phaseb_train or not guard:
        return {"adopted": False, "note": "no NUS phaseb/guard clips available"}
    _, _, b = configs(quick)
    model = IHN(size=128, iters=6).cuda()
    model.load_state_dict(torch.load(str(WEIGHTS), map_location="cuda"))
    pairs = build_pairs(phaseb_train, patch=128)
    result = phase_b_with_guard(model, pairs, lambda m: stability_metric(m, guard, print), b)
    torch.save(model.state_dict(), str(WEIGHTS))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "phase_b_result.json").write_text(json.dumps(result, indent=2))
    return result


def benchmark() -> BenchmarkReport:
    torch.manual_seed(0)
    bench_clips, _, _ = splits()
    estimators = {
        "classical": ClassicalEstimator(),
        "ihn": LearnedEstimator(weights_path=WEIGHTS if WEIGHTS.exists() else None),
    }
    report = run_benchmark(bench_clips, estimators)
    write_report(report, OUT)
    plot_report(report, OUT / "stability_by_category.png")
    return report
