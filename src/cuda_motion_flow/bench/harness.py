"""Head-to-head benchmark: run each estimator over the same clips, measure quality and cost,
and emit machine-readable result files.

Timing note: GPU kernel launches are asynchronous, so the device is synchronized before and
after every measured call. Per-call latency (p50/p95) is measured with each call individually
bracketed by a device sync; throughput (FPS) is measured over a synchronized batch so the
per-call sync overhead is excluded. Seeds and deterministic flags are set by the caller.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .._gpu import device_synchronize, used_vram_mb
from ..estimators.base import Estimator


@dataclass
class Timing:
    p50_ms: float
    p95_ms: float
    fps: float
    peak_vram_mb: float


@dataclass
class ClipResult:
    clip: str
    category: str
    estimator: str
    cropping_ratio: float
    distortion_value: float
    stability_score: float
    stability_input: float
    timing: Timing


@dataclass
class BenchmarkReport:
    results: list[ClipResult] = field(default_factory=list)
    environment: dict[str, str] = field(default_factory=dict)
    correlation_microbench: dict[str, float] = field(default_factory=dict)


def time_estimator(
    estimator: Estimator, gray_pairs: list[tuple[np.ndarray, np.ndarray]], warmup: int = 3
) -> Timing:
    """Measure per-call latency and batch throughput for one estimator over frame pairs."""
    if gray_pairs:
        h, w = gray_pairs[0][0].shape
        estimator.warmup(h, w)
    for i in range(min(warmup, len(gray_pairs))):
        estimator.estimate(*gray_pairs[i])
    device_synchronize()

    latencies: list[float] = []
    for a, b in gray_pairs:
        device_synchronize()
        t0 = time.perf_counter()
        estimator.estimate(a, b)
        device_synchronize()
        latencies.append((time.perf_counter() - t0) * 1e3)

    device_synchronize()
    t0 = time.perf_counter()
    for a, b in gray_pairs:
        estimator.estimate(a, b)
    device_synchronize()
    batch_s = time.perf_counter() - t0

    lat = np.asarray(latencies)
    fps = len(gray_pairs) / batch_s if batch_s > 0 else 0.0
    return Timing(
        p50_ms=float(np.percentile(lat, 50)),
        p95_ms=float(np.percentile(lat, 95)),
        fps=float(fps),
        peak_vram_mb=float(used_vram_mb()),
    )


def write_report(report: BenchmarkReport, out_dir: str | Path) -> None:
    """Write the report as JSON and a flat CSV under out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with (out / "benchmark.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "environment": report.environment,
                "correlation_microbench": report.correlation_microbench,
                "results": [_flatten(r) for r in report.results],
            },
            f,
            indent=2,
        )

    if report.results:
        rows = [_flatten(r) for r in report.results]
        with (out / "benchmark.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def _flatten(r: ClipResult) -> dict[str, object]:
    d = asdict(r)
    timing = d.pop("timing")
    d.update({f"timing_{k}": v for k, v in timing.items()})
    return d


def category_table(report: BenchmarkReport) -> dict[str, dict[str, dict[str, float]]]:
    """Aggregate the triplet by (category, estimator) as means for the rich table / plot."""
    agg: dict[str, dict[str, list[ClipResult]]] = {}
    for r in report.results:
        agg.setdefault(r.category, {}).setdefault(r.estimator, []).append(r)
    table: dict[str, dict[str, dict[str, float]]] = {}
    for cat, by_est in agg.items():
        table[cat] = {}
        for est, items in by_est.items():
            table[cat][est] = {
                "cropping_ratio": float(np.mean([i.cropping_ratio for i in items])),
                "distortion_value": float(np.mean([i.distortion_value for i in items])),
                "stability_score": float(np.mean([i.stability_score for i in items])),
                "fps": float(np.mean([i.timing.fps for i in items])),
            }
    return table
