"""The head-to-head benchmark: run both estimators on the same clips and write the results out.

One thing to be careful about: GPU calls are async, so timing them naively gives nonsense. I
sync the device around the measured work. For per-call latency I sync around each call, and for
throughput I time a whole batch with a single sync at the end so the sync cost isn't counted.
"""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .._gpu import device_synchronize, used_vram_mb
from ..estimators.base import Estimator
from .datasets import Clip


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
    scope: dict[str, object] = field(default_factory=dict)


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
                "scope": report.scope,
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


def correlation_microbenchmark(
    b: int = 32, c: int = 96, h: int = 16, w: int = 16, radius: int = 4, iters: int = 50
) -> dict[str, float]:
    """Time fused vs reference local correlation (forward+backward), latency and peak memory."""
    import torch

    from ..learned.correlation import FusedLocalCorrelation, local_correlation_reference

    fa = torch.randn(b, c, h, w, device="cuda", requires_grad=True)
    fb = torch.randn(b, c, h, w, device="cuda", requires_grad=True)

    def run(fused: bool) -> tuple[float, float]:
        def step() -> None:
            fa.grad = None
            fb.grad = None
            out = (
                FusedLocalCorrelation.apply(fa, fb, radius)  # type: ignore[no-untyped-call]
                if fused
                else (local_correlation_reference(fa, fb, radius))
            )
            out.sum().backward()

        for _ in range(5):
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        end = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        start.record()
        for _ in range(iters):
            step()
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / iters
        peak = torch.cuda.max_memory_allocated() / 1e6
        return ms, peak

    ref_ms, ref_mb = run(False)
    fused_ms, fused_mb = run(True)
    return {
        "reference_ms": ref_ms,
        "reference_peak_mb": ref_mb,
        "fused_ms": fused_ms,
        "fused_peak_mb": fused_mb,
        "speedup": ref_ms / fused_ms if fused_ms > 0 else 0.0,
        "memory_ratio": ref_mb / fused_mb if fused_mb > 0 else 0.0,
    }


def environment_info() -> dict[str, str]:
    import cupy as cp
    import torch

    cap = torch.cuda.get_device_capability(0)
    return {
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": f"sm_{cap[0]}{cap[1]}",
        "torch": torch.__version__,
        "cupy": cp.__version__,
        "driver": str(torch.version.cuda),
    }


def run_benchmark(
    clips: list[Clip],
    estimators: dict[str, Estimator],
    smoother: str = "kalman_rts",
    strength: float = 0.6,
    log: Callable[[str], None] = print,
) -> BenchmarkReport:
    """Run every estimator on every clip; collect the triplet, timing, and the microbench."""
    from ..pipeline import stabilize
    from ..video_io import read_video, to_gray

    report = BenchmarkReport(environment=environment_info())
    for clip in clips:
        frames, _ = read_video(clip.path)
        grays = [to_gray(f) for f in frames]
        pairs = [(grays[i - 1], grays[i]) for i in range(1, len(grays))]
        for name, est in estimators.items():
            result = stabilize(frames, est, smoother=smoother, strength=strength)  # type: ignore[arg-type]
            timing = time_estimator(est, pairs)
            m = result.metrics
            report.results.append(
                ClipResult(
                    clip=clip.path.name,
                    category=clip.category,
                    estimator=name,
                    cropping_ratio=m["cropping_ratio"],
                    distortion_value=m["distortion_value"],
                    stability_score=m["stability_score"],
                    stability_input=m["stability_input"],
                    timing=timing,
                )
            )
            log(
                f"{clip.category:14s} {clip.path.name:20s} {name:9s} "
                f"stab {m['stability_score']:.3f} crop {m['cropping_ratio']:.3f} "
                f"dist {m['distortion_value']:.3f} fps {timing.fps:.1f}"
            )
    report.correlation_microbench = correlation_microbenchmark()
    return report


def plot_report(report: BenchmarkReport, out_path: str | Path) -> None:
    """Grouped bar chart of stability score per category, classical vs ihn."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    table = category_table(report)
    cats = list(table.keys())
    ests = sorted({e for v in table.values() for e in v})
    x = np.arange(len(cats))
    width = 0.8 / max(1, len(ests))
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, est in enumerate(ests):
        vals = [table[c].get(est, {}).get("stability_score", 0.0) for c in cats]
        ax.bar(x + i * width, vals, width, label=est)
    ax.set_xticks(x + width * (len(ests) - 1) / 2)
    ax.set_xticklabels(cats, rotation=20)
    ax.set_ylabel("stability score")
    ax.set_title("NUS stability by category: classical vs IHN")
    ax.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


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
