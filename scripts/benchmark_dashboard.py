"""Render the NUS head-to-head benchmark into presentation-grade figures.

Reads outputs/benchmark.json (written by `gimbal benchmark`) and writes the tracked benchmark/
deliverables: the raw json and csv, a four-panel dashboard, a standalone quality-vs-speed figure,
and a README with the per-category table. Run after the benchmark finishes.
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

SRC = Path("outputs/benchmark.json")
SRC_CSV = Path("outputs/benchmark.csv")
OUT = Path("benchmark")

# canonical NUS order; only the categories actually present are drawn
CATEGORY_ORDER = ("Regular", "QuickRotation", "Zooming", "Parallax", "Crowd", "Running")
ESTS = ("classical", "ihn")
LABELS = {"classical": "Classical  (CUDA LK + RANSAC)", "ihn": "Learned  (IHN)"}
COLORS = {"classical": "#5C6B73", "ihn": "#00B2A9"}
INPUT_COLOR = "#C44536"


def _style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#3A3A3A",
            "axes.linewidth": 1.1,
            "axes.axisbelow": True,
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#E3E3E3",
            "grid.linewidth": 0.9,
            "font.size": 11,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.color": "#3A3A3A",
            "ytick.color": "#3A3A3A",
            "legend.frameon": False,
            "figure.dpi": 200,
        }
    )


def _aggregate(results: list[dict], metric: str) -> dict[str, dict[str, tuple[float, float]]]:
    """{estimator: {category: (mean, std)}} for one metric over its clips."""
    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        buckets[r["estimator"]][r["category"]].append(r[metric])
    return {
        est: {c: (float(np.mean(v)), float(np.std(v))) for c, v in cats.items()}
        for est, cats in buckets.items()
    }


def _ordered_categories(results: list[dict]) -> list[str]:
    present = {r["category"] for r in results}
    return [c for c in CATEGORY_ORDER if c in present]


def _short(cat: str) -> str:
    return {"QuickRotation": "Quick\nRotation"}.get(cat, cat)


def _grouped_bars(ax, cats, data, ylabel, title, ylim=None) -> None:
    x = np.arange(len(cats))
    width = 0.38
    for i, est in enumerate(ESTS):
        means = [data[est][c][0] for c in cats]
        stds = [data[est][c][1] for c in cats]
        ax.bar(
            x + (i - 0.5) * width,
            means,
            width,
            yerr=stds,
            capsize=3,
            color=COLORS[est],
            edgecolor="white",
            linewidth=0.6,
            error_kw={"elinewidth": 1, "ecolor": "#888888"},
            label=LABELS[est],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([_short(c) for c in cats], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    if ylim:
        ax.set_ylim(*ylim)


def _draw_scatter(ax, scatter, overall) -> None:
    for est in ESTS:
        xs = [p[0] for p in scatter[est]]
        ys = [p[1] for p in scatter[est]]
        ax.scatter(
            xs, ys, s=70, color=COLORS[est], alpha=0.8, edgecolor="white", linewidth=1, zorder=3
        )
        ax.scatter(
            overall[est][0],
            overall[est][1],
            s=420,
            marker="*",
            color=COLORS[est],
            edgecolor="white",
            linewidth=1.4,
            zorder=4,
            label=f"{LABELS[est]}  (mean)",
        )
    ax.annotate(
        "better",
        xy=(0.97, 0.95),
        xytext=(0.72, 0.78),
        xycoords="axes fraction",
        fontsize=10,
        color="#555555",
        ha="center",
        arrowprops={"arrowstyle": "-|>", "color": "#999999", "lw": 1.6},
    )
    ax.set_xlabel("throughput  (frames / second)")
    ax.set_ylabel("stability score")
    ax.set_title("Quality vs speed  (each point = one scene category)", loc="left")


def _draw_margin(ax, cats, stab) -> None:
    margins = [stab["ihn"][c][0] - stab["classical"][c][0] for c in cats]
    y = np.arange(len(cats))[::-1]
    colors = [COLORS["ihn"] if m >= 0 else COLORS["classical"] for m in margins]
    ax.barh(y, margins, color=colors, edgecolor="white", height=0.62)
    ax.axvline(0, color="#3A3A3A", linewidth=1.1)
    ax.set_yticks(y)
    ax.set_yticklabels(list(cats), fontsize=9)
    lim = max(abs(m) for m in margins) * 1.45
    ax.set_xlim(-lim, lim)
    for yi, m in zip(y, margins, strict=True):
        ax.text(
            m + (lim * 0.03 if m >= 0 else -lim * 0.03),
            yi,
            f"{m:+.3f}",
            va="center",
            ha="left" if m >= 0 else "right",
            fontsize=8.5,
            color="#444444",
        )
    ax.text(lim * 0.96, len(cats) - 0.4, "IHN better", ha="right", fontsize=9, color=COLORS["ihn"])
    ax.text(
        -lim * 0.96, len(cats) - 0.4, "classical better", ha="left", fontsize=9, color="#3A3A3A"
    )
    ax.set_xlabel("stability margin  (IHN minus classical)")
    ax.set_title("Where each method wins", loc="left")
    ax.grid(axis="x", alpha=0.3)


def render(data: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style()
    results = data["results"]
    cats = _ordered_categories(results)
    stab = _aggregate(results, "stability_score")
    fps = _aggregate(results, "timing_fps")
    crop = _aggregate(results, "cropping_ratio")
    dist = _aggregate(results, "distortion_value")

    inp = {c: np.mean([r["stability_input"] for r in results if r["category"] == c]) for c in cats}
    scatter = {
        est: [(fps[est][c][0], stab[est][c][0]) for c in cats if c in fps[est]] for est in ESTS
    }
    overall = {
        est: (
            float(np.mean([fps[est][c][0] for c in cats])),
            float(np.mean([stab[est][c][0] for c in cats])),
        )
        for est in ESTS
    }

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(
        "gimbal  ·  classical vs learned (IHN) video stabilization on NUS",
        fontsize=18,
        fontweight="bold",
        x=0.5,
        y=0.99,
    )
    env = data.get("environment", {})
    mb = data.get("correlation_microbench", {})
    n_clips = data.get("scope", {}).get("clips", "")
    subtitle = (
        f"{env.get('gpu', '')}  ·  torch {env.get('torch', '')}  ·  "
        f"{len(cats)} categories, {n_clips} clips  ·  higher is better on every axis"
    )
    fig.text(0.5, 0.945, subtitle, ha="center", fontsize=11, color="#666666")

    ax_a = fig.add_subplot(2, 2, 1)
    _grouped_bars(ax_a, cats, stab, "stability score", "Stabilization quality by scene category")
    ax_a.plot(
        np.arange(len(cats)),
        [inp[c] for c in cats],
        "D",
        color=INPUT_COLOR,
        markersize=7,
        markeredgecolor="white",
        label="shaky input",
        zorder=5,
        linestyle="none",
    )
    ax_a.legend(loc="lower right", fontsize=9)

    ax_b = fig.add_subplot(2, 2, 2)
    _grouped_bars(ax_b, cats, fps, "frames / second", "Throughput by scene category")
    ax_b.legend(loc="upper right", fontsize=9)

    ax_c = fig.add_subplot(2, 2, 3)
    _draw_scatter(ax_c, scatter, overall)
    ax_c.legend(loc="lower left", fontsize=9)

    ax_d = fig.add_subplot(2, 2, 4)
    _draw_margin(ax_d, cats, stab)

    footer = data.get("scope", {}).get("note", "")
    if mb:
        footer += (
            f"   |   fused correlation kernel: {mb['speedup']:.1f}x faster, "
            f"{mb['memory_ratio']:.2f}x less memory than the PyTorch reference"
        )
    fig.text(0.5, 0.012, footer, ha="center", fontsize=8.5, color="#999999", wrap=True)
    fig.tight_layout(rect=(0, 0.04, 1, 0.90))
    fig.savefig(OUT / "dashboard.png", bbox_inches="tight")
    plt.close(fig)

    fig2, ax = plt.subplots(figsize=(9, 6))
    _draw_scatter(ax, scatter, overall)
    for est in ESTS:
        for c in cats:
            ax.annotate(
                c,
                (fps[est][c][0], stab[est][c][0]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=7.5,
                color="#777777",
            )
    ax.set_title("Quality vs speed across the six NUS categories", loc="left")
    ax.legend(loc="lower left", fontsize=9)
    fig2.tight_layout()
    fig2.savefig(OUT / "quality_vs_speed.png", bbox_inches="tight")
    plt.close(fig2)

    _write_readme(data, cats, stab, crop, dist, fps, overall)


def _write_readme(data, cats, stab, crop, dist, fps, overall) -> None:
    env = data.get("environment", {})
    mb = data.get("correlation_microbench", {})
    lines = [
        "# NUS stabilization benchmark",
        "",
        "Classical (CUDA Lucas-Kanade + RANSAC homography) vs the learned IHN estimator, run "
        "head to head on the NUS benchmark clips. Higher is better on every metric.",
        "",
        "![dashboard](dashboard.png)",
        "",
        "![quality vs speed](quality_vs_speed.png)",
        "",
        f"- GPU: {env.get('gpu', '')}, torch {env.get('torch', '')}",
        f"- Scope: {data.get('scope', {}).get('note', '')}",
    ]
    if mb:
        lines.append(
            f"- Fused correlation kernel: {mb['speedup']:.1f}x faster and "
            f"{mb['memory_ratio']:.2f}x lighter than the PyTorch reference"
        )
    lines += [
        "",
        "## Per-category means",
        "",
        "| category | estimator | stability | cropping | distortion | fps |",
        "|---|---|---|---|---|---|",
    ]
    for c in cats:
        for est in ESTS:
            lines.append(
                f"| {c} | {est} | {stab[est][c][0]:.3f} | {crop[est][c][0]:.3f} | "
                f"{dist[est][c][0]:.3f} | {fps[est][c][0]:.1f} |"
            )
    lines += [
        "",
        "## Overall (mean over categories)",
        "",
        "| estimator | stability | fps |",
        "|---|---|---|",
    ]
    for est in ESTS:
        lines.append(f"| {est} | {overall[est][1]:.3f} | {overall[est][0]:.1f} |")
    lines.append("")
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"{SRC} not found; run `gimbal benchmark` first")
    OUT.mkdir(parents=True, exist_ok=True)
    data = json.loads(SRC.read_text())
    shutil.copy(SRC, OUT / "benchmark.json")
    if SRC_CSV.exists():
        shutil.copy(SRC_CSV, OUT / "benchmark.csv")
    render(data)
    print(f"wrote {OUT}/dashboard.png, quality_vs_speed.png, README.md, benchmark.json, csv")


if __name__ == "__main__":
    main()
