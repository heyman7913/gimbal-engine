"""Command-line interface: stabilize, train, benchmark, info."""

from __future__ import annotations

from typing import cast

import rich_click as click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .estimators.base import Estimator
from .trajectory import Smoother

click.rich_click.USE_RICH_MARKUP = True
click.rich_click.STYLE_OPTION = "cyan"
click.rich_click.STYLE_COMMAND = "bold green"

console = Console()
BANNER = "[bold cyan]cuda-motion-flow[/]  [dim]classical vs learned camera-motion, head to head[/]"


@click.group()
@click.version_option(package_name="cuda-motion-flow")
def main() -> None:
    """GPU video stabilization with two interchangeable motion estimators."""


@main.command()
def info() -> None:
    """Show the GPU, compute capability, memory, and library versions."""
    _require_gpu_stack()
    import cupy as cp
    import torch

    if not torch.cuda.is_available():
        console.print("[red]no CUDA device available[/]")
        raise SystemExit(1)
    cap = torch.cuda.get_device_capability(0)
    free, total = cp.cuda.runtime.memGetInfo()
    body = (
        f"[bold]{torch.cuda.get_device_name(0)}[/]\n"
        f"compute capability : sm_{cap[0]}{cap[1]}\n"
        f"memory             : {total / 1e9:.1f} GB ({(total - free) / 1e9:.1f} GB used)\n"
        f"torch              : {torch.__version__}\n"
        f"cupy               : {cp.__version__}\n"
        f"cuda (torch)       : {torch.version.cuda}"
    )
    console.print(Panel(body, title="device", border_style="cyan", expand=False))


@main.command()
@click.argument("input_video", type=click.Path(exists=True))
@click.argument("output_video", type=click.Path())
@click.option("--estimator", type=click.Choice(["classical", "ihn"]), default="classical")
@click.option(
    "--smoother", type=click.Choice(["gaussian", "kalman_rts", "l1_tv"]), default="kalman_rts"
)
@click.option("--strength", type=click.FloatRange(0.0, 1.0), default=0.6)
@click.option(
    "--export-trajectory", type=click.Path(), default=None, help="JSON file or COLMAP dir"
)
def stabilize(
    input_video: str,
    output_video: str,
    estimator: str,
    smoother: str,
    strength: float,
    export_trajectory: str | None,
) -> None:
    """Stabilize INPUT_VIDEO into OUTPUT_VIDEO."""
    _require_gpu_stack()
    from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

    from ._gpu import used_vram_mb
    from .pipeline import stabilize as run_stabilize
    from .video_io import read_video, write_video

    console.print(BANNER)
    est = _build_estimator(estimator)
    frames, fps = read_video(input_video)
    console.print(
        f"loaded [bold]{len(frames)}[/] frames @ {fps:.1f} fps, estimator=[bold]{estimator}[/]"
    )

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        tasks = {
            "estimate": prog.add_task("estimate motion", total=len(frames) - 1),
            "warp": prog.add_task("warp + crop", total=len(frames)),
        }

        def on_progress(stage: str, i: int, total: int) -> None:
            prog.update(tasks[stage], completed=i)

        result = run_stabilize(
            frames,
            est,
            smoother=cast(Smoother, smoother),
            strength=strength,
            progress=on_progress,
        )

    write_video(output_video, result.frames, fps)
    if export_trajectory:
        _export(result, frames[0].shape[1], frames[0].shape[0], export_trajectory)

    _metrics_table(result.metrics, f"{estimator} · {smoother}", used_vram_mb())
    console.print(f"wrote [green]{output_video}[/]")


@main.command()
@click.option("--quick", is_flag=True, help="tiny run to validate the pipeline")
@click.option(
    "--phase-b/--no-phase-b", default=True, help="run unsupervised Phase B if NUS is present"
)
def train(quick: bool, phase_b: bool) -> None:
    """Train the IHN: supervised on COCO, then guarded unsupervised on NUS."""
    _require_gpu_stack()
    from . import runner

    console.print(BANNER)
    summary = runner.train(quick=quick, with_phase_b=phase_b)
    console.print(
        f"IHN MACE [bold green]{summary['ihn_best_mace']:.2f}px[/] vs "
        f"regression [bold]{summary['regression_best_mace']:.2f}px[/] "
        f"(fused kernel: {'on' if summary['use_fused'] else 'off'})"
    )


@main.command()
def benchmark() -> None:
    """Run both estimators on the NUS subset and write the comparison."""
    _require_gpu_stack()
    from . import runner
    from .bench.harness import category_table

    console.print(BANNER)
    report = runner.benchmark()
    table = category_table(report)

    rt = Table(title="NUS stability by category", border_style="cyan")
    rt.add_column("category")
    for est in ("classical", "ihn"):
        rt.add_column(f"{est}\nstab", justify="right")
        rt.add_column(f"{est}\ncrop", justify="right")
        rt.add_column(f"{est}\nfps", justify="right")
    for cat, by_est in table.items():
        row = [cat]
        for est in ("classical", "ihn"):
            m = by_est.get(est, {})
            row += [
                f"{m.get('stability_score', 0):.3f}",
                f"{m.get('cropping_ratio', 0):.3f}",
                f"{m.get('fps', 0):.1f}",
            ]
        rt.add_row(*row)
    console.print(rt)

    mb = report.correlation_microbench
    if mb:
        console.print(
            f"fused correlation: [bold green]{mb['speedup']:.2f}x[/] faster, "
            f"[bold]{mb['memory_ratio']:.2f}x[/] less memory than the PyTorch reference"
        )
    console.print(f"wrote [green]{runner.OUT}/benchmark.json[/], benchmark.csv, the plot")


def _require_gpu_stack() -> None:
    """Fail with a readable message, not a traceback, when the GPU runtime is missing."""
    import importlib.util

    missing = [m for m in ("torch", "cupy") if importlib.util.find_spec(m) is None]
    if missing:
        raise click.ClickException(
            f"this command needs the GPU runtime ({' and '.join(missing)} not importable). "
            "install torch + cupy built for your CUDA, or use the project's Docker image."
        )


def _build_estimator(kind: str) -> Estimator:
    if kind == "classical":
        from .estimators.classical import ClassicalEstimator

        return ClassicalEstimator()
    from .estimators.learned import LearnedEstimator
    from .learned.download import ensure_weights

    return LearnedEstimator(weights_path=ensure_weights())


def _metrics_table(metrics: dict[str, float], title: str, vram_mb: float) -> None:
    table = Table(title=title, border_style="cyan", show_header=False)
    table.add_row(
        "stability (in -> out)",
        f"{metrics['stability_input']:.3f} -> {metrics['stability_score']:.3f}",
    )
    table.add_row("cropping ratio", f"{metrics['cropping_ratio']:.3f}")
    table.add_row("distortion value", f"{metrics['distortion_value']:.3f}")
    table.add_row("VRAM", f"{vram_mb:.0f} MB")
    console.print(table)


def _export(result: object, width: int, height: int, path: str) -> None:
    from pathlib import Path

    from .geometry import build_trajectory, estimate_intrinsics

    traj = build_trajectory(result.pairwise, estimate_intrinsics(width, height))  # type: ignore[attr-defined]
    if path.endswith(".json"):
        traj.export_json(path)
    else:
        traj.export_colmap(Path(path))
    console.print(f"exported trajectory to [green]{path}[/]")


if __name__ == "__main__":
    main()
