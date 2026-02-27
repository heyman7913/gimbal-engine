import sys
import time
from pathlib import Path
from typing import Optional

import click
import cv2
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text
from rich import box

from cuda_motion_flow.stabilizer import stabilize_video
from cuda_motion_flow.cuda_kernels import check_cuda_available, get_device_info
from cuda_motion_flow.trajectory import SMOOTHERS

console = Console()
VERSION = "0.5.0"

SMOOTHER_LABELS = {
    "gaussian": "Gaussian convolution",
    "kalman":   "Kalman RTS smoother",
    "l1":       "L1 / Total-Variation",
}


def _print_banner() -> None:
    t = Text()
    t.append("⚡ cuda-motion-flow", style="bold cyan")
    t.append(f"  v{VERSION}", style="dim")
    t.append("  ·  GPU-accelerated video stabilization", style="dim white")
    console.print(Panel(t, box=box.ROUNDED, border_style="cyan", padding=(0, 2)))


def _print_device_row() -> Optional[dict]:
    if not check_cuda_available():
        console.print("[red]  CUDA not available[/]")
        return None
    info = get_device_info()
    console.print(
        f"  [dim]Device[/]  [bold cyan]{info['device_name']}[/]"
        f"  ·  cc [cyan]{info['compute_capability']}[/]"
        f"  ·  [cyan]{info['free_memory_gb']:.1f}[/][dim]/{info['total_memory_gb']:.1f} GB free[/]"
    )
    console.print()
    return info


def _print_job_panel(
    input_video: Path,
    output_video: Path,
    width: int,
    height: int,
    fps: float,
    frame_count: int,
    smoother: str,
    smoothing: float,
) -> None:
    table = Table(box=None, show_header=False, padding=(0, 2), expand=False)
    table.add_column(style="dim", min_width=10)
    table.add_column()
    table.add_row("Input",    f"[bold]{input_video.name}[/]")
    table.add_row("Output",   f"[bold]{output_video.name}[/]")
    table.add_row("Video",    f"{width}×{height}  @  {fps:.2f} fps  ·  {frame_count} frames")
    table.add_row(
        "Smoother",
        f"[cyan]{SMOOTHER_LABELS[smoother]}[/]  ·  strength [cyan]{smoothing}[/]",
    )
    console.print(Panel(table, title="[dim]job[/]", box=box.ROUNDED, border_style="dim"))
    console.print()


def _make_progress() -> Progress:
    return Progress(
        SpinnerColumn(spinner_name="dots", style="cyan"),
        TextColumn("[bold]{task.description:<22}"),
        BarColumn(bar_width=36, style="cyan", complete_style="bright_cyan"),
        MofNCompleteColumn(),
        TextColumn("[dim]{task.fields[rate]}[/]"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )


@click.command()
@click.argument("input_video",  type=click.Path(exists=True, path_type=Path))
@click.argument("output_video", type=click.Path(path_type=Path))
@click.option(
    "--smoothing", default=0.3, show_default=True,
    type=click.FloatRange(0.0, 1.0),
    help="Smoothing strength (0 = minimal, 1 = maximum).",
)
@click.option(
    "--smoother", default="gaussian", show_default=True,
    type=click.Choice(list(SMOOTHERS), case_sensitive=False),
    help=(
        "Trajectory smoothing algorithm.  "
        "gaussian: Gaussian convolution.  "
        "kalman: Rauch-Tung-Striebel optimal smoother.  "
        "l1: Total-Variation (preserves intentional pans)."
    ),
)
@click.option("--no-crop",   is_flag=True, help="Disable auto-crop of black borders.")
@click.option("--no-resize", is_flag=True, help="Keep cropped resolution (do not upscale).")
@click.option(
    "--export-trajectory", default=None, metavar="PATH",
    help=(
        "Export camera poses.  "
        ".json path for JSON output, directory for COLMAP format "
        "(cameras.txt + images.txt) for Gaussian Splatting / SfM pipelines."
    ),
)
@click.option("--verbose", "-v", is_flag=True, help="Print per-stage timing and diagnostics.")
@click.option(
    "--device-info", is_flag=True, is_eager=True, expose_value=False,
    callback=lambda ctx, _, v: (_device_info_and_exit(ctx)) if v else None,
    help="Print CUDA device info and exit.",
)
def main(
    input_video: Path,
    output_video: Path,
    smoothing: float,
    smoother: str,
    no_crop: bool,
    no_resize: bool,
    export_trajectory: str,
    verbose: bool,
) -> None:
    """
    GPU-accelerated video stabilization via optical flow motion compensation.

    All pipeline stages run on the GPU:

    \b
      Stage 1  Shi-Tomasi corner detection      (raw CUDA kernel)
      Stage 2  Pyramidal Lucas-Kanade tracking  (vectorised CuPy)
      Stage 3  RANSAC affine estimation         (all iterations in parallel)
      Stage 4  Trajectory smoothing             (Gaussian / Kalman / L1-TV)
      Stage 5  Bilinear affine warp             (raw CUDA kernel, dual streams)

    Examples:

    \b
      cuda-motion-flow shaky.mp4 stable.mp4
      cuda-motion-flow input.mp4 out.mp4 --smoother kalman --smoothing 0.6
      cuda-motion-flow input.mp4 out.mp4 --smoother l1 --export-trajectory ./poses/
    """
    _print_banner()
    _print_device_row()

    # Read video metadata up front for the job panel
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        console.print(f"[red]Error:[/] cannot open {input_video}")
        sys.exit(1)
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps         = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    _print_job_panel(input_video, output_video, width, height, fps, frame_count, smoother, smoothing)

    t_start = time.perf_counter()

    with _make_progress() as progress:
        flow_task   = progress.add_task("Optical flow",   total=frame_count - 1, rate="")
        smooth_task = progress.add_task(
            f"Trajectory ({smoother})", total=1, rate=""
        )
        warp_task   = progress.add_task("Warp frames",    total=frame_count - 1, rate="")

        _stage_start: dict = {}

        def _cb(stage: str, current: int, _total: int) -> None:
            now = time.perf_counter()
            if stage == "flow":
                task: TaskID = flow_task
                if current == 1:
                    _stage_start["flow"] = now
                elapsed = now - _stage_start.get("flow", now)
                rate = f"{current / elapsed:.0f} fps" if elapsed > 0 else ""
                progress.update(task, completed=current, rate=rate)
            elif stage == "smooth":
                if current == 0:
                    _stage_start["smooth"] = now
                    progress.update(smooth_task, rate="running…")
                else:
                    elapsed = now - _stage_start.get("smooth", now)
                    progress.update(smooth_task, completed=1, rate=f"{elapsed * 1e3:.1f} ms")
            elif stage == "warp":
                task = warp_task
                if current == 1:
                    _stage_start["warp"] = now
                elapsed = now - _stage_start.get("warp", now)
                rate = f"{current / elapsed:.0f} fps" if elapsed > 0 else ""
                progress.update(task, completed=current, rate=rate)

        try:
            stabilize_video(
                input_path=str(input_video),
                output_path=str(output_video),
                smoothing_factor=smoothing,
                smoother=smoother,
                verbose=verbose,
                auto_crop=not no_crop,
                preserve_resolution=not no_resize,
                export_trajectory=export_trajectory,
                progress_callback=_cb,
            )
        except (RuntimeError, FileNotFoundError, ValueError) as exc:
            console.print(f"\n[red]Error:[/] {exc}")
            sys.exit(1)

    elapsed = time.perf_counter() - t_start
    size_mb = output_video.stat().st_size / 1e6 if output_video.exists() else 0.0

    _print_results(output_video, elapsed, frame_count, size_mb)


def _print_results(output_video: Path, elapsed: float, frame_count: int, size_mb: float) -> None:
    console.print()
    table = Table(box=None, show_header=False, padding=(0, 2), expand=False)
    table.add_column(style="dim", min_width=14)
    table.add_column()
    table.add_row("Total time",  f"[bold cyan]{elapsed:.1f}s[/]")
    table.add_row("Throughput",  f"[cyan]{frame_count / elapsed:.1f} fps[/] end-to-end")
    table.add_row("Output",      f"[bold]{output_video.name}[/]  [dim]({size_mb:.1f} MB)[/]")
    console.print(Panel(table, title="[dim]done[/]", box=box.ROUNDED, border_style="green"))


def _device_info_and_exit(ctx: click.Context) -> None:
    _print_banner()
    if not check_cuda_available():
        console.print("[red]CUDA is not available on this system.[/]")
    else:
        info = get_device_info()
        table = Table(box=None, show_header=False, padding=(0, 2))
        table.add_column(style="dim", min_width=22)
        table.add_column(style="cyan")
        table.add_row("Device",            info["device_name"])
        table.add_row("Device ID",         str(info["device_id"]))
        table.add_row("Compute capability", info["compute_capability"])
        table.add_row("Total memory",      f"{info['total_memory_gb']:.2f} GB")
        table.add_row("Free memory",       f"{info['free_memory_gb']:.2f} GB")
        console.print(Panel(table, title="[dim]CUDA device[/]", box=box.ROUNDED, border_style="cyan"))
    ctx.exit()
