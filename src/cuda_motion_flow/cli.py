import sys
import time
from pathlib import Path
from typing import Optional

import rich_click as click
import cv2
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich import box

from cuda_motion_flow.stabilizer import stabilize_video
from cuda_motion_flow.cuda_kernels import check_cuda_available, get_device_info
from cuda_motion_flow.trajectory import SMOOTHERS

# ── rich-click help styling ────────────────────────────────────────────────────
click.rich_click.USE_RICH_MARKUP       = True
click.rich_click.SHOW_ARGUMENTS        = True
click.rich_click.MAX_WIDTH             = 100
click.rich_click.STYLE_OPTION         = "bold cyan"
click.rich_click.STYLE_ARGUMENT       = "bold bright_cyan"
click.rich_click.STYLE_METAVAR        = "dim"
click.rich_click.STYLE_USAGE          = "bold"
click.rich_click.STYLE_USAGE_COMMAND  = "bold cyan"
click.rich_click.STYLE_HELPTEXT_FIRST_LINE = "bold"
click.rich_click.STYLE_OPTIONS_PANEL_BORDER   = "cyan"
click.rich_click.STYLE_OPTIONS_PANEL_BOX      = "ROUNDED"
click.rich_click.STYLE_COMMANDS_PANEL_BORDER  = "dim cyan"
click.rich_click.HEADER_TEXT = (
    "[bold bright_cyan]cuda-motion-flow[/]  [dim cyan]v0.5.0[/]"
    "  [dim]·  GPU-accelerated video stabilization[/]"
)
click.rich_click.FOOTER_TEXT = (
    "[dim]Examples:[/]\n\n"
    "  [bold]cuda-motion-flow[/] shaky.mp4 stable.mp4\n"
    "  [bold]cuda-motion-flow[/] input.mp4 out.mp4 "
    "[cyan]--smoother kalman --smoothing 0.6[/]\n"
    "  [bold]cuda-motion-flow[/] input.mp4 out.mp4 "
    "[cyan]--smoother l1 --export-trajectory ./poses/[/]"
)
click.rich_click.OPTION_GROUPS = {
    "cuda-motion-flow": [
        {
            "name": "Smoothing",
            "options": ["--smoother", "--smoothing"],
        },
        {
            "name": "Output",
            "options": ["--no-crop", "--no-resize", "--export-trajectory"],
        },
        {
            "name": "Diagnostics",
            "options": ["--verbose", "--device-info", "--help"],
        },
    ]
}

console = Console()
VERSION = "0.5.0"

SMOOTHER_LABELS = {
    "gaussian": "Gaussian convolution",
    "kalman":   "Kalman RTS smoother",
    "l1":       "L1 / Total-Variation",
}

SMOOTHER_COLORS = {
    "gaussian": "cyan",
    "kalman":   "bright_cyan",
    "l1":       "yellow",
}


# ── header ────────────────────────────────────────────────────────────────────

def _print_header() -> None:
    console.print()
    console.print(Rule(style="bright_cyan"))

    title = Text()
    title.append("  cuda-motion-flow", style="bold bright_cyan")
    title.append(f"  v{VERSION}", style="dim cyan")
    title.append(
        "    GPU-accelerated video stabilization  ",
        style="dim white",
    )
    console.print(title)
    console.print(Rule(style="bright_cyan"))
    console.print()


def _print_device() -> Optional[dict]:
    if not check_cuda_available():
        console.print("  [red]CUDA not available[/]")
        console.print()
        return None

    info = get_device_info()

    # VRAM bar — 24 chars wide
    used  = info["total_memory_gb"] - info["free_memory_gb"]
    frac  = used / max(info["total_memory_gb"], 1e-6)
    bar_w = 24
    filled = int(frac * bar_w)
    bar = "█" * filled + "░" * (bar_w - filled)

    console.print(
        f"  [dim]Device[/]  [bold cyan]{info['device_name']}[/]"
        f"  [dim]·[/]  cc [cyan]{info['compute_capability']}[/]"
    )
    console.print(
        f"  [dim]VRAM  [/]  [cyan]{bar}[/]"
        f"  [dim]{used:.1f} / {info['total_memory_gb']:.1f} GB[/]"
        f"  [dim]({frac * 100:.0f}%)[/]"
    )
    console.print()
    return info


def _print_job(
    input_video: Path,
    output_video: Path,
    width: int, height: int,
    fps: float,
    frame_count: int,
    smoother: str,
    smoothing: float,
) -> None:
    color = SMOOTHER_COLORS[smoother]
    table = Table(box=None, show_header=False, padding=(0, 2), expand=False)
    table.add_column(style="dim", min_width=10)
    table.add_column()
    table.add_row("Input",    f"[bold]{input_video.name}[/]")
    table.add_row("Output",   f"[bold]{output_video.name}[/]")
    table.add_row("Video",    f"{width}×{height}  @  {fps:.2f} fps  ·  {frame_count} frames")
    table.add_row(
        "Smoother",
        f"[{color}]{SMOOTHER_LABELS[smoother]}[/]  "
        f"[dim]·  strength[/] [{color}]{smoothing}[/]",
    )
    console.print(Panel(table, title="[dim]job[/]", box=box.ROUNDED, border_style="dim cyan"))
    console.print()


def _make_progress() -> Progress:
    return Progress(
        SpinnerColumn(spinner_name="dots2", style="cyan", finished_text="[green]+[/]"),
        TextColumn("[bold]{task.description:<24}"),
        BarColumn(bar_width=32, style="dim cyan", complete_style="cyan", finished_style="bright_cyan"),
        MofNCompleteColumn(),
        TextColumn("  [dim]{task.fields[rate]:<12}[/]"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )


def _print_results(
    output_video: Path,
    frame_count: int,
    t_total: float,
    stage_times: dict,
    smoother: str,
) -> None:
    color = SMOOTHER_COLORS[smoother]
    size_mb = output_video.stat().st_size / 1e6 if output_video.exists() else 0.0

    console.print()
    console.print(Rule(style="dim cyan"))
    console.print()

    t = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="dim", expand=False)
    t.add_column("Stage",      style="bold", min_width=24)
    t.add_column("Time",       justify="right", min_width=10)
    t.add_column("Frames",     justify="right", min_width=8)
    t.add_column("Throughput", justify="right", min_width=10)

    for stage_key, label in [
        ("flow",   "Optical flow"),
        ("smooth", f"Trajectory ({smoother})"),
        ("warp",   "Warp frames"),
    ]:
        elapsed = stage_times.get(stage_key, 0.0)
        if stage_key == "smooth":
            t.add_row(
                f"[green]+[/]  {label}",
                f"[{color}]{elapsed * 1e3:.1f} ms[/]" if elapsed < 1 else f"[{color}]{elapsed:.2f}s[/]",
                "[dim]—[/]",
                "[dim]—[/]",
            )
        else:
            fps_stage = frame_count / elapsed if elapsed > 0 else 0
            t.add_row(
                f"[green]+[/]  {label}",
                f"[{color}]{elapsed:.2f}s[/]",
                f"[dim]{frame_count}[/]",
                f"[{color}]{fps_stage:.1f} fps[/]",
            )

    t.add_section()
    t.add_row(
        "[bold bright_cyan]End-to-end[/]",
        f"[bold bright_cyan]{t_total:.2f}s[/]",
        f"[dim]{frame_count}[/]",
        f"[bold bright_cyan]{frame_count / t_total:.1f} fps[/]",
    )

    console.print(t)
    console.print(
        f"  [dim]Output[/]  [bold]{output_video.name}[/]"
        f"  [dim]·  {size_mb:.1f} MB[/]"
    )
    console.print()


# ── CLI entry point ────────────────────────────────────────────────────────────

@click.command("cuda-motion-flow")
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
    help="Trajectory smoothing: gaussian | kalman | l1",
)
@click.option("--no-crop",   is_flag=True, help="Disable auto-crop of black borders.")
@click.option("--no-resize", is_flag=True, help="Keep cropped resolution (do not upscale).")
@click.option(
    "--export-trajectory", default=None, metavar="PATH",
    help=".json for JSON poses, directory for COLMAP output (Gaussian Splatting / SfM).",
)
@click.option("--verbose", "-v", is_flag=True, help="Per-stage timing and diagnostics.")
@click.option(
    "--device-info", is_flag=True, is_eager=True, expose_value=False,
    callback=lambda ctx, _, v: (_device_info_cmd(ctx)) if v else None,
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

    \b
    Pipeline stages (all GPU):
      1  Shi-Tomasi corner detection    raw CUDA kernel
      2  Pyramidal Lucas-Kanade         vectorised CuPy
      3  RANSAC affine estimation       all iterations in parallel
      4  Trajectory smoothing           Gaussian / Kalman-RTS / L1-TV
      5  Bilinear affine warp           raw CUDA kernel, dual streams

    \b
    Examples:
      cuda-motion-flow input.mp4 out.mp4
      cuda-motion-flow input.mp4 out.mp4 --smoother kalman --smoothing 0.6
      cuda-motion-flow input.mp4 out.mp4 --smoother l1 --export-trajectory ./poses/
    """
    _print_header()
    _print_device()

    # Read metadata before handing off to stabilizer
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        console.print(f"[red]error:[/] cannot open {input_video}")
        sys.exit(1)
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps         = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    _print_job(input_video, output_video, width, height, fps, frame_count, smoother, smoothing)

    t_start     = time.perf_counter()
    stage_times: dict = {}
    stage_start: dict = {}

    with _make_progress() as progress:
        flow_task = progress.add_task(
            "Optical flow",
            total=frame_count - 1,
            rate="",
        )
        smooth_task = progress.add_task(
            f"Trajectory ({smoother})",
            total=1,
            rate="",
        )
        warp_task = progress.add_task(
            "Warp frames",
            total=frame_count - 1,
            rate="",
        )

        def _cb(stage: str, current: int, _total: int) -> None:
            now = time.perf_counter()

            if stage == "flow":
                if current == 1:
                    stage_start["flow"] = now
                elapsed = now - stage_start.get("flow", now)
                rate = f"{current / elapsed:.0f} fps" if elapsed > 0 else ""
                progress.update(flow_task, completed=current, rate=rate)
                if current == _total:
                    stage_times["flow"] = elapsed

            elif stage == "smooth":
                if current == 0:
                    stage_start["smooth"] = now
                    progress.update(smooth_task, rate="running…")
                else:
                    elapsed = now - stage_start.get("smooth", now)
                    stage_times["smooth"] = elapsed
                    progress.update(
                        smooth_task, completed=1,
                        rate=f"{elapsed * 1e3:.1f} ms" if elapsed < 1 else f"{elapsed:.2f}s",
                    )

            elif stage == "warp":
                if current == 1:
                    stage_start["warp"] = now
                elapsed = now - stage_start.get("warp", now)
                rate = f"{current / elapsed:.0f} fps" if elapsed > 0 else ""
                progress.update(warp_task, completed=current, rate=rate)
                if current == _total:
                    stage_times["warp"] = elapsed

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
            console.print(f"\n[red]error:[/] {exc}")
            sys.exit(1)

    t_total = time.perf_counter() - t_start
    _print_results(output_video, frame_count, t_total, stage_times, smoother)


# ── device-info subcommand ─────────────────────────────────────────────────────

def _device_info_cmd(ctx: click.Context) -> None:
    _print_header()

    if not check_cuda_available():
        console.print("  [red]CUDA not available on this system.[/]")
        ctx.exit()
        return

    info = get_device_info()
    used  = info["total_memory_gb"] - info["free_memory_gb"]
    frac  = used / max(info["total_memory_gb"], 1e-6)
    bar_w = 28
    bar   = "█" * int(frac * bar_w) + "░" * (bar_w - int(frac * bar_w))

    t = Table(box=box.SIMPLE_HEAVY, show_header=False, padding=(0, 2))
    t.add_column(style="dim",        min_width=22)
    t.add_column(style="bold cyan")
    t.add_row("Device",             info["device_name"])
    t.add_row("Device ID",          str(info["device_id"]))
    t.add_row("Compute capability", info["compute_capability"])
    t.add_row("Total memory",       f"{info['total_memory_gb']:.2f} GB")
    t.add_row("Free memory",        f"{info['free_memory_gb']:.2f} GB")
    t.add_row("VRAM usage",         f"[cyan]{bar}[/]  {frac * 100:.0f}%")

    console.print(Panel(t, title="[dim]CUDA device[/]", box=box.ROUNDED, border_style="cyan"))
    ctx.exit()
