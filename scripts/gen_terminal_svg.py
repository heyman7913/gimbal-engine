"""Render the gimbal CLI output to a terminal capture for the README.

Uses the real rich components from the CLI and real measured numbers from the benchmark, so the
image matches what a stabilize run prints: the command, the device panel, the progress, and the
result triplet. Writes media/cli.svg, which is then rendered to a PNG for a font stable image:

    rsvg-convert media/cli.svg -o media/cli.png
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table

CLIP = Path("data/nus/QuickRotation/0.avi")
OUT = Path("media/cli.svg")


def main() -> None:
    data = json.loads(Path("benchmark/benchmark.json").read_text())
    env = data["environment"]
    row = next(
        r
        for r in data["results"]
        if r["category"] == "QuickRotation" and r["clip"] == "0.avi" and r["estimator"] == "ihn"
    )

    cap = cv2.VideoCapture(str(CLIP))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    console = Console(record=True, width=82, highlight=False)
    console.print(
        "[bold #2ec4b6]$[/] [white]gimbal stabilize quickrotation.avi stabilized.mp4 "
        "--estimator ihn[/]"
    )
    console.print()
    console.print("[bold cyan]gimbal[/]   [dim]classical vs learned camera-motion, head to head[/]")
    console.print(
        Panel(
            f"[bold]{env['gpu']}[/]\n"
            f"compute capability : {env['compute_capability']}\n"
            f"torch              : {env['torch']}\n"
            f"cupy               : {env['cupy']}",
            title="device",
            border_style="cyan",
            expand=False,
        )
    )
    console.print(f"loaded {n_frames} frames @ {fps:.1f} fps, estimator ihn")

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=26, complete_style="#2ec4b6", finished_style="#2ec4b6"),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    )
    progress.add_task("estimate motion", total=n_frames - 1, completed=n_frames - 1)
    progress.add_task("warp and crop", total=n_frames, completed=n_frames)
    console.print(progress.get_renderable())

    table = Table(title="ihn · kalman_rts", border_style="cyan", show_header=False)
    table.add_row(
        "stability (in -> out)",
        f"{row['stability_input']:.3f} -> [bold #2ec4b6]{row['stability_score']:.3f}[/]",
    )
    table.add_row("cropping ratio", f"{row['cropping_ratio']:.3f}")
    table.add_row("distortion value", f"{row['distortion_value']:.3f}")
    table.add_row("throughput", f"{row['timing_fps']:.1f} fps")
    table.add_row("VRAM", f"{row['timing_peak_vram_mb']:.0f} MB")
    console.print(table)
    console.print("wrote [green]stabilized.mp4[/]")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    console.save_svg(str(OUT), title="gimbal stabilize")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
