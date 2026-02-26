import sys
import time

import click
from pathlib import Path

from cuda_motion_flow.stabilizer import stabilize_video
from cuda_motion_flow.cuda_kernels import check_cuda_available, get_device_info
from cuda_motion_flow.trajectory import SMOOTHERS


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
        "gaussian: fast GPU Gaussian convolution.  "
        "kalman: Rauch-Tung-Striebel optimal smoother (best for mixed motion).  "
        "l1: Total-Variation smoothing (preserves intentional pans)."
    ),
)
@click.option("--no-crop",   is_flag=True, help="Disable auto-crop of black borders.")
@click.option("--no-resize", is_flag=True, help="Keep cropped resolution (do not upscale).")
@click.option(
    "--export-trajectory", default=None, metavar="PATH",
    help=(
        "Export recovered camera poses.  "
        "Pass a .json path for JSON output, or a directory for COLMAP format "
        "(cameras.txt + images.txt), which can be fed directly into "
        "Gaussian Splatting or SfM pipelines."
    ),
)
@click.option("--verbose", "-v", is_flag=True, help="Print per-stage timing and diagnostics.")
@click.option("--device-info", is_flag=True, is_eager=True, expose_value=False,
              callback=lambda ctx, _p, v: (_print_device_info(), ctx.exit()) if v else None,
              help="Print CUDA device info and exit.")
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
      # Basic stabilization
      cuda-motion-flow shaky.mp4 stable.mp4

    \b
      # Kalman smoother, stronger smoothing, export COLMAP trajectory
      cuda-motion-flow input.mp4 output.mp4 --smoother kalman --smoothing 0.6 \\
          --export-trajectory ./colmap_poses/

    \b
      # L1 / Total-Variation smoother (preserves intentional pans)
      cuda-motion-flow vlog.mp4 vlog_stable.mp4 --smoother l1 --smoothing 0.4

    \b
      # Verbose diagnostics
      cuda-motion-flow input.mp4 output.mp4 --verbose
    """
    if verbose:
        click.echo(f"Input     : {input_video}")
        click.echo(f"Output    : {output_video}")
        click.echo(f"Smoother  : {smoother}  (strength={smoothing})")
        click.echo(f"Auto-crop : {'off' if no_crop else 'on'}")
        click.echo()

    t_start = time.perf_counter()

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
        )
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    elapsed = time.perf_counter() - t_start
    click.echo(f"Completed in {elapsed:.1f}s  ->  {output_video}")


def _print_device_info() -> None:
    if not check_cuda_available():
        click.echo("CUDA is not available on this system.")
        return
    info = get_device_info()
    click.echo(f"Device {info['device_id']}  |  "
               f"compute capability {info['compute_capability']}  |  "
               f"{info['total_memory_gb']:.1f} GB total  |  "
               f"{info['free_memory_gb']:.1f} GB free")
