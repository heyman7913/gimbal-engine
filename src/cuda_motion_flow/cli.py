import click
from pathlib import Path
from cuda_motion_flow.stabilizer import stabilize_video
from cuda_motion_flow.cuda_kernels import check_cuda_available, get_device_info

@click.command()
@click.option('--smoothing', default=0.3, type=float, help='Smoothing factor for video stabilization (0.0 to 1.0)')
@click.option('--verbose', is_flag=True, help='Enable verbose output')
@click.option('--no-crop', is_flag=True, help='Disable auto-crop of black borders')
@click.option('--no-resize', is_flag=True, help='Keep cropped resolution instead of resizing to original')
@click.argument('input_video', type=click.Path(exists=True, path_type=Path))
@click.argument('output_video', type=click.Path(path_type=Path))
def main(input_video: Path, output_video: Path, smoothing: float, verbose: bool, no_crop: bool, no_resize: bool):
    """
    Perform GPU-accelerated video stabilization via optical flow-based motion compensation.

    Employs CUDA-accelerated algorithms for ALL pipeline stages:
    - Sparse optical flow computation (CuPy GPU)
    - RANSAC transform estimation (CuPy GPU)
    - Trajectory smoothing (CuPy CUDA)
    - Frame warping (CuPy CUDA)

    The smoothing parameter controls the low-pass filter strength applied to the
    computed trajectory. Higher values produce more aggressive stabilization at
    the cost of increased edge cropping due to affine transformation constraints.

    Features:
    - Full GPU pipeline using CuPy CUDA (no OpenCV CUDA needed)
    - Shi-Tomasi corner detection on GPU
    - Pyramidal Lucas-Kanade optical flow on GPU
    - GPU RANSAC affine transform estimation
    - Gaussian trajectory smoothing (CUDA)
    - GPU affine warping with bilinear interpolation
    - Auto-crop to remove black borders

    Examples:

        \b
        # Full GPU acceleration (default)
        cuda-motion-flow input.mp4 output.mp4

        \b
        # Apply aggressive stabilization with elevated smoothing coefficient
        cuda-motion-flow shaky.avi stable.avi --smoothing 0.7

        \b
        # Enable diagnostic output stream for algorithmic transparency
        cuda-motion-flow input.mp4 output.mp4 --verbose
    """
    if verbose:
        click.echo(f"Verbose mode is enabled.")
        click.echo(f"Input Video Path: {input_video}")
        click.echo(f"Output Video Path: {output_video}")
        click.echo(f"Smoothing Factor: {smoothing}")
        click.echo(f"Auto-crop: {'disabled' if no_crop else 'enabled'}")
        click.echo(f"Preserve resolution: {'no' if no_resize else 'yes'}")
        click.echo()

    click.echo(f"Initializing video stabilization pipeline for {input_video}...")

    try:
        stabilize_video(
            input_path=input_video,
            output_path=output_video,
            smoothing_factor=smoothing,
            verbose=verbose,
            auto_crop=not no_crop,
            preserve_resolution=not no_resize,
        )
        click.echo(f"Video stabilization completed successfully. Output saved to {output_video}")
    except Exception as e:
        click.echo(f"An error occurred during stabilization: {e}", err=True)
        raise click.Abort()