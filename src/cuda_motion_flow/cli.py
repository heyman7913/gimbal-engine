import click
from pathlib import Path
from cuda_motion_flow.stabilizer import stabilize_video

@click.command()
@click.option('--smoothing', default=0.3, type=float, help='Smoothing factor for video stabilization (0.0 to 1.0)')
@click.option('--verbose', is_flag=True, help='Enable verbose output')
@click.argument('input_video', type=click.Path(exists=True, path_type=Path))
@click.argument('output_video', type=click.Path(path_type=Path))
def main(input_video: Path, output_video: Path, smoothing: float, verbose: bool):
    """
    Perform GPU-accelerated video stabilization via optical flow-based motion compensation.

    Employs CUDA-accelerated algorithms to compute inter-frame transformations,
    construct camera trajectory models, and apply temporal smoothing filters to
    eliminate jitter and unwanted motion artifacts from INPUT_VIDEO. The stabilized
    output is encoded to OUTPUT_VIDEO with preserved original resolution and codec
    parameters.

    The smoothing parameter controls the low-pass filter strength applied to the
    computed trajectory. Higher values produce more aggressive stabilization at
    the cost of increased edge cropping due to affine transformation constraints.

    Examples:

        \b
        # Process with default smoothing coefficient (0.3)
        cuda-motion-flow input.mp4 output.mp4

        \b
        # Apply aggressive stabilization with elevated smoothing coefficient
        cuda-motion-flow shaky.avi stable.avi --smoothing 0.7

        \b
        # Enable diagnostic output stream for algorithmic transparency
        cuda-motion-flow input.mp4 output.mp4 --verbose
    """
    if verbose:
        click.echo(f"Verbose mode is enabled.\nInput Video Path: {input_video}\nOutput Video Path: {output_video}\nSmoothing Factor: {smoothing}\n")

    click.echo(f"Initializing video stabilization pipeline for {input_video}...")

    try:
        stabilize_video(input_video, output_video, smoothing, verbose)
        click.echo(f"Video stabilization completed successfully. Output saved to {output_video}")
    except Exception as e:
        click.echo(f"An error occurred during stabilization: {e}", err=True)
        raise click.Abort()