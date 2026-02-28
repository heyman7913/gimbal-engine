"""
Video stabilization quality analysis.

Compares an original (unstabilized) video against one or more stabilized
outputs across five metric categories drawn from published stabilization
literature:

  Stability    — inter-frame motion statistics (Grundmann et al. CVPR 2011)
  Smoothness   — velocity / jerk of motion trajectory (Liu et al. 2013)
  Frequency    — PSD analysis of motion signal; high-freq = residual jitter
  Visual       — temporal SSIM, Laplacian sharpness
  Fidelity     — SSIM and PSNR vs original (distortion introduced by warp)

GPU acceleration:
  - Motion estimation via our own GPU LK optical flow (same pipeline as stabilizer)
  - SSIM via CuPy + cupyx gaussian filter (direct convolution, no cuFFT)
  - Laplacian sharpness via cupyx.scipy.ndimage.laplace
  Falls back to CPU (OpenCV) if CUDA is unavailable.

Usage:
    python compare_videos.py test.mp4 out_gaussian.mp4 out_kalman.mp4 out_l1.mp4
    python compare_videos.py original.mp4 stabilized.mp4 --labels orig stab
    python compare_videos.py original.mp4 out.mp4 --no-fidelity --cpu
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from rich.console import Console
from rich.rule import Rule
from rich.table import Table
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich import box

console = Console()

# ── GPU backend setup ─────────────────────────────────────────────────────────

_CUDA = False
try:
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter as _cp_gauss
    from cupyx.scipy.ndimage import laplace as _cp_laplace
    from cuda_motion_flow.cuda_kernels import compute_optical_flow_gpu, check_cuda_available
    _CUDA = check_cuda_available()
except Exception:
    pass

# CPU fallback params for Farneback dense flow
_FARNEBACK = dict(
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
)
_FLOW_WIDTH = 640   # downscale width for CPU Farneback only


def _use_gpu(force_cpu: bool) -> bool:
    return _CUDA and not force_cpu


# ── Per-frame metric functions ────────────────────────────────────────────────

def _motion_gpu(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """Mean inter-frame feature displacement via GPU Lucas-Kanade."""
    prev_pts, curr_pts = compute_optical_flow_gpu(prev_gray, curr_gray)
    if len(curr_pts) == 0:
        return 0.0
    d = curr_pts.astype(np.float32) - prev_pts.astype(np.float32)
    return float(np.mean(np.sqrt(d[:, 0]**2 + d[:, 1]**2)))


def _motion_cpu(prev_small: np.ndarray, curr_small: np.ndarray) -> float:
    """Mean inter-frame motion via CPU Farneback dense flow."""
    flow = cv2.calcOpticalFlowFarneback(prev_small, curr_small, None, **_FARNEBACK)
    return float(np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)))


def _resize_cpu(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    s = _FLOW_WIDTH / w
    if s >= 1.0:
        return frame
    return cv2.resize(frame, (_FLOW_WIDTH, int(h * s)), interpolation=cv2.INTER_AREA)


def _ssim_gpu(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM via CuPy gaussian filter (direct convolution — no cuFFT required)."""
    C1, C2 = 6.5025, 58.5225
    ag = cp.asarray(a, dtype=cp.float64)
    bg = cp.asarray(b, dtype=cp.float64)
    sigma = 1.5
    mu1 = _cp_gauss(ag, sigma)
    mu2 = _cp_gauss(bg, sigma)
    s1  = _cp_gauss(ag * ag, sigma) - mu1 ** 2
    s2  = _cp_gauss(bg * bg, sigma) - mu2 ** 2
    s12 = _cp_gauss(ag * bg, sigma) - mu1 * mu2
    num = (2 * mu1 * mu2 + C1) * (2 * s12 + C2)
    den = (mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2)
    return float(cp.mean(num / den))


def _ssim_cpu(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM via OpenCV gaussian filter."""
    C1, C2 = 6.5025, 58.5225
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    k = cv2.getGaussianKernel(11, 1.5)
    win = k @ k.T
    mu1 = cv2.filter2D(a, -1, win)
    mu2 = cv2.filter2D(b, -1, win)
    s1  = cv2.filter2D(a * a, -1, win) - mu1 ** 2
    s2  = cv2.filter2D(b * b, -1, win) - mu2 ** 2
    s12 = cv2.filter2D(a * b, -1, win) - mu1 * mu2
    num = (2 * mu1 * mu2 + C1) * (2 * s12 + C2)
    den = (mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2)
    return float(np.mean(num / den))


def _sharpness_gpu(gray: np.ndarray) -> float:
    g = cp.asarray(gray, dtype=cp.float64)
    return float(_cp_laplace(g).var())


def _sharpness_cpu(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ── Signal processing ─────────────────────────────────────────────────────────

def _frequency_analysis(signal: np.ndarray, fps: float) -> Tuple[float, float, float]:
    """
    High/low frequency power ratio and spectral centroid via windowed periodogram.
    Threshold: fps/4 — above is residual jitter, below is intentional motion.
    """
    n = len(signal)
    if n < 8:
        return 0.5, 0.5, 0.0
    sig = (signal - signal.mean()) * np.hanning(n)
    psd = np.abs(np.fft.rfft(sig)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    total = psd.sum() + 1e-12
    thresh = fps / 4.0
    lf = psd[freqs < thresh].sum()
    hf = psd[freqs >= thresh].sum()
    centroid = float((freqs * psd).sum() / total)
    return float(hf / total), float(lf / total), centroid


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class VideoMetrics:
    label: str
    path: str
    frames: int
    width: int
    height: int
    fps: float

    motion_mean:      float = 0.0
    motion_std:       float = 0.0
    motion_p95:       float = 0.0
    motion_max:       float = 0.0
    stability_score:  float = 0.0   # 1/(1+σ), higher = better

    velocity_std:     float = 0.0   # std of Δmotion, lower = better
    jerk_std:         float = 0.0   # std of Δ²motion, lower = better

    hf_ratio:         float = 0.0   # high-freq power ratio, lower = better
    lf_ratio:         float = 0.0   # low-freq power ratio, higher = better
    spectral_centroid: float = 0.0  # Hz, lower = better

    temporal_ssim:    float = 0.0   # mean SSIM(frame_i, frame_i+1), higher = better
    sharpness_mean:   float = 0.0   # mean Laplacian variance, higher = better
    sharpness_std:    float = 0.0   # std of sharpness, lower = better

    ssim_vs_orig:     Optional[float] = None
    psnr_vs_orig:     Optional[float] = None

    motion_signal:    np.ndarray = field(default_factory=lambda: np.array([]))


# ── Core analysis ─────────────────────────────────────────────────────────────

def analyze_video(
    path: str,
    label: str,
    orig_grays: Optional[List[np.ndarray]],
    use_gpu: bool,
    progress: Optional[Progress] = None,
    task_id=None,
) -> Optional["VideoMetrics"]:

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        console.print(f"  [red]cannot open:[/] {path}")
        return None

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    motion_vals: List[float] = []
    t_ssim_vals: List[float] = []
    sharp_vals:  List[float] = []
    ssim_orig:   List[float] = []
    psnr_orig:   List[float] = []

    prev_gray:  Optional[np.ndarray] = None
    prev_small: Optional[np.ndarray] = None   # CPU path only
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Motion ────────────────────────────────────────────────────────────
        if prev_gray is not None:
            if use_gpu:
                motion_vals.append(_motion_gpu(prev_gray, gray))
            else:
                small = _resize_cpu(gray)
                motion_vals.append(_motion_cpu(prev_small, small))
                prev_small = small
        elif not use_gpu:
            prev_small = _resize_cpu(gray)

        # ── Temporal SSIM ─────────────────────────────────────────────────────
        if prev_gray is not None:
            fn = _ssim_gpu if use_gpu else _ssim_cpu
            t_ssim_vals.append(fn(prev_gray.astype(float), gray.astype(float)))

        # ── Sharpness ─────────────────────────────────────────────────────────
        fn_s = _sharpness_gpu if use_gpu else _sharpness_cpu
        sharp_vals.append(fn_s(gray))

        # ── Fidelity vs original ──────────────────────────────────────────────
        if orig_grays is not None and idx < len(orig_grays):
            og = orig_grays[idx]
            if og.shape != (height, width):
                og = cv2.resize(og, (width, height), interpolation=cv2.INTER_AREA)
            fn_q = _ssim_gpu if use_gpu else _ssim_cpu
            ssim_orig.append(fn_q(og.astype(float), gray.astype(float)))
            mse = np.mean((og.astype(np.float64) - gray.astype(np.float64)) ** 2)
            psnr_orig.append(10 * np.log10(255**2 / mse) if mse > 0 else 100.0)

        prev_gray = gray
        idx += 1

        if progress is not None and task_id is not None:
            progress.update(task_id, completed=idx)

    cap.release()

    if not motion_vals:
        console.print(f"  [red]no frames:[/] {path}")
        return None

    motion = np.array(motion_vals)
    hf, lf, centroid = _frequency_analysis(motion, fps)
    d1, d2 = np.diff(motion), np.diff(np.diff(motion))

    m = VideoMetrics(
        label=label, path=path,
        frames=total, width=width, height=height, fps=fps,
        motion_mean    = float(motion.mean()),
        motion_std     = float(motion.std()),
        motion_p95     = float(np.percentile(motion, 95)),
        motion_max     = float(motion.max()),
        stability_score= 1.0 / (1.0 + float(motion.std())),
        velocity_std   = float(d1.std()),
        jerk_std       = float(d2.std()),
        hf_ratio       = hf,
        lf_ratio       = lf,
        spectral_centroid = centroid,
        temporal_ssim  = float(np.mean(t_ssim_vals)) if t_ssim_vals else 0.0,
        sharpness_mean = float(np.mean(sharp_vals)),
        sharpness_std  = float(np.std(sharp_vals)),
        motion_signal  = motion,
    )
    if ssim_orig:
        m.ssim_vs_orig = float(np.mean(ssim_orig))
    if psnr_orig:
        m.psnr_vs_orig = float(np.mean(psnr_orig))
    return m


# ── Rich output ───────────────────────────────────────────────────────────────

def _best_idx(vals: List, lower: bool) -> int:
    valid = [(i, v) for i, v in enumerate(vals) if v is not None]
    if not valid:
        return -1
    return min(valid, key=lambda x: x[1] if lower else -x[1])[0]


def _print_section(
    title: str,
    metrics: List[VideoMetrics],
    rows: List[Tuple],
) -> None:
    t = Table(
        title=f"[bold]{title}[/]", box=box.SIMPLE_HEAVY,
        show_header=True, header_style="dim", title_justify="left",
    )
    t.add_column("Metric", style="dim", min_width=30)
    for m in metrics:
        t.add_column(m.label, justify="right", min_width=12)
    t.add_column("Best", style="dim cyan", justify="right", min_width=10)

    for name, attr, lower, dec, unit in rows:
        vals = [getattr(m, attr) for m in metrics]
        if all(v is None for v in vals):
            continue
        bi = _best_idx(vals, lower)
        cells = []
        for i, v in enumerate(vals):
            if v is None:
                cells.append("[dim]n/a[/]")
            else:
                s = f"{v:.{dec}f}{unit}"
                cells.append(f"[bold bright_cyan]{s}[/]" if i == bi else s)
        t.add_row(name, *cells, metrics[bi].label if bi >= 0 else "")

    console.print(t)
    console.print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare video stabilization quality.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python compare_videos.py test.mp4 out_gaussian.mp4 out_kalman.mp4 out_l1.mp4\n"
            "  python compare_videos.py original.mp4 stabilized.mp4\n\n"
            "The first video is the reference (unstabilized original)."
        ),
    )
    parser.add_argument("videos", nargs="+", metavar="VIDEO")
    parser.add_argument("--labels", nargs="+", metavar="LABEL",
                        help="Display labels (default: filename stems)")
    parser.add_argument("--no-fidelity", action="store_true",
                        help="Skip SSIM/PSNR vs original (faster)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU mode (OpenCV Farneback flow)")
    args = parser.parse_args()

    paths  = args.videos
    labels = args.labels or [Path(p).stem for p in paths]

    if len(labels) != len(paths):
        console.print("[red]error:[/] --labels count must match video count")
        sys.exit(1)
    for p in paths:
        if not Path(p).exists():
            console.print(f"[red]error:[/] not found: {p}")
            sys.exit(1)

    gpu = _use_gpu(args.cpu)

    console.print()
    console.print(
        "[bold bright_cyan]cuda-motion-flow[/]"
        "  [dim]·  stabilization quality analysis[/]"
    )
    console.print(Rule(style="bright_cyan"))
    mode_str = "[cyan]GPU[/]  (LK optical flow + CuPy SSIM/Laplacian)" if gpu else "[dim]CPU[/]  (Farneback dense flow)"
    console.print(f"  mode   {mode_str}")
    console.print()

    # Load reference frames for fidelity metrics
    orig_grays: Optional[List[np.ndarray]] = None
    if not args.no_fidelity and len(paths) > 1:
        console.print(f"  [dim]loading reference:[/] {labels[0]}")
        cap = cv2.VideoCapture(paths[0])
        orig_grays = []
        while True:
            ret, f = cap.read()
            if not ret:
                break
            orig_grays.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        cap.release()
        console.print(f"  [dim]{len(orig_grays)} frames loaded[/]")
        console.print()

    frame_counts = []
    for p in paths:
        cap = cv2.VideoCapture(p)
        frame_counts.append(max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
        cap.release()

    results: List[VideoMetrics] = []
    with Progress(
        SpinnerColumn(spinner_name="dots2", style="cyan", finished_text="[green]+[/]"),
        TextColumn("[bold]{task.description:<28}"),
        BarColumn(bar_width=30, style="dim cyan", complete_style="cyan"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        for path, label, total in zip(paths, labels, frame_counts):
            task = progress.add_task(label, total=total)
            ref = orig_grays if (not args.no_fidelity and path != paths[0]) else None
            m = analyze_video(path, label, orig_grays=ref, use_gpu=gpu,
                              progress=progress, task_id=task)
            if m is None:
                sys.exit(1)
            results.append(m)

    console.print()
    console.print(Rule(style="dim cyan"))
    console.print()

    _print_section("Stability  —  inter-frame motion", results, [
        ("Mean motion",           "motion_mean",      True,  3, " px"),
        ("Motion std dev",        "motion_std",        True,  3, " px"),
        ("95th percentile",       "motion_p95",        True,  3, " px"),
        ("Peak motion",           "motion_max",        True,  3, " px"),
        ("Stability score",       "stability_score",   False, 4, ""),
    ])

    _print_section("Trajectory smoothness  —  derivative analysis", results, [
        ("Velocity std (|Δm|)",   "velocity_std",      True,  4, " px/f"),
        ("Jerk std (|Δ²m|)",      "jerk_std",          True,  4, " px/f²"),
    ])

    _print_section("Frequency  —  motion power spectrum  (threshold: fps/4)", results, [
        ("High-freq ratio",       "hf_ratio",           True,  4, ""),
        ("Low-freq ratio",        "lf_ratio",           False, 4, ""),
        ("Spectral centroid",     "spectral_centroid",   True,  3, " Hz"),
    ])

    _print_section("Visual quality", results, [
        ("Temporal SSIM",         "temporal_ssim",     False, 4, ""),
        ("Mean sharpness",        "sharpness_mean",    False, 1, ""),
        ("Sharpness consistency", "sharpness_std",      True,  1, ""),
    ])

    fidelity = [m for m in results if m.ssim_vs_orig is not None]
    if fidelity:
        _print_section("Fidelity vs original  —  distortion from warp", fidelity, [
            ("Mean SSIM vs original", "ssim_vs_orig",   False, 4, ""),
            ("Mean PSNR vs original", "psnr_vs_orig",   False, 2, " dB"),
        ])

    info = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="dim",
                 title="[bold]Video info[/]", title_justify="left")
    info.add_column("", style="dim")
    for m in results:
        info.add_column(m.label, justify="right")
    info.add_row("Resolution", *[f"{m.width}×{m.height}" for m in results])
    info.add_row("FPS",        *[f"{m.fps:.2f}"          for m in results])
    info.add_row("Frames",     *[str(m.frames)           for m in results])
    console.print(info)
    console.print()

    console.print("[dim]Stability score   = 1/(1+σ_motion)  ·  higher is better[/]")
    console.print("[dim]High-freq ratio   = motion power above fps/4  ·  lower = less jitter[/]")
    console.print("[dim]Temporal SSIM     = SSIM between consecutive frames  ·  higher = smoother[/]")
    console.print("[dim]Jerk              = std of second derivative of motion signal[/]")
    console.print("[dim]GPU motion metric = mean LK feature displacement (sparse)[/]" if gpu else
                  "[dim]CPU motion metric = mean Farneback dense flow magnitude[/]")
    console.print()


if __name__ == "__main__":
    main()
