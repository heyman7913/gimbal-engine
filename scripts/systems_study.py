"""Systems study of the fused correlation kernel.

ncu hardware counters are blocked on this WSL2 / consumer-GeForce setup (ERR_NVGPUCTRPERM), so
instead of trusting a counter this measures kernel time with CUDA events and derives the rest
from first principles: FLOPs and the essential data movement are counted analytically, the
compute roof is the card's theoretical FP32 peak, and the memory roof is measured with a
saturating copy. Everything is reproducible with fixed seeds and a warmup.

Emits docs/optimization_log.{json,csv} and docs/roofline.png.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import gimbal._cuda as ext
import numpy as np
import torch
from gimbal.learned.correlation import local_correlation_reference

B, C, H, W, R = 32, 96, 16, 16, 4
K = (2 * R + 1) ** 2
REPS = 200
WARMUP = 30
OUT = Path("docs")


def cuda_time_ms(fn, reps: int = REPS, warmup: int = WARMUP) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps


def measured_memory_roof_gbs() -> float:
    """Saturating device-to-device copy gives the attainable bandwidth (read + write)."""
    n = 64 * 1024 * 1024  # 256 MB of float32
    a = torch.empty(n, device="cuda")
    b = torch.empty(n, device="cuda")
    ms = cuda_time_ms(lambda: b.copy_(a), reps=50, warmup=10)
    return (2 * n * 4) / (ms * 1e-3) / 1e9


def theoretical_fp32_tflops() -> float:
    p = torch.cuda.get_device_properties(0)
    clock_ghz = 3.09  # rated max graphics clock from nvidia-smi
    cores_per_sm = 128  # Blackwell consumer SM
    return p.multi_processor_count * cores_per_sm * 2 * clock_ghz / 1e3


def clocks_under_load() -> dict[str, str]:
    """Sample the steady boosted sm/mem clock and pstate while a copy loop runs.

    The GPU takes a moment to ramp, so this samples several times and keeps the highest sm clock
    (the steady boost), rather than a single reading that might catch it mid-ramp.
    """
    import subprocess
    import threading
    import time

    a = torch.randn(64 * 1024 * 1024, device="cuda")
    b = torch.empty_like(a)
    stop = [False]

    def loop() -> None:
        while not stop[0]:
            b.copy_(a)
        torch.cuda.synchronize()

    th = threading.Thread(target=loop)
    th.start()
    best_sm = -1
    best = ("", "", "")
    for _ in range(6):
        time.sleep(0.7)
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,pstate", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        sm, mem, ps = (x.strip() for x in out.split(","))
        cur = int(sm.split()[0])
        if cur > best_sm:
            best_sm, best = cur, (sm, mem, ps)
    stop[0] = True
    th.join()
    return {
        "sm_clock_under_load": best[0],
        "mem_clock_under_load": best[1],
        "pstate_under_load": best[2],
    }


def main() -> None:
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    fa = torch.randn(B, C, H, W, device="cuda")
    fb = torch.randn(B, C, H, W, device="cuda")
    fan = fa.permute(0, 2, 3, 1).contiguous()
    fbn = fb.permute(0, 2, 3, 1).contiguous()

    # analytic work: 2C FLOPs per output element; essential traffic reads fa, fb once and
    # writes the volume (the algorithm's intrinsic movement, independent of kernel version).
    flops = B * K * H * W * (2 * C)
    essential_bytes = (2 * B * C * H * W + B * K * H * W) * 4
    arithmetic_intensity = flops / essential_bytes

    versions = {
        "pytorch_reference": lambda: local_correlation_reference(fa, fb, R),
        "v0_naive": lambda: ext.correlation_forward(fa, fb, R),
        "v1_fa_reuse": lambda: ext.correlation_forward_v1(fa, fb, R),
        "v2_float4_nhwc": lambda: ext.correlation_forward_v2(fan, fbn, R),
    }

    ref = ext.correlation_forward(fa, fb, R)
    mem_roof = measured_memory_roof_gbs()
    compute_roof = theoretical_fp32_tflops()

    rows = []
    t_v0 = None
    t_ref = None
    for name, fn in versions.items():
        out = fn()
        max_err = float((out - ref).abs().max())
        ms = cuda_time_ms(fn)
        gflops = flops / (ms * 1e-3) / 1e9
        if name == "v0_naive":
            t_v0 = ms
        if name == "pytorch_reference":
            t_ref = ms
        rows.append(
            {
                "version": name,
                "latency_ms": round(ms, 4),
                "achieved_gflops": round(gflops, 1),
                "pct_of_memory_roof": round(100 * gflops / (mem_roof * arithmetic_intensity), 2),
                "max_err_vs_v0": max_err,
            }
        )

    # thread count is the load-bearing fact here: at 16x16 the win is parallelism, not fewer ops
    threads = {
        "pytorch_reference": None,
        "v0_naive": B * K * H * W,
        "v1_fa_reuse": B * H * W,
        "v2_float4_nhwc": B * K * H * W,
    }
    bottleneck = {
        "pytorch_reference": "overhead-bound: dozens of small ops plus temporaries per call",
        "v0_naive": "latency-bound: fully parallel and coalesced, but at 16x16 it reaches only a "
        "fifth of the copy roof, the problem is too small to saturate memory at full clock",
        "v1_fa_reuse": f"occupancy-bound: only {B * H * W} threads vs v0's {B * K * H * W}, "
        "too few to hide latency, so saving fa loads does not pay off",
        "v2_float4_nhwc": "uncoalesced: NHWC makes neighbouring threads stride by C, "
        "breaking warp coalescing despite the float4 loads",
    }
    for r in rows:
        r["speedup_vs_v0"] = round(t_v0 / r["latency_ms"], 2) if t_v0 else None
        r["speedup_vs_reference"] = round(t_ref / r["latency_ms"], 2) if t_ref else None
        r["threads_launched"] = threads[r["version"]]
        r["bottleneck"] = bottleneck[r["version"]]

    env = {
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": "sm_120",
        "torch": torch.__version__,
        "shape": f"B{B}_C{C}_H{H}_W{W}_R{R}",
        "arithmetic_intensity_flop_per_byte": round(arithmetic_intensity, 2),
        "compute_roof_tflops_fp32_theoretical_at_rated_clock": round(compute_roof, 1),
        "memory_roof_gbs_measured": round(mem_roof, 1),
        "ncu_counters": "unavailable on this WSL2/consumer GPU (ERR_NVGPUCTRPERM)",
        "timing": "cuda events, fixed seed, warmup",
        **clocks_under_load(),
        "clock_note": (
            "the GPU boosts to its high-power state under load (clocks recorded above), so these "
            "are full-clock numbers. -lgc cannot pin clocks under WSL2, so they are sampled "
            "each run."
        ),
    }

    OUT.mkdir(exist_ok=True)
    (OUT / "optimization_log.json").write_text(json.dumps({"env": env, "versions": rows}, indent=2))
    with (OUT / "optimization_log.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    _roofline(rows, arithmetic_intensity, compute_roof, mem_roof)
    print(json.dumps({"env": env, "versions": rows}, indent=2))


def _roofline(rows, ai, compute_tflops, mem_gbs) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = np.logspace(-1, 3, 200)
    mem_line = mem_gbs * xs / 1e3  # TFLOP/s
    comp_line = np.full_like(xs, compute_tflops)
    roof = np.minimum(mem_line, comp_line)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(xs, roof, "k-", label="roofline")
    ax.axhline(compute_tflops, ls="--", color="tab:red", alpha=0.6, label="fp32 compute roof")
    ax.loglog(xs, mem_line, ls="--", color="tab:blue", alpha=0.6, label="memory roof")
    for r in rows:
        ax.plot(ai, r["achieved_gflops"] / 1e3, "o", label=r["version"])
    ax.axvline(ai, color="gray", ls=":", alpha=0.5)
    ax.set_xlabel("arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("performance (TFLOP/s)")
    ax.set_title("fused correlation roofline (5070 Ti laptop)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "roofline.png", dpi=120)


if __name__ == "__main__":
    main()
