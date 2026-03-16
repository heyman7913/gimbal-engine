"""End-to-end study: is IHN inference launch-bound, and does a CUDA graph fix it?

Each refinement iteration fires several small kernels (encoder convs, correlation, the DLT
solve, grid_sample), so at this size the GPU is likely idle between launches. This measures the
eager latency, breaks it down with the profiler, then captures the whole predict into a CUDA
graph and replays it. Writes perf_study/cuda_graph.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from gimbal.learned.model import IHN

OUT = Path("perf_study")


def cuda_time_ms(fn, reps: int = 100, warmup: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps


def kernel_vs_total(model, a, b) -> dict[str, float]:
    from torch.profiler import ProfilerActivity, profile

    with torch.no_grad(), profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        for _ in range(20):
            model.predict(a, b)
        torch.cuda.synchronize()
    ka = p.key_averages()
    cuda_us = sum(e.self_device_time_total for e in ka)
    launches = sum(e.count for e in ka if e.key == "cudaLaunchKernel")
    return {
        "cuda_kernel_ms_per_call": cuda_us / 1e3 / 20,
        "kernel_launches_per_call": launches / 20,
    }


def main() -> None:
    torch.manual_seed(0)
    model = IHN(size=128, iters=6).cuda().eval()
    a = torch.rand(1, 1, 128, 128, device="cuda")
    b = torch.rand(1, 1, 128, 128, device="cuda")

    # ground-truth answer with the default cuSOLVER solve, then switch to the capturable solve
    with torch.no_grad():
        eager_ref = model.predict(a, b).clone()
    model.graph_safe = True

    eager_ms = cuda_time_ms(lambda: model.predict(a, b))
    breakdown = kernel_vs_total(model, a, b)

    result: dict[str, object] = {
        "eager_ms_per_call": round(eager_ms, 4),
        **{k: round(v, 4) for k, v in breakdown.items()},
        "gpu_busy_fraction": round(breakdown["cuda_kernel_ms_per_call"] / eager_ms, 3),
    }

    # capture the whole predict into a graph
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(5):
                model.predict(a, b)
        torch.cuda.current_stream().wait_stream(side)

        graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(graph):
            static_out = model.predict(a, b)

        def replay() -> None:
            graph.replay()

        for _ in range(20):
            replay()
        torch.cuda.synchronize()
        graph_ms = cuda_time_ms(replay)
        graph_err = float((static_out - eager_ref).abs().max())
        result["graph_ms_per_call"] = round(graph_ms, 4)
        result["graph_speedup"] = round(eager_ms / graph_ms, 2)
        result["graph_vs_eager_max_err"] = graph_err
        result["captured"] = True
    except Exception as exc:  # noqa: BLE001
        result["captured"] = False
        result["capture_error"] = f"{type(exc).__name__}: {exc}"[:300]

    OUT.mkdir(exist_ok=True)
    (OUT / "cuda_graph.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
