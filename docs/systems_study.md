# Systems study: the fused correlation kernel

All numbers measured on an RTX 5070 Ti Laptop GPU (sm_120, 46 SMs), torch 2.11.0+cu128, inside
the project's Docker image. Raw data: `optimization_log.{json,csv}`, `roofline.png`,
`cuda_graph.json`, `precision.json`.

## Profiling environment (honest limitations)

- **Nsight Compute counters are unavailable here.** `ncu` returns `ERR_NVGPUCTRPERM` even with
  `--privileged`; on WSL2 with a consumer GeForce the GPU performance counters are not exposed
  and the host driver policy can't be changed from the container. So there are no ncu
  warp-stall / DRAM-counter numbers in this study.
- **Nsight Systems is not installed in the image**; the end-to-end timeline uses `torch.profiler`
  (CUPTI), which *does* work and gives per-kernel device time and launch counts.
- **The GPU stays in P8 under load.** At 100% utilisation it sits at 180 MHz SM / 405 MHz memory
  and `nvidia-smi -lgc` cannot lock clocks under WSL2. Absolute throughput is therefore ~1/17 of
  the rated card. The measured memory roof (copy bench) reflects this throttled state, so the
  valid takeaways are the **per-version positions and relative speedups**, not absolute peaks.
- What is measured instead: CUDA-event latency, analytic FLOPs / essential bytes, achieved
  throughput = work / measured time, occupancy reasoned from thread counts, and a roofline whose
  memory roof is a measured saturating copy.

## Optimization log (forward, B32 C96 16x16 R4)

| version | latency | vs PyTorch ref | vs v0 | bottleneck |
|---|---|---|---|---|
| pytorch_reference | 10.3 ms | 1.0x | 0.12x | overhead: many small ops + temporaries |
| **v0 naive** | **1.25 ms** | **8.3x** | 1.0x | memory-bound, ~half the (throttled) copy roof, fully parallel + coalesced |
| v1 fa-reuse (k-tiling) | 11.4 ms | 0.9x | 0.11x | occupancy: 8192 threads vs v0's 663552, too few to hide latency |
| v2 float4 (NHWC) | 3.2 ms | 3.2x | 0.39x | uncoalesced: NHWC strides neighbouring threads by C |

Arithmetic intensity is ~14.2 FLOP/byte, which puts the kernel left of the roofline ridge
(memory side). The honest result is that **the hand "optimizations" are slower than the naive
kernel**: at 16x16 the volume is tiny, so what matters is launching enough threads (v0's 663k)
and keeping loads coalesced. v1 cuts redundant `fa` traffic but collapses to 8k threads and
loses latency-hiding; v2's NHWC layout enables float4 but breaks warp coalescing. Reducing
arithmetic/loads does not help a problem that is occupancy/latency bound. v0 (custom, compiled)
is already 8.3x over the PyTorch reference and sits near the memory roof.

## End-to-end: the real bottleneck is launch overhead (CUDA graphs)

Full IHN inference profiled with CUPTI: **2179 kernel launches per call, of which only 16.7% of
the wall time is GPU-busy** (6.6 ms of kernels inside 44.5 ms). Each of the 6 refinement
iterations fires many small kernels, so launch latency dominates, exactly as expected at this
scale.

Capturing the whole `predict` into a CUDA graph required a graph-safe DLT solve, because
`torch.linalg.solve` (cuSOLVER) does a host sync that invalidates capture
(`cudaErrorStreamCaptureInvalidated`). Replacing it with a functional no-pivot Gaussian
elimination on the SPD ridge system (matches cuSOLVER to 1e-4) makes the loop capturable.

- eager: **45.8 ms/call**
- graph replay: **3.68 ms/call** -> **12.5x end-to-end speedup**
- graph vs eager max error: **4e-5** (numerically equivalent; covered by a test)

This is the load-bearing finding: at this size you do not chase the kernel, you remove the
launch overhead.

## Precision (fp16 / bf16 / fp8)

| precision | MACE | delta vs fp32 | correlation latency |
|---|---|---|---|
| fp32 | 0.863 px | - | 2.30 ms |
| fp16 | 0.865 px | +0.002 | 2.34 ms |
| bf16 | 0.892 px | +0.029 | 2.44 ms |

fp16 is essentially free on accuracy (correlation relative error 7e-4) but gives **no throughput
gain** here, because the kernel is latency/memory bound rather than compute bound, so halving the
data width buys nothing it can use. bf16 costs a small amount of MACE. fp8 (`torch.float8_e4m3fn`
+ `_scaled_mm`) is available but the 16x16 cost volume is a tiny per-pair dot product with no GEMM
to amortise, so it would be marginal and was not pursued. The senior point is knowing the op is
small and bound by something other than FLOPs, not manufacturing a tensor-core win.
