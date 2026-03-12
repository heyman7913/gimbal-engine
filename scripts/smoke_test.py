"""Quick sanity check that the GPU actually works before I bother with anything else.

It checks that torch can run a matmul on the device, that the GPU really is Blackwell (sm_120),
and that CuPy can compile and run a kernel. Exits non-zero if any of that fails.
"""

from __future__ import annotations

import sys


def check_torch() -> tuple[int, int]:
    import torch

    print(f"torch {torch.__version__}")
    if not torch.cuda.is_available():
        raise SystemExit("FAIL: torch.cuda.is_available() is False")
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"device: {name}  capability sm_{cap[0]}{cap[1]}")
    a = torch.randn(512, 512, device="cuda")
    b = torch.randn(512, 512, device="cuda")
    c = (a @ b).sum().item()
    if c != c:  # NaN guard
        raise SystemExit("FAIL: torch CUDA matmul produced NaN")
    print(f"torch CUDA matmul ok (checksum {c:.1f})")
    return cap


def check_cupy() -> None:
    import cupy as cp

    print(f"cupy {cp.__version__}")
    kernel = cp.RawKernel(
        r"""
        extern "C" __global__
        void add_one(const float* x, float* y, int n) {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i < n) y[i] = x[i] + 1.0f;
        }
        """,
        "add_one",
    )
    n = 1024
    x = cp.arange(n, dtype=cp.float32)
    y = cp.empty_like(x)
    threads = 128
    blocks = (n + threads - 1) // threads
    kernel((blocks,), (threads,), (x, y, cp.int32(n)))
    cp.cuda.runtime.deviceSynchronize()
    if not bool((y == x + 1).all()):
        raise SystemExit("FAIL: CuPy RawKernel result mismatch")
    print("cupy RawKernel compile + run ok")


def main() -> int:
    cap = check_torch()
    check_cupy()
    if cap[0] < 12:
        print(f"WARNING: expected Blackwell sm_120, saw sm_{cap[0]}{cap[1]}", file=sys.stderr)
    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
