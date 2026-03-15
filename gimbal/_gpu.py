"""CUDA-presence guard. This project is GPU-only; there is no CPU compute path."""

from __future__ import annotations

import functools


class CudaUnavailableError(RuntimeError):
    """Raised when a CUDA device is required but none is usable."""


@functools.lru_cache(maxsize=1)
def cuda_is_available() -> bool:
    """True only if both torch and CuPy can see a working CUDA device."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
    except Exception:
        return False
    try:
        import cupy as cp

        return bool(cp.cuda.runtime.getDeviceCount() > 0)
    except Exception:
        return False


def require_cuda() -> None:
    """Hard-fail with a clear message if no CUDA device is usable."""
    if not cuda_is_available():
        raise CudaUnavailableError(
            "gimbal requires an NVIDIA CUDA device visible to both torch and "
            "CuPy. No usable device was found. This project has no CPU fallback."
        )


def device_synchronize() -> None:
    """Wait for all GPU work to finish. Syncs the whole device since torch and CuPy each have
    their own streams and timing has to cover both."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
    import cupy as cp

    cp.cuda.runtime.deviceSynchronize()


def used_vram_mb() -> float:
    """Currently used device memory in MiB (total minus free)."""
    import cupy as cp

    free, total = cp.cuda.runtime.memGetInfo()
    return float(total - free) / (1024.0 * 1024.0)
