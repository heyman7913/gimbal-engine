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

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def require_cuda() -> None:
    """Hard-fail with a clear message if no CUDA device is usable."""
    if not cuda_is_available():
        raise CudaUnavailableError(
            "cuda-motion-flow requires an NVIDIA CUDA device visible to both torch and "
            "CuPy. No usable device was found. This project has no CPU fallback."
        )
