"""Glue so the cupy-based classical pipeline can call the compiled torch extension.

The classical estimator is written in cupy, but the kernels now live in the torch extension, so
I pass arrays across with dlpack (zero-copy, shared memory). Both libraries default to the same
CUDA stream so ordering is fine.
"""

from __future__ import annotations

from typing import Any


def ext() -> Any:
    import cuda_motion_flow._cuda as _c

    return _c


def to_torch(a: Any) -> Any:
    import cupy as cp
    import torch

    return torch.from_dlpack(cp.ascontiguousarray(a))  # type: ignore[attr-defined]


def to_cupy(t: Any) -> Any:
    import cupy as cp

    return cp.from_dlpack(t.contiguous())
