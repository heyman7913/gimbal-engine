"""Grabs the trained weights.

The weights don't go in the wheel (too big), so the first time you need them they get pulled
from a URL and cached. Training writes to the same place.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

DEFAULT_URL = "https://github.com/heyman7913/cuda-motion-flow/releases/latest/download/ihn.pt"


def weights_dir() -> Path:
    return Path(os.environ.get("CUDA_MOTION_FLOW_WEIGHTS_DIR", "weights"))


def local_weights_path(name: str = "ihn.pt") -> Path:
    return weights_dir() / name


def ensure_weights(name: str = "ihn.pt", url: str | None = None) -> Path:
    """Return the local weights path, downloading it on first use if missing."""
    path = local_weights_path(name)
    if path.exists():
        return path
    source = url or os.environ.get("CUDA_MOTION_FLOW_WEIGHTS_URL", DEFAULT_URL)
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(source, path)  # noqa: S310 (https release asset)
    return path
