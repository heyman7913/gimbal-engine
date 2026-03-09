"""Benchmark dataset description and loaders.

The NUS video stabilization dataset (Liu et al., SIGGRAPH 2013) is the real-footage
benchmark. A small, category-spanning subset is vendored in the repo (Git LFS) so the
benchmark runs offline; the loaders here extract that subset on demand. The full-set fetch
is a documented secondary path. Phase-B training clips and benchmark clips are kept in
disjoint splits so reported numbers are never measured on training footage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# NUS scene categories used for per-category reporting.
NUS_CATEGORIES = (
    "regular",
    "low_texture",
    "zooming",
    "parallax",
    "crowd",
    "running",
)

# Official source, documented for provenance. The vendored subset is the default.
NUS_SOURCE_URL = "http://liushuaicheng.org/SIGGRAPH2013/database.html"


@dataclass(frozen=True)
class Clip:
    path: Path
    category: str
    split: str  # "benchmark" or "phaseb"


def vendored_root() -> Path:
    return Path("data") / "nus"
