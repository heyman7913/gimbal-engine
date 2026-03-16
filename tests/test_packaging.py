"""Packaging checks: the sdist must carry the CUDA sources and the bundled weights."""

from __future__ import annotations

import os
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _build_sdist(dest: Path) -> Path:
    """Build the sdist into dest and return its path. Skipping the CUDA build keeps it fast."""
    env = {**os.environ, "CMF_SKIP_CUDA_BUILD": "1"}
    subprocess.run(
        [sys.executable, "setup.py", "sdist", "--dist-dir", str(dest)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        env=env,
    )
    sdists = list(dest.glob("*.tar.gz"))
    assert len(sdists) == 1, sdists
    return sdists[0]


def test_sdist_contains_cuda_sources_and_weights(tmp_path: Path) -> None:
    sdist = _build_sdist(tmp_path)
    with tarfile.open(sdist) as tar:
        names = tar.getnames()
    assert any(n.endswith("cuda-src/correlation.cu") for n in names), names
    assert any(n.endswith("cuda-src/classical.cu") for n in names)
    assert any(n.endswith("cuda-src/bindings.cpp") for n in names)
    assert any(n.endswith("cuda-src/kernels.h") for n in names)
    assert any(n.endswith("gimbal/weights/ihn.pt") for n in names), "bundled weights missing"


def test_bundled_weights_resolve_via_importlib_resources() -> None:
    from gimbal.learned.pretrained import bundled_weights_file

    with bundled_weights_file() as path:
        assert path.exists()
        assert path.stat().st_size > 1_000_000  # the real checkpoint, not a stub
