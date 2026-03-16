"""The CUDA extension is built here; everything else lives in pyproject.toml.

setuptools doesn't let you declare ext_modules in pyproject yet, so the compiled kernels need
this file. The extension compiles at install time, so it adapts to the user's CUDA and GPU: the
target architecture is taken from TORCH_CUDA_ARCH_LIST if set, else detected from the local GPU,
else left to PyTorch's default arch list. Set CMF_SKIP_CUDA_BUILD=1 to install the pure-Python
package without nvcc/torch (for the CLI, docs, or CI on a GPU-less box); the compiled estimators
then raise on use until the extension is built.
"""

import os
import shutil

from setuptools import setup


def _preflight() -> None:
    """Fail with a readable message, not raw compiler output, if the toolchain is missing."""
    from torch.utils.cpp_extension import CUDA_HOME

    nvcc = shutil.which("nvcc")
    if nvcc is None and CUDA_HOME:
        candidate = os.path.join(CUDA_HOME, "bin", "nvcc")
        nvcc = candidate if os.path.exists(candidate) else None
    if nvcc is None:
        raise SystemExit(
            "gimbal_engine build error: the CUDA toolkit (nvcc) was not found.\n"
            "This package compiles a CUDA extension at install time. Install a CUDA toolkit "
            "matching your PyTorch build and put nvcc on PATH or set CUDA_HOME, then reinstall."
        )


def _select_arch() -> None:
    """Target the local GPU's architecture; fall back to PyTorch's default multi-arch list."""
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return
    import torch

    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    # else: leave it unset so torch builds for the arches its own CUDA build supports


ext_modules = []
cmdclass = {}
if os.environ.get("CMF_SKIP_CUDA_BUILD") != "1":
    _preflight()
    _select_arch()
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    # relative paths, setuptools rejects absolute ones when building the wheel
    sources = [f"cuda-src/{f}" for f in ("bindings.cpp", "correlation.cu", "classical.cu")]
    ext_modules = [
        CUDAExtension(
            name="gimbal._cuda",
            sources=sources,
            include_dirs=["cuda-src"],
        )
    ]
    cmdclass = {"build_ext": BuildExtension}

setup(ext_modules=ext_modules, cmdclass=cmdclass)
