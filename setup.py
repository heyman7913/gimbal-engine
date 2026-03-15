"""The CUDA extension is built here; everything else lives in pyproject.toml.

setuptools doesn't let you declare ext_modules in pyproject yet, so the compiled kernels need
this file. Build for Blackwell with TORCH_CUDA_ARCH_LIST=12.0. Set CMF_SKIP_CUDA_BUILD=1 to
install the pure-Python package without nvcc/torch (for the CLI, docs, or CI on a GPU-less box);
the compiled estimators then raise on use until the extension is built.
"""

import os

from setuptools import setup

ext_modules = []
cmdclass = {}
if os.environ.get("CMF_SKIP_CUDA_BUILD") != "1":
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    # relative paths, setuptools rejects absolute ones when building the wheel
    sources = [f"cuda-src/{f}" for f in ("bindings.cpp", "correlation.cu", "classical.cu")]
    ext_modules = [
        CUDAExtension(
            name="cuda_motion_flow._cuda",
            sources=sources,
            include_dirs=["cuda-src"],
        )
    ]
    cmdclass = {"build_ext": BuildExtension}

setup(ext_modules=ext_modules, cmdclass=cmdclass)
