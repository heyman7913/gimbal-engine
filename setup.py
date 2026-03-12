"""The CUDA extension is built here; everything else lives in pyproject.toml.

setuptools doesn't let you declare ext_modules in pyproject yet, so the compiled kernels need
this file. Build for Blackwell with TORCH_CUDA_ARCH_LIST=12.0.
"""

from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

csrc = Path(__file__).parent / "csrc"
sources = [str(csrc / f) for f in ("bindings.cpp", "correlation.cu", "classical.cu")]

setup(
    ext_modules=[
        CUDAExtension(
            name="cuda_motion_flow._cuda",
            sources=sources,
            include_dirs=[str(csrc)],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
