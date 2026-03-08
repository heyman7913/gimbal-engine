# Blackwell (sm_120) needs CUDA 12.8+; the devel image ships nvcc/nvrtc so CuPy can JIT
# RawKernels at runtime. torch brings its own CUDA runtime via the cu128 wheel.
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev git ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv for env + dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV VIRTUAL_ENV=/opt/venv
RUN uv venv "$VIRTUAL_ENV" --python 3.12
ENV PATH="/opt/venv/bin:$PATH"

# build custom kernels for the Blackwell arch
ENV TORCH_CUDA_ARCH_LIST="12.0"

# torch first, from the Blackwell wheel index; then CuPy and the rest
RUN uv pip install torch --index-url https://download.pytorch.org/whl/cu128
RUN uv pip install \
        cupy-cuda12x \
        numpy \
        opencv-python-headless \
        rich rich-click click \
        matplotlib \
        pytest ruff mypy build twine

WORKDIR /workspace
ENV PYTHONPATH=/workspace/src
