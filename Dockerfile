# my laptop GPU is a 5070 Ti (Blackwell, sm_120) which needs CUDA 12.8+.
# using the -devel image so nvcc is available, otherwise CuPy can't compile the kernels.
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev git ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV VIRTUAL_ENV=/opt/venv
RUN uv venv "$VIRTUAL_ENV" --python 3.12
ENV PATH="/opt/venv/bin:$PATH"

ENV TORCH_CUDA_ARCH_LIST="12.0"

# the default torch wheel doesn't work on Blackwell yet, need the cu128 one
RUN uv pip install torch --index-url https://download.pytorch.org/whl/cu128
RUN uv pip install \
        cupy-cuda12x \
        numpy \
        opencv-python-headless \
        rich rich-click click \
        matplotlib \
        ninja \
        pytest ruff mypy build twine

WORKDIR /workspace
ENV PYTHONPATH=/workspace
