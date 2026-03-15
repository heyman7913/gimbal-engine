import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip tests marked `cuda` when no CUDA device is available."""
    from gimbal._gpu import cuda_is_available

    if cuda_is_available():
        return
    skip_cuda = pytest.mark.skip(reason="no CUDA device available")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip_cuda)
