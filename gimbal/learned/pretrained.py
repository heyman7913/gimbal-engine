"""Locate the trained IHN weights shipped inside the package.

The weights are package data at gimbal/weights/ihn.pt, resolved through importlib.resources so it
works the same from a wheel, an sdist, or an editable install, with no network access.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from importlib.resources import as_file, files
from pathlib import Path

WEIGHTS_NAME = "ihn.pt"


@contextmanager
def bundled_weights_file() -> Iterator[Path]:
    """Yield a real filesystem path to the bundled weights (extracted from a zip if needed)."""
    resource = files("gimbal") / "weights" / WEIGHTS_NAME
    with as_file(resource) as path:
        yield path
