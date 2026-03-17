"""Downloading and loading the datasets.

Phase A trains on a slice of COCO; the real-footage benchmark is the NUS dataset (Liu et al.
2013), grouped by scene type. Both get pulled from their sources and cached under data/. The
important detail is that the Phase-B training clips and the benchmark clips are kept in separate
splits, so I'm never reporting numbers on footage the model trained on.
"""

from __future__ import annotations

import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

# NUS scene categories used for per-category reporting (the dataset's own grouping).
NUS_CATEGORIES = ("Regular", "QuickRotation", "Zooming", "Parallax", "Crowd", "Running")

NUS_BASE = "http://liushuaicheng.org/SIGGRAPH2013/"
NUS_SOURCE_URL = NUS_BASE + "database.html"
COCO_URL = "http://images.cocodataset.org/zips/val2017.zip"


@dataclass(frozen=True)
class Clip:
    path: Path
    category: str
    split: str  # "benchmark" or "phaseb"


def nus_root() -> Path:
    return Path("data") / "nus"


def coco_root() -> Path:
    return Path("data") / "coco"


def fetch_coco(limit: int = 5000, dest: Path | None = None) -> Path:
    """Download the COCO val2017 subset and extract up to `limit` jpgs. Returns the dir."""
    out = dest or coco_root()
    out.mkdir(parents=True, exist_ok=True)
    existing = list(out.glob("*.jpg"))
    if len(existing) >= min(limit, 4000):
        return out
    zip_path = out.parent / "val2017.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(COCO_URL, zip_path)  # noqa: S310
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".jpg")][:limit]
        for m in members:
            data = zf.read(m)
            (out / Path(m).name).write_bytes(data)
    return out


def fetch_nus(categories: tuple[str, ...] = NUS_CATEGORIES, dest: Path | None = None) -> Path:
    """Download and extract the NUS category zips. Returns the dataset root."""
    out = dest or nus_root()
    out.mkdir(parents=True, exist_ok=True)
    for cat in categories:
        cat_dir = out / cat
        if cat_dir.exists() and any(cat_dir.iterdir()):
            continue
        zip_path = out / f"{cat}.zip"
        if zip_path.exists() and not _is_valid_zip(zip_path):
            zip_path.unlink()  # partial/corrupt download from a previous interrupted run
        if not zip_path.exists():
            _download_with_retry(NUS_BASE + f"data/{cat}.zip", zip_path)
        cat_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            for m in zf.namelist():
                if m.lower().endswith((".avi", ".mp4", ".mov")):
                    (cat_dir / Path(m).name).write_bytes(zf.read(m))
    return out


def _is_valid_zip(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as zf:
            zf.namelist()
        return True
    except zipfile.BadZipFile:
        return False


def _download_with_retry(url: str, dest: Path, attempts: int = 8) -> None:
    """Download with retries; the NUS host drops connections mid-transfer."""
    for i in range(attempts):
        try:
            urllib.request.urlretrieve(url, dest)  # noqa: S310
            return
        except (urllib.error.URLError, urllib.error.ContentTooShortError, TimeoutError):
            if dest.exists():
                dest.unlink()
            if i == attempts - 1:
                raise


def nus_clips(
    root: Path | None = None, per_category: int | None = 4, benchmark_fraction: float = 0.5
) -> list[Clip]:
    """Collect clips per category and split them into disjoint benchmark / phaseb sets.

    The benchmark and Phase-B splits never share a clip, so reported numbers are not measured
    on training footage. per_category=None takes every original clip in each category.
    """
    base = root or nus_root()
    clips: list[Clip] = []
    for cat in NUS_CATEGORIES:
        cat_dir = base / cat
        if not cat_dir.exists():
            continue
        # the dataset ships both the shaky originals and the authors' stabilized results
        # (named *stb.avi); benchmark only the originals
        files = sorted(
            p
            for p in cat_dir.iterdir()
            if p.suffix.lower() in {".avi", ".mp4", ".mov"} and "stb" not in p.stem.lower()
        )
        if per_category is not None:
            files = files[:per_category]
        n_bench = max(1, int(round(len(files) * benchmark_fraction)))
        for i, f in enumerate(files):
            split = "benchmark" if i < n_bench else "phaseb"
            clips.append(Clip(path=f, category=cat, split=split))
    return clips
