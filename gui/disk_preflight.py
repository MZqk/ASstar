from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from .constants import (
        DISK_SPACE_HEADROOM_RATIO,
        DISK_SPACE_MIN_HEADROOM_BYTES,
        FITS_SUFFIXES,
        LIGHT_FRAME_EXPANSION_FACTOR,
        LIGHT_PREPROCESS_SEQUENCE_COPIES,
        LINEAR_RESUME_STAGE_ARTIFACT_COPIES,
        STAGE2_RESUME_STAGE_ARTIFACT_COPIES,
        STACKED_STAGE_ARTIFACT_COPIES,
    )
except ImportError:
    from constants import (  # type: ignore[no-redef]
        DISK_SPACE_HEADROOM_RATIO,
        DISK_SPACE_MIN_HEADROOM_BYTES,
        FITS_SUFFIXES,
        LIGHT_FRAME_EXPANSION_FACTOR,
        LIGHT_PREPROCESS_SEQUENCE_COPIES,
        LINEAR_RESUME_STAGE_ARTIFACT_COPIES,
        STAGE2_RESUME_STAGE_ARTIFACT_COPIES,
        STACKED_STAGE_ARTIFACT_COPIES,
    )


@dataclass(frozen=True)
class DiskSpaceEstimate:
    mode: str
    current_work_dir_bytes: int
    input_count: int
    input_bytes: int
    estimated_peak_growth_bytes: int
    required_free_bytes: int
    available_bytes: int
    selected_input_label: str


@dataclass(frozen=True)
class RuntimeDiskEstimate:
    volume_path: Path
    volume_device: int
    current_runtime_bytes: int
    seed_growth_bytes: int
    support_growth_bytes: int
    dependency_growth_bytes: int
    required_free_bytes: int
    available_bytes: int
    bootstrap_cache_hit: bool


def format_bytes(num_bytes: int) -> str:
    value = float(max(num_bytes, 0))
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if unit == units[-1] or value < 1024.0:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{int(value)} B"


def safe_file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def directory_size_bytes(path: Path) -> int:
    total = 0
    if not path.is_dir():
        return total
    for root, _dirs, files in os.walk(path, followlinks=False):
        root_path = Path(root)
        for name in files:
            file_path = root_path / name
            try:
                if file_path.is_symlink():
                    continue
                total += file_path.stat().st_size
            except OSError:
                continue
    return total


def existing_volume_anchor(path: Path) -> Path:
    """Return the nearest existing parent suitable for disk_usage/stat."""
    candidate = path.expanduser()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate
