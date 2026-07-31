#!/usr/bin/env python3
"""Transactional downloader for Siril's local Gaia DR3 PCC catalogue."""

from __future__ import annotations

import bz2
import hashlib
import os
import shutil
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path


GAIA_CATALOG_FILENAME = "siril_cat_healpix8_astro.dat"
GAIA_CATALOG_ARCHIVE_FILENAME = f"{GAIA_CATALOG_FILENAME}.bz2"
GAIA_CATALOG_URL = (
    "https://zenodo.org/records/14692304/files/"
    f"{GAIA_CATALOG_ARCHIVE_FILENAME}?download=1"
)
GAIA_CATALOG_ARCHIVE_SHA256 = (
    "846ad4b12c50865df0cb8c5b23453f22eec78bbe9969e17d669ae19eb49d421f"
)
GAIA_CATALOG_SHA256 = (
    "2fa40c93fe115235d35c5050757f2ef60a326a6f3030f87be1598c016fcb2388"
)
GAIA_CATALOG_SIZE_BYTES = 1_521_132_640
GAIA_CATALOG_DOWNLOAD_HEADROOM_BYTES = 512 * 1024 * 1024
DOWNLOAD_CHUNK_BYTES = 4 * 1024 * 1024


class GaiaCatalogCancelled(RuntimeError):
    """Raised when a catalogue download is cancelled."""


class GaiaCatalogDownloadError(RuntimeError):
    """Raised for a catalogue download, verification, or installation failure."""


def gaia_catalog_path(runtime_home: Path) -> Path:
    """Return the user-runtime path; the project and app bundle are never targets."""
    return (
        Path(runtime_home)
        / ".local"
        / "share"
        / "siril"
        / GAIA_CATALOG_FILENAME
    )


def gaia_catalog_status(runtime_home: Path) -> dict[str, object]:
    path = gaia_catalog_path(runtime_home)
    try:
        size = path.stat().st_size if path.is_file() else 0
    except OSError:
        size = 0
    return {
        "path": path,
        "available": size == GAIA_CATALOG_SIZE_BYTES,
        "size_bytes": int(size),
        "expected_size_bytes": GAIA_CATALOG_SIZE_BYTES,
    }


def _check_cancelled(stop_event: threading.Event) -> None:
    if stop_event.is_set():
        raise GaiaCatalogCancelled()


def _sha256_file(
    path: Path,
    *,
    stop_event: threading.Event,
) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            _check_cancelled(stop_event)
            chunk = handle.read(DOWNLOAD_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download_gaia_catalog(
    runtime_home: Path,
    *,
    stop_event: threading.Event,
    progress: Callable[[str], None],
    opener: Callable[..., object] = urllib.request.urlopen,
    force: bool = False,
) -> Path:
    """Download, verify, decompress, and atomically install the Gaia catalogue."""
    destination = gaia_catalog_path(runtime_home)
    destination.parent.mkdir(parents=True, exist_ok=True)
    status = gaia_catalog_status(runtime_home)
    if status["available"] and not force:
        progress("离线 Gaia 星色目录已安装。")
        return destination

    free_bytes = shutil.disk_usage(destination.parent).free
    required_bytes = (
        GAIA_CATALOG_SIZE_BYTES * 2 + GAIA_CATALOG_DOWNLOAD_HEADROOM_BYTES
    )
    if free_bytes < required_bytes:
        raise GaiaCatalogDownloadError(
            "安装离线 Gaia 目录至少需要 "
            f"{required_bytes / (1024**3):.1f} GiB 可用空间；"
            f"当前约 {free_bytes / (1024**3):.1f} GiB。"
        )

    archive_part = destination.with_name(f".{GAIA_CATALOG_ARCHIVE_FILENAME}.part")
    catalog_part = destination.with_name(f".{GAIA_CATALOG_FILENAME}.part")
    request = urllib.request.Request(
        GAIA_CATALOG_URL,
        headers={"User-Agent": "SeestarSuperimpose/1.0 GaiaCatalogInstaller"},
    )
    try:
        _check_cancelled(stop_event)
        progress("正在从 Siril 官方 Zenodo 数据集下载 Gaia 目录（约 1.1 GB）…")
        digest = hashlib.sha256()
        downloaded = 0
        with opener(request, timeout=60) as response, archive_part.open("wb") as output:
            total = int(response.headers.get("Content-Length") or 0)
            while True:
                _check_cancelled(stop_event)
                chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                downloaded += len(chunk)
                if total > 0:
                    progress(
                        "正在下载离线 Gaia 目录："
                        f"{downloaded * 100 // total}% "
                        f"({downloaded / (1024**3):.2f}/{total / (1024**3):.2f} GiB)"
                    )
        if digest.hexdigest() != GAIA_CATALOG_ARCHIVE_SHA256:
            raise GaiaCatalogDownloadError("Gaia 目录压缩包 SHA-256 校验失败。")

        progress("下载校验通过，正在解压 Gaia 目录…")
        written = 0
        with bz2.open(archive_part, "rb") as source, catalog_part.open("wb") as output:
            while True:
                _check_cancelled(stop_event)
                chunk = source.read(DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                output.write(chunk)
                written += len(chunk)
                progress(
                    "正在解压离线 Gaia 目录："
                    f"{min(100, written * 100 // GAIA_CATALOG_SIZE_BYTES)}%"
                )
        if written != GAIA_CATALOG_SIZE_BYTES:
            raise GaiaCatalogDownloadError(
                f"Gaia 目录大小异常：{written}，期望 {GAIA_CATALOG_SIZE_BYTES} 字节。"
            )

        progress("正在验证解压后的 Gaia 星色目录…")
        if _sha256_file(catalog_part, stop_event=stop_event) != GAIA_CATALOG_SHA256:
            raise GaiaCatalogDownloadError("Gaia 目录 SHA-256 校验失败。")
        _check_cancelled(stop_event)
        os.replace(catalog_part, destination)
        progress("离线 Gaia 星色目录安装完成。")
        return destination
    except GaiaCatalogCancelled:
        raise
    except GaiaCatalogDownloadError:
        raise
    except (OSError, EOFError, ValueError, urllib.error.URLError) as error:
        raise GaiaCatalogDownloadError(f"Gaia 目录下载或安装失败：{error}") from error
    finally:
        for temporary in (archive_part, catalog_part):
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
