"""Display-only PNG previews shared by the pipeline and macOS GUI."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_PREVIEW_MAX_SIDE = 1600
LINKED_DISPLAY_CURVE_SCHEMA = "starun.linked-display-curve.v1"
LINKED_DISPLAY_SAMPLE_LIMIT = 500_000
LINKED_DISPLAY_PERCENTILES = (0.2, 50.0, 99.8)


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    length = struct.pack(">I", len(payload))
    crc = zlib.crc32(tag)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    return length + tag + payload + struct.pack(">I", crc)


def _rgb_raw_float(image: Any, *, max_side: int) -> np.ndarray:
    """Convert image data to RGB CHW without display stretching or normalization."""
    source = np.asarray(image)
    if source.size == 0:
        raise ValueError("empty image data")

    original_dtype = source.dtype
    arr = source
    while arr.ndim > 3:
        arr = arr[0]

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=0)
    elif arr.ndim == 3:
        if arr.shape[0] in (1, 3, 4):
            arr = arr[:3]
            if arr.shape[0] == 1:
                arr = np.repeat(arr, 3, axis=0)
        elif arr.shape[-1] in (1, 3, 4):
            arr = np.transpose(arr[..., :3], (2, 0, 1))
            if arr.shape[0] == 1:
                arr = np.repeat(arr, 3, axis=0)
        else:
            raise ValueError(f"unsupported image shape: {arr.shape}")
    else:
        raise ValueError(f"unsupported image ndim: {arr.ndim}")

    arr = arr.astype(np.float32, copy=False)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)

    # Integer FITS data represents its native storage range. Floating-point
    # pipeline buffers are already in Siril's canonical 0...1 range. Neither
    # path performs histogram normalization, gamma, or any other display stretch.
    if np.issubdtype(original_dtype, np.integer):
        dtype_info = np.iinfo(original_dtype)
        positive_max = max(1.0, float(dtype_info.max))
        arr = arr / positive_max
    arr = np.clip(arr, 0.0, 1.0)

    height, width = int(arr.shape[1]), int(arr.shape[2])
    longest = max(height, width)
    if longest > max_side:
        step = max(1, int(np.ceil(longest / float(max_side))))
        arr = arr[:, ::step, ::step]

    # FITS/Siril pixel buffers use bottom-up image orientation.
    return np.flip(arr, axis=1)


def _write_png_rgb16(path: Path, rgb: np.ndarray) -> None:
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ValueError(f"expected RGB CHW array, got shape={arr.shape}")

    arr = np.clip(arr, 0.0, 1.0)
    height, width = int(arr.shape[1]), int(arr.shape[2])
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid image size: {width}x{height}")

    chw_u16 = np.round(arr * 65535.0).astype(">u2")
    hwc_u16 = np.transpose(chw_u16, (1, 2, 0))
    compressor = zlib.compressobj(level=6)
    idat_parts: list[bytes] = []
    for row in hwc_u16:
        compressed = compressor.compress(b"\x00" + row.tobytes(order="C"))
        if compressed:
            idat_parts.append(compressed)
    tail = compressor.flush()
    if tail:
        idat_parts.append(tail)

    ihdr = struct.pack(">IIBBBBB", width, height, 16, 2, 0, 0, 0)
    png_data = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", b"".join(idat_parts))
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(png_data)


def _linked_display_luminance(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ValueError(f"expected RGB CHW array, got shape={arr.shape}")
    return (
        arr[0] * np.float32(0.2126)
        + arr[1] * np.float32(0.7152)
        + arr[2] * np.float32(0.0722)
    )


def _linked_display_curve_contract_from_rgb(rgb: np.ndarray) -> dict[str, Any]:
    """Measure the serializable curve used by the observer-only preview."""

    arr = np.asarray(rgb, dtype=np.float32)
    luminance = _linked_display_luminance(arr)
    sample = luminance.ravel()
    source_sample_count = int(sample.size)
    step = 1
    if sample.size > LINKED_DISPLAY_SAMPLE_LIMIT:
        # Preserve the historical preview stride exactly. In particular, this
        # deliberately uses integer division rather than ceil division.
        step = max(1, sample.size // LINKED_DISPLAY_SAMPLE_LIMIT)
        sample = sample[::step]
    sample = sample[np.isfinite(sample)]
    if sample.size < 8:
        return {
            "schema": LINKED_DISPLAY_CURVE_SCHEMA,
            "status": "unavailable",
            "accepted": False,
            "reason": "fewer than 8 finite luminance samples",
            "sampling": {
                "source_sample_count": source_sample_count,
                "finite_sample_count": int(sample.size),
                "sample_limit": LINKED_DISPLAY_SAMPLE_LIMIT,
                "stride": int(step),
            },
        }

    black, median, white = np.percentile(sample, LINKED_DISPLAY_PERCENTILES)
    span = float(white - black)
    if not np.isfinite(span) or span <= 1e-7:
        return {
            "schema": LINKED_DISPLAY_CURVE_SCHEMA,
            "status": "unavailable",
            "accepted": False,
            "reason": "display-curve luminance span is degenerate",
            "black": float(black),
            "median": float(median),
            "white": float(white),
            "span": span,
            "sampling": {
                "source_sample_count": source_sample_count,
                "finite_sample_count": int(sample.size),
                "sample_limit": LINKED_DISPLAY_SAMPLE_LIMIT,
                "stride": int(step),
            },
        }
    median_normalized = float(np.clip((median - black) / span, 1e-6, 0.999999))
    gamma = float(
        np.clip(
            np.log(0.18) / np.log(median_normalized),
            0.20,
            1.00,
        )
    )
    return {
        "schema": LINKED_DISPLAY_CURVE_SCHEMA,
        "status": "ok",
        "accepted": True,
        "method": "rec709_percentile_linked_display",
        "black": float(black),
        "median": float(median),
        "white": float(white),
        "span": span,
        "median_normalized": median_normalized,
        "gamma": gamma,
        "luminance": "rec709",
        "percentiles": {
            "black": LINKED_DISPLAY_PERCENTILES[0],
            "median": LINKED_DISPLAY_PERCENTILES[1],
            "white": LINKED_DISPLAY_PERCENTILES[2],
        },
        "sampling": {
            "source_sample_count": source_sample_count,
            "finite_sample_count": int(sample.size),
            "sample_limit": LINKED_DISPLAY_SAMPLE_LIMIT,
            "stride": int(step),
            "stride_rule": "max(1, sample_count // sample_limit)",
        },
    }


def build_linked_display_curve_contract(
    image: Any,
    *,
    max_side: int = DEFAULT_PREVIEW_MAX_SIDE,
) -> dict[str, Any]:
    """Build the GUI-linked display curve contract from preview-sampled pixels."""

    bounded_max_side = max(64, int(max_side))
    rgb = _rgb_raw_float(image, max_side=bounded_max_side)
    contract = _linked_display_curve_contract_from_rgb(rgb)
    contract["sampling"] = {
        **dict(contract.get("sampling") or {}),
        "preview_max_side": bounded_max_side,
        "sampled_shape_chw": [int(value) for value in rgb.shape],
    }
    return contract


def _apply_linked_display_curve_contract(
    rgb: np.ndarray,
    contract: dict[str, Any],
) -> np.ndarray:
    """Apply a measured GUI display contract with the historical luma gain."""

    arr = np.asarray(rgb, dtype=np.float32)
    if contract.get("status") != "ok" or contract.get("accepted") is not True:
        return np.clip(arr, 0.0, 1.0)
    luminance = _linked_display_luminance(arr)
    black = float(contract["black"])
    span = float(contract["span"])
    gamma = float(contract["gamma"])
    normalized = np.clip((arr - black) / span, 0.0, 1.0)
    normalized_luma = np.clip((luminance - black) / span, 0.0, 1.0)
    stretched_luma = np.power(normalized_luma, gamma).astype(
        np.float32,
        copy=False,
    )
    gain = np.divide(
        stretched_luma,
        normalized_luma,
        out=np.zeros_like(stretched_luma),
        where=normalized_luma > 1e-7,
    )
    return np.clip(normalized * gain[np.newaxis, :, :], 0.0, 1.0)


def apply_linked_display_curve_contract(
    rgb: np.ndarray,
    contract: dict[str, Any],
) -> np.ndarray:
    """Apply a serialized GUI display curve to an already sampled RGB array."""

    return _apply_linked_display_curve_contract(rgb, contract)


def _linked_display_stretch(rgb: np.ndarray) -> np.ndarray:
    """Apply one bounded linked screen stretch without changing source data."""

    arr = np.asarray(rgb, dtype=np.float32)
    contract = _linked_display_curve_contract_from_rgb(arr)
    return _apply_linked_display_curve_contract(arr, contract)


def write_raw_preview(
    image: Any,
    path: Path,
    *,
    max_side: int = DEFAULT_PREVIEW_MAX_SIDE,
) -> Path:
    """Atomically write a bounded 16-bit PNG without any display stretch."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        rgb = _rgb_raw_float(image, max_side=max(64, int(max_side)))
        _write_png_rgb16(temporary, rgb)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def write_display_preview(
    image: Any,
    path: Path,
    *,
    apply_stretch: bool = True,
    max_side: int = DEFAULT_PREVIEW_MAX_SIDE,
) -> Path:
    """Write a display preview; optional stretch is linked and observer-only."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        rgb = _rgb_raw_float(image, max_side=max(64, int(max_side)))
        if apply_stretch:
            rgb = _linked_display_stretch(rgb)
        _write_png_rgb16(temporary, rgb)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def write_raw_fits_preview(
    candidates: Iterable[Path],
    path: Path,
    *,
    max_side: int = DEFAULT_PREVIEW_MAX_SIDE,
) -> Path:
    """Write the first readable FIT/FITS candidate and return its source path."""
    try:
        from astropy.io import fits
    except ImportError as exc:  # pragma: no cover - runtime preflight covers this.
        raise RuntimeError("astropy is unavailable for FITS preview") from exc

    errors: list[str] = []
    for candidate in candidates:
        source = Path(candidate)
        if not source.is_file():
            continue
        try:
            data = fits.getdata(source, memmap=False)
            write_raw_preview(data, path, max_side=max_side)
            return source
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            errors.append(f"{source.name}: {exc}")
    detail = "; ".join(errors[:3]) or "no readable FIT/FITS candidate"
    raise RuntimeError(detail)


def write_display_fits_preview(
    candidates: Iterable[Path],
    path: Path,
    *,
    apply_stretch: bool = True,
    max_side: int = DEFAULT_PREVIEW_MAX_SIDE,
) -> Path:
    """Write the first readable FITS as a labeled display-only preview."""

    try:
        from astropy.io import fits
    except ImportError as exc:  # pragma: no cover - runtime preflight covers this.
        raise RuntimeError("astropy is unavailable for FITS preview") from exc

    errors: list[str] = []
    for candidate in candidates:
        source = Path(candidate)
        if not source.is_file():
            continue
        try:
            data = fits.getdata(source, memmap=False)
            write_display_preview(
                data,
                path,
                apply_stretch=apply_stretch,
                max_side=max_side,
            )
            return source
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            errors.append(f"{source.name}: {exc}")
    detail = "; ".join(errors[:3]) or "no readable FIT/FITS candidate"
    raise RuntimeError(detail)


__all__ = [
    "DEFAULT_PREVIEW_MAX_SIDE",
    "write_display_fits_preview",
    "write_display_preview",
    "write_raw_fits_preview",
    "write_raw_preview",
]
