"""Independent color-managed Stage 10 display and editable exports."""
from __future__ import annotations

import hashlib
import struct
import zlib
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

import display_rendition
import stage9_quality
from stage8_starless_finish import (
    DECODED_PIXEL_SHA256_METHOD,
    canonical_decoded_pixel_sha256,
    pixel_sha256,
)

MANAGED_OUTPUT_SCHEMA = "starun.managed-output.v2"
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_DISPLAY_LUMA_WEIGHTS = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)
_GALAXY_TARGET_TYPES = frozenset({"galaxy", "large_galaxy", "small_galaxy"})
_DIFFUSE_TARGET_MARKERS = ("galaxy", "nebula", "milky_way")
_SRGB_PROFILE_CANDIDATES = (
    Path("/System/Library/ColorSync/Profiles/sRGB Profile.icc"),
    Path("/Library/ColorSync/Profiles/sRGB Profile.icc"),
    Path("/usr/share/color/icc/colord/sRGB.icc"),
    Path("/usr/share/color/icc/sRGB.icc"),
)


def _as_display_rgb(image: Any) -> np.ndarray:
    source = np.asarray(image)
    if source.ndim != 3:
        raise ValueError("managed export requires an RGB image")
    if source.shape[0] == 3:
        rgb = source
    elif source.shape[-1] == 3:
        rgb = np.transpose(source, (2, 0, 1))
    else:
        raise ValueError(f"expected RGB input, got shape={source.shape}")
    original_dtype = source.dtype
    rgb = rgb.astype(np.float32, copy=True)
    if np.issubdtype(original_dtype, np.integer):
        rgb /= max(1.0, float(np.iinfo(original_dtype).max))
    if not np.all(np.isfinite(rgb)):
        raise ValueError("nonfinite managed-export pixels")
    # Siril/FITS buffers are bottom-up; display and TIFF scanlines are top-down.
    return np.flip(np.clip(rgb, 0.0, 1.0), axis=1)


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(tag)
    checksum = zlib.crc32(payload, checksum) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", checksum)
    )


def _write_display_rgb_png(path: Path, rgb: np.ndarray) -> Path:
    """Write already top-down display RGB into the managed PNG container."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError(f"expected RGB CHW array, got shape={rgb.shape}")
    if not np.all(np.isfinite(rgb)):
        raise ValueError("nonfinite managed-display pixels")
    rgb = np.clip(rgb, 0.0, 1.0)
    height, width = int(rgb.shape[1]), int(rgb.shape[2])
    pixels = np.transpose(
        np.round(rgb * 65535.0).astype(">u2"),
        (1, 2, 0),
    )
    try:
        with temporary.open("wb") as handle:
            handle.write(_PNG_SIGNATURE)
            handle.write(
                _png_chunk(
                    b"IHDR",
                    struct.pack(
                        ">IIBBBBB",
                        width,
                        height,
                        16,
                        2,
                        0,
                        0,
                        0,
                    ),
                )
            )
            handle.write(_png_chunk(b"sRGB", b"\x00"))
            handle.write(_png_chunk(b"gAMA", struct.pack(">I", 45455)))
            handle.write(
                _png_chunk(
                    b"cHRM",
                    struct.pack(
                        ">8I",
                        31270,
                        32900,
                        64000,
                        33000,
                        30000,
                        60000,
                        15000,
                        6000,
                    ),
                )
            )
            compressor = zlib.compressobj(level=6)
            pending = bytearray()
            for row in pixels:
                pending.extend(
                    compressor.compress(b"\x00" + row.tobytes(order="C"))
                )
                if len(pending) >= 1024 * 1024:
                    handle.write(_png_chunk(b"IDAT", bytes(pending)))
                    pending.clear()
            pending.extend(compressor.flush())
            if pending:
                handle.write(_png_chunk(b"IDAT", bytes(pending)))
            handle.write(_png_chunk(b"IEND", b""))
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def write_managed_display_png(path: Path, image: Any) -> Path:
    """Write a 16-bit RGB PNG with explicit sRGB transfer/chromaticity metadata."""
    return _write_display_rgb_png(Path(path), _as_display_rgb(image))


def _read_managed_display_png(path: Path) -> np.ndarray:
    """Decode exact pixels from a PNG emitted by ``_write_display_rgb_png``."""
    target = Path(path)
    idat = bytearray()
    width = height = bit_depth = color_type = None
    with target.open("rb") as handle:
        if handle.read(8) != _PNG_SIGNATURE:
            raise ValueError("invalid managed PNG signature")
        while True:
            header = handle.read(8)
            if len(header) != 8:
                raise ValueError("truncated managed PNG chunk header")
            length, tag = struct.unpack(">I4s", header)
            payload = handle.read(length)
            checksum = handle.read(4)
            if len(payload) != length or len(checksum) != 4:
                raise ValueError("truncated managed PNG chunk")
            expected_crc = zlib.crc32(tag)
            expected_crc = zlib.crc32(payload, expected_crc) & 0xFFFFFFFF
            if struct.unpack(">I", checksum)[0] != expected_crc:
                raise ValueError("managed PNG chunk CRC mismatch")
            if tag == b"IHDR":
                if len(payload) != 13:
                    raise ValueError("invalid managed PNG IHDR")
                width, height, bit_depth, color_type, compression, filtering, interlace = (
                    struct.unpack(">IIBBBBB", payload)
                )
                if (compression, filtering, interlace) != (0, 0, 0):
                    raise ValueError("unsupported managed PNG encoding")
            elif tag == b"IDAT":
                idat.extend(payload)
            elif tag == b"IEND":
                break
    if not width or not height or bit_depth != 16 or color_type != 2:
        raise ValueError("managed PNG is not non-interlaced 16-bit RGB")
    try:
        raw = zlib.decompress(bytes(idat))
    except zlib.error as error:
        raise ValueError(f"managed PNG IDAT decode failed: {error}") from error
    row_bytes = int(width) * 3 * 2
    expected_size = int(height) * (row_bytes + 1)
    if len(raw) != expected_size:
        raise ValueError(
            f"managed PNG pixel length mismatch: {len(raw)} != {expected_size}"
        )
    rows = []
    for row_index in range(int(height)):
        offset = row_index * (row_bytes + 1)
        if raw[offset] != 0:
            raise ValueError("managed PNG uses an unexpected row filter")
        rows.append(raw[offset + 1 : offset + 1 + row_bytes])
    pixels = np.frombuffer(b"".join(rows), dtype=">u2").reshape(
        int(height),
        int(width),
        3,
    )
    return np.transpose(pixels.astype(np.float32) / 65535.0, (2, 0, 1))


def read_managed_display_png(path: Path) -> np.ndarray:
    """Decode display RGB from a PNG written by this managed exporter."""
    return _read_managed_display_png(Path(path))


def canonical_managed_derivative_pixels(image: Any) -> np.ndarray:
    """Return the exact 16-bit top-down pixel domain used by managed exports."""

    display_rgb = _as_display_rgb(image)
    return (
        np.rint(np.clip(display_rgb, 0.0, 1.0) * 65535.0)
        .astype(np.uint16)
        .astype(np.float32)
        / np.float32(65535.0)
    )


def _box_blur_gray(gray: np.ndarray) -> np.ndarray:
    arr = np.asarray(gray, dtype=np.float32)
    height, width = arr.shape
    padded = np.pad(arr, ((1, 1), (1, 1)), mode="reflect")
    result = np.zeros_like(arr, dtype=np.float32)
    for y_offset in range(3):
        for x_offset in range(3):
            result += padded[
                y_offset : y_offset + height,
                x_offset : x_offset + width,
            ]
    return result / np.float32(9.0)


def _pooled_gray(
    gray: np.ndarray,
    *,
    max_side: int,
    reducer: str,
) -> np.ndarray:
    """Bound visibility checks while retaining either diffuse flux or star peaks."""
    arr = np.asarray(gray, dtype=np.float32)
    height, width = arr.shape
    step = max(1, int(np.ceil(max(height, width) / float(max_side))))
    if step == 1:
        return arr
    pooled_height = height // step
    pooled_width = width // step
    if pooled_height < 3 or pooled_width < 3:
        return arr[::step, ::step]
    blocks = arr[
        : pooled_height * step,
        : pooled_width * step,
    ].reshape(pooled_height, step, pooled_width, step)
    if reducer == "max":
        return np.max(blocks, axis=(1, 3))
    return np.mean(blocks, axis=(1, 3), dtype=np.float32)


def _display_luminance(rgb: np.ndarray) -> np.ndarray:
    return np.sum(
        np.asarray(rgb, dtype=np.float32) * _DISPLAY_LUMA_WEIGHTS[:, None, None],
        axis=0,
        dtype=np.float32,
    )


def _rounded(value: float) -> float:
    return round(float(value), 6)


def audit_display_visibility(
    rgb: np.ndarray,
    *,
    target_type: str = "",
    stars_required: bool = False,
    star_reference: Optional[Dict[str, Any]] = None,
    pixel_coordinate_domain: str = "display_array_top_down",
    star_visibility_config: Any = None,
) -> Dict[str, Any]:
    """Measure whether encoded display pixels are bright and astronomically legible."""
    source = np.asarray(rgb)
    original_dtype = source.dtype
    arr = np.asarray(source, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] != 3 or arr.size == 0:
        raise ValueError(f"expected non-empty RGB CHW array, got shape={arr.shape}")
    if np.issubdtype(original_dtype, np.integer):
        arr /= max(1.0, float(np.iinfo(original_dtype).max))
    if not np.all(np.isfinite(arr)):
        raise ValueError("nonfinite display-audit pixels")
    arr = np.clip(arr, 0.0, 1.0)
    luminance = _display_luminance(arr)
    p01, p10, median, p90, p99, p999 = np.quantile(
        luminance,
        (0.01, 0.10, 0.50, 0.90, 0.99, 0.999),
    )
    channel_medians = np.median(arr, axis=(1, 2))
    black_ratio = float(np.mean(luminance <= 0.01))
    white_ratio = float(np.mean(np.max(arr, axis=0) >= 0.995))
    display_range = float(p99 - p10)
    sparse_signal_range = float(p999 - p10)

    broad = _pooled_gray(luminance, max_side=512, reducer="mean")
    blur_iterations = max(2, min(8, int(min(broad.shape) // 16)))
    for _ in range(blur_iterations):
        broad = _box_blur_gray(broad)
    broad_bg, broad_p95, broad_p99 = np.quantile(broad, (0.20, 0.95, 0.99))
    broad_contrast_p95 = float(broad_p95 - broad_bg)
    broad_contrast_p99 = float(broad_p99 - broad_bg)
    broad_signal_floor = max(0.025, broad_contrast_p99 * 0.25)
    broad_signal_coverage = float(np.mean(broad > broad_bg + broad_signal_floor))

    star_map = _pooled_gray(luminance, max_side=1600, reducer="max")
    star_blur = _box_blur_gray(star_map)
    high_pass = star_map - star_blur
    high_pass_center = float(np.median(high_pass))
    high_pass_mad = float(np.median(np.abs(high_pass - high_pass_center)))
    bright_floor = max(
        float(np.quantile(star_map, 0.985)),
        float(np.median(star_map)) + 0.06,
    )
    peak_contrast_floor = max(0.015, high_pass_mad * 6.0)
    peak_count = 0
    peak_luminance_median = 0.0
    peak_contrast_median = 0.0
    if min(star_map.shape) >= 3:
        center = star_map[1:-1, 1:-1]
        center_high_pass = high_pass[1:-1, 1:-1]
        neighbor_max = np.maximum.reduce(
            [
                star_map[:-2, :-2],
                star_map[:-2, 1:-1],
                star_map[:-2, 2:],
                star_map[1:-1, :-2],
                star_map[1:-1, 2:],
                star_map[2:, :-2],
                star_map[2:, 1:-1],
                star_map[2:, 2:],
            ]
        )
        peak_mask = (
            (center >= neighbor_max)
            & (center >= bright_floor)
            & (center_high_pass >= peak_contrast_floor)
        )
        peak_count = int(np.count_nonzero(peak_mask))
        if peak_count:
            peak_luminance_median = float(np.median(center[peak_mask]))
            peak_contrast_median = float(np.median(center_high_pass[peak_mask]))

    target = str(target_type or "").strip().lower()
    galaxy_required = target in _GALAXY_TARGET_TYPES
    extended_required = galaxy_required or any(
        marker in target for marker in _DIFFUSE_TARGET_MARKERS
    )
    extended_passed = bool(
        max(broad_contrast_p95, broad_contrast_p99) >= 0.04
        and broad_signal_coverage >= 0.003
    )
    latent_extended_subject_mappable = bool(
        max(broad_contrast_p95, broad_contrast_p99) >= 0.005
        and broad_signal_coverage >= 0.003
    )
    generic_peak_passed = bool(
        peak_count >= 3
        and peak_luminance_median >= 0.20
        and peak_contrast_median >= 0.015
    )
    catalog_visibility = stage9_quality.assess_catalog_star_visibility(
        arr,
        star_reference,
        star_visibility_config,
        coordinate_domain=pixel_coordinate_domain,
    )
    star_passed = bool(
        catalog_visibility.get("available", False)
        and catalog_visibility.get("passed", False)
    )
    scene_content_visible = bool(
        sparse_signal_range >= 0.08
        or extended_passed
        or generic_peak_passed
        or star_passed
    )
    required_subject_mappable = bool(
        not extended_required
        or extended_passed
        or latent_extended_subject_mappable
    )
    underexposed = bool(
        median < 0.10
        or p90 < 0.18
        or black_ratio > 0.80
    )
    overexposed = bool(
        p10 > 0.35
        or median > 0.45
        or white_ratio > 0.20
    )
    if overexposed:
        exposure_state = "overexposed"
    elif not scene_content_visible or not required_subject_mappable:
        exposure_state = "unmappable"
    elif underexposed:
        exposure_state = "underexposed"
    else:
        exposure_state = "acceptable"
    brightness_passed = bool(
        median >= 0.10
        and p90 >= 0.18
        and scene_content_visible
        and black_ratio <= 0.80
        and white_ratio <= 0.20
    )
    upper_bounds_passed = bool(
        p10 <= 0.35
        and median <= 0.45
        and white_ratio <= 0.20
    )
    checks: Dict[str, Dict[str, Any]] = {
        "pixel_brightness": {
            "required": True,
            "passed": brightness_passed,
            "thresholds": {
                "luminance_median_min": 0.10,
                "luminance_p90_min": 0.18,
                "visible_scene_required": True,
                "sparse_signal_p999_minus_p10_min": 0.08,
                "black_pixel_ratio_max": 0.80,
                "white_clip_ratio_max": 0.20,
            },
        },
        "exposure_upper_bounds": {
            "required": True,
            "passed": upper_bounds_passed,
            "thresholds": {
                "luminance_p10_max": 0.35,
                "luminance_median_max": 0.45,
                "white_clip_ratio_max": 0.20,
            },
        },
        "galaxy_visibility": {
            "required": galaxy_required,
            "passed": extended_passed if galaxy_required else None,
            "thresholds": {
                "broad_signal_contrast_min": 0.04,
                "broad_signal_coverage_min": 0.003,
            },
        },
        "extended_subject_visibility": {
            "required": extended_required and not galaxy_required,
            "passed": (
                extended_passed
                if extended_required and not galaxy_required
                else None
            ),
            "thresholds": {
                "broad_signal_contrast_min": 0.04,
                "broad_signal_coverage_min": 0.003,
            },
        },
        "star_visibility": {
            "required": bool(stars_required),
            "passed": star_passed if stars_required else None,
            "detected": (
                star_passed if stars_required else generic_peak_passed
            ),
            "method": (
                "source_catalog_absolute_visibility"
                if stars_required
                else "not_required"
            ),
            "catalog_visibility": catalog_visibility,
            "compact_peak_diagnostic": {
                "formal_gate": False,
                "passed": generic_peak_passed,
                "reason": (
                    "generic compact peaks cannot satisfy stars_required"
                ),
            },
            "thresholds": {
                "catalog_all_visibility_ratio_min": 0.75,
                "catalog_weak_visibility_ratio_min": 0.70,
                "catalog_bright_visibility_ratio_min": 0.90,
                "source_and_candidate_local_contrast_min": 0.002,
            },
        },
    }
    failed_checks = [
        name
        for name, check in checks.items()
        if bool(check.get("required")) and check.get("passed") is not True
    ]
    return {
        "schema": "starun.display-visibility.v2",
        "status": "passed" if not failed_checks else "failed",
        "passed": not failed_checks,
        "exposure_state": exposure_state,
        "target_type": target or "unknown",
        "stars_required": bool(stars_required),
        "metrics": {
            "red_median": _rounded(channel_medians[0]),
            "green_median": _rounded(channel_medians[1]),
            "blue_median": _rounded(channel_medians[2]),
            "luminance_p01": _rounded(p01),
            "luminance_p10": _rounded(p10),
            "luminance_median": _rounded(median),
            "luminance_p90": _rounded(p90),
            "luminance_p99": _rounded(p99),
            "luminance_p999": _rounded(p999),
            "p99_minus_p10": _rounded(display_range),
            "p999_minus_p10": _rounded(sparse_signal_range),
            "scene_content_visible": scene_content_visible,
            "required_subject_mappable": required_subject_mappable,
            "latent_extended_subject_mappable": (
                latent_extended_subject_mappable
            ),
            "black_pixel_ratio": _rounded(black_ratio),
            "white_clip_ratio": _rounded(white_ratio),
            "underexposed": underexposed,
            "overexposed": overexposed,
            "broad_signal_contrast_p95": _rounded(broad_contrast_p95),
            "broad_signal_contrast_p99": _rounded(broad_contrast_p99),
            "broad_signal_coverage": _rounded(broad_signal_coverage),
            "compact_peak_count": peak_count,
            "compact_peak_diagnostic_passed": generic_peak_passed,
            "peak_luminance_median": _rounded(peak_luminance_median),
            "peak_contrast_median": _rounded(peak_contrast_median),
            "peak_contrast_floor": _rounded(peak_contrast_floor),
        },
        "checks": checks,
        "failed_checks": failed_checks,
    }


def _linked_visibility_stretch(rgb: np.ndarray) -> tuple[np.ndarray, Dict[str, Any]]:
    """Build and replay the single bounded v2 mapping for a dark source."""
    arr = np.asarray(rgb, dtype=np.float32)
    contract = display_rendition.build_linked_review_contract(
        arr,
        reason="managed_output_underexposed_source",
        source_stem="stage10_final",
    )
    return display_rendition.apply_review_contract(arr, contract), contract


def _valid_icc_profile(data: bytes) -> bool:
    if len(data) < 128:
        return False
    declared_size = struct.unpack(">I", data[:4])[0]
    return data[36:40] == b"acsp" and 128 <= declared_size <= len(data)


def find_srgb_icc_profile() -> tuple[Optional[bytes], Optional[Path]]:
    for candidate in _SRGB_PROFILE_CANDIDATES:
        try:
            data = candidate.read_bytes()
        except OSError:
            continue
        if _valid_icc_profile(data):
            return data, candidate
    return None, None


def _align(value: int, boundary: int = 4) -> int:
    return (int(value) + boundary - 1) // boundary * boundary


def write_managed_edit_tiff(
    path: Path,
    image: Any,
    *,
    icc_profile: bytes,
) -> Path:
    """Write uncompressed 16-bit RGB TIFF with an embedded ICC profile."""
    if not _valid_icc_profile(icc_profile):
        raise ValueError("valid sRGB ICC profile is required for editable TIFF")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    rgb = _as_display_rgb(image)
    height, width = int(rgb.shape[1]), int(rgb.shape[2])
    pixel_bytes = np.transpose(
        np.round(rgb * 65535.0).astype("<u2"),
        (1, 2, 0),
    ).tobytes(order="C")
    software = b"Starun AstroSuite managed export\x00"

    entry_count = 16
    external_offset = 8 + 2 + entry_count * 12 + 4
    sections: list[tuple[int, bytes]] = []

    def allocate(payload: bytes, *, alignment: int = 4) -> int:
        nonlocal external_offset
        external_offset = _align(external_offset, alignment)
        offset = external_offset
        sections.append((offset, payload))
        external_offset += len(payload)
        return offset

    bits_offset = allocate(struct.pack("<HHH", 16, 16, 16), alignment=2)
    xres_offset = allocate(struct.pack("<II", 300, 1))
    yres_offset = allocate(struct.pack("<II", 300, 1))
    software_offset = allocate(software)
    sample_format_offset = allocate(
        struct.pack("<HHH", 1, 1, 1),
        alignment=2,
    )
    icc_offset = allocate(bytes(icc_profile))
    pixel_offset = allocate(pixel_bytes, alignment=4)

    def inline_short(value: int) -> bytes:
        return struct.pack("<H", int(value)) + b"\x00\x00"

    def inline_long(value: int) -> bytes:
        return struct.pack("<I", int(value))

    entries = [
        (256, 4, 1, inline_long(width)),
        (257, 4, 1, inline_long(height)),
        (258, 3, 3, inline_long(bits_offset)),
        (259, 3, 1, inline_short(1)),
        (262, 3, 1, inline_short(2)),
        (273, 4, 1, inline_long(pixel_offset)),
        (277, 3, 1, inline_short(3)),
        (278, 4, 1, inline_long(height)),
        (279, 4, 1, inline_long(len(pixel_bytes))),
        (282, 5, 1, inline_long(xres_offset)),
        (283, 5, 1, inline_long(yres_offset)),
        (284, 3, 1, inline_short(1)),
        (296, 3, 1, inline_short(2)),
        (305, 2, len(software), inline_long(software_offset)),
        (339, 3, 3, inline_long(sample_format_offset)),
        (34675, 1, len(icc_profile), inline_long(icc_offset)),
    ]
    try:
        with temporary.open("wb") as handle:
            handle.write(b"II")
            handle.write(struct.pack("<HI", 42, 8))
            handle.write(struct.pack("<H", len(entries)))
            for tag, value_type, count, value in sorted(entries):
                handle.write(
                    struct.pack("<HHI", tag, value_type, count)
                    + value
                )
            handle.write(struct.pack("<I", 0))
            for offset, payload in sorted(sections):
                current = handle.tell()
                if current > offset:
                    raise RuntimeError("invalid TIFF section layout")
                if current < offset:
                    handle.write(b"\x00" * (offset - current))
                handle.write(payload)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _read_managed_edit_tiff(path: Path) -> np.ndarray:
    """Decode pixels from the deterministic TIFF emitted above."""

    payload = Path(path).read_bytes()
    if len(payload) < 14 or payload[:2] != b"II":
        raise ValueError("managed TIFF byte order is invalid")
    magic, ifd_offset = struct.unpack_from("<HI", payload, 2)
    if magic != 42 or ifd_offset < 8 or ifd_offset + 2 > len(payload):
        raise ValueError("managed TIFF header is invalid")
    entry_count = struct.unpack_from("<H", payload, ifd_offset)[0]
    cursor = ifd_offset + 2
    entries: dict[int, tuple[int, int, bytes]] = {}
    for _ in range(entry_count):
        if cursor + 12 > len(payload):
            raise ValueError("managed TIFF IFD is truncated")
        tag, value_type, count = struct.unpack_from("<HHI", payload, cursor)
        entries[int(tag)] = (
            int(value_type),
            int(count),
            payload[cursor + 8 : cursor + 12],
        )
        cursor += 12

    def long_value(tag: int) -> int:
        value_type, count, raw = entries[tag]
        if value_type != 4 or count != 1:
            raise ValueError(f"managed TIFF tag {tag} is invalid")
        return int(struct.unpack("<I", raw)[0])

    def short_value(tag: int) -> int:
        value_type, count, raw = entries[tag]
        if value_type != 3 or count != 1:
            raise ValueError(f"managed TIFF tag {tag} is invalid")
        return int(struct.unpack("<H", raw[:2])[0])

    try:
        width = long_value(256)
        height = long_value(257)
        pixel_offset = long_value(273)
        pixel_length = long_value(279)
        compression = short_value(259)
        photometric = short_value(262)
        samples = short_value(277)
        planar = short_value(284)
    except KeyError as error:
        raise ValueError(f"managed TIFF tag missing: {error}") from error
    expected_length = width * height * 3 * 2
    if (
        width <= 0
        or height <= 0
        or compression != 1
        or photometric != 2
        or samples != 3
        or planar != 1
        or pixel_length != expected_length
        or pixel_offset + pixel_length > len(payload)
    ):
        raise ValueError("managed TIFF pixel layout is invalid")
    pixels = np.frombuffer(
        payload[pixel_offset : pixel_offset + pixel_length],
        dtype="<u2",
    ).reshape(height, width, 3)
    return np.transpose(pixels.astype(np.float32) / 65535.0, (2, 0, 1))


def read_managed_edit_tiff(path: Path) -> np.ndarray:
    """Decode editable RGB from a TIFF written by this managed exporter."""

    return _read_managed_edit_tiff(Path(path))


def _managed_derivative_pixel_chain(
    source_pixel_sha256: str,
    expected: np.ndarray,
    decoded: np.ndarray,
) -> Dict[str, Any]:
    expected_pixels = np.ascontiguousarray(expected, dtype=np.float32)
    decoded_pixels = np.ascontiguousarray(decoded, dtype=np.float32)
    expected_sha256 = pixel_sha256(expected_pixels)
    decoded_sha256 = pixel_sha256(decoded_pixels)
    return {
        "schema": "starun.managed-output-pixel-chain.v1",
        "accepted": bool(
            expected_pixels.shape == decoded_pixels.shape
            and np.array_equal(expected_pixels, decoded_pixels)
            and expected_sha256 == decoded_sha256
        ),
        "source_pixel_sha256": source_pixel_sha256,
        "source_pixel_sha256_method": DECODED_PIXEL_SHA256_METHOD,
        "expected_pixel_sha256": expected_sha256,
        "decoded_pixel_sha256": decoded_sha256,
        "decoded_pixel_sha256_method": (
            "canonical_uint16_top_down_float32_chw_v1"
        ),
    }


def _sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _requested_formats(value: str) -> set[str]:
    requested = {
        item.strip().lower()
        for item in str(value or "all").split(",")
        if item.strip()
    }
    if not requested or "all" in requested:
        requested = {"fit", "tif", "png"}
    if "tiff" in requested:
        requested.add("tif")
    return requested


def export_managed_outputs(
    image: Any,
    *,
    work_dir: Path,
    base_filename: str,
    output_format: str,
    scientific_paths: Iterable[Path] = (),
    icc_profile_bytes: Optional[bytes] = None,
    target_type: str = "",
    stars_required: bool = False,
    star_reference: Optional[Dict[str, Any]] = None,
    star_visibility_config: Any = None,
    display_contract: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Export independently audited display/edit assets without touching FITS."""
    root = Path(work_dir)
    requested = _requested_formats(output_format)
    scientific = [Path(path) for path in scientific_paths]
    hashes_before = {
        str(path): digest
        for path in scientific
        if (digest := _sha256(path)) is not None
    }
    artifacts: list[Dict[str, Any]] = []
    issues: list[str] = []
    display_visibility: Optional[Dict[str, Any]] = None
    display_transform: Optional[Dict[str, Any]] = None
    source_display_rgb = _as_display_rgb(image)
    source_bottom_up = np.flip(source_display_rgb, axis=1)
    source_pixel_sha256 = canonical_decoded_pixel_sha256(source_bottom_up)

    if "png" in requested:
        display_path = root / f"{base_filename}_display_srgb.png"
        wrote_display = False
        try:
            # Never let a stale derivative from an earlier attempt satisfy the
            # current run after a failed visibility audit.
            display_path.unlink(missing_ok=True)
            source_rgb = source_display_rgb.copy()
            input_visibility = audit_display_visibility(
                source_rgb,
                target_type=target_type,
                stars_required=stars_required,
                star_reference=star_reference,
                pixel_coordinate_domain="display_array_top_down",
                star_visibility_config=star_visibility_config,
            )
            display_rgb = source_rgb
            if display_contract is not None:
                if not display_rendition.validate_review_contract(
                    display_contract
                ):
                    raise ValueError("required Review display contract is invalid")
                display_rgb = display_rendition.apply_review_contract(
                    source_rgb,
                    display_contract,
                )
                display_transform = dict(display_contract)
            else:
                display_transform = {
                    "name": "preserve_accepted_nonlinear_rendering",
                    "observer_only": True,
                    "source_pixels_changed": False,
                    "derivative_pixels_changed": False,
                }
                if (
                    not bool(input_visibility.get("passed", False))
                    and input_visibility.get("exposure_state") == "underexposed"
                ):
                    display_rgb, display_transform = _linked_visibility_stretch(
                        source_rgb
                    )
                elif not bool(input_visibility.get("passed", False)):
                    display_transform = {
                        "name": "preserve_unmappable_or_overexposed_source",
                        "observer_only": True,
                        "source_pixels_changed": False,
                        "derivative_pixels_changed": False,
                        "reason": str(
                            input_visibility.get("exposure_state") or "unknown"
                        ),
                    }
            _write_display_rgb_png(display_path, display_rgb)
            wrote_display = True
            expected_display = canonical_managed_derivative_pixels(
                np.flip(display_rgb, axis=1)
            )
            decoded_display = _read_managed_display_png(display_path)
            display_pixel_chain = _managed_derivative_pixel_chain(
                source_pixel_sha256,
                expected_display,
                decoded_display,
            )
            if display_pixel_chain["accepted"] is not True:
                raise ValueError("managed display decoded pixel identity mismatch")
            final_visibility = audit_display_visibility(
                decoded_display,
                target_type=target_type,
                stars_required=stars_required,
                star_reference=star_reference,
                pixel_coordinate_domain="display_array_top_down",
                star_visibility_config=star_visibility_config,
            )
            final_visibility["source"] = "decoded_final_png"
            final_visibility["path"] = str(display_path)
            display_visibility = {
                "status": final_visibility["status"],
                "passed": bool(final_visibility["passed"]),
                "input_pixels": input_visibility,
                "final_png": final_visibility,
                "transform": display_transform,
                "brightening_decision": {
                    "allowed": bool(
                        input_visibility.get("exposure_state")
                        == "underexposed"
                    ),
                    "input_exposure_state": str(
                        input_visibility.get("exposure_state") or "unmappable"
                    ),
                    "refused_reason": (
                        None
                        if input_visibility.get("exposure_state")
                        == "underexposed"
                        else "brightening_requires_underexposed_input"
                    ),
                    "lower_bounds": dict(
                        (
                            input_visibility.get("checks", {}).get(
                                "pixel_brightness",
                                {},
                            )
                            or {}
                        ).get("thresholds")
                        or {}
                    ),
                    "upper_bounds": dict(
                        (
                            input_visibility.get("checks", {}).get(
                                "exposure_upper_bounds",
                                {},
                            )
                            or {}
                        ).get("thresholds")
                        or {}
                    ),
                },
            }
            if not bool(final_visibility["passed"]):
                display_path.unlink(missing_ok=True)
                wrote_display = False
                failed = ",".join(final_visibility.get("failed_checks") or [])
                issues.append(
                    "display_png_visibility_failed: "
                    + (failed or "unknown_visibility_failure")
                )
                artifacts.append(
                    {
                        "role": "display",
                        "path": str(display_path),
                        "name": display_path.name,
                        "format": "png",
                        "status": "rejected_not_published",
                        "reason": "final PNG pixel visibility audit failed",
                        "visibility": final_visibility,
                        "display_transform": display_transform,
                    }
                )
            else:
                artifacts.append(
                    {
                        "role": "display",
                        "path": str(display_path),
                        "name": display_path.name,
                        "format": "png",
                        "bit_depth": 16,
                        "color_space": "sRGB IEC61966-2.1",
                        "profile_mechanism": "sRGB+gAMA+cHRM chunks",
                        "status": "written",
                        "sha256": _sha256(display_path),
                        "pixel_chain": display_pixel_chain,
                        "visibility": final_visibility,
                        "display_transform": display_transform,
                    }
                )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            if wrote_display:
                display_path.unlink(missing_ok=True)
            display_visibility = display_visibility or {
                "status": "unavailable",
                "passed": False,
                "error": str(error),
            }
            issues.append(f"display_png_failed: {error}")

    profile_source = None
    profile = icc_profile_bytes
    if "tif" in requested:
        if profile is None:
            profile, profile_path = find_srgb_icc_profile()
            profile_source = str(profile_path) if profile_path else None
        else:
            profile_source = "caller_supplied"
        if profile is None:
            issues.append("editable_tiff_failed: sRGB ICC profile unavailable")
        else:
            edit_path = root / f"{base_filename}_edit_srgb.tif"
            try:
                write_managed_edit_tiff(
                    edit_path,
                    image,
                    icc_profile=profile,
                )
                expected_edit = canonical_managed_derivative_pixels(image)
                decoded_edit = _read_managed_edit_tiff(edit_path)
                edit_pixel_chain = _managed_derivative_pixel_chain(
                    source_pixel_sha256,
                    expected_edit,
                    decoded_edit,
                )
                if edit_pixel_chain["accepted"] is not True:
                    edit_path.unlink(missing_ok=True)
                    raise ValueError(
                        "managed editable decoded pixel identity mismatch"
                    )
                artifacts.append(
                    {
                        "role": "editable",
                        "path": str(edit_path),
                        "name": edit_path.name,
                        "format": "tiff",
                        "bit_depth": 16,
                        "color_space": "sRGB IEC61966-2.1",
                        "profile_mechanism": "embedded ICC tag 34675",
                        "icc_profile_source": profile_source,
                        "icc_profile_bytes": len(profile),
                        "status": "written",
                        "sha256": _sha256(edit_path),
                        "pixel_chain": edit_pixel_chain,
                    }
                )
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                issues.append(f"editable_tiff_failed: {error}")

    hashes_after = {
        str(path): digest
        for path in scientific
        if (digest := _sha256(path)) is not None
    }
    scientific_unchanged = hashes_before == hashes_after
    if not scientific_unchanged:
        issues.append("scientific_archive_hash_changed")
    expected_roles = {
        role
        for format_name, role in (("png", "display"), ("tif", "editable"))
        if format_name in requested
    }
    written_roles = {
        str(artifact["role"])
        for artifact in artifacts
        if artifact.get("status") == "written"
    }
    ready = expected_roles.issubset(written_roles) and scientific_unchanged
    return {
        "schema": MANAGED_OUTPUT_SCHEMA,
        "status": "ready" if ready else "partial",
        "ready": ready,
        "mode": "independent_managed_derivatives",
        "source_pixels": {
            "checkpoint": "stage10_final.fit",
            "pixel_sha256": source_pixel_sha256,
            "pixel_sha256_method": DECODED_PIXEL_SHA256_METHOD,
            "transform": "derivative_specific",
            "display_transform": display_transform,
            "editable_transform": "preserve_accepted_nonlinear_rendering",
            "orientation": "fits_bottom_up_to_display_top_down",
            "working_primaries_assumption": "sRGB",
        },
        "display_visibility": display_visibility,
        "display_rendition_contract": (
            dict(display_contract)
            if isinstance(display_contract, dict)
            else None
        ),
        "scientific_archive": {
            "policy": "never_rewrite",
            "hashes_before": hashes_before,
            "hashes_after": hashes_after,
            "unchanged": scientific_unchanged,
        },
        "artifacts": artifacts,
        "issues": issues,
    }


__all__ = [
    "MANAGED_OUTPUT_SCHEMA",
    "audit_display_visibility",
    "canonical_managed_derivative_pixels",
    "export_managed_outputs",
    "find_srgb_icc_profile",
    "read_managed_display_png",
    "read_managed_edit_tiff",
    "write_managed_display_png",
    "write_managed_edit_tiff",
]
