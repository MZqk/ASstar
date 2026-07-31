"""Independent color-managed Stage 10 display and editable exports."""
from __future__ import annotations

import hashlib
import struct
import zlib
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np


MANAGED_OUTPUT_SCHEMA = "seestar.managed-output.v1"
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
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


def write_managed_display_png(path: Path, image: Any) -> Path:
    """Write a 16-bit RGB PNG with explicit sRGB transfer/chromaticity metadata."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    rgb = _as_display_rgb(image)
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
    software = b"Seestar AstroSuite managed export\x00"

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
) -> Dict[str, Any]:
    """Export independent display/edit assets and prove FITS files were untouched."""
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

    if "png" in requested:
        display_path = root / f"{base_filename}_display_srgb.png"
        try:
            write_managed_display_png(display_path, image)
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
                }
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
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
            "transform": "preserve_accepted_nonlinear_rendering",
            "orientation": "fits_bottom_up_to_display_top_down",
            "working_primaries_assumption": "sRGB",
        },
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
    "export_managed_outputs",
    "find_srgb_icc_profile",
    "write_managed_display_png",
    "write_managed_edit_tiff",
]
