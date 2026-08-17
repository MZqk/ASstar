"""Report-only inspection of Stage 10 export color metadata."""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


OUTPUT_COLOR_SCHEMA = "starun.output-color-manifest.v1"


def _requested_formats(output_format: str) -> set[str]:
    requested = {
        item.strip().lower()
        for item in str(output_format or "all").split(",")
        if item.strip()
    }
    if not requested or "all" in requested:
        requested = {"tif", "png", "fit"}
    if "tiff" in requested:
        requested.add("tif")
    if "fits" in requested:
        requested.add("fit")
    return requested & {"tif", "png", "fit"}


def _inspect_png(path: Path) -> Dict[str, Any]:
    chunks: list[str] = []
    bit_depth = None
    color_type = None
    with path.open("rb") as handle:
        if handle.read(8) != b"\x89PNG\r\n\x1a\n":
            raise ValueError("invalid PNG signature")
        while True:
            header = handle.read(8)
            if len(header) < 8:
                break
            length, tag = struct.unpack(">I4s", header)
            name = tag.decode("ascii", errors="replace")
            chunks.append(name)
            if name == "IHDR":
                payload = handle.read(min(length, 13))
                if length > len(payload):
                    handle.seek(length - len(payload), 1)
            else:
                payload = b""
                handle.seek(length, 1)
            if len(handle.read(4)) != 4:
                raise ValueError("truncated PNG chunk")
            if name == "IHDR" and len(payload) >= 10:
                _width, _height, bit_depth, color_type = struct.unpack(
                    ">IIBB",
                    payload[:10],
                )
            if name == "IEND":
                break
    has_icc = "iCCP" in chunks
    has_srgb = "sRGB" in chunks
    return {
        "format": "png",
        "bit_depth": bit_depth,
        "color_type": color_type,
        "metadata_chunks": [
            name for name in chunks if name in {"iCCP", "sRGB", "gAMA", "cHRM"}
        ],
        "icc_profile_present": has_icc,
        "srgb_declared": has_srgb,
        "display_profile_verified": bool(has_icc or has_srgb),
    }


def _read_tiff_values(
    handle,
    *,
    endian: str,
    value_type: int,
    count: int,
    value_or_offset: bytes,
) -> list[int]:
    type_sizes = {1: 1, 3: 2, 4: 4}
    size = type_sizes.get(value_type)
    if size is None or count <= 0:
        return []
    total_size = size * count
    if total_size <= 4:
        payload = value_or_offset[:total_size]
    else:
        offset = struct.unpack(endian + "I", value_or_offset)[0]
        position = handle.tell()
        handle.seek(offset)
        payload = handle.read(total_size)
        handle.seek(position)
    if len(payload) < total_size:
        return []
    format_code = {1: "B", 3: "H", 4: "I"}[value_type]
    return list(struct.unpack(endian + format_code * count, payload))


def _inspect_tiff(path: Path) -> Dict[str, Any]:
    with path.open("rb") as handle:
        byte_order = handle.read(2)
        endian = "<" if byte_order == b"II" else ">" if byte_order == b"MM" else ""
        if not endian:
            raise ValueError("invalid TIFF byte order")
        magic = struct.unpack(endian + "H", handle.read(2))[0]
        if magic != 42:
            return {
                "format": "tiff",
                "variant": "bigtiff_or_unknown" if magic == 43 else "unknown",
                "icc_profile_present": False,
                "profile_check_supported": False,
            }
        ifd_offset = struct.unpack(endian + "I", handle.read(4))[0]
        handle.seek(ifd_offset)
        entry_count_data = handle.read(2)
        if len(entry_count_data) != 2:
            raise ValueError("truncated TIFF IFD")
        entry_count = struct.unpack(endian + "H", entry_count_data)[0]
        bits_per_sample: list[int] = []
        icc_profile_bytes = 0
        for _ in range(entry_count):
            entry = handle.read(12)
            if len(entry) != 12:
                break
            tag, value_type, count = struct.unpack(endian + "HHI", entry[:8])
            value_or_offset = entry[8:12]
            if tag == 258:
                bits_per_sample = _read_tiff_values(
                    handle,
                    endian=endian,
                    value_type=value_type,
                    count=count,
                    value_or_offset=value_or_offset,
                )
            elif tag == 34675:
                icc_profile_bytes = int(count)
        return {
            "format": "tiff",
            "variant": "classic",
            "bits_per_sample": bits_per_sample,
            "icc_profile_present": icc_profile_bytes > 0,
            "icc_profile_bytes": icc_profile_bytes,
            "profile_check_supported": True,
        }


def _inspect_fits(path: Path) -> Dict[str, Any]:
    with path.open("rb") as handle:
        header = handle.read(1024 * 1024)
    cards = [
        header[index : index + 80].decode("ascii", errors="replace")
        for index in range(0, min(len(header), 2880 * 32), 80)
    ]
    bitpix = None
    naxis = None
    for card in cards:
        keyword = card[:8].strip()
        if keyword in {"BITPIX", "NAXIS"} and "=" in card:
            raw = card.split("=", 1)[1].split("/", 1)[0].strip()
            try:
                value = int(raw)
            except ValueError:
                continue
            if keyword == "BITPIX":
                bitpix = value
            else:
                naxis = value
        if keyword == "END":
            break
    icc_extension = b"ICCPROFILE" in header.upper() or b"ICCProfile" in header
    return {
        "format": "fits",
        "bitpix": bitpix,
        "naxis": naxis,
        "icc_extension_detected": icc_extension,
        "scientific_archive": True,
        "display_profile_required": False,
    }


def inspect_output_artifact(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    base = {
        "path": str(path),
        "name": path.name,
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
    }
    if not path.is_file():
        return base
    try:
        if suffix == ".png":
            detail = _inspect_png(path)
        elif suffix in {".tif", ".tiff"}:
            detail = _inspect_tiff(path)
        elif suffix in {".fit", ".fits"}:
            detail = _inspect_fits(path)
        else:
            detail = {"format": "unknown"}
        return {**base, "inspection_status": "ok", **detail}
    except (OSError, RuntimeError, TypeError, ValueError, struct.error) as error:
        return {
            **base,
            "inspection_status": "failed",
            "error": str(error),
        }


def _recent_outputs(
    work_dir: Path,
    *,
    names: Iterable[str],
    extensions: Iterable[str],
    exported_after: Optional[float],
) -> list[Path]:
    candidates = {
        work_dir / f"{name}.{extension}"
        for name in names
        if name
        for extension in extensions
    }
    if exported_after is not None:
        for extension in extensions:
            for path in work_dir.glob(f"*.{extension}"):
                try:
                    if path.stat().st_mtime >= exported_after - 1.0:
                        candidates.add(path)
                except OSError:
                    continue
    return sorted(
        (path for path in candidates if path.is_file()),
        key=lambda path: path.name,
    )


def build_output_color_manifest(
    *,
    work_dir: Path,
    base_filename: str,
    fit_filename: Optional[str],
    fallback_base: str,
    fallback_fit_base: str,
    output_format: str,
    channel_semantics: str,
    review_only: bool,
    exported_after: Optional[float] = None,
    managed_export_report: Optional[Dict[str, Any]] = None,
    source_color_contract: Optional[Dict[str, Any]] = None,
    display_rendition_contract: Optional[Dict[str, Any]] = None,
    export_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Inspect current exports without converting, tagging, or rewriting them."""
    requested = _requested_formats(output_format)
    names = {
        str(base_filename or ""),
        str(fallback_base or ""),
        str(fit_filename or ""),
        str(fallback_fit_base or ""),
    }
    for base in (str(base_filename or ""), str(fallback_base or "")):
        if base:
            names.add(base + "_display_srgb")
            names.add(base + "_edit_srgb")
    extensions = sorted(requested)
    if isinstance(export_report, dict):
        actual_paths: set[Path] = set()
        for output in (export_report.get("outputs") or {}).values():
            if not isinstance(output, dict):
                continue
            selected = str(output.get("selected") or "")
            if selected and str(output.get("status") or "") in {
                "primary",
                "fallback",
                "managed_review",
            }:
                candidate = Path(work_dir) / selected
                if candidate.is_file():
                    actual_paths.add(candidate)
        for artifact in (managed_export_report or {}).get("artifacts") or []:
            if not isinstance(artifact, dict) or artifact.get("status") != "written":
                continue
            candidate = Path(str(artifact.get("path") or ""))
            if candidate.is_file():
                actual_paths.add(candidate)
        paths = sorted(actual_paths, key=lambda path: path.name)
    else:
        paths = _recent_outputs(
            Path(work_dir),
            names=names,
            extensions=extensions,
            exported_after=exported_after,
        )
    artifacts = [inspect_output_artifact(path) for path in paths]
    png_artifacts = [item for item in artifacts if item.get("format") == "png"]
    tiff_artifacts = [item for item in artifacts if item.get("format") == "tiff"]
    fits_artifacts = [item for item in artifacts if item.get("format") == "fits"]
    managed_png_artifacts = [
        item
        for item in png_artifacts
        if str(item.get("name") or "").endswith("_display_srgb.png")
    ]
    managed_tiff_artifacts = [
        item
        for item in tiff_artifacts
        if str(item.get("name") or "").endswith("_edit_srgb.tif")
    ]
    requested_display = "png" in requested
    requested_edit = "tif" in requested
    display_visibility_ok: Optional[bool] = None
    if managed_export_report is not None:
        display_profiles_ok = bool(managed_png_artifacts) and all(
            bool(item.get("display_profile_verified"))
            for item in managed_png_artifacts
        )
        edit_profiles_ok = bool(managed_tiff_artifacts) and all(
            bool(item.get("icc_profile_present"))
            for item in managed_tiff_artifacts
        )
        written_managed_displays = [
            item
            for item in (managed_export_report.get("artifacts") or [])
            if item.get("role") == "display" and item.get("status") == "written"
        ]
        display_visibility_ok = (
            bool(written_managed_displays)
            and all(
                bool((item.get("visibility") or {}).get("passed", False))
                for item in written_managed_displays
            )
            if requested_display
            else None
        )
    else:
        display_profiles_ok = bool(png_artifacts) and all(
            bool(item.get("display_profile_verified"))
            for item in png_artifacts
        )
        edit_profiles_ok = bool(tiff_artifacts) and all(
            bool(item.get("icc_profile_present")) for item in tiff_artifacts
        )
    ready = bool(
        (not requested_display or display_profiles_ok)
        and (not requested_edit or edit_profiles_ok)
        and (
            managed_export_report is None
            or not requested_display
            or display_visibility_ok is True
        )
    )
    color_contract = dict(source_color_contract or {})
    working_color_state = color_contract.get("working_color_state") or {}
    working_color_state = (
        dict(working_color_state)
        if isinstance(working_color_state, dict)
        else {}
    )
    source_profile_verified = bool(
        working_color_state.get("profile_verified", False)
    )
    conversion_lineage_verified = bool(
        working_color_state.get("conversion_lineage_verified", False)
    )

    return {
        "schema": OUTPUT_COLOR_SCHEMA,
        "mode": (
            "managed_derivatives_active"
            if managed_export_report is not None
            else "report_only"
        ),
        "rewrote_outputs": False,
        "managed_export": managed_export_report,
        "channel_semantics": str(channel_semantics or "unknown"),
        "source_color_contract": color_contract or None,
        "display_rendition_contract": (
            dict(display_rendition_contract)
            if isinstance(display_rendition_contract, dict)
            else None
        ),
        "color_state_disclosure": {
            "source_profile_verified": source_profile_verified,
            "source_to_target_conversion_lineage_verified": (
                conversion_lineage_verified
            ),
            "source_to_srgb_pixel_conversion_performed_by_manifest": False,
            "managed_container_metadata_action": (
                "direct pixel quantization plus sRGB metadata/profile assignment"
                if managed_export_report is not None
                else "inspection_only"
            ),
            "limitation": (
                None
                if source_profile_verified and conversion_lineage_verified
                else "source primaries/TRC/white point are unverified; an sRGB tag "
                "alone does not prove a source-to-sRGB pixel conversion"
            ),
        },
        "review_only": bool(review_only),
        "requested_formats": sorted(requested),
        "desired_contract": {
            "fits": {
                "role": "scientific_archive",
                "color_transform": "none",
                "icc_required": False,
            },
            "tif": {
                "role": "editable_16bit",
                "color_space": "sRGB IEC61966-2.1",
                "icc_required": True,
            },
            "png": {
                "role": "display",
                "color_space": "sRGB IEC61966-2.1",
                "icc_or_srgb_chunk_required": True,
            },
        },
        "artifacts": artifacts,
        "summary": {
            "artifact_count": len(artifacts),
            "png_count": len(png_artifacts),
            "tiff_count": len(tiff_artifacts),
            "fits_count": len(fits_artifacts),
            "managed_png_count": len(managed_png_artifacts),
            "managed_tiff_count": len(managed_tiff_artifacts),
            "display_profiles_verified": display_profiles_ok,
            "display_visibility_verified": display_visibility_ok,
            "editable_profiles_verified": edit_profiles_ok,
            "ready_for_future_managed_export": ready,
            "managed_export_ready": bool(
                managed_export_report is not None
                and managed_export_report.get("ready", False)
                and ready
            ),
            "activation_blockers": [
                reason
                for condition, reason in (
                    (
                        requested_display and not display_profiles_ok,
                        "PNG sRGB/ICC metadata missing or unverified",
                    ),
                    (
                        requested_edit and not edit_profiles_ok,
                        "TIFF ICC metadata missing or unverified",
                    ),
                    (
                        managed_export_report is not None
                        and requested_display
                        and display_visibility_ok is not True,
                        "PNG pixel brightness/subject/star visibility missing or failed",
                    ),
                )
                if condition
            ],
        },
        "runtime_note": (
            "Managed display/edit derivatives are independent; existing Siril "
            "exports and FITS scientific archives are not rewritten."
            if managed_export_report is not None
            else (
                "Output metadata audit only; existing Siril export commands and "
                "pixel buffers remain unchanged."
            )
        ),
    }


__all__ = [
    "build_output_color_manifest",
    "inspect_output_artifact",
]
