"""Save/export helpers for the SeeStar processing pipeline."""
import json
import re
import struct
import zlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from sirilpy.exceptions import CommandError, DataError, SirilError


LEGACY_STAGE_READ_ALIASES = {
    "stage4_color": ("stage4_colorbalanced",),
    "stage4_colorbalanced": ("stage4_color",),
    "stage5_linear": ("stage5_denoised",),
    "stage7_starless": ("stage6_starless",),
    "stage6_starless": ("stage7_starless",),
    "stage7_starless_repaired": ("stage6_starless_repaired",),
    "stage6_starless_repaired": ("stage7_starless_repaired",),
    "stage6_input": ("stage7_input",),
    "stage7_input": ("stage6_input",),
}

LEGACY_STAGE_JSON_READ_ALIASES = {
    "stage7_quality.json": ("stage6_starless_quality.json",),
    "stage6_starless_quality.json": ("stage7_quality.json",),
    "pre_starless_gate_report.json": ("stage7_5_pre_starless_gate_report.json",),
    "stage7_5_pre_starless_gate_report.json": ("pre_starless_gate_report.json",),
}


def _stage_output_aliases(stem: str) -> Tuple[str, ...]:
    """Return legacy names for read/migration code; never write them."""
    aliases = list(LEGACY_STAGE_READ_ALIASES.get(stem, ()))
    return tuple(aliases)


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    length = struct.pack(">I", len(payload))
    crc = zlib.crc32(tag)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    return length + tag + payload + struct.pack(">I", crc)


def write_png_rgb16(path: Path, rgb: np.ndarray) -> None:
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ValueError(f"expected RGB CHW array, got shape={arr.shape}")

    arr = np.clip(arr, 0.0, 1.0)
    h, w = arr.shape[1], arr.shape[2]
    if h <= 0 or w <= 0:
        raise ValueError(f"invalid image size: {w}x{h}")

    # PNG 16-bit data must be network-byte-order (big-endian).
    chw_u16 = np.round(arr * 65535.0).astype(">u2")
    hwc_u16 = np.transpose(chw_u16, (1, 2, 0))

    compressor = zlib.compressobj(level=6)
    idat_parts: List[bytes] = []
    for row in hwc_u16:
        row_bytes = b"\x00" + row.tobytes(order="C")
        compressed = compressor.compress(row_bytes)
        if compressed:
            idat_parts.append(compressed)
    tail = compressor.flush()
    if tail:
        idat_parts.append(tail)
    idat = b"".join(idat_parts)

    ihdr = struct.pack(">IIBBBBB", w, h, 16, 2, 0, 0, 0)
    png_data = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", idat)
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(png_data)


def save_stage_output(cmd_with_check: Callable[..., Any], log: Any, stem: str) -> bool:
    """Save per-stage FITS snapshot with stage* naming."""
    try:
        cmd_with_check("save", stem)
        log.info(f"阶段产物已保存: {stem}.fit")
        return True
    except (CommandError, DataError, SirilError, OSError, RuntimeError) as e:
        log.warn(f"阶段产物保存失败 ({stem}): {e}")
        return False


def write_stage_json(
    process_dir: Optional[Path],
    log: Any,
    filename: str,
    payload: Dict[str, Any],
) -> None:
    if not process_dir:
        return
    path = process_dir / filename
    try:
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        path.write_text(
            text,
            encoding="utf-8",
        )
    except OSError as e:
        log.warn(f"写入阶段 JSON 失败 ({filename}): {e}")


def write_ai_raw_response(
    process_dir: Optional[Path],
    log: Any,
    counter: int,
    short_text: Callable[[str, int], str],
    stage_name: str,
    *,
    endpoint_url: str,
    temperature: float,
    json_mode: bool,
    response_obj: Optional[Dict[str, Any]] = None,
    content: Optional[str] = None,
    error_text: Optional[str] = None,
) -> int:
    next_counter = counter + 1
    if not process_dir:
        return next_counter

    safe_stage = re.sub(r"[^A-Za-z0-9_.-]+", "_", stage_name).strip("_") or "stage"
    stem = f"ai_raw_{next_counter:03d}_{safe_stage}"
    payload = {
        "stage": stage_name,
        "endpoint": endpoint_url,
        "temperature": temperature,
        "json_mode": json_mode,
        "error": error_text,
        "content_preview": short_text(content or "", 2000),
        "response": response_obj,
    }
    try:
        (process_dir / f"{stem}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if content:
            (process_dir / f"{stem}.txt").write_text(content, encoding="utf-8")
    except OSError as e:
        log.warn(f"写入 AI raw response 失败 ({stage_name}): {e}")
    return next_counter


def export_final_outputs(
    cmd_with_check: Callable[..., Any],
    log: Any,
    *,
    base_filename: str,
    fit_filename: Optional[str],
    fallback_base: str,
    fallback_fit_base: str,
    output_format: str = "all",
    png_preview_stretch: bool = True,
    status: str,
    messages: List[str],
    export_report: Optional[Dict[str, Any]] = None,
) -> Tuple[str, List[str]]:
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
    report: Dict[str, Any] = {
        "schema": "seestar.stage10-export.v1",
        "requested_formats": sorted(requested),
        "outputs": {},
        "fallback_used": False,
        "fallback_formats": [],
    }

    if "tif" in requested:
        tif_report: Dict[str, Any] = {
            "primary": f"{base_filename}.tif",
            "fallback": f"{fallback_base}.tif",
        }
        report["outputs"]["tif"] = tif_report
        log.info("导出高质量 TIFF...")
        try:
            cmd_with_check("savetif", base_filename, "-astro")
            log.info("TIFF 已导出")
            tif_report.update(status="primary", selected=tif_report["primary"])
        except (CommandError, DataError, SirilError, OSError, RuntimeError) as e:
            tif_report["primary_error"] = str(e)
            log.warn(f"TIFF 导出失败: {e}")
            try:
                cmd_with_check("savetif", fallback_base, "-astro")
                log.info(f"TIFF 已导出: {fallback_base}.tif")
                tif_report.update(status="fallback", selected=tif_report["fallback"])
                report["fallback_used"] = True
                report["fallback_formats"].append("tif")
            except (CommandError, DataError, SirilError, OSError, RuntimeError) as fallback_error:
                tif_report.update(status="failed", fallback_error=str(fallback_error))
                log.error("TIFF 导出完全失败")
                status = "degraded"
                messages.append("TIFF 导出完全失败")

    if "fit" in requested:
        primary_fit = fit_filename or (base_filename + "_final")
        fit_report: Dict[str, Any] = {
            "primary": f"{primary_fit}.fit",
            "fallback": f"{fallback_fit_base}.fit",
        }
        report["outputs"]["fit"] = fit_report
        log.info("保存 FITS 存档...")
        try:
            cmd_with_check("save", primary_fit)
            log.info("FITS 存档已保存")
            fit_report.update(status="primary", selected=fit_report["primary"])
        except (CommandError, DataError, SirilError, OSError, RuntimeError) as error:
            fit_report["primary_error"] = str(error)
            try:
                cmd_with_check("save", fallback_fit_base)
                log.info(f"FITS 存档已保存: {fallback_fit_base}.fit")
                fit_report.update(status="fallback", selected=fit_report["fallback"])
                report["fallback_used"] = True
                report["fallback_formats"].append("fit")
            except (CommandError, DataError, SirilError, OSError, RuntimeError) as fallback_error:
                fit_report.update(status="failed", fallback_error=str(fallback_error))
                log.error("FITS 存档保存失败")
                status = "degraded"
                messages.append("FITS 存档保存失败")

    if "png" in requested:
        png_report: Dict[str, Any] = {
            "primary": f"{base_filename}.png",
            "fallback": f"{fallback_base}.png",
            "preview_stretch_requested": bool(png_preview_stretch),
        }
        report["outputs"]["png"] = png_report
        log.info("导出 PNG 预览...")
        if png_preview_stretch:
            try:
                cmd_with_check("autostretch", "-linked")
                messages.append("PNG preview stretch applied (linked diagnostic fallback)")
                png_report["preview_stretch_status"] = "applied"
            except (CommandError, DataError, SirilError, OSError, RuntimeError) as e:
                log.warn(f"PNG 预览拉伸跳过: {e}")
                messages.append(f"PNG preview stretch failed: {e}")
                png_report.update(
                    preview_stretch_status="failed",
                    preview_stretch_error=str(e),
                )
        else:
            png_report["preview_stretch_status"] = "not_required"
            messages.append(
                "PNG preview uses accepted nonlinear Stage7 rendering; "
                "second autostretch skipped"
            )
        try:
            cmd_with_check("savepng", base_filename)
            log.info("PNG 已导出")
            png_report.update(status="primary", selected=png_report["primary"])
        except (CommandError, DataError, SirilError, OSError, RuntimeError) as e:
            png_report["primary_error"] = str(e)
            log.warn(f"PNG 导出失败: {e}")
            try:
                cmd_with_check("savepng", fallback_base)
                log.info(f"PNG 已导出: {fallback_base}.png")
                png_report.update(status="fallback", selected=png_report["fallback"])
                report["fallback_used"] = True
                report["fallback_formats"].append("png")
            except (CommandError, DataError, SirilError, OSError, RuntimeError) as fallback_error:
                png_report.update(status="failed", fallback_error=str(fallback_error))
                log.error("PNG 导出完全失败")
                status = "degraded"
                messages.append("PNG 导出完全失败")

    if export_report is not None:
        export_report.clear()
        export_report.update(report)
    return status, messages
