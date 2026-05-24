"""Save/export helpers for the SeeStar processing pipeline."""
import json
import re
import struct
import zlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


STAGE_OUTPUT_ALIASES = {
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

STAGE_JSON_ALIASES = {
    "stage7_quality.json": ("stage6_starless_quality.json",),
    "stage6_starless_quality.json": ("stage7_quality.json",),
    "pre_starless_gate_report.json": ("stage7_5_pre_starless_gate_report.json",),
    "stage7_5_pre_starless_gate_report.json": ("pre_starless_gate_report.json",),
}


def _stage_output_aliases(stem: str) -> Tuple[str, ...]:
    aliases = list(STAGE_OUTPUT_ALIASES.get(stem, ()))
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
        for alias in _stage_output_aliases(stem):
            try:
                cmd_with_check("save", alias)
                log.info(f"阶段产物已保存: {alias}.fit")
            except Exception as alias_error:
                log.warn(f"阶段产物兼容别名保存失败 ({alias}): {alias_error}")
        return True
    except Exception as e:
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
        for alias in STAGE_JSON_ALIASES.get(filename, ()):
            (process_dir / alias).write_text(
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
    fallback_base: str,
    fallback_fit_base: str,
    output_format: str = "all",
    status: str,
    messages: List[str],
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

    if "tif" in requested:
        log.info("导出高质量 TIFF...")
        try:
            cmd_with_check("savetif", base_filename, "-astro")
            log.info("TIFF 已导出")
        except Exception as e:
            log.warn(f"TIFF 导出失败: {e}")
            try:
                cmd_with_check("savetif", fallback_base, "-astro")
                log.info(f"TIFF 已导出: {fallback_base}.tif")
            except Exception:
                log.error("TIFF 导出完全失败")
                status = "degraded"
                messages.append("TIFF 导出完全失败")

    if "fit" in requested:
        log.info("保存 FITS 存档...")
        try:
            cmd_with_check("save", base_filename + "_final")
            log.info("FITS 存档已保存")
        except Exception:
            try:
                cmd_with_check("save", fallback_fit_base)
                log.info(f"FITS 存档已保存: {fallback_fit_base}.fit")
            except Exception:
                log.error("FITS 存档保存失败")
                status = "degraded"
                messages.append("FITS 存档保存失败")

    if "png" in requested:
        log.info("导出 PNG 预览...")
        try:
            cmd_with_check("autostretch")
            messages.append("PNG preview stretch applied")
        except Exception as e:
            log.warn(f"PNG 预览拉伸跳过: {e}")
            messages.append(f"PNG preview stretch failed: {e}")
        try:
            cmd_with_check("savepng", base_filename)
            log.info("PNG 已导出")
        except Exception as e:
            log.warn(f"PNG 导出失败: {e}")
            try:
                cmd_with_check("savepng", fallback_base)
                log.info(f"PNG 已导出: {fallback_base}.png")
            except Exception:
                log.error("PNG 导出完全失败")
                messages.append("PNG 导出完全失败")

    return status, messages
