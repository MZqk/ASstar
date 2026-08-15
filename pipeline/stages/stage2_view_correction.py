"""Stage 2 view correction and crop."""
import math
from typing import List, Optional, Tuple

import numpy as np

from image_metrics import _to_rgb_float_fullres
from models import PipelineStage
from stage2_crop_detector import (
    detect_field_rotation_crop,
    detect_native_contour_crop,
)
from sirilpy.exceptions import CommandError, SirilError


def _line_edge_scores(gray: np.ndarray, rgb: np.ndarray, *, axis: int, black_threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if axis == 0:
        values = gray
        r, g, b = rgb[0], rgb[1], rgb[2]
    else:
        values = gray.T
        r, g, b = rgb[0].T, rgb[1].T, rgb[2].T

    black_ratio = np.mean(values <= black_threshold, axis=0)
    median_level = np.median(values, axis=0)
    chroma = np.mean(np.abs(np.stack([r, g, b], axis=0) - values[None, :, :]), axis=0)
    blue_excess = np.maximum(b - np.maximum(r, g), 0.0)
    red_excess = np.maximum(r - np.maximum(g, b), 0.0)
    color_cast = np.median(chroma + 0.65 * np.maximum(blue_excess, red_excess), axis=0)
    return black_ratio.astype(np.float32), median_level.astype(np.float32), color_cast.astype(np.float32)


def _combine_edge_evidence(hard_black: np.ndarray, *soft_evidence: np.ndarray) -> np.ndarray:
    """Require two soft signals unless a line has direct near-black coverage."""
    votes = np.zeros(np.asarray(hard_black).shape, dtype=np.uint8)
    for evidence in soft_evidence:
        votes += np.asarray(evidence, dtype=np.uint8)
    return np.asarray(hard_black, dtype=bool) | (votes >= 2)


def _first_stable_good_index(bad: np.ndarray, max_scan: int, stable_run: int) -> int:
    trim = 0
    good_run = 0
    for idx in range(max_scan):
        if bool(bad[idx]):
            trim = idx + 1
            good_run = 0
        else:
            good_run += 1
            if good_run >= stable_run:
                break
    return trim


def _rolling_median_1d(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    if window <= 3:
        return arr
    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    return np.asarray(
        [np.median(padded[idx : idx + window]) for idx in range(arr.size)],
        dtype=np.float32,
    )


def _detect_auto_edge_crop(pipeline, is_adaptive: bool = False) -> Tuple[Optional[Tuple[int, int, int, int]], str]:
    image_data = pipeline.siril.get_image_pixeldata(preview=False)
    shape = pipeline.siril.get_image_shape()
    if image_data is None or not shape:
        return None, "auto edge crop skipped: image data unavailable"
    rgb = _to_rgb_float_fullres(np.asarray(image_data))
    if rgb.ndim != 3 or rgb.shape[1] < 80 or rgb.shape[2] < 80:
        return None, "auto edge crop skipped: image too small"

    _channels, height, width = rgb.shape
    gray = (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]).astype(np.float32)
    center = gray[
        int(height * 0.25) : int(height * 0.75),
        int(width * 0.25) : int(width * 0.75),
    ]
    if center.size < 64:
        center = gray
    center_bg = center[center <= float(np.quantile(center, 0.45))]
    if center_bg.size < 64:
        center_bg = center.reshape(-1)
    bg_median = float(np.median(center_bg))
    bg_std = float(np.std(center_bg))
    center_median = float(np.median(center))
    black_threshold = max(0.0015, min(0.018, bg_median * 0.50))

    center_cast = _edge_color_cast_score(rgb[
        :,
        int(height * 0.25) : int(height * 0.75),
        int(width * 0.25) : int(width * 0.75),
    ])
    target = float(getattr(pipeline.cfg, "stage2_edge_black_target", 0.10))
    black_line_limit = max(0.015, min(0.08, target * 0.35))
    dark_level_limit = max(black_threshold * 1.25, bg_median - max(bg_std * 2.0, 0.004))
    cast_limit = max(
        float(getattr(pipeline.cfg, "stage2_edge_cast_absolute_max", 0.010)),
        center_cast
        * float(getattr(pipeline.cfg, "stage2_edge_cast_detect_ratio_max", 2.8)),
    )
    max_scan_x = max(4, int(width * 0.25))
    max_scan_y = max(4, int(height * 0.25))
    stable_run = max(6, int(min(width, height) * 0.006))

    # Score each side through the orthogonal center band. A top/bottom black
    # strip otherwise raises the black ratio of every column (and vice versa),
    # making a modest rectangular border look like a 25% crop on all sides.
    center_y = slice(int(height * 0.25), max(int(height * 0.75), int(height * 0.25) + 1))
    center_x = slice(int(width * 0.25), max(int(width * 0.75), int(width * 0.25) + 1))
    col_black, col_median, col_cast = _line_edge_scores(
        gray[center_y, :], rgb[:, center_y, :], axis=0, black_threshold=black_threshold
    )
    row_black, row_median, row_cast = _line_edge_scores(
        gray[:, center_x], rgb[:, :, center_x], axis=1, black_threshold=black_threshold
    )
    level_window = int(getattr(pipeline.cfg, "stage2_level_artifact_window", 81))
    row_level = _rolling_median_1d(row_median, level_window)
    col_level = _rolling_median_1d(col_median, level_window)
    row_high_limit = bg_median + max(bg_std * 2.30, 0.00008)
    col_high_limit = bg_median + max(bg_std * 2.30, 0.00008)
    col_low_limit = center_median - max(bg_std * 0.62, 0.000035)

    bad_cols = _combine_edge_evidence(
        col_black > black_line_limit,
        (col_median <= dark_level_limit) & (col_black > 0.004),
        col_cast > cast_limit,
        col_level > col_high_limit,
        col_level < col_low_limit,
    )
    bad_rows = _combine_edge_evidence(
        row_black > black_line_limit,
        (row_median <= dark_level_limit) & (row_black > 0.004),
        row_cast > cast_limit,
        row_level > row_high_limit,
    )

    left = _first_stable_good_index(bad_cols, max_scan_x, stable_run)
    right = _first_stable_good_index(bad_cols[::-1], max_scan_x, stable_run)
    top = _first_stable_good_index(bad_rows, max_scan_y, stable_run)
    bottom = _first_stable_good_index(bad_rows[::-1], max_scan_y, stable_run)

    # Corner clearance for field rotation triangles
    is_black = (gray <= black_threshold)
    corner_size = 5
    # Top-Left
    while left < max_scan_x and top < max_scan_y:
        patch = is_black[top:top+corner_size, left:left+corner_size]
        if patch.size > 0 and np.any(patch):
            left += 1; top += 1
        else:
            break
    # Top-Right
    while right < max_scan_x and top < max_scan_y:
        patch = is_black[top:top+corner_size, max(0, width-right-corner_size):width-right]
        if patch.size > 0 and np.any(patch):
            right += 1; top += 1
        else:
            break
    # Bottom-Left
    while left < max_scan_x and bottom < max_scan_y:
        patch = is_black[max(0, height-bottom-corner_size):height-bottom, left:left+corner_size]
        if patch.size > 0 and np.any(patch):
            left += 1; bottom += 1
        else:
            break
    # Bottom-Right
    while right < max_scan_x and bottom < max_scan_y:
        patch = is_black[max(0, height-bottom-corner_size):height-bottom, max(0, width-right-corner_size):width-right]
        if patch.size > 0 and np.any(patch):
            right += 1; bottom += 1
        else:
            break

    guard_band = int(getattr(pipeline.cfg, "stage2_guard_band_pixels", 3))
    if left > 0:
        left += guard_band
    if right > 0:
        right += guard_band
    if top > 0:
        top += guard_band
    if bottom > 0:
        bottom += guard_band

    if is_adaptive:
        max_extra_ratio = float(getattr(pipeline.cfg, "stage2_adaptive_edge_crop_max_extra", 0.035))
        limit_x = max(guard_band, int(width * max_extra_ratio))
        limit_y = max(guard_band, int(height * max_extra_ratio))
        left = min(left, limit_x)
        right = min(right, limit_x)
        top = min(top, limit_y)
        bottom = min(bottom, limit_y)

    crop_w = width - left - right
    crop_h = height - top - bottom
    if crop_w <= 0 or crop_h <= 0:
        return None, (
            "auto edge crop skipped: invalid detected crop "
            f"(pixels={left}/{top}/{right}/{bottom})"
        )
    if left <= 0 and right <= 0 and top <= 0 and bottom <= 0:
        return None, (
            "auto edge crop not needed "
            f"(black_limit={black_line_limit:.3f}, cast_limit={cast_limit:.3f})"
        )
    return (
        (left, top, crop_w, crop_h),
        "auto edge crop detected "
        f"(pixels={left}/{top}/{right}/{bottom}, bg={bg_median:.4f}, "
        f"black_limit={black_line_limit:.3f}, cast_limit={cast_limit:.3f}, "
        f"row_high={row_high_limit:.4f}, col_low={col_low_limit:.4f})",
    )


def _stage2_shape_dict(shape) -> dict:
    if not shape:
        return {}
    channels, height, width = shape
    return {"channels": int(channels), "height": int(height), "width": int(width)}


def _stage2_center_protection(shape: dict, area_ratio: float) -> dict:
    width = int((shape or {}).get("width", 0) or 0)
    height = int((shape or {}).get("height", 0) or 0)
    if width <= 0 or height <= 0:
        return {}
    requested_ratio = max(0.50, min(0.95, float(area_ratio)))
    linear_ratio = math.sqrt(requested_ratio)
    protected_width = min(width, max(1, int(math.ceil(width * linear_ratio))))
    protected_height = min(height, max(1, int(math.ceil(height * linear_ratio))))
    if protected_width < width and protected_width % 2:
        protected_width += 1
    if protected_height < height and protected_height % 2:
        protected_height += 1
    left = (width - protected_width) // 2
    top = (height - protected_height) // 2
    if left > 0 and left % 2:
        left -= 1
    if top > 0 and top % 2:
        top -= 1
    right = left + protected_width
    bottom = top + protected_height
    return {
        "requested_area_ratio": requested_ratio,
        "actual_area_ratio": (protected_width * protected_height) / float(width * height),
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "width": protected_width,
        "height": protected_height,
    }


def _stage2_constrain_crop_to_center(
    pipeline,
    crop_report: dict,
    x: int,
    y: int,
    crop_w: int,
    crop_h: int,
    *,
    reason: str,
) -> Tuple[Optional[Tuple[int, int, int, int]], str]:
    before = crop_report.get("current_shape") or _stage2_shape_dict(pipeline.siril.get_image_shape())
    current_width = int((before or {}).get("width", 0) or 0)
    current_height = int((before or {}).get("height", 0) or 0)
    if current_width <= 0 or current_height <= 0:
        return None, "center-area protection blocked crop: current shape unavailable"

    original = crop_report.get("original_shape") or before
    if not crop_report.get("original_shape"):
        crop_report["original_shape"] = original
    original_width = int((original or {}).get("width", 0) or 0)
    original_height = int((original or {}).get("height", 0) or 0)
    if original_width <= 0 or original_height <= 0:
        return None, "center-area protection blocked crop: original shape unavailable"

    current_left = int(crop_report.get("total_left", 0) or 0)
    current_top = int(crop_report.get("total_top", 0) or 0)
    current_right = current_left + current_width
    current_bottom = current_top + current_height

    requested_left = current_left + max(0, int(x))
    requested_top = current_top + max(0, int(y))
    requested_right = min(current_right, current_left + int(x) + int(crop_w))
    requested_bottom = min(current_bottom, current_top + int(y) + int(crop_h))
    if requested_right <= requested_left or requested_bottom <= requested_top:
        return None, "center-area protection blocked crop: invalid requested rectangle"

    protection = crop_report.get("center_protection") or _stage2_center_protection(
        original,
        float(getattr(pipeline.cfg, "stage2_center_protect_area_ratio", 0.70)),
    )
    crop_report["center_protection"] = protection
    if not protection:
        return None, "center-area protection blocked crop: protection rectangle unavailable"
    if (
        current_left > int(protection["left"])
        or current_top > int(protection["top"])
        or current_right < int(protection["right"])
        or current_bottom < int(protection["bottom"])
    ):
        return None, "center-area protection blocked crop: current image already violates protection"

    applied_left = max(current_left, min(requested_left, int(protection["left"])))
    applied_top = max(current_top, min(requested_top, int(protection["top"])))
    applied_right = min(current_right, max(requested_right, int(protection["right"])))
    applied_bottom = min(current_bottom, max(requested_bottom, int(protection["bottom"])))

    # Keep every Stage 2 crop aligned to even absolute boundaries.  Expand
    # outward where possible so alignment never removes more valid pixels than
    # the detector requested; an odd outer image edge is rounded inward only
    # after the even-aligned center protection rectangle has been retained.
    if applied_left % 2:
        applied_left = max(current_left, applied_left - 1)
    if applied_top % 2:
        applied_top = max(current_top, applied_top - 1)
    if applied_right % 2:
        applied_right = (
            applied_right + 1
            if applied_right < current_right
            else applied_right - 1
        )
    if applied_bottom % 2:
        applied_bottom = (
            applied_bottom + 1
            if applied_bottom < current_bottom
            else applied_bottom - 1
        )
    applied_w = applied_right - applied_left
    applied_h = applied_bottom - applied_top
    if applied_w <= 0 or applied_h <= 0:
        return None, "center-area protection blocked crop: no valid protected rectangle"

    applied_rect = (
        applied_left - current_left,
        applied_top - current_top,
        applied_w,
        applied_h,
    )
    requested_rect = (int(x), int(y), int(crop_w), int(crop_h))
    if applied_rect == (0, 0, current_width, current_height):
        note = (
            "center-area protection blocked crop "
            f"(reason={reason}, requested={requested_rect}, "
            f"protected_area={float(protection['actual_area_ratio']):.3f})"
        )
        crop_report.setdefault("crop_limit_hits", []).append(
            {
                "reason": reason,
                "requested": requested_rect,
                "applied": None,
                "message": note,
            }
        )
        return None, note

    if applied_rect != requested_rect:
        note = (
            "center-area protection constrained crop "
            f"(reason={reason}, requested={requested_rect}, applied={applied_rect}, "
            f"protected_area={float(protection['actual_area_ratio']):.3f})"
        )
        crop_report.setdefault("crop_limit_hits", []).append(
            {
                "reason": reason,
                "requested": requested_rect,
                "applied": applied_rect,
                "message": note,
            }
        )
        return applied_rect, note
    return applied_rect, ""


def _stage2_crop_totals(crop_report: dict) -> dict:
    original = crop_report.get("original_shape") or {}
    current = crop_report.get("current_shape") or original
    left = int(crop_report.get("total_left", 0) or 0)
    top = int(crop_report.get("total_top", 0) or 0)
    width = int(current.get("width", 0) or 0)
    height = int(current.get("height", 0) or 0)
    original_width = int(original.get("width", 0) or 0)
    original_height = int(original.get("height", 0) or 0)
    return {
        "left": left,
        "top": top,
        "right": max(0, original_width - left - width),
        "bottom": max(0, original_height - top - height),
    }


def _detect_native_contour_candidate(
    pipeline,
) -> Tuple[Optional[Tuple[int, int, int, int]], str, dict]:
    """Run the pure first-party detector against the current Siril pixels."""

    try:
        image_data = pipeline.siril.get_image_pixeldata(preview=False)
    except (
        AttributeError,
        CommandError,
        SirilError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        report = {
            "method": "native_contour",
            "accepted": False,
            "reason": "image_data_read_failed",
            "candidate": None,
            "error": str(error),
        }
        return None, "native contour crop skipped: image data read failed", report
    if image_data is None:
        report = {
            "method": "native_contour",
            "accepted": False,
            "reason": "image_data_unavailable",
            "candidate": None,
        }
        return None, "native contour crop skipped: image data unavailable", report
    result = detect_native_contour_crop(
        _to_rgb_float_fullres(np.asarray(image_data))
    )
    report = result.as_report()
    if not result.accepted or result.rect is None:
        return None, f"native contour crop skipped: {result.reason}", report
    x, y, crop_w, crop_h = result.rect
    evidence = result.evidence or {}
    source = str(evidence.get("rectangle_source") or "unknown")
    note = (
        "native contour crop detected "
        f"(x={x}, y={y}, w={crop_w}, h={crop_h}, source={source}, "
        f"removed_black={float(evidence.get('removed_near_black_ratio', 0.0)):.3f})"
    )
    return result.rect, note, report


def _detect_field_rotation_candidate(
    pipeline,
) -> Tuple[Optional[Tuple[int, int, int, int]], str, dict]:
    """Detect edge-connected field-rotation coverage loss in current pixels."""

    if not bool(
        getattr(pipeline.cfg, "stage2_field_rotation_detection_enabled", True)
    ):
        report = {
            "method": "native_field_rotation",
            "accepted": False,
            "reason": "disabled_by_configuration",
            "candidate": None,
        }
        return None, "field-rotation coverage crop disabled", report
    try:
        image_data = pipeline.siril.get_image_pixeldata(preview=False)
    except (
        AttributeError,
        CommandError,
        SirilError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        report = {
            "method": "native_field_rotation",
            "accepted": False,
            "reason": "image_data_read_failed",
            "candidate": None,
            "error": str(error),
        }
        return None, "field-rotation coverage crop skipped: image data read failed", report
    if image_data is None:
        report = {
            "method": "native_field_rotation",
            "accepted": False,
            "reason": "image_data_unavailable",
            "candidate": None,
        }
        return None, "field-rotation coverage crop skipped: image data unavailable", report
    result = detect_field_rotation_crop(
        _to_rgb_float_fullres(np.asarray(image_data)),
        noise_ratio_min=float(
            getattr(pipeline.cfg, "stage2_field_rotation_noise_ratio_min", 1.35)
        ),
        chroma_ratio_min=float(
            getattr(pipeline.cfg, "stage2_field_rotation_chroma_ratio_min", 1.20)
        ),
    )
    report = result.as_report()
    if not result.accepted or result.rect is None:
        return (
            None,
            f"field-rotation coverage crop skipped: {result.reason}",
            report,
        )
    x, y, crop_w, crop_h = result.rect
    evidence = result.evidence or {}
    note = (
        "field-rotation coverage crop detected "
        f"(x={x}, y={y}, w={crop_w}, h={crop_h}, "
        f"connected={float(evidence.get('edge_connected_ratio', 0.0)):.3f}, "
        f"retained={float(evidence.get('candidate_retained_ratio', 0.0)):.3f})"
    )
    return result.rect, note, report


def _stage2_apply_crop(
    pipeline,
    crop_report: dict,
    x: int,
    y: int,
    crop_w: int,
    crop_h: int,
    *,
    reason: str,
) -> Tuple[Optional[Tuple[int, int, int, int]], str]:
    before = crop_report.get("current_shape") or _stage2_shape_dict(pipeline.siril.get_image_shape())
    before_width = int(before.get("width", 0) or 0)
    before_height = int(before.get("height", 0) or 0)
    requested_crop = {
        "x": int(x),
        "y": int(y),
        "width": int(crop_w),
        "height": int(crop_h),
    }
    applied_rect, protection_note = _stage2_constrain_crop_to_center(
        pipeline,
        crop_report,
        x,
        y,
        crop_w,
        crop_h,
        reason=reason,
    )
    if applied_rect is None:
        return None, protection_note
    x, y, crop_w, crop_h = applied_rect
    pipeline.cmd_with_check("crop", str(x), str(y), str(crop_w), str(crop_h))
    crop_report["total_left"] = int(crop_report.get("total_left", 0) or 0) + int(x)
    crop_report["total_top"] = int(crop_report.get("total_top", 0) or 0) + int(y)
    crop_report["current_shape"] = {
        "channels": int(before.get("channels", 0) or 0),
        "height": int(crop_h),
        "width": int(crop_w),
    }
    crop_report.setdefault("crops", []).append(
        {
            "reason": reason,
            "requested_crop": requested_crop,
            "center_protection_limited": bool(protection_note),
            "x": int(x),
            "y": int(y),
            "width": int(crop_w),
            "height": int(crop_h),
            "removed_left": int(x),
            "removed_top": int(y),
            "removed_right": max(0, before_width - int(x) - int(crop_w)),
            "removed_bottom": max(0, before_height - int(y) - int(crop_h)),
            "before_shape": before,
            "after_shape": crop_report["current_shape"],
        }
    )
    crop_report["total_crop"] = _stage2_crop_totals(crop_report)
    return applied_rect, protection_note


STAGE2_COLOR_ARTIFACT_MIN_RETAINED_RATIO = 0.90


def _edge_color_artifact_crop(pipeline, crop_report: Optional[dict] = None) -> str:
    image_data = pipeline.siril.get_image_pixeldata(preview=False)
    shape = pipeline.siril.get_image_shape()
    if image_data is None or not shape:
        return ""
    rgb = _to_rgb_float_fullres(np.asarray(image_data))
    if rgb.ndim != 3 or rgb.shape[1] < 80 or rgb.shape[2] < 80:
        return ""
    _channels, height, width = rgb.shape
    strip_ratio = float(
        getattr(pipeline.cfg, "stage2_color_cleanup_strip_ratio", 0.025)
    )
    strip_w = max(8, int(width * strip_ratio))
    strip_h = max(8, int(height * strip_ratio))
    center = rgb[
        :,
        int(height * 0.22) : int(height * 0.78),
        int(width * 0.22) : int(width * 0.78),
    ]
    center_cast = _edge_color_cast_score(center)
    sides = {
        "left": rgb[:, :, :strip_w],
        "right": rgb[:, :, width - strip_w :],
        "top": rgb[:, :strip_h, :],
        "bottom": rgb[:, height - strip_h :, :],
    }
    absolute_limit = float(
        getattr(pipeline.cfg, "stage2_edge_cast_absolute_max", 0.010)
    )
    relative_limit = float(
        getattr(pipeline.cfg, "stage2_edge_cast_cleanup_ratio_max", 2.6)
    )
    bad_sides = {
        name: _edge_color_cast_score(strip)
        for name, strip in sides.items()
        if _edge_color_cast_score(strip)
        > max(absolute_limit, center_cast * relative_limit)
    }
    if not bad_sides:
        return ""
    crop_left = strip_w if "left" in bad_sides else 0
    crop_right = strip_w if "right" in bad_sides else 0
    crop_top = strip_h if "top" in bad_sides else 0
    crop_bottom = strip_h if "bottom" in bad_sides else 0
    crop_w = width - crop_left - crop_right
    crop_h = height - crop_top - crop_bottom
    # This is a fixed safety invariant, not a tuning parameter: color-edge
    # cleanup must retain more than 90% of each current image dimension.
    if (
        crop_w <= width * STAGE2_COLOR_ARTIFACT_MIN_RETAINED_RATIO
        or crop_h <= height * STAGE2_COLOR_ARTIFACT_MIN_RETAINED_RATIO
    ):
        return ""
    if crop_report is not None:
        applied_rect, protection_note = _stage2_apply_crop(
            pipeline,
            crop_report,
            crop_left,
            crop_top,
            crop_w,
            crop_h,
            reason="adaptive_color_edge",
        )
        if applied_rect is None:
            return f"adaptive color-edge crop blocked; {protection_note}"
        crop_left, crop_top, crop_w, crop_h = applied_rect
        crop_right = width - crop_left - crop_w
        crop_bottom = height - crop_top - crop_h
    else:
        pipeline.cmd_with_check("crop", str(crop_left), str(crop_top), str(crop_w), str(crop_h))
    side_text = ",".join(sorted(bad_sides))
    return (
        "adaptive color-edge crop applied "
        f"(sides={side_text}, edge_cast={max(bad_sides.values()):.4f}, "
        f"center_cast={center_cast:.4f}, pixels={crop_left}/{crop_top}/{crop_right}/{crop_bottom})"
    )


def _edge_color_cast_score(rgb: np.ndarray) -> float:
    if rgb.size == 0:
        return 0.0
    arr = np.asarray(rgb, dtype=np.float32)
    gray = (0.2126 * arr[0] + 0.7152 * arr[1] + 0.0722 * arr[2]).astype(np.float32)
    chroma = np.mean(np.abs(arr - gray[None, :, :]), axis=0)
    blue_excess = np.maximum(arr[2] - np.maximum(arr[0], arr[1]), 0.0)
    red_excess = np.maximum(arr[0] - np.maximum(arr[1], arr[2]), 0.0)
    return float(np.nanmedian(chroma + 0.65 * np.maximum(blue_excess, red_excess)))


def run_stage2_view_correction(pipeline) -> None:
    """
    阶段 2: 裁切
    - 按工作流先做画面边缘裁切
    - 图像解析（天体测量）在阶段4执行
    """
    stage_label = PipelineStage.VIEW_CORRECTION.label
    pipeline.log.stage_start(stage_label)
    status = "ok"
    review_reason_code = ""
    messages: List[str] = []
    pipeline._stage2_view_review_required = False
    try:
        initial_shape = _stage2_shape_dict(pipeline.siril.get_image_shape())
    except (CommandError, SirilError, OSError, RuntimeError, TypeError, ValueError):
        initial_shape = {}
    crop_report = {
        "stage": "stage2_crop",
        "mode": "native_no_crop",
        "original_shape": initial_shape,
        "current_shape": initial_shape,
        "total_left": 0,
        "total_top": 0,
        "total_crop": {"left": 0, "top": 0, "right": 0, "bottom": 0},
        "crops": [],
        "crop_limit_hits": [],
        "requires_review": False,
        "target_edge_black_ratio": float(getattr(pipeline.cfg, "stage2_edge_black_target", 0.03)),
        "center_protection": _stage2_center_protection(
            initial_shape,
            float(getattr(pipeline.cfg, "stage2_center_protect_area_ratio", 0.70)),
        ),
    }
    pipeline.stage2_crop_report = crop_report

    if str(getattr(pipeline.cfg, "stage2_processing_mode", "auto")) == "preserve":
        crop_report["mode"] = "user_preserve"
        diagnostics = pipeline._measure_current_features()
        edge_black_ratio = (
            float(getattr(diagnostics, "edge_black_ratio", 0.0) or 0.0)
            if diagnostics is not None
            else None
        )
        requires_review = bool(
            edge_black_ratio is not None
            and edge_black_ratio
            > float(
                getattr(
                    pipeline.cfg,
                    "stage2_preserve_review_edge_black_max",
                    0.30,
                )
            )
        )
        if requires_review:
            pipeline._stage2_view_review_required = True
        stage_saved = pipeline._save_stage_output("stage2_corrected")
        crop_report.update(
            {
                "status": "ok" if stage_saved else "degraded",
                "execution": "safe_passthrough",
                "reason_code": "user_preserve",
                "requires_review": requires_review,
                "diagnostics": {"edge_black_ratio": edge_black_ratio},
                "final_shape": initial_shape,
                "messages": ["user requested original boundary preservation"],
            }
        )
        if hasattr(pipeline, "_write_stage_json"):
            pipeline._write_stage_json("stage2_crop_report.json", crop_report)
        elapsed = pipeline.log.stage_end(stage_label)
        pipeline._record_stage(
            stage_label,
            "ok" if stage_saved else "degraded",
            elapsed,
            "用户选择保留原图；已生成 Stage 2 安全直通产物",
            execution="safe_passthrough",
            reason_code="user_preserve",
            details={
                "output": "stage2_corrected" if stage_saved else None,
                "requires_review": requires_review,
                "edge_black_ratio": edge_black_ratio,
            },
        )
        return

    failure_action = str(
        getattr(pipeline.cfg, "stage2_failure_action", "auto_fallback")
    )
    if failure_action != "auto_fallback":
        if not pipeline._save_stage_output("stage2_input_checkpoint"):
            messages.append("stage2 immutable input checkpoint could not be saved")

    if bool(getattr(pipeline.cfg, "stage2_base_crop_enabled", False)):
        margin = float(getattr(pipeline.cfg, "stage2_base_crop_margin", 0.02))
        width = int(initial_shape.get("width", 0) or 0)
        height = int(initial_shape.get("height", 0) or 0)
        trim_x = max(0, int(round(width * margin)))
        trim_y = max(0, int(round(height * margin)))
        if trim_x or trim_y:
            applied_rect, protection_note = _stage2_apply_crop(
                pipeline,
                crop_report,
                trim_x,
                trim_y,
                width - 2 * trim_x,
                height - 2 * trim_y,
                reason="user_base_crop",
            )
            if protection_note:
                messages.append(protection_note)
            messages.append(
                "base crop applied" if applied_rect is not None
                else "base crop blocked by center protection"
            )
            if applied_rect is not None:
                crop_report["mode"] = "user_base_crop"

    native_contour_applied = False
    try:
        pipeline.log.info("运行第一方轮廓/最大内接矩形边界检测...")
        contour_rect, contour_note, contour_report = (
            _detect_native_contour_candidate(pipeline)
        )
        crop_report["native_contour"] = contour_report
        messages.append(contour_note)
        if contour_rect is not None:
            x, y, crop_w, crop_h = contour_rect
            applied_rect, protection_note = _stage2_apply_crop(
                pipeline,
                crop_report,
                x,
                y,
                crop_w,
                crop_h,
                reason="native_contour",
            )
            if protection_note:
                messages.append(protection_note)
            if applied_rect is not None:
                native_contour_applied = True
                crop_report["mode"] = "native_contour"
                crop_report["native_contour"]["applied_crop"] = {
                    "x": int(applied_rect[0]),
                    "y": int(applied_rect[1]),
                    "width": int(applied_rect[2]),
                    "height": int(applied_rect[3]),
                }
                crop_report["native_contour"]["center_protection_limited"] = bool(
                    protection_note
                )
                pipeline.log.info(contour_note)
            else:
                crop_report["native_contour"]["accepted"] = False
                crop_report["native_contour"]["reason"] = (
                    "center_protection_blocked"
                )
    except (CommandError, SirilError) as error:
        status = "degraded"
        crop_report["native_contour"] = {
            "method": "native_contour",
            "accepted": False,
            "reason": "crop_command_failed",
            "candidate": None,
            "error": pipeline._short_text(error, 160),
        }
        messages.append(
            "native contour crop command failed: "
            f"{pipeline._short_text(error, 160)}"
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        crop_report["native_contour"] = {
            "method": "native_contour",
            "accepted": False,
            "reason": "detector_error",
            "candidate": None,
            "error": pipeline._short_text(error, 160),
        }
        messages.append(
            "native contour crop skipped: "
            f"{pipeline._short_text(error, 160)}"
        )

    field_rotation_applied = False
    if not native_contour_applied:
        try:
            pipeline.log.info("运行第一方场旋低覆盖边缘检测...")
            field_rect, field_note, field_report = (
                _detect_field_rotation_candidate(pipeline)
            )
            crop_report["field_rotation"] = field_report
            messages.append(field_note)
            if field_rect is not None:
                x, y, crop_w, crop_h = field_rect
                applied_rect, protection_note = _stage2_apply_crop(
                    pipeline,
                    crop_report,
                    x,
                    y,
                    crop_w,
                    crop_h,
                    reason="native_field_rotation",
                )
                if protection_note:
                    messages.append(protection_note)
                if applied_rect is not None:
                    field_rotation_applied = True
                    crop_report["mode"] = "native_field_rotation"
                    crop_report["field_rotation"]["applied_crop"] = {
                        "x": int(applied_rect[0]),
                        "y": int(applied_rect[1]),
                        "width": int(applied_rect[2]),
                        "height": int(applied_rect[3]),
                    }
                    crop_report["field_rotation"][
                        "center_protection_limited"
                    ] = bool(protection_note)
                    pipeline.log.info(field_note)

                    _residual_rect, residual_note, residual_report = (
                        _detect_field_rotation_candidate(pipeline)
                    )
                    crop_report["field_rotation"]["residual"] = residual_report
                    messages.append(f"post-crop {residual_note}")
                    if bool(residual_report.get("accepted")):
                        crop_report["requires_review"] = True
                        crop_report["reason_code"] = (
                            "field_rotation_residual_review"
                        )
                        pipeline._stage2_view_review_required = True
                        status = "degraded"
                        review_reason_code = "field_rotation_residual_review"
                        messages.append(
                            "field-rotation coverage anomaly remains after "
                            "70% center protection; manual review required"
                        )
                else:
                    crop_report["field_rotation"]["accepted"] = False
                    crop_report["field_rotation"]["reason"] = (
                        "center_protection_blocked"
                    )
        except (CommandError, SirilError) as error:
            status = "degraded"
            crop_report["field_rotation"] = {
                "method": "native_field_rotation",
                "accepted": False,
                "reason": "crop_command_failed",
                "candidate": None,
                "error": pipeline._short_text(error, 160),
            }
            messages.append(
                "field-rotation crop command failed: "
                f"{pipeline._short_text(error, 160)}"
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            crop_report["field_rotation"] = {
                "method": "native_field_rotation",
                "accepted": False,
                "reason": "detector_error",
                "candidate": None,
                "error": pipeline._short_text(error, 160),
            }
            messages.append(
                "field-rotation coverage crop skipped: "
                f"{pipeline._short_text(error, 160)}"
            )

    primary_crop_applied = native_contour_applied or field_rotation_applied

    # 第一方候选没有通过安全门或没有实际裁切时，运行现有边缘检测方案。
    if not primary_crop_applied:
        pipeline.log.info("轮廓检测未裁切，自动识别黑边/叠加边缘并裁切...")
        try:
            crop_rect, crop_note = _detect_auto_edge_crop(pipeline, is_adaptive=False)
            messages.append(crop_note)
            if crop_rect:
                x, y, crop_w, crop_h = crop_rect
                applied_rect, protection_note = _stage2_apply_crop(
                    pipeline,
                    crop_report,
                    x,
                    y,
                    crop_w,
                    crop_h,
                    reason="initial_auto_edge",
                )
                if protection_note:
                    messages.append(protection_note)
                if applied_rect is None:
                    pipeline.log.warn("中心面积保护已阻止本次边界裁切")
                    status = "degraded"
                else:
                    crop_report["mode"] = "native_edge"
                    applied_x, applied_y, applied_w, applied_h = applied_rect
                    pipeline.log.info(
                        f"已自动裁切 (x={applied_x}, y={applied_y}, "
                        f"w={applied_w}, h={applied_h})"
                    )
            else:
                pipeline.log.info(crop_note)
        except (CommandError, SirilError) as e:
            pipeline.log.warn(f"裁切失败: {e}")
            status = "degraded"

    if status == "ok" and not primary_crop_applied:
        target = float(getattr(pipeline.cfg, "stage2_edge_black_target", 0.10))
        max_passes = int(getattr(pipeline.cfg, "stage2_adaptive_edge_crop_max_passes", 3))
        last_edge_black = None
        for pass_index in range(max_passes):
            edge_feat = pipeline._measure_current_features()
            if edge_feat is None:
                messages.append("adaptive edge crop skipped: feature sampling unavailable")
                break
            last_edge_black = edge_feat.edge_black_ratio
            if edge_feat.edge_black_ratio <= target:
                if pass_index == 0:
                    messages.append(
                        f"adaptive edge crop not needed (edge_black={edge_feat.edge_black_ratio:.3f})"
                    )
                break
            try:
                crop_rect, adaptive_note = _detect_auto_edge_crop(pipeline, is_adaptive=True)
                if crop_rect:
                    x, y, crop_w, crop_h = crop_rect
                    applied_rect, protection_note = _stage2_apply_crop(
                        pipeline,
                        crop_report,
                        x,
                        y,
                        crop_w,
                        crop_h,
                        reason=f"adaptive_edge_pass_{pass_index + 1}",
                    )
                    if protection_note:
                        messages.append(protection_note)
                    if applied_rect is None:
                        status = "degraded"
                        messages.append(
                            "adaptive edge crop stopped: center-area protection limit reached"
                        )
                        break
                after_feat = pipeline._measure_current_features()
                if after_feat is not None:
                    messages.append(
                        "stage2 pass "
                        f"{pass_index + 1} metrics: edge_black "
                        f"{edge_feat.edge_black_ratio:.3f}->{after_feat.edge_black_ratio:.3f}, "
                        f"global_dark={getattr(edge_feat, 'global_dark_ratio', 0.0):.3f}"
                        f"->{getattr(after_feat, 'global_dark_ratio', 0.0):.3f}"
                    )
                    improvement_min = float(
                        getattr(
                            pipeline.cfg,
                            "stage2_edge_black_improvement_min",
                            0.003,
                        )
                    )
                    if (
                        after_feat.edge_black_ratio
                        >= edge_feat.edge_black_ratio - improvement_min
                    ):
                        status = "degraded" if status == "ok" else status
                        messages.append(
                            "adaptive edge crop stopped: degraded_no_improvement "
                            f"(edge_black {edge_feat.edge_black_ratio:.3f}->{after_feat.edge_black_ratio:.3f})"
                        )
                        break
                if adaptive_note:
                    messages.append(f"stage2 pass {pass_index + 1}: {adaptive_note}")
                    if not crop_rect:
                        break
                else:
                    messages.append(
                        "adaptive edge crop skipped "
                        f"(edge_black={edge_feat.edge_black_ratio:.3f}, target={target:.3f})"
                    )
                    break
            except (CommandError, SirilError) as e:
                pipeline.log.warn(f"自适应黑边裁切失败: {e}")
                status = "degraded"
                messages.append(
                    f"adaptive edge crop failed: {pipeline._short_text(e, 160)}"
                )
                break
        final_feat = pipeline._measure_current_features()
        if final_feat is not None:
            messages.append(
                "stage2 edge_black "
                f"{last_edge_black if last_edge_black is not None else final_feat.edge_black_ratio:.3f}"
                f"->{final_feat.edge_black_ratio:.3f} (target={target:.3f})"
            )
            if final_feat.edge_black_ratio > target:
                status = "degraded" if status == "ok" else status
                messages.append(
                    f"edge_black remains above stage2 target: {final_feat.edge_black_ratio:.3f}>{target:.3f}"
                )

    if (
        status == "ok"
        and not primary_crop_applied
        and bool(
            getattr(
                pipeline.cfg,
                "stage2_color_edge_cleanup_enabled",
                True,
            )
        )
    ):
        try:
            color_edge_note = _edge_color_artifact_crop(pipeline, crop_report)
            if color_edge_note:
                messages.append(color_edge_note)
                if color_edge_note.startswith("adaptive color-edge crop blocked"):
                    status = "degraded"
        except (CommandError, SirilError) as e:
            pipeline.log.warn(f"彩色边缘裁切失败: {e}")
            status = "degraded"
            messages.append(f"adaptive color-edge crop failed: {pipeline._short_text(e, 160)}")
        except (OSError, RuntimeError, TypeError, ValueError) as e:
            pipeline.log.warn(f"彩色边缘检测失败: {e}")
            messages.append(f"adaptive color-edge crop skipped: {pipeline._short_text(e, 160)}")

    if status == "degraded" and failure_action in {"preserve_review", "stop"}:
        crop_report["failure_action"] = failure_action
        if failure_action == "preserve_review":
            try:
                pipeline.cmd_with_check("load", "stage2_input_checkpoint")
                pipeline._stage2_view_review_required = True
                crop_report["execution"] = "safe_passthrough"
                crop_report["reason_code"] = "failure_policy_preserve_review"
                messages.append(
                    "decisive Stage 2 failure restored immutable input for review"
                )
            except (CommandError, SirilError) as error:
                status = "failed"
                messages.append(
                    "Stage 2 preserve_review restore failed: "
                    f"{pipeline._short_text(error, 160)}"
                )
        else:
            status = "failed"
            messages.append("Stage 2 stop policy requested after diagnostics")

    stage_saved = pipeline._save_stage_output("stage2_corrected")
    if not stage_saved and status == "ok":
        status = "degraded"
        messages.append("stage2 输出保存失败")
    try:
        crop_report["final_shape"] = _stage2_shape_dict(pipeline.siril.get_image_shape())
    except (CommandError, SirilError, OSError, RuntimeError, TypeError, ValueError):
        crop_report["final_shape"] = crop_report.get("current_shape") or {}
    crop_report["total_crop"] = _stage2_crop_totals(crop_report)
    crop_report["status"] = status
    crop_report["failure_action"] = failure_action
    crop_report["messages"] = messages
    if hasattr(pipeline, "_write_stage_json"):
        pipeline._write_stage_json("stage2_crop_report.json", crop_report)

    elapsed = pipeline.log.stage_end(stage_label)
    preserved_for_review = bool(
        crop_report.get("reason_code") == "failure_policy_preserve_review"
    )
    pipeline._record_stage(
        stage_label,
        status,
        elapsed,
        "；".join(messages),
        execution="safe_passthrough" if preserved_for_review else "completed",
        fallback_used=status == "degraded",
        upstream_passthrough=preserved_for_review,
        reason_code=(
            "failure_policy_preserve_review"
            if preserved_for_review
            else review_reason_code
        ),
        details={
            "crop_detector": crop_report.get("mode", "native_no_crop"),
            "failure_action": failure_action,
            "final_source": (
                "stage2_input_checkpoint"
                if preserved_for_review
                else "stage2_processed"
            ),
            "requires_review": bool(
                getattr(pipeline, "_stage2_view_review_required", False)
            ),
        },
    )
