"""Stage 2 view correction and crop."""
from typing import List, Optional, Tuple

import numpy as np

from image_metrics import _to_rgb_float_fullres
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
    cast_limit = max(0.010, center_cast * 2.8)
    max_scan_x = max(4, int(width * 0.25))
    max_scan_y = max(4, int(height * 0.25))
    stable_run = max(6, int(min(width, height) * 0.006))

    col_black, col_median, col_cast = _line_edge_scores(
        gray, rgb, axis=0, black_threshold=black_threshold
    )
    row_black, row_median, row_cast = _line_edge_scores(
        gray, rgb, axis=1, black_threshold=black_threshold
    )
    level_window = int(getattr(pipeline.cfg, "stage2_level_artifact_window", 81))
    row_level = _rolling_median_1d(row_median, level_window)
    col_level = _rolling_median_1d(col_median, level_window)
    row_high_limit = bg_median + max(bg_std * 2.30, 0.00008)
    col_high_limit = bg_median + max(bg_std * 2.30, 0.00008)
    col_low_limit = center_median - max(bg_std * 0.62, 0.000035)

    bad_cols = (
        (col_black > black_line_limit)
        | ((col_median <= dark_level_limit) & (col_black > 0.004))
        | (col_cast > cast_limit)
        | (col_level > col_high_limit)
        | (col_level < col_low_limit)
    )
    bad_rows = (
        (row_black > black_line_limit)
        | ((row_median <= dark_level_limit) & (row_black > 0.004))
        | (row_cast > cast_limit)
        | (row_level > row_high_limit)
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


def _stage2_apply_crop(
    pipeline,
    crop_report: dict,
    x: int,
    y: int,
    crop_w: int,
    crop_h: int,
    *,
    reason: str,
) -> None:
    before = crop_report.get("current_shape") or _stage2_shape_dict(pipeline.siril.get_image_shape())
    before_width = int(before.get("width", 0) or 0)
    before_height = int(before.get("height", 0) or 0)
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


def _edge_color_artifact_crop(pipeline, crop_report: Optional[dict] = None) -> str:
    image_data = pipeline.siril.get_image_pixeldata(preview=False)
    shape = pipeline.siril.get_image_shape()
    if image_data is None or not shape:
        return ""
    rgb = _to_rgb_float_fullres(np.asarray(image_data))
    if rgb.ndim != 3 or rgb.shape[1] < 80 or rgb.shape[2] < 80:
        return ""
    _channels, height, width = rgb.shape
    strip_w = max(8, int(width * 0.025))
    strip_h = max(8, int(height * 0.025))
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
    bad_sides = {
        name: _edge_color_cast_score(strip)
        for name, strip in sides.items()
        if _edge_color_cast_score(strip) > max(0.010, center_cast * 2.6)
    }
    if not bad_sides:
        return ""
    crop_left = strip_w if "left" in bad_sides else 0
    crop_right = strip_w if "right" in bad_sides else 0
    crop_top = strip_h if "top" in bad_sides else 0
    crop_bottom = strip_h if "bottom" in bad_sides else 0
    crop_w = width - crop_left - crop_right
    crop_h = height - crop_top - crop_bottom
    if crop_w <= width * 0.90 or crop_h <= height * 0.90:
        return ""
    if crop_report is not None:
        _stage2_apply_crop(
            pipeline,
            crop_report,
            crop_left,
            crop_top,
            crop_w,
            crop_h,
            reason="adaptive_color_edge",
        )
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
    pipeline.log.stage_start("阶段 2: 裁切")
    status = "ok"
    messages: List[str] = []
    try:
        initial_shape = _stage2_shape_dict(pipeline.siril.get_image_shape())
    except Exception:
        initial_shape = {}
    crop_report = {
        "stage": "stage2_crop",
        "mode": "auto_edge_detection",
        "original_shape": initial_shape,
        "current_shape": initial_shape,
        "total_left": 0,
        "total_top": 0,
        "total_crop": {"left": 0, "top": 0, "right": 0, "bottom": 0},
        "crops": [],
        "target_edge_black_ratio": float(getattr(pipeline.cfg, "stage2_edge_black_target", 0.03)),
    }
    pipeline.stage2_crop_report = crop_report

    # 裁切边缘
    pipeline.log.info("自动识别黑边/叠加边缘并裁切...")
    try:
        crop_rect, crop_note = _detect_auto_edge_crop(pipeline, is_adaptive=False)
        messages.append(crop_note)
        if crop_rect:
            x, y, crop_w, crop_h = crop_rect
            _stage2_apply_crop(pipeline, crop_report, x, y, crop_w, crop_h, reason="initial_auto_edge")
            pipeline.log.info(f"已自动裁切 (x={x}, y={y}, w={crop_w}, h={crop_h})")
        else:
            pipeline.log.info(crop_note)
    except (CommandError, SirilError) as e:
        pipeline.log.warn(f"裁切失败: {e}")
        status = "degraded"

    if status == "ok":
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
                    _stage2_apply_crop(
                        pipeline,
                        crop_report,
                        x,
                        y,
                        crop_w,
                        crop_h,
                        reason=f"adaptive_edge_pass_{pass_index + 1}",
                    )
                after_feat = pipeline._measure_current_features()
                if after_feat is not None:
                    messages.append(
                        "stage2 pass "
                        f"{pass_index + 1} metrics: edge_black "
                        f"{edge_feat.edge_black_ratio:.3f}->{after_feat.edge_black_ratio:.3f}, "
                        f"global_dark={getattr(edge_feat, 'global_dark_ratio', 0.0):.3f}"
                        f"->{getattr(after_feat, 'global_dark_ratio', 0.0):.3f}"
                    )
                    if after_feat.edge_black_ratio >= edge_feat.edge_black_ratio - 0.003:
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

    if status == "ok":
        try:
            color_edge_note = _edge_color_artifact_crop(pipeline, crop_report)
            if color_edge_note:
                messages.append(color_edge_note)
        except (CommandError, SirilError) as e:
            pipeline.log.warn(f"彩色边缘裁切失败: {e}")
            status = "degraded"
            messages.append(f"adaptive color-edge crop failed: {pipeline._short_text(e, 160)}")
        except Exception as e:
            pipeline.log.warn(f"彩色边缘检测失败: {e}")
            messages.append(f"adaptive color-edge crop skipped: {pipeline._short_text(e, 160)}")

    stage_saved = pipeline._save_stage_output("stage2_corrected")
    if not stage_saved and status == "ok":
        status = "degraded"
        messages.append("stage2 输出保存失败")
    try:
        crop_report["final_shape"] = _stage2_shape_dict(pipeline.siril.get_image_shape())
    except Exception:
        crop_report["final_shape"] = crop_report.get("current_shape") or {}
    crop_report["total_crop"] = _stage2_crop_totals(crop_report)
    crop_report["status"] = status
    crop_report["messages"] = messages
    if hasattr(pipeline, "_write_stage_json"):
        pipeline._write_stage_json("stage2_crop_report.json", crop_report)

    elapsed = pipeline.log.stage_end("阶段 2: 裁切")
    pipeline._record_stage("阶段 2: 裁切", status, elapsed, "；".join(messages))
