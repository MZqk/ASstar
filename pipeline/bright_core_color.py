"""Fixed Stage 4 color-integrity guard for strict bright-core nebulae."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from image_metrics import _box_blur_gray
import stage7_quality


SCHEMA = "starun.stage4-bright-core-color-integrity.v1"
ROI_QUANTILE = 0.99
ROI_SMOOTH_PASSES = 4
ROI_SUPPORT_MIN = 64
CHROMA_DRIFT_MIN = 0.08
SATURATION_MIN = 0.30
DOMINANT_ADVANTAGE_MIN = 0.08
COMPONENT_ACCEPTED_RATIO = 0.005
COMPONENT_REPAIR_RATIO = 0.01
BROAD_PLATFORM_STRONG_ANOMALY_RATIO_MIN = 0.02
BROAD_PLATFORM_SATURATION_MIN = 0.10
BROAD_PLATFORM_COMPONENT_RATIO_MAX = 0.05
REPAIR_DILATION_RADIUS = 3
REPAIR_FEATHER_PASSES = 4
REPAIR_SUPPORT_RATIO_MAX = 0.015
REPAIR_LUMA_ERROR_P99_MAX = 0.002
REPAIR_NEW_CLIP_RATIO_MAX = 0.001
CLIP_THRESHOLD = 0.995
REC709 = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float64)


def _as_rgb_chw(image: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError("bright-core color integrity requires a 3-D RGB image")
    if arr.shape[0] in (3, 4):
        rgb = arr[:3]
        layout = "CHW"
    elif arr.shape[-1] in (3, 4):
        rgb = np.moveaxis(arr[..., :3], -1, 0)
        layout = "HWC"
    else:
        raise ValueError(f"unsupported RGB image shape: {arr.shape}")

    native = np.asarray(rgb)
    values = np.asarray(native, dtype=np.float64)
    finite_values = values[np.isfinite(values)]
    minimum = float(np.min(finite_values)) if finite_values.size else None
    maximum = float(np.max(finite_values)) if finite_values.size else None
    dtype = native.dtype
    scale = 1.0
    domain = "normalized_float"
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        if minimum is not None and minimum < 0.0:
            raise ValueError("signed RGB input contains negative integer values")
        scale = float(info.max)
        domain = f"{dtype.name}_full_scale"
    elif np.issubdtype(dtype, np.floating):
        # Siril normally exposes normalized floats or native uint16 pixels.  A
        # few wrappers convert the latter to float32 without rescaling; accept
        # that representation only when it is demonstrably integer-coded.
        if minimum is not None and maximum is not None:
            normalized_range = minimum >= -0.25 and maximum <= 4.0
            integer_coded = (
                minimum >= 0.0
                and maximum > 4.0
                and maximum <= 65535.0
                and bool(
                    np.allclose(
                        finite_values,
                        np.rint(finite_values),
                        rtol=0.0,
                        atol=1e-4,
                    )
                )
            )
            if integer_coded:
                scale = 65535.0
                domain = "uint16_full_scale_float"
            elif not normalized_range:
                raise ValueError(
                    "unsupported floating RGB pixel domain; expected normalized "
                    "values or uint16-coded integer values"
                )
    else:
        raise ValueError(f"unsupported RGB pixel dtype: {dtype}")

    return values / scale, {
        "domain": domain,
        "dtype": str(dtype),
        "layout": layout,
        "scale": scale,
        "minimum": minimum,
        "maximum": maximum,
        "native_chw": values,
    }


def _luma(rgb: np.ndarray) -> np.ndarray:
    return np.tensordot(REC709, rgb[:3], axes=(0, 0))


def _smooth_luma(rgb: np.ndarray) -> np.ndarray:
    smooth = _luma(rgb)
    for _ in range(ROI_SMOOTH_PASSES):
        smooth = _box_blur_gray(smooth)
    return np.asarray(smooth, dtype=np.float64)


def _chromaticity(rgb: np.ndarray) -> np.ndarray:
    positive = np.clip(rgb[:3], 0.0, None)
    total = np.sum(positive, axis=0)
    return positive / np.maximum(total[np.newaxis, ...], 1e-12)


def _saturation(rgb: np.ndarray) -> np.ndarray:
    positive = np.clip(rgb[:3], 0.0, None)
    high = np.max(positive, axis=0)
    low = np.min(positive, axis=0)
    return (high - low) / np.maximum(high, 1e-12)


def _dominance_margin(chroma: np.ndarray) -> np.ndarray:
    ordered = np.sort(chroma, axis=0)
    return ordered[-1] - ordered[-2]


def _neighbors8(y: int, x: int, height: int, width: int) -> Iterable[Tuple[int, int]]:
    y0 = max(y - 1, 0)
    y1 = min(y + 1, height - 1)
    x0 = max(x - 1, 0)
    x1 = min(x + 1, width - 1)
    for ny in range(y0, y1 + 1):
        for nx in range(x0, x1 + 1):
            if ny != y or nx != x:
                yield ny, nx


def _connected_components8(mask: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    foreground = np.asarray(mask, dtype=bool)
    height, width = foreground.shape
    labels = np.zeros((height, width), dtype=np.int32)
    areas: List[int] = []
    next_label = 0
    for seed_y, seed_x in np.argwhere(foreground):
        y = int(seed_y)
        x = int(seed_x)
        if labels[y, x] != 0:
            continue
        next_label += 1
        labels[y, x] = next_label
        stack = [(y, x)]
        area = 0
        while stack:
            current_y, current_x = stack.pop()
            area += 1
            for ny, nx in _neighbors8(current_y, current_x, height, width):
                if foreground[ny, nx] and labels[ny, nx] == 0:
                    labels[ny, nx] = next_label
                    stack.append((ny, nx))
        areas.append(area)
    return labels, areas


def _dilate8(mask: np.ndarray, radius: int) -> np.ndarray:
    expanded = np.asarray(mask, dtype=bool).copy()
    for _ in range(max(int(radius), 0)):
        padded = np.pad(expanded, 1, mode="constant", constant_values=False)
        height, width = expanded.shape
        expanded = np.logical_or.reduce(
            [
                padded[dy : dy + height, dx : dx + width]
                for dy in range(3)
                for dx in range(3)
            ]
        )
    return expanded


def _fixed_limits() -> Dict[str, Any]:
    return {
        "roi_quantile": ROI_QUANTILE,
        "roi_smooth_passes": ROI_SMOOTH_PASSES,
        "roi_support_min": ROI_SUPPORT_MIN,
        "chroma_l1_over_2_drift_min": CHROMA_DRIFT_MIN,
        "post_saturation_min": SATURATION_MIN,
        "dominant_channel_change_required": True,
        "dominant_channel_advantage_min": DOMINANT_ADVANTAGE_MIN,
        "largest_component_ratio": {
            "accepted": COMPONENT_ACCEPTED_RATIO,
            "repair": COMPONENT_REPAIR_RATIO,
        },
        "broad_core_chroma_platform": {
            "strong_anomaly_ratio_min": BROAD_PLATFORM_STRONG_ANOMALY_RATIO_MIN,
            "chroma_l1_over_2_drift_min": CHROMA_DRIFT_MIN,
            "post_saturation_min": BROAD_PLATFORM_SATURATION_MIN,
            "largest_component_ratio_max": BROAD_PLATFORM_COMPONENT_RATIO_MAX,
            "action": "reject_spcc_to_pcc",
        },
        "repair": {
            "component_ratio_min": COMPONENT_ACCEPTED_RATIO,
            "dilation_radius": REPAIR_DILATION_RADIUS,
            "feather_smooth_passes": REPAIR_FEATHER_PASSES,
            "support_ratio_max": REPAIR_SUPPORT_RATIO_MAX,
            "luma_abs_error_p99_max": REPAIR_LUMA_ERROR_P99_MAX,
            "new_clip_ratio_max": REPAIR_NEW_CLIP_RATIO_MAX,
            "clip_threshold": CLIP_THRESHOLD,
        },
    }


def assess_spcc_bright_core_color(
    before: Any,
    after: Any,
    *,
    target_type: str,
    target_profile: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Measure local SPCC chroma corruption and return JSON plus private masks."""
    strict_evidence = stage7_quality.strict_bright_core_target_evidence(
        target_type,
        target_profile,
    )
    report: Dict[str, Any] = {
        "schema": SCHEMA,
        "applicable": bool(strict_evidence.get("strict", False)),
        "strict_target_evidence": strict_evidence,
        "fixed_limits": _fixed_limits(),
        "status": "not_applicable",
        "accepted": True,
        "repaired": False,
        "final_action": "accept_spcc",
        "trigger_reasons": [],
    }
    if not report["applicable"]:
        return report, {}

    try:
        before_rgb, before_domain = _as_rgb_chw(before)
        after_rgb, after_domain = _as_rgb_chw(after)
    except (TypeError, ValueError) as error:
        report.update(
            status="hard_failed",
            accepted=False,
            final_action="reject_spcc_to_pcc",
            trigger_reasons=["invalid_rgb_input"],
            error=str(error),
        )
        return report, {}
    report["numeric_domain"] = {
        "before": {
            key: value
            for key, value in before_domain.items()
            if key != "native_chw"
        },
        "after": {
            key: value
            for key, value in after_domain.items()
            if key != "native_chw"
        },
        "assessment_domain": "normalized_0_1",
    }
    if before_rgb.shape != after_rgb.shape:
        report.update(
            status="hard_failed",
            accepted=False,
            final_action="reject_spcc_to_pcc",
            trigger_reasons=["shape_changed"],
            shape_before=list(before_rgb.shape),
            shape_after=list(after_rgb.shape),
        )
        return report, {}
    if not np.all(np.isfinite(before_rgb)) or not np.all(np.isfinite(after_rgb)):
        report.update(
            status="hard_failed",
            accepted=False,
            final_action="reject_spcc_to_pcc",
            trigger_reasons=["non_finite_input"],
        )
        return report, {}

    smooth = _smooth_luma(before_rgb)
    white = float(np.quantile(smooth, ROI_QUANTILE))
    roi = np.asarray(smooth >= white, dtype=bool)
    roi_support = int(np.count_nonzero(roi))
    report["roi"] = {
        "source": "stage4_pre_pcc",
        "luminance": "Rec.709",
        "quantile": ROI_QUANTILE,
        "threshold": white,
        "smooth_passes": ROI_SMOOTH_PASSES,
        "support_pixels": roi_support,
        "image_pixels": int(roi.size),
        "support_ratio": float(roi_support / max(roi.size, 1)),
        "minimum_support_pixels": ROI_SUPPORT_MIN,
    }
    if roi_support < ROI_SUPPORT_MIN:
        report.update(
            status="hard_failed",
            accepted=False,
            final_action="reject_spcc_to_pcc",
            trigger_reasons=["bright_core_roi_support_insufficient"],
        )
        return report, {"before": before_rgb, "after": after_rgb, "roi": roi}

    before_chroma = _chromaticity(before_rgb)
    after_chroma = _chromaticity(after_rgb)
    chroma_drift = 0.5 * np.sum(np.abs(after_chroma - before_chroma), axis=0)
    after_saturation = _saturation(after_rgb)
    before_dominant = np.argmax(before_chroma, axis=0)
    after_dominant = np.argmax(after_chroma, axis=0)
    dominant_changed = before_dominant != after_dominant
    dominant_advantage = _dominance_margin(after_chroma)
    anomaly = (
        roi
        & (chroma_drift >= CHROMA_DRIFT_MIN)
        & (after_saturation >= SATURATION_MIN)
        & dominant_changed
        & (dominant_advantage >= DOMINANT_ADVANTAGE_MIN)
    )
    labels, areas = _connected_components8(anomaly)
    largest_area = max(areas, default=0)
    largest_ratio = float(largest_area / roi_support)
    anomaly_pixels = int(np.count_nonzero(anomaly))
    anomaly_ratio = float(anomaly_pixels / roi_support)
    broad_platform = (
        roi
        & (chroma_drift >= CHROMA_DRIFT_MIN)
        & (after_saturation >= BROAD_PLATFORM_SATURATION_MIN)
    )
    broad_labels, broad_areas = _connected_components8(broad_platform)
    broad_largest_area = max(broad_areas, default=0)
    broad_largest_ratio = float(broad_largest_area / roi_support)
    broad_platform_rejected = bool(
        anomaly_ratio > BROAD_PLATFORM_STRONG_ANOMALY_RATIO_MIN
        and broad_largest_ratio > BROAD_PLATFORM_COMPONENT_RATIO_MAX
    )
    report["measurements"] = {
        "anomaly_pixels": anomaly_pixels,
        "anomaly_ratio_of_roi": anomaly_ratio,
        "component_count": len(areas),
        "component_areas_desc": sorted((int(area) for area in areas), reverse=True)[:32],
        "largest_component_pixels": int(largest_area),
        "largest_component_ratio_of_roi": largest_ratio,
        "roi_chroma_drift_p99": float(np.quantile(chroma_drift[roi], 0.99)),
        "roi_post_saturation_p99": float(np.quantile(after_saturation[roi], 0.99)),
        "roi_dominant_change_ratio": float(np.mean(dominant_changed[roi])),
        "broad_platform_pixels": int(np.count_nonzero(broad_platform)),
        "broad_platform_ratio_of_roi": float(
            np.count_nonzero(broad_platform) / roi_support
        ),
        "broad_platform_component_count": len(broad_areas),
        "broad_platform_component_areas_desc": sorted(
            (int(area) for area in broad_areas), reverse=True
        )[:32],
        "broad_platform_largest_component_pixels": int(broad_largest_area),
        "broad_platform_largest_component_ratio_of_roi": broad_largest_ratio,
    }
    if broad_platform_rejected:
        report.update(
            status="hard_failed",
            accepted=False,
            final_action="reject_spcc_to_pcc",
            trigger_reasons=["broad_core_chroma_platform"],
        )
    elif largest_ratio > COMPONENT_REPAIR_RATIO:
        report.update(
            status="repair_required",
            accepted=False,
            final_action="attempt_local_core_chroma_rollback",
            trigger_reasons=["largest_anomaly_component_exceeded_repair_line"],
        )
    elif largest_ratio > COMPONENT_ACCEPTED_RATIO:
        report.update(
            status="advisory",
            accepted=True,
            final_action="accept_spcc_with_advisory",
            trigger_reasons=["largest_anomaly_component_above_acceptance_line"],
        )
    else:
        report.update(status="ok", accepted=True, final_action="accept_spcc")
    return report, {
        "before": before_rgb,
        "after": after_rgb,
        "after_native": after_domain["native_chw"],
        "after_scale": after_domain["scale"],
        "roi": roi,
        "anomaly": anomaly,
        "labels": labels,
        "areas": areas,
        "broad_platform": broad_platform,
        "broad_platform_labels": broad_labels,
        "broad_platform_areas": broad_areas,
    }


def _repair_candidate(context: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    before = np.asarray(context["before"], dtype=np.float64)
    after = np.asarray(context["after"], dtype=np.float64)
    labels = np.asarray(context["labels"], dtype=np.int32)
    areas = list(context["areas"])
    roi_support = max(int(np.count_nonzero(context["roi"])), 1)
    selected_labels = [
        index + 1
        for index, area in enumerate(areas)
        if float(area / roi_support) > COMPONENT_ACCEPTED_RATIO
    ]
    selected = np.isin(labels, np.asarray(selected_labels, dtype=np.int32))
    expanded = _dilate8(selected, REPAIR_DILATION_RADIUS)
    weight = expanded.astype(np.float64)
    for _ in range(REPAIR_FEATHER_PASSES):
        weight = _box_blur_gray(weight)
    weight = np.clip(weight, 0.0, 1.0)
    support = weight > 0.0

    after_y = _luma(after)
    positive_before = np.clip(before, 0.0, None)
    before_y = _luma(positive_before)
    reference = positive_before * (
        after_y / np.maximum(before_y, 1e-12)
    )[np.newaxis, ...]
    candidate = after.copy()
    candidate[:, support] = (
        after[:, support] * (1.0 - weight[support])[np.newaxis, :]
        + reference[:, support] * weight[support][np.newaxis, :]
    )
    return candidate, {
        "selected_component_labels": selected_labels,
        "selected_component_count": len(selected_labels),
        "selected_seed_pixels": int(np.count_nonzero(selected)),
        "dilation_radius": REPAIR_DILATION_RADIUS,
        "dilated_pixels": int(np.count_nonzero(expanded)),
        "feather_smooth_passes": REPAIR_FEATHER_PASSES,
        "support": support,
        "support_pixels": int(np.count_nonzero(support)),
        "support_ratio_of_image": float(np.mean(support)),
    }


def evaluate_and_repair_spcc_bright_core(
    before: Any,
    after: Any,
    *,
    target_type: str,
    target_profile: Optional[Dict[str, Any]],
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Accept, repair, or reject one SPCC candidate under the fixed policy."""
    initial, context = assess_spcc_bright_core_color(
        before,
        after,
        target_type=target_type,
        target_profile=target_profile,
    )
    if not initial.get("applicable", False):
        return np.asarray(after).copy(), initial
    if initial.get("accepted", False):
        return np.asarray(after).copy(), initial
    if initial.get("status") != "repair_required" or not context.get("areas"):
        return None, initial

    candidate, repair = _repair_candidate(context)
    support = np.asarray(repair.pop("support"), dtype=bool)
    after_rgb = np.asarray(context["after"], dtype=np.float64)
    finite = bool(np.all(np.isfinite(candidate)))
    support_ratio = float(repair["support_ratio_of_image"])
    outside_exact = bool(np.array_equal(candidate[:, ~support], after_rgb[:, ~support]))
    if np.any(support):
        luma_error = np.abs(_luma(candidate) - _luma(after_rgb))[support]
        luma_error_p99 = float(np.quantile(luma_error, 0.99))
    else:
        luma_error_p99 = float("inf")
    before_cap = after_rgb >= CLIP_THRESHOLD
    candidate_cap = candidate >= CLIP_THRESHOLD
    new_clip_ratio = float(np.mean(candidate_cap & ~before_cap))

    post, _ = assess_spcc_bright_core_color(
        context["before"],
        candidate,
        target_type=target_type,
        target_profile=target_profile,
    )
    post_largest = float(
        (post.get("measurements") or {}).get(
            "largest_component_ratio_of_roi",
            float("inf"),
        )
    )
    checks = {
        "largest_component_ratio": {
            "value": post_largest,
            "limit": COMPONENT_ACCEPTED_RATIO,
            "passed": post_largest <= COMPONENT_ACCEPTED_RATIO,
        },
        "modified_support_ratio": {
            "value": support_ratio,
            "limit": REPAIR_SUPPORT_RATIO_MAX,
            "passed": support_ratio <= REPAIR_SUPPORT_RATIO_MAX,
        },
        "luma_abs_error_p99": {
            "value": luma_error_p99,
            "limit": REPAIR_LUMA_ERROR_P99_MAX,
            "passed": luma_error_p99 <= REPAIR_LUMA_ERROR_P99_MAX,
        },
        "new_clip_ratio": {
            "value": new_clip_ratio,
            "limit": REPAIR_NEW_CLIP_RATIO_MAX,
            "passed": new_clip_ratio <= REPAIR_NEW_CLIP_RATIO_MAX,
        },
        "finite": {"passed": finite},
        "outside_support_exact": {"passed": outside_exact},
    }
    passed = all(bool(check.get("passed")) for check in checks.values())
    report = dict(initial)
    report["repair"] = {
        **repair,
        "repair_mask": {
            "source": "anomaly_components_above_0.5_percent_of_core_roi",
            "selected_component_count": repair["selected_component_count"],
            "seed_pixels": repair["selected_seed_pixels"],
            "dilated_pixels": repair["dilated_pixels"],
            "dilation_radius": repair["dilation_radius"],
            "feather_smooth_passes": repair["feather_smooth_passes"],
            "support_pixels": repair["support_pixels"],
            "support_ratio_of_image": repair["support_ratio_of_image"],
        },
        "method": "SPCC_LOCAL_CORE_CHROMA_ROLLBACK",
        "reference_rgb_direction": "stage4_pre_pcc",
        "preserved_luminance": "SPCC Rec.709",
        "checks": checks,
        "post_repair_assessment": post,
        "passed": passed,
    }
    if passed:
        report.update(
            status="repaired",
            accepted=True,
            repaired=True,
            final_action="accept_repaired_spcc",
            trigger_reasons=list(initial.get("trigger_reasons") or []),
        )
        native_candidate = np.asarray(
            context["after_native"],
            dtype=np.float64,
        ).copy()
        native_candidate[:, support] = (
            candidate[:, support] * float(context["after_scale"])
        )
        return np.asarray(native_candidate, dtype=np.float32), report

    failed_checks = [name for name, check in checks.items() if not check["passed"]]
    report.update(
        status="hard_failed",
        accepted=False,
        repaired=False,
        final_action="reject_spcc_to_pcc",
        trigger_reasons=list(initial.get("trigger_reasons") or [])
        + [f"repair_check_failed:{name}" for name in failed_checks],
    )
    return None, report


__all__ = [
    "assess_spcc_bright_core_color",
    "evaluate_and_repair_spcc_bright_core",
]
