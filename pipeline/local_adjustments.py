"""Reusable deterministic curves, masks, and local-adjustment primitives."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np


LOCAL_ADJUSTMENT_SCHEMA = "starun.local-adjustment-recipe.v1"
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _as_rgb_float(image: Any) -> tuple[np.ndarray, np.ndarray, str]:
    source = np.asarray(image)
    if source.ndim != 3:
        raise ValueError("local adjustments require an RGB image")
    if source.shape[0] == 3:
        rgb_source = source
        layout = "chw"
    elif source.shape[-1] == 3:
        rgb_source = np.transpose(source, (2, 0, 1))
        layout = "hwc"
    else:
        raise ValueError(f"expected RGB input, got shape={source.shape}")
    rgb = rgb_source.astype(np.float32, copy=True)
    if np.issubdtype(source.dtype, np.integer):
        rgb /= max(1.0, float(np.iinfo(source.dtype).max))
    if not np.all(np.isfinite(rgb)):
        raise ValueError("nonfinite input pixels")
    return source, rgb, layout


def _restore(source: np.ndarray, rgb: np.ndarray, layout: str) -> np.ndarray:
    output = rgb if layout == "chw" else np.transpose(rgb, (1, 2, 0))
    if np.issubdtype(source.dtype, np.integer):
        maximum = float(np.iinfo(source.dtype).max)
        return np.clip(output * maximum, 0.0, maximum).astype(source.dtype)
    return output.astype(np.float32, copy=False)


def _blur3(plane: np.ndarray) -> np.ndarray:
    padded = np.pad(plane, ((1, 1), (1, 1)), mode="reflect")
    return (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 1:-1]
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    ) / 9.0


def _smoothstep(values: np.ndarray, low: float, high: float) -> np.ndarray:
    width = max(float(high) - float(low), 1e-7)
    normalized = np.clip((values - float(low)) / width, 0.0, 1.0)
    return normalized * normalized * (3.0 - 2.0 * normalized)


def _rgb_hue_and_saturation(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return HSL-style hue/saturation planes for a CHW RGB image.

    The hue-mask model is adapted from AstroColorMixer by Yannick Dutertre
    (Cuiv), itself based on Patrick Cosgrove's Astro Color Mixer workflow.
    The upstream implementation is GPL-3.0-or-later, matching this project.
    """
    red, green, blue = rgb
    maximum = np.max(rgb, axis=0)
    minimum = np.min(rgb, axis=0)
    delta = maximum - minimum
    lightness = (maximum + minimum) * 0.5

    saturation = np.zeros_like(lightness, dtype=np.float32)
    chromatic = delta > 1e-7
    low_lightness = chromatic & (lightness <= 0.5)
    high_lightness = chromatic & ~low_lightness
    saturation[low_lightness] = delta[low_lightness] / np.maximum(
        maximum[low_lightness] + minimum[low_lightness],
        1e-7,
    )
    saturation[high_lightness] = delta[high_lightness] / np.maximum(
        2.0 - maximum[high_lightness] - minimum[high_lightness],
        1e-7,
    )

    hue = np.zeros_like(lightness, dtype=np.float32)
    red_max = chromatic & (maximum == red)
    green_max = chromatic & (maximum == green)
    blue_max = chromatic & (maximum == blue)
    hue[red_max] = (
        (green[red_max] - blue[red_max]) / np.maximum(delta[red_max], 1e-7)
    ) % 6.0
    hue[green_max] = (
        (blue[green_max] - red[green_max])
        / np.maximum(delta[green_max], 1e-7)
    ) + 2.0
    hue[blue_max] = (
        (red[blue_max] - green[blue_max])
        / np.maximum(delta[blue_max], 1e-7)
    ) + 4.0
    return (hue * 60.0) % 360.0, np.clip(saturation, 0.0, 1.0)


def _circular_hue_mask(
    hue: np.ndarray,
    *,
    center: float,
    width: float,
    feather: float,
) -> np.ndarray:
    distance = np.abs((hue % 360.0) - (float(center) % 360.0))
    distance = np.minimum(distance, 360.0 - distance)
    outer = max(5.0, min(90.0, float(width)))
    feather_value = max(0.05, min(1.0, float(feather)))
    inner = outer * (1.0 - feather_value)
    if outer - inner <= 1e-7:
        return (distance <= outer).astype(np.float32)
    transition = np.clip((distance - inner) / (outer - inner), 0.0, 1.0)
    return (1.0 - transition * transition * (3.0 - 2.0 * transition)).astype(
        np.float32
    )


def _apply_hue_selective_saturation(
    rgb: np.ndarray,
    operation: Mapping[str, Any],
    effective_mask: np.ndarray,
) -> tuple[np.ndarray, list[Dict[str, Any]]]:
    bands = operation.get("bands") or ()
    if not isinstance(bands, Sequence) or isinstance(bands, (str, bytes)):
        raise ValueError("hue-selective saturation bands must be a sequence")
    if not bands:
        raise ValueError("hue-selective saturation requires at least one band")

    hue, saturation = _rgb_hue_and_saturation(rgb)
    luma = np.tensordot(_LUMA, rgb, axes=(0, 0))
    sat_floor = max(0.0, min(0.40, float(operation.get("sat_floor", 0.03))))
    sat_full = max(
        sat_floor + 1e-4,
        min(0.70, float(operation.get("sat_full", 0.18))),
    )
    dark_floor = max(0.0, min(0.50, float(operation.get("dark_floor", 0.02))))
    dark_full = max(
        dark_floor + 1e-4,
        min(0.75, float(operation.get("dark_full", 0.12))),
    )
    highlight_start = max(
        0.40,
        min(0.98, float(operation.get("highlight_start", 0.85))),
    )
    highlight_full = max(
        highlight_start + 1e-4,
        min(1.0, float(operation.get("highlight_full", 0.98))),
    )
    protection = (
        _smoothstep(saturation, sat_floor, sat_full)
        * _smoothstep(luma, dark_floor, dark_full)
        * (1.0 - _smoothstep(luma, highlight_start, highlight_full))
    )

    adjustment = np.zeros_like(luma, dtype=np.float32)
    band_reports: list[Dict[str, Any]] = []
    for index, raw_band in enumerate(bands):
        if not isinstance(raw_band, Mapping):
            raise ValueError("hue-selective saturation band must be a mapping")
        center = float(raw_band.get("center", 0.0)) % 360.0
        width = max(5.0, min(90.0, float(raw_band.get("width", 45.0))))
        feather = max(0.05, min(1.0, float(raw_band.get("feather", 0.80))))
        amount = max(-0.15, min(0.15, float(raw_band.get("amount", 0.0))))
        band_mask = _circular_hue_mask(
            hue,
            center=center,
            width=width,
            feather=feather,
        )
        adjustment += amount * band_mask
        band_reports.append(
            {
                "id": str(raw_band.get("id") or f"band-{index + 1}"),
                "center": center,
                "width": width,
                "feather": feather,
                "amount": amount,
            }
        )

    # Keep overlapping hue bands inside the same conservative color budget.
    adjustment = np.clip(adjustment, -0.15, 0.15)
    color_weight = adjustment * protection * effective_mask
    chroma = rgb - luma[None]
    adjusted = luma[None] + chroma * (1.0 + color_weight[None])
    return adjusted.astype(np.float32), band_reports


def dilate_mask(mask: Any, iterations: int = 1) -> np.ndarray:
    result = np.asarray(mask, dtype=np.float32) > 0.5
    for _ in range(max(0, int(iterations))):
        padded = np.pad(result, ((1, 1), (1, 1)), mode="constant")
        result = np.logical_or.reduce(
            [
                padded[y : y + result.shape[0], x : x + result.shape[1]]
                for y in range(3)
                for x in range(3)
            ]
        )
    return result.astype(np.float32)


def erode_mask(mask: Any, iterations: int = 1) -> np.ndarray:
    result = np.asarray(mask, dtype=np.float32) > 0.5
    for _ in range(max(0, int(iterations))):
        padded = np.pad(
            result,
            ((1, 1), (1, 1)),
            mode="constant",
            constant_values=True,
        )
        result = np.logical_and.reduce(
            [
                padded[y : y + result.shape[0], x : x + result.shape[1]]
                for y in range(3)
                for x in range(3)
            ]
        )
    return result.astype(np.float32)


def feather_mask(mask: Any, radius: int = 2) -> np.ndarray:
    result = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    for _ in range(max(0, min(8, int(radius)))):
        result = _blur3(result)
    return np.clip(result, 0.0, 1.0)


def build_local_masks(image: Any) -> Dict[str, Any]:
    """Build deterministic background, subject, faint, core, detail, and chroma masks."""
    _source, rgb, _layout = _as_rgb_float(image)
    luma = np.tensordot(_LUMA, rgb, axes=(0, 0))
    q10, q35, q55, q75, q92, q995 = (
        float(value)
        for value in np.quantile(
            luma[np.isfinite(luma)],
            (0.10, 0.35, 0.55, 0.75, 0.92, 0.995),
        )
    )
    smooth = _blur3(_blur3(luma))
    detail_raw = np.abs(luma - smooth)
    detail_scale = max(float(np.quantile(detail_raw, 0.98)), 1e-7)
    detail = np.clip(detail_raw / detail_scale, 0.0, 1.0)
    core = feather_mask(_smoothstep(luma, q92, q995), 1)
    subject = (
        _smoothstep(luma, q55, q92)
        * (1.0 - 0.92 * core)
        * (1.0 - 0.30 * detail)
    )
    faint = (
        _smoothstep(luma, q35, q75)
        * (1.0 - _smoothstep(luma, q75, q92))
        * (1.0 - 0.75 * core)
    )
    background = (
        (1.0 - _smoothstep(luma, q10, q55))
        * (1.0 - 0.60 * detail)
    )
    channel_max = np.max(rgb, axis=0)
    channel_min = np.min(rgb, axis=0)
    chroma = np.clip(
        (channel_max - channel_min) / np.maximum(luma, 1e-5),
        0.0,
        1.0,
    )
    masks = {
        "background": np.clip(background, 0.0, 1.0).astype(np.float32),
        "subject": np.clip(subject, 0.0, 1.0).astype(np.float32),
        "faint": np.clip(faint, 0.0, 1.0).astype(np.float32),
        "core": np.clip(core, 0.0, 1.0).astype(np.float32),
        "detail": detail.astype(np.float32),
        "chroma": chroma.astype(np.float32),
    }
    return {
        "schema": LOCAL_ADJUSTMENT_SCHEMA,
        "masks": masks,
        "coverage": {
            name: float(np.mean(mask > 0.05))
            for name, mask in masks.items()
        },
        "luminance_quantiles": {
            "p10": q10,
            "p35": q35,
            "p55": q55,
            "p75": q75,
            "p92": q92,
            "p99_5": q995,
        },
    }


def _validated_curve(
    points: Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray]:
    if len(points) < 2:
        raise ValueError("curve requires at least two points")
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("curve points must be [x, y] pairs")
    x_values = array[:, 0]
    y_values = array[:, 1]
    if (
        np.any(~np.isfinite(array))
        or np.any(np.diff(x_values) <= 0.0)
        or np.any(np.diff(y_values) < 0.0)
        or x_values[0] < 0.0
        or x_values[-1] > 1.0
        or np.any(y_values < 0.0)
        or np.any(y_values > 1.0)
    ):
        raise ValueError("curve must be finite and monotonic in [0, 1]")
    return x_values, y_values


def apply_monotonic_curve(
    rgb: np.ndarray,
    points: Sequence[Sequence[float]],
    mask: np.ndarray,
    *,
    opacity: float = 1.0,
) -> np.ndarray:
    x_values, y_values = _validated_curve(points)
    luma = np.tensordot(_LUMA, rgb, axes=(0, 0))
    curved_luma = np.interp(luma, x_values, y_values).astype(np.float32)
    curved = rgb * (
        curved_luma / np.maximum(luma, 1e-7)
    )[None, :, :]
    weight = (
        np.clip(mask, 0.0, 1.0)
        * max(0.0, min(1.0, float(opacity)))
    )
    return rgb * (1.0 - weight[None]) + curved * weight[None]


def apply_local_adjustment_recipe(
    image: Any,
    recipe: Mapping[str, Any],
    *,
    masks: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Apply a versioned local recipe and accept it only through safety metrics."""
    source, rgb, layout = _as_rgb_float(image)
    original = rgb.copy()
    if str(recipe.get("schema") or "") != LOCAL_ADJUSTMENT_SCHEMA:
        raise ValueError("unsupported local-adjustment recipe schema")
    mask_report = build_local_masks(rgb)
    available_masks = dict(mask_report["masks"])
    for name, value in dict(masks or {}).items():
        plane = np.asarray(value, dtype=np.float32)
        if plane.shape != rgb.shape[1:]:
            raise ValueError(
                f"mask {name!r} shape {plane.shape} != {rgb.shape[1:]}"
            )
        available_masks[str(name)] = np.clip(plane, 0.0, 1.0)

    result = rgb.copy()
    used_union = np.zeros(rgb.shape[1:], dtype=np.float32)
    operation_reports: list[Dict[str, Any]] = []
    hue_selective_used = False
    for index, operation in enumerate(recipe.get("operations") or []):
        kind = str(operation.get("type") or "")
        mask_name = str(operation.get("mask") or "subject")
        if mask_name not in available_masks:
            raise ValueError(f"unknown local-adjustment mask: {mask_name}")
        mask = available_masks[mask_name]
        opacity = max(
            0.0,
            min(1.0, float(operation.get("opacity", 1.0))),
        )
        effective_mask = mask * opacity
        used_union = np.maximum(used_union, effective_mask)
        requested_amount: float | None = None
        bounded_amount: float | None = None
        if kind == "curve":
            result = apply_monotonic_curve(
                result,
                operation.get("points") or (),
                mask,
                opacity=opacity,
            )
        elif kind == "saturation":
            requested_amount = float(operation.get("amount", 0.0))
            bounded_amount = max(
                -0.50,
                min(0.50, requested_amount),
            )
            luma = np.tensordot(_LUMA, result, axes=(0, 0))
            adjusted = luma[None] + (result - luma[None]) * (
                1.0 + bounded_amount
            )
            result = (
                result * (1.0 - effective_mask[None])
                + adjusted * effective_mask[None]
            )
        elif kind == "hue_selective_saturation":
            result, band_reports = _apply_hue_selective_saturation(
                result,
                operation,
                effective_mask,
            )
            hue_selective_used = True
        elif kind == "local_contrast":
            amount = max(
                -0.25,
                min(0.25, float(operation.get("amount", 0.0))),
            )
            radius = max(1, min(6, int(operation.get("radius", 2))))
            smooth = result
            for _ in range(radius):
                smooth = np.stack(
                    [_blur3(channel) for channel in smooth],
                    axis=0,
                )
            adjusted = result + (result - smooth) * amount
            result = (
                result * (1.0 - effective_mask[None])
                + adjusted * effective_mask[None]
            )
        else:
            raise ValueError(f"unsupported local-adjustment operation: {kind}")
        operation_report = {
            "index": index,
            "type": kind,
            "mask": mask_name,
            "opacity": opacity,
            "mask_coverage": float(np.mean(mask > 0.05)),
            "effective_mask_mean": float(np.mean(effective_mask)),
            "effective_mask_peak": float(np.max(effective_mask)),
        }
        if kind == "saturation":
            operation_report.update(
                {
                    "requested_amount": requested_amount,
                    "amount": bounded_amount,
                    "effective_amount": float(
                        bounded_amount * np.mean(effective_mask)
                    ),
                    "effective_amount_peak": float(
                        bounded_amount * np.max(effective_mask)
                    ),
                }
            )
        if kind == "hue_selective_saturation":
            maximum_band_amount = max(
                (
                    abs(float(band.get("amount", 0.0)))
                    for band in band_reports
                    if isinstance(band, dict)
                ),
                default=0.0,
            )
            operation_report.update(
                {
                    "profile": str(operation.get("profile") or "broadband"),
                    "bands": band_reports,
                    "effective_amount": float(
                        maximum_band_amount * np.mean(effective_mask)
                    ),
                    "effective_amount_peak": float(
                        maximum_band_amount * np.max(effective_mask)
                    ),
                }
            )
        operation_reports.append(operation_report)

    candidate = np.clip(result, 0.0, 1.0).astype(np.float32)
    before_luma = np.tensordot(_LUMA, original, axes=(0, 0))
    after_luma = np.tensordot(_LUMA, candidate, axes=(0, 0))
    before_chroma = np.max(original, axis=0) - np.min(original, axis=0)
    after_chroma = np.max(candidate, axis=0) - np.min(candidate, axis=0)
    background_mask = available_masks["background"] > 0.50
    core_mask = available_masks["core"] > 0.50
    active_mask = used_union > 0.05
    outside = used_union <= 1e-6
    before_clip = float(np.mean((original <= 0.0) | (original >= 1.0)))
    after_clip = float(np.mean((candidate <= 0.0) | (candidate >= 1.0)))
    background_drift = (
        abs(
            float(np.median(after_luma[background_mask]))
            - float(np.median(before_luma[background_mask]))
        )
        if np.count_nonzero(background_mask) >= 32
        else 0.0
    )
    core_drift = (
        abs(
            float(np.quantile(after_luma[core_mask], 0.99))
            - float(np.quantile(before_luma[core_mask], 0.99))
        )
        if np.count_nonzero(core_mask) >= 16
        else 0.0
    )
    outside_change = (
        float(
            np.mean(
                np.max(np.abs(candidate - original), axis=0)[outside]
                > 1e-6
            )
        )
        if np.count_nonzero(outside)
        else 0.0
    )
    changed = float(
        np.mean(np.max(np.abs(candidate - original), axis=0) > 1e-5)
    )
    background_chroma_drift = (
        abs(
            float(np.median(after_chroma[background_mask]))
            - float(np.median(before_chroma[background_mask]))
        )
        if np.count_nonzero(background_mask) >= 32
        else 0.0
    )
    active_chroma_p95_growth = (
        float(np.quantile(after_chroma[active_mask], 0.95))
        - float(np.quantile(before_chroma[active_mask], 0.95))
        if np.count_nonzero(active_mask) >= 32
        else 0.0
    )
    saturation_operation_reports = [
        operation
        for operation in operation_reports
        if str(operation.get("type"))
        in {"saturation", "hue_selective_saturation"}
    ]
    saturation_active_mask = np.zeros(rgb.shape[1:], dtype=np.float32)
    for operation in saturation_operation_reports:
        saturation_active_mask = np.maximum(
            saturation_active_mask,
            available_masks[str(operation["mask"])]
            * float(operation.get("opacity", 1.0)),
        )
    saturation_pixels = saturation_active_mask > 0.05
    saturation_chroma_abs_delta_p95 = (
        float(
            np.quantile(
                np.abs(after_chroma - before_chroma)[saturation_pixels],
                0.95,
            )
        )
        if np.count_nonzero(saturation_pixels) >= 32
        else 0.0
    )
    saturation_chroma_p95_delta = (
        float(np.quantile(after_chroma[saturation_pixels], 0.95))
        - float(np.quantile(before_chroma[saturation_pixels], 0.95))
        if np.count_nonzero(saturation_pixels) >= 32
        else 0.0
    )
    if not saturation_operation_reports:
        saturation_effect_status = "not_requested"
    elif np.count_nonzero(saturation_pixels) < 32:
        saturation_effect_status = "insufficient_evidence"
    elif saturation_chroma_abs_delta_p95 <= 1e-6:
        saturation_effect_status = "no_effect"
    else:
        saturation_effect_status = "effective"
    for operation in saturation_operation_reports:
        operation["effect_status"] = saturation_effect_status
    metrics = {
        "clip_growth": after_clip - before_clip,
        "background_median_drift": background_drift,
        "core_p99_drift": core_drift,
        "outside_mask_changed_ratio": outside_change,
        "changed_pixel_ratio": changed,
        "active_mask_coverage": float(np.mean(used_union > 0.05)),
        "background_chroma_median_drift": background_chroma_drift,
        "active_chroma_p95_growth": active_chroma_p95_growth,
        "saturation_active_chroma_abs_delta_p95": (
            saturation_chroma_abs_delta_p95
        ),
        "saturation_active_chroma_p95_delta": saturation_chroma_p95_delta,
    }
    limits = {
        "clip_growth_max": 0.002,
        "background_median_drift_max": 0.006,
        "core_p99_drift_max": 0.025,
        "outside_mask_changed_ratio_max": 0.001,
        "background_chroma_median_drift_max": 0.004,
        "active_chroma_p95_growth_max": 0.060,
    }
    issues: list[str] = []
    if not np.all(np.isfinite(candidate)):
        issues.append("nonfinite_output")
    for metric_name, limit_name in (
        ("clip_growth", "clip_growth_max"),
        ("background_median_drift", "background_median_drift_max"),
        ("core_p99_drift", "core_p99_drift_max"),
        ("outside_mask_changed_ratio", "outside_mask_changed_ratio_max"),
    ):
        if metrics[metric_name] > limits[limit_name]:
            issues.append(metric_name)
    if hue_selective_used:
        for metric_name, limit_name in (
            (
                "background_chroma_median_drift",
                "background_chroma_median_drift_max",
            ),
            ("active_chroma_p95_growth", "active_chroma_p95_growth_max"),
        ):
            if metrics[metric_name] > limits[limit_name]:
                issues.append(metric_name)
    if operation_reports and changed <= 0.0:
        issues.append("no_effect")
    accepted = not issues
    report = {
        "schema": LOCAL_ADJUSTMENT_SCHEMA,
        "status": "accepted" if accepted else "rejected",
        "accepted": accepted,
        "recipe_id": str(recipe.get("id") or "anonymous"),
        "operations": operation_reports,
        "mask_coverage": mask_report["coverage"],
        "metrics": metrics,
        "limits": limits,
        "issues": issues,
        "saturation_effect": {
            "status": saturation_effect_status,
            "active_pixel_count": int(np.count_nonzero(saturation_pixels)),
            "active_mask_coverage": float(np.mean(saturation_pixels)),
            "chroma_abs_delta_p95": saturation_chroma_abs_delta_p95,
            "chroma_p95_delta": saturation_chroma_p95_delta,
        },
        "transaction": {
            "baseline": "stage8_input_starless.fit",
            "candidate": "stage8_enhanced.fit",
            "applied_in_memory": accepted,
        },
    }
    return _restore(source, candidate, layout), report


__all__ = [
    "LOCAL_ADJUSTMENT_SCHEMA",
    "apply_local_adjustment_recipe",
    "apply_monotonic_curve",
    "build_local_masks",
    "dilate_mask",
    "erode_mask",
    "feather_mask",
]
