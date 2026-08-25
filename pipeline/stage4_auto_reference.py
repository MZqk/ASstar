"""Deterministic non-photometric Stage 4 color fallback.

This module deliberately has no Siril or filesystem side effects.  It builds
and evaluates heuristic color candidates from an immutable input image; the
Stage 4 orchestrator owns checkpointing, pixel writes and report publication.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from background_sampling import (
    build_safe_background_samples,
    split_background_sample_points,
)


AUTO_REFERENCE_SCHEMA = "starun.stage4-auto-local-reference.v2"
BACKGROUND_METHOD = "AUTO_BACKGROUND_NEUTRALIZATION"
WHITE_REGION_METHOD = "AUTO_BACKGROUND_WHITE_REGION"
# Compatibility alias for existing report consumers.  New reports use a
# single rectangular white reference, not a frame-wide ensemble.
STAR_ENSEMBLE_METHOD = WHITE_REGION_METHOD
PSEUDO_WHITE_REFERENCE = "single_rectangular_white_reference"
STAR_ENSEMBLE_OBJECT_CAP = 256
BACKGROUND_P90_GROWTH_MAX = 1.05
BACKGROUND_IMPROVED_FRACTION_MIN = 0.75
BACKGROUND_SPATIAL_CHROMA_GROWTH_MAX = 1.05
BACKGROUND_SPATIAL_CHROMA_ABSOLUTE_TOLERANCE = 0.0005


@dataclass(frozen=True)
class _ImageAdapter:
    source_shape: Tuple[int, ...]
    source_dtype: np.dtype
    scale: float
    chw: np.ndarray
    restore: Callable[[np.ndarray], np.ndarray]


def _clamp(value: Any, lower: float, upper: float, default: float) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        resolved = default
    if not math.isfinite(resolved):
        resolved = default
    return max(lower, min(upper, resolved))


def _cfg(config: Any, name: str, default: Any) -> Any:
    return getattr(config, name, default) if config is not None else default


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _resolve_scale(source: np.ndarray, chw: np.ndarray) -> float:
    if np.issubdtype(source.dtype, np.integer):
        return float(max(1, np.iinfo(source.dtype).max))
    finite = chw[np.isfinite(chw)]
    if finite.size == 0:
        return 1.0
    peak = float(np.quantile(np.abs(finite), 0.99999))
    if peak <= 1.5:
        return 1.0
    if peak <= 255.5:
        return 255.0
    if peak <= 65535.5:
        return 65535.0
    return max(peak, 1.0)


def _cast_like(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.rint(values), info.min, info.max).astype(dtype, copy=False)
    return values.astype(dtype, copy=False)


def _adapt_image(image: Any) -> _ImageAdapter:
    source = np.asarray(image)
    if source.size == 0:
        raise ValueError("image buffer is empty")
    if source.ndim == 2:
        raw_chw = source[np.newaxis, ...].astype(np.float64, copy=True)
        layout = "mono"
    elif source.ndim == 3 and source.shape[0] in (1, 3, 4) and source.shape[-1] not in (1, 3, 4):
        raw_chw = source.astype(np.float64, copy=True)
        layout = "chw"
    elif source.ndim == 3 and source.shape[-1] in (1, 3, 4):
        raw_chw = np.moveaxis(source, -1, 0).astype(np.float64, copy=True)
        layout = "hwc"
    elif source.ndim == 3 and source.shape[0] >= 3:
        raw_chw = source.astype(np.float64, copy=True)
        layout = "chw"
    else:
        raise ValueError(f"unsupported image shape: {source.shape}")

    scale = _resolve_scale(source, raw_chw)
    normalized = raw_chw / scale
    original_dtype = source.dtype

    def restore(chw: np.ndarray) -> np.ndarray:
        raw = np.asarray(chw, dtype=np.float64) * scale
        if layout == "mono":
            raw = raw[0]
        elif layout == "hwc":
            raw = np.moveaxis(raw, 0, -1)
        return _cast_like(raw, original_dtype)

    return _ImageAdapter(
        source_shape=tuple(int(value) for value in source.shape),
        source_dtype=original_dtype,
        scale=scale,
        chw=normalized,
        restore=restore,
    )


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def _robust_sigma(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    center = float(np.median(finite))
    return max(1.4826 * float(np.median(np.abs(finite - center))), 0.0)


def _patch_bounds(
    point: Tuple[float, float],
    *,
    width: int,
    height: int,
    radius: int,
) -> Tuple[int, int, int, int]:
    x = int(round(float(point[0])))
    y = height - 1 - int(round(float(point[1])))
    return (
        max(0, x - radius),
        min(width, x + radius + 1),
        max(0, y - radius),
        min(height, y + radius + 1),
    )


def _measure_rgb_patches(
    chw: np.ndarray,
    points: Sequence[Tuple[float, float]],
    *,
    patch_radius: int,
) -> Dict[str, Any]:
    rgb = np.asarray(chw[:3], dtype=np.float64)
    height, width = rgb.shape[1:]
    medians: List[List[float]] = []
    gradients: List[float] = []
    textures: List[float] = []
    accepted_points: List[List[float]] = []
    for point in points:
        x0, x1, y0, y1 = _patch_bounds(
            point,
            width=width,
            height=height,
            radius=patch_radius,
        )
        patch = rgb[:, y0:y1, x0:x1]
        if patch.shape[1] < 3 or patch.shape[2] < 3 or not np.all(np.isfinite(patch)):
            continue
        channel_medians = [float(np.median(channel)) for channel in patch]
        lum = _luminance(patch)
        dx = np.abs(np.diff(lum, axis=1))
        dy = np.abs(np.diff(lum, axis=0))
        gradients.append(
            float(0.5 * (np.median(dx) + np.median(dy)))
        )
        lum_median = float(np.median(lum))
        textures.append(float(np.median(np.abs(lum - lum_median))))
        medians.append(channel_medians)
        accepted_points.append([float(point[0]), float(point[1])])
    values = np.asarray(medians, dtype=np.float64)
    if values.size == 0:
        return {
            "status": "unavailable",
            "sample_count": 0,
            "points": [],
            "channel_medians": [],
            "median_color_error": None,
        }
    errors = np.max(values, axis=1) - np.min(values, axis=1)
    luminance_medians = values @ np.asarray((0.2126, 0.7152, 0.0722))
    return {
        "status": "ready",
        "sample_count": int(values.shape[0]),
        "points": accepted_points,
        "channel_medians": [[float(item) for item in row] for row in values],
        "robust_channel_medians": [
            float(value) for value in np.median(values, axis=0)
        ],
        "median_color_error": float(np.median(errors)),
        "p90_color_error": float(np.quantile(errors, 0.90)),
        "median_luminance": float(np.median(luminance_medians)),
        "luminance_sigma": _robust_sigma(luminance_medians),
        "gradient": float(np.median(gradients)),
        "texture": float(np.median(textures)),
    }


def _rectangle_from_point(
    point: Tuple[float, float],
    *,
    width: int,
    height: int,
    radius: int,
) -> Dict[str, Any]:
    x0, x1, y0, y1 = _patch_bounds(
        point,
        width=width,
        height=height,
        radius=radius,
    )
    return {
        "x": int(x0),
        "y": int(y0),
        "width": int(x1 - x0),
        "height": int(y1 - y0),
        "coordinate_system": "numpy_top_left",
    }


def _select_background_rectangle(
    chw: np.ndarray,
    points: Sequence[Tuple[float, float]],
    *,
    patch_radius: int,
) -> Tuple[Optional[Tuple[float, float]], Dict[str, Any]]:
    """Choose one empty-sky rectangle without using RGB balance as a score."""
    height, width = np.asarray(chw).shape[1:]
    candidates: List[Dict[str, Any]] = []
    for point in points:
        measured = _measure_rgb_patches(
            chw,
            (point,),
            patch_radius=patch_radius,
        )
        if measured.get("status") != "ready":
            continue
        candidate = {
            "point": (float(point[0]), float(point[1])),
            "rectangle": _rectangle_from_point(
                point,
                width=width,
                height=height,
                radius=patch_radius,
            ),
            "median_luminance": float(measured.get("median_luminance") or 0.0),
            "gradient": float(measured.get("gradient") or 0.0),
            "texture": float(measured.get("texture") or 0.0),
        }
        candidates.append(candidate)
    if not candidates:
        return None, {
            "status": "unavailable",
            "reason": "no_measurable_background_rectangle",
        }
    luminances = np.asarray(
        [item["median_luminance"] for item in candidates],
        dtype=np.float64,
    )
    gradients = np.asarray(
        [item["gradient"] for item in candidates],
        dtype=np.float64,
    )
    textures = np.asarray(
        [item["texture"] for item in candidates],
        dtype=np.float64,
    )

    def normalized(values: np.ndarray) -> np.ndarray:
        span = float(np.ptp(values))
        if span <= 1e-12:
            return np.zeros_like(values)
        return (values - float(np.min(values))) / span

    scores = normalized(luminances) + normalized(gradients) + normalized(textures)
    for item, score in zip(candidates, scores):
        item["selection_score"] = float(score)
    selected = min(
        candidates,
        key=lambda item: (
            float(item["selection_score"]),
            float(item["point"][1]),
            float(item["point"][0]),
        ),
    )
    return selected["point"], {
        "status": "selected",
        "selection_policy": (
            "minimum_luminance_gradient_texture_without_rgb_scoring"
        ),
        "candidate_count": len(candidates),
        **selected,
    }


def _spatial_chroma_gradient(measurements: Mapping[str, Any]) -> Optional[float]:
    points = np.asarray(measurements.get("points") or (), dtype=np.float64)
    values = np.asarray(
        measurements.get("channel_medians") or (),
        dtype=np.float64,
    )
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or values.ndim != 2
        or values.shape[1] != 3
        or points.shape[0] != values.shape[0]
        or points.shape[0] < 4
        or not np.all(np.isfinite(points))
        or not np.all(np.isfinite(values))
    ):
        return None
    spans = np.ptp(points, axis=0)
    if float(np.min(spans)) <= 1e-9:
        return None
    normalized = (points - np.min(points, axis=0)) / spans
    design = np.column_stack(
        (np.ones(points.shape[0], dtype=np.float64), normalized)
    )
    if int(np.linalg.matrix_rank(design)) < 3:
        return None
    chroma = np.column_stack(
        (values[:, 0] - values[:, 1], values[:, 2] - values[:, 1])
    )
    coefficients, *_ = np.linalg.lstsq(design, chroma, rcond=None)
    return float(np.linalg.norm(coefficients[1:, :]))


def _background_holdout_comparison(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> Dict[str, Any]:
    before_values = np.asarray(
        before.get("channel_medians") or (),
        dtype=np.float64,
    )
    after_values = np.asarray(
        after.get("channel_medians") or (),
        dtype=np.float64,
    )
    if (
        before_values.ndim != 2
        or before_values.shape[1] != 3
        or after_values.shape != before_values.shape
        or before_values.shape[0] < 4
        or not np.all(np.isfinite(before_values))
        or not np.all(np.isfinite(after_values))
    ):
        return {"status": "unavailable"}
    before_errors = np.ptp(before_values, axis=1)
    after_errors = np.ptp(after_values, axis=1)
    before_p90 = float(np.quantile(before_errors, 0.90))
    after_p90 = float(np.quantile(after_errors, 0.90))
    improved = after_errors < before_errors - 1e-12
    before_gradient = _spatial_chroma_gradient(before)
    after_gradient = _spatial_chroma_gradient(after)
    gradient_growth = (
        _ratio(after_gradient, before_gradient)
        if before_gradient is not None and after_gradient is not None
        else None
    )
    return {
        "status": "ready",
        "sample_count": int(before_errors.size),
        "baseline_p90_color_error": before_p90,
        "candidate_p90_color_error": after_p90,
        "p90_growth": _ratio(after_p90, before_p90),
        "p90_growth_max": BACKGROUND_P90_GROWTH_MAX,
        "improved_count": int(np.count_nonzero(improved)),
        "improved_fraction": float(np.mean(improved)),
        "improved_fraction_min": BACKGROUND_IMPROVED_FRACTION_MIN,
        "baseline_spatial_chroma_gradient": before_gradient,
        "candidate_spatial_chroma_gradient": after_gradient,
        "spatial_chroma_gradient_growth": gradient_growth,
        "spatial_chroma_gradient_growth_max": (
            BACKGROUND_SPATIAL_CHROMA_GROWTH_MAX
        ),
        "spatial_chroma_absolute_tolerance": (
            BACKGROUND_SPATIAL_CHROMA_ABSOLUTE_TOLERANCE
        ),
    }


def _background_holdout_tail_reasons(
    comparison: Mapping[str, Any],
) -> List[str]:
    if comparison.get("status") != "ready":
        return ["heldout_background_tail_measurement_unavailable"]
    reasons: List[str] = []
    before_p90 = float(comparison["baseline_p90_color_error"])
    after_p90 = float(comparison["candidate_p90_color_error"])
    if after_p90 > before_p90 * BACKGROUND_P90_GROWTH_MAX + 1e-8:
        reasons.append("heldout_background_p90_regressed")
    if (
        float(comparison["improved_fraction"])
        < BACKGROUND_IMPROVED_FRACTION_MIN
    ):
        reasons.append("heldout_background_improved_fraction_insufficient")
    before_chroma_gradient = comparison.get("baseline_spatial_chroma_gradient")
    after_chroma_gradient = comparison.get("candidate_spatial_chroma_gradient")
    if before_chroma_gradient is None or after_chroma_gradient is None:
        reasons.append("heldout_background_spatial_chroma_gradient_unavailable")
    elif (
        float(after_chroma_gradient)
        > float(before_chroma_gradient) * BACKGROUND_SPATIAL_CHROMA_GROWTH_MAX
        + BACKGROUND_SPATIAL_CHROMA_ABSOLUTE_TOLERANCE
    ):
        reasons.append(
            "heldout_background_spatial_chroma_gradient_growth_exceeded"
        )
    return reasons


def _subtract_to_lowest(
    chw: np.ndarray,
    fit_measurements: Mapping[str, Any],
) -> Tuple[np.ndarray, List[float]]:
    output = np.asarray(chw, dtype=np.float64).copy()
    medians = np.asarray(
        fit_measurements.get("robust_channel_medians") or (),
        dtype=np.float64,
    )
    if medians.size != 3 or not np.all(np.isfinite(medians)):
        raise ValueError("background channel medians unavailable")
    offsets = np.maximum(medians - float(np.min(medians)), 0.0)
    output[:3] = np.maximum(output[:3] - offsets[:, None, None], 0.0)
    return output, [float(value) for value in offsets]


def _ratio(after: float, before: float) -> float:
    if before <= 1e-12:
        return 1.0 if after <= 1e-12 else float("inf")
    return float(after / before)


def _clip_metrics(chw: np.ndarray) -> Dict[str, float]:
    rgb = np.asarray(chw[:3], dtype=np.float64)
    return {
        "highlight_clip_ratio": float(np.mean(np.max(rgb, axis=0) >= 0.999)),
        "black_clip_ratio": float(np.mean(np.min(rgb, axis=0) <= (1.0 / 65535.0))),
    }


def _build_subject_mask(
    chw: np.ndarray,
    *,
    background_luminance: float,
    background_sigma: float,
    star_threshold: float,
) -> np.ndarray:
    lum = _luminance(np.asarray(chw[:3], dtype=np.float64))
    finite = np.isfinite(lum)
    finite_values = lum[finite]
    if finite_values.size < 64:
        return np.zeros_like(lum, dtype=bool)
    q995 = float(np.quantile(finite_values, 0.995))
    threshold = background_luminance + max(
        5.0 * background_sigma,
        0.10 * max(q995 - background_luminance, 0.0),
        0.002,
    )
    mask = finite & (lum >= threshold) & (lum < star_threshold)
    if int(np.count_nonzero(mask)) < 64:
        return np.zeros_like(lum, dtype=bool)
    return mask


def _subject_chroma_drift(
    before: np.ndarray,
    after: np.ndarray,
    mask: np.ndarray,
) -> Optional[float]:
    if mask.shape != before.shape[1:] or int(np.count_nonzero(mask)) < 64:
        return None
    before_rgb = np.clip(np.asarray(before[:3, mask], dtype=np.float64), 0.0, None)
    after_rgb = np.clip(np.asarray(after[:3, mask], dtype=np.float64), 0.0, None)
    before_sum = np.sum(before_rgb, axis=0)
    after_sum = np.sum(after_rgb, axis=0)
    valid = (before_sum > 1e-8) & (after_sum > 1e-8)
    if int(np.count_nonzero(valid)) < 64:
        return None
    before_chroma = before_rgb[:, valid] / before_sum[valid]
    after_chroma = after_rgb[:, valid] / after_sum[valid]
    distances = np.linalg.norm(after_chroma - before_chroma, axis=0)
    return float(np.quantile(distances, 0.95))


def _safety_gate(
    before: np.ndarray,
    after: np.ndarray,
    *,
    validation_points: Sequence[Tuple[float, float]],
    patch_radius: int,
    subject_mask: np.ndarray,
    config: Any,
) -> Tuple[List[str], Dict[str, Any]]:
    reasons: List[str] = []
    if before.shape != after.shape:
        reasons.append("shape_changed")
    if not np.all(np.isfinite(after)):
        reasons.append("non_finite_pixels")
    before_clip = _clip_metrics(before)
    after_clip = _clip_metrics(after)
    highlight_growth = (
        after_clip["highlight_clip_ratio"] - before_clip["highlight_clip_ratio"]
    )
    black_growth = after_clip["black_clip_ratio"] - before_clip["black_clip_ratio"]
    highlight_max = _clamp(
        _cfg(config, "stage4_auto_reference_highlight_clip_growth_max", 0.002),
        0.0,
        0.05,
        0.002,
    )
    black_max = _clamp(
        _cfg(config, "stage4_auto_reference_black_clip_growth_max", 0.002),
        0.0,
        0.05,
        0.002,
    )
    if highlight_growth > highlight_max:
        reasons.append("highlight_clip_growth_exceeded")
    if black_growth > black_max:
        reasons.append("black_clip_growth_exceeded")

    before_validation = _measure_rgb_patches(
        before, validation_points, patch_radius=patch_radius
    )
    after_validation = _measure_rgb_patches(
        after, validation_points, patch_radius=patch_radius
    )
    gradient_ratio = _ratio(
        float(after_validation.get("gradient") or 0.0),
        float(before_validation.get("gradient") or 0.0),
    )
    texture_ratio = _ratio(
        float(after_validation.get("texture") or 0.0),
        float(before_validation.get("texture") or 0.0),
    )
    gradient_max = _clamp(
        _cfg(config, "stage4_auto_reference_gradient_growth_max", 1.05),
        1.0,
        2.0,
        1.05,
    )
    texture_max = _clamp(
        _cfg(config, "stage4_auto_reference_texture_growth_max", 1.10),
        1.0,
        2.0,
        1.10,
    )
    if gradient_ratio > gradient_max:
        reasons.append("heldout_gradient_growth_exceeded")
    if texture_ratio > texture_max:
        reasons.append("heldout_texture_growth_exceeded")

    subject_drift = _subject_chroma_drift(before, after, subject_mask)
    subject_max = _clamp(
        _cfg(config, "stage4_auto_reference_target_chroma_drift_max", 0.08),
        0.01,
        0.50,
        0.08,
    )
    if subject_drift is not None and subject_drift > subject_max:
        reasons.append("subject_chromaticity_drift_exceeded")
    return reasons, {
        "shape_preserved": before.shape == after.shape,
        "finite": bool(np.all(np.isfinite(after))),
        "clip": {
            "before": before_clip,
            "after": after_clip,
            "highlight_growth": float(highlight_growth),
            "highlight_growth_max": highlight_max,
            "black_growth": float(black_growth),
            "black_growth_max": black_max,
        },
        "heldout_structure": {
            "gradient_ratio": gradient_ratio,
            "gradient_ratio_max": gradient_max,
            "texture_ratio": texture_ratio,
            "texture_ratio_max": texture_max,
        },
        "subject_chromaticity": {
            "mask_pixels": int(np.count_nonzero(subject_mask)),
            "p95_drift": subject_drift,
            "p95_drift_max": subject_max,
        },
    }


def _component_pixels(mask: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: List[Tuple[np.ndarray, np.ndarray]] = []
    for start_y, start_x in np.argwhere(mask):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        ys: List[int] = []
        xs: List[int] = []
        while stack:
            y, x = stack.pop()
            ys.append(y)
            xs.append(x)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny = y + dy
                    nx = x + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and mask[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        components.append(
            (np.asarray(ys, dtype=np.int32), np.asarray(xs, dtype=np.int32))
        )
    return components


def _star_flux(
    chw: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    annulus_y: np.ndarray,
    annulus_x: np.ndarray,
) -> Optional[np.ndarray]:
    rgb = np.asarray(chw[:3], dtype=np.float64)
    if ys.size == 0 or annulus_y.size < 8:
        return None
    local_background = np.median(rgb[:, annulus_y, annulus_x], axis=1)
    flux = np.sum(
        np.maximum(rgb[:, ys, xs] - local_background[:, None], 0.0),
        axis=1,
    )
    if not np.all(np.isfinite(flux)) or float(np.min(flux)) <= 1e-10:
        return None
    return flux


def _find_star_ensemble(
    chw: np.ndarray,
    *,
    background_luminance: float,
    background_sigma: float,
    background_texture: float,
    subject_mask: np.ndarray,
    saturation_ratio_max: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], float]:
    rgb = np.asarray(chw[:3], dtype=np.float64)
    lum = _luminance(rgb)
    finite = np.isfinite(lum) & np.all(np.isfinite(rgb), axis=0)
    valid = lum[finite]
    if valid.size < 256:
        return [], {"status": "unavailable", "reason": "insufficient_finite_pixels"}, 1.0
    threshold = max(
        float(np.quantile(valid, 0.98)),
        background_luminance + max(5.0 * background_sigma, 0.002),
    )
    detection = finite & (lum >= threshold)
    height, width = lum.shape
    maximum_area = max(8, min(256, int(round(height * width * 0.0008))))
    records: List[Dict[str, Any]] = []
    rejection_counts: Dict[str, int] = {}

    def reject(reason: str) -> None:
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    for ys, xs in _component_pixels(detection):
        area = int(ys.size)
        if area < 2 or area > maximum_area:
            reject("component_area")
            continue
        y0, y1 = int(np.min(ys)), int(np.max(ys))
        x0, x1 = int(np.min(xs)), int(np.max(xs))
        box_height = y1 - y0 + 1
        box_width = x1 - x0 + 1
        aspect = max(box_width, box_height) / max(1, min(box_width, box_height))
        compactness = area / max(1, box_width * box_height)
        if aspect > 2.5 or compactness < 0.15:
            reject("non_compact_or_asymmetric")
            continue
        if x0 < 3 or y0 < 3 or x1 >= width - 3 or y1 >= height - 3:
            reject("edge_component")
            continue
        saturated = np.max(rgb[:, ys, xs], axis=0) >= 0.98
        saturation_fraction = float(np.mean(saturated))
        if saturation_fraction > saturation_ratio_max:
            reject("saturated")
            continue

        ay0, ay1 = max(0, y0 - 4), min(height, y1 + 5)
        ax0, ax1 = max(0, x0 - 4), min(width, x1 + 5)
        local_y, local_x = np.mgrid[ay0:ay1, ax0:ax1]
        annulus_mask = np.ones(local_y.shape, dtype=bool)
        annulus_mask[
            np.clip(ys - ay0, 0, annulus_mask.shape[0] - 1),
            np.clip(xs - ax0, 0, annulus_mask.shape[1] - 1),
        ] = False
        annulus_mask &= ~detection[ay0:ay1, ax0:ax1]
        annulus_y = local_y[annulus_mask].astype(np.int32)
        annulus_x = local_x[annulus_mask].astype(np.int32)
        if annulus_y.size < 8:
            reject("annulus_unavailable")
            continue
        annulus_lum = lum[annulus_y, annulus_x]
        local_background = float(np.median(annulus_lum))
        local_texture = float(np.median(np.abs(annulus_lum - local_background)))
        allowed_background = background_luminance + max(
            5.0 * background_sigma,
            0.01,
        )
        if local_background > allowed_background:
            reject("bright_extended_background")
            continue
        if local_texture > max(background_texture * 4.0, 0.01):
            reject("structured_annulus")
            continue
        if float(np.mean(subject_mask[ay0:ay1, ax0:ax1])) > 0.25:
            reject("subject_contamination")
            continue
        flux = _star_flux(chw, ys, xs, annulus_y, annulus_x)
        if flux is None:
            reject("invalid_background_subtracted_flux")
            continue
        log_rg = float(math.log(float(flux[0] / flux[1])))
        log_bg = float(math.log(float(flux[2] / flux[1])))
        center_x = float(np.mean(xs))
        center_y = float(np.mean(ys))
        quadrant = int(center_x >= width / 2.0) + 2 * int(center_y >= height / 2.0)
        records.append(
            {
                "x": center_x,
                "y": center_y,
                "quadrant": quadrant,
                "area": area,
                "bbox": [x0, y0, x1, y1],
                "aspect_ratio": float(aspect),
                "compactness": float(compactness),
                "saturation_fraction": saturation_fraction,
                "flux": flux,
                "log_rg": log_rg,
                "log_bg": log_bg,
                "_ys": ys,
                "_xs": xs,
                "_annulus_y": annulus_y,
                "_annulus_x": annulus_x,
            }
        )
    records.sort(key=lambda item: (item["y"], item["x"]))
    return records, {
        "status": "ready" if records else "insufficient_star_evidence",
        "detection_threshold": float(threshold),
        "detected_component_count": int(sum(rejection_counts.values()) + len(records)),
        "valid_object_count": len(records),
        "quadrants": len({int(item["quadrant"]) for item in records}),
        "maximum_component_area": maximum_area,
        "rejection_counts": rejection_counts,
    }, float(threshold)


def _split_star_records(
    records: Sequence[Dict[str, Any]],
    *,
    validation_ratio: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    source = list(records)
    if len(source) < 16:
        return [], [], {
            "status": "insufficient_samples",
            "sample_count": len(source),
            "minimum_total": 16,
        }
    validation_count = max(4, int(math.ceil(len(source) * validation_ratio)))
    validation_count = min(validation_count, len(source) - 12)
    quadrant_members: Dict[int, int] = {}
    for item in source:
        quadrant = int(item["quadrant"])
        quadrant_members[quadrant] = quadrant_members.get(quadrant, 0) + 1
    eligible = [
        index
        for index, item in enumerate(source)
        if quadrant_members[int(item["quadrant"])] >= 2
    ]
    selected: List[int] = []
    selected_set = set()
    selected_quadrants: Dict[int, int] = {}
    coordinates = np.asarray(
        [[float(item["x"]), float(item["y"])] for item in source],
        dtype=np.float64,
    )
    nearest_distance = np.full(len(source), float("inf"), dtype=np.float64)
    while len(selected) < validation_count:
        best: Optional[int] = None
        best_key: Optional[Tuple[Any, ...]] = None
        for index in eligible:
            if index in selected_set:
                continue
            item = source[index]
            quadrant = int(item["quadrant"])
            if selected_quadrants.get(quadrant, 0) >= quadrant_members[quadrant] - 1:
                continue
            distance = float(nearest_distance[index])
            key = (
                int(quadrant not in selected_quadrants),
                distance,
                -float(item["y"]),
                -float(item["x"]),
            )
            if best_key is None or key > best_key:
                best_key = key
                best = index
        if best is None:
            break
        selected.append(best)
        selected_set.add(best)
        quadrant = int(source[best]["quadrant"])
        selected_quadrants[quadrant] = selected_quadrants.get(quadrant, 0) + 1
        delta = coordinates - coordinates[best]
        distances = np.sqrt(np.sum(delta * delta, axis=1))
        nearest_distance = np.minimum(nearest_distance, distances)
    validation = [item for index, item in enumerate(source) if index in selected_set]
    fit = [item for index, item in enumerate(source) if index not in selected_set]
    fit_quadrants = len({int(item["quadrant"]) for item in fit})
    validation_quadrants = len({int(item["quadrant"]) for item in validation})
    ready = bool(
        len(fit) >= 12
        and len(validation) >= 4
        and fit_quadrants >= 3
        and validation_quadrants >= 3
    )
    return (fit if ready else []), (validation if ready else []), {
        "status": "ready" if ready else "insufficient_spatial_coverage",
        "sample_count": len(source),
        "fit_count": len(fit),
        "validation_count": len(validation),
        "fit_quadrants": fit_quadrants,
        "validation_quadrants": validation_quadrants,
        "validation_ratio": validation_ratio,
        "fit_indexes": [index for index in range(len(source)) if index not in selected_set],
        "validation_indexes": sorted(selected),
    }


def _star_morphology_key(item: Mapping[str, Any]) -> Tuple[float, ...]:
    return (
        float(item.get("saturation_fraction") or 0.0),
        abs(float(item.get("aspect_ratio") or 1.0) - 1.0),
        -float(item.get("compactness") or 0.0),
        float(item.get("y") or 0.0),
        float(item.get("x") or 0.0),
    )


def _select_star_records_for_analysis(
    records: Sequence[Dict[str, Any]],
    *,
    maximum_objects: int = STAR_ENSEMBLE_OBJECT_CAP,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    source = list(records)
    maximum_objects = max(16, min(int(maximum_objects), STAR_ENSEMBLE_OBJECT_CAP))
    if len(source) <= maximum_objects:
        return source, {
            "valid_object_count": len(source),
            "analysis_object_count": len(source),
            "object_cap": maximum_objects,
            "object_cap_applied": False,
            "discarded_object_count": 0,
            "selection_method": "all_valid_objects",
        }

    coordinates = np.asarray(
        [[float(item["x"]), float(item["y"])] for item in source],
        dtype=np.float64,
    )
    minima = np.min(coordinates, axis=0)
    spans = np.maximum(np.ptp(coordinates, axis=0), 1.0)
    normalized = (coordinates - minima) / spans
    selected: List[int] = []
    selected_mask = np.zeros(len(source), dtype=bool)

    for quadrant in sorted({int(item["quadrant"]) for item in source}):
        members = [
            index
            for index, item in enumerate(source)
            if int(item["quadrant"]) == quadrant
        ]
        if not members:
            continue
        best = min(members, key=lambda index: _star_morphology_key(source[index]))
        selected.append(best)
        selected_mask[best] = True

    nearest_distance_sq = np.full(len(source), float("inf"), dtype=np.float64)
    for index in selected:
        delta = normalized - normalized[index]
        nearest_distance_sq = np.minimum(
            nearest_distance_sq,
            np.sum(delta * delta, axis=1),
        )
    nearest_distance_sq[selected_mask] = -1.0

    while len(selected) < maximum_objects:
        best = int(np.argmax(nearest_distance_sq))
        if selected_mask[best]:
            break
        selected.append(best)
        selected_mask[best] = True
        delta = normalized - normalized[best]
        nearest_distance_sq = np.minimum(
            nearest_distance_sq,
            np.sum(delta * delta, axis=1),
        )
        nearest_distance_sq[selected_mask] = -1.0

    selected.sort()
    analysis = [source[index] for index in selected]
    return analysis, {
        "valid_object_count": len(source),
        "analysis_object_count": len(analysis),
        "object_cap": maximum_objects,
        "object_cap_applied": True,
        "discarded_object_count": len(source) - len(analysis),
        "selection_method": (
            "quadrant_seeded_normalized_farthest_point_without_color_ranking"
        ),
    }


def _select_white_reference_rectangle(
    records: Sequence[Dict[str, Any]],
    *,
    image_shape: Tuple[int, int],
    minimum_objects: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Select one luminance-driven rectangle containing usable bright objects."""
    source = list(records)
    height, width = image_shape
    if len(source) < minimum_objects:
        return [], {
            "status": "insufficient_star_evidence",
            "minimum_objects": int(minimum_objects),
            "valid_object_count": len(source),
        }
    short_side = max(1, min(height, width))
    side_lengths = sorted(
        {
            max(32, min(short_side, int(round(short_side * fraction))))
            for fraction in (0.25, 0.375, 0.50, 0.75, 1.0)
        }
    )
    proposals: List[Dict[str, Any]] = []
    centers = sorted(
        {(int(round(float(item["x"]))), int(round(float(item["y"])))) for item in source}
    )
    for side in side_lengths:
        half = side // 2
        for center_x, center_y in centers:
            x0 = max(0, min(width - side, center_x - half))
            y0 = max(0, min(height - side, center_y - half))
            x1 = min(width, x0 + side)
            y1 = min(height, y0 + side)
            members = [
                item
                for item in source
                if x0 <= float(item["x"]) < x1
                and y0 <= float(item["y"]) < y1
            ]
            if len(members) < minimum_objects:
                continue
            total_flux = float(
                sum(float(np.sum(np.asarray(item["flux"], dtype=np.float64))) for item in members)
            )
            saturation = float(
                np.median(
                    [float(item.get("saturation_fraction") or 0.0) for item in members]
                )
            )
            compactness = float(
                np.median([float(item.get("compactness") or 0.0) for item in members])
            )
            proposals.append(
                {
                    "rectangle": {
                        "x": int(x0),
                        "y": int(y0),
                        "width": int(x1 - x0),
                        "height": int(y1 - y0),
                        "coordinate_system": "numpy_top_left",
                    },
                    "members": members,
                    "object_count": len(members),
                    "total_background_subtracted_flux": total_flux,
                    "median_saturation_fraction": saturation,
                    "median_compactness": compactness,
                    "side_length": int(side),
                }
            )
        if proposals:
            break
    if not proposals:
        return [], {
            "status": "unavailable",
            "reason": "no_single_rectangle_contains_minimum_objects",
            "minimum_objects": int(minimum_objects),
            "valid_object_count": len(source),
            "tested_side_lengths": side_lengths,
        }
    selected = max(
        proposals,
        key=lambda item: (
            int(item["object_count"]),
            float(item["total_background_subtracted_flux"]),
            float(item["median_compactness"]),
            -float(item["median_saturation_fraction"]),
            -int(item["rectangle"]["y"]),
            -int(item["rectangle"]["x"]),
        ),
    )
    rectangle = selected["rectangle"]
    center_x = float(rectangle["x"] + rectangle["width"] / 2.0)
    center_y = float(rectangle["y"] + rectangle["height"] / 2.0)
    normalized_members: List[Dict[str, Any]] = []
    for item in selected["members"]:
        normalized_item = dict(item)
        normalized_item["quadrant"] = int(float(item["x"]) >= center_x) + 2 * int(
            float(item["y"]) >= center_y
        )
        normalized_members.append(normalized_item)
    return normalized_members, {
        "status": "selected",
        "selection_policy": (
            "smallest_multiscale_rectangle_then_object_count_flux_compactness"
        ),
        "color_values_used_for_selection": False,
        "proposal_count": len(proposals),
        "tested_side_lengths": side_lengths,
        **{key: value for key, value in selected.items() if key != "members"},
    }


def _ratio_stats(
    records: Sequence[Dict[str, Any]],
    *,
    image: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    values: List[Tuple[float, float]] = []
    fluxes: List[List[float]] = []
    for item in records:
        if image is None:
            flux = np.asarray(item["flux"], dtype=np.float64)
        else:
            flux = _star_flux(
                image,
                item["_ys"],
                item["_xs"],
                item["_annulus_y"],
                item["_annulus_x"],
            )
            if flux is None:
                continue
        values.append(
            (
                float(math.log(float(flux[0] / flux[1]))),
                float(math.log(float(flux[2] / flux[1]))),
            )
        )
        fluxes.append([float(value) for value in flux])
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"status": "unavailable", "sample_count": 0}
    centers = np.median(array, axis=0)
    mad = np.median(np.abs(array - centers), axis=0)
    errors = np.linalg.norm(array, axis=1)
    return {
        "status": "ready",
        "sample_count": int(array.shape[0]),
        "center_log_ratios": [float(value) for value in centers],
        "ratio_mad": [float(value) for value in mad],
        "maximum_ratio_mad": float(np.max(mad)),
        "median_neutral_error": float(np.median(errors)),
        "fluxes": fluxes,
    }


def _sigma_clip_star_fit(
    records: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    selected = list(records)
    for iteration in range(3):
        if len(selected) < 12:
            break
        array = np.asarray(
            [[item["log_rg"], item["log_bg"]] for item in selected],
            dtype=np.float64,
        )
        center = np.median(array, axis=0)
        mad = np.median(np.abs(array - center), axis=0)
        scale = np.maximum(1.4826 * mad, 0.01)
        keep = np.all(np.abs(array - center) <= 2.5 * scale, axis=1)
        next_selected = [item for item, accepted in zip(selected, keep) if accepted]
        if len(next_selected) == len(selected):
            return selected, {
                "status": "ready",
                "iterations": iteration + 1,
                "input_count": len(records),
                "output_count": len(selected),
            }
        selected = next_selected
    return selected, {
        "status": "ready" if len(selected) >= 12 else "insufficient_after_clipping",
        "iterations": 3,
        "input_count": len(records),
        "output_count": len(selected),
    }


def _report_star_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "x": float(item["x"]),
            "y": float(item["y"]),
            "quadrant": int(item["quadrant"]),
            "area": int(item["area"]),
            "bbox": [int(value) for value in item["bbox"]],
            "aspect_ratio": float(item["aspect_ratio"]),
            "compactness": float(item["compactness"]),
            "saturation_fraction": float(item["saturation_fraction"]),
            "log_rg": float(item["log_rg"]),
            "log_bg": float(item["log_bg"]),
        }
        for item in records
    ]


def _mark_star_ensemble_prerequisite_skipped(
    report: Dict[str, Any],
    *,
    global_white_enabled: bool,
) -> None:
    report["sampling"]["stars"] = {
        "status": "not_run",
        "reason": "background_neutralization_prerequisite_rejected",
        "selection": {
            "status": "not_run",
            "reason": "background_neutralization_prerequisite_rejected",
            "detected_component_count": 0,
            "valid_object_count": 0,
            "quadrants": 0,
            "rejection_counts": {},
        },
    }
    report["candidates"][STAR_ENSEMBLE_METHOD] = {
        "method": STAR_ENSEMBLE_METHOD,
        "reference": PSEUDO_WHITE_REFERENCE,
        "accepted": False,
        "would_accept": False,
        "analysis_skipped": True,
        "shadow_only": not global_white_enabled,
        "pixels_authorized": False,
        "physical_color": False,
        "requires_review": True,
        "gains": [],
        "rejection_reasons": [
            "background_neutralization_prerequisite_rejected"
        ],
    }
    report["selection"].update(
        method="PRESERVE_INPUT",
        applied=False,
        shadow_star_ensemble=False,
    )


def evaluate_auto_local_reference(
    image: Any,
    *,
    config: Any = None,
    channel_kind: str = "broadband_rgb_osc",
    linear: bool = True,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Return the selected non-physical candidate and a JSON-safe report.

    ``None`` means that no candidate was authorized.  The caller must keep or
    restore its immutable pre-color checkpoint in that case.
    """
    report: Dict[str, Any] = {
        "schema": AUTO_REFERENCE_SCHEMA,
        "status": "not_run",
        "eligibility": {
            "linear": bool(linear),
            "channel_kind": str(channel_kind),
            "eligible": False,
            "reason": None,
        },
        "sampling": {},
        "reference_regions": {
            "background": None,
            "white": None,
        },
        "candidates": {},
        "selection": {
            "method": "PRESERVE_INPUT",
            "applied": False,
            "physical_color": False,
            "requires_review": True,
        },
        "transaction": {
            "pixels_written": False,
            "owned_by": "stage4_orchestrator",
        },
        "physical_color": {"accepted": False},
        "degraded_color_correction": {"applied": False},
        "requires_review": True,
    }
    if not linear:
        report["status"] = "not_applicable"
        report["eligibility"]["reason"] = "nonlinear_input"
        return None, _json_safe(report)
    if str(channel_kind) != "broadband_rgb_osc":
        report["status"] = "not_applicable"
        report["eligibility"]["reason"] = "unsupported_channel_semantics"
        return None, _json_safe(report)

    try:
        adapter = _adapt_image(image)
    except (TypeError, ValueError) as error:
        report["status"] = "unavailable"
        report["eligibility"]["reason"] = str(error)
        return None, _json_safe(report)
    before = adapter.chw
    if before.shape[0] < 3:
        report["status"] = "not_applicable"
        report["eligibility"]["reason"] = "mono_input"
        return None, _json_safe(report)
    if not np.all(np.isfinite(before)):
        report["status"] = "rejected"
        report["eligibility"]["reason"] = "non_finite_input"
        return None, _json_safe(report)
    report["eligibility"].update(eligible=True, reason="linear_broadband_rgb_osc")
    global_white_enabled = bool(
        _cfg(config, "stage4_auto_reference_global_white_enabled", True)
    )

    target_count = int(
        _clamp(
            _cfg(config, "stage4_auto_reference_background_sample_target", 40),
            16,
            64,
            40,
        )
    )
    minimum_count = int(
        _clamp(
            _cfg(config, "stage4_auto_reference_background_sample_min", 16),
            16,
            40,
            16,
        )
    )
    minimum_count = min(minimum_count, target_count)
    holdout_ratio = _clamp(
        _cfg(config, "stage4_auto_reference_holdout_ratio", 0.25),
        0.20,
        0.40,
        0.25,
    )
    points, safe_report = build_safe_background_samples(
        before,
        target_count=target_count,
        min_count=minimum_count,
    )
    fit_points, validation_points, split_report = split_background_sample_points(
        points,
        before,
        validation_ratio=holdout_ratio,
        minimum_total=minimum_count,
        minimum_fit=12,
        minimum_validation=4,
    )
    report["sampling"]["background"] = {
        "safe_selection": safe_report,
        "split": split_report,
    }
    if safe_report.get("status") != "ready" or split_report.get("status") != "ready":
        report["status"] = "rejected"
        report["eligibility"]["reason"] = "insufficient_safe_background_coverage"
        report["candidates"][BACKGROUND_METHOD] = {
            "accepted": False,
            "rejection_reasons": ["insufficient_safe_background_coverage"],
        }
        _mark_star_ensemble_prerequisite_skipped(
            report,
            global_white_enabled=global_white_enabled,
        )
        return None, _json_safe(report)

    patch_radius = int(safe_report.get("patch_radius") or 12)
    selected_background_point, background_region = _select_background_rectangle(
        before,
        fit_points,
        patch_radius=patch_radius,
    )
    report["reference_regions"]["background"] = background_region
    selected_background_points = (
        [selected_background_point]
        if selected_background_point is not None
        else []
    )
    fit_before = _measure_rgb_patches(
        before,
        selected_background_points,
        patch_radius=patch_radius,
    )
    validation_before = _measure_rgb_patches(
        before, validation_points, patch_radius=patch_radius
    )
    report["sampling"]["background"].update(
        fit_measurements=fit_before,
        validation_measurements=validation_before,
    )
    if fit_before.get("sample_count", 0) < 1 or validation_before.get("sample_count", 0) < 4:
        report["status"] = "rejected"
        report["eligibility"]["reason"] = "background_patch_measurement_failed"
        report["candidates"][BACKGROUND_METHOD] = {
            "accepted": False,
            "rejection_reasons": ["background_patch_measurement_failed"],
        }
        _mark_star_ensemble_prerequisite_skipped(
            report,
            global_white_enabled=global_white_enabled,
        )
        return None, _json_safe(report)

    lum = _luminance(before[:3])
    finite_lum = lum[np.isfinite(lum)]
    background_luminance = float(fit_before["median_luminance"])
    background_sigma = float(fit_before["luminance_sigma"])
    subject_separation = float(np.quantile(finite_lum, 0.98)) - background_luminance
    separation_min = max(0.005, 3.0 * background_sigma)
    star_threshold = max(
        float(np.quantile(finite_lum, 0.98)),
        background_luminance + max(5.0 * background_sigma, 0.002),
    )
    subject_mask = _build_subject_mask(
        before,
        background_luminance=background_luminance,
        background_sigma=background_sigma,
        star_threshold=star_threshold,
    )
    report["sampling"]["background"]["subject_separation"] = {
        "measured": subject_separation,
        "minimum": separation_min,
        "accepted": subject_separation >= separation_min,
    }

    candidate_a, offsets_a = _subtract_to_lowest(before, fit_before)
    validation_a = _measure_rgb_patches(
        candidate_a, validation_points, patch_radius=patch_radius
    )
    holdout_tail = _background_holdout_comparison(
        validation_before,
        validation_a,
    )
    before_error = float(validation_before.get("median_color_error") or 0.0)
    after_error = float(validation_a.get("median_color_error") or 0.0)
    absolute_improvement = before_error - after_error
    relative_improvement = absolute_improvement / max(before_error, 1e-12)
    minimum_error = _clamp(
        _cfg(config, "stage4_auto_reference_background_error_min", 0.01),
        0.0,
        0.25,
        0.01,
    )
    improvement_min = _clamp(
        _cfg(config, "stage4_auto_reference_background_improvement_min", 0.10),
        0.01,
        0.90,
        0.10,
    )
    reasons_a: List[str] = []
    if subject_separation < separation_min:
        reasons_a.append("insufficient_background_subject_separation")
    if before_error < minimum_error:
        reasons_a.append("background_color_error_below_action_threshold")
    if absolute_improvement < 0.002:
        reasons_a.append("heldout_background_absolute_improvement_insufficient")
    if relative_improvement < improvement_min:
        reasons_a.append("heldout_background_relative_improvement_insufficient")
    reasons_a.extend(_background_holdout_tail_reasons(holdout_tail))
    safety_reasons_a, safety_a = _safety_gate(
        before,
        candidate_a,
        validation_points=validation_points,
        patch_radius=patch_radius,
        subject_mask=subject_mask,
        config=config,
    )
    reasons_a.extend(safety_reasons_a)
    accepted_a = not reasons_a
    report_a = {
        "method": BACKGROUND_METHOD,
        "accepted": accepted_a,
        "physical_color": False,
        "requires_review": True,
        "application": "global_additive_subtract_to_lowest_channel",
        "channel_offsets": offsets_a,
        "never_lifts_channels": True,
        "fit": fit_before,
        "holdout_before": validation_before,
        "holdout_after": validation_a,
        "metrics": {
            "baseline_color_error": before_error,
            "candidate_color_error": after_error,
            "absolute_improvement": absolute_improvement,
            "relative_improvement": relative_improvement,
            "minimum_baseline_error": minimum_error,
            "minimum_absolute_improvement": 0.002,
            "minimum_relative_improvement": improvement_min,
            "holdout_tail": holdout_tail,
        },
        "safety_gate": safety_a,
        "rejection_reasons": sorted(set(reasons_a)),
    }
    report["candidates"][BACKGROUND_METHOD] = report_a

    if not accepted_a:
        _mark_star_ensemble_prerequisite_skipped(
            report,
            global_white_enabled=global_white_enabled,
        )
        report["status"] = "rejected"
        return None, _json_safe(report)

    saturation_max = _clamp(
        _cfg(config, "stage4_auto_reference_star_saturation_ratio_max", 0.10),
        0.0,
        0.50,
        0.10,
    )
    records, star_detection_report, _detected_threshold = _find_star_ensemble(
        before,
        background_luminance=background_luminance,
        background_sigma=background_sigma,
        background_texture=float(fit_before.get("texture") or 0.0),
        subject_mask=subject_mask,
        saturation_ratio_max=saturation_max,
    )
    minimum_stars = int(
        _clamp(
            _cfg(config, "stage4_auto_reference_star_min_objects", 16),
            16,
            256,
            16,
        )
    )
    bounded_records, star_limit_report = _select_star_records_for_analysis(
        records,
    )
    rectangle_records, white_region = _select_white_reference_rectangle(
        bounded_records,
        image_shape=tuple(int(value) for value in before.shape[1:]),
        minimum_objects=minimum_stars,
    )
    report["reference_regions"]["white"] = white_region
    analysis_records = rectangle_records
    star_detection_report.update(star_limit_report)
    fit_stars, validation_stars, star_split = _split_star_records(
        analysis_records, validation_ratio=holdout_ratio
    )
    clipped_fit, sigma_clip = _sigma_clip_star_fit(fit_stars)
    fit_ratio_stats = _ratio_stats(clipped_fit)
    validation_ratio_before = _ratio_stats(validation_stars)
    report["sampling"]["stars"] = {
        "coordinate_system": "numpy_top_left",
        "selection": star_detection_report,
        "minimum_objects": minimum_stars,
        "minimum_quadrants": 3,
        "objects": _report_star_records(analysis_records),
        "split": star_split,
        "sigma_clip": sigma_clip,
        "fit_ratio_statistics": fit_ratio_stats,
        "validation_ratio_statistics": validation_ratio_before,
    }

    reasons_b: List[str] = []
    candidate_b: Optional[np.ndarray] = None
    gains: List[float] = []
    validation_b: Dict[str, Any] = {"status": "not_run"}
    validation_ratio_after: Dict[str, Any] = {"status": "not_run"}
    safety_b: Dict[str, Any] = {"status": "not_run"}
    offsets_b: List[float] = []
    if len(analysis_records) < minimum_stars:
        reasons_b.append("insufficient_independent_star_objects")
    if len({int(item["quadrant"]) for item in analysis_records}) < 3:
        reasons_b.append("insufficient_star_quadrant_coverage")
    if star_split.get("status") != "ready":
        reasons_b.append("star_fit_holdout_split_unavailable")
    if len(clipped_fit) < 12:
        reasons_b.append("insufficient_star_fit_objects_after_clipping")
    dispersion_max = _clamp(
        _cfg(config, "stage4_auto_reference_star_ratio_mad_max", 0.12),
        0.01,
        0.50,
        0.12,
    )
    fit_dispersion = float(fit_ratio_stats.get("maximum_ratio_mad") or float("inf"))
    if fit_dispersion > dispersion_max:
        reasons_b.append("star_color_ratio_dispersion_exceeded")

    if not reasons_b:
        fluxes = np.asarray(fit_ratio_stats.get("fluxes") or (), dtype=np.float64)
        medians = np.median(fluxes, axis=0)
        if medians.size != 3 or float(np.min(medians)) <= 0.0:
            reasons_b.append("invalid_star_ensemble_flux_medians")
        else:
            target = float(np.median(medians))
            gain_limit = _clamp(
                _cfg(config, "stage4_auto_reference_gain_limit", 1.10),
                1.01,
                1.20,
                1.10,
            )
            gain_values = np.clip(target / medians, 1.0 / gain_limit, gain_limit)
            gains = [float(value) for value in gain_values]
            candidate_b = candidate_a.copy()
            candidate_b[:3] = np.maximum(
                candidate_b[:3] * gain_values[:, None, None], 0.0
            )
            fit_after_gain = _measure_rgb_patches(
                candidate_b,
                selected_background_points,
                patch_radius=patch_radius,
            )
            candidate_b, offsets_b = _subtract_to_lowest(candidate_b, fit_after_gain)
            validation_b = _measure_rgb_patches(
                candidate_b, validation_points, patch_radius=patch_radius
            )
            validation_ratio_after = _ratio_stats(validation_stars, image=candidate_b)
            before_star_error = float(
                validation_ratio_before.get("median_neutral_error") or float("inf")
            )
            after_star_error = float(
                validation_ratio_after.get("median_neutral_error") or float("inf")
            )
            star_improvement = (before_star_error - after_star_error) / max(
                before_star_error, 1e-12
            )
            star_improvement_min = _clamp(
                _cfg(config, "stage4_auto_reference_star_improvement_min", 0.10),
                0.01,
                0.90,
                0.10,
            )
            if star_improvement < star_improvement_min:
                reasons_b.append("heldout_star_neutral_error_improvement_insufficient")
            before_dispersion = float(
                validation_ratio_before.get("maximum_ratio_mad") or float("inf")
            )
            after_dispersion = float(
                validation_ratio_after.get("maximum_ratio_mad") or float("inf")
            )
            if after_dispersion > before_dispersion * 1.05 + 1e-8:
                reasons_b.append("heldout_star_ratio_dispersion_growth_exceeded")
            candidate_b_bg_error = float(validation_b.get("median_color_error") or float("inf"))
            if candidate_b_bg_error > after_error * 1.05 + 0.0005:
                reasons_b.append("background_regressed_after_pseudo_white_reference")
            safety_reasons_b, safety_b = _safety_gate(
                before,
                candidate_b,
                validation_points=validation_points,
                patch_radius=patch_radius,
                subject_mask=subject_mask,
                config=config,
            )
            reasons_b.extend(safety_reasons_b)

    accepted_b = candidate_b is not None and not reasons_b
    report["candidates"][WHITE_REGION_METHOD] = {
        "method": WHITE_REGION_METHOD,
        "reference": PSEUDO_WHITE_REFERENCE,
        "accepted": bool(accepted_b),
        "would_accept": bool(accepted_b),
        "shadow_only": not global_white_enabled,
        "pixels_authorized": bool(accepted_b and global_white_enabled),
        "physical_color": False,
        "requires_review": True,
        "gains": gains,
        "gain_limit": _clamp(
            _cfg(config, "stage4_auto_reference_gain_limit", 1.10),
            1.01,
            1.20,
            1.10,
        ),
        "second_background_offsets": offsets_b,
        "fit_ratio_statistics": fit_ratio_stats,
        "holdout_ratio_before": validation_ratio_before,
        "holdout_ratio_after": validation_ratio_after,
        "holdout_background_after": validation_b,
        "safety_gate": safety_b,
        "rejection_reasons": sorted(set(reasons_b)),
    }

    selected: Optional[np.ndarray] = None
    selected_method = "PRESERVE_INPUT"
    if accepted_b and global_white_enabled and candidate_b is not None:
        selected = adapter.restore(candidate_b)
        selected_method = WHITE_REGION_METHOD
    elif accepted_a:
        selected = adapter.restore(candidate_a)
        selected_method = BACKGROUND_METHOD

    if selected is not None and tuple(selected.shape) != adapter.source_shape:
        selected = None
        selected_method = "PRESERVE_INPUT"
        report["candidates"].setdefault(selected_method, {}).setdefault(
            "rejection_reasons", []
        ).append("restored_layout_shape_changed")
    report["status"] = "accepted" if selected is not None else "rejected"
    report["selection"].update(
        method=selected_method,
        applied=selected is not None,
        shadow_star_ensemble=not global_white_enabled,
    )
    report["degraded_color_correction"].update(
        applied=selected is not None,
        method=selected_method if selected is not None else None,
        physical_color=False,
        requires_review=True,
    )
    return selected, _json_safe(report)


__all__ = [
    "AUTO_REFERENCE_SCHEMA",
    "BACKGROUND_METHOD",
    "PSEUDO_WHITE_REFERENCE",
    "STAR_ENSEMBLE_METHOD",
    "WHITE_REGION_METHOD",
    "evaluate_auto_local_reference",
]
