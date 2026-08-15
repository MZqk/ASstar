"""Pixel-space remix formula and deterministic Stage 9 quality gate."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import math

import numpy as np

from image_metrics import _box_blur_gray
import stage7_quality

try:
    from scipy import ndimage as scipy_ndimage
    from scipy import spatial as scipy_spatial
except ImportError:  # pragma: no cover - bundled runtime includes scipy
    scipy_ndimage = None
    scipy_spatial = None


_MOTTLING_LOW_ABSOLUTE_SCORE_MAX = 0.10
_MOTTLING_LOW_ABSOLUTE_DELTA_MAX = 0.03
_MOTTLING_LOW_ABSOLUTE_CHANGED_RATIO_MAX = 0.12
_STAR_REFERENCE_MIN_COMPONENT_AREA = 3
_STAR_RECOVERY_DELTA = 0.002
_HOLLOW_STRUCTURE_MIN_AREA = 4
_STAGE9_FWHM_ANCHOR_PX = 4.0
_STAGE9_SPATIAL_SCALE_MIN_SAMPLES = 4
_STAGE9_SPATIAL_SCALE_SCHEMA = "starun.stage9-fwhm-spatial-scale.v1"
_STAGE9_STARMASK_OUTPUT_PROFILE_SCHEMA = (
    "starun.stage9-starmask-output-profile.v1"
)
_STAGE9_STARMASK_OUTPUT_TOLERANCE = 1e-4
_STAGE9_STARMASK_OUTPUT_NAMES = ("faint", "mid", "bright", "peak")


def _bounded(value: Any, default: float, lower: float, upper: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(lower, min(upper, parsed))


def _image_scale(image: np.ndarray) -> float:
    arr = np.asarray(image)
    if np.issubdtype(arr.dtype, np.integer):
        return float(np.iinfo(arr.dtype).max)
    finite = np.asarray(arr, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    peak = float(np.max(np.abs(finite)))
    if peak <= 1.5:
        return 1.0
    if peak <= 255.0 * 1.05:
        return 255.0
    if peak <= 65535.0 * 1.05:
        return 65535.0
    return max(peak, 1.0)


def _normalized(image: np.ndarray, *, scale: float | None = None) -> np.ndarray:
    arr = np.nan_to_num(
        np.asarray(image).astype(np.float32, copy=False),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    divisor = float(scale if scale is not None else _image_scale(np.asarray(image)))
    return np.clip(arr / max(divisor, 1e-12), 0.0, 1.0)


def _stage9_valid_fwhm_samples(values: Any) -> np.ndarray:
    samples = np.asarray(values if values is not None else (), dtype=np.float64)
    samples = samples.reshape(-1)
    return samples[np.isfinite(samples) & (samples > 0.0)]


def _stage9_spatial_scale_summary(
    samples: Any,
    *,
    source: str,
    review_required: bool,
    source_attempts: list[Dict[str, Any]],
) -> Dict[str, Any]:
    valid = _stage9_valid_fwhm_samples(samples)
    if valid.size < _STAGE9_SPATIAL_SCALE_MIN_SAMPLES:
        raise ValueError(
            f"{source} FWHM sample count {int(valid.size)} is below "
            f"{_STAGE9_SPATIAL_SCALE_MIN_SAMPLES}"
        )
    median = float(np.median(valid))
    radius_scale = median / _STAGE9_FWHM_ANCHOR_PX
    return {
        "schema": _STAGE9_SPATIAL_SCALE_SCHEMA,
        "status": "ready",
        "reason_code": "stage9_spatial_scale_ready",
        "source": source,
        "sample_count": int(valid.size),
        "fwhm_median_px": median,
        "fwhm_p25_px": float(np.percentile(valid, 25.0)),
        "fwhm_p75_px": float(np.percentile(valid, 75.0)),
        "anchor_fwhm_px": _STAGE9_FWHM_ANCHOR_PX,
        "radius_scale": float(radius_scale),
        "area_scale": float(radius_scale * radius_scale),
        "radius_formula": "nominal_px * (FWHM_px / 4.0_px)",
        "area_formula": "nominal_px2 * (FWHM_px / 4.0_px)^2",
        "matched_display_source": source == "matched_display_fwhm",
        "stage9_psf_review_required": bool(review_required),
        "source_attempts": source_attempts,
    }


def _stage9_starmask_halfmax_samples(
    raw_starmask: np.ndarray,
    catalog: Dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap half-max FWHM from the immutable raw starmask.

    The fixed search patch used here only discovers the initial scale. Formal
    Stage 9 support and quality geometry is rebuilt from the frozen result.
    """
    peak = _pixel_peak(_normalized(np.asarray(raw_starmask)))
    if peak.ndim != 2:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    peak_y = np.asarray((catalog or {}).get("_peak_y", ()), dtype=np.int32)
    peak_x = np.asarray((catalog or {}).get("_peak_x", ()), dtype=np.int32)
    if peak_y.size == 0 or peak_y.size != peak_x.size:
        finite = peak[np.isfinite(peak)]
        if finite.size < 64 or scipy_ndimage is None:
            return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
        background = float(np.percentile(finite, 50.0))
        low = finite[finite <= np.percentile(finite, 70.0)]
        if low.size < 32:
            low = finite
        mad = float(np.median(np.abs(low - np.median(low))))
        threshold = max(background + 5.0 * max(1.4826 * mad, 1e-7), 1e-6)
        labels, count = scipy_ndimage.label(
            peak > threshold,
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        if count <= 0:
            return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
        positions = scipy_ndimage.maximum_position(
            peak,
            labels=labels,
            index=np.arange(1, count + 1, dtype=np.int32),
        )
        peak_y = np.asarray([item[0] for item in positions], dtype=np.int32)
        peak_x = np.asarray([item[1] for item in positions], dtype=np.int32)
    per_star = np.full(peak_y.size, np.nan, dtype=np.float64)
    for index, (y, x) in enumerate(zip(peak_y, peak_x)):
        measured = _measure_connected_halfmax_fwhm(
            peak,
            int(y),
            int(x),
            search_radius=2,
            patch_radius=6,
        )
        if measured.get("status") == "ok":
            per_star[index] = float(measured["fwhm_px"])
    return _stage9_valid_fwhm_samples(per_star), per_star


def resolve_stage9_spatial_scale(
    star_reference: Dict[str, Any] | None,
    *,
    stage5_stars: Any = None,
    raw_starmask: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Resolve and freeze Stage 9 pixel geometry without mutating images."""
    catalog = star_reference if isinstance(star_reference, dict) else {}
    attempts: list[Dict[str, Any]] = []

    display = np.asarray(catalog.get("_display_source_fwhm_px", ()), dtype=np.float64)
    valid = np.asarray(catalog.get("_psf_valid_flags", ()), dtype=bool)
    isolated = np.asarray(catalog.get("_psf_isolated_flags", ()), dtype=bool)
    saturated = np.asarray(catalog.get("_psf_saturated_flags", ()), dtype=bool)
    display_selected = np.asarray([], dtype=np.float64)
    display_per_star = np.full(display.size, np.nan, dtype=np.float64)
    if (
        display.size > 0
        and valid.size == display.size
        and isolated.size == display.size
        and saturated.size == display.size
    ):
        keep = valid & isolated & ~saturated & np.isfinite(display) & (display > 0.0)
        display_per_star[keep] = display[keep]
        display_selected = display[keep]
    attempts.append(
        {
            "source": "matched_display_fwhm",
            "sample_count": int(display_selected.size),
            "accepted": bool(
                display_selected.size >= _STAGE9_SPATIAL_SCALE_MIN_SAMPLES
            ),
        }
    )
    if display_selected.size >= _STAGE9_SPATIAL_SCALE_MIN_SAMPLES:
        report = _stage9_spatial_scale_summary(
            display_selected,
            source="matched_display_fwhm",
            review_required=False,
            source_attempts=attempts,
        )
        catalog["_stage9_spatial_fwhm_px"] = display_per_star.astype(np.float32)
        catalog["stage9_spatial_scale"] = report
        return report

    stage5_values = []
    for star in list(stage5_stars or []):
        if not isinstance(star, dict):
            continue
        if not bool(star.get("geometry_valid", True)) or bool(
            star.get("saturated", False)
        ):
            continue
        try:
            value = float(star.get("fwhm_geometry"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0.0:
            stage5_values.append(value)
    stage5_samples = _stage9_valid_fwhm_samples(stage5_values)
    attempts.append(
        {
            "source": "stage5_fwhm_geometry",
            "sample_count": int(stage5_samples.size),
            "accepted": bool(stage5_samples.size >= _STAGE9_SPATIAL_SCALE_MIN_SAMPLES),
        }
    )
    if stage5_samples.size >= _STAGE9_SPATIAL_SCALE_MIN_SAMPLES:
        report = _stage9_spatial_scale_summary(
            stage5_samples,
            source="stage5_fwhm_geometry",
            review_required=True,
            source_attempts=attempts,
        )
        catalog["_stage9_spatial_fwhm_px"] = np.full(
            np.asarray(catalog.get("_peak_y", ())).size,
            report["fwhm_median_px"],
            dtype=np.float32,
        )
        catalog["stage9_spatial_scale"] = report
        return report

    mask_samples = np.asarray([], dtype=np.float64)
    mask_per_star = np.asarray([], dtype=np.float64)
    if raw_starmask is not None:
        mask_samples, mask_per_star = _stage9_starmask_halfmax_samples(
            raw_starmask,
            catalog,
        )
    attempts.append(
        {
            "source": "raw_starmask_halfmax",
            "sample_count": int(mask_samples.size),
            "accepted": bool(mask_samples.size >= _STAGE9_SPATIAL_SCALE_MIN_SAMPLES),
        }
    )
    if mask_samples.size >= _STAGE9_SPATIAL_SCALE_MIN_SAMPLES:
        report = _stage9_spatial_scale_summary(
            mask_samples,
            source="raw_starmask_halfmax",
            review_required=True,
            source_attempts=attempts,
        )
        catalog["_stage9_spatial_fwhm_px"] = mask_per_star.astype(np.float32)
        catalog["stage9_spatial_scale"] = report
        return report

    report = {
        "schema": _STAGE9_SPATIAL_SCALE_SCHEMA,
        "status": "unavailable",
        "reason_code": "stage9_spatial_scale_unavailable",
        "reason": "fewer than four valid FWHM samples from all scale sources",
        "sample_count_min": _STAGE9_SPATIAL_SCALE_MIN_SAMPLES,
        "anchor_fwhm_px": _STAGE9_FWHM_ANCHOR_PX,
        "radius_formula": "nominal_px * (FWHM_px / 4.0_px)",
        "area_formula": "nominal_px2 * (FWHM_px / 4.0_px)^2",
        "stage9_psf_review_required": True,
        "source_attempts": attempts,
    }
    catalog["stage9_spatial_scale"] = report
    return report


def _stage9_catalog_scale(star_reference: Dict[str, Any] | None) -> Dict[str, Any]:
    report = (
        (star_reference or {}).get("stage9_spatial_scale")
        if isinstance(star_reference, dict)
        else None
    )
    return report if isinstance(report, dict) else {}


def _stage9_fwhm_scale(
    star_reference: Dict[str, Any] | None,
    fwhm_px: Any = None,
) -> float:
    try:
        value = float(fwhm_px)
    except (TypeError, ValueError):
        value = float("nan")
    if not math.isfinite(value) or value <= 0.0:
        scale = _stage9_catalog_scale(star_reference)
        value = float(scale.get("fwhm_median_px", _STAGE9_FWHM_ANCHOR_PX))
    return max(value / _STAGE9_FWHM_ANCHOR_PX, 1e-6)


def stage9_scale_radius(
    nominal_pixels: Any,
    star_reference: Dict[str, Any] | None,
    *,
    fwhm_px: Any = None,
    rounding: str = "ceil",
    minimum: int = 0,
) -> int:
    scaled = max(0.0, float(nominal_pixels)) * _stage9_fwhm_scale(
        star_reference,
        fwhm_px,
    )
    value = math.ceil(scaled) if rounding == "ceil" else math.floor(scaled + 0.5)
    return max(int(minimum), int(value))


def stage9_scale_distance(
    nominal_pixels: Any,
    star_reference: Dict[str, Any] | None,
    *,
    fwhm_px: Any = None,
) -> float:
    return max(0.0, float(nominal_pixels)) * _stage9_fwhm_scale(
        star_reference,
        fwhm_px,
    )


def stage9_scale_area(
    nominal_pixels: Any,
    star_reference: Dict[str, Any] | None,
    *,
    minimum: int = 1,
) -> int:
    scale = _stage9_catalog_scale(star_reference)
    area_scale = float(scale.get("area_scale", 1.0) or 1.0)
    return max(int(minimum), int(math.floor(float(nominal_pixels) * area_scale + 0.5)))


def stage9_scale_odd_window(
    nominal_size: int,
    star_reference: Dict[str, Any] | None,
    *,
    fwhm_px: Any = None,
) -> int:
    nominal_radius = max(0, (int(nominal_size) - 1) // 2)
    radius = stage9_scale_radius(
        nominal_radius,
        star_reference,
        fwhm_px=fwhm_px,
        rounding="nearest",
        minimum=0,
    )
    return 2 * radius + 1


def stage9_effective_pixel_stats(values: Any) -> Dict[str, Any]:
    samples = _stage9_valid_fwhm_samples(values)
    if samples.size == 0:
        return {"min": None, "median": None, "p95": None, "max": None}
    return {
        "min": float(np.min(samples)),
        "median": float(np.median(samples)),
        "p95": float(np.percentile(samples, 95.0)),
        "max": float(np.max(samples)),
    }


def _stage9_square_window_values(
    image: np.ndarray,
    peak_y: np.ndarray,
    peak_x: np.ndarray,
    window_sizes: np.ndarray,
    *,
    statistic: str,
) -> np.ndarray:
    sample = np.asarray(image, dtype=np.float32)
    values = np.zeros(peak_y.size, dtype=np.float32)
    height, width = sample.shape
    for index, (y, x, size) in enumerate(zip(peak_y, peak_x, window_sizes)):
        radius = max(0, (int(size) - 1) // 2)
        y0 = max(0, int(y) - radius)
        y1 = min(height, int(y) + radius + 1)
        x0 = max(0, int(x) - radius)
        x1 = min(width, int(x) + radius + 1)
        window = sample[y0:y1, x0:x1]
        if window.size == 0:
            continue
        if statistic == "sum":
            values[index] = float(np.sum(window))
        elif statistic == "max":
            values[index] = float(np.max(window))
        elif statistic == "min":
            values[index] = float(np.min(window))
        elif statistic == "median":
            values[index] = float(np.median(window))
        else:
            raise ValueError(f"unsupported window statistic: {statistic}")
    return values


def freeze_stage9_spatial_geometry(
    star_reference: Dict[str, Any],
    raw_starmask: np.ndarray,
) -> Dict[str, Any]:
    """Freeze per-star scaled windows after the one-time scale resolution."""
    scale = _stage9_catalog_scale(star_reference)
    if scale.get("status") != "ready":
        raise ValueError("stage9_spatial_scale_unavailable")
    peak_y = np.asarray(star_reference.get("_peak_y", ()), dtype=np.int32)
    peak_x = np.asarray(star_reference.get("_peak_x", ()), dtype=np.int32)
    if peak_y.size == 0 or peak_y.size != peak_x.size:
        raise ValueError("star reference coordinates unavailable")
    per_star = np.asarray(
        star_reference.get("_stage9_spatial_fwhm_px", ()),
        dtype=np.float32,
    )
    if per_star.size != peak_y.size:
        per_star = np.full(
            peak_y.size,
            float(scale["fwhm_median_px"]),
            dtype=np.float32,
        )
    invalid = ~np.isfinite(per_star) | (per_star <= 0.0)
    per_star[invalid] = float(scale["fwhm_median_px"])
    inner_windows = np.asarray(
        [
            stage9_scale_odd_window(3, star_reference, fwhm_px=value)
            for value in per_star
        ],
        dtype=np.int32,
    )
    outer_windows = np.asarray(
        [
            stage9_scale_odd_window(7, star_reference, fwhm_px=value)
            for value in per_star
        ],
        dtype=np.int32,
    )
    reference_peak = _pixel_peak(_normalized(np.asarray(raw_starmask)))
    inner_sum = _stage9_square_window_values(
        reference_peak,
        peak_y,
        peak_x,
        inner_windows,
        statistic="sum",
    )
    outer_sum = _stage9_square_window_values(
        reference_peak,
        peak_y,
        peak_x,
        outer_windows,
        statistic="sum",
    )
    wing_ratio = np.divide(
        np.maximum(outer_sum - inner_sum, 0.0),
        np.maximum(outer_sum, 1e-12),
        out=np.zeros_like(outer_sum),
        where=outer_sum > 0.0,
    )
    star_reference["_stage9_spatial_fwhm_px"] = per_star
    star_reference["_stage9_inner_window_size_px"] = inner_windows
    star_reference["_stage9_outer_window_size_px"] = outer_windows
    star_reference["_reference_wing_ratio"] = wing_ratio.astype(np.float32)
    report = {
        "status": "frozen",
        "star_count": int(peak_y.size),
        "per_star_fwhm_px": stage9_effective_pixel_stats(per_star),
        "inner_window_size_px": stage9_effective_pixel_stats(inner_windows),
        "outer_window_size_px": stage9_effective_pixel_stats(outer_windows),
        "component_min_area_effective_px": stage9_scale_area(
            _STAR_REFERENCE_MIN_COMPONENT_AREA,
            star_reference,
        ),
        "component_max_area_effective_px": stage9_scale_area(
            512,
            star_reference,
        ),
        "component_span_max_effective_px": stage9_scale_radius(
            64,
            star_reference,
            rounding="nearest",
            minimum=1,
        ),
    }
    star_reference["stage9_spatial_geometry"] = report
    return report


def _gray(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[0] <= 4:
        return np.mean(arr[:3], axis=0)
    if arr.ndim == 3 and arr.shape[-1] <= 4:
        return np.mean(arr[..., :3], axis=-1)
    return np.mean(arr, axis=0)


def _pixel_peak(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[0] <= 4:
        return np.max(arr[:3], axis=0)
    if arr.ndim == 3 and arr.shape[-1] <= 4:
        return np.max(arr[..., :3], axis=-1)
    return np.max(arr, axis=0)


def normalized_star_layer_peak(image: np.ndarray) -> np.ndarray:
    """Return a normalized 2-D peak map for immutable star-layer evidence."""
    return _pixel_peak(_normalized(np.asarray(image))).astype(
        np.float32,
        copy=False,
    )


def _pixel_floor(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[0] <= 4:
        return np.min(arr[:3], axis=0)
    if arr.ndim == 3 and arr.shape[-1] <= 4:
        return np.min(arr[..., :3], axis=-1)
    return np.min(arr, axis=0)


def _luminance(image: np.ndarray) -> np.ndarray:
    """Return the same weighted luminance used by the final background report."""
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[0] <= 4:
        return (
            0.2126 * arr[0]
            + 0.7152 * arr[min(1, arr.shape[0] - 1)]
            + 0.0722 * arr[min(2, arr.shape[0] - 1)]
        ).astype(np.float32)
    if arr.ndim == 3 and arr.shape[-1] <= 4:
        return (
            0.2126 * arr[..., 0]
            + 0.7152 * arr[..., min(1, arr.shape[-1] - 1)]
            + 0.0722 * arr[..., min(2, arr.shape[-1] - 1)]
        ).astype(np.float32)
    return _gray(arr)


def _background_mottling_score(
    gray: np.ndarray,
    *,
    exclusion_mask: np.ndarray | None = None,
) -> float:
    """Measure low-frequency background mottling using the final-report metric."""
    values = np.asarray(gray, dtype=np.float32)
    background_mask = values <= float(np.quantile(values, 0.35))
    if exclusion_mask is not None:
        excluded = np.asarray(exclusion_mask, dtype=bool)
        if excluded.shape == background_mask.shape:
            background_mask &= ~excluded
    if int(np.count_nonzero(background_mask)) < 32:
        background_mask = np.ones_like(values, dtype=bool)
        if exclusion_mask is not None and excluded.shape == background_mask.shape:
            background_mask &= ~excluded

    def weighted_mean(sample: np.ndarray) -> float:
        return float(np.mean(np.asarray(sample)[background_mask]))

    def weighted_std(sample: np.ndarray) -> float:
        return float(np.std(np.asarray(sample)[background_mask]))

    blur1 = _box_blur_gray(values)
    blur3 = values.copy()
    for _ in range(3):
        blur3 = _box_blur_gray(blur3)
    mottling = np.abs(blur1 - blur3)
    score = weighted_mean(mottling) / max(weighted_std(values) * 2.0, 0.006)
    return max(0.0, min(2.0, float(score)))


def _rgb_channels(image: np.ndarray) -> np.ndarray:
    """Return RGB data in CHW layout for local color diagnostics."""
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        return np.repeat(arr[np.newaxis, ...], 3, axis=0)
    if arr.ndim == 3 and arr.shape[0] <= 4:
        if arr.shape[0] == 1:
            return np.repeat(arr, 3, axis=0)
        return arr[:3]
    if arr.ndim == 3 and arr.shape[-1] <= 4:
        rgb = np.moveaxis(arr[..., :3], -1, 0)
        if rgb.shape[0] == 1:
            return np.repeat(rgb, 3, axis=0)
        return rgb
    raise ValueError(f"unsupported RGB image layout: shape={arr.shape}")


def _component_areas(mask: np.ndarray) -> tuple[np.ndarray, int, np.ndarray]:
    labels, count = scipy_ndimage.label(
        np.asarray(mask, dtype=bool),
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    areas = (
        np.bincount(labels.reshape(-1), minlength=count + 1)[1:]
        if count > 0
        else np.asarray([], dtype=np.int64)
    )
    return labels, int(count), np.asarray(areas, dtype=np.int64)


def _stage9_local_quality_metrics(
    base_norm: np.ndarray,
    candidate_norm: np.ndarray,
    positive_change: np.ndarray,
    cfg: Any,
    *,
    confirmed_star_support: np.ndarray | None = None,
    star_reference: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Measure local Stage 9 artifacts that whole-frame ratios can dilute."""
    component_peak_min = _bounded(
        getattr(cfg, "stage9_local_component_peak_min", 0.01),
        0.01,
        0.002,
        0.10,
    )
    component_area_max_nominal = int(
        round(
            _bounded(
                getattr(cfg, "stage9_local_component_area_max", 256),
                256.0,
                16.0,
                4096.0,
            )
        )
    )
    component_aspect_max = _bounded(
        getattr(cfg, "stage9_local_component_aspect_ratio_max", 3.0),
        3.0,
        1.2,
        10.0,
    )
    component_fill_min = _bounded(
        getattr(cfg, "stage9_local_component_fill_ratio_min", 0.15),
        0.15,
        0.02,
        0.80,
    )
    single_pixel_ratio_max = _bounded(
        getattr(cfg, "stage9_local_single_pixel_ratio_max", 0.20),
        0.20,
        0.0,
        0.90,
    )
    cyan_peak_min = _bounded(
        getattr(cfg, "stage9_local_cyan_blue_peak_min", 0.01),
        0.01,
        0.002,
        0.10,
    )
    cyan_saturation_min = _bounded(
        getattr(cfg, "stage9_local_cyan_blue_saturation_min", 0.50),
        0.50,
        0.20,
        0.95,
    )
    cyan_area_max_nominal = int(
        round(
            _bounded(
                getattr(
                    cfg,
                    "stage9_local_cyan_blue_component_area_max",
                    64,
                ),
                64.0,
                4.0,
                2048.0,
            )
        )
    )
    core_percentile = _bounded(
        getattr(cfg, "stage9_core_percentile", 90.0),
        90.0,
        70.0,
        99.0,
    )
    core_color_jump_min = _bounded(
        getattr(cfg, "stage9_core_color_jump_min", 0.10),
        0.10,
        0.03,
        0.50,
    )
    core_jump_area_max_nominal = int(
        round(
            _bounded(
                getattr(
                    cfg,
                    "stage9_core_color_jump_component_area_max",
                    64,
                ),
                64.0,
                4.0,
                2048.0,
            )
        )
    )
    component_area_max = stage9_scale_area(
        component_area_max_nominal,
        star_reference,
    )
    cyan_area_max = stage9_scale_area(cyan_area_max_nominal, star_reference)
    core_jump_area_max = stage9_scale_area(
        core_jump_area_max_nominal,
        star_reference,
    )
    nonstellar_min_area = stage9_scale_area(4, star_reference)
    fragment_area_max = stage9_scale_area(1, star_reference)
    limits = {
        "local_connected_component_max_area": float(component_area_max),
        "local_nonstellar_shape_component_count": 0.0,
        "local_single_pixel_component_ratio": single_pixel_ratio_max,
        "local_cyan_blue_component_max_area": float(cyan_area_max),
        "core_color_jump_component_max_area": float(core_jump_area_max),
        "nominal_local_connected_component_max_area": float(
            component_area_max_nominal
        ),
        "nominal_local_cyan_blue_component_max_area": float(
            cyan_area_max_nominal
        ),
        "nominal_core_color_jump_component_max_area": float(
            core_jump_area_max_nominal
        ),
        "fwhm_anchor_px": _STAGE9_FWHM_ANCHOR_PX,
        "area_scale": float(
            _stage9_catalog_scale(star_reference).get("area_scale", 1.0) or 1.0
        ),
    }
    if scipy_ndimage is None:
        return {
            "status": "unavailable",
            "reason": "scipy.ndimage unavailable",
            "limits": limits,
            "metrics": {
                "local_quality_status": "unavailable",
                "local_connected_component_max_area": 0,
                "local_nonstellar_shape_component_count": 0,
                "local_single_pixel_component_ratio": 0.0,
                "local_cyan_blue_component_max_area": 0,
                "core_color_jump_component_max_area": 0,
                "local_color_risk_score": 1.0,
            },
        }

    positive_peak = _pixel_peak(positive_change)
    positive_floor = _pixel_floor(positive_change)
    positive_saturation = np.divide(
        positive_peak - positive_floor,
        np.maximum(positive_peak, 1e-12),
        out=np.zeros_like(positive_peak),
        where=positive_peak > 0.0,
    )
    supported_star_mask = np.zeros_like(positive_peak, dtype=bool)
    if confirmed_star_support is not None:
        supplied_support = np.asarray(confirmed_star_support, dtype=bool)
        if supplied_support.shape == positive_peak.shape:
            supported_star_mask = supplied_support
    raw_component_mask = positive_peak > component_peak_min
    # Source-matched star support is expected to contain compact positive
    # additions. Keep the raw measurement for diagnostics, but gate only
    # additions outside confirmed stars.
    component_mask = raw_component_mask & ~supported_star_mask
    _, raw_component_count, raw_component_areas = _component_areas(
        raw_component_mask
    )
    raw_component_max_area = (
        int(np.max(raw_component_areas)) if raw_component_areas.size else 0
    )
    labels, component_count, component_areas = _component_areas(component_mask)
    component_max_area = int(np.max(component_areas)) if component_areas.size else 0
    single_pixel_ratio = (
        float(np.mean(component_areas <= fragment_area_max))
        if component_areas.size
        else 0.0
    )
    nonstellar_shape_areas = []
    for index, bounds in enumerate(scipy_ndimage.find_objects(labels), start=1):
        if bounds is None:
            continue
        area = int(component_areas[index - 1])
        height = int(bounds[0].stop - bounds[0].start)
        width = int(bounds[1].stop - bounds[1].start)
        longest = max(height, width)
        shortest = max(1, min(height, width))
        aspect_ratio = float(longest / shortest)
        fill_ratio = float(area / max(1, height * width))
        if area >= nonstellar_min_area and (
            aspect_ratio > component_aspect_max
            or fill_ratio < component_fill_min
        ):
            nonstellar_shape_areas.append(area)

    positive_rgb = _rgb_channels(positive_change)
    red, green, blue = positive_rgb
    raw_cyan_blue_mask = (
        (positive_peak > cyan_peak_min)
        & (positive_saturation > cyan_saturation_min)
        & (blue > red * 1.20)
        & (((green + blue) * 0.5) > red * 1.25)
    )
    # Blue/cyan additions are expected for real blue stars. Only unmatched
    # additions are an artifact gate; retain raw measurements for diagnostics.
    cyan_blue_mask = raw_cyan_blue_mask & ~supported_star_mask
    _, raw_cyan_count, raw_cyan_areas = _component_areas(raw_cyan_blue_mask)
    raw_cyan_max_area = (
        int(np.max(raw_cyan_areas)) if raw_cyan_areas.size else 0
    )
    _, cyan_count, cyan_areas = _component_areas(cyan_blue_mask)
    cyan_max_area = int(np.max(cyan_areas)) if cyan_areas.size else 0

    base_luminance = _luminance(base_norm)
    signal_floor = float(np.percentile(base_luminance, 35.0))
    signal_mask = base_luminance > signal_floor
    core_samples = base_luminance[signal_mask]
    if core_samples.size < 32:
        core_samples = base_luminance.reshape(-1)
    core_threshold = float(np.percentile(core_samples, core_percentile))
    core_mask = base_luminance >= core_threshold
    base_rgb = _rgb_channels(base_norm)
    candidate_rgb = _rgb_channels(candidate_norm)
    base_sum = np.sum(base_rgb, axis=0)
    candidate_sum = np.sum(candidate_rgb, axis=0)
    base_ratio = np.divide(
        base_rgb,
        np.maximum(base_sum, 1e-6)[np.newaxis, ...],
        out=np.zeros_like(base_rgb),
        where=base_sum[np.newaxis, ...] > 1e-6,
    )
    candidate_ratio = np.divide(
        candidate_rgb,
        np.maximum(candidate_sum, 1e-6)[np.newaxis, ...],
        out=np.zeros_like(candidate_rgb),
        where=candidate_sum[np.newaxis, ...] > 1e-6,
    )
    color_jump = np.max(np.abs(candidate_ratio - base_ratio), axis=0)
    raw_core_jump_mask = (
        core_mask
        & (positive_peak > component_peak_min)
        & (color_jump > core_color_jump_min)
    )
    # A verified star layer is allowed to restore stellar colour inside its
    # own support. Unmatched core colour jumps remain a hard artifact gate.
    core_jump_mask = raw_core_jump_mask & ~supported_star_mask
    _, raw_core_jump_count, raw_core_jump_areas = _component_areas(
        raw_core_jump_mask
    )
    raw_core_jump_max_area = (
        int(np.max(raw_core_jump_areas)) if raw_core_jump_areas.size else 0
    )
    _, core_jump_count, core_jump_areas = _component_areas(core_jump_mask)
    core_jump_max_area = (
        int(np.max(core_jump_areas)) if core_jump_areas.size else 0
    )
    core_jump_ratio = float(
        np.count_nonzero(core_jump_mask) / max(1, np.count_nonzero(core_mask))
    )
    local_color_risk_score = max(
        cyan_max_area / max(float(cyan_area_max), 1.0),
        core_jump_max_area / max(float(core_jump_area_max), 1.0),
    )
    metrics = {
        "local_quality_status": "ok",
        "local_component_peak_min": component_peak_min,
        "local_connected_component_count": component_count,
        "local_connected_component_max_area": component_max_area,
        "local_connected_component_count_raw": raw_component_count,
        "local_connected_component_max_area_raw": raw_component_max_area,
        "local_confirmed_star_support_ratio": float(
            np.mean(supported_star_mask)
        ),
        "local_nonstellar_shape_component_count": len(nonstellar_shape_areas),
        "local_nonstellar_shape_component_max_area": (
            max(nonstellar_shape_areas) if nonstellar_shape_areas else 0
        ),
        "local_single_pixel_component_ratio": single_pixel_ratio,
        "local_fragment_component_area_max_effective": fragment_area_max,
        "local_nonstellar_shape_area_min_effective": nonstellar_min_area,
        "local_cyan_blue_component_count": cyan_count,
        "local_cyan_blue_component_max_area": cyan_max_area,
        "local_cyan_blue_pixel_ratio": float(np.mean(cyan_blue_mask)),
        "local_cyan_blue_component_count_raw": raw_cyan_count,
        "local_cyan_blue_component_max_area_raw": raw_cyan_max_area,
        "local_cyan_blue_pixel_ratio_raw": float(np.mean(raw_cyan_blue_mask)),
        "local_cyan_blue_confirmed_star_pixel_ratio": float(
            np.mean(raw_cyan_blue_mask & supported_star_mask)
        ),
        "local_cyan_blue_peak_min": cyan_peak_min,
        "local_cyan_blue_saturation_min": cyan_saturation_min,
        "core_signal_percentile": core_percentile,
        "core_signal_threshold": core_threshold,
        "core_color_jump_min": core_color_jump_min,
        "core_color_jump_component_count": core_jump_count,
        "core_color_jump_component_max_area": core_jump_max_area,
        "core_color_jump_component_count_raw": raw_core_jump_count,
        "core_color_jump_component_max_area_raw": raw_core_jump_max_area,
        "core_color_jump_pixel_ratio": core_jump_ratio,
        "local_color_risk_score": min(1.0, float(local_color_risk_score)),
    }
    return {
        "status": "ok",
        "reason": "",
        "limits": limits,
        "metrics": metrics,
    }


def screen_blend(
    base: np.ndarray,
    stars: np.ndarray,
    intensity: float,
    *,
    alpha_mask: np.ndarray | None = None,
    weak_mask: np.ndarray | None = None,
    bright_mask: np.ndarray | None = None,
    weak_intensity: float | None = None,
) -> np.ndarray:
    """Apply an explicit top star layer over a starless base with Screen headroom."""
    base_arr = np.asarray(base)
    base_scale = _image_scale(base_arr)
    base_norm = _normalized(base_arr, scale=base_scale)
    stars_norm = _normalized(np.asarray(stars))
    spatial_shape = _pixel_peak(base_norm).shape
    intensity_map = np.full(spatial_shape, float(intensity), dtype=np.float32)
    if weak_mask is not None and weak_intensity is not None:
        weak_spatial = np.asarray(weak_mask, dtype=bool)
        if weak_spatial.shape != spatial_shape:
            raise ValueError(
                f"weak overlay mask shape mismatch: {weak_spatial.shape}!={spatial_shape}"
            )
        intensity_map = np.where(
            weak_spatial,
            max(float(intensity), float(weak_intensity)),
            intensity_map,
        )
    if bright_mask is not None:
        bright_spatial = np.asarray(bright_mask, dtype=bool)
        if bright_spatial.shape != spatial_shape:
            raise ValueError(
                f"bright overlay mask shape mismatch: {bright_spatial.shape}!={spatial_shape}"
            )
        intensity_map = np.where(bright_spatial, float(intensity), intensity_map)
    star_term = np.clip(
        stars_norm
        * _expanded_spatial_mask(stars_norm, intensity_map.astype(np.float32)),
        0.0,
        1.0,
    )
    screened = 1.0 - (1.0 - base_norm) * (1.0 - star_term)
    if alpha_mask is None:
        mixed_norm = screened
    else:
        alpha_spatial = np.asarray(alpha_mask, dtype=np.float32)
        if alpha_spatial.shape != spatial_shape:
            raise ValueError(
                f"star alpha shape mismatch: {alpha_spatial.shape}!={spatial_shape}"
            )
        alpha = np.clip(
            _expanded_spatial_mask(base_norm, alpha_spatial),
            0.0,
            1.0,
        )
        mixed_norm = base_norm * (1.0 - alpha) + screened * alpha

    if np.issubdtype(base_arr.dtype, np.integer):
        info = np.iinfo(base_arr.dtype)
        return np.rint(mixed_norm * base_scale).clip(info.min, info.max).astype(
            base_arr.dtype,
            copy=False,
        )
    return (mixed_norm * base_scale).astype(np.float32, copy=False)


def unscreen_layer(
    original_display: np.ndarray,
    starless_display: np.ndarray,
    *,
    denominator_floor: float = 0.08,
) -> np.ndarray:
    """Recover a display-domain star layer with the inverse Screen formula."""
    original = np.asarray(original_display)
    starless = np.asarray(starless_display)
    if original.shape != starless.shape:
        raise ValueError(
            "unscreen pair shape mismatch: "
            f"original={original.shape}, starless={starless.shape}"
        )
    if original.ndim not in (2, 3):
        raise ValueError(f"unsupported unscreen image rank: {original.ndim}")
    original_norm = _normalized(original)
    starless_norm = _normalized(starless)
    floor = _bounded(denominator_floor, 0.08, 0.02, 0.25)
    denominator = np.maximum(1.0 - starless_norm, floor)
    recovered = (original_norm - starless_norm) / denominator
    return np.clip(recovered, 0.0, 1.0).astype(np.float32, copy=False)


def build_local_chroma_recovery_layer(
    star_layer: np.ndarray,
    remix_base: np.ndarray,
    candidate_display: np.ndarray,
    support_mask: np.ndarray,
    strict_core_mask: np.ndarray,
    cfg: Any,
    *,
    attenuation: float,
) -> tuple[np.ndarray | None, Dict[str, Any]]:
    """Attenuate only formally offending additions outside the strict core."""
    report: Dict[str, Any] = {
        "schema": "starun.stage9-local-chroma-recovery.v1",
        "status": "unavailable",
        "available": False,
        "changed": False,
        "strict_core_immutable": True,
        "support_expansion": False,
        "reason_code": "stage9_local_chroma_recovery_unavailable",
    }
    try:
        stars = _normalized(np.asarray(star_layer))
        base = _normalized(np.asarray(remix_base))
        candidate = _normalized(np.asarray(candidate_display))
        if not (stars.shape == base.shape == candidate.shape):
            raise ValueError("local chroma recovery image shapes do not match")
        spatial_shape = _pixel_peak(stars).shape
        support = np.asarray(support_mask, dtype=bool)
        strict_core = np.asarray(strict_core_mask, dtype=bool)
        if support.shape != spatial_shape or strict_core.shape != spatial_shape:
            raise ValueError("local chroma recovery support shape mismatch")

        peak_min = _bounded(
            getattr(cfg, "stage9_chromatic_addition_peak_min", 0.02),
            0.02,
            0.001,
            0.25,
        )
        saturation_min = _bounded(
            getattr(cfg, "stage9_chromatic_addition_saturation_min", 0.70),
            0.70,
            0.10,
            1.00,
        )
        strength = _bounded(attenuation, 0.75, 0.05, 0.95)
        positive = np.clip(candidate - base, 0.0, 1.0)
        positive_peak = _pixel_peak(positive)
        positive_floor = _pixel_floor(positive)
        saturation = np.divide(
            positive_peak - positive_floor,
            np.maximum(positive_peak, 1e-7),
            out=np.zeros_like(positive_peak, dtype=np.float32),
            where=positive_peak > 1e-7,
        )
        raw_offending = (
            support
            & (positive_peak > peak_min)
            & (saturation > saturation_min)
        )
        protected_offending = raw_offending & strict_core
        recoverable = raw_offending & ~strict_core
        recoverable_count = int(np.count_nonzero(recoverable))
        report.update(
            available=True,
            attenuation=strength,
            chromatic_addition_peak_min=peak_min,
            chromatic_addition_saturation_min=saturation_min,
            offending_pixel_count=int(np.count_nonzero(raw_offending)),
            protected_strict_core_pixel_count=int(
                np.count_nonzero(protected_offending)
            ),
            recoverable_pixel_count=recoverable_count,
            recoverable_ratio=float(np.mean(recoverable)),
        )
        if recoverable_count <= 0:
            report.update(
                status="not_needed",
                reason_code="stage9_local_chroma_recovery_no_outer_pixels",
                reason=(
                    "all chromatic additions are absent or belong to the "
                    "immutable strict core"
                ),
            )
            return None, report

        scale = np.ones(spatial_shape, dtype=np.float32)
        scale[recoverable] = strength
        recovered = stars * _expanded_spatial_mask(stars, scale)
        recovered *= _expanded_spatial_mask(stars, support.astype(np.float32))
        strict_delta = _pixel_peak(np.abs(recovered - stars))[strict_core]
        strict_change_max = (
            float(np.max(strict_delta)) if strict_delta.size else 0.0
        )
        if strict_change_max > 1e-7:
            raise ValueError(
                "local chroma recovery changed immutable strict core: "
                f"max={strict_change_max:.9f}"
            )
        report.update(
            status="ready",
            changed=True,
            reason_code="stage9_local_chroma_recovery_ready",
            strict_core_change_max=strict_change_max,
            changed_pixel_count=int(
                np.count_nonzero(_pixel_peak(np.abs(recovered - stars)) > 1e-7)
            ),
            semantics=(
                "hue_preserving_scalar_attenuation_of_formally_offending_"
                "outer_additions_only"
            ),
        )
        return recovered.astype(np.float32, copy=False), report
    except (IndexError, TypeError, ValueError, FloatingPointError) as error:
        report["reason"] = str(error)
        return None, report


def _stage9_roundtrip_error_summary(
    actual: np.ndarray,
    reconstructed: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Return JSON-safe RGB error evidence without inventing unavailable zeros."""
    try:
        actual_rgb = _rgb_channels(np.asarray(actual, dtype=np.float32))
        reconstructed_rgb = _rgb_channels(
            np.asarray(reconstructed, dtype=np.float32)
        )
        if actual_rgb.shape != reconstructed_rgb.shape:
            raise ValueError("roundtrip image shapes do not match")
        error = np.abs(reconstructed_rgb - actual_rgb)
        if mask is None:
            samples = error.reshape(-1)
            spatial_count = int(error.shape[1] * error.shape[2])
        else:
            spatial = np.asarray(mask, dtype=bool)
            if spatial.shape != error.shape[1:]:
                raise ValueError("roundtrip mask shape does not match")
            spatial_count = int(np.count_nonzero(spatial))
            if spatial_count <= 0:
                raise ValueError("roundtrip mask contains no samples")
            samples = error[:, spatial].reshape(-1)
        if samples.size <= 0 or not np.all(np.isfinite(samples)):
            raise ValueError("roundtrip error samples are unavailable")
        return {
            "status": "ok",
            "spatial_sample_count": spatial_count,
            "rgb_mae": float(np.mean(samples)),
            "rgb_p95": float(np.percentile(samples, 95.0)),
            "rgb_max": float(np.max(samples)),
        }
    except (IndexError, TypeError, ValueError, FloatingPointError) as error:
        return {"status": "unavailable", "reason": str(error)}


def assess_linear_decomposition_roundtrip(
    original_linear: np.ndarray,
    starless_linear: np.ndarray,
) -> Dict[str, Any]:
    """Audit the no-edit O = B + (O-B) linear decomposition identity."""
    try:
        original = np.asarray(original_linear, dtype=np.float32)
        starless = np.asarray(starless_linear, dtype=np.float32)
        if original.shape != starless.shape:
            raise ValueError("linear pair shapes do not match")
        if not np.all(np.isfinite(original)) or not np.all(np.isfinite(starless)):
            raise ValueError("linear pair contains non-finite pixels")
        star_layer = original - starless
        reconstructed = starless + star_layer
        return {
            "schema": "starun.stage9-linear-roundtrip.v1",
            "status": "ok",
            "operator": "O = B + (O - B)",
            "clipping_applied": False,
            "error": _stage9_roundtrip_error_summary(
                original,
                reconstructed,
            ),
        }
    except (IndexError, TypeError, ValueError, FloatingPointError) as error:
        return {
            "schema": "starun.stage9-linear-roundtrip.v1",
            "status": "unavailable",
            "reason": str(error),
        }


def assess_unscreen_operator_roundtrip(
    original_display: np.ndarray,
    starless_display: np.ndarray,
    stabilized_unscreen: np.ndarray,
    support_mask: np.ndarray,
    *,
    denominator_floor: float,
) -> Dict[str, Any]:
    """Audit raw Unscreen closure and the stabilized candidate's deviation."""
    report: Dict[str, Any] = {
        "schema": "starun.stage9-unscreen-operator-audit.v1",
        "status": "unavailable",
        "candidate_semantics": (
            "chroma_preserving_amplitude_correction_not_channelwise_exact_inverse"
        ),
    }
    try:
        original = _normalized(np.asarray(original_display))
        starless = _normalized(np.asarray(starless_display))
        stabilized = _normalized(np.asarray(stabilized_unscreen))
        if not (original.shape == starless.shape == stabilized.shape):
            raise ValueError("Unscreen audit image shapes do not match")
        support = np.asarray(support_mask, dtype=bool)
        if support.shape != _pixel_peak(original).shape:
            raise ValueError("Unscreen audit support shape does not match")
        denominator = 1.0 - starless
        residual = original - starless
        safe = (
            support
            & (_pixel_floor(denominator) >= float(denominator_floor))
            & (_pixel_floor(residual) >= 0.0)
            & (_pixel_peak(original) < 0.995)
        )
        if not np.any(safe):
            raise ValueError("raw Unscreen safe support contains no samples")
        raw_unscreen = np.divide(
            residual,
            denominator,
            out=np.zeros_like(residual, dtype=np.float32),
            where=denominator > 0.0,
        ).astype(np.float32, copy=False)
        raw_reconstructed = 1.0 - (1.0 - starless) * (1.0 - raw_unscreen)
        alpha_composed = screen_blend(
            starless,
            stabilized,
            1.0,
            alpha_mask=support.astype(np.float32),
        )
        outside = ~support
        report.update(
            status="ok",
            denominator_floor=float(denominator_floor),
            safe_support_pixel_count=int(np.count_nonzero(safe)),
            raw_unscreen_screen_roundtrip=_stage9_roundtrip_error_summary(
                original,
                raw_reconstructed,
                mask=safe,
            ),
            stabilized_deviation_from_raw=_stage9_roundtrip_error_summary(
                raw_unscreen,
                stabilized,
                mask=safe,
            ),
            alpha_support_outside_change=_stage9_roundtrip_error_summary(
                starless,
                alpha_composed,
                mask=outside,
            ),
        )
        return report
    except (IndexError, TypeError, ValueError, FloatingPointError) as error:
        report["reason"] = str(error)
        return report


def _stage9_mono_or_rgb_layout(image: np.ndarray) -> str:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return "mono"
    if arr.ndim != 3:
        raise ValueError(f"unsupported Stage9 image rank: {arr.ndim}")
    if arr.shape[0] in (1, 3):
        return "chw"
    if arr.shape[-1] in (1, 3):
        return "hwc"
    raise ValueError(
        "Stage9 matched-domain pair must be mono or RGB: "
        f"shape={arr.shape}"
    )


def build_chroma_stable_unscreen_layer(
    original_display: np.ndarray,
    starless_display: np.ndarray,
    trusted_stars: np.ndarray,
    support_mask: np.ndarray,
    cfg: Any,
) -> tuple[np.ndarray | None, Dict[str, Any]]:
    """Build a reliable Unscreen amplitude layer with trusted RGB proportions."""
    report: Dict[str, Any] = {
        "schema": "starun.stage9-unscreen-reference.v1",
        "status": "unavailable",
        "available": False,
        "reason_code": "stage9_unscreen_reference_unavailable",
    }
    try:
        original_arr = np.asarray(original_display)
        starless_arr = np.asarray(starless_display)
        trusted_arr = np.asarray(trusted_stars)
        if not (
            original_arr.shape == starless_arr.shape == trusted_arr.shape
        ):
            raise ValueError(
                "matched-domain pair/trusted layer shape mismatch: "
                f"original={original_arr.shape}, starless={starless_arr.shape}, "
                f"trusted={trusted_arr.shape}"
            )
        layout = _stage9_mono_or_rgb_layout(original_arr)
        if _stage9_mono_or_rgb_layout(starless_arr) != layout:
            raise ValueError("matched-domain pair layout mismatch")
        if _stage9_mono_or_rgb_layout(trusted_arr) != layout:
            raise ValueError("trusted star-layer layout mismatch")
        if not (
            np.all(np.isfinite(original_arr))
            and np.all(np.isfinite(starless_arr))
            and np.all(np.isfinite(trusted_arr))
        ):
            raise ValueError("matched-domain pair contains non-finite pixels")

        original = _normalized(original_arr)
        starless = _normalized(starless_arr)
        trusted = _normalized(trusted_arr)
        spatial_shape = _pixel_peak(original).shape
        support = np.asarray(support_mask, dtype=bool)
        if support.shape != spatial_shape:
            raise ValueError(
                "unscreen support shape mismatch: "
                f"support={support.shape}, image={spatial_shape}"
            )
        support_count = int(np.count_nonzero(support))
        if support_count <= 0:
            raise ValueError("unscreen compact support is empty")

        denominator_floor = _bounded(
            getattr(cfg, "stage9_unscreen_denominator_floor", 0.08),
            0.08,
            0.02,
            0.25,
        )
        reliable_min = _bounded(
            getattr(cfg, "stage9_unscreen_reliable_support_min", 0.80),
            0.80,
            0.50,
            0.98,
        )
        peak_max = _bounded(
            getattr(cfg, "stage9_unscreen_peak_max", 0.95),
            0.95,
            0.75,
            0.98,
        )
        residual = original - starless
        denominator = 1.0 - starless
        outside = ~support
        outside_residual = _rgb_channels(residual)[:, outside]
        if outside_residual.size:
            residual_center = float(np.median(outside_residual))
            residual_mad = float(
                np.median(np.abs(outside_residual - residual_center))
            )
            negative_tolerance = float(
                np.clip(3.0 * 1.4826 * residual_mad, 2e-6, 0.002)
            )
        else:
            residual_mad = 0.0
            negative_tolerance = 0.002

        denominator_ok = _pixel_floor(denominator) >= denominator_floor
        unsaturated = _pixel_peak(original) < 0.995
        residual_ok = _pixel_floor(residual) >= -negative_tolerance
        trusted_peak = _pixel_peak(trusted)
        trusted_nonzero = trusted_peak > 1e-7
        reliable = (
            support
            & denominator_ok
            & unsaturated
            & residual_ok
            & trusted_nonzero
        )
        reliable_count = int(np.count_nonzero(reliable))
        reliable_ratio = float(reliable_count / support_count)
        raw_unscreen = np.clip(
            residual / np.maximum(denominator, denominator_floor),
            0.0,
            1.0,
        ).astype(np.float32, copy=False)
        raw_peak = _pixel_peak(raw_unscreen)
        recovered_peak = np.maximum(
            trusted_peak,
            np.minimum(raw_peak, peak_max),
        )
        output_peak = np.where(reliable, recovered_peak, trusted_peak)
        trusted_ratio = np.divide(
            trusted,
            np.maximum(
                _expanded_spatial_mask(trusted, trusted_peak),
                1e-7,
            ),
            out=np.zeros_like(trusted, dtype=np.float32),
            where=_expanded_spatial_mask(trusted, trusted_peak) > 1e-7,
        )
        stabilized = trusted_ratio * _expanded_spatial_mask(
            trusted_ratio,
            output_peak.astype(np.float32, copy=False),
        )
        stabilized *= _expanded_spatial_mask(
            stabilized,
            support.astype(np.float32),
        )
        stabilized = np.clip(stabilized, 0.0, 1.0).astype(
            np.float32,
            copy=False,
        )
        operator_audit = assess_unscreen_operator_roundtrip(
            original,
            starless,
            stabilized,
            support,
            denominator_floor=denominator_floor,
        )

        report.update(
            {
                "layout": layout,
                "support_pixel_count": support_count,
                "support_ratio": float(np.mean(support)),
                "reliable_pixel_count": reliable_count,
                "reliable_support_ratio": reliable_ratio,
                "fallback_support_ratio": max(0.0, 1.0 - reliable_ratio),
                "reliable_support_min": reliable_min,
                "denominator_floor": denominator_floor,
                "denominator_limited_ratio": float(
                    np.count_nonzero(support & ~denominator_ok) / support_count
                ),
                "saturated_reference_ratio": float(
                    np.count_nonzero(support & ~unsaturated) / support_count
                ),
                "negative_residual_ratio": float(
                    np.count_nonzero(support & ~residual_ok) / support_count
                ),
                "trusted_zero_ratio": float(
                    np.count_nonzero(support & ~trusted_nonzero) / support_count
                ),
                "outside_residual_mad": residual_mad,
                "negative_residual_tolerance": negative_tolerance,
                "peak_max": peak_max,
                "peak_capped_ratio": float(
                    np.count_nonzero(reliable & (raw_peak > peak_max))
                    / support_count
                ),
                "candidate_semantics": (
                    "chroma_preserving_amplitude_correction_not_channelwise_exact_inverse"
                ),
                "operator_audit": operator_audit,
            }
        )
        if reliable_ratio < reliable_min:
            report.update(
                status="unavailable",
                available=False,
                reason_code="stage9_unscreen_reliability_insufficient",
                reason=(
                    "reliable compact-support coverage below minimum: "
                    f"{reliable_ratio:.6f}<{reliable_min:.6f}"
                ),
            )
            return None, report
        report.update(
            status="ready",
            available=True,
            reason_code="stage9_unscreen_reference_ready",
        )
        return stabilized, report
    except (IndexError, TypeError, ValueError, FloatingPointError) as error:
        report["reason"] = str(error)
        return None, report


def build_source_wing_feather_candidate(
    original_display: np.ndarray,
    starless_display: np.ndarray,
    stabilized_stars: np.ndarray,
    strict_support_mask: np.ndarray,
    expanded_support_mask: np.ndarray,
    cfg: Any,
    *,
    feather_strength: float = 0.90,
) -> tuple[np.ndarray | None, np.ndarray | None, Dict[str, Any]]:
    """Restore only source-confirmed outer PSF wings with bounded amplitude.

    This is deliberately different from dilating the already-rendered star
    layer.  Pixels are eligible only when they belong to the independently
    rebuilt source-FWHM support and the matched O/B pair has a reliable,
    non-negative Unscreen solution.
    """
    report: Dict[str, Any] = {
        "schema": "starun.stage9-source-wing-feather.v1",
        "status": "unavailable",
        "available": False,
        "reason_code": "stage9_source_wing_feather_unavailable",
    }
    try:
        original = _normalized(np.asarray(original_display))
        starless = _normalized(np.asarray(starless_display))
        stabilized = _normalized(np.asarray(stabilized_stars))
        if not (original.shape == starless.shape == stabilized.shape):
            raise ValueError("source-wing feather image shapes do not match")
        spatial_shape = _pixel_peak(original).shape
        strict = np.asarray(strict_support_mask, dtype=bool)
        expanded = np.asarray(expanded_support_mask, dtype=bool)
        if strict.shape != spatial_shape or expanded.shape != spatial_shape:
            raise ValueError("source-wing feather support shape mismatch")
        outer_ring = expanded & ~strict
        ring_count = int(np.count_nonzero(outer_ring))
        if ring_count <= 0:
            raise ValueError("expanded source support adds no outer PSF wing")

        denominator_floor = _bounded(
            getattr(cfg, "stage9_unscreen_denominator_floor", 0.08),
            0.08,
            0.02,
            0.25,
        )
        strength = _bounded(feather_strength, 0.90, 0.10, 0.95)
        residual = original - starless
        denominator = 1.0 - starless
        reliable = (
            outer_ring
            & (_pixel_floor(denominator) >= denominator_floor)
            & (_pixel_floor(residual) >= -0.002)
            & (_pixel_peak(original) < 0.995)
        )
        reliable_count = int(np.count_nonzero(reliable))
        if reliable_count <= 0:
            raise ValueError("source-wing feather has no reliable matched pixels")
        raw_unscreen = np.clip(
            residual / np.maximum(denominator, denominator_floor),
            0.0,
            1.0,
        ).astype(np.float32, copy=False)
        feathered = np.array(stabilized, dtype=np.float32, copy=True)
        expanded_reliable = _expanded_spatial_mask(
            feathered,
            reliable.astype(np.float32),
        )
        feathered = np.where(
            expanded_reliable > 0.0,
            np.maximum(feathered, strength * raw_unscreen),
            feathered,
        )
        support = strict | reliable
        feathered *= _expanded_spatial_mask(
            feathered,
            support.astype(np.float32),
        )
        report.update(
            status="ready",
            available=True,
            reason_code="stage9_source_wing_feather_ready",
            semantics=(
                "matched_domain_raw_unscreen_outer_psf_wing_not_recursive_dilation"
            ),
            feather_strength=strength,
            expanded_ring_pixel_count=ring_count,
            reliable_ring_pixel_count=reliable_count,
            reliable_ring_ratio=float(reliable_count / ring_count),
            support_pixel_count=int(np.count_nonzero(support)),
            support_ratio=float(np.mean(support)),
            denominator_floor=denominator_floor,
        )
        return feathered.astype(np.float32, copy=False), support, report
    except (IndexError, TypeError, ValueError, FloatingPointError) as error:
        report["reason"] = str(error)
        return None, None, report


def build_selective_source_wing_candidate(
    original_display: np.ndarray,
    starless_display: np.ndarray,
    stabilized_stars: np.ndarray,
    candidate_display: np.ndarray,
    current_support_mask: np.ndarray,
    star_reference: Dict[str, Any],
    cfg: Any,
    *,
    remix_base: np.ndarray | None = None,
    visible_wing_reference: np.ndarray | None = None,
    screen_intensity: float = 1.0,
    fwhm_ratio_target: float = 1.08,
    feather_strength: float = 1.15,
    extra_pixels: int = 2,
    target_groups: tuple[str, ...] | None = None,
    recovery_alpha: float = 1.0,
) -> tuple[np.ndarray | None, np.ndarray | None, Dict[str, Any]]:
    """Restore source-confirmed outer wings only for still-small same stars.

    When a same-source linked-autostretch reference is available, selection is
    made at its locally background-subtracted 5%, 10%, and 25% peak contours.
    The 25% contour closes the visible mid-wing gap without changing the peak
    or the 50% FWHM core.  A star that is already large enough at 25% may still
    recover only its missing 5%--10% outer skirt; a star that is small at 10%
    may recover the source 5%--45% profile.  Otherwise the legacy matched-MTF
    FWHM selection remains available.  Saturated or unmeasurable stars are
    excluded.  In both modes the strict source-FWHM core is immutable, and
    every newly raised pixel is capped by both the source reference profile
    and the candidate half-max safety line.
    """
    report: Dict[str, Any] = {
        "schema": "starun.stage9-selective-source-wing.v1",
        "status": "unavailable",
        "available": False,
        "changed": False,
        "reason_code": "stage9_selective_source_wing_unavailable",
        "semantics": (
            "same_star_source_confirmed_display_visible_sub_halfmax_outer_wing"
        ),
        "strict_core_immutable": True,
        "halfmax_core_immutable": True,
        "recursive_dilation": False,
    }
    if scipy_ndimage is None:
        report["reason"] = "scipy.ndimage unavailable"
        return None, None, report
    try:
        original = _normalized(np.asarray(original_display))
        starless = _normalized(np.asarray(starless_display))
        stabilized = _normalized(np.asarray(stabilized_stars))
        candidate = _normalized(np.asarray(candidate_display))
        actual_base = _normalized(
            np.asarray(
                starless_display if remix_base is None else remix_base
            )
        )
        if not (
            original.shape
            == starless.shape
            == stabilized.shape
            == candidate.shape
            == actual_base.shape
        ):
            raise ValueError("selective source-wing image shapes do not match")
        if not isinstance(star_reference, dict) or star_reference.get("status") != "ok":
            raise ValueError("selective source-wing star reference is unavailable")

        visible_reference = None
        if visible_wing_reference is not None:
            visible_reference = _normalized(np.asarray(visible_wing_reference))
            if visible_reference.shape != candidate.shape:
                raise ValueError(
                    "visible source-wing reference shape does not match candidate"
                )

        spatial_shape = _pixel_peak(original).shape
        current_support = np.asarray(current_support_mask, dtype=bool)
        if current_support.shape != spatial_shape:
            raise ValueError("selective source-wing support shape mismatch")

        source_fwhm = np.asarray(
            star_reference.get("_display_source_fwhm_px", ()),
            dtype=np.float32,
        )
        valid = np.asarray(
            star_reference.get("_psf_valid_flags", ()),
            dtype=bool,
        )
        saturated = np.asarray(
            star_reference.get("_psf_saturated_flags", ()),
            dtype=bool,
        )
        peak_y = np.asarray(star_reference.get("_peak_y", ()), dtype=np.int32)
        peak_x = np.asarray(star_reference.get("_peak_x", ()), dtype=np.int32)
        weak_flags = np.asarray(
            star_reference.get("_weak_flags", ()),
            dtype=bool,
        )
        count = int(source_fwhm.size)
        if not (
            count > 0
            and valid.size == count
            and saturated.size == count
            and peak_y.size == count
            and peak_x.size == count
            and weak_flags.size == count
        ):
            raise ValueError("selective source-wing reference arrays are incomplete")

        target = _bounded(fwhm_ratio_target, 1.08, 0.93, 1.10)
        strength = _bounded(feather_strength, 1.15, 0.90, 1.25)
        soft_alpha = _bounded(recovery_alpha, 1.0, 0.0, 1.0)
        retry_pixels = max(0, min(int(extra_pixels), 2))
        radius_max_nominal = int(
            _bounded(
                getattr(cfg, "stage9_psf_support_radius_max", 6),
                6,
                2,
                12,
            )
        )
        candidate_peak = _pixel_peak(candidate)
        ordinary = valid & ~saturated & np.isfinite(source_fwhm) & (source_fwhm > 0.0)
        candidate_fwhm = np.full(count, np.nan, dtype=np.float32)
        candidate_valid = np.zeros(count, dtype=bool)
        candidate_background = np.full(count, np.nan, dtype=np.float32)
        candidate_signal = np.full(count, np.nan, dtype=np.float32)
        measurement_search_radii = []
        measurement_patch_radii = []
        for index in np.flatnonzero(ordinary):
            search_radius = stage9_scale_radius(
                2,
                star_reference,
                fwhm_px=source_fwhm[index],
                rounding="nearest",
                minimum=1,
            )
            patch_radius = stage9_scale_radius(
                6,
                star_reference,
                fwhm_px=source_fwhm[index],
                rounding="nearest",
                minimum=1,
            )
            measurement_search_radii.append(search_radius)
            measurement_patch_radii.append(patch_radius)
            measurement = _measure_connected_halfmax_fwhm(
                candidate_peak,
                int(peak_y[index]),
                int(peak_x[index]),
                search_radius=search_radius,
                patch_radius=patch_radius,
            )
            if measurement.get("status") != "ok":
                continue
            if (
                abs(int(measurement.get("offset_y", 99))) > search_radius
                or abs(int(measurement.get("offset_x", 99))) > search_radius
            ):
                continue
            candidate_fwhm[index] = float(measurement["fwhm_px"])
            candidate_background[index] = float(measurement["background"])
            candidate_signal[index] = float(measurement["signal"])
            candidate_valid[index] = True

        measurable = (
            ordinary
            & candidate_valid
            & np.isfinite(candidate_fwhm)
            & (candidate_fwhm > 0.0)
        )
        ratios = np.full(count, np.nan, dtype=np.float32)
        ratios[measurable] = (
            candidate_fwhm[measurable] / source_fwhm[measurable]
        )
        selection_mode = "matched_mtf_fwhm"
        visible_floor = _bounded(
            getattr(
                cfg,
                "stage9_source_autostretch_wing_floor_fraction",
                0.05,
            ),
            0.05,
            0.03,
            0.15,
        )
        visible_selection_fraction = max(0.10, 2.0 * visible_floor)
        visible_mid_fraction = 0.25
        visible_target = _bounded(
            getattr(
                cfg,
                "stage9_source_autostretch_wing_target_ratio",
                1.03,
            ),
            1.03,
            0.90,
            1.10,
        )
        visible_radius_max_nominal = int(
            _bounded(
                getattr(
                    cfg,
                    "stage9_source_autostretch_wing_radius_max",
                    10,
                ),
                10,
                6,
                16,
            )
        )
        source_visible_diameter = np.full(count, np.nan, dtype=np.float32)
        candidate_visible_diameter = np.full(count, np.nan, dtype=np.float32)
        visible_ratios = np.full(count, np.nan, dtype=np.float32)
        visible_measurable = np.zeros(count, dtype=bool)
        source_floor_diameter = np.full(count, np.nan, dtype=np.float32)
        candidate_floor_diameter = np.full(count, np.nan, dtype=np.float32)
        floor_ratios = np.full(count, np.nan, dtype=np.float32)
        floor_measurable = np.zeros(count, dtype=bool)
        source_mid_diameter = np.full(count, np.nan, dtype=np.float32)
        candidate_mid_diameter = np.full(count, np.nan, dtype=np.float32)
        mid_ratios = np.full(count, np.nan, dtype=np.float32)
        mid_measurable = np.zeros(count, dtype=bool)
        selected_at_visible_fraction = np.zeros(count, dtype=bool)
        selected_at_floor_fraction = np.zeros(count, dtype=bool)
        selected_at_mid_fraction = np.zeros(count, dtype=bool)
        if visible_reference is not None:
            selection_mode = (
                "same_source_linked_autostretch_5pct_10pct_25pct_footprint"
            )
            visible_peak = _pixel_peak(visible_reference)
            for index in np.flatnonzero(ordinary):
                visible_radius_max = stage9_scale_radius(
                    visible_radius_max_nominal,
                    star_reference,
                    fwhm_px=source_fwhm[index],
                    rounding="nearest",
                    minimum=1,
                )
                search_radius = stage9_scale_radius(
                    2,
                    star_reference,
                    fwhm_px=source_fwhm[index],
                    rounding="nearest",
                    minimum=1,
                )
                source_mid_measurement = (
                    _measure_connected_peak_fraction_footprint(
                        visible_peak,
                        int(peak_y[index]),
                        int(peak_x[index]),
                        peak_fraction=visible_mid_fraction,
                        search_radius=search_radius,
                        patch_radius=visible_radius_max,
                    )
                )
                candidate_mid_measurement = (
                    _measure_connected_peak_fraction_footprint(
                        candidate_peak,
                        int(peak_y[index]),
                        int(peak_x[index]),
                        peak_fraction=visible_mid_fraction,
                        search_radius=search_radius,
                        patch_radius=visible_radius_max,
                    )
                )
                if (
                    source_mid_measurement.get("status") == "ok"
                    and candidate_mid_measurement.get("status") == "ok"
                    and all(
                        abs(int(measurement.get(offset, 99))) <= search_radius
                        for measurement in (
                            source_mid_measurement,
                            candidate_mid_measurement,
                        )
                        for offset in ("offset_y", "offset_x")
                    )
                ):
                    source_mid = float(
                        source_mid_measurement["equivalent_diameter_px"]
                    )
                    candidate_mid = float(
                        candidate_mid_measurement["equivalent_diameter_px"]
                    )
                    if source_mid > 0.0 and candidate_mid > 0.0:
                        source_mid_diameter[index] = source_mid
                        candidate_mid_diameter[index] = candidate_mid
                        mid_ratios[index] = candidate_mid / source_mid
                        mid_measurable[index] = True
                        candidate_background[index] = float(
                            candidate_mid_measurement["background"]
                        )
                        candidate_signal[index] = float(
                            candidate_mid_measurement["signal"]
                        )

                source_measurement = _measure_connected_peak_fraction_footprint(
                    visible_peak,
                    int(peak_y[index]),
                    int(peak_x[index]),
                    peak_fraction=visible_selection_fraction,
                    search_radius=search_radius,
                    patch_radius=visible_radius_max,
                )
                candidate_measurement = _measure_connected_peak_fraction_footprint(
                    candidate_peak,
                    int(peak_y[index]),
                    int(peak_x[index]),
                    peak_fraction=visible_selection_fraction,
                    search_radius=search_radius,
                    patch_radius=visible_radius_max,
                )
                if (
                    source_measurement.get("status") != "ok"
                    or candidate_measurement.get("status") != "ok"
                ):
                    source_measurement = None
                    candidate_measurement = None
                elif any(
                    abs(int(measurement.get(offset, 99))) > search_radius
                    for measurement in (source_measurement, candidate_measurement)
                    for offset in ("offset_y", "offset_x")
                ):
                    source_measurement = None
                    candidate_measurement = None
                if source_measurement is not None and candidate_measurement is not None:
                    source_diameter = float(
                        source_measurement["equivalent_diameter_px"]
                    )
                    candidate_diameter = float(
                        candidate_measurement["equivalent_diameter_px"]
                    )
                    if source_diameter > 0.0 and candidate_diameter > 0.0:
                        source_visible_diameter[index] = source_diameter
                        candidate_visible_diameter[index] = candidate_diameter
                        visible_ratios[index] = candidate_diameter / source_diameter
                        visible_measurable[index] = True
                        # The locally background-subtracted footprint owns the
                        # display threshold used for visible-wing recovery.  Do
                        # not require a second half-max fit here: crowded/high-
                        # background bright stars can have a sound sub-halfmax
                        # contour even when that fit fails.
                        candidate_background[index] = float(
                            candidate_measurement["background"]
                        )
                        candidate_signal[index] = float(
                            candidate_measurement["signal"]
                        )

                source_floor_measurement = (
                    _measure_connected_peak_fraction_footprint(
                        visible_peak,
                        int(peak_y[index]),
                        int(peak_x[index]),
                        peak_fraction=visible_floor,
                        search_radius=search_radius,
                        patch_radius=visible_radius_max,
                    )
                )
                candidate_floor_measurement = (
                    _measure_connected_peak_fraction_footprint(
                        candidate_peak,
                        int(peak_y[index]),
                        int(peak_x[index]),
                        peak_fraction=visible_floor,
                        search_radius=search_radius,
                        patch_radius=visible_radius_max,
                    )
                )
                if (
                    source_floor_measurement.get("status") != "ok"
                    or candidate_floor_measurement.get("status") != "ok"
                ):
                    continue
                if any(
                    abs(int(measurement.get(offset, 99))) > search_radius
                    for measurement in (
                        source_floor_measurement,
                        candidate_floor_measurement,
                    )
                    for offset in ("offset_y", "offset_x")
                ):
                    continue
                source_floor = float(
                    source_floor_measurement["equivalent_diameter_px"]
                )
                candidate_floor = float(
                    candidate_floor_measurement["equivalent_diameter_px"]
                )
                if source_floor <= 0.0 or candidate_floor <= 0.0:
                    continue
                source_floor_diameter[index] = source_floor
                candidate_floor_diameter[index] = candidate_floor
                floor_ratios[index] = candidate_floor / source_floor
                floor_measurable[index] = True
            finite_candidate_profile = (
                np.isfinite(candidate_background)
                & np.isfinite(candidate_signal)
                & (candidate_signal > 0.0)
            )
            visible_assessable = visible_measurable & finite_candidate_profile
            floor_assessable = floor_measurable & finite_candidate_profile
            mid_assessable = mid_measurable & finite_candidate_profile
            selected_at_visible_fraction = visible_assessable & (
                visible_ratios < visible_target
            )
            selected_at_floor_fraction = floor_assessable & (
                floor_ratios < visible_target
            )
            selected_at_mid_fraction = mid_assessable & (
                mid_ratios < visible_target
            )
            selected = (
                selected_at_visible_fraction
                | selected_at_floor_fraction
                | selected_at_mid_fraction
            )
            visible_assessable_any = (
                visible_assessable | floor_assessable | mid_assessable
            )
            measurable_count = int(np.count_nonzero(visible_assessable_any))
        else:
            visible_assessable = np.zeros(count, dtype=bool)
            floor_assessable = np.zeros(count, dtype=bool)
            mid_assessable = np.zeros(count, dtype=bool)
            selected = measurable & (ratios < target)
            measurable_count = int(np.count_nonzero(measurable))
        normalized_target_groups = tuple(
            group
            for group in dict.fromkeys(target_groups or ())
            if group in {"weak", "bright"}
        )
        if normalized_target_groups:
            selected_groups = np.zeros(count, dtype=bool)
            if "weak" in normalized_target_groups:
                selected_groups |= weak_flags
            if "bright" in normalized_target_groups:
                selected_groups |= ~weak_flags
            selected &= selected_groups
        selected_count = int(np.count_nonzero(selected))
        report.update(
            available=True,
            status="not_needed" if selected_count <= 0 else "ready",
            reason_code=(
                "stage9_selective_source_wing_not_needed"
                if selected_count <= 0
                else "stage9_selective_source_wing_ready"
            ),
            fwhm_ratio_target=target,
            selection_mode=selection_mode,
            source_autostretch_reference_available=bool(
                visible_reference is not None
            ),
            visible_wing_floor_peak_fraction=visible_floor,
            visible_wing_selection_peak_fraction=visible_selection_fraction,
            visible_mid_wing_peak_fraction=visible_mid_fraction,
            visible_selection_peak_fractions=[
                float(visible_floor),
                float(visible_selection_fraction),
                float(visible_mid_fraction),
            ],
            visible_wing_target_ratio=visible_target,
            visible_wing_radius_max=visible_radius_max_nominal,
            visible_wing_radius_max_nominal=visible_radius_max_nominal,
            feather_strength=strength,
            recovery_alpha=soft_alpha,
            target_groups=list(normalized_target_groups),
            support_extra_pixels=retry_pixels,
            support_extra_pixels_nominal=retry_pixels,
            support_radius_max=radius_max_nominal,
            support_radius_max_nominal=radius_max_nominal,
            measurement_search_radius_effective_px=(
                stage9_effective_pixel_stats(measurement_search_radii)
            ),
            measurement_patch_radius_effective_px=(
                stage9_effective_pixel_stats(measurement_patch_radii)
            ),
            ordinary_reference_sample_count=int(np.count_nonzero(ordinary)),
            candidate_measurable_sample_count=measurable_count,
            selected_star_count=selected_count,
            selected_at_10pct_star_count=int(
                np.count_nonzero(selected_at_visible_fraction)
            ),
            selected_at_25pct_star_count=int(
                np.count_nonzero(selected_at_mid_fraction)
            ),
            selected_at_25pct_only_star_count=int(
                np.count_nonzero(
                    selected_at_mid_fraction
                    & ~selected_at_visible_fraction
                    & ~selected_at_floor_fraction
                )
            ),
            selected_at_5pct_star_count=int(
                np.count_nonzero(selected_at_floor_fraction)
            ),
            selected_at_5pct_only_star_count=int(
                np.count_nonzero(
                    selected_at_floor_fraction
                    & ~selected_at_visible_fraction
                )
            ),
            selected_weak_star_count=int(np.count_nonzero(selected & weak_flags)),
            selected_bright_star_count=int(np.count_nonzero(selected & ~weak_flags)),
            selected_star_ratio=(
                float(selected_count / measurable_count)
                if measurable_count > 0
                else 0.0
            ),
        )
        if selection_mode == "matched_mtf_fwhm" and measurable_count > 0:
            measured_ratios = ratios[measurable]
            report["before_fwhm_ratio_p25"] = float(
                np.percentile(measured_ratios, 25.0)
            )
            report["before_fwhm_ratio_median"] = float(
                np.median(measured_ratios)
            )
            report["before_fwhm_ratio_p75"] = float(
                np.percentile(measured_ratios, 75.0)
            )
        elif np.any(visible_assessable):
            measured_visible_ratios = visible_ratios[visible_assessable]
            report["before_visible_wing_ratio_p25"] = float(
                np.percentile(measured_visible_ratios, 25.0)
            )
            report["before_visible_wing_ratio_median"] = float(
                np.median(measured_visible_ratios)
            )
            report["before_visible_wing_ratio_p75"] = float(
                np.percentile(measured_visible_ratios, 75.0)
            )
        if visible_reference is not None and np.any(floor_assessable):
            measured_floor_ratios = floor_ratios[floor_assessable]
            report["before_visible_floor_ratio_p25"] = float(
                np.percentile(measured_floor_ratios, 25.0)
            )
            report["before_visible_floor_ratio_median"] = float(
                np.median(measured_floor_ratios)
            )
            report["before_visible_floor_ratio_p75"] = float(
                np.percentile(measured_floor_ratios, 75.0)
            )
        if visible_reference is not None and np.any(mid_assessable):
            measured_mid_ratios = mid_ratios[mid_assessable]
            report["before_visible_mid_ratio_p25"] = float(
                np.percentile(measured_mid_ratios, 25.0)
            )
            report["before_visible_mid_ratio_median"] = float(
                np.median(measured_mid_ratios)
            )
            report["before_visible_mid_ratio_p75"] = float(
                np.percentile(measured_mid_ratios, 75.0)
            )
        if selected_count <= 0:
            return None, None, report

        # Rebuild the ordinary strict support from the frozen source FWHM.  It
        # is used only as an immutable core exclusion, never as a dilation seed.
        _strict_weak, _strict_bright, strict_support = _catalog_support_masks(
            star_reference,
            strict=True,
            cfg=cfg,
        )
        selected_support = np.zeros(spatial_shape, dtype=bool)
        selected_weak_support = np.zeros(spatial_shape, dtype=bool)
        selected_bright_support = np.zeros(spatial_shape, dtype=bool)
        # Keep all newly raised wing samples at 90% of the current half-max
        # line.  This fixed margin prevents numeric jitter from enlarging the
        # connected half-max core while still lifting the visible outer PSF.
        halfmax_ceiling_fraction = 0.45
        display_ceiling = np.full(spatial_shape, np.inf, dtype=np.float32)
        display_target = np.full(spatial_shape, -np.inf, dtype=np.float32)
        source_visible_support = np.zeros(spatial_shape, dtype=bool)
        height, width = spatial_shape
        visible_peak = (
            _pixel_peak(visible_reference)
            if visible_reference is not None
            else None
        )
        for index in np.flatnonzero(selected):
            base_radius = int(math.ceil(float(source_fwhm[index]) / 2.0))
            visible_radius_max = stage9_scale_radius(
                visible_radius_max_nominal,
                star_reference,
                fwhm_px=source_fwhm[index],
                rounding="nearest",
                minimum=1,
            )
            radius_max = stage9_scale_radius(
                radius_max_nominal,
                star_reference,
                fwhm_px=source_fwhm[index],
                rounding="ceil",
                minimum=1,
            )
            search_radius = stage9_scale_radius(
                2,
                star_reference,
                fwhm_px=source_fwhm[index],
                rounding="nearest",
                minimum=1,
            )
            y = int(peak_y[index])
            x = int(peak_x[index])
            floor_only = bool(
                visible_reference is not None
                and selected_at_floor_fraction[index]
                and not selected_at_visible_fraction[index]
                and not selected_at_mid_fraction[index]
            )
            mid_selected = bool(
                visible_reference is not None
                and selected_at_mid_fraction[index]
            )
            visible_selected = bool(
                visible_reference is not None
                and selected_at_visible_fraction[index]
            )
            visible_measurement = None
            if visible_peak is not None:
                support_fraction = (
                    visible_floor
                    if visible_selected or selected_at_floor_fraction[index]
                    else visible_mid_fraction
                )
                visible_measurement = _measure_connected_peak_fraction_footprint(
                    visible_peak,
                    y,
                    x,
                    peak_fraction=support_fraction,
                    search_radius=search_radius,
                    patch_radius=visible_radius_max,
                    require_closed=False,
                )
            if (
                isinstance(visible_measurement, dict)
                and visible_measurement.get("status") == "ok"
                and abs(int(visible_measurement.get("offset_y", 99)))
                <= search_radius
                and abs(int(visible_measurement.get("offset_x", 99)))
                <= search_radius
            ):
                radius = visible_radius_max
            else:
                visible_measurement = None
                radius = max(
                    1,
                    min(
                        radius_max,
                        base_radius
                        + stage9_scale_radius(
                            1,
                            star_reference,
                            fwhm_px=source_fwhm[index],
                            rounding="ceil",
                        )
                        + stage9_scale_radius(
                            retry_pixels,
                            star_reference,
                            fwhm_px=source_fwhm[index],
                            rounding="ceil",
                        ),
                    ),
                )
            y0 = max(0, y - radius)
            y1 = min(height, y + radius + 1)
            x0 = max(0, x - radius)
            x1 = min(width, x + radius + 1)
            grid_y, grid_x = np.ogrid[y0:y1, x0:x1]
            disk = (grid_y - y) ** 2 + (grid_x - x) ** 2 <= radius * radius
            if visible_measurement is not None:
                my0, my1, mx0, mx1 = visible_measurement["patch_bounds"]
                source_component = np.asarray(
                    visible_measurement["component_mask"],
                    dtype=bool,
                )
                # The measurement patch may be clipped by the image edge.  Its
                # bounds are authoritative and need not equal the radius disk.
                visible_disk = np.zeros((y1 - y0, x1 - x0), dtype=bool)
                iy0 = max(y0, int(my0))
                iy1 = min(y1, int(my1))
                ix0 = max(x0, int(mx0))
                ix1 = min(x1, int(mx1))
                if iy1 > iy0 and ix1 > ix0:
                    visible_disk[
                        iy0 - y0 : iy1 - y0,
                        ix0 - x0 : ix1 - x0,
                    ] = source_component[
                        iy0 - int(my0) : iy1 - int(my0),
                        ix0 - int(mx0) : ix1 - int(mx0),
                    ]
                disk &= visible_disk
            selected_support[y0:y1, x0:x1] |= disk
            source_visible_support[y0:y1, x0:x1] |= disk
            group_support = (
                selected_weak_support
                if bool(weak_flags[index])
                else selected_bright_support
            )
            group_support[y0:y1, x0:x1] |= disk
            local_ceiling_fraction = (
                visible_selection_fraction if floor_only else halfmax_ceiling_fraction
            )
            local_ceiling = float(
                candidate_background[index]
                + local_ceiling_fraction * candidate_signal[index]
            )
            display_ceiling[y0:y1, x0:x1] = np.where(
                disk,
                np.minimum(
                    display_ceiling[y0:y1, x0:x1],
                    local_ceiling,
                ),
                display_ceiling[y0:y1, x0:x1],
            )
            if visible_measurement is not None and np.any(disk):
                local_source = visible_peak[y0:y1, x0:x1]
                raw_source_profile = (
                    local_source
                    - float(visible_measurement["background"])
                ) / max(float(visible_measurement["signal"]), 1e-7)
                source_profile = np.clip(
                    raw_source_profile,
                    0.0,
                    halfmax_ceiling_fraction,
                )
                local_target = (
                    float(candidate_background[index])
                    + source_profile * float(candidate_signal[index])
                )
                if visible_selected:
                    target_region = disk & (source_profile >= visible_floor)
                elif mid_selected:
                    target_region = disk & (
                        source_profile >= visible_mid_fraction
                    )
                    if bool(selected_at_floor_fraction[index]):
                        target_region |= (
                            disk
                            & (source_profile >= visible_floor)
                            & (raw_source_profile < visible_selection_fraction)
                        )
                else:
                    target_region = disk & (source_profile >= visible_floor)
                if floor_only:
                    # The 10% contour is already large enough.  Fill only the
                    # lower 5%--10% skirt so this recovery cannot grow it.
                    target_region &= (
                        raw_source_profile < visible_selection_fraction
                    )
                display_target[y0:y1, x0:x1] = np.where(
                    target_region,
                    np.maximum(
                        display_target[y0:y1, x0:x1],
                        local_target,
                    ),
                    display_target[y0:y1, x0:x1],
                )

        outer_wing = selected_support & ~np.asarray(strict_support, dtype=bool)
        denominator_floor = _bounded(
            getattr(cfg, "stage9_unscreen_denominator_floor", 0.08),
            0.08,
            0.02,
            0.25,
        )
        residual = original - starless
        denominator = 1.0 - starless
        reliable = (
            outer_wing
            & (_pixel_floor(denominator) >= denominator_floor)
            & (_pixel_floor(residual) >= -0.002)
            & (_pixel_peak(original) < 0.995)
            & (candidate_peak < display_ceiling - 1e-5)
        )
        if visible_reference is not None:
            reliable &= (
                source_visible_support
                & np.isfinite(display_target)
                & (candidate_peak < display_target - 1e-5)
            )
        reliable_count = int(np.count_nonzero(reliable))
        if reliable_count <= 0:
            raise ValueError("selective source-wing has no reliable matched pixels")
        raw_unscreen = np.clip(
            residual / np.maximum(denominator, denominator_floor),
            0.0,
            1.0,
        ).astype(np.float32, copy=False)
        feathered = np.array(stabilized, dtype=np.float32, copy=True)
        reliable_expanded = _expanded_spatial_mask(
            feathered,
            reliable.astype(np.float32),
        )
        proposed = np.minimum(strength * raw_unscreen, 1.0)
        # Convert the display-domain ceiling into a hue-preserving scalar cap
        # for the proposed layer using the actual Stage 8 Screen base.
        bright_intensity = _bounded(screen_intensity, 1.0, 0.10, 1.05)
        weak_intensity = max(
            bright_intensity,
            _bounded(
                getattr(cfg, "stage9_weak_star_screen_intensity_min", 0.55),
                0.55,
                0.10,
                1.05,
            ),
        )
        intensity_map = np.full(
            spatial_shape,
            bright_intensity,
            dtype=np.float32,
        )
        intensity_map[selected_weak_support] = weak_intensity
        proposed_rgb = _rgb_channels(proposed)
        base_rgb = _rgb_channels(actual_base)
        delivered_delta = (
            (1.0 - base_rgb)
            * intensity_map[np.newaxis, ...]
            * proposed_rgb
        )
        visible_boost_limited = np.zeros(spatial_shape, dtype=bool)
        if visible_reference is not None:
            delivered_peak = _pixel_peak(base_rgb + delivered_delta)
            boost = np.divide(
                display_target - _pixel_peak(base_rgb),
                np.maximum(delivered_peak - _pixel_peak(base_rgb), 1e-7),
                out=np.ones(spatial_shape, dtype=np.float32),
                where=reliable,
            )
            # The reference shape owns the target.  The 4x limit prevents a
            # noisy residual hue sample from turning into a bright halo; any
            # remaining shortfall is observable in the post-candidate audit.
            boost = np.clip(boost, 1.0, 4.0)
            visible_boost_limited = reliable & (boost >= 4.0 - 1e-6)
            proposed *= _expanded_spatial_mask(proposed, boost)
            proposed = np.minimum(proposed, 1.0)
            proposed_rgb = _rgb_channels(proposed)
            delivered_delta = (
                (1.0 - base_rgb)
                * intensity_map[np.newaxis, ...]
                * proposed_rgb
            )
        effective_display_ceiling = np.where(
            np.isfinite(display_target),
            np.minimum(display_ceiling, display_target),
            display_ceiling,
        )
        allowed_delta = effective_display_ceiling[np.newaxis, ...] - base_rgb
        channel_scale = np.divide(
            allowed_delta,
            delivered_delta,
            out=np.full_like(delivered_delta, np.inf, dtype=np.float32),
            where=delivered_delta > 1e-12,
        )
        hue_scale = np.clip(np.min(channel_scale, axis=0), 0.0, 1.0)
        proposed *= _expanded_spatial_mask(proposed, hue_scale)
        ceiling_limited = reliable & (hue_scale < 1.0 - 1e-6)
        ceiling_zeroed = reliable & (hue_scale <= 1e-7)
        target_layer = np.maximum(feathered, proposed)
        softened_target = feathered + soft_alpha * (target_layer - feathered)
        feathered = np.where(
            reliable_expanded > 0.0,
            softened_target,
            feathered,
        )
        support = current_support | reliable
        feathered *= _expanded_spatial_mask(
            feathered,
            support.astype(np.float32),
        )
        core_delta = _pixel_peak(np.abs(feathered - stabilized))[
            np.asarray(strict_support, dtype=bool)
        ]
        strict_core_change_max = (
            float(np.max(core_delta)) if core_delta.size else 0.0
        )
        if strict_core_change_max > 1e-7:
            raise ValueError(
                "selective source-wing changed the immutable strict core: "
                f"max={strict_core_change_max:.9f}"
            )
        changed_spatial = _pixel_peak(np.abs(feathered - stabilized)) > 1e-7
        changed_count = int(np.count_nonzero(changed_spatial))
        if changed_count <= 0:
            report.update(
                status="not_needed",
                reason_code="stage9_selective_source_wing_no_pixel_change",
                reliable_wing_pixel_count=reliable_count,
            )
            return None, None, report
        report.update(
            changed=True,
            selected_support_pixel_count=int(np.count_nonzero(selected_support)),
            eligible_outer_wing_pixel_count=int(np.count_nonzero(outer_wing)),
            reliable_wing_pixel_count=reliable_count,
            reliable_wing_ratio=float(
                reliable_count / max(1, int(np.count_nonzero(outer_wing)))
            ),
            changed_pixel_count=changed_count,
            changed_weak_pixel_count=int(
                np.count_nonzero(changed_spatial & selected_weak_support)
            ),
            changed_bright_pixel_count=int(
                np.count_nonzero(changed_spatial & selected_bright_support)
            ),
            support_pixel_count=int(np.count_nonzero(support)),
            support_ratio=float(np.mean(support)),
            denominator_floor=denominator_floor,
            halfmax_ceiling_fraction=halfmax_ceiling_fraction,
            halfmax_ceiling_relative_to_halfmax=2.0
            * halfmax_ceiling_fraction,
            strict_core_change_max=strict_core_change_max,
            delivered_screen_intensity=bright_intensity,
            delivered_weak_screen_intensity=weak_intensity,
            visible_target_pixel_count=int(
                np.count_nonzero(np.isfinite(display_target))
            ),
            source_reference_target_ceiling_applied=bool(
                visible_reference is not None
            ),
            visible_boost_limited_pixel_count=int(
                np.count_nonzero(visible_boost_limited)
            ),
            ceiling_limited_pixel_count=int(np.count_nonzero(ceiling_limited)),
            ceiling_zeroed_pixel_count=int(np.count_nonzero(ceiling_zeroed)),
            display_ceiling_min=float(
                np.min(effective_display_ceiling[selected_support])
            ),
            display_ceiling_median=float(
                np.median(effective_display_ceiling[selected_support])
            ),
        )
        return feathered.astype(np.float32, copy=False), support, report
    except (IndexError, KeyError, TypeError, ValueError, FloatingPointError) as error:
        report["reason"] = str(error)
        return None, None, report


def assess_unscreen_reference_fidelity(
    original_display: np.ndarray,
    starless_display: np.ndarray,
    stars: np.ndarray,
    *,
    intensity: float,
    support_mask: np.ndarray,
    alpha_mask: np.ndarray | None = None,
    weak_mask: np.ndarray | None = None,
    bright_mask: np.ndarray | None = None,
    weak_intensity: float | None = None,
) -> Dict[str, Any]:
    """Measure Screen roundtrip error against the common display-domain pair."""
    try:
        original = _normalized(np.asarray(original_display))
        starless = _normalized(np.asarray(starless_display))
        star_layer = _normalized(np.asarray(stars))
        if not (original.shape == starless.shape == star_layer.shape):
            raise ValueError("reference-fidelity image shapes do not match")
        support = np.asarray(support_mask, dtype=bool)
        if support.shape != _pixel_peak(original).shape:
            raise ValueError("reference-fidelity support shape does not match")
        if not np.any(support):
            raise ValueError("reference-fidelity support is empty")
        composed = screen_blend(
            starless,
            star_layer,
            intensity,
            alpha_mask=alpha_mask,
            weak_mask=weak_mask,
            bright_mask=bright_mask,
            weak_intensity=weak_intensity,
        )
        error_rgb = np.abs(_rgb_channels(composed) - _rgb_channels(original))
        samples = error_rgb[:, support]
        outside = ~support
        outside_change = _stage9_roundtrip_error_summary(
            starless,
            composed,
            mask=outside,
        )
        return {
            "schema": "starun.stage9-reference-fidelity.v1",
            "status": "ok",
            "composition_operator": "alpha_gated_screen",
            "premultiplied_alpha": False,
            "support_pixel_count": int(np.count_nonzero(support)),
            "support_rgb_mae": float(np.mean(samples)),
            "support_rgb_p95": float(np.percentile(samples, 95.0)),
            "support_rgb_max": float(np.max(samples)),
            "alpha_support_outside_change": outside_change,
        }
    except (IndexError, TypeError, ValueError, FloatingPointError) as error:
        return {
            "schema": "starun.stage9-reference-fidelity.v1",
            "status": "unavailable",
            "reason": str(error),
        }


def compare_unscreen_candidate(
    baseline: Dict[str, Any],
    unscreen: Dict[str, Any],
    cfg: Any,
) -> Dict[str, Any]:
    """Apply the fixed non-regression policy for an accepted Unscreen candidate."""
    result: Dict[str, Any] = {
        "schema": "starun.stage9-unscreen-comparison.v1",
        "policy": "fixed_unscreen_psf_competition_v2",
        "selected": False,
        "reason_code": "stage9_unscreen_candidate_rejected",
        "checks": {},
    }
    if not bool(baseline.get("accepted", False)):
        result.update(
            selected=bool(unscreen.get("accepted", False)),
            reason_code=(
                "stage9_unscreen_selected"
                if bool(unscreen.get("accepted", False))
                else "stage9_unscreen_candidate_rejected"
            ),
            rescue_without_baseline=True,
        )
        return result
    if not bool(unscreen.get("accepted", False)):
        return result

    baseline_fidelity = baseline.get("reference_fidelity") or {}
    unscreen_fidelity = unscreen.get("reference_fidelity") or {}
    try:
        baseline_mae = float(baseline_fidelity["support_rgb_mae"])
        unscreen_mae = float(unscreen_fidelity["support_rgb_mae"])
        if (
            baseline_fidelity.get("status") != "ok"
            or unscreen_fidelity.get("status") != "ok"
        ):
            raise ValueError("reference fidelity unavailable")
    except (KeyError, TypeError, ValueError):
        result["reason"] = "reference fidelity unavailable"
        return result

    absolute_improvement = baseline_mae - unscreen_mae
    relative_improvement = absolute_improvement / max(baseline_mae, 1e-12)
    absolute_min = _bounded(
        getattr(cfg, "stage9_unscreen_roundtrip_absolute_improvement_min", 0.005),
        0.005,
        0.0,
        0.05,
    )
    relative_min = _bounded(
        getattr(cfg, "stage9_unscreen_roundtrip_relative_improvement_min", 0.10),
        0.10,
        0.0,
        0.50,
    )
    result.update(
        baseline_support_rgb_mae=baseline_mae,
        unscreen_support_rgb_mae=unscreen_mae,
        absolute_improvement=absolute_improvement,
        relative_improvement=relative_improvement,
    )
    result["checks"]["absolute_improvement"] = absolute_improvement >= absolute_min
    result["checks"]["relative_improvement"] = relative_improvement >= relative_min

    def chroma_error(candidate: Dict[str, Any]) -> float:
        validation = candidate.get("star_color_validation") or {}
        return float((validation.get("metrics") or {})["median_chroma_error"])

    try:
        chroma_regression = chroma_error(unscreen) - chroma_error(baseline)
        chroma_limit = _bounded(
            getattr(cfg, "stage9_unscreen_chroma_regression_max", 0.02),
            0.02,
            0.0,
            0.10,
        )
        result["chroma_regression"] = chroma_regression
        result["checks"]["chroma_non_regression"] = (
            chroma_regression <= chroma_limit + 1e-12
        )
    except (KeyError, TypeError, ValueError):
        result["checks"]["chroma_non_regression"] = False
        result["chroma_regression"] = None

    recovery_limit = _bounded(
        getattr(cfg, "stage9_unscreen_recovery_regression_max", 0.02),
        0.02,
        0.0,
        0.10,
    )
    wing_limit = _bounded(
        getattr(cfg, "stage9_unscreen_wing_regression_max", 0.03),
        0.03,
        0.0,
        0.15,
    )
    baseline_metrics = baseline.get("metrics") or {}
    unscreen_metrics = unscreen.get("metrics") or {}
    comparison_metrics = (
        "weak_star_recovery_ratio",
        "star_recovery_ratio",
        "star_positive_delta_window_recovery_ratio",
        "star_wing_recovery_ratio",
    )
    for metric_name in comparison_metrics:
        try:
            baseline_value = baseline_metrics[metric_name]
            unscreen_value = unscreen_metrics[metric_name]
            regression = float(baseline_value) - float(unscreen_value)
            limit = wing_limit if metric_name == "star_wing_recovery_ratio" else recovery_limit
            result[f"{metric_name}_regression"] = regression
            result["checks"][f"{metric_name}_non_regression"] = (
                regression <= limit + 1e-12
            )
        except (KeyError, StopIteration, TypeError, ValueError):
            result["checks"][f"{metric_name}_non_regression"] = False

    try:
        baseline_fwhm_ratio = float(
            baseline_metrics["star_psf_fwhm_ratio_all"]
        )
        unscreen_fwhm_ratio = float(
            unscreen_metrics["star_psf_fwhm_ratio_all"]
        )
        fwhm_regression = abs(unscreen_fwhm_ratio - 1.0) - abs(
            baseline_fwhm_ratio - 1.0
        )
        fwhm_regression_max = _bounded(
            getattr(cfg, "stage9_unscreen_fwhm_regression_max", 0.05),
            0.05,
            0.0,
            0.25,
        )
        fwhm_min = _bounded(
            getattr(cfg, "stage9_psf_fwhm_ratio_min", 0.93),
            0.93,
            0.50,
            1.00,
        )
        fwhm_max = _bounded(
            getattr(cfg, "stage9_psf_fwhm_ratio_max", 1.10),
            1.10,
            1.00,
            1.50,
        )
        result.update(
            baseline_fwhm_ratio=baseline_fwhm_ratio,
            unscreen_fwhm_ratio=unscreen_fwhm_ratio,
            fwhm_regression=fwhm_regression,
        )
        result["checks"]["fwhm_in_range"] = bool(
            fwhm_min <= unscreen_fwhm_ratio <= fwhm_max
        )
        result["checks"]["fwhm_non_regression"] = bool(
            fwhm_regression <= fwhm_regression_max + 1e-12
        )
    except (KeyError, TypeError, ValueError):
        # PSF closure is intentionally unavailable for sparse fields and legacy
        # resumes.  Both candidates still traversed the same compatibility path.
        result["fwhm_regression"] = None
        result["checks"]["fwhm_reference_compatible"] = True

    def advisory_categories(candidate: Dict[str, Any]) -> set[str]:
        categories: set[str] = set()
        for advisory in candidate.get("advisories") or []:
            text = str(advisory).strip().lower()
            if text:
                categories.add(text.split()[0].split(":", 1)[0])
        return categories

    new_advisories = sorted(
        advisory_categories(unscreen) - advisory_categories(baseline)
    )
    result["new_advisory_categories"] = new_advisories
    result["checks"]["no_new_advisory_category"] = not new_advisories
    if all(bool(value) for value in result["checks"].values()):
        result.update(
            selected=True,
            reason_code="stage9_unscreen_selected",
        )
    else:
        result.update(
            reason_code="stage9_unscreen_no_material_improvement",
            failed_checks=[
                key for key, value in result["checks"].items() if not value
            ],
        )
    return result


def _asinh_sample(value: float, stretch: float, offset: float) -> float:
    value = float(value)
    stretch = max(1.0, float(stretch))
    offset = max(0.0, float(offset))
    if not all(math.isfinite(item) for item in (value, stretch, offset)):
        return 0.0
    if value <= offset or value <= 0.0:
        return 0.0
    denominator = value * math.asinh(stretch)
    if denominator <= 0.0:
        return 0.0
    return _bounded(
        (value - offset) * math.asinh(value * stretch) / denominator,
        0.0,
        0.0,
        1.0,
    )


def _solve_asinh_stretch(
    value: float,
    offset: float,
    target: float,
    stretch_max: float,
) -> float:
    low = 1.0
    high = max(low, float(stretch_max))
    target = _bounded(target, 0.22, 0.02, 0.95)
    if _asinh_sample(value, low, offset) >= target:
        return low
    if _asinh_sample(value, high, offset) <= target:
        return high
    for _ in range(48):
        middle = (low + high) * 0.5
        if _asinh_sample(value, middle, offset) < target:
            low = middle
        else:
            high = middle
    return high


def _stage9_starmask_output_targets(cfg: Any) -> Dict[str, float]:
    """Return the existing four light-stretch output limits in strict order."""
    faint = _bounded(
        getattr(cfg, "stage9_starmask_faint_target", 0.26),
        0.26,
        0.08,
        0.40,
    )
    peak = _bounded(
        getattr(cfg, "stage9_starmask_peak_target", 0.90),
        0.90,
        0.75,
        0.95,
    )
    mid = min(
        peak - 0.10,
        max(
            faint + 0.03,
            _bounded(
                getattr(cfg, "stage9_starmask_mid_target", 0.50),
                0.50,
                0.30,
                0.70,
            ),
        ),
    )
    bright = min(
        peak - 0.03,
        max(
            mid + 0.03,
            _bounded(
                getattr(cfg, "stage9_starmask_bright_target", 0.75),
                0.75,
                0.50,
                0.88,
            ),
        ),
    )
    return {
        "faint": float(faint),
        "mid": float(mid),
        "bright": float(bright),
        "peak": float(peak),
    }


def measure_starmask_output_profile(
    stretched_stars: np.ndarray,
    calibration: Dict[str, Any],
    *,
    source: str,
) -> Dict[str, Any]:
    """Measure the four frozen-star output anchors from an actual star layer."""
    targets = {
        name: float(
            calibration.get(
                f"{name}_target",
                (calibration.get("output_targets") or {}).get(name, 0.0),
            )
            or 0.0
        )
        for name in _STAGE9_STARMASK_OUTPUT_NAMES
    }
    result: Dict[str, Any] = {
        "schema": _STAGE9_STARMASK_OUTPUT_PROFILE_SCHEMA,
        "status": "unavailable",
        "accepted": False,
        "hard_failed": True,
        "source": str(source),
        "targets": targets,
        "tolerance": _STAGE9_STARMASK_OUTPUT_TOLERANCE,
        "actual": {},
        "exceeded_anchors": [],
    }
    try:
        output_peak = _pixel_peak(_normalized(np.asarray(stretched_stars)))
        if output_peak.ndim != 2:
            raise ValueError("starmask output peak map is not two-dimensional")
        profile_mode = str(
            calibration.get("output_profile_mode")
            or (
                "mixed_star_peak_percentiles"
                if calibration.get("mixed_star_field", False)
                else "ordinary_support_pixel_percentiles"
            )
        )
        if profile_mode == "mixed_star_peak_percentiles":
            catalog = calibration.get("_star_reference_catalog") or {}
            peak_y = np.asarray(catalog.get("_peak_y", ()), dtype=np.int64)
            peak_x = np.asarray(catalog.get("_peak_x", ()), dtype=np.int64)
            valid = (
                (peak_y >= 0)
                & (peak_x >= 0)
                & (peak_y < output_peak.shape[0])
                & (peak_x < output_peak.shape[1])
            )
            samples = output_peak[peak_y[valid], peak_x[valid]]
            percentiles = (40.0, 80.0, 90.0, 99.7)
        else:
            sample_mask = calibration.get("_output_profile_sample_mask")
            if sample_mask is None:
                raise ValueError("ordinary output-profile sample mask unavailable")
            sample_mask = np.asarray(sample_mask, dtype=bool)
            if sample_mask.shape != output_peak.shape:
                raise ValueError(
                    "ordinary output-profile sample mask shape mismatch"
                )
            samples = output_peak[sample_mask & np.isfinite(output_peak)]
            percentiles = (50.0, 75.0, 90.0, 99.7)
        samples = np.asarray(samples, dtype=np.float64)
        samples = samples[np.isfinite(samples)]
        if samples.size < 8:
            raise ValueError(
                f"starmask output-profile sample count {int(samples.size)} is below 8"
            )
        values = np.percentile(samples, percentiles)
        actual = {
            name: float(value)
            for name, value in zip(_STAGE9_STARMASK_OUTPUT_NAMES, values)
        }
        exceeded = [
            name
            for name in _STAGE9_STARMASK_OUTPUT_NAMES
            if actual[name] > targets[name] + _STAGE9_STARMASK_OUTPUT_TOLERANCE
        ]
        result.update(
            status="hard_failed" if exceeded else "ok",
            accepted=not exceeded,
            hard_failed=bool(exceeded),
            profile_mode=profile_mode,
            sample_count=int(samples.size),
            percentiles=[float(value) for value in percentiles],
            actual=actual,
            exceeded_anchors=exceeded,
            reason=(
                "stage9_starmask_output_target_exceeded: "
                + ", ".join(exceeded)
                if exceeded
                else ""
            ),
            reason_code=(
                "stage9_starmask_output_target_exceeded"
                if exceeded
                else ""
            ),
        )
    except (IndexError, TypeError, ValueError, FloatingPointError) as error:
        result["reason"] = str(error)
        result["reason_code"] = "stage9_starmask_output_profile_unavailable"
    return result


def _starmask_output_profile_gate(profile: Dict[str, Any]) -> Dict[str, Any]:
    accepted = bool(profile.get("accepted", False))
    return {
        "status": "ok" if accepted else "hard_failed",
        "accepted": accepted,
        "advisory": False,
        "hard_failed": not accepted,
        "actual": dict(profile.get("actual") or {}),
        "accepted_limit": dict(profile.get("targets") or {}),
        "tolerance": float(
            profile.get("tolerance", _STAGE9_STARMASK_OUTPUT_TOLERANCE)
        ),
        "exceeded_anchors": list(profile.get("exceeded_anchors") or []),
        "reason": str(profile.get("reason") or ""),
    }


def _compact_star_support(
    gray: np.ndarray,
    *,
    background: float,
    noise_sigma: float,
    strict: bool = False,
) -> Dict[str, Any]:
    """Find connected compact star cores and a narrow wing support."""
    if scipy_ndimage is None:
        return {"status": "unavailable", "reason": "scipy.ndimage unavailable"}
    finite = gray[np.isfinite(gray)]
    if finite.size < 64:
        return {"status": "unavailable", "reason": "insufficient finite pixels"}

    core_percentile = 99.85 if strict else 99.7
    noise_multiplier = 10.0 if strict else 8.0
    max_component_area = 256 if strict else 512
    max_component_span = 48 if strict else 64
    wing_iterations = 1 if strict else 3
    core_threshold = max(
        background + noise_multiplier * noise_sigma,
        float(np.percentile(finite, core_percentile)),
    )
    core_seed = np.asarray(gray > core_threshold, dtype=bool)
    labels, component_count = scipy_ndimage.label(
        core_seed,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    if component_count <= 0:
        return {"status": "unavailable", "reason": "no connected star cores"}

    areas = np.bincount(labels.reshape(-1), minlength=component_count + 1)
    keep_component = np.zeros(component_count + 1, dtype=bool)
    objects = scipy_ndimage.find_objects(labels)
    for index, bounds in enumerate(objects, start=1):
        if bounds is None:
            continue
        area = int(areas[index])
        height = int(bounds[0].stop - bounds[0].start)
        width = int(bounds[1].stop - bounds[1].start)
        longest = max(height, width)
        shortest = max(1, min(height, width))
        fill_ratio = area / max(1, height * width)
        if (
            1 <= area <= max_component_area
            and longest <= max_component_span
            and longest / shortest <= 4.0
            and fill_ratio >= 0.15
        ):
            keep_component[index] = True

    compact_core = keep_component[labels]
    kept_count = int(np.count_nonzero(keep_component))
    if kept_count <= 0 or int(np.count_nonzero(compact_core)) < 8:
        return {"status": "unavailable", "reason": "no compact connected star cores"}
    compact_support = scipy_ndimage.binary_dilation(
        compact_core,
        structure=np.ones((3, 3), dtype=bool),
        iterations=wing_iterations,
    )
    return {
        "status": "ok",
        "mask": compact_support,
        "support_mode": "strict_recovery" if strict else "normal",
        "core_threshold": float(core_threshold),
        "core_percentile": float(core_percentile),
        "noise_multiplier": float(noise_multiplier),
        "wing_iterations": int(wing_iterations),
        "component_count": int(component_count),
        "kept_component_count": kept_count,
        "core_coverage": float(np.mean(compact_core)),
        "support_coverage": float(np.mean(compact_support)),
    }


def build_star_reference_catalog(
    stars: np.ndarray,
    cfg: Any,
    *,
    background: float | None = None,
    noise_sigma: float | None = None,
    source_image: np.ndarray | None = None,
    spatial_scale: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a reliable component-count reference for weak/bright star recovery."""
    if scipy_ndimage is None:
        return {"status": "unavailable", "reason": "scipy.ndimage unavailable"}
    normalized = _normalized(np.asarray(stars))
    if normalized.ndim not in (2, 3):
        return {"status": "unavailable", "reason": "invalid starmask dimensions"}
    gray = _gray(normalized)
    finite = gray[np.isfinite(gray)]
    if finite.size < 64:
        return {"status": "unavailable", "reason": "insufficient finite pixels"}

    if background is None or noise_sigma is None:
        background = float(np.percentile(finite, 50.0))
        low_samples = finite[finite <= np.percentile(finite, 70.0)]
        if low_samples.size < 32:
            low_samples = finite
        mad = float(np.median(np.abs(low_samples - np.median(low_samples))))
        noise_sigma = max(1.4826 * mad, 1e-7)
    background = float(background)
    noise_sigma = max(float(noise_sigma), 1e-7)
    if source_image is not None:
        return _build_source_matched_star_catalog(
            normalized,
            source_image,
            cfg,
            background=background,
            noise_sigma=noise_sigma,
            spatial_scale=spatial_scale,
        )
    reference_sigma = _bounded(
        getattr(cfg, "stage9_star_reference_sigma", 5.0),
        5.0,
        3.0,
        8.0,
    )
    threshold = max(background + reference_sigma * noise_sigma, 1e-6)
    labels, component_count = scipy_ndimage.label(
        np.asarray(gray > threshold, dtype=bool),
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    if component_count <= 0:
        return {"status": "unavailable", "reason": "no 5-sigma star components"}

    areas = np.bincount(labels.reshape(-1), minlength=component_count + 1)
    objects = scipy_ndimage.find_objects(labels)
    keep_ids = []
    rejected_small_count = 0
    geometry_reference = {"stage9_spatial_scale": spatial_scale or {}}
    min_component_area = stage9_scale_area(
        _STAR_REFERENCE_MIN_COMPONENT_AREA,
        geometry_reference,
    )
    max_component_area = stage9_scale_area(512, geometry_reference)
    max_component_span = stage9_scale_radius(
        64,
        geometry_reference,
        rounding="nearest",
        minimum=1,
    )
    for index, bounds in enumerate(objects, start=1):
        if bounds is None:
            continue
        area = int(areas[index])
        if area < min_component_area:
            rejected_small_count += 1
            continue
        height = int(bounds[0].stop - bounds[0].start)
        width = int(bounds[1].stop - bounds[1].start)
        longest = max(height, width)
        shortest = max(1, min(height, width))
        fill_ratio = area / max(1, height * width)
        if (
            area <= max_component_area
            and longest <= max_component_span
            and longest / shortest <= 4.0
            and fill_ratio >= 0.15
        ):
            keep_ids.append(index)
    if not keep_ids:
        return {
            "status": "unavailable",
            "reason": "no reliable compact star components",
            "rejected_small_component_count": rejected_small_count,
        }

    component_ids = np.asarray(keep_ids, dtype=np.int32)
    component_peak_map = _pixel_peak(normalized)
    component_peaks = np.asarray(
        scipy_ndimage.maximum(
            component_peak_map,
            labels=labels,
            index=component_ids,
        ),
        dtype=np.float32,
    )
    peak_positions = scipy_ndimage.maximum_position(
        component_peak_map,
        labels=labels,
        index=component_ids,
    )
    peak_y = np.asarray([item[0] for item in peak_positions], dtype=np.int32)
    peak_x = np.asarray([item[1] for item in peak_positions], dtype=np.int32)
    weak_cutoff = float(np.percentile(component_peaks, 80.0))
    weak_flags = np.asarray(component_peaks <= weak_cutoff, dtype=bool)
    bright_flags = ~weak_flags
    weak_count = int(np.count_nonzero(weak_flags))
    bright_count = int(np.count_nonzero(bright_flags))
    weak_peak_median = (
        float(np.median(component_peaks[weak_flags])) if weak_count else 0.0
    )
    bright_peak_median = (
        float(np.median(component_peaks[bright_flags])) if bright_count else 0.0
    )
    peak_ratio = bright_peak_median / max(weak_peak_median, 1e-12)
    weak_count_min = int(
        _bounded(
            getattr(cfg, "stage9_mixed_star_weak_count_min", 20),
            20,
            4,
            1000,
        )
    )
    bright_count_min = int(
        _bounded(
            getattr(cfg, "stage9_mixed_star_bright_count_min", 3),
            3,
            1,
            100,
        )
    )
    mixed_ratio_min = _bounded(
        getattr(cfg, "stage9_mixed_star_peak_ratio_min", 4.0),
        4.0,
        2.0,
        20.0,
    )
    mixed_star_field = bool(
        weak_count >= weak_count_min
        and bright_count >= bright_count_min
        and peak_ratio >= mixed_ratio_min
    )

    weak_lookup = np.zeros(component_count + 1, dtype=bool)
    bright_lookup = np.zeros(component_count + 1, dtype=bool)
    weak_lookup[component_ids[weak_flags]] = True
    bright_lookup[component_ids[bright_flags]] = True
    peak_by_label = np.zeros(component_count + 1, dtype=np.float32)
    peak_by_label[component_ids] = component_peaks
    return {
        "status": "ok",
        "method": "5sigma_compact_component_catalog",
        "background": background,
        "noise_sigma": noise_sigma,
        "reference_sigma": reference_sigma,
        "reference_threshold": threshold,
        "min_component_area": min_component_area,
        "min_component_area_nominal": _STAR_REFERENCE_MIN_COMPONENT_AREA,
        "max_component_area": max_component_area,
        "max_component_area_nominal": 512,
        "max_component_span": max_component_span,
        "max_component_span_nominal": 64,
        "stage9_spatial_scale": dict(spatial_scale or {}),
        "component_count": int(component_ids.size),
        "weak_component_count": weak_count,
        "bright_component_count": bright_count,
        "rejected_small_component_count": rejected_small_count,
        "weak_peak_cutoff": weak_cutoff,
        "weak_peak_median": weak_peak_median,
        "bright_peak_median": bright_peak_median,
        "bright_to_weak_peak_ratio": float(peak_ratio),
        "mixed_peak_ratio_min": mixed_ratio_min,
        "mixed_weak_count_min": weak_count_min,
        "mixed_bright_count_min": bright_count_min,
        "mixed_star_field": mixed_star_field,
        "_labels": labels,
        "_component_ids": component_ids,
        "_component_peaks": component_peaks,
        "_peak_y": peak_y,
        "_peak_x": peak_x,
        "_weak_flags": weak_flags,
        "_weak_lookup": weak_lookup,
        "_bright_lookup": bright_lookup,
        "_peak_by_label": peak_by_label,
        "_weak_core_mask": weak_lookup[labels],
        "_bright_core_mask": bright_lookup[labels],
    }


def _measure_connected_halfmax_fwhm(
    image: np.ndarray,
    center_y: int,
    center_x: int,
    *,
    search_radius: int = 2,
    patch_radius: int = 6,
) -> Dict[str, Any]:
    """Measure one star with an annular background and connected half-max area."""
    if scipy_ndimage is None:
        return {"status": "unavailable", "reason": "scipy.ndimage unavailable"}
    sample = np.asarray(image, dtype=np.float32)
    if sample.ndim != 2:
        return {"status": "unavailable", "reason": "invalid PSF image"}
    height, width = sample.shape
    y = int(center_y)
    x = int(center_x)
    if not (0 <= y < height and 0 <= x < width):
        return {"status": "unavailable", "reason": "PSF center outside image"}

    search = max(0, int(search_radius))
    sy0 = max(0, y - search)
    sy1 = min(height, y + search + 1)
    sx0 = max(0, x - search)
    sx1 = min(width, x + search + 1)
    search_window = sample[sy0:sy1, sx0:sx1]
    if search_window.size == 0 or not np.all(np.isfinite(search_window)):
        return {"status": "unavailable", "reason": "empty PSF search window"}
    local_y, local_x = np.unravel_index(
        int(np.argmax(search_window)),
        search_window.shape,
    )
    peak_y = sy0 + int(local_y)
    peak_x = sx0 + int(local_x)

    radius = max(1, int(patch_radius))
    y0 = max(0, peak_y - radius)
    y1 = min(height, peak_y + radius + 1)
    x0 = max(0, peak_x - radius)
    x1 = min(width, peak_x + radius + 1)
    patch = sample[y0:y1, x0:x1]
    if not np.all(np.isfinite(patch)):
        return {"status": "unavailable", "reason": "non-finite PSF patch"}
    grid_y, grid_x = np.ogrid[y0:y1, x0:x1]
    distance = np.sqrt((grid_y - peak_y) ** 2 + (grid_x - peak_x) ** 2)
    annulus = (distance >= max(1.0, radius * (2.0 / 3.0))) & (
        distance <= radius
    )
    ring_values = patch[annulus]
    if ring_values.size < 8:
        border = np.zeros_like(patch, dtype=bool)
        border[[0, -1], :] = True
        border[:, [0, -1]] = True
        ring_values = patch[border]
    background = float(np.median(ring_values)) if ring_values.size else 0.0
    ring_mad = (
        float(np.median(np.abs(ring_values - background)))
        if ring_values.size
        else 0.0
    )
    noise_sigma = max(1.4826 * ring_mad, 1e-7)
    peak = float(sample[peak_y, peak_x])
    signal = peak - background
    if not math.isfinite(signal) or signal <= max(4.0 * noise_sigma, 1e-6):
        return {
            "status": "unavailable",
            "reason": "independent star signal is insufficient",
            "signal": signal,
            "noise_sigma": noise_sigma,
        }

    half_max = background + 0.5 * signal
    labels, _component_count = scipy_ndimage.label(
        patch >= half_max,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    patch_peak_y = peak_y - y0
    patch_peak_x = peak_x - x0
    label_id = int(labels[patch_peak_y, patch_peak_x])
    if label_id <= 0:
        return {"status": "unavailable", "reason": "half-max core is absent"}
    component = labels == label_id
    if (
        np.any(component[0, :])
        or np.any(component[-1, :])
        or np.any(component[:, 0])
        or np.any(component[:, -1])
    ):
        return {
            "status": "unavailable",
            "reason": "half-max core touches measurement boundary",
        }
    area = int(np.count_nonzero(component))
    if area <= 0:
        return {"status": "unavailable", "reason": "half-max area is empty"}
    fwhm = 2.0 * math.sqrt(area / math.pi)
    return {
        "status": "ok",
        "center_y": peak_y,
        "center_x": peak_x,
        "offset_y": int(peak_y - y),
        "offset_x": int(peak_x - x),
        "background": background,
        "noise_sigma": noise_sigma,
        "peak": peak,
        "signal": signal,
        "half_max_area": area,
        "fwhm_px": float(fwhm),
        "saturated": bool(peak >= 0.995),
    }


def _measure_connected_peak_fraction_footprint(
    image: np.ndarray,
    center_y: int,
    center_x: int,
    *,
    peak_fraction: float,
    search_radius: int = 2,
    patch_radius: int = 10,
    require_closed: bool = True,
) -> Dict[str, Any]:
    """Measure a locally background-subtracted connected low-light footprint.

    This is a display-morphology measurement, not aperture photometry.  It is
    intentionally separate from the 50% FWHM contract so a rendering can keep
    a stable core while recovering the 5%--10% PSF wing seen in a linked
    autostretch of the same source.
    """
    if scipy_ndimage is None:
        return {"status": "unavailable", "reason": "scipy.ndimage unavailable"}
    sample = np.asarray(image, dtype=np.float32)
    fraction = _bounded(peak_fraction, 0.10, 0.01, 0.49)
    if sample.ndim != 2:
        return {"status": "unavailable", "reason": "invalid footprint image"}
    height, width = sample.shape
    y = int(center_y)
    x = int(center_x)
    if not (0 <= y < height and 0 <= x < width):
        return {"status": "unavailable", "reason": "footprint center outside image"}

    search = max(0, int(search_radius))
    sy0 = max(0, y - search)
    sy1 = min(height, y + search + 1)
    sx0 = max(0, x - search)
    sx1 = min(width, x + search + 1)
    search_window = sample[sy0:sy1, sx0:sx1]
    if search_window.size == 0 or not np.all(np.isfinite(search_window)):
        return {"status": "unavailable", "reason": "empty footprint search window"}
    local_y, local_x = np.unravel_index(
        int(np.argmax(search_window)),
        search_window.shape,
    )
    peak_y = sy0 + int(local_y)
    peak_x = sx0 + int(local_x)

    radius = max(1, int(patch_radius))
    y0 = max(0, peak_y - radius)
    y1 = min(height, peak_y + radius + 1)
    x0 = max(0, peak_x - radius)
    x1 = min(width, peak_x + radius + 1)
    patch = sample[y0:y1, x0:x1]
    if not np.all(np.isfinite(patch)):
        return {"status": "unavailable", "reason": "non-finite footprint patch"}
    grid_y, grid_x = np.ogrid[y0:y1, x0:x1]
    distance = np.sqrt((grid_y - peak_y) ** 2 + (grid_x - peak_x) ** 2)
    annulus = (distance >= max(1.0, radius * 0.8)) & (distance <= radius)
    ring_values = patch[annulus]
    if ring_values.size < 8:
        border = np.zeros_like(patch, dtype=bool)
        border[[0, -1], :] = True
        border[:, [0, -1]] = True
        ring_values = patch[border]
    background = float(np.median(ring_values)) if ring_values.size else 0.0
    ring_mad = (
        float(np.median(np.abs(ring_values - background)))
        if ring_values.size
        else 0.0
    )
    noise_sigma = max(1.4826 * ring_mad, 1e-7)
    peak = float(sample[peak_y, peak_x])
    signal = peak - background
    if not math.isfinite(signal) or signal <= max(4.0 * noise_sigma, 1e-6):
        return {
            "status": "unavailable",
            "reason": "independent footprint signal is insufficient",
            "signal": signal,
            "noise_sigma": noise_sigma,
        }
    threshold = background + fraction * signal
    labels, _component_count = scipy_ndimage.label(
        patch >= threshold,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    patch_peak_y = peak_y - y0
    patch_peak_x = peak_x - x0
    label_id = int(labels[patch_peak_y, patch_peak_x])
    if label_id <= 0:
        return {"status": "unavailable", "reason": "footprint is absent"}
    component = labels == label_id
    boundary_touched = bool(
        np.any(component[0, :])
        or np.any(component[-1, :])
        or np.any(component[:, 0])
        or np.any(component[:, -1])
    )
    if require_closed and boundary_touched:
        return {
            "status": "unavailable",
            "reason": "footprint touches measurement boundary",
        }
    area = int(np.count_nonzero(component))
    if area <= 0:
        return {"status": "unavailable", "reason": "footprint area is empty"}
    equivalent_diameter = 2.0 * math.sqrt(area / math.pi)
    return {
        "status": "ok",
        "center_y": peak_y,
        "center_x": peak_x,
        "offset_y": int(peak_y - y),
        "offset_x": int(peak_x - x),
        "background": background,
        "noise_sigma": noise_sigma,
        "peak": peak,
        "signal": signal,
        "peak_fraction": fraction,
        "threshold": threshold,
        "area_px": area,
        "equivalent_diameter_px": float(equivalent_diameter),
        "boundary_touched": boundary_touched,
        "component_mask": component,
        "patch_bounds": (y0, y1, x0, x1),
    }


def assess_stage9_visible_wing_closure(
    candidate_display: np.ndarray,
    visible_wing_reference: np.ndarray,
    star_reference: Dict[str, Any],
    cfg: Any,
) -> Dict[str, Any]:
    """Audit 5%/10% display-visible same-star footprints against autostretch."""
    report: Dict[str, Any] = {
        "schema": "starun.stage9-visible-wing-closure.v1",
        "status": "unavailable",
        "available": False,
        "scientific_photometry_claim": False,
        "hard_gate": False,
        "semantics": (
            "local_background_subtracted_connected_peak_fraction_footprint"
        ),
    }
    try:
        candidate_peak = _pixel_peak(_normalized(np.asarray(candidate_display)))
        source_peak = _pixel_peak(
            _normalized(np.asarray(visible_wing_reference))
        )
        if candidate_peak.shape != source_peak.shape:
            raise ValueError("visible-wing audit shape mismatch")
        valid = np.asarray(
            star_reference.get("_psf_valid_flags", ()), dtype=bool
        )
        saturated = np.asarray(
            star_reference.get("_psf_saturated_flags", ()), dtype=bool
        )
        peak_y = np.asarray(star_reference.get("_peak_y", ()), dtype=np.int32)
        peak_x = np.asarray(star_reference.get("_peak_x", ()), dtype=np.int32)
        count = int(valid.size)
        if not (
            count > 0
            and saturated.size == count
            and peak_y.size == count
            and peak_x.size == count
        ):
            raise ValueError("visible-wing audit reference arrays are incomplete")
        ordinary = valid & ~saturated
        nominal_radius = int(
            _bounded(
                getattr(
                    cfg,
                    "stage9_source_autostretch_wing_radius_max",
                    10,
                ),
                10,
                6,
                16,
            )
        )
        fractions = tuple(
            dict.fromkeys(
                (
                    _bounded(
                        getattr(
                            cfg,
                            "stage9_source_autostretch_wing_floor_fraction",
                            0.05,
                        ),
                        0.05,
                        0.03,
                        0.15,
                    ),
                    0.10,
                    0.25,
                )
            )
        )
        fraction_reports: Dict[str, Any] = {}
        measurement_radii = []
        for fraction in fractions:
            ratios: list[float] = []
            for index in np.flatnonzero(ordinary):
                per_star_fwhm = float(
                    np.asarray(
                        star_reference.get("_stage9_spatial_fwhm_px", ()),
                        dtype=np.float32,
                    )[index]
                ) if np.asarray(
                    star_reference.get("_stage9_spatial_fwhm_px", ())
                ).size == count else None
                radius = stage9_scale_radius(
                    nominal_radius,
                    star_reference,
                    fwhm_px=per_star_fwhm,
                    rounding="nearest",
                    minimum=1,
                )
                search_radius = stage9_scale_radius(
                    2,
                    star_reference,
                    fwhm_px=per_star_fwhm,
                    rounding="nearest",
                    minimum=1,
                )
                measurement_radii.append(radius)
                source_measurement = _measure_connected_peak_fraction_footprint(
                    source_peak,
                    int(peak_y[index]),
                    int(peak_x[index]),
                    peak_fraction=fraction,
                    search_radius=search_radius,
                    patch_radius=radius,
                )
                candidate_measurement = _measure_connected_peak_fraction_footprint(
                    candidate_peak,
                    int(peak_y[index]),
                    int(peak_x[index]),
                    peak_fraction=fraction,
                    search_radius=search_radius,
                    patch_radius=radius,
                )
                if (
                    source_measurement.get("status") != "ok"
                    or candidate_measurement.get("status") != "ok"
                ):
                    continue
                if any(
                    abs(int(measurement.get(offset, 99))) > search_radius
                    for measurement in (source_measurement, candidate_measurement)
                    for offset in ("offset_y", "offset_x")
                ):
                    continue
                source_diameter = float(
                    source_measurement["equivalent_diameter_px"]
                )
                candidate_diameter = float(
                    candidate_measurement["equivalent_diameter_px"]
                )
                if source_diameter > 0.0 and candidate_diameter > 0.0:
                    ratios.append(candidate_diameter / source_diameter)
            key = f"{fraction:.2f}"
            if ratios:
                values = np.asarray(ratios, dtype=np.float32)
                fraction_reports[key] = {
                    "status": "measured",
                    "peak_fraction": float(fraction),
                    "sample_count": int(values.size),
                    "diameter_ratio_p25": float(np.percentile(values, 25.0)),
                    "diameter_ratio_median": float(np.median(values)),
                    "diameter_ratio_p75": float(np.percentile(values, 75.0)),
                    "candidate_smaller_ratio": float(np.mean(values < 1.0)),
                }
            else:
                fraction_reports[key] = {
                    "status": "unavailable",
                    "peak_fraction": float(fraction),
                    "sample_count": 0,
                    "reason": "no jointly measurable isolated ordinary stars",
                }
        measured = [
            item
            for item in fraction_reports.values()
            if item.get("status") == "measured"
        ]
        report.update(
            status="measured" if measured else "unavailable",
            available=bool(measured),
            ordinary_reference_sample_count=int(np.count_nonzero(ordinary)),
            measurement_radius_px=nominal_radius,
            measurement_radius_px_nominal=nominal_radius,
            measurement_radius_px_effective=stage9_effective_pixel_stats(
                measurement_radii
            ),
            fractions=fraction_reports,
        )
        if not measured:
            report["reason"] = "no visible-wing fraction was jointly measurable"
        return report
    except (IndexError, KeyError, TypeError, ValueError, FloatingPointError) as error:
        report["reason"] = str(error)
        return report


def enrich_star_reference_with_display_psf(
    catalog: Dict[str, Any],
    original_display: np.ndarray,
    cfg: Any,
) -> Dict[str, Any]:
    """Confirm catalog stars in matched display-domain O and freeze PSF data."""
    if not isinstance(catalog, dict) or catalog.get("status") != "ok":
        return catalog
    source_y = np.asarray(
        catalog.get("_source_peak_y", catalog.get("_peak_y", ())),
        dtype=np.int32,
    )
    source_x = np.asarray(
        catalog.get("_source_peak_x", catalog.get("_peak_x", ())),
        dtype=np.int32,
    )
    layer_y = np.asarray(catalog.get("_peak_y", ()), dtype=np.int32)
    layer_x = np.asarray(catalog.get("_peak_x", ()), dtype=np.int32)
    weak_flags = np.asarray(catalog.get("_weak_flags", ()), dtype=bool)
    count = int(source_y.size)
    if not (
        count > 0
        and source_x.size == count
        and layer_y.size == count
        and layer_x.size == count
        and weak_flags.size == count
    ):
        catalog.update(
            psf_reference_status="unavailable",
            psf_reference_reason="catalog coordinates are incomplete",
        )
        return catalog
    peak_image = _pixel_peak(_normalized(np.asarray(original_display)))
    if peak_image.ndim != 2:
        catalog.update(
            psf_reference_status="unavailable",
            psf_reference_reason="matched-domain reference is not spatial",
        )
        return catalog

    fwhm = np.full(count, np.nan, dtype=np.float32)
    confirmed_y = np.full(count, -1, dtype=np.int32)
    confirmed_x = np.full(count, -1, dtype=np.int32)
    confirmed = np.zeros(count, dtype=bool)
    saturated = np.zeros(count, dtype=bool)
    measurements = []
    for index, (center_y, center_x) in enumerate(zip(source_y, source_x)):
        source_fwhm_hint = np.asarray(
            catalog.get("_stage9_spatial_fwhm_px", ()),
            dtype=np.float32,
        )
        fwhm_hint = (
            float(source_fwhm_hint[index])
            if source_fwhm_hint.size == count
            else None
        )
        search_radius = stage9_scale_radius(
            2,
            catalog,
            fwhm_px=fwhm_hint,
            rounding="nearest",
            minimum=1,
        )
        patch_radius = stage9_scale_radius(
            6,
            catalog,
            fwhm_px=fwhm_hint,
            rounding="nearest",
            minimum=1,
        )
        measurement = _measure_connected_halfmax_fwhm(
            peak_image,
            int(center_y),
            int(center_x),
            search_radius=search_radius,
            patch_radius=patch_radius,
        )
        measurements.append(measurement)
        if measurement.get("status") != "ok":
            continue
        offset_ok = bool(
            abs(int(measurement["offset_y"])) <= search_radius
            and abs(int(measurement["offset_x"])) <= search_radius
        )
        if not offset_ok:
            continue
        fwhm[index] = float(measurement["fwhm_px"])
        confirmed_y[index] = int(measurement["center_y"])
        confirmed_x[index] = int(measurement["center_x"])
        saturated[index] = bool(measurement.get("saturated", False))
        confirmed[index] = True

    isolated = confirmed.copy()
    if scipy_spatial is not None and int(np.count_nonzero(confirmed)) >= 2:
        coords = np.column_stack((source_y[confirmed], source_x[confirmed]))
        distances, _indices = scipy_spatial.cKDTree(coords).query(coords, k=2)
        nearest = np.asarray(distances[:, 1], dtype=np.float32)
        confirmed_indices = np.flatnonzero(confirmed)
        required_distance = np.maximum(
            np.asarray(
                [
                    stage9_scale_distance(
                        6.0,
                        catalog,
                        fwhm_px=fwhm[index],
                    )
                    for index in confirmed_indices
                ],
                dtype=np.float32,
            ),
            2.5 * fwhm[confirmed_indices],
        )
        isolated[confirmed_indices] = nearest >= required_distance

    valid = confirmed & isolated & np.isfinite(fwhm) & (fwhm > 0.0)
    ordinary_valid = valid & ~saturated
    min_count = int(
        _bounded(
            getattr(cfg, "stage9_psf_min_sample_count", 16),
            16,
            4,
            256,
        )
    )
    valid_count = int(np.count_nonzero(valid))
    ordinary_count = int(np.count_nonzero(ordinary_valid))
    weak_count = int(np.count_nonzero(ordinary_valid & weak_flags))
    bright_count = int(np.count_nonzero(ordinary_valid & ~weak_flags))
    if ordinary_count >= min_count:
        reference_status = "ready"
        reference_reason = ""
    elif valid_count > 0:
        reference_status = "partial"
        reference_reason = "insufficient isolated unsaturated matched stars"
    else:
        reference_status = "unavailable"
        reference_reason = "insufficient isolated matched stars"
    catalog.update(
        psf_reference_status=reference_status,
        psf_reference_reason=reference_reason,
        psf_confirmation_method="trusted_starmask_candidates_confirmed_in_mtf_O",
        psf_confirmation_radius_px=2,
        psf_confirmation_radius_px_nominal=2,
        psf_sample_count=valid_count,
        psf_ordinary_sample_count=ordinary_count,
        psf_min_sample_count=min_count,
        psf_weak_sample_count=weak_count,
        psf_bright_sample_count=bright_count,
        psf_saturated_sample_count=int(np.count_nonzero(valid & saturated)),
        display_source_fwhm_median_px=(
            float(np.median(fwhm[valid])) if valid_count else None
        ),
    )
    catalog["_display_source_fwhm_px"] = fwhm
    catalog["_display_source_halfmax_area_px"] = np.asarray(
        [
            float(item.get("half_max_area", np.nan))
            if isinstance(item, dict) and item.get("status") == "ok"
            else np.nan
            for item in measurements
        ],
        dtype=np.float32,
    )
    catalog["_psf_valid_flags"] = valid
    catalog["_psf_confirmed_flags"] = confirmed
    catalog["_psf_isolated_flags"] = isolated
    catalog["_psf_saturated_flags"] = saturated
    catalog["_display_source_peak_y"] = confirmed_y
    catalog["_display_source_peak_x"] = confirmed_x
    catalog["_psf_measurements"] = measurements
    return catalog


def build_display_confirmed_starmask_catalog(
    stars: np.ndarray,
    original_display: np.ndarray,
    cfg: Any,
    *,
    spatial_scale: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build the normal Stage 9 catalog from starmask candidates confirmed in O.

    Candidate membership is frozen from the trusted starmask.  Contamination
    retries only raise the independent O-domain confirmation sigma; they never
    increase the legacy global-detail percentile and silently discard weak-star
    candidates.
    """
    if scipy_ndimage is None:
        return {"status": "unavailable", "reason": "scipy.ndimage unavailable"}
    normalized_stars = _normalized(np.asarray(stars))
    display = _normalized(np.asarray(original_display))
    if normalized_stars.ndim not in (2, 3) or display.ndim not in (2, 3):
        return {"status": "unavailable", "reason": "invalid star/display dimensions"}
    star_gray = _gray(normalized_stars)
    display_peak = _pixel_peak(display)
    if star_gray.shape != display_peak.shape:
        return {
            "status": "unavailable",
            "reason": (
                "display/starmask shape mismatch: "
                f"display={display_peak.shape}, starmask={star_gray.shape}"
            ),
        }
    finite = star_gray[np.isfinite(star_gray)]
    if finite.size < 64:
        return {"status": "unavailable", "reason": "insufficient finite starmask pixels"}

    background = float(np.percentile(finite, 50.0))
    low_samples = finite[finite <= np.percentile(finite, 70.0)]
    if low_samples.size < 32:
        low_samples = finite
    mad = float(np.median(np.abs(low_samples - np.median(low_samples))))
    noise_sigma = max(1.4826 * mad, 1e-7)
    reference_sigma = _bounded(
        getattr(cfg, "stage9_star_reference_sigma", 5.0),
        5.0,
        3.0,
        8.0,
    )
    threshold = max(background + reference_sigma * noise_sigma, 1e-6)
    labels, component_count = scipy_ndimage.label(
        np.asarray(star_gray > threshold, dtype=bool),
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    if component_count <= 0:
        return {"status": "unavailable", "reason": "no trusted starmask components"}

    areas = np.bincount(labels.reshape(-1), minlength=component_count + 1)
    candidate_ids: List[int] = []
    geometry_reference = {"stage9_spatial_scale": spatial_scale or {}}
    component_min_area = stage9_scale_area(1, geometry_reference)
    component_max_area = stage9_scale_area(512, geometry_reference)
    component_span_max = stage9_scale_radius(
        64,
        geometry_reference,
        rounding="nearest",
        minimum=1,
    )
    for component_id, bounds in enumerate(
        scipy_ndimage.find_objects(labels),
        start=1,
    ):
        if bounds is None:
            continue
        area = int(areas[component_id])
        height = int(bounds[0].stop - bounds[0].start)
        width = int(bounds[1].stop - bounds[1].start)
        longest = max(height, width)
        shortest = max(1, min(height, width))
        fill_ratio = area / max(1, height * width)
        if (
            component_min_area <= area <= component_max_area
            and longest <= component_span_max
            and longest / shortest <= 4.0
            and fill_ratio >= 0.15
        ):
            candidate_ids.append(component_id)
    if not candidate_ids:
        return {
            "status": "unavailable",
            "reason": "no compact trusted starmask candidates",
        }

    candidate_ids_array = np.asarray(candidate_ids, dtype=np.int32)
    star_peak = _pixel_peak(normalized_stars)
    positions = scipy_ndimage.maximum_position(
        star_peak,
        labels=labels,
        index=candidate_ids_array,
    )
    layer_y = np.asarray([item[0] for item in positions], dtype=np.int32)
    layer_x = np.asarray([item[1] for item in positions], dtype=np.int32)
    candidate_peaks = np.asarray(
        scipy_ndimage.maximum(
            star_peak,
            labels=labels,
            index=candidate_ids_array,
        ),
        dtype=np.float32,
    )
    measurement_search_radius = stage9_scale_radius(
        2,
        geometry_reference,
        rounding="nearest",
        minimum=1,
    )
    measurement_patch_radius = stage9_scale_radius(
        6,
        geometry_reference,
        rounding="nearest",
        minimum=1,
    )
    halfmax_area_max = stage9_scale_area(128, geometry_reference)
    fwhm_max = stage9_scale_distance(12.0, geometry_reference)
    measurements = [
        _measure_connected_halfmax_fwhm(
            display_peak,
            int(center_y),
            int(center_x),
            search_radius=measurement_search_radius,
            patch_radius=measurement_patch_radius,
        )
        for center_y, center_x in zip(layer_y, layer_x)
    ]
    measurement_ok = np.asarray(
        [
            measurement.get("status") == "ok"
            and abs(int(measurement.get("offset_y", 99)))
            <= measurement_search_radius
            and abs(int(measurement.get("offset_x", 99)))
            <= measurement_search_radius
            and 0 < int(measurement.get("half_max_area", 0)) <= halfmax_area_max
            and 0.0 < float(measurement.get("fwhm_px", 0.0)) <= fwhm_max
            for measurement in measurements
        ],
        dtype=bool,
    )

    # Multiple tiny mask fragments may point at the same O-domain maximum. Keep
    # the strongest trusted layer candidate so each physical star is counted once.
    best_by_source_peak: Dict[Tuple[int, int], int] = {}
    for index in np.flatnonzero(measurement_ok):
        measurement = measurements[int(index)]
        key = (
            int(measurement.get("center_y", -1)),
            int(measurement.get("center_x", -1)),
        )
        previous = best_by_source_peak.get(key)
        if previous is None or float(candidate_peaks[index]) > float(
            candidate_peaks[previous]
        ):
            if previous is not None:
                measurement_ok[previous] = False
            best_by_source_peak[key] = int(index)
        else:
            measurement_ok[index] = False

    density_max = _bounded(
        getattr(cfg, "stage9_source_component_density_max", 2500.0),
        2500.0,
        500.0,
        10000.0,
    )
    single_pixel_max = _bounded(
        getattr(cfg, "stage9_source_single_pixel_ratio_max", 0.20),
        0.20,
        0.10,
        0.90,
    )
    megapixels = max(float(star_gray.size) / 1_000_000.0, 1e-6)
    confirmation_sigmas = [reference_sigma]
    retry_sigma = math.ceil((reference_sigma + 0.01) * 2.0) / 2.0
    while retry_sigma <= 8.0:
        confirmation_sigmas.append(retry_sigma)
        retry_sigma += 0.5

    attempts: List[Dict[str, Any]] = []
    selected = None
    last_contaminated = None
    component_areas = areas[candidate_ids_array].astype(np.int32, copy=False)
    for confirmation_sigma in confirmation_sigmas:
        confirmed = measurement_ok.copy()
        for index in np.flatnonzero(confirmed):
            measurement = measurements[int(index)]
            confirmed[index] = float(measurement.get("signal", 0.0)) >= (
                confirmation_sigma
                * max(float(measurement.get("noise_sigma", 0.0)), 1e-7)
            )
        confirmed_count = int(np.count_nonzero(confirmed))
        confirmed_areas = component_areas[confirmed]
        density = float(confirmed_count / megapixels)
        single_pixel_ratio = (
            float(np.mean(confirmed_areas == 1))
            if confirmed_areas.size
            else 0.0
        )
        density_exceeded = density > density_max
        single_exceeded = single_pixel_ratio > single_pixel_max
        density_contamination = bool(density_exceeded and single_pixel_ratio > 0.10)
        contamination = bool(density_contamination or single_exceeded)
        attempt = {
            # Stable compatibility fields: percentile is diagnostic-only and
            # deliberately identical for every confirmation retry.
            "percentile": float(
                _bounded(
                    getattr(cfg, "stage9_source_star_detail_percentile", 98.0),
                    98.0,
                    97.0,
                    99.5,
                )
            ),
            "reference_sigma": float(confirmation_sigma),
            "component_count": int(candidate_ids_array.size),
            "matched_component_count": confirmed_count,
            "component_density_per_megapixel": density,
            "single_pixel_component_ratio": single_pixel_ratio,
            "density_limit_exceeded": bool(density_exceeded),
            "density_contamination_risk": density_contamination,
            "single_pixel_limit_exceeded": bool(single_exceeded),
            "contamination_risk": contamination,
            "reference_insufficient": confirmed_count < 4,
        }
        attempts.append(attempt)
        if contamination:
            last_contaminated = (attempt, confirmed.copy())
            continue
        if confirmed_count >= 4:
            selected = (attempt, confirmed)
            break

    detail_percentile = float(
        _bounded(
            getattr(cfg, "stage9_source_star_detail_percentile", 98.0),
            98.0,
            97.0,
            99.5,
        )
    )
    raw_density = float(candidate_ids_array.size / megapixels)
    raw_single_pixel_ratio = float(np.mean(component_areas == 1))
    if selected is None:
        if last_contaminated is not None:
            failed_attempt, _failed_mask = last_contaminated
            exceeded = []
            if failed_attempt["density_limit_exceeded"]:
                exceeded.append(
                    "component_density_per_megapixel="
                    f"{failed_attempt['component_density_per_megapixel']:.1f}>"
                    f"{density_max:.1f}"
                )
            if failed_attempt["single_pixel_limit_exceeded"]:
                exceeded.append(
                    "single_pixel_component_ratio="
                    f"{failed_attempt['single_pixel_component_ratio']:.3f}>"
                    f"{single_pixel_max:.3f}"
                )
            return {
                "status": "rejected",
                "fail_closed": True,
                "reason": "source_star_catalog_contamination_risk: "
                + ", ".join(exceeded),
                "confirmation_method": "trusted_starmask_candidates_confirmed_in_mtf_O",
                "source_detail_role": "diagnostic_compatibility_only",
                "source_detail_percentile_requested": detail_percentile,
                "source_detail_percentile": detail_percentile,
                "source_detail_attempts": attempts,
                "source_component_count": int(candidate_ids_array.size),
                "source_component_density_per_megapixel": failed_attempt[
                    "component_density_per_megapixel"
                ],
                "source_component_density_max": density_max,
                "source_single_pixel_component_ratio": failed_attempt[
                    "single_pixel_component_ratio"
                ],
                "source_single_pixel_component_ratio_max": single_pixel_max,
                "source_raw_component_density_per_megapixel": raw_density,
                "source_raw_single_pixel_component_ratio": raw_single_pixel_ratio,
            }
        return {
            "status": "unavailable",
            "reason": "too few starmask candidates confirmed in matched-domain O",
            "confirmation_method": "trusted_starmask_candidates_confirmed_in_mtf_O",
            "source_detail_role": "diagnostic_compatibility_only",
            "source_detail_percentile_requested": detail_percentile,
            "source_detail_percentile": detail_percentile,
            "source_detail_attempts": attempts,
            "source_component_count": int(candidate_ids_array.size),
        }

    selected_attempt, confirmed = selected
    selected_indices = np.flatnonzero(confirmed)
    component_ids = candidate_ids_array[confirmed]
    component_peaks = candidate_peaks[confirmed]
    component_areas = component_areas[confirmed]
    confirmed_layer_y = layer_y[confirmed]
    confirmed_layer_x = layer_x[confirmed]
    source_y = np.asarray(
        [int(measurements[index]["center_y"]) for index in selected_indices],
        dtype=np.int32,
    )
    source_x = np.asarray(
        [int(measurements[index]["center_x"]) for index in selected_indices],
        dtype=np.int32,
    )
    source_fwhm = np.asarray(
        [float(measurements[index]["fwhm_px"]) for index in selected_indices],
        dtype=np.float32,
    )
    weak_cutoff = float(np.percentile(component_peaks, 80.0))
    weak_flags = np.asarray(component_peaks <= weak_cutoff, dtype=bool)
    bright_flags = ~weak_flags
    weak_count = int(np.count_nonzero(weak_flags))
    bright_count = int(np.count_nonzero(bright_flags))
    weak_peak_median = (
        float(np.median(component_peaks[weak_flags])) if weak_count else 0.0
    )
    bright_peak_median = (
        float(np.median(component_peaks[bright_flags])) if bright_count else 0.0
    )
    peak_ratio = bright_peak_median / max(weak_peak_median, 1e-12)
    weak_count_min = int(
        _bounded(
            getattr(cfg, "stage9_mixed_star_weak_count_min", 20),
            20,
            4,
            1000,
        )
    )
    bright_count_min = int(
        _bounded(
            getattr(cfg, "stage9_mixed_star_bright_count_min", 3),
            3,
            1,
            100,
        )
    )
    mixed_ratio_min = _bounded(
        getattr(cfg, "stage9_mixed_star_peak_ratio_min", 4.0),
        4.0,
        2.0,
        20.0,
    )

    weak_lookup = np.zeros(component_count + 1, dtype=bool)
    bright_lookup = np.zeros(component_count + 1, dtype=bool)
    weak_lookup[component_ids[weak_flags]] = True
    bright_lookup[component_ids[bright_flags]] = True
    peak_by_label = np.zeros(component_count + 1, dtype=np.float32)
    peak_by_label[component_ids] = component_peaks
    weak_core = weak_lookup[labels]
    bright_core = bright_lookup[labels]
    weak_core[confirmed_layer_y[weak_flags], confirmed_layer_x[weak_flags]] = True
    bright_core[confirmed_layer_y[bright_flags], confirmed_layer_x[bright_flags]] = True

    sum3 = scipy_ndimage.uniform_filter(star_peak, size=3, mode="constant") * 9.0
    sum7 = scipy_ndimage.uniform_filter(star_peak, size=7, mode="constant") * 49.0
    total_flux = np.asarray(
        sum7[confirmed_layer_y, confirmed_layer_x],
        dtype=np.float32,
    )
    wing_flux = np.maximum(
        total_flux
        - np.asarray(sum3[confirmed_layer_y, confirmed_layer_x], dtype=np.float32),
        0.0,
    )
    wing_ratio = np.divide(
        wing_flux,
        np.maximum(total_flux, 1e-12),
        out=np.zeros_like(wing_flux),
        where=total_flux > 0.0,
    )
    source_rgb = _rgb_channels(display)
    source_samples = np.asarray(source_rgb[:, source_y, source_x], dtype=np.float32).T
    source_chroma = source_samples / np.maximum(
        np.sum(source_samples, axis=1, keepdims=True),
        1e-12,
    )

    catalog: Dict[str, Any] = {
        "status": "ok",
        "method": "trusted_starmask_candidates_confirmed_in_mtf_O",
        "confirmation_method": "trusted_starmask_candidates_confirmed_in_mtf_O",
        "source_matched": True,
        "background": background,
        "noise_sigma": noise_sigma,
        "reference_sigma_requested": reference_sigma,
        "reference_sigma": float(selected_attempt["reference_sigma"]),
        "reference_threshold": threshold,
        "min_component_area": component_min_area,
        "min_component_area_nominal": 1,
        "max_component_area": component_max_area,
        "max_component_area_nominal": 512,
        "max_component_span": component_span_max,
        "max_component_span_nominal": 64,
        "measurement_search_radius_px": measurement_search_radius,
        "measurement_search_radius_px_nominal": 2,
        "measurement_patch_radius_px": measurement_patch_radius,
        "measurement_patch_radius_px_nominal": 6,
        "halfmax_area_max": halfmax_area_max,
        "halfmax_area_max_nominal": 128,
        "stage9_spatial_scale": dict(spatial_scale or {}),
        "source_detail_role": "diagnostic_compatibility_only",
        "source_detail_percentile_requested": detail_percentile,
        "source_detail_percentile": detail_percentile,
        "source_detail_threshold": None,
        "source_detail_attempts": attempts,
        "source_detail_adaptive_retry": bool(
            float(selected_attempt["reference_sigma"]) > reference_sigma
        ),
        "source_component_count": int(candidate_ids_array.size),
        "source_component_density_per_megapixel": float(
            selected_attempt["component_density_per_megapixel"]
        ),
        "source_component_density_max": density_max,
        "source_single_pixel_component_ratio": float(
            selected_attempt["single_pixel_component_ratio"]
        ),
        "source_single_pixel_component_ratio_max": single_pixel_max,
        "source_raw_component_density_per_megapixel": raw_density,
        "source_raw_single_pixel_component_ratio": raw_single_pixel_ratio,
        "component_count": int(component_ids.size),
        "matched_component_count": int(component_ids.size),
        "unmatched_component_count": int(candidate_ids_array.size - component_ids.size),
        "weak_component_count": weak_count,
        "bright_component_count": bright_count,
        "rejected_small_component_count": 0,
        "weak_peak_cutoff": weak_cutoff,
        "weak_peak_median": weak_peak_median,
        "bright_peak_median": bright_peak_median,
        "bright_to_weak_peak_ratio": float(peak_ratio),
        "mixed_peak_ratio_min": mixed_ratio_min,
        "mixed_weak_count_min": weak_count_min,
        "mixed_bright_count_min": bright_count_min,
        "mixed_star_field": bool(
            weak_count >= weak_count_min
            and bright_count >= bright_count_min
            and peak_ratio >= mixed_ratio_min
        ),
        "source_star_core_coverage": float(np.mean(weak_core | bright_core)),
        "psf_shadow_status": "shadow",
        "psf_support_count": int(source_fwhm.size),
        "source_fwhm_median_px": float(np.median(source_fwhm)),
        "source_fwhm_p25_px": float(np.percentile(source_fwhm, 25.0)),
        "source_fwhm_p75_px": float(np.percentile(source_fwhm, 75.0)),
        "_labels": labels,
        "_component_ids": component_ids,
        "_component_peaks": component_peaks,
        "_component_areas": component_areas,
        "_peak_y": confirmed_layer_y,
        "_peak_x": confirmed_layer_x,
        "_source_peak_y": source_y,
        "_source_peak_x": source_x,
        "_weak_flags": weak_flags,
        "_weak_lookup": weak_lookup,
        "_bright_lookup": bright_lookup,
        "_peak_by_label": peak_by_label,
        "_weak_core_mask": weak_core,
        "_bright_core_mask": bright_core,
        "_reference_wing_ratio": wing_ratio,
        "_source_fwhm_px": source_fwhm,
        "_stage9_spatial_fwhm_px": source_fwhm.copy(),
        "_source_chroma": np.asarray(source_chroma, dtype=np.float32),
    }
    return enrich_star_reference_with_display_psf(catalog, display, cfg)


def _build_source_matched_star_catalog(
    normalized_stars: np.ndarray,
    source_image: np.ndarray,
    cfg: Any,
    *,
    background: float,
    noise_sigma: float,
    spatial_scale: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build the Stage 9 catalog from the original full image, then match starmask."""
    if scipy_ndimage is None:
        return {"status": "unavailable", "reason": "scipy.ndimage unavailable"}
    source = _normalized(np.asarray(source_image))
    stars = np.asarray(normalized_stars, dtype=np.float32)
    source_gray = _luminance(source)
    source_peak = _pixel_peak(source)
    star_peak = _pixel_peak(stars)
    if source_gray.shape != star_peak.shape:
        return {
            "status": "unavailable",
            "reason": (
                "source/starmask shape mismatch: "
                f"source={source_gray.shape}, starmask={star_peak.shape}"
            ),
        }

    requested_detail_percentile = _bounded(
        getattr(cfg, "stage9_source_star_detail_percentile", 98.0),
        98.0,
        97.0,
        99.5,
    )
    broad = scipy_ndimage.gaussian_filter(source_gray, sigma=2.0, mode="reflect")
    local_detail = np.maximum(source_gray - broad, 0.0)
    source_megapixels = max(float(source_gray.size) / 1_000_000.0, 1e-6)
    source_component_density_max = _bounded(
        getattr(cfg, "stage9_source_component_density_max", 2500.0),
        2500.0,
        500.0,
        10000.0,
    )
    source_single_pixel_ratio_max = _bounded(
        getattr(cfg, "stage9_source_single_pixel_ratio_max", 0.20),
        0.20,
        0.10,
        0.90,
    )
    geometry_reference = {"stage9_spatial_scale": spatial_scale or {}}
    match_radius = stage9_scale_radius(
        2,
        geometry_reference,
        rounding="nearest",
        minimum=1,
    )
    local_peak_window = stage9_scale_odd_window(5, geometry_reference)
    source_component_min_area = stage9_scale_area(1, geometry_reference)
    source_component_max_area = stage9_scale_area(128, geometry_reference)
    source_component_span_max = stage9_scale_radius(
        16,
        geometry_reference,
        rounding="nearest",
        minimum=1,
    )
    local_star_peak = scipy_ndimage.maximum_filter(
        star_peak,
        size=local_peak_window,
        mode="constant",
        cval=0.0,
    )

    def match_source_positions(
        source_y: np.ndarray,
        source_x: np.ndarray,
        threshold: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Intersect source detail with unique starmask peaks within two pixels."""
        height, width = star_peak.shape
        matched_y = np.empty_like(source_y)
        matched_x = np.empty_like(source_x)
        matched_peaks = np.zeros(source_y.size, dtype=np.float32)
        for index, (center_y, center_x) in enumerate(zip(source_y, source_x)):
            y0 = max(0, int(center_y) - match_radius)
            y1 = min(height, int(center_y) + match_radius + 1)
            x0 = max(0, int(center_x) - match_radius)
            x1 = min(width, int(center_x) + match_radius + 1)
            window = star_peak[y0:y1, x0:x1]
            local_y, local_x = np.unravel_index(
                int(np.argmax(window)),
                window.shape,
            )
            matched_y[index] = y0 + int(local_y)
            matched_x[index] = x0 + int(local_x)
            matched_peaks[index] = float(window[local_y, local_x])
        local_maximum_flags = matched_peaks >= (
            local_star_peak[matched_y, matched_x] - 1e-12
        )
        matched_flags = (matched_peaks > float(threshold)) & local_maximum_flags
        # A single trusted starmask maximum represents one star.  Fragmented
        # source-detail islands around that maximum must not inflate the catalog.
        selected_by_peak: Dict[Tuple[int, int], int] = {}
        for index in np.flatnonzero(matched_flags):
            key = (int(matched_y[index]), int(matched_x[index]))
            previous = selected_by_peak.get(key)
            if previous is None or float(source_peak[source_y[index], source_x[index]]) > float(
                source_peak[source_y[previous], source_x[previous]]
            ):
                if previous is not None:
                    matched_flags[previous] = False
                selected_by_peak[key] = int(index)
            else:
                matched_flags[index] = False
        return matched_flags, matched_y, matched_x, matched_peaks
    reference_sigma = _bounded(
        getattr(cfg, "stage9_star_reference_sigma", 5.0),
        5.0,
        3.0,
        8.0,
    )
    # Keep the source-detail population fixed.  A contaminated match is retried
    # by requiring stronger independent starmask evidence, rather than raising
    # the source percentile and silently deleting most weak stars.
    reference_sigmas = [reference_sigma]
    next_sigma = math.ceil((reference_sigma + 0.01) * 2.0) / 2.0
    while next_sigma <= 8.0:
        reference_sigmas.append(next_sigma)
        next_sigma += 0.5

    attempts = []
    selected_components = None
    detail_percentile = requested_detail_percentile
    for active_reference_sigma in reference_sigmas:
        match_threshold = max(
            float(background)
            + active_reference_sigma * float(noise_sigma),
            1e-6,
        )
        detail_threshold = float(np.percentile(local_detail, detail_percentile))
        labels, component_count = scipy_ndimage.label(
            local_detail > detail_threshold,
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        if component_count <= 0:
            continue

        areas = np.bincount(labels.reshape(-1), minlength=component_count + 1)
        source_ids = []
        for index, bounds in enumerate(
            scipy_ndimage.find_objects(labels),
            start=1,
        ):
            if bounds is None:
                continue
            area = int(areas[index])
            height = int(bounds[0].stop - bounds[0].start)
            width = int(bounds[1].stop - bounds[1].start)
            longest = max(height, width)
            shortest = max(1, min(height, width))
            fill_ratio = area / max(1, height * width)
            if (
                source_component_min_area <= area <= source_component_max_area
                and longest <= source_component_span_max
                and longest / shortest <= 3.0
                and fill_ratio >= 0.15
            ):
                source_ids.append(index)
        if not source_ids:
            continue

        source_ids_arr = np.asarray(source_ids, dtype=np.int32)
        source_component_areas = areas[source_ids_arr].astype(
            np.int32,
            copy=False,
        )
        source_positions = scipy_ndimage.maximum_position(
            source_peak,
            labels=labels,
            index=source_ids_arr,
        )
        source_y = np.asarray(
            [item[0] for item in source_positions],
            dtype=np.int32,
        )
        source_x = np.asarray(
            [item[1] for item in source_positions],
            dtype=np.int32,
        )
        matched, _matched_y, _matched_x, component_peaks_all = (
            match_source_positions(
                source_y,
                source_x,
                match_threshold,
            )
        )
        matched_component_areas = source_component_areas[matched]
        matched_component_count = int(np.count_nonzero(matched))
        raw_component_density = float(source_ids_arr.size / source_megapixels)
        raw_single_pixel_ratio = float(np.mean(source_component_areas == 1))
        source_component_density = float(
            matched_component_count / source_megapixels
        )
        source_single_pixel_ratio = (
            float(np.mean(matched_component_areas == 1))
            if matched_component_areas.size
            else 0.0
        )
        density_limit_exceeded = bool(
            source_component_density > source_component_density_max
        )
        single_pixel_limit_exceeded = bool(
            source_single_pixel_ratio > source_single_pixel_ratio_max
        )
        density_contamination_risk = bool(
            density_limit_exceeded and source_single_pixel_ratio > 0.10
        )
        contamination_risk = bool(
            density_contamination_risk or single_pixel_limit_exceeded
        )
        attempts.append(
            {
                "percentile": float(detail_percentile),
                "threshold": detail_threshold,
                "reference_sigma": float(active_reference_sigma),
                "match_threshold": float(match_threshold),
                "component_count": int(source_ids_arr.size),
                "matched_component_count": matched_component_count,
                "component_density_per_megapixel": source_component_density,
                "single_pixel_component_ratio": source_single_pixel_ratio,
                "raw_component_density_per_megapixel": raw_component_density,
                "raw_single_pixel_component_ratio": raw_single_pixel_ratio,
                "density_limit_exceeded": density_limit_exceeded,
                "density_contamination_risk": density_contamination_risk,
                "single_pixel_limit_exceeded": single_pixel_limit_exceeded,
                "contamination_risk": contamination_risk,
                "reference_insufficient": matched_component_count < 4,
            }
        )
        if not contamination_risk and matched_component_count >= 4:
            selected_components = (
                detail_percentile,
                detail_threshold,
                labels,
                component_count,
                areas,
                source_ids_arr,
                source_component_density,
                source_single_pixel_ratio,
                raw_component_density,
                raw_single_pixel_ratio,
                active_reference_sigma,
                match_threshold,
            )
            break

    if selected_components is None:
        if not attempts:
            return {
                "status": "unavailable",
                "reason": "no compact source-image star components",
            }
        last_attempt = attempts[-1]
        if bool(last_attempt.get("reference_insufficient", False)) and not bool(
            last_attempt.get("contamination_risk", False)
        ):
            contaminated_attempts = [
                item for item in attempts if bool(item.get("contamination_risk"))
            ]
            if contaminated_attempts:
                # Raising the confirmation sigma until all candidates disappear
                # does not turn an observed contamination failure into an
                # unavailable reference; preserve the last measured hard risk.
                last_attempt = contaminated_attempts[-1]
            else:
                return {
                    "status": "unavailable",
                    "reason": "too few source stars matched to starmask",
                    "source_detail_percentile_requested": requested_detail_percentile,
                    "source_detail_percentile": last_attempt["percentile"],
                    "source_detail_threshold": last_attempt["threshold"],
                    "source_detail_attempts": attempts,
                    "source_component_count": last_attempt["component_count"],
                    "matched_component_count": last_attempt[
                        "matched_component_count"
                    ],
                    "source_component_density_per_megapixel": last_attempt[
                        "component_density_per_megapixel"
                    ],
                    "source_component_density_max": source_component_density_max,
                    "source_single_pixel_component_ratio": last_attempt[
                        "single_pixel_component_ratio"
                    ],
                    "source_single_pixel_component_ratio_max": (
                        source_single_pixel_ratio_max
                    ),
                    "source_raw_component_density_per_megapixel": last_attempt[
                        "raw_component_density_per_megapixel"
                    ],
                    "source_raw_single_pixel_component_ratio": last_attempt[
                        "raw_single_pixel_component_ratio"
                    ],
                }
        exceeded_limits = []
        if last_attempt["density_limit_exceeded"]:
            exceeded_limits.append(
                "component_density_per_megapixel="
                f"{last_attempt['component_density_per_megapixel']:.1f}>"
                f"{source_component_density_max:.1f}"
            )
        if last_attempt["single_pixel_limit_exceeded"]:
            exceeded_limits.append(
                "single_pixel_component_ratio="
                f"{last_attempt['single_pixel_component_ratio']:.3f}>"
                f"{source_single_pixel_ratio_max:.3f}"
            )
        return {
            "status": "rejected",
            "reason": (
                "source_star_catalog_contamination_risk: "
                + ", ".join(exceeded_limits)
            ),
            "source_detail_percentile_requested": requested_detail_percentile,
            "source_detail_percentile": last_attempt["percentile"],
            "source_detail_threshold": last_attempt["threshold"],
            "source_detail_attempts": attempts,
            "source_component_count": last_attempt["component_count"],
            "source_component_density_per_megapixel": last_attempt[
                "component_density_per_megapixel"
            ],
            "source_component_density_max": source_component_density_max,
            "source_single_pixel_component_ratio": last_attempt[
                "single_pixel_component_ratio"
            ],
            "source_single_pixel_component_ratio_max": (
                source_single_pixel_ratio_max
            ),
            "source_raw_component_density_per_megapixel": last_attempt[
                "raw_component_density_per_megapixel"
            ],
            "source_raw_single_pixel_component_ratio": last_attempt[
                "raw_single_pixel_component_ratio"
            ],
            "fail_closed": True,
        }
    (
        detail_percentile,
        detail_threshold,
        labels,
        component_count,
        areas,
        source_ids_arr,
        source_component_density,
        source_single_pixel_ratio,
        raw_component_density,
        raw_single_pixel_ratio,
        active_reference_sigma,
        match_threshold,
    ) = selected_components
    source_positions = scipy_ndimage.maximum_position(
        source_peak,
        labels=labels,
        index=source_ids_arr,
    )
    source_y = np.asarray([item[0] for item in source_positions], dtype=np.int32)
    source_x = np.asarray([item[1] for item in source_positions], dtype=np.int32)
    matched, matched_peak_y_all, matched_peak_x_all, component_peaks_all = (
        match_source_positions(
            source_y,
            source_x,
            match_threshold,
        )
    )
    component_ids = source_ids_arr[matched]
    component_peaks = component_peaks_all[matched]
    source_peak_y = source_y[matched]
    source_peak_x = source_x[matched]
    peak_y = source_peak_y.copy()
    peak_x = source_peak_x.copy()
    if component_ids.size < 4:
        return {
            "status": "unavailable",
            "reason": "too few source stars matched to starmask",
            "source_component_count": int(source_ids_arr.size),
            "matched_component_count": int(component_ids.size),
        }

    source_rgb = _rgb_channels(source)
    source_fwhm_px = []
    source_chroma = []
    for center_y, center_x in zip(peak_y, peak_x):
        y = int(center_y)
        x = int(center_x)
        halfmax_radius = stage9_scale_radius(
            4,
            geometry_reference,
            rounding="nearest",
            minimum=1,
        )
        y0 = max(0, y - halfmax_radius)
        y1 = min(source_gray.shape[0], y + halfmax_radius + 1)
        x0 = max(0, x - halfmax_radius)
        x1 = min(source_gray.shape[1], x + halfmax_radius + 1)
        local_background = float(broad[y, x])
        local_peak = float(source_peak[y, x])
        half_max = local_background + 0.5 * max(
            local_peak - local_background,
            0.0,
        )
        half_max_area = int(
            np.count_nonzero(source_peak[y0:y1, x0:x1] >= half_max)
        )
        source_fwhm_px.append(
            2.0 * math.sqrt(max(half_max_area, 1) / math.pi)
        )
        rgb_sample = np.asarray(source_rgb[:, y, x], dtype=np.float32)
        source_chroma.append(
            rgb_sample / max(float(np.sum(rgb_sample)), 1e-12)
        )
    source_fwhm_px_arr = np.asarray(source_fwhm_px, dtype=np.float32)
    source_chroma_arr = np.asarray(source_chroma, dtype=np.float32)

    # Preserve distinct coordinates: source/O measurements remain tied to the
    # original with-stars image, while layer operations use the matched starmask
    # maximum (bounded to ±2 px).
    peak_y = matched_peak_y_all[matched]
    peak_x = matched_peak_x_all[matched]

    weak_cutoff = float(np.percentile(component_peaks, 80.0))
    weak_flags = np.asarray(component_peaks <= weak_cutoff, dtype=bool)
    bright_flags = ~weak_flags
    weak_count = int(np.count_nonzero(weak_flags))
    bright_count = int(np.count_nonzero(bright_flags))
    weak_peak_median = float(np.median(component_peaks[weak_flags]))
    bright_peak_median = (
        float(np.median(component_peaks[bright_flags])) if bright_count else 0.0
    )
    peak_ratio = bright_peak_median / max(weak_peak_median, 1e-12)
    weak_count_min = int(
        _bounded(
            getattr(cfg, "stage9_mixed_star_weak_count_min", 20),
            20,
            4,
            1000,
        )
    )
    bright_count_min = int(
        _bounded(
            getattr(cfg, "stage9_mixed_star_bright_count_min", 3),
            3,
            1,
            100,
        )
    )
    mixed_ratio_min = _bounded(
        getattr(cfg, "stage9_mixed_star_peak_ratio_min", 4.0),
        4.0,
        2.0,
        20.0,
    )

    weak_lookup = np.zeros(component_count + 1, dtype=bool)
    bright_lookup = np.zeros(component_count + 1, dtype=bool)
    weak_lookup[component_ids[weak_flags]] = True
    bright_lookup[component_ids[bright_flags]] = True
    peak_by_label = np.zeros(component_count + 1, dtype=np.float32)
    peak_by_label[component_ids] = component_peaks
    sum3 = scipy_ndimage.uniform_filter(star_peak, size=3, mode="constant") * 9.0
    sum7 = scipy_ndimage.uniform_filter(star_peak, size=7, mode="constant") * 49.0
    reference_total_flux = np.asarray(sum7[peak_y, peak_x], dtype=np.float32)
    reference_wing_flux = np.maximum(
        reference_total_flux - np.asarray(sum3[peak_y, peak_x], dtype=np.float32),
        0.0,
    )
    reference_wing_ratio = np.divide(
        reference_wing_flux,
        np.maximum(reference_total_flux, 1e-12),
        out=np.zeros_like(reference_wing_flux),
        where=reference_total_flux > 0.0,
    )
    weak_core_mask = weak_lookup[labels]
    bright_core_mask = bright_lookup[labels]
    weak_core_mask[peak_y[weak_flags], peak_x[weak_flags]] = True
    bright_core_mask[peak_y[bright_flags], peak_x[bright_flags]] = True
    core_mask = weak_core_mask | bright_core_mask
    return {
        "status": "ok",
        "method": "source_matched_local_detail_star_catalog",
        "source_matched": True,
        "background": float(background),
        "noise_sigma": float(noise_sigma),
        "reference_sigma_requested": reference_sigma,
        "reference_sigma": float(active_reference_sigma),
        "reference_threshold": match_threshold,
        "min_component_area": 1,
        "min_component_area_nominal": 1,
        "min_component_area_effective": source_component_min_area,
        "max_component_area_nominal": 128,
        "max_component_area": source_component_max_area,
        "max_component_span_nominal": 16,
        "max_component_span": source_component_span_max,
        "match_radius_nominal_px": 2,
        "match_radius_effective_px": match_radius,
        "local_peak_window_nominal_px": 5,
        "local_peak_window_effective_px": local_peak_window,
        "stage9_spatial_scale": dict(spatial_scale or {}),
        "source_detail_percentile_requested": requested_detail_percentile,
        "source_detail_percentile": detail_percentile,
        "source_detail_threshold": detail_threshold,
        "source_detail_attempts": attempts,
        "source_detail_adaptive_retry": bool(
            active_reference_sigma > reference_sigma
        ),
        "source_component_count": int(source_ids_arr.size),
        "source_component_density_per_megapixel": source_component_density,
        "source_component_density_max": source_component_density_max,
        "source_single_pixel_component_ratio": source_single_pixel_ratio,
        "source_single_pixel_component_ratio_max": source_single_pixel_ratio_max,
        "source_raw_component_density_per_megapixel": raw_component_density,
        "source_raw_single_pixel_component_ratio": raw_single_pixel_ratio,
        "component_count": int(component_ids.size),
        "matched_component_count": int(component_ids.size),
        "unmatched_component_count": int(np.count_nonzero(~matched)),
        "weak_component_count": weak_count,
        "bright_component_count": bright_count,
        "rejected_small_component_count": 0,
        "weak_peak_cutoff": weak_cutoff,
        "weak_peak_median": weak_peak_median,
        "bright_peak_median": bright_peak_median,
        "bright_to_weak_peak_ratio": float(peak_ratio),
        "mixed_peak_ratio_min": mixed_ratio_min,
        "mixed_weak_count_min": weak_count_min,
        "mixed_bright_count_min": bright_count_min,
        "mixed_star_field": bool(
            weak_count >= weak_count_min
            and bright_count >= bright_count_min
            and peak_ratio >= mixed_ratio_min
        ),
        "source_star_core_coverage": float(np.mean(core_mask)),
        "psf_shadow_status": "shadow",
        "psf_support_count": int(source_fwhm_px_arr.size),
        "source_fwhm_median_px": float(np.median(source_fwhm_px_arr)),
        "source_fwhm_p25_px": float(np.percentile(source_fwhm_px_arr, 25.0)),
        "source_fwhm_p75_px": float(np.percentile(source_fwhm_px_arr, 75.0)),
        "_labels": labels,
        "_component_ids": component_ids,
        "_component_peaks": component_peaks,
        "_component_areas": areas[component_ids].astype(np.int32, copy=False),
        "_peak_y": peak_y,
        "_peak_x": peak_x,
        "_source_peak_y": source_peak_y,
        "_source_peak_x": source_peak_x,
        "_weak_flags": weak_flags,
        "_weak_lookup": weak_lookup,
        "_bright_lookup": bright_lookup,
        "_peak_by_label": peak_by_label,
        "_weak_core_mask": weak_core_mask,
        "_bright_core_mask": bright_core_mask,
        "_reference_wing_ratio": reference_wing_ratio,
        "_source_fwhm_px": source_fwhm_px_arr,
        "_stage9_spatial_fwhm_px": source_fwhm_px_arr.copy(),
        "_source_chroma": source_chroma_arr,
    }


def star_reference_summary(catalog: Dict[str, Any] | None) -> Dict[str, Any]:
    """Return the JSON-safe public portion of a reference catalog."""
    if not isinstance(catalog, dict):
        return {"status": "unavailable", "reason": "catalog missing"}
    return {
        key: value
        for key, value in catalog.items()
        if not str(key).startswith("_")
    }


def build_stage5_bright_star_completion(
    stage5_stars: Any,
    existing_catalog: Dict[str, Any],
    original_display: np.ndarray,
    trusted_stars: np.ndarray,
    cfg: Any,
) -> Dict[str, Any]:
    """Build an independent support group for omitted bright/saturated stars.

    Siril stores star coordinates bottom-up while numpy image arrays are
    top-down.  The frozen Stage 5 coordinates are converted here, deduplicated
    against the normal Stage 9 catalog, and then confirmed by local trusted
    star-layer energy.  These entries never participate in the ordinary FWHM
    hard gate.
    """
    report: Dict[str, Any] = {
        "schema": "starun.stage9-stage5-bright-star-completion.v1",
        "status": "unavailable",
        "available": False,
        "reason_code": "stage9_stage5_bright_star_completion_unavailable",
        "gate_role": "presence_and_wing_observation_only",
        "ordinary_fwhm_gate_member": False,
    }
    if scipy_ndimage is None:
        report["reason"] = "scipy.ndimage unavailable"
        return report
    try:
        source_entries = list(stage5_stars or [])
        if not source_entries:
            raise ValueError("frozen Stage5 star catalog is empty")
        display = _normalized(np.asarray(original_display))
        trusted = _normalized(np.asarray(trusted_stars))
        display_peak = _pixel_peak(display)
        trusted_peak = _pixel_peak(trusted)
        if display_peak.shape != trusted_peak.shape:
            raise ValueError("Stage5 completion image shapes do not match")
        height, width = display_peak.shape

        existing_y = np.asarray(
            existing_catalog.get(
                "_display_source_peak_y",
                existing_catalog.get("_source_peak_y", ()),
            ),
            dtype=np.int32,
        )
        existing_x = np.asarray(
            existing_catalog.get(
                "_display_source_peak_x",
                existing_catalog.get("_source_peak_x", ()),
            ),
            dtype=np.int32,
        )
        existing_valid = (
            (existing_y >= 0)
            & (existing_y < height)
            & (existing_x >= 0)
            & (existing_x < width)
        ) if existing_y.size == existing_x.size else np.zeros(0, dtype=bool)
        existing_coords = np.column_stack(
            (existing_y[existing_valid], existing_x[existing_valid])
        )
        match_radius_nominal = _bounded(
            getattr(cfg, "stage9_stage5_bright_star_match_radius", 3.0),
            3.0,
            1.0,
            8.0,
        )
        fwhm_min = _bounded(
            getattr(cfg, "stage9_stage5_bright_star_fwhm_min", 8.0),
            8.0,
            4.0,
            20.0,
        )
        radius_max_nominal = int(
            _bounded(
                getattr(
                    cfg,
                    "stage9_stage5_bright_star_support_radius_max",
                    12,
                ),
                12,
                6,
                16,
            )
        )
        selected: list[Dict[str, Any]] = []
        match_radii: list[float] = []
        search_radii: list[int] = []
        evidence_windows: list[int] = []
        rejected = {
            "invalid_geometry": 0,
            "ordinary_non_saturated": 0,
            "already_in_stage9_catalog": 0,
            "trusted_star_evidence_missing": 0,
        }
        for entry in source_entries:
            try:
                x_float = float(entry.get("x"))
                y_siril = float(entry.get("y"))
                fwhm = float(entry.get("fwhm_geometry"))
            except (AttributeError, TypeError, ValueError):
                rejected["invalid_geometry"] += 1
                continue
            saturated = bool(entry.get("saturated", False))
            if not all(math.isfinite(value) for value in (x_float, y_siril, fwhm)):
                rejected["invalid_geometry"] += 1
                continue
            if fwhm <= 0.0 or (not saturated and fwhm < fwhm_min):
                rejected["ordinary_non_saturated"] += 1
                continue
            x = int(round(x_float))
            y = int(round((height - 1) - y_siril))
            if not (0 <= x < width and 0 <= y < height):
                rejected["invalid_geometry"] += 1
                continue
            if existing_coords.size:
                match_radius = stage9_scale_distance(
                    match_radius_nominal,
                    existing_catalog,
                    fwhm_px=fwhm,
                )
                match_radii.append(match_radius)
                distances = np.hypot(
                    existing_coords[:, 0] - y,
                    existing_coords[:, 1] - x,
                )
                if float(np.min(distances)) <= match_radius:
                    rejected["already_in_stage9_catalog"] += 1
                    continue

            search_radius = stage9_scale_radius(
                2,
                existing_catalog,
                fwhm_px=fwhm,
                rounding="nearest",
                minimum=1,
            )
            search_radii.append(search_radius)
            y0 = max(0, y - search_radius)
            y1 = min(height, y + search_radius + 1)
            x0 = max(0, x - search_radius)
            x1 = min(width, x + search_radius + 1)
            window = display_peak[y0:y1, x0:x1]
            local_y, local_x = np.unravel_index(
                int(np.argmax(window)),
                window.shape,
            )
            peak_y = y0 + int(local_y)
            peak_x = x0 + int(local_x)
            evidence_size = stage9_scale_odd_window(
                5,
                existing_catalog,
                fwhm_px=fwhm,
            )
            evidence_radius = (evidence_size - 1) // 2
            evidence_windows.append(evidence_size)
            ty0 = max(0, peak_y - evidence_radius)
            ty1 = min(height, peak_y + evidence_radius + 1)
            tx0 = max(0, peak_x - evidence_radius)
            tx1 = min(width, peak_x + evidence_radius + 1)
            trusted_evidence = float(
                np.max(trusted_peak[ty0:ty1, tx0:tx1])
            )
            threshold = max(
                float(existing_catalog.get("reference_threshold", 0.0) or 0.0),
                1e-4,
            )
            if trusted_evidence < threshold:
                rejected["trusted_star_evidence_missing"] += 1
                continue
            radius_max = stage9_scale_radius(
                radius_max_nominal,
                existing_catalog,
                fwhm_px=fwhm,
                rounding="ceil",
                minimum=1,
            )
            radius = max(
                stage9_scale_radius(
                    3,
                    existing_catalog,
                    fwhm_px=fwhm,
                    rounding="ceil",
                    minimum=1,
                ),
                min(
                    radius_max,
                    int(math.ceil(fwhm / 2.0))
                    + stage9_scale_radius(
                        2,
                        existing_catalog,
                        fwhm_px=fwhm,
                        rounding="ceil",
                    ),
                ),
            )
            selected.append(
                {
                    "source_index": int(entry.get("index", len(selected)) or 0),
                    "x": peak_x,
                    "y": peak_y,
                    "stage5_fwhm_px": fwhm,
                    "saturated": saturated,
                    "support_radius_px": radius,
                    "trusted_peak": trusted_evidence,
                }
            )
        if not selected:
            raise ValueError("no omitted Stage5 bright/saturated stars were confirmed")

        # Deduplicate the rare case where two Stage5 fits recenter to one pixel.
        unique: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for entry in selected:
            key = (int(entry["y"]), int(entry["x"]))
            previous = unique.get(key)
            if previous is None or float(entry["stage5_fwhm_px"]) > float(
                previous["stage5_fwhm_px"]
            ):
                unique[key] = entry
        selected = list(unique.values())
        support = np.zeros((height, width), dtype=bool)
        for entry in selected:
            y = int(entry["y"])
            x = int(entry["x"])
            radius = int(entry["support_radius_px"])
            y0 = max(0, y - radius)
            y1 = min(height, y + radius + 1)
            x0 = max(0, x - radius)
            x1 = min(width, x + radius + 1)
            grid_y, grid_x = np.ogrid[y0:y1, x0:x1]
            support[y0:y1, x0:x1] |= (
                (grid_y - y) ** 2 + (grid_x - x) ** 2 <= radius * radius
            )
        saturated_count = sum(bool(entry["saturated"]) for entry in selected)
        report.update(
            status="ready",
            available=True,
            reason_code="stage9_stage5_bright_star_completion_ready",
            source_schema="starun.stage5-star-reference.v1",
            coordinate_conversion="numpy_y = image_height - 1 - siril_y",
            selection_semantics=(
                "frozen_stage5_saturated_or_large_star_not_already_in_stage9_catalog_"
                "and_confirmed_by_trusted_star_layer"
            ),
            source_star_count=len(source_entries),
            selected_star_count=len(selected),
            selected_saturated_count=saturated_count,
            selected_large_unsaturated_count=len(selected) - saturated_count,
            support_pixel_count=int(np.count_nonzero(support)),
            support_ratio=float(np.mean(support)),
            support_radius_max=radius_max_nominal,
            support_radius_max_nominal=radius_max_nominal,
            support_radius_effective_px=stage9_effective_pixel_stats(
                [entry["support_radius_px"] for entry in selected]
            ),
            support_radius_median_px=float(
                np.median([entry["support_radius_px"] for entry in selected])
            ),
            stage5_fwhm_min=fwhm_min,
            existing_catalog_match_radius=match_radius_nominal,
            existing_catalog_match_radius_nominal=match_radius_nominal,
            existing_catalog_match_radius_effective_px=(
                stage9_effective_pixel_stats(match_radii)
            ),
            search_radius_nominal_px=2,
            search_radius_effective_px=stage9_effective_pixel_stats(
                search_radii
            ),
            trusted_evidence_window_nominal_px=5,
            trusted_evidence_window_effective_px=stage9_effective_pixel_stats(
                evidence_windows
            ),
            rejected_counts=rejected,
            stars=selected,
        )
        report["_support_mask"] = support
        return report
    except (IndexError, TypeError, ValueError, FloatingPointError) as error:
        report["reason"] = str(error)
        return report


def apply_stage5_bright_star_completion(
    original_display: np.ndarray,
    starless_display: np.ndarray,
    stars: np.ndarray,
    completion: Dict[str, Any],
    cfg: Any,
    *,
    remix_base: np.ndarray | None = None,
    screen_intensity: float = 1.0,
) -> tuple[np.ndarray | None, np.ndarray | None, Dict[str, Any]]:
    """Merge bright-star wings without enlarging saturated source cores.

    The raw matched-domain Unscreen layer exactly reconstructs ``original``
    only over its paired ``starless`` image.  Stage 9 composes onto the later
    Stage 8 base, whose local signal can be higher.  Bound the completion term
    by the actual Screen headroom so the composed peak cannot exceed the
    frozen source peak at the same pixel.
    """
    public_report = {
        key: value
        for key, value in dict(completion or {}).items()
        if not str(key).startswith("_")
    }
    try:
        if completion.get("status") != "ready":
            raise ValueError(str(completion.get("reason") or "completion unavailable"))
        original = _normalized(np.asarray(original_display))
        starless = _normalized(np.asarray(starless_display))
        output = _normalized(np.asarray(stars)).copy()
        if remix_base is None:
            raise ValueError("bright-star completion remix base is unavailable")
        base = _normalized(np.asarray(remix_base))
        if not (original.shape == starless.shape == output.shape == base.shape):
            raise ValueError("bright-star completion image shapes do not match")
        support = np.asarray(completion.get("_support_mask"), dtype=bool)
        if support.shape != _pixel_peak(original).shape:
            raise ValueError("bright-star completion support shape mismatch")
        denominator_floor = _bounded(
            getattr(cfg, "stage9_unscreen_denominator_floor", 0.08),
            0.08,
            0.02,
            0.25,
        )
        denominator = 1.0 - starless
        residual = original - starless
        reliable = support & (_pixel_floor(denominator) >= denominator_floor)
        raw_unscreen = np.clip(
            residual / np.maximum(denominator, denominator_floor),
            0.0,
            1.0,
        ).astype(np.float32, copy=False)
        usable = reliable & (_pixel_peak(raw_unscreen) > 1e-7)
        if not np.any(usable):
            raise ValueError("bright-star completion has no usable Unscreen pixels")

        source_peak = _pixel_peak(original)
        base_peak = _pixel_peak(base)
        raw_peak = _pixel_peak(raw_unscreen)
        intensity = max(float(screen_intensity), 1e-6)
        allowed_screen_term = np.divide(
            np.maximum(source_peak - base_peak, 0.0),
            np.maximum(1.0 - base_peak, denominator_floor),
            out=np.zeros_like(source_peak, dtype=np.float32),
            where=base_peak < 1.0,
        )
        allowed_layer_peak = np.clip(
            allowed_screen_term / intensity,
            0.0,
            1.0,
        )
        peak_scale = np.minimum(
            1.0,
            np.divide(
                allowed_layer_peak,
                np.maximum(raw_peak, 1e-12),
                out=np.zeros_like(raw_peak, dtype=np.float32),
                where=raw_peak > 0.0,
            ),
        )
        safe_unscreen = raw_unscreen * _expanded_spatial_mask(
            raw_unscreen,
            peak_scale,
        )
        reduced = usable & (peak_scale < (1.0 - 1e-6))
        safe_usable = usable & (_pixel_peak(safe_unscreen) > 1e-7)
        if not np.any(safe_usable):
            raise ValueError(
                "bright-star completion has no source-bounded Screen headroom"
            )
        merged = np.maximum(output, safe_unscreen)
        merged_peak = _pixel_peak(merged)
        merged_scale = np.minimum(
            1.0,
            np.divide(
                allowed_layer_peak,
                np.maximum(merged_peak, 1e-12),
                out=np.zeros_like(merged_peak, dtype=np.float32),
                where=merged_peak > 0.0,
            ),
        )
        bounded_merged = merged * _expanded_spatial_mask(merged, merged_scale)
        output_reduced = support & (merged_scale < (1.0 - 1e-6))
        output = np.where(
            _expanded_spatial_mask(output, support.astype(np.float32)) > 0.0,
            bounded_merged,
            output,
        )
        effective_support = support & (_pixel_peak(output) > 1e-7)
        public_report.update(
            status="ready",
            available=True,
            amplitude_semantics=(
                "matched_domain_unscreen_source_peak_bounded_for_actual_remix_base"
            ),
            source_peak_cap_applied=True,
            source_peak_cap_screen_intensity=intensity,
            source_peak_cap_reduced_pixel_count=int(np.count_nonzero(reduced)),
            source_peak_cap_reduced_support_ratio=float(
                np.count_nonzero(reduced) / max(1, np.count_nonzero(usable))
            ),
            source_peak_cap_output_reduced_pixel_count=int(
                np.count_nonzero(output_reduced)
            ),
            reliable_pixel_count=int(np.count_nonzero(reliable)),
            reliable_support_ratio=float(
                np.count_nonzero(reliable) / max(1, np.count_nonzero(support))
            ),
            effective_support_pixel_count=int(np.count_nonzero(effective_support)),
            effective_support_ratio=float(np.mean(effective_support)),
            denominator_floor=denominator_floor,
        )
        return output.astype(np.float32, copy=False), effective_support, public_report
    except (IndexError, TypeError, ValueError, FloatingPointError) as error:
        public_report.update(
            status="unavailable",
            available=False,
            reason_code="stage9_stage5_bright_star_completion_unavailable",
            reason=str(error),
        )
        return None, None, public_report


def assess_stage5_bright_star_presence(
    base_image: np.ndarray,
    candidate_image: np.ndarray,
    completion_report: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """Observe whether frozen bright/saturated coordinates gained star signal."""
    result: Dict[str, Any] = {
        "schema": "starun.stage9-stage5-bright-star-presence.v1",
        "status": "unavailable",
        "available": False,
        "gate_role": "presence_and_wing_observation_only",
        "ordinary_fwhm_gate_member": False,
    }
    try:
        report = dict(completion_report or {})
        stars = list(report.get("stars") or [])
        if report.get("status") != "ready" or not stars:
            raise ValueError(str(report.get("reason") or "completion unavailable"))
        base = _normalized(np.asarray(base_image))
        candidate = _normalized(np.asarray(candidate_image))
        if base.shape != candidate.shape:
            raise ValueError("bright-star presence image shapes do not match")
        positive_peak = _pixel_peak(np.maximum(candidate - base, 0.0))
        restored = []
        window_sums = []
        saturated_restored = []
        for star in stars:
            y = int(star["y"])
            x = int(star["x"])
            y0 = max(0, y - 3)
            y1 = min(positive_peak.shape[0], y + 4)
            x0 = max(0, x - 3)
            x1 = min(positive_peak.shape[1], x + 4)
            value = float(np.sum(positive_peak[y0:y1, x0:x1]))
            present = value >= 0.006
            restored.append(present)
            window_sums.append(value)
            if bool(star.get("saturated", False)):
                saturated_restored.append(present)
        result.update(
            status="observed",
            available=True,
            reference_star_count=len(stars),
            restored_star_count=int(np.count_nonzero(restored)),
            recovery_ratio=float(np.mean(restored)),
            saturated_reference_count=len(saturated_restored),
            saturated_restored_count=int(np.count_nonzero(saturated_restored)),
            saturated_recovery_ratio=(
                float(np.mean(saturated_restored))
                if saturated_restored
                else None
            ),
            positive_delta_window_sum_median=float(np.median(window_sums)),
            positive_delta_window_sum_min=0.006,
            window_size_px=7,
        )
        return result
    except (IndexError, KeyError, TypeError, ValueError, FloatingPointError) as error:
        result["reason"] = str(error)
        return result


def _component_retention(
    support_mask: np.ndarray,
    catalog: Dict[str, Any],
    *,
    weak_only: bool,
) -> float:
    peak_y = np.asarray(catalog.get("_peak_y", ()), dtype=np.int32)
    peak_x = np.asarray(catalog.get("_peak_x", ()), dtype=np.int32)
    weak_flags = np.asarray(catalog.get("_weak_flags", ()), dtype=bool)
    if peak_y.size == 0 or peak_y.size != peak_x.size or peak_y.size != weak_flags.size:
        return 0.0
    selected = weak_flags if weak_only else np.ones_like(weak_flags, dtype=bool)
    if not np.any(selected):
        return 1.0
    covered = np.asarray(support_mask, dtype=bool)[peak_y[selected], peak_x[selected]]
    return float(np.mean(covered))


def _catalog_support_masks(
    catalog: Dict[str, Any],
    *,
    strict: bool,
    cfg: Any | None = None,
    extra_pixels: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if scipy_ndimage is None:
        raise RuntimeError("scipy.ndimage unavailable")
    structure = np.ones((3, 3), dtype=bool)
    weak_core = np.asarray(catalog["_weak_core_mask"], dtype=bool)
    bright_core = np.asarray(catalog["_bright_core_mask"], dtype=bool)
    if bool(catalog.get("source_matched", False)):
        # A source-confirmed one/two-pixel component is a real weak star. Strict
        # recovery may reduce bright halos, but must not strip the weak PSF wing.
        weak_iterations = 1
        bright_iterations = 2 if strict else 3
    else:
        weak_iterations = 0 if strict else 1
        bright_iterations = 1 if strict else 3
    weak_iterations = stage9_scale_radius(
        weak_iterations,
        catalog,
        rounding="ceil",
        minimum=0,
    )
    bright_iterations = stage9_scale_radius(
        bright_iterations,
        catalog,
        rounding="ceil",
        minimum=0,
    )
    weak_support = (
        weak_core.copy()
        if weak_iterations <= 0
        else scipy_ndimage.binary_dilation(
            weak_core,
            structure=structure,
            iterations=weak_iterations,
        )
    )
    bright_support = scipy_ndimage.binary_dilation(
        bright_core,
        structure=structure,
        iterations=bright_iterations,
    )

    source_fwhm = np.asarray(
        catalog.get(
            "_display_source_fwhm_px",
            catalog.get("_source_fwhm_px", ()),
        ),
        dtype=np.float32,
    )
    psf_valid = np.asarray(catalog.get("_psf_valid_flags", ()), dtype=bool)
    peak_y = np.asarray(catalog.get("_peak_y", ()), dtype=np.int32)
    peak_x = np.asarray(catalog.get("_peak_x", ()), dtype=np.int32)
    weak_flags = np.asarray(catalog.get("_weak_flags", ()), dtype=bool)
    if (
        source_fwhm.size > 0
        and source_fwhm.size == psf_valid.size
        and source_fwhm.size == peak_y.size
        and source_fwhm.size == peak_x.size
        and source_fwhm.size == weak_flags.size
    ):
        nominal_radius_max = int(
            _bounded(
                getattr(cfg, "stage9_psf_support_radius_max", 6),
                6,
                2,
                12,
            )
        )
        retry_pixels = max(0, min(int(extra_pixels), 2))
        radii = np.zeros(source_fwhm.size, dtype=np.int32)
        per_star_fwhm = np.asarray(
            catalog.get("_stage9_spatial_fwhm_px", ()),
            dtype=np.float32,
        )
        if per_star_fwhm.size != source_fwhm.size:
            per_star_fwhm = source_fwhm
        height, width = weak_core.shape
        weak_support = np.zeros_like(weak_core, dtype=bool)
        bright_support = np.zeros_like(bright_core, dtype=bool)
        for index in range(source_fwhm.size):
            geometry_fwhm = float(per_star_fwhm[index])
            if not math.isfinite(geometry_fwhm) or geometry_fwhm <= 0.0:
                geometry_fwhm = None
            radius_max = stage9_scale_radius(
                nominal_radius_max,
                catalog,
                fwhm_px=geometry_fwhm,
                rounding="ceil",
                minimum=1,
            )
            if bool(psf_valid[index]):
                base_radius = int(math.ceil(float(source_fwhm[index]) / 2.0))
                radius = base_radius
                if not strict:
                    radius += stage9_scale_radius(
                        1,
                        catalog,
                        fwhm_px=geometry_fwhm,
                        rounding="ceil",
                    )
                radius += stage9_scale_radius(
                    retry_pixels,
                    catalog,
                    fwhm_px=geometry_fwhm,
                    rounding="ceil",
                )
            elif bool(weak_flags[index]):
                radius = stage9_scale_radius(
                    1 if bool(catalog.get("source_matched", False)) else 0,
                    catalog,
                    fwhm_px=geometry_fwhm,
                    rounding="ceil",
                )
                radius += stage9_scale_radius(
                    retry_pixels,
                    catalog,
                    fwhm_px=geometry_fwhm,
                    rounding="ceil",
                )
            else:
                radius = stage9_scale_radius(
                    2 if strict else 3,
                    catalog,
                    fwhm_px=geometry_fwhm,
                    rounding="ceil",
                )
                radius += stage9_scale_radius(
                    retry_pixels,
                    catalog,
                    fwhm_px=geometry_fwhm,
                    rounding="ceil",
                )
            radius = max(1, min(radius_max, radius))
            radii[index] = radius
            y = int(peak_y[index])
            x = int(peak_x[index])
            y0 = max(0, y - radius)
            y1 = min(height, y + radius + 1)
            x0 = max(0, x - radius)
            x1 = min(width, x + radius + 1)
            grid_y, grid_x = np.ogrid[y0:y1, x0:x1]
            disk = (grid_y - y) ** 2 + (grid_x - x) ** 2 <= radius * radius
            target = weak_support if bool(weak_flags[index]) else bright_support
            target[y0:y1, x0:x1] |= disk
        active_radii = radii[psf_valid]
        effective_stats = stage9_effective_pixel_stats(
            active_radii if active_radii.size else radii
        )
        catalog.update(
            psf_support_policy="source_fwhm_scaled_v2",
            psf_support_strict=bool(strict),
            psf_support_retry_pixels=retry_pixels,
            psf_support_retry_pixels_nominal=retry_pixels,
            psf_support_radius_max=nominal_radius_max,
            psf_support_radius_max_nominal=nominal_radius_max,
            psf_support_radius_median_px=(
                float(np.median(active_radii)) if active_radii.size else None
            ),
            psf_support_radius_p95_px=(
                float(np.percentile(active_radii, 95.0))
                if active_radii.size
                else None
            ),
            psf_support_effective_radius_px=effective_stats,
            psf_support_geometry_anchor_fwhm_px=_STAGE9_FWHM_ANCHOR_PX,
        )
        catalog["_psf_support_radii"] = radii
    return weak_support, bright_support, weak_support | bright_support


def build_star_overlay_masks(
    catalog: Dict[str, Any],
    *,
    strict: bool,
    cfg: Any | None = None,
    extra_pixels: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return weak, bright and union masks for explicit top-layer composition."""
    return _catalog_support_masks(
        catalog,
        strict=strict,
        cfg=cfg,
        extra_pixels=extra_pixels,
    )


def apply_compact_starmask_support(
    stars: np.ndarray,
    support_mask: np.ndarray,
) -> np.ndarray:
    """Keep only connected compact star cores and their narrow wing support."""
    source = np.asarray(stars)
    mask = np.asarray(support_mask, dtype=bool)
    if source.ndim == 2 and source.shape == mask.shape:
        expanded_mask = mask
    elif source.ndim == 3 and source.shape[1:] == mask.shape:
        expanded_mask = mask[np.newaxis, ...]
    elif source.ndim == 3 and source.shape[:2] == mask.shape:
        expanded_mask = mask[..., np.newaxis]
    else:
        raise ValueError(
            "compact support shape mismatch: "
            f"stars={source.shape}, support={mask.shape}"
        )
    return np.where(expanded_mask, source, 0).astype(source.dtype, copy=False)


def _expanded_spatial_mask(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    source = np.asarray(values)
    spatial = np.asarray(mask)
    if source.ndim == 2 and source.shape == spatial.shape:
        return spatial
    if source.ndim == 3 and source.shape[1:] == spatial.shape:
        return spatial[np.newaxis, ...]
    if source.ndim == 3 and source.shape[:2] == spatial.shape:
        return spatial[..., np.newaxis]
    raise ValueError(
        "starmask spatial mask shape mismatch: "
        f"values={source.shape}, mask={spatial.shape}"
    )


def _asinh_map(values: np.ndarray, stretch: float, offset: float) -> np.ndarray:
    sample = np.asarray(values, dtype=np.float32)
    stretch = max(1.0, float(stretch))
    offset = max(0.0, float(offset))
    output = np.zeros_like(sample, dtype=np.float32)
    active = np.isfinite(sample) & (sample > offset) & (sample > 0.0)
    if not np.any(active):
        return output
    denominator = sample[active] * math.asinh(stretch)
    mapped = (
        (sample[active] - offset)
        * np.arcsinh(sample[active] * stretch)
        / np.maximum(denominator, 1e-12)
    )
    output[active] = np.clip(mapped, 0.0, 1.0)
    return output


def _color_preserving_asinh(
    normalized: np.ndarray,
    support_mask: np.ndarray,
    *,
    stretch: float,
    offset: float,
) -> np.ndarray:
    source = np.asarray(normalized, dtype=np.float32)
    peak = _pixel_peak(source)
    mapped_peak = _asinh_map(peak, stretch, offset)
    gain = np.divide(
        mapped_peak,
        np.maximum(peak, 1e-12),
        out=np.zeros_like(mapped_peak),
        where=peak > 0.0,
    )
    if source.ndim == 2:
        output = source * gain
    elif source.ndim == 3 and source.shape[0] <= 4:
        output = source * gain[np.newaxis, ...]
    elif source.ndim == 3 and source.shape[-1] <= 4:
        output = source * gain[..., np.newaxis]
    else:
        raise ValueError(f"unsupported starmask dimensions: {source.shape}")
    return np.where(_expanded_spatial_mask(source, support_mask), output, 0.0)


def _monotonic_anchor_map(
    values: np.ndarray,
    input_anchors: np.ndarray,
    output_anchors: np.ndarray,
) -> np.ndarray:
    """Map star signal through a strictly ordered log-input anchor curve."""
    sample = np.asarray(values, dtype=np.float32)
    inputs = np.asarray(input_anchors, dtype=np.float64)
    outputs = np.asarray(output_anchors, dtype=np.float64)
    if (
        inputs.ndim != 1
        or outputs.ndim != 1
        or inputs.size < 2
        or inputs.size != outputs.size
        or not np.all(np.isfinite(inputs))
        or not np.all(np.isfinite(outputs))
        or np.any(np.diff(inputs) <= 0.0)
        or np.any(np.diff(outputs) <= 0.0)
    ):
        raise ValueError("invalid monotonic star-curve anchors")
    mapped = np.zeros_like(sample, dtype=np.float32)
    active = np.isfinite(sample) & (sample > inputs[0])
    if np.any(active):
        mapped[active] = np.interp(
            np.log(np.maximum(sample[active], inputs[0])),
            np.log(inputs),
            outputs,
        ).astype(np.float32, copy=False)
    return np.clip(mapped, 0.0, float(outputs[-1]))


def _anchor_input_threshold(
    input_anchors: np.ndarray,
    output_anchors: np.ndarray,
    output_threshold: float,
) -> float:
    """Invert the monotonic anchor curve for coverage diagnostics."""
    inputs = np.asarray(input_anchors, dtype=np.float64)
    outputs = np.asarray(output_anchors, dtype=np.float64)
    threshold = float(output_threshold)
    if threshold <= outputs[0]:
        return float(inputs[0])
    if threshold >= outputs[-1]:
        return float(inputs[-1])
    return float(
        np.exp(np.interp(threshold, outputs, np.log(inputs)))
    )


def _color_preserving_multi_anchor_curve(
    normalized: np.ndarray,
    support_mask: np.ndarray,
    *,
    input_anchors: np.ndarray,
    output_anchors: np.ndarray,
) -> np.ndarray:
    """Apply one monotonic peak curve while preserving each pixel's RGB ratio."""
    source = np.asarray(normalized, dtype=np.float32)
    peak = _pixel_peak(source)
    mapped_peak = _monotonic_anchor_map(
        peak,
        input_anchors,
        output_anchors,
    )
    gain = np.divide(
        mapped_peak,
        np.maximum(peak, 1e-12),
        out=np.zeros_like(mapped_peak),
        where=peak > 0.0,
    )
    if source.ndim == 2:
        output = source * gain
    elif source.ndim == 3 and source.shape[0] <= 4:
        output = source * gain[np.newaxis, ...]
    elif source.ndim == 3 and source.shape[-1] <= 4:
        output = source * gain[..., np.newaxis]
    else:
        raise ValueError(f"unsupported starmask dimensions: {source.shape}")
    return np.where(
        _expanded_spatial_mask(source, support_mask),
        output,
        0.0,
    ).astype(np.float32, copy=False)


def _regularize_amplified_starmask_chroma(
    source: np.ndarray,
    mapped: np.ndarray,
    *,
    faint_input: float,
    bright_input: float,
    faint_chroma_max: float,
    bright_chroma_max: float,
) -> tuple[np.ndarray, Dict[str, float]]:
    """Stabilize faint star-wing colour before a high-gain layer is remixed.

    The starless subtraction can leave channel-dominant single pixels in faint
    wings. A multi-anchor curve makes those ratios visible even though their
    linear values were near the noise floor. Use an amplitude-weighted local
    colour estimate for faint pixels and cap only extreme channel spread; bright
    cores retain their measured RGB ratios.
    """
    source_arr = np.asarray(source, dtype=np.float32)
    mapped_arr = np.asarray(mapped, dtype=np.float32)
    if source_arr.shape != mapped_arr.shape or source_arr.ndim != 3:
        return mapped_arr, {
            "regularized_pixel_ratio": 0.0,
            "saturation_p99_before": 0.0,
            "saturation_p99_after": 0.0,
        }

    channel_first = source_arr.shape[0] <= 4
    channel_last = source_arr.shape[-1] <= 4
    if channel_first:
        source_rgb = source_arr[:3]
        mapped_rgb = mapped_arr[:3]
    elif channel_last:
        source_rgb = np.moveaxis(source_arr[..., :3], -1, 0)
        mapped_rgb = np.moveaxis(mapped_arr[..., :3], -1, 0)
    else:
        return mapped_arr, {
            "regularized_pixel_ratio": 0.0,
            "saturation_p99_before": 0.0,
            "saturation_p99_after": 0.0,
        }
    if source_rgb.shape[0] < 3:
        return mapped_arr, {
            "regularized_pixel_ratio": 0.0,
            "saturation_p99_before": 0.0,
            "saturation_p99_after": 0.0,
        }

    source_peak = np.max(source_rgb, axis=0)
    mapped_peak = np.max(mapped_rgb, axis=0)
    active = mapped_peak > 0.0
    raw_ratio = np.divide(
        source_rgb,
        np.maximum(source_peak, 1e-12)[np.newaxis, ...],
        out=np.zeros_like(source_rgb),
        where=source_peak[np.newaxis, ...] > 0.0,
    )

    local_rgb = []
    for channel in source_rgb:
        smoothed = _box_blur_gray(channel)
        local_rgb.append(_box_blur_gray(smoothed))
    local_rgb_arr = np.stack(local_rgb, axis=0).astype(np.float32, copy=False)
    local_peak = np.max(local_rgb_arr, axis=0)
    local_ratio = np.divide(
        local_rgb_arr,
        np.maximum(local_peak, 1e-12)[np.newaxis, ...],
        out=np.array(raw_ratio, copy=True),
        where=local_peak[np.newaxis, ...] > 0.0,
    )

    faint = max(float(faint_input), 1e-12)
    bright = max(float(bright_input), faint * 1.001)
    reliability = np.clip(
        (np.log(np.maximum(source_peak, 1e-12)) - math.log(faint))
        / max(math.log(bright) - math.log(faint), 1e-12),
        0.0,
        1.0,
    ).astype(np.float32, copy=False)
    ratio = (
        local_ratio * (1.0 - reliability)[np.newaxis, ...]
        + raw_ratio * reliability[np.newaxis, ...]
    )
    ratio_peak = np.max(ratio, axis=0)
    ratio = np.divide(
        ratio,
        np.maximum(ratio_peak, 1e-12)[np.newaxis, ...],
        out=np.zeros_like(ratio),
        where=ratio_peak[np.newaxis, ...] > 0.0,
    )

    faint_limit = _bounded(faint_chroma_max, 0.35, 0.10, 0.80)
    bright_limit = _bounded(bright_chroma_max, 0.60, faint_limit, 0.90)
    chroma_limit = (
        faint_limit + (bright_limit - faint_limit) * reliability
    ).astype(np.float32, copy=False)
    ratio_floor = np.min(ratio, axis=0)
    saturation = (1.0 - ratio_floor).astype(np.float32, copy=False)
    chroma_scale = np.minimum(
        1.0,
        np.divide(
            chroma_limit,
            np.maximum(saturation, 1e-12),
            out=np.ones_like(saturation),
            where=saturation > 0.0,
        ),
    )
    # Move channel ratios toward neutral white while keeping the strongest
    # channel pinned at 1.0. This makes the configured channel-spread limit an
    # actual upper bound after peak normalization.
    limited_ratio = 1.0 - (
        1.0 - ratio
    ) * chroma_scale[np.newaxis, ...]
    limited_peak = np.max(limited_ratio, axis=0)
    limited_ratio = np.divide(
        limited_ratio,
        np.maximum(limited_peak, 1e-12)[np.newaxis, ...],
        out=np.zeros_like(limited_ratio),
        where=limited_peak[np.newaxis, ...] > 0.0,
    )
    regularized_rgb = np.clip(
        limited_ratio * mapped_peak[np.newaxis, ...],
        0.0,
        1.0,
    )

    result = np.array(mapped_arr, copy=True)
    if channel_first:
        result[:3] = regularized_rgb
    else:
        result[..., :3] = np.moveaxis(regularized_rgb, 0, -1)

    before_saturation = np.divide(
        mapped_peak - np.min(mapped_rgb, axis=0),
        np.maximum(mapped_peak, 1e-12),
        out=np.zeros_like(mapped_peak),
        where=mapped_peak > 0.0,
    )
    after_peak = np.max(regularized_rgb, axis=0)
    after_saturation = np.divide(
        after_peak - np.min(regularized_rgb, axis=0),
        np.maximum(after_peak, 1e-12),
        out=np.zeros_like(after_peak),
        where=after_peak > 0.0,
    )
    changed = _pixel_peak(np.abs(result - mapped_arr)) > 1e-6
    return result.astype(np.float32, copy=False), {
        "regularized_pixel_ratio": float(np.mean(changed & active)),
        "saturation_p99_before": (
            float(np.percentile(before_saturation[active], 99.0))
            if np.any(active)
            else 0.0
        ),
        "saturation_p99_after": (
            float(np.percentile(after_saturation[active], 99.0))
            if np.any(active)
            else 0.0
        ),
    }


def apply_calibrated_starmask(
    stars: np.ndarray,
    calibration: Dict[str, Any],
) -> np.ndarray:
    """Build a rank-preserving multi-anchor nonlinear star layer."""
    source = np.asarray(stars)
    scale = _image_scale(source)
    normalized = _normalized(source, scale=scale)
    support_mask = calibration.get("_compact_support_mask")
    input_anchors = calibration.get("anchor_input_values")
    output_anchors = calibration.get("anchor_output_targets")
    if support_mask is None or input_anchors is None or output_anchors is None:
        raise ValueError("multi-anchor starmask calibration is incomplete")
    output = _color_preserving_multi_anchor_curve(
        normalized,
        np.asarray(support_mask, dtype=bool),
        input_anchors=np.asarray(input_anchors, dtype=np.float32),
        output_anchors=np.asarray(output_anchors, dtype=np.float32),
    )
    if bool(calibration.get("chroma_regularization_enabled", False)):
        output, _diagnostics = _regularize_amplified_starmask_chroma(
            normalized,
            output,
            faint_input=float(calibration.get("faint_value", input_anchors[1])),
            bright_input=float(calibration.get("bright_value", input_anchors[-2])),
            faint_chroma_max=float(
                calibration.get("faint_chroma_max", 0.35)
            ),
            bright_chroma_max=float(
                calibration.get("bright_chroma_max", 0.60)
            ),
        )
    if np.issubdtype(source.dtype, np.integer):
        info = np.iinfo(source.dtype)
        return np.rint(output * scale).clip(info.min, info.max).astype(source.dtype)
    return (output * scale).astype(source.dtype, copy=False)


def _stage9_psf_scale_shadow(
    positive_change: np.ndarray,
    star_reference: Dict[str, Any] | None,
    *,
    star_overlay_mask: np.ndarray | None,
) -> Dict[str, Any]:
    """Report source-confirmed PSF/scale closure without affecting gates."""
    if not isinstance(star_reference, dict) or star_reference.get("status") != "ok":
        return {
            "status": "unavailable",
            "reason": str(
                (star_reference or {}).get("reason")
                if isinstance(star_reference, dict)
                else "star reference catalog missing"
            ),
        }
    fwhm = np.asarray(star_reference.get("_source_fwhm_px", ()), dtype=np.float32)
    peak_y = np.asarray(star_reference.get("_peak_y", ()), dtype=np.int32)
    peak_x = np.asarray(star_reference.get("_peak_x", ()), dtype=np.int32)
    source_chroma = np.asarray(
        star_reference.get("_source_chroma", ()),
        dtype=np.float32,
    )
    if (
        fwhm.size < 4
        or peak_y.size != fwhm.size
        or peak_x.size != fwhm.size
        or source_chroma.shape != (fwhm.size, 3)
    ):
        return {
            "status": "unavailable",
            "reason": "source-confirmed PSF/chroma support is insufficient",
            "support_count": int(fwhm.size),
        }

    positive_peak = _pixel_peak(positive_change)
    positive_rgb = _rgb_channels(positive_change)
    if np.any(peak_y < 0) or np.any(peak_x < 0):
        return {"status": "unavailable", "reason": "invalid star coordinates"}
    if np.any(peak_y >= positive_peak.shape[0]) or np.any(peak_x >= positive_peak.shape[1]):
        return {"status": "unavailable", "reason": "star coordinates outside candidate"}

    component_mask = positive_peak > _STAR_RECOVERY_DELTA
    _labels, component_count, component_areas = _component_areas(component_mask)
    fwhm_median = max(float(np.median(fwhm)), 1e-6)
    normalized_areas = (
        component_areas.astype(np.float32) / (fwhm_median * fwhm_median)
        if component_areas.size
        else np.asarray([], dtype=np.float32)
    )
    candidate_samples = np.moveaxis(
        positive_rgb[:, peak_y, peak_x],
        0,
        -1,
    )
    candidate_sum = np.sum(candidate_samples, axis=1, keepdims=True)
    candidate_chroma = np.divide(
        candidate_samples,
        np.maximum(candidate_sum, 1e-12),
        out=np.zeros_like(candidate_samples),
        where=candidate_sum > 1e-12,
    )
    chroma_error = np.max(np.abs(candidate_chroma - source_chroma), axis=1)
    aperture7 = (
        scipy_ndimage.uniform_filter(
            positive_peak,
            size=7,
            mode="constant",
        )
        * 49.0
        if scipy_ndimage is not None
        else positive_peak
    )
    outside_change_ratio = None
    outside_change_status = "unavailable"
    outside_change_reason = "source-confirmed support mask is unavailable"
    if star_overlay_mask is not None:
        support = np.asarray(star_overlay_mask, dtype=bool)
        if support.shape == positive_peak.shape:
            outside_change_ratio = float(
                np.mean(component_mask & ~support)
            )
            outside_change_status = "shadow"
            outside_change_reason = ""
        else:
            outside_change_reason = "source-confirmed support mask shape mismatch"
    component_area_status = "shadow" if normalized_areas.size else "unavailable"
    return {
        "status": "shadow",
        "support_count": int(fwhm.size),
        "source_fwhm_median_px": fwhm_median,
        "source_fwhm_p25_px": float(np.percentile(fwhm, 25.0)),
        "source_fwhm_p75_px": float(np.percentile(fwhm, 75.0)),
        "positive_component_count": int(component_count),
        "positive_component_area_over_fwhm2_median": (
            float(np.median(normalized_areas)) if normalized_areas.size else None
        ),
        "positive_component_area_over_fwhm2_p99": (
            float(np.percentile(normalized_areas, 99.0))
            if normalized_areas.size
            else None
        ),
        "positive_component_area_over_fwhm2_status": component_area_status,
        "positive_component_area_over_fwhm2_reason": (
            "" if normalized_areas.size else "no positive recovery components"
        ),
        "source_chroma_error_median": float(np.median(chroma_error)),
        "source_chroma_error_p99": float(np.percentile(chroma_error, 99.0)),
        "aperture7_median": float(np.median(aperture7[peak_y, peak_x])),
        "outside_confirmed_star_change_ratio": outside_change_ratio,
        "outside_confirmed_star_change_status": outside_change_status,
        "outside_confirmed_star_change_reason": outside_change_reason,
    }


def assess_stage9_psf_closure(
    candidate_image: np.ndarray,
    star_reference: Dict[str, Any] | None,
    cfg: Any,
) -> Dict[str, Any]:
    """Measure same-star display-domain FWHM and enforce the natural-size gate."""
    enabled = bool(getattr(cfg, "stage9_psf_size_gate_enabled", True))
    result: Dict[str, Any] = {
        "schema": "starun.stage9-psf-closure.v2",
        "status": "unavailable",
        "available": False,
        "accepted": True,
        "review_required": False,
        "gate_enabled": enabled,
        "groups": {},
    }
    if not isinstance(star_reference, dict) or star_reference.get("status") != "ok":
        result["reason"] = "star reference catalog missing"
        return result
    reference_status = str(
        star_reference.get("psf_reference_status") or "unavailable"
    )
    if reference_status not in {"ready", "partial"}:
        result["reason"] = str(
            star_reference.get("psf_reference_reason")
            or "matched-domain PSF reference unavailable"
        )
        result["reference_sample_count"] = int(
            star_reference.get("psf_sample_count", 0) or 0
        )
        return result

    source_fwhm = np.asarray(
        star_reference.get("_display_source_fwhm_px", ()),
        dtype=np.float32,
    )
    valid = np.asarray(star_reference.get("_psf_valid_flags", ()), dtype=bool)
    peak_y = np.asarray(star_reference.get("_peak_y", ()), dtype=np.int32)
    peak_x = np.asarray(star_reference.get("_peak_x", ()), dtype=np.int32)
    weak_flags = np.asarray(star_reference.get("_weak_flags", ()), dtype=bool)
    saturated_source = np.asarray(
        star_reference.get("_psf_saturated_flags", ()),
        dtype=bool,
    )
    source_halfmax_area = np.asarray(
        star_reference.get("_display_source_halfmax_area_px", ()),
        dtype=np.float32,
    )
    count = int(source_fwhm.size)
    if not (
        count > 0
        and valid.size == count
        and peak_y.size == count
        and peak_x.size == count
        and weak_flags.size == count
    ):
        result["reason"] = "PSF reference arrays are incomplete"
        return result

    saturation_classified = saturated_source.size == count
    if not saturation_classified:
        saturated_source = np.zeros(count, dtype=bool)
    if source_halfmax_area.size != count:
        source_halfmax_area = np.full(count, np.nan, dtype=np.float32)

    # Compare the actual display-domain candidate against O.  Measuring only
    # ``candidate - remix_base`` would omit the retained local background and
    # any trusted source flux already present in B, so it can report a star as
    # artificially small even when the delivered PSF is natural.
    candidate_peak = _pixel_peak(_normalized(np.asarray(candidate_image)))
    candidate_fwhm = np.full(count, np.nan, dtype=np.float32)
    candidate_halfmax_area = np.full(count, np.nan, dtype=np.float32)
    candidate_valid = np.zeros(count, dtype=bool)
    candidate_saturated = np.zeros(count, dtype=bool)
    per_star_fwhm = np.asarray(
        star_reference.get("_stage9_spatial_fwhm_px", ()),
        dtype=np.float32,
    )
    if per_star_fwhm.size != count:
        per_star_fwhm = source_fwhm
    measurement_search_radii = []
    measurement_patch_radii = []
    for index in np.flatnonzero(valid):
        geometry_fwhm = float(per_star_fwhm[index])
        search_radius = stage9_scale_radius(
            2,
            star_reference,
            fwhm_px=geometry_fwhm,
            rounding="nearest",
            minimum=1,
        )
        patch_radius = stage9_scale_radius(
            6,
            star_reference,
            fwhm_px=geometry_fwhm,
            rounding="nearest",
            minimum=1,
        )
        measurement_search_radii.append(search_radius)
        measurement_patch_radii.append(patch_radius)
        measurement = _measure_connected_halfmax_fwhm(
            candidate_peak,
            int(peak_y[index]),
            int(peak_x[index]),
            search_radius=search_radius,
            patch_radius=patch_radius,
        )
        if measurement.get("status") != "ok":
            continue
        if (
            abs(int(measurement.get("offset_y", 99))) > search_radius
            or abs(int(measurement.get("offset_x", 99))) > search_radius
        ):
            continue
        candidate_fwhm[index] = float(measurement["fwhm_px"])
        candidate_halfmax_area[index] = float(measurement["half_max_area"])
        candidate_saturated[index] = bool(measurement.get("saturated", False))
        candidate_valid[index] = True

    ordinary_reference = (
        valid & ~saturated_source
        if saturation_classified
        else np.zeros(count, dtype=bool)
    )
    matched = (
        ordinary_reference
        & candidate_valid
        & np.isfinite(source_fwhm)
        & np.isfinite(candidate_fwhm)
        & (source_fwhm > 0.0)
        & (candidate_fwhm > 0.0)
    )
    min_count = int(
        _bounded(
            getattr(cfg, "stage9_psf_min_sample_count", 16),
            16,
            4,
            256,
        )
    )
    reference_count = int(np.count_nonzero(ordinary_reference))
    matched_count = int(np.count_nonzero(matched))
    result.update(
        reference_status=reference_status,
        saturation_classification_status=(
            "available" if saturation_classified else "unavailable"
        ),
        reference_total_sample_count=int(np.count_nonzero(valid)),
        reference_sample_count=reference_count,
        candidate_sample_count=matched_count,
        minimum_sample_count=min_count,
        measurement_search_radius_px_nominal=2,
        measurement_search_radius_px_effective=stage9_effective_pixel_stats(
            measurement_search_radii
        ),
        measurement_patch_radius_px_nominal=6,
        measurement_patch_radius_px_effective=stage9_effective_pixel_stats(
            measurement_patch_radii
        ),
    )
    ratio_min = _bounded(
        getattr(cfg, "stage9_psf_fwhm_ratio_min", 0.93),
        0.93,
        0.50,
        1.00,
    )
    ratio_max = _bounded(
        getattr(cfg, "stage9_psf_fwhm_ratio_max", 1.10),
        1.10,
        1.00,
        1.50,
    )
    subgroup_min_count = max(4, int(math.ceil(min_count * 0.20)))
    issues: list[str] = []
    partial_groups: list[str] = []
    diagnostic_incomplete_groups: list[str] = []
    metrics: Dict[str, float] = {}
    group_masks = {
        "all": (ordinary_reference, matched),
        "weak": (ordinary_reference & weak_flags, matched & weak_flags),
        "bright": (ordinary_reference & ~weak_flags, matched & ~weak_flags),
    }
    for group_name, (reference_mask, candidate_mask) in group_masks.items():
        group_reference_count = int(np.count_nonzero(reference_mask))
        group_count = int(np.count_nonzero(candidate_mask))
        required = min_count if group_name == "all" else subgroup_min_count
        metrics[f"star_psf_fwhm_reference_sample_count_{group_name}"] = float(
            group_reference_count
        )
        metrics[f"star_psf_fwhm_sample_count_{group_name}"] = float(group_count)
        if group_reference_count < required:
            result["groups"][group_name] = {
                "status": "not_assessed",
                "reference_sample_count": group_reference_count,
                "candidate_sample_count": group_count,
                "minimum_sample_count": required,
                "reason": (
                    "insufficient isolated unsaturated reference stars"
                    if saturation_classified
                    else "reference saturation classification unavailable"
                ),
            }
            partial_groups.append(group_name)
            continue
        if group_count < required:
            result["groups"][group_name] = {
                "status": "insufficient",
                "reference_sample_count": group_reference_count,
                "candidate_sample_count": group_count,
                "minimum_sample_count": required,
                "reason": "candidate lost measurable same-star samples",
            }
            diagnostic_incomplete_groups.append(group_name)
            if enabled:
                issues.append(
                    f"star_psf_fwhm_sample_count_{group_name} "
                    f"{group_count}<{required}"
                )
            continue
        ratios = candidate_fwhm[candidate_mask] / source_fwhm[candidate_mask]
        ratio_median = float(np.median(ratios))
        group_report = {
            "status": "ok",
            "reference_sample_count": group_reference_count,
            "candidate_sample_count": group_count,
            "minimum_sample_count": required,
            "source_fwhm_median_px": float(np.median(source_fwhm[candidate_mask])),
            "candidate_fwhm_median_px": float(
                np.median(candidate_fwhm[candidate_mask])
            ),
            "fwhm_ratio_median": ratio_median,
            "fwhm_ratio_p25": float(np.percentile(ratios, 25.0)),
            "fwhm_ratio_p75": float(np.percentile(ratios, 75.0)),
            "ratio_min": ratio_min,
            "ratio_max": ratio_max,
            "accepted": bool(ratio_min <= ratio_median <= ratio_max),
        }
        result["groups"][group_name] = group_report
        metrics[f"star_psf_fwhm_ratio_{group_name}"] = ratio_median
        if enabled and not group_report["accepted"]:
            issues.append(
                f"star_psf_fwhm_ratio_{group_name} "
                f"{ratio_median:.6f} outside {ratio_min:.6f}..{ratio_max:.6f}"
            )

    saturated_mask = valid & saturated_source if saturation_classified else np.zeros(
        count,
        dtype=bool,
    )
    saturated_matched = saturated_mask & candidate_valid
    saturated_count = int(np.count_nonzero(saturated_mask))
    saturated_matched_count = int(np.count_nonzero(saturated_matched))
    metrics["star_psf_saturated_reference_count"] = float(saturated_count)
    metrics["star_psf_saturated_candidate_measurable_count"] = float(
        saturated_matched_count
    )
    if not saturation_classified:
        result["groups"]["saturated"] = {
            "status": "not_assessed",
            "reason": "reference saturation classification unavailable",
        }
    elif saturated_count <= 0:
        result["groups"]["saturated"] = {
            "status": "not_present",
            "reference_sample_count": 0,
            "candidate_sample_count": 0,
            "gate_role": "observation_only",
        }
    else:
        saturated_report: Dict[str, Any] = {
            "status": "observed",
            "reference_sample_count": saturated_count,
            "candidate_sample_count": saturated_matched_count,
            "candidate_saturated_sample_count": int(
                np.count_nonzero(saturated_matched & candidate_saturated)
            ),
            "gate_role": "observation_only",
        }
        area_valid = (
            saturated_matched
            & np.isfinite(source_halfmax_area)
            & np.isfinite(candidate_halfmax_area)
            & (source_halfmax_area > 0.0)
            & (candidate_halfmax_area > 0.0)
        )
        if np.any(area_valid):
            area_ratios = (
                candidate_halfmax_area[area_valid]
                / source_halfmax_area[area_valid]
            )
            area_ratio_median = float(np.median(area_ratios))
            saturated_report.update(
                halfmax_area_sample_count=int(np.count_nonzero(area_valid)),
                halfmax_area_ratio_median=area_ratio_median,
            )
            metrics["star_psf_saturated_halfmax_area_ratio_median"] = (
                area_ratio_median
            )
        else:
            saturated_report.update(
                halfmax_area_sample_count=0,
                halfmax_area_ratio_median=None,
            )
        result["groups"]["saturated"] = saturated_report

    accepted = not issues
    review_required = bool(enabled and accepted and partial_groups)
    if issues:
        status = "rejected"
    elif partial_groups or diagnostic_incomplete_groups:
        status = "partial"
    else:
        status = "ok"
    advisories = []
    if partial_groups:
        advisories.append(
            "PSF groups not assessed: " + ", ".join(partial_groups)
        )
    if diagnostic_incomplete_groups and not enabled:
        advisories.append(
            "PSF candidate samples incomplete while gate disabled: "
            + ", ".join(diagnostic_incomplete_groups)
        )
    result.update(
        status=status,
        available=True,
        accepted=accepted,
        review_required=review_required,
        review_reason_codes=(
            ["STAGE9_PSF_SUBGROUP_EVIDENCE_INSUFFICIENT"]
            if review_required
            else []
        ),
        advisories=advisories,
        issues=issues,
        metrics=metrics,
        limits={
            "stage9_psf_fwhm_ratio_min": ratio_min,
            "stage9_psf_fwhm_ratio_max": ratio_max,
        },
    )
    return result


def _solve_asinh_input_threshold(
    stretch: float,
    offset: float,
    output_target: float,
) -> float:
    """Find the input value whose transformed output reaches output_target."""
    target = _bounded(output_target, 0.002, 1e-7, 0.95)
    low = max(0.0, float(offset))
    high = 1.0
    if _asinh_sample(high, stretch, offset) <= target:
        return high
    for _ in range(40):
        middle = (low + high) * 0.5
        if _asinh_sample(middle, stretch, offset) < target:
            low = middle
        else:
            high = middle
    return high


def _predicted_change_coverage(
    peak_map: np.ndarray,
    *,
    stretch: float,
    offset: float,
    intensity: float,
) -> Tuple[float, float]:
    """Conservatively predict pixels changed by more than the Stage 9 gate delta."""
    reference_intensity = _bounded(intensity, 1.0, 0.10, 1.05)
    input_threshold = _solve_asinh_input_threshold(
        stretch,
        offset,
        0.002 / reference_intensity,
    )
    coverage = float(np.mean(np.asarray(peak_map) > input_threshold))
    return coverage, float(input_threshold)


def _coverage_limited_stretch(
    peak_map: np.ndarray,
    *,
    requested_stretch: float,
    offset: float,
    intensity: float,
    coverage_limit: float,
) -> Tuple[float, float, float, bool]:
    """Cap Asinh strength so predicted wide-field changes stay inside the gate."""
    requested = max(1.10, float(requested_stretch))
    requested_coverage, requested_threshold = _predicted_change_coverage(
        peak_map,
        stretch=requested,
        offset=offset,
        intensity=intensity,
    )
    if requested_coverage <= coverage_limit:
        return requested, requested_coverage, requested_threshold, False

    low = 1.10
    low_coverage, low_threshold = _predicted_change_coverage(
        peak_map,
        stretch=low,
        offset=offset,
        intensity=intensity,
    )
    if low_coverage > coverage_limit:
        return low, low_coverage, low_threshold, True
    high = requested
    best_stretch = low
    best_coverage = low_coverage
    best_threshold = low_threshold
    for _ in range(32):
        middle = (low + high) * 0.5
        coverage, input_threshold = _predicted_change_coverage(
            peak_map,
            stretch=middle,
            offset=offset,
            intensity=intensity,
        )
        if coverage <= coverage_limit:
            best_stretch = middle
            best_coverage = coverage
            best_threshold = input_threshold
            low = middle
        else:
            high = middle
    return best_stretch, best_coverage, best_threshold, True


def _coverage_limited_anchor_targets(
    normalized: np.ndarray,
    support_mask: np.ndarray,
    *,
    input_anchors: np.ndarray,
    nominal_output_anchors: np.ndarray,
    intensity: float,
    coverage_limit: float,
) -> Tuple[np.ndarray, np.ndarray, float, float, bool]:
    """Uniformly scale a monotonic star curve to the largest safe coverage."""
    reference_intensity = _bounded(intensity, 1.0, 0.10, 1.05)

    def preview_for(scale: float) -> Tuple[np.ndarray, float]:
        if scale <= 0.0:
            preview = np.zeros_like(normalized, dtype=np.float32)
        else:
            anchors = np.asarray(nominal_output_anchors, dtype=np.float64).copy()
            anchors[1:] *= float(scale)
            preview = _color_preserving_multi_anchor_curve(
                normalized,
                support_mask,
                input_anchors=input_anchors,
                output_anchors=anchors,
            )
        coverage = float(
            np.mean(
                _pixel_peak(preview) * reference_intensity
                > _STAR_RECOVERY_DELTA
            )
        )
        return preview, coverage

    nominal_preview, nominal_coverage = preview_for(1.0)
    if nominal_coverage <= coverage_limit:
        return (
            np.asarray(nominal_output_anchors, dtype=np.float32),
            nominal_preview,
            1.0,
            nominal_coverage,
            False,
        )

    low = 0.0
    high = 1.0
    best_scale = 0.0
    best_preview, best_coverage = preview_for(0.0)
    for _ in range(40):
        middle = (low + high) * 0.5
        preview, coverage = preview_for(middle)
        if coverage <= coverage_limit:
            best_scale = middle
            best_preview = preview
            best_coverage = coverage
            low = middle
        else:
            high = middle
    scaled = np.asarray(nominal_output_anchors, dtype=np.float64).copy()
    scaled[1:] *= best_scale
    return (
        scaled.astype(np.float32),
        best_preview,
        float(best_scale),
        float(best_coverage),
        True,
    )


def calibrate_starmask_asinh(
    stars: np.ndarray,
    cfg: Any,
    *,
    include_support_mask: bool = False,
    strict_support: bool = False,
    support_retry_pixels: int = 0,
    reference_catalog: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Derive a bounded star-layer Asinh from effective faint/bright star samples."""
    normalized = _normalized(np.asarray(stars))
    if normalized.ndim not in (2, 3):
        return {"status": "unavailable", "reason": "invalid starmask dimensions"}
    gray = _gray(normalized)
    finite = gray[np.isfinite(gray)]
    if finite.size < 64:
        return {"status": "unavailable", "reason": "insufficient finite pixels"}

    background = float(np.percentile(finite, 50.0))
    low_samples = finite[finite <= np.percentile(finite, 70.0)]
    if low_samples.size < 32:
        low_samples = finite
    mad = float(np.median(np.abs(low_samples - np.median(low_samples))))
    noise_sigma = max(1.4826 * mad, 1e-7)
    signal_floor = max(background + 3.0 * noise_sigma, 1e-6)
    catalog = reference_catalog or build_star_reference_catalog(
        stars,
        cfg,
        background=background,
        noise_sigma=noise_sigma,
    )
    if catalog.get("status") != "ok":
        return {
            "status": "unavailable",
            "reason": str(catalog.get("reason") or "star reference unavailable"),
            "star_reference": star_reference_summary(catalog),
        }

    initial_support = _compact_star_support(
        gray,
        background=background,
        noise_sigma=noise_sigma,
        strict=False,
    )
    initial_support_mask = (
        np.asarray(initial_support["mask"], dtype=bool)
        if initial_support.get("status") == "ok"
        else np.zeros_like(gray, dtype=bool)
    )
    initial_weak_retention = _component_retention(
        initial_support_mask,
        catalog,
        weak_only=True,
    )
    min_weak_retention = _bounded(
        getattr(cfg, "stage9_compact_weak_star_retention_min", 0.80),
        0.80,
        0.50,
        0.98,
    )
    mixed_star_field = bool(catalog.get("mixed_star_field", False))
    rebuild_support = bool(
        strict_support
        or mixed_star_field
        or initial_weak_retention < min_weak_retention
        or initial_support.get("status") != "ok"
    )
    weak_support, bright_support, catalog_support = _catalog_support_masks(
        catalog,
        strict=strict_support,
        cfg=cfg,
        extra_pixels=support_retry_pixels,
    )
    # Use the same reference catalog for both normal and strict support so the
    # strict weak=0/bright=1 wings are genuinely narrower than weak=1/bright=3.
    # The legacy support remains diagnostic input for deciding whether recovery
    # was needed, but it is not allowed to silently select a different star set.
    support_mask = catalog_support
    weak_retention = _component_retention(
        support_mask,
        catalog,
        weak_only=True,
    )
    total_retention = _component_retention(
        support_mask,
        catalog,
        weak_only=False,
    )
    weak_retention_gate = stage7_quality.stage7_9_lower_quality_gate(
        cfg,
        value=weak_retention,
        accepted_limit=min_weak_retention,
    )
    calibration_advisories: list[str] = []
    if weak_retention_gate["hard_failed"]:
        return {
            "status": "rejected",
            "reason": (
                "compact_weak_star_retention "
                f"{weak_retention:.3f}<{min_weak_retention:.3f}"
            ),
            "weak_star_retention": weak_retention,
            "weak_star_retention_min": min_weak_retention,
            "quality_gates": {
                "compact_weak_star_retention": weak_retention_gate,
            },
            "star_reference": star_reference_summary(catalog),
        }
    if weak_retention_gate["advisory"]:
        calibration_advisories.append(
            "compact_weak_star_retention "
            f"{weak_retention:.3f}<{min_weak_retention:.3f} "
            "(advisory; calibration continued)"
        )

    input_peak_map = _pixel_peak(normalized)
    profile_sample_mask = (
        support_mask & np.isfinite(input_peak_map) & (gray > signal_floor)
    )
    signal = input_peak_map[profile_sample_mask]
    if signal.size < 8:
        return {
            "status": "unavailable",
            "reason": "compact star core/wing samples unavailable",
        }

    component_peaks = np.asarray(catalog.get("_component_peaks", ()), dtype=np.float32)
    weak_flags = np.asarray(catalog.get("_weak_flags", ()), dtype=bool)
    if mixed_star_field and component_peaks.size >= 8 and np.any(weak_flags):
        # These percentiles correspond to the old lower-80% median, weak/bright
        # boundary, upper-20% median and extreme highlight. Mapping all four on
        # one curve prevents independently stretched weak stars from overtaking
        # intrinsically brighter stars.
        faint_value, mid_value, bright_value, peak_value = (
            float(value)
            for value in np.percentile(component_peaks, (40.0, 80.0, 90.0, 99.7))
        )
    else:
        faint_value = float(np.percentile(signal, 50.0))
        mid_value = float(np.percentile(signal, 75.0))
        bright_value = float(np.percentile(signal, 90.0))
        peak_value = float(np.percentile(signal, 99.7))
    configured_offset = _bounded(
        getattr(cfg, "stage9_starmask_asinh_offset", 0.001),
        0.001,
        0.00001,
        0.006,
    )
    offset_cap = max(0.00001, min(signal_floor * 0.80, faint_value * 0.35))
    offset = min(configured_offset, offset_cap)
    stretch_max = _bounded(
        getattr(cfg, "stage9_starmask_asinh_stretch_max", 1000.0),
        1000.0,
        10.0,
        1000.0,
    )
    output_targets = _stage9_starmask_output_targets(cfg)
    faint_target = output_targets["faint"]
    mid_target = output_targets["mid"]
    bright_target = output_targets["bright"]
    peak_target = output_targets["peak"]
    input_anchor_values = {
        "faint": faint_value,
        "mid": mid_value,
        "bright": bright_value,
        "peak": peak_value,
    }
    limited_stretches = {
        name: _solve_asinh_stretch(
            input_anchor_values[name],
            offset,
            output_targets[name],
            stretch_max,
        )
        for name in _STAGE9_STARMASK_OUTPUT_NAMES
    }
    faint_limited = limited_stretches["faint"]
    peak_limited = limited_stretches["peak"]
    output_limited_stretch = min(limited_stretches.values())
    adaptive_enabled = bool(
        getattr(cfg, "stage9_starmask_adaptive_stretch_enabled", True)
    )
    multi_anchor_enabled = bool(
        adaptive_enabled
        and mixed_star_field
        and component_peaks.size >= 8
        and faint_value > offset
        and faint_value < mid_value < bright_value < peak_value
    )
    chroma_regularization_enabled = bool(
        getattr(cfg, "stage9_starmask_chroma_regularization_enabled", True)
    )
    faint_chroma_max = _bounded(
        getattr(cfg, "stage9_starmask_faint_chroma_max", 0.35),
        0.35,
        0.10,
        0.80,
    )
    bright_chroma_max = _bounded(
        getattr(cfg, "stage9_starmask_bright_chroma_max", 0.60),
        0.60,
        faint_chroma_max,
        0.90,
    )
    chroma_regularization = {
        "regularized_pixel_ratio": 0.0,
        "saturation_p99_before": 0.0,
        "saturation_p99_after": 0.0,
    }
    if adaptive_enabled:
        configured_stretch_proposal = None
        target_stretch = _bounded(
            output_limited_stretch,
            2.0,
            1.10,
            stretch_max,
        )
    else:
        configured_stretch_proposal = _bounded(
            getattr(cfg, "stage9_starmask_asinh_stretch", 2.0),
            2.0,
            1.10,
            3.0,
        )
        target_stretch = min(
            configured_stretch_proposal,
            max(1.10, output_limited_stretch),
        )
    configured_coverage_limit = _bounded(
        getattr(cfg, "stage9_starmask_predicted_change_ratio_max", 0.30),
        0.30,
        0.05,
        0.60,
    )
    gate_coverage_limit = _bounded(
        getattr(cfg, "stage9_changed_pixel_ratio_max", 0.35),
        0.35,
        0.05,
        0.80,
    )
    coverage_limit = min(configured_coverage_limit, gate_coverage_limit * 0.90)
    reference_intensity = _bounded(
        getattr(cfg, "star_intensity", 1.05),
        1.05,
        0.10,
        1.05,
    )
    compact_normalized = apply_compact_starmask_support(normalized, support_mask)
    coverage_peak_map = _pixel_peak(compact_normalized)
    anchor_input_values = None
    anchor_output_targets = None
    output_target_scale = 1.0
    if multi_anchor_enabled:
        anchor_input_values = np.asarray(
            [offset, faint_value, mid_value, bright_value, peak_value],
            dtype=np.float32,
        )
        nominal_anchor_output_targets = np.asarray(
            [0.0, faint_target, mid_target, bright_target, peak_target],
            dtype=np.float32,
        )
        (
            anchor_output_targets,
            multi_anchor_preview,
            output_target_scale,
            predicted_coverage,
            coverage_limited,
        ) = _coverage_limited_anchor_targets(
            normalized,
            support_mask,
            input_anchors=anchor_input_values,
            nominal_output_anchors=nominal_anchor_output_targets,
            intensity=reference_intensity,
            coverage_limit=coverage_limit,
        )
        if chroma_regularization_enabled:
            multi_anchor_preview, chroma_regularization = (
                _regularize_amplified_starmask_chroma(
                    normalized,
                    multi_anchor_preview,
                    faint_input=faint_value,
                    bright_input=bright_value,
                    faint_chroma_max=faint_chroma_max,
                    bright_chroma_max=bright_chroma_max,
                )
            )
        predicted_coverage = float(
            np.mean(
                _pixel_peak(multi_anchor_preview) * reference_intensity
                > _STAR_RECOVERY_DELTA
            )
        )
        output_change_threshold = _STAR_RECOVERY_DELTA / reference_intensity
        change_input_threshold = (
            1.0
            if float(anchor_output_targets[-1]) <= output_change_threshold
            else _anchor_input_threshold(
                anchor_input_values,
                anchor_output_targets,
                output_change_threshold,
            )
        )
        stretch = float(output_limited_stretch)
        output_preview = multi_anchor_preview
    else:
        stretch, predicted_coverage, change_input_threshold, coverage_limited = (
            _coverage_limited_stretch(
                coverage_peak_map,
                requested_stretch=target_stretch,
                offset=offset,
                intensity=reference_intensity,
                coverage_limit=coverage_limit,
            )
        )
        output_preview = _color_preserving_asinh(
            normalized,
            support_mask,
            stretch=stretch,
            offset=offset,
        )
    output_profile_mode = (
        "mixed_star_peak_percentiles"
        if mixed_star_field and component_peaks.size >= 8
        else "ordinary_support_pixel_percentiles"
    )
    output_target_limited = False
    if multi_anchor_enabled:
        preliminary_profile = measure_starmask_output_profile(
            output_preview,
            {
                "output_profile_mode": output_profile_mode,
                "output_targets": output_targets,
                "_output_profile_sample_mask": profile_sample_mask,
                "_star_reference_catalog": catalog,
            },
            source="builtin_multi_anchor_preliminary",
        )
        preliminary_actual = dict(preliminary_profile.get("actual") or {})
        scale_cap = min(
            (
                output_targets[name] / preliminary_actual[name]
                if preliminary_actual.get(name, 0.0) > output_targets[name]
                else 1.0
            )
            for name in _STAGE9_STARMASK_OUTPUT_NAMES
        )
        if scale_cap < 1.0:
            output_target_limited = True
            output_target_scale *= scale_cap
            anchor_output_targets = np.asarray(
                anchor_output_targets,
                dtype=np.float64,
            )
            anchor_output_targets[1:] *= scale_cap
            anchor_output_targets = anchor_output_targets.astype(np.float32)
            output_preview = _color_preserving_multi_anchor_curve(
                normalized,
                support_mask,
                input_anchors=anchor_input_values,
                output_anchors=anchor_output_targets,
            )
            if chroma_regularization_enabled:
                output_preview, chroma_regularization = (
                    _regularize_amplified_starmask_chroma(
                        normalized,
                        output_preview,
                        faint_input=faint_value,
                        bright_input=bright_value,
                        faint_chroma_max=faint_chroma_max,
                        bright_chroma_max=bright_chroma_max,
                    )
                )
            predicted_coverage = float(
                np.mean(
                    _pixel_peak(output_preview) * reference_intensity
                    > _STAR_RECOVERY_DELTA
                )
            )
            output_change_threshold = _STAR_RECOVERY_DELTA / reference_intensity
            change_input_threshold = (
                1.0
                if float(anchor_output_targets[-1]) <= output_change_threshold
                else _anchor_input_threshold(
                    anchor_input_values,
                    anchor_output_targets,
                    output_change_threshold,
                )
            )
    output_profile_context = {
        "mixed_star_field": mixed_star_field,
        "output_profile_mode": output_profile_mode,
        "output_targets": output_targets,
        "faint_target": faint_target,
        "mid_target": mid_target,
        "bright_target": bright_target,
        "peak_target": peak_target,
        "_output_profile_sample_mask": profile_sample_mask,
        "_star_reference_catalog": catalog,
    }
    output_profile = measure_starmask_output_profile(
        output_preview,
        output_profile_context,
        source=(
            "builtin_multi_anchor_preview"
            if multi_anchor_enabled
            else "builtin_asinh_preview"
        ),
    )
    output_profile_gate = _starmask_output_profile_gate(output_profile)
    coverage_usable = bool(predicted_coverage <= coverage_limit + 1e-12)
    curve_usable = bool(output_profile_gate["accepted"] and coverage_usable)
    rejection_codes: list[str] = []
    if not output_profile_gate["accepted"]:
        rejection_codes.append(
            "stage9_starmask_output_target_exceeded_at_minimum_stretch"
            if not multi_anchor_enabled and output_limited_stretch < 1.10
            else str(
                output_profile.get("reason_code")
                or "stage9_starmask_output_target_exceeded"
            )
        )
    if not coverage_usable:
        rejection_codes.append("stage9_starmask_change_coverage_unavailable")
    bright_stretch = _bounded(peak_limited, 1.0, 1.0, stretch_max)
    source_matched = bool(catalog.get("source_matched", False))
    bright_wing_iterations = (
        2 if strict_support else 3
    ) if source_matched else (1 if strict_support else 3)
    weak_wing_iterations = 1 if source_matched else (0 if strict_support else 1)
    full_layer_predicted_coverage = float(
        np.mean(_pixel_peak(normalized) > change_input_threshold)
    )
    support_mode = (
        "strict_recovery"
        if strict_support
        else "mixed_multi_anchor"
        if multi_anchor_enabled
        else "weak_recovery"
        if rebuild_support
        else str(initial_support.get("support_mode", "normal"))
    )
    core_mask = np.asarray(catalog["_weak_core_mask"], dtype=bool) | np.asarray(
        catalog["_bright_core_mask"],
        dtype=bool,
    )
    result = {
        "status": "ok" if curve_usable else "rejected",
        "support_status": "ok",
        "reason": (
            " | ".join(rejection_codes) if rejection_codes else ""
        ),
        "reason_code": rejection_codes[0] if rejection_codes else "",
        "rejection_reason_codes": rejection_codes,
        "method": (
            "monotonic_multi_anchor_star_curve"
            if multi_anchor_enabled
            else "connected_compact_distribution_calibrated_asinh"
        ),
        "adaptive_enabled": adaptive_enabled,
        "multi_anchor_curve": multi_anchor_enabled,
        "chroma_regularization_enabled": bool(
            multi_anchor_enabled and chroma_regularization_enabled
        ),
        "faint_chroma_max": faint_chroma_max,
        "bright_chroma_max": bright_chroma_max,
        "chroma_regularization": chroma_regularization,
        "support_mode": support_mode,
        "psf_support_retry_pixels": max(0, min(int(support_retry_pixels), 2)),
        "stretch": float(stretch),
        "weak_stretch": float(stretch),
        "bright_stretch": float(bright_stretch),
        "derived_asinh_stretch": float(output_limited_stretch),
        "asinh_stretch_role": "derived_diagnostic",
        "configured_stretch_proposal": configured_stretch_proposal,
        "offset": float(offset),
        "background": background,
        "noise_sigma": noise_sigma,
        "signal_floor": signal_floor,
        "star_sample_count": int(signal.size),
        "faint_value": faint_value,
        "mid_value": mid_value,
        "bright_value": bright_value,
        "peak_value": peak_value,
        "faint_target": faint_target,
        "mid_target": mid_target,
        "bright_target": bright_target,
        "peak_target": peak_target,
        "output_targets": output_targets,
        "output_target_scale": float(output_target_scale),
        "output_target_limited": bool(output_target_limited),
        "output_profile_mode": output_profile_mode,
        "output_profile": output_profile,
        "light_stretch_usable": curve_usable,
        "light_stretch_contract": {
            "status": "accepted" if curve_usable else "rejected",
            "definition": "four_anchor_outputs_and_change_coverage",
            "output_profile": output_profile,
            "output_target_scale": float(output_target_scale),
            "output_target_limited": bool(output_target_limited),
            "derived_asinh_stretch": float(output_limited_stretch),
            "asinh_stretch_role": "derived_diagnostic",
            "coverage": float(predicted_coverage),
            "coverage_limit": float(coverage_limit),
            "rejection_reason_codes": rejection_codes,
        },
        "faint_limited_stretch": float(faint_limited),
        "mid_limited_stretch": float(limited_stretches["mid"]),
        "bright_limited_stretch": float(limited_stretches["bright"]),
        "peak_limited_stretch": float(peak_limited),
        "output_limited_stretch": float(output_limited_stretch),
        "target_stretch": float(target_stretch),
        "predicted_change_ratio": float(predicted_coverage),
        "predicted_change_ratio_limit": float(coverage_limit),
        "predicted_change_input_threshold": float(change_input_threshold),
        "coverage_limited": bool(coverage_limited),
        "reference_intensity": float(reference_intensity),
        "compact_component_count": int(catalog.get("component_count", 0)),
        "compact_core_coverage": float(np.mean(core_mask)),
        "compact_support_coverage": float(np.mean(support_mask)),
        "compact_core_threshold": float(
            initial_support.get("core_threshold", catalog.get("reference_threshold", 0.0))
        ),
        "compact_core_percentile": float(initial_support.get("core_percentile", 0.0)),
        "compact_noise_multiplier": float(initial_support.get("noise_multiplier", 0.0)),
        "compact_wing_iterations": int(bright_wing_iterations),
        "weak_wing_iterations": int(weak_wing_iterations),
        "bright_wing_iterations": int(bright_wing_iterations),
        "initial_weak_star_retention": float(initial_weak_retention),
        "weak_star_retention": float(weak_retention),
        "weak_star_retention_min": float(min_weak_retention),
        "star_retention": float(total_retention),
        "support_rebuilt_for_weak_stars": bool(rebuild_support),
        "advisories": calibration_advisories,
        "quality_gates": {
            "compact_weak_star_retention": weak_retention_gate,
            "starmask_output_targets": output_profile_gate,
        },
        "quality_advisory_multiplier": (
            stage7_quality.stage7_9_quality_advisory_multiplier(cfg)
        ),
        "mixed_star_field": mixed_star_field,
        "star_reference": star_reference_summary(catalog),
        "full_layer_predicted_change_ratio": full_layer_predicted_coverage,
        "removed_predicted_change_ratio": max(
            0.0,
            full_layer_predicted_coverage - float(predicted_coverage),
        ),
        "predicted_faint": float(
            (output_profile.get("actual") or {}).get("faint", 0.0)
        ),
        "predicted_mid": float(
            (output_profile.get("actual") or {}).get("mid", 0.0)
        ),
        "predicted_bright": float(
            (output_profile.get("actual") or {}).get("bright", 0.0)
        ),
        "predicted_peak": float(
            (output_profile.get("actual") or {}).get("peak", 0.0)
        ),
    }
    if multi_anchor_enabled:
        result["anchor_input_percentiles"] = [0.0, 40.0, 80.0, 90.0, 99.7]
        result["anchor_input_values"] = [
            float(value) for value in anchor_input_values
        ]
        result["anchor_output_targets"] = [
            float(value) for value in anchor_output_targets
        ]
        result["brightness_ordering_preserved"] = True
    if include_support_mask:
        result["_compact_support_mask"] = support_mask
        result["_weak_support_mask"] = weak_support
        result["_bright_support_mask"] = bright_support
        result["_output_profile_sample_mask"] = profile_sample_mask
        result["_star_reference_catalog"] = catalog
    return result


def _stage9_support_candidate_summary(
    calibration: Dict[str, Any],
    cfg: Any,
    *,
    plugin_stretched_stars: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Return JSON-safe preflight evidence for one starmask support mode."""
    status = str(calibration.get("status") or "unavailable")
    candidate_status = status
    support_mask = calibration.get("_compact_support_mask")
    support_coverage = float(
        calibration.get(
            "compact_support_coverage",
            float(np.mean(np.asarray(support_mask, dtype=bool)))
            if support_mask is not None
            else 0.0,
        )
        or 0.0
    )
    support_limit = _bounded(
        getattr(cfg, "stage9_star_support_ratio_max", 0.12),
        0.12,
        0.03,
        0.20,
    )
    support_gate = stage7_quality.stage7_9_upper_quality_gate(
        cfg,
        value=support_coverage,
        accepted_limit=support_limit,
    )

    predicted_change: float | None = float(
        calibration.get("predicted_change_ratio", 0.0) or 0.0
    )
    predicted_change_source = "calibrated_builtin_stretch"
    plugin_measurement_required = plugin_stretched_stars is not None
    plugin_measurement_available = not plugin_measurement_required
    plugin_measurement_reason = ""
    output_profile = dict(calibration.get("output_profile") or {})
    if not output_profile:
        # Legacy resumes and narrow test doubles predate the v5 light-output
        # contract. Production calibrations always carry a measured profile.
        output_profile = {
            "status": "legacy_not_assessed",
            "accepted": True,
            "hard_failed": False,
            "actual": {},
            "targets": dict(calibration.get("output_targets") or {}),
            "tolerance": _STAGE9_STARMASK_OUTPUT_TOLERANCE,
            "exceeded_anchors": [],
        }
    if plugin_stretched_stars is not None and support_mask is not None:
        try:
            plugin_layer = apply_compact_starmask_support(
                _normalized(np.asarray(plugin_stretched_stars)),
                np.asarray(support_mask, dtype=bool),
            )
            reference_intensity = _bounded(
                getattr(cfg, "star_intensity", 1.05),
                1.05,
                0.10,
                1.05,
            )
            predicted_change = float(
                np.mean(
                    _pixel_peak(plugin_layer) * reference_intensity
                    > _STAR_RECOVERY_DELTA
                )
            )
            predicted_change_source = "actual_plugin_stretched_pixels"
            output_profile = measure_starmask_output_profile(
                plugin_layer,
                calibration,
                source="actual_plugin_stretched_pixels",
            )
            plugin_measurement_available = bool(
                output_profile.get("status") != "unavailable"
            )
            if not plugin_measurement_available:
                plugin_measurement_reason = str(
                    output_profile.get("reason")
                    or "plugin output profile unavailable"
                )
            candidate_status = str(
                calibration.get("support_status") or status
            )
        except (IndexError, TypeError, ValueError, FloatingPointError) as error:
            predicted_change = None
            predicted_change_source = "plugin_measurement_unavailable"
            plugin_measurement_reason = str(error)
    elif plugin_measurement_required:
        predicted_change = None
        predicted_change_source = "plugin_measurement_unavailable"
        plugin_measurement_reason = "compact support mask unavailable"
    predicted_change_limit = float(
        calibration.get(
            "predicted_change_ratio_limit",
            min(
                _bounded(
                    getattr(
                        cfg,
                        "stage9_starmask_predicted_change_ratio_max",
                        0.30,
                    ),
                    0.30,
                    0.05,
                    0.60,
                ),
                0.90
                * _bounded(
                    getattr(cfg, "stage9_changed_pixel_ratio_max", 0.35),
                    0.35,
                    0.05,
                    0.80,
                ),
            ),
        )
        or 0.30
    )
    if plugin_measurement_required and not plugin_measurement_available:
        predicted_change_gate = {
            "status": "hard_failed",
            "accepted": False,
            "advisory": False,
            "hard_failed": True,
            "value": None,
            "accepted_limit": predicted_change_limit,
            "reason": plugin_measurement_reason,
        }
    else:
        predicted_change_gate = stage7_quality.stage7_9_upper_quality_gate(
            cfg,
            value=float(predicted_change or 0.0),
            accepted_limit=predicted_change_limit,
        )
    output_profile_gate = _starmask_output_profile_gate(output_profile)
    if plugin_measurement_required and not plugin_measurement_available:
        output_profile_gate.update(
            status="hard_failed",
            accepted=False,
            advisory=False,
            hard_failed=True,
            reason=plugin_measurement_reason,
        )
    weak_retention = float(
        calibration.get(
            "weak_star_retention",
            1.0 if status == "ok" else 0.0,
        )
        or 0.0
    )
    weak_retention_min = float(
        calibration.get(
            "weak_star_retention_min",
            getattr(cfg, "stage9_compact_weak_star_retention_min", 0.80),
        )
        or 0.80
    )
    weak_retention_gate = stage7_quality.stage7_9_lower_quality_gate(
        cfg,
        value=weak_retention,
        accepted_limit=weak_retention_min,
    )

    gates = {
        "star_support_ratio": support_gate,
        "predicted_change_ratio": predicted_change_gate,
        "starmask_output_targets": output_profile_gate,
        "compact_weak_star_retention": weak_retention_gate,
    }
    gate_statuses = [
        str(gate.get("status") or "hard_failed") for gate in gates.values()
    ]
    if candidate_status != "ok" or "hard_failed" in gate_statuses:
        risk_level = "hard_failed"
    elif "advisory" in gate_statuses:
        risk_level = "advisory"
    else:
        risk_level = "ok"
    usable = bool(candidate_status == "ok" and risk_level != "hard_failed")
    actual_anchors = dict(output_profile.get("actual") or {})
    target_anchors = dict(output_profile.get("targets") or {})
    anchor_ratios: Dict[str, float] = {}
    for anchor in ("faint", "mid", "bright", "peak"):
        try:
            actual_value = float(actual_anchors[anchor])
            target_value = float(target_anchors[anchor])
        except (KeyError, TypeError, ValueError):
            continue
        if target_value > 1e-7 and np.isfinite(actual_value):
            anchor_ratios[anchor] = float(actual_value / target_value)
    minimum_anchor_ratio = min(anchor_ratios.values(), default=None)
    psf_undersize_risk = bool(
        minimum_anchor_ratio is not None and minimum_anchor_ratio < 0.50
    )
    output_adequacy = {
        "status": "risk" if psf_undersize_risk else "ok",
        "advisory_only": True,
        "formal_gate": False,
        "anchor_ratios": anchor_ratios,
        "minimum_anchor_ratio": minimum_anchor_ratio,
        "psf_undersize_risk": psf_undersize_risk,
        "reason_code": (
            "stage9_starmask_output_undersize_risk"
            if psf_undersize_risk
            else "stage9_starmask_output_adequate"
        ),
    }
    star_reference = calibration.get("star_reference") or {}
    return {
        "status": candidate_status,
        "calibration_status": status,
        "usable": usable,
        "risk_level": risk_level,
        "reason": str(calibration.get("reason") or ""),
        "support_mode": str(calibration.get("support_mode") or "unknown"),
        "support_coverage": support_coverage,
        "support_coverage_limit": support_limit,
        "predicted_change_ratio": predicted_change,
        "predicted_change_ratio_limit": predicted_change_limit,
        "predicted_change_source": predicted_change_source,
        "plugin_measurement_required": plugin_measurement_required,
        "plugin_measurement_available": plugin_measurement_available,
        "plugin_measurement_reason": plugin_measurement_reason,
        "output_profile": output_profile,
        "output_adequacy": output_adequacy,
        "output_target_scale": float(
            calibration.get("output_target_scale", 1.0) or 0.0
        ),
        "derived_asinh_stretch": calibration.get("derived_asinh_stretch"),
        "asinh_stretch_role": str(
            calibration.get("asinh_stretch_role") or "derived_diagnostic"
        ),
        "weak_star_retention": weak_retention,
        "weak_star_retention_min": weak_retention_min,
        "star_retention": float(calibration.get("star_retention", 0.0) or 0.0),
        "psf_support_radius_max": star_reference.get("psf_support_radius_max"),
        "psf_support_radius_median_px": star_reference.get(
            "psf_support_radius_median_px"
        ),
        "psf_support_radius_p95_px": star_reference.get(
            "psf_support_radius_p95_px"
        ),
        "hard_failed": risk_level == "hard_failed",
        "advisory": risk_level == "advisory",
        "gates": gates,
        "advisories": list(calibration.get("advisories") or []),
    }


def assess_starmask_support_preflight(
    stars: np.ndarray,
    cfg: Any,
    *,
    reference_catalog: Dict[str, Any] | None = None,
    failure_action: str = "auto_fallback",
    plugin_stretched_stars: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Preflight normal/strict support without mutating the Siril image buffer."""
    compact_enabled = bool(
        getattr(cfg, "stage9_compact_starmask_enabled", True)
    )
    pre_stretch_compact_enabled = bool(
        getattr(
            cfg,
            "stage9_starmask_pre_stretch_compact_enabled",
            False,
        )
    )
    base_catalog = dict(reference_catalog or {})
    normal_catalog = dict(base_catalog)
    strict_catalog = dict(base_catalog)
    normal = calibrate_starmask_asinh(
        stars,
        cfg,
        include_support_mask=True,
        strict_support=False,
        reference_catalog=normal_catalog or None,
    )
    normal_summary = _stage9_support_candidate_summary(
        normal,
        cfg,
        plugin_stretched_stars=plugin_stretched_stars,
    )
    if plugin_stretched_stars is not None:
        normal["plugin_output_profile"] = dict(
            normal_summary.get("output_profile") or {}
        )
        normal["plugin_predicted_change_ratio"] = normal_summary.get(
            "predicted_change_ratio"
        )
        normal["plugin_predicted_change_source"] = normal_summary.get(
            "predicted_change_source"
        )
    frozen_catalog = (
        base_catalog
        or dict(normal.get("_star_reference_catalog") or {})
    )
    strict: Dict[str, Any] = {
        "status": "disabled",
        "reason": "stage9_compact_starmask_enabled=false",
    }
    strict_summary: Dict[str, Any] = {
        "status": "disabled",
        "usable": False,
        "risk_level": "disabled",
        "reason": "stage9_compact_starmask_enabled=false",
        "support_mode": "strict_recovery",
        "gates": {},
        "advisories": [],
    }
    equivalent = False
    if compact_enabled:
        strict = calibrate_starmask_asinh(
            stars,
            cfg,
            include_support_mask=True,
            strict_support=True,
            reference_catalog=frozen_catalog or strict_catalog or None,
        )
        strict_summary = _stage9_support_candidate_summary(strict, cfg)
        normal_mask = normal.get("_compact_support_mask")
        strict_mask = strict.get("_compact_support_mask")
        if normal_mask is not None and strict_mask is not None:
            try:
                equivalent = bool(
                    np.array_equal(
                        np.asarray(normal_mask, dtype=bool),
                        np.asarray(strict_mask, dtype=bool),
                    )
                )
            except (TypeError, ValueError):
                equivalent = False

    normal_usable = bool(normal_summary.get("usable", False))
    strict_usable = bool(strict_summary.get("usable", False))
    normal_risk = str(normal_summary.get("risk_level") or "hard_failed")
    normal_undersize_risk = bool(
        (normal_summary.get("output_adequacy") or {}).get(
            "psf_undersize_risk",
            False,
        )
    )
    route = "unavailable"
    reason_code = "stage9_support_preflight_no_usable_candidate"
    if not compact_enabled:
        route = "normal_only"
        reason_code = "stage9_support_preflight_compact_disabled"
    elif equivalent and normal_usable:
        route = "normal_only"
        reason_code = "stage9_support_preflight_equivalent_masks"
    elif not normal_usable and strict_usable:
        route = "strict_only"
        reason_code = "stage9_support_preflight_normal_hard_failed"
    elif normal_usable and normal_undersize_risk and strict_usable:
        if str(failure_action or "auto_fallback") == "auto_fallback":
            route = "dual_competition"
            reason_code = "stage9_support_preflight_output_adequacy_dual"
        else:
            route = "normal_only"
            reason_code = "stage9_support_preflight_output_adequacy_normal"
    elif normal_usable and normal_risk == "advisory" and strict_usable:
        if str(failure_action or "auto_fallback") == "auto_fallback":
            route = "dual_competition"
            reason_code = "stage9_support_preflight_boundary_dual"
        else:
            route = "normal_only"
            reason_code = "stage9_support_preflight_boundary_policy_normal"
    elif normal_usable and normal_risk == "advisory":
        route = "normal_only"
        reason_code = (
            "stage9_support_preflight_boundary_strict_unavailable"
        )
    elif normal_usable:
        route = "normal_only"
        reason_code = "stage9_support_preflight_normal_clear"
    elif strict_usable:
        route = "strict_only"
        reason_code = "stage9_support_preflight_strict_only_available"

    status = "ready" if route != "unavailable" else "rejected"
    planned_candidates = list(
        {
            "normal_only": ("normal",),
            "strict_only": ("strict_compact",),
            "dual_competition": ("normal", "strict_compact"),
        }.get(route, ())
    )
    skipped_candidates = []
    for support_mode, summary in (
        ("normal", normal_summary),
        ("strict_compact", strict_summary),
    ):
        if support_mode in planned_candidates:
            continue
        skipped_candidates.append(
            {
                "support_mode": support_mode,
                "reason_code": reason_code,
                "status": str(summary.get("status") or "unavailable"),
                "risk_level": str(summary.get("risk_level") or "unavailable"),
                "reason": str(summary.get("reason") or ""),
            }
        )
    return {
        "schema": "starun.stage9-starmask_support_preflight.v1",
        "status": status,
        "strategy": "adaptive_dual_route",
        "compact_enabled": compact_enabled,
        "compact_support_enabled": compact_enabled,
        "pre_stretch_compact_enabled": pre_stretch_compact_enabled,
        "failure_action": str(failure_action or "auto_fallback"),
        "route": route,
        "reason_code": reason_code,
        "support_masks_equivalent": equivalent,
        "planned_candidates": planned_candidates,
        "skipped_candidates": skipped_candidates,
        "candidates": {
            "normal": normal_summary,
            "strict_compact": strict_summary,
        },
        "executed_candidates": [],
        "selected_support_mode": None,
        "_calibrations": {
            "normal": normal,
            "strict_compact": strict,
        },
    }


def public_starmask_support_preflight(
    report: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """Strip in-memory calibration masks from a support preflight report."""
    if not isinstance(report, dict):
        return {
            "schema": "starun.stage9-starmask_support_preflight.v1",
            "status": "unavailable",
            "reason": "support preflight report missing",
        }
    return {
        key: value
        for key, value in report.items()
        if not str(key).startswith("_")
    }


def assess_remix(
    base: np.ndarray,
    candidate: np.ndarray,
    cfg: Any,
    *,
    attempt: str,
    formula: str,
    star_reference: Dict[str, Any] | None = None,
    star_overlay_mask: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Compare a Stage 9 candidate with its immutable remix base."""
    base_arr = np.asarray(base)
    candidate_arr = np.asarray(candidate)
    if base_arr.shape != candidate_arr.shape:
        return {
            "attempt": attempt,
            "formula": formula,
            "status": "rejected",
            "accepted": False,
            "issues": [
                f"shape mismatch: base={base_arr.shape}, candidate={candidate_arr.shape}"
            ],
            "metrics": {},
        }

    scale = _image_scale(base_arr)
    base_norm = _normalized(base_arr, scale=scale)
    candidate_finite = np.isfinite(candidate_arr)
    candidate_norm = _normalized(candidate_arr, scale=scale)
    base_gray = _gray(base_norm)
    candidate_gray = _gray(candidate_norm)
    base_luminance = _luminance(base_norm)
    candidate_luminance = _luminance(candidate_norm)
    delta = candidate_gray - base_gray
    base_peak = _pixel_peak(base_norm)
    candidate_peak = _pixel_peak(candidate_norm)
    change_peak = _pixel_peak(np.abs(candidate_norm - base_norm))
    positive_change = np.maximum(candidate_norm - base_norm, 0.0)
    positive_change_peak = _pixel_peak(positive_change)
    positive_change_floor = _pixel_floor(positive_change)
    positive_change_saturation = np.divide(
        positive_change_peak - positive_change_floor,
        np.maximum(positive_change_peak, 1e-12),
        out=np.zeros_like(positive_change_peak),
        where=positive_change_peak > 0.0,
    )
    chromatic_addition_peak_min = _bounded(
        getattr(cfg, "stage9_chromatic_addition_peak_min", 0.02),
        0.02,
        0.002,
        0.25,
    )
    chromatic_addition_saturation_min = _bounded(
        getattr(cfg, "stage9_chromatic_addition_saturation_min", 0.70),
        0.70,
        0.30,
        0.95,
    )
    chromatic_addition_ratio = float(
        np.mean(
            (positive_change_peak > chromatic_addition_peak_min)
            & (positive_change_saturation > chromatic_addition_saturation_min)
        )
    )
    confirmed_star_support = None
    if (
        isinstance(star_reference, dict)
        and star_reference.get("status") == "ok"
        and bool(star_reference.get("source_matched", False))
        and star_overlay_mask is not None
    ):
        supplied_support = np.asarray(star_overlay_mask, dtype=bool)
        if supplied_support.shape == positive_change_peak.shape:
            confirmed_star_support = supplied_support
    local_quality = _stage9_local_quality_metrics(
        base_norm,
        candidate_norm,
        positive_change,
        cfg,
        confirmed_star_support=confirmed_star_support,
        star_reference=star_reference,
    )
    psf_scale_shadow = _stage9_psf_scale_shadow(
        positive_change,
        star_reference,
        star_overlay_mask=star_overlay_mask,
    )
    psf_closure = assess_stage9_psf_closure(
        candidate_norm,
        star_reference,
        cfg,
    )

    hollow_delta_min = _bounded(
        getattr(cfg, "stage9_hollow_structure_delta_min", 0.05),
        0.05,
        0.01,
        0.25,
    )
    hollow_structure_count = 0
    hollow_structure_max_area = 0
    if scipy_ndimage is not None:
        significant_addition = np.asarray(
            positive_change_peak > hollow_delta_min,
            dtype=bool,
        )
        if np.any(significant_addition):
            filled_addition = scipy_ndimage.binary_fill_holes(significant_addition)
            hollow_mask = filled_addition & ~significant_addition
            hollow_labels, hollow_count = scipy_ndimage.label(
                hollow_mask,
                structure=np.ones((3, 3), dtype=np.uint8),
            )
            if hollow_count > 0:
                hollow_areas = np.bincount(hollow_labels.reshape(-1))[1:]
                hollow_min_area = stage9_scale_area(
                    _HOLLOW_STRUCTURE_MIN_AREA,
                    star_reference,
                )
                meaningful_holes = hollow_areas[hollow_areas >= hollow_min_area]
                hollow_structure_count = int(meaningful_holes.size)
                if meaningful_holes.size:
                    hollow_structure_max_area = int(np.max(meaningful_holes))

    star_exclusion_mask = None
    if star_overlay_mask is not None:
        supplied_mask = np.asarray(star_overlay_mask, dtype=bool)
        if supplied_mask.shape == positive_change_peak.shape:
            star_exclusion_mask = supplied_mask
    if isinstance(star_reference, dict) and star_reference.get("status") == "ok":
        weak_core = star_reference.get("_weak_core_mask")
        bright_core = star_reference.get("_bright_core_mask")
        if weak_core is not None and bright_core is not None:
            core_mask = np.asarray(weak_core, dtype=bool) | np.asarray(
                bright_core,
                dtype=bool,
            )
            if core_mask.shape == positive_change_peak.shape and star_exclusion_mask is None:
                try:
                    star_exclusion_mask = _catalog_support_masks(
                        star_reference,
                        strict=False,
                    )[2]
                except (KeyError, RuntimeError, ValueError):
                    star_exclusion_mask = core_mask

    metric_exclusion_mask = star_exclusion_mask
    if star_exclusion_mask is not None and scipy_ndimage is not None:
        metric_exclusion_mask = scipy_ndimage.binary_dilation(
            star_exclusion_mask,
            structure=np.ones((3, 3), dtype=bool),
            iterations=stage9_scale_radius(
                3,
                star_reference,
                rounding="nearest",
                minimum=1,
            ),
        )

    background_limit = float(np.percentile(base_gray, 35.0))
    background_mask = base_gray <= background_limit
    if metric_exclusion_mask is not None:
        background_mask &= ~metric_exclusion_mask
    if int(np.count_nonzero(background_mask)) < 32:
        background_mask = np.ones_like(base_gray, dtype=bool)
        if metric_exclusion_mask is not None:
            background_mask &= ~metric_exclusion_mask

    clip_before = float(np.mean(base_peak >= 0.995))
    clip_after = float(np.mean(candidate_peak >= 0.995))
    bright_before = float(np.mean(base_peak >= 0.90))
    bright_after = float(np.mean(candidate_peak >= 0.90))
    mottling_before = _background_mottling_score(
        base_luminance,
        exclusion_mask=metric_exclusion_mask,
    )
    mottling_after = _background_mottling_score(
        candidate_luminance,
        exclusion_mask=metric_exclusion_mask,
    )
    mottling_ratio_floor = 0.01
    mottling_delta = max(0.0, mottling_after - mottling_before)
    mottling_growth = max(
        mottling_after,
        mottling_ratio_floor,
    ) / max(mottling_before, mottling_ratio_floor)
    changed_pixel_ratio = float(np.mean(change_peak > 0.002))
    star_support_ratio = (
        float(np.mean(star_exclusion_mask))
        if star_exclusion_mask is not None
        else 0.0
    )
    unmatched_changed_ratio = (
        float(np.mean((change_peak > 0.002) & ~star_exclusion_mask))
        if star_exclusion_mask is not None
        else changed_pixel_ratio
    )
    mottling_exemption_changed_ratio_max = _bounded(
        getattr(
            cfg,
            "stage9_mottling_exemption_changed_pixel_ratio_max",
            _MOTTLING_LOW_ABSOLUTE_CHANGED_RATIO_MAX,
        ),
        _MOTTLING_LOW_ABSOLUTE_CHANGED_RATIO_MAX,
        0.02,
        0.35,
    )
    mottling_low_absolute = (
        mottling_after <= _MOTTLING_LOW_ABSOLUTE_SCORE_MAX
        and mottling_delta <= _MOTTLING_LOW_ABSOLUTE_DELTA_MAX
        and changed_pixel_ratio <= mottling_exemption_changed_ratio_max
    )
    metrics = {
        "finite_ratio": float(np.mean(candidate_finite)),
        "highlight_clip_ratio_before": clip_before,
        "highlight_clip_ratio_after": clip_after,
        "highlight_clip_growth": max(0.0, clip_after - clip_before),
        "bright_pixel_ratio_before": bright_before,
        "bright_pixel_ratio_after": bright_after,
        "bright_pixel_growth": max(0.0, bright_after - bright_before),
        "background_lift": float(np.median(delta[background_mask])),
        "background_mottling_score_before": mottling_before,
        "background_mottling_score_after": mottling_after,
        "background_mottling_delta": mottling_delta,
        "background_mottling_growth": mottling_growth,
        "background_mottling_low_absolute_exempted": mottling_low_absolute,
        "changed_pixel_ratio": changed_pixel_ratio,
        "chromatic_star_addition_ratio": chromatic_addition_ratio,
        "chromatic_star_addition_peak_min": chromatic_addition_peak_min,
        "chromatic_star_addition_saturation_min": (
            chromatic_addition_saturation_min
        ),
        "star_support_ratio": star_support_ratio,
        "unmatched_changed_ratio": unmatched_changed_ratio,
        "darkening_ratio": float(np.mean(delta < -0.001)),
        "star_exclusion_ratio": star_support_ratio,
        "residual_dark_hole_ratio": 0.0,
        "new_hollow_structure_count": hollow_structure_count,
        "new_hollow_structure_max_area": hollow_structure_max_area,
        "new_hollow_structure_delta_min": hollow_delta_min,
    }
    metrics.update(local_quality.get("metrics") or {})
    metrics.update(psf_closure.get("metrics") or {})

    recovery_status = "unavailable"
    recovery_reason = "star reference catalog missing"
    if isinstance(star_reference, dict) and star_reference.get("status") == "ok":
        peak_y = np.asarray(star_reference.get("_peak_y", ()), dtype=np.int32)
        peak_x = np.asarray(star_reference.get("_peak_x", ()), dtype=np.int32)
        weak_flags = np.asarray(
            star_reference.get("_weak_flags", ()),
            dtype=bool,
        )
        saturated_flags = np.asarray(
            star_reference.get("_psf_saturated_flags", ()),
            dtype=bool,
        )
        spatial_shape = tuple(positive_change_peak.shape)
        reference_valid = bool(
            peak_y.size > 0
            and peak_y.size == peak_x.size
            and peak_y.size == weak_flags.size
            and len(spatial_shape) == 2
            and np.all((peak_y >= 0) & (peak_y < spatial_shape[0]))
            and np.all((peak_x >= 0) & (peak_x < spatial_shape[1]))
        )
        if reference_valid:
            saturation_valid = saturated_flags.size == peak_y.size
            if not saturation_valid:
                saturated_flags = np.zeros(peak_y.size, dtype=bool)
            per_star_fwhm = np.asarray(
                star_reference.get("_stage9_spatial_fwhm_px", ()),
                dtype=np.float32,
            )
            scale_report = _stage9_catalog_scale(star_reference)
            if per_star_fwhm.size != peak_y.size:
                per_star_fwhm = np.full(
                    peak_y.size,
                    float(
                        scale_report.get(
                            "fwhm_median_px",
                            _STAGE9_FWHM_ANCHOR_PX,
                        )
                    ),
                    dtype=np.float32,
                )
            inner_windows = np.asarray(
                star_reference.get("_stage9_inner_window_size_px", ()),
                dtype=np.int32,
            )
            outer_windows = np.asarray(
                star_reference.get("_stage9_outer_window_size_px", ()),
                dtype=np.int32,
            )
            if inner_windows.size != peak_y.size:
                inner_windows = np.asarray(
                    [
                        stage9_scale_odd_window(
                            3,
                            star_reference,
                            fwhm_px=value,
                        )
                        for value in per_star_fwhm
                    ],
                    dtype=np.int32,
                )
            if outer_windows.size != peak_y.size:
                outer_windows = np.asarray(
                    [
                        stage9_scale_odd_window(
                            7,
                            star_reference,
                            fwhm_px=value,
                        )
                        for value in per_star_fwhm
                    ],
                    dtype=np.int32,
                )
            local_positive_values = _stage9_square_window_values(
                positive_change_peak,
                peak_y,
                peak_x,
                inner_windows,
                statistic="max",
            )
            restored = np.asarray(
                local_positive_values >= _STAR_RECOVERY_DELTA,
                dtype=bool,
            )
            aperture3 = _stage9_square_window_values(
                positive_change_peak,
                peak_y,
                peak_x,
                inner_windows,
                statistic="sum",
            )
            aperture7 = _stage9_square_window_values(
                positive_change_peak,
                peak_y,
                peak_x,
                outer_windows,
                statistic="sum",
            )
            aperture_restored = aperture7 >= 0.006
            candidate_wing_ratio = np.divide(
                np.maximum(aperture7 - aperture3, 0.0),
                np.maximum(aperture7, 1e-12),
                out=np.zeros_like(aperture7),
                where=aperture7 > 0.0,
            )
            reference_wing_ratio = np.asarray(
                star_reference.get("_reference_wing_ratio", ()),
                dtype=np.float32,
            )
            wing_valid = reference_wing_ratio.size == restored.size
            if wing_valid:
                wing_eligible = reference_wing_ratio >= 0.08
                wing_floor = np.maximum(
                    0.02,
                    np.minimum(reference_wing_ratio * 0.35, 0.12),
                )
                wing_restored = (
                    aperture_restored
                    & wing_eligible
                    & (candidate_wing_ratio >= wing_floor)
                )
                wing_reference_count = int(np.count_nonzero(wing_eligible))
                wing_recovery_ratio = (
                    float(np.count_nonzero(wing_restored) / wing_reference_count)
                    if wing_reference_count
                    else 1.0
                )
                saturated_wing_eligible = wing_eligible & saturated_flags
                saturated_wing_reference_count = int(
                    np.count_nonzero(saturated_wing_eligible)
                )
                saturated_wing_recovery_ratio = (
                    float(
                        np.count_nonzero(
                            wing_restored & saturated_wing_eligible
                        )
                        / saturated_wing_reference_count
                    )
                    if saturated_wing_reference_count
                    else None
                )
            else:
                wing_reference_count = 0
                wing_recovery_ratio = 1.0
                saturated_wing_reference_count = 0
                saturated_wing_recovery_ratio = None
            weak_count = int(np.count_nonzero(weak_flags))
            bright_flags = ~weak_flags
            bright_count = int(np.count_nonzero(bright_flags))
            weak_restored_count = int(np.count_nonzero(restored & weak_flags))
            bright_restored_count = int(np.count_nonzero(restored & bright_flags))
            restored_count = int(np.count_nonzero(restored))
            metrics.update(
                {
                    "star_recovery_status": "ok",
                    "star_recovery_delta_min": _STAR_RECOVERY_DELTA,
                    "star_reference_count": int(restored.size),
                    "star_restored_count": restored_count,
                    "star_recovery_ratio": float(np.mean(restored)),
                    "weak_star_reference_count": weak_count,
                    "weak_star_restored_count": weak_restored_count,
                    "weak_star_recovery_ratio": (
                        float(weak_restored_count / weak_count)
                        if weak_count
                        else 0.0
                    ),
                    "bright_star_reference_count": bright_count,
                    "bright_star_restored_count": bright_restored_count,
                    "bright_star_recovery_ratio": (
                        float(bright_restored_count / bright_count)
                        if bright_count
                        else 0.0
                    ),
                    "star_positive_delta_window_size_px": 7,
                    "star_positive_delta_window_size_px_nominal": 7,
                    "star_positive_delta_window_size_px_effective": (
                        stage9_effective_pixel_stats(outer_windows)
                    ),
                    "star_inner_window_size_px_nominal": 3,
                    "star_inner_window_size_px_effective": (
                        stage9_effective_pixel_stats(inner_windows)
                    ),
                    "star_positive_delta_window_sum_min": 0.006,
                    "star_positive_delta_window_restored_count": int(
                        np.count_nonzero(aperture_restored)
                    ),
                    "star_positive_delta_window_recovery_ratio": float(
                        np.mean(aperture_restored)
                    ),
                    "star_wing_reference_count": wing_reference_count,
                    "star_wing_recovery_ratio": wing_recovery_ratio,
                    "star_saturation_classification_status": (
                        "available" if saturation_valid else "unavailable"
                    ),
                    "saturated_star_wing_reference_count": (
                        saturated_wing_reference_count
                    ),
                    "saturated_star_wing_recovery_ratio": (
                        saturated_wing_recovery_ratio
                    ),
                }
            )
            if star_exclusion_mask is not None and scipy_ndimage is not None:
                local_background = _stage9_square_window_values(
                    base_luminance,
                    peak_y,
                    peak_x,
                    outer_windows,
                    statistic="median",
                )
                base_local_min = _stage9_square_window_values(
                    base_luminance,
                    peak_y,
                    peak_x,
                    inner_windows,
                    statistic="min",
                )
                candidate_local_peak = _stage9_square_window_values(
                    candidate_luminance,
                    peak_y,
                    peak_x,
                    inner_windows,
                    statistic="max",
                )
                initial_holes = (
                    local_background - base_local_min > 0.002
                )
                initial_hole_count = int(np.count_nonzero(initial_holes))
                residual_holes = initial_holes & (
                    local_background - candidate_local_peak > 0.001
                )
                metrics.update(
                    {
                        "star_dark_hole_reference_count": initial_hole_count,
                        "residual_dark_hole_ratio": (
                            float(np.count_nonzero(residual_holes) / initial_hole_count)
                            if initial_hole_count
                            else 0.0
                        ),
                    }
                )
            else:
                metrics.update(
                    {
                        "star_dark_hole_reference_count": 0,
                        "residual_dark_hole_ratio": 0.0,
                    }
                )
            recovery_status = "ok"
            recovery_reason = ""
        else:
            recovery_reason = "star reference coordinates invalid for candidate"
    elif isinstance(star_reference, dict):
        recovery_reason = str(
            star_reference.get("reason") or "star reference catalog unavailable"
        )
    if recovery_status != "ok":
        metrics.update(
            {
                "star_recovery_status": "unavailable",
                "star_recovery_reason": recovery_reason,
                "star_recovery_delta_min": _STAR_RECOVERY_DELTA,
            }
        )

    enabled = bool(getattr(cfg, "stage9_quality_gate_enabled", True))
    structural_issues: list[str] = []
    gate_issues: list[str] = []
    advisories: list[str] = []
    advisories.extend(
        str(item) for item in (psf_closure.get("advisories") or [])
    )
    quality_gates: Dict[str, Dict[str, Any]] = {}
    if metrics["finite_ratio"] < 1.0:
        structural_issues.append(
            f"non-finite pixels: finite_ratio={metrics['finite_ratio']:.6f}"
        )
    if local_quality.get("status") != "ok":
        structural_issues.append(
            "local_quality_metrics_unavailable: "
            f"{local_quality.get('reason') or 'unknown reason'}"
        )
    if (
        bool(getattr(cfg, "stage9_psf_size_gate_enabled", True))
        and psf_closure.get("status") == "rejected"
    ):
        structural_issues.extend(
            str(issue) for issue in (psf_closure.get("issues") or [])
        )

    limits = {
        "highlight_clip_ratio_after": _bounded(
            getattr(cfg, "stage9_highlight_clip_ratio_max", 0.015),
            0.015,
            0.001,
            0.10,
        ),
        "highlight_clip_growth": _bounded(
            getattr(cfg, "stage9_highlight_clip_growth_max", 0.006),
            0.006,
            0.0,
            0.05,
        ),
        "bright_pixel_growth": _bounded(
            getattr(cfg, "stage9_bright_pixel_growth_max", 0.025),
            0.025,
            0.0,
            0.10,
        ),
        "background_lift": _bounded(
            getattr(cfg, "stage9_background_lift_max", 0.010),
            0.010,
            0.0,
            0.05,
        ),
        "changed_pixel_ratio": _bounded(
            getattr(cfg, "stage9_changed_pixel_ratio_max", 0.35),
            0.35,
            0.05,
            0.80,
        ),
        "darkening_ratio": _bounded(
            getattr(cfg, "stage9_darkening_ratio_max", 0.005),
            0.005,
            0.0,
            0.05,
        ),
        "weak_star_recovery_ratio": _bounded(
            getattr(cfg, "stage9_weak_star_recovery_ratio_min", 0.70),
            0.70,
            0.40,
            0.95,
        ),
        "star_recovery_ratio": _bounded(
            getattr(cfg, "stage9_star_recovery_ratio_min", 0.75),
            0.75,
            0.40,
            0.98,
        ),
        "star_support_ratio": _bounded(
            getattr(cfg, "stage9_star_support_ratio_max", 0.12),
            0.12,
            0.03,
            0.20,
        ),
        "unmatched_changed_ratio": _bounded(
            getattr(cfg, "stage9_unmatched_changed_ratio_max", 0.01),
            0.01,
            0.0,
            0.05,
        ),
        "chromatic_star_addition_ratio": _bounded(
            getattr(cfg, "stage9_chromatic_addition_ratio_max", 0.003),
            0.003,
            0.0,
            0.05,
        ),
        "star_positive_delta_window_recovery_ratio": _bounded(
            getattr(cfg, "stage9_star_positive_delta_window_recovery_ratio_min", 0.75),
            0.75,
            0.40,
            0.98,
        ),
        "star_wing_recovery_ratio": _bounded(
            getattr(cfg, "stage9_star_wing_recovery_ratio_min", 0.65),
            0.65,
            0.30,
            0.95,
        ),
        "residual_dark_hole_ratio": _bounded(
            getattr(cfg, "stage9_residual_dark_hole_ratio_max", 0.15),
            0.15,
            0.0,
            0.50,
        ),
        "new_hollow_structure_max_area": _bounded(
            stage9_scale_area(
                _bounded(
                    getattr(cfg, "stage9_new_hollow_structure_area_max", 64),
                    64.0,
                    4.0,
                    4096.0,
                ),
                star_reference,
            ),
            64.0,
            1.0,
            65536.0,
        ),
        "nominal_new_hollow_structure_max_area": _bounded(
            getattr(cfg, "stage9_new_hollow_structure_area_max", 64),
            64.0,
            4.0,
            4096.0,
        ),
        "fwhm_anchor_px": _STAGE9_FWHM_ANCHOR_PX,
        "spatial_radius_scale": float(
            _stage9_catalog_scale(star_reference).get("radius_scale", 1.0)
            or 1.0
        ),
        "spatial_area_scale": float(
            _stage9_catalog_scale(star_reference).get("area_scale", 1.0)
            or 1.0
        ),
    }
    limits.update(local_quality.get("limits") or {})
    limits.update(psf_closure.get("limits") or {})
    upper_limit_names = (
        "highlight_clip_ratio_after",
        "highlight_clip_growth",
        "bright_pixel_growth",
        "background_lift",
        "changed_pixel_ratio",
        "chromatic_star_addition_ratio",
        "star_support_ratio",
        "unmatched_changed_ratio",
        "darkening_ratio",
        "residual_dark_hole_ratio",
        "new_hollow_structure_max_area",
        "local_connected_component_max_area",
        "local_nonstellar_shape_component_count",
        "local_single_pixel_component_ratio",
        "local_cyan_blue_component_max_area",
        "core_color_jump_component_max_area",
    )
    for metric_name in upper_limit_names:
        limit = float(limits[metric_name])
        value = float(metrics[metric_name])
        if value > limit:
            gate = stage7_quality.stage7_9_upper_quality_gate(
                cfg,
                value=value,
                accepted_limit=limit,
            )
            quality_gates[metric_name] = gate
            message = f"{metric_name} {value:.6f}>{limit:.6f}"
            if gate["hard_failed"]:
                gate_issues.append(message)
            elif gate["advisory"]:
                advisories.append(f"{message} (advisory; remix retained)")

    if recovery_status != "ok":
        structural_issues.append(
            f"star_recovery_metrics_unavailable: {recovery_reason}"
        )
    else:
        for metric_name in (
            "weak_star_recovery_ratio",
            "star_recovery_ratio",
            "star_positive_delta_window_recovery_ratio",
            "star_wing_recovery_ratio",
        ):
            value = float(metrics[metric_name])
            limit = float(limits[metric_name])
            if value < limit:
                gate = stage7_quality.stage7_9_lower_quality_gate(
                    cfg,
                    value=value,
                    accepted_limit=limit,
                )
                quality_gates[metric_name] = gate
                message = f"{metric_name} {value:.6f}<{limit:.6f}"
                if gate["hard_failed"]:
                    gate_issues.append(message)
                elif gate["advisory"]:
                    advisories.append(f"{message} (advisory; remix retained)")

    mottling_growth_limit = _bounded(
        getattr(cfg, "stage9_background_mottling_growth_max", 1.35),
        1.35,
        1.0,
        3.0,
    )
    limits.update(
        {
            "background_mottling_growth": mottling_growth_limit,
            "background_mottling_low_absolute_score_max": (
                _MOTTLING_LOW_ABSOLUTE_SCORE_MAX
            ),
            "background_mottling_low_absolute_delta_max": (
                _MOTTLING_LOW_ABSOLUTE_DELTA_MAX
            ),
            "background_mottling_low_absolute_changed_pixel_ratio_max": (
                mottling_exemption_changed_ratio_max
            ),
        }
    )
    if mottling_growth > mottling_growth_limit and not mottling_low_absolute:
        mottling_gate = stage7_quality.stage7_9_upper_quality_gate(
            cfg,
            value=mottling_growth,
            accepted_limit=mottling_growth_limit,
        )
        quality_gates["background_mottling_growth"] = mottling_gate
        message = (
            "background_mottling_growth "
            f"{mottling_growth:.6f}>{mottling_growth_limit:.6f} "
            f"(after={mottling_after:.6f}, delta={mottling_delta:.6f})"
        )
        if mottling_gate["hard_failed"]:
            gate_issues.append(message)
        elif mottling_gate["advisory"]:
            advisories.append(f"{message} (advisory; remix retained)")

    issues = structural_issues + gate_issues
    accepted = not structural_issues and (not enabled or not gate_issues)
    reason_codes: list[str] = []
    source_reference = (
        star_reference.get("source_reference")
        if isinstance(star_reference, dict)
        else None
    )
    source_reason = str(
        (source_reference or {}).get("reason")
        if isinstance(source_reference, dict)
        else (star_reference or {}).get("reason", "")
        if isinstance(star_reference, dict)
        else ""
    )
    if "catalog_contamination" in source_reason:
        reason_codes.append("SOURCE_CATALOG_CONTAMINATION")
    if any(
        token in issue
        for issue in gate_issues
        for token in (
            "local_connected_component",
            "local_nonstellar_shape",
            "local_single_pixel_component",
            "local_cyan_blue_component",
            "core_color_jump_component",
            "new_hollow_structure",
        )
    ):
        reason_codes.append("CANDIDATE_MORPHOLOGY_FAILURE")
    if psf_closure.get("status") == "rejected":
        reason_codes.append("STAR_PSF_FWHM_FAILURE")
    psf_review_required = bool(psf_closure.get("review_required", False))
    if psf_review_required:
        reason_codes.append("STAGE9_PSF_SUBGROUP_EVIDENCE_INSUFFICIENT")
    return {
        "attempt": attempt,
        "formula": formula,
        "status": (
            "partial" if accepted and psf_review_required
            else "ok" if accepted
            else "rejected"
        ),
        "accepted": accepted,
        "review_required": psf_review_required,
        "gate_enabled": enabled,
        "issues": issues,
        "structural_issues": structural_issues,
        "gate_issues": gate_issues,
        "advisories": advisories,
        "quality_gates": quality_gates,
        "quality_advisory_multiplier": (
            stage7_quality.stage7_9_quality_advisory_multiplier(cfg)
        ),
        "reason_codes": reason_codes,
        "shadow_metrics": {"psf_scale": psf_scale_shadow},
        "psf_closure": psf_closure,
        "metrics": metrics,
        "limits": limits,
    }
