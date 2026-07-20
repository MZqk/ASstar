"""Pixel-space remix formula and deterministic Stage 9 quality gate."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import math

import numpy as np

from image_metrics import _box_blur_gray

try:
    from scipy import ndimage as scipy_ndimage
except ImportError:  # pragma: no cover - bundled runtime includes scipy
    scipy_ndimage = None


_MOTTLING_LOW_ABSOLUTE_SCORE_MAX = 0.10
_MOTTLING_LOW_ABSOLUTE_DELTA_MAX = 0.03
_MOTTLING_LOW_ABSOLUTE_CHANGED_RATIO_MAX = 0.12
_STAR_REFERENCE_MIN_COMPONENT_AREA = 3
_STAR_RECOVERY_DELTA = 0.002
_HOLLOW_STRUCTURE_MIN_AREA = 4


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
) -> Dict[str, Any]:
    """Measure local Stage 9 artifacts that whole-frame ratios can dilute."""
    component_peak_min = _bounded(
        getattr(cfg, "stage9_local_component_peak_min", 0.01),
        0.01,
        0.002,
        0.10,
    )
    component_area_max = int(
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
    cyan_area_max = int(
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
    core_jump_area_max = int(
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
    limits = {
        "local_connected_component_max_area": float(component_area_max),
        "local_nonstellar_shape_component_count": 0.0,
        "local_single_pixel_component_ratio": single_pixel_ratio_max,
        "local_cyan_blue_component_max_area": float(cyan_area_max),
        "core_color_jump_component_max_area": float(core_jump_area_max),
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
    component_mask = positive_peak > component_peak_min
    labels, component_count, component_areas = _component_areas(component_mask)
    component_max_area = int(np.max(component_areas)) if component_areas.size else 0
    single_pixel_ratio = (
        float(np.mean(component_areas == 1)) if component_areas.size else 0.0
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
        if area >= 4 and (
            aspect_ratio > component_aspect_max
            or fill_ratio < component_fill_min
        ):
            nonstellar_shape_areas.append(area)

    positive_rgb = _rgb_channels(positive_change)
    red, green, blue = positive_rgb
    cyan_blue_mask = (
        (positive_peak > cyan_peak_min)
        & (positive_saturation > cyan_saturation_min)
        & (blue > red * 1.20)
        & (((green + blue) * 0.5) > red * 1.25)
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
    core_jump_mask = (
        core_mask
        & (positive_peak > component_peak_min)
        & (color_jump > core_color_jump_min)
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
        "local_nonstellar_shape_component_count": len(nonstellar_shape_areas),
        "local_nonstellar_shape_component_max_area": (
            max(nonstellar_shape_areas) if nonstellar_shape_areas else 0
        ),
        "local_single_pixel_component_ratio": single_pixel_ratio,
        "local_cyan_blue_component_count": cyan_count,
        "local_cyan_blue_component_max_area": cyan_max_area,
        "local_cyan_blue_pixel_ratio": float(np.mean(cyan_blue_mask)),
        "local_cyan_blue_peak_min": cyan_peak_min,
        "local_cyan_blue_saturation_min": cyan_saturation_min,
        "core_signal_percentile": core_percentile,
        "core_signal_threshold": core_threshold,
        "core_color_jump_min": core_color_jump_min,
        "core_color_jump_component_count": core_jump_count,
        "core_color_jump_component_max_area": core_jump_max_area,
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
    for index, bounds in enumerate(objects, start=1):
        if bounds is None:
            continue
        area = int(areas[index])
        if area < _STAR_REFERENCE_MIN_COMPONENT_AREA:
            rejected_small_count += 1
            continue
        height = int(bounds[0].stop - bounds[0].start)
        width = int(bounds[1].stop - bounds[1].start)
        longest = max(height, width)
        shortest = max(1, min(height, width))
        fill_ratio = area / max(1, height * width)
        if (
            area <= 512
            and longest <= 64
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
        "min_component_area": _STAR_REFERENCE_MIN_COMPONENT_AREA,
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


def _build_source_matched_star_catalog(
    normalized_stars: np.ndarray,
    source_image: np.ndarray,
    cfg: Any,
    *,
    background: float,
    noise_sigma: float,
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
    detail_percentiles = [requested_detail_percentile]
    next_percentile = math.ceil((requested_detail_percentile + 0.01) * 2.0) / 2.0
    while next_percentile <= 99.5:
        detail_percentiles.append(next_percentile)
        next_percentile += 0.5

    attempts = []
    selected_components = None
    for detail_percentile in detail_percentiles:
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
                1 <= area <= 128
                and longest <= 16
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
        source_component_density = float(source_ids_arr.size / source_megapixels)
        source_single_pixel_ratio = float(np.mean(source_component_areas == 1))
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
                "component_count": int(source_ids_arr.size),
                "component_density_per_megapixel": source_component_density,
                "single_pixel_component_ratio": source_single_pixel_ratio,
                "density_limit_exceeded": density_limit_exceeded,
                "density_contamination_risk": density_contamination_risk,
                "single_pixel_limit_exceeded": single_pixel_limit_exceeded,
                "contamination_risk": contamination_risk,
            }
        )
        if not contamination_risk:
            selected_components = (
                detail_percentile,
                detail_threshold,
                labels,
                component_count,
                areas,
                source_ids_arr,
                source_component_density,
                source_single_pixel_ratio,
            )
            break

    if selected_components is None:
        if not attempts:
            return {
                "status": "unavailable",
                "reason": "no compact source-image star components",
            }
        last_attempt = attempts[-1]
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
    ) = selected_components
    source_positions = scipy_ndimage.maximum_position(
        source_peak,
        labels=labels,
        index=source_ids_arr,
    )
    source_y = np.asarray([item[0] for item in source_positions], dtype=np.int32)
    source_x = np.asarray([item[1] for item in source_positions], dtype=np.int32)
    local_star_peak = scipy_ndimage.maximum_filter(
        star_peak,
        size=5,
        mode="constant",
        cval=0.0,
    )
    reference_sigma = _bounded(
        getattr(cfg, "stage9_star_reference_sigma", 5.0),
        5.0,
        3.0,
        8.0,
    )
    match_threshold = max(
        float(background) + reference_sigma * float(noise_sigma),
        1e-6,
    )
    component_peaks_all = np.asarray(
        local_star_peak[source_y, source_x],
        dtype=np.float32,
    )
    matched = component_peaks_all > match_threshold
    component_ids = source_ids_arr[matched]
    component_peaks = component_peaks_all[matched]
    peak_y = source_y[matched]
    peak_x = source_x[matched]
    if component_ids.size < 4:
        return {
            "status": "unavailable",
            "reason": "too few source stars matched to starmask",
            "source_component_count": int(source_ids_arr.size),
            "matched_component_count": int(component_ids.size),
        }

    # Use the actual starmask maximum within the allowed source-match window.
    # This keeps the independent source catalog authoritative while tolerating a
    # one- or two-pixel separation offset from the starless tool.
    matched_peak_y = np.empty_like(peak_y)
    matched_peak_x = np.empty_like(peak_x)
    height, width = star_peak.shape
    for index, (source_center_y, source_center_x) in enumerate(zip(peak_y, peak_x)):
        y0 = max(0, int(source_center_y) - 2)
        y1 = min(height, int(source_center_y) + 3)
        x0 = max(0, int(source_center_x) - 2)
        x1 = min(width, int(source_center_x) + 3)
        window = star_peak[y0:y1, x0:x1]
        local_y, local_x = np.unravel_index(int(np.argmax(window)), window.shape)
        matched_peak_y[index] = y0 + int(local_y)
        matched_peak_x[index] = x0 + int(local_x)
    peak_y = matched_peak_y
    peak_x = matched_peak_x

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
        "reference_sigma": reference_sigma,
        "reference_threshold": match_threshold,
        "min_component_area": 1,
        "source_detail_percentile_requested": requested_detail_percentile,
        "source_detail_percentile": detail_percentile,
        "source_detail_threshold": detail_threshold,
        "source_detail_attempts": attempts,
        "source_detail_adaptive_retry": bool(
            detail_percentile > requested_detail_percentile
        ),
        "source_component_count": int(source_ids_arr.size),
        "source_component_density_per_megapixel": source_component_density,
        "source_component_density_max": source_component_density_max,
        "source_single_pixel_component_ratio": source_single_pixel_ratio,
        "source_single_pixel_component_ratio_max": source_single_pixel_ratio_max,
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
        "_labels": labels,
        "_component_ids": component_ids,
        "_component_peaks": component_peaks,
        "_component_areas": areas[component_ids].astype(np.int32, copy=False),
        "_peak_y": peak_y,
        "_peak_x": peak_x,
        "_weak_flags": weak_flags,
        "_weak_lookup": weak_lookup,
        "_bright_lookup": bright_lookup,
        "_peak_by_label": peak_by_label,
        "_weak_core_mask": weak_core_mask,
        "_bright_core_mask": bright_core_mask,
        "_reference_wing_ratio": reference_wing_ratio,
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
    return weak_support, bright_support, weak_support | bright_support


def build_star_overlay_masks(
    catalog: Dict[str, Any],
    *,
    strict: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return weak, bright and union masks for explicit top-layer composition."""
    return _catalog_support_masks(catalog, strict=strict)


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


def calibrate_starmask_asinh(
    stars: np.ndarray,
    cfg: Any,
    *,
    include_support_mask: bool = False,
    strict_support: bool = False,
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
    if weak_retention < min_weak_retention:
        return {
            "status": "rejected",
            "reason": (
                "compact_weak_star_retention "
                f"{weak_retention:.3f}<{min_weak_retention:.3f}"
            ),
            "weak_star_retention": weak_retention,
            "weak_star_retention_min": min_weak_retention,
            "star_reference": star_reference_summary(catalog),
        }

    signal = gray[support_mask & np.isfinite(gray) & (gray > signal_floor)]
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
    faint_target = _bounded(
        getattr(cfg, "stage9_starmask_faint_target", 0.26),
        0.26,
        0.08,
        0.40,
    )
    peak_target = _bounded(
        getattr(cfg, "stage9_starmask_peak_target", 0.90),
        0.90,
        0.75,
        0.95,
    )
    mid_target = min(
        peak_target - 0.10,
        max(
            faint_target + 0.03,
            _bounded(
                getattr(cfg, "stage9_starmask_mid_target", 0.50),
                0.50,
                0.30,
                0.70,
            ),
        ),
    )
    bright_target = min(
        peak_target - 0.03,
        max(
            mid_target + 0.03,
            _bounded(
                getattr(cfg, "stage9_starmask_bright_target", 0.75),
                0.75,
                0.50,
                0.88,
            ),
        ),
    )
    faint_limited = _solve_asinh_stretch(
        faint_value,
        offset,
        faint_target,
        stretch_max,
    )
    peak_limited = _solve_asinh_stretch(
        peak_value,
        offset,
        peak_target,
        stretch_max,
    )
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
        requested_weak_stretch = _bounded(
            faint_limited,
            2.0,
            1.10,
            stretch_max,
        )
        target_stretch = _bounded(
            requested_weak_stretch
            if mixed_star_field
            else min(faint_limited, peak_limited),
            2.0,
            1.10,
            stretch_max,
        )
    else:
        target_stretch = _bounded(
            getattr(cfg, "stage9_starmask_asinh_stretch", 2.0),
            2.0,
            1.10,
            3.0,
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
        getattr(cfg, "star_intensity", 1.0),
        1.0,
        0.10,
        1.05,
    )
    compact_normalized = apply_compact_starmask_support(normalized, support_mask)
    coverage_peak_map = _pixel_peak(compact_normalized)
    anchor_input_values = None
    anchor_output_targets = None
    if multi_anchor_enabled:
        anchor_input_values = np.asarray(
            [offset, faint_value, mid_value, bright_value, peak_value],
            dtype=np.float32,
        )
        anchor_output_targets = np.asarray(
            [0.0, faint_target, mid_target, bright_target, peak_target],
            dtype=np.float32,
        )
        multi_anchor_preview = _color_preserving_multi_anchor_curve(
            normalized,
            support_mask,
            input_anchors=anchor_input_values,
            output_anchors=anchor_output_targets,
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
        change_input_threshold = _anchor_input_threshold(
            anchor_input_values,
            anchor_output_targets,
            _STAR_RECOVERY_DELTA / reference_intensity,
        )
        stretch = float(faint_limited)
        coverage_limited = False
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
        "status": "ok",
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
        "stretch": float(stretch),
        "weak_stretch": float(stretch),
        "bright_stretch": float(bright_stretch),
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
        "faint_limited_stretch": float(faint_limited),
        "peak_limited_stretch": float(peak_limited),
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
        "mixed_star_field": mixed_star_field,
        "star_reference": star_reference_summary(catalog),
        "full_layer_predicted_change_ratio": full_layer_predicted_coverage,
        "removed_predicted_change_ratio": max(
            0.0,
            full_layer_predicted_coverage - float(predicted_coverage),
        ),
        "predicted_faint": (
            faint_target
            if multi_anchor_enabled
            else _asinh_sample(faint_value, stretch, offset)
        ),
        "predicted_mid": (
            mid_target
            if multi_anchor_enabled
            else _asinh_sample(mid_value, stretch, offset)
        ),
        "predicted_bright": (
            bright_target
            if multi_anchor_enabled
            else _asinh_sample(bright_value, stretch, offset)
        ),
        "predicted_peak": (
            peak_target
            if multi_anchor_enabled
            else _asinh_sample(peak_value, stretch, offset)
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
        result["_star_reference_catalog"] = catalog
    return result


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
    """Compare a Stage 9 candidate with its Stage 8 base and apply safety limits."""
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
    local_quality = _stage9_local_quality_metrics(
        base_norm,
        candidate_norm,
        positive_change,
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
                meaningful_holes = hollow_areas[
                    hollow_areas >= _HOLLOW_STRUCTURE_MIN_AREA
                ]
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
            iterations=3,
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

    recovery_status = "unavailable"
    recovery_reason = "star reference catalog missing"
    if isinstance(star_reference, dict) and star_reference.get("status") == "ok":
        peak_y = np.asarray(star_reference.get("_peak_y", ()), dtype=np.int32)
        peak_x = np.asarray(star_reference.get("_peak_x", ()), dtype=np.int32)
        weak_flags = np.asarray(
            star_reference.get("_weak_flags", ()),
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
            local_positive_peak = (
                scipy_ndimage.maximum_filter(
                    positive_change_peak,
                    size=3,
                    mode="constant",
                    cval=0.0,
                )
                if scipy_ndimage is not None
                else positive_change_peak
            )
            restored = np.asarray(
                local_positive_peak[peak_y, peak_x] >= _STAR_RECOVERY_DELTA,
                dtype=bool,
            )
            if scipy_ndimage is not None:
                aperture3_map = (
                    scipy_ndimage.uniform_filter(
                        positive_change_peak,
                        size=3,
                        mode="constant",
                    )
                    * 9.0
                )
                aperture7_map = (
                    scipy_ndimage.uniform_filter(
                        positive_change_peak,
                        size=7,
                        mode="constant",
                    )
                    * 49.0
                )
            else:
                aperture3_map = positive_change_peak
                aperture7_map = positive_change_peak
            aperture3 = np.asarray(aperture3_map[peak_y, peak_x], dtype=np.float32)
            aperture7 = np.asarray(aperture7_map[peak_y, peak_x], dtype=np.float32)
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
            else:
                wing_reference_count = 0
                wing_recovery_ratio = 1.0
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
                    "star_aperture_flux_min": 0.006,
                    "star_aperture_restored_count": int(
                        np.count_nonzero(aperture_restored)
                    ),
                    "star_aperture_recovery_ratio": float(
                        np.mean(aperture_restored)
                    ),
                    "star_wing_reference_count": wing_reference_count,
                    "star_wing_recovery_ratio": wing_recovery_ratio,
                }
            )
            if star_exclusion_mask is not None and scipy_ndimage is not None:
                local_background = scipy_ndimage.median_filter(
                    base_luminance,
                    size=7,
                    mode="reflect",
                )
                base_local_min = scipy_ndimage.minimum_filter(
                    base_luminance,
                    size=3,
                    mode="reflect",
                )
                candidate_local_peak = scipy_ndimage.maximum_filter(
                    candidate_luminance,
                    size=3,
                    mode="reflect",
                )
                initial_holes = (
                    local_background[peak_y, peak_x]
                    - base_local_min[peak_y, peak_x]
                    > 0.002
                )
                initial_hole_count = int(np.count_nonzero(initial_holes))
                residual_holes = initial_holes & (
                    local_background[peak_y, peak_x]
                    - candidate_local_peak[peak_y, peak_x]
                    > 0.001
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
    issues = []
    if metrics["finite_ratio"] < 1.0:
        issues.append(f"non-finite pixels: finite_ratio={metrics['finite_ratio']:.6f}")
    if local_quality.get("status") != "ok":
        issues.append(
            "local_quality_metrics_unavailable: "
            f"{local_quality.get('reason') or 'unknown reason'}"
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
        "star_aperture_recovery_ratio": _bounded(
            getattr(cfg, "stage9_star_aperture_recovery_ratio_min", 0.75),
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
            getattr(cfg, "stage9_new_hollow_structure_area_max", 64),
            64.0,
            4.0,
            4096.0,
        ),
    }
    limits.update(local_quality.get("limits") or {})
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
            issues.append(f"{metric_name} {value:.6f}>{limit:.6f}")

    if recovery_status != "ok":
        issues.append(f"star_recovery_metrics_unavailable: {recovery_reason}")
    else:
        for metric_name in (
            "weak_star_recovery_ratio",
            "star_recovery_ratio",
            "star_aperture_recovery_ratio",
            "star_wing_recovery_ratio",
        ):
            value = float(metrics[metric_name])
            limit = float(limits[metric_name])
            if value < limit:
                issues.append(f"{metric_name} {value:.6f}<{limit:.6f}")

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
        issues.append(
            "background_mottling_growth "
            f"{mottling_growth:.6f}>{mottling_growth_limit:.6f} "
            f"(after={mottling_after:.6f}, delta={mottling_delta:.6f})"
        )

    accepted = not enabled or not issues
    return {
        "attempt": attempt,
        "formula": formula,
        "status": "ok" if accepted else "rejected",
        "accepted": accepted,
        "gate_enabled": enabled,
        "issues": issues,
        "metrics": metrics,
        "limits": limits,
    }
