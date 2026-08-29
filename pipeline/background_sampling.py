"""Safe Stage 3 background samples and directional-pattern diagnostics."""
from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from stage3_contract import (
    STAGE3_DENSE_STAR_FIELD_FRACTION_MIN,
    STAGE3_DENSE_STAR_SAMPLE_SUPPORT_MIN,
    STAGE3_DENSE_STAR_SAMPLING_SCHEMA,
    STAGE3_MAX_SOURCE_MASK_FRACTION,
    STAGE3_MIN_AXIS_SPAN_RATIO,
    STAGE3_MIN_SPATIAL_GRID_CELLS,
    STAGE3_MIN_SPATIAL_QUADRANTS,
    STAGE3_MIN_USABLE_SKY_FRACTION,
    STAGE3_MIN_VALIDATION_PATCHES,
    STAGE3_NEUTRAL_AXIS_CONDITION_MAX,
    STAGE3_NEUTRAL_AXIS_HEADROOM_FRACTION_MAX,
    STAGE3_NEUTRAL_AXIS_HEADROOM_MARGIN,
    STAGE3_NEUTRAL_AXIS_PROJECTION_SCHEMA,
    STAGE3_SPATIAL_OPPONENT_CORRELATION_MIN,
    STAGE3_SPATIAL_OPPONENT_PROJECTION_SCHEMA,
    STAGE3_SPATIAL_OPPONENT_RMS_IMPROVEMENT_MIN,
    STAGE3_RADIAL_BIC_DELTA_MIN,
    STAGE3_RADIAL_RESIDUAL_SPAN_RATIO_MAX,
    STAGE3_SAMPLE_MASK_SCHEMA,
    STAGE3_SIGNIFICANCE_SIGMA,
    normalize_stage3_gate_profile,
    stage3_gate_thresholds,
)


STAGE3_PROCESS_EVIDENCE_SCHEMA = "starun.stage3-process-evidence.v1"
STAGE3_BACKGROUND_VALIDATION_SCHEMA = "starun.stage3-background-validation.v1"


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def _as_luminance(image: Any) -> np.ndarray:
    source = np.asarray(image)
    if source.size == 0:
        raise ValueError("image buffer is empty")
    if source.ndim == 2:
        luminance = source
    elif source.ndim == 3:
        if source.shape[0] in (1, 3, 4) and source.shape[-1] not in (1, 3, 4):
            channels = source[:3]
        elif source.shape[-1] in (1, 3, 4):
            channels = np.moveaxis(source[..., :3], -1, 0)
        elif source.shape[0] >= 3:
            channels = source[:3]
        else:
            raise ValueError(f"unsupported image shape: {source.shape}")
        if channels.shape[0] == 1:
            luminance = channels[0]
        else:
            luminance = (
                0.2126 * channels[0]
                + 0.7152 * channels[1]
                + 0.0722 * channels[2]
            )
    else:
        raise ValueError(f"unsupported image ndim: {source.ndim}")

    values = np.asarray(luminance, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size < 256:
        raise ValueError("not enough finite pixels")
    low, high = (float(value) for value in np.quantile(finite, (0.005, 0.995)))
    span = high - low
    if not math.isfinite(span) or span <= 1e-12:
        raise ValueError("image dynamic range is insufficient")
    return np.clip(
        np.nan_to_num((values - low) / span, nan=0.0, posinf=1.0, neginf=0.0),
        0.0,
        1.0,
    )


def _native_channels(image: Any) -> np.ndarray:
    """Return native-value channels in CHW order without renormalization."""
    source = np.asarray(image)
    if source.size == 0:
        raise ValueError("image buffer is empty")
    if source.ndim == 2:
        channels = source[None, :, :]
    elif source.ndim == 3:
        if source.shape[0] in (1, 3, 4) and source.shape[-1] not in (1, 3, 4):
            channels = source[:3]
        elif source.shape[-1] in (1, 3, 4):
            channels = np.moveaxis(source[..., :3], -1, 0)
        elif source.shape[0] >= 3:
            channels = source[:3]
        else:
            raise ValueError(f"unsupported image shape: {source.shape}")
    else:
        raise ValueError(f"unsupported image ndim: {source.ndim}")
    values = np.asarray(channels, dtype=np.float64)
    if values.shape[0] not in (1, 3) or values.shape[1] < 2 or values.shape[2] < 2:
        raise ValueError(f"unsupported channel layout: {source.shape}")
    if int(np.count_nonzero(np.isfinite(values))) < 256:
        raise ValueError("not enough finite channel pixels")
    return values


def _expand_analysis_mask(
    mask: np.ndarray,
    *,
    height: int,
    width: int,
    stride: int,
) -> np.ndarray:
    """Project a bounded analysis mask back to native pixels deterministically."""
    source = np.asarray(mask, dtype=bool)
    y_index = np.clip(
        np.rint(np.arange(height, dtype=np.float64) / max(1, stride)).astype(int),
        0,
        source.shape[0] - 1,
    )
    x_index = np.clip(
        np.rint(np.arange(width, dtype=np.float64) / max(1, stride)).astype(int),
        0,
        source.shape[1] - 1,
    )
    return source[np.ix_(y_index, x_index)]


def _mask_sha256(mask: np.ndarray) -> str:
    packed = np.packbits(np.asarray(mask, dtype=np.uint8).reshape(-1))
    return hashlib.sha256(packed.tobytes()).hexdigest()


def _smooth3(values: np.ndarray, passes: int) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64)
    height, width = output.shape
    for _ in range(max(1, int(passes))):
        padded = np.pad(output, 1, mode="reflect")
        output = sum(
            padded[dy : dy + height, dx : dx + width]
            for dy in range(3)
            for dx in range(3)
        ) / 9.0
    return output


def _smooth_profile(values: np.ndarray, radius: int = 3) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    radius = max(1, min(int(radius), max(1, source.size // 4)))
    padded = np.pad(source, radius, mode="reflect")
    kernel = np.ones(2 * radius + 1, dtype=np.float64) / (2 * radius + 1)
    return np.convolve(padded, kernel, mode="valid")


def _robust_sigma(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    center = float(np.median(finite))
    return 1.4826 * float(np.median(np.abs(finite - center)))


def _patch_median_uncertainty(values: np.ndarray) -> Tuple[float, Dict[str, Any]]:
    """Estimate a patch median uncertainty without assuming independent pixels.

    Registered/debayered/denoised astronomy images commonly contain correlated
    neighbouring pixels.  The ordinary ``sigma / sqrt(pixel_count)`` estimate
    is therefore only a lower bound.  Non-overlapping block medians provide a
    conservative effective-sample estimate while keeping the calculation
    deterministic and dependency-free.
    """
    patch = np.asarray(values, dtype=np.float64)
    finite = patch[np.isfinite(patch)]
    if patch.ndim != 2 or finite.size < 16:
        return math.inf, {
            "method": "unavailable",
            "reason": "patch is not a complete two-dimensional sample",
            "block_count": 0,
        }

    pixel_sigma = _robust_sigma(finite)
    pixel_standard_error = 1.2533 * pixel_sigma / math.sqrt(finite.size)
    block_side = max(2, min(patch.shape) // 5)
    block_medians: List[float] = []
    for y0 in range(0, patch.shape[0], block_side):
        for x0 in range(0, patch.shape[1], block_side):
            block = patch[
                y0 : min(patch.shape[0], y0 + block_side),
                x0 : min(patch.shape[1], x0 + block_side),
            ]
            block_finite = block[np.isfinite(block)]
            if block_finite.size < max(4, block.size // 2):
                continue
            block_medians.append(float(np.median(block_finite)))

    if len(block_medians) >= 6:
        block_sigma = _robust_sigma(np.asarray(block_medians, dtype=np.float64))
        block_standard_error = (
            1.2533 * block_sigma / math.sqrt(len(block_medians))
        )
        uncertainty = max(pixel_standard_error, block_standard_error)
        method = "max_pixel_and_nonoverlapping_block_median_standard_error"
    else:
        block_standard_error = None
        uncertainty = pixel_standard_error
        method = "pixel_standard_error_fallback"

    return max(float(uncertainty), np.finfo(np.float64).eps), {
        "method": method,
        "pixel_standard_error": float(pixel_standard_error),
        "block_standard_error": (
            float(block_standard_error)
            if block_standard_error is not None
            else None
        ),
        "block_side": int(block_side),
        "block_count": len(block_medians),
    }


def background_span_standard_error(validation: Dict[str, Any]) -> float:
    """Return the one-sigma uncertainty of a P90-P10 patch-median span."""
    try:
        patch_uncertainty = float(validation["patch_median_uncertainty"])
    except (KeyError, TypeError, ValueError):
        try:
            patch_rms = float(validation["patch_mad_median"])
            radius = max(1, int(validation.get("patch_radius", 12) or 12))
        except (KeyError, TypeError, ValueError):
            return math.inf
        patch_pixels = float((2 * radius + 1) ** 2)
        patch_uncertainty = 1.2533 * patch_rms / math.sqrt(patch_pixels)
    if not math.isfinite(patch_uncertainty) or patch_uncertainty < 0.0:
        return math.inf
    # A span is the difference between two independently estimated quantiles.
    return math.sqrt(2.0) * max(
        patch_uncertainty,
        np.finfo(np.float64).eps,
    )


def _dilate_mask(mask: np.ndarray, passes: int = 2) -> np.ndarray:
    """Grow a boolean mask without adding a scipy runtime dependency."""
    output = np.asarray(mask, dtype=bool)
    height, width = output.shape
    for _ in range(max(0, int(passes))):
        padded = np.pad(output, 1, mode="constant", constant_values=False)
        output = np.logical_or.reduce(
            [
                padded[dy : dy + height, dx : dx + width]
                for dy in range(3)
                for dx in range(3)
            ]
        )
    return output


def _build_source_and_coverage_masks(
    luminance: np.ndarray,
    *,
    max_side: int = 512,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]:
    """Build explicit source/blank masks on a bounded analysis grid.

    The source mask is derived from an iteratively clipped quadratic sky fit,
    then grown to include stellar wings and coherent extended structure.  The
    coverage mask is deliberately narrower: it only marks non-finite pixels or
    a repeated detector/image floor, not merely the darkest valid sky.
    """
    height, width = luminance.shape
    stride = max(1, int(math.ceil(max(height, width) / max(64, int(max_side)))))
    small = np.asarray(luminance[::stride, ::stride], dtype=np.float64)
    finite = np.isfinite(small)
    finite_values = small[finite]
    if finite_values.size < 256:
        raise ValueError("not enough finite pixels for source masking")

    low, high = (float(value) for value in np.quantile(finite_values, (0.005, 0.995)))
    dynamic = max(high - low, 1e-8)
    floor = float(np.min(finite_values))
    floor_tolerance = max(dynamic * 1e-7, np.finfo(np.float64).eps * 64.0)
    repeated_floor = np.abs(small - floor) <= floor_tolerance
    floor_fraction = float(np.mean(repeated_floor & finite))
    coverage_mask = ~finite
    if floor_fraction >= 0.002:
        coverage_mask |= repeated_floor

    grid_y, grid_x = np.mgrid[: small.shape[0], : small.shape[1]]
    nx = grid_x / max(small.shape[1] - 1, 1)
    ny = grid_y / max(small.shape[0] - 1, 1)
    design = np.stack(
        (
            np.ones_like(nx),
            nx,
            ny,
            nx * nx,
            nx * ny,
            ny * ny,
        ),
        axis=-1,
    ).reshape(-1, 6)
    values = small.reshape(-1)
    fit_mask = (~coverage_mask).reshape(-1)
    coefficients = np.zeros(6, dtype=np.float64)
    for _ in range(5):
        if int(np.count_nonzero(fit_mask)) < 64:
            break
        coefficients, *_ = np.linalg.lstsq(
            design[fit_mask],
            values[fit_mask],
            rcond=None,
        )
        residual = values - design @ coefficients
        center = float(np.median(residual[fit_mask]))
        sigma = max(_robust_sigma(residual[fit_mask]), dynamic * 1e-4)
        next_mask = (
            (~coverage_mask).reshape(-1)
            & (residual >= center - 3.5 * sigma)
            & (residual <= center + 1.6 * sigma)
        )
        if np.array_equal(next_mask, fit_mask):
            break
        fit_mask = next_mask

    fitted = (design @ coefficients).reshape(small.shape)
    residual = small - fitted
    usable = ~coverage_mask
    center = float(np.median(residual[usable]))
    sigma = max(_robust_sigma(residual[usable]), dynamic * 1e-4)
    coherent_support = _smooth3(
        (residual > center + 1.50 * sigma).astype(np.float64),
        passes=2,
    )
    coherent = (
        coherent_support >= 0.70
    ) & (residual > center + 0.50 * sigma)
    high_frequency = small - _smooth3(small, passes=2)
    high_frequency_sigma = max(_robust_sigma(high_frequency[usable]), dynamic * 1e-4)
    compact = high_frequency > 4.0 * high_frequency_sigma
    compact_mask = _dilate_mask(compact & usable, passes=1)
    extended_structure_mask = _dilate_mask(
        coherent & usable,
        passes=2,
    )
    bright_core_seed = (
        (
            (residual > center + 5.0 * sigma)
            | (small >= float(np.quantile(finite_values, 0.995)))
        )
        & usable
    )
    # A high residual alone is not evidence of an extended bright core: in a
    # dense Milky-Way field it is commonly an isolated stellar peak that is
    # already protected by both compact-source and scene-support masks.  Keep
    # the bright-core layer tied to coherent positive structure so it cannot
    # grow thousands of stellar seeds into a near-full-frame duplicate mask.
    bright_core_mask = bright_core_seed & extended_structure_mask
    negative_support = _smooth3(
        (residual < center - 1.50 * sigma).astype(np.float64),
        passes=2,
    )
    dark_structure_mask = _dilate_mask(
        (
            (negative_support >= 0.70)
            & (residual < center - 0.50 * sigma)
        )
        & usable,
        passes=1,
    )
    outer_halo_mask = _dilate_mask(
        (
            (coherent_support >= 0.55)
            & (residual > center + 0.35 * sigma)
            & usable
        ),
        passes=3,
    )
    source_mask = compact_mask | extended_structure_mask | bright_core_mask
    source_mask &= usable

    sky_mask = usable & ~source_mask
    sky_cells = set()
    for cell_y in range(4):
        y0 = int(round(cell_y * small.shape[0] / 4.0))
        y1 = int(round((cell_y + 1) * small.shape[0] / 4.0))
        for cell_x in range(4):
            x0 = int(round(cell_x * small.shape[1] / 4.0))
            x1 = int(round((cell_x + 1) * small.shape[1] / 4.0))
            cell = sky_mask[y0:y1, x0:x1]
            if cell.size and float(np.mean(cell)) >= 0.10:
                sky_cells.add((cell_x, cell_y))
    layers = {
        "compact_source": compact_mask,
        "extended_structure": extended_structure_mask,
        "bright_core": bright_core_mask,
        "dark_structure": dark_structure_mask,
        "outer_halo": outer_halo_mask,
    }
    report = {
        "status": "ready",
        "analysis_shape": [int(small.shape[0]), int(small.shape[1])],
        "downsample_stride": stride,
        "source_mask_fraction": float(np.mean(source_mask)),
        "coverage_mask_fraction": float(np.mean(coverage_mask)),
        "usable_sky_fraction": float(np.mean(sky_mask)),
        "usable_sky_grid_cells": len(sky_cells),
        "usable_sky_grid_cell_ratio": len(sky_cells) / 16.0,
        "fit_residual_sigma": sigma,
        "mask_method": "iterative_quadratic_clip_plus_coherent_structure_growth",
        "layer_fractions": {
            name: float(np.mean(mask)) for name, mask in layers.items()
        },
    }
    return source_mask, coverage_mask, layers, report


def _offset_correlation(
    residual: np.ndarray,
    mask: np.ndarray,
    *,
    dx: int,
    dy: int,
) -> float:
    height, width = residual.shape
    if abs(dx) >= width or abs(dy) >= height:
        return 0.0
    x0 = max(0, -dx)
    x1 = min(width, width - dx)
    y0 = max(0, -dy)
    y1 = min(height, height - dy)
    left = residual[y0:y1, x0:x1]
    right = residual[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
    valid = mask[y0:y1, x0:x1] & mask[
        y0 + dy : y1 + dy,
        x0 + dx : x1 + dx,
    ]
    if int(np.count_nonzero(valid)) < 128:
        return 0.0
    a = left[valid]
    b = right[valid]
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    denominator = math.sqrt(float(np.sum(a * a)) * float(np.sum(b * b)))
    if denominator <= 1e-12:
        return 0.0
    return _clamp(float(np.sum(a * b)) / denominator, -1.0, 1.0)


def analyze_directional_pattern_noise(
    image: Any,
    *,
    detection_threshold: float = 0.55,
    walking_threshold: float = 0.50,
    max_side: int = 512,
) -> Dict[str, Any]:
    """Measure row/column banding and diagonal walking-noise coherence."""
    detection_threshold = _clamp(detection_threshold, 0.25, 0.90)
    walking_threshold = _clamp(walking_threshold, 0.25, 0.90)
    try:
        luminance = _as_luminance(image)
    except (TypeError, ValueError) as error:
        return {
            "schema_version": STAGE3_BACKGROUND_VALIDATION_SCHEMA,
            "status": "unavailable",
            "detected": False,
            "error": str(error),
            "thresholds": {
                "pattern_score_min": detection_threshold,
                "walking_noise_score_min": walking_threshold,
            },
        }

    height, width = luminance.shape
    stride = max(1, int(math.ceil(max(height, width) / max(64, int(max_side)))))
    small = luminance[::stride, ::stride]
    background_limit = float(np.quantile(small, 0.60))
    background_mask = np.isfinite(small) & (small <= background_limit)
    if int(np.count_nonzero(background_mask)) < 512:
        return {
            "status": "unavailable",
            "detected": False,
            "error": "insufficient background pixels for directional analysis",
            "thresholds": {
                "pattern_score_min": detection_threshold,
                "walking_noise_score_min": walking_threshold,
            },
        }

    local_model = _smooth3(small, passes=4)
    residual = small - local_model
    residual_sigma = max(_robust_sigma(residual[background_mask]), 1e-8)
    correlations = {
        "horizontal": _offset_correlation(
            residual,
            background_mask,
            dx=1,
            dy=0,
        ),
        "vertical": _offset_correlation(
            residual,
            background_mask,
            dx=0,
            dy=1,
        ),
        "diagonal_down": _offset_correlation(
            residual,
            background_mask,
            dx=1,
            dy=1,
        ),
        "diagonal_up": _offset_correlation(
            residual,
            background_mask,
            dx=1,
            dy=-1,
        ),
    }

    broad_residual = small - _smooth3(small, passes=12)
    row_profile = np.median(broad_residual, axis=1)
    column_profile = np.median(broad_residual, axis=0)
    row_ratio = _robust_sigma(row_profile - _smooth_profile(row_profile)) / residual_sigma
    column_ratio = (
        _robust_sigma(column_profile - _smooth_profile(column_profile))
        / residual_sigma
    )
    row_band_score = _clamp((row_ratio - 0.04) / 0.30)
    column_band_score = _clamp((column_ratio - 0.04) / 0.30)
    stripe_score = max(row_band_score, column_band_score)

    axial_correlation = max(
        0.0,
        correlations["horizontal"],
        correlations["vertical"],
    )
    diagonal_name = max(
        ("diagonal_down", "diagonal_up"),
        key=lambda name: correlations[name],
    )
    diagonal_correlation = max(0.0, correlations[diagonal_name])
    # True walking-noise streaks retain materially more coherence along one
    # diagonal than along either image axis. Plain row/column banding also has
    # diagonal correlation, so comparing against the strongest axial value is
    # essential to keep the routes separate.
    walking_score = _clamp(
        (
            diagonal_correlation
            - axial_correlation
            - 0.04
        )
        / 0.25
    )
    sorted_correlations = sorted(
        (max(0.0, float(value)) for value in correlations.values()),
        reverse=True,
    )
    directional_correlation = sorted_correlations[0]
    other_direction_median = float(np.median(sorted_correlations[1:]))
    directional_anisotropy = max(
        0.0,
        directional_correlation - other_direction_median,
    )
    # Natural nebulosity and broadly correlated background texture can retain
    # high correlation in every sampled direction after local smoothing.  A
    # directional artifact must therefore be anisotropic, not merely
    # correlated.  Row/column banding and diagonal walking noise keep their
    # dedicated scores above; this term only catches other dominant directions.
    directional_score = _clamp((directional_anisotropy - 0.08) / 0.30)
    pattern_score = max(stripe_score, walking_score, directional_score)
    walking_detected = walking_score >= walking_threshold
    detected = bool(
        pattern_score >= detection_threshold or walking_detected
    )
    if not detected:
        kind = "none"
        orientation = None
    elif walking_detected:
        kind = "diagonal_walking_noise"
        orientation = diagonal_name
    elif row_band_score >= column_band_score and stripe_score >= directional_score:
        kind = "horizontal_banding"
        orientation = "horizontal"
    elif column_band_score > row_band_score and stripe_score >= directional_score:
        kind = "vertical_banding"
        orientation = "vertical"
    else:
        kind = "directional_pattern_noise"
        orientation = max(correlations, key=correlations.get)

    return {
        "status": "ok",
        "detected": detected,
        "kind": kind,
        "orientation": orientation,
        "pattern_score": pattern_score,
        "stripe_score": stripe_score,
        "walking_noise_score": walking_score,
        "directional_score": directional_score,
        "directional_anisotropy": directional_anisotropy,
        "row_banding_ratio": row_ratio,
        "column_banding_ratio": column_ratio,
        "directional_correlations": correlations,
        "analysis_shape": [int(small.shape[0]), int(small.shape[1])],
        "source_shape": [int(height), int(width)],
        "downsample_stride": stride,
        "background_pixel_count": int(np.count_nonzero(background_mask)),
        "thresholds": {
            "pattern_score_min": detection_threshold,
            "walking_noise_score_min": walking_threshold,
        },
        "interpretation": (
            "directional pattern noise is not a low-frequency sky gradient"
            if detected
            else "no material directional pattern noise detected"
        ),
    }


def build_safe_background_samples(
    image: Any,
    *,
    target_count: int = 40,
    min_count: int = 16,
    patch_radius: int = 12,
    brightness_quantile_max: float = 0.70,
    texture_quantile_max: float = 0.55,
    candidate_refinement: bool = True,
    preserve_regular_grid: bool = False,
    masked_catalog_statistics: bool = False,
    refinement_max_steps: int = 8,
    shared_valid_mask: Optional[np.ndarray] = None,
    shared_saturation_map: Optional[np.ndarray] = None,
    shared_star_catalog: Optional[Sequence[Dict[str, Any]]] = None,
    shared_star_catalog_sha256: Optional[str] = None,
    protection_policy: Optional[Dict[str, Any]] = None,
    return_candidate_independent_support_mask: bool = False,
) -> Tuple[List[Tuple[float, float]], Dict[str, Any]]:
    """Select spatially distributed low-signal, low-texture Siril samples.

    A deterministic dark-patch refinement pass may propose extra coordinates
    between the regular grid points.  It never relaxes the thresholds learned
    from the original grid: every proposal must pass the same source/coverage
    masks, brightness, texture, star-excess and spacing audits before it can be
    used by Siril or by the held-out validation split.
    """
    target_count = max(16, min(64, int(target_count)))
    min_count = max(12, min(target_count, int(min_count)))
    brightness_quantile_max = _clamp(brightness_quantile_max, 0.50, 0.85)
    texture_quantile_max = _clamp(texture_quantile_max, 0.25, 0.75)
    candidate_refinement = bool(candidate_refinement)
    preserve_regular_grid = bool(preserve_regular_grid)
    masked_catalog_statistics = bool(masked_catalog_statistics)
    return_candidate_independent_support_mask = bool(
        return_candidate_independent_support_mask
    )
    refinement_max_steps = max(0, min(12, int(refinement_max_steps)))
    try:
        luminance = _as_luminance(image)
        native_channels = (
            _native_channels(image)
            if masked_catalog_statistics
            else None
        )
    except (TypeError, ValueError) as error:
        return [], {
            "status": "unavailable",
            "error": str(error),
            "sample_count": 0,
        }

    height, width = luminance.shape
    protection_policy = dict(protection_policy or {})
    shared_valid = None
    shared_saturated = None
    shared_star_mask = None
    shared_support_status: Dict[str, Any] = {
        "valid_mask": "unavailable",
        "saturation_map": "unavailable",
        "star_catalog": "unavailable",
    }
    if shared_valid_mask is not None:
        candidate = np.asarray(shared_valid_mask, dtype=bool)
        if candidate.shape == (height, width):
            shared_valid = candidate
            shared_support_status["valid_mask"] = "applied"
    if shared_saturation_map is not None:
        candidate = np.asarray(shared_saturation_map)
        if candidate.shape == (height, width):
            shared_saturated = candidate > 0
            shared_support_status["saturation_map"] = "applied"
    if shared_star_catalog is not None:
        shared_star_mask = np.zeros((height, width), dtype=bool)
        accepted_stars = 0
        for star in shared_star_catalog:
            if not isinstance(star, dict):
                continue
            try:
                x = float(star.get("x"))
                y = float(star.get("y"))
                fwhm = float(star.get("fwhm_px"))
            except (TypeError, ValueError):
                continue
            if not all(math.isfinite(value) for value in (x, y, fwhm)) or fwhm <= 0:
                continue
            radius_scale = (
                4.0
                if bool(protection_policy.get("protect_star_halo", False))
                else 2.5
            )
            radius_cap = 48.0 if radius_scale > 2.5 else 32.0
            radius = max(3.0, min(radius_cap, radius_scale * fwhm))
            extent = int(math.ceil(radius))
            x0 = max(0, int(math.floor(x)) - extent)
            x1 = min(width, int(math.floor(x)) + extent + 1)
            y0 = max(0, int(math.floor(y)) - extent)
            y1 = min(height, int(math.floor(y)) + extent + 1)
            if x0 >= x1 or y0 >= y1:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1]
            shared_star_mask[y0:y1, x0:x1] |= (
                (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
            )
            accepted_stars += 1
        shared_support_status["star_catalog"] = (
            "applied" if accepted_stars else "available_empty"
        )
        shared_support_status["catalog_star_count"] = accepted_stars
    patch_radius = max(
        4,
        min(int(patch_radius), 24, max(4, min(height, width) // 12)),
    )
    margin = max(patch_radius + 2, int(round(min(height, width) * 0.04)))
    if width <= margin * 2 + 4 or height <= margin * 2 + 4:
        return [], {
            "status": "unavailable",
            "error": "image is too small for safe sample margins",
            "sample_count": 0,
        }

    try:
        (
            source_mask,
            coverage_mask,
            derived_masks,
            mask_report,
        ) = _build_source_and_coverage_masks(luminance)
    except (TypeError, ValueError, np.linalg.LinAlgError) as error:
        return [], {
            "status": "unavailable",
            "error": f"source/coverage mask failed: {error}",
            "sample_count": 0,
        }
    mask_stride = max(1, int(mask_report.get("downsample_stride", 1) or 1))
    noncatalog_mask = source_mask | coverage_mask | derived_masks["dark_structure"]
    combined_mask = noncatalog_mask.copy()
    applied_layers = {
        "invalid_or_uncovered": coverage_mask,
        "image_stars_and_sources": derived_masks["compact_source"],
        "positive_structure_nebulosity": derived_masks["extended_structure"],
        "bright_core": derived_masks["bright_core"],
        "dark_structure": derived_masks["dark_structure"],
    }
    if bool(protection_policy.get("protect_outer_halo", False)):
        applied_layers["outer_halo"] = derived_masks["outer_halo"]
        combined_mask |= derived_masks["outer_halo"]
        noncatalog_mask |= derived_masks["outer_halo"]

    star_fraction = (
        float(np.mean(shared_star_mask))
        if shared_star_mask is not None
        else None
    )
    if shared_star_mask is not None:
        star_small = shared_star_mask[::mask_stride, ::mask_stride]
        star_small = star_small[: combined_mask.shape[0], : combined_mask.shape[1]]
        combined_mask[: star_small.shape[0], : star_small.shape[1]] |= star_small
    strict_unmasked_sky_fraction = float(np.mean(~combined_mask))
    nonstellar_sky_fraction = float(np.mean(~noncatalog_mask))
    effective_usable_sky_fraction = (
        nonstellar_sky_fraction
        if masked_catalog_statistics
        else strict_unmasked_sky_fraction
    )
    mask_evidence = {
        "schema_version": STAGE3_SAMPLE_MASK_SCHEMA,
        "method": "multiscale_quadratic_residual_plus_scene_support",
        "applied_to_sampling": True,
        "layers": {
            name: {
                "requested": True,
                "available": True,
                "applied": True,
                "pixel_fraction": float(np.mean(mask)),
                "method": {
                    "invalid_or_uncovered": "finite_and_repeated_floor",
                    "image_stars_and_sources": (
                        "high_frequency_4sigma_compact_source_growth"
                    ),
                    "positive_structure_nebulosity": (
                        "coherent_positive_quadratic_residual_growth"
                    ),
                    "bright_core": (
                        "coherent_positive_structure_intersect_5sigma_or_p995"
                    ),
                    "dark_structure": (
                        "coherent_negative_quadratic_residual_growth"
                    ),
                    "outer_halo": (
                        "low_threshold_connected_positive_residual_growth"
                    ),
                }[name],
                "reason": None,
            }
            for name, mask in applied_layers.items()
        },
        "combined_excluded_fraction": float(np.mean(combined_mask)),
        "usable_sky_fraction": effective_usable_sky_fraction,
        "strict_unmasked_sky_fraction": strict_unmasked_sky_fraction,
        "nonstellar_sky_fraction": nonstellar_sky_fraction,
        "effective_usable_sky_fraction": effective_usable_sky_fraction,
        "effective_definition": (
            "nonstellar_sky_with_catalog_points_masked_in_sample_statistics"
            if masked_catalog_statistics
            else "strict_zero_overlap_combined_mask"
        ),
        "scene_support_stars": {
            "requested": True,
            "available": shared_star_mask is not None,
            "applied": shared_star_mask is not None,
            "pixel_fraction": star_fraction,
            "method": (
                "scene_support_catalog_4x_fwhm"
                if bool(protection_policy.get("protect_star_halo", False))
                else "scene_support_catalog_2_5x_fwhm"
            ),
            "reason": (
                None
                if shared_star_mask is not None
                else "scene_support_star_catalog_unavailable_image_source_mask_used"
            ),
        },
    }
    mask_evidence["layers"]["scene_support_stars"] = mask_evidence.pop(
        "scene_support_stars"
    )
    if "outer_halo" not in mask_evidence["layers"]:
        mask_evidence["layers"]["outer_halo"] = {
            "requested": False,
            "available": True,
            "applied": False,
            "pixel_fraction": float(np.mean(derived_masks["outer_halo"])),
            "method": "low_threshold_connected_positive_residual_growth",
            "reason": "protect_outer_halo_not_requested",
        }

    compact_native = None
    hard_exclusion_native = None
    masked_pixel_support = None
    full_combined_exclusion = _expand_analysis_mask(
        combined_mask,
        height=height,
        width=width,
        stride=mask_stride,
    )
    candidate_independent_sky_support = ~full_combined_exclusion
    if shared_valid is not None:
        candidate_independent_sky_support &= shared_valid
    if shared_saturated is not None:
        candidate_independent_sky_support &= ~shared_saturated
    if shared_star_mask is not None:
        candidate_independent_sky_support &= ~shared_star_mask
    masked_audit_layers_native: Dict[str, np.ndarray] = {}
    if masked_catalog_statistics:
        if shared_star_mask is None or shared_valid is None or shared_saturated is None:
            return [], {
                "status": "unavailable",
                "error": "masked catalog statistics require valid, saturation and star evidence",
                "sample_count": 0,
                "mask_evidence": mask_evidence,
            }
        compact_native = _expand_analysis_mask(
            derived_masks["compact_source"],
            height=height,
            width=width,
            stride=mask_stride,
        )
        hard_small = (
            coverage_mask
            | derived_masks["extended_structure"]
            | derived_masks["bright_core"]
            | derived_masks["dark_structure"]
        )
        if bool(protection_policy.get("protect_outer_halo", False)):
            hard_small |= derived_masks["outer_halo"]
        hard_exclusion_native = _expand_analysis_mask(
            hard_small,
            height=height,
            width=width,
            stride=mask_stride,
        )
        masked_audit_layers_native = {
            "shared_invalid": ~shared_valid,
            "shared_saturated": shared_saturated,
            "coverage": _expand_analysis_mask(
                coverage_mask,
                height=height,
                width=width,
                stride=mask_stride,
            ),
            "catalog_star": shared_star_mask,
            "compact_source": compact_native,
            "positive_structure_nebulosity": _expand_analysis_mask(
                derived_masks["extended_structure"],
                height=height,
                width=width,
                stride=mask_stride,
            ),
            "bright_core": _expand_analysis_mask(
                derived_masks["bright_core"],
                height=height,
                width=width,
                stride=mask_stride,
            ),
            "dark_structure": _expand_analysis_mask(
                derived_masks["dark_structure"],
                height=height,
                width=width,
                stride=mask_stride,
            ),
            "outer_halo": _expand_analysis_mask(
                derived_masks["outer_halo"],
                height=height,
                width=width,
                stride=mask_stride,
            ),
        }
        masked_pixel_support = (
            shared_valid
            & ~shared_saturated
            & ~hard_exclusion_native
            & ~shared_star_mask
            & ~compact_native
        )
        candidate_independent_sky_support = masked_pixel_support.copy()
        mask_evidence["masked_catalog_statistics"] = {
            "schema_version": STAGE3_DENSE_STAR_SAMPLING_SCHEMA,
            "enabled": True,
            "minimum_patch_support_fraction": (
                STAGE3_DENSE_STAR_SAMPLE_SUPPORT_MIN
            ),
            "maximum_masked_point_source_fraction": (
                1.0 - STAGE3_DENSE_STAR_SAMPLE_SUPPORT_MIN
            ),
            "support_mask_sha256": _mask_sha256(masked_pixel_support),
            "catalog_mask_sha256": _mask_sha256(shared_star_mask),
            "catalog_records_sha256": shared_star_catalog_sha256,
            "catalog_radius_method": (
                "scene_support_catalog_4x_fwhm"
                if bool(protection_policy.get("protect_star_halo", False))
                else "scene_support_catalog_2_5x_fwhm"
            ),
            "siril_recalculate": False,
        }

    finite_values = luminance[np.isfinite(luminance)]
    low_limit = float(np.quantile(finite_values, 0.015))
    absolute_brightness_limit = float(np.quantile(finite_values, 0.90))
    dynamic = max(
        float(np.quantile(finite_values, 0.98)) - low_limit,
        1e-8,
    )
    aspect = width / max(height, 1)
    candidate_grid_multiplier = (
        10
        if masked_catalog_statistics
        else 6
        if (star_fraction or 0.0) >= STAGE3_DENSE_STAR_FIELD_FRACTION_MIN
        else 4
    )
    candidate_budget = target_count * candidate_grid_multiplier
    columns = max(8, int(round(math.sqrt(candidate_budget * aspect))))
    rows = max(6, int(math.ceil(candidate_budget / columns)))
    x_values = np.linspace(margin, width - margin - 1, columns)
    y_values = np.linspace(margin, height - margin - 1, rows)

    candidates: List[Dict[str, Any]] = []
    candidate_coordinates = set()
    rejected_nonfinite = 0

    def measure_candidate(
        x: int,
        y: int,
        *,
        source: str,
    ) -> Optional[Dict[str, Any]]:
        if (
            x < margin
            or x >= width - margin
            or y < margin
            or y >= height - margin
        ):
            return None
        patch = luminance[
            y - patch_radius : y + patch_radius + 1,
            x - patch_radius : x + patch_radius + 1,
        ]
        if patch.shape != (2 * patch_radius + 1, 2 * patch_radius + 1):
            return None
        finite_ratio = float(np.mean(np.isfinite(patch)))
        if finite_ratio < 0.999:
            return None
        patch_support = None
        channel_medians = None
        native_luminance_mean = None
        native_luminance_min = None
        native_luminance_max = None
        if masked_catalog_statistics:
            assert masked_pixel_support is not None
            assert native_channels is not None
            patch_support = masked_pixel_support[
                y - patch_radius : y + patch_radius + 1,
                x - patch_radius : x + patch_radius + 1,
            ]
            supported = patch[patch_support]
            if supported.size < 16:
                return None
            median = float(np.median(supported))
            mad = float(np.median(np.abs(supported - median)))
            p95 = float(np.quantile(supported, 0.95))
            x_support = patch_support[:, 1:] & patch_support[:, :-1]
            y_support = patch_support[1:, :] & patch_support[:-1, :]
            dx_all = np.abs(np.diff(patch, axis=1))
            dy_all = np.abs(np.diff(patch, axis=0))
            dx = dx_all[x_support]
            dy = dy_all[y_support]
            if dx.size < 8 or dy.size < 8:
                return None
            texture = float(0.5 * (np.median(dx) + np.median(dy)))
            native_patch = native_channels[
                :,
                y - patch_radius : y + patch_radius + 1,
                x - patch_radius : x + patch_radius + 1,
            ]
            channel_medians = [
                float(np.median(channel[patch_support]))
                for channel in native_patch
            ]
            if native_patch.shape[0] == 1:
                native_luminance = native_patch[0]
            else:
                native_luminance = (
                    0.2126 * native_patch[0]
                    + 0.7152 * native_patch[1]
                    + 0.0722 * native_patch[2]
                )
            native_values = native_luminance[patch_support]
            native_luminance_mean = float(np.mean(native_values))
            native_luminance_min = float(np.min(native_values))
            native_luminance_max = float(np.max(native_values))
        else:
            median = float(np.median(patch))
            mad = float(np.median(np.abs(patch - median)))
            p95 = float(np.quantile(patch, 0.95))
            dx = np.abs(np.diff(patch, axis=1))
            dy = np.abs(np.diff(patch, axis=0))
            texture = float(0.5 * (np.median(dx) + np.median(dy)))
        mask_x = min(
            source_mask.shape[1] - 1,
            max(0, int(round(x / mask_stride))),
        )
        mask_y = min(
            source_mask.shape[0] - 1,
            max(0, int(round(y / mask_stride))),
        )
        mask_radius = max(1, int(math.ceil(patch_radius / mask_stride)))
        mx0 = max(0, mask_x - mask_radius)
        mx1 = min(source_mask.shape[1], mask_x + mask_radius + 1)
        my0 = max(0, mask_y - mask_radius)
        my1 = min(source_mask.shape[0], mask_y + mask_radius + 1)
        local_source_mask = source_mask[my0:my1, mx0:mx1]
        local_coverage_mask = coverage_mask[my0:my1, mx0:mx1]
        local_combined_mask = combined_mask[my0:my1, mx0:mx1]
        native_slice = (
            slice(y - patch_radius, y + patch_radius + 1),
            slice(x - patch_radius, x + patch_radius + 1),
        )
        local_compact_native = (
            compact_native[native_slice]
            if compact_native is not None
            else None
        )
        local_hard_native = (
            hard_exclusion_native[native_slice]
            if hard_exclusion_native is not None
            else None
        )
        shared_star_patch = (
            shared_star_mask[native_slice]
            if shared_star_mask is not None
            else None
        )
        point_source_mask = (
            shared_star_patch | local_compact_native
            if shared_star_patch is not None and local_compact_native is not None
            else shared_star_patch
        )
        mask_overlap_fractions = (
            {
                name: float(np.mean(mask[native_slice]))
                for name, mask in masked_audit_layers_native.items()
            }
            if masked_catalog_statistics
            else None
        )
        if mask_overlap_fractions is not None:
            mask_overlap_fractions["outer_halo_enforced"] = bool(
                protection_policy.get("protect_outer_halo", False)
            )
        return {
            "x": int(x),
            "y": int(y),
            "median": median,
            "mad": mad,
            "texture": texture,
            "star_excess": max(0.0, p95 - median),
            "source_mask_center": bool(source_mask[mask_y, mask_x]),
            "source_mask_fraction": float(np.mean(local_source_mask)),
            "coverage_mask_fraction": float(np.mean(local_coverage_mask)),
            "combined_mask_fraction": float(np.mean(local_combined_mask)),
            "hard_exclusion_fraction": (
                float(np.mean(local_hard_native))
                if local_hard_native is not None
                else None
            ),
            "compact_source_fraction": (
                float(np.mean(local_compact_native))
                if local_compact_native is not None
                else None
            ),
            "point_source_mask_fraction": (
                float(np.mean(point_source_mask))
                if point_source_mask is not None
                else None
            ),
            "masked_support_fraction": (
                float(np.mean(patch_support))
                if patch_support is not None
                else None
            ),
            "masked_support_count": (
                int(np.count_nonzero(patch_support))
                if patch_support is not None
                else None
            ),
            "sample_size": int(2 * patch_radius + 1),
            "channel_count": (
                int(native_channels.shape[0])
                if native_channels is not None
                else None
            ),
            "channel_medians": channel_medians,
            "native_luminance_mean": native_luminance_mean,
            "native_luminance_min": native_luminance_min,
            "native_luminance_max": native_luminance_max,
            "mask_overlap_fractions": mask_overlap_fractions,
            "candidate_source": source,
            "shared_valid_fraction": (
                float(np.mean(shared_valid[
                    y - patch_radius : y + patch_radius + 1,
                    x - patch_radius : x + patch_radius + 1,
                ]))
                if shared_valid is not None
                else None
            ),
            "shared_saturated_fraction": (
                float(np.mean(shared_saturated[
                    y - patch_radius : y + patch_radius + 1,
                    x - patch_radius : x + patch_radius + 1,
                ]))
                if shared_saturated is not None
                else None
            ),
            "shared_star_fraction": (
                float(np.mean(shared_star_patch))
                if shared_star_patch is not None
                else None
            ),
            "shared_star_center": (
                bool(shared_star_mask[y, x])
                if shared_star_mask is not None
                else None
            ),
            "compact_source_center": (
                bool(compact_native[y, x])
                if compact_native is not None
                else None
            ),
        }

    for raw_y in y_values:
        y = int(round(float(raw_y)))
        for raw_x in x_values:
            x = int(round(float(raw_x)))
            record = measure_candidate(x, y, source="regular_grid")
            if record is None:
                rejected_nonfinite += 1
                continue
            candidates.append(record)
            candidate_coordinates.add((x, y))

    if not candidates:
        return [], {
            "status": "unavailable",
            "error": "no finite sample candidates",
            "sample_count": 0,
        }
    base_candidate_count = len(candidates)
    coordinates = np.asarray(
        [
            (
                1.0,
                record["x"] / max(width - 1, 1),
                record["y"] / max(height - 1, 1),
                (record["x"] / max(width - 1, 1)) ** 2,
                (record["y"] / max(height - 1, 1)) ** 2,
                (record["x"] / max(width - 1, 1))
                * (record["y"] / max(height - 1, 1)),
            )
            for record in candidates
        ],
        dtype=np.float64,
    )
    medians = np.asarray(
        [record["median"] for record in candidates],
        dtype=np.float64,
    )
    fit_mask = np.ones(len(candidates), dtype=bool)
    coefficients = np.zeros(coordinates.shape[1], dtype=np.float64)
    try:
        for _ in range(3):
            if int(np.count_nonzero(fit_mask)) < max(12, coordinates.shape[1] * 2):
                break
            coefficients, *_ = np.linalg.lstsq(
                coordinates[fit_mask],
                medians[fit_mask],
                rcond=None,
            )
            residuals = medians - coordinates @ coefficients
            center = float(np.median(residuals[fit_mask]))
            sigma = max(_robust_sigma(residuals[fit_mask]), dynamic * 0.003)
            next_mask = (residuals >= center - 2.5 * sigma) & (
                residuals <= center + 1.5 * sigma
            )
            if np.array_equal(next_mask, fit_mask):
                break
            fit_mask = next_mask
    except np.linalg.LinAlgError as error:
        return [], {
            "status": "unavailable",
            "error": f"safe-sample background fit failed: {error}",
            "sample_count": 0,
        }
    residuals = medians - coordinates @ coefficients
    residual_low = float(np.quantile(residuals, 0.10))
    residual_high = float(
        np.quantile(residuals, brightness_quantile_max)
    )
    for record, residual in zip(candidates, residuals):
        record["brightness_residual"] = float(residual)
    texture_limit = max(
        float(
            np.quantile(
                [record["texture"] for record in candidates],
                texture_quantile_max,
            )
        ),
        dynamic * 1e-5,
    )
    star_excess_limit = max(
        float(
            np.quantile(
                [record["star_excess"] for record in candidates],
                0.60,
            )
        ),
        dynamic * 0.003,
    )

    def decorate_candidate(record: Dict[str, Any]) -> None:
        nx = record["x"] / max(width - 1, 1)
        ny = record["y"] / max(height - 1, 1)
        design_row = np.asarray(
            (1.0, nx, ny, nx * nx, ny * ny, nx * ny),
            dtype=np.float64,
        )
        record["brightness_residual"] = float(
            record["median"] - design_row @ coefficients
        )

    def audit_candidate(record: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        if (
            record.get("shared_valid_fraction") is not None
            and float(record["shared_valid_fraction"]) < 0.999
        ):
            return False, "shared_invalid_region"
        if float(record.get("shared_saturated_fraction") or 0.0) > 0.0:
            return False, "shared_saturated"
        if masked_catalog_statistics:
            if bool(record.get("shared_star_center")) or bool(
                record.get("compact_source_center")
            ):
                return False, "masked_point_source_center"
            if float(record.get("hard_exclusion_fraction") or 0.0) > 0.0:
                return False, "exclusion_masked"
            if float(record.get("masked_support_fraction") or 0.0) < (
                STAGE3_DENSE_STAR_SAMPLE_SUPPORT_MIN
            ):
                return False, "masked_point_source_support_insufficient"
        else:
            if float(record.get("shared_star_fraction") or 0.0) > 0.0:
                return False, "shared_catalog_star"
            if record["combined_mask_fraction"] > 0.0:
                return False, "exclusion_masked"
        if record["median"] <= low_limit:
            return False, "clipped_or_too_dark"
        if (
            record["median"] > absolute_brightness_limit
            or record["brightness_residual"] > residual_high
        ):
            return False, "too_bright"
        if record["brightness_residual"] < residual_low:
            return False, "clipped_or_too_dark"
        if record["texture"] > texture_limit:
            return False, "structured"
        if record["star_excess"] > star_excess_limit:
            return False, "star_contaminated"
        brightness_risk = (
            record["brightness_residual"] - residual_low
        ) / max(residual_high - residual_low, 1e-8)
        texture_risk = record["texture"] / max(texture_limit, 1e-8)
        star_risk = record["star_excess"] / max(star_excess_limit, 1e-8)
        record["risk"] = (
            0.45 * brightness_risk
            + 0.35 * texture_risk
            + 0.20 * star_risk
        )
        return True, None

    base_safe_candidate_count = sum(
        1 for record in candidates if audit_candidate(record)[0]
    )
    refinement_block_reasons: List[str] = []
    if not candidate_refinement:
        refinement_block_reasons.append("disabled_by_target_policy")
    if refinement_max_steps <= 0:
        refinement_block_reasons.append("disabled_by_step_limit")
    if (
        float(mask_evidence.get("usable_sky_fraction", 0.0) or 0.0)
        < STAGE3_MIN_USABLE_SKY_FRACTION
    ):
        refinement_block_reasons.append("usable_sky_fraction_below_0_50")
    if (
        float(mask_report.get("source_mask_fraction", 1.0) or 1.0)
        > STAGE3_MAX_SOURCE_MASK_FRACTION
    ):
        refinement_block_reasons.append("source_mask_fraction_above_0_50")
    refinement_enabled = not refinement_block_reasons
    refinement_report: Dict[str, Any] = {
        "enabled": refinement_enabled,
        "method": "deterministic_mask_constrained_dark_patch_walk",
        "seed_count": base_candidate_count,
        "generated_candidate_count": 0,
        "accepted_candidate_count": 0,
        "max_steps": refinement_max_steps,
        "block_reasons": refinement_block_reasons,
        "minimum_usable_sky_fraction": STAGE3_MIN_USABLE_SKY_FRACTION,
        "maximum_source_mask_fraction": STAGE3_MAX_SOURCE_MASK_FRACTION,
    }
    if refinement_enabled:
        initial_step = max(
            patch_radius,
            int(round(min(height, width) * 0.018)),
        )
        minimum_step = max(2, patch_radius // 2)
        refined_candidates: List[Dict[str, Any]] = []
        for seed in list(candidates):
            current_x = int(seed["x"])
            current_y = int(seed["y"])
            seed_ok, _seed_reason = audit_candidate(seed)
            current = seed if seed_ok else None
            step = initial_step
            steps_used = 0
            while step >= minimum_step and steps_used < refinement_max_steps:
                neighbours: List[Dict[str, Any]] = []
                for offset_y in (-step, 0, step):
                    for offset_x in (-step, 0, step):
                        if offset_x == 0 and offset_y == 0:
                            continue
                        x = current_x + offset_x
                        y = current_y + offset_y
                        proposal = measure_candidate(
                            x,
                            y,
                            source="dark_patch_refinement",
                        )
                        if proposal is None:
                            continue
                        decorate_candidate(proposal)
                        proposal_ok, _proposal_reason = audit_candidate(proposal)
                        if proposal_ok:
                            neighbours.append(proposal)
                if neighbours:
                    best = min(
                        neighbours,
                        key=lambda item: (item["risk"], item["y"], item["x"]),
                    )
                    if current is None or best["risk"] + 1e-12 < current["risk"]:
                        current = best
                        current_x = int(best["x"])
                        current_y = int(best["y"])
                        steps_used += 1
                        continue
                step //= 2
            if current is seed or current is None:
                continue
            key = (int(current["x"]), int(current["y"]))
            if key in candidate_coordinates:
                continue
            candidate_coordinates.add(key)
            refined_candidates.append(current)
        candidates.extend(refined_candidates)
        refinement_report["generated_candidate_count"] = len(refined_candidates)
        refinement_report["accepted_candidate_count"] = len(refined_candidates)

    safe: List[Dict[str, Any]] = []
    rejection_counts = {
        "nonfinite": rejected_nonfinite,
        "exclusion_masked": 0,
        "clipped_or_too_dark": 0,
        "too_bright": 0,
        "structured": 0,
        "star_contaminated": 0,
        "source_masked": 0,
        "coverage_masked": 0,
        "shared_invalid_region": 0,
        "shared_saturated": 0,
        "shared_catalog_star": 0,
        "masked_point_source_center": 0,
        "masked_point_source_support_insufficient": 0,
    }
    for record in candidates:
        accepted, rejection_reason = audit_candidate(record)
        if not accepted:
            rejection_counts[str(rejection_reason)] += 1
            continue
        safe.append(record)

    coverage_rows = 4
    coverage_columns = 4
    candidates_by_cell: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for record in safe:
        cell_x = min(
            coverage_columns - 1,
            int(record["x"] * coverage_columns / max(width, 1)),
        )
        cell_y = min(
            coverage_rows - 1,
            int(record["y"] * coverage_rows / max(height, 1)),
        )
        key = (cell_x, cell_y)
        candidates_by_cell.setdefault(key, []).append(record)

    minimum_spacing = max(
        patch_radius * 2,
        int(round(min(height, width) * 0.035)),
    )
    for records in candidates_by_cell.values():
        records.sort(key=lambda item: (item["risk"], item["y"], item["x"]))
    # Cover sparse cells first.  Every cell representative still has to obey
    # the global spacing contract; adjacent cell boundaries must not create
    # near-duplicate evidence.
    cell_order = sorted(
        candidates_by_cell,
        key=lambda key: (
            len(candidates_by_cell[key]),
            candidates_by_cell[key][0]["risk"],
            key[1],
            key[0],
        ),
    )
    selected: List[Dict[str, Any]] = []
    selected_keys = set()
    selected_grid_cells = set()

    def select_first_spaced(records: Sequence[Dict[str, Any]]) -> bool:
        for record in records:
            if len(selected) >= target_count:
                return False
            key = (record["x"], record["y"])
            if key in selected_keys:
                continue
            if any(
                math.hypot(
                    record["x"] - item["x"],
                    record["y"] - item["y"],
                )
                < minimum_spacing
                for item in selected
            ):
                continue
            selected.append(record)
            selected_keys.add(key)
            selected_grid_cells.add(
                (
                    min(
                        coverage_columns - 1,
                        int(record["x"] * coverage_columns / max(width, 1)),
                    ),
                    min(
                        coverage_rows - 1,
                        int(record["y"] * coverage_rows / max(height, 1)),
                    ),
                )
            )
            return True
        return False

    # Conservative recovery must retain the regular-grid evidence that made
    # the recovery eligible. Refinement may fill missing cells, but must not
    # displace all regular points merely because local risk is lower. This is
    # opt-in so ordinary Stage 3 and Stage 4 selection remain unchanged.
    if preserve_regular_grid:
        regular_by_cell: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        for cell, records in candidates_by_cell.items():
            regular_records = [
                record
                for record in records
                if record.get("candidate_source") == "regular_grid"
            ]
            if regular_records:
                regular_by_cell[cell] = regular_records
        regular_cell_order = sorted(
            regular_by_cell,
            key=lambda key: (
                len(regular_by_cell[key]),
                regular_by_cell[key][0]["risk"],
                key[1],
                key[0],
            ),
        )
        for cell in regular_cell_order:
            select_first_spaced(regular_by_cell[cell])

    for cell in cell_order:
        select_first_spaced(candidates_by_cell[cell])

    for record in sorted(safe, key=lambda item: item["risk"]):
        if len(selected) >= target_count:
            break
        key = (record["x"], record["y"])
        if key in selected_keys:
            continue
        select_first_spaced((record,))

    selected = sorted(selected[:target_count], key=lambda item: (item["y"], item["x"]))
    quadrants = {
        (int(item["x"] >= width / 2), int(item["y"] >= height / 2))
        for item in selected
    }
    x_span = (
        (max(item["x"] for item in selected) - min(item["x"] for item in selected))
        / max(width - 1, 1)
        if selected
        else 0.0
    )
    y_span = (
        (max(item["y"] for item in selected) - min(item["y"] for item in selected))
        / max(height - 1, 1)
        if selected
        else 0.0
    )
    ready = bool(
        len(selected) >= min_count
        and float(mask_evidence["usable_sky_fraction"])
        >= STAGE3_MIN_USABLE_SKY_FRACTION
        and len(quadrants) >= STAGE3_MIN_SPATIAL_QUADRANTS
        and len(selected_grid_cells) >= STAGE3_MIN_SPATIAL_GRID_CELLS
        and x_span >= STAGE3_MIN_AXIS_SPAN_RATIO
        and y_span >= STAGE3_MIN_AXIS_SPAN_RATIO
    )
    points = [
        (float(item["x"]), float(height - 1 - item["y"]))
        for item in selected
    ]
    selected_source_counts: Dict[str, int] = {}
    selected_samples: List[Dict[str, Any]] = []
    for item in selected:
        source = str(item.get("candidate_source") or "unknown")
        selected_source_counts[source] = selected_source_counts.get(source, 0) + 1
        sample_x = float(item["x"])
        sample_y = float(height - 1 - item["y"])
        selected_samples.append(
            {
                "point": [sample_x, sample_y],
                "source": source,
                "grid_cell": [
                    min(
                        coverage_columns - 1,
                        int(item["x"] * coverage_columns / max(width, 1)),
                    ),
                    min(
                        coverage_rows - 1,
                        int(item["y"] * coverage_rows / max(height, 1)),
                    ),
                ],
                "sample_size": item.get("sample_size"),
                "masked_support_count": item.get("masked_support_count"),
                "masked_support_fraction": item.get("masked_support_fraction"),
                "shared_star_fraction": item.get("shared_star_fraction"),
                "compact_source_fraction": item.get("compact_source_fraction"),
                "point_source_mask_fraction": item.get(
                    "point_source_mask_fraction"
                ),
                "hard_exclusion_fraction": item.get("hard_exclusion_fraction"),
                "channel_count": item.get("channel_count"),
                "channel_medians": item.get("channel_medians"),
                "native_luminance_mean": item.get("native_luminance_mean"),
                "native_luminance_min": item.get("native_luminance_min"),
                "native_luminance_max": item.get("native_luminance_max"),
                "mask_overlap_fractions": item.get(
                    "mask_overlap_fractions"
                ),
            }
        )
    report = {
        "status": "ready" if ready else "insufficient_safe_coverage",
        "coordinate_system": "siril_bottom_left",
        "sample_count": len(points) if ready else 0,
        "selected_candidate_count": len(selected),
        "target_count": target_count,
        "minimum_count": min_count,
        "candidate_count": len(candidates),
        "base_candidate_count": base_candidate_count,
        "safe_candidate_count": len(safe),
        "base_safe_candidate_count": base_safe_candidate_count,
        "candidate_search": {
            "method": "regular_grid_plus_deterministic_dark_patch_refinement",
            "grid_multiplier": candidate_grid_multiplier,
            "dense_star_field_expansion": candidate_grid_multiplier > 4,
            "scene_support_star_fraction": star_fraction,
        },
        "refinement": refinement_report,
        "preserve_regular_grid": preserve_regular_grid,
        "masked_catalog_statistics": masked_catalog_statistics,
        "patch_radius": patch_radius,
        "margin_pixels": margin,
        "minimum_spacing_pixels": minimum_spacing,
        "selected_candidate_sources": selected_source_counts,
        "selected_samples": selected_samples,
        "coverage": {
            "quadrants": len(quadrants),
            "grid_cells": len(selected_grid_cells),
            "available_grid_cells": len(candidates_by_cell),
            "grid_cell_ratio": len(selected_grid_cells)
            / (coverage_rows * coverage_columns),
            "x_span_ratio": x_span,
            "y_span_ratio": y_span,
        },
        "masks": mask_report,
        "mask_evidence": mask_evidence,
        "shared_scene_support": shared_support_status,
        "thresholds": {
            "brightness_quantile_max": brightness_quantile_max,
            "absolute_brightness_limit": absolute_brightness_limit,
            "brightness_residual_low": residual_low,
            "brightness_residual_high": residual_high,
            "low_limit": low_limit,
            "texture_quantile_max": texture_quantile_max,
            "texture_limit": texture_limit,
            "star_excess_limit": star_excess_limit,
        },
        "rejection_counts": rejection_counts,
        "points": [[x, y] for x, y in points] if ready else [],
        "safety_contract": (
            [
                "require zero overlap with invalid, saturated and non-point-source structure masks",
                "exclude catalog and compact point-source pixels from patch statistics",
                "require at least 80 percent frozen pixel support per patch",
                "reject patch centers inside catalog or compact point-source masks",
                "dark-patch proposals reuse thresholds frozen from the regular grid",
                "refined proposals remain deterministic, deduplicated and spacing-audited",
                "require broad spatial coverage before subsky -existing",
            ]
            if masked_catalog_statistics
            else [
                "require zero overlap with the combined multiscale exclusion mask",
                "reject clipped/bright/structured/star-contaminated patches",
                "dark-patch proposals reuse thresholds frozen from the regular grid",
                "refined proposals remain deterministic, deduplicated and spacing-audited",
                "require broad spatial coverage before subsky -existing",
            ]
        ),
    }
    if masked_catalog_statistics:
        report["dense_star_masked_sampling"] = {
            "schema_version": STAGE3_DENSE_STAR_SAMPLING_SCHEMA,
            "status": "ready" if ready else "insufficient_safe_coverage",
            "minimum_patch_support_fraction": (
                STAGE3_DENSE_STAR_SAMPLE_SUPPORT_MIN
            ),
            "sample_statistics": "masked_native_channel_median",
            "siril_recalculate": False,
            "selected_support_fraction_min": (
                min(
                    float(item["masked_support_fraction"])
                    for item in selected
                )
                if selected
                else None
            ),
            "selected_support_fraction_median": (
                float(
                    np.median(
                        [
                            float(item["masked_support_fraction"])
                            for item in selected
                        ]
                    )
                )
                if selected
                else None
            ),
            "support_mask_sha256": (
                (mask_evidence.get("masked_catalog_statistics") or {}).get(
                    "support_mask_sha256"
                )
            ),
            "catalog_mask_sha256": (
                (mask_evidence.get("masked_catalog_statistics") or {}).get(
                    "catalog_mask_sha256"
                )
            ),
            "catalog_records_sha256": shared_star_catalog_sha256,
        }
        report["_masked_pixel_support_mask"] = masked_pixel_support
    report["candidate_independent_sky_support"] = {
        "schema_version": "starun.stage3-candidate-independent-sky-support.v1",
        "status": "available",
        "pixel_count": int(np.count_nonzero(candidate_independent_sky_support)),
        "coverage": float(np.mean(candidate_independent_sky_support)),
        "sha256": _mask_sha256(candidate_independent_sky_support),
        "definition": (
            "masked_catalog_nonstellar_full_resolution_sky"
            if masked_catalog_statistics
            else "full_resolution_multiscale_exclusion_safe_sky"
        ),
    }
    if return_candidate_independent_support_mask:
        report["_candidate_independent_sky_support_mask"] = (
            candidate_independent_sky_support
        )
    return (points if ready else []), report


def _native_luminance(
    image: Any,
    *,
    value_scale: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """Return luminance on a stable native scale for before/after comparison."""
    source = np.asarray(image)
    if source.size == 0:
        raise ValueError("image buffer is empty")
    source_dtype = source.dtype
    if source.ndim == 2:
        luminance = source
    elif source.ndim == 3:
        if source.shape[0] in (1, 3, 4) and source.shape[-1] not in (1, 3, 4):
            channels = source[:3]
        elif source.shape[-1] in (1, 3, 4):
            channels = np.moveaxis(source[..., :3], -1, 0)
        elif source.shape[0] >= 3:
            channels = source[:3]
        else:
            raise ValueError(f"unsupported image shape: {source.shape}")
        if channels.shape[0] == 1:
            luminance = channels[0]
        else:
            luminance = (
                0.2126 * channels[0]
                + 0.7152 * channels[1]
                + 0.0722 * channels[2]
            )
    else:
        raise ValueError(f"unsupported image ndim: {source.ndim}")

    values = np.asarray(luminance, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size < 64:
        raise ValueError("not enough finite pixels")
    if value_scale is None:
        if np.issubdtype(source_dtype, np.integer):
            value_scale = float(np.iinfo(source_dtype).max)
        else:
            robust_high = float(np.quantile(np.abs(finite), 0.999))
            if robust_high <= 1.5:
                value_scale = 1.0
            elif robust_high <= 255.0 * 1.5:
                value_scale = 255.0
            elif robust_high <= 65535.0 * 1.5:
                value_scale = 65535.0
            else:
                value_scale = max(robust_high, 1.0)
    value_scale = float(value_scale)
    if not math.isfinite(value_scale) or value_scale <= 0.0:
        raise ValueError("validation value scale is invalid")
    return values / value_scale, value_scale


def _stage3_rgb_chw(image: Any) -> Tuple[np.ndarray, str, np.dtype]:
    """Return a finite three-channel image and its reversible layout."""
    source = np.asarray(image)
    if source.ndim != 3:
        raise ValueError("neutral-axis projection requires a three-channel image")
    if source.shape[0] == 3 and source.shape[-1] != 3:
        rgb = source
        layout = "chw"
    elif source.shape[-1] == 3:
        rgb = np.moveaxis(source, -1, 0)
        layout = "hwc"
    else:
        raise ValueError(f"unsupported RGB image shape: {source.shape}")
    values = np.asarray(rgb, dtype=np.float64)
    if values.shape[0] != 3 or values.shape[1] < 2 or values.shape[2] < 2:
        raise ValueError(f"invalid RGB image shape: {source.shape}")
    if not bool(np.all(np.isfinite(values))):
        raise ValueError("RGB image contains non-finite pixels")
    return values, layout, source.dtype


def _restore_stage3_rgb_layout(
    rgb: np.ndarray,
    *,
    layout: str,
    dtype: np.dtype,
) -> np.ndarray:
    restored = rgb if layout == "chw" else np.moveaxis(rgb, 0, -1)
    output_dtype = dtype if np.issubdtype(dtype, np.floating) else np.dtype(np.float32)
    return np.asarray(restored, dtype=output_dtype)


def _stage3_patch_mask(
    points: Sequence[Tuple[float, float]],
    *,
    width: int,
    height: int,
    patch_radius: int,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for raw_x, raw_y in points:
        x = int(round(float(raw_x)))
        y = int(round(height - 1 - float(raw_y)))
        x0 = max(0, x - patch_radius)
        x1 = min(width, x + patch_radius + 1)
        y0 = max(0, y - patch_radius)
        y1 = min(height, y + patch_radius + 1)
        if x0 < x1 and y0 < y1:
            mask[y0:y1, x0:x1] = True
    return mask


def project_stage3_neutral_axis_poly1(
    baseline_image: Any,
    siril_proposal_image: Any,
    fit_points: Sequence[Tuple[float, float]],
    validation_points: Sequence[Tuple[float, float]],
    *,
    patch_radius: int = 12,
    minimum_fit: int = 8,
    condition_number_max: float = STAGE3_NEUTRAL_AXIS_CONDITION_MAX,
    headroom_fraction_max: float = (
        STAGE3_NEUTRAL_AXIS_HEADROOM_FRACTION_MAX
    ),
    headroom_margin: float = STAGE3_NEUTRAL_AXIS_HEADROOM_MARGIN,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Project a Siril RGB Polynomial proposal onto the additive neutral axis.

    Siril owns the degree-1 proposal. Only its Rec.709 slope is retained; the
    per-channel correction and DC term are discarded. Validation points never
    participate in the fit or anchor and are inspected only after construction.
    """
    report: Dict[str, Any] = {
        "schema": STAGE3_NEUTRAL_AXIS_PROJECTION_SCHEMA,
        "status": "rejected",
        "accepted": False,
        "reason_code": "stage3_neutral_axis_projection_unavailable",
        "source_model": "siril_subsky_degree1_existing",
        "luminance": "rec709",
        "projection": "additive_rgb_neutral_axis",
        "anchor": {
            "type": "geometric_center_zero_mean",
            "x": 0.5,
            "y": 0.5,
            "constant_term_applied": False,
        },
        "fit_count": len(fit_points),
        "validation_count": len(validation_points),
        "patch_radius": int(patch_radius),
        "issues": [],
    }

    def reject(reason_code: str, issue: str) -> Tuple[None, Dict[str, Any]]:
        report["reason_code"] = reason_code
        report["issues"] = list(dict.fromkeys([*report["issues"], issue]))
        return None, report

    patch_radius = max(1, int(patch_radius))
    minimum_fit = max(3, int(minimum_fit))
    if len(fit_points) < minimum_fit:
        return reject(
            "stage3_neutral_axis_fit_samples_insufficient",
            "neutral-axis projection has insufficient fit samples",
        )
    try:
        baseline, layout, source_dtype = _stage3_rgb_chw(baseline_image)
        proposal, proposal_layout, _proposal_dtype = _stage3_rgb_chw(
            siril_proposal_image
        )
    except (TypeError, ValueError) as error:
        return reject(
            "stage3_neutral_axis_rgb_unavailable",
            str(error),
        )
    if proposal_layout != layout or proposal.shape != baseline.shape:
        return reject(
            "stage3_neutral_axis_proposal_shape_mismatch",
            "Siril proposal shape or layout differs from Stage 3 baseline",
        )
    if bool(np.array_equal(proposal, baseline)):
        return reject(
            "stage3_neutral_axis_proposal_unchanged",
            "Siril Polynomial proposal did not change any pixels",
        )

    height, width = baseline.shape[1:]
    rec709 = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float64)
    delta_luminance = np.tensordot(
        rec709,
        proposal - baseline,
        axes=(0, 0),
    )
    report["raw_proposal"] = {
        "shape": list(proposal.shape),
        "dtype": str(np.asarray(siril_proposal_image).dtype),
        "finite": bool(np.all(np.isfinite(proposal))),
        "changed_pixel_fraction": float(
            np.mean(np.any(proposal != baseline, axis=0))
        ),
        "rec709_delta_min": float(np.min(delta_luminance)),
        "rec709_delta_max": float(np.max(delta_luminance)),
        "rec709_delta_median": float(np.median(delta_luminance)),
    }
    design_rows: List[List[float]] = []
    patch_values: List[float] = []
    for raw_x, raw_y in fit_points:
        x = int(round(float(raw_x)))
        y = int(round(height - 1 - float(raw_y)))
        x0 = max(0, x - patch_radius)
        x1 = min(width, x + patch_radius + 1)
        y0 = max(0, y - patch_radius)
        y1 = min(height, y + patch_radius + 1)
        patch = delta_luminance[y0:y1, x0:x1]
        if patch.size == 0 or not bool(np.all(np.isfinite(patch))):
            return reject(
                "stage3_neutral_axis_fit_patch_invalid",
                "neutral-axis fit patch is empty or non-finite",
            )
        design_rows.append(
            [
                1.0,
                float(raw_x) / max(width - 1, 1),
                float(y) / max(height - 1, 1),
            ]
        )
        patch_values.append(float(np.median(patch)))
    design = np.asarray(design_rows, dtype=np.float64)
    values = np.asarray(patch_values, dtype=np.float64)
    try:
        coefficients, _residuals, rank, singular_values = np.linalg.lstsq(
            design,
            values,
            rcond=None,
        )
        condition_number = float(np.linalg.cond(design))
    except np.linalg.LinAlgError as error:
        return reject(
            "stage3_neutral_axis_fit_failed",
            f"neutral-axis degree-1 fit failed: {error}",
        )
    fitted_values = design @ coefficients
    residual_rms = float(np.sqrt(np.mean((values - fitted_values) ** 2)))
    report["model"] = {
        "coordinate_system": "normalized_top_left",
        "coefficients": [float(value) for value in coefficients],
        "rank": int(rank),
        "condition_number": condition_number,
        "condition_number_max": float(condition_number_max),
        "singular_values": [float(value) for value in singular_values],
        "fit_patch_residual_rms": residual_rms,
    }
    if int(rank) != 3:
        return reject(
            "stage3_neutral_axis_fit_rank_failed",
            f"neutral-axis fit rank is {int(rank)}, expected 3",
        )
    if not math.isfinite(condition_number) or condition_number > float(
        condition_number_max
    ):
        return reject(
            "stage3_neutral_axis_fit_condition_failed",
            "neutral-axis fit condition number exceeds the fixed limit",
        )

    yy, xx = np.mgrid[:height, :width]
    correction = (
        float(coefficients[1]) * (xx / max(width - 1, 1) - 0.5)
        + float(coefficients[2]) * (yy / max(height - 1, 1) - 0.5)
    )
    correction_abs_max = float(np.max(np.abs(correction)))
    if not math.isfinite(correction_abs_max) or correction_abs_max <= 1e-12:
        return reject(
            "stage3_neutral_axis_correction_not_material",
            "neutral-axis correction is non-finite or numerically unchanged",
        )

    channel_min = np.min(baseline, axis=0)
    channel_max = np.max(baseline, axis=0)
    alpha = np.ones_like(correction, dtype=np.float64)
    margin_scale = max(0.0, min(1.0, 1.0 - float(headroom_margin)))
    negative = correction < 0.0
    positive = correction > 0.0
    alpha[negative] = np.minimum(
        1.0,
        margin_scale
        * np.maximum(channel_min[negative], 0.0)
        / np.maximum(-correction[negative], 1e-15),
    )
    alpha[positive] = np.minimum(
        1.0,
        margin_scale
        * np.maximum(1.0 - channel_max[positive], 0.0)
        / np.maximum(correction[positive], 1e-15),
    )
    baseline_clipped = (channel_min <= 0.0) | (channel_max >= 1.0)
    alpha[baseline_clipped] = 0.0
    alpha = np.clip(alpha, 0.0, 1.0)
    meaningful = np.abs(correction) > 1e-12
    attenuated = meaningful & (alpha < 1.0 - 1e-12)
    attenuated_fraction = float(np.mean(attenuated))
    fit_mask = _stage3_patch_mask(
        fit_points,
        width=width,
        height=height,
        patch_radius=patch_radius,
    )
    validation_mask = _stage3_patch_mask(
        validation_points,
        width=width,
        height=height,
        patch_radius=patch_radius,
    )
    fit_attenuated = int(np.count_nonzero(attenuated & fit_mask))
    validation_attenuated = int(
        np.count_nonzero(attenuated & validation_mask)
    )
    report["headroom"] = {
        "method": "continuous_common_axis_taper",
        "margin": float(headroom_margin),
        "attenuated_fraction": attenuated_fraction,
        "attenuated_fraction_max": float(headroom_fraction_max),
        "attenuated_pixel_count": int(np.count_nonzero(attenuated)),
        "fit_patch_attenuated_pixel_count": fit_attenuated,
        "validation_patch_attenuated_pixel_count": validation_attenuated,
        "baseline_clipped_pixel_count": int(np.count_nonzero(baseline_clipped)),
    }
    if attenuated_fraction > float(headroom_fraction_max):
        return reject(
            "stage3_neutral_axis_headroom_fraction_exceeded",
            "neutral-axis headroom attenuation exceeds 0.1 percent",
        )
    if fit_attenuated or validation_attenuated:
        return reject(
            "stage3_neutral_axis_sample_headroom_attenuated",
            "neutral-axis headroom attenuation intersects fit or validation patches",
        )

    applied_correction = correction * alpha
    candidate_rgb = baseline + applied_correction[None, :, :]
    candidate = _restore_stage3_rgb_layout(
        candidate_rgb,
        layout=layout,
        dtype=source_dtype,
    )
    persisted_rgb, _layout, persisted_dtype = _stage3_rgb_chw(candidate)
    delta = persisted_rgb - baseline
    dtype_epsilon = (
        float(np.finfo(persisted_dtype).eps)
        if np.issubdtype(persisted_dtype, np.floating)
        else float(np.finfo(np.float32).eps)
    )
    value_scale = max(1.0, float(np.max(np.abs(baseline))))
    opponent_tolerance = max(1e-7, 8.0 * dtype_epsilon * value_scale)
    rg_drift = float(
        np.max(
            np.abs(
                (persisted_rgb[0] - persisted_rgb[1])
                - (baseline[0] - baseline[1])
            )
        )
    )
    bg_drift = float(
        np.max(
            np.abs(
                (persisted_rgb[2] - persisted_rgb[1])
                - (baseline[2] - baseline[1])
            )
        )
    )
    baseline_low = baseline <= 0.0
    baseline_high = baseline >= 1.0
    new_low_clip = int(np.count_nonzero((persisted_rgb <= 0.0) & ~baseline_low))
    new_high_clip = int(np.count_nonzero((persisted_rgb >= 1.0) & ~baseline_high))
    channel_mean_delta = np.mean(delta, axis=(1, 2))
    report["correction"] = {
        "requested_min": float(np.min(correction)),
        "requested_max": float(np.max(correction)),
        "applied_min": float(np.min(applied_correction)),
        "applied_max": float(np.max(applied_correction)),
        "geometric_center_value": 0.0,
        "channel_mean_delta": [float(value) for value in channel_mean_delta],
    }
    report["invariants"] = {
        "opponent_tolerance": opponent_tolerance,
        "rg_max_abs_drift": rg_drift,
        "bg_max_abs_drift": bg_drift,
        "new_low_clip_count": new_low_clip,
        "new_high_clip_count": new_high_clip,
        "finite": bool(np.all(np.isfinite(persisted_rgb))),
        "shape_preserved": persisted_rgb.shape == baseline.shape,
        "pixels_changed": not bool(np.array_equal(persisted_rgb, baseline)),
    }
    invariant_issues: List[str] = []
    if not report["invariants"]["finite"]:
        invariant_issues.append("neutral-axis candidate contains non-finite pixels")
    if not report["invariants"]["shape_preserved"]:
        invariant_issues.append("neutral-axis candidate shape changed")
    if not report["invariants"]["pixels_changed"]:
        invariant_issues.append("neutral-axis candidate did not change pixels")
    if rg_drift > opponent_tolerance or bg_drift > opponent_tolerance:
        invariant_issues.append("neutral-axis opponent-channel drift exceeds tolerance")
    if new_low_clip or new_high_clip:
        invariant_issues.append("neutral-axis candidate introduced clipping")
    if invariant_issues:
        report["issues"] = invariant_issues
        report["reason_code"] = "stage3_neutral_axis_invariant_failed"
        return None, report
    report.update(
        status="ready",
        accepted=True,
        reason_code="stage3_neutral_axis_projection_ready",
        issues=[],
    )
    return candidate, report


def verify_stage3_neutral_axis_persistence(
    baseline_image: Any,
    candidate_image: Any,
) -> Tuple[bool, Dict[str, Any]]:
    """Verify the neutral-axis invariants after Siril/FITS persistence."""
    report: Dict[str, Any] = {
        "status": "rejected",
        "accepted": False,
        "issues": [],
    }
    try:
        baseline, baseline_layout, _baseline_dtype = _stage3_rgb_chw(
            baseline_image
        )
        candidate, candidate_layout, candidate_dtype = _stage3_rgb_chw(
            candidate_image
        )
    except (TypeError, ValueError) as error:
        report["issues"] = [str(error)]
        return False, report
    if baseline_layout != candidate_layout or baseline.shape != candidate.shape:
        report["issues"] = [
            "persisted neutral-axis candidate shape or layout changed"
        ]
        return False, report

    dtype_epsilon = (
        float(np.finfo(candidate_dtype).eps)
        if np.issubdtype(candidate_dtype, np.floating)
        else float(np.finfo(np.float32).eps)
    )
    value_scale = max(1.0, float(np.max(np.abs(baseline))))
    tolerance = max(1e-7, 8.0 * dtype_epsilon * value_scale)
    rg_drift = float(
        np.max(
            np.abs(
                (candidate[0] - candidate[1])
                - (baseline[0] - baseline[1])
            )
        )
    )
    bg_drift = float(
        np.max(
            np.abs(
                (candidate[2] - candidate[1])
                - (baseline[2] - baseline[1])
            )
        )
    )
    baseline_low = baseline <= 0.0
    baseline_high = baseline >= 1.0
    new_low_clip = int(np.count_nonzero((candidate <= 0.0) & ~baseline_low))
    new_high_clip = int(np.count_nonzero((candidate >= 1.0) & ~baseline_high))
    finite = bool(np.all(np.isfinite(candidate)))
    changed = not bool(np.array_equal(candidate, baseline))
    issues: List[str] = []
    if not finite:
        issues.append("persisted neutral-axis candidate contains non-finite pixels")
    if not changed:
        issues.append("persisted neutral-axis candidate is unchanged")
    if rg_drift > tolerance or bg_drift > tolerance:
        issues.append("persisted opponent-channel drift exceeds tolerance")
    if new_low_clip or new_high_clip:
        issues.append("persisted neutral-axis candidate introduced clipping")
    report.update(
        opponent_tolerance=tolerance,
        rg_max_abs_drift=rg_drift,
        bg_max_abs_drift=bg_drift,
        new_low_clip_count=new_low_clip,
        new_high_clip_count=new_high_clip,
        finite=finite,
        shape_preserved=True,
        pixels_changed=changed,
        issues=issues,
    )
    if issues:
        return False, report
    report.update(status="accepted", accepted=True)
    return True, report


def _stage3_patch_medians(
    plane: np.ndarray,
    points: Sequence[Tuple[float, float]],
    *,
    patch_radius: int,
    support_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return normalized top-left coordinates and frozen patch medians."""
    height, width = plane.shape
    if support_mask is not None:
        support = np.asarray(support_mask, dtype=bool)
        if support.shape != plane.shape:
            raise ValueError("Stage 3 spatial support mask shape mismatch")
    else:
        support = None
    rows: List[List[float]] = []
    medians: List[float] = []
    for raw_x, raw_y in points:
        x = int(round(float(raw_x)))
        y = int(round(height - 1 - float(raw_y)))
        x0 = max(0, x - patch_radius)
        x1 = min(width, x + patch_radius + 1)
        y0 = max(0, y - patch_radius)
        y1 = min(height, y + patch_radius + 1)
        patch = np.asarray(plane[y0:y1, x0:x1], dtype=np.float64)
        if support is not None:
            patch = patch[support[y0:y1, x0:x1]]
        else:
            patch = patch.reshape(-1)
        finite = patch[np.isfinite(patch)]
        if finite.size == 0:
            raise ValueError("Stage 3 spatial opponent patch has no finite support")
        rows.append(
            [
                1.0,
                float(raw_x) / max(width - 1, 1),
                float(y) / max(height - 1, 1),
            ]
        )
        medians.append(float(np.median(finite)))
    return np.asarray(rows, dtype=np.float64), np.asarray(medians, dtype=np.float64)


def _stage3_plane_fit(
    design: np.ndarray,
    values: np.ndarray,
) -> Dict[str, Any]:
    coefficients, _residuals, rank, singular_values = np.linalg.lstsq(
        design,
        values,
        rcond=None,
    )
    fitted = design @ coefficients
    residual = values - fitted
    residual_rms = float(np.sqrt(np.mean(np.square(residual))))
    condition_number = float(np.linalg.cond(design))
    dof = max(int(design.shape[0]) - 3, 1)
    sigma2 = float(np.sum(np.square(residual)) / dof)
    covariance = sigma2 * np.linalg.pinv(design.T @ design)
    slope = np.asarray(coefficients[1:3], dtype=np.float64)
    slope_variance = max(
        float(np.trace(covariance[1:3, 1:3])),
        1e-30,
    )
    significance = float(np.linalg.norm(slope) / math.sqrt(slope_variance))
    return {
        "coefficients": coefficients,
        "rank": int(rank),
        "condition_number": condition_number,
        "singular_values": singular_values,
        "residual_rms": residual_rms,
        "slope_significance_sigma": significance,
    }


def _stage3_centered_plane_values(
    design: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    return (
        float(coefficients[1]) * (design[:, 1] - 0.5)
        + float(coefficients[2]) * (design[:, 2] - 0.5)
    )


def project_stage3_spatial_opponent_poly1(
    baseline_image: Any,
    neutral_image: Any,
    siril_proposal_image: Any,
    fit_points: Sequence[Tuple[float, float]],
    validation_points: Sequence[Tuple[float, float]],
    *,
    patch_radius: int = 12,
    minimum_fit: int = 8,
    minimum_validation: int = 4,
    condition_number_max: float = STAGE3_NEUTRAL_AXIS_CONDITION_MAX,
    significance_sigma: float = STAGE3_SIGNIFICANCE_SIGMA,
    correlation_min: float = STAGE3_SPATIAL_OPPONENT_CORRELATION_MIN,
    rms_improvement_min: float = STAGE3_SPATIAL_OPPONENT_RMS_IMPROVEMENT_MIN,
    headroom_fraction_max: float = STAGE3_NEUTRAL_AXIS_HEADROOM_FRACTION_MAX,
    headroom_margin: float = STAGE3_NEUTRAL_AXIS_HEADROOM_MARGIN,
    support_mask: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Project independently proven Siril opponent slopes at zero Rec.709 luma."""
    report: Dict[str, Any] = {
        "schema": STAGE3_SPATIAL_OPPONENT_PROJECTION_SCHEMA,
        "status": "rejected",
        "accepted": False,
        "reason_code": "stage3_spatial_opponent_lineage_unverified",
        "anchor": {
            "type": "geometric_center_zero_mean",
            "x": 0.5,
            "y": 0.5,
            "constant_term_applied": False,
        },
        "fit_count": len(fit_points),
        "validation_count": len(validation_points),
        "patch_radius": int(patch_radius),
        "components": {},
        "selected_components": [],
        "unresolved_components": [],
        "issues": [],
    }

    def reject(reason: str, issue: str) -> Tuple[None, Dict[str, Any]]:
        report["reason_code"] = reason
        report["issues"] = list(dict.fromkeys([*report["issues"], issue]))
        return None, report

    if len(fit_points) < max(3, int(minimum_fit)):
        return reject(
            "stage3_spatial_opponent_lineage_unverified",
            "spatial opponent fit samples are insufficient",
        )
    if len(validation_points) < max(3, int(minimum_validation)):
        return reject(
            "stage3_spatial_opponent_lineage_unverified",
            "spatial opponent validation samples are insufficient",
        )
    try:
        baseline, layout, _baseline_dtype = _stage3_rgb_chw(baseline_image)
        neutral, neutral_layout, neutral_dtype = _stage3_rgb_chw(neutral_image)
        proposal, proposal_layout, _proposal_dtype = _stage3_rgb_chw(
            siril_proposal_image
        )
    except (TypeError, ValueError) as error:
        return reject("stage3_spatial_opponent_lineage_unverified", str(error))
    if (
        neutral_layout != layout
        or proposal_layout != layout
        or neutral.shape != baseline.shape
        or proposal.shape != baseline.shape
    ):
        return reject(
            "stage3_spatial_opponent_lineage_unverified",
            "spatial opponent source shapes or layouts differ",
        )

    patch_radius = max(1, int(patch_radius))
    opponent_planes = {
        "R-G": (baseline[0] - baseline[1], (proposal[0] - proposal[1]) - (baseline[0] - baseline[1])),
        "B-G": (baseline[2] - baseline[1], (proposal[2] - proposal[1]) - (baseline[2] - baseline[1])),
    }
    selected: Dict[str, np.ndarray] = {}
    unresolved: List[str] = []
    fit_design: Optional[np.ndarray] = None
    validation_design: Optional[np.ndarray] = None
    for name, (baseline_plane, proposal_delta) in opponent_planes.items():
        try:
            component_fit_design, baseline_fit = _stage3_patch_medians(
                baseline_plane,
                fit_points,
                patch_radius=patch_radius,
                support_mask=support_mask,
            )
            validation_design, baseline_validation = _stage3_patch_medians(
                baseline_plane,
                validation_points,
                patch_radius=patch_radius,
                support_mask=support_mask,
            )
            delta_fit_design, proposal_fit = _stage3_patch_medians(
                proposal_delta,
                fit_points,
                patch_radius=patch_radius,
                support_mask=support_mask,
            )
            if not np.array_equal(component_fit_design, delta_fit_design):
                raise ValueError("spatial opponent fit coordinates changed")
            baseline_model = _stage3_plane_fit(component_fit_design, baseline_fit)
            proposal_model = _stage3_plane_fit(component_fit_design, proposal_fit)
            validation_model = _stage3_plane_fit(validation_design, baseline_validation)
        except (ValueError, np.linalg.LinAlgError) as error:
            report["components"][name] = {
                "status": "unverified",
                "selected": False,
                "reason": str(error),
            }
            unresolved.append(name)
            continue
        fit_design = component_fit_design
        model_valid = bool(
            baseline_model["rank"] == 3
            and proposal_model["rank"] == 3
            and validation_model["rank"] == 3
            and math.isfinite(baseline_model["condition_number"])
            and math.isfinite(proposal_model["condition_number"])
            and baseline_model["condition_number"] <= float(condition_number_max)
            and proposal_model["condition_number"] <= float(condition_number_max)
            and math.isfinite(validation_model["condition_number"])
            and validation_model["condition_number"] <= float(condition_number_max)
        )
        correction_validation = _stage3_centered_plane_values(
            validation_design,
            proposal_model["coefficients"],
        )
        validation_centered = baseline_validation - float(
            np.median(baseline_validation)
        )
        removal = -correction_validation
        if float(np.std(removal)) <= 1e-15 or float(np.std(validation_centered)) <= 1e-15:
            correlation = 0.0
        else:
            correlation = float(np.corrcoef(removal, validation_centered)[0, 1])
        after_validation = baseline_validation + correction_validation
        before_rms = float(np.sqrt(np.mean(np.square(validation_centered))))
        after_centered = after_validation - float(np.median(after_validation))
        after_rms = float(np.sqrt(np.mean(np.square(after_centered))))
        improvement = float((before_rms - after_rms) / max(before_rms, 1e-15))
        after_model = _stage3_plane_fit(validation_design, after_validation)
        before_slope = np.asarray(validation_model["coefficients"][1:3])
        after_slope = np.asarray(after_model["coefficients"][1:3])
        fit_slope_span = float(
            abs(float(baseline_model["coefficients"][1]))
            + abs(float(baseline_model["coefficients"][2]))
        )
        validation_slope_span = float(
            abs(float(validation_model["coefficients"][1]))
            + abs(float(validation_model["coefficients"][2]))
        )
        material_floor = max(
            1e-7,
            8.0
            * float(np.finfo(neutral_dtype).eps)
            * max(1.0, float(np.max(np.abs(baseline_plane)))),
        )
        reversed_direction = bool(
            float(np.dot(before_slope, after_slope)) < 0.0
            and float(after_model["slope_significance_sigma"])
            > float(significance_sigma)
        )
        fit_significant = bool(
            fit_slope_span > material_floor
            and
            float(baseline_model["slope_significance_sigma"])
            >= float(significance_sigma)
        )
        component_selected = bool(
            model_valid
            and fit_significant
            and math.isfinite(correlation)
            and correlation >= float(correlation_min)
            and improvement >= float(rms_improvement_min)
            and not reversed_direction
        )
        validation_unresolved = bool(
            not component_selected
            and validation_slope_span > material_floor
            and float(validation_model["slope_significance_sigma"])
            >= float(significance_sigma)
        )
        if component_selected:
            selected[name] = np.asarray(
                proposal_model["coefficients"], dtype=np.float64
            )
        elif validation_unresolved:
            unresolved.append(name)
        report["components"][name] = {
            "status": (
                "selected"
                if component_selected
                else "unresolved"
                if validation_unresolved
                else "not_required"
            ),
            "selected": component_selected,
            "model_valid": model_valid,
            "fit_slope_significance_sigma": float(
                baseline_model["slope_significance_sigma"]
            ),
            "fit_slope_span": fit_slope_span,
            "validation_slope_significance_sigma": float(
                validation_model["slope_significance_sigma"]
            ),
            "validation_slope_span": validation_slope_span,
            "material_floor": material_floor,
            "direction_correlation": correlation,
            "validation_rms_before": before_rms,
            "validation_rms_after": after_rms,
            "validation_rms_improvement": improvement,
            "direction_reversed_over_3sigma": reversed_direction,
            "baseline_model": {
                "coefficients": [float(value) for value in baseline_model["coefficients"]],
                "rank": int(baseline_model["rank"]),
                "condition_number": float(baseline_model["condition_number"]),
            },
            "proposal_model": {
                "coefficients": [float(value) for value in proposal_model["coefficients"]],
                "rank": int(proposal_model["rank"]),
                "condition_number": float(proposal_model["condition_number"]),
            },
        }

    report["selected_components"] = list(selected)
    report["unresolved_components"] = list(dict.fromkeys(unresolved))
    if not selected:
        if unresolved:
            return reject(
                "stage3_spatial_opponent_lineage_unverified",
                "significant held-out spatial opponent gradient remains unresolved",
            )
        report.update(
            status="not_required",
            accepted=True,
            reason_code="stage3_spatial_opponent_not_required",
            issues=[],
        )
        return np.asarray(neutral_image).copy(), report
    assert fit_design is not None

    height, width = baseline.shape[1:]
    yy, xx = np.mgrid[:height, :width]
    normalized_x = xx / max(width - 1, 1) - 0.5
    normalized_y = yy / max(height - 1, 1) - 0.5
    d_rg = np.zeros((height, width), dtype=np.float64)
    d_bg = np.zeros((height, width), dtype=np.float64)
    if "R-G" in selected:
        d_rg = selected["R-G"][1] * normalized_x + selected["R-G"][2] * normalized_y
    if "B-G" in selected:
        d_bg = selected["B-G"][1] * normalized_x + selected["B-G"][2] * normalized_y
    correction = np.empty_like(neutral, dtype=np.float64)
    correction[1] = -0.2126 * d_rg - 0.0722 * d_bg
    correction[0] = correction[1] + d_rg
    correction[2] = correction[1] + d_bg

    margin_scale = max(0.0, min(1.0, 1.0 - float(headroom_margin)))
    alpha = np.ones((height, width), dtype=np.float64)
    for channel in range(3):
        channel_correction = correction[channel]
        negative = channel_correction < 0.0
        positive = channel_correction > 0.0
        alpha[negative] = np.minimum(
            alpha[negative],
            margin_scale * np.maximum(neutral[channel][negative], 0.0)
            / np.maximum(-channel_correction[negative], 1e-15),
        )
        alpha[positive] = np.minimum(
            alpha[positive],
            margin_scale * np.maximum(1.0 - neutral[channel][positive], 0.0)
            / np.maximum(channel_correction[positive], 1e-15),
        )
    baseline_clipped = np.any((neutral <= 0.0) | (neutral >= 1.0), axis=0)
    alpha[baseline_clipped] = 0.0
    alpha = np.clip(alpha, 0.0, 1.0)
    meaningful = np.max(np.abs(correction), axis=0) > 1e-12
    attenuated = meaningful & (alpha < 1.0 - 1e-12)
    fit_mask = _stage3_patch_mask(
        fit_points, width=width, height=height, patch_radius=patch_radius
    )
    validation_mask = _stage3_patch_mask(
        validation_points, width=width, height=height, patch_radius=patch_radius
    )
    attenuated_fraction = float(np.mean(attenuated))
    fit_attenuated = int(np.count_nonzero(attenuated & fit_mask))
    validation_attenuated = int(np.count_nonzero(attenuated & validation_mask))
    report["headroom"] = {
        "method": "continuous_common_vector_taper",
        "margin": float(headroom_margin),
        "attenuated_fraction": attenuated_fraction,
        "attenuated_fraction_max": float(headroom_fraction_max),
        "fit_patch_attenuated_pixel_count": fit_attenuated,
        "validation_patch_attenuated_pixel_count": validation_attenuated,
        "baseline_clipped_pixel_count": int(np.count_nonzero(baseline_clipped)),
    }
    if attenuated_fraction > float(headroom_fraction_max):
        return reject(
            "stage3_spatial_opponent_candidate_rejected",
            "spatial opponent headroom attenuation exceeds 0.1 percent",
        )
    if fit_attenuated or validation_attenuated:
        return reject(
            "stage3_spatial_opponent_candidate_rejected",
            "spatial opponent headroom attenuation intersects fit or validation patches",
        )

    candidate_rgb = neutral + correction * alpha[None, :, :]
    candidate = _restore_stage3_rgb_layout(
        candidate_rgb,
        layout=layout,
        dtype=neutral_dtype,
    )
    persisted, _persisted_layout, persisted_dtype = _stage3_rgb_chw(candidate)
    dtype_epsilon = (
        float(np.finfo(persisted_dtype).eps)
        if np.issubdtype(persisted_dtype, np.floating)
        else float(np.finfo(np.float32).eps)
    )
    tolerance = max(1e-7, 8.0 * dtype_epsilon * max(1.0, float(np.max(np.abs(neutral)))))
    rec709 = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float64)
    luma_drift = float(np.max(np.abs(np.tensordot(rec709, persisted - neutral, axes=(0, 0)))))
    rg_drift = float(np.max(np.abs((persisted[0] - persisted[1]) - (neutral[0] - neutral[1]))))
    bg_drift = float(np.max(np.abs((persisted[2] - persisted[1]) - (neutral[2] - neutral[1]))))
    new_low = int(np.count_nonzero((persisted <= 0.0) & ~(neutral <= 0.0)))
    new_high = int(np.count_nonzero((persisted >= 1.0) & ~(neutral >= 1.0)))
    issues: List[str] = []
    if not bool(np.all(np.isfinite(persisted))):
        issues.append("spatial opponent candidate contains non-finite pixels")
    if luma_drift > tolerance:
        issues.append("spatial opponent candidate changed Rec.709 luminance")
    if "R-G" not in selected and rg_drift > tolerance:
        issues.append("unselected R-G component changed")
    if "B-G" not in selected and bg_drift > tolerance:
        issues.append("unselected B-G component changed")
    if new_low or new_high:
        issues.append("spatial opponent candidate introduced clipping")
    report["invariants"] = {
        "tolerance": tolerance,
        "rec709_luma_max_abs_drift": luma_drift,
        "rg_max_abs_delta": rg_drift,
        "bg_max_abs_delta": bg_drift,
        "new_low_clip_count": new_low,
        "new_high_clip_count": new_high,
        "finite": bool(np.all(np.isfinite(persisted))),
    }
    if issues:
        return reject("stage3_spatial_opponent_candidate_rejected", " | ".join(issues))
    report.update(
        status="ready",
        accepted=True,
        reason_code="stage3_spatial_opponent_correction_applied",
        issues=[],
    )
    return candidate, report


def verify_stage3_spatial_opponent_persistence(
    neutral_image: Any,
    candidate_image: Any,
    projection_report: Dict[str, Any],
) -> Tuple[bool, Dict[str, Any]]:
    """Verify zero-luma and unselected-component invariants after persistence."""
    report: Dict[str, Any] = {"status": "rejected", "accepted": False, "issues": []}
    try:
        neutral, neutral_layout, _neutral_dtype = _stage3_rgb_chw(neutral_image)
        candidate, candidate_layout, candidate_dtype = _stage3_rgb_chw(candidate_image)
    except (TypeError, ValueError) as error:
        report["issues"] = [str(error)]
        return False, report
    if neutral_layout != candidate_layout or neutral.shape != candidate.shape:
        report["issues"] = ["persisted spatial opponent candidate shape changed"]
        return False, report
    dtype_epsilon = (
        float(np.finfo(candidate_dtype).eps)
        if np.issubdtype(candidate_dtype, np.floating)
        else float(np.finfo(np.float32).eps)
    )
    tolerance = max(1e-7, 8.0 * dtype_epsilon * max(1.0, float(np.max(np.abs(neutral)))))
    rec709 = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float64)
    luma_drift = float(np.max(np.abs(np.tensordot(rec709, candidate - neutral, axes=(0, 0)))))
    rg_drift = float(np.max(np.abs((candidate[0] - candidate[1]) - (neutral[0] - neutral[1]))))
    bg_drift = float(np.max(np.abs((candidate[2] - candidate[1]) - (neutral[2] - neutral[1]))))
    selected = set(projection_report.get("selected_components") or [])
    new_low = int(np.count_nonzero((candidate <= 0.0) & ~(neutral <= 0.0)))
    new_high = int(np.count_nonzero((candidate >= 1.0) & ~(neutral >= 1.0)))
    issues: List[str] = []
    if not bool(np.all(np.isfinite(candidate))):
        issues.append("persisted spatial opponent candidate contains non-finite pixels")
    if luma_drift > tolerance:
        issues.append("persisted spatial opponent candidate changed Rec.709 luminance")
    if "R-G" not in selected and rg_drift > tolerance:
        issues.append("persisted unselected R-G component changed")
    if "B-G" not in selected and bg_drift > tolerance:
        issues.append("persisted unselected B-G component changed")
    if new_low or new_high:
        issues.append("persisted spatial opponent candidate introduced clipping")
    report.update(
        tolerance=tolerance,
        rec709_luma_max_abs_drift=luma_drift,
        rg_max_abs_delta=rg_drift,
        bg_max_abs_delta=bg_drift,
        new_low_clip_count=new_low,
        new_high_clip_count=new_high,
        selected_components=sorted(selected),
        finite=bool(np.all(np.isfinite(candidate))),
        pixels_changed=not bool(np.array_equal(candidate, neutral)),
        issues=issues,
    )
    if issues:
        return False, report
    report.update(status="accepted", accepted=True)
    return True, report


def _sample_coverage(
    points: List[Tuple[float, float]],
    *,
    width: int,
    height: int,
) -> Dict[str, Any]:
    if not points:
        return {
            "quadrants": 0,
            "grid_cells": 0,
            "x_span_ratio": 0.0,
            "y_span_ratio": 0.0,
        }
    normalized = [
        (
            _clamp(float(x) / max(width - 1, 1)),
            _clamp(float(y) / max(height - 1, 1)),
        )
        for x, y in points
    ]
    quadrants = {
        (int(x >= 0.5), int(y >= 0.5))
        for x, y in normalized
    }
    cells = {
        (
            min(3, int(x * 4.0)),
            min(3, int(y * 4.0)),
        )
        for x, y in normalized
    }
    x_values = [point[0] for point in normalized]
    y_values = [point[1] for point in normalized]
    return {
        "quadrants": len(quadrants),
        "grid_cells": len(cells),
        "x_span_ratio": max(x_values) - min(x_values),
        "y_span_ratio": max(y_values) - min(y_values),
    }


def _coverage_ready(
    coverage: Dict[str, Any],
    *,
    minimum_cells: int,
) -> bool:
    return bool(
        int(coverage.get("quadrants", 0) or 0)
        >= STAGE3_MIN_SPATIAL_QUADRANTS
        and int(coverage.get("grid_cells", 0) or 0) >= minimum_cells
        and float(coverage.get("x_span_ratio", 0.0) or 0.0)
        >= STAGE3_MIN_AXIS_SPAN_RATIO
        and float(coverage.get("y_span_ratio", 0.0) or 0.0)
        >= STAGE3_MIN_AXIS_SPAN_RATIO
    )


def split_background_sample_points(
    points: List[Tuple[float, float]],
    image: Any,
    *,
    point_sources: Optional[Sequence[str]] = None,
    minimum_regular_validation: int = 0,
    validation_ratio: float = 0.25,
    minimum_total: int = 12,
    minimum_fit: int = 8,
    minimum_validation: int = 4,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], Dict[str, Any]]:
    """Deterministically reserve spatially distributed samples for validation.

    ``point_sources`` is an optional list aligned with ``points``.  Stage 3's
    conservative recovery uses it to keep a spatially independent regular-grid
    subset in validation while older callers retain the coordinate-only split.
    """
    source_points = [
        (float(point[0]), float(point[1]))
        for point in points
    ]
    if point_sources is None:
        normalized_sources = ["unknown"] * len(source_points)
    elif len(point_sources) != len(source_points):
        return [], [], {
            "status": "unavailable",
            "reason": "point provenance count does not match sample count",
            "sample_count": len(source_points),
            "provenance_count": len(point_sources),
        }
    else:
        normalized_sources = [
            str(source or "unknown").strip().lower()
            for source in point_sources
        ]
    minimum_regular_validation = max(0, int(minimum_regular_validation))
    minimum_total = max(12, int(minimum_total))
    minimum_fit = max(8, int(minimum_fit))
    minimum_validation = max(4, int(minimum_validation))
    fit_minimum_cells = min(STAGE3_MIN_SPATIAL_GRID_CELLS, minimum_fit)
    validation_minimum_cells = min(
        STAGE3_MIN_SPATIAL_GRID_CELLS,
        minimum_validation,
    )
    validation_ratio = _clamp(validation_ratio, 0.10, 0.40)
    try:
        luminance, _scale = _native_luminance(image)
    except (TypeError, ValueError) as error:
        return [], [], {
            "status": "unavailable",
            "reason": str(error),
            "sample_count": len(source_points),
        }
    height, width = luminance.shape
    if len(source_points) < minimum_total:
        return [], [], {
            "status": "insufficient_samples",
            "reason": (
                f"audited samples {len(source_points)}<{minimum_total}"
            ),
            "sample_count": len(source_points),
            "minimum_total": minimum_total,
        }

    validation_count = max(
        minimum_validation,
        int(math.ceil(len(source_points) * validation_ratio)),
    )
    validation_count = min(validation_count, len(source_points) - minimum_fit)
    if validation_count < minimum_validation:
        return [], [], {
            "status": "insufficient_samples",
            "reason": "fit/validation minima cannot both be satisfied",
            "sample_count": len(source_points),
            "minimum_fit": minimum_fit,
            "minimum_validation": minimum_validation,
        }

    normalized = [
        (
            _clamp(x / max(width - 1, 1)),
            _clamp(y / max(height - 1, 1)),
        )
        for x, y in source_points
    ]
    cell_by_index = {
        index: (
            min(3, int(point[0] * 4.0)),
            min(3, int(point[1] * 4.0)),
        )
        for index, point in enumerate(normalized)
    }
    members_by_cell: Dict[Tuple[int, int], List[int]] = {}
    for index, cell in cell_by_index.items():
        members_by_cell.setdefault(cell, []).append(index)
    eligible = [
        index
        for index in range(len(source_points))
        if len(members_by_cell[cell_by_index[index]]) >= 2
    ]
    if len(eligible) < validation_count:
        return [], [], {
            "status": "insufficient_safe_coverage",
            "reason": "not enough repeat-covered grid cells to freeze validation points",
            "sample_count": len(source_points),
            "eligible_validation_count": len(eligible),
            "validation_target_count": validation_count,
        }

    def source_counts(indexes: Sequence[int]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for index in indexes:
            source = normalized_sources[index]
            counts[source] = counts.get(source, 0) + 1
        return counts

    def regular_coverage(indexes: Sequence[int]) -> Dict[str, Any]:
        regular_points = [
            source_points[index]
            for index in indexes
            if normalized_sources[index] == "regular_grid"
        ]
        return {
            "count": len(regular_points),
            **_sample_coverage(
                regular_points,
                width=width,
                height=height,
            ),
        }

    def build_selection(start: int) -> List[int]:
        chosen = [start]
        chosen_set = {start}
        chosen_per_cell = {cell_by_index[start]: 1}
        while len(chosen) < validation_count:
            best_index = None
            best_key = None
            for index in eligible:
                if index in chosen_set:
                    continue
                cell = cell_by_index[index]
                selected_in_cell = chosen_per_cell.get(cell, 0)
                if selected_in_cell >= len(members_by_cell[cell]) - 1:
                    continue
                trial_indexes = chosen + [index]
                trial_points = [source_points[item] for item in trial_indexes]
                coverage = _sample_coverage(
                    trial_points,
                    width=width,
                    height=height,
                )
                point = normalized[index]
                regular = regular_coverage(trial_indexes)
                minimum_distance = min(
                    math.hypot(
                        point[0] - normalized[other][0],
                        point[1] - normalized[other][1],
                    )
                    for other in chosen
                )
                key = (
                    min(
                        int(regular["count"]),
                        minimum_regular_validation,
                    ),
                    min(
                        int(regular["quadrants"]),
                        STAGE3_MIN_SPATIAL_QUADRANTS,
                    ),
                    min(
                        int(regular["grid_cells"]),
                        min(
                            STAGE3_MIN_SPATIAL_GRID_CELLS,
                            minimum_regular_validation,
                        ),
                    ),
                    min(
                        int(coverage["quadrants"]),
                        STAGE3_MIN_SPATIAL_QUADRANTS,
                    ),
                    min(
                        int(coverage["grid_cells"]),
                        STAGE3_MIN_SPATIAL_GRID_CELLS,
                    ),
                    min(
                        float(coverage["x_span_ratio"]),
                        STAGE3_MIN_AXIS_SPAN_RATIO,
                    )
                    + min(
                        float(coverage["y_span_ratio"]),
                        STAGE3_MIN_AXIS_SPAN_RATIO,
                    ),
                    minimum_distance,
                    -index,
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_index = index
            if best_index is None:
                break
            chosen.append(best_index)
            chosen_set.add(best_index)
            cell = cell_by_index[best_index]
            chosen_per_cell[cell] = chosen_per_cell.get(cell, 0) + 1
        return chosen

    best_selection: List[int] = []
    best_selection_key = None
    for start in eligible:
        chosen = build_selection(start)
        if len(chosen) != validation_count:
            continue
        chosen_set = set(chosen)
        validation_points = [source_points[index] for index in chosen]
        fit_points = [
            point
            for index, point in enumerate(source_points)
            if index not in chosen_set
        ]
        validation_coverage = _sample_coverage(
            validation_points,
            width=width,
            height=height,
        )
        fit_coverage = _sample_coverage(
            fit_points,
            width=width,
            height=height,
        )
        validation_ready = _coverage_ready(
            validation_coverage,
            minimum_cells=validation_minimum_cells,
        )
        fit_ready = _coverage_ready(
            fit_coverage,
            minimum_cells=fit_minimum_cells,
        )
        validation_regular = regular_coverage(chosen)
        regular_ready = bool(
            minimum_regular_validation <= 0
            or (
                int(validation_regular["count"])
                >= minimum_regular_validation
                and int(validation_regular["quadrants"])
                >= STAGE3_MIN_SPATIAL_QUADRANTS
                and int(validation_regular["grid_cells"])
                >= min(
                    STAGE3_MIN_SPATIAL_GRID_CELLS,
                    minimum_regular_validation,
                )
            )
        )
        selection_key = (
            int(validation_ready and fit_ready and regular_ready),
            min(
                int(validation_regular["count"]),
                minimum_regular_validation,
            ),
            min(
                int(validation_regular["quadrants"]),
                STAGE3_MIN_SPATIAL_QUADRANTS,
            ),
            min(
                int(validation_regular["grid_cells"]),
                min(
                    STAGE3_MIN_SPATIAL_GRID_CELLS,
                    minimum_regular_validation,
                ),
            ),
            min(
                int(validation_coverage["quadrants"]),
                STAGE3_MIN_SPATIAL_QUADRANTS,
            ),
            min(
                int(validation_coverage["grid_cells"]),
                STAGE3_MIN_SPATIAL_GRID_CELLS,
            ),
            min(
                float(validation_coverage["x_span_ratio"]),
                STAGE3_MIN_AXIS_SPAN_RATIO,
            )
            + min(
                float(validation_coverage["y_span_ratio"]),
                STAGE3_MIN_AXIS_SPAN_RATIO,
            ),
            min(
                int(fit_coverage["quadrants"]),
                STAGE3_MIN_SPATIAL_QUADRANTS,
            ),
            min(
                int(fit_coverage["grid_cells"]),
                STAGE3_MIN_SPATIAL_GRID_CELLS,
            ),
            min(
                float(fit_coverage["x_span_ratio"]),
                STAGE3_MIN_AXIS_SPAN_RATIO,
            )
            + min(
                float(fit_coverage["y_span_ratio"]),
                STAGE3_MIN_AXIS_SPAN_RATIO,
            ),
            tuple(-index for index in sorted(chosen)),
        )
        if best_selection_key is None or selection_key > best_selection_key:
            best_selection_key = selection_key
            best_selection = chosen

    selected_set = set(best_selection)
    validation_points = [
        point
        for index, point in enumerate(source_points)
        if index in selected_set
    ]
    fit_points = [
        point
        for index, point in enumerate(source_points)
        if index not in selected_set
    ]
    fit_coverage = _sample_coverage(fit_points, width=width, height=height)
    validation_coverage = _sample_coverage(
        validation_points,
        width=width,
        height=height,
    )
    validation_regular_coverage = regular_coverage(best_selection)
    regular_validation_ready = bool(
        minimum_regular_validation <= 0
        or (
            int(validation_regular_coverage["count"])
            >= minimum_regular_validation
            and int(validation_regular_coverage["quadrants"])
            >= STAGE3_MIN_SPATIAL_QUADRANTS
            and int(validation_regular_coverage["grid_cells"])
            >= min(
                STAGE3_MIN_SPATIAL_GRID_CELLS,
                minimum_regular_validation,
            )
        )
    )
    ready = bool(
        len(fit_points) >= minimum_fit
        and len(validation_points) >= minimum_validation
        and _coverage_ready(
            fit_coverage,
            minimum_cells=fit_minimum_cells,
        )
        and _coverage_ready(
            validation_coverage,
            minimum_cells=validation_minimum_cells,
        )
        and regular_validation_ready
    )
    fit_indexes = [
        index for index in range(len(source_points)) if index not in selected_set
    ]
    validation_indexes = [
        index for index in range(len(source_points)) if index in selected_set
    ]
    report = {
        "status": "ready" if ready else "insufficient_safe_coverage",
        "coordinate_system": "siril_bottom_left",
        "sample_count": len(source_points),
        "fit_count": len(fit_points),
        "validation_count": len(validation_points),
        "validation_ratio": validation_ratio,
        "minimum_total": minimum_total,
        "minimum_fit": minimum_fit,
        "minimum_validation": minimum_validation,
        "fit_minimum_grid_cells": fit_minimum_cells,
        "validation_minimum_grid_cells": validation_minimum_cells,
        "minimum_regular_validation": minimum_regular_validation,
        "fit_coverage": fit_coverage,
        "validation_coverage": validation_coverage,
        "validation_regular_coverage": validation_regular_coverage,
        "regular_validation_ready": regular_validation_ready,
        "fit_source_counts": source_counts(fit_indexes),
        "validation_source_counts": source_counts(validation_indexes),
        "fit_points": [[x, y] for x, y in fit_points] if ready else [],
        "validation_points": (
            [[x, y] for x, y in validation_points]
            if ready
            else []
        ),
        "safety_contract": [
            "validation points never participate in Polynomial or RBF fitting",
            "fit and validation pools both require broad spatial coverage",
            "recovery validation retains spatial regular-grid evidence",
            "split is deterministic for identical audited samples and image geometry",
        ],
    }
    if not ready:
        report["reason"] = (
            "regular-grid validation provenance is insufficient"
            if not regular_validation_ready
            else "fit or validation spatial coverage is insufficient"
        )
        return [], [], report
    return fit_points, validation_points, report


def measure_background_validation(
    image: Any,
    points: List[Tuple[float, float]],
    *,
    patch_radius: int = 12,
    minimum_count: int = 8,
    value_scale: Optional[float] = None,
    support_mask: Optional[np.ndarray] = None,
    minimum_support_fraction: float = STAGE3_DENSE_STAR_SAMPLE_SUPPORT_MIN,
) -> Dict[str, Any]:
    """Measure held-out patch medians without independently renormalizing images."""
    try:
        luminance, resolved_scale = _native_luminance(
            image,
            value_scale=value_scale,
        )
    except (TypeError, ValueError) as error:
        return {
            "status": "unavailable",
            "reason": str(error),
            "sample_count": 0,
            "expected_count": len(points),
        }
    height, width = luminance.shape
    frozen_support = None
    if support_mask is not None:
        candidate_support = np.asarray(support_mask, dtype=bool)
        if candidate_support.shape != (height, width):
            return {
                "status": "unavailable",
                "reason": "validation support mask shape mismatch",
                "sample_count": 0,
                "expected_count": len(points),
            }
        frozen_support = candidate_support
    minimum_support_fraction = _clamp(minimum_support_fraction, 0.50, 1.0)
    patch_radius = max(
        2,
        min(int(patch_radius), 24, max(2, min(height, width) // 12)),
    )
    medians: List[float] = []
    patch_mads: List[float] = []
    patch_uncertainties: List[float] = []
    uncertainty_details: List[Dict[str, Any]] = []
    support_fractions: List[float] = []
    absolute_minimum = math.inf
    absolute_maximum = -math.inf
    supported_pixel_count = 0
    low_clip_count = 0
    high_clip_count = 0
    for raw_x, raw_y in points:
        x = int(round(float(raw_x)))
        y = int(round(float(height - 1 - raw_y)))
        if (
            x - patch_radius < 0
            or x + patch_radius >= width
            or y - patch_radius < 0
            or y + patch_radius >= height
        ):
            continue
        patch = luminance[
            y - patch_radius : y + patch_radius + 1,
            x - patch_radius : x + patch_radius + 1,
        ]
        if frozen_support is not None:
            patch_support = frozen_support[
                y - patch_radius : y + patch_radius + 1,
                x - patch_radius : x + patch_radius + 1,
            ]
            support_fraction = float(np.mean(patch_support))
            if support_fraction < minimum_support_fraction:
                continue
            finite = patch[patch_support & np.isfinite(patch)]
            if finite.size != int(np.count_nonzero(patch_support)):
                continue
            uncertainty_patch = np.where(patch_support, patch, np.nan)
            support_fractions.append(support_fraction)
        else:
            finite = patch[np.isfinite(patch)]
            if finite.size != patch.size:
                continue
            uncertainty_patch = patch
            support_fractions.append(1.0)
        median = float(np.median(finite))
        absolute_minimum = min(absolute_minimum, float(np.min(finite)))
        absolute_maximum = max(absolute_maximum, float(np.max(finite)))
        supported_pixel_count += int(finite.size)
        low_clip_count += int(np.count_nonzero(finite <= 0.0))
        high_clip_count += int(np.count_nonzero(finite >= 1.0))
        medians.append(median)
        patch_mads.append(float(np.median(np.abs(finite - median))))
        uncertainty, detail = _patch_median_uncertainty(uncertainty_patch)
        patch_uncertainties.append(uncertainty)
        uncertainty_details.append(detail)

    expected_count = len(points)
    ready = bool(
        len(medians) == expected_count
        and len(medians) >= max(4, int(minimum_count))
    )
    if not ready:
        return {
            "schema_version": STAGE3_BACKGROUND_VALIDATION_SCHEMA,
            "status": "unavailable",
            "reason": "validation patches are incomplete",
            "sample_count": len(medians),
            "expected_count": expected_count,
            "patch_radius": patch_radius,
            "value_scale": resolved_scale,
            "masked_support": frozen_support is not None,
        }
    values = np.asarray(medians, dtype=np.float64)
    center = float(np.median(values))
    p10, p90 = (float(value) for value in np.quantile(values, (0.10, 0.90)))
    return {
        "schema_version": STAGE3_BACKGROUND_VALIDATION_SCHEMA,
        "status": "ready",
        "sample_count": len(medians),
        "expected_count": expected_count,
        "patch_radius": patch_radius,
        "value_scale": resolved_scale,
        "masked_support": frozen_support is not None,
        "minimum_support_fraction": (
            minimum_support_fraction if frozen_support is not None else None
        ),
        "support_fraction_min": float(min(support_fractions)),
        "support_fraction_median": float(np.median(support_fractions)),
        "minimum": float(absolute_minimum),
        "maximum": float(absolute_maximum),
        "supported_pixel_count": int(supported_pixel_count),
        "low_clip_count": int(low_clip_count),
        "high_clip_count": int(high_clip_count),
        "median": center,
        "p10": p10,
        "p90": p90,
        "robust_span": max(0.0, p90 - p10),
        "mad": 1.4826 * float(np.median(np.abs(values - center))),
        "patch_mad_median": float(np.median(patch_mads)),
        "patch_median_uncertainty": float(np.median(patch_uncertainties)),
        "span_standard_error": background_span_standard_error(
            {"patch_median_uncertainty": float(np.median(patch_uncertainties))}
        ),
        "uncertainty_method": (
            "correlation_aware_nonoverlapping_block_medians"
            if any(
                detail.get("method", "").startswith("max_pixel_and_")
                for detail in uncertainty_details
            )
            else "pixel_standard_error_fallback"
        ),
        "uncertainty_block_count_median": float(
            np.median(
                [float(detail.get("block_count", 0) or 0) for detail in uncertainty_details]
            )
        ),
    }


def assess_background_direction_reversal(
    baseline_image: Any,
    candidate_image: Any,
    points: Sequence[Tuple[float, float]],
    *,
    patch_radius: int = 12,
    significance_sigma: float = STAGE3_SIGNIFICANCE_SIGMA,
    support_mask: Optional[np.ndarray] = None,
    minimum_support_fraction: float = STAGE3_DENSE_STAR_SAMPLE_SUPPORT_MIN,
) -> Tuple[bool, Dict[str, Any]]:
    """Reject a statistically significant held-out degree-1 gradient reversal."""

    def fit_plane(
        image: Any,
        *,
        value_scale: Optional[float],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[float], Optional[str]]:
        try:
            luminance, resolved_scale = _native_luminance(
                image,
                value_scale=value_scale,
            )
        except (TypeError, ValueError) as error:
            return None, value_scale, str(error)
        height, width = luminance.shape
        frozen_support = None
        if support_mask is not None:
            candidate_support = np.asarray(support_mask, dtype=bool)
            if candidate_support.shape != (height, width):
                return None, resolved_scale, "direction support mask shape mismatch"
            frozen_support = candidate_support
        radius = max(2, min(int(patch_radius), 24))
        rows: List[List[float]] = []
        values: List[float] = []
        uncertainties: List[float] = []
        for raw_x, raw_y in points:
            x = int(round(float(raw_x)))
            y = int(round(height - 1 - float(raw_y)))
            if (
                x - radius < 0
                or x + radius >= width
                or y - radius < 0
                or y + radius >= height
            ):
                continue
            patch = luminance[
                y - radius : y + radius + 1,
                x - radius : x + radius + 1,
            ]
            if frozen_support is not None:
                patch_support = frozen_support[
                    y - radius : y + radius + 1,
                    x - radius : x + radius + 1,
                ]
                if float(np.mean(patch_support)) < _clamp(
                    minimum_support_fraction,
                    0.50,
                    1.0,
                ):
                    continue
                values_patch = patch[patch_support & np.isfinite(patch)]
                if values_patch.size != int(np.count_nonzero(patch_support)):
                    continue
                uncertainty_patch = np.where(patch_support, patch, np.nan)
            else:
                if not bool(np.all(np.isfinite(patch))):
                    continue
                values_patch = patch.reshape(-1)
                uncertainty_patch = patch
            rows.append(
                [
                    1.0,
                    float(raw_x) / max(width - 1, 1),
                    float(y) / max(height - 1, 1),
                ]
            )
            values.append(float(np.median(values_patch)))
            uncertainty, _detail = _patch_median_uncertainty(
                uncertainty_patch
            )
            uncertainties.append(float(uncertainty))
        if len(rows) != len(points) or len(rows) < 4:
            return None, resolved_scale, "held-out directional patches are incomplete"
        design = np.asarray(rows, dtype=np.float64)
        samples = np.asarray(values, dtype=np.float64)
        try:
            coefficients, _residuals, rank, _singular = np.linalg.lstsq(
                design,
                samples,
                rcond=None,
            )
            if int(rank) != 3:
                return None, resolved_scale, "held-out directional fit rank is invalid"
            fitted = design @ coefficients
            dof = max(1, len(samples) - 3)
            residual_variance = float(
                np.sum((samples - fitted) ** 2) / dof
            )
            patch_variance = float(np.median(uncertainties)) ** 2
            covariance = max(residual_variance, patch_variance, 1e-24) * np.linalg.inv(
                design.T @ design
            )
        except np.linalg.LinAlgError as error:
            return None, resolved_scale, f"held-out directional fit failed: {error}"
        slope = np.asarray(coefficients[1:3], dtype=np.float64)
        slope_se = np.sqrt(np.maximum(np.diag(covariance)[1:3], 0.0))
        return (
            {
                "coefficients": [float(value) for value in coefficients],
                "gradient": [float(value) for value in slope],
                "gradient_standard_error": [
                    float(value) for value in slope_se
                ],
                "gradient_norm": float(np.linalg.norm(slope)),
                "gradient_norm_standard_error": float(np.linalg.norm(slope_se)),
                "residual_rms": float(math.sqrt(max(residual_variance, 0.0))),
                "sample_count": len(samples),
            },
            resolved_scale,
            None,
        )

    before, value_scale, before_error = fit_plane(
        baseline_image,
        value_scale=None,
    )
    after, _resolved_scale, after_error = fit_plane(
        candidate_image,
        value_scale=value_scale,
    )
    report: Dict[str, Any] = {
        "status": "rejected",
        "accepted": False,
        "severity": "hard_rejected",
        "significance_sigma": float(significance_sigma),
        "uncertainty_method": "heldout_patch_plane_covariance",
        "before": before,
        "after": after,
        "issues": [],
    }
    if before is None or after is None:
        report["issues"] = [
            before_error
            or after_error
            or "held-out directional evidence is unavailable"
        ]
        return False, report

    before_gradient = np.asarray(before["gradient"], dtype=np.float64)
    after_gradient = np.asarray(after["gradient"], dtype=np.float64)
    before_se = np.asarray(
        before["gradient_standard_error"],
        dtype=np.float64,
    )
    after_se = np.asarray(
        after["gradient_standard_error"],
        dtype=np.float64,
    )
    component_reversals: List[str] = []
    for name, before_value, after_value, before_error, after_error in zip(
        ("x", "y"),
        before_gradient,
        after_gradient,
        before_se,
        after_se,
    ):
        if (
            before_value * after_value < 0.0
            and abs(before_value) > significance_sigma * before_error
            and abs(after_value) > significance_sigma * after_error
        ):
            component_reversals.append(name)
    dot_product = float(np.dot(before_gradient, after_gradient))
    before_significant = bool(
        float(before["gradient_norm"])
        > significance_sigma * float(before["gradient_norm_standard_error"])
    )
    after_significant = bool(
        float(after["gradient_norm"])
        > significance_sigma * float(after["gradient_norm_standard_error"])
    )
    vector_reversed = bool(
        dot_product < 0.0 and before_significant and after_significant
    )
    issues: List[str] = []
    if component_reversals:
        issues.append(
            "held-out gradient component reversed beyond three-sigma: "
            + ",".join(component_reversals)
        )
    if vector_reversed:
        issues.append("held-out gradient vector reversed beyond three-sigma")
    report.update(
        dot_product=dot_product,
        component_reversals=component_reversals,
        vector_reversed=vector_reversed,
        before_significant=before_significant,
        after_significant=after_significant,
        issues=issues,
    )
    if issues:
        return False, report
    report.update(
        status="accepted",
        accepted=True,
        severity="normal",
    )
    return True, report


def _sample_spatial_model_diagnostics(
    image: Any,
    points: List[Tuple[float, float]],
    *,
    patch_radius: int = 12,
    support_mask: Optional[np.ndarray] = None,
    minimum_support_fraction: float = STAGE3_DENSE_STAR_SAMPLE_SUPPORT_MIN,
) -> Dict[str, Any]:
    """Compare low-order directional and radial models on audited sky only."""
    try:
        luminance, value_scale = _native_luminance(image)
    except (TypeError, ValueError) as error:
        return {
            "schema_version": STAGE3_BACKGROUND_VALIDATION_SCHEMA,
            "status": "unavailable",
            "reason": str(error),
        }
    height, width = luminance.shape
    frozen_support = None
    if support_mask is not None:
        candidate_support = np.asarray(support_mask, dtype=bool)
        if candidate_support.shape != (height, width):
            return {
                "schema_version": STAGE3_BACKGROUND_VALIDATION_SCHEMA,
                "status": "unavailable",
                "reason": "spatial support mask shape mismatch",
            }
        frozen_support = candidate_support
    radius = max(2, min(int(patch_radius), 24))
    records: List[Tuple[float, float, float, float, float]] = []
    for raw_x, raw_y in points:
        x = int(round(float(raw_x)))
        y = int(round(float(height - 1 - raw_y)))
        if (
            x - radius < 0
            or x + radius >= width
            or y - radius < 0
            or y + radius >= height
        ):
            continue
        patch = luminance[y - radius : y + radius + 1, x - radius : x + radius + 1]
        if frozen_support is not None:
            patch_support = frozen_support[
                y - radius : y + radius + 1,
                x - radius : x + radius + 1,
            ]
            if float(np.mean(patch_support)) < _clamp(
                minimum_support_fraction,
                0.50,
                1.0,
            ):
                continue
            finite = patch[patch_support & np.isfinite(patch)]
            if finite.size != int(np.count_nonzero(patch_support)):
                continue
            uncertainty_patch = np.where(patch_support, patch, np.nan)
        else:
            finite = patch[np.isfinite(patch)]
            if finite.size != patch.size:
                continue
            uncertainty_patch = patch
        median = float(np.median(finite))
        mad = 1.4826 * float(np.median(np.abs(finite - median)))
        median_uncertainty, _uncertainty_detail = _patch_median_uncertainty(
            uncertainty_patch
        )
        records.append(
            (
                float(x) / max(width - 1, 1) - 0.5,
                float(y) / max(height - 1, 1) - 0.5,
                median,
                mad,
                median_uncertainty,
            )
        )
    if len(records) < STAGE3_MIN_VALIDATION_PATCHES:
        return {
            "schema_version": STAGE3_BACKGROUND_VALIDATION_SCHEMA,
            "status": "unavailable",
            "reason": "insufficient complete audited sky patches",
            "sample_count": len(records),
        }

    values = np.asarray([record[2] for record in records], dtype=np.float64)
    x = np.asarray([record[0] for record in records], dtype=np.float64)
    y = np.asarray([record[1] for record in records], dtype=np.float64)
    r2 = x * x + y * y
    plane_design = np.column_stack((np.ones_like(x), x, y))
    radial_design = np.column_stack((np.ones_like(r2), r2))

    def fit(design: np.ndarray) -> Dict[str, Any]:
        coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
        residual = values - design @ coefficients
        rss = max(float(np.sum(residual * residual)), np.finfo(np.float64).tiny)
        count = values.size
        parameter_count = design.shape[1]
        bic = count * math.log(rss / count) + parameter_count * math.log(count)
        return {
            "coefficients": [float(value) for value in coefficients],
            "residual_mad": _robust_sigma(residual),
            "bic": float(bic),
        }

    try:
        plane = fit(plane_design)
        radial = fit(radial_design)
    except np.linalg.LinAlgError as error:
        return {
            "schema_version": STAGE3_BACKGROUND_VALIDATION_SCHEMA,
            "status": "unavailable",
            "reason": f"spatial fit failed: {error}",
        }

    center = float(np.median(values))
    p10, p90 = (float(value) for value in np.quantile(values, (0.10, 0.90)))
    span = max(0.0, p90 - p10)
    patch_mad = float(np.median([record[3] for record in records]))
    median_standard_error = float(np.median([record[4] for record in records]))
    span_standard_error = math.sqrt(2.0) * median_standard_error
    significance_limit = max(
        STAGE3_SIGNIFICANCE_SIGMA * span_standard_error,
        1e-12,
    )
    spatial_variation_supported = bool(span > significance_limit)
    radial_edge_delta = float(radial["coefficients"][1]) * 0.5
    radial_preferred = bool(
        spatial_variation_supported
        and radial_edge_delta < -significance_limit
        and float(radial["bic"]) + STAGE3_RADIAL_BIC_DELTA_MIN
        < float(plane["bic"])
        and float(radial["residual_mad"])
        <= max(
            span * STAGE3_RADIAL_RESIDUAL_SPAN_RATIO_MAX,
            significance_limit,
        )
    )
    return {
        "schema_version": STAGE3_BACKGROUND_VALIDATION_SCHEMA,
        "status": "ready",
        "sample_count": len(records),
        "value_scale": value_scale,
        "median": center,
        "p10": p10,
        "p90": p90,
        "robust_span": span,
        "patch_mad_median": patch_mad,
        "patch_median_standard_error": median_standard_error,
        "span_standard_error": span_standard_error,
        "spatial_significance_limit_3sigma": significance_limit,
        "uncertainty_method": "correlation_aware_nonoverlapping_block_medians",
        "spatial_variation_supported": spatial_variation_supported,
        "plane_model": plane,
        "radial_model": radial,
        "radial_edge_delta": radial_edge_delta,
        "radial_multiplicative_pattern_supported": radial_preferred,
    }


def assess_background_process(
    image: Any,
    points: List[Tuple[float, float]],
    safe_sample_report: Dict[str, Any],
    baseline_validation: Dict[str, Any],
    pattern_report: Dict[str, Any],
    *,
    input_profile: Optional[Dict[str, Any]] = None,
    diffuse_context: Optional[Dict[str, Any]] = None,
    patch_radius: int = 12,
    support_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Authorize Stage 3 from process evidence instead of project score gates."""
    profile = input_profile or {}
    state = str(profile.get("state") or "unknown").strip().lower()
    linear_confirmed = bool(
        state == "linear" and profile.get("safe_for_linear_steps", True) is not False
    )
    sample_ready = bool((safe_sample_report or {}).get("status") == "ready")
    validation_ready = bool((baseline_validation or {}).get("status") == "ready")
    masks = (safe_sample_report or {}).get("masks") or {}
    mask_evidence = (safe_sample_report or {}).get("mask_evidence") or {}
    coverage = (safe_sample_report or {}).get("coverage") or {}
    usable_sky_fraction = mask_evidence.get(
        "usable_sky_fraction",
        masks.get("usable_sky_fraction"),
    )
    sky_grid_cells = int(
        coverage.get(
            "available_grid_cells",
            masks.get("usable_sky_grid_cells", 0),
        )
        or 0
    )
    sky_supported = bool(
        sample_ready
        and validation_ready
        and sky_grid_cells >= STAGE3_MIN_SPATIAL_GRID_CELLS
    )
    spatial = _sample_spatial_model_diagnostics(
        image,
        points,
        patch_radius=patch_radius,
        support_mask=support_mask,
    )
    spatial_supported = bool(
        spatial.get("status") == "ready"
        and spatial.get("spatial_variation_supported")
    )
    pattern_detected = bool((pattern_report or {}).get("detected", False))
    radial_shape_supported = bool(
        spatial.get("radial_multiplicative_pattern_supported", False)
    )
    diffuse = bool(
        (diffuse_context or {}).get("diffuse")
        or (diffuse_context or {}).get("emission_diffuse")
        or (diffuse_context or {}).get("pixel_signal_protection")
    )
    target_type = str((diffuse_context or {}).get("target_type") or "").lower()

    if not sky_supported:
        mechanism = "target_signal_or_sky_limited"
        correction_mode = "preserve_and_review"
    elif pattern_detected and not spatial_supported:
        mechanism = "directional_pattern_or_walking_noise"
        correction_mode = "defer_to_calibration_or_stacking"
    elif radial_shape_supported:
        # A radial profile in one already-integrated image is not sufficient
        # evidence to distinguish multiplicative flat-field error from an
        # additive radial sky gradient.  Keep the flat-field interpretation as
        # an advisory, but let the normal held-out-sky and target-fidelity gates
        # decide whether a bounded subtraction candidate is safe.
        mechanism = "radial_low_frequency_gradient_ambiguous"
        correction_mode = "subtract_with_master_flat_advisory"
    elif spatial_supported:
        mechanism = "additive_low_frequency_gradient"
        correction_mode = "subtract"
    else:
        mechanism = "no_measurable_low_frequency_gradient"
        correction_mode = "preserve"

    hard_blocks: List[str] = []
    if not linear_confirmed:
        hard_blocks.append("input_linearity_not_confirmed")
    if not sky_supported:
        hard_blocks.append("insufficient_source_masked_true_sky_support")
    if mechanism == "directional_pattern_or_walking_noise":
        hard_blocks.append("directional_noise_is_not_a_sky_gradient")
    low_complexity_required = bool(
        diffuse
        or target_type in {"large_galaxy", "galaxy", "dark_nebula"}
        or float(usable_sky_fraction or 0.0)
        < STAGE3_MIN_USABLE_SKY_FRACTION
    )
    should_evaluate = bool(
        not hard_blocks
        and mechanism in {
            "additive_low_frequency_gradient",
            "radial_low_frequency_gradient_ambiguous",
        }
    )
    advisory_reasons: List[str] = []
    if radial_shape_supported:
        advisory_reasons.append(
            "radial_shape_cannot_distinguish_additive_gradient_from_flat_error"
        )
    return {
        "schema_version": STAGE3_PROCESS_EVIDENCE_SCHEMA,
        "calibration_status": "process_safety_contract_not_industry_standard",
        "status": "ready" if linear_confirmed and sample_ready else "review_required",
        "linear_input": {
            "confirmed": linear_confirmed,
            "state": state,
            "confidence": profile.get("confidence"),
            "source": profile.get("source"),
        },
        "true_sky_support": {
            "supported": sky_supported,
            "safe_samples_ready": sample_ready,
            "heldout_validation_ready": validation_ready,
            "usable_sky_fraction": usable_sky_fraction,
            "usable_sky_grid_cells": sky_grid_cells,
        },
        "spatial_models": spatial,
        "mechanism": mechanism,
        "correction_mode": correction_mode,
        "spatial_variation_supported": spatial_supported,
        "directional_pattern_detected": pattern_detected,
        "radial_shape_supported": radial_shape_supported,
        "advisory_reasons": advisory_reasons,
        "low_complexity_required": low_complexity_required,
        "model_complexity_limit": (
            "polynomial_degree_1"
            if low_complexity_required
            else "polynomial_then_validated_rbf"
        ),
        "should_evaluate": should_evaluate,
        "hard_block_reasons": hard_blocks,
        "evidence_basis": [
            "linear-input provenance",
            "explicit source and coverage masks",
            "spatially distributed fit and held-out sky samples",
            "three-sigma patch-median spatial evidence",
            "low-order directional versus radial model comparison",
        ],
    }


def _stage3_gate_result(
    profile: str,
    warnings: List[str],
    hard_issues: List[str],
    **payload: Any,
) -> Tuple[bool, Dict[str, Any]]:
    """Build a compatible normal/soft-warning/hard-rejection gate report."""
    normalized = normalize_stage3_gate_profile(profile)
    warnings = list(dict.fromkeys(str(item) for item in warnings if item))
    hard_issues = list(dict.fromkeys(str(item) for item in hard_issues if item))
    accepted = not hard_issues
    severity = (
        "hard_rejected"
        if hard_issues
        else ("soft_warning" if warnings else "normal")
    )
    report = {
        "status": (
            "rejected"
            if hard_issues
            else ("accepted_with_warnings" if warnings else "accepted")
        ),
        "accepted": accepted,
        "severity": severity,
        "warnings": warnings,
        "hard_issues": hard_issues,
        "issues": hard_issues + warnings,
        "profile": normalized,
        "effective_thresholds": stage3_gate_thresholds(normalized),
        **payload,
    }
    return accepted, report


def assess_single_background_validation(
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
    *,
    gate_profile: str = "output_first",
) -> Tuple[bool, Dict[str, Any]]:
    """Classify held-out background evidence using the selected gate profile."""
    profile = normalize_stage3_gate_profile(gate_profile)
    thresholds = stage3_gate_thresholds(profile)
    strict = bool(thresholds.get("strict_legacy"))
    unavailable = "held-out background/RMS metrics are unavailable"
    if baseline.get("status") != "ready" or candidate.get("status") != "ready":
        return _stage3_gate_result(
            profile,
            [] if strict else [unavailable],
            [unavailable] if strict else [],
            baseline=baseline or {},
            candidate=candidate or {},
        )
    try:
        baseline_span = float(baseline["robust_span"])
        candidate_span = float(candidate["robust_span"])
        baseline_noise = float(baseline["patch_mad_median"])
        candidate_noise = float(candidate["patch_mad_median"])
    except (KeyError, TypeError, ValueError):
        incomplete = "held-out background/RMS metrics are incomplete"
        return _stage3_gate_result(
            profile,
            [] if strict else [incomplete],
            [incomplete] if strict else [],
        )
    if not all(
        math.isfinite(value)
        for value in (
            baseline_span,
            candidate_span,
            baseline_noise,
            candidate_noise,
        )
    ):
        nonfinite = "held-out background/RMS metrics contain non-finite values"
        return _stage3_gate_result(
            profile,
            [] if strict else [nonfinite],
            [nonfinite] if strict else [],
        )
    baseline_span_se = background_span_standard_error(baseline)
    candidate_span_se = background_span_standard_error(candidate)
    three_sigma = max(
        STAGE3_SIGNIFICANCE_SIGMA
        * math.hypot(baseline_span_se, candidate_span_se),
        1e-12,
    )
    span_improvement = baseline_span - candidate_span
    span_not_worse = bool(candidate_span <= baseline_span + three_sigma)
    material_improvement = bool(span_improvement > three_sigma)
    rms_not_worse = bool(candidate_noise <= baseline_noise + three_sigma)
    warnings: List[str] = []
    if not span_not_worse:
        warnings.append("held-out spatial background span worsened")
    if not material_improvement:
        warnings.append("held-out span improvement is below sampling uncertainty")
    if not rms_not_worse:
        warnings.append("held-out background RMS increased beyond sampling uncertainty")

    hard_issues: List[str] = []
    if strict:
        hard_issues.extend(warnings)
        warnings = []
    else:
        sigma_unit = max(three_sigma / STAGE3_SIGNIFICANCE_SIGMA, 1e-12)
        hard_sigma = float(thresholds["hard_significance_sigma"])
        span_worsening = max(0.0, candidate_span - baseline_span)
        rms_worsening = max(0.0, candidate_noise - baseline_noise)
        span_ratio = candidate_span / max(baseline_span, 1e-12)
        rms_ratio = candidate_noise / max(baseline_noise, 1e-12)
        if (
            span_ratio > float(thresholds["span_hard_ratio"])
            and span_worsening / sigma_unit > hard_sigma
        ):
            hard_issues.append(
                "held-out spatial background span is excessively worse "
                f"({span_ratio:.2f}x, {span_worsening / sigma_unit:.2f} sigma)"
            )
        if (
            rms_ratio > float(thresholds["rms_hard_ratio"])
            and rms_worsening / sigma_unit > hard_sigma
        ):
            hard_issues.append(
                "held-out background RMS is excessively worse "
                f"({rms_ratio:.2f}x, {rms_worsening / sigma_unit:.2f} sigma)"
            )

    accepted, report = _stage3_gate_result(
        profile,
        warnings,
        hard_issues,
        baseline_span=baseline_span,
        candidate_span=candidate_span,
        span_improvement=span_improvement,
        sampling_uncertainty_3sigma=three_sigma,
        uncertainty_method=(
            "correlation_aware_span_difference"
            if "patch_median_uncertainty" in baseline
            and "patch_median_uncertainty" in candidate
            else "legacy_pixel_independence_fallback"
        ),
        material_improvement=material_improvement,
        span_not_worse=span_not_worse,
        baseline_background_rms=baseline_noise,
        candidate_background_rms=candidate_noise,
        background_rms_not_worse=rms_not_worse,
        baseline_median=baseline.get("median"),
        candidate_median=candidate.get("median"),
    )
    return accepted, report


def assess_target_fidelity(
    preservation: Dict[str, Any],
    *,
    low_complexity_required: bool,
    gate_profile: str = "output_first",
) -> Tuple[bool, Dict[str, Any]]:
    """Classify target-fidelity changes as normal, warning, or hard rejection.

    The comparison is meaningful only when target flux is referenced to a
    degree-1 sky plane fitted on the held-out samples.  A three-sigma decision
    boundary is applied to measured uncertainty rather than to a product-wide
    flux-retention percentage.
    """
    profile = normalize_stage3_gate_profile(gate_profile)
    thresholds = stage3_gate_thresholds(profile)
    strict = bool(thresholds.get("strict_legacy"))
    report = preservation or {}
    warnings: List[str] = []
    hard_issues: List[str] = []
    if not report.get("available"):
        (hard_issues if strict else warnings).append(
            "target fidelity metrics are unavailable"
        )
    if report.get("target_sky_reference") != "heldout_sky_plane_degree_1":
        (hard_issues if strict else warnings).append(
            "target flux lacks an independent held-out sky reference"
        )

    try:
        flux_significance = float(report["target_flux_change_significance"])
    except (KeyError, TypeError, ValueError):
        flux_significance = math.nan
    if not math.isfinite(flux_significance):
        (hard_issues if strict else warnings).append(
            "target flux-change significance is unavailable"
        )

    try:
        flux_retention = float(report["target_flux_retention_ratio"])
    except (KeyError, TypeError, ValueError):
        flux_retention = math.nan
    if not math.isfinite(flux_retention):
        (hard_issues if strict else warnings).append(
            "target flux-retention ratio is unavailable"
        )
    if math.isfinite(flux_significance):
        if flux_significance < -STAGE3_SIGNIFICANCE_SIGMA:
            message = (
                "background-referenced target flux loss exceeds three-sigma "
                f"uncertainty ({flux_significance:.2f} sigma)"
            )
            (hard_issues if strict else warnings).append(message)
        elif flux_significance > STAGE3_SIGNIFICANCE_SIGMA and not strict:
            warnings.append(
                "background-referenced target flux growth exceeds three-sigma "
                f"uncertainty (+{flux_significance:.2f} sigma)"
            )
    if not strict and math.isfinite(flux_retention):
        if not (
            float(thresholds["flux_retention_soft_min"])
            <= flux_retention
            <= float(thresholds["flux_retention_soft_max"])
        ):
            warnings.append(
                f"target flux retention is outside the preferred range ({flux_retention:.3f})"
            )
        hard_sigma = float(thresholds["hard_significance_sigma"])
        if (
            flux_retention < float(thresholds["flux_retention_hard_min"])
            and math.isfinite(flux_significance)
            and flux_significance < -hard_sigma
        ):
            hard_issues.append(
                f"target flux loss is excessive ({flux_retention:.3f}, {flux_significance:.2f} sigma)"
            )
        if (
            flux_retention > float(thresholds["flux_retention_hard_max"])
            and math.isfinite(flux_significance)
            and flux_significance > hard_sigma
        ):
            hard_issues.append(
                f"target flux growth is excessive ({flux_retention:.3f}, +{flux_significance:.2f} sigma)"
            )

    try:
        morphology = float(report["target_morphology_correlation"])
    except (KeyError, TypeError, ValueError):
        morphology = math.nan
        (hard_issues if strict else warnings).append(
            "target morphology correlation is unavailable"
        )
    if not math.isfinite(morphology):
        (hard_issues if strict else warnings).append(
            "target morphology correlation is non-finite"
        )
    elif not strict:
        if morphology < float(thresholds["morphology_soft_min"]):
            warnings.append(
                f"target morphology correlation is low ({morphology:.5f})"
            )
        if morphology < float(thresholds["morphology_hard_min"]):
            hard_issues.append(
                f"target morphology correlation is excessively low ({morphology:.5f})"
            )

    try:
        structure_significance = float(
            report["target_change_residual_significance"]
        )
    except (KeyError, TypeError, ValueError):
        structure_significance = math.nan
    if low_complexity_required and not math.isfinite(structure_significance):
        (hard_issues if strict else warnings).append(
            "target change-structure significance is unavailable"
        )
    if (
        low_complexity_required
        and math.isfinite(structure_significance)
        and structure_significance > STAGE3_SIGNIFICANCE_SIGMA
    ):
        message = (
            "candidate introduced target-region structure beyond the authorized "
            f"degree-1 model ({structure_significance:.2f} sigma)"
        )
        (hard_issues if strict else warnings).append(message)
    if (
        not strict
        and low_complexity_required
        and math.isfinite(structure_significance)
        and structure_significance > float(thresholds["structure_hard_sigma"])
    ):
        hard_issues.append(
            "candidate introduced excessive target-region structure "
            f"({structure_significance:.2f} sigma)"
        )

    try:
        centroid_shift = float(report["target_centroid_shift_fraction"])
    except (KeyError, TypeError, ValueError):
        centroid_shift = math.nan
    if not strict and not math.isfinite(centroid_shift):
        warnings.append("target centroid shift is unavailable")
    if not strict and math.isfinite(centroid_shift):
        if centroid_shift > float(thresholds["centroid_soft_max"]):
            warnings.append(
                f"target centroid shift is elevated ({centroid_shift:.5f})"
            )
        if centroid_shift > float(thresholds["centroid_hard_max"]):
            hard_issues.append(
                f"target centroid shift is excessive ({centroid_shift:.5f})"
            )

    accepted, gate = _stage3_gate_result(
        profile,
        warnings,
        hard_issues,
        uncertainty_basis="held-out_sky_plane_three_sigma",
        low_complexity_required=bool(low_complexity_required),
        target_flux_retention_ratio=(
            flux_retention if math.isfinite(flux_retention) else None
        ),
        target_flux_change_significance=(
            flux_significance if math.isfinite(flux_significance) else None
        ),
        target_morphology_correlation=(
            morphology if math.isfinite(morphology) else None
        ),
        target_change_residual_significance=(
            structure_significance
            if math.isfinite(structure_significance)
            else None
        ),
        target_centroid_shift_fraction=(
            centroid_shift if math.isfinite(centroid_shift) else None
        ),
    )
    return accepted, gate


def assess_compound_background_validation(
    baseline: Dict[str, Any],
    best_single: Dict[str, Any],
    polynomial: Dict[str, Any],
    compound: Dict[str, Any],
    *,
    improvement_min: float = 0.10,
    zero_point_abs_max: float = 0.01,
    zero_point_rel_max: float = 0.15,
    gate_profile: str = "output_first",
) -> Tuple[bool, Dict[str, Any]]:
    """Classify compound-model evidence without hiding output-first warnings."""
    profile = normalize_stage3_gate_profile(gate_profile)
    strict = bool(stage3_gate_thresholds(profile).get("strict_legacy"))
    reports = {
        "baseline": baseline or {},
        "best_single": best_single or {},
        "polynomial": polynomial or {},
        "compound": compound or {},
    }
    unavailable = [
        name
        for name, report in reports.items()
        if report.get("status") != "ready"
    ]
    if unavailable:
        message = "validation metrics unavailable: " + ", ".join(unavailable)
        return _stage3_gate_result(
            profile,
            [] if strict else [message],
            [message] if strict else [],
            measurements=reports,
        )
    counts = {
        int(report.get("sample_count", 0) or 0)
        for report in reports.values()
    }
    if (
        len(counts) != 1
        or not counts
        or min(counts) < STAGE3_MIN_VALIDATION_PATCHES
    ):
        message = "validation sample counts are inconsistent"
        return _stage3_gate_result(
            profile,
            [] if strict else [message],
            [message] if strict else [],
            measurements=reports,
        )
    try:
        baseline_span = float(baseline["robust_span"])
        single_span = float(best_single["robust_span"])
        compound_span = float(compound["robust_span"])
        polynomial_median = float(polynomial["median"])
        compound_median = float(compound["median"])
    except (KeyError, TypeError, ValueError):
        message = "validation metrics are incomplete"
        return _stage3_gate_result(
            profile,
            [] if strict else [message],
            [message] if strict else [],
            measurements=reports,
        )
    numeric_values = (
        baseline_span,
        single_span,
        compound_span,
        polynomial_median,
        compound_median,
    )
    if not all(math.isfinite(value) for value in numeric_values):
        message = "validation metrics contain non-finite values"
        return _stage3_gate_result(
            profile,
            [] if strict else [message],
            [message] if strict else [],
            measurements=reports,
        )
    if single_span <= 1e-10:
        message = "best single-stage candidate has no measurable residual span"
        return _stage3_gate_result(
            profile,
            [] if strict else [message],
            [message] if strict else [],
            measurements=reports,
        )

    improvement_min = _clamp(improvement_min, 0.0, 0.80)
    improvement = (single_span - compound_span) / single_span
    baseline_tolerance = max(1e-10, baseline_span * 1e-6)
    baseline_not_worse = compound_span <= baseline_span + baseline_tolerance
    zero_point_drift = abs(compound_median - polynomial_median)
    zero_point_limit = max(
        max(0.0, float(zero_point_abs_max)),
        abs(polynomial_median) * _clamp(zero_point_rel_max, 0.0, 1.0),
    )
    issues: List[str] = []
    if improvement + 1e-12 < improvement_min:
        issues.append(
            f"held-out span improvement {improvement:.3f}<{improvement_min:.3f}"
        )
    if not baseline_not_worse:
        issues.append(
            "compound held-out span is worse than the immutable baseline"
        )
    if zero_point_drift > zero_point_limit + 1e-12:
        issues.append(
            "held-out zero-point drift "
            f"{zero_point_drift:.6f}>{zero_point_limit:.6f}"
        )
    _single_ok, single_gate = assess_single_background_validation(
        baseline,
        compound,
        gate_profile=profile,
    )
    warnings = list(issues) + list(single_gate.get("warnings") or [])
    hard_issues = list(single_gate.get("hard_issues") or [])
    if strict:
        hard_issues.extend(warnings)
        warnings = []
    return _stage3_gate_result(
        profile,
        warnings,
        hard_issues,
        span_improvement_ratio=improvement,
        span_improvement_min=improvement_min,
        baseline_not_worse=baseline_not_worse,
        zero_point_drift=zero_point_drift,
        zero_point_limit=zero_point_limit,
        zero_point_abs_max=float(zero_point_abs_max),
        zero_point_rel_max=float(zero_point_rel_max),
        single_candidate_gate=single_gate,
        measurements=reports,
    )


def select_background_route(
    adaptive: Dict[str, Any],
    pattern_report: Dict[str, Any],
    *,
    process_report: Optional[Dict[str, Any]] = None,
    gradient_apply_min: float = 0.08,
    dirty_apply_min: float = 0.18,
) -> Dict[str, Any]:
    """Separate low-frequency sky gradients from directional pattern noise."""
    metrics = adaptive or {}
    try:
        gradient = float(metrics.get("gradient_score", 0.0) or 0.0)
        dirty = float(metrics.get("dirty_background_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        gradient = 0.0
        dirty = 0.0
    detected = bool((pattern_report or {}).get("detected", False))
    pattern_kind = str((pattern_report or {}).get("kind") or "none")
    if isinstance(process_report, dict) and process_report:
        gradient_supported = bool(
            process_report.get("spatial_variation_supported", False)
            and process_report.get("correction_mode") in {
                "subtract",
                "subtract_with_master_flat_advisory",
            }
        )
        evidence_basis = "source_masked_heldout_spatial_process"
    else:
        # Compatibility path for callers that have not yet supplied the Stage 3
        # process report. Production Stage 3 always supplies it.
        gradient_supported = bool(
            gradient >= max(0.01, float(gradient_apply_min))
            and dirty >= max(0.01, float(dirty_apply_min))
        )
        evidence_basis = "legacy_adaptive_diagnostics"
    if detected and gradient_supported:
        route = "mixed_gradient_and_pattern_noise"
        reason = (
            "low-frequency gradient is supported, but directional pattern noise "
            "must remain a separate review signal"
        )
    elif detected:
        route = "pattern_noise_deferred"
        reason = (
            "stripe/walking noise is not a sky-gradient model; preserve pixels "
            "and defer correction to acquisition/stacking review"
        )
    else:
        route = "low_frequency_gradient"
        reason = "no directional-pattern diversion is required"
    if pattern_kind == "diagonal_walking_noise":
        pattern_branch = "walking_noise_review"
        recommended_actions = [
            "re-stack with registration rejection and adequate acquisition dithering",
            "inspect calibrated subframes for correlated diagonal drift",
            "review a dedicated walking-noise model against an unchanged baseline",
        ]
        correction_owner = "acquisition_or_stacking_review"
    elif pattern_kind in {"horizontal_banding", "vertical_banding"}:
        pattern_branch = "banding_review"
        recommended_actions = [
            "inspect bias/dark calibration and sensor readout consistency",
            "confirm the same stripe orientation in separated background regions",
            "review dedicated banding reduction against an unchanged baseline",
        ]
        correction_owner = "calibration_or_sensor_review"
    elif detected:
        pattern_branch = "directional_pattern_review"
        recommended_actions = [
            "inspect calibrated subframes and stacking residuals",
            "review a dedicated artifact correction against an unchanged baseline",
        ]
        correction_owner = "acquisition_or_stacking_review"
    else:
        pattern_branch = "none"
        recommended_actions = []
        correction_owner = "stage3_background_extraction"
    return {
        "route": route,
        "reason": reason,
        "pattern_detected": detected,
        "pattern_kind": pattern_kind,
        "pattern_branch": pattern_branch,
        "gradient_supported": gradient_supported,
        "gradient_evidence_basis": evidence_basis,
        "gradient_score": gradient,
        "dirty_background_score": dirty,
        "subsky_existing_allowed": route != "pattern_noise_deferred",
        "requires_review": detected,
        "correction_owner": correction_owner,
        "recommended_actions": recommended_actions,
    }


def pattern_candidate_gate(
    before: Dict[str, Any],
    after: Dict[str, Any],
    *,
    growth_max: float = 0.12,
    gate_profile: str = "output_first",
) -> Tuple[bool, Dict[str, Any]]:
    """Classify introduced/amplified directional pattern noise."""
    profile = normalize_stage3_gate_profile(gate_profile)
    thresholds = stage3_gate_thresholds(profile)
    strict = bool(thresholds.get("strict_legacy"))
    if before.get("status") != "ok" or after.get("status") != "ok":
        if strict:
            return True, {
                "status": "not_available",
                "accepted": True,
                "severity": "normal",
                "warnings": [],
                "hard_issues": [],
                "issues": [],
                "profile": profile,
                "effective_thresholds": thresholds,
                "reason": "directional pattern metrics unavailable",
            }
        return _stage3_gate_result(
            profile,
            ["directional pattern metrics are unavailable"],
            [],
            reason="directional pattern metrics unavailable",
        )
    before_score = float(before.get("pattern_score", 0.0) or 0.0)
    after_score = float(after.get("pattern_score", 0.0) or 0.0)
    growth = after_score - before_score
    threshold = float(
        (after.get("thresholds") or {}).get("pattern_score_min", 0.55)
        or 0.55
    )
    growth_max = _clamp(growth_max, 0.02, 0.40)
    introduced = bool(after.get("detected")) and not bool(before.get("detected"))
    worsened = bool(growth > growth_max and after_score >= threshold)
    warning_message = "candidate introduced or materially worsened directional pattern noise"
    warnings = [warning_message] if introduced or worsened else []
    hard_issues: List[str] = []
    if strict:
        hard_issues = warnings
        warnings = []
    elif (
        after_score > float(thresholds["pattern_hard_score"])
        and growth > float(thresholds["pattern_hard_growth"])
    ):
        hard_issues.append(
            "candidate introduced excessive directional pattern noise "
            f"(score={after_score:.3f}, growth={growth:.3f})"
        )
    return _stage3_gate_result(
        profile,
        warnings,
        hard_issues,
        pattern_score_before=before_score,
        pattern_score_after=after_score,
        pattern_score_growth=growth,
        growth_max=growth_max,
        introduced_pattern_noise=introduced,
        worsened_pattern_noise=worsened,
        after_kind=after.get("kind"),
    )


__all__ = [
    "STAGE3_BACKGROUND_VALIDATION_SCHEMA",
    "STAGE3_PROCESS_EVIDENCE_SCHEMA",
    "assess_background_process",
    "assess_compound_background_validation",
    "assess_single_background_validation",
    "assess_target_fidelity",
    "analyze_directional_pattern_noise",
    "build_safe_background_samples",
    "background_span_standard_error",
    "measure_background_validation",
    "pattern_candidate_gate",
    "select_background_route",
    "split_background_sample_points",
]
