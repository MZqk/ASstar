"""Target-local diagnostics for Stage 7 starless stretch candidates."""
from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional

import numpy as np
import stage7_quality
import ui_preview

from image_metrics import (
    _box_blur_gray,
    _component_areas,
    _to_rgb_float_fullres,
    _to_rgb_float_image,
)
from stage7_pixel_domain import canonicalize_stage7_pixels_01


CORE_PROTECT_TARGETS = {
    "bright_emission_reflection_nebula",
    "large_galaxy",
    "small_galaxy",
}
FAINT_SIGNAL_TARGETS = {
    "bright_emission_reflection_nebula",
    "emission_nebula_widefield",
    "large_galaxy",
    "small_galaxy",
    "dark_nebula_low_contrast",
}
STATISTICAL_MTF_REFERENCE_SOURCE = (
    "resources/siril_plugins/vendor/siril-scripts/processing/"
    "Statistical_Stretch.py"
)
LOWER_HALF_RECENTERED_MAD_SCALE = 1.4826
# The usual 1.4826 multiplier is almost exactly Gaussian-consistent for the
# full MAD and for a lower-side deviation measured around the global median.
FULL_MAD_NORMAL_SIGMA_RATIO = 0.9999985036407106
# For an ideal normal distribution, recentering the x <= median half before
# applying the usual MAD scale estimates only this fraction of the true sigma.
LOWER_HALF_RECENTERED_MAD_NORMAL_SIGMA_RATIO = 0.5916931999771552
STRETCH_SEMANTICS_SCHEMA = "starun.siril-stretch-semantics.v1"
TRANSFORM_LOSS_SCHEMA = "starun.stage7-transform-loss.v1"
MULTISCALE_CONTRAST_SCHEMA = "starun.stage7-multiscale-contrast.v1"
RENDITION_METRICS_SCHEMA = "starun.stage7-rendition-metrics.v1"
RENDITION_CHROMA_SCHEMA = "starun.stage7-subject-chroma.v1"
MULTISCALE_CONTRAST_RADII = (1, 2, 4, 8, 16)
SIRIL_MINIMUM_VERSION_CONTRACT = "1.4.0"
SIRIL_BUNDLED_REFERENCE_VERSION = "1.4.4"
TRANSFORM_ZERO_EPSILON = 1e-7
TRANSFORM_NEAR_BLACK = 0.010
TRANSFORM_NEAR_HIGHLIGHT = 0.995
CONDITIONAL_STRETCH_SCHEMA = "starun.stage7-conditional-stretch.v1"
CONDITIONAL_SOURCE_PROFILE_SCHEMA = "starun.stage7-stretch-source-profile.v1"
CONDITIONAL_STRETCH_LUT_SIZE = 65536
CONDITIONAL_STRETCH_OUTPUT_HEADROOM = 0.995
CONDITIONAL_STRETCH_REFERENCE = (
    "https://github.com/MZqk/starun/blob/"
    "c82773038fcb1aef1699efb6a677537ba7f79d25/"
    "deep-sky-processor/scripts/adaptive_stretch.py"
)
CONDITIONAL_GHS_REFERENCE = (
    "https://github.com/MZqk/starun/blob/"
    "c82773038fcb1aef1699efb6a677537ba7f79d25/"
    "deep-sky-processor/scripts/stretch.py"
)
CONDITIONAL_STRETCH_SOURCE_LICENSE = "GPL-3.0-only"
DISPLAY90_STRETCH_SCHEMA = "starun.stage7-display90-calibration.v1"
STAGE7_MATCHED_DOMAIN_TRANSFER_SCHEMA = (
    "starun.stage7-matched-domain-transfer.v1"
)
DISPLAY90_STRENGTH_MIN = 0.50
DISPLAY90_STRENGTH_MAX = 0.95
DISPLAY90_ANALYSIS_MAX_SIDE = ui_preview.DEFAULT_PREVIEW_MAX_SIDE
DISPLAY90_REPORT_PERCENTILES = (0.2, 1.0, 10.0, 50.0, 90.0, 99.0, 99.8)
DISPLAY90_CONFORMANCE_PERCENTILES = (50.0, 90.0, 99.0)


def _stage7_mask(
    masks: Optional[Dict[str, Any]],
    name: str,
    shape: tuple[int, int],
) -> Optional[np.ndarray]:
    """Return one finite full-resolution mask without rebuilding it per candidate."""

    if not isinstance(masks, dict) or masks.get(name) is None:
        return None
    mask = np.asarray(masks[name], dtype=np.float32)
    if mask.ndim != 2 or tuple(mask.shape) != tuple(shape):
        return None
    if not np.all(np.isfinite(mask)):
        return None
    return np.clip(mask, 0.0, 1.0)


def _stage7_subject_weight(
    masks: Optional[Dict[str, Any]],
    shape: tuple[int, int],
) -> np.ndarray:
    """Build the stable subject ROI from masks frozen on the Stage 6 source."""

    explicit = _stage7_mask(masks, "subject_mask", shape)
    if explicit is not None:
        return explicit
    layers = [
        mask
        for name in (
            "core_mask",
            "nebula_mask",
            "faint_nebula_mask",
            "galaxy_signal_mask",
            "star_mask",
        )
        if (mask := _stage7_mask(masks, name, shape)) is not None
    ]
    if layers:
        return np.maximum.reduce(layers).astype(np.float32, copy=False)
    background = _stage7_mask(masks, "background_mask", shape)
    if background is not None:
        return (1.0 - background).astype(np.float32, copy=False)
    return np.ones(shape, dtype=np.float32)


def measure_frozen_rendition_metrics(
    image: np.ndarray,
    masks: Optional[Dict[str, Any]],
    *,
    max_samples: int = 300_000,
) -> Dict[str, Any]:
    """Measure presentation quality on Stage 6-derived, candidate-invariant ROIs."""

    try:
        rgb = _stage7_rgb_float_fullres(np.asarray(image))
        if not np.all(np.isfinite(rgb)):
            raise ValueError("rendition image contains non-finite pixels")
        shape = tuple(int(value) for value in rgb.shape[1:])
        subject_weight = _stage7_subject_weight(masks, shape)
        background_weight = _stage7_mask(masks, "background_mask", shape)
        subject = subject_weight > 0.25
        if int(np.count_nonzero(subject)) < 64:
            raise ValueError("frozen subject ROI contains too few pixels")
        background = (
            background_weight > 0.50
            if background_weight is not None
            else ~subject
        )
        background &= ~subject
        if int(np.count_nonzero(background)) < 64:
            raise ValueError("frozen background ROI contains too few pixels")

        luminance = (
            0.2126 * rgb[0]
            + 0.7152 * rgb[1]
            + 0.0722 * rgb[2]
        ).astype(np.float32)
        peak = np.max(rgb, axis=0)
        trough = np.min(rgb, axis=0)
        saturation = np.divide(
            peak - trough,
            np.maximum(peak, 1e-6),
            out=np.zeros_like(peak, dtype=np.float32),
            where=peak > 1e-6,
        )
        local_detail = np.abs(luminance - _box_blur_gray(luminance))

        def sampled(values: np.ndarray, region: np.ndarray) -> np.ndarray:
            selected = np.asarray(values[region], dtype=np.float32)
            if selected.size > max_samples:
                stride = int(np.ceil(selected.size / float(max_samples)))
                selected = selected[::stride]
            return selected

        subject_luma = sampled(luminance, subject)
        subject_saturation = sampled(saturation, subject)
        subject_detail = sampled(local_detail, subject)
        background_luma = sampled(luminance, background)
        bg_median = float(np.median(background_luma))
        bg_mad = float(np.median(np.abs(background_luma - bg_median)))
        subject_p50, subject_p75, subject_p99 = np.percentile(
            subject_luma,
            [50.0, 75.0, 99.0],
        )
        noise_sigma = max(1.4826 * bg_mad, 1e-4)
        return {
            "schema": RENDITION_METRICS_SCHEMA,
            "status": "available",
            "mask_source": "stage6_frozen_roi",
            "subject_coverage": float(np.mean(subject)),
            "background_coverage": float(np.mean(background)),
            "metrics": {
                "visibility": float(max(0.0, subject_p75 - bg_median) / noise_sigma),
                "subject_span": float(max(0.0, subject_p99 - subject_p50)),
                "saturation_median": float(np.median(subject_saturation)),
                "saturation_p95": float(np.percentile(subject_saturation, 95.0)),
                "microcontrast": float(np.percentile(subject_detail, 75.0)),
                "subject_p50": float(subject_p50),
                "subject_p99": float(subject_p99),
                "background_median": bg_median,
                "background_mad": bg_mad,
            },
        }
    except (IndexError, TypeError, ValueError, FloatingPointError) as error:
        return {
            "schema": RENDITION_METRICS_SCHEMA,
            "status": "unavailable",
            "mask_source": "stage6_frozen_roi",
            "reason": str(error),
            "metrics": {},
        }


def rendition_metric_retention(
    candidate: Dict[str, Any],
    preview: Dict[str, Any],
) -> Dict[str, Any]:
    """Return stable candidate/preview ratios used by the Stage 7 selector."""

    candidate_metrics = dict(candidate.get("metrics") or {})
    preview_metrics = dict(preview.get("metrics") or {})
    ratios: Dict[str, Any] = {}
    for name in (
        "visibility",
        "subject_span",
        "saturation_median",
        "saturation_p95",
        "microcontrast",
    ):
        try:
            value = float(candidate_metrics[name])
            reference = float(preview_metrics[name])
        except (KeyError, TypeError, ValueError):
            ratios[name] = {"available": False, "ratio": None}
            continue
        available = bool(
            np.isfinite(value)
            and np.isfinite(reference)
            and reference > 1e-9
        )
        ratios[name] = {
            "available": available,
            "candidate": value,
            "preview": reference,
            "ratio": float(value / reference) if available else None,
        }
    return {
        "schema": RENDITION_METRICS_SCHEMA,
        "status": (
            "available"
            if any(item.get("available") for item in ratios.values())
            else "unavailable"
        ),
        "metrics": ratios,
    }


def apply_subject_chroma_rendition(
    image: np.ndarray,
    masks: Optional[Dict[str, Any]],
    *,
    factor: float,
    output_headroom: float = 0.995,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Boost only broad subject chroma while preserving luminance and headroom."""

    rgb = _stage7_rgb_float_fullres(np.asarray(image))
    if not np.all(np.isfinite(rgb)):
        raise ValueError("subject chroma source contains non-finite pixels")
    shape = tuple(int(value) for value in rgb.shape[1:])
    factor = float(np.clip(float(factor), 1.0, 1.20))
    headroom = float(np.clip(float(output_headroom), 0.95, 0.999))
    has_frozen_roi = any(
        _stage7_mask(masks, name, shape) is not None
        for name in (
            "subject_mask",
            "background_mask",
            "core_mask",
            "nebula_mask",
            "faint_nebula_mask",
            "galaxy_signal_mask",
            "star_mask",
        )
    )
    if not has_frozen_roi:
        raise ValueError("vivid-safe chroma requires a valid frozen Stage 6 ROI")
    subject_weight = _stage7_subject_weight(masks, shape)
    core_weight = _stage7_mask(masks, "core_mask", shape)
    if core_weight is not None:
        subject_weight *= 1.0 - 0.90 * core_weight
    star_weight = _stage7_mask(masks, "star_mask", shape)
    if star_weight is not None:
        subject_weight *= 1.0 - 0.90 * star_weight
    source_peak = np.max(rgb, axis=0)
    source_floor = np.min(rgb, axis=0)
    range_weight = np.minimum(
        np.clip((headroom - source_peak) / 0.04, 0.0, 1.0),
        np.clip(source_floor / 0.02, 0.0, 1.0),
    )
    boost_weight = np.clip(subject_weight * range_weight, 0.0, 1.0)
    luminance = (
        0.2126 * rgb[0]
        + 0.7152 * rgb[1]
        + 0.0722 * rgb[2]
    ).astype(np.float32)
    chroma = rgb - luminance[None, :, :]
    broad_chroma = np.stack(
        [_box_blur_gray(_box_blur_gray(channel)) for channel in chroma],
        axis=0,
    ).astype(np.float32)
    delta = (factor - 1.0) * boost_weight[None, :, :] * broad_chroma
    delta_luma = (
        0.2126 * delta[0]
        + 0.7152 * delta[1]
        + 0.0722 * delta[2]
    )
    delta -= delta_luma[None, :, :]

    safe_scale = np.ones(shape, dtype=np.float32)
    for channel in range(3):
        positive = delta[channel] > 0.0
        negative = delta[channel] < 0.0
        channel_scale = np.ones(shape, dtype=np.float32)
        channel_scale[positive] = np.maximum(
            0.0,
            (headroom - rgb[channel][positive])
            / np.maximum(delta[channel][positive], 1e-12),
        )
        channel_scale[negative] = np.maximum(
            0.0,
            rgb[channel][negative]
            / np.maximum(-delta[channel][negative], 1e-12),
        )
        safe_scale = np.minimum(safe_scale, channel_scale)
    safe_scale = np.clip(safe_scale, 0.0, 1.0)
    rendered = rgb + delta * safe_scale[None, :, :]
    rendered_luma = (
        0.2126 * rendered[0]
        + 0.7152 * rendered[1]
        + 0.0722 * rendered[2]
    )
    rendered_peak = np.max(rendered, axis=0)
    headroom_limited = (
        (rendered_peak >= headroom - 1e-7)
        & (source_peak < headroom - 1e-7)
    )
    newly_clipped = (
        (rendered_peak >= 1.0 - TRANSFORM_ZERO_EPSILON)
        & (source_peak < 1.0 - TRANSFORM_ZERO_EPSILON)
    )
    return np.asarray(np.clip(rendered, 0.0, 1.0), dtype=np.float32), {
        "schema": RENDITION_CHROMA_SCHEMA,
        "mode": "frozen_subject_broad_chroma",
        "factor": factor,
        "output_headroom": headroom,
        "subject_coverage": float(np.mean(subject_weight > 0.25)),
        "boosted_coverage": float(np.mean(boost_weight > 0.05)),
        "core_protection_applied": core_weight is not None,
        "star_protection_applied": star_weight is not None,
        "mean_effective_scale": float(np.mean(safe_scale[boost_weight > 0.05]))
        if np.any(boost_weight > 0.05)
        else 0.0,
        "max_luminance_error": float(np.max(np.abs(rendered_luma - luminance))),
        "newly_clipped_ratio": float(np.mean(newly_clipped)),
        "headroom_limited_ratio": float(np.mean(headroom_limited)),
        "background_unchanged": bool(
            np.allclose(
                rendered[:, subject_weight <= 1e-6],
                rgb[:, subject_weight <= 1e-6],
                atol=2e-7,
            )
        ),
    }


def _stage7_rgb_float_fullres(image: np.ndarray) -> np.ndarray:
    pixels, _provenance = canonicalize_stage7_pixels_01(image)
    return _to_rgb_float_fullres(pixels)


def _stage7_rgb_float_image(
    image: np.ndarray,
    *,
    max_side: int = 1024,
) -> np.ndarray:
    pixels, _provenance = canonicalize_stage7_pixels_01(image)
    return _to_rgb_float_image(pixels, max_side=max_side)


def _stage7_rgb_analysis_grid(
    image: np.ndarray,
    *,
    max_side: int,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Validate the full domain, then canonicalize only a spatial grid."""
    source = np.asarray(image)
    if source.size == 0:
        raise ValueError("empty image data")
    extrema = np.asarray([np.min(source), np.max(source)], dtype=source.dtype)
    canonicalize_stage7_pixels_01(extrema)

    layout = ""
    if source.ndim == 2:
        height, width = source.shape
        layout = "mono_2d"
    elif source.ndim == 3:
        if source.shape[0] == 1 and source.shape[-1] not in (1, 3):
            height, width = source.shape[1:]
            layout = "chw_mono"
        elif source.shape[0] == 3 and source.shape[-1] not in (1, 3):
            height, width = source.shape[1:]
            layout = "chw_rgb"
        elif source.shape[-1] == 1:
            height, width = source.shape[:2]
            layout = "hwc_mono"
        elif source.shape[-1] >= 3:
            height, width = source.shape[:2]
            layout = "hwc_rgb"
        elif source.shape[0] >= 3:
            height, width = source.shape[1:]
            layout = "chw_rgb"
        else:
            raise ValueError(f"unsupported image shape: {source.shape}")
    else:
        raise ValueError(f"unsupported image ndim: {source.ndim}")

    bounded_max_side = max(1, int(max_side))
    stride = max(1, int(np.ceil(max(height, width) / float(bounded_max_side))))
    if layout == "mono_2d":
        sampled = source[::stride, ::stride]
    elif layout.startswith("chw"):
        sampled = source[:3, ::stride, ::stride]
    else:
        sampled = source[::stride, ::stride, :3]
    canonical, _provenance = canonicalize_stage7_pixels_01(sampled)
    rgb = _to_rgb_float_fullres(canonical)
    return rgb, {
        "source_layout": layout,
        "source_spatial_shape": [int(height), int(width)],
        "analysis_stride": stride,
        "analysis_stride_unit": "source_spatial_pixels",
        "analysis_grid_shape": [
            int(rgb.shape[1]),
            int(rgb.shape[2]),
        ],
    }


def _stage7_rgb_triplet_sample(
    image: np.ndarray,
    *,
    max_samples: int,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Sample complete spatial RGB pixels without copying the full image."""
    source = np.asarray(image)
    if source.size == 0:
        raise ValueError("empty image data")
    # Validate the full source domain from its extrema, then canonicalize only
    # the sampled pixels. Stage 7 already validates every full candidate at the
    # Siril pixel boundary; this avoids another full-size float allocation for
    # a report-only diagnostic.
    extrema = np.asarray(
        [np.min(source), np.max(source)],
        dtype=source.dtype,
    )
    canonicalize_stage7_pixels_01(extrema)

    sample_limit = max(64, int(max_samples))
    layout = ""
    if source.ndim == 2:
        height, width = source.shape
        pixel_count = int(height * width)
        stride = max(1, int(np.ceil(pixel_count / float(sample_limit))))
        mono = source.reshape(-1)[::stride]
        sampled = np.repeat(mono[:, None], 3, axis=1)
        layout = "mono_2d"
    elif source.ndim == 3:
        if source.shape[0] == 1 and source.shape[-1] not in (1, 3):
            height, width = source.shape[1:]
            pixel_count = int(height * width)
            stride = max(1, int(np.ceil(pixel_count / float(sample_limit))))
            mono = source[0].reshape(-1)[::stride]
            sampled = np.repeat(mono[:, None], 3, axis=1)
            layout = "chw_mono"
        elif source.shape[0] == 3 and source.shape[-1] not in (1, 3):
            height, width = source.shape[1:]
            pixel_count = int(height * width)
            stride = max(1, int(np.ceil(pixel_count / float(sample_limit))))
            sampled = source[:3].reshape(3, -1)[:, ::stride].T
            layout = "chw_rgb"
        elif source.shape[-1] == 1:
            height, width = source.shape[:2]
            pixel_count = int(height * width)
            stride = max(1, int(np.ceil(pixel_count / float(sample_limit))))
            mono = source[..., 0].reshape(-1)[::stride]
            sampled = np.repeat(mono[:, None], 3, axis=1)
            layout = "hwc_mono"
        elif source.shape[-1] >= 3:
            height, width = source.shape[:2]
            pixel_count = int(height * width)
            stride = max(1, int(np.ceil(pixel_count / float(sample_limit))))
            sampled = source[..., :3].reshape(-1, 3)[::stride]
            layout = "hwc_rgb"
        elif source.shape[0] >= 3:
            height, width = source.shape[1:]
            pixel_count = int(height * width)
            stride = max(1, int(np.ceil(pixel_count / float(sample_limit))))
            sampled = source[:3].reshape(3, -1)[:, ::stride].T
            layout = "chw_rgb"
        else:
            raise ValueError(f"unsupported image shape: {source.shape}")
    else:
        raise ValueError(f"unsupported image ndim: {source.ndim}")

    canonical, _provenance = canonicalize_stage7_pixels_01(sampled)
    return np.asarray(canonical, dtype=np.float32), {
        "source_shape": [int(value) for value in source.shape],
        "spatial_shape": [int(height), int(width)],
        "source_layout": layout,
        "spatial_pixel_count": pixel_count,
        "sample_pixel_count": int(canonical.shape[0]),
        "sample_stride": stride,
        "sample_stride_unit": "spatial_pixels",
    }


def _bounded(value: Any, default: float, lower: float, upper: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(lower, min(upper, parsed))


def build_siril_stretch_semantics(
    method: str,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Describe the exact stretch command contract without executing it."""
    method_name = str(method or "").strip().lower()
    values = dict(params or {})
    base: Dict[str, Any] = {
        "schema": STRETCH_SEMANTICS_SCHEMA,
        "status": "available",
        "engine": "siril",
        "method": method_name,
        "minimum_siril_version": SIRIL_MINIMUM_VERSION_CONTRACT,
        "bundled_reference_version": SIRIL_BUNDLED_REFERENCE_VERSION,
        "steps": [],
    }

    if method_name in {"asinh", "asinh_ghs", "bright_nebula_hdr_masked"}:
        stretch = str(values.get("asinh_stretch", 3.0))
        offset = str(values.get("asinh_offset", 0.001))
        base.update(
            {
                "luminance_mode": "mean_rgb",
                "human_weighted": False,
                "clip_mode": "rgbblend",
                "default_dependent_fields": ["luminance_mode"],
            }
        )
        base["steps"].append(
            {
                "command": "asinh",
                "argv": [stretch, offset, "-clipmode=rgbblend"],
                "full_argv": [
                    "asinh",
                    stretch,
                    offset,
                    "-clipmode=rgbblend",
                ],
                "luminance_mode": "mean_rgb",
                "human_weighted": False,
                "clip_mode": "rgbblend",
                "clip_mode_explicit": True,
                "luminance_mode_encoded_by": "absence_of_-human",
            }
        )
        if method_name == "asinh_ghs":
            base["steps"].append(
                {
                    "command": "autoghs",
                    "argv": [
                        "-linked",
                        str(values.get("ghs_shadowsclip", -2.8)),
                        str(values.get("ghs_stretchamount", 2.0)),
                    ],
                    "full_argv": [
                        "autoghs",
                        "-linked",
                        str(values.get("ghs_shadowsclip", -2.8)),
                        str(values.get("ghs_stretchamount", 2.0)),
                    ],
                }
            )
        elif method_name == "bright_nebula_hdr_masked":
            base["steps"].append(
                {
                    "command": "numpy_bright_nebula_hdr_masked",
                    "argv": [],
                    "spatially_varying": True,
                }
            )
        return base

    if method_name == "linked_mtf":
        shadows = float(values.get("mtf_shadows", 0.0))
        midtones = float(values.get("mtf_midtones", 0.5))
        highlights = float(values.get("mtf_highlights", 1.0))
        base.update(
            {
                "luminance_mode": "per_channel_common_curve",
                "human_weighted": False,
                "clip_mode": "mtf_endpoints",
            }
        )
        base["steps"].append(
            {
                "command": "mtf",
                "argv": [
                    f"{shadows:.6f}",
                    f"{midtones:.6f}",
                    f"{highlights:.6f}",
                ],
                "full_argv": [
                    "mtf",
                    f"{shadows:.6f}",
                    f"{midtones:.6f}",
                    f"{highlights:.6f}",
                ],
            }
        )
        return base

    if method_name == "adaptive_quantile":
        base.update(
            {
                "engine": "numpy",
                "luminance_mode": "linked_rgb_curve",
                "human_weighted": False,
                "clip_mode": "bounded_monotonic_curve",
                "steps": [
                    {
                        "command": "numpy_adaptive_quantile",
                        "argv": [],
                    }
                ],
            }
        )
        return base

    if method_name in {
        "iterative_masked_mtf",
        "dual_stage_mtf_ghs",
        "display90_linked_lut",
    }:
        is_display90 = method_name == "display90_linked_lut"
        base.update(
            {
                "engine": "numpy",
                "luminance_mode": "linked_rgb_curve",
                "human_weighted": False,
                "clip_mode": "bounded_monotonic_curve",
                "minimum_siril_version": None,
                "bundled_reference_version": None,
                "linked_rgb": True,
                "spatially_invariant": True,
                "output_headroom": CONDITIONAL_STRETCH_OUTPUT_HEADROOM,
                "external_reference": (
                    None if is_display90 else CONDITIONAL_STRETCH_REFERENCE
                ),
                "external_source_license": (
                    None if is_display90 else CONDITIONAL_STRETCH_SOURCE_LICENSE
                ),
                "pixel_equivalence_claimed": False,
                "steps": [
                    {
                        "command": f"numpy_{method_name}",
                        "argv": [],
                        "lut_size": CONDITIONAL_STRETCH_LUT_SIZE,
                    }
                ],
            }
        )
        return base

    return {
        **base,
        "status": "not_applicable",
        "reason": "stretch semantics are not defined for this method",
    }


def _effective_blackpoint(
    method: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    method_name = str(method or "").strip().lower()
    if method_name in {"asinh", "asinh_ghs", "bright_nebula_hdr_masked"}:
        key = "asinh_offset"
    elif method_name == "linked_mtf":
        key = "mtf_shadows"
    else:
        return {
            "status": "not_applicable",
            "value": None,
            "source": None,
        }
    try:
        value = float(params.get(key, 0.0))
    except (TypeError, ValueError):
        return {
            "status": "unavailable",
            "value": None,
            "source": key,
            "reason": "effective blackpoint is not numeric",
        }
    if not np.isfinite(value):
        return {
            "status": "unavailable",
            "value": None,
            "source": key,
            "reason": "effective blackpoint is not finite",
        }
    return {
        "status": "available",
        "value": float(value),
        "source": key,
    }


def _ratio_by_channel(mask: np.ndarray) -> Dict[str, float]:
    values = np.asarray(mask, dtype=bool)
    return {
        channel: float(np.mean(values[index]))
        for index, channel in enumerate(("r", "g", "b"))
    }


def _transform_loss_region(
    source_rgb: np.ndarray,
    candidate_rgb: np.ndarray,
    *,
    spatial_mask: Optional[np.ndarray],
    effective_blackpoint: Dict[str, Any],
) -> Dict[str, Any]:
    source = np.asarray(source_rgb, dtype=np.float32)
    candidate = np.asarray(candidate_rgb, dtype=np.float32)
    if spatial_mask is None:
        selected_source = source.reshape(3, -1)
        selected_candidate = candidate.reshape(3, -1)
        region = "full_image"
    else:
        selected = np.asarray(spatial_mask, dtype=bool)
        selected_source = source[:, selected]
        selected_candidate = candidate[:, selected]
        region = "frozen_background_mask"
    if selected_source.shape[1] <= 0:
        return {
            "status": "unavailable",
            "region": region,
            "reason": "region contains no spatial pixels",
        }

    zero_before = selected_source <= TRANSFORM_ZERO_EPSILON
    zero_after = selected_candidate <= TRANSFORM_ZERO_EPSILON
    newly_zeroed = (~zero_before) & zero_after
    hard_high_before = selected_source >= 1.0 - TRANSFORM_ZERO_EPSILON
    hard_high_after = selected_candidate >= 1.0 - TRANSFORM_ZERO_EPSILON
    newly_hard_high = (~hard_high_before) & hard_high_after
    near_black_before = selected_source <= TRANSFORM_NEAR_BLACK
    near_black_after = selected_candidate <= TRANSFORM_NEAR_BLACK
    near_high_before = selected_source >= TRANSFORM_NEAR_HIGHLIGHT
    near_high_after = selected_candidate >= TRANSFORM_NEAR_HIGHLIGHT

    def ratio_block(before: np.ndarray, after: np.ndarray) -> Dict[str, Any]:
        before_ratio = float(np.mean(before))
        after_ratio = float(np.mean(after))
        before_channels = _ratio_by_channel(before)
        after_channels = _ratio_by_channel(after)
        return {
            "before": before_ratio,
            "after": after_ratio,
            "growth": after_ratio - before_ratio,
            "before_by_channel": before_channels,
            "after_by_channel": after_channels,
            "growth_by_channel": {
                channel: after_channels[channel] - before_channels[channel]
                for channel in ("r", "g", "b")
            },
        }

    blackpoint_block: Dict[str, Any] = {
        "status": effective_blackpoint.get("status", "not_applicable"),
        "value": effective_blackpoint.get("value"),
        "source": effective_blackpoint.get("source"),
    }
    if effective_blackpoint.get("status") == "available":
        below = selected_source <= float(effective_blackpoint["value"])
        unexpected_newly_zeroed = newly_zeroed & ~below
        blackpoint_block.update(
            {
                "source_below_blackpoint_ratio": float(np.mean(below)),
                "source_below_blackpoint_ratio_by_channel": _ratio_by_channel(below),
                "spatial_any_channel_below_ratio": float(np.mean(np.any(below, axis=0))),
                "spatial_all_channels_below_ratio": float(np.mean(np.all(below, axis=0))),
                "unexpected_newly_zeroed_ratio": float(
                    np.mean(unexpected_newly_zeroed)
                ),
                "unexpected_newly_zeroed_ratio_by_channel": _ratio_by_channel(
                    unexpected_newly_zeroed
                ),
            }
        )
    else:
        unexpected_newly_zeroed = newly_zeroed

    zero_block = {
        **ratio_block(zero_before, zero_after),
        "newly_zeroed_ratio": float(np.mean(newly_zeroed)),
        "newly_zeroed_ratio_by_channel": _ratio_by_channel(newly_zeroed),
    }
    hard_high_block = {
        **ratio_block(hard_high_before, hard_high_after),
        "newly_clipped_ratio": float(np.mean(newly_hard_high)),
        "newly_clipped_ratio_by_channel": _ratio_by_channel(newly_hard_high),
    }
    near_black_block = ratio_block(near_black_before, near_black_after)
    near_highlight_block = ratio_block(near_high_before, near_high_after)
    result = {
        "status": "available",
        "region": region,
        "spatial_pixel_count": int(selected_source.shape[1]),
        "channel_sample_count": int(selected_source.size),
        "effective_blackpoint": blackpoint_block,
        "zero_ratio_before": zero_block["before"],
        "zero_ratio_after": zero_block["after"],
        "zero_ratio_growth": zero_block["growth"],
        "newly_zeroed_ratio": zero_block["newly_zeroed_ratio"],
        "unexpected_newly_zeroed_ratio": float(
            np.mean(unexpected_newly_zeroed)
        ),
        "unexpected_newly_zeroed_ratio_by_channel": _ratio_by_channel(
            unexpected_newly_zeroed
        ),
        "zero": zero_block,
        "hard_clip_ratio_before": hard_high_block["before"],
        "hard_clip_ratio_after": hard_high_block["after"],
        "hard_clip_ratio_growth": hard_high_block["growth"],
        "newly_hard_clipped_ratio": hard_high_block["newly_clipped_ratio"],
        "hard_high_clip": hard_high_block,
        "near_black_ratio_before": near_black_block["before"],
        "near_black_ratio_after": near_black_block["after"],
        "near_black_ratio_growth": near_black_block["growth"],
        "near_black": near_black_block,
        "near_highlight_ratio_before": near_highlight_block["before"],
        "near_highlight_ratio_after": near_highlight_block["after"],
        "near_highlight_ratio_growth": near_highlight_block["growth"],
        "near_highlight": near_highlight_block,
    }
    if effective_blackpoint.get("status") == "available":
        result["source_below_blackpoint_ratio"] = blackpoint_block[
            "source_below_blackpoint_ratio"
        ]
        result["source_below_blackpoint_ratio_by_channel"] = blackpoint_block[
            "source_below_blackpoint_ratio_by_channel"
        ]
    else:
        result["source_below_blackpoint_ratio"] = None
        result["source_below_blackpoint_ratio_by_channel"] = None
    return result


def assess_transform_loss(
    source_image: np.ndarray,
    candidate_image: np.ndarray,
    *,
    method: str,
    params: Optional[Dict[str, Any]] = None,
    background_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Measure exact before/after clipping diagnostics without gating output."""
    base: Dict[str, Any] = {
        "schema": TRANSFORM_LOSS_SCHEMA,
        "role": "report_only",
        "enforced": False,
        "participates_in_selection": False,
        "method": str(method or ""),
        "zero_epsilon": TRANSFORM_ZERO_EPSILON,
        "near_black": TRANSFORM_NEAR_BLACK,
        "near_highlight": TRANSFORM_NEAR_HIGHLIGHT,
        "near_black_threshold": TRANSFORM_NEAR_BLACK,
        "near_highlight_threshold": TRANSFORM_NEAR_HIGHLIGHT,
    }
    try:
        source_rgb = _stage7_rgb_float_fullres(np.asarray(source_image))
        candidate_rgb = _stage7_rgb_float_fullres(np.asarray(candidate_image))
        if source_rgb.shape != candidate_rgb.shape:
            raise ValueError(
                "source/candidate shape mismatch: "
                f"source={source_rgb.shape}, candidate={candidate_rgb.shape}"
            )
        if not np.all(np.isfinite(source_rgb)) or not np.all(np.isfinite(candidate_rgb)):
            raise ValueError("source/candidate contains non-finite pixels")
        blackpoint = _effective_blackpoint(str(method or ""), dict(params or {}))
        global_report = _transform_loss_region(
            source_rgb,
            candidate_rgb,
            spatial_mask=None,
            effective_blackpoint=blackpoint,
        )
        background_report: Dict[str, Any]
        if background_mask is None:
            background_report = {
                "status": "unavailable",
                "region": "frozen_background_mask",
                "reason": "frozen background mask unavailable",
            }
        else:
            mask = np.asarray(background_mask, dtype=np.float32)
            expected_shape = tuple(int(value) for value in source_rgb.shape[1:])
            if mask.ndim != 2 or mask.shape != expected_shape:
                background_report = {
                    "status": "unavailable",
                    "region": "frozen_background_mask",
                    "reason": (
                        "background mask shape mismatch: "
                        f"expected={expected_shape}, actual={mask.shape}"
                    ),
                }
            elif not np.all(np.isfinite(mask)):
                background_report = {
                    "status": "unavailable",
                    "region": "frozen_background_mask",
                    "reason": "background mask contains non-finite values",
                }
            else:
                background_report = _transform_loss_region(
                    source_rgb,
                    candidate_rgb,
                    spatial_mask=mask > 0.50,
                    effective_blackpoint=blackpoint,
                )
        return {
            **base,
            "status": "available",
            "effective_blackpoint": blackpoint,
            "global": global_report,
            "background_roi": background_report,
        }
    except (IndexError, TypeError, ValueError, FloatingPointError) as error:
        return {
            **base,
            "status": "unavailable",
            "reason": str(error),
        }


def solve_linked_mtf_midpoint(source_value: float, target_value: float) -> float:
    """Solve the closed-form linked MTF midpoint for one normalized level."""
    source_raw = float(source_value)
    target_raw = float(target_value)
    if not np.isfinite(source_raw) or not np.isfinite(target_raw):
        raise ValueError("linked MTF source and target must be finite")
    source = max(1e-6, min(0.999999, source_raw))
    target = max(1e-6, min(0.999999, target_raw))
    denominator = source * (1.0 - 2.0 * target) + target
    if not np.isfinite(denominator) or abs(denominator) < 1e-12:
        raise ValueError("cannot solve a stable linked MTF midpoint")
    midpoint = source * (1.0 - target) / denominator
    if not np.isfinite(midpoint) or not 0.0 < midpoint < 1.0:
        raise ValueError("linked MTF midpoint is outside (0,1)")
    return float(midpoint)


def linked_mtf_sample(
    value: float,
    shadows: float,
    midtones: float,
    highlights: float = 1.0,
) -> float:
    """Evaluate the scalar linked MTF used by Siril and Stage 7."""
    value = float(value)
    shadows = float(shadows)
    midtones = float(midtones)
    highlights = float(highlights)
    if not all(
        np.isfinite(item) for item in (value, shadows, midtones, highlights)
    ):
        raise ValueError("linked MTF inputs must be finite")
    if not 0.0 <= shadows < highlights <= 1.0:
        raise ValueError("linked MTF shadows/highlights are invalid")
    if not 0.0 < midtones < 1.0:
        raise ValueError("linked MTF midpoint is outside (0,1)")
    if value <= shadows:
        return 0.0
    if value >= highlights:
        return 1.0
    normalized = (value - shadows) / max(highlights - shadows, 1e-12)
    denominator = (2.0 * midtones - 1.0) * normalized - midtones
    if abs(denominator) < 1e-12:
        raise ValueError("linked MTF denominator is unstable")
    mapped = (midtones - 1.0) * normalized / denominator
    return float(np.clip(mapped, 0.0, 1.0))


def apply_linked_mtf(
    values: np.ndarray,
    shadows: float,
    midtones: float,
    highlights: float = 1.0,
) -> np.ndarray:
    """Vectorized closed-form linked MTF used for matched-domain references."""
    shadows = float(shadows)
    midtones = float(midtones)
    highlights = float(highlights)
    if not all(
        np.isfinite(item) for item in (shadows, midtones, highlights)
    ):
        raise ValueError("linked MTF parameters must be finite")
    if not 0.0 <= shadows < highlights <= 1.0:
        raise ValueError("linked MTF shadows/highlights are invalid")
    if not 0.0 < midtones < 1.0:
        raise ValueError("linked MTF midpoint is outside (0,1)")

    source = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(source)):
        raise ValueError("linked MTF source pixels must be finite")
    normalized = np.clip(
        (source - shadows) / max(highlights - shadows, 1e-12),
        0.0,
        1.0,
    )
    denominator = (2.0 * midtones - 1.0) * normalized - midtones
    unstable = (normalized > 0.0) & (normalized < 1.0) & (
        np.abs(denominator) < 1e-12
    )
    if np.any(unstable):
        raise ValueError("linked MTF denominator is unstable")
    mapped = np.divide(
        (midtones - 1.0) * normalized,
        denominator,
        out=np.zeros_like(normalized, dtype=np.float32),
        where=np.abs(denominator) >= 1e-12,
    )
    mapped = np.where(source <= shadows, 0.0, mapped)
    mapped = np.where(source >= highlights, 1.0, mapped)
    return np.clip(mapped, 0.0, 1.0).astype(np.float32, copy=False)


def _statistical_scale_estimator_report(
    *,
    method: str,
    raw_mad: float,
    normal_gaussian_sigma_ratio: float,
    source_p50: float,
    source_min: float,
    blackpoint_sigma: float,
    sample: np.ndarray,
    analysis_values: np.ndarray,
) -> Dict[str, Any]:
    robust_sigma = LOWER_HALF_RECENTERED_MAD_SCALE * max(float(raw_mad), 0.0)
    blackpoint = max(
        float(source_min),
        float(source_p50) - float(blackpoint_sigma) * robust_sigma,
    )
    blackpoint = min(blackpoint, 0.99)
    return {
        "method": method,
        "raw_mad": float(raw_mad),
        "mad_scale_factor": LOWER_HALF_RECENTERED_MAD_SCALE,
        "robust_sigma": float(robust_sigma),
        "normal_gaussian_sigma_ratio": float(normal_gaussian_sigma_ratio),
        "nominal_blackpoint_sigma_multiplier": float(blackpoint_sigma),
        "normal_gaussian_equivalent_blackpoint_sigma": float(
            blackpoint_sigma * normal_gaussian_sigma_ratio
        ),
        "blackpoint": float(blackpoint),
        "sample_below_or_equal_blackpoint_ratio": float(
            np.mean(np.asarray(sample) <= blackpoint)
        ),
        "analysis_below_or_equal_blackpoint_ratio": float(
            np.mean(np.asarray(analysis_values) <= blackpoint)
        ),
    }


def _zscale_interval_reference(
    sample: np.ndarray,
    analysis_values: np.ndarray,
) -> Dict[str, Any]:
    """Build an independent display-interval diagnostic, never a curve."""
    try:
        from astropy.visualization import ZScaleInterval

        values = np.asarray(sample, dtype=np.float64).reshape(-1)
        if values.size < 64 or not np.all(np.isfinite(values)):
            raise ValueError("too few finite samples for ZScale reference")
        n_samples = min(1000, int(values.size))
        interval = ZScaleInterval(
            n_samples=n_samples,
            contrast=0.25,
            max_reject=0.5,
            min_npixels=5,
            krej=2.5,
            max_iterations=5,
        )
        z1, z2 = (float(value) for value in interval.get_limits(values))
        if not np.isfinite(z1) or not np.isfinite(z2) or z1 >= z2:
            raise ValueError(f"ZScale returned an invalid interval: {z1}, {z2}")
        analysis = np.asarray(analysis_values, dtype=np.float64).reshape(-1)
        return {
            "status": "available",
            "role": "report_only_interval",
            "participates_in_selection": False,
            "implementation": "astropy.visualization.ZScaleInterval",
            "interval_only": True,
            "blackpoint_candidate": z1,
            "whitepoint_candidate": z2,
            "analysis_below_blackpoint_ratio": float(np.mean(analysis <= z1)),
            "analysis_above_whitepoint_ratio": float(np.mean(analysis >= z2)),
            "sample_count": int(values.size),
            "fit_sample_limit": n_samples,
            "parameters": {
                "contrast": 0.25,
                "max_reject": 0.5,
                "min_npixels": 5,
                "krej": 2.5,
                "max_iterations": 5,
            },
        }
    except (ImportError, IndexError, TypeError, ValueError, FloatingPointError) as error:
        return {
            "status": "unavailable",
            "role": "report_only_interval",
            "participates_in_selection": False,
            "implementation": "astropy.visualization.ZScaleInterval",
            "interval_only": True,
            "reason": str(error),
        }


def build_statistical_mtf_reference(
    image: np.ndarray,
    target_p50: float,
    *,
    blackpoint_sigma: float = 5.0,
    max_samples: int = 400_000,
    reference_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Build a reference-only linked MTF using the vendored blackpoint rule.

    The active shadow keeps Statistical Stretch's recentered lower-half MAD.
    Full MAD, a true lower-side deviation around the global median, and ZScale
    are emitted as independent report-only comparators. Complete spatial RGB
    triplets avoid the vendored flat-stride channel bias. No value returned by
    this function creates a candidate or participates in final selection.
    """
    try:
        rgb = _stage7_rgb_float_fullres(np.asarray(image))
        rgb_pixels = np.moveaxis(
            np.asarray(rgb, dtype=np.float32),
            0,
            -1,
        ).reshape(-1, 3)
        if rgb_pixels.size < 64 or not np.all(np.isfinite(rgb_pixels)):
            raise ValueError("too few finite pixels for statistical MTF reference")

        reference_region = "full_image"
        mask_fallback_reason: Optional[str] = None
        analysis_pixels = rgb_pixels
        if reference_mask is not None:
            mask = np.asarray(reference_mask, dtype=np.float32)
            expected_shape = tuple(int(value) for value in rgb.shape[1:])
            if mask.ndim != 2 or mask.shape != expected_shape:
                mask_fallback_reason = (
                    "reference mask shape mismatch: "
                    f"expected={expected_shape}, actual={mask.shape}"
                )
            elif not np.all(np.isfinite(mask)):
                mask_fallback_reason = "reference mask contains non-finite values"
            else:
                selected = mask.reshape(-1) > 0.50
                selected_pixels = rgb_pixels[selected]
                if selected_pixels.size >= 64:
                    analysis_pixels = selected_pixels
                    reference_region = "frozen_background_mask"
                else:
                    mask_fallback_reason = (
                        "reference mask contains fewer than 64 RGB samples"
                    )

        sample_limit = max(64, int(max_samples))
        pixel_sample_limit = max(1, sample_limit // 3)
        stride = max(
            1,
            int(np.ceil(analysis_pixels.shape[0] / float(pixel_sample_limit))),
        )
        sample_pixels = analysis_pixels[::stride]
        sample = sample_pixels.reshape(-1)
        sample_median = float(np.median(sample))
        lower_half = sample[sample <= sample_median]
        full_mad = float(np.median(np.abs(sample - sample_median)))
        one_sided_global_center_mad = (
            float(np.median(sample_median - lower_half))
            if lower_half.size
            else full_mad
        )
        if lower_half.size < 16:
            lower_half_recentered_mad = full_mad
            lower_half_fallback = "full_mad_due_to_small_lower_half"
        else:
            lower_center = float(np.median(lower_half))
            lower_half_recentered_mad = float(
                np.median(np.abs(lower_half - lower_center))
            )
            lower_half_fallback = None

        analysis_flat = analysis_pixels.reshape(-1)
        source_p50 = float(np.median(analysis_flat))
        source_p99 = float(np.percentile(analysis_flat, 99.0))
        source_min = float(np.min(analysis_flat))
        sigma = _bounded(blackpoint_sigma, 5.0, 0.5, 8.0)
        scale_estimators = {
            "full_mad": _statistical_scale_estimator_report(
                method="full_mad",
                raw_mad=full_mad,
                normal_gaussian_sigma_ratio=FULL_MAD_NORMAL_SIGMA_RATIO,
                source_p50=source_p50,
                source_min=source_min,
                blackpoint_sigma=sigma,
                sample=sample,
                analysis_values=analysis_flat,
            ),
            "one_sided_global_center_mad": _statistical_scale_estimator_report(
                method="one_sided_global_center_mad",
                raw_mad=one_sided_global_center_mad,
                normal_gaussian_sigma_ratio=FULL_MAD_NORMAL_SIGMA_RATIO,
                source_p50=source_p50,
                source_min=source_min,
                blackpoint_sigma=sigma,
                sample=sample,
                analysis_values=analysis_flat,
            ),
            "lower_half_recentered_mad": _statistical_scale_estimator_report(
                method="lower_half_recentered_mad",
                raw_mad=lower_half_recentered_mad,
                normal_gaussian_sigma_ratio=(
                    LOWER_HALF_RECENTERED_MAD_NORMAL_SIGMA_RATIO
                ),
                source_p50=source_p50,
                source_min=source_min,
                blackpoint_sigma=sigma,
                sample=sample,
                analysis_values=analysis_flat,
            ),
        }
        active_scale_estimator = scale_estimators[
            "lower_half_recentered_mad"
        ]
        robust_sigma = float(active_scale_estimator["robust_sigma"])
        blackpoint = float(active_scale_estimator["blackpoint"])
        if not blackpoint < source_p50 < 1.0:
            raise ValueError("statistical blackpoint does not leave a usable median")

        normalized_p50 = (source_p50 - blackpoint) / max(
            1.0 - blackpoint,
            1e-12,
        )
        midpoint = solve_linked_mtf_midpoint(normalized_p50, target_p50)
        predicted_p50 = linked_mtf_sample(
            source_p50,
            blackpoint,
            midpoint,
        )
        predicted_p99 = linked_mtf_sample(
            source_p99,
            blackpoint,
            midpoint,
        )
        clipped_ratio = float(np.mean(sample <= blackpoint))
        analysis_clipped_ratio = float(np.mean(analysis_flat <= blackpoint))
        return {
            "status": "available",
            "role": "reference_only",
            "method": "closed_form_linked_mtf",
            "source": STATISTICAL_MTF_REFERENCE_SOURCE,
            "equivalence_scope": "linked_rgb_no_curves_no_hdr_no_normalize",
            "blackpoint_method": "lower_half_recentered_mad",
            "blackpoint_method_detail": "lower_half_recentered_mad",
            "active_scale_estimator": "lower_half_recentered_mad",
            "blackpoint_sigma": sigma,
            "mad_scale_factor": LOWER_HALF_RECENTERED_MAD_SCALE,
            "normal_gaussian_sigma_ratio": (
                LOWER_HALF_RECENTERED_MAD_NORMAL_SIGMA_RATIO
            ),
            "normal_gaussian_equivalent_blackpoint_sigma": (
                sigma * LOWER_HALF_RECENTERED_MAD_NORMAL_SIGMA_RATIO
            ),
            "blackpoint": float(blackpoint),
            "robust_sigma": float(robust_sigma),
            "source_p50": source_p50,
            "source_p99": source_p99,
            "target_p50": float(target_p50),
            "midtones": midpoint,
            "highlights": 1.0,
            "predicted_p50": predicted_p50,
            "predicted_p99": predicted_p99,
            "estimated_source_below_blackpoint_ratio": clipped_ratio,
            "analysis_source_below_blackpoint_ratio": analysis_clipped_ratio,
            "scale_estimators": scale_estimators,
            "lower_half_estimator_fallback": lower_half_fallback,
            "zscale_interval_reference": _zscale_interval_reference(
                sample,
                analysis_flat,
            ),
            "sample_count": int(sample.size),
            "sample_pixel_count": int(sample_pixels.shape[0]),
            "sample_channel_count": 3,
            "sample_channel_medians": [
                float(value) for value in np.median(sample_pixels, axis=0)
            ],
            "sample_layout": "spatial_rgb_triplets",
            "sample_stride": stride,
            "sample_stride_unit": "spatial_pixels",
            "sampling_equivalence_scope": (
                "estimator_equivalent_not_vendor_flat_stride_equivalent"
            ),
            "reference_region": reference_region,
            "reference_mask_fallback_reason": mask_fallback_reason,
            "final_candidate": False,
        }
    except (IndexError, TypeError, ValueError, FloatingPointError) as error:
        return {
            "status": "unavailable",
            "role": "reference_only",
            "method": "closed_form_linked_mtf",
            "source": STATISTICAL_MTF_REFERENCE_SOURCE,
            "equivalence_scope": "linked_rgb_no_curves_no_hdr_no_normalize",
            "final_candidate": False,
            "reason": str(error),
        }


def _box_mean_gray_at_radius(gray: np.ndarray, radius: int) -> np.ndarray:
    """Return a reflect-padded square box mean in O(H*W) time."""
    arr = np.asarray(gray, dtype=np.float32)
    radius = int(radius)
    if arr.ndim != 2:
        raise ValueError(f"expected gray image, got shape={arr.shape}")
    if radius < 1:
        raise ValueError(f"box radius must be positive, got {radius}")
    if min(arr.shape) <= 2 * radius:
        raise ValueError(
            f"analysis grid {arr.shape} is too small for radius={radius}"
        )
    padded = np.pad(arr, ((radius, radius), (radius, radius)), mode="reflect")
    integral = np.pad(
        padded.astype(np.float64, copy=False),
        ((1, 0), (1, 0)),
        mode="constant",
    )
    integral = np.cumsum(np.cumsum(integral, axis=0), axis=1)
    diameter = 2 * radius + 1
    window_sum = (
        integral[diameter:, diameter:]
        - integral[:-diameter, diameter:]
        - integral[diameter:, :-diameter]
        + integral[:-diameter, :-diameter]
    )
    return np.asarray(window_sum / float(diameter * diameter), dtype=np.float32)


def assess_multiscale_contrast_reference(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    max_side: int = 1024,
    radii: tuple[int, ...] = MULTISCALE_CONTRAST_RADII,
) -> Dict[str, Any]:
    """Audit contrast transfer at fixed scales without affecting selection.

    This is a deterministic box-residual diagnostic inspired by the need for
    multiscale review. It is deliberately not described as an implementation
    of PixInsight MultiscaleAdaptiveStretch and never acts as a quality gate.
    """
    base: Dict[str, Any] = {
        "schema": MULTISCALE_CONTRAST_SCHEMA,
        "status": "unavailable",
        "role": "report_only",
        "enforced": False,
        "participates_in_selection": False,
        "reference_model": "rec709_gray_minus_reflect_box_mean",
        "mas_equivalent": False,
    }
    try:
        source_array = np.asarray(baseline)
        candidate_array = np.asarray(candidate)
        if source_array.shape != candidate_array.shape:
            raise ValueError(
                "shape mismatch: "
                f"baseline={source_array.shape}, candidate={candidate_array.shape}"
            )
        bounded_max_side = max(64, min(int(max_side), 2048))
        source_rgb, source_sampling = _stage7_rgb_analysis_grid(
            source_array,
            max_side=bounded_max_side,
        )
        candidate_rgb, candidate_sampling = _stage7_rgb_analysis_grid(
            candidate_array,
            max_side=bounded_max_side,
        )
        if (
            source_rgb.shape != candidate_rgb.shape
            or source_sampling["analysis_stride"]
            != candidate_sampling["analysis_stride"]
        ):
            raise ValueError(
                "analysis grid mismatch: "
                f"baseline={source_rgb.shape}, candidate={candidate_rgb.shape}"
            )
        source_gray = (
            0.2126 * source_rgb[0]
            + 0.7152 * source_rgb[1]
            + 0.0722 * source_rgb[2]
        ).astype(np.float32)
        candidate_gray = (
            0.2126 * candidate_rgb[0]
            + 0.7152 * candidate_rgb[1]
            + 0.0722 * candidate_rgb[2]
        ).astype(np.float32)
        requested_radii = tuple(dict.fromkeys(int(value) for value in radii))
        if not requested_radii or any(value < 1 for value in requested_radii):
            raise ValueError(f"invalid multiscale radii: {radii}")
        if min(source_gray.shape) <= 2 * max(requested_radii):
            raise ValueError(
                f"analysis grid {source_gray.shape} is too small for "
                f"radius={max(requested_radii)}"
            )

        source_p01, source_p99 = np.percentile(source_gray, [1.0, 99.0])
        candidate_p01, candidate_p99 = np.percentile(
            candidate_gray,
            [1.0, 99.0],
        )
        source_span = max(float(source_p99 - source_p01), 1e-7)
        candidate_span = max(float(candidate_p99 - candidate_p01), 1e-7)
        scale_reports = []
        for radius in requested_radii:
            source_residual = source_gray - _box_mean_gray_at_radius(
                source_gray,
                radius,
            )
            candidate_residual = candidate_gray - _box_mean_gray_at_radius(
                candidate_gray,
                radius,
            )
            source_abs = np.abs(source_residual).astype(np.float64, copy=False)
            candidate_abs = np.abs(candidate_residual).astype(
                np.float64,
                copy=False,
            )
            source_rms = float(
                np.sqrt(np.mean(source_residual.astype(np.float64) ** 2))
            )
            candidate_rms = float(
                np.sqrt(np.mean(candidate_residual.astype(np.float64) ** 2))
            )
            source_p90 = float(np.percentile(source_abs, 90.0))
            candidate_p90 = float(np.percentile(candidate_abs, 90.0))
            detail_floor = max(float(np.percentile(source_abs, 75.0)), 1e-7)
            detail_support = source_abs >= detail_floor
            detail_count = int(np.count_nonzero(detail_support))
            sign_reversal_ratio = None
            if detail_count >= 64:
                sign_reversal_ratio = float(
                    np.mean(
                        np.signbit(source_residual[detail_support])
                        != np.signbit(candidate_residual[detail_support])
                    )
                )
            scale_reports.append(
                {
                    "radius": radius,
                    "diameter": 2 * radius + 1,
                    "source_residual_rms": source_rms,
                    "candidate_residual_rms": candidate_rms,
                    "absolute_rms_gain": float(
                        candidate_rms / max(source_rms, 1e-12)
                    ),
                    "span_normalized_rms_gain": float(
                        (candidate_rms / candidate_span)
                        / max(source_rms / source_span, 1e-12)
                    ),
                    "source_abs_residual_p90": source_p90,
                    "candidate_abs_residual_p90": candidate_p90,
                    "absolute_p90_gain": float(
                        candidate_p90 / max(source_p90, 1e-12)
                    ),
                    "source_detail_support_count": detail_count,
                    "sign_reversal_ratio_on_source_detail_support": (
                        sign_reversal_ratio
                    ),
                    "sign_reversal_is_ringing_proxy_only": True,
                }
            )
        return {
            **base,
            "status": "available",
            "analysis_grid_shape": [
                int(source_gray.shape[0]),
                int(source_gray.shape[1]),
            ],
            "analysis_max_side": bounded_max_side,
            "source_layout": source_sampling["source_layout"],
            "analysis_stride": source_sampling["analysis_stride"],
            "analysis_stride_unit": source_sampling["analysis_stride_unit"],
            "scale_unit": "analysis_grid_box_radius_pixels",
            "source_pixel_scale_equivalence": "not_claimed",
            "source_luminance_span_p01_p99": source_span,
            "candidate_luminance_span_p01_p99": candidate_span,
            "requested_radii": list(requested_radii),
            "scales": scale_reports,
        }
    except (IndexError, TypeError, ValueError, FloatingPointError) as error:
        return {
            **base,
            "reason": str(error),
        }


def assess_rec709_vector_color_reference(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    max_samples: int = 200_000,
) -> Dict[str, Any]:
    """Report RGB-direction drift from a Rec.709 luminance-vector reference.

    A luminance-vector stretch scales the three channels of each pixel by one
    shared factor, so normalized RGB direction is the report-only color anchor.
    Clipped and near-black samples are excluded because no invertible vector
    reference exists there. This diagnostic never changes candidate ranking.
    """
    try:
        source_array = np.asarray(baseline)
        candidate_array = np.asarray(candidate)
        if source_array.shape != candidate_array.shape:
            raise ValueError(
                "shape mismatch: "
                f"baseline={source_array.shape}, candidate={candidate_array.shape}"
            )
        source_sample, source_sampling = _stage7_rgb_triplet_sample(
            source_array,
            max_samples=max_samples,
        )
        candidate_sample, candidate_sampling = _stage7_rgb_triplet_sample(
            candidate_array,
            max_samples=max_samples,
        )
        if (
            source_sampling["spatial_shape"]
            != candidate_sampling["spatial_shape"]
            or source_sampling["sample_stride"]
            != candidate_sampling["sample_stride"]
            or source_sample.shape != candidate_sample.shape
        ):
            raise ValueError("source/candidate sampling grids do not match")
        if source_sample.shape[0] < 64:
            raise ValueError("too few pixels for vector color reference")
        source_sample = source_sample.astype(np.float64, copy=False)
        candidate_sample = candidate_sample.astype(
            np.float64,
            copy=False,
        )
        if not np.all(np.isfinite(source_sample)) or not np.all(
            np.isfinite(candidate_sample)
        ):
            raise ValueError("vector color samples contain non-finite values")

        rec709 = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)
        source_luminance = source_sample @ rec709
        candidate_luminance = candidate_sample @ rec709
        positive_source = source_luminance[source_luminance > 1e-6]
        if positive_source.size < 64:
            raise ValueError("insufficient positive luminance support")
        signal_floor = max(
            1e-5,
            float(np.percentile(positive_source, 10.0)),
        )
        signal_support = (
            (source_luminance >= signal_floor)
            & (candidate_luminance > 1e-6)
        )
        if int(np.count_nonzero(signal_support)) < 64:
            raise ValueError("insufficient signal support for vector color reference")

        luminance_scale = np.divide(
            candidate_luminance,
            source_luminance,
            out=np.zeros_like(candidate_luminance),
            where=source_luminance > 1e-12,
        )
        candidate_peak = np.max(candidate_sample, axis=1)
        vector_reference_peak = np.max(source_sample, axis=1) * luminance_scale
        support = (
            signal_support
            & (candidate_peak < 0.995)
            & (vector_reference_peak < 0.995)
        )
        support_count = int(np.count_nonzero(support))
        if support_count < 64:
            raise ValueError("too few unclipped samples for vector color reference")

        source_supported = source_sample[support]
        candidate_supported = candidate_sample[support]
        source_chromaticity = source_supported / np.maximum(
            np.sum(source_supported, axis=1, keepdims=True),
            1e-12,
        )
        candidate_chromaticity = candidate_supported / np.maximum(
            np.sum(candidate_supported, axis=1, keepdims=True),
            1e-12,
        )
        chromaticity_error = 0.5 * np.sum(
            np.abs(candidate_chromaticity - source_chromaticity),
            axis=1,
        )
        signal_count = max(int(np.count_nonzero(signal_support)), 1)
        supported_scales = luminance_scale[support]
        return {
            "schema": "starun.stage7-color-vector-reference.v1",
            "status": "available",
            "role": "report_only",
            "enforced": False,
            "participates_in_selection": False,
            "reference_model": "rec709_luminance_scalar_rgb_vector",
            "ideal_chromaticity_error": 0.0,
            "metrics": {
                "chromaticity_l1_half_median": float(
                    np.median(chromaticity_error)
                ),
                "chromaticity_l1_half_p95": float(
                    np.percentile(chromaticity_error, 95.0)
                ),
                "chromaticity_l1_half_p99": float(
                    np.percentile(chromaticity_error, 99.0)
                ),
                "luminance_scale_p50": float(np.median(supported_scales)),
                "candidate_clip_exclusion_ratio": float(
                    np.count_nonzero(signal_support & (candidate_peak >= 0.995))
                    / signal_count
                ),
                "vector_gamut_exclusion_ratio": float(
                    np.count_nonzero(
                        signal_support & (vector_reference_peak >= 0.995)
                    )
                    / signal_count
                ),
                "support_coverage": float(support_count / source_sample.shape[0]),
                "signal_floor": signal_floor,
            },
            "sample_layout": "spatial_rgb_triplets",
            "sample_pixel_count": int(source_sample.shape[0]),
            "support_pixel_count": support_count,
            "source_layout": source_sampling["source_layout"],
            "sample_stride": source_sampling["sample_stride"],
            "sample_stride_unit": "spatial_pixels",
        }
    except (IndexError, TypeError, ValueError, FloatingPointError) as error:
        return {
            "schema": "starun.stage7-color-vector-reference.v1",
            "status": "unavailable",
            "role": "report_only",
            "enforced": False,
            "participates_in_selection": False,
            "reference_model": "rec709_luminance_scalar_rgb_vector",
            "reason": str(error),
        }


def assess_closed_form_mtf_conformance(
    params: Dict[str, Any],
    actual_p50: Any,
    *,
    relative_error_max: float = 0.05,
    absolute_error_max: float = 0.005,
) -> Dict[str, Any]:
    """Verify that a linked-MTF candidate matches its closed-form prediction."""
    limits = {
        "relative_error_max": _bounded(relative_error_max, 0.05, 0.01, 0.25),
        "absolute_error_max": _bounded(
            absolute_error_max,
            0.005,
            0.0001,
            0.03,
        ),
    }
    try:
        shadows = float(params["mtf_shadows"])
        midtones = float(params["mtf_midtones"])
        highlights = float(params.get("mtf_highlights", 1.0))
        source_p50 = float(params["source_background"])
        target_p50 = float(params["target_background"])
        measured_p50 = float(actual_p50)
        if not all(
            np.isfinite(item)
            for item in (
                shadows,
                midtones,
                highlights,
                source_p50,
                target_p50,
                measured_p50,
            )
        ):
            raise ValueError("closed-form MTF conformance inputs must be finite")

        normalized_source = (source_p50 - shadows) / max(
            highlights - shadows,
            1e-12,
        )
        expected_midpoint = solve_linked_mtf_midpoint(
            normalized_source,
            target_p50,
        )
        predicted_p50 = linked_mtf_sample(
            source_p50,
            shadows,
            midtones,
            highlights,
        )
        absolute_error = abs(measured_p50 - predicted_p50)
        relative_error = absolute_error / max(predicted_p50, 1e-6)
        effective_tolerance = max(
            limits["absolute_error_max"],
            predicted_p50 * limits["relative_error_max"],
        )
        midpoint_tolerance = max(5e-6, abs(expected_midpoint) * 1e-4)
        midpoint_error = abs(midtones - expected_midpoint)
        issues = []
        if midpoint_error > midpoint_tolerance:
            issues.append(
                "closed_form_mtf_midpoint_error "
                f"{midpoint_error:.8f}>{midpoint_tolerance:.8f}"
            )
        if absolute_error > effective_tolerance:
            issues.append(
                "closed_form_mtf_p50_error "
                f"{absolute_error:.6f}>{effective_tolerance:.6f} "
                f"(actual={measured_p50:.6f}, predicted={predicted_p50:.6f})"
            )
        return {
            "status": "ok" if not issues else "rejected",
            "accepted": not issues,
            "role": "reference_anchor",
            "method": "closed_form_linked_mtf",
            "issues": issues,
            "limits": limits,
            "metrics": {
                "source_p50": source_p50,
                "target_p50": target_p50,
                "predicted_p50": predicted_p50,
                "actual_p50": measured_p50,
                "absolute_error": absolute_error,
                "relative_error": relative_error,
                "effective_tolerance": effective_tolerance,
                "expected_midpoint": expected_midpoint,
                "actual_midpoint": midtones,
                "midpoint_error": midpoint_error,
                "midpoint_tolerance": midpoint_tolerance,
            },
        }
    except (KeyError, TypeError, ValueError, FloatingPointError) as error:
        return {
            "status": "unavailable",
            "accepted": False,
            "role": "reference_anchor",
            "method": "closed_form_linked_mtf",
            "issues": ["closed_form_mtf_reference_unavailable"],
            "limits": limits,
            "metrics": {},
            "reason": str(error),
        }


def _rank_normalized_gray(gray: np.ndarray) -> np.ndarray:
    """Map luminance to approximate percentile ranks.

    A global monotonic stretch preserves these ranks, so local changes measured
    after this mapping describe structural edits instead of simple brightness
    amplification.
    """
    values = np.asarray(gray, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size < 64:
        raise ValueError("too few finite pixels for rank normalization")
    if finite.size > 1_000_000:
        sample_step = int(np.ceil(finite.size / 1_000_000.0))
        finite = finite[::sample_step]
    levels = np.linspace(0.0, 1.0, 129, dtype=np.float32)
    anchors = np.quantile(finite, levels).astype(np.float32)
    unique_anchors, unique_indices = np.unique(anchors, return_index=True)
    if unique_anchors.size < 4:
        raise ValueError("insufficient luminance range for rank normalization")
    unique_levels = levels[unique_indices]
    ranked = np.interp(
        values.reshape(-1),
        unique_anchors,
        unique_levels,
    ).reshape(values.shape)
    return np.asarray(ranked, dtype=np.float32)


def assess_starless_structure_growth(
    baseline: np.ndarray,
    candidate: np.ndarray,
    starmask: Optional[np.ndarray],
    cfg: Any,
) -> Dict[str, Any]:
    """Gate star-like structural growth in a stretched Starless image.

    The comparison is performed on luminance percentile-rank maps and only in
    regions identified by the Stage 6 starmask. This makes the diagnostic
    insensitive to an ordinary global monotonic stretch while retaining
    sensitivity to newly enlarged residuals or halos around removed stars.
    """
    if not bool(getattr(cfg, "stage7_starless_structure_gate_enabled", True)):
        return {
            "status": "disabled",
            "accepted": True,
            "issues": [],
            "risk_score": 0.0,
            "metrics": {},
        }
    if starmask is None:
        return {
            "status": "unavailable",
            "accepted": True,
            "issues": [],
            "risk_score": 0.0,
            "metrics": {},
            "reason": "starmask unavailable",
        }

    try:
        source_rgb = _stage7_rgb_float_image(np.asarray(baseline), max_side=1024)
        candidate_rgb = _stage7_rgb_float_image(np.asarray(candidate), max_side=1024)
        starmask_rgb = _stage7_rgb_float_image(np.asarray(starmask), max_side=1024)
        if source_rgb.shape != candidate_rgb.shape or source_rgb.shape != starmask_rgb.shape:
            raise ValueError(
                "shape mismatch: "
                f"baseline={source_rgb.shape}, candidate={candidate_rgb.shape}, "
                f"starmask={starmask_rgb.shape}"
            )

        source_gray = (
            0.2126 * source_rgb[0]
            + 0.7152 * source_rgb[1]
            + 0.0722 * source_rgb[2]
        ).astype(np.float32)
        candidate_gray = (
            0.2126 * candidate_rgb[0]
            + 0.7152 * candidate_rgb[1]
            + 0.0722 * candidate_rgb[2]
        ).astype(np.float32)
        starmask_gray = (
            0.2126 * starmask_rgb[0]
            + 0.7152 * starmask_rgb[1]
            + 0.0722 * starmask_rgb[2]
        ).astype(np.float32)

        mask_floor = float(np.quantile(starmask_gray, 0.50))
        mask_signal = np.clip(starmask_gray - mask_floor, 0.0, None)
        positive = mask_signal[mask_signal > 0.0]
        if positive.size < 8:
            raise ValueError("starmask contains too little positive signal")
        mask_scale = float(np.quantile(positive, 0.995))
        if not np.isfinite(mask_scale) or mask_scale <= 1e-7:
            raise ValueError("starmask signal scale is invalid")
        mask_weight = np.clip(mask_signal / mask_scale, 0.0, 1.0)
        seed = mask_weight >= 0.10
        if int(np.count_nonzero(seed)) < 4:
            seed_threshold = float(np.quantile(mask_weight[mask_weight > 0.0], 0.75))
            seed = mask_weight >= max(seed_threshold, 1e-4)

        halo_weight = seed.astype(np.float32)
        for _ in range(3):
            halo_weight = _box_blur_gray(halo_weight)
        support_weight = np.maximum(mask_weight, np.clip(halo_weight * 4.0, 0.0, 1.0))
        support = support_weight > 0.02
        if int(np.count_nonzero(support)) < 16:
            raise ValueError("starmask-local support contains too few samples")

        source_rank = _rank_normalized_gray(source_gray)
        candidate_rank = _rank_normalized_gray(candidate_gray)
        rank_delta = candidate_rank - source_rank
        absolute_drift_p95 = float(np.quantile(np.abs(rank_delta[support]), 0.95))
        brightening_p95 = float(
            np.quantile(np.clip(rank_delta[support], 0.0, None), 0.95)
        )

        source_detail = np.abs(source_rank - _box_blur_gray(source_rank))
        candidate_detail = np.abs(candidate_rank - _box_blur_gray(candidate_rank))
        weights = support_weight[support]
        weight_sum = max(float(np.sum(weights)), 1e-6)
        source_detail_level = float(np.sum(source_detail[support] * weights) / weight_sum)
        candidate_detail_level = float(
            np.sum(candidate_detail[support] * weights) / weight_sum
        )
        detail_delta = candidate_detail_level - source_detail_level
        detail_ratio = candidate_detail_level / max(source_detail_level, 0.002)

        drift_max = _bounded(
            getattr(cfg, "stage7_starless_masked_rank_drift_p95_max", 0.18),
            0.18,
            0.02,
            0.50,
        )
        detail_ratio_max = _bounded(
            getattr(
                cfg,
                "stage7_starless_halo_detail_growth_ratio_max",
                1.60,
            ),
            1.60,
            1.05,
            4.0,
        )
        detail_delta_min = _bounded(
            getattr(cfg, "stage7_starless_halo_detail_delta_min", 0.010),
            0.010,
            0.001,
            0.10,
        )
        issues = []
        advisories = []
        quality_gates: Dict[str, Dict[str, Any]] = {}
        drift_gate = stage7_quality.stage7_9_upper_quality_gate(
            cfg,
            value=absolute_drift_p95,
            accepted_limit=drift_max,
        )
        quality_gates["starless_masked_rank_drift_p95"] = drift_gate
        if drift_gate["hard_failed"]:
            issues.append(
                "starless_masked_rank_drift_p95 "
                f"{absolute_drift_p95:.3f}>{drift_max:.3f}"
            )
        elif drift_gate["advisory"]:
            advisories.append(
                "starless_masked_rank_drift_p95 "
                f"{absolute_drift_p95:.3f}>{drift_max:.3f} advisory"
            )
        detail_gate = stage7_quality.stage7_9_upper_quality_gate(
            cfg,
            value=detail_ratio,
            accepted_limit=detail_ratio_max,
        )
        quality_gates["starless_halo_detail_growth"] = detail_gate
        if detail_delta > detail_delta_min and detail_gate["hard_failed"]:
            issues.append(
                "starless_halo_detail_growth "
                f"{detail_ratio:.3f}>{detail_ratio_max:.3f} "
                f"(delta={detail_delta:.4f}>{detail_delta_min:.4f})"
            )
        elif detail_delta > detail_delta_min and detail_gate["advisory"]:
            advisories.append(
                "starless_halo_detail_growth "
                f"{detail_ratio:.3f}>{detail_ratio_max:.3f} "
                f"(delta={detail_delta:.4f}>{detail_delta_min:.4f}) advisory"
            )

        risk_score = absolute_drift_p95 / max(drift_max, 1e-6) * 0.5
        if detail_delta > detail_delta_min:
            risk_score += max(0.0, detail_ratio - 1.0)
        metrics = {
            "masked_rank_drift_p95": absolute_drift_p95,
            "masked_rank_brightening_p95": brightening_p95,
            "source_halo_detail_level": source_detail_level,
            "candidate_halo_detail_level": candidate_detail_level,
            "halo_detail_growth_ratio": detail_ratio,
            "halo_detail_delta": detail_delta,
            "support_coverage": float(np.mean(support)),
            "starmask_seed_coverage": float(np.mean(seed)),
            "rank_drift_p95_max": drift_max,
            "halo_detail_growth_ratio_max": detail_ratio_max,
            "halo_detail_delta_min": detail_delta_min,
        }
        return {
            "status": "ok" if not issues else "rejected",
            "accepted": not issues,
            "issues": issues,
            "advisories": advisories,
            "quality_gates": quality_gates,
            "risk_score": float(risk_score),
            "metrics": metrics,
        }
    except (IndexError, TypeError, ValueError, FloatingPointError) as error:
        return {
            "status": "unavailable",
            "accepted": True,
            "issues": [],
            "risk_score": 0.0,
            "metrics": {},
            "reason": str(error),
        }


def _conditional_profile_mask(
    masks: Optional[Dict[str, Any]],
    name: str,
    shape: tuple[int, int],
) -> Optional[np.ndarray]:
    if not isinstance(masks, dict) or masks.get(name) is None:
        return None
    values = np.asarray(masks[name], dtype=np.float32)
    if values.ndim != 2 or tuple(values.shape) != tuple(shape):
        return None
    if not np.all(np.isfinite(values)):
        return None
    return np.clip(values, 0.0, 1.0)


def build_conditional_stretch_source_profile(
    image: np.ndarray,
    frozen_masks: Optional[Dict[str, Any]],
    background_sampling: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Measure source-domain anchors for conditional Stage 7 curves.

    Rec.709 luminance is used only to measure the frozen background and galaxy
    support.  Production pixels are later transformed with one common RGB LUT.
    """
    base: Dict[str, Any] = {
        "schema": CONDITIONAL_SOURCE_PROFILE_SCHEMA,
        "status": "unavailable",
        "measurement_domain": "source_rec709_luminance",
        "application_domain": "linked_rgb_common_curve",
        "luminance_weights": [0.2126, 0.7152, 0.0722],
        "candidate_independent": True,
    }
    try:
        rgb = _stage7_rgb_float_fullres(np.asarray(image))
        if not np.all(np.isfinite(rgb)):
            raise ValueError("source image contains non-finite pixels")
        gray = (
            0.2126 * rgb[0]
            + 0.7152 * rgb[1]
            + 0.0722 * rgb[2]
        ).astype(np.float32)
        if min(gray.shape) < 5 or gray.size < 64:
            raise ValueError("source image is too small for a stretch profile")

        background_weight = _conditional_profile_mask(
            frozen_masks,
            "background_mask",
            tuple(gray.shape),
        )
        if background_weight is None:
            raise ValueError("frozen background mask unavailable")
        for signal_name in (
            "core_mask",
            "nebula_mask",
            "faint_nebula_mask",
            "galaxy_signal_mask",
        ):
            signal_weight = _conditional_profile_mask(
                frozen_masks,
                signal_name,
                tuple(gray.shape),
            )
            if signal_weight is not None:
                background_weight *= 1.0 - signal_weight
        background_support = background_weight > 0.50
        background_count = int(np.count_nonzero(background_support))
        if background_count < 64:
            raise ValueError("frozen signal-excluded background has fewer than 64 pixels")

        sampling = (
            dict(background_sampling)
            if isinstance(background_sampling, dict)
            else {}
        )
        sampling_coverage = float(
            sampling.get(
                "coverage_gt_0_50",
                float(np.mean(background_support)),
            )
            or 0.0
        )
        trusted_background = bool(
            sampling.get("status") == "available"
            and sampling.get("candidate_independent") is True
            and sampling_coverage > 0.01
            and float(np.mean(background_support)) > 0.01
            and background_count >= 64
        )

        background_values = gray[background_support]
        background_median = float(np.median(background_values))
        background_mad = float(
            np.median(np.abs(background_values - background_median))
        )
        background_sigma = float(1.4826 * background_mad)
        p99, p99_5, p99_9 = (
            float(value)
            for value in np.percentile(gray, [99.0, 99.5, 99.9])
        )

        max_radius = max(1, (min(gray.shape) - 1) // 2)
        extended_radius = min(
            max_radius,
            max(1, min(24, int(round(min(gray.shape) / 110.0)))),
        )
        extended_gray = _box_mean_gray_at_radius(gray, extended_radius)
        extended_background_values = extended_gray[background_support]
        extended_background_median = float(
            np.median(extended_background_values)
        )
        extended_background_mad = float(
            np.median(
                np.abs(
                    extended_background_values - extended_background_median
                )
            )
        )
        extended_background_sigma = float(1.4826 * extended_background_mad)

        subject_weight = _conditional_profile_mask(
            frozen_masks,
            "galaxy_signal_mask",
            tuple(gray.shape),
        )
        subject_input_support = (
            subject_weight > 0.12
            if subject_weight is not None
            else np.zeros_like(gray, dtype=bool)
        )
        subject_input_count = int(np.count_nonzero(subject_input_support))
        subject_signal_floor: Optional[float] = None
        subject_measurement_method = "galaxy_signal_mask_unavailable"
        subject_measurement_support = np.zeros_like(gray, dtype=bool)
        if subject_input_count >= 64:
            subject_signal_floor = float(
                extended_background_median
                + max(1.25 * extended_background_sigma, 1e-7)
            )
            source_signal_support = (
                subject_input_support
                & (extended_gray >= subject_signal_floor)
            )
            if int(np.count_nonzero(source_signal_support)) >= 64:
                subject_measurement_support = source_signal_support
                subject_measurement_method = (
                    "frozen_galaxy_mask_and_extended_signal_v1"
                )
            else:
                subject_measurement_support = subject_input_support
                subject_measurement_method = "frozen_galaxy_mask_fallback_v1"

        subject_count = int(np.count_nonzero(subject_measurement_support))
        subject_p50: Optional[float] = None
        subject_p75: Optional[float] = None
        subject_p90: Optional[float] = None
        extended_subject_p75: Optional[float] = None
        if subject_count >= 64:
            subject_values = gray[subject_measurement_support]
            subject_p50, subject_p75, subject_p90 = (
                float(value)
                for value in np.percentile(subject_values, [50.0, 75.0, 90.0])
            )
            extended_subject_p75 = float(
                np.percentile(extended_gray[subject_measurement_support], 75.0)
            )

        faint_signal_contrast = max(
            0.0,
            (
                extended_subject_p75
                if extended_subject_p75 is not None
                else float(np.percentile(extended_gray, 99.0))
            )
            - extended_background_median,
        )
        faint_signal_snr_proxy = float(
            faint_signal_contrast / max(extended_background_sigma, 1e-9)
        )
        noise_regime = (
            "low_snr_proxy"
            if faint_signal_snr_proxy < 3.5
            else "medium_snr_proxy"
            if faint_signal_snr_proxy < 8.0
            else "high_snr_proxy"
        )

        galaxy_sampling = sampling.get("galaxy_signal_exclusion") or {}
        trusted_galaxy_roi = bool(
            trusted_background
            and isinstance(galaxy_sampling, dict)
            and galaxy_sampling.get("applicable") is True
            and galaxy_sampling.get("available") is True
            and subject_count >= 64
        )
        return {
            **base,
            "status": "available",
            "spatial_shape": [int(gray.shape[0]), int(gray.shape[1])],
            "background_available": True,
            "trusted_background": trusted_background,
            "background_sample_count": background_count,
            "background_coverage": float(np.mean(background_support)),
            "background_median": background_median,
            "background_mad": background_mad,
            "background_sigma": background_sigma,
            "p99": p99,
            "p99_5": p99_5,
            "p99_9": p99_9,
            "extended_filter": {
                "method": "reflect_box_mean_v1",
                "radius": extended_radius,
            },
            "extended_background_median": extended_background_median,
            "extended_background_sigma": extended_background_sigma,
            "subject_mask_input_available": subject_input_count >= 64,
            "subject_mask_available": subject_count >= 64,
            "trusted_galaxy_roi": trusted_galaxy_roi,
            "subject_input_sample_count": subject_input_count,
            "subject_sample_count": subject_count,
            "subject_input_coverage": float(np.mean(subject_input_support)),
            "subject_measurement_coverage": float(
                np.mean(subject_measurement_support)
            ),
            "subject_measurement_method": subject_measurement_method,
            "subject_signal_floor": subject_signal_floor,
            "subject_p50": subject_p50,
            "subject_p75": subject_p75,
            "subject_p90": subject_p90,
            "extended_subject_p75": extended_subject_p75,
            "faint_signal_contrast": faint_signal_contrast,
            "faint_signal_snr_proxy": faint_signal_snr_proxy,
            "stretch_noise_regime": noise_regime,
            "physical_snr": False,
            "sampling_contract": {
                "status": sampling.get("status", "unavailable"),
                "method": sampling.get("method"),
                "candidate_independent": bool(
                    sampling.get("candidate_independent", False)
                ),
                "coverage_gt_0_50": sampling_coverage,
                "galaxy_signal_exclusion": {
                    "applicable": bool(galaxy_sampling.get("applicable", False))
                    if isinstance(galaxy_sampling, dict)
                    else False,
                    "available": bool(galaxy_sampling.get("available", False))
                    if isinstance(galaxy_sampling, dict)
                    else False,
                    "coverage": galaxy_sampling.get("coverage")
                    if isinstance(galaxy_sampling, dict)
                    else None,
                    "reason": galaxy_sampling.get("reason")
                    if isinstance(galaxy_sampling, dict)
                    else None,
                },
            },
        }
    except (IndexError, TypeError, ValueError, FloatingPointError) as error:
        return {
            **base,
            "reason": str(error),
        }


def _conditional_mtf_curve(values: np.ndarray, midtones: float) -> np.ndarray:
    midpoint = float(np.clip(midtones, 1e-5, 1.0 - 1e-5))
    source = np.asarray(values, dtype=np.float64)
    denominator = (2.0 * midpoint - 1.0) * source - midpoint
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        mapped = ((midpoint - 1.0) * source) / denominator
    return np.clip(mapped, 0.0, 1.0)


def _conditional_lut_digest(lut: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(lut, dtype="<f4"))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _conditional_lut_contract(
    lut: np.ndarray,
    *,
    max_derivative: float,
) -> Dict[str, Any]:
    values = np.asarray(lut, dtype=np.float64)
    issues = []
    if values.ndim != 1 or values.size != CONDITIONAL_STRETCH_LUT_SIZE:
        issues.append("lut_size_invalid")
    if not np.all(np.isfinite(values)):
        issues.append("lut_non_finite")
    if issues:
        return {
            "status": "rejected",
            "accepted": False,
            "issues": issues,
            "size": int(values.size) if values.ndim == 1 else None,
        }
    differences = np.diff(values)
    minimum_step = float(np.min(differences))
    derivative_max = float(np.max(differences) * (values.size - 1))
    if minimum_step < -1e-10:
        issues.append("lut_not_monotonic")
    if abs(float(values[0])) > 2e-6 or abs(float(values[-1]) - 1.0) > 2e-6:
        issues.append("lut_endpoints_invalid")
    if derivative_max > float(max_derivative) + 1e-6:
        issues.append("lut_derivative_exceeds_limit")
    return {
        "status": "ok" if not issues else "rejected",
        "accepted": not issues,
        "issues": issues,
        "size": int(values.size),
        "dtype_digest": "float32-little-endian",
        "sha256": _conditional_lut_digest(values),
        "endpoint_zero": float(values[0]),
        "endpoint_one": float(values[-1]),
        "minimum_step": minimum_step,
        "maximum_derivative": derivative_max,
        "maximum_derivative_limit": float(max_derivative),
        "monotonic": minimum_step >= -1e-10,
    }


def _display90_quantile_key(percentile: float) -> str:
    value = float(percentile)
    if value.is_integer():
        return f"p{int(value)}"
    return f"p{str(value).replace('.', '_')}"


def _display90_quantiles(rgb: np.ndarray) -> Dict[str, Dict[str, float]]:
    values = np.asarray(rgb, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != 3:
        raise ValueError(f"expected RGB CHW array, got shape={values.shape}")
    luminance = (
        values[0] * np.float32(0.2126)
        + values[1] * np.float32(0.7152)
        + values[2] * np.float32(0.0722)
    )
    rgb_values = np.percentile(
        values.reshape(-1),
        DISPLAY90_REPORT_PERCENTILES,
    )
    luma_values = np.percentile(
        luminance.reshape(-1),
        DISPLAY90_REPORT_PERCENTILES,
    )
    return {
        "rgb_flat": {
            _display90_quantile_key(percentile): float(value)
            for percentile, value in zip(
                DISPLAY90_REPORT_PERCENTILES,
                rgb_values,
            )
        },
        "rec709_luminance": {
            _display90_quantile_key(percentile): float(value)
            for percentile, value in zip(
                DISPLAY90_REPORT_PERCENTILES,
                luma_values,
            )
        },
    }


def _display90_lut(
    display_curve: Dict[str, Any],
    strength: float,
) -> np.ndarray:
    if (
        not isinstance(display_curve, dict)
        or display_curve.get("schema") != ui_preview.LINKED_DISPLAY_CURVE_SCHEMA
        or display_curve.get("status") != "ok"
        or display_curve.get("accepted") is not True
    ):
        raise ValueError("GUI linked display curve contract is unavailable")
    black = float(display_curve["black"])
    white = float(display_curve["white"])
    gamma = float(display_curve["gamma"])
    blend = float(strength)
    if not all(np.isfinite(value) for value in (black, white, gamma, blend)):
        raise ValueError("Display90 curve parameters must be finite")
    if not 0.0 <= black < white <= 1.0:
        raise ValueError("Display90 black/white points are outside the unit domain")
    if not 0.20 <= gamma <= 1.0:
        raise ValueError("Display90 gamma is outside the GUI contract")
    if not DISPLAY90_STRENGTH_MIN <= blend <= DISPLAY90_STRENGTH_MAX:
        raise ValueError(
            "Display90 strength must be within "
            f"{DISPLAY90_STRENGTH_MIN:.2f}..{DISPLAY90_STRENGTH_MAX:.2f}"
        )
    grid = np.linspace(
        0.0,
        1.0,
        CONDITIONAL_STRETCH_LUT_SIZE,
        dtype=np.float64,
    )
    display = np.power(
        np.clip((grid - black) / (white - black), 0.0, 1.0),
        gamma,
    )
    # The canonical LUT is float32 because that exact byte representation is
    # authenticated and reused by both Stage 7 and Stage 9.
    return np.asarray(
        np.clip((1.0 - blend) * grid + blend * display, 0.0, 1.0),
        dtype=np.float32,
    )


def _apply_authenticated_lut_rgb(
    rgb: np.ndarray,
    lut: np.ndarray,
) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float32)
    grid = np.linspace(0.0, 1.0, lut.size, dtype=np.float64)
    mapped = np.interp(
        np.clip(values.reshape(-1), 0.0, 1.0),
        grid,
        lut,
    ).reshape(values.shape)
    return np.asarray(
        np.clip(mapped, 0.0, CONDITIONAL_STRETCH_OUTPUT_HEADROOM),
        dtype=np.float32,
    )


def calibrate_display90_linked_lut(
    image: np.ndarray,
    display_curve: Dict[str, Any],
    *,
    strength: float,
    max_derivative: float,
) -> Dict[str, Any]:
    """Build the authenticated shared-RGB LUT derived from the GUI D curve."""

    try:
        derivative_limit = float(max_derivative)
        if not np.isfinite(derivative_limit) or derivative_limit <= 0.0:
            raise ValueError("Display90 derivative limit must be positive")
        lut = _display90_lut(display_curve, strength)
        lut_contract = _conditional_lut_contract(
            lut,
            max_derivative=derivative_limit,
        )
        if not lut_contract.get("accepted"):
            raise ValueError(
                "Display90 LUT failed validation: "
                + ",".join(lut_contract.get("issues") or [])
            )
        analysis_rgb, sampling = _stage7_rgb_analysis_grid(
            image,
            max_side=DISPLAY90_ANALYSIS_MAX_SIDE,
        )
        gui_reference = ui_preview.apply_linked_display_curve_contract(
            analysis_rgb,
            display_curve,
        )
        target_rgb = _apply_authenticated_lut_rgb(analysis_rgb, lut)
        source_quantiles = _display90_quantiles(analysis_rgb)
        gui_quantiles = _display90_quantiles(gui_reference)
        target_quantiles = _display90_quantiles(target_rgb)
        rgb_targets = target_quantiles["rgb_flat"]
        return {
            "schema": DISPLAY90_STRETCH_SCHEMA,
            "status": "ok",
            "method": "display90_linked_lut",
            "display_curve": dict(display_curve),
            "parameters": {
                "strength": float(strength),
                "strength_min": DISPLAY90_STRENGTH_MIN,
                "strength_max": DISPLAY90_STRENGTH_MAX,
                "max_derivative": derivative_limit,
                "lut_size": CONDITIONAL_STRETCH_LUT_SIZE,
                "output_headroom": CONDITIONAL_STRETCH_OUTPUT_HEADROOM,
                "formula": (
                    "(1-strength)*x + strength*"
                    "clip((x-black)/(white-black),0,1)^gamma"
                ),
            },
            "lut_contract": lut_contract,
            "analysis_sampling": sampling,
            "source_quantiles": source_quantiles,
            "gui_display_reference_quantiles": gui_quantiles,
            "d90_target_quantiles": target_quantiles,
            "predicted_p50": rgb_targets["p50"],
            "predicted_p90": rgb_targets["p90"],
            "predicted_p99": rgb_targets["p99"],
            "target_p50": rgb_targets["p50"],
            "target_p90": rgb_targets["p90"],
            "target_p99": rgb_targets["p99"],
            "calibrated_stretch": float(strength),
            "stretch_max": DISPLAY90_STRENGTH_MAX,
        }
    except (
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        FloatingPointError,
    ) as error:
        return {
            "schema": DISPLAY90_STRETCH_SCHEMA,
            "status": "unavailable",
            "method": "display90_linked_lut",
            "reason": str(error),
            "display_curve": (
                dict(display_curve) if isinstance(display_curve, dict) else None
            ),
        }


def rebuild_display90_linked_lut(
    calibration: Dict[str, Any],
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Rebuild and authenticate a Display90 LUT without trusting stored pixels."""

    if (
        not isinstance(calibration, dict)
        or calibration.get("schema") != DISPLAY90_STRETCH_SCHEMA
        or calibration.get("status") != "ok"
        or calibration.get("method") != "display90_linked_lut"
    ):
        raise ValueError("Display90 calibration is invalid")
    parameters = dict(calibration.get("parameters") or {})
    stored_contract = calibration.get("lut_contract")
    if (
        not isinstance(stored_contract, dict)
        or stored_contract.get("status") != "ok"
        or stored_contract.get("accepted") is not True
    ):
        raise ValueError("Display90 LUT contract is missing or rejected")
    if int(parameters.get("lut_size", 0)) != CONDITIONAL_STRETCH_LUT_SIZE:
        raise ValueError("Display90 LUT size contract mismatch")
    if not np.isclose(
        float(parameters.get("output_headroom", float("nan"))),
        CONDITIONAL_STRETCH_OUTPUT_HEADROOM,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Display90 output headroom contract mismatch")
    derivative_limit = float(parameters["max_derivative"])
    if not np.isfinite(derivative_limit) or derivative_limit <= 0.0:
        raise ValueError("Display90 derivative limit is invalid")
    lut = _display90_lut(
        dict(calibration.get("display_curve") or {}),
        float(parameters["strength"]),
    )
    contract = _conditional_lut_contract(
        lut,
        max_derivative=derivative_limit,
    )
    if not contract.get("accepted"):
        raise ValueError(
            "rebuilt Display90 LUT failed validation: "
            + ",".join(contract.get("issues") or [])
        )
    expected = str(stored_contract.get("sha256") or "")
    if not expected or contract.get("sha256") != expected:
        raise ValueError("Display90 LUT digest mismatch")
    if (
        int(stored_contract.get("size", 0)) != CONDITIONAL_STRETCH_LUT_SIZE
        or stored_contract.get("dtype_digest") != "float32-little-endian"
        or stored_contract.get("monotonic") is not True
    ):
        raise ValueError("Display90 LUT summary contract mismatch")
    for field in (
        "endpoint_zero",
        "endpoint_one",
        "minimum_step",
        "maximum_derivative",
        "maximum_derivative_limit",
    ):
        try:
            stored_value = float(stored_contract[field])
            rebuilt_value = float(contract[field])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Display90 LUT summary field is invalid: {field}"
            ) from error
        if not np.isclose(
            stored_value,
            rebuilt_value,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError(f"Display90 LUT summary mismatch: {field}")
    return lut, contract


def apply_display90_linked_rgb_stretch(
    image: np.ndarray,
    calibration: Dict[str, Any],
) -> np.ndarray:
    """Apply an authenticated Display90 LUT independently to R/G/B."""

    rgb = _stage7_rgb_float_fullres(np.asarray(image))
    lut, _contract = rebuild_display90_linked_lut(calibration)
    return _apply_authenticated_lut_rgb(rgb, lut)


def build_display90_gui_linked_reference(
    image: np.ndarray,
    calibration: Dict[str, Any],
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Rebuild the exact GUI linked-D reference from an authenticated contract.

    The production Display90 candidate intentionally uses one shared per-channel
    LUT.  Its safety reference is different: this function reproduces the GUI's
    existing Rec.709 luminance-gain display path without changing that candidate
    or its Stage 9 matched-domain transfer.
    """

    _lut, lut_contract = rebuild_display90_linked_lut(calibration)
    rgb = _stage7_rgb_float_fullres(np.asarray(image))
    reference = ui_preview.apply_linked_display_curve_contract(
        rgb,
        dict(calibration.get("display_curve") or {}),
    )
    values = np.asarray(reference, dtype=np.float32)
    if values.shape != rgb.shape or not np.all(np.isfinite(values)):
        raise ValueError("Display90 GUI linked reference is invalid")
    np.clip(values, 0.0, 1.0, out=values)
    return values, lut_contract


def assess_display90_curve_conformance(
    calibration: Dict[str, Any],
    actual_image: np.ndarray,
    *,
    relative_error_max: float = 0.05,
    absolute_error_max: float = 0.005,
) -> Dict[str, Any]:
    """Require measured P50/P90/P99 to agree with the authenticated LUT."""

    try:
        _lut, rebuilt_contract = rebuild_display90_linked_lut(calibration)
        actual_rgb, sampling = _stage7_rgb_analysis_grid(
            actual_image,
            max_side=DISPLAY90_ANALYSIS_MAX_SIDE,
        )
        actual_quantiles = _display90_quantiles(actual_rgb)
        expected_quantiles = dict(
            calibration.get("d90_target_quantiles") or {}
        )
        relative_limit = float(np.clip(relative_error_max, 0.0, 0.50))
        absolute_limit = float(np.clip(absolute_error_max, 0.0, 0.10))
        issues = []
        metrics: Dict[str, Any] = {}
        for domain in ("rgb_flat", "rec709_luminance"):
            expected_domain = dict(expected_quantiles.get(domain) or {})
            actual_domain = dict(actual_quantiles.get(domain) or {})
            domain_metrics: Dict[str, Any] = {}
            for percentile in DISPLAY90_CONFORMANCE_PERCENTILES:
                key = _display90_quantile_key(percentile)
                expected = float(expected_domain[key])
                actual = float(actual_domain[key])
                absolute_error = abs(actual - expected)
                relative_error = absolute_error / max(abs(expected), 1e-6)
                accepted = bool(
                    absolute_error <= absolute_limit
                    or relative_error <= relative_limit
                )
                domain_metrics[key] = {
                    "expected": expected,
                    "actual": actual,
                    "absolute_error": absolute_error,
                    "relative_error": relative_error,
                    "accepted": accepted,
                }
                if not accepted:
                    issues.append(
                        "display90_curve_"
                        f"{domain}_{key}_error "
                        f"abs={absolute_error:.6f}>{absolute_limit:.6f} "
                        f"rel={relative_error:.4f}>{relative_limit:.4f}"
                    )
            metrics[domain] = domain_metrics
        return {
            "schema": "starun.stage7-display90-conformance.v1",
            "status": "ok" if not issues else "rejected",
            "accepted": not issues,
            "issues": issues,
            "relative_error_max": relative_limit,
            "absolute_error_max": absolute_limit,
            "metrics": metrics,
            "analysis_sampling": sampling,
            "actual_quantiles": actual_quantiles,
            "lut_contract": rebuilt_contract,
        }
    except (
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        FloatingPointError,
    ) as error:
        return {
            "schema": "starun.stage7-display90-conformance.v1",
            "status": "rejected",
            "accepted": False,
            "issues": ["display90_curve_contract_invalid"],
            "reason": str(error),
            "metrics": {},
        }


# The following point-curve solvers are modified from the GPL-3.0-only
# deep-sky-processor implementation at CONDITIONAL_STRETCH_REFERENCE; the
# standard GHS evaluator is modified from CONDITIONAL_GHS_REFERENCE. This
# project applies the resulting curve as linked RGB and intentionally does not
# copy that project's shared-luminance gain or highlight soft-knee.
def _iterative_masked_mtf_lut(
    *,
    source_background: float,
    target_background: float,
    iterations: int,
) -> tuple[np.ndarray, Dict[str, Any]]:
    grid = np.linspace(
        0.0,
        1.0,
        CONDITIONAL_STRETCH_LUT_SIZE,
        dtype=np.float64,
    )
    curve = grid.copy()
    midpoint_path = []
    target_path = []
    for index in range(iterations):
        desired_background = float(
            source_background
            + (target_background - source_background)
            * (index + 1)
            / max(iterations, 1)
        )
        lower_midtones = 1e-5
        upper_midtones = 0.49999
        for _probe in range(36):
            midtones = (lower_midtones + upper_midtones) * 0.5
            mapped = _conditional_mtf_curve(curve, midtones)
            candidate = curve * curve + mapped * (1.0 - curve)
            mapped_background = float(
                np.interp(source_background, grid, candidate)
            )
            if mapped_background > desired_background:
                lower_midtones = midtones
            else:
                upper_midtones = midtones
        midtones = (lower_midtones + upper_midtones) * 0.5
        mapped = _conditional_mtf_curve(curve, midtones)
        curve = curve * curve + mapped * (1.0 - curve)
        midpoint_path.append(float(midtones))
        target_path.append(desired_background)
    curve = np.clip(curve, 0.0, 1.0)
    return curve, {
        "iterations": int(iterations),
        "mask": "inverse_current_intensity_point_curve",
        "midpoint_path": midpoint_path,
        "target_background_path": target_path,
        "mapped_background": float(
            np.interp(source_background, grid, curve)
        ),
    }


def _conditional_ghs_base_transform(
    distance: np.ndarray,
    stretch_factor: float,
    local_intensity: float,
) -> np.ndarray:
    values = np.asarray(distance, dtype=np.float64)
    factor = float(np.clip(stretch_factor, 0.0, 20.0))
    d_value = float(np.expm1(factor))
    b_value = float(local_intensity)
    if factor <= 1e-12:
        return values.copy()
    if abs(b_value + 1.0) <= 1e-8:
        return np.log1p(d_value * values)
    if abs(b_value) <= 1e-8:
        return -np.expm1(-d_value * values)
    if b_value < 0.0:
        exponent = (b_value + 1.0) / b_value
        return (
            1.0 - np.power(1.0 - b_value * d_value * values, exponent)
        ) / (d_value * (b_value + 1.0))
    return 1.0 - np.power(
        1.0 + b_value * d_value * values,
        -1.0 / b_value,
    )


def _conditional_ghs_base_derivative(
    distance: float,
    stretch_factor: float,
    local_intensity: float,
) -> float:
    factor = float(np.clip(stretch_factor, 0.0, 20.0))
    d_value = float(np.expm1(factor))
    b_value = float(local_intensity)
    value = float(distance)
    if factor <= 1e-12:
        return 1.0
    if abs(b_value + 1.0) <= 1e-8:
        return d_value / (1.0 + d_value * value)
    if abs(b_value) <= 1e-8:
        return d_value * float(np.exp(-d_value * value))
    if b_value < 0.0:
        return float(
            np.power(1.0 - b_value * d_value * value, 1.0 / b_value)
        )
    return float(
        d_value
        * np.power(
            1.0 + b_value * d_value * value,
            -(1.0 + b_value) / b_value,
        )
    )


def _conditional_ghs_curve(
    values: np.ndarray,
    *,
    stretch_factor: float,
    b: float,
    sp: float,
    lp: float,
    hp: float,
) -> np.ndarray:
    source = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    factor = float(np.clip(stretch_factor, 0.0, 20.0))
    if factor <= 1e-12:
        return source.copy()
    symmetry = float(np.clip(sp, 0.0, 1.0))
    shadow_protection = float(np.clip(lp, 0.0, symmetry))
    highlight_protection = float(np.clip(hp, symmetry, 1.0))
    local_intensity = float(np.clip(b, -10.0, 10.0))

    left_at_lp = -float(
        _conditional_ghs_base_transform(
            np.asarray(symmetry - shadow_protection),
            factor,
            local_intensity,
        )
    )
    left_slope = _conditional_ghs_base_derivative(
        symmetry - shadow_protection,
        factor,
        local_intensity,
    )
    right_at_hp = float(
        _conditional_ghs_base_transform(
            np.asarray(highlight_protection - symmetry),
            factor,
            local_intensity,
        )
    )
    right_slope = _conditional_ghs_base_derivative(
        highlight_protection - symmetry,
        factor,
        local_intensity,
    )

    raw = np.empty_like(source, dtype=np.float64)
    shadow_region = source < shadow_protection
    lower_region = (source >= shadow_protection) & (source < symmetry)
    upper_region = (source >= symmetry) & (source < highlight_protection)
    highlight_region = source >= highlight_protection
    raw[shadow_region] = left_at_lp + left_slope * (
        source[shadow_region] - shadow_protection
    )
    raw[lower_region] = -_conditional_ghs_base_transform(
        symmetry - source[lower_region],
        factor,
        local_intensity,
    )
    raw[upper_region] = _conditional_ghs_base_transform(
        source[upper_region] - symmetry,
        factor,
        local_intensity,
    )
    raw[highlight_region] = right_at_hp + right_slope * (
        source[highlight_region] - highlight_protection
    )

    raw_zero = (
        left_at_lp - left_slope * shadow_protection
        if shadow_protection > 0.0
        else -float(
            _conditional_ghs_base_transform(
                np.asarray(symmetry),
                factor,
                local_intensity,
            )
        )
    )
    raw_one = (
        right_at_hp + right_slope * (1.0 - highlight_protection)
        if highlight_protection < 1.0
        else float(
            _conditional_ghs_base_transform(
                np.asarray(1.0 - symmetry),
                factor,
                local_intensity,
            )
        )
    )
    span = raw_one - raw_zero
    if not np.isfinite(span) or span <= 1e-15:
        raise ValueError("GHS parameters produced a degenerate curve")
    return np.clip((raw - raw_zero) / span, 0.0, 1.0)


def _compose_conditional_background_adjustment(
    curve: np.ndarray,
    *,
    source_background: float,
    target_background: float,
) -> tuple[np.ndarray, Dict[str, Any]]:
    grid = np.linspace(0.0, 1.0, curve.size, dtype=np.float64)
    intermediate_background = float(
        np.interp(source_background, grid, curve)
    )
    midpoint = solve_linked_mtf_midpoint(
        intermediate_background,
        target_background,
    )
    return _conditional_mtf_curve(curve, midpoint), {
        "method": "mtf_background_anchor",
        "midtones": midpoint,
        "source_background": intermediate_background,
        "target_background": float(target_background),
    }


def _dual_stage_mtf_ghs_lut(
    *,
    source_background: float,
    background_sigma: float,
    source_subject_p90: float,
    source_p99_9: float,
    target_background: float,
    target_subject_p90: float,
    ghs_b: float,
    ghs_d_min: float,
    ghs_d_max: float,
    ghs_search_steps: int,
    max_derivative: float,
) -> tuple[np.ndarray, Dict[str, Any]]:
    grid = np.linspace(
        0.0,
        1.0,
        CONDITIONAL_STRETCH_LUT_SIZE,
        dtype=np.float64,
    )
    highlight_anchor = float(
        np.clip(
            max(
                source_p99_9,
                source_subject_p90 + max(background_sigma * 3.0, 1e-5),
            ),
            source_subject_p90,
            0.999999,
        )
    )
    base_midtones = solve_linked_mtf_midpoint(
        source_background,
        target_background,
    )
    base_curve = _conditional_mtf_curve(grid, base_midtones)
    transformed_background = float(
        np.interp(source_background, grid, base_curve)
    )
    transformed_highlight = float(
        np.interp(highlight_anchor, grid, base_curve)
    )
    soft_derivative_limit = min(float(max_derivative), 4800.0)
    best_key: Optional[tuple[float, float, float]] = None
    best_curve: Optional[np.ndarray] = None
    best_factor = 0.0
    best_derivative = 0.0
    best_mapped_subject = 0.0
    best_adjustment: Dict[str, Any] = {}
    for stretch_factor in np.linspace(
        ghs_d_min,
        ghs_d_max,
        ghs_search_steps,
        dtype=np.float64,
    ):
        candidate = _conditional_ghs_curve(
            base_curve,
            stretch_factor=float(stretch_factor),
            b=ghs_b,
            sp=transformed_background,
            lp=0.0,
            hp=transformed_highlight,
        )
        candidate, adjustment = _compose_conditional_background_adjustment(
            candidate,
            source_background=source_background,
            target_background=target_background,
        )
        derivative_max = float(
            np.max(np.diff(candidate)) * (candidate.size - 1)
        )
        if derivative_max > float(max_derivative) + 1e-6:
            continue
        mapped_subject = float(
            np.interp(source_subject_p90, grid, candidate)
        )
        subject_error = abs(mapped_subject - target_subject_p90)
        slope_penalty = max(
            0.0,
            derivative_max - soft_derivative_limit,
        ) / 1000.0
        ranking = (
            subject_error + slope_penalty,
            subject_error,
            float(stretch_factor),
        )
        if best_key is None or ranking < best_key:
            best_key = ranking
            best_curve = candidate
            best_factor = float(stretch_factor)
            best_derivative = derivative_max
            best_mapped_subject = mapped_subject
            best_adjustment = adjustment
    if best_curve is None:
        raise ValueError("no Dual-stage MTF+GHS curve passed the derivative limit")
    mapped_background = float(
        np.interp(source_background, grid, best_curve)
    )
    return np.clip(best_curve, 0.0, 1.0), {
        "base_mtf_midtones": float(base_midtones),
        "source_background": float(source_background),
        "target_background": float(target_background),
        "mapped_background": mapped_background,
        "background_target_error": abs(
            mapped_background - float(target_background)
        ),
        "source_subject_p90": float(source_subject_p90),
        "target_subject_p90": float(target_subject_p90),
        "mapped_subject_p90": best_mapped_subject,
        "source_highlight_anchor": highlight_anchor,
        "ghs_stretch_factor": best_factor,
        "ghs_stretch_factor_semantics": "ln(D+1)",
        "ghs_b": float(ghs_b),
        "ghs_sp": transformed_background,
        "ghs_lp": 0.0,
        "ghs_hp": transformed_highlight,
        "ghs_search_range": [float(ghs_d_min), float(ghs_d_max)],
        "ghs_search_steps": int(ghs_search_steps),
        "maximum_derivative": best_derivative,
        "background_adjustment": best_adjustment,
    }


def _conditional_global_predictions(
    lut: np.ndarray,
    source_p50: float,
    source_p99: float,
) -> Dict[str, float]:
    grid = np.linspace(0.0, 1.0, lut.size, dtype=np.float64)
    return {
        "predicted_global_p50": float(np.interp(source_p50, grid, lut)),
        "predicted_global_p99": float(np.interp(source_p99, grid, lut)),
    }


def calibrate_iterative_masked_mtf(
    source_profile: Dict[str, Any],
    *,
    target_background: float,
    iterations: int,
    max_derivative: float,
    source_p50: float,
    source_p99: float,
) -> Dict[str, Any]:
    try:
        if source_profile.get("status") != "available":
            raise ValueError("conditional source profile unavailable")
        if source_profile.get("trusted_background") is not True:
            raise ValueError("trusted frozen background unavailable")
        source_background = float(source_profile["background_median"])
        target = float(target_background)
        resolved_iterations = max(8, min(32, int(iterations)))
        derivative_limit = _bounded(max_derivative, 5000.0, 250.0, 20000.0)
        global_p50 = float(source_p50)
        global_p99 = float(source_p99)
        if not all(
            np.isfinite(value)
            for value in (
                source_background,
                target,
                global_p50,
                global_p99,
            )
        ):
            raise ValueError("Iterative Masked MTF anchors must be finite")
        if not 1e-7 < source_background < target < 0.50:
            raise ValueError("Iterative Masked MTF background path is invalid")
        if not 0.0 < global_p50 < global_p99 <= 1.0:
            raise ValueError("global P50/P99 anchors are invalid")
        lut, resolved = _iterative_masked_mtf_lut(
            source_background=source_background,
            target_background=target,
            iterations=resolved_iterations,
        )
        if abs(float(resolved["mapped_background"]) - target) > 2e-5:
            raise ValueError(
                "Iterative Masked MTF did not attain the background anchor"
            )
        contract = _conditional_lut_contract(
            lut,
            max_derivative=derivative_limit,
        )
        if not contract.get("accepted"):
            raise ValueError(
                "Iterative Masked MTF LUT rejected: "
                + ",".join(contract.get("issues") or [])
            )
        return {
            "schema": CONDITIONAL_STRETCH_SCHEMA,
            "status": "ok",
            "method": "iterative_masked_mtf",
            "curve_application": "linked_rgb_common_curve",
            "source_reference": CONDITIONAL_STRETCH_REFERENCE,
            "source_license": CONDITIONAL_STRETCH_SOURCE_LICENSE,
            "source_profile": {
                "background_median": source_background,
            },
            "parameters": {
                "target_background": target,
                "iterations": resolved_iterations,
                "max_derivative": derivative_limit,
            },
            "resolved": resolved,
            "lut_contract": contract,
            **_conditional_global_predictions(lut, global_p50, global_p99),
        }
    except (KeyError, TypeError, ValueError, FloatingPointError) as error:
        return {
            "schema": CONDITIONAL_STRETCH_SCHEMA,
            "status": "unavailable",
            "method": "iterative_masked_mtf",
            "reason": str(error),
        }


def calibrate_dual_stage_mtf_ghs(
    source_profile: Dict[str, Any],
    *,
    target_background: float,
    target_subject_p90: float,
    ghs_b: float,
    ghs_d_min: float,
    ghs_d_max: float,
    ghs_search_steps: int,
    max_derivative: float,
    source_p50: float,
    source_p99: float,
) -> Dict[str, Any]:
    try:
        if source_profile.get("status") != "available":
            raise ValueError("conditional source profile unavailable")
        if source_profile.get("trusted_galaxy_roi") is not True:
            raise ValueError("trusted frozen galaxy ROI unavailable")
        source_background = float(source_profile["background_median"])
        background_sigma = float(source_profile["background_sigma"])
        source_subject_p90 = float(source_profile["subject_p90"])
        source_p99_9 = float(source_profile["p99_9"])
        target = float(target_background)
        subject_target = float(target_subject_p90)
        resolved_b = _bounded(ghs_b, 5.0, 2.0, 8.0)
        resolved_d_min = _bounded(ghs_d_min, 0.5, 0.10, 8.0)
        resolved_d_max = _bounded(ghs_d_max, 12.0, 0.50, 16.0)
        if resolved_d_max <= resolved_d_min:
            raise ValueError("Dual-stage GHS search range is invalid")
        resolved_steps = max(9, min(97, int(ghs_search_steps)))
        derivative_limit = _bounded(max_derivative, 5000.0, 250.0, 20000.0)
        global_p50 = float(source_p50)
        global_p99 = float(source_p99)
        if not all(
            np.isfinite(value)
            for value in (
                source_background,
                background_sigma,
                source_subject_p90,
                source_p99_9,
                target,
                subject_target,
                global_p50,
                global_p99,
            )
        ):
            raise ValueError("Dual-stage MTF+GHS anchors must be finite")
        if not 1e-7 < source_background < target < 0.50:
            raise ValueError("Dual-stage background path is invalid")
        if not source_background < source_subject_p90 < 1.0:
            raise ValueError("Dual-stage subject P90 anchor is invalid")
        if not target < subject_target < 0.60:
            raise ValueError("Dual-stage target subject P90 is invalid")
        if not 0.0 < global_p50 < global_p99 <= 1.0:
            raise ValueError("global P50/P99 anchors are invalid")
        lut, resolved = _dual_stage_mtf_ghs_lut(
            source_background=source_background,
            background_sigma=max(background_sigma, 0.0),
            source_subject_p90=source_subject_p90,
            source_p99_9=source_p99_9,
            target_background=target,
            target_subject_p90=subject_target,
            ghs_b=resolved_b,
            ghs_d_min=resolved_d_min,
            ghs_d_max=resolved_d_max,
            ghs_search_steps=resolved_steps,
            max_derivative=derivative_limit,
        )
        background_target_error = abs(
            float(resolved["mapped_background"]) - target
        )
        subject_target_error = abs(
            float(resolved["mapped_subject_p90"]) - subject_target
        )
        subject_target_tolerance = max(0.012, 0.08 * subject_target)
        resolved.update(
            {
                "background_target_error": background_target_error,
                "background_target_tolerance": 2e-5,
                "background_target_attained": background_target_error <= 2e-5,
                "subject_target_error": subject_target_error,
                "subject_target_tolerance": subject_target_tolerance,
                "subject_target_attained": (
                    subject_target_error <= subject_target_tolerance
                ),
            }
        )
        if background_target_error > 2e-5:
            raise ValueError(
                "Dual-stage MTF+GHS did not attain the background anchor"
            )
        if subject_target_error > subject_target_tolerance:
            raise ValueError(
                "Dual-stage MTF+GHS subject P90 target is infeasible within "
                "the bounded GHS search"
            )
        contract = _conditional_lut_contract(
            lut,
            max_derivative=derivative_limit,
        )
        if not contract.get("accepted"):
            raise ValueError(
                "Dual-stage MTF+GHS LUT rejected: "
                + ",".join(contract.get("issues") or [])
            )
        return {
            "schema": CONDITIONAL_STRETCH_SCHEMA,
            "status": "ok",
            "method": "dual_stage_mtf_ghs",
            "curve_application": "linked_rgb_common_curve",
            "source_reference": CONDITIONAL_STRETCH_REFERENCE,
            "ghs_source_reference": CONDITIONAL_GHS_REFERENCE,
            "source_license": CONDITIONAL_STRETCH_SOURCE_LICENSE,
            "source_profile": {
                "background_median": source_background,
                "background_sigma": max(background_sigma, 0.0),
                "subject_p90": source_subject_p90,
                "p99_9": source_p99_9,
            },
            "parameters": {
                "target_background": target,
                "target_subject_p90": subject_target,
                "ghs_b": resolved_b,
                "ghs_d_min": resolved_d_min,
                "ghs_d_max": resolved_d_max,
                "ghs_search_steps": resolved_steps,
                "max_derivative": derivative_limit,
            },
            "resolved": resolved,
            "lut_contract": contract,
            **_conditional_global_predictions(lut, global_p50, global_p99),
        }
    except (KeyError, TypeError, ValueError, FloatingPointError) as error:
        return {
            "schema": CONDITIONAL_STRETCH_SCHEMA,
            "status": "unavailable",
            "method": "dual_stage_mtf_ghs",
            "reason": str(error),
        }


def rebuild_conditional_stretch_lut(
    calibration: Dict[str, Any],
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Rebuild and authenticate a compact conditional-stretch calibration."""
    if (
        not isinstance(calibration, dict)
        or calibration.get("schema") != CONDITIONAL_STRETCH_SCHEMA
        or calibration.get("status") != "ok"
    ):
        raise ValueError("conditional stretch calibration is invalid")
    method = str(calibration.get("method") or "")
    profile = dict(calibration.get("source_profile") or {})
    parameters = dict(calibration.get("parameters") or {})
    if method == "iterative_masked_mtf":
        lut, _resolved = _iterative_masked_mtf_lut(
            source_background=float(profile["background_median"]),
            target_background=float(parameters["target_background"]),
            iterations=int(parameters["iterations"]),
        )
    elif method == "dual_stage_mtf_ghs":
        lut, _resolved = _dual_stage_mtf_ghs_lut(
            source_background=float(profile["background_median"]),
            background_sigma=float(profile["background_sigma"]),
            source_subject_p90=float(profile["subject_p90"]),
            source_p99_9=float(profile["p99_9"]),
            target_background=float(parameters["target_background"]),
            target_subject_p90=float(parameters["target_subject_p90"]),
            ghs_b=float(parameters["ghs_b"]),
            ghs_d_min=float(parameters["ghs_d_min"]),
            ghs_d_max=float(parameters["ghs_d_max"]),
            ghs_search_steps=int(parameters["ghs_search_steps"]),
            max_derivative=float(parameters["max_derivative"]),
        )
    else:
        raise ValueError(f"unsupported conditional stretch method: {method}")
    contract = _conditional_lut_contract(
        lut,
        max_derivative=float(parameters["max_derivative"]),
    )
    if not contract.get("accepted"):
        raise ValueError(
            "rebuilt conditional LUT failed validation: "
            + ",".join(contract.get("issues") or [])
        )
    expected = str(
        (calibration.get("lut_contract") or {}).get("sha256") or ""
    )
    if not expected or contract.get("sha256") != expected:
        raise ValueError("conditional stretch LUT digest mismatch")
    return lut, contract


def apply_conditional_linked_rgb_stretch(
    image: np.ndarray,
    calibration: Dict[str, Any],
) -> np.ndarray:
    """Apply an authenticated one-dimensional LUT independently to R/G/B."""
    rgb = _stage7_rgb_float_fullres(np.asarray(image))
    lut, _contract = rebuild_conditional_stretch_lut(calibration)
    grid = np.linspace(0.0, 1.0, lut.size, dtype=np.float64)
    mapped = np.interp(
        np.clip(rgb.reshape(-1), 0.0, 1.0),
        grid,
        lut,
    ).reshape(rgb.shape)
    return np.asarray(
        np.clip(mapped, 0.0, CONDITIONAL_STRETCH_OUTPUT_HEADROOM),
        dtype=np.float32,
    )


def calibrate_adaptive_quantile_stretch(
    image: np.ndarray,
    adaptation: Dict[str, Any],
    cfg: Any,
) -> Dict[str, Any]:
    """Build a bounded linked curve from source quantiles to preview targets."""
    if not bool(getattr(cfg, "stage7_quantile_fallback_enabled", True)):
        return {
            "status": "disabled",
            "reason": "adaptive quantile fallback disabled by config",
        }
    try:
        preview_calibration = adaptation.get("preview_calibration") or {}
        candidate_a = preview_calibration.get("candidate_a") or {}
        # Conditional cand_a methods rebase their own P50 contract onto the
        # analytically predicted global median.  The last-resort quantile
        # fallback deliberately retains the original preview ruler.
        target_p50 = float(
            candidate_a.get(
                "auto_asinh_target_p50",
                candidate_a.get("target_p50", 0.0),
            )
            or 0.0
        )
        target_p99 = float(
            candidate_a.get(
                "auto_asinh_target_p99",
                candidate_a.get("target_p99", 0.0),
            )
            or 0.0
        )
        if (
            not np.isfinite(target_p50)
            or not np.isfinite(target_p99)
            or target_p50 <= 0.0
            or target_p99 <= target_p50
        ):
            raise ValueError("preview P50/P99 targets unavailable")

        rgb = _stage7_rgb_float_fullres(np.asarray(image))
        finite = rgb[np.isfinite(rgb)]
        if finite.size < 64:
            raise ValueError("too few finite source pixels")
        if finite.size > 2_000_000:
            sample_step = int(np.ceil(finite.size / 2_000_000.0))
            finite = finite[::sample_step]
        input_percentiles = np.asarray(
            [0.1, 1.0, 50.0, 90.0, 99.0, 99.9, 100.0],
            dtype=np.float32,
        )
        input_values = np.percentile(finite, input_percentiles).astype(np.float64)

        shadow_low = min(max(target_p50 * 0.08, 0.0005), 0.008)
        shadow_high = min(max(target_p50 * 0.22, shadow_low + 0.001), 0.018)
        target_p90 = target_p50 + 0.55 * (target_p99 - target_p50)
        peak_target = min(0.970, max(target_p99 + 0.040, target_p99 * 1.06))
        maximum_target = min(0.985, max(peak_target + 0.010, target_p99 + 0.080))
        output_values = np.asarray(
            [
                shadow_low,
                shadow_high,
                target_p50,
                target_p90,
                target_p99,
                peak_target,
                maximum_target,
            ],
            dtype=np.float64,
        )
        output_values = np.maximum.accumulate(output_values)

        # Flat shadows can produce duplicate input quantiles. Keep the later
        # (brighter) target at an identical source value so the P50 contract is
        # not silently lost.
        unique_inputs = []
        unique_outputs = []
        unique_percentiles = []
        for percentile, source_value, target_value in zip(
            input_percentiles,
            input_values,
            output_values,
        ):
            source_value = float(source_value)
            target_value = float(target_value)
            if not np.isfinite(source_value) or not np.isfinite(target_value):
                continue
            if unique_inputs and source_value <= unique_inputs[-1] + 1e-8:
                unique_inputs[-1] = max(unique_inputs[-1], source_value)
                unique_outputs[-1] = max(unique_outputs[-1], target_value)
                unique_percentiles[-1] = float(percentile)
                continue
            unique_inputs.append(source_value)
            unique_outputs.append(target_value)
            unique_percentiles.append(float(percentile))
        if len(unique_inputs) < 4 or unique_inputs[-1] <= unique_inputs[0] + 1e-6:
            raise ValueError("source quantiles do not define a usable curve")
        if any(
            later <= earlier
            for earlier, later in zip(unique_inputs, unique_inputs[1:])
        ):
            raise ValueError("source quantile anchors are not strictly increasing")

        return {
            "status": "ok",
            "method": "linked_piecewise_linear_quantile_curve",
            "input_percentiles": unique_percentiles,
            "input_anchors": unique_inputs,
            "output_anchors": unique_outputs,
            "target_p50": target_p50,
            "target_p99": target_p99,
            "brightness_ordering_preserved": True,
            "channel_curve_linked": True,
            "source": "stage7_preview_ref candidate_a P50/P99",
        }
    except (IndexError, TypeError, ValueError, FloatingPointError) as error:
        return {
            "status": "unavailable",
            "reason": str(error),
        }


def apply_adaptive_quantile_stretch(
    image: np.ndarray,
    calibration: Dict[str, Any],
) -> np.ndarray:
    """Apply one shared monotonic curve to all RGB samples."""
    source = np.asarray(image)
    rgb = _stage7_rgb_float_fullres(source)
    inputs = np.asarray(calibration.get("input_anchors"), dtype=np.float64)
    outputs = np.asarray(calibration.get("output_anchors"), dtype=np.float64)
    if (
        str(calibration.get("status") or "") != "ok"
        or inputs.ndim != 1
        or outputs.ndim != 1
        or inputs.size < 4
        or inputs.size != outputs.size
        or not np.all(np.isfinite(inputs))
        or not np.all(np.isfinite(outputs))
        or np.any(np.diff(inputs) <= 0.0)
        or np.any(np.diff(outputs) < 0.0)
    ):
        raise ValueError("adaptive quantile calibration is invalid")
    mapped = np.interp(
        rgb.reshape(-1),
        inputs,
        outputs,
        left=float(outputs[0]),
        right=float(outputs[-1]),
    ).reshape(rgb.shape)
    return np.clip(mapped, 0.0, 0.995).astype(np.float32, copy=False)


def assess_target_local_stretch(
    baseline: np.ndarray,
    candidate: np.ndarray,
    target_type: str,
    cfg: Any,
    *,
    target_profile: Optional[Dict[str, Any]] = None,
    starmask: Optional[np.ndarray] = None,
    frozen_reference_available: bool = True,
) -> Dict[str, Any]:
    """Measure core, faint-structure and dark-lane regions derived from linear data."""
    normalized_target = str(target_type or "generic_low_snr_safe").strip().lower()
    strict_evidence = stage7_quality.strict_bright_core_target_evidence(
        normalized_target,
        target_profile,
    )
    strict_target = bool(strict_evidence.get("strict", False))
    if not bool(getattr(cfg, "stage7_target_local_metrics_enabled", True)):
        if strict_target:
            return {
                "status": "rejected",
                "accepted": False,
                "issues": ["local_core_reference_gate_disabled"],
                "advisories": [],
                "quality_gates": {
                    "local_core_reference_available": {
                        "status": "hard_failed",
                        "hard_failed": True,
                        "advisory": False,
                        "fixed_limit": True,
                    }
                },
                "risk_score": 1.0,
                "metrics": {},
                "strict_target_evidence": strict_evidence,
            }
        return {
            "status": "disabled",
            "accepted": True,
            "issues": [],
            "risk_score": 0.0,
            "metrics": {},
        }
    if strict_target and not frozen_reference_available:
        return {
            "status": "rejected",
            "accepted": False,
            "issues": ["local_core_frozen_reference_missing"],
            "advisories": [],
            "quality_gates": {
                "local_core_reference_available": {
                    "status": "hard_failed",
                    "hard_failed": True,
                    "advisory": False,
                    "fixed_limit": True,
                }
            },
            "risk_score": 1.0,
            "metrics": {},
            "target_type": normalized_target,
            "strict_target_evidence": strict_evidence,
        }
    try:
        source_rgb = _stage7_rgb_float_fullres(np.asarray(baseline))
        candidate_rgb = _stage7_rgb_float_fullres(np.asarray(candidate))
        if source_rgb.shape != candidate_rgb.shape:
            raise ValueError(
                f"shape mismatch: baseline={source_rgb.shape}, candidate={candidate_rgb.shape}"
            )
        source_gray = (
            0.2126 * source_rgb[0]
            + 0.7152 * source_rgb[1]
            + 0.0722 * source_rgb[2]
        ).astype(np.float32)
        candidate_gray = (
            0.2126 * candidate_rgb[0]
            + 0.7152 * candidate_rgb[1]
            + 0.0722 * candidate_rgb[2]
        ).astype(np.float32)
        candidate_peak = np.max(candidate_rgb[:3], axis=0)
        broad = source_gray.copy()
        for _ in range(4):
            broad = _box_blur_gray(broad)
        q35, q55, q90, q99 = np.percentile(broad, [35.0, 55.0, 90.0, 99.0])
        background_mask = broad <= q35
        dark_mask = (broad > q35) & (broad <= q55)
        faint_mask = (broad > q55) & (broad <= q90)
        core_mask = broad > q99
        core_roi_evidence: Dict[str, Any] = {
            "available": True,
            "method": "target_local_top1pct",
            "support": int(np.count_nonzero(core_mask)),
        }
        if strict_target:
            strict_core_mask, core_roi_evidence = (
                stage7_quality.build_bright_core_roi(baseline, starmask)
            )
            if strict_core_mask is None or not bool(
                core_roi_evidence.get("available", False)
            ):
                raise ValueError(
                    "strict bright-core ROI unavailable: "
                    f"{core_roi_evidence.get('reason', 'unknown')}"
                )
            core_mask = strict_core_mask
            starmask_rgb = _stage7_rgb_float_fullres(np.asarray(starmask))
            star_pixels = (
                np.max(starmask_rgb, axis=0)
                > stage7_quality.BRIGHT_CORE_STARMASK_THRESHOLD
            )
            star_exclusion = stage7_quality._dilate_binary_mask(
                star_pixels,
                stage7_quality.BRIGHT_CORE_STARMASK_EXPANSION,
            )
            background_mask &= ~star_exclusion
            dark_mask &= ~star_exclusion
            faint_mask &= ~star_exclusion
        if min(
            int(np.count_nonzero(background_mask)),
            int(np.count_nonzero(dark_mask)),
            int(np.count_nonzero(faint_mask)),
            int(np.count_nonzero(core_mask)),
        ) < 16:
            raise ValueError("target-local masks contain too few samples")

        background_values = candidate_gray[background_mask]
        dark_values = candidate_gray[dark_mask]
        faint_values = candidate_gray[faint_mask]
        core_values = candidate_gray[core_mask]
        background_median = float(np.median(background_values))
        background_std = max(float(np.std(background_values)), 1e-6)
        dark_median = float(np.median(dark_values))
        faint_median = float(np.median(faint_values))
        faint_contrast = faint_median - background_median
        dark_separation = faint_median - dark_median
        core_clip_ratio = float(np.mean(candidate_peak[core_mask] >= 0.995))
        core_p99 = float(np.percentile(core_values, 99.0))
        metrics = {
            "background_median": background_median,
            "background_std": background_std,
            "faint_median": faint_median,
            "faint_contrast": faint_contrast,
            "faint_snr": faint_contrast / background_std,
            "dark_median": dark_median,
            "dark_separation": dark_separation,
            "core_median": float(np.median(core_values)),
            "core_p99": core_p99,
            "core_clip_ratio": core_clip_ratio,
            "background_coverage": float(np.mean(background_mask)),
            "faint_coverage": float(np.mean(faint_mask)),
            "core_coverage": float(np.mean(core_mask)),
        }

        issues = []
        advisories = []
        quality_gates: Dict[str, Dict[str, Any]] = {}
        risk_score = 0.0
        if normalized_target in CORE_PROTECT_TARGETS:
            if strict_target:
                channel_clip_ratios = [
                    float(np.mean(candidate_rgb[channel][core_mask] >= 0.995))
                    for channel in range(3)
                ]
                core_clip_ratio = max(channel_clip_ratios)
                metrics["core_clip_ratio"] = core_clip_ratio
                metrics["core_channel_clip_ratios"] = channel_clip_ratios
                core_clip_max = 0.01
                core_clip_gate = stage7_quality._fixed_upper_gate(
                    core_clip_ratio,
                    accepted_limit=core_clip_max,
                    hard_limit=0.015,
                )
            else:
                core_clip_max = _bounded(
                    getattr(cfg, "stage7_local_core_clip_ratio_max", 0.12),
                    0.12,
                    0.01,
                    0.30,
                )
                core_clip_gate = stage7_quality.stage7_9_upper_quality_gate(
                    cfg,
                    value=core_clip_ratio,
                    accepted_limit=core_clip_max,
                )
            quality_gates["local_core_clip_ratio"] = core_clip_gate
            if core_clip_gate["hard_failed"]:
                issues.append(
                    f"local_core_clip_ratio {core_clip_ratio:.4f}>{core_clip_max:.4f}"
                )
            elif core_clip_gate["advisory"]:
                advisories.append(
                    "local_core_clip_ratio "
                    f"{core_clip_ratio:.4f}>{core_clip_max:.4f} advisory"
                )
            risk_score += core_clip_ratio * 8.0
            risk_score += max(0.0, core_p99 - 0.985) * 30.0

            if strict_target:
                capped_channels = candidate_rgb >= 0.995
                colored_plateau = (
                    np.any(capped_channels, axis=0)
                    & ~np.all(capped_channels, axis=0)
                    & core_mask
                )
                component_areas = _component_areas(colored_plateau)
                colored_component_ratio = float(
                    max(component_areas, default=0)
                ) / float(max(int(np.count_nonzero(core_mask)), 1))
                plateau_gate = stage7_quality._fixed_upper_gate(
                    colored_component_ratio,
                    accepted_limit=0.005,
                    hard_limit=0.01,
                )
                quality_gates[
                    "local_core_colored_plateau_component_ratio"
                ] = plateau_gate
                metrics[
                    "core_colored_plateau_component_ratio"
                ] = colored_component_ratio
                metrics["core_colored_plateau_pixels"] = int(
                    np.count_nonzero(colored_plateau)
                )
                if plateau_gate["hard_failed"]:
                    issues.append(
                        "local_core_colored_plateau_component_ratio "
                        f"{colored_component_ratio:.4f}>0.0100"
                    )
                elif plateau_gate["advisory"]:
                    advisories.append(
                        "local_core_colored_plateau_component_ratio "
                        f"{colored_component_ratio:.4f}>0.0050 advisory"
                    )

                parity_means = []
                parity_spans = []
                parity_complete = True
                for channel in range(3):
                    channel_means = []
                    for phase_y in range(2):
                        for phase_x in range(2):
                            phase_mask = core_mask[phase_y::2, phase_x::2]
                            values = candidate_rgb[
                                channel, phase_y::2, phase_x::2
                            ][phase_mask]
                            if values.size:
                                channel_means.append(float(np.mean(values)))
                            else:
                                channel_means.append(None)
                                parity_complete = False
                    available = [
                        value for value in channel_means if value is not None
                    ]
                    parity_means.append(channel_means)
                    parity_spans.append(
                        float(max(available) - min(available))
                        if available
                        else float("inf")
                    )
                parity_span = max(parity_spans)
                parity_gate = stage7_quality._fixed_upper_gate(
                    parity_span,
                    accepted_limit=0.01,
                    hard_limit=0.015,
                )
                if not parity_complete:
                    parity_gate.update(
                        {
                            "status": "hard_failed",
                            "hard_failed": True,
                            "advisory": False,
                        }
                    )
                quality_gates["local_core_parity_phase_span"] = parity_gate
                metrics["core_parity_phase_means"] = parity_means
                metrics["core_parity_phase_spans"] = parity_spans
                metrics["core_parity_phase_span"] = parity_span
                if parity_gate["hard_failed"]:
                    issues.append(
                        "local_core_parity_phase_span "
                        f"{parity_span:.4f}>0.0150"
                    )
                elif parity_gate["advisory"]:
                    advisories.append(
                        "local_core_parity_phase_span "
                        f"{parity_span:.4f}>0.0100 advisory"
                    )
                quality_gates["local_core_reference_available"] = {
                    "status": "ok",
                    "hard_failed": False,
                    "advisory": False,
                    "fixed_limit": True,
                }
                metrics["core_roi"] = core_roi_evidence
                risk_score += colored_component_ratio * 20.0
                risk_score += parity_span * 20.0

        if normalized_target in FAINT_SIGNAL_TARGETS:
            faint_snr_min = _bounded(
                getattr(cfg, "stage7_local_faint_snr_min", 0.25),
                0.25,
                0.0,
                2.0,
            )
            faint_snr_gate = stage7_quality.stage7_9_lower_quality_gate(
                cfg,
                value=metrics["faint_snr"],
                accepted_limit=faint_snr_min,
            )
            quality_gates["local_faint_snr"] = faint_snr_gate
            if faint_snr_gate["hard_failed"]:
                issues.append(
                    f"local_faint_snr {metrics['faint_snr']:.4f}<{faint_snr_min:.4f}"
                )
            elif faint_snr_gate["advisory"]:
                advisories.append(
                    "local_faint_snr "
                    f"{metrics['faint_snr']:.4f}<{faint_snr_min:.4f} advisory"
                )
            risk_score += max(0.0, faint_snr_min - metrics["faint_snr"]) * 2.0

        if normalized_target == "dark_nebula_low_contrast":
            separation_min = _bounded(
                getattr(cfg, "stage7_local_dark_separation_min", 0.001),
                0.001,
                0.0,
                0.02,
            )
            dark_separation_gate = stage7_quality.stage7_9_lower_quality_gate(
                cfg,
                value=dark_separation,
                accepted_limit=separation_min,
            )
            quality_gates["local_dark_separation"] = dark_separation_gate
            if dark_separation_gate["hard_failed"]:
                issues.append(
                    f"local_dark_separation {dark_separation:.5f}<{separation_min:.5f}"
                )
            elif dark_separation_gate["advisory"]:
                advisories.append(
                    "local_dark_separation "
                    f"{dark_separation:.5f}<{separation_min:.5f} advisory"
                )
            risk_score += max(0.0, separation_min - dark_separation) * 100.0

        return {
            "status": "ok" if not issues else "rejected",
            "accepted": not issues,
            "issues": issues,
            "advisories": advisories,
            "quality_gates": quality_gates,
            "risk_score": float(risk_score),
            "metrics": metrics,
            "target_type": normalized_target,
            "strict_target_evidence": strict_evidence,
        }
    except (IndexError, TypeError, ValueError, FloatingPointError) as error:
        if strict_target:
            return {
                "status": "rejected",
                "accepted": False,
                "issues": [f"local_core_frozen_reference_unavailable: {error}"],
                "advisories": [],
                "quality_gates": {
                    "local_core_reference_available": {
                        "status": "hard_failed",
                        "hard_failed": True,
                        "advisory": False,
                        "fixed_limit": True,
                    }
                },
                "risk_score": 1.0,
                "metrics": {},
                "target_type": normalized_target,
                "strict_target_evidence": strict_evidence,
                "reason": str(error),
            }
        return {
            "status": "unavailable",
            "accepted": True,
            "issues": [],
            "risk_score": 0.0,
            "metrics": {},
            "reason": str(error),
        }
