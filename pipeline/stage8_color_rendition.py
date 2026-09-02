"""Bounded target-aware subject colour rendition for Stage 8 Starless images."""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

try:
    from .image_metrics import _box_blur_gray, _to_rgb_float_fullres
    from .stage7_pixel_domain import canonicalize_stage7_pixels_01
except ImportError:
    from image_metrics import _box_blur_gray, _to_rgb_float_fullres
    from stage7_pixel_domain import canonicalize_stage7_pixels_01


STAGE8_SUBJECT_CHROMA_SCHEMA = "starun.stage8-subject-chroma.v1"
STAGE8_SUBJECT_CHROMA_HEADROOM = 0.995


def _mask(
    masks: Optional[Dict[str, Any]],
    name: str,
    shape: tuple[int, int],
) -> Optional[np.ndarray]:
    if not isinstance(masks, dict) or masks.get(name) is None:
        return None
    value = np.asarray(masks[name], dtype=np.float32)
    if value.ndim != 2 or tuple(value.shape) != tuple(shape):
        return None
    if not np.all(np.isfinite(value)):
        return None
    return np.clip(value, 0.0, 1.0)


def _subject_weight(
    masks: Optional[Dict[str, Any]],
    shape: tuple[int, int],
) -> np.ndarray:
    explicit = _mask(masks, "subject_mask", shape)
    if explicit is not None:
        return explicit
    layers = [
        value
        for name in (
            "core_mask",
            "nebula_mask",
            "faint_nebula_mask",
            "galaxy_signal_mask",
        )
        if (value := _mask(masks, name, shape)) is not None
    ]
    if layers:
        return np.maximum.reduce(layers).astype(np.float32, copy=False)
    background = _mask(masks, "background_mask", shape)
    if background is not None:
        return (1.0 - background).astype(np.float32, copy=False)
    return np.zeros(shape, dtype=np.float32)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return (
        0.2126 * rgb[0]
        + 0.7152 * rgb[1]
        + 0.0722 * rgb[2]
    ).astype(np.float32)


def _saturation_proxy(rgb: np.ndarray) -> np.ndarray:
    peak = np.max(rgb, axis=0)
    floor = np.min(rgb, axis=0)
    return np.divide(
        peak - floor,
        np.maximum(peak, 1e-6),
        out=np.zeros_like(peak, dtype=np.float32),
        where=peak > 1e-6,
    ).astype(np.float32)


def subject_saturation_median(
    image: np.ndarray,
    masks: Optional[Dict[str, Any]],
) -> Optional[float]:
    """Return broad non-background saturation for Stage 8 factor selection."""

    canonical, _provenance = canonicalize_stage7_pixels_01(image)
    rgb = _to_rgb_float_fullres(canonical)
    shape = tuple(int(value) for value in rgb.shape[1:])
    background = _mask(masks, "background_mask", shape)
    subject = _subject_weight(masks, shape) > 0.25
    if background is not None:
        subject |= background <= 0.50
        subject &= background < 0.80
    if int(np.count_nonzero(subject)) < 64:
        return None
    value = float(np.median(_saturation_proxy(rgb)[subject]))
    return value if math.isfinite(value) else None


def subject_saturation_distribution(
    image: np.ndarray,
    masks: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Measure the protected subject saturation budget on one display image."""

    canonical, _provenance = canonicalize_stage7_pixels_01(image)
    rgb = _to_rgb_float_fullres(canonical)
    shape = tuple(int(value) for value in rgb.shape[1:])
    subject = _subject_weight(masks, shape) > 0.05
    for name, threshold in (
        ("background_mask", 0.80),
        ("core_mask", 0.50),
        ("star_mask", 0.05),
        ("star_halo_guard_mask", 0.05),
    ):
        value = _mask(masks, name, shape)
        if value is not None:
            subject &= value < threshold
    count = int(np.count_nonzero(subject))
    if count < 64:
        return {"status": "unavailable", "support_count": count}
    saturation = _saturation_proxy(rgb)[subject]
    return {
        "status": "available",
        "support_count": count,
        "p50": float(np.percentile(saturation, 50.0)),
        "p95": float(np.percentile(saturation, 95.0)),
    }


def target_aware_chroma_factor(
    profile_name: str,
    *,
    subject_saturation: Optional[float],
    effective_saturation_budget: float,
) -> Dict[str, Any]:
    """Resolve the former vivid-safe factor under the Stage 8 colour budget."""

    profile = str(profile_name or "generic_balanced").strip().lower()
    budget = float(np.clip(float(effective_saturation_budget), 0.0, 0.65))
    absolute_goal = 0.30 if profile == "bright_core_composite_reveal" else None
    if profile == "bright_core_composite_reveal":
        try:
            absolute = float(subject_saturation)
        except (TypeError, ValueError):
            absolute = 0.0
        if math.isfinite(absolute) and absolute > 1e-6:
            raw_factor = 1.0 + 2.0 * max(0.0, 0.30 / absolute - 1.0)
            raw_factor = max(1.12, min(4.0, raw_factor))
        else:
            raw_factor = 1.12
    elif profile in {
        "bright_core_protect",
        "widefield_nebulosity",
        "widefield_faint_signal",
        "widefield_subject_separation",
        "dark_nebula_separation",
    }:
        raw_factor = 1.12
    elif profile == "galaxy_core_halo_balance":
        raw_factor = 1.08
    elif profile == "star_colour_preserve":
        raw_factor = 1.06
    else:
        raw_factor = 1.08
    budget_scale = float(np.clip(budget / 0.40, 0.0, 1.0))
    factor = 1.0 + (raw_factor - 1.0) * budget_scale
    return {
        "profile": profile,
        "factor": float(np.clip(factor, 1.0, 4.0)),
        "raw_factor": float(raw_factor),
        "budget_scale": budget_scale,
        "effective_saturation_budget": budget,
        "absolute_subject_saturation": subject_saturation,
        "absolute_subject_saturation_goal": absolute_goal,
    }


def apply_subject_chroma_rendition(
    image: np.ndarray,
    masks: Optional[Dict[str, Any]],
    *,
    factor: float,
    output_headroom: float = STAGE8_SUBJECT_CHROMA_HEADROOM,
    expand_faint_signal: bool = False,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Boost coherent subject chroma while preserving luminance and sky pixels."""

    canonical, _provenance = canonicalize_stage7_pixels_01(image)
    rgb = _to_rgb_float_fullres(canonical)
    if not np.all(np.isfinite(rgb)):
        raise ValueError("Stage8 subject chroma source contains non-finite pixels")
    shape = tuple(int(value) for value in rgb.shape[1:])
    factor = float(np.clip(float(factor), 1.0, 4.0))
    headroom = float(np.clip(float(output_headroom), 0.95, 0.999))
    has_roi = any(
        _mask(masks, name, shape) is not None
        for name in (
            "subject_mask",
            "background_mask",
            "core_mask",
            "nebula_mask",
            "faint_nebula_mask",
            "galaxy_signal_mask",
        )
    )
    if not has_roi:
        raise ValueError("Stage8 subject chroma requires a valid frozen ROI")

    subject_weight = _subject_weight(masks, shape)
    background = _mask(masks, "background_mask", shape)
    core_weight = _mask(masks, "core_mask", shape)
    star_weight = _mask(masks, "star_mask", shape)
    halo_weight = _mask(masks, "star_halo_guard_mask", shape)
    non_background_smoothing_passes = 0
    non_background_opening_passes = 0
    non_background_star_exclusion_applied = False

    if expand_faint_signal:
        broad_layers = []
        for name in (
            "nebula_mask",
            "faint_nebula_mask",
            "galaxy_signal_mask",
        ):
            layer = _mask(masks, name, shape)
            if layer is not None:
                broad_layers.append(layer)
        if background is not None:
            broad_signal = (1.0 - background).astype(np.float32, copy=False)
            broad_signal[background >= 0.80] = 0.0
            if star_weight is not None:
                star_exclusion = (star_weight > 0.01).astype(np.float32)
                for _ in range(2):
                    star_exclusion = (
                        _box_blur_gray(star_exclusion) > 1e-6
                    ).astype(np.float32)
                broad_signal *= 1.0 - star_exclusion
                non_background_star_exclusion_applied = True
            non_background_opening_passes = 8
            for _ in range(non_background_opening_passes):
                broad_signal = (
                    _box_blur_gray(broad_signal) >= 1.0 - 1e-6
                ).astype(np.float32)
            for _ in range(non_background_opening_passes):
                broad_signal = (
                    _box_blur_gray(broad_signal) > 1e-6
                ).astype(np.float32)
            non_background_smoothing_passes = 8
            for _ in range(non_background_smoothing_passes):
                broad_signal = _box_blur_gray(broad_signal)
            broad_layers.append(broad_signal)
        subject_weight = (
            np.maximum.reduce(broad_layers).astype(np.float32, copy=False)
            if broad_layers
            else np.zeros(shape, dtype=np.float32)
        )

    if background is not None:
        subject_weight[background >= 0.80] = 0.0
    if core_weight is not None:
        subject_weight *= 1.0 - 0.90 * core_weight
        subject_weight[core_weight >= 0.50] = 0.0
    star_protection_applied = bool(
        star_weight is not None and not expand_faint_signal
    )
    if star_weight is not None:
        if star_protection_applied:
            subject_weight *= 1.0 - 0.90 * star_weight
        subject_weight[star_weight >= 0.05] = 0.0
    if halo_weight is not None:
        subject_weight[halo_weight >= 0.05] = 0.0

    subject_support = subject_weight > 0.05
    if int(np.count_nonzero(subject_support)) < 64:
        raise ValueError("Stage8 subject chroma mask has insufficient support")

    source_peak = np.max(rgb, axis=0)
    source_floor = np.min(rgb, axis=0)
    range_weight = np.minimum(
        np.clip((headroom - source_peak) / 0.04, 0.0, 1.0),
        np.clip(source_floor / 0.02, 0.0, 1.0),
    )
    boost_weight = np.clip(subject_weight * range_weight, 0.0, 1.0)
    luminance = _luminance(rgb)
    chroma = rgb - luminance[None, :, :]
    chroma_smoothing_passes = 6 if expand_faint_signal else 2
    broad_channels = []
    for channel in chroma:
        broad_channel = channel
        for _ in range(chroma_smoothing_passes):
            broad_channel = _box_blur_gray(broad_channel)
        broad_channels.append(broad_channel)
    broad_chroma = np.stack(broad_channels, axis=0).astype(np.float32)
    delta = (factor - 1.0) * boost_weight[None, :, :] * broad_chroma
    delta_luma = _luminance(delta)
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
    rendered = np.asarray(
        np.clip(rgb + delta * safe_scale[None, :, :], 0.0, headroom),
        dtype=np.float32,
    )
    rendered_luma = _luminance(rendered)
    rendered_peak = np.max(rendered, axis=0)
    newly_clipped = (
        (rendered_peak >= 1.0 - 1e-7)
        & (source_peak < 1.0 - 1e-7)
    )
    outside = subject_weight <= 1e-6
    background_support = (
        background >= 0.80
        if background is not None
        else outside
    )
    delta_abs = np.max(np.abs(rendered - rgb), axis=0)
    source_sat = _saturation_proxy(rgb)
    rendered_sat = _saturation_proxy(rendered)
    measured_support = boost_weight > 0.05
    subject_gain = (
        float(np.median(rendered_sat[measured_support]))
        - float(np.median(source_sat[measured_support]))
        if int(np.count_nonzero(measured_support)) >= 64
        else 0.0
    )
    core_support = core_weight >= 0.50 if core_weight is not None else None
    star_support = star_weight >= 0.05 if star_weight is not None else None
    halo_support = halo_weight >= 0.05 if halo_weight is not None else None
    return rendered, {
        "schema": STAGE8_SUBJECT_CHROMA_SCHEMA,
        "mode": "frozen_subject_broad_chroma",
        "factor": factor,
        "output_headroom": headroom,
        "subject_support_count": int(np.count_nonzero(subject_support)),
        "subject_coverage": float(np.mean(subject_weight > 0.25)),
        "boosted_coverage": float(np.mean(measured_support)),
        "faint_signal_expansion_applied": bool(expand_faint_signal),
        "non_background_signal_expansion_applied": bool(
            expand_faint_signal and background is not None
        ),
        "non_background_signal_smoothing_passes": int(
            non_background_smoothing_passes
        ),
        "non_background_signal_opening_passes": int(
            non_background_opening_passes
        ),
        "non_background_star_exclusion_applied": bool(
            non_background_star_exclusion_applied
        ),
        "chroma_smoothing_passes": int(chroma_smoothing_passes),
        "core_protection_applied": core_weight is not None,
        "star_protection_applied": star_protection_applied,
        "halo_protection_applied": halo_weight is not None,
        "star_protection_skipped_for_starless_expansion": bool(
            star_weight is not None and expand_faint_signal
        ),
        "mean_effective_scale": float(np.mean(safe_scale[measured_support]))
        if np.any(measured_support)
        else 0.0,
        "max_luminance_error": float(np.max(np.abs(rendered_luma - luminance))),
        "newly_clipped_ratio": float(np.mean(newly_clipped)),
        "candidate_max": float(np.max(rendered)),
        "outside_subject_max_abs_change": (
            float(np.max(delta_abs[outside])) if np.any(outside) else 0.0
        ),
        "background_max_abs_change": (
            float(np.max(delta_abs[background_support]))
            if np.any(background_support)
            else 0.0
        ),
        "core_max_abs_change": (
            float(np.max(delta_abs[core_support]))
            if core_support is not None and np.any(core_support)
            else 0.0
        ),
        "star_max_abs_change": (
            float(np.max(delta_abs[star_support]))
            if star_support is not None and np.any(star_support)
            else 0.0
        ),
        "halo_max_abs_change": (
            float(np.max(delta_abs[halo_support]))
            if halo_support is not None and np.any(halo_support)
            else 0.0
        ),
        "subject_saturation_median_before": (
            float(np.median(source_sat[measured_support]))
            if int(np.count_nonzero(measured_support)) >= 64
            else None
        ),
        "subject_saturation_median_after": (
            float(np.median(rendered_sat[measured_support]))
            if int(np.count_nonzero(measured_support)) >= 64
            else None
        ),
        "subject_saturation_median_gain": subject_gain,
        "background_unchanged": bool(
            np.allclose(
                rendered[:, background_support],
                rgb[:, background_support],
                rtol=0.0,
                atol=2e-7,
            )
        ),
    }


def assess_subject_chroma_candidate(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Apply the Stage 8 colour-only hard gate to one rendered candidate."""

    limits = {
        "subject_support_count_min": 64,
        "max_luminance_error_max": 1e-6,
        "newly_clipped_ratio_max": 0.0,
        "candidate_max_max": STAGE8_SUBJECT_CHROMA_HEADROOM + 1e-7,
        "outside_subject_max_abs_change_max": 2e-7,
        "background_max_abs_change_max": 2e-7,
        "core_max_abs_change_max": 2e-7,
        "star_max_abs_change_max": 2e-7,
        "halo_max_abs_change_max": 2e-7,
        "subject_saturation_median_gain_min_exclusive": 1e-6,
    }
    issues = []
    finite_fields = (
        "max_luminance_error",
        "newly_clipped_ratio",
        "candidate_max",
        "outside_subject_max_abs_change",
        "background_max_abs_change",
        "core_max_abs_change",
        "star_max_abs_change",
        "halo_max_abs_change",
        "subject_saturation_median_gain",
    )
    for field in finite_fields:
        try:
            value = float(metadata[field])
        except (KeyError, TypeError, ValueError):
            issues.append(f"{field}_unavailable")
            continue
        if not math.isfinite(value):
            issues.append(f"{field}_nonfinite")
    if int(metadata.get("subject_support_count", 0) or 0) < 64:
        issues.append("subject_support_insufficient")
    for field in (
        "max_luminance_error",
        "newly_clipped_ratio",
        "candidate_max",
        "outside_subject_max_abs_change",
        "background_max_abs_change",
        "core_max_abs_change",
        "star_max_abs_change",
        "halo_max_abs_change",
    ):
        try:
            if float(metadata[field]) > float(limits[f"{field}_max"]) + 1e-12:
                issues.append(field)
        except (KeyError, TypeError, ValueError):
            pass
    try:
        if float(metadata["subject_saturation_median_gain"]) <= 1e-6:
            issues.append("subject_chroma_no_positive_gain")
    except (KeyError, TypeError, ValueError):
        pass
    return {
        "accepted": not issues,
        "status": "accepted" if not issues else "rejected",
        "limits": limits,
        "issues": list(dict.fromkeys(issues)),
    }


__all__ = [
    "STAGE8_SUBJECT_CHROMA_SCHEMA",
    "apply_subject_chroma_rendition",
    "assess_subject_chroma_candidate",
    "subject_saturation_median",
    "subject_saturation_distribution",
    "target_aware_chroma_factor",
]
