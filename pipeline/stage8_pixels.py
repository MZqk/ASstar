from __future__ import annotations

import importlib
import math
import os
import shutil
import subprocess
import sys
import types
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from channel_semantics import BROADBAND_RGB_OSC
from color_quality_metrics import physical_broadband_anchor_accepted
import stage7_quality
from image_metrics import (
    _box_blur_gray,
    _clamp_float,
    _to_rgb_float_fullres,
    _to_rgb_float_image,
    measure_image_features,
    measure_quality_metrics,
)
from local_adjustments import (
    LOCAL_ADJUSTMENT_SCHEMA,
    apply_local_adjustment_recipe,
    dilate_mask,
    feather_mask,
)
from models import ImageFeatures, QualityMetrics

_FINAL_COMPACT_HALO_NEAR_LIMIT_FACTOR = 1.10
_FINAL_GLOBAL_HALO_SAFE_FACTOR = 0.75
_FINAL_COMPACT_RESIDUAL_SCORE_SAFE_MAX = 0.02
_FINAL_COMPACT_RESIDUAL_COVERAGE_SAFE_MAX = 0.005
_FINAL_STARLESS_ARTIFACT_SAFE_MAX = 0.20
_FINAL_LOCAL_PATCH_VARIANCE_STRICT_MAX = 0.00016
_FINAL_LOCAL_PATCH_VARIANCE_MAX = 0.00022
_FINAL_LARGE_GALAXY_LOCAL_PATCH_VARIANCE_MAX = 0.00032
_FINAL_NOISE_EXTREME_MAX = 0.90
_FINAL_TEXTURE_OUTLIER_SCORE_HARD_MAX = 4.0
_FINAL_TEXTURE_AFFECTED_RATIO_HARD_MAX = 0.35
_FINAL_NOISE_GROWTH_RATIO_HARD_MAX = 1.50
_FINAL_NOISE_ABSOLUTE_GROWTH_HARD_MIN = 0.10
_FINAL_TEXTURE_P90_GROWTH_HARD_MIN = 0.0015


# These are conservative target routes, not copies of AstroColorMixer's stronger
# interactive presets. Their amounts are derived from the already-capped Stage8
# saturation request in ``stage8_broadband_hue_saturation_bands`` below.
_STAGE8_BROADBAND_HUE_PROFILES: Dict[str, Tuple[Dict[str, float | str], ...]] = {
    "galaxy": (
        {
            "id": "warm_core",
            "center": 30.0,
            "width": 50.0,
            "feather": 0.80,
            "scale": 1.00,
        },
        {
            "id": "blue_structure",
            "center": 235.0,
            "width": 48.0,
            "feather": 0.82,
            "scale": 0.75,
        },
    ),
    "emission": (
        {
            "id": "observed_red",
            "center": 0.0,
            "width": 50.0,
            "feather": 0.82,
            "scale": 1.00,
        },
        {
            "id": "observed_cyan",
            "center": 190.0,
            "width": 45.0,
            "feather": 0.82,
            "scale": 0.55,
        },
    ),
    "reflection": (
        {
            "id": "observed_blue",
            "center": 235.0,
            "width": 50.0,
            "feather": 0.84,
            "scale": 1.00,
        },
        {
            "id": "observed_cyan",
            "center": 190.0,
            "width": 42.0,
            "feather": 0.84,
            "scale": 0.45,
        },
    ),
    "mixed_nebula": (
        {
            "id": "observed_red",
            "center": 0.0,
            "width": 48.0,
            "feather": 0.84,
            "scale": 0.75,
        },
        {
            "id": "observed_blue",
            "center": 235.0,
            "width": 48.0,
            "feather": 0.84,
            "scale": 0.65,
        },
    ),
}

_STAGE8_BROADBAND_TARGET_PROFILES = {
    "galaxy": "galaxy",
    "large_galaxy": "galaxy",
    "small_galaxy": "galaxy",
    "emission_nebula": "emission",
    "emission_nebula_widefield": "emission",
    "reflection_nebula": "reflection",
    "reflection_nebula_cluster": "reflection",
    "bright_emission_reflection_nebula": "mixed_nebula",
}

def _clamp_int(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(value)))


def stage8_broadband_hue_saturation_bands(
    target_type: str,
    saturation: float,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Build the single conservative hue profile for a recognized broadband target."""
    profile_name = _STAGE8_BROADBAND_TARGET_PROFILES.get(
        str(target_type or "").strip().lower(),
        "",
    )
    profile = _STAGE8_BROADBAND_HUE_PROFILES.get(profile_name, ())
    base_amount = min(0.04, max(0.0, float(saturation)) * 0.16)
    if not profile or base_amount <= 1e-6:
        return "", []
    return profile_name, [
        {
            "id": str(band["id"]),
            "center": float(band["center"]),
            "width": float(band["width"]),
            "feather": float(band["feather"]),
            "amount": base_amount * float(band["scale"]),
        }
        for band in profile
    ]


def stage8_target_color_route_allowed(pipeline) -> Tuple[bool, str]:
    """Fail closed to generic color preservation for low-confidence targets."""
    frozen = getattr(pipeline, "_frozen_primary_target", None)
    profile = frozen if isinstance(frozen, dict) and frozen else None
    if profile is None:
        candidate = getattr(pipeline, "target_profile", None)
        if isinstance(candidate, dict) and candidate:
            primary = candidate.get("primary_target")
            profile = primary if isinstance(primary, dict) and primary else candidate
    if not isinstance(profile, dict) or not profile:
        return False, "target_profile_unavailable"
    try:
        confidence = float(
            profile.get("confidence", profile.get("target_confidence", 0.0)) or 0.0
        )
    except (TypeError, ValueError):
        confidence = 0.0
    method = str(
        profile.get("method", profile.get("classification_method", "")) or ""
    ).strip().lower()
    if confidence < 0.55:
        return False, f"target_confidence={confidence:.3f}<0.550"
    if method in {"fallback", "unavailable", "unknown"}:
        return False, f"target_method={method}"
    return True, f"target_confidence={confidence:.3f}"

try:
    from sirilpy.exceptions import CommandError, SirilError
except ImportError:  # Tests may import with lightweight fakes.
    CommandError = RuntimeError
    SirilError = RuntimeError

def stage8_restore_rgb_like(pipeline, source_data: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    source = np.asarray(source_data)
    out = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
    if out.ndim != 3 or out.shape[0] != 3:
        raise ValueError(f"expected RGB CHW output, got shape={out.shape}")

    if source.ndim == 2:
        restored = (0.2126 * out[0] + 0.7152 * out[1] + 0.0722 * out[2]).astype(np.float32)
    elif source.ndim == 3 and source.shape[0] in (1, 3):
        restored = out[: source.shape[0], :, :]
        if source.shape[0] == 1:
            restored = (0.2126 * out[0] + 0.7152 * out[1] + 0.0722 * out[2])[None, :, :]
    elif source.ndim == 3 and source.shape[-1] in (1, 3):
        if source.shape[-1] == 1:
            restored = (0.2126 * out[0] + 0.7152 * out[1] + 0.0722 * out[2])[:, :, None]
        else:
            restored = np.transpose(out, (1, 2, 0))
    else:
        raise ValueError(f"unsupported source image shape: {source.shape}")

    if np.issubdtype(source.dtype, np.integer):
        max_value = float(np.iinfo(source.dtype).max)
        return np.clip(restored * max_value, 0, max_value).astype(source.dtype, copy=False)
    return restored.astype(np.float32, copy=False)

def stage8_soften_mask(pipeline, mask: np.ndarray, passes: int = 3) -> np.ndarray:
    softened = np.asarray(mask, dtype=np.float32)
    softened = np.clip(softened, 0.0, 1.0)
    for _ in range(max(0, int(passes))):
        softened = _box_blur_gray(softened)
    return np.clip(softened, 0.0, 1.0)

def stage8_low_signal_thresholds(
    *,
    bg_median: float,
    bg_std: float,
    p90: float,
    p99: float,
) -> Dict[str, Any]:
    signal_span = max(float(p99) - float(bg_median), float(p90) - float(bg_median), 0.0)
    low_signal = (
        float(bg_median) < 0.080
        and float(p99) < 0.120
        and signal_span < 0.015
    )
    if not low_signal:
        return {
            "low_signal": False,
            "nebula_floor": 0.025,
            "faint_floor": 0.008,
            "ramp_floor": 0.020,
            "std_floor": 0.010,
        }
    noise_floor = max(4.0 * float(bg_std), signal_span * 0.25, 0.00025)
    return {
        "low_signal": True,
        "nebula_floor": min(0.006, max(0.00035, noise_floor * 1.2)),
        "faint_floor": min(0.003, max(0.00015, noise_floor * 0.45)),
        "ramp_floor": min(0.008, max(0.00050, signal_span * 0.75, noise_floor * 1.4)),
        "std_floor": 0.0005,
    }

def stage8_generate_starless_masks(pipeline, image_data: np.ndarray) -> Dict[str, Any]:
    rgb = _to_rgb_float_fullres(image_data)
    gray = (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]).astype(np.float32)
    channel_peak = np.max(rgb, axis=0)
    bg_threshold = float(np.quantile(gray, 0.22))
    bg_mask = gray <= bg_threshold
    if int(np.count_nonzero(bg_mask)) < 64:
        bg_mask = gray <= float(np.quantile(gray, 0.30))
    bg_values = gray[bg_mask] if np.any(bg_mask) else gray.reshape(-1)
    bg_median = float(np.median(bg_values))
    bg_std = float(np.std(bg_values))
    p90 = float(np.quantile(gray, 0.90))
    p99 = float(np.quantile(gray, 0.99))
    low_signal_thresholds = stage8_low_signal_thresholds(
        bg_median=bg_median,
        bg_std=bg_std,
        p90=p90,
        p99=p99,
    )
    nebula_floor = float(low_signal_thresholds["nebula_floor"])
    faint_floor = float(low_signal_thresholds["faint_floor"])
    ramp_floor = float(low_signal_thresholds["ramp_floor"])
    std_floor = float(low_signal_thresholds["std_floor"])

    core_threshold = max(
        float(np.quantile(gray, 0.992)),
        bg_median + max(5.5 * bg_std, 0.07),
    )
    # Stage 7 output is already transformed: a genuinely bright/colorful core
    # can have luminance well below an absolute 0.82 floor while one channel is
    # close to clipping.  Keep an explicit hard seed so the core does not vanish
    # when the top quantile is a flat plateau, then add a feathered boundary.
    core_seed_mask = (
        (gray >= core_threshold)
        | (channel_peak >= 0.985)
    ).astype(np.float32)
    core_hard_mask = dilate_mask(core_seed_mask, iterations=2)
    core_feather = feather_mask(
        dilate_mask(core_hard_mask, iterations=2),
        radius=3,
    )
    core_mask = np.maximum(core_hard_mask, core_feather).astype(np.float32)

    limited_core_expand = _clamp_int(
        getattr(
            getattr(pipeline, "cfg", None),
            "stage8_limited_core_exclusion_expand",
            8,
        ),
        2,
        16,
    )
    limited_core_hard_mask = dilate_mask(
        core_hard_mask,
        iterations=limited_core_expand,
    )
    limited_core_feather = feather_mask(
        dilate_mask(limited_core_hard_mask, iterations=2),
        radius=2,
    )
    limited_core_exclusion_mask = np.maximum(
        limited_core_hard_mask,
        limited_core_feather,
    ).astype(np.float32)

    nebula_threshold = max(
        float(np.quantile(gray, 0.68)),
        bg_median + max(1.6 * bg_std, nebula_floor),
    )
    nebula_ramp = (gray - nebula_threshold) / max(ramp_floor, 3.0 * max(bg_std, std_floor))
    nebula_mask = pipeline._stage8_soften_mask(np.clip(nebula_ramp, 0.0, 1.0), passes=3)
    nebula_mask = np.clip(nebula_mask * (1.0 - 0.90 * core_mask), 0.0, 1.0)

    faint_threshold = max(
        float(np.quantile(gray, 0.40)),
        bg_median + max(0.55 * bg_std, faint_floor),
    )
    faint_ramp = (gray - faint_threshold) / max(
        nebula_threshold - faint_threshold,
        2.0 * max(bg_std, std_floor),
        ramp_floor,
    )
    faint_nebula_mask = pipeline._stage8_soften_mask(np.clip(faint_ramp, 0.0, 1.0), passes=4)
    faint_nebula_mask = np.clip(
        faint_nebula_mask * (1.0 - 0.85 * nebula_mask) * (1.0 - core_mask),
        0.0,
        1.0,
    )

    protected_signal = np.maximum.reduce([core_mask, nebula_mask, faint_nebula_mask])
    background_mask = pipeline._stage8_soften_mask(1.0 - protected_signal, passes=2)
    background_mask = np.clip(background_mask, 0.0, 1.0)

    coverage = {
        "core": float(np.mean(core_mask > 0.12)),
        "limited_core_exclusion": float(
            np.mean(limited_core_exclusion_mask > 0.12)
        ),
        "limited_core_exclusion_hard": float(
            np.mean(limited_core_hard_mask > 0.50)
        ),
        "nebula": float(np.mean(nebula_mask > 0.12)),
        "faint_nebula": float(np.mean(faint_nebula_mask > 0.12)),
        "background": float(np.mean(background_mask > 0.50)),
    }
    return {
        "rgb": rgb,
        "gray": gray,
        "bg_median": bg_median,
        "bg_std": bg_std,
        "core_threshold": core_threshold,
        "core_channel_peak_threshold": 0.985,
        "core_seed_mask": core_seed_mask,
        "core_hard_mask": core_hard_mask,
        "core_mask": core_mask,
        "limited_core_exclusion_expand": limited_core_expand,
        "limited_core_exclusion_hard_mask": limited_core_hard_mask,
        "limited_core_exclusion_mask": limited_core_exclusion_mask,
        "nebula_mask": nebula_mask,
        "faint_nebula_mask": faint_nebula_mask,
        "background_mask": background_mask,
        "coverage": coverage,
    }


def stage8_starless_readiness_report(
    pipeline,
    image_data: np.ndarray,
    masks: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Describe Starless artifact risks without introducing new hard gates."""
    raw = np.asarray(image_data)
    finite_ratio = float(np.mean(np.isfinite(raw))) if raw.size else 0.0
    resolved_masks = (
        dict(masks)
        if isinstance(masks, dict) and masks
        else stage8_generate_starless_masks(pipeline, image_data)
    )
    rgb = np.asarray(resolved_masks["rgb"], dtype=np.float32)
    gray = np.asarray(resolved_masks["gray"], dtype=np.float32)
    background = np.asarray(
        resolved_masks["background_mask"], dtype=np.float32
    )
    subject = np.maximum.reduce(
        (
            np.asarray(resolved_masks["core_mask"], dtype=np.float32),
            np.asarray(resolved_masks["nebula_mask"], dtype=np.float32),
            np.asarray(resolved_masks["faint_nebula_mask"], dtype=np.float32),
        )
    )
    local_gray = _box_blur_gray(gray)
    high_frequency = gray - local_gray
    hf_median = float(np.median(high_frequency))
    hf_mad = float(np.median(np.abs(high_frequency - hf_median)))
    artifact_floor = max(1.4826 * hf_mad * 8.0, 0.006)
    subject_support = subject > 0.12
    background_support = (background > 0.60) & (~subject_support)

    def ratio(mask: np.ndarray, condition: np.ndarray) -> Optional[float]:
        count = int(np.count_nonzero(mask))
        if count < 32:
            return None
        return float(np.count_nonzero(condition & mask) / count)

    dark_hole_proxy = ratio(
        subject_support,
        high_frequency < -artifact_floor,
    )
    ringing_proxy = ratio(
        subject_support,
        np.abs(high_frequency) > artifact_floor,
    )
    chroma = np.max(rgb, axis=0) - np.min(rgb, axis=0)
    local_chroma = _box_blur_gray(chroma.astype(np.float32))
    chroma_residual = chroma - local_chroma
    background_chroma = chroma_residual[background_support]
    if background_chroma.size >= 32:
        chroma_median = float(np.median(background_chroma))
        chroma_mad = float(
            np.median(np.abs(background_chroma - chroma_median))
        )
        chroma_floor = max(1.4826 * chroma_mad * 8.0, 0.004)
        chroma_fragment_proxy = float(
            np.mean(np.abs(background_chroma - chroma_median) > chroma_floor)
        )
    else:
        chroma_floor = None
        chroma_fragment_proxy = None

    starmask_file = getattr(pipeline, "starmask_file", None)
    starmask_available = bool(
        starmask_file is not None
        and callable(getattr(starmask_file, "exists", None))
        and starmask_file.exists()
    )
    return {
        "schema": "starun.stage8-starless-readiness.v1",
        "status": "reported",
        "mode": "report_only",
        "used_for_gate": False,
        "finite_ratio": finite_ratio,
        "mask_coverage": dict(resolved_masks.get("coverage", {})),
        "artifact_proxies": {
            "dark_hole_subject_ratio": dark_hole_proxy,
            "ringing_subject_ratio": ringing_proxy,
            "background_chroma_fragment_ratio": chroma_fragment_proxy,
            "high_frequency_data_driven_floor": artifact_floor,
            "background_chroma_data_driven_floor": chroma_floor,
        },
        "reference_availability": {
            "starmask": starmask_available,
            "registered_pre_starless_reference": False,
            "filament_topology_comparison": "unavailable",
        },
        "limitations": [
            "artifact proxies are diagnostic and are not automatic thresholds",
            "filament/structure deletion requires a registered pre-starless reference",
            "missing structure is never synthesized by this report",
        ],
    }


def build_signal_excluded_background_masks(
    pipeline,
    image_data: np.ndarray,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Freeze background coordinates while excluding target/galaxy signal."""
    masks = dict(stage8_generate_starless_masks(pipeline, image_data))
    target_type = (
        str(pipeline._active_target_type() or "").strip().lower()
        if hasattr(pipeline, "_active_target_type")
        else ""
    )
    galaxy_report: Dict[str, Any] = {
        "applicable": target_type in {"galaxy", "large_galaxy", "small_galaxy"},
        "available": False,
    }
    if galaxy_report["applicable"]:
        galaxy_masks = stage7_quality.stage7_galaxy_structure_masks(
            np.asarray(masks["gray"], dtype=np.float32),
            float(masks.get("bg_median", 0.0) or 0.0),
            float(masks.get("bg_std", 0.0) or 0.0),
        )
        galaxy_report.update(
            {
                "available": bool(galaxy_masks.get("available", False)),
                "reason": galaxy_masks.get("reason"),
            }
        )
        if galaxy_report["available"]:
            disk_mask = np.asarray(galaxy_masks["disk_mask"], dtype=np.float32)
            if disk_mask.shape == np.asarray(masks["background_mask"]).shape:
                masks["galaxy_signal_mask"] = np.clip(disk_mask, 0.0, 1.0)
                galaxy_report["coverage"] = float(np.mean(disk_mask > 0.12))
            else:
                galaxy_report.update(
                    available=False,
                    reason="galaxy_mask_shape_mismatch",
                )

    signal_keys = [
        key
        for key in (
            "core_mask",
            "nebula_mask",
            "faint_nebula_mask",
            "galaxy_signal_mask",
        )
        if masks.get(key) is not None
    ]
    exclusive = _stage8_exclusive_background_weight(
        masks,
        np.asarray(masks["background_mask"], dtype=np.float32),
    )
    report = {
        "status": "available",
        "method": "frozen_signal_excluded_background_mask_v2",
        "candidate_independent": True,
        "signal_exclusion_applied": bool(signal_keys),
        "signal_exclusion_keys": signal_keys,
        "coverage_gt_0_50": float(np.mean(exclusive > 0.50)),
        "galaxy_signal_exclusion": galaxy_report,
    }
    return masks, report

def stage8_masked_metrics(
    pipeline,
    image_data: Optional[np.ndarray],
    masks: Optional[Dict[str, Any]],
) -> Dict[str, float]:
    if image_data is None or not masks:
        return {}
    try:
        rgb = _to_rgb_float_fullres(image_data)
        gray = (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]).astype(np.float32)
        blurred = _box_blur_gray(gray)
        background = np.asarray(masks["background_mask"], dtype=np.float32)
        core = np.asarray(masks["core_mask"], dtype=np.float32)
        nebula = np.asarray(masks["nebula_mask"], dtype=np.float32)
        faint = np.asarray(masks["faint_nebula_mask"], dtype=np.float32)

        def weighted_mean(values: np.ndarray, weight: np.ndarray) -> float:
            total = float(np.sum(weight))
            if total <= 1e-6:
                return 0.0
            return float(np.sum(values * weight) / total)

        def weighted_std(values: np.ndarray, weight: np.ndarray) -> float:
            total = float(np.sum(weight))
            if total <= 1e-6:
                return 0.0
            mean = float(np.sum(values * weight) / total)
            return float(np.sqrt(np.sum(((values - mean) ** 2) * weight) / total))

        background_weight = _stage8_exclusive_background_weight(masks, background)
        core_weight = np.clip(core, 0.0, 1.0)
        diffuse_weight = np.clip(nebula + 0.65 * faint, 0.0, 1.0)
        texture_weight = np.clip(background + 0.35 * faint, 0.0, 1.0)
        return {
            "background_median": weighted_mean(gray, background_weight),
            "background_std": weighted_std(gray, background_weight),
            "background_brightness": weighted_mean(gray, background_weight),
            "core_clip_ratio": weighted_mean(
                ((gray >= 0.985) | (np.max(rgb, axis=0) >= 0.995)).astype(np.float32),
                core_weight,
            ),
            "diffuse_signal": weighted_mean(np.clip(gray, 0.0, 1.0), diffuse_weight),
            "texture_artifact_score": weighted_mean(np.abs(gray - blurred), texture_weight),
        }
    except (TypeError, ValueError, IndexError, FloatingPointError) as e:
        pipeline.log.warn(f"stage8 masked metrics unavailable: {e}")
        return {}

def background_quality_metrics(
    pipeline,
    image_data: Optional[np.ndarray],
    masks: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if image_data is None:
        return {}
    try:
        rgb = _to_rgb_float_fullres(image_data)
        gray = (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]).astype(np.float32)
        if masks and "background_mask" in masks:
            background = np.asarray(masks["background_mask"], dtype=np.float32)
            bg_weight = _stage8_exclusive_background_weight(masks, background)
        else:
            bg_weight = (gray <= float(np.quantile(gray, 0.35))).astype(np.float32)
        if float(np.sum(bg_weight)) <= 1e-6:
            bg_weight = np.ones_like(gray, dtype=np.float32)

        def weighted_mean(values: np.ndarray) -> float:
            total = max(float(np.sum(bg_weight)), 1e-6)
            return float(np.sum(values * bg_weight) / total)

        def weighted_std(values: np.ndarray) -> float:
            mean = weighted_mean(values)
            total = max(float(np.sum(bg_weight)), 1e-6)
            return float(np.sqrt(np.sum(((values - mean) ** 2) * bg_weight) / total))

        # Channel spread is colour magnitude, not noise.  Measure chroma noise
        # from only the high-frequency part of the two opponent-colour planes,
        # otherwise smooth H-alpha nebulosity is misclassified as background
        # noise after a nonlinear stretch.
        chroma_bias = np.std(rgb, axis=0)
        red_green = rgb[0] - rgb[1]
        blue_green = rgb[2] - rgb[1]
        red_green_noise = red_green - _box_blur_gray(red_green)
        blue_green_noise = blue_green - _box_blur_gray(blue_green)
        chroma_noise = np.sqrt(
            (red_green_noise * red_green_noise + blue_green_noise * blue_green_noise)
            * 0.5
        )
        blur1 = _box_blur_gray(gray)
        blur3 = gray.copy()
        for _ in range(3):
            blur3 = _box_blur_gray(blur3)
        local_texture = np.abs(gray - blur1)
        mottling = np.abs(blur1 - blur3)
        patch = 16
        h, w = gray.shape
        patch_vars: List[float] = []
        patch_residual_rms: List[float] = []
        texture_residual = gray - blur1
        for y in range(0, max(h - patch + 1, 1), patch):
            for x in range(0, max(w - patch + 1, 1), patch):
                tile_weight = bg_weight[y:y + patch, x:x + patch]
                if tile_weight.size and float(np.mean(tile_weight)) > 0.65:
                    tile_gray = gray[y:y + patch, x:x + patch]
                    tile_residual = texture_residual[y:y + patch, x:x + patch]
                    patch_vars.append(float(np.var(tile_gray)))
                    weight_total = max(float(np.sum(tile_weight)), 1e-6)
                    residual_mean = float(
                        np.sum(tile_residual * tile_weight) / weight_total
                    )
                    residual_rms = float(
                        np.sqrt(
                            np.sum(
                                ((tile_residual - residual_mean) ** 2)
                                * tile_weight
                            )
                            / weight_total
                        )
                    )
                    patch_residual_rms.append(residual_rms)
        patch_variance = float(np.median(patch_vars)) if patch_vars else 0.0
        if patch_residual_rms:
            residual_tiles = np.asarray(patch_residual_rms, dtype=np.float64)
            residual_median = float(np.median(residual_tiles))
            residual_p90 = float(np.quantile(residual_tiles, 0.90))
            residual_mad = float(
                np.median(np.abs(residual_tiles - residual_median))
            )
            residual_scale = max(
                1.4826 * residual_mad,
                residual_median * 0.15,
                1e-6,
            )
            residual_z = np.maximum(
                0.0,
                (residual_tiles - residual_median) / residual_scale,
            )
            residual_outlier_score = float(np.quantile(residual_z, 0.90))
            residual_affected_ratio = float(
                np.mean(residual_z > _FINAL_TEXTURE_OUTLIER_SCORE_HARD_MAX)
            )
        else:
            residual_median = 0.0
            residual_p90 = 0.0
            residual_outlier_score = 0.0
            residual_affected_ratio = 0.0
        red = weighted_mean(rgb[0])
        green = weighted_mean(rgb[1])
        blue = weighted_mean(rgb[2])
        background_level = weighted_mean(gray)
        chroma_bias_mean = weighted_mean(chroma_bias)
        background_scale = max(background_level, 1e-4)
        green_excess = green - 0.5 * (red + blue)
        blue_excess = max(0.0, blue / max(green, 1e-6) - max(1.08, red / max(green, 1e-6) + 0.12))
        return {
            "bg_median": background_level,
            "bg_std": weighted_std(gray),
            "bg_dirty_score": _clamp_float(weighted_std(gray) / max(background_level, 0.015), 0.0, 2.0),
            "chroma_noise_score": _clamp_float(weighted_mean(chroma_noise) / max(weighted_std(gray) * 2.0, 0.01), 0.0, 2.0),
            "background_chroma_bias_score": _clamp_float(chroma_bias_mean / max(background_level, 0.015), 0.0, 2.0),
            "background_chroma_load": chroma_bias_mean / background_scale,
            "background_red_mean": red,
            "background_green_mean": green,
            "background_blue_mean": blue,
            "background_green_excess": green_excess / background_scale,
            "blue_excess_score": _clamp_float(blue_excess, 0.0, 2.0),
            "background_mottling_score": _clamp_float(weighted_mean(mottling) / max(weighted_std(gray) * 2.0, 0.006), 0.0, 2.0),
            "local_patch_variance": patch_variance,
            "local_texture_residual_median": residual_median,
            "local_texture_residual_p90": residual_p90,
            "local_texture_residual_outlier_score": residual_outlier_score,
            "local_texture_affected_patch_ratio": residual_affected_ratio,
            "local_texture_patch_count": len(patch_residual_rms),
            "core_clip_score": float(np.mean(gray >= 0.985)),
            "starless_artifact_score": _clamp_float(weighted_mean(local_texture) / max(weighted_std(gray) * 3.0, 0.006), 0.0, 2.0),
        }
    except (TypeError, ValueError, IndexError, FloatingPointError) as e:
        pipeline.log.warn(f"background quality metrics unavailable: {e}")
        return {}


def _stage8_exclusive_background_weight(
    masks: Dict[str, Any],
    background: np.ndarray,
) -> np.ndarray:
    """Keep Stage 8 quality sampling outside every protected signal mask.

    The generated masks are deliberately feathered, so the background mask can
    remain strong where a nebula/faint-signal mask is also active.  Enhancement
    in that overlap must not be reported as new *background* noise.
    """
    bg_weight = np.clip(np.asarray(background, dtype=np.float32), 0.0, 1.0)
    signal_keys = tuple(
        key
        for key in (
            "core_mask",
            "nebula_mask",
            "faint_nebula_mask",
            "galaxy_signal_mask",
        )
        if masks.get(key) is not None
    )
    if not signal_keys:
        return bg_weight
    try:
        signal = np.maximum.reduce(
            [
                np.clip(np.asarray(masks[key], dtype=np.float32), 0.0, 1.0)
                for key in signal_keys
            ]
        )
        if signal.shape != bg_weight.shape:
            return bg_weight
        # 0.12 is the same support threshold used by mask coverage reporting.
        # Retain a short feather below it instead of introducing a hard edge.
        signal_exclusion = np.clip(signal / 0.12, 0.0, 1.0)
        exclusive = bg_weight * (1.0 - signal_exclusion)
        if float(np.sum(exclusive)) >= 32.0:
            return exclusive.astype(np.float32, copy=False)
    except (TypeError, ValueError, FloatingPointError):
        pass
    return bg_weight

def stage8_enhancement_quality_report(pipeline) -> Dict[str, Any]:
    before_data = pipeline._read_image_by_stem("stage8_input_starless")
    after_data = pipeline._read_image_by_stem("stage8_enhanced")
    availability_issues: List[str] = []
    if before_data is None:
        availability_issues.append("stage8_input_starless_unavailable")
    if after_data is None:
        availability_issues.append("stage8_enhanced_unavailable")
    masks = None
    if before_data is not None:
        try:
            masks = pipeline._stage8_generate_starless_masks(before_data)
        except (CommandError, SirilError, RuntimeError, TypeError, ValueError):
            masks = None
    before = pipeline._background_quality_metrics(before_data, masks)
    after = pipeline._background_quality_metrics(after_data, masks)
    if before_data is not None and not before:
        availability_issues.append("stage8_input_quality_metrics_unavailable")
    if after_data is not None and not after:
        availability_issues.append("stage8_enhanced_quality_metrics_unavailable")
    issues: List[str] = list(availability_issues)
    advisories: List[str] = []
    quality_gates: Dict[str, Dict[str, Any]] = {}

    def record_upper_gate(
        metric_name: str,
        value: float,
        limit: float,
        issue_text: str,
    ) -> None:
        gate = stage7_quality.stage7_9_upper_quality_gate(
            pipeline.cfg,
            value=value,
            accepted_limit=limit,
        )
        quality_gates[metric_name] = gate
        if gate["hard_failed"]:
            issues.append(issue_text)
        elif gate["advisory"]:
            advisories.append(f"{issue_text} (advisory; enhancement retained)")

    bg_growth = None
    chroma_growth = None
    if before and after:
        bg_growth = after.get("bg_dirty_score", 0.0) / max(before.get("bg_dirty_score", 0.0), 1e-6)
        chroma_growth = after.get("chroma_noise_score", 0.0) / max(before.get("chroma_noise_score", 0.0), 1e-6)
        dirty_growth_issue = _stage8_bg_noise_growth_issue(
            pipeline,
            growth=bg_growth,
            baseline_std=float(before.get("bg_std", 0.0) or 0.0),
            candidate_std=float(after.get("bg_std", 0.0) or 0.0),
            candidate_dirty_score=float(after.get("bg_dirty_score", 0.0) or 0.0),
            label="bg_dirty_score_growth",
        )
        if dirty_growth_issue:
            record_upper_gate(
                "bg_dirty_score_growth",
                bg_growth,
                float(pipeline.cfg.stage8_bg_std_growth_max),
                dirty_growth_issue,
            )
        if chroma_growth > 1.10:
            record_upper_gate(
                "chroma_noise_score_growth",
                chroma_growth,
                1.10,
                f"chroma_noise_score_growth {chroma_growth:.3f}>1.100",
            )
        if after.get("background_mottling_score", 0.0) > 0.55:
            mottling_score = float(after.get("background_mottling_score", 0.0))
            record_upper_gate(
                "background_mottling_score",
                mottling_score,
                0.55,
                f"background_mottling_score {mottling_score:.3f}>0.550",
            )
    return {
        "stage": "stage8_enhancement",
        "policy": pipeline._active_policy_name() if hasattr(pipeline, "_active_policy_name") else "",
        "target_type": pipeline._active_target_type() if hasattr(pipeline, "_active_target_type") else "",
        "before": before,
        "after": after,
        "quality_before": before,
        "blue_excess_before": before.get("blue_excess_score") if before else None,
        "blue_excess_after": after.get("blue_excess_score") if after else None,
        "bg_std_growth": bg_growth,
        "chroma_noise_growth": chroma_growth,
        "conservative_rerun_applied": False,
        "final_source": pipeline._stage8_final_source,
        "fallback_used": pipeline._stage8_fallback_used,
        "final_quality": pipeline._stage8_final_quality,
        "status": "poor" if issues else "ok",
        "issues": issues,
        "advisories": advisories,
        "quality_gates": quality_gates,
        "quality_advisory_multiplier": (
            stage7_quality.stage7_9_quality_advisory_multiplier(pipeline.cfg)
        ),
    }

def _stage8_bg_noise_growth_issue(
    pipeline,
    *,
    growth: float,
    baseline_std: float,
    candidate_std: float,
    candidate_dirty_score: float,
    label: str = "bg_std_growth",
) -> Optional[str]:
    if growth <= pipeline.cfg.stage8_bg_std_growth_max:
        return None

    absolute_growth = max(0.0, candidate_std - baseline_std)
    low_absolute_noise = (
        candidate_std <= 0.00075
        and absolute_growth <= 0.00050
        and candidate_dirty_score <= 0.060
    )
    if low_absolute_noise:
        return None

    return f"{label} {growth:.3f}>{pipeline.cfg.stage8_bg_std_growth_max:.3f}"

def rollback_stage8_to_input(pipeline) -> bool:
    try:
        pipeline.cmd_with_check("load", "stage8_input_starless")
        pipeline._save_stage_output("stage8_enhanced")
        return True
    except (CommandError, SirilError) as e:
        pipeline.log.warn(f"Stage8 rollback to input failed: {e}")
        return False

def final_quality_report(pipeline, stem: str = "stage10_final") -> Dict[str, Any]:
    image_data = pipeline._read_image_by_stem(stem)
    issues: List[str] = []
    advisories: List[str] = []
    stage4_core_color_integrity = dict(
        (getattr(pipeline, "color_calibration_report", {}) or {}).get(
            "bright_core_color_integrity"
        )
        or {}
    )
    if bool(stage4_core_color_integrity.get("applicable", False)) and str(
        stage4_core_color_integrity.get("status") or ""
    ) not in {"ok", "repaired"}:
        issues.append("stage4_bright_core_color_integrity_unresolved")
    try:
        image_array = np.asarray(image_data)
        opaque_test_double = bool(
            image_array.ndim == 0 and image_array.dtype == np.dtype("O")
        )
        if not opaque_test_double:
            if image_array.size == 0:
                issues.append("final_image_empty")
            if image_array.ndim not in (2, 3):
                issues.append(f"final_image_invalid_dimensions:{image_array.shape}")
            if image_array.size and not np.all(np.isfinite(image_array)):
                issues.append("final_image_non_finite_pixels")
    except (TypeError, ValueError) as error:
        issues.append(f"final_image_validation_failed:{error}")
    unmasked_metrics = pipeline._background_quality_metrics(image_data)
    metrics = dict(unmasked_metrics or {})
    final_masks: Optional[Dict[str, Any]] = None
    final_signal_excluded_background_metrics: Dict[str, Any] = {}
    final_signal_excluded_background_sampling: Dict[str, Any] = {
        "status": "unavailable",
    }
    try:
        frozen_masks = getattr(
            pipeline,
            "_stage10_quality_frozen_background_masks",
            None,
        )
        frozen_sampling = getattr(
            pipeline,
            "_stage10_quality_frozen_background_sampling",
            None,
        )
        if isinstance(frozen_masks, dict) and frozen_masks:
            final_masks = frozen_masks
            final_signal_excluded_background_sampling = (
                dict(frozen_sampling)
                if isinstance(frozen_sampling, dict)
                else {
                    "status": "available",
                    "method": "frozen_signal_excluded_background_mask_v2",
                }
            )
        else:
            final_masks, final_signal_excluded_background_sampling = (
                build_signal_excluded_background_masks(pipeline, image_data)
            )
            pipeline._stage10_quality_frozen_background_masks = final_masks
            pipeline._stage10_quality_frozen_background_sampling = dict(
                final_signal_excluded_background_sampling
            )
        coverage_value = final_signal_excluded_background_sampling.get(
            "coverage_gt_0_50"
        )
        if coverage_value is not None and float(coverage_value) <= 0.01:
            raise ValueError(
                "signal-excluded background coverage is insufficient: "
                f"{float(coverage_value):.6f}<=0.010000"
            )
        final_signal_excluded_background_metrics = (
            pipeline._background_quality_metrics(image_data, final_masks)
        )
        if not final_signal_excluded_background_metrics:
            raise ValueError("empty signal-excluded background metrics")
        metrics = dict(final_signal_excluded_background_metrics)
    except (
        AttributeError,
        IndexError,
        RuntimeError,
        TypeError,
        ValueError,
        FloatingPointError,
    ) as error:
        final_masks = None
        try:
            pipeline._stage10_quality_frozen_background_masks = None
            pipeline._stage10_quality_frozen_background_sampling = None
        except AttributeError:
            pass
        final_signal_excluded_background_metrics = {}
        final_signal_excluded_background_sampling = {
            "status": "unavailable",
            "method": "darkest_35_percent_fallback",
            "error": str(error),
        }
        advisories.append(
            "signal_excluded_background_sampling_unavailable "
            "(advisory; darkest 35% fallback retained)"
        )

    baseline_metrics: Dict[str, Any] = {}
    baseline_sampling: Dict[str, Any] = {"status": "not_available"}
    baseline_stem = str(
        getattr(pipeline, "_stage10_quality_baseline_stem", "") or ""
    )
    if baseline_stem and baseline_stem != stem:
        try:
            baseline_image = pipeline._read_image_by_stem(baseline_stem)
            baseline_metrics = pipeline._background_quality_metrics(
                baseline_image,
                final_masks,
            )
            if not baseline_metrics:
                raise ValueError("empty Stage10 baseline metrics")
            baseline_sampling = {
                "status": "available",
                "stem": baseline_stem,
                "shared_background_mask": bool(final_masks),
            }
        except (
            AttributeError,
            IndexError,
            RuntimeError,
            TypeError,
            ValueError,
            FloatingPointError,
        ) as error:
            baseline_metrics = {}
            baseline_sampling = {
                "status": "unavailable",
                "stem": baseline_stem,
                "error": str(error),
            }
            advisories.append(
                "stage10_input_noise_baseline_unavailable "
                "(advisory; absolute final metrics used)"
            )
    background_color_review_gate = getattr(
        pipeline,
        "_stage7_background_color_review_gate",
        {},
    ) or {}
    background_color_review_gate = (
        background_color_review_gate
        if isinstance(background_color_review_gate, dict)
        else {}
    )
    stage7_background_color_review_required_raw = bool(
        getattr(
            pipeline,
            "_stage7_background_color_review_required",
            False,
        )
        or background_color_review_gate.get("requires_review", False)
    )
    stage7_forced_delivery = bool(
        getattr(pipeline, "_stage7_stretch_forced_delivery", False)
    )
    stage7_background_color_review_required = bool(
        stage7_background_color_review_required_raw
        and not stage7_forced_delivery
    )
    final_background_color_review_required = False
    final_background_chroma_load: Optional[float] = None
    final_background_color_quality_gate: Dict[str, Any] = {}
    if bool(background_color_review_gate.get("applicable", False)):
        try:
            if not final_signal_excluded_background_metrics:
                raise ValueError("signal-excluded background metrics unavailable")
            final_load = float(
                final_signal_excluded_background_metrics.get(
                    "background_chroma_load"
                )
            )
            final_limit = float(background_color_review_gate.get("limit", 0.12))
            if not math.isfinite(final_load):
                raise ValueError("non-finite final background chroma load")
            final_background_chroma_load = final_load
            final_background_color_quality_gate = (
                stage7_quality.stage7_9_upper_quality_gate(
                    getattr(pipeline, "cfg", None),
                    value=final_load,
                    accepted_limit=final_limit,
                )
            )
            final_background_color_review_required = bool(
                final_background_color_quality_gate.get("hard_failed", False)
            )
            if final_background_color_quality_gate.get("advisory", False):
                advisories.append(
                    "uncalibrated_final_background_chroma_load "
                    f"{final_load:.3f}>{final_limit:.3f} "
                    "(advisory; final output retained)"
                )
        except (
            AttributeError,
            IndexError,
            RuntimeError,
            TypeError,
            ValueError,
            FloatingPointError,
        ) as error:
            final_signal_excluded_background_metrics = {}
            final_signal_excluded_background_sampling = {
                "status": "unavailable",
                "error": str(error),
            }
            final_background_color_review_required = True
    uncalibrated_background_color_review_required_raw = bool(
        stage7_background_color_review_required_raw
        or final_background_color_review_required
    )
    uncalibrated_background_color_review_required = bool(
        uncalibrated_background_color_review_required_raw
        and not stage7_forced_delivery
    )
    if (
        stage7_forced_delivery
        and (
            stage7_background_color_review_required_raw
            or final_background_color_review_required
        )
    ):
        advisories.append(
            "stage7_forced_delivery_overrode_background_colour_review "
            "(appearance-only; technical gates remained enforced)"
        )
    halo_residue = pipeline._stage7_halo_residue_score()
    selected_stage7_quality = getattr(pipeline, "_stage7_selected_quality", None)
    stage7_quality_derived = (
        selected_stage7_quality.get("derived") or {}
        if isinstance(selected_stage7_quality, dict)
        else {}
    )
    try:
        compact_halo_residue = float(
            stage7_quality_derived.get("compact_halo_residue_score", 0.0) or 0.0
        )
    except (TypeError, ValueError):
        compact_halo_residue = 0.0
    try:
        global_halo_residue = float(
            stage7_quality_derived.get("global_halo_residue_score", halo_residue)
            or 0.0
        )
    except (TypeError, ValueError):
        global_halo_residue = halo_residue
    stage7_halo_limit = 0.70
    if hasattr(pipeline, "_stage7_effective_halo_threshold"):
        try:
            stage7_halo_limit = float(pipeline._stage7_effective_halo_threshold())
        except (TypeError, ValueError):
            stage7_halo_limit = 0.70
    compact_halo_raw_limit_exceeded = bool(
        compact_halo_residue > stage7_halo_limit
    )
    bypassed_bad_starless = bool(getattr(pipeline, "_stage9_bypassed_bad_starless", False))
    stage9_contract_known = hasattr(pipeline, "_stage9_stars_applied")
    stage9_stars_required = bool(
        getattr(pipeline, "_stage9_stars_required", False)
    )
    stage9_stars_applied = bool(
        getattr(pipeline, "_stage9_stars_applied", False)
    )
    stage9_remix_formally_accepted = bool(
        getattr(
            pipeline,
            "_stage9_remix_formally_accepted",
            stage9_stars_applied,
        )
    )
    stage9_review_candidate_selected = bool(
        getattr(pipeline, "_stage9_review_candidate_selected", False)
    )
    stage9_missing_required_stars = bool(
        stage9_contract_known
        and stage9_stars_required
        and not stage9_stars_applied
    )
    stage9_starmask_stretch_failed = bool(
        getattr(pipeline, "_stage9_starmask_stretch_failed", False)
    )
    stage9_starmask_preparation_failed = bool(
        getattr(pipeline, "_stage9_starmask_preparation_failed", False)
    )
    stage9_star_reference_degraded = bool(
        getattr(pipeline, "_stage9_star_reference_degraded", False)
    )
    selected_stage9_quality = getattr(
        pipeline,
        "_stage9_selected_remix_quality",
        None,
    )
    stage9_quality_metrics = (
        selected_stage9_quality.get("metrics") or {}
        if isinstance(selected_stage9_quality, dict)
        else {}
    )
    stage9_quality_limits = (
        selected_stage9_quality.get("limits") or {}
        if isinstance(selected_stage9_quality, dict)
        else {}
    )
    advisories.extend(
        str(item)
        for item in (
            selected_stage9_quality.get("advisories") or []
            if isinstance(selected_stage9_quality, dict)
            else []
        )
        if str(item)
    )
    stage9_chromatic_addition_ratio = None
    if "chromatic_star_addition_ratio" in stage9_quality_metrics:
        try:
            stage9_chromatic_addition_ratio = float(
                stage9_quality_metrics["chromatic_star_addition_ratio"]
            )
        except (TypeError, ValueError):
            stage9_chromatic_addition_ratio = None
    try:
        stage9_local_color_risk_score = float(
            stage9_quality_metrics.get("local_color_risk_score", 0.0) or 0.0
        )
    except (TypeError, ValueError):
        stage9_local_color_risk_score = 0.0
    stage10_saturation_guard = getattr(
        pipeline,
        "_stage10_saturation_guard",
        None,
    )
    stage9_chromatic_addition_limit_value = stage9_quality_limits.get(
        "chromatic_star_addition_ratio",
        getattr(
            getattr(pipeline, "cfg", None),
            "stage9_chromatic_addition_ratio_max",
            0.003,
        ),
    )
    try:
        stage9_chromatic_addition_limit = float(
            stage9_chromatic_addition_limit_value
        )
    except (TypeError, ValueError):
        stage9_chromatic_addition_limit = 0.003
    stage9_chromatic_addition_gate: Dict[str, Any] = {}
    if stage9_chromatic_addition_ratio is not None:
        stage9_chromatic_addition_gate = (
            stage7_quality.stage7_9_upper_quality_gate(
                getattr(pipeline, "cfg", None),
                value=stage9_chromatic_addition_ratio,
                accepted_limit=stage9_chromatic_addition_limit,
            )
        )
    conservative_stage8_skip = (
        str(getattr(pipeline, "_stage8_final_quality", "")) == "conservative_skipped"
    )
    active_target_type = (
        pipeline._active_target_type()
        if hasattr(pipeline, "_active_target_type")
        else ""
    )
    try:
        effective_stage7_halo = float(
            stage7_quality_derived.get("halo_residue_score", halo_residue)
            or 0.0
        )
    except (TypeError, ValueError):
        effective_stage7_halo = halo_residue
    try:
        compact_residual_star_score = float(
            stage7_quality_derived.get("compact_residual_star_score", 1.0)
            or 0.0
        )
    except (TypeError, ValueError):
        compact_residual_star_score = 1.0
    try:
        compact_residual_coverage = float(
            stage7_quality_derived.get("compact_residual_coverage", 1.0)
            or 0.0
        )
    except (TypeError, ValueError):
        compact_residual_coverage = 1.0
    try:
        final_starless_artifact = float(
            metrics.get("starless_artifact_score", 1.0) if metrics else 1.0
        )
    except (TypeError, ValueError):
        final_starless_artifact = 1.0
    stage7_status = (
        str(selected_stage7_quality.get("status", ""))
        if isinstance(selected_stage7_quality, dict)
        else ""
    )
    # A compact-halo score just over the threshold can be contaminated by real
    # bright-nebula structure. Exempt only the narrow case already accepted by
    # Stage 6/7 and independently confirmed by safe Stage 8 plus an applied,
    # quality-gated star remix. Larger/unsupported compact halos still fail.
    compact_halo_target_aware_exempted = bool(
        compact_halo_raw_limit_exceeded
        and active_target_type == "bright_emission_reflection_nebula"
        and compact_halo_residue
        <= stage7_halo_limit * _FINAL_COMPACT_HALO_NEAR_LIMIT_FACTOR
        and global_halo_residue
        <= stage7_halo_limit * _FINAL_GLOBAL_HALO_SAFE_FACTOR
        and effective_stage7_halo <= stage7_halo_limit
        and compact_residual_star_score
        <= _FINAL_COMPACT_RESIDUAL_SCORE_SAFE_MAX
        and compact_residual_coverage
        <= _FINAL_COMPACT_RESIDUAL_COVERAGE_SAFE_MAX
        and stage7_status == "ok"
        and str(getattr(pipeline, "_stage8_final_quality", "")) == "ok"
        and not bool(getattr(pipeline, "_stage8_fallback_used", False))
        and not bypassed_bad_starless
        and stage9_contract_known
        and stage9_stars_required
        and stage9_stars_applied
        and stage9_remix_formally_accepted
        and not stage9_review_candidate_selected
        and not stage9_starmask_preparation_failed
        and not stage9_starmask_stretch_failed
        and final_starless_artifact <= _FINAL_STARLESS_ARTIFACT_SAFE_MAX
    )
    compact_halo_limit_exceeded = bool(
        compact_halo_raw_limit_exceeded
        and not compact_halo_target_aware_exempted
    )
    strict_gate = (
        (
            bool(getattr(pipeline, "_stage8_fallback_used", False))
            and not conservative_stage8_skip
        )
        or bypassed_bad_starless
        or stage9_missing_required_stars
        or stage9_starmask_preparation_failed
        or stage9_starmask_stretch_failed
        or uncalibrated_background_color_review_required
        or halo_residue > 0.70
        or compact_halo_limit_exceeded
    )
    if uncalibrated_background_color_review_required:
        value = (
            final_background_chroma_load
            if final_background_color_review_required
            else background_color_review_gate.get("value")
        )
        limit = background_color_review_gate.get("limit")
        if final_background_color_review_required and value is None:
            issues.append(
                "uncalibrated_final_signal_excluded_background_chroma_load_"
                "unavailable"
            )
        elif value is None:
            issues.append(
                "uncalibrated_signal_excluded_background_chroma_load_unavailable"
            )
        else:
            issues.append(
                "uncalibrated_background_chroma_load "
                f"{float(value):.3f}>{float(limit or 0.12):.3f}"
            )
    if compact_halo_limit_exceeded:
        issues.append(
            "stage7_compact_halo_residue_score "
            f"{compact_halo_residue:.3f}>{stage7_halo_limit:.3f}"
        )
    if strict_gate:
        patch_limit = _FINAL_LOCAL_PATCH_VARIANCE_STRICT_MAX
    elif active_target_type == "large_galaxy":
        try:
            configured_patch_limit = float(
                getattr(
                    getattr(pipeline, "cfg", None),
                    "stage10_large_galaxy_local_patch_variance_max",
                    _FINAL_LARGE_GALAXY_LOCAL_PATCH_VARIANCE_MAX,
                )
            )
        except (TypeError, ValueError):
            configured_patch_limit = _FINAL_LARGE_GALAXY_LOCAL_PATCH_VARIANCE_MAX
        patch_limit = _clamp_float(
            configured_patch_limit,
            _FINAL_LOCAL_PATCH_VARIANCE_MAX,
            0.00100,
        )
    else:
        patch_limit = _FINAL_LOCAL_PATCH_VARIANCE_MAX
    noise_hard_issues: List[str] = []
    noise_warning_metrics: List[str] = []
    noise_growth: Dict[str, Any] = {}
    chroma_limit = 0.34 if strict_gate else 0.42
    mottling_limit = 0.45 if strict_gate else 0.55
    artifact_limit = 0.52 if strict_gate else 0.62
    if not metrics:
        issues.append("final_quality_metrics_unavailable")
    else:
        required_metric_names = (
            "chroma_noise_score",
            "background_mottling_score",
            "local_patch_variance",
            "core_clip_score",
            "starless_artifact_score",
        )
        invalid_metric_names: List[str] = []
        for metric_name in required_metric_names:
            try:
                metric_value = float(metrics[metric_name])
            except (KeyError, TypeError, ValueError):
                invalid_metric_names.append(metric_name)
                continue
            if not math.isfinite(metric_value):
                invalid_metric_names.append(metric_name)
        if invalid_metric_names:
            issues.append(
                "final_quality_metrics_invalid:"
                + ",".join(invalid_metric_names)
            )

        chroma = float(metrics.get("chroma_noise_score", 0.0) or 0.0)
        mottling = float(metrics.get("background_mottling_score", 0.0) or 0.0)
        patch_var = float(metrics.get("local_patch_variance", 0.0) or 0.0)
        texture_p90 = float(
            metrics.get("local_texture_residual_p90", 0.0) or 0.0
        )
        texture_outlier = float(
            metrics.get("local_texture_residual_outlier_score", 0.0) or 0.0
        )
        texture_affected = float(
            metrics.get("local_texture_affected_patch_ratio", 0.0) or 0.0
        )
        core_clip = float(metrics.get("core_clip_score", 0.0) or 0.0)
        artifact = float(metrics.get("starless_artifact_score", 0.0) or 0.0)

        if chroma > chroma_limit:
            noise_warning_metrics.append("chroma")
            advisories.append(
                f"background_chroma_noise_score {chroma:.3f}>{chroma_limit:.3f} "
                "(advisory unless extreme or corroborated by growth)"
            )
        if mottling > mottling_limit:
            noise_warning_metrics.append("mottling")
            advisories.append(
                f"background_mottling_score {mottling:.3f}>{mottling_limit:.3f} "
                "(advisory unless extreme or corroborated by growth)"
            )
        if patch_var > patch_limit:
            noise_warning_metrics.append("texture")
            advisories.append(
                f"local_patch_variance {patch_var:.6f}>{patch_limit:.6f} "
                "(diagnostic advisory; residual texture gate used)"
            )
        elif (
            texture_outlier > _FINAL_TEXTURE_OUTLIER_SCORE_HARD_MAX
            or texture_affected > 0.10
        ):
            noise_warning_metrics.append("texture")
            advisories.append(
                "local_texture_residual "
                f"outlier={texture_outlier:.3f}, affected={texture_affected:.3f} "
                "(advisory unless spatially extreme)"
            )
        if artifact > artifact_limit:
            noise_warning_metrics.append("artifact")
            advisories.append(
                f"starless_artifact_score {artifact:.3f}>{artifact_limit:.3f} "
                "(advisory unless extreme or corroborated by growth)"
            )

        if chroma > _FINAL_NOISE_EXTREME_MAX:
            noise_hard_issues.append(
                "background_chroma_noise_extreme "
                f"{chroma:.3f}>{_FINAL_NOISE_EXTREME_MAX:.3f}"
            )
        if mottling > _FINAL_NOISE_EXTREME_MAX:
            noise_hard_issues.append(
                "background_mottling_extreme "
                f"{mottling:.3f}>{_FINAL_NOISE_EXTREME_MAX:.3f}"
            )
        if artifact > _FINAL_NOISE_EXTREME_MAX:
            noise_hard_issues.append(
                "starless_artifact_extreme "
                f"{artifact:.3f}>{_FINAL_NOISE_EXTREME_MAX:.3f}"
            )
        if (
            texture_outlier > _FINAL_TEXTURE_OUTLIER_SCORE_HARD_MAX
            and texture_affected > _FINAL_TEXTURE_AFFECTED_RATIO_HARD_MAX
        ):
            noise_hard_issues.append(
                "local_texture_residual_extreme "
                f"outlier={texture_outlier:.3f}>"
                f"{_FINAL_TEXTURE_OUTLIER_SCORE_HARD_MAX:.3f}, "
                f"affected={texture_affected:.3f}>"
                f"{_FINAL_TEXTURE_AFFECTED_RATIO_HARD_MAX:.3f}"
            )

        def metric_growth(
            name: str,
            value: float,
            *,
            absolute_min: float,
        ) -> Dict[str, Any]:
            try:
                baseline_value = float(baseline_metrics[name])
            except (KeyError, TypeError, ValueError):
                return {
                    "available": False,
                    "significant": False,
                    "value": value,
                    "baseline": None,
                }
            if not math.isfinite(baseline_value) or baseline_value < 0.0:
                return {
                    "available": False,
                    "significant": False,
                    "value": value,
                    "baseline": None,
                }
            absolute_growth = value - baseline_value
            ratio = value / max(baseline_value, 1e-9)
            return {
                "available": True,
                "significant": bool(
                    ratio > _FINAL_NOISE_GROWTH_RATIO_HARD_MAX
                    and absolute_growth > absolute_min
                ),
                "value": value,
                "baseline": baseline_value,
                "ratio": ratio,
                "absolute_growth": absolute_growth,
            }

        noise_growth = {
            "chroma": metric_growth(
                "chroma_noise_score",
                chroma,
                absolute_min=_FINAL_NOISE_ABSOLUTE_GROWTH_HARD_MIN,
            ),
            "mottling": metric_growth(
                "background_mottling_score",
                mottling,
                absolute_min=_FINAL_NOISE_ABSOLUTE_GROWTH_HARD_MIN,
            ),
            "artifact": metric_growth(
                "starless_artifact_score",
                artifact,
                absolute_min=_FINAL_NOISE_ABSOLUTE_GROWTH_HARD_MIN,
            ),
            "texture": metric_growth(
                "local_texture_residual_p90",
                texture_p90,
                absolute_min=_FINAL_TEXTURE_P90_GROWTH_HARD_MIN,
            ),
        }
        warning_metric_names = list(dict.fromkeys(noise_warning_metrics))
        corroborated_growth = any(
            bool(noise_growth.get(name, {}).get("significant", False))
            for name in warning_metric_names
        )
        if len(warning_metric_names) >= 2 and corroborated_growth:
            noise_hard_issues.append(
                "background_noise_combined_growth "
                f"metrics={','.join(warning_metric_names)}"
            )

        issues.extend(noise_hard_issues)
        if core_clip > 0.012:
            issues.append(f"core_clip_score {core_clip:.4f}>0.0120")
        if strict_gate and not issues:
            if stage9_missing_required_stars:
                issues.append("stage9_required_stars_not_applied")
            elif stage9_starmask_stretch_failed:
                issues.append("stage9_starmask_stretch_failed")
            else:
                issues.append(
                    "strict_gate_requires_review_due_to_stage8_fallback_"
                    "or_stage9_bypass_or_stage7_halo"
                )
    if (
        stage9_missing_required_stars
        and "stage9_required_stars_not_applied" not in issues
    ):
        issues.append("stage9_required_stars_not_applied")
    if (
        stage9_starmask_preparation_failed
        and "stage9_starmask_preparation_failed" not in issues
    ):
        issues.append("stage9_starmask_preparation_failed")
    if (
        stage9_starmask_stretch_failed
        and "stage9_starmask_stretch_failed" not in issues
    ):
        issues.append("stage9_starmask_stretch_failed")
    if (
        stage9_chromatic_addition_ratio is not None
        and stage9_chromatic_addition_gate.get("hard_failed", False)
    ):
        issues.append(
            "stage9_chromatic_star_addition_ratio "
            f"{stage9_chromatic_addition_ratio:.6f}>"
            f"{stage9_chromatic_addition_limit:.6f}"
        )
    elif stage9_chromatic_addition_gate.get("advisory", False):
        advisory = (
            "stage9_chromatic_star_addition_ratio "
            f"{stage9_chromatic_addition_ratio:.6f}>"
            f"{stage9_chromatic_addition_limit:.6f} "
            "(advisory; final output retained)"
        )
        if advisory not in advisories:
            advisories.append(advisory)
    normalized_metrics = {
        "background_chroma_noise_score": metrics.get("chroma_noise_score") if metrics else None,
        "background_chroma_bias_score": metrics.get("background_chroma_bias_score") if metrics else None,
        "background_chroma_load": metrics.get("background_chroma_load") if metrics else None,
        "uncalibrated_background_color_review_required": (
            uncalibrated_background_color_review_required
        ),
        "uncalibrated_background_color_review_required_raw": (
            uncalibrated_background_color_review_required_raw
        ),
        "stage7_forced_delivery_override": stage7_forced_delivery,
        "uncalibrated_background_color_review_gate": (
            background_color_review_gate
        ),
        "final_signal_excluded_background_metrics": (
            final_signal_excluded_background_metrics
        ),
        "final_signal_excluded_background_sampling": (
            final_signal_excluded_background_sampling
        ),
        "unmasked_background_metrics": unmasked_metrics,
        "stage10_input_background_metrics": baseline_metrics,
        "stage10_input_background_sampling": baseline_sampling,
        "final_background_color_quality_gate": (
            final_background_color_quality_gate
        ),
        "background_mottling_score": metrics.get("background_mottling_score") if metrics else None,
        "local_patch_variance": metrics.get("local_patch_variance") if metrics else None,
        "local_patch_variance_score": metrics.get("local_patch_variance") if metrics else None,
        "local_patch_variance_max": patch_limit,
        "local_texture_residual_median": (
            metrics.get("local_texture_residual_median") if metrics else None
        ),
        "local_texture_residual_p90": (
            metrics.get("local_texture_residual_p90") if metrics else None
        ),
        "local_texture_residual_outlier_score": (
            metrics.get("local_texture_residual_outlier_score")
            if metrics
            else None
        ),
        "local_texture_affected_patch_ratio": (
            metrics.get("local_texture_affected_patch_ratio")
            if metrics
            else None
        ),
        "local_texture_patch_count": (
            metrics.get("local_texture_patch_count") if metrics else None
        ),
        "noise_growth": noise_growth,
        "noise_gate_limits": {
            "chroma_advisory_max": chroma_limit,
            "mottling_advisory_max": mottling_limit,
            "artifact_advisory_max": artifact_limit,
            "extreme_max": _FINAL_NOISE_EXTREME_MAX,
            "texture_outlier_score_hard_max": (
                _FINAL_TEXTURE_OUTLIER_SCORE_HARD_MAX
            ),
            "texture_affected_ratio_hard_max": (
                _FINAL_TEXTURE_AFFECTED_RATIO_HARD_MAX
            ),
            "growth_ratio_hard_max": _FINAL_NOISE_GROWTH_RATIO_HARD_MAX,
            "noise_absolute_growth_hard_min": (
                _FINAL_NOISE_ABSOLUTE_GROWTH_HARD_MIN
            ),
            "texture_p90_growth_hard_min": (
                _FINAL_TEXTURE_P90_GROWTH_HARD_MIN
            ),
        },
        "core_clip_score": metrics.get("core_clip_score") if metrics else None,
        "starless_artifact_score": metrics.get("starless_artifact_score") if metrics else None,
        "halo_artifact_score": halo_residue,
        "stage7_global_halo_residue_score": global_halo_residue,
        "stage7_compact_halo_residue_score": compact_halo_residue,
        "stage7_compact_halo_mask_coverage": stage7_quality_derived.get(
            "compact_halo_mask_coverage"
        ),
        "stage7_compact_halo_source_level": stage7_quality_derived.get(
            "compact_halo_source_level"
        ),
        "stage7_compact_halo_starless_level": stage7_quality_derived.get(
            "compact_halo_starless_level"
        ),
        "stage7_halo_residue_score_max": stage7_halo_limit,
        "stage7_compact_halo_raw_limit_exceeded": (
            compact_halo_raw_limit_exceeded
        ),
        "stage7_compact_halo_target_aware_exempted": (
            compact_halo_target_aware_exempted
        ),
        "bg_dirty_score": metrics.get("bg_dirty_score") if metrics else None,
        "bg_std": metrics.get("bg_std") if metrics else None,
        "stage9_bypassed_bad_starless": bypassed_bad_starless,
        "stage9_stars_required": stage9_stars_required,
        "stage9_stars_applied": stage9_stars_applied,
        "stage9_remix_formally_accepted": stage9_remix_formally_accepted,
        "stage9_review_candidate_selected": stage9_review_candidate_selected,
        "stage9_star_reference_degraded": stage9_star_reference_degraded,
        "stage9_starmask_preparation_failed": (
            stage9_starmask_preparation_failed
        ),
        "stage9_starmask_preparation_failure_reason": str(
            getattr(
                pipeline,
                "_stage9_starmask_preparation_failure_reason",
                "",
            )
            or ""
        ),
        "stage9_starmask_stretch_failed": stage9_starmask_stretch_failed,
        "stage9_chromatic_star_addition_ratio": stage9_chromatic_addition_ratio,
        "stage9_chromatic_star_addition_ratio_max": (
            stage9_chromatic_addition_limit
        ),
        "stage9_chromatic_star_addition_quality_gate": (
            stage9_chromatic_addition_gate
        ),
        "stage9_local_quality_status": str(
            stage9_quality_metrics.get("local_quality_status", "not_available")
        ),
        "stage9_local_color_risk_score": stage9_local_color_risk_score,
        "stage9_local_connected_component_max_area": stage9_quality_metrics.get(
            "local_connected_component_max_area"
        ),
        "stage9_local_single_pixel_component_ratio": stage9_quality_metrics.get(
            "local_single_pixel_component_ratio"
        ),
        "stage9_local_cyan_blue_component_max_area": stage9_quality_metrics.get(
            "local_cyan_blue_component_max_area"
        ),
        "stage9_local_cyan_blue_component_max_area_raw": (
            stage9_quality_metrics.get(
                "local_cyan_blue_component_max_area_raw"
            )
        ),
        "stage9_local_cyan_blue_confirmed_star_pixel_ratio": (
            stage9_quality_metrics.get(
                "local_cyan_blue_confirmed_star_pixel_ratio"
            )
        ),
        "stage9_core_color_jump_component_max_area": stage9_quality_metrics.get(
            "core_color_jump_component_max_area"
        ),
        "stage10_stage9_saturation_factor": (
            stage10_saturation_guard.get("saturation_factor")
            if isinstance(stage10_saturation_guard, dict)
            else None
        ),
        "stage10_effective_final_saturation": (
            stage10_saturation_guard.get("effective_saturation")
            if isinstance(stage10_saturation_guard, dict)
            else None
        ),
        "stage9_stars_application_mode": str(
            getattr(pipeline, "_stage9_stars_application_mode", "unknown")
        ),
        "stage8_conservative_skipped": conservative_stage8_skip,
        "stage4_bright_core_color_integrity": stage4_core_color_integrity,
    }
    hard_issues = list(dict.fromkeys(issues))
    warnings = list(dict.fromkeys(advisories))
    noise_hard_set = set(noise_hard_issues)
    upstream_hard_issues = [
        issue for issue in hard_issues if issue not in noise_hard_set
    ]
    primary_issues = (
        upstream_hard_issues
        if upstream_hard_issues
        else list(dict.fromkeys(noise_hard_issues))
    )
    secondary_diagnostics = list(warnings)
    if upstream_hard_issues:
        secondary_diagnostics.extend(
            issue for issue in noise_hard_issues if issue not in primary_issues
        )
    severity = (
        "hard_reject"
        if hard_issues
        else "soft_warning" if warnings else "normal"
    )
    repair_report = getattr(pipeline, "_stage10_quality_repair_report", None)
    if not isinstance(repair_report, dict):
        repair_report = {
            "attempted": False,
            "status": "not_requested",
        }
    return {
        "schema": "starun.final-quality.v2",
        "stage": "stage10_final_quality",
        "file": f"{stem}.fit",
        "policy": pipeline._active_policy_name() if hasattr(pipeline, "_active_policy_name") else "",
        "target_type": active_target_type,
        "severity": severity,
        "status": "needs_conservative_rerun" if hard_issues else "ok",
        "final_quality": "poor" if hard_issues else "ok",
        "needs_conservative_rerun": bool(hard_issues),
        "strict_gate": strict_gate,
        "stage4_bright_core_color_integrity": stage4_core_color_integrity,
        "metrics": normalized_metrics,
        "warnings": warnings,
        "hard_issues": hard_issues,
        "primary_issues": list(dict.fromkeys(primary_issues)),
        "secondary_diagnostics": list(dict.fromkeys(secondary_diagnostics)),
        "repair": dict(repair_report),
        "issues": hard_issues,
        "advisories": warnings,
    }

def stage8_input_enhancement_guard(pipeline) -> Dict[str, Any]:
    hard_reasons: List[str] = []
    subject_reasons: List[str] = []
    handoff = getattr(pipeline, "_stage8_handoff", {}) or {}
    handoff = handoff if isinstance(handoff, dict) else {}
    requested_policy = str(
        handoff.get("processing_policy")
        or handoff.get("requested_policy")
        or ""
    ).strip().lower()
    upstream_requested_policy = requested_policy or "full"
    user_processing_mode = str(
        getattr(pipeline.cfg, "stage8_processing_mode", "auto") or "auto"
    ).strip().lower()
    if user_processing_mode not in {"auto", "limited", "background_only", "preserve"}:
        user_processing_mode = "auto"
    user_policy_cap = {
        "auto": "full",
        "limited": "limited",
        "background_only": "background_only",
        "preserve": "skip",
    }[user_processing_mode]
    policy_rank = {"skip": 0, "background_only": 1, "limited": 2, "full": 3}
    if requested_policy in policy_rank and policy_rank[user_policy_cap] < policy_rank[requested_policy]:
        requested_policy = user_policy_cap
    handoff_reason_text = str(handoff.get("reason_text") or "").strip()
    handoff_reason_details = list(handoff.get("reasons") or [])
    guard_reason_code = str(handoff.get("reason_code") or "")
    guard_reason_text = handoff_reason_text
    advisories: List[str] = []
    if user_processing_mode != "auto" and requested_policy == user_policy_cap:
        user_reason = (
            "user_preserve"
            if user_processing_mode == "preserve"
            else f"user_stage8_cap={user_processing_mode}"
        )
        guard_reason_code = guard_reason_code or user_reason
        guard_reason_text = guard_reason_text or user_reason
        if user_processing_mode != "preserve":
            advisories.append(user_reason)
    if requested_policy == "skip":
        hard_reasons.append(
            handoff_reason_text
            or guard_reason_text
            or str(handoff.get("reason_code") or "stage8_handoff_requested_skip")
        )
    elif requested_policy == "background_only":
        if handoff_reason_text:
            advisories.append(handoff_reason_text)
    elif requested_policy == "limited":
        if handoff_reason_text:
            advisories.append(handoff_reason_text)
    elif not requested_policy and bool(
        getattr(pipeline, "_stage8_conservative_mode", False)
    ):
        # Old checkpoints only carried a boolean. Keep them fail-closed without
        # inventing a repair reason that is not present in the evidence.
        requested_policy = "skip"
        hard_reasons.append("stage8_conservative_mode_active")
    elif not requested_policy:
        requested_policy = "full"
    elif requested_policy not in {"full", "limited", "background_only", "skip"}:
        hard_reasons.append(f"unsupported_stage8_policy={requested_policy}")
        requested_policy = "skip"
    if policy_rank[user_policy_cap] < policy_rank[requested_policy]:
        requested_policy = user_policy_cap
        user_reason = (
            "user_preserve"
            if user_processing_mode == "preserve"
            else f"user_stage8_cap={user_processing_mode}"
        )
        guard_reason_code = guard_reason_code or user_reason
        guard_reason_text = guard_reason_text or user_reason
        if user_processing_mode == "preserve":
            hard_reasons.append(user_reason)
        else:
            advisories.append(user_reason)
    if requested_policy == "limited":
        if not bool(
            getattr(pipeline.cfg, "stage8_masked_enhancement_enabled", False)
        ):
            hard_reasons.append("stage8_limited_masked_enhancement_disabled")
        starmask_file = getattr(pipeline, "starmask_file", None)
        if not (
            starmask_file is not None
            and callable(getattr(starmask_file, "exists", None))
            and starmask_file.exists()
        ):
            hard_reasons.append("stage8_limited_starmask_unavailable")
    quality = getattr(pipeline, "_stage7_selected_quality", None)
    derived = quality.get("derived") if isinstance(quality, dict) else {}
    if bool(getattr(pipeline, "_stage7_starless_skipped", False)):
        hard_reasons.append("stage6_star_separation_unavailable")
    status = str(quality.get("status", "")) if isinstance(quality, dict) else ""
    if status and status not in {"ok"}:
        subject_reasons.append(f"stage7_quality_status={status}")
    if isinstance(quality, dict):
        advisories.extend(
            str(item).strip()
            for item in (quality.get("advisories") or [])
            if str(item).strip()
        )

    residual = 0.0
    halo = pipeline._stage7_halo_residue_score()
    compact_halo = 0.0
    noise_gain = 0.0
    if isinstance(derived, dict):
        try:
            residual = float(derived.get("residual_star_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            residual = 0.0
        try:
            halo = max(halo, float(derived.get("halo_residue_score", 0.0) or 0.0))
        except (TypeError, ValueError):
            pass
        try:
            compact_halo = max(
                compact_halo,
                float(derived.get("compact_halo_residue_score", 0.0) or 0.0),
            )
            halo = max(halo, compact_halo)
        except (TypeError, ValueError):
            pass
        try:
            noise_gain = float(derived.get("starless_noise_gain", 0.0) or 0.0)
        except (TypeError, ValueError):
            noise_gain = 0.0
    residual_gate = stage7_quality.stage7_upper_quality_gate(
        pipeline.cfg,
        value=residual,
        accepted_limit=pipeline.cfg.stage7_residual_star_score_max,
    )
    if residual_gate["hard_failed"]:
        subject_reasons.append(
            f"stage7_residual_star_score {residual:.3f}>"
            f"{pipeline.cfg.stage7_residual_star_score_max:.3f}"
        )
    elif residual_gate["advisory"]:
        advisories.append(
            f"stage7_residual_star_score {residual:.3f}>"
            f"{pipeline.cfg.stage7_residual_star_score_max:.3f} advisory"
        )
    halo_threshold = pipeline._stage7_effective_halo_threshold()
    base_halo_limit = float(
        getattr(pipeline.cfg, "stage7_halo_residue_score_max", 0.35)
    )
    target_type = (
        str(pipeline._active_target_type() or "")
        if hasattr(pipeline, "_active_target_type")
        else ""
    )
    if (
        requested_policy == "full"
        and target_type == "bright_emission_reflection_nebula"
        and halo > base_halo_limit
        and halo <= halo_threshold
    ):
        requested_policy = "limited"
        guard_reason_code = "bright_nebula_halo_advisory"
        guard_reason_text = (
            "bright_nebula_halo_advisory: "
            f"{halo:.3f} > {base_halo_limit:.3f}, "
            f"accepted_limit={halo_threshold:.3f}"
        )
        advisories.append(guard_reason_text)
        handoff_reason_details.append(
            {
                "code": guard_reason_code,
                "source_stage": 8,
                "value": halo,
                "base_limit": base_halo_limit,
                "accepted_limit": halo_threshold,
            }
        )
        if not bool(
            getattr(pipeline.cfg, "stage8_masked_enhancement_enabled", False)
        ):
            hard_reasons.append("stage8_limited_masked_enhancement_disabled")
        starmask_file = getattr(pipeline, "starmask_file", None)
        if not (
            starmask_file is not None
            and callable(getattr(starmask_file, "exists", None))
            and starmask_file.exists()
        ):
            hard_reasons.append("stage8_limited_starmask_unavailable")
    halo_gate = stage7_quality.stage7_upper_quality_gate(
        pipeline.cfg,
        value=halo,
        accepted_limit=halo_threshold,
    )
    if halo_gate["hard_failed"]:
        subject_reasons.append(
            f"stage7_halo_residue_score {halo:.3f}>{halo_threshold:.3f}"
        )
    elif halo_gate["advisory"]:
        advisories.append(
            f"stage7_halo_residue_score {halo:.3f}>{halo_threshold:.3f} advisory"
        )
    noise_gate = stage7_quality.stage7_upper_quality_gate(
        pipeline.cfg,
        value=noise_gain,
        accepted_limit=pipeline.cfg.stage7_starless_noise_gain_max,
    )
    if noise_gate["hard_failed"]:
        hard_reasons.append(
            f"stage7_starless_noise_gain {noise_gain:.3f}>"
            f"{pipeline.cfg.stage7_starless_noise_gain_max:.3f}"
        )
    elif noise_gate["advisory"]:
        advisories.append(
            f"stage7_starless_noise_gain {noise_gain:.3f}>"
            f"{pipeline.cfg.stage7_starless_noise_gain_max:.3f} advisory"
        )

    coverage: Dict[str, float] = {}
    mask_signal_coverage = None
    background_pixel_count = 0
    readiness_report: Dict[str, Any] = {
        "schema": "starun.stage8-starless-readiness.v1",
        "status": "unavailable",
        "mode": "report_only",
        "used_for_gate": False,
    }
    try:
        image_data = pipeline.siril.get_image_pixeldata(preview=False)
        if image_data is not None:
            masks = pipeline._stage8_generate_starless_masks(np.asarray(image_data))
            readiness_builder = getattr(
                pipeline,
                "_stage8_starless_readiness_report",
                None,
            )
            if callable(readiness_builder):
                readiness_report = readiness_builder(np.asarray(image_data), masks)
            else:
                readiness_report = stage8_starless_readiness_report(
                    pipeline,
                    np.asarray(image_data),
                    masks,
                )
            raw_coverage = masks.get("coverage", {})
            if isinstance(raw_coverage, dict):
                coverage = {str(k): float(v) for k, v in raw_coverage.items()}
            background_mask = np.asarray(
                masks.get("background_mask", np.zeros(np.asarray(image_data).shape[-2:])),
                dtype=np.float32,
            )
            background_pixel_count = int(np.count_nonzero(background_mask > 0.60))
            mask_signal_coverage = float(coverage.get("nebula", 0.0)) + float(
                coverage.get("faint_nebula", 0.0)
            )
            if mask_signal_coverage < pipeline.cfg.stage8_mask_signal_coverage_min:
                subject_reasons.append(
                    "stage8_mask_signal_coverage "
                    f"{mask_signal_coverage:.4f}<{pipeline.cfg.stage8_mask_signal_coverage_min:.4f}"
                )
    except (CommandError, SirilError, OSError, RuntimeError, TypeError, ValueError) as e:
        hard_reasons.append(
            f"stage8_mask_guard_unavailable={pipeline._short_text(e, 120)}"
        )

    background_available = background_pixel_count >= 64
    if hard_reasons:
        processing_policy = "skip"
    elif requested_policy == "background_only":
        processing_policy = "background_only"
    elif subject_reasons:
        processing_policy = "background_only" if background_available else "skip"
        if processing_policy == "skip":
            hard_reasons.append("stage8_background_support_unavailable")
    elif requested_policy == "limited":
        processing_policy = "limited"
    else:
        processing_policy = "full"

    reasons = [*hard_reasons, *subject_reasons]
    if processing_policy == "background_only":
        guard_reason_code = guard_reason_code or "stage8_subject_risk_background_only"
        guard_reason_text = guard_reason_text or ", ".join(subject_reasons[:2]) or (
            "stage8_background_only_requested"
        )
    conservative_skip = bool(
        processing_policy == "skip"
        and requested_policy in {"limited", "skip"}
    )
    skip_status = (
        "conservative_skipped"
        if conservative_skip
        else "skipped"
        if processing_policy == "skip"
        else "background_only_passthrough"
        if processing_policy == "background_only"
        else "ok"
    )
    return {
        "skip_enhancement": processing_policy == "skip",
        "background_only": processing_policy == "background_only",
        "processing_policy": processing_policy,
        "requested_policy": requested_policy,
        "upstream_requested_policy": upstream_requested_policy,
        "user_processing_mode": user_processing_mode,
        "conservative_mode": processing_policy in {
            "limited",
            "background_only",
            "skip",
        },
        "status": skip_status,
        "final_quality": skip_status,
        "reasons": reasons,
        "hard_reasons": hard_reasons,
        "subject_reasons": subject_reasons,
        "advisories": list(dict.fromkeys(advisories)),
        "reason_details": handoff_reason_details,
        "reason_code": guard_reason_code,
        "reason_text": guard_reason_text,
        "mask_coverage": coverage,
        "mask_signal_coverage": mask_signal_coverage,
        "background_pixel_count": background_pixel_count,
        "background_available": background_available,
        "starless_readiness": readiness_report,
        "derived": {
            "residual_star_score": residual,
            "halo_residue_score": halo,
            "compact_halo_residue_score": compact_halo,
            "starless_noise_gain": noise_gain,
        },
    }

def apply_stage8_masked_pixel_enhancement(
    pipeline,
    image_data: np.ndarray,
    plan: Dict[str, Any],
    *,
    label: str,
    plugin_candidate: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any], List[str]]:
    masks = pipeline._stage8_generate_starless_masks(image_data)
    base = masks["rgb"]
    gray = masks["gray"]
    core = masks["core_mask"]
    nebula = masks["nebula_mask"]
    faint = masks["faint_nebula_mask"]
    background = masks["background_mask"]
    coverage = dict(masks.get("coverage", {}))
    handoff = getattr(pipeline, "_stage8_handoff", {}) or {}
    limited_mode = bool(
        isinstance(handoff, dict)
        and str(handoff.get("processing_policy") or "").strip().lower()
        == "limited"
    )
    limited_core_exclusion = np.asarray(
        masks.get("limited_core_exclusion_mask", core),
        dtype=np.float32,
    )
    limited_core_hard = np.asarray(
        masks.get("limited_core_exclusion_hard_mask", core > 0.50),
        dtype=np.float32,
    )
    effective_core = limited_core_exclusion if limited_mode else core
    core_guard = np.clip(1.0 - effective_core, 0.0, 1.0)

    # The restricted candidate is deliberately weak-signal-only.  Removing the
    # soft-mask tail below 0.05 keeps the recipe from touching most of the frame,
    # while the additional nebula suppression avoids lifting the bright subject.
    weak_signal = np.clip(faint * (1.0 - nebula) * core_guard, 0.0, 1.0)
    weak_signal = np.clip((weak_signal - 0.05) / 0.95, 0.0, 1.0)
    if limited_mode:
        nebula_effect = weak_signal
        faint_effect = weak_signal
    else:
        nebula_effect = np.clip(nebula * core_guard, 0.0, 1.0)
        faint_effect = np.clip(faint * core_guard, 0.0, 1.0)
    coverage["limited_weak_signal"] = float(np.mean(weak_signal > 0.05))
    target_type = pipeline._active_target_type() if hasattr(pipeline, "_active_target_type") else ""
    high_halo_risk = pipeline._stage7_halo_residue_score() > pipeline._stage7_effective_halo_threshold()
    object_mask_only = target_type == "bright_emission_reflection_nebula" or high_halo_risk
    mask_signal_coverage = float(coverage.get("nebula", 0.0)) + float(
        coverage.get("faint_nebula", 0.0)
    )
    mask_quality_scale = 1.0
    messages: List[str] = []

    physical_color_anchor = physical_broadband_anchor_accepted(
        getattr(pipeline, "channel_profile", {}) or {},
        getattr(pipeline, "color_calibration_report", {}) or {},
    )
    if physical_color_anchor:
        messages.append(
            f"{label} Stage4 physical color anchor frozen; global channel rebalance disabled"
        )

    saturation = _clamp_float(plan.get("saturation", pipeline.cfg.nebula_saturation), 0.0, 0.65)
    unsharp_amount = min(
        _clamp_float(plan.get("unsharp_amount", 0.35), 0.0, 0.60),
        float(pipeline.cfg.stage8_masked_unsharp_amount_max),
    )
    if not bool(getattr(pipeline.cfg, "stage8_nebula_saturation_enabled", True)):
        saturation = 0.0
    if not bool(getattr(pipeline.cfg, "stage8_masked_unsharp_enabled", True)):
        unsharp_amount = 0.0
    if masks["bg_std"] > pipeline.cfg.stage7_bg_std_high:
        unsharp_amount *= 0.40
        saturation *= 0.72
        messages.append(f"{label} stage8 high-bg-noise guard reduced saturation/detail")

    if object_mask_only:
        if high_halo_risk:
            saturation = min(saturation, 0.05)
        saturation *= 0.82
        unsharp_amount *= 0.70
        mask_quality_scale = 0.85
        messages.append(f"{label} stage8 bright-nebula object-mask-only mode")
        if mask_signal_coverage < 0.015:
            mask_quality_scale = 0.45
            saturation *= 0.55
            unsharp_amount *= 0.40
            messages.append(
                f"{label} stage8 low mask coverage reduced enhancement "
                f"(coverage={mask_signal_coverage:.4f})"
            )

    if limited_mode:
        saturation = min(
            saturation,
            float(getattr(pipeline.cfg, "stage8_limited_saturation_max", 0.05)),
        )
        unsharp_amount = 0.0
        messages.append(
            f"{label} stage8 limited weak-signal-only mode "
            "(core curves/saturation/faint boost disabled, "
            f"core_expand={int(masks.get('limited_core_exclusion_expand', 8))})"
        )

    core_clip = float(np.mean(gray >= 0.985))
    if core_clip > pipeline.cfg.stage8_highlight_clip_ratio_max:
        unsharp_amount = 0.0
        saturation *= 0.65
        messages.append(f"{label} stage8 highlight guard disabled masked unsharp")

    denoise_strength = (
        0.0
        if limited_mode
        else float(pipeline.cfg.stage8_background_denoise_strength)
    )
    if not bool(getattr(pipeline.cfg, "stage8_background_denoise_enabled", True)):
        denoise_strength = 0.0
    faint_boost = min(float(pipeline.cfg.stage8_faint_nebula_boost_max), 0.20 * saturation) * mask_quality_scale
    contrast_strength = min(float(pipeline.cfg.stage8_nebula_contrast_max), 0.28 * saturation) * mask_quality_scale
    if not bool(getattr(pipeline.cfg, "stage8_faint_nebula_boost_enabled", True)):
        faint_boost = 0.0
    if not bool(getattr(pipeline.cfg, "stage8_nebula_contrast_enabled", True)):
        contrast_strength = 0.0
    signal_weight = np.clip(nebula_effect + 0.60 * faint_effect, 0.0, 1.0)
    color_sample_weight = np.clip(
        signal_weight * core_guard * (1.0 - 0.85 * background),
        0.0,
        1.0,
    )
    color_weight_total = float(np.sum(color_sample_weight))
    blue_pre_gain = 1.0
    if color_weight_total > 1e-6 and not physical_color_anchor:
        red_dom = float(
            np.sum((base[0] + 1e-6) / (base[1] + 1e-6) * color_sample_weight)
            / color_weight_total
        )
        blue_dom = float(
            np.sum((base[2] + 1e-6) / (base[1] + 1e-6) * color_sample_weight)
            / color_weight_total
        )
        blue_excess = max(0.0, blue_dom - max(1.04, red_dom + 0.08))
        blue_target = max(0.0, float(pipeline.cfg.stage8_blue_excess_max) * 0.55)
        if blue_excess > blue_target:
            blue_pre_gain = _clamp_float(
                1.0 - min(
                    0.18,
                    (blue_excess - blue_target)
                    * float(pipeline.cfg.stage8_blue_precontrol_strength),
                ),
                0.82,
                1.0,
            )
            saturation *= _clamp_float(1.0 - (1.0 - blue_pre_gain) * 1.4, 0.70, 1.0)
            messages.append(
                f"{label} stage8 blue pre-control "
                f"(blue_excess={blue_excess:.3f}, b_gain={blue_pre_gain:.3f})"
            )

    result = base.copy()
    blurred = pipeline._box_blur_rgb(result)
    denoise_weight = np.clip(
        denoise_strength * (1.35 * background + 0.30 * faint + 0.10),
        0.0,
        0.42,
    ) * core_guard
    result = result * (1.0 - denoise_weight[None, :, :]) + blurred * denoise_weight[None, :, :]
    denoised_base = result.copy()

    if blue_pre_gain < 0.999:
        blue_weight = np.clip(
            (nebula_effect + 0.45 * faint_effect)
            * core_guard
            * (1.0 - background),
            0.0,
            1.0,
        )
        result[2] = result[2] + result[2] * (blue_pre_gain - 1.0) * blue_weight

    if faint_boost > 1e-6:
        background_guard = 1.0 - (0.98 if object_mask_only else 0.80) * background
        lift = (
            faint_boost
            * faint_effect
            * background_guard
            * core_guard
            * np.clip(1.0 - gray, 0.0, 1.0)
        )
        result = result + lift[None, :, :]

    if contrast_strength > 1e-6:
        background_guard = 1.0 - (1.0 if object_mask_only else 0.90) * background
        signal_weight = np.clip(
            (nebula_effect + 0.30 * faint_effect) * background_guard,
            0.0,
            1.0,
        )
        total = float(np.sum(signal_weight))
        center = (
            float(np.sum(gray * signal_weight) / total)
            if total > 1e-6
            else float(np.median(gray))
        )
        contrast_weight = contrast_strength * nebula_effect * core_guard
        result = result + (result - center) * contrast_weight[None, :, :]

    if unsharp_amount > 1e-6:
        blurred = pipeline._box_blur_rgb(result)
        background_guard = 1.0 - (1.0 if object_mask_only else 0.98) * background
        detail_weight = np.clip(
            unsharp_amount * nebula_effect * core_guard * background_guard,
            0.0,
            float(pipeline.cfg.stage8_masked_unsharp_amount_max),
        )
        result = result + (result - blurred) * detail_weight[None, :, :]

    if plugin_candidate is not None:
        plugin_rgb = _to_rgb_float_fullres(plugin_candidate)
        if blue_pre_gain < 0.999:
            blue_weight = np.clip(
                (nebula_effect + 0.45 * faint_effect)
                * core_guard
                * (1.0 - background),
                0.0,
                1.0,
            )
            plugin_rgb[2] = (
                plugin_rgb[2]
                + plugin_rgb[2] * (blue_pre_gain - 1.0) * blue_weight
            )
        background_guard = 1.0 - (1.0 if object_mask_only else 0.98) * background
        plugin_weight = np.clip(
            (0.34 * nebula_effect + 0.16 * faint_effect)
            * core_guard
            * background_guard,
            0.0,
            0.24 if object_mask_only else 0.38,
        )
        result = result + (plugin_rgb - result) * plugin_weight[None, :, :]
        messages.append(f"{label} SASP output blended through Starless masks")

    core_protection = float(pipeline.cfg.stage8_core_protection_strength)
    if not limited_mode:
        background_restore_strength = 1.0 if object_mask_only else 0.985
        background_restore = np.clip(background_restore_strength * background, 0.0, 1.0)
        result = (
            result * (1.0 - background_restore[None, :, :])
            + denoised_base * background_restore[None, :, :]
        )
        core_restore = np.clip(core_protection * core, 0.0, 1.0)
        result = (
            result * (1.0 - core_restore[None, :, :])
            + base * core_restore[None, :, :]
        )

    local_adjustment_report: Dict[str, Any] = {
        "schema": LOCAL_ADJUSTMENT_SCHEMA,
        "status": "disabled",
        "accepted": False,
    }
    if bool(
        getattr(
            pipeline.cfg,
            "stage8_local_adjustment_engine_enabled",
            True,
        )
    ):
        local_operations: List[Dict[str, Any]] = []
        curve_mask_name = (
            "limited_weak_signal" if limited_mode else "faint_nebula"
        )
        subject_mask_name = (
            "limited_weak_signal" if limited_mode else "nebula"
        )
        channel_semantics = str(
            getattr(pipeline, "_channel_semantics", "unknown") or "unknown"
        )
        selective_profile = ""
        selective_bands: List[Dict[str, Any]] = []
        target_color_route = "generic_color_preserve"
        target_route_allowed, target_route_reason = (
            stage8_target_color_route_allowed(pipeline)
        )
        if (
            saturation > 1e-6
            and not limited_mode
            and channel_semantics == BROADBAND_RGB_OSC
            and target_route_allowed
        ):
            selective_profile, selective_bands = (
                stage8_broadband_hue_saturation_bands(
                    target_type,
                    saturation,
                )
            )
            if selective_profile:
                target_color_route = selective_profile
        elif channel_semantics == BROADBAND_RGB_OSC and not target_route_allowed:
            messages.append(
                f"{label} target color route failed closed to generic preserve "
                f"({target_route_reason})"
            )
        if faint_boost > 1e-6:
            local_operations.append(
                {
                    "type": "curve",
                    "mask": curve_mask_name,
                    "points": (
                        (0.0, 0.0),
                        (0.18, 0.19),
                        (0.50, 0.51),
                        (1.0, 1.0),
                    ),
                    "opacity": min(
                        0.60,
                        float(
                            getattr(
                                pipeline.cfg,
                                "stage8_local_curve_opacity",
                                0.30,
                            )
                        )
                        * mask_quality_scale,
                    ),
                }
            )
        if saturation > 1e-6:
            if selective_bands:
                local_operations.append(
                    {
                        "type": "hue_selective_saturation",
                        "mask": subject_mask_name,
                        "profile": selective_profile,
                        "bands": selective_bands,
                        "opacity": 0.50 * mask_quality_scale,
                    }
                )
            else:
                local_operations.append(
                    {
                        "type": "saturation",
                        "mask": subject_mask_name,
                        "amount": min(0.05, saturation * 0.12),
                        "opacity": 0.50 * mask_quality_scale,
                    }
                )
        if contrast_strength > 1e-6:
            local_operations.append(
                {
                    "type": "local_contrast",
                    "mask": subject_mask_name,
                    "amount": min(0.03, contrast_strength * 0.25),
                    "radius": 2,
                    "opacity": 0.50 * mask_quality_scale,
                }
            )
        local_candidate, local_adjustment_report = (
            apply_local_adjustment_recipe(
                result,
                {
                    "schema": LOCAL_ADJUSTMENT_SCHEMA,
                    "id": "stage8_nebula_local_v2",
                    "operations": local_operations,
                },
                masks={
                    "background": background,
                    "core": effective_core,
                    "nebula": nebula_effect,
                    "faint_nebula": faint_effect,
                    "limited_weak_signal": weak_signal,
                },
            )
        )
        if bool(local_adjustment_report.get("accepted", False)):
            result = np.asarray(local_candidate, dtype=np.float32)
            if selective_profile:
                messages.append(
                    f"{label} broadband hue-selective saturation accepted "
                    f"(target={target_type}, profile={selective_profile}, "
                    f"bands={len(selective_bands)})"
                )
            messages.append(
                f"{label} local curves/masks recipe accepted "
                f"(operations={len(local_operations)}, "
                "changed="
                f"{float((local_adjustment_report.get('metrics') or {}).get('changed_pixel_ratio', 0.0)):.3f})"
            )
        else:
            messages.append(
                f"{label} local curves/masks recipe rejected; "
                "kept pre-recipe candidate"
            )
        local_adjustment_report["color_route"] = target_color_route
        local_adjustment_report["target_route_allowed"] = bool(
            target_route_allowed
        )
        local_adjustment_report["target_route_reason"] = target_route_reason

    if limited_mode:
        # Enforce the scope after all operations, including the local recipe.
        # This is an invariant, not merely a quality-gate expectation: pixels
        # outside the weak-signal mask and the expanded hard core stay bitwise at
        # the Stage 8 input value.
        result[:, weak_signal <= 0.0] = base[:, weak_signal <= 0.0]
        soft_core_edge = limited_core_exclusion * (1.0 - limited_core_hard)
        result = result + (base - result) * soft_core_edge[None, :, :]
        result[:, limited_core_hard > 0.50] = base[:, limited_core_hard > 0.50]

    result = np.clip(result, 0.0, 1.0)
    scope_delta = np.max(np.abs(result - base), axis=0)
    processing_scope = {
        "mode": (
            "limited_weak_signal_only" if limited_mode else "masked_full"
        ),
        "core_exclusion_expand": int(
            masks.get("limited_core_exclusion_expand", 8)
        ),
        "core_threshold": float(masks.get("core_threshold", 0.0)),
        "core_channel_peak_threshold": float(
            masks.get("core_channel_peak_threshold", 0.985)
        ),
    }
    if limited_mode:
        hard_core_pixels = limited_core_hard > 0.50
        outside_weak_pixels = weak_signal <= 0.0
        processing_scope.update(
            {
                "core_operation_weight_max": (
                    float(np.max(weak_signal[hard_core_pixels]))
                    if np.any(hard_core_pixels)
                    else 0.0
                ),
                "core_max_abs_change": (
                    float(np.max(scope_delta[hard_core_pixels]))
                    if np.any(hard_core_pixels)
                    else 0.0
                ),
                "outside_weak_signal_max_abs_change": (
                    float(np.max(scope_delta[outside_weak_pixels]))
                    if np.any(outside_weak_pixels)
                    else 0.0
                ),
            }
        )

    restored = pipeline._stage8_restore_rgb_like(image_data, result)
    saturation_operations = [
        operation
        for operation in local_adjustment_report.get("operations", [])
        if isinstance(operation, dict)
        and str(operation.get("type"))
        in {"saturation", "hue_selective_saturation"}
    ]
    saturation_applied = bool(
        local_adjustment_report.get("accepted", False)
        and saturation_operations
    )
    saturation_amounts: List[float] = []
    for operation in saturation_operations:
        opacity = _clamp_float(operation.get("opacity", 1.0), 0.0, 1.0)
        if str(operation.get("type")) == "hue_selective_saturation":
            bands = operation.get("bands") or []
            saturation_amounts.extend(
                abs(float(band.get("amount", 0.0) or 0.0)) * opacity
                for band in bands
                if isinstance(band, dict)
            )
        else:
            saturation_amounts.append(
                abs(float(operation.get("amount", 0.0) or 0.0)) * opacity
            )
    applied_saturation_amount = max(saturation_amounts, default=0.0)
    diagnostics = {
        "mask_coverage": coverage,
        "masked_metrics": pipeline._stage8_masked_metrics(restored, masks),
        "protection_actions": messages,
        "local_adjustment_engine": local_adjustment_report,
        "saturation_execution": {
            "requested": saturation,
            "applied": saturation_applied,
            "applied_amount": applied_saturation_amount,
            "passes": 1 if saturation_applied else 0,
            "position": "after_structure_and_plugin_blend",
            "method": (
                "local_adjustment_recipe"
                if local_adjustment_report.get("accepted", False)
                else "none"
            ),
        },
        "processing_scope": processing_scope,
    }
    messages.append(
        f"{label} masked Starless enhancement "
        f"(sat={saturation:.3f}, faint={faint_boost:.3f}, "
        f"contrast={contrast_strength:.3f}, unsharp={unsharp_amount:.3f})"
    )
    return restored, diagnostics, messages

def apply_stage8_builtin_enhancement(
    pipeline,
    plan: Dict[str, Any],
    *,
    label: str,
) -> List[str]:
    messages: List[str] = []
    saturation = _clamp_float(plan.get("saturation", pipeline.cfg.nebula_saturation), 0.0, 0.65)
    bg_factor = _clamp_int(plan.get("bg_factor", pipeline.cfg.nebula_bg_factor), 0, 3)
    unsharp_radius = _clamp_float(plan.get("unsharp_radius", 0.8), 0.0, 1.2)
    unsharp_amount = _clamp_float(plan.get("unsharp_amount", 0.35), 0.0, 0.60)
    if not bool(getattr(pipeline.cfg, "stage8_nebula_saturation_enabled", True)):
        saturation = 0.0
    if not bool(getattr(pipeline.cfg, "stage8_masked_unsharp_enabled", True)):
        unsharp_amount = 0.0
    masked_enhancement_enabled = bool(
        getattr(pipeline.cfg, "stage8_masked_enhancement_enabled", False)
    )
    if (
        not masked_enhancement_enabled
        and physical_broadband_anchor_accepted(
            getattr(pipeline, "channel_profile", {}) or {},
            getattr(pipeline, "color_calibration_report", {}) or {},
        )
    ):
        saturation = 0.0
        messages.append(
            f"{label} global satu skipped: Stage4 physical color anchor "
            "requires masked color recovery"
        )

    if masked_enhancement_enabled:
        image_data = pipeline.siril.get_image_pixeldata(preview=False)
        if image_data is None:
            raise RuntimeError("image buffer is empty")
        enhanced, diagnostics, masked_messages = pipeline._apply_stage8_masked_pixel_enhancement(
            np.asarray(image_data),
            {
                **plan,
                "saturation": saturation,
                "unsharp_amount": min(unsharp_amount, pipeline.cfg.stage8_masked_unsharp_amount_max),
            },
            label=label,
        )
        pipeline._set_current_image_pixeldata(enhanced, label=f"{label} stage8 masked enhancement")
        pipeline._last_stage8_masked_diagnostics = diagnostics
        pipeline._stage8_saturation_execution = dict(
            diagnostics.get("saturation_execution") or {}
        )
        return masked_messages

    if unsharp_radius > 1e-6 and unsharp_amount > 1e-6:
        pipeline.cmd_with_check(
            "unsharp",
            f"{unsharp_radius:.4f}",
            f"{unsharp_amount:.4f}",
        )
        messages.append(
            f"{label} Starless unsharp ({unsharp_radius:.3f}, {unsharp_amount:.3f})"
        )
    else:
        messages.append(f"{label} Starless unsharp skipped")

    saturation_applied = False
    if saturation > 1e-6:
        pipeline.cmd_with_check("satu", f"{saturation:.6f}", str(bg_factor))
        saturation_applied = True
        messages.append(
            f"{label} Starless satu (sat={saturation:.4f}, bg={bg_factor})"
        )
    else:
        messages.append(f"{label} Starless satu skipped (sat=0)")
    pipeline._stage8_saturation_execution = {
        "requested": saturation,
        "applied": saturation_applied,
        "applied_amount": saturation if saturation_applied else 0.0,
        "passes": 1 if saturation_applied else 0,
        "position": "after_structure_unmasked_path",
        "method": "siril_satu" if saturation_applied else "none",
    }
    return messages

def apply_stage8_color_correction_from_quality(
    pipeline,
    quality_record: Dict[str, Any],
) -> Optional[str]:
    candidate_metrics = quality_record.get("candidate_metrics")
    if not isinstance(candidate_metrics, dict):
        return None
    blue_excess = float(candidate_metrics.get("blue_excess", 0.0))
    target_blue_excess = pipeline._stage8_target_blue_excess(quality_record)
    correction_floor = target_blue_excess + 0.012
    if blue_excess <= correction_floor:
        return None
    b_gain = _clamp_float(
        1.0 - min(0.14, max(0.0, blue_excess - target_blue_excess) * 0.45),
        0.84,
        0.98,
    )
    pipeline.cmd_with_check("load", "stage8_enhanced")
    pipeline.cmd_with_check(
        "ccm",
        "1.000000",
        "0",
        "0",
        "0",
        "1.000000",
        "0",
        "0",
        "0",
        f"{b_gain:.6f}",
    )
    pipeline._save_stage_output("stage8_enhanced")
    return (
        "Stage8 color correction applied "
        f"(blue_excess={blue_excess:.3f}, target={target_blue_excess:.3f}, "
        f"b_gain={b_gain:.3f})"
    )

def stage8_target_blue_excess(pipeline, quality_record: Optional[Dict[str, Any]]) -> float:
    return float(pipeline.cfg.stage8_blue_excess_max)


def stage8_limited_halo_texture_report(
    pipeline,
    baseline_data: Optional[np.ndarray],
    candidate_data: Optional[np.ndarray],
    starmask_data: Optional[np.ndarray],
) -> Dict[str, Any]:
    """Compare local texture in a starmask-derived star-halo annulus."""
    growth_limit = float(
        getattr(pipeline.cfg, "stage8_limited_halo_texture_growth_max", 1.05)
    )
    delta_limit = float(
        getattr(pipeline.cfg, "stage8_limited_halo_texture_delta_max", 0.00075)
    )
    report: Dict[str, Any] = {
        "available": False,
        "accepted": False,
        "growth_limit": growth_limit,
        "absolute_delta_limit": delta_limit,
        "reason": "required image data unavailable",
    }
    if baseline_data is None or candidate_data is None or starmask_data is None:
        return report
    try:
        baseline_rgb = _to_rgb_float_image(baseline_data, max_side=1024)
        candidate_rgb = _to_rgb_float_image(candidate_data, max_side=1024)
        starmask_rgb = _to_rgb_float_image(starmask_data, max_side=1024)
        if not (
            baseline_rgb.shape == candidate_rgb.shape == starmask_rgb.shape
        ):
            report["reason"] = "baseline/candidate/starmask shape mismatch"
            return report

        baseline_gray = (
            0.2126 * baseline_rgb[0]
            + 0.7152 * baseline_rgb[1]
            + 0.0722 * baseline_rgb[2]
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
        mask_floor = float(np.quantile(starmask_gray, 0.55))
        mask_signal = np.clip(starmask_gray - mask_floor, 0.0, None)
        positive = mask_signal[mask_signal > 0.0]
        if positive.size < 16:
            report["reason"] = "starmask compact support unavailable"
            return report
        mask_scale = max(float(np.quantile(positive, 0.995)), 1e-7)
        normalized = np.clip(mask_signal / mask_scale, 0.0, 1.0)
        # A cleaned starmask can still contain a low-amplitude diffuse pedestal.
        # A fixed threshold then turns almost half the frame into "star cores" and
        # makes the annulus unusable.  Select the strongest compact support first,
        # relaxing only when there are too few core pixels for a stable sample.
        support_attempts: List[Dict[str, Any]] = []
        ring_weight: Optional[np.ndarray] = None
        ring_coverage = 0.0
        weight_total = 0.0
        selected_threshold = 0.0
        selected_quantile = 0.0
        selected_core_coverage = 0.0
        normalized_positive = normalized[normalized > 0.0]
        for support_quantile in (0.995, 0.99, 0.98, 0.95, 0.90):
            core_threshold = max(
                0.12,
                float(np.quantile(normalized_positive, support_quantile)),
            )
            star_core = (normalized >= core_threshold).astype(np.float32)
            core_pixels = int(np.count_nonzero(star_core))
            core_coverage = float(np.mean(star_core > 0.0))
            attempt: Dict[str, Any] = {
                "support_quantile": support_quantile,
                "core_threshold": core_threshold,
                "core_pixels": core_pixels,
                "core_coverage": core_coverage,
            }
            if core_pixels < 8:
                attempt["reason"] = "core support too small"
                support_attempts.append(attempt)
                continue

            inner = dilate_mask(star_core, iterations=1)
            outer = dilate_mask(star_core, iterations=5)
            candidate_ring = np.clip(
                feather_mask(outer, radius=1) - feather_mask(inner, radius=1),
                0.0,
                1.0,
            )
            candidate_coverage = float(np.mean(candidate_ring > 0.05))
            candidate_weight = float(np.sum(candidate_ring))
            attempt.update(
                ring_coverage=candidate_coverage,
                ring_weight_total=candidate_weight,
            )
            support_attempts.append(attempt)
            if candidate_weight < 16.0 or candidate_coverage > 0.45:
                continue
            ring_weight = candidate_ring
            ring_coverage = candidate_coverage
            weight_total = candidate_weight
            selected_threshold = core_threshold
            selected_quantile = support_quantile
            selected_core_coverage = core_coverage
            break

        if ring_weight is None:
            report.update(
                reason="starmask compact halo annulus unavailable",
                support_attempts=support_attempts,
            )
            return report

        def local_texture(gray: np.ndarray) -> np.ndarray:
            smooth = gray
            for _ in range(3):
                smooth = _box_blur_gray(smooth)
            return np.abs(gray - smooth)

        baseline_texture = local_texture(baseline_gray)
        candidate_texture = local_texture(candidate_gray)
        baseline_level = float(
            np.sum(baseline_texture * ring_weight) / weight_total
        )
        candidate_level = float(
            np.sum(candidate_texture * ring_weight) / weight_total
        )
        growth = candidate_level / max(baseline_level, 1e-7)
        absolute_delta = max(0.0, candidate_level - baseline_level)
        nominal_exceeded = bool(
            growth > growth_limit and absolute_delta > delta_limit
        )
        growth_gate = stage7_quality.stage7_9_upper_quality_gate(
            pipeline.cfg,
            value=growth,
            accepted_limit=growth_limit,
        )
        delta_gate = stage7_quality.stage7_9_upper_quality_gate(
            pipeline.cfg,
            value=absolute_delta,
            accepted_limit=delta_limit,
        )
        hard_failed = bool(
            nominal_exceeded
            and (growth_gate["hard_failed"] or delta_gate["hard_failed"])
        )
        advisory = bool(nominal_exceeded and not hard_failed)
        accepted = not hard_failed
        report.update(
            {
                "available": True,
                "accepted": accepted,
                "reason": (
                    "halo annulus texture growth exceeded"
                    if hard_failed
                    else "halo annulus texture growth advisory"
                    if advisory
                    else ""
                ),
                "advisory": advisory,
                "advisories": (
                    [
                        "limited_halo_texture_growth "
                        f"{growth:.3f}>{growth_limit:.3f}, delta "
                        f"{absolute_delta:.6f}>{delta_limit:.6f} advisory"
                    ]
                    if advisory
                    else []
                ),
                "quality_gates": {
                    "growth": growth_gate,
                    "absolute_delta": delta_gate,
                },
                "ring_coverage": ring_coverage,
                "core_coverage": selected_core_coverage,
                "core_threshold": selected_threshold,
                "support_quantile": selected_quantile,
                "support_attempts": support_attempts,
                "baseline_level": baseline_level,
                "candidate_level": candidate_level,
                "growth": growth,
                "absolute_delta": absolute_delta,
                "low_absolute_growth_exempted": (
                    growth > growth_limit and absolute_delta <= delta_limit
                ),
            }
        )
        return report
    except (IndexError, TypeError, ValueError, FloatingPointError) as error:
        report["reason"] = str(error)
        return report

def stage8_quality_assessment(
    pipeline,
    *,
    baseline_stem: str = "stage8_input_starless",
    candidate_stem: str = "stage8_enhanced",
) -> Dict[str, Any]:
    handoff = getattr(pipeline, "_stage8_handoff", {}) or {}
    limited_mode = bool(
        isinstance(handoff, dict)
        and str(handoff.get("processing_policy") or "").strip().lower()
        == "limited"
    )
    starmask_data = None
    if limited_mode:
        starmask_file = getattr(pipeline, "starmask_file", None)
        starmask_stem = getattr(starmask_file, "stem", "")
        if starmask_stem:
            loaded_starmask = pipeline._read_image_by_stem(starmask_stem)
            if loaded_starmask is not None:
                starmask_data = np.array(loaded_starmask, copy=True)
    baseline_data = pipeline._read_image_by_stem(baseline_stem)
    candidate_data = pipeline._read_image_by_stem(candidate_stem)
    unavailable_issues: List[str] = []
    if baseline_data is None:
        unavailable_issues.append(f"stage8_baseline_unavailable={baseline_stem}")
    if candidate_data is None:
        unavailable_issues.append(f"stage8_candidate_unavailable={candidate_stem}")
    if baseline_data is not None and candidate_data is not None:
        try:
            baseline_array = np.asarray(baseline_data)
            candidate_array = np.asarray(candidate_data)
            if baseline_array.shape != candidate_array.shape:
                unavailable_issues.append(
                    "stage8_shape_mismatch="
                    f"{baseline_array.shape}!={candidate_array.shape}"
                )
            if not bool(np.all(np.isfinite(baseline_array))):
                unavailable_issues.append("stage8_baseline_nonfinite_pixels")
            if not bool(np.all(np.isfinite(candidate_array))):
                unavailable_issues.append("stage8_candidate_nonfinite_pixels")
        except (TypeError, ValueError):
            unavailable_issues.append("stage8_pixel_validation_unavailable")
    if unavailable_issues:
        empty_metrics = asdict(QualityMetrics())
        return {
            "status": "poor",
            "issues": unavailable_issues,
            "local_issues": unavailable_issues,
            "advisories": [],
            "local_advisories": [],
            "quality_gates": {},
            "baseline_metrics": empty_metrics,
            "candidate_metrics": empty_metrics,
            "masked_metrics": {"baseline": {}, "candidate": {}},
            "mask_coverage": {},
            "limited_halo_texture": {
                "available": False,
                "accepted": False,
                "reason": "required image data unavailable",
            },
            "protection_actions": getattr(
                pipeline,
                "_last_stage8_masked_diagnostics",
                {},
            ).get("protection_actions", []),
            "derived": {},
        }
    baseline_metrics = (
        measure_quality_metrics(baseline_data) if baseline_data is not None else QualityMetrics()
    )
    candidate_metrics = (
        measure_quality_metrics(candidate_data) if candidate_data is not None else QualityMetrics()
    )
    masks = (
        pipeline._stage8_generate_starless_masks(baseline_data)
        if baseline_data is not None
        else None
    )
    baseline_masked_metrics = pipeline._stage8_masked_metrics(baseline_data, masks)
    candidate_masked_metrics = pipeline._stage8_masked_metrics(candidate_data, masks)

    saturation_growth = (
        candidate_metrics.saturation_p95 / max(baseline_metrics.saturation_p95, 0.05)
    )
    microcontrast_growth = (
        candidate_metrics.microcontrast / max(baseline_metrics.microcontrast, 1e-4)
    )
    bg_std_growth = (
        candidate_masked_metrics.get("background_std", 0.0)
        / max(baseline_masked_metrics.get("background_std", 0.0), 1e-5)
    )
    background_brightening = (
        candidate_masked_metrics.get("background_brightness", 0.0)
        - baseline_masked_metrics.get("background_brightness", 0.0)
    )
    core_clip_growth = (
        candidate_masked_metrics.get("core_clip_ratio", 0.0)
        - baseline_masked_metrics.get("core_clip_ratio", 0.0)
    )
    diffuse_gain = (
        candidate_masked_metrics.get("diffuse_signal", 0.0)
        / max(baseline_masked_metrics.get("diffuse_signal", 0.0), 1e-5)
    )
    texture_artifact_growth = (
        candidate_masked_metrics.get("texture_artifact_score", 0.0)
        / max(baseline_masked_metrics.get("texture_artifact_score", 0.0), 1e-5)
    )
    issues: List[str] = []
    advisories: List[str] = []
    quality_gates: Dict[str, Dict[str, Any]] = {}

    def record_upper_gate(
        metric_name: str,
        value: float,
        limit: float,
        issue_text: str,
    ) -> None:
        gate = stage7_quality.stage7_9_upper_quality_gate(
            pipeline.cfg,
            value=value,
            accepted_limit=limit,
        )
        quality_gates[metric_name] = gate
        if gate["hard_failed"]:
            issues.append(issue_text)
        elif gate["advisory"]:
            advisories.append(f"{issue_text} (advisory; candidate retained)")

    halo_texture_report: Dict[str, Any] = {
        "available": False,
        "accepted": True,
        "reason": "not_required",
    }
    if limited_mode:
        halo_texture_report = stage8_limited_halo_texture_report(
            pipeline,
            baseline_data,
            candidate_data,
            starmask_data,
        )
        if not bool(halo_texture_report.get("available")):
            issues.append(
                "limited_halo_texture_gate_unavailable="
                + str(halo_texture_report.get("reason") or "unknown")
            )
        elif not bool(halo_texture_report.get("accepted")):
            issues.append(
                "limited_halo_texture_growth "
                f"{float(halo_texture_report.get('growth', 0.0)):.3f}>"
                f"{float(halo_texture_report.get('growth_limit', 1.05)):.3f}, "
                "delta "
                f"{float(halo_texture_report.get('absolute_delta', 0.0)):.6f}>"
                f"{float(halo_texture_report.get('absolute_delta_limit', 0.0)):.6f}"
            )
        else:
            advisories.extend(halo_texture_report.get("advisories") or [])
            for metric_name, gate in (
                halo_texture_report.get("quality_gates") or {}
            ).items():
                quality_gates[f"limited_halo_texture_{metric_name}"] = gate
    blue_issue_threshold = pipeline.cfg.stage8_blue_excess_max + 0.012
    if candidate_metrics.blue_excess > blue_issue_threshold:
        record_upper_gate(
            "blue_excess",
            candidate_metrics.blue_excess,
            blue_issue_threshold,
            f"blue_excess {candidate_metrics.blue_excess:.3f}>{blue_issue_threshold:.3f}",
        )
    if (
        saturation_growth > pipeline.cfg.stage8_saturation_growth_ratio_max
        and candidate_metrics.saturation_p95 > 0.28
    ):
        record_upper_gate(
            "saturation_growth",
            saturation_growth,
            float(pipeline.cfg.stage8_saturation_growth_ratio_max),
            "saturation_growth "
            f"{saturation_growth:.3f}>{pipeline.cfg.stage8_saturation_growth_ratio_max:.3f}",
        )
    if (
        microcontrast_growth > pipeline.cfg.stage8_microcontrast_growth_ratio_max
        and candidate_metrics.microcontrast > baseline_metrics.microcontrast + 0.004
    ):
        record_upper_gate(
            "microcontrast_growth",
            microcontrast_growth,
            float(pipeline.cfg.stage8_microcontrast_growth_ratio_max),
            "microcontrast_growth "
            f"{microcontrast_growth:.3f}>{pipeline.cfg.stage8_microcontrast_growth_ratio_max:.3f}",
        )
    if candidate_metrics.highlight_clip_ratio > pipeline.cfg.stage8_highlight_clip_ratio_max:
        record_upper_gate(
            "highlight_clip_ratio",
            candidate_metrics.highlight_clip_ratio,
            float(pipeline.cfg.stage8_highlight_clip_ratio_max),
            "highlight_clip_ratio "
            f"{candidate_metrics.highlight_clip_ratio:.4f}>{pipeline.cfg.stage8_highlight_clip_ratio_max:.4f}",
        )
    bg_noise_issue = _stage8_bg_noise_growth_issue(
        pipeline,
        growth=bg_std_growth,
        baseline_std=float(baseline_masked_metrics.get("background_std", 0.0) or 0.0),
        candidate_std=float(candidate_masked_metrics.get("background_std", 0.0) or 0.0),
        candidate_dirty_score=float(
            candidate_masked_metrics.get("background_std", 0.0) or 0.0
        )
        / max(float(candidate_masked_metrics.get("background_brightness", 0.0) or 0.0), 0.015),
    )
    if bg_noise_issue:
        record_upper_gate(
            "bg_std_growth",
            bg_std_growth,
            float(pipeline.cfg.stage8_bg_std_growth_max),
            bg_noise_issue,
        )
    if (
        background_brightening > 0.020
        and candidate_masked_metrics.get("background_brightness", 0.0) > 0.08
    ):
        record_upper_gate(
            "background_brightening",
            background_brightening,
            0.020,
            f"background_brightening {background_brightening:.4f}>0.0200",
        )
    if core_clip_growth > 0.010:
        record_upper_gate(
            "core_clip_growth",
            core_clip_growth,
            0.010,
            f"core_clip_growth {core_clip_growth:.4f}>0.0100",
        )
    if (
        texture_artifact_growth > pipeline.cfg.stage8_texture_artifact_growth_max
        and candidate_masked_metrics.get("texture_artifact_score", 0.0) > 0.003
    ):
        record_upper_gate(
            "texture_artifact_growth",
            texture_artifact_growth,
            float(pipeline.cfg.stage8_texture_artifact_growth_max),
            "texture_artifact_growth "
            f"{texture_artifact_growth:.3f}>{pipeline.cfg.stage8_texture_artifact_growth_max:.3f}",
        )

    observations = {
        "baseline_metrics": asdict(baseline_metrics),
        "candidate_metrics": asdict(candidate_metrics),
        "baseline_masked_metrics": baseline_masked_metrics,
        "candidate_masked_metrics": candidate_masked_metrics,
        "mask_coverage": masks.get("coverage", {}) if masks else {},
        "derived": {
            "saturation_growth": saturation_growth,
            "microcontrast_growth": microcontrast_growth,
            "bg_std_growth": bg_std_growth,
            "background_brightening": background_brightening,
            "core_clip_growth": core_clip_growth,
            "diffuse_gain": diffuse_gain,
            "texture_artifact_growth": texture_artifact_growth,
        },
        "limited_halo_texture": halo_texture_report,
        "local_issues": issues,
        "local_advisories": advisories,
        "quality_gates": quality_gates,
    }
    all_issues = issues
    return {
        "status": "ok" if not all_issues else "poor",
        "issues": all_issues,
        "local_issues": issues,
        "advisories": advisories,
        "local_advisories": advisories,
        "quality_gates": quality_gates,
        "quality_advisory_multiplier": (
            stage7_quality.stage7_9_quality_advisory_multiplier(pipeline.cfg)
        ),
        "baseline_metrics": asdict(baseline_metrics),
        "candidate_metrics": asdict(candidate_metrics),
        "masked_metrics": {
            "baseline": baseline_masked_metrics,
            "candidate": candidate_masked_metrics,
        },
        "mask_coverage": observations["mask_coverage"],
        "limited_halo_texture": halo_texture_report,
        "protection_actions": getattr(
            pipeline,
            "_last_stage8_masked_diagnostics",
            {},
        ).get("protection_actions", []),
        "derived": observations["derived"],
    }

def stage8_conservative_rerun(pipeline, original_saturation: float) -> Dict[str, Any]:
    safe_saturation = _clamp_float(min(original_saturation * 0.5, 0.18), 0.0, 0.18)
    saturation_enabled = bool(
        getattr(pipeline.cfg, "stage8_nebula_saturation_enabled", True)
    )
    unsharp_enabled = bool(
        getattr(pipeline.cfg, "stage8_masked_unsharp_enabled", True)
    )
    if not saturation_enabled:
        safe_saturation = 0.0
    masked_enhancement_enabled = bool(
        getattr(pipeline.cfg, "stage8_masked_enhancement_enabled", False)
    )
    physical_color_anchor = physical_broadband_anchor_accepted(
        getattr(pipeline, "channel_profile", {}) or {},
        getattr(pipeline, "color_calibration_report", {}) or {},
    )
    if physical_color_anchor and not masked_enhancement_enabled:
        safe_saturation = 0.0
    result: Dict[str, Any] = {
        "status": "failed",
        "safe_saturation": safe_saturation,
        "issues": [],
    }
    try:
        pipeline.cmd_with_check("load", "stage8_input_starless")
        if masked_enhancement_enabled:
            image_data = pipeline.siril.get_image_pixeldata(preview=False)
            if image_data is None:
                raise RuntimeError("image buffer is empty")
            enhanced, diagnostics, messages = pipeline._apply_stage8_masked_pixel_enhancement(
                np.asarray(image_data),
                {
                    "saturation": safe_saturation,
                    "unsharp_amount": min(0.08, pipeline.cfg.stage8_masked_unsharp_amount_max),
                },
                label="conservative",
            )
            pipeline._set_current_image_pixeldata(enhanced, label="stage8 conservative masked rerun")
            pipeline._last_stage8_masked_diagnostics = diagnostics
            pipeline._stage8_saturation_execution = dict(
                diagnostics.get("saturation_execution") or {}
            )
            result["issues"].extend(messages)
            pipeline._save_stage_output("stage8_enhanced")
            pipeline._save_stage_output("stage8_conservative_enhanced")
            assessment = pipeline._stage8_quality_assessment()
            result.update(
                {
                    "status": "ok" if assessment["status"] == "ok" else "poor",
                    "assessment": assessment,
                }
            )
            return result
        pipeline._stage8_saturation_execution = {
            "requested": safe_saturation,
            "applied": False,
            "applied_amount": 0.0,
            "passes": 0,
            "position": "after_structure_unmasked_path",
            "method": "none",
        }
        if physical_color_anchor:
            result["issues"].append(
                "global saturation skipped: Stage4 physical color anchor "
                "requires masked color recovery"
            )
        if unsharp_enabled:
            try:
                pipeline.cmd_with_check("unsharp", "0.4", "0.20")
            except (CommandError, SirilError) as e:
                result["issues"].append(
                    f"conservative unsharp skipped: {pipeline._short_text(e, 120)}"
                )
        else:
            result["issues"].append(
                "conservative unsharp disabled by processing parameters"
            )
        if safe_saturation > 1e-6:
            pipeline.cmd_with_check(
                "satu",
                f"{safe_saturation:.6f}",
                str(pipeline.cfg.nebula_bg_factor),
            )
            pipeline._stage8_saturation_execution.update(
                {
                    "applied": True,
                    "applied_amount": safe_saturation,
                    "passes": 1,
                    "method": "siril_satu",
                }
            )
        pipeline._save_stage_output("stage8_enhanced")
        pipeline._save_stage_output("stage8_conservative_enhanced")
        assessment = pipeline._stage8_quality_assessment()
        result.update(
            {
                "status": "ok" if assessment["status"] == "ok" else "poor",
                "assessment": assessment,
            }
        )
        return result
    except (CommandError, SirilError) as e:
        result["issues"].append(pipeline._short_text(e, 180))
        return result

def stage8_needs_conservative_rerun(pipeline, quality_record: Dict[str, Any]) -> bool:
    issues = [str(item) for item in quality_record.get("issues", [])]
    issue_text = " ".join(issues).lower()
    if any(
        token in issue_text
        for token in (
            "blue_excess",
            "saturation_growth",
            "microcontrast_growth",
            "highlight_clip_ratio",
            "bg_std_growth",
            "background_brightening",
            "core_clip_growth",
            "texture_artifact_growth",
        )
    ):
        return True
    return False
