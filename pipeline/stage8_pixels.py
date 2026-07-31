from __future__ import annotations

import importlib
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

def _clamp_int(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(value)))

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
        0.82,
    )
    core_ramp = (gray - core_threshold) / max(1.0 - core_threshold, 1e-3)
    core_mask = pipeline._stage8_soften_mask(np.clip(core_ramp, 0.0, 1.0), passes=4)

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
        "nebula": float(np.mean(nebula_mask > 0.12)),
        "faint_nebula": float(np.mean(faint_nebula_mask > 0.12)),
        "background": float(np.mean(background_mask > 0.50)),
    }
    return {
        "rgb": rgb,
        "gray": gray,
        "bg_median": bg_median,
        "bg_std": bg_std,
        "core_mask": core_mask,
        "nebula_mask": nebula_mask,
        "faint_nebula_mask": faint_nebula_mask,
        "background_mask": background_mask,
        "coverage": coverage,
    }

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

        background_weight = np.clip(background, 0.0, 1.0)
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
) -> Dict[str, float]:
    if image_data is None:
        return {}
    try:
        rgb = _to_rgb_float_fullres(image_data)
        gray = (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]).astype(np.float32)
        if masks and "background_mask" in masks:
            bg_weight = np.clip(np.asarray(masks["background_mask"], dtype=np.float32), 0.0, 1.0)
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
        for y in range(0, max(h - patch + 1, 1), patch):
            for x in range(0, max(w - patch + 1, 1), patch):
                tile_weight = bg_weight[y:y + patch, x:x + patch]
                if tile_weight.size and float(np.mean(tile_weight)) > 0.65:
                    patch_vars.append(float(np.var(gray[y:y + patch, x:x + patch])))
        patch_variance = float(np.median(patch_vars)) if patch_vars else 0.0
        red = weighted_mean(rgb[0])
        green = weighted_mean(rgb[1])
        blue = weighted_mean(rgb[2])
        background_level = weighted_mean(gray)
        chroma_bias_mean = weighted_mean(chroma_bias)
        blue_excess = max(0.0, blue / max(green, 1e-6) - max(1.08, red / max(green, 1e-6) + 0.12))
        return {
            "bg_median": background_level,
            "bg_std": weighted_std(gray),
            "bg_dirty_score": _clamp_float(weighted_std(gray) / max(background_level, 0.015), 0.0, 2.0),
            "chroma_noise_score": _clamp_float(weighted_mean(chroma_noise) / max(weighted_std(gray) * 2.0, 0.01), 0.0, 2.0),
            "background_chroma_bias_score": _clamp_float(chroma_bias_mean / max(background_level, 0.015), 0.0, 2.0),
            "background_chroma_load": chroma_bias_mean / max(background_level, 1e-4),
            "blue_excess_score": _clamp_float(blue_excess, 0.0, 2.0),
            "background_mottling_score": _clamp_float(weighted_mean(mottling) / max(weighted_std(gray) * 2.0, 0.006), 0.0, 2.0),
            "local_patch_variance": patch_variance,
            "core_clip_score": float(np.mean(gray >= 0.985)),
            "starless_artifact_score": _clamp_float(weighted_mean(local_texture) / max(weighted_std(gray) * 3.0, 0.006), 0.0, 2.0),
        }
    except (TypeError, ValueError, IndexError, FloatingPointError) as e:
        pipeline.log.warn(f"background quality metrics unavailable: {e}")
        return {}

def stage8_enhancement_quality_report(pipeline) -> Dict[str, Any]:
    before_data = pipeline._read_image_by_stem("stage8_input_starless")
    after_data = pipeline._read_image_by_stem("stage8_enhanced")
    masks = None
    if before_data is not None:
        try:
            masks = pipeline._stage8_generate_starless_masks(before_data)
        except (CommandError, SirilError, RuntimeError, TypeError, ValueError):
            masks = None
    before = pipeline._background_quality_metrics(before_data, masks)
    after = pipeline._background_quality_metrics(after_data, masks)
    issues: List[str] = []
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
            issues.append(dirty_growth_issue)
        if chroma_growth > 1.10:
            issues.append(f"chroma_noise_score_growth {chroma_growth:.3f}>1.100")
        if after.get("background_mottling_score", 0.0) > 0.55:
            issues.append(f"background_mottling_score {after.get('background_mottling_score', 0.0):.3f}>0.550")
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
        pipeline._save_stage_output("starless_enhanced")
        pipeline._save_stage_output("stage8_enhanced")
        return True
    except (CommandError, SirilError) as e:
        pipeline.log.warn(f"Stage8 rollback to input failed: {e}")
        return False

def final_quality_report(pipeline, stem: str = "stage10_final") -> Dict[str, Any]:
    image_data = pipeline._read_image_by_stem(stem)
    metrics = pipeline._background_quality_metrics(image_data)
    issues: List[str] = []
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
    stage9_missing_required_stars = bool(
        stage9_contract_known
        and stage9_stars_required
        and not stage9_stars_applied
    )
    stage9_starmask_stretch_failed = bool(
        getattr(pipeline, "_stage9_starmask_stretch_failed", False)
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
        or stage9_starmask_stretch_failed
        or halo_residue > 0.70
        or compact_halo_limit_exceeded
    )
    if compact_halo_limit_exceeded:
        issues.append(
            "stage7_compact_halo_residue_score "
            f"{compact_halo_residue:.3f}>{stage7_halo_limit:.3f}"
        )
    if not metrics:
        issues.append("final_quality_metrics_unavailable")
    else:
        chroma = float(metrics.get("chroma_noise_score", 0.0) or 0.0)
        mottling = float(metrics.get("background_mottling_score", 0.0) or 0.0)
        patch_var = float(metrics.get("local_patch_variance", 0.0) or 0.0)
        core_clip = float(metrics.get("core_clip_score", 0.0) or 0.0)
        artifact = float(metrics.get("starless_artifact_score", 0.0) or 0.0)
        chroma_limit = 0.34 if strict_gate else 0.42
        mottling_limit = 0.45 if strict_gate else 0.55
        patch_limit = 0.00016 if strict_gate else 0.00022
        artifact_limit = 0.52 if strict_gate else 0.62
        if chroma > chroma_limit:
            issues.append(f"background_chroma_noise_score {chroma:.3f}>{chroma_limit:.3f}")
        if mottling > mottling_limit:
            issues.append(f"background_mottling_score {mottling:.3f}>{mottling_limit:.3f}")
        if patch_var > patch_limit:
            issues.append(f"local_patch_variance {patch_var:.6f}>{patch_limit:.6f}")
        if core_clip > 0.012:
            issues.append(f"core_clip_score {core_clip:.4f}>0.0120")
        if artifact > artifact_limit:
            issues.append(f"starless_artifact_score {artifact:.3f}>{artifact_limit:.3f}")
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
        stage9_starmask_stretch_failed
        and "stage9_starmask_stretch_failed" not in issues
    ):
        issues.append("stage9_starmask_stretch_failed")
    if (
        stage9_chromatic_addition_ratio is not None
        and stage9_chromatic_addition_ratio > stage9_chromatic_addition_limit
    ):
        issues.append(
            "stage9_chromatic_star_addition_ratio "
            f"{stage9_chromatic_addition_ratio:.6f}>"
            f"{stage9_chromatic_addition_limit:.6f}"
        )
    normalized_metrics = {
        "background_chroma_noise_score": metrics.get("chroma_noise_score") if metrics else None,
        "background_chroma_bias_score": metrics.get("background_chroma_bias_score") if metrics else None,
        "background_chroma_load": metrics.get("background_chroma_load") if metrics else None,
        "background_mottling_score": metrics.get("background_mottling_score") if metrics else None,
        "local_patch_variance": metrics.get("local_patch_variance") if metrics else None,
        "local_patch_variance_score": metrics.get("local_patch_variance") if metrics else None,
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
        "stage9_starmask_stretch_failed": stage9_starmask_stretch_failed,
        "stage9_chromatic_star_addition_ratio": stage9_chromatic_addition_ratio,
        "stage9_chromatic_star_addition_ratio_max": (
            stage9_chromatic_addition_limit
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
    }
    return {
        "stage": "stage10_final_quality",
        "file": f"{stem}.fit",
        "policy": pipeline._active_policy_name() if hasattr(pipeline, "_active_policy_name") else "",
        "target_type": active_target_type,
        "status": "needs_conservative_rerun" if issues else "ok",
        "final_quality": "poor" if issues else "ok",
        "needs_conservative_rerun": bool(issues),
        "strict_gate": strict_gate,
        "metrics": normalized_metrics,
        "issues": issues,
    }

def stage8_input_enhancement_guard(pipeline) -> Dict[str, Any]:
    reasons: List[str] = []
    handoff = getattr(pipeline, "_stage8_handoff", {}) or {}
    handoff = handoff if isinstance(handoff, dict) else {}
    requested_policy = str(
        handoff.get("processing_policy")
        or handoff.get("requested_policy")
        or ""
    ).strip().lower()
    handoff_reason_text = str(handoff.get("reason_text") or "").strip()
    handoff_reason_details = list(handoff.get("reasons") or [])
    guard_reason_code = str(handoff.get("reason_code") or "")
    guard_reason_text = handoff_reason_text
    advisories: List[str] = []
    if requested_policy == "skip":
        reasons.append(
            handoff_reason_text
            or str(handoff.get("reason_code") or "stage8_handoff_requested_skip")
        )
    elif requested_policy == "limited":
        if handoff_reason_text:
            advisories.append(handoff_reason_text)
    elif not requested_policy and bool(
        getattr(pipeline, "_stage8_conservative_mode", False)
    ):
        # Old checkpoints only carried a boolean. Keep them fail-closed without
        # inventing a repair reason that is not present in the evidence.
        requested_policy = "skip"
        reasons.append("stage8_conservative_mode_legacy")
    elif not requested_policy:
        requested_policy = "full"
    if requested_policy == "limited":
        if not bool(
            getattr(pipeline.cfg, "stage8_masked_enhancement_enabled", False)
        ):
            reasons.append("stage8_limited_masked_enhancement_disabled")
        starmask_file = getattr(pipeline, "starmask_file", None)
        if not (
            starmask_file is not None
            and callable(getattr(starmask_file, "exists", None))
            and starmask_file.exists()
        ):
            reasons.append("stage8_limited_starmask_unavailable")
    quality = getattr(pipeline, "_stage7_selected_quality", None)
    derived = quality.get("derived") if isinstance(quality, dict) else {}
    if bool(getattr(pipeline, "_stage7_starless_skipped", False)):
        reasons.append("stage7_starless_skipped_by_pre_starless_gate")
    status = str(quality.get("status", "")) if isinstance(quality, dict) else ""
    if status and status not in {"ok"}:
        reasons.append(f"stage7_quality_status={status}")

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
    if residual > pipeline.cfg.stage7_residual_star_score_max:
        reasons.append(
            f"stage7_residual_star_score {residual:.3f}>{pipeline.cfg.stage7_residual_star_score_max:.3f}"
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
            reasons.append("stage8_limited_masked_enhancement_disabled")
        starmask_file = getattr(pipeline, "starmask_file", None)
        if not (
            starmask_file is not None
            and callable(getattr(starmask_file, "exists", None))
            and starmask_file.exists()
        ):
            reasons.append("stage8_limited_starmask_unavailable")
    if halo > halo_threshold:
        reasons.append(
            f"stage7_halo_residue_score {halo:.3f}>{halo_threshold:.3f}"
        )
    if noise_gain > pipeline.cfg.stage7_starless_noise_gain_max:
        reasons.append(
            f"stage7_starless_noise_gain {noise_gain:.3f}>{pipeline.cfg.stage7_starless_noise_gain_max:.3f}"
        )

    coverage: Dict[str, float] = {}
    mask_signal_coverage = None
    try:
        image_data = pipeline.siril.get_image_pixeldata(preview=False)
        if image_data is not None:
            masks = pipeline._stage8_generate_starless_masks(np.asarray(image_data))
            raw_coverage = masks.get("coverage", {})
            if isinstance(raw_coverage, dict):
                coverage = {str(k): float(v) for k, v in raw_coverage.items()}
            mask_signal_coverage = float(coverage.get("nebula", 0.0)) + float(
                coverage.get("faint_nebula", 0.0)
            )
            if mask_signal_coverage < pipeline.cfg.stage8_mask_signal_coverage_min:
                reasons.append(
                    "stage8_mask_signal_coverage "
                    f"{mask_signal_coverage:.4f}<{pipeline.cfg.stage8_mask_signal_coverage_min:.4f}"
                )
    except (CommandError, SirilError, OSError, RuntimeError, TypeError, ValueError) as e:
        reasons.append(f"stage8_mask_guard_unavailable={pipeline._short_text(e, 120)}")

    processing_policy = (
        "skip"
        if reasons
        else "limited"
        if requested_policy == "limited"
        else "full"
    )
    conservative_skip = bool(reasons and requested_policy in {"limited", "skip"})
    skip_status = (
        "conservative_skipped"
        if conservative_skip
        else "skipped"
        if reasons
        else "ok"
    )
    return {
        "skip_enhancement": bool(reasons),
        "processing_policy": processing_policy,
        "requested_policy": requested_policy,
        "conservative_mode": processing_policy in {"limited", "skip"},
        "status": skip_status,
        "final_quality": skip_status,
        "reasons": reasons,
        "advisories": advisories,
        "reason_details": handoff_reason_details,
        "reason_code": guard_reason_code,
        "reason_text": guard_reason_text,
        "mask_coverage": coverage,
        "mask_signal_coverage": mask_signal_coverage,
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
    coverage = masks.get("coverage", {})
    target_type = pipeline._active_target_type() if hasattr(pipeline, "_active_target_type") else ""
    high_halo_risk = pipeline._stage7_halo_residue_score() > pipeline._stage7_effective_halo_threshold()
    object_mask_only = target_type == "bright_emission_reflection_nebula" or high_halo_risk
    mask_signal_coverage = float(coverage.get("nebula", 0.0)) + float(
        coverage.get("faint_nebula", 0.0)
    )
    mask_quality_scale = 1.0
    messages: List[str] = []

    saturation = _clamp_float(plan.get("saturation", pipeline.cfg.nebula_saturation), 0.0, 0.65)
    unsharp_amount = min(
        _clamp_float(plan.get("unsharp_amount", 0.35), 0.0, 0.60),
        float(pipeline.cfg.stage8_masked_unsharp_amount_max),
    )
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

    core_clip = float(np.mean(gray >= 0.985))
    if core_clip > pipeline.cfg.stage8_highlight_clip_ratio_max:
        unsharp_amount = 0.0
        saturation *= 0.65
        messages.append(f"{label} stage8 highlight guard disabled masked unsharp")

    denoise_strength = float(pipeline.cfg.stage8_background_denoise_strength)
    faint_boost = min(float(pipeline.cfg.stage8_faint_nebula_boost_max), 0.20 * saturation) * mask_quality_scale
    contrast_strength = min(float(pipeline.cfg.stage8_nebula_contrast_max), 0.28 * saturation) * mask_quality_scale
    signal_weight = np.clip(nebula + 0.60 * faint, 0.0, 1.0)
    color_sample_weight = np.clip(signal_weight * (1.0 - core) * (1.0 - 0.85 * background), 0.0, 1.0)
    color_weight_total = float(np.sum(color_sample_weight))
    blue_pre_gain = 1.0
    if color_weight_total > 1e-6:
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
    )
    result = result * (1.0 - denoise_weight[None, :, :]) + blurred * denoise_weight[None, :, :]
    denoised_base = result.copy()

    if blue_pre_gain < 0.999:
        blue_weight = np.clip((nebula + 0.45 * faint) * (1.0 - core) * (1.0 - background), 0.0, 1.0)
        result[2] = result[2] * (1.0 - blue_weight + blue_weight * blue_pre_gain)

    if faint_boost > 1e-6:
        background_guard = 1.0 - (0.98 if object_mask_only else 0.80) * background
        lift = faint_boost * faint * background_guard * np.clip(1.0 - gray, 0.0, 1.0)
        result = result + lift[None, :, :]

    if contrast_strength > 1e-6:
        background_guard = 1.0 - (1.0 if object_mask_only else 0.90) * background
        signal_weight = np.clip((nebula + 0.30 * faint) * background_guard, 0.0, 1.0)
        total = float(np.sum(signal_weight))
        center = (
            float(np.sum(gray * signal_weight) / total)
            if total > 1e-6
            else float(np.median(gray))
        )
        contrast_weight = contrast_strength * nebula * (1.0 - core)
        result = center + (result - center) * (1.0 + contrast_weight[None, :, :])

    if unsharp_amount > 1e-6:
        blurred = pipeline._box_blur_rgb(result)
        background_guard = 1.0 - (1.0 if object_mask_only else 0.98) * background
        detail_weight = np.clip(
            unsharp_amount * nebula * (1.0 - core) * background_guard,
            0.0,
            float(pipeline.cfg.stage8_masked_unsharp_amount_max),
        )
        result = result + (result - blurred) * detail_weight[None, :, :]

    if saturation > 1e-6:
        gray_after = (
            0.2126 * result[0] + 0.7152 * result[1] + 0.0722 * result[2]
        ).astype(np.float32)
        background_guard = 1.0 - (1.0 if object_mask_only else 0.98) * background
        sat_weight = np.clip(
            saturation * (0.78 * nebula + 0.18 * faint) * (1.0 - core) * background_guard,
            0.0,
            0.35,
        )
        result = gray_after[None, :, :] + (result - gray_after[None, :, :]) * (
            1.0 + sat_weight[None, :, :]
        )

    if plugin_candidate is not None:
        plugin_rgb = _to_rgb_float_fullres(plugin_candidate)
        if blue_pre_gain < 0.999:
            blue_weight = np.clip((nebula + 0.45 * faint) * (1.0 - core) * (1.0 - background), 0.0, 1.0)
            plugin_rgb[2] = plugin_rgb[2] * (1.0 - blue_weight + blue_weight * blue_pre_gain)
        background_guard = 1.0 - (1.0 if object_mask_only else 0.98) * background
        plugin_weight = np.clip(
            (0.34 * nebula + 0.16 * faint) * (1.0 - core) * background_guard,
            0.0,
            0.24 if object_mask_only else 0.38,
        )
        result = result * (1.0 - plugin_weight[None, :, :]) + plugin_rgb * plugin_weight[None, :, :]
        messages.append(f"{label} SASP output blended through Starless masks")

    core_protection = float(pipeline.cfg.stage8_core_protection_strength)
    background_restore_strength = 1.0 if object_mask_only else 0.985
    background_restore = np.clip(background_restore_strength * background, 0.0, 1.0)
    result = result * (1.0 - background_restore[None, :, :]) + denoised_base * background_restore[None, :, :]
    core_restore = np.clip(core_protection * core, 0.0, 1.0)
    result = result * (1.0 - core_restore[None, :, :]) + base * core_restore[None, :, :]

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
        if faint_boost > 1e-6:
            local_operations.append(
                {
                    "type": "curve",
                    "mask": "faint_nebula",
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
            local_operations.append(
                {
                    "type": "saturation",
                    "mask": "nebula",
                    "amount": min(0.05, saturation * 0.12),
                    "opacity": 0.50 * mask_quality_scale,
                }
            )
        if contrast_strength > 1e-6:
            local_operations.append(
                {
                    "type": "local_contrast",
                    "mask": "nebula",
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
                    "id": "stage8_nebula_local_v1",
                    "operations": local_operations,
                },
                masks={
                    "background": background,
                    "core": core,
                    "nebula": nebula,
                    "faint_nebula": faint,
                },
            )
        )
        if bool(local_adjustment_report.get("accepted", False)):
            result = np.asarray(local_candidate, dtype=np.float32)
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

    restored = pipeline._stage8_restore_rgb_like(image_data, np.clip(result, 0.0, 1.0))
    diagnostics = {
        "mask_coverage": masks["coverage"],
        "masked_metrics": pipeline._stage8_masked_metrics(restored, masks),
        "protection_actions": messages,
        "local_adjustment_engine": local_adjustment_report,
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

    if bool(getattr(pipeline.cfg, "stage8_masked_enhancement_enabled", False)):
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
        return masked_messages

    if saturation > 1e-6:
        pipeline.cmd_with_check("satu", f"{saturation:.6f}", str(bg_factor))
        messages.append(
            f"{label} Starless satu (sat={saturation:.4f}, bg={bg_factor})"
        )
    else:
        messages.append(f"{label} Starless satu skipped (sat=0)")

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
    pipeline._save_stage_output("starless_enhanced")
    pipeline._save_stage_output("stage8_enhanced")
    return (
        "AI stage8 color correction applied "
        f"(blue_excess={blue_excess:.3f}, target={target_blue_excess:.3f}, "
        f"b_gain={b_gain:.3f})"
    )

def stage8_target_blue_excess(pipeline, quality_record: Optional[Dict[str, Any]]) -> float:
    ai_assessment = (quality_record or {}).get("ai_assessment")
    if isinstance(ai_assessment, dict):
        value = ai_assessment.get("target_blue_excess")
        if value is not None:
            try:
                return _clamp_float(value, 0.05, 0.16)
            except (TypeError, ValueError):
                pass
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
        accepted = growth <= growth_limit or absolute_delta <= delta_limit
        report.update(
            {
                "available": True,
                "accepted": accepted,
                "reason": "" if accepted else "halo annulus texture growth exceeded",
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
    blue_issue_threshold = pipeline.cfg.stage8_blue_excess_max + 0.012
    if candidate_metrics.blue_excess > blue_issue_threshold:
        issues.append(
            f"blue_excess {candidate_metrics.blue_excess:.3f}>{blue_issue_threshold:.3f}"
        )
    if (
        saturation_growth > pipeline.cfg.stage8_saturation_growth_ratio_max
        and candidate_metrics.saturation_p95 > 0.28
    ):
        issues.append(
            "saturation_growth "
            f"{saturation_growth:.3f}>{pipeline.cfg.stage8_saturation_growth_ratio_max:.3f}"
        )
    if (
        microcontrast_growth > pipeline.cfg.stage8_microcontrast_growth_ratio_max
        and candidate_metrics.microcontrast > baseline_metrics.microcontrast + 0.004
    ):
        issues.append(
            "microcontrast_growth "
            f"{microcontrast_growth:.3f}>{pipeline.cfg.stage8_microcontrast_growth_ratio_max:.3f}"
        )
    if candidate_metrics.highlight_clip_ratio > pipeline.cfg.stage8_highlight_clip_ratio_max:
        issues.append(
            "highlight_clip_ratio "
            f"{candidate_metrics.highlight_clip_ratio:.4f}>{pipeline.cfg.stage8_highlight_clip_ratio_max:.4f}"
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
        issues.append(bg_noise_issue)
    if (
        background_brightening > 0.020
        and candidate_masked_metrics.get("background_brightness", 0.0) > 0.08
    ):
        issues.append(f"background_brightening {background_brightening:.4f}>0.0200")
    if core_clip_growth > 0.010:
        issues.append(f"core_clip_growth {core_clip_growth:.4f}>0.0100")
    if (
        texture_artifact_growth > pipeline.cfg.stage8_texture_artifact_growth_max
        and candidate_masked_metrics.get("texture_artifact_score", 0.0) > 0.003
    ):
        issues.append(
            "texture_artifact_growth "
            f"{texture_artifact_growth:.3f}>{pipeline.cfg.stage8_texture_artifact_growth_max:.3f}"
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
    }
    ai_assessment = pipeline._request_stage8_quality_ai(observations)
    ai_issues: List[str] = []
    if ai_assessment:
        if ai_assessment["verdict"] != "ok":
            ai_issues.append(f"ai_verdict={ai_assessment['verdict']}")
        if ai_assessment["oversaturated"]:
            ai_issues.append("ai_oversaturated")
        if ai_assessment["blue_bias"]:
            ai_issues.append("ai_blue_bias")
        if ai_assessment["microcontrast_overdone"]:
            ai_issues.append("ai_microcontrast_overdone")

    all_issues = issues + ai_issues
    return {
        "status": "ok" if not all_issues else "poor",
        "issues": all_issues,
        "local_issues": issues,
        "ai_assessment": ai_assessment,
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
    result: Dict[str, Any] = {
        "status": "failed",
        "safe_saturation": safe_saturation,
        "issues": [],
    }
    try:
        pipeline.cmd_with_check("load", "stage8_input_starless")
        if bool(getattr(pipeline.cfg, "stage8_masked_enhancement_enabled", False)):
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
            result["issues"].extend(messages)
            pipeline._save_stage_output("starless_enhanced")
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
        if safe_saturation > 1e-6:
            pipeline.cmd_with_check(
                "satu",
                f"{safe_saturation:.6f}",
                str(pipeline.cfg.nebula_bg_factor),
            )
        try:
            pipeline.cmd_with_check("unsharp", "0.4", "0.20")
        except (CommandError, SirilError) as e:
            result["issues"].append(
                f"conservative unsharp skipped: {pipeline._short_text(e, 120)}"
            )
        pipeline._save_stage_output("starless_enhanced")
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
            "ai_blue_bias",
            "saturation_growth",
            "microcontrast_growth",
            "highlight_clip_ratio",
            "bg_std_growth",
            "background_brightening",
            "core_clip_growth",
            "texture_artifact_growth",
            "ai_oversaturated",
            "ai_microcontrast_overdone",
        )
    ):
        return True
    ai_assessment = quality_record.get("ai_assessment")
    if isinstance(ai_assessment, dict):
        return bool(
            ai_assessment.get("oversaturated")
            or ai_assessment.get("blue_bias")
            or ai_assessment.get("microcontrast_overdone")
            or ai_assessment.get("recommended_action") in {"conservative_rerun", "rollback"}
        )
    return False
