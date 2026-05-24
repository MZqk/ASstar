from __future__ import annotations

import importlib
import math
import os
import shutil
import subprocess
import sys
import types
import zipfile
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
from models import ImageFeatures, QualityMetrics

try:
    from sirilpy.exceptions import CommandError, DataError, SirilError
except Exception:  # Tests may import with lightweight fakes.
    CommandError = Exception
    DataError = Exception
    SirilError = Exception

def stage7_clean_starmask(
    pipeline,
    *,
    label: str = "initial",
    source_stem: Optional[str] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "label": label,
        "status": "skipped",
        "reason": "",
        "metrics": {},
    }
    if not pipeline.cfg.stage7_starmask_clean_enabled:
        result["reason"] = "disabled"
        return result
    if not pipeline.starmask_file or not pipeline.starmask_file.exists():
        result["reason"] = "starmask_missing"
        return result

    source_stem = source_stem or pipeline.stretched_name or "stage7_stretched"
    try:
        source_data = pipeline._read_image_by_stem(source_stem)
        if source_data is None:
            result["reason"] = "source_unavailable"
            return result

        pipeline.cmd_with_check("load", pipeline.starmask_file.stem)
        starmask_data = pipeline.siril.get_image_pixeldata(preview=False)
        if starmask_data is None:
            result["reason"] = "starmask_buffer_empty"
            return result

        stars_arr = np.asarray(starmask_data)
        output_dtype = stars_arr.dtype
        stars = np.nan_to_num(
            stars_arr.astype(np.float32, copy=False),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        stars = np.clip(stars, 0.0, None)
        if not stars.size or float(np.max(stars)) <= 1e-8:
            result["reason"] = "starmask_empty"
            return result

        source = pipeline._match_star_layer_shape(np.asarray(source_data), stars)
        source_rgb = _to_rgb_float_image(source, max_side=100_000)
        source_gray = (
            0.2126 * source_rgb[0]
            + 0.7152 * source_rgb[1]
            + 0.0722 * source_rgb[2]
        ).astype(np.float32)
        stars_rgb = _to_rgb_float_image(stars, max_side=100_000)
        stars_gray = (
            0.2126 * stars_rgb[0]
            + 0.7152 * stars_rgb[1]
            + 0.0722 * stars_rgb[2]
        ).astype(np.float32)
        if source_gray.shape != stars_gray.shape:
            result["reason"] = "shape_mismatch"
            return result

        cleaned = stars.copy()
        positive = stars_gray[stars_gray > 1e-7]
        if positive.size:
            floor = float(np.percentile(
                positive,
                pipeline.cfg.stage7_starmask_background_floor_percentile,
            ))
            high = float(np.percentile(positive, 99.5))
            if math.isfinite(floor) and math.isfinite(high) and high > floor:
                keep_weight = np.clip((stars_gray - floor) / (high - floor), 0.0, 1.0)
                keep_weight = keep_weight ** 0.70
                cleaned *= (
                    keep_weight[None, :, :]
                    if cleaned.ndim == 3
                    else keep_weight
                )
            small_limit = float(np.percentile(positive, 90.0))
            small_region = (stars_gray > floor) & (stars_gray < small_limit)
            if int(np.count_nonzero(small_region)) > 0:
                small_weight = (
                    np.broadcast_to(small_region, cleaned.shape)
                    if cleaned.ndim == 3
                    else small_region
                )
                cleaned = np.where(
                    small_weight,
                    cleaned * float(pipeline.cfg.stage7_starmask_small_star_scale),
                    cleaned,
                )
        else:
            floor = 0.0

        source_features = measure_image_features(source)
        source_bg = float(source_features.bg_median)
        source_std = max(float(source_features.bg_std), 1e-5)
        core_threshold = max(
            float(np.quantile(source_gray, 0.995)),
            source_bg + max(6.0 * source_std, 0.08),
        )
        core_mask = source_gray > core_threshold
        halo_region = np.zeros_like(source_gray, dtype=bool)
        if int(np.count_nonzero(core_mask)) > 0:
            halo_weight = core_mask.astype(np.float32)
            for _ in range(4):
                halo_weight = _box_blur_gray(halo_weight)
            halo_region = halo_weight > 0.006
            if (
                int(np.count_nonzero(halo_region)) > 16
                and pipeline.cfg.stage7_starmask_halo_blur_strength > 1e-4
            ):
                blur_strength = float(pipeline.cfg.stage7_starmask_halo_blur_strength)
                if cleaned.ndim == 3:
                    blurred = np.empty_like(cleaned)
                    for idx in range(cleaned.shape[0]):
                        blurred[idx] = _box_blur_gray(cleaned[idx])
                    halo_weight_3d = np.broadcast_to(halo_region, cleaned.shape)
                    cleaned = np.where(
                        halo_weight_3d,
                        cleaned * (1.0 - blur_strength) + blurred * blur_strength,
                        cleaned,
                    )
                else:
                    blurred = _box_blur_gray(cleaned)
                    cleaned = np.where(
                        halo_region,
                        cleaned * (1.0 - blur_strength) + blurred * blur_strength,
                        cleaned,
                    )

        object_threshold = max(
            float(np.quantile(source_gray, 0.70)),
            source_bg + max(1.8 * source_std, 0.02),
        )
        diffuse_mask = source_gray > object_threshold
        star_weight = core_mask.astype(np.float32)
        for _ in range(4):
            star_weight = _box_blur_gray(star_weight)
        nebula_mask = diffuse_mask & (star_weight <= 0.003)
        nebula_pixels = int(np.count_nonzero(nebula_mask))
        if nebula_pixels > 16 and pipeline.cfg.stage7_starmask_nebula_suppression > 1e-4:
            suppression = float(pipeline.cfg.stage7_starmask_nebula_suppression)
            nebula_weight = (
                np.broadcast_to(nebula_mask, cleaned.shape)
                if cleaned.ndim == 3
                else nebula_mask
            )
            cleaned = np.where(
                nebula_weight,
                cleaned * (1.0 - suppression),
                cleaned,
            )

        before_signal = float(np.sum(stars))
        after_signal = float(np.sum(cleaned))
        cleaned = np.clip(cleaned, 0.0, None)
        if np.issubdtype(output_dtype, np.integer):
            info = np.iinfo(output_dtype)
            output = np.clip(cleaned, info.min, info.max).astype(output_dtype, copy=False)
        else:
            output = cleaned.astype(np.float32, copy=False)

        pipeline.cmd_with_check("load", pipeline.starmask_file.stem)
        lock_factory = getattr(pipeline.siril, "image_lock", None)
        if callable(lock_factory):
            with lock_factory():
                pipeline.siril.set_image_pixeldata(output)
        else:
            pipeline.siril.set_image_pixeldata(output)
        pipeline.cmd_with_check("save", "starmask")
        pipeline.starmask_file = pipeline.process_dir / "starmask.fit"

        result.update(
            {
                "status": "applied",
                "metrics": {
                    "background_floor": float(floor),
                    "signal_before": before_signal,
                    "signal_after": after_signal,
                    "signal_ratio": after_signal / max(before_signal, 1e-7),
                    "halo_pixels": int(np.count_nonzero(halo_region)),
                    "nebula_suppressed_pixels": nebula_pixels,
                },
            }
        )
        pipeline.log.info(
            "Stage7 starmask cleanup applied "
            f"(label={label}, signal_ratio={result['metrics']['signal_ratio']:.3f}, "
            f"nebula_pixels={nebula_pixels})"
        )
        return result
    except (CommandError, SirilError, DataError, RuntimeError, ValueError) as e:
        result["status"] = "failed"
        result["reason"] = pipeline._short_text(e, 180)
        pipeline.log.warn(f"Stage7 starmask cleanup failed: {e}")
        return result

def apply_stage7_residual_suppression(
    pipeline,
    strength: float,
    *,
    source_stem: Optional[str] = None,
) -> Optional[str]:
    strength = _clamp_float(strength, 0.0, 0.25)
    if strength <= 1e-4 or not pipeline.starmask_file or not pipeline.starmask_file.exists():
        return None
    try:
        source_data = pipeline._read_image_by_stem(
            source_stem or pipeline.stretched_name or "stage7_stretched"
        )
        pipeline.cmd_with_check("load", "starless")
        starless_data = pipeline.siril.get_image_pixeldata(preview=False)
        pipeline.cmd_with_check("load", pipeline.starmask_file.stem)
        starmask_data = pipeline.siril.get_image_pixeldata(preview=False)
        if starless_data is None or starmask_data is None:
            return None
        starless = np.nan_to_num(
            np.asarray(starless_data).astype(np.float32, copy=False),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        stars = pipeline._match_star_layer_shape(np.asarray(starmask_data), starless)
        stars = np.nan_to_num(
            stars.astype(np.float32, copy=False),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        stars = np.clip(stars, 0.0, None)
        scale = float(np.percentile(stars, 99.7))
        if not math.isfinite(scale) or scale <= 1e-6:
            scale = float(np.max(stars)) if stars.size else 0.0
        if scale <= 1e-6:
            return None
        mask = np.clip(stars / scale, 0.0, 1.0)
        residual_mask = None
        if source_data is not None:
            source = pipeline._match_star_layer_shape(np.asarray(source_data), starless)
            source = np.nan_to_num(
                source.astype(np.float32, copy=False),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            residual = np.clip(source - starless, 0.0, None)
            residual_rgb = _to_rgb_float_image(residual, max_side=100_000)
            residual_luma = (
                0.2126 * residual_rgb[0]
                + 0.7152 * residual_rgb[1]
                + 0.0722 * residual_rgb[2]
            )
            floor = float(np.percentile(residual_luma, 70.0))
            residual_luma = np.clip(residual_luma - floor, 0.0, None)
            residual_scale = float(np.percentile(residual_luma, 99.7))
            if math.isfinite(residual_scale) and residual_scale > 1e-7:
                residual_mask_2d = np.clip(residual_luma / residual_scale, 0.0, 1.0)
                residual_mask = (
                    np.broadcast_to(residual_mask_2d, starless.shape)
                    if starless.ndim == 3
                    else residual_mask_2d
                )
                mask = np.maximum(mask, residual_mask * 0.85)
        starless_rgb = _to_rgb_float_fullres(starless)
        smooth_rgb = starless_rgb.copy()
        for _ in range(2):
            smooth_rgb = pipeline._box_blur_rgb(smooth_rgb)
        smooth_starless = pipeline._stage8_restore_rgb_like(starless, smooth_rgb).astype(
            np.float32,
            copy=False,
        )
        blend_mask = np.clip(mask * strength, 0.0, 0.45)
        suppressed = starless * (1.0 - blend_mask) + smooth_starless * blend_mask
        if source_data is not None and residual_mask is not None:
            source = pipeline._match_star_layer_shape(np.asarray(source_data), starless).astype(
                np.float32,
                copy=False,
            )
            residual = np.clip(source - starless, 0.0, None)
            suppressed = np.clip(
                suppressed - residual * residual_mask * (strength * 0.08),
                0.0,
                None,
            )
        pipeline.cmd_with_check("load", "starless")
        lock_factory = getattr(pipeline.siril, "image_lock", None)
        if callable(lock_factory):
            with lock_factory():
                pipeline.siril.set_image_pixeldata(suppressed.astype(np.float32, copy=False))
        else:
            pipeline.siril.set_image_pixeldata(suppressed.astype(np.float32, copy=False))
        pipeline._save_stage_output("starless")
        pipeline.starless_file = pipeline.process_dir / "starless.fit"
        mode = "starmask+residual-map" if residual_mask is not None else "starmask"
        return f"stage7 residual suppression applied (strength={strength:.3f}, mode={mode})"
    except (CommandError, SirilError, DataError, RuntimeError, ValueError) as e:
        pipeline.log.warn(f"stage7 residual suppression skipped: {e}")
        return None

def apply_stage7_starless_pixel_repair(
    pipeline,
    *,
    source_stem: str,
    label: str,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "label": label,
        "source_stem": source_stem,
        "status": "skipped",
        "metrics": {},
    }
    if not bool(getattr(pipeline.cfg, "stage7_starless_pixel_repair_enabled", True)):
        result["reason"] = "disabled"
        return result
    try:
        source_data = pipeline._read_image_by_stem(source_stem)
        if source_data is None:
            result["reason"] = "source_unavailable"
            return result

        pipeline.cmd_with_check("load", "starless")
        starless_data = pipeline.siril.get_image_pixeldata(preview=False)
        if starless_data is None:
            result["reason"] = "starless_buffer_empty"
            return result

        starless_arr = np.nan_to_num(
            np.asarray(starless_data).astype(np.float32, copy=False),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        source_arr = pipeline._match_star_layer_shape(np.asarray(source_data), starless_arr)
        source_rgb = _to_rgb_float_fullres(source_arr)
        starless_rgb = _to_rgb_float_fullres(starless_arr)
        source_gray = (
            0.2126 * source_rgb[0]
            + 0.7152 * source_rgb[1]
            + 0.0722 * source_rgb[2]
        ).astype(np.float32)
        starless_gray = (
            0.2126 * starless_rgb[0]
            + 0.7152 * starless_rgb[1]
            + 0.0722 * starless_rgb[2]
        ).astype(np.float32)
        if source_gray.shape != starless_gray.shape:
            result["reason"] = "shape_mismatch"
            return result

        source_features = measure_image_features(source_arr)
        starless_features = measure_image_features(starless_arr)
        source_bg = float(source_features.bg_median)
        source_std = max(float(source_features.bg_std), 1e-5)
        starless_std = max(float(starless_features.bg_std), 1e-5)

        local_rgb = starless_rgb.copy()
        for _ in range(3):
            local_rgb = pipeline._box_blur_rgb(local_rgb)
        strong_local_rgb = local_rgb.copy()
        for _ in range(3):
            strong_local_rgb = pipeline._box_blur_rgb(strong_local_rgb)
        local_gray = (
            0.2126 * local_rgb[0]
            + 0.7152 * local_rgb[1]
            + 0.0722 * local_rgb[2]
        ).astype(np.float32)

        source_blur = _box_blur_gray(source_gray)
        source_detail = np.clip(source_gray - source_blur, 0.0, None)
        detail_threshold = max(
            float(np.percentile(source_detail, 98.8)),
            source_std * 1.6,
            0.006,
        )
        small_star_seed = (source_detail > detail_threshold).astype(np.float32)

        core_threshold = max(
            float(np.quantile(source_gray, 0.995)),
            source_bg + max(6.0 * source_std, 0.08),
        )
        bright_star_core = (source_gray > core_threshold).astype(np.float32)
        halo_weight = bright_star_core.copy()
        for _ in range(6):
            halo_weight = _box_blur_gray(halo_weight)
        halo_weight = np.clip(halo_weight - bright_star_core, 0.0, 1.0)

        starmask_weight = np.zeros_like(source_gray, dtype=np.float32)
        if pipeline.starmask_file and pipeline.starmask_file.exists():
            starmask_data = pipeline._read_image_by_stem(pipeline.starmask_file.stem)
            if starmask_data is not None:
                starmask_arr = pipeline._match_star_layer_shape(
                    np.asarray(starmask_data),
                    starless_arr,
                )
                starmask_rgb = _to_rgb_float_fullres(starmask_arr)
                starmask_gray = (
                    0.2126 * starmask_rgb[0]
                    + 0.7152 * starmask_rgb[1]
                    + 0.0722 * starmask_rgb[2]
                ).astype(np.float32)
                scale = float(np.percentile(starmask_gray, 99.7))
                if not math.isfinite(scale) or scale <= 1e-7:
                    scale = float(np.max(starmask_gray)) if starmask_gray.size else 0.0
                if scale > 1e-7:
                    starmask_weight = np.clip(starmask_gray / scale, 0.0, 1.0)
        expanded_star_weight = starmask_weight.copy()
        for _ in range(5):
            blurred_star = _box_blur_gray(expanded_star_weight)
            expanded_star_weight = np.maximum(expanded_star_weight, np.clip(blurred_star * 1.9, 0.0, 1.0))
        expanded_star_weight = pipeline._stage8_soften_mask(
            np.clip(expanded_star_weight, 0.0, 1.0),
            passes=1,
        )

        residual_detail = np.clip(
            starless_gray - local_gray - max(1.4 * starless_std, 0.004),
            0.0,
            None,
        )
        residual_scale = float(np.percentile(residual_detail, 99.4))
        if not math.isfinite(residual_scale) or residual_scale <= 1e-7:
            residual_scale = float(np.max(residual_detail)) if residual_detail.size else 0.0
        if residual_scale > 1e-7:
            residual_weight = np.clip(residual_detail / residual_scale, 0.0, 1.0)
        else:
            residual_weight = np.zeros_like(source_gray, dtype=np.float32)
        star_seed = np.maximum(expanded_star_weight, small_star_seed)
        residual_weight = np.maximum(
            residual_weight * np.clip(star_seed + 0.35, 0.0, 1.0),
            star_seed * 0.42,
        )
        residual_weight = pipeline._stage8_soften_mask(np.clip(residual_weight, 0.0, 1.0), passes=2)

        repaired = starless_rgb.copy()
        residual_strength = float(pipeline.cfg.stage7_starless_repair_strength)
        residual_blend = np.clip(residual_weight * residual_strength, 0.0, 0.82)
        residual_target = strong_local_rgb * 0.72 + local_rgb * 0.28
        repaired = repaired * (1.0 - residual_blend[None, :, :]) + residual_target * residual_blend[None, :, :]

        halo_strength = float(pipeline.cfg.stage7_starless_halo_repair_strength)
        halo_weight = np.maximum(halo_weight, expanded_star_weight * 0.55)
        halo_blend = np.clip(halo_weight * halo_strength, 0.0, 0.78)
        if float(np.max(halo_blend)) > 1e-6:
            halo_rgb = repaired.copy()
            for _ in range(6):
                halo_rgb = pipeline._box_blur_rgb(halo_rgb)
            halo_target = halo_rgb * 0.70 + strong_local_rgb * 0.30
            repaired = repaired * (1.0 - halo_blend[None, :, :]) + halo_target * halo_blend[None, :, :]
            repaired_gray = (
                0.2126 * repaired[0]
                + 0.7152 * repaired[1]
                + 0.0722 * repaired[2]
            ).astype(np.float32)
            broad_rgb = repaired.copy()
            for _ in range(10):
                broad_rgb = pipeline._box_blur_rgb(broad_rgb)
            broad_gray = (
                0.2126 * broad_rgb[0]
                + 0.7152 * broad_rgb[1]
                + 0.0722 * broad_rgb[2]
            ).astype(np.float32)
            halo_floor = np.minimum(
                broad_gray,
                float(starless_features.bg_median) + max(3.0 * starless_std, 0.006),
            )
            damped_gray = halo_floor + np.clip(repaired_gray - halo_floor, 0.0, None) * (
                1.0 - np.clip(halo_blend * 0.68, 0.0, 0.62)
            )
            gray_scale = np.divide(
                damped_gray,
                np.maximum(repaired_gray, 1e-6),
                out=np.ones_like(repaired_gray, dtype=np.float32),
                where=repaired_gray > 1e-6,
            )
            repaired = repaired * (1.0 - halo_blend[None, :, :]) + (
                repaired * gray_scale[None, :, :]
            ) * halo_blend[None, :, :]

        repaired_gray = (
            0.2126 * repaired[0]
            + 0.7152 * repaired[1]
            + 0.0722 * repaired[2]
        ).astype(np.float32)
        dark_defect = np.clip(
            local_gray - repaired_gray - max(2.0 * starless_std, 0.008),
            0.0,
            None,
        )
        dark_scale = float(np.percentile(dark_defect, 99.5))
        if math.isfinite(dark_scale) and dark_scale > 1e-7:
            dark_weight = np.clip(dark_defect / dark_scale, 0.0, 1.0)
            dark_weight = pipeline._stage8_soften_mask(
                np.clip(dark_weight * np.maximum(residual_weight, halo_weight), 0.0, 1.0),
                passes=1,
            )
            repaired = repaired * (1.0 - 0.50 * dark_weight[None, :, :]) + local_rgb * (
                0.50 * dark_weight[None, :, :]
            )
        else:
            dark_weight = np.zeros_like(source_gray, dtype=np.float32)

        object_threshold = max(
            float(np.quantile(source_gray, 0.68)),
            source_bg + max(1.7 * source_std, 0.020),
        )
        object_ramp = (source_gray - object_threshold) / max(3.0 * source_std, 0.030)
        object_mask = pipeline._stage8_soften_mask(np.clip(object_ramp, 0.0, 1.0), passes=3)
        protected = np.clip(object_mask + residual_weight + halo_weight, 0.0, 1.0)
        background_weight = pipeline._stage8_soften_mask(1.0 - protected, passes=2)
        background_weight = np.clip(background_weight, 0.0, 1.0)
        chroma_strength = float(pipeline.cfg.stage7_starless_chroma_denoise_strength)
        repaired_gray = (
            0.2126 * repaired[0]
            + 0.7152 * repaired[1]
            + 0.0722 * repaired[2]
        ).astype(np.float32)
        chroma = repaired - repaired_gray[None, :, :]
        chroma_keep = 1.0 - np.clip(background_weight * chroma_strength, 0.0, 0.82)
        repaired = repaired_gray[None, :, :] + chroma * chroma_keep[None, :, :]

        bg_smooth = repaired.copy()
        for _ in range(2):
            bg_smooth = pipeline._box_blur_rgb(bg_smooth)
        bg_blend = np.clip(background_weight * chroma_strength * 0.16, 0.0, 0.14)
        repaired = repaired * (1.0 - bg_blend[None, :, :]) + bg_smooth * bg_blend[None, :, :]

        output = pipeline._stage8_restore_rgb_like(starless_arr, np.clip(repaired, 0.0, 1.0))
        pipeline.cmd_with_check("load", "starless")
        pipeline._set_current_image_pixeldata(
            output,
            label="stage7 starless pixel repair",
        )
        pipeline._save_stage_output("starless")
        pipeline._save_stage_output("stage7_starless_repaired")
        pipeline.starless_file = pipeline.process_dir / "starless.fit"

        metrics = {
            "residual_repair_coverage": float(np.mean(residual_weight > 0.05)),
            "expanded_star_repair_coverage": float(np.mean(expanded_star_weight > 0.05)),
            "halo_repair_coverage": float(np.mean(halo_blend > 0.02)),
            "dark_defect_coverage": float(np.mean(dark_weight > 0.05)),
            "background_chroma_denoise_coverage": float(np.mean(background_weight > 0.50)),
            "residual_strength": residual_strength,
            "halo_strength": halo_strength,
            "chroma_strength": chroma_strength,
        }
        result.update({"status": "applied", "metrics": metrics})
        pipeline.log.info(
            "Stage7 starless pixel repair applied "
            f"(residual_cov={metrics['residual_repair_coverage']:.4f}, "
            f"halo_cov={metrics['halo_repair_coverage']:.4f}, "
            f"bg_cov={metrics['background_chroma_denoise_coverage']:.4f})"
        )
        return result
    except (CommandError, SirilError, DataError, RuntimeError, ValueError) as e:
        result["status"] = "failed"
        result["reason"] = pipeline._short_text(e, 180)
        pipeline.log.warn(f"Stage7 starless pixel repair failed: {e}")
        return result
