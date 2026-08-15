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
import starmask_cleanup
import stage7_quality

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
except ImportError:  # Tests may import with lightweight fakes.
    CommandError = RuntimeError
    DataError = RuntimeError
    SirilError = RuntimeError


def _stage7_galaxy_protection_mask(
    pipeline,
    source_data: np.ndarray,
) -> Optional[np.ndarray]:
    """Return the fitted galaxy disk that generic repair must not rewrite."""
    target_type = (
        str(pipeline._active_target_type() or "").strip().lower()
        if hasattr(pipeline, "_active_target_type")
        else ""
    )
    if (
        target_type not in stage7_quality.GALAXY_TARGET_TYPES
        or not bool(
            getattr(
                pipeline.cfg,
                "stage7_galaxy_roi_halo_gate_enabled",
                True,
            )
        )
    ):
        return None
    source_arr = np.asarray(source_data)
    source_rgb = _to_rgb_float_image(source_arr, max_side=1024)
    source_gray = (
        0.2126 * source_rgb[0]
        + 0.7152 * source_rgb[1]
        + 0.0722 * source_rgb[2]
    ).astype(np.float32)
    source_features = measure_image_features(source_rgb)
    masks = stage7_quality.stage7_galaxy_structure_masks(
        source_gray,
        float(source_features.bg_median),
        max(float(source_features.bg_std), 1e-5),
    )
    if not bool(masks.get("available", False)):
        return None
    disk_mask = np.asarray(masks["disk_mask"], dtype=bool)
    full_rgb = _to_rgb_float_fullres(source_arr)
    full_shape = full_rgb.shape[1:]
    if disk_mask.shape == full_shape:
        return disk_mask
    source_height, source_width = disk_mask.shape
    target_height, target_width = full_shape
    y_index = np.minimum(
        (np.arange(target_height) * source_height / target_height).astype(int),
        source_height - 1,
    )
    x_index = np.minimum(
        (np.arange(target_width) * source_width / target_width).astype(int),
        source_width - 1,
    )
    return disk_mask[y_index[:, None], x_index[None, :]]

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
    if not pipeline.starmask_file or not pipeline.starmask_file.exists():
        result["reason"] = "starmask_missing"
        return result

    process_dir = getattr(pipeline, "process_dir", None) or pipeline.starmask_file.parent
    raw_path = process_dir / "starmask_raw.fit"
    try:
        if pipeline.starmask_file != raw_path:
            shutil.copy2(pipeline.starmask_file, raw_path)
        result["raw_file"] = raw_path.name
    except OSError as e:
        result["status"] = "failed"
        result["reason"] = f"raw starmask preservation failed: {pipeline._short_text(e, 160)}"
        return result
    if not pipeline.cfg.stage7_starmask_clean_enabled:
        result["reason"] = "disabled; raw starmask preserved"
        return result

    try:
        pipeline.cmd_with_check("load", pipeline.starmask_file.stem)
        starmask_data = pipeline.siril.get_image_pixeldata(preview=False)
        if starmask_data is None:
            result["reason"] = "starmask_buffer_empty"
            return result
        output, metrics = starmask_cleanup.clean_starmask_pixels(
            np.asarray(starmask_data),
            pipeline.cfg,
        )
        result["metrics"] = metrics
        result["source_stem"] = source_stem or pipeline.stretched_name or "stage5_linear"
        if not bool(metrics.get("accepted", False)):
            hard_gate_failed = bool(
                metrics.get("diffuse_hard_gate_failed", False)
                or metrics.get("compact_hard_gate_failed", False)
                or metrics.get("faint_compact_hard_gate_failed", False)
            )
            result["status"] = (
                "hard_rejected" if hard_gate_failed else "rolled_back"
            )
            result["hard_gate_failed"] = hard_gate_failed
            result["reason"] = ", ".join(str(item) for item in metrics.get("issues", []))
            pipeline.log.warn(
                "Stage6 starmask cleanup rejected; original mask retained "
                f"(label={label}, reason={result['reason']})"
            )
            return result

        pipeline.cmd_with_check("load", pipeline.starmask_file.stem)
        lock_factory = getattr(pipeline.siril, "image_lock", None)
        if callable(lock_factory):
            with lock_factory():
                pipeline.siril.set_image_pixeldata(output)
        else:
            pipeline.siril.set_image_pixeldata(output)
        pipeline.cmd_with_check("save", "starmask_clean")
        pipeline.starmask_file = process_dir / "starmask_clean.fit"
        result["clean_file"] = "starmask_clean.fit"

        result["status"] = "applied"
        pipeline.log.info(
            "Stage6 starmask multi-scale cleanup applied "
            f"(label={label}, signal_ratio={metrics['signal_ratio']:.3f}, "
            f"compact_retention={metrics['compact_retention']:.3f}, "
            f"diffuse_residual={metrics['diffuse_residual_ratio']:.3f})"
        )
        return result
    except (CommandError, SirilError, DataError, RuntimeError, ValueError) as e:
        result["status"] = "failed"
        result["reason"] = pipeline._short_text(e, 180)
        pipeline.log.warn(f"Stage6 starmask cleanup failed: {e}")
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
        galaxy_protect_like: Optional[np.ndarray] = None
        if source_data is not None:
            source = pipeline._match_star_layer_shape(np.asarray(source_data), starless)
            source = np.nan_to_num(
                source.astype(np.float32, copy=False),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            galaxy_protect_2d = _stage7_galaxy_protection_mask(
                pipeline,
                source,
            )
            if galaxy_protect_2d is not None:
                galaxy_protect_rgb = np.broadcast_to(
                    galaxy_protect_2d[None, :, :],
                    (3, *galaxy_protect_2d.shape),
                ).astype(np.float32)
                galaxy_protect_like = (
                    pipeline._stage8_restore_rgb_like(
                        starless,
                        galaxy_protect_rgb,
                    )
                    > 0.5
                )
                mask = np.where(galaxy_protect_like, 0.0, mask)
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
                if galaxy_protect_like is not None:
                    residual_mask = np.where(
                        galaxy_protect_like,
                        0.0,
                        residual_mask,
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
        if galaxy_protect_like is not None:
            # Generic smoothing/subtraction must never rewrite the fitted
            # bulge, arms or dust lanes. A bad core/disk stays visible to the
            # quality gate and forces a new separation candidate instead.
            suppressed = np.where(galaxy_protect_like, starless, suppressed)
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
        if galaxy_protect_like is not None:
            mode += "+galaxy-disk-protected"
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
        galaxy_protect_mask = _stage7_galaxy_protection_mask(
            pipeline,
            source_arr,
        )
        if galaxy_protect_mask is None:
            galaxy_protect_mask = np.zeros_like(source_gray, dtype=bool)
        generic_repair_mask = (~galaxy_protect_mask).astype(np.float32)

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
        small_star_seed = (
            (source_detail > detail_threshold).astype(np.float32)
            * generic_repair_mask
        )

        core_threshold = max(
            float(np.quantile(source_gray, 0.995)),
            source_bg + max(6.0 * source_std, 0.08),
        )
        bright_star_core = (
            (source_gray > core_threshold).astype(np.float32)
            * generic_repair_mask
        )
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
        starmask_weight *= generic_repair_mask
        expanded_star_weight = starmask_weight.copy()
        for _ in range(5):
            blurred_star = _box_blur_gray(expanded_star_weight)
            expanded_star_weight = np.maximum(expanded_star_weight, np.clip(blurred_star * 1.9, 0.0, 1.0))
        expanded_star_weight = pipeline._stage8_soften_mask(
            np.clip(expanded_star_weight, 0.0, 1.0),
            passes=1,
        )
        expanded_star_weight *= generic_repair_mask

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
        residual_weight *= generic_repair_mask
        star_seed = np.maximum(expanded_star_weight, small_star_seed)
        residual_weight = np.maximum(
            residual_weight * np.clip(star_seed + 0.35, 0.0, 1.0),
            star_seed * 0.42,
        )
        residual_weight = pipeline._stage8_soften_mask(np.clip(residual_weight, 0.0, 1.0), passes=2)
        residual_weight *= generic_repair_mask

        repaired = starless_rgb.copy()
        residual_strength = float(pipeline.cfg.stage7_starless_repair_strength)
        residual_blend = np.clip(residual_weight * residual_strength, 0.0, 0.82)
        residual_target = strong_local_rgb * 0.72 + local_rgb * 0.28
        repaired = repaired * (1.0 - residual_blend[None, :, :]) + residual_target * residual_blend[None, :, :]

        halo_strength = float(pipeline.cfg.stage7_starless_halo_repair_strength)
        halo_weight = np.maximum(halo_weight, expanded_star_weight * 0.55)
        halo_weight *= generic_repair_mask
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
        protected = np.clip(
            object_mask
            + residual_weight
            + halo_weight
            + galaxy_protect_mask.astype(np.float32),
            0.0,
            1.0,
        )
        background_weight = pipeline._stage8_soften_mask(1.0 - protected, passes=2)
        background_weight = np.clip(background_weight, 0.0, 1.0)
        background_weight[galaxy_protect_mask] = 0.0
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

        # Preserve the accepted starless pixels in the fitted galaxy disk
        # exactly. Generic residual/halo repair is allowed only outside it;
        # disk halo or core loss remains a fail-closed candidate-selection
        # problem and is never concealed by smoothing real galaxy structure.
        repaired[:, galaxy_protect_mask] = starless_rgb[:, galaxy_protect_mask]

        output = pipeline._stage8_restore_rgb_like(starless_arr, np.clip(repaired, 0.0, 1.0))
        background_quality_before = pipeline._background_quality_metrics(starless_arr)
        background_quality_after = pipeline._background_quality_metrics(output)
        pipeline.cmd_with_check("load", "starless")
        pipeline._set_current_image_pixeldata(
            output,
            label="stage7 starless pixel repair",
        )
        pipeline._save_stage_output("starless")
        pipeline._save_stage_output("stage6_starless_repaired")
        pipeline.starless_file = pipeline.process_dir / "starless.fit"

        metrics = {
            "residual_repair_coverage": float(np.mean(residual_weight > 0.05)),
            "expanded_star_repair_coverage": float(np.mean(expanded_star_weight > 0.05)),
            "halo_repair_coverage": float(np.mean(halo_blend > 0.02)),
            "dark_defect_coverage": float(np.mean(dark_weight > 0.05)),
            "background_chroma_denoise_coverage": float(np.mean(background_weight > 0.50)),
            "galaxy_structure_protection_coverage": float(
                np.mean(galaxy_protect_mask)
            ),
            "residual_strength": residual_strength,
            "halo_strength": halo_strength,
            "chroma_strength": chroma_strength,
            "background_quality_before": background_quality_before,
            "background_quality_after": background_quality_after,
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
