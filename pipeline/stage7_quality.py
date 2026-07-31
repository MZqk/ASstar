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
    from sirilpy.exceptions import CommandError, SirilError
except ImportError:  # Tests may import with lightweight fakes.
    CommandError = RuntimeError
    SirilError = RuntimeError


DIFFUSE_EMISSION_NEBULA_TARGET_TYPES = frozenset(
    {
        "emission_nebula",
        "emission_nebula_widefield",
        "bright_emission_reflection_nebula",
    }
)
DIFFUSE_EMISSION_HALO_RESIDUE_SCORE_MAX = 0.45


def stage7_dynamic_range_assessment(
    cfg,
    *,
    dynamic_range_ratio: float,
    peak_signal: float,
    background_level: float,
) -> Dict[str, Any]:
    """Calibrate the linear starless collapse gate against its background floor."""
    dynamic_range_ratio = float(dynamic_range_ratio)
    peak_signal = float(peak_signal)
    background_level = float(background_level)
    if not math.isfinite(dynamic_range_ratio):
        dynamic_range_ratio = 0.0
    if not math.isfinite(peak_signal):
        peak_signal = 0.0
    if not math.isfinite(background_level):
        background_level = 0.0
    dynamic_range_ratio = max(dynamic_range_ratio, 0.0)
    peak_signal = max(peak_signal, 0.0)
    background_level = max(background_level, 0.0)
    dynamic_threshold = float(
        getattr(cfg, "stage7_starless_dynamic_range_min_ratio", 0.55)
    )
    peak_threshold = float(
        getattr(cfg, "stage7_starless_peak_signal_min", 0.006)
    )
    peak_background_ratio_threshold = float(
        getattr(cfg, "stage7_starless_peak_background_ratio_min", 4.0)
    )
    peak_background_ratio = peak_signal / max(background_level, 1e-4)
    collapsed = bool(
        dynamic_range_ratio < dynamic_threshold
        and peak_signal < peak_threshold
        and peak_background_ratio < peak_background_ratio_threshold
    )
    return {
        "collapsed": collapsed,
        "dynamic_range_ratio_min": dynamic_threshold,
        "peak_signal_min": peak_threshold,
        "peak_background_ratio": peak_background_ratio,
        "peak_background_ratio_min": peak_background_ratio_threshold,
    }


def stage7_starless_artifact_scores(
    pipeline,
    source_data: Optional[np.ndarray],
    starless_data: Optional[np.ndarray],
    starmask_data: Optional[np.ndarray],
    source_features: ImageFeatures,
    starless_features: ImageFeatures,
) -> Dict[str, float]:
    scores = {
        "halo_residue_score": 0.0,
        "global_halo_residue_score": 0.0,
        "compact_halo_residue_score": 0.0,
        "compact_halo_mask_coverage": 0.0,
        "compact_halo_source_level": 0.0,
        "compact_halo_starless_level": 0.0,
        "black_hole_score": 0.0,
        "compact_residual_star_score": 0.0,
        "compact_residual_coverage": 0.0,
        "starmask_contamination": 0.0,
        "starless_noise_gain": 1.0,
        "starless_dynamic_range_ratio": 1.0,
        "source_dynamic_range": 0.0,
        "starless_dynamic_range": 0.0,
        "source_peak_signal": 0.0,
        "starless_peak_signal": 0.0,
    }
    if source_data is None or starless_data is None:
        return scores

    try:
        target_type = pipeline._active_target_type() if hasattr(pipeline, "_active_target_type") else ""
        is_protected_nebula = target_type in DIFFUSE_EMISSION_NEBULA_TARGET_TYPES
        source_rgb = _to_rgb_float_image(source_data, max_side=1024)
        starless_rgb = _to_rgb_float_image(starless_data, max_side=1024)
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
            return scores

        source_bg = float(source_features.bg_median)
        source_std = max(float(source_features.bg_std), 1e-5)
        starless_bg = float(starless_features.bg_median)
        starless_std = max(float(starless_features.bg_std), 1e-5)
        try:
            source_q01, source_q99 = np.quantile(source_gray, (0.01, 0.99))
            starless_q01, starless_q99 = np.quantile(starless_gray, (0.01, 0.99))
            source_range = max(float(source_q99 - source_q01), 1e-7)
            starless_range = max(float(starless_q99 - starless_q01), 0.0)
            scores["source_dynamic_range"] = source_range
            scores["starless_dynamic_range"] = starless_range
            scores["starless_dynamic_range_ratio"] = _clamp_float(
                starless_range / source_range,
                0.0,
                10.0,
            )
            scores["source_peak_signal"] = float(np.nanmax(source_gray))
            scores["starless_peak_signal"] = float(np.nanmax(starless_gray))
        except (TypeError, ValueError, FloatingPointError):
            pass
        broad_source = source_gray.copy()
        for _ in range(5):
            broad_source = _box_blur_gray(broad_source)
        nebula_threshold = max(
            float(np.quantile(broad_source, 0.76)),
            source_bg + max(1.8 * source_std, 0.018),
        )
        diffuse_nebula_protect = (
            broad_source > nebula_threshold
            if is_protected_nebula
            else np.zeros_like(source_gray, dtype=bool)
        )
        scores["starless_noise_gain"] = _clamp_float(
            starless_std / max(source_std, 1e-5),
            0.0,
            10.0,
        )

        core_threshold = max(
            float(np.quantile(source_gray, 0.995)),
            source_bg + max(6.0 * source_std, 0.08),
        )
        core_mask = source_gray > core_threshold
        if int(np.count_nonzero(core_mask)) > 0:
            halo_weight = core_mask.astype(np.float32)
            for _ in range(5):
                halo_weight = _box_blur_gray(halo_weight)
            halo_mask = (halo_weight > 0.004) & (~core_mask)
            if is_protected_nebula:
                halo_mask &= ~diffuse_nebula_protect
            if int(np.count_nonzero(halo_mask)) > 16:
                source_halo = np.clip(source_gray[halo_mask] - source_bg, 0.0, None)
                starless_halo = np.clip(starless_gray[halo_mask] - starless_bg, 0.0, None)
                source_level = float(np.mean(source_halo)) if source_halo.size else 0.0
                starless_level = float(np.mean(starless_halo)) if starless_halo.size else 0.0
                if source_level > 1e-5:
                    scores["halo_residue_score"] = _clamp_float(
                        starless_level / source_level,
                        0.0,
                        3.0,
                    )

            star_area = core_mask.astype(np.float32)
            for _ in range(3):
                star_area = _box_blur_gray(star_area)
            star_neighborhood = star_area > 0.010
            if int(np.count_nonzero(star_neighborhood)) > 16:
                dark_threshold = starless_bg - max(2.0 * starless_std, 0.015)
                dark_values = np.clip(
                    dark_threshold - starless_gray[star_neighborhood],
                    0.0,
                    None,
                )
                dark_fraction = float(np.mean(dark_values > 0.0))
                dark_depth = (
                    float(np.mean(dark_values)) / max(starless_bg, 0.03)
                    if dark_values.size
                    else 0.0
                )
                scores["black_hole_score"] = _clamp_float(
                    dark_fraction * 0.7 + dark_depth * 0.3,
                    0.0,
                    1.0,
                )
        scores["global_halo_residue_score"] = scores["halo_residue_score"]

        if starmask_data is not None:
            starmask_rgb = _to_rgb_float_image(starmask_data, max_side=1024)
            starmask_gray = (
                0.2126 * starmask_rgb[0]
                + 0.7152 * starmask_rgb[1]
                + 0.0722 * starmask_rgb[2]
            ).astype(np.float32)
            if starmask_gray.shape == source_gray.shape:
                object_threshold = max(
                    float(np.quantile(broad_source, 0.70)),
                    source_bg + max(1.8 * source_std, 0.02),
                )
                diffuse_mask = broad_source > object_threshold
                star_weight = core_mask.astype(np.float32)
                for _ in range(4):
                    star_weight = _box_blur_gray(star_weight)
                nebula_mask = diffuse_mask & (star_weight <= 0.003)
                protected_nebula_mask = (
                    nebula_mask | diffuse_nebula_protect
                    if is_protected_nebula
                    else nebula_mask
                )
                mask_signal = np.clip(starmask_gray, 0.0, None)
                total_mask_signal = float(np.sum(mask_signal))
                if total_mask_signal > 1e-7 and int(np.count_nonzero(nebula_mask)) > 16:
                    contaminated_signal = float(np.sum(mask_signal[nebula_mask]))
                    scores["starmask_contamination"] = _clamp_float(
                        contaminated_signal / total_mask_signal,
                        0.0,
                        1.0,
                    )

                compact_weight = np.zeros_like(source_gray, dtype=np.float32)
                if total_mask_signal > 1e-7:
                    mask_scale = float(np.percentile(starmask_gray, 99.7))
                    if not math.isfinite(mask_scale) or mask_scale <= 1e-7:
                        mask_scale = float(np.max(starmask_gray)) if starmask_gray.size else 0.0
                    if mask_scale > 1e-7:
                        compact_weight = np.clip(starmask_gray / mask_scale, 0.0, 1.0)
                source_detail = np.clip(source_gray - _box_blur_gray(source_gray), 0.0, None)
                detail_threshold = max(float(np.percentile(source_detail, 98.5)), source_std * 1.4, 0.004)
                detail_seed = (source_detail > detail_threshold).astype(np.float32)
                compact_weight = np.maximum(compact_weight, detail_seed * 0.65)
                for _ in range(3):
                    compact_weight = np.maximum(
                        compact_weight,
                        np.clip(_box_blur_gray(compact_weight) * 1.8, 0.0, 1.0),
                    )
                compact_weight = np.clip(
                    compact_weight * (1.0 - 0.85 * protected_nebula_mask.astype(np.float32)),
                    0.0,
                    1.0,
                )
                if float(np.sum(compact_weight)) > 1e-6:
                    starless_detail = np.clip(
                        starless_gray - _box_blur_gray(starless_gray) - max(starless_std * 1.2, 0.003),
                        0.0,
                        None,
                    )
                    source_compact_signal = np.clip(
                        source_detail - max(source_std * 0.6, 0.002),
                        0.0,
                        None,
                    )
                    residual_signal = float(np.sum(starless_detail * compact_weight))
                    reference_signal = float(np.sum(source_compact_signal * compact_weight))
                    if reference_signal > 1e-7:
                        scores["compact_residual_star_score"] = _clamp_float(
                            residual_signal / reference_signal,
                            0.0,
                            3.0,
                        )
                    scores["compact_residual_coverage"] = float(
                        np.sum((starless_detail > max(starless_std * 1.5, 0.004)).astype(np.float32) * compact_weight)
                        / max(float(np.sum(compact_weight)), 1e-6)
                    )
                star_core_weight = compact_weight.copy()
                # Keep the compact-halo diagnostic local. Repeated max-dilation
                # blankets dense star fields and turns this into a background
                # brightness ratio instead of a star-halo measurement.
                halo_weight = _box_blur_gray(
                    (star_core_weight > 0.08).astype(np.float32)
                )
                compact_halo_mask = (
                    (halo_weight > 0.02)
                    & (star_core_weight < 0.28)
                    & (~protected_nebula_mask)
                )
                if is_protected_nebula:
                    compact_halo_mask &= ~diffuse_nebula_protect
                scores["compact_halo_mask_coverage"] = float(
                    np.mean(compact_halo_mask)
                )
                if int(np.count_nonzero(compact_halo_mask)) > 16:
                    source_local = source_gray.copy()
                    starless_local = starless_gray.copy()
                    for _ in range(4):
                        source_local = _box_blur_gray(source_local)
                        starless_local = _box_blur_gray(starless_local)
                    source_compact_halo = np.clip(
                        source_gray[compact_halo_mask]
                        - source_local[compact_halo_mask],
                        0.0,
                        None,
                    )
                    starless_compact_halo = np.clip(
                        starless_gray[compact_halo_mask]
                        - starless_local[compact_halo_mask],
                        0.0,
                        None,
                    )
                    source_compact_level = float(np.mean(source_compact_halo)) if source_compact_halo.size else 0.0
                    starless_compact_level = float(np.mean(starless_compact_halo)) if starless_compact_halo.size else 0.0
                    scores["compact_halo_source_level"] = source_compact_level
                    scores["compact_halo_starless_level"] = starless_compact_level
                    if source_compact_level > 1e-5:
                        scores["compact_halo_residue_score"] = _clamp_float(
                            starless_compact_level / source_compact_level,
                            0.0,
                            3.0,
                        )
    except (TypeError, ValueError, IndexError, FloatingPointError) as e:
        pipeline.log.debug(f"stage7 artifact scoring skipped: {e}")
    return scores

def stage7_quality_assessment(
    pipeline,
    attempt_name: str,
    *,
    tool_label: str,
    use_ai: bool = True,
    source_stem: Optional[str] = None,
) -> Dict[str, Any]:
    source_stem = source_stem or pipeline.stretched_name or "stage7_stretched"
    source_data = pipeline._read_image_by_stem(source_stem)
    starless_data = pipeline._read_image_by_stem("starless")
    starmask_data = None
    if pipeline.starmask_file and pipeline.starmask_file.exists():
        starmask_data = pipeline._read_image_by_stem(pipeline.starmask_file.stem)

    source_metrics = measure_quality_metrics(source_data) if source_data is not None else QualityMetrics()
    starless_metrics = measure_quality_metrics(starless_data) if starless_data is not None else QualityMetrics()
    source_features = measure_image_features(source_data) if source_data is not None else ImageFeatures()
    starless_features = measure_image_features(starless_data) if starless_data is not None else ImageFeatures()
    starmask_metrics = (
        measure_quality_metrics(starmask_data) if starmask_data is not None else QualityMetrics()
    )

    baseline_star_energy = max(source_metrics.star_energy_ratio, 1e-5)
    baseline_star_coverage = max(source_metrics.star_coverage_ratio, 1e-5)
    global_residual_star_score = max(
        starless_metrics.star_energy_ratio / baseline_star_energy,
        starless_metrics.star_coverage_ratio / baseline_star_coverage,
    )
    if source_metrics.star_density <= 1e-7:
        global_residual_star_score = 0.0

    starmask_coverage_ratio = (
        starmask_metrics.star_coverage_ratio / baseline_star_coverage
        if starmask_data is not None
        else 0.0
    )
    starmask_width_ratio = (
        starmask_metrics.median_star_size / max(source_metrics.median_star_size, 1e-4)
        if starmask_data is not None and source_metrics.median_star_size > 0
        else 0.0
    )
    artifact_scores = pipeline._stage7_starless_artifact_scores(
        source_data,
        starless_data,
        starmask_data,
        source_features,
        starless_features,
    )
    global_halo_residue_score = artifact_scores.get(
        "global_halo_residue_score",
        artifact_scores["halo_residue_score"],
    )
    compact_halo_residue_score = artifact_scores.get("compact_halo_residue_score", 0.0)
    halo_residue_score = artifact_scores["halo_residue_score"]
    black_hole_score = artifact_scores["black_hole_score"]
    compact_residual_star_score = artifact_scores.get("compact_residual_star_score", 0.0)
    compact_residual_coverage = artifact_scores.get("compact_residual_coverage", 0.0)
    starmask_contamination = artifact_scores["starmask_contamination"]
    starless_noise_gain = artifact_scores["starless_noise_gain"]
    starless_dynamic_range_ratio = float(
        artifact_scores.get("starless_dynamic_range_ratio", 1.0) or 0.0
    )
    starless_peak_signal = float(
        artifact_scores.get("starless_peak_signal", 0.0) or 0.0
    )
    residual_coverage_score = (
        starless_metrics.star_coverage_ratio / baseline_star_coverage
        if source_metrics.star_density > 1e-7
        else 0.0
    )
    target_type = pipeline._active_target_type() if hasattr(pipeline, "_active_target_type") else ""
    if compact_residual_star_score > 0.0 and target_type == "bright_emission_reflection_nebula":
        residual_star_score = min(
            global_residual_star_score,
            max(compact_residual_star_score, residual_coverage_score * 0.75),
        )
    elif compact_residual_star_score > 0.0:
        residual_star_score = min(
            global_residual_star_score,
            max(compact_residual_star_score, residual_coverage_score * 0.85),
        )
    else:
        residual_star_score = global_residual_star_score
    if compact_halo_residue_score > 0.0 and target_type == "bright_emission_reflection_nebula":
        halo_residue_score = min(global_halo_residue_score, compact_halo_residue_score)

    issues: List[str] = []
    if residual_star_score > pipeline.cfg.stage7_residual_star_score_max:
        issues.append(
            "residual_stars "
            f"{residual_star_score:.3f}>{pipeline.cfg.stage7_residual_star_score_max:.3f}"
        )
    halo_threshold = pipeline._stage7_effective_halo_threshold()
    if halo_residue_score > halo_threshold:
        issues.append(
            "halo_residue "
            f"{halo_residue_score:.3f}>{halo_threshold:.3f}"
        )
    if compact_halo_residue_score > halo_threshold:
        issues.append(
            "compact_halo_residue "
            f"{compact_halo_residue_score:.3f}>{halo_threshold:.3f}"
        )
    if black_hole_score > pipeline.cfg.stage7_black_hole_score_max:
        issues.append(
            "black_hole "
            f"{black_hole_score:.3f}>{pipeline.cfg.stage7_black_hole_score_max:.3f}"
        )
    if (
        starmask_data is not None
        and starmask_contamination > pipeline.cfg.stage7_starmask_contamination_max
    ):
        issues.append(
            "starmask_contamination "
            f"{starmask_contamination:.3f}>{pipeline.cfg.stage7_starmask_contamination_max:.3f}"
        )
    if starless_noise_gain > pipeline.cfg.stage7_starless_noise_gain_max:
        issues.append(
            "starless_noise_gain "
            f"{starless_noise_gain:.3f}>{pipeline.cfg.stage7_starless_noise_gain_max:.3f}"
        )
    dynamic_assessment = stage7_dynamic_range_assessment(
        pipeline.cfg,
        dynamic_range_ratio=starless_dynamic_range_ratio,
        peak_signal=starless_peak_signal,
        background_level=float(starless_features.bg_median),
    )
    if dynamic_assessment["collapsed"]:
        issues.append(
            "starless_dynamic_range_collapse "
            f"{starless_dynamic_range_ratio:.3f}<"
            f"{dynamic_assessment['dynamic_range_ratio_min']:.3f}, "
            f"peak={starless_peak_signal:.5f}<"
            f"{dynamic_assessment['peak_signal_min']:.5f}, "
            f"peak/bg={dynamic_assessment['peak_background_ratio']:.2f}<"
            f"{dynamic_assessment['peak_background_ratio_min']:.2f}"
        )
    if starmask_data is None:
        issues.append("starmask_missing")
    elif starmask_coverage_ratio < pipeline.cfg.stage7_starmask_coverage_min_ratio:
        issues.append(
            "starmask_coverage_ratio "
            f"{starmask_coverage_ratio:.3f}<{pipeline.cfg.stage7_starmask_coverage_min_ratio:.3f}"
        )
    if (
        starmask_data is not None
        and source_metrics.median_star_size > 0
        and starmask_metrics.median_star_size > 0
        and starmask_width_ratio > pipeline.cfg.stage7_starmask_width_ratio_max
    ):
        issues.append(
            "starmask_width_ratio "
            f"{starmask_width_ratio:.3f}>{pipeline.cfg.stage7_starmask_width_ratio_max:.3f}"
        )

    observations = {
        "attempt": attempt_name,
        "tool_label": tool_label,
        "source_stem": source_stem,
        "source_metrics": asdict(source_metrics),
        "starless_metrics": asdict(starless_metrics),
        "starmask_metrics": asdict(starmask_metrics) if starmask_data is not None else None,
        "derived": {
            "residual_star_score": residual_star_score,
            "global_residual_star_score": global_residual_star_score,
            "compact_residual_star_score": compact_residual_star_score,
            "compact_residual_coverage": compact_residual_coverage,
            "halo_residue_score": halo_residue_score,
            "global_halo_residue_score": global_halo_residue_score,
            "compact_halo_residue_score": compact_halo_residue_score,
            "compact_halo_mask_coverage": artifact_scores.get(
                "compact_halo_mask_coverage",
                0.0,
            ),
            "compact_halo_source_level": artifact_scores.get(
                "compact_halo_source_level",
                0.0,
            ),
            "compact_halo_starless_level": artifact_scores.get(
                "compact_halo_starless_level",
                0.0,
            ),
            "black_hole_score": black_hole_score,
            "starmask_contamination": starmask_contamination,
            "starless_noise_gain": starless_noise_gain,
            "starless_dynamic_range_ratio": starless_dynamic_range_ratio,
            "source_dynamic_range": artifact_scores.get("source_dynamic_range", 0.0),
            "starless_dynamic_range": artifact_scores.get("starless_dynamic_range", 0.0),
            "source_peak_signal": artifact_scores.get("source_peak_signal", 0.0),
            "starless_peak_signal": starless_peak_signal,
            "starless_background_level": float(starless_features.bg_median),
            "starless_peak_background_ratio": dynamic_assessment[
                "peak_background_ratio"
            ],
            "dynamic_range_collapse": dynamic_assessment["collapsed"],
            "starmask_coverage_ratio": starmask_coverage_ratio,
            "starmask_width_ratio": starmask_width_ratio,
            "halo_threshold": halo_threshold,
        },
        "local_issues": issues,
    }
    ai_assessment = pipeline._request_stage7_quality_ai(observations) if use_ai else None
    ai_issues: List[str] = []
    if ai_assessment:
        if ai_assessment["verdict"] != "ok":
            ai_issues.append(f"ai_verdict={ai_assessment['verdict']}")
        if ai_assessment["residual_stars"]:
            ai_issues.append("ai_residual_stars")
        if ai_assessment["starmask_missing"]:
            ai_issues.append("ai_starmask_missing")
        if ai_assessment["starmask_too_wide"]:
            ai_issues.append("ai_starmask_too_wide")

    all_issues = issues + ai_issues
    return {
        "attempt": attempt_name,
        "tool_label": tool_label,
        "source_stem": source_stem,
        "status": "ok" if not all_issues else "poor",
        "issues": all_issues,
        "local_issues": issues,
        "ai_assessment": ai_assessment,
        "source_metrics": asdict(source_metrics),
        "starless_metrics": asdict(starless_metrics),
        "starmask_metrics": asdict(starmask_metrics) if starmask_data is not None else None,
        "derived": observations["derived"],
    }

def stage7_quality_score(pipeline, quality: Optional[Dict[str, Any]]) -> float:
    if not quality:
        return 1_000_000.0
    derived = quality.get("derived")
    if not isinstance(derived, dict):
        return 1_000_000.0
    residual = float(derived.get("residual_star_score", 1_000.0))
    halo = float(derived.get("halo_residue_score", 0.0))
    compact_halo = float(derived.get("compact_halo_residue_score", 0.0))
    black_hole = float(derived.get("black_hole_score", 0.0))
    contamination = float(derived.get("starmask_contamination", 0.0))
    noise_gain = float(derived.get("starless_noise_gain", 1.0))
    dynamic_range_ratio = float(derived.get("starless_dynamic_range_ratio", 1.0))
    peak_signal = float(derived.get("starless_peak_signal", 1.0))
    coverage_ratio = float(derived.get("starmask_coverage_ratio", 0.0))
    width_ratio = float(derived.get("starmask_width_ratio", 1.0))
    coverage_penalty = max(0.0, pipeline.cfg.stage7_starmask_coverage_min_ratio - coverage_ratio)
    width_penalty = max(0.0, width_ratio - pipeline.cfg.stage7_starmask_width_ratio_max)
    halo_penalty = max(0.0, halo - pipeline._stage7_effective_halo_threshold())
    compact_halo_penalty = max(
        0.0,
        compact_halo - pipeline._stage7_effective_halo_threshold(),
    )
    black_hole_penalty = max(0.0, black_hole - pipeline.cfg.stage7_black_hole_score_max)
    contamination_penalty = max(
        0.0,
        contamination - pipeline.cfg.stage7_starmask_contamination_max,
    )
    noise_penalty = max(0.0, noise_gain - pipeline.cfg.stage7_starless_noise_gain_max)
    dynamic_penalty = 0.0
    dynamic_threshold = float(
        getattr(pipeline.cfg, "stage7_starless_dynamic_range_min_ratio", 0.55)
    )
    collapse = derived.get("dynamic_range_collapse")
    if collapse is None:
        peak_threshold = float(
            getattr(pipeline.cfg, "stage7_starless_peak_signal_min", 0.006)
        )
        collapse = dynamic_range_ratio < dynamic_threshold and peak_signal < peak_threshold
    if bool(collapse):
        dynamic_penalty = (dynamic_threshold - dynamic_range_ratio) * 2.0
    return (
        residual
        + coverage_penalty * 2.0
        + width_penalty
        + halo_penalty
        + compact_halo_penalty
        + black_hole_penalty * 2.0
        + contamination_penalty
        + noise_penalty
        + dynamic_penalty
    )

def stage7_repair_triggers(pipeline, quality: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(quality, dict):
        return []
    derived = quality.get("derived")
    if not isinstance(derived, dict):
        return []
    triggers: List[str] = []
    try:
        residual = float(derived.get("residual_star_score", 0.0))
    except (TypeError, ValueError):
        residual = 0.0
    try:
        black_hole = float(derived.get("black_hole_score", 0.0))
    except (TypeError, ValueError):
        black_hole = 0.0
    try:
        halo = float(derived.get("halo_residue_score", 0.0))
    except (TypeError, ValueError):
        halo = 0.0
    try:
        compact_halo = float(derived.get("compact_halo_residue_score", 0.0))
    except (TypeError, ValueError):
        compact_halo = 0.0
    try:
        dynamic_range_ratio = float(derived.get("starless_dynamic_range_ratio", 1.0))
    except (TypeError, ValueError):
        dynamic_range_ratio = 1.0
    try:
        peak_signal = float(derived.get("starless_peak_signal", 1.0))
    except (TypeError, ValueError):
        peak_signal = 1.0
    if residual > float(pipeline.cfg.stage7_residual_star_score_max):
        triggers.append("residual_stars")
    if halo > float(pipeline._stage7_effective_halo_threshold()):
        triggers.append("halo_residue")
    if compact_halo > float(pipeline._stage7_effective_halo_threshold()):
        triggers.append("compact_halo_residue")
    if black_hole > float(pipeline.cfg.stage7_black_hole_score_max):
        triggers.append("black_hole")
    collapse = derived.get("dynamic_range_collapse")
    if collapse is None:
        collapse = (
            dynamic_range_ratio
            < float(getattr(pipeline.cfg, "stage7_starless_dynamic_range_min_ratio", 0.55))
            and peak_signal
            < float(getattr(pipeline.cfg, "stage7_starless_peak_signal_min", 0.006))
        )
    if bool(collapse):
        triggers.append("dynamic_range_collapse")
    return triggers

def stage7_update_star_remix_from_quality(
    pipeline,
    quality: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    pipeline._stage7_selected_quality = quality
    pipeline._stage7_residual_star_score = 0.0
    pipeline._stage9_star_intensity_scale = 1.0
    pipeline._stage9_star_intensity_reason = ""

    derived = quality.get("derived") if isinstance(quality, dict) else None
    residual_score = 0.0
    if isinstance(derived, dict):
        try:
            residual_score = float(derived.get("residual_star_score", 0.0))
        except (TypeError, ValueError):
            residual_score = 0.0
        try:
            halo_score = float(derived.get("halo_residue_score", 0.0))
        except (TypeError, ValueError):
            halo_score = 0.0
        try:
            contamination_score = float(derived.get("starmask_contamination", 0.0))
        except (TypeError, ValueError):
            contamination_score = 0.0
    else:
        halo_score = 0.0
        contamination_score = 0.0
    pipeline._stage7_residual_star_score = max(0.0, residual_score)

    threshold = max(float(pipeline.cfg.stage7_residual_star_score_max), 1e-4)
    halo_threshold = max(float(pipeline._stage7_effective_halo_threshold()), 1e-4)
    contamination_threshold = max(float(pipeline.cfg.stage7_starmask_contamination_max), 1e-4)
    issues = [str(item).lower() for item in (quality or {}).get("issues", [])]
    residual_flagged = (
        pipeline._stage7_residual_star_score > threshold
        or any("residual_star" in item or "residual_stars" in item for item in issues)
    )
    halo_flagged = (
        halo_score > halo_threshold
        or any("halo_residue" in item for item in issues)
    )
    contamination_flagged = (
        contamination_score > contamination_threshold
        or any("starmask_contamination" in item for item in issues)
    )
    scale_candidates: List[Tuple[float, str]] = []
    if residual_flagged:
        ai_assessment = (quality or {}).get("ai_assessment")
        ai_scale: Optional[float] = None
        if isinstance(ai_assessment, dict):
            value = ai_assessment.get("stage9_star_intensity_scale")
            if value is not None:
                try:
                    ai_scale = _clamp_float(value, 0.35, 1.0)
                except (TypeError, ValueError):
                    ai_scale = None
        if pipeline._stage7_residual_star_score > threshold:
            over_ratio = pipeline._stage7_residual_star_score / threshold
            scale = 1.0 - 0.36 * max(0.0, over_ratio - 1.0)
            reason = (
                "stage7 residual stars high "
                f"({pipeline._stage7_residual_star_score:.3f}>{threshold:.3f})"
            )
        else:
            scale = 0.85
            reason = "stage7 AI residual-star diagnostic"
        if ai_scale is not None:
            scale = min(scale, ai_scale)
            reason += f"; AI scale={ai_scale:.3f}"
        scale_candidates.append((_clamp_float(scale, 0.35, 0.95), reason))

    if halo_flagged:
        over_ratio = halo_score / halo_threshold if halo_score > 0 else 1.0
        halo_scale = 1.0 - 0.16 * max(0.0, over_ratio - 1.0)
        scale_candidates.append(
            (
                _clamp_float(halo_scale, 0.55, 0.92),
                f"stage7 halo residue high ({halo_score:.3f}>{halo_threshold:.3f})",
            )
        )

    if contamination_flagged:
        over_ratio = contamination_score / contamination_threshold if contamination_score > 0 else 1.0
        contamination_scale = 1.0 - 0.28 * max(0.0, over_ratio - 1.0)
        scale_candidates.append(
            (
                _clamp_float(contamination_scale, 0.45, 0.88),
                (
                    "stage7 starmask contamination high "
                    f"({contamination_score:.3f}>{contamination_threshold:.3f})"
                ),
            )
        )

    if scale_candidates:
        scale, reason = min(scale_candidates, key=lambda item: item[0])
        pipeline._stage9_star_intensity_scale = scale
        pipeline._stage9_star_intensity_reason = reason

    return {
        "residual_star_score": pipeline._stage7_residual_star_score,
        "threshold": threshold,
        "halo_residue_score": halo_score,
        "halo_threshold": halo_threshold,
        "starmask_contamination": contamination_score,
        "starmask_contamination_threshold": contamination_threshold,
        "intensity_scale": pipeline._stage9_star_intensity_scale,
        "reason": pipeline._stage9_star_intensity_reason,
    }

def stage7_residual_suppression_strength(
    pipeline,
    quality: Optional[Dict[str, Any]],
) -> float:
    ai_assessment = (quality or {}).get("ai_assessment")
    if isinstance(ai_assessment, dict):
        value = ai_assessment.get("residual_suppression_strength")
        if value is not None:
            try:
                return _clamp_float(value, 0.0, 0.25)
            except (TypeError, ValueError):
                pass
    derived = (quality or {}).get("derived")
    residual_score = 0.0
    if isinstance(derived, dict):
        try:
            residual_score = float(derived.get("residual_star_score", 0.0))
        except (TypeError, ValueError):
            residual_score = 0.0
    threshold = max(float(pipeline.cfg.stage7_residual_star_score_max), 1e-4)
    if residual_score <= threshold:
        return 0.0
    over_ratio = residual_score / threshold
    return _clamp_float(0.06 + 0.08 * max(0.0, over_ratio - 1.0), 0.04, 0.18)
