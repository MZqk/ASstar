"""Service mixins for SeestarPostProcessor."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import ai_advisory
import cosmic_clarity
import plugin_runner
import sasp_runner
import scunet_denoise
import syqon_starless
import stage7_quality
import stage7_repair
import stage7_stretch_metrics
import stage8_pixels
from image_metrics import (
    _box_blur_gray,
    _clamp_float,
    _to_rgb_float_fullres,
    format_feature_summary,
    measure_quality_metrics,
)
from models import ImageFeatures, QualityMetrics, Stage6StretchStrategy, StageResult, TargetType
from save_utils import save_stage_output, write_ai_raw_response, write_stage_json

try:
    from sirilpy.exceptions import CommandError, DataError, SirilError
except ImportError:
    CommandError = RuntimeError
    DataError = RuntimeError
    SirilError = RuntimeError

try:
    from image_feature_analyzer import analyze_image as analyze_adaptive_image
    from policy_selector import DEFAULT_POLICY, policy_for_profile
    from stretch_candidate_evaluator import (
        build_candidate_spec,
        candidate_modes,
        choose_best as choose_best_stretch_candidate,
        score_candidate as score_stretch_candidate,
    )
    from target_profiler import build_target_profile
except (ImportError, RuntimeError):
    analyze_adaptive_image = None
    DEFAULT_POLICY = {
        "policy_name": "generic_low_snr_safe",
        "stage6_stretch": {"fallback_candidate": "asinh_core_protect"},
    }
    policy_for_profile = None
    build_candidate_spec = None
    candidate_modes = None
    choose_best_stretch_candidate = None
    score_stretch_candidate = None
    build_target_profile = None

ENV_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
ENV_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
ENV_DEBUG_MODE_KEY = "SEESTAR_DEBUG_MODE"
ENV_INPUT_MODE_KEY = "SEESTAR_INPUT_MODE"
INPUT_MODE_AUTO = "auto"
INPUT_MODE_LINEAR_RESUME = "result_linear_resume"
RESULT_BASENAME_TEMPLATE = (
    "$OBJECT:%s$_$STACKCNT:%d$x$EXPTIME:%d$sec"
    "_$DATE-OBS:dm12$_processed"
)
STAGE7_ASINH_STRETCH_MIN = 1.0
STAGE7_ASINH_STRETCH_MAX = 1000.0


def _stage7_expanded_star_halo_protection(
    starmask: Optional[np.ndarray],
    target_shape: Tuple[int, int],
    *,
    expand_iterations: int,
) -> Optional[np.ndarray]:
    """Build a soft full-resolution protection mask around Stage 6 stars."""
    if starmask is None:
        return None
    try:
        mask_rgb = _to_rgb_float_fullres(np.asarray(starmask))
        if tuple(mask_rgb.shape[1:]) != tuple(target_shape):
            return None
        mask_gray = (
            0.2126 * mask_rgb[0]
            + 0.7152 * mask_rgb[1]
            + 0.0722 * mask_rgb[2]
        ).astype(np.float32)
        mask_floor = float(np.quantile(mask_gray, 0.50))
        mask_signal = np.clip(mask_gray - mask_floor, 0.0, None)
        positive = mask_signal[mask_signal > 0.0]
        if positive.size < 8:
            return None
        mask_scale = float(np.quantile(positive, 0.995))
        if not math.isfinite(mask_scale) or mask_scale <= 1e-7:
            return None
        mask_weight = np.clip(mask_signal / mask_scale, 0.0, 1.0)
        seed = mask_weight >= 0.10
        if int(np.count_nonzero(seed)) < 4:
            seed_threshold = float(
                np.quantile(mask_weight[mask_weight > 0.0], 0.75)
            )
            seed = mask_weight >= max(seed_threshold, 1e-4)

        expanded = seed.astype(np.float32)
        for _ in range(max(1, min(int(expand_iterations), 8))):
            expanded = _box_blur_gray(expanded)
        return np.maximum(
            mask_weight,
            np.clip(expanded * 4.0, 0.0, 1.0),
        ).astype(np.float32)
    except (IndexError, TypeError, ValueError, FloatingPointError):
        return None


def _stage7_asinh_sample(value: float, stretch: float, offset: float) -> float:
    """Approximate Siril's scalar Asinh transform for parameter calibration."""
    value = float(value)
    stretch = max(STAGE7_ASINH_STRETCH_MIN, float(stretch))
    offset = max(0.0, float(offset))
    if not all(math.isfinite(item) for item in (value, stretch, offset)):
        return 0.0
    if value <= 0.0 or value <= offset:
        return 0.0
    denominator = value * math.asinh(stretch)
    if denominator <= 0.0:
        return 0.0
    transformed = (
        (value - offset) * math.asinh(value * stretch) / denominator
    )
    return _clamp_float(transformed, 0.0, 1.0)


def _stage7_solve_asinh_stretch(
    value: float,
    offset: float,
    target: float,
    stretch_max: float = STAGE7_ASINH_STRETCH_MAX,
) -> float:
    """Find a bounded Siril Asinh strength that reaches a scalar target."""
    target = _clamp_float(target, 0.0, 1.0)
    low = STAGE7_ASINH_STRETCH_MIN
    high = _clamp_float(
        stretch_max,
        STAGE7_ASINH_STRETCH_MIN,
        STAGE7_ASINH_STRETCH_MAX,
    )
    if _stage7_asinh_sample(value, low, offset) >= target:
        return low
    if _stage7_asinh_sample(value, high, offset) <= target:
        return high
    for _ in range(48):
        middle = (low + high) * 0.5
        if _stage7_asinh_sample(value, middle, offset) < target:
            low = middle
        else:
            high = middle
    return high


def _stage7_linked_mtf_midtones(source_value: float, target_value: float) -> float:
    """Solve a linked MTF midpoint for one normalized source level."""
    source = _clamp_float(float(source_value), 1e-6, 0.999999)
    target = _clamp_float(float(target_value), 1e-6, 0.999999)
    denominator = source * (1.0 - 2.0 * target) + target
    if not math.isfinite(denominator) or abs(denominator) < 1e-12:
        raise ValueError("cannot solve a stable Stage7 linked MTF midpoint")
    midpoint = source * (1.0 - target) / denominator
    if not math.isfinite(midpoint) or not 0.0 < midpoint < 1.0:
        raise ValueError("Stage7 linked MTF midpoint is outside (0,1)")
    return midpoint


def _stage7_mtf_sample(
    value: float,
    shadows: float,
    midtones: float,
    highlights: float = 1.0,
) -> float:
    """Evaluate Siril's MTF mapping for calibration diagnostics."""
    value = float(value)
    shadows = float(shadows)
    midtones = float(midtones)
    highlights = float(highlights)
    if not all(math.isfinite(item) for item in (value, shadows, midtones, highlights)):
        return 0.0
    if value <= shadows:
        return 0.0
    if value >= highlights:
        return 1.0
    normalized = (value - shadows) / max(highlights - shadows, 1e-12)
    denominator = (2.0 * midtones - 1.0) * normalized - midtones
    if abs(denominator) < 1e-12:
        return 0.0
    mapped = (midtones - 1.0) * normalized / denominator
    return _clamp_float(mapped, 0.0, 1.0)


def _stage7_preview_calibrated_stretch(
    baseline_stats: Dict[str, Any],
    preview_stats: Dict[str, Any],
    *,
    offset: float,
    preview_scale: float,
    fallback: float,
    highlight_scale: float = 0.90,
    stretch_max: float = STAGE7_ASINH_STRETCH_MAX,
) -> Tuple[float, Dict[str, Any]]:
    """Use linked-autostretch distribution as a conservative Asinh ruler."""
    try:
        baseline_p50 = float(baseline_stats.get("p50", 0.0) or 0.0)
        baseline_p99 = float(baseline_stats.get("p99", 0.0) or 0.0)
        preview_p50 = float(preview_stats.get("p50", 0.0) or 0.0)
        preview_p99 = float(preview_stats.get("p99", 0.0) or 0.0)
    except (TypeError, ValueError):
        baseline_p50 = baseline_p99 = preview_p50 = preview_p99 = 0.0
    if (
        not all(
            math.isfinite(item)
            for item in (baseline_p50, baseline_p99, preview_p50, preview_p99)
        )
        or baseline_p50 <= max(float(offset), 0.0)
        or baseline_p99 <= baseline_p50
        or preview_p50 <= 0.0
        or preview_p99 <= preview_p50
    ):
        return float(fallback), {}

    target_p50 = _clamp_float(preview_p50 * preview_scale, 0.025, 0.120)
    target_p99 = _clamp_float(
        preview_p99 * _clamp_float(highlight_scale, 0.75, 0.95),
        0.20,
        0.85,
    )
    median_limited = _stage7_solve_asinh_stretch(
        baseline_p50,
        offset,
        target_p50,
        stretch_max,
    )
    highlight_limited = _stage7_solve_asinh_stretch(
        baseline_p99,
        offset,
        target_p99,
        stretch_max,
    )
    calibrated = _clamp_float(
        min(median_limited, highlight_limited),
        STAGE7_ASINH_STRETCH_MIN,
        stretch_max,
    )
    return calibrated, {
        "baseline_p50": baseline_p50,
        "baseline_p99": baseline_p99,
        "preview_p50": preview_p50,
        "preview_p99": preview_p99,
        "target_p50": target_p50,
        "target_p99": target_p99,
        "median_limited_stretch": median_limited,
        "highlight_limited_stretch": highlight_limited,
        "calibrated_stretch": calibrated,
        "stretch_max": float(stretch_max),
        "predicted_p50": _stage7_asinh_sample(baseline_p50, calibrated, offset),
        "predicted_p99": _stage7_asinh_sample(baseline_p99, calibrated, offset),
    }


def _stage7_preview_target_attainment(
    candidate_name: str,
    pixel_stats: Dict[str, Any],
    adaptation: Dict[str, Any],
    *,
    min_ratio: float = 0.55,
    max_ratio: float = 1.50,
) -> Dict[str, Any]:
    """Keep calibrated candidates within the allowed preview P50 target band."""
    preview_calibration = adaptation.get("preview_calibration") or {}
    calibration_key = {
        "cand_a": "candidate_a",
        "cand_b": "candidate_b",
    }.get(str(candidate_name))
    calibration = (
        preview_calibration.get(calibration_key) if calibration_key else None
    )
    if not isinstance(calibration, dict) or not calibration:
        return {
            "status": "unavailable",
            "accepted": True,
            "issues": [],
        }

    try:
        actual_p50 = float(pixel_stats.get("p50", 0.0) or 0.0)
        target_p50 = float(calibration.get("target_p50", 0.0) or 0.0)
        calibrated_stretch = float(
            calibration.get("calibrated_stretch", 0.0) or 0.0
        )
        stretch_max = float(calibration.get("stretch_max", 0.0) or 0.0)
        predicted_p50 = float(calibration.get("predicted_p50", 0.0) or 0.0)
    except (TypeError, ValueError):
        return {
            "status": "unavailable",
            "accepted": True,
            "issues": [],
        }
    if (
        not all(
            math.isfinite(value)
            for value in (
                actual_p50,
                target_p50,
                calibrated_stretch,
                stretch_max,
                predicted_p50,
            )
        )
        or actual_p50 <= 0.0
        or target_p50 <= 0.0
    ):
        return {
            "status": "unavailable",
            "accepted": True,
            "issues": [],
        }

    minimum = _clamp_float(min_ratio, 0.25, 0.90)
    maximum = _clamp_float(max_ratio, 1.00, 3.00)
    ratio = actual_p50 / target_p50
    saturated = bool(
        stretch_max > 0.0
        and calibrated_stretch >= stretch_max * 0.999
        and predicted_p50 < target_p50
    )
    issues: List[str] = []
    if ratio < minimum:
        issues.append(
            "preview_target_p50_ratio "
            f"{ratio:.3f}<{minimum:.3f} "
            f"(actual={actual_p50:.5f}, target={target_p50:.5f}, "
            f"stretch_saturated={str(saturated).lower()})"
        )
    if ratio > maximum:
        issues.append(
            "preview_target_p50_ratio_above_max "
            f"{ratio:.3f}>{maximum:.3f} "
            f"(actual={actual_p50:.5f}, target={target_p50:.5f})"
        )
    accepted = not issues
    return {
        "status": "ok" if accepted else "poor",
        "accepted": accepted,
        "candidate": str(candidate_name),
        "actual_p50": actual_p50,
        "target_p50": target_p50,
        "attainment_ratio": ratio,
        "minimum_ratio": minimum,
        "maximum_ratio": maximum,
        "calibrated_stretch": calibrated_stretch,
        "stretch_max": stretch_max,
        "stretch_saturated": saturated,
        "issues": issues,
    }


def stage6_effective_bg_median_min(configured_min: float) -> float:
    """Return the Stage6 dark-floor gate with room for FITS sampling noise."""
    configured = max(float(configured_min), 1e-4)
    tolerance = min(0.0005, configured * 0.025)
    return max(1e-4, configured - tolerance)

class Stage6ServiceMixin:
    def _ai_stage_advisory_enabled(self, attr_name: str) -> bool:
        if not ai_advisory.network_mode_enabled():
            return False
        if not bool(getattr(self.cfg, "ai_post_enabled", False)):
            return False
        if not bool(getattr(self.cfg, attr_name, True)):
            return False
        return bool(
            self.cfg.ai_endpoint.strip()
            and self.cfg.ai_model.strip()
            and self.cfg.ai_api_key.strip()
        )


    def _extract_stage_advisory_from_text(
        self,
        stage_name: str,
        text: str,
    ) -> Optional[Dict[str, Any]]:
        return ai_advisory.extract_stage_advisory_from_text(self, stage_name, text)


    def _request_stage_ai_advisory(
        self,
        stage_name: str,
        schema_text: str,
        observations: Dict[str, Any],
        *,
        max_tokens: int = 700,
        image_paths: Optional[List[Tuple[str, Path]]] = None,
        allow_text_fallback: bool = True,
    ) -> Dict[str, Any]:
        return ai_advisory.request_stage_ai_advisory(
            self,
            stage_name,
            schema_text,
            observations,
            max_tokens=max_tokens,
            image_paths=image_paths,
            allow_text_fallback=allow_text_fallback,
        )


    def _apply_adaptive_edge_crop(
        self,
        feat: ImageFeatures,
        *,
        trigger: Optional[float] = None,
        target: Optional[float] = None,
        max_extra_margin: Optional[float] = None,
    ) -> Optional[str]:
        trigger = 0.14 if trigger is None else float(trigger)
        target = 0.10 if target is None else float(target)
        max_extra_margin = 0.015 if max_extra_margin is None else float(max_extra_margin)
        if feat.edge_black_ratio < trigger:
            return None
        if self.cfg.crop_margin >= 0.055:
            return (
                "adaptive edge crop skipped "
                f"(edge_black={feat.edge_black_ratio:.3f}, crop_margin already high)"
            )
        shape = self.siril.get_image_shape()
        if not shape:
            return (
                "adaptive edge crop skipped "
                f"(edge_black={feat.edge_black_ratio:.3f}, image shape unavailable)"
            )
        _channels, height, width = shape
        edge_excess = max(0.0, feat.edge_black_ratio - target)
        extra_margin_ratio = min(
            max_extra_margin,
            max(0.006, 0.006 + edge_excess * 0.16),
        )
        margin_x = int(width * extra_margin_ratio)
        margin_y = int(height * extra_margin_ratio)
        crop_w = width - 2 * margin_x
        crop_h = height - 2 * margin_y
        if margin_x <= 0 or margin_y <= 0 or crop_w <= 0 or crop_h <= 0:
            return (
                "adaptive edge crop skipped "
                f"(edge_black={feat.edge_black_ratio:.3f}, image too small)"
            )
        self.cmd_with_check("crop", str(margin_x), str(margin_y), str(crop_w), str(crop_h))
        return (
            "adaptive edge crop applied "
            f"(edge_black={feat.edge_black_ratio:.3f}, extra_margin={extra_margin_ratio:.3f}, "
            f"target={target:.3f}, pixels={margin_x}x{margin_y})"
        )


    def _apply_weak_object_tuning(self) -> Optional[str]:
        feat = getattr(self.auto_tune_result, "features", None)
        if feat is None:
            return None
        if not (feat.object_area_ratio < 0.01 and feat.diffuse_ratio < 0.05):
            return None
        stretch_before = float(self.cfg.asinh_stretch)
        saturation_before = float(self.cfg.nebula_saturation)
        stretch_bump = 0.25 if feat.core_brightness_ratio > 0.12 else 0.35
        self.cfg.asinh_stretch = _clamp_float(stretch_before + stretch_bump, 1.6, 3.6)
        self.cfg.nebula_saturation = _clamp_float(saturation_before + 0.06, 0.0, 0.65)
        return (
            "weak-object tuning applied "
            f"(object_area={feat.object_area_ratio:.3f}, diffuse={feat.diffuse_ratio:.3f}, "
            f"asinh={stretch_before:.2f}->{self.cfg.asinh_stretch:.2f}, "
            f"nebula_sat={saturation_before:.2f}->{self.cfg.nebula_saturation:.2f})"
        )


    def _stage6_target_hint_lower(self) -> str:
        parts: List[str] = []
        for value in (self.source_file, self.work_dir):
            if isinstance(value, Path):
                parts.extend([value.name, value.stem])
        return " ".join(parts).lower()


    def _stage6_stretch_candidate(
        self,
        method: str,
        *,
        source: str = "strategy",
        asinh_stretch: Optional[float] = None,
        asinh_offset: Optional[float] = None,
        ghs_shadowsclip: Optional[float] = None,
        ghs_stretchamount: Optional[float] = None,
        summary: str = "",
    ) -> Dict[str, Any]:
        return {
            "source": source,
            "method": method,
            "summary": summary,
            "params": {
                "asinh_stretch": _clamp_float(
                    self.cfg.asinh_stretch if asinh_stretch is None else asinh_stretch,
                    1.6,
                    3.6,
                ),
                "asinh_offset": _clamp_float(
                    self.cfg.asinh_offset if asinh_offset is None else asinh_offset,
                    0.0005,
                    0.006,
                ),
                "ghs_shadowsclip": _clamp_float(
                    self.cfg.ghs_shadowsclip if ghs_shadowsclip is None else ghs_shadowsclip,
                    -3.6,
                    -1.8,
                ),
                "ghs_stretchamount": _clamp_float(
                    self.cfg.ghs_stretchamount if ghs_stretchamount is None else ghs_stretchamount,
                    1.0,
                    2.8,
                ),
            },
        }


    def _stage6_strategy_from_features(
        self,
        baseline_features: Optional[ImageFeatures],
        baseline_quality: Optional[QualityMetrics],
    ) -> Stage6StretchStrategy:
        feat = baseline_features or getattr(self.auto_tune_result, "features", None)
        target_type = getattr(
            getattr(self, "auto_tune_result", None),
            "target_type",
            TargetType.UNKNOWN,
        )
        hint = self._stage6_target_hint_lower()
        quality_bg_std = float(getattr(baseline_quality, "black_pixel_ratio", 0.0) or 0.0)

        if feat is None:
            return Stage6StretchStrategy(
                "default_asinh_then_ghs",
                "feature unavailable: Asinh first, GHS fallback",
                [
                    self._stage6_stretch_candidate("asinh"),
                    self._stage6_stretch_candidate("ghs"),
                    self._stage6_stretch_candidate("autostretch"),
                ],
                use_curves=True,
                curves_label="default curves",
            )

        noisy_background = (
            feat.bg_std > 0.055
            or feat.bg_std / max(feat.bg_median, 1e-4) > 0.55
            or feat.edge_black_ratio > 0.18
            or quality_bg_std > 0.20
        )
        high_dynamic_nebula = (
            target_type in {TargetType.EMISSION_NEBULA, TargetType.REFLECTION_NEBULA}
            and feat.diffuse_ratio > 0.24
            and feat.core_brightness_ratio > 0.055
        ) or any(token in hint for token in ("m42", "orion", "great_orion"))
        dark_weak_nebula = (
            target_type in {TargetType.EMISSION_NEBULA, TargetType.REFLECTION_NEBULA}
            and feat.object_area_ratio < 0.18
            and feat.bg_median < 0.11
        )
        bright_core_dark_outer = (
            feat.core_brightness_ratio > 0.075
            and feat.object_area_ratio < 0.36
            and feat.bg_median < 0.13
        )

        cautious_ghs_amount = min(float(self.cfg.ghs_stretchamount), 1.45)
        protected_ghs_amount = min(float(self.cfg.ghs_stretchamount), 1.65)

        if target_type == TargetType.CLUSTER:
            return Stage6StretchStrategy(
                "cluster_asinh",
                "star cluster: Asinh only, avoid GHS star bloat",
                [
                    self._stage6_stretch_candidate("asinh"),
                    self._stage6_stretch_candidate("autostretch"),
                ],
                use_curves=False,
            )

        if noisy_background:
            return Stage6StretchStrategy(
                "noisy_background_asinh_cautious_ghs",
                "dirty background/low SNR: prefer Asinh, keep GHS conservative",
                [
                    self._stage6_stretch_candidate("asinh"),
                    self._stage6_stretch_candidate(
                        "asinh_ghs",
                        ghs_stretchamount=cautious_ghs_amount,
                        summary="cautious GHS after Asinh",
                    ),
                    self._stage6_stretch_candidate(
                        "ghs",
                        ghs_stretchamount=cautious_ghs_amount,
                    ),
                    self._stage6_stretch_candidate("autostretch"),
                ],
                use_curves=False,
                protection_note="GHS guarded because background/noise metrics are high",
            )

        if high_dynamic_nebula:
            return Stage6StretchStrategy(
                "high_dynamic_nebula_asinh_ghs_curves",
                "high dynamic nebula: Asinh plus GHS/curves",
                [
                    self._stage6_stretch_candidate("asinh_ghs"),
                    self._stage6_stretch_candidate("asinh"),
                    self._stage6_stretch_candidate("ghs"),
                    self._stage6_stretch_candidate("autostretch"),
                ],
                use_curves=True,
                curves_label="high dynamic nebula curves",
            )

        if dark_weak_nebula:
            return Stage6StretchStrategy(
                "dark_weak_nebula_asinh_ghs_guarded",
                "dark/weak nebula: Asinh plus guarded GHS with noise protection",
                [
                    self._stage6_stretch_candidate(
                        "asinh_ghs",
                        ghs_stretchamount=protected_ghs_amount,
                        summary="guarded GHS after Asinh",
                    ),
                    self._stage6_stretch_candidate("asinh"),
                    self._stage6_stretch_candidate(
                        "ghs",
                        ghs_stretchamount=protected_ghs_amount,
                    ),
                    self._stage6_stretch_candidate("autostretch"),
                ],
                use_curves=False,
                protection_note="weak nebula protected with conservative GHS and quality gate",
            )

        if bright_core_dark_outer:
            return Stage6StretchStrategy(
                "bright_core_dark_outer_asinh_ghs",
                "bright core/dark outer halo: Asinh plus GHS",
                [
                    self._stage6_stretch_candidate("asinh_ghs"),
                    self._stage6_stretch_candidate("asinh"),
                    self._stage6_stretch_candidate("ghs"),
                    self._stage6_stretch_candidate("autostretch"),
                ],
                use_curves=True,
                curves_label="core/outer balance curves",
            )

        if target_type == TargetType.GALAXY:
            return Stage6StretchStrategy(
                "galaxy_asinh_light_curves",
                "ordinary galaxy: Asinh plus light curves",
                [
                    self._stage6_stretch_candidate("asinh"),
                    self._stage6_stretch_candidate("ghs", ghs_stretchamount=cautious_ghs_amount),
                    self._stage6_stretch_candidate("autostretch"),
                ],
                use_curves=True,
                curves_label="light galaxy curves",
            )

        return Stage6StretchStrategy(
            "default_asinh_then_ghs",
            "default: Asinh first, GHS fallback",
            [
                self._stage6_stretch_candidate("asinh"),
                self._stage6_stretch_candidate("ghs"),
                self._stage6_stretch_candidate("autostretch"),
            ],
            use_curves=True,
            curves_label="default curves",
        )


    def _normalize_stage6_ai_plan(self, obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return ai_advisory.normalize_stage6_ai_plan(self, obj)


    def _request_stage6_stretch_plan(
        self,
        baseline_features: Optional[ImageFeatures],
        baseline_quality: Optional[QualityMetrics],
    ) -> Optional[Dict[str, Any]]:
        return ai_advisory.request_stage6_stretch_plan(
            self,
            baseline_features,
            baseline_quality,
        )


    def _normalize_stage7_stretch_selection(
        self,
        obj: Dict[str, Any],
        allowed_candidate_ids: List[str],
    ) -> Optional[Dict[str, Any]]:
        return ai_advisory.normalize_stage7_stretch_selection(
            self,
            obj,
            allowed_candidate_ids,
        )


    def _request_stage7_stretch_selection(
        self,
        accepted_attempts: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        return ai_advisory.request_stage7_stretch_selection(
            self,
            accepted_attempts,
        )


    def _stage6_candidate_specs(
        self,
        ai_plan: Optional[Dict[str, Any]],
        strategy: Stage6StretchStrategy,
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        strategy_methods = {str(item.get("method")) for item in strategy.candidates}
        if ai_plan:
            ai_method = str(ai_plan["method"])
            if ai_method in strategy_methods:
                candidates.append(
                    {
                        "source": "ai",
                        "method": ai_method,
                        "params": ai_plan["params"],
                        "summary": ai_plan.get("summary", ""),
                    }
                )
            else:
                self.log.warn(
                    f"[AI] stage6 plan method {ai_method} skipped by local strategy "
                    f"{strategy.name}"
                )
        if self.cfg.workflow_plugin_probe_enabled:
            candidates.append({"source": "plugin", "method": "plugin", "params": {}})

        candidates.extend(strategy.candidates)

        deduped: List[Dict[str, Any]] = []
        seen = set()
        for candidate in candidates:
            params = candidate.get("params", {})
            key = (
                candidate.get("source"),
                candidate.get("method"),
                tuple(sorted((str(k), round(float(v), 6)) for k, v in params.items())),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        return deduped


    def _stage6_candidate_label(self, candidate: Dict[str, Any]) -> str:
        method = candidate.get("method")
        source = candidate.get("source", "default")
        params = candidate.get("params", {})
        if method == "asinh":
            return (
                f"{source}: Asinh "
                f"({params.get('asinh_stretch', self.cfg.asinh_stretch):.3f}, "
                f"{params.get('asinh_offset', self.cfg.asinh_offset):.5f})"
            )
        if method == "asinh_ghs":
            return (
                f"{source}: Asinh+GHS "
                f"(asinh={params.get('asinh_stretch', self.cfg.asinh_stretch):.3f}, "
                f"offset={params.get('asinh_offset', self.cfg.asinh_offset):.5f}, "
                f"ghs={params.get('ghs_stretchamount', self.cfg.ghs_stretchamount):.3f})"
            )
        if method == "linked_mtf":
            shadows = float(params.get("mtf_shadows", 0.0) or 0.0)
            variant = "noise-floor" if shadows > 1e-9 else "zero-shadow"
            return (
                f"{source}: {variant} linked MTF "
                f"(s={shadows:.6f}, "
                f"m={params.get('mtf_midtones', 0.5):.6f}, "
                f"target={params.get('target_background', 0.0):.5f})"
            )
        if method == "ghs":
            return (
                f"{source}: GHS "
                f"({params.get('ghs_shadowsclip', self.cfg.ghs_shadowsclip):.3f}, "
                f"{params.get('ghs_stretchamount', self.cfg.ghs_stretchamount):.3f})"
            )
        if method == "plugin":
            return "plugin: VeraLux HyperMetric Stretch"
        if method == "bright_nebula_hdr_masked":
            source = candidate.get("source", "policy")
            params = candidate.get("params", {})
            return (
                f"{source}: Bright nebula HDR masked "
                f"(asinh={params.get('asinh_stretch', self.cfg.asinh_stretch):.3f}, "
                f"pedestal={params.get('bg_pedestal', 0.0):.4f})"
            )
        return f"{source}: autostretch"


    def _apply_stage6_bright_nebula_hdr_masked(
        self,
        params: Dict[str, Any],
        starmask_image_data: Optional[np.ndarray] = None,
    ) -> None:
        image_data = self.siril.get_image_pixeldata(preview=False)
        if image_data is None:
            raise RuntimeError("image buffer is empty")
        source = np.asarray(image_data)
        rgb = _to_rgb_float_fullres(source)
        gray = (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]).astype(np.float32)
        finite = gray[np.isfinite(gray)]
        if finite.size == 0:
            raise RuntimeError("image buffer has no finite pixels")

        p50, p82, p96, p985, p997 = np.percentile(
            finite,
            [50.0, 82.0, 96.0, 98.5, 99.7],
        )
        bg_target = _clamp_float(float(params.get("bg_pedestal", 0.024)), 0.012, 0.045)
        faint_boost = _clamp_float(float(params.get("faint_boost", 0.018)), 0.0, 0.045)
        core_protection = _clamp_float(float(params.get("core_protection", 0.72)), 0.0, 0.95)
        shadow_chroma_damping = _clamp_float(
            float(params.get("shadow_chroma_damping", 0.28)), 0.0, 0.60
        )
        faint_saturation_boost = _clamp_float(
            float(params.get("faint_saturation_boost", 0.026)), 0.0, 0.08
        )
        star_mask_expand = int(
            _clamp_float(float(params.get("star_mask_expand", 4)), 1.0, 8.0)
        )
        star_faint_suppression = _clamp_float(
            float(params.get("star_faint_suppression", 0.85)), 0.0, 1.0
        )
        star_detail_suppression = _clamp_float(
            float(params.get("star_detail_suppression", 0.18)), 0.0, 0.60
        )
        star_protection = _stage7_expanded_star_halo_protection(
            starmask_image_data,
            tuple(gray.shape),
            expand_iterations=star_mask_expand,
        )

        shadow_anchor = max(float(p82), bg_target * 2.6, float(p50) + 0.012)
        shadow_weight = np.clip(1.0 - gray / max(shadow_anchor, 1e-4), 0.0, 1.0) ** 1.35
        pedestal = max(0.0, bg_target - float(p50))

        faint_low = max(float(p50), bg_target * 0.60)
        faint_high = max(float(p96), faint_low + 0.025)
        faint_mask = np.clip((gray - faint_low) / max(faint_high - faint_low, 1e-4), 0.0, 1.0)
        faint_mask = _box_blur_gray(faint_mask.astype(np.float32))
        core_floor = max(float(p985), faint_high + 0.030, 0.55)
        core_top = max(float(p997), core_floor + 0.040)
        core_mask = np.clip((gray - core_floor) / max(core_top - core_floor, 1e-4), 0.0, 1.0)
        core_mask = _box_blur_gray(core_mask.astype(np.float32))

        result = rgb.copy()
        pedestal_weight = np.clip(0.70 + 0.30 * shadow_weight - 0.80 * core_mask, 0.0, 1.0)
        result += pedestal * pedestal_weight[None, :, :]
        faint_lift_weight = faint_mask * (1.0 - 0.85 * core_mask)
        if star_protection is not None:
            faint_lift_weight *= np.clip(
                1.0 - star_faint_suppression * star_protection,
                0.0,
                1.0,
            )
        result += faint_boost * faint_lift_weight[None, :, :]

        new_gray = (0.2126 * result[0] + 0.7152 * result[1] + 0.0722 * result[2]).astype(np.float32)
        bg_chroma_weight = np.clip(
            shadow_weight * (1.0 - 0.65 * faint_mask) * (1.0 - core_mask),
            0.0,
            1.0,
        )
        chroma_damp = 1.0 - shadow_chroma_damping * bg_chroma_weight
        for idx in range(3):
            result[idx] = new_gray + (result[idx] - new_gray) * chroma_damp

        new_gray = (0.2126 * result[0] + 0.7152 * result[1] + 0.0722 * result[2]).astype(np.float32)
        sat_weight = np.clip(faint_mask * (1.0 - 0.90 * shadow_weight) * (1.0 - core_mask), 0.0, 1.0)
        if star_protection is not None:
            sat_weight *= np.clip(
                1.0 - star_faint_suppression * star_protection,
                0.0,
                1.0,
            )
        sat_gain = 1.0 + faint_saturation_boost * sat_weight
        for idx in range(3):
            result[idx] = new_gray + (result[idx] - new_gray) * sat_gain

        new_gray = (0.2126 * result[0] + 0.7152 * result[1] + 0.0722 * result[2]).astype(np.float32)
        highlight = np.maximum(new_gray - core_floor, 0.0)
        compressed_gray = core_floor + highlight / (
            1.0 + core_protection * highlight / max(1.0 - core_floor, 1e-3)
        )
        core_scale = np.divide(
            compressed_gray,
            np.maximum(new_gray, 1e-6),
            out=np.ones_like(new_gray, dtype=np.float32),
            where=new_gray > 1e-6,
        )
        compressed = result * core_scale[None, :, :]
        result = result * (1.0 - core_mask[None, :, :]) + compressed * core_mask[None, :, :]

        if star_protection is not None and star_detail_suppression > 0.0:
            # The global Asinh can make residual star footprints cross the
            # component-size threshold even when rank structure is stable.
            # Attenuate only positive local detail inside the expanded Stage 6
            # star/halo support, preserving colour ratios and diffuse nebulosity.
            protected_gray = (
                0.2126 * result[0]
                + 0.7152 * result[1]
                + 0.0722 * result[2]
            ).astype(np.float32)
            local_floor = protected_gray
            for _ in range(3):
                local_floor = _box_blur_gray(local_floor)
            positive_detail = np.maximum(protected_gray - local_floor, 0.0)
            detail_reduction = (
                star_detail_suppression * star_protection * positive_detail
            )
            detail_scale = np.divide(
                np.maximum(protected_gray - detail_reduction, 0.0),
                np.maximum(protected_gray, 1e-6),
                out=np.ones_like(protected_gray, dtype=np.float32),
                where=protected_gray > 1e-6,
            )
            result *= detail_scale[None, :, :]

        restored = self._stage8_restore_rgb_like(source, np.clip(result, 0.0, 0.995))
        self._set_current_image_pixeldata(restored, label="stage6 bright-nebula HDR masked")


    def _execute_stage6_stretch_candidate(
        self,
        candidate: Dict[str, Any],
        *,
        starmask_image_data: Optional[np.ndarray] = None,
    ) -> Tuple[bool, str]:
        method = candidate.get("method")
        params = candidate.get("params", {})
        try:
            if method == "plugin":
                used = self._run_first_available_command(
                    "拉伸",
                    [
                        ("VeraLux HyperMetric Stretch", ("veralux_hypermetric_stretch",)),
                        ("HyperMetricStretch", ("hypermetric_stretch",)),
                    ],
                )
                if used:
                    return True, used
                return False, "stretch plugin unavailable"
            if method == "asinh":
                self.cmd_with_check(
                    "asinh",
                    str(params.get("asinh_stretch", self.cfg.asinh_stretch)),
                    str(params.get("asinh_offset", self.cfg.asinh_offset)),
                )
                return True, "Asinh"
            if method == "asinh_ghs":
                self.cmd_with_check(
                    "asinh",
                    str(params.get("asinh_stretch", self.cfg.asinh_stretch)),
                    str(params.get("asinh_offset", self.cfg.asinh_offset)),
                )
                self.cmd_with_check(
                    "autoghs", "-linked",
                    str(params.get("ghs_shadowsclip", self.cfg.ghs_shadowsclip)),
                    str(params.get("ghs_stretchamount", self.cfg.ghs_stretchamount)),
                )
                return True, "Asinh+GHS"
            if method == "linked_mtf":
                shadows = float(params.get("mtf_shadows", 0.0))
                midtones = float(params.get("mtf_midtones", 0.5))
                highlights = float(params.get("mtf_highlights", 1.0))
                if not (
                    0.0 <= shadows < highlights <= 1.0
                    and 0.0 < midtones < 1.0
                ):
                    raise ValueError(
                        "invalid Stage7 linked MTF parameters: "
                        f"s={shadows}, m={midtones}, h={highlights}"
                    )
                self.cmd_with_check(
                    "mtf",
                    f"{shadows:.6f}",
                    f"{midtones:.6f}",
                    f"{highlights:.6f}",
                )
                variant = "noise-floor" if shadows > 1e-9 else "zero-shadow"
                return True, f"{variant} linked MTF"
            if method == "bright_nebula_hdr_masked":
                self.cmd_with_check(
                    "asinh",
                    str(params.get("asinh_stretch", self.cfg.asinh_stretch)),
                    str(params.get("asinh_offset", self.cfg.asinh_offset)),
                )
                self._apply_stage6_bright_nebula_hdr_masked(
                    params,
                    starmask_image_data,
                )
                return True, "bright_nebula_hdr_masked"
            if method == "ghs":
                self.cmd_with_check(
                    "autoghs", "-linked",
                    str(params.get("ghs_shadowsclip", self.cfg.ghs_shadowsclip)),
                    str(params.get("ghs_stretchamount", self.cfg.ghs_stretchamount)),
                )
                return True, "GHS"
            if method == "adaptive_quantile":
                image_data = self.siril.get_image_pixeldata(preview=False)
                if image_data is None:
                    raise RuntimeError("image buffer is empty")
                source = np.asarray(image_data)
                mapped_rgb = (
                    stage7_stretch_metrics.apply_adaptive_quantile_stretch(
                        source,
                        dict(params.get("calibration") or {}),
                    )
                )
                restored = self._stage8_restore_rgb_like(source, mapped_rgb)
                self._set_current_image_pixeldata(
                    restored,
                    label="Stage7 adaptive quantile fallback",
                )
                return True, "adaptive_quantile"
            if method == "autostretch":
                self.cmd_with_check("autostretch")
                return True, "autostretch"
            return False, f"unsupported stage6 method: {method}"
        except (CommandError, SirilError, RuntimeError, ValueError) as e:
            return False, self._short_text(e, 180)


    def _stage7_feedback_retry_candidate(
        self,
        candidate: Dict[str, Any],
        attempt: Dict[str, Any],
        retry_index: int,
    ) -> Optional[Dict[str, Any]]:
        """Build one same-baseline retry from measured post-transform P50."""
        try:
            retry_max = int(
                getattr(self.cfg, "stage7_stretch_feedback_retry_max", 1)
            )
        except (TypeError, ValueError):
            retry_max = 1
        retry_max = max(0, min(retry_max, 1))
        if retry_index < 1 or retry_index > retry_max:
            return None
        if attempt.get("status") != "ok" or not attempt.get("stem"):
            return None

        diagnostics = [
            str(item).strip()
            for item in (attempt.get("diagnostics") or [])
            if str(item).strip()
        ]
        feedback_prefixes = (
            "preview_target_p50_ratio ",
            "preview_target_p50_ratio_above_max ",
        )
        if not diagnostics or not all(
            item.startswith(feedback_prefixes) for item in diagnostics
        ):
            return None

        attainment = attempt.get("preview_target_attainment") or {}
        try:
            ratio = float(attainment.get("attainment_ratio", 0.0) or 0.0)
            minimum = float(attainment.get("minimum_ratio", 0.0) or 0.0)
            maximum = float(attainment.get("maximum_ratio", 0.0) or 0.0)
        except (TypeError, ValueError):
            return None
        if (
            not all(math.isfinite(value) for value in (ratio, minimum, maximum))
            or ratio <= 0.0
            or minimum <= 0.0
            or maximum <= minimum
            or minimum <= ratio <= maximum
        ):
            return None

        adjusted = copy.deepcopy(candidate)
        root_name = str(
            adjusted.get("calibration_candidate")
            or (adjusted.get("feedback") or {}).get("root_name")
            or adjusted.get("name")
            or "candidate"
        )
        adjusted["calibration_candidate"] = root_name
        adjusted["name"] = f"{root_name}_feedback_{retry_index}"
        adjusted["stem"] = f"stage7_{root_name}_feedback_{retry_index}"
        adjusted["explicit_fallback"] = True

        method = str(adjusted.get("method") or "")
        params = dict(adjusted.get("params") or {})
        feedback: Dict[str, Any] = {
            "mode": "post_transform_p50_calibration",
            "root_name": root_name,
            "retry_index": retry_index,
            "source_attempt": str(attempt.get("name") or ""),
            "measured_ratio": ratio,
            "measured_post_transform_p50": attainment.get("actual_p50"),
            "target_p50": attainment.get("target_p50"),
            "minimum_ratio": minimum,
            "maximum_ratio": maximum,
        }

        try:
            current_stretch = float(params.get("asinh_stretch", 0.0) or 0.0)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(current_stretch) or current_stretch <= 0.0:
            return None

        # Move just inside the unchanged acceptance band.  For cand_b this
        # correction is based on the P50 measured after both Asinh and GHS;
        # GHS remains enabled and the immutable linear source is rerun once.
        target_ratio = (
            max(minimum, maximum * 0.98)
            if ratio > maximum
            else min(maximum, minimum * 1.02)
        )
        correction = _clamp_float(target_ratio / ratio, 0.35, 1.80)
        next_stretch = _clamp_float(
            current_stretch * correction,
            STAGE7_ASINH_STRETCH_MIN,
            STAGE7_ASINH_STRETCH_MAX,
        )
        if abs(next_stretch - current_stretch) < 1e-6:
            return None
        params["asinh_stretch"] = round(next_stretch, 3)
        feedback.update(
            {
                "adjustment": "scale_asinh_from_post_transform_p50",
                "adjusted_parameter": "asinh_stretch",
                "previous_asinh_stretch": current_stretch,
                "next_asinh_stretch": next_stretch,
                "correction_factor": correction,
                "calibration_target_ratio": target_ratio,
                "ghs_retained": method == "asinh_ghs",
                "ghs_stretchamount": (
                    params.get("ghs_stretchamount")
                    if method == "asinh_ghs"
                    else None
                ),
                "reason": (
                    "parameter correction uses P50 measured after the complete "
                    "transform; the candidate is rerun once from the linear source"
                ),
            }
        )

        adjusted["params"] = params
        adjusted["feedback"] = feedback
        return adjusted


    def _stage7_evaluate_stretch_candidate(
        self,
        candidate: Dict[str, Any],
        *,
        source_stem: str,
        baseline_quality: Optional[QualityMetrics],
        baseline_image_data: Optional[np.ndarray],
        starmask_image_data: Optional[np.ndarray],
        baseline_background_quality: Dict[str, Any],
        frozen_background_masks: Optional[Dict[str, Any]],
        target_stretch: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run one Stage 7 candidate from the immutable source and evaluate all gates."""
        name = str(candidate.get("name") or "candidate")
        stem = str(candidate.get("stem") or f"stage7_{name}")
        common = {
            "name": name,
            "file": None,
            "stem": None,
            "method": candidate.get("method"),
            "params": dict(candidate.get("params") or {}),
            "adaptation": candidate.get("adaptation"),
            "explicit_fallback": bool(candidate.get("explicit_fallback")),
            "feedback": candidate.get("feedback"),
            "calibration_candidate": candidate.get("calibration_candidate"),
        }
        try:
            self.cmd_with_check("load", source_stem)
        except (CommandError, SirilError) as error:
            return {
                **common,
                "status": "failed",
                "reason": self._short_text(error, 160),
            }

        ok, used_or_error = self._execute_stage6_stretch_candidate(
            candidate,
            starmask_image_data=starmask_image_data,
        )
        if not ok:
            return {
                **common,
                "status": "failed",
                "reason": used_or_error,
            }

        local_quality: Dict[str, Any] = {
            "status": "unavailable",
            "accepted": True,
            "issues": [],
            "risk_score": 0.0,
            "metrics": {},
        }
        starless_structure_quality: Dict[str, Any] = {
            "status": "unavailable",
            "accepted": True,
            "issues": [],
            "risk_score": 0.0,
            "metrics": {},
            "reason": "Starless structure inputs unavailable",
        }
        candidate_image_data: Optional[np.ndarray] = None
        try:
            candidate_data = self.siril.get_image_pixeldata(preview=False)
            if candidate_data is not None:
                candidate_image_data = np.asarray(candidate_data)
                if baseline_image_data is not None:
                    local_quality = stage7_stretch_metrics.assess_target_local_stretch(
                        baseline_image_data,
                        candidate_image_data,
                        str(
                            target_stretch.get("target_type")
                            or "generic_low_snr_safe"
                        ),
                        self.cfg,
                    )
                    if source_stem == "stage6_starless":
                        starless_structure_quality = (
                            stage7_stretch_metrics.assess_starless_structure_growth(
                                baseline_image_data,
                                candidate_image_data,
                                starmask_image_data,
                                self.cfg,
                            )
                        )
        except (
            CommandError,
            DataError,
            SirilError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            candidate_image_data = None

        use_starless_structure_gate = (
            starless_structure_quality.get("status") in {"ok", "rejected"}
        )
        enforce_star_growth = self._stage7_should_enforce_star_growth(
            use_starless_structure_gate
        )
        quality_ok, issues, metrics = self._validate_stage6_stretch_quality(
            baseline_quality,
            enforce_star_growth=enforce_star_growth,
        )
        pixel_stats = self._current_pixel_distribution_stats()
        invalid_reasons = [
            reason
            for key, reason in (
                ("is_nearly_black", "nearly_black"),
                ("is_visibility_too_low", "visibility_too_low"),
                ("is_nearly_white", "nearly_white"),
                ("invalid_dynamic_range", "invalid_dynamic_range"),
            )
            if pixel_stats.get(key)
        ]
        if invalid_reasons:
            quality_ok = False
            issues = [*issues, *invalid_reasons]
        visibility_gate = self._stage7_candidate_visibility_gate(
            pixel_stats,
            target_stretch,
        )
        if not bool(visibility_gate.get("accepted", True)):
            quality_ok = False
            issues = [
                *issues,
                *list(visibility_gate.get("issues") or []),
            ]
        calibration_name = str(
            candidate.get("calibration_candidate") or candidate.get("name") or ""
        )
        preview_target_attainment = _stage7_preview_target_attainment(
            calibration_name,
            pixel_stats,
            dict(candidate.get("adaptation") or {}),
            min_ratio=getattr(
                self.cfg,
                "stage7_preview_target_p50_min_ratio",
                0.55,
            ),
            max_ratio=getattr(
                self.cfg,
                "stage7_preview_target_p50_max_ratio",
                1.50,
            ),
        )
        if not bool(preview_target_attainment.get("accepted", True)):
            quality_ok = False
            issues = [
                *issues,
                *list(preview_target_attainment.get("issues") or []),
            ]

        candidate_background_masks = frozen_background_masks
        if (
            candidate_image_data is not None
            and isinstance(frozen_background_masks, dict)
            and frozen_background_masks.get("background_mask") is not None
            and hasattr(self, "_stage8_generate_starless_masks")
        ):
            try:
                candidate_signal_masks = self._stage8_generate_starless_masks(
                    candidate_image_data
                )
                candidate_background_masks = dict(frozen_background_masks)
                for mask_name in (
                    "core_mask",
                    "nebula_mask",
                    "faint_nebula_mask",
                ):
                    signal_mask = candidate_signal_masks.get(mask_name)
                    if signal_mask is not None:
                        candidate_background_masks[mask_name] = signal_mask
                candidate_signal_exclusion_applied = any(
                    candidate_background_masks.get(mask_name) is not None
                    for mask_name in (
                        "core_mask",
                        "nebula_mask",
                        "faint_nebula_mask",
                    )
                )
            except (
                IndexError,
                RuntimeError,
                TypeError,
                ValueError,
                FloatingPointError,
            ):
                candidate_background_masks = frozen_background_masks
                candidate_signal_exclusion_applied = False
        else:
            candidate_signal_exclusion_applied = False

        candidate_background_quality = (
            self._background_quality_metrics(
                candidate_image_data,
                candidate_background_masks,
            )
            if candidate_image_data is not None
            else {}
        )
        if candidate_background_quality:
            candidate_background_quality["signal_exclusion_applied"] = bool(
                candidate_signal_exclusion_applied
            )
        background_quality_gate = self._stage7_stretch_background_gate(
            baseline_background_quality,
            candidate_background_quality,
        )
        if not bool(background_quality_gate.get("accepted", False)):
            quality_ok = False
            issues = [
                *issues,
                *list(background_quality_gate.get("issues") or []),
            ]
        if not bool(local_quality.get("accepted", True)):
            quality_ok = False
            issues = [*issues, *list(local_quality.get("issues") or [])]
        if not bool(starless_structure_quality.get("accepted", True)):
            quality_ok = False
            issues = [
                *issues,
                *list(starless_structure_quality.get("issues") or []),
            ]
        risk_score = self._stage6_stretch_risk_score(
            metrics,
            issues,
            baseline_quality,
            enforce_star_growth=enforce_star_growth,
        )
        risk_score += float(local_quality.get("risk_score", 0.0) or 0.0)
        risk_score += float(
            starless_structure_quality.get("risk_score", 0.0) or 0.0
        )
        candidate_saved = self._save_stage_output(stem)
        return {
            **common,
            "file": f"{stem}.fit" if candidate_saved else None,
            "stem": stem if candidate_saved else None,
            "status": "ok" if candidate_saved else "failed",
            "used": used_or_error,
            "quality_ok": quality_ok,
            "diagnostics": issues,
            "metrics": asdict(metrics) if metrics else None,
            "adaptive_metrics": (
                self._adaptive_features_current()
                if hasattr(self, "_adaptive_features_current")
                else None
            ),
            "pixel_stats": pixel_stats,
            "visibility_gate": visibility_gate,
            "preview_target_attainment": preview_target_attainment,
            "target_local_quality": local_quality,
            "starless_structure_quality": starless_structure_quality,
            "background_quality_gate": background_quality_gate,
            "risk_score": risk_score,
            "allowed_as_final": bool(quality_ok),
        }


    def _stage7_candidate_visibility_gate(
        self,
        pixel_stats: Dict[str, Any],
        target_stretch: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Require material diffuse visibility for diffuse-target final candidates."""
        profile_name = str(target_stretch.get("name") or "")
        applies = profile_name in {
            "widefield_nebulosity",
            "dark_nebula_separation",
        }
        minimum = float(
            getattr(self.cfg, "stage7_diffuse_visibility_score_min", 0.08)
        )
        score = float(
            pixel_stats.get("safe_preview_visibility_score", 0.0) or 0.0
        )
        accepted = not applies or (
            math.isfinite(score) and score >= minimum
        )
        return {
            "status": "ok" if accepted else "poor",
            "accepted": accepted,
            "applies": applies,
            "issues": (
                []
                if accepted
                else [f"diffuse_visibility_score {score:.3f}<{minimum:.3f}"]
            ),
            "metrics": {
                "safe_preview_visibility_score": score,
                "minimum": minimum,
                "profile": profile_name,
            },
        }


    def _stage7_effective_star_growth_ratio_max(self) -> float:
        """Allow bright-nebula starless structure without weakening other targets."""
        generic_limit = float(self.cfg.stage6_star_growth_ratio_max)
        target_type = (
            str(self._active_target_type() or "").strip().lower()
            if hasattr(self, "_active_target_type")
            else ""
        )
        if target_type != "bright_emission_reflection_nebula":
            return generic_limit
        return max(
            generic_limit,
            float(
                getattr(
                    self.cfg,
                    "stage7_bright_nebula_star_growth_ratio_max",
                    generic_limit,
                )
            ),
        )


    def _stage7_should_enforce_star_growth(
        self,
        use_starless_structure_gate: bool,
    ) -> bool:
        """Keep the 1.50x bright-nebula size gate alongside rank diagnostics."""
        if not use_starless_structure_gate:
            return True
        target_type = (
            str(self._active_target_type() or "").strip().lower()
            if hasattr(self, "_active_target_type")
            else ""
        )
        return target_type == "bright_emission_reflection_nebula"


    def _validate_stage6_stretch_quality(
        self,
        baseline_quality: Optional[QualityMetrics],
        *,
        enforce_star_growth: bool = True,
    ) -> Tuple[bool, List[str], Optional[QualityMetrics]]:
        metrics = self._measure_current_quality()
        if metrics is None:
            return True, ["quality sampling unavailable"], None

        issues: List[str] = []
        bg_median_min = stage6_effective_bg_median_min(self.cfg.stage6_bg_median_min)
        if metrics.bg_median < bg_median_min:
            issues.append(
                f"bg_median {metrics.bg_median:.4f}<{self.cfg.stage6_bg_median_min:.4f}"
            )
        if metrics.black_pixel_ratio > self.cfg.stage6_black_pixel_ratio_max:
            issues.append(
                "black_pixel_ratio "
                f"{metrics.black_pixel_ratio:.3f}>{self.cfg.stage6_black_pixel_ratio_max:.3f}"
            )
        if metrics.highlight_clip_ratio > self.cfg.stage6_highlight_clip_ratio_max:
            issues.append(
                "highlight_clip_ratio "
                f"{metrics.highlight_clip_ratio:.4f}>{self.cfg.stage6_highlight_clip_ratio_max:.4f}"
            )
        if (
            enforce_star_growth
            and baseline_quality
            and baseline_quality.median_star_size > 0.2
            and metrics.median_star_size > 0
        ):
            star_growth = metrics.median_star_size / max(baseline_quality.median_star_size, 1e-4)
            star_growth_limit = self._stage7_effective_star_growth_ratio_max()
            if star_growth > star_growth_limit:
                issues.append(
                    f"star_size_growth {star_growth:.3f}>{star_growth_limit:.3f}"
                )
        return len(issues) == 0, issues, metrics


    @staticmethod
    def _stage7_background_chroma_load(metrics: Dict[str, Any]) -> float:
        """Return background chroma deviation relative to the background level."""
        direct_load = metrics.get("background_chroma_load")
        if direct_load is not None:
            try:
                value = float(direct_load)
                if math.isfinite(value):
                    return max(value, 0.0)
            except (TypeError, ValueError):
                pass
        chroma_score = max(float(metrics.get("chroma_noise_score", 0.0) or 0.0), 0.0)
        bg_std = max(float(metrics.get("bg_std", 0.0) or 0.0), 0.0)
        bg_median = max(float(metrics.get("bg_median", 0.0) or 0.0), 1e-4)
        mean_chroma = chroma_score * max(2.0 * bg_std, 0.01)
        return mean_chroma / bg_median


    def _stage7_stretch_background_gate(
        self,
        baseline: Dict[str, Any],
        candidate: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Reject stretched candidates with unsafe chroma or background mottling."""
        limits = {
            "chroma_noise_score_max": float(
                getattr(self.cfg, "stage7_stretch_chroma_noise_score_max", 0.34)
            ),
            "background_mottling_score_max": float(
                getattr(self.cfg, "stage7_stretch_background_mottling_score_max", 0.45)
            ),
            "chroma_load_growth_max": float(
                getattr(self.cfg, "stage7_stretch_chroma_load_growth_max", 1.35)
            ),
            "chroma_load_low_absolute_max": float(
                getattr(self.cfg, "stage7_stretch_chroma_load_low_absolute_max", 0.05)
            ),
            "chroma_load_low_absolute_tolerance": float(
                getattr(
                    self.cfg,
                    "stage7_stretch_chroma_load_low_absolute_tolerance",
                    0.0005,
                )
            ),
            "chroma_load_signal_excluded_max": float(
                getattr(
                    self.cfg,
                    "stage7_stretch_chroma_load_signal_excluded_max",
                    0.06,
                )
            ),
        }
        issues: List[str] = []
        if not candidate:
            return {
                "accepted": False,
                "status": "unavailable",
                "issues": ["background_quality_metrics_unavailable"],
                "limits": limits,
                "metrics": {},
            }

        chroma = max(float(candidate.get("chroma_noise_score", 0.0) or 0.0), 0.0)
        mottling = max(
            float(candidate.get("background_mottling_score", 0.0) or 0.0),
            0.0,
        )
        candidate_load = self._stage7_background_chroma_load(candidate)
        baseline_load = self._stage7_background_chroma_load(baseline) if baseline else 0.0
        load_growth = candidate_load / max(baseline_load, 1e-6) if baseline_load > 0.0 else None
        baseline_bg_median = 0.0
        if baseline:
            baseline_bg_median = max(
                float(baseline.get("bg_median", 0.0) or 0.0),
                0.0,
            )
        extreme_low_background = bool(
            baseline and 0.0 < baseline_bg_median <= 0.005
        )
        low_absolute_tolerance = max(
            limits["chroma_load_low_absolute_tolerance"],
            0.0,
        )
        low_absolute_effective_max = (
            limits["chroma_load_low_absolute_max"] + low_absolute_tolerance
        )
        limits["chroma_load_low_absolute_effective_max"] = (
            low_absolute_effective_max
        )
        low_absolute_load_exempted = bool(
            extreme_low_background
            and load_growth is not None
            and load_growth > limits["chroma_load_growth_max"]
            and candidate_load <= low_absolute_effective_max
        )
        signal_exclusion_applied = bool(
            candidate.get("signal_exclusion_applied", False)
        )
        signal_excluded_load_exempted = bool(
            signal_exclusion_applied
            and load_growth is not None
            and load_growth > limits["chroma_load_growth_max"]
            and candidate_load <= limits["chroma_load_signal_excluded_max"]
            and chroma <= limits["chroma_noise_score_max"]
            and mottling <= limits["background_mottling_score_max"]
        )
        load_growth_exempted = bool(
            low_absolute_load_exempted or signal_excluded_load_exempted
        )
        effective_load_limit = (
            limits["chroma_load_signal_excluded_max"]
            if signal_excluded_load_exempted
            else low_absolute_effective_max
        )

        if chroma > limits["chroma_noise_score_max"]:
            issues.append(
                "background_chroma_noise_score "
                f"{chroma:.3f}>{limits['chroma_noise_score_max']:.3f}"
            )
        if mottling > limits["background_mottling_score_max"]:
            issues.append(
                "background_mottling_score "
                f"{mottling:.3f}>{limits['background_mottling_score_max']:.3f}"
            )
        if (
            load_growth is not None
            and load_growth > limits["chroma_load_growth_max"]
            and not load_growth_exempted
        ):
            issues.append(
                "background_chroma_load_growth "
                f"{load_growth:.3f}>{limits['chroma_load_growth_max']:.3f}"
            )

        return {
            "accepted": not issues,
            "status": "ok" if not issues else "poor",
            "issues": issues,
            "limits": limits,
            "metrics": {
                "chroma_noise_score": chroma,
                "background_mottling_score": mottling,
                "chroma_load": candidate_load,
                "baseline_chroma_load": baseline_load if baseline else None,
                "chroma_load_growth": load_growth,
                "chroma_load_growth_low_absolute_exempted": load_growth_exempted,
                "chroma_load_growth_extreme_low_exempted": low_absolute_load_exempted,
                "chroma_load_growth_signal_excluded_exempted": signal_excluded_load_exempted,
                "signal_exclusion_applied": signal_exclusion_applied,
                "chroma_load_low_absolute_effective_max": (
                    effective_load_limit
                ),
                "chroma_load_low_absolute_tolerance": low_absolute_tolerance,
                "extreme_low_background": extreme_low_background,
                "bg_median": candidate.get("bg_median"),
                "bg_std": candidate.get("bg_std"),
            },
        }


    def _stage7_chroma_rescue_strengths(self) -> List[float]:
        raw_levels = getattr(
            self.cfg,
            "stage7_chroma_rescue_strength_levels",
            (0.10, 0.20, 0.35),
        )
        if not isinstance(raw_levels, (list, tuple)):
            raw_levels = (raw_levels,)
        levels: List[float] = []
        for raw_value in raw_levels:
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            value = _clamp_float(value, 0.10, 0.75)
            if not any(abs(value - existing) < 1e-6 for existing in levels):
                levels.append(value)
        return sorted(levels) or [0.10, 0.20, 0.35]


    @staticmethod
    def _stage7_candidate_selection_key(attempt: Dict[str, Any]) -> Tuple[Any, ...]:
        """Rank saved candidates without allowing a failed hard gate to become final."""
        status_penalty = 0 if attempt.get("status") == "ok" and attempt.get("stem") else 1
        final_penalty = 0 if bool(attempt.get("allowed_as_final", False)) else 1
        diagnostics = [
            str(item).strip()
            for item in (attempt.get("diagnostics") or [])
            if str(item).strip()
        ]
        soft_prefixes = (
            "background_chroma_noise_score",
            "background_chroma_load_growth",
        )
        hard_issue_count = sum(
            1 for item in diagnostics if not item.startswith(soft_prefixes)
        )

        normalized_excess = 0.0
        normalized_quality_load = 0.0
        background_gate = attempt.get("background_quality_gate") or {}
        gate_metrics = background_gate.get("metrics") or {}
        gate_limits = background_gate.get("limits") or {}
        for metric_name, limit_name in (
            ("chroma_noise_score", "chroma_noise_score_max"),
            ("background_mottling_score", "background_mottling_score_max"),
            ("chroma_load_growth", "chroma_load_growth_max"),
        ):
            try:
                if (
                    metric_name == "chroma_load_growth"
                    and bool(
                        gate_metrics.get(
                            "chroma_load_growth_low_absolute_exempted"
                        )
                    )
                ):
                    metric_value = float(
                        gate_metrics.get("chroma_load", 0.0) or 0.0
                    )
                    limit_value = float(
                        gate_limits.get(
                            "chroma_load_low_absolute_effective_max",
                            gate_limits.get("chroma_load_low_absolute_max", 0.0),
                        )
                        or 0.0
                    )
                else:
                    metric_value = float(
                        gate_metrics.get(metric_name, 0.0) or 0.0
                    )
                    limit_value = float(
                        gate_limits.get(limit_name, 0.0) or 0.0
                    )
            except (TypeError, ValueError):
                continue
            if (
                math.isfinite(metric_value)
                and math.isfinite(limit_value)
                and limit_value > 0.0
            ):
                normalized_value = max(0.0, metric_value / limit_value)
                normalized_quality_load += normalized_value
                normalized_excess += max(0.0, normalized_value - 1.0)

        target_attainment = attempt.get("preview_target_attainment") or {}
        try:
            attainment_ratio = float(
                target_attainment.get("attainment_ratio", 1.0) or 0.0
            )
            minimum_ratio = float(
                target_attainment.get("minimum_ratio", 0.0) or 0.0
            )
            maximum_ratio = float(
                target_attainment.get("maximum_ratio", 0.0) or 0.0
            )
        except (TypeError, ValueError):
            attainment_ratio = 1.0
            minimum_ratio = 0.0
            maximum_ratio = 0.0
        if minimum_ratio > 0.0 and math.isfinite(attainment_ratio):
            normalized_excess += max(
                0.0,
                (minimum_ratio - attainment_ratio) / minimum_ratio,
            )
        if maximum_ratio > 0.0 and math.isfinite(attainment_ratio):
            normalized_excess += max(
                0.0,
                (attainment_ratio - maximum_ratio) / maximum_ratio,
            )
        brightness_distance = (
            abs(math.log(attainment_ratio))
            if math.isfinite(attainment_ratio) and attainment_ratio > 0.0
            else math.inf
        )

        try:
            risk_score = float(attempt.get("risk_score", 1_000_000.0))
        except (TypeError, ValueError):
            risk_score = 1_000_000.0
        if not math.isfinite(risk_score):
            risk_score = 1_000_000.0
        return (
            status_penalty,
            final_penalty,
            hard_issue_count,
            normalized_excess,
            len(diagnostics),
            normalized_quality_load,
            brightness_distance,
            risk_score,
            str(attempt.get("name") or ""),
        )


    @staticmethod
    def _stage7_validated_fallback_reason(attempt: Dict[str, Any]) -> str:
        """Return the reason code for a selected fallback that passed all gates."""
        if not bool(attempt.get("explicit_fallback")):
            return ""
        method = str(attempt.get("method") or "")
        feedback_mode = str((attempt.get("feedback") or {}).get("mode") or "")
        if method == "background_chroma_rescue":
            return "validated_chroma_rescue"
        if feedback_mode in {
            "closed_loop_brightness",
            "post_transform_p50_calibration",
        }:
            return "validated_brightness_feedback"
        if (
            method == "adaptive_quantile"
            or feedback_mode == "adaptive_quantile_fallback"
        ):
            return "validated_quantile_fallback"
        return "validated_stretch_fallback"


    @staticmethod
    def _stage7_review_candidate_is_safe(attempt: Dict[str, Any]) -> bool:
        """Allow review delivery only for structurally safe brightness/chroma rejects."""
        if attempt.get("status") != "ok" or not attempt.get("stem"):
            return False
        local_quality = attempt.get("target_local_quality") or {}
        if not bool(local_quality.get("accepted", True)):
            return False
        pixel_stats = attempt.get("pixel_stats") or {}
        if any(
            bool(pixel_stats.get(key))
            for key in (
                "is_nearly_black",
                "is_visibility_too_low",
                "is_nearly_white",
                "invalid_dynamic_range",
            )
        ):
            return False
        if "p50" not in pixel_stats or "dynamic_range" not in pixel_stats:
            return False
        for key in ("p01", "p50", "p99", "max", "dynamic_range"):
            if key not in pixel_stats:
                continue
            try:
                value = float(pixel_stats.get(key))
            except (TypeError, ValueError):
                return False
            if not math.isfinite(value):
                return False
            if key in {"p50", "dynamic_range"} and value <= 0.0:
                return False
        target_attainment = attempt.get("preview_target_attainment") or {}
        try:
            attainment_ratio = float(target_attainment.get("attainment_ratio"))
        except (TypeError, ValueError):
            return False
        if not math.isfinite(attainment_ratio) or attainment_ratio <= 0.0:
            return False
        diagnostics = [
            str(item).strip()
            for item in (attempt.get("diagnostics") or [])
            if str(item).strip()
        ]
        reviewable_prefixes = (
            "preview_target_p50_ratio_above_max",
            "background_chroma_noise_score",
            "background_chroma_load_growth",
        )
        return bool(diagnostics) and all(
            item.startswith(reviewable_prefixes) for item in diagnostics
        )


    @classmethod
    def _stage7_review_candidate_selection_key(
        cls,
        attempt: Dict[str, Any],
    ) -> Tuple[Any, ...]:
        """Rank safe review rejects by gate severity before brightness proximity."""
        return cls._stage7_candidate_selection_key(attempt)


    def _stage7_attempt_allows_chroma_rescue(self, attempt: Dict[str, Any]) -> bool:
        """Only repair candidates rejected exclusively for background chroma."""
        if not bool(getattr(self.cfg, "stage7_chroma_rescue_enabled", True)):
            return False
        if attempt.get("status") != "ok" or not attempt.get("stem"):
            return False
        if str(attempt.get("method") or "") in {
            "adaptive_quantile",
            "background_chroma_rescue",
        }:
            return False
        local_quality = attempt.get("target_local_quality") or {}
        if not bool(local_quality.get("accepted", True)):
            return False
        diagnostics = [
            str(item).strip()
            for item in (attempt.get("diagnostics") or [])
            if str(item).strip()
        ]
        allowed_prefixes = (
            "background_chroma_noise_score",
            "background_chroma_load_growth",
        )
        return bool(diagnostics) and all(
            item.startswith(allowed_prefixes) for item in diagnostics
        )


    def _stage7_background_chroma_rescue_pixels(
        self,
        image_data: np.ndarray,
        *,
        strength: float,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reduce background chroma while preserving luminance and signal masks."""
        strength = _clamp_float(strength, 0.10, 0.75)
        masks = self._stage8_generate_starless_masks(np.asarray(image_data))
        rgb = np.asarray(masks["rgb"], dtype=np.float32)
        gray = np.asarray(masks["gray"], dtype=np.float32)
        background_weight = np.clip(
            np.asarray(masks["background_mask"], dtype=np.float32),
            0.0,
            1.0,
        )
        signal_layers = [
            np.asarray(masks[name], dtype=np.float32)
            for name in ("core_mask", "nebula_mask", "faint_nebula_mask")
            if name in masks
        ]
        if signal_layers:
            signal_protection = np.clip(
                np.maximum.reduce(signal_layers),
                0.0,
                1.0,
            )
            # Softening the background mask can bleed back into faint signal.
            # Re-apply the original signal masks before damping any chroma.
            background_weight *= 1.0 - signal_protection
        chroma = rgb - gray[None, :, :]
        chroma_keep = 1.0 - np.clip(background_weight * strength, 0.0, 0.75)
        rescued_rgb = gray[None, :, :] + chroma * chroma_keep[None, :, :]
        rescued = self._stage8_restore_rgb_like(
            np.asarray(image_data),
            np.clip(rescued_rgb, 0.0, 1.0),
        )
        return rescued, {
            "mode": "background_chroma_rescue",
            "strength": strength,
            "background_coverage": float(np.mean(background_weight > 0.50)),
            "mean_chroma_keep": float(np.mean(chroma_keep)),
            "luminance_preserved": True,
            "signal_masks": {
                "core": float((masks.get("coverage") or {}).get("core", 0.0)),
                "nebula": float((masks.get("coverage") or {}).get("nebula", 0.0)),
                "faint_nebula": float(
                    (masks.get("coverage") or {}).get("faint_nebula", 0.0)
                ),
            },
        }


    def _stage6_stretch_risk_score(
        self,
        metrics: Optional[QualityMetrics],
        issues: List[str],
        baseline_quality: Optional[QualityMetrics],
        *,
        enforce_star_growth: bool = True,
    ) -> float:
        if metrics is None:
            return 1_000_000.0 + len(issues)
        score = float(len(issues)) * 5.0
        bg_min = stage6_effective_bg_median_min(self.cfg.stage6_bg_median_min)
        if metrics.bg_median < bg_min:
            score += ((bg_min - metrics.bg_median) / bg_min) * 8.0
        if metrics.bg_median < 0.005:
            score += ((0.005 - metrics.bg_median) / 0.005) * 12.0
        black_max = max(float(self.cfg.stage6_black_pixel_ratio_max), 1e-4)
        if metrics.black_pixel_ratio > black_max:
            score += ((metrics.black_pixel_ratio - black_max) / black_max) * 3.0
        clip_max = max(float(self.cfg.stage6_highlight_clip_ratio_max), 1e-5)
        if metrics.highlight_clip_ratio > clip_max:
            score += ((metrics.highlight_clip_ratio - clip_max) / clip_max) * 6.0
        if (
            enforce_star_growth
            and baseline_quality
            and baseline_quality.median_star_size > 0.2
            and metrics.median_star_size > 0
        ):
            star_growth = metrics.median_star_size / max(baseline_quality.median_star_size, 1e-4)
            star_growth_limit = self._stage7_effective_star_growth_ratio_max()
            if star_growth > star_growth_limit:
                score += (star_growth - star_growth_limit) * 8.0
        return float(score)


    def _stage7_baseline_background_stats(
        self,
        baseline_quality: Optional[QualityMetrics],
        baseline_adaptive: Optional[Dict[str, Any]],
    ) -> Dict[str, float]:
        bg_median = 0.0
        if baseline_quality is not None:
            bg_median = float(getattr(baseline_quality, "bg_median", 0.0) or 0.0)
        adaptive = baseline_adaptive or {}
        if bg_median <= 0.0 and isinstance(adaptive, dict):
            try:
                bg_median = float(adaptive.get("bg_median", 0.0) or 0.0)
            except (TypeError, ValueError):
                bg_median = 0.0
        bg_std = 0.0
        if isinstance(adaptive, dict):
            try:
                bg_std = float(adaptive.get("bg_std", 0.0) or 0.0)
            except (TypeError, ValueError):
                bg_std = 0.0
        return {"bg_median": bg_median, "bg_std": bg_std}


    def _stage7_target_stretch_profile(self) -> Dict[str, Any]:
        """Map the active target policy onto the fixed Stage 7 A/B candidates."""
        target_type = (
            str(self._active_target_type() or "generic_low_snr_safe").strip().lower()
            if hasattr(self, "_active_target_type")
            else "generic_low_snr_safe"
        )
        policy_value = getattr(self, "pipeline_policy", {}) or {}
        policy = policy_value if isinstance(policy_value, dict) else {}
        policy_name = str(policy.get("policy_name") or "generic_low_snr_safe").strip().lower()
        stage_policy_value = policy.get("stage6_stretch") or {}
        stage_policy = stage_policy_value if isinstance(stage_policy_value, dict) else {}
        candidate_modes = [
            str(item).strip().lower()
            for item in (stage_policy.get("candidate_mode") or [])
            if str(item).strip()
        ]
        enabled = bool(getattr(self.cfg, "stage7_target_aware_stretch_enabled", True))

        profile = {
            "enabled": enabled,
            "name": "generic_balanced",
            "target_type": target_type,
            "policy_name": policy_name,
            "policy_candidate_modes": candidate_modes,
            "cand_a_p50_multiplier": 1.0,
            "cand_b_p50_multiplier": 1.0,
            "highlight_scale": 0.90,
            "cand_a_stretch_multiplier": 1.0,
            "cand_b_stretch_multiplier": 1.0,
            "cand_a_method": "asinh",
            "cand_a_pixel_params": {},
            "cand_b_method": "asinh_ghs",
            "cand_b_ghs_amount": None,
            "reason": "generic balanced stretch",
        }
        if not enabled:
            profile["reason"] = "target-aware stretch disabled by config"
            return profile

        mode_set = set(candidate_modes)
        if (
            target_type == "bright_emission_reflection_nebula"
            or policy_name == "bright_nebula_hdr_conservative"
            or "bright_nebula_hdr_masked" in mode_set
        ):
            profile.update(
                {
                    "name": "bright_core_protect",
                    "cand_a_p50_multiplier": 0.80,
                    "cand_b_p50_multiplier": 0.72,
                    "highlight_scale": 0.82,
                    "cand_a_stretch_multiplier": 0.92,
                    "cand_b_stretch_multiplier": 0.90,
                    "cand_a_method": "bright_nebula_hdr_masked",
                    "cand_a_pixel_params": {
                        "bg_pedestal": 0.024,
                        "faint_boost": 0.018,
                        "core_protection": 0.72,
                        "shadow_chroma_damping": 0.28,
                        "faint_saturation_boost": 0.026,
                        "star_mask_expand": int(
                            getattr(
                                self.cfg,
                                "stage7_bright_nebula_star_mask_expand",
                                4,
                            )
                        ),
                        "star_faint_suppression": float(
                            getattr(
                                self.cfg,
                                "stage7_bright_nebula_star_faint_suppression",
                                0.85,
                            )
                        ),
                        "star_detail_suppression": float(
                            getattr(
                                self.cfg,
                                "stage7_bright_nebula_star_detail_suppression",
                                0.18,
                            )
                        ),
                    },
                    "cand_b_ghs_amount": 1.00,
                    "reason": "protect bright nebula core and preserve highlight colour",
                }
            )
        elif (
            target_type in {"globular_cluster", "open_cluster", "reflection_nebula_cluster"}
            or "star_color_preserving_stretch" in mode_set
        ):
            profile.update(
                {
                    "name": "star_colour_preserve",
                    "cand_a_p50_multiplier": 0.82,
                    "cand_b_p50_multiplier": 0.72,
                    "highlight_scale": 0.82,
                    "cand_a_stretch_multiplier": 0.92,
                    "cand_b_stretch_multiplier": 0.88,
                    "cand_b_method": "asinh",
                    "reason": "avoid linked GHS star bloat and protect stellar colour",
                }
            )
        elif (
            target_type in {"large_galaxy", "small_galaxy"}
            or policy_name == "large_galaxy_core_protect"
            or "masked_galaxy_stretch" in mode_set
        ):
            profile.update(
                {
                    "name": "galaxy_core_halo_balance",
                    "cand_a_p50_multiplier": 0.95,
                    "cand_b_p50_multiplier": 0.88,
                    "highlight_scale": 0.85,
                    "cand_a_stretch_multiplier": 0.96,
                    "cand_b_stretch_multiplier": 0.94,
                    "cand_b_ghs_amount": 1.02,
                    "reason": "protect the galaxy core while retaining the outer halo",
                }
            )
        elif (
            target_type == "dark_nebula_low_contrast"
            or policy_name == "dark_nebula_low_contrast"
            or "dark_nebula_masked_lift" in mode_set
        ):
            profile.update(
                {
                    "name": "dark_nebula_separation",
                    "cand_a_p50_multiplier": 1.20,
                    "cand_b_p50_multiplier": 1.16,
                    "highlight_scale": 0.88,
                    "cand_a_stretch_multiplier": 1.08,
                    "cand_b_stretch_multiplier": 1.05,
                    "cand_a_method": "bright_nebula_hdr_masked",
                    "cand_a_pixel_params": {
                        "bg_pedestal": 0.022,
                        "faint_boost": 0.012,
                        "core_protection": 0.84,
                        "shadow_chroma_damping": 0.38,
                        "faint_saturation_boost": 0.014,
                    },
                    "cand_b_ghs_amount": 1.02,
                    "reason": "lift faint surrounding signal without crushing dark dust lanes",
                }
            )
        elif target_type == "emission_nebula_widefield" or policy_name == "emission_nebula_widefield":
            profile.update(
                {
                    "name": "widefield_nebulosity",
                    "cand_a_p50_multiplier": 1.12,
                    "cand_b_p50_multiplier": 1.08,
                    "highlight_scale": 0.90,
                    "cand_a_stretch_multiplier": 1.05,
                    "cand_b_stretch_multiplier": 1.03,
                    "cand_b_ghs_amount": 1.04,
                    "reason": "prioritise diffuse nebulosity while keeping GHS conservative",
                }
            )
        return profile


    def _stage7_compact_stretch_candidates(
        self,
        baseline_quality: Optional[QualityMetrics],
        baseline_adaptive: Optional[Dict[str, Any]],
        baseline_pixel_stats: Optional[Dict[str, Any]] = None,
        preview_pixel_stats: Optional[Dict[str, Any]] = None,
        *,
        starless_recomposition_planned: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        stats = self._stage7_baseline_background_stats(baseline_quality, baseline_adaptive)
        bg_median = float(stats.get("bg_median", 0.0) or 0.0)
        bg_std = float(stats.get("bg_std", 0.0) or 0.0)
        adaptation: Dict[str, Any] = {
            "mode": "default_compact",
            "bg_median": bg_median,
            "bg_std": bg_std,
            "reason": "baseline background not extremely low",
        }
        pixel_stats = baseline_pixel_stats or {}
        try:
            p01 = float(pixel_stats.get("p01", 0.0) or 0.0)
            p99 = float(pixel_stats.get("p99", 0.0) or 0.0)
            max_v = float(pixel_stats.get("max", 0.0) or 0.0)
        except (TypeError, ValueError):
            p01 = 0.0
            p99 = 0.0
            max_v = 0.0
        cand_a_params = {"asinh_stretch": 2.2, "asinh_offset": 0.002}
        cand_b_params = {
            "asinh_stretch": 2.1,
            "asinh_offset": 0.002,
            "ghs_shadowsclip": -2.1,
            "ghs_stretchamount": 1.05,
        }
        target_stretch = Stage6ServiceMixin._stage7_target_stretch_profile(self)

        if 0.0 < bg_median <= 0.005:
            severity = _clamp_float((0.005 - bg_median) / 0.003, 0.0, 1.0)
            safe_offset_candidates = [bg_median * 0.50]
            if p01 > 1e-6:
                safe_offset_candidates.append(p01 * 0.80)
            noise_floor = bg_median - 2.5 * bg_std
            if noise_floor > 1e-6:
                safe_offset_candidates.append(noise_floor)
            safe_offset = max(1e-6, min(safe_offset_candidates))
            gentler_offset = max(1e-6, min(safe_offset, bg_median * 0.25))
            cand_a_params = {
                "asinh_stretch": round(2.20 + 0.20 * severity, 3),
                "asinh_offset": round(safe_offset, 6),
            }
            cand_b_params = {
                "asinh_stretch": round(2.10 + 0.10 * severity, 3),
                "asinh_offset": round(gentler_offset, 6),
                "ghs_shadowsclip": -2.1,
                "ghs_stretchamount": round(1.02 - 0.02 * severity, 3),
            }
            adaptation.update(
                {
                    "mode": "extreme_low_background",
                    "severity": severity,
                    "offset_cap": {
                        "p01": p01,
                        "bg_median": bg_median,
                        "bg_std": bg_std,
                        "cap": safe_offset,
                        "reason": "keep Asinh offset below the measured background floor",
                    },
                    "reason": "bg_median<=0.005; keep offset below the background floor and retain enough stretch for faint signal",
                }
            )
        elif 0.005 < bg_median < 0.010:
            severity = _clamp_float((0.010 - bg_median) / 0.005, 0.0, 1.0)
            cand_a_params = {
                "asinh_stretch": round(2.20 - 0.40 * severity, 3),
                "asinh_offset": round(0.002 + 0.003 * severity, 5),
            }
            cand_b_params = {
                "asinh_stretch": round(2.10 - 0.25 * severity, 3),
                "asinh_offset": round(0.002 + 0.0025 * severity, 5),
                "ghs_shadowsclip": -2.1,
                "ghs_stretchamount": round(1.05 - 0.03 * severity, 3),
            }
            adaptation.update(
                {
                    "mode": "low_background",
                    "severity": severity,
                    "reason": "bg_median<0.010; moderately raise offset and reduce stretch",
                }
            )

        if target_stretch.get("enabled"):
            cand_a_params["asinh_stretch"] = round(
                _clamp_float(
                    float(cand_a_params["asinh_stretch"])
                    * float(target_stretch["cand_a_stretch_multiplier"]),
                    STAGE7_ASINH_STRETCH_MIN,
                    STAGE7_ASINH_STRETCH_MAX,
                ),
                3,
            )
            cand_b_params["asinh_stretch"] = round(
                _clamp_float(
                    float(cand_b_params["asinh_stretch"])
                    * float(target_stretch["cand_b_stretch_multiplier"]),
                    STAGE7_ASINH_STRETCH_MIN,
                    STAGE7_ASINH_STRETCH_MAX,
                ),
                3,
            )
            target_ghs_amount = target_stretch.get("cand_b_ghs_amount")
            if target_ghs_amount is not None:
                cand_b_params["ghs_stretchamount"] = min(
                    float(cand_b_params["ghs_stretchamount"]),
                    float(target_ghs_amount),
                )
        adaptation["target_aware"] = target_stretch
        cand_a_params.update(target_stretch.get("cand_a_pixel_params") or {})

        cap_candidates = [
            value
            for value in (p99 * 0.85, max_v * 0.80)
            if math.isfinite(value) and value > 0.0005
        ]
        if cap_candidates and adaptation.get("mode") != "extreme_low_background":
            offset_cap = max(0.0005, min(cap_candidates))
            capped: List[str] = []
            for name, params in (("cand_a", cand_a_params), ("cand_b", cand_b_params)):
                current_offset = float(params.get("asinh_offset", 0.002) or 0.002)
                if current_offset > offset_cap:
                    params["asinh_offset"] = round(offset_cap, 5)
                    capped.append(name)
            if capped:
                adaptation["offset_cap"] = {
                    "capped_candidates": capped,
                    "p99": p99,
                    "max": max_v,
                    "cap": offset_cap,
                    "reason": "asinh_offset must stay below starless effective signal range",
                }
                adaptation["reason"] = (
                    str(adaptation.get("reason") or "")
                    + "; capped offset below starless p99/max signal"
                ).strip("; ")

        calibration_enabled = bool(
            getattr(self.cfg, "stage7_preview_calibration_enabled", True)
        )
        preview_stats = (preview_pixel_stats or {}) if calibration_enabled else {}
        cand_a_preview_scale = _clamp_float(
            getattr(self.cfg, "stage7_preview_cand_a_p50_ratio", 0.35)
            * float(target_stretch.get("cand_a_p50_multiplier", 1.0)),
            0.10,
            0.60,
        )
        cand_b_preview_scale = _clamp_float(
            getattr(self.cfg, "stage7_preview_cand_b_p50_ratio", 0.25)
            * float(target_stretch.get("cand_b_p50_multiplier", 1.0)),
            0.10,
            cand_a_preview_scale,
        )
        preview_stretch_max = _clamp_float(
            getattr(self.cfg, "stage7_preview_asinh_stretch_max", 1000.0),
            10.0,
            STAGE7_ASINH_STRETCH_MAX,
        )
        cand_a_stretch, cand_a_calibration = _stage7_preview_calibrated_stretch(
            pixel_stats,
            preview_stats,
            offset=float(cand_a_params["asinh_offset"]),
            preview_scale=cand_a_preview_scale,
            highlight_scale=float(target_stretch.get("highlight_scale", 0.90)),
            fallback=float(cand_a_params["asinh_stretch"]),
            stretch_max=preview_stretch_max,
        )
        cand_b_stretch, cand_b_calibration = _stage7_preview_calibrated_stretch(
            pixel_stats,
            preview_stats,
            offset=float(cand_b_params["asinh_offset"]),
            preview_scale=cand_b_preview_scale,
            highlight_scale=float(target_stretch.get("highlight_scale", 0.90)),
            fallback=float(cand_b_params["asinh_stretch"]),
            stretch_max=preview_stretch_max,
        )
        if cand_a_calibration and cand_b_calibration:
            cand_a_params["asinh_stretch"] = round(cand_a_stretch, 3)
            cand_b_params["asinh_stretch"] = round(cand_b_stretch, 3)
            adaptation["preview_calibration"] = {
                "source": "stage7_preview_ref",
                "candidate_a": cand_a_calibration,
                "candidate_b": cand_b_calibration,
                "reason": (
                    "linked autostretch defines bounded P50/P99 targets; "
                    "preview remains reference-only"
                ),
            }
            adaptation["reason"] = (
                str(adaptation.get("reason") or "")
                + "; Asinh strengths calibrated from stage7_preview_ref"
            ).strip("; ")

        cand_a_method = str(target_stretch.get("cand_a_method") or "asinh")
        cand_b_method = str(target_stretch.get("cand_b_method") or "asinh_ghs")
        if starless_recomposition_planned and cand_b_calibration:
            try:
                source_p50 = float(pixel_stats.get("p50", 0.0) or 0.0)
                calibrated_target_p50 = float(
                    cand_b_calibration.get("target_p50", 0.0) or 0.0
                )
                preview_p50 = float(
                    cand_b_calibration.get("preview_p50", 0.0) or 0.0
                )
                profile_name = str(target_stretch.get("name") or "")
                target_floor = float(
                    getattr(
                        self.cfg,
                        "stage7_starless_linked_mtf_p50_min",
                        0.15,
                    )
                )
                if profile_name in {"widefield_nebulosity", "dark_nebula_separation"}:
                    target_floor = max(
                        target_floor,
                        float(
                            getattr(
                                self.cfg,
                                "stage7_starless_linked_mtf_diffuse_p50_min",
                                0.17,
                            )
                        ),
                    )
                preview_ratio = _clamp_float(
                    getattr(
                        self.cfg,
                        "stage7_starless_linked_mtf_preview_p50_ratio",
                        0.70,
                    ),
                    0.40,
                    0.90,
                )
                target_ceiling = _clamp_float(
                    getattr(
                        self.cfg,
                        "stage7_starless_linked_mtf_p50_max",
                        0.22,
                    ),
                    0.15,
                    0.35,
                )
                target_p50 = _clamp_float(
                    max(
                        calibrated_target_p50,
                        target_floor,
                        preview_p50 * preview_ratio,
                    ),
                    min(target_floor, target_ceiling),
                    target_ceiling,
                )

                p01 = float(pixel_stats.get("p01", 0.0) or 0.0)
                min_value = float(pixel_stats.get("min", 0.0) or 0.0)
                noise_sigma = _clamp_float(
                    getattr(
                        self.cfg,
                        "stage7_starless_linked_mtf_shadow_noise_sigma",
                        3.0,
                    ),
                    2.0,
                    6.0,
                )
                shadow_margin = max(
                    noise_sigma * max(bg_std, 0.0),
                    max(source_p50 - p01, 0.0) * 0.25,
                    1e-6,
                )
                shadow_candidates = [
                    value - shadow_margin
                    for value in (p01, min_value)
                    if math.isfinite(value) and value > 0.0
                ]
                shadows = max(0.0, min(shadow_candidates)) if shadow_candidates else 0.0
                shadows = min(
                    shadows,
                    max(0.0, source_p50 - shadow_margin),
                )
                normalized_source_p50 = (
                    (source_p50 - shadows) / max(1.0 - shadows, 1e-12)
                )
                midpoint = _stage7_linked_mtf_midtones(
                    normalized_source_p50,
                    target_p50,
                )
                cand_b_method = "linked_mtf"
                cand_b_params = {
                    "mtf_shadows": round(shadows, 9),
                    "mtf_midtones": round(midpoint, 9),
                    "mtf_highlights": 1.0,
                    "source_background": source_p50,
                    "target_background": target_p50,
                }
                cand_b_calibration.update(
                    {
                        "calibration_method": "noise_floor_linked_mtf",
                        "calibrated_parameter": "mtf_midtones",
                        "calibrated_stretch": midpoint,
                        "stretch_max": 1.0,
                        "target_p50_before_visibility_floor": calibrated_target_p50,
                        "target_p50": target_p50,
                        "predicted_p50": _stage7_mtf_sample(
                            source_p50,
                            shadows,
                            midpoint,
                        ),
                        "predicted_p99": _stage7_mtf_sample(
                            float(pixel_stats.get("p99", 0.0) or 0.0),
                            shadows,
                            midpoint,
                        ),
                        "mtf": [shadows, midpoint, 1.0],
                        "shadow_margin": shadow_margin,
                    }
                )
                adaptation["starless_recomposition_candidate"] = {
                    "candidate": "cand_b",
                    "method": "noise_floor_linked_mtf",
                    "source_p50": source_p50,
                    "target_p50": target_p50,
                    "mtf": [shadows, midpoint, 1.0],
                    "shadow_margin": shadow_margin,
                    "reason": (
                        "expand the narrow Starless pedestal from a measured "
                        "noise-floor shadow while keeping the shadow below the "
                        "sampled minimum when available"
                    ),
                }
            except (TypeError, ValueError) as error:
                adaptation["starless_recomposition_candidate"] = {
                    "candidate": "cand_b",
                    "method": cand_b_method,
                    "status": "linked_mtf_unavailable",
                    "reason": str(error),
                }

        candidates = [
            {
                "name": "cand_a",
                "stem": "stage7_cand_a",
                "method": cand_a_method,
                "params": cand_a_params,
                "adaptation": adaptation,
            },
            {
                "name": "cand_b",
                "stem": "stage7_cand_b",
                "method": cand_b_method,
                "params": cand_b_params,
                "adaptation": adaptation,
            },
        ]
        return candidates, adaptation


    def _stage7_quantile_fallback_candidate(
        self,
        baseline_image_data: Optional[np.ndarray],
        stretch_adaptation: Dict[str, Any],
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """Build the deterministic last-resort curve from measured quantiles."""
        if baseline_image_data is None:
            unavailable = {
                "status": "unavailable",
                "reason": "baseline pixels unavailable",
            }
            return None, unavailable
        calibration = stage7_stretch_metrics.calibrate_adaptive_quantile_stretch(
            baseline_image_data,
            stretch_adaptation,
            self.cfg,
        )
        if calibration.get("status") != "ok":
            return None, calibration
        adaptation = copy.deepcopy(stretch_adaptation)
        adaptation["quantile_fallback"] = calibration
        return (
            {
                "name": "cand_quantile",
                "stem": "stage7_cand_quantile",
                "method": "adaptive_quantile",
                "params": {"calibration": calibration},
                "adaptation": adaptation,
                "calibration_candidate": "cand_a",
                "explicit_fallback": True,
                "feedback": {
                    "mode": "adaptive_quantile_fallback",
                    "source": calibration.get("source"),
                },
            },
            calibration,
        )


    def _run_stage6_ai_stretching(
        self,
        allow_ai: bool = True,
    ) -> Tuple[bool, bool, List[str], str]:
        self._stage7_stretch_validated_rescue = False
        self._stage7_stretch_fallback_reason = None
        self._stage7_review_source = None
        messages: List[str] = []
        source_stem = str(
            getattr(self, "_stage7_stretch_source", None) or "stage6_starless"
        )
        try:
            self.cmd_with_check("load", source_stem)
        except (CommandError, SirilError) as e:
            messages.append(f"stage7 source load failed: {self._short_text(e, 180)}")
            return False, False, messages, ""

        baseline_quality = self._measure_current_quality()
        baseline_adaptive = (
            self._adaptive_features_current()
            if hasattr(self, "_adaptive_features_current")
            else {}
        )
        baseline_pixel_stats = (
            self._current_pixel_distribution_stats()
            if hasattr(self, "_current_pixel_distribution_stats")
            else {}
        )
        baseline_image_data: Optional[np.ndarray] = None
        try:
            current_data = self.siril.get_image_pixeldata(preview=False)
            if current_data is not None:
                baseline_image_data = np.array(current_data, copy=True)
        except (CommandError, DataError, SirilError, RuntimeError, TypeError, ValueError):
            baseline_image_data = None
        frozen_background_masks: Optional[Dict[str, Any]] = None
        frozen_background_sampling: Dict[str, Any] = {
            "status": "unavailable",
            "method": "candidate_local_background_fallback",
        }
        if baseline_image_data is not None:
            try:
                source_masks = self._stage8_generate_starless_masks(
                    baseline_image_data
                )
                background_mask = np.asarray(
                    source_masks.get("background_mask"),
                    dtype=np.float32,
                )
                if (
                    background_mask.ndim == 2
                    and background_mask.size > 0
                    and np.all(np.isfinite(background_mask))
                ):
                    frozen_background_masks = {
                        "background_mask": np.array(
                            background_mask,
                            dtype=np.float32,
                            copy=True,
                        )
                    }
                    frozen_background_sampling = {
                        "status": "available",
                        "method": "frozen_stage6_source_background_mask_v1",
                        "candidate_independent": True,
                        "coverage_gt_0_50": float(
                            np.mean(background_mask > 0.50)
                        ),
                        "source": f"{source_stem}.fit",
                    }
            except (
                IndexError,
                RuntimeError,
                TypeError,
                ValueError,
                FloatingPointError,
            ):
                frozen_background_masks = None
        baseline_background_quality = (
            self._background_quality_metrics(
                baseline_image_data,
                frozen_background_masks,
            )
            if baseline_image_data is not None
            else {}
        )
        starmask_image_data: Optional[np.ndarray] = None
        if source_stem == "stage6_starless":
            starmask_file = getattr(self, "starmask_file", None)
            starmask_stem = str(getattr(starmask_file, "stem", "") or "")
            if starmask_stem:
                loaded_starmask = self._read_image_by_stem(starmask_stem)
                if loaded_starmask is not None:
                    starmask_image_data = np.array(loaded_starmask, copy=True)
            try:
                self.cmd_with_check("load", source_stem)
            except (CommandError, SirilError) as error:
                messages.append(
                    "stage7 source reload after starmask sampling failed: "
                    f"{self._short_text(error, 160)}"
                )
                return False, False, messages, ""
        messages.append(
            "stage7 primary stretch candidates: stage7_cand_a, stage7_cand_b; "
            "preview=stage7_preview_ref; chroma rescue is conditional"
        )
        messages.append(
            "stage7 Starless structure gate="
            + ("starmask_rank_local" if starmask_image_data is not None else "generic_fallback")
        )
        if allow_ai:
            messages.append(
                "stage7 keeps the fixed compact candidate set; AI does not expand it"
            )

        if self.process_dir:
            for pattern in (
                "stage6_candidate_*.fit",
                "stage7_candidate_*.fit",
                "stage7_cand_*.fit",
                "stage7_preview_ref.fit",
                "stage6_selected*.fit",
                "stage7_selected*.fit",
                "stage6_stretched.fit",
                "stage7_stretched.fit",
            ):
                for stale_path in self.process_dir.glob(pattern):
                    try:
                        stale_path.unlink()
                    except OSError as e:
                        self.log.warn(f"[Stage7] stale stretch candidate cleanup failed: {e}")

        preview_saved = False
        preview_pixel_stats: Dict[str, Any] = {}
        preview_quality: Optional[QualityMetrics] = None
        try:
            self.cmd_with_check("load", source_stem)
            self.cmd_with_check("autostretch", "-linked")
            preview_saved = self._save_stage_output("stage7_preview_ref")
            if preview_saved:
                preview_pixel_stats = self._current_pixel_distribution_stats()
                preview_quality = self._measure_current_quality()
        except (CommandError, SirilError) as e:
            messages.append(f"stage7 preview_ref failed: {self._short_text(e, 160)}")

        candidate_list, stretch_adaptation = self._stage7_compact_stretch_candidates(
            baseline_quality,
            baseline_adaptive,
            baseline_pixel_stats,
            preview_pixel_stats,
            starless_recomposition_planned=source_stem == "stage6_starless",
        )
        target_stretch = stretch_adaptation.get("target_aware") or {}
        messages.append(
            "stage7 target-aware stretch "
            f"profile={target_stretch.get('name', 'generic_balanced')}, "
            f"target={target_stretch.get('target_type', 'generic_low_snr_safe')}, "
            f"policy={target_stretch.get('policy_name', 'generic_low_snr_safe')}"
        )
        preview_calibration = stretch_adaptation.get("preview_calibration")
        if isinstance(preview_calibration, dict):
            candidate_a = preview_calibration.get("candidate_a") or {}
            candidate_b = preview_calibration.get("candidate_b") or {}
            if str(candidate_list[1].get("method") or "") == "linked_mtf":
                messages.append(
                    "stage7 preview calibration "
                    f"p50={float(candidate_a.get('preview_p50', 0.0) or 0.0):.4f}, "
                    "cand_a_asinh="
                    f"{float(candidate_a.get('calibrated_stretch', 0.0) or 0.0):.1f}, "
                    "cand_b_linked_mtf_midtones="
                    f"{float(candidate_b.get('calibrated_stretch', 0.0) or 0.0):.6f}"
                )
            else:
                messages.append(
                    "stage7 preview calibration "
                    f"p50={float(candidate_a.get('preview_p50', 0.0) or 0.0):.4f}, "
                    "asinh=("
                    f"{float(candidate_a.get('calibrated_stretch', 0.0) or 0.0):.1f},"
                    f"{float(candidate_b.get('calibrated_stretch', 0.0) or 0.0):.1f})"
                )
        if stretch_adaptation.get("mode") != "default_compact":
            messages.append(
                "stage7 low-background stretch adaptation "
                f"mode={stretch_adaptation.get('mode')}, "
                f"bg_median={float(stretch_adaptation.get('bg_median', 0.0) or 0.0):.4f}"
            )
        attempts: List[Dict[str, Any]] = []
        best_attempt: Optional[Dict[str, Any]] = None

        for candidate in candidate_list:
            attempt = self._stage7_evaluate_stretch_candidate(
                candidate,
                source_stem=source_stem,
                baseline_quality=baseline_quality,
                baseline_image_data=baseline_image_data,
                starmask_image_data=starmask_image_data,
                baseline_background_quality=baseline_background_quality,
                frozen_background_masks=frozen_background_masks,
                target_stretch=target_stretch,
            )
            attempts.append(attempt)
            retry_source = candidate
            retry_attempt = attempt
            try:
                feedback_retry_max = int(
                    getattr(self.cfg, "stage7_stretch_feedback_retry_max", 1)
                )
            except (TypeError, ValueError):
                feedback_retry_max = 1
            feedback_retry_max = max(0, min(feedback_retry_max, 1))
            for retry_index in range(1, feedback_retry_max + 1):
                if bool(retry_attempt.get("allowed_as_final", False)):
                    break
                feedback_candidate = self._stage7_feedback_retry_candidate(
                    retry_source,
                    retry_attempt,
                    retry_index,
                )
                if feedback_candidate is None:
                    break
                retry_attempt = self._stage7_evaluate_stretch_candidate(
                    feedback_candidate,
                    source_stem=source_stem,
                    baseline_quality=baseline_quality,
                    baseline_image_data=baseline_image_data,
                    starmask_image_data=starmask_image_data,
                    baseline_background_quality=baseline_background_quality,
                    frozen_background_masks=frozen_background_masks,
                    target_stretch=target_stretch,
                )
                attempts.append(retry_attempt)
                retry_source = feedback_candidate
                if bool(retry_attempt.get("allowed_as_final", False)):
                    messages.append(
                        "stage7 post-transform P50 calibration passed all gates "
                        f"(source={candidate.get('name')}, "
                        f"retry={retry_index}, "
                        f"method={feedback_candidate.get('method')})"
                    )
                    break

        accepted_before_chroma_rescue = [
            attempt
            for attempt in attempts
            if attempt.get("status") == "ok"
            and attempt.get("stem")
            and bool(attempt.get("allowed_as_final", False))
        ]
        best_attempt = (
            min(
                accepted_before_chroma_rescue,
                key=self._stage7_candidate_selection_key,
            )
            if accepted_before_chroma_rescue
            else None
        )

        if best_attempt is None:
            quantile_candidate, quantile_calibration = (
                self._stage7_quantile_fallback_candidate(
                    baseline_image_data,
                    stretch_adaptation,
                )
            )
            stretch_adaptation["quantile_fallback"] = quantile_calibration
            if quantile_candidate is not None:
                quantile_attempt = self._stage7_evaluate_stretch_candidate(
                    quantile_candidate,
                    source_stem=source_stem,
                    baseline_quality=baseline_quality,
                    baseline_image_data=baseline_image_data,
                    starmask_image_data=starmask_image_data,
                    baseline_background_quality=baseline_background_quality,
                    frozen_background_masks=frozen_background_masks,
                    target_stretch=target_stretch,
                )
                attempts.append(quantile_attempt)
                if bool(quantile_attempt.get("allowed_as_final", False)):
                    best_attempt = quantile_attempt
                    messages.append(
                        "stage7 adaptive quantile fallback passed all gates "
                        "(linked preview-calibrated monotonic curve)"
                    )
            elif quantile_calibration.get("status") != "disabled":
                messages.append(
                    "stage7 adaptive quantile fallback unavailable: "
                    f"{self._short_text(quantile_calibration.get('reason'), 160)}"
                )

        if best_attempt is None:
            rejected_attempts = [
                attempt
                for attempt in attempts
                if attempt.get("status") == "ok" and attempt.get("stem")
            ]
            if rejected_attempts:
                eligible_rescue_attempts = sorted(
                    (
                        attempt
                        for attempt in rejected_attempts
                        if self._stage7_attempt_allows_chroma_rescue(attempt)
                    ),
                    key=self._stage7_candidate_selection_key,
                )
                if eligible_rescue_attempts:
                    rescue_source_attempt = eligible_rescue_attempts[0]
                    rescue_source = str(rescue_source_attempt["stem"])
                    rescue_root_candidate = {
                        "name": str(
                            rescue_source_attempt.get("name") or "candidate"
                        ),
                        "method": rescue_source_attempt.get("method"),
                        "params": dict(
                            rescue_source_attempt.get("params") or {}
                        ),
                        "adaptation": rescue_source_attempt.get("adaptation"),
                    }
                    rescue_calibration_candidate = str(
                        rescue_source_attempt.get("calibration_candidate")
                        or rescue_source_attempt.get("name")
                        or ""
                    )
                    for rescue_index, rescue_strength in enumerate(
                        self._stage7_chroma_rescue_strengths(),
                        start=1,
                    ):
                        rescue_name = f"chroma_rescue_{rescue_index}"
                        rescue_stem = f"stage7_cand_rescue_{rescue_index}"
                        try:
                            self.cmd_with_check("load", source_stem)
                            replay_ok, replay_used = (
                                self._execute_stage6_stretch_candidate(
                                    rescue_root_candidate,
                                    starmask_image_data=starmask_image_data,
                                )
                            )
                            if not replay_ok:
                                raise RuntimeError(
                                    "failed to replay chroma-rescue parent from "
                                    f"the frozen source: {replay_used}"
                                )
                            rejected_data = self.siril.get_image_pixeldata(preview=False)
                            if rejected_data is None:
                                raise RuntimeError("rejected candidate pixel buffer is empty")
                            rescued_data, rescue_adaptation = (
                                self._stage7_background_chroma_rescue_pixels(
                                    np.asarray(rejected_data),
                                    strength=rescue_strength,
                                )
                            )
                            self._set_current_image_pixeldata(
                                rescued_data,
                                label=f"Stage7 {rescue_name}",
                            )

                            starless_structure_quality: Dict[str, Any] = {
                                "status": "unavailable",
                                "accepted": True,
                                "issues": [],
                                "risk_score": 0.0,
                                "metrics": {},
                                "reason": "Starless structure inputs unavailable",
                            }
                            if baseline_image_data is not None and source_stem == "stage6_starless":
                                starless_structure_quality = (
                                    stage7_stretch_metrics.assess_starless_structure_growth(
                                        baseline_image_data,
                                        rescued_data,
                                        starmask_image_data,
                                        self.cfg,
                                    )
                                )
                            use_starless_structure_gate = (
                                starless_structure_quality.get("status")
                                in {"ok", "rejected"}
                            )
                            enforce_star_growth = (
                                self._stage7_should_enforce_star_growth(
                                    use_starless_structure_gate
                                )
                            )
                            quality_ok, issues, metrics = (
                                self._validate_stage6_stretch_quality(
                                    baseline_quality,
                                    enforce_star_growth=enforce_star_growth,
                                )
                            )
                            pixel_stats = self._current_pixel_distribution_stats()
                            invalid_reasons = [
                                reason
                                for key, reason in (
                                    ("is_nearly_black", "nearly_black"),
                                    ("is_visibility_too_low", "visibility_too_low"),
                                    ("is_nearly_white", "nearly_white"),
                                    ("invalid_dynamic_range", "invalid_dynamic_range"),
                                )
                                if pixel_stats.get(key)
                            ]
                            if invalid_reasons:
                                quality_ok = False
                                issues = [*issues, *invalid_reasons]
                            preview_target_attainment = (
                                _stage7_preview_target_attainment(
                                    rescue_calibration_candidate,
                                    pixel_stats,
                                    dict(
                                        rescue_source_attempt.get("adaptation") or {}
                                    ),
                                    min_ratio=getattr(
                                        self.cfg,
                                        "stage7_preview_target_p50_min_ratio",
                                        0.55,
                                    ),
                                    max_ratio=getattr(
                                        self.cfg,
                                        "stage7_preview_target_p50_max_ratio",
                                        1.50,
                                    ),
                                )
                            )
                            if not bool(
                                preview_target_attainment.get("accepted", True)
                            ):
                                quality_ok = False
                                issues = [
                                    *issues,
                                    *list(
                                        preview_target_attainment.get("issues") or []
                                    ),
                                ]

                            local_quality: Dict[str, Any] = {
                                "status": "unavailable",
                                "accepted": True,
                                "issues": [],
                                "risk_score": 0.0,
                                "metrics": {},
                            }
                            if baseline_image_data is not None:
                                local_quality = (
                                    stage7_stretch_metrics.assess_target_local_stretch(
                                        baseline_image_data,
                                        rescued_data,
                                        str(
                                            target_stretch.get("target_type")
                                            or "generic_low_snr_safe"
                                        ),
                                        self.cfg,
                                    )
                                )
                            rescued_background_quality = self._background_quality_metrics(
                                rescued_data,
                                frozen_background_masks,
                            )
                            background_quality_gate = self._stage7_stretch_background_gate(
                                baseline_background_quality,
                                rescued_background_quality,
                            )
                            if not bool(background_quality_gate.get("accepted", False)):
                                quality_ok = False
                                issues = [
                                    *issues,
                                    *list(background_quality_gate.get("issues") or []),
                                ]
                            if not bool(local_quality.get("accepted", True)):
                                quality_ok = False
                                issues = [
                                    *issues,
                                    *list(local_quality.get("issues") or []),
                                ]
                            if not bool(
                                starless_structure_quality.get("accepted", True)
                            ):
                                quality_ok = False
                                issues = [
                                    *issues,
                                    *list(
                                        starless_structure_quality.get("issues")
                                        or []
                                    ),
                                ]

                            risk_score = self._stage6_stretch_risk_score(
                                metrics,
                                issues,
                                baseline_quality,
                                enforce_star_growth=enforce_star_growth,
                            )
                            risk_score += float(
                                local_quality.get("risk_score", 0.0) or 0.0
                            )
                            risk_score += float(
                                starless_structure_quality.get("risk_score", 0.0)
                                or 0.0
                            )
                            rescue_saved = self._save_stage_output(rescue_stem)
                            rescue_attempt = {
                                "name": rescue_name,
                                "file": f"{rescue_stem}.fit" if rescue_saved else None,
                                "stem": rescue_stem if rescue_saved else None,
                                "method": "background_chroma_rescue",
                                "params": {
                                    "source_candidate": rescue_source,
                                    "frozen_source": source_stem,
                                    "strength": rescue_strength,
                                },
                                "adaptation": rescue_adaptation,
                                "status": "ok" if rescue_saved else "failed",
                                "used": (
                                    "replayed from frozen source then applied "
                                    "background-only luminance-preserving chroma rescue"
                                ),
                                "quality_ok": quality_ok,
                                "diagnostics": issues,
                                "metrics": asdict(metrics) if metrics else None,
                                "adaptive_metrics": (
                                    self._adaptive_features_current()
                                    if hasattr(self, "_adaptive_features_current")
                                    else None
                                ),
                                "pixel_stats": pixel_stats,
                                "preview_target_attainment": preview_target_attainment,
                                "target_local_quality": local_quality,
                                "starless_structure_quality": starless_structure_quality,
                                "background_quality_gate": background_quality_gate,
                                "risk_score": risk_score,
                                "allowed_as_final": bool(quality_ok),
                                "explicit_fallback": True,
                                "calibration_candidate": (
                                    rescue_calibration_candidate
                                ),
                                "feedback": {
                                    "mode": "background_chroma_rescue",
                                    "source_candidate": rescue_source,
                                    "frozen_source": source_stem,
                                    "replayed_transform": replay_used,
                                },
                            }
                            attempts.append(rescue_attempt)
                            if rescue_saved and quality_ok:
                                messages.append(
                                    "stage7 background chroma rescue passed gates "
                                    f"(source={rescue_source}, strength={rescue_strength:.2f})"
                                )
                        except (
                            CommandError,
                            DataError,
                            SirilError,
                            RuntimeError,
                            TypeError,
                            ValueError,
                        ) as error:
                            attempts.append(
                                {
                                    "name": rescue_name,
                                    "file": None,
                                    "stem": None,
                                    "method": "background_chroma_rescue",
                                    "params": {
                                        "source_candidate": rescue_source,
                                        "frozen_source": source_stem,
                                        "strength": rescue_strength,
                                    },
                                    "status": "failed",
                                    "reason": self._short_text(error, 180),
                                    "explicit_fallback": True,
                                }
                            )

        saved_attempts = [
            attempt
            for attempt in attempts
            if attempt.get("status") == "ok" and attempt.get("stem")
        ]
        accepted_attempts = [
            attempt
            for attempt in saved_attempts
            if bool(attempt.get("allowed_as_final", False))
        ]
        rejected_attempts = [
            attempt
            for attempt in saved_attempts
            if not bool(attempt.get("allowed_as_final", False))
        ]
        deterministic_best_attempt = (
            min(accepted_attempts, key=self._stage7_candidate_selection_key)
            if accepted_attempts
            else None
        )
        ai_selection: Optional[Dict[str, Any]] = None
        best_attempt = deterministic_best_attempt
        if allow_ai and accepted_attempts:
            ai_selection = self._request_stage7_stretch_selection(
                accepted_attempts
            )
            if ai_selection:
                selected_candidate_id = str(
                    ai_selection.get("selected_candidate_id") or ""
                )
                selected_attempt = next(
                    (
                        attempt
                        for attempt in accepted_attempts
                        if str(attempt.get("name") or "")
                        == selected_candidate_id
                        and not bool(attempt.get("explicit_fallback"))
                    ),
                    None,
                )
                if selected_attempt is not None:
                    best_attempt = selected_attempt
                else:
                    self.log.warn(
                        "[AI] stage7 stretch selection was not revalidated; "
                        "using deterministic quality rank"
                    )
                    ai_selection = None
        best_rejected_attempt = (
            min(rejected_attempts, key=self._stage7_candidate_selection_key)
            if rejected_attempts
            else None
        )
        safe_review_attempts = [
            attempt
            for attempt in rejected_attempts
            if self._stage7_review_candidate_is_safe(attempt)
        ]
        review_attempt = (
            min(
                safe_review_attempts,
                key=self._stage7_review_candidate_selection_key,
            )
            if best_attempt is None and safe_review_attempts
            else None
        )
        selection_ranks = {
            id(attempt): rank
            for rank, attempt in enumerate(
                sorted(saved_attempts, key=self._stage7_candidate_selection_key),
                start=1,
            )
        }
        for attempt in attempts:
            attempt["selection_rank"] = selection_ranks.get(id(attempt))
            if attempt is best_attempt:
                attempt["selection_role"] = "selected_final"
            elif attempt is review_attempt:
                attempt["selection_role"] = "selected_review"
            else:
                attempt["selection_role"] = "not_selected"

        if best_attempt is not None:
            if ai_selection:
                messages.append(
                    "stage7 AI selected a hard-gate-passing candidate id "
                    f"(name={best_attempt.get('name')}, "
                    f"risk={float(best_attempt.get('risk_score', 0.0) or 0.0):.3f})"
                )
            else:
                messages.append(
                    "stage7 deterministic quality-ranked candidate selected "
                    f"(name={best_attempt.get('name')}, "
                    f"risk={float(best_attempt.get('risk_score', 0.0) or 0.0):.3f})"
                )
        elif review_attempt is not None:
            self._stage7_review_source = str(review_attempt.get("stem") or "") or None
            messages.append(
                "stage7 rejected all final candidates; quality-ranked safe review source="
                f"{review_attempt.get('name')} "
                f"(stem={self._stage7_review_source}, "
                f"ratio={float((review_attempt.get('preview_target_attainment') or {}).get('attainment_ratio', 0.0) or 0.0):.3f}, "
                f"risk={float(review_attempt.get('risk_score', 0.0) or 0.0):.3f})"
            )

        selected_fallback_reason = (
            self._stage7_validated_fallback_reason(best_attempt)
            if best_attempt is not None
            else ""
        )
        selection_summary = {
            "strategy": "hard_gate_then_quality_rank_with_safe_review",
            "selector": (
                "ai_candidate_id"
                if ai_selection
                else "deterministic_quality_rank"
            ),
            "model_output_fields": ["selected_candidate_id"],
            "parameters_owned_by": "code",
            "allowed_candidate_ids": [
                str(attempt.get("name"))
                for attempt in accepted_attempts
                if not bool(attempt.get("explicit_fallback"))
            ],
            "ai_selection": ai_selection,
            "deterministic_fallback": (
                str(deterministic_best_attempt.get("name"))
                if deterministic_best_attempt
                else None
            ),
            "saved_candidate_count": len(saved_attempts),
            "accepted_candidate_count": len(accepted_attempts),
            "safe_review_candidate_count": len(safe_review_attempts),
            "selected_final": (
                str(best_attempt.get("name")) if best_attempt else None
            ),
            "selected_fallback_reason": selected_fallback_reason or None,
            "selected_review": (
                str(review_attempt.get("name"))
                if review_attempt
                else None
            ),
            "review_source": self._stage7_review_source,
        }

        self._write_stage_json(
            "stage7_stretch_quality.json",
            {
                "stage": "stage7_stretch",
                "input": f"{source_stem}.fit",
                "preview": {
                    "name": "preview_ref",
                    "file": "stage7_preview_ref.fit" if preview_saved else None,
                    "method": "autostretch -linked",
                    "reference_only": True,
                    "pixel_stats": preview_pixel_stats,
                    "quality": asdict(preview_quality) if preview_quality else None,
                },
                "baseline_adaptive": baseline_adaptive,
                "baseline_background_quality": baseline_background_quality,
                "background_sampling": frozen_background_sampling,
                "baseline_pixel_stats": baseline_pixel_stats,
                "stretch_adaptation": stretch_adaptation,
                "attempts": attempts,
                "selected": best_attempt,
                "best_rejected": best_rejected_attempt,
                "selection": selection_summary,
            },
        )
        self._write_stage_json(
            "stretch_candidates_report.json",
            {
                "stage": "stage7_stretch",
                "input": f"{source_stem}.fit",
                "stretch_adaptation": stretch_adaptation,
                "baseline_background_quality": baseline_background_quality,
                "background_sampling": frozen_background_sampling,
                "baseline_pixel_stats": baseline_pixel_stats,
                "candidates": [
                    {
                        "name": item.get("name"),
                        "file": item.get("file"),
                        "method": item.get("method"),
                        "params": item.get("params"),
                        "adaptation": item.get("adaptation"),
                        "quality_ok": item.get("quality_ok"),
                        "risk_score": item.get("risk_score"),
                        "pixel_stats": item.get("pixel_stats"),
                        "preview_target_attainment": item.get("preview_target_attainment"),
                        "target_local_quality": item.get("target_local_quality"),
                        "background_quality_gate": item.get("background_quality_gate"),
                        "status": item.get("status"),
                        "diagnostics": item.get("diagnostics"),
                        "feedback": item.get("feedback"),
                        "explicit_fallback": bool(item.get("explicit_fallback")),
                        "selection_rank": item.get("selection_rank"),
                        "selection_role": item.get("selection_role"),
                    }
                    for item in attempts
                ],
                "preview": "stage7_preview_ref.fit" if preview_saved else None,
                "selected": (
                    {
                        "name": best_attempt.get("name"),
                        "source_file": best_attempt.get("file"),
                        "file": "stage7_stretched.fit",
                        "normal_selected": not bool(best_attempt.get("explicit_fallback")),
                        "validated_rescue": bool(best_attempt.get("explicit_fallback")),
                        "reason_code": selected_fallback_reason or "accepted",
                    }
                    if best_attempt
                    else None
                ),
                "review_selected": (
                    {
                        "name": review_attempt.get("name"),
                        "source_file": review_attempt.get("file"),
                        "stem": review_attempt.get("stem"),
                    }
                    if review_attempt
                    else None
                ),
                "selection": selection_summary,
            },
        )
        self._stage7_stretch_candidates = attempts
        self._stage7_stretch_selected = (
            str(best_attempt.get("name")) if best_attempt else None
        )

        if not best_attempt or not best_attempt.get("stem"):
            try:
                self.cmd_with_check("load", source_stem)
            except (CommandError, SirilError):
                pass
            return False, False, messages, ""

        self.cmd_with_check("load", str(best_attempt["stem"]))
        self.log.info(
            "[Stage7] selected="
            f"{best_attempt.get('name')} risk={best_attempt.get('risk_score')}"
        )
        messages.append(
            "stage7_selected="
            f"{best_attempt.get('name')}; "
            f"fallback_used={str(bool(best_attempt.get('explicit_fallback'))).lower()}; "
            f"reason_code={selected_fallback_reason or 'accepted'}"
        )
        pixel_stats = best_attempt.get("pixel_stats") or {}
        if pixel_stats:
            messages.append(
                "stage7_selected_distribution="
                f"p50={float(pixel_stats.get('p50', 0.0) or 0.0):.4f}, "
                f"p99={float(pixel_stats.get('p99', 0.0) or 0.0):.4f}, "
                f"dynamic={float(pixel_stats.get('dynamic_range', 0.0) or 0.0):.4f}"
            )
        selected_method = str(best_attempt.get("used") or best_attempt.get("method") or "")
        self._stage7_stretch_validated_rescue = bool(
            best_attempt.get("explicit_fallback")
        )
        self._stage7_stretch_fallback_reason = selected_fallback_reason or None
        return True, bool(best_attempt.get("explicit_fallback")), messages, selected_method
