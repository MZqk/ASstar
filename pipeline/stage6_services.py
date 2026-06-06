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

def stage6_effective_bg_median_min(configured_min: float) -> float:
    """Return the Stage6 dark-floor gate with room for FITS sampling noise."""
    configured = max(float(configured_min), 1e-4)
    tolerance = min(0.0005, configured * 0.025)
    return max(1e-4, configured - tolerance)

class Stage6ServiceMixin:
    def _ai_stage_advisory_enabled(self, attr_name: str) -> bool:
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
    ) -> Dict[str, Any]:
        return ai_advisory.request_stage_ai_advisory(
            self,
            stage_name,
            schema_text,
            observations,
            max_tokens=max_tokens,
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


    def _apply_stage6_bright_nebula_hdr_masked(self, params: Dict[str, Any]) -> None:
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
        result += faint_boost * faint_mask[None, :, :] * (1.0 - 0.85 * core_mask[None, :, :])

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

        restored = self._stage8_restore_rgb_like(source, np.clip(result, 0.0, 0.995))
        self._set_current_image_pixeldata(restored, label="stage6 bright-nebula HDR masked")


    def _execute_stage6_stretch_candidate(self, candidate: Dict[str, Any]) -> Tuple[bool, str]:
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
            if method == "bright_nebula_hdr_masked":
                self.cmd_with_check(
                    "asinh",
                    str(params.get("asinh_stretch", self.cfg.asinh_stretch)),
                    str(params.get("asinh_offset", self.cfg.asinh_offset)),
                )
                self._apply_stage6_bright_nebula_hdr_masked(params)
                return True, "bright_nebula_hdr_masked"
            if method == "ghs":
                self.cmd_with_check(
                    "autoghs", "-linked",
                    str(params.get("ghs_shadowsclip", self.cfg.ghs_shadowsclip)),
                    str(params.get("ghs_stretchamount", self.cfg.ghs_stretchamount)),
                )
                return True, "GHS"
            if method == "autostretch":
                self.cmd_with_check("autostretch")
                return True, "autostretch"
            return False, f"unsupported stage6 method: {method}"
        except (CommandError, SirilError, RuntimeError, ValueError) as e:
            return False, self._short_text(e, 180)


    def _validate_stage6_stretch_quality(
        self,
        baseline_quality: Optional[QualityMetrics],
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
        if baseline_quality and baseline_quality.median_star_size > 0.2 and metrics.median_star_size > 0:
            star_growth = metrics.median_star_size / max(baseline_quality.median_star_size, 1e-4)
            if star_growth > self.cfg.stage6_star_growth_ratio_max:
                issues.append(
                    f"star_size_growth {star_growth:.3f}>{self.cfg.stage6_star_growth_ratio_max:.3f}"
                )
        return len(issues) == 0, issues, metrics


    def _stage6_stretch_risk_score(
        self,
        metrics: Optional[QualityMetrics],
        issues: List[str],
        baseline_quality: Optional[QualityMetrics],
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
        if baseline_quality and baseline_quality.median_star_size > 0.2 and metrics.median_star_size > 0:
            star_growth = metrics.median_star_size / max(baseline_quality.median_star_size, 1e-4)
            if star_growth > self.cfg.stage6_star_growth_ratio_max:
                score += (star_growth - self.cfg.stage6_star_growth_ratio_max) * 8.0
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


    def _stage7_compact_stretch_candidates(
        self,
        baseline_quality: Optional[QualityMetrics],
        baseline_adaptive: Optional[Dict[str, Any]],
        baseline_pixel_stats: Optional[Dict[str, Any]] = None,
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
        cand_a_params = {"asinh_stretch": 2.2, "asinh_offset": 0.002}
        cand_b_params = {
            "asinh_stretch": 2.1,
            "asinh_offset": 0.002,
            "ghs_shadowsclip": -2.1,
            "ghs_stretchamount": 1.05,
        }

        if 0.0 < bg_median <= 0.005:
            severity = _clamp_float((0.005 - bg_median) / 0.003, 0.0, 1.0)
            cand_a_params = {
                "asinh_stretch": round(1.80 - 0.30 * severity, 3),
                "asinh_offset": round(0.005 + 0.003 * severity, 5),
            }
            cand_b_params = {
                "asinh_stretch": round(1.85 - 0.25 * severity, 3),
                "asinh_offset": round(0.005 + 0.002 * severity, 5),
                "ghs_shadowsclip": -2.1,
                "ghs_stretchamount": round(1.02 - 0.02 * severity, 3),
            }
            adaptation.update(
                {
                    "mode": "extreme_low_background",
                    "severity": severity,
                    "reason": "bg_median<=0.005; lower Asinh stretch and raise offset to avoid crushing faint background",
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

        pixel_stats = baseline_pixel_stats or {}
        try:
            p99 = float(pixel_stats.get("p99", 0.0) or 0.0)
            max_v = float(pixel_stats.get("max", 0.0) or 0.0)
        except (TypeError, ValueError):
            p99 = 0.0
            max_v = 0.0
        cap_candidates = [
            value
            for value in (p99 * 0.85, max_v * 0.80)
            if math.isfinite(value) and value > 0.0005
        ]
        if cap_candidates:
            offset_cap = max(0.0005, min(cap_candidates))
            capped: List[str] = []
            for name, params in (("cand_a", cand_a_params), ("cand_b", cand_b_params)):
                current_offset = float(params.get("asinh_offset", 0.002) or 0.002)
                if current_offset >= p99 and current_offset > offset_cap:
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

        candidates = [
            {
                "name": "cand_a",
                "stem": "stage7_cand_a",
                "method": "asinh",
                "params": cand_a_params,
                "adaptation": adaptation,
            },
            {
                "name": "cand_b",
                "stem": "stage7_cand_b",
                "method": "asinh_ghs",
                "params": cand_b_params,
                "adaptation": adaptation,
            },
        ]
        return candidates, adaptation


    def _run_stage6_ai_stretching(
        self,
        allow_ai: bool = True,
    ) -> Tuple[bool, bool, List[str], str]:
        messages: List[str] = []
        source_stem = "stage6_starless"
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
        messages.append("stage7 stretch candidates: stage7_cand_a, stage7_cand_b; preview=stage7_preview_ref")
        if allow_ai:
            messages.append("stage7 stretch ignores AI/policy expansion; fixed compact candidate set")

        if self.process_dir:
            for pattern in (
                "stage6_candidate_*.fit",
                "stage7_candidate_*.fit",
                "stage6_selected*.fit",
                "stage7_selected*.fit",
                "stage6_stretched.fit",
            ):
                for stale_path in self.process_dir.glob(pattern):
                    try:
                        stale_path.unlink()
                    except OSError as e:
                        self.log.warn(f"[Stage7] stale stretch candidate cleanup failed: {e}")

        preview_saved = False
        try:
            self.cmd_with_check("load", source_stem)
            self.cmd_with_check("autostretch", "-linked")
            preview_saved = self._save_stage_output("stage7_preview_ref")
        except (CommandError, SirilError) as e:
            messages.append(f"stage7 preview_ref failed: {self._short_text(e, 160)}")

        candidate_list, stretch_adaptation = self._stage7_compact_stretch_candidates(
            baseline_quality,
            baseline_adaptive,
            baseline_pixel_stats,
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
            stem = str(candidate["stem"])
            try:
                self.cmd_with_check("load", source_stem)
            except (CommandError, SirilError) as e:
                attempts.append(
                    {
                        "name": candidate["name"],
                        "file": f"{stem}.fit",
                        "method": candidate["method"],
                        "params": candidate["params"],
                        "adaptation": candidate.get("adaptation"),
                        "status": "failed",
                        "reason": self._short_text(e, 160),
                    }
                )
                continue

            ok, used_or_error = self._execute_stage6_stretch_candidate(candidate)
            if not ok:
                attempts.append(
                    {
                        "name": candidate["name"],
                        "file": f"{stem}.fit",
                        "method": candidate["method"],
                        "params": candidate["params"],
                        "adaptation": candidate.get("adaptation"),
                        "status": "failed",
                        "reason": used_or_error,
                    }
                )
                continue

            quality_ok, issues, metrics = self._validate_stage6_stretch_quality(baseline_quality)
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
            risk_score = self._stage6_stretch_risk_score(metrics, issues, baseline_quality)
            candidate_saved = self._save_stage_output(stem)
            attempt = {
                "name": candidate["name"],
                "file": f"{stem}.fit" if candidate_saved else None,
                "stem": stem if candidate_saved else None,
                "method": candidate["method"],
                "params": candidate["params"],
                "adaptation": candidate.get("adaptation"),
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
                "risk_score": risk_score,
                "allowed_as_final": bool(quality_ok),
            }
            attempts.append(attempt)
            if candidate_saved and quality_ok and (
                best_attempt is None
                or risk_score < float(best_attempt.get("risk_score", 1_000_000.0))
            ):
                best_attempt = attempt

        if best_attempt is None:
            best_attempt = next(
                (
                    attempt
                    for attempt in attempts
                    if attempt.get("status") == "ok" and attempt.get("stem")
                ),
                None,
            )
            if best_attempt is not None:
                best_attempt["explicit_fallback"] = True
                messages.append("stage7 selected degraded fallback because no candidate passed quality gate")

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
                },
                "baseline_adaptive": baseline_adaptive,
                "baseline_pixel_stats": baseline_pixel_stats,
                "stretch_adaptation": stretch_adaptation,
                "attempts": attempts,
                "selected": best_attempt,
            },
        )
        self._write_stage_json(
            "stretch_candidates_report.json",
            {
                "stage": "stage7_stretch",
                "input": f"{source_stem}.fit",
                "stretch_adaptation": stretch_adaptation,
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
                        "status": item.get("status"),
                        "diagnostics": item.get("diagnostics"),
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
                    }
                    if best_attempt
                    else None
                ),
            },
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
            f"fallback_used={str(bool(best_attempt.get('explicit_fallback'))).lower()}"
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
        return True, bool(best_attempt.get("explicit_fallback")), messages, selected_method
