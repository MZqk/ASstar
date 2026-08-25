"""Service mixins for StarunPostProcessor."""
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

import cosmic_clarity
import plugin_runner
import sasp_runner
import scunet_denoise
import scene_support
import syqon_starless
import stage7_quality
import stage7_repair
import stage7_stretch_metrics
import stage8_pixels
import ui_preview
from image_metrics import (
    _box_blur_gray,
    _clamp_float,
    _to_rgb_float_fullres,
    format_feature_summary,
    measure_quality_metrics,
)
from models import (
    ImageFeatures,
    QualityMetrics,
    StageResult,
    StarSeparationState,
)
from save_utils import save_stage_output, write_stage_json
from stage7_pixel_domain import canonicalize_stage7_pixels_01

try:
    from sirilpy.exceptions import CommandError, DataError, SirilError
except ImportError:
    CommandError = RuntimeError
    DataError = RuntimeError
    SirilError = RuntimeError

try:
    from image_feature_analyzer import analyze_image as analyze_adaptive_image
    from policy_selector import DEFAULT_POLICY, policy_for_profile
    from target_profiler import build_target_profile
except (ImportError, RuntimeError):
    analyze_adaptive_image = None
    DEFAULT_POLICY = {
        "policy_name": "generic_low_snr_safe",
        "stage7_stretch": {"fallback_candidate": "asinh_core_protect"},
    }
    policy_for_profile = None
    build_target_profile = None

ENV_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
ENV_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
ENV_DEBUG_MODE_KEY = "STARUN_DEBUG_MODE"
ENV_INPUT_MODE_KEY = "STARUN_INPUT_MODE"
INPUT_MODE_AUTO = "auto"
INPUT_MODE_LINEAR_RESUME = "stage5_linear_resume"
RESULT_BASENAME_TEMPLATE = (
    "$OBJECT:%s$_$STACKCNT:%d$x$EXPTIME:%d$sec"
    "_$DATE-OBS:dm12$_processed"
)
STAGE7_ASINH_STRETCH_MIN = 1.0
STAGE7_ASINH_STRETCH_MAX = 1000.0
STAGE7_CANDIDATE_RANKING_POLICY = "hard_gate_bounded_subject_brightness_v6"
STAGE7_CANDIDATE_RANKING_FIELDS = (
    "status",
    "hard_gate_eligibility",
    "technical_safety",
    "stretch_saturated_penalty",
    "subject_brightness_floor",
    "subject_brightness_goals_capped",
    "bounded_presentation_score",
    "preview_visibility_retention_capped_at_goal",
    "preview_subject_span_retention_capped_at_goal",
    "preview_saturation_median_retention_capped_at_goal",
    "preview_microcontrast_retention_capped_at_goal",
    "background_and_colour_safety_headroom",
    "advisory_count",
    "fixed_risk_score",
    "candidate_name",
)


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
        mask_pixels, _pixel_domain = canonicalize_stage7_pixels_01(starmask)
        mask_rgb = _to_rgb_float_fullres(mask_pixels)
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
    return stage7_stretch_metrics.solve_linked_mtf_midpoint(
        source_value,
        target_value,
    )


def _stage7_mtf_sample(
    value: float,
    shadows: float,
    midtones: float,
    highlights: float = 1.0,
) -> float:
    """Evaluate Siril's MTF mapping for calibration diagnostics."""
    try:
        return stage7_stretch_metrics.linked_mtf_sample(
            value,
            shadows,
            midtones,
            highlights,
        )
    except (TypeError, ValueError, FloatingPointError):
        return 0.0


def _stage7_preview_calibrated_stretch(
    baseline_stats: Dict[str, Any],
    preview_stats: Dict[str, Any],
    *,
    offset: float,
    preview_scale: float,
    fallback: float,
    highlight_scale: float = 0.90,
    stretch_max: float = STAGE7_ASINH_STRETCH_MAX,
    target_p50_max: float = 0.22,
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

    target_p50_ceiling = _clamp_float(target_p50_max, 0.12, 0.35)
    target_p50 = _clamp_float(
        preview_p50 * preview_scale,
        0.025,
        target_p50_ceiling,
    )
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
        "target_p50_ceiling": target_p50_ceiling,
        "target_p99": target_p99,
        "median_limited_stretch": median_limited,
        "highlight_limited_stretch": highlight_limited,
        "calibrated_stretch": calibrated,
        "stretch_max": float(stretch_max),
        "predicted_p50": _stage7_asinh_sample(baseline_p50, calibrated, offset),
        "predicted_p99": _stage7_asinh_sample(baseline_p99, calibrated, offset),
    }


def _stage7_rebase_manual_asinh_calibration(
    calibration: Dict[str, Any],
    baseline_stats: Dict[str, Any],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Rebase the preview target onto the explicit Asinh parameter contract."""
    if not isinstance(calibration, dict) or not calibration:
        return {}
    try:
        baseline_p50 = float(
            calibration.get("baseline_p50", baseline_stats.get("p50", 0.0))
            or 0.0
        )
        baseline_p99 = float(
            calibration.get("baseline_p99", baseline_stats.get("p99", 0.0))
            or 0.0
        )
        stretch = float(params.get("asinh_stretch", 0.0) or 0.0)
        offset = float(params.get("asinh_offset", 0.0) or 0.0)
    except (TypeError, ValueError):
        return {}
    if not all(
        math.isfinite(value)
        for value in (baseline_p50, baseline_p99, stretch, offset)
    ) or baseline_p50 <= 0.0 or baseline_p99 <= baseline_p50 or stretch <= 0.0:
        return {}

    predicted_p50 = _stage7_asinh_sample(baseline_p50, stretch, offset)
    predicted_p99 = _stage7_asinh_sample(baseline_p99, stretch, offset)
    if predicted_p50 <= 0.0 or predicted_p99 <= predicted_p50:
        return {}
    rebased = dict(calibration)
    rebased.update(
        {
            "calibration_method": "manual_contract_rebased",
            "target_contract": "explicit_asinh_prediction",
            "auto_target_p50": calibration.get("target_p50"),
            "auto_target_p99": calibration.get("target_p99"),
            "target_p50": predicted_p50,
            "target_p99": predicted_p99,
            "calibrated_stretch": stretch,
            "manual_asinh_offset": offset,
            "predicted_p50": predicted_p50,
            "predicted_p99": predicted_p99,
        }
    )
    return rebased


def _stage7_preview_target_attainment(
    candidate_name: str,
    pixel_stats: Dict[str, Any],
    adaptation: Dict[str, Any],
    *,
    min_ratio: float = 0.90,
    hard_min_ratio: float = 0.80,
    max_ratio: float = 1.50,
    advisory_multiplier: float = 1.0,
) -> Dict[str, Any]:
    """Keep calibrated candidates within the allowed preview P50 target band."""
    preview_calibration = adaptation.get("preview_calibration") or {}
    calibration_key = {
        "cand_a": "candidate_a",
        "cand_b": "candidate_b",
        "cand_display70": "candidate_display70",
        "cand_display82": "candidate_display82",
        "cand_display86": "candidate_display86",
        "cand_display90": "candidate_display90",
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
    multiplier = _clamp_float(advisory_multiplier, 1.0, 2.0)
    hard_minimum = _clamp_float(hard_min_ratio, 0.25, minimum)
    hard_maximum = maximum * multiplier
    ratio = actual_p50 / target_p50
    saturated = bool(
        stretch_max > 0.0
        and calibrated_stretch >= stretch_max * 0.999
        and predicted_p50 < target_p50
    )
    issues: List[str] = []
    advisories: List[str] = []
    if ratio < minimum:
        message = (
            "preview_target_p50_ratio "
            f"{ratio:.3f}<{minimum:.3f} "
            f"(actual={actual_p50:.5f}, target={target_p50:.5f}, "
            f"stretch_saturated={str(saturated).lower()})"
        )
        (issues if ratio < hard_minimum else advisories).append(message)
    if ratio > maximum:
        message = (
            "preview_target_p50_ratio_above_max "
            f"{ratio:.3f}>{maximum:.3f} "
            f"(actual={actual_p50:.5f}, target={target_p50:.5f})"
        )
        (issues if ratio > hard_maximum else advisories).append(message)
    accepted = not issues
    return {
        "status": "poor" if issues else "advisory" if advisories else "ok",
        "accepted": accepted,
        "candidate": str(candidate_name),
        "actual_p50": actual_p50,
        "target_p50": target_p50,
        "attainment_ratio": ratio,
        "minimum_ratio": minimum,
        "maximum_ratio": maximum,
        "hard_minimum_ratio": hard_minimum,
        "hard_maximum_ratio": hard_maximum,
        "advisory_multiplier": multiplier,
        "calibrated_stretch": calibrated_stretch,
        "stretch_max": stretch_max,
        "stretch_saturated": saturated,
        "issues": issues,
        "advisories": advisories,
    }


def _stage7_matched_domain_transfer_contract(
    selected: Optional[Dict[str, Any]],
    closed_form_mtf_reference: Dict[str, Any],
) -> Dict[str, Any]:
    """Serialize the exact Stage 9 display-domain transfer for the winner."""

    base: Dict[str, Any] = {
        "schema": (
            stage7_stretch_metrics.STAGE7_MATCHED_DOMAIN_TRANSFER_SCHEMA
        ),
        "status": "unavailable",
        "selected_candidate_id": (
            str(selected.get("name") or "") if isinstance(selected, dict) else None
        ),
    }
    if not isinstance(selected, dict):
        return {**base, "reason": "Stage7 selected candidate is unavailable"}
    selected_name = str(selected.get("name") or "")
    selected_method = str(selected.get("method") or "")
    tone_candidate = selected
    tone_candidate_id = selected_name
    if selected_method == "vivid_safe_chroma":
        parent = dict((selected.get("params") or {}).get("parent_candidate") or {})
        if parent:
            tone_candidate = parent
            tone_candidate_id = str(
                (selected.get("params") or {}).get("parent_name")
                or selected.get("tone_parent")
                or parent.get("name")
                or ""
            )
    tone_method = str(tone_candidate.get("method") or "")
    calibration = copy.deepcopy(
        (tone_candidate.get("params") or {}).get("calibration") or {}
    )
    calibration_method = str(calibration.get("method") or "")
    if (
        tone_method in stage7_stretch_metrics.DISPLAY_LUT_METHODS
        or calibration_method in stage7_stretch_metrics.DISPLAY_LUT_METHODS
        or tone_candidate_id.startswith("cand_display")
    ):
        if (
            tone_method in stage7_stretch_metrics.DISPLAY_LUT_METHODS
            and calibration_method in stage7_stretch_metrics.DISPLAY_LUT_METHODS
            and tone_method != calibration_method
        ):
            return {
                **base,
                "method": tone_method,
                "reason": (
                    "selected Display90 candidate method does not match its "
                    "calibration semantics"
                ),
            }
        display_method = calibration_method or tone_method
        try:
            _lut, lut_contract = (
                stage7_stretch_metrics.rebuild_display90_linked_lut(
                    calibration
                )
            )
        except (KeyError, TypeError, ValueError, FloatingPointError) as error:
            return {
                **base,
                "method": display_method or None,
                "reason": f"selected Display90 contract is invalid: {error}",
            }
        return {
            **base,
            "schema": (
                stage7_stretch_metrics.STAGE7_MATCHED_DOMAIN_TRANSFER_SCHEMA_V1
                if calibration.get("schema")
                == stage7_stretch_metrics.DISPLAY90_STRETCH_SCHEMA_V1
                else stage7_stretch_metrics.STAGE7_MATCHED_DOMAIN_TRANSFER_SCHEMA_V2
            ),
            "status": "active",
            "method": display_method,
            "source": "selected_stage7_candidate",
            "tone_candidate_id": tone_candidate_id,
            "calibration": calibration,
            "lut_contract": lut_contract,
            "fallback_to_linked_mtf_allowed": False,
        }

    reference = dict(closed_form_mtf_reference or {})
    active_anchor = dict(reference.get("active_anchor") or {})
    params = dict(active_anchor.get("params") or {})
    if reference.get("status") != "active" or not params:
        return {
            **base,
            "method": "closed_form_linked_mtf",
            "reason": "closed-form linked-MTF reference is unavailable",
        }
    return {
        **base,
        "status": "active",
        "method": "closed_form_linked_mtf",
        "source": "candidate_b_closed_form_reference",
        "params": params,
        "reference_schema": reference.get("schema"),
        "fallback_to_linked_mtf_allowed": True,
    }


def _stage7_candidates_for_policy(
    candidates: List[Dict[str, Any]],
    policy: str,
) -> List[Dict[str, Any]]:
    """Apply the signed Stage7 primary-candidate policy without adding IDs."""

    normalized = str(policy or "auto_display90").strip().lower()
    allowed = {
        "auto_display90": {
            "cand_a",
            "cand_b",
            "cand_display70",
            "cand_display82",
            "cand_display86",
            "cand_display90",
        },
        "auto_dual": {"cand_a", "cand_b"},
        "candidate_a_only": {"cand_a"},
        "candidate_b_only": {"cand_b"},
        "display90_only": {"cand_display90"},
    }.get(normalized)
    if allowed is None:
        return list(candidates)
    return [
        candidate
        for candidate in candidates
        if str(candidate.get("name") or "") in allowed
    ]


def _stage7_preview_retention(
    pixel_stats: Dict[str, Any],
    quality_metrics: Dict[str, Any],
    preview_pixel_stats: Dict[str, Any],
    preview_quality_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Measure subject/display retention against the linked preview reference."""

    candidate_distribution = dict(pixel_stats or {})
    preview_distribution = dict(preview_pixel_stats or {})
    try:
        candidate_distribution["subject_span"] = max(
            0.0,
            float(candidate_distribution.get("p99", 0.0) or 0.0)
            - float(candidate_distribution.get("p50", 0.0) or 0.0),
        )
        preview_distribution["subject_span"] = max(
            0.0,
            float(preview_distribution.get("p99", 0.0) or 0.0)
            - float(preview_distribution.get("p50", 0.0) or 0.0),
        )
    except (TypeError, ValueError):
        candidate_distribution.pop("subject_span", None)
        preview_distribution.pop("subject_span", None)

    metric_sources = {
        "p50": (candidate_distribution, preview_distribution, "p50"),
        "object_signal": (
            candidate_distribution,
            preview_distribution,
            "object_signal_ratio",
        ),
        "visibility": (
            candidate_distribution,
            preview_distribution,
            "safe_preview_visibility_score",
        ),
        "subject_span": (
            candidate_distribution,
            preview_distribution,
            "subject_span",
        ),
        "saturation_median": (
            quality_metrics,
            preview_quality_metrics,
            "saturation_median",
        ),
        "saturation_p95": (
            quality_metrics,
            preview_quality_metrics,
            "saturation_p95",
        ),
        "microcontrast": (
            quality_metrics,
            preview_quality_metrics,
            "microcontrast",
        ),
    }
    measured: Dict[str, Dict[str, Any]] = {}
    for name, (candidate_source, preview_source, key) in metric_sources.items():
        try:
            candidate_value = float(candidate_source.get(key))
            preview_value = float(preview_source.get(key))
        except (AttributeError, TypeError, ValueError):
            measured[name] = {
                "available": False,
                "candidate": None,
                "preview": None,
                "ratio": None,
                "ranking_ratio": None,
            }
            continue
        available = bool(
            math.isfinite(candidate_value)
            and math.isfinite(preview_value)
            and candidate_value >= 0.0
            and preview_value > 1e-12
        )
        ratio = candidate_value / preview_value if available else None
        measured[name] = {
            "available": available,
            "candidate": candidate_value if math.isfinite(candidate_value) else None,
            "preview": preview_value if math.isfinite(preview_value) else None,
            "ratio": ratio,
            # Retention above the preview is not rewarded; over-processing stays
            # governed by the existing brightness, background, core and clip gates.
            "ranking_ratio": min(max(float(ratio), 0.0), 1.0) if ratio is not None else None,
        }

    available_count = sum(
        1 for item in measured.values() if bool(item.get("available"))
    )
    return {
        "schema": "starun.stage7-preview-retention.v2",
        "status": (
            "active"
            if available_count == len(measured)
            else "partial"
            if available_count
            else "unavailable"
        ),
        "role": "selection_signal",
        "hard_gate_metrics": ["visibility"],
        "selection_metrics": [
            "visibility",
            "object_signal",
            "subject_span",
            "saturation_p95",
            "microcontrast",
        ],
        "available_count": available_count,
        "metrics": measured,
    }


def stage7_effective_bg_median_min(configured_min: float) -> float:
    """Return the Stage6 dark-floor gate with room for FITS sampling noise."""
    configured = max(float(configured_min), 1e-4)
    tolerance = min(0.0005, configured * 0.025)
    return max(1e-4, configured - tolerance)


def _stage7_core_gate_failures(attempt: Dict[str, Any]) -> List[str]:
    local_gates = (
        (attempt.get("target_local_quality") or {}).get("quality_gates") or {}
    )
    return [
        str(name)
        for name, gate in local_gates.items()
        if str(name).startswith("local_core_")
        and bool((gate or {}).get("hard_failed", False))
    ]


def _stage7_all_saved_candidates_fail_core_gates(
    saved_attempts: List[Dict[str, Any]],
    *,
    strict_target: bool,
) -> Tuple[bool, List[str]]:
    """Return the non-overridable Stage 7 core rejection decision."""
    if not strict_target or not saved_attempts:
        return False, []
    failures = [_stage7_core_gate_failures(attempt) for attempt in saved_attempts]
    if any(not attempt_failures for attempt_failures in failures):
        return False, []
    return True, list(
        dict.fromkeys(
            failure
            for attempt_failures in failures
            for failure in attempt_failures
        )
    )

class Stage6ServiceMixin:
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
        if self.cfg.stage2_base_crop_margin >= 0.055:
            return (
                "adaptive edge crop skipped "
                "(edge_black="
                f"{feat.edge_black_ratio:.3f}, stage2 base crop already high)"
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


    def _stage7_target_hint_lower(self) -> str:
        parts: List[str] = []
        for value in (self.source_file, self.work_dir):
            if isinstance(value, Path):
                parts.extend([value.name, value.stem])
        return " ".join(parts).lower()


    def _stage7_stretch_candidate(
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


    def _apply_stage7_bright_nebula_hdr_masked(
        self,
        params: Dict[str, Any],
        starmask_image_data: Optional[np.ndarray] = None,
    ) -> None:
        image_data = self.siril.get_image_pixeldata(preview=False)
        if image_data is None:
            raise RuntimeError("image buffer is empty")
        source = np.asarray(image_data)
        canonical_source, _pixel_domain = canonicalize_stage7_pixels_01(source)
        rgb = _to_rgb_float_fullres(canonical_source)
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


    def _execute_stage7_stretch_candidate(
        self,
        candidate: Dict[str, Any],
        *,
        starmask_image_data: Optional[np.ndarray] = None,
        frozen_masks: Optional[Dict[str, Any]] = None,
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
                    "-clipmode=rgbblend",
                )
                return True, "Asinh"
            if method == "asinh_ghs":
                self.cmd_with_check(
                    "asinh",
                    str(params.get("asinh_stretch", self.cfg.asinh_stretch)),
                    str(params.get("asinh_offset", self.cfg.asinh_offset)),
                    "-clipmode=rgbblend",
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
                    "-clipmode=rgbblend",
                )
                self._apply_stage7_bright_nebula_hdr_masked(
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
            if method in {"iterative_masked_mtf", "dual_stage_mtf_ghs"}:
                image_data = self.siril.get_image_pixeldata(preview=False)
                if image_data is None:
                    raise RuntimeError("image buffer is empty")
                source = np.asarray(image_data)
                calibration = dict(params.get("calibration") or {})
                if str(calibration.get("method") or "") != method:
                    raise ValueError(
                        "conditional stretch method/calibration mismatch"
                    )
                mapped_rgb = (
                    stage7_stretch_metrics.apply_conditional_linked_rgb_stretch(
                        source,
                        calibration,
                    )
                )
                restored = self._stage8_restore_rgb_like(source, mapped_rgb)
                self._set_current_image_pixeldata(
                    restored,
                    label=f"Stage7 {method}",
                )
                return True, method
            if method in stage7_stretch_metrics.DISPLAY_LUT_METHODS:
                image_data = self.siril.get_image_pixeldata(preview=False)
                if image_data is None:
                    raise RuntimeError("image buffer is empty")
                source = np.asarray(image_data)
                calibration = dict(params.get("calibration") or {})
                if str(calibration.get("method") or "") != method:
                    raise ValueError(
                        "Display90 stretch method/calibration mismatch"
                    )
                mapped_rgb = (
                    stage7_stretch_metrics.apply_display90_linked_rgb_stretch(
                        source,
                        calibration,
                    )
                )
                restored = self._stage8_restore_rgb_like(source, mapped_rgb)
                self._set_current_image_pixeldata(
                    restored,
                    label="Stage7 Display90 linked LUT",
                )
                return True, method
            if method == "vivid_safe_chroma":
                parent = dict(params.get("parent_candidate") or {})
                if not parent or str(parent.get("method") or "") == method:
                    raise ValueError("vivid-safe parent candidate is invalid")
                replay_ok, replay_used = self._execute_stage7_stretch_candidate(
                    parent,
                    starmask_image_data=starmask_image_data,
                    frozen_masks=frozen_masks,
                )
                if not replay_ok:
                    raise RuntimeError(
                        f"vivid-safe parent replay failed: {replay_used}"
                    )
                image_data = self.siril.get_image_pixeldata(preview=False)
                if image_data is None:
                    raise RuntimeError("vivid-safe parent pixel buffer is empty")
                rendered_rgb, rendition = (
                    stage7_stretch_metrics.apply_subject_chroma_rendition(
                        np.asarray(image_data),
                        frozen_masks,
                        factor=float(params.get("factor", 1.08)),
                    )
                )
                restored = self._stage8_restore_rgb_like(
                    np.asarray(image_data),
                    rendered_rgb,
                )
                self._set_current_image_pixeldata(
                    restored,
                    label="Stage7 vivid-safe subject chroma",
                )
                self._stage7_last_vivid_chroma_execution = rendition
                return True, "vivid_safe_chroma"
            if method == "autostretch":
                self.cmd_with_check("autostretch")
                return True, "autostretch"
            return False, f"unsupported stage6 method: {method}"
        except (
            CommandError,
            SirilError,
            RuntimeError,
            KeyError,
            TypeError,
            ValueError,
            FloatingPointError,
        ) as e:
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


    def _stage7_current_pixel_snapshot(
        self,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Read the active Siril image once in the canonical Stage 7 domain."""
        image_data = self.siril.get_image_pixeldata(preview=False)
        if image_data is None:
            raise RuntimeError("Stage7 image buffer is empty")
        pixels, pixel_domain = canonicalize_stage7_pixels_01(image_data)
        pixel_stats = self._pixel_distribution_stats_from_canonical(
            pixels,
            pixel_domain,
        )
        return pixels, pixel_stats


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
        preview_pixel_stats: Dict[str, Any],
        preview_quality_metrics: Dict[str, Any],
        preview_rendition_metrics: Optional[Dict[str, Any]] = None,
        target_local_reference_data: Optional[np.ndarray] = None,
        target_local_reference_available: bool = True,
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
            "transform_semantics": (
                stage7_stretch_metrics.build_siril_stretch_semantics(
                    str(candidate.get("method") or ""),
                    dict(candidate.get("params") or {}),
                )
            ),
        }
        try:
            self.cmd_with_check("load", source_stem)
        except (CommandError, SirilError) as error:
            return {
                **common,
                "status": "failed",
                "reason": self._short_text(error, 160),
            }

        ok, used_or_error = self._execute_stage7_stretch_candidate(
            candidate,
            starmask_image_data=starmask_image_data,
            frozen_masks=frozen_background_masks,
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
        color_vector_reference: Dict[str, Any] = {
            "schema": "starun.stage7-color-vector-reference.v1",
            "status": "unavailable",
            "role": "report_only",
            "enforced": False,
            "participates_in_selection": False,
            "reason": "Stage7 color reference inputs unavailable",
        }
        multiscale_contrast_reference: Dict[str, Any] = {
            "schema": stage7_stretch_metrics.MULTISCALE_CONTRAST_SCHEMA,
            "status": "unavailable",
            "role": "report_only",
            "enforced": False,
            "participates_in_selection": False,
            "mas_equivalent": False,
            "reason": "Stage7 multiscale reference inputs unavailable",
        }
        candidate_image_data: Optional[np.ndarray] = None
        pixel_stats: Dict[str, Any] = {}
        transform_loss: Dict[str, Any] = {
            "schema": stage7_stretch_metrics.TRANSFORM_LOSS_SCHEMA,
            "status": "unavailable",
            "role": "report_only",
            "enforced": False,
            "participates_in_selection": False,
            "reason": "Stage7 source/candidate pixels unavailable",
        }
        try:
            candidate_image_data, pixel_stats = (
                self._stage7_current_pixel_snapshot()
            )
            if baseline_image_data is not None:
                transform_loss = stage7_stretch_metrics.assess_transform_loss(
                    baseline_image_data,
                    candidate_image_data,
                    method=str(candidate.get("method") or ""),
                    params=dict(candidate.get("params") or {}),
                    background_mask=(
                        frozen_background_masks.get("background_mask")
                        if isinstance(frozen_background_masks, dict)
                        else None
                    ),
                )
                local_quality = stage7_stretch_metrics.assess_target_local_stretch(
                    (
                        target_local_reference_data
                        if target_local_reference_data is not None
                        else baseline_image_data
                    ),
                    candidate_image_data,
                    str(
                        target_stretch.get("target_type")
                        or "generic_low_snr_safe"
                    ),
                    self.cfg,
                    target_profile=getattr(self, "target_profile", None),
                    starmask=starmask_image_data,
                    frozen_reference_available=target_local_reference_available,
                    valid_mask=(
                        frozen_background_masks.get("shared_valid_mask")
                        if isinstance(frozen_background_masks, dict)
                        else None
                    ),
                    original_saturation_map=(
                        frozen_background_masks.get("original_saturation_map")
                        if isinstance(frozen_background_masks, dict)
                        else None
                    ),
                )
                color_vector_reference = (
                    stage7_stretch_metrics.assess_rec709_vector_color_reference(
                        baseline_image_data,
                        candidate_image_data,
                    )
                )
                multiscale_contrast_reference = (
                    stage7_stretch_metrics.assess_multiscale_contrast_reference(
                        baseline_image_data,
                        candidate_image_data,
                    )
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
        ) as error:
            return {
                **common,
                "status": "failed",
                "reason": (
                    "Stage7 pixel-domain validation failed: "
                    f"{self._short_text(error, 160)}"
                ),
            }

        use_starless_structure_gate = (
            starless_structure_quality.get("status") in {"ok", "rejected"}
        )
        enforce_star_growth = self._stage7_should_enforce_star_growth(
            use_starless_structure_gate
        )
        quality_ok, issues, metrics = self._validate_stage7_stretch_quality(
            baseline_quality,
            enforce_star_growth=enforce_star_growth,
            current_image_data=candidate_image_data,
        )
        advisories = list(
            getattr(metrics, "_stage7_quality_advisories", [])
            if metrics is not None
            else []
        )
        quality_gates: Dict[str, Any] = {
            "base_quality": dict(
                getattr(metrics, "_stage7_quality_gates", {})
                if metrics is not None
                else {}
            )
        }
        transform_loss_gate = self._stage7_transform_loss_gate(transform_loss)
        color_vector_gate = self._stage7_color_vector_gate(
            color_vector_reference
        )
        transform_loss = dict(transform_loss)
        transform_loss.update(
            role="technical_quality_gate",
            enforced=True,
            participates_in_selection=True,
        )
        color_vector_reference = dict(color_vector_reference)
        color_vector_reference.update(
            role="appearance_quality_gate",
            enforced=True,
            participates_in_selection=True,
        )
        if not bool(transform_loss_gate.get("accepted", True)):
            quality_ok = False
            issues = [
                *issues,
                *list(transform_loss_gate.get("issues") or []),
            ]
        if not bool(color_vector_gate.get("accepted", True)):
            quality_ok = False
            issues = [
                *issues,
                *list(color_vector_gate.get("issues") or []),
            ]
        advisories.extend(transform_loss_gate.get("advisories") or [])
        advisories.extend(color_vector_gate.get("advisories") or [])
        quality_gates["transform_loss"] = transform_loss_gate
        quality_gates["color_vector"] = color_vector_gate
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
        quality_metrics_dict = asdict(metrics) if metrics else {}
        preview_retention = _stage7_preview_retention(
            pixel_stats,
            quality_metrics_dict,
            preview_pixel_stats,
            preview_quality_metrics,
        )
        candidate_rendition_metrics = (
            stage7_stretch_metrics.measure_frozen_rendition_metrics(
                candidate_image_data,
                frozen_background_masks,
            )
            if candidate_image_data is not None
            else {
                "schema": stage7_stretch_metrics.RENDITION_METRICS_SCHEMA,
                "status": "unavailable",
                "metrics": {},
                "reason": "candidate pixels unavailable",
            }
        )
        rendition_metrics = {
            "candidate": candidate_rendition_metrics,
            "preview": dict(preview_rendition_metrics or {}),
            "retention": stage7_stretch_metrics.rendition_metric_retention(
                candidate_rendition_metrics,
                dict(preview_rendition_metrics or {}),
            ),
        }
        subject_brightness = (
            stage7_stretch_metrics.subject_brightness_selection(
                candidate_rendition_metrics,
                dict(preview_rendition_metrics or {}),
                profile_name=str(target_stretch.get("name") or ""),
            )
        )
        quality_gates["subject_brightness"] = subject_brightness
        if not bool(subject_brightness.get("formal_floor_passed", False)):
            quality_ok = False
            issues = [
                *issues,
                "stage7_subject_brightness_floor_unmet",
            ]
        quality_ok, issues, visibility_gate = (
            self._stage7_apply_candidate_visibility_gate(
                quality_ok,
                issues,
                pixel_stats,
                target_stretch,
                preview_retention,
            )
        )
        advisories.extend(visibility_gate.get("advisories") or [])
        quality_gates["visibility"] = visibility_gate.get("quality_gate")
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
                0.90,
            ),
            hard_min_ratio=getattr(
                self.cfg,
                "stage7_preview_target_p50_hard_min_ratio",
                0.80,
            ),
            max_ratio=getattr(
                self.cfg,
                "stage7_preview_target_p50_max_ratio",
                1.50,
            ),
            advisory_multiplier=(
                stage7_quality.stage7_9_quality_advisory_multiplier(
                    self.cfg
                )
            ),
        )
        if not bool(preview_target_attainment.get("accepted", True)):
            quality_ok = False
            issues = [
                *issues,
                *list(preview_target_attainment.get("issues") or []),
            ]
        advisories.extend(preview_target_attainment.get("advisories") or [])
        quality_gates["preview_target_attainment"] = {
            "hard_minimum_ratio": preview_target_attainment.get(
                "hard_minimum_ratio"
            ),
            "hard_maximum_ratio": preview_target_attainment.get(
                "hard_maximum_ratio"
            ),
            "advisory_multiplier": preview_target_attainment.get(
                "advisory_multiplier"
            ),
        }

        mtf_reference_quality: Dict[str, Any] = {
            "status": "not_applicable",
            "accepted": True,
            "role": "reference_anchor",
            "method": candidate.get("method"),
            "issues": [],
            "metrics": {},
        }
        if str(candidate.get("method") or "") == "linked_mtf":
            mtf_reference_quality = (
                stage7_stretch_metrics.assess_closed_form_mtf_conformance(
                    dict(candidate.get("params") or {}),
                    pixel_stats.get("p50"),
                    relative_error_max=getattr(
                        self.cfg,
                        "stage7_mtf_reference_p50_relative_error_max",
                        0.05,
                    ),
                    absolute_error_max=getattr(
                        self.cfg,
                        "stage7_mtf_reference_p50_absolute_error_max",
                        0.005,
                    ),
                )
            )
            if not bool(mtf_reference_quality.get("accepted", False)):
                quality_ok = False
                issues = [
                    *issues,
                    *list(mtf_reference_quality.get("issues") or []),
                ]

        display90_curve_quality: Dict[str, Any] = {
            "status": "not_applicable",
            "accepted": True,
            "issues": [],
            "method": candidate.get("method"),
        }
        if str(candidate.get("method") or "") in (
            stage7_stretch_metrics.DISPLAY_LUT_METHODS
        ):
            display90_curve_quality = (
                stage7_stretch_metrics.assess_display90_curve_conformance(
                    dict(
                        (candidate.get("params") or {}).get("calibration")
                        or {}
                    ),
                    candidate_image_data,
                    relative_error_max=getattr(
                        self.cfg,
                        "stage7_mtf_reference_p50_relative_error_max",
                        0.05,
                    ),
                    absolute_error_max=getattr(
                        self.cfg,
                        "stage7_mtf_reference_p50_absolute_error_max",
                        0.005,
                    ),
                )
            )
            if not bool(display90_curve_quality.get("accepted", False)):
                quality_ok = False
                issues = [
                    *issues,
                    *list(display90_curve_quality.get("issues") or []),
                ]
            quality_gates["display90_curve_conformance"] = {
                "relative_error_max": display90_curve_quality.get(
                    "relative_error_max"
                ),
                "absolute_error_max": display90_curve_quality.get(
                    "absolute_error_max"
                ),
            }

        # Every candidate is measured on the exact same Stage 6-derived masks.
        # Rebuilding signal masks after nonlinear stretching made strong curves
        # shrink their own measured background and receive an unfair advantage.
        candidate_background_masks = frozen_background_masks
        candidate_signal_exclusion_applied = bool(
            isinstance(frozen_background_masks, dict)
            and frozen_background_masks.get("background_mask") is not None
            and any(
                frozen_background_masks.get(mask_name) is not None
                for mask_name in (
                    "subject_mask",
                    "core_mask",
                    "nebula_mask",
                    "faint_nebula_mask",
                    "galaxy_signal_mask",
                    "star_mask",
                )
            )
        )

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
        display90_background_reference = (
            self._stage7_display90_background_reference(
                candidate,
                source_stem=source_stem,
                baseline_image_data=baseline_image_data,
                background_masks=candidate_background_masks,
                signal_exclusion_applied=candidate_signal_exclusion_applied,
                curve_quality=display90_curve_quality,
                non_background_hard_gates_accepted=bool(
                    quality_ok
                    and local_quality.get("accepted", True)
                    and starless_structure_quality.get("accepted", True)
                ),
            )
        )
        background_quality_gate = self._stage7_stretch_background_gate(
            baseline_background_quality,
            candidate_background_quality,
            candidate_name=name,
            candidate_method=str(candidate.get("method") or ""),
            display90_reference=display90_background_reference,
        )
        if not bool(background_quality_gate.get("accepted", False)):
            quality_ok = False
            issues = [
                *issues,
                *list(background_quality_gate.get("issues") or []),
            ]
        advisories.extend(background_quality_gate.get("advisories") or [])
        quality_gates["background"] = background_quality_gate.get(
            "quality_gates"
        )
        if not bool(local_quality.get("accepted", True)):
            quality_ok = False
            issues = [*issues, *list(local_quality.get("issues") or [])]
        advisories.extend(local_quality.get("advisories") or [])
        quality_gates["target_local"] = local_quality.get("quality_gates")
        if not bool(starless_structure_quality.get("accepted", True)):
            quality_ok = False
            issues = [
                *issues,
                *list(starless_structure_quality.get("issues") or []),
            ]
        advisories.extend(starless_structure_quality.get("advisories") or [])
        quality_gates["starless_structure"] = starless_structure_quality.get(
            "quality_gates"
        )
        risk_score = self._stage7_stretch_risk_score(
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
        attempt = {
            **common,
            "file": f"{stem}.fit" if candidate_saved else None,
            "stem": stem if candidate_saved else None,
            "status": "ok" if candidate_saved else "failed",
            "used": used_or_error,
            "quality_ok": quality_ok,
            "diagnostics": issues,
            "advisories": list(dict.fromkeys(str(item) for item in advisories)),
            "quality_gates": quality_gates,
            "metrics": quality_metrics_dict or None,
            "adaptive_metrics": (
                self._adaptive_features_from_image(candidate_image_data)
                if candidate_image_data is not None
                and hasattr(self, "_adaptive_features_from_image")
                else None
            ),
            "pixel_stats": pixel_stats,
            "preview_retention": preview_retention,
            "rendition_metrics": rendition_metrics,
            "subject_brightness_selection": subject_brightness,
            "visibility_gate": visibility_gate,
            "preview_target_attainment": preview_target_attainment,
            "mtf_reference_quality": mtf_reference_quality,
            "display90_curve_quality": display90_curve_quality,
            "color_vector_reference": color_vector_reference,
            "color_vector_gate": color_vector_gate,
            "multiscale_contrast_reference": multiscale_contrast_reference,
            "target_local_quality": local_quality,
            "starless_structure_quality": starless_structure_quality,
            "background_quality_gate": background_quality_gate,
            "transform_loss": transform_loss,
            "transform_loss_gate": transform_loss_gate,
            "risk_score": risk_score,
            "allowed_as_final": bool(quality_ok),
        }
        attempt["technical_safe"] = self._stage7_candidate_is_technically_safe(
            attempt
        )
        attempt["presentation_score"] = self._stage7_presentation_score(
            attempt
        )
        return attempt


    def _stage7_candidate_visibility_gate(
        self,
        pixel_stats: Dict[str, Any],
        target_stretch: Dict[str, Any],
        preview_retention: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Require absolute and preview-relative subject visibility for all targets."""
        profile_name = str(target_stretch.get("name") or "")
        minimum = float(
            getattr(self.cfg, "stage7_diffuse_visibility_score_min", 0.08)
        )
        try:
            score = float(
                pixel_stats.get("safe_preview_visibility_score", 0.0) or 0.0
            )
        except (TypeError, ValueError):
            score = 0.0
        if not math.isfinite(score):
            score = 0.0
        absolute_gate = stage7_quality.stage7_9_lower_quality_gate(
            self.cfg,
            value=score,
            accepted_limit=minimum,
        )
        retention_metrics = preview_retention.get("metrics") or {}
        visibility_retention = retention_metrics.get("visibility") or {}
        relative_available = bool(visibility_retention.get("available"))
        relative_ratio = visibility_retention.get("ratio")
        relative_minimum = float(
            getattr(
                self.cfg,
                "stage7_preview_visibility_retention_min",
                0.60,
            )
        )
        if profile_name == "star_colour_preserve":
            # Cluster candidates are judged on preserved stellar visibility,
            # not on retaining a diffuse-nebula preview lift.
            relative_minimum = min(relative_minimum, 0.18)
        if relative_available:
            relative_gate = stage7_quality.stage7_9_lower_quality_gate(
                self.cfg,
                value=float(relative_ratio),
                accepted_limit=relative_minimum,
            )
        else:
            relative_gate = {
                "status": "unavailable",
                "advisory": False,
                "hard_failed": False,
                "value": None,
                "accepted_limit": relative_minimum,
                "hard_limit": None,
                "severity_ratio": None,
                "advisory_multiplier": (
                    stage7_quality.stage7_9_quality_advisory_multiplier(
                        self.cfg
                    )
                ),
            }

        issues: List[str] = []
        advisories: List[str] = []
        absolute_message = (
            "visibility_score_below_minimum "
            f"{score:.3f}<{minimum:.3f}"
        )
        if bool(absolute_gate.get("hard_failed")):
            issues.append(absolute_message)
        elif bool(absolute_gate.get("advisory")):
            advisories.append(absolute_message + " advisory")
        if relative_available:
            relative_message = (
                "preview_visibility_retention_below_minimum "
                f"{float(relative_ratio):.3f}<{relative_minimum:.3f}"
            )
            if bool(relative_gate.get("hard_failed")):
                issues.append(relative_message)
            elif bool(relative_gate.get("advisory")):
                advisories.append(relative_message + " advisory")

        accepted = not issues
        return {
            "status": (
                "poor" if not accepted else "advisory" if advisories else "ok"
            ),
            "accepted": accepted,
            "applies": True,
            "issues": issues,
            "advisories": advisories,
            "quality_gate": {
                "absolute_visibility": absolute_gate,
                "preview_visibility_retention": relative_gate,
            },
            "metrics": {
                "safe_preview_visibility_score": score,
                "minimum": minimum,
                "profile": profile_name,
                "preview_visibility_retention": (
                    float(relative_ratio) if relative_available else None
                ),
                "preview_visibility_retention_minimum": relative_minimum,
                "relative_contract": (
                    "stellar_subject"
                    if profile_name == "star_colour_preserve"
                    else "diffuse_subject"
                ),
            },
        }


    def _stage7_apply_candidate_visibility_gate(
        self,
        quality_ok: bool,
        issues: List[str],
        pixel_stats: Dict[str, Any],
        target_stretch: Dict[str, Any],
        preview_retention: Dict[str, Any],
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Apply the same absolute/relative visibility contract to every final path."""
        visibility_gate = self._stage7_candidate_visibility_gate(
            pixel_stats,
            target_stretch,
            preview_retention,
        )
        merged_issues = list(issues)
        if not bool(visibility_gate.get("accepted", True)):
            quality_ok = False
            merged_issues.extend(
                str(item)
                for item in (visibility_gate.get("issues") or [])
                if str(item)
            )
        return bool(quality_ok), merged_issues, visibility_gate


    def _stage7_effective_star_growth_ratio_max(self) -> float:
        """Allow bright-nebula starless structure without weakening other targets."""
        generic_limit = float(self.cfg.stage7_star_growth_ratio_max)
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


    def _validate_stage7_stretch_quality(
        self,
        baseline_quality: Optional[QualityMetrics],
        *,
        enforce_star_growth: bool = True,
        current_image_data: Optional[np.ndarray] = None,
    ) -> Tuple[bool, List[str], Optional[QualityMetrics]]:
        metrics = (
            measure_quality_metrics(current_image_data)
            if current_image_data is not None
            else self._measure_current_quality()
        )
        if metrics is None:
            return True, ["quality sampling unavailable"], None

        issues: List[str] = []
        advisories: List[str] = []
        quality_gates: Dict[str, Dict[str, Any]] = {}

        def record_gate(
            name: str,
            gate: Dict[str, Any],
            message: str,
        ) -> None:
            quality_gates[name] = gate
            if gate["hard_failed"]:
                issues.append(message)
            elif gate["advisory"]:
                advisories.append(message + " advisory")

        bg_median_min = stage7_effective_bg_median_min(self.cfg.stage7_bg_median_min)
        record_gate(
            "bg_median",
            stage7_quality.stage7_9_lower_quality_gate(
                self.cfg,
                value=metrics.bg_median,
                accepted_limit=bg_median_min,
            ),
            f"bg_median {metrics.bg_median:.4f}<{self.cfg.stage7_bg_median_min:.4f}",
        )
        record_gate(
            "black_pixel_ratio",
            stage7_quality.stage7_9_upper_quality_gate(
                self.cfg,
                value=metrics.black_pixel_ratio,
                accepted_limit=self.cfg.stage7_black_pixel_ratio_max,
            ),
            "black_pixel_ratio "
            f"{metrics.black_pixel_ratio:.3f}>{self.cfg.stage7_black_pixel_ratio_max:.3f}",
        )
        record_gate(
            "highlight_clip_ratio",
            stage7_quality.stage7_9_upper_quality_gate(
                self.cfg,
                value=metrics.highlight_clip_ratio,
                accepted_limit=self.cfg.stage7_highlight_clip_ratio_max,
            ),
            "highlight_clip_ratio "
            f"{metrics.highlight_clip_ratio:.4f}>{self.cfg.stage7_highlight_clip_ratio_max:.4f}",
        )
        if (
            enforce_star_growth
            and baseline_quality
            and baseline_quality.median_star_size > 0.2
            and metrics.median_star_size > 0
        ):
            star_growth = metrics.median_star_size / max(baseline_quality.median_star_size, 1e-4)
            star_growth_limit = self._stage7_effective_star_growth_ratio_max()
            record_gate(
                "star_size_growth",
                stage7_quality.stage7_9_upper_quality_gate(
                    self.cfg,
                    value=star_growth,
                    accepted_limit=star_growth_limit,
                ),
                f"star_size_growth {star_growth:.3f}>{star_growth_limit:.3f}",
            )
        setattr(metrics, "_stage7_quality_advisories", advisories)
        setattr(metrics, "_stage7_quality_gates", quality_gates)
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


    def _stage7_display90_background_reference(
        self,
        candidate: Dict[str, Any],
        *,
        source_stem: str,
        baseline_image_data: Optional[np.ndarray],
        background_masks: Optional[Dict[str, Any]],
        signal_exclusion_applied: bool,
        curve_quality: Dict[str, Any],
        non_background_hard_gates_accepted: bool,
    ) -> Dict[str, Any]:
        """Measure exact GUI linked-D chroma on the Display90 candidate mask."""

        candidate_name = str(candidate.get("name") or "")
        candidate_method = str(candidate.get("method") or "")
        calibration = dict(
            (candidate.get("params") or {}).get("calibration") or {}
        )
        eligibility = dict(calibration.get("eligibility") or {})
        color_report = getattr(self, "color_calibration_report", {}) or {}
        color_report = color_report if isinstance(color_report, dict) else {}
        physical = color_report.get("physical_color") or {}
        physical = physical if isinstance(physical, dict) else {}
        physical_method = str(
            physical.get("method") or color_report.get("method") or ""
        )
        channel_semantics = str(
            getattr(self, "_channel_semantics", "unknown") or "unknown"
        )
        signal_keys = [
            key
            for key in (
                "core_mask",
                "nebula_mask",
                "faint_nebula_mask",
                "galaxy_signal_mask",
            )
            if isinstance(background_masks, dict)
            and background_masks.get(key) is not None
        ]
        report: Dict[str, Any] = {
            "schema": "starun.stage7-display90-background-reference.v1",
            "status": "not_applicable",
            "applicable": False,
            "matched": False,
            "reason_code": "candidate_not_display_ladder",
            "candidate_name": candidate_name or None,
            "candidate_method": candidate_method or None,
            "source_stem": str(source_stem or ""),
            "channel_semantics": channel_semantics,
            "physical_calibration_accepted": bool(
                physical.get("accepted", False)
            ),
            "physical_calibration_method": physical_method or None,
            "curve_conformance_accepted": bool(
                curve_quality.get("accepted", False)
            ),
            "curve_authenticated": False,
            "non_background_hard_gates_accepted": bool(
                non_background_hard_gates_accepted
            ),
            "signal_exclusion_applied": bool(signal_exclusion_applied),
            "signal_exclusion_keys": signal_keys,
            "mask_scope": "stage6_frozen_signal_excluded_background_mask",
            "reference_application": "gui_rec709_luminance_gain",
            "candidate_application": (
                "rec709_luminance_uniform_rgb_gain"
                if candidate_method
                == stage7_stretch_metrics.DISPLAY_LUMINANCE_VECTOR_METHOD
                else "linked_rgb_common_lut"
            ),
            "reference_metrics": {},
        }
        if not (
            str(candidate_name or "").startswith("cand_display")
            and candidate_method in stage7_stretch_metrics.DISPLAY_LUT_METHODS
        ):
            return report

        report["applicable"] = True
        automatic_route = bool(
            str(getattr(self.cfg, "stage7_processing_mode", "auto") or "auto")
            .strip()
            .lower()
            == "auto"
            and str(source_stem or "") == "stage6_starless"
            and eligibility.get("automatic_parameter_mode") is True
            and eligibility.get("starless_recomposition_planned") is True
            and str(eligibility.get("source_stem") or "")
            == "stage6_starless"
            and str(eligibility.get("star_separation_state") or "")
            == StarSeparationState.ACCEPTED.value
        )
        report["automatic_accepted_starless_route"] = automatic_route
        if not automatic_route:
            report["reason_code"] = "display90_route_not_eligible"
            return report
        if channel_semantics != "narrowband_composite":
            report["reason_code"] = "channel_semantics_not_narrowband_composite"
            return report
        if not (
            bool(physical.get("accepted", False))
            and physical_method == "SPCC_NARROWBAND"
        ):
            report["reason_code"] = "spcc_narrowband_not_accepted"
            return report
        if not bool(curve_quality.get("accepted", False)):
            report.update(
                status="unavailable",
                reason_code="display90_curve_conformance_failed",
            )
            return report
        if not non_background_hard_gates_accepted:
            report.update(
                status="not_applicable",
                reason_code="non_background_hard_gate_failed",
            )
            return report
        if not signal_exclusion_applied or not signal_keys:
            report.update(
                status="unavailable",
                reason_code="signal_excluded_background_mask_unavailable",
            )
            return report
        if baseline_image_data is None:
            report.update(
                status="unavailable",
                reason_code="stage6_starless_pixels_unavailable",
            )
            return report

        try:
            reference_pixels, lut_contract = (
                stage7_stretch_metrics.build_display90_gui_linked_reference(
                    baseline_image_data,
                    calibration,
                )
            )
            reference_metrics = self._background_quality_metrics(
                reference_pixels,
                background_masks,
            )
            if not reference_metrics:
                raise ValueError("GUI linked reference background metrics unavailable")
            reference_metrics["signal_exclusion_applied"] = True
            reference_load = self._stage7_background_chroma_load(
                reference_metrics
            )
            if not math.isfinite(reference_load) or reference_load <= 0.0:
                raise ValueError("GUI linked reference chroma load is invalid")
            for metric_name in (
                "chroma_noise_score",
                "background_mottling_score",
            ):
                metric_value = float(reference_metrics.get(metric_name))
                if not math.isfinite(metric_value):
                    raise ValueError(
                        f"GUI linked reference {metric_name} is invalid"
                    )
            metric_keys = (
                "background_chroma_load",
                "chroma_noise_score",
                "background_mottling_score",
                "bg_median",
                "bg_std",
                "background_red_mean",
                "background_green_mean",
                "background_blue_mean",
                "background_green_excess",
                "signal_exclusion_applied",
            )
            report.update(
                status="available",
                reason_code="gui_linked_reference_measured",
                curve_authenticated=True,
                lut_sha256=lut_contract.get("sha256"),
                reference_metrics={
                    key: reference_metrics.get(key) for key in metric_keys
                },
                reference_chroma_load=reference_load,
            )
            return report
        except (
            IndexError,
            KeyError,
            MemoryError,
            RuntimeError,
            TypeError,
            ValueError,
            FloatingPointError,
        ) as error:
            report.update(
                status="unavailable",
                reason_code="gui_linked_reference_measurement_unavailable",
                reason=self._short_text(error, 200),
            )
            return report


    def _stage7_stretch_background_gate(
        self,
        baseline: Dict[str, Any],
        candidate: Dict[str, Any],
        *,
        candidate_name: str = "",
        candidate_method: str = "",
        display90_reference: Optional[Dict[str, Any]] = None,
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
                getattr(self.cfg, "stage7_stretch_chroma_load_growth_max", 1.37)
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
            "display90_reference_chroma_load_ratio_max": _clamp_float(
                getattr(
                    self.cfg,
                    "stage7_display90_reference_chroma_load_ratio_max",
                    1.05,
                ),
                1.00,
                1.20,
            ),
            "display90_reference_chroma_load_absolute_max": _clamp_float(
                getattr(
                    self.cfg,
                    "stage7_display90_reference_chroma_load_absolute_max",
                    0.40,
                ),
                0.15,
                0.50,
            ),
        }
        issues: List[str] = []
        advisories: List[str] = []
        quality_gates: Dict[str, Dict[str, Any]] = {}
        if not candidate:
            return {
                "accepted": False,
                "status": "unavailable",
                "issues": ["background_quality_metrics_unavailable"],
                "limits": limits,
                "metrics": {},
                "display90_reference_match": (
                    copy.deepcopy(display90_reference)
                    if isinstance(display90_reference, dict)
                    else {
                        "schema": (
                            "starun.stage7-display90-background-reference.v1"
                        ),
                        "status": "not_applicable",
                        "applicable": False,
                        "matched": False,
                        "reason_code": "candidate_metrics_unavailable",
                    }
                ),
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
        low_absolute_growth_exempted = bool(
            low_absolute_load_exempted or signal_excluded_load_exempted
        )
        effective_load_limit = (
            limits["chroma_load_signal_excluded_max"]
            if signal_excluded_load_exempted
            else low_absolute_effective_max
        )

        def record_gate(
            name: str,
            value: float,
            limit: float,
            message: str,
        ) -> None:
            gate = stage7_quality.stage7_9_upper_quality_gate(
                self.cfg,
                value=value,
                accepted_limit=limit,
            )
            quality_gates[name] = gate
            if gate["hard_failed"]:
                issues.append(message)
            elif gate["advisory"]:
                advisories.append(message + " advisory")

        record_gate(
            "background_chroma_noise_score",
            chroma,
            limits["chroma_noise_score_max"],
            "background_chroma_noise_score "
            f"{chroma:.3f}>{limits['chroma_noise_score_max']:.3f}",
        )
        record_gate(
            "background_mottling_score",
            mottling,
            limits["background_mottling_score_max"],
            "background_mottling_score "
            f"{mottling:.3f}>{limits['background_mottling_score_max']:.3f}",
        )

        reference_match = (
            copy.deepcopy(display90_reference)
            if isinstance(display90_reference, dict)
            else {
                "schema": "starun.stage7-display90-background-reference.v1",
                "status": "not_applicable",
                "applicable": False,
                "matched": False,
                "reason_code": "reference_not_requested",
            }
        )
        reference_match.setdefault(
            "schema",
            "starun.stage7-display90-background-reference.v1",
        )
        reference_match["measurement_status"] = reference_match.get("status")
        reference_match["matched"] = False
        reference_match["candidate_chroma_load"] = candidate_load
        reference_match["baseline_chroma_load"] = (
            baseline_load if baseline else None
        )
        reference_match["nominal_chroma_load_growth"] = load_growth
        reference_match["limits"] = {
            "candidate_to_gui_reference_ratio_max": limits[
                "display90_reference_chroma_load_ratio_max"
            ],
            "candidate_absolute_chroma_load_max": limits[
                "display90_reference_chroma_load_absolute_max"
            ],
            "nominal_linear_baseline_growth_max": limits[
                "chroma_load_growth_max"
            ],
            "uses_stage7_9_advisory_multiplier": False,
        }
        reference_metrics = reference_match.get("reference_metrics") or {}
        try:
            reference_load = float(
                reference_match.get(
                    "reference_chroma_load",
                    self._stage7_background_chroma_load(reference_metrics),
                )
            )
        except (TypeError, ValueError):
            reference_load = math.nan
        reference_ratio = (
            candidate_load / reference_load
            if math.isfinite(reference_load) and reference_load > 0.0
            else math.nan
        )
        reference_match["reference_chroma_load"] = (
            reference_load if math.isfinite(reference_load) else None
        )
        reference_match["candidate_to_gui_reference_ratio"] = (
            reference_ratio if math.isfinite(reference_ratio) else None
        )
        ratio_limit = limits["display90_reference_chroma_load_ratio_max"]
        absolute_limit = limits[
            "display90_reference_chroma_load_absolute_max"
        ]
        ratio_within_limit = bool(
            math.isfinite(reference_ratio)
            and reference_ratio <= ratio_limit + 1e-12
        )
        absolute_within_limit = bool(
            math.isfinite(candidate_load)
            and candidate_load <= absolute_limit + 1e-12
        )
        reference_applicable = bool(reference_match.get("applicable") is True)
        ratio_gate_status = (
            "ok"
            if reference_applicable and ratio_within_limit
            else "rejected"
            if reference_applicable and math.isfinite(reference_ratio)
            else "unavailable"
            if reference_applicable
            else "not_applicable"
        )
        absolute_gate_status = (
            "ok"
            if reference_applicable and absolute_within_limit
            else "rejected"
            if reference_applicable
            else "not_applicable"
        )
        reference_match["quality_gates"] = {
            "candidate_to_gui_reference_ratio": {
                "status": ratio_gate_status,
                "value": (
                    reference_ratio if math.isfinite(reference_ratio) else None
                ),
                "accepted_limit": ratio_limit,
                "advisory": False,
                "hard_failed": bool(
                    reference_applicable and not ratio_within_limit
                ),
            },
            "candidate_absolute_chroma_load": {
                "status": absolute_gate_status,
                "value": candidate_load,
                "accepted_limit": absolute_limit,
                "advisory": False,
                "hard_failed": bool(
                    reference_applicable and not absolute_within_limit
                ),
            },
        }
        authenticated_digest = str(reference_match.get("lut_sha256") or "")
        reference_context_eligible = bool(
            reference_match.get("schema")
            == "starun.stage7-display90-background-reference.v1"
            and reference_match.get("measurement_status") == "available"
            and reference_match.get("applicable") is True
            and str(candidate_name or "").startswith("cand_display")
            and str(candidate_method or "")
            in stage7_stretch_metrics.DISPLAY_LUT_METHODS
            and reference_match.get("automatic_accepted_starless_route") is True
            and reference_match.get("channel_semantics")
            == "narrowband_composite"
            and reference_match.get("physical_calibration_accepted") is True
            and reference_match.get("physical_calibration_method")
            == "SPCC_NARROWBAND"
            and reference_match.get("curve_conformance_accepted") is True
            and reference_match.get("curve_authenticated") is True
            and reference_match.get("non_background_hard_gates_accepted")
            is True
            and bool(re.fullmatch(r"[0-9a-f]{64}", authenticated_digest))
            and signal_exclusion_applied
            and reference_metrics.get("signal_exclusion_applied") is True
        )
        reference_match["context_eligible"] = reference_context_eligible
        nominal_growth_gate: Optional[Dict[str, Any]] = None
        if (
            load_growth is not None
            and load_growth > limits["chroma_load_growth_max"]
        ):
            nominal_growth_gate = stage7_quality.stage7_9_upper_quality_gate(
                self.cfg,
                value=load_growth,
                accepted_limit=limits["chroma_load_growth_max"],
            )
        noise_or_mottling_hard_failed = bool(
            quality_gates["background_chroma_noise_score"].get(
                "hard_failed", False
            )
            or quality_gates["background_mottling_score"].get(
                "hard_failed", False
            )
        )
        display_reference_exempted = bool(
            nominal_growth_gate
            and nominal_growth_gate.get("hard_failed") is True
            and not low_absolute_growth_exempted
            and reference_context_eligible
            and ratio_within_limit
            and absolute_within_limit
            and not noise_or_mottling_hard_failed
        )
        load_growth_exempted = bool(
            low_absolute_growth_exempted or display_reference_exempted
        )

        if display_reference_exempted and nominal_growth_gate is not None:
            reference_match.update(
                status="matched_advisory",
                matched=True,
                reason_code="display90_narrowband_gui_reference_matched",
                nominal_growth_gate=copy.deepcopy(nominal_growth_gate),
                nominal_growth_hard_failure_exempted=True,
            )
            quality_gates["background_chroma_load_growth"] = {
                **dict(nominal_growth_gate),
                "status": "display_reference_matched_advisory",
                "advisory": True,
                "hard_failed": False,
                "reason_code": "display90_narrowband_gui_reference_matched",
                "nominal_gate": copy.deepcopy(nominal_growth_gate),
            }
            advisories.append(
                "background_chroma_load_growth "
                f"{load_growth:.3f}>{limits['chroma_load_growth_max']:.3f} "
                "display_reference_matched advisory"
            )
        elif nominal_growth_gate is not None and not low_absolute_growth_exempted:
            if reference_context_eligible:
                if not absolute_within_limit:
                    reference_match.update(
                        status="rejected",
                        reason_code=(
                            "display90_reference_absolute_chroma_load_exceeded"
                        ),
                    )
                elif not ratio_within_limit:
                    reference_match.update(
                        status="rejected",
                        reason_code="display90_reference_chroma_ratio_exceeded",
                    )
                elif noise_or_mottling_hard_failed:
                    reference_match.update(
                        status="rejected",
                        reason_code=(
                            "background_noise_or_mottling_hard_failed"
                        ),
                    )
            quality_gates["background_chroma_load_growth"] = nominal_growth_gate
            message = (
                "background_chroma_load_growth "
                f"{load_growth:.3f}>{limits['chroma_load_growth_max']:.3f}"
            )
            if nominal_growth_gate["hard_failed"]:
                issues.append(message)
            elif nominal_growth_gate["advisory"]:
                advisories.append(message + " advisory")
        elif low_absolute_growth_exempted:
            if reference_match.get("applicable") is True:
                reference_match.update(
                    status="not_needed",
                    reason_code="existing_low_absolute_exemption_applied",
                )
        elif reference_match.get("measurement_status") == "available":
            reference_match.update(
                status="not_needed",
                reason_code="nominal_growth_gate_not_exceeded",
            )

        return {
            "accepted": not issues,
            "status": (
                "poor" if issues else "advisory" if advisories else "ok"
            ),
            "issues": issues,
            "advisories": advisories,
            "quality_gates": quality_gates,
            "limits": limits,
            "display90_reference_match": reference_match,
            "metrics": {
                "chroma_noise_score": chroma,
                "background_mottling_score": mottling,
                "chroma_load": candidate_load,
                "baseline_chroma_load": baseline_load if baseline else None,
                "chroma_load_growth": load_growth,
                "chroma_load_growth_exempted": load_growth_exempted,
                "chroma_load_growth_low_absolute_exempted": (
                    low_absolute_growth_exempted
                ),
                "chroma_load_growth_extreme_low_exempted": low_absolute_load_exempted,
                "chroma_load_growth_signal_excluded_exempted": signal_excluded_load_exempted,
                "chroma_load_growth_display_reference_exempted": (
                    display_reference_exempted
                ),
                "display_reference_chroma_load": (
                    reference_load if math.isfinite(reference_load) else None
                ),
                "display_reference_chroma_load_ratio": (
                    reference_ratio if math.isfinite(reference_ratio) else None
                ),
                "display_reference_chroma_load_ratio_max": ratio_limit,
                "display_reference_chroma_load_absolute_max": absolute_limit,
                "signal_exclusion_applied": signal_exclusion_applied,
                "chroma_load_low_absolute_effective_max": (
                    effective_load_limit
                ),
                "chroma_load_low_absolute_tolerance": low_absolute_tolerance,
                "extreme_low_background": extreme_low_background,
                "bg_median": candidate.get("bg_median"),
                "bg_std": candidate.get("bg_std"),
                "background_red_mean": candidate.get("background_red_mean"),
                "background_green_mean": candidate.get("background_green_mean"),
                "background_blue_mean": candidate.get("background_blue_mean"),
                "background_green_excess": candidate.get(
                    "background_green_excess"
                ),
            },
        }


    def _stage7_uncalibrated_background_color_review_gate(
        self,
        selected_attempt: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Require review for absolute background cast without physical color."""
        report = getattr(self, "color_calibration_report", {}) or {}
        report = report if isinstance(report, dict) else {}
        physical = report.get("physical_color") or {}
        physical = physical if isinstance(physical, dict) else {}
        method = str(physical.get("method") or report.get("method") or "")
        # Current Stage 4 reports an explicit acceptance bit.  Do not let a
        # failed SPCC/PCC attempt bypass this review gate merely because its
        # attempted method name is still present in a diagnostic report.
        physical_accepted = bool(physical.get("accepted", False))
        channel_semantics = str(
            getattr(self, "_channel_semantics", "unknown") or "unknown"
        )
        try:
            limit = float(
                getattr(
                    self.cfg,
                    "stage7_uncalibrated_background_chroma_load_review_max",
                    0.12,
                )
            )
        except (TypeError, ValueError):
            limit = 0.12
        if not math.isfinite(limit):
            limit = 0.12
        limit = max(0.04, min(limit, 0.50))
        gate: Dict[str, Any] = {
            "schema": "starun.stage7-background-color-review.v1",
            "applicable": False,
            "status": "not_applicable",
            "requires_review": False,
            "reason_code": "physical_calibration_accepted",
            "physical_calibration_accepted": physical_accepted,
            "physical_calibration_method": method or None,
            "channel_semantics": channel_semantics,
            "metric": "background_chroma_load",
            "value": None,
            "limit": limit,
            "measurement": "selected_stage7_signal_excluded_background",
            "signal_exclusion_applied": False,
            "global_white_balance_applied": False,
            "global_white_balance_prohibited": True,
            "action": "review_only_no_global_white_balance",
        }
        if physical_accepted or channel_semantics == "mono":
            if channel_semantics == "mono":
                gate["reason_code"] = "mono_not_applicable"
            return gate

        gate["applicable"] = True
        background_gate = (
            selected_attempt.get("background_quality_gate") or {}
            if isinstance(selected_attempt, dict)
            else {}
        )
        metrics = (
            background_gate.get("metrics") or {}
            if isinstance(background_gate, dict)
            else {}
        )
        signal_exclusion_applied = bool(
            metrics.get("signal_exclusion_applied", False)
        )
        gate["signal_exclusion_applied"] = signal_exclusion_applied
        for key in (
            "background_red_mean",
            "background_green_mean",
            "background_blue_mean",
            "background_green_excess",
        ):
            gate[key] = metrics.get(key)
        try:
            chroma_load = float(metrics.get("chroma_load"))
        except (TypeError, ValueError):
            chroma_load = math.nan
        if not signal_exclusion_applied or not math.isfinite(chroma_load):
            gate.update(
                status="unavailable_review_required",
                requires_review=True,
                reason_code="signal_excluded_background_measurement_unavailable",
            )
            return gate

        chroma_load = max(chroma_load, 0.0)
        gate["value"] = chroma_load
        tolerance_gate = stage7_quality.stage7_9_upper_quality_gate(
            self.cfg,
            value=chroma_load,
            accepted_limit=limit,
        )
        gate["quality_gate"] = tolerance_gate
        gate["hard_limit"] = tolerance_gate["hard_limit"]
        if tolerance_gate["hard_failed"]:
            gate.update(
                status="review_required",
                requires_review=True,
                reason_code="uncalibrated_background_chroma_load_exceeded",
            )
        elif tolerance_gate["advisory"]:
            gate.update(
                status="advisory",
                requires_review=False,
                reason_code="uncalibrated_background_chroma_load_advisory",
                advisories=[
                    "uncalibrated_background_chroma_load "
                    f"{chroma_load:.3f}>{limit:.3f} advisory"
                ],
            )
        else:
            gate.update(status="ok", reason_code="within_review_limit")
        return gate


    def _stage7_transform_loss_gate(
        self,
        report: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Promote exact clipping diagnostics to immutable technical gates."""

        warn_limit = _clamp_float(
            getattr(self.cfg, "stage7_transform_new_hard_clip_ratio_warn", 0.0001),
            0.0,
            0.005,
        )
        hard_limit = _clamp_float(
            getattr(self.cfg, "stage7_transform_new_hard_clip_ratio_max", 0.0005),
            max(warn_limit, 1e-7),
            0.01,
        )
        zero_limit = _clamp_float(
            getattr(self.cfg, "stage7_transform_unexpected_zero_ratio_max", 0.001),
            0.0001,
            0.02,
        )
        result: Dict[str, Any] = {
            "status": "unavailable",
            "accepted": True,
            "technical_gate": True,
            "issues": [],
            "advisories": [],
            "limits": {
                "newly_hard_clipped_ratio_warn": warn_limit,
                "newly_hard_clipped_ratio_max": hard_limit,
                "unexpected_newly_zeroed_ratio_max": zero_limit,
            },
        }
        if report.get("status") != "available":
            result.update(
                status="rejected",
                accepted=False,
                issues=["transform_loss_unavailable"],
            )
            return result
        global_metrics = dict(report.get("global") or {})
        try:
            newly_clipped = float(
                global_metrics.get("newly_hard_clipped_ratio", 0.0) or 0.0
            )
            newly_zeroed = float(
                global_metrics.get("newly_zeroed_ratio", 0.0) or 0.0
            )
            unexpected_zeroed = float(
                global_metrics.get(
                    "unexpected_newly_zeroed_ratio",
                    newly_zeroed,
                )
                or 0.0
            )
        except (TypeError, ValueError):
            result.update(
                status="rejected",
                accepted=False,
                issues=["transform_loss_metrics_invalid"],
            )
            return result
        if not all(
            math.isfinite(value)
            for value in (newly_clipped, newly_zeroed, unexpected_zeroed)
        ):
            result.update(
                status="rejected",
                accepted=False,
                issues=["transform_loss_metrics_nonfinite"],
            )
            return result
        issues: List[str] = []
        advisories: List[str] = []
        if newly_clipped > hard_limit:
            issues.append(
                "transform_new_hard_clip_ratio "
                f"{newly_clipped:.6f}>{hard_limit:.6f}"
            )
        elif newly_clipped > warn_limit:
            advisories.append(
                "transform_new_hard_clip_ratio "
                f"{newly_clipped:.6f}>{warn_limit:.6f} advisory"
            )
        if unexpected_zeroed > zero_limit:
            issues.append(
                "transform_unexpected_zero_ratio "
                f"{unexpected_zeroed:.6f}>{zero_limit:.6f}"
            )
        result.update(
            status="rejected" if issues else "advisory" if advisories else "ok",
            accepted=not issues,
            issues=issues,
            advisories=advisories,
            metrics={
                "newly_hard_clipped_ratio": newly_clipped,
                "newly_zeroed_ratio": newly_zeroed,
                "unexpected_newly_zeroed_ratio": unexpected_zeroed,
            },
        )
        return result


    def _stage7_color_vector_gate(
        self,
        report: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Gate colour-direction drift without treating intended narrowband as RGB."""

        channel_semantics = str(
            getattr(self, "_channel_semantics", "unknown") or "unknown"
        ).strip().lower()
        narrowband = channel_semantics == "narrowband_composite"
        advisory_limit = _clamp_float(
            getattr(
                self.cfg,
                "stage7_narrowband_color_vector_p95_advisory_max"
                if narrowband
                else "stage7_color_vector_p95_advisory_max",
                0.10 if narrowband else 0.04,
            ),
            0.01,
            0.30,
        )
        hard_limit = _clamp_float(
            getattr(
                self.cfg,
                "stage7_narrowband_color_vector_p95_hard_max"
                if narrowband
                else "stage7_color_vector_p95_hard_max",
                0.20 if narrowband else 0.08,
            ),
            advisory_limit,
            0.40,
        )
        result: Dict[str, Any] = {
            "status": "unavailable",
            "accepted": True,
            "technical_gate": False,
            "issues": [],
            "advisories": [],
            "channel_semantics": channel_semantics,
            "limits": {
                "chromaticity_l1_half_p95_advisory_max": advisory_limit,
                "chromaticity_l1_half_p95_hard_max": hard_limit,
            },
        }
        if report.get("status") != "available":
            result["advisories"] = ["color_vector_reference_unavailable advisory"]
            return result
        try:
            value = float(
                (report.get("metrics") or {})["chromaticity_l1_half_p95"]
            )
        except (KeyError, TypeError, ValueError):
            result["advisories"] = ["color_vector_reference_invalid advisory"]
            return result
        if not math.isfinite(value):
            result["advisories"] = ["color_vector_reference_nonfinite advisory"]
            return result
        issues: List[str] = []
        advisories: List[str] = []
        if value > hard_limit:
            issues.append(
                "color_vector_chromaticity_p95 "
                f"{value:.3f}>{hard_limit:.3f}"
            )
        elif value > advisory_limit:
            advisories.append(
                "color_vector_chromaticity_p95 "
                f"{value:.3f}>{advisory_limit:.3f} advisory"
            )
        result.update(
            status="rejected" if issues else "advisory" if advisories else "ok",
            accepted=not issues,
            issues=issues,
            advisories=advisories,
            metrics={"chromaticity_l1_half_p95": value},
        )
        return result


    def _stage7_chroma_rescue_strengths(
        self,
        attempt: Optional[Dict[str, Any]] = None,
    ) -> List[float]:
        if isinstance(attempt, dict):
            background_gate = attempt.get("background_quality_gate") or {}
            metrics = background_gate.get("metrics") or {}
            limits = background_gate.get("limits") or {}
            try:
                current_load = float(metrics.get("chroma_load", 0.0) or 0.0)
                target_load = float(
                    metrics.get(
                        "chroma_load_low_absolute_effective_max",
                        limits.get(
                            "chroma_load_signal_excluded_max",
                            limits.get("chroma_load_low_absolute_max", 0.05),
                        ),
                    )
                    or 0.0
                )
            except (TypeError, ValueError):
                current_load = 0.0
                target_load = 0.0
            if (
                math.isfinite(current_load)
                and math.isfinite(target_load)
                and current_load > target_load > 0.0
            ):
                max_strength = _clamp_float(
                    getattr(self.cfg, "stage7_chroma_rescue_max_strength", 0.90),
                    0.10,
                    0.90,
                )
                needed = _clamp_float(
                    1.0 - 0.95 * target_load / current_load,
                    0.10,
                    max_strength,
                )
                levels = [
                    max(0.10, needed * 0.75),
                    needed,
                    min(max_strength, needed * 1.15),
                ]
                unique = []
                for level in levels:
                    rounded = round(float(level), 4)
                    if not any(abs(rounded - item) < 1e-6 for item in unique):
                        unique.append(rounded)
                try:
                    max_attempts = int(
                        getattr(self.cfg, "stage7_chroma_rescue_max_attempts", 3)
                    )
                except (TypeError, ValueError):
                    max_attempts = 3
                return unique[: max(0, min(max_attempts, 3))]

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
            value = _clamp_float(value, 0.10, 0.90)
            if not any(abs(value - existing) < 1e-6 for existing in levels):
                levels.append(value)
        ordered = sorted(levels) or [0.10, 0.20, 0.35]
        try:
            max_attempts = int(
                getattr(self.cfg, "stage7_chroma_rescue_max_attempts", 3)
            )
        except (TypeError, ValueError):
            max_attempts = 3
        return ordered[: max(0, min(max_attempts, 3))]


    @staticmethod
    def _stage7_retention_ratio(
        attempt: Dict[str, Any],
        metric_name: str,
    ) -> Optional[float]:
        rendition = (
            (attempt.get("rendition_metrics") or {}).get("retention") or {}
        )
        item = (rendition.get("metrics") or {}).get(metric_name) or {}
        try:
            value = float(item.get("ratio"))
        except (TypeError, ValueError):
            value = math.nan
        if bool(item.get("available")) and math.isfinite(value):
            return max(0.0, value)

        preview = (attempt.get("preview_retention") or {}).get("metrics") or {}
        fallback_name = metric_name
        if metric_name == "saturation_median" and fallback_name not in preview:
            fallback_name = "saturation_p95"
        fallback = preview.get(fallback_name) or {}
        try:
            value = float(
                fallback.get("ranking_ratio", fallback.get("ratio"))
            )
        except (TypeError, ValueError):
            return None
        return max(0.0, value) if math.isfinite(value) else None


    @classmethod
    def _stage7_presentation_score(
        cls,
        attempt: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Bound rewards at useful goals, then use safety headroom as 30%."""

        profile_name = str(
            (
                ((attempt.get("adaptation") or {}).get("target_aware") or {}).get(
                    "name"
                )
                or "generic_balanced"
            )
        ).strip().lower()
        if profile_name in {
            "bright_core_protect",
            "widefield_nebulosity",
            "widefield_faint_signal",
            "widefield_subject_separation",
            "dark_nebula_separation",
        }:
            profile = "nebula"
        elif profile_name == "galaxy_core_halo_balance":
            profile = "galaxy"
        elif profile_name == "star_colour_preserve":
            profile = "cluster"
        else:
            profile = "generic"
        goals = {
            "nebula": {
                "visibility": 0.90,
                "subject_span": 0.80,
                "saturation_median": 0.90,
                "microcontrast": 0.80,
            },
            "galaxy": {
                "visibility": 0.85,
                "subject_span": 0.80,
                "saturation_median": 0.75,
                "microcontrast": 0.85,
            },
            "cluster": {
                "visibility": 0.80,
                "subject_span": 0.70,
                "saturation_median": 0.80,
                "microcontrast": 0.75,
            },
            "generic": {
                "visibility": 0.85,
                "subject_span": 0.75,
                "saturation_median": 0.80,
                "microcontrast": 0.75,
            },
        }[profile]
        subject_brightness = attempt.get("subject_brightness_selection")
        lift_reliable = bool(
            not isinstance(subject_brightness, dict)
            or (subject_brightness.get("preview_reliability") or {}).get(
                "subject_lift_reliable",
                True,
            )
        )
        low_lift_nebula = bool(profile == "nebula" and not lift_reliable)
        weights = (
            {
                "visibility": 0.25,
                "subject_span": 0.30,
                "saturation_median": 0.15,
                "microcontrast": 0.30,
            }
            if low_lift_nebula
            else {
                "nebula": {
                    "visibility": 0.28,
                    "subject_span": 0.24,
                    "saturation_median": 0.28,
                    "microcontrast": 0.20,
                },
                "galaxy": {
                    "visibility": 0.30,
                    "subject_span": 0.27,
                    "saturation_median": 0.18,
                    "microcontrast": 0.25,
                },
                "cluster": {
                    "visibility": 0.27,
                    "subject_span": 0.23,
                    "saturation_median": 0.25,
                    "microcontrast": 0.25,
                },
                "generic": {
                    "visibility": 0.28,
                    "subject_span": 0.25,
                    "saturation_median": 0.25,
                    "microcontrast": 0.22,
                },
            }[profile]
        )
        utilities: Dict[str, float] = {}
        missing: List[str] = []
        for name, goal in goals.items():
            ratio = cls._stage7_retention_ratio(attempt, name)
            if ratio is None:
                missing.append(name)
                utilities[name] = 0.0
            else:
                utilities[name] = _clamp_float(ratio / goal, 0.0, 1.0)
        presentation = sum(
            utilities[name] * weights[name] for name in weights
        )

        safety_values: Dict[str, float] = {}
        background_gate = attempt.get("background_quality_gate") or {}
        background_metrics = background_gate.get("metrics") or {}
        background_limits = background_gate.get("limits") or {}

        def headroom(value: Any, limit: Any) -> float:
            try:
                numeric_value = max(0.0, float(value))
                numeric_limit = float(limit)
            except (TypeError, ValueError):
                return 0.50
            if not math.isfinite(numeric_value) or not math.isfinite(numeric_limit) or numeric_limit <= 0.0:
                return 0.50
            return 1.0 - _clamp_float(numeric_value / numeric_limit, 0.0, 1.0)

        load_limit = background_metrics.get(
            "display_reference_chroma_load_absolute_max"
            if background_metrics.get("chroma_load_growth_display_reference_exempted")
            else "chroma_load_low_absolute_effective_max"
        )
        if load_limit is None:
            load_limit = background_limits.get(
                "chroma_load_signal_excluded_max",
                background_limits.get("chroma_load_low_absolute_max"),
            )
        safety_values["background_chroma_load"] = headroom(
            background_metrics.get("chroma_load"),
            load_limit,
        )
        safety_values["chroma_noise"] = headroom(
            background_metrics.get("chroma_noise_score"),
            background_limits.get("chroma_noise_score_max"),
        )
        safety_values["mottling"] = headroom(
            background_metrics.get("background_mottling_score"),
            background_limits.get("background_mottling_score_max"),
        )
        color_gate = attempt.get("color_vector_gate") or {}
        safety_values["color_vector"] = headroom(
            (color_gate.get("metrics") or {}).get("chromaticity_l1_half_p95"),
            (color_gate.get("limits") or {}).get(
                "chromaticity_l1_half_p95_hard_max"
            ),
        )
        transform_gate = attempt.get("transform_loss_gate") or {}
        safety_values["new_hard_clip"] = headroom(
            (transform_gate.get("metrics") or {}).get(
                "newly_hard_clipped_ratio"
            ),
            (transform_gate.get("limits") or {}).get(
                "newly_hard_clipped_ratio_max"
            ),
        )
        local_quality = attempt.get("target_local_quality") or {}
        local_gates = local_quality.get("quality_gates") or {}
        core_headrooms = []
        for gate_name in (
            "local_core_clip_ratio",
            "local_core_colored_plateau_component_ratio",
            "local_core_parity_phase_span",
        ):
            gate = local_gates.get(gate_name) or {}
            if "value" in gate and "hard_limit" in gate:
                core_headrooms.append(
                    headroom(gate.get("value"), gate.get("hard_limit"))
                )
        safety_values["target_core_safety"] = (
            min(core_headrooms) if core_headrooms else 0.50
        )
        if profile == "nebula":
            safety_weights = {
                "background_chroma_load": 0.25,
                "chroma_noise": 0.10,
                "mottling": 0.10,
                "color_vector": 0.15,
                "new_hard_clip": 0.15,
                "target_core_safety": 0.25,
            }
        else:
            safety_weights = {
                "background_chroma_load": 0.30,
                "chroma_noise": 0.15,
                "mottling": 0.15,
                "color_vector": 0.25,
                "new_hard_clip": 0.15,
            }
        safety = sum(
            safety_values[name] * safety_weights[name]
            for name in safety_weights
        )
        score = 0.70 * presentation + 0.30 * safety
        return {
            "policy": STAGE7_CANDIDATE_RANKING_POLICY,
            "profile": profile,
            "selection_mode": (
                "low_lift_structure_first"
                if low_lift_nebula
                else "bounded_subject_brightness"
            ),
            "score": float(score),
            "presentation_utility": float(presentation),
            "safety_headroom": float(safety),
            "goals": goals,
            "utilities": utilities,
            "safety": safety_values,
            "missing_metrics": missing,
            "saturation_p95_role": "gate_and_diagnostic_only",
            "subject_brightness": subject_brightness,
        }


    @classmethod
    def _stage7_candidate_selection_key(
        cls,
        attempt: Dict[str, Any],
    ) -> Tuple[Any, ...]:
        """Filter hard gates first, then rank bounded presentation plus safety."""

        status_penalty = 0 if attempt.get("status") == "ok" and attempt.get("stem") else 1
        final_penalty = 0 if bool(attempt.get("allowed_as_final", False)) else 1
        technical_penalty = 0 if bool(attempt.get("technical_safe", True)) else 1
        local_quality = attempt.get("target_local_quality") or {}
        strict_target = bool(
            (local_quality.get("strict_target_evidence") or {}).get(
                "strict", False
            )
        )
        stretch_saturated_penalty = int(
            strict_target
            and bool(
                (attempt.get("preview_target_attainment") or {}).get(
                    "stretch_saturated", False
                )
            )
        )
        subject_brightness = attempt.get("subject_brightness_selection")
        if isinstance(subject_brightness, dict):
            brightness_floor_penalty = int(
                not bool(
                    subject_brightness.get("formal_floor_passed", False)
                )
            )
            brightness_ranking = subject_brightness.get("ranking") or {}
            try:
                brightness_goal_count = max(
                    0,
                    int(brightness_ranking.get("goal_count", 0) or 0),
                )
            except (TypeError, ValueError):
                brightness_goal_count = 0
            try:
                brightness_utility = float(
                    brightness_ranking.get("utility", 0.0) or 0.0
                )
            except (TypeError, ValueError):
                brightness_utility = 0.0
            if not math.isfinite(brightness_utility):
                brightness_utility = 0.0
            brightness_utility = _clamp_float(
                brightness_utility,
                0.0,
                1.0,
            )
        else:
            brightness_floor_penalty = 0
            brightness_goal_count = 0
            brightness_utility = 0.0
        report = attempt.get("presentation_score") or cls._stage7_presentation_score(
            attempt
        )
        try:
            score = float(report.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        if not math.isfinite(score):
            score = 0.0
        advisory_count = len(
            {
                str(item).strip()
                for item in (attempt.get("advisories") or [])
                if str(item).strip()
            }
        )
        try:
            risk_score = float(attempt.get("risk_score", 1_000_000.0))
        except (TypeError, ValueError):
            risk_score = 1_000_000.0
        if not math.isfinite(risk_score):
            risk_score = 1_000_000.0
        lift_reliable = bool(
            not isinstance(subject_brightness, dict)
            or (subject_brightness.get("preview_reliability") or {}).get(
                "subject_lift_reliable",
                True,
            )
        )
        if lift_reliable:
            ranking_tail = (
                -brightness_goal_count,
                -brightness_utility,
                -score,
                advisory_count,
                risk_score,
            )
        else:
            ranking_tail = (
                -score,
                advisory_count,
                risk_score,
                -brightness_goal_count,
                -brightness_utility,
            )
        return (
            status_penalty,
            final_penalty,
            technical_penalty,
            stretch_saturated_penalty,
            brightness_floor_penalty,
            *ranking_tail,
            str(attempt.get("name") or ""),
        )


    @classmethod
    def _stage7_forced_candidate_selection_key(
        cls,
        attempt: Dict[str, Any],
    ) -> Tuple[Any, ...]:
        """For forced output, minimise appearance-gate excess before vividness."""

        local_quality = attempt.get("target_local_quality") or {}
        strict_target = bool(
            (local_quality.get("strict_target_evidence") or {}).get(
                "strict", False
            )
        )
        stretch_saturated_penalty = int(
            strict_target
            and bool(
                (attempt.get("preview_target_attainment") or {}).get(
                    "stretch_saturated", False
                )
            )
        )

        excesses: List[float] = []
        background = attempt.get("background_quality_gate") or {}
        metrics = background.get("metrics") or {}
        limits = background.get("limits") or {}
        for metric_name, limit_name in (
            ("chroma_noise_score", "chroma_noise_score_max"),
            ("background_mottling_score", "background_mottling_score_max"),
        ):
            try:
                value = float(metrics.get(metric_name))
                limit = float(limits.get(limit_name))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and math.isfinite(limit) and limit > 0.0:
                excesses.append(max(0.0, value / limit - 1.0))
        if metrics.get("chroma_load_growth_display_reference_exempted"):
            load_value = metrics.get("display_reference_chroma_load_ratio")
            load_limit = metrics.get("display_reference_chroma_load_ratio_max")
        elif metrics.get("chroma_load_growth_low_absolute_exempted"):
            load_value = metrics.get("chroma_load")
            load_limit = metrics.get("chroma_load_low_absolute_effective_max")
        else:
            load_value = metrics.get("chroma_load_growth")
            load_limit = limits.get("chroma_load_growth_max")
        try:
            numeric_load = float(load_value)
            numeric_load_limit = float(load_limit)
        except (TypeError, ValueError):
            numeric_load = numeric_load_limit = math.nan
        if (
            math.isfinite(numeric_load)
            and math.isfinite(numeric_load_limit)
            and numeric_load_limit > 0.0
        ):
            excesses.append(
                max(0.0, numeric_load / numeric_load_limit - 1.0)
            )
        color_gate = attempt.get("color_vector_gate") or {}
        try:
            color_value = float(
                (color_gate.get("metrics") or {}).get("chromaticity_l1_half_p95")
            )
            color_limit = float(
                (color_gate.get("limits") or {}).get(
                    "chromaticity_l1_half_p95_hard_max"
                )
            )
        except (TypeError, ValueError):
            color_value = color_limit = math.nan
        if math.isfinite(color_value) and math.isfinite(color_limit) and color_limit > 0.0:
            excesses.append(max(0.0, color_value / color_limit - 1.0))
        score_report = attempt.get("presentation_score") or cls._stage7_presentation_score(
            attempt
        )
        try:
            score = float(score_report.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        if not math.isfinite(score):
            score = 0.0
        return (
            stretch_saturated_penalty,
            max(excesses, default=0.0),
            sum(excesses),
            -score,
            len(attempt.get("diagnostics") or []),
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
    def _stage7_candidate_is_technically_safe(attempt: Dict[str, Any]) -> bool:
        """Enforce the non-overridable floor for any formal Stage 7 delivery."""

        if attempt.get("status") != "ok" or not attempt.get("stem"):
            return False
        pixel_stats = attempt.get("pixel_stats") or {}
        if any(
            bool(pixel_stats.get(key))
            for key in (
                "is_nearly_black",
                "is_nearly_white",
                "invalid_dynamic_range",
            )
        ):
            return False
        for key in ("p50", "p99", "max", "dynamic_range"):
            if key not in pixel_stats:
                continue
            try:
                value = float(pixel_stats[key])
            except (TypeError, ValueError):
                return False
            if not math.isfinite(value):
                return False
            if key in {"p50", "dynamic_range"} and value <= 0.0:
                return False

        base_gates = (attempt.get("quality_gates") or {}).get("base_quality") or {}
        for name in ("black_pixel_ratio", "highlight_clip_ratio", "star_size_growth"):
            if bool((base_gates.get(name) or {}).get("hard_failed", False)):
                return False
        local_gates = (
            (attempt.get("target_local_quality") or {}).get("quality_gates") or {}
        )
        if any(
            str(name).startswith("local_core_")
            and bool((gate or {}).get("hard_failed", False))
            for name, gate in local_gates.items()
        ):
            return False
        if not bool(
            (attempt.get("starless_structure_quality") or {}).get(
                "accepted",
                True,
            )
        ):
            return False
        if not bool(
            (attempt.get("transform_loss_gate") or {}).get("accepted", True)
        ):
            return False
        method = str(attempt.get("method") or "")
        required_contract = {
            "linked_mtf": "mtf_reference_quality",
            stage7_stretch_metrics.DISPLAY90_LEGACY_METHOD: (
                "display90_curve_quality"
            ),
            stage7_stretch_metrics.DISPLAY_LUMINANCE_VECTOR_METHOD: (
                "display90_curve_quality"
            ),
        }.get(method)
        if required_contract and not bool(
            (attempt.get(required_contract) or {}).get("accepted", False)
        ):
            return False
        for contract_name in ("mtf_reference_quality", "display90_curve_quality"):
            contract = attempt.get(contract_name) or {}
            if contract.get("status") not in {None, "not_applicable", "unavailable"} and not bool(
                contract.get("accepted", False)
            ):
                return False
        return True


    @classmethod
    def _stage7_review_candidate_is_safe(cls, attempt: Dict[str, Any]) -> bool:
        """Allow review delivery only for structurally safe brightness/chroma rejects."""
        if not cls._stage7_candidate_is_technically_safe(attempt):
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
        frozen_masks: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reduce background chroma while preserving luminance and signal masks."""
        max_strength = _clamp_float(
            getattr(self.cfg, "stage7_chroma_rescue_max_strength", 0.90),
            0.10,
            0.90,
        )
        strength = _clamp_float(strength, 0.10, max_strength)
        source = np.asarray(image_data)
        canonical_source, _pixel_domain = canonicalize_stage7_pixels_01(source)
        if (
            isinstance(frozen_masks, dict)
            and frozen_masks.get("background_mask") is not None
        ):
            rgb = _to_rgb_float_fullres(canonical_source).astype(
                np.float32,
                copy=False,
            )
            gray = (
                0.2126 * rgb[0]
                + 0.7152 * rgb[1]
                + 0.0722 * rgb[2]
            ).astype(np.float32)
            masks = dict(frozen_masks)
            mask_source = "stage6_frozen_roi"
        else:
            masks = self._stage8_generate_starless_masks(canonical_source)
            rgb = np.asarray(masks["rgb"], dtype=np.float32)
            gray = np.asarray(masks["gray"], dtype=np.float32)
            mask_source = "candidate_fallback"
        background_weight = np.clip(
            np.asarray(masks["background_mask"], dtype=np.float32),
            0.0,
            1.0,
        )
        signal_layers = [
            np.asarray(masks[name], dtype=np.float32)
            for name in (
                "subject_mask",
                "core_mask",
                "nebula_mask",
                "faint_nebula_mask",
                "galaxy_signal_mask",
                "star_mask",
            )
            if masks.get(name) is not None
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
            source,
            np.clip(rescued_rgb, 0.0, 1.0),
        )
        return rescued, {
            "mode": "background_chroma_rescue",
            "strength": strength,
            "mask_source": mask_source,
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


    def _stage7_stretch_risk_score(
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
        bg_min = stage7_effective_bg_median_min(self.cfg.stage7_bg_median_min)
        if metrics.bg_median < bg_min:
            score += ((bg_min - metrics.bg_median) / bg_min) * 8.0
        if metrics.bg_median < 0.005:
            score += ((0.005 - metrics.bg_median) / 0.005) * 12.0
        black_max = max(float(self.cfg.stage7_black_pixel_ratio_max), 1e-4)
        if metrics.black_pixel_ratio > black_max:
            score += ((metrics.black_pixel_ratio - black_max) / black_max) * 3.0
        clip_max = max(float(self.cfg.stage7_highlight_clip_ratio_max), 1e-5)
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
        stage_policy_value = policy.get("stage7_stretch") or {}
        stage_policy = stage_policy_value if isinstance(stage_policy_value, dict) else {}
        candidate_modes = [
            str(item).strip().lower()
            for item in (stage_policy.get("candidate_mode") or [])
            if str(item).strip()
        ]
        enabled = bool(getattr(self.cfg, "stage7_target_aware_stretch_enabled", True))
        target_profile_value = getattr(self, "target_profile", {}) or {}
        target_profile = (
            target_profile_value
            if isinstance(target_profile_value, dict)
            else {}
        )
        secondary_labels = {
            str(item).strip()
            for item in (target_profile.get("secondary_labels") or [])
            if str(item).strip()
        }
        composite_target_names = {
            str(item.get("name") or "").strip().lower()
            for item in (target_profile.get("composite_targets") or [])
            if isinstance(item, dict) and str(item.get("name") or "").strip()
        }

        profile = {
            "enabled": enabled,
            "name": "generic_balanced",
            "target_type": target_type,
            "policy_name": policy_name,
            "policy_candidate_modes": candidate_modes,
            "secondary_labels": sorted(secondary_labels),
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
                    "cand_a_p50_multiplier": 0.82,
                    "cand_b_p50_multiplier": 0.78,
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
            if (
                {"lagoon nebula", "trifid nebula"}.issubset(
                    composite_target_names
                )
                and {"emission_red", "reflection_blue"}.issubset(
                    secondary_labels
                )
            ):
                profile.update(
                    {
                        "name": "bright_core_composite_reveal",
                        "cand_a_p50_multiplier": 0.94,
                        "cand_b_p50_multiplier": 0.92,
                        "highlight_scale": 0.80,
                        "cand_a_stretch_multiplier": 0.96,
                        "cand_b_stretch_multiplier": 0.94,
                        "cand_a_pixel_params": {
                            **profile["cand_a_pixel_params"],
                            "faint_boost": 0.026,
                            "core_protection": 0.80,
                            "shadow_chroma_damping": 0.20,
                            "faint_saturation_boost": 0.040,
                        },
                        "reason": (
                            "reveal Lagoon and Trifid in one physical field while "
                            "retaining masked core and star protection"
                        ),
                    }
                )
        elif (
            target_type in {"globular_cluster", "open_cluster", "reflection_nebula_cluster"}
            or "star_color_preserving_stretch" in mode_set
        ):
            has_nebulosity_context = bool(
                secondary_labels
                & {"large_nebulosity", "faint_outer_cloud", "emission_red"}
            )
            profile.update(
                {
                    "name": "star_colour_preserve",
                    "cand_a_p50_multiplier": (
                        0.68 if has_nebulosity_context else 0.85
                    ),
                    "cand_b_p50_multiplier": (
                        0.62 if has_nebulosity_context else 0.80
                    ),
                    "highlight_scale": (
                        0.78 if has_nebulosity_context else 0.82
                    ),
                    "cand_a_stretch_multiplier": (
                        0.84 if has_nebulosity_context else 0.92
                    ),
                    "cand_b_stretch_multiplier": (
                        0.80 if has_nebulosity_context else 0.88
                    ),
                    "cand_b_method": "asinh",
                    "secondary_context_overlay": (
                        "stellar_primary_with_nebulosity"
                        if has_nebulosity_context
                        else None
                    ),
                    "reason": (
                        "protect stellar colour while lowering the field "
                        "background around real emission nebulosity"
                        if has_nebulosity_context
                        else "avoid linked GHS star bloat and protect stellar colour"
                    ),
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
                    "cand_a_p50_multiplier": 0.94,
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
                    "cand_a_p50_multiplier": 1.00,
                    "cand_b_p50_multiplier": 1.00,
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
            if secondary_labels & {"faint_outer_cloud", "large_nebulosity"}:
                profile.update(
                    {
                        "name": "widefield_faint_signal",
                        "cand_a_p50_multiplier": 0.98,
                        "cand_b_p50_multiplier": 0.96,
                        "highlight_scale": 0.88,
                        "cand_a_stretch_multiplier": 1.03,
                        "cand_b_stretch_multiplier": 1.00,
                        "cand_b_ghs_amount": 1.03,
                        "reason": (
                            "retain catalogued faint outer nebulosity with a "
                            "bounded widefield lift"
                        ),
                    }
                )
            else:
                profile.update(
                    {
                        "name": "widefield_subject_separation",
                        "cand_a_p50_multiplier": 0.88,
                        "cand_b_p50_multiplier": 0.82,
                        "highlight_scale": 0.86,
                        "cand_a_stretch_multiplier": 0.98,
                        "cand_b_stretch_multiplier": 0.96,
                        "cand_b_ghs_amount": 1.02,
                        "reason": (
                            "separate the widefield subject without lifting the "
                            "whole field to the faint-cloud target"
                        ),
                    }
                )
        return profile


    def _stage7_vivid_chroma_factor(
        self,
        target_stretch: Dict[str, Any],
        *,
        saturation_ratio: Optional[float] = None,
        saturation_goal: Optional[float] = None,
    ) -> float:
        """Return the bounded target-aware colour factor for vivid-safe."""

        profile_name = str(target_stretch.get("name") or "").strip().lower()
        channel_semantics = str(
            getattr(self, "_channel_semantics", "unknown") or "unknown"
        ).strip().lower()
        if (
            channel_semantics == "narrowband_composite"
            and profile_name
            in {
                "bright_core_protect",
                "widefield_nebulosity",
                "widefield_faint_signal",
                "widefield_subject_separation",
                "dark_nebula_separation",
                "generic_balanced",
            }
        ):
            return 1.18
        if profile_name in {
            "bright_core_protect",
            "widefield_nebulosity",
            "widefield_faint_signal",
            "widefield_subject_separation",
            "dark_nebula_separation",
        }:
            return 1.12
        if profile_name == "galaxy_core_halo_balance":
            try:
                ratio = float(saturation_ratio)
                goal = min(float(saturation_goal), 0.50)
            except (TypeError, ValueError):
                ratio = 0.0
                goal = 0.0
            if math.isfinite(ratio) and math.isfinite(goal) and ratio > 1e-6:
                return max(1.08, min(6.0, goal / ratio))
            return 1.08
        if profile_name == "star_colour_preserve":
            return 1.06
        return 1.08


    def _stage7_conditional_candidate_a(
        self,
        *,
        baseline_method: str,
        baseline_params: Dict[str, Any],
        target_stretch: Dict[str, Any],
        source_profile: Optional[Dict[str, Any]],
        cand_a_calibration: Dict[str, Any],
        baseline_pixel_stats: Dict[str, Any],
        manual_parameter_mode: bool,
        starless_recomposition_planned: bool,
        source_stem: str,
        star_separation_state: str,
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Conditionally replace cand_a without changing the configured pool."""
        target_type = str(
            target_stretch.get("target_type") or "generic_low_snr_safe"
        ).strip().lower()
        source_role = (
            "starless_linear"
            if starless_recomposition_planned
            else "star_bearing_linear"
        )
        profile = (
            dict(source_profile)
            if isinstance(source_profile, dict)
            else {
                "status": "unavailable",
                "reason": "source profile unavailable",
            }
        )
        profile_evidence = {
            key: profile.get(key)
            for key in (
                "status",
                "trusted_background",
                "background_sample_count",
                "background_coverage",
                "background_median",
                "background_mad",
                "background_sigma",
                "p99",
                "p99_5",
                "p99_9",
                "extended_background_median",
                "extended_background_sigma",
                "trusted_galaxy_roi",
                "subject_mask_available",
                "subject_sample_count",
                "subject_measurement_coverage",
                "subject_measurement_method",
                "subject_p50",
                "subject_p75",
                "subject_p90",
                "extended_subject_p75",
                "faint_signal_contrast",
                "faint_signal_snr_proxy",
                "stretch_noise_regime",
                "physical_snr",
                "reason",
            )
        }
        candidate_policy = str(
            getattr(self.cfg, "stage7_candidate_policy", "auto_display90")
            or "auto_display90"
        ).strip().lower()
        report: Dict[str, Any] = {
            "schema": "starun.stage7-candidate-a-routing.v1",
            "status": "not_applicable",
            "reason_code": "target_not_in_strict_scope",
            "requested_algorithm": None,
            "applied_method": baseline_method,
            "baseline_method": baseline_method,
            "source_role": source_role,
            "source_stem": source_stem,
            "star_separation_state": star_separation_state,
            "target_type": target_type,
            "candidate_pool": (
                "cand_a_cand_b_display90"
                if candidate_policy == "auto_display90"
                else "cand_a_cand_b"
            ),
            "quality_failure_fallback": (
                "cand_b_then_existing_quantile_and_chroma_fallbacks"
            ),
            "source_profile": profile_evidence,
            "constraint_results": {
                "automatic_parameter_mode": not manual_parameter_mode,
                "target_aware_enabled": bool(target_stretch.get("enabled", False)),
                "candidate_policy": candidate_policy,
                "candidate_policy_auto_pool": candidate_policy
                in {"auto_display90", "auto_dual"},
                "source_profile_available": profile.get("status") == "available",
                "strict_route_match": False,
            },
        }
        preview_contract = dict(cand_a_calibration or {})
        if manual_parameter_mode:
            report.update(
                status="not_applied",
                reason_code="manual_parameter_mode",
                reason="manual Asinh/GHS contract disables adaptive replacement",
            )
            return baseline_method, dict(baseline_params), report, preview_contract
        if not bool(target_stretch.get("enabled", False)):
            report.update(
                status="not_applied",
                reason_code="target_aware_stretch_disabled",
                reason="target-aware Stage7 stretch disabled by config",
            )
            return baseline_method, dict(baseline_params), report, preview_contract
        if candidate_policy not in {"auto_display90", "auto_dual"}:
            report.update(
                status="not_applied",
                reason_code="candidate_policy_preserves_configured_algorithms",
                reason=(
                    f"{candidate_policy} keeps the configured Stage7 candidate "
                    "algorithm contract"
                ),
                candidate_policy=candidate_policy,
            )
            return baseline_method, dict(baseline_params), report, preview_contract

        algorithm = None
        if (
            not starless_recomposition_planned
            and source_stem == "stage6_passthrough"
            and star_separation_state == "target_bypass"
            and target_type
            in {
                "globular_cluster",
                "open_cluster",
                "reflection_nebula_cluster",
            }
        ):
            algorithm = "iterative_masked_mtf"
        elif (
            starless_recomposition_planned
            and source_stem == "stage6_starless"
            and target_type in {"large_galaxy", "small_galaxy"}
        ):
            algorithm = "dual_stage_mtf_ghs"
        if algorithm is None:
            return baseline_method, dict(baseline_params), report, preview_contract
        report["requested_algorithm"] = algorithm
        report["constraint_results"]["strict_route_match"] = True

        if algorithm == "iterative_masked_mtf" and not bool(
            getattr(
                self.cfg,
                "stage7_iterative_masked_mtf_enabled",
                True,
            )
        ):
            report.update(
                status="not_applied",
                reason_code="algorithm_disabled",
                reason="Iterative Masked MTF disabled by config",
            )
            return baseline_method, dict(baseline_params), report, preview_contract
        if algorithm == "dual_stage_mtf_ghs" and not bool(
            getattr(
                self.cfg,
                "stage7_dual_stage_mtf_ghs_enabled",
                True,
            )
        ):
            report.update(
                status="not_applied",
                reason_code="algorithm_disabled",
                reason="Dual-stage MTF+GHS disabled by config",
            )
            return baseline_method, dict(baseline_params), report, preview_contract

        if profile.get("status") != "available":
            report.update(
                status="not_applied",
                reason_code="source_profile_unavailable",
                reason=str(profile.get("reason") or "source profile unavailable"),
            )
            return baseline_method, dict(baseline_params), report, preview_contract
        try:
            target_background = float(
                cand_a_calibration.get("target_p50", 0.0) or 0.0
            )
            source_p50 = float(
                baseline_pixel_stats.get("p50", 0.0) or 0.0
            )
            source_p99 = float(
                baseline_pixel_stats.get("p99", 0.0) or 0.0
            )
        except (TypeError, ValueError):
            target_background = source_p50 = source_p99 = 0.0
        if (
            not cand_a_calibration
            or not all(
                math.isfinite(value)
                for value in (target_background, source_p50, source_p99)
            )
            or target_background <= 0.0
            or source_p50 <= 0.0
            or source_p99 <= source_p50
        ):
            report.update(
                status="not_applied",
                reason_code="preview_calibration_unavailable",
                reason="conditional cand_a requires a valid preview P50/P99 contract",
            )
            return baseline_method, dict(baseline_params), report, preview_contract

        max_derivative = _clamp_float(
            getattr(
                self.cfg,
                "stage7_conditional_lut_max_derivative",
                5000.0,
            ),
            250.0,
            20000.0,
        )
        if algorithm == "iterative_masked_mtf":
            if profile.get("trusted_background") is not True:
                report.update(
                    status="not_applied",
                    reason_code="trusted_background_unavailable",
                    reason="strict target-bypass route requires a frozen background ROI",
                )
                return baseline_method, dict(baseline_params), report, preview_contract
            calibration = stage7_stretch_metrics.calibrate_iterative_masked_mtf(
                profile,
                target_background=target_background,
                iterations=getattr(
                    self.cfg,
                    "stage7_iterative_masked_mtf_iterations",
                    16,
                ),
                max_derivative=max_derivative,
                source_p50=source_p50,
                source_p99=source_p99,
            )
        else:
            if profile.get("trusted_galaxy_roi") is not True:
                report.update(
                    status="not_applied",
                    reason_code="trusted_galaxy_roi_unavailable",
                    reason="strict Starless galaxy route requires a trusted frozen ROI",
                )
                return baseline_method, dict(baseline_params), report, preview_contract
            try:
                snr_proxy = float(profile.get("faint_signal_snr_proxy"))
            except (TypeError, ValueError):
                snr_proxy = math.inf
            snr_max = _clamp_float(
                getattr(self.cfg, "stage7_dual_stage_weak_snr_max", 8.0),
                3.5,
                12.0,
            )
            if not math.isfinite(snr_proxy) or snr_proxy >= snr_max:
                report.update(
                    status="not_applied",
                    reason_code="signal_not_weak",
                    reason=(
                        "Dual-stage route requires finite frozen ROI proxy SNR "
                        f"below {snr_max:.3f}; measured={snr_proxy}"
                    ),
                    weak_snr_max=snr_max,
                )
                return baseline_method, dict(baseline_params), report, preview_contract
            subject_min = _clamp_float(
                getattr(
                    self.cfg,
                    "stage7_dual_stage_subject_p90_min",
                    0.20,
                ),
                0.10,
                0.35,
            )
            subject_max = _clamp_float(
                getattr(
                    self.cfg,
                    "stage7_dual_stage_subject_p90_max",
                    0.26,
                ),
                0.15,
                0.50,
            )
            if subject_max <= subject_min:
                subject_max = min(0.50, subject_min + 0.01)
            snr_fraction = _clamp_float(
                (snr_proxy - 3.5) / max(snr_max - 3.5, 1e-6),
                0.0,
                1.0,
            )
            configured_target_subject_p90 = (
                subject_min + (subject_max - subject_min) * snr_fraction
            )
            # The stronger preview-linked background target must retain a
            # meaningful subject/background separation.  Without this floor,
            # the former absolute 0.20-0.26 subject target can fall too close
            # to the new ~75-80% galaxy background target and make the bounded
            # dual-stage calibration infeasible.
            subject_contrast_floor_ratio = 2.0
            target_subject_p90 = min(
                0.50,
                max(
                    configured_target_subject_p90,
                    target_background * subject_contrast_floor_ratio,
                ),
            )
            calibration = stage7_stretch_metrics.calibrate_dual_stage_mtf_ghs(
                profile,
                target_background=target_background,
                target_subject_p90=target_subject_p90,
                ghs_b=getattr(self.cfg, "stage7_dual_stage_ghs_b", 5.0),
                ghs_d_min=getattr(
                    self.cfg,
                    "stage7_dual_stage_ghs_d_min",
                    0.5,
                ),
                ghs_d_max=getattr(
                    self.cfg,
                    "stage7_dual_stage_ghs_d_max",
                    12.0,
                ),
                ghs_search_steps=getattr(
                    self.cfg,
                    "stage7_dual_stage_ghs_search_steps",
                    47,
                ),
                max_derivative=max_derivative,
                source_p50=source_p50,
                source_p99=source_p99,
            )
            report.update(
                weak_snr_max=snr_max,
                weak_snr_fraction=snr_fraction,
                configured_target_subject_p90=configured_target_subject_p90,
                subject_contrast_floor_ratio=subject_contrast_floor_ratio,
                target_subject_p90=target_subject_p90,
            )

        if calibration.get("status") != "ok":
            report.update(
                status="not_applied",
                reason_code="calibration_failed",
                reason=str(calibration.get("reason") or "LUT calibration failed"),
                calibration=calibration,
            )
            return baseline_method, dict(baseline_params), report, preview_contract

        predicted_p50 = float(calibration["predicted_global_p50"])
        predicted_p99 = float(calibration["predicted_global_p99"])
        resolved = dict(calibration.get("resolved") or {})
        calibrated_parameter = (
            float(resolved.get("iterations", 0.0) or 0.0)
            if algorithm == "iterative_masked_mtf"
            else float(resolved.get("ghs_stretch_factor", 0.0) or 0.0)
        )
        parameter_max = (
            32.0
            if algorithm == "iterative_masked_mtf"
            else float(
                (calibration.get("parameters") or {}).get(
                    "ghs_d_max",
                    12.0,
                )
            )
        )
        preview_contract.update(
            {
                "calibration_method": algorithm,
                "target_contract": "conditional_linked_lut_prediction",
                "auto_asinh_target_p50": cand_a_calibration.get("target_p50"),
                "auto_asinh_target_p99": cand_a_calibration.get("target_p99"),
                "target_background_roi": target_background,
                "target_p50": predicted_p50,
                "target_p99": predicted_p99,
                "predicted_p50": predicted_p50,
                "predicted_p99": predicted_p99,
                "calibrated_parameter": algorithm,
                "calibrated_stretch": calibrated_parameter,
                "stretch_max": parameter_max,
            }
        )
        report.update(
            status="applied",
            reason_code="strict_conditions_passed",
            reason="conditional cand_a replacement passed source and LUT preflight",
            applied_method=algorithm,
            calibration=calibration,
            linked_rgb=True,
            runtime_external_dependency=False,
        )
        report["constraint_results"].update(
            {
                "trusted_background": profile.get("trusted_background") is True,
                "trusted_galaxy_roi": profile.get("trusted_galaxy_roi") is True,
                "weak_signal": (
                    algorithm != "dual_stage_mtf_ghs"
                    or float(profile.get("faint_signal_snr_proxy"))
                    < float(report.get("weak_snr_max", math.inf))
                ),
                "lut_contract_accepted": bool(
                    (calibration.get("lut_contract") or {}).get("accepted")
                ),
            }
        )
        return (
            algorithm,
            {"calibration": calibration},
            report,
            preview_contract,
        )


    def _stage7_compact_stretch_candidates(
        self,
        baseline_quality: Optional[QualityMetrics],
        baseline_adaptive: Optional[Dict[str, Any]],
        baseline_pixel_stats: Optional[Dict[str, Any]] = None,
        preview_pixel_stats: Optional[Dict[str, Any]] = None,
        *,
        starless_recomposition_planned: bool = False,
        source_profile: Optional[Dict[str, Any]] = None,
        source_stem: str = "",
        star_separation_state: str = "",
        baseline_image_data: Optional[np.ndarray] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        stats = self._stage7_baseline_background_stats(baseline_quality, baseline_adaptive)
        bg_median = float(stats.get("bg_median", 0.0) or 0.0)
        bg_std = float(stats.get("bg_std", 0.0) or 0.0)
        adaptation: Dict[str, Any] = {
            "mode": "default_compact",
            "bg_median": bg_median,
            "bg_std": bg_std,
            "reason": "baseline background not extremely low",
            "conditional_source_profile": (
                dict(source_profile)
                if isinstance(source_profile, dict)
                else {
                    "schema": (
                        stage7_stretch_metrics.CONDITIONAL_SOURCE_PROFILE_SCHEMA
                    ),
                    "status": "unavailable",
                    "reason": "source profile not supplied",
                }
            ),
        }
        manual_stretch_fields = {
            field
            for field in getattr(self, "_task_manual_override_fields", ())
            if field
            in {
                "asinh_stretch",
                "asinh_offset",
                "ghs_shadowsclip",
                "ghs_stretchamount",
            }
        }
        explicit_manual_stretch_fields = set(manual_stretch_fields)
        configured_parameter_mode = str(
            getattr(self.cfg, "stage7_processing_mode", "auto") or "auto"
        ).strip().lower()
        manual_parameter_mode = bool(
            configured_parameter_mode == "manual" or manual_stretch_fields
        )
        if manual_parameter_mode and not manual_stretch_fields:
            manual_stretch_fields = {
                "asinh_stretch",
                "asinh_offset",
                "ghs_shadowsclip",
                "ghs_stretchamount",
            }
        adaptation["parameter_mode"] = (
            "manual" if manual_parameter_mode else "auto"
        )
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
            getattr(self.cfg, "stage7_preview_cand_a_p50_ratio", 0.85)
            * float(target_stretch.get("cand_a_p50_multiplier", 1.0)),
            0.10,
            0.90,
        )
        cand_b_preview_scale = _clamp_float(
            getattr(self.cfg, "stage7_preview_cand_b_p50_ratio", 0.85)
            * float(target_stretch.get("cand_b_p50_multiplier", 1.0)),
            0.10,
            cand_a_preview_scale,
        )
        preview_asinh_p50_max = _clamp_float(
            getattr(self.cfg, "stage7_preview_asinh_p50_max", 0.26),
            0.12,
            0.35,
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
            target_p50_max=preview_asinh_p50_max,
        )
        cand_b_stretch, cand_b_calibration = _stage7_preview_calibrated_stretch(
            pixel_stats,
            preview_stats,
            offset=float(cand_b_params["asinh_offset"]),
            preview_scale=cand_b_preview_scale,
            highlight_scale=float(target_stretch.get("highlight_scale", 0.90)),
            fallback=float(cand_b_params["asinh_stretch"]),
            stretch_max=preview_stretch_max,
            target_p50_max=preview_asinh_p50_max,
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
            adaptation["effective_preview_p50_ratios"] = {
                "candidate_a": cand_a_preview_scale,
                "candidate_b": cand_b_preview_scale,
            }
            adaptation["reason"] = (
                str(adaptation.get("reason") or "")
                + "; Asinh strengths calibrated from stage7_preview_ref"
            ).strip("; ")

        cand_a_method = str(target_stretch.get("cand_a_method") or "asinh")
        cand_b_method = str(target_stretch.get("cand_b_method") or "asinh_ghs")
        if manual_stretch_fields:
            manual_values = {
                field: float(getattr(self.cfg, field))
                for field in sorted(manual_stretch_fields)
            }
            for field in ("asinh_stretch", "asinh_offset"):
                if field in manual_values:
                    cand_a_params[field] = manual_values[field]
                    cand_b_params[field] = manual_values[field]
            for field in ("ghs_shadowsclip", "ghs_stretchamount"):
                if field in manual_values:
                    cand_b_params[field] = manual_values[field]
            if explicit_manual_stretch_fields.intersection(
                {"ghs_shadowsclip", "ghs_stretchamount"}
            ):
                cand_b_method = "asinh_ghs"
            preview_contract = adaptation.get("preview_calibration")
            preview_contract = (
                preview_contract
                if isinstance(preview_contract, dict)
                else {}
            )
            target_contracts: Dict[str, Any] = {}
            for candidate_key, method, params in (
                ("candidate_a", cand_a_method, cand_a_params),
                ("candidate_b", cand_b_method, cand_b_params),
            ):
                calibration = preview_contract.get(candidate_key)
                if method == "asinh":
                    rebased = _stage7_rebase_manual_asinh_calibration(
                        calibration if isinstance(calibration, dict) else {},
                        pixel_stats,
                        params,
                    )
                    preview_contract[candidate_key] = rebased
                    target_contracts[candidate_key] = {
                        "mode": (
                            "manual_contract_rebased"
                            if rebased
                            else "manual_contract_unavailable"
                        ),
                        "safety_gates": (
                            "rebased_preview_target_and_independent_gates"
                            if rebased
                            else "independent_safety_gates"
                        ),
                    }
                else:
                    # A preview target derived for an Asinh-only transform is
                    # not a valid contract for GHS or local masked transforms.
                    preview_contract[candidate_key] = {}
                    target_contracts[candidate_key] = {
                        "mode": "manual_independent_safety_gates",
                        "method": method,
                        "safety_gates": (
                            "visibility_background_core_highlight_and_structure"
                        ),
                    }
            if preview_contract:
                adaptation["preview_calibration"] = preview_contract
            adaptation["manual_parameter_overrides"] = {
                "fields": sorted(manual_stretch_fields),
                "values": manual_values,
                "source": "signed_processing_parameters",
                "adaptive_replacement_disabled": True,
                "target_contracts": target_contracts,
            }
        if (
            starless_recomposition_planned
            and cand_b_calibration
            and not manual_stretch_fields
        ):
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
                        0.85,
                    ),
                    0.40,
                    0.90,
                )
                target_ceiling = _clamp_float(
                    getattr(
                        self.cfg,
                        "stage7_starless_linked_mtf_p50_max",
                        0.26,
                    ),
                    0.15,
                    0.35,
                )
                linked_profile_multiplier = _clamp_float(
                    target_stretch.get("cand_b_p50_multiplier", 1.0),
                    0.50,
                    1.00,
                )
                effective_preview_ratio = _clamp_float(
                    preview_ratio * linked_profile_multiplier,
                    0.40,
                    0.90,
                )
                target_p50 = _clamp_float(
                    max(
                        calibrated_target_p50,
                        target_floor,
                        preview_p50 * effective_preview_ratio,
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
                    "preview_p50_ratio": effective_preview_ratio,
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

        (
            cand_a_method,
            cand_a_params,
            candidate_a_replacement,
            conditional_preview_contract,
        ) = Stage6ServiceMixin._stage7_conditional_candidate_a(
            self,
            baseline_method=cand_a_method,
            baseline_params=cand_a_params,
            target_stretch=target_stretch,
            source_profile=source_profile,
            cand_a_calibration=dict(
                (
                    adaptation.get("preview_calibration") or {}
                ).get("candidate_a")
                or cand_a_calibration
                or {}
            ),
            baseline_pixel_stats=pixel_stats,
            manual_parameter_mode=manual_parameter_mode,
            starless_recomposition_planned=starless_recomposition_planned,
            source_stem=str(source_stem or ""),
            star_separation_state=str(star_separation_state or ""),
        )
        adaptation["candidate_a_replacement"] = candidate_a_replacement
        preview_contract = adaptation.get("preview_calibration")
        if (
            isinstance(preview_contract, dict)
            and conditional_preview_contract
        ):
            preview_contract["candidate_a"] = conditional_preview_contract

        candidate_policy = str(
            getattr(self.cfg, "stage7_candidate_policy", "auto_display90")
            or "auto_display90"
        ).strip().lower()
        display90_requested = candidate_policy in {
            "auto_display90",
            "display90_only",
        }
        display90_eligible = bool(
            display90_requested
            and not manual_parameter_mode
            and starless_recomposition_planned
            and str(source_stem or "") == "stage6_starless"
            and str(star_separation_state or "")
            == StarSeparationState.ACCEPTED.value
            and baseline_image_data is not None
        )
        display90_calibration: Dict[str, Any] = {
            "schema": stage7_stretch_metrics.DISPLAY90_STRETCH_SCHEMA,
            "status": "not_applicable",
            "method": stage7_stretch_metrics.DISPLAY_LUMINANCE_VECTOR_METHOD,
            "eligibility": {
                "policy_requested": display90_requested,
                "automatic_parameter_mode": not manual_parameter_mode,
                "starless_recomposition_planned": bool(
                    starless_recomposition_planned
                ),
                "source_stem": str(source_stem or ""),
                "star_separation_state": str(star_separation_state or ""),
                "baseline_pixels_available": baseline_image_data is not None,
            },
            "reason": "Display90 is outside the active Stage7 route",
        }
        display_candidates: List[Dict[str, Any]] = []
        display_ladder: Dict[str, Any] = {
            "status": "not_applicable",
            "policy": "bounded_low_mid_high",
            "tiers": [],
        }
        if display90_eligible:
            requested_strength = float(
                getattr(self.cfg, "stage7_display90_strength", 0.90)
            )
            high_strength = _clamp_float(
                requested_strength,
                stage7_stretch_metrics.DISPLAY90_STRENGTH_MIN,
                stage7_stretch_metrics.DISPLAY90_STRENGTH_MAX,
            )
            strength_ladder = [
                (
                    "cand_display82",
                    "stage7_cand_display82",
                    max(
                        stage7_stretch_metrics.DISPLAY90_STRENGTH_MIN,
                        high_strength - 0.08,
                    ),
                    "mid",
                ),
            ]
            if str(target_stretch.get("name") or "").strip().lower() == (
                "galaxy_core_halo_balance"
            ):
                strength_ladder.append(
                    (
                        "cand_display86",
                        "stage7_cand_display86",
                        max(
                            stage7_stretch_metrics.DISPLAY90_STRENGTH_MIN,
                            high_strength - 0.04,
                        ),
                        "mid_high",
                    )
                )
            strength_ladder.extend((
                (
                    "cand_display70",
                    "stage7_cand_display70",
                    max(
                        stage7_stretch_metrics.DISPLAY90_STRENGTH_MIN,
                        high_strength - 0.20,
                    ),
                    "low",
                ),
                (
                    "cand_display90",
                    "stage7_cand_display90",
                    high_strength,
                    "high",
                ),
            ))
            max_derivative = _clamp_float(
                getattr(
                    self.cfg,
                    "stage7_conditional_lut_max_derivative",
                    5000.0,
                ),
                250.0,
                20000.0,
            )
            try:
                display_curve = ui_preview.build_linked_display_curve_contract(
                    baseline_image_data,
                    max_side=ui_preview.DEFAULT_PREVIEW_MAX_SIDE,
                )
                preview_contract = adaptation.setdefault(
                    "preview_calibration",
                    {
                        "source": "stage7_gui_linked_display_curve",
                        "reason": (
                            "Display ladder uses serialized GUI D curves; "
                            "preview remains observer-only"
                        ),
                    },
                )
                seen_strengths: set[float] = set()
                for candidate_name, candidate_stem, strength, tier in strength_ladder:
                    rounded_strength = round(float(strength), 6)
                    if rounded_strength in seen_strengths:
                        continue
                    seen_strengths.add(rounded_strength)
                    calibration = stage7_stretch_metrics.calibrate_display90_linked_lut(
                        baseline_image_data,
                        display_curve,
                        strength=rounded_strength,
                        max_derivative=max_derivative,
                        target_type=self._active_target_type(),
                    )
                    calibration["eligibility"] = {
                        "policy_requested": True,
                        "automatic_parameter_mode": True,
                        "starless_recomposition_planned": True,
                        "source_stem": "stage6_starless",
                        "star_separation_state": StarSeparationState.ACCEPTED.value,
                        "baseline_pixels_available": True,
                    }
                    calibration["requested_strength"] = requested_strength
                    calibration["effective_strength"] = rounded_strength
                    calibration["strength_tier"] = tier
                    display_ladder["tiers"].append(
                        {
                            "name": candidate_name,
                            "tier": tier,
                            "strength": rounded_strength,
                            "status": calibration.get("status"),
                        }
                    )
                    if tier == "high":
                        display90_calibration = calibration
                    if calibration.get("status") != "ok":
                        continue
                    preview_contract[
                        "candidate_" + candidate_name.removeprefix("cand_")
                    ] = {
                        "calibration_method": (
                            stage7_stretch_metrics.DISPLAY_LUMINANCE_VECTOR_METHOD
                        ),
                        "target_contract": (
                            "authenticated_display_ladder_lut_prediction"
                        ),
                        "preview_p50": (
                            calibration[
                                "gui_display_reference_quantiles"
                            ]["rgb_flat"]["p50"]
                        ),
                        "target_p50": calibration["target_p50"],
                        "target_p90": calibration["target_p90"],
                        "target_p99": calibration["target_p99"],
                        "predicted_p50": calibration["predicted_p50"],
                        "predicted_p90": calibration["predicted_p90"],
                        "predicted_p99": calibration["predicted_p99"],
                        "calibrated_parameter": "display90_strength",
                        "calibrated_stretch": rounded_strength,
                        "stretch_max": (
                            stage7_stretch_metrics.DISPLAY90_STRENGTH_MAX
                        ),
                    }
                    display_candidates.append({
                        "name": candidate_name,
                        "stem": candidate_stem,
                        "method": (
                            stage7_stretch_metrics.DISPLAY_LUMINANCE_VECTOR_METHOD
                        ),
                        "params": {
                            "calibration": calibration,
                            "strength_tier": tier,
                        },
                        "adaptation": adaptation,
                    })
                display_ladder["status"] = (
                    "ok" if display_candidates else "unavailable"
                )
            except (
                IndexError,
                TypeError,
                ValueError,
                FloatingPointError,
            ) as error:
                display90_calibration = {
                    "schema": stage7_stretch_metrics.DISPLAY90_STRETCH_SCHEMA,
                    "status": "unavailable",
                    "method": (
                        stage7_stretch_metrics.DISPLAY_LUMINANCE_VECTOR_METHOD
                    ),
                    "reason": str(error),
                }
                display_ladder = {
                    "status": "unavailable",
                    "policy": "bounded_low_mid_high",
                    "tiers": [],
                    "reason": str(error),
                }
        adaptation["display90_calibration"] = display90_calibration
        adaptation["display_ladder"] = display_ladder

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
        candidates.extend(display_candidates)
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


    def _run_stage7_stretching_candidates(
        self,
    ) -> Tuple[bool, bool, List[str], str]:
        self._stage7_matched_domain_transfer = None
        self._stage7_stretch_validated_rescue = False
        self._stage7_stretch_forced_delivery = False
        self._stage7_forced_delivery_reasons = []
        self._stage7_destructive_core_rejected = False
        self._stage7_bright_core_integrity_rejected_reasons = []
        self._stage7_revoked_pair_id = None
        self._stage7_last_vivid_chroma_execution = {}
        self._stage7_stretch_fallback_reason = None
        self._stage7_review_source = None
        self._stage7_background_color_review_required = False
        self._stage7_background_color_review_gate = {
            "status": "not_run",
            "requires_review": False,
        }
        messages: List[str] = []
        source_stem = str(
            getattr(self, "_stage7_stretch_source", None) or "stage6_starless"
        )
        try:
            self.cmd_with_check("load", source_stem)
        except (CommandError, SirilError) as e:
            messages.append(f"stage7 source load failed: {self._short_text(e, 180)}")
            return False, False, messages, ""

        try:
            baseline_image_data, baseline_pixel_stats = (
                self._stage7_current_pixel_snapshot()
            )
        except (
            CommandError,
            DataError,
            SirilError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            messages.append(
                "stage7 pixel-domain contract failed: "
                f"{self._short_text(error, 180)}"
            )
            return False, False, messages, ""
        try:
            baseline_quality = measure_quality_metrics(baseline_image_data)
        except (RuntimeError, TypeError, ValueError, IndexError, FloatingPointError):
            baseline_quality = None
        baseline_adaptive = (
            self._adaptive_features_from_image(baseline_image_data)
            if hasattr(self, "_adaptive_features_from_image")
            else {}
        )
        source_pixel_domain = dict(
            baseline_pixel_stats.get("pixel_domain") or {}
        )
        messages.append(
            "stage7 pixel domain canonicalized "
            f"source_dtype={source_pixel_domain.get('source_dtype', 'unknown')} "
            f"scale={source_pixel_domain.get('normalization_scale', 1.0)} "
            "canonical=float32[0,1]"
        )
        frozen_background_masks: Optional[Dict[str, Any]] = None
        existing_scene_runtime = getattr(self, "_stage3_scene_support", None)
        reuse_existing_scene_runtime = bool(
            isinstance(existing_scene_runtime, dict)
            and str(
                ((existing_scene_runtime.get("manifest") or {}).get("reason_code"))
                or ""
            )
            == "legacy_checkpoint_without_scene_support"
        )
        shared_scene_runtime = (
            existing_scene_runtime
            if reuse_existing_scene_runtime
            else scene_support.load_scene_support(
                self.process_dir,
                expected_shape=tuple(np.asarray(baseline_image_data).shape),
            )
            if self.process_dir is not None and baseline_image_data is not None
            else {
                "status": "unavailable",
                "manifest": scene_support.unavailable_scene_support(
                    "stage7 baseline pixels unavailable",
                    reason_code="scene_support_unavailable",
                ),
                "valid_mask": None,
                "saturation_map": None,
            }
        )
        self._stage3_scene_support = shared_scene_runtime
        shared_scene_support_summary = scene_support.scene_support_summary(
            shared_scene_runtime
        )
        frozen_background_sampling: Dict[str, Any] = {
            "status": "unavailable",
            "method": "candidate_local_background_fallback",
        }
        if baseline_image_data is not None:
            try:
                source_masks, frozen_background_sampling = (
                    stage8_pixels.build_signal_excluded_background_masks(
                        self,
                        baseline_image_data,
                    )
                )
                background_mask = np.asarray(source_masks["background_mask"])
                if (
                    background_mask.ndim == 2
                    and background_mask.size > 0
                    and np.all(np.isfinite(background_mask))
                ):
                    frozen_background_masks = {
                        key: np.array(value, dtype=np.float32, copy=True)
                        for key, value in source_masks.items()
                        if key
                        in {
                            "background_mask",
                            "core_mask",
                            "nebula_mask",
                            "faint_nebula_mask",
                            "galaxy_signal_mask",
                        }
                    }
                    subject_layers = [
                        frozen_background_masks[name]
                        for name in (
                            "core_mask",
                            "nebula_mask",
                            "faint_nebula_mask",
                            "galaxy_signal_mask",
                        )
                        if frozen_background_masks.get(name) is not None
                    ]
                    if subject_layers:
                        frozen_background_masks["subject_mask"] = np.clip(
                            np.maximum.reduce(subject_layers),
                            0.0,
                            1.0,
                        ).astype(np.float32, copy=False)
                    else:
                        frozen_background_masks["subject_mask"] = np.clip(
                            1.0 - background_mask,
                            0.0,
                            1.0,
                        ).astype(np.float32, copy=False)
                    frozen_background_sampling["source"] = f"{source_stem}.fit"
            except (
                IndexError,
                RuntimeError,
                TypeError,
                ValueError,
                FloatingPointError,
            ):
                frozen_background_masks = None
        if isinstance(frozen_background_masks, dict):
            spatial_shape = tuple(
                int(value) for value in frozen_background_masks["background_mask"].shape
            )
            shared_valid = shared_scene_runtime.get("valid_mask")
            shared_saturation = shared_scene_runtime.get("saturation_map")
            valid_weight = (
                np.asarray(shared_valid, dtype=np.float32)
                if shared_valid is not None
                and np.asarray(shared_valid).shape == spatial_shape
                else None
            )
            saturation_weight = (
                (np.asarray(shared_saturation) > 0).astype(np.float32)
                if shared_saturation is not None
                and np.asarray(shared_saturation).shape == spatial_shape
                else None
            )
            catalog_weight = scene_support.catalog_aperture_mask(
                shared_scene_runtime,
                spatial_shape,
            )
            if valid_weight is not None:
                frozen_background_masks["background_mask"] = (
                    np.asarray(
                        frozen_background_masks["background_mask"],
                        dtype=np.float32,
                    )
                    * valid_weight
                ).astype(np.float32, copy=False)
                frozen_background_masks["shared_valid_mask"] = np.asarray(
                    shared_valid,
                    dtype=np.uint8,
                )
            if (
                shared_saturation is not None
                and np.asarray(shared_saturation).shape == spatial_shape
            ):
                frozen_background_masks["original_saturation_map"] = np.asarray(
                    shared_saturation,
                    dtype=np.uint8,
                )
            protected_layers = [
                value
                for value in (saturation_weight, catalog_weight)
                if value is not None
            ]
            if protected_layers:
                protected = np.clip(
                    np.maximum.reduce(protected_layers), 0.0, 1.0
                ).astype(np.float32, copy=False)
                frozen_background_masks["background_mask"] *= 1.0 - protected
            frozen_background_sampling["shared_scene_support"] = (
                shared_scene_support_summary
            )
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
                    try:
                        starmask_pixels, _starmask_pixel_domain = (
                            canonicalize_stage7_pixels_01(loaded_starmask)
                        )
                        starmask_image_data = starmask_pixels
                        if (
                            isinstance(frozen_background_masks, dict)
                            and baseline_image_data is not None
                        ):
                            baseline_rgb = _to_rgb_float_fullres(
                                baseline_image_data
                            )
                            star_weight = _stage7_expanded_star_halo_protection(
                                starmask_pixels,
                                tuple(baseline_rgb.shape[1:]),
                                expand_iterations=1,
                            )
                            if star_weight is not None:
                                frozen_background_masks["star_mask"] = star_weight
                                frozen_background_masks["subject_mask"] = np.maximum(
                                    frozen_background_masks["subject_mask"],
                                    star_weight,
                                ).astype(np.float32, copy=False)
                    except (TypeError, ValueError) as error:
                        messages.append(
                            "stage7 starmask pixel-domain validation failed: "
                            f"{self._short_text(error, 160)}"
                        )
            try:
                self.cmd_with_check("load", source_stem)
            except (CommandError, SirilError) as error:
                messages.append(
                    "stage7 source reload after starmask sampling failed: "
                    f"{self._short_text(error, 160)}"
                )
                return False, False, messages, ""
        active_target_type = str(
            self._active_target_type() or "generic_low_snr_safe"
            if hasattr(self, "_active_target_type")
            else "generic_low_snr_safe"
        ).strip().lower()
        strict_core_evidence = stage7_quality.strict_bright_core_target_evidence(
            active_target_type,
            getattr(self, "target_profile", None),
        )
        target_local_reference_data: Optional[np.ndarray] = baseline_image_data
        target_local_reference_available = True
        if bool(strict_core_evidence.get("strict", False)):
            frozen_stage6_input = self._read_image_by_stem("stage6_input")
            if frozen_stage6_input is None:
                target_local_reference_data = None
                target_local_reference_available = False
                messages.append(
                    "strict bright-core target is missing frozen stage6_input; "
                    "formal Starless delivery will be rejected"
                )
            else:
                try:
                    target_local_reference_data, _reference_domain = (
                        canonicalize_stage7_pixels_01(frozen_stage6_input)
                    )
                    messages.append(
                        "strict bright-core target-local masks frozen from "
                        "stage6_input with Stage6 starmask exclusion"
                    )
                except (TypeError, ValueError) as error:
                    target_local_reference_data = None
                    target_local_reference_available = False
                    messages.append(
                        "strict bright-core frozen reference validation failed: "
                        f"{self._short_text(error, 160)}"
                    )
        messages.append(
            "stage7 primary stretch candidates: stage7_cand_a, stage7_cand_b, "
            "optional stage7_cand_display70/82/90; "
            "galaxy routes also evaluate stage7_cand_display86; "
            "preview=stage7_preview_ref; vivid-safe and chroma rescue are conditional"
        )
        messages.append(
            "stage7 Starless structure gate="
            + ("starmask_rank_local" if starmask_image_data is not None else "generic_fallback")
        )
        if self.process_dir:
            for pattern in (
                "stage7_candidate_*.fit",
                "stage7_cand_*.fit",
                "stage7_with_stars_hdr*.fit",
                "stage7_preview_ref.fit",
                "stage7_selected*.fit",
                "stage7_stretched.fit",
            ):
                for stale_path in self.process_dir.glob(pattern):
                    try:
                        stale_path.unlink()
                    except OSError as e:
                        self.log.warn(f"[Stage7] stale stretch candidate cleanup failed: {e}")

        preview_saved = False
        preview_image_data: Optional[np.ndarray] = None
        preview_pixel_stats: Dict[str, Any] = {}
        preview_quality: Optional[QualityMetrics] = None
        try:
            self.cmd_with_check("load", source_stem)
            self.cmd_with_check("autostretch", "-linked")
            preview_saved = self._save_stage_output("stage7_preview_ref")
            if preview_saved:
                preview_image_data, preview_pixel_stats = (
                    self._stage7_current_pixel_snapshot()
                )
                preview_quality = measure_quality_metrics(preview_image_data)
        except (CommandError, SirilError, RuntimeError, TypeError, ValueError) as e:
            messages.append(f"stage7 preview_ref failed: {self._short_text(e, 160)}")
        preview_quality_metrics = (
            asdict(preview_quality) if preview_quality is not None else {}
        )
        preview_rendition_metrics = (
            stage7_stretch_metrics.measure_frozen_rendition_metrics(
                preview_image_data,
                frozen_background_masks,
            )
            if preview_image_data is not None
            else {
                "schema": stage7_stretch_metrics.RENDITION_METRICS_SCHEMA,
                "status": "unavailable",
                "metrics": {},
                "reason": "preview pixels unavailable",
            }
        )

        conditional_target_type = str(
            self._active_target_type() or "generic_low_snr_safe"
            if hasattr(self, "_active_target_type")
            else "generic_low_snr_safe"
        ).strip().lower()
        conditional_parameter_mode = str(
            getattr(self.cfg, "stage7_processing_mode", "auto") or "auto"
        ).strip().lower()
        conditional_manual_fields = {
            str(field)
            for field in getattr(self, "_task_manual_override_fields", ())
            if str(field)
            in {
                "asinh_stretch",
                "asinh_offset",
                "ghs_shadowsclip",
                "ghs_stretchamount",
            }
        }
        conditional_candidate_policy = str(
            getattr(self.cfg, "stage7_candidate_policy", "auto_display90")
            or "auto_display90"
        ).strip().lower()
        conditional_cluster_route = bool(
            source_stem == "stage6_passthrough"
            and str(getattr(self, "_star_separation_state", "") or "")
            == "target_bypass"
            and conditional_target_type
            in {
                "globular_cluster",
                "open_cluster",
                "reflection_nebula_cluster",
            }
            and getattr(
                self.cfg,
                "stage7_iterative_masked_mtf_enabled",
                True,
            )
        )
        conditional_galaxy_route = bool(
            source_stem == "stage6_starless"
            and conditional_target_type in {"large_galaxy", "small_galaxy"}
            and getattr(
                self.cfg,
                "stage7_dual_stage_mtf_ghs_enabled",
                True,
            )
        )
        conditional_profile_requested = bool(
            baseline_image_data is not None
            and conditional_parameter_mode == "auto"
            and not conditional_manual_fields
            and bool(
                getattr(
                    self.cfg,
                    "stage7_target_aware_stretch_enabled",
                    True,
                )
            )
            and conditional_candidate_policy
            in {"auto_display90", "auto_dual"}
            and (conditional_cluster_route or conditional_galaxy_route)
        )
        conditional_source_profile = (
            stage7_stretch_metrics.build_conditional_stretch_source_profile(
                baseline_image_data,
                frozen_background_masks,
                frozen_background_sampling,
            )
            if conditional_profile_requested
            else {
                "schema": (
                    stage7_stretch_metrics.CONDITIONAL_SOURCE_PROFILE_SCHEMA
                ),
                "status": "unavailable",
                "measurement_skipped": True,
                "reason": (
                    "conditional source profile not required by the active "
                    "Stage7 routing contract"
                    if baseline_image_data is not None
                    else "baseline pixels unavailable"
                ),
            }
        )
        candidate_list, stretch_adaptation = self._stage7_compact_stretch_candidates(
            baseline_quality,
            baseline_adaptive,
            baseline_pixel_stats,
            preview_pixel_stats,
            starless_recomposition_planned=source_stem == "stage6_starless",
            source_profile=conditional_source_profile,
            source_stem=source_stem,
            star_separation_state=str(
                getattr(self, "_star_separation_state", "") or ""
            ),
            baseline_image_data=baseline_image_data,
        )
        closed_form_mtf_reference: Dict[str, Any] = {
            "schema": "starun.stage7-mtf-reference.v1",
            "status": "not_applicable",
            "role": "candidate_evaluation_reference",
            "standard_scope": "closed_form_parameter_and_p50_conformance",
            "reference_only": True,
            "final_candidate": False,
            "reason": "cand_b does not use linked MTF",
        }
        if (
            len(candidate_list) > 1
            and str(candidate_list[1].get("method") or "") == "linked_mtf"
        ):
            anchor_params = dict(candidate_list[1].get("params") or {})
            if baseline_image_data is None:
                statistical_shadow_reference: Dict[str, Any] = {
                    "status": "unavailable",
                    "role": "reference_only",
                    "method": "closed_form_linked_mtf",
                    "source": (
                        stage7_stretch_metrics.STATISTICAL_MTF_REFERENCE_SOURCE
                    ),
                    "equivalence_scope": (
                        "linked_rgb_no_curves_no_hdr_no_normalize"
                    ),
                    "final_candidate": False,
                    "reason": "baseline pixels unavailable",
                }
            else:
                statistical_shadow_reference = (
                    stage7_stretch_metrics.build_statistical_mtf_reference(
                        baseline_image_data,
                        anchor_params.get("target_background", 0.18),
                        blackpoint_sigma=getattr(
                            self.cfg,
                            "stage7_mtf_reference_blackpoint_sigma",
                            5.0,
                        ),
                        reference_mask=(
                            frozen_background_masks.get("background_mask")
                            if isinstance(frozen_background_masks, dict)
                            else None
                        ),
                    )
                )
            closed_form_mtf_reference = {
                "schema": "starun.stage7-mtf-reference.v1",
                "status": "active",
                "role": "candidate_evaluation_reference",
                "standard_scope": "closed_form_parameter_and_p50_conformance",
                "reference_only": True,
                "final_candidate": False,
                "active_anchor": {
                    "candidate": "cand_b",
                    "method": "closed_form_linked_mtf",
                    "blackpoint_method": "noise_floor_safety_margin",
                    "params": anchor_params,
                },
                "statistical_shadow_reference": statistical_shadow_reference,
            }
        self._stage7_closed_form_mtf_reference = copy.deepcopy(
            closed_form_mtf_reference
        )
        stretch_adaptation["closed_form_mtf_reference"] = (
            closed_form_mtf_reference
        )
        stretch_adaptation["candidate_ranking"] = {
            "policy": STAGE7_CANDIDATE_RANKING_POLICY,
            "configurable_weights": False,
            "key_order": list(STAGE7_CANDIDATE_RANKING_FIELDS),
        }
        target_stretch = stretch_adaptation.get("target_aware") or {}
        candidate_a_replacement = (
            stretch_adaptation.get("candidate_a_replacement") or {}
        )
        messages.append(
            "stage7 conditional cand_a replacement "
            f"status={candidate_a_replacement.get('status', 'not_applicable')}, "
            f"method={candidate_a_replacement.get('applied_method', 'asinh')}, "
            f"reason={candidate_a_replacement.get('reason_code', 'unknown')}"
        )
        messages.append(
            "stage7 target-aware stretch "
            f"profile={target_stretch.get('name', 'generic_balanced')}, "
            f"target={target_stretch.get('target_type', 'generic_low_snr_safe')}, "
            f"policy={target_stretch.get('policy_name', 'generic_low_snr_safe')}"
        )
        preview_calibration = stretch_adaptation.get("preview_calibration")
        if closed_form_mtf_reference.get("status") == "active":
            statistical_reference = (
                closed_form_mtf_reference.get("statistical_shadow_reference")
                or {}
            )
            messages.append(
                "stage7 closed-form MTF reference anchor=cand_b; "
                "statistical lower-half-recentered MAD shadow="
                f"{statistical_reference.get('status', 'unavailable')} "
                "(reference-only)"
            )
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
        candidate_policy = str(
            getattr(self.cfg, "stage7_candidate_policy", "auto_display90")
            or "auto_display90"
        ).strip().lower()
        candidate_list = _stage7_candidates_for_policy(
            candidate_list,
            candidate_policy,
        )
        rendition_intent = str(
            getattr(self.cfg, "stage7_rendition_intent", "vivid_safe")
            or "vivid_safe"
        ).strip().lower()
        if rendition_intent not in {"vivid_safe", "balanced", "conservative"}:
            rendition_intent = "vivid_safe"
        if candidate_policy == "auto_display90":
            if rendition_intent == "balanced":
                candidate_list = [
                    candidate
                    for candidate in candidate_list
                    if str(candidate.get("name") or "")
                    not in {"cand_display70", "cand_display90"}
                ]
            elif rendition_intent == "conservative":
                candidate_list = [
                    candidate
                    for candidate in candidate_list
                    if str(candidate.get("name") or "")
                    not in {
                        "cand_display82",
                        "cand_display86",
                        "cand_display90",
                    }
                ]
        stretch_adaptation["candidate_policy"] = candidate_policy
        stretch_adaptation["rendition_intent"] = rendition_intent
        stretch_adaptation["delivery_mode"] = "starless"
        stretch_adaptation["enabled_candidates"] = [
            str(candidate.get("name")) for candidate in candidate_list
        ]
        failure_action = str(
            getattr(self.cfg, "stage7_failure_action", "auto_fallback")
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
                preview_pixel_stats=preview_pixel_stats,
                preview_quality_metrics=preview_quality_metrics,
                preview_rendition_metrics=preview_rendition_metrics,
                target_local_reference_data=target_local_reference_data,
                target_local_reference_available=target_local_reference_available,
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
                    preview_pixel_stats=preview_pixel_stats,
                    preview_quality_metrics=preview_quality_metrics,
                    preview_rendition_metrics=preview_rendition_metrics,
                    target_local_reference_data=target_local_reference_data,
                    target_local_reference_available=target_local_reference_available,
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

        vivid_parent_attempts = [
            attempt
            for attempt in attempts
            if attempt.get("status") == "ok"
            and attempt.get("stem")
            and bool(attempt.get("allowed_as_final", False))
            and str(attempt.get("method") or "") != "vivid_safe_chroma"
            and (
                ((attempt.get("rendition_metrics") or {}).get("candidate") or {}).get(
                    "status"
                )
                == "available"
            )
        ]
        if (
            rendition_intent == "vivid_safe"
            and bool(
                getattr(self.cfg, "stage7_vivid_subject_chroma_enabled", True)
            )
            and str(
                getattr(self.cfg, "stage7_processing_mode", "auto") or "auto"
            ).strip().lower()
            == "auto"
            and vivid_parent_attempts
            and candidate_policy in {"auto_display90", "auto_dual"}
            and preview_rendition_metrics.get("status") == "available"
        ):
            vivid_parent = min(
                vivid_parent_attempts,
                key=self._stage7_candidate_selection_key,
            )
            score_report = vivid_parent.get("presentation_score") or {}
            saturation_ratio = self._stage7_retention_ratio(
                vivid_parent,
                "saturation_median",
            )
            saturation_goal = float(
                (score_report.get("goals") or {}).get(
                    "saturation_median",
                    0.80,
                )
            )
            safety_headroom = float(
                score_report.get("safety_headroom", 0.0) or 0.0
            )
            if (
                (saturation_ratio is None or saturation_ratio < saturation_goal)
                and safety_headroom >= 0.15
            ):
                vivid_factor = self._stage7_vivid_chroma_factor(
                    target_stretch,
                    saturation_ratio=saturation_ratio,
                    saturation_goal=saturation_goal,
                )
                vivid_candidate = {
                    "name": "cand_vivid_safe",
                    "stem": "stage7_cand_vivid_safe",
                    "method": "vivid_safe_chroma",
                    "params": {
                        "factor": vivid_factor,
                        "parent_name": vivid_parent.get("name"),
                        "parent_candidate": {
                            "name": vivid_parent.get("name"),
                            "method": vivid_parent.get("method"),
                            "params": dict(vivid_parent.get("params") or {}),
                        },
                    },
                    "adaptation": vivid_parent.get("adaptation"),
                    "calibration_candidate": str(
                        vivid_parent.get("calibration_candidate")
                        or vivid_parent.get("name")
                        or ""
                    ),
                    "tone_parent": str(vivid_parent.get("name") or ""),
                }
                vivid_attempt = self._stage7_evaluate_stretch_candidate(
                    vivid_candidate,
                    source_stem=source_stem,
                    baseline_quality=baseline_quality,
                    baseline_image_data=baseline_image_data,
                    starmask_image_data=starmask_image_data,
                    baseline_background_quality=baseline_background_quality,
                    frozen_background_masks=frozen_background_masks,
                    target_stretch=target_stretch,
                    preview_pixel_stats=preview_pixel_stats,
                    preview_quality_metrics=preview_quality_metrics,
                    preview_rendition_metrics=preview_rendition_metrics,
                    target_local_reference_data=target_local_reference_data,
                    target_local_reference_available=target_local_reference_available,
                )
                vivid_attempt["tone_parent"] = str(
                    vivid_parent.get("name") or ""
                )
                vivid_attempt["vivid_chroma_execution"] = dict(
                    getattr(self, "_stage7_last_vivid_chroma_execution", {})
                    or {}
                )
                attempts.append(vivid_attempt)
                messages.append(
                    "stage7 vivid-safe subject chroma candidate evaluated "
                    f"(parent={vivid_parent.get('name')}, "
                    f"factor={vivid_factor:.2f}, "
                    f"accepted={str(bool(vivid_attempt.get('allowed_as_final'))).lower()})"
                )

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

        if (
            best_attempt is None
            and failure_action == "auto_fallback"
        ):
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
                    preview_pixel_stats=preview_pixel_stats,
                    preview_quality_metrics=preview_quality_metrics,
                    preview_rendition_metrics=preview_rendition_metrics,
                    target_local_reference_data=target_local_reference_data,
                    target_local_reference_available=target_local_reference_available,
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

        if (
            best_attempt is None
            and failure_action == "auto_fallback"
        ):
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
                        self._stage7_chroma_rescue_strengths(
                            rescue_source_attempt
                        ),
                        start=1,
                    ):
                        rescue_name = f"chroma_rescue_{rescue_index}"
                        rescue_stem = f"stage7_cand_rescue_{rescue_index}"
                        try:
                            self.cmd_with_check("load", source_stem)
                            replay_ok, replay_used = (
                                self._execute_stage7_stretch_candidate(
                                    rescue_root_candidate,
                                    starmask_image_data=starmask_image_data,
                                    frozen_masks=frozen_background_masks,
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
                                    frozen_masks=frozen_background_masks,
                                )
                            )
                            self._set_current_image_pixeldata(
                                rescued_data,
                                label=f"Stage7 {rescue_name}",
                            )
                            rescued_image_data, pixel_stats = (
                                self._stage7_current_pixel_snapshot()
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
                                        rescued_image_data,
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
                                self._validate_stage7_stretch_quality(
                                    baseline_quality,
                                    enforce_star_growth=enforce_star_growth,
                                    current_image_data=rescued_image_data,
                                )
                            )
                            advisories = list(
                                getattr(
                                    metrics,
                                    "_stage7_quality_advisories",
                                    [],
                                )
                                if metrics is not None
                                else []
                            )
                            quality_gates: Dict[str, Any] = {
                                "base_quality": dict(
                                    getattr(
                                        metrics,
                                        "_stage7_quality_gates",
                                        {},
                                    )
                                    if metrics is not None
                                    else {}
                                )
                            }
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
                            quality_metrics_dict = asdict(metrics) if metrics else {}
                            preview_retention = _stage7_preview_retention(
                                pixel_stats,
                                quality_metrics_dict,
                                preview_pixel_stats,
                                preview_quality_metrics,
                            )
                            quality_ok, issues, visibility_gate = (
                                self._stage7_apply_candidate_visibility_gate(
                                    quality_ok,
                                    issues,
                                    pixel_stats,
                                    target_stretch,
                                    preview_retention,
                                )
                            )
                            advisories.extend(
                                visibility_gate.get("advisories") or []
                            )
                            quality_gates["visibility"] = (
                                visibility_gate.get("quality_gate")
                            )
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
                                        0.90,
                                    ),
                                    hard_min_ratio=getattr(
                                        self.cfg,
                                        "stage7_preview_target_p50_hard_min_ratio",
                                        0.80,
                                    ),
                                    max_ratio=getattr(
                                        self.cfg,
                                        "stage7_preview_target_p50_max_ratio",
                                        1.50,
                                    ),
                                    advisory_multiplier=(
                                        stage7_quality.stage7_9_quality_advisory_multiplier(
                                            self.cfg
                                        )
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
                            advisories.extend(
                                preview_target_attainment.get("advisories") or []
                            )
                            quality_gates["preview_target_attainment"] = {
                                "hard_minimum_ratio": preview_target_attainment.get(
                                    "hard_minimum_ratio"
                                ),
                                "hard_maximum_ratio": preview_target_attainment.get(
                                    "hard_maximum_ratio"
                                ),
                                "advisory_multiplier": preview_target_attainment.get(
                                    "advisory_multiplier"
                                ),
                            }

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
                                        (
                                            target_local_reference_data
                                            if target_local_reference_data is not None
                                            else baseline_image_data
                                        ),
                                        rescued_image_data,
                                        str(
                                            target_stretch.get("target_type")
                                            or "generic_low_snr_safe"
                                        ),
                                        self.cfg,
                                        target_profile=getattr(
                                            self, "target_profile", None
                                        ),
                                        starmask=starmask_image_data,
                                        frozen_reference_available=(
                                            target_local_reference_available
                                        ),
                                        valid_mask=(
                                            frozen_background_masks.get(
                                                "shared_valid_mask"
                                            )
                                            if isinstance(
                                                frozen_background_masks, dict
                                            )
                                            else None
                                        ),
                                        original_saturation_map=(
                                            frozen_background_masks.get(
                                                "original_saturation_map"
                                            )
                                            if isinstance(
                                                frozen_background_masks, dict
                                            )
                                            else None
                                        ),
                                    )
                                )
                            color_vector_reference: Dict[str, Any] = {
                                "schema": (
                                    "starun.stage7-color-vector-reference.v1"
                                ),
                                "status": "unavailable",
                                "role": "report_only",
                                "enforced": False,
                                "participates_in_selection": False,
                                "reason": (
                                    "Stage7 color reference inputs unavailable"
                                ),
                            }
                            if baseline_image_data is not None:
                                color_vector_reference = (
                                    stage7_stretch_metrics.assess_rec709_vector_color_reference(
                                        baseline_image_data,
                                        rescued_image_data,
                                    )
                                )
                            multiscale_contrast_reference: Dict[str, Any] = {
                                "schema": (
                                    stage7_stretch_metrics.MULTISCALE_CONTRAST_SCHEMA
                                ),
                                "status": "unavailable",
                                "role": "report_only",
                                "enforced": False,
                                "participates_in_selection": False,
                                "mas_equivalent": False,
                                "reason": (
                                    "Stage7 multiscale reference inputs unavailable"
                                ),
                            }
                            if baseline_image_data is not None:
                                multiscale_contrast_reference = (
                                    stage7_stretch_metrics.assess_multiscale_contrast_reference(
                                        baseline_image_data,
                                        rescued_image_data,
                                    )
                                )
                            rescued_background_quality = self._background_quality_metrics(
                                rescued_image_data,
                                frozen_background_masks,
                            )
                            if rescued_background_quality:
                                rescued_background_quality[
                                    "signal_exclusion_applied"
                                ] = bool(
                                    isinstance(frozen_background_masks, dict)
                                    and any(
                                        frozen_background_masks.get(mask_name)
                                        is not None
                                        for mask_name in (
                                            "core_mask",
                                            "nebula_mask",
                                            "faint_nebula_mask",
                                            "galaxy_signal_mask",
                                        )
                                    )
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
                            advisories.extend(
                                background_quality_gate.get("advisories") or []
                            )
                            quality_gates["background"] = (
                                background_quality_gate.get("quality_gates")
                            )
                            if not bool(local_quality.get("accepted", True)):
                                quality_ok = False
                                issues = [
                                    *issues,
                                    *list(local_quality.get("issues") or []),
                                ]
                            advisories.extend(
                                local_quality.get("advisories") or []
                            )
                            quality_gates["target_local"] = local_quality.get(
                                "quality_gates"
                            )
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
                            advisories.extend(
                                starless_structure_quality.get("advisories") or []
                            )
                            quality_gates["starless_structure"] = (
                                starless_structure_quality.get("quality_gates")
                            )

                            risk_score = self._stage7_stretch_risk_score(
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
                            rescue_semantics = (
                                stage7_stretch_metrics.build_siril_stretch_semantics(
                                    str(rescue_root_candidate.get("method") or ""),
                                    dict(rescue_root_candidate.get("params") or {}),
                                )
                            )
                            rescue_semantics = dict(rescue_semantics)
                            rescue_semantics["reported_method"] = (
                                "background_chroma_rescue"
                            )
                            rescue_semantics["steps"] = [
                                *list(rescue_semantics.get("steps") or []),
                                {
                                    "command": "numpy_background_chroma_rescue",
                                    "argv": [f"strength={rescue_strength:.6f}"],
                                    "full_argv": [
                                        "numpy_background_chroma_rescue",
                                        f"strength={rescue_strength:.6f}",
                                    ],
                                    "role": "post_transform",
                                },
                            ]
                            rescue_transform_loss = {
                                "schema": (
                                    stage7_stretch_metrics.TRANSFORM_LOSS_SCHEMA
                                ),
                                "status": "unavailable",
                                "role": "report_only",
                                "enforced": False,
                                "participates_in_selection": False,
                                "reason": "Stage7 immutable source pixels unavailable",
                            }
                            if baseline_image_data is not None:
                                rescue_transform_loss = (
                                    stage7_stretch_metrics.assess_transform_loss(
                                        baseline_image_data,
                                        rescued_image_data,
                                        method=str(
                                            rescue_root_candidate.get("method") or ""
                                        ),
                                        params=dict(
                                            rescue_root_candidate.get("params") or {}
                                        ),
                                        background_mask=(
                                            frozen_background_masks.get(
                                                "background_mask"
                                            )
                                            if isinstance(
                                                frozen_background_masks,
                                                dict,
                                            )
                                            else None
                                        ),
                                    )
                                )
                            rescue_transform_loss_gate = (
                                self._stage7_transform_loss_gate(
                                    rescue_transform_loss
                                )
                            )
                            rescue_color_vector_gate = (
                                self._stage7_color_vector_gate(
                                    color_vector_reference
                                )
                            )
                            rescue_transform_loss = dict(rescue_transform_loss)
                            rescue_transform_loss.update(
                                role="technical_quality_gate",
                                enforced=True,
                                participates_in_selection=True,
                            )
                            color_vector_reference = dict(
                                color_vector_reference
                            )
                            color_vector_reference.update(
                                role="appearance_quality_gate",
                                enforced=True,
                                participates_in_selection=True,
                            )
                            if not bool(
                                rescue_transform_loss_gate.get(
                                    "accepted",
                                    True,
                                )
                            ):
                                quality_ok = False
                                issues = [
                                    *issues,
                                    *list(
                                        rescue_transform_loss_gate.get(
                                            "issues"
                                        )
                                        or []
                                    ),
                                ]
                            if not bool(
                                rescue_color_vector_gate.get(
                                    "accepted",
                                    True,
                                )
                            ):
                                quality_ok = False
                                issues = [
                                    *issues,
                                    *list(
                                        rescue_color_vector_gate.get("issues")
                                        or []
                                    ),
                                ]
                            advisories.extend(
                                rescue_transform_loss_gate.get("advisories")
                                or []
                            )
                            advisories.extend(
                                rescue_color_vector_gate.get("advisories")
                                or []
                            )
                            quality_gates["transform_loss"] = (
                                rescue_transform_loss_gate
                            )
                            quality_gates["color_vector"] = (
                                rescue_color_vector_gate
                            )
                            candidate_rendition_metrics = (
                                stage7_stretch_metrics.measure_frozen_rendition_metrics(
                                    rescued_image_data,
                                    frozen_background_masks,
                                )
                            )
                            rendition_metrics = {
                                "candidate": candidate_rendition_metrics,
                                "preview": dict(
                                    preview_rendition_metrics or {}
                                ),
                                "retention": (
                                    stage7_stretch_metrics.rendition_metric_retention(
                                        candidate_rendition_metrics,
                                        dict(
                                            preview_rendition_metrics or {}
                                        ),
                                    )
                                ),
                            }
                            subject_brightness = (
                                stage7_stretch_metrics.subject_brightness_selection(
                                    candidate_rendition_metrics,
                                    dict(preview_rendition_metrics or {}),
                                    profile_name=str(
                                        target_stretch.get("name") or ""
                                    ),
                                )
                            )
                            quality_gates["subject_brightness"] = (
                                subject_brightness
                            )
                            if not bool(
                                subject_brightness.get(
                                    "formal_floor_passed",
                                    False,
                                )
                            ):
                                quality_ok = False
                                issues = [
                                    *issues,
                                    "stage7_subject_brightness_floor_unmet",
                                ]
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
                                "advisories": list(
                                    dict.fromkeys(
                                        str(item) for item in advisories
                                    )
                                ),
                                "quality_gates": quality_gates,
                                "metrics": quality_metrics_dict or None,
                                "adaptive_metrics": (
                                    self._adaptive_features_from_image(
                                        rescued_image_data
                                    )
                                    if hasattr(self, "_adaptive_features_from_image")
                                    else None
                                ),
                                "pixel_stats": pixel_stats,
                                "preview_retention": preview_retention,
                                "rendition_metrics": rendition_metrics,
                                "subject_brightness_selection": (
                                    subject_brightness
                                ),
                                "visibility_gate": visibility_gate,
                                "preview_target_attainment": preview_target_attainment,
                                "color_vector_reference": color_vector_reference,
                                "color_vector_gate": rescue_color_vector_gate,
                                "multiscale_contrast_reference": (
                                    multiscale_contrast_reference
                                ),
                                "target_local_quality": local_quality,
                                "starless_structure_quality": starless_structure_quality,
                                "background_quality_gate": background_quality_gate,
                                "transform_semantics": rescue_semantics,
                                "transform_loss": rescue_transform_loss,
                                "transform_loss_gate": (
                                    rescue_transform_loss_gate
                                ),
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
                            rescue_attempt["technical_safe"] = (
                                self._stage7_candidate_is_technically_safe(
                                    rescue_attempt
                                )
                            )
                            rescue_attempt["presentation_score"] = (
                                self._stage7_presentation_score(rescue_attempt)
                            )
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
        destructive_core_rejected, core_rejection_reasons = (
            _stage7_all_saved_candidates_fail_core_gates(
                saved_attempts,
                strict_target=bool(strict_core_evidence.get("strict", False)),
            )
        )
        if destructive_core_rejected:
            self._stage7_destructive_core_rejected = True
            self._stage7_bright_core_integrity_rejected_reasons = list(
                dict.fromkeys(core_rejection_reasons)
            ) or ["local_core_integrity_unavailable"]
            self._stage7_revoked_pair_id = getattr(
                self,
                "_selected_syqon_pair_id",
                None,
            )
            messages.append(
                "stage7_bright_core_integrity_rejected: every saved candidate "
                "failed a non-overridable local_core_* gate; formal output "
                "will be rejected"
            )
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
        best_attempt = deterministic_best_attempt
        forced_attempt: Optional[Dict[str, Any]] = None
        forced_delivery_enabled = bool(
            getattr(self.cfg, "stage7_forced_delivery_enabled", True)
        )
        if (
            best_attempt is None
            and failure_action == "auto_fallback"
            and forced_delivery_enabled
        ):
            forced_candidates = [
                attempt
                for attempt in rejected_attempts
                if self._stage7_candidate_is_technically_safe(attempt)
            ]
            if forced_candidates:
                forced_attempt = min(
                    forced_candidates,
                    key=self._stage7_forced_candidate_selection_key,
                )
                forced_attempt["forced_delivery"] = True
                forced_attempt["formal_accepted"] = False
                forced_attempt["delivery_class"] = "review_only"
                forced_attempt["review_only"] = True
                forced_attempt["forced_delivery_reason_codes"] = list(
                    dict.fromkeys(
                        str(item).split(" ", 1)[0]
                        for item in (forced_attempt.get("diagnostics") or [])
                        if str(item).strip()
                    )
                )
                if not forced_attempt["forced_delivery_reason_codes"]:
                    forced_attempt["forced_delivery_reason_codes"] = [
                        "appearance_quality_gate"
                    ]
                self._stage7_forced_delivery_reasons = list(
                    forced_attempt["forced_delivery_reason_codes"]
                )
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
            forced_attempt
            if best_attempt is None and forced_attempt is not None
            else min(
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
                attempt["selection_role"] = (
                    "selected_review_evidence"
                    if attempt is forced_attempt
                    else "selected_review"
                )
            else:
                attempt["selection_role"] = "not_selected"

        if forced_attempt is not None:
            messages.append(
                "stage7 retained a technically safe appearance reject as "
                f"review evidence only (name={forced_attempt.get('name')}, "
                "reasons="
                + ",".join(
                    forced_attempt.get("forced_delivery_reason_codes") or []
                )
                + ")"
            )
            self.log.warn(
                "Stage7 所有正式候选均未通过画质门；仅保留最佳失败候选"
                "作为诊断证据，正式交付保持关闭"
            )
        elif best_attempt is not None:
            messages.append(
                "stage7 deterministic quality-ranked candidate selected "
                f"(name={best_attempt.get('name')}, "
                f"risk={float(best_attempt.get('risk_score', 0.0) or 0.0):.3f})"
            )
            selected_advisories = [
                str(item)
                for item in (best_attempt.get("advisories") or [])
                if str(item)
            ]
            if selected_advisories:
                advisory_text = ", ".join(selected_advisories[:3])
                advisory_percent = (
                    stage7_quality.stage7_9_quality_advisory_multiplier(
                        self.cfg
                    )
                    - 1.0
                ) * 100.0
                messages.append(
                    "stage7 quality advisory accepted without review-only "
                    f"fallback: {advisory_text}"
                )
                self.log.warn(
                    "Stage7 数值门禁处于 "
                    f"{advisory_percent:.0f}% "
                    "容忍带，保留候选并继续："
                    + advisory_text
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

        background_color_review_gate = (
            self._stage7_uncalibrated_background_color_review_gate(best_attempt)
        )
        self._stage7_background_color_review_gate = dict(
            background_color_review_gate
        )
        self._stage7_background_color_review_required = bool(
            background_color_review_gate.get("requires_review", False)
        )
        if self._stage7_background_color_review_required:
            value = background_color_review_gate.get("value")
            value_text = "unavailable" if value is None else f"{float(value):.3f}"
            messages.append(
                "uncalibrated signal-excluded background color review gate="
                f"{background_color_review_gate.get('status')} "
                f"(chroma_load={value_text}, "
                f"limit={float(background_color_review_gate.get('limit', 0.12)):.3f})"
            )
            self.log.warn(
                "Stage7 未接受物理校色且背景绝对色偏超过复核门；"
                "保留图像像素，不做全局白平衡，最终仅允许 review-only 交付"
            )
        elif background_color_review_gate.get("status") == "advisory":
            messages.append(
                "uncalibrated signal-excluded background color advisory "
                f"(chroma_load={float(background_color_review_gate.get('value', 0.0)):.3f}, "
                f"limit={float(background_color_review_gate.get('limit', 0.12)):.3f}, "
                f"hard_limit={float(background_color_review_gate.get('hard_limit', 0.18)):.3f})"
            )

        selected_fallback_reason = (
            self._stage7_validated_fallback_reason(best_attempt)
            if best_attempt is not None
            else ""
        )
        matched_domain_transfer = _stage7_matched_domain_transfer_contract(
            best_attempt,
            closed_form_mtf_reference,
        )
        self._stage7_matched_domain_transfer = copy.deepcopy(
            matched_domain_transfer
        )
        selection_summary = {
            "strategy": "hard_gate_then_quality_rank_with_safe_review",
            "candidate_policy": candidate_policy,
            "failure_action": failure_action,
            "selector": "deterministic_quality_rank",
            "parameters_owned_by": "code",
            "allowed_candidate_ids": [
                str(attempt.get("name"))
                for attempt in accepted_attempts
                if not bool(attempt.get("explicit_fallback"))
            ],
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
            "rendition_intent": rendition_intent,
            "forced_delivery": False,
            "forced_review_evidence": bool(forced_attempt is not None),
            "forced_delivery_reason_codes": list(
                self._stage7_forced_delivery_reasons
            ),
            "technical_floor": (
                "finite_shape_dynamic_range_clip_core_structure_and_curve_contract"
            ),
            "matched_domain_transfer_method": matched_domain_transfer.get(
                "method"
            ),
            "selected_review": (
                str(review_attempt.get("name"))
                if review_attempt
                else None
            ),
            "review_source": self._stage7_review_source,
            "background_color_review_gate": background_color_review_gate,
            "strict_bright_core_evidence": strict_core_evidence,
            "delivery_mode": "starless",
            "formal_accepted": bool(best_attempt is not None),
            "delivery_class": (
                "formal" if best_attempt is not None else "review_only"
            ),
            "stage7_bright_core_integrity_rejected": bool(
                self._stage7_destructive_core_rejected
            ),
            "stage7_bright_core_integrity_rejected_reasons": list(
                self._stage7_bright_core_integrity_rejected_reasons
            ),
            "revoked_pair_id": self._stage7_revoked_pair_id,
            "quality_advisory_multiplier": (
                stage7_quality.stage7_9_quality_advisory_multiplier(self.cfg)
            ),
        }

        self._write_stage_json(
            "stage7_stretch_quality.json",
            {
                "stage": "stage7_stretch",
                "candidate_policy": candidate_policy,
                "rendition_intent": rendition_intent,
                "failure_action": failure_action,
                "input": f"{source_stem}.fit",
                "preview": {
                    "name": "preview_ref",
                    "file": "stage7_preview_ref.fit" if preview_saved else None,
                    "method": "autostretch -linked",
                    "reference_only": True,
                    "pixel_stats": preview_pixel_stats,
                    "quality": asdict(preview_quality) if preview_quality else None,
                    "rendition_metrics": preview_rendition_metrics,
                },
                "baseline_adaptive": baseline_adaptive,
                "baseline_background_quality": baseline_background_quality,
                "background_sampling": frozen_background_sampling,
                "shared_scene_support": shared_scene_support_summary,
                "background_color_review_gate": background_color_review_gate,
                "strict_bright_core_evidence": strict_core_evidence,
                "delivery_mode": "starless",
                "formal_accepted": bool(best_attempt is not None),
                "delivery_class": (
                    "formal" if best_attempt is not None else "review_only"
                ),
                "stage7_bright_core_integrity_rejected": bool(
                    self._stage7_destructive_core_rejected
                ),
                "stage7_bright_core_integrity_rejected_reasons": list(
                    self._stage7_bright_core_integrity_rejected_reasons
                ),
                "revoked_pair_id": self._stage7_revoked_pair_id,
                "baseline_pixel_stats": baseline_pixel_stats,
                "stretch_adaptation": stretch_adaptation,
                "closed_form_mtf_reference": closed_form_mtf_reference,
                "matched_domain_transfer": matched_domain_transfer,
                "attempts": attempts,
                "selected": best_attempt,
                "galaxy_roi": copy.deepcopy(
                    getattr(
                        self,
                        "_stage6_galaxy_roi_diagnostics",
                        {"status": "not_run", "available": False},
                    )
                ),
                "best_rejected": best_rejected_attempt,
                "selection": selection_summary,
            },
        )
        self._write_stage_json(
            "stretch_candidates_report.json",
            {
                "stage": "stage7_stretch",
                "candidate_policy": candidate_policy,
                "rendition_intent": rendition_intent,
                "failure_action": failure_action,
                "input": f"{source_stem}.fit",
                "stretch_adaptation": stretch_adaptation,
                "closed_form_mtf_reference": closed_form_mtf_reference,
                "matched_domain_transfer": matched_domain_transfer,
                "baseline_background_quality": baseline_background_quality,
                "background_sampling": frozen_background_sampling,
                "shared_scene_support": shared_scene_support_summary,
                "background_color_review_gate": background_color_review_gate,
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
                        "visibility_gate": item.get("visibility_gate"),
                        "preview_target_attainment": item.get("preview_target_attainment"),
                        "rendition_metrics": item.get("rendition_metrics"),
                        "presentation_score": item.get("presentation_score"),
                        "mtf_reference_quality": item.get("mtf_reference_quality"),
                        "display90_curve_quality": item.get(
                            "display90_curve_quality"
                        ),
                        "color_vector_reference": item.get("color_vector_reference"),
                        "color_vector_gate": item.get("color_vector_gate"),
                        "multiscale_contrast_reference": item.get(
                            "multiscale_contrast_reference"
                        ),
                        "target_local_quality": item.get("target_local_quality"),
                        "background_quality_gate": item.get("background_quality_gate"),
                        "transform_semantics": item.get("transform_semantics"),
                        "transform_loss": item.get("transform_loss"),
                        "transform_loss_gate": item.get("transform_loss_gate"),
                        "technical_safe": item.get("technical_safe"),
                        "forced_delivery": bool(item.get("forced_delivery")),
                        "forced_delivery_reason_codes": item.get(
                            "forced_delivery_reason_codes"
                        ),
                        "status": item.get("status"),
                        "diagnostics": item.get("diagnostics"),
                        "advisories": item.get("advisories"),
                        "quality_gates": item.get("quality_gates"),
                        "feedback": item.get("feedback"),
                        "explicit_fallback": bool(item.get("explicit_fallback")),
                        "selection_rank": item.get("selection_rank"),
                        "selection_role": item.get("selection_role"),
                    }
                    for item in attempts
                ],
                "preview": "stage7_preview_ref.fit" if preview_saved else None,
                "preview_rendition_metrics": preview_rendition_metrics,
                "selected": (
                    {
                        "name": best_attempt.get("name"),
                        "source_file": best_attempt.get("file"),
                        "file": "stage7_stretched.fit",
                        "normal_selected": not bool(
                            best_attempt.get("explicit_fallback")
                            or best_attempt.get("forced_delivery")
                        ),
                        "validated_rescue": bool(
                            best_attempt.get("explicit_fallback")
                            and best_attempt.get("allowed_as_final", False)
                        ),
                        "forced_delivery": bool(
                            best_attempt.get("forced_delivery")
                        ),
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
        self._stage7_review_evidence = (
            copy.deepcopy(review_attempt) if review_attempt is not None else None
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
            and best_attempt.get("allowed_as_final", False)
        )
        self._stage7_stretch_fallback_reason = selected_fallback_reason or None
        return (
            True,
            bool(
                best_attempt.get("explicit_fallback")
                or best_attempt.get("forced_delivery")
            ),
            messages,
            selected_method,
        )
