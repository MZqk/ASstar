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
    _component_areas,
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
GALAXY_TARGET_TYPES = frozenset({"galaxy", "large_galaxy", "small_galaxy"})
STARLESS_RANGE_STARMASK_THRESHOLD = 0.02
STARLESS_RANGE_STARMASK_QUANTILE = 0.997
STARLESS_RANGE_EXCLUSION_RADIUS = 3
STARLESS_RANGE_SUPPORT_MIN = 0.05
STARLESS_RANGE_CORRELATION_MIN = 0.85
STARLESS_RANGE_SOURCE_FRACTION_MIN = 0.10

BRIGHT_CORE_STRICT_TARGET_TYPE = "bright_emission_reflection_nebula"
BRIGHT_CORE_ROI_QUANTILE = 0.99
BRIGHT_CORE_ROI_SMOOTH_PASSES = 4
BRIGHT_CORE_STARMASK_THRESHOLD = 0.02
BRIGHT_CORE_STARMASK_EXPANSION = 3
BRIGHT_CORE_ROI_SUPPORT_MIN = 64
BRIGHT_CORE_CAP_THRESHOLD = 0.995
BRIGHT_CORE_OVERSHOOT_DELTA = 0.01
BRIGHT_CORE_INTEGRITY_LIMITS = {
    "new_channel_cap_ratio_max": {"accepted": 0.01, "hard": 0.02},
    "largest_cap_component_ratio": {"accepted": 0.01, "hard": 0.02},
    "closure_abs_error_p99": {"accepted": 0.05, "hard": 0.10},
    "starless_overshoot_ratio_max": {"accepted": 0.10, "hard": 0.20},
    "parity_phase_span_max": {"accepted": 0.01, "hard": 0.02},
}


def strict_bright_core_target_evidence(
    target_type: str,
    target_profile: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return the frozen evidence that enables strict bright-core protection."""
    normalized_type = str(target_type or "").strip().lower()
    profile = target_profile if isinstance(target_profile, dict) else {}
    raw_labels = profile.get("secondary_labels", [])
    labels = {
        str(item).strip().lower()
        for item in raw_labels
        if str(item).strip()
    } if isinstance(raw_labels, (list, tuple, set)) else set()
    raw_features = profile.get("features", {})
    features = raw_features if isinstance(raw_features, dict) else {}
    risks = profile.get("risks", {})
    risks = risks if isinstance(risks, dict) else {}
    bright_core_label = "bright_core" in labels or bool(
        features.get("bright_core", False)
    )
    core_blowout = str(risks.get("core_blowout") or "").strip().lower()
    strict = (
        normalized_type == BRIGHT_CORE_STRICT_TARGET_TYPE
        and core_blowout == "high"
    )
    return {
        "strict": strict,
        "target_type": normalized_type,
        "target_type_matched": normalized_type == BRIGHT_CORE_STRICT_TARGET_TYPE,
        "bright_core_label": bright_core_label,
        "core_blowout": core_blowout or "unknown",
        "routing_rule": (
            "bright_emission_reflection_nebula AND "
            "core_blowout=high; bright_core label is diagnostic-only"
        ),
    }


def is_strict_bright_core_target(
    target_type: str,
    target_profile: Optional[Dict[str, Any]],
) -> bool:
    return bool(
        strict_bright_core_target_evidence(target_type, target_profile).get(
            "strict", False
        )
    )


def _fixed_upper_gate(
    value: float,
    *,
    accepted_limit: float,
    hard_limit: float,
) -> Dict[str, Any]:
    measured = max(float(value), 0.0)
    if measured <= accepted_limit:
        status = "ok"
    elif measured <= hard_limit:
        status = "advisory"
    else:
        status = "hard_failed"
    return {
        "status": status,
        "advisory": status == "advisory",
        "hard_failed": status == "hard_failed",
        "value": measured,
        "accepted_limit": float(accepted_limit),
        "hard_limit": float(hard_limit),
        "severity_ratio": measured / max(float(accepted_limit), 1e-12),
        "fixed_limit": True,
    }


def _dilate_binary_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    expanded = np.asarray(mask, dtype=bool)
    for _ in range(max(int(iterations), 0)):
        padded = np.pad(expanded, 1, mode="constant", constant_values=False)
        expanded = np.logical_or.reduce(
            [
                padded[dy : dy + expanded.shape[0], dx : dx + expanded.shape[1]]
                for dy in range(3)
                for dx in range(3)
            ]
        )
    return expanded


def build_bright_core_roi(
    source_data: Any,
    starmask_data: Any,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Build the fixed Stage 5 bright-core ROI shared by Stage 6 and Stage 7."""
    evidence: Dict[str, Any] = {
        "method": "rec709_box3x3_x4_top1pct_excluding_starmask_dilate3",
        "source_quantile": BRIGHT_CORE_ROI_QUANTILE,
        "smooth_passes": BRIGHT_CORE_ROI_SMOOTH_PASSES,
        "starmask_threshold": BRIGHT_CORE_STARMASK_THRESHOLD,
        "starmask_expansion_pixels": BRIGHT_CORE_STARMASK_EXPANSION,
        "support_min": BRIGHT_CORE_ROI_SUPPORT_MIN,
        "available": False,
        "support": 0,
    }
    if source_data is None or starmask_data is None:
        evidence["reason"] = (
            "source_missing" if source_data is None else "starmask_missing"
        )
        return None, evidence
    try:
        source_rgb = _to_rgb_float_fullres(source_data)
        starmask_rgb = _to_rgb_float_fullres(starmask_data)
    except (TypeError, ValueError, IndexError, FloatingPointError) as error:
        evidence["reason"] = f"invalid_reference:{error}"
        return None, evidence
    if source_rgb.shape != starmask_rgb.shape:
        evidence["reason"] = "reference_shape_mismatch"
        evidence["source_shape"] = list(source_rgb.shape)
        evidence["starmask_shape"] = list(starmask_rgb.shape)
        return None, evidence

    luminance = (
        0.2126 * source_rgb[0]
        + 0.7152 * source_rgb[1]
        + 0.0722 * source_rgb[2]
    ).astype(np.float32)
    broad = luminance
    for _ in range(BRIGHT_CORE_ROI_SMOOTH_PASSES):
        broad = _box_blur_gray(broad)
    try:
        threshold = float(np.quantile(broad, BRIGHT_CORE_ROI_QUANTILE))
    except (TypeError, ValueError, FloatingPointError) as error:
        evidence["reason"] = f"roi_quantile_failed:{error}"
        return None, evidence
    roi = broad >= threshold
    star_pixels = np.max(starmask_rgb, axis=0) > BRIGHT_CORE_STARMASK_THRESHOLD
    excluded = _dilate_binary_mask(star_pixels, BRIGHT_CORE_STARMASK_EXPANSION)
    roi &= ~excluded
    support = int(np.count_nonzero(roi))
    evidence.update(
        {
            "threshold": threshold,
            "support": support,
            "support_ratio": support / float(max(roi.size, 1)),
            "excluded_starmask_pixels": int(np.count_nonzero(excluded)),
            "available": support >= BRIGHT_CORE_ROI_SUPPORT_MIN,
            "reason": (
                "ok" if support >= BRIGHT_CORE_ROI_SUPPORT_MIN else "roi_support_insufficient"
            ),
        }
    )
    return roi, evidence


def assess_bright_core_integrity(
    source_data: Any,
    starless_data: Any,
    starmask_data: Any,
    *,
    target_type: str,
    target_profile: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Detect destructive Stage 6 artifacts inside a frozen bright-core ROI."""
    strict_evidence = strict_bright_core_target_evidence(
        target_type,
        target_profile,
    )
    result: Dict[str, Any] = {
        "schema": "starun.bright-core-integrity.v1",
        "applicable": bool(strict_evidence["strict"]),
        "strict_target_evidence": strict_evidence,
        "limits": {
            name: dict(values)
            for name, values in BRIGHT_CORE_INTEGRITY_LIMITS.items()
        },
        "cap_threshold": BRIGHT_CORE_CAP_THRESHOLD,
        "overshoot_delta": BRIGHT_CORE_OVERSHOOT_DELTA,
        "status": "not_applicable",
        "hard_failed": False,
        "advisory": False,
        "trigger_reasons": [],
        "metrics": {},
        "gates": {},
    }
    if not strict_evidence["strict"]:
        return result

    roi, roi_evidence = build_bright_core_roi(source_data, starmask_data)
    result["roi"] = roi_evidence
    if roi is None or not bool(roi_evidence.get("available", False)):
        reason = str(roi_evidence.get("reason") or "roi_unavailable")
        result.update(
            {
                "status": "hard_failed",
                "hard_failed": True,
                "trigger_reasons": [reason],
            }
        )
        return result
    if starless_data is None:
        result.update(
            {
                "status": "hard_failed",
                "hard_failed": True,
                "trigger_reasons": ["starless_missing"],
            }
        )
        return result

    try:
        source_rgb = _to_rgb_float_fullres(source_data)
        starless_rgb = _to_rgb_float_fullres(starless_data)
        starmask_rgb = _to_rgb_float_fullres(starmask_data)
    except (TypeError, ValueError, IndexError, FloatingPointError) as error:
        result.update(
            {
                "status": "hard_failed",
                "hard_failed": True,
                "trigger_reasons": [f"invalid_pair:{error}"],
            }
        )
        return result
    if source_rgb.shape != starless_rgb.shape or source_rgb.shape != starmask_rgb.shape:
        result.update(
            {
                "status": "hard_failed",
                "hard_failed": True,
                "trigger_reasons": ["pair_shape_mismatch"],
            }
        )
        return result

    support = max(int(np.count_nonzero(roi)), 1)
    newly_capped = (starless_rgb >= BRIGHT_CORE_CAP_THRESHOLD) & (
        source_rgb < BRIGHT_CORE_CAP_THRESHOLD
    )
    cap_ratios = [
        float(np.count_nonzero(newly_capped[channel] & roi)) / float(support)
        for channel in range(3)
    ]
    component_ratios: List[float] = []
    for channel in range(3):
        areas = _component_areas(newly_capped[channel] & roi)
        component_ratios.append(
            float(max(areas, default=0)) / float(support)
        )
    closure = np.abs(source_rgb - (starless_rgb + starmask_rgb))
    closure_values = closure[:, roi]
    closure_p99 = (
        float(np.quantile(closure_values, 0.99))
        if closure_values.size
        else float("inf")
    )
    delta = starless_rgb - source_rgb
    overshoot_ratios = [
        float(np.count_nonzero((delta[channel] > BRIGHT_CORE_OVERSHOOT_DELTA) & roi))
        / float(support)
        for channel in range(3)
    ]
    parity_spans: List[float] = []
    parity_means: List[List[Optional[float]]] = []
    parity_complete = True
    for channel in range(3):
        channel_means: List[Optional[float]] = []
        for phase_y in range(2):
            for phase_x in range(2):
                phase_roi = roi[phase_y::2, phase_x::2]
                phase_values = delta[channel, phase_y::2, phase_x::2][phase_roi]
                if phase_values.size:
                    channel_means.append(float(np.mean(phase_values)))
                else:
                    channel_means.append(None)
                    parity_complete = False
        available_means = [value for value in channel_means if value is not None]
        parity_spans.append(
            float(max(available_means) - min(available_means))
            if available_means
            else float("inf")
        )
        parity_means.append(channel_means)

    metrics = {
        "new_channel_cap_ratios": cap_ratios,
        "new_channel_cap_ratio_max": max(cap_ratios),
        "cap_component_ratios": component_ratios,
        "largest_cap_component_ratio": max(component_ratios),
        "closure_abs_error_p99": closure_p99,
        "starless_overshoot_ratios": overshoot_ratios,
        "starless_overshoot_ratio_max": max(overshoot_ratios),
        "parity_phase_means": parity_means,
        "parity_phase_spans": parity_spans,
        "parity_phase_span_max": max(parity_spans),
        "parity_complete": parity_complete,
    }
    result["metrics"] = metrics
    for name, limits in BRIGHT_CORE_INTEGRITY_LIMITS.items():
        result["gates"][name] = _fixed_upper_gate(
            float(metrics[name]),
            accepted_limit=float(limits["accepted"]),
            hard_limit=float(limits["hard"]),
        )
    if not parity_complete:
        result["gates"]["parity_phase_span_max"].update(
            {"status": "hard_failed", "hard_failed": True, "advisory": False}
        )
        result["trigger_reasons"].append("parity_support_incomplete")

    hard_reasons = [
        name
        for name, gate in result["gates"].items()
        if bool(gate.get("hard_failed", False))
    ]
    advisory_reasons = [
        name
        for name, gate in result["gates"].items()
        if bool(gate.get("advisory", False))
    ]
    result["trigger_reasons"] = list(
        dict.fromkeys(result["trigger_reasons"] + hard_reasons)
    )
    result["hard_failed"] = bool(hard_reasons or not parity_complete)
    result["advisory"] = bool(advisory_reasons) and not result["hard_failed"]
    result["status"] = (
        "hard_failed"
        if result["hard_failed"]
        else "advisory"
        if result["advisory"]
        else "ok"
    )
    return result


def stage7_quality_advisory_multiplier(cfg) -> float:
    """Return the bounded abnormality ratio that remains advisory-only."""
    try:
        value = float(getattr(cfg, "stage7_quality_advisory_multiplier", 2.0))
    except (TypeError, ValueError):
        value = 2.0
    if not math.isfinite(value):
        value = 2.0
    return _clamp_float(value, 1.0, 4.0)


def stage7_9_quality_advisory_multiplier(cfg) -> float:
    """Return the bounded advisory ratio used by recoverable Stage 7-9 gates."""
    try:
        value = float(
            getattr(cfg, "stage7_9_quality_advisory_multiplier", 1.5)
        )
    except (TypeError, ValueError):
        value = 1.5
    if not math.isfinite(value):
        value = 1.5
    return _clamp_float(value, 1.0, 2.0)


def _upper_quality_gate(
    *,
    value: float,
    accepted_limit: float,
    multiplier: float,
) -> Dict[str, Any]:
    measured = max(float(value), 0.0)
    limit = max(float(accepted_limit), 0.0)
    hard_limit = limit * multiplier
    if measured <= limit:
        status = "ok"
    elif limit > 0.0 and measured <= hard_limit:
        status = "advisory"
    else:
        status = "hard_failed"
    severity_ratio = measured / max(limit, 1e-12)
    return {
        "status": status,
        "advisory": status == "advisory",
        "hard_failed": status == "hard_failed",
        "value": measured,
        "accepted_limit": limit,
        "hard_limit": hard_limit,
        "severity_ratio": severity_ratio,
        "advisory_multiplier": multiplier,
    }


def _lower_quality_gate(
    *,
    value: float,
    accepted_limit: float,
    multiplier: float,
) -> Dict[str, Any]:
    measured = max(float(value), 0.0)
    limit = max(float(accepted_limit), 0.0)
    hard_limit = limit / multiplier if multiplier > 0.0 else limit
    if measured >= limit or limit <= 0.0:
        status = "ok"
    elif measured >= hard_limit:
        status = "advisory"
    else:
        status = "hard_failed"
    severity_ratio = limit / max(measured, 1e-12) if limit > 0.0 else 0.0
    return {
        "status": status,
        "advisory": status == "advisory",
        "hard_failed": status == "hard_failed",
        "value": measured,
        "accepted_limit": limit,
        "hard_limit": hard_limit,
        "severity_ratio": severity_ratio,
        "advisory_multiplier": multiplier,
    }


def stage7_upper_quality_gate(
    cfg,
    *,
    value: float,
    accepted_limit: float,
) -> Dict[str, Any]:
    """Classify an upper-bound metric as ok, advisory, or hard failure."""
    multiplier = stage7_quality_advisory_multiplier(cfg)
    return _upper_quality_gate(
        value=value,
        accepted_limit=accepted_limit,
        multiplier=multiplier,
    )


def stage7_lower_quality_gate(
    cfg,
    *,
    value: float,
    accepted_limit: float,
) -> Dict[str, Any]:
    """Classify a lower-bound metric using the symmetric 1/multiplier floor."""
    multiplier = stage7_quality_advisory_multiplier(cfg)
    return _lower_quality_gate(
        value=value,
        accepted_limit=accepted_limit,
        multiplier=multiplier,
    )


def stage7_9_upper_quality_gate(
    cfg,
    *,
    value: float,
    accepted_limit: float,
) -> Dict[str, Any]:
    """Classify a recoverable Stage 7-9 upper-bound metric."""
    return _upper_quality_gate(
        value=value,
        accepted_limit=accepted_limit,
        multiplier=stage7_9_quality_advisory_multiplier(cfg),
    )


def stage7_9_lower_quality_gate(
    cfg,
    *,
    value: float,
    accepted_limit: float,
) -> Dict[str, Any]:
    """Classify a recoverable Stage 7-9 lower-bound metric."""
    return _lower_quality_gate(
        value=value,
        accepted_limit=accepted_limit,
        multiplier=stage7_9_quality_advisory_multiplier(cfg),
    )


def stage7_has_diffuse_nebula_context(pipeline) -> bool:
    """Return whether image evidence requires diffuse-nebula halo protection.

    The frozen primary target remains the sole processing-policy router, but a
    mixed field such as Horsehead/IC 434 must not let that catalog label turn
    real emission or reflection structure into a star-halo measurement.
    Require multiple image-derived secondary signals before enabling this
    measurement-only protection.
    """
    target_type = (
        str(pipeline._active_target_type() or "").strip().lower()
        if hasattr(pipeline, "_active_target_type")
        else ""
    )
    if target_type in DIFFUSE_EMISSION_NEBULA_TARGET_TYPES:
        return True

    profile = getattr(pipeline, "target_profile", None)
    if not isinstance(profile, dict):
        return False
    raw_labels = profile.get("secondary_labels", [])
    labels = {
        str(item).strip().lower()
        for item in raw_labels
        if str(item).strip()
    } if isinstance(raw_labels, (list, tuple, set)) else set()
    raw_features = profile.get("features", {})
    features = raw_features if isinstance(raw_features, dict) else {}

    def present(name: str) -> bool:
        return name in labels or bool(features.get(name, False))

    broad_structure = present("large_nebulosity") and present("faint_outer_cloud")
    physical_color_context = present("emission_red") or present("reflection_blue")
    return bool(broad_structure and physical_color_context)


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
    component_gates = {
        "dynamic_range_ratio": stage7_lower_quality_gate(
            cfg,
            value=dynamic_range_ratio,
            accepted_limit=dynamic_threshold,
        ),
        "peak_signal": stage7_lower_quality_gate(
            cfg,
            value=peak_signal,
            accepted_limit=peak_threshold,
        ),
        "peak_background_ratio": stage7_lower_quality_gate(
            cfg,
            value=peak_background_ratio,
            accepted_limit=peak_background_ratio_threshold,
        ),
    }
    hard_failed = bool(
        collapsed
        and any(gate["hard_failed"] for gate in component_gates.values())
    )
    advisory = bool(collapsed and not hard_failed)
    return {
        "collapsed": collapsed,
        "advisory": advisory,
        "hard_failed": hard_failed,
        "status": (
            "hard_failed" if hard_failed else "advisory" if advisory else "ok"
        ),
        "dynamic_range_ratio_min": dynamic_threshold,
        "peak_signal_min": peak_threshold,
        "peak_background_ratio": peak_background_ratio,
        "peak_background_ratio_min": peak_background_ratio_threshold,
        "component_gates": component_gates,
        "advisory_multiplier": stage7_quality_advisory_multiplier(cfg),
    }


def _expand_boolean_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    expanded = np.asarray(mask, dtype=bool).copy()
    for _ in range(max(0, int(radius))):
        padded = np.pad(expanded, 1, mode="constant", constant_values=False)
        expanded = np.logical_or.reduce(
            tuple(
                padded[y : y + expanded.shape[0], x : x + expanded.shape[1]]
                for y in range(3)
                for x in range(3)
            )
        )
    return expanded


def stage7_calibrate_starless_dynamic_range(
    source_gray: np.ndarray,
    starless_gray: np.ndarray,
    starmask_gray: Optional[np.ndarray],
) -> Dict[str, Any]:
    """Measure paired non-stellar structure instead of comparing star peaks.

    Removing stars is expected to reduce the full-frame source P99 and maximum.
    A valid starmask lets the gate compare the exact same non-stellar coordinates
    before and after SyQon.  The calibrated value is used only when enough
    support remains and the paired morphology is still strongly correlated;
    otherwise callers retain the original full-frame fail-closed measurement.
    """
    source = np.nan_to_num(
        np.asarray(source_gray, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    starless = np.nan_to_num(
        np.asarray(starless_gray, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    report: Dict[str, Any] = {
        "available": False,
        "method": "full_frame_percentile_fallback",
        "reason": "starmask_unavailable",
        "support_ratio": 0.0,
        "structure_correlation": 0.0,
        "source_range": 0.0,
        "starless_range": 0.0,
        "range_ratio": 0.0,
        "source_range_fraction": 0.0,
        "source_peak_signal": 0.0,
        "starless_peak_signal": 0.0,
        "starmask_scale": 0.0,
        "starmask_threshold": STARLESS_RANGE_STARMASK_THRESHOLD,
        "starmask_quantile": STARLESS_RANGE_STARMASK_QUANTILE,
        "exclusion_radius": STARLESS_RANGE_EXCLUSION_RADIUS,
        "support_min": STARLESS_RANGE_SUPPORT_MIN,
        "correlation_min": STARLESS_RANGE_CORRELATION_MIN,
        "source_range_fraction_min": STARLESS_RANGE_SOURCE_FRACTION_MIN,
    }
    if source.ndim != 2 or source.shape != starless.shape:
        report["reason"] = "source_starless_shape_mismatch"
        return report
    if starmask_gray is None:
        return report
    starmask = np.nan_to_num(
        np.asarray(starmask_gray, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if starmask.shape != source.shape:
        report["reason"] = "starmask_shape_mismatch"
        return report

    try:
        global_q01, global_q99 = np.quantile(source, (0.01, 0.99))
        global_source_range = max(float(global_q99 - global_q01), 1e-7)
        starmask_scale = float(
            np.quantile(np.clip(starmask, 0.0, None), STARLESS_RANGE_STARMASK_QUANTILE)
        )
    except (TypeError, ValueError, FloatingPointError):
        report["reason"] = "percentile_measurement_failed"
        return report
    report["starmask_scale"] = starmask_scale
    if not math.isfinite(starmask_scale) or starmask_scale <= 1e-7:
        report["reason"] = "starmask_signal_insufficient"
        return report

    normalized_starmask = np.clip(starmask / starmask_scale, 0.0, 1.0)
    stellar_seed = normalized_starmask >= STARLESS_RANGE_STARMASK_THRESHOLD
    nonstellar_support = ~_expand_boolean_mask(
        stellar_seed,
        STARLESS_RANGE_EXCLUSION_RADIUS,
    )
    support_count = int(np.count_nonzero(nonstellar_support))
    support_ratio = float(support_count / max(nonstellar_support.size, 1))
    report["support_ratio"] = support_ratio
    minimum_count = max(4096, int(nonstellar_support.size * STARLESS_RANGE_SUPPORT_MIN))
    if support_count < minimum_count:
        report["reason"] = "nonstellar_support_insufficient"
        return report

    source_values = source[nonstellar_support]
    starless_values = starless[nonstellar_support]
    try:
        source_q01, source_q99, source_peak = np.quantile(
            source_values,
            (0.01, 0.99, 0.999),
        )
        starless_q01, starless_q99, starless_peak = np.quantile(
            starless_values,
            (0.01, 0.99, 0.999),
        )
    except (TypeError, ValueError, FloatingPointError):
        report["reason"] = "nonstellar_percentile_measurement_failed"
        return report
    source_range = max(float(source_q99 - source_q01), 1e-7)
    starless_range = max(float(starless_q99 - starless_q01), 0.0)
    range_ratio = _clamp_float(starless_range / source_range, 0.0, 10.0)
    source_range_fraction = source_range / global_source_range
    report.update(
        {
            "source_range": source_range,
            "starless_range": starless_range,
            "range_ratio": range_ratio,
            "source_range_fraction": source_range_fraction,
            "source_peak_signal": float(source_peak),
            "starless_peak_signal": float(starless_peak),
        }
    )
    if source_range_fraction < STARLESS_RANGE_SOURCE_FRACTION_MIN:
        report["reason"] = "nonstellar_source_range_insufficient"
        return report

    source_centered = source_values.astype(np.float64) - float(
        np.mean(source_values, dtype=np.float64)
    )
    starless_centered = starless_values.astype(np.float64) - float(
        np.mean(starless_values, dtype=np.float64)
    )
    denominator = math.sqrt(
        float(np.dot(source_centered, source_centered))
        * float(np.dot(starless_centered, starless_centered))
    )
    if not math.isfinite(denominator) or denominator <= 1e-12:
        report["reason"] = "nonstellar_correlation_unavailable"
        return report
    correlation = _clamp_float(
        float(np.dot(source_centered, starless_centered)) / denominator,
        -1.0,
        1.0,
    )
    report["structure_correlation"] = correlation
    if correlation < STARLESS_RANGE_CORRELATION_MIN:
        report["reason"] = "nonstellar_structure_correlation_low"
        return report

    report.update(
        {
            "available": True,
            "method": "starmask_excluded_paired_percentiles",
            "reason": "accepted",
        }
    )
    return report


def stage7_galaxy_structure_masks(
    source_gray: np.ndarray,
    source_bg: float,
    source_std: float,
) -> Dict[str, Any]:
    """Locate a conservative elliptical galaxy disk/core from low frequencies."""
    gray = np.nan_to_num(
        np.asarray(source_gray).astype(np.float32, copy=False),
        nan=float(source_bg),
        posinf=float(source_bg),
        neginf=float(source_bg),
    )
    if gray.ndim != 2 or min(gray.shape) < 24:
        return {"available": False, "reason": "image_too_small"}

    broad = gray.copy()
    for _ in range(24):
        broad = _box_blur_gray(broad)
    signal = np.clip(broad - float(source_bg), 0.0, None)
    height, width = gray.shape
    peak_y, peak_x = np.unravel_index(int(np.argmax(broad)), broad.shape)
    yy, xx = np.mgrid[:height, :width]
    search_radius = max(12.0, float(min(height, width)) * 0.42)
    search_mask = (
        np.square(xx.astype(np.float32) - float(peak_x))
        + np.square(yy.astype(np.float32) - float(peak_y))
    ) <= search_radius * search_radius
    search_values = signal[search_mask]
    floor = max(
        float(np.quantile(search_values, 0.70)),
        max(float(source_std), 1e-5) * 1.5,
        0.0025,
    )
    weights = np.clip(signal - floor, 0.0, None)
    weights[~search_mask] = 0.0
    edge = max(2, min(height, width) // 50)
    weights[:edge, :] = 0.0
    weights[-edge:, :] = 0.0
    weights[:, :edge] = 0.0
    weights[:, -edge:] = 0.0
    positive = weights[weights > 0.0]
    if positive.size < max(96, int(gray.size * 0.005)):
        return {"available": False, "reason": "insufficient_broad_signal"}
    cap = float(np.quantile(positive, 0.92))
    if math.isfinite(cap) and cap > 0.0:
        weights = np.minimum(weights, cap)

    total = float(np.sum(weights))
    if not math.isfinite(total) or total <= 1e-6:
        return {"available": False, "reason": "invalid_broad_signal"}
    center_x = float(np.sum(weights * xx) / total)
    center_y = float(np.sum(weights * yy) / total)
    dx = xx.astype(np.float32) - center_x
    dy = yy.astype(np.float32) - center_y
    covariance = np.array(
        [
            [
                float(np.sum(weights * dx * dx) / total),
                float(np.sum(weights * dx * dy) / total),
            ],
            [
                float(np.sum(weights * dx * dy) / total),
                float(np.sum(weights * dy * dy) / total),
            ],
        ],
        dtype=np.float64,
    )
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    except np.linalg.LinAlgError:
        return {"available": False, "reason": "invalid_shape_covariance"}
    if not np.all(np.isfinite(eigenvalues)) or float(eigenvalues[-1]) <= 1.0:
        return {"available": False, "reason": "degenerate_shape_covariance"}

    minor_sigma = math.sqrt(max(float(eigenvalues[0]), 1.0))
    major_sigma = math.sqrt(max(float(eigenvalues[1]), 1.0))
    min_side = float(min(height, width))
    max_side = float(max(height, width))
    major_radius = _clamp_float(
        major_sigma * 2.5,
        max(8.0, min_side * 0.06),
        max_side * 0.46,
    )
    minor_radius = _clamp_float(
        minor_sigma * 2.5,
        max(6.0, min_side * 0.04),
        min_side * 0.38,
    )
    major_vector = eigenvectors[:, 1]
    minor_vector = eigenvectors[:, 0]
    major_coord = dx * float(major_vector[0]) + dy * float(major_vector[1])
    minor_coord = dx * float(minor_vector[0]) + dy * float(minor_vector[1])
    disk_radius2 = (
        np.square(major_coord / max(major_radius, 1.0))
        + np.square(minor_coord / max(minor_radius, 1.0))
    )
    disk_mask = disk_radius2 <= 1.0
    disk_coverage = float(np.mean(disk_mask))
    if (
        bool(np.any(disk_mask[0, :]))
        or bool(np.any(disk_mask[-1, :]))
        or bool(np.any(disk_mask[:, 0]))
        or bool(np.any(disk_mask[:, -1]))
    ):
        return {
            "available": False,
            "reason": "truncated_disk_roi",
            "disk_coverage": disk_coverage,
        }
    if not 0.003 <= disk_coverage <= 0.72:
        return {
            "available": False,
            "reason": "implausible_disk_coverage",
            "disk_coverage": disk_coverage,
        }

    core_major = _clamp_float(
        major_radius * 0.18,
        3.0,
        min_side * 0.09,
    )
    core_minor = _clamp_float(
        minor_radius * 0.24,
        3.0,
        min_side * 0.07,
    )
    core_radius2 = (
        np.square(major_coord / max(core_major, 1.0))
        + np.square(minor_coord / max(core_minor, 1.0))
    )
    core_mask = core_radius2 <= 1.0
    core_ring_mask = (core_radius2 > 1.0) & (core_radius2 <= 4.0) & disk_mask
    if int(np.count_nonzero(core_ring_mask)) < 16:
        core_ring_mask = disk_mask & (~core_mask)

    return {
        "available": True,
        "reason": "",
        "disk_mask": disk_mask,
        "core_mask": core_mask,
        "core_ring_mask": core_ring_mask,
        "broad_source": broad,
        "center_x": center_x,
        "center_y": center_y,
        "major_radius": float(major_radius),
        "minor_radius": float(minor_radius),
        "disk_coverage": disk_coverage,
        "core_coverage": float(np.mean(core_mask)),
    }


def _stage7_galaxy_artifact_scores(
    source_gray: np.ndarray,
    starless_gray: np.ndarray,
    starmask_gray: Optional[np.ndarray],
    *,
    source_bg: float,
    source_std: float,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Compare galaxy disk/core under one shared local range stretch."""
    metrics: Dict[str, float] = {
        "galaxy_roi_available": 0.0,
        "galaxy_disk_halo_evidence_available": 0.0,
        "galaxy_disk_halo_residue_score": 0.0,
        "galaxy_disk_halo_mask_coverage": 0.0,
        "galaxy_disk_star_seed_coverage": 0.0,
        "galaxy_core_preservation_ratio": 1.0,
        "galaxy_core_contrast_ratio": 1.0,
        "galaxy_core_damage_score": 0.0,
        "galaxy_structure_starmask_leakage": 0.0,
        "galaxy_range_black": 0.0,
        "galaxy_range_white": 0.0,
        "galaxy_roi_center_x": 0.0,
        "galaxy_roi_center_y": 0.0,
        "galaxy_roi_major_radius": 0.0,
        "galaxy_roi_minor_radius": 0.0,
        "galaxy_roi_disk_coverage": 0.0,
        "galaxy_roi_core_coverage": 0.0,
    }
    masks = stage7_galaxy_structure_masks(source_gray, source_bg, source_std)
    if not bool(masks.get("available", False)):
        return metrics, masks

    disk_mask = np.asarray(masks["disk_mask"], dtype=bool)
    core_mask = np.asarray(masks["core_mask"], dtype=bool)
    core_ring_mask = np.asarray(masks["core_ring_mask"], dtype=bool)
    disk_range_mask = disk_mask & (~core_mask)
    if int(np.count_nonzero(disk_range_mask)) < 32:
        disk_range_mask = disk_mask
    source_values = source_gray[disk_range_mask]
    black = float(np.quantile(source_values, 0.08))
    white = float(np.quantile(source_values, 0.975))
    minimum_range = max(4.0 * max(float(source_std), 1e-5), 0.008)
    if not math.isfinite(black) or not math.isfinite(white) or white - black < minimum_range:
        masks = dict(masks)
        masks["available"] = False
        masks["reason"] = "insufficient_disk_range"
        return metrics, masks

    denominator = max(white - black, 1e-6)

    def shared_stretch(gray: np.ndarray) -> np.ndarray:
        normalized = np.clip((gray - black) / denominator, 0.0, 1.0)
        return (
            np.arcsinh(8.0 * normalized) / math.asinh(8.0)
        ).astype(np.float32)

    source_stretched = shared_stretch(source_gray)
    starless_stretched = shared_stretch(starless_gray)
    # The disk range intentionally saturates a bright nucleus so faint disk
    # halos become visible. Assess the nucleus separately in the original
    # linear domain, with matched low-frequency smoothing, otherwise a removed
    # M31 core can look numerically identical after clipping to white.
    source_broad = np.asarray(masks["broad_source"], dtype=np.float32)
    starless_broad = starless_gray.astype(np.float32, copy=True)
    for _ in range(24):
        starless_broad = _box_blur_gray(starless_broad)
    outside_disk = ~disk_mask
    starless_bg = (
        float(np.median(starless_broad[outside_disk]))
        if int(np.count_nonzero(outside_disk)) >= 32
        else float(np.median(starless_broad))
    )
    source_core = max(
        float(np.median(source_broad[core_mask])) - float(source_bg),
        0.0,
    )
    starless_core = max(
        float(np.median(starless_broad[core_mask])) - starless_bg,
        0.0,
    )
    source_ring = max(
        float(np.median(source_broad[core_ring_mask])) - float(source_bg),
        0.0,
    )
    starless_ring = max(
        float(np.median(starless_broad[core_ring_mask])) - starless_bg,
        0.0,
    )
    core_preservation = _clamp_float(
        starless_core / max(source_core, 1e-5),
        0.0,
        3.0,
    )
    source_contrast = max(source_core - source_ring, 1e-5)
    starless_contrast = max(starless_core - starless_ring, 0.0)
    core_contrast = _clamp_float(
        starless_contrast / source_contrast,
        0.0,
        3.0,
    )

    source_local = source_stretched.copy()
    for _ in range(4):
        source_local = _box_blur_gray(source_local)
    source_detail = np.clip(source_stretched - source_local, 0.0, None)
    detail_values = source_detail[disk_range_mask]
    detail_median = float(np.median(detail_values))
    detail_mad = float(np.median(np.abs(detail_values - detail_median)))
    detail_threshold = max(
        float(np.quantile(detail_values, 0.995)),
        detail_median + 6.0 * max(detail_mad, 1e-5),
        0.012,
    )

    def local_peak_mask(values: np.ndarray) -> np.ndarray:
        padded = np.pad(values, ((1, 1), (1, 1)), mode="reflect")
        neighbors = []
        for offset_y in range(3):
            for offset_x in range(3):
                if offset_y == 1 and offset_x == 1:
                    continue
                neighbors.append(
                    padded[
                        offset_y : offset_y + values.shape[0],
                        offset_x : offset_x + values.shape[1],
                    ]
                )
        neighbor_max = np.maximum.reduce(neighbors)
        return values >= neighbor_max

    source_peak = local_peak_mask(source_detail)
    source_peak_candidate = (
        source_peak
        & (source_detail > detail_threshold)
        & disk_range_mask
    )
    star_seed = source_peak_candidate.copy()

    normalized_starmask: Optional[np.ndarray] = None
    if starmask_gray is not None and starmask_gray.shape == source_gray.shape:
        mask_scale = float(np.quantile(starmask_gray, 0.997))
        if not math.isfinite(mask_scale) or mask_scale <= 1e-7:
            mask_scale = float(np.max(starmask_gray)) if starmask_gray.size else 0.0
        if mask_scale > 1e-7:
            normalized_starmask = np.clip(starmask_gray / mask_scale, 0.0, 1.0)
            mask_local = normalized_starmask.copy()
            for _ in range(3):
                mask_local = _box_blur_gray(mask_local)
            mask_detail = np.clip(normalized_starmask - mask_local, 0.0, None)
            mask_values = mask_detail[disk_range_mask]
            mask_threshold = max(float(np.quantile(mask_values, 0.992)), 0.06)
            mask_peak = local_peak_mask(mask_detail)
            very_strong_source = source_detail > max(
                float(np.quantile(detail_values, 0.9985)),
                detail_threshold * 1.35,
            )
            star_seed = source_peak_candidate & (
                ((mask_peak & (mask_detail > mask_threshold)))
                | very_strong_source
            )

    seed_coverage = float(np.mean(star_seed & disk_mask))
    compact_support = star_seed.astype(np.float32)
    for _ in range(2):
        compact_support = np.maximum(
            compact_support,
            np.clip(_box_blur_gray(compact_support) * 2.0, 0.0, 1.0),
        )
    halo_weight = compact_support.copy()
    for _ in range(5):
        expanded = np.clip(_box_blur_gray(halo_weight) * 2.1, 0.0, 1.0)
        halo_weight = np.maximum(halo_weight, expanded)
    gradient_y, gradient_x = np.gradient(source_local)
    structure_gradient = np.hypot(gradient_x, gradient_y)
    gradient_limit = float(np.quantile(structure_gradient[disk_range_mask], 0.88))
    halo_mask = (
        (halo_weight > 0.018)
        & (compact_support < 0.10)
        & disk_range_mask
        & (structure_gradient <= max(gradient_limit, 0.002))
    )
    halo_count = int(np.count_nonzero(halo_mask))
    if int(np.count_nonzero(star_seed)) >= 3 and halo_count > 16:
        # Use one common local baseline for original and starless. Separate
        # local blurs let the original star lift its own baseline and can make
        # an unchanged spiral arm look stronger in starless than in original.
        source_halo_local = source_stretched.copy()
        starless_halo_local = starless_stretched.copy()
        for _ in range(12):
            source_halo_local = _box_blur_gray(source_halo_local)
            starless_halo_local = _box_blur_gray(starless_halo_local)
        common_local = np.minimum(source_halo_local, starless_halo_local)
        source_halo_signal = np.clip(
            source_stretched - common_local,
            0.0,
            None,
        )
        starless_halo_signal = np.clip(
            starless_stretched - common_local,
            0.0,
            None,
        )
        seed_points = np.argwhere(star_seed)
        seed_points = sorted(
            seed_points,
            key=lambda point: float(source_detail[int(point[0]), int(point[1])]),
            reverse=True,
        )
        selected_points: List[Tuple[int, int]] = []
        for point in seed_points:
            point_y, point_x = int(point[0]), int(point[1])
            if any(
                (point_y - prior_y) ** 2 + (point_x - prior_x) ** 2 < 25
                for prior_y, prior_x in selected_points
            ):
                continue
            selected_points.append((point_y, point_x))
            if len(selected_points) >= 128:
                break

        local_scores: List[float] = []
        for point_y, point_x in selected_points:
            y0 = max(0, point_y - 9)
            y1 = min(source_gray.shape[0], point_y + 10)
            x0 = max(0, point_x - 9)
            x1 = min(source_gray.shape[1], point_x + 10)
            local_mask = halo_mask[y0:y1, x0:x1]
            if int(np.count_nonzero(local_mask)) < 4:
                continue
            local_source = source_halo_signal[y0:y1, x0:x1][local_mask]
            local_starless = starless_halo_signal[y0:y1, x0:x1][local_mask]
            local_source_level = float(np.quantile(local_source, 0.75))
            if local_source_level <= 1e-5:
                continue
            local_starless_level = float(np.quantile(local_starless, 0.75))
            local_scores.append(
                _clamp_float(
                    local_starless_level / local_source_level,
                    0.0,
                    3.0,
                )
            )
        if local_scores:
            metrics["galaxy_disk_halo_evidence_available"] = 1.0
            # One bright disk star can reveal a damaging halo even when many
            # correctly removed stars dilute a global mean. Compact evidence
            # has already been shape/gradient filtered, so retain the worst
            # local ring rather than averaging it away.
            metrics["galaxy_disk_halo_residue_score"] = float(
                np.max(local_scores)
            )
            metrics["galaxy_disk_halo_mask_coverage"] = float(np.mean(halo_mask))

    if normalized_starmask is not None:
        structure_only = disk_mask & (compact_support < 0.08)
        total_signal = float(np.sum(normalized_starmask))
        if total_signal > 1e-7 and int(np.count_nonzero(structure_only)) > 16:
            metrics["galaxy_structure_starmask_leakage"] = _clamp_float(
                float(np.sum(normalized_starmask[structure_only])) / total_signal,
                0.0,
                1.0,
            )

    metrics.update(
        {
            "galaxy_roi_available": 1.0,
            "galaxy_disk_star_seed_coverage": seed_coverage,
            "galaxy_core_preservation_ratio": core_preservation,
            "galaxy_core_contrast_ratio": core_contrast,
            "galaxy_core_damage_score": max(
                0.0,
                1.0 - min(core_preservation, core_contrast),
            ),
            "galaxy_range_black": black,
            "galaxy_range_white": white,
            "galaxy_roi_center_x": float(masks["center_x"]) / max(source_gray.shape[1], 1),
            "galaxy_roi_center_y": float(masks["center_y"]) / max(source_gray.shape[0], 1),
            "galaxy_roi_major_radius": float(masks["major_radius"]) / max(source_gray.shape),
            "galaxy_roi_minor_radius": float(masks["minor_radius"]) / max(source_gray.shape),
            "galaxy_roi_disk_coverage": float(masks["disk_coverage"]),
            "galaxy_roi_core_coverage": float(masks["core_coverage"]),
        }
    )
    return metrics, masks


def stage7_starless_artifact_scores(
    pipeline,
    source_data: Optional[np.ndarray],
    starless_data: Optional[np.ndarray],
    starmask_data: Optional[np.ndarray],
    source_features: ImageFeatures,
    starless_features: ImageFeatures,
) -> Dict[str, Any]:
    scores = {
        "halo_residue_score": 0.0,
        "global_halo_residue_score": 0.0,
        "compact_halo_residue_score": 0.0,
        "compact_halo_mask_coverage": 0.0,
        "compact_halo_source_level": 0.0,
        "compact_halo_starless_level": 0.0,
        "compact_halo_evidence_available": 0.0,
        "diffuse_nebula_context": 0.0,
        "diffuse_nebula_protection_coverage": 0.0,
        "black_hole_score": 0.0,
        "compact_residual_star_score": 0.0,
        "compact_residual_coverage": 0.0,
        "starmask_contamination": 0.0,
        "starless_noise_gain": 1.0,
        "starless_dynamic_range_ratio": 1.0,
        "starless_dynamic_range_ratio_raw": 1.0,
        "source_dynamic_range": 0.0,
        "starless_dynamic_range": 0.0,
        "source_dynamic_range_raw": 0.0,
        "starless_dynamic_range_raw": 0.0,
        "source_peak_signal": 0.0,
        "starless_peak_signal": 0.0,
        "dynamic_range_calibration_available": 0.0,
        "dynamic_range_calibration_support_ratio": 0.0,
        "dynamic_range_calibration_correlation": 0.0,
        "dynamic_range_calibration_source_fraction": 0.0,
        "dynamic_range_calibration_source_peak_signal": 0.0,
        "dynamic_range_calibration_starless_peak_signal": 0.0,
        "galaxy_roi_available": 0.0,
        "galaxy_disk_halo_evidence_available": 0.0,
        "galaxy_disk_halo_residue_score": 0.0,
        "galaxy_disk_halo_mask_coverage": 0.0,
        "galaxy_disk_star_seed_coverage": 0.0,
        "galaxy_core_preservation_ratio": 1.0,
        "galaxy_core_contrast_ratio": 1.0,
        "galaxy_core_damage_score": 0.0,
        "galaxy_structure_starmask_leakage": 0.0,
        "galaxy_range_black": 0.0,
        "galaxy_range_white": 0.0,
        "galaxy_roi_center_x": 0.0,
        "galaxy_roi_center_y": 0.0,
        "galaxy_roi_major_radius": 0.0,
        "galaxy_roi_minor_radius": 0.0,
        "galaxy_roi_disk_coverage": 0.0,
        "galaxy_roi_core_coverage": 0.0,
    }
    if source_data is None or starless_data is None:
        return scores

    try:
        target_type = (
            str(pipeline._active_target_type() or "").strip().lower()
            if hasattr(pipeline, "_active_target_type")
            else ""
        )
        is_protected_nebula = stage7_has_diffuse_nebula_context(pipeline)
        scores["diffuse_nebula_context"] = float(is_protected_nebula)
        is_galaxy = bool(
            target_type in GALAXY_TARGET_TYPES
            and getattr(
                pipeline.cfg,
                "stage7_galaxy_roi_halo_gate_enabled",
                True,
            )
        )
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

        starmask_gray: Optional[np.ndarray] = None
        if starmask_data is not None:
            starmask_rgb = _to_rgb_float_image(starmask_data, max_side=1024)
            candidate_starmask_gray = (
                0.2126 * starmask_rgb[0]
                + 0.7152 * starmask_rgb[1]
                + 0.0722 * starmask_rgb[2]
            ).astype(np.float32)
            if candidate_starmask_gray.shape == source_gray.shape:
                starmask_gray = candidate_starmask_gray

        source_bg = float(source_features.bg_median)
        source_std = max(float(source_features.bg_std), 1e-5)
        starless_bg = float(starless_features.bg_median)
        starless_std = max(float(starless_features.bg_std), 1e-5)
        try:
            source_q01, source_q99 = np.quantile(source_gray, (0.01, 0.99))
            starless_q01, starless_q99 = np.quantile(starless_gray, (0.01, 0.99))
            source_range = max(float(source_q99 - source_q01), 1e-7)
            starless_range = max(float(starless_q99 - starless_q01), 0.0)
            raw_range_ratio = _clamp_float(
                starless_range / source_range,
                0.0,
                10.0,
            )
            scores["source_dynamic_range"] = source_range
            scores["starless_dynamic_range"] = starless_range
            scores["starless_dynamic_range_ratio"] = raw_range_ratio
            scores["source_dynamic_range_raw"] = source_range
            scores["starless_dynamic_range_raw"] = starless_range
            scores["starless_dynamic_range_ratio_raw"] = raw_range_ratio
            scores["source_peak_signal"] = float(np.nanmax(source_gray))
            scores["starless_peak_signal"] = float(np.nanmax(starless_gray))
        except (TypeError, ValueError, FloatingPointError):
            pass
        dynamic_calibration = stage7_calibrate_starless_dynamic_range(
            source_gray,
            starless_gray,
            starmask_gray,
        )
        scores["dynamic_range_calibration_available"] = float(
            bool(dynamic_calibration.get("available", False))
        )
        scores["dynamic_range_calibration_support_ratio"] = float(
            dynamic_calibration.get("support_ratio", 0.0) or 0.0
        )
        scores["dynamic_range_calibration_correlation"] = float(
            dynamic_calibration.get("structure_correlation", 0.0) or 0.0
        )
        scores["dynamic_range_calibration_source_fraction"] = float(
            dynamic_calibration.get("source_range_fraction", 0.0) or 0.0
        )
        scores["dynamic_range_calibration_source_peak_signal"] = float(
            dynamic_calibration.get("source_peak_signal", 0.0) or 0.0
        )
        scores["dynamic_range_calibration_starless_peak_signal"] = float(
            dynamic_calibration.get("starless_peak_signal", 0.0) or 0.0
        )
        scores["dynamic_range_calibration_method"] = str(
            dynamic_calibration.get("method") or "full_frame_percentile_fallback"
        )
        scores["dynamic_range_calibration_reason"] = str(
            dynamic_calibration.get("reason") or "unknown"
        )
        if bool(dynamic_calibration.get("available", False)):
            scores["source_dynamic_range"] = float(
                dynamic_calibration["source_range"]
            )
            scores["starless_dynamic_range"] = float(
                dynamic_calibration["starless_range"]
            )
            scores["starless_dynamic_range_ratio"] = float(
                dynamic_calibration["range_ratio"]
            )
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
        if (
            is_protected_nebula
            and starmask_gray is not None
            and starmask_gray.shape == source_gray.shape
        ):
            # Broad smoothing also makes saturated stars look like diffuse
            # nebulosity. Carve confirmed point-source neighborhoods back out
            # of the protection mask so their wings remain measurable.
            starmask_scale = float(np.quantile(starmask_gray, 0.997))
            if not math.isfinite(starmask_scale) or starmask_scale <= 1e-7:
                starmask_scale = (
                    float(np.max(starmask_gray)) if starmask_gray.size else 0.0
                )
            if starmask_scale > 1e-7:
                normalized_stars = np.clip(
                    starmask_gray / starmask_scale,
                    0.0,
                    1.0,
                )
                star_neighborhood_weight = (
                    normalized_stars > 0.08
                ).astype(np.float32)
                for _ in range(5):
                    star_neighborhood_weight = np.maximum(
                        star_neighborhood_weight,
                        np.clip(
                            _box_blur_gray(star_neighborhood_weight) * 2.0,
                            0.0,
                            1.0,
                        ),
                    )
                diffuse_nebula_protect &= star_neighborhood_weight <= 0.010
        scores["diffuse_nebula_protection_coverage"] = float(
            np.mean(diffuse_nebula_protect)
        )
        scores["starless_noise_gain"] = _clamp_float(
            starless_std / max(source_std, 1e-5),
            0.0,
            10.0,
        )

        galaxy_structure_mask = np.zeros_like(source_gray, dtype=bool)
        if is_galaxy:
            galaxy_scores, galaxy_masks = _stage7_galaxy_artifact_scores(
                source_gray,
                starless_gray,
                starmask_gray,
                source_bg=source_bg,
                source_std=source_std,
            )
            scores.update(galaxy_scores)
            if bool(galaxy_masks.get("available", False)):
                galaxy_structure_mask = np.asarray(
                    galaxy_masks["disk_mask"],
                    dtype=bool,
                )

        core_threshold = max(
            float(np.quantile(source_gray, 0.995)),
            source_bg + max(6.0 * source_std, 0.08),
        )
        # A bright galaxy nucleus is not a star. The fitted disk receives its
        # own same-stretch ROI check below, so the generic corner/field halo
        # diagnostic must not form a star ring around the bulge.
        core_mask = (source_gray > core_threshold) & (~galaxy_structure_mask)
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

        if starmask_gray is not None:
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
                protected_structure_mask = (
                    protected_nebula_mask | galaxy_structure_mask
                )
                mask_signal = np.clip(starmask_gray, 0.0, None)
                total_mask_signal = float(np.sum(mask_signal))
                generic_contamination_mask = nebula_mask & (~galaxy_structure_mask)
                if (
                    total_mask_signal > 1e-7
                    and int(np.count_nonzero(generic_contamination_mask)) > 16
                ):
                    contaminated_signal = float(
                        np.sum(mask_signal[generic_contamination_mask])
                    )
                    scores["starmask_contamination"] = _clamp_float(
                        contaminated_signal / total_mask_signal,
                        0.0,
                        1.0,
                    )
                if scores["galaxy_roi_available"] > 0.5:
                    scores["starmask_contamination"] = max(
                        scores["starmask_contamination"] * 0.50,
                        scores["galaxy_structure_starmask_leakage"],
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
                    compact_weight
                    * (1.0 - 0.85 * protected_nebula_mask.astype(np.float32))
                    * (~galaxy_structure_mask).astype(np.float32),
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
                    & (~protected_structure_mask)
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
                        scores["compact_halo_evidence_available"] = 1.0
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
    compact_halo_evidence_available = (
        artifact_scores.get("compact_halo_evidence_available", 0.0) > 0.5
    )
    diffuse_nebula_context = (
        artifact_scores.get("diffuse_nebula_context", 0.0) > 0.5
    )
    halo_residue_score = artifact_scores["halo_residue_score"]
    galaxy_roi_available = artifact_scores.get("galaxy_roi_available", 0.0) > 0.5
    galaxy_disk_halo_residue_score = artifact_scores.get(
        "galaxy_disk_halo_residue_score",
        0.0,
    )
    galaxy_disk_halo_evidence_available = (
        artifact_scores.get("galaxy_disk_halo_evidence_available", 0.0) > 0.5
    )
    galaxy_core_preservation_ratio = artifact_scores.get(
        "galaxy_core_preservation_ratio",
        1.0,
    )
    galaxy_core_contrast_ratio = artifact_scores.get(
        "galaxy_core_contrast_ratio",
        1.0,
    )
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
    target_type = (
        str(pipeline._active_target_type() or "").strip().lower()
        if hasattr(pipeline, "_active_target_type")
        else ""
    )
    bright_core_integrity = assess_bright_core_integrity(
        source_data,
        starless_data,
        starmask_data,
        target_type=target_type,
        target_profile=getattr(pipeline, "target_profile", None),
    )
    if target_type in GALAXY_TARGET_TYPES and galaxy_roi_available:
        # Global star-density metrics see the bulge, arms and dust lanes as
        # point-like energy. Use only compact evidence outside the fitted disk;
        # the disk itself is evaluated by its dedicated same-stretch ROI gate.
        residual_star_score = min(
            global_residual_star_score,
            max(
                compact_residual_star_score,
                compact_residual_coverage * 0.70,
            ),
        )
    elif compact_residual_star_score > 0.0 and target_type == "bright_emission_reflection_nebula":
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
    if (
        compact_halo_evidence_available
        and compact_halo_residue_score > 0.0
        and (
            target_type == "bright_emission_reflection_nebula"
            or diffuse_nebula_context
        )
    ):
        # In mixed diffuse-nebula fields the global bright-core annulus can
        # overlap real IC 434/NGC 2023/NGC 2024 structure. The compact metric
        # is constrained by the cleaned starmask and removes a matched local
        # low-frequency baseline, so it owns the artifact decision here.
        halo_residue_score = min(global_halo_residue_score, compact_halo_residue_score)
    elif target_type in GALAXY_TARGET_TYPES and galaxy_roi_available:
        halo_residue_score = max(
            global_halo_residue_score,
            galaxy_disk_halo_residue_score
            if galaxy_disk_halo_evidence_available
            else 0.0,
        )

    issues: List[str] = []
    advisories: List[str] = []
    quality_gates: Dict[str, Dict[str, Any]] = {}

    def record_gate(
        code: str,
        gate: Dict[str, Any],
        message: str,
    ) -> None:
        quality_gates[code] = gate
        if bool(gate.get("hard_failed", False)):
            issues.append(message)
        elif bool(gate.get("advisory", False)):
            advisories.append(message)

    if bool(bright_core_integrity.get("applicable", False)):
        bright_core_status = str(
            bright_core_integrity.get("status") or "hard_failed"
        )
        bright_core_gate = {
            "status": bright_core_status,
            "advisory": bright_core_status == "advisory",
            "hard_failed": bright_core_status == "hard_failed",
            "fixed_limit": True,
            "trigger_reasons": list(
                bright_core_integrity.get("trigger_reasons") or []
            ),
        }
        quality_gates["bright_core_integrity"] = bright_core_gate
        for metric_name, metric_gate in (
            bright_core_integrity.get("gates") or {}
        ).items():
            quality_gates[f"bright_core_{metric_name}"] = dict(metric_gate)
        bright_core_message = "bright_core_integrity " + ",".join(
            bright_core_gate["trigger_reasons"] or [bright_core_status]
        )
        if bright_core_gate["hard_failed"]:
            issues.append(bright_core_message)
        elif bright_core_gate["advisory"]:
            advisories.append(bright_core_message)

    residual_gate = stage7_upper_quality_gate(
        pipeline.cfg,
        value=residual_star_score,
        accepted_limit=pipeline.cfg.stage7_residual_star_score_max,
    )
    record_gate(
        "residual_stars",
        residual_gate,
        "residual_stars "
        f"{residual_star_score:.3f}>{pipeline.cfg.stage7_residual_star_score_max:.3f}",
    )
    halo_threshold = pipeline._stage7_effective_halo_threshold()
    halo_gate = stage7_upper_quality_gate(
        pipeline.cfg,
        value=halo_residue_score,
        accepted_limit=halo_threshold,
    )
    if halo_gate["status"] != "ok":
        if (
            target_type in GALAXY_TARGET_TYPES
            and galaxy_disk_halo_evidence_available
            and galaxy_disk_halo_residue_score >= global_halo_residue_score
        ):
            record_gate(
                "galaxy_disk_halo_residue",
                halo_gate,
                "galaxy_disk_halo_residue "
                f"{galaxy_disk_halo_residue_score:.3f}>{halo_threshold:.3f}",
            )
        else:
            record_gate(
                "halo_residue",
                halo_gate,
                "halo_residue "
                f"{halo_residue_score:.3f}>{halo_threshold:.3f}",
            )
    else:
        quality_gates["halo_residue"] = halo_gate
    compact_halo_gate = stage7_upper_quality_gate(
        pipeline.cfg,
        value=compact_halo_residue_score,
        accepted_limit=halo_threshold,
    )
    record_gate(
        "compact_halo_residue",
        compact_halo_gate,
        (
            "compact_halo_residue "
            f"{compact_halo_residue_score:.3f}>{halo_threshold:.3f}"
        ),
    )
    black_hole_gate = stage7_upper_quality_gate(
        pipeline.cfg,
        value=black_hole_score,
        accepted_limit=pipeline.cfg.stage7_black_hole_score_max,
    )
    record_gate(
        "black_hole",
        black_hole_gate,
        (
            "black_hole "
            f"{black_hole_score:.3f}>{pipeline.cfg.stage7_black_hole_score_max:.3f}"
        ),
    )
    if target_type in GALAXY_TARGET_TYPES and galaxy_roi_available:
        core_preservation_min = float(
            getattr(
                pipeline.cfg,
                "stage7_galaxy_core_preservation_ratio_min",
                0.72,
            )
        )
        core_contrast_min = float(
            getattr(
                pipeline.cfg,
                "stage7_galaxy_core_contrast_ratio_min",
                0.60,
            )
        )
        core_preservation_gate = stage7_lower_quality_gate(
            pipeline.cfg,
            value=galaxy_core_preservation_ratio,
            accepted_limit=core_preservation_min,
        )
        record_gate(
            "galaxy_core_preservation",
            core_preservation_gate,
            (
                "galaxy_core_preservation "
                f"{galaxy_core_preservation_ratio:.3f}<"
                f"{core_preservation_min:.3f}"
            ),
        )
        core_contrast_gate = stage7_lower_quality_gate(
            pipeline.cfg,
            value=galaxy_core_contrast_ratio,
            accepted_limit=core_contrast_min,
        )
        record_gate(
            "galaxy_core_contrast",
            core_contrast_gate,
            (
                "galaxy_core_contrast "
                f"{galaxy_core_contrast_ratio:.3f}<"
                f"{core_contrast_min:.3f}"
            ),
        )
    if (
        starmask_data is not None
    ):
        contamination_gate = stage7_upper_quality_gate(
            pipeline.cfg,
            value=starmask_contamination,
            accepted_limit=pipeline.cfg.stage7_starmask_contamination_max,
        )
        record_gate(
            "starmask_contamination",
            contamination_gate,
            (
            "starmask_contamination "
            f"{starmask_contamination:.3f}>{pipeline.cfg.stage7_starmask_contamination_max:.3f}"
            ),
        )
    noise_gain_gate = stage7_upper_quality_gate(
        pipeline.cfg,
        value=starless_noise_gain,
        accepted_limit=pipeline.cfg.stage7_starless_noise_gain_max,
    )
    record_gate(
        "starless_noise_gain",
        noise_gain_gate,
        (
            "starless_noise_gain "
            f"{starless_noise_gain:.3f}>{pipeline.cfg.stage7_starless_noise_gain_max:.3f}"
        ),
    )
    dynamic_assessment = stage7_dynamic_range_assessment(
        pipeline.cfg,
        dynamic_range_ratio=starless_dynamic_range_ratio,
        peak_signal=starless_peak_signal,
        background_level=float(starless_features.bg_median),
    )
    dynamic_assessment["measurement"] = {
        "method": artifact_scores.get(
            "dynamic_range_calibration_method",
            "full_frame_percentile_fallback",
        ),
        "reason": artifact_scores.get(
            "dynamic_range_calibration_reason",
            "unavailable",
        ),
        "available": bool(
            artifact_scores.get("dynamic_range_calibration_available", 0.0)
        ),
        "support_ratio": artifact_scores.get(
            "dynamic_range_calibration_support_ratio",
            0.0,
        ),
        "structure_correlation": artifact_scores.get(
            "dynamic_range_calibration_correlation",
            0.0,
        ),
        "source_range_fraction": artifact_scores.get(
            "dynamic_range_calibration_source_fraction",
            0.0,
        ),
        "raw_range_ratio": artifact_scores.get(
            "starless_dynamic_range_ratio_raw",
            starless_dynamic_range_ratio,
        ),
    }
    quality_gates["starless_dynamic_range_collapse"] = dynamic_assessment
    if dynamic_assessment["collapsed"]:
        dynamic_message = (
            "starless_dynamic_range_collapse "
            f"{starless_dynamic_range_ratio:.3f}<"
            f"{dynamic_assessment['dynamic_range_ratio_min']:.3f}, "
            f"peak={starless_peak_signal:.5f}<"
            f"{dynamic_assessment['peak_signal_min']:.5f}, "
            f"peak/bg={dynamic_assessment['peak_background_ratio']:.2f}<"
            f"{dynamic_assessment['peak_background_ratio_min']:.2f}"
        )
        if dynamic_assessment["hard_failed"]:
            issues.append(dynamic_message)
        else:
            advisories.append(dynamic_message)
    if starmask_data is None:
        issues.append("starmask_missing")
    else:
        coverage_gate = stage7_lower_quality_gate(
            pipeline.cfg,
            value=starmask_coverage_ratio,
            accepted_limit=pipeline.cfg.stage7_starmask_coverage_min_ratio,
        )
        record_gate(
            "starmask_coverage_ratio",
            coverage_gate,
            (
            "starmask_coverage_ratio "
            f"{starmask_coverage_ratio:.3f}<{pipeline.cfg.stage7_starmask_coverage_min_ratio:.3f}"
            ),
        )
    if (
        starmask_data is not None
        and source_metrics.median_star_size > 0
        and starmask_metrics.median_star_size > 0
    ):
        width_gate = stage7_upper_quality_gate(
            pipeline.cfg,
            value=starmask_width_ratio,
            accepted_limit=pipeline.cfg.stage7_starmask_width_ratio_max,
        )
        record_gate(
            "starmask_width_ratio",
            width_gate,
            (
            "starmask_width_ratio "
            f"{starmask_width_ratio:.3f}>{pipeline.cfg.stage7_starmask_width_ratio_max:.3f}"
            ),
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
            "compact_halo_evidence_available": float(
                compact_halo_evidence_available
            ),
            "diffuse_nebula_context": float(diffuse_nebula_context),
            "diffuse_nebula_protection_coverage": artifact_scores.get(
                "diffuse_nebula_protection_coverage",
                0.0,
            ),
            "black_hole_score": black_hole_score,
            "starmask_contamination": starmask_contamination,
            "starless_noise_gain": starless_noise_gain,
            "starless_dynamic_range_ratio": starless_dynamic_range_ratio,
            "starless_dynamic_range_ratio_raw": artifact_scores.get(
                "starless_dynamic_range_ratio_raw",
                starless_dynamic_range_ratio,
            ),
            "source_dynamic_range": artifact_scores.get("source_dynamic_range", 0.0),
            "starless_dynamic_range": artifact_scores.get("starless_dynamic_range", 0.0),
            "source_dynamic_range_raw": artifact_scores.get(
                "source_dynamic_range_raw",
                0.0,
            ),
            "starless_dynamic_range_raw": artifact_scores.get(
                "starless_dynamic_range_raw",
                0.0,
            ),
            "source_peak_signal": artifact_scores.get("source_peak_signal", 0.0),
            "starless_peak_signal": starless_peak_signal,
            "dynamic_range_calibration": dict(
                dynamic_assessment["measurement"]
            ),
            "starless_background_level": float(starless_features.bg_median),
            "starless_peak_background_ratio": dynamic_assessment[
                "peak_background_ratio"
            ],
            "dynamic_range_collapse": dynamic_assessment["collapsed"],
            "dynamic_range_collapse_advisory": dynamic_assessment["advisory"],
            "dynamic_range_collapse_hard_failed": dynamic_assessment["hard_failed"],
            "starmask_coverage_ratio": starmask_coverage_ratio,
            "starmask_width_ratio": starmask_width_ratio,
            "halo_threshold": halo_threshold,
            "bright_core_integrity_status": bright_core_integrity.get("status"),
            **{
                f"bright_core_{name}": value
                for name, value in (
                    bright_core_integrity.get("metrics") or {}
                ).items()
                if isinstance(value, (int, float, bool))
            },
            **{
                name: value
                for name, value in artifact_scores.items()
                if name.startswith("galaxy_")
            },
        },
        "local_issues": issues,
        "local_advisories": advisories,
        "quality_gates": quality_gates,
        "bright_core_integrity": bright_core_integrity,
    }
    all_issues = issues
    return {
        "attempt": attempt_name,
        "tool_label": tool_label,
        "source_stem": source_stem,
        "status": "ok" if not all_issues else "poor",
        "issues": all_issues,
        "local_issues": issues,
        "advisories": advisories,
        "local_advisories": advisories,
        "quality_gates": quality_gates,
        "bright_core_integrity": bright_core_integrity,
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
    galaxy_roi_available = float(derived.get("galaxy_roi_available", 0.0)) > 0.5
    galaxy_core_preservation = float(
        derived.get("galaxy_core_preservation_ratio", 1.0)
    )
    galaxy_core_contrast = float(
        derived.get("galaxy_core_contrast_ratio", 1.0)
    )
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
    galaxy_core_penalty = 0.0
    if galaxy_roi_available:
        galaxy_core_penalty = 2.0 * (
            max(
                0.0,
                float(
                    getattr(
                        pipeline.cfg,
                        "stage7_galaxy_core_preservation_ratio_min",
                        0.72,
                    )
                )
                - galaxy_core_preservation,
            )
            + max(
                0.0,
                float(
                    getattr(
                        pipeline.cfg,
                        "stage7_galaxy_core_contrast_ratio_min",
                        0.60,
                    )
                )
                - galaxy_core_contrast,
            )
        )
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
        + galaxy_core_penalty
    )

def stage7_repair_triggers(pipeline, quality: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(quality, dict):
        return []
    derived = quality.get("derived")
    if not isinstance(derived, dict):
        return []
    triggers: List[str] = []
    bright_core_integrity = quality.get("bright_core_integrity")
    if (
        isinstance(bright_core_integrity, dict)
        and bool(bright_core_integrity.get("applicable", False))
        and bool(bright_core_integrity.get("hard_failed", False))
    ):
        triggers.append("bright_core_integrity")
    try:
        residual = float(derived.get("residual_star_score", 0.0))
    except (TypeError, ValueError):
        residual = 0.0
    try:
        compact_residual = float(
            derived.get("compact_residual_star_score", 0.0)
        )
    except (TypeError, ValueError):
        compact_residual = 0.0
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
    try:
        galaxy_roi_available = float(derived.get("galaxy_roi_available", 0.0)) > 0.5
        galaxy_core_preservation = float(
            derived.get("galaxy_core_preservation_ratio", 1.0)
        )
        galaxy_core_contrast = float(
            derived.get("galaxy_core_contrast_ratio", 1.0)
        )
    except (TypeError, ValueError):
        galaxy_roi_available = False
        galaxy_core_preservation = 1.0
        galaxy_core_contrast = 1.0
    if residual > float(pipeline.cfg.stage7_residual_star_score_max):
        triggers.append("residual_stars")
    if compact_residual > float(pipeline.cfg.stage7_residual_star_score_max):
        triggers.append("compact_residual_stars")
    if halo > float(pipeline._stage7_effective_halo_threshold()):
        triggers.append("halo_residue")
    if compact_halo > float(pipeline._stage7_effective_halo_threshold()):
        triggers.append("compact_halo_residue")
    if black_hole > float(pipeline.cfg.stage7_black_hole_score_max):
        triggers.append("black_hole")
    if galaxy_roi_available and (
        galaxy_core_preservation
        < float(
            getattr(
                pipeline.cfg,
                "stage7_galaxy_core_preservation_ratio_min",
                0.72,
            )
        )
        or galaxy_core_contrast
        < float(
            getattr(
                pipeline.cfg,
                "stage7_galaxy_core_contrast_ratio_min",
                0.60,
            )
        )
    ):
        triggers.append("galaxy_core_damage")
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
        cleanup_borderline = bool(
            derived.get("starmask_cleanup_borderline", False)
        )
    else:
        halo_score = 0.0
        contamination_score = 0.0
        cleanup_borderline = False
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
        if pipeline._stage7_residual_star_score > threshold:
            over_ratio = pipeline._stage7_residual_star_score / threshold
            scale = 1.0 - 0.36 * max(0.0, over_ratio - 1.0)
            reason = (
                "stage7 residual stars high "
                f"({pipeline._stage7_residual_star_score:.3f}>{threshold:.3f})"
            )
        else:
            scale = 0.85
            reason = "stage7 residual-star diagnostic"
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

    if cleanup_borderline:
        borderline_scale = _clamp_float(
            getattr(
                pipeline.cfg,
                "stage7_starmask_diffuse_borderline_star_intensity_scale",
                0.70,
            ),
            0.35,
            1.0,
        )
        scale_candidates.append(
            (
                borderline_scale,
                "stage6 starmask diffuse residual inside advisory band",
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
        "starmask_cleanup_borderline": cleanup_borderline,
        "intensity_scale": pipeline._stage9_star_intensity_scale,
        "reason": pipeline._stage9_star_intensity_reason,
    }

def stage7_residual_suppression_strength(
    pipeline,
    quality: Optional[Dict[str, Any]],
) -> float:
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
