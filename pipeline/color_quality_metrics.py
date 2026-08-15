"""Report-only color contracts and cross-stage color-delta measurements.

The current pipeline does not yet carry a verified ICC working profile through
every FITS checkpoint.  Metrics in this module therefore use normalized RGB
chromaticity and opponent-hue proxies.  They are intentionally diagnostic and
must not be presented as CIE Delta-E or used as automatic gates.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

import numpy as np

from channel_semantics import BROADBAND_RGB_OSC, NARROWBAND_COMPOSITE


COLOR_CONTRACT_SCHEMA = "starun.color-contract.v1"
COLOR_QUALITY_SCHEMA = "starun.color-quality-report.v1"
COLOR_LEDGER_SCHEMA = "starun.color-adjustment-ledger.v1"


def _to_rgb_float(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[None, :, :], 3, axis=0)
    elif array.ndim == 3 and array.shape[0] >= 3:
        array = array[:3]
    elif array.ndim == 3 and array.shape[-1] >= 3:
        array = np.transpose(array[..., :3], (2, 0, 1))
    else:
        raise ValueError(f"unsupported RGB shape: {array.shape}")
    if np.issubdtype(array.dtype, np.integer):
        maximum = float(np.iinfo(array.dtype).max)
        array = array.astype(np.float32) / max(maximum, 1.0)
    else:
        array = array.astype(np.float32, copy=False)
        finite = np.abs(array[np.isfinite(array)])
        peak = float(np.max(finite)) if finite.size else 0.0
        if peak > 1.5:
            if peak <= 255.0 * 1.05:
                scale = 255.0
            elif peak <= 65535.0 * 1.05:
                scale = 65535.0
            else:
                scale = max(peak, 1.0)
            array = array / scale
    return np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> Optional[float]:
    values = np.asarray(values, dtype=np.float64).ravel()
    weights = np.asarray(weights, dtype=np.float64).ravel()
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 1e-6)
    if int(np.count_nonzero(valid)) < 16:
        return None
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values, kind="stable")
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    total = float(cumulative[-1])
    if total <= 1e-12:
        return None
    target = max(0.0, min(1.0, float(quantile))) * total
    index = min(int(np.searchsorted(cumulative, target, side="left")), values.size - 1)
    return float(values[index])


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> Optional[float]:
    values = np.asarray(values, dtype=np.float64).ravel()
    weights = np.asarray(weights, dtype=np.float64).ravel()
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 1e-6)
    if int(np.count_nonzero(valid)) < 16:
        return None
    total = float(np.sum(weights[valid]))
    if total <= 1e-12:
        return None
    return float(np.sum(values[valid] * weights[valid]) / total)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return (
        0.2126 * rgb[0]
        + 0.7152 * rgb[1]
        + 0.0722 * rgb[2]
    ).astype(np.float32)


def _chroma_proxy(rgb: np.ndarray) -> np.ndarray:
    maximum = np.max(rgb, axis=0)
    minimum = np.min(rgb, axis=0)
    mean = np.mean(np.abs(rgb), axis=0)
    return ((maximum - minimum) / np.maximum(mean, 0.02)).astype(np.float32)


def _chromaticity(rgb: np.ndarray) -> np.ndarray:
    denominator = np.sum(np.maximum(rgb, 0.0), axis=0, keepdims=True)
    return np.maximum(rgb, 0.0) / np.maximum(denominator, 1e-6)


def _opponent_hue(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    red_green = rgb[0] - rgb[1]
    blue_green = rgb[2] - rgb[1]
    hue = np.arctan2(blue_green, red_green)
    magnitude = np.sqrt(red_green * red_green + blue_green * blue_green)
    return hue.astype(np.float32), magnitude.astype(np.float32)


def _rounded(value: Optional[float]) -> Optional[float]:
    return None if value is None or not math.isfinite(value) else round(float(value), 7)


def _roi_metrics(
    baseline: np.ndarray,
    candidate: np.ndarray,
    weight: np.ndarray,
) -> Dict[str, Any]:
    weight = np.clip(np.asarray(weight, dtype=np.float32), 0.0, 1.0)
    baseline_luma = _luminance(baseline)
    candidate_luma = _luminance(candidate)
    baseline_chroma = _chroma_proxy(baseline)
    candidate_chroma = _chroma_proxy(candidate)
    baseline_xy = _chromaticity(baseline)
    candidate_xy = _chromaticity(candidate)
    chromaticity_delta = np.sqrt(np.sum((candidate_xy - baseline_xy) ** 2, axis=0))
    baseline_hue, baseline_hue_strength = _opponent_hue(baseline_xy)
    candidate_hue, candidate_hue_strength = _opponent_hue(candidate_xy)
    hue_delta = np.abs(
        np.arctan2(
            np.sin(candidate_hue - baseline_hue),
            np.cos(candidate_hue - baseline_hue),
        )
    )
    hue_weight = weight * np.clip(
        np.minimum(baseline_hue_strength, candidate_hue_strength) / 0.08,
        0.0,
        1.0,
    )
    luma_delta = np.abs(candidate_luma - baseline_luma)
    significant_change = np.max(np.abs(candidate - baseline), axis=0) > 1e-4
    weighted_count = float(np.sum(weight))
    selected_count = int(np.count_nonzero(weight > 0.05))

    baseline_chroma_p50 = _weighted_quantile(baseline_chroma, weight, 0.50)
    candidate_chroma_p50 = _weighted_quantile(candidate_chroma, weight, 0.50)
    return {
        "selected_pixel_count": selected_count,
        "weight_sum": round(weighted_count, 3),
        "baseline": {
            "luminance_p50": _rounded(
                _weighted_quantile(baseline_luma, weight, 0.50)
            ),
            "luminance_p95": _rounded(
                _weighted_quantile(baseline_luma, weight, 0.95)
            ),
            "chroma_proxy_p50": _rounded(baseline_chroma_p50),
            "chroma_proxy_p95": _rounded(
                _weighted_quantile(baseline_chroma, weight, 0.95)
            ),
        },
        "candidate": {
            "luminance_p50": _rounded(
                _weighted_quantile(candidate_luma, weight, 0.50)
            ),
            "luminance_p95": _rounded(
                _weighted_quantile(candidate_luma, weight, 0.95)
            ),
            "chroma_proxy_p50": _rounded(candidate_chroma_p50),
            "chroma_proxy_p95": _rounded(
                _weighted_quantile(candidate_chroma, weight, 0.95)
            ),
        },
        "delta": {
            "luminance_abs_p50": _rounded(
                _weighted_quantile(luma_delta, weight, 0.50)
            ),
            "luminance_abs_p95": _rounded(
                _weighted_quantile(luma_delta, weight, 0.95)
            ),
            "chromaticity_distance_p50": _rounded(
                _weighted_quantile(chromaticity_delta, weight, 0.50)
            ),
            "chromaticity_distance_p95": _rounded(
                _weighted_quantile(chromaticity_delta, weight, 0.95)
            ),
            "opponent_hue_drift_degrees_p50": _rounded(
                None
                if (value := _weighted_quantile(hue_delta, hue_weight, 0.50)) is None
                else math.degrees(value)
            ),
            "opponent_hue_drift_degrees_p95": _rounded(
                None
                if (value := _weighted_quantile(hue_delta, hue_weight, 0.95)) is None
                else math.degrees(value)
            ),
            "chroma_proxy_p50_change": _rounded(
                None
                if baseline_chroma_p50 is None or candidate_chroma_p50 is None
                else candidate_chroma_p50 - baseline_chroma_p50
            ),
            "changed_pixel_ratio": _rounded(
                _weighted_mean(significant_change.astype(np.float32), weight)
            ),
        },
    }


def _default_roi_masks(rgb: np.ndarray) -> Dict[str, np.ndarray]:
    luminance = _luminance(rgb)
    low = float(np.quantile(luminance, 0.35))
    high = float(np.quantile(luminance, 0.60))
    return {
        "background": (luminance <= low).astype(np.float32),
        "subject": (luminance >= high).astype(np.float32),
        "global": np.ones_like(luminance, dtype=np.float32),
    }


def _normalize_roi_masks(
    baseline: np.ndarray,
    masks: Optional[Mapping[str, Any]],
) -> Dict[str, np.ndarray]:
    if not isinstance(masks, Mapping):
        return _default_roi_masks(baseline)
    shape = baseline.shape[1:]

    def mask(name: str) -> np.ndarray:
        value = np.asarray(masks.get(name, np.zeros(shape)), dtype=np.float32)
        if value.shape != shape:
            return np.zeros(shape, dtype=np.float32)
        return np.clip(value, 0.0, 1.0)

    core = mask("core_mask")
    nebula = mask("nebula_mask")
    faint = mask("faint_nebula_mask")
    subject = np.maximum.reduce((core, nebula, faint))
    background = mask("background_mask") * (1.0 - subject)
    if int(np.count_nonzero(subject > 0.05)) < 16:
        return _default_roi_masks(baseline)
    return {
        "background": np.clip(background, 0.0, 1.0),
        "subject": np.clip(subject, 0.0, 1.0),
        "core": core,
        "global": np.ones(shape, dtype=np.float32),
    }


def resolve_color_contract(
    *,
    channel_profile: Optional[Mapping[str, Any]],
    color_report: Optional[Mapping[str, Any]],
    palette_report: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    profile = dict(channel_profile or {})
    report = dict(color_report or {})
    palette = dict(palette_report or {})
    report_channel_profile = report.get("channel_policy") or {}
    report_kind = (
        report_channel_profile.get("kind")
        if isinstance(report_channel_profile, Mapping)
        else None
    )
    kind = str(profile.get("kind") or report_kind or "unknown")
    physical = report.get("physical_color") or {}
    physical_anchor = bool(
        isinstance(physical, Mapping)
        and physical.get("accepted", False)
        and physical.get("feeds_main_pipeline", False)
    )
    palette_accepted = bool(palette.get("accepted", False))

    if palette_accepted:
        rendition_intent = "artistic_false_color"
        calibration_anchor = "stage8_pre_palette"
    elif kind == BROADBAND_RGB_OSC and physical_anchor:
        rendition_intent = "photometrically_anchored"
        calibration_anchor = "stage4_physical_color"
    elif kind == BROADBAND_RGB_OSC:
        rendition_intent = "natural_color_approximation"
        calibration_anchor = "stage4_degraded_or_preserved_color"
    elif kind == NARROWBAND_COMPOSITE:
        rendition_intent = "representative_color"
        calibration_anchor = "stage4_narrowband_parent"
    else:
        rendition_intent = "preserve_input_review"
        calibration_anchor = "unverified_input"

    return {
        "schema": COLOR_CONTRACT_SCHEMA,
        "channel_semantics": kind,
        "channel_axes": profile.get("axes") or {},
        "rendition_intent": rendition_intent,
        "calibration_anchor": calibration_anchor,
        "physical_anchor_accepted": physical_anchor,
        "calibration_method": physical.get("method") if isinstance(physical, Mapping) else None,
        "working_color_state": {
            "profile": "unknown",
            "primaries": "unknown",
            "transfer_curve": "nonlinear_stage7_output",
            "white_point": "unknown",
            "profile_verified": False,
            "conversion_lineage_verified": False,
        },
        "operation_policy": {
            "repeat_global_white_balance": False,
            "unconditional_scnr": False,
            "unbounded_global_channel_matrix": False,
            "masked_hue_preserving_chroma": kind == BROADBAND_RGB_OSC,
            "artistic_palette": palette_accepted,
        },
        "disclosure": {
            "palette": palette.get("palette") if palette_accepted else None,
            "synthetic_sii": bool(palette.get("synthetic_sii", False)),
            "profile_dependent_metrics_available": False,
        },
    }


def physical_broadband_anchor_accepted(
    channel_profile: Optional[Mapping[str, Any]],
    color_report: Optional[Mapping[str, Any]],
) -> bool:
    contract = resolve_color_contract(
        channel_profile=channel_profile,
        color_report=color_report,
    )
    return bool(
        contract["channel_semantics"] == BROADBAND_RGB_OSC
        and contract["physical_anchor_accepted"]
    )


def build_color_quality_report(
    baseline_image: np.ndarray,
    candidate_image: np.ndarray,
    *,
    stage: str,
    baseline_name: str,
    candidate_name: str,
    contract: Mapping[str, Any],
    masks: Optional[Mapping[str, Any]] = None,
    requested_saturation: float = 0.0,
    effective_saturation: float = 0.0,
    applied_saturation: float = 0.0,
    operation: str = "report_only",
) -> Dict[str, Any]:
    try:
        baseline = _to_rgb_float(baseline_image)
        candidate = _to_rgb_float(candidate_image)
    except (TypeError, ValueError) as error:
        return {
            "schema": COLOR_QUALITY_SCHEMA,
            "stage": stage,
            "status": "unavailable",
            "mode": "report_only",
            "used_for_gate": False,
            "issues": [str(error)],
            "contract": dict(contract),
        }
    if baseline.shape != candidate.shape:
        return {
            "schema": COLOR_QUALITY_SCHEMA,
            "stage": stage,
            "status": "unavailable",
            "mode": "report_only",
            "used_for_gate": False,
            "issues": [
                f"shape_mismatch baseline={baseline.shape} candidate={candidate.shape}"
            ],
            "contract": dict(contract),
        }

    roi_masks = _normalize_roi_masks(baseline, masks)
    roi_reports = {
        name: _roi_metrics(baseline, candidate, weight)
        for name, weight in roi_masks.items()
    }
    baseline_out_of_range = np.any((baseline < 0.0) | (baseline > 1.0), axis=0)
    candidate_out_of_range = np.any((candidate < 0.0) | (candidate > 1.0), axis=0)
    baseline_clip = np.any((baseline <= 1e-6) | (baseline >= 1.0 - 1e-6), axis=0)
    candidate_clip = np.any((candidate <= 1e-6) | (candidate >= 1.0 - 1e-6), axis=0)
    subject_delta = roi_reports.get("subject", {}).get("delta", {})
    background_delta = roi_reports.get("background", {}).get("delta", {})

    ledger_entry = {
        "schema": COLOR_LEDGER_SCHEMA,
        "stage": stage,
        "operation": operation,
        "baseline": baseline_name,
        "candidate": candidate_name,
        "requested_saturation": round(float(requested_saturation), 7),
        "effective_saturation": round(float(effective_saturation), 7),
        "applied_saturation": round(float(applied_saturation), 7),
        "measured_subject_chroma_proxy_p50_change": subject_delta.get(
            "chroma_proxy_p50_change"
        ),
        "measured_background_chroma_proxy_p50_change": background_delta.get(
            "chroma_proxy_p50_change"
        ),
        "profile_dependent_measurement": False,
    }
    return {
        "schema": COLOR_QUALITY_SCHEMA,
        "stage": stage,
        "status": "reported",
        "mode": "report_only",
        "used_for_gate": False,
        "baseline": baseline_name,
        "candidate": candidate_name,
        "contract": dict(contract),
        "measurement_domain": {
            "space": "normalized_rgb_chromaticity_opponent_proxy",
            "luminance_weights": "Rec.709 coefficients used as an engineering proxy",
            "profile_verified": False,
            "brightness_matching": "chromaticity normalization",
        },
        "rois": roi_reports,
        "gamut_and_clipping": {
            "baseline_out_of_range_ratio": round(float(np.mean(baseline_out_of_range)), 8),
            "candidate_out_of_range_ratio": round(float(np.mean(candidate_out_of_range)), 8),
            "baseline_channel_clip_ratio": round(float(np.mean(baseline_clip)), 8),
            "candidate_channel_clip_ratio": round(float(np.mean(candidate_clip)), 8),
            "pre_clip_gamut_scale_available": False,
        },
        "profile_dependent_metrics": {
            "status": "unavailable",
            "delta_e00": None,
            "oklab_delta": None,
            "reason": "working profile/primaries/TRC/white point lineage is not yet verified",
        },
        "ledger_entry": ledger_entry,
        "issues": [],
    }


__all__ = [
    "COLOR_CONTRACT_SCHEMA",
    "COLOR_LEDGER_SCHEMA",
    "COLOR_QUALITY_SCHEMA",
    "build_color_quality_report",
    "physical_broadband_anchor_accepted",
    "resolve_color_contract",
]
