"""Compatibility pre-starless quality gate."""
from __future__ import annotations

from typing import Any, Dict, List


CONSERVATIVE_TYPES = {
    "bright_emission_reflection_nebula",
    "reflection_nebula_cluster",
    "dark_nebula_low_contrast",
    "generic_low_snr_safe",
}


def evaluate_pre_starless_gate(
    metrics: Dict[str, Any],
    profile: Dict[str, Any],
    policy: Dict[str, Any],
) -> Dict[str, Any]:
    target_type = str(profile.get("target_type") or "generic_low_snr_safe")
    gate = policy.get("stage6_5_pre_starless_gate", {}) if isinstance(policy, dict) else {}
    max_dirty = float(gate.get("max_bg_dirty_score", 0.35))
    max_clip = float(gate.get("max_core_clip_ratio", 0.01))
    max_halo = float(gate.get("max_star_halo_risk", 0.65))
    max_global_dark = float(gate.get("max_global_dark_ratio", gate.get("max_black_pixel_ratio", 0.70)))
    min_bg = float(gate.get("min_bg_median", 0.005))
    max_edge = float(gate.get("max_edge_black_ratio", 0.10))
    dirty = float(metrics.get("dirty_background_score", 0.0) or 0.0)
    clip = float(metrics.get("core_clip_ratio", 0.0) or 0.0)
    halo = float(metrics.get("halo_risk_score", 0.0) or 0.0)
    global_dark = float(metrics.get("global_dark_ratio", metrics.get("black_pixel_ratio", 0.0)) or 0.0)
    bg_median = float(metrics.get("bg_median", 0.0) or 0.0)
    edge = float(metrics.get("edge_black_ratio", 0.0) or 0.0)

    reasons: List[str] = []
    if dirty > max_dirty:
        reasons.append("background_dirty_score too high")
    if clip > max_clip:
        reasons.append("core too bright")
    if halo > max_halo:
        reasons.append("star_halo_risk high")
    if global_dark > max_global_dark:
        reasons.append("global_dark_ratio too high")
    if bg_median > 0.0 and bg_median < min_bg:
        reasons.append("background median too low")
    if edge > max_edge:
        reasons.append("edge_black_ratio too high")

    recommendation = "stage7_stretched"
    ready = True
    if target_type in {"globular_cluster", "open_cluster"}:
        recommendation = "stage7_stretched"
        ready = True
        if not reasons:
            reasons.append("star-preserve target; standard starless should be optional")
    elif reasons and (target_type in CONSERVATIVE_TYPES or gate.get("require_conservative_starless_input", False)):
        recommendation = str(gate.get("default_starless_input") or "stage7_ultra_conservative_asinh")
        ready = False
    elif target_type in {"large_galaxy", "small_galaxy"} and reasons:
        recommendation = "stage7_conservative_asinh"
        ready = False

    return {
        "stage": "stage6_5_pre_starless_gate",
        "ready_for_starless": ready,
        "reason": reasons,
        "recommended_starless_input": recommendation,
        "fallback_created": False,
        "target_type": target_type,
        "metrics": metrics,
        "thresholds": {
            "max_bg_dirty_score": max_dirty,
            "max_core_clip_ratio": max_clip,
            "max_star_halo_risk": max_halo,
            "max_global_dark_ratio": max_global_dark,
            "max_black_pixel_ratio": max_global_dark,
            "min_bg_median": min_bg,
            "max_edge_black_ratio": max_edge,
        },
    }
