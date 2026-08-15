"""Cross-stage safety rules for target routing and color/denoise budgets."""
from __future__ import annotations

from typing import Any, Dict, Mapping


STAR_PRESERVE_TARGET_TYPES = frozenset(
    {
        "globular_cluster",
        "open_cluster",
        "reflection_nebula_cluster",
    }
)


def should_bypass_star_separation(target_type: str, *, enabled: bool = True) -> bool:
    """Return whether stars are part of the subject and must stay in the main image."""
    return bool(enabled and str(target_type or "").strip() in STAR_PRESERVE_TARGET_TYPES)


def color_safety_limits(
    policy: Mapping[str, Any] | None,
    color_report: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    """Resolve Stage 4 policy limits, preferring the concrete calibration report."""
    stage4_policy = {}
    if isinstance(policy, Mapping):
        candidate = policy.get("stage4_color", {})
        if isinstance(candidate, Mapping):
            stage4_policy = dict(candidate)

    adjustments = {}
    if isinstance(color_report, Mapping):
        candidate = color_report.get("policy_adjustments", {})
        if isinstance(candidate, Mapping):
            adjustments = dict(candidate)

    def number(name: str, default: float) -> float:
        value = adjustments.get(name, stage4_policy.get(name, default))
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    max_saturation = max(0.0, min(0.65, number("max_allowed_saturation_boost", 0.10)))
    reduce_saturation = bool(adjustments.get("reduce_saturation_boost", False))
    if reduce_saturation:
        max_saturation *= 0.5

    return {
        "max_saturation_boost": max_saturation,
        "red_gain_limit": max(0.80, min(1.25, number("red_gain_limit", 1.08))),
        "blue_gain_limit": max(0.80, min(1.25, number("blue_gain_limit", 1.00))),
        "reduce_saturation_boost": reduce_saturation,
    }


def clamp_saturation_boost(
    requested: float,
    *,
    already_applied: float,
    limits: Mapping[str, Any],
) -> float:
    """Clamp a positive saturation request to the remaining cross-stage budget."""
    value = float(requested)
    if value <= 0.0:
        return value
    maximum = max(0.0, float(limits.get("max_saturation_boost", 0.10)))
    remaining = max(0.0, maximum - max(0.0, float(already_applied)))
    return min(value, remaining)


def should_skip_final_denoise(
    *,
    stage5_denoise_applied: bool,
    stage8_final_quality: str,
    stage8_fallback_used: bool,
) -> bool:
    """Avoid a second denoise when the earlier denoise and later quality gate passed."""
    return bool(
        stage5_denoise_applied
        and not stage8_fallback_used
        and str(stage8_final_quality or "").lower()
        in {"ok", "conservative_skipped", "star_preserve_bypass"}
    )
