"""Stage 6 stretch candidate scoring."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


MODE_ALIASES = {
    "asinh_core_protect": "asinh",
    "asinh_mild_ghs": "asinh_ghs",
    "masked_curve_dark_boost": "asinh",
    "bright_nebula_hdr_masked": "bright_nebula_hdr_masked",
    "mild_histogram": "ghs",
    "masked_galaxy_stretch": "asinh_ghs",
    "star_color_preserving_stretch": "asinh",
    "autostretch_reference": "autostretch",
}


def candidate_modes(policy: Dict[str, Any], target_type: str) -> List[str]:
    stage6 = policy.get("stage6_stretch") if isinstance(policy, dict) else {}
    modes = stage6.get("candidate_mode") if isinstance(stage6, dict) else None
    if isinstance(modes, list) and modes:
        return [str(mode) for mode in modes]
    if target_type in {"bright_emission_reflection_nebula", "generic_low_snr_safe"}:
        return [
            "bright_nebula_hdr_masked",
            "asinh_core_protect",
            "asinh_mild_ghs",
            "masked_curve_dark_boost",
            "autostretch_reference",
        ]
    if target_type in {"large_galaxy", "small_galaxy"}:
        return ["masked_galaxy_stretch", "asinh_mild_ghs", "mild_histogram", "autostretch_reference"]
    if target_type in {"globular_cluster", "open_cluster"}:
        return ["star_color_preserving_stretch", "asinh_core_protect", "autostretch_reference"]
    return ["asinh_core_protect", "asinh_mild_ghs", "autostretch_reference"]


def build_candidate_spec(mode: str, cfg: Any) -> Dict[str, Any]:
    method = MODE_ALIASES.get(mode, mode)
    asinh = float(getattr(cfg, "asinh_stretch", 3.0))
    offset = float(getattr(cfg, "asinh_offset", 0.001))
    ghs_amount = float(getattr(cfg, "ghs_stretchamount", 2.0))
    shadows = float(getattr(cfg, "ghs_shadowsclip", -2.8))
    if mode == "asinh_core_protect":
        asinh = min(asinh, 2.25)
    elif mode == "bright_nebula_hdr_masked":
        asinh = min(max(asinh, 2.20), 2.80)
        offset = max(offset, 0.0018)
    elif mode == "asinh_mild_ghs":
        asinh = min(asinh, 2.20)
        ghs_amount = min(ghs_amount, 1.05)
        shadows = max(shadows, -2.10)
    elif mode == "masked_curve_dark_boost":
        asinh = min(asinh, 2.05)
        offset = max(offset, 0.0018)
    elif mode == "mild_histogram":
        ghs_amount = min(ghs_amount, 1.20)
        shadows = max(shadows, -2.45)
    elif mode == "masked_galaxy_stretch":
        asinh = min(asinh, 2.35)
        ghs_amount = min(ghs_amount, 1.45)
    elif mode == "star_color_preserving_stretch":
        asinh = min(asinh, 2.00)
    return {
        "source": "policy",
        "mode": mode,
        "method": method,
        "params": {
            "asinh_stretch": asinh,
            "asinh_offset": offset,
            "ghs_shadowsclip": shadows,
            "ghs_stretchamount": ghs_amount,
            "bg_pedestal": 0.024 if mode == "bright_nebula_hdr_masked" else 0.0,
            "faint_boost": 0.018 if mode == "bright_nebula_hdr_masked" else 0.0,
            "core_protection": 0.72 if mode == "bright_nebula_hdr_masked" else 0.0,
            "shadow_chroma_damping": 0.28 if mode == "bright_nebula_hdr_masked" else 0.0,
            "faint_saturation_boost": 0.026 if mode == "bright_nebula_hdr_masked" else 0.0,
        },
        "summary": f"policy candidate {mode}",
    }


def score_candidate(
    mode: str,
    metrics: Optional[Dict[str, Any]],
    baseline: Optional[Dict[str, Any]],
    policy: Dict[str, Any],
) -> Dict[str, Any]:
    metrics = metrics or {}
    baseline = baseline or {}
    stage6 = policy.get("stage6_stretch") if isinstance(policy, dict) else {}
    weights = stage6.get("scoring", {}) if isinstance(stage6, dict) else {}
    bg_dirty = float(metrics.get("dirty_background_score", 0.0) or 0.0)
    core_clip = float(metrics.get("core_clip_ratio", 0.0) or 0.0)
    chroma_noise = float(metrics.get("chroma_noise_score", 0.0) or 0.0)
    baseline_chroma = float(baseline.get("chroma_noise_score", chroma_noise) or chroma_noise or 1e-6)
    chroma_growth = chroma_noise / max(baseline_chroma, 1e-6)
    detail = float(metrics.get("faint_structure_score", metrics.get("object_detail_score", 0.0)) or 0.0)
    star_bloat = float(metrics.get("star_bloat_score", 0.0) or 0.0)
    color_shift = _color_shift(metrics, baseline)
    core_weight = float(weights.get("core_blowout_weight", 0.30))
    bg_weight = float(weights.get("bg_noise_weight", 0.35))
    detail_weight = float(weights.get("nebulosity_weight", weights.get("detail_weight", 0.20)))
    star_weight = float(weights.get("star_bloat_weight", 0.10))
    color_weight = float(weights.get("color_shift_weight", 0.05))
    risk = (
        core_weight * min(core_clip * 30.0, 1.0)
        + bg_weight * bg_dirty
        + star_weight * star_bloat
        + color_weight * color_shift
    )
    score = max(0.0, min(1.0, 0.55 + detail_weight * detail - risk))
    allowed, reason = allowed_as_final(mode, metrics, policy)
    return {
        "score": round(float(score), 4),
        "metrics": {
            "bg_dirty_score": bg_dirty,
            "core_clip_score": min(core_clip * 30.0, 1.0),
            "object_detail_score": detail,
            "star_bloat_score": star_bloat,
            "color_shift_score": color_shift,
            "chroma_noise_growth": chroma_growth,
        },
        "allowed_as_final": allowed,
        "reject_reason": reason,
    }


def allowed_as_final(mode: str, metrics: Dict[str, Any], policy: Dict[str, Any]) -> tuple[bool, str]:
    stage6 = policy.get("stage6_stretch") if isinstance(policy, dict) else {}
    forbidden = stage6.get("forbidden_when_dirty", []) if isinstance(stage6, dict) else []
    dirty = float(metrics.get("dirty_background_score", 0.0) or 0.0)
    core_clip = float(metrics.get("core_clip_ratio", 0.0) or 0.0)
    chroma_noise = float(metrics.get("chroma_noise_score", 0.0) or 0.0)
    chroma_growth = float(metrics.get("chroma_noise_growth", 1.0) or 1.0)
    thresholds = stage6.get("hard_reject", {}) if isinstance(stage6, dict) else {}
    mode_overrides = thresholds.get("mode_overrides", {}) if isinstance(thresholds, dict) else {}
    mode_thresholds = mode_overrides.get(mode, {}) if isinstance(mode_overrides, dict) else {}
    if not isinstance(mode_thresholds, dict):
        mode_thresholds = {}
    max_dirty = float(mode_thresholds.get("max_bg_dirty_score", thresholds.get("max_bg_dirty_score", 0.42)))
    max_core = float(mode_thresholds.get("max_core_clip_score", thresholds.get("max_core_clip_score", 0.18)))
    max_chroma_growth = float(
        mode_thresholds.get("max_chroma_noise_growth", thresholds.get("max_chroma_noise_growth", 1.35))
    )
    max_chroma_noise = mode_thresholds.get("max_chroma_noise_score")
    if "autostretch" in str(mode) and ("autostretch" in forbidden or stage6.get("allow_autostretch_as_reference_only")):
        if dirty > max_dirty:
            return False, "dirty_background_policy_forbids_autostretch"
        return False, "autostretch_reference_only"
    if dirty > max_dirty:
        return False, "bg_dirty_score_hard_reject"
    if min(core_clip * 30.0, 1.0) > max_core:
        return False, "core_clip_score_hard_reject"
    if chroma_growth > max_chroma_growth:
        if max_chroma_noise is not None and chroma_noise <= float(max_chroma_noise):
            return True, ""
        return False, "chroma_noise_growth_hard_reject"
    return True, ""


def choose_best(candidates: List[Dict[str, Any]], fallback_mode: str) -> Optional[Dict[str, Any]]:
    allowed = [item for item in candidates if item.get("allowed_as_final") and item.get("status") == "ok"]
    if allowed:
        return max(allowed, key=lambda item: float(item.get("score", 0.0)))
    return None


def _color_shift(metrics: Dict[str, Any], baseline: Dict[str, Any]) -> float:
    red = abs(float(metrics.get("red_dominance", 1.0) or 1.0) - float(baseline.get("red_dominance", 1.0) or 1.0))
    blue = abs(float(metrics.get("blue_dominance", 1.0) or 1.0) - float(baseline.get("blue_dominance", 1.0) or 1.0))
    green = abs(float(metrics.get("green_cast", 1.0) or 1.0) - float(baseline.get("green_cast", 1.0) or 1.0))
    return max(0.0, min(1.0, (red + blue + green) / 1.5))
