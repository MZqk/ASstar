"""Policy loading and selection for adaptive pipeline stages."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_POLICY_NAME = "generic_low_snr_safe"
POLICY_DIR = Path(__file__).resolve().parent / "configs" / "policies"


DEFAULT_POLICY: Dict[str, Any] = {
    "policy_name": DEFAULT_POLICY_NAME,
    "applies_to": {"target_types": ["generic_low_snr_safe"]},
    "stage3_background": {
        "model_priority": ["rbf_smooth", "polynomial_low_order"],
        "protect_nebulosity": True,
        "protect_bright_core": True,
        "max_bg_std_growth": 1.05,
        "fallback_to_safe_model": True,
    },
    "stage4_color": {
        "prefer_spcc": True,
        "reduce_saturation_if_solution_imprecise": True,
        "blue_gain_limit": 0.90,
        "red_gain_limit": 1.08,
        "max_allowed_saturation_boost": 0.10,
    },
    "stage5_linear": {
        "denoise_mode": "chroma_first",
        "sharpen_mode": "conservative",
        "protect_background": True,
        "protect_star_halo": True,
        "avoid_global_sharpen": True,
    },
    "stage6_stretch": {
        "candidate_mode": ["asinh_core_protect", "asinh_mild_ghs", "autostretch_reference"],
        "forbidden_when_dirty": ["autostretch"],
        "allow_autostretch_as_reference_only": True,
        "fallback_candidate": "asinh_core_protect",
        "scoring": {
            "core_blowout_weight": 0.30,
            "bg_noise_weight": 0.35,
            "nebulosity_weight": 0.20,
            "star_bloat_weight": 0.10,
            "color_shift_weight": 0.05,
        },
    },
    "stage6_5_pre_starless_gate": {
        "max_bg_dirty_score": 0.35,
        "max_core_clip_ratio": 0.01,
        "max_star_halo_risk": 0.60,
        "require_conservative_starless_input": True,
        "default_starless_input": "stage7_ultra_conservative_asinh",
    },
}

BUILTIN_POLICY_OVERLAYS: Dict[str, Dict[str, Any]] = {
    "bright_nebula_hdr_conservative": {
        "policy_name": "bright_nebula_hdr_conservative",
        "applies_to": {"target_types": ["bright_emission_reflection_nebula"]},
        "stage3_background": {
            "model_priority": ["rbf_smooth", "polynomial_low_order"],
            "protect_nebulosity": True,
            "reject_samples_on_nebula": True,
            "protect_bright_core": True,
            "max_bg_std_growth": 1.03,
            "fallback_to_safe_model": True,
        },
        "stage4_color": {
            "prefer_spcc": True,
            "allow_pcc_warning": False,
            "reduce_saturation_if_solution_imprecise": True,
            "blue_gain_limit": 0.85,
            "red_gain_limit": 1.10,
            "max_allowed_saturation_boost": 0.14,
        },
        "stage5_linear": {
            "denoise_mode": "chroma_first",
            "sharpen_mode": "object_masked",
            "protect_background": True,
            "protect_star_halo": True,
            "avoid_global_sharpen": True,
        },
        "stage6_stretch": {
            "candidate_mode": [
                "bright_nebula_hdr_masked",
                "asinh_core_protect",
                "asinh_mild_ghs",
                "masked_curve_dark_boost",
                "autostretch_reference",
            ],
            "forbidden_when_dirty": ["autostretch"],
            "allow_autostretch_as_reference_only": True,
            "fallback_candidate": "asinh_core_protect",
            "hard_reject": {
                "max_bg_dirty_score": 0.42,
                "max_core_clip_score": 0.18,
                "max_chroma_noise_growth": 1.35,
                "mode_overrides": {
                    "bright_nebula_hdr_masked": {
                        "max_bg_dirty_score": 0.10,
                        "max_chroma_noise_growth": 1.65,
                        "max_chroma_noise_score": 0.018,
                    },
                },
            },
            "scoring": {
                "core_blowout_weight": 0.35,
                "bg_noise_weight": 0.35,
                "nebulosity_weight": 0.20,
                "star_bloat_weight": 0.10,
                "color_shift_weight": 0.05,
            },
        },
        "stage6_5_pre_starless_gate": {
            "max_bg_dirty_score": 0.35,
            "max_core_clip_ratio": 0.01,
            "max_star_halo_risk": 0.65,
            "require_conservative_starless_input": True,
            "default_starless_input": "stage7_ultra_conservative_asinh",
        },
    }
}


def _load_json_or_yaml(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    try:
        import yaml  # type: ignore

        parsed = yaml.safe_load(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    raise ValueError(f"policy file is not valid JSON/YAML: {path}")


def load_policy(policy_name: str, *, policy_dir: Optional[Path] = None) -> Dict[str, Any]:
    clean_name = (policy_name or DEFAULT_POLICY_NAME).strip() or DEFAULT_POLICY_NAME
    root = policy_dir or POLICY_DIR
    for suffix in (".yaml", ".yml", ".json"):
        path = root / f"{clean_name}{suffix}"
        if path.is_file():
            policy = _load_json_or_yaml(path)
            merged = copy.deepcopy(DEFAULT_POLICY)
            _deep_update(merged, policy)
            merged["policy_name"] = str(policy.get("policy_name") or clean_name)
            return merged
    if clean_name in BUILTIN_POLICY_OVERLAYS:
        merged = copy.deepcopy(DEFAULT_POLICY)
        _deep_update(merged, BUILTIN_POLICY_OVERLAYS[clean_name])
        merged["policy_name"] = clean_name
        return merged
    if clean_name != DEFAULT_POLICY_NAME:
        return load_policy(DEFAULT_POLICY_NAME, policy_dir=root)
    return copy.deepcopy(DEFAULT_POLICY)


def _deep_update(base: Dict[str, Any], overlay: Dict[str, Any]) -> None:
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def policy_for_profile(profile: Dict[str, Any], *, policy_dir: Optional[Path] = None) -> Dict[str, Any]:
    policy_name = (
        profile.get("pipeline")
        or profile.get("default_policy")
        or DEFAULT_POLICY_NAME
    )
    try:
        return load_policy(str(policy_name), policy_dir=policy_dir)
    except Exception:
        return copy.deepcopy(DEFAULT_POLICY)
