"""Versioned invariants for the Stage 3 background-safety contract."""
from __future__ import annotations

from typing import Any, Dict


STAGE3_ALGORITHM_CONTRACT_VERSION = "1.3.0"
STAGE3_BACKGROUND_QUALITY_SCHEMA = "starun.stage3-background-quality.v5"

STAGE3_SIGNIFICANCE_SIGMA = 3.0
STAGE3_MIN_VALIDATION_PATCHES = 4
STAGE3_MIN_TARGET_SKY_PLANE_SAMPLES = 4
STAGE3_MIN_SPATIAL_QUADRANTS = 3
STAGE3_MIN_SPATIAL_GRID_CELLS = 8
STAGE3_MIN_AXIS_SPAN_RATIO = 0.55
STAGE3_MIN_USABLE_SKY_FRACTION = 0.50
STAGE3_MAX_SOURCE_MASK_FRACTION = 0.50
STAGE3_RADIAL_BIC_DELTA_MIN = 6.0
STAGE3_RADIAL_RESIDUAL_SPAN_RATIO_MAX = 0.40

STAGE3_FINAL_DIRTY_WARNING_MIN = 0.35
STAGE3_FINAL_GRADIENT_RETENTION_WARNING = 0.92
STAGE3_DIRECTIONAL_PATTERN_PENALTY_WEIGHT = 0.45

STAGE3_GATE_PROFILE_DEFAULT = "output_first"
STAGE3_GATE_PROFILES: Dict[str, Dict[str, Any]] = {
    "output_first": {
        "strict_legacy": False,
        "hard_significance_sigma": 6.0,
        "flux_retention_soft_min": 0.65,
        "flux_retention_soft_max": 1.50,
        "flux_retention_hard_min": 0.20,
        "flux_retention_hard_max": 3.00,
        "morphology_soft_min": 0.85,
        "morphology_hard_min": 0.40,
        "centroid_soft_max": 0.05,
        "centroid_hard_max": 0.15,
        "structure_hard_sigma": 20.0,
        "span_hard_ratio": 2.00,
        "rms_hard_ratio": 1.75,
        "pattern_hard_score": 0.90,
        "pattern_hard_growth": 0.35,
        "sufficient_max_background_score": 0.65,
        "sufficient_dirty_score_max": 0.55,
        "sufficient_max_bg_std_growth": 1.25,
        "sufficient_color_shift_max": 0.30,
    },
    "balanced": {
        "strict_legacy": False,
        "hard_significance_sigma": 6.0,
        "flux_retention_soft_min": 0.75,
        "flux_retention_soft_max": 1.25,
        "flux_retention_hard_min": 0.40,
        "flux_retention_hard_max": 2.00,
        "morphology_soft_min": 0.90,
        "morphology_hard_min": 0.60,
        "centroid_soft_max": 0.03,
        "centroid_hard_max": 0.10,
        "structure_hard_sigma": 12.0,
        "span_hard_ratio": 1.50,
        "rms_hard_ratio": 1.35,
        "pattern_hard_score": 0.85,
        "pattern_hard_growth": 0.25,
        "sufficient_max_background_score": 0.50,
        "sufficient_dirty_score_max": 0.40,
        "sufficient_max_bg_std_growth": 1.15,
        "sufficient_color_shift_max": 0.24,
    },
    "strict": {
        "strict_legacy": True,
        "hard_significance_sigma": STAGE3_SIGNIFICANCE_SIGMA,
        "structure_hard_sigma": STAGE3_SIGNIFICANCE_SIGMA,
        "sufficient_max_background_score": 0.34,
        "sufficient_dirty_score_max": 0.32,
        "sufficient_max_bg_std_growth": 1.08,
        "sufficient_color_shift_max": 0.18,
    },
}


def normalize_stage3_gate_profile(value: Any) -> str:
    """Return a supported Stage 3 gate profile, defaulting output-first."""
    profile = str(value or STAGE3_GATE_PROFILE_DEFAULT).strip().lower()
    if profile not in STAGE3_GATE_PROFILES:
        return STAGE3_GATE_PROFILE_DEFAULT
    return profile


def stage3_gate_thresholds(value: Any) -> Dict[str, Any]:
    """Return an isolated threshold snapshot for reports and gate evaluation."""
    profile = normalize_stage3_gate_profile(value)
    return {"profile": profile, **dict(STAGE3_GATE_PROFILES[profile])}

STAGE3_BACKGROUND_SCORE_WEIGHTS: Dict[str, float] = {
    "dirty_background_score": 1.25,
    "gradient_score": 0.85,
    "chroma_noise_score": 0.45,
    "bg_std_growth": 0.35,
    "color_shift": 0.45,
}


def stage3_static_contract_manifest() -> Dict[str, Any]:
    """Return immutable Stage 3 thresholds for plans and quality reports."""
    return {
        "algorithm_contract_version": STAGE3_ALGORITHM_CONTRACT_VERSION,
        "quality_report_schema": STAGE3_BACKGROUND_QUALITY_SCHEMA,
        "statistical_significance_sigma": STAGE3_SIGNIFICANCE_SIGMA,
        "minimum_validation_patches": STAGE3_MIN_VALIDATION_PATCHES,
        "minimum_target_sky_plane_samples": (
            STAGE3_MIN_TARGET_SKY_PLANE_SAMPLES
        ),
        "spatial_coverage": {
            "minimum_quadrants": STAGE3_MIN_SPATIAL_QUADRANTS,
            "minimum_grid_cells": STAGE3_MIN_SPATIAL_GRID_CELLS,
            "minimum_axis_span_ratio": STAGE3_MIN_AXIS_SPAN_RATIO,
        },
        "source_masked_sky": {
            "minimum_usable_sky_fraction": STAGE3_MIN_USABLE_SKY_FRACTION,
            "maximum_source_mask_fraction": STAGE3_MAX_SOURCE_MASK_FRACTION,
        },
        "radial_model_review": {
            "minimum_bic_delta": STAGE3_RADIAL_BIC_DELTA_MIN,
            "maximum_residual_span_ratio": (
                STAGE3_RADIAL_RESIDUAL_SPAN_RATIO_MAX
            ),
        },
        "final_consistency_warning": {
            "dirty_background_score_min": STAGE3_FINAL_DIRTY_WARNING_MIN,
            "gradient_retention_ratio": (
                STAGE3_FINAL_GRADIENT_RETENTION_WARNING
            ),
        },
        "background_score_weights": dict(STAGE3_BACKGROUND_SCORE_WEIGHTS),
        "directional_pattern_penalty_weight": (
            STAGE3_DIRECTIONAL_PATTERN_PENALTY_WEIGHT
        ),
        "gate_profiles": {
            name: dict(thresholds)
            for name, thresholds in STAGE3_GATE_PROFILES.items()
        },
        "default_gate_profile": STAGE3_GATE_PROFILE_DEFAULT,
    }


__all__ = [
    "STAGE3_ALGORITHM_CONTRACT_VERSION",
    "STAGE3_BACKGROUND_QUALITY_SCHEMA",
    "STAGE3_BACKGROUND_SCORE_WEIGHTS",
    "STAGE3_FINAL_DIRTY_WARNING_MIN",
    "STAGE3_FINAL_GRADIENT_RETENTION_WARNING",
    "STAGE3_GATE_PROFILE_DEFAULT",
    "STAGE3_GATE_PROFILES",
    "STAGE3_DIRECTIONAL_PATTERN_PENALTY_WEIGHT",
    "STAGE3_MAX_SOURCE_MASK_FRACTION",
    "STAGE3_MIN_AXIS_SPAN_RATIO",
    "STAGE3_MIN_SPATIAL_GRID_CELLS",
    "STAGE3_MIN_SPATIAL_QUADRANTS",
    "STAGE3_MIN_USABLE_SKY_FRACTION",
    "STAGE3_MIN_VALIDATION_PATCHES",
    "STAGE3_MIN_TARGET_SKY_PLANE_SAMPLES",
    "STAGE3_RADIAL_BIC_DELTA_MIN",
    "STAGE3_RADIAL_RESIDUAL_SPAN_RATIO_MAX",
    "STAGE3_SIGNIFICANCE_SIGMA",
    "normalize_stage3_gate_profile",
    "stage3_gate_thresholds",
    "stage3_static_contract_manifest",
]
