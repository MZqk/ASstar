"""Versioned invariants for the Stage 3 background-safety contract."""
from __future__ import annotations

from typing import Any, Dict


STAGE3_ALGORITHM_CONTRACT_VERSION = "1.1.0"
STAGE3_BACKGROUND_QUALITY_SCHEMA = "seestar.stage3-background-quality.v3"

STAGE3_SIGNIFICANCE_SIGMA = 3.0
STAGE3_MIN_VALIDATION_PATCHES = 8
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
    }


__all__ = [
    "STAGE3_ALGORITHM_CONTRACT_VERSION",
    "STAGE3_BACKGROUND_QUALITY_SCHEMA",
    "STAGE3_BACKGROUND_SCORE_WEIGHTS",
    "STAGE3_FINAL_DIRTY_WARNING_MIN",
    "STAGE3_FINAL_GRADIENT_RETENTION_WARNING",
    "STAGE3_DIRECTIONAL_PATTERN_PENALTY_WEIGHT",
    "STAGE3_MAX_SOURCE_MASK_FRACTION",
    "STAGE3_MIN_AXIS_SPAN_RATIO",
    "STAGE3_MIN_SPATIAL_GRID_CELLS",
    "STAGE3_MIN_SPATIAL_QUADRANTS",
    "STAGE3_MIN_USABLE_SKY_FRACTION",
    "STAGE3_MIN_VALIDATION_PATCHES",
    "STAGE3_RADIAL_BIC_DELTA_MIN",
    "STAGE3_RADIAL_RESIDUAL_SPAN_RATIO_MAX",
    "STAGE3_SIGNIFICANCE_SIGMA",
    "stage3_static_contract_manifest",
]
