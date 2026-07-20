"""Target-local diagnostics for Stage 7 starless stretch candidates."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from image_metrics import _box_blur_gray, _to_rgb_float_fullres


CORE_PROTECT_TARGETS = {
    "bright_emission_reflection_nebula",
    "large_galaxy",
    "small_galaxy",
}
FAINT_SIGNAL_TARGETS = {
    "bright_emission_reflection_nebula",
    "emission_nebula_widefield",
    "large_galaxy",
    "small_galaxy",
    "dark_nebula_low_contrast",
}


def _bounded(value: Any, default: float, lower: float, upper: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(lower, min(upper, parsed))


def assess_target_local_stretch(
    baseline: np.ndarray,
    candidate: np.ndarray,
    target_type: str,
    cfg: Any,
) -> Dict[str, Any]:
    """Measure core, faint-structure and dark-lane regions derived from linear data."""
    if not bool(getattr(cfg, "stage7_target_local_metrics_enabled", True)):
        return {
            "status": "disabled",
            "accepted": True,
            "issues": [],
            "risk_score": 0.0,
            "metrics": {},
        }
    try:
        source_rgb = _to_rgb_float_fullres(np.asarray(baseline))
        candidate_rgb = _to_rgb_float_fullres(np.asarray(candidate))
        if source_rgb.shape != candidate_rgb.shape:
            raise ValueError(
                f"shape mismatch: baseline={source_rgb.shape}, candidate={candidate_rgb.shape}"
            )
        source_gray = (
            0.2126 * source_rgb[0]
            + 0.7152 * source_rgb[1]
            + 0.0722 * source_rgb[2]
        ).astype(np.float32)
        candidate_gray = (
            0.2126 * candidate_rgb[0]
            + 0.7152 * candidate_rgb[1]
            + 0.0722 * candidate_rgb[2]
        ).astype(np.float32)
        candidate_peak = np.max(candidate_rgb[:3], axis=0)
        broad = source_gray.copy()
        for _ in range(4):
            broad = _box_blur_gray(broad)
        q35, q55, q90, q99 = np.percentile(broad, [35.0, 55.0, 90.0, 99.0])
        background_mask = broad <= q35
        dark_mask = (broad > q35) & (broad <= q55)
        faint_mask = (broad > q55) & (broad <= q90)
        core_mask = broad > q99
        if min(
            int(np.count_nonzero(background_mask)),
            int(np.count_nonzero(dark_mask)),
            int(np.count_nonzero(faint_mask)),
            int(np.count_nonzero(core_mask)),
        ) < 16:
            raise ValueError("target-local masks contain too few samples")

        background_values = candidate_gray[background_mask]
        dark_values = candidate_gray[dark_mask]
        faint_values = candidate_gray[faint_mask]
        core_values = candidate_gray[core_mask]
        background_median = float(np.median(background_values))
        background_std = max(float(np.std(background_values)), 1e-6)
        dark_median = float(np.median(dark_values))
        faint_median = float(np.median(faint_values))
        faint_contrast = faint_median - background_median
        dark_separation = faint_median - dark_median
        core_clip_ratio = float(np.mean(candidate_peak[core_mask] >= 0.995))
        core_p99 = float(np.percentile(core_values, 99.0))
        metrics = {
            "background_median": background_median,
            "background_std": background_std,
            "faint_median": faint_median,
            "faint_contrast": faint_contrast,
            "faint_snr": faint_contrast / background_std,
            "dark_median": dark_median,
            "dark_separation": dark_separation,
            "core_median": float(np.median(core_values)),
            "core_p99": core_p99,
            "core_clip_ratio": core_clip_ratio,
            "background_coverage": float(np.mean(background_mask)),
            "faint_coverage": float(np.mean(faint_mask)),
            "core_coverage": float(np.mean(core_mask)),
        }

        normalized_target = str(target_type or "generic_low_snr_safe").strip().lower()
        issues = []
        risk_score = 0.0
        if normalized_target in CORE_PROTECT_TARGETS:
            core_clip_max = _bounded(
                getattr(cfg, "stage7_local_core_clip_ratio_max", 0.12),
                0.12,
                0.01,
                0.30,
            )
            if core_clip_ratio > core_clip_max:
                issues.append(
                    f"local_core_clip_ratio {core_clip_ratio:.4f}>{core_clip_max:.4f}"
                )
            risk_score += core_clip_ratio * 8.0
            risk_score += max(0.0, core_p99 - 0.985) * 30.0

        if normalized_target in FAINT_SIGNAL_TARGETS:
            faint_snr_min = _bounded(
                getattr(cfg, "stage7_local_faint_snr_min", 0.25),
                0.25,
                0.0,
                2.0,
            )
            if metrics["faint_snr"] < faint_snr_min:
                issues.append(
                    f"local_faint_snr {metrics['faint_snr']:.4f}<{faint_snr_min:.4f}"
                )
            risk_score += max(0.0, faint_snr_min - metrics["faint_snr"]) * 2.0

        if normalized_target == "dark_nebula_low_contrast":
            separation_min = _bounded(
                getattr(cfg, "stage7_local_dark_separation_min", 0.001),
                0.001,
                0.0,
                0.02,
            )
            if dark_separation < separation_min:
                issues.append(
                    f"local_dark_separation {dark_separation:.5f}<{separation_min:.5f}"
                )
            risk_score += max(0.0, separation_min - dark_separation) * 100.0

        return {
            "status": "ok" if not issues else "rejected",
            "accepted": not issues,
            "issues": issues,
            "risk_score": float(risk_score),
            "metrics": metrics,
            "target_type": normalized_target,
        }
    except (IndexError, TypeError, ValueError, FloatingPointError) as error:
        return {
            "status": "unavailable",
            "accepted": True,
            "issues": [],
            "risk_score": 0.0,
            "metrics": {},
            "reason": str(error),
        }
