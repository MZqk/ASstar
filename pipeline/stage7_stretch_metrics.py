"""Target-local diagnostics for Stage 7 starless stretch candidates."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from image_metrics import (
    _box_blur_gray,
    _to_rgb_float_fullres,
    _to_rgb_float_image,
)


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


def _rank_normalized_gray(gray: np.ndarray) -> np.ndarray:
    """Map luminance to approximate percentile ranks.

    A global monotonic stretch preserves these ranks, so local changes measured
    after this mapping describe structural edits instead of simple brightness
    amplification.
    """
    values = np.asarray(gray, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size < 64:
        raise ValueError("too few finite pixels for rank normalization")
    if finite.size > 1_000_000:
        sample_step = int(np.ceil(finite.size / 1_000_000.0))
        finite = finite[::sample_step]
    levels = np.linspace(0.0, 1.0, 129, dtype=np.float32)
    anchors = np.quantile(finite, levels).astype(np.float32)
    unique_anchors, unique_indices = np.unique(anchors, return_index=True)
    if unique_anchors.size < 4:
        raise ValueError("insufficient luminance range for rank normalization")
    unique_levels = levels[unique_indices]
    ranked = np.interp(
        values.reshape(-1),
        unique_anchors,
        unique_levels,
    ).reshape(values.shape)
    return np.asarray(ranked, dtype=np.float32)


def assess_starless_structure_growth(
    baseline: np.ndarray,
    candidate: np.ndarray,
    starmask: Optional[np.ndarray],
    cfg: Any,
) -> Dict[str, Any]:
    """Gate star-like structural growth in a stretched Starless image.

    The comparison is performed on luminance percentile-rank maps and only in
    regions identified by the Stage 6 starmask. This makes the diagnostic
    insensitive to an ordinary global monotonic stretch while retaining
    sensitivity to newly enlarged residuals or halos around removed stars.
    """
    if not bool(getattr(cfg, "stage7_starless_structure_gate_enabled", True)):
        return {
            "status": "disabled",
            "accepted": True,
            "issues": [],
            "risk_score": 0.0,
            "metrics": {},
        }
    if starmask is None:
        return {
            "status": "unavailable",
            "accepted": True,
            "issues": [],
            "risk_score": 0.0,
            "metrics": {},
            "reason": "starmask unavailable",
        }

    try:
        source_rgb = _to_rgb_float_image(np.asarray(baseline), max_side=1024)
        candidate_rgb = _to_rgb_float_image(np.asarray(candidate), max_side=1024)
        starmask_rgb = _to_rgb_float_image(np.asarray(starmask), max_side=1024)
        if source_rgb.shape != candidate_rgb.shape or source_rgb.shape != starmask_rgb.shape:
            raise ValueError(
                "shape mismatch: "
                f"baseline={source_rgb.shape}, candidate={candidate_rgb.shape}, "
                f"starmask={starmask_rgb.shape}"
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
        starmask_gray = (
            0.2126 * starmask_rgb[0]
            + 0.7152 * starmask_rgb[1]
            + 0.0722 * starmask_rgb[2]
        ).astype(np.float32)

        mask_floor = float(np.quantile(starmask_gray, 0.50))
        mask_signal = np.clip(starmask_gray - mask_floor, 0.0, None)
        positive = mask_signal[mask_signal > 0.0]
        if positive.size < 8:
            raise ValueError("starmask contains too little positive signal")
        mask_scale = float(np.quantile(positive, 0.995))
        if not np.isfinite(mask_scale) or mask_scale <= 1e-7:
            raise ValueError("starmask signal scale is invalid")
        mask_weight = np.clip(mask_signal / mask_scale, 0.0, 1.0)
        seed = mask_weight >= 0.10
        if int(np.count_nonzero(seed)) < 4:
            seed_threshold = float(np.quantile(mask_weight[mask_weight > 0.0], 0.75))
            seed = mask_weight >= max(seed_threshold, 1e-4)

        halo_weight = seed.astype(np.float32)
        for _ in range(3):
            halo_weight = _box_blur_gray(halo_weight)
        support_weight = np.maximum(mask_weight, np.clip(halo_weight * 4.0, 0.0, 1.0))
        support = support_weight > 0.02
        if int(np.count_nonzero(support)) < 16:
            raise ValueError("starmask-local support contains too few samples")

        source_rank = _rank_normalized_gray(source_gray)
        candidate_rank = _rank_normalized_gray(candidate_gray)
        rank_delta = candidate_rank - source_rank
        absolute_drift_p95 = float(np.quantile(np.abs(rank_delta[support]), 0.95))
        brightening_p95 = float(
            np.quantile(np.clip(rank_delta[support], 0.0, None), 0.95)
        )

        source_detail = np.abs(source_rank - _box_blur_gray(source_rank))
        candidate_detail = np.abs(candidate_rank - _box_blur_gray(candidate_rank))
        weights = support_weight[support]
        weight_sum = max(float(np.sum(weights)), 1e-6)
        source_detail_level = float(np.sum(source_detail[support] * weights) / weight_sum)
        candidate_detail_level = float(
            np.sum(candidate_detail[support] * weights) / weight_sum
        )
        detail_delta = candidate_detail_level - source_detail_level
        detail_ratio = candidate_detail_level / max(source_detail_level, 0.002)

        drift_max = _bounded(
            getattr(cfg, "stage7_starless_masked_rank_drift_p95_max", 0.18),
            0.18,
            0.02,
            0.50,
        )
        detail_ratio_max = _bounded(
            getattr(
                cfg,
                "stage7_starless_halo_detail_growth_ratio_max",
                1.60,
            ),
            1.60,
            1.05,
            4.0,
        )
        detail_delta_min = _bounded(
            getattr(cfg, "stage7_starless_halo_detail_delta_min", 0.010),
            0.010,
            0.001,
            0.10,
        )
        issues = []
        if absolute_drift_p95 > drift_max:
            issues.append(
                "starless_masked_rank_drift_p95 "
                f"{absolute_drift_p95:.3f}>{drift_max:.3f}"
            )
        if detail_ratio > detail_ratio_max and detail_delta > detail_delta_min:
            issues.append(
                "starless_halo_detail_growth "
                f"{detail_ratio:.3f}>{detail_ratio_max:.3f} "
                f"(delta={detail_delta:.4f}>{detail_delta_min:.4f})"
            )

        risk_score = absolute_drift_p95 / max(drift_max, 1e-6) * 0.5
        if detail_delta > detail_delta_min:
            risk_score += max(0.0, detail_ratio - 1.0)
        metrics = {
            "masked_rank_drift_p95": absolute_drift_p95,
            "masked_rank_brightening_p95": brightening_p95,
            "source_halo_detail_level": source_detail_level,
            "candidate_halo_detail_level": candidate_detail_level,
            "halo_detail_growth_ratio": detail_ratio,
            "halo_detail_delta": detail_delta,
            "support_coverage": float(np.mean(support)),
            "starmask_seed_coverage": float(np.mean(seed)),
            "rank_drift_p95_max": drift_max,
            "halo_detail_growth_ratio_max": detail_ratio_max,
            "halo_detail_delta_min": detail_delta_min,
        }
        return {
            "status": "ok" if not issues else "rejected",
            "accepted": not issues,
            "issues": issues,
            "risk_score": float(risk_score),
            "metrics": metrics,
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


def calibrate_adaptive_quantile_stretch(
    image: np.ndarray,
    adaptation: Dict[str, Any],
    cfg: Any,
) -> Dict[str, Any]:
    """Build a bounded linked curve from source quantiles to preview targets."""
    if not bool(getattr(cfg, "stage7_quantile_fallback_enabled", True)):
        return {
            "status": "disabled",
            "reason": "adaptive quantile fallback disabled by config",
        }
    try:
        preview_calibration = adaptation.get("preview_calibration") or {}
        candidate_a = preview_calibration.get("candidate_a") or {}
        target_p50 = float(candidate_a.get("target_p50", 0.0) or 0.0)
        target_p99 = float(candidate_a.get("target_p99", 0.0) or 0.0)
        if (
            not np.isfinite(target_p50)
            or not np.isfinite(target_p99)
            or target_p50 <= 0.0
            or target_p99 <= target_p50
        ):
            raise ValueError("preview P50/P99 targets unavailable")

        rgb = _to_rgb_float_fullres(np.asarray(image))
        finite = rgb[np.isfinite(rgb)]
        if finite.size < 64:
            raise ValueError("too few finite source pixels")
        if finite.size > 2_000_000:
            sample_step = int(np.ceil(finite.size / 2_000_000.0))
            finite = finite[::sample_step]
        input_percentiles = np.asarray(
            [0.1, 1.0, 50.0, 90.0, 99.0, 99.9, 100.0],
            dtype=np.float32,
        )
        input_values = np.percentile(finite, input_percentiles).astype(np.float64)

        shadow_low = min(max(target_p50 * 0.08, 0.0005), 0.008)
        shadow_high = min(max(target_p50 * 0.22, shadow_low + 0.001), 0.018)
        target_p90 = target_p50 + 0.55 * (target_p99 - target_p50)
        peak_target = min(0.970, max(target_p99 + 0.040, target_p99 * 1.06))
        maximum_target = min(0.985, max(peak_target + 0.010, target_p99 + 0.080))
        output_values = np.asarray(
            [
                shadow_low,
                shadow_high,
                target_p50,
                target_p90,
                target_p99,
                peak_target,
                maximum_target,
            ],
            dtype=np.float64,
        )
        output_values = np.maximum.accumulate(output_values)

        # Flat shadows can produce duplicate input quantiles. Keep the later
        # (brighter) target at an identical source value so the P50 contract is
        # not silently lost.
        unique_inputs = []
        unique_outputs = []
        unique_percentiles = []
        for percentile, source_value, target_value in zip(
            input_percentiles,
            input_values,
            output_values,
        ):
            source_value = float(source_value)
            target_value = float(target_value)
            if not np.isfinite(source_value) or not np.isfinite(target_value):
                continue
            if unique_inputs and source_value <= unique_inputs[-1] + 1e-8:
                unique_inputs[-1] = max(unique_inputs[-1], source_value)
                unique_outputs[-1] = max(unique_outputs[-1], target_value)
                unique_percentiles[-1] = float(percentile)
                continue
            unique_inputs.append(source_value)
            unique_outputs.append(target_value)
            unique_percentiles.append(float(percentile))
        if len(unique_inputs) < 4 or unique_inputs[-1] <= unique_inputs[0] + 1e-6:
            raise ValueError("source quantiles do not define a usable curve")
        if any(
            later <= earlier
            for earlier, later in zip(unique_inputs, unique_inputs[1:])
        ):
            raise ValueError("source quantile anchors are not strictly increasing")

        return {
            "status": "ok",
            "method": "linked_piecewise_linear_quantile_curve",
            "input_percentiles": unique_percentiles,
            "input_anchors": unique_inputs,
            "output_anchors": unique_outputs,
            "target_p50": target_p50,
            "target_p99": target_p99,
            "brightness_ordering_preserved": True,
            "channel_curve_linked": True,
            "source": "stage7_preview_ref candidate_a P50/P99",
        }
    except (IndexError, TypeError, ValueError, FloatingPointError) as error:
        return {
            "status": "unavailable",
            "reason": str(error),
        }


def apply_adaptive_quantile_stretch(
    image: np.ndarray,
    calibration: Dict[str, Any],
) -> np.ndarray:
    """Apply one shared monotonic curve to all RGB samples."""
    source = np.asarray(image)
    rgb = _to_rgb_float_fullres(source)
    inputs = np.asarray(calibration.get("input_anchors"), dtype=np.float64)
    outputs = np.asarray(calibration.get("output_anchors"), dtype=np.float64)
    if (
        str(calibration.get("status") or "") != "ok"
        or inputs.ndim != 1
        or outputs.ndim != 1
        or inputs.size < 4
        or inputs.size != outputs.size
        or not np.all(np.isfinite(inputs))
        or not np.all(np.isfinite(outputs))
        or np.any(np.diff(inputs) <= 0.0)
        or np.any(np.diff(outputs) < 0.0)
    ):
        raise ValueError("adaptive quantile calibration is invalid")
    mapped = np.interp(
        rgb.reshape(-1),
        inputs,
        outputs,
        left=float(outputs[0]),
        right=float(outputs[-1]),
    ).reshape(rgb.shape)
    return np.clip(mapped, 0.0, 0.995).astype(np.float32, copy=False)


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
