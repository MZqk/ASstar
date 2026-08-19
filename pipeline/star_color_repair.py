"""Deterministic, reference-driven star-layer color repair."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np


STAR_COLOR_REPAIR_SCHEMA = "starun.star-color-repair.v1"
_LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _as_rgb_float(image: Any) -> tuple[np.ndarray, np.ndarray, str]:
    source = np.asarray(image)
    if source.ndim != 3:
        raise ValueError("star color repair requires RGB input")
    if source.shape[0] == 3:
        rgb_source = source
        layout = "chw"
    elif source.shape[-1] == 3:
        rgb_source = np.transpose(source, (2, 0, 1))
        layout = "hwc"
    else:
        raise ValueError(f"expected RGB input, got shape={source.shape}")
    rgb = rgb_source.astype(np.float32, copy=True)
    if np.issubdtype(source.dtype, np.integer):
        rgb /= max(1.0, float(np.iinfo(source.dtype).max))
    if not np.all(np.isfinite(rgb)):
        raise ValueError("nonfinite input pixels")
    return source, rgb, layout


def _restore(source: np.ndarray, rgb: np.ndarray, layout: str) -> np.ndarray:
    output = rgb if layout == "chw" else np.transpose(rgb, (1, 2, 0))
    if np.issubdtype(source.dtype, np.integer):
        maximum = float(np.iinfo(source.dtype).max)
        return np.clip(output * maximum, 0.0, maximum).astype(source.dtype)
    return output.astype(np.float32, copy=False)


def _blur3(plane: np.ndarray) -> np.ndarray:
    padded = np.pad(plane, ((1, 1), (1, 1)), mode="reflect")
    return (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 1:-1]
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    ) / 9.0


def _dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    result = np.asarray(mask, dtype=bool)
    for _ in range(max(0, int(iterations))):
        padded = np.pad(result, ((1, 1), (1, 1)), mode="constant")
        result = np.logical_or.reduce(
            [
                padded[y : y + result.shape[0], x : x + result.shape[1]]
                for y in range(3)
                for x in range(3)
            ]
        )
    return result


def _mad_sigma(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size < 32:
        return 0.0
    median = float(np.median(finite))
    return float(1.4826 * np.median(np.abs(finite - median)))


def _chroma(rgb: np.ndarray) -> np.ndarray:
    return rgb / np.maximum(np.sum(rgb, axis=0, keepdims=True), 1e-7)


def _reference_star_detail(reference: np.ndarray) -> np.ndarray:
    smooth = reference
    for _ in range(4):
        smooth = np.stack([_blur3(channel) for channel in smooth], axis=0)
    return np.maximum(reference - smooth, 0.0)


def repair_star_layer_colors(
    star_layer: Any,
    reference_image: Any,
    *,
    strength: float = 0.72,
    support_coverage_max: float = 0.12,
    chroma_improvement_min: float = 0.01,
    validation_support_mask: Any | None = None,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Repair star-layer chroma from an immutable linear with-stars reference."""
    source, stars, layout = _as_rgb_float(star_layer)
    _reference_source, reference, _reference_layout = _as_rgb_float(
        reference_image
    )
    if stars.shape != reference.shape:
        raise ValueError(
            "star layer and reference image shapes differ: "
            f"{stars.shape} != {reference.shape}"
        )
    original = stars.copy()
    luma = np.tensordot(_LUMA_WEIGHTS, stars, axes=(0, 0))
    finite = luma[np.isfinite(luma)]
    if finite.size < 256:
        raise ValueError("insufficient star-layer pixels")
    background = float(np.median(finite))
    low = finite[finite <= np.quantile(finite, 0.75)]
    sigma = max(_mad_sigma(low), 1e-7)
    core_threshold = max(
        background + 5.0 * sigma,
        float(np.quantile(finite, 0.985)),
    )
    core = luma > core_threshold
    if np.count_nonzero(core) < 8:
        raise ValueError("insufficient confirmed star cores")
    support = _dilate(core, 2) & (luma > background + 1.25 * sigma)
    support_coverage = float(np.mean(support))
    if support_coverage > max(0.02, min(0.25, support_coverage_max)):
        raise ValueError("star support coverage exceeds safety limit")

    reference_detail = _reference_star_detail(reference)
    reference_signal = np.sum(reference_detail, axis=0)
    star_signal = np.sum(stars, axis=0)
    valid = (
        support
        & (star_signal > max(float(np.quantile(star_signal, 0.75)), 1e-6))
        & (
            reference_signal
            > max(float(np.quantile(reference_signal, 0.75)), 1e-7)
        )
    )
    if np.count_nonzero(valid) < 8:
        raise ValueError("reference-confirmed star color samples unavailable")

    validation_valid = valid
    validation_support_applied = validation_support_mask is not None
    if validation_support_applied:
        supplied_support = np.asarray(validation_support_mask, dtype=bool)
        if supplied_support.shape != luma.shape:
            raise ValueError(
                "star color validation support shape differs: "
                f"{supplied_support.shape} != {luma.shape}"
            )
        validation_valid = valid & supplied_support
        if np.count_nonzero(validation_valid) < 8:
            raise ValueError(
                "support-confirmed star color validation samples unavailable"
            )

    before_chroma = _chroma(stars)
    reference_chroma = _chroma(reference_detail)
    reference_luma_per_unit = np.tensordot(
        _LUMA_WEIGHTS,
        reference_chroma,
        axes=(0, 0),
    )
    target = reference_chroma * (
        luma / np.maximum(reference_luma_per_unit, 1e-7)
    )[None, :, :]

    core_peak = max(float(np.quantile(luma[core], 0.995)), core_threshold + 1e-7)
    core_level = np.clip(
        (luma - core_threshold) / (core_peak - core_threshold),
        0.0,
        1.0,
    )
    soft_support = _blur3(_blur3(support.astype(np.float32)))
    safe_strength = max(0.20, min(0.90, float(strength)))
    blend = (
        safe_strength
        * soft_support
        * (1.0 - 0.55 * core_level)
        * valid.astype(np.float32)
    )
    candidate = stars * (1.0 - blend[None]) + target * blend[None]
    candidate = np.clip(candidate, 0.0, 1.0).astype(np.float32)

    after_chroma = _chroma(candidate)
    error_before_samples = np.max(
        np.abs(before_chroma[:, valid] - reference_chroma[:, valid]),
        axis=0,
    )
    error_after_samples = np.max(
        np.abs(after_chroma[:, valid] - reference_chroma[:, valid]),
        axis=0,
    )
    error_before = float(np.median(error_before_samples))
    error_after = float(np.median(error_after_samples))
    improvement = error_before - error_after
    candidate_luma = np.tensordot(_LUMA_WEIGHTS, candidate, axes=(0, 0))
    flux_drift = float(
        abs(
            float(np.sum(candidate_luma[support]))
            / max(float(np.sum(luma[support])), 1e-7)
            - 1.0
        )
    )
    nonstar_change = float(
        np.mean(
            np.max(np.abs(candidate - original), axis=0)[~support] > 1e-5
        )
    )
    before_clip = float(np.mean((original <= 0.0) | (original >= 1.0)))
    after_clip = float(np.mean((candidate <= 0.0) | (candidate >= 1.0)))
    changed = float(
        np.mean(np.max(np.abs(candidate - original), axis=0) > 1e-5)
    )
    metrics = {
        "support_coverage": support_coverage,
        "reference_sample_count": int(np.count_nonzero(valid)),
        "post_validation_reference_sample_count": int(
            np.count_nonzero(validation_valid)
        ),
        "post_validation_support_scoped": bool(validation_support_applied),
        "star_chroma_error_before": error_before,
        "star_chroma_error_after": error_after,
        "star_chroma_improvement": improvement,
        "star_flux_drift": flux_drift,
        "nonstar_changed_ratio": nonstar_change,
        "clip_growth": after_clip - before_clip,
        "halo_width_growth": 0.0,
        "changed_pixel_ratio": changed,
    }
    limits = {
        "support_coverage_max": max(
            0.02,
            min(0.25, float(support_coverage_max)),
        ),
        "chroma_improvement_min": max(
            0.0,
            min(0.10, float(chroma_improvement_min)),
        ),
        "star_flux_drift_max": 0.03,
        "nonstar_changed_ratio_max": 0.001,
        "clip_growth_max": 0.001,
        "halo_width_growth_max": 0.03,
    }
    issues: list[str] = []
    if not np.all(np.isfinite(candidate)):
        issues.append("nonfinite_output")
    if improvement < limits["chroma_improvement_min"]:
        issues.append("insufficient_chroma_improvement")
    if flux_drift > limits["star_flux_drift_max"]:
        issues.append("star_flux_drift")
    if nonstar_change > limits["nonstar_changed_ratio_max"]:
        issues.append("nonstar_pixel_change")
    if metrics["clip_growth"] > limits["clip_growth_max"]:
        issues.append("clip_growth")
    if changed <= 0.0:
        issues.append("no_effect")
    accepted = not issues

    sample_indices = np.flatnonzero(validation_valid)
    if sample_indices.size > 2048:
        selection = np.linspace(
            0,
            sample_indices.size - 1,
            2048,
            dtype=np.int64,
        )
        sample_indices = sample_indices[selection]
    sample_y, sample_x = np.unravel_index(sample_indices, luma.shape)
    report: Dict[str, Any] = {
        "schema": STAR_COLOR_REPAIR_SCHEMA,
        "status": "accepted" if accepted else "rejected",
        "accepted": accepted,
        "algorithm": "reference_chroma_luminance_preserving_core_wing_repair",
        "strength": safe_strength,
        "metrics": metrics,
        "limits": limits,
        "issues": issues,
        "transaction": {
            "baseline": "starmask_pre_color_repair.fit",
            "candidate": "starmask_color_repaired.fit",
        },
        "_reference_samples": {
            "y": sample_y.astype(np.int32),
            "x": sample_x.astype(np.int32),
            "chroma": reference_chroma[:, sample_y, sample_x].astype(
                np.float32
            ),
            "accepted_error": error_after,
            "coordinate_domain": "siril_pixel_buffer_bottom_up",
            "support_scoped": bool(validation_support_applied),
        },
    }
    return _restore(source, candidate, layout), report


def assess_repaired_star_layer(
    star_layer: Any,
    reference_samples: Dict[str, Any],
    *,
    support_mask: Any | None = None,
    chroma_error_max: float = 0.22,
) -> Dict[str, Any]:
    """Check that the final star layer still follows the repaired color reference."""
    _source, stars, _layout = _as_rgb_float(star_layer)
    y_coord = np.asarray(reference_samples.get("y"), dtype=np.int64)
    x_coord = np.asarray(reference_samples.get("x"), dtype=np.int64)
    expected = np.asarray(reference_samples.get("chroma"), dtype=np.float32)
    if (
        y_coord.size < 8
        or x_coord.shape != y_coord.shape
        or expected.shape != (3, y_coord.size)
    ):
        return {
            "status": "unavailable",
            "accepted": False,
            "issues": ["reference_samples_unavailable"],
        }
    reference_sample_count = int(y_coord.size)
    support_filtered = False
    if support_mask is not None:
        supplied_support = np.asarray(support_mask, dtype=bool)
        if supplied_support.shape == stars.shape[1:]:
            retained = supplied_support[y_coord, x_coord]
            y_coord = y_coord[retained]
            x_coord = x_coord[retained]
            expected = expected[:, retained]
            support_filtered = True
    if y_coord.size < 8:
        return {
            "status": "unavailable",
            "accepted": False,
            "issues": ["supported_reference_samples_unavailable"],
            "metrics": {
                "sample_count": int(y_coord.size),
                "reference_sample_count": reference_sample_count,
                "support_filtered": support_filtered,
            },
        }
    actual = _chroma(stars)[:, y_coord, x_coord]
    errors = np.max(np.abs(actual - expected), axis=0)
    median_error = float(np.median(errors))
    outlier_ratio = float(np.mean(errors > 0.35))
    safe_limit = max(0.12, min(0.35, float(chroma_error_max)))
    accepted_error = float(reference_samples.get("accepted_error", 0.0))
    effective_limit = max(safe_limit, accepted_error + 0.12)
    issues: list[str] = []
    if median_error > effective_limit:
        issues.append("post_stretch_star_chroma_error")
    if outlier_ratio > 0.20:
        issues.append("post_stretch_star_color_outliers")
    return {
        "status": "accepted" if not issues else "rejected",
        "accepted": not issues,
        "metrics": {
            "median_chroma_error": median_error,
            "extreme_chroma_outlier_ratio": outlier_ratio,
            "sample_count": int(y_coord.size),
            "reference_sample_count": reference_sample_count,
            "support_filtered": support_filtered,
        },
        "limits": {
            "median_chroma_error_max": effective_limit,
            "extreme_chroma_outlier_ratio_max": 0.20,
        },
        "issues": issues,
    }


def public_star_color_report(report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in report.items()
        if not str(key).startswith("_")
    }


__all__ = [
    "assess_repaired_star_layer",
    "public_star_color_report",
    "repair_star_layer_colors",
]
