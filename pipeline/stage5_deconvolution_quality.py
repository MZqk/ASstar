"""Pure Stage 5 PSF and deconvolution star-quality diagnostics.

The reports in this module deliberately separate the small set of structural
PSF failures that may skip Siril RL from the enforced, fixed-catalog local-star
acceptance guard.  No function mutates Siril state.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


PSF_QUALITY_SCHEMA = "starun.stage5-psf-quality.v1"
STAR_REFERENCE_SCHEMA = "starun.stage5-star-reference.v1"
LOCAL_STAR_GUARD_SCHEMA = "starun.stage5-local-star-guard.v2"
TARGET_STRUCTURE_MASK_THRESHOLD = 0.12
TARGET_STRUCTURE_OVERLAP_MAX = 0.25
LOCAL_STAR_MIN_COUNT = 5
LOCAL_STAR_EDGE_FWHM_MIN = 5.0
LOCAL_STAR_NEIGHBOR_FWHM_MIN = 6.0
LOCAL_STAR_SIGNAL_SNR_MIN = 5.0
LOCAL_STAR_RING_SIGMA_MIN = 3.0
LOCAL_STAR_SIGNAL_ABS_MIN = 1e-6


def _finite_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _star_value(star: Any, names: Sequence[str]) -> Any:
    for name in names:
        if isinstance(star, dict) and name in star:
            return star.get(name)
        if hasattr(star, name):
            return getattr(star, name)
    return None


def _star_bool(star: Any, names: Sequence[str]) -> bool:
    value = _star_value(star, names)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _summary(values: Iterable[Any]) -> Dict[str, Any]:
    finite = np.asarray(
        [value for raw in values if (value := _finite_float(raw)) is not None],
        dtype=np.float64,
    )
    if finite.size == 0:
        return {"status": "unavailable", "count": 0}
    median = float(np.median(finite))
    return {
        "status": "available",
        "count": int(finite.size),
        "median": median,
        "mad": float(np.median(np.abs(finite - median))),
        "p10": float(np.percentile(finite, 10.0)),
        "p90": float(np.percentile(finite, 90.0)),
        "p05": float(np.percentile(finite, 5.0)),
        "p95": float(np.percentile(finite, 95.0)),
    }


def _spatial_shape(image_or_shape: Any) -> Tuple[int, int]:
    if isinstance(image_or_shape, tuple) and len(image_or_shape) == 2:
        return int(image_or_shape[0]), int(image_or_shape[1])
    shape = tuple(int(value) for value in np.asarray(image_or_shape).shape)
    if len(shape) == 2:
        return shape
    if len(shape) != 3:
        raise ValueError(f"unsupported Stage5 image shape: {shape}")
    if shape[0] in (1, 3) and shape[-1] not in (1, 3):
        return shape[1], shape[2]
    if shape[-1] in (1, 3):
        return shape[0], shape[1]
    if shape[0] >= 3:
        return shape[1], shape[2]
    raise ValueError(f"unsupported Stage5 image shape: {shape}")


def _to_chw_float(image: Any) -> np.ndarray:
    source = np.asarray(image)
    if source.size == 0:
        raise ValueError("empty Stage5 image")
    if not np.all(np.isfinite(source)):
        raise ValueError("non-finite Stage5 image")
    if source.ndim == 2:
        chw = np.repeat(source[None, :, :], 3, axis=0)
    elif source.ndim == 3:
        if source.shape[0] == 1 and source.shape[-1] not in (1, 3):
            chw = np.repeat(source, 3, axis=0)
        elif source.shape[0] == 3 and source.shape[-1] not in (1, 3):
            chw = source
        elif source.shape[-1] == 1:
            chw = np.repeat(np.moveaxis(source, -1, 0), 3, axis=0)
        elif source.shape[-1] >= 3:
            chw = np.moveaxis(source[..., :3], -1, 0)
        elif source.shape[0] >= 3:
            chw = source[:3]
        else:
            raise ValueError(f"unsupported Stage5 image shape: {source.shape}")
    else:
        raise ValueError(f"unsupported Stage5 image ndim: {source.ndim}")

    if np.issubdtype(chw.dtype, np.unsignedinteger):
        scale = float(np.iinfo(chw.dtype).max)
        return chw.astype(np.float32) / scale
    if not np.issubdtype(chw.dtype, np.floating):
        raise ValueError(f"unsupported Stage5 image dtype: {chw.dtype}")
    return chw.astype(np.float32, copy=False)


def build_target_structure_mask(
    masks: Optional[Dict[str, Any]],
    image_or_shape: Any,
) -> Optional[np.ndarray]:
    """Union existing Stage 8 signal-exclusion masks at the frozen threshold."""
    if not isinstance(masks, dict):
        return None
    expected_shape = _spatial_shape(image_or_shape)
    union = np.zeros(expected_shape, dtype=bool)
    available = False
    for key in (
        "core_mask",
        "nebula_mask",
        "faint_nebula_mask",
        "galaxy_signal_mask",
    ):
        value = masks.get(key)
        if value is None:
            continue
        array = np.asarray(value, dtype=np.float32)
        if array.shape != expected_shape or not np.all(np.isfinite(array)):
            continue
        union |= array > TARGET_STRUCTURE_MASK_THRESHOLD
        available = True
    return union if available else None


def _local_mask_fraction(
    mask: Optional[np.ndarray],
    x: Optional[float],
    y: Optional[float],
    radius: Optional[float],
) -> Optional[float]:
    if mask is None or x is None or y is None or radius is None or radius <= 0.0:
        return None
    height, width = mask.shape
    x0 = max(0, int(math.floor(x - radius)))
    x1 = min(width, int(math.ceil(x + radius)) + 1)
    y0 = max(0, int(math.floor(y - radius)))
    y1 = min(height, int(math.ceil(y + radius)) + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    yy, xx = np.ogrid[y0:y1, x0:x1]
    disk = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
    if not bool(np.any(disk)):
        return None
    return float(np.mean(np.asarray(mask[y0:y1, x0:x1], dtype=bool)[disk]))


def _unverified_report(
    *,
    source_checkpoint: str,
    catalog_role: str,
    reason: str,
    findstar_succeeded: bool,
) -> Dict[str, Any]:
    return {
        "schema": PSF_QUALITY_SCHEMA,
        "status": "unavailable",
        "source_checkpoint": source_checkpoint,
        "catalog_role": catalog_role,
        "findstar_succeeded": bool(findstar_succeeded),
        "decision": "proceed_unverified",
        "hard_skip_rl": False,
        "reason_code": "star_api_unavailable",
        "reason": str(reason),
        "structural_checks_enforced": True,
        "shadow_thresholds_enforced": False,
        "stars": [],
    }


def unavailable_psf_quality_report(
    reason: str,
    *,
    source_checkpoint: str = "stage5_input_linear.fit",
    catalog_role: str = "rl_psf",
    findstar_succeeded: bool = False,
) -> Dict[str, Any]:
    """Report API absence/errors without changing the existing RL behavior."""
    return _unverified_report(
        source_checkpoint=source_checkpoint,
        catalog_role=catalog_role,
        reason=reason,
        findstar_succeeded=findstar_succeeded,
    )


def not_run_psf_quality_report() -> Dict[str, Any]:
    return {
        "schema": PSF_QUALITY_SCHEMA,
        "status": "not_run",
        "source_checkpoint": "stage5_input_linear.fit",
        "catalog_role": "rl_psf",
        "decision": "not_run",
        "hard_skip_rl": False,
        "reason_code": "rl_not_attempted",
        "structural_checks_enforced": True,
        "shadow_thresholds_enforced": False,
        "stars": [],
    }


def build_psf_quality_report(
    stars: Optional[Sequence[Any]],
    image_or_shape: Any,
    *,
    target_structure_mask: Optional[np.ndarray] = None,
    source_checkpoint: str = "stage5_input_linear.fit",
    catalog_role: str = "rl_psf",
    findstar_succeeded: bool = True,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Convert Siril PSFStar objects into a stable audit and fixed catalog."""
    height, width = _spatial_shape(image_or_shape)
    if target_structure_mask is not None:
        target_structure_mask = np.asarray(target_structure_mask, dtype=bool)
        if target_structure_mask.shape != (height, width):
            target_structure_mask = None
    source_stars = list(stars or [])
    parsed: List[Dict[str, Any]] = []
    for index, star in enumerate(source_stars):
        x = _finite_float(_star_value(star, ("xpos", "x", "x0")))
        y = _finite_float(_star_value(star, ("ypos", "y", "y0")))
        fwhm_x = _finite_float(_star_value(star, ("fwhmx", "fwhm_x")))
        fwhm_y = _finite_float(_star_value(star, ("fwhmy", "fwhm_y")))
        geometry_valid = bool(
            x is not None
            and y is not None
            and fwhm_x is not None
            and fwhm_y is not None
            and fwhm_x > 0.0
            and fwhm_y > 0.0
        )
        fwhm = (
            float(math.sqrt(fwhm_x * fwhm_y))
            if geometry_valid and fwhm_x is not None and fwhm_y is not None
            else None
        )
        ellipticity = (
            1.0 - min(fwhm_x, fwhm_y) / max(fwhm_x, fwhm_y)
            if geometry_valid and fwhm_x is not None and fwhm_y is not None
            else None
        )
        edge_distance = (
            float(min(x, y, width - 1.0 - x, height - 1.0 - y))
            if x is not None and y is not None
            else None
        )
        overlap = _local_mask_fraction(
            target_structure_mask,
            x,
            y,
            2.0 * fwhm if fwhm is not None else None,
        )
        parsed.append(
            {
                "index": int(index),
                "name": _star_value(star, ("star_name", "name")),
                "x": x,
                "y": y,
                "amplitude": _finite_float(_star_value(star, ("A", "amplitude"))),
                "background": _finite_float(_star_value(star, ("B", "background"))),
                "saturation_level": _finite_float(_star_value(star, ("sat",))),
                "saturated": _star_bool(
                    star,
                    ("has_saturated", "saturated", "is_saturated"),
                ),
                "fwhm_x": fwhm_x,
                "fwhm_y": fwhm_y,
                "fwhm_geometry": fwhm,
                "ellipticity": ellipticity,
                "rmse": _finite_float(_star_value(star, ("rmse", "RMSE"))),
                "edge_distance": edge_distance,
                "nearest_neighbor_distance": None,
                "nearest_neighbor_fwhm": None,
                "target_structure_overlap": overlap,
                "target_structure_overlap_threshold": (
                    TARGET_STRUCTURE_OVERLAP_MAX
                ),
                "geometry_valid": geometry_valid,
            }
        )

    coordinate_indices = [
        index
        for index, star in enumerate(parsed)
        if star["x"] is not None and star["y"] is not None
    ]
    if len(coordinate_indices) >= 2:
        coords = np.asarray(
            [[parsed[index]["x"], parsed[index]["y"]] for index in coordinate_indices],
            dtype=np.float64,
        )
        distances = np.sqrt(
            np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2)
        )
        np.fill_diagonal(distances, np.inf)
        nearest = np.min(distances, axis=1)
        for local_index, star_index in enumerate(coordinate_indices):
            distance = float(nearest[local_index])
            parsed[star_index]["nearest_neighbor_distance"] = distance
            fwhm = parsed[star_index]["fwhm_geometry"]
            if fwhm is not None and fwhm > 0.0:
                parsed[star_index]["nearest_neighbor_fwhm"] = distance / fwhm

    for star in parsed:
        reasons: List[str] = []
        if star["saturated"]:
            reasons.append("saturated")
        if not star["geometry_valid"]:
            reasons.append("invalid_geometry")
        fwhm = star["fwhm_geometry"]
        edge = star["edge_distance"]
        nearest_ratio = star["nearest_neighbor_fwhm"]
        overlap = star["target_structure_overlap"]
        if fwhm is not None and edge is not None and edge < LOCAL_STAR_EDGE_FWHM_MIN * fwhm:
            reasons.append("edge_distance_lt_5_fwhm")
        if nearest_ratio is not None and nearest_ratio < LOCAL_STAR_NEIGHBOR_FWHM_MIN:
            reasons.append("nearest_neighbor_lt_6_fwhm")
        if overlap is not None and overlap > TARGET_STRUCTURE_OVERLAP_MAX:
            reasons.append("target_structure_overlap_gt_0_25")
        star["exclusion_reasons"] = reasons
        star["usable_for_psf_summary"] = bool(
            star["geometry_valid"] and not star["saturated"]
        )
        star["isolated"] = bool(
            star["usable_for_psf_summary"]
            and nearest_ratio is not None
            and nearest_ratio >= LOCAL_STAR_NEIGHBOR_FWHM_MIN
        )
        star["eligible_for_local_guard"] = not reasons

    total_count = len(parsed)
    saturated_count = sum(bool(star["saturated"]) for star in parsed)
    valid_geometry_count = sum(bool(star["geometry_valid"]) for star in parsed)
    usable = [star for star in parsed if star["usable_for_psf_summary"]]
    isolated_count = sum(bool(star["isolated"]) for star in parsed)
    overlap_known = [
        star for star in usable if star["target_structure_overlap"] is not None
    ]
    overlap_count = sum(
        float(star["target_structure_overlap"]) > TARGET_STRUCTURE_OVERLAP_MAX
        for star in overlap_known
    )
    fwhm_summary = _summary(star["fwhm_geometry"] for star in usable)
    ellipticity_summary = _summary(star["ellipticity"] for star in usable)
    fwhm_relative_mad = None
    if fwhm_summary.get("status") == "available":
        median = float(fwhm_summary["median"])
        if median > 0.0:
            fwhm_relative_mad = float(fwhm_summary["mad"]) / median
    saturated_ratio = saturated_count / total_count if total_count else None
    overlap_ratio = overlap_count / len(overlap_known) if overlap_known else None

    hard_reason = ""
    if findstar_succeeded and total_count == 0:
        hard_reason = "empty_star_catalog"
    elif total_count > 0 and saturated_count == total_count:
        hard_reason = "all_stars_saturated"
    elif total_count > 0 and valid_geometry_count == 0:
        hard_reason = "no_finite_positive_fwhm_coordinates"
    decision = "skip_rl" if hard_reason else "proceed"

    def shadow_check(
        name: str,
        observed: Optional[float],
        *,
        operator: str,
        limit: float,
    ) -> Dict[str, Any]:
        if observed is None:
            return {
                "name": name,
                "status": "unavailable",
                "observed": None,
                "operator": operator,
                "limit": limit,
                "passed": None,
            }
        passed = observed >= limit if operator == ">=" else observed <= limit
        return {
            "name": name,
            "status": "available",
            "observed": float(observed),
            "operator": operator,
            "limit": float(limit),
            "passed": bool(passed),
        }

    shadow_checks = [
        shadow_check("usable_star_count", float(len(usable)), operator=">=", limit=8.0),
        shadow_check("isolated_star_count", float(isolated_count), operator=">=", limit=5.0),
        shadow_check("saturated_ratio", saturated_ratio, operator="<=", limit=0.10),
        shadow_check("fwhm_mad_over_median", fwhm_relative_mad, operator="<=", limit=0.25),
        shadow_check(
            "ellipticity_median",
            ellipticity_summary.get("median"),
            operator="<=",
            limit=0.25,
        ),
        shadow_check(
            "ellipticity_mad",
            ellipticity_summary.get("mad"),
            operator="<=",
            limit=0.10,
        ),
        shadow_check("target_overlap_star_ratio", overlap_ratio, operator="<=", limit=0.25),
    ]
    report = {
        "schema": PSF_QUALITY_SCHEMA,
        "status": "available",
        "source_checkpoint": source_checkpoint,
        "catalog_role": catalog_role,
        "findstar_succeeded": bool(findstar_succeeded),
        "decision": decision,
        "hard_skip_rl": bool(hard_reason),
        "reason_code": hard_reason or "structurally_valid",
        "structural_checks_enforced": True,
        "structural_skip_conditions": [
            "empty_star_catalog_after_successful_findstar",
            "all_stars_saturated",
            "no_finite_coordinates_and_positive_fwhm",
        ],
        "shadow_thresholds_enforced": False,
        "shadow_thresholds": {
            "role": "report_only",
            "participates_in_rl_decision": False,
            "checks": shadow_checks,
            "would_warn": any(check.get("passed") is False for check in shadow_checks),
        },
        "target_structure_mask": {
            "available": target_structure_mask is not None,
            "union_threshold": TARGET_STRUCTURE_MASK_THRESHOLD,
            "local_overlap_limit": TARGET_STRUCTURE_OVERLAP_MAX,
        },
        "counts": {
            "total": total_count,
            "saturated": saturated_count,
            "valid_geometry": valid_geometry_count,
            "usable": len(usable),
            "isolated": isolated_count,
            "target_overlap_known": len(overlap_known),
            "target_overlapped": overlap_count,
            "eligible_for_local_guard": sum(
                bool(star["eligible_for_local_guard"]) for star in parsed
            ),
        },
        "summary": {
            "amplitude": _summary(star["amplitude"] for star in parsed),
            "background": _summary(star["background"] for star in parsed),
            "fwhm_x": _summary(star["fwhm_x"] for star in usable),
            "fwhm_y": _summary(star["fwhm_y"] for star in usable),
            "fwhm_geometry": fwhm_summary,
            "fwhm_mad_over_median": fwhm_relative_mad,
            "ellipticity": ellipticity_summary,
            "rmse": _summary(star["rmse"] for star in usable),
            "edge_distance": _summary(star["edge_distance"] for star in parsed),
            "nearest_neighbor_fwhm": _summary(
                star["nearest_neighbor_fwhm"] for star in usable
            ),
            "target_structure_overlap": _summary(
                star["target_structure_overlap"] for star in overlap_known
            ),
            "saturated_ratio": saturated_ratio,
            "target_overlap_star_ratio": overlap_ratio,
        },
        "stars": parsed,
    }
    return report, parsed


def _safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return None
    if abs(denominator) <= 1e-12:
        return None
    value = numerator / denominator
    return float(value) if math.isfinite(value) else None


def _centroid(
    residual: np.ndarray,
    mask: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
) -> Optional[Tuple[float, float]]:
    weights = np.maximum(np.asarray(residual, dtype=np.float64), 0.0) * mask
    total = float(np.sum(weights))
    if total <= 1e-12:
        return None
    return float(np.sum(weights * xx) / total), float(np.sum(weights * yy) / total)


def _one_signal_metrics(
    before: np.ndarray,
    after: np.ndarray,
    *,
    x: float,
    y: float,
    fwhm: float,
) -> Optional[Dict[str, Any]]:
    height, width = before.shape
    radius = 5.0 * fwhm
    x0 = max(0, int(math.floor(x - radius)))
    x1 = min(width, int(math.ceil(x + radius)) + 1)
    y0 = max(0, int(math.floor(y - radius)))
    y1 = min(height, int(math.ceil(y + radius)) + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    radial = np.sqrt((xx - x) ** 2 + (yy - y) ** 2) / fwhm
    core = radial <= 0.75
    inner_ring = (radial >= 1.25) & (radial <= 2.25)
    outer_ring = (radial >= 2.25) & (radial <= 3.5)
    ring = inner_ring | outer_ring
    background = (radial >= 3.5) & (radial <= 5.0)
    patch = radial <= 3.5
    centroid_support = radial <= 2.25
    if not all(bool(np.any(mask)) for mask in (core, ring, background, patch)):
        return None
    before_patch = np.asarray(before[y0:y1, x0:x1], dtype=np.float64)
    after_patch = np.asarray(after[y0:y1, x0:x1], dtype=np.float64)
    before_background = float(np.median(before_patch[background]))
    after_background = float(np.median(after_patch[background]))
    before_background_mad = float(
        np.median(np.abs(before_patch[background] - before_background))
    )
    after_background_mad = float(
        np.median(np.abs(after_patch[background] - after_background))
    )
    before_noise_sigma = max(1.4826 * before_background_mad, 1e-7)
    after_noise_sigma = max(1.4826 * after_background_mad, 1e-7)
    before_residual = before_patch - before_background
    after_residual = after_patch - after_background
    before_peak = float(np.max(before_residual[core]))
    after_peak = float(np.max(after_residual[core]))
    before_core_flux = float(np.sum(np.maximum(before_residual[core], 0.0)))
    after_core_flux = float(np.sum(np.maximum(after_residual[core], 0.0)))
    before_patch_flux = float(np.sum(np.maximum(before_residual[patch], 0.0)))
    after_patch_flux = float(np.sum(np.maximum(after_residual[patch], 0.0)))
    peak_floor = max(
        LOCAL_STAR_SIGNAL_SNR_MIN * before_noise_sigma,
        LOCAL_STAR_SIGNAL_ABS_MIN,
    )
    flux_floor = max(
        LOCAL_STAR_SIGNAL_SNR_MIN
        * before_noise_sigma
        * math.sqrt(float(np.count_nonzero(patch))),
        LOCAL_STAR_SIGNAL_ABS_MIN,
    )
    if before_peak < peak_floor or before_patch_flux <= flux_floor:
        return {
            "status": "excluded_low_signal",
            "reason_code": "baseline_star_signal_below_noise_floor",
            "before_peak": before_peak,
            "before_patch_flux": before_patch_flux,
            "before_noise_sigma": before_noise_sigma,
            "after_noise_sigma": after_noise_sigma,
            "peak_floor": peak_floor,
            "flux_floor": flux_floor,
            "signal_snr_min": LOCAL_STAR_SIGNAL_SNR_MIN,
        }

    normalization = before_peak
    ring_delta = after_residual[ring] - before_residual[ring]
    positive_ring_absolute = float(max(0.0, np.percentile(ring_delta, 95.0)))
    negative_ring_absolute = float(max(0.0, -np.percentile(ring_delta, 5.0)))
    ring_noise_sigma = math.sqrt(
        before_noise_sigma * before_noise_sigma
        + after_noise_sigma * after_noise_sigma
    )
    ring_absolute_floor = max(
        LOCAL_STAR_RING_SIGMA_MIN * ring_noise_sigma,
        LOCAL_STAR_SIGNAL_ABS_MIN,
    )
    before_centroid = _centroid(
        before_residual,
        centroid_support,
        xx,
        yy,
    )
    after_centroid = _centroid(
        after_residual,
        centroid_support,
        xx,
        yy,
    )
    centroid_shift = None
    if before_centroid is not None and after_centroid is not None:
        centroid_shift = float(
            math.hypot(
                after_centroid[0] - before_centroid[0],
                after_centroid[1] - before_centroid[1],
            )
            / fwhm
        )
    return {
        "status": "ok",
        "core_peak_ratio": _safe_ratio(after_peak, before_peak),
        "core_flux_ratio": _safe_ratio(after_core_flux, before_core_flux),
        "positive_ring_residual": positive_ring_absolute / normalization,
        "negative_ring_residual": negative_ring_absolute / normalization,
        "positive_ring_residual_absolute": positive_ring_absolute,
        "negative_ring_residual_absolute": negative_ring_absolute,
        "ring_noise_sigma": ring_noise_sigma,
        "ring_absolute_floor": ring_absolute_floor,
        "positive_ring_triggered": bool(
            positive_ring_absolute / normalization > 0.10
            and positive_ring_absolute > ring_absolute_floor
        ),
        "negative_ring_triggered": bool(
            negative_ring_absolute / normalization > 0.08
            and negative_ring_absolute > ring_absolute_floor
        ),
        "centroid_shift_fwhm": centroid_shift,
        "patch_flux_ratio": _safe_ratio(after_patch_flux, before_patch_flux),
        "before_peak": before_peak,
        "before_patch_flux": before_patch_flux,
        "before_noise_sigma": before_noise_sigma,
        "after_noise_sigma": after_noise_sigma,
        "peak_floor": peak_floor,
        "flux_floor": flux_floor,
    }


def _aggregate_signal(stars: Sequence[Dict[str, Any]], signal: str) -> Dict[str, Any]:
    blocks = [
        star.get("signals", {}).get(signal)
        for star in stars
        if isinstance(star.get("signals", {}).get(signal), dict)
    ]
    return {
        "evaluated_star_count": len(blocks),
        "core_peak_ratio": _summary(block.get("core_peak_ratio") for block in blocks),
        "core_flux_ratio": _summary(block.get("core_flux_ratio") for block in blocks),
        "positive_ring_residual": _summary(
            block.get("positive_ring_residual") for block in blocks
        ),
        "negative_ring_residual": _summary(
            block.get("negative_ring_residual") for block in blocks
        ),
        "positive_ring_residual_absolute": _summary(
            block.get("positive_ring_residual_absolute") for block in blocks
        ),
        "negative_ring_residual_absolute": _summary(
            block.get("negative_ring_residual_absolute") for block in blocks
        ),
        "ring_absolute_floor": _summary(
            block.get("ring_absolute_floor") for block in blocks
        ),
        "centroid_shift_fwhm": _summary(
            block.get("centroid_shift_fwhm") for block in blocks
        ),
        "patch_flux_ratio": _summary(block.get("patch_flux_ratio") for block in blocks),
    }


def unavailable_local_star_guard_report(
    reason: str,
    *,
    method: str = "none",
    reason_code: str = "measurement_failed",
) -> Dict[str, Any]:
    """Return the fail-closed contract used when local stars cannot be measured."""
    return {
        "schema": LOCAL_STAR_GUARD_SCHEMA,
        "status": "unavailable",
        "mode": "enforced",
        "enforced": True,
        "participates_in_acceptance": True,
        "method": str(method or "none"),
        "source_checkpoint": "stage5_input_linear.fit",
        "coordinate_policy": "fixed_baseline_catalog_no_output_redetection",
        "minimum_star_count": LOCAL_STAR_MIN_COUNT,
        "reason_code": str(reason_code or "measurement_failed"),
        "reason": str(reason),
        "decision_reasons": [str(reason_code or "measurement_failed")],
        "accepted": False,
        "rollback_required": True,
        "decision": "rollback",
        "would_rollback": True,
    }


def not_run_local_star_guard_report() -> Dict[str, Any]:
    """Return the stable contract for a deconvolution candidate not attempted."""
    return {
        "schema": LOCAL_STAR_GUARD_SCHEMA,
        "status": "not_run",
        "mode": "enforced",
        "enforced": True,
        "participates_in_acceptance": True,
        "method": "none",
        "source_checkpoint": "stage5_input_linear.fit",
        "reason_code": "deconvolution_not_applied",
        "decision_reasons": ["deconvolution_not_applied"],
        "accepted": False,
        "rollback_required": False,
        "decision": "not_run",
        "would_rollback": False,
    }


def assess_local_star_guard(
    before_image: Any,
    after_image: Any,
    star_catalog: Optional[Sequence[Dict[str, Any]]],
    *,
    method: str,
) -> Dict[str, Any]:
    """Compare fixed baseline stars and return an enforced acceptance decision."""
    base: Dict[str, Any] = {
        "schema": LOCAL_STAR_GUARD_SCHEMA,
        "mode": "enforced",
        "enforced": True,
        "participates_in_acceptance": True,
        "method": str(method or "none"),
        "source_checkpoint": "stage5_input_linear.fit",
        "coordinate_policy": "fixed_baseline_catalog_no_output_redetection",
        "minimum_star_count": LOCAL_STAR_MIN_COUNT,
        "regions_fwhm": {
            "core": [0.0, 0.75],
            "inner_ring": [1.25, 2.25],
            "outer_ring": [2.25, 3.5],
            "local_background": [3.5, 5.0],
        },
    }
    try:
        before_rgb = _to_chw_float(before_image)
        after_rgb = _to_chw_float(after_image)
        if before_rgb.shape != after_rgb.shape:
            raise ValueError(
                f"Stage5 baseline/output shape mismatch: {before_rgb.shape}!={after_rgb.shape}"
            )
        eligible = [
            dict(star)
            for star in (star_catalog or [])
            if bool(star.get("eligible_for_local_guard"))
        ]
        if len(eligible) < LOCAL_STAR_MIN_COUNT:
            return {
                **base,
                "status": "unavailable",
                "reason_code": "insufficient_eligible_baseline_stars",
                "decision_reasons": [
                    "insufficient_eligible_baseline_stars"
                ],
                "eligible_star_count": len(eligible),
                "evaluated_star_count": 0,
                "excluded_low_signal_star_count": 0,
                "accepted": False,
                "rollback_required": True,
                "decision": "rollback",
                "would_rollback": True,
            }
        signals_before = {
            "rec709": (
                0.2126 * before_rgb[0]
                + 0.7152 * before_rgb[1]
                + 0.0722 * before_rgb[2]
            ),
            "r": before_rgb[0],
            "g": before_rgb[1],
            "b": before_rgb[2],
        }
        signals_after = {
            "rec709": (
                0.2126 * after_rgb[0]
                + 0.7152 * after_rgb[1]
                + 0.0722 * after_rgb[2]
            ),
            "r": after_rgb[0],
            "g": after_rgb[1],
            "b": after_rgb[2],
        }
        evaluated: List[Dict[str, Any]] = []
        excluded_low_signal: List[Dict[str, Any]] = []
        for star in eligible:
            x = _finite_float(star.get("x"))
            y = _finite_float(star.get("y"))
            fwhm = _finite_float(star.get("fwhm_geometry"))
            if x is None or y is None or fwhm is None or fwhm <= 0.0:
                continue
            signal_metrics: Dict[str, Any] = {}
            signal_exclusions: Dict[str, Any] = {}
            for signal in ("rec709", "r", "g", "b"):
                metrics = _one_signal_metrics(
                    signals_before[signal],
                    signals_after[signal],
                    x=x,
                    y=y,
                    fwhm=fwhm,
                )
                if metrics is not None and metrics.get("status") == "ok":
                    signal_metrics[signal] = metrics
                elif metrics is not None:
                    signal_exclusions[signal] = metrics
            if "rec709" not in signal_metrics:
                excluded_low_signal.append(
                    {
                        "index": star.get("index"),
                        "x": x,
                        "y": y,
                        "fwhm_geometry": fwhm,
                        "signals": signal_exclusions,
                        "reason_code": str(
                            (signal_exclusions.get("rec709") or {}).get(
                                "reason_code"
                            )
                            or "rec709_measurement_unavailable"
                        ),
                    }
                )
                continue
            rec709 = signal_metrics["rec709"]
            peak = rec709.get("core_peak_ratio")
            patch_flux = rec709.get("patch_flux_ratio")
            centroid = rec709.get("centroid_shift_fwhm")
            anomalous = bool(
                bool(rec709.get("positive_ring_triggered", False))
                or bool(rec709.get("negative_ring_triggered", False))
                or (peak is not None and (float(peak) > 2.5 or float(peak) < 0.65))
                or (
                    patch_flux is not None
                    and not 0.90 <= float(patch_flux) <= 1.10
                )
                or (centroid is not None and float(centroid) > 0.25)
            )
            evaluated.append(
                {
                    "index": star.get("index"),
                    "x": x,
                    "y": y,
                    "fwhm_geometry": fwhm,
                    "signals": signal_metrics,
                    "signal_exclusions": signal_exclusions,
                    "anomalous": anomalous,
                }
            )
        if len(evaluated) < LOCAL_STAR_MIN_COUNT:
            return {
                **base,
                "status": "unavailable",
                "reason_code": "insufficient_evaluable_baseline_stars",
                "decision_reasons": [
                    "insufficient_evaluable_baseline_stars"
                ],
                "eligible_star_count": len(eligible),
                "evaluated_star_count": len(evaluated),
                "excluded_low_signal_star_count": len(excluded_low_signal),
                "excluded_low_signal_stars": excluded_low_signal,
                "accepted": False,
                "rollback_required": True,
                "decision": "rollback",
                "would_rollback": True,
            }

        aggregates = {
            signal: _aggregate_signal(evaluated, signal)
            for signal in ("rec709", "r", "g", "b")
        }
        rec709 = aggregates["rec709"]
        positive_p95 = rec709["positive_ring_residual"].get("p95")
        negative_p95 = rec709["negative_ring_residual"].get("p95")
        positive_absolute_p95 = rec709[
            "positive_ring_residual_absolute"
        ].get("p95")
        negative_absolute_p95 = rec709[
            "negative_ring_residual_absolute"
        ].get("p95")
        positive_ring_triggered_count = sum(
            bool(star["signals"]["rec709"].get("positive_ring_triggered"))
            for star in evaluated
        )
        negative_ring_triggered_count = sum(
            bool(star["signals"]["rec709"].get("negative_ring_triggered"))
            for star in evaluated
        )
        peak_p95 = rec709["core_peak_ratio"].get("p95")
        peak_p05 = rec709["core_peak_ratio"].get("p05")
        flux_median = rec709["patch_flux_ratio"].get("median")
        centroid_p95 = rec709["centroid_shift_fwhm"].get("p95")
        anomalous_ratio = float(np.mean([star["anomalous"] for star in evaluated]))
        checks = {
            "positive_ring_residual_p95": {
                "observed": positive_p95,
                "observed_absolute": positive_absolute_p95,
                "significant_star_count": positive_ring_triggered_count,
                "would_trigger": positive_ring_triggered_count > 0,
                "limit": 0.10,
                "absolute_limit": (
                    f">{LOCAL_STAR_RING_SIGMA_MIN:.1f}x local delta noise"
                ),
                "operator": ">",
            },
            "negative_ring_residual_p95": {
                "observed": negative_p95,
                "observed_absolute": negative_absolute_p95,
                "significant_star_count": negative_ring_triggered_count,
                "would_trigger": negative_ring_triggered_count > 0,
                "limit": 0.08,
                "absolute_limit": (
                    f">{LOCAL_STAR_RING_SIGMA_MIN:.1f}x local delta noise"
                ),
                "operator": ">",
            },
            "core_peak_ratio_p95": {
                "observed": peak_p95,
                "would_trigger": peak_p95 is not None and peak_p95 > 2.5,
                "limit": 2.5,
                "operator": ">",
            },
            "core_peak_ratio_p05": {
                "observed": peak_p05,
                "would_trigger": peak_p05 is not None and peak_p05 < 0.65,
                "limit": 0.65,
                "operator": "<",
            },
            "patch_flux_ratio_median": {
                "observed": flux_median,
                "would_trigger": (
                    flux_median is not None
                    and not 0.90 <= float(flux_median) <= 1.10
                ),
                "limits": [0.90, 1.10],
                "operator": "outside",
            },
            "centroid_shift_fwhm_p95": {
                "observed": centroid_p95,
                "would_trigger": centroid_p95 is not None and centroid_p95 > 0.25,
                "limit": 0.25,
                "operator": ">",
            },
            "anomalous_star_ratio": {
                "observed": anomalous_ratio,
                "would_trigger": anomalous_ratio > 0.20,
                "limit": 0.20,
                "operator": ">",
            },
        }
        reasons = [
            name for name, check in checks.items() if bool(check["would_trigger"])
        ]
        accepted = not reasons
        return {
            **base,
            "status": "available",
            "reason_code": (
                "local_star_guard_rejected"
                if reasons
                else "local_star_guard_accepted"
            ),
            "eligible_star_count": len(eligible),
            "evaluated_star_count": len(evaluated),
            "excluded_low_signal_star_count": len(excluded_low_signal),
            "excluded_low_signal_stars": excluded_low_signal,
            "anomalous_star_count": sum(star["anomalous"] for star in evaluated),
            "anomalous_star_ratio": anomalous_ratio,
            "decision_basis": "rec709",
            "accepted": accepted,
            "rollback_required": not accepted,
            "decision": "accept" if accepted else "rollback",
            "would_rollback": bool(reasons),
            "would_rollback_reasons": reasons,
            "decision_reasons": (
                reasons if reasons else ["all_local_star_checks_passed"]
            ),
            "checks": checks,
            "aggregate": aggregates,
            "stars": evaluated,
        }
    except (IndexError, TypeError, ValueError, FloatingPointError) as error:
        return {
            **base,
            "status": "unavailable",
            "reason_code": "measurement_failed",
            "reason": str(error),
            "decision_reasons": ["measurement_failed"],
            "accepted": False,
            "rollback_required": True,
            "decision": "rollback",
            "would_rollback": True,
        }
