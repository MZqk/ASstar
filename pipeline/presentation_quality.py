"""Stage 10 presentation-quality gate anchored to the frozen Stage 7 result.

This module never consumes an external reference image.  It compares the final
candidate with the authenticated, internal Stage 7 presentation reference on
candidate-invariant masks frozen from Stage 6.
"""
from __future__ import annotations

import math
import hashlib
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from image_metrics import _box_blur_gray
import stage7_stretch_metrics


PRESENTATION_QUALITY_SCHEMA = "starun.presentation-quality.v1"
DELIVERY_GATES_SCHEMA = "starun.final-delivery-gates.v1"
STAGE7_PRESENTATION_REFERENCE_SCHEMA = (
    "starun.stage7-presentation-reference.v1"
)


def _finite(value: Any) -> Optional[float]:
    try:
        measured = float(value)
    except (TypeError, ValueError):
        return None
    return measured if math.isfinite(measured) else None


def _cfg_float(
    cfg: Any,
    name: str,
    default: float,
    lower: float,
    upper: float,
) -> float:
    value = _finite(getattr(cfg, name, default))
    if value is None:
        value = default
    return float(np.clip(value, lower, upper))


def _locked_lower_cfg(
    cfg: Any,
    name: str,
    locked_minimum: float,
    upper: float,
) -> float:
    """Allow a task to tighten, but never weaken, a fixed lower gate."""

    return max(
        float(locked_minimum),
        _cfg_float(cfg, name, locked_minimum, locked_minimum, upper),
    )


def _locked_upper_cfg(
    cfg: Any,
    name: str,
    locked_maximum: float,
    lower: float,
) -> float:
    """Allow a task to tighten, but never weaken, a fixed upper gate."""

    return min(
        float(locked_maximum),
        _cfg_float(cfg, name, locked_maximum, lower, locked_maximum),
    )


def _lower_gate(value: Optional[float], limit: float) -> Dict[str, Any]:
    accepted = bool(value is not None and value + 1e-12 >= limit)
    return {
        "status": "ok" if accepted else "rejected",
        "accepted": accepted,
        "value": value,
        "limit": float(limit),
        "direction": "greater_than_or_equal",
    }


def _upper_gate(value: Optional[float], limit: float) -> Dict[str, Any]:
    accepted = bool(value is not None and value <= limit + 1e-12)
    return {
        "status": "ok" if accepted else "rejected",
        "accepted": accepted,
        "value": value,
        "limit": float(limit),
        "direction": "less_than_or_equal",
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_stage7_presentation_reference(
    report: Mapping[str, Any],
    pixels: Any,
    artifact_path: Path,
) -> Dict[str, Any]:
    """Verify the frozen Stage 7 ruler before Stage 10 consumes its pixels."""

    payload = dict(report or {})
    if payload.get("schema") != STAGE7_PRESENTATION_REFERENCE_SCHEMA:
        raise ValueError("Stage7 presentation reference schema mismatch")
    if payload.get("status") != "ready" or payload.get("accepted") is not True:
        raise ValueError("Stage7 presentation reference is not accepted")
    expected_report_sha = str(payload.get("report_sha256") or "")
    unsigned = dict(payload)
    unsigned.pop("report_sha256", None)
    if (
        len(expected_report_sha) != 64
        or stage7_stretch_metrics.canonical_json_sha256(unsigned)
        != expected_report_sha
    ):
        raise ValueError("Stage7 presentation reference report digest mismatch")
    artifact = dict(payload.get("artifact") or {})
    path = Path(artifact_path)
    if artifact.get("file") != path.name or not path.is_file():
        raise ValueError("Stage7 presentation reference artifact unavailable")
    container_sha = _sha256_file(path)
    if container_sha != str(artifact.get("container_sha256") or ""):
        raise ValueError("Stage7 presentation reference container SHA mismatch")
    pixel_array = np.asarray(pixels)
    pixel_sha = stage7_stretch_metrics.stage7_pixel_sha256(pixel_array)
    if pixel_sha != str(artifact.get("pixel_sha256") or ""):
        raise ValueError("Stage7 presentation reference pixel SHA mismatch")
    if [int(value) for value in pixel_array.shape] != list(
        artifact.get("shape") or []
    ):
        raise ValueError("Stage7 presentation reference shape mismatch")
    selected = dict(payload.get("selected_candidate") or {})
    formal = dict(payload.get("source_artifact") or {})
    if (
        selected.get("pixel_sha256") != pixel_sha
        or formal.get("pixel_sha256") != pixel_sha
    ):
        raise ValueError("Stage7 formal/selected pixel binding mismatch")
    binding_payload = {
        "linear_source": payload.get("linear_source"),
        "selected_candidate": payload.get("selected_candidate"),
        "matched_domain": payload.get("matched_domain"),
        "formal_source_artifact": payload.get("source_artifact"),
    }
    if (
        stage7_stretch_metrics.canonical_json_sha256(binding_payload)
        != str(payload.get("source_binding_sha256") or "")
    ):
        raise ValueError("Stage7 presentation source binding digest mismatch")
    return {
        "status": "verified",
        "accepted": True,
        "container_sha256": container_sha,
        "pixel_sha256": pixel_sha,
        "report_sha256": expected_report_sha,
        "source_binding_sha256": payload.get("source_binding_sha256"),
    }


def _profile_name(target_type: str, explicit_profile: str) -> str:
    profile = str(explicit_profile or "").strip().lower()
    if profile:
        return profile
    target = str(target_type or "").strip().lower()
    if target in {"open_cluster", "star_cluster", "star_preserve"}:
        return "star_colour_preserve"
    if target in {"large_galaxy", "small_galaxy"}:
        return "galaxy_core_halo_balance"
    if target == "bright_emission_reflection_nebula":
        return "bright_core_composite_reveal"
    return "generic_balanced"


def _frozen_support(
    masks: Optional[Mapping[str, Any]],
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    mask_map = dict(masks or {})
    subject = stage7_stretch_metrics._stage7_subject_weight(mask_map, shape) > 0.25
    background_weight = stage7_stretch_metrics._stage7_mask(
        mask_map,
        "background_mask",
        shape,
    )
    if background_weight is None:
        background = ~subject
    else:
        background = background_weight > 0.50
        background &= ~subject
    if int(np.count_nonzero(subject)) < 64:
        raise ValueError("frozen subject ROI contains too few pixels")
    if int(np.count_nonzero(background)) < 64:
        raise ValueError("frozen background ROI contains too few pixels")
    return subject, background


def _subject_color_report(
    reference: np.ndarray,
    candidate: np.ndarray,
    masks: Optional[Mapping[str, Any]],
    cfg: Any,
) -> Dict[str, Any]:
    limits = {
        "saturation_p50_retention_min": _locked_lower_cfg(
            cfg,
            "stage10_presentation_color_p50_retention_min",
            0.35,
            1.50,
        ),
        "saturation_p95_retention_min": _locked_lower_cfg(
            cfg,
            "stage10_presentation_color_p95_retention_min",
            0.50,
            1.50,
        ),
        "opponent_energy_retention_min": _locked_lower_cfg(
            cfg,
            "stage10_presentation_opponent_energy_retention_min",
            0.40,
            1.50,
        ),
        "opponent_direction_correlation_min": _locked_lower_cfg(
            cfg,
            "stage10_presentation_color_direction_correlation_min",
            0.70,
            1.0,
        ),
    }
    report: Dict[str, Any] = {
        "status": "unavailable",
        "accepted": False,
        "applicable": True,
        "limits": limits,
        "metrics": {},
        "gates": {},
        "issues": ["presentation_color_measurement_unavailable"],
    }
    try:
        source_rgb = stage7_stretch_metrics._stage7_rgb_float_fullres(reference)
        output_rgb = stage7_stretch_metrics._stage7_rgb_float_fullres(candidate)
        if source_rgb.shape != output_rgb.shape:
            raise ValueError(
                f"presentation reference/final shape mismatch: "
                f"{source_rgb.shape}!={output_rgb.shape}"
            )
        if source_rgb.shape[0] != 3:
            report.update(
                status="not_applicable",
                accepted=True,
                applicable=False,
                issues=[],
                reason="mono_source",
            )
            return report
        if not np.all(np.isfinite(source_rgb)) or not np.all(
            np.isfinite(output_rgb)
        ):
            raise ValueError("presentation pixels contain non-finite values")
        subject, background = _frozen_support(
            masks,
            tuple(int(value) for value in source_rgb.shape[1:]),
        )

        def low_frequency(rgb: np.ndarray) -> np.ndarray:
            result = np.empty_like(rgb, dtype=np.float32)
            for channel in range(3):
                plane = np.asarray(rgb[channel], dtype=np.float32)
                for _ in range(2):
                    plane = _box_blur_gray(plane)
                result[channel] = plane
            return result

        source_low = low_frequency(source_rgb)
        output_low = low_frequency(output_rgb)
        source_peak = np.max(source_low, axis=0)
        output_peak = np.max(output_low, axis=0)
        source_sat = (source_peak - np.min(source_low, axis=0)) / np.maximum(
            source_peak,
            1e-6,
        )
        output_sat = (output_peak - np.min(output_low, axis=0)) / np.maximum(
            output_peak,
            1e-6,
        )
        source_background = np.asarray(
            [np.median(source_low[channel][background]) for channel in range(3)],
            dtype=np.float32,
        )
        output_background = np.asarray(
            [np.median(output_low[channel][background]) for channel in range(3)],
            dtype=np.float32,
        )
        source_signal = np.clip(
            source_low - source_background[:, None, None],
            0.0,
            None,
        )
        output_signal = np.clip(
            output_low - output_background[:, None, None],
            0.0,
            None,
        )
        source_vectors = np.stack(
            (
                source_signal[0][subject] - source_signal[1][subject],
                source_signal[2][subject] - source_signal[1][subject],
            ),
            axis=0,
        )
        output_vectors = np.stack(
            (
                output_signal[0][subject] - output_signal[1][subject],
                output_signal[2][subject] - output_signal[1][subject],
            ),
            axis=0,
        )
        source_sat_values = np.asarray(source_sat[subject], dtype=np.float32)
        output_sat_values = np.asarray(output_sat[subject], dtype=np.float32)
        source_energy = float(np.sqrt(np.mean(source_vectors * source_vectors)))
        output_energy = float(np.sqrt(np.mean(output_vectors * output_vectors)))
        source_sat_p50 = float(np.median(source_sat_values))
        source_sat_p95 = float(np.quantile(source_sat_values, 0.95))
        output_sat_p50 = float(np.median(output_sat_values))
        output_sat_p95 = float(np.quantile(output_sat_values, 0.95))

        low_sat = _locked_upper_cfg(
            cfg,
            "stage7_low_chroma_source_saturation_max",
            0.02,
            0.0,
        )
        low_energy = _locked_upper_cfg(
            cfg,
            "stage7_low_chroma_source_opponent_rms_max",
            0.0001,
            0.0,
        )
        limits["low_chroma_source_saturation_max"] = low_sat
        limits["low_chroma_source_opponent_rms_max"] = low_energy
        if source_sat_p50 < low_sat and source_energy < low_energy:
            report.update(
                status="not_applicable",
                accepted=True,
                applicable=False,
                issues=[],
                reason="stage7_reference_chroma_below_measurement_floor",
                metrics={
                    "source_saturation_p50": source_sat_p50,
                    "source_opponent_rms": source_energy,
                },
            )
            return report

        denominator = float(
            np.sqrt(np.sum(source_vectors * source_vectors))
            * np.sqrt(np.sum(output_vectors * output_vectors))
        )
        direction = (
            float(np.sum(source_vectors * output_vectors) / denominator)
            if denominator > 1e-12
            else 0.0
        )
        metrics = {
            "source_saturation_p50": source_sat_p50,
            "source_saturation_p95": source_sat_p95,
            "output_saturation_p50": output_sat_p50,
            "output_saturation_p95": output_sat_p95,
            "saturation_p50_retention": output_sat_p50
            / max(source_sat_p50, 1e-12),
            "saturation_p95_retention": output_sat_p95
            / max(source_sat_p95, 1e-12),
            "source_opponent_rms": source_energy,
            "output_opponent_rms": output_energy,
            "opponent_energy_retention": output_energy
            / max(source_energy, 1e-12),
            "opponent_direction_correlation": float(np.clip(direction, -1.0, 1.0)),
            "source_background_rgb": [float(value) for value in source_background],
            "output_background_rgb": [float(value) for value in output_background],
            "support_count": int(np.count_nonzero(subject)),
        }
        gate_metrics = {
            "saturation_p50_retention_min": "saturation_p50_retention",
            "saturation_p95_retention_min": "saturation_p95_retention",
            "opponent_energy_retention_min": "opponent_energy_retention",
            "opponent_direction_correlation_min": (
                "opponent_direction_correlation"
            ),
        }
        gates = {
            name: _lower_gate(metrics[metric_name], limits[name])
            for name, metric_name in gate_metrics.items()
        }
        issues = [name for name, gate in gates.items() if not gate["accepted"]]
        report.update(
            status="ok" if not issues else "rejected",
            accepted=not issues,
            applicable=True,
            metrics=metrics,
            gates=gates,
            issues=issues,
        )
        return report
    except (IndexError, TypeError, ValueError, FloatingPointError) as error:
        report["reason"] = str(error)
        return report


def _psf_ratios(stage9_quality: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    quality = dict(stage9_quality or {})
    closure = quality.get("psf_closure") or {}
    groups = closure.get("groups") or {}
    ratios: Dict[str, float] = {}
    for name in ("all", "weak", "bright"):
        group = groups.get(name)
        if not isinstance(group, Mapping) or str(group.get("status")) != "ok":
            continue
        ratio = _finite(group.get("fwhm_ratio_median"))
        if ratio is not None and ratio > 0.0:
            ratios[name] = ratio
    return ratios


def _presentation_brightness_report(
    candidate_metrics: Dict[str, Any],
    reference_metrics: Dict[str, Any],
    cfg: Any,
    *,
    profile_name: str,
) -> Dict[str, Any]:
    """Reuse the Stage 7 profile contract while locking its P50 floor."""

    report = dict(
        stage7_stretch_metrics.subject_brightness_selection(
            candidate_metrics,
            reference_metrics,
            profile_name=profile_name,
        )
    )
    normalized = str(profile_name or "generic_balanced").strip().lower()
    field, locked_floor = (
        (
            "stage10_presentation_star_preserve_brightness_retention_min",
            0.20,
        )
        if normalized == "star_colour_preserve"
        else (
            "stage10_presentation_galaxy_brightness_retention_min",
            0.58,
        )
        if normalized == "galaxy_core_halo_balance"
        else (
            "stage10_presentation_composite_brightness_retention_min",
            0.45,
        )
        if normalized == "bright_core_composite_reveal"
        else (
            "stage10_presentation_generic_brightness_retention_min",
            0.60,
        )
    )
    floor = _locked_lower_cfg(cfg, field, locked_floor, 1.50)
    retention = _finite((report.get("retention") or {}).get("subject_p50"))
    p50_gate = _lower_gate(retention, floor)
    prior_accepted = bool(report.get("formal_floor_passed", False))
    accepted = bool(prior_accepted and p50_gate["accepted"])
    issues = list(report.get("issues") or [])
    if not p50_gate["accepted"] and "subject_p50_retention_below_floor" not in issues:
        issues.append("subject_p50_retention_below_floor")
    floors = dict(report.get("floors") or {})
    floors["subject_p50_retention"] = floor
    report.update(
        status="ok" if accepted else "rejected",
        formal_floor_passed=accepted,
        reason_code=(
            "stage7_subject_brightness_floor_passed"
            if accepted
            else "stage7_subject_brightness_floor_unmet"
        ),
        issues=issues,
        floors=floors,
        presentation_p50_gate=p50_gate,
    )
    return report


def _stars_report(
    cfg: Any,
    stage9_quality: Optional[Mapping[str, Any]],
    *,
    stars_required: bool,
    stars_not_required_verified: bool,
) -> Dict[str, Any]:
    hard_min = _cfg_float(cfg, "stage9_psf_fwhm_ratio_min", 0.93, 0.50, 1.00)
    hard_max = _cfg_float(cfg, "stage9_psf_fwhm_ratio_max", 1.10, 1.00, 1.50)
    soft_min = max(
        hard_min,
        _locked_lower_cfg(
            cfg,
            "stage9_psf_recovery_target_min",
            0.97,
            1.00,
        ),
    )
    soft_max = min(
        hard_max,
        _locked_upper_cfg(
            cfg,
            "stage9_psf_recovery_target_max",
            1.05,
            1.00,
        ),
    )
    if not stars_required:
        return {
            "status": "not_applicable" if stars_not_required_verified else "unavailable",
            "accepted": bool(stars_not_required_verified),
            "applicable": False,
            "reason": (
                "verified_stars_not_required"
                if stars_not_required_verified
                else "stars_not_required_identity_unverified"
            ),
            "limits": {
                "soft_min": soft_min,
                "soft_max": soft_max,
                "hard_min": hard_min,
                "hard_max": hard_max,
            },
            "ratios": {},
        }
    ratios = _psf_ratios(stage9_quality)
    required_groups = ("all", "weak", "bright")
    missing_groups = [name for name in required_groups if name not in ratios]
    groups_complete = not missing_groups
    hard_passed = bool(
        groups_complete
        and all(
            hard_min <= ratios[name] <= hard_max
            for name in required_groups
        )
    )
    soft_passed = bool(
        groups_complete
        and all(
            soft_min <= ratios[name] <= soft_max
            for name in required_groups
        )
    )
    return {
        "status": "ok" if hard_passed and soft_passed else "rejected",
        "accepted": bool(hard_passed and soft_passed),
        "applicable": True,
        "hard_science_gate_passed": hard_passed,
        "soft_presentation_target_passed": soft_passed,
        "required_groups": list(required_groups),
        "groups_complete": groups_complete,
        "missing_groups": missing_groups,
        "limits": {
            "soft_min": soft_min,
            "soft_max": soft_max,
            "hard_min": hard_min,
            "hard_max": hard_max,
        },
        "ratios": ratios,
        "issues": (
            []
            if hard_passed and soft_passed
            else [
                "stage9_psf_groups_incomplete"
                if not groups_complete
                else "stage9_psf_hard_gate_failed"
                if not hard_passed
                else "stage9_psf_presentation_target_unmet"
            ]
        ),
    }


def _frozen_mask_for_presentation(
    masks: Optional[Mapping[str, Any]],
    name: str,
    shape: tuple[int, int],
    *,
    required: bool,
) -> Optional[np.ndarray]:
    """Load one frozen mask without silently accepting malformed evidence."""

    if not isinstance(masks, Mapping) or masks.get(name) is None:
        if required:
            raise ValueError(f"frozen {name} evidence is unavailable")
        return None
    mask = np.asarray(masks[name], dtype=np.float32)
    if mask.ndim != 2 or tuple(mask.shape) != tuple(shape):
        raise ValueError(
            f"frozen {name} shape mismatch: {mask.shape}!={shape}"
        )
    if not np.all(np.isfinite(mask)):
        raise ValueError(f"frozen {name} contains non-finite values")
    return np.clip(mask, 0.0, 1.0)


def _same_content_microdetail_report(
    reference: np.ndarray,
    candidate: np.ndarray,
    masks: Optional[Mapping[str, Any]],
    *,
    stars_required: bool,
    retention_min: float,
    growth_max: float,
) -> Dict[str, Any]:
    """Measure non-stellar subject detail on one frozen boolean support.

    Stage 7's formal reference is Starless when Stage 9 must add stars back.
    Measuring all subject pixels would therefore treat legitimate star remixing
    as newly generated microdetail.  This gate keeps the existing whole-subject
    rendition metrics as diagnostics, but formally compares only the shared
    Stage 6 subject support after frozen stellar/core/saturation guards are
    removed from both images.
    """

    report: Dict[str, Any] = {
        "schema": "starun.presentation-microdetail-same-content.v1",
        "status": "unavailable",
        "accepted": False,
        "measurement_domain": (
            "stage6_frozen_nonstellar_subject_same_boolean_support"
        ),
        "retention": _lower_gate(None, retention_min),
        "growth": _upper_gate(None, growth_max),
        "metrics": {},
        "support": {},
        "issues": ["microdetail_same_content_measurement_unavailable"],
    }
    try:
        reference_rgb = stage7_stretch_metrics._stage7_rgb_float_fullres(
            reference
        )
        candidate_rgb = stage7_stretch_metrics._stage7_rgb_float_fullres(
            candidate
        )
        if reference_rgb.shape != candidate_rgb.shape:
            raise ValueError(
                "microdetail reference/candidate shape mismatch: "
                f"{reference_rgb.shape}!={candidate_rgb.shape}"
            )
        if not np.all(np.isfinite(reference_rgb)) or not np.all(
            np.isfinite(candidate_rgb)
        ):
            raise ValueError("microdetail pixels contain non-finite values")
        shape = tuple(int(value) for value in reference_rgb.shape[1:])
        subject_weight = _frozen_mask_for_presentation(
            masks,
            "subject_mask",
            shape,
            required=True,
        )
        if subject_weight is None:
            raise ValueError("frozen subject_mask evidence is unavailable")
        support = subject_weight > 0.25
        consumed_masks: Dict[str, Dict[str, Any]] = {}

        def record_consumed_mask(
            name: str,
            mask: np.ndarray,
            threshold: float,
        ) -> None:
            canonical = np.ascontiguousarray(mask, dtype="<f4")
            consumed_masks[name] = {
                "source": "stage6_frozen_rendition_masks",
                "shape": [int(value) for value in canonical.shape],
                "dtype": "float32-le",
                "threshold": float(threshold),
                "sha256": hashlib.sha256(
                    canonical.tobytes(order="C")
                ).hexdigest(),
            }

        record_consumed_mask("subject_mask", subject_weight, 0.25)

        star_mask = _frozen_mask_for_presentation(
            masks,
            "star_mask",
            shape,
            required=stars_required,
        )
        star_halo_guard = _frozen_mask_for_presentation(
            masks,
            "star_halo_guard_mask",
            shape,
            required=stars_required,
        )
        exclusion_specs = (
            ("star_mask", star_mask, 0.05),
            ("star_halo_guard_mask", star_halo_guard, 0.05),
            (
                "core_mask",
                _frozen_mask_for_presentation(
                    masks,
                    "core_mask",
                    shape,
                    required=False,
                ),
                0.05,
            ),
            (
                "limited_core_exclusion_mask",
                _frozen_mask_for_presentation(
                    masks,
                    "limited_core_exclusion_mask",
                    shape,
                    required=False,
                ),
                0.05,
            ),
            (
                "original_saturation_map",
                _frozen_mask_for_presentation(
                    masks,
                    "original_saturation_map",
                    shape,
                    required=False,
                ),
                0.0,
            ),
            (
                "saturation_map",
                _frozen_mask_for_presentation(
                    masks,
                    "saturation_map",
                    shape,
                    required=False,
                ),
                0.0,
            ),
        )
        exclusion_counts: Dict[str, int] = {}
        for name, mask, threshold in exclusion_specs:
            if mask is None:
                continue
            record_consumed_mask(name, mask, threshold)
            excluded = mask > threshold
            exclusion_counts[name] = int(np.count_nonzero(support & excluded))
            support &= ~excluded

        valid_mask = _frozen_mask_for_presentation(
            masks,
            "shared_valid_mask",
            shape,
            required=False,
        )
        if valid_mask is not None:
            record_consumed_mask("shared_valid_mask", valid_mask, 0.0)
            invalid = valid_mask <= 0.0
            exclusion_counts["shared_invalid_mask"] = int(
                np.count_nonzero(support & invalid)
            )
            support &= ~invalid
        support_count = int(np.count_nonzero(support))
        if support_count < 64:
            raise ValueError(
                "frozen same-content microdetail support contains too few "
                f"pixels: {support_count}<64"
            )

        def local_detail(rgb: np.ndarray) -> np.ndarray:
            luminance = (
                0.2126 * rgb[0]
                + 0.7152 * rgb[1]
                + 0.0722 * rgb[2]
            ).astype(np.float32)
            return np.abs(luminance - _box_blur_gray(luminance))

        def sampled(values: np.ndarray) -> np.ndarray:
            selected = np.asarray(values[support], dtype=np.float32)
            if selected.size > 300_000:
                stride = int(np.ceil(selected.size / 300_000.0))
                selected = selected[::stride]
            return selected

        reference_detail = float(
            np.percentile(sampled(local_detail(reference_rgb)), 75.0)
        )
        candidate_detail = float(
            np.percentile(sampled(local_detail(candidate_rgb)), 75.0)
        )
        if reference_detail <= 1e-12:
            raise ValueError(
                "frozen reference microdetail is below the measurement floor"
            )
        detail_ratio = candidate_detail / reference_detail
        retention = _lower_gate(detail_ratio, retention_min)
        growth = _upper_gate(detail_ratio, growth_max)
        accepted = bool(retention["accepted"] and growth["accepted"])
        support_bytes = np.ascontiguousarray(
            support,
            dtype=np.uint8,
        ).tobytes(order="C")
        issues = []
        if not retention["accepted"]:
            issues.append("microdetail_retention_below_floor")
        if not growth["accepted"]:
            issues.append("microdetail_growth_above_limit")
        report.update(
            status="ok" if accepted else "rejected",
            accepted=accepted,
            retention=retention,
            growth=growth,
            metrics={
                "reference_microdetail": reference_detail,
                "candidate_microdetail": candidate_detail,
                "retention": detail_ratio,
            },
            support={
                "threshold": 0.25,
                "count": support_count,
                "coverage": support_count / float(support.size),
                "sha256": hashlib.sha256(support_bytes).hexdigest(),
                "shared_between_reference_and_candidate": True,
                "exclusion_counts": exclusion_counts,
                "star_halo_evidence": (
                    "frozen_star_mask_plus_star_halo_guard"
                    if star_mask is not None and star_halo_guard is not None
                    else "optional_frozen_star_or_halo_guard"
                    if star_mask is not None or star_halo_guard is not None
                    else "not_required"
                ),
                "mask_provenance": {
                    "source": "stage6_frozen_rendition_masks",
                    "candidate_independent": True,
                    "consumed_masks": consumed_masks,
                    "digest_sha256": (
                        stage7_stretch_metrics.canonical_json_sha256(
                            consumed_masks
                        )
                    ),
                },
            },
            issues=issues,
        )
        return report
    except (IndexError, TypeError, ValueError, FloatingPointError) as error:
        report["reason"] = str(error)
        return report


def build_presentation_quality_report(
    reference: Any,
    candidate: Any,
    masks: Optional[Mapping[str, Any]],
    cfg: Any,
    *,
    target_type: str,
    profile_name: str = "",
    stage9_quality: Optional[Mapping[str, Any]] = None,
    stars_required: bool,
    stars_not_required_verified: bool,
    scientific_report: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the fail-closed internal presentation gate for Stage 10."""

    report: Dict[str, Any] = {
        "schema": PRESENTATION_QUALITY_SCHEMA,
        "status": "unavailable",
        "accepted": False,
        "external_reference_used": False,
        "reference": "stage7_presentation_reference",
        "measurement_domain": "canonical_float_0_1_stage6_frozen_roi",
        "target_type": str(target_type or "unknown"),
        "profile": _profile_name(target_type, profile_name),
        "gates": {},
        "issues": ["presentation_quality_measurement_unavailable"],
    }
    try:
        reference_array = np.asarray(reference)
        candidate_array = np.asarray(candidate)
        if reference_array.size == 0 or candidate_array.size == 0:
            raise ValueError("presentation reference/final pixels are empty")
        if reference_array.shape != candidate_array.shape:
            raise ValueError(
                f"presentation reference/final shape mismatch: "
                f"{reference_array.shape}!={candidate_array.shape}"
            )
        if not np.all(np.isfinite(reference_array)) or not np.all(
            np.isfinite(candidate_array)
        ):
            raise ValueError("presentation reference/final contains non-finite pixels")

        reference_metrics = stage7_stretch_metrics.measure_frozen_rendition_metrics(
            reference_array,
            dict(masks or {}),
        )
        candidate_metrics = stage7_stretch_metrics.measure_frozen_rendition_metrics(
            candidate_array,
            dict(masks or {}),
        )
        if reference_metrics.get("status") != "available":
            raise ValueError("Stage7 presentation reference metrics unavailable")
        if candidate_metrics.get("status") != "available":
            raise ValueError("Stage10 presentation metrics unavailable")

        color = _subject_color_report(
            reference_array,
            candidate_array,
            masks,
            cfg,
        )
        brightness = _presentation_brightness_report(
            candidate_metrics,
            reference_metrics,
            cfg,
            profile_name=report["profile"],
        )
        reference_values = dict(reference_metrics.get("metrics") or {})
        candidate_values = dict(candidate_metrics.get("metrics") or {})

        def ratio(name: str) -> Optional[float]:
            source = _finite(reference_values.get(name))
            output = _finite(candidate_values.get(name))
            if source is None or output is None or source <= 1e-12:
                return None
            return output / source

        visibility_limit = _locked_lower_cfg(
            cfg,
            "stage10_presentation_visibility_retention_min",
            0.60,
            1.50,
        )
        detail_min = _locked_lower_cfg(
            cfg,
            "stage10_presentation_microdetail_retention_min",
            0.70,
            1.50,
        )
        detail_max = _locked_upper_cfg(
            cfg,
            "stage10_presentation_microdetail_growth_max",
            1.60,
            1.0,
        )
        visibility_ratio = ratio("visibility")
        overall_detail_ratio = ratio("microcontrast")
        visibility = _lower_gate(visibility_ratio, visibility_limit)
        detail = _same_content_microdetail_report(
            reference_array,
            candidate_array,
            masks,
            stars_required=stars_required,
            retention_min=detail_min,
            growth_max=detail_max,
        )
        detail_ratio = _finite((detail.get("metrics") or {}).get("retention"))
        stars = _stars_report(
            cfg,
            stage9_quality,
            stars_required=stars_required,
            stars_not_required_verified=stars_not_required_verified,
        )
        scientific = dict(scientific_report or {})
        scientific_accepted = bool(
            str(scientific.get("status") or "").strip().lower() == "ok"
            and str(scientific.get("final_quality") or "").strip().lower() == "ok"
            and scientific.get("needs_conservative_rerun") is False
            and not list(scientific.get("issues") or [])
        )
        scientific_gate = {
            "status": "ok" if scientific_accepted else "rejected",
            "accepted": scientific_accepted,
            "report_schema": scientific.get("schema"),
            "spatial_background": dict(
                scientific.get("spatial_background_gradient") or {}
            ),
        }
        gates = {
            "scientific": scientific_gate,
            "color": color,
            "visibility": visibility,
            "brightness": brightness,
            "microdetail": detail,
            "stars": stars,
        }
        accepted = bool(
            scientific_accepted
            and color.get("accepted") is True
            and visibility.get("accepted") is True
            and brightness.get("formal_floor_passed") is True
            and detail.get("accepted") is True
            and stars.get("accepted") is True
        )
        issues = [
            name
            for name, gate in gates.items()
            if not bool(
                gate.get("accepted")
                if name != "brightness"
                else gate.get("formal_floor_passed")
            )
        ]
        report.update(
            status="ok" if accepted else "review_required",
            accepted=accepted,
            reference_pixel_sha256=stage7_stretch_metrics.stage7_pixel_sha256(
                reference_array
            ),
            candidate_pixel_sha256=stage7_stretch_metrics.stage7_pixel_sha256(
                candidate_array
            ),
            reference_metrics=reference_metrics,
            candidate_metrics=candidate_metrics,
            metrics={
                "visibility_retention": visibility_ratio,
                "microdetail_retention": detail_ratio,
                "overall_rendition_microdetail_retention_diagnostic": (
                    overall_detail_ratio
                ),
                "subject_p50_retention": (
                    (brightness.get("retention") or {}).get("subject_p50")
                ),
                "subject_lift_retention": (
                    (brightness.get("retention") or {}).get("subject_lift")
                ),
            },
            gates=gates,
            issues=issues,
        )
        return report
    except (IndexError, TypeError, ValueError, FloatingPointError) as error:
        report["reason"] = str(error)
        return report


def unavailable_presentation_report(reason: str) -> Dict[str, Any]:
    return {
        "schema": PRESENTATION_QUALITY_SCHEMA,
        "status": "unavailable",
        "accepted": False,
        "external_reference_used": False,
        "reference": "stage7_presentation_reference",
        "issues": ["presentation_quality_measurement_unavailable"],
        "reason": str(reason),
    }


__all__ = [
    "DELIVERY_GATES_SCHEMA",
    "PRESENTATION_QUALITY_SCHEMA",
    "STAGE7_PRESENTATION_REFERENCE_SCHEMA",
    "build_presentation_quality_report",
    "unavailable_presentation_report",
    "verify_stage7_presentation_reference",
]
