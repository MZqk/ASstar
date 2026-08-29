"""Stage 3 background extraction."""
import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from astropy.io import fits

import scene_support
import spatial_background_lineage

from background_sampling import (
    STAGE3_PROCESS_EVIDENCE_SCHEMA,
    assess_background_direction_reversal,
    assess_background_process,
    assess_compound_background_validation,
    assess_single_background_validation,
    assess_target_fidelity,
    analyze_directional_pattern_noise,
    background_span_standard_error,
    build_safe_background_samples,
    measure_background_validation,
    pattern_candidate_gate,
    project_stage3_neutral_axis_poly1,
    project_stage3_spatial_opponent_poly1,
    select_background_route,
    split_background_sample_points,
    verify_stage3_neutral_axis_persistence,
    verify_stage3_spatial_opponent_persistence,
)
from models import PipelineStage
from sirilpy.exceptions import CommandError, SirilError
try:
    from sirilpy.models import BGSample
except (ImportError, ModuleNotFoundError):
    class BGSample:  # pragma: no cover - exercised by the lightweight test runtime
        """Minimal compatibility object for tests that stub only sirilpy.exceptions."""

        def __init__(
            self,
            *,
            position,
            median=(0.0, 0.0, 0.0),
            mean=0.0,
            min=0.0,
            max=0.0,
            size=25,
            valid=True,
        ):
            self.position = position
            self.median = median
            self.mean = mean
            self.min = min
            self.max = max
            self.size = size
            self.valid = valid
from stage3_contract import (
    STAGE3_ALGORITHM_CONTRACT_VERSION,
    STAGE3_BACKGROUND_QUALITY_SCHEMA,
    STAGE3_BACKGROUND_SCORE_WEIGHTS,
    STAGE3_DIRECTIONAL_PATTERN_PENALTY_WEIGHT,
    STAGE3_DENSE_STAR_FIELD_FRACTION_MIN,
    STAGE3_FINAL_DIRTY_WARNING_MIN,
    STAGE3_FINAL_GRADIENT_RETENTION_WARNING,
    STAGE3_MAX_SOURCE_MASK_FRACTION,
    STAGE3_MIN_AXIS_SPAN_RATIO,
    STAGE3_MIN_SPATIAL_GRID_CELLS,
    STAGE3_MIN_SPATIAL_QUADRANTS,
    STAGE3_MIN_USABLE_SKY_FRACTION,
    STAGE3_MIN_VALIDATION_PATCHES,
    STAGE3_SIGNIFICANCE_SIGMA,
    normalize_stage3_gate_profile,
    stage3_gate_thresholds,
    stage3_static_contract_manifest,
)


STAGE3_SPATIAL_LINEAGE_SCHEMA = spatial_background_lineage.LINEAGE_SCHEMA


EMISSION_NEBULA_TARGET_TYPES = {
    "emission_nebula",
    "emission_nebula_widefield",
    "bright_emission_reflection_nebula",
}
DEFAULT_DIFFUSE_OBJECT_AREA_MIN = 0.15
DEFAULT_DIFFUSE_NEBULOSITY_AREA_MIN = 0.18
DEFAULT_DIFFUSE_FAINT_STRUCTURE_MIN = 0.65
DEFAULT_FAINT_NEBULA_AREA_MIN = 0.10
DEFAULT_FAINT_NEBULA_STRUCTURE_MIN = 0.40
DEFAULT_NEBULA_PRESERVATION_WEIGHT = 1.6
DEFAULT_FAINT_NEBULA_PRESERVATION_WEIGHT_MAX = 2.5
STAGE3_PRIMARY_GRAXPERT_LABEL = "GraXpert-AI BGE CPU"
STAGE3_GRAXPERT_BGE_MODEL_NAME = "model_v2_0_1"
STAGE3_GRAXPERT_BGE_MODEL_SHA256 = (
    "26d9e68370dfc079698aece805240a41782364f48c75f18ee4ff262c3f2ea8d2"
)
STAGE3_GRAXPERT_BGE_MODEL_RELATIVE = Path(
    "graxpert/bge-ai-models/model_v2_0_1/model.onnx"
)
STAGE3_GRAXPERT_SCRIPT_SHA256 = (
    "543bcce1fe1b4845ccee39c4b2a1b9d59088ca180ab5cd21e2a7d8bf014ce7fc"
)


def _stage3_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stage3_array_sha256(image: Any) -> str:
    values = np.ascontiguousarray(np.asarray(image, dtype="<f4"))
    digest = hashlib.sha256()
    digest.update(str(tuple(int(value) for value in values.shape)).encode("ascii"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _stage3_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stage3_write_spatial_background_lineage(
    pipeline,
    *,
    baseline_image: Any,
    fit_points: List[Tuple[float, float]],
    validation_points: List[Tuple[float, float]],
    patch_radius: int,
    support_mask: Optional[np.ndarray],
    projection: Dict[str, Any],
    review_required: bool,
    processing_route: str,
) -> Dict[str, Any]:
    """Persist the exact Stage 3 sky support used by downstream gradient gates."""
    report: Dict[str, Any] = {
        "schema": STAGE3_SPATIAL_LINEAGE_SCHEMA,
        "status": "unavailable",
        "accepted": False,
        "review_required": bool(review_required),
        "support_artifact": "stage3_spatial_background_support.fit",
        "issues": [],
    }
    process_dir = getattr(pipeline, "process_dir", None)
    if process_dir is None:
        report["issues"] = ["Stage 3 process directory is unavailable"]
        return report
    try:
        route = str(processing_route or "")
        if route not in {"background_correction", "verified_noop"}:
            raise ValueError("Stage 3 formal spatial lineage route is invalid")
        if review_required:
            raise ValueError(
                "review-required Stage 3 output cannot publish formal spatial lineage"
            )
        run_id = str(
            getattr(pipeline, "_run_id", "")
            or Path(str(getattr(pipeline, "work_dir", "") or "")).name
            or ""
        ).strip()
        if not run_id:
            raise ValueError("Stage 3 formal spatial lineage run_id is unavailable")
        if not fit_points or not validation_points:
            raise ValueError(
                "Stage 3 formal spatial lineage requires fit and validation samples"
            )
        source = np.asarray(baseline_image)
        channel_layout = "unknown"
        if source.ndim == 2:
            height, width = source.shape
            channel_layout = "mono_replicated_for_plane_metrics"
        elif source.ndim == 3 and source.shape[0] in (3, 4):
            height, width = source.shape[1:3]
            channel_layout = "rgb_chw"
        elif source.ndim == 3 and source.shape[-1] in (3, 4):
            height, width = source.shape[:2]
            channel_layout = "rgb_hwc"
        else:
            raise ValueError("Stage 3 spatial lineage RGB layout is invalid")
        if support_mask is None:
            raise ValueError(
                "Stage 3 formal lineage requires candidate-independent sky support"
            )
        raw_support = np.asarray(support_mask)
        if (
            raw_support.shape != (height, width)
            or raw_support.dtype.kind in {"f", "c"}
            and not bool(np.all(np.isfinite(raw_support)))
        ):
            raise ValueError("Stage 3 frozen support mask shape/content mismatch")
        frozen = np.asarray(raw_support, dtype=bool).copy()
        if int(np.count_nonzero(frozen)) < 64:
            raise ValueError("Stage 3 frozen support mask is too small")
        sample_patch_support, point_support_counts = (
            spatial_background_lineage.build_sample_patch_support(
                frozen.shape,
                [*fit_points, *validation_points],
                frozen,
                patch_radius=patch_radius,
            )
        )
        sample_patch_support_pixel_count = int(
            np.count_nonzero(sample_patch_support)
        )
        if sample_patch_support_pixel_count < 64:
            raise ValueError("Stage 3 frozen sample support is too small")
        artifact = Path(process_dir) / report["support_artifact"]
        header = fits.Header()
        header["STARSCMA"] = STAGE3_SPATIAL_LINEAGE_SCHEMA
        header["STARFIT"] = len(fit_points)
        header["STARVAL"] = len(validation_points)
        header["STARSKY"] = "FULL_SAFE"
        fits.PrimaryHDU(frozen.astype(np.uint8), header=header).writeto(
            artifact,
            overwrite=True,
            checksum=True,
        )
        canonical = Path(process_dir) / "stage3_bgremoved.fit"
        neutral = Path(process_dir) / "stage3_candidate_neutral_axis_poly1.fit"
        baseline = Path(process_dir) / "stage3_bg_input.fit"
        if not baseline.is_file() or not canonical.is_file():
            raise ValueError(
                "Stage 3 input/output artifacts are required for spatial lineage"
            )
        reference_metrics: Dict[str, Any] = {}
        with fits.open(
            canonical,
            memmap=False,
            do_not_scale_image_data=False,
        ) as hdul:
            canonical_pixels = np.asarray(hdul[0].data)
        with fits.open(
            baseline,
            memmap=False,
            do_not_scale_image_data=False,
        ) as hdul:
            baseline_pixels = np.asarray(hdul[0].data)
        baseline_pixel_sha256 = _stage3_array_sha256(baseline_pixels)
        canonical_pixel_sha256 = _stage3_array_sha256(canonical_pixels)
        if _stage3_array_sha256(source) != baseline_pixel_sha256:
            raise ValueError(
                "Stage 3 baseline artifact/input pixel identity mismatch"
            )
        if (
            route == "verified_noop"
            and baseline_pixel_sha256 != canonical_pixel_sha256
        ):
            raise ValueError(
                "verified Stage 3 no-op input/output pixel identity mismatch"
            )
        reference_pixels = canonical_pixels
        if reference_pixels.ndim == 2:
            reference_pixels = np.repeat(
                reference_pixels[None, :, :],
                3,
                axis=0,
            )
        reference_metrics = (
            spatial_background_lineage.measure_spatial_background_planes(
                reference_pixels,
                frozen,
                [*fit_points, *validation_points],
                patch_radius=patch_radius,
            )
        )
        reference_plane = {
            name: {
                "coefficients": list(metrics.get("coefficients") or []),
                "slope_span": metrics.get("slope_span"),
                "slope_significance_sigma": metrics.get(
                    "slope_significance_sigma"
                ),
            }
            for name, metrics in reference_metrics.items()
        }
        report.update(
            status="review_required" if review_required else "accepted",
            accepted=not review_required,
            run_id=run_id,
            image_shape=[int(height), int(width)],
            channel_layout=channel_layout,
            processing_route=route,
            patch_radius=int(patch_radius),
            fit_points=[[float(x), float(y)] for x, y in fit_points],
            validation_points=[
                [float(x), float(y)] for x, y in validation_points
            ],
            support_kind="candidate_independent_full_sky_mask",
            support_pixel_count=int(np.count_nonzero(frozen)),
            support_coverage=(
                float(np.count_nonzero(frozen)) / float(frozen.size)
            ),
            sample_patch_support_pixel_count=sample_patch_support_pixel_count,
            sample_patch_min_support_pixel_count=min(point_support_counts),
            support_sha256=_stage3_sha256(artifact),
            stage3_input_sha256=(
                _stage3_sha256(baseline) if baseline.is_file() else None
            ),
            stage3_input_pixel_sha256=baseline_pixel_sha256,
            neutral_checkpoint_sha256=(
                _stage3_sha256(neutral) if neutral.is_file() else None
            ),
            stage3_output_sha256=(
                _stage3_sha256(canonical) if canonical.is_file() else None
            ),
            stage3_output_pixel_sha256=canonical_pixel_sha256,
            reference_metrics=reference_metrics,
            reference_plane={
                "coordinate_system": "normalized_image_xy",
                "components": reference_plane,
                "sha256": _stage3_json_sha256(reference_plane),
            },
            projection_schema=projection.get("schema"),
            projection_reason_code=projection.get("reason_code"),
            selected_components=list(
                projection.get("selected_components") or []
            ),
            unresolved_components=list(
                projection.get("unresolved_components") or []
            ),
            issues=[],
        )
        report = spatial_background_lineage.seal_lineage(report)
        pipeline._write_stage_json(
            "stage3_spatial_background_lineage.json",
            report,
        )
        verified = spatial_background_lineage.load_lineage(Path(process_dir))
        if (
            verified.get("accepted") is not True
            or str(
                (verified.get("chain_digest") or {}).get("sha256") or ""
            )
            != str((report.get("chain_digest") or {}).get("sha256") or "")
        ):
            raise ValueError(
                "Stage 3 spatial background lineage self-verification failed: "
                + ", ".join(str(item) for item in verified.get("issues") or [])
            )
        return report
    except (OSError, TypeError, ValueError) as error:
        report.update(
            status="unavailable",
            accepted=False,
            review_required=True,
            processing_route=str(processing_route or "unknown"),
            issues=[str(error)],
        )
        try:
            pipeline._write_stage_json(
                "stage3_spatial_background_lineage.json",
                report,
            )
        except (AttributeError, OSError, TypeError, ValueError):
            pass
        return report


def _stage3_candidate_stem(label: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", label.strip().lower()).strip("_")
    return f"stage3_candidate_{safe or 'background'}"


def _stage3_background_score_components(
    before: Dict[str, Any],
    after: Dict[str, Any],
) -> Dict[str, Any]:
    """Return an auditable decomposition of the legacy ranking score."""
    before = before or {}
    after = after or {}
    before_std = max(float(before.get("bg_std", 0.0) or 0.0), 1e-7)
    after_std = float(after.get("bg_std", 0.0) or 0.0)
    dirty = float(after.get("dirty_background_score", 0.0) or 0.0)
    gradient = float(after.get("gradient_score", 0.0) or 0.0)
    chroma = float(after.get("chroma_noise_score", 0.0) or 0.0)
    color_shift = _stage3_color_shift(before, after)
    std_growth = max(0.0, after_std / before_std - 1.0)
    components = {
        "dirty_background_score": dirty,
        "gradient_score": gradient,
        "chroma_noise_score": chroma,
        "bg_std_growth": std_growth,
        "color_shift": color_shift,
    }
    weighted = {
        name: float(value) * float(STAGE3_BACKGROUND_SCORE_WEIGHTS[name])
        for name, value in components.items()
    }
    return {
        "components": components,
        "weights": dict(STAGE3_BACKGROUND_SCORE_WEIGHTS),
        "weighted_components": weighted,
        "total": sum(weighted.values()),
    }


def _stage3_quality_diagnostic_message(
    before: Any,
    after: Any,
    preservation: Optional[Dict[str, Any]],
) -> str:
    """Describe candidate diagnostics without pretending to be an acceptance gate."""
    if before is None or after is None:
        return "feature diagnostics unavailable; auditable hard gates own acceptance"
    notes: List[str] = []
    if preservation and preservation.get("available"):
        for key in (
            "star_retention_ratio",
            "target_flux_retention_ratio",
            "target_morphology_correlation",
            "target_centroid_shift_fraction",
        ):
            value = preservation.get(key)
            if value is not None:
                notes.append(f"{key}={float(value):.5f}")
    before_bg_std = float(getattr(before, "bg_std", float("nan")))
    after_bg_std = float(getattr(after, "bg_std", float("nan")))
    before_bg_median = float(getattr(before, "bg_median", float("nan")))
    after_bg_median = float(getattr(after, "bg_median", float("nan")))
    message = (
        "source-fidelity diagnostics recorded; held-out sky and pixel gates own "
        f"acceptance, bg_std {before_bg_std:.4f}->{after_bg_std:.4f}, "
        f"bg_median {before_bg_median:.4f}->{after_bg_median:.4f}"
    )
    if notes:
        message += ", " + ", ".join(notes)
    return message


def _stage3_candidate_quality_record_fields(
    *,
    gate_ok: bool,
    gate_profile: str,
    gate_warnings: List[str],
    hard_gate_metrics_available: bool,
    color_shift: float,
) -> Dict[str, Any]:
    """Build the shared ordinary/compound candidate quality audit fields."""
    warnings = list(dict.fromkeys(str(item) for item in gate_warnings if item))
    return {
        "status": (
            "accepted_with_warnings"
            if gate_ok and warnings
            else "accepted"
            if gate_ok
            else "rejected"
        ),
        "severity": (
            "soft_warning"
            if gate_ok and warnings
            else "normal"
            if gate_ok
            else "hard_rejected"
        ),
        "gate_profile": gate_profile,
        "gate_warnings": warnings,
        "hard_gate_metrics_available": bool(hard_gate_metrics_available),
        "color_shift": float(color_shift),
    }


def _stage3_color_shift(
    before: Dict[str, Any],
    after: Dict[str, Any],
) -> float:
    before = before or {}
    after = after or {}
    shifts: List[float] = []
    for key in ("red_dominance", "blue_dominance", "green_cast"):
        if key not in before or key not in after:
            continue
        before_value = float(before.get(key, 1.0) or 1.0)
        after_value = float(after.get(key, 1.0) or 1.0)
        shifts.append(abs(after_value - before_value))
    if "color_balance_score" in before and "color_balance_score" in after:
        shifts.append(
            max(
                0.0,
                float(before.get("color_balance_score", 1.0) or 1.0)
                - float(after.get("color_balance_score", 1.0) or 1.0),
            )
        )
    return max(shifts) if shifts else 0.0


def _stage3_verified_background_color_normalization(
    before: Dict[str, Any],
    candidate: Dict[str, Any],
    final_output_validation: Dict[str, Any],
    *,
    gate_profile: str,
) -> Dict[str, Any]:
    """Identify a fully evidenced Stage 3 sky-color normalization false alarm."""
    report: Dict[str, Any] = {
        "schema": "starun.stage3-verified-background-color-normalization.v1",
        "applied": False,
        "accepted": False,
        "reason_code": "not_applicable",
    }
    generic_warning = "candidate does not meet clean-output sufficiency thresholds"
    warnings = list(candidate.get("gate_warnings") or [])
    if normalize_stage3_gate_profile(gate_profile) != "output_first":
        report["reason_code"] = "strict_profile"
        return report
    if str(candidate.get("source") or "") != "builtin":
        report["reason_code"] = "non_builtin_candidate"
        return report
    if warnings != [generic_warning]:
        report["reason_code"] = "candidate_has_non_sufficiency_warnings"
        return report

    after = candidate.get("after_adaptive") or {}
    preservation = candidate.get("preservation") or {}
    score_components = candidate.get("background_score_components") or {}
    components = score_components.get("components") or {}
    weighted = score_components.get("weighted_components") or {}
    raw_score = float(candidate.get("score", math.inf) or math.inf)
    color_shift = float(components.get("color_shift", math.inf) or math.inf)
    color_penalty = float(weighted.get("color_shift", math.inf) or math.inf)
    effective_score = raw_score - color_penalty
    max_score = float(
        stage3_gate_thresholds("output_first")[
            "sufficient_max_background_score"
        ]
    )

    def finite_metric(mapping: Dict[str, Any], key: str, default: float) -> float:
        try:
            value = float(mapping.get(key, default))
        except (TypeError, ValueError):
            return default
        return value if math.isfinite(value) else default

    def neutral_distance(mapping: Dict[str, Any]) -> float:
        return sum(
            abs(finite_metric(mapping, key, math.inf) - 1.0)
            for key in ("red_dominance", "blue_dominance", "green_cast")
        )

    before_std = max(finite_metric(before, "bg_std", math.inf), 1.0e-7)
    after_std = finite_metric(after, "bg_std", math.inf)
    before_gradient = finite_metric(before, "gradient_score", math.inf)
    after_gradient = finite_metric(after, "gradient_score", math.inf)
    before_neutrality = neutral_distance(before)
    after_neutrality = neutral_distance(after)
    final_pixel_gate = final_output_validation.get("pixel_integrity_gate") or {}
    gate_checks = {
        name: bool((candidate.get(name) or {}).get("accepted", False))
        for name in (
            "pixel_integrity_gate",
            "target_fidelity_gate",
            "validation_gate",
            "pattern_quality_gate",
        )
    }
    thresholds = {
        "raw_score_min": max_score,
        "effective_score_max": max_score,
        "color_shift_min": float(
            stage3_gate_thresholds("output_first")[
                "sufficient_color_shift_max"
            ]
        ),
        "neutral_distance_ratio_max": 0.50,
        "gradient_after_max": 0.04,
        "gradient_retention_ratio_max": 0.50,
        "dirty_background_score_max": 0.05,
        "chroma_noise_score_max": 0.05,
        "bg_std_growth_max": 1.0,
        "target_flux_retention_min": 0.95,
        "target_flux_retention_max": 1.05,
        "target_morphology_correlation_min": 0.98,
        "target_centroid_shift_fraction_max": 0.02,
        "target_change_residual_significance_max": 1.0,
    }
    evidence = {
        "raw_score": raw_score,
        "color_shift": color_shift,
        "color_shift_penalty": color_penalty,
        "effective_score_without_verified_color_normalization": effective_score,
        "before_neutral_distance": before_neutrality,
        "after_neutral_distance": after_neutrality,
        "neutral_distance_ratio": (
            after_neutrality / before_neutrality
            if before_neutrality > 1.0e-12
            else math.inf
        ),
        "gradient_before": before_gradient,
        "gradient_after": after_gradient,
        "dirty_background_score_after": finite_metric(
            after, "dirty_background_score", math.inf
        ),
        "chroma_noise_score_after": finite_metric(
            after, "chroma_noise_score", math.inf
        ),
        "bg_std_growth": after_std / before_std,
        "color_balance_before": finite_metric(
            before, "color_balance_score", -math.inf
        ),
        "color_balance_after": finite_metric(
            after, "color_balance_score", -math.inf
        ),
        "target_flux_retention_ratio": finite_metric(
            preservation, "target_flux_retention_ratio", math.inf
        ),
        "target_morphology_correlation": finite_metric(
            preservation, "target_morphology_correlation", -math.inf
        ),
        "target_centroid_shift_fraction": finite_metric(
            preservation, "target_centroid_shift_fraction", math.inf
        ),
        "target_change_residual_significance": finite_metric(
            preservation, "target_change_residual_significance", math.inf
        ),
        "candidate_gate_checks": gate_checks,
        "final_output_accepted": bool(
            final_output_validation.get("accepted", False)
        ),
        "final_output_severity": str(
            final_output_validation.get("severity") or ""
        ),
        "final_pixel_gate_accepted": bool(final_pixel_gate.get("accepted", False)),
    }
    issues: List[str] = []
    if not all(gate_checks.values()):
        issues.append("candidate_quality_gate_not_fully_accepted")
    if not bool(candidate.get("hard_gate_metrics_available", False)):
        issues.append("candidate_hard_gate_metrics_unavailable")
    if not (
        evidence["final_output_accepted"]
        and evidence["final_output_severity"] == "normal"
        and evidence["final_pixel_gate_accepted"]
    ):
        issues.append("final_saved_output_not_normally_accepted")
    if not raw_score > thresholds["raw_score_min"]:
        issues.append("raw_score_was_not_color_shift_limited")
    if not effective_score <= thresholds["effective_score_max"]:
        issues.append("non_color_background_score_remains_high")
    if not color_shift > thresholds["color_shift_min"]:
        issues.append("color_shift_did_not_trigger_sufficiency")
    if not (
        evidence["neutral_distance_ratio"]
        <= thresholds["neutral_distance_ratio_max"]
        and evidence["color_balance_after"] >= evidence["color_balance_before"]
    ):
        issues.append("sky_color_did_not_converge_toward_neutral")
    if not (
        after_gradient <= thresholds["gradient_after_max"]
        and after_gradient
        <= before_gradient * thresholds["gradient_retention_ratio_max"]
    ):
        issues.append("held_out_gradient_not_materially_reduced")
    if not (
        evidence["dirty_background_score_after"]
        <= thresholds["dirty_background_score_max"]
        and evidence["chroma_noise_score_after"]
        <= thresholds["chroma_noise_score_max"]
        and evidence["bg_std_growth"] <= thresholds["bg_std_growth_max"]
    ):
        issues.append("clean_background_evidence_failed")
    if not (
        thresholds["target_flux_retention_min"]
        <= evidence["target_flux_retention_ratio"]
        <= thresholds["target_flux_retention_max"]
        and evidence["target_morphology_correlation"]
        >= thresholds["target_morphology_correlation_min"]
        and evidence["target_centroid_shift_fraction"]
        <= thresholds["target_centroid_shift_fraction_max"]
        and evidence["target_change_residual_significance"]
        <= thresholds["target_change_residual_significance_max"]
    ):
        issues.append("target_signal_preservation_evidence_failed")

    report.update(
        accepted=not issues,
        applied=not issues,
        reason_code=(
            "verified_background_color_normalization" if not issues else issues[0]
        ),
        evidence=evidence,
        thresholds=thresholds,
        issues=issues,
        cleared_warning=(generic_warning if not issues else None),
    )
    return report


def _stage3_color_normalization_candidate_evidence(
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    """Let a strongly preserved built-in candidate enter the clean tier.

    This does not clear review by itself.  The selected, saved output must still
    pass :func:`_stage3_verified_background_color_normalization` before the
    generic sufficiency warning can be removed.
    """
    generic_warning = "candidate does not meet clean-output sufficiency thresholds"
    report: Dict[str, Any] = {
        "schema": "starun.stage3-color-normalization-candidate-evidence.v1",
        "eligible": False,
        "reason_code": "not_applicable",
    }
    if str(candidate.get("source") or "") != "builtin":
        report["reason_code"] = "non_builtin_candidate"
        return report
    if list(candidate.get("gate_warnings") or []) != [generic_warning]:
        report["reason_code"] = "candidate_has_non_sufficiency_warnings"
        return report

    def finite(mapping: Dict[str, Any], key: str, default: float) -> float:
        try:
            value = float(mapping.get(key, default))
        except (TypeError, ValueError):
            return default
        return value if math.isfinite(value) else default

    after = candidate.get("after_adaptive") or {}
    preservation = candidate.get("preservation") or {}
    score_components = candidate.get("background_score_components") or {}
    components = score_components.get("components") or {}
    weighted = score_components.get("weighted_components") or {}
    raw_score = finite(candidate, "score", math.inf)
    color_shift = finite(components, "color_shift", math.inf)
    color_penalty = finite(weighted, "color_shift", math.inf)
    effective_score = raw_score - color_penalty
    thresholds = {
        "background_score_max": float(
            stage3_gate_thresholds("output_first")[
                "sufficient_max_background_score"
            ]
        ),
        "color_shift_min": float(
            stage3_gate_thresholds("output_first")[
                "sufficient_color_shift_max"
            ]
        ),
        "gradient_after_max": 0.04,
        "dirty_background_score_max": 0.05,
        "chroma_noise_score_max": 0.05,
        "target_flux_retention_min": 0.95,
        "target_flux_retention_max": 1.05,
        "target_morphology_correlation_min": 0.98,
        "target_centroid_shift_fraction_max": 0.02,
        "target_change_residual_significance_max": 1.0,
    }
    gate_checks = {
        name: bool((candidate.get(name) or {}).get("accepted", False))
        for name in (
            "pixel_integrity_gate",
            "target_fidelity_gate",
            "validation_gate",
            "pattern_quality_gate",
        )
    }
    evidence = {
        "raw_score": raw_score,
        "color_shift": color_shift,
        "color_shift_penalty": color_penalty,
        "effective_score_without_color_shift": effective_score,
        "gradient_after": finite(after, "gradient_score", math.inf),
        "dirty_background_score_after": finite(
            after, "dirty_background_score", math.inf
        ),
        "chroma_noise_score_after": finite(
            after, "chroma_noise_score", math.inf
        ),
        "target_flux_retention_ratio": finite(
            preservation, "target_flux_retention_ratio", math.inf
        ),
        "target_morphology_correlation": finite(
            preservation, "target_morphology_correlation", -math.inf
        ),
        "target_centroid_shift_fraction": finite(
            preservation, "target_centroid_shift_fraction", math.inf
        ),
        "target_change_residual_significance": finite(
            preservation, "target_change_residual_significance", math.inf
        ),
        "candidate_gate_checks": gate_checks,
    }
    issues: List[str] = []
    if not all(gate_checks.values()):
        issues.append("candidate_quality_gate_not_fully_accepted")
    if not bool(candidate.get("hard_gate_metrics_available", False)):
        issues.append("candidate_hard_gate_metrics_unavailable")
    if not (
        raw_score > thresholds["background_score_max"]
        and effective_score <= thresholds["background_score_max"]
        and color_shift > thresholds["color_shift_min"]
    ):
        issues.append("generic_warning_not_isolated_to_color_shift")
    if not (
        evidence["gradient_after"] <= thresholds["gradient_after_max"]
        and evidence["dirty_background_score_after"]
        <= thresholds["dirty_background_score_max"]
        and evidence["chroma_noise_score_after"]
        <= thresholds["chroma_noise_score_max"]
    ):
        issues.append("candidate_background_not_clean")
    if not (
        thresholds["target_flux_retention_min"]
        <= evidence["target_flux_retention_ratio"]
        <= thresholds["target_flux_retention_max"]
        and evidence["target_morphology_correlation"]
        >= thresholds["target_morphology_correlation_min"]
        and evidence["target_centroid_shift_fraction"]
        <= thresholds["target_centroid_shift_fraction_max"]
        and evidence["target_change_residual_significance"]
        <= thresholds["target_change_residual_significance_max"]
    ):
        issues.append("candidate_target_signal_not_strongly_preserved")

    report.update(
        eligible=not issues,
        reason_code=(
            "candidate_color_normalization_evidence_ready"
            if not issues
            else issues[0]
        ),
        evidence=evidence,
        thresholds=thresholds,
        issues=issues,
        final_saved_output_revalidation_required=True,
    )
    return report


def _stage3_policy_float(
    stage3_policy: Optional[Dict[str, Any]],
    key: str,
    default: float,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    policy = stage3_policy or {}
    try:
        value = float(policy.get(key, default) if isinstance(policy, dict) else default)
    except (TypeError, ValueError):
        value = default
    if minimum is not None:
        value = max(float(minimum), value)
    if maximum is not None:
        value = min(float(maximum), value)
    return value


def _stage3_nebula_preservation_weight(
    diffuse_context: Optional[Dict[str, Any]] = None,
    stage3_policy: Optional[Dict[str, Any]] = None,
) -> float:
    base_weight = _stage3_policy_float(
        stage3_policy,
        "nebula_preservation_penalty_weight",
        DEFAULT_NEBULA_PRESERVATION_WEIGHT,
        minimum=0.0,
        maximum=10.0,
    )
    max_weight = _stage3_policy_float(
        stage3_policy,
        "faint_nebula_preservation_penalty_weight_max",
        DEFAULT_FAINT_NEBULA_PRESERVATION_WEIGHT_MAX,
        minimum=0.0,
        maximum=10.0,
    )
    max_weight = max(base_weight, max_weight)
    context = diffuse_context or {}
    try:
        faint_structure_score = float(context.get("faint_structure_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        faint_structure_score = 0.0
    faint_min = _stage3_policy_float(
        stage3_policy,
        "faint_nebula_structure_min",
        DEFAULT_FAINT_NEBULA_STRUCTURE_MIN,
        minimum=0.0,
        maximum=1.0,
    )
    if not context.get("faint_nebula_protection") and faint_structure_score <= faint_min:
        return base_weight
    span = max(1e-6, 1.0 - faint_min)
    t = max(0.0, min(1.0, (faint_structure_score - faint_min) / span))
    return base_weight + (max_weight - base_weight) * t


def _stage3_preservation_penalty(
    preservation: Dict[str, Any],
    *,
    diffuse_context: Optional[Dict[str, Any]] = None,
    stage3_policy: Optional[Dict[str, Any]] = None,
    nebula_weight: Optional[float] = None,
    gate_profile: str = "output_first",
) -> float:
    if not isinstance(preservation, dict) or not preservation.get("available"):
        return 0.0

    penalty = 0.0
    nebula_weight = (
        float(nebula_weight)
        if nebula_weight is not None
        else _stage3_nebula_preservation_weight(diffuse_context, stage3_policy)
    )
    target_flux_retention = preservation.get("target_flux_retention_ratio")
    if target_flux_retention is not None:
        try:
            retention = float(target_flux_retention)
            if normalize_stage3_gate_profile(gate_profile) == "strict":
                penalty += max(0.0, 1.0 - retention) * nebula_weight
            else:
                penalty += abs(1.0 - retention) * nebula_weight
        except (TypeError, ValueError):
            pass
    else:
        # Compatibility for older callers that have not supplied held-out sky
        # referenced target flux metrics.
        nebula_change = preservation.get("nebula_mean_change_ratio")
        if nebula_change is not None:
            try:
                change = max(0.0, float(nebula_change))
                penalty += max(0.0, change - 0.015) * nebula_weight
            except (TypeError, ValueError):
                pass

    morphology = preservation.get("target_morphology_correlation")
    if morphology is not None:
        try:
            penalty += max(0.0, 1.0 - float(morphology)) * 0.75
        except (TypeError, ValueError):
            pass

    star_retention = preservation.get("star_retention_ratio")
    if star_retention is not None:
        try:
            retention = float(star_retention)
            penalty += max(0.0, 1.0 - retention) * 0.45
        except (TypeError, ValueError):
            pass
    return penalty


def _stage3_candidate_sufficient(
    before: Dict[str, Any],
    after: Dict[str, Any],
    score: float,
    stage3_policy: Optional[Dict[str, Any]] = None,
    gate_profile: str = "output_first",
) -> bool:
    before = before or {}
    after = after or {}
    profile = normalize_stage3_gate_profile(gate_profile)
    thresholds = stage3_gate_thresholds(profile)
    strict = bool(thresholds.get("strict_legacy"))
    max_score = (
        _stage3_policy_float(
            stage3_policy,
            "sufficient_max_background_score",
            float(thresholds["sufficient_max_background_score"]),
            minimum=0.0,
        )
        if strict
        else float(thresholds["sufficient_max_background_score"])
    )
    dirty_max = (
        _stage3_policy_float(
            stage3_policy,
            "sufficient_dirty_score_max",
            float(thresholds["sufficient_dirty_score_max"]),
            minimum=0.0,
        )
        if strict
        else float(thresholds["sufficient_dirty_score_max"])
    )
    dirty_gradient_ratio = _stage3_policy_float(
        stage3_policy,
        "sufficient_dirty_gradient_retention_ratio",
        0.88,
        minimum=0.0,
        maximum=1.5,
    )
    dirty_gradient_floor = _stage3_policy_float(
        stage3_policy,
        "sufficient_dirty_gradient_floor",
        0.04,
        minimum=0.0,
    )
    initial_gradient_min = _stage3_policy_float(
        stage3_policy,
        "sufficient_initial_gradient_min",
        0.06,
        minimum=0.0,
    )
    high_gradient_ratio = _stage3_policy_float(
        stage3_policy,
        "sufficient_high_gradient_retention_ratio",
        0.96,
        minimum=0.0,
        maximum=1.5,
    )
    max_std_growth = (
        _stage3_policy_float(
            stage3_policy,
            "max_bg_std_growth",
            float(thresholds["sufficient_max_bg_std_growth"]),
            minimum=1.0,
            maximum=2.0,
        )
        if strict
        else float(thresholds["sufficient_max_bg_std_growth"])
    )
    color_shift_max = (
        _stage3_policy_float(
            stage3_policy,
            "sufficient_color_shift_max",
            float(thresholds["sufficient_color_shift_max"]),
            minimum=0.0,
            maximum=2.0,
        )
        if strict
        else float(thresholds["sufficient_color_shift_max"])
    )
    dirty = float(after.get("dirty_background_score", 0.0) or 0.0)
    gradient_before = float(before.get("gradient_score", 0.0) or 0.0)
    gradient_after = float(after.get("gradient_score", 0.0) or 0.0)
    before_std = max(float(before.get("bg_std", 0.0) or 0.0), 1e-7)
    after_std = float(after.get("bg_std", 0.0) or 0.0)
    color_shift = _stage3_color_shift(before, after)
    sufficient = score <= max_score and not (
        dirty > dirty_max and gradient_after >= max(gradient_before * dirty_gradient_ratio, dirty_gradient_floor)
    ) and not (
        gradient_before >= initial_gradient_min and gradient_after > gradient_before * high_gradient_ratio
    ) and not (
        after_std / before_std > max_std_growth
    ) and not (
        color_shift > color_shift_max
    )
    if strict:
        return sufficient
    return bool(
        score <= max_score
        and dirty <= dirty_max
        and after_std / before_std <= max_std_growth
        and color_shift <= color_shift_max
    )


def _stage3_quote_arg(pipeline, value: Path | str) -> str:
    if hasattr(pipeline, "_quote_siril_arg"):
        return pipeline._quote_siril_arg(value)
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def _stage3_find_script(pipeline, *relative_candidates: str) -> Optional[Path]:
    if hasattr(pipeline, "_find_plugin_script"):
        try:
            found = pipeline._find_plugin_script(tuple(relative_candidates))
            if found is not None:
                return found
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            if hasattr(pipeline, "log"):
                pipeline.log.debug(f"stage3 plugin script lookup skipped: {exc}")
    scripts_root = None
    if hasattr(pipeline, "_resolve_siril_scripts_root"):
        try:
            scripts_root = pipeline._resolve_siril_scripts_root()
        except (OSError, RuntimeError, TypeError, ValueError):
            scripts_root = None
    if scripts_root is None:
        plugin_dir = getattr(pipeline, "siril_plugin_dir", None)
        if plugin_dir:
            root = Path(plugin_dir)
            for candidate_root in (
                root / "vendor" / "siril-scripts",
                root / "vendor" / "siril-scripts" / "siril-scripts",
            ):
                if (candidate_root / "processing").is_dir():
                    scripts_root = candidate_root
                    break
    if scripts_root is None:
        return None
    for rel in relative_candidates:
        candidate = Path(scripts_root) / rel
        if candidate.is_file():
            return candidate
    return None


def _stage3_ensure_graxpert_bge_model(pipeline) -> bool:
    plugin_dir = getattr(pipeline, "siril_plugin_dir", None)
    if not plugin_dir:
        return False
    plugin_root = Path(plugin_dir)
    source = plugin_root / STAGE3_GRAXPERT_BGE_MODEL_RELATIVE
    if not source.is_file():
        if hasattr(pipeline, "log"):
            pipeline.log.warn(
                "Stage3 GraXpert BGE model missing: "
                f"{STAGE3_GRAXPERT_BGE_MODEL_RELATIVE}"
            )
        return False
    try:
        source_sha = _stage3_sha256(source)
    except OSError as exc:
        if hasattr(pipeline, "log"):
            pipeline.log.warn(f"Stage3 GraXpert BGE model hash failed: {exc}")
        return False
    if source_sha != STAGE3_GRAXPERT_BGE_MODEL_SHA256:
        if hasattr(pipeline, "log"):
            pipeline.log.warn(
                "Stage3 GraXpert BGE model checksum mismatch: "
                f"expected={STAGE3_GRAXPERT_BGE_MODEL_SHA256} actual={source_sha}"
            )
        return False

    target = (
        Path(os.path.expanduser("~/Library/Application Support/GraXpert"))
        / "bge-ai-models"
        / STAGE3_GRAXPERT_BGE_MODEL_NAME
        / "model.onnx"
    )
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target_sha = _stage3_sha256(target) if target.is_file() else ""
        if target_sha != STAGE3_GRAXPERT_BGE_MODEL_SHA256:
            shutil.copy2(source, target)
            if hasattr(pipeline, "log"):
                pipeline.log.info(f"Stage3 GraXpert BGE model installed: {target}")
        installed_sha = _stage3_sha256(target) if target.is_file() else ""
        if installed_sha != STAGE3_GRAXPERT_BGE_MODEL_SHA256:
            raise OSError("installed model checksum mismatch")
        pipeline._stage3_graxpert_provenance = {
            "backend": "graxpert_bge",
            "model_name": STAGE3_GRAXPERT_BGE_MODEL_NAME,
            "model_sha256": STAGE3_GRAXPERT_BGE_MODEL_SHA256,
            "model_source": str(source),
            "model_runtime_path": str(target),
            "correction": "subtraction",
            "smoothing": 0.5,
            "compute": "cpu",
            "background_model_artifact": None,
        }
        return True
    except (OSError, RuntimeError, shutil.Error) as exc:
        if hasattr(pipeline, "log"):
            pipeline.log.warn(f"Stage3 GraXpert BGE model install failed: {exc}")
        return False


def _stage3_graxpert_candidates(pipeline=None) -> List[Tuple[str, Tuple[str, ...], str]]:
    if pipeline is not None:
        script = _stage3_find_script(
            pipeline,
            "processing/GraXpert-AI.py",
        )
        if script is not None and _stage3_ensure_graxpert_bge_model(pipeline):
            try:
                script_sha = _stage3_sha256(script)
                if script_sha != STAGE3_GRAXPERT_SCRIPT_SHA256:
                    if hasattr(pipeline, "log"):
                        pipeline.log.warn(
                            "Stage3 GraXpert script checksum mismatch: "
                            f"expected={STAGE3_GRAXPERT_SCRIPT_SHA256} "
                            f"actual={script_sha}"
                        )
                    return []
                pipeline._stage3_graxpert_provenance["script_path"] = str(script)
                pipeline._stage3_graxpert_provenance["script_sha256"] = script_sha
            except OSError as exc:
                if hasattr(pipeline, "log"):
                    pipeline.log.warn(f"Stage3 GraXpert script hash failed: {exc}")
                return []
            script_arg = _stage3_quote_arg(pipeline, script)
            return [
                (
                    STAGE3_PRIMARY_GRAXPERT_LABEL,
                    (
                        "pyscript",
                        script_arg,
                        "-bge",
                        "-model",
                        STAGE3_GRAXPERT_BGE_MODEL_NAME,
                        "-correction",
                        "subtraction",
                        "-keep_bg",
                        "-nogpu",
                    ),
                    "graxpert",
                ),
            ]
        return []
    return []


def _stage3_capture_graxpert_background_model(
    pipeline,
    *,
    label: str,
) -> Optional[str]:
    """Normalize the background model emitted by GraXpert ``-keep_bg``."""
    process_dir = getattr(pipeline, "process_dir", None)
    if process_dir is None:
        return None
    process_root = Path(process_dir)
    candidates: List[Path] = []
    getter = getattr(getattr(pipeline, "siril", None), "get_image_filename", None)
    if callable(getter):
        try:
            current = Path(str(getter()))
            candidates.extend(
                current.with_name(f"{current.stem}_bg{suffix}")
                for suffix in (current.suffix, ".fit", ".fits", ".fts")
            )
        except (OSError, RuntimeError, TypeError, ValueError):
            pass
    for stem in ("stage3_bg_input", "stage3_bgremoved"):
        candidates.extend(
            process_root / f"{stem}_bg{suffix}"
            for suffix in (".fit", ".fits", ".fts")
        )
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        if hasattr(pipeline, "log"):
            pipeline.log.warn(
                f"Stage3 GraXpert background model missing after {label}"
            )
        return None
    destination = process_root / (
        f"{_stage3_candidate_stem(label)}_background{source.suffix.lower()}"
    )
    try:
        if destination.exists() and destination != source:
            destination.unlink()
        if destination != source:
            source.replace(destination)
        return str(destination)
    except OSError as exc:
        if hasattr(pipeline, "log"):
            pipeline.log.warn(
                f"Stage3 GraXpert background model normalization failed: {exc}"
            )
        return None


def _stage3_finalize_graxpert_background_model(
    pipeline,
    selected: Dict[str, Any],
) -> Optional[str]:
    raw_path = selected.get("background_model_artifact")
    if not raw_path:
        return None
    source = Path(str(raw_path))
    if not source.is_file():
        return None
    destination = source.with_name(f"stage3_background_model{source.suffix.lower()}")
    try:
        if destination.exists() and destination != source:
            destination.unlink()
        if destination != source:
            source.replace(destination)
        selected["background_model_artifact"] = str(destination)
        provenance = getattr(pipeline, "_stage3_graxpert_provenance", None)
        if isinstance(provenance, dict):
            provenance["background_model_artifact"] = str(destination)
        return str(destination)
    except OSError as exc:
        if hasattr(pipeline, "log"):
            pipeline.log.warn(
                f"Stage3 GraXpert selected background model finalize failed: {exc}"
            )
        return None


def _stage3_theoretical_plugin_candidates(pipeline=None) -> List[Tuple[str, Tuple[str, ...], str]]:
    # Order by expected background-model quality first, then by automation risk.
    # Every candidate still goes through the same Stage3 quality gate.
    autobge = None
    if pipeline is not None:
        autobge = _stage3_find_script(pipeline, "processing/AutoBGE.py")
    if autobge is not None:
        script_arg = _stage3_quote_arg(pipeline, autobge)
        background_plugins = [
            ("ADBE", ("pyscript", script_arg, "-npoints", "80", "-polydegree", "2", "-rbfsmooth", "0.08"), "plugin"),
            ("DBE", ("pyscript", script_arg, "-npoints", "120", "-polydegree", "3", "-rbfsmooth", "0.12"), "plugin"),
            ("AutoDBE", ("pyscript", script_arg, "-npoints", "100", "-polydegree", "2", "-rbfsmooth", "0.10"), "plugin"),
        ]
    else:
        background_plugins = []
    return [
        *_stage3_graxpert_candidates(pipeline),
        *background_plugins,
    ]


def _stage3_background_candidate_chain(
    pipeline,
    *,
    rbf_attempts: List[Tuple[str, Tuple[str, ...], str]],
    poly_attempt: List[Tuple[str, Tuple[str, ...], str]],
    poly_first: bool,
) -> Tuple[List[Tuple[str, Tuple[str, ...], str]], List[str], str]:
    plugin_attempts = _stage3_theoretical_plugin_candidates(pipeline)
    primary_attempts = [
        record
        for record in plugin_attempts
        if record[2] == "graxpert"
    ]
    backup_plugin_attempts = [
        record
        for record in plugin_attempts
        if record[2] != "graxpert"
    ]
    if poly_first:
        builtin_attempts = poly_attempt + rbf_attempts
        builtin_order_reason = "diffuse_signal_safe_samples_poly_before_rbf"
    else:
        builtin_attempts = rbf_attempts + poly_attempt
        builtin_order_reason = "safe_samples_rbf_before_poly"

    cfg = getattr(pipeline, "cfg", None)
    backend_policy = str(
        getattr(cfg, "stage3_backend_policy", "auto_chain")
    )
    plugin_fallback_enabled = bool(
        getattr(cfg, "stage3_plugin_fallback_enabled", True)
    )
    if backend_policy == "graxpert_only":
        primary_attempts = [
            record for record in plugin_attempts if record[2] == "graxpert"
        ]
        builtin_attempts = []
        backup_plugin_attempts = []
        chain = primary_attempts
    elif backend_policy == "builtin_only":
        primary_attempts = []
        backup_plugin_attempts = []
        chain = builtin_attempts
    elif not plugin_fallback_enabled:
        primary_attempts = []
        backup_plugin_attempts = []
        chain = builtin_attempts
    else:
        # Audited built-in models own the default route. GraXpert and the
        # remaining external plugins are conditional output-oriented backups.
        chain = builtin_attempts + primary_attempts + backup_plugin_attempts

    seen = set()
    ordered: List[Tuple[str, Tuple[str, ...], str]] = []
    for label, command, source in chain:
        key = (label, command)
        if key in seen:
            continue
        seen.add(key)
        ordered.append((label, command, source))

    attempt_limit = int(
        getattr(cfg, "stage3_candidate_attempt_limit", 0) or 0
    )
    if attempt_limit > 0:
        ordered = ordered[:attempt_limit]

    if hasattr(pipeline, "log"):
        pipeline.log.info(
            "[Stage3] Background extraction chain: "
            + " -> ".join(label for label, _command, _source in ordered)
        )
    return ordered, [record[0] for record in builtin_attempts], builtin_order_reason


def _stage3_is_graxpert_attempt(label: str, command: Tuple[str, ...], source: str) -> bool:
    if str(source).lower() == "graxpert":
        return True
    text = " ".join(str(part) for part in (label, *command)).lower()
    return "graxpert-ai.py" in text or "graxpert" in text or "gxp" in text


def _stage3_graxpert_runtime_error_reason(
    error: Exception,
    command: Tuple[str, ...],
) -> Optional[str]:
    text = str(error).strip()
    lowered = text.lower()
    command_text = " ".join(str(part) for part in command).lower()
    runtime_markers = (
        "graxpert-ai.py",
        "graxpert ai",
        "background extraction",
        "too many indices for array",
        "onnx",
        "onnxruntime",
        "error initializing application",
        "traceback",
        "model_v2_0_1",
    )
    if "graxpert-ai.py" in command_text or any(marker in lowered for marker in runtime_markers):
        return f"graxpert_runtime_error: {text or type(error).__name__}"
    return None


def _stage3_pyscript_path(command: Tuple[str, ...]) -> Optional[Path]:
    if len(command) < 2 or str(command[0]).lower() != "pyscript":
        return None
    raw_path = str(command[1]).strip()
    if len(raw_path) >= 2 and raw_path[0] == raw_path[-1] == '"':
        raw_path = raw_path[1:-1]
    raw_path = raw_path.replace('\\"', '"').replace("\\\\", "\\")
    return Path(raw_path) if raw_path else None


def _stage3_image_fingerprint(pipeline) -> Optional[str]:
    if not hasattr(pipeline, "_current_image_fingerprint"):
        return None
    try:
        return pipeline._current_image_fingerprint()
    except (CommandError, SirilError, OSError, RuntimeError, TypeError, ValueError) as exc:
        if hasattr(pipeline, "log"):
            pipeline.log.debug(f"stage3 image fingerprint skipped: {exc}")
        return None


def _stage3_candidate_pixel_gate(
    baseline_image: Any,
    candidate_image: Any,
    *,
    gate_profile: str = "output_first",
) -> Tuple[bool, Dict[str, Any]]:
    """Hard-reject unusable or byte-for-byte unchanged candidate pixels."""
    profile = normalize_stage3_gate_profile(gate_profile)
    thresholds = stage3_gate_thresholds(profile)
    if baseline_image is None:
        return True, {
            "status": "not_enforced",
            "accepted": True,
            "severity": "normal",
            "warnings": [],
            "hard_issues": [],
            "issues": [],
            "profile": profile,
            "effective_thresholds": thresholds,
            "reason": "baseline pixels are unavailable for legacy caller",
        }
    hard_issues: List[str] = []
    try:
        baseline = np.asarray(baseline_image)
        candidate = np.asarray(candidate_image)
    except (TypeError, ValueError) as error:
        hard_issues.append(f"candidate pixels are unreadable: {error}")
        baseline = np.asarray([])
        candidate = np.asarray([])
    if not hard_issues:
        if baseline.ndim < 2 or candidate.ndim < 2:
            hard_issues.append("candidate image dimensions are invalid")
        elif candidate.shape != baseline.shape:
            hard_issues.append(
                "candidate image dimensions changed "
                f"({tuple(candidate.shape)} != {tuple(baseline.shape)})"
            )
        elif not bool(np.all(np.isfinite(candidate))):
            hard_issues.append("candidate image contains non-finite pixels")
        elif not bool(np.all(np.isfinite(baseline))):
            hard_issues.append("Stage 3 baseline contains non-finite pixels")
        elif bool(np.array_equal(candidate, baseline)):
            hard_issues.append("candidate command did not change any pixels")
    accepted = not hard_issues
    return accepted, {
        "status": "accepted" if accepted else "rejected",
        "accepted": accepted,
        "severity": "normal" if accepted else "hard_rejected",
        "warnings": [],
        "hard_issues": hard_issues,
        "issues": list(hard_issues),
        "profile": profile,
        "effective_thresholds": thresholds,
        "baseline_shape": (
            list(baseline.shape) if getattr(baseline, "ndim", 0) else None
        ),
        "candidate_shape": (
            list(candidate.shape) if getattr(candidate, "ndim", 0) else None
        ),
    }


def _stage3_recovery_channel_layout(image: Any) -> str:
    """Classify only the channel layouts supported by conservative recovery."""
    try:
        source = np.asarray(image)
    except (TypeError, ValueError):
        return "unknown"
    if source.ndim == 2:
        return "mono"
    if source.ndim != 3:
        return "unknown"
    if source.shape[0] == 1 and source.shape[-1] != 1:
        return "mono"
    if source.shape[-1] == 1:
        return "mono"
    if source.shape[0] == 3 and source.shape[-1] != 3:
        return "rgb_chw"
    if source.shape[-1] == 3:
        return "rgb_hwc"
    return "unknown"


def _stage3_write_neutral_axis_pixels(
    pipeline,
    pixels: Any,
) -> Tuple[bool, Optional[str]]:
    """Write a projected candidate only while holding Siril's image lock."""
    lock_factory = getattr(pipeline.siril, "image_lock", None)
    set_pixels = getattr(pipeline.siril, "set_image_pixeldata", None)
    if not callable(lock_factory):
        return False, "Siril image_lock is unavailable"
    if not callable(set_pixels):
        return False, "Siril pixel writer is unavailable"
    try:
        with lock_factory():
            set_pixels(pixels)
    except (
        CommandError,
        SirilError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        return False, str(error)
    return True, None


def _stage3_try_background_command(
    pipeline,
    label: str,
    command: Tuple[str, ...],
    source: str,
) -> Tuple[bool, Optional[str]]:
    is_graxpert = _stage3_is_graxpert_attempt(label, command, source)
    script_path = _stage3_pyscript_path(command)
    runtime_error_prefix = (
        "graxpert_runtime_error" if is_graxpert else "plugin_runtime_error"
    )
    if script_path is not None and hasattr(
        pipeline,
        "_validate_plugin_script_prerequisites",
    ):
        try:
            prerequisites_ok, prerequisites_reason = (
                pipeline._validate_plugin_script_prerequisites(script_path)
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            prerequisites_ok = False
            prerequisites_reason = f"prerequisite check failed: {exc}"
        if not prerequisites_ok:
            return False, (
                f"{runtime_error_prefix}: prerequisites unavailable: "
                f"{prerequisites_reason or 'unknown reason'}"
            )

    before_fingerprint = (
        _stage3_image_fingerprint(pipeline) if script_path is not None else None
    )
    try:
        pipeline.cmd_with_check(*command, quiet=True)
        after_fingerprint = (
            _stage3_image_fingerprint(pipeline) if script_path is not None else None
        )
        if (
            before_fingerprint
            and after_fingerprint
            and before_fingerprint == after_fingerprint
        ):
            return False, (
                f"{runtime_error_prefix}: command returned success "
                "but image did not change"
            )
        return True, None
    except (CommandError, SirilError, OSError, RuntimeError) as exc:
        if is_graxpert:
            reason = _stage3_graxpert_runtime_error_reason(exc, command)
            if reason is None:
                reason = f"graxpert_command_failed: {str(exc).strip() or type(exc).__name__}"
            return False, reason
        return False, f"command_failed: {str(exc).strip() or type(exc).__name__}"


def _stage3_cfg_float(
    pipeline,
    name: str,
    default: float,
    lower: float,
    upper: float,
) -> float:
    try:
        value = float(getattr(pipeline.cfg, name, default))
    except (TypeError, ValueError):
        value = default
    return max(lower, min(upper, value))


def _stage3_cfg_int(
    pipeline,
    name: str,
    default: int,
    lower: int,
    upper: int,
) -> int:
    try:
        value = int(getattr(pipeline.cfg, name, default))
    except (TypeError, ValueError):
        value = default
    return max(lower, min(upper, value))


def _stage3_decision_thresholds(
    pipeline,
    stage3_policy: Optional[Dict[str, Any]] = None,
    *,
    gate_profile_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Freeze the effective Stage 3 decision and ranking thresholds."""
    contract = stage3_static_contract_manifest()
    gate_profile = normalize_stage3_gate_profile(
        gate_profile_override
        if gate_profile_override is not None
        else getattr(
            getattr(pipeline, "cfg", None),
            "stage3_gate_profile",
            "output_first",
        )
    )
    gate_thresholds = stage3_gate_thresholds(gate_profile)
    strict = bool(gate_thresholds.get("strict_legacy"))
    return {
        **contract,
        "active_gate_profile": gate_profile,
        "active_gate_thresholds": gate_thresholds,
        "safe_samples": {
            "target_count": _stage3_cfg_int(
                pipeline, "stage3_safe_sample_target_count", 40, 16, 64
            ),
            "minimum_count": _stage3_cfg_int(
                pipeline, "stage3_safe_sample_min_count", 12, 12, 48
            ),
            "patch_radius": _stage3_cfg_int(
                pipeline, "stage3_safe_sample_patch_radius", 12, 4, 24
            ),
            "brightness_quantile_max": _stage3_cfg_float(
                pipeline,
                "stage3_safe_sample_brightness_quantile_max",
                0.70,
                0.50,
                0.85,
            ),
            "texture_quantile_max": _stage3_cfg_float(
                pipeline,
                "stage3_safe_sample_texture_quantile_max",
                0.55,
                0.25,
                0.75,
            ),
        },
        "candidate_sufficiency": {
            "maximum_background_score": _stage3_policy_float(
                stage3_policy,
                "sufficient_max_background_score",
                float(gate_thresholds["sufficient_max_background_score"]),
                minimum=0.0,
            ) if strict else float(gate_thresholds["sufficient_max_background_score"]),
            "maximum_dirty_score": _stage3_policy_float(
                stage3_policy,
                "sufficient_dirty_score_max",
                float(gate_thresholds["sufficient_dirty_score_max"]),
                minimum=0.0,
            ) if strict else float(gate_thresholds["sufficient_dirty_score_max"]),
            "maximum_bg_std_growth": _stage3_policy_float(
                stage3_policy,
                "max_bg_std_growth",
                float(gate_thresholds["sufficient_max_bg_std_growth"]),
                minimum=1.0,
                maximum=2.0,
            ) if strict else float(gate_thresholds["sufficient_max_bg_std_growth"]),
            "maximum_color_shift": _stage3_policy_float(
                stage3_policy,
                "sufficient_color_shift_max",
                float(gate_thresholds["sufficient_color_shift_max"]),
                minimum=0.0,
                maximum=2.0,
            ) if strict else float(gate_thresholds["sufficient_color_shift_max"]),
        },
        "compound_candidate": {
            "minimum_span_improvement_ratio": _stage3_cfg_float(
                pipeline,
                "stage3_compound_validation_improvement_min",
                0.10,
                0.10,
                0.40,
            ),
            "minimum_score_absolute_improvement": _stage3_cfg_float(
                pipeline,
                "stage3_compound_score_abs_improvement_min",
                0.03,
                0.03,
                0.15,
            ),
            "minimum_score_relative_improvement": _stage3_cfg_float(
                pipeline,
                "stage3_compound_score_rel_improvement_min",
                0.10,
                0.10,
                0.40,
            ),
        },
        "directional_pattern": {
            "pattern_score_min": _stage3_cfg_float(
                pipeline, "stage3_pattern_score_min", 0.55, 0.25, 0.90
            ),
            "walking_noise_score_min": _stage3_cfg_float(
                pipeline,
                "stage3_walking_noise_score_min",
                0.50,
                0.25,
                0.90,
            ),
            "maximum_pattern_score_growth": _stage3_cfg_float(
                pipeline,
                "stage3_pattern_score_growth_max",
                0.12,
                0.02,
                0.40,
            ),
        },
        "final_output_revalidation": {
            "enforced_for_profiled_runs": True,
            "gate_profile": gate_profile,
            "three_sigma_action": (
                "hard_reject" if strict else "soft_warning"
            ),
            "hard_thresholds": gate_thresholds,
            "maximum_bg_std_growth_warning": float(
                gate_thresholds["sufficient_max_bg_std_growth"]
            ),
        },
        "statistical_selection": {
            "method": "pareto_dense_rank_sum_v2",
            "runtime_selection_affected": True,
            "lower_is_better": [
                "residual_span_significance_sigma",
                "target_flux_deviation",
                "target_morphology_loss",
                "target_centroid_shift_fraction",
                "target_change_residual_significance",
                "directional_pattern_penalty",
                "soft_warning_count",
            ],
        },
    }


def _stage3_select_candidate(
    candidates: List[Dict[str, Any]],
    current_selected: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return the deterministic runtime Pareto/statistical candidate order."""
    current_label = str((current_selected or {}).get("label") or "") or None
    rows: List[Dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        if str(candidate.get("severity") or "") == "hard_rejected":
            continue
        validation = candidate.get("validation") or {}
        gate = candidate.get("validation_gate") or {}
        if gate.get("accepted") is False:
            continue
        preservation = candidate.get("preservation") or {}

        def finite_value(value: Any, default: float) -> float:
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                return default
            return parsed if math.isfinite(parsed) else default

        residual_span = finite_value(validation.get("robust_span"), 999.0)
        span_standard_error = finite_value(
            background_span_standard_error(validation)
            if validation.get("status") == "ready"
            else None,
            1.0,
        )
        if span_standard_error <= 0.0:
            span_standard_error = 1.0
        retention = finite_value(
            preservation.get("target_flux_retention_ratio"),
            1.0,
        )
        morphology = finite_value(
            preservation.get("target_morphology_correlation"),
            1.0,
        )
        centroid = max(
            0.0,
            finite_value(preservation.get("target_centroid_shift_fraction"), 0.0),
        )
        structure = max(
            0.0,
            finite_value(
                preservation.get("target_change_residual_significance"),
                0.0,
            ),
        )
        pattern_penalty = max(
            0.0,
            finite_value(candidate.get("directional_pattern_penalty"), 0.0),
        )
        runtime_score = finite_value(candidate.get("score"), 999.0)
        gate_warnings = list(candidate.get("gate_warnings") or [])
        color_normalization_evidence = (
            _stage3_color_normalization_candidate_evidence(candidate)
        )
        color_normalization_eligible = bool(
            color_normalization_evidence.get("eligible", False)
        )
        soft_warning_count = 0 if color_normalization_eligible else len(
            gate_warnings
        )
        candidate_tier = 0 if (
            color_normalization_eligible
            or (
                not gate_warnings
                and bool(candidate.get("sufficient", True))
            )
        ) else 1
        uncertainty_3sigma = gate.get("sampling_uncertainty_3sigma")
        span_improvement = gate.get("span_improvement")
        improvement_sigma = None
        try:
            uncertainty_value = float(uncertainty_3sigma)
            improvement_value = float(span_improvement)
            if uncertainty_value > 0.0:
                improvement_sigma = (
                    STAGE3_SIGNIFICANCE_SIGMA
                    * improvement_value
                    / uncertainty_value
                )
        except (TypeError, ValueError):
            pass
        rows.append(
            {
                "candidate_index": index,
                "label": str(candidate.get("label") or f"candidate_{index + 1}"),
                "source": candidate.get("source"),
                "chain_index": index,
                "candidate_tier": candidate_tier,
                "soft_warning_count": soft_warning_count,
                "gate_warnings": gate_warnings,
                "verified_color_normalization_candidate": (
                    color_normalization_eligible
                ),
                "color_normalization_candidate_evidence": (
                    color_normalization_evidence
                ),
                "runtime_selected": str(candidate.get("label") or "")
                == current_label,
                "residual_span": residual_span,
                "residual_span_standard_error": span_standard_error,
                "residual_span_significance_sigma": (
                    residual_span / span_standard_error
                ),
                "span_improvement_sigma": improvement_sigma,
                "target_flux_deviation": abs(retention - 1.0),
                "target_morphology_loss": max(0.0, 1.0 - morphology),
                "target_centroid_shift_fraction": centroid,
                "target_change_residual_significance": structure,
                "directional_pattern_penalty": pattern_penalty,
                "runtime_background_score": runtime_score,
            }
        )

    if not rows:
        return {
            "status": "unavailable",
            "method": "pareto_dense_rank_sum_v2",
            "runtime_selection_affected": True,
            "current_runtime_candidate": current_label,
            "reason": "no candidate has complete held-out statistical evidence",
            "candidates": [],
        }

    best_tier = min(int(row["candidate_tier"]) for row in rows)
    eligible_rows = [row for row in rows if int(row["candidate_tier"]) == best_tier]
    criteria = (
        "soft_warning_count",
        "residual_span_significance_sigma",
        "target_flux_deviation",
        "target_morphology_loss",
        "target_centroid_shift_fraction",
        "target_change_residual_significance",
        "directional_pattern_penalty",
    )

    def dominates(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
        return bool(
            all(float(left[key]) <= float(right[key]) for key in criteria)
            and any(float(left[key]) < float(right[key]) for key in criteria)
        )

    pareto_labels = [
        str(row["label"])
        for row in eligible_rows
        if not any(
            other is not row and dominates(other, row)
            for other in eligible_rows
        )
    ]
    for key in criteria:
        ordered_values = sorted({float(row[key]) for row in eligible_rows})
        ranks = {value: rank for rank, value in enumerate(ordered_values)}
        for row in eligible_rows:
            row[f"{key}_rank"] = ranks[float(row[key])]
    for row in eligible_rows:
        row["balanced_rank_sum"] = sum(
            int(row[f"{key}_rank"])
            for key in criteria
        )
        row["pareto_front"] = str(row["label"]) in pareto_labels

    selection_order = sorted(
        eligible_rows,
        key=lambda row: (
            int(row["balanced_rank_sum"]),
            not bool(row["pareto_front"]),
            float(row["residual_span_significance_sigma"]),
            float(row["runtime_background_score"]),
            int(row["chain_index"]),
        ),
    )
    proposed_label = str(selection_order[0]["label"])
    return {
        "status": "ready",
        "method": "pareto_dense_rank_sum_v2",
        "runtime_selection_affected": True,
        "selected_tier": best_tier,
        "current_runtime_candidate": current_label,
        "recommended_candidate": proposed_label,
        "selection_would_change": bool(
            current_label and proposed_label != current_label
        ),
        "pareto_front": pareto_labels,
        "statistical_order": [str(row["label"]) for row in selection_order],
        "candidates": selection_order,
    }


def _stage3_outer_halo_selection_override(
    legacy_selected: Dict[str, Any],
    statistical_selected: Dict[str, Any],
    stage3_policy: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Keep the safer low-order model when an outer-halo RBF is not better.

    Large galaxies can occupy enough of the frame that a locally smoother RBF
    residual is achieved by fitting real outer-halo signal.  Only override the
    statistical order when the Polynomial candidate has the better aggregate
    background score and no worse fixed-target fidelity.
    """

    report: Dict[str, Any] = {
        "applied": False,
        "reason_code": "not_applicable",
    }
    if not bool(stage3_policy.get("protect_outer_halo", False)):
        return statistical_selected, report
    legacy_label = str(legacy_selected.get("label") or "").lower()
    selected_label = str(statistical_selected.get("label") or "").lower()
    if "poly" not in legacy_label or "rbf" not in selected_label:
        return statistical_selected, report

    def finite(value: Any, default: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return parsed if math.isfinite(parsed) else default

    legacy_score = finite(legacy_selected.get("score"), 999.0)
    selected_score = finite(statistical_selected.get("score"), 999.0)
    legacy_preservation = legacy_selected.get("preservation") or {}
    selected_preservation = statistical_selected.get("preservation") or {}
    legacy_flux_error = abs(
        finite(legacy_preservation.get("target_flux_retention_ratio"), 1.0)
        - 1.0
    )
    selected_flux_error = abs(
        finite(selected_preservation.get("target_flux_retention_ratio"), 1.0)
        - 1.0
    )
    legacy_morphology_loss = max(
        0.0,
        1.0
        - finite(
            legacy_preservation.get("target_morphology_correlation"),
            1.0,
        ),
    )
    selected_morphology_loss = max(
        0.0,
        1.0
        - finite(
            selected_preservation.get("target_morphology_correlation"),
            1.0,
        ),
    )
    legacy_validation = legacy_selected.get("validation") or {}
    selected_validation = statistical_selected.get("validation") or {}
    legacy_residual_span = finite(
        legacy_validation.get("robust_span"),
        float("inf"),
    )
    selected_residual_span = finite(
        selected_validation.get("robust_span"),
        float("inf"),
    )
    selected_centroid_shift = abs(
        finite(
            selected_preservation.get("target_centroid_shift_fraction"),
            float("inf"),
        )
    )
    # A low-order preference must not preserve a visible coverage boundary.
    # Retain the statistically selected RBF only when its held-out residual
    # span is materially smaller and every fixed-target fidelity measurement
    # remains inside a deliberately tight budget.  The bounded aggregate-score
    # allowance prevents a small colour/noise term from overruling the direct
    # spatial-background evidence while still rejecting broadly worse models.
    materially_cleaner_background = bool(
        math.isfinite(legacy_residual_span)
        and legacy_residual_span > 0.0
        and math.isfinite(selected_residual_span)
        and selected_residual_span
        <= 0.50 * legacy_residual_span
    )
    aggregate_score_bounded = bool(
        math.isfinite(legacy_score)
        and legacy_score > 0.0
        and math.isfinite(selected_score)
        and selected_score <= 1.15 * legacy_score
    )
    selected_fidelity_bounded = bool(
        selected_flux_error <= 0.01
        and selected_morphology_loss <= 1.0e-4
        and selected_centroid_shift <= 0.005
    )
    if (
        materially_cleaner_background
        and aggregate_score_bounded
        and selected_fidelity_bounded
    ):
        return statistical_selected, {
            "applied": False,
            "reason_code": (
                "rbf_material_background_gain_within_outer_halo_fidelity_budget"
            ),
            "preserved_statistical_selection": True,
            "statistical_candidate": statistical_selected.get("label"),
            "selected_candidate": statistical_selected.get("label"),
            "aggregate_background_score": {
                "polynomial": legacy_score,
                "rbf": selected_score,
                "rbf_to_polynomial_max": 1.15,
            },
            "residual_span": {
                "polynomial": legacy_residual_span,
                "rbf": selected_residual_span,
                "rbf_to_polynomial_max": 0.50,
            },
            "target_flux_deviation": selected_flux_error,
            "target_morphology_loss": selected_morphology_loss,
            "target_centroid_shift_fraction": selected_centroid_shift,
        }
    if (
        selected_score >= legacy_score
        and selected_flux_error >= legacy_flux_error
        and selected_morphology_loss >= legacy_morphology_loss
    ):
        return legacy_selected, {
            "applied": True,
            "reason_code": "outer_halo_low_order_fidelity_preferred",
            "statistical_candidate": statistical_selected.get("label"),
            "selected_candidate": legacy_selected.get("label"),
            "aggregate_background_score": {
                "polynomial": legacy_score,
                "rbf": selected_score,
            },
            "target_flux_deviation": {
                "polynomial": legacy_flux_error,
                "rbf": selected_flux_error,
            },
            "target_morphology_loss": {
                "polynomial": legacy_morphology_loss,
                "rbf": selected_morphology_loss,
            },
        }
    return statistical_selected, report


def _stage3_final_output_validation(
    pipeline,
    *,
    baseline_image: Any,
    baseline_validation: Dict[str, Any],
    validation_points: List[Tuple[float, float]],
    patch_radius: int,
    minimum_count: int,
    enforced: bool,
    gate_profile: str = "output_first",
    reload_stem: Optional[str] = None,
    neutral_axis_enforced: bool = False,
    spatial_opponent_projection: Optional[Dict[str, Any]] = None,
    support_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Re-run the active profile gate on the reloaded, saved final buffer."""
    report: Dict[str, Any] = {
        "status": "not_enforced" if not enforced else "running",
        "enforced": bool(enforced),
        "evidence_basis": "reloaded_saved_output_active_stage3_gate_profile",
    }
    if not enforced:
        report["reason"] = "legacy caller has no profiled production input"
        return report
    try:
        neutral_checkpoint_image = None
        spatial_applied = bool(
            spatial_opponent_projection
            and str(spatial_opponent_projection.get("reason_code") or "")
            == "stage3_spatial_opponent_correction_applied"
        )
        if spatial_applied:
            pipeline.cmd_with_check(
                "load",
                "stage3_candidate_neutral_axis_poly1",
                quiet=True,
            )
            neutral_checkpoint_image = pipeline.siril.get_image_pixeldata(
                preview=False
            )
        if reload_stem:
            pipeline.cmd_with_check("load", reload_stem, quiet=True)
            report["reloaded_stem"] = reload_stem
        image = pipeline.siril.get_image_pixeldata(preview=False)
        pixel_gate_ok, pixel_gate = _stage3_candidate_pixel_gate(
            baseline_image,
            image,
            gate_profile=gate_profile,
        )
        candidate_validation = measure_background_validation(
            image,
            validation_points,
            patch_radius=patch_radius,
            minimum_count=minimum_count,
            value_scale=baseline_validation.get("value_scale"),
            support_mask=support_mask,
        )
        if neutral_axis_enforced:
            if spatial_applied:
                neutral_axis_ok, neutral_axis_gate = (
                    verify_stage3_neutral_axis_persistence(
                        baseline_image,
                        neutral_checkpoint_image,
                    )
                )
                spatial_opponent_ok, spatial_opponent_gate = (
                    verify_stage3_spatial_opponent_persistence(
                        neutral_checkpoint_image,
                        image,
                        spatial_opponent_projection or {},
                    )
                )
            else:
                neutral_axis_ok, neutral_axis_gate = (
                    verify_stage3_neutral_axis_persistence(
                        baseline_image,
                        image,
                    )
                )
                spatial_opponent_ok = True
                spatial_opponent_gate = {
                    "status": "not_applicable",
                    "accepted": True,
                }
            direction_ok, direction_gate = assess_background_direction_reversal(
                baseline_image,
                image,
                validation_points,
                patch_radius=patch_radius,
                support_mask=support_mask,
            )
        else:
            neutral_axis_ok = True
            spatial_opponent_ok = True
            direction_ok = True
            neutral_axis_gate = {
                "status": "not_applicable",
                "accepted": True,
            }
            spatial_opponent_gate = {
                "status": "not_applicable",
                "accepted": True,
            }
            direction_gate = {
                "status": "not_applicable",
                "accepted": True,
            }
    except (
        AttributeError,
        CommandError,
        OSError,
        RuntimeError,
        SirilError,
        TypeError,
        ValueError,
    ) as error:
        profile = normalize_stage3_gate_profile(gate_profile)
        pixel_gate_ok = False
        pixel_gate = {
            "status": "rejected",
            "accepted": False,
            "severity": "hard_rejected",
            "warnings": [],
            "hard_issues": [f"final saved output pixels are unavailable: {error}"],
            "issues": [f"final saved output pixels are unavailable: {error}"],
            "profile": profile,
            "effective_thresholds": stage3_gate_thresholds(profile),
        }
        candidate_validation = {
            "status": "unavailable",
            "reason": str(error),
        }
        neutral_axis_ok = not neutral_axis_enforced
        spatial_opponent_ok = not neutral_axis_enforced
        direction_ok = not neutral_axis_enforced
        neutral_axis_gate = {
            "status": "rejected" if neutral_axis_enforced else "not_applicable",
            "accepted": not neutral_axis_enforced,
            "issues": [str(error)] if neutral_axis_enforced else [],
        }
        spatial_opponent_gate = {
            "status": "rejected" if neutral_axis_enforced else "not_applicable",
            "accepted": not neutral_axis_enforced,
            "issues": [str(error)] if neutral_axis_enforced else [],
        }
        direction_gate = {
            "status": "rejected" if neutral_axis_enforced else "not_applicable",
            "accepted": not neutral_axis_enforced,
            "issues": [str(error)] if neutral_axis_enforced else [],
        }
    accepted, gate = assess_single_background_validation(
        baseline_validation,
        candidate_validation,
        gate_profile=gate_profile,
    )
    if not pixel_gate_ok:
        accepted = False
    if not neutral_axis_ok or not spatial_opponent_ok or not direction_ok:
        accepted = False
    report.update(
        {
            "status": "accepted" if accepted else "rejected",
            "accepted": accepted,
            "severity": (
                "hard_rejected"
                if not pixel_gate_ok or not accepted
                else str(gate.get("severity") or "normal")
            ),
            "pixel_integrity_gate": pixel_gate,
            "validation": candidate_validation,
            "validation_gate": gate,
            "neutral_axis_persistence": neutral_axis_gate,
            "spatial_opponent_persistence": spatial_opponent_gate,
            "directional_gradient_gate": direction_gate,
        }
    )
    return report


def _stage3_compound_target_guard(
    target_profile: Dict[str, Any],
    diffuse_context: Dict[str, Any],
    stage3_policy: Dict[str, Any],
    noise_route: Dict[str, Any],
) -> Dict[str, Any]:
    """Hard-disable compound fitting where large-scale signal is ambiguous."""
    target_type = str((target_profile or {}).get("target_type") or "").lower()
    reasons: List[str] = []
    context = diffuse_context or {}
    policy = stage3_policy or {}
    if bool((noise_route or {}).get("requires_review", False)):
        reasons.append("directional_pattern_noise_requires_review")
    if bool(
        context.get("diffuse")
        or context.get("emission_diffuse")
        or context.get("large_nebulosity_feature")
    ):
        reasons.append("large_or_diffuse_nebula_signal")
    if bool(context.get("faint_nebula_protection")):
        reasons.append("low_contrast_faint_structure")
    if "dark_nebula" in target_type or bool(policy.get("protect_dark_structure")):
        reasons.append("dark_nebula_structure")
    if "low_contrast" in target_type:
        reasons.append("low_contrast_target_profile")
    unique_reasons = list(dict.fromkeys(reasons))
    return {
        "status": "eligible" if not unique_reasons else "excluded",
        "eligible": not unique_reasons,
        "reasons": unique_reasons,
        "target_type": target_type or None,
    }


def _stage3_compound_residual_gate(
    validation: Dict[str, Any],
) -> Dict[str, Any]:
    """Require residual sky variation above held-out sampling uncertainty."""
    if (validation or {}).get("status") != "ready":
        return {
            "status": "not_supported",
            "supported": False,
            "reason": "held-out validation is unavailable",
        }
    try:
        robust_span = float(validation["robust_span"])
        patch_rms = float(validation["patch_mad_median"])
    except (KeyError, TypeError, ValueError):
        return {
            "status": "not_supported",
            "supported": False,
            "reason": "held-out validation is incomplete",
        }
    span_standard_error = background_span_standard_error(validation)
    significance_limit = max(
        STAGE3_SIGNIFICANCE_SIGMA * span_standard_error,
        1e-12,
    )
    supported = bool(
        math.isfinite(robust_span)
        and math.isfinite(significance_limit)
        and robust_span > significance_limit
    )
    return {
        "status": "supported" if supported else "not_supported",
        "supported": supported,
        "robust_span": robust_span,
        "patch_rms": patch_rms,
        "patch_median_standard_error": validation.get(
            "patch_median_uncertainty"
        ),
        "span_standard_error": span_standard_error,
        "spatial_significance_limit_3sigma": significance_limit,
        "evidence_basis": "correlation_aware_heldout_sky_sampling_uncertainty",
    }


def _stage3_compound_score_gate(
    best_single_score: float,
    compound_score: float,
    *,
    absolute_improvement_min: float = 0.03,
    relative_improvement_min: float = 0.10,
) -> Tuple[bool, Dict[str, Any]]:
    try:
        best_score = float(best_single_score)
        candidate_score = float(compound_score)
    except (TypeError, ValueError):
        return False, {
            "status": "rejected",
            "accepted": False,
            "issues": ["background scores are unavailable"],
        }
    if not math.isfinite(best_score) or not math.isfinite(candidate_score):
        return False, {
            "status": "rejected",
            "accepted": False,
            "issues": ["background scores are non-finite"],
        }
    absolute_min = max(0.0, float(absolute_improvement_min))
    relative_min = max(0.0, float(relative_improvement_min))
    absolute_improvement = best_score - candidate_score
    relative_improvement = absolute_improvement / max(abs(best_score), 1e-7)
    issues: List[str] = []
    if absolute_improvement + 1e-12 < absolute_min:
        issues.append(
            "absolute score improvement "
            f"{absolute_improvement:.3f}<{absolute_min:.3f}"
        )
    if relative_improvement + 1e-12 < relative_min:
        issues.append(
            "relative score improvement "
            f"{relative_improvement:.3f}<{relative_min:.3f}"
        )
    accepted = not issues
    return accepted, {
        "status": "accepted" if accepted else "rejected",
        "accepted": accepted,
        "issues": issues,
        "best_single_score": best_score,
        "compound_score": candidate_score,
        "absolute_improvement": absolute_improvement,
        "absolute_improvement_min": absolute_min,
        "relative_improvement": relative_improvement,
        "relative_improvement_min": relative_min,
    }


def _stage3_clear_background_samples(pipeline) -> None:
    clear_samples = getattr(pipeline.siril, "clear_image_bgsamples", None)
    if not callable(clear_samples):
        return
    try:
        clear_samples()
    except (
        AttributeError,
        CommandError,
        OSError,
        RuntimeError,
        SirilError,
        TypeError,
        ValueError,
    ) as error:
        if hasattr(pipeline, "log"):
            pipeline.log.debug(f"Stage3 background sample cleanup skipped: {error}")


def _stage3_install_safe_background_samples(
    pipeline,
    points: List[Tuple[float, float]],
    *,
    minimum_count: Optional[int] = None,
    sample_contract: str = "safe_background",
    sample_records: Optional[List[Dict[str, Any]]] = None,
    masked_statistics: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """Install and audit Siril's recalculated sample set before ``-existing``.

    Siril may discard a sample whose 25-pixel statistics cannot be
    recalculated.  That is not itself a failure: the returned set remains
    usable only when every surviving coordinate came from the audited request,
    the configured minimum is retained, and its spatial coverage is still
    sufficient.  ``minimum_count`` belongs to the caller's sampling contract:
    the ordinary candidate chain uses the global safe-sample minimum, while a
    compound Polynomial→RBF fit uses its smaller, separately validated fit
    minimum.  Unknown samples or collapsed coverage fail closed.
    """
    if minimum_count is None:
        try:
            required_count = int(
                getattr(
                    getattr(pipeline, "cfg", None),
                    "stage3_safe_sample_min_count",
                    12,
                )
            )
        except (TypeError, ValueError):
            required_count = 12
    else:
        try:
            required_count = int(minimum_count)
        except (TypeError, ValueError):
            required_count = STAGE3_MIN_VALIDATION_PATCHES
    required_count = max(
        STAGE3_MIN_VALIDATION_PATCHES,
        min(64, required_count),
    )
    contract_name = str(sample_contract or "safe_background")
    masked_statistics = bool(masked_statistics)
    setter = getattr(pipeline.siril, "set_image_bgsamples", None)
    if not points:
        return False, {
            "status": "unavailable",
            "installed": False,
            "reason": "safe background sample set is empty",
            "sample_contract": contract_name,
            "minimum_count": required_count,
        }
    if not callable(setter):
        return False, {
            "status": "unsupported",
            "installed": False,
            "reason": "Siril Python API does not expose set_image_bgsamples",
            "sample_contract": contract_name,
            "minimum_count": required_count,
        }
    requested_samples: Any = points
    requested_records: List[Dict[str, Any]] = []
    statistics_digest = None
    transport_padding_applied = False
    if masked_statistics:
        if sample_records is None or len(sample_records) != len(points):
            return False, {
                "status": "failed",
                "installed": False,
                "reason": "masked BGSample records do not match fit points",
                "reason_code": "stage3_dense_star_bg_sample_roundtrip_failed",
                "sample_contract": contract_name,
                "minimum_count": required_count,
            }
        by_point = {
            (
                round(float(record.get("point", [math.nan, math.nan])[0]), 6),
                round(float(record.get("point", [math.nan, math.nan])[1]), 6),
            ): record
            for record in sample_records
            if isinstance(record, dict)
            and isinstance(record.get("point"), (list, tuple))
            and len(record.get("point")) >= 2
        }
        built_samples = []
        digest_rows = []
        for point in points:
            key = (round(float(point[0]), 6), round(float(point[1]), 6))
            record = by_point.get(key)
            if record is None:
                return False, {
                    "status": "failed",
                    "installed": False,
                    "reason": "masked BGSample provenance is missing a fit point",
                    "reason_code": "stage3_dense_star_bg_sample_roundtrip_failed",
                    "sample_contract": contract_name,
                    "minimum_count": required_count,
                }
            medians = [float(value) for value in record.get("channel_medians") or []]
            channel_count = int(record.get("channel_count") or len(medians))
            if channel_count == 1 and len(medians) == 1:
                median_tuple = (medians[0], 0.0, 0.0)
            elif channel_count == 3 and len(medians) == 3:
                median_tuple = tuple(medians)
            else:
                return False, {
                    "status": "failed",
                    "installed": False,
                    "reason": "masked BGSample channel statistics are invalid",
                    "reason_code": "stage3_dense_star_bg_sample_roundtrip_failed",
                    "sample_contract": contract_name,
                    "minimum_count": required_count,
                }
            sample_size = int(record.get("sample_size") or 0)
            if sample_size <= 0 or sample_size % 2 == 0:
                return False, {
                    "status": "failed",
                    "installed": False,
                    "reason": "masked BGSample size is invalid",
                    "reason_code": "stage3_dense_star_bg_sample_roundtrip_failed",
                    "sample_contract": contract_name,
                    "minimum_count": required_count,
                }
            numeric = (
                *median_tuple,
                float(record.get("native_luminance_mean")),
                float(record.get("native_luminance_min")),
                float(record.get("native_luminance_max")),
            )
            if not all(math.isfinite(value) for value in numeric):
                return False, {
                    "status": "failed",
                    "installed": False,
                    "reason": "masked BGSample statistics are non-finite",
                    "reason_code": "stage3_dense_star_bg_sample_roundtrip_failed",
                    "sample_contract": contract_name,
                    "minimum_count": required_count,
                }
            built_samples.append(
                BGSample(
                    position=(float(point[0]), float(point[1])),
                    median=median_tuple,
                    mean=numeric[3],
                    min=numeric[4],
                    max=numeric[5],
                    size=sample_size,
                    valid=True,
                )
            )
            requested_records.append(record)
            digest_rows.append((key, median_tuple, sample_size))
        requested_samples = built_samples
        statistics_digest = hashlib.sha256(
            repr(sorted(digest_rows)).encode("utf-8")
        ).hexdigest()
    transmitted_samples = requested_samples
    setter_module = str(getattr(setter, "__module__", ""))
    interface_module = str(type(pipeline.siril).__module__)
    bg_sample_module = str(getattr(BGSample, "__module__", ""))
    if masked_statistics and (
        setter_module.startswith("sirilpy")
        or interface_module.startswith("sirilpy")
        or bg_sample_module.startswith("sirilpy")
    ):
        # sirilpy 1.4.4 serializes N native-aligned background structs as
        # ``80*N-4`` bytes, so Siril receives only N-1 complete records.  An
        # audited duplicate sentinel restores the final record.  The strict
        # round-trip below rejects a runtime where the sentinel is retained,
        # preventing it from changing the fit on fixed/newer transports.
        transmitted_samples = [*requested_samples, requested_samples[-1]]
        transport_padding_applied = True
    try:
        _stage3_clear_background_samples(pipeline)
        result = setter(
            transmitted_samples,
            show_samples=False,
            recalculate=not masked_statistics,
        )
        if result is False:
            raise RuntimeError("set_image_bgsamples returned false")
        observed_count = None
        observed_positions: List[Tuple[float, float]] = []
        observed_samples: List[Any] = []
        rejected_positions: List[List[float]] = []
        observed_coverage: Dict[str, Any] = {}
        getter = getattr(pipeline.siril, "get_image_bgsamples", None)
        if masked_statistics and not callable(getter):
            raise RuntimeError(
                "Siril does not expose BGSample round-trip verification"
            )
        if callable(getter):
            observed = getter()
            if observed is None:
                raise RuntimeError(
                    "Siril did not return the installed background samples"
                )
            observed_count = len(observed)
            for sample in observed:
                position = getattr(sample, "position", sample)
                if not isinstance(position, (tuple, list)) or len(position) < 2:
                    raise RuntimeError(
                        "Siril returned a background sample without coordinates"
                    )
                observed_positions.append(
                    (float(position[0]), float(position[1]))
                )
                observed_samples.append(sample)

            remaining = list(observed_positions)
            for requested_x, requested_y in points:
                match_index = next(
                    (
                        index
                        for index, (observed_x, observed_y) in enumerate(remaining)
                        if math.hypot(
                            observed_x - float(requested_x),
                            observed_y - float(requested_y),
                        ) <= 0.75
                    ),
                    None,
                )
                if match_index is None:
                    rejected_positions.append(
                        [float(requested_x), float(requested_y)]
                    )
                else:
                    remaining.pop(match_index)
            if remaining:
                raise RuntimeError(
                    "Siril returned background samples outside the audited set"
                )

            if masked_statistics:
                unmatched = list(range(len(observed_samples)))
                tolerance = 1e-7
                for requested, record in zip(requested_samples, requested_records):
                    match_index = next(
                        (
                            index
                            for index in unmatched
                            if math.hypot(
                                float(observed_samples[index].position[0])
                                - float(requested.position[0]),
                                float(observed_samples[index].position[1])
                                - float(requested.position[1]),
                            )
                            <= 0.75
                        ),
                        None,
                    )
                    if match_index is None:
                        raise RuntimeError(
                            "masked BGSample was not returned by Siril: "
                            f"position=({float(requested.position[0]):.6f},"
                            f"{float(requested.position[1]):.6f}) "
                            f"observed_count={len(observed_samples)}"
                        )
                    observed_sample = observed_samples[match_index]
                    unmatched.remove(match_index)
                    source_medians = [
                        float(value)
                        for value in record.get("channel_medians") or []
                    ]
                    expected_medians = (
                        (source_medians[0], 0.0, 0.0)
                        if int(record.get("channel_count") or 0) == 1
                        else tuple(source_medians)
                    )
                    observed_medians = tuple(
                        float(value) for value in observed_sample.median
                    )
                    value_scale = max(
                        1.0,
                        *(abs(value) for value in expected_medians),
                    )
                    sample_tolerance = max(tolerance, 8.0 * np.finfo(np.float32).eps * value_scale)
                    if len(observed_medians) != 3 or any(
                        abs(observed_value - expected_value) > sample_tolerance
                        for observed_value, expected_value in zip(
                            observed_medians,
                            expected_medians,
                        )
                    ):
                        raise RuntimeError(
                            "Siril altered masked BGSample channel medians"
                        )
                    if int(getattr(observed_sample, "size", 0)) != int(
                        record.get("sample_size") or 0
                    ):
                        raise RuntimeError("Siril altered masked BGSample size")
                    if not bool(getattr(observed_sample, "valid", False)):
                        raise RuntimeError("Siril invalidated a masked BGSample")

            if observed_count < required_count:
                raise RuntimeError(
                    "Siril retained too few audited background samples: "
                    f"contract={contract_name} minimum={required_count} "
                    f"observed={observed_count}"
                )

            requested_x = [float(point[0]) for point in points]
            requested_y = [float(point[1]) for point in points]
            x_low, x_high = min(requested_x), max(requested_x)
            y_low, y_high = min(requested_y), max(requested_y)
            x_span = max(x_high - x_low, 1.0)
            y_span = max(y_high - y_low, 1.0)
            cells = {
                (
                    min(3, max(0, int((x - x_low) / x_span * 4.0))),
                    min(3, max(0, int((y - y_low) / y_span * 4.0))),
                )
                for x, y in observed_positions
            }
            quadrants = {
                (int(x >= (x_low + x_high) / 2.0), int(y >= (y_low + y_high) / 2.0))
                for x, y in observed_positions
            }
            observed_coverage = {
                "quadrants": len(quadrants),
                "grid_cells": len(cells),
                "x_span_ratio_of_requested_envelope": (
                    max(x for x, _y in observed_positions)
                    - min(x for x, _y in observed_positions)
                ) / x_span,
                "y_span_ratio_of_requested_envelope": (
                    max(y for _x, y in observed_positions)
                    - min(y for _x, y in observed_positions)
                ) / y_span,
            }
            if (
                len(quadrants) < STAGE3_MIN_SPATIAL_QUADRANTS
                or len(cells) < STAGE3_MIN_SPATIAL_GRID_CELLS
                or observed_coverage["x_span_ratio_of_requested_envelope"]
                < STAGE3_MIN_AXIS_SPAN_RATIO
                or observed_coverage["y_span_ratio_of_requested_envelope"]
                < STAGE3_MIN_AXIS_SPAN_RATIO
            ):
                raise RuntimeError(
                    "Siril recalculation collapsed audited sample coverage: "
                    f"quadrants={len(quadrants)} grid_cells={len(cells)} "
                    "x_span_ratio="
                    f"{observed_coverage['x_span_ratio_of_requested_envelope']:.3f} "
                    "y_span_ratio="
                    f"{observed_coverage['y_span_ratio_of_requested_envelope']:.3f}"
                )
        return True, {
            "status": "installed",
            "installed": True,
            "sample_count": observed_count if observed_count is not None else len(points),
            "requested_count": len(points),
            "observed_count": observed_count,
            "siril_rejected_count": len(rejected_positions),
            "siril_rejected_positions": rejected_positions,
            "observed_coverage": observed_coverage,
            "command_contract": "subsky -existing",
            "sample_contract": contract_name,
            "minimum_count": required_count,
            "statistics_mode": (
                "masked_native_channel_bg_sample"
                if masked_statistics
                else "siril_recalculated_coordinates"
            ),
            "siril_recalculate": not masked_statistics,
            "statistics_sha256": statistics_digest,
            "roundtrip_verified": bool(masked_statistics),
            "transport_padding_applied": transport_padding_applied,
            "transmitted_count": len(transmitted_samples),
            "transport_interface": {
                "setter_module": setter_module,
                "interface_module": interface_module,
                "bg_sample_module": bg_sample_module,
            },
        }
    except (
        AttributeError,
        CommandError,
        OSError,
        RuntimeError,
        SirilError,
        TypeError,
        ValueError,
    ) as error:
        _stage3_clear_background_samples(pipeline)
        return False, {
            "status": "failed",
            "installed": False,
            "sample_count": len(points),
            "reason": str(error),
            "sample_contract": contract_name,
            "minimum_count": required_count,
            "reason_code": (
                "stage3_dense_star_bg_sample_roundtrip_failed"
                if masked_statistics
                else None
            ),
            "transport_padding_applied": transport_padding_applied,
            "transmitted_count": len(transmitted_samples),
            "transport_interface": {
                "setter_module": setter_module,
                "interface_module": interface_module,
                "bg_sample_module": bg_sample_module,
            },
        }


def _stage3_subsky_uses_existing(command: Tuple[str, ...]) -> bool:
    return bool(
        command
        and str(command[0]).lower() == "subsky"
        and "-existing" in command
    )


def _stage3_metric(
    target_profile: Dict[str, Any],
    adaptive: Dict[str, Any],
    key: str,
) -> float:
    profile = target_profile or {}
    for section_name in ("object_stats", "image_stats", "color_stats", "star_stats"):
        section = profile.get(section_name) if isinstance(profile, dict) else None
        if isinstance(section, dict) and key in section:
            try:
                return float(section.get(key) or 0.0)
            except (TypeError, ValueError):
                return 0.0
    if isinstance(adaptive, dict) and key in adaptive:
        try:
            return float(adaptive.get(key) or 0.0)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _stage3_diffuse_nebula_context(
    target_profile: Dict[str, Any],
    adaptive: Dict[str, Any],
    *,
    stage3_policy: Optional[Dict[str, Any]] = None,
    object_area_min: float = DEFAULT_DIFFUSE_OBJECT_AREA_MIN,
    nebulosity_area_min: float = DEFAULT_DIFFUSE_NEBULOSITY_AREA_MIN,
    faint_structure_min: float = DEFAULT_DIFFUSE_FAINT_STRUCTURE_MIN,
) -> Tuple[bool, Dict[str, Any]]:
    profile = target_profile or {}
    object_area_min = _stage3_policy_float(
        stage3_policy,
        "diffuse_nebula_object_area_min",
        object_area_min,
        minimum=0.0,
        maximum=1.0,
    )
    nebulosity_area_min = _stage3_policy_float(
        stage3_policy,
        "diffuse_nebula_nebulosity_area_min",
        nebulosity_area_min,
        minimum=0.0,
        maximum=1.0,
    )
    faint_structure_min = _stage3_policy_float(
        stage3_policy,
        "diffuse_nebula_faint_structure_min",
        faint_structure_min,
        minimum=0.0,
        maximum=1.0,
    )
    faint_nebula_area_min = _stage3_policy_float(
        stage3_policy,
        "faint_nebula_nebulosity_area_min",
        DEFAULT_FAINT_NEBULA_AREA_MIN,
        minimum=0.0,
        maximum=1.0,
    )
    faint_nebula_structure_min = _stage3_policy_float(
        stage3_policy,
        "faint_nebula_structure_min",
        DEFAULT_FAINT_NEBULA_STRUCTURE_MIN,
        minimum=0.0,
        maximum=1.0,
    )
    target_type = str(profile.get("target_type") or "").strip().lower()
    secondary_labels = {
        str(label).strip()
        for label in (profile.get("secondary_labels") or [])
    }
    features = profile.get("features") if isinstance(profile, dict) else {}
    feature_large = bool(
        isinstance(features, dict)
        and features.get("large_nebulosity")
    )
    feature_large = bool(
        feature_large or "large_nebulosity" in secondary_labels
    )
    object_area_ratio = _stage3_metric(profile, adaptive or {}, "object_area_ratio")
    nebulosity_area_ratio = _stage3_metric(profile, adaptive or {}, "nebulosity_area_ratio")
    faint_structure_score = _stage3_metric(profile, adaptive or {}, "faint_structure_score")
    is_emission = target_type in EMISSION_NEBULA_TARGET_TYPES
    emission_context = bool(is_emission or "emission_red" in secondary_labels)
    emission_diffuse = bool(
        emission_context
        and (
            object_area_ratio >= object_area_min
            or nebulosity_area_ratio >= nebulosity_area_min
            or faint_structure_score >= faint_structure_min
            or feature_large
        )
    )
    faint_nebula_protection = bool(
        nebulosity_area_ratio > faint_nebula_area_min
        and faint_structure_score > faint_nebula_structure_min
    )
    pixel_signal_protection = bool(faint_nebula_protection or feature_large)
    diffuse = bool(emission_diffuse or pixel_signal_protection)
    if emission_diffuse:
        protection_reason = "emission_target_signal"
    elif faint_nebula_protection:
        protection_reason = "faint_nebula_signal"
    elif feature_large:
        protection_reason = "large_nebulosity_feature"
    else:
        protection_reason = "none"
    return diffuse, {
        "target_type": target_type,
        "is_emission_target": is_emission,
        "secondary_labels": sorted(secondary_labels),
        "secondary_emission_context": bool(
            "emission_red" in secondary_labels
        ),
        "emission_diffuse": emission_diffuse,
        "faint_nebula_protection": faint_nebula_protection,
        "pixel_signal_protection": pixel_signal_protection,
        "protection_reason": protection_reason,
        "object_area_ratio": object_area_ratio,
        "nebulosity_area_ratio": nebulosity_area_ratio,
        "faint_structure_score": faint_structure_score,
        "large_nebulosity_feature": feature_large,
        "object_area_min": object_area_min,
        "nebulosity_area_min": nebulosity_area_min,
        "faint_structure_min": faint_structure_min,
        "faint_nebula_nebulosity_area_min": faint_nebula_area_min,
        "faint_nebula_structure_min": faint_nebula_structure_min,
    }


def _stage3_prefers_poly_first(
    target_profile: Dict[str, Any],
    adaptive: Dict[str, Any],
    *,
    stage3_policy: Optional[Dict[str, Any]] = None,
    object_area_min: float = DEFAULT_DIFFUSE_OBJECT_AREA_MIN,
) -> bool:
    target_type = str((target_profile or {}).get("target_type") or "").lower()
    if target_type in {"large_galaxy", "galaxy", "dark_nebula"}:
        return True
    diffuse, _context = _stage3_diffuse_nebula_context(
        target_profile,
        adaptive,
        stage3_policy=stage3_policy,
        object_area_min=object_area_min,
    )
    return diffuse


def _stage3_should_exhaust_builtin_search(
    target_profile: Dict[str, Any],
    adaptive: Dict[str, Any],
    stage3_policy: Dict[str, Any],
) -> Tuple[bool, Dict[str, Any]]:
    diffuse, context = _stage3_diffuse_nebula_context(
        target_profile,
        adaptive,
        stage3_policy=stage3_policy,
    )
    policy_requests_protection = bool(
        (stage3_policy or {}).get("reject_samples_on_nebula")
        or (stage3_policy or {}).get("protect_nebulosity")
    )
    pixel_signal_requests_protection = bool(
        context.get("faint_nebula_protection")
        or context.get("large_nebulosity_feature")
    )
    return bool(diffuse and (policy_requests_protection or pixel_signal_requests_protection)), context


def _stage3_background_decision(
    pipeline,
    adaptive: Dict[str, Any],
    *,
    diffuse_context: Optional[Dict[str, Any]] = None,
    process_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve Stage 3 from process evidence before destructive commands."""
    if isinstance(process_report, dict) and process_report:
        mechanism = str(process_report.get("mechanism") or "unknown")
        common_process = {
            "metrics": dict(adaptive or {}),
            "diffuse_context": dict(diffuse_context or {}),
            "process_evidence": process_report,
            "process_evidence_schema": process_report.get(
                "schema_version",
                STAGE3_PROCESS_EVIDENCE_SCHEMA,
            ),
            "threshold_basis": "source-masked held-out process evidence",
        }
        hard_blocks = list(process_report.get("hard_block_reasons") or [])
        if hard_blocks:
            return {
                **common_process,
                "decision": "review_required",
                "source": "process_evidence",
                "confidence": 1.0,
                "reason": "; ".join(str(reason) for reason in hard_blocks),
            }
        if bool(process_report.get("should_evaluate", False)):
            return {
                **common_process,
                "decision": "apply",
                "source": "process_evidence",
                "confidence": 1.0,
                "reason": (
                    "linear input, source-masked true sky and held-out spatial "
                    "variation authorize a bounded low-complexity model"
                ),
            }
        if mechanism == "no_measurable_low_frequency_gradient":
            return {
                **common_process,
                "decision": "preserve",
                "source": "process_evidence",
                "confidence": 1.0,
                "reason": (
                    "source-masked held-out sky shows no spatial variation above "
                    "patch-median uncertainty"
                ),
            }
        return {
            **common_process,
            "decision": "review_required",
            "source": "process_evidence",
            "confidence": 1.0,
            "reason": f"background mechanism requires review: {mechanism}",
        }

    return {
        "decision": "review_required",
        "source": "process_evidence",
        "confidence": 0.0,
        "reason": "source-masked Stage 3 process evidence is unavailable",
        "metrics": dict(adaptive or {}),
        "diffuse_context": dict(diffuse_context or {}),
        "threshold_basis": "process evidence required",
    }


def _stage3_outcome_reason_code(
    *,
    policy_abort_candidate_search: bool,
    failure_action: str,
    final_output_validation_rejected: bool,
    bg_ok: bool,
    stage_saved: bool,
    pattern_review_required: bool,
    compound_selected_degraded: bool,
    selected_gate_warnings: List[str],
    background_backup_used: bool,
    profile_fallback_used: bool,
    fallback_warning: bool,
    decision: str = "apply",
    user_preserve: bool = False,
    review_required: bool = False,
) -> str:
    """Return the single audited outcome reason used by every Stage 3 sink."""
    return (
        "failure_policy_stop"
        if policy_abort_candidate_search and failure_action == "stop"
        else "failure_policy_preserve_review"
        if policy_abort_candidate_search and failure_action == "preserve_review"
        else "final_output_validation_rejected"
        if final_output_validation_rejected
        else "stage3_output_save_failed"
        if not stage_saved
        else "background_not_required"
        if decision == "skip"
        else "user_preserve"
        if decision == "preserve" and user_preserve
        else "background_not_required_after_process_validation"
        if decision == "preserve"
        else "pattern_noise_deferred"
        if decision == "review_required" and pattern_review_required
        else "background_review_required"
        if decision == "review_required"
        else "no_background_candidate_accepted"
        if not bg_ok
        else "mixed_gradient_pattern_noise_review"
        if pattern_review_required
        else "compound_poly_residual_rbf_degraded_review"
        if compound_selected_degraded
        else "background_accepted_with_soft_warnings"
        if selected_gate_warnings
        else "background_backup_accepted"
        if background_backup_used
        else "target_profiler_fallback"
        if profile_fallback_used
        else "background_improvement_limited"
        if fallback_warning
        else "background_review_required"
        if review_required
        else "background_accepted"
    )


STAGE3_BELOW_UNCERTAINTY_WARNING = (
    "held-out span improvement is below sampling uncertainty"
)


def _stage3_below_uncertainty_only_validation(
    validation_gate: Any,
) -> Dict[str, Any]:
    """Identify the one rejected validation outcome allowed for a formal no-op."""
    payload = validation_gate if isinstance(validation_gate, dict) else {}
    required = {
        "accepted",
        "material_improvement",
        "span_not_worse",
        "background_rms_not_worse",
    }
    issues = list(
        dict.fromkeys(
            str(item)
            for item in [
                *(payload.get("warnings") or []),
                *(payload.get("issues") or []),
                *(payload.get("hard_issues") or []),
            ]
            if str(item)
        )
    )
    checks = {
        "complete": required.issubset(payload),
        "candidate_rejected": payload.get("accepted") is False,
        "below_three_sigma": payload.get("material_improvement") is False,
        "span_not_worse": bool(payload.get("span_not_worse", False)),
        "background_rms_not_worse": bool(
            payload.get("background_rms_not_worse", False)
        ),
        "only_allowed_issue": bool(
            issues
            and all(
                issue == STAGE3_BELOW_UNCERTAINTY_WARNING
                for issue in issues
            )
        ),
    }
    return {
        "eligible": all(checks.values()),
        "checks": checks,
        "issues": issues,
    }


def _stage3_replay_below_uncertainty_evidence(
    baseline_validation: Any,
    candidate_validation: Any,
    validation_gate: Any,
) -> Dict[str, Any]:
    """Recompute the held-out span decision instead of trusting gate booleans."""

    baseline = baseline_validation if isinstance(baseline_validation, dict) else {}
    candidate = (
        candidate_validation if isinstance(candidate_validation, dict) else {}
    )
    gate = validation_gate if isinstance(validation_gate, dict) else {}
    report: Dict[str, Any] = {
        "schema": "starun.stage3-noop-uncertainty-replay.v1",
        "status": "rejected",
        "accepted": False,
        "issues": [],
    }
    required_validation = {
        "status",
        "robust_span",
        "patch_mad_median",
        "patch_median_uncertainty",
        "patch_radius",
    }
    required_gate = {
        "baseline_span",
        "candidate_span",
        "span_improvement",
        "sampling_uncertainty_3sigma",
        "material_improvement",
        "span_not_worse",
        "background_rms_not_worse",
    }
    try:
        if not required_validation.issubset(baseline):
            raise ValueError("baseline held-out uncertainty evidence is incomplete")
        if not required_validation.issubset(candidate):
            raise ValueError("candidate held-out uncertainty evidence is incomplete")
        if not required_gate.issubset(gate):
            raise ValueError("reported held-out uncertainty decision is incomplete")
        if baseline.get("status") != "ready" or candidate.get("status") != "ready":
            raise ValueError("held-out uncertainty evidence is not ready")
        baseline_span = float(baseline["robust_span"])
        candidate_span = float(candidate["robust_span"])
        baseline_rms = float(baseline["patch_mad_median"])
        candidate_rms = float(candidate["patch_mad_median"])
        baseline_span_se = float(background_span_standard_error(baseline))
        candidate_span_se = float(background_span_standard_error(candidate))
        values = (
            baseline_span,
            candidate_span,
            baseline_rms,
            candidate_rms,
            baseline_span_se,
            candidate_span_se,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("held-out uncertainty evidence contains non-finite values")
        three_sigma = max(
            STAGE3_SIGNIFICANCE_SIGMA
            * math.hypot(baseline_span_se, candidate_span_se),
            1e-12,
        )
        span_improvement = baseline_span - candidate_span
        span_not_worse = bool(candidate_span <= baseline_span + three_sigma)
        material_improvement = bool(span_improvement > three_sigma)
        rms_not_worse = bool(candidate_rms <= baseline_rms + three_sigma)

        def reported_matches(name: str, expected: float) -> bool:
            try:
                reported = float(gate[name])
            except (KeyError, TypeError, ValueError):
                return False
            return bool(
                math.isfinite(reported)
                and math.isclose(
                    reported,
                    expected,
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                )
            )

        checks = {
            "baseline_span_matches": reported_matches(
                "baseline_span", baseline_span
            ),
            "candidate_span_matches": reported_matches(
                "candidate_span", candidate_span
            ),
            "span_improvement_matches": reported_matches(
                "span_improvement", span_improvement
            ),
            "three_sigma_matches": reported_matches(
                "sampling_uncertainty_3sigma", three_sigma
            ),
            "below_three_sigma": not material_improvement,
            "material_decision_matches": gate.get("material_improvement")
            is material_improvement,
            "span_not_worse": span_not_worse,
            "span_decision_matches": gate.get("span_not_worse")
            is span_not_worse,
            "background_rms_not_worse": rms_not_worse,
            "rms_decision_matches": gate.get("background_rms_not_worse")
            is rms_not_worse,
        }
        issues = [name for name, accepted in checks.items() if not accepted]
        report.update(
            status="accepted" if not issues else "rejected",
            accepted=not issues,
            issues=issues,
            checks=checks,
            recomputed={
                "baseline_span": baseline_span,
                "candidate_span": candidate_span,
                "span_improvement": span_improvement,
                "baseline_span_standard_error": baseline_span_se,
                "candidate_span_standard_error": candidate_span_se,
                "sampling_uncertainty_3sigma": three_sigma,
                "material_improvement": material_improvement,
                "span_not_worse": span_not_worse,
                "background_rms_not_worse": rms_not_worse,
            },
        )
        return report
    except (KeyError, TypeError, ValueError) as error:
        report["issues"] = [str(error)]
        return report


def _stage3_verified_noop_candidate_audit(
    process_report: Dict[str, Any],
    pattern_report: Dict[str, Any],
    noise_route: Dict[str, Any],
    attempts: List[Dict[str, Any]],
    *,
    baseline_validation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Prove that evaluated candidates add no benefit beyond three sigma."""
    true_sky = dict((process_report or {}).get("true_sky_support") or {})
    process_checks = {
        "true_sky_supported": bool(true_sky.get("supported", False)),
        "linear_input_confirmed": bool(
            ((process_report or {}).get("linear_input") or {}).get(
                "confirmed",
                False,
            )
        ),
        "no_process_hard_blocks": not bool(
            (process_report or {}).get("hard_block_reasons")
        ),
        "directional_pattern_measured": str(
            (pattern_report or {}).get("status") or ""
        )
        == "ok",
        "directional_pattern_clear": not bool(
            (pattern_report or {}).get("detected", False)
        ),
        "pattern_route_clear": not bool(
            (noise_route or {}).get("requires_review", False)
        ),
    }
    assessed: List[Dict[str, Any]] = []
    blockers: List[str] = []
    for record in attempts:
        label = str(record.get("label") or "candidate")
        validation_gate = record.get("validation_gate")
        required_gate_values = {
            "pixel_integrity": record.get("pixel_integrity_gate"),
            "target_fidelity": record.get("target_fidelity_gate"),
            "directional_pattern": record.get("pattern_quality_gate"),
            "directional_gradient": record.get("directional_gradient_gate"),
            "color_shift": record.get("color_shift_gate"),
        }
        gate_presence = {
            name: isinstance(gate, dict) and "accepted" in gate
            for name, gate in required_gate_values.items()
        }
        gate_checks = {
            name: bool(gate.get("accepted", False))
            if isinstance(gate, dict)
            else False
            for name, gate in required_gate_values.items()
        }
        validation_payload = (
            validation_gate if isinstance(validation_gate, dict) else {}
        )
        validation_evidence = _stage3_below_uncertainty_only_validation(
            validation_payload
        )
        uncertainty_replay = _stage3_replay_below_uncertainty_evidence(
            baseline_validation,
            record.get("validation"),
            validation_payload,
        )
        validation_checks = {
            "accepted": bool(validation_payload.get("accepted", False)),
            "candidate_rejected": validation_payload.get("accepted") is False,
            "below_three_sigma": (
                validation_payload.get("material_improvement") is False
            ),
            "span_not_worse": bool(
                validation_payload.get("span_not_worse", False)
            ),
            "background_rms_not_worse": bool(
                validation_payload.get("background_rms_not_worse", False)
            ),
        }
        warnings = list(
            dict.fromkeys(
                str(item)
                for item in [
                    *(validation_payload.get("warnings") or []),
                    *(validation_payload.get("issues") or []),
                    *(validation_payload.get("hard_issues") or []),
                    *(record.get("gate_warnings") or []),
                    *(record.get("issues") or []),
                    *(record.get("hard_issues") or []),
                ]
                if str(item)
            )
        )
        unexpected_warnings = [
            warning
            for warning in warnings
            if warning != STAGE3_BELOW_UNCERTAINTY_WARNING
        ]
        hard_metrics = bool(record.get("hard_gate_metrics_available", False))
        status = str(record.get("status") or "")
        checkpoint = record.get("candidate_checkpoint") or {}
        technical_checks = {
            "validation_gate_complete": bool(
                validation_evidence["checks"]["complete"]
            ),
            "required_gates_complete": all(gate_presence.values()),
            "candidate_checkpoint_saved": bool(
                record.get("candidate_stem")
                and isinstance(checkpoint, dict)
                and checkpoint.get("accepted") is True
                and checkpoint.get("status") == "accepted"
            ),
            "candidate_status_assessed": status == "rejected",
            "no_failure_reason": not bool(record.get("failure_reason")),
        }
        candidate_ok = bool(
            hard_metrics
            and all(technical_checks.values())
            and all(gate_checks.values())
            and validation_evidence["eligible"]
            and uncertainty_replay.get("accepted") is True
            and not unexpected_warnings
            and STAGE3_BELOW_UNCERTAINTY_WARNING in warnings
        )
        if not candidate_ok:
            blockers.append(label)
        assessed.append(
            {
                "label": label,
                "accepted_as_noop_evidence": candidate_ok,
                "status": status or "missing",
                "hard_gate_metrics_available": hard_metrics,
                "technical_checks": technical_checks,
                "gate_presence": gate_presence,
                "gate_checks": gate_checks,
                "validation_checks": validation_checks,
                "validation_evidence": validation_evidence,
                "uncertainty_replay": uncertainty_replay,
                "warnings": warnings,
                "unexpected_warnings": unexpected_warnings,
            }
        )
    eligible = bool(
        assessed
        and all(process_checks.values())
        and not blockers
        and all(
            bool(record.get("accepted_as_noop_evidence", False))
            for record in assessed
        )
    )
    return {
        "schema": "starun.stage3-verified-noop-candidate-audit.v1",
        "status": "eligible" if eligible else "ineligible",
        "eligible": eligible,
        "process_checks": process_checks,
        "assessed_candidate_count": len(assessed),
        "candidate_blockers": blockers,
        "candidates": assessed,
        "allowed_candidate_issue": STAGE3_BELOW_UNCERTAINTY_WARNING,
    }


def _stage3_restored_noop_pixel_gate(
    baseline_image: Any,
    restored_image: Any,
    *,
    gate_profile: str,
) -> Tuple[bool, Dict[str, Any]]:
    """Require exact pixels while ignoring only the ordinary must-change rule."""
    baseline = np.asarray(baseline_image)
    restored = np.asarray(restored_image)
    pixel_exact = bool(
        baseline.shape == restored.shape
        and baseline.dtype == restored.dtype
        and np.array_equal(baseline, restored, equal_nan=True)
    )
    ordinary_ok, ordinary_gate = _stage3_candidate_pixel_gate(
        baseline,
        restored,
        gate_profile=gate_profile,
    )
    ordinary_issues = list(
        dict.fromkeys(
            str(item)
            for item in [
                *(ordinary_gate.get("issues") or []),
                *(ordinary_gate.get("hard_issues") or []),
            ]
            if str(item)
        )
    )
    allowed_issue = "candidate command did not change any pixels"
    unexpected_issues = [
        issue for issue in ordinary_issues if issue != allowed_issue
    ]
    accepted = bool(
        pixel_exact
        and not unexpected_issues
        and (ordinary_ok or ordinary_issues == [allowed_issue])
    )
    report = {
        "schema": "starun.stage3-restored-noop-pixel-integrity.v1",
        "status": "accepted" if accepted else "rejected",
        "accepted": accepted,
        "pixel_exact": pixel_exact,
        "baseline_shape": [int(value) for value in baseline.shape],
        "restored_shape": [int(value) for value in restored.shape],
        "baseline_dtype": str(baseline.dtype),
        "restored_dtype": str(restored.dtype),
        "ordinary_candidate_gate": ordinary_gate,
        "allowed_ordinary_issue": allowed_issue,
        "unexpected_issues": unexpected_issues,
        "issues": (
            []
            if accepted
            else list(
                dict.fromkeys(
                    [
                        *(unexpected_issues or []),
                        *([] if pixel_exact else ["restored pixels are not exact"]),
                    ]
                )
            )
        ),
    }
    return accepted, report


def _stage3_absolute_background_gate(validation: Any) -> Dict[str, Any]:
    """Require true-sky samples to occupy an unclipped absolute unit range."""

    payload = validation if isinstance(validation, dict) else {}
    report: Dict[str, Any] = {
        "schema": "starun.stage3-absolute-background.v1",
        "status": "rejected",
        "accepted": False,
        "issues": [],
    }
    required = {
        "minimum",
        "maximum",
        "p10",
        "median",
        "p90",
        "robust_span",
        "supported_pixel_count",
        "low_clip_count",
        "high_clip_count",
    }
    try:
        if payload.get("status") != "ready" or not required.issubset(payload):
            raise ValueError("absolute background evidence is incomplete")
        minimum = float(payload["minimum"])
        maximum = float(payload["maximum"])
        p10 = float(payload["p10"])
        median = float(payload["median"])
        p90 = float(payload["p90"])
        span = float(payload["robust_span"])
        supported = int(payload["supported_pixel_count"])
        low_clip_count = int(payload["low_clip_count"])
        high_clip_count = int(payload["high_clip_count"])
        sample_count = int(payload.get("sample_count", 0) or 0)
        expected_count = int(payload.get("expected_count", 0) or 0)
        if not all(
            math.isfinite(value)
            for value in (minimum, maximum, p10, median, p90, span)
        ):
            raise ValueError("absolute background evidence contains non-finite values")
        checks = {
            "validation_samples_complete": bool(
                sample_count >= 4 and sample_count == expected_count
            ),
            "absolute_range_ordered": bool(
                0.0 < minimum <= p10 <= median <= p90 <= maximum < 1.0
                and 0.0 <= span < 1.0
            ),
            "black_level_above_zero": bool(minimum > 0.0),
            "bright_level_below_one": bool(maximum < 1.0),
            "sky_pixels_available": bool(supported > 0),
            "sky_clipping_absent": bool(
                low_clip_count == 0 and high_clip_count == 0
            ),
        }
        issues = [name for name, accepted in checks.items() if not accepted]
        report.update(
            status="accepted" if not issues else "rejected",
            accepted=not issues,
            issues=issues,
            checks=checks,
            thresholds={
                "domain_min_exclusive": 0.0,
                "domain_max_exclusive": 1.0,
                "low_clip_count_max": 0,
                "high_clip_count_max": 0,
            },
            measurements={
                "minimum": minimum,
                "p10": p10,
                "median": median,
                "p90": p90,
                "maximum": maximum,
                "robust_span": span,
                "supported_pixel_count": supported,
                "low_clip_count": low_clip_count,
                "high_clip_count": high_clip_count,
            },
        )
        return report
    except (TypeError, ValueError) as error:
        report["issues"] = [str(error)]
        return report


def _stage3_verify_restored_noop(
    pipeline,
    *,
    baseline_image: Any,
    baseline_validation: Dict[str, Any],
    validation_points: List[Tuple[float, float]],
    patch_radius: int,
    minimum_count: int,
    support_mask: Optional[np.ndarray],
    gate_profile: str,
    process_report: Dict[str, Any],
    pattern_report: Dict[str, Any],
) -> Dict[str, Any]:
    """Re-measure an exact rollback before it may become a formal no-op."""
    report: Dict[str, Any] = {
        "schema": "starun.stage3-verified-noop.v1",
        "status": "rejected",
        "accepted": False,
        "issues": [],
    }
    try:
        baseline = np.asarray(baseline_image)
        restored = np.asarray(
            pipeline.siril.get_image_pixeldata(preview=False)
        )
        pixel_ok, pixel_gate = _stage3_restored_noop_pixel_gate(
            baseline,
            restored,
            gate_profile=gate_profile,
        )
        pixel_exact = bool(pixel_gate.get("pixel_exact", False))
        candidate_validation = measure_background_validation(
            restored,
            validation_points,
            patch_radius=patch_radius,
            minimum_count=minimum_count,
            value_scale=baseline_validation.get("value_scale"),
            support_mask=support_mask,
        )
        validation_ok, validation_gate = assess_single_background_validation(
            baseline_validation,
            candidate_validation,
            gate_profile=gate_profile,
        )
        uncertainty_replay = _stage3_replay_below_uncertainty_evidence(
            baseline_validation,
            candidate_validation,
            validation_gate,
        )
        validation_issues = list(validation_gate.get("issues") or [])
        noop_validation_ok = bool(
            validation_ok
            or (
                validation_gate.get("material_improvement") is False
                and bool(validation_gate.get("span_not_worse", False))
                and bool(
                    validation_gate.get("background_rms_not_worse", False)
                )
                and all(
                    str(issue) == STAGE3_BELOW_UNCERTAINTY_WARNING
                    for issue in validation_issues
                )
            )
        )
        preservation = pipeline._stage3_signal_preservation_metrics(
            baseline,
            restored,
        )
        fidelity_ok, fidelity_gate = assess_target_fidelity(
            preservation,
            low_complexity_required=bool(
                (process_report or {}).get("low_complexity_required", False)
            ),
            gate_profile=gate_profile,
        )
        restored_pattern = analyze_directional_pattern_noise(
            restored,
            detection_threshold=_stage3_cfg_float(
                pipeline,
                "stage3_pattern_score_min",
                0.55,
                0.25,
                0.90,
            ),
            walking_threshold=_stage3_cfg_float(
                pipeline,
                "stage3_walking_noise_score_min",
                0.50,
                0.25,
                0.90,
            ),
        )
        pattern_ok, pattern_gate = pattern_candidate_gate(
            pattern_report,
            restored_pattern,
            growth_max=_stage3_cfg_float(
                pipeline,
                "stage3_pattern_score_growth_max",
                0.12,
                0.02,
                0.40,
            ),
            gate_profile=gate_profile,
        )
        direction_ok, direction_gate = assess_background_direction_reversal(
            baseline,
            restored,
            validation_points,
            patch_radius=patch_radius,
            support_mask=support_mask,
        )
        true_sky_ok = bool(
            ((process_report or {}).get("true_sky_support") or {}).get(
                "supported",
                False,
            )
        )
        absolute_background_gate = _stage3_absolute_background_gate(
            candidate_validation
        )
        absolute_background_ok = bool(
            absolute_background_gate.get("accepted", False)
        )
        checks = {
            "pixel_exact": pixel_exact,
            "pixel_integrity": bool(pixel_ok),
            "true_sky_support": true_sky_ok,
            "absolute_background": absolute_background_ok,
            "uncertainty_replay": bool(
                uncertainty_replay.get("accepted", False)
            ),
            "background_validation": noop_validation_ok,
            "target_fidelity": bool(fidelity_ok),
            "directional_pattern": bool(pattern_ok),
            "directional_gradient": bool(direction_ok),
        }
        issues = [name for name, accepted in checks.items() if not accepted]
        accepted = not issues
        report.update(
            status="accepted" if accepted else "rejected",
            accepted=accepted,
            issues=issues,
            checks=checks,
            baseline_pixel_sha256=_stage3_array_sha256(baseline),
            restored_pixel_sha256=_stage3_array_sha256(restored),
            pixel_integrity_gate=pixel_gate,
            absolute_background_gate=absolute_background_gate,
            validation=candidate_validation,
            validation_gate=validation_gate,
            uncertainty_replay=uncertainty_replay,
            target_fidelity_gate=fidelity_gate,
            pattern_quality_gate=pattern_gate,
            directional_gradient_gate=direction_gate,
        )
        return report
    except (
        AttributeError,
        CommandError,
        OSError,
        RuntimeError,
        SirilError,
        TypeError,
        ValueError,
    ) as error:
        report["issues"] = [str(error)]
        return report


def _stage3_verify_persisted_noop_output(
    pipeline,
    *,
    baseline_image: Any,
    output_stem: str,
) -> Dict[str, Any]:
    """Reload a saved no-op artifact and prove its decoded pixels are identical."""
    report: Dict[str, Any] = {
        "schema": "starun.stage3-persisted-noop-output.v1",
        "status": "rejected",
        "accepted": False,
        "output_stem": output_stem,
        "issues": [],
    }
    try:
        pipeline.cmd_with_check("load", output_stem, quiet=True)
        baseline = np.asarray(baseline_image)
        persisted = np.asarray(
            pipeline.siril.get_image_pixeldata(preview=False)
        )
        exact = bool(
            baseline.shape == persisted.shape
            and baseline.dtype == persisted.dtype
            and np.array_equal(baseline, persisted, equal_nan=True)
        )
        report.update(
            status="accepted" if exact else "rejected",
            accepted=exact,
            checks={
                "shape_equal": baseline.shape == persisted.shape,
                "dtype_equal": baseline.dtype == persisted.dtype,
                "pixels_exact": exact,
            },
            baseline_pixel_sha256=_stage3_array_sha256(baseline),
            persisted_pixel_sha256=_stage3_array_sha256(persisted),
            issues=(
                []
                if exact
                else ["persisted Stage 3 no-op pixels differ from the input"]
            ),
        )
        return report
    except (
        AttributeError,
        CommandError,
        OSError,
        RuntimeError,
        SirilError,
        TypeError,
        ValueError,
    ) as error:
        report["issues"] = [str(error)]
        return report


def run_stage3_background_extraction(pipeline) -> None:
    """
    阶段 3: 背景提取
    - 先评估目标感知的内置 Polynomial/RBF 候选
    - 内置候选不足时依次评估复合模型、GraXpert 和外部插件
    - 输出优先门禁将一般偏差降为软告警，仅过度异常硬拒绝
    - 条纹和 walking noise 与天空梯度分流，禁止用背景模型静默吞掉结构噪声
    - 每个候选成功后执行质量门控，避免过度扣背景
    - 候选命令失败或未达到充分质量时，继续尝试下一个备用候选
    """
    stage_label = PipelineStage.BACKGROUND_EXTRACTION.label
    pipeline.log.stage_start(stage_label)
    pipeline._clear_stage_reviews(3)
    pipeline._background_review_required = False
    pipeline._stage3_graxpert_provenance = None
    bg_ok = False
    selected_source = ""
    preflight_message = ""
    if hasattr(pipeline, "_run_target_profile_preflight"):
        preflight_message = pipeline._run_target_profile_preflight(
            source="Stage3 preflight",
            metadata_candidates=("stage2_corrected", getattr(pipeline, "source_file", None)),
            preview_name="stage3_target_preview.png",
        )
    stage_message = preflight_message
    policy = getattr(pipeline, "pipeline_policy", {}) or {}
    policy_name = policy.get("policy_name", "generic_low_snr_safe") if isinstance(policy, dict) else "generic_low_snr_safe"
    stage3_policy = policy.get("stage3_background", {}) if isinstance(policy, dict) else {}
    configured_gate_profile = normalize_stage3_gate_profile(
        getattr(pipeline.cfg, "stage3_gate_profile", "output_first")
    )
    gate_profile = configured_gate_profile
    decision_thresholds = _stage3_decision_thresholds(
        pipeline,
        stage3_policy,
        gate_profile_override=gate_profile,
    )
    pipeline.log.info(
        "[Stage3] Background policy: "
        f"policy={policy_name} protect_nebulosity={bool(stage3_policy.get('protect_nebulosity', False))} "
        f"model={','.join(stage3_policy.get('model_priority', []) or [])} "
        f"gate_profile={configured_gate_profile}"
    )

    baseline_stem = "stage3_bg_input"
    baseline_saved = False
    rollback_events: List[Dict[str, Any]] = []
    try:
        pipeline.cmd_with_check("save", baseline_stem)
        baseline_saved = True
    except (CommandError, SirilError) as e:
        pipeline.log.warn(
            "stage3 baseline save failed; skip destructive background candidates: "
            f"{e}"
        )

    def restore_baseline(context: str) -> bool:
        nonlocal baseline_saved
        if not baseline_saved:
            rollback_events.append(
                {
                    "context": context,
                    "status": "unavailable",
                    "reason": "stage3 baseline checkpoint unavailable",
                }
            )
            return False
        try:
            pipeline.cmd_with_check("load", baseline_stem, quiet=True)
            rollback_events.append({"context": context, "status": "restored"})
            return True
        except (CommandError, SirilError) as e:
            baseline_saved = False
            rollback_events.append(
                {
                    "context": context,
                    "status": "failed",
                    "reason": str(e),
                }
            )
            pipeline.log.warn(f"failed to restore stage3 baseline ({context}): {e}")
            return False

    before_image = None
    try:
        before_image = pipeline.siril.get_image_pixeldata(preview=False)
    except (CommandError, SirilError, OSError, RuntimeError, TypeError, ValueError) as e:
        pipeline.log.debug(f"stage3 baseline image sampling skipped: {e}")
    scene_support_runtime: Dict[str, Any] = {
        "status": "unavailable",
        "manifest": scene_support.unavailable_scene_support(
            "stage3 baseline pixels unavailable",
            reason_code="stage3_scene_support_pixels_unavailable",
        ),
        "valid_mask": None,
        "saturation_map": None,
    }
    if getattr(pipeline, "process_dir", None) is not None:
        try:
            if before_image is None:
                manifest = scene_support.write_unavailable_scene_support(
                    pipeline.process_dir,
                    "stage3 baseline pixels unavailable",
                    reason_code="stage3_scene_support_pixels_unavailable",
                )
                scene_support_runtime["manifest"] = manifest
            else:
                scene_support.build_scene_support(
                    before_image,
                    pipeline.process_dir,
                    source_path=pipeline.process_dir / f"{baseline_stem}.fit",
                )
                scene_support_runtime = scene_support.load_scene_support(
                    pipeline.process_dir,
                    expected_shape=tuple(np.asarray(before_image).shape),
                )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            pipeline.log.debug(f"stage3 shared scene support unavailable: {error}")
            try:
                manifest = scene_support.write_unavailable_scene_support(
                    pipeline.process_dir,
                    str(error),
                    reason_code="stage3_scene_support_build_failed",
                )
                scene_support_runtime["manifest"] = manifest
            except OSError:
                pass
    pipeline._stage3_scene_support = scene_support_runtime
    scene_support_report = scene_support.scene_support_summary(
        scene_support_runtime
    )
    before_feat = pipeline._stage3_measure_features("before")
    before_adaptive = (
        pipeline._adaptive_features_current()
        if hasattr(pipeline, "_adaptive_features_current")
        else {}
    )

    pattern_routing_enabled = bool(
        getattr(pipeline.cfg, "stage3_pattern_routing_enabled", True)
    )
    if pattern_routing_enabled and before_image is not None:
        pattern_report = analyze_directional_pattern_noise(
            before_image,
            detection_threshold=_stage3_cfg_float(
                pipeline,
                "stage3_pattern_score_min",
                0.55,
                0.25,
                0.90,
            ),
            walking_threshold=_stage3_cfg_float(
                pipeline,
                "stage3_walking_noise_score_min",
                0.50,
                0.25,
                0.90,
            ),
        )
    else:
        pattern_report = {
            "status": "disabled" if not pattern_routing_enabled else "unavailable",
            "detected": False,
            "reason": (
                "disabled by configuration"
                if not pattern_routing_enabled
                else "baseline pixels unavailable"
            ),
        }
    noise_route: Dict[str, Any] = {}

    attempt_records: List[Dict[str, Any]] = []
    selected_preservation: Dict[str, Any] = {}
    selected: Dict[str, Any] = {}
    selected_loaded = False
    accepted_candidates: List[Dict[str, Any]] = []
    builtin_sufficient = False
    graxpert_attempted = False
    graxpert_runtime_error = False
    graxpert_error_reasons: List[str] = []
    selected_label = ""
    selected_gate_warnings: List[str] = []
    selected_pattern_report: Dict[str, Any] = {}
    selected_spatial_opponent_projection: Dict[str, Any] = {}
    selected_spatial_opponent_review_required = False
    verified_color_normalization: Dict[str, Any] = {
        "schema": "starun.stage3-verified-background-color-normalization.v1",
        "applied": False,
        "accepted": False,
        "reason_code": "not_evaluated",
    }
    builtin_order_reason = "safe_samples_rbf_before_poly"
    builtin_search_mode = "safe_samples_primary"
    diffuse_context: Dict[str, Any] = {}
    safe_sample_points: List[Tuple[float, float]] = []
    safe_sample_report: Dict[str, Any] = {
        "status": "not_run",
        "sample_count": 0,
    }
    safe_sample_recovery_used = False
    masked_catalog_sampling_used = False
    safe_sample_support_mask: Optional[np.ndarray] = None
    masked_fit_sample_records: List[Dict[str, Any]] = []
    safe_sample_recovery_report: Dict[str, Any] = {
        "schema_version": "starun.stage3-safe-sample-recovery.v1",
        "status": "not_applicable",
        "triggered": False,
        "reason_code": None,
    }
    compound_fit_points: List[Tuple[float, float]] = []
    compound_validation_points: List[Tuple[float, float]] = []
    compound_split_report: Dict[str, Any] = {
        "status": "not_run",
    }
    compound_target_guard: Dict[str, Any] = {
        "status": "not_evaluated",
        "eligible": False,
        "reasons": [],
    }
    baseline_validation: Dict[str, Any] = {
        "status": "not_run",
    }
    compound_report: Dict[str, Any] = {
        "status": "not_triggered",
        "triggered": False,
    }
    compound_selected = False
    compound_selected_degraded = False
    selection_report: Dict[str, Any] = {
        "status": "not_run",
        "method": "pareto_dense_rank_sum_v2",
        "runtime_selection_affected": True,
    }
    final_output_validation: Dict[str, Any] = {
        "status": "not_run",
        "enforced": False,
    }
    final_output_validation_rejected = False
    spatial_background_lineage: Dict[str, Any] = {
        "schema": STAGE3_SPATIAL_LINEAGE_SCHEMA,
        "status": "not_applicable",
        "accepted": False,
    }
    verified_noop = False
    verified_noop_candidate_audit: Dict[str, Any] = {
        "schema": "starun.stage3-verified-noop-candidate-audit.v1",
        "status": "not_evaluated",
        "eligible": False,
    }
    verified_noop_report: Dict[str, Any] = {
        "schema": "starun.stage3-verified-noop.v1",
        "status": "not_evaluated",
        "accepted": False,
    }
    failure_action = str(
        getattr(pipeline.cfg, "stage3_failure_action", "auto_fallback")
    )
    candidate_attempt_limit = max(
        0,
        int(getattr(pipeline.cfg, "stage3_candidate_attempt_limit", 0) or 0),
    )
    policy_abort_candidate_search = False
    policy_abort_reason = ""
    attempted_selected_label = ""

    target_profile = getattr(pipeline, "target_profile", {}) or {}
    profile_fallback_used = bool(
        isinstance(target_profile, dict)
        and str(
            target_profile.get("classification_method") or ""
        ).strip().lower()
        == "fallback"
    )
    _diffuse, diffuse_context = _stage3_diffuse_nebula_context(
        target_profile,
        before_adaptive,
        stage3_policy=stage3_policy,
    )
    diffuse_context["diffuse"] = bool(_diffuse)
    protection_policy_flags = {
        "protect_nebulosity": bool(stage3_policy.get("protect_nebulosity")),
        "reject_samples_on_nebula": bool(
            stage3_policy.get("reject_samples_on_nebula")
        ),
        "protect_bright_core": bool(stage3_policy.get("protect_bright_core")),
        "protect_star_halo": bool(stage3_policy.get("protect_star_halo")),
        "protect_outer_halo": bool(stage3_policy.get("protect_outer_halo")),
        "protect_dark_structure": bool(
            stage3_policy.get("protect_dark_structure")
        ),
        "faint_nebula_signal": bool(
            diffuse_context.get("faint_nebula_protection")
        ),
    }
    sample_refinement_blocked = bool(
        diffuse_context.get("diffuse")
        or diffuse_context.get("emission_diffuse")
        or diffuse_context.get("large_nebulosity_feature")
        or diffuse_context.get("faint_nebula_protection")
        or diffuse_context.get("pixel_signal_protection")
    )
    input_profile = getattr(pipeline, "input_profile", None)
    process_profile = (
        input_profile
        if isinstance(input_profile, dict)
        else None
    )
    input_state = str((process_profile or {}).get("state") or "unknown").lower()
    linear_confirmed = bool(
        input_state == "linear"
        and (process_profile or {}).get("safe_for_linear_steps", True) is not False
    )

    def sample_report_summary(report: Dict[str, Any]) -> Dict[str, Any]:
        coverage = report.get("coverage") or {}
        minimum_count = int(report.get("minimum_count") or 12)
        selected_count = int(report.get("selected_candidate_count") or 0)
        grid_cells = int(coverage.get("grid_cells") or 0)
        return {
            "status": report.get("status"),
            "selected_candidate_count": selected_count,
            "safe_candidate_count": int(report.get("safe_candidate_count") or 0),
            "count_deficit": max(0, minimum_count - selected_count),
            "grid_cell_deficit": max(
                0,
                STAGE3_MIN_SPATIAL_GRID_CELLS - grid_cells,
            ),
            "coverage": coverage,
            "selected_candidate_sources": (
                report.get("selected_candidate_sources") or {}
            ),
            "rejection_counts": report.get("rejection_counts") or {},
        }

    if before_image is not None:
        shared_manifest = scene_support_runtime.get("manifest") or {}
        shared_components = shared_manifest.get("components") or {}
        shared_catalog = shared_components.get("star_catalog") or {}
        sample_kwargs = {
            "target_count": _stage3_cfg_int(
                pipeline,
                "stage3_safe_sample_target_count",
                40,
                16,
                64,
            ),
            "min_count": _stage3_cfg_int(
                pipeline,
                "stage3_safe_sample_min_count",
                12,
                12,
                48,
            ),
            "patch_radius": _stage3_cfg_int(
                pipeline,
                "stage3_safe_sample_patch_radius",
                12,
                4,
                24,
            ),
            "brightness_quantile_max": _stage3_cfg_float(
                pipeline,
                "stage3_safe_sample_brightness_quantile_max",
                0.70,
                0.50,
                0.85,
            ),
            "texture_quantile_max": _stage3_cfg_float(
                pipeline,
                "stage3_safe_sample_texture_quantile_max",
                0.55,
                0.25,
                0.75,
            ),
            "shared_valid_mask": scene_support_runtime.get("valid_mask"),
            "shared_saturation_map": scene_support_runtime.get("saturation_map"),
            "shared_star_catalog": (
                shared_catalog.get("records")
                if shared_catalog.get("status") == "available"
                else None
            ),
            "shared_star_catalog_sha256": shared_catalog.get(
                "records_sha256"
            ),
            "protection_policy": stage3_policy,
            "return_candidate_independent_support_mask": True,
        }
        safe_sample_points, safe_sample_report = build_safe_background_samples(
            before_image,
            candidate_refinement=not sample_refinement_blocked,
            **sample_kwargs,
        )
        safe_sample_support_mask = safe_sample_report.pop(
            "_candidate_independent_sky_support_mask",
            None,
        )
        safe_sample_report.pop("_masked_pixel_support_mask", None)
        if sample_refinement_blocked:
            base_report = safe_sample_report
            base_points = safe_sample_points
            base_masks = base_report.get("masks") or {}
            base_mask_evidence = base_report.get("mask_evidence") or {}
            base_shared = base_report.get("shared_scene_support") or {}
            backend_policy = str(
                getattr(pipeline.cfg, "stage3_backend_policy", "auto_chain")
            ).strip().lower()
            common_eligibility_checks = {
                "base_grid_insufficient": base_report.get("status") != "ready",
                "linear_input_confirmed": linear_confirmed,
                "source_mask_fraction": float(
                    base_masks.get("source_mask_fraction", 1.0) or 1.0
                ) <= STAGE3_MAX_SOURCE_MASK_FRACTION,
                "mask_evidence_applied": bool(
                    base_mask_evidence.get("applied_to_sampling")
                ),
                "valid_mask_applied": base_shared.get("valid_mask") == "applied",
                "saturation_map_applied": (
                    base_shared.get("saturation_map") == "applied"
                ),
                "star_catalog_trusted": base_shared.get("star_catalog")
                in {"applied", "available_empty"},
                "directional_pattern_cleared": bool(
                    pattern_report.get("status") == "ok"
                    and not pattern_report.get("detected", False)
                ),
                "builtin_backend_allowed": backend_policy
                in {"auto_chain", "builtin_only"},
            }
            strict_eligibility_checks = {
                **common_eligibility_checks,
                "usable_sky_fraction": float(
                    base_mask_evidence.get("usable_sky_fraction", 0.0) or 0.0
                ) >= STAGE3_MIN_USABLE_SKY_FRACTION,
            }
            base_rejections = base_report.get("rejection_counts") or {}
            base_candidate_count = int(
                base_report.get("base_candidate_count")
                or base_report.get("candidate_count")
                or 0
            )
            catalog_rejections = int(
                base_rejections.get("shared_catalog_star") or 0
            )
            catalog_dominated = bool(
                catalog_rejections
                >= max(
                    int(sample_kwargs["min_count"]),
                    int(math.ceil(base_candidate_count * 0.50)),
                )
            )
            scene_star_fraction = float(
                ((base_mask_evidence.get("layers") or {}).get(
                    "scene_support_stars"
                ) or {}).get("pixel_fraction")
                or 0.0
            )
            dense_eligibility_checks = {
                **common_eligibility_checks,
                "catalog_rejection_dominated": catalog_dominated,
                "dense_star_field": scene_star_fraction
                >= STAGE3_DENSE_STAR_FIELD_FRACTION_MIN,
                "nonstellar_sky_fraction": float(
                    base_masks.get("usable_sky_fraction", 0.0) or 0.0
                ) >= STAGE3_MIN_USABLE_SKY_FRACTION,
                "strict_sky_requires_masked_statistics": float(
                    base_mask_evidence.get("usable_sky_fraction", 0.0) or 0.0
                ) < STAGE3_MIN_USABLE_SKY_FRACTION,
            }
            strict_recovery_eligible = all(strict_eligibility_checks.values())
            dense_recovery_eligible = all(dense_eligibility_checks.values())
            recovery_eligible = bool(
                strict_recovery_eligible or dense_recovery_eligible
            )
            recovery_mode = (
                "strict_zero_overlap"
                if strict_recovery_eligible
                else "masked_catalog_statistics"
                if dense_recovery_eligible
                else None
            )
            safe_sample_recovery_report = {
                **safe_sample_recovery_report,
                "status": (
                    "not_required"
                    if base_report.get("status") == "ready"
                    else "eligible"
                    if recovery_eligible
                    else "ineligible"
                ),
                "triggered": bool(
                    base_report.get("status") != "ready" and recovery_eligible
                ),
                "reason_code": (
                    None
                    if base_report.get("status") == "ready"
                    else "stage3_safe_sample_recovery_applied"
                    if recovery_eligible
                    else "stage3_safe_sample_recovery_ineligible"
                ),
                "eligibility": strict_eligibility_checks,
                "dense_star_eligibility": dense_eligibility_checks,
                "recovery_mode": recovery_mode,
                "strict_unmasked_sky_fraction": float(
                    base_mask_evidence.get(
                        "strict_unmasked_sky_fraction",
                        base_mask_evidence.get("usable_sky_fraction", 0.0),
                    )
                    or 0.0
                ),
                "nonstellar_sky_fraction": float(
                    base_masks.get("usable_sky_fraction", 0.0) or 0.0
                ),
                "base_grid": sample_report_summary(base_report),
                "configured_gate_profile": configured_gate_profile,
                "effective_gate_profile": configured_gate_profile,
            }
            if recovery_eligible:
                refined_points, refined_report = build_safe_background_samples(
                    before_image,
                    candidate_refinement=True,
                    preserve_regular_grid=True,
                    masked_catalog_statistics=dense_recovery_eligible,
                    **sample_kwargs,
                )
                refined_support_mask = refined_report.pop(
                    "_candidate_independent_sky_support_mask",
                    None,
                )
                refined_report.pop("_masked_pixel_support_mask", None)
                if dense_recovery_eligible:
                    thresholds_frozen = bool(
                        refined_report.get("thresholds")
                        and int(refined_report.get("base_candidate_count") or 0)
                        >= int(sample_kwargs["min_count"])
                    )
                else:
                    thresholds_frozen = bool(
                        refined_report.get("thresholds")
                        == base_report.get("thresholds")
                    )
                refined_ready = bool(
                    refined_report.get("status") == "ready"
                    and refined_points
                    and thresholds_frozen
                    and refined_support_mask is not None
                )
                safe_sample_recovery_report.update(
                    status="applied" if refined_ready else "failed",
                    reason_code=(
                        "stage3_dense_star_masked_sampling_applied"
                        if refined_ready and dense_recovery_eligible
                        else "stage3_safe_sample_recovery_applied"
                        if refined_ready
                        else "stage3_dense_star_masked_sampling_ineligible"
                        if dense_recovery_eligible
                        else "stage3_safe_sample_recovery_ineligible"
                    ),
                    thresholds_frozen=thresholds_frozen,
                    refined=sample_report_summary(refined_report),
                )
                if refined_ready:
                    safe_sample_support_mask = refined_support_mask
                    safe_sample_recovery_used = True
                    masked_catalog_sampling_used = bool(
                        dense_recovery_eligible
                    )
                    gate_profile = "strict"
                    decision_thresholds = _stage3_decision_thresholds(
                        pipeline,
                        stage3_policy,
                        gate_profile_override=gate_profile,
                    )
                    safe_sample_points = refined_points
                    safe_sample_report = refined_report
                    safe_sample_recovery_report["effective_gate_profile"] = (
                        gate_profile
                    )
                else:
                    safe_sample_support_mask = None
                    safe_sample_points = []
                    safe_sample_report = refined_report
            else:
                safe_sample_points = base_points
                safe_sample_report = base_report
        safe_sample_report = {
            **safe_sample_report,
            "recovery": safe_sample_recovery_report,
        }
    else:
        safe_sample_report = {
            "status": "unavailable",
            "sample_count": 0,
            "error": "baseline pixels unavailable",
            "recovery": safe_sample_recovery_report,
        }

    selected_sample_records = safe_sample_report.get("selected_samples") or []
    selected_point_sources = (
        [
            str(sample.get("source") or "unknown")
            for sample in selected_sample_records
        ]
        if len(selected_sample_records) == len(safe_sample_points)
        else None
    )

    def sample_records_for_points(
        points: List[Tuple[float, float]],
    ) -> List[Dict[str, Any]]:
        by_point = {
            (
                round(float(record["point"][0]), 6),
                round(float(record["point"][1]), 6),
            ): record
            for record in selected_sample_records
            if isinstance(record, dict)
            and isinstance(record.get("point"), (list, tuple))
            and len(record.get("point")) >= 2
        }
        return [
            by_point.get(
                (round(float(point[0]), 6), round(float(point[1]), 6))
            )
            for point in points
            if by_point.get(
                (round(float(point[0]), 6), round(float(point[1]), 6))
            )
            is not None
        ]

    if before_image is not None and safe_sample_points:
        (
            compound_fit_points,
            compound_validation_points,
            compound_split_report,
        ) = split_background_sample_points(
            safe_sample_points,
            before_image,
            point_sources=selected_point_sources,
            minimum_regular_validation=(4 if safe_sample_recovery_used else 0),
            validation_ratio=_stage3_cfg_float(
                pipeline,
                "stage3_compound_validation_ratio",
                0.25,
                0.15,
                0.35,
            ),
            minimum_total=_stage3_cfg_int(
                pipeline,
                "stage3_compound_min_sample_count",
                12,
                12,
                64,
            ),
            minimum_fit=_stage3_cfg_int(
                pipeline,
                "stage3_compound_fit_min_count",
                8,
                8,
                56,
            ),
            minimum_validation=_stage3_cfg_int(
                pipeline,
                "stage3_compound_validation_min_count",
                4,
                4,
                20,
            ),
        )
        if compound_split_report.get("status") == "ready":
            if masked_catalog_sampling_used:
                masked_fit_sample_records = sample_records_for_points(
                    compound_fit_points
                )
                if len(masked_fit_sample_records) != len(compound_fit_points):
                    compound_split_report = {
                        **compound_split_report,
                        "status": "unavailable",
                        "reason": (
                            "masked BGSample fit provenance does not match split"
                        ),
                    }
            if compound_split_report.get("status") == "ready":
                baseline_validation = measure_background_validation(
                    before_image,
                    compound_validation_points,
                    patch_radius=_stage3_cfg_int(
                        pipeline,
                        "stage3_safe_sample_patch_radius",
                        12,
                        4,
                        24,
                    ),
                    minimum_count=_stage3_cfg_int(
                        pipeline,
                        "stage3_compound_validation_min_count",
                        4,
                        4,
                        20,
                    ),
                    support_mask=(
                        safe_sample_support_mask
                    ),
                )
        if (
            compound_split_report.get("status") != "ready"
            and safe_sample_recovery_used
        ):
            safe_sample_recovery_used = False
            masked_catalog_sampling_used = False
            safe_sample_support_mask = None
            masked_fit_sample_records = []
            safe_sample_recovery_report.update(
                status="failed",
                reason_code="stage3_safe_sample_provenance_split_failed",
                effective_gate_profile="strict",
                split_status=compound_split_report.get("status"),
                split_reason=compound_split_report.get("reason"),
            )
            safe_sample_points = []
            safe_sample_report = {
                **safe_sample_report,
                "status": "insufficient_safe_coverage",
                "sample_count": 0,
                "recovery": safe_sample_recovery_report,
            }
    else:
        compound_split_report = {
            "status": "unavailable",
            "reason": "audited samples or baseline pixels are unavailable",
        }

    process_report = assess_background_process(
        before_image,
        safe_sample_points,
        safe_sample_report,
        baseline_validation,
        pattern_report,
        input_profile=process_profile,
        diffuse_context=diffuse_context,
        patch_radius=_stage3_cfg_int(
            pipeline,
            "stage3_safe_sample_patch_radius",
            12,
            4,
            24,
        ),
        support_mask=(
            safe_sample_support_mask
        ),
    ) if before_image is not None else {
        "status": "review_required",
        "should_evaluate": False,
        "mechanism": "unavailable",
        "hard_block_reasons": ["baseline_pixels_unavailable"],
    }
    noise_route = select_background_route(
        before_adaptive,
        pattern_report,
        process_report=process_report,
    )
    if not pattern_routing_enabled:
        noise_route.update(
            route="routing_disabled",
            pattern_detected=False,
            subsky_existing_allowed=True,
            requires_review=False,
            reason="directional-pattern routing disabled by configuration",
        )
    pipeline._stage3_pattern_noise_report = {
        "analysis": pattern_report,
        "route": noise_route,
    }
    compound_target_guard = _stage3_compound_target_guard(
        target_profile,
        diffuse_context,
        stage3_policy,
        noise_route,
    )
    safe_sample_report = {
        **safe_sample_report,
        "fit_validation_split": compound_split_report,
        "compound_target_guard": compound_target_guard,
        "baseline_validation": baseline_validation,
    }
    pipeline._stage3_safe_sample_report = safe_sample_report
    pipeline.log.info(
        "[Stage3] Process evidence: "
        f"linear={bool((process_report.get('linear_input') or {}).get('confirmed'))} "
        f"samples={safe_sample_report.get('status')}:"
        f"{int(safe_sample_report.get('sample_count') or 0)} "
        f"holdout={compound_split_report.get('status')} "
        f"mechanism={process_report.get('mechanism')}"
    )
    background_decision = _stage3_background_decision(
        pipeline,
        before_adaptive,
        diffuse_context=diffuse_context,
        process_report=(
            process_report
            if isinstance(input_profile, dict)
            else None
        ),
    )
    if noise_route.get("route") == "pattern_noise_deferred":
        background_decision = {
            **background_decision,
            "decision": "review_required",
            "source": "pattern_noise_router",
            "confidence": max(
                float(background_decision.get("confidence") or 0.0),
                float(pattern_report.get("pattern_score") or 0.0),
            ),
            "reason": str(noise_route.get("reason") or "pattern noise deferred"),
            "pre_route_decision": dict(background_decision),
            "noise_route": noise_route,
        }
    elif noise_route.get("requires_review") and str(
        background_decision.get("decision") or "review_required"
    ) != "apply":
        background_decision = {
            **background_decision,
            "decision": "review_required",
            "source": "pattern_noise_router",
            "confidence": max(
                float(background_decision.get("confidence") or 0.0),
                float(pattern_report.get("pattern_score") or 0.0),
            ),
            "reason": str(
                noise_route.get("reason")
                or "directional pattern noise requires review"
            ),
            "pre_route_decision": dict(background_decision),
            "noise_route": noise_route,
        }
    else:
        background_decision["noise_route"] = noise_route
    user_preserve = (
        str(getattr(pipeline.cfg, "stage3_processing_mode", "auto"))
        == "preserve"
    )
    if user_preserve:
        diagnostic_decision = dict(background_decision)
        if str(diagnostic_decision.get("decision") or "") == "review_required":
            pipeline._background_review_required = True
        background_decision = {
            **background_decision,
            "decision": "preserve",
            "source": "user_processing_parameters",
            "confidence": 1.0,
            "reason": "user requested background preservation",
            "diagnostic_decision": diagnostic_decision,
        }
    pipeline._stage3_background_decision = background_decision
    decision = str(background_decision.get("decision") or "review_required")
    pipeline.log.info(
        "[Stage3] Background decision: "
        f"decision={decision} source={background_decision.get('source')} "
        f"confidence={float(background_decision.get('confidence') or 0.0):.2f}"
    )
    if decision != "apply":
        rollback_verified = restore_baseline(f"decision:{decision}")
        stage_saved = pipeline._save_stage_output("stage3_bgremoved")
        after_adaptive = dict(before_adaptive or {})
        # A pre-candidate skip/preserve has not passed the strict verified-noop
        # audit below. It is diagnostic passthrough only, never a formal
        # substitute for an accepted correction or verified_noop.
        pipeline._background_review_required = True
        reason = str(
            background_decision.get("reason")
            or "background extraction was not authorized"
        )
        reason_code = _stage3_outcome_reason_code(
            policy_abort_candidate_search=False,
            failure_action=failure_action,
            final_output_validation_rejected=False,
            bg_ok=False,
            stage_saved=stage_saved,
            pattern_review_required=bool(noise_route.get("requires_review", False)),
            compound_selected_degraded=False,
            selected_gate_warnings=[],
            background_backup_used=False,
            profile_fallback_used=profile_fallback_used,
            fallback_warning=False,
            decision=decision,
            user_preserve=user_preserve,
        )
        if decision in {"skip", "preserve"}:
            reason_code = "stage3_passthrough_requires_verified_noop"
        if (
            decision == "review_required"
            and "insufficient_source_masked_true_sky_support"
            in (process_report.get("hard_block_reasons") or [])
        ):
            reason_code = "insufficient_source_masked_true_sky_support"
        if not rollback_verified:
            reason_code = "stage3_passthrough_rollback_unverified"
        pipeline._require_review(3, reason_code)
        spatial_background_lineage = _stage3_write_spatial_background_lineage(
            pipeline,
            baseline_image=before_image,
            fit_points=compound_fit_points,
            validation_points=compound_validation_points,
            patch_radius=_stage3_cfg_int(
                pipeline,
                "stage3_safe_sample_patch_radius",
                12,
                4,
                24,
            ),
            support_mask=(
                safe_sample_support_mask
            ),
            projection={},
            review_required=True,
            processing_route=f"unverified_{decision}_passthrough",
        )
        if hasattr(pipeline, "_write_stage_json"):
            pipeline._write_stage_json(
                "background_quality_report.json",
                {
                    "schema_version": STAGE3_BACKGROUND_QUALITY_SCHEMA,
                    "algorithm_contract_version": (
                        STAGE3_ALGORITHM_CONTRACT_VERSION
                    ),
                    "stage": "stage3_background",
                    "reason_code": reason_code,
                    "configured_gate_profile": configured_gate_profile,
                    "effective_gate_profile": gate_profile,
                    "policy": policy_name,
                    "decision_thresholds": decision_thresholds,
                    "decision": background_decision,
                    "process_evidence": process_report,
                    "model_used": None,
                    "backend_provenance": {
                        "graxpert": getattr(
                            pipeline,
                            "_stage3_graxpert_provenance",
                            None,
                        ),
                    },
                    "candidate_order": [],
                    "attempts": [],
                    "neutral_axis_projection": None,
                    "selection": {
                        **selection_report,
                        "status": "not_applicable",
                        "reason": f"decision={decision}",
                    },
                    "final_output_validation": {
                        **final_output_validation,
                        "status": "not_applicable",
                        "reason": f"decision={decision}",
                    },
                    "rollback_events": rollback_events,
                    "diffuse_nebula_context": diffuse_context,
                    "directional_pattern_noise": pattern_report,
                    "noise_route": noise_route,
                    "safe_samples": safe_sample_report,
                    "protection_policy_flags": protection_policy_flags,
                    "shared_scene_support": scene_support_report,
                    "spatial_background_lineage": spatial_background_lineage,
                    "before": before_adaptive,
                    "after": after_adaptive,
                    "quality": "review_required",
                    "review_required": True,
                    "fallback_used": False,
                },
            )
        message = f"decision={decision}; {reason}"
        if preflight_message:
            message = f"{preflight_message}; {message}"
        if not stage_saved:
            message += "; stage3 输出保存失败"
        elapsed = pipeline.log.stage_end(stage_label)
        pipeline._record_stage(
            stage_label,
            "degraded",
            elapsed,
            message,
            execution="safe_passthrough",
            fallback_used=profile_fallback_used,
            reason_code=reason_code,
            components={
                "target_profile": {
                    "status": "applied",
                    "method": target_profile.get("classification_method"),
                    "reason_code": (
                        "target_profiler_fallback"
                        if profile_fallback_used
                        else "accepted"
                    ),
                    "fallback_used": profile_fallback_used,
                },
                "background_extraction": {
                    "status": (
                        "skipped"
                        if decision == "skip"
                        else "preserved"
                        if decision == "preserve"
                        else "rolled_back"
                    ),
                    "method": None,
                    "reason_code": (
                        "background_not_required"
                        if decision == "skip"
                        else "background_not_required_after_process_validation"
                        if decision == "preserve"
                        else "background_review_required"
                    ),
                    "input": baseline_stem,
                    "output": "stage3_bgremoved" if stage_saved else None,
                    "fallback_used": False,
                },
                "directional_pattern_router": {
                    "status": (
                        "review_required"
                        if noise_route.get("requires_review")
                        else "accepted"
                    ),
                    "method": noise_route.get("route"),
                    "reason_code": (
                        "pattern_noise_deferred"
                        if noise_route.get("route") == "pattern_noise_deferred"
                        else "mixed_gradient_pattern_noise_review"
                        if noise_route.get("route") == "mixed_gradient_and_pattern_noise"
                        else "no_directional_pattern_detected"
                    ),
                    "fallback_used": False,
                },
            },
            review_reasons=[reason_code],
        )
        return

    def evaluate_attempts(
        attempts: List[Tuple[str, Tuple[str, ...], str]],
        *,
        phase: str,
    ) -> bool:
        nonlocal baseline_saved, graxpert_runtime_error
        nonlocal policy_abort_candidate_search, policy_abort_reason
        phase_sufficient = False
        if not baseline_saved:
            pipeline.log.warn(
                f"[Stage3] Skip {phase}: rollback checkpoint is unavailable"
            )
            return False
        for label, command, source in attempts:
            if (
                candidate_attempt_limit > 0
                and len(attempt_records) >= candidate_attempt_limit
            ):
                pipeline.log.info(
                    "[Stage3] Candidate attempt limit reached; "
                    f"skip remaining {phase} candidates"
                )
                break
            if not restore_baseline(f"before:{label}"):
                pipeline.log.warn(
                    f"[Stage3] Stop {phase}: unable to establish clean baseline "
                    f"before {label}"
                )
                break

            sample_install_report: Dict[str, Any] = {
                "status": "not_required",
                "installed": False,
            }
            if _stage3_subsky_uses_existing(command):
                installed_points = (
                    compound_fit_points
                    if source == "builtin"
                    and compound_split_report.get("status") == "ready"
                    else safe_sample_points
                )
                sample_ok, sample_install_report = (
                    _stage3_install_safe_background_samples(
                        pipeline,
                        installed_points,
                        minimum_count=(
                            _stage3_cfg_int(
                                pipeline,
                                "stage3_compound_fit_min_count",
                                8,
                                8,
                                56,
                            )
                            if source == "builtin"
                            and compound_split_report.get("status") == "ready"
                            else None
                        ),
                        sample_contract=(
                            "compound_fit"
                            if source == "builtin"
                            and compound_split_report.get("status") == "ready"
                            else "safe_background"
                        ),
                        sample_records=(
                            masked_fit_sample_records
                            if masked_catalog_sampling_used
                            and source == "builtin"
                            and compound_split_report.get("status") == "ready"
                            else None
                        ),
                        masked_statistics=bool(
                            masked_catalog_sampling_used
                            and source == "builtin"
                            and compound_split_report.get("status") == "ready"
                        ),
                    )
                )
                if not sample_ok:
                    if masked_catalog_sampling_used:
                        safe_sample_recovery_report.update(
                            status="failed",
                            reason_code=(
                                "stage3_dense_star_bg_sample_roundtrip_failed"
                            ),
                            sample_install=sample_install_report,
                        )
                    attempt_records.append(
                        {
                            "label": label,
                            "source": source,
                            "phase": phase,
                            "command": list(command),
                            "status": "safe_sample_install_failed",
                            "failure_reason": sample_install_report.get("reason"),
                            "safe_samples": sample_install_report,
                            "fallback_triggered": True,
                        }
                    )
                    pipeline.log.warn(
                        f"{label} 禁止执行：自定义安全样点未完整安装；"
                        "不会让 subsky 自动重采样"
                    )
                    if not restore_baseline(f"sample_install_failed:{label}"):
                        break
                    if failure_action != "auto_fallback":
                        policy_abort_candidate_search = True
                        policy_abort_reason = (
                            f"{label}: safe background samples unavailable"
                        )
                        break
                    continue
            else:
                _stage3_clear_background_samples(pipeline)

            pipeline.log.info(f"尝试背景提取: {label}")
            command_ok, failure_reason = _stage3_try_background_command(
                pipeline,
                label,
                command,
                source,
            )
            if not command_ok:
                is_graxpert = _stage3_is_graxpert_attempt(label, command, source)
                is_graxpert_runtime = bool(
                    is_graxpert
                    and failure_reason
                    and failure_reason.startswith("graxpert_runtime_error:")
                )
                is_plugin_runtime = bool(
                    failure_reason
                    and failure_reason.startswith("plugin_runtime_error:")
                )
                if is_graxpert_runtime:
                    graxpert_runtime_error = True
                    graxpert_error_reasons.append(failure_reason)
                    pipeline.log.warn(
                        f"{label} 运行失败，自动切换到下一个背景提取候选: {failure_reason}"
                    )
                elif is_plugin_runtime:
                    pipeline.log.warn(
                        f"{label} 未产生有效图像变更，自动切换到下一个背景提取候选: "
                        f"{failure_reason}"
                    )
                attempt_records.append(
                    {
                        "label": label,
                        "source": source,
                        "phase": phase,
                        "command": list(command),
                        "status": (
                            "graxpert_runtime_error"
                            if is_graxpert_runtime
                            else (
                                "plugin_runtime_error"
                                if is_plugin_runtime
                                else "command_failed"
                            )
                        ),
                        "failure_reason": failure_reason,
                        "safe_samples": sample_install_report,
                        "fallback_triggered": bool(
                            is_graxpert or is_plugin_runtime
                        ),
                    }
                )
                if not restore_baseline(f"failed:{label}"):
                    break
                if failure_action != "auto_fallback":
                    policy_abort_candidate_search = True
                    policy_abort_reason = (
                        f"{label}: {failure_reason or 'candidate command failed'}"
                    )
                    break
                continue

            background_model_artifact = (
                _stage3_capture_graxpert_background_model(
                    pipeline,
                    label=label,
                )
                if source == "graxpert"
                else None
            )

            neutral_axis_projection: Optional[Dict[str, Any]] = None
            spatial_opponent_projection: Optional[Dict[str, Any]] = None
            spatial_opponent_review_required = False
            spatial_opponent_applied = False
            candidate_stem_override: Optional[str] = None
            neutral_checkpoint_pixels = None
            if safe_sample_recovery_used and label == "neutral-axis-poly1":
                try:
                    siril_proposal = pipeline.siril.get_image_pixeldata(
                        preview=False
                    )
                except (
                    CommandError,
                    SirilError,
                    OSError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ) as error:
                    siril_proposal = None
                    neutral_axis_projection = {
                        "status": "rejected",
                        "accepted": False,
                        "reason_code": (
                            "stage3_neutral_axis_proposal_unavailable"
                        ),
                        "issues": [str(error)],
                    }
                projected_pixels = None
                if siril_proposal is not None:
                    (
                        projected_pixels,
                        neutral_axis_projection,
                    ) = project_stage3_neutral_axis_poly1(
                        before_image,
                        siril_proposal,
                        compound_fit_points,
                        compound_validation_points,
                        patch_radius=_stage3_cfg_int(
                            pipeline,
                            "stage3_safe_sample_patch_radius",
                            12,
                            4,
                            24,
                        ),
                        minimum_fit=_stage3_cfg_int(
                            pipeline,
                            "stage3_compound_fit_min_count",
                            8,
                            8,
                            56,
                        ),
                    )
                if projected_pixels is None:
                    attempt_records.append(
                        {
                            "label": label,
                            "source": source,
                            "phase": phase,
                            "command": list(command),
                            "status": "neutral_axis_projection_rejected",
                            "accepted": False,
                            "severity": "hard_rejected",
                            "failure_reason": (
                                neutral_axis_projection or {}
                            ).get("reason_code"),
                            "safe_samples": sample_install_report,
                            "neutral_axis_projection": neutral_axis_projection,
                            "fallback_triggered": False,
                        }
                    )
                    if not restore_baseline(
                        f"neutral_axis_projection_rejected:{label}"
                    ):
                        break
                    continue
                write_ok, write_error = _stage3_write_neutral_axis_pixels(
                    pipeline,
                    projected_pixels,
                )
                if not write_ok:
                    neutral_axis_projection["writeback"] = {
                        "status": "failed",
                        "reason": write_error,
                    }
                    attempt_records.append(
                        {
                            "label": label,
                            "source": source,
                            "phase": phase,
                            "command": list(command),
                            "status": "neutral_axis_writeback_failed",
                            "accepted": False,
                            "severity": "hard_rejected",
                            "failure_reason": write_error,
                            "safe_samples": sample_install_report,
                            "neutral_axis_projection": neutral_axis_projection,
                            "fallback_triggered": False,
                        }
                    )
                    if not restore_baseline(
                        f"neutral_axis_writeback_failed:{label}"
                    ):
                        break
                    continue
                try:
                    projected_writeback = pipeline.siril.get_image_pixeldata(
                        preview=False
                    )
                    writeback_ok, writeback_report = (
                        verify_stage3_neutral_axis_persistence(
                            before_image,
                            projected_writeback,
                        )
                    )
                except (
                    CommandError,
                    SirilError,
                    OSError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ) as error:
                    writeback_ok = False
                    writeback_report = {
                        "status": "rejected",
                        "accepted": False,
                        "issues": [str(error)],
                    }
                neutral_axis_projection["writeback"] = writeback_report
                if not writeback_ok:
                    attempt_records.append(
                        {
                            "label": label,
                            "source": source,
                            "phase": phase,
                            "command": list(command),
                            "status": "neutral_axis_writeback_rejected",
                            "accepted": False,
                            "severity": "hard_rejected",
                            "failure_reason": (
                                "stage3_neutral_axis_invariant_failed"
                            ),
                            "safe_samples": sample_install_report,
                            "neutral_axis_projection": neutral_axis_projection,
                            "fallback_triggered": False,
                        }
                    )
                    if not restore_baseline(
                        f"neutral_axis_writeback_rejected:{label}"
                    ):
                        break
                    continue
                neutral_checkpoint_saved = pipeline._save_stage_output(
                    "stage3_candidate_neutral_axis_poly1"
                )
                if neutral_checkpoint_saved:
                    try:
                        pipeline.cmd_with_check(
                            "load",
                            "stage3_candidate_neutral_axis_poly1",
                            quiet=True,
                        )
                        neutral_checkpoint_pixels = (
                            pipeline.siril.get_image_pixeldata(preview=False)
                        )
                        (
                            neutral_checkpoint_ok,
                            neutral_checkpoint_report,
                        ) = verify_stage3_neutral_axis_persistence(
                            before_image,
                            neutral_checkpoint_pixels,
                        )
                    except (
                        CommandError,
                        SirilError,
                        OSError,
                        RuntimeError,
                        TypeError,
                        ValueError,
                    ) as error:
                        neutral_checkpoint_ok = False
                        neutral_checkpoint_report = {
                            "status": "rejected",
                            "accepted": False,
                            "issues": [str(error)],
                        }
                else:
                    neutral_checkpoint_ok = False
                    neutral_checkpoint_report = {
                        "status": "rejected",
                        "accepted": False,
                        "issues": [
                            "neutral-axis candidate checkpoint could not be saved"
                        ],
                    }
                neutral_axis_projection["candidate_checkpoint"] = (
                    neutral_checkpoint_report
                )
                if not neutral_checkpoint_ok or neutral_checkpoint_pixels is None:
                    attempt_records.append(
                        {
                            "label": label,
                            "source": source,
                            "phase": phase,
                            "command": list(command),
                            "status": "neutral_axis_checkpoint_rejected",
                            "accepted": False,
                            "severity": "hard_rejected",
                            "failure_reason": (
                                "stage3_neutral_axis_checkpoint_failed"
                            ),
                            "safe_samples": sample_install_report,
                            "neutral_axis_projection": neutral_axis_projection,
                            "fallback_triggered": False,
                        }
                    )
                    if not restore_baseline(
                        f"neutral_axis_checkpoint_rejected:{label}"
                    ):
                        break
                    continue

                opponent_pixels, spatial_opponent_projection = (
                    project_stage3_spatial_opponent_poly1(
                        before_image,
                        neutral_checkpoint_pixels,
                        siril_proposal,
                        compound_fit_points,
                        compound_validation_points,
                        patch_radius=_stage3_cfg_int(
                            pipeline,
                            "stage3_safe_sample_patch_radius",
                            12,
                            4,
                            24,
                        ),
                        minimum_fit=_stage3_cfg_int(
                            pipeline,
                            "stage3_compound_fit_min_count",
                            8,
                            8,
                            56,
                        ),
                        minimum_validation=_stage3_cfg_int(
                            pipeline,
                            "stage3_compound_validation_min_count",
                            4,
                            4,
                            20,
                        ),
                        support_mask=safe_sample_support_mask,
                    )
                )
                opponent_reason = str(
                    spatial_opponent_projection.get("reason_code") or ""
                )
                if (
                    opponent_pixels is not None
                    and opponent_reason
                    == "stage3_spatial_opponent_correction_applied"
                ):
                    opponent_write_ok, opponent_write_error = (
                        _stage3_write_neutral_axis_pixels(
                            pipeline,
                            opponent_pixels,
                        )
                    )
                    if opponent_write_ok:
                        try:
                            opponent_writeback = (
                                pipeline.siril.get_image_pixeldata(preview=False)
                            )
                            (
                                opponent_writeback_ok,
                                opponent_writeback_report,
                            ) = verify_stage3_spatial_opponent_persistence(
                                neutral_checkpoint_pixels,
                                opponent_writeback,
                                spatial_opponent_projection,
                            )
                        except (
                            CommandError,
                            SirilError,
                            OSError,
                            RuntimeError,
                            TypeError,
                            ValueError,
                        ) as error:
                            opponent_writeback_ok = False
                            opponent_writeback_report = {
                                "status": "rejected",
                                "accepted": False,
                                "issues": [str(error)],
                            }
                    else:
                        opponent_writeback_ok = False
                        opponent_writeback_report = {
                            "status": "rejected",
                            "accepted": False,
                            "issues": [opponent_write_error or "writeback failed"],
                        }
                    spatial_opponent_projection["writeback"] = (
                        opponent_writeback_report
                    )
                    if opponent_writeback_ok:
                        spatial_opponent_applied = True
                        candidate_stem_override = (
                            "stage3_candidate_spatial_opponent_poly1"
                        )
                    else:
                        spatial_opponent_projection.update(
                            status="rejected",
                            accepted=False,
                            reason_code=(
                                "stage3_spatial_opponent_candidate_rejected"
                            ),
                        )
                        spatial_opponent_review_required = True
                        try:
                            pipeline.cmd_with_check(
                                "load",
                                "stage3_candidate_neutral_axis_poly1",
                                quiet=True,
                            )
                        except (CommandError, SirilError) as error:
                            raise RuntimeError(
                                "Stage3 spatial opponent rollback to the accepted "
                                f"neutral checkpoint failed: {error}"
                            ) from error
                elif opponent_reason == "stage3_spatial_opponent_not_required":
                    pipeline.cmd_with_check(
                        "load",
                        "stage3_candidate_neutral_axis_poly1",
                        quiet=True,
                    )
                else:
                    spatial_opponent_review_required = True
                    pipeline.cmd_with_check(
                        "load",
                        "stage3_candidate_neutral_axis_poly1",
                        quiet=True,
                    )

            after_feat = pipeline._stage3_measure_features(label)
            after_image = None
            try:
                after_image = pipeline.siril.get_image_pixeldata(preview=False)
            except (CommandError, SirilError, OSError, RuntimeError, TypeError, ValueError) as e:
                pipeline.log.debug(f"stage3 candidate image sampling skipped ({label}): {e}")
            preservation = pipeline._stage3_signal_preservation_metrics(
                before_image,
                after_image,
            )
            pixel_gate_ok, pixel_gate = _stage3_candidate_pixel_gate(
                before_image,
                after_image,
                gate_profile=gate_profile,
            )
            fidelity_ok, fidelity_gate = assess_target_fidelity(
                preservation,
                low_complexity_required=bool(
                    process_report.get("low_complexity_required", False)
                ),
                gate_profile=gate_profile,
            )
            fidelity_enforced = isinstance(input_profile, dict)
            gate_ok = True
            gate_msg = _stage3_quality_diagnostic_message(
                before_feat,
                after_feat,
                preservation,
            )
            if fidelity_enforced and not fidelity_ok:
                gate_ok = False
                issues = ", ".join(fidelity_gate.get("issues") or [])
                gate_msg = (
                    f"{gate_msg}; target fidelity gate rejected"
                    + (f": {issues}" if issues else "")
                )
            if fidelity_enforced and not pixel_gate_ok:
                gate_ok = False
                issues = ", ".join(pixel_gate.get("hard_issues") or [])
                gate_msg = (
                    f"{gate_msg}; candidate pixel integrity gate rejected"
                    + (f": {issues}" if issues else "")
                )
            if pattern_routing_enabled and after_image is not None:
                after_pattern_report = analyze_directional_pattern_noise(
                    after_image,
                    detection_threshold=_stage3_cfg_float(
                        pipeline,
                        "stage3_pattern_score_min",
                        0.55,
                        0.25,
                        0.90,
                    ),
                    walking_threshold=_stage3_cfg_float(
                        pipeline,
                        "stage3_walking_noise_score_min",
                        0.50,
                        0.25,
                        0.90,
                    ),
                )
            else:
                after_pattern_report = {
                    "status": "unavailable",
                    "detected": False,
                }
            pattern_ok, pattern_gate_report = pattern_candidate_gate(
                pattern_report,
                after_pattern_report,
                growth_max=_stage3_cfg_float(
                    pipeline,
                    "stage3_pattern_score_growth_max",
                    0.12,
                    0.02,
                    0.40,
                ),
                gate_profile=gate_profile,
            )
            if not pattern_ok:
                gate_ok = False
                gate_msg = (
                    f"{gate_msg}; directional pattern gate rejected "
                    f"growth={float(pattern_gate_report.get('pattern_score_growth') or 0.0):.3f}"
                )
            after_adaptive_candidate = (
                pipeline._adaptive_features_current()
                if hasattr(pipeline, "_adaptive_features_current")
                else {}
            )
            if compound_validation_points and after_image is not None:
                candidate_validation = measure_background_validation(
                    after_image,
                    compound_validation_points,
                    patch_radius=_stage3_cfg_int(
                        pipeline,
                        "stage3_safe_sample_patch_radius",
                        12,
                        4,
                        24,
                    ),
                    minimum_count=_stage3_cfg_int(
                        pipeline,
                        "stage3_compound_validation_min_count",
                        4,
                        4,
                        20,
                    ),
                    value_scale=baseline_validation.get("value_scale"),
                    support_mask=safe_sample_support_mask,
                )
            else:
                candidate_validation = {
                    "status": "not_run",
                    "reason": "held-out validation pool is unavailable",
                }
            validation_ok, validation_gate = assess_single_background_validation(
                baseline_validation,
                candidate_validation,
                gate_profile=gate_profile,
            )
            validation_enforced = isinstance(input_profile, dict)
            if validation_enforced and not validation_ok:
                gate_ok = False
                issues = ", ".join(validation_gate.get("issues") or [])
                gate_msg = (
                    f"{gate_msg}; held-out background/RMS gate rejected"
                    + (f": {issues}" if issues else "")
                )
            directional_gradient_ok = True
            directional_gradient_gate: Dict[str, Any] = {
                "status": "not_applicable",
                "accepted": True,
            }
            if neutral_axis_projection is not None:
                if after_image is None:
                    directional_gradient_ok = False
                    directional_gradient_gate = {
                        "status": "rejected",
                        "accepted": False,
                        "issues": [
                            "projected pixels unavailable for held-out direction gate"
                        ],
                    }
                else:
                    (
                        directional_gradient_ok,
                        directional_gradient_gate,
                    ) = assess_background_direction_reversal(
                        before_image,
                        after_image,
                        compound_validation_points,
                        patch_radius=_stage3_cfg_int(
                            pipeline,
                            "stage3_safe_sample_patch_radius",
                            12,
                            4,
                            24,
                        ),
                        support_mask=safe_sample_support_mask,
                    )
                if not directional_gradient_ok:
                    gate_ok = False
                    issues = ", ".join(
                        directional_gradient_gate.get("issues") or []
                    )
                    gate_msg = (
                        f"{gate_msg}; held-out gradient direction gate rejected"
                        + (f": {issues}" if issues else "")
                    )
            color_shift = _stage3_color_shift(before_adaptive, after_adaptive_candidate)
            color_shift_limit = float(
                stage3_gate_thresholds("strict")["sufficient_color_shift_max"]
            )
            color_shift_ok = bool(
                neutral_axis_projection is None
                or color_shift <= color_shift_limit
            )
            color_shift_gate = (
                {
                    "status": "accepted" if color_shift_ok else "rejected",
                    "accepted": color_shift_ok,
                    "measured": float(color_shift),
                    "maximum": color_shift_limit,
                    "profile": "strict",
                }
                if neutral_axis_projection is not None
                else {
                    "status": "not_applicable",
                    "accepted": True,
                }
            )
            if not color_shift_ok:
                gate_ok = False
                gate_msg = (
                    f"{gate_msg}; strict color-change gate rejected "
                    f"{color_shift:.3f}>{color_shift_limit:.3f}"
                )
            required_adaptive_metrics = {
                "bg_std",
                "gradient_score",
                "dirty_background_score",
                "chroma_noise_score",
            }
            adaptive_metrics_available = bool(
                required_adaptive_metrics.issubset(before_adaptive)
                and required_adaptive_metrics.issubset(after_adaptive_candidate)
            )
            gate_warnings = list(
                dict.fromkeys(
                    [
                        *(
                            fidelity_gate.get("warnings") or []
                            if fidelity_enforced
                            else []
                        ),
                        *(
                            validation_gate.get("warnings") or []
                            if validation_enforced
                            else []
                        ),
                        *(
                            pattern_gate_report.get("warnings") or []
                            if pattern_routing_enabled
                            else []
                        ),
                    ]
                )
            )
            hard_gate_metrics_available = bool(
                before_feat is not None
                and after_feat is not None
                and after_image is not None
                and pixel_gate_ok
                and preservation.get("available")
                and adaptive_metrics_available
                and directional_gradient_ok
                and color_shift_ok
                and (
                    not validation_enforced
                    or candidate_validation.get("status") == "ready"
                )
                and (
                    not pattern_routing_enabled
                    or (
                        pattern_report.get("status") == "ok"
                        and after_pattern_report.get("status") == "ok"
                    )
                )
            )
            record = {
                "label": label,
                "source": source,
                "phase": phase,
                "command": list(command),
                **_stage3_candidate_quality_record_fields(
                    gate_ok=gate_ok,
                    gate_profile=gate_profile,
                    gate_warnings=gate_warnings,
                    hard_gate_metrics_available=hard_gate_metrics_available,
                    color_shift=color_shift,
                ),
                "quality_message": gate_msg,
                "preservation": preservation,
                "pixel_integrity_gate": pixel_gate,
                "target_fidelity_gate": fidelity_gate,
                "target_fidelity_enforced": fidelity_enforced,
                "safe_samples": sample_install_report,
                "directional_pattern_noise": after_pattern_report,
                "pattern_quality_gate": pattern_gate_report,
                "after_adaptive": after_adaptive_candidate,
                "validation": candidate_validation,
                "validation_gate": validation_gate,
                "validation_enforced": validation_enforced,
                "directional_gradient_gate": directional_gradient_gate,
                "color_shift_gate": color_shift_gate,
                "neutral_axis_projection": neutral_axis_projection,
                "spatial_opponent_projection": spatial_opponent_projection,
                "spatial_opponent_review_required": (
                    spatial_opponent_review_required
                ),
            }
            below_uncertainty_only = bool(
                validation_enforced
                and _stage3_below_uncertainty_only_validation(
                    validation_gate
                ).get("eligible", False)
                and pixel_gate_ok
                and fidelity_ok
                and pattern_ok
                and directional_gradient_ok
                and color_shift_ok
                and not spatial_opponent_review_required
            )
            if not gate_ok and below_uncertainty_only:
                candidate_stem = (
                    candidate_stem_override or _stage3_candidate_stem(label)
                )
                candidate_saved = pipeline._save_stage_output(candidate_stem)
                checkpoint_report: Dict[str, Any] = {
                    "schema": "starun.stage3-candidate-checkpoint.v1",
                    "status": "rejected",
                    "accepted": False,
                    "candidate_stem": candidate_stem,
                    "issues": [],
                }
                if candidate_saved:
                    try:
                        pipeline.cmd_with_check(
                            "load",
                            candidate_stem,
                            quiet=True,
                        )
                        checkpoint_pixels = np.asarray(
                            pipeline.siril.get_image_pixeldata(preview=False)
                        )
                        measured_pixels = np.asarray(after_image)
                        pixel_exact = bool(
                            measured_pixels.shape == checkpoint_pixels.shape
                            and measured_pixels.dtype == checkpoint_pixels.dtype
                            and np.array_equal(
                                measured_pixels,
                                checkpoint_pixels,
                                equal_nan=True,
                            )
                        )
                        persistence_ok = True
                        persistence_report: Dict[str, Any] = {
                            "status": "not_applicable",
                            "accepted": True,
                        }
                        if neutral_axis_projection is not None:
                            if spatial_opponent_applied:
                                persistence_ok, persistence_report = (
                                    verify_stage3_spatial_opponent_persistence(
                                        neutral_checkpoint_pixels,
                                        checkpoint_pixels,
                                        spatial_opponent_projection or {},
                                    )
                                )
                            else:
                                persistence_ok, persistence_report = (
                                    verify_stage3_neutral_axis_persistence(
                                        before_image,
                                        checkpoint_pixels,
                                    )
                                )
                        checkpoint_ok = bool(pixel_exact and persistence_ok)
                        checkpoint_report.update(
                            status=(
                                "accepted" if checkpoint_ok else "rejected"
                            ),
                            accepted=checkpoint_ok,
                            pixel_exact=pixel_exact,
                            measured_pixel_sha256=(
                                _stage3_array_sha256(measured_pixels)
                            ),
                            checkpoint_pixel_sha256=(
                                _stage3_array_sha256(checkpoint_pixels)
                            ),
                            persistence=persistence_report,
                            issues=(
                                []
                                if checkpoint_ok
                                else [
                                    "candidate checkpoint pixels or invariants "
                                    "did not round-trip exactly"
                                ]
                            ),
                        )
                    except (
                        CommandError,
                        SirilError,
                        OSError,
                        RuntimeError,
                        TypeError,
                        ValueError,
                    ) as error:
                        checkpoint_report["issues"] = [str(error)]
                else:
                    checkpoint_report["issues"] = [
                        "candidate checkpoint could not be saved"
                    ]
                record["candidate_checkpoint"] = checkpoint_report
                record["candidate_stem"] = (
                    candidate_stem
                    if checkpoint_report.get("accepted") is True
                    else None
                )
                if checkpoint_report.get("accepted") is not True:
                    record.update(
                        status="candidate_checkpoint_rejected",
                        failure_reason="stage3_candidate_checkpoint_failed",
                    )
            if not gate_ok:
                attempt_records.append(record)
                pipeline.log.warn(
                    f"{label} rejected by quality gate, try next candidate: {gate_msg}"
                )
                if not restore_baseline(f"rejected:{label}"):
                    break
                if failure_action != "auto_fallback":
                    policy_abort_candidate_search = True
                    policy_abort_reason = f"{label}: {gate_msg}"
                    break
                continue

            candidate_stem = (
                candidate_stem_override or _stage3_candidate_stem(label)
            )
            candidate_saved = pipeline._save_stage_output(candidate_stem)
            if neutral_axis_projection is not None:
                checkpoint_ok = False
                checkpoint_report: Dict[str, Any]
                if not candidate_saved:
                    checkpoint_report = {
                        "status": "rejected",
                        "accepted": False,
                        "issues": [
                            "neutral-axis candidate checkpoint could not be saved"
                        ],
                    }
                else:
                    try:
                        pipeline.cmd_with_check(
                            "load",
                            candidate_stem,
                            quiet=True,
                        )
                        checkpoint_pixels = (
                            pipeline.siril.get_image_pixeldata(preview=False)
                        )
                        if spatial_opponent_applied:
                            checkpoint_ok, checkpoint_report = (
                                verify_stage3_spatial_opponent_persistence(
                                    neutral_checkpoint_pixels,
                                    checkpoint_pixels,
                                    spatial_opponent_projection or {},
                                )
                            )
                        else:
                            checkpoint_ok, checkpoint_report = (
                                verify_stage3_neutral_axis_persistence(
                                    before_image,
                                    checkpoint_pixels,
                                )
                            )
                    except (
                        CommandError,
                        SirilError,
                        OSError,
                        RuntimeError,
                        TypeError,
                        ValueError,
                    ) as error:
                        checkpoint_report = {
                            "status": "rejected",
                            "accepted": False,
                            "issues": [str(error)],
                        }
                neutral_axis_projection["candidate_checkpoint"] = (
                    checkpoint_report
                )
                if not checkpoint_ok:
                    record.update(
                        status="neutral_axis_checkpoint_rejected",
                        accepted=False,
                        severity="hard_rejected",
                        failure_reason=(
                            "stage3_neutral_axis_checkpoint_failed"
                        ),
                        candidate_stem=None,
                    )
                    attempt_records.append(record)
                    process_dir = getattr(pipeline, "process_dir", None)
                    if process_dir is not None:
                        try:
                            (
                                Path(process_dir)
                                / f"{candidate_stem}.fit"
                            ).unlink(missing_ok=True)
                        except OSError as error:
                            pipeline.log.debug(
                                "stage3 rejected neutral-axis candidate cleanup "
                                f"skipped: {error}"
                            )
                    if not restore_baseline(
                        f"neutral_axis_checkpoint_rejected:{label}"
                    ):
                        break
                    continue
            background_score_components = _stage3_background_score_components(
                before_adaptive,
                after_adaptive_candidate,
            )
            base_candidate_score = float(background_score_components["total"])
            nebula_preservation_weight = _stage3_nebula_preservation_weight(
                diffuse_context,
                stage3_policy,
            )
            preservation_penalty = _stage3_preservation_penalty(
                preservation,
                diffuse_context=diffuse_context,
                stage3_policy=stage3_policy,
                nebula_weight=nebula_preservation_weight,
                gate_profile=gate_profile,
            )
            pattern_penalty = max(
                0.0,
                float(pattern_gate_report.get("pattern_score_growth", 0.0) or 0.0),
            ) * STAGE3_DIRECTIONAL_PATTERN_PENALTY_WEIGHT
            candidate_score = (
                base_candidate_score
                + preservation_penalty
                + pattern_penalty
            )
            sufficient = _stage3_candidate_sufficient(
                before_adaptive,
                after_adaptive_candidate,
                candidate_score,
                stage3_policy,
                gate_profile,
            )
            if not sufficient:
                gate_warnings = list(
                    dict.fromkeys(
                        gate_warnings
                        + ["candidate does not meet clean-output sufficiency thresholds"]
                    )
                )
            quality_record_fields = _stage3_candidate_quality_record_fields(
                gate_ok=True,
                gate_profile=gate_profile,
                gate_warnings=gate_warnings,
                hard_gate_metrics_available=hard_gate_metrics_available,
                color_shift=color_shift,
            )
            record.update(
                {
                    **quality_record_fields,
                    "candidate_stem": candidate_stem if candidate_saved else None,
                    "background_model_artifact": background_model_artifact,
                    "base_background_score": base_candidate_score,
                    "background_score_components": background_score_components,
                    "preservation_penalty": preservation_penalty,
                    "directional_pattern_penalty": pattern_penalty,
                    "nebula_preservation_penalty_weight": nebula_preservation_weight,
                    "background_score": candidate_score,
                    "sufficient": sufficient,
                }
            )
            if safe_sample_recovery_used and (not sufficient or gate_warnings):
                record.update(
                    status="rejected",
                    accepted=False,
                    severity="hard_rejected",
                    hard_issues=list(gate_warnings),
                    issues=list(gate_warnings),
                    recovery_strict_rejection=True,
                )
                attempt_records.append(record)
                pipeline.log.warn(
                    f"{label} rejected by conservative recovery strict gate: "
                    + " | ".join(gate_warnings)
                )
                if not restore_baseline(f"recovery_strict_rejected:{label}"):
                    break
                if failure_action != "auto_fallback":
                    policy_abort_candidate_search = True
                    policy_abort_reason = (
                        f"{label}: conservative recovery strict gate rejected"
                    )
                    break
                continue
            attempt_records.append(record)
            if candidate_saved:
                accepted_candidates.append(
                    {
                        "label": label,
                        "source": source,
                        "phase": phase,
                        "command": list(command),
                        "stem": candidate_stem,
                        "background_model_artifact": background_model_artifact,
                        "base_score": base_candidate_score,
                        "background_score_components": (
                            background_score_components
                        ),
                        "preservation_penalty": preservation_penalty,
                        "directional_pattern_penalty": pattern_penalty,
                        "nebula_preservation_penalty_weight": nebula_preservation_weight,
                        "score": candidate_score,
                        "quality_message": gate_msg,
                        "preservation": preservation,
                        "pixel_integrity_gate": pixel_gate,
                        "target_fidelity_gate": fidelity_gate,
                        "target_fidelity_enforced": fidelity_enforced,
                        "directional_pattern_noise": after_pattern_report,
                        "pattern_quality_gate": pattern_gate_report,
                        "after_adaptive": after_adaptive_candidate,
                        "validation": candidate_validation,
                        "validation_gate": validation_gate,
                        "validation_enforced": validation_enforced,
                        "directional_gradient_gate": (
                            directional_gradient_gate
                        ),
                        "color_shift_gate": color_shift_gate,
                        "neutral_axis_projection": neutral_axis_projection,
                        "spatial_opponent_projection": (
                            spatial_opponent_projection
                        ),
                        "spatial_opponent_review_required": (
                            spatial_opponent_review_required
                        ),
                        "sufficient": sufficient,
                        **quality_record_fields,
                    }
                )
            if not restore_baseline(f"evaluated:{label}"):
                break
            clean_sufficient = bool(sufficient and not gate_warnings)
            if sufficient and candidate_saved:
                pipeline.log.info(
                    f"背景提取候选{'足够干净' if clean_sufficient else '带软告警可用'}: "
                    f"{label} score={candidate_score:.3f}"
                )
                phase_sufficient = phase_sufficient or clean_sufficient
                pipeline.log.info(
                    f"{label} 已合格；继续评估剩余候选以保护弥散星云"
                )
                continue
            pipeline.log.info(
                f"背景提取候选通过但残余背景偏高，继续搜索: {label} score={candidate_score:.3f}"
            )
        return phase_sufficient

    def evaluate_compound_candidate() -> bool:
        nonlocal compound_report
        label = "subsky-poly-residual-rbf"
        phase = "compound_fallback"
        accepted_builtin_candidates = [
            candidate
            for candidate in accepted_candidates
            if candidate.get("source") == "builtin"
        ]
        unverified_builtin_candidates = [
            candidate.get("label")
            for candidate in accepted_builtin_candidates
            if not bool(candidate.get("hard_gate_metrics_available", False))
        ]
        builtin_candidates = [
            candidate
            for candidate in accepted_builtin_candidates
            if bool(candidate.get("hard_gate_metrics_available", False))
        ]
        rbf_candidates = [
            candidate
            for candidate in builtin_candidates
            if "-rbf" in tuple(candidate.get("command") or ())
        ]
        hard_rejections = [
            record.get("label")
            for record in attempt_records
            if record.get("source") == "builtin"
            and record.get("status") == "rejected"
        ]
        eligibility_issues: List[str] = []
        if not compound_target_guard.get("eligible"):
            eligibility_issues.extend(
                str(reason)
                for reason in compound_target_guard.get("reasons", [])
            )
        if compound_split_report.get("status") != "ready":
            eligibility_issues.append("deterministic_fit_validation_split_unavailable")
        if baseline_validation.get("status") != "ready":
            eligibility_issues.append("baseline_validation_unavailable")
        if hard_rejections:
            eligibility_issues.append("single_stage_hard_gate_rejection_present")
        if unverified_builtin_candidates:
            eligibility_issues.append("single_stage_hard_gate_metrics_unavailable")
        if not builtin_candidates:
            eligibility_issues.append("no_hard_gate_accepted_builtin_candidate")
        if not rbf_candidates:
            eligibility_issues.append("no_hard_gate_accepted_rbf_candidate")

        best_single = (
            min(
                builtin_candidates,
                key=lambda item: float(item.get("score", 999.0)),
            )
            if builtin_candidates
            else None
        )
        best_rbf = (
            min(
                rbf_candidates,
                key=lambda item: float(item.get("score", 999.0)),
            )
            if rbf_candidates
            else None
        )
        if best_single is not None:
            residual_gate = _stage3_compound_residual_gate(
                best_single.get("validation") or {},
            )
            if not residual_gate.get("supported"):
                eligibility_issues.append("low_frequency_residual_not_supported")
            if (best_single.get("validation") or {}).get("status") != "ready":
                eligibility_issues.append("best_single_validation_unavailable")
        else:
            residual_gate = {"status": "not_run", "supported": False}
        if best_rbf is not None and not bool(
            best_rbf.get("hard_gate_metrics_available", False)
        ):
            eligibility_issues.append("best_rbf_hard_gate_metrics_unavailable")

        eligibility_issues = list(dict.fromkeys(eligibility_issues))
        if eligibility_issues:
            compound_report = {
                "status": "not_triggered",
                "triggered": False,
                "eligibility_issues": eligibility_issues,
                "target_guard": compound_target_guard,
                "sample_split": compound_split_report,
                "residual_gate": residual_gate,
                "hard_rejected_single_candidates": hard_rejections,
                "unverified_single_candidates": unverified_builtin_candidates,
                "best_single_candidate": (
                    best_single.get("label") if best_single else None
                ),
                "reused_rbf_candidate": (
                    best_rbf.get("label") if best_rbf else None
                ),
            }
            return False

        assert best_single is not None
        assert best_rbf is not None
        compound_report = {
            "status": "running",
            "triggered": True,
            "target_guard": compound_target_guard,
            "sample_split": compound_split_report,
            "residual_gate": residual_gate,
            "best_single_candidate": best_single.get("label"),
            "reused_rbf_candidate": best_rbf.get("label"),
            "reused_rbf_command": list(best_rbf.get("command") or []),
        }
        pipeline.log.info(
            "[Stage3] Single-stage built-ins remain insufficient; "
            "try frozen-validation Polynomial→residual-RBF before external plugins"
        )
        base_record: Dict[str, Any] = {
            "label": label,
            "source": "compound",
            "phase": phase,
            "status": "running",
            "fallback_triggered": True,
            "best_single_candidate": best_single.get("label"),
            "reused_rbf_candidate": best_rbf.get("label"),
            "steps": [
                ["subsky", "1", "-existing"],
                list(best_rbf.get("command") or []),
            ],
            "sample_split": compound_split_report,
            "baseline_validation": baseline_validation,
        }

        def reject(
            status: str,
            reason: str,
            **details: Any,
        ) -> bool:
            record = {
                **base_record,
                **details,
                "status": status,
                "failure_reason": reason,
            }
            attempt_records.append(record)
            compound_report.update(
                {
                    "status": status,
                    "accepted": False,
                    "failure_reason": reason,
                    **details,
                }
            )
            pipeline.log.warn(
                f"[Stage3] Compound Polynomial→RBF candidate rejected: {reason}"
            )
            return False

        if not restore_baseline(f"before:{label}"):
            return reject(
                "rollback_unavailable",
                "immutable baseline could not be restored",
            )

        rollback_attempted = False
        rollback_completed = False
        try:
            poly_ok, poly_samples = _stage3_install_safe_background_samples(
                pipeline,
                compound_fit_points,
                minimum_count=_stage3_cfg_int(
                    pipeline,
                    "stage3_compound_fit_min_count",
                    8,
                    8,
                    56,
                ),
                sample_contract="compound_fit_polynomial",
            )
            if not poly_ok:
                return reject(
                    "safe_sample_install_failed",
                    str(poly_samples.get("reason") or "Polynomial fit samples unavailable"),
                    polynomial_samples=poly_samples,
                )
            polynomial_command = ("subsky", "1", "-existing")
            command_ok, failure_reason = _stage3_try_background_command(
                pipeline,
                f"{label}-polynomial",
                polynomial_command,
                "compound",
            )
            if not command_ok:
                return reject(
                    "command_failed",
                    str(failure_reason or "Polynomial command failed"),
                    polynomial_samples=poly_samples,
                )
            intermediate_stem = "stage3_compound_poly_intermediate"
            if not pipeline._save_stage_output(intermediate_stem):
                return reject(
                    "intermediate_save_failed",
                    "Polynomial transaction intermediate could not be saved",
                    polynomial_samples=poly_samples,
                    intermediate_stem=intermediate_stem,
                )
            try:
                polynomial_image = pipeline.siril.get_image_pixeldata(preview=False)
            except (
                CommandError,
                SirilError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as error:
                return reject(
                    "validation_unavailable",
                    f"Polynomial intermediate pixels unavailable: {error}",
                    polynomial_samples=poly_samples,
                    intermediate_stem=intermediate_stem,
                )
            polynomial_validation = measure_background_validation(
                polynomial_image,
                compound_validation_points,
                patch_radius=_stage3_cfg_int(
                    pipeline,
                    "stage3_safe_sample_patch_radius",
                    12,
                    4,
                    24,
                ),
                minimum_count=_stage3_cfg_int(
                    pipeline,
                    "stage3_compound_validation_min_count",
                    4,
                    4,
                    20,
                ),
                value_scale=baseline_validation.get("value_scale"),
            )
            if polynomial_validation.get("status") != "ready":
                return reject(
                    "validation_unavailable",
                    "Polynomial held-out validation is unavailable",
                    polynomial_samples=poly_samples,
                    polynomial_validation=polynomial_validation,
                    intermediate_stem=intermediate_stem,
                )

            rbf_ok, rbf_samples = _stage3_install_safe_background_samples(
                pipeline,
                compound_fit_points,
                minimum_count=_stage3_cfg_int(
                    pipeline,
                    "stage3_compound_fit_min_count",
                    8,
                    8,
                    56,
                ),
                sample_contract="compound_fit_rbf",
            )
            if not rbf_ok:
                return reject(
                    "safe_sample_install_failed",
                    str(rbf_samples.get("reason") or "RBF fit samples unavailable"),
                    polynomial_samples=poly_samples,
                    rbf_samples=rbf_samples,
                    polynomial_validation=polynomial_validation,
                    intermediate_stem=intermediate_stem,
                )
            rbf_command = tuple(best_rbf.get("command") or ())
            command_ok, failure_reason = _stage3_try_background_command(
                pipeline,
                f"{label}-rbf",
                rbf_command,
                "compound",
            )
            if not command_ok:
                return reject(
                    "command_failed",
                    str(failure_reason or "residual RBF command failed"),
                    polynomial_samples=poly_samples,
                    rbf_samples=rbf_samples,
                    polynomial_validation=polynomial_validation,
                    intermediate_stem=intermediate_stem,
                )

            after_feat = pipeline._stage3_measure_features(label)
            try:
                after_image = pipeline.siril.get_image_pixeldata(preview=False)
            except (
                CommandError,
                SirilError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as error:
                after_image = None
                pipeline.log.debug(
                    f"stage3 compound candidate image sampling skipped: {error}"
                )
            preservation = pipeline._stage3_signal_preservation_metrics(
                before_image,
                after_image,
            )
            pixel_gate_ok, pixel_gate = _stage3_candidate_pixel_gate(
                before_image,
                after_image,
                gate_profile=gate_profile,
            )
            fidelity_ok, fidelity_gate = assess_target_fidelity(
                preservation,
                low_complexity_required=bool(
                    process_report.get("low_complexity_required", False)
                ),
                gate_profile=gate_profile,
            )
            fidelity_enforced = isinstance(input_profile, dict)
            gate_ok = True
            gate_message = _stage3_quality_diagnostic_message(
                before_feat,
                after_feat,
                preservation,
            )
            if fidelity_enforced and not fidelity_ok:
                gate_ok = False
                issues = ", ".join(fidelity_gate.get("issues") or [])
                gate_message = (
                    f"{gate_message}; target fidelity gate rejected"
                    + (f": {issues}" if issues else "")
                )
            if fidelity_enforced and not pixel_gate_ok:
                gate_ok = False
                issues = ", ".join(pixel_gate.get("hard_issues") or [])
                gate_message = (
                    f"{gate_message}; candidate pixel integrity gate rejected"
                    + (f": {issues}" if issues else "")
                )
            if pattern_routing_enabled and after_image is not None:
                after_pattern_report = analyze_directional_pattern_noise(
                    after_image,
                    detection_threshold=_stage3_cfg_float(
                        pipeline,
                        "stage3_pattern_score_min",
                        0.55,
                        0.25,
                        0.90,
                    ),
                    walking_threshold=_stage3_cfg_float(
                        pipeline,
                        "stage3_walking_noise_score_min",
                        0.50,
                        0.25,
                        0.90,
                    ),
                )
            else:
                after_pattern_report = {
                    "status": "unavailable",
                    "detected": False,
                }
            pattern_ok, pattern_gate_report = pattern_candidate_gate(
                pattern_report,
                after_pattern_report,
                growth_max=_stage3_cfg_float(
                    pipeline,
                    "stage3_pattern_score_growth_max",
                    0.12,
                    0.02,
                    0.40,
                ),
                gate_profile=gate_profile,
            )
            if not pattern_ok:
                gate_ok = False
                gate_message = (
                    f"{gate_message}; directional pattern gate rejected"
                )
            after_adaptive_candidate = (
                pipeline._adaptive_features_current()
                if hasattr(pipeline, "_adaptive_features_current")
                else {}
            )
            color_shift = _stage3_color_shift(
                before_adaptive,
                after_adaptive_candidate,
            )
            required_adaptive_metrics = {
                "bg_std",
                "gradient_score",
                "dirty_background_score",
                "chroma_noise_score",
            }
            adaptive_metrics_available = bool(
                required_adaptive_metrics.issubset(before_adaptive)
                and required_adaptive_metrics.issubset(after_adaptive_candidate)
            )
            color_shift_limit = float(
                stage3_gate_thresholds(gate_profile)["sufficient_color_shift_max"]
            )
            if color_shift > color_shift_limit:
                if gate_profile == "strict":
                    gate_ok = False
                gate_message = (
                    f"{gate_message}; color shift "
                    f"{color_shift:.3f}>{color_shift_limit:.3f}"
                )
            hard_gate_metrics_available = bool(
                before_feat is not None
                and after_feat is not None
                and after_image is not None
                and pixel_gate_ok
                and preservation.get("available")
                and adaptive_metrics_available
                and (
                    not pattern_routing_enabled
                    or (
                        pattern_report.get("status") == "ok"
                        and after_pattern_report.get("status") == "ok"
                    )
                )
            )
            missing_metric_warning = "compound hard-gate metrics unavailable"
            if not hard_gate_metrics_available and gate_profile == "strict":
                gate_ok = False
                gate_message = (
                    f"{gate_message}; {missing_metric_warning}"
                )
            candidate_validation = (
                measure_background_validation(
                    after_image,
                    compound_validation_points,
                    patch_radius=_stage3_cfg_int(
                        pipeline,
                        "stage3_safe_sample_patch_radius",
                        12,
                        4,
                        24,
                    ),
                    minimum_count=_stage3_cfg_int(
                        pipeline,
                        "stage3_compound_validation_min_count",
                        4,
                        4,
                        20,
                    ),
                    value_scale=baseline_validation.get("value_scale"),
                )
                if after_image is not None
                else {"status": "unavailable", "reason": "pixels unavailable"}
            )
            common_details = {
                "polynomial_samples": poly_samples,
                "rbf_samples": rbf_samples,
                "intermediate_stem": intermediate_stem,
                "polynomial_validation": polynomial_validation,
                "validation": candidate_validation,
                "quality_message": gate_message,
                "preservation": preservation,
                "pixel_integrity_gate": pixel_gate,
                "target_fidelity_gate": fidelity_gate,
                "target_fidelity_enforced": fidelity_enforced,
                "directional_pattern_noise": after_pattern_report,
                "pattern_quality_gate": pattern_gate_report,
                "after_adaptive": after_adaptive_candidate,
                "color_shift": color_shift,
                "hard_gate_metrics_available": hard_gate_metrics_available,
            }
            if not gate_ok:
                return reject(
                    "rejected",
                    gate_message,
                    **common_details,
                )

            background_score_components = _stage3_background_score_components(
                before_adaptive,
                after_adaptive_candidate,
            )
            base_candidate_score = float(background_score_components["total"])
            nebula_preservation_weight = _stage3_nebula_preservation_weight(
                diffuse_context,
                stage3_policy,
            )
            preservation_penalty = _stage3_preservation_penalty(
                preservation,
                diffuse_context=diffuse_context,
                stage3_policy=stage3_policy,
                nebula_weight=nebula_preservation_weight,
                gate_profile=gate_profile,
            )
            pattern_penalty = max(
                0.0,
                float(pattern_gate_report.get("pattern_score_growth", 0.0) or 0.0),
            ) * STAGE3_DIRECTIONAL_PATTERN_PENALTY_WEIGHT
            candidate_score = (
                base_candidate_score
                + preservation_penalty
                + pattern_penalty
            )
            validation_ok, validation_gate = (
                assess_compound_background_validation(
                    baseline_validation,
                    best_single.get("validation") or {},
                    polynomial_validation,
                    candidate_validation,
                    improvement_min=_stage3_cfg_float(
                        pipeline,
                        "stage3_compound_validation_improvement_min",
                        0.10,
                        0.10,
                        0.40,
                    ),
                    zero_point_abs_max=_stage3_cfg_float(
                        pipeline,
                        "stage3_compound_zero_point_abs_max",
                        0.01,
                        0.002,
                        0.010,
                    ),
                    zero_point_rel_max=_stage3_cfg_float(
                        pipeline,
                        "stage3_compound_zero_point_rel_max",
                        0.15,
                        0.05,
                        0.15,
                    ),
                    gate_profile=gate_profile,
                )
            )
            score_ok, score_gate = _stage3_compound_score_gate(
                float(best_single.get("score", 999.0)),
                candidate_score,
                absolute_improvement_min=_stage3_cfg_float(
                    pipeline,
                    "stage3_compound_score_abs_improvement_min",
                    0.03,
                    0.03,
                    0.15,
                ),
                relative_improvement_min=_stage3_cfg_float(
                    pipeline,
                    "stage3_compound_score_rel_improvement_min",
                    0.10,
                    0.10,
                    0.40,
                ),
            )
            common_details.update(
                {
                    "base_background_score": base_candidate_score,
                    "background_score_components": background_score_components,
                    "preservation_penalty": preservation_penalty,
                    "directional_pattern_penalty": pattern_penalty,
                    "nebula_preservation_penalty_weight": (
                        nebula_preservation_weight
                    ),
                    "background_score": candidate_score,
                    "validation_gate": validation_gate,
                    "score_gate": score_gate,
                }
            )
            gate_warnings = list(
                dict.fromkeys(
                    [
                        *(fidelity_gate.get("warnings") or []),
                        *(pattern_gate_report.get("warnings") or []),
                        *(validation_gate.get("warnings") or []),
                        *(
                            [missing_metric_warning]
                            if not hard_gate_metrics_available
                            and gate_profile != "strict"
                            else []
                        ),
                        *(
                            [
                                f"color shift {color_shift:.3f}>{color_shift_limit:.3f}"
                            ]
                            if color_shift > color_shift_limit
                            else []
                        ),
                    ]
                )
            )
            if not validation_ok and gate_profile == "strict":
                return reject(
                    "validation_rejected",
                    "; ".join(validation_gate.get("issues") or [
                        "held-out validation rejected the compound candidate"
                    ]),
                    **common_details,
                )
            if not validation_ok:
                gate_warnings.extend(validation_gate.get("issues") or [
                    "compound held-out validation did not show sufficient improvement"
                ])
            if not score_ok and gate_profile == "strict":
                return reject(
                    "score_rejected",
                    "; ".join(score_gate.get("issues") or [
                        "compound score improvement is insufficient"
                    ]),
                    **common_details,
                )
            if not score_ok:
                gate_warnings.extend(score_gate.get("issues") or [
                    "compound score improvement is insufficient"
                ])

            candidate_stem = _stage3_candidate_stem(label)
            if not pipeline._save_stage_output(candidate_stem):
                return reject(
                    "candidate_save_failed",
                    "validated compound candidate could not be saved",
                    **common_details,
                )
            _stage3_clear_background_samples(pipeline)
            rollback_attempted = True
            rollback_completed = restore_baseline(f"evaluated:{label}")
            if not rollback_completed:
                return reject(
                    "rollback_failed",
                    "validated compound candidate was invalidated because the immutable baseline could not be restored",
                    candidate_stem=candidate_stem,
                    **common_details,
                )
            sufficient = _stage3_candidate_sufficient(
                before_adaptive,
                after_adaptive_candidate,
                candidate_score,
                stage3_policy,
                gate_profile,
            )
            if not sufficient:
                gate_warnings.append(
                    "candidate does not meet clean-output sufficiency thresholds"
                )
            gate_warnings = list(dict.fromkeys(gate_warnings))
            clean_sufficient = bool(sufficient and not gate_warnings)
            quality_record_fields = _stage3_candidate_quality_record_fields(
                gate_ok=True,
                gate_profile=gate_profile,
                gate_warnings=gate_warnings,
                hard_gate_metrics_available=hard_gate_metrics_available,
                color_shift=color_shift,
            )
            record = {
                **base_record,
                **common_details,
                **quality_record_fields,
                "candidate_stem": candidate_stem,
                "sufficient": sufficient,
            }
            attempt_records.append(record)
            accepted_candidates.append(
                {
                    "label": label,
                    "source": "compound",
                    "phase": phase,
                    "command": list(rbf_command),
                    "stem": candidate_stem,
                    "base_score": base_candidate_score,
                    "background_score_components": background_score_components,
                    "preservation_penalty": preservation_penalty,
                    "directional_pattern_penalty": pattern_penalty,
                    "nebula_preservation_penalty_weight": (
                        nebula_preservation_weight
                    ),
                    "score": candidate_score,
                    "quality_message": gate_message,
                    "preservation": preservation,
                    "directional_pattern_noise": after_pattern_report,
                    "pattern_quality_gate": pattern_gate_report,
                    "after_adaptive": after_adaptive_candidate,
                    "validation": candidate_validation,
                    "validation_gate": validation_gate,
                    "score_gate": score_gate,
                    "sufficient": sufficient,
                    **quality_record_fields,
                }
            )
            compound_report.update(
                {
                    "status": "accepted" if clean_sufficient else "accepted_degraded",
                    "accepted": True,
                    "sufficient": sufficient,
                    "severity": "soft_warning" if gate_warnings else "normal",
                    "warnings": gate_warnings,
                    "candidate_stem": candidate_stem,
                    "intermediate_stem": intermediate_stem,
                    "validation_gate": validation_gate,
                    "score_gate": score_gate,
                    "background_score": candidate_score,
                }
            )
            pipeline.log.info(
                "[Stage3] Compound Polynomial→RBF candidate accepted: "
                f"score={candidate_score:.3f} sufficient={sufficient}"
            )
            return clean_sufficient
        finally:
            if not rollback_attempted:
                _stage3_clear_background_samples(pipeline)
                restore_baseline(f"evaluated:{label}")

    mixed_pattern_route = bool(
        noise_route.get("route") == "mixed_gradient_and_pattern_noise"
    )
    if mixed_pattern_route:
        pipeline._background_review_required = True
        builtin_search_mode = "mixed_gradient_pattern_noise_review"
        pipeline.log.warn(
            "[Stage3] Directional pattern noise coexists with a supported gradient; "
            "only the low-frequency gradient may be modeled; pattern noise remains review-only"
        )

    rbf_attempts: List[Tuple[str, Tuple[str, ...], str]] = []
    for idx, raw_command in enumerate(
        pipeline._stage3_subsky_rbf_candidates(),
        start=1,
    ):
        command = tuple(raw_command)
        if "-existing" not in command:
            command += ("-existing",)
        rbf_attempts.append((f"subsky-rbf-existing-{idx}", command, "builtin"))
    poly_command = ("subsky", "1", "-existing")
    poly_attempt = [("subsky-poly-existing", poly_command, "builtin")]

    poly_first = _stage3_prefers_poly_first(
        target_profile,
        before_adaptive,
        stage3_policy=stage3_policy,
    )
    exhaustive_builtin, diffuse_context = _stage3_should_exhaust_builtin_search(
        target_profile,
        before_adaptive,
        stage3_policy,
    )
    if exhaustive_builtin:
        builtin_search_mode = (
            "mixed_gradient_pattern_noise_review"
            if mixed_pattern_route
            else "safe_samples_with_diffuse_signal_protection"
        )
        pipeline.log.info(
            "[Stage3] Diffuse nebulosity protection enabled; every candidate still requires preservation gate"
        )

    ordered_attempts, builtin_attempt_labels, builtin_order_reason = (
        _stage3_background_candidate_chain(
            pipeline,
            rbf_attempts=rbf_attempts,
            poly_attempt=poly_attempt,
            poly_first=poly_first,
        )
    )
    if safe_sample_recovery_used:
        recovery_layout = _stage3_recovery_channel_layout(before_image)
        if recovery_layout.startswith("rgb_"):
            ordered_attempts = [
                ("neutral-axis-poly1", poly_command, "builtin")
            ]
            builtin_attempt_labels = ["neutral-axis-poly1"]
            safe_sample_recovery_report["pixel_route"] = {
                "status": "enabled",
                "channel_layout": recovery_layout,
                "implementation": "siril_poly1_neutral_axis_projection",
            }
        elif recovery_layout == "mono":
            ordered_attempts = list(poly_attempt)
            builtin_attempt_labels = [poly_attempt[0][0]]
            safe_sample_recovery_report["pixel_route"] = {
                "status": "enabled",
                "channel_layout": recovery_layout,
                "implementation": "siril_subsky_degree1_existing",
            }
        else:
            ordered_attempts = []
            builtin_attempt_labels = []
            safe_sample_recovery_report.update(
                status="failed",
                reason_code="stage3_neutral_axis_rgb_unavailable",
                pixel_route={
                    "status": "rejected",
                    "channel_layout": recovery_layout,
                    "implementation": None,
                    "reason": "unsupported or malformed channel layout",
                },
            )
        builtin_order_reason = (
            "conservative_sample_recovery_polynomial_degree_1_only"
        )
        builtin_search_mode = (
            "conservative_sample_recovery_neutral_axis_poly1"
            if recovery_layout.startswith("rgb_")
            else "conservative_sample_recovery_strict_poly1"
        )
        pipeline.log.info(
            "[Stage3] Conservative sample recovery active; restrict candidate "
            "chain to one strict degree-1 Polynomial proposal"
        )
    primary_attempts_ordered = [
        record
        for record in ordered_attempts
        if record[2] == "graxpert"
    ]
    builtin_attempts_ordered = [
        record
        for record in ordered_attempts
        if record[2] == "builtin"
    ]
    external_attempts_ordered = [
        record
        for record in ordered_attempts
        if record[2] != "builtin"
        and record[2] != "graxpert"
    ]
    builtin_sufficient = evaluate_attempts(
        builtin_attempts_ordered,
        phase="builtin_primary",
    )
    compound_sufficient = False
    primary_sufficient = False
    external_sufficient = False
    if policy_abort_candidate_search:
        compound_report = {
            "status": "skipped_by_failure_policy",
            "triggered": False,
            "reason": policy_abort_reason,
            "target_guard": compound_target_guard,
            "sample_split": compound_split_report,
        }
    elif builtin_sufficient:
        compound_report = {
            "status": "not_required",
            "triggered": False,
            "reason": "builtin_primary_candidate_is_clean_and_sufficient",
            "target_guard": compound_target_guard,
            "sample_split": compound_split_report,
        }
    elif safe_sample_recovery_used:
        compound_report = {
            "status": "prohibited_by_conservative_sample_recovery",
            "triggered": False,
            "reason": "strict degree-1 Polynomial is the only permitted route",
            "target_guard": compound_target_guard,
            "sample_split": compound_split_report,
        }
    elif (
        candidate_attempt_limit > 0
        and len(attempt_records) >= candidate_attempt_limit
    ):
        compound_report = {
            "status": "not_triggered",
            "triggered": False,
            "reason": "candidate_attempt_limit_reached",
            "target_guard": compound_target_guard,
            "sample_split": compound_split_report,
        }
    elif bool(
        getattr(pipeline.cfg, "stage3_compound_candidate_enabled", True)
    ):
        compound_sufficient = evaluate_compound_candidate()
    if (
        not policy_abort_candidate_search
        and not safe_sample_recovery_used
        and not builtin_sufficient
        and not compound_sufficient
    ):
        primary_sufficient = evaluate_attempts(
            primary_attempts_ordered,
            phase="graxpert_conditional_backup",
        )
    if (
        not policy_abort_candidate_search
        and not safe_sample_recovery_used
        and not builtin_sufficient
        and not compound_sufficient
        and not primary_sufficient
    ):
        external_sufficient = evaluate_attempts(
            external_attempts_ordered,
            phase="external_backup",
        )
    if external_sufficient:
        pipeline.log.info("[Stage3] External backup produced a clean sufficient candidate")
    _stage3_clear_background_samples(pipeline)
    graxpert_attempted = any(
        record.get("source") == "graxpert" for record in attempt_records
    )
    graxpert_runtime_error = graxpert_runtime_error or any(
        record.get("status") == "graxpert_runtime_error"
        for record in attempt_records
    )

    if policy_abort_candidate_search:
        accepted_candidates.clear()
        restore_baseline("failure_policy_candidate_abort")
        pipeline._background_review_required = True
        if hasattr(pipeline, "_record_stage_policy_event"):
            pipeline._record_stage_policy_event(
                3,
                event="candidate_search_stopped",
                reason=policy_abort_reason,
                source="candidate_gate",
            )

    if accepted_candidates:
        legacy_selected = min(
            accepted_candidates,
            key=lambda item: (
                not bool(item.get("sufficient")),
                float(item.get("score", 999.0)),
            ),
        )
        selection_report = _stage3_select_candidate(
            accepted_candidates,
            legacy_selected,
        )
        recommended_label = str(
            selection_report.get("recommended_candidate") or ""
        )
        selected = next(
            (
                candidate
                for candidate in accepted_candidates
                if str(candidate.get("label") or "") == recommended_label
            ),
            legacy_selected,
        )
        selected, outer_halo_selection = _stage3_outer_halo_selection_override(
            legacy_selected,
            selected,
            stage3_policy,
        )
        selection_report["outer_halo_safety_override"] = outer_halo_selection
        selection_report["recommended_candidate"] = str(
            selected.get("label") or ""
        )
        selection_report["runtime_selected_candidate"] = str(
            selected.get("label") or ""
        )
        attempted_selected_label = str(selected.get("label") or "")
        if selected is not legacy_selected:
            pipeline.log.info(
                "[Stage3] Statistical selection applied: "
                f"legacy={legacy_selected.get('label')} "
                f"selected={selected.get('label')}"
            )
        elif outer_halo_selection.get("applied"):
            pipeline.log.info(
                "[Stage3] Outer-halo safety override kept "
                f"{selected.get('label')} instead of "
                f"{outer_halo_selection.get('statistical_candidate')}"
            )
        selected_loaded = False
        try:
            pipeline.cmd_with_check("load", str(selected["stem"]))
            selected_loaded = True
        except (CommandError, SirilError) as e:
            pipeline.log.warn(
                "failed to load best stage3 candidate; restoring baseline: "
                f"{e}"
            )
            restore_baseline(f"selected_load_failed:{selected.get('label')}")
            failure_message = (
                f"selected candidate load failed: {selected.get('label')}"
            )
            stage_message = (
                f"{preflight_message}; {failure_message}"
                if preflight_message
                else failure_message
            )
        if selected_loaded:
            bg_ok = True
            selected_source = str(selected.get("source") or "")
            selected_label = str(selected.get("label") or "")
            selected_gate_warnings = list(selected.get("gate_warnings") or [])
            compound_selected = selected_source == "compound"
            compound_selected_degraded = bool(
                compound_selected and not selected.get("sufficient")
            )
            if compound_selected_degraded:
                pipeline._background_review_required = True
            selected_preservation = selected.get("preservation") or {}
            selected_pattern_report = (
                selected.get("directional_pattern_noise") or {}
            )
            selected_spatial_opponent_projection = dict(
                selected.get("spatial_opponent_projection") or {}
            )
            selected_spatial_opponent_review_required = bool(
                selected.get("spatial_opponent_review_required", False)
            )
            if selected_spatial_opponent_review_required:
                pipeline._background_review_required = True
            selected_message = (
                f"method={selected.get('label')}; {selected.get('quality_message')}; "
                f"background_score={float(selected.get('score', 0.0)):.3f}"
            )
            if selected_gate_warnings:
                selected_message += (
                    "; soft_warnings=" + " | ".join(selected_gate_warnings)
                )
            stage_message = (
                f"{preflight_message}; {selected_message}"
                if preflight_message
                else selected_message
            )
            if selected_source == "plugin":
                pipeline.workflow_command_used["背景提取插件链"] = str(selected.get("label"))
            elif selected_source == "graxpert":
                pipeline.workflow_command_used["GraXpert 背景提取"] = str(selected.get("label"))
                _stage3_finalize_graxpert_background_model(pipeline, selected)
            elif selected_source == "compound":
                pipeline.workflow_command_used["复合背景提取"] = str(selected.get("label"))
            pipeline.log.info(
                "背景提取最终选择: "
                f"{selected.get('label')} score={float(selected.get('score', 0.0)):.3f}"
            )
    elif not bg_ok:
        restore_baseline("no_candidate_accepted")
        pipeline._background_review_required = True
        pipeline.log.error("背景提取完全失败，图像可能有梯度残留")

    _stage3_clear_background_samples(pipeline)
    pipeline._stage3_pattern_noise_report = {
        "analysis": pattern_report,
        "route": noise_route,
        "selected_candidate": selected_pattern_report or None,
    }
    stage_saved = pipeline._save_stage_output("stage3_bgremoved")
    if bg_ok and stage_saved:
        final_output_validation = _stage3_final_output_validation(
            pipeline,
            baseline_image=before_image,
            baseline_validation=baseline_validation,
            validation_points=compound_validation_points,
            patch_radius=_stage3_cfg_int(
                pipeline,
                "stage3_safe_sample_patch_radius",
                12,
                4,
                24,
            ),
            minimum_count=_stage3_cfg_int(
                pipeline,
                "stage3_compound_validation_min_count",
                4,
                4,
                20,
            ),
            enforced=isinstance(input_profile, dict),
            gate_profile=gate_profile,
            reload_stem=(
                "stage3_bgremoved"
                if safe_sample_recovery_used
                and attempted_selected_label == "neutral-axis-poly1"
                else None
            ),
            neutral_axis_enforced=bool(
                safe_sample_recovery_used
                and attempted_selected_label == "neutral-axis-poly1"
            ),
            spatial_opponent_projection=(
                selected_spatial_opponent_projection or None
            ),
            support_mask=safe_sample_support_mask,
        )
        final_output_validation["selected_candidate"] = (
            attempted_selected_label or None
        )
        final_validation_warnings = list(
            (final_output_validation.get("validation_gate") or {}).get("warnings")
            or []
        )
        if final_validation_warnings:
            selected_gate_warnings = list(
                dict.fromkeys(selected_gate_warnings + final_validation_warnings)
            )
        if (
            final_output_validation.get("enforced")
            and not final_output_validation.get("accepted", False)
        ):
            final_output_validation_rejected = True
            pipeline._background_review_required = True
            issues = ", ".join(
                dict.fromkeys(
                    str(issue)
                    for gate_name in (
                        "validation_gate",
                        "neutral_axis_persistence",
                        "spatial_opponent_persistence",
                        "directional_gradient_gate",
                    )
                    for issue in (
                        final_output_validation.get(gate_name) or {}
                    ).get("issues", [])
                )
            )
            validation_message = (
                "final saved output rejected by held-out background/RMS gate"
                + (f": {issues}" if issues else "")
            )
            pipeline.log.warn(f"[Stage3] {validation_message}")
            stage_message = (
                f"{stage_message}; {validation_message}"
                if stage_message
                else validation_message
            )
            rollback_completed = restore_baseline(
                "final_output_validation_rejected"
            )
            final_output_validation["rollback"] = {
                "attempted": True,
                "completed": rollback_completed,
            }
            if not rollback_completed:
                raise RuntimeError(
                    "Stage3 final output failed its hard gate and the immutable "
                    "baseline could not be restored"
                )
            stage_saved = pipeline._save_stage_output("stage3_bgremoved")
            final_output_validation["rollback"]["output_saved"] = stage_saved
            bg_ok = False
            selected_source = ""
            selected_label = ""
            selected_gate_warnings = []
            selected_preservation = {}
            selected_pattern_report = {}
            selected_spatial_opponent_projection = {}
            selected_spatial_opponent_review_required = False
            compound_selected = False
            compound_selected_degraded = False
            pipeline._stage3_pattern_noise_report["selected_candidate"] = None
        elif selected_loaded:
            if (
                safe_sample_recovery_used
                and attempted_selected_label == "neutral-axis-poly1"
            ):
                spatial_applied = bool(
                    str(
                        selected_spatial_opponent_projection.get(
                            "reason_code"
                        )
                        or ""
                    )
                    == "stage3_spatial_opponent_correction_applied"
                )
                verified_color_normalization = {
                    "schema": (
                        "starun.stage3-verified-background-color-normalization.v1"
                    ),
                    "applied": spatial_applied,
                    "accepted": spatial_applied,
                    "reason_code": (
                        "spatial_opponent_projection_verified"
                        if spatial_applied
                        else "neutral_axis_projection_preserves_channel_chroma"
                    ),
                }
            else:
                verified_color_normalization = (
                    _stage3_verified_background_color_normalization(
                        before_adaptive,
                        selected,
                        final_output_validation,
                        gate_profile=gate_profile,
                    )
                )
            final_output_validation["verified_background_color_normalization"] = (
                verified_color_normalization
            )
            if verified_color_normalization.get("applied", False):
                cleared_warning = str(
                    verified_color_normalization.get("cleared_warning") or ""
                )
                selected_gate_warnings = [
                    warning
                    for warning in selected_gate_warnings
                    if str(warning) != cleared_warning
                ]
                selected["gate_warnings"] = list(selected_gate_warnings)
                selected["severity"] = (
                    "soft_warning" if selected_gate_warnings else "normal"
                )
                selected["verified_background_color_normalization"] = (
                    verified_color_normalization
                )
                for record in attempt_records:
                    if str(record.get("label") or "") == str(
                        selected.get("label") or ""
                    ):
                        record["gate_warnings"] = list(selected_gate_warnings)
                        record["severity"] = selected["severity"]
                        record["status"] = (
                            "accepted_with_warnings"
                            if selected_gate_warnings
                            else "accepted_verified_color_normalization"
                        )
                        record[
                            "verified_background_color_normalization"
                        ] = verified_color_normalization
                        break
                pipeline.log.info(
                    "[Stage3] Verified background color normalization cleared "
                    "the generic clean-output sufficiency warning"
                )
                normalization_note = (
                    "verified background color normalization; target flux, "
                    "morphology, centroid, held-out sky and saved pixels accepted"
                )
                stage_message = (
                    f"{stage_message}; {normalization_note}"
                    if stage_message
                    else normalization_note
                )
    elif bg_ok:
        final_output_validation = {
            "status": "not_run",
            "enforced": isinstance(input_profile, dict),
            "selected_candidate": attempted_selected_label or None,
            "reason": "stage3 final output could not be saved",
        }
    else:
        final_output_validation = {
            "status": "not_applicable",
            "enforced": False,
            "reason": "no background candidate was selected",
        }
    verified_noop_candidate_audit = _stage3_verified_noop_candidate_audit(
        process_report,
        pattern_report,
        noise_route,
        attempt_records,
        baseline_validation=baseline_validation,
    )
    if (
        stage_saved
        and verified_noop_candidate_audit.get("eligible", False)
        and not policy_abort_candidate_search
    ):
        noop_rollback_completed = restore_baseline(
            "verified_noop_below_sampling_uncertainty"
        )
        if not noop_rollback_completed:
            raise RuntimeError(
                "Stage3 verified-noop candidate audit passed, but the immutable "
                "baseline could not be restored"
            )
        verified_noop_report = _stage3_verify_restored_noop(
            pipeline,
            baseline_image=before_image,
            baseline_validation=baseline_validation,
            validation_points=compound_validation_points,
            patch_radius=_stage3_cfg_int(
                pipeline,
                "stage3_safe_sample_patch_radius",
                12,
                4,
                24,
            ),
            minimum_count=_stage3_cfg_int(
                pipeline,
                "stage3_compound_validation_min_count",
                4,
                4,
                20,
            ),
            support_mask=safe_sample_support_mask,
            gate_profile=gate_profile,
            process_report=process_report,
            pattern_report=pattern_report,
        )
        verified_noop_report["candidate_audit"] = (
            verified_noop_candidate_audit
        )
        verified_noop_report["rollback"] = {
            "attempted": True,
            "completed": True,
        }
        stage_saved = pipeline._save_stage_output("stage3_bgremoved")
        verified_noop_report["rollback"]["output_saved"] = stage_saved
        persisted_noop_output = (
            _stage3_verify_persisted_noop_output(
                pipeline,
                baseline_image=before_image,
                output_stem="stage3_bgremoved",
            )
            if stage_saved
            else {
                "schema": "starun.stage3-persisted-noop-output.v1",
                "status": "rejected",
                "accepted": False,
                "issues": ["Stage 3 no-op output could not be saved"],
            }
        )
        verified_noop_report["persisted_output"] = persisted_noop_output
        if persisted_noop_output.get("accepted") is not True:
            verified_noop_report.update(
                status="rejected",
                accepted=False,
                issues=list(
                    dict.fromkeys(
                        [
                            *(verified_noop_report.get("issues") or []),
                            "persisted_output_identity",
                        ]
                    )
                ),
            )
        if verified_noop_report.get("accepted", False) and stage_saved:
            verified_noop = True
            bg_ok = True
            pipeline._background_review_required = False
            selected_source = ""
            selected_label = ""
            selected_gate_warnings = []
            selected_preservation = {}
            selected_pattern_report = {}
            selected_spatial_opponent_projection = {}
            selected_spatial_opponent_review_required = False
            compound_selected = False
            compound_selected_degraded = False
            pipeline._stage3_pattern_noise_report["selected_candidate"] = None
            final_output_validation = {
                "schema": "starun.stage3-final-output-validation.v1",
                "status": "accepted",
                "accepted": True,
                "enforced": True,
                "processing_route": "verified_noop",
                "verified_noop_checks": dict(
                    verified_noop_report.get("checks") or {}
                ),
            }
            noop_note = (
                "verified_noop: every scientifically assessed candidate "
                "improved held-out sky by less than three-sigma uncertainty; "
                "the immutable input was restored exactly"
            )
            stage_message = (
                f"{stage_message}; {noop_note}"
                if stage_message
                else noop_note
            )
        else:
            restore_baseline("verified_noop_persisted_output_rejected")
            pipeline._background_review_required = True
            stage_message = (
                f"{stage_message}; verified_noop revalidation failed"
                if stage_message
                else "verified_noop revalidation failed"
            )
    after_adaptive = (
        pipeline._adaptive_features_current()
        if hasattr(pipeline, "_adaptive_features_current")
        else {}
    )
    max_bg_std_growth = float(
        stage3_gate_thresholds(gate_profile)["sufficient_max_bg_std_growth"]
    )
    fallback_warning = False
    if bg_ok and not verified_noop and before_adaptive and after_adaptive:
        before_std = max(float(before_adaptive.get("bg_std", 0.0) or 0.0), 1e-7)
        after_std = float(after_adaptive.get("bg_std", 0.0) or 0.0)
        dirty = float(after_adaptive.get("dirty_background_score", 0.0) or 0.0)
        gradient_before = float(before_adaptive.get("gradient_score", 0.0) or 0.0)
        gradient_after = float(after_adaptive.get("gradient_score", 0.0) or 0.0)
        if after_std / before_std > max_bg_std_growth or (
            dirty > STAGE3_FINAL_DIRTY_WARNING_MIN
            and gradient_after
            >= gradient_before * STAGE3_FINAL_GRADIENT_RETENTION_WARNING
        ):
            fallback_warning = True
            pipeline._background_review_required = True
            warning_msg = (
                "background improvement limited "
                f"(dirty={dirty:.3f}, std_growth={after_std / before_std:.3f})"
            )
            pipeline.log.warn(f"[Stage3] {warning_msg}")
            stage_message = f"{stage_message}; {warning_msg}" if stage_message else warning_msg
    if safe_sample_recovery_used and fallback_warning:
        rollback_completed = restore_baseline(
            "conservative_recovery_final_warning"
        )
        if not rollback_completed:
            raise RuntimeError(
                "Stage3 conservative recovery produced a final warning and the "
                "immutable baseline could not be restored"
            )
        stage_saved = pipeline._save_stage_output("stage3_bgremoved")
        if not stage_saved:
            raise RuntimeError(
                "Stage3 conservative recovery rollback could not publish the "
                "canonical baseline output"
            )
        final_output_validation["recovery_warning_rollback"] = {
            "attempted": True,
            "completed": True,
            "output_saved": True,
        }
        bg_ok = False
        selected_source = ""
        selected_label = ""
        selected_gate_warnings = []
        selected_preservation = {}
        selected_pattern_report = {}
        pipeline._stage3_pattern_noise_report["selected_candidate"] = None
    background_backup_used = bool(
        bg_ok and selected_source not in ("builtin", "")
    )
    background_backup_reason = (
        "builtin_primary_insufficient_compound_selected"
        if background_backup_used and selected_source == "compound"
        else "builtin_and_compound_not_clean_graxpert_selected"
        if background_backup_used and selected_source == "graxpert"
        else "graxpert_runtime_error_external_selected"
        if background_backup_used and graxpert_runtime_error
        else "builtin_compound_and_graxpert_not_clean_external_selected"
        if background_backup_used
        else None
    )
    custom_sample_attempted = any(
        record.get("source") == "builtin" for record in attempt_records
    )
    custom_sample_backup_used = bool(
        bg_ok
        and selected_source not in ("", "builtin", "compound")
        and custom_sample_attempted
    )
    safe_sample_install_failed = any(
        record.get("status") == "safe_sample_install_failed"
        for record in attempt_records
    )
    safe_sample_report = {
        **safe_sample_report,
        "subsky_existing_enforced": True,
        "install_failed": safe_sample_install_failed,
        "selected_source": selected_source or None,
        "backup_used": background_backup_used,
        "fallback_used": False,
    }
    pipeline._stage3_safe_sample_report = safe_sample_report
    stage_fallback_used = bool(
        profile_fallback_used
        or final_output_validation_rejected
    )
    fallback_reasons = [
        reason
        for reason, enabled in (
            ("target_profiler_fallback", profile_fallback_used),
            (
                "final_output_validation_rejected_baseline_restored",
                final_output_validation_rejected,
            ),
        )
        if enabled
    ]
    pattern_review_required = bool(noise_route.get("requires_review", False))
    background_review_required = bool(
        pattern_review_required
        or compound_selected_degraded
        or selected_spatial_opponent_review_required
        or final_output_validation_rejected
        or fallback_warning
        or not bg_ok
        or not stage_saved
        or bool(getattr(pipeline, "_background_review_required", False))
    )
    if bg_ok and stage_saved:
        spatial_background_lineage = _stage3_write_spatial_background_lineage(
            pipeline,
            baseline_image=before_image,
            fit_points=compound_fit_points,
            validation_points=compound_validation_points,
            patch_radius=_stage3_cfg_int(
                pipeline,
                "stage3_safe_sample_patch_radius",
                12,
                4,
                24,
            ),
            support_mask=safe_sample_support_mask,
            projection=selected_spatial_opponent_projection,
            review_required=background_review_required,
            processing_route=(
                "verified_noop"
                if verified_noop
                else "background_correction"
            ),
        )
        if spatial_background_lineage.get("accepted") is not True:
            selected_spatial_opponent_review_required = True
            background_review_required = True
            pipeline._background_review_required = True
            selected_spatial_opponent_projection = {
                **selected_spatial_opponent_projection,
                "reason_code": "stage3_spatial_opponent_lineage_unverified",
                "lineage_issues": list(
                    spatial_background_lineage.get("issues") or []
                ),
            }
    reason_code = _stage3_outcome_reason_code(
        policy_abort_candidate_search=policy_abort_candidate_search,
        failure_action=failure_action,
        final_output_validation_rejected=final_output_validation_rejected,
        bg_ok=bg_ok,
        stage_saved=stage_saved,
        pattern_review_required=pattern_review_required,
        compound_selected_degraded=compound_selected_degraded,
        selected_gate_warnings=selected_gate_warnings,
        background_backup_used=background_backup_used,
        profile_fallback_used=profile_fallback_used,
        fallback_warning=fallback_warning,
        review_required=background_review_required,
    )
    if (
        safe_sample_recovery_used
        and bg_ok
        and not background_review_required
    ):
        spatial_reason = str(
            selected_spatial_opponent_projection.get("reason_code") or ""
        )
        reason_code = (
            "stage3_spatial_opponent_correction_applied"
            if spatial_reason == "stage3_spatial_opponent_correction_applied"
            else "stage3_safe_sample_recovery_applied"
        )
    elif safe_sample_recovery_used and selected_spatial_opponent_review_required:
        reason_code = str(
            selected_spatial_opponent_projection.get("reason_code")
            or "stage3_spatial_opponent_lineage_unverified"
        )
    elif safe_sample_recovery_used and not bg_ok:
        true_sky_supported = bool(
            ((process_report or {}).get("true_sky_support") or {}).get(
                "supported",
                False,
            )
        )
        reason_code = (
            "insufficient_source_masked_true_sky_support"
            if not true_sky_supported
            else "stage3_conservative_recovery_candidate_rejected"
        )
    if verified_noop and not background_review_required:
        reason_code = "verified_noop_below_sampling_uncertainty"
    if background_review_required:
        pipeline._require_review(
            3,
            reason_code,
        )
    report_quality = (
        "review_required"
        if background_review_required
        else "ok"
        if bg_ok
        else "degraded"
    )
    if pattern_review_required:
        pattern_note = (
            "directional pattern noise remains unresolved; "
            f"route={noise_route.get('route')}"
        )
        stage_message = (
            f"{stage_message}; {pattern_note}"
            if stage_message
            else pattern_note
        )
    if compound_selected_degraded:
        compound_note = (
            "compound Polynomial→RBF backup passed safety validation but "
            "did not reach sufficient background quality; review-only output required"
        )
        stage_message = (
            f"{stage_message}; {compound_note}"
            if stage_message
            else compound_note
        )
    if hasattr(pipeline, "_write_stage_json"):
        pipeline._write_stage_json(
            "background_quality_report.json",
            {
                "schema_version": STAGE3_BACKGROUND_QUALITY_SCHEMA,
                "algorithm_contract_version": STAGE3_ALGORITHM_CONTRACT_VERSION,
                "stage": "stage3_background",
                "reason_code": reason_code,
                "backend_policy": str(
                    getattr(pipeline.cfg, "stage3_backend_policy", "auto_chain")
                ),
                "gate_profile": gate_profile,
                "configured_gate_profile": configured_gate_profile,
                "effective_gate_profile": gate_profile,
                "failure_action": failure_action,
                "candidate_search_stopped": policy_abort_candidate_search,
                "candidate_search_stop_reason": policy_abort_reason or None,
                "policy": policy_name,
                "decision_thresholds": decision_thresholds,
                "decision": background_decision,
                "process_evidence": process_report,
                "model_used": selected_label or None,
                "backend_provenance": {
                    "graxpert": getattr(
                        pipeline,
                        "_stage3_graxpert_provenance",
                        None,
                    ),
                },
                "attempted_selected_model": attempted_selected_label or None,
                "graxpert_attempted": graxpert_attempted,
                "graxpert_runtime_error": graxpert_runtime_error,
                "graxpert_error_reasons": graxpert_error_reasons,
                "fallback_triggered_by_graxpert_error": bool(
                    graxpert_runtime_error and selected_source != "graxpert"
                ),
                "preferred_candidate": (
                    "strict degree-1 Polynomial"
                    if safe_sample_recovery_used
                    else "target-aware builtin Polynomial/RBF"
                ),
                "preferred_candidate_sufficient": builtin_sufficient,
                "backup_used": background_backup_used,
                "backup_reason": background_backup_reason,
                "custom_sample_backup_used": custom_sample_backup_used,
                "builtin_order_reason": builtin_order_reason,
                "candidate_order": [record[0] for record in ordered_attempts],
                "evaluated_candidate_order": [
                    str(record.get("label") or "")
                    for record in attempt_records
                ],
                "builtin_candidate_order": builtin_attempt_labels,
                "builtin_search_mode": builtin_search_mode,
                "builtin_sufficient": builtin_sufficient,
                "compound_fallback": compound_report,
                "diffuse_nebula_context": diffuse_context,
                "safe_samples": safe_sample_report,
                "shared_scene_support": scene_support_report,
                "subsky_existing_enforced": True,
                "directional_pattern_noise": pattern_report,
                "selected_directional_pattern_noise": (
                    selected_pattern_report or None
                ),
                "noise_route": noise_route,
                "protection_policy_flags": protection_policy_flags,
                "before": before_adaptive,
                "after": after_adaptive,
                "attempts": attempt_records,
                "neutral_axis_projection": next(
                    (
                        record.get("neutral_axis_projection")
                        for record in reversed(attempt_records)
                        if record.get("neutral_axis_projection") is not None
                    ),
                    None,
                ),
                "spatial_opponent_projection": (
                    selected_spatial_opponent_projection or next(
                        (
                            record.get("spatial_opponent_projection")
                            for record in reversed(attempt_records)
                            if record.get("spatial_opponent_projection")
                            is not None
                        ),
                        None,
                    )
                ),
                "spatial_background_lineage": spatial_background_lineage,
                "verified_noop_candidate_audit": (
                    verified_noop_candidate_audit
                ),
                "verified_noop": verified_noop_report,
                "selection": selection_report,
                "selected_gate_warnings": selected_gate_warnings,
                "verified_background_color_normalization": (
                    verified_color_normalization
                ),
                "final_output_validation": final_output_validation,
                "rollback_events": rollback_events,
                "selected_preservation": selected_preservation,
                "quality": report_quality,
                "review_required": background_review_required,
                "fallback_used": stage_fallback_used,
                "fallback_reasons": fallback_reasons,
                "fallback_reason": (
                    "final_output_validation_rejected_baseline_restored"
                    if final_output_validation_rejected
                    else "target_profiler_fallback"
                    if profile_fallback_used
                    else None
                ),
            },
        )
    if not stage_saved:
        stage_message = (
            f"{stage_message}; stage3 输出保存失败"
            if stage_message
            else "stage3 输出保存失败"
        )
    elif hasattr(pipeline, "_create_stage_review_bundle"):
        review = pipeline._create_stage_review_bundle(
            "stage3_background_extraction",
            baseline_stem,
            "stage3_bgremoved",
            context={
                "method": selected_label or None,
                "quality": report_quality,
                "noise_route": noise_route.get("route"),
            },
            candidates=attempt_records,
            selected_candidate=selected_label or None,
        )
        if review.get("report_path"):
            review_note = f"review_bundle={review['report_path']}"
            stage_message = f"{stage_message}; {review_note}" if stage_message else review_note

    elapsed = pipeline.log.stage_end(stage_label)
    components = {
        "target_profile": {
            "status": "applied",
            "method": target_profile.get("classification_method"),
            "reason_code": (
                "target_profiler_fallback"
                if profile_fallback_used
                else "accepted"
            ),
            "fallback_used": profile_fallback_used,
        },
        "background_extraction": {
            "status": (
                "rolled_back"
                if final_output_validation_rejected
                else "skipped"
                if verified_noop
                else "review_required"
                if background_review_required and bg_ok
                else "applied"
                if bg_ok
                else "rolled_back"
            ),
            "method": selected_label or None,
            "attempted_method": attempted_selected_label or None,
            "reason_code": (
                "final_output_validation_rejected"
                if final_output_validation_rejected
                else "verified_noop_below_sampling_uncertainty"
                if verified_noop
                else "compound_poly_residual_rbf_degraded_review"
                if compound_selected_degraded
                else "accepted_with_soft_warnings"
                if selected_gate_warnings
                else "backup_accepted"
                if background_backup_used
                else "accepted"
                if bg_ok
                else "no_candidate_accepted"
            ),
            "input": baseline_stem,
            "output": "stage3_bgremoved" if stage_saved else None,
            "fallback_used": bool(final_output_validation_rejected),
            "backup_used": background_backup_used,
            "backup_reason": background_backup_reason,
        },
        "directional_pattern_router": {
            "status": (
                "review_required" if pattern_review_required else "accepted"
            ),
            "method": noise_route.get("route"),
            "reason_code": (
                "mixed_gradient_pattern_noise_review"
                if noise_route.get("route") == "mixed_gradient_and_pattern_noise"
                else "no_directional_pattern_detected"
            ),
            "fallback_used": False,
        },
    }
    if bg_ok:
        status = (
            "degraded"
            if background_review_required or not stage_saved
            else "ok"
        )
        pipeline._record_stage(
            stage_label,
            status,
            elapsed,
            stage_message,
            execution="skipped" if verified_noop else "completed",
            fallback_used=stage_fallback_used,
            reason_code=reason_code,
            components=components,
            review_reasons=pipeline._stage_review_reasons(3),
        )
        if selected_source == "builtin":
            pipeline.log.info("阶段3按策略使用内置 subsky/RBF 背景提取")
    else:
        degrade_message = (
            stage_message
            if final_output_validation_rejected and stage_message
            else "背景提取失败，图像可能有梯度残留"
        )
        if not stage_saved:
            degrade_message += "；stage3 输出保存失败"
        pipeline._record_stage(
            stage_label,
            "failed" if failure_action == "stop" else "degraded",
            elapsed,
            degrade_message,
            execution="safe_passthrough",
            fallback_used=bool(
                stage_fallback_used or policy_abort_candidate_search
            ),
            upstream_passthrough=bool(policy_abort_candidate_search),
            reason_code=reason_code,
            details={
                "backend_policy": str(
                    getattr(pipeline.cfg, "stage3_backend_policy", "auto_chain")
                ),
                "failure_action": failure_action,
                "candidate_search_stopped": policy_abort_candidate_search,
            },
            components=components,
            review_reasons=(
                pipeline._stage_review_reasons(3)
                if failure_action != "stop"
                else []
            ),
        )
