"""Stage 8 starless nebula enhancement."""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from sirilpy.exceptions import CommandError, SirilError

from channel_semantics import BROADBAND_RGB_OSC, NARROWBAND_COMPOSITE
from color_quality_metrics import (
    build_color_quality_report,
    physical_broadband_anchor_accepted,
    resolve_color_contract,
)
from dualband_palette import (
    DUALBAND_PALETTE_SCHEMA,
    DUALBAND_PALETTE_SOURCE,
    PALETTE_CHANNELS,
    build_dualband_palette_candidate,
    stage8_palette_eligibility,
)
from image_metrics import format_feature_summary
from models import PipelineStage, StarSeparationState
from pipeline_safety import color_safety_limits, clamp_saturation_boost
import run_manifest
from stage8_handoff import (
    seal_stage8_handoff,
    verify_stage8_handoff_integrity,
)
import stage7_stretch_metrics
import star_halo_guard
import spatial_background_lineage
from stage8_pixels import (
    stage8_large_galaxy_structure_masks,
    stage8_mixed_nebula_composite_context,
    stage8_outside_target_identity_report,
    stage8_restore_rgb_like,
    stage8_star_preserve_nebulosity_context,
    stage8_star_preserve_nebulosity_overlay,
    stage8_subject_boundary_retry_candidate,
    stage8_subject_boundary_seam_report,
    stage8_target_structure_masks,
)
from stage8_color_rendition import (
    STAGE8_SUBJECT_CHROMA_SCHEMA,
    apply_subject_chroma_rendition,
    assess_subject_chroma_candidate,
    subject_saturation_median,
    target_aware_chroma_factor,
)
from stage8_starless_finish import (
    DECODED_PIXEL_SHA256_METHOD,
    FITS_DATA_SHA256_METHOD,
    STAGE8_STARLESS_FINISH_SCHEMA,
    assess_finish_candidate,
    canonical_decoded_pixel_sha256,
    pixel_sha256,
    persisted_fits_decoded_pixel_sha256,
    project_linked_luminance_candidate,
    project_luminance_locked_color_candidate,
)


STAGE8_HANDOFF_SCHEMA = "starun.stage8-handoff.v3"

_STAGE8_LIMITED_SAFE_REASON_CODES = {
    "bright_nebula_halo_advisory",
    "stage6_quality_advisory",
    "stage6_starless_pixel_repair_accepted",
    "user_stage8_cap=limited",
}


def _write_stage8_handoff(
    pipeline,
    handoff: Dict[str, Any],
) -> Dict[str, Any]:
    """Seal, self-verify, and persist the exact Stage 8 handoff record."""

    sealed = seal_stage8_handoff(handoff)
    verification = verify_stage8_handoff_integrity(sealed)
    if verification.get("accepted") is not True:
        raise RuntimeError("stage8_handoff_writer_self_verification_failed")
    pipeline._stage8_handoff = sealed
    pipeline._write_stage_json("stage8_handoff.json", sealed)
    return sealed


def _stage8_source_identity(pipeline, source_stem: Optional[str]) -> Dict[str, Any]:
    """Resolve the persisted Stage8 artifact identity without changing images."""

    stem = str(source_stem or "").strip()
    filename = f"{stem}.fit" if stem else None
    process_dir = getattr(pipeline, "process_dir", None)
    path = process_dir / filename if process_dir is not None and filename else None
    container_sha256 = (
        run_manifest.sha256_file(path)
        if path is not None and path.is_file()
        else None
    )
    fits_data_digest = None
    decoded_pixel_digest = None
    fingerprint = None
    fingerprint_reader = getattr(pipeline, "_fits_stage_fingerprint", None)
    if callable(fingerprint_reader) and path is not None and path.is_file():
        try:
            fingerprint = fingerprint_reader(path)
        except (OSError, RuntimeError, TypeError, ValueError):
            fingerprint = None
    if isinstance(fingerprint, dict):
        fits_data_digest = str(fingerprint.get("data_sha256") or "") or None
    if path is not None and path.is_file():
        try:
            decoded_pixel_digest = persisted_fits_decoded_pixel_sha256(path)
        except (OSError, RuntimeError, TypeError, ValueError):
            decoded_pixel_digest = None
    if fits_data_digest is None or decoded_pixel_digest is None:
        saved = getattr(pipeline, "saved_image_pixels", None)
        if isinstance(saved, dict) and stem in saved:
            try:
                if fits_data_digest is None:
                    fits_data_digest = pixel_sha256(saved[stem])
                if decoded_pixel_digest is None:
                    decoded_pixel_digest = canonical_decoded_pixel_sha256(
                        saved[stem]
                    )
            except (TypeError, ValueError):
                pass
    identity_status = (
        "verified"
        if container_sha256 and fits_data_digest and decoded_pixel_digest
        else "container_verified"
        if container_sha256
        else "unavailable"
    )
    return {
        "artifact": filename,
        "sha256": container_sha256,
        # ``pixel_sha256`` remains the v3 FITS logical-data identity for
        # compatibility.  The decoded buffer uses its own explicitly named
        # digest domain and must never be compared directly with this value.
        "pixel_sha256": fits_data_digest,
        "pixel_sha256_method": FITS_DATA_SHA256_METHOD,
        "fits_data_sha256": fits_data_digest,
        "fits_data_sha256_method": FITS_DATA_SHA256_METHOD,
        "decoded_pixel_sha256": decoded_pixel_digest,
        "decoded_pixel_sha256_method": DECODED_PIXEL_SHA256_METHOD,
        "identity_status": identity_status,
    }


def _set_stage8_handoff(
    pipeline,
    *,
    source_stem: Optional[str],
    passthrough: bool,
    restricted_downstream: bool,
    final_quality: str,
    processing_route: str = "review_only",
    formal_eligible: bool = False,
    reason_code: str = "",
    reason_text: str = "",
) -> Dict[str, Any]:
    """Finalize Stage 8 provenance without overloading fallback semantics."""
    handoff = dict(getattr(pipeline, "_stage8_handoff", {}) or {})
    reasons = list(handoff.get("reasons") or [])
    effective_reason_code = str(
        reason_code or handoff.get("reason_code") or ""
    )
    effective_reason_text = str(
        reason_text or handoff.get("reason_text") or effective_reason_code
    )
    if reason_code and not any(
        isinstance(item, dict) and item.get("code") == reason_code
        for item in reasons
    ):
        reasons.append({"code": reason_code, "source_stage": 8})
    source_identity = _stage8_source_identity(pipeline, source_stem)
    input_stem = str(getattr(pipeline, "_stage8_input_source", "") or "")
    input_identity = _stage8_source_identity(pipeline, input_stem)
    source_verified = bool(
        source_identity.get("sha256")
        and source_identity.get("fits_data_sha256")
        and source_identity.get("decoded_pixel_sha256")
    )
    input_verified = bool(
        not input_stem
        or (
            input_identity.get("sha256")
            and input_identity.get("fits_data_sha256")
            and input_identity.get("decoded_pixel_sha256")
        )
    )
    lineage_verified = bool(source_verified and input_verified)
    cumulative = dict(
        getattr(pipeline, "_stage8_final_cumulative_quality_report", {}) or {}
    )
    cumulative_required = processing_route in {
        "structure_enhanced",
        "safe_passthrough_color_only",
    }
    cumulative_verified = bool(
        not cumulative_required
        or (
            cumulative.get("schema")
            == "starun.stage8-final-cumulative-quality.v1"
            and cumulative.get("status") == "accepted"
            and cumulative.get("accepted") is True
            and cumulative.get("fresh_evaluation") is True
            and not list(cumulative.get("issues") or [])
        )
    )
    effective_restricted_downstream = bool(
        restricted_downstream
        or (cumulative_required and not cumulative_verified)
    )
    effective_formal_eligible = bool(
        formal_eligible
        and not effective_restricted_downstream
        and lineage_verified
        and cumulative_verified
        and processing_route
        in {
            "structure_enhanced",
            "safe_passthrough_color_only",
            "star_preserve_secondary_nebulosity",
        }
    )
    handoff.update(
        {
            "schema": STAGE8_HANDOFF_SCHEMA,
            "source_stem": source_stem,
            "passthrough": bool(passthrough),
            "restricted_downstream": effective_restricted_downstream,
            "processing_route": str(processing_route),
            "formal_eligible": effective_formal_eligible,
            "reason_code": effective_reason_code,
            "reason_text": effective_reason_text,
            "reasons": reasons,
            "final_quality": final_quality,
            "source_artifact": source_identity,
            "artifact_sha256": source_identity.get("sha256"),
            "pixel_sha256": source_identity.get("pixel_sha256"),
            "pixel_sha256_method": source_identity.get("pixel_sha256_method"),
            "fits_data_sha256": source_identity.get("fits_data_sha256"),
            "fits_data_sha256_method": source_identity.get(
                "fits_data_sha256_method"
            ),
            "decoded_pixel_sha256": source_identity.get(
                "decoded_pixel_sha256"
            ),
            "decoded_pixel_sha256_method": source_identity.get(
                "decoded_pixel_sha256_method"
            ),
            "lineage": {
                "input_stem": input_stem or None,
                "input_artifact": input_identity if input_stem else None,
                "output_stem": source_stem,
                "output_artifact": source_identity,
            },
            "starless_finish_report": "stage8_starless_finish_report.json",
            "final_cumulative_quality_report": (
                "stage8_final_cumulative_quality.json"
                if cumulative_required
                else None
            ),
            "final_cumulative_quality_verified": cumulative_verified,
        }
    )
    handoff["lineage_verified"] = lineage_verified
    return _write_stage8_handoff(pipeline, handoff)


def _stage8_review_requirements_through_stage8(pipeline) -> List[Dict[str, Any]]:
    """Return persisted review requirements that already block formal Stage8."""

    payload_builder = getattr(pipeline, "_review_requirements_payload", None)
    if callable(payload_builder):
        try:
            payload = payload_builder(through_stage=8)
        except (TypeError, ValueError):
            payload = []
        if isinstance(payload, list):
            return [dict(item) for item in payload if isinstance(item, dict)]
    reasons: List[Dict[str, Any]] = []
    reason_reader = getattr(pipeline, "_stage_review_reasons", None)
    if callable(reason_reader):
        for stage in range(1, 9):
            try:
                stage_reasons = reason_reader(stage)
            except (TypeError, ValueError):
                stage_reasons = []
            reasons.extend(
                {"stage": stage, "code": str(code)}
                for code in (stage_reasons or [])
                if str(code).strip()
            )
    return reasons


def _stage8_limited_safe_passthrough_eligibility(
    pipeline,
    *,
    stage8_guard_report: Dict[str, Any],
    final_source: str,
    final_quality: str,
    user_processing_mode: str,
    external_override: bool,
) -> Dict[str, Any]:
    """Separate an upstream advisory from a retained scientific hard failure."""

    incoming = dict(getattr(pipeline, "_stage8_handoff", {}) or {})
    guard = dict(stage8_guard_report or {})
    reason_codes = {
        str(item.get("code") or "").strip()
        for item in (incoming.get("reasons") or [])
        if isinstance(item, dict) and str(item.get("code") or "").strip()
    }
    primary_reason = str(
        guard.get("reason_code") or incoming.get("reason_code") or ""
    ).strip()
    if primary_reason:
        reason_codes.add(primary_reason)

    metrics = dict(incoming.get("metrics") or {})
    metric_pairs = (
        ("residual_star_score", "residual_star_hard_limit"),
        ("starless_noise_gain", "starless_noise_gain_hard_limit"),
        ("effective_halo_residue_score", "halo_residue_hard_limit"),
    )
    metric_evidence_complete = True
    upstream_hard_metric_clear = True
    metric_checks: Dict[str, Any] = {}
    for value_name, limit_name in metric_pairs:
        try:
            value = float(metrics[value_name])
            limit = float(metrics[limit_name])
            accepted = bool(np.isfinite(value) and np.isfinite(limit) and value <= limit)
        except (KeyError, TypeError, ValueError):
            value = None
            limit = None
            accepted = False
            metric_evidence_complete = False
        upstream_hard_metric_clear = bool(upstream_hard_metric_clear and accepted)
        metric_checks[value_name] = {
            "value": value,
            "hard_limit": limit,
            "accepted": accepted,
        }

    review_requirements = _stage8_review_requirements_through_stage8(pipeline)
    reason_codes_supported = bool(
        reason_codes
        and reason_codes.issubset(_STAGE8_LIMITED_SAFE_REASON_CODES)
    )
    if not reason_codes and str(user_processing_mode).strip().lower() == "limited":
        reason_codes_supported = True
    checks = {
        "limited_policy": str(guard.get("processing_policy") or "").lower()
        == "limited",
        "guard_status": str(guard.get("status") or "").lower() == "ok",
        "guard_hard_reasons_clear": not list(guard.get("hard_reasons") or []),
        "guard_subject_reasons_clear": not list(
            guard.get("subject_reasons") or []
        ),
        "upstream_quality_ok": str(incoming.get("quality_status") or "").lower()
        == "ok",
        "upstream_reason_is_safe": reason_codes_supported,
        "upstream_hard_metric_evidence_complete": metric_evidence_complete,
        "upstream_hard_metrics_clear": upstream_hard_metric_clear,
        "final_candidate_ready": (
            str(final_source) == "stage8_enhanced"
            and str(final_quality) == "ok"
        ),
        "processing_mode_safe": str(user_processing_mode).strip().lower()
        in {"auto", "limited"},
        "no_external_override": not bool(external_override),
        "review_requirement_free": not review_requirements,
    }
    issues = [name for name, accepted in checks.items() if not accepted]
    return {
        "schema": "starun.stage8-limited-safe-passthrough-eligibility.v1",
        "status": "eligible" if not issues else "rejected",
        "accepted": not issues,
        "checks": checks,
        "issues": issues,
        "reason_codes": sorted(reason_codes),
        "upstream_hard_metrics": metric_checks,
        "review_requirements": review_requirements,
    }


def _stage8_safe_passthrough_preflight(
    pipeline,
    *,
    source_mode: str = "structure_rollback",
    eligibility: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Verify an exact structure rollback before any color-only transaction."""

    report: Dict[str, Any] = {
        "schema": "starun.stage8-safe-passthrough-preflight.v1",
        "status": "rejected",
        "accepted": False,
        "route": "safe_passthrough_color_only",
        "source_mode": str(source_mode),
        "eligibility": dict(eligibility or {}),
        "checks": {},
        "issues": [],
    }
    try:
        baseline = pipeline._read_image_by_stem("stage8_input_starless")
        canonical = pipeline._read_image_by_stem("stage8_enhanced")
        if baseline is None or canonical is None:
            raise ValueError("safe_passthrough_pixels_unavailable")
        baseline_array = np.asarray(baseline)
        canonical_array = np.asarray(canonical)
        baseline_identity = _stage8_source_identity(
            pipeline, "stage8_input_starless"
        )
        canonical_identity = _stage8_source_identity(pipeline, "stage8_enhanced")
        rollback_identity = bool(
            baseline_array.shape == canonical_array.shape
            and np.array_equal(baseline_array, canonical_array)
            and baseline_identity.get("pixel_sha256")
            and baseline_identity.get("pixel_sha256")
            == canonical_identity.get("pixel_sha256")
        )
        report["checks"]["exact_structure_rollback"] = {
            "accepted": rollback_identity,
            "baseline": baseline_identity,
            "canonical": canonical_identity,
        }

        presentation = dict(
            getattr(pipeline, "_stage7_presentation_reference_report", {}) or {}
        )
        presentation_path = (
            Path(pipeline.process_dir) / "stage7_presentation_reference.json"
            if getattr(pipeline, "process_dir", None) is not None
            else None
        )
        if not presentation and presentation_path is not None:
            presentation = json.loads(presentation_path.read_text(encoding="utf-8"))
        unsigned_presentation = dict(presentation)
        expected_report_sha = str(
            unsigned_presentation.pop("report_sha256", "") or ""
        )
        source_artifact = dict(presentation.get("source_artifact") or {})
        selected_candidate = dict(presentation.get("selected_candidate") or {})
        baseline_presentation_pixel_sha = (
            stage7_stretch_metrics.stage7_pixel_sha256(baseline_array)
        )
        binding_payload = {
            "linear_source": presentation.get("linear_source"),
            "selected_candidate": presentation.get("selected_candidate"),
            "matched_domain": presentation.get("matched_domain"),
            "formal_source_artifact": source_artifact,
        }
        presentation_accepted = bool(
            presentation.get("schema")
            == "starun.stage7-presentation-reference.v1"
            and presentation.get("status") == "ready"
            and presentation.get("accepted") is True
            and presentation.get("reference_only") is True
            and expected_report_sha
            == stage7_stretch_metrics.canonical_json_sha256(
                unsigned_presentation
            )
            and str(presentation.get("source_binding_sha256") or "")
            == stage7_stretch_metrics.canonical_json_sha256(binding_payload)
            and source_artifact.get("container_sha256")
            and source_artifact.get("pixel_sha256")
            and selected_candidate.get("container_sha256")
            and selected_candidate.get("pixel_sha256")
            and source_artifact.get("pixel_sha256")
            == baseline_presentation_pixel_sha
            and selected_candidate.get("pixel_sha256")
            == baseline_presentation_pixel_sha
        )
        report["checks"]["stage7_presentation_reference"] = {
            "accepted": presentation_accepted,
            "schema": presentation.get("schema"),
            "status": presentation.get("status"),
            "source_binding_sha256": presentation.get("source_binding_sha256"),
            "report_sha256": expected_report_sha or None,
            "expected_pixel_sha256": source_artifact.get("pixel_sha256"),
            "actual_pixel_sha256": baseline_presentation_pixel_sha,
        }

        spatial = spatial_background_lineage.assess_final_spatial_background(
            getattr(pipeline, "process_dir", None),
            baseline_array,
        )
        report["checks"]["spatial_background"] = spatial
        seam = stage8_subject_boundary_seam_report(
            pipeline,
            baseline_array,
            baseline_array,
        )
        report["checks"]["subject_boundary_seam"] = seam
        masks = pipeline._stage8_generate_starless_masks(baseline_array)
        halo = star_halo_guard.assess_candidate(
            baseline_array,
            baseline_array,
            masks.get("star_halo_guard_mask"),
            mode="color",
        )
        report["checks"]["star_halo"] = halo
        finite = bool(np.all(np.isfinite(baseline_array)))
        clipping = {
            "accepted": finite,
            "finite": finite,
            "clip_ratio": float(
                np.mean((baseline_array <= 0.0) | (baseline_array >= 1.0))
            ),
            "clip_growth": 0.0,
        }
        report["checks"]["clipping"] = clipping
        checks = {
            "exact_structure_rollback": rollback_identity,
            "stage7_presentation_reference": presentation_accepted,
            "spatial_background": bool(spatial.get("accepted", False)),
            "subject_boundary_seam": bool(seam.get("accepted", False)),
            "star_halo": bool(halo.get("accepted", False)),
            "clipping": bool(clipping.get("accepted", False)),
        }
        report["issues"] = [name for name, accepted in checks.items() if not accepted]
        report["accepted"] = not report["issues"]
        report["status"] = "accepted" if report["accepted"] else "rejected"
    except (
        AttributeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        report["issues"] = [str(error)]
    pipeline._stage8_safe_passthrough_color_only_preflight = dict(report)
    pipeline._write_stage_json("stage8_safe_passthrough_preflight.json", report)
    return report


def _stage8_frozen_color_masks(
    pipeline,
    image_data: np.ndarray,
) -> tuple[Dict[str, Any], List[str]]:
    """Rebuild the same frozen Stage 7/8 ROI set used by color transactions."""

    generated_masks = pipeline._stage8_generate_starless_masks(image_data)
    if not isinstance(generated_masks, dict):
        raise ValueError("Stage8 color masks are unavailable")
    background = np.asarray(generated_masks.get("background_mask"))
    if background.ndim != 2:
        raise ValueError("Stage8 background mask is invalid")
    spatial_shape = tuple(int(value) for value in background.shape)
    masks: Dict[str, Any] = {
        key: np.array(value, copy=True)
        for key, value in generated_masks.items()
        if value is not None
        and np.asarray(value).ndim == 2
        and tuple(np.asarray(value).shape) == spatial_shape
    }
    frozen_masks = getattr(pipeline, "_stage7_frozen_rendition_masks", {})
    frozen_keys: List[str] = []
    if isinstance(frozen_masks, dict):
        for key, value in frozen_masks.items():
            if value is None:
                continue
            array = np.asarray(value)
            if array.ndim != 2 or tuple(array.shape) != spatial_shape:
                continue
            if not np.all(np.isfinite(array)):
                continue
            masks[key] = np.array(array, dtype=np.float32, copy=True)
            frozen_keys.append(str(key))
    if masks.get("subject_mask") is None:
        subject_layers = [
            np.asarray(masks[name], dtype=np.float32)
            for name in (
                "core_mask",
                "nebula_mask",
                "faint_nebula_mask",
                "galaxy_signal_mask",
            )
            if masks.get(name) is not None
        ]
        masks["subject_mask"] = (
            np.maximum.reduce(subject_layers)
            if subject_layers
            else np.clip(
                1.0 - np.asarray(masks["background_mask"], dtype=np.float32),
                0.0,
                1.0,
            )
        ).astype(np.float32, copy=False)
    return masks, sorted(frozen_keys)


def _stage8_mutation_union_outside_identity(
    pipeline,
    baseline_array: np.ndarray,
    candidate_array: np.ndarray,
    *,
    subject_chroma_report: Dict[str, Any],
    palette_report: Dict[str, Any],
    include_structure: bool,
) -> Dict[str, Any]:
    """Verify exact pixels outside every independently authorized mutation ROI."""

    target_type = (
        str(pipeline._active_target_type() or "").strip().lower()
        if hasattr(pipeline, "_active_target_type")
        else ""
    )
    baseline_shape = np.asarray(baseline_array).shape
    if len(baseline_shape) == 2:
        spatial_shape = tuple(int(value) for value in baseline_shape)
    elif len(baseline_shape) == 3:
        spatial_shape = tuple(int(value) for value in baseline_shape[-2:])
    else:
        raise ValueError("stage8_mutation_union_pixel_layout_unsupported")
    allowed = np.zeros(spatial_shape, dtype=bool)
    support_sources: List[str] = []
    structure_mask_report: Dict[str, Any] = {}
    structure_target_types = {
        "large_galaxy",
        "emission_nebula",
        "emission_nebula_widefield",
        "bright_emission_reflection_nebula",
    }
    needs_masks = bool(
        (include_structure and target_type in structure_target_types)
        or subject_chroma_report.get("accepted", False)
        or palette_report.get("accepted", False)
    )
    masks: Dict[str, Any] = {}
    frozen_keys: List[str] = []
    if needs_masks:
        masks, frozen_keys = _stage8_frozen_color_masks(
            pipeline,
            baseline_array,
        )
    if include_structure and target_type in structure_target_types:
        routed_masks, structure_mask_report = stage8_target_structure_masks(
            pipeline,
            dict(masks),
        )
        if routed_masks is None:
            raise ValueError(
                "stage8_mutation_union_structure_support_unavailable:"
                + str(structure_mask_report.get("reason") or "unknown")
            )
        structure_support = np.asarray(
            routed_masks.get("enhancement_support_weight"),
            dtype=np.float32,
        )
        if structure_support.shape != spatial_shape:
            raise ValueError("stage8_mutation_union_structure_shape_mismatch")
        allowed |= structure_support > 0.0
        support_sources.append("target_structure_weight")

    if bool(subject_chroma_report.get("accepted", False)):
        subject_support = np.asarray(
            masks.get("subject_mask"),
            dtype=np.float32,
        )
        if subject_support.shape != spatial_shape or not np.all(
            np.isfinite(subject_support)
        ):
            raise ValueError("stage8_mutation_union_subject_support_invalid")
        # The color transaction independently proves exact identity outside
        # this frozen ROI.  Use the whole frozen ROI here rather than the
        # narrower structure mask; candidate pixels never define permission.
        allowed |= subject_support > 1.0e-6
        support_sources.append("frozen_subject_chroma_roi")

    if bool(palette_report.get("accepted", False)):
        core = np.asarray(masks.get("core_mask"), dtype=np.float32)
        nebula = np.asarray(masks.get("nebula_mask"), dtype=np.float32)
        faint = np.asarray(masks.get("faint_nebula_mask"), dtype=np.float32)
        background = np.asarray(
            masks.get("background_mask"),
            dtype=np.float32,
        )
        stars = np.asarray(masks.get("star_mask"), dtype=np.float32)
        halo = np.asarray(
            masks.get("star_halo_guard_mask"),
            dtype=np.float32,
        )
        if any(
            value.shape != spatial_shape
            for value in (core, nebula, faint, background, stars, halo)
        ):
            raise ValueError("stage8_mutation_union_palette_support_invalid")
        palette_strength = float(
            np.clip(
                float(
                    getattr(
                        pipeline.cfg,
                        "stage8_dualband_palette_strength",
                        0.85,
                    )
                ),
                0.10,
                1.0,
            )
        )
        palette_weight = np.clip(
            np.maximum(nebula, faint)
            * (1.0 - background)
            * (1.0 - np.maximum.reduce((core, stars, halo))),
            0.0,
            1.0,
        ) * palette_strength
        palette_weight[background >= 0.80] = 0.0
        palette_weight[palette_weight <= 0.01] = 0.0
        allowed |= palette_weight > 0.0
        support_sources.append("dualband_palette_subject_roi")

    identity_masks = {
        "enhancement_support_weight": allowed.astype(np.float32),
        "mask_route": "accepted_mutation_roi_union_v1",
    }
    report = stage8_outside_target_identity_report(
        baseline_array,
        candidate_array,
        identity_masks,
        target_type=target_type,
    )
    report.update(
        operation_support_contract=(
            "candidate_independent_frozen_roi_union_v1"
        ),
        support_sources=support_sources,
        frozen_mask_keys=frozen_keys,
        structure_included=bool(include_structure),
        structure_mask_report=structure_mask_report,
        mutation_support_pixel_count=int(np.count_nonzero(allowed)),
        mutation_support_coverage=float(np.mean(allowed)),
        mutation_support_sha256=pixel_sha256(
            np.ascontiguousarray(allowed.astype(np.uint8))
        ),
    )
    if not support_sources and np.array_equal(baseline_array, candidate_array):
        report["reason"] = "exact_verified_noop_identity"
    return report


def _stage8_quality_with_mutation_union(
    pipeline,
    *,
    baseline_stem: str,
    candidate_stem: str,
    subject_chroma_report: Dict[str, Any],
    palette_report: Dict[str, Any],
    include_structure: bool,
) -> Dict[str, Any]:
    """Rebind only the outside-pixel gate to the accepted mutation ledger."""

    try:
        quality = dict(
            pipeline._stage8_quality_assessment(
                baseline_stem=baseline_stem,
                candidate_stem=candidate_stem,
            )
        )
    except TypeError:
        # Small test doubles and historical runtime shims exposed the older
        # no-keyword form.  The production method accepts the explicit stems.
        quality = dict(pipeline._stage8_quality_assessment())
    baseline = pipeline._read_image_by_stem(baseline_stem)
    candidate = pipeline._read_image_by_stem(candidate_stem)
    if baseline is None or candidate is None:
        return quality
    outside_identity = _stage8_mutation_union_outside_identity(
        pipeline,
        np.asarray(baseline),
        np.asarray(candidate),
        subject_chroma_report=subject_chroma_report,
        palette_report=palette_report,
        include_structure=include_structure,
    )
    outside_issue_prefix = "outside_target_pixel_identity_gate_failed="
    issues = [
        str(issue)
        for issue in list(quality.get("issues") or [])
        if not str(issue).startswith(outside_issue_prefix)
    ]
    if not bool(outside_identity.get("accepted", False)):
        issues.append(
            outside_issue_prefix
            + str(outside_identity.get("reason") or "unknown")
        )
    quality["outside_target_identity"] = outside_identity
    quality["issues"] = issues
    quality["local_issues"] = list(issues)
    quality_gates = dict(quality.get("quality_gates") or {})
    if bool(outside_identity.get("applicable", False)):
        outside_accepted = bool(outside_identity.get("accepted", False))
        quality_gates["outside_target_pixel_identity"] = {
            "value": bool(
                outside_identity.get("exact_pixel_identity", False)
            ),
            "accepted_limit": True,
            "accepted": outside_accepted,
            "hard_failed": not outside_accepted,
            "advisory": False,
        }
    quality["quality_gates"] = quality_gates
    quality["status"] = "ok" if not issues else "poor"
    quality["mutation_union_revalidation"] = {
        "schema": "starun.stage8-mutation-union-revalidation.v1",
        "status": "accepted" if not issues else "rejected",
        "accepted": not issues,
        "include_structure": bool(include_structure),
        "support_sources": list(outside_identity.get("support_sources") or []),
        "support_sha256": outside_identity.get("mutation_support_sha256"),
    }
    return quality


def _stage8_nonmutating_color_terminal(
    report: Dict[str, Any],
    *,
    expected_schema: str,
    skipped_status: str,
    rejected_status: str,
    require_canonical_restore: bool,
) -> Dict[str, Any]:
    """Verify that a Stage8 color transaction reached a safe no-op terminal."""

    value = report if isinstance(report, dict) else {}
    status = str(value.get("status") or "")
    transaction = value.get("transaction")
    transaction = transaction if isinstance(transaction, dict) else {}
    eligibility = value.get("eligibility")
    eligibility = eligibility if isinstance(eligibility, dict) else {}
    common = bool(
        value.get("schema") == expected_schema
        and value.get("accepted") is False
        and value.get("feeds_main_pipeline") is False
    )
    skipped = bool(
        common
        and status == skipped_status
        and eligibility.get("eligible") is False
        and transaction.get("baseline_saved") is False
        and transaction.get("candidate_saved") is False
        and transaction.get("rollback_performed") is False
    )
    rejected = bool(
        common
        and status == rejected_status
        and transaction.get("baseline_saved") is True
        and transaction.get("candidate_saved") is False
        and transaction.get("rollback_performed") is True
        and transaction.get("rollback_ok") is True
        and (
            not require_canonical_restore
            or transaction.get("canonical_saved") is True
        )
    )
    return {
        "schema": "starun.stage8-nonmutating-color-terminal.v1",
        "status": "verified" if skipped or rejected else "unverified",
        "accepted": bool(skipped or rejected),
        "report_schema": value.get("schema"),
        "report_status": status or None,
        "terminal_mode": (
            "ineligible_without_mutation"
            if skipped
            else "quality_rejected_and_rolled_back"
            if rejected
            else None
        ),
    }


def _stage8_safe_passthrough_final_validation(
    pipeline,
    *,
    subject_chroma_report: Dict[str, Any],
    palette_report: Dict[str, Any],
) -> Dict[str, Any]:
    """Re-run every independent gate on the color-only output."""

    preflight = dict(
        getattr(
            pipeline,
            "_stage8_safe_passthrough_color_only_preflight",
            {},
        )
        or {}
    )
    report: Dict[str, Any] = {
        "schema": "starun.stage8-safe-passthrough-final.v1",
        "status": "rejected",
        "accepted": False,
        "route": "safe_passthrough_color_only",
        "preflight": preflight,
        "checks": {},
        "issues": [],
    }
    try:
        baseline = pipeline._read_image_by_stem("stage8_input_starless")
        candidate = pipeline._read_image_by_stem("stage8_enhanced")
        if baseline is None or candidate is None:
            raise ValueError("safe_passthrough_final_pixels_unavailable")
        baseline_array = np.asarray(baseline)
        candidate_array = np.asarray(candidate)
        color_operation_accepted = bool(
            subject_chroma_report.get("accepted", False)
            or palette_report.get("accepted", False)
        )
        exact_color_identity = bool(
            baseline_array.shape == candidate_array.shape
            and np.array_equal(baseline_array, candidate_array)
        )
        subject_noop = _stage8_nonmutating_color_terminal(
            subject_chroma_report,
            expected_schema=STAGE8_SUBJECT_CHROMA_SCHEMA,
            skipped_status="skipped_ineligible",
            rejected_status="rejected_by_quality_gate",
            require_canonical_restore=True,
        )
        palette_noop = _stage8_nonmutating_color_terminal(
            palette_report,
            expected_schema=DUALBAND_PALETTE_SCHEMA,
            skipped_status="skipped_ineligible",
            rejected_status="rejected_by_palette_quality_gate",
            require_canonical_restore=False,
        )
        limited_verified_noop = bool(
            preflight.get("source_mode") == "limited_safe_passthrough"
            and preflight.get("accepted") is True
            and exact_color_identity
        )
        post_cumulative_verified_noop = bool(
            preflight.get("source_mode")
            == "post_cumulative_structure_rollback"
            and preflight.get("accepted") is True
            and exact_color_identity
            and subject_noop.get("accepted") is True
            and palette_noop.get("accepted") is True
        )
        color_accepted = bool(
            color_operation_accepted
            or limited_verified_noop
            or post_cumulative_verified_noop
        )
        quality = _stage8_quality_with_mutation_union(
            pipeline,
            baseline_stem="stage8_input_starless",
            candidate_stem="stage8_enhanced",
            subject_chroma_report=subject_chroma_report,
            palette_report=palette_report,
            include_structure=False,
        )
        outside_target_identity = dict(
            quality.get("outside_target_identity") or {}
        )
        spatial = spatial_background_lineage.assess_final_spatial_background(
            getattr(pipeline, "process_dir", None),
            candidate_array,
        )
        masks = pipeline._stage8_generate_starless_masks(baseline_array)
        halo = star_halo_guard.assess_candidate(
            baseline_array,
            candidate_array,
            masks.get("star_halo_guard_mask"),
            mode="color",
        )
        finite_and_shape = bool(
            baseline_array.shape == candidate_array.shape
            and np.all(np.isfinite(candidate_array))
        )
        report["checks"] = {
            "preflight": bool(preflight.get("accepted", False)),
            "color": {
                "accepted": color_accepted,
                "mode": (
                    "bounded_color_operation"
                    if color_operation_accepted
                    else "verified_pixel_identity"
                    if limited_verified_noop
                    else "verified_color_noop_after_structure_rollback"
                    if post_cumulative_verified_noop
                    else "unverified"
                ),
                "exact_pixel_identity": exact_color_identity,
                "subject_chroma": bool(
                    subject_chroma_report.get("accepted", False)
                ),
                "palette": bool(palette_report.get("accepted", False)),
                "nonmutating_terminal_evidence": {
                    "subject_chroma": subject_noop,
                    "palette": palette_noop,
                },
            },
            "background_seam_clip_presentation": quality,
            "outside_target_pixel_identity": outside_target_identity,
            "spatial_background": spatial,
            "star_halo": halo,
            "artifact": {
                "accepted": finite_and_shape,
                "identity": _stage8_source_identity(
                    pipeline, "stage8_enhanced"
                ),
            },
        }
        checks = {
            "preflight": bool(preflight.get("accepted", False)),
            "color": color_accepted,
            "background_seam_clip_presentation": quality.get("status") == "ok",
            "outside_target_pixel_identity": bool(
                outside_target_identity.get("accepted", False)
            ),
            "spatial_background": bool(spatial.get("accepted", False)),
            "star_halo": bool(halo.get("accepted", False)),
            "artifact": finite_and_shape,
        }
        report["issues"] = [name for name, accepted in checks.items() if not accepted]
        report["accepted"] = not report["issues"]
        report["status"] = "accepted" if report["accepted"] else "rejected"
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as error:
        report["issues"] = [str(error)]
    pipeline._stage8_safe_passthrough_color_only_final = dict(report)
    pipeline._write_stage_json("stage8_safe_passthrough_final.json", report)
    return report


def _stage8_final_cumulative_validation(
    pipeline,
    *,
    subject_chroma_report: Dict[str, Any],
    palette_report: Dict[str, Any],
    starless_finish_report: Dict[str, Any],
) -> Dict[str, Any]:
    """Re-run every Stage8 gate on the final persisted pixel accumulation.

    Local acceptance of structure, Starless finish, chroma, and palette
    transactions is intentionally insufficient here.  Every measurement is
    recomputed from the immutable Stage8 input to the final canonical FITS so
    two individually safe deltas cannot cumulatively cross a hard limit.
    """

    report: Dict[str, Any] = {
        "schema": "starun.stage8-final-cumulative-quality.v1",
        "status": "rejected",
        "accepted": False,
        "evaluation_scope": (
            "stage8_input_starless_to_final_persisted_pixels"
        ),
        "fresh_evaluation": True,
        "baseline": "stage8_input_starless.fit",
        "candidate": "stage8_enhanced.fit",
        "checks": {},
        "issues": [],
        "mutation_ledger": {
            "starless_finish": {
                "status": starless_finish_report.get("status"),
                "accepted": bool(starless_finish_report.get("accepted", False)),
                "accepted_steps": list(
                    starless_finish_report.get("accepted_steps") or []
                ),
            },
            "subject_chroma": {
                "status": subject_chroma_report.get("status"),
                "accepted": bool(subject_chroma_report.get("accepted", False)),
            },
            "palette": {
                "status": palette_report.get("status"),
                "accepted": bool(palette_report.get("accepted", False)),
                "palette": palette_report.get("palette"),
            },
        },
    }
    try:
        baseline = pipeline._read_image_by_stem("stage8_input_starless")
        candidate = pipeline._read_image_by_stem("stage8_enhanced")
        if baseline is None or candidate is None:
            raise ValueError("stage8_final_cumulative_pixels_unavailable")
        baseline_array = np.asarray(baseline)
        candidate_array = np.asarray(candidate)
        shape_dtype_finite = bool(
            baseline_array.shape == candidate_array.shape
            and baseline_array.dtype == candidate_array.dtype
            and np.all(np.isfinite(baseline_array))
            and np.all(np.isfinite(candidate_array))
        )
        baseline_identity = _stage8_source_identity(
            pipeline,
            "stage8_input_starless",
        )
        candidate_identity = _stage8_source_identity(
            pipeline,
            "stage8_enhanced",
        )
        artifact_accepted = bool(
            shape_dtype_finite
            and baseline_identity.get("identity_status") == "verified"
            and candidate_identity.get("identity_status") == "verified"
            and candidate_identity.get("sha256")
            and candidate_identity.get("fits_data_sha256")
            and candidate_identity.get("decoded_pixel_sha256")
        )
        artifact = {
            "accepted": artifact_accepted,
            "shape_dtype_finite": shape_dtype_finite,
            "baseline": baseline_identity,
            "candidate": candidate_identity,
        }

        # Do not reuse the pre-finish structure report.  The production
        # assessment rebuilds the target mask from the immutable input and
        # compares it with the final persisted candidate.
        safe_preflight = dict(
            getattr(
                pipeline,
                "_stage8_safe_passthrough_color_only_preflight",
                {},
            )
            or {}
        )
        safe_color_only = bool(
            safe_preflight.get("accepted", False)
            and str(safe_preflight.get("source_mode") or "")
            in {
                "structure_rollback",
                "limited_safe_passthrough",
                "post_cumulative_structure_rollback",
            }
        )
        quality = _stage8_quality_with_mutation_union(
            pipeline,
            baseline_stem="stage8_input_starless",
            candidate_stem="stage8_enhanced",
            subject_chroma_report=subject_chroma_report,
            palette_report=palette_report,
            include_structure=not safe_color_only,
        )
        seam = dict(quality.get("subject_boundary_seam") or {})
        frozen_noise = dict(quality.get("frozen_sky_visible_noise") or {})
        outside_identity = dict(quality.get("outside_target_identity") or {})
        quality_accepted = bool(
            str(quality.get("status") or "") == "ok"
            and not list(quality.get("issues") or [])
        )
        seam_accepted = bool(
            seam.get("accepted", False)
            and str(seam.get("status") or "") != "hard_failed"
            and (
                seam.get("available", False)
                or (
                    seam.get("applicable") is False
                    and str(seam.get("status") or "") == "not_applicable"
                )
            )
        )
        frozen_noise_accepted = bool(
            frozen_noise.get("available", False)
            and frozen_noise.get("accepted", False)
        )
        outside_identity_accepted = bool(
            outside_identity.get("available", False)
            and outside_identity.get("accepted", False)
        )

        spatial = spatial_background_lineage.assess_final_spatial_background(
            getattr(pipeline, "process_dir", None),
            candidate_array,
        )
        masks = pipeline._stage8_generate_starless_masks(baseline_array)
        if not isinstance(masks, dict):
            raise ValueError("stage8_final_cumulative_masks_unavailable")
        halo = star_halo_guard.assess_candidate(
            baseline_array,
            candidate_array,
            masks.get("star_halo_guard_mask"),
            mode="color",
        )

        channel_profile = getattr(pipeline, "channel_profile", {}) or {}
        if not isinstance(channel_profile, dict) or not channel_profile:
            channel_profile = {
                "kind": str(
                    getattr(pipeline, "_channel_semantics", "unknown")
                    or "unknown"
                )
            }
        contract = resolve_color_contract(
            channel_profile=channel_profile,
            color_report=getattr(pipeline, "color_calibration_report", {}) or {},
            palette_report=palette_report,
        )
        color_presentation = build_color_quality_report(
            baseline_array,
            candidate_array,
            stage="stage8_final_cumulative_preflight",
            baseline_name="stage8_input_starless.fit",
            candidate_name="stage8_enhanced.fit",
            contract=contract,
            masks=masks,
            operation="final_cumulative_color_presentation_preflight",
        )
        color_presentation["mode"] = "final_cumulative_preflight"
        color_presentation["used_for_gate"] = True
        color_presentation["accepted"] = bool(
            str(color_presentation.get("status") or "") == "reported"
            and not list(color_presentation.get("issues") or [])
        )

        report["checks"] = {
            "artifact": artifact,
            "subject_boundary_seam": seam,
            "frozen_sky_visible_noise": frozen_noise,
            "outside_target_pixel_identity": outside_identity,
            "background_clipping_contrast_presentation": quality,
            "spatial_background": spatial,
            "star_halo": halo,
            "color_presentation_preflight": color_presentation,
        }
        checks = {
            "artifact": artifact_accepted,
            "subject_boundary_seam": seam_accepted,
            "frozen_sky_visible_noise": frozen_noise_accepted,
            "outside_target_pixel_identity": outside_identity_accepted,
            "background_clipping_contrast_presentation": quality_accepted,
            "spatial_background": bool(spatial.get("accepted", False)),
            "star_halo": bool(halo.get("accepted", False)),
            "color_presentation_preflight": bool(
                color_presentation.get("accepted", False)
            ),
        }
        report["issues"] = [
            name for name, accepted in checks.items() if not accepted
        ]
        report["accepted"] = not report["issues"]
        report["status"] = "accepted" if report["accepted"] else "rejected"
    except (
        AttributeError,
        FloatingPointError,
        IndexError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        report["issues"] = [str(error)]
    return report


def _stage8_exact_final_cumulative_rollback(pipeline) -> Dict[str, Any]:
    """Restore and independently prove Stage8's immutable input identity."""

    report: Dict[str, Any] = {
        "target": "stage8_input_starless",
        "canonical": "stage8_enhanced",
        "status": "failed",
        "accepted": False,
        "attempted": True,
    }
    try:
        rollback = getattr(pipeline, "_rollback_stage8_to_input", None)
        if callable(rollback):
            rollback_invoked = bool(rollback())
        else:
            pipeline.cmd_with_check("load", "stage8_input_starless")
            rollback_invoked = bool(
                pipeline._save_stage_output("stage8_enhanced")
            )
        baseline = pipeline._read_image_by_stem("stage8_input_starless")
        canonical = pipeline._read_image_by_stem("stage8_enhanced")
        if baseline is None or canonical is None:
            raise ValueError("stage8_final_cumulative_rollback_pixels_unavailable")
        baseline_array = np.asarray(baseline)
        canonical_array = np.asarray(canonical)
        baseline_identity = _stage8_source_identity(
            pipeline,
            "stage8_input_starless",
        )
        canonical_identity = _stage8_source_identity(
            pipeline,
            "stage8_enhanced",
        )
        exact_pixels = bool(
            baseline_array.shape == canonical_array.shape
            and baseline_array.dtype == canonical_array.dtype
            and np.array_equal(baseline_array, canonical_array)
            and pixel_sha256(baseline_array) == pixel_sha256(canonical_array)
        )
        exact_persisted_identity = bool(
            baseline_identity.get("fits_data_sha256")
            and baseline_identity.get("fits_data_sha256")
            == canonical_identity.get("fits_data_sha256")
            and baseline_identity.get("decoded_pixel_sha256")
            and baseline_identity.get("decoded_pixel_sha256")
            == canonical_identity.get("decoded_pixel_sha256")
        )
        accepted = bool(
            rollback_invoked and exact_pixels and exact_persisted_identity
        )
        report.update(
            status="restored" if accepted else "failed",
            accepted=accepted,
            rollback_invoked=rollback_invoked,
            exact_pixels=exact_pixels,
            exact_persisted_identity=exact_persisted_identity,
            baseline=baseline_identity,
            canonical=canonical_identity,
        )
    except (
        AttributeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        report["error"] = str(error)
    return report


def _stage8_enforce_final_cumulative_validation(
    pipeline,
    *,
    subject_chroma_report: Dict[str, Any],
    palette_report: Dict[str, Any],
    starless_finish_report: Dict[str, Any],
    defer_review_on_exact_rollback: bool = False,
) -> Dict[str, Any]:
    """Persist the final cumulative gate and fail closed on rejection."""

    report = _stage8_final_cumulative_validation(
        pipeline,
        subject_chroma_report=subject_chroma_report,
        palette_report=palette_report,
        starless_finish_report=starless_finish_report,
    )
    if not bool(report.get("accepted", False)):
        report["rejected_candidate_identity"] = _stage8_source_identity(
            pipeline,
            "stage8_enhanced",
        )
        rollback = _stage8_exact_final_cumulative_rollback(pipeline)
        report["rollback"] = rollback
        if bool(rollback.get("accepted", False)):
            pipeline._stage8_final_source = "stage8_input_starless"
            pipeline._stage8_final_quality = "final_cumulative_qa_rejected"
            reason_code = "stage8_final_cumulative_qa_rejected"
        else:
            pipeline._stage8_final_quality = (
                "final_cumulative_qa_rollback_failed"
            )
            pipeline._stage8_fallback_used = True
            reason_code = "stage8_final_cumulative_qa_rollback_failed"
        pipeline._stage8_fallback_used = True
        report["reason_code"] = reason_code
        review_deferred = bool(
            defer_review_on_exact_rollback
            and rollback.get("accepted", False)
        )
        report["review_deferred_for_safe_passthrough"] = review_deferred
        if not review_deferred and hasattr(pipeline, "_require_review"):
            pipeline._require_review(8, reason_code)
    else:
        report["reason_code"] = "accepted"
    pipeline._stage8_final_cumulative_quality_report = dict(report)
    pipeline._write_stage_json(
        "stage8_final_cumulative_quality.json",
        report,
    )
    return report


def _load_stage8_input(
    pipeline,
    messages: List[str],
    *,
    explicit_source: Optional[str] = None,
) -> str:
    """Load this run's accepted Stage 7 output, then conservative fallbacks."""
    stage7_preferred = str(
        getattr(pipeline, "_stage7_stretch_output", None)
        or getattr(pipeline, "stretched_name", None)
        or "stage7_stretched"
    )
    preferred = str(explicit_source or stage7_preferred)
    stage7_accepted = bool(getattr(pipeline, "_stage7_stretch_accepted", False))
    candidates: List[str] = []
    if explicit_source:
        candidates.append(str(explicit_source))
        messages.append(f"stage8 explicit external input requested: {explicit_source}")

    if stage7_accepted:
        preferred_path = pipeline.process_dir / f"{stage7_preferred}.fit"
        if preferred_path.exists():
            if stage7_preferred not in candidates:
                candidates.append(stage7_preferred)
        else:
            messages.append(
                "stage8 preferred Stage7 input missing: "
                f"{stage7_preferred}.fit; using fallback"
            )
    else:
        messages.append("stage8 Stage7 output not accepted; using linear starless fallback")

    for fallback in ("starless", "stage6_starless"):
        if fallback not in candidates:
            candidates.append(fallback)

    last_error: Optional[Exception] = None
    for source_stem in candidates:
        try:
            pipeline.cmd_with_check("load", source_stem)
            pipeline._stage8_input_source = source_stem
            pipeline._stage8_input_fallback_used = source_stem != preferred
            pipeline.starless_file = pipeline.process_dir / f"{source_stem}.fit"
            messages.append(f"stage8_input_source={source_stem}")
            pipeline.log.info(f"Stage8 输入源: {source_stem}")
            return source_stem
        except (CommandError, SirilError) as error:
            last_error = error
            messages.append(
                f"stage8 input load failed: {source_stem}: "
                f"{pipeline._short_text(error, 160)}"
            )

    if last_error is not None:
        raise last_error
    raise RuntimeError("Stage8 has no usable input source")


def _write_stage8_color_quality_report(
    pipeline,
    *,
    final_source: str,
    requested_saturation: float,
    effective_saturation: float,
    applied_saturation: float,
    palette_report: Optional[Dict[str, Any]] = None,
    subject_chroma_report: Optional[Dict[str, Any]] = None,
    vectra_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write the final Stage8 color and star-halo gating ledger entry."""
    channel_profile = getattr(pipeline, "channel_profile", {}) or {}
    if not isinstance(channel_profile, dict) or not channel_profile:
        channel_profile = {
            "kind": str(
                getattr(pipeline, "_channel_semantics", "unknown") or "unknown"
            )
        }
    contract = resolve_color_contract(
        channel_profile=channel_profile,
        color_report=getattr(pipeline, "color_calibration_report", {}) or {},
        palette_report=palette_report or {},
    )
    def read_checkpoint(stem: str):
        reader = getattr(pipeline, "_read_image_by_stem", None)
        if callable(reader):
            try:
                return reader(stem)
            except (OSError, RuntimeError, TypeError, ValueError):
                return None
        # Lightweight unit/integration harnesses keep checkpoints in memory.
        saved = getattr(pipeline, "saved_image_pixels", None)
        if isinstance(saved, dict) and stem in saved:
            return np.array(saved[stem], copy=True)
        return None

    baseline = read_checkpoint("stage8_input_starless")
    candidate = read_checkpoint(final_source)
    if baseline is None or candidate is None:
        report = {
            "schema": "starun.color-quality-report.v1",
            "stage": "stage8",
            "status": "unavailable",
            "mode": "transactional_gate",
            "used_for_gate": True,
            "baseline": "stage8_input_starless.fit",
            "candidate": f"{final_source}.fit",
            "contract": contract,
            "issues": [
                "stage8 color report baseline or candidate is unavailable"
            ],
        }
    else:
        try:
            mask_builder = getattr(pipeline, "_stage8_generate_starless_masks", None)
            masks = mask_builder(baseline) if callable(mask_builder) else None
        except (OSError, RuntimeError, TypeError, ValueError):
            masks = None
        saturation_execution = dict(
            getattr(pipeline, "_stage8_saturation_execution", {}) or {}
        )
        report = build_color_quality_report(
            baseline,
            candidate,
            stage="stage8",
            baseline_name="stage8_input_starless.fit",
            candidate_name=f"{final_source}.fit",
            contract=contract,
            masks=masks,
            requested_saturation=requested_saturation,
            effective_saturation=effective_saturation,
            applied_saturation=applied_saturation,
            operation=(
                "artistic_dualband_palette"
                if bool((palette_report or {}).get("accepted", False))
                else "vectra_exclusive_color_route"
                if bool((vectra_report or {}).get("accepted", False))
                else "masked_subject_chroma_rendition"
                if bool((subject_chroma_report or {}).get("accepted", False))
                else "masked_single_chroma_recovery"
                if saturation_execution.get("applied", False)
                else "structure_only_or_passthrough"
            ),
        )
        report["saturation_execution"] = saturation_execution
        halo_gate = star_halo_guard.assess_candidate(
            baseline,
            candidate,
            (masks or {}).get("star_halo_guard_mask")
            if isinstance(masks, dict)
            else None,
            mode="color",
        )
        report["mode"] = "transactional_gate"
        report["used_for_gate"] = True
        report["star_halo_local_gate"] = halo_gate
        if not bool(halo_gate.get("accepted", False)):
            report["status"] = "rejected"
            report.setdefault("issues", []).extend(
                list(halo_gate.get("issues") or [])
            )
    guard_report = dict(
        getattr(pipeline, "_stage8_star_halo_guard_report", {}) or {}
    )
    guard_metrics = (
        dict(guard_report.get("metrics") or {})
        if isinstance(guard_report.get("metrics"), dict)
        else {}
    )
    guard_components = list(guard_metrics.get("components") or [])
    hard_components = [
        dict(component)
        for component in guard_components
        if isinstance(component, dict) and bool(component.get("hard_anomaly", False))
    ]
    report["guard_lineage"] = {
        "verified": bool(
            getattr(pipeline, "_stage8_star_halo_guard_verified", False)
        ),
        "schema": guard_report.get("schema"),
        "run_id": guard_report.get("run_id"),
        "status": guard_report.get("status"),
        "reason_code": guard_report.get("reason_code"),
        "artifact": guard_report.get("artifact"),
        "artifact_sha256": guard_report.get("artifact_sha256"),
        "source": dict(guard_report.get("source") or {}),
    }
    report["component_anomalies"] = {
        "component_count": int(guard_metrics.get("component_count", 0) or 0),
        "hard_anomaly_count": int(
            guard_metrics.get("hard_anomaly_count", len(hard_components)) or 0
        ),
        "components": hard_components,
    }
    finish_report = dict(
        getattr(pipeline, "_stage8_starless_finish_report", {}) or {}
    )
    finish_retries = []
    for step in list(finish_report.get("steps") or []):
        if not isinstance(step, dict):
            continue
        finish_retries.append(
            {
                "name": step.get("name") or step.get("step"),
                "status": step.get("status"),
                "weakened_retry": dict(step.get("weakened_retry") or {}),
            }
        )
    palette_retries = []
    for palette_candidate in list((palette_report or {}).get("candidates") or []):
        if not isinstance(palette_candidate, dict):
            continue
        palette_retries.append(
            {
                "palette": palette_candidate.get("palette"),
                "status": palette_candidate.get("status"),
                "weakened_retry": dict(
                    palette_candidate.get("weakened_retry") or {}
                ),
            }
        )
    report["weakened_retry"] = {
        "starless_finish_steps": finish_retries,
        "subject_chroma": dict(
            (subject_chroma_report or {}).get("weakened_retry") or {}
        ),
        "vectra": dict((vectra_report or {}).get("weakened_retry") or {}),
        "palette": dict((palette_report or {}).get("weakened_retry") or {}),
        "palette_candidates": palette_retries,
    }
    report["final_pixel_identity"] = _stage8_source_identity(
        pipeline,
        final_source,
    )
    ledger = list(getattr(pipeline, "_color_adjustment_ledger", []) or [])
    entry = report.get("ledger_entry")
    if isinstance(entry, dict):
        ledger.append(dict(entry))
    pipeline._color_adjustment_ledger = ledger
    report["cross_stage_ledger"] = list(ledger)
    pipeline._stage8_color_quality_report = dict(report)
    pipeline._write_stage_json("stage8_color_quality_report.json", report)
    handoff = dict(getattr(pipeline, "_stage8_handoff", {}) or {})
    if handoff:
        handoff["color_contract"] = report.get("contract")
        handoff["color_quality_report"] = "stage8_color_quality_report.json"
        handoff["color_gate"] = {
            "report": "stage8_color_quality_report.json",
            "used_for_gate": bool(report.get("used_for_gate", False)),
            "status": report.get("status"),
            "guard_lineage_verified": bool(
                (report.get("guard_lineage") or {}).get("verified", False)
            ),
            "final_pixel_identity": dict(
                report.get("final_pixel_identity") or {}
            ),
        }
        _write_stage8_handoff(pipeline, handoff)
    return report


def _stage8_set_image_pixels(pipeline, pixels, *, label: str) -> None:
    setter = getattr(pipeline, "_set_current_image_pixeldata", None)
    if callable(setter):
        setter(pixels, label=label)
        return
    lock_factory = getattr(pipeline.siril, "image_lock", None)
    if callable(lock_factory):
        with lock_factory():
            pipeline.siril.set_image_pixeldata(pixels)
        return
    pipeline.siril.set_image_pixeldata(pixels)


def _stage8_write_starless_finish_report(
    pipeline,
    *,
    status: str,
    reason_code: str,
    **updates: Any,
) -> Dict[str, Any]:
    """Persist the Stage8-owned post-structure Starless finish decision."""

    report: Dict[str, Any] = {
        "schema": STAGE8_STARLESS_FINISH_SCHEMA,
        "status": str(status),
        "accepted": False,
        "reason_code": str(reason_code),
        "source": None,
        "final_source": None,
        "steps": [],
        "color_route": {"selected": "none", "reason": reason_code},
    }
    previous = getattr(pipeline, "_stage8_starless_finish_report", None)
    if isinstance(previous, dict):
        report.update(previous)
    report.update(updates)
    pipeline._stage8_starless_finish_report = dict(report)
    pipeline._write_stage_json("stage8_starless_finish_report.json", report)
    return report


def _stage8_finish_masks(pipeline, image_data: np.ndarray) -> Dict[str, Any]:
    masks = pipeline._stage8_generate_starless_masks(image_data)
    if not isinstance(masks, dict):
        raise ValueError("Stage8 finish masks are unavailable")
    target_type = (
        str(pipeline._active_target_type() or "").strip().lower()
        if hasattr(pipeline, "_active_target_type")
        else ""
    )
    if target_type == "large_galaxy":
        masks, disk_report = stage8_large_galaxy_structure_masks(
            pipeline,
            masks,
        )
        if not bool(disk_report.get("available", False)):
            raise ValueError(
                "large_galaxy_structure_mask_unavailable:"
                + str(disk_report.get("reason") or "unknown")
            )
        masks["galaxy_signal_mask"] = np.asarray(
            masks.get("nebula_mask"),
            dtype=np.float32,
        )
    return masks


def _stage8_restore_finish_step(pipeline, baseline_stem: str) -> tuple[bool, Optional[str]]:
    try:
        pipeline.cmd_with_check("load", baseline_stem)
        return True, None
    except (CommandError, SirilError) as error:
        return False, str(error)


def _stage8_run_starless_finish_step(
    pipeline,
    messages: List[str],
    *,
    step_id: str,
    step_key: str,
    command_candidates: List[tuple[str, tuple[str, ...]]],
    base_stem: str,
    mode: str,
    effective_saturation_budget: float = 0.0,
) -> Dict[str, Any]:
    """Run, project, gate, persist and exactly roll back one plugin step."""

    pre_stem = f"stage8_pre_{step_id}"
    candidate_stem = f"stage8_{step_id}_candidate"
    step: Dict[str, Any] = {
        "id": step_id,
        "mode": mode,
        "status": "skipped_unavailable",
        "accepted": False,
        "implementation": None,
        "input": f"{base_stem}.fit",
        "baseline": f"{pre_stem}.fit",
        "candidate": f"{candidate_stem}.fit",
        "output": None,
        "transaction": {
            "baseline_saved": False,
            "candidate_saved": False,
            "canonical_saved": False,
            "rollback_performed": False,
            "rollback_ok": None,
        },
    }
    try:
        pipeline.cmd_with_check("load", base_stem)
        if not pipeline._save_stage_output(pre_stem):
            step.update(
                status="prohibited_baseline_save_failed",
                reason_code="stage8_finish_baseline_save_failed",
            )
            return step
        step["transaction"]["baseline_saved"] = True
        baseline = pipeline.siril.get_image_pixeldata(preview=False)
        if baseline is None:
            raise RuntimeError("Stage8 finish baseline pixels are unavailable")
        baseline = np.asarray(baseline, dtype=np.float32)
        implementation = pipeline._run_first_available_command(
            step_key,
            command_candidates,
        )
        step["implementation"] = implementation
        if not implementation:
            _stage8_restore_finish_step(pipeline, pre_stem)
            step["reason_code"] = "plugin_command_unavailable"
            return step
        plugin_output = pipeline.siril.get_image_pixeldata(preview=False)
        if plugin_output is None:
            raise RuntimeError("Stage8 finish plugin pixels are unavailable")
        masks = _stage8_finish_masks(pipeline, baseline)
        if mode == "color":
            candidate, projection = project_luminance_locked_color_candidate(
                baseline,
                plugin_output,
                masks,
                effective_saturation_budget=effective_saturation_budget,
            )
        else:
            candidate, projection = project_linked_luminance_candidate(
                baseline,
                plugin_output,
                masks,
            )
        gate = assess_finish_candidate(
            baseline,
            candidate,
            masks,
            mode=mode,
            highlight_clip_ratio_max=float(
                getattr(pipeline.cfg, "stage8_highlight_clip_ratio_max", 0.012)
            ),
            texture_growth_max=float(
                getattr(pipeline.cfg, "stage8_texture_artifact_growth_max", 1.25)
            ),
            effective_saturation_budget=effective_saturation_budget,
        )
        halo_gate = star_halo_guard.assess_candidate(
            baseline,
            candidate,
            masks.get("star_halo_guard_mask"),
            mode="color" if mode == "color" else "luminance",
        )
        if not bool(halo_gate.get("accepted", False)):
            gate["accepted"] = False
            gate["status"] = "rejected"
            gate.setdefault("issues", []).extend(
                list(halo_gate.get("issues") or [])
            )
        step["star_halo_local_gate"] = halo_gate
        seam = stage8_subject_boundary_seam_report(
            pipeline,
            baseline,
            candidate,
        )
        seam_failed = bool(
            seam.get("applicable", False)
            and (
                not bool(seam.get("available", False))
                or str(seam.get("status") or "") == "hard_failed"
            )
        )
        if seam_failed:
            gate["accepted"] = False
            gate["status"] = "rejected"
            gate.setdefault("issues", []).append(
                "subject_boundary_gate_failed"
            )
        if (
            not bool(gate.get("accepted", False))
            and int(getattr(pipeline.cfg, "stage8_quality_retry_max", 1) or 0)
            > 0
        ):
            reduced_candidate, boundary_retry = (
                stage8_subject_boundary_retry_candidate(
                    pipeline,
                    baseline,
                    candidate,
                    seam_report=seam,
                )
            )
            reduced_gate = assess_finish_candidate(
                baseline,
                reduced_candidate,
                masks,
                mode=mode,
                highlight_clip_ratio_max=float(
                    getattr(pipeline.cfg, "stage8_highlight_clip_ratio_max", 0.012)
                ),
                texture_growth_max=float(
                    getattr(
                        pipeline.cfg,
                        "stage8_texture_artifact_growth_max",
                        1.25,
                    )
                ),
                effective_saturation_budget=effective_saturation_budget,
            )
            reduced_halo_gate = star_halo_guard.assess_candidate(
                baseline,
                reduced_candidate,
                masks.get("star_halo_guard_mask"),
                mode="color" if mode == "color" else "luminance",
            )
            if not bool(reduced_halo_gate.get("accepted", False)):
                reduced_gate["accepted"] = False
                reduced_gate["status"] = "rejected"
                reduced_gate.setdefault("issues", []).extend(
                    list(reduced_halo_gate.get("issues") or [])
                )
            reduced_seam = stage8_subject_boundary_seam_report(
                pipeline,
                baseline,
                reduced_candidate,
            )
            reduced_seam_failed = bool(
                reduced_seam.get("applicable", False)
                and (
                    not bool(reduced_seam.get("available", False))
                    or str(reduced_seam.get("status") or "") == "hard_failed"
                )
            )
            if reduced_seam_failed:
                reduced_gate["accepted"] = False
                reduced_gate["status"] = "rejected"
                reduced_gate.setdefault("issues", []).append(
                    "subject_boundary_gate_failed"
                )
            step["weakened_retry"] = {
                "attempted": True,
                "mode": "adaptive_subject_boundary_delta_scaling",
                "delta_scale": {
                    "boundary": 0.25,
                    "interior": 0.50,
                },
                "boundary_retry": boundary_retry,
                "quality_gate": reduced_gate,
                "star_halo_local_gate": reduced_halo_gate,
                "subject_boundary_seam": reduced_seam,
                "candidate_pixel_sha256": pixel_sha256(reduced_candidate),
                "accepted": bool(reduced_gate.get("accepted", False)),
            }
            if bool(reduced_gate.get("accepted", False)):
                candidate = reduced_candidate
                gate = reduced_gate
                halo_gate = reduced_halo_gate
                projection = dict(projection)
                projection["weakened_retry_delta_scale"] = {
                    "boundary": 0.25,
                    "interior": 0.50,
                }
                seam = reduced_seam
        step.update(
            projection=projection,
            quality_gate=gate,
            subject_boundary_seam=seam,
        )
        if not bool(gate.get("accepted", False)):
            rollback_ok, rollback_error = _stage8_restore_finish_step(
                pipeline,
                pre_stem,
            )
            step["transaction"].update(
                rollback_performed=True,
                rollback_ok=rollback_ok,
            )
            if rollback_error:
                step["transaction"]["rollback_error"] = rollback_error
            step.update(
                status=("rejected_rolled_back" if rollback_ok else "rejected_rollback_failed"),
                reason_code=(
                    "stage8_finish_quality_gate_rejected"
                    if rollback_ok
                    else "stage8_finish_rollback_failed"
                ),
            )
            return step
        _stage8_set_image_pixels(
            pipeline,
            candidate,
            label=f"Stage8 {step_id} projected candidate",
        )
        if not pipeline._save_stage_output(candidate_stem):
            raise RuntimeError(f"{candidate_stem} save failed")
        step["transaction"]["candidate_saved"] = True
        pipeline.cmd_with_check("load", candidate_stem)
        persisted = pipeline.siril.get_image_pixeldata(preview=False)
        if persisted is None or pixel_sha256(persisted) != pixel_sha256(candidate):
            raise RuntimeError(f"{candidate_stem} persisted pixel identity mismatch")
        if not pipeline._save_stage_output("stage8_enhanced"):
            raise RuntimeError("stage8_enhanced canonical save failed")
        step["transaction"]["canonical_saved"] = True
        step.update(
            status="accepted",
            accepted=True,
            output="stage8_enhanced.fit",
            candidate_pixel_sha256=pixel_sha256(candidate),
        )
        messages.append(
            f"Stage8 {step_id} accepted via {implementation}"
        )
        return step
    except (
        AttributeError,
        CommandError,
        OSError,
        RuntimeError,
        SirilError,
        TypeError,
        ValueError,
    ) as error:
        rollback_ok, rollback_error = _stage8_restore_finish_step(
            pipeline,
            pre_stem if step["transaction"]["baseline_saved"] else base_stem,
        )
        step["transaction"].update(
            rollback_performed=True,
            rollback_ok=rollback_ok,
        )
        if rollback_error:
            step["transaction"]["rollback_error"] = rollback_error
        step.update(
            status="failed_rolled_back" if rollback_ok else "failed_rollback_failed",
            reason_code=(
                "stage8_finish_step_failed"
                if rollback_ok
                else "stage8_finish_rollback_failed"
            ),
            error=str(error),
        )
        messages.append(
            f"Stage8 {step_id} failed; rollback={'ok' if rollback_ok else 'failed'}"
        )
        return step


def _stage8_run_starless_finish(
    pipeline,
    messages: List[str],
    *,
    base_stem: str,
    channel_semantics: str,
    processing_policy: str,
    user_processing_mode: str,
    external_override: bool,
    vectra_route_selected: bool,
    effective_saturation_budget: float,
) -> Dict[str, Any]:
    """Own every post-structure Starless plugin pixel mutation in Stage8."""

    def restrict_after_rollback_failure() -> None:
        pipeline._stage8_final_quality = "starless_finish_rollback_failed"
        pipeline._stage8_fallback_used = True
        handoff = dict(getattr(pipeline, "_stage8_handoff", {}) or {})
        handoff.update(
            restricted_downstream=True,
            reason_code="stage8_starless_finish_rollback_failed",
            reason_text="stage8_starless_finish_rollback_failed",
        )
        pipeline._stage8_handoff = handoff
        if hasattr(pipeline, "_require_review"):
            pipeline._require_review(
                8,
                "stage8_starless_finish_rollback_failed",
            )

    incoming_handoff = dict(getattr(pipeline, "_stage8_handoff", {}) or {})
    issues: List[str] = []
    if str(processing_policy or "").strip().lower() != "full":
        issues.append("processing_policy_not_full")
    if str(user_processing_mode or "").strip().lower() != "auto":
        issues.append("processing_mode_not_auto")
    if not bool(getattr(pipeline, "_stage7_stretch_accepted", False)):
        issues.append("stage7_stretch_not_accepted")
    if external_override:
        issues.append("external_starless_override")
    if bool(incoming_handoff.get("restricted_downstream", False)):
        issues.append("upstream_handoff_restricted")
    if bool(getattr(pipeline, "_star_preserve_target_bypass", False)):
        issues.append("star_preserve_route")
    safe_passthrough_color_only = bool(
        (
            getattr(
                pipeline,
                "_stage8_safe_passthrough_color_only_preflight",
                {},
            )
            or {}
        ).get("accepted", False)
    )
    if (
        str(getattr(pipeline, "_stage8_final_quality", "unknown")) != "ok"
        and not safe_passthrough_color_only
    ):
        issues.append("stage8_structure_quality_not_ok")

    color_route = (
        "dualband_palette"
        if channel_semantics == NARROWBAND_COMPOSITE
        else "vectra"
        if vectra_route_selected
        else "subject_chroma"
        if channel_semantics == BROADBAND_RGB_OSC
        else "none"
    )
    report = _stage8_write_starless_finish_report(
        pipeline,
        status="skipped_ineligible" if issues else "eligible",
        reason_code=issues[0] if issues else "eligible",
        source=f"{base_stem}.fit",
        final_source=f"{base_stem}.fit",
        eligibility={
            "eligible": not issues,
            "issues": issues,
            "processing_policy": processing_policy,
            "processing_mode": user_processing_mode,
            "stage7_accepted": bool(
                getattr(pipeline, "_stage7_stretch_accepted", False)
            ),
            "external_override": bool(external_override),
        },
        steps=[],
        color_route={"selected": color_route, "reason": "runtime_contract"},
    )
    pipeline._stage8_vectra_applied = False
    pipeline._stage8_vectra_report = {
        "status": "not_selected",
        "accepted": False,
    }
    if issues:
        messages.append("Stage8 Starless finish skipped: " + ",".join(issues))
        return report
    if safe_passthrough_color_only:
        # The exact rollback is the trusted structure state.  A color-only
        # bypass must never re-enter Revela/Curves/Vectra plugin mutations;
        # only the independently gated internal subject/palette transaction
        # below may create a candidate from this point.
        report.update(
            status="skipped_safe_passthrough_color_only",
            accepted=False,
            reason_code="safe_passthrough_prohibits_starless_finish_plugins",
            steps=[],
            color_route={
                "selected": color_route,
                "reason": "safe_passthrough_internal_color_only",
                "vectra_accepted": False,
            },
        )
        pipeline._write_stage_json("stage8_starless_finish_report.json", report)
        messages.append(
            "Stage8 safe color-only passthrough skipped all Starless finish plugins"
        )
        return report
    if not bool(getattr(pipeline.cfg, "workflow_plugin_probe_enabled", False)):
        report.update(
            status="skipped_disabled",
            reason_code="workflow_plugin_probe_disabled",
        )
        pipeline._write_stage_json("stage8_starless_finish_report.json", report)
        return report

    current_stem = base_stem
    steps: List[Dict[str, Any]] = []
    for step_id, step_key, candidates in (
        (
            "revela",
            "细节/结构增强2",
            [
                ("VeraLux Revela", ("veralux_revela",)),
                ("Revela", ("revela",)),
            ],
        ),
        (
            "subject_curves",
            "最终微调颜色",
            [
                ("VeraLux Curves", ("veralux_curves",)),
                ("Curves", ("curves",)),
            ],
        ),
    ):
        step = _stage8_run_starless_finish_step(
            pipeline,
            messages,
            step_id=step_id,
            step_key=step_key,
            command_candidates=candidates,
            base_stem=current_stem,
            mode="structure",
        )
        steps.append(step)
        if bool(step.get("accepted", False)):
            current_stem = "stage8_enhanced"
        if str(step.get("status") or "").endswith("rollback_failed"):
            restrict_after_rollback_failure()
            report.update(
                status="failed_rollback_failed",
                reason_code="stage8_finish_rollback_failed",
                accepted=False,
                steps=steps,
                final_source=f"{current_stem}.fit",
            )
            pipeline._write_stage_json("stage8_starless_finish_report.json", report)
            return report

    if vectra_route_selected:
        vectra_step = _stage8_run_starless_finish_step(
            pipeline,
            messages,
            step_id="vectra",
            step_key="调色2（可选）",
            command_candidates=[
                ("VeraLux Vectra", ("veralux_vectra",)),
                ("Vectra", ("vectra",)),
            ],
            base_stem=current_stem,
            mode="color",
            effective_saturation_budget=effective_saturation_budget,
        )
        steps.append(vectra_step)
        pipeline._stage8_vectra_report = dict(vectra_step)
        pipeline._stage8_vectra_applied = bool(vectra_step.get("accepted", False))
        if pipeline._stage8_vectra_applied:
            current_stem = "stage8_enhanced"
            pipeline._stage8_saturation_execution = {
                "schema": STAGE8_STARLESS_FINISH_SCHEMA,
                "applied": True,
                "method": "vectra_exclusive_color_route",
                "requested_amount": float(effective_saturation_budget),
                "applied_amount": float(effective_saturation_budget),
                "passes": 1,
                "generic_saturation_suppressed": True,
                "suppression_reason": "reserved_for_stage8_vectra",
            }

    accepted_steps = [step for step in steps if bool(step.get("accepted", False))]
    rollback_failed = any(
        str(step.get("status") or "").endswith("rollback_failed")
        for step in steps
    )
    if rollback_failed:
        restrict_after_rollback_failure()
    report.update(
        status=(
            "failed_rollback_failed"
            if rollback_failed
            else "accepted"
            if accepted_steps
            else "no_plugin_applied"
        ),
        accepted=bool(accepted_steps) and not rollback_failed,
        reason_code=(
            "stage8_finish_rollback_failed"
            if rollback_failed
            else "accepted"
            if accepted_steps
            else "plugin_commands_unavailable_or_rejected"
        ),
        steps=steps,
        final_source=f"{current_stem}.fit",
        accepted_steps=[str(step.get("id")) for step in accepted_steps],
        color_route={
            "selected": color_route,
            "reason": "exclusive_stage8_color_route",
            "vectra_accepted": bool(pipeline._stage8_vectra_applied),
        },
    )
    pipeline._stage8_final_source = current_stem
    pipeline._write_stage_json("stage8_starless_finish_report.json", report)
    return report


def _stage8_frozen_palette_selection(
    pipeline,
) -> tuple[Dict[str, Any], Optional[str]]:
    raw = getattr(pipeline, "_stage8_palette_selection", None)
    if not isinstance(raw, dict) or not raw:
        return {}, "stage8_palette_selection_missing"
    selection = dict(raw)
    palette = str(selection.get("palette") or "").strip().upper()
    automatic = str(selection.get("automatic_palette") or "").strip().upper()
    requested = str(selection.get("requested_palette") or "").strip()
    manual = selection.get("manual_override")
    mode = str(selection.get("selection_mode") or "")
    target = selection.get("target")
    if palette not in PALETTE_CHANNELS or automatic not in PALETTE_CHANNELS:
        return {}, "stage8_palette_selection_invalid"
    if not isinstance(target, dict) or not bool(target.get("frozen", False)):
        return {}, "stage8_palette_selection_invalid"
    if manual is False:
        if requested != "auto" or mode != "automatic_target_mapping":
            return {}, "stage8_palette_selection_invalid"
        if palette != automatic:
            return {}, "stage8_palette_selection_invalid"
    elif manual is True:
        requested_palette = requested.upper()
        if (
            requested_palette not in PALETTE_CHANNELS
            or requested_palette != palette
            or mode != "explicit_user_palette"
        ):
            return {}, "stage8_palette_selection_invalid"
    else:
        return {}, "stage8_palette_selection_invalid"
    selection["palette"] = palette
    selection["automatic_palette"] = automatic
    selection["requested_palette"] = "auto" if manual is False else palette
    return selection, None


def _stage8_write_subject_chroma_report(
    pipeline,
    *,
    status: str,
    reason_code: str,
    **updates: Any,
) -> Dict[str, Any]:
    """Persist the Stage8 positive-chroma decision on every runtime route."""

    report: Dict[str, Any] = {
        "schema": STAGE8_SUBJECT_CHROMA_SCHEMA,
        "status": str(status),
        "accepted": False,
        "role": "target_aware_subject_chroma",
        "feeds_main_pipeline": False,
        "reason_code": str(reason_code),
        "output": None,
    }
    report.update(updates)
    pipeline._stage8_subject_chroma_report = dict(report)
    pipeline._write_stage_json("stage8_subject_chroma_report.json", report)
    return report


def _stage8_run_subject_chroma(
    pipeline,
    messages: List[str],
    *,
    base_stem: str,
    channel_semantics: str,
    processing_policy: str,
    user_processing_mode: str,
    external_override: bool,
    requested_saturation_budget: float,
    effective_saturation_budget: float,
    generic_saturation_suppressed: bool,
    vectra_route_selected: bool = False,
) -> Dict[str, Any]:
    """Apply the single bounded broadband positive-chroma transaction."""

    issues: List[str] = []
    if not bool(
        getattr(pipeline.cfg, "stage8_target_aware_chroma_enabled", True)
    ):
        issues.append("disabled_by_configuration")
    if not bool(
        getattr(pipeline.cfg, "stage8_nebula_saturation_enabled", True)
    ):
        issues.append("stage8_nebula_saturation_disabled")
    if channel_semantics != BROADBAND_RGB_OSC:
        issues.append("channel_semantics_not_broadband")
    if str(user_processing_mode or "").strip().lower() != "auto":
        issues.append("processing_mode_not_auto")
    if str(processing_policy or "").strip().lower() != "full":
        issues.append("processing_policy_not_full")
    if not bool(getattr(pipeline, "_stage7_stretch_accepted", False)):
        issues.append("stage7_stretch_not_accepted")
    safe_passthrough_color_only = bool(
        (
            getattr(
                pipeline,
                "_stage8_safe_passthrough_color_only_preflight",
                {},
            )
            or {}
        ).get("accepted", False)
    )
    if (
        str(getattr(pipeline, "_stage8_final_quality", "unknown")) != "ok"
        and not safe_passthrough_color_only
    ):
        issues.append("stage8_structure_quality_not_ok")
    if external_override:
        issues.append("external_starless_override")
    if vectra_route_selected:
        issues.append("vectra_exclusive_color_route")
    if bool(getattr(pipeline, "_star_preserve_target_bypass", False)):
        issues.append("star_preserve_route")
    if not generic_saturation_suppressed:
        issues.append("generic_saturation_budget_not_reserved")
    try:
        budget = float(effective_saturation_budget)
    except (TypeError, ValueError):
        budget = 0.0
    if not np.isfinite(budget) or budget <= 0.0:
        issues.append("saturation_budget_unavailable")
        budget = 0.0

    report: Dict[str, Any] = {
        "schema": STAGE8_SUBJECT_CHROMA_SCHEMA,
        "status": "skipped_ineligible",
        "accepted": False,
        "role": "target_aware_subject_chroma",
        "feeds_main_pipeline": False,
        "reason_code": issues[0] if issues else "eligible",
        "source": f"{base_stem}.fit",
        "output": None,
        "eligibility": {
            "eligible": not issues,
            "issues": list(issues),
            "channel_semantics": channel_semantics,
            "processing_mode": user_processing_mode,
            "processing_policy": processing_policy,
            "stage7_accepted": bool(
                getattr(pipeline, "_stage7_stretch_accepted", False)
            ),
            "stage8_quality": str(
                getattr(pipeline, "_stage8_final_quality", "unknown")
            ),
            "safe_passthrough_color_only": safe_passthrough_color_only,
            "external_override": bool(external_override),
        },
        "effective_saturation_budget": budget,
        "requested_saturation_budget": float(requested_saturation_budget),
        "generic_saturation_execution": {
            "suppressed": bool(generic_saturation_suppressed),
            "reason": (
                "reserved_for_stage8_target_aware_subject_chroma"
                if generic_saturation_suppressed
                else "not_reserved"
            ),
            "runtime_execution": dict(
                getattr(pipeline, "_stage8_saturation_execution", {}) or {}
            ),
        },
        "transaction": {
            "baseline": "stage8_pre_subject_chroma.fit",
            "baseline_saved": False,
            "candidate_saved": False,
            "canonical_saved": False,
            "rollback_performed": False,
            "rollback_ok": None,
        },
    }

    def finish() -> Dict[str, Any]:
        pipeline._stage8_subject_chroma_report = dict(report)
        pipeline._write_stage_json("stage8_subject_chroma_report.json", report)
        return report

    if issues:
        messages.append(
            "Stage8 target-aware subject chroma skipped: "
            + ",".join(issues)
        )
        return finish()

    try:
        pipeline.cmd_with_check("load", base_stem)
        if not pipeline._save_stage_output("stage8_pre_subject_chroma"):
            report.update(
                status="prohibited_baseline_save_failed",
                reason_code="stage8_pre_subject_chroma_save_failed",
            )
            messages.append(
                "Stage8 subject chroma prohibited: immutable baseline save failed"
            )
            return finish()
        report["transaction"]["baseline_saved"] = True
        image_data = pipeline.siril.get_image_pixeldata(preview=False)
        if image_data is None:
            raise RuntimeError("Stage8 subject chroma image buffer is empty")

        masks, frozen_keys = _stage8_frozen_color_masks(
            pipeline,
            np.asarray(image_data),
        )

        profile_resolver = getattr(
            pipeline,
            "_stage7_target_stretch_profile",
            None,
        )
        target_profile = (
            profile_resolver() if callable(profile_resolver) else {}
        )
        if not isinstance(target_profile, dict):
            target_profile = {}
        profile_name = str(target_profile.get("name") or "generic_balanced")
        baseline_saturation = subject_saturation_median(image_data, masks)
        factor_report = target_aware_chroma_factor(
            profile_name,
            subject_saturation=baseline_saturation,
            effective_saturation_budget=budget,
        )
        candidate_rgb, rendition = apply_subject_chroma_rendition(
            image_data,
            masks,
            factor=float(factor_report["factor"]),
            expand_faint_signal=(
                profile_name.strip().lower()
                == "bright_core_composite_reveal"
            ),
        )
        quality_gate = assess_subject_chroma_candidate(rendition)
        candidate_pixels = stage8_restore_rgb_like(
            pipeline,
            np.asarray(image_data),
            candidate_rgb,
        )
        halo_gate = star_halo_guard.assess_candidate(
            image_data,
            candidate_pixels,
            masks.get("star_halo_guard_mask"),
            mode="color",
        )
        weakened_retry: Dict[str, Any] = {"attempted": False}
        if (
            not bool(halo_gate.get("accepted", False))
            and int(getattr(pipeline.cfg, "stage8_quality_retry_max", 1) or 0)
            > 0
        ):
            baseline_pixels = np.asarray(image_data, dtype=np.float32)
            reduced_pixels = baseline_pixels + 0.5 * (
                np.asarray(candidate_pixels, dtype=np.float32) - baseline_pixels
            )
            reduced_halo_gate = star_halo_guard.assess_candidate(
                baseline_pixels,
                reduced_pixels,
                masks.get("star_halo_guard_mask"),
                mode="color",
            )
            weakened_retry = {
                "attempted": True,
                "delta_scale": 0.5,
                "accepted": bool(reduced_halo_gate.get("accepted", False)),
                "star_halo_local_gate": reduced_halo_gate,
            }
            if bool(reduced_halo_gate.get("accepted", False)):
                candidate_pixels = reduced_pixels
                halo_gate = reduced_halo_gate
        if not bool(halo_gate.get("accepted", False)):
            quality_gate["accepted"] = False
            quality_gate.setdefault("issues", []).extend(
                list(halo_gate.get("issues") or [])
            )
        report.update(
            target_profile=target_profile,
            factor=factor_report,
            masks={
                "source": (
                    "stage7_frozen_plus_stage8_generated"
                    if frozen_keys
                    else "stage8_generated"
                ),
                "frozen_keys": sorted(frozen_keys),
            },
            candidate=rendition,
            quality_gate=quality_gate,
            star_halo_local_gate=halo_gate,
            weakened_retry=weakened_retry,
        )
        if not bool(quality_gate.get("accepted", False)):
            pipeline.cmd_with_check("load", "stage8_pre_subject_chroma")
            rollback_ok = bool(pipeline._save_stage_output("stage8_enhanced"))
            report["transaction"].update(
                rollback_performed=True,
                rollback_ok=rollback_ok,
                canonical_saved=rollback_ok,
            )
            report.update(
                status=(
                    "rejected_by_quality_gate"
                    if rollback_ok
                    else "rejected_rollback_failed"
                ),
                reason_code=(
                    "stage8_subject_chroma_quality_gate_rejected"
                    if rollback_ok
                    else "stage8_subject_chroma_rollback_failed"
                ),
            )
            if rollback_ok:
                pipeline._stage8_final_source = "stage8_enhanced"
                messages.append(
                    "Stage8 subject chroma rejected; retained structure-only baseline: "
                    + ",".join(
                        str(item) for item in quality_gate.get("issues", [])
                    )
                )
            else:
                pipeline._stage8_final_quality = (
                    "subject_chroma_rollback_failed"
                )
                pipeline._stage8_final_source = (
                    "stage8_pre_subject_chroma"
                )
                pipeline._stage8_fallback_used = True
                if hasattr(pipeline, "_require_review"):
                    pipeline._require_review(
                        8,
                        "stage8_subject_chroma_rollback_failed",
                    )
                messages.append(
                    "Stage8 subject chroma quality rejection could not restore "
                    "a verified canonical checkpoint"
                )
            return finish()

        _stage8_set_image_pixels(
            pipeline,
            candidate_pixels,
            label="Stage8 target-aware subject chroma",
        )
        candidate_saved = pipeline._save_stage_output(
            "stage8_subject_chroma_candidate"
        )
        report["transaction"]["candidate_saved"] = bool(candidate_saved)
        if not candidate_saved:
            raise RuntimeError("stage8_subject_chroma_candidate save failed")
        canonical_saved = pipeline._save_stage_output("stage8_enhanced")
        report["transaction"]["canonical_saved"] = bool(canonical_saved)
        if not canonical_saved:
            raise RuntimeError("Stage8 subject chroma canonical save failed")
        report.update(
            status="accepted",
            accepted=True,
            feeds_main_pipeline=True,
            reason_code="accepted",
            output="stage8_subject_chroma_candidate.fit",
            final_source="stage8_enhanced.fit",
        )
        pipeline._stage8_final_source = "stage8_enhanced"
        pipeline._stage8_saturation_execution = {
            "schema": STAGE8_SUBJECT_CHROMA_SCHEMA,
            "applied": True,
            "method": "masked_subject_chroma_rendition",
            "requested_amount": budget,
            "applied_amount": budget,
            "factor": float(factor_report["factor"]),
            "target_profile": profile_name,
            "generic_saturation_suppressed": bool(
                generic_saturation_suppressed
            ),
            "suppression_reason": (
                "reserved_for_stage8_target_aware_subject_chroma"
                if generic_saturation_suppressed
                else None
            ),
        }
        messages.append(
            "Stage8 target-aware subject chroma accepted "
            f"(profile={profile_name}, factor={float(factor_report['factor']):.3f})"
        )
        return finish()
    except (
        AttributeError,
        CommandError,
        FloatingPointError,
        IndexError,
        KeyError,
        OSError,
        RuntimeError,
        SirilError,
        TypeError,
        ValueError,
    ) as error:
        report.update(
            status="failed",
            accepted=False,
            feeds_main_pipeline=False,
            reason_code="stage8_subject_chroma_failed",
            error=str(error),
        )
        rollback_ok = False
        if bool(report["transaction"].get("baseline_saved", False)):
            report["transaction"]["rollback_performed"] = True
            try:
                pipeline.cmd_with_check("load", "stage8_pre_subject_chroma")
                rollback_ok = bool(
                    pipeline._save_stage_output("stage8_enhanced")
                )
            except (CommandError, SirilError):
                rollback_ok = False
        report["transaction"]["rollback_ok"] = rollback_ok
        report["transaction"]["canonical_saved"] = rollback_ok
        if rollback_ok:
            report["status"] = "failed_rolled_back"
            pipeline._stage8_final_source = "stage8_enhanced"
            messages.append(
                "Stage8 subject chroma failed and rolled back: "
                f"{pipeline._short_text(error, 160)}"
            )
        else:
            report["status"] = "failed_rollback_failed"
            report["reason_code"] = "stage8_subject_chroma_rollback_failed"
            pipeline._stage8_final_quality = "subject_chroma_rollback_failed"
            pipeline._stage8_final_source = "stage8_pre_subject_chroma"
            pipeline._stage8_fallback_used = True
            if hasattr(pipeline, "_require_review"):
                pipeline._require_review(
                    8,
                    "stage8_subject_chroma_rollback_failed",
                )
            messages.append(
                "Stage8 subject chroma rollback failed: "
                f"{pipeline._short_text(error, 160)}"
            )
        return finish()


def _stage8_run_dualband_palette(
    pipeline,
    messages: List[str],
    *,
    base_stem: str,
    channel_semantics: str,
    processing_policy: str,
    external_override: bool,
) -> Dict[str, Any]:
    """Apply one target-selected palette transaction after structural QA."""
    channel_profile = getattr(pipeline, "channel_profile", None)
    if not isinstance(channel_profile, dict):
        channel_profile = {}
    mapping = getattr(pipeline, "narrowband_channel_mapping", None)
    mapping_source = "stage4_runtime_contract"
    if not isinstance(mapping, dict) or not mapping:
        mapping = channel_profile.get("narrowband_mapping")
        mapping_source = "channel_profile_contract"
    if not isinstance(mapping, dict):
        mapping = {}
        mapping_source = "missing"
    eligibility = stage8_palette_eligibility(
        enabled=bool(
            getattr(pipeline.cfg, "stage8_dualband_palette_enabled", True)
        ),
        channel_semantics=channel_semantics,
        mapping=mapping,
        mapping_confidence_min=float(
            getattr(pipeline.cfg, "stage4_nbn_mapping_confidence_min", 0.85)
        ),
        processing_policy=processing_policy,
        stage8_quality=(
            "ok"
            if bool(
                (
                    getattr(
                        pipeline,
                        "_stage8_safe_passthrough_color_only_preflight",
                        {},
                    )
                    or {}
                ).get("accepted", False)
            )
            else str(getattr(pipeline, "_stage8_final_quality", "unknown"))
        ),
        stage7_accepted=bool(
            getattr(pipeline, "_stage7_stretch_accepted", False)
        ),
        external_override=external_override,
    )
    selection, selection_issue = _stage8_frozen_palette_selection(pipeline)
    if selection_issue:
        eligibility = dict(eligibility)
        eligibility["eligible"] = False
        eligibility["issues"] = [
            *list(eligibility.get("issues") or []),
            selection_issue,
        ]
    color_report = getattr(pipeline, "color_calibration_report", {}) or {}
    physical_color = color_report.get("physical_color") or {}
    degraded_color = color_report.get("degraded_color_correction") or {}
    color_parent = {
        "method": color_report.get("method"),
        "physical_color_accepted": bool(physical_color.get("accepted", False)),
        "degraded_pcc_applied": bool(degraded_color.get("applied", False)),
        "requires_review": bool(color_report.get("requires_review", False)),
        "degraded_pcc_mapping_policy": "allowed_with_existing_review_notice",
    }
    report: Dict[str, Any] = {
        "schema": DUALBAND_PALETTE_SCHEMA,
        "algorithm_source": dict(DUALBAND_PALETTE_SOURCE),
        "status": "skipped",
        "accepted": False,
        "role": "artistic_false_color",
        "feeds_main_pipeline": False,
        "source": f"{base_stem}.fit",
        "mapping_evidence": mapping,
        "mapping_source": mapping_source,
        "eligibility": eligibility,
        "planned_palette": selection.get("palette"),
        "requested_palette": selection.get("requested_palette"),
        "automatic_palette": selection.get("automatic_palette"),
        "selection_mode": selection.get("selection_mode"),
        "manual_override": selection.get("manual_override"),
        "target_selection": selection,
        "color_parent": color_parent,
        "transaction": {
            "baseline": "stage8_pre_palette.fit",
            "baseline_saved": False,
            "candidate_saved": False,
            "rollback_performed": False,
            "rollback_ok": None,
        },
    }

    def finish() -> Dict[str, Any]:
        pipeline._stage8_palette_report = dict(report)
        pipeline._write_stage_json("stage8_palette_report.json", report)
        return report

    if not bool(eligibility.get("eligible", False)):
        report["status"] = "skipped_ineligible"
        messages.append(
            "Stage8 dual-band palette skipped: "
            + ",".join(str(item) for item in eligibility.get("issues", []))
        )
        return finish()

    try:
        pipeline.cmd_with_check("load", base_stem)
        if not pipeline._save_stage_output("stage8_pre_palette"):
            report.update(
                status="prohibited_baseline_save_failed",
                issues=["stage8_pre_palette_save_failed"],
            )
            messages.append(
                "Stage8 dual-band palette prohibited: pre-palette baseline save failed"
            )
            return finish()
        report["transaction"]["baseline_saved"] = True

        image_data = pipeline.siril.get_image_pixeldata(preview=False)
        if image_data is None:
            raise RuntimeError("Stage8 dual-band palette image buffer is empty")
        masks, frozen_mask_keys = _stage8_frozen_color_masks(
            pipeline,
            image_data,
        )
        manual_override = bool(selection.get("manual_override", False))
        primary_palette = str(selection["palette"]).upper()
        palette_order: List[str] = []
        for palette_id in (
            (primary_palette,)
            if manual_override
            else (primary_palette, "HOO", *PALETTE_CHANNELS.keys())
        ):
            normalized = str(palette_id).upper()
            if normalized not in palette_order:
                palette_order.append(normalized)

        candidate_records: List[Dict[str, Any]] = []
        best_candidate: Optional[np.ndarray] = None
        best_candidate_report: Optional[Dict[str, Any]] = None
        best_score: Optional[tuple[float, float, float, float, int]] = None
        for candidate_index, palette_id in enumerate(palette_order):
            candidate_pixels, candidate_report = (
                build_dualband_palette_candidate(
                    image_data,
                    palette=palette_id,
                    core_mask=masks["core_mask"],
                    nebula_mask=masks["nebula_mask"],
                    faint_nebula_mask=masks["faint_nebula_mask"],
                    background_mask=masks["background_mask"],
                    star_mask=masks.get("star_mask"),
                    star_halo_guard_mask=masks.get("star_halo_guard_mask"),
                    strength=float(
                        getattr(
                            pipeline.cfg,
                            "stage8_dualband_palette_strength",
                            0.85,
                        )
                    ),
                    luma_drift_p95_max=float(
                        getattr(
                            pipeline.cfg,
                            "stage8_dualband_palette_luma_drift_max",
                            0.005,
                        )
                    ),
                    clip_growth_max=float(
                        getattr(
                            pipeline.cfg,
                            "stage8_dualband_palette_clip_growth_max",
                            0.002,
                        )
                    ),
                    quality_warning_tolerance=float(
                        getattr(
                            pipeline.cfg,
                            "stage8_dualband_palette_quality_warning_tolerance",
                            0.50,
                        )
                    ),
                    subject_chroma_separation_gain_min=1.0e-4,
                    subject_saturation_input_ratio_min=float(
                        getattr(
                            pipeline.cfg,
                            "stage8_palette_subject_saturation_input_ratio_min",
                            0.50,
                        )
                    ),
                    subject_saturation_absolute_min=float(
                        getattr(
                            pipeline.cfg,
                            "stage8_palette_subject_saturation_absolute_min",
                            0.08,
                        )
                    ),
                )
            )
            halo_gate = star_halo_guard.assess_candidate(
                image_data,
                candidate_pixels,
                masks.get("star_halo_guard_mask"),
                mode="color",
            )
            weakened_retry: Dict[str, Any] = {"attempted": False}
            if (
                not bool(halo_gate.get("accepted", False))
                and int(
                    getattr(pipeline.cfg, "stage8_quality_retry_max", 1) or 0
                )
                > 0
            ):
                baseline_pixels = np.asarray(image_data, dtype=np.float32)
                reduced_pixels = baseline_pixels + 0.5 * (
                    np.asarray(candidate_pixels, dtype=np.float32)
                    - baseline_pixels
                )
                reduced_halo_gate = star_halo_guard.assess_candidate(
                    baseline_pixels,
                    reduced_pixels,
                    masks.get("star_halo_guard_mask"),
                    mode="color",
                )
                weakened_retry = {
                    "attempted": True,
                    "delta_scale": 0.5,
                    "accepted": bool(
                        reduced_halo_gate.get("accepted", False)
                    ),
                    "star_halo_local_gate": reduced_halo_gate,
                }
                if bool(reduced_halo_gate.get("accepted", False)):
                    candidate_pixels = reduced_pixels
                    halo_gate = reduced_halo_gate
            candidate_report["star_halo_local_gate"] = halo_gate
            candidate_report["frozen_mask_keys"] = frozen_mask_keys
            candidate_report["weakened_retry"] = weakened_retry
            if not bool(halo_gate.get("accepted", False)):
                candidate_report["accepted"] = False
                candidate_report.setdefault("issues", []).extend(
                    list(halo_gate.get("issues") or [])
                )
            metrics = candidate_report.get("metrics") or {}
            separation_gain = float(
                metrics.get(
                    "subject_background_chroma_separation_gain",
                    0.0,
                )
                or 0.0
            )
            subject_gain = float(
                metrics.get("subject_saturation_p50_gain", 0.0) or 0.0
            )
            luma_drift = float(metrics.get("luminance_drift_p95", 1.0) or 0.0)
            clip_growth = float(metrics.get("clip_growth", 1.0) or 0.0)
            chroma_gain_passed = bool(separation_gain > 1.0e-4)
            selection_score = (
                -separation_gain,
                -subject_gain,
                luma_drift,
                clip_growth,
                candidate_index,
            )
            candidate_report["automatic_selection"] = {
                "eligible": bool(
                    candidate_report.get("accepted", False)
                    and chroma_gain_passed
                ),
                "subject_chroma_gain_required": True,
                "subject_chroma_gain_min_exclusive": 1.0e-4,
                "subject_chroma_gain_passed": chroma_gain_passed,
                "selection_score": [float(value) for value in selection_score],
                "candidate_order": candidate_index,
            }
            candidate_records.append(candidate_report)
            if not bool(candidate_report["automatic_selection"]["eligible"]):
                continue
            if best_score is None or selection_score < best_score:
                best_score = selection_score
                best_candidate = np.array(candidate_pixels, copy=True)
                best_candidate_report = candidate_report

        report["candidates"] = candidate_records
        report["candidate_count"] = len(candidate_records)
        report["selection_execution_mode"] = (
            "explicit_single_candidate"
            if manual_override
            else "automatic_candidate_competition"
        )
        if best_candidate is None or best_candidate_report is None:
            report.update(
                status="rejected_by_palette_quality_gate",
                issues=["auto_palette_subject_chroma_gain_unmet"],
            )
            pipeline.cmd_with_check("load", "stage8_pre_palette")
            report["transaction"].update(
                rollback_performed=True,
                rollback_ok=True,
            )
            messages.append(
                "Stage8 dual-band palette candidates rejected; retained "
                "pre-palette enhancement"
            )
            return finish()

        candidate = best_candidate
        candidate_report = best_candidate_report
        selected_palette = str(candidate_report.get("palette") or primary_palette)
        report["candidate"] = candidate_report
        report["selection_score"] = [
            float(value) for value in (best_score or ())
        ]

        _stage8_set_image_pixels(
            pipeline,
            candidate,
            label=f"Stage8 dual-band {selected_palette} palette",
        )
        candidate_saved = pipeline._save_stage_output("stage8_palette_selected")
        report["transaction"]["candidate_saved"] = bool(candidate_saved)
        if not candidate_saved:
            raise RuntimeError("stage8_palette_selected save failed")
        canonical_saved = pipeline._save_stage_output("stage8_enhanced")
        report["transaction"]["canonical_saved"] = bool(canonical_saved)
        if not canonical_saved:
            raise RuntimeError("Stage8 canonical output save failed")

        candidate_warnings = list(candidate_report.get("warnings") or [])
        report.update(
            status=(
                "accepted_with_warning"
                if candidate_warnings
                else "accepted"
            ),
            accepted=True,
            feeds_main_pipeline=True,
            palette=selected_palette,
            synthetic_sii=bool(candidate_report.get("synthetic_sii", False)),
            output="stage8_palette_selected.fit",
            final_source="stage8_enhanced.fit",
            warnings=candidate_warnings,
        )
        pipeline._stage8_artistic_palette_applied = True
        pipeline._stage8_final_source = "stage8_enhanced"
        if candidate_warnings:
            messages.append(
                "Stage8 dual-band artistic palette accepted with quality reminder "
                f"(target={selection['category']}, palette={selected_palette}, "
                f"warnings={','.join(candidate_warnings)})"
            )
        else:
            messages.append(
                "Stage8 dual-band artistic palette accepted "
                f"(target={selection['category']}, palette={selected_palette}, "
                f"candidates={len(candidate_records)})"
            )
        return finish()
    except (
        AttributeError,
        CommandError,
        FloatingPointError,
        IndexError,
        KeyError,
        OSError,
        RuntimeError,
        SirilError,
        TypeError,
        ValueError,
    ) as error:
        report.update(
            status="failed",
            accepted=False,
            feeds_main_pipeline=False,
            error=str(error),
        )
        rollback_ok = False
        if bool(report["transaction"].get("baseline_saved", False)):
            report["transaction"]["rollback_performed"] = True
            try:
                pipeline.cmd_with_check("load", "stage8_pre_palette")
                rollback_ok = bool(
                    pipeline._save_stage_output("stage8_enhanced")
                )
            except (CommandError, SirilError):
                rollback_ok = False
        report["transaction"]["rollback_ok"] = rollback_ok
        if rollback_ok:
            report["status"] = "failed_rolled_back"
            report["reason_code"] = "stage8_palette_failed_rolled_back"
            messages.append(
                "Stage8 dual-band palette failed and rolled back: "
                f"{pipeline._short_text(error, 160)}"
            )
        else:
            report["status"] = "failed_rollback_failed"
            report["reason_code"] = "stage8_palette_rollback_failed"
            pipeline._stage8_final_quality = "palette_rollback_failed"
            pipeline._stage8_final_source = "stage8_pre_palette"
            pipeline._stage8_fallback_used = True
            if hasattr(pipeline, "_require_review"):
                pipeline._require_review(
                    8,
                    "stage8_palette_rollback_failed",
                )
            messages.append(
                "Stage8 dual-band palette rollback failed: "
                f"{pipeline._short_text(error, 160)}"
            )
        return finish()


def run_stage8_nebula_enhancement(pipeline) -> None:
    """
    阶段 8: Starless 深加工（含外部回写）
    - 优先使用本轮通过质量门的 stage7_stretched
    - Stage 7 不合格时回退 starless / stage6_starless
    - SASP WaveScale/DSE 输出必须经 Starless soft mask 回混
    - 插件不可用时回退到内置分区增强
    """
    stage_label = PipelineStage.NEBULA_ENHANCEMENT.label
    pipeline.log.stage_start(stage_label)
    pipeline._clear_stage_reviews(8)
    status = 'ok'
    messages: List[str] = []
    failure_action = str(
        getattr(pipeline.cfg, "stage8_failure_action", "auto_fallback")
        or "auto_fallback"
    )
    user_processing_mode = str(
        getattr(pipeline.cfg, "stage8_processing_mode", "auto") or "auto"
    )
    pipeline._stage8_final_source = "stage8_enhanced"
    pipeline._stage8_fallback_used = False
    pipeline._stage8_final_quality = "unknown"
    pipeline._stage8_input_source = None
    pipeline._stage8_input_fallback_used = False
    pipeline._stage8_artistic_palette_applied = False
    pipeline._stage8_palette_report = {}
    pipeline._stage8_saturation_execution = {}
    pipeline._stage8_color_quality_report = {}
    pipeline._stage8_final_cumulative_quality_report = {}
    pipeline._stage8_subject_chroma_report = {}
    pipeline._stage8_starless_finish_report = {}
    pipeline._stage8_vectra_applied = False
    pipeline._stage8_vectra_report = {
        "status": "not_run",
        "accepted": False,
    }
    _stage8_write_subject_chroma_report(
        pipeline,
        status="not_run",
        reason_code="stage8_route_not_resolved",
    )
    _stage8_write_starless_finish_report(
        pipeline,
        status="not_run",
        reason_code="stage8_route_not_resolved",
    )
    channel_semantics = str(
        getattr(pipeline, "_channel_semantics", "unknown") or "unknown"
    )
    pipeline._stage8_subject_boundary_seam_report = {}
    stage8_initial_quality = "unknown"
    stage8_reject_reason = ""
    stage8_policy_failure_recorded = False
    separation_state = str(
        getattr(
            pipeline,
            "_star_separation_state",
            StarSeparationState.ACCEPTED.value,
        )
    )
    target_bypass_stage7_rejected = bool(
        separation_state == StarSeparationState.TARGET_BYPASS.value
        and not bool(getattr(pipeline, "_stage7_stretch_accepted", False))
    )
    if separation_state in {
        StarSeparationState.REJECTED.value,
        StarSeparationState.TOOL_FAILED.value,
    } or target_bypass_stage7_rejected:
        upstream_handoff = dict(
            getattr(pipeline, "_stage8_handoff", {}) or {}
        )
        upstream_reason_code = str(
            upstream_handoff.get("reason_code") or ""
        )
        review_reason_code = (
            "stage7_stretch_not_accepted_target_bypass"
            if target_bypass_stage7_rejected
            else upstream_reason_code
            if upstream_reason_code == "stage7_stretch_not_accepted"
            else "star_separation_unavailable"
        )
        final_quality = (
            "stage7_stretch_review_target_bypass"
            if target_bypass_stage7_rejected
            else "stage7_stretch_review_only"
            if review_reason_code == "stage7_stretch_not_accepted"
            else "star_separation_unavailable"
        )
        source_candidates = [
            getattr(pipeline, "_stage7_review_source", None),
            "stage7_review_with_stars",
            getattr(pipeline, "_stage6_passthrough_source", None),
            "stage6_passthrough",
        ]
        selected_source = None
        load_errors: List[str] = []
        for candidate in dict.fromkeys(
            str(item) for item in source_candidates if item
        ):
            try:
                pipeline.cmd_with_check("load", candidate)
                selected_source = candidate
                break
            except (CommandError, SirilError) as error:
                load_errors.append(f"{candidate}: {error}")

        stage8_saved = False
        if selected_source:
            stage8_saved = pipeline._save_stage_output(
                "stage8_review_with_stars"
            )
        pipeline._stage8_input_source = selected_source
        pipeline._stage8_input_fallback_used = True
        pipeline._stage8_fallback_used = True
        pipeline._stage8_final_quality = final_quality
        if target_bypass_stage7_rejected:
            # Keep Stage 9 pinned to the required with-stars checkpoint even
            # when it could not be produced.  This makes the next stage fail
            # closed instead of resolving its generic Starless default.
            pipeline._stage8_final_source = "stage8_review_with_stars"
        else:
            pipeline._stage8_final_source = (
                "stage8_review_with_stars"
                if stage8_saved
                else selected_source or "stage6_passthrough"
            )
        handoff = _set_stage8_handoff(
            pipeline,
            source_stem=pipeline._stage8_final_source,
            passthrough=True,
            restricted_downstream=True,
            final_quality=pipeline._stage8_final_quality,
            reason_code=review_reason_code,
            reason_text=review_reason_code,
        )
        subject_chroma_report = _stage8_write_subject_chroma_report(
            pipeline,
            status="bypassed_with_stars_review",
            reason_code=review_reason_code,
            source=selected_source,
        )
        palette_report = _stage8_run_dualband_palette(
            pipeline,
            messages,
            base_stem=str(pipeline._stage8_final_source),
            channel_semantics=channel_semantics,
            processing_policy="skip",
            external_override=False,
        )
        handoff["color_rendition"] = {
            "mode": "bypassed_with_stars_review",
            "accepted": False,
            "report": "stage8_subject_chroma_report.json",
        }
        pipeline._stage8_handoff = handoff
        pipeline.starless_file = None
        if target_bypass_stage7_rejected:
            decisive_status = (
                "failed"
                if failure_action == "stop" or not stage8_saved
                else "degraded"
            )
        else:
            decisive_status = "failed" if failure_action == "stop" else (
                "degraded" if failure_action == "preserve_review" else (
                    "degraded" if stage8_saved else "failed"
                )
            )
        report = {
            "stage": "stage8_nebula_enhancement",
            "status": decisive_status,
            "mode": "with_stars_review_passthrough",
            "star_separation_state": separation_state,
            "source": selected_source,
            "final_source": pipeline._stage8_final_source,
            "starless_enhancement_applied": False,
            "load_errors": load_errors,
            "handoff": handoff,
            "processing_mode": user_processing_mode,
            "failure_action": failure_action,
            "subject_chroma": subject_chroma_report,
            "dualband_palette": palette_report,
        }
        pipeline._write_stage_json("stage8_enhancement_report.json", report)
        messages.append(
            "starless-only enhancement skipped; using with-stars review "
            f"passthrough because {review_reason_code}"
        )
        if selected_source:
            messages.append(f"with_stars_source={selected_source}")
        elif target_bypass_stage7_rejected:
            messages.append(
                "no with-stars review source could be loaded; delivery withheld"
            )
        if load_errors:
            messages.append("load_errors=" + " | ".join(load_errors))
        if stage8_saved and hasattr(pipeline, "_create_stage_review_bundle"):
            review = pipeline._create_stage_review_bundle(
                "stage8_nebula_enhancement",
                selected_source,
                "stage8_review_with_stars",
                context={
                    "mode": "with_stars_review_passthrough",
                    "dualband_palette": palette_report,
                    "subject_chroma": subject_chroma_report,
                },
                candidates=[
                    {
                        "name": "with_stars_review_passthrough",
                        "stem": "stage8_review_with_stars",
                        "status": decisive_status,
                        "selected": True,
                    }
                ],
                selected_candidate="with_stars_review_passthrough",
            )
            if review.get("report_path"):
                messages.append(f"review_bundle={review['report_path']}")
        elapsed = pipeline.log.stage_end(stage_label)
        if failure_action != "auto_fallback" and hasattr(
            pipeline, "_record_stage_policy_event"
        ):
            pipeline._record_stage_policy_event(
                8,
                event="decisive_failure",
                reason=review_reason_code,
                source="input_contract",
            )
        if failure_action == "preserve_review" and not target_bypass_stage7_rejected:
            pipeline._require_review(8, review_reason_code)
        pipeline._record_stage(
            stage_label,
            decisive_status,
            elapsed,
            "；".join(messages),
            execution="safe_passthrough",
            reason_code=review_reason_code,
            details={
                "stage8_handoff": handoff,
                "dualband_palette": palette_report,
            },
            review_reasons=pipeline._stage_review_reasons(8),
        )
        return

    color_limits = color_safety_limits(
        getattr(pipeline, "pipeline_policy", {}) or {},
        getattr(pipeline, "color_calibration_report", {}) or {},
    )
    broadband_color_allowed = channel_semantics == BROADBAND_RGB_OSC
    physical_broadband_anchor = physical_broadband_anchor_accepted(
        getattr(pipeline, "channel_profile", {}) or {},
        getattr(pipeline, "color_calibration_report", {}) or {},
    )
    global_color_rebalance_allowed = bool(
        broadband_color_allowed and not physical_broadband_anchor
    )
    if physical_broadband_anchor:
        messages.append(
            "Stage4 physical broadband color anchor frozen; repeat white balance, "
            "global channel matrix and unconditional green removal are prohibited"
        )
    effective_stage8_saturation = 0.0
    incoming_handoff = dict(getattr(pipeline, "_stage8_handoff", {}) or {})
    spatial_lineage_path = (
        Path(pipeline.process_dir) / "stage3_spatial_background_lineage.json"
        if getattr(pipeline, "process_dir", None) is not None
        else None
    )
    stage3_spatial_lineage = (
        spatial_background_lineage.load_lineage(pipeline.process_dir)
        if spatial_lineage_path is not None and spatial_lineage_path.is_file()
        else {
            "schema": spatial_background_lineage.LINEAGE_SCHEMA,
            "status": "unavailable",
            "accepted": False,
            "issues": ["stage3_spatial_background_lineage_unavailable"],
        }
    )
    if not bool(stage3_spatial_lineage.get("accepted", False)):
        incoming_handoff.update(
            requested_policy="background_only",
            processing_policy="background_only",
            restricted_downstream=True,
            reason_code="stage3_spatial_opponent_lineage_unverified",
            reason_text=(
                "Stage3 spatial background lineage is unresolved; positive "
                "Stage8 structure enhancement is prohibited"
            ),
            quality_status="restricted",
        )
        pipeline._stage8_handoff = incoming_handoff
        pipeline._stage8_conservative_mode = True
        pipeline._require_review(
            8,
            "stage3_spatial_opponent_lineage_unverified",
        )
        messages.append(
            "Stage3 spatial background lineage unresolved; Stage8 restricted"
        )
    incoming_policy = str(
        incoming_handoff.get("processing_policy")
        or incoming_handoff.get("requested_policy")
        or "full"
    )
    incoming_reason_text = str(incoming_handoff.get("reason_text") or "")
    if incoming_policy != "full":
        messages.append(
            f"stage8 requested_policy={incoming_policy}"
            + (f"; {incoming_reason_text}" if incoming_reason_text else "")
        )

    if (
        bool(getattr(pipeline, "_star_preserve_target_bypass", False))
        and bool(getattr(pipeline, "_stage7_stretch_accepted", False))
    ):
        source_stem = pipeline.stretched_name or "stage7_stretched"
        try:
            pipeline.cmd_with_check("load", source_stem)
            secondary_context = stage8_star_preserve_nebulosity_context(
                pipeline
            )
            secondary_overlay: Dict[str, Any] = {
                **secondary_context,
                "status": "not_requested",
                "accepted": False,
            }
            if bool(secondary_context.get("eligible", False)):
                try:
                    source_pixels = pipeline.siril.get_image_pixeldata(
                        preview=False
                    )
                    if source_pixels is None:
                        raise RuntimeError(
                            "star-preserve secondary-nebulosity pixels unavailable"
                        )
                    overlay_pixels, secondary_overlay = (
                        stage8_star_preserve_nebulosity_overlay(
                            pipeline,
                            source_pixels,
                        )
                    )
                    if bool(secondary_overlay.get("accepted", False)):
                        _stage8_set_image_pixels(
                            pipeline,
                            overlay_pixels,
                            label="Stage8 star-preserve secondary nebulosity",
                        )
                except (RuntimeError, TypeError, ValueError) as error:
                    secondary_overlay = {
                        **secondary_context,
                        "status": "failed_safe_passthrough",
                        "accepted": False,
                        "error": pipeline._short_text(error, 160),
                    }
                    pipeline.log.warn(
                        "Stage8 secondary-nebulosity overlay failed; preserving "
                        f"the accepted stellar source unchanged: {error}"
                    )
            pipeline._stage8_input_source = source_stem
            pipeline._stage8_input_fallback_used = False
            stage8_saved = pipeline._save_stage_output("stage8_enhanced")
            pipeline._stage8_final_source = (
                "stage8_enhanced" if stage8_saved else source_stem
            )
            overlay_accepted = bool(secondary_overlay.get("accepted", False))
            pipeline._stage8_final_quality = (
                "star_preserve_secondary_nebulosity"
                if overlay_accepted
                else "star_preserve_bypass"
            )
            pipeline._stage8_fallback_used = False
            spatial_lineage_verified = bool(
                stage3_spatial_lineage.get("accepted", False)
                and stage3_spatial_lineage.get("support_sha256")
            )
            star_preserve_formal = bool(
                overlay_accepted and stage8_saved and spatial_lineage_verified
            )
            handoff = _set_stage8_handoff(
                pipeline,
                source_stem=pipeline._stage8_final_source,
                passthrough=not overlay_accepted,
                restricted_downstream=not star_preserve_formal,
                final_quality=pipeline._stage8_final_quality,
                processing_route=(
                    "star_preserve_secondary_nebulosity"
                    if overlay_accepted
                    else "review_only"
                ),
                formal_eligible=star_preserve_formal,
                reason_code=(
                    "star_preserve_secondary_nebulosity"
                    if overlay_accepted
                    else "star_preserve_target_bypass"
                ),
                reason_text=(
                    "stellar primary preserved; diffuse secondary nebulosity "
                    "received one bounded local adjustment"
                    if overlay_accepted
                    else "star_preserve_target_bypass"
                ),
            )
            handoff["spatial_background_lineage"] = {
                "schema": stage3_spatial_lineage.get("schema"),
                "status": stage3_spatial_lineage.get("status"),
                "accepted": spatial_lineage_verified,
                "support_sha256": stage3_spatial_lineage.get("support_sha256"),
                "issues": list(stage3_spatial_lineage.get("issues") or []),
            }
            if not star_preserve_formal:
                handoff["formal_eligible"] = False
                handoff["restricted_downstream"] = True
                if hasattr(pipeline, "_require_review"):
                    pipeline._require_review(
                        8,
                        "star_preserve_stage8_formal_handoff_unverified",
                    )
            subject_chroma_report = _stage8_write_subject_chroma_report(
                pipeline,
                status="bypassed_star_preserve",
                reason_code=(
                    "star_preserve_secondary_nebulosity"
                    if overlay_accepted
                    else "star_preserve_target_bypass"
                ),
                source=f"{source_stem}.fit",
            )
            palette_report = _stage8_run_dualband_palette(
                pipeline,
                messages,
                base_stem=str(pipeline._stage8_final_source),
                channel_semantics=channel_semantics,
                processing_policy="skip",
                external_override=False,
            )
            handoff["color_rendition"] = {
                "mode": "bypassed_star_preserve",
                "accepted": False,
                "report": "stage8_subject_chroma_report.json",
            }
            handoff["star_preserve_secondary_nebulosity"] = dict(
                secondary_overlay
            )
            handoff = _write_stage8_handoff(pipeline, handoff)
            pipeline.starless_file = pipeline.process_dir / f"{pipeline._stage8_final_source}.fit"
            report = {
                "stage": "stage8_nebula_enhancement",
                "status": "ok" if overlay_accepted else "skipped",
                "mode": (
                    "star_preserve_secondary_nebulosity"
                    if overlay_accepted
                    else "star_preserve_target_bypass"
                ),
                "source": source_stem,
                "final_source": pipeline._stage8_final_source,
                "target_type": (
                    pipeline._active_target_type()
                    if hasattr(pipeline, "_active_target_type")
                    else "generic_low_snr_safe"
                ),
                "secondary_context": secondary_context,
                "secondary_nebulosity_overlay": secondary_overlay,
                "subject_chroma": subject_chroma_report,
                "dualband_palette": palette_report,
                "handoff": handoff,
            }
            pipeline._write_stage_json("stage8_enhancement_report.json", report)
            messages.append(
                (
                    "star-preserve target kept stellar pixels exact and applied "
                    "one bounded diffuse-nebulosity overlay "
                    if overlay_accepted
                    else "star-preserve target bypassed Starless-only enhancement "
                )
                + f"(source={source_stem})"
            )
            if stage8_saved and hasattr(pipeline, "_create_stage_review_bundle"):
                review = pipeline._create_stage_review_bundle(
                    "stage8_nebula_enhancement",
                    source_stem,
                    "stage8_enhanced",
                    context={
                        "mode": report["mode"],
                        "dualband_palette": palette_report,
                        "subject_chroma": subject_chroma_report,
                    },
                    candidates=[
                        {
                            "name": (
                                "star_preserve_secondary_nebulosity"
                                if overlay_accepted
                                else "star_preserve_bypass"
                            ),
                            "stem": "stage8_enhanced",
                            "status": "accepted" if overlay_accepted else "skipped",
                            "selected": True,
                        }
                    ],
                    selected_candidate=(
                        "star_preserve_secondary_nebulosity"
                        if overlay_accepted
                        else "star_preserve_bypass"
                    ),
                )
                if review.get("report_path"):
                    messages.append(f"review_bundle={review['report_path']}")
            elapsed = pipeline.log.stage_end(stage_label)
            stage_status = (
                "ok"
                if stage8_saved and overlay_accepted
                else "skipped"
                if stage8_saved
                else "degraded"
            )
            pipeline._record_stage(
                stage_label,
                stage_status,
                elapsed,
                "；".join(messages),
                execution=(
                    "completed"
                    if overlay_accepted
                    else "safe_passthrough"
                ),
                reason_code=str(handoff.get("reason_code") or ""),
                details={
                    "stage8_handoff": handoff,
                    "secondary_nebulosity_overlay": secondary_overlay,
                    "dualband_palette": palette_report,
                },
            )
            return
        except (CommandError, SirilError) as error:
            pipeline._star_preserve_target_bypass = False
            pipeline.log.warn(
                "Star-preserve Stage8 bypass failed; using conservative Stage8 path: "
                f"{error}"
            )
            messages.append(
                "star-preserve Stage8 bypass failed: "
                f"{pipeline._short_text(error, 160)}"
            )
    elif bool(getattr(pipeline, "_star_preserve_target_bypass", False)):
        pipeline._star_preserve_target_bypass = False
        messages.append(
            "star-preserve Stage8 bypass disabled because Stage7 output was not accepted"
        )

    external_starless_source: Optional[str] = None
    external_starless = pipeline._find_external_fit(
        [
            "sasp_starless.fit",
            "starless_sasp.fit",
            "starless_from_sasp.fit",
        ]
    )
    if external_starless:
        try:
            imported = pipeline._import_external_fit(external_starless, "starless_external")
            if imported:
                pipeline.cmd_with_check("save", "starless")
                external_starless_source = "starless"
                pipeline.starless_file = pipeline.process_dir / "starless.fit"
                pipeline.log.info(f"已导入外部 Starless: {external_starless.name}")
                messages.append(f"导入外部 Starless: {external_starless.name}")
        except (OSError, CommandError, SirilError) as e:
            pipeline.log.warn(f"导入外部 Starless 失败，继续使用本地 starless: {e}")
            messages.append(f"外部 Starless 导入失败: {pipeline._short_text(e, 160)}")

    pipeline.log.info("加载 Stage8 输入图像...")
    stage8_input_source = _load_stage8_input(
        pipeline,
        messages,
        explicit_source=external_starless_source,
    )
    pipeline._write_stage_json(
        "stage8_input_selection.json",
        {
            "stage": "stage8_input_selection",
            "stage7_accepted": bool(
                getattr(pipeline, "_stage7_stretch_accepted", False)
            ),
            "preferred_source": external_starless_source
            or (
                getattr(pipeline, "_stage7_stretch_output", None)
                or getattr(pipeline, "stretched_name", None)
                or "stage7_stretched"
            ),
            "selected_source": stage8_input_source,
            "fallback_used": bool(pipeline._stage8_input_fallback_used),
            "external_override": bool(external_starless_source),
            "external_source_file": (
                str(external_starless) if external_starless_source else None
            ),
        },
    )
    pipeline._last_stage8_masked_diagnostics = {}
    # The local input guard and immutable rollback baseline are mandatory even
    # even when masked enhancement is disabled. In particular,
    # a Stage 6 `skip` handoff must never depend on optional Stage 8 features.
    stage8_diagnostics_enabled = True
    stage8_processing_plan: Optional[Dict[str, Any]] = None
    stage8_guard_report: Dict[str, Any] = {}
    stage8_limited_mode = False
    stage8_high_risk = False
    if hasattr(pipeline, "_stage7_halo_residue_score") and hasattr(
        pipeline, "_stage7_effective_halo_threshold"
    ):
        stage8_high_risk = (
            pipeline._stage7_halo_residue_score()
            > pipeline._stage7_effective_halo_threshold()
        )
    if stage8_diagnostics_enabled:
        if pipeline._save_stage_output("stage8_input_starless"):
            messages.append("stage8_input_starless saved for masked diagnostics")
            stage8_guard_report = pipeline._stage8_input_enhancement_guard()
            current_handoff = dict(getattr(pipeline, "_stage8_handoff", {}) or {})
            current_handoff.update(
                {
                    "processing_policy": stage8_guard_report.get(
                        "processing_policy",
                        current_handoff.get("processing_policy", "full"),
                    ),
                    "reason_code": stage8_guard_report.get("reason_code")
                    or current_handoff.get("reason_code", ""),
                    "reason_text": stage8_guard_report.get("reason_text")
                    or current_handoff.get("reason_text", ""),
                    "reasons": stage8_guard_report.get("reason_details")
                    or current_handoff.get("reasons", []),
                }
            )
            pipeline._stage8_handoff = current_handoff
            if stage8_guard_report.get("background_only"):
                pipeline.cmd_with_check("load", "stage8_input_starless")
                stage8_saved = pipeline._save_stage_output("stage8_enhanced")
                pipeline._stage8_final_source = "stage8_input_starless"
                pipeline._stage8_fallback_used = False
                pipeline._stage8_final_quality = "background_only_passthrough"
                pipeline.starless_file = (
                    pipeline.process_dir / "stage8_input_starless.fit"
                )
                reason_code = str(
                    stage8_guard_report.get("reason_code")
                    or "stage8_subject_risk_background_only"
                )
                reason_text = str(
                    stage8_guard_report.get("reason_text")
                    or "Stage8 subject enhancement unsafe; baseline retained"
                )
                handoff = _set_stage8_handoff(
                    pipeline,
                    source_stem="stage8_input_starless",
                    passthrough=True,
                    restricted_downstream=True,
                    final_quality=pipeline._stage8_final_quality,
                    reason_code=reason_code,
                    reason_text=reason_text,
                )
                subject_chroma_report = _stage8_write_subject_chroma_report(
                    pipeline,
                    status="bypassed_background_only",
                    reason_code=reason_code,
                    source="stage8_input_starless.fit",
                )
                palette_report = _stage8_run_dualband_palette(
                    pipeline,
                    messages,
                    base_stem="stage8_input_starless",
                    channel_semantics=channel_semantics,
                    processing_policy="background_only",
                    external_override=bool(external_starless_source),
                )
                handoff["color_rendition"] = {
                    "mode": "bypassed_background_only",
                    "accepted": False,
                    "report": "stage8_subject_chroma_report.json",
                }
                handoff["processing_policy"] = "background_only"
                handoff["outcome_reason_code"] = (
                    "stage8_background_only_no_incremental_damage"
                )
                pipeline._stage8_handoff = handoff
                color_quality = _write_stage8_color_quality_report(
                    pipeline,
                    final_source="stage8_input_starless",
                    requested_saturation=float(pipeline.cfg.nebula_saturation),
                    effective_saturation=0.0,
                    applied_saturation=0.0,
                    palette_report=palette_report,
                )
                payload = {
                    **stage8_guard_report,
                    "stage": "stage8_input_guard",
                    "status": (
                        "background_only_passthrough"
                        if stage8_saved
                        else "background_only_checkpoint_failed"
                    ),
                    "input_source": stage8_input_source,
                    "final_source": pipeline._stage8_final_source,
                    "fallback_used": False,
                    "passthrough": True,
                    "processing_policy": "background_only",
                    "background_operation": "none",
                    "background_operation_reason": (
                        "Stage8 has introduced no candidate damage; retain immutable "
                        "baseline instead of repeating Stage5/7/10 denoise"
                    ),
                    "color_quality": color_quality,
                    "subject_chroma": subject_chroma_report,
                    "dualband_palette": palette_report,
                    "handoff": handoff,
                    "final_quality": pipeline._stage8_final_quality,
                    "processing_mode": user_processing_mode,
                    "failure_action": failure_action,
                }
                pipeline._write_stage_json(
                    "stage8_quality.json",
                    {
                        "mode": "background_only_passthrough",
                        "initial": payload,
                        "final": payload,
                    },
                )
                pipeline._write_stage_json(
                    "stage8_enhancement_report.json",
                    payload,
                )
                messages.append(
                    "stage8 background_only route retained immutable baseline; "
                    "no duplicate background denoise/SCNR applied"
                )
                if hasattr(pipeline, "_create_stage_review_bundle"):
                    review = pipeline._create_stage_review_bundle(
                        "stage8_nebula_enhancement",
                        "stage8_input_starless",
                        "stage8_enhanced",
                        context={
                            "mode": "background_only_passthrough",
                            "reason": reason_text,
                            "dualband_palette": palette_report,
                            "subject_chroma": subject_chroma_report,
                        },
                        candidates=[
                            {
                                "name": "background_only_passthrough",
                                "stem": "stage8_enhanced",
                                "status": payload["status"],
                                "selected": True,
                            }
                        ],
                        selected_candidate="background_only_passthrough",
                    )
                    if review.get("report_path"):
                        messages.append(f"review_bundle={review['report_path']}")
                elapsed = pipeline.log.stage_end(stage_label)
                pipeline._record_stage(
                    stage_label,
                    "ok" if stage8_saved else "degraded",
                    elapsed,
                    "；".join(messages),
                    execution="safe_passthrough",
                    reason_code=reason_code,
                    details={
                        "reason_text": reason_text,
                        "stage8_handoff": handoff,
                        "dualband_palette": palette_report,
                    },
                )
                return
            if stage8_guard_report.get("skip_enhancement"):
                guard_reasons = [
                    str(item) for item in stage8_guard_report.get("reasons", []) if item
                ]
                user_preserve = (
                    str(stage8_guard_report.get("user_processing_mode") or "")
                    == "preserve"
                )
                decisive_guard_failure = bool(
                    stage8_guard_report.get("hard_reasons")
                ) and not user_preserve
                conservative_skip = bool(
                    getattr(pipeline, "_stage8_conservative_mode", False)
                    or stage8_guard_report.get("conservative_mode", False)
                    or "stage8_conservative_mode_after_stage7_starless_repair"
                    in guard_reasons
                )
                guard_status = "user_preserve" if user_preserve else (
                    "conservative_skipped" if conservative_skip else "skipped"
                )
                pipeline._stage8_final_source = "stage8_input_starless"
                pipeline._stage8_fallback_used = not user_preserve
                pipeline._stage8_final_quality = guard_status
                pipeline.cmd_with_check("load", "stage8_input_starless")
                pipeline._save_stage_output("stage8_enhanced")
                pipeline.starless_file = pipeline.process_dir / "stage8_input_starless.fit"
                stage8_reject_reason = ", ".join(guard_reasons[:3])
                handoff = _set_stage8_handoff(
                    pipeline,
                    source_stem=pipeline._stage8_final_source,
                    passthrough=True,
                    restricted_downstream=True,
                    final_quality=pipeline._stage8_final_quality,
                    reason_code=str(
                        stage8_guard_report.get("reason_code")
                        or "stage8_input_guard_skip"
                    ),
                    reason_text=str(
                        stage8_guard_report.get("reason_text")
                        or stage8_reject_reason
                        or "stage8_input_guard_skip"
                    ),
                )
                if handoff.get("formal_eligible") is not True:
                    pipeline._require_review(
                        8,
                        str(
                            handoff.get("reason_code")
                            or "stage8_input_guard_skip"
                        ),
                    )
                subject_chroma_report = _stage8_write_subject_chroma_report(
                    pipeline,
                    status="bypassed_input_guard",
                    reason_code=str(
                        stage8_guard_report.get("reason_code")
                        or "stage8_input_guard_skip"
                    ),
                    source="stage8_input_starless.fit",
                )
                palette_report = _stage8_run_dualband_palette(
                    pipeline,
                    messages,
                    base_stem="stage8_input_starless",
                    channel_semantics=channel_semantics,
                    processing_policy=str(
                        stage8_guard_report.get("processing_policy") or "skip"
                    ),
                    external_override=bool(external_starless_source),
                )
                handoff["color_rendition"] = {
                    "mode": "bypassed_input_guard",
                    "accepted": False,
                    "report": "stage8_subject_chroma_report.json",
                }
                pipeline._stage8_handoff = handoff
                messages.append(
                    "stage8 enhancement skipped by input guard: "
                    + (stage8_reject_reason or "unsafe starless input")
                )
                color_quality = _write_stage8_color_quality_report(
                    pipeline,
                    final_source="stage8_input_starless",
                    requested_saturation=float(pipeline.cfg.nebula_saturation),
                    effective_saturation=0.0,
                    applied_saturation=0.0,
                    palette_report=palette_report,
                )
                guard_payload = {
                    **stage8_guard_report,
                    "stage": "stage8_input_guard",
                    "status": guard_status,
                    "input_source": stage8_input_source,
                    "final_source": pipeline._stage8_final_source,
                    "fallback_used": pipeline._stage8_fallback_used,
                    "passthrough": True,
                    "handoff": handoff,
                    "final_quality": pipeline._stage8_final_quality,
                    "color_quality": color_quality,
                    "subject_chroma": subject_chroma_report,
                    "dualband_palette": palette_report,
                    "processing_mode": user_processing_mode,
                    "failure_action": failure_action,
                }
                pipeline._write_stage_json(
                    "stage8_quality.json",
                    {
                        "mode": "input_guard_skip",
                        "initial": guard_payload,
                        "final": guard_payload,
                    },
                )
                pipeline._write_stage_json("stage8_enhancement_report.json", guard_payload)
                if hasattr(pipeline, "_create_stage_review_bundle"):
                    review = pipeline._create_stage_review_bundle(
                        "stage8_nebula_enhancement",
                        "stage8_input_starless",
                        "stage8_enhanced",
                        context={
                            "mode": "input_guard_skip",
                            "reasons": guard_reasons,
                            "dualband_palette": palette_report,
                            "subject_chroma": subject_chroma_report,
                        },
                        candidates=[
                            {
                                "name": "input_guard_skip",
                                "stem": "stage8_enhanced",
                                "status": guard_payload["status"],
                                "issues": guard_reasons,
                                "selected": True,
                            }
                        ],
                        selected_candidate="input_guard_skip",
                    )
                    if review.get("report_path"):
                        messages.append(f"review_bundle={review['report_path']}")
                elapsed = pipeline.log.stage_end(stage_label)
                record_status = (
                    "skipped"
                    if user_preserve
                    else "failed"
                    if decisive_guard_failure and failure_action == "stop"
                    else "degraded"
                    if decisive_guard_failure and failure_action == "preserve_review"
                    else "degraded"
                )
                if decisive_guard_failure and failure_action != "auto_fallback":
                    if hasattr(pipeline, "_record_stage_policy_event"):
                        pipeline._record_stage_policy_event(
                            8,
                            event="decisive_failure",
                            reason=str(handoff.get("reason_text") or "stage8_input_guard_skip"),
                            source="input_guard",
                        )
                    if failure_action == "preserve_review":
                        pipeline._require_review(
                            8,
                            str(handoff.get("reason_code") or "stage8_input_guard_skip"),
                        )
                pipeline._record_stage(
                    stage_label,
                    record_status,
                    elapsed,
                    "；".join(messages),
                    execution="safe_passthrough",
                    reason_code=str(handoff.get("reason_code") or ""),
                    details={
                        "reason_text": str(handoff.get("reason_text") or ""),
                        "stage8_handoff": handoff,
                        "dualband_palette": palette_report,
                    },
                    review_reasons=pipeline._stage_review_reasons(8),
                )
                return
            stage8_limited_mode = (
                stage8_guard_report.get("processing_policy") == "limited"
            )
            if stage8_guard_report.get("advisories"):
                messages.extend(
                    "stage8 advisory: " + str(item)
                    for item in stage8_guard_report["advisories"]
                    if item
                )
            if stage8_high_risk or stage8_limited_mode:
                stage8_processing_plan = {
                    **(stage8_processing_plan or {}),
                    "saturation": min(
                        float(
                            (stage8_processing_plan or {}).get(
                                "saturation",
                                pipeline.cfg.nebula_saturation,
                            )
                            or 0.0
                        ),
                        float(
                            getattr(
                                pipeline.cfg,
                                "stage8_limited_saturation_max",
                                0.05,
                            )
                        ),
                    ),
                    "bg_factor": 0,
                    "unsharp_radius": 0.0,
                    "unsharp_amount": 0.0,
                    "apply_after_plugins": False,
                }
                messages.append(
                    "stage8 limited halo policy: skip SASP/global color/unsharp; "
                    "generate object-mask-only candidate"
                )
        else:
            messages.append(
                "stage8_input_starless 保存失败；无法建立质量门基线，安全旁路阶段8增强"
            )
            pipeline._stage8_final_source = stage8_input_source
            pipeline._stage8_fallback_used = True
            pipeline._stage8_final_quality = "baseline_save_failed"
            handoff = _set_stage8_handoff(
                pipeline,
                source_stem=stage8_input_source,
                passthrough=True,
                restricted_downstream=True,
                final_quality=pipeline._stage8_final_quality,
                reason_code="stage8_baseline_save_failed",
                reason_text="stage8_baseline_save_failed",
            )
            subject_chroma_report = _stage8_write_subject_chroma_report(
                pipeline,
                status="prohibited_baseline_save_failed",
                reason_code="stage8_baseline_save_failed",
                source=f"{stage8_input_source}.fit",
            )
            palette_report = _stage8_run_dualband_palette(
                pipeline,
                messages,
                base_stem=stage8_input_source,
                channel_semantics=channel_semantics,
                processing_policy="skip",
                external_override=bool(external_starless_source),
            )
            handoff["color_rendition"] = {
                "mode": "prohibited_baseline_save_failed",
                "accepted": False,
                "report": "stage8_subject_chroma_report.json",
            }
            pipeline._stage8_handoff = handoff
            failure_payload = {
                "stage": "stage8_nebula_enhancement",
                "status": "degraded",
                "mode": "baseline_save_failed_safe_passthrough",
                "requested_policy": incoming_policy,
                "input_source": stage8_input_source,
                "final_source": stage8_input_source,
                "fallback_used": True,
                "passthrough": True,
                "final_quality": pipeline._stage8_final_quality,
                "handoff": handoff,
                "subject_chroma": subject_chroma_report,
                "dualband_palette": palette_report,
            }
            pipeline._write_stage_json(
                "stage8_quality.json",
                {
                    "mode": "baseline_unavailable",
                    "initial": failure_payload,
                    "final": failure_payload,
                },
            )
            pipeline._write_stage_json(
                "stage8_enhancement_report.json",
                failure_payload,
            )
            elapsed = pipeline.log.stage_end(stage_label)
            pipeline._record_stage(
                stage_label,
                "degraded",
                elapsed,
                "；".join(messages),
                execution="safe_passthrough",
                reason_code="stage8_baseline_save_failed",
                details={
                    "stage8_handoff": handoff,
                    "dualband_palette": palette_report,
                },
            )
            return

    requested_saturation = float(
        (stage8_processing_plan or {}).get(
            "saturation",
            pipeline.cfg.nebula_saturation,
        )
    )
    mixed_composite_context = stage8_mixed_nebula_composite_context(pipeline)
    if (
        broadband_color_allowed
        and not stage8_high_risk
        and not stage8_limited_mode
        and mixed_composite_context.get("eligible", False)
    ):
        composite_floor = min(
            0.14,
            float(color_limits.get("max_saturation_boost", 0.10)),
        )
        if requested_saturation < composite_floor:
            messages.append(
                "resolved red/blue composite field raised the single masked "
                f"color request {requested_saturation:.3f}->{composite_floor:.3f}"
            )
            requested_saturation = composite_floor
    if not broadband_color_allowed:
        requested_saturation = 0.0
        messages.append(
            "Stage8 global color transforms skipped by channel semantics "
            f"({channel_semantics})"
        )
    effective_stage8_saturation = clamp_saturation_boost(
        requested_saturation,
        already_applied=float(getattr(pipeline, "_saturation_boost_applied", 0.0)),
        limits=color_limits,
    )
    effective_plan: Dict[str, Any] = {
        "saturation": effective_stage8_saturation,
        "bg_factor": pipeline.cfg.nebula_bg_factor,
        "unsharp_radius": 0.8,
        "unsharp_amount": 0.35,
    }
    effective_plan.update(stage8_processing_plan or {})
    effective_plan["saturation"] = effective_stage8_saturation
    effective_plan["color_policy_limits"] = color_limits
    resolved_processing_policy = str(
        stage8_guard_report.get("processing_policy")
        or ("limited" if stage8_limited_mode else incoming_policy)
        or "full"
    ).strip().lower()
    vectra_route_selected = bool(
        getattr(pipeline.cfg, "optional_color_transform_enabled", False)
        and getattr(pipeline.cfg, "workflow_plugin_probe_enabled", False)
        and global_color_rebalance_allowed
        and broadband_color_allowed
        and not physical_broadband_anchor
        and user_processing_mode.strip().lower() == "auto"
        and resolved_processing_policy == "full"
        and not stage8_limited_mode
        and not stage8_high_risk
        and bool(getattr(pipeline, "_stage7_stretch_accepted", False))
        and not bool(external_starless_source)
        and effective_stage8_saturation > 0.0
    )
    subject_chroma_budget_reserved = bool(
        getattr(
            pipeline.cfg,
            "stage8_target_aware_chroma_enabled",
            True,
        )
        and getattr(
            pipeline.cfg,
            "stage8_nebula_saturation_enabled",
            True,
        )
        and broadband_color_allowed
        and user_processing_mode.strip().lower() == "auto"
        and resolved_processing_policy == "full"
        and not stage8_limited_mode
        and not stage8_high_risk
        and bool(getattr(pipeline, "_stage7_stretch_accepted", False))
        and not bool(external_starless_source)
        and not vectra_route_selected
        and effective_stage8_saturation > 0.0
    )
    positive_chroma_budget_reserved = bool(
        subject_chroma_budget_reserved or vectra_route_selected
    )
    if positive_chroma_budget_reserved:
        effective_plan["saturation"] = 0.0
        if vectra_route_selected:
            messages.append(
                "Stage8 positive chroma budget reserved exclusively for the "
                "post-structure Vectra transaction"
            )
        else:
            messages.append(
                "Stage8 positive chroma budget reserved for the post-structure "
                "target-aware subject transaction"
            )
    stage8_processing_plan = effective_plan
    if effective_stage8_saturation != requested_saturation:
        messages.append(
            "Stage4 color policy capped Stage8 saturation "
            f"{requested_saturation:.3f}->{effective_stage8_saturation:.3f} "
            f"(budget={color_limits['max_saturation_boost']:.3f})"
        )

    processed = False
    limited_processing_failed = False
    stage8_restricted_mode = stage8_high_risk or stage8_limited_mode
    sasp_api_used = (
        None
        if stage8_restricted_mode
        else pipeline._run_sasp_stage8_api(stage8_processing_plan)
    )
    if sasp_api_used:
        processed = True
        messages.append(f"SASP Starless 深加工使用 {sasp_api_used}")
    elif not stage8_restricted_mode and getattr(
        pipeline,
        "_last_sasp_stage8_error",
        None,
    ):
        messages.append(
            "SASP Starless 深加工 API 不可用: "
            f"{pipeline._short_text(pipeline._last_sasp_stage8_error, 160)}"
        )

    if not sasp_api_used and pipeline.cfg.workflow_plugin_probe_enabled:
        messages.append("SASP Siril 深加工命令不可用，跳过实验性 sasp_* 命令探测")

    if vectra_route_selected:
        messages.append(
            "optional color transform deferred to the exclusive Stage8 "
            "Vectra transaction"
        )
    elif (
        pipeline.cfg.optional_color_transform_enabled
        and physical_broadband_anchor
    ):
        messages.append(
            "optional global color plugin skipped: Stage4 physical color anchor frozen"
        )

    if processed and stage8_processing_plan and stage8_processing_plan.get("apply_after_plugins"):
        # The SASP runner already feeds its structure candidate through the same
        # masked local recipe. Replaying the built-in chain here would apply a
        # second saturation/contrast pass and consume the color budget twice.
        messages.append(
            "supplemental replay skipped: SASP candidate already completed "
            "the single masked post-structure color-recovery pass"
        )

    if not processed:
        pipeline.log.info("未检测到可用 Starless 深加工命令，使用内置增强链...")
        builtin_plan = stage8_processing_plan or {
            "saturation": pipeline.cfg.nebula_saturation,
            "bg_factor": pipeline.cfg.nebula_bg_factor,
            "unsharp_radius": 0.8,
            "unsharp_amount": 0.35,
        }
        try:
            messages.extend(
                pipeline._apply_stage8_builtin_enhancement(
                    builtin_plan,
                    label="内置",
                )
            )
            pipeline.log.info("Starless 内置增强链完成")
        except (
            AttributeError,
            CommandError,
            RuntimeError,
            SirilError,
            TypeError,
            ValueError,
        ) as e:
            pipeline.log.warn(f"Starless 内置增强跳过: {e}")
            status = 'degraded'
            limited_processing_failed = stage8_limited_mode
            messages.append(f"Starless 内置增强失败: {pipeline._short_text(e, 160)}")

    limited_candidate_checkpoint_saved = False
    limited_candidate_rolled_back = False
    stage_saved = pipeline._save_stage_output("stage8_enhanced")
    if stage_saved:
        stage8_saved = True
        if stage8_saved:
            if stage8_limited_mode:
                limited_candidate_checkpoint_saved = pipeline._save_stage_output(
                    "stage8_limited_candidate"
                )
                messages.append(
                    "stage8 limited candidate checkpoint "
                    + ("saved" if limited_candidate_checkpoint_saved else "save failed")
                )
            diff_note = pipeline._stage_diff_note("stage8_enhanced", "stage8_input_starless")
            if diff_note:
                messages.append(diff_note)
            feature = pipeline._measure_current_features()
            if feature:
                messages.append(f"Starless 后特征: {format_feature_summary(feature)}")
                blue_guard_note = (
                    pipeline._apply_starless_blue_guard(feature)
                    if global_color_rebalance_allowed and not stage8_limited_mode
                    else None
                )
                if blue_guard_note:
                    messages.append(blue_guard_note)
                    guarded_feature = pipeline._measure_current_features()
                    if guarded_feature:
                        messages.append(
                            "Starless 蓝色门控后特征: "
                            f"{format_feature_summary(guarded_feature)}"
                        )
                        before_excess = max(
                            0.0,
                            feature.blue_dominance
                            - max(1.08, feature.red_dominance + 0.12),
                        )
                        after_excess = max(
                            0.0,
                            guarded_feature.blue_dominance
                            - max(1.08, guarded_feature.red_dominance + 0.12),
                        )
                        guard_worse = (
                            guarded_feature.blue_dominance
                            > feature.blue_dominance + 0.03
                            or after_excess > before_excess + 0.01
                        )
                        if guard_worse:
                            pipeline.log.warn(
                                "Starless 蓝色门控后指标未改善，回滚到门控前结果"
                            )
                            messages.append(
                                "Starless 蓝色门控回滚 "
                                f"(blue_dom {feature.blue_dominance:.3f}"
                                f"->{guarded_feature.blue_dominance:.3f}, "
                                f"blue_excess {before_excess:.3f}->{after_excess:.3f})"
                            )
                            try:
                                pipeline.cmd_with_check("load", "stage8_enhanced")
                            except (CommandError, SirilError) as e:
                                status = 'degraded'
                                messages.append(
                                    "Starless 蓝色门控回滚失败: "
                                    f"{pipeline._short_text(e, 160)}"
                                )
                        else:
                            pipeline._save_stage_output("stage8_enhanced")
                    else:
                        pipeline._save_stage_output("stage8_enhanced")

            if stage8_diagnostics_enabled:
                quality_record = pipeline._stage8_quality_assessment()
                if stage8_limited_mode and limited_processing_failed:
                    quality_record = dict(quality_record)
                    quality_record["status"] = "poor"
                    quality_record["issues"] = list(
                        quality_record.get("issues") or []
                    ) + ["stage8_limited_processing_failed"]
                if stage8_limited_mode and not limited_candidate_checkpoint_saved:
                    quality_record = dict(quality_record)
                    quality_record["status"] = "poor"
                    quality_record["issues"] = list(
                        quality_record.get("issues") or []
                    ) + ["stage8_limited_candidate_checkpoint_save_failed"]
                initial_advisories = list(
                    quality_record.get("advisories") or []
                )
                if initial_advisories:
                    messages.append(
                        "stage8 quality advisory; continuing without rollback: "
                        + ", ".join(str(item) for item in initial_advisories[:3])
                    )
                stage8_initial_quality = str(quality_record.get("status", "unknown"))
                quality_payload: Dict[str, Any] = {
                    "initial": quality_record,
                    "final": quality_record,
                    "mode": "masked_parameter_optimization",
                    "processing_plan": stage8_processing_plan,
                    "masked_diagnostics": getattr(
                        pipeline,
                        "_last_stage8_masked_diagnostics",
                        {},
                    ),
                }
                if quality_record["status"] != "ok":
                    stage8_reject_reason = ", ".join(quality_record.get("issues", [])[:3])
                    messages.append(
                        "stage8_quality diagnostic: "
                        + stage8_reject_reason
                    )
                    latest_quality = quality_record
                    try:
                        correction_note = (
                            pipeline._apply_stage8_color_correction_from_quality(
                                quality_record
                            )
                            if failure_action == "auto_fallback"
                            and global_color_rebalance_allowed
                            and not stage8_limited_mode
                            else None
                        )
                        if correction_note:
                            messages.append(correction_note)
                            corrected_quality = pipeline._stage8_quality_assessment()
                            quality_payload["color_correction"] = corrected_quality
                            quality_payload["final"] = corrected_quality
                            latest_quality = corrected_quality
                            corrected_metrics = corrected_quality.get("candidate_metrics")
                            if isinstance(corrected_metrics, dict):
                                corrected_blue_excess = float(
                                    corrected_metrics.get("blue_excess", 0.0)
                                )
                                messages.append(
                                    "stage8_quality recalculated after blue correction "
                                    f"(status={corrected_quality.get('status')}, "
                                    f"blue_excess={corrected_blue_excess:.3f})"
                                )
                                correction_passes: List[Dict[str, Any]] = []
                                for pass_index in range(2):
                                    target_blue_excess = pipeline._stage8_target_blue_excess(
                                        corrected_quality
                                    )
                                    if corrected_blue_excess <= target_blue_excess:
                                        break
                                    extra_note = pipeline._apply_stage8_color_correction_from_quality(
                                        corrected_quality
                                    )
                                    if not extra_note:
                                        break
                                    messages.append(
                                        "stage8 extra blue correction applied "
                                        f"(pass={pass_index + 2})"
                                    )
                                    corrected_quality = pipeline._stage8_quality_assessment()
                                    correction_passes.append(corrected_quality)
                                    quality_payload["final"] = corrected_quality
                                    latest_quality = corrected_quality
                                    corrected_metrics = corrected_quality.get("candidate_metrics")
                                    if not isinstance(corrected_metrics, dict):
                                        break
                                    corrected_blue_excess = float(
                                        corrected_metrics.get("blue_excess", 0.0)
                                    )
                                    messages.append(
                                        "stage8_quality recalculated after extra blue correction "
                                        f"(status={corrected_quality.get('status')}, "
                                        f"blue_excess={corrected_blue_excess:.3f})"
                                    )
                                if correction_passes:
                                    quality_payload["extra_color_corrections"] = correction_passes
                        elif stage8_limited_mode:
                            messages.append(
                                "stage8 quality color correction disabled by limited policy"
                            )
                        elif failure_action != "auto_fallback":
                            messages.append(
                                "stage8 quality correction disabled by failure_action="
                                f"{failure_action}"
                            )
                        elif not broadband_color_allowed:
                            messages.append(
                                "stage8 quality color correction skipped by "
                                f"channel semantics ({channel_semantics})"
                            )
                        elif physical_broadband_anchor:
                            messages.append(
                                "stage8 global color correction skipped: Stage4 "
                                "physical color anchor is frozen"
                            )
                        else:
                            messages.append("stage8_quality did not require color correction")
                    except (CommandError, SirilError) as e:
                        messages.append(
                            "stage8 color correction failed: "
                            f"{pipeline._short_text(e, 160)}"
                        )
                    needs_conservative_rerun = bool(
                        latest_quality.get("status") != "ok"
                        and not stage8_limited_mode
                        and pipeline._stage8_needs_conservative_rerun(latest_quality)
                    )
                    if needs_conservative_rerun and failure_action != "auto_fallback":
                        stage8_policy_failure_recorded = True
                        reason = ", ".join(
                            str(item) for item in latest_quality.get("issues", [])[:3]
                        ) or "stage8_quality_gate_failed"
                        if hasattr(pipeline, "_record_stage_policy_event"):
                            pipeline._record_stage_policy_event(
                                8,
                                event="decisive_failure",
                                reason=reason,
                                source="quality_gate",
                            )
                        if failure_action == "preserve_review":
                            pipeline._require_review(8, "stage8_quality_gate_failed")
                            status = "degraded"
                        elif failure_action == "stop":
                            status = "failed"
                        messages.append(
                            "stage8 candidate search stopped by failure_action="
                            f"{failure_action}"
                        )
                    retry_allowed = int(
                        getattr(pipeline.cfg, "stage8_quality_retry_max", 1) or 0
                    ) > 0
                    if needs_conservative_rerun and not retry_allowed:
                        messages.append("stage8 conservative quality retry disabled")
                    if (
                        needs_conservative_rerun
                        and retry_allowed
                        and failure_action == "auto_fallback"
                    ):
                        original_saturation = (
                            float(stage8_processing_plan.get("saturation"))
                            if stage8_processing_plan
                            else float(pipeline.cfg.nebula_saturation)
                        )
                        rerun = pipeline._stage8_conservative_rerun(original_saturation)
                        quality_payload["conservative_rerun"] = rerun
                        quality_payload["masked_diagnostics"] = getattr(
                            pipeline,
                            "_last_stage8_masked_diagnostics",
                            {},
                        )
                        final_assessment = rerun.get("assessment")
                        if isinstance(final_assessment, dict):
                            quality_payload["final"] = final_assessment
                            pipeline._stage8_final_quality = str(final_assessment.get("status", "unknown"))
                            if final_assessment.get("status") != "ok":
                                fallback_source = (
                                    "stage8_conservative_enhanced"
                                    if (pipeline.process_dir / "stage8_conservative_enhanced.fit").exists()
                                    else "stage8_input_starless"
                                )
                                pipeline._stage8_final_source = fallback_source
                                pipeline._stage8_fallback_used = True
                            else:
                                pipeline._stage8_final_source = "stage8_enhanced"
                                pipeline._stage8_fallback_used = True
                                status = "degraded" if status == "ok" else status
                                messages.append(
                                    "stage8 conservative rerun passed; "
                                    "candidate retained as degraded fallback"
                                )
                        safe_sat = rerun.get("safe_saturation")
                        safe_sat_text = (
                            f"{float(safe_sat):.3f}"
                            if isinstance(safe_sat, (int, float))
                            else str(safe_sat)
                        )
                        messages.append(
                            "stage8 conservative masked rerun applied "
                            f"safe_sat={safe_sat_text} status={rerun.get('status')}"
                        )
                final_quality_record = quality_payload.get("final") or quality_record
                assessed_final_status = str(
                    final_quality_record.get("status") or "unknown"
                )
                if not stage8_limited_mode:
                    pipeline._stage8_final_quality = assessed_final_status
                if (
                    stage8_limited_mode
                    and str(final_quality_record.get("status") or "") != "ok"
                ):
                    if failure_action != "auto_fallback" and not stage8_policy_failure_recorded:
                        stage8_policy_failure_recorded = True
                        reason = ", ".join(
                            str(item) for item in final_quality_record.get("issues", [])[:3]
                        ) or "stage8_limited_quality_gate_failed"
                        if hasattr(pipeline, "_record_stage_policy_event"):
                            pipeline._record_stage_policy_event(
                                8,
                                event="decisive_failure",
                                reason=reason,
                                source="quality_gate",
                            )
                        if failure_action == "preserve_review":
                            pipeline._require_review(
                                8, "stage8_limited_quality_gate_failed"
                            )
                            status = "degraded"
                        elif failure_action == "stop":
                            status = "failed"
                    rollback_ok = pipeline._rollback_stage8_to_input()
                    limited_candidate_rolled_back = bool(rollback_ok)
                    quality_payload["limited_candidate_rollback"] = {
                        "target": "stage8_input_starless",
                        "status": "ok" if rollback_ok else "failed",
                    }
                    messages.append(
                        "stage8 limited candidate rejected; rollback "
                        f"status={quality_payload['limited_candidate_rollback']['status']}"
                    )
                    if rollback_ok:
                        pipeline._stage8_final_source = "stage8_input_starless"
                        pipeline._stage8_fallback_used = True
                        pipeline._stage8_final_quality = "limited_candidate_rejected"
                    else:
                        status = "degraded"
                        pipeline._stage8_final_quality = "limited_rollback_failed"
                elif stage8_limited_mode:
                    pipeline._stage8_final_source = "stage8_enhanced"
                    pipeline._stage8_final_quality = "ok"
                    messages.append(
                        "stage8 limited candidate accepted by quality and halo gates"
                    )
                pipeline._write_stage_json("stage8_quality.json", quality_payload)
            enhancement_report = (
                pipeline._stage8_enhancement_quality_report()
                if hasattr(pipeline, "_stage8_enhancement_quality_report")
                else {}
            )
            if (
                not enhancement_report
                and str(pipeline._stage8_final_quality or "") == "poor"
            ):
                enhancement_report = {
                    "stage": "stage8_enhancement",
                    "status": "poor",
                    "issues": ["stage8_assessment_gate_failed"],
                }
            if enhancement_report:
                enhancement_advisories = list(
                    enhancement_report.get("advisories") or []
                )
                if enhancement_advisories:
                    messages.append(
                        "stage8 enhancement advisory; retained candidate: "
                        + ", ".join(
                            str(item) for item in enhancement_advisories[:3]
                        )
                    )
                enhancement_report["conservative_rerun_applied"] = bool(
                    "conservative_rerun" in locals().get("quality_payload", {})
                )
                assessment_gate_failed = (
                    str(pipeline._stage8_final_quality or "") == "poor"
                )
                enhancement_report["assessment_gate_status"] = (
                    pipeline._stage8_final_quality
                )
                if assessment_gate_failed:
                    enhancement_report["assessment_gate_issues"] = list(
                        (locals().get("final_quality_record") or {}).get("issues")
                        or []
                    )
                if (
                    enhancement_report.get("status") == "poor"
                    or assessment_gate_failed
                ) and not limited_candidate_rolled_back:
                    if failure_action != "auto_fallback" and not stage8_policy_failure_recorded:
                        stage8_policy_failure_recorded = True
                        reason = ", ".join(
                            str(item) for item in enhancement_report.get("issues", [])[:3]
                        ) or "stage8_enhancement_quality_gate_failed"
                        if hasattr(pipeline, "_record_stage_policy_event"):
                            pipeline._record_stage_policy_event(
                                8,
                                event="decisive_failure",
                                reason=reason,
                                source="quality_gate",
                            )
                        if failure_action == "preserve_review":
                            pipeline._require_review(
                                8, "stage8_enhancement_quality_gate_failed"
                            )
                            status = "degraded"
                        elif failure_action == "stop":
                            status = "failed"
                    rollback_ok = pipeline._rollback_stage8_to_input()
                    enhancement_report["rollback"] = {
                        "target": "stage8_input_starless",
                        "status": "ok" if rollback_ok else "failed",
                    }
                    messages.append(
                        "stage8 enhancement quality gate rollback "
                        f"status={enhancement_report['rollback']['status']}"
                    )
                    if rollback_ok:
                        if not stage8_limited_mode:
                            status = "degraded" if status == "ok" else status
                        pipeline._stage8_final_source = "stage8_input_starless"
                        pipeline._stage8_fallback_used = True
                        pipeline._stage8_final_quality = (
                            "limited_candidate_rejected"
                            if stage8_limited_mode
                            else "poor"
                        )
                        limited_candidate_rolled_back = stage8_limited_mode
                    else:
                        status = "degraded"
                enhancement_report["final_source"] = pipeline._stage8_final_source
                enhancement_report["input_source"] = stage8_input_source
                enhancement_report["input_fallback_used"] = bool(
                    pipeline._stage8_input_fallback_used
                )
                enhancement_report["fallback_used"] = pipeline._stage8_fallback_used
                enhancement_report["final_quality"] = pipeline._stage8_final_quality
                if enhancement_report.get("issues"):
                    stage8_reject_reason = ", ".join(str(x) for x in enhancement_report.get("issues", [])[:3])
                enhancement_report["stage8_initial_quality"] = stage8_initial_quality
                enhancement_report["stage8_reject_reason"] = stage8_reject_reason or None
    elif status == 'ok':
        status = 'degraded'
        messages.append("stage8_enhanced 输出保存失败")
    if pipeline._stage8_final_quality == "unknown":
        pipeline._stage8_final_quality = "ok" if status == "ok" else status
    safe_passthrough_preflight: Dict[str, Any] = {
        "schema": "starun.stage8-safe-passthrough-preflight.v1",
        "status": "not_applicable",
        "accepted": False,
        "route": "safe_passthrough_color_only",
    }
    limited_safe_passthrough_eligibility = (
        _stage8_limited_safe_passthrough_eligibility(
            pipeline,
            stage8_guard_report=stage8_guard_report,
            final_source=str(pipeline._stage8_final_source),
            final_quality=str(pipeline._stage8_final_quality),
            user_processing_mode=user_processing_mode,
            external_override=bool(external_starless_source),
        )
        if stage8_limited_mode
        else {
            "schema": (
                "starun.stage8-limited-safe-passthrough-eligibility.v1"
            ),
            "status": "not_applicable",
            "accepted": False,
            "checks": {},
            "issues": [],
        }
    )
    full_rollback_candidate = bool(
        pipeline._stage8_final_source == "stage8_input_starless"
        and not stage8_limited_mode
        and resolved_processing_policy == "full"
        and user_processing_mode == "auto"
        and not bool(external_starless_source)
        and not _stage8_review_requirements_through_stage8(pipeline)
    )
    limited_safe_candidate = bool(
        limited_safe_passthrough_eligibility.get("accepted", False)
    )
    if full_rollback_candidate or limited_safe_candidate:
        safe_passthrough_preflight = _stage8_safe_passthrough_preflight(
            pipeline,
            source_mode=(
                "limited_safe_passthrough"
                if limited_safe_candidate
                else "structure_rollback"
            ),
            eligibility=(
                limited_safe_passthrough_eligibility
                if limited_safe_candidate
                else None
            ),
        )
        if bool(safe_passthrough_preflight.get("accepted", False)):
            messages.append(
                "Stage8 exact structure identity passed independent preflight; "
                + (
                    "limited advisory safe passthrough enabled"
                    if limited_safe_candidate
                    else "bounded color-only route enabled"
                )
            )
        else:
            messages.append(
                "Stage8 color-only passthrough prohibited: "
                + ",".join(
                    str(item)
                    for item in safe_passthrough_preflight.get("issues", [])[:3]
                )
            )
    stage8_starless_finish_report = _stage8_run_starless_finish(
        pipeline,
        messages,
        base_stem=str(pipeline._stage8_final_source or "stage8_enhanced"),
        channel_semantics=channel_semantics,
        processing_policy=(
            "limited" if stage8_limited_mode else resolved_processing_policy
        ),
        user_processing_mode=user_processing_mode,
        external_override=bool(external_starless_source),
        vectra_route_selected=vectra_route_selected,
        effective_saturation_budget=effective_stage8_saturation,
    )
    if stage8_starless_finish_report.get("status") == "failed_rollback_failed":
        status = "degraded"
        pipeline._stage8_final_quality = "starless_finish_rollback_failed"
        pipeline._stage8_fallback_used = True
        if hasattr(pipeline, "_require_review"):
            pipeline._require_review(8, "stage8_starless_finish_rollback_failed")
    stage8_subject_chroma_report = _stage8_run_subject_chroma(
        pipeline,
        messages,
        base_stem=str(pipeline._stage8_final_source or "stage8_enhanced"),
        channel_semantics=channel_semantics,
        processing_policy=(
            "limited" if stage8_limited_mode else resolved_processing_policy
        ),
        user_processing_mode=user_processing_mode,
        external_override=bool(external_starless_source),
        requested_saturation_budget=float(
            getattr(pipeline.cfg, "nebula_saturation", requested_saturation)
        ),
        effective_saturation_budget=effective_stage8_saturation,
        generic_saturation_suppressed=positive_chroma_budget_reserved,
        vectra_route_selected=vectra_route_selected,
    )
    if (
        stage8_subject_chroma_report.get("status")
        in {"failed", "failed_rollback_failed", "rejected_rollback_failed"}
        and not bool(
            (stage8_subject_chroma_report.get("transaction") or {}).get(
                "rollback_ok", False
            )
        )
    ):
        status = "degraded"
    stage8_palette_report = _stage8_run_dualband_palette(
        pipeline,
        messages,
        base_stem=str(pipeline._stage8_final_source or "stage8_enhanced"),
        channel_semantics=channel_semantics,
        processing_policy=(
            "limited" if stage8_limited_mode else resolved_processing_policy
        ),
        external_override=bool(external_starless_source),
    )
    if (
        stage8_palette_report.get("status")
        in {"failed", "failed_rollback_failed"}
        and not bool(
            (stage8_palette_report.get("transaction") or {}).get(
                "rollback_ok", False
            )
        )
    ):
        status = "degraded"
    safe_passthrough_final: Dict[str, Any] = {
        "schema": "starun.stage8-safe-passthrough-final.v1",
        "status": "not_applicable",
        "accepted": False,
        "route": "safe_passthrough_color_only",
    }
    post_cumulative_safe_passthrough_used = False
    structure_cumulative_rejection: Dict[str, Any] = {}
    if bool(safe_passthrough_preflight.get("accepted", False)):
        safe_passthrough_final = _stage8_safe_passthrough_final_validation(
            pipeline,
            subject_chroma_report=stage8_subject_chroma_report,
            palette_report=stage8_palette_report,
        )
        if bool(safe_passthrough_final.get("accepted", False)):
            pipeline._stage8_final_quality = "safe_passthrough_color_only"
            pipeline._stage8_final_source = "stage8_enhanced"
            pipeline._stage8_fallback_used = True
            messages.append(
                "Stage8 safe_passthrough_color_only passed all independent gates"
            )
        else:
            # A failed color-only candidate is never allowed to weaken the
            # exact structure rollback.  Restore that canonical input and keep
            # the handoff review-only.
            rollback_ok = pipeline._rollback_stage8_to_input()
            pipeline._stage8_final_source = "stage8_input_starless"
            pipeline._stage8_final_quality = "poor"
            pipeline._stage8_fallback_used = True
            safe_passthrough_final["rollback"] = {
                "status": "restored" if rollback_ok else "failed",
                "source": "stage8_input_starless",
            }
            if hasattr(pipeline, "_require_review"):
                pipeline._require_review(
                    8,
                    "stage8_safe_passthrough_color_only_rejected",
                )
            status = "degraded"
    final_cumulative_quality: Dict[str, Any] = {
        "schema": "starun.stage8-final-cumulative-quality.v1",
        "status": "not_applicable",
        "accepted": False,
        "evaluation_scope": (
            "stage8_input_starless_to_final_persisted_pixels"
        ),
        "fresh_evaluation": False,
        "issues": [],
    }
    final_cumulative_candidate = bool(
        pipeline._stage8_final_source == "stage8_enhanced"
        and not _stage8_review_requirements_through_stage8(pipeline)
        and (
            bool(safe_passthrough_final.get("accepted", False))
            or (
                not stage8_limited_mode
                and pipeline._stage8_final_quality == "ok"
            )
        )
    )
    if final_cumulative_candidate:
        structure_candidate_cumulative = bool(
            not safe_passthrough_final.get("accepted", False)
            and not stage8_limited_mode
            and pipeline._stage8_final_quality == "ok"
        )
        final_cumulative_quality = (
            _stage8_enforce_final_cumulative_validation(
                pipeline,
                subject_chroma_report=stage8_subject_chroma_report,
                palette_report=stage8_palette_report,
                starless_finish_report=stage8_starless_finish_report,
                defer_review_on_exact_rollback=structure_candidate_cumulative,
            )
        )
        if bool(final_cumulative_quality.get("accepted", False)):
            messages.append(
                "Stage8 final cumulative pixels passed fresh full QA"
            )
        else:
            status = "degraded"
            messages.append(
                "Stage8 final cumulative pixels rejected; "
                + str(
                    (
                        final_cumulative_quality.get("rollback") or {}
                    ).get("status")
                    or "review_only"
                )
                + ": "
                + ",".join(
                    str(item)
                    for item in final_cumulative_quality.get("issues", [])[:3]
                )
            )
            retry_eligible = bool(
                structure_candidate_cumulative
                and (
                    final_cumulative_quality.get("rollback") or {}
                ).get("accepted", False)
                and final_cumulative_quality.get(
                    "review_deferred_for_safe_passthrough", False
                )
                and pipeline._stage8_final_source == "stage8_input_starless"
                and resolved_processing_policy == "full"
                and user_processing_mode == "auto"
                and not bool(external_starless_source)
                and not _stage8_review_requirements_through_stage8(pipeline)
            )
            if retry_eligible:
                structure_cumulative_rejection = dict(
                    final_cumulative_quality
                )
                pipeline._write_stage_json(
                    "stage8_structure_cumulative_quality.json",
                    structure_cumulative_rejection,
                )
                safe_passthrough_preflight = (
                    _stage8_safe_passthrough_preflight(
                        pipeline,
                        source_mode="post_cumulative_structure_rollback",
                    )
                )
                if bool(safe_passthrough_preflight.get("accepted", False)):
                    # The rejected structure/plugin delta is gone.  Re-run
                    # only the bounded internal color transaction from the
                    # exact Stage7 pixels; do not re-enter any plugin.
                    stage8_starless_finish_report = (
                        _stage8_run_starless_finish(
                            pipeline,
                            messages,
                            base_stem="stage8_input_starless",
                            channel_semantics=channel_semantics,
                            processing_policy=resolved_processing_policy,
                            user_processing_mode=user_processing_mode,
                            external_override=bool(external_starless_source),
                            vectra_route_selected=vectra_route_selected,
                            effective_saturation_budget=(
                                effective_stage8_saturation
                            ),
                        )
                    )
                    stage8_subject_chroma_report = (
                        _stage8_run_subject_chroma(
                            pipeline,
                            messages,
                            base_stem="stage8_input_starless",
                            channel_semantics=channel_semantics,
                            processing_policy=resolved_processing_policy,
                            user_processing_mode=user_processing_mode,
                            external_override=bool(external_starless_source),
                            requested_saturation_budget=float(
                                getattr(
                                    pipeline.cfg,
                                    "nebula_saturation",
                                    requested_saturation,
                                )
                            ),
                            effective_saturation_budget=(
                                effective_stage8_saturation
                            ),
                            generic_saturation_suppressed=(
                                positive_chroma_budget_reserved
                            ),
                            vectra_route_selected=vectra_route_selected,
                        )
                    )
                    stage8_palette_report = _stage8_run_dualband_palette(
                        pipeline,
                        messages,
                        base_stem="stage8_input_starless",
                        channel_semantics=channel_semantics,
                        processing_policy=resolved_processing_policy,
                        external_override=bool(external_starless_source),
                    )
                    safe_passthrough_final = (
                        _stage8_safe_passthrough_final_validation(
                            pipeline,
                            subject_chroma_report=(
                                stage8_subject_chroma_report
                            ),
                            palette_report=stage8_palette_report,
                        )
                    )
                    if bool(safe_passthrough_final.get("accepted", False)):
                        pipeline._stage8_final_source = "stage8_enhanced"
                        pipeline._stage8_final_quality = (
                            "safe_passthrough_color_only"
                        )
                        pipeline._stage8_fallback_used = True
                        final_cumulative_quality = (
                            _stage8_enforce_final_cumulative_validation(
                                pipeline,
                                subject_chroma_report=(
                                    stage8_subject_chroma_report
                                ),
                                palette_report=stage8_palette_report,
                                starless_finish_report=(
                                    stage8_starless_finish_report
                                ),
                            )
                        )
                        post_cumulative_safe_passthrough_used = bool(
                            final_cumulative_quality.get("accepted", False)
                        )
                        if post_cumulative_safe_passthrough_used:
                            messages.append(
                                "Stage8 cumulative structure rejection "
                                "recovered through exact rollback and "
                                "independently gated color-only passthrough"
                            )
                    else:
                        rollback_ok = pipeline._rollback_stage8_to_input()
                        pipeline._stage8_final_source = "stage8_input_starless"
                        pipeline._stage8_final_quality = "poor"
                        safe_passthrough_final["rollback"] = {
                            "status": "restored" if rollback_ok else "failed",
                            "source": "stage8_input_starless",
                        }
                        if hasattr(pipeline, "_require_review"):
                            pipeline._require_review(
                                8,
                                "stage8_safe_passthrough_color_only_rejected",
                            )
                else:
                    if hasattr(pipeline, "_require_review"):
                        pipeline._require_review(
                            8,
                            "stage8_safe_passthrough_color_only_rejected",
                        )
    else:
        pipeline._stage8_final_cumulative_quality_report = dict(
            final_cumulative_quality
        )
        pipeline._write_stage_json(
            "stage8_final_cumulative_quality.json",
            final_cumulative_quality,
        )
    stage8_starless_finish_report.update(
        final_source=f"{pipeline._stage8_final_source}.fit",
        final_source_identity=_stage8_source_identity(
            pipeline,
            pipeline._stage8_final_source,
        ),
        downstream_color_route={
            "palette_accepted": bool(stage8_palette_report.get("accepted", False)),
            "vectra_selected": bool(vectra_route_selected),
            "vectra_accepted": bool(
                getattr(pipeline, "_stage8_vectra_applied", False)
            ),
            "subject_chroma_accepted": bool(
                stage8_subject_chroma_report.get("accepted", False)
            ),
        },
    )
    pipeline._stage8_starless_finish_report = dict(
        stage8_starless_finish_report
    )
    pipeline._write_stage_json(
        "stage8_starless_finish_report.json",
        stage8_starless_finish_report,
    )
    source_stem_passthrough = (
        pipeline._stage8_final_source == "stage8_input_starless"
    )
    spatial_lineage_verified = bool(
        stage3_spatial_lineage.get("accepted", False)
        and stage3_spatial_lineage.get("support_sha256")
    )
    review_requirement_free = not _stage8_review_requirements_through_stage8(
        pipeline
    )
    safe_passthrough_formal = bool(
        safe_passthrough_final.get("accepted", False)
        and final_cumulative_quality.get("accepted", False)
        and pipeline._stage8_final_source == "stage8_enhanced"
        and review_requirement_free
    )
    stage8_passthrough = bool(source_stem_passthrough or safe_passthrough_formal)
    structure_formal = bool(
        not stage8_limited_mode
        and not source_stem_passthrough
        and pipeline._stage8_final_quality == "ok"
        and final_cumulative_quality.get("accepted", False)
        and review_requirement_free
    )
    processing_route = (
        "safe_passthrough_color_only"
        if safe_passthrough_formal
        else "structure_enhanced"
        if structure_formal
        else "review_only"
    )
    formal_eligible = bool(
        spatial_lineage_verified
        and (safe_passthrough_formal or structure_formal)
    )
    final_reason_code = ""
    final_reason_text = ""
    if str(final_cumulative_quality.get("status") or "") == "rejected":
        final_reason_code = str(
            final_cumulative_quality.get("reason_code")
            or "stage8_final_cumulative_qa_rejected"
        )
        final_reason_text = final_reason_code
    elif post_cumulative_safe_passthrough_used:
        final_reason_code = (
            "stage8_structure_cumulative_rollback_"
            "safe_passthrough_accepted"
        )
        final_reason_text = final_reason_code
    elif stage8_passthrough and not str(
        (getattr(pipeline, "_stage8_handoff", {}) or {}).get("reason_code") or ""
    ):
        final_reason_code = "stage8_enhancement_quality_rollback"
        final_reason_text = "stage8_enhancement_quality_rollback"
    handoff = _set_stage8_handoff(
        pipeline,
        source_stem=pipeline._stage8_final_source,
        passthrough=stage8_passthrough,
        restricted_downstream=not formal_eligible,
        final_quality=pipeline._stage8_final_quality,
        processing_route=processing_route,
        formal_eligible=formal_eligible,
        reason_code=final_reason_code,
        reason_text=final_reason_text,
    )
    halo_guard_report = dict(
        getattr(pipeline, "_stage8_star_halo_guard_report", {}) or {}
    )
    if halo_guard_report:
        handoff["star_halo_guard"] = {
            "status": halo_guard_report.get("status"),
            "reason_code": halo_guard_report.get("reason_code"),
            "report": star_halo_guard.REPORT_NAME,
            "artifact": halo_guard_report.get("artifact"),
            "artifact_sha256": halo_guard_report.get("artifact_sha256"),
            "verified": bool(
                getattr(pipeline, "_stage8_star_halo_guard_verified", False)
            ),
        }
    handoff["spatial_background_lineage"] = {
        "schema": stage3_spatial_lineage.get("schema"),
        "status": stage3_spatial_lineage.get("status"),
        "accepted": bool(stage3_spatial_lineage.get("accepted", False)),
        "support_sha256": stage3_spatial_lineage.get("support_sha256"),
        "issues": list(stage3_spatial_lineage.get("issues") or []),
    }
    handoff["outcome_reason_code"] = (
        str(
            final_cumulative_quality.get("reason_code")
            or "stage8_final_cumulative_qa_rejected"
        )
        if str(final_cumulative_quality.get("status") or "") == "rejected"
        else "stage8_structure_cumulative_rollback_safe_passthrough_accepted"
        if post_cumulative_safe_passthrough_used
        else "stage8_limited_safe_passthrough_accepted"
        if safe_passthrough_formal and stage8_limited_mode
        else "stage8_limited_candidate_rejected"
        if limited_candidate_rolled_back
        else "stage8_limited_candidate_accepted"
        if stage8_limited_mode and not stage8_passthrough
        else "stage8_enhancement_quality_rollback"
        if stage8_passthrough
        else ""
    )
    if bool(stage8_palette_report.get("accepted", False)):
        handoff.update(
            {
                "color_role": "artistic_false_color",
                "palette": stage8_palette_report.get("palette"),
                "synthetic_sii": bool(
                    stage8_palette_report.get("synthetic_sii", False)
                ),
                "physical_parent": "stage8_pre_palette.fit",
            }
        )
    elif bool(getattr(pipeline, "_stage8_vectra_applied", False)):
        handoff.update(
            {
                "color_role": "vectra_exclusive_color_route",
                "color_rendition": {
                    "mode": "vectra_exclusive_color_route",
                    "accepted": True,
                    "report": "stage8_starless_finish_report.json",
                },
            }
        )
    elif bool(stage8_subject_chroma_report.get("accepted", False)):
        handoff.update(
            {
                "color_role": "target_aware_subject_chroma",
                "color_rendition": {
                    "mode": "masked_subject_chroma_rendition",
                    "accepted": True,
                    "report": "stage8_subject_chroma_report.json",
                    "factor": (
                        stage8_subject_chroma_report.get("factor") or {}
                    ).get("factor"),
                },
            }
        )
    else:
        handoff["color_rendition"] = {
            "mode": str(
                stage8_subject_chroma_report.get("status")
                or "skipped"
            ),
            "accepted": False,
            "report": "stage8_subject_chroma_report.json",
        }
    handoff["subject_chroma"] = {
        "status": stage8_subject_chroma_report.get("status"),
        "accepted": bool(stage8_subject_chroma_report.get("accepted", False)),
        "report": "stage8_subject_chroma_report.json",
    }
    handoff["starless_finish"] = {
        "status": stage8_starless_finish_report.get("status"),
        "accepted": bool(stage8_starless_finish_report.get("accepted", False)),
        "accepted_steps": list(
            stage8_starless_finish_report.get("accepted_steps") or []
        ),
        "report": "stage8_starless_finish_report.json",
    }
    handoff["final_cumulative_quality"] = {
        "status": final_cumulative_quality.get("status"),
        "accepted": bool(final_cumulative_quality.get("accepted", False)),
        "fresh_evaluation": bool(
            final_cumulative_quality.get("fresh_evaluation", False)
        ),
        "report": "stage8_final_cumulative_quality.json",
        "issues": list(final_cumulative_quality.get("issues") or []),
        "rollback": dict(final_cumulative_quality.get("rollback") or {}),
    }
    handoff["structure_cumulative_retry"] = {
        "attempted": bool(structure_cumulative_rejection),
        "accepted": bool(post_cumulative_safe_passthrough_used),
        "source_mode": (
            "post_cumulative_structure_rollback"
            if structure_cumulative_rejection
            else None
        ),
        "report": (
            "stage8_structure_cumulative_quality.json"
            if structure_cumulative_rejection
            else None
        ),
        "initial_issues": list(
            structure_cumulative_rejection.get("issues") or []
        ),
    }
    pipeline._stage8_handoff = handoff
    saturation_execution = dict(
        getattr(pipeline, "_stage8_saturation_execution", {}) or {}
    )
    applied_saturation = (
        max(0.0, float(saturation_execution.get("applied_amount", 0.0) or 0.0))
        if bool(saturation_execution.get("applied", False))
        else 0.0
    )
    if pipeline._stage8_final_source == "stage8_input_starless":
        applied_saturation = 0.0
    pipeline._saturation_boost_applied = float(
        getattr(pipeline, "_saturation_boost_applied", 0.0)
    ) + applied_saturation
    color_quality_report = _write_stage8_color_quality_report(
        pipeline,
        final_source=str(pipeline._stage8_final_source),
        requested_saturation=requested_saturation,
        effective_saturation=effective_stage8_saturation,
        applied_saturation=applied_saturation,
        palette_report=stage8_palette_report,
        subject_chroma_report=stage8_subject_chroma_report,
        vectra_report=getattr(pipeline, "_stage8_vectra_report", {}) or {},
    )
    if (
        bool(color_quality_report.get("used_for_gate", False))
        and str(color_quality_report.get("status") or "") == "rejected"
    ):
        handoff["restricted_downstream"] = True
        handoff["formal_eligible"] = False
        handoff["final_quality"] = "stage8_color_or_halo_gate_rejected"
        handoff["reason_code"] = "stage8_color_or_halo_gate_rejected"
        handoff["reason_text"] = "stage8_color_or_halo_gate_rejected"
        pipeline._stage8_final_quality = "stage8_color_or_halo_gate_rejected"
        status = "degraded"
        if hasattr(pipeline, "_require_review"):
            pipeline._require_review(8, "stage8_color_or_halo_gate_rejected")
    handoff["color_contract"] = color_quality_report.get("contract")
    handoff["color_quality_report"] = "stage8_color_quality_report.json"
    handoff["color_gate"] = {
        "report": "stage8_color_quality_report.json",
        "used_for_gate": bool(color_quality_report.get("used_for_gate", False)),
        "status": color_quality_report.get("status"),
        "guard_lineage_verified": bool(
            (color_quality_report.get("guard_lineage") or {}).get("verified", False)
        ),
        "final_pixel_identity": dict(
            color_quality_report.get("final_pixel_identity") or {}
        ),
    }
    handoff["saturation_execution"] = saturation_execution
    handoff["safe_passthrough_color_only"] = {
        "limited_eligibility": limited_safe_passthrough_eligibility,
        "preflight": safe_passthrough_preflight,
        "final_validation": safe_passthrough_final,
    }
    if str(handoff.get("processing_route") or "") == "safe_passthrough_color_only":
        final_pixel_identity = dict(
            color_quality_report.get("final_pixel_identity") or {}
        )
        source_artifact = dict(handoff.get("source_artifact") or {})
        safe_color_gate_verified = bool(
            color_quality_report.get("used_for_gate", False)
            and str(color_quality_report.get("status") or "")
            in {"accepted", "ok", "reported"}
            and not list(color_quality_report.get("issues") or [])
            and bool(
                (color_quality_report.get("guard_lineage") or {}).get(
                    "verified", False
                )
            )
            and str(final_pixel_identity.get("sha256") or "")
            == str(source_artifact.get("sha256") or "")
            and str(final_pixel_identity.get("pixel_sha256") or "")
            == str(source_artifact.get("pixel_sha256") or "")
        )
        handoff["safe_passthrough_color_only"]["color_gate_verified"] = (
            safe_color_gate_verified
        )
        if not safe_color_gate_verified:
            handoff["restricted_downstream"] = True
            handoff["formal_eligible"] = False
            handoff["reason_code"] = "stage8_safe_passthrough_color_gate_unverified"
            handoff["reason_text"] = "stage8_safe_passthrough_color_gate_unverified"
            handoff["final_quality"] = (
                "stage8_safe_passthrough_color_gate_unverified"
            )
            pipeline._stage8_final_quality = (
                "stage8_safe_passthrough_color_gate_unverified"
            )
            status = "degraded"
            if hasattr(pipeline, "_require_review"):
                pipeline._require_review(
                    8,
                    "stage8_safe_passthrough_color_gate_unverified",
                )
    if bool(handoff.get("restricted_downstream", True)):
        handoff["formal_eligible"] = False
    handoff = _write_stage8_handoff(pipeline, handoff)
    enhancement_report_value = locals().get("enhancement_report")
    if not isinstance(enhancement_report_value, dict) or not enhancement_report_value:
        enhancement_report_value = {
            "stage": "stage8_nebula_enhancement",
            "status": status,
        }
    if isinstance(enhancement_report_value, dict) and enhancement_report_value:
        enhancement_report_value["final_quality"] = pipeline._stage8_final_quality
        enhancement_report_value["final_source"] = pipeline._stage8_final_source
        enhancement_report_value["fallback_used"] = pipeline._stage8_fallback_used
        enhancement_report_value["passthrough"] = stage8_passthrough
        enhancement_report_value["processing_policy"] = (
            "limited" if stage8_limited_mode else "full"
        )
        enhancement_report_value["processing_mode"] = user_processing_mode
        enhancement_report_value["failure_action"] = failure_action
        enhancement_report_value["quality_retry_max"] = int(
            getattr(pipeline.cfg, "stage8_quality_retry_max", 1) or 0
        )
        enhancement_report_value["substeps"] = {
            "target_aware_subject_chroma": bool(
                getattr(
                    pipeline.cfg,
                    "stage8_target_aware_chroma_enabled",
                    True,
                )
            ),
            "nebula_saturation": bool(
                getattr(pipeline.cfg, "stage8_nebula_saturation_enabled", True)
            ),
            "background_denoise": bool(
                getattr(pipeline.cfg, "stage8_background_denoise_enabled", True)
            ),
            "faint_nebula_boost": bool(
                getattr(pipeline.cfg, "stage8_faint_nebula_boost_enabled", True)
            ),
            "nebula_contrast": bool(
                getattr(pipeline.cfg, "stage8_nebula_contrast_enabled", True)
            ),
            "masked_unsharp": bool(
                getattr(pipeline.cfg, "stage8_masked_unsharp_enabled", True)
            ),
        }
        enhancement_report_value["dualband_palette"] = stage8_palette_report or None
        enhancement_report_value["subject_chroma"] = (
            stage8_subject_chroma_report
        )
        enhancement_report_value["starless_finish"] = (
            stage8_starless_finish_report
        )
        enhancement_report_value["color_quality"] = color_quality_report
        enhancement_report_value["final_cumulative_quality"] = (
            final_cumulative_quality
        )
        enhancement_report_value["handoff"] = handoff
        pipeline._write_stage_json(
            "stage8_enhancement_report.json",
            enhancement_report_value,
        )
    summary_fields = {
        "stage8_input_source": stage8_input_source,
        "stage8_input_fallback_used": str(
            bool(pipeline._stage8_input_fallback_used)
        ).lower(),
        "stage8_initial_quality": stage8_initial_quality,
        "stage8_final_quality": pipeline._stage8_final_quality,
        "stage8_final_source": pipeline._stage8_final_source,
        "stage8_fallback_used": str(pipeline._stage8_fallback_used).lower(),
        "stage8_processing_policy": (
            "limited" if stage8_limited_mode else "full"
        ),
        "stage8_processing_mode": user_processing_mode,
        "stage8_failure_action": failure_action,
        "stage8_passthrough": str(stage8_passthrough).lower(),
        "stage8_reason": str(handoff.get("reason_text") or "none"),
        "stage8_conservative_mode": str(bool(getattr(pipeline, "_stage8_conservative_mode", False))).lower(),
        "stage8_reject_reason": stage8_reject_reason or "none",
        "stage8_artistic_palette_applied": str(
            bool(stage8_palette_report.get("accepted", False))
        ).lower(),
        "stage8_artistic_palette": str(
            stage8_palette_report.get("palette") or "none"
        ),
        "stage8_subject_chroma_status": str(
            stage8_subject_chroma_report.get("status") or "unknown"
        ),
        "stage8_subject_chroma_applied": str(
            bool(stage8_subject_chroma_report.get("accepted", False))
        ).lower(),
        "stage8_starless_finish_status": str(
            stage8_starless_finish_report.get("status") or "unknown"
        ),
        "stage8_vectra_applied": str(
            bool(getattr(pipeline, "_stage8_vectra_applied", False))
        ).lower(),
        "channel_semantics": channel_semantics,
    }
    for key, value in summary_fields.items():
        line = f"{key}={value}"
        pipeline.log.info(f"[Stage8] {line}")
        messages.append(line)
    pipeline.starless_file = pipeline.process_dir / f"{pipeline._stage8_final_source}.fit"
    if hasattr(pipeline, "_create_stage_review_bundle"):
        stage8_candidates: List[Dict[str, Any]] = [
            {
                "name": "initial_enhancement",
                "stem": "stage8_enhanced",
                "status": stage8_initial_quality,
            }
        ]
        quality_payload_value = locals().get("quality_payload")
        if isinstance(quality_payload_value, dict):
            if stage8_limited_mode:
                stage8_candidates.append(
                    {
                        "name": "limited_candidate",
                        "stem": "stage8_limited_candidate",
                        "status": stage8_initial_quality,
                        "selected": not stage8_passthrough,
                    }
                )
            if quality_payload_value.get("color_correction"):
                stage8_candidates.append(
                    {
                        "name": "color_corrected",
                        "stem": "stage8_enhanced",
                        "status": str(
                            quality_payload_value["color_correction"].get("status", "unknown")
                        ),
                    }
                )
            if quality_payload_value.get("conservative_rerun"):
                rerun_record = quality_payload_value["conservative_rerun"]
                stage8_candidates.append(
                    {
                        "name": "conservative_rerun",
                        "stem": "stage8_conservative_enhanced",
                        "status": str(rerun_record.get("status", "unknown")),
                        "safe_saturation": rerun_record.get("safe_saturation"),
                    }
                )
        if bool(stage8_palette_report.get("accepted", False)):
            stage8_candidates.append(
                {
                    "name": "dualband_palette",
                    "stem": "stage8_palette_selected",
                    "status": str(
                        stage8_palette_report.get("status") or "accepted"
                    ),
                    "palette": stage8_palette_report.get("palette"),
                    "selected": False,
                }
            )
        if bool(stage8_subject_chroma_report.get("accepted", False)):
            stage8_candidates.append(
                {
                    "name": "target_aware_subject_chroma",
                    "stem": "stage8_subject_chroma_candidate",
                    "status": str(
                        stage8_subject_chroma_report.get("status")
                        or "accepted"
                    ),
                    "factor": (
                        stage8_subject_chroma_report.get("factor") or {}
                    ).get("factor"),
                    "selected": False,
                }
            )
        stage8_candidates.append(
            {
                "name": "final",
                "stem": pipeline._stage8_final_source,
                "status": pipeline._stage8_final_quality,
                "fallback_used": pipeline._stage8_fallback_used,
                "selected": True,
            }
        )
        review = pipeline._create_stage_review_bundle(
            "stage8_nebula_enhancement",
            "stage8_input_starless",
            pipeline._stage8_final_source,
            context={
                "input_source": stage8_input_source,
                "final_quality": pipeline._stage8_final_quality,
                "fallback_used": pipeline._stage8_fallback_used,
                "color_policy_limits": color_limits,
                "channel_semantics": channel_semantics,
                "dualband_palette": stage8_palette_report or None,
                "subject_chroma": stage8_subject_chroma_report,
                "handoff": handoff,
            },
            candidates=stage8_candidates,
            selected_candidate="final",
        )
        if review.get("report_path"):
            messages.append(f"review_bundle={review['report_path']}")

    elapsed = pipeline.log.stage_end(stage_label)
    pipeline._record_stage(
        stage_label,
        status,
        elapsed,
        "；".join(messages),
        execution="safe_passthrough" if stage8_passthrough else "completed",
        fallback_used=bool(
            pipeline._stage8_fallback_used and not stage8_passthrough
        ),
        reason_code=str(handoff.get("reason_code") or ""),
        details={
            "reason_text": str(handoff.get("reason_text") or ""),
            "stage8_handoff": handoff,
            "processing_mode": user_processing_mode,
            "failure_action": failure_action,
            "dualband_palette": stage8_palette_report or None,
            "subject_chroma": stage8_subject_chroma_report,
        },
        review_reasons=pipeline._stage_review_reasons(8),
    )
