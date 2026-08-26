"""Stage 9 star processing and remixing."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from models import PipelineStage, StarSeparationState
import stage7_quality
import stage7_stretch_metrics
import stage9_quality
import syqon_starless
import run_manifest
from stage7_pixel_domain import (
    STAGE7_FLOAT_DOMAIN_TOLERANCE,
    canonicalize_stage7_pixels_01,
)
from star_color_repair import (
    assess_repaired_star_layer,
    public_star_color_report,
    repair_star_layer_colors,
)
from sirilpy.exceptions import CommandError, SirilError


def _clamp_float(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, float(value))))


def _stage9_signed_task_master_source(
    pipeline,
) -> tuple[Optional[Path], str]:
    """Resolve the immutable original master even during Stage 5 resume."""
    configured = str(os.getenv("STARUN_TASK_RUN_MANIFEST", "") or "").strip()
    if not configured:
        return None, "signed task-run manifest is not configured"
    manifest_path = Path(configured).expanduser().resolve()
    work_dir = Path(getattr(pipeline, "work_dir", manifest_path.parent)).resolve()
    if manifest_path.parent != work_dir:
        return None, "signed task-run manifest is outside the current run"
    payload = run_manifest.load_json(manifest_path)
    if not isinstance(payload, Mapping) or payload.get("schema") != (
        "starun.task-run.v1"
    ):
        return None, "signed task-run manifest is unavailable or unsupported"
    expected_hash = str(payload.get("manifest_hash") or "")
    unsigned = dict(payload)
    unsigned.pop("manifest_hash", None)
    if not expected_hash or expected_hash != run_manifest.canonical_payload_hash(
        unsigned
    ):
        return None, "signed task-run manifest hash is invalid"
    source = payload.get("source")
    if not isinstance(source, Mapping) or source.get("read_only") is not True:
        return None, "signed task source is not immutable"
    if str(source.get("kind") or "").strip().lower() != "master_file":
        return None, "task source is not a single master image"
    files = source.get("files")
    if not isinstance(files, list) or len(files) != 1:
        return None, "signed master source record is not singular"
    record = files[0]
    if not isinstance(record, Mapping):
        return None, "signed master source record is invalid"
    path = Path(str(record.get("path") or "")).expanduser().resolve()
    if not path.is_file():
        return None, "signed master source file is missing"
    try:
        if int(record.get("size")) != int(path.stat().st_size):
            return None, "signed master source size changed"
    except (OSError, TypeError, ValueError):
        return None, "signed master source size is invalid"
    expected_sha256 = str(record.get("sha256") or "")
    actual_sha256 = run_manifest.sha256_file(path)
    if not expected_sha256 or actual_sha256 != expected_sha256:
        return None, "signed master source SHA-256 changed"
    return path, "verified signed immutable master source"


def _stage9_upstream_handoff(pipeline, source_stem: str) -> Dict[str, Any]:
    handoff = dict(getattr(pipeline, "_stage8_handoff", {}) or {})
    if not handoff:
        passthrough_sources = {
            "stage8_input_starless",
            "stage8_review_with_stars",
            "stage7_review_with_stars",
            "stage6_passthrough",
        }
        handoff = {
            "schema": "starun.stage8-handoff.v1",
            "source_stem": source_stem,
            "passthrough": source_stem in passthrough_sources,
            "restricted_downstream": True,
            "reason_code": "stage8_handoff_missing",
        }
    return handoff


def _stage9_local_fallback(
    mode: str,
    selected: Optional[Dict[str, Any]],
    application_mode: str,
) -> tuple[bool, Optional[str]]:
    normalized_mode = str(mode or "").strip().lower()
    normalized_application = str(application_mode or "").strip().lower()
    review_mode_reasons = {
        "best_failed_review_candidate": "best_failed_candidate_review",
        "stage8_starmask_review_fallback": (
            "stage8_starmask_review_fallback"
        ),
        "stage5_review_fallback": "all_remix_candidates_outside_review_range_stage5",
    }
    if normalized_mode in review_mode_reasons:
        return True, review_mode_reasons[normalized_mode]
    mode_reasons = {
        "unsafe_starless_bypass": "unsafe_starless_bypass",
        "with_stars_review_fallback": "required_stars_preserved_in_review_fallback",
        "required_stars_output_withheld": "required_stars_output_withheld",
        "rejected_keep_starless": "all_remix_candidates_rejected",
        "starmask_preparation_failed": "starmask_preparation_failed_keep_upstream",
        "starmask_stretch_failed": "starmask_stretch_failed_keep_upstream",
    }
    if normalized_mode in mode_reasons:
        return True, mode_reasons[normalized_mode]
    if normalized_application in {"screen_save_failed", "starcomposer_save_failed"}:
        return True, "output_save_failed_keep_upstream"
    return False, None


def _stage9_required_stars_output_mode(saved: bool) -> str:
    return (
        "with_stars_review_fallback"
        if saved
        else "required_stars_output_withheld"
    )


def _stage9_preserve_with_stars_review_output(
    pipeline,
    messages: List[str],
    *,
    reason: str,
    prefer_stage5: bool = False,
) -> tuple[bool, Optional[str]]:
    """Build a review-only final source that is guaranteed to retain stars."""
    nonlinear_sources = (
        "stage8_review_with_stars",
        "stage7_review_with_stars",
    )
    linear_sources = tuple(
        dict.fromkeys(
            str(source)
            for source in (
                getattr(pipeline, "_stage6_passthrough_source", None),
                "stage6_passthrough",
                "stage5_linear",
                "stage1_prepared",
                "working",
            )
            if source
        )
    )
    load_errors: List[str] = []
    source_candidates = [
        *((stem, False) for stem in nonlinear_sources),
        *((stem, True) for stem in linear_sources),
    ]
    if prefer_stage5:
        source_candidates = [
            ("stage5_linear", True),
            *(
                item
                for item in source_candidates
                if item[0] != "stage5_linear"
            ),
        ]
    for source_stem, needs_display_stretch in source_candidates:
        source_path = (
            pipeline.process_dir / f"{source_stem}.fit"
            if getattr(pipeline, "process_dir", None)
            else None
        )
        if source_path is None or not source_path.is_file():
            continue
        try:
            pipeline.cmd_with_check("load", source_stem)
            if needs_display_stretch:
                pipeline.cmd_with_check("autostretch", "-linked")
            review_saved = pipeline._save_stage_output(
                "stage9_review_with_stars"
            )
            if not review_saved:
                load_errors.append(f"{source_stem}: review save failed")
                continue
            canonical_saved = False
            try:
                canonical_saved = bool(
                    pipeline._save_stage_output("stage9_remixed")
                )
            except (CommandError, RuntimeError, SirilError) as error:
                load_errors.append(
                    f"{source_stem}: canonical save failed: {error}"
                )
            pipeline._stage9_output_contains_stars = True
            pipeline._stage9_output_withheld = False
            pipeline._stage9_final_source = "stage9_review_with_stars"
            pipeline._stage9_stars_applied = False
            pipeline._stage9_remix_formally_accepted = False
            pipeline._stage9_review_candidate_selected = False
            pipeline._stage9_stars_application_mode = (
                "with_stars_review_fallback"
            )
            pipeline._stage9_bypassed_bad_starless = True
            if str(reason) != "user_preserve_with_stars":
                pipeline._require_review(
                    9,
                    "with_stars_review_fallback",
                    {"reason": str(reason), "source": source_stem},
                )
            messages.append(
                "required-stars contract preserved via with-stars review "
                f"source={source_stem}; canonical_saved="
                f"{str(bool(canonical_saved)).lower()}; reason={reason}"
            )
            return True, source_stem
        except (CommandError, RuntimeError, SirilError) as error:
            load_errors.append(f"{source_stem}: {error}")

    pipeline._stage9_output_contains_stars = False
    pipeline._stage9_output_withheld = True
    pipeline._stage9_final_source = ""
    pipeline._stage9_stars_applied = False
    pipeline._stage9_remix_formally_accepted = False
    pipeline._stage9_review_candidate_selected = False
    pipeline._stage9_stars_application_mode = (
        "withheld_no_with_stars_review_source"
    )
    messages.append(
        "required-stars output withheld because no with-stars review source "
        f"could be produced; reason={reason}"
        + (f"; errors={' | '.join(load_errors)}" if load_errors else "")
    )
    return False, None


def _stage9_existing_fit_path(pipeline, stem: str) -> Optional[Path]:
    process_dir = getattr(pipeline, "process_dir", None)
    normalized = str(stem or "").strip()
    if process_dir is None or not normalized:
        return None
    return next(
        (
            Path(process_dir) / f"{normalized}{suffix}"
            for suffix in (".fit", ".fits", ".fts")
            if (Path(process_dir) / f"{normalized}{suffix}").is_file()
        ),
        None,
    )


def _stage9_current_canonical_pixels(
    pipeline,
    *,
    label: str,
) -> tuple[np.ndarray, Dict[str, Any]]:
    get_pixels = getattr(pipeline.siril, "get_image_pixeldata", None)
    if not callable(get_pixels):
        raise RuntimeError(f"{label} pixel reader unavailable")
    pixels = get_pixels(preview=False)
    if pixels is None:
        raise RuntimeError(f"{label} image buffer is empty")
    try:
        return canonicalize_stage7_pixels_01(pixels)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{label} pixel domain invalid: {error}") from error


def _stage9_fallback_starmask_shape_compatible(
    starmask: np.ndarray,
    base: np.ndarray,
) -> bool:
    """Match the channel-layout compatibility accepted by the Screen composer."""
    if starmask.shape == base.shape:
        return True
    if base.ndim == 3 and starmask.ndim == 2:
        return starmask.shape == base.shape[1:]
    if base.ndim == 2 and starmask.ndim == 3:
        return starmask.shape[1:] == base.shape
    if base.ndim == 3 and starmask.ndim == 3:
        return bool(
            starmask.shape[1:] == base.shape[1:]
            and (
                (starmask.shape[0] == 1 and base.shape[0] == 3)
                or (starmask.shape[0] == 3 and base.shape[0] == 1)
            )
        )
    return False


def _stage9_minimal_fallback_safety(
    pipeline,
    base: np.ndarray,
    candidate: np.ndarray,
    *,
    base_domain: Mapping[str, Any],
    candidate_domain: Mapping[str, Any],
) -> Dict[str, Any]:
    """Apply only fatal pixel checks to a review-only Stage 9 fallback."""
    tolerance = max(
        float(STAGE7_FLOAT_DOMAIN_TOLERANCE),
        float(base_domain.get("float_tolerance", 0.0) or 0.0),
        float(candidate_domain.get("float_tolerance", 0.0) or 0.0),
    )
    report: Dict[str, Any] = {
        "schema": "starun.stage9-minimal-fallback-safety.v1",
        "status": "failed",
        "tolerance": tolerance,
        "checks": {},
        "issues": [],
    }
    checks = report["checks"]
    issues = report["issues"]
    same_shape = base.shape == candidate.shape
    checks["shape_compatible"] = bool(same_shape)
    checks["base_shape"] = [int(value) for value in base.shape]
    checks["candidate_shape"] = [int(value) for value in candidate.shape]
    if not same_shape:
        issues.append("candidate shape differs from Stage8 fallback base")
        return report

    finite = bool(np.all(np.isfinite(base)) and np.all(np.isfinite(candidate)))
    checks["finite_pixels"] = finite
    if not finite:
        issues.append("base or candidate contains non-finite pixels")
        return report

    candidate_min = float(np.min(candidate))
    candidate_max = float(np.max(candidate))
    in_range = bool(
        candidate_min >= -tolerance and candidate_max <= 1.0 + tolerance
    )
    checks.update(
        normalized_range=in_range,
        candidate_min=candidate_min,
        candidate_max=candidate_max,
    )
    if not in_range:
        issues.append("candidate pixels are outside normalized 0..1 range")

    delta = candidate.astype(np.float64) - base.astype(np.float64)
    min_delta = float(np.min(delta))
    monotonic_screen = min_delta >= -tolerance
    checks["screen_non_darkening"] = bool(monotonic_screen)
    checks["minimum_delta"] = min_delta
    if not monotonic_screen:
        issues.append("Screen fallback produced a material negative delta")

    delta_peak = np.max(delta, axis=0) if delta.ndim == 3 else delta
    positive = delta_peak > tolerance
    positive_count = int(np.count_nonzero(positive))
    checks["positive_delta_pixel_count"] = positive_count
    checks["maximum_positive_delta"] = float(np.max(delta_peak))

    support = getattr(pipeline, "_stage9_last_star_overlay_mask", None)
    support_available = isinstance(support, np.ndarray) and support.size > 0
    if support_available:
        support_mask = np.asarray(support, dtype=bool)
        if support_mask.ndim == 3:
            support_mask = np.any(support_mask, axis=0)
        support_shape_ok = support_mask.shape == delta_peak.shape
        checks["star_support_shape_compatible"] = bool(support_shape_ok)
        if support_shape_ok:
            supported_positive_count = int(
                np.count_nonzero(positive & support_mask)
            )
        else:
            supported_positive_count = 0
            issues.append("star support shape differs from fallback candidate")
        checks["supported_positive_delta_pixel_count"] = (
            supported_positive_count
        )
        real_star_delta = bool(support_shape_ok and supported_positive_count > 0)
    else:
        star_layer = getattr(pipeline, "_stage9_last_star_layer", None)
        star_layer_array = (
            np.asarray(star_layer)
            if isinstance(star_layer, np.ndarray) and star_layer.size
            else None
        )
        star_layer_peak = (
            float(np.max(star_layer_array))
            if star_layer_array is not None
            and np.all(np.isfinite(star_layer_array))
            else 0.0
        )
        checks["star_layer_peak"] = star_layer_peak
        star_layer_support = (
            np.max(star_layer_array, axis=0) > tolerance
            if star_layer_array is not None and star_layer_array.ndim == 3
            else star_layer_array > tolerance
            if star_layer_array is not None and star_layer_array.ndim == 2
            else None
        )
        star_layer_shape_ok = bool(
            isinstance(star_layer_support, np.ndarray)
            and star_layer_support.shape == delta_peak.shape
        )
        checks["star_layer_support_shape_compatible"] = star_layer_shape_ok
        supported_positive_count = int(
            np.count_nonzero(positive & star_layer_support)
        ) if star_layer_shape_ok else 0
        checks["supported_positive_delta_pixel_count"] = (
            supported_positive_count
        )
        real_star_delta = bool(star_layer_shape_ok and supported_positive_count > 0)
    checks["real_star_delta"] = real_star_delta
    if not real_star_delta:
        issues.append("fallback did not add measurable signal on a real star layer")

    if not issues:
        report["status"] = "passed"
    return report


def _stage9_try_stage8_starmask_review_fallback(
    pipeline,
    messages: List[str],
    remix_attempts: List[Dict[str, Any]],
    *,
    trigger_reason: str,
    stage8_source_stem: str,
    raw_starmask_stem: str,
    intensity: float,
    allow_stretch: bool = True,
) -> tuple[bool, Optional[Dict[str, Any]]]:
    """Try Stage8 + stretch-only/raw starmask before any Stage5 fallback."""
    fallback_report: Dict[str, Any] = {
        "schema": "starun.stage9-fallback-remix.v1",
        "status": "failed",
        "trigger_reason": str(trigger_reason or "stage9_remix_rejected"),
        "base_source_stem": str(stage8_source_stem or ""),
        "raw_starmask_source_stem": str(raw_starmask_stem or ""),
        "candidate_order": ["stretch_only", "raw"],
        "intensity": float(intensity),
        "attempts": [],
        "selected_variant": None,
    }
    pipeline._stage9_fallback_remix_report = fallback_report
    base_path = _stage9_existing_fit_path(pipeline, stage8_source_stem)
    raw_path = _stage9_existing_fit_path(pipeline, raw_starmask_stem)
    if base_path is None:
        fallback_report["issues"] = ["Stage8 fallback base is unavailable"]
        return False, None
    if raw_path is None:
        fallback_report["issues"] = ["raw Stage9 starmask is unavailable"]
        return False, None

    variants: List[Dict[str, Any]] = []
    stretch_enabled = bool(
        getattr(pipeline.cfg, "stage9_starmask_stretch_enabled", True)
    )
    if allow_stretch and stretch_enabled:
        stretched_name = "starmask_fallback_stretched"
        prepared_name = _prepare_stage9_starmask_for_pixel_remix(
            pipeline,
            raw_starmask_stem,
            star_stretch_used=False,
            messages=messages,
            strict_support=False,
            output_name=stretched_name,
            precomputed_calibration=None,
            candidate_local=True,
            allow_pre_stretch_compact=False,
        )
        calibration = copy.deepcopy(
            getattr(pipeline, "_stage9_starmask_calibration", {}) or {}
        )
        prepared = bool(
            prepared_name == stretched_name
            and calibration.get("stretch_applied") is True
            and str(calibration.get("status") or "")
            not in {"failed", "rejected", "unavailable"}
        )
        if prepared:
            variants.append(
                {
                    "variant": "stretch_only",
                    "starmask": stretched_name,
                    "calibration": calibration,
                }
            )
        else:
            fallback_report["attempts"].append(
                {
                    "variant": "stretch_only",
                    "phase": "prepare",
                    "status": "unavailable",
                    "starmask": stretched_name,
                    "reason": str(
                        calibration.get("reason")
                        or "controlled stretch was not produced"
                    ),
                    "calibration_status": calibration.get("status"),
                }
            )
    else:
        fallback_report["attempts"].append(
            {
                "variant": "stretch_only",
                "phase": "prepare",
                "status": "skipped",
                "reason": (
                    "prior stretch execution failed"
                    if not allow_stretch
                    else "starmask stretch disabled by configuration"
                ),
            }
        )
    variants.append(
        {
            "variant": "raw",
            "starmask": raw_starmask_stem,
            "calibration": {
                "status": "review_only_raw",
                "support_mode": "normal",
                "stretch_applied": False,
                "reason": "unaltered canonical Stage6 starmask fallback",
            },
        }
    )

    previous_remix_base_stem = str(
        getattr(pipeline, "_stage9_remix_base_stem", "") or ""
    )
    pipeline._stage9_remix_base_stem = stage8_source_stem
    for variant in variants:
        variant_name = str(variant["variant"])
        starmask_name = str(variant["starmask"])
        attempt_name = f"screen_stage8_starmask_{variant_name}_fallback"
        pipeline._stage9_starmask_calibration = copy.deepcopy(
            variant.get("calibration") or {}
        )
        attempt: Dict[str, Any] = {
            "attempt": attempt_name,
            "formula": "screen",
            "status": "failed",
            "accepted": False,
            "delivery_accepted": False,
            "gate_role": "fatal_safety_only",
            "review_required": True,
            "fallback_variant": variant_name,
            "trigger_reason": str(trigger_reason),
            "base_source_stem": stage8_source_stem,
            "starmask": starmask_name,
            "support_starmask": starmask_name,
            "support_mode": "normal",
            "intensity": float(intensity),
            "recovery_kind": "minimal_stage8_starmask_fallback",
            "recovery_strength": 0.0,
            "recovery_target_groups": [],
            "issues": [],
            "metrics": {},
            "reason_codes": ["STAGE9_MINIMAL_STARMASK_REVIEW_FALLBACK"],
        }
        try:
            pipeline.cmd_with_check("load", stage8_source_stem)
            base_pixels, base_domain = _stage9_current_canonical_pixels(
                pipeline,
                label="Stage8 fallback base",
            )
            starmask_path = _stage9_existing_fit_path(pipeline, starmask_name)
            if starmask_path is None:
                raise RuntimeError(
                    f"Stage9 {variant_name} fallback starmask file is unavailable"
                )
            pipeline.cmd_with_check("load", starmask_name)
            starmask_pixels, starmask_domain = (
                _stage9_current_canonical_pixels(
                    pipeline,
                    label=f"Stage9 {variant_name} fallback starmask",
                )
            )
            starmask_shape_compatible = (
                _stage9_fallback_starmask_shape_compatible(
                    starmask_pixels,
                    base_pixels,
                )
            )
            attempt["source_safety"] = {
                "status": (
                    "passed" if starmask_shape_compatible else "failed"
                ),
                "base_file": str(base_path),
                "starmask_file": str(starmask_path),
                "base_domain": base_domain,
                "starmask_domain": starmask_domain,
                "shape_compatible": bool(starmask_shape_compatible),
            }
            if not starmask_shape_compatible:
                raise RuntimeError(
                    "Stage8 fallback base and starmask dimensions are incompatible"
                )
            pipeline._stage9_minimal_fallback_active = True
            try:
                applied = bool(
                    pipeline._apply_previous_stage_star_remix(
                        stage8_source_stem,
                        starmask_name,
                        intensity,
                    )
                )
            finally:
                pipeline._stage9_minimal_fallback_active = False
            if not applied:
                raise RuntimeError("Stage8 + starmask Screen execution failed")
            candidate_pixels, candidate_domain = (
                _stage9_current_canonical_pixels(
                    pipeline,
                    label=f"Stage9 {variant_name} fallback",
                )
            )
            safety = _stage9_minimal_fallback_safety(
                pipeline,
                base_pixels,
                candidate_pixels,
                base_domain=base_domain,
                candidate_domain=candidate_domain,
            )
            attempt["fatal_safety"] = safety
            if safety.get("status") != "passed":
                attempt["issues"] = list(safety.get("issues") or [])
                fallback_report["attempts"].append(copy.deepcopy(attempt))
                remix_attempts.append(attempt)
                continue

            try:
                formal_quality = _assess_stage9_candidate(
                    pipeline,
                    stage8_source_stem,
                    attempt=f"{attempt_name}_diagnostic",
                    formula="screen",
                )
                if not isinstance(formal_quality, dict):
                    raise TypeError("formal quality diagnostics returned no report")
            except (
                AttributeError,
                CommandError,
                OSError,
                RuntimeError,
                SirilError,
                TypeError,
                ValueError,
            ) as diagnostic_error:
                formal_quality = {
                    "status": "unavailable",
                    "accepted": False,
                    "issues": [str(diagnostic_error)],
                    "metrics": {},
                    "diagnostic_only": True,
                }
            attempt["formal_quality"] = formal_quality
            attempt["metrics"] = copy.deepcopy(
                formal_quality.get("metrics") or {}
            )
            attempt["formal_quality_accepted"] = bool(
                formal_quality.get("accepted", False)
            )
            attempt["formal_quality_issues"] = list(
                formal_quality.get("issues") or []
            )
            pipeline.cmd_with_check("load", stage8_source_stem)
            safe_pixel_writer = getattr(
                pipeline,
                "_set_current_image_pixeldata",
                None,
            )
            if callable(safe_pixel_writer):
                safe_pixel_writer(
                    candidate_pixels,
                    label=f"Stage9 {variant_name} minimal fallback restore",
                )
            else:
                set_pixels = getattr(pipeline.siril, "set_image_pixeldata", None)
                if not callable(set_pixels):
                    raise RuntimeError(
                        "minimal Stage9 fallback pixel writer unavailable"
                    )
                lock_factory = getattr(pipeline.siril, "image_lock", None)
                if callable(lock_factory):
                    with lock_factory():
                        set_pixels(candidate_pixels)
                else:
                    set_pixels(candidate_pixels)
            review_saved = bool(
                pipeline._save_stage_output("stage9_review_with_stars")
            )
            canonical_saved = bool(
                review_saved
                and pipeline._save_stage_output("stage9_remixed")
            )
            attempt["save_status"] = {
                "review_saved": review_saved,
                "canonical_saved": canonical_saved,
            }
            if not (review_saved and canonical_saved):
                attempt["issues"] = [
                    "minimal Stage8 + starmask fallback save failed"
                ]
                fallback_report["attempts"].append(copy.deepcopy(attempt))
                remix_attempts.append(attempt)
                continue
        except (
            AttributeError,
            CommandError,
            OSError,
            RuntimeError,
            SirilError,
            TypeError,
            ValueError,
        ) as error:
            attempt["issues"] = [str(error)]
            fallback_report["attempts"].append(copy.deepcopy(attempt))
            remix_attempts.append(attempt)
            continue

        attempt.update(
            status="selected_review_only",
            delivery_accepted=True,
        )
        remix_attempts.append(attempt)
        fallback_report["attempts"].append(copy.deepcopy(attempt))
        fallback_report.update(
            status="selected",
            selected_variant=variant_name,
            selected_starmask=starmask_name,
            final_source="stage9_review_with_stars",
        )
        pipeline._stage9_fallback_remix_report = fallback_report
        pipeline._stage9_selected_remix_quality = dict(attempt)
        pipeline._stage9_star_layer_decomposition = (
            "minimal_stage6_starmask_screen"
        )
        pipeline._stage9_stars_applied = True
        pipeline._stage9_output_contains_stars = True
        pipeline._stage9_output_withheld = False
        pipeline._stage9_remix_formally_accepted = False
        pipeline._stage9_review_candidate_selected = False
        pipeline._stage9_stars_application_mode = (
            "screen_minimal_review_fallback"
        )
        pipeline._stage9_final_source = "stage9_review_with_stars"
        pipeline._stage9_bypassed_bad_starless = False
        pipeline._require_review(
            9,
            "stage8_starmask_review_fallback",
            {
                "trigger_reason": str(trigger_reason),
                "base_source": stage8_source_stem,
                "starmask": starmask_name,
                "variant": variant_name,
            },
        )
        messages.append(
            "Stage9 review-only fallback retained the Stage8 Starless base "
            f"with {variant_name} starmask Screen composition "
            f"(base={stage8_source_stem}, starmask={starmask_name})"
        )
        return True, attempt

    pipeline._stage9_remix_base_stem = previous_remix_base_stem
    pipeline._stage9_fallback_remix_report = fallback_report
    messages.append(
        "Stage9 Stage8 + minimal starmask fallback candidates were unavailable; "
        "continuing to the terminal with-stars fallback"
    )
    return False, None


def _stage9_remix_intensity_candidates(
    pipeline,
    *,
    primary_intensity: float,
    remix_scale: float,
    reference_degraded: bool = False,
) -> List[tuple[str, float]]:
    """Build a genuinely descending remix ladder after the primary candidate."""
    configured_levels = getattr(
        pipeline.cfg,
        "stage9_fallback_intensity_levels",
        (0.75, 0.55, 0.40),
    )
    if not isinstance(configured_levels, (list, tuple)):
        configured_levels = (0.75, 0.55, 0.40)
    fallback_cap = _clamp_float(
        getattr(pipeline.cfg, "stage9_fallback_intensity_cap", 0.95),
        0.40,
        1.05,
    )
    fallback_floor = _clamp_float(
        getattr(pipeline.cfg, "stage9_fallback_intensity_floor", 0.40),
        0.40,
        0.75,
    )
    fallback_retry_max = max(
        0,
        min(3, int(getattr(pipeline.cfg, "stage9_fallback_retry_max", 3) or 0)),
    )
    if reference_degraded:
        safe_levels = []
        for raw_level in configured_levels:
            try:
                safe_levels.append(
                    max(fallback_floor, min(float(raw_level), fallback_cap))
                )
            except (TypeError, ValueError):
                continue
        conservative_level = min(safe_levels or [0.40])
        conservative_intensity = min(
            primary_intensity,
            _clamp_float(conservative_level * remix_scale, 0.10, 1.05),
        )
        return [("reference_degraded_strict", conservative_intensity)]

    candidates: List[tuple[str, float]] = [("primary", primary_intensity)]
    previous = primary_intensity
    fallback_count = 0
    for raw_level in configured_levels:
        if fallback_count >= fallback_retry_max:
            break
        try:
            base_level = max(
                fallback_floor,
                min(float(raw_level), fallback_cap),
            )
        except (TypeError, ValueError):
            continue
        effective = _clamp_float(base_level * remix_scale, 0.10, 1.05)
        if effective >= previous - 1e-6:
            continue
        label = f"fallback_{int(round(effective * 100)):03d}"
        candidates.append((label, effective))
        previous = effective
        fallback_count += 1
    return candidates


def _assess_stage9_candidate(
    pipeline,
    source_stem: str,
    *,
    attempt: str,
    formula: str,
) -> Dict[str, Any]:
    assessor = getattr(pipeline, "_stage9_assess_current_remix", None)
    if not callable(assessor):
        return {
            "attempt": attempt,
            "formula": formula,
            "status": "not_measured",
            "accepted": True,
            "gate_enabled": False,
            "issues": ["quality assessor unavailable"],
            "metrics": {},
        }
    report = assessor(
        source_stem,
        attempt=attempt,
        formula=formula,
    )
    reference_samples = getattr(
        pipeline,
        "_stage9_star_color_reference_samples",
        None,
    )
    star_layer = getattr(pipeline, "_stage9_last_star_layer", None)
    star_color_gate_enabled = bool(
        getattr(
            pipeline.cfg,
            "stage9_star_color_post_validation_enabled",
            True,
        )
    )
    if isinstance(reference_samples, dict) and star_layer is not None:
        validation = assess_repaired_star_layer(
            star_layer,
            reference_samples,
            support_mask=getattr(
                pipeline,
                "_stage9_last_star_overlay_mask",
                None,
            ),
            chroma_error_max=float(
                getattr(
                    pipeline.cfg,
                    "stage9_star_color_post_chroma_error_max",
                    0.22,
                )
            ),
        )
        validation["gate_enabled"] = star_color_gate_enabled
        validation["enforced"] = star_color_gate_enabled
        validation_metrics = validation.get("metrics") or {}
        validation_limits = validation.get("limits") or {}
        validation_gates: Dict[str, Dict[str, Any]] = {}
        validation_advisories: List[str] = []
        for metric_name, limit_name in (
            ("median_chroma_error", "median_chroma_error_max"),
            (
                "extreme_chroma_outlier_ratio",
                "extreme_chroma_outlier_ratio_max",
            ),
        ):
            if (
                metric_name not in validation_metrics
                or limit_name not in validation_limits
            ):
                continue
            value = float(validation_metrics[metric_name])
            limit = float(validation_limits[limit_name])
            gate = stage7_quality.stage7_9_upper_quality_gate(
                pipeline.cfg,
                value=value,
                accepted_limit=limit,
            )
            validation_gates[metric_name] = gate
            if gate["advisory"]:
                validation_advisories.append(
                    f"{metric_name} {value:.6f}>{limit:.6f} "
                    "(advisory; star layer retained)"
                )
        if validation_gates:
            validation["quality_gates"] = validation_gates
            validation["quality_advisory_multiplier"] = (
                stage7_quality.stage7_9_quality_advisory_multiplier(
                    pipeline.cfg
                )
            )
        if (
            not bool(validation.get("accepted", False))
            and {
                "median_chroma_error",
                "extreme_chroma_outlier_ratio",
            }.issubset(validation_gates)
            and not any(
                bool(gate.get("hard_failed", False))
                for gate in validation_gates.values()
            )
        ):
            validation["original_status"] = validation.get("status")
            validation["original_issues"] = list(validation.get("issues") or [])
            validation["status"] = "advisory"
            validation["accepted"] = True
            validation["issues"] = []
            validation["advisories"] = validation_advisories
        report["star_color_validation"] = validation
        pipeline._stage9_star_color_post_validation = validation
        if validation.get("advisories"):
            report.setdefault("advisories", []).extend(
                validation.get("advisories") or []
            )
        if star_color_gate_enabled and not bool(validation.get("accepted", False)):
            report["accepted"] = False
            report["status"] = "rejected"
            report.setdefault("issues", []).extend(
                validation.get("issues") or ["star_color_validation_failed"]
            )
    return report


def _stage9_pixel_hash(pixels: np.ndarray) -> str:
    """Return a stable hash over the exact persisted pixel representation."""
    array = np.ascontiguousarray(np.asarray(pixels))
    digest = hashlib.sha256()
    digest.update(str(tuple(int(value) for value in array.shape)).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _persist_stage9_sep_crossmatch_evidence(
    pipeline,
    evidence: Dict[str, Any],
) -> Dict[str, Any]:
    """Persist the full SEP evidence and return the compact v10 reference."""
    payload = copy.deepcopy(evidence)
    summary = stage9_quality.stage9_sep_crossmatch_summary(payload)
    summary["artifact"] = "stage9_sep_crossmatch.json"
    writer = getattr(pipeline, "_write_stage_json", None)
    process_dir = getattr(pipeline, "process_dir", None)
    if not callable(writer) or process_dir is None:
        summary.update(
            status="unavailable",
            accepted=False,
            reason_code="stage9_sep_crossmatch_artifact_unavailable",
            reason="Stage9 JSON artifact writer or process directory is unavailable",
            artifact_sha256=None,
        )
        pipeline._stage9_sep_crossmatch_report = payload
        pipeline._stage9_sep_crossmatch_summary = summary
        return summary
    try:
        writer("stage9_sep_crossmatch.json", payload)
        artifact_path = Path(process_dir) / "stage9_sep_crossmatch.json"
        artifact_sha256 = run_manifest.sha256_file(artifact_path)
        if not artifact_sha256:
            raise OSError("stage9_sep_crossmatch.json was not durably written")
        summary["artifact_sha256"] = artifact_sha256
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        summary.update(
            status="unavailable",
            accepted=False,
            reason_code="stage9_sep_crossmatch_artifact_unavailable",
            reason=str(error),
            artifact_sha256=None,
        )
    pipeline._stage9_sep_crossmatch_report = payload
    pipeline._stage9_sep_crossmatch_summary = summary
    return summary


def _validate_stage9_persisted_output(
    pipeline,
    source_stem: str,
    selected_quality: Dict[str, Any],
) -> Dict[str, Any]:
    """Reload the canonical FITS and repeat every formal Stage9 gate."""
    report: Dict[str, Any] = {
        "schema": "starun.stage9-persisted-output-validation.v1",
        "status": "rejected",
        "accepted": False,
        "output_stem": "stage9_remixed",
        "selected_attempt": str(selected_quality.get("attempt") or "unknown"),
        "selected_formula": str(selected_quality.get("formula") or "screen"),
        "reason_code": "stage9_persisted_output_validation_failed",
    }
    try:
        getter = getattr(pipeline.siril, "get_image_pixeldata", None)
        if not callable(getter):
            raise RuntimeError("Siril pixel reader is unavailable")
        active_pixels = getter(preview=False)
        if active_pixels is None:
            raise RuntimeError("active Stage9 candidate pixels are unavailable")
        active = np.asarray(active_pixels)
        active_hash = _stage9_pixel_hash(active)

        pipeline.cmd_with_check("load", "stage9_remixed")
        persisted_pixels = getter(preview=False)
        if persisted_pixels is None:
            raise RuntimeError("persisted stage9_remixed pixels are unavailable")
        persisted = np.asarray(persisted_pixels)
        persisted_hash = _stage9_pixel_hash(persisted)
        shape_matches = bool(active.shape == persisted.shape)
        max_abs_error = None
        pixels_match = False
        if shape_matches:
            active_float = active.astype(np.float64, copy=False)
            persisted_float = persisted.astype(np.float64, copy=False)
            max_abs_error = float(np.max(np.abs(active_float - persisted_float)))
            pixels_match = bool(
                np.isfinite(max_abs_error) and max_abs_error <= 1e-6
            )

        reloaded_quality = _assess_stage9_candidate(
            pipeline,
            source_stem,
            attempt=f"{report['selected_attempt']}_persisted_reload",
            formula=report["selected_formula"],
        )
        catalog_visibility = dict(
            reloaded_quality.get("catalog_visibility") or {}
        )
        group_reports = dict(catalog_visibility.get("groups") or {})
        visibility_groups_passed = bool(
            catalog_visibility.get("available", False)
            and all(
                (group_reports.get(name) or {}).get("passed") is True
                for name in ("all", "weak", "bright")
            )
        )
        quality_accepted = bool(reloaded_quality.get("accepted", False))
        report.update(
            active_pixel_hash=active_hash,
            persisted_pixel_hash=persisted_hash,
            active_shape=list(active.shape),
            persisted_shape=list(persisted.shape),
            active_dtype=str(active.dtype),
            persisted_dtype=str(persisted.dtype),
            shape_matches=shape_matches,
            pixels_match=pixels_match,
            maximum_absolute_pixel_error=max_abs_error,
            quality_accepted=quality_accepted,
            catalog_visibility_groups_passed=visibility_groups_passed,
            coordinate_contract=catalog_visibility.get("coordinate_contract"),
            reloaded_quality=reloaded_quality,
        )
        matched_context = getattr(
            pipeline,
            "_stage9_matched_domain_context",
            None,
        )
        original_display = (
            matched_context.get("original_display")
            if isinstance(matched_context, dict)
            else None
        )
        if (
            original_display is None
            or (matched_context or {}).get("available") is not True
        ):
            matched_report = (
                dict(matched_context.get("report") or {})
                if isinstance(matched_context, dict)
                else {}
            )
            not_applicable = stage9_quality.stage9_sep_crossmatch_not_applicable(
                str(
                    matched_report.get("reason")
                    or "verified Stage6 matched-domain original is unavailable"
                )
            )
            sep_summary = _persist_stage9_sep_crossmatch_evidence(
                pipeline,
                not_applicable,
            )
            failures = ["persisted_sep_crossmatch_not_applicable"]
            if not shape_matches:
                failures.append("persisted_frame_shape_mismatch")
            if not pixels_match:
                failures.append("persisted_frame_pixels_mismatch")
            if not quality_accepted:
                failures.append("persisted_quality_gate_failed")
            if not visibility_groups_passed:
                failures.append("persisted_catalog_visibility_failed")
            report.update(
                status="rejected",
                accepted=False,
                reason_code="stage9_persisted_output_validation_failed",
                reason=(
                    "independent SEP is not applicable without a verified "
                    "Stage6 matched domain"
                ),
                sep_crossmatch_accepted=False,
                sep_crossmatch=sep_summary,
                restored_after_sep=True,
                restored_pixel_hash=persisted_hash,
                failures=failures,
            )
            pipeline._stage9_persisted_output_validation = report
            return report

        remix_base_stem = str(
            getattr(pipeline, "_stage9_remix_base_stem", source_stem)
            or source_stem
        )
        pipeline.cmd_with_check("load", remix_base_stem)
        before_pixels = getter(preview=False)
        if before_pixels is None:
            raise RuntimeError("Stage9 pre-remix base pixels are unavailable")
        before = np.asarray(before_pixels)
        original = np.asarray(original_display)
        sep_evidence = stage9_quality.assess_independent_sep_crossmatch(
            original,
            before,
            persisted,
            pipeline.cfg,
            original_pixel_sha256=_stage9_pixel_hash(original),
            before_pixel_sha256=_stage9_pixel_hash(before),
            after_pixel_sha256=persisted_hash,
            spatial_scale=getattr(pipeline, "_stage9_spatial_scale", None),
            source_names={
                "O": "verified_stage6_input_in_stage7_matched_domain",
                "B": remix_base_stem,
                "C": "stage9_remixed",
            },
        )
        sep_summary = _persist_stage9_sep_crossmatch_evidence(
            pipeline,
            sep_evidence,
        )
        sep_crossmatch_accepted = bool(sep_summary.get("accepted", False))

        # SEP may temporarily load B, but Stage9 must leave the persisted C
        # buffer active for Stage10 and for the exact-pixel delivery audit.
        pipeline.cmd_with_check("load", "stage9_remixed")
        restored_pixels = getter(preview=False)
        if restored_pixels is None:
            raise RuntimeError("stage9_remixed could not be restored after SEP")
        restored_hash = _stage9_pixel_hash(np.asarray(restored_pixels))
        restored_after_sep = bool(restored_hash == persisted_hash)

        accepted = bool(
            shape_matches
            and pixels_match
            and quality_accepted
            and visibility_groups_passed
            and sep_crossmatch_accepted
            and restored_after_sep
        )
        report.update(
            status="ok" if accepted else "rejected",
            accepted=accepted,
            reason_code=(
                "stage9_persisted_output_validation_ok"
                if accepted
                else "stage9_persisted_output_validation_failed"
            ),
            active_pixel_hash=active_hash,
            persisted_pixel_hash=persisted_hash,
            active_shape=list(active.shape),
            persisted_shape=list(persisted.shape),
            active_dtype=str(active.dtype),
            persisted_dtype=str(persisted.dtype),
            shape_matches=shape_matches,
            pixels_match=pixels_match,
            maximum_absolute_pixel_error=max_abs_error,
            quality_accepted=quality_accepted,
            catalog_visibility_groups_passed=visibility_groups_passed,
            coordinate_contract=catalog_visibility.get("coordinate_contract"),
            sep_crossmatch_accepted=sep_crossmatch_accepted,
            sep_crossmatch=sep_summary,
            restored_after_sep=restored_after_sep,
            restored_pixel_hash=restored_hash,
            reloaded_quality=reloaded_quality,
        )
        if not accepted:
            failures = []
            if not shape_matches:
                failures.append("persisted_frame_shape_mismatch")
            if not pixels_match:
                failures.append("persisted_frame_pixels_mismatch")
            if not quality_accepted:
                failures.append("persisted_quality_gate_failed")
            if not visibility_groups_passed:
                failures.append("persisted_catalog_visibility_failed")
            if not sep_crossmatch_accepted:
                failures.append("persisted_sep_crossmatch_failed")
            if not restored_after_sep:
                failures.append("persisted_stage9_buffer_restore_failed")
            report["failures"] = failures
    except (
        AttributeError,
        CommandError,
        RuntimeError,
        SirilError,
        TypeError,
        ValueError,
        FloatingPointError,
    ) as error:
        report["reason"] = str(error)
        failures = ["persisted_output_validation_unavailable"]
        if report.get("shape_matches") is False:
            failures.append("persisted_frame_shape_mismatch")
        if report.get("pixels_match") is False:
            failures.append("persisted_frame_pixels_mismatch")
        if report.get("quality_accepted") is False:
            failures.append("persisted_quality_gate_failed")
        if report.get("catalog_visibility_groups_passed") is False:
            failures.append("persisted_catalog_visibility_failed")
        report["failures"] = failures
        sep_summary = getattr(pipeline, "_stage9_sep_crossmatch_summary", None)
        if not isinstance(sep_summary, dict):
            sep_summary = _persist_stage9_sep_crossmatch_evidence(
                pipeline,
                stage9_quality._stage9_sep_unavailable(str(error)),
            )
        report["sep_crossmatch_accepted"] = False
        report["sep_crossmatch"] = sep_summary
        try:
            pipeline.cmd_with_check("load", "stage9_remixed")
        except (CommandError, SirilError, OSError, RuntimeError):
            pass
    pipeline._stage9_persisted_output_validation = report
    return report


def _record_stage9_quality_advisories(
    pipeline,
    messages: List[str],
    quality: Dict[str, Any],
    *,
    label: str,
) -> None:
    advisories = list(quality.get("advisories") or [])
    if not advisories:
        return
    advisory_text = ", ".join(str(item) for item in advisories[:3])
    message = f"Stage9 {label} advisory; continuing: {advisory_text}"
    messages.append(message)
    pipeline.log.warn(message)


def _stage9_needs_compact_mask_recovery(quality: Dict[str, Any]) -> bool:
    """Return whether a rejected candidate indicates broad starmask contamination."""
    if bool(quality.get("accepted", False)):
        return False
    issue_text = " ".join(str(item) for item in quality.get("issues", [])).lower()
    if _stage9_psf_size_direction(quality) == "large":
        return True
    if any(
        token in issue_text
        for token in (
            "background_mottling_growth",
            "changed_pixel_ratio",
            "background_lift",
            "chromatic_star_addition_ratio",
            "new_hollow_structure_max_area",
            "local_connected_component_max_area",
            "local_nonstellar_shape_component_count",
            "local_single_pixel_component_ratio",
            "local_cyan_blue_component_max_area",
            "core_color_jump_component_max_area",
        )
    ):
        return True
    metrics = quality.get("metrics") or {}
    limits = quality.get("limits") or {}
    try:
        changed_ratio = float(metrics.get("changed_pixel_ratio", 0.0) or 0.0)
        recovery_limit = min(
            float(limits.get("changed_pixel_ratio", 0.35) or 0.35),
            float(
                limits.get(
                    "background_mottling_low_absolute_changed_pixel_ratio_max",
                    0.12,
                )
                or 0.12
            ),
        )
    except (TypeError, ValueError):
        return False
    return changed_ratio > recovery_limit


def _stage9_psf_size_direction(quality: Dict[str, Any]) -> str | None:
    """Return whether a formal PSF closure failed below or above its limits."""
    closure = quality.get("psf_closure") or {}
    if closure.get("status") != "rejected":
        return None
    limits = closure.get("limits") or {}
    groups = closure.get("groups") or {}
    try:
        lower = float(limits["stage9_psf_fwhm_ratio_min"])
        upper = float(limits["stage9_psf_fwhm_ratio_max"])
    except (KeyError, TypeError, ValueError):
        return None
    ratios = []
    for group in groups.values():
        if isinstance(group, dict) and group.get("status") in {"ok", "insufficient"}:
            try:
                ratios.append(float(group["fwhm_ratio_median"]))
            except (KeyError, TypeError, ValueError):
                continue
    if any(value < lower for value in ratios):
        return "small"
    if any(value > upper for value in ratios):
        return "large"
    return None


def _stage9_psf_group_ratios(quality: Dict[str, Any]) -> Dict[str, float]:
    """Return assessable ordinary-star FWHM ratios from one quality report."""
    closure = quality.get("psf_closure") or {}
    groups = closure.get("groups") or {}
    ratios: Dict[str, float] = {}
    for group_name in ("all", "weak", "bright"):
        group = groups.get(group_name)
        if not isinstance(group, dict) or group.get("status") != "ok":
            continue
        try:
            ratio = float(group["fwhm_ratio_median"])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(ratio) and ratio > 0.0:
            ratios[group_name] = ratio
    return ratios


def _stage9_psf_uncertainty_exemption_used(
    quality: Dict[str, Any],
) -> bool:
    """Return whether formal PSF acceptance crossed a limit by uncertainty."""
    closure = quality.get("psf_closure") or {}
    if bool(closure.get("uncertainty_exemption_used", False)):
        return True
    return any(
        isinstance(group, dict)
        and bool(group.get("accepted_within_uncertainty", False))
        for group in (closure.get("groups") or {}).values()
    )


def _stage9_review_fwhm_ratio_max(pipeline) -> float:
    """Return the independent, non-profile-scaled review-candidate ceiling."""
    return _clamp_float(
        getattr(pipeline.cfg, "stage9_psf_review_fwhm_ratio_max", 1.65),
        1.10,
        1.65,
    )


def _stage9_failure_record(
    *,
    metric: str,
    failure_class: str,
    direction: str,
    value: Any = None,
    formal_limit: Any = None,
    severity_ratio: Any = None,
    group: Optional[str] = None,
    reason: str = "",
) -> Dict[str, Any]:
    def finite_float(raw: Any) -> Optional[float]:
        try:
            parsed = float(raw)
        except (TypeError, ValueError):
            return None
        return parsed if np.isfinite(parsed) else None

    record: Dict[str, Any] = {
        "metric": str(metric or "unknown"),
        "class": str(failure_class),
        "group": str(group) if group else None,
        "direction": str(direction or "unknown"),
        "value": finite_float(value),
        "formal_limit": finite_float(formal_limit),
        "severity_ratio": finite_float(severity_ratio),
        "reason": str(reason or ""),
    }
    return record


def _stage9_structured_failure_classification(
    quality: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split a rejected attempt into structured structural/numeric failures."""
    structural: List[Dict[str, Any]] = []
    numeric: List[Dict[str, Any]] = []
    accounted_issue_text: set[str] = set()

    closure = quality.get("psf_closure") or {}
    closure_limits = closure.get("limits") or {}
    closure_issues = {
        str(item) for item in (closure.get("issues") or []) if str(item)
    }
    accounted_issue_text.update(closure_issues)
    try:
        ratio_min = float(closure_limits["stage9_psf_fwhm_ratio_min"])
        ratio_max = float(closure_limits["stage9_psf_fwhm_ratio_max"])
    except (KeyError, TypeError, ValueError):
        ratio_min = ratio_max = float("nan")
    if bool(closure.get("gate_enabled", True)):
        for group_name in ("all", "weak", "bright"):
            group = (closure.get("groups") or {}).get(group_name)
            if not isinstance(group, dict):
                if not bool(quality.get("accepted", False)):
                    structural.append(
                        _stage9_failure_record(
                            metric="star_psf_fwhm_evidence",
                            failure_class="structural",
                            direction="unavailable",
                            group=group_name,
                            reason="candidate PSF group evidence is unavailable",
                        )
                    )
                continue
            status = str(group.get("status") or "")
            if status == "insufficient":
                structural.append(
                    _stage9_failure_record(
                        metric="star_psf_fwhm_sample_count",
                        failure_class="structural",
                        direction="insufficient",
                        value=group.get("candidate_sample_count"),
                        formal_limit=group.get("minimum_sample_count"),
                        group=group_name,
                        reason=str(
                            group.get("reason")
                            or "candidate PSF samples are insufficient"
                        ),
                    )
                )
                continue
            if status == "not_assessed" and not bool(
                quality.get("accepted", False)
            ):
                structural.append(
                    _stage9_failure_record(
                        metric="star_psf_fwhm_evidence",
                        failure_class="structural",
                        direction="insufficient",
                        value=group.get("reference_sample_count"),
                        formal_limit=group.get("minimum_sample_count"),
                        group=group_name,
                        reason=str(
                            group.get("reason")
                            or "reference PSF evidence is insufficient"
                        ),
                    )
                )
                continue
            if status != "ok":
                if not bool(quality.get("accepted", False)):
                    structural.append(
                        _stage9_failure_record(
                            metric="star_psf_fwhm_evidence",
                            failure_class="structural",
                            direction="invalid",
                            group=group_name,
                            reason=(
                                str(group.get("reason") or "")
                                or f"candidate PSF group status={status or 'unknown'}"
                            ),
                        )
                    )
                continue
            try:
                ratio = float(group["fwhm_ratio_median"])
            except (KeyError, TypeError, ValueError):
                structural.append(
                    _stage9_failure_record(
                        metric="star_psf_fwhm_ratio",
                        failure_class="structural",
                        direction="invalid",
                        group=group_name,
                        reason="candidate PSF ratio is unavailable or invalid",
                    )
                )
                continue
            if not np.isfinite(ratio) or ratio <= 0.0:
                structural.append(
                    _stage9_failure_record(
                        metric="star_psf_fwhm_ratio",
                        failure_class="structural",
                        direction="invalid",
                        value=ratio,
                        group=group_name,
                        reason="candidate PSF ratio is non-finite or non-positive",
                    )
                )
            elif np.isfinite(ratio_min) and ratio < ratio_min:
                numeric.append(
                    _stage9_failure_record(
                        metric="star_psf_fwhm_ratio",
                        failure_class="numeric",
                        direction="lower",
                        value=ratio,
                        formal_limit=ratio_min,
                        severity_ratio=ratio_min / max(ratio, 1e-12),
                        group=group_name,
                        reason="PSF FWHM is below the formal lower limit",
                    )
                )
            elif np.isfinite(ratio_max) and ratio > ratio_max:
                numeric.append(
                    _stage9_failure_record(
                        metric="star_psf_fwhm_ratio",
                        failure_class="numeric",
                        direction="upper",
                        value=ratio,
                        formal_limit=ratio_max,
                        severity_ratio=ratio / max(ratio_max, 1e-12),
                        group=group_name,
                        reason="PSF FWHM exceeds the formal upper limit",
                    )
                )

    def add_quality_gates(
        gates: Any,
        *,
        prefix: str = "",
    ) -> None:
        if not isinstance(gates, dict):
            return
        for metric_name, gate in gates.items():
            if not isinstance(gate, dict) or not bool(
                gate.get("hard_failed", False)
            ):
                continue
            value = gate.get("value")
            limit = gate.get("accepted_limit")
            try:
                direction = (
                    "lower"
                    if float(value) < float(limit)
                    else "upper"
                )
            except (TypeError, ValueError):
                direction = "unknown"
            numeric.append(
                _stage9_failure_record(
                    metric=prefix + str(metric_name),
                    failure_class="numeric",
                    direction=direction,
                    value=value,
                    formal_limit=limit,
                    severity_ratio=gate.get("severity_ratio"),
                    reason="numeric quality gate hard-failed",
                )
            )

    add_quality_gates(quality.get("quality_gates"))
    gate_issues = {
        str(item) for item in (quality.get("gate_issues") or []) if str(item)
    }
    accounted_issue_text.update(gate_issues)

    validation = quality.get("star_color_validation") or {}
    validation_issues = {
        str(item) for item in (validation.get("issues") or []) if str(item)
    }
    accounted_issue_text.update(validation_issues)
    validation_enforced = bool(
        validation.get(
            "enforced",
            validation.get("gate_enabled", True),
        )
    )
    if validation_enforced:
        add_quality_gates(
            validation.get("quality_gates"),
            prefix="star_color_",
        )
        if validation and not bool(
            validation.get("accepted", False)
        ) and not any(
            item["metric"].startswith("star_color_") for item in numeric
        ):
            structural.append(
                _stage9_failure_record(
                    metric="star_color_validation",
                    failure_class="structural",
                    direction="unavailable",
                    reason="; ".join(validation_issues)
                    or str(
                        validation.get("reason")
                        or "star color validation failed"
                    ),
                )
            )

    for issue in quality.get("structural_issues") or []:
        issue_text = str(issue)
        accounted_issue_text.add(issue_text)
        if issue_text in closure_issues:
            continue
        structural.append(
            _stage9_failure_record(
                metric="candidate_structure",
                failure_class="structural",
                direction="invalid",
                reason=issue_text,
            )
        )

    if str(quality.get("status") or "").lower() in {
        "failed",
        "not_measured",
        "unavailable",
    }:
        structural.append(
            _stage9_failure_record(
                metric="candidate_execution",
                failure_class="structural",
                direction="invalid",
                reason=f"candidate status={quality.get('status')}",
            )
        )

    for issue in quality.get("issues") or []:
        issue_text = str(issue)
        if issue_text in accounted_issue_text:
            continue
        structural.append(
            _stage9_failure_record(
                metric="unclassified_rejection",
                failure_class="structural",
                direction="unknown",
                reason=issue_text,
            )
        )

    if (
        not bool(quality.get("accepted", False))
        and not structural
        and not numeric
    ):
        structural.append(
            _stage9_failure_record(
                metric="unclassified_rejection",
                failure_class="structural",
                direction="unknown",
                reason="candidate was rejected without structured gate evidence",
            )
        )
    return structural, numeric


def _stage9_review_candidate_score(
    quality: Dict[str, Any],
    *,
    attempt_order: int,
) -> tuple[float, int, float, float, float, float, int]:
    """Rank bounded review candidates, keeping safety ahead of completeness."""
    severities: List[float] = []

    def collect_gate_severities(gates: Any) -> None:
        if not isinstance(gates, dict):
            return
        for gate in gates.values():
            if not isinstance(gate, dict):
                continue
            try:
                severity = float(gate.get("severity_ratio", 1.0) or 1.0)
            except (TypeError, ValueError):
                continue
            if np.isfinite(severity):
                severities.append(max(0.0, severity))

    collect_gate_severities(quality.get("quality_gates"))
    collect_gate_severities(
        (quality.get("star_color_validation") or {}).get("quality_gates")
    )
    numeric_failures = list(quality.get("numeric_failures") or [])
    for failure in numeric_failures:
        try:
            severity = float(failure.get("severity_ratio"))
        except (AttributeError, TypeError, ValueError):
            continue
        if np.isfinite(severity):
            severities.append(max(0.0, severity))
    worst_severity = max(severities or [1.0])

    metrics = quality.get("metrics") or {}
    limits = quality.get("limits") or {}
    recovery_margins: List[float] = []
    for metric_name in (
        "weak_star_recovery_ratio",
        "star_recovery_ratio",
        "star_positive_delta_window_recovery_ratio",
        "star_wing_recovery_ratio",
    ):
        try:
            value = float(metrics[metric_name])
            limit = float(limits[metric_name])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value) and np.isfinite(limit) and limit > 0.0:
            recovery_margins.append(value / limit)
    minimum_recovery_margin = min(recovery_margins or [0.0])

    def finite_metric(name: str) -> float:
        try:
            value = float(metrics.get(name, float("inf")))
        except (TypeError, ValueError):
            return float("inf")
        return value if np.isfinite(value) else float("inf")

    try:
        intensity = float(quality.get("intensity", 0.0) or 0.0)
    except (TypeError, ValueError):
        intensity = 0.0
    if not np.isfinite(intensity):
        intensity = 0.0
    return (
        float(worst_severity),
        len(numeric_failures),
        -float(minimum_recovery_margin),
        -float(intensity),
        finite_metric("highlight_clip_growth"),
        finite_metric("bright_pixel_growth"),
        max(0, int(attempt_order)),
    )


def _stage9_review_candidate_eligibility(
    pipeline,
    quality: Dict[str, Any],
    *,
    attempt_order: int,
) -> Dict[str, Any]:
    """Classify one formal failure for the bounded review-only candidate pool."""
    structural, numeric = _stage9_structured_failure_classification(quality)
    star_color_gate_enabled = bool(
        getattr(
            pipeline.cfg,
            "stage9_star_color_post_validation_enabled",
            True,
        )
    )
    candidate_rejected = not bool(quality.get("accepted", False))
    if candidate_rejected:
        star_color_validation = quality.get("star_color_validation")
        if not isinstance(star_color_validation, dict):
            structural.append(
                _stage9_failure_record(
                    metric="star_color_validation",
                    failure_class="structural",
                    direction="unavailable",
                    reason="star color posterior evidence is unavailable",
                )
            )
        else:
            validation_metrics = star_color_validation.get("metrics") or {}
            validation_limits = star_color_validation.get("limits") or {}
            try:
                validation_sample_count = int(
                    validation_metrics["sample_count"]
                )
                validation_chroma_error = float(
                    validation_metrics["median_chroma_error"]
                )
                validation_outlier_ratio = float(
                    validation_metrics["extreme_chroma_outlier_ratio"]
                )
                validation_chroma_limit = float(
                    validation_limits["median_chroma_error_max"]
                )
                validation_outlier_limit = float(
                    validation_limits["extreme_chroma_outlier_ratio_max"]
                )
            except (KeyError, TypeError, ValueError):
                validation_sample_count = 0
                validation_chroma_error = float("nan")
                validation_outlier_ratio = float("nan")
                validation_chroma_limit = float("nan")
                validation_outlier_limit = float("nan")
            if (
                validation_sample_count < 8
                or not np.isfinite(validation_chroma_error)
                or not np.isfinite(validation_outlier_ratio)
                or not np.isfinite(validation_chroma_limit)
                or not np.isfinite(validation_outlier_limit)
            ):
                structural.append(
                    _stage9_failure_record(
                        metric="star_color_validation_evidence",
                        failure_class="structural",
                        direction="insufficient",
                        value=validation_sample_count,
                        formal_limit=8,
                        reason=(
                            "star color posterior samples or metrics are "
                            "incomplete"
                        ),
                    )
                )
            if star_color_gate_enabled and not bool(
                star_color_validation.get(
                    "enforced",
                    star_color_validation.get("gate_enabled", True),
                )
            ):
                structural.append(
                    _stage9_failure_record(
                        metric="star_color_validation",
                        failure_class="structural",
                        direction="disabled",
                        reason="star color posterior evidence was not enforced",
                    )
                )
    if candidate_rejected:
        metrics = quality.get("metrics") or {}
        limits = quality.get("limits") or {}
        for metric_name in (
            "weak_star_recovery_ratio",
            "star_recovery_ratio",
            "star_positive_delta_window_recovery_ratio",
            "star_wing_recovery_ratio",
        ):
            try:
                metric_value = float(metrics[metric_name])
                metric_limit = float(limits[metric_name])
            except (KeyError, TypeError, ValueError):
                metric_value = metric_limit = float("nan")
            if (
                not np.isfinite(metric_value)
                or not np.isfinite(metric_limit)
                or metric_limit <= 0.0
            ):
                structural.append(
                    _stage9_failure_record(
                        metric=metric_name,
                        failure_class="structural",
                        direction="unavailable",
                        value=metric_value,
                        formal_limit=metric_limit,
                        reason="candidate recovery evidence is unavailable",
                    )
                )
        for metric_name in (
            "highlight_clip_growth",
            "bright_pixel_growth",
        ):
            try:
                metric_value = float(metrics[metric_name])
            except (KeyError, TypeError, ValueError):
                metric_value = float("nan")
            if not np.isfinite(metric_value):
                structural.append(
                    _stage9_failure_record(
                        metric=metric_name,
                        failure_class="structural",
                        direction="unavailable",
                        value=metric_value,
                        reason="candidate ranking evidence is unavailable",
                    )
                )
        try:
            candidate_intensity = float(quality["intensity"])
        except (KeyError, TypeError, ValueError):
            candidate_intensity = float("nan")
        if not np.isfinite(candidate_intensity) or candidate_intensity <= 0.0:
            structural.append(
                _stage9_failure_record(
                    metric="remix_intensity",
                    failure_class="structural",
                    direction="unavailable",
                    value=candidate_intensity,
                    reason="candidate remix intensity is unavailable",
                )
            )
    quality["structural_failures"] = structural
    quality["numeric_failures"] = numeric
    review_max = _stage9_review_fwhm_ratio_max(pipeline)
    reasons: List[str] = []
    eligible = False

    if bool(quality.get("accepted", False)):
        reasons.append("formally_accepted")
    elif structural:
        reasons.append("structural_failure")
    else:
        non_psf = [
            item
            for item in numeric
            if item.get("metric") != "star_psf_fwhm_ratio"
        ]
        psf = [
            item
            for item in numeric
            if item.get("metric") == "star_psf_fwhm_ratio"
        ]
        if non_psf:
            reasons.append("non_psf_numeric_hard_failure")
        if any(item.get("direction") == "lower" for item in psf):
            reasons.append("psf_below_formal_lower_limit")
        upper_psf = [item for item in psf if item.get("direction") == "upper"]
        if len(psf) != 1 or len(upper_psf) != 1:
            reasons.append("psf_failure_count_not_one")
        elif upper_psf[0].get("value") is None:
            reasons.append("psf_upper_ratio_unavailable")
        elif float(upper_psf[0]["value"]) > review_max + 1e-12:
            reasons.append("psf_above_review_upper_limit")
        if not reasons:
            eligible = True
            reasons.append("single_psf_upper_failure_within_review_limit")

    score = _stage9_review_candidate_score(
        quality,
        attempt_order=attempt_order,
    )
    return {
        "eligible": eligible,
        "reasons": reasons,
        "formal_fwhm_ratio_max": (
            (quality.get("psf_closure") or {}).get("limits") or {}
        ).get(
            "stage9_psf_fwhm_ratio_max",
            getattr(pipeline.cfg, "stage9_psf_fwhm_ratio_max", 1.10),
        ),
        "review_fwhm_ratio_max": review_max,
        "checkpoint_saved": False,
        "selected": False,
        "selection_attempted": False,
        "checkpoint_state": {
            "status": "not_attempted",
            "image_checkpoint_saved": False,
        },
        "restore_status": "not_attempted",
        "final_save_status": "not_attempted",
        "selection_score": [
            None if not np.isfinite(value) else float(value)
            for value in score
        ],
    }


def _stage9_psf_recovery_target_min(pipeline, quality: Dict[str, Any]) -> float:
    """Resolve the soft natural-size target without changing the hard gate."""
    closure = quality.get("psf_closure") or {}
    limits = closure.get("limits") or {}
    try:
        hard_min = float(
            limits.get(
                "stage9_psf_fwhm_ratio_min",
                getattr(pipeline.cfg, "stage9_psf_fwhm_ratio_min", 0.93),
            )
        )
    except (TypeError, ValueError):
        hard_min = 0.93
    try:
        hard_max = float(
            limits.get(
                "stage9_psf_fwhm_ratio_max",
                getattr(pipeline.cfg, "stage9_psf_fwhm_ratio_max", 1.10),
            )
        )
    except (TypeError, ValueError):
        hard_max = 1.10
    configured = _clamp_float(
        getattr(pipeline.cfg, "stage9_psf_recovery_target_min", 0.97),
        0.50,
        1.00,
    )
    return float(max(hard_min, min(configured, hard_max, 1.00)))


def _stage9_psf_recovery_target_max(pipeline, quality: Dict[str, Any]) -> float:
    """Resolve the soft upper guard for a uniform source-wing expansion."""
    closure = quality.get("psf_closure") or {}
    limits = closure.get("limits") or {}
    try:
        hard_max = float(
            limits.get(
                "stage9_psf_fwhm_ratio_max",
                getattr(pipeline.cfg, "stage9_psf_fwhm_ratio_max", 1.10),
            )
        )
    except (TypeError, ValueError):
        hard_max = 1.10
    configured = _clamp_float(
        getattr(pipeline.cfg, "stage9_psf_recovery_target_max", 1.05),
        1.00,
        1.50,
    )
    return float(max(1.00, min(configured, hard_max)))


def _stage9_needs_progressive_psf_recovery(
    pipeline,
    quality: Dict[str, Any],
) -> bool:
    """Route undersized, measurable stars into source-wing recovery."""
    if not bool(getattr(pipeline.cfg, "stage9_psf_size_gate_enabled", True)):
        return False
    ratios = _stage9_psf_group_ratios(quality)
    if not ratios:
        return False
    target_min = _stage9_psf_recovery_target_min(pipeline, quality)
    target_max = _stage9_psf_recovery_target_max(pipeline, quality)
    if any(ratio > target_max for ratio in ratios.values()):
        return False
    if not bool(quality.get("accepted", False)):
        issues = [str(item).lower() for item in quality.get("issues", [])]
        if any("star_psf_fwhm" not in issue for issue in issues):
            return False
    return any(ratio < target_min for ratio in ratios.values())


def _stage9_psf_candidate_score(
    quality: Dict[str, Any],
    *,
    recovery_pixels: int,
) -> tuple[float, float, float, int, float, float]:
    """Rank safe candidates by worst PSF error, then expansion and highlights."""
    if not bool(quality.get("accepted", False)):
        return (float("inf"),) * 6
    ratios = tuple(_stage9_psf_group_ratios(quality).values())
    if not ratios:
        return (float("inf"),) * 6
    deviations = tuple(abs(value - 1.0) for value in ratios)
    metrics = quality.get("metrics") or {}

    def metric(name: str) -> float:
        try:
            value = float(metrics.get(name, 0.0) or 0.0)
        except (TypeError, ValueError):
            return float("inf")
        return value if np.isfinite(value) else float("inf")

    return (
        1.0 if _stage9_psf_uncertainty_exemption_used(quality) else 0.0,
        float(max(deviations)),
        float(sum(deviations) / len(deviations)),
        max(0, int(recovery_pixels)),
        metric("highlight_clip_growth"),
        metric("bright_pixel_growth"),
    )


def _stage9_has_recovery_shortfall(quality: Dict[str, Any]) -> bool:
    """Return whether lowering Screen intensity cannot improve the rejection."""
    issue_text = " ".join(str(item) for item in quality.get("issues", [])).lower()
    return any(
        token in issue_text
        for token in (
            "weak_star_recovery_ratio",
            "star_recovery_ratio",
            "star_positive_delta_window_recovery_ratio",
            "star_wing_recovery_ratio",
            "residual_dark_hole_ratio",
            "new_hollow_structure_max_area",
            "star_recovery_metrics_unavailable",
            "star_psf_fwhm_ratio",
            "star_psf_fwhm_sample_count",
        )
    )


def _prepare_stage9_star_reference(
    pipeline,
    starmask_name: str,
    messages: List[str],
) -> Dict[str, Any]:
    """Build a source-confirmed star catalog before any star plugin runs."""
    catalog: Dict[str, Any] = {
        "status": "unavailable",
        "reason": "original starmask pixels unavailable",
    }
    primary_summary: Dict[str, Any] = dict(catalog)
    reference_source = "starmask_only"
    starmask_pixels = None
    try:
        pipeline.cmd_with_check("load", starmask_name)
        get_pixels = getattr(pipeline.siril, "get_image_pixeldata", None)
        if callable(get_pixels):
            pixels = get_pixels(preview=False)
            if pixels is not None:
                starmask_pixels = np.array(pixels, copy=True)
                # Preserve a small immutable evidence map before plugin/color
                # preparation.  The ordinary catalog intentionally excludes
                # very large or saturated sources, so its compact layer cannot
                # later prove that an omitted Stage 5 source is genuine.
                pipeline._stage9_immutable_trusted_starmask_peak = np.array(
                    stage9_quality.normalized_star_layer_peak(starmask_pixels),
                    copy=True,
                )
                matched_context = _prepare_stage9_matched_domain_context(
                    pipeline,
                    messages,
                )
                matched_report = matched_context.get("report") or {}
                source_pixels = None
                if matched_context.get("available") is True:
                    reference_source = "stage6_pair_linked_mtf_O"
                    catalog = (
                        stage9_quality.build_display_confirmed_starmask_catalog(
                            starmask_pixels,
                            matched_context["original_display"],
                            pipeline.cfg,
                        )
                    )
                else:
                    # A current-run handoff can be unavailable after a safe
                    # degradation. The baseline Screen catalog may still be
                    # measured, but Unscreen and formal PSF remain disabled.
                    process_dir = getattr(pipeline, "process_dir", None)
                    for source_stem in (
                        "stage5_linear",
                        "stage6_input",
                        "working",
                    ):
                        source_path = (
                            process_dir / f"{source_stem}.fit"
                            if process_dir is not None
                            else None
                        )
                        if source_path is None or not source_path.exists():
                            continue
                        pipeline.cmd_with_check("load", source_stem)
                        source_data = get_pixels(preview=False)
                        if source_data is not None:
                            source_pixels = np.array(source_data, copy=True)
                            reference_source = source_stem
                            break
                    catalog = stage9_quality.build_star_reference_catalog(
                        starmask_pixels,
                        pipeline.cfg,
                        source_image=source_pixels,
                    )
                    catalog.update(
                        psf_reference_status="unavailable",
                        psf_reference_reason=str(
                            matched_report.get("reason")
                            or "matched-domain Stage6 pair unavailable"
                        ),
                    )
                catalog["reference_source"] = reference_source
                primary_summary = stage9_quality.star_reference_summary(catalog)
                if source_pixels is None:
                    if matched_context.get("available") is not True:
                        catalog["reference_degraded"] = True
                        catalog["reference_degraded_reason"] = (
                            "independent with-stars reference unavailable"
                        )
                if catalog.get("status") != "ok":
                    fallback_catalog = stage9_quality.build_star_reference_catalog(
                        starmask_pixels,
                        pipeline.cfg,
                    )
                    if fallback_catalog.get("status") == "ok":
                        fallback_catalog.update(
                            {
                                "reference_source": "starmask_only",
                                "reference_degraded": True,
                                "reference_degraded_reason": str(
                                    catalog.get("reason")
                                    or "source-confirmed reference unavailable"
                                ),
                                "source_reference": primary_summary,
                            }
                        )
                        catalog = fallback_catalog
                    else:
                        catalog["starmask_fallback_reference"] = (
                            stage9_quality.star_reference_summary(
                                fallback_catalog
                            )
                        )
                stage5_reference = _stage9_stage5_star_reference_report(pipeline)
                spatial_scale = stage9_quality.resolve_stage9_spatial_scale(
                    catalog,
                    stage5_stars=stage5_reference.get("stars") or [],
                    raw_starmask=starmask_pixels,
                )
                # Scale is independent evidence. Freeze it before rebuilding
                # the catalog so a contaminated/invalid scaled catalog cannot
                # erase a valid FWHM measurement.
                pipeline._stage9_spatial_scale = dict(spatial_scale)
                scale_review = bool(
                    spatial_scale.get("stage9_psf_review_required", False)
                )
                pipeline._stage9_spatial_scale_review_required = scale_review
                pipeline._stage9_psf_review_required = bool(
                    getattr(pipeline, "_stage9_psf_review_required", False)
                    or scale_review
                )
                scaled_catalog_validation: Dict[str, Any] = {
                    "schema": "starun.stage9-scaled-catalog-validation.v1",
                    "status": "not_run",
                    "scale_status": str(
                        spatial_scale.get("status") or "unavailable"
                    ),
                    "bootstrap_catalog": stage9_quality.star_reference_summary(
                        catalog
                    ),
                    "selected_route": None,
                    "source_confirmed": None,
                    "strict_starmask_only": None,
                }
                pipeline._stage9_scaled_catalog_validation = (
                    scaled_catalog_validation
                )
                if spatial_scale.get("status") == "ready":
                    # The first catalog is bootstrap evidence only. Once the
                    # scale is known, rebuild formal membership with scaled
                    # component bounds and freeze that result for every route.
                    if matched_context.get("available") is True:
                        rebuilt_catalog = (
                            stage9_quality.build_display_confirmed_starmask_catalog(
                                starmask_pixels,
                                matched_context["original_display"],
                                pipeline.cfg,
                                spatial_scale=spatial_scale,
                            )
                        )
                    elif source_pixels is not None:
                        rebuilt_catalog = stage9_quality.build_star_reference_catalog(
                            starmask_pixels,
                            pipeline.cfg,
                            source_image=source_pixels,
                            spatial_scale=spatial_scale,
                        )
                    else:
                        rebuilt_catalog = {
                            "status": "unavailable",
                            "reason": (
                                "scaled source-confirmed reference image "
                                "unavailable"
                            ),
                        }
                    scaled_catalog_validation["source_confirmed"] = (
                        stage9_quality.star_reference_summary(rebuilt_catalog)
                    )
                    if rebuilt_catalog.get("status") == "ok":
                        scaled_catalog_validation.update(
                            status="ok",
                            selected_route="source_confirmed",
                        )
                    else:
                        strict_catalog = (
                            stage9_quality.build_star_reference_catalog(
                                starmask_pixels,
                                pipeline.cfg,
                                spatial_scale=spatial_scale,
                            )
                        )
                        scaled_catalog_validation["strict_starmask_only"] = (
                            stage9_quality.star_reference_summary(
                                strict_catalog
                            )
                        )
                        if strict_catalog.get("status") == "ok":
                            strict_catalog.update(
                                reference_source="starmask_only",
                                reference_degraded=True,
                                reference_degraded_reason=(
                                    "scaled source-confirmed catalog rejected: "
                                    + str(
                                        rebuilt_catalog.get("reason")
                                        or "unknown reason"
                                    )
                                ),
                                source_reference=(
                                    scaled_catalog_validation[
                                        "source_confirmed"
                                    ]
                                ),
                                psf_reference_status="unavailable",
                                psf_reference_reason=(
                                    "strict scaled starmask-only catalog"
                                ),
                            )
                            rebuilt_catalog = strict_catalog
                            scaled_catalog_validation.update(
                                status="degraded",
                                selected_route="strict_starmask_only",
                                reason=(
                                    "source-confirmed scaled catalog failed; "
                                    "strict starmask-only catalog accepted"
                                ),
                            )
                        else:
                            scaled_catalog_validation.update(
                                status="unavailable",
                                selected_route="none",
                                reason=(
                                    "both scaled catalog validations failed"
                                ),
                            )
                            catalog = {
                                "status": "unavailable",
                                "reason": (
                                    "scaled Stage9 star-catalog validation "
                                    "failed for source-confirmed and strict "
                                    "starmask-only routes"
                                ),
                                "bootstrap_reference": (
                                    scaled_catalog_validation[
                                        "bootstrap_catalog"
                                    ]
                                ),
                                "scaled_catalog_validation": dict(
                                    scaled_catalog_validation
                                ),
                            }
                            rebuilt_catalog = None
                    if rebuilt_catalog is None:
                        pipeline._stage9_scaled_catalog_validation = dict(
                            scaled_catalog_validation
                        )
                    else:
                        rebuilt_catalog.update(
                            reference_source=rebuilt_catalog.get(
                                "reference_source",
                                catalog.get(
                                    "reference_source",
                                    reference_source,
                                ),
                            ),
                            reference_degraded=bool(
                                rebuilt_catalog.get(
                                    "reference_degraded",
                                    catalog.get("reference_degraded", False),
                                )
                            ),
                            reference_degraded_reason=str(
                                rebuilt_catalog.get(
                                    "reference_degraded_reason",
                                    catalog.get(
                                        "reference_degraded_reason"
                                    )
                                    or "",
                                )
                            ),
                            stage9_spatial_scale=dict(spatial_scale),
                        )
                    if rebuilt_catalog is not None:
                        per_star_fwhm = np.asarray(
                            rebuilt_catalog.get(
                                "_display_source_fwhm_px",
                                rebuilt_catalog.get("_source_fwhm_px", ()),
                            ),
                            dtype=np.float32,
                        )
                        star_count = np.asarray(
                            rebuilt_catalog.get("_peak_y", ())
                        ).size
                        if per_star_fwhm.size != star_count:
                            per_star_fwhm = np.full(
                                star_count,
                                float(spatial_scale["fwhm_median_px"]),
                                dtype=np.float32,
                            )
                        invalid_fwhm = ~np.isfinite(per_star_fwhm) | (
                            per_star_fwhm <= 0.0
                        )
                        per_star_fwhm[invalid_fwhm] = float(
                            spatial_scale["fwhm_median_px"]
                        )
                        rebuilt_catalog["_stage9_spatial_fwhm_px"] = (
                            per_star_fwhm
                        )
                        catalog = rebuilt_catalog
                        stage9_quality.freeze_stage9_spatial_geometry(
                            catalog,
                            starmask_pixels,
                        )
                else:
                    scaled_catalog_validation.update(
                        status="not_run",
                        reason=(
                            str(spatial_scale.get("reason_code"))
                            or "spatial scale unavailable"
                        ),
                    )
                pipeline._stage9_scaled_catalog_validation = dict(
                    scaled_catalog_validation
                )
                pipeline.cmd_with_check("load", starmask_name)
    except (
        CommandError,
        SirilError,
        RuntimeError,
        AttributeError,
        TypeError,
        ValueError,
        IndexError,
    ) as error:
        catalog = {"status": "unavailable", "reason": str(error)}
        validation = dict(
            getattr(pipeline, "_stage9_scaled_catalog_validation", {}) or {}
        )
        validation.update(
            status="unavailable",
            selected_route="none",
            reason=f"Stage9 catalog preparation failed: {error}",
        )
        pipeline._stage9_scaled_catalog_validation = validation

    try:
        pipeline.cmd_with_check("load", starmask_name)
    except (CommandError, SirilError, RuntimeError, AttributeError) as restore_error:
        if catalog.get("status") == "ok":
            catalog = {
                "status": "unavailable",
                "reason": f"failed to restore starmask after cataloging: {restore_error}",
            }

    reference_degraded = bool(
        catalog.get("status") == "ok"
        and catalog.get("reference_degraded", False)
    )
    pipeline._stage9_star_reference_degraded = reference_degraded
    pipeline._stage9_star_reference_primary_summary = primary_summary
    pipeline._stage9_star_reference_catalog = catalog
    summary = stage9_quality.star_reference_summary(catalog)
    pipeline._stage9_star_reference_summary = summary
    spatial_scale = dict(
        getattr(pipeline, "_stage9_spatial_scale", {}) or {}
    )
    if spatial_scale.get("status") == "ready":
        messages.append(
            "Stage9 FWHM spatial scale frozen "
            f"source={spatial_scale.get('source')}, "
            f"samples={int(spatial_scale.get('sample_count', 0))}, "
            f"fwhm={float(spatial_scale.get('fwhm_median_px', 0.0)):.3f}px, "
            f"radius_scale={float(spatial_scale.get('radius_scale', 0.0)):.3f}, "
            f"area_scale={float(spatial_scale.get('area_scale', 0.0)):.3f}"
        )
    else:
        reason = str(
            spatial_scale.get("reason_code")
            or "stage9_spatial_scale_unavailable"
        )
        messages.append(f"Stage9 FWHM spatial scale unavailable: {reason}")
    if summary.get("status") == "ok" and not reference_degraded:
        messages.append(
            "Stage9 source-confirmed star reference "
            f"source={summary.get('reference_source', reference_source)}, "
            f"method={summary.get('method', 'starmask_catalog')}, "
            f"components={int(summary.get('component_count', 0))}, "
            f"weak={int(summary.get('weak_component_count', 0))}, "
            f"bright={int(summary.get('bright_component_count', 0))}, "
            f"peak_ratio={float(summary.get('bright_to_weak_peak_ratio', 0.0)):.2f}, "
            f"mixed={str(bool(summary.get('mixed_star_field', False))).lower()}, "
            "detail_percentile="
            f"{float(summary.get('source_detail_percentile', 0.0)):.1f}"
        )
    elif summary.get("status") == "ok":
        reason = str(
            summary.get("reference_degraded_reason")
            or "independent source reference unavailable"
        )
        messages.append(
            "Stage9 star reference degraded to starmask-only catalog; "
            "strict compact support and low-intensity quality-gated remix required "
            f"(reason={reason})"
        )
        pipeline.log.warn(
            "Stage9 source-confirmed star reference unavailable; using a "
            "starmask-only catalog only for the strict low-intensity candidate: "
            f"{reason}"
        )
    else:
        reason = str(summary.get("reason") or "unknown")
        messages.append(f"Stage9 original starmask reference unavailable: {reason}")
        pipeline.log.warn(
            "Stage9 star reference unavailable after starmask fallback: "
            f"{reason}"
        )
    return catalog


def _prepare_stage9_star_color_repair(
    pipeline,
    starmask_name: str,
    messages: List[str],
) -> str:
    """Create and validate a reversible reference-driven star-color candidate."""
    report: Dict[str, Any] = {
        "schema": "starun.star-color-repair.v1",
        "status": "not_run",
        "accepted": False,
    }
    pipeline._stage9_star_color_reference_samples = None
    if not bool(
        getattr(pipeline.cfg, "stage9_star_color_repair_enabled", False)
    ):
        report.update(status="disabled", issues=["disabled_by_configuration"])
        pipeline._stage9_star_color_repair_report = report
        pipeline._write_stage_json("stage9_star_color_repair.json", report)
        messages.append("Stage9 deterministic star-color repair disabled")
        return starmask_name
    try:
        pipeline.cmd_with_check("load", starmask_name)
        if not pipeline._save_stage_output("starmask_pre_color_repair"):
            raise RuntimeError("immutable star-color baseline save failed")
        get_pixels = getattr(pipeline.siril, "get_image_pixeldata", None)
        if not callable(get_pixels):
            raise RuntimeError("star-layer pixel reader unavailable")
        star_data = get_pixels(preview=False)
        if star_data is None:
            raise RuntimeError("star-layer pixels unavailable")
        star_pixels = np.array(star_data, copy=True)

        reference_pixels = None
        reference_source = ""
        process_dir = getattr(pipeline, "process_dir", None)
        for source_stem in ("stage5_linear", "stage6_input", "working"):
            source_path = (
                process_dir / f"{source_stem}.fit"
                if process_dir is not None
                else None
            )
            if source_path is None or not source_path.exists():
                continue
            pipeline.cmd_with_check("load", source_stem)
            source_data = get_pixels(preview=False)
            if source_data is not None:
                reference_pixels = np.array(source_data, copy=True)
                reference_source = source_stem
                break
        if reference_pixels is None:
            raise RuntimeError("immutable linear with-stars reference unavailable")

        validation_support_mask = None
        catalog = getattr(pipeline, "_stage9_star_reference_catalog", None)
        if isinstance(catalog, dict) and catalog.get("status") == "ok":
            _weak_support, _bright_support, validation_support_mask = (
                stage9_quality.build_star_overlay_masks(
                    catalog,
                    strict=False,
                    cfg=pipeline.cfg,
                )
            )
        candidate, report = repair_star_layer_colors(
            star_pixels,
            reference_pixels,
            strength=float(
                getattr(pipeline.cfg, "stage9_star_color_repair_strength", 0.72)
            ),
            support_coverage_max=float(
                getattr(
                    pipeline.cfg,
                    "stage9_star_color_support_ratio_max",
                    0.12,
                )
            ),
            chroma_improvement_min=float(
                getattr(
                    pipeline.cfg,
                    "stage9_star_color_improvement_min",
                    0.01,
                )
            ),
            validation_support_mask=validation_support_mask,
        )
        report["reference_source"] = reference_source
        report["transaction"]["baseline_saved"] = True
        reference_samples = report.get("_reference_samples")
        public_report = public_star_color_report(report)
        if not bool(report.get("accepted", False)):
            pipeline.cmd_with_check("load", starmask_name)
            public_report["transaction"]["rollback_performed"] = False
            pipeline._stage9_star_color_repair_report = public_report
            pipeline._write_stage_json(
                "stage9_star_color_repair.json",
                public_report,
            )
            messages.append(
                "Stage9 deterministic star-color candidate rejected: "
                + ",".join(report.get("issues") or [])
            )
            return starmask_name

        pipeline.cmd_with_check("load", starmask_name)
        writer = getattr(pipeline, "_set_current_image_pixeldata", None)
        if callable(writer):
            writer(candidate, label="Stage9 deterministic star-color repair")
        else:
            set_pixels = getattr(pipeline.siril, "set_image_pixeldata", None)
            if not callable(set_pixels):
                raise RuntimeError("star-layer pixel writer unavailable")
            lock_factory = getattr(pipeline.siril, "image_lock", None)
            if callable(lock_factory):
                with lock_factory():
                    set_pixels(candidate)
            else:
                set_pixels(candidate)
        if not pipeline._save_stage_output("starmask_color_repaired"):
            raise RuntimeError("star-color candidate save failed")
        public_report["transaction"].update(
            candidate_saved=True,
            rollback_performed=False,
        )
        pipeline._stage9_star_color_reference_samples = reference_samples
        pipeline._stage9_star_color_repair_report = public_report
        pipeline._write_stage_json(
            "stage9_star_color_repair.json",
            public_report,
        )
        metrics = public_report.get("metrics") or {}
        messages.append(
            "Stage9 deterministic star-color repair accepted "
            f"(samples={int(metrics.get('reference_sample_count', 0))}, "
            f"chroma_improvement={float(metrics.get('star_chroma_improvement', 0.0)):.3f}, "
            f"flux_drift={float(metrics.get('star_flux_drift', 0.0)):.4f})"
        )
        return "starmask_color_repaired"
    except (
        AttributeError,
        CommandError,
        OSError,
        RuntimeError,
        SirilError,
        TypeError,
        ValueError,
    ) as error:
        report = public_star_color_report(report)
        report.update(status="failed", accepted=False, error=str(error))
        try:
            pipeline.cmd_with_check("load", "starmask_pre_color_repair")
            report.setdefault("transaction", {}).update(
                rollback_performed=True,
            )
        except (CommandError, SirilError) as rollback_error:
            report.setdefault("transaction", {}).update(
                rollback_performed=False,
                rollback_error=str(rollback_error),
            )
        pipeline._stage9_star_color_repair_report = report
        pipeline._write_stage_json("stage9_star_color_repair.json", report)
        messages.append(
            "Stage9 deterministic star-color repair unavailable; "
            f"baseline retained: {error}"
        )
        return starmask_name


def _update_stage9_star_delivery_contract(pipeline) -> bool:
    """Freeze the final Stage9 delivery truth after every route decision."""
    stars_required = bool(getattr(pipeline, "_stage9_stars_required", True))
    stars_applied = bool(getattr(pipeline, "_stage9_stars_applied", False))
    output_contains_stars = bool(
        getattr(pipeline, "_stage9_output_contains_stars", False)
    )
    accepted = bool(
        getattr(pipeline, "_stage9_remix_formally_accepted", False)
        and str(getattr(pipeline, "_stage9_final_source", "") or "")
        == "stage9_remixed"
        and not bool(getattr(pipeline, "_stage9_output_withheld", False))
        and output_contains_stars
        and (not stars_required or stars_applied)
    )
    pipeline._stage9_star_delivery_contract_accepted = accepted
    return accepted


def _write_stage9_quality_report(
    pipeline,
    attempts: List[Dict[str, Any]],
    selected: Optional[Dict[str, Any]],
    *,
    source_stem: str,
    mode: str,
) -> None:
    stars_required = bool(getattr(pipeline, "_stage9_stars_required", True))
    stars_applied = bool(getattr(pipeline, "_stage9_stars_applied", False))
    stars_application_mode = str(
        getattr(pipeline, "_stage9_stars_application_mode", mode) or mode
    )
    output_contains_stars = bool(
        getattr(pipeline, "_stage9_output_contains_stars", stars_applied)
    )
    _update_stage9_star_delivery_contract(pipeline)
    upstream_handoff = _stage9_upstream_handoff(pipeline, source_stem)
    stage9_fallback_used, stage9_fallback_reason = _stage9_local_fallback(
        mode,
        selected,
        stars_application_mode,
    )
    pipeline._stage9_fallback_used = stage9_fallback_used
    pipeline._stage9_fallback_reason = stage9_fallback_reason
    pipeline._stage9_delivery_fallback_used = stage9_fallback_used
    support_preflight = dict(
        getattr(pipeline, "_stage9_starmask_support_preflight", {}) or {}
    )
    support_route = str(support_preflight.get("route") or "unavailable")
    remix_base_stem = str(
        getattr(pipeline, "_stage9_remix_base_stem", source_stem)
        or source_stem
    )

    def infer_recovery_kind(attempt_report: Dict[str, Any]) -> str:
        explicit = str(attempt_report.get("recovery_kind") or "")
        if explicit:
            return explicit
        attempt_name = str(attempt_report.get("attempt") or "").lower()
        if "local_chroma" in attempt_name:
            return "local_chroma_attenuation"
        if "soft_psf" in attempt_name or "selective_size" in attempt_name:
            return "group_fractional_source_wing"
        if "psf_support_recovery" in attempt_name:
            return "legacy_integer_source_wing"
        if "source_presence" in attempt_name:
            return "source_presence_extension"
        if "compact" in attempt_name or "reference_degraded_strict" in attempt_name:
            return "strict_support_switch"
        if "fallback" in attempt_name:
            return "global_intensity_feasibility"
        return "none"

    for attempt_order, attempt_report in enumerate(attempts):
        if not isinstance(attempt_report, dict):
            continue
        if "review_eligibility" not in attempt_report:
            attempt_report["review_eligibility"] = (
                _stage9_review_candidate_eligibility(
                    pipeline,
                    attempt_report,
                    attempt_order=attempt_order,
                )
            )
        attempt_report.setdefault("support_preflight_route", support_route)
        attempt_report.setdefault("parent_attempt", None)
        attempt_report.setdefault("base_source_stem", remix_base_stem)
        attempt_report.setdefault(
            "support_starmask",
            attempt_report.get("starmask"),
        )
        attempt_report.setdefault(
            "support_mode",
            str(attempt_report.get("support_mode") or "unknown"),
        )
        attempt_report.setdefault(
            "recovery_kind",
            infer_recovery_kind(attempt_report),
        )
        attempt_report.setdefault("recovery_strength", 0.0)
        attempt_report.setdefault("recovery_target_groups", [])
    if isinstance(selected, dict):
        selected.setdefault("support_preflight_route", support_route)
        selected.setdefault("parent_attempt", None)
        selected.setdefault("base_source_stem", remix_base_stem)
        selected.setdefault("support_starmask", selected.get("starmask"))
        selected.setdefault(
            "support_mode",
            str(selected.get("support_mode") or "unknown"),
        )
        selected.setdefault("recovery_kind", infer_recovery_kind(selected))
        selected.setdefault("recovery_strength", 0.0)
        selected.setdefault("recovery_target_groups", [])
    spatial_scale = dict(
        getattr(pipeline, "_stage9_spatial_scale", {}) or {}
    )
    scale_source = str(spatial_scale.get("source") or "unavailable")
    for attempt_report in attempts:
        if not isinstance(attempt_report, dict):
            continue
        attempt_report.setdefault("spatial_scale_source", scale_source)
        attempt_report.setdefault(
            "spatial_scale_anchor_fwhm_px",
            spatial_scale.get("anchor_fwhm_px", 4.0),
        )
        nominal_retry = int(
            attempt_report.get("psf_support_recovery_pixels", 0) or 0
        )
        attempt_report.setdefault(
            "psf_support_recovery_pixels_nominal",
            nominal_retry,
        )
        catalog = getattr(pipeline, "_stage9_star_reference_catalog", {}) or {}
        radii = np.asarray(catalog.get("_psf_support_radii", ()))
        if radii.size:
            attempt_report.setdefault(
                "effective_support_radius_px",
                stage9_quality.stage9_effective_pixel_stats(radii),
            )
    if isinstance(selected, dict):
        selected.setdefault("spatial_scale_source", scale_source)
        selected.setdefault(
            "spatial_scale_anchor_fwhm_px",
            spatial_scale.get("anchor_fwhm_px", 4.0),
        )
    writer = getattr(pipeline, "_write_stage_json", None)
    if not callable(writer):
        return
    sep_summary = getattr(pipeline, "_stage9_sep_crossmatch_summary", None)
    if not isinstance(sep_summary, dict) or not sep_summary.get(
        "artifact_sha256"
    ):
        sep_evidence = getattr(pipeline, "_stage9_sep_crossmatch_report", None)
        if not isinstance(sep_evidence, dict):
            sep_evidence = stage9_quality.stage9_sep_crossmatch_not_applicable(
                "Stage9 did not enter a formal persisted remix route"
            )
        sep_summary = _persist_stage9_sep_crossmatch_evidence(
            pipeline,
            sep_evidence,
        )
    pipeline.log.info(
        "[Stage9] star application contract "
        f"required={str(stars_required).lower()}, "
        f"applied={str(stars_applied).lower()}, "
        f"output_contains_stars={str(output_contains_stars).lower()}, "
        f"mode={stars_application_mode}"
    )
    report_formula = str(
        (selected or {}).get("formula")
        or (attempts[-1].get("formula") if attempts else "none")
    )
    matched_context = getattr(pipeline, "_stage9_matched_domain_context", None)
    matched_report = (
        (matched_context or {}).get("report")
        if isinstance(matched_context, dict)
        else None
    )
    selected_quality = (
        selected
        if isinstance(selected, dict)
        else attempts[-1]
        if attempts
        else None
    )
    formal_accepted = bool(
        getattr(
            pipeline,
            "_stage9_remix_formally_accepted",
            bool(selected and selected.get("accepted", False) and stars_applied),
        )
    )
    review_candidate_selected = bool(
        getattr(pipeline, "_stage9_review_candidate_selected", False)
    )
    selected_attempt = str((selected or {}).get("attempt") or "").lower()
    candidate_recovery_used = bool(
        isinstance(selected, dict)
        and (
            str(selected.get("recovery_kind") or "none") != "none"
            or selected.get("parent_attempt") is not None
            or "unscreen" in selected_attempt
            or "compact" in selected_attempt
            or "reference_degraded_strict" in selected_attempt
        )
    )
    pipeline._stage9_candidate_recovery_used = candidate_recovery_used
    if formal_accepted:
        selection_class = "formal"
    elif review_candidate_selected:
        selection_class = "review_candidate"
    elif mode == "stage8_starmask_review_fallback":
        selection_class = "stage8_starmask_fallback"
    elif mode == "stage5_review_fallback":
        selection_class = "stage5_fallback"
    elif bool(getattr(pipeline, "_stage9_output_withheld", False)):
        selection_class = "withheld"
    elif output_contains_stars:
        selection_class = "with_stars_fallback"
    else:
        selection_class = "none"
    unscreen_report = getattr(
        pipeline,
        "_stage9_unscreen_reference",
        _stage9_unscreen_unavailable("not attempted"),
    )
    linear_roundtrip = (
        (matched_report or {}).get("linear_decomposition_roundtrip")
        if isinstance(matched_report, dict)
        else None
    ) or {
        "status": "unavailable",
        "reason": "verified linear Stage6 pair is unavailable",
    }
    unscreen_operator_audit = (
        (unscreen_report or {}).get("operator_audit")
        if isinstance(unscreen_report, dict)
        else None
    ) or {
        "status": "unavailable",
        "reason": str(
            (
                (unscreen_report or {}).get("reason")
                or (unscreen_report or {}).get("reason_code")
                or "Unscreen candidate was not prepared"
            )
            if isinstance(unscreen_report, dict)
            else "Unscreen candidate was not prepared"
        ),
    }
    selected_composition_fidelity = (
        (selected_quality or {}).get("reference_fidelity")
        if isinstance(selected_quality, dict)
        else None
    ) or {
        "status": "unavailable",
        "reason": "selected candidate matched-domain fidelity is unavailable",
    }
    attempt_history_reason_codes = list(
        dict.fromkeys(
            [
                *(
                    [str((unscreen_report or {}).get("reason_code"))]
                    if isinstance(unscreen_report, dict)
                    and (unscreen_report or {}).get("reason_code")
                    else []
                ),
                *[
                    str(code)
                    for attempt_report in attempts
                    if isinstance(attempt_report, dict)
                    for code in (
                        [attempt_report.get("reason_code")]
                        + list(attempt_report.get("reason_codes") or [])
                    )
                    if str(code or "").strip()
                ],
            ]
        )
    )
    final_reason_codes = list(
        dict.fromkeys(
            [
                *(
                    [str(selected.get("reason_code"))]
                    if isinstance(selected, dict) and selected.get("reason_code")
                    else []
                ),
                *(
                    [
                        str(code)
                        for code in (selected.get("reason_codes") or [])
                        if str(code)
                    ]
                    if isinstance(selected, dict)
                    else []
                ),
                *([stage9_fallback_reason] if stage9_fallback_reason else []),
                *(
                    ["stage9_output_withheld"]
                    if bool(getattr(pipeline, "_stage9_output_withheld", False))
                    else []
                ),
            ]
        )
    )
    if formal_accepted and not final_reason_codes:
        final_reason_codes = ["accepted"]
    writer(
        "stage9_remix_quality.json",
        {
            "schema": "starun.stage9-remix-quality.v10",
            "selection_policy": (
                "sep_catalog_visibility_psf_fidelity_recovery_v8"
            ),
            "selection_class": selection_class,
            "formal_accepted": formal_accepted,
            "review_candidate_selected": review_candidate_selected,
            "review_candidate_policy": {
                "mode": "best_bounded_formal_failure_review_only",
                "enabled_for_failure_action": "auto_fallback",
                "formal_fwhm_ratio_min": float(
                    getattr(pipeline.cfg, "stage9_psf_fwhm_ratio_min", 0.93)
                ),
                "formal_fwhm_ratio_max": float(
                    getattr(pipeline.cfg, "stage9_psf_fwhm_ratio_max", 1.10)
                ),
                "review_fwhm_ratio_max": _stage9_review_fwhm_ratio_max(
                    pipeline
                ),
                "lower_fwhm_failure_action": (
                    "stage8_starmask_minimal_fallback"
                ),
                "maximum_numeric_failure_count": 1,
                "non_psf_hard_failure_allowed": False,
                "existing_numeric_advisory_multiplier_reused": True,
                "additional_numeric_hard_failure_multiplier": 1.0,
                "selection_order": [
                    "minimum_worst_numeric_severity",
                    "minimum_formal_failure_count",
                    "maximum_minimum_star_recovery_margin",
                    "maximum_remix_intensity",
                    "minimum_highlight_clip_growth",
                    "minimum_bright_pixel_growth",
                    "original_attempt_order",
                ],
            },
            "stage9_spatial_scale": getattr(
                pipeline,
                "_stage9_spatial_scale",
                {
                    "schema": "starun.stage9-fwhm-spatial-scale.v1",
                    "status": "unavailable",
                    "reason_code": "stage9_spatial_scale_unavailable",
                },
            ),
            "scaled_catalog_validation": getattr(
                pipeline,
                "_stage9_scaled_catalog_validation",
                {
                    "schema": (
                        "starun.stage9-scaled-catalog-validation.v1"
                    ),
                    "status": "not_run",
                    "reason": "Stage9 star reference was not prepared",
                },
            ),
            "objective": {
                "reconstruction_target": (
                    "faithful_same_source_star_display_restoration"
                ),
                "authenticity": "same_source_controlled_display_reconstruction",
                "scientific_photometry_claim": False,
                "synthetic_or_replacement_stars": False,
            },
            "operator_contract": {
                "source_role": "verified_with_stars_source",
                "backdrop_role": "processed_starless_remix_base",
                "linear_decomposition": "star_layer = O_linear - B_linear",
                "display_composition": "Screen(B, k*stars)",
                "operation_order": [
                    "scale_star_rgb_once",
                    "screen_with_backdrop",
                    "interpolate_backdrop_to_screen_result_by_spatial_alpha",
                ],
                "alpha_semantics": "binary_compact_spatial_support",
                "premultiplied_alpha": False,
                "intensity_semantics": (
                    "single_rgb_scalar_with_weak_star_floor_before_screen"
                ),
                "working_range": "normalized_float_0_1",
                "final_display_is_photometric": False,
            },
            "source_autostretch_wing_reference": (
                (
                    (matched_report or {}).get(
                        "source_autostretch_wing_reference"
                    )
                    if isinstance(matched_report, dict)
                    else None
                )
                or {
                    "status": "unavailable",
                    "available": False,
                    "reason": "matched-domain context is unavailable",
                }
            ),
            "zero_edit_operator_audit": {
                "schema": "starun.stage9-operator-audit.v1",
                "linear_decomposition_roundtrip": linear_roundtrip,
                "raw_unscreen_and_stabilization": unscreen_operator_audit,
                "selected_composition_fidelity": selected_composition_fidelity,
            },
            "mode": mode,
            "formula": report_formula,
            "source_stem": source_stem,
            "delivery_input_source_stem": source_stem,
            "upstream_source_stem": str(
                getattr(pipeline, "_stage9_upstream_source_stem", source_stem)
                or source_stem
            ),
            "remix_base_stem": remix_base_stem,
            "quality_gate_enabled": bool(
                getattr(pipeline.cfg, "stage9_quality_gate_enabled", True)
            ),
            "processing_mode": str(
                getattr(pipeline.cfg, "stage9_processing_mode", "auto") or "auto"
            ),
            "failure_action": str(
                getattr(pipeline.cfg, "stage9_failure_action", "auto_fallback")
                or "auto_fallback"
            ),
            "fallback_retry_max": int(
                getattr(pipeline.cfg, "stage9_fallback_retry_max", 3) or 0
            ),
            "targeted_recovery_enabled": bool(
                getattr(
                    pipeline.cfg,
                    "stage9_targeted_recovery_enabled",
                    True,
                )
            ),
            "targeted_recovery_retry_max": _stage9_targeted_recovery_retry_max(
                pipeline
            ),
            "fallback_intensity_floor": float(
                getattr(pipeline.cfg, "stage9_fallback_intensity_floor", 0.40)
                or 0.40
            ),
            "psf_size_recovery_policy": {
                "mode": (
                    "bidirectional_group_psf_recovery_with_bounded_binary_search"
                    if bool(
                        getattr(
                            pipeline.cfg,
                            "stage9_targeted_recovery_enabled",
                            True,
                        )
                    )
                    else "progressive_source_support_then_same_source_autostretch_visible_wing"
                ),
                "gate_enabled": bool(
                    getattr(pipeline.cfg, "stage9_psf_size_gate_enabled", True)
                ),
                "hard_fwhm_ratio_min": float(
                    getattr(pipeline.cfg, "stage9_psf_fwhm_ratio_min", 0.93)
                ),
                "hard_fwhm_ratio_max": float(
                    getattr(pipeline.cfg, "stage9_psf_fwhm_ratio_max", 1.10)
                ),
                "soft_recovery_target_min": _stage9_psf_recovery_target_min(
                    pipeline,
                    selected_quality or {},
                ),
                "soft_recovery_target_max": _stage9_psf_recovery_target_max(
                    pipeline,
                    selected_quality or {},
                ),
                "oversize_contraction": {
                    "trigger": "pure_fwhm_upper_limit_failure_only",
                    "target_groups": "failed_weak_bright_or_all_only",
                    "operator": "component_local_rgb_shared_u_power",
                    "operator_formula": "gain=u^(gamma-1)",
                    "gamma_bounds": [1.0, 4.0],
                    "retry_budget_shared_with_targeted_recovery": True,
                    "candidate_rebuild_source": "immutable_parent_star_layer",
                    "peak_preserved": True,
                    "centroid_guard_max_px": 0.05,
                    "channel_ratio_preserved_by_construction": True,
                    "all_existing_formal_gates_reapplied": True,
                    "rollback": "exact_immutable_parent",
                },
                "selective_low_tail_enabled": bool(
                    getattr(
                        pipeline.cfg,
                        "stage9_psf_selective_wing_enabled",
                        True,
                    )
                ),
                "selective_low_tail_target_ratio": float(
                    getattr(
                        pipeline.cfg,
                        "stage9_psf_selective_wing_target_ratio",
                        1.08,
                    )
                ),
                "selective_low_tail_strength_max": float(
                    getattr(
                        pipeline.cfg,
                        "stage9_psf_selective_wing_strength_max",
                        1.15,
                    )
                ),
                "visible_wing_reference_enabled": bool(
                    getattr(
                        pipeline.cfg,
                        "stage9_source_autostretch_wing_reference_enabled",
                        True,
                    )
                ),
                "visible_wing_floor_peak_fraction": float(
                    getattr(
                        pipeline.cfg,
                        "stage9_source_autostretch_wing_floor_fraction",
                        0.05,
                    )
                ),
                "visible_wing_target_ratio": float(
                    getattr(
                        pipeline.cfg,
                        "stage9_source_autostretch_wing_target_ratio",
                        1.03,
                    )
                ),
                "visible_wing_radius_max": int(
                    getattr(
                        pipeline.cfg,
                        "stage9_source_autostretch_wing_radius_max",
                        10,
                    )
                ),
                "pixel_geometry": {
                    "anchor_fwhm_px": 4.0,
                    "radius_formula": "nominal_px * (FWHM_px / 4.0_px)",
                    "area_formula": "nominal_px2 * (FWHM_px / 4.0_px)^2",
                    "candidate_pixels_are_nominal_anchor_levels": True,
                },
                "candidate_pixels": list(
                    range(
                        0,
                        max(
                            0,
                            min(
                                2,
                                int(
                                    getattr(
                                        pipeline.cfg,
                                        "stage9_psf_support_retry_pixels",
                                        2,
                                    )
                                    or 0
                                ),
                            ),
                        )
                        + 1,
                    )
                ),
                "selection_order": [
                    "minimum_worst_group_absolute_fwhm_error_from_1",
                    "minimum_mean_group_absolute_fwhm_error_from_1",
                    "minimum_added_source_wing_pixels",
                    "minimum_highlight_clip_growth",
                    "minimum_bright_pixel_growth",
                ],
                "candidate_rebuild_source": "immutable_trusted_star_layer",
                "synthetic_or_recursive_dilation": False,
            },
            "upstream_handoff": upstream_handoff,
            "upstream_passthrough": bool(
                upstream_handoff.get("passthrough", False)
            ),
            "upstream_restricted": bool(
                upstream_handoff.get("restricted_downstream", False)
            ),
            "stage9_fallback_used": stage9_fallback_used,
            "fallback_used": stage9_fallback_used,
            "stage9_fallback_reason": stage9_fallback_reason,
            "candidate_recovery_used": candidate_recovery_used,
            "delivery_fallback_used": stage9_fallback_used,
            "fallback_remix": copy.deepcopy(
                getattr(
                    pipeline,
                    "_stage9_fallback_remix_report",
                    {
                        "schema": "starun.stage9-fallback-remix.v1",
                        "status": "not_attempted",
                        "attempts": [],
                    },
                )
            ),
            "psf_review_required": bool(
                getattr(pipeline, "_stage9_psf_review_required", False)
            ),
            "remix_formally_accepted": formal_accepted,
            "star_delivery_contract_accepted": bool(
                getattr(
                    pipeline,
                    "_stage9_star_delivery_contract_accepted",
                    False,
                )
            ),
            "bright_core_with_stars_fallback": copy.deepcopy(
                getattr(pipeline, "_bright_core_with_stars_fallback", {}) or {}
            ),
            "stars_required": stars_required,
            "stars_applied": stars_applied,
            "output_contains_stars": output_contains_stars,
            "output_withheld": bool(
                getattr(pipeline, "_stage9_output_withheld", False)
            ),
            "stars_application_mode": stars_application_mode,
            "final_source": str(
                getattr(pipeline, "_stage9_final_source", "") or ""
            ),
            "final_delivery_source_stem": str(
                getattr(pipeline, "_stage9_final_source", "") or ""
            ),
            "star_reference": getattr(
                pipeline,
                "_stage9_star_reference_summary",
                {"status": "unavailable", "reason": "not prepared"},
            ),
            "star_catalog_confirmation": {
                "method": "trusted_starmask_candidates_confirmed_in_mtf_O",
                "reference": getattr(
                    pipeline,
                    "_stage9_star_reference_summary",
                    {"status": "unavailable", "reason": "not prepared"},
                ),
            },
            "source_presence": getattr(
                pipeline,
                "_stage9_source_presence_report",
                {
                    "schema": "starun.stage9-source-presence.v1",
                    "status": "not_run",
                    "available": False,
                },
            ),
            "matched_domain_reference": matched_report,
            "stage6_pair_handoff": (
                (matched_report or {}).get("pair_handoff")
                if isinstance(matched_report, dict)
                else None
            ),
            "star_reference_degraded": bool(
                getattr(pipeline, "_stage9_star_reference_degraded", False)
            ),
            "star_reference_primary": getattr(
                pipeline,
                "_stage9_star_reference_primary_summary",
                None,
            ),
            "starmask_calibration": getattr(
                pipeline,
                "_stage9_starmask_calibration",
                None,
            ),
            "starmask_support_preflight": getattr(
                pipeline,
                "_stage9_starmask_support_preflight",
                {
            "schema": "starun.stage9-starmask-support-preflight.v2",
                    "status": "unavailable",
                    "reason": "support preflight was not recorded",
                },
            ),
            "starmask_preparation_failed": bool(
                getattr(pipeline, "_stage9_starmask_preparation_failed", False)
            ),
            "starmask_preparation_failure_reason": str(
                getattr(
                    pipeline,
                    "_stage9_starmask_preparation_failure_reason",
                    "",
                )
                or ""
            ),
            "starmask_stretch_failed": bool(
                getattr(pipeline, "_stage9_starmask_stretch_failed", False)
            ),
            "star_color_repair": getattr(
                pipeline,
                "_stage9_star_color_repair_report",
                {
                    "schema": "starun.star-color-repair.v1",
                    "status": "not_run",
                    "accepted": False,
                },
            ),
            "star_color_post_validation": getattr(
                pipeline,
                "_stage9_star_color_post_validation",
                None,
            ),
            "star_color_post_validation_gate_enabled": bool(
                getattr(
                    pipeline.cfg,
                    "stage9_star_color_post_validation_enabled",
                    True,
                )
            ),
            "star_plugin_preprocessing": getattr(
                pipeline,
                "_stage9_star_plugin_preprocessing",
                {"status": "not_run", "applied_steps": []},
            ),
            "remix_base_identity": getattr(
                pipeline,
                "_stage9_remix_base_identity",
                None,
            ),
            "star_layer_decomposition": {
                "baseline": "linear_original_minus_starless_stretched",
                "selected": str(
                    getattr(
                        pipeline,
                        "_stage9_star_layer_decomposition",
                        "linear_original_minus_starless_stretched",
                    )
                    or "linear_original_minus_starless_stretched"
                ),
                "final_composition": "screen",
            },
            "unscreen_reference": unscreen_report,
            "reason_codes": attempt_history_reason_codes,
            "reason_codes_scope": "attempt_history_legacy",
            "attempt_history_reason_codes": attempt_history_reason_codes,
            "final_reason_codes": final_reason_codes,
            "shadow_metrics": (
                (selected or {}).get("shadow_metrics")
                if isinstance(selected, dict)
                else (attempts[-1].get("shadow_metrics") if attempts else None)
            ),
            "psf_closure": (
                (selected or {}).get("psf_closure")
                if isinstance(selected, dict)
                else (attempts[-1].get("psf_closure") if attempts else None)
            ),
            "catalog_visibility": (
                (selected or {}).get("catalog_visibility")
                if isinstance(selected, dict)
                else (
                    attempts[-1].get("catalog_visibility")
                    if attempts
                    else None
                )
            ),
            "sep_crossmatch": copy.deepcopy(sep_summary),
            "persisted_output_validation": copy.deepcopy(
                getattr(
                    pipeline,
                    "_stage9_persisted_output_validation",
                    {
                        "schema": (
                            "starun.stage9-persisted-output-validation.v1"
                        ),
                        "status": "not_run",
                        "accepted": False,
                    },
                )
            ),
            "visible_wing_closure": (
                (selected or {}).get("visible_wing_closure")
                if isinstance(selected, dict)
                else (
                    attempts[-1].get("visible_wing_closure")
                    if attempts
                    else None
                )
            ) or {
                "schema": "starun.stage9-visible-wing-closure.v1",
                "status": "unavailable",
                "available": False,
                "reason": (
                    "selected candidate has no linked-autostretch wing audit"
                ),
            },
            "attempts": attempts,
            "selected": selected,
        },
    )


def _append_stage9_review_bundle(
    pipeline,
    messages: List[str],
    attempts: List[Dict[str, Any]],
    selected: Optional[Dict[str, Any]],
    *,
    source_stem: str,
    mode: str,
    stage_saved: bool,
) -> None:
    """Create Stage 9 before/after evidence for accepted and degraded outputs."""
    creator = getattr(pipeline, "_create_stage_review_bundle", None)
    if not stage_saved or not callable(creator):
        return
    review_candidates = list(attempts)
    selected_attempt = str((selected or {}).get("attempt") or "").strip() or None
    if attempts and selected is None:
        selected_attempt = "stage9_safe_rollback"
        review_candidates.append(
            {
                "id": selected_attempt,
                "name": "stage9_remixed",
                "status": "selected",
                "selected": True,
                "reason": "all remix candidates rejected; retained safe source",
            }
        )
    review = creator(
        "stage9_star_remixing",
        source_stem,
        "stage9_remixed",
        context={
            "mode": mode,
            "upstream_source_stem": str(
                getattr(pipeline, "_stage9_upstream_source_stem", source_stem)
                or source_stem
            ),
            "remix_base_stem": source_stem,
            "stars_required": bool(
                getattr(pipeline, "_stage9_stars_required", True)
            ),
            "stars_applied": bool(
                getattr(pipeline, "_stage9_stars_applied", False)
            ),
            "remix_formally_accepted": bool(
                getattr(pipeline, "_stage9_remix_formally_accepted", False)
            ),
            "review_candidate_selected": bool(
                getattr(pipeline, "_stage9_review_candidate_selected", False)
            ),
            "stars_application_mode": str(
                getattr(pipeline, "_stage9_stars_application_mode", mode) or mode
            ),
            "star_reference": getattr(
                pipeline,
                "_stage9_star_reference_summary",
                {"status": "unavailable", "reason": "not prepared"},
            ),
            "star_reference_degraded": bool(
                getattr(pipeline, "_stage9_star_reference_degraded", False)
            ),
            "star_reference_primary": getattr(
                pipeline,
                "_stage9_star_reference_primary_summary",
                None,
            ),
            "source_presence": getattr(
                pipeline,
                "_stage9_source_presence_report",
                None,
            ),
            "starmask_calibration": getattr(
                pipeline,
                "_stage9_starmask_calibration",
                None,
            ),
            "starmask_support_preflight": getattr(
                pipeline,
                "_stage9_starmask_support_preflight",
                None,
            ),
            "starmask_preparation_failed": bool(
                getattr(pipeline, "_stage9_starmask_preparation_failed", False)
            ),
            "star_plugin_preprocessing": getattr(
                pipeline,
                "_stage9_star_plugin_preprocessing",
                {"status": "not_run", "applied_steps": []},
            ),
            "quality_gate_enabled": bool(
                getattr(pipeline.cfg, "stage9_quality_gate_enabled", True)
            ),
            "star_color_post_validation_gate_enabled": bool(
                getattr(
                    pipeline.cfg,
                    "stage9_star_color_post_validation_enabled",
                    True,
                )
            ),
        },
        candidates=review_candidates,
        selected_candidate=selected_attempt,
    )
    if review.get("status") == "ready":
        messages.append(f"review_bundle={review['report_path']}")


def _stage9_starmask_support_preflight(
    pipeline,
    starmask_name: str,
    *,
    star_stretch_used: bool,
    failure_action: str,
    messages: List[str],
) -> Dict[str, Any]:
    """Read starmask pixels and classify normal/strict support before remix."""
    try:
        spatial_scale = dict(
            getattr(pipeline, "_stage9_spatial_scale", {}) or {}
        )
        if spatial_scale.get("status") != "ready":
            raise RuntimeError("stage9_spatial_scale_unavailable")
        get_pixels = getattr(pipeline.siril, "get_image_pixeldata", None)
        if not callable(get_pixels):
            raise RuntimeError("Siril pixel reader is unavailable")
        pipeline.cmd_with_check("load", starmask_name)
        raw_pixels = get_pixels(preview=False)
        if raw_pixels is None:
            raise RuntimeError("raw starmask pixels are unavailable")
        plugin_pixels = None
        if star_stretch_used:
            pipeline.cmd_with_check("load", "starmask_stretched")
            plugin_pixels = get_pixels(preview=False)
            if plugin_pixels is None:
                raise RuntimeError("plugin-stretched starmask pixels are unavailable")
        preflight = stage9_quality.assess_starmask_support_preflight(
            np.asarray(raw_pixels),
            pipeline.cfg,
            reference_catalog=getattr(
                pipeline,
                "_stage9_star_reference_catalog",
                None,
            ),
            failure_action=failure_action,
            plugin_stretched_stars=(
                np.asarray(plugin_pixels) if plugin_pixels is not None else None
            ),
        )
        public = stage9_quality.public_starmask_support_preflight(preflight)
        pipeline._stage9_starmask_support_preflight = public
        if star_stretch_used:
            plugin_eligibility = dict(
                public.get("plugin_formal_eligibility") or {}
            )
            preprocessing = dict(
                getattr(pipeline, "_stage9_star_plugin_preprocessing", {}) or {}
            )
            preprocessing.update(
                formal_eligible=bool(plugin_eligibility.get("eligible", False)),
                output_adequacy=plugin_eligibility,
                selected_stretch_source=public.get("selected_stretch_source"),
            )
            if not bool(plugin_eligibility.get("eligible", False)):
                preprocessing.update(
                    status="plugin_stretch_rejected_output_inadequate",
                    builtin_stretch_required=True,
                    fallback_reason=public.get("fallback_reason"),
                )
                messages.append(
                    "Stage9 rejected the plugin-stretched star layer because "
                    "its measured four-anchor output was inadequate; rebuilt "
                    "the normal candidate from the raw starmask"
                )
            else:
                preprocessing.update(
                    status="plugin_stretched_formally_qualified",
                    builtin_stretch_required=False,
                )
            pipeline._stage9_star_plugin_preprocessing = preprocessing
        normal = ((public.get("candidates") or {}).get("normal") or {})
        strict = ((public.get("candidates") or {}).get("strict_compact") or {})
        messages.append(
            "Stage9 starmask support preflight "
            f"route={public.get('route')}, "
            f"normal={normal.get('risk_level', 'unknown')}/"
            f"{float(normal.get('support_coverage', 0.0) or 0.0):.4f}, "
            f"strict={strict.get('risk_level', 'unknown')}/"
            f"{float(strict.get('support_coverage', 0.0) or 0.0):.4f}"
        )
        return preflight
    except (
        AttributeError,
        CommandError,
        RuntimeError,
        SirilError,
        TypeError,
        ValueError,
    ) as error:
        report = {
            "schema": "starun.stage9-starmask-support-preflight.v2",
            "status": "rejected",
            "strategy": "adaptive_dual_route",
            "compact_enabled": bool(
                getattr(
                    pipeline.cfg,
                    "stage9_compact_starmask_enabled",
                    True,
                )
            ),
            "compact_support_enabled": bool(
                getattr(
                    pipeline.cfg,
                    "stage9_compact_starmask_enabled",
                    True,
                )
            ),
            "pre_stretch_compact_enabled": bool(
                getattr(
                    pipeline.cfg,
                    "stage9_starmask_pre_stretch_compact_enabled",
                    False,
                )
            ),
            "route": "unavailable",
            "reason_code": "stage9_support_preflight_unavailable",
            "reason": str(error),
            "planned_candidates": [],
            "skipped_candidates": [
                {
                    "support_mode": support_mode,
                    "reason_code": "stage9_support_preflight_unavailable",
                    "status": "unavailable",
                    "risk_level": "unavailable",
                    "reason": str(error),
                }
                for support_mode in ("normal", "strict_compact")
            ],
            "executed_candidates": [],
            "selected_support_mode": None,
            "_calibrations": {},
        }
        pipeline._stage9_starmask_support_preflight = (
            stage9_quality.public_starmask_support_preflight(report)
        )
        messages.append(f"Stage9 starmask support preflight failed: {error}")
        return report


def _prepare_stage9_starmask_for_pixel_remix(
    pipeline,
    starmask_name: str,
    *,
    star_stretch_used: bool,
    messages: List[str],
    strict_support: bool = False,
    support_retry_pixels: int = 0,
    output_name: str = "starmask_stretched",
    precomputed_calibration: Dict[str, Any] | None = None,
    candidate_local: bool = False,
    compact_output_name: str | None = None,
    allow_pre_stretch_compact: bool = True,
) -> str:
    stretch_execution_started = False
    stretch_enabled = bool(
        getattr(pipeline.cfg, "stage9_starmask_stretch_enabled", True)
    )
    not_run_semantics: Dict[str, Any] = {
        "schema": stage7_stretch_metrics.STRETCH_SEMANTICS_SCHEMA,
        "status": "not_run",
        "engine": "siril",
        "method": "stage9_star_layer_pending",
        "minimum_siril_version": (
            stage7_stretch_metrics.SIRIL_MINIMUM_VERSION_CONTRACT
        ),
        "bundled_reference_version": (
            stage7_stretch_metrics.SIRIL_BUNDLED_REFERENCE_VERSION
        ),
        "scope": "stage9_star_layer",
        "steps": [],
    }
    calibration: Dict[str, Any] = {
        "status": "unavailable",
        "reason": "starmask preparation has not started",
        "stretch_semantics": not_run_semantics,
    }
    stretched_name = output_name
    if star_stretch_used:
        plugin_calibration = {
            key: value
            for key, value in dict(precomputed_calibration or {}).items()
            if not str(key).startswith("_")
        }
        plugin_calibration.update({
            "status": "plugin_stretched",
            "adaptive_status": str(
                (precomputed_calibration or {}).get("status") or "not_run"
            ),
            "reason": "plugin-provided nonlinear star layer",
            "compact_support_enabled": bool(
                getattr(
                    pipeline.cfg,
                    "stage9_compact_starmask_enabled",
                    True,
                )
            ),
            "pre_stretch_compact_enabled": bool(
                allow_pre_stretch_compact
                and getattr(
                    pipeline.cfg,
                    "stage9_starmask_pre_stretch_compact_enabled",
                    False,
                )
            ),
            "compact_layer_applied": False,
            "compact_layer_disabled": True,
            "stretch_semantics": {
                "schema": stage7_stretch_metrics.STRETCH_SEMANTICS_SCHEMA,
                "status": "not_applicable",
                "engine": "external_plugin",
                "method": "plugin_stretched",
                "minimum_siril_version": (
                    stage7_stretch_metrics.SIRIL_MINIMUM_VERSION_CONTRACT
                ),
                "bundled_reference_version": (
                    stage7_stretch_metrics.SIRIL_BUNDLED_REFERENCE_VERSION
                ),
                "reason": "plugin supplied an already nonlinear star layer",
                "steps": [],
            },
        })
        pipeline._stage9_starmask_calibration = plugin_calibration
        messages.append("Stage9 starmask uses plugin-stretched star layer for pixel remix")
        return stretched_name
    try:
        pipeline.cmd_with_check("load", starmask_name)
        calibration = {
            "status": "unavailable",
            "reason": "starmask pixels unavailable",
            "stretch_semantics": not_run_semantics,
        }
        get_pixels = getattr(pipeline.siril, "get_image_pixeldata", None)
        starmask_data = None
        support_mask = None
        if callable(get_pixels):
            starmask_data = get_pixels(preview=False)
            if starmask_data is not None:
                if precomputed_calibration is not None:
                    calibration = dict(precomputed_calibration)
                else:
                    calibration = stage9_quality.calibrate_starmask_asinh(
                        starmask_data,
                        pipeline.cfg,
                        include_support_mask=True,
                        strict_support=strict_support,
                        support_retry_pixels=support_retry_pixels,
                        reference_catalog=getattr(
                            pipeline,
                            "_stage9_star_reference_catalog",
                            None,
                        ),
                    )
                support_mask = calibration.get("_compact_support_mask")
                calibration_advisories = list(
                    calibration.get("advisories") or []
                )
                if calibration_advisories:
                    advisory_text = ", ".join(
                        str(item) for item in calibration_advisories[:3]
                    )
                    messages.append(
                        "Stage9 starmask calibration advisory; continuing: "
                        + advisory_text
                    )
                    pipeline.log.warn(
                        "Stage9 starmask calibration advisory; continuing: "
                        + advisory_text
                    )
        calibration.setdefault("stretch_semantics", not_run_semantics)
        compact_support_enabled = bool(
            getattr(
                pipeline.cfg,
                "stage9_compact_starmask_enabled",
                True,
            )
        )
        pre_stretch_compact_enabled = bool(
            allow_pre_stretch_compact
            and getattr(
                pipeline.cfg,
                "stage9_starmask_pre_stretch_compact_enabled",
                False,
            )
        )
        calibration["compact_support_enabled"] = compact_support_enabled
        calibration[
            "pre_stretch_compact_enabled"
        ] = pre_stretch_compact_enabled
        compact_applied = False
        stretch_input_data = starmask_data

        def write_pixels(pixels, *, label: str) -> bool:
            safe_pixel_writer = getattr(
                pipeline,
                "_set_current_image_pixeldata",
                None,
            )
            if callable(safe_pixel_writer):
                safe_pixel_writer(pixels, label=label)
                return True
            set_pixels = getattr(pipeline.siril, "set_image_pixeldata", None)
            if not callable(set_pixels):
                return False
            lock_factory = getattr(pipeline.siril, "image_lock", None)
            if callable(lock_factory):
                with lock_factory():
                    set_pixels(pixels)
            else:
                pipeline.log.warn(
                    f"{label}: image_lock unavailable, writing pixels without thread lock"
                )
                set_pixels(pixels)
            return True

        if (
            calibration.get("status") == "ok"
            and pre_stretch_compact_enabled
        ):
            if starmask_data is not None and support_mask is not None:
                compact_pixels = stage9_quality.apply_compact_starmask_support(
                    starmask_data,
                    support_mask,
                )
                compact_applied = write_pixels(
                    compact_pixels,
                    label="Stage9 compact starmask",
                )
                if compact_applied:
                    stretch_input_data = compact_pixels
                    calibration["_compact_support_preweighted"] = True
            if compact_applied:
                compact_name = compact_output_name or (
                    "starmask_compact_recovery"
                    if strict_support
                    else "starmask_compact"
                )
                pipeline.cmd_with_check("save", compact_name)
                calibration["compact_layer_applied"] = True
                calibration["compact_layer_stem"] = compact_name
                messages.append(
                    "Stage9 compact starmask applied before Asinh "
                    f"(mode={calibration.get('support_mode', 'normal')}, "
                    f"support={float(calibration.get('compact_support_coverage', 0.0)):.3f}, "
                    "removed_predicted_change="
                    f"{float(calibration.get('removed_predicted_change_ratio', 0.0)):.3f})"
                )
            else:
                calibration["adaptive_status"] = "ok"
                calibration["status"] = "fallback_safe"
                calibration["reason"] = "compact support pixel write unavailable"
                calibration["compact_layer_applied"] = False
        elif calibration.get("status") == "ok":
            calibration["compact_layer_applied"] = False
            calibration["compact_layer_disabled"] = True
            messages.append(
                "Stage9 pre-stretch starmask compaction skipped "
                f"(support_routing={'enabled' if compact_support_enabled else 'normal_only'}, "
                f"mode={calibration.get('support_mode', 'normal')})"
            )
        if calibration.get("status") != "ok":
            calibration.setdefault("compact_layer_applied", compact_applied)
            public_calibration = {
                key: value
                for key, value in calibration.items()
                if not str(key).startswith("_")
            }
            public_calibration["fail_closed"] = True
            pipeline._stage9_starmask_calibration = public_calibration
            reason = str(calibration.get("reason") or "starmask calibration unavailable")
            if not candidate_local and (
                not strict_support
                or bool(
                    getattr(pipeline, "_stage9_star_reference_degraded", False)
                )
            ):
                pipeline._stage9_starmask_preparation_failed = True
                pipeline._stage9_starmask_preparation_failure_reason = reason
            messages.append(
                "Stage9 starmask preparation rejected before stretch execution; "
                "an unmeasured configured Asinh fallback is not eligible "
                f"for formal delivery ({reason})"
            )
            return starmask_name

        stretch = _clamp_float(
            float(calibration["stretch"]),
            1.10,
            1000.0,
        )
        execution_stretch = max(1.10, math.floor(stretch * 1000.0) / 1000.0)
        calibration["executed_asinh_stretch"] = float(execution_stretch)
        offset = _clamp_float(
            float(calibration["offset"]),
            0.00001,
            0.0060,
        )
        messages.append(
            "Stage9 adaptive starmask calibration "
            f"samples={int(calibration.get('star_sample_count', 0))}, "
            f"components={int(calibration.get('compact_component_count', 0))}, "
            f"faint={float(calibration.get('faint_value', 0.0)):.5f}, "
            f"peak={float(calibration.get('peak_value', 0.0)):.5f}, "
            "predicted_change="
            f"{float(calibration.get('predicted_change_ratio', 0.0)):.3f}/"
            f"{float(calibration.get('predicted_change_ratio_limit', 0.0)):.3f}"
        )

        multi_anchor_curve = bool(
            calibration.get("status") == "ok"
            and calibration.get("multi_anchor_curve", False)
        )
        if multi_anchor_curve:
            stretch_semantics: Dict[str, Any] = {
                "schema": stage7_stretch_metrics.STRETCH_SEMANTICS_SCHEMA,
                "status": "available",
                "engine": "numpy",
                "method": "monotonic_multi_anchor_star_curve",
                "minimum_siril_version": (
                    stage7_stretch_metrics.SIRIL_MINIMUM_VERSION_CONTRACT
                ),
                "bundled_reference_version": (
                    stage7_stretch_metrics.SIRIL_BUNDLED_REFERENCE_VERSION
                ),
                "luminance_mode": "linked_rgb_curve",
                "human_weighted": False,
                "clip_mode": "bounded_monotonic_curve",
                "steps": [
                    {
                        "command": "numpy_monotonic_multi_anchor_star_curve",
                        "argv": [],
                        "full_argv": [
                            "numpy_monotonic_multi_anchor_star_curve"
                        ],
                    }
                ],
            }
        else:
            stretch_semantics = (
                stage7_stretch_metrics.build_siril_stretch_semantics(
                    "asinh",
                    {
                        "asinh_stretch": f"{execution_stretch:.3f}",
                        "asinh_offset": f"{offset:.5f}",
                    },
                )
            )
        stretch_semantics["scope"] = "stage9_star_layer"
        calibration["stretch_semantics"] = stretch_semantics

        if calibration.get("status") == "ok" and not stretch_enabled:
            calibration["starmask_stretch_disabled"] = True
            calibration["stretch_applied"] = False
            pipeline.cmd_with_check("save", stretched_name)
            pipeline._stage9_starmask_calibration = {
                key: value
                for key, value in calibration.items()
                if not str(key).startswith("_")
            }
            messages.append(
                "Stage9 starmask stretch disabled; retained validated compact support"
            )
            if pipeline.process_dir:
                pipeline._stage9_stretched_starmask_file = (
                    pipeline.process_dir / f"{stretched_name}.fit"
                )
            return stretched_name

        stretch_execution_started = True

        def validate_runtime_output(pixels, *, source: str) -> None:
            if not calibration.get("output_profile"):
                return
            runtime_profile = stage9_quality.measure_starmask_output_profile(
                pixels,
                calibration,
                source=source,
            )
            calibration["runtime_output_profile"] = runtime_profile
            light_contract = dict(calibration.get("light_stretch_contract") or {})
            light_contract["runtime_output_profile"] = runtime_profile
            calibration["light_stretch_contract"] = light_contract
            if not bool(runtime_profile.get("accepted", False)):
                raise RuntimeError(
                    str(
                        runtime_profile.get("reason")
                        or "stage9_starmask_output_profile_unavailable"
                    )
                )

        if multi_anchor_curve:
            curved_pixels = stage9_quality.apply_calibrated_starmask(
                stretch_input_data,
                calibration,
            )
            validate_runtime_output(
                curved_pixels,
                source="actual_builtin_multi_anchor_pixels",
            )
            if not write_pixels(
                curved_pixels,
                label="Stage9 monotonic multi-anchor starmask",
            ):
                raise RuntimeError("multi-anchor starmask pixel write unavailable")
            calibration["multi_anchor_curve_applied"] = True
            calibration["stretch_applied"] = True
            stretch_method = "monotonic_multi_anchor_star_curve"
        else:
            pipeline.cmd_with_check(
                "asinh",
                f"{execution_stretch:.3f}",
                f"{offset:.5f}",
                "-clipmode=rgbblend",
            )
            if calibration.get("output_profile"):
                if not callable(get_pixels):
                    raise RuntimeError(
                        "stage9_starmask_output_profile_unavailable: "
                        "Siril pixel reader unavailable after Asinh"
                    )
                stretched_pixels = get_pixels(preview=False)
                if stretched_pixels is None:
                    raise RuntimeError(
                        "stage9_starmask_output_profile_unavailable: "
                        "Siril Asinh pixels unavailable"
                    )
                validate_runtime_output(
                    stretched_pixels,
                    source="actual_siril_asinh_pixels",
                )
            calibration["stretch_applied"] = True
            stretch_method = "asinh"
        pipeline._stage9_starmask_calibration = {
            key: value
            for key, value in calibration.items()
            if not str(key).startswith("_")
        }
        messages.append(
            "Stage9 starmask stretched before pixel remix "
            f"(method={stretch_method}, stretch={execution_stretch:.3f}, "
            f"offset={offset:.5f})"
        )
        pipeline.cmd_with_check("save", stretched_name)
        if pipeline.process_dir:
            pipeline._stage9_stretched_starmask_file = (
                pipeline.process_dir / f"{stretched_name}.fit"
            )
        return stretched_name
    except (
        CommandError,
        SirilError,
        RuntimeError,
        AttributeError,
        TypeError,
        ValueError,
        IndexError,
    ) as e:
        pipeline._stage9_starmask_calibration = {
            "status": "failed",
            "reason": str(e),
            "strict_support": bool(strict_support),
            "compact_support_enabled": bool(
                getattr(
                    pipeline.cfg,
                    "stage9_compact_starmask_enabled",
                    True,
                )
            ),
            "pre_stretch_compact_enabled": bool(
                allow_pre_stretch_compact
                and getattr(
                    pipeline.cfg,
                    "stage9_starmask_pre_stretch_compact_enabled",
                    False,
                )
            ),
            "compact_layer_applied": False,
            "stretch_semantics": calibration.get(
                "stretch_semantics",
                not_run_semantics,
            ),
            "failure_phase": (
                "stretch_execution"
                if stretch_execution_started
                else "starmask_preparation"
            ),
        }
        primary_or_degraded_attempt = bool(
            not strict_support
            or getattr(pipeline, "_stage9_star_reference_degraded", False)
        )
        if (
            not candidate_local
            and stretch_execution_started
            and primary_or_degraded_attempt
        ):
            pipeline._stage9_starmask_stretch_failed = True
            pipeline.log.warn(f"Stage9 starmask stretch execution failed: {e}")
            messages.append(f"Stage9 starmask stretch execution failed: {e}")
        else:
            if not candidate_local and primary_or_degraded_attempt:
                pipeline._stage9_starmask_preparation_failed = True
                pipeline._stage9_starmask_preparation_failure_reason = str(e)
            pipeline.log.warn(f"Stage9 starmask preparation failed: {e}")
            messages.append(f"Stage9 starmask preparation failed: {e}")
        return starmask_name


def _stage9_unscreen_unavailable(
    reason: str,
    *,
    reason_code: str = "stage9_unscreen_reference_unavailable",
) -> Dict[str, Any]:
    return {
        "schema": "starun.stage9-unscreen-reference.v1",
        "status": "unavailable",
        "available": False,
        "reason_code": reason_code,
        "reason": str(reason),
    }


def _stage9_unscreen_context_report(context: Dict[str, Any]) -> Dict[str, Any]:
    nested = context.get("report")
    if isinstance(nested, dict):
        return dict(nested)
    if context.get("schema") == "starun.stage9-unscreen-reference.v1":
        return dict(context)
    return _stage9_unscreen_unavailable("preparation returned no report")


def _stage9_resolve_matched_domain_transfer(pipeline) -> Dict[str, Any]:
    """Resolve the selected Stage 7 transfer from runtime state or its report."""

    transfer = getattr(pipeline, "_stage7_matched_domain_transfer", None)
    reference = getattr(pipeline, "_stage7_closed_form_mtf_reference", None)
    selected_candidate_id = str(
        (transfer or {}).get("selected_candidate_id") or ""
    )
    transfer_source = "runtime"
    reference_source = "runtime"
    process_dir = getattr(pipeline, "process_dir", None)
    report_path = (
        process_dir / "stage7_stretch_quality.json"
        if process_dir is not None
        else None
    )
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(transfer, dict):
            transfer = payload.get("matched_domain_transfer")
            if isinstance(transfer, dict):
                transfer_source = "stage7_stretch_quality.json"
        if not isinstance(reference, dict):
            reference = payload.get("closed_form_mtf_reference")
            if isinstance(reference, dict):
                reference_source = "stage7_stretch_quality.json"
        selected = payload.get("selected") or {}
        selected_candidate_id = str(
            selected_candidate_id or selected.get("name") or ""
        )
    except (AttributeError, json.JSONDecodeError, OSError, TypeError):
        pass

    if isinstance(transfer, dict) and transfer.get("status") == "active":
        method = str(transfer.get("method") or "")
        selected_candidate_id = str(
            transfer.get("selected_candidate_id")
            or selected_candidate_id
            or ""
        )
        transfer_schema = str(transfer.get("schema") or "")
        if transfer_schema not in {
            stage7_stretch_metrics.STAGE7_MATCHED_DOMAIN_TRANSFER_SCHEMA_V1,
            stage7_stretch_metrics.STAGE7_MATCHED_DOMAIN_TRANSFER_SCHEMA_V2,
        }:
            return {
                "status": "unavailable",
                "reason_code": (
                    "stage9_display90_transfer_invalid"
                    if method in stage7_stretch_metrics.DISPLAY_LUT_METHODS
                    or selected_candidate_id.startswith("cand_display")
                    else "stage9_matched_domain_transfer_invalid"
                ),
                "reason": "Stage7 matched-domain transfer schema is invalid",
                "selected_candidate_id": selected_candidate_id,
            }
        if method in stage7_stretch_metrics.DISPLAY_LUT_METHODS:
            expected_method = (
                stage7_stretch_metrics.DISPLAY90_LEGACY_METHOD
                if transfer_schema
                == stage7_stretch_metrics.STAGE7_MATCHED_DOMAIN_TRANSFER_SCHEMA_V1
                else stage7_stretch_metrics.DISPLAY_LUMINANCE_VECTOR_METHOD
            )
            if method != expected_method:
                return {
                    "status": "unavailable",
                    "reason_code": "stage9_display90_transfer_invalid",
                    "reason": (
                        "Stage7 matched-domain schema/method semantics do not match"
                    ),
                    "selected_candidate_id": selected_candidate_id,
                }
            tone_candidate_id = str(
                transfer.get("tone_candidate_id")
                or selected_candidate_id
                or ""
            )
            if not tone_candidate_id.startswith("cand_display"):
                return {
                    "status": "unavailable",
                    "reason_code": "stage9_display90_transfer_invalid",
                    "reason": "Display90 transfer does not match the selected candidate",
                    "selected_candidate_id": selected_candidate_id,
                }
            calibration = dict(transfer.get("calibration") or {})
            try:
                _lut, lut_contract = (
                    stage7_stretch_metrics.rebuild_display90_linked_lut(
                        calibration
                    )
                )
            except (KeyError, TypeError, ValueError, FloatingPointError) as error:
                return {
                    "status": "unavailable",
                    "reason_code": "stage9_display90_transfer_invalid",
                    "reason": f"Display90 LUT authentication failed: {error}",
                    "selected_candidate_id": selected_candidate_id,
                }
            if (
                dict(transfer.get("lut_contract") or {}) != lut_contract
                or transfer.get("fallback_to_linked_mtf_allowed") is not False
            ):
                return {
                    "status": "unavailable",
                    "reason_code": "stage9_display90_transfer_invalid",
                    "reason": (
                        "Display90 matched-domain LUT summary contract is invalid"
                    ),
                    "selected_candidate_id": selected_candidate_id,
                }
            return {
                "status": "ready",
                "method": method,
                "source": transfer_source,
                "selected_candidate_id": selected_candidate_id,
                "tone_candidate_id": tone_candidate_id,
                "calibration": calibration,
                "lut_contract": lut_contract,
                "fallback_to_linked_mtf_allowed": False,
            }
        if method == "closed_form_linked_mtf":
            params = dict(transfer.get("params") or {})
            reference_source = str(
                transfer.get("source") or transfer_source
            )
        else:
            return {
                "status": "unavailable",
                "reason_code": "stage9_matched_domain_transfer_invalid",
                "reason": f"unsupported Stage7 matched-domain method: {method}",
                "selected_candidate_id": selected_candidate_id,
            }
    else:
        if selected_candidate_id.startswith("cand_display"):
            return {
                "status": "unavailable",
                "reason_code": "stage9_display90_transfer_invalid",
                "reason": (
                    "selected Display90 candidate has no authenticated matched-domain "
                    "transfer; linked-MTF fallback is forbidden"
                ),
                "selected_candidate_id": selected_candidate_id,
            }
        if not isinstance(reference, dict) or reference.get("status") != "active":
            return {
                "status": "unavailable",
                "reason_code": "stage9_mtf_reference_unavailable",
                "reason": "Stage7 closed-form linked-MTF anchor is unavailable",
                "selected_candidate_id": selected_candidate_id or None,
            }
        active_anchor = reference.get("active_anchor") or {}
        params = dict(active_anchor.get("params") or {})

    try:
        shadows = float(params["mtf_shadows"])
        midtones = float(params["mtf_midtones"])
        highlights = float(params.get("mtf_highlights", 1.0))
    except (KeyError, TypeError, ValueError):
        return {
            "status": "unavailable",
            "reason_code": "stage9_mtf_reference_unavailable",
            "reason": "Stage7 linked-MTF anchor parameters are incomplete",
            "selected_candidate_id": selected_candidate_id or None,
        }
    if not (
        0.0 <= shadows < highlights <= 1.0
        and 0.0 < midtones < 1.0
    ):
        return {
            "status": "unavailable",
            "reason_code": "stage9_mtf_reference_unavailable",
            "reason": "Stage7 linked-MTF anchor parameters are invalid",
            "selected_candidate_id": selected_candidate_id or None,
        }
    return {
        "status": "ready",
        "method": "closed_form_linked_mtf",
        "source": reference_source,
        "selected_candidate_id": selected_candidate_id or None,
        "params": {
            "mtf_shadows": shadows,
            "mtf_midtones": midtones,
            "mtf_highlights": highlights,
        },
        "fallback_to_linked_mtf_allowed": True,
    }


def _stage9_apply_matched_domain_transfer(
    image: np.ndarray,
    transfer: Dict[str, Any],
) -> np.ndarray:
    """Apply the already-authenticated Stage7 transfer to one Stage6 role."""

    method = str(transfer.get("method") or "")
    if transfer.get("status") != "ready":
        raise ValueError("Stage7 matched-domain transfer is not ready")
    if method in stage7_stretch_metrics.DISPLAY_LUT_METHODS:
        return stage7_stretch_metrics.apply_display90_linked_rgb_stretch(
            image,
            dict(transfer.get("calibration") or {}),
        )
    if method == "closed_form_linked_mtf":
        params = dict(transfer.get("params") or {})
        return stage7_stretch_metrics.apply_linked_mtf(
            image,
            float(params["mtf_shadows"]),
            float(params["mtf_midtones"]),
            float(params.get("mtf_highlights", 1.0)),
        )
    raise ValueError(f"unsupported matched-domain transfer: {method}")


def _prepare_stage9_matched_domain_context(
    pipeline,
    messages: List[str],
) -> Dict[str, Any]:
    """Verify and load the immutable Stage6 pair in the selected Stage7 domain."""
    cached = getattr(pipeline, "_stage9_matched_domain_context", None)
    if isinstance(cached, dict):
        return cached

    def unavailable(reason: str, reason_code: str) -> Dict[str, Any]:
        context = {
            "available": False,
            "report": {
                "schema": "starun.stage9-matched-domain.v1",
                "status": "unavailable",
                "available": False,
                "reason_code": reason_code,
                "reason": str(reason),
            },
        }
        pipeline._stage9_matched_domain_context = context
        return context

    if bool(getattr(pipeline, "_star_preserve_target_bypass", False)):
        return unavailable(
            "target-bypass keeps the with-stars source",
            "stage9_stage6_pair_handoff_unavailable",
        )
    if str(getattr(pipeline, "_star_separation_state", "") or "") != (
        StarSeparationState.ACCEPTED.value
    ):
        return unavailable(
            "Stage6 star separation was not accepted",
            "stage9_stage6_pair_handoff_unavailable",
        )
    if bool(getattr(pipeline, "_stage6_quality_hard_failed_retained", False)):
        return unavailable(
            "Stage6 hard-failed pair was retained",
            "stage9_stage6_pair_mismatch",
        )
    if not bool(getattr(pipeline, "_stage7_stretch_accepted", False)):
        return unavailable(
            "Stage7 stretch was not accepted",
            "stage9_stage6_pair_handoff_unavailable",
        )

    pair_verification = syqon_starless.verify_stage6_pair_handoff(pipeline)
    if pair_verification.get("accepted") is not True:
        return unavailable(
            str(
                pair_verification.get("reason")
                or pair_verification.get("status")
                or "Stage6 pair verification failed"
            ),
            str(
                pair_verification.get("reason_code")
                or "stage9_stage6_pair_mismatch"
            ),
        )

    matched_transfer = _stage9_resolve_matched_domain_transfer(pipeline)
    if matched_transfer.get("status") != "ready":
        return unavailable(
            str(
                matched_transfer.get("reason")
                or "Stage7 matched-domain transfer is unavailable"
            ),
            str(
                matched_transfer.get("reason_code")
                or "stage9_matched_domain_transfer_invalid"
            ),
        )

    get_pixels = getattr(pipeline.siril, "get_image_pixeldata", None)
    if not callable(get_pixels):
        return unavailable(
            "Siril pixel reader is unavailable",
            "stage9_stage6_pair_handoff_unavailable",
        )
    try:
        pair_pixels: Dict[str, np.ndarray] = {}
        pair_domains: Dict[str, Any] = {}
        pair_paths = pair_verification.get("paths") or {}
        for role in ("stage6_input", "stage6_starless"):
            path = Path(str(pair_paths.get(role) or ""))
            if not path.is_file():
                raise RuntimeError(f"verified Stage6 role is unavailable: {role}")
            pipeline.cmd_with_check("load", path.stem)
            pixels = get_pixels(preview=False)
            if pixels is None:
                raise RuntimeError(f"{role} image buffer is empty")
            pair_pixels[role], pair_domains[role] = (
                canonicalize_stage7_pixels_01(pixels)
            )
        original_linear = pair_pixels["stage6_input"]
        starless_linear = pair_pixels["stage6_starless"]
        if original_linear.shape != starless_linear.shape:
            raise RuntimeError(
                "verified Stage6 pair shape mismatch: "
                f"{original_linear.shape}!={starless_linear.shape}"
            )
        transfer_method = str(matched_transfer.get("method") or "")
        original_display = _stage9_apply_matched_domain_transfer(
            original_linear,
            matched_transfer,
        )
        starless_display = _stage9_apply_matched_domain_transfer(
            starless_linear,
            matched_transfer,
        )
        linear_roundtrip = stage9_quality.assess_linear_decomposition_roundtrip(
            original_linear,
            starless_linear,
        )
        source_autostretch_display = None
        source_autostretch_report: Dict[str, Any] = {
            "schema": "starun.stage9-source-autostretch-wing-reference.v1",
            "status": "disabled",
            "available": False,
            "reason_code": "stage9_source_autostretch_wing_reference_disabled",
            "purpose": "display_visible_low_intensity_psf_wing_shape_only",
            "scientific_photometry_claim": False,
            "changes_stage7_anchor": False,
            "changes_fwhm_hard_gate": False,
        }
        if bool(
            getattr(
                pipeline.cfg,
                "stage9_source_autostretch_wing_reference_enabled",
                True,
            )
        ):
            source_autostretch_report.update(
                status="unavailable",
                reason_code="stage9_source_autostretch_wing_reference_unavailable",
                reason="no shape-compatible same-source linear image was available",
            )
            reference_candidates: List[tuple[str, Path, bool]] = []
            signed_source, signed_source_detail = (
                _stage9_signed_task_master_source(pipeline)
            )
            if signed_source is not None:
                reference_candidates.append(
                    ("signed_original_task_source", signed_source, True)
                )
            else:
                source_autostretch_report["signed_source_detail"] = (
                    signed_source_detail
                )
            source_file = getattr(pipeline, "source_file", None)
            if source_file is not None:
                source_path = Path(source_file)
                if source_path.is_file() and all(
                    path.resolve() != source_path.resolve()
                    for _role, path, _external in reference_candidates
                ):
                    reference_candidates.append(
                        ("original_task_source", source_path, True)
                    )
            stage6_input_path = Path(
                str((pair_verification.get("paths") or {}).get("stage6_input") or "")
            )
            if stage6_input_path.is_file() and all(
                path.resolve() != stage6_input_path.resolve()
                for _role, path, _external in reference_candidates
            ):
                reference_candidates.append(
                    ("verified_stage6_input", stage6_input_path, False)
                )

            failure_reasons: List[str] = []
            for role, path, external_path in reference_candidates:
                try:
                    process_dir = getattr(pipeline, "process_dir", None)
                    if external_path:
                        pipeline.cmd_with_check("cd", f'"{path.parent}"')
                        pipeline.cmd_with_check("load", f'"{path.name}"')
                        if process_dir is not None:
                            pipeline.cmd_with_check("cd", f'"{process_dir}"')
                    else:
                        if process_dir is not None:
                            pipeline.cmd_with_check("cd", f'"{process_dir}"')
                        pipeline.cmd_with_check("load", path.stem)
                    reference_linear_pixels = get_pixels(preview=False)
                    if reference_linear_pixels is None:
                        raise RuntimeError("image buffer is empty")
                    reference_linear, reference_domain = (
                        canonicalize_stage7_pixels_01(reference_linear_pixels)
                    )
                    if reference_linear.shape != original_linear.shape:
                        raise RuntimeError(
                            "shape mismatch: "
                            f"{reference_linear.shape}!={original_linear.shape}"
                        )
                    pipeline.cmd_with_check("autostretch", "-linked")
                    stretched_pixels = get_pixels(preview=False)
                    if stretched_pixels is None:
                        raise RuntimeError("linked autostretch buffer is empty")
                    stretched, stretched_domain = canonicalize_stage7_pixels_01(
                        stretched_pixels
                    )
                    if stretched.shape != original_display.shape:
                        raise RuntimeError(
                            "stretched shape mismatch: "
                            f"{stretched.shape}!={original_display.shape}"
                        )
                    source_autostretch_display = np.asarray(
                        stretched,
                        dtype=np.float32,
                    )
                    source_autostretch_report.update(
                        status="ready",
                        available=True,
                        reason_code="stage9_source_autostretch_wing_reference_ready",
                        source_role=role,
                        source_path=str(path),
                        method="siril_autostretch_linked",
                        command="autostretch -linked",
                        default_shadow_sigma=-2.8,
                        default_target_background=0.25,
                        source_domain=reference_domain,
                        display_domain=stretched_domain,
                    )
                    source_autostretch_report.pop("reason", None)
                    break
                except (
                    AttributeError,
                    CommandError,
                    OSError,
                    RuntimeError,
                    SirilError,
                    TypeError,
                    ValueError,
                ) as reference_error:
                    failure_reasons.append(f"{role}: {reference_error}")
                    process_dir = getattr(pipeline, "process_dir", None)
                    if process_dir is not None:
                        try:
                            pipeline.cmd_with_check("cd", f'"{process_dir}"')
                        except (CommandError, SirilError, OSError, RuntimeError):
                            pass
            if source_autostretch_display is None and failure_reasons:
                source_autostretch_report["reason"] = "; ".join(failure_reasons)

            # Leave the immutable Stage6 backdrop active for the next caller;
            # the autostretch reference only exists in memory and is never
            # allowed to become the Stage7 or final rendering source.
            stage6_starless_path = Path(
                str(
                    (pair_verification.get("paths") or {}).get(
                        "stage6_starless"
                    )
                    or ""
                )
            )
            if stage6_starless_path.is_file():
                pipeline.cmd_with_check(
                    "load",
                    stage6_starless_path.stem,
                )
        context = {
            "available": True,
            "original_display": original_display,
            "starless_display": starless_display,
            "pair_domains": pair_domains,
            "report": {
                "schema": "starun.stage9-matched-domain.v1",
                "status": "ready",
                "available": True,
                "reason_code": "stage9_stage6_pair_verified",
                "pair_handoff": pair_verification,
                "pair_sources": ["stage6_input", "stage6_starless"],
                "pair_domains": pair_domains,
                "linear_decomposition_roundtrip": linear_roundtrip,
                "source_autostretch_wing_reference": source_autostretch_report,
                "matched_domain_transfer": {
                    key: value
                    for key, value in matched_transfer.items()
                    if key != "calibration"
                },
                "mtf_reference": (
                    {
                        "source": matched_transfer.get("source"),
                        "method": transfer_method,
                        "params": dict(matched_transfer.get("params") or {}),
                    }
                    if transfer_method == "closed_form_linked_mtf"
                    else {
                        "status": "not_applicable",
                        "method": transfer_method,
                    }
                ),
            },
        }
        if source_autostretch_display is not None:
            context["source_autostretch_display"] = source_autostretch_display
        pipeline._stage9_matched_domain_context = context
        messages.append(
            "Stage9 verified immutable Stage6 pair and restored the selected "
            f"Stage7 matched domain ({transfer_method})"
        )
        if source_autostretch_display is not None:
            messages.append(
                "Stage9 captured same-source linked-autostretch reference for "
                "display-visible low-intensity PSF wings only"
            )
        return context
    except (
        AttributeError,
        CommandError,
        OSError,
        RuntimeError,
        SirilError,
        TypeError,
        ValueError,
    ) as error:
        return unavailable(str(error), "stage9_stage6_pair_mismatch")


def _prepare_stage9_unscreen_candidate(
    pipeline,
    trusted_starmask_name: str,
    messages: List[str],
    *,
    output_name: str = "starmask_unscreen_stabilized",
    support_mode: str = "unknown",
) -> Dict[str, Any]:
    """Prepare the matched-domain, chroma-stable Unscreen star layer."""
    if not bool(
        getattr(pipeline.cfg, "stage9_unscreen_candidate_enabled", True)
    ):
        return _stage9_unscreen_unavailable(
            "disabled by configuration",
            reason_code="stage9_unscreen_candidate_disabled",
        )
    if bool(getattr(pipeline, "_star_preserve_target_bypass", False)):
        return _stage9_unscreen_unavailable("target-bypass keeps the with-stars source")
    if str(getattr(pipeline, "_star_separation_state", "") or "") != (
        StarSeparationState.ACCEPTED.value
    ):
        return _stage9_unscreen_unavailable("Stage6 star separation was not accepted")
    if bool(getattr(pipeline, "_stage6_quality_hard_failed_retained", False)):
        return _stage9_unscreen_unavailable("Stage6 hard-failed pair was retained")
    if not bool(getattr(pipeline, "_stage7_stretch_accepted", False)):
        return _stage9_unscreen_unavailable("Stage7 stretch was not accepted")
    catalog = getattr(pipeline, "_stage9_star_reference_catalog", None)
    if (
        not isinstance(catalog, dict)
        or catalog.get("status") != "ok"
        or not bool(catalog.get("source_matched", False))
        or bool(getattr(pipeline, "_stage9_star_reference_degraded", False))
    ):
        return _stage9_unscreen_unavailable(
            "independent source star reference is unavailable or degraded"
        )

    matched_context = _prepare_stage9_matched_domain_context(
        pipeline,
        messages,
    )
    matched_report = matched_context.get("report") or {}
    if matched_context.get("available") is not True:
        report = _stage9_unscreen_unavailable(
            str(
                matched_report.get("reason")
                or "immutable Stage6 matched-domain reference is unavailable"
            ),
            reason_code=str(
                matched_report.get("reason_code")
                or "stage9_unscreen_reference_unavailable"
            ),
        )
        report["matched_domain"] = matched_report
        return {"available": False, "report": report}

    pair_verification = matched_report.get("pair_handoff") or {}
    get_pixels = getattr(pipeline.siril, "get_image_pixeldata", None)
    if not callable(get_pixels):
        return _stage9_unscreen_unavailable("Siril pixel reader is unavailable")

    try:
        trusted_pixels = getattr(pipeline, "_stage9_last_star_layer", None)
        if trusted_pixels is None:
            pipeline.cmd_with_check("load", trusted_starmask_name)
            trusted_pixels = get_pixels(preview=False)
        if trusted_pixels is None:
            raise RuntimeError("trusted stretched star layer is unavailable")
        trusted, trusted_domain = canonicalize_stage7_pixels_01(trusted_pixels)
        original_display = np.asarray(
            matched_context["original_display"],
            dtype=np.float32,
        )
        starless_display = np.asarray(
            matched_context["starless_display"],
            dtype=np.float32,
        )
        if trusted.shape != original_display.shape:
            matcher = getattr(pipeline, "_match_star_layer_shape", None)
            if not callable(matcher):
                raise RuntimeError("trusted star-layer shape matcher unavailable")
            trusted, trusted_domain = canonicalize_stage7_pixels_01(
                matcher(trusted, original_display)
            )
        support_source = getattr(
            pipeline,
            "_stage9_last_star_overlay_mask",
            None,
        )
        if support_source is None:
            raise RuntimeError("current compact star overlay support is unavailable")
        support = np.asarray(support_source, dtype=np.float32) > 1e-6
        stabilized, public_report = (
            stage9_quality.build_chroma_stable_unscreen_layer(
                original_display,
                starless_display,
                trusted,
                support,
                pipeline.cfg,
            )
        )
        public_report.update(
            {
                "pair_verification": pair_verification,
                "pair_sources": matched_report.get("pair_sources") or [],
                "pair_domains": matched_report.get("pair_domains") or {},
                "trusted_star_domain": trusted_domain,
                "matched_domain": matched_report,
                "mtf_reference": matched_report.get("mtf_reference") or {},
            }
        )
        if stabilized is None:
            messages.append(
                "Stage9 matched-domain Unscreen unavailable: "
                f"{public_report.get('reason') or public_report.get('reason_code')}"
            )
            return {"available": False, "report": public_report}

        pipeline.cmd_with_check("load", trusted_starmask_name)
        writer = getattr(pipeline, "_set_current_image_pixeldata", None)
        if callable(writer):
            writer(stabilized, label="Stage9 chroma-stable Unscreen star layer")
        else:
            set_pixels = getattr(pipeline.siril, "set_image_pixeldata", None)
            if not callable(set_pixels):
                raise RuntimeError("Siril pixel writer is unavailable")
            lock_factory = getattr(pipeline.siril, "image_lock", None)
            if callable(lock_factory):
                with lock_factory():
                    set_pixels(stabilized)
            else:
                set_pixels(stabilized)
        if not pipeline._save_stage_output(output_name):
            raise RuntimeError("stabilized Unscreen star-layer save failed")
        weak_mask = getattr(pipeline, "_stage9_last_weak_overlay_mask", None)
        bright_mask = getattr(pipeline, "_stage9_last_bright_overlay_mask", None)
        candidate_masks = dict(
            getattr(pipeline, "_stage9_candidate_overlay_masks", {}) or {}
        )
        candidate_masks[output_name] = {
            "weak": None if weak_mask is None else np.array(weak_mask, copy=True),
            "bright": (
                None if bright_mask is None else np.array(bright_mask, copy=True)
            ),
            "alpha": np.array(support, copy=True),
        }
        pipeline._stage9_candidate_overlay_masks = candidate_masks
        public_report.update(
            support_mode=support_mode,
            support_starmask=trusted_starmask_name,
            output_starmask=output_name,
        )
        messages.append(
            "Stage9 matched-domain Unscreen reference ready "
            f"(reliable_support={float(public_report.get('reliable_support_ratio', 0.0)):.3f}, "
            f"fallback_support={float(public_report.get('fallback_support_ratio', 0.0)):.3f})"
        )
        return {
            "available": True,
            "report": public_report,
            "original_display": original_display,
            "starless_display": starless_display,
            **(
                {
                    "source_autostretch_display": matched_context[
                        "source_autostretch_display"
                    ]
                }
                if matched_context.get("source_autostretch_display") is not None
                else {}
            ),
            "trusted_stars": trusted,
            "unscreen_stars": stabilized,
            "support_mask": support,
            "weak_mask": weak_mask,
            "bright_mask": bright_mask,
            "starmask": output_name,
            "support_mode": support_mode,
            "support_starmask": trusted_starmask_name,
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
        report = _stage9_unscreen_unavailable(str(error))
        report.update(
            pair_verification=pair_verification,
            matched_domain=matched_report,
            mtf_reference=matched_report.get("mtf_reference") or {},
        )
        messages.append(f"Stage9 matched-domain Unscreen unavailable: {error}")
        return {"available": False, "report": report}


def _stage9_stage5_star_reference_report(pipeline) -> Dict[str, Any]:
    """Load the frozen Stage5 Siril star reference from memory or its report."""
    runtime_report = getattr(pipeline, "_stage5_star_reference_report", None)
    if isinstance(runtime_report, dict) and runtime_report.get("stars"):
        return runtime_report
    resume_context = getattr(pipeline, "_resume_semantic_context", None)
    if isinstance(resume_context, dict):
        resumed_report = resume_context.get("stage5_star_reference_report")
        if isinstance(resumed_report, dict) and resumed_report.get("stars"):
            return resumed_report
    process_dir = getattr(pipeline, "process_dir", None)
    report_path = (
        Path(process_dir) / "stage5_linear_report.json"
        if process_dir is not None
        else None
    )
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        report = ((payload.get("deconvolution") or {}).get("star_reference") or {})
        return report if isinstance(report, dict) else {}
    except (AttributeError, json.JSONDecodeError, OSError, TypeError):
        return {}


def _prepare_stage9_source_presence_candidate(
    pipeline,
    context: Dict[str, Any],
    messages: List[str],
    *,
    feather_strength: float = 0.90,
    screen_intensity: float = 1.0,
) -> Dict[str, Any]:
    """Add bounded source wings and independently frozen bright-star support."""
    if not bool(context.get("available", False)):
        return {
            **context,
            "source_presence_report": {
                "schema": "starun.stage9-source-presence.v1",
                "status": "unavailable",
                "available": False,
                "changed": False,
                "reason_code": "stage9_source_presence_context_unavailable",
                "reason": "accepted Unscreen context is unavailable",
            },
        }
    catalog = getattr(pipeline, "_stage9_star_reference_catalog", None)
    if not isinstance(catalog, dict) or catalog.get("status") != "ok":
        return {
            **context,
            "source_presence_report": {
                "schema": "starun.stage9-source-presence.v1",
                "status": "unavailable",
                "available": False,
                "changed": False,
                "reason_code": "stage9_source_presence_catalog_unavailable",
                "reason": "frozen Stage9 star reference catalog is unavailable",
            },
        }
    baseline_stars = context.get("unscreen_stars")
    baseline_support = context.get("support_mask")
    trusted = context.get("trusted_stars")
    if baseline_stars is None or baseline_support is None:
        return {
            **context,
            "source_presence_report": {
                "schema": "starun.stage9-source-presence.v1",
                "status": "unavailable",
                "available": False,
                "changed": False,
                "reason_code": "stage9_source_presence_arrays_unavailable",
                "reason": "accepted Unscreen stars or support mask is unavailable",
            },
        }

    candidate = np.asarray(baseline_stars, dtype=np.float32)
    support = np.asarray(baseline_support, dtype=bool)
    weak_mask = np.asarray(
        getattr(pipeline, "_stage9_last_weak_overlay_mask", support),
        dtype=bool,
    )
    bright_mask = np.asarray(
        getattr(pipeline, "_stage9_last_bright_overlay_mask", support),
        dtype=bool,
    )
    presence_report: Dict[str, Any] = {
        "schema": "starun.stage9-source-presence.v1",
        "status": "ready",
        "available": True,
        "source_wing_feather": {
            "status": "not_run",
            "reason": "ordinary source support not available",
        },
        "independent_source_presence": {
            "status": "not_run",
            "available": False,
            "changed": False,
            "reason": "independent same-source recovery is disabled",
        },
        "stage5_bright_star_completion": {
            "schema": "starun.stage9-stage5-bright-star-completion.v2",
            "status": "not_run",
            "available": False,
            "reason_code": "stage9_stage5_bright_star_completion_not_run",
            "reason": "disabled or frozen Stage5 reference unavailable",
            "coordinate_contract": {
                "schema": "starun.pixel-coordinate-contract.v1",
                "source_coordinate_domain": "siril_star_catalog_bottom_up",
                "array_coordinate_domain": "siril_pixel_buffer_bottom_up",
                "conversion": "y_array = y_siril",
                "validated": True,
            },
        },
    }

    try:
        normal_weak, normal_bright, normal_support = (
            stage9_quality.build_star_overlay_masks(
                catalog,
                strict=False,
                cfg=pipeline.cfg,
            )
        )
        feathered, feather_support, feather_report = (
            stage9_quality.build_source_wing_feather_candidate(
                context["original_display"],
                context["starless_display"],
                candidate,
                support,
                normal_support,
                pipeline.cfg,
                feather_strength=feather_strength,
            )
        )
        presence_report["source_wing_feather"] = feather_report
        if feathered is not None and feather_support is not None:
            added_support = np.asarray(feather_support, dtype=bool) & ~support
            candidate = feathered
            support = np.asarray(feather_support, dtype=bool)
            weak_mask |= np.asarray(normal_weak, dtype=bool) & added_support
            bright_mask |= np.asarray(normal_bright, dtype=bool) & added_support
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        presence_report["source_wing_feather"] = {
            "schema": "starun.stage9-source-wing-feather.v1",
            "status": "unavailable",
            "available": False,
            "reason": str(error),
        }

    if bool(
        getattr(
            pipeline.cfg,
            "stage9_independent_source_presence_enabled",
            True,
        )
    ):
        independent, independent_support, independent_report = (
            stage9_quality.build_independent_source_presence_candidate(
                context["original_display"],
                context["starless_display"],
                candidate,
                support,
                pipeline.cfg,
                spatial_scale=getattr(pipeline, "_stage9_spatial_scale", None),
                strength=feather_strength,
            )
        )
        presence_report["independent_source_presence"] = independent_report
        if independent is not None and independent_support is not None:
            added_support = np.asarray(independent_support, dtype=bool) & ~support
            candidate = independent
            support = np.asarray(independent_support, dtype=bool)
            weak_mask |= added_support

    stage5_report = _stage9_stage5_star_reference_report(pipeline)
    completion_enabled = bool(
        getattr(
            pipeline.cfg,
            "stage9_stage5_bright_star_completion_enabled",
            True,
        )
    )
    if completion_enabled and stage5_report.get("stars"):
        completion_evidence = getattr(
            pipeline,
            "_stage9_immutable_trusted_starmask_peak",
            None,
        )
        if completion_evidence is None:
            completion_evidence = trusted
        if completion_evidence is None:
            completion_report = {
                **presence_report["stage5_bright_star_completion"],
                "status": "unavailable",
                "reason_code": (
                    "stage9_stage5_bright_star_completion_evidence_unavailable"
                ),
                "reason": "immutable trusted starmask evidence is unavailable",
            }
        else:
            completion = stage9_quality.build_stage5_bright_star_completion(
                stage5_report.get("stars"),
                catalog,
                context["original_display"],
                completion_evidence,
                pipeline.cfg,
                coordinate_domain="siril_pixel_buffer_bottom_up",
            )
            completed, completion_support, completion_report = (
                stage9_quality.apply_stage5_bright_star_completion(
                    context["original_display"],
                    context["starless_display"],
                    candidate,
                    completion,
                    pipeline.cfg,
                    remix_base=context.get("remix_base"),
                    screen_intensity=screen_intensity,
                )
            )
            if completed is not None and completion_support is not None:
                candidate = completed
                bright_mask |= np.asarray(completion_support, dtype=bool)
                support |= np.asarray(completion_support, dtype=bool)
        presence_report["stage5_bright_star_completion"] = completion_report

    changed = bool(np.any(np.abs(candidate - baseline_stars) > 1e-7))
    presence_report.update(
        changed=changed,
        support_pixel_count=int(np.count_nonzero(support)),
        support_ratio=float(np.mean(support)),
        semantics=(
            "matched_source_psf_wings_plus_independent_same_source_weak_stars_"
            "plus_frozen_stage5_bright_star_completion"
        ),
        ordinary_fwhm_gate_unchanged=True,
    )
    if not changed:
        return {**context, "source_presence_report": presence_report}
    report = dict(context.get("report") or {})
    report["source_presence"] = presence_report
    report["operator_audit"] = stage9_quality.assess_unscreen_operator_roundtrip(
        context["original_display"],
        context["starless_display"],
        candidate,
        support,
        denominator_floor=float(
            getattr(pipeline.cfg, "stage9_unscreen_denominator_floor", 0.08)
        ),
    )
    messages.append(
        "Stage9 prepared source-presence candidate "
        f"(support={float(np.mean(support)):.4f}, "
        "ordinary_fwhm_gate=unchanged)"
    )
    return {
        **context,
        "report": report,
        "unscreen_stars": candidate,
        "support_mask": support,
        "weak_mask": weak_mask,
        "bright_mask": bright_mask,
        "source_presence_report": presence_report,
    }


def _prepare_stage9_selective_size_candidate(
    pipeline,
    context: Dict[str, Any],
    candidate_display: np.ndarray,
    messages: List[str],
    *,
    screen_intensity: float,
    fwhm_ratio_target: float,
    feather_strength: float,
    support_extra_pixels: int,
    target_groups: tuple[str, ...] | None = None,
    recovery_alpha: float = 1.0,
) -> Dict[str, Any]:
    """Apply a source-confirmed outer wing only to still-small ordinary stars."""
    report: Dict[str, Any] = {
        "schema": "starun.stage9-selective-source-wing.v1",
        "status": "unavailable",
        "available": False,
        "changed": False,
        "reason": "selective size candidate prerequisites are unavailable",
    }
    if not bool(context.get("available", False)):
        return {**context, "selective_source_wing_report": report}
    catalog = getattr(pipeline, "_stage9_star_reference_catalog", None)
    stars = context.get("unscreen_stars")
    support_source = context.get("support_mask")
    if (
        not isinstance(catalog, dict)
        or catalog.get("status") != "ok"
        or stars is None
        or support_source is None
    ):
        return {**context, "selective_source_wing_report": report}

    selective, selective_support, report = (
        stage9_quality.build_selective_source_wing_candidate(
            context["original_display"],
            context["starless_display"],
            np.asarray(stars),
            candidate_display,
            np.asarray(support_source, dtype=bool),
            catalog,
            pipeline.cfg,
            remix_base=context.get("remix_base"),
            visible_wing_reference=context.get("source_autostretch_display"),
            screen_intensity=screen_intensity,
            fwhm_ratio_target=fwhm_ratio_target,
            feather_strength=feather_strength,
            extra_pixels=support_extra_pixels,
            target_groups=target_groups,
            recovery_alpha=recovery_alpha,
        )
    )
    if selective is None or selective_support is None:
        return {**context, "selective_source_wing_report": report}

    previous_support = np.asarray(support_source, dtype=bool)
    added_support = np.asarray(selective_support, dtype=bool) & ~previous_support
    weak_mask = np.asarray(
        context.get("weak_mask", previous_support),
        dtype=bool,
    ).copy()
    bright_mask = np.asarray(
        context.get("bright_mask", previous_support),
        dtype=bool,
    ).copy()
    try:
        expanded_weak, expanded_bright, _expanded_support = (
            stage9_quality.build_star_overlay_masks(
                catalog,
                strict=False,
                cfg=pipeline.cfg,
                extra_pixels=support_extra_pixels,
            )
        )
        weak_mask |= np.asarray(expanded_weak, dtype=bool) & added_support
        bright_mask |= np.asarray(expanded_bright, dtype=bool) & added_support
    except (KeyError, RuntimeError, TypeError, ValueError):
        # The alpha support remains authoritative.  Existing weak/bright masks
        # are retained if group-mask reconstruction is unavailable.
        pass

    public_report = dict(context.get("report") or {})
    public_report["selective_source_wing"] = report
    public_report["operator_audit"] = (
        stage9_quality.assess_unscreen_operator_roundtrip(
            context["original_display"],
            context["starless_display"],
            selective,
            selective_support,
            denominator_floor=float(
                getattr(pipeline.cfg, "stage9_unscreen_denominator_floor", 0.08)
            ),
        )
    )
    messages.append(
        "Stage9 prepared same-star selective size candidate "
        f"(selected={int(report.get('selected_star_count', 0) or 0)}, "
        f"mode={str(report.get('selection_mode') or 'unknown')}, "
        f"target={float(report.get('visible_wing_target_ratio', report.get('fwhm_ratio_target', 0.0))):.3f}, "
        f"extra_pixels={support_extra_pixels})"
    )
    return {
        **context,
        "report": public_report,
        "unscreen_stars": selective,
        "support_mask": np.asarray(selective_support, dtype=bool),
        "weak_mask": weak_mask,
        "bright_mask": bright_mask,
        "selective_source_wing_report": report,
    }


def _save_stage9_unscreen_context_layer(pipeline, context: Dict[str, Any]) -> bool:
    """Persist a prepared Unscreen variant using the existing transactional stem."""
    try:
        stars = context.get("unscreen_stars")
        if stars is None:
            return False
        output_name = str(
            context.get("starmask") or "starmask_unscreen_stabilized"
        )
        pipeline.cmd_with_check("load", output_name)
        writer = getattr(pipeline, "_set_current_image_pixeldata", None)
        if callable(writer):
            writer(stars, label="Stage9 source-presence Unscreen star layer")
        else:
            set_pixels = getattr(pipeline.siril, "set_image_pixeldata", None)
            if not callable(set_pixels):
                return False
            lock_factory = getattr(pipeline.siril, "image_lock", None)
            if callable(lock_factory):
                with lock_factory():
                    set_pixels(stars)
            else:
                set_pixels(stars)
        pipeline._stage9_candidate_overlay_masks = {
            **dict(getattr(pipeline, "_stage9_candidate_overlay_masks", {}) or {}),
            output_name: {
                "weak": context.get("weak_mask"),
                "bright": context.get("bright_mask"),
                "alpha": context.get("support_mask"),
            }
        }
        return bool(pipeline._save_stage_output(output_name))
    except (AttributeError, CommandError, RuntimeError, SirilError, ValueError):
        return False


def _stage9_observe_bright_star_presence(
    pipeline,
    *,
    source_stem: str,
    star_layer_name: str,
    intensity: float,
    completion_report: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """Measure bright-star presence and transactionally restore the candidate."""
    unavailable: Dict[str, Any] = {
        "schema": "starun.stage9-stage5-bright-star-presence.v1",
        "status": "unavailable",
        "available": False,
        "gate_role": "presence_and_wing_observation_only",
        "ordinary_fwhm_gate_member": False,
    }
    candidate_pixels = None
    restored = False
    try:
        get_pixels = getattr(pipeline.siril, "get_image_pixeldata", None)
        if not callable(get_pixels):
            raise RuntimeError("Siril pixel reader unavailable")
        current = get_pixels(preview=False)
        if current is None:
            raise RuntimeError("source-presence candidate pixels unavailable")
        candidate_pixels = np.array(current, copy=True)
        pipeline.cmd_with_check("load", source_stem)
        base = get_pixels(preview=False)
        if base is None:
            raise RuntimeError("Stage 9 remix base pixels unavailable")
        return stage9_quality.assess_stage5_bright_star_presence(
            np.asarray(base),
            candidate_pixels,
            completion_report,
        )
    except (
        CommandError,
        SirilError,
        RuntimeError,
        AttributeError,
        TypeError,
        ValueError,
    ) as error:
        return {**unavailable, "reason": str(error)}
    finally:
        if candidate_pixels is not None:
            try:
                pipeline.cmd_with_check("load", source_stem)
                writer = getattr(pipeline, "_set_current_image_pixeldata", None)
                if callable(writer):
                    writer(
                        candidate_pixels,
                        label=(
                            "Stage9 restore source-presence candidate after "
                            "bright-star audit"
                        ),
                    )
                    restored = True
            except (
                CommandError,
                SirilError,
                RuntimeError,
                AttributeError,
                TypeError,
                ValueError,
            ):
                restored = False
        if not restored:
            try:
                pipeline._apply_previous_stage_star_remix(
                    source_stem,
                    star_layer_name,
                    intensity,
                )
            except (
                CommandError,
                SirilError,
                RuntimeError,
                AttributeError,
                TypeError,
                ValueError,
            ):
                pass


def _stage9_extend_rescue_with_source_presence(
    pipeline,
    *,
    source_stem: str,
    accepted_context: Dict[str, Any],
    accepted_quality: Dict[str, Any],
    intensity: float,
    messages: List[str],
    remix_attempts: List[Dict[str, Any]],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Compete bounded source-wing strengths without relaxing hard gates."""
    accepted_context = dict(accepted_context)
    try:
        pipeline.cmd_with_check("load", source_stem)
        get_pixels = getattr(pipeline.siril, "get_image_pixeldata", None)
        remix_base = get_pixels(preview=False) if callable(get_pixels) else None
        if remix_base is None:
            raise RuntimeError("Stage9 immutable remix base pixels are unavailable")
        accepted_context["remix_base"] = np.array(remix_base, copy=True)
    except (
        AttributeError,
        CommandError,
        RuntimeError,
        SirilError,
        TypeError,
        ValueError,
    ) as error:
        messages.append(
            "Stage9 source-presence completion cannot bind saturated cores to "
            f"the actual remix base: {error}"
        )
    # The source-wing builder has a hard 0.95 amplitude ceiling.  Start at
    # that ceiling, then descend until the unchanged 1.10 FWHM gate accepts
    # a candidate.  This uses the remaining safe diameter headroom without
    # turning a stronger, rejected candidate into an implicit relaxation.
    strength_candidates = (0.95, 0.90, 0.85, 0.80)
    candidate_summaries: List[Dict[str, Any]] = []
    last_source_report: Dict[str, Any] = {}
    for feather_strength in strength_candidates:
        source_context = _prepare_stage9_source_presence_candidate(
            pipeline,
            accepted_context,
            messages,
            feather_strength=feather_strength,
            screen_intensity=intensity,
        )
        source_report = dict(source_context.get("source_presence_report") or {})
        last_source_report = source_report
        if not bool(source_report.get("changed", False)):
            break

        source_layer_name = str(
            source_context.get("starmask") or "starmask_unscreen_stabilized"
        )

        applied = bool(
            _save_stage9_unscreen_context_layer(pipeline, source_context)
            and pipeline._apply_previous_stage_star_remix(
                source_stem,
                source_layer_name,
                intensity,
            )
        )
        if not applied:
            candidate_summaries.append(
                {
                    "feather_strength": feather_strength,
                    "status": "failed",
                    "accepted": False,
                    "issues": ["source-presence remix execution failed"],
                }
            )
            break

        attempt_suffix = int(round(feather_strength * 100.0))
        quality = _assess_stage9_candidate(
            pipeline,
            source_stem,
            attempt=f"screen_unscreen_source_presence_{attempt_suffix}",
            formula="screen",
        )
        quality.update(
            intensity=intensity,
            starmask=source_layer_name,
            support_mode=str(
                accepted_quality.get("support_mode")
                or accepted_context.get("support_mode")
                or "unknown"
            ),
            support_starmask=str(
                accepted_quality.get("support_starmask")
                or accepted_context.get("support_starmask")
                or source_layer_name
            ),
            parent_attempt=accepted_quality.get("attempt"),
            base_source_stem=str(
                accepted_quality.get("base_source_stem") or source_stem
            ),
            recovery_kind="source_presence_extension",
            recovery_strength=feather_strength,
            recovery_target_groups=list(
                accepted_quality.get("recovery_target_groups") or []
            ),
            decomposition_method="matched_mtf_unscreen_source_presence",
            source_presence=source_report,
            source_wing_feather_strength=feather_strength,
        )
        quality["bright_star_presence"] = _stage9_observe_bright_star_presence(
            pipeline,
            source_stem=source_stem,
            star_layer_name=source_layer_name,
            intensity=intensity,
            completion_report=source_report.get("stage5_bright_star_completion"),
        )
        quality["reference_fidelity"] = _stage9_reference_fidelity(
            pipeline,
            source_context,
            source_context["unscreen_stars"],
            intensity,
            {
                "alpha_mask": source_context.get("support_mask"),
                "weak_mask": source_context.get("weak_mask"),
                "bright_mask": source_context.get("bright_mask"),
            },
        )
        candidate_summaries.append(
            {
                "feather_strength": feather_strength,
                "attempt": quality.get("attempt"),
                "status": quality.get("status"),
                "accepted": bool(quality.get("accepted", False)),
                "issues": list(quality.get("issues") or []),
                "fwhm_ratios": _stage9_psf_group_ratios(quality),
            }
        )
        source_report["candidate_comparison"] = copy.deepcopy(
            candidate_summaries
        )
        quality["source_wing_candidate_comparison"] = copy.deepcopy(
            candidate_summaries
        )
        remix_attempts.append(copy.deepcopy(quality))
        if bool(quality.get("accepted", False)):
            source_report["selected_feather_strength"] = feather_strength
            quality.setdefault("reason_codes", []).append(
                "stage9_unscreen_source_presence_selected"
            )
            pipeline._stage9_source_presence_report = source_report
            messages.append(
                "Stage9 source-presence extension passed all ordinary "
                "PSF/highlight/structure gates after Unscreen rescue "
                f"(feather_strength={feather_strength:.2f})"
            )
            return quality, source_context

        messages.append(
            "Stage9 source-presence candidate rejected; retrying with lower "
            f"source-wing feather strength ({feather_strength:.2f})"
        )

    pipeline._stage9_source_presence_report = (
        last_source_report
        if last_source_report
        else {
            "schema": "starun.stage9-source-presence.v1",
            "status": "not_run",
            "available": False,
        }
    )
    pipeline._stage9_source_presence_report["candidate_comparison"] = (
        candidate_summaries
    )
    _save_stage9_unscreen_context_layer(pipeline, accepted_context)
    accepted_layer_name = str(
        accepted_context.get("starmask") or "starmask_unscreen_stabilized"
    )
    pipeline._apply_previous_stage_star_remix(
        source_stem,
        accepted_layer_name,
        intensity,
    )
    messages.append(
        "Stage9 source-presence strengths were rejected; restored the accepted "
        "Unscreen rescue"
    )
    return accepted_quality, accepted_context


def _stage9_extend_with_selective_size(
    pipeline,
    *,
    source_stem: str,
    accepted_context: Dict[str, Any],
    accepted_quality: Dict[str, Any],
    intensity: float,
    messages: List[str],
    remix_attempts: List[Dict[str, Any]],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Compete per-star low-tail wing rescue without relaxing global gates."""
    if not bool(
        getattr(pipeline.cfg, "stage9_psf_selective_wing_enabled", True)
    ):
        return accepted_quality, accepted_context
    if not bool(accepted_quality.get("accepted", False)):
        return accepted_quality, accepted_context

    get_pixels = getattr(pipeline.siril, "get_image_pixeldata", None)
    if not callable(get_pixels):
        messages.append(
            "Stage9 selective size rescue unavailable: candidate pixel reader missing"
        )
        return accepted_quality, accepted_context
    current_pixels = get_pixels(preview=False)
    if current_pixels is None:
        messages.append(
            "Stage9 selective size rescue unavailable: accepted candidate pixels missing"
        )
        return accepted_quality, accepted_context
    candidate_display = np.array(current_pixels, copy=True)

    target = _clamp_float(
        getattr(pipeline.cfg, "stage9_psf_selective_wing_target_ratio", 1.08),
        float(getattr(pipeline.cfg, "stage9_psf_fwhm_ratio_min", 0.93)),
        float(getattr(pipeline.cfg, "stage9_psf_fwhm_ratio_max", 1.10)),
    )
    strength_max = _clamp_float(
        getattr(pipeline.cfg, "stage9_psf_selective_wing_strength_max", 1.15),
        0.90,
        1.25,
    )
    retry_max = max(
        0,
        min(
            2,
            int(
                getattr(
                    pipeline.cfg,
                    "stage9_psf_support_retry_pixels",
                    2,
                )
                or 0
            ),
        ),
    )
    # Larger support is the primary diameter control; amplitude steps then
    # descend inside the same source-bounded support if a hard gate rejects.
    support_candidates = tuple(range(retry_max, -1, -1))
    strength_candidates = tuple(
        dict.fromkeys(
            round(float(value), 2)
            for value in (strength_max, 1.10, 1.05, 1.00)
            if float(value) <= strength_max + 1e-9
        )
    )
    candidate_summaries: List[Dict[str, Any]] = []
    for support_extra_pixels in support_candidates:
        for feather_strength in strength_candidates:
            selective_context = _prepare_stage9_selective_size_candidate(
                pipeline,
                accepted_context,
                candidate_display,
                messages,
                screen_intensity=intensity,
                fwhm_ratio_target=target,
                feather_strength=feather_strength,
                support_extra_pixels=support_extra_pixels,
            )
            selective_report = dict(
                selective_context.get("selective_source_wing_report") or {}
            )
            if not bool(selective_report.get("changed", False)):
                if not candidate_summaries:
                    pipeline._stage9_source_presence_report.setdefault(
                        "selective_size_rescue",
                        selective_report,
                    )
                return accepted_quality, accepted_context

            applied = bool(
                _save_stage9_unscreen_context_layer(pipeline, selective_context)
                and pipeline._apply_previous_stage_star_remix(
                    source_stem,
                    "starmask_unscreen_stabilized",
                    intensity,
                )
            )
            attempt_suffix = (
                f"{support_extra_pixels}px_"
                f"{int(round(feather_strength * 100.0))}"
            )
            if not applied:
                candidate_summaries.append(
                    {
                        "attempt": f"screen_unscreen_selective_size_{attempt_suffix}",
                        "status": "failed",
                        "accepted": False,
                        "support_extra_pixels": support_extra_pixels,
                        "feather_strength": feather_strength,
                        "issues": ["selective size remix execution failed"],
                    }
                )
                break

            quality = _assess_stage9_candidate(
                pipeline,
                source_stem,
                attempt=f"screen_unscreen_selective_size_{attempt_suffix}",
                formula="screen",
            )
            quality.update(
                intensity=intensity,
                starmask="starmask_unscreen_stabilized",
                decomposition_method=(
                    "matched_mtf_unscreen_selective_source_wing"
                ),
                source_presence=selective_context.get(
                    "source_presence_report",
                    accepted_context.get("source_presence_report"),
                ),
                selective_source_wing=selective_report,
                selective_source_wing_target=target,
                selective_source_wing_strength=feather_strength,
                selective_source_wing_extra_pixels=support_extra_pixels,
            )
            visible_reference = selective_context.get(
                "source_autostretch_display"
            )
            catalog = getattr(
                pipeline,
                "_stage9_star_reference_catalog",
                None,
            )
            if visible_reference is not None and isinstance(catalog, dict):
                delivered_pixels = get_pixels(preview=False)
                if delivered_pixels is not None:
                    quality["visible_wing_closure"] = (
                        stage9_quality.assess_stage9_visible_wing_closure(
                            delivered_pixels,
                            visible_reference,
                            catalog,
                            pipeline.cfg,
                        )
                    )
            # The selective candidate never edits saturated Stage 5
            # completions.  Carry the already measured presence observation
            # forward instead of silently dropping that audit from the final
            # selected quality object.
            bright_presence = accepted_quality.get("bright_star_presence")
            if isinstance(bright_presence, dict):
                quality["bright_star_presence"] = copy.deepcopy(bright_presence)
            quality["reference_fidelity"] = _stage9_reference_fidelity(
                pipeline,
                selective_context,
                selective_context["unscreen_stars"],
                intensity,
                {
                    "alpha_mask": selective_context.get("support_mask"),
                    "weak_mask": selective_context.get("weak_mask"),
                    "bright_mask": selective_context.get("bright_mask"),
                },
            )
            summary = {
                "attempt": quality.get("attempt"),
                "status": quality.get("status"),
                "accepted": bool(quality.get("accepted", False)),
                "support_extra_pixels": support_extra_pixels,
                "feather_strength": feather_strength,
                "selected_star_count": int(
                    selective_report.get("selected_star_count", 0) or 0
                ),
                "selected_star_ratio": float(
                    selective_report.get("selected_star_ratio", 0.0) or 0.0
                ),
                "fwhm_ratios": _stage9_psf_group_ratios(quality),
                "issues": list(quality.get("issues") or []),
            }
            candidate_summaries.append(summary)
            quality["selective_size_candidate_comparison"] = copy.deepcopy(
                candidate_summaries
            )
            remix_attempts.append(copy.deepcopy(quality))
            if bool(quality.get("accepted", False)):
                selective_report["candidate_comparison"] = copy.deepcopy(
                    candidate_summaries
                )
                selective_report["selected"] = True
                quality.setdefault("reason_codes", []).append(
                    "stage9_unscreen_selective_size_selected"
                )
                source_presence = dict(
                    getattr(pipeline, "_stage9_source_presence_report", {}) or {}
                )
                source_presence["selective_size_rescue"] = selective_report
                pipeline._stage9_source_presence_report = source_presence
                messages.append(
                    "Stage9 selective same-star size candidate passed all "
                    "ordinary PSF/highlight/structure gates "
                    f"(selected={summary['selected_star_count']}, "
                    f"extra_pixels={support_extra_pixels}, "
                    f"strength={feather_strength:.2f})"
                )
                return quality, selective_context

            messages.append(
                "Stage9 selective same-star size candidate rejected; "
                f"retrying (extra_pixels={support_extra_pixels}, "
                f"strength={feather_strength:.2f})"
            )
            direction = _stage9_psf_size_direction(quality)
            if direction not in {"large", None}:
                break

    source_presence = dict(
        getattr(pipeline, "_stage9_source_presence_report", {}) or {}
    )
    source_presence["selective_size_rescue"] = {
        "schema": "starun.stage9-selective-source-wing.v1",
        "status": "rejected",
        "available": True,
        "changed": False,
        "candidate_comparison": candidate_summaries,
        "reason": "all selective size candidates failed unchanged hard gates",
    }
    pipeline._stage9_source_presence_report = source_presence
    _save_stage9_unscreen_context_layer(pipeline, accepted_context)
    pipeline._apply_previous_stage_star_remix(
        source_stem,
        "starmask_unscreen_stabilized",
        intensity,
    )
    messages.append(
        "Stage9 selective size candidates were rejected; restored the accepted "
        "source-presence candidate"
    )
    return accepted_quality, accepted_context


def _stage9_reference_fidelity(
    pipeline,
    context: Dict[str, Any],
    stars: np.ndarray,
    intensity: float,
    state: Dict[str, Any],
) -> Dict[str, Any]:
    configured_weak_intensity = _clamp_float(
        getattr(pipeline.cfg, "stage9_weak_star_screen_intensity_min", 0.55),
        0.10,
        1.05,
    )
    return stage9_quality.assess_unscreen_reference_fidelity(
        context["original_display"],
        context["starless_display"],
        stars,
        intensity=float(intensity),
        support_mask=context["support_mask"],
        alpha_mask=state.get("alpha_mask"),
        weak_mask=state.get("weak_mask"),
        bright_mask=state.get("bright_mask"),
        weak_intensity=max(float(intensity), configured_weak_intensity),
    )


def _capture_stage9_candidate_state(pipeline) -> Dict[str, Any]:
    def copy_array(name: str) -> np.ndarray | None:
        value = getattr(pipeline, name, None)
        return None if value is None else np.array(value, copy=True)

    return {
        "star_layer": copy_array("_stage9_last_star_layer"),
        "alpha_mask": copy_array("_stage9_last_star_overlay_mask"),
        "weak_mask": copy_array("_stage9_last_weak_overlay_mask"),
        "bright_mask": copy_array("_stage9_last_bright_overlay_mask"),
        "star_color_validation": copy.deepcopy(
            getattr(pipeline, "_stage9_star_color_post_validation", None)
        ),
        "starmask_calibration": copy.deepcopy(
            getattr(pipeline, "_stage9_starmask_calibration", None)
        ),
        "unscreen_reference": copy.deepcopy(
            getattr(pipeline, "_stage9_unscreen_reference", None)
        ),
        "source_presence_report": copy.deepcopy(
            getattr(pipeline, "_stage9_source_presence_report", None)
        ),
        "star_layer_decomposition": str(
            getattr(pipeline, "_stage9_star_layer_decomposition", "") or ""
        ),
    }


def _stage9_support_candidate_score(
    quality: Dict[str, Any],
    *,
    support_mode: str,
) -> tuple[float, ...]:
    """Rank accepted normal/compact primary candidates deterministically."""
    if not bool(quality.get("accepted", False)):
        return (float("inf"),) * 10
    gates = quality.get("quality_gates") or {}
    advisories = list(quality.get("advisories") or [])
    advisory_present = bool(
        str(quality.get("status") or "").lower() == "advisory"
        or advisories
        or any(
            isinstance(gate, dict) and bool(gate.get("advisory", False))
            for gate in gates.values()
        )
    )

    ratios = list(_stage9_psf_group_ratios(quality).values())
    worst_psf_error = (
        max(abs(float(value) - 1.0) for value in ratios)
        if ratios
        else float("inf")
    )
    mean_psf_error = (
        float(np.mean([abs(float(value) - 1.0) for value in ratios]))
        if ratios
        else float("inf")
    )
    metrics = quality.get("metrics") or {}
    limits = quality.get("limits") or {}
    visibility_names = (
        "catalog_star_visibility_ratio_all",
        "catalog_star_visibility_ratio_weak",
        "catalog_star_visibility_ratio_bright",
    )
    if any(name in metrics for name in visibility_names):
        visibility_deficit = 0.0
        for name in visibility_names:
            try:
                ratio = float(metrics[name])
            except (KeyError, TypeError, ValueError):
                visibility_deficit = float("inf")
                break
            if not np.isfinite(ratio):
                visibility_deficit = float("inf")
                break
            visibility_deficit += max(0.0, 1.0 - ratio)
    else:
        visibility_deficit = 0.0
    recovery_margins: List[float] = []
    for metric_name in (
        "weak_star_recovery_ratio",
        "star_recovery_ratio",
        "star_positive_delta_window_recovery_ratio",
        "star_wing_recovery_ratio",
    ):
        try:
            limit = float(limits[metric_name])
            value = float(metrics[metric_name])
        except (KeyError, TypeError, ValueError):
            continue
        if limit > 0.0:
            recovery_margins.append(value / limit)
    minimum_recovery_margin = min(recovery_margins or [0.0])
    try:
        support_ratio = float(metrics.get("star_support_ratio", float("inf")))
    except (TypeError, ValueError):
        support_ratio = float("inf")
    try:
        highlight_growth = float(
            metrics.get("highlight_clip_growth", float("inf"))
        )
    except (TypeError, ValueError):
        highlight_growth = float("inf")
    try:
        bright_growth = float(metrics.get("bright_pixel_growth", float("inf")))
    except (TypeError, ValueError):
        bright_growth = float("inf")
    normal_tie_break = 0.0 if support_mode == "normal" else 1.0
    return (
        1.0 if _stage9_psf_uncertainty_exemption_used(quality) else 0.0,
        visibility_deficit,
        1.0 if advisory_present else 0.0,
        worst_psf_error,
        mean_psf_error,
        -minimum_recovery_margin,
        support_ratio,
        highlight_growth,
        bright_growth,
        normal_tie_break,
    )


def _stage9_support_failure_allows_intensity_retry(
    quality: Dict[str, Any],
) -> bool:
    """Return whether lowering Screen intensity can plausibly fix rejection."""
    if _stage9_has_recovery_shortfall(quality):
        return False
    issue_text = " ".join(str(item) for item in quality.get("issues", [])).lower()
    structural_tokens = (
        "unavailable",
        "shape mismatch",
        "non-finite",
        "finite_ratio",
        "star_recovery_metrics_unavailable",
        "local_quality",
        "candidate_sample_count",
        "reference catalog",
    )
    return not any(token in issue_text for token in structural_tokens)


def _stage9_failed_support_candidate_score(
    quality: Dict[str, Any],
    *,
    support_mode: str,
) -> tuple[float, int, int]:
    gates = quality.get("quality_gates") or {}
    severities = [
        float(gate.get("severity_ratio", 1.0) or 1.0)
        for gate in gates.values()
        if isinstance(gate, dict) and bool(gate.get("hard_failed", False))
    ]
    return (
        max(severities or [float(len(quality.get("issues") or []))]),
        len(quality.get("issues") or []),
        0 if support_mode == "strict_compact" else 1,
    )


def _restore_stage9_candidate_state(
    pipeline,
    state: Dict[str, Any],
    *,
    checkpoint_stem: str,
) -> None:
    pipeline.cmd_with_check("load", checkpoint_stem)
    pipeline._stage9_last_star_layer = state.get("star_layer")
    pipeline._stage9_last_star_overlay_mask = state.get("alpha_mask")
    pipeline._stage9_last_weak_overlay_mask = state.get("weak_mask")
    pipeline._stage9_last_bright_overlay_mask = state.get("bright_mask")
    pipeline._stage9_star_color_post_validation = state.get(
        "star_color_validation"
    )
    pipeline._stage9_starmask_calibration = state.get("starmask_calibration")
    if state.get("unscreen_reference") is not None:
        pipeline._stage9_unscreen_reference = copy.deepcopy(
            state.get("unscreen_reference")
        )
    if state.get("source_presence_report") is not None:
        pipeline._stage9_source_presence_report = copy.deepcopy(
            state.get("source_presence_report")
        )
    if state.get("star_layer_decomposition"):
        pipeline._stage9_star_layer_decomposition = str(
            state.get("star_layer_decomposition")
        )


def _activate_stage9_candidate_state(pipeline, state: Dict[str, Any]) -> None:
    """Activate a frozen star-layer context without changing the current image."""
    for attribute, key in (
        ("_stage9_last_star_layer", "star_layer"),
        ("_stage9_last_star_overlay_mask", "alpha_mask"),
        ("_stage9_last_weak_overlay_mask", "weak_mask"),
        ("_stage9_last_bright_overlay_mask", "bright_mask"),
    ):
        value = state.get(key)
        setattr(
            pipeline,
            attribute,
            None if value is None else np.array(value, copy=True),
        )
    pipeline._stage9_star_color_post_validation = copy.deepcopy(
        state.get("star_color_validation")
    )
    pipeline._stage9_starmask_calibration = copy.deepcopy(
        state.get("starmask_calibration")
    )


def _stage9_targeted_recovery_retry_max(pipeline) -> int:
    return max(
        0,
        min(
            4,
            int(
                getattr(
                    pipeline.cfg,
                    "stage9_targeted_recovery_retry_max",
                    3,
                )
                or 0
            ),
        ),
    )


def _stage9_psf_recovery_target_groups(
    pipeline,
    quality: Dict[str, Any],
) -> tuple[str, ...]:
    """Return concrete weak/bright groups that need size recovery."""
    ratios = _stage9_psf_group_ratios(quality)
    if not ratios:
        return ()
    closure = quality.get("psf_closure") or {}
    limits = closure.get("limits") or {}
    try:
        hard_min = float(
            limits.get(
                "stage9_psf_fwhm_ratio_min",
                getattr(pipeline.cfg, "stage9_psf_fwhm_ratio_min", 0.93),
            )
        )
    except (TypeError, ValueError):
        hard_min = 0.93
    target_min = _stage9_psf_recovery_target_min(pipeline, quality)
    groups = tuple(
        group
        for group in ("weak", "bright")
        if group in ratios and ratios[group] < hard_min
    )
    if groups:
        return groups
    if bool(quality.get("accepted", False)):
        groups = tuple(
            group
            for group in ("weak", "bright")
            if group in ratios and ratios[group] < target_min
        )
        if groups:
            return groups
    if ratios.get("all", target_min) < hard_min:
        return tuple(group for group in ("weak", "bright") if group in ratios) or (
            "weak",
            "bright",
        )
    return ()


def _stage9_is_psf_small_only_failure(
    pipeline,
    quality: Dict[str, Any],
) -> bool:
    if bool(quality.get("accepted", False)):
        return False
    if _stage9_psf_size_direction(quality) != "small":
        return False
    if not _stage9_psf_recovery_target_groups(pipeline, quality):
        return False
    issues = [str(item).lower() for item in quality.get("issues", [])]
    return bool(issues) and all("star_psf_fwhm" in issue for issue in issues)


def _stage9_psf_contraction_target_groups(
    pipeline,
    quality: Dict[str, Any],
) -> tuple[str, ...]:
    """Return only the frozen ordinary-star groups above the hard PSF limit."""
    ratios = _stage9_psf_group_ratios(quality)
    if not ratios:
        return ()
    closure = quality.get("psf_closure") or {}
    limits = closure.get("limits") or {}
    try:
        hard_max = float(
            limits.get(
                "stage9_psf_fwhm_ratio_max",
                getattr(pipeline.cfg, "stage9_psf_fwhm_ratio_max", 1.10),
            )
        )
    except (TypeError, ValueError):
        hard_max = 1.10
    groups = tuple(
        group
        for group in ("weak", "bright")
        if group in ratios and ratios[group] > hard_max
    )
    if groups:
        return groups
    if ratios.get("all", 0.0) > hard_max:
        return ("all",)
    return ()


def _stage9_is_psf_large_only_failure(
    pipeline,
    quality: Dict[str, Any],
) -> bool:
    if bool(quality.get("accepted", False)):
        return False
    if _stage9_psf_size_direction(quality) != "large":
        return False
    if not _stage9_psf_contraction_target_groups(pipeline, quality):
        return False
    issues = [str(item).lower() for item in quality.get("issues", [])]
    return bool(issues) and all("star_psf_fwhm" in issue for issue in issues)


def _stage9_is_chroma_only_failure(quality: Dict[str, Any]) -> bool:
    if bool(quality.get("accepted", False)):
        return False
    ratios = _stage9_psf_group_ratios(quality)
    closure = quality.get("psf_closure") or {}
    limits = closure.get("limits") or {}
    try:
        lower = float(limits.get("stage9_psf_fwhm_ratio_min", 0.93))
        upper = float(limits.get("stage9_psf_fwhm_ratio_max", 1.10))
    except (TypeError, ValueError):
        lower, upper = 0.93, 1.10
    if ratios and any(not lower <= value <= upper for value in ratios.values()):
        return False
    structural, numeric = _stage9_structured_failure_classification(quality)
    if structural:
        return False
    if numeric:
        return all(
            str(item.get("metric") or "") == "chromatic_star_addition_ratio"
            for item in numeric
        )
    issues = [str(item).lower() for item in quality.get("issues", [])]
    return bool(issues) and all(
        "chromatic_star_addition_ratio" in issue for issue in issues
    )


def _save_stage9_candidate_star_layer(
    pipeline,
    *,
    source_starmask: str,
    output_name: str,
    stars: np.ndarray,
    support_mask: np.ndarray,
    weak_mask: np.ndarray | None,
    bright_mask: np.ndarray | None,
    label: str,
) -> bool:
    """Persist one candidate star layer together with its exact overlay masks."""
    try:
        pipeline.cmd_with_check("load", source_starmask)
        writer = getattr(pipeline, "_set_current_image_pixeldata", None)
        if callable(writer):
            writer(stars, label=label)
        else:
            set_pixels = getattr(pipeline.siril, "set_image_pixeldata", None)
            if not callable(set_pixels):
                return False
            lock_factory = getattr(pipeline.siril, "image_lock", None)
            if callable(lock_factory):
                with lock_factory():
                    set_pixels(stars)
            else:
                set_pixels(stars)
        masks = dict(
            getattr(pipeline, "_stage9_candidate_overlay_masks", {}) or {}
        )
        masks[output_name] = {
            "weak": None if weak_mask is None else np.array(weak_mask, copy=True),
            "bright": (
                None if bright_mask is None else np.array(bright_mask, copy=True)
            ),
            "alpha": np.array(support_mask, copy=True),
        }
        pipeline._stage9_candidate_overlay_masks = masks
        return bool(pipeline._save_stage_output(output_name))
    except (AttributeError, CommandError, RuntimeError, SirilError, ValueError):
        return False


def _stage9_formal_candidate_score(
    quality: Dict[str, Any],
    *,
    support_mode: str,
) -> tuple[float, ...]:
    """Rank every formally accepted Screen/Unscreen candidate uniformly."""
    if not bool(quality.get("accepted", False)):
        return (float("inf"),) * 9
    ratios = tuple(_stage9_psf_group_ratios(quality).values())
    worst_psf = (
        max(abs(value - 1.0) for value in ratios) if ratios else float("inf")
    )
    metrics = quality.get("metrics") or {}

    def finite_metric(name: str, default: float) -> float:
        try:
            value = float(metrics.get(name, default))
        except (TypeError, ValueError):
            return default
        return value if np.isfinite(value) else default

    fidelity = quality.get("reference_fidelity") or {}
    try:
        fidelity_mae = float(fidelity.get("support_rgb_mae", float("inf")))
    except (TypeError, ValueError):
        fidelity_mae = float("inf")
    recoveries = [
        finite_metric(name, 0.0)
        for name in (
            "weak_star_recovery_ratio",
            "star_recovery_ratio",
            "star_positive_delta_window_recovery_ratio",
        )
    ]
    wing = finite_metric("star_wing_recovery_ratio", 0.0)
    visibility_metric_names = (
        ("catalog_star_visibility_ratio_all", 1.0),
        ("catalog_star_visibility_ratio_weak", 1.0),
        ("catalog_star_visibility_ratio_bright", 1.0),
    )
    visibility_metrics_present = any(
        name in metrics for name, _target in visibility_metric_names
    )
    if visibility_metrics_present:
        visibility_deficit = sum(
            max(0.0, target - finite_metric(name, float("-inf")))
            for name, target in visibility_metric_names
        )
    else:
        # Legacy test doubles and v8 resume reports predate the absolute
        # visibility metrics. They remain rankable, but a new production
        # assessment cannot be formally accepted without the hard gate.
        visibility_deficit = 0.0
    return (
        1.0 if _stage9_psf_uncertainty_exemption_used(quality) else 0.0,
        visibility_deficit,
        worst_psf,
        fidelity_mae,
        -min(recoveries),
        -wing,
        finite_metric("highlight_clip_growth", float("inf")),
        finite_metric("bright_pixel_growth", float("inf")),
        0.0 if support_mode == "normal" else 1.0,
    )


def _stage9_targeted_local_chroma_recovery(
    pipeline,
    *,
    source_stem: str,
    parent_quality: Dict[str, Any],
    parent_context: Dict[str, Any],
    intensity: float,
    support_mode: str,
    messages: List[str],
    remix_attempts: List[Dict[str, Any]],
    review_candidate_registry: List[Dict[str, Any]],
    retry_budget: int | None = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Try local outer-pixel attenuation for a chroma-only rejection."""
    if not _stage9_is_chroma_only_failure(parent_quality):
        return parent_quality, parent_context
    retry_max = _stage9_targeted_recovery_retry_max(pipeline)
    if retry_budget is not None:
        retry_max = min(retry_max, max(0, int(retry_budget)))
    if retry_max <= 0:
        return parent_quality, parent_context
    stars = parent_context.get("stars", parent_context.get("unscreen_stars"))
    support = parent_context.get("support_mask")
    parent_starmask = str(parent_context.get("starmask") or "")
    if stars is None or support is None or not parent_starmask:
        return parent_quality, parent_context
    catalog = getattr(pipeline, "_stage9_star_reference_catalog", None)
    if not isinstance(catalog, dict) or catalog.get("status") != "ok":
        return parent_quality, parent_context
    get_pixels = getattr(pipeline.siril, "get_image_pixeldata", None)
    if not callable(get_pixels):
        return parent_quality, parent_context
    candidate_display = get_pixels(preview=False)
    if candidate_display is None:
        return parent_quality, parent_context
    parent_checkpoint = (
        f"stage9_candidate_{support_mode}_"
        f"{len(remix_attempts):03d}_local_chroma_parent"
    )
    if not pipeline._save_stage_output(parent_checkpoint):
        messages.append(
            "Stage9 skipped local chroma recovery because the parent checkpoint "
            "could not be saved"
        )
        return parent_quality, parent_context
    parent_state = _capture_stage9_candidate_state(pipeline)
    try:
        pipeline.cmd_with_check("load", source_stem)
        remix_base = get_pixels(preview=False)
        if remix_base is None:
            raise RuntimeError("immutable remix base pixels are unavailable")
        _strict_weak, _strict_bright, strict_core = (
            stage9_quality.build_star_overlay_masks(
                catalog,
                strict=True,
                cfg=pipeline.cfg,
            )
        )
    except (CommandError, RuntimeError, SirilError, TypeError, ValueError) as error:
        _restore_stage9_candidate_state(
            pipeline,
            parent_state,
            checkpoint_stem=parent_checkpoint,
        )
        messages.append(f"Stage9 local chroma recovery unavailable: {error}")
        return parent_quality, parent_context

    configured_levels = tuple(
        float(value)
        for value in getattr(
            pipeline.cfg,
            "stage9_fallback_intensity_levels",
            (0.75, 0.55, 0.40),
        )
        if 0.05 <= float(value) < 1.0
    )
    levels = tuple(dict.fromkeys(configured_levels))[:retry_max]
    comparison: List[Dict[str, Any]] = []
    parent_attempt = str(parent_quality.get("attempt") or "unknown")
    for attenuation in levels:
        recovered, recovery_report = (
            stage9_quality.build_local_chroma_recovery_layer(
                np.asarray(stars),
                np.asarray(remix_base),
                np.asarray(candidate_display),
                np.asarray(support),
                np.asarray(strict_core),
                pipeline.cfg,
                attenuation=attenuation,
            )
        )
        if recovered is None or not bool(recovery_report.get("changed", False)):
            break
        suffix = int(round(attenuation * 100.0))
        output_name = f"starmask_{support_mode}_local_chroma_{suffix:02d}"
        saved = _save_stage9_candidate_star_layer(
            pipeline,
            source_starmask=parent_starmask,
            output_name=output_name,
            stars=recovered,
            support_mask=np.asarray(support),
            weak_mask=parent_context.get("weak_mask"),
            bright_mask=parent_context.get("bright_mask"),
            label="Stage9 local chroma recovery star layer",
        )
        applied = bool(
            saved
            and pipeline._apply_previous_stage_star_remix(
                source_stem,
                output_name,
                intensity,
            )
        )
        attempt_name = f"{parent_attempt}_local_chroma_{suffix:02d}"
        quality = (
            _assess_stage9_candidate(
                pipeline,
                source_stem,
                attempt=attempt_name,
                formula="screen",
            )
            if applied
            else {
                "attempt": attempt_name,
                "formula": "screen",
                "status": "failed",
                "accepted": False,
                "issues": ["local chroma recovery remix execution failed"],
                "metrics": {},
            }
        )
        quality.update(
            intensity=intensity,
            starmask=output_name,
            support_mode=support_mode,
            support_starmask=parent_context.get(
                "support_starmask",
                parent_starmask,
            ),
            parent_attempt=parent_attempt,
            base_source_stem=source_stem,
            recovery_kind="local_chroma_attenuation",
            recovery_strength=attenuation,
            recovery_target_groups=[],
            local_chroma_recovery=recovery_report,
            decomposition_method=(
                str(parent_quality.get("decomposition_method") or "screen")
                + "_local_chroma_bounded"
            ),
        )
        _stage9_consider_review_candidate(
            pipeline,
            quality,
            attempt_order=len(remix_attempts),
            registry=review_candidate_registry,
            messages=messages,
        )
        remix_attempts.append(copy.deepcopy(quality))
        comparison.append(
            {
                "attempt": attempt_name,
                "attenuation": attenuation,
                "accepted": bool(quality.get("accepted", False)),
                "status": str(quality.get("status") or "unknown"),
                "issues": list(quality.get("issues") or []),
                "fwhm_ratios": _stage9_psf_group_ratios(quality),
                "chromatic_star_addition_ratio": (
                    (quality.get("metrics") or {}).get(
                        "chromatic_star_addition_ratio"
                    )
                ),
            }
        )
        quality["local_chroma_candidate_comparison"] = copy.deepcopy(comparison)
        if bool(quality.get("accepted", False)):
            quality.setdefault("reason_codes", []).append(
                "stage9_local_chroma_recovery_selected"
            )
            context = {
                **parent_context,
                "stars": recovered,
                "unscreen_stars": recovered,
                "starmask": output_name,
                "local_chroma_recovery": recovery_report,
            }
            messages.append(
                "Stage9 selected local chroma recovery without changing the "
                f"strict core (support={support_mode}, attenuation={attenuation:.2f})"
            )
            return quality, context
        if not _stage9_is_chroma_only_failure(quality):
            break

    _restore_stage9_candidate_state(
        pipeline,
        parent_state,
        checkpoint_stem=parent_checkpoint,
    )
    parent_quality["local_chroma_candidate_comparison"] = comparison
    messages.append(
        "Stage9 local chroma candidates were rejected; restored the exact parent "
        f"attempt={parent_attempt}"
    )
    return parent_quality, parent_context


def _stage9_targeted_soft_psf_recovery(
    pipeline,
    *,
    source_stem: str,
    parent_quality: Dict[str, Any],
    parent_context: Dict[str, Any],
    intensity: float,
    support_mode: str,
    messages: List[str],
    remix_attempts: List[Dict[str, Any]],
    review_candidate_registry: List[Dict[str, Any]],
    retry_budget: int | None = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Search a bounded fractional source-wing weight for failed star groups."""
    if not _stage9_is_psf_small_only_failure(pipeline, parent_quality):
        return parent_quality, parent_context
    retry_max = _stage9_targeted_recovery_retry_max(pipeline)
    if retry_budget is not None:
        retry_max = min(retry_max, max(0, int(retry_budget)))
    target_groups = _stage9_psf_recovery_target_groups(pipeline, parent_quality)
    if retry_max <= 0 or not target_groups:
        return parent_quality, parent_context
    get_pixels = getattr(pipeline.siril, "get_image_pixeldata", None)
    if not callable(get_pixels):
        return parent_quality, parent_context
    parent_display = get_pixels(preview=False)
    if parent_display is None:
        return parent_quality, parent_context
    parent_checkpoint = (
        f"stage9_candidate_{support_mode}_"
        f"{len(remix_attempts):03d}_soft_psf_parent"
    )
    if not pipeline._save_stage_output(parent_checkpoint):
        messages.append(
            "Stage9 skipped targeted PSF recovery because the parent checkpoint "
            "could not be saved"
        )
        return parent_quality, parent_context
    parent_state = _capture_stage9_candidate_state(pipeline)
    context = dict(parent_context)
    try:
        pipeline.cmd_with_check("load", source_stem)
        remix_base = get_pixels(preview=False)
        if remix_base is None:
            raise RuntimeError("immutable remix base pixels are unavailable")
        context["remix_base"] = np.array(remix_base, copy=True)
    except (CommandError, RuntimeError, SirilError) as error:
        _restore_stage9_candidate_state(
            pipeline,
            parent_state,
            checkpoint_stem=parent_checkpoint,
        )
        messages.append(f"Stage9 targeted PSF recovery unavailable: {error}")
        return parent_quality, parent_context

    target_min = _stage9_psf_recovery_target_min(pipeline, parent_quality)
    target_max = _stage9_psf_recovery_target_max(pipeline, parent_quality)
    strength = _clamp_float(
        getattr(pipeline.cfg, "stage9_psf_selective_wing_strength_max", 1.15),
        0.90,
        1.25,
    )
    support_extra_pixels = max(
        0,
        min(
            2,
            int(getattr(pipeline.cfg, "stage9_psf_support_retry_pixels", 2) or 0),
        ),
    )
    low_alpha = 0.0
    high_alpha = 1.0
    best: Dict[str, Any] | None = None
    comparisons: List[Dict[str, Any]] = []
    parent_attempt = str(parent_quality.get("attempt") or "unknown")
    for _retry in range(retry_max):
        recovery_alpha = 0.5 * (low_alpha + high_alpha)
        selective_context = _prepare_stage9_selective_size_candidate(
            pipeline,
            context,
            np.asarray(parent_display),
            messages,
            screen_intensity=intensity,
            fwhm_ratio_target=target_min,
            feather_strength=strength,
            support_extra_pixels=support_extra_pixels,
            target_groups=target_groups,
            recovery_alpha=recovery_alpha,
        )
        recovery_report = dict(
            selective_context.get("selective_source_wing_report") or {}
        )
        if not bool(recovery_report.get("changed", False)):
            break
        suffix = int(round(recovery_alpha * 1000.0))
        output_name = f"starmask_{support_mode}_soft_psf_{suffix:03d}"
        recovered_stars = selective_context.get("unscreen_stars")
        recovered_support = selective_context.get("support_mask")
        saved = bool(
            recovered_stars is not None
            and recovered_support is not None
            and _save_stage9_candidate_star_layer(
                pipeline,
                source_starmask=str(context.get("starmask") or ""),
                output_name=output_name,
                stars=np.asarray(recovered_stars),
                support_mask=np.asarray(recovered_support),
                weak_mask=selective_context.get("weak_mask"),
                bright_mask=selective_context.get("bright_mask"),
                label="Stage9 targeted fractional PSF star layer",
            )
        )
        applied = bool(
            saved
            and pipeline._apply_previous_stage_star_remix(
                source_stem,
                output_name,
                intensity,
            )
        )
        attempt_name = f"{parent_attempt}_soft_psf_{suffix:03d}"
        quality = (
            _assess_stage9_candidate(
                pipeline,
                source_stem,
                attempt=attempt_name,
                formula="screen",
            )
            if applied
            else {
                "attempt": attempt_name,
                "formula": "screen",
                "status": "failed",
                "accepted": False,
                "issues": ["targeted soft PSF remix execution failed"],
                "metrics": {},
            }
        )
        quality.update(
            intensity=intensity,
            starmask=output_name,
            support_mode=support_mode,
            support_starmask=context.get("support_starmask"),
            parent_attempt=parent_attempt,
            base_source_stem=source_stem,
            recovery_kind="group_fractional_source_wing",
            recovery_strength=recovery_alpha,
            recovery_target_groups=list(target_groups),
            selective_source_wing=recovery_report,
            decomposition_method=(
                str(parent_quality.get("decomposition_method") or "screen")
                + "_group_fractional_source_wing"
            ),
        )
        if recovered_stars is not None and all(
            key in context for key in ("original_display", "starless_display")
        ):
            quality["reference_fidelity"] = _stage9_reference_fidelity(
                pipeline,
                selective_context,
                np.asarray(recovered_stars),
                intensity,
                {
                    "alpha_mask": recovered_support,
                    "weak_mask": selective_context.get("weak_mask"),
                    "bright_mask": selective_context.get("bright_mask"),
                },
            )
        _stage9_consider_review_candidate(
            pipeline,
            quality,
            attempt_order=len(remix_attempts),
            registry=review_candidate_registry,
            messages=messages,
        )
        remix_attempts.append(copy.deepcopy(quality))
        ratios = _stage9_psf_group_ratios(quality)
        comparisons.append(
            {
                "attempt": attempt_name,
                "recovery_alpha": recovery_alpha,
                "target_groups": list(target_groups),
                "accepted": bool(quality.get("accepted", False)),
                "status": str(quality.get("status") or "unknown"),
                "issues": list(quality.get("issues") or []),
                "fwhm_ratios": ratios,
            }
        )
        quality["targeted_psf_candidate_comparison"] = copy.deepcopy(comparisons)
        if bool(quality.get("accepted", False)):
            checkpoint = f"stage9_candidate_{support_mode}_soft_psf_{suffix:03d}"
            if pipeline._save_stage_output(checkpoint):
                record = {
                    "quality": quality,
                    "context": {
                        **selective_context,
                        "starmask": output_name,
                    },
                    "checkpoint": checkpoint,
                    "state": _capture_stage9_candidate_state(pipeline),
                    "score": _stage9_formal_candidate_score(
                        quality,
                        support_mode=support_mode,
                    ),
                }
                if best is None or record["score"] < best["score"]:
                    best = record
            within_soft_target = bool(ratios) and all(
                target_min <= ratio <= target_max for ratio in ratios.values()
            )
            if within_soft_target:
                break
            low_alpha = recovery_alpha
        else:
            direction = _stage9_psf_size_direction(quality)
            if direction == "large":
                high_alpha = recovery_alpha
            elif direction == "small":
                low_alpha = recovery_alpha
            else:
                break

    if best is None:
        _restore_stage9_candidate_state(
            pipeline,
            parent_state,
            checkpoint_stem=parent_checkpoint,
        )
        parent_quality["targeted_psf_candidate_comparison"] = comparisons
        messages.append(
            "Stage9 targeted soft PSF candidates were rejected; restored the "
            f"exact parent attempt={parent_attempt}"
        )
        return parent_quality, parent_context
    _restore_stage9_candidate_state(
        pipeline,
        best["state"],
        checkpoint_stem=str(best["checkpoint"]),
    )
    selected_quality = best["quality"]
    selected_quality["targeted_psf_candidate_comparison"] = comparisons
    selected_quality.setdefault("reason_codes", []).append(
        "stage9_targeted_soft_psf_recovery_selected"
    )
    messages.append(
        "Stage9 selected bounded group-aware soft PSF recovery "
        f"(groups={','.join(target_groups)}, strength="
        f"{float(selected_quality.get('recovery_strength', 0.0)):.3f})"
    )
    return selected_quality, best["context"]


def _stage9_targeted_psf_contraction(
    pipeline,
    *,
    source_stem: str,
    parent_quality: Dict[str, Any],
    parent_context: Dict[str, Any],
    intensity: float,
    support_mode: str,
    messages: List[str],
    remix_attempts: List[Dict[str, Any]],
    review_candidate_registry: List[Dict[str, Any]],
    retry_budget: int | None = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Search bounded component-local contraction for a pure large-PSF failure."""
    if not _stage9_is_psf_large_only_failure(pipeline, parent_quality):
        return parent_quality, parent_context
    retry_max = _stage9_targeted_recovery_retry_max(pipeline)
    if retry_budget is not None:
        retry_max = min(retry_max, max(0, int(retry_budget)))
    target_groups = _stage9_psf_contraction_target_groups(
        pipeline,
        parent_quality,
    )
    stars = parent_context.get("stars", parent_context.get("unscreen_stars"))
    support = parent_context.get("support_mask")
    weak_mask = parent_context.get("weak_mask")
    bright_mask = parent_context.get("bright_mask")
    parent_starmask = str(parent_context.get("starmask") or "")
    catalog = getattr(pipeline, "_stage9_star_reference_catalog", None)
    if (
        retry_max <= 0
        or not target_groups
        or stars is None
        or support is None
        or not parent_starmask
        or not isinstance(catalog, dict)
        or catalog.get("status") != "ok"
    ):
        return parent_quality, parent_context

    parent_checkpoint = (
        f"stage9_candidate_{support_mode}_"
        f"{len(remix_attempts):03d}_psf_contraction_parent"
    )
    if not pipeline._save_stage_output(parent_checkpoint):
        messages.append(
            "Stage9 skipped PSF contraction because the immutable parent "
            "checkpoint could not be saved"
        )
        return parent_quality, parent_context
    parent_state = _capture_stage9_candidate_state(pipeline)
    immutable_parent_stars = np.array(stars, copy=True)
    low_gamma = 1.0
    high_gamma = 4.0
    target_ratio = 1.0
    best: Dict[str, Any] | None = None
    comparisons: List[Dict[str, Any]] = []
    parent_attempt = str(parent_quality.get("attempt") or "unknown")
    parent_ratios = _stage9_psf_group_ratios(parent_quality)
    parent_target_values = [
        parent_ratios[group]
        for group in target_groups
        if group in parent_ratios
    ]
    if not parent_target_values and "all" in parent_ratios:
        parent_target_values = [parent_ratios["all"]]
    if not parent_target_values:
        return parent_quality, parent_context
    feedback_ratio = float(max(parent_target_values))
    requested_gamma = float(
        np.clip(
            (feedback_ratio / target_ratio) ** 2,
            low_gamma,
            high_gamma,
        )
    )
    if requested_gamma <= 1.0 + 1.0e-6:
        return parent_quality, parent_context
    gamma = requested_gamma

    for _retry in range(retry_max):
        contracted, contraction_report = (
            stage9_quality.contract_star_layer_components(
                immutable_parent_stars,
                catalog,
                support_mask=np.asarray(support),
                weak_mask=(
                    None if weak_mask is None else np.asarray(weak_mask)
                ),
                bright_mask=(
                    None if bright_mask is None else np.asarray(bright_mask)
                ),
                target_groups=target_groups,
                gamma=gamma,
            )
        )
        suffix = int(round(gamma * 1000.0))
        attempt_name = f"{parent_attempt}_psf_contract_{suffix:04d}"
        output_name = f"starmask_{support_mode}_psf_contract_{suffix:04d}"
        if contracted is None or not bool(
            contraction_report.get("changed", False)
        ):
            comparisons.append(
                {
                    "attempt": attempt_name,
                    "gamma": gamma,
                    "target_groups": list(target_groups),
                    "accepted": False,
                    "status": str(
                        contraction_report.get("status") or "unavailable"
                    ),
                    "issues": [
                        str(
                            contraction_report.get("reason")
                            or "component contraction made no safe change"
                        )
                    ],
                    "fwhm_ratios": {},
                    "contraction": contraction_report,
                }
            )
            break
        saved = _save_stage9_candidate_star_layer(
            pipeline,
            source_starmask=parent_starmask,
            output_name=output_name,
            stars=np.asarray(contracted),
            support_mask=np.asarray(support),
            weak_mask=(None if weak_mask is None else np.asarray(weak_mask)),
            bright_mask=(
                None if bright_mask is None else np.asarray(bright_mask)
            ),
            label="Stage9 component-local PSF contraction star layer",
        )
        applied = bool(
            saved
            and pipeline._apply_previous_stage_star_remix(
                source_stem,
                output_name,
                intensity,
            )
        )
        quality = (
            _assess_stage9_candidate(
                pipeline,
                source_stem,
                attempt=attempt_name,
                formula="screen",
            )
            if applied
            else {
                "attempt": attempt_name,
                "formula": "screen",
                "status": "failed",
                "accepted": False,
                "issues": ["component PSF contraction remix execution failed"],
                "metrics": {},
            }
        )
        quality.update(
            intensity=intensity,
            starmask=output_name,
            support_mode=support_mode,
            support_starmask=parent_context.get(
                "support_starmask",
                parent_starmask,
            ),
            parent_attempt=parent_attempt,
            base_source_stem=source_stem,
            recovery_kind="group_component_psf_contraction",
            recovery_strength=gamma,
            recovery_target_groups=list(target_groups),
            psf_contraction=contraction_report,
            decomposition_method=(
                str(parent_quality.get("decomposition_method") or "screen")
                + "_component_psf_contraction"
            ),
        )
        candidate_context = {
            **parent_context,
            "stars": contracted,
            "unscreen_stars": contracted,
            "starmask": output_name,
            "psf_contraction": contraction_report,
        }
        if all(
            key in parent_context
            for key in ("original_display", "starless_display")
        ):
            quality["reference_fidelity"] = _stage9_reference_fidelity(
                pipeline,
                candidate_context,
                np.asarray(contracted),
                intensity,
                {
                    "alpha_mask": support,
                    "weak_mask": weak_mask,
                    "bright_mask": bright_mask,
                },
            )
        _stage9_consider_review_candidate(
            pipeline,
            quality,
            attempt_order=len(remix_attempts),
            registry=review_candidate_registry,
            messages=messages,
        )
        remix_attempts.append(copy.deepcopy(quality))
        ratios = _stage9_psf_group_ratios(quality)
        comparison = {
            "attempt": attempt_name,
            "gamma": gamma,
            "feedback_input_ratio": feedback_ratio,
            "feedback_target_ratio": target_ratio,
            "target_groups": list(target_groups),
            "accepted": bool(quality.get("accepted", False)),
            "status": str(quality.get("status") or "unknown"),
            "issues": list(quality.get("issues") or []),
            "fwhm_ratios": ratios,
            "contraction": contraction_report,
        }
        comparisons.append(comparison)
        quality["psf_contraction_candidate_comparison"] = copy.deepcopy(
            comparisons
        )
        if bool(quality.get("accepted", False)):
            checkpoint = (
                f"stage9_candidate_{support_mode}_psf_contract_{suffix:04d}"
            )
            target_values = [
                ratios[group]
                for group in target_groups
                if group in ratios
            ]
            if not target_values and "all" in ratios:
                target_values = [ratios["all"]]
            psf_error = (
                max(abs(value - 1.0) for value in target_values)
                if target_values
                else float("inf")
            )
            mean_error = (
                float(np.mean([abs(value - 1.0) for value in target_values]))
                if target_values
                else float("inf")
            )
            if pipeline._save_stage_output(checkpoint):
                record = {
                    "quality": quality,
                    "context": candidate_context,
                    "checkpoint": checkpoint,
                    "state": _capture_stage9_candidate_state(pipeline),
                    "score": (
                        psf_error,
                        mean_error,
                        *_stage9_formal_candidate_score(
                            quality,
                            support_mode=support_mode,
                        ),
                        gamma,
                    ),
                }
                if best is None or record["score"] < best["score"]:
                    best = record

        target_values = [
            ratios[group] for group in target_groups if group in ratios
        ]
        if not target_values and "all" in ratios:
            target_values = [ratios["all"]]
        if target_values:
            feedback_ratio = float(max(target_values))
            if feedback_ratio > target_ratio:
                low_gamma = gamma
            else:
                high_gamma = gamma
        else:
            direction = _stage9_psf_size_direction(quality)
            if direction == "large":
                low_gamma = gamma
            elif direction == "small":
                high_gamma = gamma
            else:
                break
        if not bool(quality.get("accepted", False)) and (
            _stage9_psf_size_direction(quality) not in {"large", "small"}
        ):
            break
        if high_gamma - low_gamma <= 1.0e-4:
            break
        feedback_gamma = float(
            gamma * (feedback_ratio / target_ratio) ** 2
        )
        next_gamma = float(
            np.clip(feedback_gamma, low_gamma, high_gamma)
        )
        if feedback_ratio > target_ratio and next_gamma <= gamma + 1.0e-4:
            next_gamma = 0.5 * (gamma + high_gamma)
        elif (
            feedback_ratio < target_ratio
            and next_gamma >= gamma - 1.0e-4
        ):
            next_gamma = 0.5 * (low_gamma + gamma)
        if abs(next_gamma - gamma) <= 1.0e-4:
            break
        comparison["feedback_next_gamma"] = next_gamma
        comparison["feedback_unclamped_gamma"] = feedback_gamma
        gamma = next_gamma

    if best is None:
        _restore_stage9_candidate_state(
            pipeline,
            parent_state,
            checkpoint_stem=parent_checkpoint,
        )
        parent_quality["psf_contraction_candidate_comparison"] = comparisons
        parent_quality["psf_contraction_rollback"] = {
            "performed": True,
            "restored": "immutable_parent",
            "selected": False,
        }
        messages.append(
            "Stage9 PSF contraction candidates were rejected; restored the "
            f"exact immutable parent attempt={parent_attempt}"
        )
        return parent_quality, parent_context

    _restore_stage9_candidate_state(
        pipeline,
        best["state"],
        checkpoint_stem=str(best["checkpoint"]),
    )
    selected_quality = best["quality"]
    selected_quality["psf_contraction_candidate_comparison"] = comparisons
    selected_quality["psf_contraction_rollback"] = {
        "performed": False,
        "restored": "selected_candidate_checkpoint",
        "selected": True,
    }
    selected_quality.setdefault("reason_codes", []).append(
        "stage9_component_psf_contraction_selected"
    )
    messages.append(
        "Stage9 selected component-local PSF contraction after all formal "
        f"gates (groups={','.join(target_groups)}, gamma="
        f"{float(selected_quality.get('recovery_strength', 1.0)):.4f})"
    )
    return selected_quality, best["context"]


def _stage9_targeted_unscreen_competition(
    pipeline,
    *,
    source_stem: str,
    primary_support_results: List[Dict[str, Any]],
    selected_screen: Optional[Dict[str, Any]],
    messages: List[str],
    remix_attempts: List[Dict[str, Any]],
    review_candidate_registry: List[Dict[str, Any]],
) -> tuple[Optional[Dict[str, Any]], str]:
    """Compete normal/strict Unscreen contexts from their frozen Screen states."""
    formal_records: List[Dict[str, Any]] = []
    for record in primary_support_results:
        quality = record.get("quality") or {}
        if (
            bool(quality.get("accepted", False))
            and record.get("checkpoint")
            and isinstance(record.get("state"), dict)
        ):
            formal_records.append(record)

    unscreen_reports: List[Dict[str, Any]] = []
    unscreen_records: List[Dict[str, Any]] = []
    for record in primary_support_results:
        screen_quality = record.get("quality") or {}
        state = record.get("state")
        if not isinstance(state, dict) or state.get("star_layer") is None:
            continue
        support_mode = str(record.get("support_mode") or "unknown")
        support_starmask = str(record.get("starmask") or "")
        if not support_starmask:
            continue
        _activate_stage9_candidate_state(pipeline, state)
        output_name = f"starmask_unscreen_{support_mode}"
        context = _prepare_stage9_unscreen_candidate(
            pipeline,
            support_starmask,
            messages,
            output_name=output_name,
            support_mode=support_mode,
        )
        context_report = _stage9_unscreen_context_report(context)
        support_report: Dict[str, Any] = {
            "support_mode": support_mode,
            "support_starmask": support_starmask,
            "output_starmask": output_name,
            "status": str(context_report.get("status") or "unavailable"),
            "available": bool(context.get("available", False)),
            "reason_code": context_report.get("reason_code"),
        }
        if not bool(context.get("available", False)):
            unscreen_reports.append(support_report)
            continue
        intensity = float(screen_quality.get("intensity", 1.0) or 1.0)
        applied = pipeline._apply_previous_stage_star_remix(
            source_stem,
            output_name,
            intensity,
        )
        attempt_name = f"screen_unscreen_{support_mode}_primary"
        quality = (
            _assess_stage9_candidate(
                pipeline,
                source_stem,
                attempt=attempt_name,
                formula="screen",
            )
            if applied
            else {
                "attempt": attempt_name,
                "formula": "screen",
                "status": "failed",
                "accepted": False,
                "issues": ["Unscreen support competition execution failed"],
                "metrics": {},
            }
        )
        quality.update(
            intensity=intensity,
            starmask=output_name,
            support_mode=support_mode,
            support_starmask=support_starmask,
            parent_attempt=screen_quality.get("attempt"),
            base_source_stem=source_stem,
            recovery_kind=(
                "strict_support_unscreen"
                if support_mode == "strict_compact"
                else "unscreen_amplitude_recovery"
            ),
            recovery_strength=0.0,
            recovery_target_groups=[],
            decomposition_method="matched_mtf_unscreen_chroma_stabilized",
        )
        quality["reference_fidelity"] = _stage9_reference_fidelity(
            pipeline,
            context,
            context["unscreen_stars"],
            intensity,
            state,
        )
        _stage9_consider_review_candidate(
            pipeline,
            quality,
            attempt_order=len(remix_attempts),
            registry=review_candidate_registry,
            messages=messages,
        )
        remix_attempts.append(copy.deepcopy(quality))

        targeted_attempt_start = len(remix_attempts)
        targeted_attempt_budget = _stage9_targeted_recovery_retry_max(pipeline)
        quality, context = _stage9_targeted_soft_psf_recovery(
            pipeline,
            source_stem=source_stem,
            parent_quality=quality,
            parent_context=context,
            intensity=intensity,
            support_mode=support_mode,
            messages=messages,
            remix_attempts=remix_attempts,
            review_candidate_registry=review_candidate_registry,
            retry_budget=targeted_attempt_budget,
        )
        targeted_attempt_budget = max(
            0,
            targeted_attempt_budget
            - (len(remix_attempts) - targeted_attempt_start),
        )
        contraction_attempt_start = len(remix_attempts)
        quality, context = _stage9_targeted_psf_contraction(
            pipeline,
            source_stem=source_stem,
            parent_quality=quality,
            parent_context=context,
            intensity=intensity,
            support_mode=support_mode,
            messages=messages,
            remix_attempts=remix_attempts,
            review_candidate_registry=review_candidate_registry,
            retry_budget=targeted_attempt_budget,
        )
        targeted_attempt_budget = max(
            0,
            targeted_attempt_budget
            - (len(remix_attempts) - contraction_attempt_start),
        )
        quality, context = _stage9_targeted_local_chroma_recovery(
            pipeline,
            source_stem=source_stem,
            parent_quality=quality,
            parent_context=context,
            intensity=intensity,
            support_mode=support_mode,
            messages=messages,
            remix_attempts=remix_attempts,
            review_candidate_registry=review_candidate_registry,
            retry_budget=targeted_attempt_budget,
        )
        if bool(quality.get("accepted", False)):
            quality, context = _stage9_extend_rescue_with_source_presence(
                pipeline,
                source_stem=source_stem,
                accepted_context=context,
                accepted_quality=quality,
                intensity=intensity,
                messages=messages,
                remix_attempts=remix_attempts,
            )
        source_presence_selected = bool(
            str(quality.get("attempt") or "").startswith(
                "screen_unscreen_source_presence"
            )
            and quality.get("accepted", False)
        )
        if source_presence_selected:
            eligible = True
            comparison = {
                "schema": "starun.stage9-unscreen-comparison.v1",
                "selected": True,
                "reason_code": "stage9_unscreen_source_presence_selected",
                "selection_basis": (
                    "same_source_completeness_after_all_unchanged_formal_gates"
                ),
            }
        elif bool(screen_quality.get("accepted", False)):
            comparison = stage9_quality.compare_unscreen_candidate(
                screen_quality,
                quality,
                pipeline.cfg,
            )
            eligible = bool(comparison.get("selected", False))
        else:
            eligible = bool(quality.get("accepted", False))
            comparison = {
                "schema": "starun.stage9-unscreen-comparison.v1",
                "selected": eligible,
                "reason_code": (
                    "stage9_unscreen_selected_rescue"
                    if eligible
                    else "stage9_unscreen_candidate_rejected"
                ),
                "rescue_without_formal_screen": True,
                "selection_basis": "all_unchanged_formal_quality_gates",
            }
        quality["comparison_to_support_screen"] = comparison
        quality.setdefault("reason_codes", []).append(
            str(comparison.get("reason_code") or "stage9_unscreen_candidate_rejected")
        )
        support_report.update(
            attempt=str(quality.get("attempt") or attempt_name),
            accepted=bool(quality.get("accepted", False)),
            eligible=eligible,
            comparison=comparison,
            recovery_kind=str(quality.get("recovery_kind") or "none"),
            fwhm_ratios=_stage9_psf_group_ratios(quality),
            issues=list(quality.get("issues") or []),
        )
        unscreen_reports.append(support_report)
        if not eligible:
            continue
        safe_mode = "".join(
            character if character.isalnum() else "_"
            for character in support_mode
        )
        checkpoint = f"stage9_candidate_unscreen_{safe_mode}"
        if not pipeline._save_stage_output(checkpoint):
            support_report["eligible"] = False
            support_report["checkpoint_status"] = "save_failed"
            continue
        candidate_record = {
            "support_mode": support_mode,
            "starmask": str(context.get("starmask") or output_name),
            "calibration": copy.deepcopy(
                getattr(pipeline, "_stage9_starmask_calibration", {}) or {}
            ),
            "quality": quality,
            "checkpoint": checkpoint,
            "state": _capture_stage9_candidate_state(pipeline),
            "score": _stage9_formal_candidate_score(
                quality,
                support_mode=support_mode,
            ),
            "context": context,
            "candidate_family": "unscreen",
        }
        unscreen_records.append(candidate_record)
        formal_records.append(candidate_record)

    selected_record = (
        min(
            formal_records,
            key=lambda item: _stage9_formal_candidate_score(
                item.get("quality") or {},
                support_mode=str(item.get("support_mode") or "unknown"),
            ),
        )
        if formal_records
        else None
    )
    selected_quality = selected_screen
    selected_starmask = str(
        (selected_screen or {}).get("starmask")
        or (primary_support_results[0].get("starmask") if primary_support_results else "")
    )
    selected_family = "screen"
    selected_context: Dict[str, Any] | None = None
    selected_context_report: Dict[str, Any] | None = None
    if selected_record is not None:
        checkpoint = str(selected_record.get("checkpoint") or "")
        state = selected_record.get("state")
        if checkpoint and isinstance(state, dict):
            _restore_stage9_candidate_state(
                pipeline,
                state,
                checkpoint_stem=checkpoint,
            )
            selected_quality = selected_record.get("quality")
            selected_starmask = str(selected_record.get("starmask") or "")
            selected_family = str(
                selected_record.get("candidate_family") or "screen"
            )
            context = selected_record.get("context")
            if isinstance(context, dict):
                selected_context = context
                selected_context_report = _stage9_unscreen_context_report(
                    selected_context
                )

    if (
        selected_family == "unscreen"
        and isinstance(selected_quality, dict)
        and isinstance(selected_context, dict)
        and bool(selected_quality.get("accepted", False))
        and not str(selected_quality.get("attempt") or "").startswith(
            "screen_unscreen_source_presence"
        )
    ):
        support_comparison = selected_quality.get(
            "comparison_to_support_screen"
        )
        selected_quality, selected_context = (
            _stage9_extend_rescue_with_source_presence(
                pipeline,
                source_stem=source_stem,
                accepted_context=selected_context,
                accepted_quality=selected_quality,
                intensity=float(selected_quality.get("intensity", 1.0) or 1.0),
                messages=messages,
                remix_attempts=remix_attempts,
            )
        )
        if support_comparison is not None:
            selected_quality.setdefault(
                "comparison_to_support_screen",
                support_comparison,
            )
        selected_starmask = str(
            selected_quality.get("starmask")
            or selected_context.get("starmask")
            or selected_starmask
        )
        selected_context_report = _stage9_unscreen_context_report(
            selected_context
        )

    aggregate_report = dict(
        selected_context_report
        or _stage9_unscreen_unavailable(
            "no Unscreen support candidate won the formal competition"
        )
    )
    aggregate_report.update(
        support_candidates=unscreen_reports,
        selected_support_mode=(
            str((selected_quality or {}).get("support_mode") or "")
            if selected_family == "unscreen"
            else None
        ),
        selection_status=(
            "selected" if selected_family == "unscreen" else "retained_screen"
        ),
        reason_code=(
            "stage9_unscreen_selected"
            if selected_family == "unscreen"
            else "stage9_screen_selected_after_unscreen_competition"
        ),
    )
    pipeline._stage9_unscreen_reference = aggregate_report
    if isinstance(selected_quality, dict):
        pipeline._stage9_selected_remix_quality = dict(selected_quality)
        if selected_family == "unscreen":
            pipeline._stage9_star_layer_decomposition = str(
                selected_quality.get("decomposition_method")
                or "matched_mtf_unscreen_chroma_stabilized"
            )
    messages.append(
        "Stage9 completed explicit normal/strict Screen-Unscreen competition "
        f"(unscreen_candidates={len(unscreen_reports)}, selected={selected_family}, "
        f"attempt={str((selected_quality or {}).get('attempt') or 'none')})"
    )
    return selected_quality, selected_starmask


def _stage9_targeted_intensity_feasibility(
    pipeline,
    *,
    source_stem: str,
    primary_support_results: List[Dict[str, Any]],
    candidates: List[tuple[str, float]],
    route: str,
    messages: List[str],
    remix_attempts: List[Dict[str, Any]],
    review_candidate_registry: List[Dict[str, Any]],
) -> tuple[Optional[Dict[str, Any]], str]:
    """Probe the configured floor before spending attempts on middle levels."""
    if len(candidates) <= 1:
        return None, ""
    actionable = [
        item
        for item in primary_support_results
        if _stage9_support_failure_allows_intensity_retry(item.get("quality") or {})
    ]
    if not actionable:
        return None, ""
    support = min(
        actionable,
        key=lambda item: _stage9_failed_support_candidate_score(
            item.get("quality") or {},
            support_mode=str(item.get("support_mode") or "unknown"),
        ),
    )
    starmask = str(support.get("starmask") or "")
    support_mode = str(support.get("support_mode") or "unknown")
    pipeline._stage9_starmask_calibration = copy.deepcopy(
        support.get("calibration") or {}
    )
    floor_label, floor_intensity = candidates[-1]

    def assess_level(label: str, level: float) -> Dict[str, Any]:
        applied = pipeline._apply_previous_stage_star_remix(
            source_stem,
            starmask,
            level,
        )
        attempt_name = f"screen_{label}"
        quality = (
            _assess_stage9_candidate(
                pipeline,
                source_stem,
                attempt=attempt_name,
                formula="screen",
            )
            if applied
            else {
                "attempt": attempt_name,
                "formula": "screen",
                "status": "failed",
                "accepted": False,
                "issues": ["pixel remix execution failed"],
                "metrics": {},
            }
        )
        quality.update(
            intensity=level,
            starmask=starmask,
            support_mode=support_mode,
            support_starmask=starmask,
            parent_attempt=(support.get("quality") or {}).get("attempt"),
            base_source_stem=source_stem,
            recovery_kind="global_intensity_feasibility",
            recovery_strength=level,
            recovery_target_groups=[],
            support_preflight_route=route,
        )
        _stage9_consider_review_candidate(
            pipeline,
            quality,
            attempt_order=len(remix_attempts),
            registry=review_candidate_registry,
            messages=messages,
        )
        remix_attempts.append(quality)
        return quality

    floor_quality = assess_level(floor_label, floor_intensity)
    if not bool(floor_quality.get("accepted", False)):
        messages.append(
            "Stage9 intensity feasibility floor failed; skipped all middle "
            f"levels (support={support_mode}, intensity={floor_intensity:.3f})"
        )
        return None, starmask
    floor_checkpoint = "stage9_candidate_intensity_feasibility_floor"
    if not pipeline._save_stage_output(floor_checkpoint):
        return floor_quality, starmask
    floor_state = _capture_stage9_candidate_state(pipeline)
    selected = floor_quality
    for label, level in candidates[1:-1]:
        quality = assess_level(label, level)
        if bool(quality.get("accepted", False)):
            selected = quality
            floor_checkpoint = ""
            floor_state = {}
            break
    if floor_checkpoint:
        _restore_stage9_candidate_state(
            pipeline,
            floor_state,
            checkpoint_stem=floor_checkpoint,
        )
    selected.setdefault("reason_codes", []).append(
        "stage9_intensity_feasibility_selected"
    )
    messages.append(
        "Stage9 selected the highest accepted intensity after a successful "
        f"floor feasibility probe (intensity={float(selected.get('intensity', 0.0)):.3f})"
    )
    return selected, starmask


def _stage9_consider_review_candidate(
    pipeline,
    quality: Dict[str, Any],
    *,
    attempt_order: int,
    registry: List[Dict[str, Any]],
    messages: List[str],
) -> None:
    """Annotate an attempt and transactionally retain an eligible candidate."""
    eligibility = _stage9_review_candidate_eligibility(
        pipeline,
        quality,
        attempt_order=attempt_order,
    )
    failure_action = str(
        getattr(pipeline.cfg, "stage9_failure_action", "auto_fallback")
        or "auto_fallback"
    )
    if (
        bool(eligibility.get("eligible", False))
        and failure_action != "auto_fallback"
    ):
        eligibility["eligible"] = False
        eligibility["reasons"] = [
            "review_policy_disabled_for_failure_action"
        ]
    quality["review_eligibility"] = eligibility
    if not bool(eligibility.get("eligible", False)):
        return

    attempt = str(quality.get("attempt") or f"attempt_{attempt_order}")
    safe_attempt = "".join(
        character if character.isalnum() else "_"
        for character in attempt
    ).strip("_") or f"attempt_{attempt_order}"
    checkpoint = f"stage9_review_candidate_{attempt_order:03d}_{safe_attempt}"
    try:
        state = _capture_stage9_candidate_state(pipeline)
    except (RuntimeError, TypeError, ValueError) as error:
        eligibility["eligible"] = False
        eligibility["reasons"] = ["candidate_state_checkpoint_failed"]
        eligibility["checkpoint_state"] = {
            "status": "capture_failed",
            "image_checkpoint_saved": False,
            "error": str(error),
        }
        messages.append(
            "Stage9 excluded bounded review candidate because its associated "
            f"state snapshot failed (attempt={attempt}): {error}"
        )
        return
    if state.get("star_color_validation") is None and isinstance(
        quality.get("star_color_validation"),
        dict,
    ):
        state["star_color_validation"] = copy.deepcopy(
            quality.get("star_color_validation")
        )
    checkpoint_state = {
        "status": "captured",
        "image_checkpoint_saved": False,
        "star_layer_captured": state.get("star_layer") is not None,
        "alpha_mask_captured": state.get("alpha_mask") is not None,
        "weak_mask_captured": state.get("weak_mask") is not None,
        "bright_mask_captured": state.get("bright_mask") is not None,
        "calibration_captured": isinstance(
            state.get("starmask_calibration"),
            dict,
        )
        and bool(state.get("starmask_calibration")),
        "star_color_validation_captured": isinstance(
            state.get("star_color_validation"),
            dict,
        ),
    }
    eligibility["checkpoint_state"] = checkpoint_state
    if not all(
        value
        for key, value in checkpoint_state.items()
        if key not in {"status", "image_checkpoint_saved"}
    ):
        checkpoint_state["status"] = "incomplete"
        eligibility["eligible"] = False
        eligibility["reasons"] = ["candidate_state_checkpoint_incomplete"]
        messages.append(
            "Stage9 excluded bounded review candidate because its star-layer, "
            f"mask, calibration, or color-validation state was incomplete "
            f"(attempt={attempt})"
        )
        return
    try:
        checkpoint_saved = bool(pipeline._save_stage_output(checkpoint))
    except (CommandError, RuntimeError, SirilError) as error:
        checkpoint_saved = False
        eligibility["checkpoint_error"] = str(error)
    if not checkpoint_saved:
        checkpoint_state["status"] = "save_failed"
        eligibility["eligible"] = False
        eligibility["reasons"] = ["candidate_checkpoint_save_failed"]
        eligibility["checkpoint_saved"] = False
        messages.append(
            "Stage9 excluded bounded review candidate because its checkpoint "
            f"could not be saved (attempt={attempt})"
        )
        return

    eligibility["checkpoint_saved"] = True
    checkpoint_state["status"] = "saved"
    checkpoint_state["image_checkpoint_saved"] = True
    eligibility["checkpoint_stem"] = checkpoint
    score = tuple(
        float("inf") if value is None else float(value)
        for value in eligibility.get("selection_score") or []
    )
    registry.append(
        {
            "attempt_order": max(0, int(attempt_order)),
            "quality": quality,
            "checkpoint": checkpoint,
            "state": state,
            "score": score,
        }
    )
    messages.append(
        "Stage9 retained bounded review candidate checkpoint "
        f"attempt={attempt}, review_fwhm_max="
        f"{float(eligibility['review_fwhm_ratio_max']):.3f}"
    )


def _stage9_psf_candidate_summary(
    quality: Dict[str, Any],
    *,
    recovery_pixels: int,
) -> Dict[str, Any]:
    score = _stage9_psf_candidate_score(
        quality,
        recovery_pixels=recovery_pixels,
    )
    return {
        "attempt": str(quality.get("attempt") or ""),
        "status": str(quality.get("status") or "unknown"),
        "accepted": bool(quality.get("accepted", False)),
        "review_required": bool(quality.get("review_required", False)),
        "recovery_pixels": max(0, int(recovery_pixels)),
        "fwhm_ratios": _stage9_psf_group_ratios(quality),
        "selection_score": [
            None if not np.isfinite(value) else float(value)
            for value in score
        ],
    }


def _stage9_progressive_screen_psf_recovery(
    pipeline,
    *,
    source_stem: str,
    trusted_starmask_name: str,
    baseline_starmask_name: str,
    candidate_intensity: float,
    baseline_quality: Dict[str, Any],
    messages: List[str],
    remix_attempts: List[Dict[str, Any]],
    review_candidate_registry: List[Dict[str, Any]],
) -> tuple[Optional[Dict[str, Any]], str, Dict[str, Any]]:
    """Try +1 then +2 source-confirmed wing pixels and retain the safest fit."""
    retry_max = max(
        0,
        min(
            2,
            int(
                getattr(
                    pipeline.cfg,
                    "stage9_psf_support_retry_pixels",
                    2,
                )
                or 0
            ),
        ),
    )
    target_min = _stage9_psf_recovery_target_min(pipeline, baseline_quality)
    baseline_quality.setdefault("psf_support_recovery_pixels", 0)
    baseline_quality["psf_support_recovery_target_min"] = target_min
    candidate_summaries = [
        _stage9_psf_candidate_summary(
            baseline_quality,
            recovery_pixels=0,
        )
    ]
    best_quality: Optional[Dict[str, Any]] = None
    best_starmask_name = baseline_starmask_name
    best_checkpoint = ""
    best_state: Dict[str, Any] | None = None
    best_score = (float("inf"),) * 5
    current_quality = baseline_quality

    if bool(baseline_quality.get("accepted", False)):
        baseline_checkpoint = "stage9_candidate_screen_psf_recovery_0px"
        if not pipeline._save_stage_output(baseline_checkpoint):
            messages.append(
                "Stage9 progressive PSF recovery skipped because the accepted "
                "baseline checkpoint could not be saved"
            )
            return baseline_quality, baseline_starmask_name, baseline_quality
        best_quality = baseline_quality
        best_checkpoint = baseline_checkpoint
        best_state = _capture_stage9_candidate_state(pipeline)
        best_score = _stage9_psf_candidate_score(
            baseline_quality,
            recovery_pixels=0,
        )

    for retry_pixels in range(1, retry_max + 1):
        if not _stage9_needs_progressive_psf_recovery(pipeline, current_quality):
            break
        output_name = f"starmask_stretched_psf_recovery_{retry_pixels}px"
        recovery_starmask_name = _prepare_stage9_starmask_for_pixel_remix(
            pipeline,
            trusted_starmask_name,
            star_stretch_used=False,
            messages=messages,
            strict_support=False,
            support_retry_pixels=retry_pixels,
            output_name=output_name,
            candidate_local=True,
        )
        recovered = bool(
            recovery_starmask_name
            and pipeline._apply_previous_stage_star_remix(
                source_stem,
                recovery_starmask_name,
                candidate_intensity,
            )
        )
        attempt_name = f"screen_psf_support_recovery_{retry_pixels}px"
        if not recovered:
            failed_quality = {
                "attempt": attempt_name,
                "formula": "screen",
                "intensity": candidate_intensity,
                "starmask": recovery_starmask_name,
                "psf_support_recovery_pixels": retry_pixels,
                "psf_support_recovery_target_min": target_min,
                "status": "failed",
                "accepted": False,
                "issues": ["source-wing PSF remix execution failed"],
                "metrics": {},
            }
            remix_attempts.append(failed_quality)
            candidate_summaries.append(
                _stage9_psf_candidate_summary(
                    failed_quality,
                    recovery_pixels=retry_pixels,
                )
            )
            current_quality = failed_quality
            messages.append(
                "Stage9 progressive source-wing PSF recovery execution failed "
                f"at +{retry_pixels} px"
            )
            break

        recovery_quality = _assess_stage9_candidate(
            pipeline,
            source_stem,
            attempt=attempt_name,
            formula="screen",
        )
        recovery_quality.update(
            intensity=candidate_intensity,
            starmask=recovery_starmask_name,
            psf_support_recovery_pixels=retry_pixels,
            psf_support_recovery_target_min=target_min,
        )
        _stage9_consider_review_candidate(
            pipeline,
            recovery_quality,
            attempt_order=len(remix_attempts),
            registry=review_candidate_registry,
            messages=messages,
        )
        remix_attempts.append(recovery_quality)
        candidate_summaries.append(
            _stage9_psf_candidate_summary(
                recovery_quality,
                recovery_pixels=retry_pixels,
            )
        )
        current_quality = recovery_quality
        ratios = _stage9_psf_group_ratios(recovery_quality)
        ratio_note = ",".join(
            f"{name}={value:.3f}" for name, value in ratios.items()
        ) or "unavailable"
        messages.append(
            "Stage9 progressive source-wing PSF candidate "
            f"+{retry_pixels} px: accepted="
            f"{str(bool(recovery_quality.get('accepted', False))).lower()}, "
            f"ratios={ratio_note}"
        )

        if bool(recovery_quality.get("accepted", False)):
            checkpoint = f"stage9_candidate_screen_psf_recovery_{retry_pixels}px"
            candidate_score = _stage9_psf_candidate_score(
                recovery_quality,
                recovery_pixels=retry_pixels,
            )
            if pipeline._save_stage_output(checkpoint):
                if candidate_score < best_score:
                    best_quality = recovery_quality
                    best_starmask_name = recovery_starmask_name
                    best_checkpoint = checkpoint
                    best_state = _capture_stage9_candidate_state(pipeline)
                    best_score = candidate_score
            elif best_quality is None:
                # The current image is valid but cannot be compared safely with
                # another retry without a rollback checkpoint.
                best_quality = recovery_quality
                best_starmask_name = recovery_starmask_name
                best_checkpoint = ""
                best_state = _capture_stage9_candidate_state(pipeline)
                best_score = candidate_score
                messages.append(
                    "Stage9 stopped progressive PSF recovery because the current "
                    f"+{retry_pixels} px candidate checkpoint could not be saved"
                )
                break

        direction = _stage9_psf_size_direction(recovery_quality)
        if direction == "large":
            messages.append(
                "Stage9 stopped progressive PSF recovery after an oversized "
                f"+{retry_pixels} px candidate"
            )
            break
        if (
            not bool(recovery_quality.get("accepted", False))
            and direction != "small"
        ):
            messages.append(
                "Stage9 stopped progressive PSF recovery after a non-size "
                f"quality rejection at +{retry_pixels} px"
            )
            break

    if best_quality is None:
        try:
            pipeline.cmd_with_check("load", source_stem)
        except (CommandError, RuntimeError, SirilError) as error:
            messages.append(f"Stage9 PSF recovery rollback failed: {error}")
        return None, baseline_starmask_name, current_quality

    if best_checkpoint and best_state is not None:
        _restore_stage9_candidate_state(
            pipeline,
            best_state,
            checkpoint_stem=best_checkpoint,
        )
    selected_pixels = int(best_quality.get("psf_support_recovery_pixels", 0) or 0)
    best_quality["psf_support_recovery_target_min"] = target_min
    best_quality["psf_support_candidate_comparison"] = candidate_summaries
    best_quality["psf_support_selection_score"] = [float(value) for value in best_score]
    selected_calibration = dict(
        getattr(pipeline, "_stage9_starmask_calibration", {}) or {}
    )
    selected_calibration.update(
        psf_recovery_attempted=True,
        psf_recovery_mode="progressive_source_confirmed_wing_support",
        psf_recovery_target_min=target_min,
        psf_recovery_pixels=selected_pixels,
        psf_recovery_candidate_comparison=candidate_summaries,
        psf_recovery_accepted=True,
    )
    pipeline._stage9_starmask_calibration = selected_calibration
    messages.append(
        "Stage9 selected the smallest safe source-wing candidate closest to "
        f"the original display PSF (pixels={selected_pixels}, "
        f"target_min={target_min:.3f})"
    )
    return best_quality, best_starmask_name, current_quality


def _stage9_progressive_unscreen_psf_recovery(
    pipeline,
    *,
    source_stem: str,
    trusted_starmask_name: str,
    baseline_intensity: float,
    screen_baseline_state: Dict[str, Any],
    initial_context: Dict[str, Any],
    initial_quality: Dict[str, Any],
    messages: List[str],
    remix_attempts: List[Dict[str, Any]],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Progressively rebuild an Unscreen candidate from +1/+2 wing support."""
    retry_max = max(
        0,
        min(
            2,
            int(
                getattr(
                    pipeline.cfg,
                    "stage9_psf_support_retry_pixels",
                    2,
                )
                or 0
            ),
        ),
    )
    target_min = _stage9_psf_recovery_target_min(pipeline, initial_quality)
    initial_quality.setdefault("psf_support_recovery_pixels", 0)
    initial_quality["psf_support_recovery_target_min"] = target_min
    if retry_max <= 0 or not _stage9_needs_progressive_psf_recovery(
        pipeline,
        initial_quality,
    ):
        return initial_quality, initial_context

    initial_checkpoint = "stage9_candidate_unscreen_psf_recovery_0px"
    if not pipeline._save_stage_output(initial_checkpoint):
        messages.append(
            "Stage9 progressive Unscreen PSF recovery skipped because the "
            "initial checkpoint could not be saved"
        )
        return initial_quality, initial_context
    initial_state = _capture_stage9_candidate_state(pipeline)
    initial_record = copy.deepcopy(initial_quality)
    initial_record.setdefault("reason_codes", []).append(
        "stage9_unscreen_progressive_psf_support_retry"
    )
    remix_attempts.append(initial_record)
    candidate_summaries = [
        _stage9_psf_candidate_summary(
            initial_quality,
            recovery_pixels=0,
        )
    ]
    best_quality: Optional[Dict[str, Any]] = None
    best_context: Dict[str, Any] = initial_context
    best_checkpoint = ""
    best_state: Dict[str, Any] | None = None
    best_score = (float("inf"),) * 5
    if bool(initial_quality.get("accepted", False)):
        best_quality = initial_quality
        best_checkpoint = initial_checkpoint
        best_state = initial_state
        best_score = _stage9_psf_candidate_score(
            initial_quality,
            recovery_pixels=0,
        )
    current_quality = initial_quality

    for retry_pixels in range(1, retry_max + 1):
        if not _stage9_needs_progressive_psf_recovery(pipeline, current_quality):
            break
        recovery_starmask = _prepare_stage9_starmask_for_pixel_remix(
            pipeline,
            trusted_starmask_name,
            star_stretch_used=False,
            messages=messages,
            strict_support=False,
            support_retry_pixels=retry_pixels,
            output_name=(
                f"starmask_stretched_unscreen_psf_recovery_{retry_pixels}px"
            ),
        )
        recovery_context = (
            _prepare_stage9_unscreen_candidate(
                pipeline,
                recovery_starmask,
                messages,
            )
            if recovery_starmask
            else {"available": False}
        )
        attempt_name = f"screen_unscreen_psf_support_recovery_{retry_pixels}px"
        applied = bool(
            recovery_context.get("available", False)
            and pipeline._apply_previous_stage_star_remix(
                source_stem,
                "starmask_unscreen_stabilized",
                baseline_intensity,
            )
        )
        if not applied:
            failed_quality = {
                "attempt": attempt_name,
                "formula": "screen",
                "intensity": baseline_intensity,
                "starmask": "starmask_unscreen_stabilized",
                "decomposition_method": (
                    "matched_mtf_unscreen_chroma_stabilized"
                ),
                "psf_support_recovery_pixels": retry_pixels,
                "psf_support_recovery_target_min": target_min,
                "status": "failed",
                "accepted": False,
                "issues": ["progressive Unscreen PSF remix execution failed"],
                "metrics": {},
            }
            remix_attempts.append(failed_quality)
            candidate_summaries.append(
                _stage9_psf_candidate_summary(
                    failed_quality,
                    recovery_pixels=retry_pixels,
                )
            )
            current_quality = failed_quality
            messages.append(
                "Stage9 progressive Unscreen PSF recovery execution failed "
                f"at +{retry_pixels} px"
            )
            break

        recovery_quality = _assess_stage9_candidate(
            pipeline,
            source_stem,
            attempt=attempt_name,
            formula="screen",
        )
        recovery_quality.update(
            intensity=baseline_intensity,
            starmask="starmask_unscreen_stabilized",
            decomposition_method="matched_mtf_unscreen_chroma_stabilized",
            psf_support_recovery_pixels=retry_pixels,
            psf_support_recovery_target_min=target_min,
        )
        recovery_quality["reference_fidelity"] = _stage9_reference_fidelity(
            pipeline,
            recovery_context,
            recovery_context["unscreen_stars"],
            baseline_intensity,
            screen_baseline_state,
        )
        remix_attempts.append(copy.deepcopy(recovery_quality))
        candidate_summaries.append(
            _stage9_psf_candidate_summary(
                recovery_quality,
                recovery_pixels=retry_pixels,
            )
        )
        current_quality = recovery_quality
        ratios = _stage9_psf_group_ratios(recovery_quality)
        ratio_note = ",".join(
            f"{name}={value:.3f}" for name, value in ratios.items()
        ) or "unavailable"
        messages.append(
            "Stage9 progressive Unscreen source-wing candidate "
            f"+{retry_pixels} px: accepted="
            f"{str(bool(recovery_quality.get('accepted', False))).lower()}, "
            f"ratios={ratio_note}"
        )

        if bool(recovery_quality.get("accepted", False)):
            checkpoint = f"stage9_candidate_unscreen_psf_recovery_{retry_pixels}px"
            candidate_score = _stage9_psf_candidate_score(
                recovery_quality,
                recovery_pixels=retry_pixels,
            )
            if pipeline._save_stage_output(checkpoint):
                if candidate_score < best_score:
                    best_quality = recovery_quality
                    best_context = recovery_context
                    best_checkpoint = checkpoint
                    best_state = _capture_stage9_candidate_state(pipeline)
                    best_score = candidate_score
            elif best_quality is None:
                best_quality = recovery_quality
                best_context = recovery_context
                best_checkpoint = ""
                best_state = _capture_stage9_candidate_state(pipeline)
                best_score = candidate_score
                messages.append(
                    "Stage9 stopped progressive Unscreen PSF recovery because "
                    f"the +{retry_pixels} px checkpoint could not be saved"
                )
                break

        direction = _stage9_psf_size_direction(recovery_quality)
        if direction == "large":
            messages.append(
                "Stage9 stopped progressive Unscreen PSF recovery after an "
                f"oversized +{retry_pixels} px candidate"
            )
            break
        if (
            not bool(recovery_quality.get("accepted", False))
            and direction != "small"
        ):
            messages.append(
                "Stage9 stopped progressive Unscreen PSF recovery after a "
                f"non-size quality rejection at +{retry_pixels} px"
            )
            break

    if best_quality is None:
        _restore_stage9_candidate_state(
            pipeline,
            initial_state,
            checkpoint_stem=initial_checkpoint,
        )
        messages.append(
            "Stage9 retained the initial Unscreen candidate because no "
            "progressive source-wing candidate passed the hard quality gates"
        )
        return initial_quality, initial_context

    if best_checkpoint and best_state is not None:
        _restore_stage9_candidate_state(
            pipeline,
            best_state,
            checkpoint_stem=best_checkpoint,
        )
    selected_pixels = int(best_quality.get("psf_support_recovery_pixels", 0) or 0)
    best_quality["psf_support_recovery_target_min"] = target_min
    best_quality["psf_support_candidate_comparison"] = candidate_summaries
    best_quality["psf_support_selection_score"] = [float(value) for value in best_score]
    selected_calibration = dict(
        getattr(pipeline, "_stage9_starmask_calibration", {}) or {}
    )
    selected_calibration.update(
        psf_recovery_attempted=True,
        psf_recovery_mode="progressive_unscreen_source_confirmed_wing_support",
        psf_recovery_target_min=target_min,
        psf_recovery_pixels=selected_pixels,
        psf_recovery_candidate_comparison=candidate_summaries,
        psf_recovery_accepted=True,
    )
    pipeline._stage9_starmask_calibration = selected_calibration
    pipeline._stage9_unscreen_reference = _stage9_unscreen_context_report(
        best_context
    )
    messages.append(
        "Stage9 selected the safest Unscreen source-wing candidate closest "
        f"to the original display PSF (pixels={selected_pixels}, "
        f"target_min={target_min:.3f})"
    )
    return best_quality, best_context


def run_stage9_star_remixing(pipeline) -> None:
    """
    阶段 9: 星点处理与合成
    - 对齐工作流中的 Star Stretch / SCNR / Curves / StarComposer
    - 插件不可用时使用阶段 8 的 stage8_enhanced 作为主图，再回混非线性 starmask
    """
    stage_label = PipelineStage.STAR_REMIXING.label
    pipeline.log.stage_start(stage_label)
    pipeline._clear_stage_reviews(9)
    messages: List[str] = []
    processing_mode = str(
        getattr(pipeline.cfg, "stage9_processing_mode", "auto") or "auto"
    )
    failure_action = str(
        getattr(pipeline.cfg, "stage9_failure_action", "auto_fallback")
        or "auto_fallback"
    )
    pipeline._stage9_bypassed_bad_starless = False
    pipeline._stage9_stars_required = not bool(
        getattr(pipeline, "_star_preserve_target_bypass", False)
    )
    pipeline._stage9_stars_applied = False
    pipeline._stage9_stars_application_mode = "pending"
    pipeline._stage9_output_contains_stars = False
    pipeline._stage9_output_withheld = False
    pipeline._stage9_psf_review_required = False
    pipeline._stage9_remix_formally_accepted = False
    pipeline._stage9_star_delivery_contract_accepted = False
    pipeline._stage9_review_candidate_selected = False
    pipeline._stage9_persisted_output_validation = {
        "schema": "starun.stage9-persisted-output-validation.v1",
        "status": "not_run",
        "accepted": False,
        "reason_code": "stage9_persisted_output_validation_not_run",
    }
    pipeline._stage9_sep_crossmatch_report = (
        stage9_quality.stage9_sep_crossmatch_not_applicable(
            "Stage9 formal persisted remix route has not run"
        )
    )
    pipeline._stage9_sep_crossmatch_summary = None
    pipeline._stage9_spatial_scale_review_required = False
    pipeline._stage9_spatial_scale = {
        "schema": "starun.stage9-fwhm-spatial-scale.v1",
        "status": "not_run",
        "reason_code": "stage9_spatial_scale_not_run",
        "anchor_fwhm_px": 4.0,
    }
    pipeline._stage9_scaled_catalog_validation = {
        "schema": "starun.stage9-scaled-catalog-validation.v1",
        "status": "not_run",
        "scale_status": "not_run",
        "selected_route": None,
        "reason": "Stage9 star reference was not prepared",
    }
    pipeline._stage9_starmask_stretch_failed = False
    pipeline._stage9_starmask_preparation_failed = False
    pipeline._stage9_starmask_preparation_failure_reason = ""
    pipeline._stage9_last_star_overlay_mask = None
    pipeline._stage9_last_weak_overlay_mask = None
    pipeline._stage9_last_bright_overlay_mask = None
    pipeline._stage9_last_star_layer = None
    pipeline._stage9_candidate_overlay_masks = {}
    pipeline._stage9_starmask_support_preflight = {
        "schema": "starun.stage9-starmask-support-preflight.v2",
        "status": "not_run",
        "strategy": "adaptive_dual_route",
        "route": "unavailable",
        "planned_candidates": [],
        "skipped_candidates": [],
        "executed_candidates": [],
        "selected_support_mode": None,
    }
    pipeline._stage9_source_presence_report = {
        "schema": "starun.stage9-source-presence.v1",
        "status": "not_run",
        "available": False,
    }
    pipeline._stage9_immutable_trusted_starmask_peak = None
    pipeline._stage9_selected_remix_quality = None
    pipeline._stage9_star_reference_catalog = {
        "status": "unavailable",
        "reason": "not prepared",
    }
    pipeline._stage9_star_reference_summary = stage9_quality.star_reference_summary(
        pipeline._stage9_star_reference_catalog
    )
    pipeline._stage9_star_reference_degraded = False
    pipeline._stage9_star_reference_primary_summary = None
    pipeline._stage9_star_color_reference_samples = None
    pipeline._stage9_star_color_repair_report = {
        "schema": "starun.star-color-repair.v1",
        "status": "not_run",
        "accepted": False,
    }
    pipeline._stage9_star_color_post_validation = None
    pipeline._stage9_star_plugin_preprocessing = {
        "status": "not_run",
        "applied_steps": [],
    }
    pipeline._stage9_fallback_used = False
    pipeline._stage9_fallback_reason = None
    pipeline._stage9_candidate_recovery_used = False
    pipeline._stage9_delivery_fallback_used = False
    pipeline._stage9_minimal_fallback_active = False
    pipeline._stage9_fallback_remix_report = {
        "schema": "starun.stage9-fallback-remix.v1",
        "status": "not_attempted",
        "attempts": [],
    }
    pipeline._stage9_raw_starmask_stem = ""
    pipeline._stage9_stage8_fallback_base_stem = ""
    pipeline._stage9_remix_base_identity = None
    pipeline._stage9_star_layer_decomposition = "not_applicable"
    pipeline._stage9_matched_domain_context = None
    pipeline._stage9_unscreen_reference = _stage9_unscreen_unavailable(
        "not attempted"
    )
    review_candidate_registry: List[Dict[str, Any]] = []
    source_stem = getattr(pipeline, "_stage8_final_source", "stage8_enhanced") or "stage8_enhanced"
    pipeline._stage9_remix_base_stem = source_stem
    pipeline._stage9_upstream_source_stem = source_stem
    pipeline._stage9_stage8_fallback_base_stem = source_stem
    upstream_handoff = _stage9_upstream_handoff(pipeline, source_stem)
    upstream_passthrough = bool(upstream_handoff.get("passthrough", False))
    upstream_restricted = bool(
        upstream_handoff.get("restricted_downstream", False)
    )
    messages.append(
        "stage9_starless_source="
        f"{source_stem}; "
        f"upstream_passthrough={str(upstream_passthrough).lower()}; "
        f"upstream_restricted={str(upstream_restricted).lower()}"
    )

    def update_psf_review_requirement(
        selected_quality: Optional[Dict[str, Any]],
    ) -> bool:
        closure = (
            selected_quality.get("psf_closure") or {}
            if isinstance(selected_quality, dict)
            else {}
        )
        required = bool(
            closure.get(
                "review_required",
                (selected_quality or {}).get("review_required", False)
                if isinstance(selected_quality, dict)
                else False,
            )
        ) or bool(
            getattr(pipeline, "_stage9_spatial_scale_review_required", False)
        )
        pipeline._stage9_psf_review_required = required
        if required:
            pipeline._require_review(9, "stage9_psf_subgroup_evidence_insufficient")
            note = (
                "Stage9 PSF subgroup evidence is incomplete; accepted stars "
                "are retained but normal delivery is routed to review-only output"
            )
            if note not in messages:
                messages.append(note)
        return required

    def result_metadata() -> Dict[str, Any]:
        stage9_fallback_used = bool(
            getattr(
                pipeline,
                "_stage9_delivery_fallback_used",
                getattr(pipeline, "_stage9_fallback_used", False),
            )
        )
        stage9_fallback_reason = str(
            getattr(pipeline, "_stage9_fallback_reason", None) or ""
        )
        return {
            "fallback_used": stage9_fallback_used,
            "upstream_passthrough": upstream_passthrough,
            "reason_code": (
                "best_failed_candidate_review"
                if bool(
                    getattr(pipeline, "_stage9_review_candidate_selected", False)
                )
                else "stage9_psf_subgroup_evidence_insufficient"
                if bool(
                    getattr(pipeline, "_stage9_psf_review_required", False)
                )
                else stage9_fallback_reason
                or ("upstream_safe_passthrough" if upstream_passthrough else "")
            ),
            "details": {
                "reason_text": (
                    "使用 Stage 8 安全旁路源"
                    if upstream_passthrough
                    else ""
                ),
                "upstream_handoff": upstream_handoff,
                "stage9_fallback_used": stage9_fallback_used,
                "stage9_fallback_reason": stage9_fallback_reason or None,
                "candidate_recovery_used": bool(
                    getattr(pipeline, "_stage9_candidate_recovery_used", False)
                ),
                "delivery_fallback_used": stage9_fallback_used,
                "final_delivery_source_stem": str(
                    getattr(pipeline, "_stage9_final_source", "") or ""
                ),
                "stage9_psf_review_required": bool(
                    getattr(pipeline, "_stage9_psf_review_required", False)
                ),
                "stage9_remix_formally_accepted": bool(
                    getattr(pipeline, "_stage9_remix_formally_accepted", False)
                ),
                "stage9_review_candidate_selected": bool(
                    getattr(pipeline, "_stage9_review_candidate_selected", False)
                ),
                "processing_mode": processing_mode,
                "failure_action": failure_action,
            },
            "review_reasons": pipeline._stage_review_reasons(9),
        }

    def decisive_failure_status(
        saved: bool,
        *,
        reason: str,
        source: str,
    ) -> str:
        if failure_action != "auto_fallback":
            if hasattr(pipeline, "_record_stage_policy_event"):
                pipeline._record_stage_policy_event(
                    9,
                    event="decisive_failure",
                    reason=reason,
                    source=source,
                )
            if failure_action == "preserve_review":
                pipeline._require_review(9, str(reason or "stage9_review_required"))
        if failure_action == "stop":
            return "failed"
        return "degraded" if saved else "failed"

    if processing_mode == "preserve_with_stars":
        stage_saved, verified_source = _stage9_preserve_with_stars_review_output(
            pipeline,
            messages,
            reason="user_preserve_with_stars",
        )
        pipeline._require_review(9, "user_preserve_with_stars")
        report_mode = (
            "user_preserve_with_stars"
            if stage_saved
            else "required_stars_output_withheld"
        )
        _write_stage9_quality_report(
            pipeline,
            [],
            None,
            source_stem=str(verified_source or source_stem),
            mode=report_mode,
        )
        _append_stage9_review_bundle(
            pipeline,
            messages,
            [],
            None,
            source_stem=str(verified_source or source_stem),
            mode=report_mode,
            stage_saved=stage_saved,
        )
        messages.append(
            "Stage9 user preserve mode selected a verified with-stars review source"
            if stage_saved
            else "Stage9 user preserve mode withheld output: no verified with-stars source"
        )
        elapsed = pipeline.log.stage_end(stage_label)
        pipeline._record_stage(
            stage_label,
            "ok" if stage_saved else "failed",
            elapsed,
            "；".join(messages),
            execution="safe_passthrough" if stage_saved else "completed",
            **result_metadata(),
        )
        return

    separation_state = str(
        getattr(
            pipeline,
            "_star_separation_state",
            StarSeparationState.ACCEPTED.value,
        )
    )
    if separation_state in {
        StarSeparationState.REJECTED.value,
        StarSeparationState.TOOL_FAILED.value,
    }:
        stage_saved, verified_source = _stage9_preserve_with_stars_review_output(
            pipeline,
            messages,
            reason="star_separation_unavailable",
        )
        pipeline._stage9_stars_required = True
        pipeline._stage9_stars_applied = False
        pipeline._stage9_output_contains_stars = bool(stage_saved)
        if stage_saved:
            pipeline._stage9_stars_application_mode = (
                "not_applied_star_separation_unavailable"
            )
        else:
            pipeline._stage9_output_withheld = True
        _write_stage9_quality_report(
            pipeline,
            [],
            None,
            source_stem=str(verified_source or source_stem),
            mode=_stage9_required_stars_output_mode(stage_saved),
        )
        _append_stage9_review_bundle(
            pipeline,
            messages,
            [],
            None,
            source_stem=str(verified_source or source_stem),
            mode=_stage9_required_stars_output_mode(stage_saved),
            stage_saved=stage_saved,
        )
        messages.append(
            "star remix skipped; input already contains stars and normal delivery "
            f"is disabled (star_separation_state={separation_state})"
        )
        elapsed = pipeline.log.stage_end(stage_label)
        pipeline._record_stage(
            stage_label,
            decisive_failure_status(
                stage_saved,
                reason="star_separation_unavailable",
                source="input_contract",
            ),
            elapsed,
            "；".join(messages),
            **result_metadata(),
        )
        return

    if bool(getattr(pipeline, "_star_preserve_target_bypass", False)):
        target_bypass_review_only = bool(
            upstream_restricted
            or str(upstream_handoff.get("reason_code") or "")
            == "stage7_stretch_not_accepted_target_bypass"
        )
        if target_bypass_review_only:
            pipeline._require_review(
                9,
                "target_bypass_review_only",
                {"upstream_handoff": upstream_handoff},
            )
        try:
            pipeline.cmd_with_check("load", source_stem)
            stage_saved = pipeline._save_stage_output("stage9_remixed")
            pipeline._stage9_final_source = source_stem
            pipeline._stage9_output_contains_stars = bool(stage_saved)
            pipeline._stage9_bypassed_bad_starless = target_bypass_review_only
            pipeline._stage9_stars_application_mode = (
                "not_required_star_preserve_review"
                if target_bypass_review_only
                else "not_required_star_preserve"
            )
            report_mode = (
                "star_preserve_target_bypass_review"
                if target_bypass_review_only
                else "star_preserve_target_bypass"
            )
            _write_stage9_quality_report(
                pipeline,
                [],
                None,
                source_stem=source_stem,
                mode=report_mode,
            )
            _append_stage9_review_bundle(
                pipeline,
                messages,
                [],
                None,
                source_stem=source_stem,
                mode=report_mode,
                stage_saved=stage_saved,
            )
            messages.append(
                "star-preserve target bypassed starmask import and star remix "
                f"(source={source_stem}, "
                f"review_only={str(target_bypass_review_only).lower()})"
            )
            elapsed = pipeline.log.stage_end(stage_label)
            pipeline._record_stage(
                stage_label,
                (
                    "degraded"
                    if target_bypass_review_only or not stage_saved
                    else "skipped"
                ),
                elapsed,
                "；".join(messages),
                **result_metadata(),
            )
            return
        except (CommandError, SirilError) as error:
            if target_bypass_review_only:
                pipeline._stage9_bypassed_bad_starless = True
                pipeline._stage9_output_withheld = True
                pipeline._stage9_final_source = ""
                pipeline._stage9_stars_application_mode = (
                    "with_stars_review_source_unavailable"
                )
                messages.append(
                    "target-bypass with-stars review source could not be loaded; "
                    "output withheld: "
                    f"{pipeline._short_text(error, 160)}"
                )
                elapsed = pipeline.log.stage_end(stage_label)
                pipeline._record_stage(
                    stage_label,
                    "failed",
                    elapsed,
                    "；".join(messages),
                    **result_metadata(),
                )
                return
            pipeline._stage9_stars_required = True
            pipeline._stage9_stars_application_mode = (
                "pending_after_star_preserve_bypass_failure"
            )
            pipeline.log.warn(
                "Star-preserve Stage9 bypass failed; continuing with regular remix path: "
                f"{error}"
            )
            messages.append(
                "star-preserve Stage9 bypass failed: "
                f"{pipeline._short_text(error, 160)}"
            )
    bad_starless_reason = pipeline._stage9_bad_starless_reason()
    starless_advisories = list(
        getattr(pipeline, "_stage9_starless_advisories", []) or []
    )
    if starless_advisories:
        advisory_text = ", ".join(str(item) for item in starless_advisories)
        messages.append(
            "Stage9 accepted-stretch advisory; continuing controlled star remix: "
            f"{advisory_text}"
        )
        pipeline.log.info(
            "Stage9 continues controlled star remix because Stage7 stretched output "
            f"passed its quality gate: {advisory_text}"
        )
    if bad_starless_reason:
        pipeline.log.warn(
            "Stage9 bypasses starless remix because selected starless is unsafe: "
            f"{bad_starless_reason}"
        )
        fallback_saved, fallback_source = (
            _stage9_preserve_with_stars_review_output(
                pipeline,
                messages,
                reason=bad_starless_reason,
            )
        )
        if fallback_saved:
            _write_stage9_quality_report(
                pipeline,
                [],
                None,
                source_stem=str(fallback_source or "stage9_review_with_stars"),
                mode="with_stars_review_fallback",
            )
            _append_stage9_review_bundle(
                pipeline,
                messages,
                [],
                None,
                source_stem="stage9_review_with_stars",
                mode="with_stars_review_fallback",
                stage_saved=True,
            )
            diff_note = pipeline._stage_diff_note(
                "stage9_review_with_stars",
                str(fallback_source or ""),
            )
            if diff_note:
                messages.append(diff_note)
            elapsed = pipeline.log.stage_end(stage_label)
            pipeline._record_stage(
                stage_label,
                decisive_failure_status(
                    True,
                    reason=bad_starless_reason,
                    source="input_guard",
                ),
                elapsed,
                "；".join(messages),
                **result_metadata(),
            )
            return
        _write_stage9_quality_report(
            pipeline,
            [],
            None,
            source_stem=source_stem,
            mode="required_stars_output_withheld",
        )
        elapsed = pipeline.log.stage_end(stage_label)
        pipeline._record_stage(
            stage_label,
            decisive_failure_status(
                False,
                reason=bad_starless_reason,
                source="input_guard",
            ),
            elapsed,
            "；".join(messages),
            **result_metadata(),
        )
        return

    external_starmask = pipeline._find_external_fit(
        [
            "sasp_starmask.fit",
            "starmask_sasp.fit",
            "starmask_from_sasp.fit",
        ]
    )
    if external_starmask:
        try:
            imported = pipeline._import_external_fit(external_starmask, "starmask_external")
            if imported:
                pipeline.cmd_with_check("save", "starmask_external_raw")
                pipeline.starmask_file = pipeline.process_dir / "starmask_external_raw.fit"
                pipeline.log.info(f"已导入外部 Starmask: {external_starmask.name}")
        except (OSError, CommandError, SirilError) as e:
            pipeline.log.warn(f"导入外部 Starmask 失败，继续使用本地 starmask: {e}")

    if pipeline.starmask_file and pipeline.starmask_file.exists():
        pipeline._stage9_raw_starmask_stem = pipeline.starmask_file.stem

    star_stretch_used = False
    if pipeline.starmask_file and pipeline.starmask_file.exists():
        _prepare_stage9_star_reference(
            pipeline,
            pipeline.starmask_file.stem,
            messages,
        )
        repaired_starmask_name = _prepare_stage9_star_color_repair(
            pipeline,
            pipeline.starmask_file.stem,
            messages,
        )
        repaired_path = (
            pipeline.process_dir / f"{repaired_starmask_name}.fit"
        )
        if (
            repaired_starmask_name == "starmask_color_repaired"
            and repaired_path.exists()
        ):
            pipeline.starmask_file = repaired_path
        direct_plugin_reports = {}
        try:
            pipeline.cmd_with_check("load", pipeline.starmask_file.stem)
            direct_sasp_stretch = getattr(
                pipeline,
                "_run_sasp_star_stretch_api",
                None,
            )
            direct_nb_to_rgb = getattr(
                pipeline,
                "_run_nb_to_rgb_stars_api",
                None,
            )
            nb_to_rgb_label = None
            star_stretch_label = None
            if callable(direct_sasp_stretch) or callable(direct_nb_to_rgb):
                # 这两个 SASP 工具没有注册为 Siril CLI 命令；必须直接调用随包
                # Python API，且 NB→RGB 在单一非线性 Star Stretch 之前完成。
                if callable(direct_nb_to_rgb):
                    nb_to_rgb_label = direct_nb_to_rgb()
                    nb_to_rgb_report = getattr(
                        pipeline,
                        "_stage9_nb_to_rgb_stars_report",
                        None,
                    )
                    if isinstance(nb_to_rgb_report, dict):
                        direct_plugin_reports["nb_to_rgb_stars"] = dict(
                            nb_to_rgb_report
                        )
                    if nb_to_rgb_label:
                        pipeline.cmd_with_check("save", "starmask_nb_to_rgb")
                        pipeline.starmask_file = (
                            pipeline.process_dir / "starmask_nb_to_rgb.fit"
                        )
                        messages.append(
                            "Stage9 applied NB to RGB Stars before SASP Star Stretch"
                        )
                if callable(direct_sasp_stretch):
                    star_stretch_label = direct_sasp_stretch()
                    stretch_report = getattr(
                        pipeline,
                        "_stage9_sasp_star_stretch_report",
                        None,
                    )
                    if isinstance(stretch_report, dict):
                        direct_plugin_reports["sasp_star_stretch"] = dict(
                            stretch_report
                        )
                if not star_stretch_label and nb_to_rgb_label:
                    # The direct stretch adapter restores its in-memory input on
                    # failure. Reload the persisted NB layer as an additional
                    # deterministic guard before optional downstream operations.
                    pipeline.cmd_with_check("load", pipeline.starmask_file.stem)
            star_scnr_label = pipeline._run_first_available_command(
                "星点去紫",
                [
                    ("SASP Invert/SCNR", ("sasp_invert_scnr",)),
                    ("SCNR", ("scnr",)),
                ],
            )
            star_curves_label = pipeline._run_first_available_command(
                "星点微调",
                [
                    ("SASP Curves Editor", ("sasp_curves_editor",)),
                    ("Curves", ("curves",)),
                ],
            )
            star_stretch_used = bool(star_stretch_label)
            applied_steps = [
                {"step": step, "implementation": label}
                for step, label in (
                    ("nb_to_rgb", nb_to_rgb_label),
                    ("stretch", star_stretch_label),
                    ("scnr", star_scnr_label),
                    ("curves", star_curves_label),
                )
                if label
            ]
            if star_stretch_used:
                pipeline.cmd_with_check("save", "starmask_stretched")
                pipeline._stage9_stretched_starmask_file = (
                    pipeline.process_dir / "starmask_stretched.fit"
                )
                pipeline._stage9_star_plugin_preprocessing = {
                    "status": "plugin_stretched",
                    "applied_steps": applied_steps,
                    "output_stem": "starmask_stretched",
                    "builtin_stretch_required": False,
                    "direct_plugin_reports": direct_plugin_reports,
                }
            elif applied_steps:
                # Preserve successful SCNR/Curves work even when no plugin supplied
                # the nonlinear stretch. The validated built-in stretch will use
                # this intermediate instead of silently returning to the raw layer.
                pipeline.cmd_with_check("save", "starmask_plugin_processed")
                pipeline.starmask_file = (
                    pipeline.process_dir / "starmask_plugin_processed.fit"
                )
                pipeline._stage9_star_plugin_preprocessing = {
                    "status": "partial_plugin_processing",
                    "applied_steps": applied_steps,
                    "output_stem": "starmask_plugin_processed",
                    "builtin_stretch_required": True,
                    "direct_plugin_reports": direct_plugin_reports,
                }
                messages.append(
                    "Stage9 retained partial star-plugin processing before "
                    "validated built-in starmask stretch"
                )
            else:
                pipeline._stage9_star_plugin_preprocessing = {
                    "status": "no_plugin_applied",
                    "applied_steps": [],
                    "builtin_stretch_required": True,
                    "direct_plugin_reports": direct_plugin_reports,
                }
        except (
            AttributeError,
            CommandError,
            OSError,
            RuntimeError,
            SirilError,
            TypeError,
            ValueError,
        ) as e:
            star_stretch_used = False
            rollback_error = None
            try:
                pipeline.cmd_with_check("load", pipeline.starmask_file.stem)
            except (CommandError, SirilError) as restore_error:
                rollback_error = str(restore_error)
            pipeline._stage9_star_plugin_preprocessing = {
                "status": "failed_rolled_back",
                "applied_steps": [],
                "error": str(e),
                "rollback_error": rollback_error,
                "builtin_stretch_required": True,
                "direct_plugin_reports": direct_plugin_reports,
            }
            pipeline.log.warn(f"星点处理插件链失败，使用原始 starmask: {e}")

    # 按工作流先在 Siril 侧做 Starless 二次细化，再进行星点合成
    if upstream_restricted:
        messages.append(
            "Stage9 skipped starless secondary enhancement because the Stage8 "
            "handoff is restricted"
        )
    else:
        try:
            pipeline.cmd_with_check("load", source_stem)
            starless_refinement_steps = []
            revela_label = pipeline._run_first_available_command(
                "细节/结构增强2",
                [
                    ("VeraLux Revela", ("veralux_revela",)),
                    ("Revela", ("revela",)),
                ],
            )
            if revela_label:
                starless_refinement_steps.append(revela_label)
            if (
                pipeline.cfg.optional_color_transform_enabled
                and not bool(
                    getattr(pipeline, "_stage8_artistic_palette_applied", False)
                )
            ):
                vectra_label = pipeline._run_first_available_command(
                    "调色2（可选）",
                    [
                        ("VeraLux Vectra", ("veralux_vectra",)),
                        ("Vectra", ("vectra",)),
                    ],
                )
                if vectra_label:
                    starless_refinement_steps.append(vectra_label)
            elif bool(
                getattr(pipeline, "_stage8_artistic_palette_applied", False)
            ):
                messages.append(
                    "Stage9 optional Vectra skipped: Stage8 dual-band palette already applied"
                )
            curves_label = pipeline._run_first_available_command(
                "最终微调颜色",
                [
                    ("VeraLux Curves", ("veralux_curves",)),
                    ("Curves", ("curves",)),
                ],
            )
            if curves_label:
                starless_refinement_steps.append(curves_label)
            if starless_refinement_steps:
                if not pipeline._save_stage_output("stage9_starless_base"):
                    raise RuntimeError("Stage9 refined Starless base save failed")
                source_stem = "stage9_starless_base"
                messages.append(
                    "Stage9 Starless secondary refinement saved to an independent "
                    "base; Stage8 checkpoint retained unchanged "
                    f"(steps={','.join(starless_refinement_steps)})"
                )
            else:
                messages.append(
                    "Stage9 Starless secondary refinement unavailable; retained "
                    "the immutable Stage8 source"
                )
        except (CommandError, SirilError, RuntimeError) as e:
            upstream_source = str(
                getattr(pipeline, "_stage9_upstream_source_stem", source_stem)
                or source_stem
            )
            try:
                pipeline.cmd_with_check("load", upstream_source)
            except (CommandError, SirilError) as restore_error:
                messages.append(
                    "Stage9 Starless refinement rollback failed: "
                    f"{restore_error}"
                )
            source_stem = upstream_source
            pipeline.log.warn(
                f"Starless 二次细化失败，回滚并沿用 {source_stem}: {e}"
            )

    remix_scale = _clamp_float(
        getattr(pipeline, "_stage9_star_intensity_scale", 1.0),
        0.45,
        1.0,
    )
    if upstream_restricted:
        messages.append("Stage8 restricted source active; using controlled pixel remix")
        remix_scale = min(remix_scale, 0.95 / max(float(pipeline.cfg.star_intensity), 1e-6))
        messages.append("Stage8 restricted-source star remix intensity capped at 0.950")
    minimal_fallback_intensity = _clamp_float(
        pipeline.cfg.star_intensity * remix_scale,
        0.10,
        1.05,
    )
    messages.append(
        "Stage9 bypassed StarComposer; formal remix uses explicit "
        "starmask-top/starless-bottom Alpha+Screen composition"
    )
    composer_used = None
    remix_attempts: List[Dict[str, Any]] = []
    selected_remix: Optional[Dict[str, Any]] = None
    composer_rejected = False
    if composer_used:
        composer_quality = _assess_stage9_candidate(
            pipeline,
            source_stem,
            attempt="starcomposer",
            formula="plugin_starcomposer",
        )
        remix_attempts.append(composer_quality)
        if not bool(composer_quality.get("accepted", False)):
            composer_rejected = True
            issues = ", ".join(str(item) for item in composer_quality.get("issues", [])[:3])
            messages.append(f"Stage9 gate rejected StarComposer: {issues}")
            pipeline.log.warn(f"Stage9 gate rejected StarComposer: {issues}")
            try:
                pipeline.cmd_with_check("load", source_stem)
            except (CommandError, SirilError) as error:
                messages.append(f"Stage9 StarComposer rollback failed: {error}")
            composer_used = None
        else:
            selected_remix = composer_quality
            pipeline._stage9_selected_remix_quality = dict(composer_quality)
            _record_stage9_quality_advisories(
                pipeline,
                messages,
                composer_quality,
                label="StarComposer",
            )

    if composer_used and selected_remix is not None:
        psf_review_required = update_psf_review_requirement(selected_remix)
        remix_saved = pipeline._save_stage_output("stage9_remixed")
        composer_failure_reason = "starcomposer_save_failed"
        if remix_saved:
            persisted_validation = _validate_stage9_persisted_output(
                pipeline,
                source_stem,
                selected_remix,
            )
            remix_saved = bool(persisted_validation.get("accepted", False))
            if not remix_saved:
                composer_failure_reason = (
                    "stage9_persisted_output_validation_failed"
                )
                messages.append(
                    "Stage9 rejected persisted StarComposer output after reload; "
                    "routing to the existing with-stars review fallback"
                )
        stage_saved = remix_saved
        pipeline._stage9_stars_applied = bool(remix_saved)
        pipeline._stage9_output_contains_stars = bool(remix_saved)
        pipeline._stage9_remix_formally_accepted = bool(remix_saved)
        pipeline._stage9_stars_application_mode = (
            "starcomposer" if remix_saved else "starcomposer_save_failed"
        )
        pipeline._stage9_final_source = (
            "stage9_remixed" if remix_saved else source_stem
        )
        _update_stage9_star_delivery_contract(pipeline)
        report_source = source_stem
        report_mode = "starcomposer"
        report_selected = selected_remix
        if not remix_saved:
            fallback_saved = False
            fallback_source = None
            minimal_selected = None
            stage8_fallback_base = str(
                getattr(
                    pipeline,
                    "_stage9_stage8_fallback_base_stem",
                    source_stem,
                )
                or source_stem
            )
            if failure_action in {"auto_fallback", "preserve_review"}:
                fallback_saved, minimal_selected = (
                    _stage9_try_stage8_starmask_review_fallback(
                        pipeline,
                        messages,
                        remix_attempts,
                        trigger_reason=composer_failure_reason,
                        stage8_source_stem=stage8_fallback_base,
                        raw_starmask_stem=str(
                            getattr(
                                pipeline,
                                "_stage9_raw_starmask_stem",
                                "",
                            )
                            or ""
                        ),
                        intensity=minimal_fallback_intensity,
                    )
                )
            if fallback_saved:
                stage_saved = True
                report_source = stage8_fallback_base
                report_mode = "stage8_starmask_review_fallback"
                report_selected = minimal_selected
            else:
                fallback_saved, fallback_source = (
                    _stage9_preserve_with_stars_review_output(
                    pipeline,
                    messages,
                    reason=composer_failure_reason,
                    )
                )
                stage_saved = fallback_saved
                report_source = str(fallback_source or source_stem)
                report_mode = _stage9_required_stars_output_mode(
                    fallback_saved
                )
        _write_stage9_quality_report(
            pipeline,
            remix_attempts,
            report_selected,
            source_stem=report_source,
            mode=report_mode,
        )
        _append_stage9_review_bundle(
            pipeline,
            messages,
            remix_attempts,
            report_selected,
            source_stem=report_source,
            mode=report_mode,
            stage_saved=stage_saved,
        )
        diff_note = pipeline._stage_diff_note("stage9_remixed", "stage8_enhanced")
        if diff_note:
            messages.append(diff_note)
        stage7_diff_note = pipeline._stage_diff_note("stage9_remixed", "stage7_stretched")
        if stage7_diff_note:
            messages.append(stage7_diff_note)
        elapsed = pipeline.log.stage_end(stage_label)
        if remix_saved:
            pipeline._record_stage(
                stage_label,
                "degraded" if psf_review_required else "ok",
                elapsed,
                "；".join(messages),
                **result_metadata(),
            )
        else:
            if not stage_saved:
                messages.append("stage9 输出保存失败且无可用含星审阅源")
            pipeline._record_stage(
                stage_label,
                decisive_failure_status(
                    stage_saved,
                    reason="starcomposer_save_failed",
                    source="output_contract",
                ),
                elapsed,
                "；".join(messages),
                **result_metadata(),
            )
        return

    pipeline._stage9_remix_base_stem = source_stem
    pipeline.log.info("执行基于本轮不可变 remix base 的星点合成...")
    if not pipeline.starmask_file or not pipeline.starmask_file.exists():
        pipeline.log.warn("无星点蒙版，跳过混合阶段")
        fallback_reason = (
            "starcomposer_rejected_without_starmask"
            if composer_rejected
            else "starmask_unavailable"
        )
        stage_saved, fallback_source = _stage9_preserve_with_stars_review_output(
            pipeline,
            messages,
            reason=fallback_reason,
        )
        elapsed = pipeline.log.stage_end(stage_label)
        _write_stage9_quality_report(
            pipeline,
            remix_attempts,
            None,
            source_stem=str(fallback_source or source_stem),
            mode=_stage9_required_stars_output_mode(stage_saved),
        )
        pipeline._record_stage(
            stage_label,
            decisive_failure_status(
                stage_saved,
                reason=fallback_reason,
                source="input_contract",
            ),
            elapsed,
            "；".join(messages),
            **result_metadata(),
        )
        return

    intensity = minimal_fallback_intensity
    if remix_scale < 0.999:
        reason = getattr(pipeline, "_stage9_star_intensity_reason", "")
        if not reason:
            reason = (
                "stage8 restricted-source star intensity cap"
                if upstream_restricted
                else "stage7 residual stars"
            )
        messages.append(
            "Stage9 star remix intensity reduced from safety diagnostics "
            f"(base={pipeline.cfg.star_intensity:.3f}, effective={intensity:.3f}, "
            f"reason={reason})"
        )
    starmask_name = pipeline.starmask_file.stem
    reference_degraded = bool(
        getattr(pipeline, "_stage9_star_reference_degraded", False)
    )
    if reference_degraded and star_stretch_used:
        preprocessing = dict(
            getattr(pipeline, "_stage9_star_plugin_preprocessing", {}) or {}
        )
        preprocessing.update(
            {
                "status": "plugin_stretch_bypassed_reference_degraded",
                "plugin_output_preserved": True,
                "builtin_stretch_required": True,
            }
        )
        pipeline._stage9_star_plugin_preprocessing = preprocessing
        messages.append(
            "Stage9 bypassed the plugin-stretched layer for the degraded-reference "
            "candidate and rebuilt a validated strict built-in stretch"
        )
    support_preflight = _stage9_starmask_support_preflight(
        pipeline,
        starmask_name,
        star_stretch_used=star_stretch_used and not reference_degraded,
        failure_action=failure_action,
        messages=messages,
    )
    if reference_degraded and support_preflight.get("route") != "unavailable":
        support_preflight["route"] = "strict_only"
        support_preflight["reason_code"] = (
            "stage9_support_preflight_reference_degraded_strict"
        )
        support_preflight["planned_candidates"] = ["strict_compact"]
        support_preflight["skipped_candidates"] = [
            {
                "support_mode": "normal",
                "reason_code": (
                    "stage9_support_preflight_reference_degraded_strict"
                ),
                "status": str(
                    (
                        (support_preflight.get("candidates") or {}).get(
                            "normal"
                        )
                        or {}
                    ).get("status")
                    or "unavailable"
                ),
                "risk_level": str(
                    (
                        (support_preflight.get("candidates") or {}).get(
                            "normal"
                        )
                        or {}
                    ).get("risk_level")
                    or "unavailable"
                ),
                "reason": "degraded star reference requires strict support",
            }
        ]
        pipeline._stage9_starmask_support_preflight = (
            stage9_quality.public_starmask_support_preflight(
                support_preflight
            )
        )

    route = str(support_preflight.get("route") or "unavailable")
    targeted_recovery_enabled = bool(
        failure_action == "auto_fallback"
        and getattr(pipeline.cfg, "stage9_targeted_recovery_enabled", True)
    )
    calibrations = dict(support_preflight.get("_calibrations") or {})
    normal_summary = (
        ((support_preflight.get("candidates") or {}).get("normal") or {})
    )
    normal_stretch_source = str(
        normal_summary.get("stretch_source")
        or support_preflight.get("selected_stretch_source")
        or "builtin_calibrated"
    )
    planned_support_modes = {
        "normal_only": ("normal",),
        "strict_only": ("strict_compact",),
        "dual_competition": ("normal", "strict_compact"),
    }.get(route, ())
    strict_summary = (
        ((support_preflight.get("candidates") or {}).get("strict_compact") or {})
    )
    if (
        targeted_recovery_enabled
        and route == "normal_only"
        and bool(strict_summary.get("usable", False))
        and not bool(support_preflight.get("support_masks_equivalent", False))
    ):
        planned_support_modes = ("normal", "strict_compact")
        support_preflight["planned_candidates"] = list(planned_support_modes)
        support_preflight["targeted_competition_expanded"] = True
        support_preflight["targeted_competition_reason"] = (
            "freeze_normal_and_strict_supports_from_the_same_remix_base"
        )
    prepared_support_candidates: List[Dict[str, Any]] = []
    support_preparation_failures: List[Dict[str, Any]] = []
    for support_mode in planned_support_modes:
        strict_support = support_mode == "strict_compact"
        output_name = (
            "starmask_stretched_reference_fallback"
            if reference_degraded and strict_support
            else "starmask_stretched_compact_primary"
            if strict_support
            else "starmask_stretched"
        )
        compact_output_name = (
            "starmask_compact_reference_fallback"
            if reference_degraded and strict_support
            else "starmask_compact_primary"
            if strict_support
            else "starmask_compact"
        )
        prepared_name = _prepare_stage9_starmask_for_pixel_remix(
            pipeline,
            starmask_name,
            star_stretch_used=bool(
                support_mode == "normal"
                and star_stretch_used
                and not reference_degraded
                and normal_stretch_source == "plugin_stretched"
            ),
            messages=messages,
            strict_support=strict_support,
            output_name=output_name,
            precomputed_calibration=calibrations.get(support_mode),
            candidate_local=True,
            compact_output_name=compact_output_name,
        )
        calibration = copy.deepcopy(
            getattr(pipeline, "_stage9_starmask_calibration", {}) or {}
        )
        preparation_status = str(calibration.get("status") or "unavailable")
        prepared = bool(
            prepared_name == output_name
            and preparation_status
            not in {"failed", "rejected", "unavailable"}
        )
        preparation_record = {
            "support_mode": support_mode,
            "stretch_source": (
                normal_stretch_source
                if support_mode == "normal"
                else "builtin_calibrated"
            ),
            "output_stem": output_name,
            "status": "ready" if prepared else "failed",
            "calibration_status": preparation_status,
            "reason": str(calibration.get("reason") or ""),
            "failure_phase": calibration.get("failure_phase"),
        }
        if prepared:
            prepared_support_candidates.append(
                {
                    **preparation_record,
                    "starmask": prepared_name,
                    "calibration": calibration,
                }
            )
        else:
            support_preparation_failures.append(preparation_record)

    public_preflight = dict(
        getattr(pipeline, "_stage9_starmask_support_preflight", {}) or {}
    )
    public_preflight.setdefault("executed_candidates", [])
    public_preflight["prepared_candidates"] = [
        {
            key: value
            for key, value in candidate.items()
            if key not in {"calibration", "starmask"}
        }
        for candidate in prepared_support_candidates
    ]
    public_preflight["preparation_failures"] = support_preparation_failures
    pipeline._stage9_starmask_support_preflight = public_preflight

    if not prepared_support_candidates:
        stretch_failed = any(
            item.get("failure_phase") == "stretch_execution"
            for item in support_preparation_failures
        )
        pipeline._stage9_starmask_stretch_failed = stretch_failed
        pipeline._stage9_starmask_preparation_failed = not stretch_failed
        pipeline._stage9_starmask_preparation_failure_reason = (
            " | ".join(
                str(item.get("reason") or item.get("calibration_status"))
                for item in support_preparation_failures
            )
            or str(
                support_preflight.get("reason")
                or support_preflight.get("reason_code")
                or "support preflight rejected every candidate"
            )
        )
        failure_mode = (
            "starmask_stretch_failed"
            if stretch_failed
            else "starmask_preparation_failed"
        )
        minimal_saved = False
        minimal_selected = None
        stage8_fallback_base = str(
            getattr(
                pipeline,
                "_stage9_stage8_fallback_base_stem",
                source_stem,
            )
            or source_stem
        )
        if failure_action in {"auto_fallback", "preserve_review"}:
            minimal_saved, minimal_selected = (
                _stage9_try_stage8_starmask_review_fallback(
                    pipeline,
                    messages,
                    remix_attempts,
                    trigger_reason=failure_mode,
                    stage8_source_stem=stage8_fallback_base,
                    raw_starmask_stem=str(
                        getattr(pipeline, "_stage9_raw_starmask_stem", "")
                        or ""
                    ),
                    intensity=intensity,
                    allow_stretch=not stretch_failed,
                )
            )
        if minimal_saved:
            _write_stage9_quality_report(
                pipeline,
                remix_attempts,
                minimal_selected,
                source_stem=stage8_fallback_base,
                mode="stage8_starmask_review_fallback",
            )
            _append_stage9_review_bundle(
                pipeline,
                messages,
                remix_attempts,
                minimal_selected,
                source_stem=stage8_fallback_base,
                mode="stage8_starmask_review_fallback",
                stage_saved=True,
            )
            messages.append(
                "Stage9 starmask preparation failed, but review delivery "
                "retained the Stage8 Starless base"
            )
            elapsed = pipeline.log.stage_end(stage_label)
            pipeline._record_stage(
                stage_label,
                decisive_failure_status(
                    True,
                    reason=failure_mode,
                    source="starmask_preparation",
                ),
                elapsed,
                "；".join(messages),
                **result_metadata(),
            )
            return
        stage_saved, fallback_source = _stage9_preserve_with_stars_review_output(
            pipeline,
            messages,
            reason=failure_mode,
        )
        _write_stage9_quality_report(
            pipeline,
            remix_attempts,
            None,
            source_stem=str(fallback_source or source_stem),
            mode=_stage9_required_stars_output_mode(stage_saved),
        )
        _append_stage9_review_bundle(
            pipeline,
            messages,
            remix_attempts,
            None,
            source_stem=str(fallback_source or source_stem),
            mode=_stage9_required_stars_output_mode(stage_saved),
            stage_saved=stage_saved,
        )
        if stretch_failed:
            messages.append(
                "Stage9 did not remix the original linear starmask after an actual "
                "stretch execution failure; "
                + (
                    "used a verified with-stars review fallback"
                    if stage_saved
                    else "withheld output because no with-stars review source was available"
                )
            )
        else:
            messages.append(
                "Stage9 did not remix an unprepared raw starmask; preparation failed "
                "before stretch execution; "
                + (
                    "used a verified with-stars review fallback"
                    if stage_saved
                    else "withheld output because no with-stars review source was available"
                )
            )
        if not stage_saved:
            messages.append("stage9 输出保存失败")
        elapsed = pipeline.log.stage_end(stage_label)
        pipeline._record_stage(
            stage_label,
            decisive_failure_status(
                stage_saved,
                reason=failure_mode,
                source="starmask_preparation",
            ),
            elapsed,
            "；".join(messages),
            **result_metadata(),
        )
        return
    pipeline._stage9_star_layer_decomposition = (
        "linear_original_minus_starless_stretched"
    )
    candidates = _stage9_remix_intensity_candidates(
        pipeline,
        primary_intensity=intensity,
        remix_scale=remix_scale,
        reference_degraded=reference_degraded,
    )
    if failure_action != "auto_fallback" and len(candidates) > 1:
        candidates = candidates[:1]
        messages.append(
            "Stage9 fallback intensity ladder disabled by failure_action="
            f"{failure_action}"
        )
    messages.append(
        "Stage9 remix intensity ladder="
        + " -> ".join(f"{value:.3f}" for _, value in candidates)
    )

    primary_label, primary_intensity = candidates[0]
    primary_support_results: List[Dict[str, Any]] = []
    remix_starmask_name = str(prepared_support_candidates[0]["starmask"])
    retry_limit = max(
        0,
        min(
            2,
            int(
                getattr(
                    pipeline.cfg,
                    "stage9_psf_support_retry_pixels",
                    2,
                )
                or 0
            ),
        ),
    )

    for prepared in prepared_support_candidates:
        support_mode = str(prepared["support_mode"])
        candidate_starmask = str(prepared["starmask"])
        pipeline._stage9_starmask_calibration = copy.deepcopy(
            prepared.get("calibration") or {}
        )
        attempt_name = (
            "screen_reference_degraded_strict"
            if reference_degraded
            else "screen_compact_primary"
            if support_mode == "strict_compact"
            else "screen_primary"
        )
        applied = pipeline._apply_previous_stage_star_remix(
            source_stem,
            candidate_starmask,
            primary_intensity,
        )
        if applied:
            quality = _assess_stage9_candidate(
                pipeline,
                source_stem,
                attempt=attempt_name,
                formula="screen",
            )
        else:
            quality = {
                "attempt": attempt_name,
                "formula": "screen",
                "status": "failed",
                "accepted": False,
                "issues": ["pixel remix execution failed"],
                "metrics": {},
            }
        quality.update(
            intensity=primary_intensity,
            starmask=candidate_starmask,
            support_mode=support_mode,
            support_starmask=candidate_starmask,
            parent_attempt=None,
            base_source_stem=source_stem,
            recovery_kind="none",
            recovery_strength=0.0,
            recovery_target_groups=[],
            support_preflight_route=route,
            psf_support_recovery_pixels=0,
        )
        active_catalog = getattr(
            pipeline,
            "_stage9_star_reference_catalog",
            {},
        ) or {}
        active_radii = np.asarray(active_catalog.get("_psf_support_radii", ()))
        quality["spatial_scale_source"] = str(
            (getattr(pipeline, "_stage9_spatial_scale", {}) or {}).get(
                "source"
            )
            or "unavailable"
        )
        quality["psf_support_recovery_pixels_nominal"] = 0
        if active_radii.size:
            quality["effective_support_radius_px"] = (
                stage9_quality.stage9_effective_pixel_stats(active_radii)
            )
        _stage9_consider_review_candidate(
            pipeline,
            quality,
            attempt_order=len(remix_attempts),
            registry=review_candidate_registry,
            messages=messages,
        )
        remix_attempts.append(quality)

        if applied and targeted_recovery_enabled:
            parent_context = {
                "available": True,
                "stars": getattr(pipeline, "_stage9_last_star_layer", None),
                "support_mask": getattr(
                    pipeline,
                    "_stage9_last_star_overlay_mask",
                    None,
                ),
                "weak_mask": getattr(
                    pipeline,
                    "_stage9_last_weak_overlay_mask",
                    None,
                ),
                "bright_mask": getattr(
                    pipeline,
                    "_stage9_last_bright_overlay_mask",
                    None,
                ),
                "starmask": candidate_starmask,
                "support_mode": support_mode,
                "support_starmask": candidate_starmask,
            }
            targeted_attempt_start = len(remix_attempts)
            targeted_attempt_budget = _stage9_targeted_recovery_retry_max(
                pipeline
            )
            quality, parent_context = _stage9_targeted_psf_contraction(
                pipeline,
                source_stem=source_stem,
                parent_quality=quality,
                parent_context=parent_context,
                intensity=primary_intensity,
                support_mode=support_mode,
                messages=messages,
                remix_attempts=remix_attempts,
                review_candidate_registry=review_candidate_registry,
                retry_budget=targeted_attempt_budget,
            )
            targeted_attempt_budget = max(
                0,
                targeted_attempt_budget
                - (len(remix_attempts) - targeted_attempt_start),
            )
            quality, parent_context = _stage9_targeted_local_chroma_recovery(
                pipeline,
                source_stem=source_stem,
                parent_quality=quality,
                parent_context=parent_context,
                intensity=primary_intensity,
                support_mode=support_mode,
                messages=messages,
                remix_attempts=remix_attempts,
                review_candidate_registry=review_candidate_registry,
                retry_budget=targeted_attempt_budget,
            )
            candidate_starmask = str(
                parent_context.get("starmask")
                or quality.get("starmask")
                or candidate_starmask
            )

        if (
            applied
            and support_mode == "normal"
            and failure_action == "auto_fallback"
            and not targeted_recovery_enabled
            and retry_limit > 0
            and _stage9_needs_progressive_psf_recovery(pipeline, quality)
        ):
            recovered_quality, recovered_starmask, last_psf_quality = (
                _stage9_progressive_screen_psf_recovery(
                    pipeline,
                    source_stem=source_stem,
                    trusted_starmask_name=starmask_name,
                    baseline_starmask_name=candidate_starmask,
                    candidate_intensity=primary_intensity,
                    baseline_quality=quality,
                    messages=messages,
                    remix_attempts=remix_attempts,
                    review_candidate_registry=review_candidate_registry,
                )
            )
            if recovered_quality is not None:
                quality = recovered_quality
                quality.update(
                    support_mode="normal",
                    support_preflight_route=route,
                )
                candidate_starmask = recovered_starmask
            else:
                quality = last_psf_quality
                quality.update(
                    support_mode="normal",
                    support_preflight_route=route,
                )
            candidate_starmask = str(
                quality.get("starmask") or candidate_starmask
            )

        checkpoint = ""
        state = _capture_stage9_candidate_state(pipeline) if applied else None
        if bool(quality.get("accepted", False)) and (
            len(prepared_support_candidates) > 1 or targeted_recovery_enabled
        ):
            checkpoint = f"stage9_candidate_support_{support_mode}_primary"
            if not pipeline._save_stage_output(checkpoint):
                quality["accepted"] = False
                quality["status"] = "failed"
                quality.setdefault("issues", []).append(
                    "support candidate checkpoint save failed"
                )
                checkpoint = ""
                state = None
        score = _stage9_support_candidate_score(
            quality,
            support_mode=support_mode,
        )
        quality["support_candidate_selection_score"] = [
            None if not np.isfinite(value) else float(value)
            for value in score
        ]
        primary_support_results.append(
            {
                "support_mode": support_mode,
                "starmask": candidate_starmask,
                "calibration": copy.deepcopy(
                    getattr(pipeline, "_stage9_starmask_calibration", {}) or {}
                ),
                "quality": quality,
                "checkpoint": checkpoint,
                "state": state,
                "score": score,
                "candidate_family": "screen",
            }
        )
        public_preflight["executed_candidates"].append(
            {
                "attempt": str(quality.get("attempt") or attempt_name),
                "support_mode": support_mode,
                "status": str(quality.get("status") or "unknown"),
                "accepted": bool(quality.get("accepted", False)),
                "issues": list(quality.get("issues") or []),
            }
        )
        if not bool(quality.get("accepted", False)):
            issues = ", ".join(
                str(item) for item in quality.get("issues", [])[:3]
            )
            messages.append(
                f"Stage9 gate rejected {attempt_name}: {issues}"
            )
            pipeline.log.warn(f"Stage9 gate rejected {attempt_name}: {issues}")

    accepted_support_results = [
        item
        for item in primary_support_results
        if bool((item.get("quality") or {}).get("accepted", False))
    ]
    selected_support_result = (
        min(accepted_support_results, key=lambda item: item["score"])
        if accepted_support_results
        else None
    )
    if selected_support_result is not None:
        checkpoint = str(selected_support_result.get("checkpoint") or "")
        state = selected_support_result.get("state")
        if checkpoint and isinstance(state, dict):
            _restore_stage9_candidate_state(
                pipeline,
                state,
                checkpoint_stem=checkpoint,
            )
        selected_remix = selected_support_result["quality"]
        remix_starmask_name = str(selected_support_result["starmask"])
        selected_support_mode = str(selected_support_result["support_mode"])
        pipeline._stage9_starmask_calibration = copy.deepcopy(
            selected_support_result.get("calibration") or {}
        )
        if selected_support_mode == "strict_compact":
            reason_code = (
                "stage9_support_dual_compact_selected"
                if route == "dual_competition"
                else "stage9_support_preflight_strict_selected"
            )
            selected_remix.setdefault("reason_codes", []).append(reason_code)
        pipeline._stage9_selected_remix_quality = dict(selected_remix)
        public_preflight["selected_support_mode"] = selected_support_mode
        public_preflight["selected_attempt"] = selected_remix.get("attempt")
        _record_stage9_quality_advisories(
            pipeline,
            messages,
            selected_remix,
            label=f"{selected_support_mode} primary Screen remix",
        )
        messages.append(
            "Stage9 selected preflighted Screen support candidate "
            f"(route={route}, support={selected_support_mode}, "
            f"attempt={selected_remix.get('attempt')}, "
            f"intensity={primary_intensity:.3f})"
        )
        messages.append(
            "previous_stage_star_remix "
            f"source={source_stem}, starmask={remix_starmask_name}, "
            f"attempt={selected_remix.get('attempt')}, formula=screen, "
            f"intensity={primary_intensity:.3f}"
        )

    # Keep the reactive tighter-support path for risks that preflight could
    # not predict. Pixel compaction is now an independent preparation option.
    if (
        selected_remix is None
        and not targeted_recovery_enabled
        and route == "normal_only"
        and failure_action == "auto_fallback"
        and primary_support_results
        and _stage9_needs_compact_mask_recovery(
            primary_support_results[0]["quality"]
        )
        and bool(getattr(pipeline.cfg, "stage9_compact_starmask_enabled", True))
    ):
        initial_calibration = dict(
            primary_support_results[0].get("calibration") or {}
        )
        recovery_starmask_name = _prepare_stage9_starmask_for_pixel_remix(
            pipeline,
            starmask_name,
            star_stretch_used=False,
            messages=messages,
            strict_support=True,
            output_name="starmask_stretched_recovery",
            precomputed_calibration=calibrations.get("strict_compact"),
            candidate_local=True,
            compact_output_name="starmask_compact_recovery",
        )
        recovery_calibration = dict(
            getattr(pipeline, "_stage9_starmask_calibration", {}) or {}
        )
        recovery_applied = bool(
            recovery_starmask_name == "starmask_stretched_recovery"
            and str(recovery_calibration.get("status") or "unavailable")
            not in {"failed", "rejected", "unavailable"}
        )
        combined_calibration = dict(recovery_calibration)
        combined_calibration.update(
            recovery_attempted=True,
            recovery_applied=recovery_applied,
            recovery_compact_layer_applied=bool(
                recovery_calibration.get("compact_layer_applied", False)
            ),
            initial=initial_calibration,
        )
        pipeline._stage9_starmask_calibration = combined_calibration
        if recovery_applied:
            remix_starmask_name = recovery_starmask_name
            recovered = pipeline._apply_previous_stage_star_remix(
                source_stem,
                remix_starmask_name,
                primary_intensity,
            )
            recovery_quality = (
                _assess_stage9_candidate(
                    pipeline,
                    source_stem,
                    attempt="screen_compact_recovery",
                    formula="screen",
                )
                if recovered
                else {
                    "attempt": "screen_compact_recovery",
                    "formula": "screen",
                    "status": "failed",
                    "accepted": False,
                    "issues": ["compact-mask remix execution failed"],
                    "metrics": {},
                }
            )
            recovery_quality.update(
                intensity=primary_intensity,
                starmask=remix_starmask_name,
                support_mode="strict_compact",
                support_preflight_route=route,
            )
            _stage9_consider_review_candidate(
                pipeline,
                recovery_quality,
                attempt_order=len(remix_attempts),
                registry=review_candidate_registry,
                messages=messages,
            )
            remix_attempts.append(recovery_quality)
            public_preflight["executed_candidates"].append(
                {
                    "attempt": "screen_compact_recovery",
                    "support_mode": "strict_compact",
                    "status": str(recovery_quality.get("status") or "unknown"),
                    "accepted": bool(recovery_quality.get("accepted", False)),
                    "issues": list(recovery_quality.get("issues") or []),
                }
            )
            if bool(recovery_quality.get("accepted", False)):
                selected_remix = recovery_quality
                pipeline._stage9_selected_remix_quality = dict(recovery_quality)
                public_preflight["selected_support_mode"] = "strict_compact"
                public_preflight["selected_attempt"] = "screen_compact_recovery"
                _record_stage9_quality_advisories(
                    pipeline,
                    messages,
                    recovery_quality,
                    label="compact-mask recovery",
                )
                messages.append(
                    "Stage9 selected emergency compact-mask recovery after an "
                    "unpredicted normal-support quality rejection"
                )
            else:
                primary_support_results.append(
                    {
                        "support_mode": "strict_compact",
                        "starmask": remix_starmask_name,
                        "calibration": combined_calibration,
                        "quality": recovery_quality,
                        "checkpoint": "",
                        "state": None,
                        "score": (float("inf"),) * 8,
                    }
                )
        else:
            messages.append(
                "Stage9 compact-mask recovery unavailable after a runtime-only "
                "normal-support rejection"
            )

    pipeline._stage9_starmask_support_preflight = public_preflight

    if (
        selected_remix is None
        and failure_action == "auto_fallback"
        and not targeted_recovery_enabled
    ):
        actionable = [
            item
            for item in primary_support_results
            if _stage9_support_failure_allows_intensity_retry(item["quality"])
        ]
        fallback_support = (
            min(
                actionable,
                key=lambda item: _stage9_failed_support_candidate_score(
                    item["quality"],
                    support_mode=str(item["support_mode"]),
                ),
            )
            if actionable
            else None
        )
        if fallback_support is not None:
            remix_starmask_name = str(fallback_support["starmask"])
            fallback_support_mode = str(fallback_support["support_mode"])
            pipeline._stage9_starmask_calibration = copy.deepcopy(
                fallback_support.get("calibration") or {}
            )
            for attempt_label, candidate_intensity in candidates[1:]:
                applied = pipeline._apply_previous_stage_star_remix(
                    source_stem,
                    remix_starmask_name,
                    candidate_intensity,
                )
                attempt_name = f"screen_{attempt_label}"
                quality = (
                    _assess_stage9_candidate(
                        pipeline,
                        source_stem,
                        attempt=attempt_name,
                        formula="screen",
                    )
                    if applied
                    else {
                        "attempt": attempt_name,
                        "formula": "screen",
                        "status": "failed",
                        "accepted": False,
                        "issues": ["pixel remix execution failed"],
                        "metrics": {},
                    }
                )
                quality.update(
                    intensity=candidate_intensity,
                    starmask=remix_starmask_name,
                    support_mode=fallback_support_mode,
                    support_preflight_route=route,
                )
                _stage9_consider_review_candidate(
                    pipeline,
                    quality,
                    attempt_order=len(remix_attempts),
                    registry=review_candidate_registry,
                    messages=messages,
                )
                remix_attempts.append(quality)
                public_preflight["executed_candidates"].append(
                    {
                        "attempt": attempt_name,
                        "support_mode": fallback_support_mode,
                        "status": str(quality.get("status") or "unknown"),
                        "accepted": bool(quality.get("accepted", False)),
                        "issues": list(quality.get("issues") or []),
                    }
                )
                if bool(quality.get("accepted", False)):
                    selected_remix = quality
                    pipeline._stage9_selected_remix_quality = dict(quality)
                    public_preflight["selected_support_mode"] = fallback_support_mode
                    public_preflight["selected_attempt"] = attempt_name
                    _record_stage9_quality_advisories(
                        pipeline,
                        messages,
                        quality,
                        label=f"{attempt_label} Screen remix",
                    )
                    messages.append(
                        "Stage9 selected reduced-intensity Screen candidate "
                        f"(support={fallback_support_mode}, "
                        f"intensity={candidate_intensity:.3f})"
                    )
                    break
                if not _stage9_support_failure_allows_intensity_retry(quality):
                    messages.append(
                        "Stage9 stopped intensity fallback because the latest "
                        "rejection is structural or recovery-limited"
                    )
                    break
        else:
            messages.append(
                "Stage9 skipped intensity fallback because every primary support "
                "candidate was structural or recovery-limited"
            )
    elif selected_remix is None and failure_action != "auto_fallback":
        messages.append(
            "Stage9 stopped candidate search after decisive rejection; "
            f"failure_action={failure_action}"
        )

    pipeline._stage9_starmask_support_preflight = public_preflight

    if targeted_recovery_enabled:
        selected_remix, remix_starmask_name = (
            _stage9_targeted_unscreen_competition(
                pipeline,
                source_stem=source_stem,
                primary_support_results=primary_support_results,
                selected_screen=selected_remix,
                messages=messages,
                remix_attempts=remix_attempts,
                review_candidate_registry=review_candidate_registry,
            )
        )
        if selected_remix is None:
            selected_remix, floor_starmask = (
                _stage9_targeted_intensity_feasibility(
                    pipeline,
                    source_stem=source_stem,
                    primary_support_results=primary_support_results,
                    candidates=candidates,
                    route=route,
                    messages=messages,
                    remix_attempts=remix_attempts,
                    review_candidate_registry=review_candidate_registry,
                )
            )
            if floor_starmask:
                remix_starmask_name = floor_starmask
        if isinstance(selected_remix, dict):
            pipeline._stage9_selected_remix_quality = dict(selected_remix)
            public_preflight["selected_support_mode"] = str(
                selected_remix.get("support_mode") or ""
            )
            public_preflight["selected_attempt"] = str(
                selected_remix.get("attempt") or ""
            )
            pipeline._stage9_starmask_support_preflight = public_preflight

    # The subtraction-derived Screen candidate remains the transactional
    # baseline. A matched-domain Unscreen layer may replace it only after the
    # same quality gates and the fixed reference/non-regression comparison.
    if selected_remix is not None and not targeted_recovery_enabled:
        baseline_state = _capture_stage9_candidate_state(pipeline)
        baseline_intensity = float(selected_remix.get("intensity", intensity))
        baseline_checkpoint = "stage9_candidate_subtraction_screen"
        if pipeline._save_stage_output(baseline_checkpoint):
            unscreen_context = _prepare_stage9_unscreen_candidate(
                pipeline,
                remix_starmask_name,
                messages,
            )
            pipeline._stage9_unscreen_reference = (
                _stage9_unscreen_context_report(unscreen_context)
            )
            if bool(unscreen_context.get("available", False)):
                selected_remix["decomposition_method"] = (
                    "linear_original_minus_starless_stretched"
                )
                selected_remix["reference_fidelity"] = (
                    _stage9_reference_fidelity(
                        pipeline,
                        unscreen_context,
                        baseline_state["star_layer"],
                        baseline_intensity,
                        baseline_state,
                    )
                )
                baseline_attempt = str(
                    selected_remix.get("attempt") or "screen_primary"
                )
                baseline_suffix = (
                    baseline_attempt[len("screen_") :]
                    if baseline_attempt.startswith("screen_")
                    else baseline_attempt
                )
                unscreen_attempt = f"screen_unscreen_{baseline_suffix}"
                applied = pipeline._apply_previous_stage_star_remix(
                    source_stem,
                    "starmask_unscreen_stabilized",
                    baseline_intensity,
                )
                if applied:
                    unscreen_quality = _assess_stage9_candidate(
                        pipeline,
                        source_stem,
                        attempt=unscreen_attempt,
                        formula="screen",
                    )
                    unscreen_quality.update(
                        intensity=baseline_intensity,
                        starmask="starmask_unscreen_stabilized",
                        decomposition_method=(
                            "matched_mtf_unscreen_chroma_stabilized"
                        ),
                    )
                    unscreen_quality["reference_fidelity"] = (
                        _stage9_reference_fidelity(
                            pipeline,
                            unscreen_context,
                            unscreen_context["unscreen_stars"],
                            baseline_intensity,
                            baseline_state,
                        )
                    )
                    unscreen_quality, unscreen_context = (
                        _stage9_progressive_unscreen_psf_recovery(
                            pipeline,
                            source_stem=source_stem,
                            trusted_starmask_name=starmask_name,
                            baseline_intensity=baseline_intensity,
                            screen_baseline_state=baseline_state,
                            initial_context=unscreen_context,
                            initial_quality=unscreen_quality,
                            messages=messages,
                            remix_attempts=remix_attempts,
                        )
                    )
                    unscreen_quality, unscreen_context = (
                        _stage9_extend_rescue_with_source_presence(
                            pipeline,
                            source_stem=source_stem,
                            accepted_context=unscreen_context,
                            accepted_quality=unscreen_quality,
                            intensity=baseline_intensity,
                            messages=messages,
                            remix_attempts=remix_attempts,
                        )
                    )
                    unscreen_quality, unscreen_context = (
                        _stage9_extend_with_selective_size(
                            pipeline,
                            source_stem=source_stem,
                            accepted_context=unscreen_context,
                            accepted_quality=unscreen_quality,
                            intensity=baseline_intensity,
                            messages=messages,
                            remix_attempts=remix_attempts,
                        )
                    )
                    if str(unscreen_quality.get("attempt") or "").startswith(
                        (
                            "screen_unscreen_source_presence",
                            "screen_unscreen_selective_size",
                        )
                    ):
                        pipeline._stage9_unscreen_reference = (
                            _stage9_unscreen_context_report(unscreen_context)
                        )
                    comparison = stage9_quality.compare_unscreen_candidate(
                        selected_remix,
                        unscreen_quality,
                        pipeline.cfg,
                    )
                    if (
                        str(unscreen_quality.get("attempt") or "").startswith(
                            (
                                "screen_unscreen_source_presence",
                                "screen_unscreen_selective_size",
                            )
                        )
                        and bool(unscreen_quality.get("accepted", False))
                    ):
                        # Source-presence/selective-size candidates extend an
                        # accepted, materially-better Unscreen candidate. Their
                        # purpose is completeness/visible PSF support, not a
                        # second MAE race against the subtraction baseline.
                        comparison = {
                            **comparison,
                            "selected": True,
                            "reason_code": (
                                "stage9_unscreen_selective_size_selected"
                                if str(
                                    unscreen_quality.get("attempt") or ""
                                ).startswith("screen_unscreen_selective_size")
                                else "stage9_unscreen_source_presence_selected"
                            ),
                            "selection_basis": (
                                "accepted_unscreen_extension_with_source_presence_"
                                "or_selective_sub_halfmax_wings_and_all_existing_"
                                "quality_gates"
                            ),
                        }
                    unscreen_quality["comparison_to_baseline"] = comparison
                    unscreen_quality.setdefault("reason_codes", []).append(
                        comparison["reason_code"]
                    )
                    matching_attempt_index = next(
                        (
                            index
                            for index, prior_attempt in enumerate(remix_attempts)
                            if prior_attempt.get("attempt")
                            == unscreen_quality.get("attempt")
                        ),
                        None,
                    )
                    if matching_attempt_index is None:
                        remix_attempts.append(unscreen_quality)
                    else:
                        remix_attempts[matching_attempt_index] = copy.deepcopy(
                            unscreen_quality
                        )
                    pipeline._save_stage_output("stage9_candidate_unscreen")
                    pipeline._stage9_unscreen_reference.update(
                        selection_status=(
                            "selected" if comparison.get("selected") else "retained_baseline"
                        ),
                        reason_code=comparison["reason_code"],
                        comparison=comparison,
                    )
                    if bool(comparison.get("selected", False)):
                        selected_remix = unscreen_quality
                        pipeline._stage9_selected_remix_quality = dict(
                            unscreen_quality
                        )
                        pipeline._stage9_star_layer_decomposition = (
                            str(
                                selected_remix.get("decomposition_method")
                                or "matched_mtf_unscreen_chroma_stabilized"
                            )
                        )
                        _record_stage9_quality_advisories(
                            pipeline,
                            messages,
                            unscreen_quality,
                            label="matched-domain Unscreen remix",
                        )
                        messages.append(
                            "Stage9 selected matched-domain chroma-stable Unscreen "
                            f"amplitude candidate (intensity={baseline_intensity:.3f}, "
                            f"relative_mae_improvement={float(comparison.get('relative_improvement', 0.0)):.3f}, "
                            f"absolute_mae_improvement={float(comparison.get('absolute_improvement', 0.0)):.4f})"
                        )
                    else:
                        try:
                            _restore_stage9_candidate_state(
                                pipeline,
                                baseline_state,
                                checkpoint_stem=baseline_checkpoint,
                            )
                        except (CommandError, RuntimeError, SirilError) as restore_error:
                            messages.append(
                                "Stage9 baseline transaction restore failed: "
                                f"{restore_error}"
                            )
                            selected_remix = None
                            pipeline._stage9_unscreen_reference.update(
                                selection_status="transaction_restore_failed",
                                reason_code="stage9_unscreen_candidate_rejected",
                                restore_error=str(restore_error),
                            )
                        if selected_remix is not None:
                            pipeline._stage9_selected_remix_quality = dict(
                                selected_remix
                            )
                            messages.append(
                                "Stage9 retained subtraction-derived Screen baseline; "
                                f"Unscreen reason={comparison['reason_code']}"
                            )
                else:
                    failed_unscreen = {
                        "attempt": unscreen_attempt,
                        "formula": "screen",
                        "intensity": baseline_intensity,
                        "decomposition_method": (
                            "matched_mtf_unscreen_chroma_stabilized"
                        ),
                        "status": "failed",
                        "accepted": False,
                        "issues": ["Unscreen pixel remix execution failed"],
                        "metrics": {},
                        "reason_codes": ["stage9_unscreen_candidate_rejected"],
                    }
                    remix_attempts.append(failed_unscreen)
                    pipeline._stage9_unscreen_reference.update(
                        selection_status="retained_baseline",
                        reason_code="stage9_unscreen_candidate_rejected",
                    )
                    try:
                        _restore_stage9_candidate_state(
                            pipeline,
                            baseline_state,
                            checkpoint_stem=baseline_checkpoint,
                        )
                    except (CommandError, RuntimeError, SirilError) as restore_error:
                        messages.append(
                            "Stage9 baseline transaction restore failed: "
                            f"{restore_error}"
                        )
                        selected_remix = None
                        pipeline._stage9_unscreen_reference.update(
                            selection_status="transaction_restore_failed",
                            restore_error=str(restore_error),
                        )
            else:
                try:
                    _restore_stage9_candidate_state(
                        pipeline,
                        baseline_state,
                        checkpoint_stem=baseline_checkpoint,
                    )
                except (CommandError, RuntimeError, SirilError) as restore_error:
                    messages.append(
                        "Stage9 baseline restore after unavailable Unscreen failed: "
                        f"{restore_error}"
                    )
                    selected_remix = None
                    pipeline._stage9_unscreen_reference.update(
                        selection_status="transaction_restore_failed",
                        reason_code="stage9_unscreen_candidate_rejected",
                        restore_error=str(restore_error),
                    )
        else:
            pipeline._stage9_unscreen_reference = _stage9_unscreen_unavailable(
                "transactional baseline checkpoint save failed"
            )
            messages.append(
                "Stage9 skipped Unscreen competition because the transactional "
                "baseline checkpoint could not be saved"
            )
    elif failure_action == "auto_fallback" and not targeted_recovery_enabled:
        # A trustworthy Unscreen layer may make a final formal attempt when all
        # subtraction-derived candidates were rejected. It still traverses the
        # complete existing quality gate and intensity ladder.
        unscreen_context = _prepare_stage9_unscreen_candidate(
            pipeline,
            remix_starmask_name,
            messages,
        )
        pipeline._stage9_unscreen_reference = _stage9_unscreen_context_report(
            unscreen_context
        )
        if bool(unscreen_context.get("available", False)):
            rescue_state = _capture_stage9_candidate_state(pipeline)
            for attempt_label, candidate_intensity in candidates:
                applied = pipeline._apply_previous_stage_star_remix(
                    source_stem,
                    "starmask_unscreen_stabilized",
                    candidate_intensity,
                )
                attempt_name = f"screen_unscreen_{attempt_label}"
                if not applied:
                    remix_attempts.append(
                        {
                            "attempt": attempt_name,
                            "formula": "screen",
                            "intensity": candidate_intensity,
                            "decomposition_method": (
                                "matched_mtf_unscreen_chroma_stabilized"
                            ),
                            "status": "failed",
                            "accepted": False,
                            "issues": ["Unscreen rescue execution failed"],
                            "metrics": {},
                            "reason_codes": [
                                "stage9_unscreen_candidate_rejected"
                            ],
                        }
                    )
                    continue
                quality = _assess_stage9_candidate(
                    pipeline,
                    source_stem,
                    attempt=attempt_name,
                    formula="screen",
                )
                quality.update(
                    intensity=candidate_intensity,
                    starmask="starmask_unscreen_stabilized",
                    decomposition_method=(
                        "matched_mtf_unscreen_chroma_stabilized"
                    ),
                )
                quality["reference_fidelity"] = _stage9_reference_fidelity(
                    pipeline,
                    unscreen_context,
                    unscreen_context["unscreen_stars"],
                    candidate_intensity,
                    rescue_state,
                )
                quality.setdefault("reason_codes", []).append(
                    "stage9_unscreen_selected"
                    if bool(quality.get("accepted", False))
                    else "stage9_unscreen_candidate_rejected"
                )
                _stage9_consider_review_candidate(
                    pipeline,
                    quality,
                    attempt_order=len(remix_attempts),
                    registry=review_candidate_registry,
                    messages=messages,
                )
                remix_attempts.append(quality)
                pipeline._save_stage_output(
                    f"stage9_candidate_unscreen_{attempt_label}"
                )
                if bool(quality.get("accepted", False)):
                    quality, unscreen_context = (
                        _stage9_extend_rescue_with_source_presence(
                            pipeline,
                            source_stem=source_stem,
                            accepted_context=unscreen_context,
                            accepted_quality=quality,
                            intensity=candidate_intensity,
                            messages=messages,
                            remix_attempts=remix_attempts,
                        )
                    )
                    quality, unscreen_context = (
                        _stage9_extend_with_selective_size(
                            pipeline,
                            source_stem=source_stem,
                            accepted_context=unscreen_context,
                            accepted_quality=quality,
                            intensity=candidate_intensity,
                            messages=messages,
                            remix_attempts=remix_attempts,
                        )
                    )
                    selected_remix = quality
                    pipeline._stage9_selected_remix_quality = dict(quality)
                    pipeline._stage9_star_layer_decomposition = (
                        str(
                            quality.get("decomposition_method")
                            or "matched_mtf_unscreen_chroma_stabilized"
                        )
                    )
                    pipeline._stage9_unscreen_reference.update(
                        selection_status="selected_rescue",
                        reason_code=(
                            "stage9_unscreen_source_presence_selected"
                            if str(quality.get("attempt") or "").startswith(
                                (
                                    "screen_unscreen_source_presence",
                                    "screen_unscreen_selective_size",
                                )
                            )
                            else "stage9_unscreen_selected"
                        ),
                        rescue_without_baseline=True,
                    )
                    _record_stage9_quality_advisories(
                        pipeline,
                        messages,
                        quality,
                        label=f"{attempt_label} Unscreen rescue",
                    )
                    messages.append(
                        "Stage9 selected chroma-stable Unscreen as the final "
                        "formal rescue candidate after baseline rejection "
                        f"(intensity={candidate_intensity:.3f})"
                    )
                    break
                try:
                    pipeline.cmd_with_check("load", source_stem)
                except (CommandError, SirilError) as rollback_error:
                    messages.append(
                        "Stage9 Unscreen rescue rollback failed: "
                        f"{rollback_error}"
                    )
            if selected_remix is None:
                pipeline._stage9_unscreen_reference.update(
                    selection_status="rejected",
                    reason_code="stage9_unscreen_candidate_rejected",
                )
    else:
        pipeline._stage9_unscreen_reference = _stage9_unscreen_unavailable(
            "Unscreen rescue disabled by the active Stage9 failure action",
            reason_code="stage9_unscreen_candidate_rejected",
        )

    if (
        selected_remix is None
        and failure_action == "auto_fallback"
        and review_candidate_registry
    ):
        best_review = min(
            review_candidate_registry,
            key=lambda item: item["score"],
        )
        review_quality = best_review["quality"]
        review_attempt = str(review_quality.get("attempt") or "unknown")
        review_eligibility = review_quality.get("review_eligibility") or {}
        review_eligibility["selection_attempted"] = True
        review_eligibility["restore_status"] = "pending"
        review_restored = False
        try:
            _restore_stage9_candidate_state(
                pipeline,
                best_review["state"],
                checkpoint_stem=str(best_review["checkpoint"]),
            )
            review_restored = True
            review_eligibility["restore_status"] = "restored"
        except (CommandError, RuntimeError, SirilError) as error:
            review_eligibility["restore_status"] = "failed"
            review_eligibility["restore_error"] = str(error)
            messages.append(
                "Stage9 bounded review candidate restore failed; "
                f"trying the minimal Stage8 + starmask fallback: {error}"
            )

        if review_restored:
            review_saved = False
            canonical_saved = False
            try:
                review_saved = bool(
                    pipeline._save_stage_output(
                        "stage9_review_with_stars"
                    )
                )
            except (CommandError, RuntimeError, SirilError) as error:
                review_eligibility["final_save_error"] = str(error)
            if review_saved:
                try:
                    canonical_saved = bool(
                        pipeline._save_stage_output("stage9_remixed")
                    )
                except (CommandError, RuntimeError, SirilError) as error:
                    review_eligibility["canonical_save_error"] = str(error)
            review_eligibility["final_save_status"] = (
                "saved"
                if review_saved and canonical_saved
                else "failed"
            )
            review_eligibility["review_output_saved"] = bool(review_saved)
            review_eligibility["canonical_output_saved"] = bool(
                canonical_saved
            )
            if review_saved and canonical_saved:
                review_eligibility["selected"] = True
                review_quality["selection_class"] = "review_candidate"
                review_quality["review_selected"] = True
                review_quality.setdefault("reason_codes", []).append(
                    "STAGE9_BEST_FAILED_CANDIDATE_REVIEW"
                )
                pipeline._stage9_selected_remix_quality = dict(review_quality)
                pipeline._stage9_star_layer_decomposition = str(
                    review_quality.get("decomposition_method")
                    or getattr(
                        pipeline,
                        "_stage9_star_layer_decomposition",
                        "linear_subtraction",
                    )
                    or "linear_subtraction"
                )
                pipeline._stage9_stars_applied = True
                pipeline._stage9_output_contains_stars = True
                pipeline._stage9_output_withheld = False
                pipeline._stage9_remix_formally_accepted = False
                pipeline._stage9_review_candidate_selected = True
                pipeline._stage9_psf_review_required = True
                pipeline._require_review(9, "best_failed_candidate_review")
                pipeline._stage9_stars_application_mode = (
                    "screen_review_candidate"
                )
                pipeline._stage9_final_source = "stage9_review_with_stars"
                if isinstance(pipeline._stage9_unscreen_reference, dict):
                    pipeline._stage9_unscreen_reference.update(
                        selection_status="selected_review_candidate",
                        reason_code="stage9_best_failed_candidate_review",
                    )
                _write_stage9_quality_report(
                    pipeline,
                    remix_attempts,
                    review_quality,
                    source_stem=source_stem,
                    mode="best_failed_review_candidate",
                )
                _append_stage9_review_bundle(
                    pipeline,
                    messages,
                    remix_attempts,
                    review_quality,
                    source_stem=source_stem,
                    mode="best_failed_review_candidate",
                    stage_saved=True,
                )
                messages.append(
                    "Stage9 selected the best bounded formal-failure candidate "
                    "for review-only delivery "
                    f"(attempt={review_attempt}, intensity="
                    f"{float(review_quality.get('intensity', 0.0) or 0.0):.3f}, "
                    f"review_fwhm_max="
                    f"{_stage9_review_fwhm_ratio_max(pipeline):.3f})"
                )
                elapsed = pipeline.log.stage_end(stage_label)
                pipeline._record_stage(
                    stage_label,
                    "degraded",
                    elapsed,
                    "；".join(messages),
                    **result_metadata(),
                )
                return
            messages.append(
                "Stage9 bounded review candidate final save failed; "
                "trying the minimal Stage8 + starmask fallback"
            )

    if selected_remix is None:
        fallback_reason = (
            "best_failed_candidate_restore_or_save_failed"
            if review_candidate_registry
            else "all_remix_candidates_outside_review_range"
        )
        minimal_saved = False
        minimal_selected = None
        stage8_fallback_base = str(
            getattr(
                pipeline,
                "_stage9_stage8_fallback_base_stem",
                source_stem,
            )
            or source_stem
        )
        if failure_action in {"auto_fallback", "preserve_review"}:
            minimal_saved, minimal_selected = (
                _stage9_try_stage8_starmask_review_fallback(
                    pipeline,
                    messages,
                    remix_attempts,
                    trigger_reason=fallback_reason,
                    stage8_source_stem=stage8_fallback_base,
                    raw_starmask_stem=str(
                        getattr(pipeline, "_stage9_raw_starmask_stem", "")
                        or ""
                    ),
                    intensity=intensity,
                )
            )
        if minimal_saved:
            _write_stage9_quality_report(
                pipeline,
                remix_attempts,
                minimal_selected,
                source_stem=stage8_fallback_base,
                mode="stage8_starmask_review_fallback",
            )
            _append_stage9_review_bundle(
                pipeline,
                messages,
                remix_attempts,
                minimal_selected,
                source_stem=stage8_fallback_base,
                mode="stage8_starmask_review_fallback",
                stage_saved=True,
            )
            elapsed = pipeline.log.stage_end(stage_label)
            messages.append(
                "Stage9 gate rejected all formal remix candidates; retained "
                "Stage8 + minimal starmask as a review-only output"
            )
            pipeline._record_stage(
                stage_label,
                decisive_failure_status(
                    True,
                    reason="all_remix_candidates_rejected",
                    source="quality_gate",
                ),
                elapsed,
                "；".join(messages),
                **result_metadata(),
            )
            return
        stage_saved, fallback_source = _stage9_preserve_with_stars_review_output(
            pipeline,
            messages,
            reason=fallback_reason,
            prefer_stage5=True,
        )
        fallback_mode = (
            "stage5_review_fallback"
            if stage_saved and fallback_source == "stage5_linear"
            else _stage9_required_stars_output_mode(stage_saved)
        )
        _write_stage9_quality_report(
            pipeline,
            remix_attempts,
            None,
            source_stem=str(fallback_source or source_stem),
            mode=fallback_mode,
        )
        _append_stage9_review_bundle(
            pipeline,
            messages,
            remix_attempts,
            None,
            source_stem=str(fallback_source or source_stem),
            mode=fallback_mode,
            stage_saved=stage_saved,
        )
        elapsed = pipeline.log.stage_end(stage_label)
        messages.append(
            "Stage9 gate rejected all remix candidates; "
            + (
                "used a verified with-stars review fallback"
                if stage_saved
                else "withheld output because no with-stars review source was available"
            )
        )
        if not stage_saved:
            messages.append("stage9 输出保存失败")
        pipeline._record_stage(
            stage_label,
            decisive_failure_status(
                stage_saved,
                reason="all_remix_candidates_rejected",
                source="quality_gate",
            ),
            elapsed,
            "；".join(messages),
            **result_metadata(),
        )
        return

    psf_review_required = update_psf_review_requirement(selected_remix)
    remix_saved = pipeline._save_stage_output("stage9_remixed")
    screen_failure_reason = "screen_save_failed"
    if remix_saved:
        persisted_validation = _validate_stage9_persisted_output(
            pipeline,
            source_stem,
            selected_remix,
        )
        remix_saved = bool(persisted_validation.get("accepted", False))
        if not remix_saved:
            screen_failure_reason = (
                "stage9_persisted_output_validation_failed"
            )
            messages.append(
                "Stage9 rejected stage9_remixed after persisted reload quality "
                "and catalog-star validation; using the with-stars review path"
            )
    stage_saved = remix_saved
    pipeline._stage9_stars_applied = bool(remix_saved)
    pipeline._stage9_output_contains_stars = bool(remix_saved)
    pipeline._stage9_remix_formally_accepted = bool(remix_saved)
    pipeline._stage9_stars_application_mode = (
        "screen" if remix_saved else "screen_save_failed"
    )
    pipeline._stage9_final_source = (
        "stage9_remixed" if remix_saved else source_stem
    )
    _update_stage9_star_delivery_contract(pipeline)
    report_source = source_stem
    report_mode = "screen"
    report_selected = selected_remix
    if not remix_saved:
        fallback_saved = False
        fallback_source = None
        minimal_selected = None
        stage8_fallback_base = str(
            getattr(
                pipeline,
                "_stage9_stage8_fallback_base_stem",
                source_stem,
            )
            or source_stem
        )
        if failure_action in {"auto_fallback", "preserve_review"}:
            fallback_saved, minimal_selected = (
                _stage9_try_stage8_starmask_review_fallback(
                    pipeline,
                    messages,
                    remix_attempts,
                    trigger_reason=screen_failure_reason,
                    stage8_source_stem=stage8_fallback_base,
                    raw_starmask_stem=str(
                        getattr(pipeline, "_stage9_raw_starmask_stem", "")
                        or ""
                    ),
                    intensity=intensity,
                )
            )
        if fallback_saved:
            stage_saved = True
            report_source = stage8_fallback_base
            report_mode = "stage8_starmask_review_fallback"
            report_selected = minimal_selected
        else:
            fallback_saved, fallback_source = (
                _stage9_preserve_with_stars_review_output(
                    pipeline,
                    messages,
                    reason=screen_failure_reason,
                )
            )
            stage_saved = fallback_saved
            report_source = str(fallback_source or source_stem)
            report_mode = _stage9_required_stars_output_mode(fallback_saved)
    _write_stage9_quality_report(
        pipeline,
        remix_attempts,
        report_selected,
        source_stem=report_source,
        mode=report_mode,
    )
    _append_stage9_review_bundle(
        pipeline,
        messages,
        remix_attempts,
        report_selected,
        source_stem=report_source,
        mode=report_mode,
        stage_saved=stage_saved,
    )
    diff_note = pipeline._stage_diff_note("stage9_remixed", "stage8_enhanced")
    if diff_note:
        messages.append(diff_note)
    stage7_diff_note = pipeline._stage_diff_note("stage9_remixed", "stage7_stretched")
    if stage7_diff_note:
        messages.append(stage7_diff_note)

    elapsed = pipeline.log.stage_end(stage_label)
    if remix_saved:
        pipeline._record_stage(
            stage_label,
            "degraded" if psf_review_required else "ok",
            elapsed,
            "；".join(messages),
            **result_metadata(),
        )
    else:
        if not stage_saved:
            messages.append("stage9 输出保存失败且无可用含星审阅源")
        pipeline._record_stage(
            stage_label,
            decisive_failure_status(
                stage_saved,
                reason="screen_save_failed",
                source="output_contract",
            ),
            elapsed,
            "；".join(messages),
            **result_metadata(),
        )
