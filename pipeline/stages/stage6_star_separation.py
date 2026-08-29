"""Star separation and star-mask preparation."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from models import PipelineStage, StarSeparationState
from pipeline_safety import should_bypass_star_separation
from sirilpy.exceptions import CommandError, SirilError
import scene_support
import stage5_handoff
import stage7_repair
import stage7_quality
import star_halo_guard
import syqon_starless


SYQON_SCRIPT_CANDIDATES = (
    "SyQon/Starless.py",
)


def _stage6_galaxy_roi_diagnostics(
    derived: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return the additive Stage 6/7 audit view of galaxy ROI fitting."""
    values = derived if isinstance(derived, dict) else {}
    return {
        "status": str(values.get("galaxy_roi_status") or "not_run"),
        "available": bool(
            float(values.get("galaxy_roi_available", 0.0) or 0.0) > 0.5
        ),
        "failure_reason": str(
            values.get("galaxy_roi_failure_reason") or ""
        ),
        "seed_position_px": {
            "x": values.get("galaxy_roi_seed_x_px"),
            "y": values.get("galaxy_roi_seed_y_px"),
        },
        "signal_floor": values.get("galaxy_roi_signal_floor"),
        "signal_floor_components": dict(
            values.get("galaxy_roi_signal_floor_components") or {}
        ),
        "raw_covariance_scale_px": {
            "major_sigma": values.get(
                "galaxy_roi_raw_covariance_major_sigma"
            ),
            "minor_sigma": values.get(
                "galaxy_roi_raw_covariance_minor_sigma"
            ),
            "minimum_extent": values.get(
                "galaxy_roi_minimum_covariance_extent"
            ),
        },
        "star_clip_percentile": values.get(
            "galaxy_roi_star_clip_percentile"
        ),
        "star_clip_value": values.get("galaxy_roi_star_clip_value"),
    }


def _stage6_repair_acceptance(
    *,
    score_before: float,
    score_after: float,
    configured_max_score_growth: Any,
    non_regression_passed: bool,
    trigger_improved: bool,
    chroma_improved: bool,
) -> Dict[str, Any]:
    """Evaluate and fully audit the transactional Stage 6 repair gate."""
    try:
        configured_limit = float(configured_max_score_growth)
    except (TypeError, ValueError):
        configured_limit = 0.0
    effective_limit = min(0.20, max(0.0, configured_limit))
    actual_growth = float(score_after) - float(score_before)
    score_gate_passed = actual_growth <= effective_limit + 1e-12
    improvement_gate_passed = bool(trigger_improved or chroma_improved)
    accepted = (
        score_gate_passed
        and bool(non_regression_passed)
        and improvement_gate_passed
    )
    return {
        "accepted": accepted,
        "score_before": float(score_before),
        "score_after": float(score_after),
        "score_growth": actual_growth,
        "score_growth_configured": configured_limit,
        "score_growth_max": effective_limit,
        "score_growth_limit_clamped": configured_limit != effective_limit,
        "score_gate_passed": score_gate_passed,
        "non_regression_gate_passed": bool(non_regression_passed),
        "improvement_gate_passed": improvement_gate_passed,
        "gate_conclusion": "accepted" if accepted else "rejected",
    }


def _stage7_chroma_repair_acceptance(
    cfg,
    before: Dict[str, Any],
    after: Dict[str, Any],
    *,
    residual_not_worse: bool,
    halo_not_worse: bool,
) -> Dict[str, Any]:
    before_chroma = max(float(before.get("chroma_noise_score", 0.0) or 0.0), 0.0)
    after_chroma = max(float(after.get("chroma_noise_score", 0.0) or 0.0), 0.0)
    chroma_delta = max(before_chroma - after_chroma, 0.0)
    chroma_reduction = chroma_delta / max(before_chroma, 1e-7)
    min_reduction = float(
        getattr(cfg, "stage7_starless_repair_chroma_reduction_min", 0.20)
    )
    min_delta = float(
        getattr(cfg, "stage7_starless_repair_chroma_delta_min", 0.0005)
    )
    significant = (
        before_chroma > 0.0
        and chroma_delta >= min_delta
        and chroma_reduction >= min_reduction
    )
    accepted = significant and residual_not_worse and halo_not_worse
    return {
        "accepted": accepted,
        "significant": significant,
        "before": before_chroma,
        "after": after_chroma,
        "delta": chroma_delta,
        "reduction_ratio": chroma_reduction,
        "minimum_delta": min_delta,
        "minimum_reduction_ratio": min_reduction,
        "residual_not_worse": residual_not_worse,
        "halo_not_worse": halo_not_worse,
    }


_REPAIR_UPPER_BOUND_METRICS = (
    "residual_star_score",
    "global_residual_star_score",
    "compact_residual_star_score",
    "halo_residue_score",
    "global_halo_residue_score",
    "compact_halo_residue_score",
    "galaxy_disk_halo_residue_score",
    "black_hole_score",
)
_REPAIR_LOWER_BOUND_METRICS = (
    "starless_dynamic_range_ratio",
    "galaxy_core_preservation_ratio",
    "galaxy_core_contrast_ratio",
)


def _stage7_repair_non_regression(
    before_quality: Dict[str, Any],
    after_quality: Dict[str, Any],
    *,
    tolerance: float = 0.002,
) -> Dict[str, Any]:
    """Require every protected quality dimension to remain non-worse."""
    before = before_quality.get("derived") or {}
    after = after_quality.get("derived") or {}
    before = before if isinstance(before, dict) else {}
    after = after if isinstance(after, dict) else {}
    comparisons: Dict[str, Dict[str, Any]] = {}
    violations: List[str] = []

    for name in _REPAIR_UPPER_BOUND_METRICS:
        if name not in before and name not in after:
            continue
        if name not in before or name not in after:
            comparisons[name] = {
                "direction": "upper",
                "passed": False,
                "reason": "metric unavailable in one side of same-domain retest",
            }
            violations.append(name)
            continue
        before_value = float(before.get(name, 0.0) or 0.0)
        after_value = float(after.get(name, 0.0) or 0.0)
        passed = after_value <= before_value + tolerance
        comparisons[name] = {
            "direction": "upper",
            "before": before_value,
            "after": after_value,
            "tolerance": tolerance,
            "passed": passed,
        }
        if not passed:
            violations.append(name)

    for name in _REPAIR_LOWER_BOUND_METRICS:
        if name not in before and name not in after:
            continue
        if name not in before or name not in after:
            comparisons[name] = {
                "direction": "lower",
                "passed": False,
                "reason": "metric unavailable in one side of same-domain retest",
            }
            violations.append(name)
            continue
        before_value = float(before.get(name, 0.0) or 0.0)
        after_value = float(after.get(name, 0.0) or 0.0)
        passed = after_value + tolerance >= before_value
        comparisons[name] = {
            "direction": "lower",
            "before": before_value,
            "after": after_value,
            "tolerance": tolerance,
            "passed": passed,
        }
        if not passed:
            violations.append(name)

    status_rank = {
        "ok": 0,
        "advisory": 1,
        "poor": 2,
        "rejected": 3,
        "failed": 3,
    }
    before_status = str(before_quality.get("status") or "failed").lower()
    after_status = str(after_quality.get("status") or "failed").lower()
    status_not_worse = status_rank.get(after_status, 3) <= status_rank.get(
        before_status,
        3,
    )
    if not status_not_worse:
        violations.append("quality_status")

    gate_comparisons: Dict[str, Dict[str, Any]] = {}
    before_gates = before_quality.get("quality_gates") or {}
    after_gates = after_quality.get("quality_gates") or {}
    if isinstance(before_gates, dict) and isinstance(after_gates, dict):
        for name in sorted(set(before_gates) | set(after_gates)):
            before_gate = before_gates.get(name)
            after_gate = after_gates.get(name)
            if not isinstance(before_gate, dict) or not isinstance(after_gate, dict):
                gate_comparisons[name] = {
                    "passed": False,
                    "reason": "gate unavailable in one side of same-domain retest",
                }
                violations.append(f"gate:{name}")
                continue
            before_gate_status = str(before_gate.get("status") or "hard_failed").lower()
            after_gate_status = str(after_gate.get("status") or "hard_failed").lower()
            gate_rank = {"ok": 0, "advisory": 1, "borderline": 1, "hard_failed": 2}
            passed = gate_rank.get(after_gate_status, 2) <= gate_rank.get(
                before_gate_status,
                2,
            )
            gate_comparisons[name] = {
                "before": before_gate_status,
                "after": after_gate_status,
                "passed": passed,
            }
            if not passed:
                violations.append(f"gate:{name}")

    new_hard_failures = []
    for name, after_value in after.items():
        if not str(name).endswith("_hard_failed") or not bool(after_value):
            continue
        if not bool(before.get(name, False)):
            new_hard_failures.append(str(name))
    violations.extend(new_hard_failures)
    return {
        "accepted": not violations,
        "tolerance": tolerance,
        "status_before": before_status,
        "status_after": after_status,
        "status_not_worse": status_not_worse,
        "comparisons": comparisons,
        "gate_comparisons": gate_comparisons,
        "new_hard_failures": new_hard_failures,
        "violations": list(dict.fromkeys(violations)),
    }


def _stage7_trigger_improvement(
    before_quality: Dict[str, Any],
    after_quality: Dict[str, Any],
    triggers: List[str],
    *,
    bright_nebula_halo_advisory: bool = False,
    minimum_delta: float = 0.005,
) -> Dict[str, Any]:
    """Require at least one actual trigger metric to improve materially."""
    before = before_quality.get("derived") or {}
    after = after_quality.get("derived") or {}
    before = before if isinstance(before, dict) else {}
    after = after if isinstance(after, dict) else {}
    trigger_metrics = {
        "residual_stars": (("residual_star_score", "upper"),),
        "compact_residual_stars": (("compact_residual_star_score", "upper"),),
        "halo_residue": (
            ("halo_residue_score", "upper"),
            ("global_halo_residue_score", "upper"),
        ),
        "compact_halo_residue": (("compact_halo_residue_score", "upper"),),
        "black_hole": (("black_hole_score", "upper"),),
        "dynamic_range_collapse": (("starless_dynamic_range_ratio", "lower"),),
        "galaxy_core_damage": (
            ("galaxy_core_preservation_ratio", "lower"),
            ("galaxy_core_contrast_ratio", "lower"),
        ),
    }
    selected = list(triggers)
    if bright_nebula_halo_advisory:
        selected.extend(("halo_residue", "compact_halo_residue"))
    comparisons: Dict[str, Dict[str, Any]] = {}
    improved = False
    for trigger in dict.fromkeys(selected):
        for name, direction in trigger_metrics.get(trigger, ()):
            if name not in before or name not in after:
                continue
            before_value = float(before.get(name, 0.0) or 0.0)
            after_value = float(after.get(name, 0.0) or 0.0)
            delta = (
                before_value - after_value
                if direction == "upper"
                else after_value - before_value
            )
            metric_improved = delta >= minimum_delta
            comparisons[name] = {
                "trigger": trigger,
                "direction": direction,
                "before": before_value,
                "after": after_value,
                "improvement": delta,
                "minimum_delta": minimum_delta,
                "improved": metric_improved,
            }
            improved = improved or metric_improved
    return {
        "accepted": improved,
        "minimum_delta": minimum_delta,
        "triggers": list(dict.fromkeys(selected)),
        "comparisons": comparisons,
    }


def _stage7_starless_pixel_repair_trigger(
    pipeline,
    quality: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Decide whether the accepted Stage 6 starless still needs pixel repair."""
    if not isinstance(quality, dict):
        return {
            "triggered": False,
            "reason": "quality_unavailable",
        }

    status = str(quality.get("status", "") or "").strip().lower()
    derived = quality.get("derived") or {}
    if not isinstance(derived, dict):
        derived = {}

    def _metric(name: str) -> float:
        try:
            return max(float(derived.get(name, 0.0) or 0.0), 0.0)
        except (TypeError, ValueError):
            return 0.0

    halo_score = _metric("halo_residue_score")
    global_halo_score = _metric("global_halo_residue_score")
    compact_halo_score = _metric("compact_halo_residue_score")
    galaxy_disk_halo_score = _metric("galaxy_disk_halo_residue_score")
    measured_halo = max(
        halo_score,
        global_halo_score,
        compact_halo_score,
        galaxy_disk_halo_score,
    )
    base_limit = float(pipeline.cfg.stage7_halo_residue_score_max)
    target_limit = float(pipeline._stage7_effective_halo_threshold())
    target_type = str(pipeline._active_target_type() or "")

    reason = ""
    if status != "ok":
        reason = f"quality_status={status or 'unknown'}"
    elif (
        target_type == "bright_emission_reflection_nebula"
        and measured_halo > base_limit
    ):
        # Bright nebulae have a relaxed acceptance limit so real nebulosity is
        # not rejected as halo. The same relaxed limit previously prevented the
        # transactional halo repair from running, even though Stage 8 treated
        # the base-limit exceedance as an enhancement advisory.
        reason = "bright_nebula_halo_advisory"

    return {
        "triggered": bool(reason),
        "reason": reason,
        "quality_status": status or "unknown",
        "target_type": target_type,
        "halo_residue_score": halo_score,
        "global_halo_residue_score": global_halo_score,
        "compact_halo_residue_score": compact_halo_score,
        "galaxy_disk_halo_residue_score": galaxy_disk_halo_score,
        "galaxy_core_preservation_ratio": _metric(
            "galaxy_core_preservation_ratio"
        ),
        "galaxy_core_contrast_ratio": _metric(
            "galaxy_core_contrast_ratio"
        ),
        "measured_halo_score": measured_halo,
        "base_halo_limit": base_limit,
        "target_halo_limit": target_limit,
        "within_target_limit": measured_halo <= target_limit,
    }


def _stage6_quality_hard_failure_summary(
    pipeline,
    quality: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Recompute frozen hard gates so status/metric inconsistencies fail safely."""
    quality = quality if isinstance(quality, dict) else {}
    derived = quality.get("derived") or {}
    derived = derived if isinstance(derived, dict) else {}

    def metric(name: str) -> float:
        try:
            return max(float(derived.get(name, 0.0) or 0.0), 0.0)
        except (TypeError, ValueError):
            return 0.0

    combined_halo = metric("halo_residue_score")
    global_halo = (
        metric("global_halo_residue_score")
        if "global_halo_residue_score" in derived
        else combined_halo
    )
    compact_halo = metric("compact_halo_residue_score")
    galaxy_disk_halo = metric("galaxy_disk_halo_residue_score")
    single_local_galaxy_halo_override = (
        stage7_quality.stage7_single_local_galaxy_halo_override_active(
            pipeline,
            quality,
        )
    )
    if single_local_galaxy_halo_override:
        effective_halo = max(global_halo, compact_halo)
    else:
        effective_halo = max(
            combined_halo,
            global_halo,
            compact_halo,
            galaxy_disk_halo,
        )
    gates = {
        "residual": stage7_quality.stage7_upper_quality_gate(
            pipeline.cfg,
            value=metric("residual_star_score"),
            accepted_limit=float(pipeline.cfg.stage7_residual_star_score_max),
        ),
        "noise_gain": stage7_quality.stage7_upper_quality_gate(
            pipeline.cfg,
            value=metric("starless_noise_gain"),
            accepted_limit=float(pipeline.cfg.stage7_starless_noise_gain_max),
        ),
        "halo": stage7_quality.stage7_upper_quality_gate(
            pipeline.cfg,
            value=effective_halo,
            accepted_limit=float(pipeline._stage7_effective_halo_threshold()),
        ),
    }
    bright_core_integrity = quality.get("bright_core_integrity")
    destructive_core_failure = bool(
        isinstance(bright_core_integrity, dict)
        and bright_core_integrity.get("applicable", False)
        and bright_core_integrity.get("hard_failed", False)
    )
    if isinstance(bright_core_integrity, dict) and bool(
        bright_core_integrity.get("applicable", False)
    ):
        gates["bright_core_integrity"] = {
            "status": bright_core_integrity.get("status", "hard_failed"),
            "hard_failed": destructive_core_failure,
            "advisory": bool(bright_core_integrity.get("advisory", False)),
            "fixed_limit": True,
            "trigger_reasons": list(
                bright_core_integrity.get("trigger_reasons") or []
            ),
        }
    hard_metrics = [
        name for name, gate in gates.items() if bool(gate.get("hard_failed"))
    ]
    status = str(quality.get("status") or "failed").strip().lower()
    triggers = stage7_quality.stage7_repair_triggers(pipeline, quality)
    codes = _syqon_quality_failure_codes(triggers)
    if "residual" in hard_metrics and "RESIDUAL_GLOBAL" not in codes:
        codes.append("RESIDUAL_GLOBAL")
    if "halo" in hard_metrics and "HALO" not in codes:
        codes.append("HALO")
    if destructive_core_failure and "BRIGHT_CORE_INTEGRITY" not in codes:
        codes.append("BRIGHT_CORE_INTEGRITY")
    return {
        "hard_failed": status != "ok" or bool(hard_metrics),
        "destructive_core_failure": destructive_core_failure,
        "quality_status": status,
        "hard_metrics": hard_metrics,
        "failure_codes": list(dict.fromkeys(codes)),
        "gates": gates,
    }


def _stage8_handoff_from_stage6(
    pipeline,
    quality: Optional[Dict[str, Any]],
    pixel_repairs: List[Dict[str, Any]],
    *,
    separation_accepted: bool,
) -> Dict[str, Any]:
    """Build the typed Stage 6 -> Stage 8 processing decision."""
    quality = quality if isinstance(quality, dict) else {}
    derived = quality.get("derived") or {}
    derived = derived if isinstance(derived, dict) else {}

    def metric(name: str) -> float:
        try:
            return max(float(derived.get(name, 0.0) or 0.0), 0.0)
        except (TypeError, ValueError):
            return 0.0

    combined_halo = metric("halo_residue_score")
    global_halo = (
        metric("global_halo_residue_score")
        if "global_halo_residue_score" in derived
        else combined_halo
    )
    compact_halo = metric("compact_halo_residue_score")
    galaxy_disk_halo = metric("galaxy_disk_halo_residue_score")
    single_local_galaxy_halo_override = (
        stage7_quality.stage7_single_local_galaxy_halo_override_active(
            pipeline,
            quality,
        )
    )
    if single_local_galaxy_halo_override:
        effective_halo = max(global_halo, compact_halo)
    else:
        effective_halo = max(
            combined_halo,
            global_halo,
            compact_halo,
            galaxy_disk_halo,
        )
    residual_score = metric("residual_star_score")
    noise_gain = metric("starless_noise_gain")
    base_limit = float(pipeline.cfg.stage7_halo_residue_score_max)
    accepted_limit = float(pipeline._stage7_effective_halo_threshold())
    target_type = str(pipeline._active_target_type() or "")
    quality_status = str(quality.get("status") or "unknown").strip().lower()
    quality_hard_failed_retained = bool(
        getattr(pipeline, "_stage6_quality_hard_failed_retained", False)
    )
    raw_quality_advisories = [
        str(item).strip()
        for item in (quality.get("advisories") or [])
        if str(item).strip()
    ]
    suppressed_quality_advisories: List[str] = []
    quality_advisories = raw_quality_advisories
    if single_local_galaxy_halo_override:
        suppressed_quality_advisories = [
            item
            for item in raw_quality_advisories
            if item.startswith("galaxy_disk_halo_residue ")
        ]
        quality_advisories = [
            item
            for item in raw_quality_advisories
            if item not in suppressed_quality_advisories
        ]
    cleanup_borderline = bool(
        derived.get("starmask_cleanup_borderline", False)
    )
    accepted_repair = next(
        (
            item
            for item in reversed(pixel_repairs)
            if isinstance(item, dict) and bool(item.get("accepted"))
        ),
        None,
    )
    repair_trigger = (
        accepted_repair.get("trigger") or {}
        if isinstance(accepted_repair, dict)
        else {}
    )
    repair_trigger = repair_trigger if isinstance(repair_trigger, dict) else {}

    def trigger_metric(name: str) -> float:
        try:
            return max(float(repair_trigger.get(name, 0.0) or 0.0), 0.0)
        except (TypeError, ValueError):
            return 0.0

    trigger_global_halo = trigger_metric("halo_residue_score")
    trigger_compact_halo = trigger_metric("compact_halo_residue_score")
    trigger_effective_halo = max(trigger_global_halo, trigger_compact_halo)
    bright_advisory_triggered = (
        str(repair_trigger.get("reason") or "")
        == "bright_nebula_halo_advisory"
        and trigger_effective_halo > base_limit
    )
    reasons: List[Dict[str, Any]] = []
    policy = "full"
    residual_gate = stage7_quality.stage7_upper_quality_gate(
        pipeline.cfg,
        value=residual_score,
        accepted_limit=float(pipeline.cfg.stage7_residual_star_score_max),
    )
    noise_gate = stage7_quality.stage7_upper_quality_gate(
        pipeline.cfg,
        value=noise_gain,
        accepted_limit=float(pipeline.cfg.stage7_starless_noise_gain_max),
    )
    halo_gate = stage7_quality.stage7_upper_quality_gate(
        pipeline.cfg,
        value=effective_halo,
        accepted_limit=accepted_limit,
    )
    gate_advisories: List[str] = []
    if residual_gate["advisory"]:
        gate_advisories.append(
            f"residual_stars {residual_score:.3f}>"
            f"{float(pipeline.cfg.stage7_residual_star_score_max):.3f}"
        )
    if noise_gate["advisory"]:
        gate_advisories.append(
            f"starless_noise_gain {noise_gain:.3f}>"
            f"{float(pipeline.cfg.stage7_starless_noise_gain_max):.3f}"
        )
    if halo_gate["advisory"]:
        gate_advisories.append(
            f"halo_residue {effective_halo:.3f}>{accepted_limit:.3f}"
        )
    quality_advisories = list(
        dict.fromkeys([*quality_advisories, *gate_advisories])
    )

    def add_reason(code: str, **values: Any) -> None:
        reasons.append({"code": code, "source_stage": 6, **values})

    if not separation_accepted:
        policy = "skip"
        failure_codes = list(
            getattr(pipeline, "_stage6_quality_failure_codes", []) or []
        )
        definite_failure_codes = [
            str(code)
            for code in failure_codes
            if str(code).upper() in _STAGE6_DEFINITE_QUALITY_REJECTION_CODES
        ]
        if definite_failure_codes:
            if "SUBJECT_CHROMA_COLLAPSE" in definite_failure_codes:
                add_reason(
                    "stage6_subject_chroma_collapse",
                    quality_status=quality_status,
                    failure_codes=definite_failure_codes,
                )
            elif "SUBJECT_CHROMA_LINEAGE_UNVERIFIED" in definite_failure_codes:
                add_reason(
                    "stage6_subject_chroma_lineage_unverified",
                    quality_status=quality_status,
                    failure_codes=definite_failure_codes,
                )
            else:
                add_reason(
                    "star_separation_quality_rejected",
                    quality_status=quality_status,
                    failure_codes=definite_failure_codes,
                )
        else:
            add_reason(
                "star_separation_unavailable",
                quality_status=quality_status,
                failure_codes=failure_codes,
            )
    elif (
        quality_hard_failed_retained
        or residual_gate["hard_failed"]
        or noise_gate["hard_failed"]
        or halo_gate["hard_failed"]
    ):
        policy = "limited"
        hard_metrics = [
            name
            for name, gate in (
                ("residual", residual_gate),
                ("noise_gain", noise_gate),
                ("halo", halo_gate),
            )
            if gate["hard_failed"]
        ]
        add_reason(
            "stage6_quality_hard_failed_retained",
            quality_status=quality_status,
            hard_metrics=hard_metrics,
            failure_codes=list(
                getattr(pipeline, "_stage6_quality_failure_codes", []) or []
            ),
            star_intensity_cap=0.70,
        )
    elif quality_status != "ok":
        policy = "skip"
        add_reason(
            "stage6_starless_quality_not_ok",
            quality_status=quality_status,
        )
    elif cleanup_borderline:
        policy = "limited"
        add_reason(
            "starmask_diffuse_residual_borderline",
            value=metric("starmask_diffuse_residual_ratio"),
            accepted_limit=metric("starmask_diffuse_residual_ratio_max"),
            uncertainty_abs=metric("starmask_diffuse_uncertainty_abs"),
            effective_hard_limit=metric(
                "starmask_diffuse_effective_hard_limit"
            ),
        )
    elif (
        target_type == "bright_emission_reflection_nebula"
        and (effective_halo > base_limit or bright_advisory_triggered)
    ):
        policy = "limited"
        advisory_global = (
            trigger_global_halo if bright_advisory_triggered else global_halo
        )
        advisory_compact = (
            trigger_compact_halo if bright_advisory_triggered else compact_halo
        )
        advisory_effective = max(advisory_global, advisory_compact)
        advisory_value = (
            advisory_global
            if advisory_global > base_limit
            else advisory_compact
        )
        add_reason(
            "bright_nebula_halo_advisory",
            metric=(
                "halo_residue_score"
                if advisory_global > base_limit
                else "compact_halo_residue_score"
            ),
            value=advisory_value,
            effective_value=advisory_effective,
            global_value=advisory_global,
            compact_value=advisory_compact,
            post_repair_global_value=global_halo,
            post_repair_compact_value=compact_halo,
            base_limit=base_limit,
            accepted_limit=accepted_limit,
            within_accepted_limit=advisory_effective <= accepted_limit,
        )
    elif quality_advisories:
        policy = "limited"
        add_reason(
            "stage6_quality_advisory",
            advisories=quality_advisories,
            advisory_multiplier=stage7_quality.stage7_quality_advisory_multiplier(
                pipeline.cfg
            ),
        )
    elif accepted_repair is not None:
        policy = "limited"
        add_reason(
            "stage6_starless_pixel_repair_accepted",
            acceptance_path=accepted_repair.get("acceptance_path"),
            global_value=global_halo,
            compact_value=compact_halo,
            base_limit=base_limit,
            accepted_limit=accepted_limit,
        )

    primary_reason = reasons[0] if reasons else {}
    reason_code = str(primary_reason.get("code") or "")
    reason_text = ""
    if reason_code == "bright_nebula_halo_advisory":
        reason_text = (
            "bright_nebula_halo_advisory: "
            f"{float(primary_reason['value']):.3f} > {base_limit:.3f}, "
            f"accepted_limit={accepted_limit:.3f}"
        )
    elif reason_code:
        if reason_code == "stage6_quality_advisory":
            reason_text = (
                "stage6_quality_advisory: "
                + ", ".join(str(item) for item in primary_reason.get("advisories", [])[:3])
            )
        else:
            reason_text = reason_code

    handoff_metrics: Dict[str, Any] = {
        "halo_residue_score": combined_halo,
        "combined_halo_residue_score": combined_halo,
        "global_halo_residue_score": global_halo,
        "compact_halo_residue_score": compact_halo,
        "galaxy_disk_halo_residue_score": galaxy_disk_halo,
        "effective_halo_residue_score": effective_halo,
        "base_halo_limit": base_limit,
        "accepted_halo_limit": accepted_limit,
        "quality_advisory_multiplier": (
            stage7_quality.stage7_quality_advisory_multiplier(pipeline.cfg)
        ),
        "residual_star_hard_limit": float(residual_gate["hard_limit"]),
        "starless_noise_gain_hard_limit": float(noise_gate["hard_limit"]),
        "halo_residue_hard_limit": float(halo_gate["hard_limit"]),
        "residual_star_score": residual_score,
        "starless_noise_gain": noise_gain,
        "starmask_diffuse_residual_ratio": metric(
            "starmask_diffuse_residual_ratio"
        ),
        "starmask_diffuse_residual_ratio_max": metric(
            "starmask_diffuse_residual_ratio_max"
        ),
        "starmask_diffuse_uncertainty_abs": metric(
            "starmask_diffuse_uncertainty_abs"
        ),
        "starmask_cleanup_borderline": cleanup_borderline,
        "galaxy_roi_available": metric("galaxy_roi_available"),
        "galaxy_core_preservation_ratio": metric(
            "galaxy_core_preservation_ratio"
        ),
        "galaxy_core_contrast_ratio": metric(
            "galaxy_core_contrast_ratio"
        ),
    }
    if bright_advisory_triggered:
        handoff_metrics.update(
            {
                "trigger_halo_residue_score": trigger_global_halo,
                "trigger_compact_halo_residue_score": trigger_compact_halo,
                "trigger_effective_halo_residue_score": trigger_effective_halo,
            }
        )

    return {
        "schema": "starun.stage8-handoff.v2",
        "requested_policy": policy,
        "processing_policy": policy,
        "source_stage": 6,
        "source_stem": None,
        "attempt_id": getattr(pipeline, "_selected_syqon_attempt_id", None),
        "pair_id": getattr(pipeline, "_selected_syqon_pair_id", None),
        "passthrough": False,
        "restricted_downstream": policy != "full",
        "reason_code": reason_code,
        "reason_text": reason_text,
        "reasons": reasons,
        "quality_status": quality_status,
        "advisories": raw_quality_advisories,
        "suppressed_advisories": suppressed_quality_advisories,
        "metrics": handoff_metrics,
        "repair": {
            "attempted": bool(pixel_repairs),
            "accepted": accepted_repair is not None,
            "acceptance_path": (
                accepted_repair.get("acceptance_path")
                if accepted_repair is not None
                else None
            ),
        },
    }


def _apply_starmask_cleanup_hard_gate(
    quality: Dict[str, Any],
    cleanup: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Carry the diffuse-residual hard gate into Stage 6 candidate quality."""
    if not isinstance(quality, dict):
        return quality
    cleanup_metrics = (
        cleanup.get("metrics") or {}
        if isinstance(cleanup, dict)
        else {}
    )
    hard_failed = bool(cleanup_metrics.get("diffuse_hard_gate_failed", False))
    borderline = bool(cleanup_metrics.get("diffuse_borderline", False))
    cleanup_advisories = [
        str(item).strip()
        for item in (cleanup_metrics.get("advisories") or [])
        if str(item).strip()
    ]
    derived = dict(quality.get("derived") or {})
    limits = cleanup_metrics.get("limits") or {}
    advisory_multiplier = float(
        limits.get("advisory_multiplier", 2.0) or 2.0
    )
    accepted_limit = float(
        limits.get("max_diffuse_residual_ratio", 0.08) or 0.08
    )
    effective_hard_limit = float(
        limits.get(
            "effective_diffuse_hard_limit",
            accepted_limit * advisory_multiplier,
        )
        or accepted_limit * advisory_multiplier
    )
    derived.update(
        {
            "starmask_diffuse_residual_ratio": float(
                cleanup_metrics.get("diffuse_residual_ratio", 0.0) or 0.0
            ),
            "starmask_diffuse_residual_ratio_max": accepted_limit,
            "starmask_diffuse_uncertainty_abs": float(
                limits.get("diffuse_uncertainty_abs", 0.0005) or 0.0
            ),
            "starmask_diffuse_advisory_multiplier": advisory_multiplier,
            "starmask_diffuse_effective_hard_limit": effective_hard_limit,
            "starmask_cleanup_borderline": borderline,
            "starmask_cleanup_hard_failed": hard_failed,
            "starmask_cleanup_advisories": cleanup_advisories,
        }
    )
    quality["derived"] = derived
    if hard_failed:
        issue = (
            "starmask_diffuse_residual_ratio "
            f"{derived['starmask_diffuse_residual_ratio']:.3f}>"
            f"{derived['starmask_diffuse_residual_ratio_max']:.3f}"
        )
        issues = list(quality.get("issues") or [])
        if issue not in issues:
            issues.append(issue)
        quality["issues"] = issues
        quality["status"] = "poor"
    else:
        if borderline and not cleanup_advisories:
            cleanup_advisories.append(
                "starmask_diffuse_residual_ratio "
                f"{derived['starmask_diffuse_residual_ratio']:.3f}>"
                f"{derived['starmask_diffuse_residual_ratio_max']:.3f}"
            )
        advisories = list(quality.get("advisories") or [])
        for advisory in cleanup_advisories:
            if advisory not in advisories:
                advisories.append(advisory)
        quality["advisories"] = advisories
        local_advisories = list(quality.get("local_advisories") or [])
        for advisory in cleanup_advisories:
            if advisory not in local_advisories:
                local_advisories.append(advisory)
        quality["local_advisories"] = local_advisories
    return quality


_QUALITY_FAILURE_CODES = {
    "residual_stars": "RESIDUAL_GLOBAL",
    "compact_residual_stars": "RESIDUAL_COMPACT",
    "halo_residue": "HALO",
    "compact_halo_residue": "HALO",
    "black_hole": "BLACK_HOLE",
    "dynamic_range_collapse": "DYNAMIC_RANGE_COLLAPSE",
    "galaxy_core_damage": "GALAXY_CORE_DAMAGE",
    "bright_core_integrity": "BRIGHT_CORE_INTEGRITY",
    "starmask_coverage": "STARMASK_COVERAGE",
    "starmask_coverage_unavailable": "STARMASK_COVERAGE_UNAVAILABLE",
    "subject_chroma_collapse": "SUBJECT_CHROMA_COLLAPSE",
    "subject_chroma_lineage_unverified": "SUBJECT_CHROMA_LINEAGE_UNVERIFIED",
}

_STAGE6_DEFINITE_QUALITY_REJECTION_CODES = frozenset(
    {
        "HALO",
        "RESIDUAL_GLOBAL",
        "RESIDUAL_COMPACT",
        "BLACK_HOLE",
        "DYNAMIC_RANGE_COLLAPSE",
        "GALAXY_CORE_DAMAGE",
        "BRIGHT_CORE_INTEGRITY",
        "STARMASK_COVERAGE",
        "SUBJECT_CHROMA_COLLAPSE",
        "SUBJECT_CHROMA_LINEAGE_UNVERIFIED",
        "SCRIPT_CONTRACT",
        "TILE_ARTIFACT",
    }
)
_STAGE6_RETAINABLE_MEASUREMENT_UNCERTAINTY_CODES = frozenset(
    {"STARMASK_COVERAGE_UNAVAILABLE"}
)


def _syqon_quality_failure_codes(repair_triggers: List[str]) -> List[str]:
    """Normalize Stage6 quality symptoms without authorizing parameter search."""
    codes = [
        _QUALITY_FAILURE_CODES.get(str(trigger), "SCRIPT_CONTRACT")
        for trigger in repair_triggers
    ]
    return list(dict.fromkeys(codes))


def _stage7_quality_selection_key(pipeline, quality: Optional[Dict[str, Any]]) -> Tuple[int, float]:
    """Prefer a hard-gate-passing candidate before comparing soft quality scores."""
    status = str((quality or {}).get("status") or "").strip().lower()
    return (0 if status == "ok" else 1, pipeline._stage7_quality_score(quality))


def _stage6_subject_chroma_failure_code(
    quality: Optional[Dict[str, Any]],
) -> Optional[str]:
    lineage = (
        quality.get("subject_chroma_lineage")
        if isinstance(quality, dict)
        else None
    )
    if not isinstance(lineage, dict) or not bool(lineage.get("hard_failed", False)):
        return None
    reason_code = str(lineage.get("reason_code") or "")
    if reason_code == "stage6_subject_chroma_collapse":
        return "SUBJECT_CHROMA_COLLAPSE"
    return "SUBJECT_CHROMA_LINEAGE_UNVERIFIED"


def _stage6_chroma_retry_passed(quality: Optional[Dict[str, Any]]) -> bool:
    lineage = (
        quality.get("subject_chroma_lineage")
        if isinstance(quality, dict)
        else None
    )
    return bool(
        isinstance(quality, dict)
        and str(quality.get("status") or "").lower() == "ok"
        and isinstance(lineage, dict)
        and lineage.get("accepted") is True
        and not lineage.get("hard_failed", True)
    )


def _stage6_chroma_retry_plan(
    quality: Optional[Dict[str, Any]],
    *,
    retry_max: int,
    syqon_available: bool,
    failure_action: str,
) -> Dict[str, Any]:
    failure_code = _stage6_subject_chroma_failure_code(quality)
    configured_retry_max = max(int(retry_max), 0)
    eligible = failure_code == "SUBJECT_CHROMA_COLLAPSE"
    should_attempt = bool(
        eligible
        and configured_retry_max >= 1
        and syqon_available
        and str(failure_action) == "auto_fallback"
    )
    if failure_code is None:
        status = "not_triggered"
    elif should_attempt:
        status = "ready"
    elif failure_code == "SUBJECT_CHROMA_LINEAGE_UNVERIFIED":
        status = "direct_reject_unverified_lineage"
    elif configured_retry_max < 1:
        status = "blocked_by_retry_limit"
    elif not syqon_available:
        status = "blocked_backend_unavailable"
    else:
        status = "blocked_by_failure_action"
    return {
        "triggered": failure_code is not None,
        "failure_code": failure_code,
        "eligible": eligible,
        "should_attempt": should_attempt,
        "status": status,
        "configured_retry_max": configured_retry_max,
        "attempt_limit": 1 if should_attempt else 0,
    }
def _stage6_can_retain_hard_failed_pair(
    final_quality_failure: Dict[str, Any],
    *,
    pair_valid: bool,
    bright_core_retry_terminal_failure: bool = False,
) -> bool:
    """Retain only a contract-valid pair with pure measurement uncertainty."""
    failure_codes = {
        str(code).strip().upper()
        for code in (final_quality_failure.get("failure_codes") or [])
        if str(code).strip()
    }
    return bool(
        final_quality_failure.get("hard_failed", False)
        and pair_valid
        and not final_quality_failure.get("destructive_core_failure", False)
        and not bright_core_retry_terminal_failure
        and failure_codes
        and failure_codes.issubset(
            _STAGE6_RETAINABLE_MEASUREMENT_UNCERTAINTY_CODES
        )
    )


def _stage6_exchange_failure_semantics(
    exchange_report: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Separate completed-inference quality rejection from backend failure."""
    report = dict(exchange_report or {})
    failure_code = str(report.get("failure_code") or "").strip().upper()
    status = str(report.get("status") or "").strip().lower()
    quality_rejected = bool(
        report.get("accepted") is not True
        and status == "rejected"
        and failure_code in _STAGE6_DEFINITE_QUALITY_REJECTION_CODES
    )
    return {
        "quality_rejected": quality_rejected,
        "star_separation_state": (
            StarSeparationState.REJECTED.value
            if quality_rejected
            else StarSeparationState.TOOL_FAILED.value
        ),
        "reason_code": (
            "star_separation_quality_rejected"
            if quality_rejected
            else "star_separation_tool_failed"
        ),
        "underlying_failure_code": failure_code or None,
        "retry_history": [
            dict(item)
            for item in (report.get("attempts") or [])
            if isinstance(item, dict)
        ],
    }


def _stage6_bright_core_retry_plan(
    quality: Optional[Dict[str, Any]],
    *,
    retry_max: int,
    syqon_available: bool,
) -> Dict[str, Any]:
    """Freeze the strict bright-core retry budget at zero or one attempt."""
    bright_core = (
        quality.get("bright_core_integrity")
        if isinstance(quality, dict)
        else None
    )
    triggered = bool(
        isinstance(bright_core, dict)
        and bright_core.get("applicable", False)
        and bright_core.get("hard_failed", False)
    )
    bright_core_gates = (
        bright_core.get("gates")
        if isinstance(bright_core, dict)
        and isinstance(bright_core.get("gates"), dict)
        else {}
    )
    recoverable_artifact = bool(
        triggered
        and any(
            bool((gate or {}).get("hard_failed", False))
            for gate in bright_core_gates.values()
        )
    )
    configured_retry_max = max(int(retry_max or 0), 0)
    should_attempt = bool(
        recoverable_artifact
        and configured_retry_max > 0
        and syqon_available
    )
    status = (
        "ready"
        if should_attempt
        else "direct_reject"
        if triggered and not recoverable_artifact
        else "disabled"
        if triggered and configured_retry_max <= 0
        else "unavailable"
        if triggered and not syqon_available
        else "not_triggered"
    )
    return {
        "triggered": triggered,
        "recoverable_artifact": recoverable_artifact,
        "should_attempt": should_attempt,
        "attempt_limit": 1 if should_attempt else 0,
        "configured_retry_max": configured_retry_max,
        "status": status,
    }


def _stage6_bright_core_retry_passed(
    quality: Optional[Dict[str, Any]],
) -> bool:
    """Require the recovery pair to pass all ordinary and strict gates."""
    if not isinstance(quality, dict) or quality.get("status") != "ok":
        return False
    bright_core = quality.get("bright_core_integrity") or {}
    return bool(
        isinstance(bright_core, dict)
        and bright_core.get("applicable", False)
        and not bright_core.get("hard_failed", True)
    )


def _stage6_bright_core_with_stars_fallback_contract(
    pipeline,
    quality: Optional[Dict[str, Any]],
    retry: Optional[Dict[str, Any]],
    *,
    separation_accepted: bool,
) -> Dict[str, Any]:
    """Record the terminal with-stars review route after pair rejection.

    The field name is retained for compatibility with existing diagnostics,
    but a rejected strict bright-core pair may no longer authorize formal HDR
    delivery.  Stage 7 must consume this record as review-only evidence.
    """
    target_type = (
        pipeline._active_target_type()
        if hasattr(pipeline, "_active_target_type")
        else "generic_low_snr_safe"
    )
    strict_evidence = stage7_quality.strict_bright_core_target_evidence(
        target_type,
        getattr(pipeline, "target_profile", None),
    )
    selected = quality if isinstance(quality, dict) else {}
    bright_core = selected.get("bright_core_integrity") or {}
    bright_core = bright_core if isinstance(bright_core, dict) else {}
    roi = bright_core.get("roi") or {}
    roi = roi if isinstance(roi, dict) else {}
    gates = bright_core.get("gates") or {}
    gates = gates if isinstance(gates, dict) else {}
    destructive_gate_names = [
        str(name)
        for name, gate in gates.items()
        if isinstance(gate, dict) and bool(gate.get("hard_failed", False))
    ]
    retry_record = retry if isinstance(retry, dict) else {}
    retry_trigger_reasons = list(retry_record.get("trigger_reasons") or [])
    recovery_terminal_failure = bool(
        retry_trigger_reasons
        and not bool(retry_record.get("accepted", False))
        and str(retry_record.get("status") or "")
        not in {"not_triggered", "accepted"}
    )
    rejected_to_review = bool(
        not separation_accepted
        and strict_evidence.get("strict", False)
        and (
            (
                bright_core.get("applicable", False)
                and bright_core.get("hard_failed", False)
            )
            or recovery_terminal_failure
        )
    )
    blocked_reasons: List[str] = []
    if separation_accepted:
        blocked_reasons.append("star_separation_accepted")
    if not strict_evidence.get("strict", False):
        blocked_reasons.append("strict_target_evidence_missing")
    if not bright_core.get("hard_failed", False) and not recovery_terminal_failure:
        blocked_reasons.append("bright_core_integrity_not_hard_failed")
    if not roi.get("available", False):
        blocked_reasons.append(str(roi.get("reason") or "bright_core_roi_unavailable"))
    if not destructive_gate_names:
        blocked_reasons.append("destructive_bright_core_gate_evidence_missing")
    return {
        "schema": "starun.bright-core-with-stars-fallback.v1",
        "eligible": False,
        "accepted": False,
        "status": (
            "rejected_to_review" if rejected_to_review else "not_eligible"
        ),
        "source_stem": "stage6_input",
        "delivery_mode": "with_stars_review_only",
        "review_only": rejected_to_review,
        "review_output": (
            "stage7_review_with_stars" if rejected_to_review else None
        ),
        "strict_target_evidence": strict_evidence,
        "trigger_reasons": list(
            dict.fromkeys(
                [
                    *list(bright_core.get("trigger_reasons") or []),
                    *retry_trigger_reasons,
                ]
            )
        ),
        "destructive_gate_names": destructive_gate_names,
        "bright_core_roi": dict(roi),
        "rejected_pair_id": (
            selected.get("pair_id")
            or getattr(pipeline, "_selected_syqon_pair_id", None)
        ),
        "rejected_attempt": selected.get("attempt"),
        "retry": {
            "profile": retry_record.get("profile"),
            "attempted": bool(retry_record.get("attempted", False)),
            "accepted": bool(retry_record.get("accepted", False)),
            "status": retry_record.get("status", "not_triggered"),
        },
        "blocked_reasons": list(dict.fromkeys(blocked_reasons)),
    }


def _prepare_star_separation_source(pipeline) -> Tuple[str, str, List[Dict[str, Any]]]:
    lineage = stage5_handoff.verify_stage5_handoff(pipeline)
    linear_source = stage5_handoff.STAGE5_SOURCE_STEM
    records: List[Dict[str, Any]] = [
        {
            "mode": "linear_star_separation",
            "source_stem": linear_source,
            "status": "selected",
            "method": "linear",
            "domain": "linear",
            "source_lineage": lineage,
        }
    ]
    try:
        pipeline.cmd_with_check("cd", f'"{pipeline.process_dir}"')
        pipeline.cmd_with_check("load", linear_source)
    except (CommandError, SirilError) as error:
        raise stage5_handoff.Stage5HandoffError(
            stage5_handoff.REASON_SOURCE_UNAVAILABLE,
            f"canonical Stage 5 source cannot be loaded: {error}",
        ) from error
    if not pipeline._save_stage_output("stage6_input"):
        raise stage5_handoff.Stage5HandoffError(
            stage5_handoff.REASON_INPUT_CHECKPOINT_FAILED,
            "Stage 6 immutable input checkpoint could not be saved",
        )

    pipeline.stretched_name = linear_source
    pipeline._stage7_starless_first_source = linear_source
    return linear_source, "linear_star_separation", records


def run_stage6_star_separation(pipeline) -> None:
    """
    阶段 6: 去星与星点层准备
    - 自动链仅使用 Zenith-only `SyQon/Starless.py` 文件模式
    - SASP Dark Star 仅在显式 `sasp_only` 后端策略下运行
    - 生成并导出 starless/starmask 交换文件供外部工具使用
    """
    stage_label = PipelineStage.STAR_SEPARATION.label
    pipeline._clear_stage_reviews(6)
    pipeline.log.stage_start(stage_label)
    pipeline._stage7_starless_skipped = False
    pipeline._stage8_conservative_mode = False
    pipeline._stage6_quality_hard_failed_retained = False
    pipeline._stage6_quality_failure_codes = []
    pipeline._selected_syqon_pair_id = None
    pipeline._selected_syqon_attempt_id = None
    pipeline._stage6_pair_handoff = None
    existing_scene_support = getattr(pipeline, "_stage3_scene_support", None)
    legacy_scene_support = bool(
        isinstance(existing_scene_support, dict)
        and str(
            ((existing_scene_support.get("manifest") or {}).get("reason_code"))
            or ""
        )
        == "legacy_checkpoint_without_scene_support"
    )
    shared_scene_support_summary = scene_support.scene_support_summary(
        getattr(pipeline, "_stage3_scene_support", None)
    )
    pipeline._stage6_galaxy_roi_diagnostics = (
        _stage6_galaxy_roi_diagnostics()
    )
    pipeline._stage8_handoff = {
        "schema": "starun.stage8-handoff.v2",
        "requested_policy": "full",
        "processing_policy": "full",
        "source_stage": 6,
        "source_stem": None,
        "passthrough": False,
        "restricted_downstream": False,
        "reason_code": "",
        "reason_text": "",
        "reasons": [],
        "quality_status": "pending",
        "metrics": {},
        "repair": {"attempted": False, "accepted": False},
    }
    pipeline._star_separation_state = StarSeparationState.PENDING.value
    pipeline._stage6_passthrough_source = None
    pipeline._stage6_starmask_borderline_review_required = False
    pipeline._bright_core_with_stars_fallback = {
        "schema": "starun.bright-core-with-stars-fallback.v1",
        "eligible": False,
        "accepted": False,
        "status": "not_evaluated",
    }
    try:
        selected_source_stem, star_separation_mode, mode_input_records = (
            _prepare_star_separation_source(pipeline)
        )
    except stage5_handoff.Stage5HandoffError as error:
        reason_code = str(
            error.reason_code or stage5_handoff.REASON_LINEAGE_UNVERIFIED
        )
        detail = str(error.detail or error)
        source_lineage = stage5_handoff.public_handoff(
            getattr(pipeline, "_stage5_linear_handoff", None)
        )
        pipeline.starless_file = None
        pipeline.starmask_file = None
        pipeline._stage7_starless_skipped = True
        pipeline._stage8_conservative_mode = True
        pipeline._star_separation_state = StarSeparationState.REJECTED.value
        pipeline._stage8_handoff.update(
            {
                "requested_policy": "skip",
                "processing_policy": "skip",
                "source_stem": None,
                "passthrough": False,
                "restricted_downstream": True,
                "reason_code": reason_code,
                "reason_text": detail,
                "reasons": [
                    {
                        "code": reason_code,
                        "source_stage": 5,
                        "error": detail,
                    }
                ],
                "quality_status": "failed",
            }
        )
        pipeline._require_review(6, reason_code)
        pipeline._write_stage_json(
            "stage6_starless_quality.json",
            {
                "attempts": [],
                "shared_scene_support": shared_scene_support_summary,
                "selected": None,
                "galaxy_roi": pipeline._stage6_galaxy_roi_diagnostics,
                "mode": "upstream_source_rejected",
                "star_separation_state": pipeline._star_separation_state,
                "star_separation_mode": None,
                "input_domain": "linear",
                "selected_source_stem": None,
                "source_lineage": source_lineage,
                "stage8_conservative_mode": True,
                "stage8_handoff": pipeline._stage8_handoff,
                "reason_code": reason_code,
                "error": detail,
                "bright_core_with_stars_fallback": (
                    pipeline._bright_core_with_stars_fallback
                ),
            },
        )
        elapsed = pipeline.log.stage_end(stage_label)
        pipeline._record_stage(
            stage_label,
            "failed",
            elapsed,
            f"Stage 6 refused an unverified Stage 5 source: {detail}",
            execution="completed",
            reason_code=reason_code,
            details={
                "output": None,
                "source_lineage": source_lineage,
                "stage8_handoff": pipeline._stage8_handoff,
                "star_separation_state": pipeline._star_separation_state,
            },
            components={
                "input_source": {
                    "status": "failed",
                    "fatal": True,
                    "reason_code": reason_code,
                    "fallback_used": False,
                    "message": detail,
                }
            },
            review_reasons=pipeline._stage_review_reasons(6),
        )
        raise SirilError(
            f"Stage 6 upstream source rejected ({reason_code}): {detail}"
        ) from error
    if not legacy_scene_support and getattr(pipeline, "process_dir", None) is not None:
        try:
            stage6_source_pixels = pipeline.siril.get_image_pixeldata(preview=False)
            if stage6_source_pixels is None:
                raise ValueError("Stage 6 source pixel buffer is unavailable")
            pipeline._stage3_scene_support = scene_support.load_scene_support(
                pipeline.process_dir,
                expected_shape=tuple(np.asarray(stage6_source_pixels).shape),
            )
        except (
            AttributeError,
            CommandError,
            SirilError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            debug_log = getattr(pipeline.log, "debug", None)
            if callable(debug_log):
                debug_log(
                    f"stage6 shared scene support validation unavailable: {error}"
                )
            pipeline._stage3_scene_support = {
                "status": "unavailable",
                "manifest": scene_support.unavailable_scene_support(
                    str(error),
                    reason_code="stage6_scene_support_validation_unavailable",
                ),
                "valid_mask": None,
                "saturation_map": None,
            }
        shared_scene_support_summary = scene_support.scene_support_summary(
            pipeline._stage3_scene_support
        )
    target_type = (
        pipeline._active_target_type()
        if hasattr(pipeline, "_active_target_type")
        else "generic_low_snr_safe"
    )
    secondary_labels = list(
        (getattr(pipeline, "target_profile", {}) or {}).get(
            "secondary_labels", []
        )
    )
    user_preserve = (
        str(getattr(pipeline.cfg, "stage6_processing_mode", "auto"))
        == "preserve"
    )
    failure_action = str(
        getattr(pipeline.cfg, "stage6_failure_action", "auto_fallback")
    )
    backend_policy = str(
        getattr(pipeline.cfg, "stage6_starless_backend_policy", "auto_chain")
    )
    target_preserve = should_bypass_star_separation(
        target_type,
        enabled=bool(
            getattr(pipeline.cfg, "stage6_star_preserve_target_bypass_enabled", True)
        ),
    )
    if user_preserve or target_preserve:
        bypass_reason = (
            "user_preserve" if user_preserve else "star_preserve_target_bypass"
        )
        message_parts = [
            (
                "user requested star-preserve passthrough"
                if user_preserve
                else f"star-preserve target bypassed SyQon/SASP ({target_type})"
            ),
            f"source={selected_source_stem}",
        ]
        try:
            pipeline.cmd_with_check("load", selected_source_stem)
            stage_saved = pipeline._save_stage_output("stage6_passthrough")
            passthrough_source = (
                "stage6_passthrough" if stage_saved else selected_source_stem
            )
            pipeline._stage6_passthrough_source = passthrough_source
            pipeline.starless_file = None
            pipeline.starmask_file = None
            pipeline._stage7_starless_skipped = True
            pipeline._star_preserve_target_bypass = True
            pipeline._star_separation_state = StarSeparationState.TARGET_BYPASS.value
            pipeline._stage8_handoff.update(
                {
                    "requested_policy": "skip",
                    "processing_policy": "skip",
                    "source_stem": passthrough_source,
                    "passthrough": True,
                    "restricted_downstream": False,
                    "reason_code": bypass_reason,
                    "reason_text": bypass_reason,
                    "reasons": [
                        {
                            "code": bypass_reason,
                            "source_stage": 6,
                            "target_type": target_type,
                        }
                    ],
                    "quality_status": "skipped",
                }
            )
            quality_record = {
                "attempt": bypass_reason,
                "tool_label": "none",
                "status": "skipped",
                "issues": [
                    "user requested star preservation"
                    if user_preserve
                    else "stars are part of the target subject"
                ],
                "derived": {
                    "residual_star_score": 0.0,
                    "halo_residue_score": 0.0,
                    "starmask_contamination": 0.0,
                },
            }
            pipeline._write_stage_json(
                "stage6_starless_quality.json",
                {
                    "attempts": [quality_record],
                    "shared_scene_support": shared_scene_support_summary,
                    "selected": quality_record,
                    "galaxy_roi": pipeline._stage6_galaxy_roi_diagnostics,
                    "mode": bypass_reason,
                    "star_separation_state": pipeline._star_separation_state,
                    "target_type": target_type,
                    "secondary_labels": secondary_labels,
                    "routing_basis": "primary_target_only",
                    "star_separation_mode": star_separation_mode,
                    "input_domain": "linear",
                    "selected_source_stem": selected_source_stem,
                    "conservative_inputs": mode_input_records,
                    "stage9_star_remix": {
                        "scale": 1.0,
                        "reason": "no remix required for preserved star field",
                    },
                    "stage8_handoff": pipeline._stage8_handoff,
                    "retry_max": 0,
                    "bright_core_with_stars_fallback": (
                        pipeline._bright_core_with_stars_fallback
                    ),
                },
            )
            if stage_saved and hasattr(pipeline, "_create_stage_review_bundle"):
                review = pipeline._create_stage_review_bundle(
                    "stage6_star_separation",
                    "stage6_input",
                    "stage6_passthrough",
                    context={"mode": bypass_reason},
                    candidates=[quality_record],
                    selected_candidate=str(quality_record.get("attempt")),
                )
                if review.get("report_path"):
                    message_parts.append(f"review_bundle={review['report_path']}")
            if not stage_saved:
                message_parts.append("stage6 star-preserve output save failed")
            elapsed = pipeline.log.stage_end(stage_label)
            pipeline._record_stage(
                stage_label,
                "ok" if stage_saved else "degraded",
                elapsed,
                "；".join(message_parts),
                execution="safe_passthrough",
                reason_code=bypass_reason,
                details={
                    "output": "stage6_passthrough" if stage_saved else None,
                    "star_separation_state": pipeline._star_separation_state,
                },
            )
            return
        except (CommandError, SirilError) as error:
            if user_preserve:
                pipeline._stage6_passthrough_source = selected_source_stem
                pipeline.starless_file = None
                pipeline.starmask_file = None
                pipeline._stage7_starless_skipped = True
                pipeline._star_preserve_target_bypass = True
                pipeline._star_separation_state = (
                    StarSeparationState.TARGET_BYPASS.value
                )
                pipeline._stage8_handoff.update(
                    {
                        "requested_policy": "skip",
                        "processing_policy": "skip",
                        "source_stem": selected_source_stem,
                        "passthrough": True,
                        "restricted_downstream": True,
                        "reason_code": "user_preserve_passthrough_failed",
                        "reason_text": str(error),
                        "reasons": [
                            {
                                "code": "user_preserve_passthrough_failed",
                                "source_stage": 6,
                                "target_type": target_type,
                            }
                        ],
                        "quality_status": "failed",
                    }
                )
                pipeline._write_stage_json(
                    "stage6_starless_quality.json",
                    {
                        "attempts": [],
                        "shared_scene_support": shared_scene_support_summary,
                        "selected": None,
                        "galaxy_roi": pipeline._stage6_galaxy_roi_diagnostics,
                        "mode": "user_preserve",
                        "star_separation_state": pipeline._star_separation_state,
                        "input_domain": "linear",
                        "selected_source_stem": selected_source_stem,
                        "stage8_handoff": pipeline._stage8_handoff,
                        "bright_core_with_stars_fallback": (
                            pipeline._bright_core_with_stars_fallback
                        ),
                        "error": str(error),
                    },
                )
                elapsed = pipeline.log.stage_end(stage_label)
                pipeline._record_stage(
                    stage_label,
                    "failed",
                    elapsed,
                    "用户要求保留含星图，但 Stage 6 直通产物无法建立："
                    f"{pipeline._short_text(error, 160)}",
                    execution="completed",
                    reason_code="user_preserve_passthrough_failed",
                    details={
                        "output": None,
                        "fallback_source": selected_source_stem,
                        "star_separation_state": pipeline._star_separation_state,
                    },
                )
                return
            pipeline.log.warn(
                "Star-preserve bypass failed; continuing with regular star separation: "
                f"{error}"
            )
            pipeline._star_preserve_target_bypass = False

    pipeline.stretched_name = selected_source_stem

    pipeline.log.info("执行去星流程...")
    try:
        pipeline.starless_file = None
        pipeline.starmask_file = None
        pipeline._stage7_starmask_cleanup_hard_failed = False
        syqon_failure_reason: Optional[str] = None
        stage_messages: List[str] = []
        quality_records: List[Dict[str, Any]] = []
        selected_quality: Optional[Dict[str, Any]] = None
        starless_used = None
        stage7_preflight: Optional[Dict[str, Any]] = None
        starmask_cleanup_records: List[Dict[str, Any]] = []
        repair_records: List[Dict[str, Any]] = []
        starless_pixel_repair_records: List[Dict[str, Any]] = []
        chroma_retry: Dict[str, Any] = {
            "profile": syqon_starless.SYQON_CHROMA_RECOVERY_PROFILE.manifest(),
            "eligible": False,
            "attempted": False,
            "accepted": False,
            "status": "not_triggered",
        }
        chroma_retry_terminal_failure = False
        bright_core_retry: Dict[str, Any] = {
            "profile": syqon_starless.SYQON_BRIGHT_CORE_RECOVERY_PROFILE.manifest(),
            "eligible": False,
            "attempted": False,
            "accepted": False,
            "status": "not_triggered",
            "trigger_reasons": [],
        }
        bright_core_retry_terminal_failure = False
        conservative_input_records: List[Dict[str, Any]] = list(mode_input_records)
        selected_source_stem = pipeline.stretched_name or selected_source_stem
        stage_messages.append(
            "star_separation_mode="
            f"{star_separation_mode}; input_domain=linear; "
            f"external_prestretch=false; source={selected_source_stem}"
        )

        if hasattr(pipeline, "_stage7_preflight_check"):
            stage7_preflight = pipeline._stage7_preflight_check()
        else:
            stage7_preflight = {"risk_level": "ok", "issues": []}
        preflight_summary = (
            pipeline._stage7_preflight_summary(stage7_preflight)
            if hasattr(pipeline, "_stage7_preflight_summary")
            else ""
        )
        if stage7_preflight.get("risk_level") != "ok":
            stage_messages.append(preflight_summary)

        syqon_profile = syqon_starless.SYQON_BASELINE_PROFILE
        syqon_script = (
            pipeline._find_plugin_script(SYQON_SCRIPT_CANDIDATES)
            if backend_policy != "sasp_only"
            else None
        )
        if syqon_script is not None:
            pipeline._last_syqon_exchange_report = {}
            _syqon_args, _syqon_timeout, syqon_device_note = pipeline._syqon_starless_cli_options(
                profile=syqon_profile,
            )
            pipeline.log.info(
                "执行 Zenith-only SyQon/Starless.py 文件模式 CLI 子进程"
                f"（{syqon_device_note}）"
            )
            syqon_used = pipeline._stage7_try_syqon_variant(
                syqon_script,
                attempt_name="initial",
                profile=syqon_profile,
            )
            initial_exchange = getattr(pipeline, "_last_syqon_exchange_report", {})
            if (
                not syqon_used
                and isinstance(initial_exchange, dict)
                and initial_exchange.get("failure_code") == "TILE_ARTIFACT"
                and bool(
                    getattr(
                        pipeline.cfg,
                        "stage6_syqon_seam_retry_enabled",
                        True,
                    )
                )
            ):
                stage_messages.append(
                    "SyQon tile artifact gate rejected the initial pair; "
                    "retrying once with 512/128 CPU FP32"
                )
                syqon_profile = (
                    syqon_starless.SYQON_TILE_ARTIFACT_RECOVERY_PROFILE
                )
                _syqon_args, _syqon_timeout, syqon_device_note = (
                    pipeline._syqon_starless_cli_options(profile=syqon_profile)
                )
                syqon_used = pipeline._stage7_try_syqon_variant(
                    syqon_script,
                    attempt_name="tile_artifact_cpu_recovery",
                    profile=syqon_profile,
                )
            elif (
                not syqon_used
                and isinstance(initial_exchange, dict)
                and initial_exchange.get("failure_code") == "OOM_DEVICE"
            ):
                stage_messages.append(
                    "SyQon GPU OOM; retrying the same Zenith baseline once on CPU"
                )
                syqon_profile = syqon_starless.SYQON_CPU_RECOVERY_PROFILE
                _syqon_args, _syqon_timeout, syqon_device_note = (
                    pipeline._syqon_starless_cli_options(profile=syqon_profile)
                )
                syqon_used = pipeline._stage7_try_syqon_variant(
                    syqon_script,
                    attempt_name="oom_cpu_recovery",
                    profile=syqon_profile,
                )
            if syqon_used:
                stage_messages.append(
                    f"SyQon model: Zenith ({syqon_profile.profile_id}; "
                    f"{syqon_device_note})"
                )
                if pipeline.starmask_file:
                    pipeline.log.info(
                        "SyQon 产物已归一化: starless.fit, starmask_raw.fit"
                    )
                else:
                    pipeline.log.info("SyQon 产物已归一化: starless.fit")
                starless_used = syqon_used
            else:
                syqon_failure_reason = (
                    getattr(pipeline, "_last_plugin_script_error", None)
                    or f"SyQon 脚本执行失败: {syqon_script.name}"
                )
        elif backend_policy != "sasp_only":
            syqon_failure_reason = (
                "SyQon/Starless.py 缺失；自动链不启用未标定替代去星模型"
            )
        else:
            syqon_failure_reason = "SyQon skipped by SASP-only backend policy"

        if not starless_used and syqon_failure_reason:
            pipeline.log.warn(syqon_failure_reason)

        if not starless_used and backend_policy == "sasp_only":
            starless_used = pipeline._run_first_available_command(
                "去星",
                [
                    ("SASP Dark Star", ("sasp_dark_star",)),
                ],
                allow_when_probe_disabled=True,
            )
            if starless_used:
                stage_messages.append(
                    pipeline._fallback_summary(
                        "SyQon/Starless.py",
                        syqon_failure_reason or "SyQon unavailable",
                        starless_used,
                        True,
                    )
                )
        if not starless_used:
            raise SirilError(
                "未找到符合后端策略的可用去星命令 "
                f"(policy={backend_policy})"
            )

        if not pipeline.starless_file:
            pipeline.cmd_with_check("save", "starless")
            pipeline.starless_file = pipeline.process_dir / "starless.fit"
            pipeline.log.info("去星图像已保存: starless.fit")

        # 先用手动差分构建星点层（更可控）
        pipeline._stage7_prepare_starmask()
        starmask_cleanup = pipeline._stage7_clean_starmask(label="initial")
        starmask_cleanup_records.append(starmask_cleanup)
        if starmask_cleanup.get("status") == "applied":
            metrics = starmask_cleanup.get("metrics") or {}
            syqon_starless.record_syqon_derived_generation(
                pipeline,
                generation="clean",
                details={"starmask_cleanup": metrics},
            )
            stage_messages.append(
                "stage6 starmask multi-scale cleanup applied "
                f"(signal_ratio={float(metrics.get('signal_ratio', 1.0)):.3f}, "
                f"compact_retention={float(metrics.get('compact_retention', 1.0)):.3f})"
            )
        elif starmask_cleanup.get("status") in {
            "rolled_back",
            "hard_rejected",
            "failed",
        }:
            stage_messages.append(
                "stage6 starmask cleanup retained original mask "
                f"(status={starmask_cleanup.get('status')}, "
                f"reason={starmask_cleanup.get('reason') or 'quality gate'})"
            )

        selected_quality = pipeline._stage7_quality_assessment(
            "initial",
            tool_label=str(starless_used or "unknown"),
            source_stem=selected_source_stem,
        )
        selected_quality = _apply_starmask_cleanup_hard_gate(
            selected_quality,
            starmask_cleanup,
        )
        selected_quality["retry_profile"] = syqon_profile.manifest()
        quality_records.append(selected_quality)
        parameter_retries_done = 0

        chroma_retry_plan = _stage6_chroma_retry_plan(
            selected_quality,
            retry_max=getattr(pipeline.cfg, "stage7_quality_retry_max", 0),
            syqon_available=syqon_script is not None,
            failure_action=failure_action,
        )
        initial_chroma_failure = chroma_retry_plan["failure_code"]
        if bool(chroma_retry_plan["triggered"]):
            configured_retry_max = chroma_retry_plan["configured_retry_max"]
            chroma_retry.update(
                {
                    "eligible": chroma_retry_plan["eligible"],
                    "status": chroma_retry_plan["status"],
                    "initial_attempt": selected_quality.get("attempt"),
                    "initial_failure_code": initial_chroma_failure,
                    "configured_retry_max": configured_retry_max,
                    "attempt_limit": chroma_retry_plan["attempt_limit"],
                    "failure_action": failure_action,
                }
            )
            selected_quality["final_selection_state"] = (
                "rejected_subject_chroma_lineage"
            )
            should_retry_chroma = bool(chroma_retry_plan["should_attempt"])
            if should_retry_chroma:
                recovery_profile = syqon_starless.SYQON_CHROMA_RECOVERY_PROFILE
                chroma_snapshot = pipeline._stage7_snapshot_current_outputs(
                    "before_chroma_linked_mtf_recovery"
                )
                chroma_retry.update(attempted=True, status="running")
                parameter_retries_done = 1
                stage_messages.append(
                    "SUBJECT_CHROMA_COLLAPSE hard rejection; retrying once with "
                    "zenith_chroma_linked_mtf_recovery (linked MTF, FP32)"
                )
                retry_used = pipeline._stage7_try_syqon_variant(
                    syqon_script,
                    attempt_name="chroma_linked_mtf_recovery",
                    profile=recovery_profile,
                )
                if retry_used:
                    starless_used = retry_used
                    syqon_profile = recovery_profile
                    if not pipeline.starless_file:
                        pipeline.cmd_with_check("save", "starless")
                        pipeline.starless_file = pipeline.process_dir / "starless.fit"
                    pipeline._stage7_prepare_starmask()
                    retry_cleanup = pipeline._stage7_clean_starmask(
                        label="chroma_linked_mtf_recovery"
                    )
                    starmask_cleanup_records.append(retry_cleanup)
                    if retry_cleanup.get("status") == "applied":
                        syqon_starless.record_syqon_derived_generation(
                            pipeline,
                            generation="clean",
                            details={
                                "starmask_cleanup": retry_cleanup.get("metrics") or {},
                                "profile_id": recovery_profile.profile_id,
                            },
                        )
                    retry_quality = pipeline._stage7_quality_assessment(
                        "chroma_linked_mtf_recovery",
                        tool_label=str(retry_used),
                        source_stem=selected_source_stem,
                    )
                    retry_quality = _apply_starmask_cleanup_hard_gate(
                        retry_quality,
                        retry_cleanup,
                    )
                    retry_quality["retry_profile"] = recovery_profile.manifest()
                    quality_records.append(retry_quality)
                    retry_safe = _stage6_chroma_retry_passed(retry_quality)
                    chroma_retry.update(
                        {
                            "accepted": retry_safe,
                            "status": "accepted" if retry_safe else "rejected",
                            "quality_status": retry_quality.get("status"),
                            "quality_after": retry_quality,
                        }
                    )
                    if retry_safe:
                        selected_quality["final_selection_state"] = (
                            "superseded_by_chroma_linked_mtf_recovery"
                        )
                        retry_quality["final_selection_state"] = "selected"
                        selected_quality = retry_quality
                        starmask_cleanup = retry_cleanup
                        stage_messages.append(
                            "zenith_chroma_linked_mtf_recovery passed the full "
                            "Stage 6 quality contract"
                        )
                    else:
                        retry_quality["final_selection_state"] = (
                            "rejected_subject_chroma_or_quality"
                        )
                        try:
                            pipeline._stage7_restore_snapshot(chroma_snapshot)
                            chroma_retry["rollback"] = "restored_initial_pair"
                        except (OSError, RuntimeError, ValueError) as error:
                            chroma_retry["rollback"] = "failed"
                            chroma_retry["rollback_error"] = str(error)
                        chroma_retry_terminal_failure = True
                        stage_messages.append(
                            "linked-MTF chroma recovery failed the full Stage 6 "
                            "recheck; candidate rolled back and will be rejected"
                        )
                else:
                    try:
                        pipeline._stage7_restore_snapshot(chroma_snapshot)
                        chroma_retry["rollback"] = "restored_initial_pair"
                    except (OSError, RuntimeError, ValueError) as error:
                        chroma_retry["rollback"] = "failed"
                        chroma_retry["rollback_error"] = str(error)
                    chroma_retry_terminal_failure = True
                    chroma_retry.update(
                        {
                            "status": "failed",
                            "failure_reason": (
                                getattr(pipeline, "_last_plugin_script_error", None)
                                or "SyQon linked-MTF recovery unavailable"
                            ),
                        }
                    )
                    stage_messages.append(
                        "zenith_chroma_linked_mtf_recovery failed or timed out; "
                        "the desaturated baseline pair remains rejected"
                    )
            else:
                chroma_retry_terminal_failure = True
                chroma_retry["status"] = chroma_retry_plan["status"]
                stage_messages.append(
                    f"{initial_chroma_failure} cannot enter the Starless main "
                    f"chain (retry_status={chroma_retry['status']})"
                )

        initial_bright_core = selected_quality.get("bright_core_integrity") or {}
        bright_core_retry_plan = _stage6_bright_core_retry_plan(
            selected_quality,
            retry_max=(
                0
                if parameter_retries_done or chroma_retry_terminal_failure
                else getattr(pipeline.cfg, "stage7_quality_retry_max", 0)
            ),
            syqon_available=(
                syqon_script is not None and not chroma_retry_terminal_failure
            ),
        )
        if bool(bright_core_retry_plan["triggered"]):
            bright_core_retry.update(
                {
                    "eligible": bright_core_retry_plan[
                        "recoverable_artifact"
                    ],
                    "status": "blocked_by_retry_limit",
                    "trigger_reasons": list(
                        initial_bright_core.get("trigger_reasons") or []
                    ),
                    "initial_attempt": selected_quality.get("attempt"),
                    "attempt_limit": bright_core_retry_plan["attempt_limit"],
                    "configured_retry_max": bright_core_retry_plan[
                        "configured_retry_max"
                    ],
                }
            )
            selected_quality["final_selection_state"] = (
                "rejected_bright_core_integrity"
            )
            retry_max = bright_core_retry_plan["configured_retry_max"]
            if bool(bright_core_retry_plan["should_attempt"]):
                recovery_profile = (
                    syqon_starless.SYQON_BRIGHT_CORE_RECOVERY_PROFILE
                )
                bright_core_retry.update(
                    {
                        "attempted": True,
                        "status": "running",
                    }
                )
                parameter_retries_done = 1
                stage_messages.append(
                    "BRIGHT_CORE_INTEGRITY hard rejection; retrying once with "
                    "zenith_bright_core_ihs_recovery (IHS 0.15, FP32)"
                )
                retry_used = pipeline._stage7_try_syqon_variant(
                    syqon_script,
                    attempt_name="bright_core_ihs_recovery",
                    profile=recovery_profile,
                )
                if retry_used:
                    starless_used = retry_used
                    syqon_profile = recovery_profile
                    if not pipeline.starless_file:
                        pipeline.cmd_with_check("save", "starless")
                        pipeline.starless_file = pipeline.process_dir / "starless.fit"
                    pipeline._stage7_prepare_starmask()
                    retry_cleanup = pipeline._stage7_clean_starmask(
                        label="bright_core_ihs_recovery"
                    )
                    starmask_cleanup_records.append(retry_cleanup)
                    if retry_cleanup.get("status") == "applied":
                        syqon_starless.record_syqon_derived_generation(
                            pipeline,
                            generation="clean",
                            details={
                                "starmask_cleanup": retry_cleanup.get("metrics") or {},
                                "profile_id": recovery_profile.profile_id,
                            },
                        )
                    retry_quality = pipeline._stage7_quality_assessment(
                        "bright_core_ihs_recovery",
                        tool_label=str(retry_used),
                        source_stem=selected_source_stem,
                    )
                    retry_quality = _apply_starmask_cleanup_hard_gate(
                        retry_quality,
                        retry_cleanup,
                    )
                    retry_quality["retry_profile"] = recovery_profile.manifest()
                    quality_records.append(retry_quality)
                    retry_bright_core = (
                        retry_quality.get("bright_core_integrity") or {}
                    )
                    retry_safe = _stage6_bright_core_retry_passed(
                        retry_quality
                    )
                    bright_core_retry.update(
                        {
                            "accepted": retry_safe,
                            "status": "accepted" if retry_safe else "rejected",
                            "quality_status": retry_quality.get("status"),
                            "bright_core_integrity": retry_bright_core,
                        }
                    )
                    selected_quality = retry_quality
                    starmask_cleanup = retry_cleanup
                    if retry_safe:
                        quality_records[0]["final_selection_state"] = (
                            "superseded_by_bright_core_recovery"
                        )
                        retry_quality["final_selection_state"] = "selected"
                        stage_messages.append(
                            "zenith_bright_core_ihs_recovery passed all Stage 6 "
                            "quality and bright-core gates"
                        )
                    else:
                        retry_quality["final_selection_state"] = (
                            "rejected_bright_core_integrity"
                        )
                        repair_snapshot = pipeline._stage7_snapshot_current_outputs(
                            "before_bright_core_overshoot_repair"
                        )
                        bright_core_repair = (
                            stage7_repair.apply_stage6_bright_core_overshoot_repair(
                                pipeline,
                                source_stem=selected_source_stem,
                                label="bright_core_ihs_bounded_overshoot_repair",
                            )
                        )
                        starless_pixel_repair_records.append(bright_core_repair)
                        if bright_core_repair.get("status") == "applied":
                            repaired_quality = pipeline._stage7_quality_assessment(
                                "bright_core_ihs_after_bounded_overshoot_repair",
                                tool_label="bounded bright-core overshoot repair",
                                source_stem=selected_source_stem,
                            )
                            repaired_quality = _apply_starmask_cleanup_hard_gate(
                                repaired_quality,
                                retry_cleanup,
                            )
                            repaired_quality["retry_profile"] = (
                                recovery_profile.manifest()
                            )
                            quality_records.append(repaired_quality)
                            repair_safe = _stage6_bright_core_retry_passed(
                                repaired_quality
                            )
                            bright_core_repair["accepted"] = repair_safe
                            bright_core_repair["quality_after"] = repaired_quality
                            bright_core_retry["bounded_overshoot_repair"] = (
                                bright_core_repair
                            )
                            if repair_safe:
                                quality_records[0]["final_selection_state"] = (
                                    "superseded_by_bright_core_recovery"
                                )
                                retry_quality["final_selection_state"] = (
                                    "superseded_by_bounded_overshoot_repair"
                                )
                                repaired_quality["final_selection_state"] = "selected"
                                selected_quality = repaired_quality
                                bright_core_retry.update(
                                    {
                                        "accepted": True,
                                        "status": "accepted_after_bounded_repair",
                                        "quality_status": repaired_quality.get("status"),
                                        "bright_core_integrity": repaired_quality.get(
                                            "bright_core_integrity"
                                        ),
                                    }
                                )
                                stage_messages.append(
                                    "zenith_bright_core_ihs_recovery accepted after "
                                    "bounded frozen-source overshoot repair"
                                )
                            else:
                                pipeline._stage7_restore_snapshot(repair_snapshot)
                                selected_quality = retry_quality
                                bright_core_retry_terminal_failure = True
                                stage_messages.append(
                                    "bounded bright-core overshoot repair failed the "
                                    "full Stage 6 recheck and was rolled back"
                                )
                        else:
                            pipeline._stage7_restore_snapshot(repair_snapshot)
                            selected_quality = retry_quality
                            bright_core_retry["bounded_overshoot_repair"] = (
                                bright_core_repair
                            )
                            bright_core_retry_terminal_failure = True
                            stage_messages.append(
                                "zenith_bright_core_ihs_recovery rejected; bounded "
                                "overshoot repair was unavailable, so the bad pair "
                                "will be purged"
                            )
                else:
                    bright_core_retry_terminal_failure = True
                    bright_core_retry.update(
                        {
                            "status": "failed",
                            "failure_reason": (
                                getattr(pipeline, "_last_plugin_script_error", None)
                                or "SyQon recovery unavailable"
                            ),
                        }
                    )
                    stage_messages.append(
                        "zenith_bright_core_ihs_recovery failed or timed out; "
                        "baseline pair remains rejected"
                    )
            else:
                bright_core_retry_terminal_failure = True
                bright_core_retry["status"] = bright_core_retry_plan["status"]
                if bright_core_retry_plan["status"] == "direct_reject":
                    stage_messages.append(
                        "BRIGHT_CORE_INTEGRITY reference/ROI is unavailable; "
                        "strict target is rejected without an IHS retry"
                    )
                else:
                    stage_messages.append(
                        "BRIGHT_CORE_INTEGRITY rejection cannot recover: "
                        + (
                            "stage7_quality_retry_max=0"
                            if retry_max <= 0
                            else "SyQon recovery profile unavailable"
                        )
                    )

        best_quality = selected_quality
        best_cleanup = starmask_cleanup
        best_snapshot = (
            pipeline._stage7_snapshot_current_outputs("best_initial")
            if pipeline.cfg.stage7_conservative_repair_enabled
            else None
        )
        best_label = "initial"
        best_source_stem = selected_source_stem

        if selected_quality["status"] != "ok":
            stage_messages.append(
                "stage7_quality diagnostic: "
                + ", ".join(selected_quality.get("issues", [])[:3])
            )
        elif selected_quality.get("advisories"):
            stage_messages.append(
                "stage7_quality advisory (continue): "
                + ", ".join(selected_quality.get("advisories", [])[:3])
            )

        repair_triggers = pipeline._stage7_repair_triggers(selected_quality)
        quality_failure_codes = _syqon_quality_failure_codes(repair_triggers)
        if (
            bright_core_retry.get("eligible")
            and "BRIGHT_CORE_INTEGRITY" not in quality_failure_codes
        ):
            quality_failure_codes.append("BRIGHT_CORE_INTEGRITY")
        if (
            selected_quality["status"] != "ok"
            and not bright_core_retry_terminal_failure
            and bool(pipeline.cfg.stage7_conservative_repair_enabled)
            and failure_action == "auto_fallback"
        ):
            stage_messages.append(
                "stage7_quality automatic parameter search withheld: no free, "
                "calibrated Zenith alternate profile for "
                + (", ".join(quality_failure_codes) or "UNKNOWN_QUALITY")
            )

        quality_mode = (
            "local_parameter_optimization"
            if parameter_retries_done
            else "local_quality"
        )

        pixel_repair_trigger = _stage7_starless_pixel_repair_trigger(
            pipeline,
            selected_quality,
        )
        if (
            selected_quality
            and pixel_repair_trigger.get("triggered")
            and not bright_core_retry_terminal_failure
            and not chroma_retry_terminal_failure
            and _stage6_subject_chroma_failure_code(selected_quality) is None
            and bool(getattr(pipeline.cfg, "stage7_starless_pixel_repair_enabled", True))
            and failure_action == "auto_fallback"
        ):
            stage_messages.append(
                "Stage6 starless pixel repair triggered "
                f"({pixel_repair_trigger['reason']}, "
                f"halo={pixel_repair_trigger['halo_residue_score']:.3f}, "
                f"compact_halo={pixel_repair_trigger['compact_halo_residue_score']:.3f}, "
                f"base_limit={pixel_repair_trigger['base_halo_limit']:.3f}, "
                f"target_limit={pixel_repair_trigger['target_halo_limit']:.3f})"
            )
            before_repair_quality = selected_quality
            before_repair_score = pipeline._stage7_quality_score(before_repair_quality)
            repair_snapshot = pipeline._stage7_snapshot_current_outputs("before_pixel_repair")
            pixel_repair = pipeline._apply_stage7_starless_pixel_repair(
                source_stem=selected_source_stem,
                label="selected_starless_pixel_repair",
            )
            pixel_repair["trigger"] = pixel_repair_trigger
            starless_pixel_repair_records.append(pixel_repair)
            if pixel_repair.get("status") == "applied":
                repaired_quality = pipeline._stage7_quality_assessment(
                    "selected_after_starless_pixel_repair",
                    tool_label="stage7 starless pixel repair",
                    source_stem=selected_source_stem,
                )
                repaired_quality = _apply_starmask_cleanup_hard_gate(
                    repaired_quality,
                    starmask_cleanup,
                )
                quality_records.append(repaired_quality)
                repaired_score = pipeline._stage7_quality_score(repaired_quality)
                non_regression = _stage7_repair_non_regression(
                    before_repair_quality,
                    repaired_quality,
                )
                before_derived = before_repair_quality.get("derived", {})
                repaired_derived = repaired_quality.get("derived", {})
                before_residual = float(before_derived.get("residual_star_score", 0.0) or 0.0)
                after_residual = float(repaired_derived.get("residual_star_score", 0.0) or 0.0)
                before_halo = float(before_derived.get("halo_residue_score", 0.0) or 0.0)
                after_halo = float(repaired_derived.get("halo_residue_score", 0.0) or 0.0)
                before_compact_halo = float(
                    before_derived.get("compact_halo_residue_score", 0.0) or 0.0
                )
                after_compact_halo = float(
                    repaired_derived.get("compact_halo_residue_score", 0.0) or 0.0
                )
                residual_not_worse = after_residual <= before_residual + 0.002
                halo_not_worse = (
                    after_halo <= before_halo + 0.002
                    and after_compact_halo <= before_compact_halo + 0.002
                )
                trigger_improvement = _stage7_trigger_improvement(
                    before_repair_quality,
                    repaired_quality,
                    repair_triggers,
                    bright_nebula_halo_advisory=(
                        pixel_repair_trigger.get("reason")
                        == "bright_nebula_halo_advisory"
                    ),
                )
                repair_metrics = pixel_repair.get("metrics") or {}
                chroma_acceptance = _stage7_chroma_repair_acceptance(
                    pipeline.cfg,
                    repair_metrics.get("background_quality_before") or {},
                    repair_metrics.get("background_quality_after") or {},
                    residual_not_worse=residual_not_worse,
                    halo_not_worse=halo_not_worse,
                )
                acceptance_audit = _stage6_repair_acceptance(
                    score_before=before_repair_score,
                    score_after=repaired_score,
                    configured_max_score_growth=(
                        pipeline.cfg.stage7_starless_repair_max_score_growth
                    ),
                    non_regression_passed=bool(non_regression["accepted"]),
                    trigger_improved=bool(trigger_improvement.get("accepted")),
                    chroma_improved=bool(chroma_acceptance.get("accepted")),
                )
                accepted = bool(acceptance_audit["accepted"])
                acceptance_path = (
                    "chroma_reduction"
                    if chroma_acceptance.get("accepted")
                    else "trigger_metric_improvement"
                    if trigger_improvement.get("accepted")
                    else "rejected"
                )
                pixel_repair.update(
                    {
                        "accepted": accepted,
                        "acceptance_path": acceptance_path,
                        "chroma_acceptance": chroma_acceptance,
                        "non_regression": non_regression,
                        "trigger_improvement": trigger_improvement,
                        **acceptance_audit,
                        "residual_before": before_residual,
                        "residual_after": after_residual,
                        "halo_before": before_halo,
                        "halo_after": after_halo,
                        "compact_halo_before": before_compact_halo,
                        "compact_halo_after": after_compact_halo,
                        "quality_after": repaired_quality,
                    }
                )
                if accepted:
                    selected_quality = repaired_quality
                    quality_mode = f"{quality_mode}+starless_pixel_repair"
                    syqon_starless.record_syqon_derived_generation(
                        pipeline,
                        generation="repaired",
                        details={
                            "acceptance_path": acceptance_path,
                            "non_regression": non_regression,
                            **acceptance_audit,
                        },
                    )
                    pipeline._stage8_conservative_mode = bool(
                        pipeline.cfg.stage8_force_conservative_after_stage7_repair
                    )
                    stage_messages.append(
                        "stage7 starless pixel repair accepted "
                        f"via {acceptance_path} "
                        f"(score {before_repair_score:.3f}->{repaired_score:.3f}, "
                        "chroma "
                        f"{float(chroma_acceptance.get('before', 0.0)):.5f}->"
                        f"{float(chroma_acceptance.get('after', 0.0)):.5f})"
                    )
                else:
                    pipeline._stage7_restore_snapshot(repair_snapshot)
                    selected_quality = before_repair_quality
                    stage_messages.append(
                        "stage7 starless pixel repair rolled back "
                        f"(score {before_repair_score:.3f}->{repaired_score:.3f}; "
                        "violations="
                        + ",".join(non_regression.get("violations", []))
                        + ")"
                    )
            elif pixel_repair.get("reason"):
                stage_messages.append(
                    "stage7 starless pixel repair skipped: "
                    f"{pixel_repair.get('reason')}"
                )

        selected_derived = (
            selected_quality.get("derived") or {}
            if isinstance(selected_quality, dict)
            else {}
        )
        pipeline._stage6_galaxy_roi_diagnostics = (
            _stage6_galaxy_roi_diagnostics(selected_derived)
        )
        selected_advisories = [
            str(item).strip()
            for item in (selected_quality or {}).get("advisories", [])
            if str(item).strip()
        ]
        cleanup_hard_failed = bool(
            selected_derived.get("starmask_cleanup_hard_failed", False)
        )
        cleanup_borderline = bool(
            selected_derived.get("starmask_cleanup_borderline", False)
        )
        pipeline._stage7_starmask_cleanup_hard_failed = cleanup_hard_failed
        pipeline._stage6_starmask_borderline_review_required = (
            cleanup_borderline and not cleanup_hard_failed
        )
        if pipeline._stage6_starmask_borderline_review_required:
            pipeline._require_review(6, "starmask_cleanup_borderline")
        if cleanup_hard_failed:
            pipeline.starmask_file = None
            stage_messages.append(
                "stage6 starmask diffuse-residual hard gate disabled star remix "
                "for this candidate"
            )
        elif cleanup_borderline:
            stage_messages.append(
                "stage6 starmask diffuse residual is inside the 200% advisory "
                "band; retained with limited downstream policy and "
                "review-only delivery"
            )
        if selected_advisories:
            stage_messages.append(
                "stage6 quality advisory accepted without rollback: "
                + ", ".join(selected_advisories[:3])
            )

        stage9_remix_quality = selected_quality

        pair_verification = syqon_starless.verify_selected_syqon_pair(pipeline)
        if pair_verification.get("status") == "rejected":
            stage_messages.append(
                "SyQon selected pair rejected before handoff: "
                f"{pair_verification.get('reason', 'PAIR_MISMATCH')}"
            )
            pipeline.starless_file = None
            pipeline.starmask_file = None
            pipeline._selected_syqon_pair_id = None
            pipeline._selected_syqon_attempt_id = None

        final_quality_failure = _stage6_quality_hard_failure_summary(
            pipeline,
            selected_quality,
        )
        # Attempts retain their own historical failure records.  The Stage 8
        # handoff must expose only failures that remain on the selected pair;
        # otherwise a successfully repaired generation is rejected by a stale
        # BRIGHT_CORE_INTEGRITY code from its immutable parent.
        quality_failure_codes = list(
            dict.fromkeys(final_quality_failure["failure_codes"])
        )
        quality_rejection_reason_code = (
            "stage6_subject_chroma_collapse"
            if "SUBJECT_CHROMA_COLLAPSE" in quality_failure_codes
            else "stage6_subject_chroma_lineage_unverified"
            if "SUBJECT_CHROMA_LINEAGE_UNVERIFIED" in quality_failure_codes
            else "star_separation_quality_rejected"
        )
        quality_gate_passed = bool(
            selected_quality
            and selected_quality.get("status") == "ok"
            and not final_quality_failure["hard_failed"]
            and not bright_core_retry_terminal_failure
            and not chroma_retry_terminal_failure
            and pipeline.starless_file
            and not cleanup_hard_failed
        )
        syqon_pair_valid = bool(
            getattr(pipeline, "_selected_syqon_pair_id", None)
            and pipeline.starless_file
            and pipeline.starless_file.is_file()
        )
        poor_candidate_retained = _stage6_can_retain_hard_failed_pair(
            final_quality_failure,
            pair_valid=syqon_pair_valid,
            bright_core_retry_terminal_failure=(
                bright_core_retry_terminal_failure
            ),
        )
        separation_accepted = quality_gate_passed or poor_candidate_retained
        quality_rejected = bool(
            not separation_accepted
            and (
                final_quality_failure["hard_failed"]
                or bright_core_retry_terminal_failure
                or cleanup_hard_failed
            )
        )
        pipeline._bright_core_with_stars_fallback = (
            _stage6_bright_core_with_stars_fallback_contract(
                pipeline,
                selected_quality,
                bright_core_retry,
                separation_accepted=separation_accepted,
            )
        )
        pipeline._stage6_quality_hard_failed_retained = poor_candidate_retained
        pipeline._stage6_quality_failure_codes = quality_failure_codes
        pipeline._star_separation_state = (
            StarSeparationState.REVIEW_REQUIRED.value
            if poor_candidate_retained
            else StarSeparationState.ACCEPTED.value
            if quality_gate_passed
            else StarSeparationState.REJECTED.value
        )
        if poor_candidate_retained:
            pipeline._require_review(6, "stage6_quality_hard_failed_retained")
            pipeline._stage8_conservative_mode = True
            stage_messages.append(
                "stage6_quality_hard_failed_retained: keeping the best "
                "contract-valid Zenith pair for limited processing and review"
            )
        if not separation_accepted:
            pipeline._stage7_starless_skipped = True
            if quality_rejected:
                pipeline._require_review(6, quality_rejection_reason_code)
                if hasattr(pipeline, "_record_stage_policy_event"):
                    pipeline._record_stage_policy_event(
                        6,
                        event="candidate_rejected",
                        reason=(
                            "starless candidate failed active quality gates: "
                            + ", ".join(
                                quality_failure_codes or ["UNKNOWN_QUALITY"]
                            )
                        ),
                        source="starless_quality_gate",
                    )
            elif failure_action != "auto_fallback":
                pipeline._require_review(6, "star_separation_candidate_rejected")
                if hasattr(pipeline, "_record_stage_policy_event"):
                    pipeline._record_stage_policy_event(
                        6,
                        event="candidate_search_stopped",
                        reason="initial starless candidate failed active hard gates",
                        source="starless_quality_gate",
                    )
            removed_artifacts = syqon_starless.purge_unaccepted_star_separation_outputs(
                pipeline
            )
            pipeline._stage7_update_star_remix_from_quality(None)
            stage9_remix_quality = None
            stage_messages.append(
                (
                    "star_separation_quality_rejected: bad Starless pair cannot "
                    "enter downstream processing; using with-stars review passthrough"
                )
                if quality_rejected
                else (
                    "star separation candidate rejected; downstream uses with-stars "
                    "review passthrough"
                )
            )
            if removed_artifacts:
                stage_messages.append(
                    "removed rejected Starless artifacts: "
                    + ", ".join(sorted(removed_artifacts))
                )
        else:
            pipeline._stage7_starless_skipped = False

        pipeline._stage8_handoff = _stage8_handoff_from_stage6(
            pipeline,
            selected_quality,
            starless_pixel_repair_records,
            separation_accepted=separation_accepted,
        )
        pipeline._stage8_conservative_mode = (
            pipeline._stage8_handoff["processing_policy"] != "full"
        )
        handoff_reason_text = str(
            pipeline._stage8_handoff.get("reason_text") or ""
        )
        if handoff_reason_text:
            stage_messages.append(
                "stage8_processing_policy="
                f"{pipeline._stage8_handoff['processing_policy']}; "
                f"{handoff_reason_text}"
            )

        stage9_remix_record = pipeline._stage7_update_star_remix_from_quality(
            stage9_remix_quality
        )
        if poor_candidate_retained:
            pipeline._stage9_star_intensity_scale = min(
                float(pipeline._stage9_star_intensity_scale),
                0.70,
            )
            pipeline._stage9_star_intensity_reason = (
                "stage6 quality hard failure retained for review"
            )
            stage9_remix_record.update(
                {
                    "intensity_scale": pipeline._stage9_star_intensity_scale,
                    "reason": pipeline._stage9_star_intensity_reason,
                    "review_required": True,
                }
            )

        for quality_record in quality_records:
            if "final_selection_state" in quality_record:
                continue
            if quality_record is selected_quality and separation_accepted:
                quality_record["final_selection_state"] = (
                    "retained_for_review"
                    if poor_candidate_retained
                    else "selected"
                )
            elif quality_record is selected_quality:
                quality_record["final_selection_state"] = "rejected"
            else:
                quality_record["final_selection_state"] = "not_selected"

        selected_chroma_lineage = (
            selected_quality.get("subject_chroma_lineage")
            if isinstance(selected_quality, dict)
            else None
        )
        chroma_attempts = [
            {
                "attempt": record.get("attempt"),
                "tool_label": record.get("tool_label"),
                "source_stem": record.get("source_stem"),
                "retry_profile": record.get("retry_profile"),
                "final_selection_state": record.get("final_selection_state"),
                "lineage": record.get("subject_chroma_lineage"),
            }
            for record in quality_records
            if isinstance(record, dict)
        ]
        pipeline._write_stage_json(
            "stage6_subject_chroma_lineage.json",
            {
                "schema": stage7_quality.STAGE6_SUBJECT_CHROMA_SCHEMA,
                "run_id": str(getattr(pipeline, "_run_id", "") or ""),
                "status": (
                    selected_chroma_lineage.get("status")
                    if isinstance(selected_chroma_lineage, dict)
                    else (
                        "not_applicable"
                        if pipeline._star_separation_state == "target_bypass"
                        else "unverified"
                    )
                ),
                "accepted": (
                    bool(selected_chroma_lineage.get("accepted", False))
                    if isinstance(selected_chroma_lineage, dict)
                    else pipeline._star_separation_state == "target_bypass"
                ),
                "hard_failed": (
                    bool(selected_chroma_lineage.get("hard_failed", True))
                    if isinstance(selected_chroma_lineage, dict)
                    else pipeline._star_separation_state != "target_bypass"
                ),
                "reason_code": (
                    selected_chroma_lineage.get("reason_code")
                    if isinstance(selected_chroma_lineage, dict)
                    else (
                        None
                        if pipeline._star_separation_state == "target_bypass"
                        else "stage6_subject_chroma_lineage_unverified"
                    )
                ),
                "source_stem": selected_source_stem,
                "star_separation_state": pipeline._star_separation_state,
                "attempts": chroma_attempts,
                "selected": selected_chroma_lineage,
                "selected_attempt": (
                    selected_quality.get("attempt")
                    if isinstance(selected_quality, dict)
                    else None
                ),
                "chroma_retry": chroma_retry,
                "quality_failure_codes": quality_failure_codes,
                "rejection_reason_code": (
                    quality_rejection_reason_code if quality_rejected else None
                ),
                "shared_scene_support": shared_scene_support_summary,
            },
        )

        pipeline._write_stage_json(
            "stage6_starless_quality.json",
            {
                "attempts": quality_records,
                "shared_scene_support": shared_scene_support_summary,
                "selected": selected_quality,
                "galaxy_roi": pipeline._stage6_galaxy_roi_diagnostics,
                "mode": quality_mode,
                "backend_policy": backend_policy,
                "failure_action": failure_action,
                "star_separation_state": pipeline._star_separation_state,
                "star_separation_mode": star_separation_mode,
                "input_domain": "linear",
                "selected_source_stem": selected_source_stem,
                "preflight": stage7_preflight,
                "starmask_cleanup": starmask_cleanup_records,
                "repairs": repair_records,
                "starless_pixel_repairs": starless_pixel_repair_records,
                "conservative_inputs": conservative_input_records,
                "stage8_conservative_mode": pipeline._stage8_conservative_mode,
                "stage8_handoff": pipeline._stage8_handoff,
                "stage9_star_remix": stage9_remix_record,
                "retry_max": pipeline.cfg.stage7_quality_retry_max,
                "automatic_quality_retries": parameter_retries_done,
                "quality_failure_codes": quality_failure_codes,
                "rejection_reason_code": (
                    quality_rejection_reason_code
                    if quality_rejected
                    else None
                ),
                "quality_hard_failed_retained": poor_candidate_retained,
                "bright_core_integrity": (
                    (selected_quality or {}).get("bright_core_integrity")
                ),
                "bright_core_retry": bright_core_retry,
                "chroma_retry": chroma_retry,
                "subject_chroma_lineage_report": {
                    "schema": stage7_quality.STAGE6_SUBJECT_CHROMA_SCHEMA,
                    "artifact": "stage6_subject_chroma_lineage.json",
                    "selected_status": (
                        selected_chroma_lineage.get("status")
                        if isinstance(selected_chroma_lineage, dict)
                        else None
                    ),
                },
                "bright_core_with_stars_fallback": (
                    pipeline._bright_core_with_stars_fallback
                ),
                "pair_verification": pair_verification,
            },
        )
        if pipeline._stage9_star_intensity_scale < 0.999:
            stage_messages.append(
                "stage9 star remix intensity linked to stage7 residuals "
                f"(scale={pipeline._stage9_star_intensity_scale:.3f}, "
                f"reason={pipeline._stage9_star_intensity_reason})"
            )

        pipeline._export_sasp_exchange_files()
        if separation_accepted:
            pipeline.cmd_with_check("load", pipeline.starless_file.stem)
            stage_output_stem = "stage6_starless"
            review_before_stem = "stage6_input"
        else:
            pipeline.cmd_with_check("load", selected_source_stem)
            stage_output_stem = "stage6_passthrough"
            review_before_stem = "stage6_input"
            pipeline._stage6_passthrough_source = stage_output_stem
            pipeline.starless_file = None
        stage_saved = pipeline._save_stage_output(stage_output_stem)
        if stage_saved and separation_accepted:
            pair_handoff = syqon_starless.record_stage6_pair_handoff(
                pipeline,
                source_path=pipeline.process_dir / "stage6_input.fit",
                starless_path=pipeline.process_dir / "stage6_starless.fit",
            )
            if pair_handoff.get("accepted") is True:
                stage_messages.append(
                    "stage6 pair handoff frozen for matched-domain Stage9"
                )
            else:
                stage_messages.append(
                    "stage6 pair handoff unavailable; Stage9 Unscreen/PSF will "
                    "fall back safely: "
                    f"{pair_handoff.get('reason', 'unknown error')}"
                )
            if pipeline.starmask_file is not None:
                guard_report = star_halo_guard.persist_stage6_guard(
                    pipeline,
                    starless_path=pipeline.process_dir / "stage6_starless.fit",
                    starmask_path=Path(pipeline.starmask_file),
                )
            else:
                guard_report = {
                    "schema": star_halo_guard.SCHEMA,
                    "status": "failed",
                    "reason_code": "stage6_star_halo_guard_starmask_unavailable",
                }
            pipeline._stage6_star_halo_guard_report = dict(guard_report)
            pipeline._stage8_handoff["star_halo_guard"] = {
                "status": guard_report.get("status"),
                "reason_code": guard_report.get("reason_code"),
                "report": star_halo_guard.REPORT_NAME,
                "artifact": guard_report.get("artifact"),
                "artifact_sha256": guard_report.get("artifact_sha256"),
            }
            if str(guard_report.get("status") or "") in {
                "failed",
                "hard_failed",
            }:
                pipeline._stage8_handoff["restricted_downstream"] = True
                pipeline._stage8_handoff["processing_policy"] = "limited"
                pipeline._stage8_handoff["quality_status"] = "degraded"
                pipeline._stage8_conservative_mode = True
                pipeline._require_review(
                    6,
                    str(guard_report.get("reason_code") or "stage6_star_halo_guard_failed"),
                )
                stage_messages.append(
                    "stage6 star-halo guard restricted Stage8: "
                    + str(guard_report.get("reason_code") or "unknown")
                )
            else:
                stage_messages.append(
                    "stage6 star-halo guard frozen for Stage8 "
                    f"(coverage={float((guard_report.get('metrics') or {}).get('coverage', 0.0)):.4f})"
                )
        if stage_saved and hasattr(pipeline, "_create_stage_review_bundle"):
            review = pipeline._create_stage_review_bundle(
                "stage6_star_separation",
                review_before_stem,
                stage_output_stem,
                context={
                    "mode": quality_mode,
                    "star_separation_state": pipeline._star_separation_state,
                },
                candidates=quality_records,
                selected_candidate=str((selected_quality or {}).get("attempt") or ""),
            )
            if review.get("report_path"):
                stage_messages.append(f"review_bundle={review['report_path']}")
        stage_message_text = "；".join(stage_messages)

        elapsed = pipeline.log.stage_end(stage_label)
        if stage_saved:
            selected_status = str((selected_quality or {}).get("status", "ok")).lower()
            halo_guard_status = str(
                (
                    getattr(pipeline, "_stage6_star_halo_guard_report", {})
                    or {}
                ).get("status")
                or ""
            )
            stage_status = (
                "ok"
                if (
                    separation_accepted
                    and selected_status == "ok"
                    and not selected_advisories
                    and halo_guard_status not in {"failed", "hard_failed"}
                )
                else "degraded"
            )
            stage6_fallback_used = bool(
                separation_accepted
                and syqon_failure_reason
                and starless_used
                and backend_policy == "auto_chain"
            )
            if not separation_accepted and failure_action == "stop":
                stage_status = "failed"
            pipeline._record_stage(
                stage_label,
                stage_status,
                elapsed,
                stage_message_text,
                fallback_used=stage6_fallback_used,
                reason_code=(
                    "stage6_quality_hard_failed_retained"
                    if poor_candidate_retained
                    else "star_separation_quality_rejected"
                    if quality_rejected
                    else "failure_policy_stop"
                    if not separation_accepted and failure_action == "stop"
                    else "failure_policy_preserve_review"
                    if not separation_accepted
                    and failure_action == "preserve_review"
                    else
                    "syqon_to_alternate_starless"
                    if stage6_fallback_used
                    else ""
                ),
                execution=(
                    "safe_passthrough"
                    if not separation_accepted
                    else "completed"
                ),
                upstream_passthrough=not separation_accepted,
                details={
                    "stage8_handoff": pipeline._stage8_handoff,
                    "backend_policy": backend_policy,
                    "failure_action": failure_action,
                },
                review_reasons=pipeline._stage_review_reasons(6),
            )
        else:
            if stage_message_text:
                stage_message_text = f"{stage_message_text}；stage7 输出保存失败"
            else:
                stage_message_text = "stage7 输出保存失败"
            pipeline._record_stage(
                stage_label, 'degraded', elapsed, stage_message_text
            )

    except (CommandError, SirilError) as e:
        syqon_exchange = dict(
            getattr(pipeline, "_last_syqon_exchange_report", {}) or {}
        )
        failure_semantics = _stage6_exchange_failure_semantics(syqon_exchange)
        quality_rejected = bool(failure_semantics["quality_rejected"])
        reason_code = str(failure_semantics["reason_code"])
        underlying_failure_code = failure_semantics[
            "underlying_failure_code"
        ]
        if quality_rejected:
            pipeline.log.error(
                "去星推理产物被质量门拒绝: "
                f"{underlying_failure_code or pipeline._short_text(e, 180)}"
            )
        else:
            pipeline.log.error(f"去星流程失败: {e}")
            pipeline.log.error("请检查哈希锁定的 SyQon Zenith 环境与模型配置")
        pipeline.starless_file = None
        pipeline.starmask_file = None
        pipeline._stage7_starless_skipped = True
        pipeline._stage8_conservative_mode = True
        pipeline._star_separation_state = failure_semantics[
            "star_separation_state"
        ]
        if quality_rejected:
            pipeline._require_review(6, reason_code)
        elif failure_action != "auto_fallback":
            pipeline._require_review(6, reason_code)
        if quality_rejected or failure_action != "auto_fallback":
            if hasattr(pipeline, "_record_stage_policy_event"):
                pipeline._record_stage_policy_event(
                    6,
                    event=(
                        "candidate_rejected"
                        if quality_rejected
                        else "backend_failed"
                    ),
                    reason=pipeline._short_text(e, 180),
                    source=(
                        "starless_quality_gate"
                        if quality_rejected
                        else "starless_backend"
                    ),
                )
        syqon_starless.purge_unaccepted_star_separation_outputs(pipeline)
        pipeline._stage8_handoff.update(
            {
                "requested_policy": "skip",
                "processing_policy": "skip",
                "restricted_downstream": True,
                "reason_code": reason_code,
                "reason_text": reason_code,
                "reasons": [
                    {
                        "code": reason_code,
                        "source_stage": 6,
                        "error": pipeline._short_text(e, 180),
                        "underlying_failure_code": underlying_failure_code,
                        "retry_history": failure_semantics["retry_history"],
                    }
                ],
                "quality_status": "rejected" if quality_rejected else "failed",
            }
        )
        pipeline._stage7_update_star_remix_from_quality(None)
        # 保留固定的含星线性 Stage 6 输入，仅供后续复核路径使用。
        pipeline.cmd_with_check("load", pipeline.stretched_name)
        stage_saved = pipeline._save_stage_output("stage6_passthrough")
        pipeline._stage6_passthrough_source = "stage6_passthrough"
        pipeline._write_stage_json(
            "stage6_starless_quality.json",
            {
                "attempts": [
                    {
                        "attempt": "tool_failed_with_stars_passthrough",
                        "tool_label": "none",
                        "status": "degraded",
                        "issues": [pipeline._short_text(e, 180)],
                        "reason_code": reason_code,
                        "underlying_failure_code": underlying_failure_code,
                    }
                ],
                "shared_scene_support": shared_scene_support_summary,
                "selected": None,
                "galaxy_roi": pipeline._stage6_galaxy_roi_diagnostics,
                "mode": "with_stars_review_passthrough",
                "backend_policy": backend_policy,
                "failure_action": failure_action,
                "star_separation_state": pipeline._star_separation_state,
                "reason_code": reason_code,
                "underlying_failure_code": underlying_failure_code,
                "syqon_exchange": syqon_exchange,
                "input_domain": "linear",
                "selected_source_stem": pipeline.stretched_name,
                "preflight": locals().get("stage7_preflight"),
                "starmask_cleanup": locals().get("starmask_cleanup_records", []),
                "repairs": locals().get("repair_records", []),
                "starless_pixel_repairs": locals().get("starless_pixel_repair_records", []),
                "conservative_inputs": locals().get("conservative_input_records", []),
                "stage8_handoff": pipeline._stage8_handoff,
                "retry_max": pipeline.cfg.stage7_quality_retry_max,
                "bright_core_with_stars_fallback": (
                    pipeline._bright_core_with_stars_fallback
                ),
            },
        )
        pipeline._export_sasp_exchange_files()

        elapsed = pipeline.log.stage_end(stage_label)
        message = (
            "去星产物未通过质量门，已切换为含星复核路径"
            if quality_rejected
            else "无可用去星工具，已切换为含星复核路径"
        )
        if not stage_saved:
            message += "；stage7 输出保存失败"
        pipeline._record_stage(
            stage_label,
            "failed" if failure_action == "stop" else "degraded",
            elapsed,
            message,
            execution="safe_passthrough",
            fallback_used=True,
            upstream_passthrough=True,
            reason_code=(
                reason_code
                if quality_rejected
                else "failure_policy_stop"
                if failure_action == "stop"
                else "failure_policy_preserve_review"
                if failure_action == "preserve_review"
                else "star_separation_tool_failed"
            ),
            details={
                "backend_policy": backend_policy,
                "failure_action": failure_action,
                "output": "stage6_passthrough" if stage_saved else None,
            },
            review_reasons=(
                pipeline._stage_review_reasons(6)
                if failure_action != "stop"
                else []
            ),
            issues=[
                {
                    "component": "star_separation",
                    "severity": "fatal" if failure_action == "stop" else "error",
                    "code": reason_code,
                    "underlying_failure_code": underlying_failure_code,
                    "recovered": bool(stage_saved and failure_action != "stop"),
                    "message": pipeline._short_text(e, 180),
                }
            ],
        )
