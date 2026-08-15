"""Stage 8 starless nebula enhancement."""
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


def _set_stage8_handoff(
    pipeline,
    *,
    source_stem: Optional[str],
    passthrough: bool,
    restricted_downstream: bool,
    final_quality: str,
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
    handoff.update(
        {
            "schema": "starun.stage8-handoff.v1",
            "source_stem": source_stem,
            "passthrough": bool(passthrough),
            "restricted_downstream": bool(restricted_downstream),
            "reason_code": effective_reason_code,
            "reason_text": effective_reason_text,
            "reasons": reasons,
            "final_quality": final_quality,
        }
    )
    pipeline._stage8_handoff = handoff
    return handoff


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
) -> Dict[str, Any]:
    """Write the P0 report-only color baseline and measured ledger entry."""
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
            "mode": "report_only",
            "used_for_gate": False,
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
                else "masked_single_chroma_recovery"
                if saturation_execution.get("applied", False)
                else "structure_only_or_passthrough"
            ),
        )
        report["saturation_execution"] = saturation_execution
    ledger = list(getattr(pipeline, "_color_adjustment_ledger", []) or [])
    entry = report.get("ledger_entry")
    if isinstance(entry, dict):
        ledger.append(dict(entry))
    pipeline._color_adjustment_ledger = ledger
    report["cross_stage_ledger"] = list(ledger)
    pipeline._stage8_color_quality_report = dict(report)
    pipeline._write_stage_json("stage8_color_quality_report.json", report)
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
        stage8_quality=str(getattr(pipeline, "_stage8_final_quality", "unknown")),
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
        masks = pipeline._stage8_generate_starless_masks(image_data)
        candidate, candidate_report = build_dualband_palette_candidate(
            image_data,
            palette=str(selection["palette"]),
            core_mask=masks["core_mask"],
            nebula_mask=masks["nebula_mask"],
            faint_nebula_mask=masks["faint_nebula_mask"],
            background_mask=masks["background_mask"],
            strength=float(
                getattr(pipeline.cfg, "stage8_dualband_palette_strength", 0.85)
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
        )
        report["candidate"] = candidate_report
        if not bool(candidate_report.get("accepted", False)):
            report.update(
                status="rejected_by_palette_quality_gate",
                issues=list(candidate_report.get("issues") or []),
            )
            pipeline.cmd_with_check("load", "stage8_pre_palette")
            report["transaction"].update(
                rollback_performed=True,
                rollback_ok=True,
            )
            messages.append(
                "Stage8 dual-band palette rejected; retained pre-palette enhancement"
            )
            return finish()

        _stage8_set_image_pixels(
            pipeline,
            candidate,
            label=f"Stage8 dual-band {selection['palette']} palette",
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
            palette=selection["palette"],
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
                f"(target={selection['category']}, palette={selection['palette']}, "
                f"warnings={','.join(candidate_warnings)})"
            )
        else:
            messages.append(
                "Stage8 dual-band artistic palette accepted "
                f"(target={selection['category']}, palette={selection['palette']})"
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
            messages.append(
                "Stage8 dual-band palette failed and rolled back: "
                f"{pipeline._short_text(error, 160)}"
            )
        else:
            pipeline._stage8_final_quality = "palette_rollback_failed"
            pipeline._stage8_fallback_used = True
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
        review_reason_code = (
            "stage7_stretch_not_accepted_target_bypass"
            if target_bypass_stage7_rejected
            else "star_separation_unavailable"
        )
        final_quality = (
            "stage7_stretch_review_target_bypass"
            if target_bypass_stage7_rejected
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
            pipeline._background_review_required = True
        pipeline._record_stage(
            stage_label,
            decisive_status,
            elapsed,
            "；".join(messages),
            execution="safe_passthrough",
            reason_code=review_reason_code,
            details={"stage8_handoff": handoff},
        )
        return

    color_limits = color_safety_limits(
        getattr(pipeline, "pipeline_policy", {}) or {},
        getattr(pipeline, "color_calibration_report", {}) or {},
    )
    channel_semantics = str(
        getattr(pipeline, "_channel_semantics", "unknown") or "unknown"
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
            pipeline._stage8_input_source = source_stem
            pipeline._stage8_input_fallback_used = False
            stage8_saved = pipeline._save_stage_output("stage8_enhanced")
            pipeline._stage8_final_source = "stage8_enhanced" if stage8_saved else source_stem
            pipeline._stage8_final_quality = "star_preserve_bypass"
            pipeline._stage8_fallback_used = False
            handoff = _set_stage8_handoff(
                pipeline,
                source_stem=pipeline._stage8_final_source,
                passthrough=True,
                restricted_downstream=False,
                final_quality=pipeline._stage8_final_quality,
                reason_code="star_preserve_target_bypass",
                reason_text="star_preserve_target_bypass",
            )
            pipeline.starless_file = pipeline.process_dir / f"{pipeline._stage8_final_source}.fit"
            report = {
                "stage": "stage8_nebula_enhancement",
                "status": "skipped",
                "mode": "star_preserve_target_bypass",
                "source": source_stem,
                "final_source": pipeline._stage8_final_source,
                "target_type": (
                    pipeline._active_target_type()
                    if hasattr(pipeline, "_active_target_type")
                    else "generic_low_snr_safe"
                ),
                "handoff": handoff,
            }
            pipeline._write_stage_json("stage8_enhancement_report.json", report)
            messages.append(
                "star-preserve target bypassed Starless-only enhancement "
                f"(source={source_stem})"
            )
            if stage8_saved and hasattr(pipeline, "_create_stage_review_bundle"):
                review = pipeline._create_stage_review_bundle(
                    "stage8_nebula_enhancement",
                    source_stem,
                    "stage8_enhanced",
                    context={"mode": "star_preserve_target_bypass"},
                    candidates=[
                        {
                            "name": "star_preserve_bypass",
                            "stem": "stage8_enhanced",
                            "status": "skipped",
                            "selected": True,
                        }
                    ],
                    selected_candidate="star_preserve_bypass",
                )
                if review.get("report_path"):
                    messages.append(f"review_bundle={review['report_path']}")
            elapsed = pipeline.log.stage_end(stage_label)
            pipeline._record_stage(
                stage_label,
                "skipped" if stage8_saved else "degraded",
                elapsed,
                "；".join(messages),
                execution="safe_passthrough",
                reason_code="star_preserve_target_bypass",
                details={"stage8_handoff": handoff},
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
                    else "ok"
                    if conservative_skip
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
                        pipeline._background_review_required = True
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
                    },
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
                details={"stage8_handoff": handoff},
            )
            return

    requested_saturation = float(
        (stage8_processing_plan or {}).get(
            "saturation",
            pipeline.cfg.nebula_saturation,
        )
    )
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

    if (
        pipeline.cfg.optional_color_transform_enabled
        and global_color_rebalance_allowed
        and not stage8_restricted_mode
    ):
        pipeline._run_first_available_command(
            "调色1（可选）",
            [
                ("SASP Selective Color Correction", ("sasp_selective_color_correction",)),
                ("Selective Color Correction", ("selective_color_correction",)),
            ],
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
                            pipeline._background_review_required = True
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
                            pipeline._background_review_required = True
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
                            pipeline._background_review_required = True
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
    stage8_palette_report: Dict[str, Any] = {}
    if channel_semantics == NARROWBAND_COMPOSITE:
        stage8_palette_report = _stage8_run_dualband_palette(
            pipeline,
            messages,
            base_stem=str(pipeline._stage8_final_source or "stage8_enhanced"),
            channel_semantics=channel_semantics,
            processing_policy="limited" if stage8_limited_mode else "full",
            external_override=bool(external_starless_source),
        )
        if (
            stage8_palette_report.get("status") == "failed"
            and not bool(
                (stage8_palette_report.get("transaction") or {}).get(
                    "rollback_ok", False
                )
            )
        ):
            status = "degraded"
    stage8_passthrough = pipeline._stage8_final_source == "stage8_input_starless"
    final_reason_code = ""
    final_reason_text = ""
    if stage8_passthrough and not str(
        (getattr(pipeline, "_stage8_handoff", {}) or {}).get("reason_code") or ""
    ):
        final_reason_code = "stage8_enhancement_quality_rollback"
        final_reason_text = "stage8_enhancement_quality_rollback"
    handoff = _set_stage8_handoff(
        pipeline,
        source_stem=pipeline._stage8_final_source,
        passthrough=stage8_passthrough,
        restricted_downstream=bool(
            stage8_limited_mode
            or pipeline._stage8_fallback_used
            or stage8_passthrough
            or pipeline._stage8_final_quality != "ok"
        ),
        final_quality=pipeline._stage8_final_quality,
        reason_code=final_reason_code,
        reason_text=final_reason_text,
    )
    handoff["outcome_reason_code"] = (
        "stage8_limited_candidate_rejected"
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
    )
    handoff["color_contract"] = color_quality_report.get("contract")
    handoff["color_quality_report"] = "stage8_color_quality_report.json"
    handoff["saturation_execution"] = saturation_execution
    pipeline._stage8_handoff = handoff
    enhancement_report_value = locals().get("enhancement_report")
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
        enhancement_report_value["color_quality"] = color_quality_report
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
        },
    )
