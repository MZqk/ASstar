"""Stretch selection and execution."""
import copy
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from color_quality_metrics import physical_broadband_anchor_accepted
import display_rendition
from managed_output import audit_display_visibility
from models import PipelineStage, StarSeparationState
import scene_support
import stage5_handoff
from sirilpy.exceptions import CommandError, SirilError
import stage7_stretch_metrics
import syqon_starless


def _freeze_review_subject_chroma_plan(pipeline) -> Dict[str, Any]:
    """Freeze the observer-only weak-colour evidence and replay masks."""

    target_type = str(
        pipeline._active_target_type()
        if hasattr(pipeline, "_active_target_type")
        else ""
    ).strip().lower()
    try:
        budget = float(getattr(pipeline.cfg, "nebula_saturation", 0.0) or 0.0)
    except (TypeError, ValueError):
        budget = 0.0
    budget = float(np.clip(budget, 0.0, 0.65))
    eligibility = {
        "galaxy_target": target_type
        in {"galaxy", "large_galaxy", "small_galaxy"},
        "physical_broadband_anchor": physical_broadband_anchor_accepted(
            getattr(pipeline, "channel_profile", {}) or {},
            getattr(pipeline, "color_calibration_report", {}) or {},
        ),
        "target_aware_chroma_enabled": bool(
            getattr(pipeline.cfg, "stage8_target_aware_chroma_enabled", True)
        ),
        "nebula_saturation_enabled": bool(
            getattr(pipeline.cfg, "stage8_nebula_saturation_enabled", True)
        ),
        "positive_budget": budget > 0.0,
    }
    plan: Dict[str, Any] = {
        "schema": "starun.review-subject-chroma-plan.v1",
        "status": "ineligible",
        "accepted": False,
        "target_type": target_type,
        "eligibility": eligibility,
        "effective_saturation_budget": budget,
        "mask_artifact": {},
        "evidence": {},
    }
    if not all(eligibility.values()):
        plan["issues"] = [name for name, accepted in eligibility.items() if not accepted]
        pipeline._review_subject_chroma_plan = plan
        pipeline._write_stage_json("display_subject_chroma_plan.json", plan)
        return plan
    try:
        source = pipeline._read_image_by_stem(stage5_handoff.STAGE5_SOURCE_STEM)
        if source is None:
            raise ValueError("stage5_linear pixels unavailable")
        artifact_root = Path(
            getattr(pipeline, "work_dir", None)
            or Path(pipeline.process_dir).parent
        )
        masks = dict(getattr(pipeline, "_stage7_frozen_rendition_masks", {}) or {})
        mask_artifact = display_rendition.persist_display_rendition_masks(
            artifact_root,
            masks,
        )
        evidence = display_rendition.assess_weak_subject_chroma_source(
            source,
            masks,
        )
        plan.update(
            status="accepted" if evidence.get("accepted") else "rejected",
            accepted=bool(evidence.get("accepted", False)),
            mask_artifact=mask_artifact,
            evidence=evidence,
            source_binding=copy.deepcopy(
                getattr(pipeline, "_stage7_stage5_source_binding", {}) or {}
            ),
        )
        if not plan["accepted"]:
            plan["issues"] = list(evidence.get("issues") or [])
    except (OSError, RuntimeError, TypeError, ValueError, FloatingPointError) as error:
        plan.update(
            status="rejected",
            accepted=False,
            issues=[str(error)],
        )
    pipeline._review_subject_chroma_plan = plan
    pipeline._write_stage_json("display_subject_chroma_plan.json", plan)
    return plan


def _bind_stage7_formal_presentation_reference(
    pipeline,
    output_stem: str,
) -> Dict[str, Any]:
    """Bind the frozen internal ruler to the canonical formal Stage 7 FIT."""

    report = copy.deepcopy(
        getattr(pipeline, "_stage7_presentation_reference_report", {}) or {}
    )
    if report.get("status") != "ready" or not report.get("accepted"):
        return report
    try:
        if pipeline.process_dir is None:
            raise ValueError("process directory unavailable")
        output_path = Path(pipeline.process_dir) / f"{output_stem}.fit"
        container_sha = pipeline._sha256_file(output_path)
        pixels = pipeline.siril.get_image_pixeldata(preview=False)
        if not container_sha or pixels is None:
            raise ValueError("formal Stage7 artifact identity unavailable")
        pixel_sha = stage7_stretch_metrics.stage7_pixel_sha256(
            np.asarray(pixels)
        )
        reference_pixel_sha = str(
            (report.get("artifact") or {}).get("pixel_sha256") or ""
        )
        if not reference_pixel_sha or pixel_sha != reference_pixel_sha:
            raise ValueError("formal Stage7 pixels differ from frozen reference")
        formal_source = {
            "stem": output_stem,
            "file": output_path.name,
            "container_sha256": container_sha,
            "pixel_sha256": pixel_sha,
        }
        report["source_artifact"] = formal_source
        binding_payload = {
            "linear_source": report.get("linear_source"),
            "selected_candidate": report.get("selected_candidate"),
            "matched_domain": report.get("matched_domain"),
            "formal_source_artifact": formal_source,
        }
        report["source_binding_sha256"] = (
            stage7_stretch_metrics.canonical_json_sha256(binding_payload)
        )
        unsigned = dict(report)
        unsigned.pop("report_sha256", None)
        report["report_sha256"] = (
            stage7_stretch_metrics.canonical_json_sha256(unsigned)
        )
        pipeline._write_stage_json(
            "stage7_presentation_reference.json",
            report,
        )
        pipeline._stage7_presentation_reference_report = copy.deepcopy(
            report
        )
        return report
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as error:
        report.update(
            status="unavailable",
            accepted=False,
            reason_code="stage7_formal_source_binding_failed",
            reason=pipeline._short_text(error, 180),
        )
        report.pop("report_sha256", None)
        pipeline._write_stage_json(
            "stage7_presentation_reference.json",
            report,
        )
        pipeline._stage7_presentation_reference_report = copy.deepcopy(
            report
        )
        return report


def _run_with_stars_review_stretch(
    pipeline,
    separation_state: str,
    *,
    source_stem: str | None = None,
    reason_code: str | None = None,
    upstream_separation_state: str | None = None,
    candidates_evaluated: bool = False,
    selected_candidate_stem: str | None = None,
) -> None:
    """Create a conservative review image without invoking starless-only logic."""
    stage_label = PipelineStage.STRETCHING.label
    upstream_state = str(upstream_separation_state or separation_state)
    selected_candidate = str(
        getattr(pipeline, "_stage7_stretch_selected", "") or ""
    )
    candidate_domain = str(
        getattr(pipeline, "_stage7_candidate_domain", "starless") or "starless"
    )
    candidate_selected = bool(selected_candidate_stem and selected_candidate)
    stretch_state = (
        "review_candidate_selected"
        if candidate_selected
        else "rejected"
        if candidates_evaluated
        else "skipped"
    )
    candidate_execution = (
        "evaluated_accepted"
        if candidate_selected
        else "evaluated_not_accepted"
        if candidates_evaluated
        else "skipped"
    )
    messages: List[str] = [
        f"stage6_star_separation_state={upstream_state}",
        f"stage7_stretch_state={stretch_state}",
        (
            "with-stars stretch candidate passed every hard gate and was "
            "selected for review-only use"
            if candidate_selected
            else (
                "stretch candidates evaluated but none accepted; immutable "
                "with-stars review fallback activated"
                if candidate_domain == "with_stars"
                else "starless-only stretch candidates evaluated but none accepted; "
                "with-stars review fallback activated"
            )
            if candidates_evaluated
            else "stretch candidates skipped"
        ),
    ]
    source_stem = str(
        source_stem
        or getattr(pipeline, "_stage6_passthrough_source", None)
        or "stage6_passthrough"
    )
    review_reason = str(
        reason_code or f"star_separation_{separation_state}"
    )
    pipeline._stage7_stretch_accepted = False
    pipeline._stage7_stretch_output = None
    pipeline._stage7_stretch_forced_delivery = False
    pipeline._stage7_forced_delivery_reasons = []
    pipeline._stage7_background_color_review_required = False
    pipeline._stage7_background_color_review_gate = {
        "status": "not_run",
        "requires_review": False,
    }
    pipeline._stage7_presentation_reference_report = {
        "schema": "starun.stage7-presentation-reference.v1",
        "status": "not_run",
        "accepted": False,
    }
    pipeline._stage7_review_source = None
    display_contract = {}
    pipeline._require_review(7, review_reason)
    process_dir = getattr(pipeline, "process_dir", None)
    if process_dir is not None:
        for pattern in ("stage7_with_stars_hdr*.fit", "stage7_with_stars_hdr*.fits"):
            for stale_path in process_dir.glob(pattern):
                try:
                    stale_path.unlink()
                except OSError as error:
                    messages.append(
                        "stale formal HDR artifact cleanup failed: "
                        f"{pipeline._short_text(error, 120)}"
                    )
    saved = False
    try:
        pipeline.cmd_with_check(
            "load",
            str(selected_candidate_stem or source_stem),
        )
        messages.append(
            "hard-gated nonlinear with-stars candidate saved as review-only FITS"
            if candidate_selected
            else "immutable Stage5 with-stars pixels preserved as the review fallback"
        )
        saved = pipeline._save_stage_output("stage7_review_with_stars")
        if saved:
            pipeline.stretched_name = "stage7_review_with_stars"
            pipeline._stage7_review_source = pipeline.stretched_name
            pipeline._review_display_route = True
            try:
                review_pixels = pipeline.siril.get_image_pixeldata(
                    preview=False
                )
                visibility = audit_display_visibility(
                    review_pixels,
                    target_type=str(
                        pipeline._active_target_type()
                        if hasattr(pipeline, "_active_target_type")
                        else ""
                    ),
                    stars_required=True,
                )
                display_contract = display_rendition.build_review_contract(
                    review_pixels,
                    reason=review_reason,
                    source_stem="stage7_review_with_stars",
                    input_visibility=visibility,
                    subject_chroma_plan=dict(
                        getattr(pipeline, "_review_subject_chroma_plan", {}) or {}
                    ),
                    artifact_root=(
                        getattr(pipeline, "work_dir", None)
                        or Path(pipeline.process_dir).parent
                    ),
                    pixel_coordinate_domain=(
                        display_rendition.PIXEL_DOMAIN_BOTTOM_UP
                    ),
                )
            except (
                AttributeError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as error:
                display_contract = display_rendition.unavailable_contract(
                    reason=review_reason,
                    error=str(error),
                )
                messages.append(
                    "linked Review display contract unavailable: "
                    f"{pipeline._short_text(error, 160)}"
                )
            display_contract_valid = (
                display_rendition.validate_review_contract(display_contract)
            )
            if display_contract_valid:
                pipeline._display_rendition_contract = display_contract
            else:
                pipeline._display_rendition_contract = {}
                messages.append(
                    "invalid Review display contract retained as diagnostic only"
                )
            pipeline._write_stage_json(
                "display_rendition_contract.json",
                display_contract,
            )
            if hasattr(pipeline, "_create_stage_review_bundle"):
                review = pipeline._create_stage_review_bundle(
                    "stage7_stretching",
                    source_stem,
                    "stage7_review_with_stars",
                    context={
                        "delivery_mode": "with_stars_review_only",
                        "reason_code": review_reason,
                    },
                )
                if review.get("report_path"):
                    messages.append(f"review_bundle={review['report_path']}")
    except (CommandError, SirilError) as error:
        messages.append(
            "with-stars review source unavailable: "
            f"{pipeline._short_text(error, 160)}"
        )

    background_color_review_gate = dict(
        getattr(pipeline, "_stage7_background_color_review_gate", {}) or {}
    )
    bright_core_fallback = dict(
        getattr(pipeline, "_bright_core_with_stars_fallback", {}) or {}
    )
    stage4_core_color = dict(
        (
            getattr(pipeline, "color_calibration_report", {}) or {}
        ).get("bright_core_color_integrity")
        or {}
    )
    shared_scene_support_summary = scene_support.scene_support_summary(
        getattr(pipeline, "_stage3_scene_support", None)
    )
    review_quality_report = {
            "stage": "stage7_stretch",
            "status": "review_only" if saved else "failed",
            "delivery_mode": "with_stars_review_only",
            "formal_accepted": False,
            "delivery_class": "review_only",
            "reason_code": review_reason,
            "review_required": True,
            "input": f"{source_stem}.fit",
            "source_stem": source_stem,
            "candidate_domain": "with_stars",
            "candidate_execution": candidate_execution,
            "source_binding": copy.deepcopy(
                getattr(pipeline, "_stage7_source_binding", {}) or {}
            ),
            "stage5_source_binding": copy.deepcopy(
                getattr(pipeline, "_stage7_stage5_source_binding", {}) or {}
            ),
            "upstream_star_separation_state": upstream_state,
            "stage7_stretch_state": stretch_state,
            "starless_candidate_execution": candidate_execution,
            "shared_scene_support": shared_scene_support_summary,
            "review_output": "stage7_review_with_stars" if saved else None,
            "attempts": copy.deepcopy(
                getattr(pipeline, "_stage7_stretch_candidates", []) or []
            ),
            "candidates": copy.deepcopy(
                getattr(pipeline, "_stage7_stretch_candidates", []) or []
            ),
            "selected": (
                {
                    "name": selected_candidate,
                    "source_stem": str(selected_candidate_stem),
                    "source_file": f"{selected_candidate_stem}.fit",
                    "file": "stage7_review_with_stars.fit",
                    "formal_accepted": False,
                    "delivery_class": "review_only",
                }
                if candidate_selected
                else None
            ),
            "review_evidence": copy.deepcopy(
                getattr(pipeline, "_stage7_review_evidence", None)
            ),
            "galaxy_roi": copy.deepcopy(
                getattr(
                    pipeline,
                    "_stage6_galaxy_roi_diagnostics",
                    {"status": "not_run", "available": False},
                )
            ),
            "strict_bright_core_evidence": dict(
                bright_core_fallback.get("strict_target_evidence") or {}
            ),
            "bright_core_with_stars_fallback": bright_core_fallback,
            "stage4_bright_core_color_integrity": stage4_core_color,
            "stage7_bright_core_integrity_rejected": bool(
                review_reason == "stage7_bright_core_integrity_rejected"
            ),
            "stage7_bright_core_integrity_rejected_reasons": list(
                getattr(
                    pipeline,
                    "_stage7_bright_core_integrity_rejected_reasons",
                    [],
                )
                or []
            ),
            "revoked_pair_id": getattr(
                pipeline, "_stage7_revoked_pair_id", None
            ),
            "background_color_review_gate": background_color_review_gate,
            "display_rendition_contract": display_contract,
            "display_rendition_contract_active": bool(
                display_rendition.validate_review_contract(display_contract)
            ),
        }
    pipeline._write_stage_json(
        "stage7_stretch_quality.json",
        review_quality_report,
    )
    pipeline._write_stage_json(
        "stretch_candidates_report.json",
        review_quality_report,
    )
    elapsed = pipeline.log.stage_end(stage_label)
    if not saved:
        messages.append("stage7 review output save failed")
    pipeline._record_stage(
        stage_label,
        "degraded" if saved else "failed",
        elapsed,
        "；".join(messages),
        execution="safe_passthrough" if saved else "completed",
        reason_code=review_reason,
        details={
            "source_stem": source_stem,
            "review_output": "stage7_review_with_stars" if saved else None,
            "upstream_star_separation_state": upstream_state,
            "stage7_stretch_state": stretch_state,
            "starless_candidate_execution": candidate_execution,
            "candidate_domain": "with_stars",
            "candidate_execution": candidate_execution,
            "stage5_source_binding": copy.deepcopy(
                getattr(pipeline, "_stage7_stage5_source_binding", {}) or {}
            ),
            "background_color_review_gate": background_color_review_gate,
        },
        review_reasons=pipeline._stage_review_reasons(7),
    )


def _run_rejected_stage6_with_stars_candidates(
    pipeline,
    separation_state: str,
    *,
    reason_code: str | None = None,
) -> None:
    """Evaluate immutable Stage 5 with-stars candidates, then remain review-only."""

    try:
        verified_handoff = stage5_handoff.verify_stage5_handoff(pipeline)
    except stage5_handoff.Stage5HandoffError as error:
        pipeline._stage7_candidate_domain = "with_stars"
        pipeline._stage7_stage5_source_binding = {
            "status": "rejected",
            "accepted": False,
            "reason_code": error.reason_code,
            "detail": error.detail,
        }
        pipeline._review_subject_chroma_plan = {
            "schema": "starun.review-subject-chroma-plan.v1",
            "status": "rejected",
            "accepted": False,
            "issues": ["stage5_handoff_unverified"],
        }
        pipeline._stage7_stretch_candidates = []
        pipeline._stage7_stretch_selected = None
        _run_with_stars_review_stretch(
            pipeline,
            separation_state,
            source_stem=str(
                getattr(pipeline, "_stage6_passthrough_source", None)
                or "stage6_passthrough"
            ),
            reason_code="stage7_stage5_handoff_unverified",
            upstream_separation_state=separation_state,
            candidates_evaluated=False,
        )
        return

    pipeline._stage7_candidate_domain = "with_stars"
    pipeline._stage7_stage5_source_binding = stage5_handoff.public_handoff(
        verified_handoff
    )
    channel_semantics = str(
        getattr(pipeline, "_channel_semantics", "unknown") or "unknown"
    ).strip().lower()
    stage5_pixels = pipeline._read_image_by_stem(stage5_handoff.STAGE5_SOURCE_STEM)
    stage5_array = np.asarray(stage5_pixels) if stage5_pixels is not None else None
    stage5_is_rgb = bool(
        stage5_array is not None
        and stage5_array.ndim == 3
        and (stage5_array.shape[0] == 3 or stage5_array.shape[-1] == 3)
        and channel_semantics != "mono"
    )
    if not stage5_is_rgb:
        pipeline._stage7_stretch_candidates = []
        pipeline._stage7_stretch_selected = None
        pipeline._review_subject_chroma_plan = {
            "schema": "starun.review-subject-chroma-plan.v1",
            "status": "ineligible",
            "accepted": False,
            "issues": ["with_stars_rgb_source_unavailable"],
        }
        _run_with_stars_review_stretch(
            pipeline,
            separation_state,
            source_stem=stage5_handoff.STAGE5_SOURCE_STEM,
            reason_code="stage7_with_stars_rgb_source_unavailable",
            upstream_separation_state=separation_state,
            candidates_evaluated=False,
        )
        return
    pipeline._stage7_stretch_source = stage5_handoff.STAGE5_SOURCE_STEM
    stretched, _degraded, stretch_messages, _stretch_method = (
        pipeline._run_stage7_stretching_candidates()
    )
    for message in stretch_messages:
        pipeline.log.info(f"[Stage7 with-stars] {message}")
    selected_stem = None
    if stretched:
        selected_name = str(
            getattr(pipeline, "_stage7_stretch_selected", "") or ""
        )
        for attempt in getattr(pipeline, "_stage7_stretch_candidates", []) or []:
            if (
                str(attempt.get("name") or "") == selected_name
                and bool(attempt.get("allowed_as_final", False))
                and attempt.get("stem")
            ):
                selected_stem = str(attempt["stem"])
                break
    pipeline._stage7_stretch_accepted = False
    pipeline._stage7_stretch_output = None
    pipeline._stage7_stretch_forced_delivery = False
    pipeline._stage7_forced_delivery_reasons = []
    _freeze_review_subject_chroma_plan(pipeline)
    _run_with_stars_review_stretch(
        pipeline,
        separation_state,
        source_stem=stage5_handoff.STAGE5_SOURCE_STEM,
        reason_code=reason_code or f"star_separation_{separation_state}",
        upstream_separation_state=separation_state,
        candidates_evaluated=True,
        selected_candidate_stem=selected_stem,
    )


def run_stage7_stretching(pipeline) -> None:
    """
    阶段 7: 主体拉伸
    - 常规输入为 stage6_starless.fit；target_bypass 使用 stage6_passthrough.fit
    - 自动 Starless 路径默认生成 stage7_cand_a / stage7_cand_b / Display ladder
      和一个 stage7_preview_ref；星系路线增加 0.86 安全档，旧 auto_dual 仍只生成 A/B
    - 严格目标条件可用 Iterative Masked MTF 或 Dual-stage MTF+GHS 替换 cand_a
    - 亮核心星云 cand_a 使用扩张的星点/halo 掩膜保护并保留 1.50x 尺寸门
    - 完整变换后实测 P50；偏离时从线性源校准参数并重跑一次
    - 候选仅因背景色度门控失败时，可追加受控色度救援候选
    - 输出 stage7_stretched.fit
    """
    stage_label = PipelineStage.STRETCHING.label
    pipeline.log.stage_start(stage_label)
    pipeline._clear_stage_reviews(7)
    pipeline._stage7_stretch_accepted = False
    pipeline._stage7_stretch_output = None
    pipeline._stage7_stretch_forced_delivery = False
    pipeline._stage7_forced_delivery_reasons = []
    pipeline._stage7_background_color_review_required = False
    pipeline._stage7_background_color_review_gate = {
        "status": "not_run",
        "requires_review": False,
    }
    pipeline._stage7_presentation_reference_report = {
        "schema": "starun.stage7-presentation-reference.v1",
        "status": "not_run",
        "accepted": False,
    }
    stretched = False
    stage_degraded = False
    messages: List[str] = []
    stretch_method = ""
    failure_action = str(
        getattr(pipeline.cfg, "stage7_failure_action", "auto_fallback")
    )
    separation_state = str(
        getattr(
            pipeline,
            "_star_separation_state",
            StarSeparationState.ACCEPTED.value,
        )
    )
    bright_core_fallback = dict(
        getattr(pipeline, "_bright_core_with_stars_fallback", {}) or {}
    )
    if (
        separation_state == StarSeparationState.REJECTED.value
        and str(bright_core_fallback.get("status") or "")
        in {"eligible", "accepted", "rejected"}
    ):
        bright_core_fallback.update(
            eligible=False,
            accepted=False,
            status="rejected_to_review",
            delivery_mode="with_stars_review_only",
            review_only=True,
            review_output="stage7_review_with_stars",
        )
        pipeline._bright_core_with_stars_fallback = bright_core_fallback
    bright_core_review_only = bool(
        separation_state == StarSeparationState.REJECTED.value
        and bright_core_fallback.get("status") == "rejected_to_review"
    )
    color_report = dict(
        getattr(pipeline, "color_calibration_report", {}) or {}
    )
    stage4_core_color = dict(
        color_report.get("bright_core_color_integrity") or {}
    )
    unresolved_stage4_core_color = bool(
        stage4_core_color.get("applicable", False)
        and str(stage4_core_color.get("status") or "")
        not in {"ok", "repaired"}
    )
    if unresolved_stage4_core_color:
        syqon_starless.purge_unaccepted_star_separation_outputs(pipeline)
        pipeline._star_separation_state = StarSeparationState.REJECTED.value
        pipeline.starless_file = None
        pipeline.starmask_file = None
        pipeline._selected_syqon_pair_id = None
        pipeline._selected_syqon_attempt_id = None
        pipeline._stage7_starless_skipped = True
        pipeline._stage6_passthrough_source = "stage6_input"
        pipeline._stage6_pair_handoff = None
        _run_with_stars_review_stretch(
            pipeline,
            StarSeparationState.REJECTED.value,
            source_stem="stage6_input",
            reason_code="stage4_bright_core_color_integrity_unresolved",
            upstream_separation_state=separation_state,
        )
        return
    if separation_state in {
        StarSeparationState.REJECTED.value,
        StarSeparationState.TOOL_FAILED.value,
    }:
        _run_rejected_stage6_with_stars_candidates(
            pipeline,
            separation_state,
            reason_code=(
                "bright_core_starless_rejected_after_recovery"
                if bright_core_review_only
                else None
            ),
        )
        return
    pipeline._stage7_stretch_source = (
        "stage6_passthrough"
        if separation_state == StarSeparationState.TARGET_BYPASS.value
        else "stage6_starless"
    )
    stretched, stage_degraded, stretch_messages, stretch_method = (
        pipeline._run_stage7_stretching_candidates()
    )
    messages.extend(stretch_messages)
    if stretch_method:
        messages.append(f"拉伸使用 {stretch_method}")
    if bool(getattr(pipeline, "_stage7_stretch_forced_delivery", False)):
        stretched = False
        stage_degraded = True
        messages.append(
            "legacy forced-delivery candidate reclassified as review evidence; "
            "formal Stage7 acceptance withheld"
        )

    if bool(getattr(pipeline, "_stage7_destructive_core_rejected", False)):
        revoked_pair_id = getattr(pipeline, "_stage7_revoked_pair_id", None)
        removed_artifacts = syqon_starless.purge_unaccepted_star_separation_outputs(
            pipeline
        )
        pipeline._star_separation_state = StarSeparationState.REJECTED.value
        pipeline._stage7_starless_skipped = True
        pipeline._stage6_passthrough_source = "stage6_input"
        pipeline._stage6_pair_handoff = None
        pipeline._stage7_matched_domain_transfer = None
        handoff = dict(getattr(pipeline, "_stage8_handoff", {}) or {})
        handoff.update(
            {
                "requested_policy": "skip",
                "processing_policy": "skip",
                "source_stage": 7,
                "source_stem": "stage7_review_with_stars",
                "passthrough": True,
                "restricted_downstream": True,
                "reason_code": "stage7_bright_core_integrity_rejected",
                "reason_text": (
                    "all Stage7 candidates failed non-overridable "
                    "bright-core integrity gates"
                ),
                "reasons": [
                    {
                        "code": "stage7_bright_core_integrity_rejected",
                        "source_stage": 7,
                        "revoked_pair_id": revoked_pair_id,
                        "gates": list(
                            getattr(
                                pipeline,
                                "_stage7_bright_core_integrity_rejected_reasons",
                                [],
                            )
                            or []
                        ),
                    }
                ],
                "quality_status": "rejected",
            }
        )
        pipeline._stage8_handoff = handoff
        if removed_artifacts:
            pipeline.log.info(
                "Stage7 bright-core rejection purged Starless pair artifacts: "
                + ", ".join(sorted(removed_artifacts))
            )
        _run_with_stars_review_stretch(
            pipeline,
            StarSeparationState.REJECTED.value,
            source_stem="stage6_input",
            reason_code="stage7_bright_core_integrity_rejected",
            upstream_separation_state=separation_state,
            candidates_evaluated=True,
        )
        return

    if not stretched and failure_action == "auto_fallback":
        target_bypass_review = bool(
            separation_state == StarSeparationState.TARGET_BYPASS.value
        )
        review_input = (
            "stage6_passthrough"
            if target_bypass_review
            else "stage6_input"
        )
        if not target_bypass_review:
            pipeline._star_separation_state = StarSeparationState.REJECTED.value
        pipeline._stage7_starless_skipped = True
        pipeline._stage6_passthrough_source = review_input
        pipeline._stage6_pair_handoff = None
        pipeline._stage7_matched_domain_transfer = None
        pipeline.starless_file = None
        pipeline.starmask_file = None
        handoff = dict(getattr(pipeline, "_stage8_handoff", {}) or {})
        handoff.update(
            {
                "requested_policy": "skip",
                "processing_policy": "skip",
                "source_stage": 7,
                "source_stem": "stage7_review_with_stars",
                "passthrough": True,
                "restricted_downstream": True,
                "reason_code": (
                    "stage7_stretch_not_accepted_target_bypass"
                    if target_bypass_review
                    else "stage7_stretch_not_accepted"
                ),
                "reason_text": (
                    "all Stage7 stretch candidates failed formal quality gates"
                ),
                "quality_status": "rejected",
            }
        )
        pipeline._stage8_handoff = handoff
        _run_with_stars_review_stretch(
            pipeline,
            (
                StarSeparationState.TARGET_BYPASS.value
                if target_bypass_review
                else StarSeparationState.REJECTED.value
            ),
            source_stem=review_input,
            reason_code=(
                "stage7_stretch_not_accepted_target_bypass"
                if target_bypass_review
                else "stage7_stretch_not_accepted"
            ),
            upstream_separation_state=separation_state,
            candidates_evaluated=True,
        )
        return

    compare_stem = pipeline._stage7_stretch_source
    failure_policy_triggered = bool(
        not stretched and failure_action != "auto_fallback"
    )
    if failure_policy_triggered:
        pipeline._require_review(7, "no_stretch_candidate_passed_quality_gate")
        if hasattr(pipeline, "_record_stage_policy_event"):
            pipeline._record_stage_policy_event(
                7,
                event="candidate_search_stopped",
                reason="no enabled stretch candidate passed all hard gates",
                source="stretch_quality_gate",
            )
        if failure_action == "preserve_review":
            try:
                pipeline.cmd_with_check("load", compare_stem)
                if pipeline._save_stage_output("stage7_review_preserved"):
                    pipeline._stage7_review_source = "stage7_review_preserved"
                    messages.append(
                        "Stage7 immutable input preserved as review output"
                    )
            except (CommandError, SirilError) as error:
                messages.append(
                    "Stage7 preserve_review output failed: "
                    f"{pipeline._short_text(error, 160)}"
                )
    pipeline.stretched_name = "stage7_stretched"
    # 拉伸后必须保存，后续 Stage8/9 需要按名加载。
    stage_saved = pipeline._save_stage_output(pipeline.stretched_name) if stretched else False
    presentation_reference = (
        _bind_stage7_formal_presentation_reference(
            pipeline,
            pipeline.stretched_name,
        )
        if stage_saved
        else dict(
            getattr(
                pipeline,
                "_stage7_presentation_reference_report",
                {},
            )
            or {}
        )
    )
    presentation_contract_required = bool(
        presentation_reference.get("status") != "not_run"
    )
    presentation_reference_ready = bool(
        not presentation_contract_required
        or (
            presentation_reference.get("status") == "ready"
            and presentation_reference.get("accepted") is True
        )
    )
    if stage_saved and not presentation_reference_ready:
        pipeline._require_review(
            7,
            "stage7_presentation_reference_unavailable",
        )
        messages.append(
            "Stage7 formal source was saved but its presentation reference "
            "binding failed; formal acceptance withheld"
        )
    # A degraded result remains unsafe unless the stretch service explicitly
    # marks it as a rescue that re-passed every Stage 7 quality gate.
    validated_rescue = bool(
        getattr(pipeline, "_stage7_stretch_validated_rescue", False)
    )
    forced_delivery = bool(
        getattr(pipeline, "_stage7_stretch_forced_delivery", False)
    )
    fallback_reason = str(
        getattr(pipeline, "_stage7_stretch_fallback_reason", "") or ""
    )
    if validated_rescue and not fallback_reason:
        fallback_reason = "validated_stretch_fallback"
    pipeline._stage7_stretch_accepted = bool(
        stretched
        and stage_saved
        and presentation_reference_ready
        and (not stage_degraded or validated_rescue)
        and not forced_delivery
    )
    if pipeline._stage7_stretch_accepted:
        pipeline._stage7_stretch_output = pipeline.stretched_name
    if forced_delivery and pipeline._stage7_stretch_accepted:
        pipeline._stage8_conservative_mode = True
        handoff = dict(getattr(pipeline, "_stage8_handoff", {}) or {})
        handoff.update(
            processing_policy="background_only",
            requested_policy="background_only",
            reason_code="stage7_forced_quality_delivery",
            reason_text=(
                "Stage7 forced delivery retained a technically safe image; "
                "skip further positive colour enhancement"
            ),
            reasons=list(
                getattr(pipeline, "_stage7_forced_delivery_reasons", []) or []
            ),
        )
        pipeline._stage8_handoff = handoff
    if stage_saved:
        diff_note = pipeline._stage_diff_note(pipeline.stretched_name, compare_stem)
        if diff_note:
            messages.append(diff_note)
        if hasattr(pipeline, "_create_stage_review_bundle"):
            review = pipeline._create_stage_review_bundle(
                "stage7_stretching",
                compare_stem,
                pipeline.stretched_name,
                context={"stretch_method": stretch_method},
                candidates=getattr(pipeline, "_stage7_stretch_candidates", []),
                selected_candidate=getattr(pipeline, "_stage7_stretch_selected", None),
            )
            if review.get("report_path"):
                messages.append(f"review_bundle={review['report_path']}")
        feature_note = pipeline._feature_summary_note("拉伸后特征")
        if feature_note:
            pipeline.log.info(f"[Stage7] {feature_note}")
            feat = pipeline._measure_current_features()
            if feat is not None:
                messages.append(
                    "stage7_features="
                    f"bg_median={feat.bg_median:.4f}, "
                    f"object_area={feat.object_area_ratio:.3f}, "
                    f"edge_black={feat.edge_black_ratio:.3f}"
                )

    background_color_review_gate = dict(
        getattr(pipeline, "_stage7_background_color_review_gate", {}) or {}
    )
    elapsed = pipeline.log.stage_end(stage_label)
    message_text = "；".join(messages)
    if stretched and stage_saved and pipeline._stage7_stretch_accepted:
        status = 'degraded' if forced_delivery else (
            'degraded' if stage_degraded and not validated_rescue else 'ok'
        )
        pipeline._record_stage(
            stage_label,
            status,
            elapsed,
            message_text,
            fallback_used=bool(validated_rescue or forced_delivery),
            reason_code=(
                fallback_reason
                if validated_rescue or forced_delivery
                else ""
            ),
            components={
                "stretch": {
                    "status": "accepted",
                    "method": stretch_method or "unknown",
                    "source": compare_stem,
                    "output": pipeline.stretched_name,
                    "reason_code": (
                        fallback_reason
                        if validated_rescue or forced_delivery
                        else "accepted"
                    ),
                    "fallback_used": bool(
                        validated_rescue or forced_delivery
                    ),
                    "forced_delivery": forced_delivery,
                }
            },
            details={
                "background_color_review_gate": background_color_review_gate,
                "presentation_reference": presentation_reference,
                "background_color_review_required": bool(
                    getattr(
                        pipeline,
                        "_stage7_background_color_review_required",
                        False,
                    )
                ),
                "forced_delivery": forced_delivery,
                "forced_delivery_reasons": list(
                    getattr(
                        pipeline,
                        "_stage7_forced_delivery_reasons",
                        [],
                    )
                    or []
                ),
                "bright_core_with_stars_fallback": dict(
                    getattr(
                        pipeline,
                        "_bright_core_with_stars_fallback",
                        {},
                    )
                    or {}
                ),
            },
            review_reasons=pipeline._stage_review_reasons(7),
        )
    elif stretched and stage_saved and not presentation_reference_ready:
        if message_text:
            message_text = (
                f"{message_text}；Stage7 表现参照或正式源绑定失败，"
                "仅允许复核"
            )
        else:
            message_text = "Stage7 表现参照或正式源绑定失败，仅允许复核"
        pipeline._record_stage(
            stage_label,
            "degraded",
            elapsed,
            message_text,
            execution="safe_passthrough",
            reason_code="stage7_presentation_reference_unavailable",
            details={
                "background_color_review_gate": background_color_review_gate,
                "presentation_reference": presentation_reference,
            },
            review_reasons=pipeline._stage_review_reasons(7),
        )
    elif stretched and not stage_saved:
        if message_text:
            message_text = f"{message_text}；stage7 输出保存失败"
        else:
            message_text = "stage7 输出保存失败"
        pipeline._record_stage(
            stage_label,
            'degraded',
            elapsed,
            message_text,
            reason_code="stage7_output_save_failed",
            details={
                "background_color_review_gate": background_color_review_gate,
            },
        )
    elif getattr(pipeline, "_stage7_review_source", None):
        if message_text:
            message_text = f"{message_text}；仅保留 Stage7 复核候选，禁止正式交付"
        else:
            message_text = "仅保留 Stage7 复核候选，禁止正式交付"
        pipeline._record_stage(
            stage_label,
            "failed" if failure_action == "stop" else "degraded",
            elapsed,
            message_text,
            execution="safe_passthrough",
            reason_code=(
                "failure_policy_stop"
                if failure_action == "stop"
                else "failure_policy_preserve_review"
                if failure_policy_triggered
                else "no_stretch_candidate_passed_quality_gate"
            ),
            upstream_passthrough=failure_policy_triggered,
            details={
                "background_color_review_gate": background_color_review_gate,
                "candidate_policy": str(
                    getattr(
                        pipeline.cfg,
                        "stage7_candidate_policy",
                        "auto_display90",
                    )
                ),
                "failure_action": failure_action,
            },
            review_reasons=(
                pipeline._stage_review_reasons(7)
                if failure_action != "stop"
                else []
            ),
        )
    else:
        if message_text:
            message_text = f"{message_text}；所有拉伸方法均失败"
        else:
            message_text = "所有拉伸方法均失败"
        pipeline._record_stage(
            stage_label,
            'failed',
            elapsed,
            message_text,
            reason_code=(
                "failure_policy_stop"
                if failure_policy_triggered and failure_action == "stop"
                else "failure_policy_preserve_review_restore_failed"
                if failure_policy_triggered
                else "all_stretch_candidates_failed"
            ),
            details={
                "background_color_review_gate": background_color_review_gate,
                "candidate_policy": str(
                    getattr(
                        pipeline.cfg,
                        "stage7_candidate_policy",
                        "auto_display90",
                    )
                ),
                "failure_action": failure_action,
            },
        )
