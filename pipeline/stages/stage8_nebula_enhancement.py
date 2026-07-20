"""Stage 8 starless nebula enhancement."""
from typing import Any, Dict, List, Optional

from sirilpy.exceptions import CommandError, SirilError

from image_metrics import format_feature_summary
from models import PipelineStage
from pipeline_safety import color_safety_limits, clamp_saturation_boost


def _load_stage8_input(pipeline, messages: List[str]) -> str:
    """Load this run's accepted Stage 7 output, then conservative fallbacks."""
    preferred = str(
        getattr(pipeline, "_stage7_stretch_output", None)
        or getattr(pipeline, "stretched_name", None)
        or "stage7_stretched"
    )
    stage7_accepted = bool(getattr(pipeline, "_stage7_stretch_accepted", False))
    candidates: List[str] = []
    if stage7_accepted:
        preferred_path = pipeline.process_dir / f"{preferred}.fit"
        if preferred_path.exists():
            candidates.append(preferred)
        else:
            messages.append(
                f"stage8 preferred Stage7 input missing: {preferred}.fit; using fallback"
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
    pipeline._stage8_final_source = "starless_enhanced"
    pipeline._stage8_fallback_used = False
    pipeline._stage8_final_quality = "unknown"
    pipeline._stage8_input_source = None
    pipeline._stage8_input_fallback_used = False
    stage8_initial_quality = "unknown"
    stage8_reject_reason = ""
    color_limits = color_safety_limits(
        getattr(pipeline, "pipeline_policy", {}) or {},
        getattr(pipeline, "color_calibration_report", {}) or {},
    )
    effective_stage8_saturation = 0.0
    stage8_ai_plan_applied = False
    if bool(getattr(pipeline, "_stage8_conservative_mode", False)):
        messages.append("stage8 conservative mode requested by Stage7 starless repair/quality gate")

    if (
        bool(getattr(pipeline, "_star_preserve_target_bypass", False))
        and bool(getattr(pipeline, "_stage7_stretch_accepted", False))
    ):
        source_stem = pipeline.stretched_name or "stage7_stretched"
        try:
            pipeline.cmd_with_check("load", source_stem)
            pipeline._stage8_input_source = source_stem
            pipeline._stage8_input_fallback_used = False
            saved = pipeline._save_stage_output("starless_enhanced")
            stage8_saved = pipeline._save_stage_output("stage8_enhanced") if saved else False
            pipeline._stage8_final_source = "stage8_enhanced" if stage8_saved else source_stem
            pipeline._stage8_final_quality = "star_preserve_bypass"
            pipeline._stage8_fallback_used = False
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
                pipeline.starless_file = pipeline.process_dir / "starless.fit"
                pipeline.log.info(f"已导入外部 Starless: {external_starless.name}")
                messages.append(f"导入外部 Starless: {external_starless.name}")
        except (OSError, CommandError, SirilError) as e:
            pipeline.log.warn(f"导入外部 Starless 失败，继续使用本地 starless: {e}")
            messages.append(f"外部 Starless 导入失败: {pipeline._short_text(e, 160)}")

    pipeline.log.info("加载 Stage8 输入图像...")
    stage8_input_source = _load_stage8_input(pipeline, messages)
    pipeline._write_stage_json(
        "stage8_input_selection.json",
        {
            "stage": "stage8_input_selection",
            "stage7_accepted": bool(
                getattr(pipeline, "_stage7_stretch_accepted", False)
            ),
            "preferred_source": (
                getattr(pipeline, "_stage7_stretch_output", None)
                or getattr(pipeline, "stretched_name", None)
                or "stage7_stretched"
            ),
            "selected_source": stage8_input_source,
            "fallback_used": bool(pipeline._stage8_input_fallback_used),
        },
    )
    pipeline._last_stage8_masked_diagnostics = {}
    stage8_quality_enabled = pipeline._ai_stage_advisory_enabled("ai_stage8_enabled")
    stage8_masked_enabled = bool(
        getattr(pipeline.cfg, "stage8_masked_enhancement_enabled", False)
    )
    stage8_diagnostics_enabled = (
        stage8_quality_enabled or stage8_masked_enabled
    )
    stage8_processing_plan: Optional[Dict[str, Any]] = None
    stage8_guard_report: Dict[str, Any] = {}
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
            if stage8_guard_report.get("skip_enhancement"):
                guard_reasons = [
                    str(item) for item in stage8_guard_report.get("reasons", []) if item
                ]
                conservative_skip = bool(
                    getattr(pipeline, "_stage8_conservative_mode", False)
                    or stage8_guard_report.get("conservative_mode", False)
                    or "stage8_conservative_mode_after_stage7_starless_repair"
                    in guard_reasons
                )
                guard_status = (
                    "conservative_skipped" if conservative_skip else "skipped"
                )
                pipeline._stage8_final_source = "stage8_input_starless"
                pipeline._stage8_fallback_used = True
                pipeline._stage8_final_quality = guard_status
                pipeline.cmd_with_check("load", "stage8_input_starless")
                pipeline._save_stage_output("starless_enhanced")
                pipeline._save_stage_output("stage8_enhanced")
                pipeline.starless_file = pipeline.process_dir / "stage8_input_starless.fit"
                stage8_reject_reason = ", ".join(guard_reasons[:3])
                messages.append(
                    "stage8 enhancement skipped by input guard: "
                    + (stage8_reject_reason or "unsafe starless input")
                )
                guard_payload = {
                    **stage8_guard_report,
                    "stage": "stage8_input_guard",
                    "status": guard_status,
                    "input_source": stage8_input_source,
                    "final_source": pipeline._stage8_final_source,
                    "fallback_used": pipeline._stage8_fallback_used,
                    "final_quality": pipeline._stage8_final_quality,
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
                pipeline._record_stage(
                    stage_label,
                    "ok" if conservative_skip else "degraded",
                    elapsed,
                    "；".join(messages),
                )
                return
            if stage8_quality_enabled:
                stage8_processing_plan = pipeline._request_stage8_processing_plan()
                if stage8_processing_plan:
                    plan_note = (
                        "AI stage8 processing params applied "
                        f"(sat={stage8_processing_plan['saturation']:.4f}, "
                        f"bg={stage8_processing_plan['bg_factor']}, "
                        f"unsharp={stage8_processing_plan['unsharp_radius']:.3f}/"
                        f"{stage8_processing_plan['unsharp_amount']:.3f}, "
                        f"after_plugins={stage8_processing_plan['apply_after_plugins']})"
                    )
                    if stage8_processing_plan.get("summary"):
                        plan_note += f": {stage8_processing_plan['summary']}"
                    messages.append(plan_note)
            if stage8_high_risk:
                stage8_processing_plan = {
                    **(stage8_processing_plan or {}),
                    "saturation": min(
                        float((stage8_processing_plan or {}).get("saturation", 0.0) or 0.0),
                        0.05,
                    ),
                    "bg_factor": 0,
                    "unsharp_radius": 0.0,
                    "unsharp_amount": 0.0,
                    "apply_after_plugins": False,
                }
                messages.append(
                    "stage8 high halo risk guard: skip global WaveScale/DarkEnhancer; "
                    "object-mask-only weak enhancement"
                )
        else:
            messages.append("stage8_input_starless 保存失败，跳过阶段8质量诊断")
            stage8_quality_enabled = False
            stage8_diagnostics_enabled = False

    stage8_ai_plan_applied = stage8_processing_plan is not None
    requested_saturation = float(
        (stage8_processing_plan or {}).get(
            "saturation",
            pipeline.cfg.nebula_saturation,
        )
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
    sasp_api_used = None if stage8_high_risk else pipeline._run_sasp_stage8_api(stage8_processing_plan)
    if sasp_api_used:
        processed = True
        messages.append(f"SASP Starless 深加工使用 {sasp_api_used}")
    elif pipeline._last_sasp_stage8_error:
        messages.append(
            "SASP Starless 深加工 API 不可用: "
            f"{pipeline._short_text(pipeline._last_sasp_stage8_error, 160)}"
        )

    if not sasp_api_used and pipeline.cfg.workflow_plugin_probe_enabled:
        messages.append("SASP Siril 深加工命令不可用，跳过实验性 sasp_* 命令探测")

    if pipeline.cfg.optional_color_transform_enabled:
        pipeline._run_first_available_command(
            "调色1（可选）",
            [
                ("SASP Selective Color Correction", ("sasp_selective_color_correction",)),
                ("Selective Color Correction", ("selective_color_correction",)),
            ],
        )

    if processed and stage8_processing_plan and stage8_processing_plan.get("apply_after_plugins"):
        try:
            messages.extend(
                pipeline._apply_stage8_builtin_enhancement(
                    stage8_processing_plan,
                    label="AI supplemental",
                )
            )
        except (CommandError, SirilError) as e:
            pipeline.log.warn(f"AI supplemental Starless enhancement skipped: {e}")
            messages.append(
                "AI supplemental Starless enhancement failed: "
                f"{pipeline._short_text(e, 160)}"
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
                    label="AI" if stage8_ai_plan_applied else "内置",
                )
            )
            pipeline.log.info("Starless 内置增强链完成")
        except (CommandError, SirilError) as e:
            pipeline.log.warn(f"Starless 内置增强跳过: {e}")
            status = 'degraded'
            messages.append(f"Starless 内置增强失败: {pipeline._short_text(e, 160)}")

    stage_saved = pipeline._save_stage_output("starless_enhanced")
    if stage_saved:
        stage8_saved = pipeline._save_stage_output("stage8_enhanced")
        if not stage8_saved and status == 'ok':
            status = 'degraded'
            messages.append("stage8 输出保存失败")
        else:
            diff_note = pipeline._stage_diff_note("stage8_enhanced", "stage8_input_starless")
            if diff_note:
                messages.append(diff_note)
            feature = pipeline._measure_current_features()
            if feature:
                messages.append(f"Starless 后特征: {format_feature_summary(feature)}")
                blue_guard_note = pipeline._apply_starless_blue_guard(feature)
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
                                pipeline._save_stage_output("starless_enhanced")
                            except (CommandError, SirilError) as e:
                                status = 'degraded'
                                messages.append(
                                    "Starless 蓝色门控回滚失败: "
                                    f"{pipeline._short_text(e, 160)}"
                                )
                        else:
                            pipeline._save_stage_output("starless_enhanced")
                            pipeline._save_stage_output("stage8_enhanced")
                    else:
                        pipeline._save_stage_output("starless_enhanced")
                        pipeline._save_stage_output("stage8_enhanced")

            if stage8_diagnostics_enabled:
                quality_record = pipeline._stage8_quality_assessment()
                stage8_initial_quality = str(quality_record.get("status", "unknown"))
                quality_payload: Dict[str, Any] = {
                    "initial": quality_record,
                    "final": quality_record,
                    "mode": (
                        "masked_parameter_optimization"
                        if stage8_quality_enabled
                        else "masked_local_diagnostics"
                    ),
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
                        correction_note = pipeline._apply_stage8_color_correction_from_quality(
                            quality_record
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
                                        "AI stage8 extra blue correction applied "
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
                        else:
                            messages.append("stage8_quality did not require color correction")
                    except (CommandError, SirilError) as e:
                        messages.append(
                            "AI stage8 color correction failed: "
                            f"{pipeline._short_text(e, 160)}"
                        )
                    if (
                        latest_quality.get("status") != "ok"
                        and pipeline._stage8_needs_conservative_rerun(latest_quality)
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
                pipeline._write_stage_json("stage8_quality.json", quality_payload)
            enhancement_report = (
                pipeline._stage8_enhancement_quality_report()
                if hasattr(pipeline, "_stage8_enhancement_quality_report")
                else {}
            )
            if enhancement_report:
                enhancement_report["conservative_rerun_applied"] = bool(
                    "conservative_rerun" in locals().get("quality_payload", {})
                )
                if enhancement_report.get("status") == "poor":
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
                        status = "degraded" if status == "ok" else status
                        pipeline._stage8_final_source = "stage8_input_starless"
                        pipeline._stage8_fallback_used = True
                        pipeline._stage8_final_quality = "poor"
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
                pipeline._write_stage_json("stage8_enhancement_report.json", enhancement_report)
    elif status == 'ok':
        status = 'degraded'
        messages.append("starless_enhanced 输出保存失败")
    if pipeline._stage8_final_quality == "unknown":
        pipeline._stage8_final_quality = "ok" if status == "ok" else status
    summary_fields = {
        "stage8_input_source": stage8_input_source,
        "stage8_input_fallback_used": str(
            bool(pipeline._stage8_input_fallback_used)
        ).lower(),
        "stage8_initial_quality": stage8_initial_quality,
        "stage8_final_quality": pipeline._stage8_final_quality,
        "stage8_final_source": pipeline._stage8_final_source,
        "stage8_fallback_used": str(pipeline._stage8_fallback_used).lower(),
        "stage8_conservative_mode": str(bool(getattr(pipeline, "_stage8_conservative_mode", False))).lower(),
        "stage8_reject_reason": stage8_reject_reason or "none",
    }
    for key, value in summary_fields.items():
        line = f"{key}={value}"
        pipeline.log.info(f"[Stage8] {line}")
        messages.append(line)
    pipeline.starless_file = pipeline.process_dir / f"{pipeline._stage8_final_source}.fit"
    applied_saturation = effective_stage8_saturation
    safe_sat = locals().get("safe_sat")
    if isinstance(safe_sat, (int, float)):
        applied_saturation = min(applied_saturation, float(safe_sat))
    if pipeline._stage8_final_source == "stage8_input_starless":
        applied_saturation = 0.0
    pipeline._saturation_boost_applied = float(
        getattr(pipeline, "_saturation_boost_applied", 0.0)
    ) + max(0.0, applied_saturation)
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
            },
            candidates=stage8_candidates,
            selected_candidate="final",
        )
        if review.get("report_path"):
            messages.append(f"review_bundle={review['report_path']}")

    elapsed = pipeline.log.stage_end(stage_label)
    pipeline._record_stage(stage_label, status, elapsed, "；".join(messages))
