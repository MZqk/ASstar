"""Stage 7 star separation and star-mask preparation."""
from typing import Any, Dict, List, Optional, Tuple

from sirilpy.exceptions import CommandError, SirilError


def run_stage7_star_separation(pipeline) -> None:
    """
    阶段 7: 星点分离
    - 优先 SyQon-Starless.py / SASP Dark Star
    - 生成并导出 starless/starmask 交换文件供外部工具使用
    """
    pipeline.log.stage_start("阶段 7: 去星与星点层准备")
    pipeline._stage7_starless_skipped = False
    pipeline._stage8_conservative_mode = False
    gate_report = getattr(pipeline, "pre_starless_gate_report", {}) or {}
    recommended = str(gate_report.get("recommended_starless_input") or "").strip()
    if recommended.endswith(".fit"):
        recommended = recommended[:-4]
    if recommended and pipeline.process_dir and (pipeline.process_dir / f"{recommended}.fit").exists():
        previous_stretched = pipeline.stretched_name or "stage6_stretched"
        if recommended != previous_stretched:
            pipeline.log.info(
                "[Stage7] Using pre-starless recommended input: "
                f"{recommended} (previous={previous_stretched})"
            )
            pipeline.stretched_name = recommended

    if (
        gate_report
        and not bool(gate_report.get("ready_for_starless", True))
        and bool(getattr(pipeline.cfg, "stage7_skip_unready_starless", True))
    ):
        source_stem = pipeline.stretched_name or "stage6_stretched"
        reasons = [str(item) for item in gate_report.get("reason", []) if item]
        message_parts = [
            "Stage6.5 ready_for_starless=false; skipped SyQon/SASP starless",
            f"source={source_stem}",
        ]
        if reasons:
            message_parts.append("reason=" + ", ".join(reasons[:3]))
        try:
            pipeline.cmd_with_check("load", source_stem)
            pipeline.cmd_with_check("save", "starless")
            pipeline.starless_file = pipeline.process_dir / "starless.fit"
            pipeline.starmask_file = None
            pipeline._stage7_starless_skipped = True
            quality_record = {
                "attempt": "skipped_by_pre_starless_gate",
                "tool_label": "none",
                "status": "skipped",
                "issues": reasons or ["pre-starless gate marked input unsafe"],
                "derived": {
                    "residual_star_score": 0.0,
                    "halo_residue_score": 0.0,
                    "starmask_contamination": 0.0,
                },
            }
            stage9_record = pipeline._stage7_update_star_remix_from_quality(quality_record)
            pipeline._write_stage_json(
                "stage7_quality.json",
                {
                    "attempts": [quality_record],
                    "selected": quality_record,
                    "mode": "skipped_by_pre_starless_gate",
                    "selected_source_stem": source_stem,
                    "pre_starless_gate": gate_report,
                    "stage9_star_remix": stage9_record,
                    "retry_max": 0,
                },
            )
            pipeline._export_sasp_exchange_files()
            stage_saved = pipeline._save_stage_output("stage7_starless")
            if not stage_saved:
                message_parts.append("stage7 输出保存失败")
            elapsed = pipeline.log.stage_end("阶段 7: 去星与星点层准备")
            pipeline._record_stage(
                "阶段 7: 去星与星点层准备",
                "skipped" if stage_saved else "degraded",
                elapsed,
                "；".join(message_parts),
            )
            return
        except (CommandError, SirilError) as e:
            pipeline.log.warn(f"Stage7 gate skip fallback failed, continuing with starless tools: {e}")

    pipeline.log.info("执行去星流程...")
    try:
        pipeline.starless_file = None
        pipeline.starmask_file = None
        syqon_failure_reason: Optional[str] = None
        stage_messages: List[str] = []
        quality_records: List[Dict[str, Any]] = []
        selected_quality: Optional[Dict[str, Any]] = None
        starless_used = None
        stage7_plan: Optional[Dict[str, Any]] = None
        stage7_preflight: Optional[Dict[str, Any]] = None
        starmask_cleanup_records: List[Dict[str, Any]] = []
        repair_records: List[Dict[str, Any]] = []
        starless_pixel_repair_records: List[Dict[str, Any]] = []
        conservative_input_records: List[Dict[str, Any]] = []
        selected_source_stem = pipeline.stretched_name or "stage6_stretched"

        stage7_preflight = pipeline._stage7_preflight_check()
        preflight_summary = pipeline._stage7_preflight_summary(stage7_preflight)
        if stage7_preflight.get("risk_level") != "ok":
            stage_messages.append(preflight_summary)

        syqon_script = pipeline._find_plugin_script(("processing/SyQon-Starless.py",))
        if syqon_script is not None:
            stage7_plan = pipeline._request_stage7_starless_plan()
            plan_tile_size = 512
            plan_overlap = 64
            plan_axiom = False
            if stage7_plan:
                plan_tile_size = int(stage7_plan["tile_size"])
                plan_overlap = int(stage7_plan["overlap"])
                plan_axiom = bool(stage7_plan["use_axiom"])
                plan_note = (
                    "AI stage7 SyQon params applied "
                    f"(tile={plan_tile_size}, overlap={plan_overlap}, "
                    f"axiom={plan_axiom})"
                )
                if stage7_plan.get("summary"):
                    plan_note += f": {stage7_plan['summary']}"
                stage_messages.append(plan_note)

            _syqon_args, _syqon_timeout, syqon_device_note = pipeline._syqon_starless_cli_options(
                tile_size=plan_tile_size,
                overlap=plan_overlap,
                axiom=plan_axiom,
            )
            pipeline.log.info(
                "优先尝试 SyQon-Starless.py CLI 子进程"
                f"（{syqon_device_note}）"
            )
            syqon_used = pipeline._stage7_try_syqon_variant(
                syqon_script,
                attempt_name="initial",
                tile_size=plan_tile_size,
                overlap=plan_overlap,
                axiom=plan_axiom,
            )
            if syqon_used:
                syqon_model_note = "Axiom v2" if plan_axiom else "Zenith v1"
                stage_messages.append(f"SyQon model: {syqon_model_note} ({syqon_device_note})")
                if pipeline.starmask_file:
                    pipeline.log.info("SyQon 产物已归一化: starless.fit, starmask.fit")
                else:
                    pipeline.log.info("SyQon 产物已归一化: starless.fit")
                starless_used = syqon_used
            else:
                syqon_failure_reason = (
                    getattr(pipeline, "_last_plugin_script_error", None)
                    or f"SyQon 脚本执行失败: {syqon_script.name}"
                )
        else:
            syqon_failure_reason = "SyQon-Starless.py 缺失，回退到 SASP Dark Star"

        if not starless_used and syqon_failure_reason:
            pipeline.log.warn(syqon_failure_reason)

        if not starless_used:
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
                        "SyQon-Starless.py",
                        syqon_failure_reason or "SyQon unavailable",
                        starless_used,
                        True,
                    )
                )
        if not starless_used:
            raise SirilError("未找到可用去星命令")

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
            stage_messages.append(
                "stage7 starmask cleanup applied "
                f"(signal_ratio={float(metrics.get('signal_ratio', 1.0)):.3f})"
            )

        quality_enabled = pipeline._ai_stage_advisory_enabled("ai_stage7_enabled")
        selected_quality = pipeline._stage7_quality_assessment(
            "initial",
            tool_label=str(starless_used or "unknown"),
            use_ai=quality_enabled,
            source_stem=selected_source_stem,
        )
        quality_records.append(selected_quality)
        best_quality = selected_quality
        best_snapshot = (
            pipeline._stage7_snapshot_current_outputs("ai_best_initial")
            if quality_enabled or pipeline.cfg.stage7_conservative_repair_enabled
            else None
        )
        best_label = "initial"
        best_source_stem = selected_source_stem

        if selected_quality["status"] != "ok":
            stage_messages.append(
                "stage7_quality diagnostic: "
                + ", ".join(selected_quality.get("issues", [])[:3])
            )

        repair_triggers = pipeline._stage7_repair_triggers(selected_quality)
        repair_attempts_done = 0
        if (
            repair_triggers
            and pipeline.cfg.stage7_conservative_repair_enabled
            and syqon_script is not None
            and pipeline.cfg.stage7_quality_retry_max > 0
        ):
            conservative_input_records = pipeline._stage7_build_conservative_starless_inputs()
            candidates = pipeline._stage7_conservative_input_candidates()
            if candidates:
                stage_messages.append(
                    "stage7_quality triggers conservative starless retry: "
                    + ",".join(repair_triggers)
                )
            for source_candidate in candidates:
                if repair_attempts_done >= max(1, int(pipeline.cfg.stage7_quality_retry_max)):
                    break
                repair_attempts_done += 1
                attempt_name = (
                    f"conservative_repair_{repair_attempts_done}_"
                    f"{source_candidate}"
                )
                used = pipeline._stage7_try_syqon_with_source(
                    syqon_script,
                    source_stem=source_candidate,
                    attempt_name=attempt_name,
                    tile_size=int(stage7_plan.get("tile_size", 512)) if stage7_plan else 512,
                    overlap=int(stage7_plan.get("overlap", 64)) if stage7_plan else 64,
                    axiom=bool(stage7_plan.get("use_axiom", False)) if stage7_plan else False,
                )
                if not used:
                    repair_record = {
                        "attempt": attempt_name,
                        "source_stem": source_candidate,
                        "status": "failed",
                        "triggers": repair_triggers,
                        "issues": [
                            getattr(
                                pipeline,
                                "_last_plugin_script_error",
                                "conservative starless retry failed",
                            )
                        ],
                    }
                    repair_records.append(repair_record)
                    quality_records.append(repair_record)
                    continue
                retry_cleanup = pipeline._stage7_clean_starmask(
                    label=attempt_name,
                    source_stem=source_candidate,
                )
                starmask_cleanup_records.append(retry_cleanup)
                retry_quality = pipeline._stage7_quality_assessment(
                    attempt_name,
                    tool_label=used,
                    use_ai=quality_enabled,
                    source_stem=source_candidate,
                )
                quality_records.append(retry_quality)
                retry_snapshot = pipeline._stage7_snapshot_current_outputs(
                    f"ai_best_{attempt_name}"
                )
                retry_score = pipeline._stage7_quality_score(retry_quality)
                best_score = pipeline._stage7_quality_score(best_quality)
                repair_record = {
                    "attempt": attempt_name,
                    "source_stem": source_candidate,
                    "status": "ok",
                    "triggers": repair_triggers,
                    "score": retry_score,
                    "selected": retry_score < best_score,
                }
                repair_records.append(repair_record)
                if retry_score < best_score:
                    best_quality = retry_quality
                    best_snapshot = retry_snapshot
                    best_label = attempt_name
                    best_source_stem = source_candidate

            if repair_attempts_done > 0 and best_snapshot is not None:
                pipeline._stage7_restore_snapshot(best_snapshot)
                selected_quality = best_quality
                selected_source_stem = best_source_stem
                stage_messages.append(
                    "stage7_quality selected repaired starless "
                    f"({best_label}, source={selected_source_stem}, "
                    f"score={pipeline._stage7_quality_score(best_quality):.3f})"
                )

        if quality_enabled and selected_quality["status"] != "ok":
            retry_limit = max(
                0,
                int(pipeline.cfg.stage7_quality_retry_max) - repair_attempts_done,
            )
            if retry_limit <= 0:
                stage_messages.append("stage7_quality refinement skipped: retry budget consumed")
            elif syqon_script is not None:
                stage_messages.append("stage7_quality triggers SyQon parameter refinement")
                variants: List[Tuple[int, int, bool]] = [
                    (512, 96, False),
                    (1024, 64, False),
                    (1024, 96, False),
                ]
                if pipeline._syqon_axiom_model_available():
                    variants.append((512, 64, True))

                seen_variants = {
                    (
                        int(stage7_plan.get("tile_size", 512)) if stage7_plan else 512,
                        int(stage7_plan.get("overlap", 64)) if stage7_plan else 64,
                        bool(stage7_plan.get("use_axiom", False)) if stage7_plan else False,
                    )
                }
                retries_done = 0
                for tile_size, overlap, axiom in variants:
                    if retries_done >= retry_limit:
                        break
                    variant_key = (tile_size, overlap, axiom)
                    if variant_key in seen_variants:
                        continue
                    seen_variants.add(variant_key)
                    retries_done += 1
                    attempt_name = (
                        f"syqon_refine_{retries_done}_tile{tile_size}_"
                        f"overlap{overlap}{'_axiom' if axiom else ''}"
                    )
                    used = pipeline._stage7_try_syqon_variant(
                        syqon_script,
                        attempt_name=attempt_name,
                        tile_size=tile_size,
                        overlap=overlap,
                        axiom=axiom,
                    )
                    if not used:
                        quality_records.append(
                            {
                                "attempt": attempt_name,
                                "status": "failed",
                                "issues": [
                                    getattr(
                                        pipeline,
                                        "_last_plugin_script_error",
                                        "SyQon refinement failed",
                                    )
                                ],
                            }
                        )
                        continue
                    retry_cleanup = pipeline._stage7_clean_starmask(label=attempt_name)
                    starmask_cleanup_records.append(retry_cleanup)
                    retry_quality = pipeline._stage7_quality_assessment(
                        attempt_name,
                        tool_label=used,
                        use_ai=True,
                        source_stem=pipeline.stretched_name or "stage6_stretched",
                    )
                    quality_records.append(retry_quality)
                    retry_snapshot = pipeline._stage7_snapshot_current_outputs(
                        f"ai_best_{attempt_name}"
                    )
                    if pipeline._stage7_quality_score(retry_quality) < pipeline._stage7_quality_score(best_quality):
                        best_quality = retry_quality
                        best_snapshot = retry_snapshot
                        best_label = attempt_name
                        best_source_stem = pipeline.stretched_name or "stage6_stretched"

                if best_snapshot is not None:
                    pipeline._stage7_restore_snapshot(best_snapshot)
                selected_quality = best_quality
                selected_source_stem = best_source_stem
                stage_messages.append(
                    "stage7_quality selected optimized starless "
                    f"({best_label}, score={pipeline._stage7_quality_score(best_quality):.3f})"
                )
            else:
                stage_messages.append("stage7_quality refinement skipped: SyQon script unavailable")

        if quality_enabled:
            suppression_strength = pipeline._stage7_residual_suppression_strength(
                selected_quality
            )
            suppression_note = pipeline._apply_stage7_residual_suppression(
                suppression_strength,
                source_stem=selected_source_stem,
            )
            if suppression_note:
                stage_messages.append(suppression_note)
                selected_quality = pipeline._stage7_quality_assessment(
                    "selected_after_residual_suppression",
                    tool_label="stage7 residual suppression",
                    use_ai=False,
                    source_stem=selected_source_stem,
                )
                quality_records.append(selected_quality)
            quality_mode = "parameter_optimization"
        else:
            quality_mode = "local_quality"

        if (
            selected_quality
            and selected_quality.get("status") != "ok"
            and bool(getattr(pipeline.cfg, "stage7_starless_pixel_repair_enabled", True))
        ):
            before_repair_quality = selected_quality
            before_repair_score = pipeline._stage7_quality_score(before_repair_quality)
            repair_snapshot = pipeline._stage7_snapshot_current_outputs("before_pixel_repair")
            pixel_repair = pipeline._apply_stage7_starless_pixel_repair(
                source_stem=selected_source_stem,
                label="selected_starless_pixel_repair",
            )
            starless_pixel_repair_records.append(pixel_repair)
            if pixel_repair.get("status") == "applied":
                repaired_quality = pipeline._stage7_quality_assessment(
                    "selected_after_starless_pixel_repair",
                    tool_label="stage7 starless pixel repair",
                    use_ai=False,
                    source_stem=selected_source_stem,
                )
                quality_records.append(repaired_quality)
                repaired_score = pipeline._stage7_quality_score(repaired_quality)
                max_growth = float(pipeline.cfg.stage7_starless_repair_max_score_growth)
                before_derived = before_repair_quality.get("derived", {})
                repaired_derived = repaired_quality.get("derived", {})
                before_residual = float(before_derived.get("residual_star_score", 0.0) or 0.0)
                after_residual = float(repaired_derived.get("residual_star_score", 0.0) or 0.0)
                before_halo = float(before_derived.get("halo_residue_score", 0.0) or 0.0)
                after_halo = float(repaired_derived.get("halo_residue_score", 0.0) or 0.0)
                residual_improved = after_residual < before_residual - 0.005
                halo_improved = after_halo < before_halo - 0.005
                residual_not_worse = after_residual <= before_residual + 0.002
                halo_not_worse = after_halo <= before_halo + 0.002
                accepted = (
                    repaired_score <= before_repair_score + max_growth
                    and residual_not_worse
                    and halo_not_worse
                    and (residual_improved or halo_improved)
                )
                pixel_repair.update(
                    {
                        "accepted": accepted,
                        "score_before": before_repair_score,
                        "score_after": repaired_score,
                        "residual_before": before_residual,
                        "residual_after": after_residual,
                        "halo_before": before_halo,
                        "halo_after": after_halo,
                        "quality_after": repaired_quality,
                    }
                )
                if accepted:
                    selected_quality = repaired_quality
                    quality_mode = f"{quality_mode}+starless_pixel_repair"
                    pipeline._stage8_conservative_mode = bool(
                        pipeline.cfg.stage8_force_conservative_after_stage7_repair
                    )
                    stage_messages.append(
                        "stage7 starless pixel repair accepted "
                        f"(score {before_repair_score:.3f}->{repaired_score:.3f})"
                    )
                else:
                    pipeline._stage7_restore_snapshot(repair_snapshot)
                    selected_quality = before_repair_quality
                    stage_messages.append(
                        "stage7 starless pixel repair rolled back "
                        f"(score {before_repair_score:.3f}->{repaired_score:.3f})"
                    )
            elif pixel_repair.get("reason"):
                stage_messages.append(
                    "stage7 starless pixel repair skipped: "
                    f"{pixel_repair.get('reason')}"
                )

        if selected_quality and selected_quality.get("status") != "ok":
            pipeline._stage8_conservative_mode = bool(
                pipeline.cfg.stage8_force_conservative_after_stage7_repair
            )
        elif selected_quality:
            derived = selected_quality.get("derived") if isinstance(selected_quality, dict) else {}
            halo_for_base_guard = (
                float(derived.get("halo_residue_score", 0.0) or 0.0)
                if isinstance(derived, dict)
                else 0.0
            )
            if (
                pipeline._active_target_type() == "bright_emission_reflection_nebula"
                and halo_for_base_guard > float(pipeline.cfg.stage7_halo_residue_score_max)
            ):
                pipeline._stage8_conservative_mode = bool(
                    pipeline.cfg.stage8_force_conservative_after_stage7_repair
                )
                stage_messages.append(
                    "stage8 conservative mode requested for bright-nebula halo advisory "
                    f"({halo_for_base_guard:.3f}>{pipeline.cfg.stage7_halo_residue_score_max:.3f})"
                )
        stage9_remix_quality = selected_quality

        pipeline._write_stage_json(
            "stage7_quality.json",
            {
                "attempts": quality_records,
                "selected": selected_quality,
                "mode": quality_mode,
                "selected_source_stem": selected_source_stem,
                "preflight": stage7_preflight,
                "starmask_cleanup": starmask_cleanup_records,
                "repairs": repair_records,
                "starless_pixel_repairs": starless_pixel_repair_records,
                "conservative_inputs": conservative_input_records,
                "processing_plan": stage7_plan,
                "stage8_conservative_mode": pipeline._stage8_conservative_mode,
                "stage9_star_remix": pipeline._stage7_update_star_remix_from_quality(
                    stage9_remix_quality
                ),
                "retry_max": pipeline.cfg.stage7_quality_retry_max,
            },
        )
        if pipeline._stage9_star_intensity_scale < 0.999:
            stage_messages.append(
                "stage9 star remix intensity linked to stage7 residuals "
                f"(scale={pipeline._stage9_star_intensity_scale:.3f}, "
                f"reason={pipeline._stage9_star_intensity_reason})"
            )

        pipeline._export_sasp_exchange_files()
        pipeline.cmd_with_check("load", pipeline.starless_file.stem)
        stage_saved = pipeline._save_stage_output("stage7_starless")
        stage_message_text = "；".join(stage_messages)

        elapsed = pipeline.log.stage_end("阶段 7: 去星与星点层准备")
        if stage_saved:
            selected_status = str((selected_quality or {}).get("status", "ok")).lower()
            stage_status = "ok" if selected_status == "ok" else "degraded"
            pipeline._record_stage("阶段 7: 去星与星点层准备", stage_status, elapsed, stage_message_text)
        else:
            if stage_message_text:
                stage_message_text = f"{stage_message_text}；stage7 输出保存失败"
            else:
                stage_message_text = "stage7 输出保存失败"
            pipeline._record_stage(
                "阶段 7: 去星与星点层准备", 'degraded', elapsed, stage_message_text
            )

    except (CommandError, SirilError) as e:
        pipeline.log.error(f"去星流程失败: {e}")
        pipeline.log.error("请检查 SyQon-Starless.py / SASP Dark Star 环境与模型配置")
        pipeline.starless_file = None
        pipeline.starmask_file = None
        pipeline._stage7_update_star_remix_from_quality(None)
        # 使用拉伸后图像作为 starless 继续
        pipeline.cmd_with_check("load", pipeline.stretched_name)
        pipeline.cmd_with_check("save", "starless")
        pipeline.starless_file = pipeline.process_dir / "starless.fit"
        if pipeline._ai_stage_advisory_enabled("ai_stage7_enabled"):
            pipeline._write_stage_json(
                "stage7_quality.json",
                {
                    "attempts": [
                        {
                            "attempt": "degraded_stage6_stretched",
                            "tool_label": "none",
                            "status": "degraded",
                            "issues": [pipeline._short_text(e, 180)],
                        }
                    ],
                    "selected": {
                        "attempt": "degraded_stage6_stretched",
                        "status": "degraded",
                    },
                    "mode": "degraded_fallback",
                    "preflight": locals().get("stage7_preflight"),
                    "starmask_cleanup": locals().get("starmask_cleanup_records", []),
                    "repairs": locals().get("repair_records", []),
                    "starless_pixel_repairs": locals().get("starless_pixel_repair_records", []),
                    "conservative_inputs": locals().get("conservative_input_records", []),
                    "retry_max": pipeline.cfg.stage7_quality_retry_max,
                },
            )
        pipeline._export_sasp_exchange_files()
        pipeline.cmd_with_check("load", "starless")
        stage_saved = pipeline._save_stage_output("stage7_starless")

        elapsed = pipeline.log.stage_end("阶段 7: 去星与星点层准备")
        message = "无可用去星工具，已退化为直接使用拉伸图继续"
        if not stage_saved:
            message += "；stage7 输出保存失败"
        pipeline._record_stage(
            "阶段 7: 去星与星点层准备", 'degraded', elapsed, message)

