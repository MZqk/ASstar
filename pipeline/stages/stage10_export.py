"""Stage 10 final denoise and export."""
from typing import List

from sirilpy.exceptions import CommandError, SirilError

from save_utils import export_final_outputs


def run_stage10_export(pipeline) -> None:
    """
    阶段 10: 最终降噪与导出
    - 最终色彩微调
    - SCUNet 最终降噪（若可用）
    - 导出 TIFF/PNG/FITS
    """
    pipeline.log.stage_start("阶段 10: 最终降噪与导出")
    status = "ok"
    messages: List[str] = []

    # 按优先级加载最终图像
    final_file = "stage9_remixed"
    final_loaded = False
    for candidate in [
        final_file,
        "starless_enhanced",
        pipeline.stretched_name or "stage6_stretched",
    ]:
        candidate_path = pipeline.process_dir / f"{candidate}.fit"
        if not candidate_path.exists():
            messages.append(f"final_candidate_missing={candidate}")
            continue
        try:
            pipeline.cmd_with_check("load", candidate)
            final_file = candidate
            final_loaded = True
            break
        except (CommandError, SirilError):
            messages.append(f"final_candidate_load_failed={candidate}")
            continue
    if not final_loaded:
        status = "degraded"
        messages.append("最终候选图加载失败，沿用当前 Siril 图像")
    pipeline.log.info(f"使用最终图像: {final_file}")

    # 色彩微调
    pipeline.log.info("色彩最终优化...")
    try:
        pipeline.cmd_with_check(
            "satu",
            str(pipeline.cfg.final_saturation),
            str(pipeline.cfg.final_bg_factor),
        )
    except (CommandError, SirilError) as e:
        pipeline.log.warn(f"最终饱和度调整跳过: {e}")
        status = "degraded"
        messages.append(f"最终饱和度调整失败: {e}")

    final_denoise_used = None
    final_scunet_used = None
    denoise_primary = "CosmicClarity Denoise in-process script"
    denoise_primary_status = "skipped"
    denoise_effective = "none"
    denoise_effective_status = "skipped"
    final_denoise_script = pipeline._find_plugin_script(
        ("processing/CosmicClarity_Denoise.py",)
    )
    final_denoise_executable_args = pipeline._classic_cosmic_clarity_args(
        "sirilcc_denoise.conf",
        "CosmicClarity Denoise",
    )
    if final_denoise_script is not None and final_denoise_executable_args is not None:
        cli_args: List[str] = [
            "-denoising_mode",
            "luminance",
            "-denoise_strength",
            "0.5",
            "-use_gpu",
            *final_denoise_executable_args,
        ]

        final_denoise_used = pipeline._run_plugin_script_by_path(
            "最终降噪",
            "CosmicClarity Denoise",
            final_denoise_script,
            args=tuple(cli_args),
        )
        if final_denoise_used:
            denoise_primary_status = "success"
            denoise_effective = final_denoise_used
            denoise_effective_status = "success"
        else:
            denoise_primary_status = "failed"
            script_error = (
                getattr(pipeline, "_last_plugin_script_error", None)
                or final_denoise_script.name
            )
            cli_denoise_used = pipeline._run_plugin_script_cli_subprocess(
                "最终降噪",
                "CosmicClarity Denoise",
                final_denoise_script,
                args=tuple(cli_args),
                timeout_sec=pipeline._final_denoise_cli_timeout_sec(),
            )
            if cli_denoise_used:
                final_denoise_used = cli_denoise_used
                denoise_effective = cli_denoise_used
                denoise_effective_status = "success"
                messages.append(
                    pipeline._fallback_summary(
                        "CosmicClarity Denoise in-process",
                        script_error,
                        cli_denoise_used,
                        True,
                    )
                )
            else:
                cli_error = (
                    getattr(pipeline, "_last_plugin_script_error", None)
                    or final_denoise_script.name
                )
                native_denoise_used = pipeline._run_cosmic_clarity_native_denoise_fallback(
                    "最终降噪回退"
                )
                if native_denoise_used:
                    final_denoise_used = native_denoise_used
                    denoise_effective = native_denoise_used
                    denoise_effective_status = "success"
                    messages.append(
                        pipeline._fallback_summary(
                            "CosmicClarity Denoise CLI subprocess",
                            cli_error,
                            native_denoise_used,
                            True,
                        )
                    )
                else:
                    native_error = (
                        getattr(pipeline, "_last_plugin_script_error", None)
                        or "CosmicClarity_Native.py unavailable"
                    )
                    final_scunet_used = pipeline._run_siril_scunet_denoise_fallback(
                        "最终降噪回退",
                        0.28,
                    )
                    if final_scunet_used:
                        denoise_effective = final_scunet_used
                        denoise_effective_status = "success"
                        messages.append(
                            pipeline._fallback_summary(
                                "CosmicClarity Native Denoise",
                                native_error,
                                final_scunet_used,
                                True,
                            )
                        )
                    else:
                        scunet_reason = getattr(
                            pipeline,
                            "_last_scunet_fallback_error",
                            None,
                        )
                        messages.append(
                            pipeline._fallback_summary(
                                "CosmicClarity Native Denoise",
                                native_error,
                                "Siril-SCUNet Denoise",
                                False,
                            )
                        )
                        if scunet_reason:
                            messages.append(
                                f"Siril-SCUNet Denoise 回退不可用: {scunet_reason}"
                            )
                        else:
                            messages.append("Siril-SCUNet Denoise 回退不可用")
    elif final_denoise_script is not None:
        denoise_primary = "CosmicClarity Native Denoise cli-subprocess"
        pipeline.log.info(
            "CosmicClarity Denoise classic 路径未启用，使用 Native Denoise"
        )
        native_denoise_used = pipeline._run_cosmic_clarity_native_denoise_fallback(
            "最终降噪"
        )
        if native_denoise_used:
            final_denoise_used = native_denoise_used
            denoise_effective = native_denoise_used
            denoise_primary_status = "success"
            denoise_effective_status = "success"
            messages.append("CosmicClarity classic 路径未启用，已选择 Native Denoise")
        else:
            denoise_primary_status = "failed"
            native_error = (
                getattr(pipeline, "_last_plugin_script_error", None)
                or "CosmicClarity_Native.py unavailable"
            )
            final_scunet_used = pipeline._run_siril_scunet_denoise_fallback(
                "最终降噪回退",
                0.28,
            )
            if final_scunet_used:
                denoise_effective = final_scunet_used
                denoise_effective_status = "success"
                messages.append(
                    pipeline._fallback_summary(
                        "CosmicClarity Native Denoise",
                        native_error,
                        final_scunet_used,
                        True,
                    )
                )
            else:
                scunet_reason = getattr(pipeline, "_last_scunet_fallback_error", None)
                messages.append(
                    pipeline._fallback_summary(
                        "CosmicClarity Native Denoise",
                        native_error,
                        "Siril-SCUNet Denoise",
                        False,
                    )
                )
                if scunet_reason:
                    messages.append(f"Siril-SCUNet Denoise 回退不可用: {scunet_reason}")
    else:
        denoise_primary_status = "missing"
        native_denoise_used = pipeline._run_cosmic_clarity_native_denoise_fallback(
            "最终降噪回退"
        )
        if native_denoise_used:
            final_denoise_used = native_denoise_used
            denoise_effective = native_denoise_used
            denoise_effective_status = "success"
            messages.append(
                pipeline._fallback_summary(
                    "CosmicClarity_Denoise.py",
                    "script missing",
                    native_denoise_used,
                    True,
                )
            )
        else:
            native_error = (
                getattr(pipeline, "_last_plugin_script_error", None)
                or "CosmicClarity_Native.py unavailable"
            )
            final_scunet_used = pipeline._run_siril_scunet_denoise_fallback(
                "最终降噪回退",
                0.28,
            )
            if final_scunet_used:
                denoise_effective = final_scunet_used
                denoise_effective_status = "success"
                messages.append(
                    pipeline._fallback_summary(
                        "CosmicClarity Native Denoise",
                        native_error,
                        final_scunet_used,
                        True,
                    )
                )
            else:
                scunet_reason = getattr(pipeline, "_last_scunet_fallback_error", None)
                if scunet_reason:
                    messages.append(
                        pipeline._fallback_summary(
                            "CosmicClarity Native Denoise",
                            native_error,
                            "Siril-SCUNet Denoise",
                            False,
                        )
                    )
                    messages.append(f"Siril-SCUNet Denoise 回退不可用: {scunet_reason}")

    if final_denoise_used:
        pipeline.log.info("已执行最终降噪（作为最后一步处理）")
        messages.append(f"最终降噪使用 {final_denoise_used}")
    elif final_scunet_used:
        pipeline.log.info("已执行 Siril-SCUNet 最终降噪（代码回退）")
        messages.append(f"最终降噪使用 {final_scunet_used}")
    elif getattr(pipeline.cfg, "aberration_api_enabled", False):
        pipeline.log.warn("最终降噪脚本不可用，尝试 Aberration API 作为回退")
        local_model = pipeline._resolve_local_aberration_model()
        aberration_used = pipeline._run_aberration_api("最终降噪", model_path=local_model)
        if aberration_used:
            denoise_effective = aberration_used
            denoise_effective_status = "success"
            messages.append(
                pipeline._fallback_summary(
                    "CosmicClarity/SCUNet final denoise",
                    "previous denoise candidates unavailable",
                    aberration_used,
                    True,
                )
            )
        else:
            pipeline.log.warn("最终降噪脚本与 Aberration API 均不可用，跳过最终降噪")
            if pipeline._last_aberration_api_error:
                messages.append(
                    "Aberration API 不可用: "
                    f"{pipeline._short_text(pipeline._last_aberration_api_error, 160)}"
                )
            messages.append("最终降噪脚本与 Aberration API 均不可用")
            denoise_effective_status = "failed"
            status = "degraded" if status == "ok" else status
    else:
        pipeline.log.info("最终降噪脚本不可用，且 Aberration API 默认关闭，跳过最终降噪")
        messages.append("最终降噪未执行（script/scunet unavailable, Aberration API disabled）")
        status = "degraded" if status == "ok" else status

    messages.append(
        f"final_denoise_primary={denoise_primary}; "
        f"primary_status={denoise_primary_status}; "
        f"final_denoise_effective={denoise_effective}; "
        f"effective_status={denoise_effective_status}"
    )

    stage_saved = pipeline._save_stage_output("stage10_final")
    if not stage_saved and status == "ok":
        status = "degraded"
        messages.append("stage10 输出保存失败")
    elif stage_saved:
        diff_note = pipeline._stage_diff_note("stage10_final", final_file)
        if diff_note:
            messages.append(diff_note)
        feature_note = pipeline._feature_summary_note("最终导出前特征")
        if feature_note:
            messages.append(feature_note)
        if hasattr(pipeline, "_final_quality_report"):
            try:
                final_quality = pipeline._final_quality_report("stage10_final")
                pipeline._write_stage_json("final_quality_report.json", final_quality)
                pipeline.log.info(
                    "[Stage10] final_quality="
                    f"{final_quality.get('final_quality')} "
                    f"status={final_quality.get('status')} "
                    f"needs_conservative_rerun={str(bool(final_quality.get('needs_conservative_rerun'))).lower()}"
                )
                messages.append(
                    "final_quality="
                    f"{final_quality.get('final_quality')} "
                    f"status={final_quality.get('status')}"
                )
                if final_quality.get("final_quality") != "ok":
                    status = "degraded" if status == "ok" else status
                    issues = final_quality.get("issues", [])
                    if isinstance(issues, list) and issues:
                        issue_text = ", ".join(str(x) for x in issues[:2])
                        pipeline.log.warn(f"[Stage10] final_quality_issues={issue_text}")
                        messages.append("final_quality_issues=" + issue_text)
                    else:
                        messages.append("final_quality=poor")
            except Exception as e:
                pipeline.log.warn(f"final quality report failed: {e}")
                messages.append("final_quality_report 写入失败")
                status = "degraded" if status == "ok" else status

    # 切换回原工作目录导出
    pipeline.cmd_with_check("cd", f'"{pipeline.work_dir}"')

    base_filename = pipeline._result_output_basename()
    fallback_base = "result_processed"
    fallback_fit_base = "result_final"
    if pipeline._stage1_input_mode == "linear_resume":
        fallback_base = "result_processed_linear"
        fallback_fit_base = "result_final_linear"

    status, messages = export_final_outputs(
        pipeline.cmd_with_check,
        pipeline.log,
        base_filename=base_filename,
        fallback_base=fallback_base,
        fallback_fit_base=fallback_fit_base,
        status=status,
        messages=messages,
    )

    elapsed = pipeline.log.stage_end("阶段 10: 最终降噪与导出")
    pipeline._record_stage(
        "阶段 10: 最终降噪与导出",
        status,
        elapsed,
        "；".join(messages),
    )
