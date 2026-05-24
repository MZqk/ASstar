"""Stage 5 linear correction, sharpen, and denoise."""
import os
from typing import List, Optional

from sirilpy.exceptions import CommandError, SirilError


ENV_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
ENV_COSMIC_CLASSIC_ENABLE_KEY = "SEESTAR_COSMIC_CLASSIC_ENABLE"


def run_stage5_linear_denoise(pipeline) -> None:
    """
    阶段 5: 星点矫正 / 锐化 / 初步降噪 / 可选转色
    - 对齐工作流中的 Aberrations Remover、CC Sharpen Both、GXP/Silentium
    - 锐化/降噪脚本失败时，优先回退 Siril-CC Sharpen Both 0.1/3/0.5（纯命令）
    - 最终仍可回退到内置 denoise
    """
    pipeline.log.stage_start("阶段 5: 星点矫正 / 锐化 / 初步降噪")
    status = 'ok'
    messages: List[str] = []
    policy = getattr(pipeline, "pipeline_policy", {}) or {}
    stage5_policy = policy.get("stage5_linear", {}) if isinstance(policy, dict) else {}
    if stage5_policy:
        pipeline.log.info(
            "[Stage5] Linear policy: "
            f"denoise={stage5_policy.get('denoise_mode', 'default')} "
            f"sharpen={stage5_policy.get('sharpen_mode', 'default')} "
            f"protect_background={bool(stage5_policy.get('protect_background', False))}"
        )
    cc_sharpen_fallback_used: Optional[str] = None
    linear_export_ok = pipeline._export_linear_intermediate()
    if not linear_export_ok:
        status = 'degraded'
        messages.append("导出 result_linear.fit 失败")

    aberration_script = pipeline._find_plugin_script(("processing/AberrationRemover.py",))
    if aberration_script is not None:
        pipeline.log.info(
            "检测到 AberrationRemover.py；按稳定策略优先使用内置 Aberration API 路径"
        )

    aberration_used = None
    local_model = pipeline._resolve_local_aberration_model()
    if local_model is not None:
        pipeline.log.info(f"矫正星点尝试本地 Aberration 模型: {local_model}")
        aberration_used = pipeline._run_aberration_api("矫正星点", model_path=local_model)
    elif getattr(pipeline.cfg, "aberration_api_enabled", False):
        aberration_used = pipeline._run_aberration_api("矫正星点")
    if not aberration_used and pipeline._last_aberration_api_error:
        messages.append(
            "Aberration API 不可用: "
            f"{pipeline._short_text(pipeline._last_aberration_api_error, 160)}"
        )
    if aberration_used:
        messages.append(f"星点矫正使用 {aberration_used}")
    if not aberration_used:
        messages.append("未执行星点矫正（Aberration API 不可用）")

    sharpen_used = None
    sharpen_failure_reason = ""
    sharpen_script = pipeline._find_plugin_script(("processing/CosmicClarity_Sharpen.py",))
    sharpen_executable_args = pipeline._classic_cosmic_clarity_args(
        "sirilcc_sharpen.conf",
        "CosmicClarity Sharpen",
    )
    sharpen_device_args, sharpen_device_note = pipeline._classic_cosmic_clarity_device_args()
    if sharpen_script is not None and sharpen_executable_args is not None:
        sharpen_used = pipeline._run_plugin_script_by_path(
            "锐化",
            "CosmicClarity Sharpen",
            sharpen_script,
            args=(
                "-sharpening_mode",
                "Both",
                "-stellar_amount",
                "0.35",
                "-non_stellar_strength",
                "3",
                "-non_stellar_amount",
                "0.35",
                *sharpen_device_args,
                *sharpen_executable_args,
            ),
        )
        if not sharpen_used:
            script_error = (
                getattr(pipeline, "_last_plugin_script_error", None)
                or sharpen_script.name
            )
            sharpen_failure_reason = script_error
    elif sharpen_script is not None:
        sharpen_failure_reason = (
            "CosmicClarity classic disabled; using Native path"
            if os.getenv(ENV_COSMIC_CLASSIC_ENABLE_KEY, "").strip().lower()
            not in ENV_TRUE_VALUES
            else "CosmicClarity executable not configured"
        )
    else:
        sharpen_failure_reason = "CosmicClarity_Sharpen.py script missing"

    if not sharpen_used:
        native_sharpen_used = pipeline._run_cosmic_clarity_native_sharpen_fallback("锐化")
        if native_sharpen_used:
            sharpen_used = native_sharpen_used
            if pipeline._is_classic_cc_not_configured(sharpen_failure_reason):
                messages.append(
                    "CosmicClarity Sharpen classic executable 未配置，"
                    f"已直接选择 {native_sharpen_used}"
                )
            else:
                messages.append(
                    pipeline._fallback_summary(
                        "CosmicClarity Sharpen",
                        sharpen_failure_reason or f"classic unavailable ({sharpen_device_note})",
                        native_sharpen_used,
                        True,
                    )
                )

    if not sharpen_used:
        cc_sharpen_fallback_used = pipeline._run_siril_cc_sharpen_fallback("锐化回退")
        if cc_sharpen_fallback_used:
            sharpen_used = cc_sharpen_fallback_used
            messages.append(
                pipeline._fallback_summary(
                    "CosmicClarity Sharpen",
                    sharpen_failure_reason or "script unavailable",
                    cc_sharpen_fallback_used,
                    True,
                )
            )
        else:
            messages.append(
                pipeline._fallback_summary(
                    "CosmicClarity Sharpen",
                    sharpen_failure_reason or "script unavailable",
                    "Siril-CC Sharpen Both",
                    False,
                )
            )

    if not sharpen_used:
        sharpen_used = pipeline._run_first_available_command(
            "锐化",
            [
                ("Unsharp fallback", ("unsharp", "1.0", "0.8")),
            ],
            allow_when_probe_disabled=True,
        )
        if sharpen_used:
            messages.append(
                pipeline._fallback_summary(
                    "Siril-CC Sharpen Both",
                    "fallback unavailable",
                    sharpen_used,
                    True,
                )
            )
    if not sharpen_used:
        messages.append("未检测到锐化插件，跳过锐化")
    else:
        messages.append(f"锐化使用 {sharpen_used}")

    plugin_denoise_used = None
    denoise_failure_reason = ""
    denoise_script = pipeline._find_plugin_script(("processing/CosmicClarity_Denoise.py",))
    denoise_executable_args = pipeline._classic_cosmic_clarity_args(
        "sirilcc_denoise.conf",
        "CosmicClarity Denoise",
    )
    denoise_device_args, denoise_device_note = pipeline._classic_cosmic_clarity_device_args()
    if denoise_script is not None and denoise_executable_args is not None:
        plugin_denoise_used = pipeline._run_plugin_script_by_path(
            "初步降噪",
            "CosmicClarity Denoise",
            denoise_script,
            args=(
                "-denoising_mode",
                "luminance",
                "-denoise_strength",
                "0.30",
                *denoise_device_args,
                *denoise_executable_args,
            ),
        )
        if not plugin_denoise_used:
            script_error = (
                getattr(pipeline, "_last_plugin_script_error", None)
                or denoise_script.name
            )
            denoise_failure_reason = script_error
            if cc_sharpen_fallback_used is None:
                cc_sharpen_fallback_used = pipeline._run_siril_cc_sharpen_fallback("初步降噪回退")
            if cc_sharpen_fallback_used:
                messages.append(
                    pipeline._fallback_summary(
                        "CosmicClarity Denoise",
                        denoise_failure_reason,
                        cc_sharpen_fallback_used,
                        True,
                    )
                )
            else:
                messages.append(
                    pipeline._fallback_summary(
                        "CosmicClarity Denoise",
                        denoise_failure_reason,
                        "Siril-CC Sharpen Both",
                        False,
                    )
                )
    elif denoise_script is not None:
        denoise_failure_reason = (
            "CosmicClarity classic disabled; using Native path"
            if os.getenv(ENV_COSMIC_CLASSIC_ENABLE_KEY, "").strip().lower()
            not in ENV_TRUE_VALUES
            else "CosmicClarity executable not configured"
        )
    else:
        denoise_failure_reason = "CosmicClarity_Denoise.py script missing"

    if not plugin_denoise_used:
        native_denoise_used = pipeline._run_cosmic_clarity_native_denoise_fallback("初步降噪")
        if native_denoise_used:
            plugin_denoise_used = native_denoise_used
            if pipeline._is_classic_cc_not_configured(denoise_failure_reason):
                messages.append(
                    "CosmicClarity Denoise classic executable 未配置，"
                    f"已直接选择 {native_denoise_used}"
                )
            else:
                messages.append(
                    pipeline._fallback_summary(
                        "CosmicClarity Denoise",
                        denoise_failure_reason or f"classic unavailable ({denoise_device_note})",
                        native_denoise_used,
                        True,
                    )
                )

    if plugin_denoise_used:
        pipeline.log.info("已完成插件初步降噪")
        messages.append(f"初步降噪使用 {plugin_denoise_used}")
    elif not pipeline.cfg.denoise_enabled:
        if aberration_used or sharpen_used:
            pipeline.log.info("内置线性降噪关闭，保留已完成的插件处理结果")
            messages.append("初步降噪未执行（denoise_enabled=False）")
        else:
            pipeline.log.info("内置线性降噪关闭，且未命中插件降噪，保持当前图像")
            status = 'skipped' if status == 'ok' else status
            messages.append("初步降噪未执行（插件不可用且 denoise_enabled=False）")
    else:
        pipeline.log.info("插件降噪不可用，回退内置线性降噪...")
        denoise_mod = min(max(pipeline.cfg.denoise_mod, 0.0), 1.0)
        denoise_safety_max = max(0.2, min(0.55, float(pipeline.cfg.denoise_safety_max)))
        if pipeline.auto_tune_result and pipeline.auto_tune_result.features:
            feat = pipeline.auto_tune_result.features
            if feat.star_density > 0.0040:
                denoise_safety_max = min(denoise_safety_max, 0.40)
                pipeline.log.info("高星密度场景，收紧降噪上限到 0.40")
            if feat.core_brightness_ratio > 0.08:
                denoise_safety_max = min(denoise_safety_max, 0.36)
                pipeline.log.info("亮核占比较高，收紧降噪上限到 0.36")
            if feat.object_area_ratio < 0.10 and feat.star_density > 0.0030:
                denoise_safety_max = min(denoise_safety_max, 0.34)
                pipeline.log.info("小目标高星场，收紧降噪上限到 0.34")

        if denoise_mod > denoise_safety_max:
            pipeline.log.warn(
                f"降噪参数过高({denoise_mod})，自动限制为 {denoise_safety_max}")
            denoise_mod = denoise_safety_max
        try:
            pipeline.cmd_with_check("denoise", f"-mod={denoise_mod}")
            pipeline.log.info(f"线性降噪完成 (mod={denoise_mod})")
            if not plugin_denoise_used:
                messages.append(
                    pipeline._fallback_summary(
                        "CosmicClarity Denoise",
                        denoise_failure_reason or "plugin unavailable",
                        f"built-in linear denoise (mod={denoise_mod})",
                        True,
                    )
                )
        except (CommandError, SirilError) as e:
            pipeline.log.warn(f"线性降噪跳过: {e}")
            status = 'degraded'
            messages.append(
                pipeline._fallback_summary(
                    "CosmicClarity Denoise",
                    denoise_failure_reason or "plugin unavailable",
                    "built-in linear denoise",
                    False,
                )
            )
            messages.append("线性降噪失败，沿用当前线性图像")

    if pipeline.cfg.optional_color_transform_enabled:
        transform_used = pipeline._run_first_available_command(
            "可选转色",
            [
                ("VeraLux Alchemy", ("veralux_alchemy",)),
                ("Hubble Palette From Dual Band", ("hubble_palette_from_dual_band",)),
                ("NB to RGB", ("nb_to_rgb",)),
            ],
        )
        if not transform_used:
            messages.append("可选转色已启用但插件不可用，保持原色彩")

    stage_saved = pipeline._save_stage_output("stage5_denoised")
    if not stage_saved and status == 'ok':
        status = 'degraded'
        messages.append("stage5 输出保存失败")
    elif stage_saved:
        diff_note = pipeline._stage_diff_note("stage5_denoised", "stage4_colorbalanced")
        if diff_note:
            messages.append(diff_note)
    pipeline._write_stage_json(
        "stage5_linear_report.json",
        {
            "stage": "stage5_linear",
            "policy": (
                pipeline._active_policy_name()
                if hasattr(pipeline, "_active_policy_name")
                else str(policy.get("policy_name", "generic_low_snr_safe"))
            ),
            "target_type": (
                pipeline._active_target_type()
                if hasattr(pipeline, "_active_target_type")
                else "generic_low_snr_safe"
            ),
            "denoise_mode": stage5_policy.get("denoise_mode", "legacy"),
            "sharpen_mode": stage5_policy.get("sharpen_mode", "legacy"),
            "protect_background": bool(stage5_policy.get("protect_background", False)),
            "protect_star_halo": bool(stage5_policy.get("protect_star_halo", False)),
            "avoid_global_sharpen": bool(stage5_policy.get("avoid_global_sharpen", False)),
            "status": status,
            "messages": messages,
        },
    )

    elapsed = pipeline.log.stage_end("阶段 5: 星点矫正 / 锐化 / 初步降噪")
    pipeline._record_stage(
        "阶段 5: 星点矫正 / 锐化 / 初步降噪",
        status,
        elapsed,
        "；".join(messages),
    )

