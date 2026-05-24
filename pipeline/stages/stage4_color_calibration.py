"""Stage 4 plate solving and color calibration."""
import os
from typing import List

from sirilpy.exceptions import CommandError, SirilError


ENV_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def run_stage4_color_calibration(pipeline) -> None:
    """
    阶段 4: 图像解析 + 色彩校准
    - 先做图像解析（天体测量）
    - 默认优先 SPCC，再回退 PCC -> CCM
    - 可通过 SEESTAR_SPCC_ENABLE=0 禁用 SPCC
    """
    pipeline.log.stage_start("阶段 4: 图像解析 + 色彩校准")
    status = 'ok'
    color_ok = False
    ccm_fallback_used = False
    color_method = "none"
    color_warning = ""
    color_confidence = 0.0
    messages: List[str] = []
    policy = getattr(pipeline, "pipeline_policy", {}) or {}
    stage4_policy = policy.get("stage4_color", {}) if isinstance(policy, dict) else {}

    allow_light_spcc = (
        os.getenv("SEESTAR_SPCC_ALLOW_LIGHT_PREPROCESS", "0")
        .strip()
        .lower()
        in ENV_TRUE_VALUES
    )
    spcc_runtime_allowed = True
    if pipeline.cfg.spcc_enabled and pipeline._stage1_input_mode == "light_preprocess":
        if allow_light_spcc:
            pipeline.log.warn(
                "检测到 Light_ 预处理输入，按 SEESTAR_SPCC_ALLOW_LIGHT_PREPROCESS=1 继续尝试 SPCC"
            )
        else:
            spcc_runtime_allowed = False
            pipeline.log.warn(
                "检测到 Light_ 预处理输入，跳过 SPCC 以规避 siril-cli 在该路径下的已知崩溃风险；"
                "如需强制尝试可设置 SEESTAR_SPCC_ALLOW_LIGHT_PREPROCESS=1"
            )
            messages.append(
                "SPCC skipped on Light_ preprocess mode to avoid siril-cli crash risk"
            )

    pipeline.platesolve_ok = False
    try:
        pipeline.log.info("执行图像解析 (天体测量)...")
        pipeline.cmd_with_check("platesolve")
        pipeline.platesolve_ok = True
        pipeline.log.info("图像解析成功")
    except (CommandError, SirilError) as e:
        pipeline.log.warn(f"图像解析失败: {e}")
        messages.append(f"platesolve 失败: {e}")

    if not pipeline.platesolve_ok:
        pipeline.log.warn("图像解析失败，仍尝试 SPCC/PCC 以最大化色彩校准成功率")
        messages.append("platesolve 不可用，继续尝试 SPCC/PCC")

    if pipeline.cfg.spcc_enabled and spcc_runtime_allowed:
        try:
            pipeline.cmd_with_check("spcc")
            pipeline.log.info("分光光度色彩校准成功 (SPCC)")
            color_ok = True
            color_method = "SPCC"
            color_confidence = 0.90
        except (CommandError, SirilError) as e:
            pipeline.log.warn(f"SPCC 失败: {e}")
            messages.append(f"SPCC 失败: {e}")
            pipeline.log.info("SPCC 不可用或失败，尝试 PCC...")
    else:
        if not pipeline.cfg.spcc_enabled:
            pipeline.log.warn("SPCC 已被显式禁用（SEESTAR_SPCC_ENABLE=0），直接尝试 PCC")
            messages.append("SPCC disabled by SEESTAR_SPCC_ENABLE=0")
        else:
            pipeline.log.info("SPCC 运行时保护已启用，直接尝试 PCC")

    if not color_ok:
        try:
            pipeline.cmd_with_check("pcc")
            pipeline.log.info("PCC 色彩校准成功")
            color_ok = True
            color_method = "PCC"
            color_confidence = 0.72 if pipeline.platesolve_ok else 0.58
            if not pipeline.platesolve_ok:
                color_warning = "imprecise_solution"
        except (CommandError, SirilError) as e:
            pipeline.log.warn(f"PCC 失败: {e}")
            messages.append(f"PCC 失败: {e}")

    if not color_ok:
        ccm_fallback_used = True
        pipeline.log.info("执行 CCM 灰世界色彩校准回退...")
        ccm_ok, ccm_message = pipeline._run_ccm_color_fallback()
        if ccm_ok:
            color_ok = True
            color_method = "CCM"
            color_warning = "non_photometric_fallback"
            color_confidence = 0.42
            messages.append(ccm_message)
        else:
            pipeline.log.warn(f"CCM 回退失败: {ccm_message}")
            status = 'degraded'
            messages.append(f"CCM 回退失败: {ccm_message}")

    if ccm_fallback_used and not pipeline.platesolve_ok and status == 'ok':
        status = 'degraded'
        pipeline.log.warn("platesolve 失败且阶段4使用 CCM 非光度回退，标记为 degraded")
        messages.append("platesolve 失败，已使用非光度 CCM 回退")
    if color_warning and stage4_policy.get("reduce_saturation_if_solution_imprecise", False):
        messages.append("color policy limits later saturation/color gains due to imprecise solution")

    stage_saved = pipeline._save_stage_output("stage4_colorbalanced")
    if not stage_saved and status == 'ok':
        status = 'degraded'
        messages.append("stage4 输出保存失败")
    if stage_saved and pipeline.platesolve_ok:
        metadata = pipeline._read_fits_header_metadata("stage4_colorbalanced")
        auto_hint = pipeline._auto_target_hint()
        if auto_hint:
            metadata["AUTO_TARGET_TYPE"] = auto_hint
        refresh_note = pipeline._refresh_target_profile_from_metadata(
            metadata,
            stage_label="Stage4",
        )
        if refresh_note:
            messages.append(refresh_note)
            policy = getattr(pipeline, "pipeline_policy", {}) or {}
            stage4_policy = policy.get("stage4_color", {}) if isinstance(policy, dict) else {}
    pipeline.color_calibration_report = {
        "stage": "stage4_color",
        "method": color_method,
        "status": "success_with_warning" if color_ok and color_warning else ("success" if color_ok else "degraded"),
        "warning": color_warning or None,
        "color_confidence": color_confidence,
        "policy": (
            pipeline._active_policy_name()
            if hasattr(pipeline, "_active_policy_name")
            else str(policy.get("policy_name", "generic_low_snr_safe"))
        ),
        "policy_adjustments": {
            "reduce_saturation_boost": bool(stage4_policy.get("reduce_saturation_if_solution_imprecise", False) and color_warning),
            "blue_gain_limit": stage4_policy.get("blue_gain_limit"),
            "red_gain_limit": stage4_policy.get("red_gain_limit"),
            "max_allowed_saturation_boost": stage4_policy.get("max_allowed_saturation_boost"),
        },
        "messages": messages,
    }
    pipeline._write_stage_json("color_calibration_report.json", pipeline.color_calibration_report)

    elapsed = pipeline.log.stage_end("阶段 4: 图像解析 + 色彩校准")
    pipeline._record_stage(
        "阶段 4: 图像解析 + 色彩校准",
        status,
        elapsed,
        "；".join(messages),
    )

