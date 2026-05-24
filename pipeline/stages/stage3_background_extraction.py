"""Stage 3 background extraction."""
from typing import List, Tuple

from sirilpy.exceptions import CommandError, SirilError


def run_stage3_background_extraction(pipeline) -> None:
    """
    阶段 3: 背景提取
    - 优先尝试插件链：DBE/ADBE/GXP/NOX（含常见别名）
    - 插件或回退命令成功后执行质量门控，避免过度扣背景
    - 插件链不可用时回退内置 subsky RBF，最后回退线性多项式
    """
    pipeline.log.stage_start("阶段 3: 背景提取")
    bg_ok = False
    selected_source = ""
    stage_message = ""
    policy = getattr(pipeline, "pipeline_policy", {}) or {}
    policy_name = policy.get("policy_name", "generic_low_snr_safe") if isinstance(policy, dict) else "generic_low_snr_safe"
    stage3_policy = policy.get("stage3_background", {}) if isinstance(policy, dict) else {}
    pipeline.log.info(
        "[Stage3] Background policy: "
        f"policy={policy_name} protect_nebulosity={bool(stage3_policy.get('protect_nebulosity', False))} "
        f"model={','.join(stage3_policy.get('model_priority', []) or [])}"
    )

    baseline_stem = "stage3_bg_input"
    baseline_saved = False
    try:
        pipeline.cmd_with_check("save", baseline_stem)
        baseline_saved = True
    except (CommandError, SirilError) as e:
        pipeline.log.warn(f"stage3 baseline save failed, fallback without rollback: {e}")

    before_feat = pipeline._stage3_measure_features("before")
    before_adaptive = (
        pipeline._adaptive_features_current()
        if hasattr(pipeline, "_adaptive_features_current")
        else {}
    )

    attempts: List[Tuple[str, Tuple[str, ...], str]] = []
    if pipeline.cfg.workflow_plugin_probe_enabled:
        attempts.extend([
            ("DBE", ("dbe",), "plugin"),
            ("ADBE", ("adbe",), "plugin"),
            ("GXP", ("gxp",), "plugin"),
            ("NOX", ("nox",), "plugin"),
            ("AutoDBE", ("autodbe",), "plugin"),
            ("AutoBGE", ("autobge",), "plugin"),
            ("VeraLux NOX", ("veralux_nox",), "plugin"),
        ])

    for idx, cmd in enumerate(pipeline._stage3_subsky_rbf_candidates(), start=1):
        attempts.append((f"subsky-rbf-{idx}", cmd, "builtin"))
    attempts.append(("subsky-poly", ("subsky", "1"), "builtin"))

    for label, command, source in attempts:
        if baseline_saved:
            try:
                pipeline.cmd_with_check("load", baseline_stem, quiet=True)
            except (CommandError, SirilError) as e:
                pipeline.log.warn(f"failed to restore stage3 baseline: {e}")
                baseline_saved = False

        pipeline.log.info(f"尝试背景提取: {label}")
        if not pipeline._try_cmd(*command):
            continue

        after_feat = pipeline._stage3_measure_features(label)
        gate_ok, gate_msg = pipeline._stage3_quality_gate(before_feat, after_feat)
        if not gate_ok:
            pipeline.log.warn(
                f"{label} rejected by quality gate, try next candidate: {gate_msg}"
            )
            continue

        bg_ok = True
        selected_source = source
        stage_message = f"method={label}; {gate_msg}"
        if source == "plugin":
            pipeline.workflow_command_used["背景提取插件链"] = label
        pipeline.log.info(f"背景提取命中: {label}")
        break

    if not bg_ok:
        pipeline.log.error("背景提取完全失败，图像可能有梯度残留")

    stage_saved = pipeline._save_stage_output("stage3_bgremoved")
    after_adaptive = (
        pipeline._adaptive_features_current()
        if hasattr(pipeline, "_adaptive_features_current")
        else {}
    )
    max_bg_std_growth = float(stage3_policy.get("max_bg_std_growth", 1.10) or 1.10)
    fallback_warning = False
    if before_adaptive and after_adaptive:
        before_std = max(float(before_adaptive.get("bg_std", 0.0) or 0.0), 1e-7)
        after_std = float(after_adaptive.get("bg_std", 0.0) or 0.0)
        dirty = float(after_adaptive.get("dirty_background_score", 0.0) or 0.0)
        gradient_before = float(before_adaptive.get("gradient_score", 0.0) or 0.0)
        gradient_after = float(after_adaptive.get("gradient_score", 0.0) or 0.0)
        if after_std / before_std > max_bg_std_growth or (
            dirty > 0.35 and gradient_after >= gradient_before * 0.92
        ):
            fallback_warning = True
            warning_msg = (
                "background improvement limited "
                f"(dirty={dirty:.3f}, std_growth={after_std / before_std:.3f})"
            )
            pipeline.log.warn(f"[Stage3] {warning_msg}")
            stage_message = f"{stage_message}; {warning_msg}" if stage_message else warning_msg
    if hasattr(pipeline, "_write_stage_json"):
        pipeline._write_stage_json(
            "background_quality_report.json",
            {
                "stage": "stage3_background",
                "policy": policy_name,
                "model_used": stage_message.split(";")[0].replace("method=", "") if stage_message else None,
                "protected_masks": [
                    name
                    for name, enabled in (
                        ("nebulosity_mask", stage3_policy.get("protect_nebulosity")),
                        ("bright_core_mask", stage3_policy.get("protect_bright_core")),
                        ("star_halo_mask", stage3_policy.get("protect_star_halo")),
                        ("outer_halo_mask", stage3_policy.get("protect_outer_halo")),
                        ("dark_structure_mask", stage3_policy.get("protect_dark_structure")),
                    )
                    if enabled
                ],
                "before": before_adaptive,
                "after": after_adaptive,
                "quality": "warning" if fallback_warning else ("ok" if bg_ok else "degraded"),
                "fallback_used": not bg_ok or fallback_warning,
            },
        )
    if not stage_saved:
        stage_message = (
            f"{stage_message}; stage3 输出保存失败"
            if stage_message
            else "stage3 输出保存失败"
        )

    elapsed = pipeline.log.stage_end("阶段 3: 背景提取")
    if bg_ok:
        status = "ok" if stage_saved else "degraded"
        pipeline._record_stage("阶段 3: 背景提取", status, elapsed, stage_message)
        if selected_source == "builtin":
            if pipeline.cfg.workflow_plugin_probe_enabled:
                pipeline.log.warn(
                    "阶段3未命中 DBE/ADBE/GXP/NOX 插件，已使用内置背景提取回退"
                )
            else:
                pipeline.log.info("阶段3按策略使用内置背景提取（插件探测关闭）")
    else:
        degrade_message = "背景提取失败，图像可能有梯度残留"
        if not stage_saved:
            degrade_message += "；stage3 输出保存失败"
        pipeline._record_stage(
            "阶段 3: 背景提取",
            "degraded",
            elapsed,
            degrade_message,
        )
