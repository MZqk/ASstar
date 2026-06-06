"""Stage 9 star processing and remixing."""
from typing import List

from models import PipelineStage
from sirilpy.exceptions import CommandError, SirilError


def _clamp_float(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, float(value))))


def _prepare_stage9_starmask_for_pixel_remix(
    pipeline,
    starmask_name: str,
    *,
    star_stretch_used: bool,
    messages: List[str],
) -> str:
    if not bool(getattr(pipeline.cfg, "stage9_starmask_stretch_enabled", True)):
        messages.append("Stage9 starmask stretch disabled; using original starmask")
        return starmask_name

    stretched_name = "starmask_stretched"
    try:
        pipeline.cmd_with_check("load", starmask_name)
        if not star_stretch_used:
            stretch = _clamp_float(
                getattr(pipeline.cfg, "stage9_starmask_asinh_stretch", 2.0),
                1.10,
                3.00,
            )
            offset = _clamp_float(
                getattr(pipeline.cfg, "stage9_starmask_asinh_offset", 0.001),
                0.0005,
                0.0060,
            )
            pipeline.cmd_with_check("asinh", f"{stretch:.3f}", f"{offset:.5f}")
            messages.append(
                "Stage9 starmask stretched before pixel remix "
                f"(method=asinh, stretch={stretch:.3f}, offset={offset:.5f})"
            )
        else:
            messages.append("Stage9 starmask uses plugin-stretched star layer for pixel remix")
        pipeline.cmd_with_check("save", stretched_name)
        if pipeline.process_dir:
            pipeline._stage9_stretched_starmask_file = (
                pipeline.process_dir / f"{stretched_name}.fit"
            )
        return stretched_name
    except (CommandError, SirilError) as e:
        pipeline.log.warn(f"Stage9 starmask stretch failed, using original starmask: {e}")
        messages.append(f"Stage9 starmask stretch failed; using original starmask: {e}")
        return starmask_name


def run_stage9_star_remixing(pipeline) -> None:
    """
    阶段 9: 星点处理与合成
    - 对齐工作流中的 Star Stretch / SCNR / Curves / StarComposer
    - 插件不可用时使用阶段 8 的 starless_enhanced 作为主图，再回混非线性 starmask
    """
    stage_label = PipelineStage.STAR_REMIXING.label
    pipeline.log.stage_start(stage_label)
    messages: List[str] = []
    source_stem = getattr(pipeline, "_stage8_final_source", "starless_enhanced") or "starless_enhanced"
    fallback_used = bool(getattr(pipeline, "_stage8_fallback_used", False))
    messages.append(
        "stage9_starless_source="
        f"{source_stem}; stage8_fallback_used={str(fallback_used).lower()}"
    )
    bad_starless_reason = pipeline._stage9_bad_starless_reason()
    if bad_starless_reason:
        safe_source = pipeline._stage9_review_safe_source()
        pipeline.log.warn(
            "Stage9 bypasses starless remix because selected starless is unsafe: "
            f"{bad_starless_reason}"
        )
        messages.append(
            "stage9_bad_starless_bypass fallback_used=true "
            f"source={safe_source}; reason={bad_starless_reason}"
        )
        try:
            pipeline.cmd_with_check("load", safe_source)
            stage_saved = pipeline._save_stage_output("stage9_remixed")
            pipeline._stage9_bypassed_bad_starless = True
            pipeline._stage9_final_source = safe_source
            diff_note = pipeline._stage_diff_note("stage9_remixed", safe_source)
            if diff_note:
                messages.append(diff_note)
            elapsed = pipeline.log.stage_end(stage_label)
            pipeline._record_stage(
                stage_label,
                "ok" if stage_saved else "degraded",
                elapsed,
                "；".join(messages),
            )
            return
        except (CommandError, SirilError) as e:
            messages.append(f"stage9 safe-source fallback failed: {e}")
            pipeline.log.warn(f"Stage9 safe-source fallback failed: {e}")

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
                pipeline.cmd_with_check("save", "starmask")
                pipeline.starmask_file = pipeline.process_dir / "starmask.fit"
                pipeline.log.info(f"已导入外部 Starmask: {external_starmask.name}")
        except (OSError, CommandError, SirilError) as e:
            pipeline.log.warn(f"导入外部 Starmask 失败，继续使用本地 starmask: {e}")

    star_stretch_used = False
    if pipeline.starmask_file and pipeline.starmask_file.exists():
        try:
            pipeline.cmd_with_check("load", pipeline.starmask_file.stem)
            star_stretch_label = pipeline._run_first_available_command(
                "星点拉伸",
                [
                    ("SASP Star Stretch", ("sasp_star_stretch",)),
                    ("NB to RGB Stars", ("nb_to_rgb_stars",)),
                ],
            )
            pipeline._run_first_available_command(
                "星点去紫",
                [
                    ("SASP Invert/SCNR", ("sasp_invert_scnr",)),
                    ("SCNR", ("scnr",)),
                ],
            )
            pipeline._run_first_available_command(
                "星点微调",
                [
                    ("SASP Curves Editor", ("sasp_curves_editor",)),
                    ("Curves", ("curves",)),
                ],
            )
            pipeline.cmd_with_check("save", "starmask")
            star_stretch_used = bool(star_stretch_label)
            pipeline.starmask_file = pipeline.process_dir / "starmask.fit"
            pipeline._export_sasp_exchange_files()
        except (CommandError, SirilError) as e:
            star_stretch_used = False
            pipeline.log.warn(f"星点处理插件链失败，使用原始 starmask: {e}")

    # 按工作流先在 Siril 侧做 Starless 二次细化，再进行星点合成
    if fallback_used:
        messages.append("Stage9 skipped starless secondary enhancement because Stage8 used fallback")
    else:
        try:
            pipeline.cmd_with_check("load", source_stem)
            pipeline._run_first_available_command(
                "细节/结构增强2",
                [
                    ("VeraLux Revela", ("veralux_revela",)),
                    ("Revela", ("revela",)),
                ],
            )
            if pipeline.cfg.optional_color_transform_enabled:
                pipeline._run_first_available_command(
                    "调色2（可选）",
                    [
                        ("VeraLux Vectra", ("veralux_vectra",)),
                        ("Vectra", ("vectra",)),
                    ],
                )
            pipeline._run_first_available_command(
                "最终微调颜色",
                [
                    ("VeraLux Curves", ("veralux_curves",)),
                    ("Curves", ("curves",)),
                ],
            )
            pipeline.cmd_with_check("save", source_stem)
        except (CommandError, SirilError) as e:
            pipeline.log.warn(f"Starless 二次细化失败，沿用当前 {source_stem}: {e}")

    remix_scale = _clamp_float(
        getattr(pipeline, "_stage9_star_intensity_scale", 1.0),
        0.45,
        1.0,
    )
    if fallback_used:
        messages.append("Stage8 fallback source active; bypass StarComposer for controlled remix")
        remix_scale = min(remix_scale, 0.95 / max(float(pipeline.cfg.star_intensity), 1e-6))
        messages.append("Stage8 fallback star remix intensity capped at 0.950")
        composer_used = None
    elif remix_scale < 0.999:
        messages.append(
            "Stage7 residual stars detected; bypass StarComposer for controlled star remix "
            f"(scale={remix_scale:.3f})"
        )
        composer_used = None
    else:
        composer_used = pipeline._run_first_available_command(
            "星点合成",
            [
                ("VeraLux StarComposer", ("veralux_starcomposer",)),
                ("StarComposer", ("starcomposer",)),
            ],
        )
    if composer_used:
        stage_saved = pipeline._save_stage_output("stage9_remixed")
        diff_note = pipeline._stage_diff_note("stage9_remixed", "stage8_enhanced")
        if diff_note:
            messages.append(diff_note)
        stage7_diff_note = pipeline._stage_diff_note("stage9_remixed", "stage7_stretched")
        if stage7_diff_note:
            messages.append(stage7_diff_note)
        elapsed = pipeline.log.stage_end(stage_label)
        if stage_saved:
            pipeline._record_stage(
                stage_label,
                'ok',
                elapsed,
                "；".join(messages),
            )
        else:
            messages.append("stage9 输出保存失败")
            pipeline._record_stage(
                stage_label,
                'degraded',
                elapsed,
                "；".join(messages),
            )
        return

    pipeline.log.info("执行基于上一阶段的星点合成...")
    if not pipeline.starmask_file or not pipeline.starmask_file.exists():
        pipeline.log.warn("无星点蒙版，跳过混合阶段")
        elapsed = pipeline.log.stage_end(stage_label)
        pipeline._record_stage(
            stage_label, 'skipped', elapsed, "无星点蒙版")
        return

    intensity = _clamp_float(pipeline.cfg.star_intensity * remix_scale, 0.10, 1.05)
    fallback_intensity = _clamp_float(
        pipeline.cfg.star_fallback_intensity * remix_scale,
        0.10,
        1.05,
    )
    if remix_scale < 0.999:
        reason = getattr(pipeline, "_stage9_star_intensity_reason", "")
        if not reason:
            reason = "stage8 fallback star intensity cap" if fallback_used else "stage7 residual stars"
        messages.append(
            "Stage9 star remix intensity reduced from safety diagnostics "
            f"(base={pipeline.cfg.star_intensity:.3f}, effective={intensity:.3f}, "
            f"fallback={fallback_intensity:.3f}, reason={reason})"
        )
    starmask_name = pipeline.starmask_file.stem
    remix_starmask_name = _prepare_stage9_starmask_for_pixel_remix(
        pipeline,
        starmask_name,
        star_stretch_used=star_stretch_used,
        messages=messages,
    )
    if pipeline._apply_previous_stage_star_remix(source_stem, remix_starmask_name, intensity):
        messages.append(
            "previous_stage_star_remix "
            f"source={source_stem}, starmask={remix_starmask_name}, "
            f"intensity={intensity:.3f}"
        )
    else:
        pipeline.log.warn("主混合失败，尝试替代强度...")
        if pipeline._apply_previous_stage_star_remix(
            source_stem, remix_starmask_name, fallback_intensity
        ):
            messages.append(
                "previous_stage_star_remix fallback "
                f"source={source_stem}, starmask={remix_starmask_name}, "
                f"intensity={fallback_intensity:.3f}"
            )
            pipeline.log.info(
                "使用替代强度完成混合 "
                f"({fallback_intensity})")
        else:
            elapsed = pipeline.log.stage_end(stage_label)
            pipeline._record_stage(
                stage_label, 'degraded', elapsed,
                "星点混合失败，使用上一阶段无星结果")
            return

    stage_saved = pipeline._save_stage_output("stage9_remixed")
    diff_note = pipeline._stage_diff_note("stage9_remixed", "stage8_enhanced")
    if diff_note:
        messages.append(diff_note)
    stage7_diff_note = pipeline._stage_diff_note("stage9_remixed", "stage7_stretched")
    if stage7_diff_note:
        messages.append(stage7_diff_note)

    elapsed = pipeline.log.stage_end(stage_label)
    if stage_saved:
        pipeline._record_stage(
            stage_label,
            'ok',
            elapsed,
            "；".join(messages),
        )
    else:
        messages.append("stage9 输出保存失败")
        pipeline._record_stage(
            stage_label,
            'degraded',
            elapsed,
            "；".join(messages),
        )
