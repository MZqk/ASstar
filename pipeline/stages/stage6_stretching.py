"""Stretch selection and execution."""
from typing import List

from models import PipelineStage, StarSeparationState
from sirilpy.exceptions import CommandError, SirilError


def _run_with_stars_review_stretch(pipeline, separation_state: str) -> None:
    """Create a conservative review image without invoking starless-only logic."""
    stage_label = PipelineStage.STRETCHING.label
    messages: List[str] = [
        f"star_separation_state={separation_state}",
        "starless-only stretch candidates skipped",
    ]
    source_stem = str(
        getattr(pipeline, "_stage6_passthrough_source", None)
        or "stage6_passthrough"
    )
    pipeline._stage7_stretch_accepted = False
    pipeline._stage7_stretch_output = None
    pipeline._stage7_review_source = None
    saved = False
    try:
        pipeline.cmd_with_check("load", source_stem)
        try:
            pipeline.cmd_with_check("autostretch", "-linked")
            messages.append("linked autostretch applied for review preview only")
        except (CommandError, SirilError) as error:
            pipeline.cmd_with_check("load", source_stem)
            messages.append(
                "review autostretch failed; retained linear passthrough: "
                f"{pipeline._short_text(error, 160)}"
            )
        saved = pipeline._save_stage_output("stage7_review_with_stars")
        if saved:
            pipeline.stretched_name = "stage7_review_with_stars"
            pipeline._stage7_review_source = pipeline.stretched_name
    except (CommandError, SirilError) as error:
        messages.append(
            "with-stars review source unavailable: "
            f"{pipeline._short_text(error, 160)}"
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
        reason_code=f"star_separation_{separation_state}",
        details={
            "source_stem": source_stem,
            "review_output": "stage7_review_with_stars" if saved else None,
        },
    )


def run_stage7_stretching(pipeline) -> None:
    """
    阶段 7: 主体拉伸
    - 输入固定为 stage6_starless.fit
    - 固定生成 stage7_cand_a / stage7_cand_b 和一个 stage7_preview_ref
    - 亮核心星云 cand_a 使用扩张的星点/halo 掩膜保护并保留 1.50x 尺寸门
    - 完整变换后实测 P50；偏离时从线性源校准参数并重跑一次
    - 候选仅因背景色度门控失败时，可追加受控色度救援候选
    - 输出 stage7_stretched.fit
    """
    stage_label = PipelineStage.STRETCHING.label
    pipeline.log.stage_start(stage_label)
    pipeline._stage7_stretch_accepted = False
    pipeline._stage7_stretch_output = None
    stretched = False
    stage_degraded = False
    messages: List[str] = []
    stretch_method = ""
    separation_state = str(
        getattr(
            pipeline,
            "_star_separation_state",
            StarSeparationState.ACCEPTED.value,
        )
    )
    if separation_state in {
        StarSeparationState.REJECTED.value,
        StarSeparationState.TOOL_FAILED.value,
    }:
        _run_with_stars_review_stretch(pipeline, separation_state)
        return
    pipeline._stage7_stretch_source = (
        "stage6_passthrough"
        if separation_state == StarSeparationState.TARGET_BYPASS.value
        else "stage6_starless"
    )
    if pipeline._ai_stage_advisory_enabled("ai_stage6_enabled"):
        stretched, stage_degraded, ai_messages, stretch_method = (
            pipeline._run_stage6_ai_stretching(allow_ai=True)
        )
        messages.extend(ai_messages)
    else:
        stretched, stage_degraded, stretch_messages, stretch_method = (
            pipeline._run_stage6_ai_stretching(allow_ai=False)
        )
        messages.extend(stretch_messages)
    if stretch_method:
        messages.append(f"拉伸使用 {stretch_method}")

    compare_stem = pipeline._stage7_stretch_source
    pipeline.stretched_name = "stage7_stretched"
    # 拉伸后必须保存，后续 Stage8/9 需要按名加载。
    stage_saved = pipeline._save_stage_output(pipeline.stretched_name) if stretched else False
    # A degraded result remains unsafe unless the stretch service explicitly
    # marks it as a rescue that re-passed every Stage 7 quality gate.
    validated_rescue = bool(
        getattr(pipeline, "_stage7_stretch_validated_rescue", False)
    )
    fallback_reason = str(
        getattr(pipeline, "_stage7_stretch_fallback_reason", "") or ""
    )
    if validated_rescue and not fallback_reason:
        fallback_reason = "validated_stretch_fallback"
    pipeline._stage7_stretch_accepted = bool(
        stretched and stage_saved and (not stage_degraded or validated_rescue)
    )
    if pipeline._stage7_stretch_accepted:
        pipeline._stage7_stretch_output = pipeline.stretched_name
    if stage_saved:
        diff_note = pipeline._stage_diff_note("stage7_stretched", compare_stem)
        if diff_note:
            messages.append(diff_note)
        if hasattr(pipeline, "_create_stage_review_bundle"):
            review = pipeline._create_stage_review_bundle(
                "stage7_stretching",
                compare_stem,
                "stage7_stretched",
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

    elapsed = pipeline.log.stage_end(stage_label)
    message_text = "；".join(messages)
    if stretched and stage_saved:
        status = 'degraded' if stage_degraded and not validated_rescue else 'ok'
        pipeline._record_stage(
            stage_label,
            status,
            elapsed,
            message_text,
            fallback_used=validated_rescue,
            reason_code=(fallback_reason if validated_rescue else ""),
            components={
                "stretch": {
                    "status": "accepted",
                    "method": stretch_method or "unknown",
                    "source": compare_stem,
                    "output": pipeline.stretched_name,
                    "reason_code": (
                        fallback_reason
                        if validated_rescue
                        else "accepted"
                    ),
                    "fallback_used": validated_rescue,
                }
            },
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
        )
    elif getattr(pipeline, "_stage7_review_source", None):
        if message_text:
            message_text = f"{message_text}；仅保留 Stage7 复核候选，禁止正式交付"
        else:
            message_text = "仅保留 Stage7 复核候选，禁止正式交付"
        pipeline._record_stage(
            stage_label,
            'degraded',
            elapsed,
            message_text,
            execution="safe_passthrough",
            reason_code="no_stretch_candidate_passed_quality_gate",
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
            reason_code="all_stretch_candidates_failed",
        )


def run_stage6_stretching(pipeline) -> None:
    """Deprecated compatibility alias for the formal Stage 7 stretch."""
    pipeline.log.warn(
        "run_stage6_stretching() is a legacy alias; use run_stage7_stretching()"
    )
    return run_stage7_stretching(pipeline)
