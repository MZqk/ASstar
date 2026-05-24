"""Stage 6 stretch selection and execution."""
from typing import List


def run_stage6_stretching(pipeline) -> None:
    """
    阶段 6: 主体拉伸
    - 按目标类型/背景/核心动态范围自动选择 Asinh、Asinh+GHS、GHS
    - 星团只用 Asinh；高动态/暗弱星云使用受保护的 Asinh+GHS
    - 按策略决定是否尝试 VeraLux Curves 曲线微调
    """
    pipeline.log.stage_start("阶段 6: 主体拉伸")
    stretched = False
    stage_degraded = False
    messages: List[str] = []
    stretch_method = ""
    weak_object_note = pipeline._apply_weak_object_tuning()
    if weak_object_note:
        messages.append(weak_object_note)

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

    pipeline.stretched_name = "stage6_stretched"
    # 拉伸后必须保存 (阶段7需要按名加载)
    stage_saved = pipeline._save_stage_output(pipeline.stretched_name)
    if stage_saved:
        diff_note = pipeline._stage_diff_note("stage6_stretched", "stage5_denoised")
        if diff_note:
            messages.append(diff_note)
        feature_note = pipeline._feature_summary_note("拉伸后特征")
        if feature_note:
            pipeline.log.info(f"[Stage6] {feature_note}")
            feat = pipeline._measure_current_features()
            if feat is not None:
                messages.append(
                    "stage6_features="
                    f"bg_median={feat.bg_median:.4f}, "
                    f"object_area={feat.object_area_ratio:.3f}, "
                    f"edge_black={feat.edge_black_ratio:.3f}"
                )

    elapsed = pipeline.log.stage_end("阶段 6: 主体拉伸")
    message_text = "；".join(messages)
    if stretched and stage_saved:
        status = 'degraded' if stage_degraded else 'ok'
        pipeline._record_stage("阶段 6: 主体拉伸", status, elapsed, message_text)
    elif stretched and not stage_saved:
        if message_text:
            message_text = f"{message_text}；stage6 输出保存失败"
        else:
            message_text = "stage6 输出保存失败"
        pipeline._record_stage("阶段 6: 主体拉伸", 'degraded', elapsed, message_text)
    else:
        if message_text:
            message_text = f"{message_text}；所有拉伸方法均失败"
        else:
            message_text = "所有拉伸方法均失败"
        pipeline._record_stage("阶段 6: 主体拉伸", 'failed', elapsed, message_text)

