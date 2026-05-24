"""Stage 2 view correction and crop."""
from typing import List

from sirilpy.exceptions import CommandError, SirilError


def run_stage2_view_correction(pipeline) -> None:
    """
    阶段 2: 裁切
    - 按工作流先做画面边缘裁切
    - 图像解析（天体测量）在阶段4执行
    """
    pipeline.log.stage_start("阶段 2: 裁切")
    status = "ok"
    messages: List[str] = []

    # 裁切边缘
    pipeline.log.info(f"裁切边缘 ({pipeline.cfg.crop_margin:.0%})...")
    try:
        shape = pipeline.siril.get_image_shape()
        if shape:
            channels, height, width = shape
            margin_x = int(width * pipeline.cfg.crop_margin)
            margin_y = int(height * pipeline.cfg.crop_margin)
            crop_w = width - 2 * margin_x
            crop_h = height - 2 * margin_y
            if crop_w > 0 and crop_h > 0:
                pipeline.cmd_with_check(
                    "crop",
                    str(margin_x),
                    str(margin_y),
                    str(crop_w),
                    str(crop_h),
                )
                pipeline.log.info(f"已裁切 (margin: {margin_x}x{margin_y} px)")
            else:
                pipeline.log.warn("图像太小，跳过裁切")
        else:
            pipeline.log.warn("无法获取图像尺寸，跳过裁切")
    except (CommandError, SirilError) as e:
        pipeline.log.warn(f"裁切失败: {e}")
        status = "degraded"

    if status == "ok":
        target = float(getattr(pipeline.cfg, "stage2_edge_black_target", 0.10))
        max_passes = int(getattr(pipeline.cfg, "stage2_adaptive_edge_crop_max_passes", 3))
        max_extra = float(getattr(pipeline.cfg, "stage2_adaptive_edge_crop_max_extra", 0.035))
        last_edge_black = None
        for pass_index in range(max_passes):
            edge_feat = pipeline._measure_current_features()
            if edge_feat is None:
                messages.append("adaptive edge crop skipped: feature sampling unavailable")
                break
            last_edge_black = edge_feat.edge_black_ratio
            if edge_feat.edge_black_ratio <= target:
                if pass_index == 0:
                    messages.append(
                        f"adaptive edge crop not needed (edge_black={edge_feat.edge_black_ratio:.3f})"
                    )
                break
            try:
                adaptive_note = pipeline._apply_adaptive_edge_crop(
                    edge_feat,
                    trigger=target,
                    target=target,
                    max_extra_margin=max_extra,
                )
                after_feat = pipeline._measure_current_features()
                if after_feat is not None:
                    messages.append(
                        "stage2 pass "
                        f"{pass_index + 1} metrics: edge_black "
                        f"{edge_feat.edge_black_ratio:.3f}->{after_feat.edge_black_ratio:.3f}, "
                        f"global_dark={getattr(edge_feat, 'global_dark_ratio', 0.0):.3f}"
                        f"->{getattr(after_feat, 'global_dark_ratio', 0.0):.3f}"
                    )
                    if after_feat.edge_black_ratio >= edge_feat.edge_black_ratio - 0.003:
                        status = "degraded" if status == "ok" else status
                        messages.append(
                            "adaptive edge crop stopped: degraded_no_improvement "
                            f"(edge_black {edge_feat.edge_black_ratio:.3f}->{after_feat.edge_black_ratio:.3f})"
                        )
                        break
                if adaptive_note:
                    messages.append(f"stage2 pass {pass_index + 1}: {adaptive_note}")
                else:
                    messages.append(
                        "adaptive edge crop skipped "
                        f"(edge_black={edge_feat.edge_black_ratio:.3f}, target={target:.3f})"
                    )
                    break
            except (CommandError, SirilError) as e:
                pipeline.log.warn(f"自适应黑边裁切失败: {e}")
                status = "degraded"
                messages.append(
                    f"adaptive edge crop failed: {pipeline._short_text(e, 160)}"
                )
                break
        final_feat = pipeline._measure_current_features()
        if final_feat is not None:
            messages.append(
                "stage2 edge_black "
                f"{last_edge_black if last_edge_black is not None else final_feat.edge_black_ratio:.3f}"
                f"->{final_feat.edge_black_ratio:.3f} (target={target:.3f})"
            )
            if final_feat.edge_black_ratio > target:
                status = "degraded" if status == "ok" else status
                messages.append(
                    f"edge_black remains above stage2 target: {final_feat.edge_black_ratio:.3f}>{target:.3f}"
                )

    stage_saved = pipeline._save_stage_output("stage2_corrected")
    if not stage_saved and status == "ok":
        status = "degraded"
        messages.append("stage2 输出保存失败")

    elapsed = pipeline.log.stage_end("阶段 2: 裁切")
    pipeline._record_stage("阶段 2: 裁切", status, elapsed, "；".join(messages))
