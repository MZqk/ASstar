"""Stage 1 preparation."""
from typing import List

from models import PipelineStage
from sirilpy.exceptions import SirilError


def run_stage1_preparation(pipeline) -> None:
    """
    阶段 1: 前期准备
    - 创建处理文件夹（每次强制干净目录，支持重复运行）
    - 自动检测输入类型:
      A) 已叠加的 .fit 文件 → 直接加载
      B) Light_ 单帧文件 → 执行预处理
    """
    stage_label = PipelineStage.PREPARATION.label
    pipeline.log.stage_start(stage_label)
    pipeline._prepare_process_dir()

    # 一次遍历查找所有 .fit/.fits 文件
    fit_files = pipeline._find_fit_files()

    # 分类: 候选叠加文件 vs Light_ 原始单帧
    stacked_files = [f for f in fit_files if pipeline._is_candidate_stacked(f)]

    light_files = [
        f for f in fit_files
        if f.name.lower().startswith("light_")
        and f.parent == pipeline.work_dir
    ]

    pipeline._stage1_registration_stats = None

    if stacked_files:
        pipeline._load_stacked_file(stacked_files)
    elif light_files:
        pipeline.log.info(
            f"未找到叠加文件，但发现 {len(light_files)} 个 Light_ 单帧"
        )
        pipeline.log.info("将自动执行预处理: 去拜耳 → 对齐 → 叠加")
        pipeline._stage1_registration_stats = pipeline._preprocess_light_frames(
            light_files
        )
    else:
        raise SirilError(f"未找到任何 .fit 文件，请检查工作目录: {pipeline.work_dir}")

    stage_saved = pipeline._save_stage_output("stage1_prepared")
    stage_status = "ok" if stage_saved else "degraded"
    stage_messages: List[str] = []
    if not stage_saved:
        stage_messages.append("stage1 输出保存失败")

    stats = pipeline._stage1_registration_stats
    if stats:
        total = int(stats.get("total", 0))
        registered = int(stats.get("registered", 0))
        failed = int(stats.get("failed", 0))
        fail_ratio = float(stats.get("fail_ratio", 0.0))
        ratio_limit = max(
            0.0,
            min(1.0, float(pipeline.cfg.stage1_register_fail_ratio_max)),
        )
        stats_msg = (
            f"registration failed={failed}/{total} "
            f"({fail_ratio:.1%}, limit={ratio_limit:.1%})"
        )
        if fail_ratio > ratio_limit:
            stage_status = "degraded"
            stage_messages.append(stats_msg)
            pipeline.log.warn(f"阶段1注册质量门控触发: {stats_msg}")
        else:
            pipeline.log.info(
                f"阶段1注册质量门控通过: failed={failed}/{total}, "
                f"registered={registered}"
            )

    elapsed = pipeline.log.stage_end(stage_label)
    pipeline._record_stage(
        stage_label,
        stage_status,
        elapsed,
        "；".join(stage_messages),
    )
