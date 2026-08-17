"""Stage 1 preparation."""
import os
from pathlib import Path
from typing import Any, List, Mapping, Optional, Tuple

from models import PipelineStage
import run_manifest
from stage_contracts import PIPELINE_CONTRACT_SCHEMA, PIPELINE_CONTRACT_VERSION
from sirilpy.exceptions import SirilError


TASK_RUN_MANIFEST_ENV = "STARUN_TASK_RUN_MANIFEST"
TASK_RUN_MANIFEST_SCHEMA = "starun.task-run.v1"
_TASK_SOURCE_KINDS = frozenset(
    {"master_file", "light_directory", "review_file"}
)


def _load_task_run_source(pipeline) -> Optional[Tuple[str, List[Path]]]:
    """Verify the frozen task-run source list before touching image data."""

    configured = str(os.getenv(TASK_RUN_MANIFEST_ENV, "") or "").strip()
    if not configured:
        return None
    manifest_path = Path(configured).expanduser().resolve()
    work_dir = Path(pipeline.work_dir).resolve()
    if manifest_path.parent != work_dir:
        raise SirilError("任务运行清单必须位于当前独立运行目录")
    payload = run_manifest.load_json(manifest_path)
    if payload is None or str(payload.get("schema") or "") != TASK_RUN_MANIFEST_SCHEMA:
        raise SirilError("任务运行清单缺失或 schema 不受支持")
    expected_hash = str(payload.get("manifest_hash") or "")
    unsigned = dict(payload)
    unsigned.pop("manifest_hash", None)
    if not expected_hash or expected_hash != run_manifest.canonical_payload_hash(
        unsigned
    ):
        raise SirilError("任务运行清单哈希无效")
    contract = payload.get("pipeline_contract")
    if not isinstance(contract, Mapping) or str(
        contract.get("schema") or ""
    ) != PIPELINE_CONTRACT_SCHEMA or str(
        contract.get("version") or ""
    ) != PIPELINE_CONTRACT_VERSION:
        raise SirilError("任务运行清单的 pipeline 契约不兼容")
    source = payload.get("source")
    if not isinstance(source, Mapping) or source.get("read_only") is not True:
        raise SirilError("任务运行清单没有只读来源记录")
    kind = str(source.get("kind") or "").strip().lower()
    if kind not in _TASK_SOURCE_KINDS:
        raise SirilError(f"任务运行清单包含不支持的来源类型: {kind}")
    raw_files = source.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise SirilError("任务运行清单没有来源文件")
    files: List[Path] = []
    for index, raw_record in enumerate(raw_files, start=1):
        if not isinstance(raw_record, Mapping):
            raise SirilError(f"来源文件记录 {index} 无效")
        path = Path(str(raw_record.get("path") or "")).expanduser().resolve()
        if not path.is_file():
            raise SirilError(f"来源文件不存在: {path}")
        expected_size = raw_record.get("size")
        try:
            size_matches = int(expected_size) == int(path.stat().st_size)
        except (OSError, TypeError, ValueError):
            size_matches = False
        if not size_matches:
            raise SirilError(f"来源文件大小已变化: {path}")
        expected_sha256 = str(raw_record.get("sha256") or "")
        actual_sha256 = run_manifest.sha256_file(path)
        if not expected_sha256 or not actual_sha256 or actual_sha256 != expected_sha256:
            raise SirilError(f"来源文件 SHA-256 已变化: {path}")
        files.append(path)
    if kind in {"master_file", "review_file"} and len(files) != 1:
        raise SirilError("单文件任务的来源数量必须为 1")
    pipeline.log.info(
        f"[TaskInput] verified kind={kind} files={len(files)} "
        f"manifest={manifest_path.name}"
    )
    return kind, files


def _load_explicit_master(pipeline, source: Path, source_kind: str) -> None:
    """Load a master read-only and materialize the first FITS inside the task."""

    pipeline._stage1_input_mode = "explicit_master"
    pipeline.source_file = source
    pipeline._stage1_registration_stats = None
    if source.suffix.lower() == ".xisf":
        pipeline.log.info("XISF 以只读方式加载，并在 Stage 1 转换为任务内 FITS")
    elif source_kind == "review_file":
        pipeline.log.warn("渲染图输入只用于安全状态检查与复核导出")
    pipeline.cmd_with_check("cd", f'"{source.parent}"')
    pipeline.cmd_with_check("load", f'"{source.name}"')
    pipeline.cmd_with_check("cd", f'"{pipeline.process_dir}"')
    pipeline.cmd_with_check("save", "working")
    pipeline.cmd_with_check("load", "working")
    pipeline.log.info(f"明确母版已导入任务目录: {source.name} -> working.fit")


def run_stage1_preparation(pipeline) -> None:
    """
    阶段 1: 前期准备
    - 创建处理文件夹（每次强制干净目录，支持重复运行）
    - 自动检测输入类型:
      A) 已叠加的 .fit 文件 → 直接加载
      B) Light_ 单帧文件 → 执行预处理
    """
    stage_label = PipelineStage.PREPARATION.label
    pipeline._clear_stage_reviews(1)
    pipeline.log.stage_start(stage_label)
    pipeline._prepare_process_dir()

    task_source = _load_task_run_source(pipeline)
    if task_source is not None:
        source_kind, source_files = task_source
        if source_kind == "light_directory":
            pipeline.log.info(
                f"签名任务清单指定 {len(source_files)} 个 Light，只处理该分组"
            )
            pipeline._stage1_registration_stats = pipeline._preprocess_light_frames(
                source_files
            )
        else:
            _load_explicit_master(pipeline, source_files[0], source_kind)
        fit_files = []
        stacked_files = []
        light_files = []
    else:
        # Standalone developer runs may still start from an explicit work dir.
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
            raise SirilError(
                f"未找到任何 .fit 文件，请检查工作目录: {pipeline.work_dir}"
            )

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
