"""GUI-facing preparation of deterministic, serial product-task queues."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

try:
    from pipeline.input_discovery import (
        InputDiscovery,
        InputKind,
        LightGroup,
        discover_input,
    )
    from pipeline.task_plan import build_resume_fingerprints
    from pipeline.task_workspace import (
        TaskRun,
        TaskWorkspace,
        WorkspaceError,
        begin_task_run,
        build_source_record,
        ensure_task_workspace,
        inspect_task_workspace,
        open_task_workspace,
    )
except ImportError:  # Support direct execution from the gui directory.
    import sys

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from pipeline.input_discovery import (
        InputDiscovery,
        InputKind,
        LightGroup,
        discover_input,
    )
    from pipeline.task_plan import build_resume_fingerprints
    from pipeline.task_workspace import (
        TaskRun,
        TaskWorkspace,
        WorkspaceError,
        begin_task_run,
        build_source_record,
        ensure_task_workspace,
        inspect_task_workspace,
        open_task_workspace,
    )


INPUT_MODE_AUTO = "auto"
INPUT_MODE_STAGE1_PREPARED_RESUME = "stage1_prepared_resume"
INPUT_MODE_STAGE2_CORRECTED_RESUME = "stage2_corrected_resume"
INPUT_MODE_STAGE5_LINEAR_RESUME = "result_linear_resume"
TASK_RUN_MANIFEST_ENV = "SEESTAR_TASK_RUN_MANIFEST"
RESUME_CHECKPOINT_PATH_ENV = "SEESTAR_RESUME_CHECKPOINT_PATH"


@dataclass(frozen=True)
class PreparedTask:
    queue_index: int
    queue_total: int
    workspace: TaskWorkspace
    run: TaskRun
    source_record: Mapping[str, Any]
    input_mode: str
    resume_after_stage: Optional[int]
    checkpoint_fingerprints: Mapping[str, Mapping[str, Any]]
    runtime_overrides: Mapping[str, str]
    display_label: str


@dataclass(frozen=True)
class PreparedTaskQueue:
    tasks: Tuple[PreparedTask, ...]

    def __post_init__(self) -> None:
        if not self.tasks:
            raise WorkspaceError("任务队列不能为空")


@dataclass(frozen=True)
class TaskPlanPresentation:
    summary: str
    linear_phase: str
    nonlinear_phase: str


def describe_input_plan(discovery: InputDiscovery) -> TaskPlanPresentation:
    """Return one concise, truthful plan for the task setup card."""

    if discovery.kind == InputKind.PRODUCT_TASK and discovery.resume_after_stage:
        stage_number = int(discovery.resume_after_stage)
        return TaskPlanPresentation(
            summary=(
                f"自动计划：Stage 1–{stage_number} 已验证，"
                f"从 Stage {stage_number + 1} 继续。"
            ),
            linear_phase=(
                f"✓ Stage 1–{stage_number} · ○ Stage {stage_number + 1}–6"
            ),
            nonlinear_phase="○ 将执行 · Stage 7–10",
        )
    if discovery.kind == InputKind.REVIEW_FILE:
        return TaskPlanPresentation(
            summary=(
                "自动计划：Stage 1 导入、Stage 2 安全校正，"
                "跳过 Stage 3–9，由 Stage 10 生成复核输出。"
            ),
            linear_phase="○ 安全导入 · Stage 1–2",
            nonlinear_phase="○ 仅复核导出 · Stage 10",
        )
    if discovery.kind == InputKind.LIGHT_DIRECTORY:
        return TaskPlanPresentation(
            summary=(
                f"自动计划：{len(discovery.light_groups)} 个分组分别从 Stage 1 "
                "重新叠加并串行处理；线性校验不通过时自动改为复核输出。"
            ),
            linear_phase="○ 重新叠加与线性处理 · Stage 1–6",
            nonlinear_phase="○ 校验通过后执行 · Stage 7–10",
        )
    if discovery.kind == InputKind.LEGACY_DIRECTORY:
        return TaskPlanPresentation(
            summary=(
                "自动计划：保持旧目录只读，将已验证旧断点"
                "作为外部母版从 Stage 1 安全导入。"
            ),
            linear_phase="○ 只读迁移与线性处理 · Stage 1–6",
            nonlinear_phase="○ 校验通过后执行 · Stage 7–10",
        )
    if discovery.kind in {InputKind.MASTER_FILE, InputKind.PRODUCT_TASK}:
        return TaskPlanPresentation(
            summary=(
                "自动计划：从 Stage 1 导入并检查线性状态；"
                "通过后执行 Stage 2–10，未通过则只生成复核输出。"
            ),
            linear_phase="○ 导入与线性处理 · Stage 1–6",
            nonlinear_phase="○ 校验通过后执行 · Stage 7–10",
        )
    return TaskPlanPresentation(
        summary="自动计划：当前输入不可执行，请先修正输入。",
        linear_phase="— 线性处理未计划",
        nonlinear_phase="— 非线性处理未计划",
    )


def stage_config_from_processing_settings(
    settings: Mapping[str, Any],
) -> Dict[int, Dict[str, Any]]:
    """Map only result-affecting Stage 1-5 settings into resume hashes."""

    values = dict(settings or {})
    return {
        1: {
            "source_import": "read_only_task_manifest",
            "light_preprocess": "debayer_register_stack_v1",
        },
        2: {"boundary_correction": "autocrop_v1"},
        3: {"background_policy": "safe_candidate_v1"},
        4: {
            key: values.get(key)
            for key in (
                "color_calibration",
                "filter_hint",
                "pcc_timeout_sec",
                "local_wb_gain_limit",
            )
        },
        5: {
            key: values.get(key)
            for key in (
                "denoise_mode",
                "deconvolution_mode",
                "graxpert_model_path",
                "compute_mode",
                "builtin_denoise_strength",
                "graxpert_deconv_strength",
                "rl_iterations",
                "rl_maxstars",
            )
        },
    }


def _run_id(now: Optional[datetime] = None) -> str:
    current = now or datetime.now(timezone.utc)
    stamp = current.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def _light_group_source_record(
    discovery: InputDiscovery,
    group: LightGroup,
    *,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    return build_source_record(
        source_kind=InputKind.LIGHT_DIRECTORY.value,
        selected_path=discovery.selected_path,
        files=group.files,
        group={
            "key": group.key,
            "target": group.target,
            "filter": group.filter_name,
            "camera": group.camera,
            "geometry": group.geometry,
        },
        cancel_check=cancel_check,
    )


def _external_source_records(
    discovery: InputDiscovery,
    *,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Tuple[Tuple[Dict[str, Any], str], ...]:
    if discovery.kind in {InputKind.MASTER_FILE, InputKind.REVIEW_FILE}:
        if discovery.master_file is None:
            raise WorkspaceError("单文件识别结果缺少明确文件")
        source = build_source_record(
            source_kind=discovery.kind.value,
            selected_path=discovery.selected_path,
            files=(discovery.master_file,),
            cancel_check=cancel_check,
        )
        return ((source, discovery.master_file.stem),)
    if discovery.kind == InputKind.LEGACY_DIRECTORY:
        if discovery.master_file is None:
            raise WorkspaceError("旧版目录没有已验证的迁移源")
        source = build_source_record(
            source_kind=InputKind.MASTER_FILE.value,
            selected_path=discovery.selected_path,
            files=(discovery.master_file,),
            group={
                "migration": "legacy_read_only_v1",
                "legacy_directory": str(discovery.selected_path),
                "legacy_checkpoint": discovery.details.get("checkpoint"),
                "legacy_stage": discovery.details.get("legacy_stage"),
                "legacy_manifest_hash": discovery.details.get("manifest_hash"),
                "legacy_plan_hash": discovery.details.get("plan_hash"),
            },
            cancel_check=cancel_check,
        )
        return ((source, f"旧版迁移 · {discovery.master_file.name}"),)
    if discovery.kind == InputKind.LIGHT_DIRECTORY:
        return tuple(
            (
                _light_group_source_record(
                    discovery,
                    group,
                    cancel_check=cancel_check,
                ),
                group.display_label,
            )
            for group in discovery.light_groups
        )
    raise WorkspaceError(f"输入类型不能创建新任务：{discovery.kind.value}")


def _mode_for_resume(stage_number: Optional[int]) -> str:
    return {
        1: INPUT_MODE_STAGE1_PREPARED_RESUME,
        2: INPUT_MODE_STAGE2_CORRECTED_RESUME,
        5: INPUT_MODE_STAGE5_LINEAR_RESUME,
    }.get(stage_number, INPUT_MODE_AUTO)


def discover_input_for_processing_settings(
    selected_path: Path,
    *,
    processing_settings: Mapping[str, Any],
) -> InputDiscovery:
    """Discover a product task against the exact Stage 1-5 settings in the UI."""

    expanded = selected_path.expanduser()
    try:
        path = expanded.resolve()
    except OSError:
        path = expanded
    if not path.is_dir() or not (path / "task-manifest.json").is_file():
        return discover_input(path)
    try:
        workspace, _source = open_task_workspace(path)
        fingerprints = build_resume_fingerprints(
            input_fingerprint=workspace.source_fingerprint,
            stage_config=stage_config_from_processing_settings(
                processing_settings
            ),
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        return discover_input(path)
    return discover_input(
        path,
        current_resume_fingerprints=fingerprints,
    )


def prepare_task_queue(
    discovery: InputDiscovery,
    *,
    processing_settings: Mapping[str, Any],
    task_container: Optional[Path] = None,
    now: Optional[datetime] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> PreparedTaskQueue:
    """Materialize one or more tasks; execution remains strictly serial."""

    if not discovery.accepted:
        raise WorkspaceError(discovery.errors[0] if discovery.errors else discovery.summary)
    stage_config = stage_config_from_processing_settings(processing_settings)
    prepared: list[PreparedTask] = []

    if discovery.kind == InputKind.PRODUCT_TASK:
        if cancel_check is not None and cancel_check():
            raise InterruptedError("任务准备已取消")
        if discovery.task_directory is None:
            raise WorkspaceError("产品任务识别结果缺少任务目录")
        workspace, source = open_task_workspace(discovery.task_directory)
        current_fingerprints = build_resume_fingerprints(
            input_fingerprint=workspace.source_fingerprint,
            stage_config=stage_config,
        )
        inspection = inspect_task_workspace(
            workspace.root,
            current_resume_fingerprints=current_fingerprints,
        )
        if cancel_check is not None and cancel_check():
            raise InterruptedError("任务准备已取消")
        resume_stage = (
            int(inspection["resume_after_stage"])
            if inspection.get("verified") and inspection.get("resume_after_stage")
            else None
        )
        resume_record = (
            inspection.get("resume_record")
            if resume_stage is not None
            else None
        )
        run = begin_task_run(
            workspace=workspace,
            source_record=source,
            run_id=_run_id(now),
            resume_record=(
                resume_record if isinstance(resume_record, Mapping) else None
            ),
            checkpoint_fingerprints=current_fingerprints,
        )
        overrides = {TASK_RUN_MANIFEST_ENV: str(run.manifest_path)}
        if isinstance(resume_record, Mapping):
            overrides[RESUME_CHECKPOINT_PATH_ENV] = str(resume_record["path"])
        prepared.append(
            PreparedTask(
                queue_index=1,
                queue_total=1,
                workspace=workspace,
                run=run,
                source_record=source,
                input_mode=_mode_for_resume(resume_stage),
                resume_after_stage=resume_stage,
                checkpoint_fingerprints=current_fingerprints,
                runtime_overrides=overrides,
                display_label=workspace.task_id,
            )
        )
    else:
        records = _external_source_records(
            discovery,
            cancel_check=cancel_check,
        )
        total = len(records)
        for index, (source, label) in enumerate(records, start=1):
            if cancel_check is not None and cancel_check():
                raise InterruptedError("任务准备已取消")
            current_fingerprints = build_resume_fingerprints(
                input_fingerprint=str(source["fingerprint"]),
                stage_config=stage_config,
            )
            workspace = ensure_task_workspace(
                source_record=source,
                selected_path=discovery.selected_path,
                task_container=task_container,
            )
            run = begin_task_run(
                workspace=workspace,
                source_record=source,
                run_id=_run_id(now),
                checkpoint_fingerprints=current_fingerprints,
            )
            prepared.append(
                PreparedTask(
                    queue_index=index,
                    queue_total=total,
                    workspace=workspace,
                    run=run,
                    source_record=source,
                    input_mode=INPUT_MODE_AUTO,
                    resume_after_stage=None,
                    checkpoint_fingerprints=current_fingerprints,
                    runtime_overrides={
                        TASK_RUN_MANIFEST_ENV: str(run.manifest_path),
                    },
                    display_label=label or workspace.task_id,
                )
            )
    return PreparedTaskQueue(tasks=tuple(prepared))


__all__ = [
    "PreparedTask",
    "PreparedTaskQueue",
    "RESUME_CHECKPOINT_PATH_ENV",
    "TASK_RUN_MANIFEST_ENV",
    "TaskPlanPresentation",
    "describe_input_plan",
    "discover_input_for_processing_settings",
    "prepare_task_queue",
    "stage_config_from_processing_settings",
]
