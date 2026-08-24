"""Independent, source-read-only product task workspaces and checkpoints."""
from __future__ import annotations

import hashlib
import re
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

try:
    from . import outcome, run_manifest, scene_support, task_plan
    from .processing_parameters import (
        default_processing_parameters,
        normalize_processing_parameters,
        processing_gate_profile_audit,
    )
    from .stage_contracts import (
        FORMAL_RESUME_STAGES,
        PIPELINE_CONTRACT_SCHEMA,
        PIPELINE_CONTRACT_VERSION,
        pipeline_contract_manifest,
        stage_contract,
    )
except ImportError:
    import outcome
    import run_manifest
    import scene_support
    import task_plan
    from processing_parameters import (
        default_processing_parameters,
        normalize_processing_parameters,
        processing_gate_profile_audit,
    )
    from stage_contracts import (
        FORMAL_RESUME_STAGES,
        PIPELINE_CONTRACT_SCHEMA,
        PIPELINE_CONTRACT_VERSION,
        pipeline_contract_manifest,
        stage_contract,
    )


TASK_MANIFEST_SCHEMA = "starun.task-manifest.v1"
CHECKPOINT_MANIFEST_SCHEMA = "starun.checkpoint-manifest.v1"
RUN_MANIFEST_SCHEMA = "starun.task-run.v1"
RESUME_SEMANTIC_SCHEMA_V1 = "starun.resume-semantics.v1"
RESUME_SEMANTIC_SCHEMA_V2 = "starun.resume-semantics.v2"
RESUME_SEMANTIC_SCHEMA = RESUME_SEMANTIC_SCHEMA_V2
SUPPORTED_RESUME_SEMANTIC_SCHEMAS = frozenset(
    {RESUME_SEMANTIC_SCHEMA_V1, RESUME_SEMANTIC_SCHEMA_V2}
)
TASK_MANIFEST_NAME = "task-manifest.json"
RUN_MANIFEST_NAME = "run-manifest.json"
CHECKPOINT_MANIFEST_REL = Path("checkpoints") / "checkpoint-manifest.json"
LATEST_RESULT_MANIFEST_REL = Path("results") / "latest-result.json"
RETENTION_MANIFEST_REL = Path("results") / "retention.json"
TASK_CONTAINER_NAME = "Starun"
STAGE2_STACKED_FOOTPRINT_SCHEMA = (
    "starun.stage2-stacked-footprint-evidence.v1"
)


class WorkspaceError(RuntimeError):
    pass


def _normalize_stage2_footprint_grid(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise WorkspaceError("Stage 2 footprint 网格必须为映射")
    if str(value.get("encoding") or "") != "rle-u8-row-major-v1":
        raise WorkspaceError("Stage 2 footprint 网格编码不受支持")
    rows = value.get("rows")
    columns = value.get("columns")
    if (
        isinstance(rows, bool)
        or not isinstance(rows, int)
        or isinstance(columns, bool)
        or not isinstance(columns, int)
    ):
        raise WorkspaceError("Stage 2 footprint 网格尺寸无效")
    if not (1 <= rows <= 64 and 1 <= columns <= 64):
        raise WorkspaceError("Stage 2 footprint 网格尺寸超出 64×64")
    raw_runs = value.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise WorkspaceError("Stage 2 footprint 网格缺少 RLE 数据")
    decoded = bytearray()
    normalized_runs: list[list[int]] = []
    for raw_run in raw_runs:
        if not isinstance(raw_run, (list, tuple)) or len(raw_run) != 2:
            raise WorkspaceError("Stage 2 footprint RLE 条目无效")
        grid_value = raw_run[0]
        count = raw_run[1]
        if (
            isinstance(grid_value, bool)
            or not isinstance(grid_value, int)
            or isinstance(count, bool)
            or not isinstance(count, int)
        ):
            raise WorkspaceError("Stage 2 footprint RLE 数值无效")
        if not (0 <= grid_value <= 100) or count <= 0:
            raise WorkspaceError("Stage 2 footprint RLE 范围无效")
        if len(decoded) + count > rows * columns:
            raise WorkspaceError("Stage 2 footprint RLE 长度超出网格")
        decoded.extend([grid_value] * count)
        normalized_runs.append([grid_value, count])
    if len(decoded) != rows * columns:
        raise WorkspaceError("Stage 2 footprint RLE 长度与网格不匹配")
    claimed_sha256 = value.get("sha256")
    actual_sha256 = hashlib.sha256(bytes(decoded)).hexdigest()
    if not isinstance(claimed_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}",
        claimed_sha256,
    ):
        raise WorkspaceError("Stage 2 footprint 网格 SHA-256 无效")
    if claimed_sha256 != actual_sha256:
        raise WorkspaceError("Stage 2 footprint 网格 SHA-256 不匹配")
    return {
        "rows": rows,
        "columns": columns,
        "encoding": "rle-u8-row-major-v1",
        "runs": normalized_runs,
        "sha256": claimed_sha256,
    }


def _normalize_stage2_stacked_footprint(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise WorkspaceError("Stage 2 footprint 证据必须为映射")
    normalized = _json_safe_contract_value(dict(value))
    if not isinstance(normalized, dict):
        raise WorkspaceError("Stage 2 footprint 证据无法规范化")
    if normalized.get("schema") != STAGE2_STACKED_FOOTPRINT_SCHEMA:
        raise WorkspaceError("Stage 2 footprint schema 不受支持")
    if normalized.get("source_mode") != "stacked_master_inference":
        raise WorkspaceError("Stage 2 footprint 来源模式无效")
    if normalized.get("observer_only") is not True:
        raise WorkspaceError("Stage 2 footprint 必须保持 observer-only")
    if normalized.get("captured_before_crop") is not True:
        raise WorkspaceError("Stage 2 footprint 必须在裁切前采集")
    status = str(normalized.get("status") or "")
    if status not in {"available", "partial", "unavailable"}:
        raise WorkspaceError("Stage 2 footprint 状态无效")
    source_sha256 = normalized.get("source_sha256")
    if source_sha256 is not None and (
        not isinstance(source_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", source_sha256)
    ):
        raise WorkspaceError("Stage 2 footprint 来源 SHA-256 无效")
    if status != "unavailable" and source_sha256 is None:
        raise WorkspaceError("Stage 2 footprint 缺少 64 位来源 SHA-256")
    source_artifact = normalized.get("source_artifact")
    if status != "unavailable" and not str(source_artifact or "").strip():
        raise WorkspaceError("Stage 2 footprint 缺少来源文件名")
    if not isinstance(normalized.get("input_shape"), Mapping):
        raise WorkspaceError("Stage 2 footprint 缺少输入尺寸")
    limitations = normalized.get("limitations")
    if not isinstance(limitations, list) or not {
        "not_per_frame_registration_footprint",
        "not_crop_authority",
    }.issubset({str(item) for item in limitations}):
        raise WorkspaceError("Stage 2 footprint 缺少证据边界")
    layers = normalized.get("layers")
    if not isinstance(layers, Mapping):
        raise WorkspaceError("Stage 2 footprint 缺少证据层")

    available_layers = 0
    normalized_layers: Dict[str, Any] = {}
    for key in ("fill_support", "relative_coverage"):
        layer = layers.get(key)
        if layer is None and status == "unavailable":
            continue
        if not isinstance(layer, Mapping):
            raise WorkspaceError(f"Stage 2 footprint 缺少 {key}")
        layer_status = str(layer.get("status") or "")
        if layer_status not in {"available", "unavailable"}:
            raise WorkspaceError(f"Stage 2 footprint {key} 状态无效")
        normalized_layer = dict(layer)
        grid = layer.get("grid")
        if layer_status == "available":
            normalized_layer["grid"] = _normalize_stage2_footprint_grid(grid)
            available_layers += 1
        elif grid is not None:
            raise WorkspaceError(f"Stage 2 footprint {key} 不可用但包含网格")
        normalized_layers[key] = normalized_layer
    expected_status = (
        "available"
        if available_layers == 2
        else "partial"
        if available_layers == 1
        else "unavailable"
    )
    if status != expected_status:
        raise WorkspaceError("Stage 2 footprint 状态与可用证据层不一致")
    normalized["layers"] = normalized_layers
    return normalized


def _json_safe_contract_value(value: Any) -> Any:
    """Normalize checkpoint metadata without admitting executable objects."""
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe_contract_value(child)
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_contract_value(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    scalar = getattr(value, "item", None)
    if callable(scalar):
        try:
            return _json_safe_contract_value(scalar())
        except (TypeError, ValueError):
            pass
    return str(value)


def _normalize_resume_semantic_context(
    value: Any,
    *,
    stage_number: int,
) -> Optional[Dict[str, Any]]:
    """Validate the signed upstream semantics carried by a formal checkpoint."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise WorkspaceError("续跑语义契约必须为映射")
    normalized = _json_safe_contract_value(
        run_manifest.redact_sensitive(dict(value))
    )
    if not isinstance(normalized, dict):
        raise WorkspaceError("续跑语义契约无法规范化")
    source_schema = str(normalized.get("schema") or "")
    if source_schema not in SUPPORTED_RESUME_SEMANTIC_SCHEMAS:
        raise WorkspaceError("续跑语义契约 schema 不受支持")
    try:
        semantic_stage = int(normalized.get("checkpoint_stage"))
    except (TypeError, ValueError) as error:
        raise WorkspaceError("续跑语义契约缺少 checkpoint_stage") from error
    if semantic_stage != stage_number:
        raise WorkspaceError("续跑语义契约与断点阶段不匹配")
    if source_schema == RESUME_SEMANTIC_SCHEMA_V1:
        if stage_number != 5:
            raise WorkspaceError("旧版续跑语义仅支持 Stage 5")
        legacy_review = normalized.get("upstream_review")
        if not isinstance(legacy_review, Mapping):
            raise WorkspaceError("Stage 5 旧版续跑语义缺少 upstream_review")
        legacy_mapping = (
            (
                "stage2_view_review_required",
                2,
                "legacy_stage2_view_review_required",
            ),
            (
                "background_review_required",
                3,
                "legacy_stage3_background_review_required",
            ),
            (
                "color_review_required",
                4,
                "legacy_stage4_color_review_required",
            ),
        )
        normalized["review_requirements"] = [
            outcome.normalize_review_requirement(
                {"stage": stage, "code": code, "details": {}},
                legacy_inferred=True,
            )
            for field, stage, code in legacy_mapping
            if bool(legacy_review.get(field, False))
        ]
        normalized["source_schema"] = source_schema
        normalized["legacy_inferred"] = True
        normalized["schema"] = RESUME_SEMANTIC_SCHEMA_V2
    else:
        raw_reviews = normalized.get("review_requirements")
        if not isinstance(raw_reviews, list):
            raise WorkspaceError("续跑语义缺少结构化 review_requirements")
        try:
            normalized["review_requirements"] = (
                outcome.deduplicate_review_requirements(
                    item for item in raw_reviews if isinstance(item, Mapping)
                )
            )
        except ValueError as error:
            raise WorkspaceError(f"续跑复核语义无效：{error}") from error

    if stage_number == 2:
        stage2_crop = normalized.get("stage2_crop")
        if not isinstance(stage2_crop, Mapping):
            raise WorkspaceError("Stage 2 续跑语义缺少裁切语义")
        for key in (
            "original_dimensions",
            "final_dimensions",
            "cumulative_crop",
            "final_residual_detection",
        ):
            if not isinstance(stage2_crop.get(key), Mapping):
                raise WorkspaceError(f"Stage 2 续跑裁切语义缺少 {key}")
        try:
            field_rotation_passes = int(
                stage2_crop.get("field_rotation_passes", 0)
            )
        except (TypeError, ValueError) as error:
            raise WorkspaceError("Stage 2 场旋裁切轮次无效") from error
        if field_rotation_passes not in range(0, 3):
            raise WorkspaceError("Stage 2 场旋裁切轮次超出 0-2")
        normalized_stage2_crop = dict(stage2_crop)
        stacked_footprint = stage2_crop.get("stacked_master_footprint")
        if stacked_footprint is not None:
            normalized_stage2_crop["stacked_master_footprint"] = (
                _normalize_stage2_stacked_footprint(stacked_footprint)
            )
        normalized["stage2_crop"] = normalized_stage2_crop
    if stage_number == 5:
        if not str(normalized.get("channel_semantics") or "").strip():
            raise WorkspaceError("Stage 5 续跑语义缺少通道语义")
        required_mappings = (
            "channel_profile",
            "target_profile",
            "pipeline_policy",
            "color_calibration_report",
        )
        for key in required_mappings:
            if not isinstance(normalized.get(key), Mapping):
                raise WorkspaceError(f"Stage 5 续跑语义缺少 {key}")
        if not normalized.get("narrowband_channel_mapping"):
            raise WorkspaceError("Stage 5 续跑语义缺少通道映射契约")
        star_reference = normalized.get("stage5_star_reference_report")
        if star_reference is not None and not isinstance(star_reference, Mapping):
            raise WorkspaceError("Stage 5 续跑星表语义无效")
        shared_support = normalized.get("stage3_scene_support")
        if shared_support is not None:
            if not isinstance(shared_support, Mapping) or str(
                shared_support.get("schema") or ""
            ) != scene_support.SCENE_SUPPORT_SCHEMA:
                raise WorkspaceError("Stage 5 共享场景支持语义无效")
            for field in (
                "source_file_sha256",
                "source_pixel_sha256",
                "manifest_sha256",
                "arrays_sha256",
            ):
                value = shared_support.get(field)
                if value is not None and (
                    len(str(value)) != 64
                    or any(char not in "0123456789abcdef" for char in str(value).lower())
                ):
                    raise WorkspaceError(f"Stage 5 共享场景支持 {field} 无效")
    return normalized


@dataclass(frozen=True)
class TaskWorkspace:
    task_id: str
    root: Path
    manifest_path: Path
    source_fingerprint: str
    reused: bool

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"


@dataclass(frozen=True)
class TaskRun:
    run_id: str
    root: Path
    manifest_path: Path
    task: TaskWorkspace


def _signed_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    signed = dict(payload)
    signed["manifest_hash"] = run_manifest.canonical_payload_hash(signed)
    return signed


def _load_signed_payload(path: Path, schema: str) -> tuple[Optional[Dict[str, Any]], str]:
    payload = run_manifest.load_json(path)
    if payload is None:
        return None, f"{path.name} is missing or invalid"
    if str(payload.get("schema") or "") != schema:
        return None, f"unsupported {path.name} schema"
    expected_hash = str(payload.get("manifest_hash") or "")
    unsigned = dict(payload)
    unsigned.pop("manifest_hash", None)
    actual_hash = run_manifest.canonical_payload_hash(unsigned)
    if not expected_hash or expected_hash != actual_hash:
        return None, f"{path.name} hash is missing or invalid"
    return payload, "verified"


def _safe_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except OSError:
        return 0


def _source_file_record(
    path: Path,
    *,
    display_path: str,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    sha256 = run_manifest.sha256_file(path, cancel_check=cancel_check)
    if not sha256:
        raise WorkspaceError(f"无法读取输入文件：{path}")
    return {
        "path": str(path),
        "display_path": display_path,
        "size": _safe_size(path),
        "sha256": sha256,
    }


def build_source_record(
    *,
    source_kind: str,
    selected_path: Path,
    files: Sequence[Path],
    group: Optional[Mapping[str, Any]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """Hash an exact master or one Light group without copying source data."""

    kind = str(source_kind or "").strip().lower()
    if kind not in {"master_file", "light_directory", "review_file"}:
        raise WorkspaceError(f"不支持创建任务的输入类型：{source_kind!r}")
    selected = selected_path.expanduser().resolve()
    normalized_files = tuple(path.expanduser().resolve() for path in files)
    if not normalized_files:
        raise WorkspaceError("任务输入至少需要一个文件")
    if kind in {"master_file", "review_file"} and len(normalized_files) != 1:
        raise WorkspaceError("单文件任务只能包含一个明确输入")

    records = []
    for path in sorted(normalized_files, key=lambda item: item.as_posix().lower()):
        if not path.is_file():
            raise WorkspaceError(f"输入文件不存在：{path}")
        try:
            relative = path.relative_to(selected if selected.is_dir() else selected.parent)
            display_path = relative.as_posix()
        except ValueError:
            display_path = path.name
        records.append(
            _source_file_record(
                path,
                display_path=display_path,
                cancel_check=cancel_check,
            )
        )

    fingerprint_payload = {
        "source_kind": kind,
        "group": dict(group or {}),
        "files": [
            {
                "display_path": record["display_path"],
                "size": record["size"],
                "sha256": record["sha256"],
            }
            for record in records
        ],
    }
    source_fingerprint = run_manifest.canonical_payload_hash(fingerprint_payload)
    return {
        "kind": kind,
        "selected_path": str(selected),
        "read_only": True,
        "fingerprint": source_fingerprint,
        "file_count": len(records),
        "total_bytes": sum(int(record["size"]) for record in records),
        "files": records,
        "group": dict(group or {}),
    }


def _task_slug(value: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9]+", str(value or "").lower())
    return "-".join(tokens)[:48].strip("-") or "task"


def _source_label(source: Mapping[str, Any]) -> str:
    group = source.get("group")
    if isinstance(group, Mapping):
        target = str(group.get("target") or "").strip()
        if target and target.lower() != "unknown":
            return target
    files = source.get("files")
    if isinstance(files, Sequence) and files:
        first = files[0]
        if isinstance(first, Mapping):
            return Path(str(first.get("display_path") or "task")).stem
    return "task"


def default_task_container(selected_path: Path) -> Path:
    selected = selected_path.expanduser().resolve()
    return selected.parent / TASK_CONTAINER_NAME


def ensure_task_workspace(
    *,
    source_record: Mapping[str, Any],
    selected_path: Path,
    task_container: Optional[Path] = None,
    created_at: Optional[str] = None,
) -> TaskWorkspace:
    """Create or reuse the one task matching a source content fingerprint."""

    source = dict(source_record)
    fingerprint = str(source.get("fingerprint") or "").strip()
    if not fingerprint:
        raise WorkspaceError("输入记录缺少内容指纹")
    if source.get("read_only") is not True:
        raise WorkspaceError("任务输入必须声明为只读")
    task_id = f"{_task_slug(_source_label(source))}-{fingerprint[:12]}"
    container = (
        task_container.expanduser().resolve()
        if task_container is not None
        else default_task_container(selected_path)
    )
    root = container / task_id
    manifest_path = root / TASK_MANIFEST_NAME

    if root.exists():
        existing, detail = _load_signed_payload(manifest_path, TASK_MANIFEST_SCHEMA)
        if existing is None:
            raise WorkspaceError(
                f"任务目录已存在但清单不可验证，未覆盖：{root} ({detail})"
            )
        existing_source = existing.get("source")
        existing_fingerprint = (
            str(existing_source.get("fingerprint") or "")
            if isinstance(existing_source, Mapping)
            else ""
        )
        if existing.get("task_id") != task_id or existing_fingerprint != fingerprint:
            raise WorkspaceError(f"任务目录与当前输入指纹不匹配，未覆盖：{root}")
        for directory in (root / "runs", root / "checkpoints", root / "results"):
            directory.mkdir(parents=True, exist_ok=True)
        return TaskWorkspace(
            task_id=task_id,
            root=root,
            manifest_path=manifest_path,
            source_fingerprint=fingerprint,
            reused=True,
        )

    root.mkdir(parents=True, exist_ok=False)
    for directory in (root / "runs", root / "checkpoints", root / "results"):
        directory.mkdir()
    payload = _signed_payload(
        {
            "schema": TASK_MANIFEST_SCHEMA,
            "task_id": task_id,
            "created_at": str(created_at or run_manifest.utc_timestamp()),
            "pipeline_contract": {
                "schema": PIPELINE_CONTRACT_SCHEMA,
                "version": PIPELINE_CONTRACT_VERSION,
            },
            "source": source,
            "layout": {
                "runs": "runs",
                "checkpoints": "checkpoints",
                "results": "results",
            },
        }
    )
    # The source is never moved or rewritten.  A partial task without this
    # signed manifest is deliberately not reusable.
    run_manifest.atomic_write_json(manifest_path, payload)
    return TaskWorkspace(
        task_id=task_id,
        root=root,
        manifest_path=manifest_path,
        source_fingerprint=fingerprint,
        reused=False,
    )


def open_task_workspace(task_root: Path) -> tuple[TaskWorkspace, Dict[str, Any]]:
    """Open a current product task without mutating or guessing its state."""

    root = task_root.expanduser().resolve()
    payload, detail = _load_signed_payload(root / TASK_MANIFEST_NAME, TASK_MANIFEST_SCHEMA)
    if payload is None:
        raise WorkspaceError(detail)
    contract = payload.get("pipeline_contract")
    if not isinstance(contract, Mapping) or str(
        contract.get("schema") or ""
    ) != PIPELINE_CONTRACT_SCHEMA or str(
        contract.get("version") or ""
    ) != PIPELINE_CONTRACT_VERSION:
        raise WorkspaceError("任务 pipeline 契约不兼容")
    source = payload.get("source")
    if not isinstance(source, Mapping):
        raise WorkspaceError("任务缺少来源记录")
    fingerprint = str(source.get("fingerprint") or "").strip()
    task_id = str(payload.get("task_id") or "").strip()
    if not fingerprint or not task_id:
        raise WorkspaceError("任务缺少 ID 或来源指纹")
    return (
        TaskWorkspace(
            task_id=task_id,
            root=root,
            manifest_path=root / TASK_MANIFEST_NAME,
            source_fingerprint=fingerprint,
            reused=True,
        ),
        dict(source),
    )


def begin_task_run(
    *,
    workspace: TaskWorkspace,
    source_record: Mapping[str, Any],
    run_id: str,
    resume_record: Optional[Mapping[str, Any]] = None,
    checkpoint_fingerprints: Optional[Mapping[str, Mapping[str, Any]]] = None,
    processing_parameters: Optional[Mapping[str, Any]] = None,
    generated_at: Optional[str] = None,
) -> TaskRun:
    """Create one immutable-input run directory below an existing task."""

    normalized_run_id = str(run_id or "").strip()
    if not normalized_run_id or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}",
        normalized_run_id,
    ):
        raise WorkspaceError("run_id 只能包含字母、数字、点、下划线和连字符")
    source = dict(source_record)
    if str(source.get("fingerprint") or "") != workspace.source_fingerprint:
        raise WorkspaceError("运行输入与任务来源指纹不一致")
    if source.get("read_only") is not True:
        raise WorkspaceError("运行输入必须为只读来源")
    normalized_fingerprints: Dict[str, Dict[str, Any]] = {}
    allowed_fingerprint_keys = {
        f"stage{stage_number}" for stage_number in FORMAL_RESUME_STAGES
    }
    raw_fingerprints = checkpoint_fingerprints or {}
    if not isinstance(raw_fingerprints, Mapping):
        raise WorkspaceError("断点配置指纹必须为映射")
    unexpected_fingerprints = set(raw_fingerprints) - allowed_fingerprint_keys
    if unexpected_fingerprints:
        raise WorkspaceError("断点配置指纹包含非正式阶段")
    for key, value in raw_fingerprints.items():
        if not isinstance(value, Mapping):
            raise WorkspaceError(f"{key} 断点配置指纹无效")
        record = dict(value)
        stage_number = int(str(key).removeprefix("stage"))
        contract = stage_contract(stage_number)
        if record.get("stage") != stage_number or str(
            record.get("artifact") or ""
        ) != contract.primary_artifact or str(
            record.get("contract_version") or ""
        ) != PIPELINE_CONTRACT_VERSION or not str(
            record.get("fingerprint") or ""
        ).strip():
            raise WorkspaceError(f"{key} 断点配置指纹不符合阶段契约")
        normalized_fingerprints[str(key)] = record

    run_root = workspace.runs_dir / normalized_run_id
    if run_root.exists():
        raise WorkspaceError(f"运行目录已存在，未覆盖：{run_root}")
    normalized_resume: Optional[Dict[str, Any]] = None
    if resume_record is not None:
        normalized_resume = dict(resume_record)
        try:
            resume_stage = int(normalized_resume.get("stage"))
        except (TypeError, ValueError) as error:
            raise WorkspaceError("续跑记录缺少有效 stage") from error
        if resume_stage not in FORMAL_RESUME_STAGES:
            raise WorkspaceError("续跑记录不是 Stage 1、2、5 正式断点")
        contract = stage_contract(resume_stage)
        if str(normalized_resume.get("artifact") or "") != contract.primary_artifact:
            raise WorkspaceError("续跑记录的规范产物名不匹配")
        checkpoint_path = Path(
            str(normalized_resume.get("path") or "")
        ).expanduser().resolve()
        try:
            checkpoint_path.relative_to(workspace.checkpoints_dir.resolve())
        except ValueError as error:
            raise WorkspaceError("续跑文件不在任务 checkpoints 目录") from error
        if checkpoint_path.name != contract.primary_artifact:
            raise WorkspaceError("续跑文件名不符合阶段契约")
        actual_sha256 = run_manifest.sha256_file(checkpoint_path)
        expected_sha256 = str(normalized_resume.get("sha256") or "")
        if not actual_sha256 or not expected_sha256 or actual_sha256 != expected_sha256:
            raise WorkspaceError("续跑文件 SHA-256 不匹配")
        if str(normalized_resume.get("state") or "").lower() != "linear":
            raise WorkspaceError("续跑文件不是已验证线性状态")
        if not str(normalized_resume.get("run_manifest_hash") or "").strip() or not str(
            normalized_resume.get("config_fingerprint") or ""
        ).strip():
            raise WorkspaceError("续跑记录缺少运行清单或配置指纹")
        normalized_resume["semantic_context"] = _normalize_resume_semantic_context(
            normalized_resume.get("semantic_context"),
            stage_number=resume_stage,
        )
        if resume_stage in {2, 5} and normalized_resume["semantic_context"] is None:
            raise WorkspaceError(f"Stage {resume_stage} 续跑记录缺少语义契约")
        normalized_resume["semantic_context_status"] = (
            "verified"
            if normalized_resume["semantic_context"] is not None
            else "not_applicable"
        )
        normalized_auxiliary_artifacts = _normalize_checkpoint_auxiliary_artifacts(
            normalized_resume.get("auxiliary_artifacts"),
            task_root=workspace.root,
            stage_number=resume_stage,
        )
        if normalized_auxiliary_artifacts:
            normalized_resume["auxiliary_artifacts"] = (
                normalized_auxiliary_artifacts
            )
        else:
            normalized_resume.pop("auxiliary_artifacts", None)
        normalized_resume["path"] = str(checkpoint_path)

    try:
        normalized_processing_parameters, processing_parameter_adjustments = (
            normalize_processing_parameters(
                processing_parameters or default_processing_parameters(),
                validate_paths=True,
            )
        )
    except (TypeError, ValueError) as error:
        raise WorkspaceError(f"处理参数无效：{error}") from error
    processing_gate_profile = processing_gate_profile_audit(
        normalized_processing_parameters
    )

    run_root.mkdir(parents=True, exist_ok=False)
    manifest_path = run_root / RUN_MANIFEST_NAME
    payload = _signed_payload(
        {
            "schema": RUN_MANIFEST_SCHEMA,
            "run_id": normalized_run_id,
            "generated_at": str(generated_at or run_manifest.utc_timestamp()),
            "task_id": workspace.task_id,
            "task_directory": str(workspace.root),
            "source_fingerprint": workspace.source_fingerprint,
            "pipeline_contract": {
                "schema": PIPELINE_CONTRACT_SCHEMA,
                "version": PIPELINE_CONTRACT_VERSION,
            },
            "source": source,
            "resume": normalized_resume,
            "checkpoint_fingerprints": normalized_fingerprints,
            "processing_parameters": normalized_processing_parameters,
            "processing_gate_profile": processing_gate_profile,
            "processing_parameter_adjustments": processing_parameter_adjustments,
        }
    )
    run_manifest.atomic_write_json(manifest_path, payload)
    return TaskRun(
        run_id=normalized_run_id,
        root=run_root,
        manifest_path=manifest_path,
        task=workspace,
    )


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        shutil.copy2(source, temp_path)
        with temp_path.open("rb") as handle:
            os.fsync(handle.fileno())
        temp_path.replace(destination)
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def publish_formal_checkpoint(
    *,
    run_manifest_path: Path,
    stage_number: int,
    artifact_path: Path,
    semantic_context: Optional[Mapping[str, Any]] = None,
    auxiliary_artifacts: Optional[Mapping[str, Path]] = None,
    completed_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Atomically promote one accepted linear stage to task-level retention."""

    if stage_number not in FORMAL_RESUME_STAGES:
        raise WorkspaceError("只能发布 Stage 1、2、5 正式断点")
    run_payload, detail = _load_signed_payload(
        run_manifest_path.expanduser().resolve(),
        RUN_MANIFEST_SCHEMA,
    )
    if run_payload is None:
        raise WorkspaceError(detail)
    run_root = run_manifest_path.expanduser().resolve().parent
    artifact = artifact_path.expanduser().resolve()
    contract = stage_contract(stage_number)
    if artifact.parent != (run_root / "process").resolve() or (
        artifact.name != contract.primary_artifact
    ):
        raise WorkspaceError("阶段产物不在当前 run/process 或名称不符合契约")
    artifact_sha256 = run_manifest.sha256_file(artifact)
    if not artifact_sha256:
        raise WorkspaceError(f"无法读取阶段产物：{artifact}")
    source_auxiliaries: Dict[str, Dict[str, Any]] = {}
    if auxiliary_artifacts is not None:
        if stage_number != 5 or set(auxiliary_artifacts) != set(
            _STAGE5_SCENE_SUPPORT_AUXILIARIES
        ):
            raise WorkspaceError("Stage 5 场景支持辅助产物集合无效")
        loaded_support = scene_support.load_scene_support(artifact.parent)
        if loaded_support.get("status") not in {"available", "partial"}:
            raise WorkspaceError("Stage 5 场景支持辅助产物无法验证")
        for name in sorted(_STAGE5_SCENE_SUPPORT_AUXILIARIES):
            source_path = Path(auxiliary_artifacts[name]).expanduser().resolve()
            if source_path.parent != artifact.parent or source_path.name != name:
                raise WorkspaceError(f"Stage 5 辅助产物 {name} 来源路径无效")
            source_hash = run_manifest.sha256_file(source_path)
            if not source_hash:
                raise WorkspaceError(f"Stage 5 辅助产物 {name} 无法读取")
            source_auxiliaries[name] = {
                "source": source_path,
                "size": _safe_size(source_path),
                "sha256": source_hash,
            }
    normalized_semantic_context = _normalize_resume_semantic_context(
        semantic_context,
        stage_number=stage_number,
    )
    if stage_number in {2, 5} and normalized_semantic_context is None:
        raise WorkspaceError(f"Stage {stage_number} 正式断点必须包含语义契约")

    task_root = Path(str(run_payload.get("task_directory") or "")).resolve()
    workspace, source = open_task_workspace(task_root)
    source_fingerprint = str(source.get("fingerprint") or "")
    if (
        run_payload.get("task_id") != workspace.task_id
        or str(run_payload.get("source_fingerprint") or "") != source_fingerprint
    ):
        raise WorkspaceError("运行清单与任务来源不匹配")
    fingerprints = run_payload.get("checkpoint_fingerprints")
    key = f"stage{stage_number}"
    fingerprint_record = (
        fingerprints.get(key) if isinstance(fingerprints, Mapping) else None
    )
    if not isinstance(fingerprint_record, Mapping) or str(
        fingerprint_record.get("artifact") or ""
    ) != contract.primary_artifact or not str(
        fingerprint_record.get("fingerprint") or ""
    ).strip():
        raise WorkspaceError(f"运行清单缺少 {key} 累计配置指纹")

    manifest_path = workspace.root / CHECKPOINT_MANIFEST_REL
    if manifest_path.exists():
        existing, detail = _load_signed_payload(
            manifest_path,
            CHECKPOINT_MANIFEST_SCHEMA,
        )
        if existing is None:
            raise WorkspaceError(f"现有断点清单不可验证，未覆盖：{detail}")
        if existing.get("task_id") != workspace.task_id or str(
            existing.get("source_fingerprint") or ""
        ) != source_fingerprint or existing.get(
            "pipeline_contract"
        ) != pipeline_contract_manifest():
            raise WorkspaceError("现有断点清单与任务或契约不匹配")
        records = dict(existing.get("checkpoints") or {})
    else:
        records = {}

    destination = workspace.checkpoints_dir / contract.primary_artifact
    destination_existed = destination.is_file()
    backup_path: Optional[Path] = None
    if destination_existed:
        fd, backup_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".bak",
            dir=str(destination.parent),
        )
        os.close(fd)
        backup_path = Path(backup_name)
        backup_path.unlink(missing_ok=True)
        try:
            os.link(destination, backup_path)
        except OSError:
            shutil.copy2(destination, backup_path)

    auxiliary_destinations: Dict[str, Path] = {}
    auxiliary_backups: Dict[str, Optional[Path]] = {}
    auxiliary_existed: Dict[str, bool] = {}
    for name in sorted(source_auxiliaries):
        auxiliary_destination = workspace.checkpoints_dir / name
        auxiliary_destinations[name] = auxiliary_destination
        existed = auxiliary_destination.is_file()
        auxiliary_existed[name] = existed
        auxiliary_backup: Optional[Path] = None
        if existed:
            fd, backup_name = tempfile.mkstemp(
                prefix=f".{name}.",
                suffix=".bak",
                dir=str(auxiliary_destination.parent),
            )
            os.close(fd)
            auxiliary_backup = Path(backup_name)
            auxiliary_backup.unlink(missing_ok=True)
            try:
                os.link(auxiliary_destination, auxiliary_backup)
            except OSError:
                shutil.copy2(auxiliary_destination, auxiliary_backup)
        auxiliary_backups[name] = auxiliary_backup

    committed = False
    try:
        _atomic_copy(artifact, destination)
        copied_sha256 = run_manifest.sha256_file(destination)
        if copied_sha256 != artifact_sha256:
            raise WorkspaceError(f"{key} 断点复制后 SHA-256 不一致")
        copied_auxiliaries: Dict[str, Dict[str, Any]] = {}
        for name, source_record in source_auxiliaries.items():
            auxiliary_destination = auxiliary_destinations[name]
            _atomic_copy(source_record["source"], auxiliary_destination)
            copied_hash = run_manifest.sha256_file(auxiliary_destination)
            if copied_hash != source_record["sha256"]:
                raise WorkspaceError(f"Stage 5 辅助产物 {name} 复制后 SHA-256 不一致")
            copied_auxiliaries[name] = {
                "path": auxiliary_destination.relative_to(workspace.root).as_posix(),
                "size": _safe_size(auxiliary_destination),
                "sha256": copied_hash,
            }
        if copied_auxiliaries:
            loaded_checkpoint_support = scene_support.load_scene_support(
                workspace.checkpoints_dir
            )
            if loaded_checkpoint_support.get("status") not in {
                "available",
                "partial",
            }:
                raise WorkspaceError("Stage 5 辅助产物复制后内容校验失败")

        for downstream_stage in FORMAL_RESUME_STAGES:
            downstream_key = f"stage{downstream_stage}"
            if downstream_stage <= stage_number or downstream_key not in records:
                continue
            current_fingerprint = (
                fingerprints.get(downstream_key)
                if isinstance(fingerprints, Mapping)
                else None
            )
            saved_fingerprint = str(
                (records.get(downstream_key) or {}).get("config_fingerprint") or ""
            )
            expected_fingerprint = (
                str(current_fingerprint.get("fingerprint") or "")
                if isinstance(current_fingerprint, Mapping)
                else ""
            )
            if not expected_fingerprint or saved_fingerprint != expected_fingerprint:
                records.pop(downstream_key, None)

        records[key] = {
            "stage": stage_number,
            "artifact": contract.primary_artifact,
            "path": destination.relative_to(workspace.root).as_posix(),
            "size": _safe_size(destination),
            "sha256": copied_sha256,
            "state": "linear",
            "run_manifest_hash": str(run_payload.get("manifest_hash") or ""),
            "config_fingerprint": str(fingerprint_record["fingerprint"]),
            "run_id": run_payload.get("run_id"),
            "completed_at": str(completed_at or run_manifest.utc_timestamp()),
            "semantic_context": normalized_semantic_context,
            "semantic_context_status": (
                "verified"
                if normalized_semantic_context is not None
                else "not_applicable"
            ),
            **(
                {"auxiliary_artifacts": copied_auxiliaries}
                if copied_auxiliaries
                else {}
            ),
        }
        checkpoint_manifest = build_checkpoint_manifest(
            task_id=workspace.task_id,
            source_fingerprint=source_fingerprint,
            checkpoints=records,
            generated_at=completed_at,
        )
        run_manifest.atomic_write_json(manifest_path, checkpoint_manifest)
        committed = True
    finally:
        if not committed:
            if backup_path is not None and backup_path.is_file():
                backup_path.replace(destination)
            elif not destination_existed:
                destination.unlink(missing_ok=True)
            for name, auxiliary_destination in auxiliary_destinations.items():
                auxiliary_backup = auxiliary_backups.get(name)
                if auxiliary_backup is not None and auxiliary_backup.is_file():
                    auxiliary_backup.replace(auxiliary_destination)
                elif not auxiliary_existed.get(name, False):
                    auxiliary_destination.unlink(missing_ok=True)
        if backup_path is not None:
            backup_path.unlink(missing_ok=True)
        for auxiliary_backup in auxiliary_backups.values():
            if auxiliary_backup is not None:
                auxiliary_backup.unlink(missing_ok=True)
    return dict(records[key])


def publish_latest_result_index(*, run_manifest_path: Path) -> Dict[str, Any]:
    """Point the task at the latest hash-verified delivery files in run history."""

    run_payload, detail = _load_signed_payload(
        run_manifest_path.expanduser().resolve(),
        RUN_MANIFEST_SCHEMA,
    )
    if run_payload is None:
        raise WorkspaceError(detail)
    run_root = run_manifest_path.expanduser().resolve().parent
    task_root = Path(str(run_payload.get("task_directory") or "")).resolve()
    workspace, source = open_task_workspace(task_root)
    if run_payload.get("task_id") != workspace.task_id or str(
        run_payload.get("source_fingerprint") or ""
    ) != str(source.get("fingerprint") or ""):
        raise WorkspaceError("运行清单与任务来源不匹配")

    result_path = run_root / "pipeline-result.json"
    result = run_manifest.load_json(result_path)
    if result is None or str(result.get("schema") or "") not in (
        outcome.SUPPORTED_PIPELINE_RESULT_SCHEMAS
    ):
        raise WorkspaceError("pipeline-result.json 缺失或 schema 不受支持")
    expected_hash = str(result.get("manifest_hash") or "")
    unsigned = dict(result)
    unsigned.pop("manifest_hash", None)
    if not expected_hash or expected_hash != run_manifest.canonical_payload_hash(
        unsigned
    ):
        raise WorkspaceError("pipeline-result.json 哈希无效")
    try:
        result = outcome.normalize_pipeline_result(result)
    except ValueError as error:
        raise WorkspaceError(f"pipeline-result.json 无法归一化：{error}") from error
    plan = run_manifest.load_json(run_root / "processing-plan.json")
    plan_verification = task_plan.verify_processing_plan(plan or {})
    if not plan_verification.get("verified"):
        raise WorkspaceError(
            "processing-plan.json 校验失败："
            + str(plan_verification.get("detail") or "unknown error")
        )
    plan_hash = str(plan.get("plan_hash") or "") if isinstance(plan, Mapping) else ""
    if str(result.get("plan_hash") or "") != plan_hash:
        raise WorkspaceError("pipeline-result.json 引用的处理计划哈希不匹配")
    if str(result.get("run_id") or "") != str(run_payload.get("run_id") or "") or str(
        plan.get("run_id") or ""
    ) != str(run_payload.get("run_id") or ""):
        raise WorkspaceError("处理计划、运行清单与结果的 run_id 不一致")
    status = str(result.get("status") or "")
    if status not in {"success", "partial_success", "review_required"}:
        raise WorkspaceError(f"运行状态 {status or 'unknown'} 不能发布为最新结果")
    outputs = result.get("outputs")
    if not isinstance(outputs, Mapping):
        raise WorkspaceError("pipeline-result.json 没有输出记录")

    verified_outputs: Dict[str, Dict[str, Any]] = {}
    for name, raw_record in outputs.items():
        if not isinstance(raw_record, Mapping):
            continue
        relative_path = Path(str(raw_record.get("path") or name))
        if relative_path.is_absolute():
            continue
        candidate = (run_root / relative_path).resolve()
        try:
            candidate.relative_to(run_root)
        except ValueError:
            continue
        expected_sha256 = str(raw_record.get("sha256") or "")
        actual_sha256 = run_manifest.sha256_file(candidate)
        if not expected_sha256 or not actual_sha256 or actual_sha256 != expected_sha256:
            continue
        verified_outputs[str(name)] = {
            "path": str(candidate),
            "size": _safe_size(candidate),
            "sha256": actual_sha256,
        }
    if not verified_outputs:
        raise WorkspaceError("本轮没有通过 SHA-256 校验的交付文件")

    payload = _signed_payload(
        {
            "schema": "starun.latest-result.v1",
            "generated_at": run_manifest.utc_timestamp(),
            "task_id": workspace.task_id,
            "run_id": run_payload.get("run_id"),
            "run_directory": str(run_root),
            "status": status,
            "outputs": verified_outputs,
        }
    )
    run_manifest.atomic_write_json(
        workspace.root / LATEST_RESULT_MANIFEST_REL,
        payload,
    )
    return payload


def latest_result_files(
    task_root: Path,
    *,
    suffixes: Optional[Iterable[str]] = None,
) -> tuple[Path, ...]:
    """Return still-valid files referenced by the task's latest result index."""

    payload, _detail = _load_signed_payload(
        task_root.expanduser().resolve() / LATEST_RESULT_MANIFEST_REL,
        "starun.latest-result.v1",
    )
    if payload is None:
        return ()
    outputs = payload.get("outputs")
    if not isinstance(outputs, Mapping):
        return ()
    normalized_suffixes = (
        {str(suffix).lower() for suffix in suffixes}
        if suffixes is not None
        else None
    )
    files = []
    for record in outputs.values():
        if not isinstance(record, Mapping):
            continue
        path = Path(str(record.get("path") or "")).expanduser().resolve()
        if (
            normalized_suffixes is not None
            and path.suffix.lower() not in normalized_suffixes
        ):
            continue
        expected_sha256 = str(record.get("sha256") or "")
        if expected_sha256 and run_manifest.sha256_file(path) == expected_sha256:
            files.append(path)
    return tuple(sorted(files, key=lambda path: path.name.lower()))


def latest_result_directory(task_root: Path) -> Optional[Path]:
    """Return the verified run directory behind a task's latest-result index."""

    root = task_root.expanduser().resolve()
    payload, _detail = _load_signed_payload(
        root / LATEST_RESULT_MANIFEST_REL,
        "starun.latest-result.v1",
    )
    if payload is None:
        return None
    run_directory = Path(str(payload.get("run_directory") or "")).expanduser().resolve()
    try:
        relative_run = run_directory.relative_to((root / "runs").resolve())
    except ValueError:
        return None
    if len(relative_run.parts) != 1 or not run_directory.is_dir():
        return None
    result_files = latest_result_files(root)
    if not result_files:
        return None
    for path in result_files:
        try:
            path.relative_to(run_directory)
        except ValueError:
            return None
    return run_directory


def _run_requested_intermediate_retention(run_root: Path) -> bool:
    """Return whether verified settings require a full process directory."""
    plan = run_manifest.load_json(run_root / "processing-plan.json")
    if not isinstance(plan, Mapping) or not task_plan.verify_processing_plan(
        plan
    ).get("verified"):
        return False
    metadata = plan.get("metadata")
    config = metadata.get("config") if isinstance(metadata, Mapping) else None
    return bool(
        isinstance(config, Mapping)
        and config.get("debug_mode") is True
        and config.get("checkpoint_mode") is not True
    )


def apply_task_retention(
    task_root: Path,
) -> Dict[str, Any]:
    """Keep the latest delivery and compact old runs by each frozen Debug flag.
    """

    root = task_root.expanduser().resolve()
    workspace, source = open_task_workspace(root)
    latest_run = latest_result_directory(root)
    if latest_run is None:
        raise WorkspaceError("最新结果索引不可验证，未执行保留策略")

    deleted_files: list[str] = []
    removed_process_dirs: list[str] = []
    preserved_debug_process_dirs: list[str] = []
    skipped: list[str] = []
    preserved_output_names = {
        "processing-plan.json",
        "pipeline-result.json",
        "run-manifest.json",
        "starun_diagnostics.zip",
    }
    for run_root in sorted(workspace.runs_dir.iterdir(), key=lambda path: path.name):
        if not run_root.is_dir() or run_root.is_symlink():
            continue
        run_payload, detail = _load_signed_payload(
            run_root / RUN_MANIFEST_NAME,
            RUN_MANIFEST_SCHEMA,
        )
        if run_payload is None:
            skipped.append(f"{run_root.name}: {detail}")
            continue
        if run_payload.get("task_id") != workspace.task_id or str(
            run_payload.get("source_fingerprint") or ""
        ) != str(source.get("fingerprint") or ""):
            skipped.append(f"{run_root.name}: run manifest does not belong to task")
            continue

        is_latest_run = run_root.resolve() == latest_run.resolve()
        process_dir = run_root / "process"
        run_preserves_intermediates = _run_requested_intermediate_retention(
            run_root
        )
        if (
            not is_latest_run
            and not run_preserves_intermediates
            and process_dir.is_dir()
            and not process_dir.is_symlink()
        ):
            try:
                shutil.rmtree(process_dir)
                removed_process_dirs.append(
                    process_dir.relative_to(root).as_posix()
                )
            except OSError as error:
                skipped.append(f"{run_root.name}/process: {error}")
        elif (
            not is_latest_run
            and run_preserves_intermediates
            and process_dir.is_dir()
            and not process_dir.is_symlink()
        ):
            preserved_debug_process_dirs.append(
                process_dir.relative_to(root).as_posix()
            )

        if is_latest_run:
            continue
        result_path = run_root / "pipeline-result.json"
        result = run_manifest.load_json(result_path)
        if result is None or str(result.get("schema") or "") not in (
            outcome.SUPPORTED_PIPELINE_RESULT_SCHEMAS
        ):
            skipped.append(f"{run_root.name}: pipeline result is unavailable")
            continue
        claimed_hash = str(result.get("manifest_hash") or "")
        unsigned_result = dict(result)
        unsigned_result.pop("manifest_hash", None)
        if not claimed_hash or claimed_hash != run_manifest.canonical_payload_hash(
            unsigned_result
        ):
            skipped.append(f"{run_root.name}: pipeline result hash is invalid")
            continue
        outputs = result.get("outputs")
        if not isinstance(outputs, Mapping):
            continue
        for raw_record in outputs.values():
            if not isinstance(raw_record, Mapping):
                continue
            relative_path = Path(str(raw_record.get("path") or ""))
            if relative_path.is_absolute() or not relative_path.as_posix():
                continue
            candidate = (run_root / relative_path).resolve()
            try:
                candidate.relative_to(run_root.resolve())
            except ValueError:
                continue
            if candidate.name.lower() in preserved_output_names:
                continue
            expected_sha256 = str(raw_record.get("sha256") or "")
            if (
                not expected_sha256
                or run_manifest.sha256_file(candidate) != expected_sha256
            ):
                continue
            try:
                candidate.unlink()
                deleted_files.append(candidate.relative_to(root).as_posix())
            except OSError as error:
                skipped.append(f"{candidate.relative_to(root)}: {error}")

    payload = _signed_payload(
        {
            "schema": "starun.retention-report.v1",
            "generated_at": run_manifest.utc_timestamp(),
            "task_id": workspace.task_id,
            "latest_run_id": latest_run.name,
            "retention_scope": "per_run_frozen_debug_and_checkpoint_settings",
            "deleted_files": deleted_files,
            "removed_process_directories": removed_process_dirs,
            "preserved_debug_process_directories": preserved_debug_process_dirs,
            "skipped": skipped,
        }
    )
    run_manifest.atomic_write_json(root / RETENTION_MANIFEST_REL, payload)
    return payload


def build_checkpoint_manifest(
    *,
    task_id: str,
    source_fingerprint: str,
    checkpoints: Mapping[str, Mapping[str, Any]],
    generated_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a signed index; checkpoint files themselves remain hash-verified."""

    allowed = {f"stage{stage}" for stage in FORMAL_RESUME_STAGES}
    unexpected = set(checkpoints) - allowed
    if unexpected:
        raise WorkspaceError(
            "断点清单包含非正式续跑阶段："
            + ", ".join(sorted(str(key) for key in unexpected))
        )
    return _signed_payload(
        {
            "schema": CHECKPOINT_MANIFEST_SCHEMA,
            "generated_at": str(generated_at or run_manifest.utc_timestamp()),
            "task_id": str(task_id),
            "source_fingerprint": str(source_fingerprint),
            "pipeline_contract": pipeline_contract_manifest(),
            "checkpoints": {
                str(key): dict(value) for key, value in checkpoints.items()
            },
        }
    )


def _resolve_checkpoint_path(task_root: Path, relative_path: Any) -> Optional[Path]:
    raw_path = Path(str(relative_path or ""))
    if not raw_path.as_posix() or raw_path.is_absolute():
        return None
    root = task_root.resolve()
    candidate = (root / raw_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


_STAGE5_SCENE_SUPPORT_AUXILIARIES = frozenset(
    {scene_support.SCENE_SUPPORT_JSON, scene_support.SCENE_SUPPORT_ARRAYS}
)


def _normalize_checkpoint_auxiliary_artifacts(
    value: Any,
    *,
    task_root: Path,
    stage_number: int,
) -> Dict[str, Dict[str, Any]]:
    if value is None:
        return {}
    if stage_number != 5 or not isinstance(value, Mapping):
        raise WorkspaceError("断点辅助产物契约无效")
    if set(value) != set(_STAGE5_SCENE_SUPPORT_AUXILIARIES):
        raise WorkspaceError("Stage 5 场景支持辅助产物必须成对存在")
    normalized: Dict[str, Dict[str, Any]] = {}
    checkpoint_root = (task_root / "checkpoints").resolve()
    for name in sorted(_STAGE5_SCENE_SUPPORT_AUXILIARIES):
        record = value.get(name)
        if not isinstance(record, Mapping):
            raise WorkspaceError(f"Stage 5 辅助产物 {name} 记录无效")
        path = _resolve_checkpoint_path(task_root, record.get("path"))
        if path is None or path.parent != checkpoint_root or path.name != name:
            raise WorkspaceError(f"Stage 5 辅助产物 {name} 路径无效")
        expected_hash = str(record.get("sha256") or "")
        actual_hash = run_manifest.sha256_file(path)
        if not expected_hash or actual_hash != expected_hash:
            raise WorkspaceError(f"Stage 5 辅助产物 {name} SHA-256 不匹配")
        try:
            size = int(record.get("size"))
        except (TypeError, ValueError) as error:
            raise WorkspaceError(f"Stage 5 辅助产物 {name} 大小无效") from error
        if size != _safe_size(path):
            raise WorkspaceError(f"Stage 5 辅助产物 {name} 大小不匹配")
        normalized[name] = {
            "path": path.relative_to(task_root.resolve()).as_posix(),
            "size": size,
            "sha256": expected_hash,
        }
    loaded = scene_support.load_scene_support(checkpoint_root)
    if loaded.get("status") not in {"available", "partial"}:
        raise WorkspaceError("Stage 5 场景支持辅助产物内容校验失败")
    return normalized


def _checkpoint_compatible(
    *,
    task_root: Path,
    stage_number: int,
    record: Mapping[str, Any],
    current_resume_fingerprints: Optional[Mapping[str, Mapping[str, Any]]],
) -> tuple[bool, str, Optional[Path]]:
    key = f"stage{stage_number}"
    contract = stage_contract(stage_number)
    if record.get("stage") != stage_number:
        return False, f"{key} stage number mismatch", None
    if str(record.get("artifact") or "") != contract.primary_artifact:
        return False, f"{key} artifact contract mismatch", None
    if str(record.get("state") or "").lower() != "linear":
        return False, f"{key} state is not linear", None
    if not str(record.get("run_manifest_hash") or "").strip():
        return False, f"{key} run manifest hash is missing", None
    try:
        semantic_context = _normalize_resume_semantic_context(
            record.get("semantic_context"),
            stage_number=stage_number,
        )
    except WorkspaceError as error:
        return False, f"{key} semantic context is invalid: {error}", None
    if stage_number in {2, 5} and semantic_context is None:
        return False, f"{key} semantic context is missing", None
    config_fingerprint = str(record.get("config_fingerprint") or "").strip()
    if not config_fingerprint:
        return False, f"{key} config fingerprint is missing", None
    if current_resume_fingerprints is not None:
        current = current_resume_fingerprints.get(key)
        current_fingerprint = (
            str(current.get("fingerprint") or "")
            if isinstance(current, Mapping)
            else ""
        )
        if not current_fingerprint or current_fingerprint != config_fingerprint:
            return False, f"{key} configuration is incompatible", None
    path = _resolve_checkpoint_path(task_root, record.get("path"))
    if path is None or path.name != contract.primary_artifact:
        return False, f"{key} path is outside task or has wrong name", None
    expected_hash = str(record.get("sha256") or "")
    actual_hash = run_manifest.sha256_file(path)
    if not expected_hash or not actual_hash or expected_hash != actual_hash:
        return False, f"{key} SHA-256 mismatch", path
    try:
        _normalize_checkpoint_auxiliary_artifacts(
            record.get("auxiliary_artifacts"),
            task_root=task_root,
            stage_number=stage_number,
        )
    except WorkspaceError as error:
        return False, f"{key} auxiliary artifacts are invalid: {error}", path
    return True, "verified", path


def inspect_task_workspace(
    task_root: Path,
    *,
    current_resume_fingerprints: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Find the latest compatible Stage 1/2/5 checkpoint, never by filename."""

    root = task_root.expanduser().resolve()
    result: Dict[str, Any] = {
        "recognized": False,
        "verified": False,
        "task_directory": str(root),
        "resume_after_stage": None,
    }
    task, detail = _load_signed_payload(root / TASK_MANIFEST_NAME, TASK_MANIFEST_SCHEMA)
    if task is None:
        result["detail"] = detail
        return result
    contract = task.get("pipeline_contract")
    if not isinstance(contract, Mapping) or str(
        contract.get("schema") or ""
    ) != PIPELINE_CONTRACT_SCHEMA or str(
        contract.get("version") or ""
    ) != PIPELINE_CONTRACT_VERSION:
        result["detail"] = "task pipeline contract is incompatible"
        return result
    result["recognized"] = True
    result["task_id"] = task.get("task_id")
    source = task.get("source")
    source_fingerprint = (
        str(source.get("fingerprint") or "")
        if isinstance(source, Mapping)
        else ""
    )
    if not source_fingerprint:
        result["detail"] = "task source fingerprint is missing"
        return result

    checkpoint_manifest, detail = _load_signed_payload(
        root / CHECKPOINT_MANIFEST_REL,
        CHECKPOINT_MANIFEST_SCHEMA,
    )
    if checkpoint_manifest is None:
        result["detail"] = detail
        return result
    if checkpoint_manifest.get("task_id") != task.get("task_id") or str(
        checkpoint_manifest.get("source_fingerprint") or ""
    ) != source_fingerprint:
        result["detail"] = "checkpoint manifest does not belong to this task input"
        return result
    if checkpoint_manifest.get("pipeline_contract") != pipeline_contract_manifest():
        result["detail"] = "checkpoint pipeline contract was modified or is incompatible"
        return result
    records = checkpoint_manifest.get("checkpoints")
    if not isinstance(records, Mapping):
        result["detail"] = "checkpoint manifest has no checkpoint records"
        return result
    unexpected = set(records) - {f"stage{stage}" for stage in FORMAL_RESUME_STAGES}
    if unexpected:
        result["detail"] = "checkpoint manifest contains non-formal stages"
        return result

    rejections: Dict[str, str] = {}
    for stage_number in reversed(FORMAL_RESUME_STAGES):
        key = f"stage{stage_number}"
        record = records.get(key)
        if not isinstance(record, Mapping):
            rejections[key] = "missing"
            continue
        compatible, checkpoint_detail, checkpoint_path = _checkpoint_compatible(
            task_root=root,
            stage_number=stage_number,
            record=record,
            current_resume_fingerprints=current_resume_fingerprints,
        )
        if not compatible:
            rejections[key] = checkpoint_detail
            continue
        result.update(
            {
                "verified": True,
                "resume_after_stage": stage_number,
                "checkpoint": key,
                "checkpoint_path": str(checkpoint_path),
                "run_manifest_hash": record.get("run_manifest_hash"),
                "config_fingerprint": record.get("config_fingerprint"),
                "resume_record": {
                    **dict(record),
                    "path": str(checkpoint_path),
                },
                "detail": f"latest compatible checkpoint is {key}",
                "rejections": rejections,
            }
        )
        return result
    result["detail"] = "no compatible formal checkpoint was verified"
    result["rejections"] = rejections
    return result


__all__ = [
    "CHECKPOINT_MANIFEST_REL",
    "CHECKPOINT_MANIFEST_SCHEMA",
    "RUN_MANIFEST_NAME",
    "RUN_MANIFEST_SCHEMA",
    "RESUME_SEMANTIC_SCHEMA",
    "RESUME_SEMANTIC_SCHEMA_V1",
    "RESUME_SEMANTIC_SCHEMA_V2",
    "SUPPORTED_RESUME_SEMANTIC_SCHEMAS",
    "LATEST_RESULT_MANIFEST_REL",
    "RETENTION_MANIFEST_REL",
    "TASK_CONTAINER_NAME",
    "TASK_MANIFEST_NAME",
    "TASK_MANIFEST_SCHEMA",
    "TaskWorkspace",
    "TaskRun",
    "WorkspaceError",
    "build_checkpoint_manifest",
    "build_source_record",
    "begin_task_run",
    "apply_task_retention",
    "default_task_container",
    "ensure_task_workspace",
    "inspect_task_workspace",
    "latest_result_directory",
    "latest_result_files",
    "open_task_workspace",
    "publish_formal_checkpoint",
    "publish_latest_result_index",
]
