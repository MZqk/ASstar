"""Persistent GUI processing history with fail-closed task deletion checks."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Collection, Mapping, Sequence

try:
    from .run_presentation import (
        VerifiedOutput,
        VerifiedRunBundle,
        formal_output_allowlist,
    )
except ImportError:  # Support direct execution from the gui directory.
    from run_presentation import (  # type: ignore
        VerifiedOutput,
        VerifiedRunBundle,
        formal_output_allowlist,
    )

try:
    from pipeline import outcome, run_manifest, task_plan
    from pipeline.task_workspace import (
        RUN_MANIFEST_NAME,
        RUN_MANIFEST_SCHEMA,
        TASK_CONTAINER_NAME,
        TaskWorkspace,
        WorkspaceError,
        open_task_workspace,
    )
except ImportError:  # Support direct execution from the gui directory.
    import sys

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from pipeline import outcome, run_manifest, task_plan  # type: ignore[no-redef]
    from pipeline.task_workspace import (  # type: ignore[no-redef]
        RUN_MANIFEST_NAME,
        RUN_MANIFEST_SCHEMA,
        TASK_CONTAINER_NAME,
        TaskWorkspace,
        WorkspaceError,
        open_task_workspace,
    )


HISTORY_SCHEMA = "starun.gui-history.v1"
HISTORY_FILENAME = "history-index.json"

STATUS_PREPARING = "preparing"
STATUS_RUNNING = "running"
STATUS_SUCCESS = "success"
STATUS_PARTIAL_SUCCESS = "partial_success"
STATUS_REVIEW_REQUIRED = "review_required"
STATUS_FAILED = "failed"
STATUS_STOPPED = "stopped"
STATUS_INTERRUPTED = "interrupted"

ACTIVE_STATUSES = frozenset({STATUS_PREPARING, STATUS_RUNNING})
TERMINAL_STATUSES = frozenset(
    {
        STATUS_SUCCESS,
        STATUS_PARTIAL_SUCCESS,
        STATUS_REVIEW_REQUIRED,
        STATUS_FAILED,
        STATUS_STOPPED,
        STATUS_INTERRUPTED,
    }
)
KNOWN_STATUSES = ACTIVE_STATUSES | TERMINAL_STATUSES

STATUS_LABELS = {
    STATUS_PREPARING: "正在准备",
    STATUS_RUNNING: "运行中",
    STATUS_SUCCESS: "成功",
    STATUS_PARTIAL_SUCCESS: "降级完成",
    STATUS_REVIEW_REQUIRED: "需要复核",
    STATUS_FAILED: "失败",
    STATUS_STOPPED: "已中止",
    STATUS_INTERRUPTED: "异常中断",
}

DELIVERABLE_IMAGE_SUFFIXES = frozenset(
    {
        ".fit",
        ".fits",
        ".fts",
        ".xisf",
        ".tif",
        ".tiff",
        ".png",
        ".jpg",
        ".jpeg",
    }
)


class HistoryStoreError(RuntimeError):
    """Raised when the history index or a referenced record is invalid."""


class UnsafeTaskDeletionError(HistoryStoreError):
    """Raised when a path is not one exact, verified product task root."""


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def default_history_path() -> Path:
    return (
        Path.home()
        / "Library"
        / "Application Support"
        / "Starun"
        / HISTORY_FILENAME
    )


def display_name_from_source(source: Mapping[str, Any]) -> str:
    group = source.get("group")
    if isinstance(group, Mapping):
        target = str(group.get("target") or "").strip()
        if target and target.lower() != "unknown":
            return target
    files = source.get("files")
    if isinstance(files, (list, tuple)) and files:
        first = files[0]
        if isinstance(first, Mapping):
            display_path = str(
                first.get("display_path") or first.get("path") or ""
            ).strip()
            if display_path:
                return Path(display_path).stem or Path(display_path).name
    selected_path = str(source.get("selected_path") or "").strip()
    if selected_path:
        return Path(selected_path).stem or Path(selected_path).name
    return "未命名任务"


def history_task_key(task_directory: Path) -> str:
    path = task_directory.expanduser().resolve()
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:20]
    return f"task-{digest}"


def _empty_payload() -> dict[str, Any]:
    return {
        "schema": HISTORY_SCHEMA,
        "updated_at": utc_timestamp(),
        "tasks": [],
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _verified_signed_payload(
    path: Path,
    schema: str | Collection[str],
) -> dict[str, Any]:
    payload = run_manifest.load_json(path)
    supported = {schema} if isinstance(schema, str) else set(schema)
    if payload is None or str(payload.get("schema") or "") not in supported:
        raise HistoryStoreError(f"清单缺失或 schema 不受支持：{path}")
    claimed_hash = str(payload.get("manifest_hash") or "")
    unsigned = dict(payload)
    unsigned.pop("manifest_hash", None)
    if not claimed_hash or claimed_hash != run_manifest.canonical_payload_hash(
        unsigned
    ):
        raise HistoryStoreError(f"清单哈希无效：{path}")
    return dict(payload)


def verify_history_run(
    task_directory: Path,
    run_id: str,
) -> tuple[TaskWorkspace, dict[str, Any], Path]:
    workspace, source = open_task_workspace(task_directory)
    run_root = (workspace.runs_dir / str(run_id)).resolve()
    try:
        relative = run_root.relative_to(workspace.runs_dir.resolve())
    except ValueError as error:
        raise HistoryStoreError("运行目录不在任务 runs 目录内") from error
    if len(relative.parts) != 1 or not run_root.is_dir() or run_root.is_symlink():
        raise HistoryStoreError("运行目录不存在或不是直接子目录")
    payload = _verified_signed_payload(
        run_root / RUN_MANIFEST_NAME,
        RUN_MANIFEST_SCHEMA,
    )
    if (
        str(payload.get("run_id") or "") != str(run_id)
        or str(payload.get("task_id") or "") != workspace.task_id
        or str(payload.get("source_fingerprint") or "")
        != str(source.get("fingerprint") or "")
        or Path(str(payload.get("task_directory") or "")).resolve()
        != workspace.root.resolve()
    ):
        raise HistoryStoreError("运行清单与任务身份不匹配")
    return workspace, payload, run_root


def load_verified_pipeline_result(run_root: Path) -> dict[str, Any] | None:
    path = run_root.resolve() / "pipeline-result.json"
    if not path.is_file():
        return None
    payload = _verified_signed_payload(
        path,
        outcome.SUPPORTED_PIPELINE_RESULT_SCHEMAS,
    )
    try:
        return outcome.normalize_pipeline_result(payload)
    except ValueError as error:
        raise HistoryStoreError(f"流水线结果无法归一化：{error}") from error


def load_normalized_run_state(run_root: Path) -> dict[str, Any] | None:
    """Load v1/v2 run state; v2 and signed v1 records fail closed on hash."""

    path = run_root.resolve() / "run-state.json"
    if not path.is_file():
        return None
    payload = run_manifest.load_json(path)
    if payload is None or str(payload.get("schema") or "") not in (
        outcome.SUPPORTED_RUN_STATE_SCHEMAS
    ):
        raise HistoryStoreError(f"运行状态缺失或 schema 不受支持：{path}")
    claimed_hash = str(payload.get("manifest_hash") or "")
    if claimed_hash:
        unsigned = dict(payload)
        unsigned.pop("manifest_hash", None)
        if claimed_hash != run_manifest.canonical_payload_hash(unsigned):
            raise HistoryStoreError(f"运行状态哈希无效：{path}")
    elif str(payload.get("schema") or "") == outcome.RUN_STATE_SCHEMA_V2:
        raise HistoryStoreError(f"v2 运行状态缺少哈希：{path}")
    try:
        normalized = outcome.normalize_run_state(payload)
    except ValueError as error:
        raise HistoryStoreError(f"运行状态无法归一化：{error}") from error
    normalized["integrity_status"] = (
        "verified" if claimed_hash else "legacy_unsigned"
    )
    return normalized


def verified_result_files(
    run_root: Path,
    result: Mapping[str, Any],
    *,
    suffixes: set[str] | None = None,
) -> tuple[Path, ...]:
    root = run_root.expanduser().resolve()
    outputs = result.get("outputs")
    if not isinstance(outputs, Mapping):
        return ()
    normalized_suffixes = (
        {suffix.lower() for suffix in suffixes} if suffixes is not None else None
    )
    verified: list[Path] = []
    for raw_record in outputs.values():
        if not isinstance(raw_record, Mapping):
            continue
        relative = Path(str(raw_record.get("path") or ""))
        if relative.is_absolute() or not relative.as_posix():
            continue
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        if (
            normalized_suffixes is not None
            and candidate.suffix.lower() not in normalized_suffixes
        ):
            continue
        expected_sha256 = str(raw_record.get("sha256") or "")
        if expected_sha256 and run_manifest.sha256_file(candidate) == expected_sha256:
            verified.append(candidate)
    return tuple(sorted(verified, key=lambda path: path.name.casefold()))


def _output_kind(
    name: str,
    path: Path,
    *,
    result_status: str,
    formal_output_names: Collection[str],
) -> str:
    tokens = f"{name} {path.name}".casefold()
    basename = path.name.casefold()
    if basename == "processing-plan.json" or basename == "result_linear.fit" or (
        basename.startswith("starun_diagnostics")
    ):
        return "auxiliary"
    if path.suffix.casefold() not in DELIVERABLE_IMAGE_SUFFIXES:
        return "auxiliary"
    if "result_review" in tokens or result_status == STATUS_REVIEW_REQUIRED:
        return "review"
    if (
        result_status in {STATUS_SUCCESS, STATUS_PARTIAL_SUCCESS}
        and name in formal_output_names
    ):
        return "formal"
    return "auxiliary"


def select_verified_png(
    outputs: Sequence[VerifiedOutput],
    *,
    status: str = "",
) -> Path | None:
    """Choose a deterministic preview only from SHA-verified result records."""

    normalized_status = str(status or "").strip().lower()
    selected_kind = (
        "review"
        if normalized_status == STATUS_REVIEW_REQUIRED
        else "formal"
        if normalized_status in {STATUS_SUCCESS, STATUS_PARTIAL_SUCCESS}
        else ""
    )
    pngs = tuple(
        output
        for output in outputs
        if output.path.suffix.casefold() == ".png"
        and output.kind == selected_kind
    )
    if not pngs:
        return None

    def rank(output: VerifiedOutput) -> tuple[int, int, str, str]:
        basename = output.path.name.casefold()
        display_rank = 0 if "display_srgb" in basename else 1
        canonical_rank = 0 if basename in {
            "result_processed.png",
            "result_review.png",
        } else 1
        return (
            display_rank,
            canonical_rank,
            basename,
            output.name.casefold(),
        )

    return min(pngs, key=rank).path


def _verified_output_records(
    run_root: Path,
    result: Mapping[str, Any],
) -> tuple[tuple[VerifiedOutput, ...], tuple[str, ...]]:
    root = run_root.expanduser().resolve()
    raw_outputs = result.get("outputs")
    result_status = str(result.get("status") or "").strip().lower()
    formal_output_names = formal_output_allowlist(result) or frozenset()
    if not isinstance(raw_outputs, Mapping):
        return (), ("pipeline-result.json 没有有效的 outputs 映射",)

    verified: list[VerifiedOutput] = []
    errors: list[str] = []
    seen_paths: set[Path] = set()
    for raw_name, raw_record in sorted(
        raw_outputs.items(),
        key=lambda item: str(item[0]).casefold(),
    ):
        name = str(raw_name)
        if not isinstance(raw_record, Mapping):
            errors.append(f"输出记录 {name} 不是对象")
            continue
        relative_text = str(raw_record.get("path") or "").strip()
        relative = Path(relative_text)
        if not relative_text or relative.is_absolute():
            errors.append(f"输出记录 {name} 的路径无效")
            continue
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            errors.append(f"输出记录 {name} 越出运行目录")
            continue
        expected_sha256 = str(raw_record.get("sha256") or "").strip().casefold()
        if not expected_sha256:
            errors.append(f"输出记录 {name} 缺少 SHA-256")
            continue
        actual_sha256 = run_manifest.sha256_file(candidate)
        if not actual_sha256 or actual_sha256.casefold() != expected_sha256:
            errors.append(f"输出记录 {name} 的 SHA-256 不匹配")
            continue
        if candidate in seen_paths:
            continue
        seen_paths.add(candidate)
        try:
            size = int(candidate.stat().st_size)
        except OSError:
            errors.append(f"输出记录 {name} 无法读取")
            continue
        verified.append(
            VerifiedOutput(
                name=name,
                path=candidate,
                sha256=actual_sha256,
                size=size,
                kind=_output_kind(
                    name,
                    candidate,
                    result_status=result_status,
                    formal_output_names=formal_output_names,
                ),
            )
        )
    return tuple(verified), tuple(dict.fromkeys(errors))


def load_verified_run_bundle(
    task_directory: Path,
    run_id: str,
) -> VerifiedRunBundle:
    """Read one run's signed evidence without mutating or guessing artifacts.

    The signed run manifest and task identity are a hard entry requirement.
    Invalid or incomplete plan/result/output evidence is returned as a
    fail-closed bundle so the result page can explain the problem without
    treating history status as delivery truth.
    """

    _workspace, run_payload, run_root = verify_history_run(task_directory, run_id)
    expected_run_id = str(run_payload.get("run_id") or "")
    expected_source_fingerprint = str(run_payload.get("source_fingerprint") or "")
    expected_run_manifest_hash = str(run_payload.get("manifest_hash") or "")
    integrity_errors: list[str] = []
    verification_issues: list[str] = []

    plan: dict[str, Any] | None = None
    plan_verified = False
    plan_path = run_root / "processing-plan.json"
    raw_plan = run_manifest.load_json(plan_path)
    if raw_plan is None:
        if plan_path.exists():
            integrity_errors.append("processing-plan.json 无法解析")
        else:
            verification_issues.append("processing-plan.json 缺失")
    else:
        plan_check = task_plan.verify_processing_plan(raw_plan)
        if not plan_check.get("verified"):
            integrity_errors.append(
                "processing-plan.json 校验失败："
                + str(plan_check.get("detail") or "unknown error")
            )
        elif str(raw_plan.get("run_id") or "") != expected_run_id:
            integrity_errors.append("processing-plan.json 的 run_id 与运行清单不一致")
        else:
            input_record = raw_plan.get("input")
            input_fingerprint = (
                str(input_record.get("fingerprint") or "")
                if isinstance(input_record, Mapping)
                else ""
            )
            if input_fingerprint and input_fingerprint != expected_source_fingerprint:
                integrity_errors.append("processing-plan.json 的输入指纹与运行清单不一致")
            else:
                metadata = raw_plan.get("metadata")
                recorded_manifest_hash = (
                    str(metadata.get("task_run_manifest_hash") or "")
                    if isinstance(metadata, Mapping)
                    else ""
                )
                if (
                    recorded_manifest_hash
                    and recorded_manifest_hash != expected_run_manifest_hash
                ):
                    integrity_errors.append(
                        "processing-plan.json 引用的运行清单哈希不一致"
                    )
                else:
                    plan = dict(raw_plan)
                    plan_verified = True

    result: dict[str, Any] | None = None
    result_verified = False
    result_path = run_root / "pipeline-result.json"
    if not result_path.is_file():
        if result_path.exists():
            integrity_errors.append("pipeline-result.json 不是普通文件")
        else:
            verification_issues.append("pipeline-result.json 缺失")
    else:
        try:
            normalized_result = load_verified_pipeline_result(run_root)
        except HistoryStoreError as error:
            integrity_errors.append(str(error))
        else:
            if normalized_result is None:
                verification_issues.append("pipeline-result.json 缺失")
            elif str(normalized_result.get("run_id") or "") != expected_run_id:
                integrity_errors.append("pipeline-result.json 的 run_id 与运行清单不一致")
            else:
                result = dict(normalized_result)
                result_verified = True

    lineage_verified = False
    if plan_verified and result_verified and plan is not None and result is not None:
        plan_hash = str(plan.get("plan_hash") or "")
        result_plan_hash = str(result.get("plan_hash") or "")
        if not result_plan_hash or result_plan_hash != plan_hash:
            integrity_errors.append(
                "pipeline-result.json 引用的处理计划哈希不匹配"
            )
        else:
            lineage_verified = True

    verified_outputs: tuple[VerifiedOutput, ...] = ()
    if result_verified and result is not None:
        verified_outputs, output_errors = _verified_output_records(run_root, result)
        integrity_errors.extend(output_errors)
        result_status = str(result.get("status") or "")
        if result_status in {
            STATUS_SUCCESS,
            STATUS_PARTIAL_SUCCESS,
            STATUS_REVIEW_REQUIRED,
        } and not verified_outputs:
            verification_issues.append("流水线结果没有通过 SHA-256 校验的输出文件")
    else:
        result_status = ""

    verified_png = select_verified_png(verified_outputs, status=result_status)
    return VerifiedRunBundle(
        run_root=run_root,
        run_manifest=dict(run_payload),
        plan=plan,
        result=result,
        verified_outputs=verified_outputs,
        verified_png=verified_png,
        plan_verified=plan_verified,
        result_verified=result_verified,
        lineage_verified=lineage_verified,
        integrity_errors=tuple(dict.fromkeys(integrity_errors)),
        verification_issues=tuple(dict.fromkeys(verification_issues)),
    )


def validate_deletable_task_root(candidate: Path) -> TaskWorkspace:
    """Return the verified task or reject every broader/ambiguous target."""

    raw = candidate.expanduser()
    if raw.is_symlink():
        raise UnsafeTaskDeletionError("任务目录是符号链接，拒绝删除")
    if not raw.exists() or not raw.is_dir():
        raise UnsafeTaskDeletionError("任务目录不存在或不是目录")
    if raw.name == TASK_CONTAINER_NAME:
        raise UnsafeTaskDeletionError("不能删除 Starun 容器目录")
    if raw.parent.name != TASK_CONTAINER_NAME:
        raise UnsafeTaskDeletionError(
            "任务目录必须是 Starun 的直接子目录"
        )

    try:
        resolved = raw.resolve(strict=True)
    except OSError as error:
        raise UnsafeTaskDeletionError(f"无法解析任务目录：{error}") from error
    if resolved.is_symlink() or resolved.name == TASK_CONTAINER_NAME:
        raise UnsafeTaskDeletionError("删除目标不是普通任务目录")
    if resolved.parent.name != TASK_CONTAINER_NAME:
        raise UnsafeTaskDeletionError(
            "解析后的任务目录不在 Starun 容器下"
        )

    try:
        workspace, _source = open_task_workspace(resolved)
    except (OSError, WorkspaceError, ValueError, TypeError) as error:
        raise UnsafeTaskDeletionError(f"任务清单无法验签：{error}") from error
    if workspace.root.resolve() != resolved:
        raise UnsafeTaskDeletionError("任务清单解析到了不同目录")
    if workspace.task_id != resolved.name:
        raise UnsafeTaskDeletionError("task_id 与任务目录名不一致")
    return workspace


class HistoryStore:
    """Atomic, path-scoped history registry for runs started by this GUI."""

    def __init__(
        self,
        path: Path | None = None,
        *,
        session_id: str | None = None,
        owner_pid: int | None = None,
    ) -> None:
        self.path = (path or default_history_path()).expanduser()
        self.session_id = str(session_id or uuid.uuid4().hex)
        self.owner_pid = int(owner_pid if owner_pid is not None else os.getpid())
        self.last_recovery_path: Path | None = None

    def _recover_corrupt_index(self) -> dict[str, Any]:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = self.path.with_name(f"{self.path.stem}.corrupt-{stamp}.json")
        index = 1
        while backup.exists():
            backup = self.path.with_name(
                f"{self.path.stem}.corrupt-{stamp}-{index}.json"
            )
            index += 1
        try:
            self.path.replace(backup)
        except OSError as error:
            raise HistoryStoreError(f"历史索引损坏且无法备份：{error}") from error
        self.last_recovery_path = backup
        return _empty_payload()

    def _read(self) -> dict[str, Any]:
        if not self.path.is_file():
            return _empty_payload()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return self._recover_corrupt_index()
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != HISTORY_SCHEMA
            or not isinstance(payload.get("tasks"), list)
        ):
            return self._recover_corrupt_index()
        return payload

    def _write(self, payload: dict[str, Any]) -> None:
        payload["schema"] = HISTORY_SCHEMA
        payload["updated_at"] = utc_timestamp()
        _atomic_write_json(self.path, payload)

    @staticmethod
    def _task_by_key(payload: Mapping[str, Any], task_key: str) -> dict[str, Any] | None:
        tasks = payload.get("tasks")
        if not isinstance(tasks, list):
            return None
        return next(
            (
                task
                for task in tasks
                if isinstance(task, dict)
                and str(task.get("task_key") or "") == str(task_key)
            ),
            None,
        )

    @staticmethod
    def _run_by_id(task: Mapping[str, Any], run_id: str) -> dict[str, Any] | None:
        runs = task.get("runs")
        if not isinstance(runs, list):
            return None
        return next(
            (
                run
                for run in runs
                if isinstance(run, dict)
                and str(run.get("run_id") or "") == str(run_id)
            ),
            None,
        )

    @staticmethod
    def _refresh_task_summary(task: dict[str, Any]) -> None:
        runs = [run for run in task.get("runs", []) if isinstance(run, dict)]
        runs.sort(
            key=lambda run: str(
                run.get("completed_at") or run.get("started_at") or ""
            ),
            reverse=True,
        )
        task["runs"] = runs
        if runs:
            latest = runs[0]
            task["latest_activity_at"] = str(
                latest.get("completed_at") or latest.get("started_at") or ""
            )
            task["latest_status"] = str(
                latest.get("status") or STATUS_INTERRUPTED
            )

    def register_run(
        self,
        *,
        task_id: str,
        task_directory: Path,
        source_fingerprint: str,
        source_record: Mapping[str, Any],
        run_id: str,
        run_directory: Path,
        input_mode: str,
        started_at: str | None = None,
    ) -> dict[str, Any]:
        task_path = task_directory.expanduser().resolve()
        run_path = run_directory.expanduser().resolve()
        try:
            relative = run_path.relative_to((task_path / "runs").resolve())
        except ValueError as error:
            raise HistoryStoreError("运行目录不在任务 runs 目录内") from error
        if len(relative.parts) != 1 or relative.name != str(run_id):
            raise HistoryStoreError("运行目录与 run_id 不匹配")

        payload = self._read()
        tasks = payload.setdefault("tasks", [])
        exact_path = str(task_path)
        task = next(
            (
                item
                for item in tasks
                if isinstance(item, dict)
                and str(item.get("task_directory") or "") == exact_path
            ),
            None,
        )
        if task is None:
            movable = next(
                (
                    item
                    for item in tasks
                    if isinstance(item, dict)
                    and str(item.get("task_id") or "") == str(task_id)
                    and str(item.get("source_fingerprint") or "")
                    == str(source_fingerprint)
                    and not Path(
                        str(item.get("task_directory") or "")
                    ).expanduser().exists()
                ),
                None,
            )
            task = movable
        if task is None:
            task = {"runs": []}
            tasks.append(task)

        task.update(
            {
                "task_key": history_task_key(task_path),
                "task_id": str(task_id),
                "task_directory": exact_path,
                "source_fingerprint": str(source_fingerprint),
                "display_name": display_name_from_source(source_record),
            }
        )
        runs = task.setdefault("runs", [])
        run = self._run_by_id(task, run_id)
        timestamp = str(started_at or utc_timestamp())
        if run is None:
            run = {"run_id": str(run_id)}
            runs.append(run)
        run.update(
            {
                "run_directory": str(run_path),
                "started_at": timestamp,
                "completed_at": None,
                "status": STATUS_PREPARING,
                "input_mode": str(input_mode),
                "owner_pid": self.owner_pid,
                "owner_session": self.session_id,
                "failure_reason": None,
                "exit_code": None,
                "log_path": None,
            }
        )
        self._refresh_task_summary(task)
        self._write(payload)
        return copy.deepcopy(task)

    def update_run(
        self,
        *,
        task_key: str,
        run_id: str,
        status: str,
        completed_at: str | None = None,
        failure_reason: str | None = None,
        exit_code: int | None = None,
        log_path: Path | None = None,
    ) -> dict[str, Any]:
        normalized_status = str(status).strip().lower()
        if normalized_status not in KNOWN_STATUSES:
            raise HistoryStoreError(f"未知历史状态：{status}")
        payload = self._read()
        task = self._task_by_key(payload, task_key)
        if task is None:
            raise HistoryStoreError("历史任务不存在")
        run = self._run_by_id(task, run_id)
        if run is None:
            raise HistoryStoreError("历史运行不存在")
        run["status"] = normalized_status
        if normalized_status in TERMINAL_STATUSES:
            run["completed_at"] = str(completed_at or utc_timestamp())
        if failure_reason is not None:
            run["failure_reason"] = str(failure_reason)
        if exit_code is not None:
            run["exit_code"] = int(exit_code)
        if log_path is not None:
            run["log_path"] = str(log_path.expanduser().resolve())
        self._refresh_task_summary(task)
        self._write(payload)
        return copy.deepcopy(run)

    def mark_incomplete_runs_interrupted(self) -> int:
        payload = self._read()
        changed = 0
        for task in payload.get("tasks", []):
            if not isinstance(task, dict):
                continue
            for run in task.get("runs", []):
                if not isinstance(run, dict) or run.get("status") not in ACTIVE_STATUSES:
                    continue
                if str(run.get("owner_session") or "") == self.session_id:
                    continue
                try:
                    owner_pid = int(run.get("owner_pid") or 0)
                except (TypeError, ValueError):
                    owner_pid = 0
                if _pid_is_alive(owner_pid):
                    continue
                run["status"] = STATUS_INTERRUPTED
                run["completed_at"] = utc_timestamp()
                run["failure_reason"] = "应用或处理进程未正常结束"
                changed += 1
            self._refresh_task_summary(task)
        if changed:
            self._write(payload)
        return changed

    def tasks(self) -> list[dict[str, Any]]:
        payload = self._read()
        tasks = [
            copy.deepcopy(task)
            for task in payload.get("tasks", [])
            if isinstance(task, dict)
        ]
        for task in tasks:
            self._refresh_task_summary(task)
            path = Path(str(task.get("task_directory") or "")).expanduser()
            task["available"] = bool(path.is_dir() and not path.is_symlink())
        tasks.sort(
            key=lambda task: str(task.get("latest_activity_at") or ""),
            reverse=True,
        )
        return tasks

    def find_task(self, task_key: str) -> dict[str, Any] | None:
        return next(
            (
                task
                for task in self.tasks()
                if str(task.get("task_key") or "") == str(task_key)
            ),
            None,
        )

    def remove_task(self, task_key: str) -> bool:
        payload = self._read()
        tasks = payload.get("tasks", [])
        if not isinstance(tasks, list):
            return False
        remaining = [
            task
            for task in tasks
            if not isinstance(task, dict)
            or str(task.get("task_key") or "") != str(task_key)
        ]
        if len(remaining) == len(tasks):
            return False
        payload["tasks"] = remaining
        self._write(payload)
        return True


__all__ = [
    "ACTIVE_STATUSES",
    "HISTORY_SCHEMA",
    "HistoryStore",
    "HistoryStoreError",
    "KNOWN_STATUSES",
    "STATUS_FAILED",
    "STATUS_INTERRUPTED",
    "STATUS_LABELS",
    "STATUS_PARTIAL_SUCCESS",
    "STATUS_PREPARING",
    "STATUS_REVIEW_REQUIRED",
    "STATUS_RUNNING",
    "STATUS_STOPPED",
    "STATUS_SUCCESS",
    "TERMINAL_STATUSES",
    "UnsafeTaskDeletionError",
    "default_history_path",
    "display_name_from_source",
    "history_task_key",
    "load_normalized_run_state",
    "load_verified_run_bundle",
    "load_verified_pipeline_result",
    "select_verified_png",
    "utc_timestamp",
    "validate_deletable_task_root",
    "verified_result_files",
    "verify_history_run",
]
