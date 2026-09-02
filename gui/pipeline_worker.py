#!/usr/bin/env python3
"""Pipeline worker for the Starun GUI."""

from __future__ import annotations

import json
import os
import platform
import queue
import re
import signal
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QThread, Signal

try:
    from .common import *
except ImportError:
    from common import *  # type: ignore[no-redef]

try:
    from .constants import FITS_SUFFIXES
    from .disk_preflight import format_bytes, safe_file_size
except ImportError:
    from constants import FITS_SUFFIXES  # type: ignore[no-redef]
    from disk_preflight import format_bytes, safe_file_size  # type: ignore[no-redef]

try:
    from .native_pipeline_runtime import (
        NATIVE_RUNTIME_MANIFEST_NAME,
        NativePipelineValidationError,
        probe_native_imports,
        stage_native_runtime_payload,
    )
except ImportError:
    from native_pipeline_runtime import (  # type: ignore[no-redef]
        NATIVE_RUNTIME_MANIFEST_NAME,
        NativePipelineValidationError,
        probe_native_imports,
        stage_native_runtime_payload,
    )


BOOTSTRAP_TIMEOUT_ENV = "STARUN_BOOTSTRAP_TIMEOUT_SEC"
DEFAULT_BOOTSTRAP_TIMEOUT_SEC = 300
MIN_BOOTSTRAP_TIMEOUT_SEC = 60
MAX_BOOTSTRAP_TIMEOUT_SEC = 3600
BOOTSTRAP_TIMEOUT_SEC_PER_GIB = 120
BYTES_PER_GIB = 1024 * 1024 * 1024
TEMP_CLEANUP_TIMEOUT_ENV = "STARUN_TEMP_CLEANUP_TIMEOUT_SEC"
DEFAULT_TEMP_CLEANUP_TIMEOUT_SEC = 30
MIN_TEMP_CLEANUP_TIMEOUT_SEC = 1
MAX_TEMP_CLEANUP_TIMEOUT_SEC = 300
WATCHDOG_IDLE_TIMEOUT_ENV = "STARUN_WATCHDOG_IDLE_TIMEOUT_SEC"
DEFAULT_WATCHDOG_IDLE_TIMEOUT_SEC = 900
MIN_WATCHDOG_IDLE_TIMEOUT_SEC = 60
MAX_WATCHDOG_IDLE_TIMEOUT_SEC = 7200
EXPORT_TAIL_TIMEOUT_ENV = "STARUN_EXPORT_TAIL_TIMEOUT_SEC"
DEFAULT_EXPORT_TAIL_TIMEOUT_SEC = 120
MIN_EXPORT_TAIL_TIMEOUT_SEC = 60
MAX_EXPORT_TAIL_TIMEOUT_SEC = 120
WATCHDOG_ARTIFACT_SUFFIXES = frozenset({
    ".fit", ".fits", ".fts", ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".json",
})
_PIPELINE_STAGE_RE = re.compile(
    r"\[(?:INFO|WARN|ERROR|DEBUG)\]\s+阶段\s+(\d+)\s*:\s*([^\r\n]*)",
    re.IGNORECASE,
)
_PIPELINE_STAGE_RESULT_RE = re.compile(
    r"\[PIPELINE_STAGE_RESULT\]\s+"
    r"stage=(\d+)\s+status=([a-z_]+)\s+"
    r"duration=([0-9]+(?:\.[0-9]+)?)\s+title=([^\r\n]*)",
    re.IGNORECASE,
)
_PIPELINE_RUN_SUMMARY_RE = re.compile(
    r"\[PIPELINE_RUN_SUMMARY\]\s+failed=(\d+)\s+degraded=(\d+)",
    re.IGNORECASE,
)
_PIPELINE_RESULT_RE = re.compile(
    r"\[PIPELINE_RESULT\]\s+status="
    r"(success|partial_success|review_required|failed)\b",
    re.IGNORECASE,
)
_SIRIL_PROGRESS_RE = re.compile(
    r"^\s*(?:log:\s*)?progress:\s*(.*?)(?:,\s*)?"
    r"([0-9]+(?:\.[0-9]+)?)%\s*$",
    re.IGNORECASE,
)
_PLUGIN_PROGRESS_RE = re.compile(
    r"\[([^\]\r\n]+\.py)\]\s+\[[#-]+\]\s+"
    r"([0-9]+(?:\.[0-9]+)?)%\s+([^\r\n]+)",
    re.IGNORECASE,
)
_SIRIL_ICC_HDU_SKIP_RE = re.compile(
    r"^\s*(?:log:\s*)?Skipping HDU \d+ with EXTNAME=ICCProfile\s*$",
    re.IGNORECASE,
)
_SIRIL_DENOISE_SRC_DIAGNOSTIC_RE = re.compile(
    r"^\s*(?:log:\s*)?error:\s*no suitable data in src fits\s*$",
    re.IGNORECASE,
)
_TASK_RUNTIME_ENV_KEYS = frozenset(
    {
        "STARUN_TASK_RUN_MANIFEST",
        "STARUN_RUNTIME_CAPABILITIES_MANIFEST",
    }
)
_REMOVED_RUNTIME_ENV_KEYS = frozenset({"STARUN_RESUME_CHECKPOINT_PATH"})
_WORKER_RUNTIME_ENV_ALLOWED_KEYS = (
    RUNTIME_ENV_ALLOWED_KEYS
) | _TASK_RUNTIME_ENV_KEYS
_SEMANTIC_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
_SIRIL_LOG_LINE_RE = re.compile(r"^\s*log:\s?(.*)$")
_MAX_PENDING_PREVIEW_JSON_CHARS = 64 * 1024


def _newest_graxpert_object_model(family_roots: tuple[Path, ...]) -> Path | None:
    candidates: list[tuple[tuple[int, int, int], Path]] = []
    for family_root in family_roots:
        try:
            version_dirs = list(family_root.iterdir()) if family_root.is_dir() else []
        except OSError:
            continue
        for version_dir in version_dirs:
            match = _SEMANTIC_VERSION_RE.fullmatch(version_dir.name)
            model = version_dir / "model.onnx"
            try:
                valid = model.is_file() and model.stat().st_size > 0
            except OSError:
                valid = False
            if match and valid:
                candidates.append(
                    (tuple(int(part) for part in match.groups()), model)
                )
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def default_graxpert_object_model(
    plugin_dir: Path,
    *,
    user_home: Path | None = None,
) -> tuple[Path | None, str]:
    """Return the default app/GraXpert Object Deconvolution model."""
    bundled_model = _newest_graxpert_object_model(
        (
            plugin_dir / "deconvolution-object-ai-models",
            plugin_dir / "graxpert" / "deconvolution-object-ai-models",
            plugin_dir / "models" / "deconvolution-object-ai-models",
        )
    )
    if bundled_model is not None:
        return bundled_model, "starun_app"

    if user_home is None:
        return None, ""
    home = user_home
    graxpert_model = _newest_graxpert_object_model(
        (
            home
            / "Library"
            / "Application Support"
            / "GraXpert"
            / "GraXpert"
            / "deconvolution-object-ai-models",
            home
            / "Library"
            / "Application Support"
            / "GraXpert"
            / "deconvolution-object-ai-models",
            home
            / ".local"
            / "share"
            / "GraXpert"
            / "deconvolution-object-ai-models",
        )
    )
    return graxpert_model, "graxpert_app" if graxpert_model is not None else ""


class PipelineWorker(QThread):
    """Runs Siril processing in a worker thread via siril-cli subprocess."""

    log = Signal(str)
    state = Signal(str)
    progress = Signal(int, str, str)
    stage_detail = Signal(int, object)
    preview = Signal(int, str, str, str)
    done = Signal(str, int, bool, str)

    def __init__(
        self,
        work_dir: Path,
        config_template: Path,
        pipeline_path: Path,
        siril_plugin_dir: Path,
        resources: Path,
        runtime_home: Path,
        siril_candidates: list[Path],
        input_mode: str = INPUT_MODE_AUTO,
        debug_mode: bool = False,
        network_mode: bool = True,
        runtime_overrides: dict[str, str] | None = None,
        runtime_unset_keys: set[str] | None = None,
        graxpert_application_home: Path | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.work_dir = work_dir
        self.config_template = config_template
        self.pipeline_path = pipeline_path
        self.siril_plugin_dir = siril_plugin_dir
        self.resources = resources
        self.runtime_home = runtime_home
        self.siril_candidates = siril_candidates
        self.input_mode = (
            input_mode
            if input_mode in {
                INPUT_MODE_AUTO,
                INPUT_MODE_STAGE1_PREPARED_RESUME,
                INPUT_MODE_LINEAR_RESUME,
                INPUT_MODE_STAGE2_CORRECTED_RESUME,
            }
            else INPUT_MODE_AUTO
        )
        self.debug_mode = bool(debug_mode)
        self.network_mode = bool(network_mode)
        self.runtime_overrides = {
            str(key): str(value)
            for key, value in (runtime_overrides or {}).items()
            if str(key) in _WORKER_RUNTIME_ENV_ALLOWED_KEYS
        }
        self.runtime_unset_keys = {
            str(key)
            for key in (runtime_unset_keys or set())
            if str(key) in _WORKER_RUNTIME_ENV_ALLOWED_KEYS
        }
        self.graxpert_application_home = graxpert_application_home

        self._stop_event = threading.Event()
        self._proc: subprocess.Popen[str] | None = None
        self._proc_pgid: int | None = None
        self._active_mode = "python"
        self._run_had_fatal_errors = False
        self._last_output_ts = 0.0
        self._pyscript_seen_at: float | None = None
        self._pipeline_output_seen = False
        self._python_env_issue = False
        self._python_env_repair_attempted = False
        self._native_process_terminated_detected = False
        self._current_pipeline_stage: int | None = None
        self._pipeline_stage_states: dict[int, str] = {}
        self._pipeline_stage_durations: dict[int, float] = {}
        self._native_termination_stage: int | None = None
        self._native_termination_command = ""
        self._pipeline_summary_failed = 0
        self._pipeline_summary_degraded = 0
        self._pipeline_result_status: str | None = None
        self._pending_preview_json: str | None = None
        self._recent_process_output: deque[str] = deque(maxlen=80)
        self._last_command = ""
        self._saving_png_seen_at: float | None = None
        self._export_tail_ready_at: float | None = None
        self._export_tail_disarmed = False
        self._export_tail_timeout_recovered = False
        self._artifact_snapshot: dict[Path, tuple[int, int]] = {}
        self._last_artifact_scan_ts = 0.0
        self._runtime_env_sources: list[str] = []
        self._runtime_env_applied_keys: list[str] = []
        self._runtime_env_warnings: list[str] = []
        self._runtime_plugin_dir: Path | None = None
        self._temp_cleanup_timeout_sec = DEFAULT_TEMP_CLEANUP_TIMEOUT_SEC
        self._watchdog_idle_timeout_sec = DEFAULT_WATCHDOG_IDLE_TIMEOUT_SEC
        self._export_tail_timeout_sec = DEFAULT_EXPORT_TAIL_TIMEOUT_SEC
        self._progress_log_state: dict[str, tuple[float, int]] = {}
        self._native_log_notice_keys: set[str] = set()
        self._prepared_native_runtime_dir: Path | None = None
        self._prepared_native_manifest_hash: str | None = None
        self._spcc_seed_warning_emitted = False

    def stop(self) -> None:
        self._stop_event.set()
        self._signal_active_processes(signal.SIGTERM)

    def _timestamp(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _append_event(self, msg: str) -> None:
        self.log.emit(f"[{self._timestamp()}] {msg}\n")

    @staticmethod
    def _json_object_is_complete(payload: str) -> bool:
        """Return whether a JSON object closes outside strings."""

        text = payload.lstrip()
        if not text.startswith("{"):
            return False
        depth = 0
        in_string = False
        escaped = False
        for character in text:
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
                continue
            if character == '"':
                in_string = True
            elif character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    return True
        return False

    def _emit_preview_payload(self, raw_payload: str) -> None:
        try:
            preview_payload = json.loads(raw_payload)
            preview_stage = int(preview_payload.get("stage", 0))
            preview_title = str(preview_payload.get("title") or "").strip()
            preview_status = str(
                preview_payload.get("status") or "unavailable"
            ).strip().lower()
            preview_value = str(preview_payload.get("payload") or "").strip()
            if not 1 <= preview_stage <= 10 or preview_status not in {
                "ready",
                "unavailable",
            }:
                raise ValueError("invalid preview event fields")
            if preview_status == "ready" and not Path(preview_value).is_file():
                preview_status = "unavailable"
                preview_value = "preview file is missing"
            self.preview.emit(
                preview_stage,
                preview_title,
                preview_status,
                preview_value,
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            self._append_event(f"忽略无效的阶段预览事件：{exc}")

    def _consume_preview_event(self, text: str) -> None:
        """Reassemble preview JSON split across Siril ``log:`` lines."""

        preview_marker = "[PIPELINE_PREVIEW]"
        preview_index = text.find(preview_marker)
        if preview_index >= 0:
            if self._pending_preview_json is not None:
                self._append_event("忽略未完成的阶段预览事件：收到新的预览事件")
            fragment = text[
                preview_index + len(preview_marker):
            ].rstrip("\r\n").lstrip()
            self._pending_preview_json = fragment
        elif self._pending_preview_json is not None:
            line = text.rstrip("\r\n")
            continuation = _SIRIL_LOG_LINE_RE.match(line)
            if continuation is None:
                self._append_event("忽略未完成的阶段预览事件：日志续行缺失")
                self._pending_preview_json = None
                return
            fragment = continuation.group(1)
            if not self._pending_preview_json and not fragment.lstrip().startswith("{"):
                self._append_event("忽略无效的阶段预览事件：JSON 起始内容缺失")
                self._pending_preview_json = None
                return
            self._pending_preview_json += fragment
        else:
            return

        payload = self._pending_preview_json
        if payload is None:
            return
        if len(payload) > _MAX_PENDING_PREVIEW_JSON_CHARS:
            self._append_event("忽略无效的阶段预览事件：JSON 超出长度限制")
            self._pending_preview_json = None
            return
        if payload and not payload.lstrip().startswith("{"):
            self._pending_preview_json = None
            self._emit_preview_payload(payload)
            return
        if self._json_object_is_complete(payload):
            self._pending_preview_json = None
            self._emit_preview_payload(payload)

    def _progress_log_signature(self, text: str) -> tuple[str, float] | None:
        plugin_match = _PLUGIN_PROGRESS_RE.search(text)
        if plugin_match:
            label = (
                f"plugin:{plugin_match.group(1).strip().lower()}:"
                f"{plugin_match.group(3).strip().lower()}"
            )
            return label, float(plugin_match.group(2))

        siril_match = _SIRIL_PROGRESS_RE.match(text)
        if not siril_match:
            return None
        label = siril_match.group(1).strip(" ,").lower() or "unnamed"
        return f"siril:{label}", float(siril_match.group(2))

    def _should_emit_process_output(self, text: str) -> bool:
        progress = self._progress_log_signature(text)
        if progress is None:
            return True

        label, percent = progress
        stage_key = self._current_pipeline_stage or 0
        key = f"stage:{stage_key}:{label}"
        bucket = max(0, min(10, int(percent // 10)))
        previous = self._progress_log_state.get(key)
        if previous is None:
            self._progress_log_state[key] = (percent, bucket)
            return True

        previous_percent, previous_bucket = previous
        restarted = previous_percent >= 90.0 and percent <= 10.0
        emit = restarted or bucket > previous_bucket or (
            percent >= 100.0 and previous_percent < 100.0
        )
        if restarted:
            self._progress_log_state[key] = (percent, bucket)
        else:
            self._progress_log_state[key] = (
                max(previous_percent, percent),
                max(previous_bucket, bucket),
            )
        return emit

    def _native_output_notice(self, text: str) -> tuple[str, str] | None:
        if _SIRIL_ICC_HDU_SKIP_RE.match(text):
            return (
                "icc_profile_hdu",
                "log: [INFO] Siril 已忽略非图像 ICCProfile FITS 扩展（重复行已折叠）\n",
            )
        if (
            self._current_pipeline_stage == 5
            and "denoise" in self._last_command.lower()
            and _SIRIL_DENOISE_SRC_DIAGNOSTIC_RE.match(text)
        ):
            return (
                "stage5_denoise_src",
                "log: [INFO] Siril NL-Bayes 初始化未使用可选 src FITS 数据；"
                "主图像处理继续，最终以 Stage 5 结果为准。\n",
            )
        return None

    def _emit_process_output(self, text: str) -> None:
        native_notice = self._native_output_notice(text)
        if native_notice is not None:
            key, replacement = native_notice
            if key not in self._native_log_notice_keys:
                self._native_log_notice_keys.add(key)
                self.log.emit(replacement)
            return
        if self._should_emit_process_output(text):
            self.log.emit(text)

    def _runtime_env_candidates(self) -> list[tuple[Path, str]]:
        return [
            (self.resources / DEFAULT_ENV_RESOURCE_REL, "应用默认配置"),
        ]

    def _load_runtime_env_defaults(self) -> tuple[dict[str, str], list[str], list[str]]:
        merged: dict[str, str] = {}
        sources: list[str] = []
        warnings: list[str] = []

        for path, source_label in self._runtime_env_candidates():
            if not path.exists() or not path.is_file():
                continue
            parsed, parse_warnings = parse_runtime_env_file(path)
            if parsed:
                sources.append(source_label)
                merged.update(parsed)
            warnings.extend(
                warning.replace(str(path), source_label)
                for warning in parse_warnings
            )

        return merged, sources, warnings

    def _inspect_output_for_errors(self, text: str) -> None:
        lowered = text.lower()
        stripped = text.strip()
        if (
            stripped
            and self._progress_log_signature(text) is None
            and self._native_output_notice(text) is None
        ):
            self._recent_process_output.append(stripped)
        stage_match = _PIPELINE_STAGE_RE.search(text)
        if stage_match:
            previous_stage = self._current_pipeline_stage
            current_stage = int(stage_match.group(1))
            self._current_pipeline_stage = current_stage
            if (
                current_stage != previous_stage
                or self._pipeline_stage_states.get(current_stage) != "running"
            ):
                self._pipeline_stage_states[current_stage] = "running"
                self.progress.emit(
                    current_stage,
                    stage_match.group(2).strip(),
                    "running",
                )
        result_match = _PIPELINE_STAGE_RESULT_RE.search(text)
        if result_match:
            stage = int(result_match.group(1))
            raw_state = result_match.group(2).strip().lower()
            duration_seconds = float(result_match.group(3))
            state = {
                "ok": "completed",
                "degraded": "degraded",
                "failed": "failed",
                "skipped": "skipped",
            }.get(raw_state, raw_state)
            self._current_pipeline_stage = stage
            self._pipeline_stage_states[stage] = state
            self._pipeline_stage_durations[stage] = (
                self._pipeline_stage_durations.get(stage, 0.0) + duration_seconds
            )
            self.progress.emit(stage, result_match.group(4).strip(), state)
        stage_detail_marker = "[PIPELINE_STAGE_DETAIL]"
        stage_detail_index = text.find(stage_detail_marker)
        if stage_detail_index >= 0:
            raw_payload = text[
                stage_detail_index + len(stage_detail_marker):
            ].strip()
            try:
                stage_detail_payload = json.loads(raw_payload)
                stage_detail_stage = int(stage_detail_payload.get("stage", 0))
                if stage_detail_stage > 0:
                    self.stage_detail.emit(
                        stage_detail_stage,
                        stage_detail_payload,
                    )
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        self._consume_preview_event(text)
        summary_match = _PIPELINE_RUN_SUMMARY_RE.search(text)
        if summary_match:
            self._pipeline_summary_failed = int(summary_match.group(1))
            self._pipeline_summary_degraded = int(summary_match.group(2))
        result_status_match = _PIPELINE_RESULT_RE.search(text)
        if result_status_match:
            self._pipeline_result_status = result_status_match.group(1).lower()
        command_markers = (
            "running command:",
            "running command ",
            "input command:",
            "siril command:",
        )
        if stripped and any(marker in lowered for marker in command_markers):
            self._last_command = stripped
        if (
            "saving png" in lowered
            or (
                "savepng" in lowered
                and any(marker in lowered for marker in command_markers)
            )
        ):
            if self._saving_png_seen_at is None:
                self._saving_png_seen_at = time.time()
        error_markers = (
            "script execution failed",
            "failed to install python module",
            "python not ready yet",
            "python validation failed",
            "python version check failed",
            "failed to initialize python virtual environment",
            "unable to spawn python",
            "error in line",
            "unknown error",
            "exiting batch processing",
            "程序中断:",
            "[siril_native_process_terminated]",
            "sirilnativeprocessterminated",
        )
        python_env_markers = (
            "failed to install python module",
            "python not ready yet",
            "unable to install or update the siril python module",
            "failed to execute pip",
            "pip command failed",
            "python validation failed",
            "python version check failed",
            "failed to initialize python virtual environment",
            "error finding venv python path",
            "unable to spawn python",
            "failed to create python connection",
            "failed to execute python script",
        )
        if any(marker in lowered for marker in error_markers):
            self._run_had_fatal_errors = True
        if (
            "[siril_native_process_terminated]" in lowered
            or "sirilnativeprocessterminated" in lowered
        ):
            self._native_process_terminated_detected = True
            if self._native_termination_stage is None:
                self._native_termination_stage = self._current_pipeline_stage
                self._native_termination_command = self._last_command
        if any(marker in lowered for marker in python_env_markers):
            self._python_env_issue = True
            # Treat Python environment problems as hard pipeline errors even when
            # siril-cli exits 0, so caller can retry/reset/fallback correctly.
            self._run_had_fatal_errors = True
        if (
            ("running command: pyscript" in lowered or "running command pyscript" in lowered)
            and self._pyscript_seen_at is None
        ):
            self._pyscript_seen_at = time.time()
        if self._pyscript_seen_at is not None and stripped:
            # Any non-empty output after entering pyscript means the pipeline is alive.
            if "running command: pyscript" not in lowered and "running command pyscript" not in lowered:
                self._pipeline_output_seen = True
        pipeline_markers = ("stage 1", "阶段 1", "[info]")
        if any(marker in lowered for marker in pipeline_markers):
            self._pipeline_output_seen = True

    def _siril_venv_dir(self) -> Path:
        return self._siril_state_root() / "venv"

    def _siril_state_root(self) -> Path:
        return siril_state_root_from_home(self.runtime_home)

    def _restore_offline_siril_seed(self) -> bool:
        state_root = self._siril_state_root()
        venv_dir = state_root / "venv"
        module_dir = state_root / ".python_module"
        seed_root = self.resources / "SirilPythonSeed"
        seed_venv = seed_root / "venv"
        seed_module = seed_root / ".python_module"

        if not seed_venv.exists() or not seed_module.exists():
            self._append_event(
                f"应用资源中缺少离线 Siril seed：{seed_root}"
            )
            return False

        try:
            state_root.mkdir(parents=True, exist_ok=True)
            if not venv_dir.exists():
                shutil.copytree(seed_venv, venv_dir, symlinks=True)
                self._append_event(f"已从离线 seed 恢复 Siril venv：{venv_dir}")
            if not module_dir.exists():
                shutil.copytree(seed_module, module_dir, symlinks=True)
                self._append_event(
                    f"已从离线 seed 恢复 Siril Python 模块：{module_dir}"
                )
            elif not (module_dir / "sirilpy").exists():
                shutil.rmtree(module_dir, ignore_errors=True)
                shutil.copytree(seed_module, module_dir, symlinks=True)
                self._append_event(
                    f"已重新从离线 seed 恢复 Siril Python 模块：{module_dir}"
                )

            # Rewrite venv interpreter links/config to current bundled Siril path.
            py_bin = (
                self.resources
                / "Siril.app"
                / "Contents"
                / "Frameworks"
                / "Python.framework"
                / "Versions"
                / "3.12"
                / "bin"
                / "python3.12"
            )
            if not py_bin.exists():
                self._append_event(f"内置 Siril Python 缺失：{py_bin}")
                return False

            bin_dir = venv_dir / "bin"
            bin_dir.mkdir(parents=True, exist_ok=True)
            for name in ("python3.12", "python3", "python"):
                dst = bin_dir / name
                if dst.exists() or dst.is_symlink():
                    try:
                        dst.unlink()
                    except Exception:
                        pass
                dst.symlink_to(py_bin)

            cfg = venv_dir / "pyvenv.cfg"
            cfg_lines: list[str] = []
            if cfg.exists():
                cfg_lines = cfg.read_text(encoding="utf-8", errors="replace").splitlines()
            siril_python = self.resources / "Siril.app" / "Contents" / "MacOS" / "python3"
            replacements = {
                "home": str(py_bin.parent),
                "executable": str(py_bin),
                "command": f"{siril_python} -m venv {venv_dir}",
            }
            seen: set[str] = set()
            out_lines: list[str] = []
            for line in cfg_lines:
                if "=" not in line:
                    out_lines.append(line)
                    continue
                key, _value = line.split("=", 1)
                k = key.strip()
                if k in replacements:
                    out_lines.append(f"{k} = {replacements[k]}")
                    seen.add(k)
                else:
                    out_lines.append(line)
            for k, v in replacements.items():
                if k not in seen:
                    out_lines.append(f"{k} = {v}")
            cfg.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

            site_dir = resolve_venv_site_packages(venv_dir)
            repaired = repair_site_packages_from_pip_vendor(site_dir)
            if repaired:
                self._append_event(
                    "已从 pip vendor 修补 Siril venv 依赖："
                    + ", ".join(repaired)
                )
            ok, detail = verify_siril_offline_seed_venv(venv_dir)
            if not ok:
                self._append_event(f"离线 Siril seed 校验失败：{detail}")
                return False
            self._append_event(f"离线 Siril seed 校验通过：{detail}")

            return True
        except Exception as e:
            self._append_event(f"恢复离线 Siril seed 失败：{e}")
            return False

    def _process_identity(self, pid: int) -> tuple[str, str] | None:
        try:
            cp = subprocess.run(
                ["/bin/ps", "-p", str(pid), "-o", "lstart=,command="],
                capture_output=True,
                text=True,
                check=False,
            )
            if cp.returncode != 0:
                return None
            fields = cp.stdout.strip().split(maxsplit=5)
            if len(fields) != 6:
                return None
            return " ".join(fields[:5]), fields[5]
        except Exception:
            return None

    def _collect_processes(
        self,
        needle_parts: tuple[str, ...],
    ) -> list[tuple[int, str, str]]:
        matches: list[tuple[int, str, str]] = []
        try:
            cp = subprocess.run(
                ["/bin/ps", "-ax", "-o", "pid=,lstart=,command="],
                capture_output=True,
                text=True,
                check=False,
            )
            if cp.returncode != 0:
                return matches
            for line in cp.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                fields = line.split(maxsplit=6)
                if len(fields) != 7:
                    continue
                pid_txt = fields[0]
                start_time = " ".join(fields[1:6])
                command = fields[6]
                if not pid_txt.isdigit():
                    continue
                pid = int(pid_txt)
                if pid <= 0 or pid == os.getpid():
                    continue
                if all(part in command for part in needle_parts):
                    matches.append((pid, start_time, command))
        except Exception:
            return matches
        return matches

    def _terminate_processes(self, procs: list[tuple[int, str, str]]) -> int:
        if not procs:
            return 0
        terminated = 0
        unique_procs = {pid: (start_time, command) for pid, start_time, command in procs}
        for pid, expected_identity in unique_procs.items():
            if self._process_identity(pid) != expected_identity:
                continue
            try:
                os.kill(pid, signal.SIGTERM)
                terminated += 1
            except Exception:
                continue
        time.sleep(1.0)
        for pid, expected_identity in unique_procs.items():
            if self._process_identity(pid) != expected_identity:
                continue
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                continue
        return terminated

    def _reset_siril_python_venv(self) -> bool:
        venv_dir = self._siril_venv_dir()
        python_token = str(venv_dir / "bin/python")
        pip_token = str(venv_dir / "bin/pip")
        venv_target = str(venv_dir)

        # Terminate stale venv python/pip processes first.
        stale = self._collect_processes((python_token,))
        stale += self._collect_processes((pip_token,))
        stale += self._collect_processes(("python3 -m venv", venv_target))
        if stale:
            terminated = self._terminate_processes(stale)
            self._append_event(
                f"已终止 {terminated} 个残留的 Siril Python/venv 进程。"
            )

        if not venv_dir.exists():
            self._append_event(
                "Siril Python venv 不存在，开始恢复离线 seed。"
            )
            return self._restore_offline_siril_seed()

        try:
            shutil.rmtree(venv_dir)
            self._append_event(f"已删除 Siril Python venv：{venv_dir}")
            return self._restore_offline_siril_seed()
        except Exception as e:
            self._append_event(f"删除 Siril Python venv 失败：{e}")
            return False

    def _reader(self, stream, out_queue: queue.Queue[str | None]) -> None:
        try:
            for line in iter(stream.readline, ""):
                out_queue.put(line)
        finally:
            out_queue.put(None)

    def _prepare_runtime_files(self, temp_dir: Path) -> tuple[Path, Path, Path]:
        run_ssf = temp_dir / "run_job_embedded.ssf"
        run_ini = temp_dir / "config.1.4.ini"
        run_py = temp_dir / self.pipeline_path.name
        pipeline_dir = self.pipeline_path.parent
        cpu_limit = compute_siril_cpu_limit()
        self._prepared_native_runtime_dir = None
        self._prepared_native_manifest_hash = None

        if not self.pipeline_path.exists():
            raise FileNotFoundError(f"未找到流水线脚本：{self.pipeline_path}")
        for py_file in pipeline_dir.glob("*.py"):
            shutil.copy2(py_file, temp_dir / py_file.name)
        native_manifest = pipeline_dir / NATIVE_RUNTIME_MANIFEST_NAME
        if native_manifest.exists() or native_manifest.is_symlink():
            staged_manifest = stage_native_runtime_payload(pipeline_dir, temp_dir)
            self._prepared_native_runtime_dir = temp_dir
            self._prepared_native_manifest_hash = str(
                staged_manifest["manifest_payload_sha256"]
            )
            self._append_event(
                "CPython 3.12 arm64 原生流水线已校验并写入本轮运行目录。"
            )
        elif is_frozen():
            raise NativePipelineValidationError(
                "正式 App 缺少 CPython 3.12 arm64 原生流水线运行清单；"
                "禁止回退到同名源码模块。"
            )
        stages_dir = pipeline_dir / "stages"
        if stages_dir.exists() and stages_dir.is_dir():
            shutil.copytree(stages_dir, temp_dir / "stages", dirs_exist_ok=True)
        configs_dir = pipeline_dir / "configs"
        if configs_dir.exists() and configs_dir.is_dir():
            shutil.copytree(configs_dir, temp_dir / "configs", dirs_exist_ok=True)
        self._runtime_plugin_dir = None
        if self.siril_plugin_dir.exists() and self.siril_plugin_dir.is_dir():
            plugin_dst = temp_dir / "siril_plugins"
            shared_resource_names = {
                "downloads",
                "syqon_starless",
                "cosmic_clarity",
                "graxpert",
                "model_v2_0_1.onnx",
            }

            def ignore_large_resources(path: str, names: list[str]) -> list[str]:
                if Path(path) != self.siril_plugin_dir:
                    return []
                return [name for name in names if name in shared_resource_names]

            shutil.copytree(
                self.siril_plugin_dir,
                plugin_dst,
                dirs_exist_ok=True,
                ignore=ignore_large_resources,
            )
            linked_resources: list[str] = []
            for name in sorted(shared_resource_names):
                source = self.siril_plugin_dir / name
                if not source.exists():
                    continue
                target = plugin_dst / name
                target.symlink_to(
                    source.resolve(),
                    target_is_directory=source.is_dir(),
                )
                linked_resources.append(name)
            if apply_siril_runtime_patches(plugin_dst):
                self._append_event("已应用 Siril 插件运行时兼容补丁")
            self._runtime_plugin_dir = plugin_dst
            if linked_resources:
                self._append_event(
                    "运行时插件覆盖层已就绪；大资源保持只读引用: "
                    + ", ".join(linked_resources)
                )

        ssf_lines = [
            "requires 1.4.0",
        ]
        if cpu_limit is not None:
            ssf_lines.append(f"setcpu {cpu_limit}")
        ssf_lines.extend(
            [
                f'cd "{shell_quote_path(self.work_dir)}"',
                f'pyscript "{shell_quote_path(run_py)}"',
                "close",
            ]
        )
        run_ssf.write_text("\n".join(ssf_lines) + "\n", encoding="utf-8")

        template_text = self.config_template.read_text(
            encoding="utf-8", errors="replace"
        )
        local_astrometric_catalog = (
            self.runtime_home
            / ".local"
            / "share"
            / "siril"
            / "siril_cat_healpix8_astro.dat"
        )
        local_photometric_catalog = (
            self.runtime_home
            / ".local"
            / "share"
            / "siril"
            / "siril_cat1_healpix8_xpsamp"
        )
        patched = normalize_siril_config_template(
            template_text,
            gaia_photo_catalog=local_photometric_catalog,
            gaia_astro_catalog=local_astrometric_catalog,
        )
        run_ini.write_text(patched, encoding="utf-8")
        return run_ssf, run_ini, run_py

    def _build_env(self, siril_cli: Path) -> dict[str, str]:
        env = scrub_python_env(os.environ.copy())
        for task_key in _TASK_RUNTIME_ENV_KEYS | _REMOVED_RUNTIME_ENV_KEYS:
            env.pop(task_key, None)
        runtime_env, runtime_sources, runtime_warnings = (
            self._load_runtime_env_defaults()
        )
        applied_keys: list[str] = []
        for key, value in runtime_env.items():
            if not env.get(key):
                env[key] = value
                applied_keys.append(key)
        for key in self.runtime_unset_keys:
            env.pop(key, None)
        for key, value in self.runtime_overrides.items():
            env[key] = value
        self._runtime_env_sources = runtime_sources
        self._runtime_env_applied_keys = sorted(applied_keys)
        self._runtime_env_warnings = runtime_warnings

        default_object_model, default_model_source = default_graxpert_object_model(
            self.siril_plugin_dir,
            user_home=self.graxpert_application_home,
        )
        user_graxpert_model = env.get("STARUN_GRAXPERT_OBJECT_MODEL_PATH", "").strip()
        if (
            env.get("STARUN_STAGE5_GRAXPERT_DECONV_ENABLE", "1").strip().lower()
            not in {"0", "false", "no", "off"}
            and default_object_model is not None
        ):
            env["STARUN_GRAXPERT_OBJECT_MODEL_PATH"] = str(default_object_model)
            source_label = (
                "Starun App 内"
                if default_model_source == "starun_app"
                else "本机 GraXpert 应用"
            )
            self._append_event(
                f"Stage 5 将优先使用{source_label}对象反卷积模型："
                f"{default_object_model.parent.name}"
            )
        elif user_graxpert_model:
            expanded_model = Path(os.path.expandvars(user_graxpert_model)).expanduser()
            if not expanded_model.is_absolute():
                expanded_model = self.work_dir / expanded_model
            env["STARUN_GRAXPERT_OBJECT_MODEL_PATH"] = str(expanded_model)
            if expanded_model.exists():
                self._append_event(f"已配置用户 GraXpert 对象反卷积模型：{expanded_model}")
            else:
                self._append_event(
                    "用户 GraXpert 对象反卷积模型路径不存在；"
                    f"Stage 5 将安全回退 Siril RL：{expanded_model}"
                )

        # Finder-launched apps may lack UTF-8 locale vars.
        env["HOME"] = str(self.runtime_home)
        env["LANG"] = "en_US.UTF-8"
        env["LC_ALL"] = "en_US.UTF-8"
        env["LC_CTYPE"] = "en_US.UTF-8"
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        # Siril pyscript plugins are non-interactive.  Keep them on their own
        # runtime Qt and prevent a Cocoa plugin from pulling the frozen GUI's
        # PySide6 frameworks into a PyQt6 process.
        env["QT_QPA_PLATFORM"] = "offscreen"
        env.setdefault(
            BOOTSTRAP_TIMEOUT_ENV,
            str(DEFAULT_BOOTSTRAP_TIMEOUT_SEC),
        )
        env.setdefault(
            TEMP_CLEANUP_TIMEOUT_ENV,
            str(DEFAULT_TEMP_CLEANUP_TIMEOUT_SEC),
        )
        env.setdefault(
            WATCHDOG_IDLE_TIMEOUT_ENV,
            str(DEFAULT_WATCHDOG_IDLE_TIMEOUT_SEC),
        )
        env.setdefault(
            EXPORT_TAIL_TIMEOUT_ENV,
            str(DEFAULT_EXPORT_TAIL_TIMEOUT_SEC),
        )
        try:
            cleanup_timeout = int(round(float(env[TEMP_CLEANUP_TIMEOUT_ENV])))
        except (OverflowError, TypeError, ValueError):
            cleanup_timeout = DEFAULT_TEMP_CLEANUP_TIMEOUT_SEC
            self._append_event(
                f"{TEMP_CLEANUP_TIMEOUT_ENV}={env[TEMP_CLEANUP_TIMEOUT_ENV]!r} 无效，"
                f"使用默认值 {DEFAULT_TEMP_CLEANUP_TIMEOUT_SEC}s。"
            )
        self._temp_cleanup_timeout_sec = max(
            MIN_TEMP_CLEANUP_TIMEOUT_SEC,
            min(MAX_TEMP_CLEANUP_TIMEOUT_SEC, cleanup_timeout),
        )
        self._watchdog_idle_timeout_sec = self._bounded_timeout_from_env(
            env,
            WATCHDOG_IDLE_TIMEOUT_ENV,
            DEFAULT_WATCHDOG_IDLE_TIMEOUT_SEC,
            MIN_WATCHDOG_IDLE_TIMEOUT_SEC,
            MAX_WATCHDOG_IDLE_TIMEOUT_SEC,
        )
        self._export_tail_timeout_sec = self._bounded_timeout_from_env(
            env,
            EXPORT_TAIL_TIMEOUT_ENV,
            DEFAULT_EXPORT_TAIL_TIMEOUT_SEC,
            MIN_EXPORT_TAIL_TIMEOUT_SEC,
            MAX_EXPORT_TAIL_TIMEOUT_SEC,
        )
        env.setdefault("STARUN_SIRILPY_TIMEOUT_SEC", "300")
        env["PIP_NO_INDEX"] = "1"
        env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        pip_find_links: list[str] = []
        bundled_downloads = self.resources / "siril_plugins" / "downloads"
        if bundled_downloads.is_dir():
            pip_find_links.append(str(bundled_downloads))
        plugin_downloads = self.siril_plugin_dir / "downloads"
        if plugin_downloads.is_dir():
            pip_find_links.append(str(plugin_downloads))
        if self._runtime_plugin_dir:
            runtime_downloads = self._runtime_plugin_dir / "downloads"
            if runtime_downloads.is_dir():
                pip_find_links.append(str(runtime_downloads))
        if pip_find_links:
            env["PIP_FIND_LINKS"] = " ".join(dict.fromkeys(pip_find_links))
        runtime_python_candidates = (
            self._siril_venv_dir() / "bin" / "python3.12",
            self._siril_venv_dir() / "bin" / "python3",
            self._siril_venv_dir() / "bin" / "python",
        )
        runtime_py = next(
            (
                candidate
                for candidate in runtime_python_candidates
                if candidate.exists() and os.access(candidate, os.X_OK)
            ),
            None,
        )
        bundled_py = (
            self.resources
            / "Siril.app"
            / "Contents"
            / "Frameworks"
            / "Python.framework"
            / "Versions"
            / "3.12"
            / "bin"
            / "python3.12"
        )
        python_cli = runtime_py or (bundled_py if bundled_py.exists() else None)
        if python_cli is not None:
            env["SIRIL_PYTHON_CLI"] = str(python_cli)
            env["STARUN_SIRIL_PYTHON_CLI"] = str(python_cli)

        bundled_siril_cli = self.resources / "Siril.app" / "Contents" / "MacOS" / "siril-cli"
        if siril_cli == bundled_siril_cli:
            relocated = self.resources / "Siril.app" / "Contents" / "Resources"
            env["SIRIL_RELOCATED_RES_DIR"] = str(relocated)

        if self._runtime_plugin_dir and self._runtime_plugin_dir.exists():
            env["STARUN_SIRIL_PLUGIN_DIR"] = str(self._runtime_plugin_dir)
            classic_wrapper = self._runtime_plugin_dir / "bin" / "CosmicClarity"
            if classic_wrapper.is_file() and os.access(classic_wrapper, os.X_OK):
                env.setdefault("STARUN_COSMIC_CLARITY_EXECUTABLE", str(classic_wrapper))
            scripts_dir = resolve_siril_scripts_root(self._runtime_plugin_dir)
            if scripts_dir is not None:
                env["SIRIL_SCRIPTS_DIR"] = str(scripts_dir)
                env["SIRIL_SCRIPTS_PATH"] = str(scripts_dir)

        syqon_models = self.siril_plugin_dir / SYQON_STARLESS_BUNDLE_REL
        if (
            (syqon_models / "zenith.pt").is_file()
            and (syqon_models / "zenith.pt.sha256").is_file()
        ):
            env["STARUN_SYQON_MODEL_DIR"] = str(syqon_models)
        cosmic_models = self.siril_plugin_dir / COSMIC_CLARITY_BUNDLE_REL
        if cosmic_models.is_dir():
            env["STARUN_COSMIC_CLARITY_MODEL_DIR"] = str(cosmic_models)

        env["STARUN_DEBUG_MODE"] = "1" if self.debug_mode else "0"
        env["STARUN_INPUT_MODE"] = self.input_mode
        env["STARUN_NETWORK_MODE"] = "1" if self.network_mode else "0"
        local_catalog_root = self.runtime_home / ".local" / "share" / "siril"
        env["STARUN_GAIA_PHOTO_CATALOG"] = str(
            local_catalog_root / "siril_cat1_healpix8_xpsamp"
        )
        env["STARUN_GAIA_ASTRO_CATALOG"] = str(
            local_catalog_root / "siril_cat_healpix8_astro.dat"
        )
        env["STARUN_SPCC_DATABASE_DIR"] = str(
            siril_spcc_database_root_from_home(self.runtime_home)
        )
        try:
            seed_result = sync_siril_spcc_database_seed(
                default_siril_spcc_database_seed_dir(self.resources),
                self.runtime_home,
            )
            copied_files = seed_result.get("copied_files", [])
            if copied_files:
                self._append_event(
                    "已校验并部署 Siril SPCC 传感器/滤镜数据库："
                    f"{len(copied_files)} 个文件"
                )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            env["STARUN_SPCC_ENABLE"] = "0"
            if not self._spcc_seed_warning_emitted:
                self._spcc_seed_warning_emitted = True
                self._append_event(
                    "Siril SPCC 固定元数据不可用，已在启动前禁用 SPCC；"
                    f"宽带将进入 PCC 异常回退：{error}"
                )
        return env

    def _bounded_timeout_from_env(
        self,
        env: dict[str, str],
        key: str,
        default: int,
        minimum: int,
        maximum: int,
    ) -> int:
        raw_value = env.get(key, str(default))
        try:
            parsed = int(round(float(raw_value)))
        except (OverflowError, TypeError, ValueError):
            parsed = default
            self._append_event(
                f"{key}={raw_value!r} 无效，使用默认值 {default}s。"
            )
        bounded = max(minimum, min(maximum, parsed))
        if bounded != parsed:
            self._append_event(
                f"{key}={parsed}s 超出范围，已限制为 {bounded}s。"
            )
        return bounded

    def _bootstrap_input_size_bytes(self) -> int:
        if not self.work_dir.is_dir():
            return 0
        total = 0
        try:
            candidates = self.work_dir.iterdir()
            for path in candidates:
                if path.is_file() and path.suffix.lower() in FITS_SUFFIXES:
                    total += safe_file_size(path)
        except OSError:
            return total
        return total

    def _bootstrap_timeout_sec(
        self,
        env: dict[str, str] | None = None,
    ) -> tuple[int, int, int]:
        source_env = env if env is not None else os.environ
        raw_value = source_env.get(
            BOOTSTRAP_TIMEOUT_ENV,
            str(DEFAULT_BOOTSTRAP_TIMEOUT_SEC),
        )
        try:
            base_timeout = int(round(float(raw_value)))
        except (OverflowError, TypeError, ValueError):
            base_timeout = DEFAULT_BOOTSTRAP_TIMEOUT_SEC
            self._append_event(
                f"{BOOTSTRAP_TIMEOUT_ENV}={raw_value!r} 无效，"
                f"使用默认值 {DEFAULT_BOOTSTRAP_TIMEOUT_SEC}s。"
            )
        base_timeout = max(
            MIN_BOOTSTRAP_TIMEOUT_SEC,
            min(MAX_BOOTSTRAP_TIMEOUT_SEC, base_timeout),
        )

        input_bytes = self._bootstrap_input_size_bytes()
        adaptive_extra = (
            input_bytes * BOOTSTRAP_TIMEOUT_SEC_PER_GIB + BYTES_PER_GIB - 1
        ) // BYTES_PER_GIB
        effective_timeout = min(
            MAX_BOOTSTRAP_TIMEOUT_SEC,
            base_timeout + adaptive_extra,
        )
        return effective_timeout, base_timeout, input_bytes

    def _cleanup_temp_dir(
        self,
        temp_dir: Path,
        timeout_sec: float | None = None,
    ) -> bool:
        timeout = (
            float(self._temp_cleanup_timeout_sec)
            if timeout_sec is None
            else max(0.0, float(timeout_sec))
        )
        errors: list[Exception] = []
        removed = threading.Event()

        def remove() -> None:
            try:
                shutil.rmtree(temp_dir)
            except FileNotFoundError:
                removed.set()
            except Exception as exc:
                errors.append(exc)
            else:
                removed.set()

        cleanup_thread = threading.Thread(
            target=remove,
            name="starun-temp-cleanup",
            daemon=True,
        )
        cleanup_thread.start()
        cleanup_thread.join(timeout)

        if cleanup_thread.is_alive():
            self._append_event(
                f"临时目录清理超过 {timeout:g}s，已转为后台清理：{temp_dir}"
            )
            return False
        if errors:
            self._append_event(f"临时目录清理失败，已保留：{temp_dir} ({errors[0]})")
            return False
        if not removed.is_set():
            self._append_event(f"临时目录清理异常中止，已保留：{temp_dir}")
            return False
        return True

    def _signal_active_processes(self, sig: signal.Signals) -> bool:
        """Signal only the process group created for the current run."""
        pgid = self._proc_pgid
        if pgid is not None:
            try:
                os.killpg(pgid, sig)
                return True
            except ProcessLookupError:
                return False
            except (OSError, PermissionError):
                pass

        proc = self._proc
        if proc is None or proc.poll() is not None:
            return False
        try:
            proc.send_signal(sig)
            return True
        except (OSError, ProcessLookupError):
            return False

    def _process_group_alive(self) -> bool:
        pgid = self._proc_pgid
        if pgid is None:
            proc = self._proc
            return bool(proc is not None and proc.poll() is None)
        try:
            os.killpg(pgid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False

    def _terminate_active_processes(self, grace_sec: float = 5.0) -> None:
        """Terminate Siril and every child in this run's isolated process group."""
        self._signal_active_processes(signal.SIGTERM)
        deadline = time.monotonic() + max(0.0, grace_sec)
        while self._process_group_alive() and time.monotonic() < deadline:
            proc = self._proc
            if proc is not None:
                proc.poll()
            time.sleep(0.1)
        if self._process_group_alive():
            self._signal_active_processes(signal.SIGKILL)
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass

    def _artifact_paths(self) -> list[Path]:
        paths: list[Path] = []
        try:
            for path in self.work_dir.iterdir():
                if path.is_file() and path.suffix.lower() in WATCHDOG_ARTIFACT_SUFFIXES:
                    paths.append(path)
        except OSError:
            pass

        process_dir = self.work_dir / "process"
        if process_dir.is_dir():
            try:
                paths.extend(
                    path
                    for path in process_dir.rglob("*")
                    if path.is_file()
                    and path.suffix.lower() in WATCHDOG_ARTIFACT_SUFFIXES
                )
            except OSError:
                pass
        return sorted(set(paths), key=lambda path: str(path))

    @staticmethod
    def _artifact_signature(path: Path) -> tuple[int, int] | None:
        try:
            stat_result = path.stat()
            return stat_result.st_mtime_ns, stat_result.st_size
        except OSError:
            return None

    def _capture_artifact_snapshot(self) -> None:
        self._artifact_snapshot = {
            path: signature
            for path in self._artifact_paths()
            if (signature := self._artifact_signature(path)) is not None
        }

    def _generated_artifacts(self) -> list[Path]:
        generated: list[Path] = []
        for path in self._artifact_paths():
            signature = self._artifact_signature(path)
            if signature is None or signature[1] <= 0:
                continue
            if self._artifact_snapshot.get(path) != signature:
                generated.append(path)
        return generated

    def _refresh_export_tail_state(self, now: float) -> None:
        if (
            self._saving_png_seen_at is None
            or self._export_tail_ready_at is not None
            or self._export_tail_disarmed
            or now - self._last_artifact_scan_ts < 1.0
        ):
            return
        self._last_artifact_scan_ts = now
        generated_pngs = [
            path
            for path in self._generated_artifacts()
            if path.parent == self.work_dir and path.suffix.lower() == ".png"
        ]
        if not generated_pngs:
            return
        self._export_tail_ready_at = now
        names = ", ".join(path.name for path in generated_pngs[:3])
        self._append_event(
            "Watchdog 已确认 PNG 导出产物："
            f"{names}；若持续无输出且进程未退出，将在 "
            f"{self._export_tail_timeout_sec}s 后清理收尾残留进程。"
        )

    def _watchdog_timeout_reason(self, now: float) -> tuple[str, float] | None:
        if self._export_tail_ready_at is not None and not self._export_tail_disarmed:
            tail_progress_ts = max(self._export_tail_ready_at, self._last_output_ts)
            tail_idle_sec = max(0.0, now - tail_progress_ts)
            if tail_idle_sec >= self._export_tail_timeout_sec:
                return "export_tail", tail_idle_sec

        idle_sec = max(0.0, now - self._last_output_ts)
        if idle_sec >= self._watchdog_idle_timeout_sec:
            return "idle", idle_sec
        return None

    def _process_group_snapshot(self) -> list[str]:
        pgid = self._proc_pgid
        proc = self._proc
        root_pid = proc.pid if proc is not None else None
        if pgid is None and root_pid is None:
            return []
        try:
            cp = subprocess.run(
                [
                    "/bin/ps",
                    "-ax",
                    "-o",
                    "pid=,ppid=,pgid=,stat=,etime=,%cpu=,%mem=,command=",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return []
        if cp.returncode != 0:
            return []

        rows: list[str] = []
        for raw_line in cp.stdout.splitlines():
            fields = raw_line.strip().split(maxsplit=7)
            if len(fields) != 8:
                continue
            try:
                pid = int(fields[0])
                row_pgid = int(fields[2])
            except ValueError:
                continue
            if (pgid is not None and row_pgid != pgid) or (
                pgid is None and pid != root_pid
            ):
                continue
            rows.append(
                "pid={pid} ppid={ppid} pgid={pgid} stat={stat} "
                "etime={etime} cpu={cpu}% mem={mem}% cmd={command}".format(
                    pid=fields[0],
                    ppid=fields[1],
                    pgid=fields[2],
                    stat=fields[3],
                    etime=fields[4],
                    cpu=fields[5],
                    mem=fields[6],
                    command=fields[7],
                )
            )
        return rows

    def _append_watchdog_diagnostics(
        self,
        *,
        reason: str,
        idle_sec: float,
    ) -> None:
        reason_label = {
            "bootstrap": "pyscript 启动超时",
            "idle": "流水线长时间无输出",
            "export_tail": "导出完成后收尾未退出",
        }.get(reason, reason)
        self._append_event(
            f"Watchdog 触发：{reason_label}（无输出 {idle_sec:.1f}s）。"
        )
        self._append_event(f"Watchdog 最后命令：{self._last_command or '<未识别>'}")

        proc = self._proc
        poll_state = proc.poll() if proc is not None else None
        process_rows = self._process_group_snapshot()
        self._append_event(
            "Watchdog 进程状态："
            f"root_pid={proc.pid if proc is not None else '<无>'}, "
            f"pgid={self._proc_pgid if self._proc_pgid is not None else '<无>'}, "
            f"poll={poll_state if poll_state is not None else 'running'}"
        )
        if process_rows:
            for row in process_rows:
                self._append_event(f"  {row}")
        else:
            self._append_event("  <未发现存活的本次运行进程>")

        artifacts = self._generated_artifacts()
        self._append_event("Watchdog 已生成产物：")
        if not artifacts:
            self._append_event("  <未检测到本次运行新建或改写的产物>")
            return
        for path in artifacts[:40]:
            try:
                display_path = path.relative_to(self.work_dir)
            except ValueError:
                display_path = path
            self._append_event(
                f"  {display_path} ({format_bytes(safe_file_size(path))})"
            )
        if len(artifacts) > 40:
            self._append_event(f"  ... 另有 {len(artifacts) - 40} 个产物")

    def _completed_run_status(self) -> str:
        if self._pipeline_result_status == "failed":
            return "Failed"
        if (
            self._export_tail_timeout_recovered
            or self._pipeline_summary_failed > 0
            or self._pipeline_summary_degraded > 0
            or self._pipeline_result_status
            in {"partial_success", "review_required"}
        ):
            return "CompletedWithWarning"
        return "Completed"

    def _run_once(self, siril_cli: Path, run_ssf: Path, run_ini: Path) -> tuple[bool, int]:
        self._run_had_fatal_errors = False
        self._python_env_issue = False
        self._pyscript_seen_at = None
        self._pipeline_output_seen = False
        self._native_process_terminated_detected = False
        self._current_pipeline_stage = None
        self._native_termination_stage = None
        self._native_termination_command = ""
        self._pipeline_summary_failed = 0
        self._pipeline_summary_degraded = 0
        self._pipeline_result_status = None
        self._pending_preview_json = None
        self._recent_process_output.clear()
        self._last_command = ""
        self._saving_png_seen_at = None
        self._export_tail_ready_at = None
        self._export_tail_disarmed = False
        self._export_tail_timeout_recovered = False
        self._progress_log_state.clear()
        self._native_log_notice_keys.clear()
        self._last_artifact_scan_ts = 0.0
        self._proc_pgid = None
        self._last_output_ts = time.time()
        self._capture_artifact_snapshot()

        cmd = build_siril_cli_command(
            siril_cli=siril_cli,
            work_dir=self.work_dir,
            run_ini=run_ini,
            run_ssf=run_ssf,
            offline_mode=not self.network_mode,
        )
        self._append_event(f"开始启动进程（{self._active_mode}），使用 {siril_cli}")
        self._append_event("命令：" + " ".join(cmd))
        proc_env = self._build_env(siril_cli)
        proc_env["STARUN_SIRIL_CLI"] = str(siril_cli)
        proc_env["STARUN_SIRIL_CONFIG"] = str(run_ini)
        self._append_event(f"Siril 运行时主目录：{proc_env.get('HOME', '')}")
        self._append_event(
            "Siril Python CLI："
            + proc_env.get("SIRIL_PYTHON_CLI", "<未设置>")
        )
        self._append_event(
            f"调试模式: {'ON' if self.debug_mode else 'OFF'}"
        )
        self._append_event(
            f"联网模式: {'ON' if self.network_mode else 'OFF'}"
        )
        self._append_event(
            f"输入模式: {proc_env.get('STARUN_INPUT_MODE', self.input_mode)}"
        )
        bootstrap_timeout_sec, bootstrap_base_sec, bootstrap_input_bytes = (
            self._bootstrap_timeout_sec(proc_env)
        )
        self._append_event(
            "pyscript bootstrap 超时: "
            f"{bootstrap_timeout_sec}s "
            f"(base={bootstrap_base_sec}s, FITS={format_bytes(bootstrap_input_bytes)}, "
            f"rate={BOOTSTRAP_TIMEOUT_SEC_PER_GIB}s/GiB)"
        )
        self._append_event(
            "运行 watchdog: "
            f"普通无输出={self._watchdog_idle_timeout_sec}s，"
            f"PNG 导出后收尾无输出={self._export_tail_timeout_sec}s"
        )
        if self._runtime_env_sources:
            self._append_event(
                "运行环境配置来源: " + ", ".join(self._runtime_env_sources)
            )
            if self._runtime_env_applied_keys:
                self._append_event(
                    "运行环境已注入键: "
                    + ", ".join(self._runtime_env_applied_keys)
                )
        for warning in self._runtime_env_warnings:
            self._append_event(f"运行环境配置警告: {warning}")

        if (
            self._prepared_native_runtime_dir is not None
            and self._prepared_native_manifest_hash is not None
        ):
            runtime_python = proc_env.get("SIRIL_PYTHON_CLI", "")
            if not runtime_python:
                self._append_event(
                    "启动前原生流水线复核失败：未解析到 Siril CPython 3.12。"
                )
                self._run_had_fatal_errors = True
                return False, -1
            try:
                probe_native_imports(
                    Path(runtime_python),
                    self._prepared_native_runtime_dir,
                    expected_manifest_payload_sha256=(
                        self._prepared_native_manifest_hash
                    ),
                )
            except NativePipelineValidationError as error:
                self._append_event(f"启动前原生流水线复核失败：{error}")
                self._run_had_fatal_errors = True
                return False, -1
            self._append_event(
                "已在启动 Siril 前复核本轮临时目录中的原生模块。"
            )

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=proc_env,
                start_new_session=True,
            )
            self._proc_pgid = self._proc.pid
        except Exception as e:
            self._append_event(f"启动进程失败：{e}")
            self._run_had_fatal_errors = True
            return False, -1

        if self._proc.stdout is None:
            self._append_event("无法捕获进程输出。")
            self._run_had_fatal_errors = True
            return False, -1

        out_queue: queue.Queue[str | None] = queue.Queue()
        reader_t = threading.Thread(target=self._reader, args=(self._proc.stdout, out_queue), daemon=True)
        reader_t.start()

        bootstrap_timeout = False
        watchdog_timeout: str | None = None
        reader_done = False
        stop_termination_requested = False

        while True:
            drained = False
            while True:
                try:
                    item = out_queue.get_nowait()
                except queue.Empty:
                    break
                drained = True
                if item is None:
                    reader_done = True
                    break
                self._last_output_ts = time.time()
                self._inspect_output_for_errors(item)
                self._emit_process_output(item)

            proc_ret = self._proc.poll()

            if self._stop_event.is_set() and not stop_termination_requested:
                stop_termination_requested = True
                self._append_event("已请求停止...")
                self._terminate_active_processes()

            now = time.time()
            self._refresh_export_tail_state(now)
            if (
                proc_ret is None
                and self._active_mode == "python"
                and self._pyscript_seen_at
                and not self._pipeline_output_seen
                and now - self._pyscript_seen_at > bootstrap_timeout_sec
            ):
                bootstrap_timeout = True
                self._run_had_fatal_errors = True
                self._python_env_issue = True
                self._append_event(
                    "pyscript 启动超时"
                    f"（>{bootstrap_timeout_sec}s）：Siril Python 环境疑似卡住。"
                )
                self._append_event(
                    "提示：关闭 Siril，删除 "
                    f"'{self._siril_venv_dir()}' 后重试。"
                )
                self._append_watchdog_diagnostics(
                    reason="bootstrap",
                    idle_sec=now - self._pyscript_seen_at,
                )
                self._terminate_active_processes()

            process_pending = proc_ret is None or not reader_done
            watchdog_reason = self._watchdog_timeout_reason(now)
            if (
                process_pending
                and watchdog_timeout is None
                and not bootstrap_timeout
                and not self._stop_event.is_set()
                and watchdog_reason is not None
            ):
                watchdog_timeout, idle_sec = watchdog_reason
                if watchdog_timeout == "export_tail":
                    self._export_tail_timeout_recovered = True
                else:
                    self._run_had_fatal_errors = True
                self._append_watchdog_diagnostics(
                    reason=watchdog_timeout,
                    idle_sec=idle_sec,
                )
                self._terminate_active_processes()

            proc_ret = self._proc.poll()
            if proc_ret is not None and reader_done and out_queue.empty():
                break

            if not drained:
                time.sleep(0.1)

        # Drain remaining output after process exit.
        while True:
            try:
                item = out_queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                continue
            self._inspect_output_for_errors(item)
            self._emit_process_output(item)

        try:
            self._proc.stdout.close()
        except Exception:
            pass
        reader_t.join(timeout=1)

        exit_code = self._proc.returncode if self._proc.returncode is not None else -1
        self._proc = None
        self._proc_pgid = None

        if bootstrap_timeout:
            return False, exit_code
        if watchdog_timeout == "export_tail":
            self._append_event(
                "最终导出产物已确认；残留进程已终止，"
                "本次按“导出成功、收尾异常”完成。"
            )
            return True, exit_code
        if watchdog_timeout is not None:
            return False, exit_code

        if (
            self._active_mode == "python"
            and self._pyscript_seen_at is not None
            and not self._pipeline_output_seen
            and not self._stop_event.is_set()
        ):
            if not self._run_had_fatal_errors:
                self._append_event(
                    "pyscript 启动后未检测到流水线阶段输出，本次运行按失败处理。"
                )
            self._run_had_fatal_errors = True

        success = (
            exit_code == 0
            and not self._run_had_fatal_errors
            and not self._stop_event.is_set()
        )
        return success, exit_code

    def run(self) -> None:
        self.state.emit("Running")
        run_status = "Failed"
        exit_code = -1
        cli_used = ""

        temp_dir = Path(tempfile.mkdtemp(prefix="starun_embedded_"))
        try:
            run_ssf, run_ini, _run_py = self._prepare_runtime_files(temp_dir)

            for attempt, siril_cli in enumerate(self.siril_candidates, start=1):
                if self._stop_event.is_set():
                    run_status = "Stopped"
                    break

                cli_used = str(siril_cli)
                success, exit_code = self._run_once(siril_cli, run_ssf, run_ini)
                if self._stop_event.is_set():
                    run_status = "Stopped"
                    break

                if success:
                    run_status = self._completed_run_status()
                    break

                if self._python_env_issue and not self._python_env_repair_attempted:
                    self._python_env_repair_attempted = True
                    self._append_event(
                        "检测到 Siril Python 环境异常。"
                        "正在重置 Siril Python venv，并重试一次..."
                    )
                    if self._reset_siril_python_venv():
                        success, exit_code = self._run_once(siril_cli, run_ssf, run_ini)
                        if self._stop_event.is_set():
                            run_status = "Stopped"
                            break
                        if success:
                            run_status = self._completed_run_status()
                            break
                    else:
                        self._append_event(
                            "自动重置 venv 失败。"
                        )

                if attempt < len(self.siril_candidates):
                    self._append_event(
                        "主流水线失败或卡住，"
                        "正在使用备用 Siril 运行时重试完整内置流水线..."
                    )
            else:
                run_status = "Failed"

        except Exception as e:
            self._append_event(f"Worker 内部错误：{e}")
            self._run_had_fatal_errors = True
            run_status = "Failed"
            exit_code = -1
        finally:
            self._cleanup_temp_dir(temp_dir)

        if self._stop_event.is_set() and run_status != "Stopped":
            run_status = "Stopped"
        if run_status == "Failed":
            self._run_had_fatal_errors = True

        self.done.emit(
            run_status,
            exit_code,
            self._run_had_fatal_errors,
            cli_used,
        )
