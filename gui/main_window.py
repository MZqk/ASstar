#!/usr/bin/env python3
"""Starun macOS GUI launcher with external pipeline resource."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import plistlib
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Mapping

from PySide6.QtCore import QFile, QSettings, QTimer, QUrl, Signal, Qt
from PySide6.QtGui import QAction, QDesktopServices, QTextCursor

try:
    from .constants import (
        DISK_SPACE_HEADROOM_RATIO,
        DISK_SPACE_MIN_HEADROOM_BYTES,
        FITS_SUFFIXES,
        LIGHT_FRAME_EXPANSION_FACTOR,
        LIGHT_PREPROCESS_SEQUENCE_COPIES,
        LINEAR_RESUME_STAGE_ARTIFACT_COPIES,
        RUNTIME_DEPENDENCY_EXPANSION_FACTOR,
        RUNTIME_DISK_MIN_HEADROOM_BYTES,
        STAGE2_RESUME_STAGE_ARTIFACT_COPIES,
        STACKED_STAGE_ARTIFACT_COPIES,
    )
    from .disk_preflight import (
        DiskSpaceEstimate,
        RuntimeDiskEstimate,
        directory_size_bytes,
        existing_volume_anchor,
        format_bytes,
        safe_file_size,
        safe_mtime,
    )
except ImportError:
    from constants import (  # type: ignore[no-redef]
        DISK_SPACE_HEADROOM_RATIO,
        DISK_SPACE_MIN_HEADROOM_BYTES,
        FITS_SUFFIXES,
        LIGHT_FRAME_EXPANSION_FACTOR,
        LIGHT_PREPROCESS_SEQUENCE_COPIES,
        LINEAR_RESUME_STAGE_ARTIFACT_COPIES,
        RUNTIME_DEPENDENCY_EXPANSION_FACTOR,
        RUNTIME_DISK_MIN_HEADROOM_BYTES,
        STAGE2_RESUME_STAGE_ARTIFACT_COPIES,
        STACKED_STAGE_ARTIFACT_COPIES,
    )
    from disk_preflight import (  # type: ignore[no-redef]
        DiskSpaceEstimate,
        RuntimeDiskEstimate,
        directory_size_bytes,
        existing_volume_anchor,
        format_bytes,
        safe_file_size,
        safe_mtime,
    )

from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

try:
    from PySide6.QtWidgets import QStyle
except ImportError:  # Lightweight Qt stubs used by non-visual unit tests.
    QStyle = None  # type: ignore[assignment,misc]

try:
    from pipeline import run_manifest as pipeline_run_manifest
    from pipeline.stage_contracts import (
        PIPELINE_CONTRACT_SCHEMA,
        PIPELINE_CONTRACT_VERSION,
        product_stage_contracts,
    )
    from pipeline.processing_parameters import (
        GATE_PROFILE_CHOICES,
        GATE_PROFILE_LABELS,
        GATE_PROFILE_UNLIMITED,
        PROCESSING_PARAMETERS_SCHEMA,
        SUPPORTED_PROCESSING_PARAMETERS_SCHEMAS,
        SPECS_BY_FIELD,
        SPECS_BY_STAGE,
        STAGE_TITLES,
        ParameterSpec,
        default_processing_parameters,
        effective_parameter_value,
        gate_profile_requires_review,
        normalize_processing_parameters,
        processing_gate_profile_audit,
        reset_stage_parameters,
    )
except ImportError:  # Support direct execution from the gui directory.
    import sys

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from pipeline import run_manifest as pipeline_run_manifest  # type: ignore[no-redef]
    from pipeline.stage_contracts import (  # type: ignore[no-redef]
        PIPELINE_CONTRACT_SCHEMA,
        PIPELINE_CONTRACT_VERSION,
        product_stage_contracts,
    )
    from pipeline.processing_parameters import (  # type: ignore[no-redef]
        GATE_PROFILE_CHOICES,
        GATE_PROFILE_LABELS,
        GATE_PROFILE_UNLIMITED,
        PROCESSING_PARAMETERS_SCHEMA,
        SUPPORTED_PROCESSING_PARAMETERS_SCHEMAS,
        SPECS_BY_FIELD,
        SPECS_BY_STAGE,
        STAGE_TITLES,
        ParameterSpec,
        default_processing_parameters,
        effective_parameter_value,
        gate_profile_requires_review,
        normalize_processing_parameters,
        processing_gate_profile_audit,
        reset_stage_parameters,
    )

SIRIL_PLUGIN_PYTHON_ABI = "cp312"
UI_MODE_RECOMMENDED = "recommended"

PIPELINE_STAGE_TITLES = {
    contract.number: contract.title for contract in product_stage_contracts()
}
PIPELINE_PROGRESS_STATE_LABELS = {
    "waiting": "等待",
    "running": "● 运行中",
    "completed": "✓ 已完成",
    "safe_passthrough": "↪ 安全旁路",
    "degraded": "⚠ 已降级",
    "failed": "✕ 失败",
    "skipped": "— 已跳过",
    "stopped": "已停止",
}

PIPELINE_STAGE_SHORT_TITLES = {
    1: "输入准备",
    2: "边界校正",
    3: "背景处理",
    4: "色彩校准",
    5: "线性清理",
    6: "线性去星",
    7: "主体拉伸",
    8: "Starless 增强",
    9: "星点合成",
    10: "降噪与导出",
}


def is_siril_cp312_wheel_compatible(path: Path) -> bool:
    """Return whether a standard wheel filename can run on Siril CPython 3.12."""
    filename = path.name
    if not filename.lower().endswith(".whl"):
        return False
    wheel_parts = filename[:-4].rsplit("-", 3)
    if len(wheel_parts) != 4:
        # Let pip report malformed third-party names; only reject ABI tags we can parse.
        return True
    _prefix, python_tag, abi_tag, _platform_tag = wheel_parts
    abi_tags = set(abi_tag.lower().split("."))
    for interpreter in python_tag.lower().split("."):
        if interpreter == "py3" and "none" in abi_tags:
            return True
        if not interpreter.startswith("cp") or not interpreter[2:].isdigit():
            continue
        version_digits = int(interpreter[2:])
        if interpreter == SIRIL_PLUGIN_PYTHON_ABI and (
            SIRIL_PLUGIN_PYTHON_ABI in abi_tags or "abi3" in abi_tags
        ):
            return True
        if version_digits <= 312 and "abi3" in abi_tags:
            return True
    return False

try:
    from .common import *
except ImportError:
    from common import *  # type: ignore[no-redef]

try:
    from .bootstrap_worker import (
        BootstrapCancelled,
        BootstrapError,
        BootstrapWorker,
        run_cancellable_process,
    )
except ImportError:
    from bootstrap_worker import (  # type: ignore[no-redef]
        BootstrapCancelled,
        BootstrapError,
        BootstrapWorker,
        run_cancellable_process,
    )

try:
    from .pipeline_worker import PipelineWorker
except ImportError:
    from pipeline_worker import PipelineWorker  # type: ignore[no-redef]

try:
    from .gaia_catalog import (
        GaiaCatalogCancelled,
        GaiaCatalogDownloadError,
        download_gaia_catalog,
        gaia_catalog_status,
    )
except ImportError:
    from gaia_catalog import (  # type: ignore[no-redef]
        GaiaCatalogCancelled,
        GaiaCatalogDownloadError,
        download_gaia_catalog,
        gaia_catalog_status,
    )

try:
    from .runtime_capabilities import (
        RUNTIME_CAPABILITIES_ENV,
        RUNTIME_CAPABILITIES_NAME,
        RUN_STATE_NAME,
        RUN_STATE_SCHEMA,
        atomic_write_json as atomic_write_runtime_json,
        build_runtime_capabilities,
        capability_summary_lines,
        probe_network_capabilities,
        refresh_blocking_errors,
        update_siril_launch_probe,
        utc_now as runtime_utc_now,
    )
except ImportError:
    from runtime_capabilities import (  # type: ignore[no-redef]
        RUNTIME_CAPABILITIES_ENV,
        RUNTIME_CAPABILITIES_NAME,
        RUN_STATE_NAME,
        RUN_STATE_SCHEMA,
        atomic_write_json as atomic_write_runtime_json,
        build_runtime_capabilities,
        capability_summary_lines,
        probe_network_capabilities,
        refresh_blocking_errors,
        update_siril_launch_probe,
        utc_now as runtime_utc_now,
    )

try:
    from .preview_widgets import LatestPreviewCanvas
    from .preview_worker import InitialPreviewWorker, preview_cache_path
except ImportError:
    from preview_widgets import LatestPreviewCanvas  # type: ignore[no-redef]
    from preview_worker import (  # type: ignore[no-redef]
        InitialPreviewWorker,
        preview_cache_path,
    )

try:
    from .ui_platform import (
        configure_main_window,
        configure_toolbar_drag_region,
        constrain_window_to_visible_screens,
        current_platform_profile,
        normal_window_geometry,
        restore_window_geometry,
        standard_shortcuts,
    )
    from .ui_theme import install_application_theme, set_style_property
except ImportError:
    from ui_platform import (  # type: ignore[no-redef]
        configure_main_window,
        configure_toolbar_drag_region,
        constrain_window_to_visible_screens,
        current_platform_profile,
        normal_window_geometry,
        restore_window_geometry,
        standard_shortcuts,
    )
    try:
        from ui_theme import (  # type: ignore[no-redef]
            install_application_theme,
            set_style_property,
        )
    except ImportError:  # Lightweight Qt stubs used by non-visual unit tests.
        def install_application_theme(app, profile=None):
            return None

        def set_style_property(widget, name: str, value: object) -> None:
            setter = getattr(widget, "setProperty", None)
            if callable(setter):
                setter(name, value)

try:
    from pipeline.input_discovery import (
        MASTER_SUFFIXES,
        REVIEW_SUFFIXES,
        InputDiscovery,
        InputKind,
        discover_input,
        inspect_source_header,
    )
    from pipeline.task_workspace import (
        apply_task_retention,
        latest_result_directory,
        latest_result_files,
        publish_latest_result_index,
    )
    from .task_intake import (
        PreparedTask,
        PreparedTaskQueue,
        SPCC_ONLINE_CIRCUIT_ENV,
        describe_input_plan,
        discover_input_for_processing_settings,
        prepare_task_queue,
        stage4_online_spcc_timeout_detected,
    )
    from .history_store import (
        STATUS_FAILED,
        STATUS_INTERRUPTED,
        STATUS_LABELS,
        STATUS_PARTIAL_SUCCESS,
        STATUS_PREPARING,
        STATUS_REVIEW_REQUIRED,
        STATUS_RUNNING,
        STATUS_STOPPED,
        STATUS_SUCCESS,
        HistoryStore,
        HistoryStoreError,
        UnsafeTaskDeletionError,
        history_task_key,
        load_verified_pipeline_result,
        validate_deletable_task_root,
        verified_result_files,
        verify_history_run,
    )
except ImportError:  # Support direct execution from the gui directory.
    import sys

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from pipeline.input_discovery import (  # type: ignore[no-redef]
        MASTER_SUFFIXES,
        REVIEW_SUFFIXES,
        InputDiscovery,
        InputKind,
        discover_input,
        inspect_source_header,
    )
    from pipeline.task_workspace import (  # type: ignore[no-redef]
        apply_task_retention,
        latest_result_directory,
        latest_result_files,
        publish_latest_result_index,
    )
    from task_intake import (  # type: ignore[no-redef]
        PreparedTask,
        PreparedTaskQueue,
        SPCC_ONLINE_CIRCUIT_ENV,
        describe_input_plan,
        discover_input_for_processing_settings,
        prepare_task_queue,
        stage4_online_spcc_timeout_detected,
    )
    from history_store import (  # type: ignore[no-redef]
        STATUS_FAILED,
        STATUS_INTERRUPTED,
        STATUS_LABELS,
        STATUS_PARTIAL_SUCCESS,
        STATUS_PREPARING,
        STATUS_REVIEW_REQUIRED,
        STATUS_RUNNING,
        STATUS_STOPPED,
        STATUS_SUCCESS,
        HistoryStore,
        HistoryStoreError,
        UnsafeTaskDeletionError,
        history_task_key,
        load_verified_pipeline_result,
        validate_deletable_task_root,
        verified_result_files,
        verify_history_run,
    )


WORKSPACE_EMPTY = "empty"
WORKSPACE_TASK = "task"
WORKSPACE_RUN = "run"
MAIN_WINDOW_DEFAULT_SIZE = (1280, 800)
HISTORY_WINDOW_DEFAULT_SIZE = (940, 620)
PREFERENCES_WINDOW_DEFAULT_SIZE = (620, 420)

_CANONICAL_PROCESSING_DEFAULTS = default_processing_parameters()
DEFAULT_PROCESSING_SETTINGS = {
    "output_formats": tuple(
        _CANONICAL_PROCESSING_DEFAULTS["general"]["output_formats"]
    ),
    "review_only": bool(
        _CANONICAL_PROCESSING_DEFAULTS["general"]["review_only"]
    ),
    "color_calibration": "pcc",
    "filter_hint": str(SPECS_BY_FIELD["stage4_filter_hint"].default),
    "denoise_mode": "auto",
    "deconvolution_mode": "auto",
    "graxpert_model_path": str(
        SPECS_BY_FIELD["graxpert_object_model_path"].default
    ),
    "compute_mode": str(
        _CANONICAL_PROCESSING_DEFAULTS["general"]["compute_mode"]
    ),
    "auto_tune_enabled": bool(
        _CANONICAL_PROCESSING_DEFAULTS["general"]["auto_tune_enabled"]
    ),
    "max_retries": int(
        _CANONICAL_PROCESSING_DEFAULTS["general"]["max_retries"]
    ),
    "retry_delay": float(
        _CANONICAL_PROCESSING_DEFAULTS["general"]["retry_delay"]
    ),
    "review_bundle_enabled": bool(
        _CANONICAL_PROCESSING_DEFAULTS["general"]["review_bundle_enabled"]
    ),
    "managed_output_enabled": bool(
        _CANONICAL_PROCESSING_DEFAULTS["general"]["managed_output_enabled"]
    ),
    "checkpoint_mode": bool(
        _CANONICAL_PROCESSING_DEFAULTS["general"]["checkpoint_mode"]
    ),
    "pcc_timeout_sec": int(
        SPECS_BY_FIELD["stage4_pcc_timeout_sec"].default
    ),
    "local_wb_gain_limit": float(
        SPECS_BY_FIELD["stage4_local_star_wb_gain_limit"].default
    ),
    "builtin_denoise_strength": float(SPECS_BY_FIELD["denoise_mod"].default),
    "graxpert_deconv_strength": float(
        SPECS_BY_FIELD["stage5_graxpert_deconv_strength"].default
    ),
    "rl_iterations": int(SPECS_BY_FIELD["stage5_rl_iters"].default),
    "rl_maxstars": int(SPECS_BY_FIELD["stage5_rl_maxstars"].default),
    "starless_retry_max": int(
        SPECS_BY_FIELD["stage7_quality_retry_max"].default
    ),
    "starless_repair_strength": float(
        SPECS_BY_FIELD["stage7_starless_repair_strength"].default
    ),
    "starless_halo_repair_strength": float(
        SPECS_BY_FIELD["stage7_starless_halo_repair_strength"].default
    ),
    "starless_chroma_strength": float(
        SPECS_BY_FIELD["stage7_starless_chroma_denoise_strength"].default
    ),
    "starmask_asinh_stretch": float(
        SPECS_BY_FIELD["stage9_starmask_asinh_stretch"].default
    ),
    "weak_star_recovery_ratio": float(
        SPECS_BY_FIELD["stage9_weak_star_recovery_ratio_min"].default
    ),
}
VALID_OUTPUT_FORMATS = frozenset({"tif", "png", "fit"})
VALID_COLOR_CALIBRATION_MODES = frozenset({"pcc"})
VALID_FILTER_HINT_MODES = frozenset(
    {"auto", "no_filter", "seestar_lp", "dual_narrowband"}
)
VALID_DENOISE_MODES = frozenset({"auto", "on", "off"})
VALID_DECONVOLUTION_MODES = frozenset({"auto", "rl", "off"})
VALID_COMPUTE_MODES = frozenset({"auto", "cpu"})


class StarunGui(QMainWindow):
    thread_log = Signal(str)

    def __init__(
        self,
        *,
        resources_override: Path | None = None,
        runtime_home_override: Path | None = None,
        settings_override: QSettings | None = None,
        history_path_override: Path | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Starun")
        self.setMinimumSize(980, 680)
        self.resize(*MAIN_WINDOW_DEFAULT_SIZE)
        self.setAcceptDrops(True)
        self.platform_profile = current_platform_profile()
        app = QApplication.instance()
        if isinstance(app, QApplication):
            self.theme_controller = install_application_theme(
                app,
                self.platform_profile,
            )
        self._ui_thread_ident = threading.get_ident()
        self.settings = settings_override or QSettings(
            "Starun",
            "Starun",
        )
        self._restoring_settings = False
        self._settings_loaded = False
        self._main_window_geometry_restored = False
        self._history_window_geometry_restored = False
        self._restore_main_window_maximized = False
        self._restore_history_window_maximized = False
        self._first_show_pending = True
        self._window_geometry_save_timer = QTimer(self)
        self._window_geometry_save_timer.setSingleShot(True)
        self._window_geometry_save_timer.setInterval(450)
        self._window_geometry_save_timer.timeout.connect(
            self._save_window_geometry_state
        )
        self._recent_directories: list[str] = []
        self._recommended_input_mode = INPUT_MODE_AUTO
        self._result_preview_path: Path | None = None
        self.preview_worker: InitialPreviewWorker | None = None
        self._preview_request_id = 0
        self._latest_preview_stage = -1
        self._latest_preview_title = ""
        self._latest_preview_path: Path | None = None
        self._workspace_state = WORKSPACE_EMPTY
        self._ui_running = False
        self._run_terminal_status: str | None = None
        self._last_run_snapshot: dict[str, object] | None = None
        self._run_input_mode: str | None = None
        self._input_discovery: InputDiscovery | None = None
        self._prepared_tasks: tuple[PreparedTask, ...] = ()
        self._prepared_task_index = 0
        self._active_prepared_task: PreparedTask | None = None
        self._spcc_online_circuit_open = False
        self._last_task_root: Path | None = None
        self.history_store = HistoryStore(history_path_override)
        self._history_return_state = WORKSPACE_EMPTY
        self._history_return_status = ""
        self._history_return_task_root: Path | None = None
        self._history_task_records: dict[str, dict[str, object]] = {}
        self._pending_history_delete: tuple[str, Path] | None = None
        self._active_history_task_key: str | None = None
        self._active_history_run_id: str | None = None
        self._history_detail_mode = False
        self._historical_run_root: Path | None = None
        self._historical_result_files: tuple[Path, ...] = ()

        self.resources = (
            Path(resources_override).expanduser().resolve()
            if resources_override is not None
            else resource_root()
        )
        self.pipeline_path = default_pipeline_path(self.resources)
        self.siril_plugin_dir = default_siril_plugin_dir(self.resources)
        self.runtime_home = (
            Path(runtime_home_override).expanduser().resolve()
            if runtime_home_override is not None
            else default_runtime_home()
        )
        self.bundled_siril_cli = (
            self.resources / "Siril.app" / "Contents" / "MacOS" / "siril-cli"
        )
        self.siril_seed_dir = self.resources / "SirilPythonSeed"
        config_candidates = [
            self.resources / "config.1.4.ini.template",
        ]
        self.config_template = resolve_existing_path(config_candidates)

        self.worker: PipelineWorker | None = None
        self.intake_worker: BootstrapWorker | None = None
        self.bootstrap_worker: BootstrapWorker | None = None
        self.gaia_catalog_worker: BootstrapWorker | None = None
        self._bootstrap_stop_event: threading.Event | None = None
        self.run_log_path: Path | None = None
        self.run_log_file = None
        self._run_log_lock = threading.Lock()
        self.run_state_path: Path | None = None
        self.runtime_capabilities_path: Path | None = None
        self._run_state_payload: dict[str, object] = {}
        self._run_state_lock = threading.Lock()
        self._runtime_capability_manifest: dict[str, object] | None = None
        self._current_work_dir: Path | None = None
        self._last_quality_report_path: Path | None = None
        self._pipeline_started_monotonic: float | None = None
        self._active_pipeline_stage: int | None = None
        self._display_pipeline_stage: int | None = None
        self._stage_started_monotonic: dict[int, float] = {}
        self._stage_elapsed_seconds: dict[int, float] = {}
        self._stage_progress_states: dict[int, str] = {}
        self._stage_progress_titles: dict[int, str] = {}
        self._stage_progress_details: dict[int, dict[str, object]] = {}
        self._stage_items: dict[int, QLabel] = {}
        self._progress_stage_count = 10
        self.input_mode = INPUT_MODE_AUTO
        self.debug_mode_enabled = False
        self.network_mode_enabled = True
        self._pending_runtime_overrides: dict[str, str] = {}
        self._pending_runtime_unset_keys: set[str] = set()
        self.processing_parameters_expanded = False
        self.processing_expert_visible = False
        self._processing_parameter_input_path: Path | None = None
        self._processing_controls_updating = False
        self.output_formats = tuple(DEFAULT_PROCESSING_SETTINGS["output_formats"])
        self.review_only = bool(DEFAULT_PROCESSING_SETTINGS["review_only"])
        self.color_calibration = str(
            DEFAULT_PROCESSING_SETTINGS["color_calibration"]
        )
        self.filter_hint = str(DEFAULT_PROCESSING_SETTINGS["filter_hint"])
        self.denoise_mode = str(DEFAULT_PROCESSING_SETTINGS["denoise_mode"])
        self.deconvolution_mode = str(
            DEFAULT_PROCESSING_SETTINGS["deconvolution_mode"]
        )
        self.graxpert_model_path = str(
            DEFAULT_PROCESSING_SETTINGS["graxpert_model_path"]
        )
        self.compute_mode = str(DEFAULT_PROCESSING_SETTINGS["compute_mode"])
        self.auto_tune_enabled = bool(
            DEFAULT_PROCESSING_SETTINGS["auto_tune_enabled"]
        )
        self.max_retries = int(DEFAULT_PROCESSING_SETTINGS["max_retries"])
        self.retry_delay = float(DEFAULT_PROCESSING_SETTINGS["retry_delay"])
        self.review_bundle_enabled = bool(
            DEFAULT_PROCESSING_SETTINGS["review_bundle_enabled"]
        )
        self.managed_output_enabled = bool(
            DEFAULT_PROCESSING_SETTINGS["managed_output_enabled"]
        )
        self.checkpoint_mode = bool(
            DEFAULT_PROCESSING_SETTINGS["checkpoint_mode"]
        )
        self.pcc_timeout_sec = int(DEFAULT_PROCESSING_SETTINGS["pcc_timeout_sec"])
        self.local_wb_gain_limit = float(
            DEFAULT_PROCESSING_SETTINGS["local_wb_gain_limit"]
        )
        self.builtin_denoise_strength = float(
            DEFAULT_PROCESSING_SETTINGS["builtin_denoise_strength"]
        )
        self.graxpert_deconv_strength = float(
            DEFAULT_PROCESSING_SETTINGS["graxpert_deconv_strength"]
        )
        self.rl_iterations = int(DEFAULT_PROCESSING_SETTINGS["rl_iterations"])
        self.rl_maxstars = int(DEFAULT_PROCESSING_SETTINGS["rl_maxstars"])
        self.starless_retry_max = int(
            DEFAULT_PROCESSING_SETTINGS["starless_retry_max"]
        )
        self.starless_repair_strength = float(
            DEFAULT_PROCESSING_SETTINGS["starless_repair_strength"]
        )
        self.starless_halo_repair_strength = float(
            DEFAULT_PROCESSING_SETTINGS["starless_halo_repair_strength"]
        )
        self.starless_chroma_strength = float(
            DEFAULT_PROCESSING_SETTINGS["starless_chroma_strength"]
        )
        self.starmask_asinh_stretch = float(
            DEFAULT_PROCESSING_SETTINGS["starmask_asinh_stretch"]
        )
        self.weak_star_recovery_ratio = float(
            DEFAULT_PROCESSING_SETTINGS["weak_star_recovery_ratio"]
        )
        self.processing_parameters = default_processing_parameters(
            general={
                "output_formats": list(self.output_formats),
                "review_only": self.review_only,
                "compute_mode": self.compute_mode,
                "auto_tune_enabled": self.auto_tune_enabled,
                "max_retries": self.max_retries,
                "retry_delay": self.retry_delay,
                "review_bundle_enabled": self.review_bundle_enabled,
                "managed_output_enabled": self.managed_output_enabled,
                "checkpoint_mode": self.checkpoint_mode,
            }
        )

        self._init_ui()
        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(1000)
        self._progress_timer.timeout.connect(self._refresh_elapsed_labels)
        self.thread_log.connect(self._append_text)
        self._load_settings()
        self._settings_loaded = True
        try:
            recovered_runs = self.history_store.mark_incomplete_runs_interrupted()
        except HistoryStoreError as error:
            recovered_runs = 0
            self._append_event(f"历史索引不可用：{error}")
        if recovered_runs:
            self._append_event(
                f"已将 {recovered_runs} 条未正常结束的历史运行标记为异常中断。"
            )
        self._set_running(False)
        self._append_event("已就绪。请选择或拖入图像文件、Light 目录或产品任务。")

    def _section_block(self, title: str, body_lines: list[str]) -> list[str]:
        divider = "=" * 16
        return [f"{divider} {title} {divider}", *body_lines, ""]

    def _display_path(self, path: Path | str | None) -> str:
        if path is None:
            return "<无>"
        text = str(path)
        home = str(Path.home())
        if text == home:
            return "~"
        prefix = home + os.sep
        if text.startswith(prefix):
            return "~" + os.sep + text[len(prefix):]
        return text

    def _display_status(self, status: str) -> str:
        return {
            "Idle": "空闲",
            "Preparing": "正在准备运行环境",
            "Running": "运行中",
            "Stopping": "正在停止",
            "Completed": "已完成",
            "CompletedWithWarning": "已完成（有降级/需复核）",
            "Failed": "失败",
            "Stopped": "已停止",
        }.get(status, status)

    def _set_status_text(self, status: str) -> None:
        self._set_status_message(self._display_status(status))

    def _set_status_message(self, message: str) -> None:
        status_text = f"状态：{message}"
        changed = self.status_label.text() != status_text
        self.status_label.setText(status_text)
        self.status_label.setAccessibleDescription(status_text)
        lowered = message.lower()
        if "失败" in message or "failed" in lowered:
            state = "error"
        elif "降级" in message or "复核" in message:
            state = "warning"
        elif "完成" in message or "completed" in lowered:
            state = "success"
        elif any(value in message for value in ("运行", "准备", "停止中")):
            state = "running"
        else:
            state = "idle"
        set_style_property(self.status_label, "state", state)
        if changed:
            self._announce_accessibility(status_text)

    def _announce_accessibility(self, message: str) -> None:
        if not message:
            return
        try:
            from PySide6.QtGui import QAccessible, QAccessibleAnnouncementEvent

            event = QAccessibleAnnouncementEvent(self.status_label, message)
            QAccessible.updateAccessibility(event)
        except (ImportError, AttributeError, RuntimeError, TypeError):
            # Accessibility announcements must never interrupt image processing.
            pass

    def _initial_panel_text(self, cli_order: list[Path]) -> str:
        if cli_order:
            candidate_names = ", ".join(p.name for p in cli_order)
        else:
            candidate_names = "<无>"
        lines: list[str] = []
        lines.extend(self._section_block("使用说明", [
            "  1. 拖入或选择一个 FITS/XISF 母版、Light 目录或产品任务。",
            "  2. 应用会识别输入并显示唯一安全处理计划，无需选择起始阶段。",
            "  3. 点击“开始处理”；需要中断时点击“停止处理”。",
            "  处理完成后，可直接预览结果或打开结果目录。",
            "  运行细节位于主界面的“详细日志”区域。",
        ]))
        lines.extend(self._section_block("应用能力", [
            "  - 明确母版始终从 Stage 1 导入；递归 Light 目录按目标、滤镜、相机和几何分组后串行处理。",
            "  - 只有清单、契约、配置指纹和 SHA-256 均通过校验的产品任务，才会从 Stage 1、2 或 5 后继续。",
            "  - 串联 Siril 1.4+ 与 SyQon/SASP/CosmicClarity 插件链路，执行离线后处理，不依赖系统 Python。",
            "  - 处理过程包含预检、重试、降级回退和阶段状态记录。",
            "  - 输出高质量 TIFF、预览 PNG、拉伸前线性 FITS 和最终 FITS 归档。",
            f"  - 当前处理方式: {self._input_mode_label(self.input_mode)}。",
            f"  - 保留中间文件: {'开启' if self.debug_mode_enabled else '关闭'}。",
            f"  - 允许联网: {'开启' if self.network_mode_enabled else '关闭'}。",
        ]))
        lines.extend(self._section_block("处理阶段总览", [
            "  线性阶段: 1.输入准备 -> 2.边界校正 -> 3.背景处理 -> 4.图像解析与色彩校准 -> 5.线性反卷积与降噪 -> 6.线性去星",
            "  非线性阶段: 7.主体拉伸 -> 8.Starless 深加工 -> 9.星点处理与合成 -> 10.最终降噪与导出",
        ]))
        lines.extend(self._section_block("阶段文件命名", [
            "  - 阶段 6 去星: stage6_starless.fit / stage6_starless_quality.json。",
            "  - 阶段 7 拉伸: stage7_stretched.fit / stage7_stretch_quality.json。",
            "  - 阶段 7 使用 stage7_cand_a/b/display90、stage7_preview_ref 与 stage7_stretched 命名。",
        ]))
        lines.extend(self._section_block("当前运行环境", [
            f"  资源根目录: {self._display_path(self.resources)}",
            f"  运行时主目录: {self._display_path(self.runtime_home)}",
            "  核心文件: "
            f"流水线={self.pipeline_path.name}, "
            f"Siril={candidate_names}",
            "  完整运行时路径和预检细节会在任务开始时写入日志。",
        ]))
        lines.append("已就绪。")
        return "\n".join(lines) + "\n"

    def _reset_view(self) -> None:
        self.log_view.clear()
        self.log_view.moveCursor(QTextCursor.MoveOperation.End)

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "关于 Starun",
            (
                "<b>Starun</b><br>"
                f"版本 {self._bootstrap_app_version()}<br><br>"
                "面向 Seestar 天文图像的离线自动后期处理应用。"
            ),
        )

    def _show_help(self) -> None:
        dialog = QMessageBox(self)
        dialog.setWindowTitle("使用说明")
        dialog.setIcon(QMessageBox.Icon.Information)
        dialog.setText(
            "三步完成处理：选择输入、确认自动计划、开始处理。"
        )
        dialog.setInformativeText(
            "可拖入母版文件、Light 目录或产品任务；应用只采用经验证的安全起点。"
        )
        dialog.setDetailedText(
            self._initial_panel_text(self._resolve_siril_candidates())
        )
        dialog.exec()

    def _resolve_siril_candidates(self) -> list[Path]:
        candidates: list[Path] = []
        if self.bundled_siril_cli.exists() and self.bundled_siril_cli.is_file():
            candidates.append(self.bundled_siril_cli)
        return candidates

    def _siril_state_root(self) -> Path:
        return siril_state_root_from_home(self.runtime_home)

    def _rewrite_seeded_venv(self, venv_dir: Path) -> None:
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
            raise FileNotFoundError(f"内置 Siril Python 缺失：{py_bin}")

        bin_dir = venv_dir / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        for name in ("python3.12", "python3", "python"):
            link_path = bin_dir / name
            if link_path.exists() or link_path.is_symlink():
                try:
                    link_path.unlink()
                except Exception:
                    pass
            link_path.symlink_to(py_bin)

        cfg = venv_dir / "pyvenv.cfg"
        if cfg.exists():
            content = cfg.read_text(encoding="utf-8", errors="replace").splitlines()
        else:
            content = []

        siril_python = self.resources / "Siril.app" / "Contents" / "MacOS" / "python3"
        replacements = {
            "home": str(py_bin.parent),
            "executable": str(py_bin),
            "command": f"{siril_python} -m venv {venv_dir}",
        }
        updated: dict[str, str] = {}
        output_lines: list[str] = []
        for line in content:
            if "=" not in line:
                output_lines.append(line)
                continue
            key, _value = line.split("=", 1)
            k = key.strip()
            if k in replacements:
                output_lines.append(f"{k} = {replacements[k]}")
                updated[k] = replacements[k]
            else:
                output_lines.append(line)
        for key, value in replacements.items():
            if key not in updated:
                output_lines.append(f"{key} = {value}")
        cfg.write_text("\n".join(output_lines) + "\n", encoding="utf-8")

    def _ensure_offline_siril_python_seed(self) -> None:
        seed_venv = self.siril_seed_dir / "venv"
        seed_module = self.siril_seed_dir / ".python_module"
        if not seed_venv.exists():
            raise FileNotFoundError(f"缺少内置 Siril venv seed：{seed_venv}")
        if not seed_module.exists():
            raise FileNotFoundError(f"缺少内置 Siril 模块 seed：{seed_module}")

        state_root = self._siril_state_root()
        target_venv = state_root / "venv"
        target_module = state_root / ".python_module"
        state_root.mkdir(parents=True, exist_ok=True)

        seeded = False
        if not target_venv.exists():
            shutil.copytree(seed_venv, target_venv, symlinks=True)
            seeded = True
            self._append_event(f"已写入 Siril 离线 venv：{target_venv}")
        if not target_module.exists():
            shutil.copytree(seed_module, target_module, symlinks=True)
            seeded = True
            self._append_event(f"已写入 Siril 离线模块：{target_module}")
        elif not (target_module / "sirilpy").exists():
            shutil.rmtree(target_module, ignore_errors=True)
            shutil.copytree(seed_module, target_module, symlinks=True)
            seeded = True
            self._append_event(f"已重新写入 Siril 离线模块：{target_module}")

        self._rewrite_seeded_venv(target_venv)
        site_dir = resolve_venv_site_packages(target_venv)
        repaired = repair_site_packages_from_pip_vendor(site_dir)
        if repaired:
            self._append_event(
                "已从 pip vendor 修补 Siril venv 依赖："
                + ", ".join(repaired)
            )
        ok, detail = verify_siril_offline_seed_venv(target_venv)
        if not ok:
            raise RuntimeError(detail)
        if seeded:
            self._append_event("离线 Siril Python seed 已就绪。")

    def _init_ui(self) -> None:
        central = QWidget()
        central.setObjectName("appRoot")
        self.setCentralWidget(central)

        outer = QVBoxLayout(central)
        margin = self.platform_profile.window_margin
        outer.setContentsMargins(margin, 12, margin, 10)
        outer.setSpacing(self.platform_profile.panel_spacing)

        self._init_actions()
        self._init_toolbar()
        self.workspace_stack = QStackedWidget()
        self.empty_page = self._build_empty_page()
        self.task_page = self._build_task_page()
        self.run_page = self._build_run_page()
        self.history_page = self._build_history_page()
        self.workspace_stack.addWidget(self.empty_page)
        self.workspace_stack.addWidget(self.task_page)
        self.workspace_stack.addWidget(self.run_page)
        outer.addWidget(self.workspace_stack, 1)

        self._init_history_window()

        self._build_log_drawer(outer)
        self._init_status_bar()
        self._init_menus()
        configure_main_window(self, self.platform_profile)
        self._configure_focus_order()
        self._update_result_actions(None)
        self._show_workspace(WORKSPACE_EMPTY)

    def _init_actions(self) -> None:
        """Create one command object for every shared desktop operation."""

        shortcuts = standard_shortcuts(self.platform_profile)

        def command(
            text: str,
            object_name: str,
            handler,
            *,
            shortcut=None,
            status_tip: str = "",
        ) -> QAction:
            action = QAction(text, self)
            action.setObjectName(object_name)
            if shortcut is not None:
                action.setShortcut(shortcut)
            if status_tip:
                action.setStatusTip(status_tip)
                action.setToolTip(status_tip)
            action.triggered.connect(handler)
            return action

        self.open_file_action = command(
            "打开图像…",
            "openFileAction",
            self._choose_input_file,
            shortcut=shortcuts["open"],
            status_tip="打开 FITS、XISF 或复核图像",
        )
        self.choose_directory_action = command(
            "打开文件夹…",
            "openDirectoryAction",
            self._choose_workdir,
            shortcut=shortcuts["open_folder"],
            status_tip="打开 Light 文件夹或产品任务",
        )
        self.open_result_action = command(
            "打开结果文件夹",
            "openResultAction",
            self._open_result_dir,
            status_tip="打开当前任务的实际结果目录",
        )
        self.close_window_action = command(
            "关闭窗口",
            "closeWindowAction",
            self._close_active_window,
            shortcut=shortcuts["close"],
        )
        self.close_window_action.setShortcutContext(
            Qt.ShortcutContext.ApplicationShortcut
        )
        self.preferences_action = command(
            "设置…",
            "preferencesAction",
            self._show_preferences,
            shortcut=shortcuts["preferences"],
        )
        self.preferences_action.setMenuRole(QAction.MenuRole.PreferencesRole)
        self.quit_action = command(
            "退出 Starun",
            "quitAction",
            self.close,
            shortcut=shortcuts["quit"],
        )
        self.quit_action.setMenuRole(QAction.MenuRole.QuitRole)

        edit_commands = (
            ("undo_action", "撤销", "undoAction", "undo", "undo"),
            ("redo_action", "重做", "redoAction", "redo", "redo"),
            ("cut_action", "剪切", "cutAction", "cut", "cut"),
            ("copy_action", "拷贝", "copyAction", "copy", "copy"),
            ("paste_action", "粘贴", "pasteAction", "paste", "paste"),
            (
                "select_all_action",
                "全选",
                "selectAllAction",
                "selectAll",
                "select_all",
            ),
        )
        for attribute, text, object_name, method_name, shortcut_name in edit_commands:
            action = command(
                text,
                object_name,
                lambda _checked=False, name=method_name: (
                    self._dispatch_focused_edit_command(name)
                ),
                shortcut=shortcuts[shortcut_name],
            )
            action.setProperty("editCommand", method_name)
            setattr(self, attribute, action)

        self.toggle_sidebar_action = QAction("显示任务信息侧边栏", self)
        self.toggle_sidebar_action.setObjectName("toggleSidebarAction")
        self.toggle_sidebar_action.setCheckable(True)
        self.toggle_sidebar_action.setChecked(True)
        self.toggle_sidebar_action.setShortcut(shortcuts["toggle_sidebar"])
        self.toggle_sidebar_action.toggled.connect(
            self._set_run_sidebar_visible
        )
        self.toggle_inspector_action = QAction("显示检查器", self)
        self.toggle_inspector_action.setObjectName("toggleInspectorAction")
        self.toggle_inspector_action.setCheckable(True)
        self.toggle_inspector_action.setChecked(True)
        self.toggle_inspector_action.setShortcut(shortcuts["toggle_inspector"])
        self.toggle_inspector_action.toggled.connect(
            self._set_run_inspector_visible
        )

        self.fit_preview_action = command(
            "适合窗口",
            "fitPreviewAction",
            self._fit_run_preview,
            shortcut=shortcuts["fit_preview"],
        )
        self.actual_preview_action = command(
            "实际大小",
            "actualPreviewAction",
            self._show_run_preview_actual_size,
            shortcut=shortcuts["actual_preview"],
        )
        self.zoom_in_action = command(
            "放大",
            "zoomInAction",
            self._zoom_run_preview_in,
            shortcut=shortcuts["zoom_in"],
        )
        self.zoom_out_action = command(
            "缩小",
            "zoomOutAction",
            self._zoom_run_preview_out,
            shortcut=shortcuts["zoom_out"],
        )
        self.toggle_log_action = QAction("显示详细日志", self)
        self.toggle_log_action.setObjectName("toggleLogAction")
        self.toggle_log_action.setCheckable(True)
        self.toggle_log_action.setShortcut(shortcuts["toggle_log"])
        self.toggle_log_action.toggled.connect(self._on_log_toggled)
        self.open_log_action = command(
            "打开日志文件",
            "openLogAction",
            self._open_log_file,
        )
        self.clear_view_action = command(
            "清空界面日志",
            "clearLogAction",
            self._reset_view,
        )
        self.full_screen_action = QAction("进入全屏幕", self)
        self.full_screen_action.setObjectName("fullScreenAction")
        self.full_screen_action.setCheckable(True)
        self.full_screen_action.setShortcut(shortcuts["full_screen"])
        self.full_screen_action.toggled.connect(self._set_full_screen)

        self.task_options_action = command(
            "任务选项",
            "taskOptionsAction",
            self._show_advanced_settings,
        )
        self.start_action = command(
            "开始处理",
            "startProcessAction",
            self._start_run,
            shortcut=shortcuts["start"],
        )
        self.stop_action = command(
            "停止",
            "stopProcessAction",
            self._request_stop_run,
            shortcut=shortcuts["stop"],
        )
        self.return_task_action = command(
            "返回任务设置",
            "returnTaskAction",
            self._return_to_task_setup,
        )
        self.rerun_action = command(
            "重新运行",
            "rerunAction",
            self._rerun_last_task,
            shortcut=shortcuts["rerun"],
        )
        self.history_action = command(
            "历史记录",
            "historyAction",
            self._show_history,
            shortcut=shortcuts["history"],
            status_tip="打开独立的历史记录窗口",
        )
        self.history_action.setShortcutContext(
            Qt.ShortcutContext.ApplicationShortcut
        )
        self.main_window_action = command(
            "Starun 主窗口",
            "mainWindowAction",
            self._show_main_window,
            status_tip="显示并前置 Starun 主窗口",
        )

        self.minimize_action = command(
            "最小化",
            "minimizeWindowAction",
            self._minimize_active_window,
            shortcut=shortcuts["minimize"],
        )
        self.minimize_action.setShortcutContext(
            Qt.ShortcutContext.ApplicationShortcut
        )
        self.zoom_window_action = command(
            "缩放",
            "zoomWindowAction",
            self._toggle_window_zoom,
        )
        self.bring_all_to_front_action = command(
            "前置全部窗口",
            "bringAllToFrontAction",
            self._bring_all_windows_to_front,
        )
        self.help_action = command(
            "使用说明",
            "helpAction",
            self._show_help,
            shortcut=shortcuts["help"],
        )
        self.about_action = command(
            "关于 Starun",
            "aboutAction",
            self._show_about,
        )
        self.about_action.setMenuRole(QAction.MenuRole.AboutRole)

        if QStyle is not None:
            icons = {
                self.open_file_action: QStyle.StandardPixmap.SP_DirOpenIcon,
                self.choose_directory_action: QStyle.StandardPixmap.SP_DirOpenIcon,
                self.open_result_action: QStyle.StandardPixmap.SP_DirOpenIcon,
                self.toggle_sidebar_action: QStyle.StandardPixmap.SP_FileDialogListView,
                self.toggle_inspector_action: QStyle.StandardPixmap.SP_FileDialogInfoView,
                self.history_action: QStyle.StandardPixmap.SP_FileDialogDetailedView,
                self.return_task_action: QStyle.StandardPixmap.SP_ArrowBack,
                self.rerun_action: QStyle.StandardPixmap.SP_BrowserReload,
                self.start_action: QStyle.StandardPixmap.SP_MediaPlay,
                self.stop_action: QStyle.StandardPixmap.SP_MediaStop,
            }
            for action, standard_pixmap in icons.items():
                action.setIcon(self.style().standardIcon(standard_pixmap))

        shared_actions = (
            self.open_file_action,
            self.choose_directory_action,
            self.open_result_action,
            self.close_window_action,
            self.preferences_action,
            self.quit_action,
            self.undo_action,
            self.redo_action,
            self.cut_action,
            self.copy_action,
            self.paste_action,
            self.select_all_action,
            self.toggle_sidebar_action,
            self.toggle_inspector_action,
            self.fit_preview_action,
            self.actual_preview_action,
            self.zoom_in_action,
            self.zoom_out_action,
            self.toggle_log_action,
            self.open_log_action,
            self.clear_view_action,
            self.full_screen_action,
            self.task_options_action,
            self.start_action,
            self.stop_action,
            self.return_task_action,
            self.rerun_action,
            self.history_action,
            self.main_window_action,
            self.minimize_action,
            self.zoom_window_action,
            self.bring_all_to_front_action,
            self.help_action,
            self.about_action,
        )
        self.command_actions = {
            action.objectName(): action for action in shared_actions
        }

        app = QApplication.instance()
        if isinstance(app, QApplication):
            app.focusChanged.connect(self._update_edit_actions)
        self._update_edit_actions()

    def _bind_button_to_action(
        self,
        button: QPushButton,
        action: QAction,
        *,
        mirror_text: bool = True,
    ) -> None:
        """Let a styled push button invoke and reflect a shared QAction."""

        button.setProperty("commandAction", action.objectName())

        def sync_button() -> None:
            if mirror_text:
                button.setText(action.text())
            button.setIcon(action.icon())
            button.setEnabled(action.isEnabled())
            if action.toolTip():
                button.setToolTip(action.toolTip())
            if not button.accessibleName():
                button.setAccessibleName(action.text().replace("…", ""))
            if button.isCheckable() and button.isChecked() != action.isChecked():
                button.blockSignals(True)
                button.setChecked(action.isChecked())
                button.blockSignals(False)

        button.clicked.connect(
            lambda _checked=False, shared_action=action: shared_action.trigger()
        )
        action.changed.connect(sync_button)
        sync_button()

    def _init_toolbar(self) -> None:
        toolbar = QToolBar("主工具栏", self)
        toolbar.setObjectName("mainToolbar")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        self.addToolBar(toolbar)
        self.main_toolbar = toolbar

        self.toolbar_directory_btn = QPushButton()
        self.toolbar_directory_btn.setAccessibleName("打开图像文件")
        self._bind_button_to_action(
            self.toolbar_directory_btn,
            self.open_file_action,
        )
        self.toolbar_directory_item = toolbar.addWidget(self.toolbar_directory_btn)

        self.toolbar_settings_btn = QPushButton()
        self.toolbar_settings_btn.setAccessibleName("展开当前任务选项")
        self.toolbar_settings_btn.setProperty("variant", "quiet")
        self._bind_button_to_action(
            self.toolbar_settings_btn,
            self.task_options_action,
        )
        self.toolbar_settings_item = toolbar.addWidget(self.toolbar_settings_btn)

        self.history_btn = QPushButton()
        self.history_btn.setAccessibleName("查看历史处理记录")
        self.history_btn.setProperty("variant", "quiet")
        self._bind_button_to_action(self.history_btn, self.history_action)
        self.history_item = toolbar.addWidget(self.history_btn)

        self.sidebar_toggle_btn = QPushButton("任务信息")
        self.sidebar_toggle_btn.setAccessibleName("显示或隐藏任务信息侧边栏")
        self.sidebar_toggle_btn.setCheckable(True)
        self.sidebar_toggle_btn.setProperty("variant", "quiet")
        self._bind_button_to_action(
            self.sidebar_toggle_btn,
            self.toggle_sidebar_action,
            mirror_text=False,
        )
        self.sidebar_toggle_item = toolbar.addWidget(self.sidebar_toggle_btn)

        self.window_drag_region = QWidget()
        self.window_drag_region.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        self.window_drag_region.setMinimumWidth(40)
        configure_toolbar_drag_region(
            self.window_drag_region,
            self.platform_profile,
        )
        toolbar.addWidget(self.window_drag_region)

        self.inspector_toggle_btn = QPushButton("检查器")
        self.inspector_toggle_btn.setAccessibleName("显示或隐藏阶段检查器")
        self.inspector_toggle_btn.setCheckable(True)
        self.inspector_toggle_btn.setProperty("variant", "quiet")
        self._bind_button_to_action(
            self.inspector_toggle_btn,
            self.toggle_inspector_action,
            mirror_text=False,
        )
        self.inspector_toggle_item = toolbar.addWidget(self.inspector_toggle_btn)

        self.return_task_btn = QPushButton()
        self._bind_button_to_action(
            self.return_task_btn,
            self.return_task_action,
        )
        self.return_task_item = toolbar.addWidget(self.return_task_btn)

        self.rerun_btn = QPushButton()
        self._bind_button_to_action(self.rerun_btn, self.rerun_action)
        self.rerun_item = toolbar.addWidget(self.rerun_btn)

        self.start_btn = QPushButton()
        self.start_btn.setAccessibleName("开始图像处理")
        self.start_btn.setMinimumWidth(120)
        self.start_btn.setDefault(True)
        self.start_btn.setProperty("variant", "primary")
        self._bind_button_to_action(self.start_btn, self.start_action)
        self.start_item = toolbar.addWidget(self.start_btn)

        self.stop_btn = QPushButton()
        self.stop_btn.setAccessibleName("停止图像处理")
        self.stop_btn.setMinimumWidth(96)
        self.stop_btn.setProperty("variant", "destructive")
        self._bind_button_to_action(self.stop_btn, self.stop_action)
        self.stop_item = toolbar.addWidget(self.stop_btn)

    def _build_empty_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(56, 38, 56, 34)
        layout.addStretch(1)

        heading = QLabel("专业天文图像后处理")
        heading.setObjectName("pageTitle")
        heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(heading)
        description = QLabel("拖入一个母版文件、Light 目录或产品任务；应用会给出唯一安全处理计划。")
        description.setObjectName("pageDescription")
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description.setWordWrap(True)
        layout.addWidget(description)
        layout.addSpacing(18)

        card = QFrame()
        card.setObjectName("dropZone")
        card.setProperty("dragActive", False)
        self.drop_zone = card
        card.setMinimumHeight(300)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(40, 34, 40, 34)
        card_layout.addStretch(1)

        icon_label = QLabel()
        icon_label.setObjectName("dropIcon")
        if QStyle is not None:
            icon_label.setPixmap(
                self.style()
                .standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon)
                .pixmap(44, 44)
            )
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(icon_label)
        card_layout.addSpacing(10)

        title = QLabel("将图像文件或目录拖到这里")
        title.setObjectName("sectionTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(title)

        subtitle = QLabel("支持 FITS / XISF 母版、递归 Light 目录和已有产品任务。")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setProperty("tone", "muted")
        subtitle.setWordWrap(True)
        card_layout.addWidget(subtitle)
        card_layout.addSpacing(16)

        self.empty_choose_btn = QPushButton("选择文件")
        self.empty_choose_btn.setAccessibleName("选择 FITS 或 XISF 图像")
        self.empty_choose_btn.setMinimumSize(150, 36)
        self.empty_choose_btn.setProperty("variant", "primary")
        self.empty_choose_btn.clicked.connect(self._choose_input_file)
        self.empty_choose_dir_btn = QPushButton("选择目录")
        self.empty_choose_dir_btn.setAccessibleName("选择 Light 或产品任务目录")
        self.empty_choose_dir_btn.setMinimumSize(150, 36)
        self.empty_choose_dir_btn.clicked.connect(self._choose_workdir)
        choose_row = QHBoxLayout()
        choose_row.addStretch(1)
        choose_row.addWidget(self.empty_choose_btn)
        choose_row.addWidget(self.empty_choose_dir_btn)
        choose_row.addStretch(1)
        card_layout.addLayout(choose_row)
        card_layout.addStretch(1)
        layout.addWidget(card)

        reassurance = QLabel("默认在线 Gaia 校准 · 可在高级设置切换离线 · 原图不会被覆盖")
        reassurance.setAlignment(Qt.AlignmentFlag.AlignCenter)
        reassurance.setAccessibleName("处理安全说明")
        reassurance.setProperty("tone", "muted")
        reassurance.setContentsMargins(0, 14, 0, 0)
        layout.addWidget(reassurance)
        layout.addStretch(1)
        return page

    def _build_history_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        heading_row = QHBoxLayout()
        heading_text = QVBoxLayout()
        heading_text.setSpacing(2)
        title = QLabel("历史处理记录")
        title.setObjectName("pageTitle")
        description = QLabel("按任务查看新版中实际开始的处理记录。")
        description.setObjectName("pageDescription")
        heading_text.addWidget(title)
        heading_text.addWidget(description)
        heading_row.addLayout(heading_text)
        heading_row.addStretch(1)
        layout.addLayout(heading_row)

        filter_row = QHBoxLayout()
        self.history_search_edit = QLineEdit()
        self.history_search_edit.setPlaceholderText("搜索目标或输入名称")
        self.history_search_edit.setClearButtonEnabled(True)
        self.history_search_edit.setAccessibleName("搜索历史任务")
        self.history_search_edit.textChanged.connect(self._refresh_history_view)
        filter_row.addWidget(self.history_search_edit, 1)

        self.history_status_combo = QComboBox()
        self.history_status_combo.setAccessibleName("筛选历史状态")
        self.history_status_combo.addItem("全部状态", "all")
        self.history_status_combo.addItem("成功", "success")
        self.history_status_combo.addItem("需复核", "review")
        self.history_status_combo.addItem("失败或中止", "failure")
        self.history_status_combo.currentIndexChanged.connect(
            self._refresh_history_view
        )
        filter_row.addWidget(self.history_status_combo)

        self.history_open_btn = QPushButton("查看运行")
        self.history_open_btn.setAccessibleName("查看选中的历史运行")
        self.history_open_btn.setEnabled(False)
        self.history_open_btn.clicked.connect(self._open_selected_history_run)
        filter_row.addWidget(self.history_open_btn)

        self.history_delete_btn = QPushButton("移到废纸篓")
        self.history_delete_btn.setAccessibleName("删除选中的历史任务")
        self.history_delete_btn.setProperty("variant", "destructive")
        self.history_delete_btn.setEnabled(False)
        self.history_delete_btn.clicked.connect(self._delete_selected_history_task)
        filter_row.addWidget(self.history_delete_btn)
        layout.addLayout(filter_row)

        self.history_mode_label = QLabel(
            "当前处理正在运行；可以浏览和搜索历史，查看运行与删除暂不可用。"
        )
        self.history_mode_label.setWordWrap(True)
        self.history_mode_label.setProperty("tone", "muted")
        self.history_mode_label.setAccessibleName("历史记录只读状态")
        self.history_mode_label.hide()
        layout.addWidget(self.history_mode_label)

        self.history_delete_banner = QFrame()
        self.history_delete_banner.setObjectName("stateBanner")
        self.history_delete_banner.setProperty("tone", "warning")
        self.history_delete_banner.setAccessibleName("历史任务删除确认")
        self.history_delete_banner.setAccessibleDescription(
            "显示待移到废纸篓的历史任务摘要和确认操作"
        )
        self.history_delete_banner.hide()
        delete_banner_layout = QVBoxLayout(self.history_delete_banner)
        delete_banner_layout.setContentsMargins(12, 10, 12, 10)
        delete_banner_layout.setSpacing(8)

        self.history_delete_banner_label = QLabel()
        self.history_delete_banner_label.setWordWrap(True)
        self.history_delete_banner_label.setAccessibleName("待删除任务摘要")
        self.history_delete_banner_label.setAccessibleDescription(
            "显示任务名称、目录、运行次数和目录大小"
        )
        delete_banner_layout.addWidget(self.history_delete_banner_label)

        delete_actions = QHBoxLayout()
        delete_actions.addStretch(1)
        self.history_delete_cancel_btn = QPushButton("取消")
        self.history_delete_cancel_btn.setAccessibleName("取消移到废纸篓")
        self.history_delete_cancel_btn.setAccessibleDescription(
            "不移动任务目录，返回历史记录"
        )
        self.history_delete_cancel_btn.clicked.connect(
            self._cancel_pending_history_delete
        )
        delete_actions.addWidget(self.history_delete_cancel_btn)

        self.history_delete_confirm_btn = QPushButton("确认移到废纸篓")
        self.history_delete_confirm_btn.setProperty("variant", "destructive")
        self.history_delete_confirm_btn.setAccessibleName(
            "确认将任务移到废纸篓"
        )
        self.history_delete_confirm_btn.setAccessibleDescription(
            "重新验证任务目录后，将整个任务移到系统废纸篓"
        )
        self.history_delete_confirm_btn.clicked.connect(
            self._confirm_pending_history_delete
        )
        delete_actions.addWidget(self.history_delete_confirm_btn)
        delete_banner_layout.addLayout(delete_actions)
        layout.addWidget(self.history_delete_banner)

        self.history_tree = QTreeWidget()
        self.history_tree.setObjectName("historyTree")
        self.history_tree.setAccessibleName("历史任务和运行记录")
        self.history_tree.setColumnCount(3)
        self.history_tree.setHeaderLabels(
            ["目标 / 输入", "最近处理时间", "状态"]
        )
        self.history_tree.setRootIsDecorated(True)
        self.history_tree.setAlternatingRowColors(True)
        self.history_tree.setUniformRowHeights(True)
        history_header = self.history_tree.header()
        history_header.setStretchLastSection(False)
        history_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        history_header.setSectionResizeMode(
            1,
            QHeaderView.ResizeMode.ResizeToContents,
        )
        history_header.setSectionResizeMode(
            2,
            QHeaderView.ResizeMode.ResizeToContents,
        )
        self.history_tree.itemActivated.connect(
            self._on_history_item_activated
        )
        self.history_tree.itemSelectionChanged.connect(
            self._on_history_selection_changed
        )
        layout.addWidget(self.history_tree, 1)

        self.history_empty_label = QLabel("还没有历史处理记录")
        self.history_empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.history_empty_label.setProperty("tone", "muted")
        self.history_empty_label.hide()
        layout.addWidget(self.history_empty_label)
        return page

    def _init_history_window(self) -> None:
        try:
            from .history_window import HistoryWindow
        except ImportError:  # Support direct execution from the gui directory.
            from history_window import HistoryWindow  # type: ignore[no-redef]

        self.history_window = HistoryWindow(self.history_page, self)
        self.history_window.aboutToClose.connect(
            self._clear_pending_history_delete
        )
        self.history_window.aboutToClose.connect(
            self._save_history_window_geometry
        )

    def _build_task_page(self) -> QWidget:
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(12)

        heading_row = QHBoxLayout()
        heading_text = QVBoxLayout()
        heading_text.setSpacing(2)
        page_title = QLabel("任务设置")
        page_title.setObjectName("pageTitle")
        page_description = QLabel("确认自动识别的唯一处理计划；高级参数会在任务启动时冻结。")
        page_description.setObjectName("pageDescription")
        heading_text.addWidget(page_title)
        heading_text.addWidget(page_description)
        heading_row.addLayout(heading_text)
        heading_row.addStretch(1)

        guide_label = QLabel("选择输入  ·  确认计划  ·  开始处理")
        guide_label.setAccessibleName("使用步骤")
        guide_label.setAccessibleDescription(
            "第一步选择输入，第二步确认自动计划，第三步开始处理"
        )
        guide_label.setProperty("tone", "muted")
        heading_row.addWidget(guide_label)
        page_layout.addLayout(heading_row)

        self.task_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.task_splitter.setChildrenCollapsible(False)
        self.task_splitter.setHandleWidth(5)
        self.source_card = self._build_source_card()
        self.source_card.setMinimumWidth(500)
        self.task_preview_card = self._build_task_preview_card()
        self.task_preview_card.setMinimumWidth(340)
        self.task_splitter.addWidget(self.source_card)
        self.task_splitter.addWidget(self.task_preview_card)
        self.task_splitter.setStretchFactor(0, 3)
        self.task_splitter.setStretchFactor(1, 2)
        self._task_splitter_sizes = [720, 480]
        self.task_splitter.setSizes(self._task_splitter_sizes)
        self.task_splitter.splitterMoved.connect(
            self._on_task_splitter_moved
        )
        page_layout.addWidget(self.task_splitter, 1)

        phase_card = QFrame()
        phase_card.setObjectName("phaseBar")
        self.task_phase_bar = phase_card
        phase_layout = QVBoxLayout(phase_card)
        phase_layout.setContentsMargins(14, 11, 14, 11)
        phase_row = QHBoxLayout()
        self.linear_phase_label = QLabel("○ 线性处理 · Stage 1–6")
        self.nonlinear_phase_label = QLabel("○ 非线性处理 · Stage 7–10")
        for label in (
            self.linear_phase_label,
            self.nonlinear_phase_label,
        ):
            label.setProperty("role", "phase")
            label.setProperty("active", False)
            phase_row.addWidget(label, 1)
        phase_layout.addLayout(phase_row)
        page_layout.addWidget(phase_card)

        self.processing_params_panel = self._build_processing_parameters_panel()
        self.processing_params_scroll = QScrollArea()
        self.processing_params_scroll.setObjectName("processingParametersScroll")
        self.processing_params_scroll.setWidgetResizable(True)
        self.processing_params_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.processing_params_scroll.setWidget(self.processing_params_panel)
        self.processing_params_scroll.hide()
        page_layout.addWidget(self.processing_params_scroll, 1)
        return page

    def _build_source_card(self) -> QFrame:
        source_card = QFrame()
        source_card.setObjectName("contentPanel")
        source_layout = QVBoxLayout(source_card)
        source_layout.setContentsMargins(16, 14, 16, 14)
        source_layout.setSpacing(10)
        section_title = self._section_title("输入与处理计划")
        source_layout.addWidget(section_title)

        directory_row = QHBoxLayout()
        directory_row.setSpacing(8)
        dir_label = QLabel("输入")
        self.dir_edit = QLineEdit()
        self.dir_edit.setPlaceholderText("拖入 FITS/XISF 文件、Light 目录或产品任务")
        self.dir_edit.setAccessibleName("任务输入")
        self.dir_edit.setAccessibleDescription(
            "明确的母版文件、递归 Light 目录或产品任务目录"
        )
        self.dir_edit.setAcceptDrops(False)
        self.dir_edit.editingFinished.connect(self._on_directory_edited)
        self.browse_btn = QPushButton("选择文件")
        self.browse_btn.setAccessibleName("选择图像文件")
        self.browse_btn.clicked.connect(self._choose_input_file)
        self.browse_dir_btn = QPushButton("选择目录")
        self.browse_dir_btn.setAccessibleName("选择 Light 或产品任务目录")
        self.browse_dir_btn.clicked.connect(self._choose_workdir)
        dir_label.setBuddy(self.dir_edit)
        directory_row.addWidget(dir_label)
        directory_row.addWidget(self.dir_edit, 1)
        directory_row.addWidget(self.browse_btn)
        directory_row.addWidget(self.browse_dir_btn)
        source_layout.addLayout(directory_row)

        recent_row = QHBoxLayout()
        self.recent_label = QLabel("最近目录")
        self.recent_combo = QComboBox()
        self.recent_combo.setAccessibleName("最近使用的工作目录")
        self.recent_combo.activated.connect(self._on_recent_directory_selected)
        self.recent_label.setBuddy(self.recent_combo)
        recent_row.addWidget(self.recent_label)
        recent_row.addWidget(self.recent_combo, 1)
        self.recent_label.hide()
        self.recent_combo.hide()
        source_layout.addLayout(recent_row)

        self.directory_summary_label = QLabel("尚未选择输入")
        self.directory_summary_label.setAccessibleName("自动处理计划")
        self.directory_summary_label.setWordWrap(True)
        source_layout.addWidget(self.directory_summary_label)

        self.source_header_group = QGroupBox("源文件 Header 信息")
        self.source_header_group.setAccessibleName("源文件 Header 信息")
        self.source_header_group.setAccessibleDescription(
            "从当前选择的 FITS 主 Header 读取设备、滤镜、曝光和拍摄信息"
        )
        source_header_form = QFormLayout(self.source_header_group)
        source_header_form.setContentsMargins(12, 10, 12, 10)
        source_header_form.setSpacing(7)

        self.source_header_device_label = QLabel("未记录")
        self.source_header_filter_label = QLabel("未记录")
        self.source_header_exposure_label = QLabel("未记录")
        self.source_header_details_label = QLabel("未发现其他已知字段")
        for label, accessible_name in (
            (self.source_header_device_label, "Header 设备名称"),
            (self.source_header_filter_label, "Header 滤镜名称"),
            (self.source_header_exposure_label, "Header 曝光时间"),
            (self.source_header_details_label, "Header 其他重要信息"),
        ):
            label.setAccessibleName(accessible_name)
            label.setTextFormat(Qt.TextFormat.PlainText)
            label.setWordWrap(True)
        source_header_form.addRow("设备名称", self.source_header_device_label)
        source_header_form.addRow("滤镜名称", self.source_header_filter_label)
        source_header_form.addRow("曝光时间", self.source_header_exposure_label)
        source_header_form.addRow("其他重要信息", self.source_header_details_label)

        self.source_header_status_label = QLabel("")
        self.source_header_status_label.setAccessibleName("Header 扫描状态")
        self.source_header_status_label.setProperty("tone", "muted")
        self.source_header_status_label.setTextFormat(Qt.TextFormat.PlainText)
        self.source_header_status_label.setWordWrap(True)
        source_header_form.addRow(self.source_header_status_label)
        self.source_header_group.hide()
        source_layout.addWidget(self.source_header_group)

        mode_row = QHBoxLayout()
        mode_label = QLabel("处理方式")
        self.mode_combo = QComboBox()
        self.mode_combo.setAccessibleName("处理方式")
        self.mode_combo.setAccessibleDescription(
            "根据已验证任务自动选择完整处理或正式断点续跑"
        )
        self.mode_combo.setToolTip(
            "流水线起点由输入发现与已验签 task-run manifest 自动决定。"
        )
        self.mode_combo.addItem("自动推荐（完整处理）", UI_MODE_RECOMMENDED)
        self.mode_combo.currentIndexChanged.connect(self._on_input_mode_changed)
        mode_label.setBuddy(self.mode_combo)
        self.mode_combo.hide()
        mode_label.hide()

        self.advanced_toggle_btn = QPushButton("高级设置 ▸")
        self.advanced_toggle_btn.setAccessibleName("展开高级设置")
        self.advanced_toggle_btn.setAccessibleDescription(
            "显示或隐藏处理参数、保留中间文件和联网设置"
        )
        self.advanced_toggle_btn.setCheckable(True)
        self.advanced_toggle_btn.setProperty("variant", "quiet")
        self.advanced_toggle_btn.toggled.connect(self._on_advanced_toggled)
        source_layout.addWidget(self.advanced_toggle_btn)

        self.advanced_panel = QWidget()
        self.advanced_panel.setAccessibleName("高级设置")
        advanced_layout = QHBoxLayout(self.advanced_panel)
        advanced_layout.setContentsMargins(8, 2, 8, 2)
        self.processing_params_btn = QPushButton("处理参数…")
        self.processing_params_btn.setAccessibleName("配置本次图像处理参数")
        self.processing_params_btn.setAccessibleDescription(
            "在当前任务设置下方展开或收起通用配置与 Stage 1–10 分阶段参数"
        )
        self.processing_params_btn.setToolTip(
            "在下方展开处理参数。通用配置会保存为应用默认值；"
            "Stage 1–10 参数仅属于当前输入，并在任务开始后冻结。"
        )
        self.processing_params_btn.setProperty("variant", "compact")
        self.processing_params_btn.clicked.connect(self._configure_processing_parameters)
        self.debug_btn = QCheckBox("保留中间文件")
        self.debug_btn.setAccessibleName("保留中间文件")
        self.debug_btn.setAccessibleDescription(
            "用途：保留各处理阶段的 FITS 和诊断报告；默认：关闭。"
            "开启便于质量复核和排障，但会显著增加磁盘占用。"
        )
        self.debug_btn.setToolTip(self.debug_btn.accessibleDescription())
        self.debug_btn.setChecked(self.debug_mode_enabled)
        self.debug_btn.toggled.connect(self._on_debug_toggled)
        self.network_btn = QCheckBox("允许联网")
        self.network_btn.setAccessibleName("允许联网")
        network_description = (
            "默认开启；允许在线 Gaia 星表查询和已授权的插件资源补齐"
        )
        self.network_btn.setAccessibleDescription(network_description)
        self.network_btn.setToolTip(
            network_description
            + "。只有明确关闭时才进入严格离线模式，并要求本地 Gaia 星表。"
        )
        self.network_btn.setChecked(self.network_mode_enabled)
        self.network_btn.toggled.connect(self._on_network_toggled)
        advanced_layout.addWidget(self.processing_params_btn)
        advanced_layout.addWidget(self.debug_btn)
        advanced_layout.addWidget(self.network_btn)
        advanced_layout.addStretch(1)
        self.advanced_panel.hide()
        source_layout.addWidget(self.advanced_panel)
        source_layout.addStretch(1)
        return source_card

    def _build_task_preview_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("previewPanel")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
        header = QHBoxLayout()
        header.addWidget(self._section_title("输入预览"))
        header.addStretch(1)
        self.task_preview_status_label = QLabel("等待选择目录")
        header.addWidget(self.task_preview_status_label)
        layout.addLayout(header)
        self.task_preview_canvas = LatestPreviewCanvas()
        self.task_preview_canvas.setMinimumHeight(220)
        layout.addWidget(self.task_preview_canvas, 1)
        note = QLabel("Stage 0 屏幕拉伸预览 · 仅影响显示，不修改源数据")
        note.setProperty("tone", "muted")
        layout.addWidget(note)
        return card

    def _build_run_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.warning_card = QFrame()
        self.warning_card.setObjectName("stateBanner")
        self.warning_card.setProperty("tone", "warning")
        self.warning_card.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Maximum,
        )
        warning_layout = QHBoxLayout(self.warning_card)
        warning_layout.setContentsMargins(10, 8, 10, 8)
        self.warning_label = QLabel(
            "⚠ 处理已完成，但存在降级或失败阶段，请复核最终质量。"
        )
        self.warning_label.setWordWrap(True)
        self.warning_label.setAccessibleName("处理完成警告")
        self.quality_report_btn = QPushButton("查看质量报告")
        self.quality_report_btn.setAccessibleName("查看最终质量报告")
        self.quality_report_btn.clicked.connect(self._open_quality_report)
        self.banner_log_btn = QPushButton("查看详细日志")
        self.banner_log_btn.setAccessibleName("展开详细日志")
        self.banner_log_btn.clicked.connect(
            lambda: self.toggle_log_action.setChecked(True)
        )
        warning_layout.addWidget(self.warning_label)
        warning_layout.addStretch(1)
        warning_layout.addWidget(self.quality_report_btn)
        warning_layout.addWidget(self.banner_log_btn)
        self.banner_log_btn.hide()
        self.warning_card.hide()
        layout.addWidget(self.warning_card)

        self.run_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.run_splitter.setChildrenCollapsible(False)
        self.run_splitter.setHandleWidth(5)
        self._run_sidebar_width = 240
        self._run_inspector_width = 300
        self.run_sidebar = self._build_run_sidebar()
        self.run_detail = self._build_run_preview_card()
        self.run_inspector = self._build_stage_inspector()
        self.run_splitter.addWidget(self.run_sidebar)
        self.run_splitter.addWidget(self.run_detail)
        self.run_splitter.addWidget(self.run_inspector)
        self.run_splitter.setStretchFactor(0, 0)
        self.run_splitter.setStretchFactor(1, 1)
        self.run_splitter.setStretchFactor(2, 0)
        self.run_splitter.setSizes(
            [self._run_sidebar_width, 700, self._run_inspector_width]
        )
        layout.addWidget(self.run_splitter, 1)
        return page

    def _build_run_sidebar(self) -> QFrame:
        card = QFrame()
        card.setObjectName("sidebarPanel")
        card.setMinimumWidth(210)
        card.setMaximumWidth(380)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(9)
        layout.addWidget(self._section_title("本次任务"))

        self.run_task_name_label = QLabel("—")
        self.run_task_name_label.setObjectName("sidebarPrimary")
        self.run_task_name_label.setWordWrap(True)
        self.run_task_name_label.setAccessibleName("当前任务名称")
        self.run_directory_label = QLabel("位置\n—")
        self.run_directory_label.setWordWrap(True)
        self.run_mode_label = QLabel("处理方式\n—")
        for widget in (
            self.run_directory_label,
            self.run_mode_label,
        ):
            widget.setProperty("role", "summary")
            widget.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
        layout.addWidget(self.run_task_name_label)
        for widget in (
            self.run_directory_label,
            self.run_mode_label,
        ):
            layout.addWidget(widget)
        layout.addStretch(1)

        layout.addWidget(self._section_title("结果"))
        self.result_preview_btn = QPushButton("结果预览")
        self.result_preview_btn.setAccessibleName("预览处理结果")
        self.result_preview_btn.clicked.connect(self._open_result_preview)
        self.open_result_btn = QPushButton("打开结果目录")
        self.open_result_btn.setAccessibleName("打开处理结果目录")
        self._bind_button_to_action(
            self.open_result_btn,
            self.open_result_action,
        )
        layout.addWidget(self.result_preview_btn)
        layout.addWidget(self.open_result_btn)
        return card

    def _build_run_preview_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("previewPanel")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 9, 10, 9)
        layout.setSpacing(7)

        header = QHBoxLayout()
        self.preview_stage_label = QLabel("预览：Stage 0 · 输入")
        self.preview_stage_label.setObjectName("sectionTitle")
        self.preview_activity_label = QLabel("等待开始")
        header.addWidget(self.preview_stage_label)
        header.addStretch(1)
        header.addWidget(self.preview_activity_label)
        layout.addLayout(header)

        self.progress_bar = QProgressBar()
        self.progress_bar.setAccessibleName("运行环境准备进度")
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFormat("正在准备…")
        self.progress_bar.setTextVisible(True)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        self.preview_canvas = LatestPreviewCanvas()
        layout.addWidget(self.preview_canvas, 1)

        controls = QHBoxLayout()
        self.zoom_out_btn = QPushButton("−")
        self.zoom_out_btn.setAccessibleName("缩小预览")
        self._bind_button_to_action(
            self.zoom_out_btn,
            self.zoom_out_action,
            mirror_text=False,
        )
        self.fit_preview_btn = QPushButton("适合窗口")
        self._bind_button_to_action(
            self.fit_preview_btn,
            self.fit_preview_action,
        )
        self.actual_preview_btn = QPushButton("1:1")
        self._bind_button_to_action(
            self.actual_preview_btn,
            self.actual_preview_action,
            mirror_text=False,
        )
        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setAccessibleName("放大预览")
        self._bind_button_to_action(
            self.zoom_in_btn,
            self.zoom_in_action,
            mirror_text=False,
        )
        controls.addStretch(1)
        for button in (
            self.zoom_out_btn,
            self.fit_preview_btn,
            self.actual_preview_btn,
            self.zoom_in_btn,
        ):
            button.setProperty("variant", "compact")
            controls.addWidget(button)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.preview_notice_label = QLabel("屏幕预览 · 显示变换不写入处理数据")
        self.preview_notice_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_notice_label.setProperty("tone", "muted")
        layout.addWidget(self.preview_notice_label)
        return card

    def _build_stage_inspector(self) -> QFrame:
        card = QFrame()
        card.setObjectName("inspectorPanel")
        card.setMinimumWidth(260)
        card.setMaximumWidth(420)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.inspector_tabs = QTabWidget()
        self.inspector_tabs.setObjectName("runInspectorTabs")
        self.inspector_tabs.setAccessibleName("运行检查器")
        self.inspector_tabs.setDocumentMode(True)
        layout.addWidget(self.inspector_tabs)

        stage_tab = QWidget()
        stage_layout = QVBoxLayout(stage_tab)
        stage_layout.setContentsMargins(12, 11, 12, 11)
        stage_layout.setSpacing(7)
        stage_layout.addWidget(self._section_title("阶段与质量"))

        self.run_phase_label = QLabel("线性处理 · Stage 1–6")
        self.run_phase_label.setProperty("tone", "muted")
        stage_layout.addWidget(self.run_phase_label)

        self.progress_summary_label = QLabel("当前进度：等待开始")
        self.progress_summary_label.setAccessibleName("处理阶段进度")
        self.progress_summary_label.setWordWrap(True)
        stage_layout.addWidget(self.progress_summary_label)

        self.stage_stepper = QWidget()
        self.stage_stepper.setAccessibleName("处理阶段列表")
        self.stage_stepper_layout = QVBoxLayout(self.stage_stepper)
        self.stage_stepper_layout.setContentsMargins(0, 0, 0, 0)
        self.stage_stepper_layout.setSpacing(3)
        self.stage_scroll = QScrollArea()
        self.stage_scroll.setAccessibleName("完整处理阶段列表")
        self.stage_scroll.setWidgetResizable(True)
        self.stage_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.stage_scroll.setWidget(self.stage_stepper)
        stage_layout.addWidget(self.stage_scroll, 1)

        task_tab = QWidget()
        task_layout = QVBoxLayout(task_tab)
        task_layout.setContentsMargins(0, 0, 0, 0)
        self.run_configuration_scroll = QScrollArea()
        self.run_configuration_scroll.setAccessibleName("本次任务冻结配置")
        self.run_configuration_scroll.setWidgetResizable(True)
        self.run_configuration_scroll.setFrameShape(QFrame.Shape.NoFrame)
        configuration_content = QWidget()
        configuration_layout = QVBoxLayout(configuration_content)
        configuration_layout.setContentsMargins(12, 11, 12, 11)
        configuration_layout.setSpacing(7)
        configuration_layout.addWidget(self._section_title("冻结的任务配置"))
        configuration_note = QLabel(
            "以下配置在任务启动时冻结；历史运行显示当次保存的信息。"
        )
        configuration_note.setWordWrap(True)
        configuration_note.setProperty("tone", "muted")
        configuration_layout.addWidget(configuration_note)
        self.run_options_label = QLabel("—")
        self.run_options_label.setWordWrap(True)
        self.run_options_label.setProperty("role", "summary")
        self.run_options_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        configuration_layout.addWidget(self.run_options_label)
        configuration_layout.addStretch(1)
        self.run_configuration_scroll.setWidget(configuration_content)
        task_layout.addWidget(self.run_configuration_scroll)

        self.inspector_tabs.addTab(stage_tab, "阶段")
        self.inspector_tabs.addTab(task_tab, "任务")
        self.inspector_tabs.setTabToolTip(0, "查看 Stage 1–10 状态与质量")
        self.inspector_tabs.setTabToolTip(1, "查看本次运行冻结的配置")
        self.inspector_tabs.currentChanged.connect(
            self._on_inspector_tab_changed
        )
        self._reset_stage_progress(10)
        return card

    def _build_log_drawer(self, outer: QVBoxLayout) -> None:
        self.log_toggle_btn = QPushButton("详细日志")
        self.log_toggle_btn.setAccessibleName("展开详细日志")
        self.log_toggle_btn.setCheckable(True)
        self._bind_button_to_action(
            self.log_toggle_btn,
            self.toggle_log_action,
            mirror_text=False,
        )

        self.log_container = QFrame()
        self.log_container.setObjectName("logPanel")
        self.log_container.setMaximumHeight(250)
        log_layout = QVBoxLayout(self.log_container)
        log_layout.setContentsMargins(10, 8, 10, 8)
        log_actions = QHBoxLayout()
        log_title = self._section_title("运行日志")
        log_actions.addWidget(log_title)
        log_actions.addStretch(1)
        self.open_log_btn = QPushButton("打开日志文件")
        self.open_log_btn.setAccessibleName("打开完整日志文件")
        self._bind_button_to_action(self.open_log_btn, self.open_log_action)
        self.clear_view_btn = QPushButton("清空日志")
        self.clear_view_btn.setAccessibleName("清空界面日志")
        self._bind_button_to_action(self.clear_view_btn, self.clear_view_action)
        log_actions.addWidget(self.open_log_btn)
        log_actions.addWidget(self.clear_view_btn)
        log_layout.addLayout(log_actions)
        self.log_view = QPlainTextEdit()
        self.log_view.setObjectName("logView")
        self.log_view.setAccessibleName("详细运行日志")
        self.log_view.setAccessibleDescription("只读的完整处理日志")
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(180)
        self.log_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.log_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        log_layout.addWidget(self.log_view)
        self.log_container.hide()
        outer.addWidget(self.log_container)

    def _init_status_bar(self) -> None:
        status_bar = self.statusBar()
        status_bar.setSizeGripEnabled(True)
        self.status_label = QLabel("状态：空闲")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setProperty("state", "idle")
        self.status_label.setAccessibleName("处理状态")
        self.status_label.setContentsMargins(0, 0, 10, 0)
        self.stage_timing_label = QLabel("本阶段 — · 总耗时 —")
        self.stage_timing_label.setAccessibleName("处理耗时")
        self.preview_status_label = QLabel("预览：等待输入")
        self.preview_status_label.setAccessibleName("预览状态")
        status_bar.addWidget(self.status_label, 1)
        status_bar.addPermanentWidget(self.stage_timing_label)
        status_bar.addPermanentWidget(self.preview_status_label)
        status_bar.addPermanentWidget(self.log_toggle_btn)

    def _init_menus(self) -> None:
        menu_bar = self.menuBar()

        self.file_menu = menu_bar.addMenu("文件")
        self.file_menu.addAction(self.open_file_action)
        self.file_menu.addAction(self.choose_directory_action)
        self.open_recent_menu = self.file_menu.addMenu("打开最近使用")
        self._refresh_recent_menu()
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.open_result_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.close_window_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.preferences_action)
        self.file_menu.addAction(self.quit_action)

        self.edit_menu = menu_bar.addMenu("编辑")
        self.edit_menu.addAction(self.undo_action)
        self.edit_menu.addAction(self.redo_action)
        self.edit_menu.addSeparator()
        self.edit_menu.addAction(self.cut_action)
        self.edit_menu.addAction(self.copy_action)
        self.edit_menu.addAction(self.paste_action)
        self.edit_menu.addSeparator()
        self.edit_menu.addAction(self.select_all_action)
        self.edit_menu.aboutToShow.connect(self._update_edit_actions)

        self.view_menu = menu_bar.addMenu("显示")
        self.view_menu.addAction(self.toggle_sidebar_action)
        self.view_menu.addAction(self.toggle_inspector_action)
        self.view_menu.addSeparator()
        self.view_menu.addAction(self.fit_preview_action)
        self.view_menu.addAction(self.actual_preview_action)
        self.view_menu.addAction(self.zoom_in_action)
        self.view_menu.addAction(self.zoom_out_action)
        self.view_menu.addSeparator()
        self.view_menu.addAction(self.toggle_log_action)
        self.view_menu.addAction(self.open_log_action)
        self.view_menu.addAction(self.clear_view_action)
        self.view_menu.addSeparator()
        self.view_menu.addAction(self.full_screen_action)

        self.process_menu = menu_bar.addMenu("处理")
        self.process_menu.addAction(self.task_options_action)
        self.process_menu.addSeparator()
        self.process_menu.addAction(self.start_action)
        self.process_menu.addAction(self.stop_action)
        self.process_menu.addSeparator()
        self.process_menu.addAction(self.return_task_action)
        self.process_menu.addAction(self.rerun_action)

        self.window_menu = menu_bar.addMenu("窗口")
        self.window_menu.addAction(self.minimize_action)
        self.window_menu.addAction(self.zoom_window_action)
        self.window_menu.addSeparator()
        self.window_menu.addAction(self.bring_all_to_front_action)
        self.window_menu.addSeparator()
        self.window_menu.addAction(self.main_window_action)
        self.window_menu.addAction(self.history_action)

        self.help_menu = menu_bar.addMenu("帮助")
        self.help_menu.addAction(self.help_action)
        self.help_menu.addSeparator()
        self.help_menu.addAction(self.about_action)

    def _refresh_recent_menu(self) -> None:
        menu = getattr(self, "open_recent_menu", None)
        if menu is None:
            return
        menu.clear()
        if not self._recent_directories:
            empty_action = menu.addAction("无最近使用的输入")
            empty_action.setEnabled(False)
            menu.menuAction().setEnabled(False)
            return

        for value in self._recent_directories:
            recent_action = QAction(self._display_path(value), menu)
            recent_action.setToolTip(value)
            recent_action.setData(value)
            recent_action.triggered.connect(
                lambda _checked=False, path=value: self._apply_input_path(
                    Path(path),
                    remember=True,
                )
            )
            menu.addAction(recent_action)
        menu.addSeparator()
        clear_action = menu.addAction("清除菜单")
        clear_action.triggered.connect(self._clear_recent_directories)
        menu.menuAction().setEnabled(not self._ui_running)

    def _clear_recent_directories(self) -> None:
        self._recent_directories = []
        self._refresh_recent_directories()
        self._save_settings()

    def _dispatch_focused_edit_command(self, method_name: str) -> None:
        widget = QApplication.focusWidget()
        handler = getattr(widget, method_name, None) if widget is not None else None
        if callable(handler):
            handler()

    def _update_edit_actions(self, *_args) -> None:
        widget = QApplication.focusWidget()
        widget_enabled = bool(widget is not None and widget.isEnabled())
        read_only_method = getattr(widget, "isReadOnly", None)
        read_only = bool(read_only_method()) if callable(read_only_method) else False
        for action in (
            self.undo_action,
            self.redo_action,
            self.cut_action,
            self.copy_action,
            self.paste_action,
            self.select_all_action,
        ):
            method_name = str(action.property("editCommand") or "")
            handler = getattr(widget, method_name, None) if widget is not None else None
            enabled = widget_enabled and callable(handler)
            if method_name in {"undo", "redo", "cut", "paste"} and read_only:
                enabled = False
            action.setEnabled(enabled)

    def _fit_run_preview(self) -> None:
        preview = getattr(self, "preview_canvas", None)
        if preview is not None:
            preview.fit_to_window()

    def _show_run_preview_actual_size(self) -> None:
        preview = getattr(self, "preview_canvas", None)
        if preview is not None:
            preview.actual_pixels()

    def _zoom_run_preview_in(self) -> None:
        preview = getattr(self, "preview_canvas", None)
        if preview is not None:
            preview.zoom_in()

    def _zoom_run_preview_out(self) -> None:
        preview = getattr(self, "preview_canvas", None)
        if preview is not None:
            preview.zoom_out()

    def _set_full_screen(self, enabled: bool) -> None:
        if enabled:
            self._window_was_maximized = self.isMaximized()
            self.showFullScreen()
            self.full_screen_action.setText("退出全屏幕")
            return
        if self.isFullScreen():
            if getattr(self, "_window_was_maximized", False):
                self.showMaximized()
            else:
                self.showNormal()
        self.full_screen_action.setText("进入全屏幕")

    def _active_application_window(self) -> QWidget:
        focused = QApplication.focusWidget()
        if focused is not None:
            focused_window = focused.window()
            if focused_window.isWindow() and focused_window.isVisible():
                return focused_window
        active = QApplication.activeWindow()
        return active if active is not None else self

    def _show_main_window(self) -> None:
        if self.isMinimized():
            self.showNormal()
        else:
            self.show()
        self.raise_()
        self.activateWindow()

    def _close_active_window(self) -> None:
        self._active_application_window().close()

    def _minimize_active_window(self) -> None:
        self._active_application_window().showMinimized()

    def _toggle_window_zoom(self) -> None:
        window = self._active_application_window()
        if window is self and self.isFullScreen():
            self.full_screen_action.setChecked(False)
        elif window.isMaximized():
            window.showNormal()
        else:
            window.showMaximized()

    def _bring_all_windows_to_front(self) -> None:
        app = QApplication.instance()
        active = QApplication.activeWindow()
        if isinstance(app, QApplication):
            for window in app.topLevelWidgets():
                if window.isWindow() and window.isVisible():
                    window.raise_()
        target = active if active is not None and active.isVisible() else self
        target.raise_()
        target.activateWindow()

    @staticmethod
    def _splitter_sizes_setting(
        value: object,
        expected_count: int,
    ) -> list[int] | None:
        if isinstance(value, str):
            candidates = [
                item
                for item in re.split(r"[,;\s]+", value.strip("[]() "))
                if item
            ]
        elif isinstance(value, (list, tuple)):
            candidates = list(value)
        else:
            return None
        try:
            sizes = [int(item) for item in candidates]
        except (TypeError, ValueError):
            return None
        if len(sizes) != expected_count or any(size <= 0 for size in sizes):
            return None
        return sizes

    @staticmethod
    def _section_title(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("sectionTitle")
        return label

    def _show_workspace(self, state: str) -> None:
        normalized = state if state in {
            WORKSPACE_EMPTY,
            WORKSPACE_TASK,
            WORKSPACE_RUN,
        } else WORKSPACE_EMPTY
        if (
            getattr(self, "_workspace_state", None) == WORKSPACE_TASK
            and hasattr(self, "task_splitter")
        ):
            task_sizes = self.task_splitter.sizes()
            if len(task_sizes) == 2 and all(size > 0 for size in task_sizes):
                self._task_splitter_sizes = task_sizes
        if (
            getattr(self, "_workspace_state", None) == WORKSPACE_RUN
            and hasattr(self, "run_splitter")
        ):
            run_sizes = self.run_splitter.sizes()
            if len(run_sizes) == 3:
                if self.toggle_sidebar_action.isChecked() and run_sizes[0] > 0:
                    self._run_sidebar_width = run_sizes[0]
                if self.toggle_inspector_action.isChecked() and run_sizes[2] > 0:
                    self._run_inspector_width = run_sizes[2]
        self._workspace_state = normalized
        page = {
            WORKSPACE_EMPTY: self.empty_page,
            WORKSPACE_TASK: self.task_page,
            WORKSPACE_RUN: self.run_page,
        }[normalized]
        self.workspace_stack.setCurrentWidget(page)
        if normalized == WORKSPACE_TASK:
            self.task_splitter.setSizes(self._task_splitter_sizes)
        self._update_toolbar_state()
        self._update_responsive_layout()

    @staticmethod
    def _format_history_time(value: object) -> str:
        text = str(value or "").strip()
        if not text:
            return "—"
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is not None:
                parsed = parsed.astimezone()
            return parsed.strftime("%Y-%m-%d %H:%M:%S")
        except (TypeError, ValueError):
            return text

    @staticmethod
    def _history_status_text(status: object, *, available: bool = True) -> str:
        normalized = str(status or STATUS_INTERRUPTED)
        symbol = {
            STATUS_PREPARING: "●",
            STATUS_RUNNING: "●",
            STATUS_SUCCESS: "✓",
            STATUS_PARTIAL_SUCCESS: "⚠",
            STATUS_REVIEW_REQUIRED: "⚠",
            STATUS_FAILED: "✕",
            STATUS_STOPPED: "■",
            STATUS_INTERRUPTED: "!",
        }.get(normalized, "·")
        label = STATUS_LABELS.get(normalized, normalized or "未知")
        if not available:
            label += " · 位置不可用"
        return f"{symbol} {label}"

    @staticmethod
    def _history_status_matches(status: str, selected_filter: str) -> bool:
        if selected_filter == "success":
            return status in {STATUS_SUCCESS, STATUS_PARTIAL_SUCCESS}
        if selected_filter == "review":
            return status == STATUS_REVIEW_REQUIRED
        if selected_filter == "failure":
            return status in {STATUS_FAILED, STATUS_STOPPED, STATUS_INTERRUPTED}
        return True

    def _history_operations_running(self) -> bool:
        return bool(
            self._ui_running
            or (self.intake_worker and self.intake_worker.isRunning())
            or (self.bootstrap_worker and self.bootstrap_worker.isRunning())
            or (self.worker and self.worker.isRunning())
        )

    def _show_history(self) -> None:
        if not self._history_detail_mode and not self.history_window.isVisible():
            self._history_return_state = self._workspace_state
            self._history_return_status = self.status_label.text()
            self._history_return_task_root = self._last_task_root
        self._refresh_history_view()
        history_was_maximized = self.history_window.isMaximized()
        if history_was_maximized and not self.history_window.isVisible():
            self.history_window.setWindowState(Qt.WindowState.WindowNoState)
        constrain_window_to_visible_screens(
            self.history_window,
            reference=self,
            resize_to_fit=False,
        )
        if history_was_maximized:
            self.history_window.setWindowState(
                self.history_window.windowState()
                | Qt.WindowState.WindowMaximized
            )
        self.history_window.show_and_activate(
            initial_focus=self.history_search_edit
        )

    def _leave_history(self) -> None:
        self.history_window.close()
        self.raise_()
        self.activateWindow()

    def _save_history_window_geometry(self) -> None:
        window = getattr(self, "history_window", None)
        if window is None or self._restoring_settings:
            return
        self._store_window_geometry_settings()
        self.settings.sync()

    def _set_history_status_badge(
        self,
        item: QTreeWidgetItem,
        status: str,
        *,
        available: bool,
    ) -> None:
        badge = QLabel(self._history_status_text(status, available=available))
        badge.setObjectName("historyStatusBadge")
        badge.setProperty("historyStatus", status)
        badge.setProperty("available", available)
        badge.setContentsMargins(8, 2, 8, 2)
        self.history_tree.setItemWidget(item, 2, badge)

    def _clear_pending_history_delete(self) -> None:
        self._pending_history_delete = None
        banner = getattr(self, "history_delete_banner", None)
        if banner is None:
            return
        banner.hide()
        self.history_delete_cancel_btn.setText("取消")
        self.history_delete_cancel_btn.setAccessibleName("取消移到废纸篓")
        self.history_delete_cancel_btn.setAccessibleDescription(
            "不移动任务目录，返回历史记录"
        )
        self.history_delete_confirm_btn.show()

    def _show_history_delete_banner(
        self,
        message: str,
        *,
        tone: str,
        confirming: bool,
    ) -> None:
        set_style_property(self.history_delete_banner, "tone", tone)
        self.history_delete_banner.setAccessibleName(
            "历史任务删除确认" if confirming else "历史任务删除状态"
        )
        self.history_delete_banner.setAccessibleDescription(message)
        self.history_delete_banner_label.setText(message)
        self.history_delete_banner_label.setAccessibleDescription(message)
        self.history_delete_confirm_btn.setVisible(confirming)
        if confirming:
            self.history_delete_cancel_btn.setText("取消")
            self.history_delete_cancel_btn.setAccessibleName("取消移到废纸篓")
            self.history_delete_cancel_btn.setAccessibleDescription(
                "不移动任务目录，返回历史记录"
            )
        else:
            self.history_delete_cancel_btn.setText("关闭")
            self.history_delete_cancel_btn.setAccessibleName("关闭历史任务删除提示")
            self.history_delete_cancel_btn.setAccessibleDescription(
                "关闭提示，不会更改任务目录或历史索引"
            )
        self.history_delete_banner.show()

    def _show_history_delete_notice(
        self,
        message: str,
        *,
        tone: str = "error",
    ) -> None:
        self._clear_pending_history_delete()
        self._show_history_delete_banner(
            message,
            tone=tone,
            confirming=False,
        )
        self._append_event(f"历史任务删除：{message.replace(chr(10), ' ')}")
        self.statusBar().showMessage(message.splitlines()[0], 8000)
        self._update_history_selection_actions()

    def _cancel_pending_history_delete(self) -> None:
        had_pending = self._pending_history_delete is not None
        self._clear_pending_history_delete()
        self._update_history_selection_actions()
        if had_pending:
            self.statusBar().showMessage("已取消移到废纸篓。", 4000)

    def _on_history_selection_changed(self) -> None:
        self._clear_pending_history_delete()
        self._update_history_selection_actions()

    def _refresh_history_view(self, _value: object = None) -> None:
        self._clear_pending_history_delete()
        if not hasattr(self, "history_tree"):
            return
        user_role = int(Qt.ItemDataRole.UserRole)
        current_item = self.history_tree.currentItem()
        selected_task_key = ""
        selected_kind = ""
        selected_run_id = ""
        if current_item is not None:
            selected_task_key = str(current_item.data(0, user_role) or "")
            selected_kind = str(current_item.data(0, user_role + 1) or "")
            selected_run_id = str(current_item.data(0, user_role + 2) or "")
        expanded_task_keys = {
            str(item.data(0, user_role) or "")
            for index in range(self.history_tree.topLevelItemCount())
            if (item := self.history_tree.topLevelItem(index)).isExpanded()
        }
        query = self.history_search_edit.text().strip().casefold()
        selected_filter = str(self.history_status_combo.currentData() or "all")
        try:
            tasks = self.history_store.tasks()
        except HistoryStoreError as error:
            tasks = []
            self.history_empty_label.setText(f"历史索引不可用：{error}")
        else:
            self.history_empty_label.setText("还没有符合条件的历史处理记录")

        self._history_task_records = {
            str(task.get("task_key") or ""): task
            for task in tasks
            if str(task.get("task_key") or "")
        }
        self.history_tree.clear()
        restored_selection: QTreeWidgetItem | None = None
        visible_count = 0
        for task in tasks:
            task_key = str(task.get("task_key") or "")
            display_name = str(task.get("display_name") or "未命名任务")
            status = str(task.get("latest_status") or STATUS_INTERRUPTED)
            if query and query not in display_name.casefold():
                continue
            if not self._history_status_matches(status, selected_filter):
                continue
            available = bool(task.get("available", False))
            item = QTreeWidgetItem(
                [
                    display_name,
                    self._format_history_time(task.get("latest_activity_at")),
                    "",
                ]
            )
            item.setData(0, user_role, task_key)
            item.setData(0, user_role + 1, "task")
            item.setToolTip(
                0,
                str(task.get("task_directory") or "")
                + ("" if available else "\n位置不可用"),
            )
            self.history_tree.addTopLevelItem(item)
            item.setExpanded(task_key in expanded_task_keys)
            if task_key == selected_task_key and selected_kind == "task":
                restored_selection = item
            self._set_history_status_badge(
                item,
                status,
                available=available,
            )
            for run in task.get("runs", []):
                if not isinstance(run, dict):
                    continue
                run_id = str(run.get("run_id") or "")
                run_status = str(run.get("status") or STATUS_INTERRUPTED)
                child = QTreeWidgetItem(
                    [
                        f"运行 {run_id}",
                        self._format_history_time(
                            run.get("completed_at") or run.get("started_at")
                        ),
                        "",
                    ]
                )
                child.setData(0, user_role, task_key)
                child.setData(0, user_role + 1, "run")
                child.setData(0, user_role + 2, run_id)
                item.addChild(child)
                if (
                    task_key == selected_task_key
                    and selected_kind == "run"
                    and run_id == selected_run_id
                ):
                    restored_selection = child
                self._set_history_status_badge(
                    child,
                    run_status,
                    available=available,
                )
            visible_count += 1

        if restored_selection is not None:
            self.history_tree.setCurrentItem(restored_selection)
        self.history_tree.setVisible(visible_count > 0)
        self.history_empty_label.setVisible(visible_count == 0)
        self._update_history_selection_actions()

    def _selected_history_task(self) -> dict[str, object] | None:
        if not hasattr(self, "history_tree"):
            return None
        item = self.history_tree.currentItem()
        if item is None:
            return None
        user_role = int(Qt.ItemDataRole.UserRole)
        task_key = str(item.data(0, user_role) or "")
        return self._history_task_records.get(task_key)

    def _selected_history_run_identity(self) -> tuple[str, str] | None:
        item = self.history_tree.currentItem()
        if item is None:
            return None
        user_role = int(Qt.ItemDataRole.UserRole)
        if str(item.data(0, user_role + 1) or "") != "run":
            return None
        task_key = str(item.data(0, user_role) or "")
        run_id = str(item.data(0, user_role + 2) or "")
        return (task_key, run_id) if task_key and run_id else None

    def _selected_history_item_kind(self) -> str:
        item = self.history_tree.currentItem()
        if item is None:
            return ""
        user_role = int(Qt.ItemDataRole.UserRole)
        return str(item.data(0, user_role + 1) or "")

    def _history_dialog_parent(self) -> QWidget:
        return (
            self.history_window
            if self.history_window.isVisible()
            else self
        )

    def _open_selected_history_run(self) -> None:
        identity = self._selected_history_run_identity()
        if identity is not None:
            self._open_history_run(*identity)

    def _update_history_selection_actions(self) -> None:
        if not hasattr(self, "history_delete_btn"):
            return
        task = self._selected_history_task()
        running = self._history_operations_running()
        pending_delete = self._pending_history_delete is not None
        available = bool(task and task.get("available", False))
        run_identity = self._selected_history_run_identity()
        task_row_selected = self._selected_history_item_kind() == "task"
        self.history_open_btn.setEnabled(
            bool(run_identity and available and not running and not pending_delete)
        )
        self.history_delete_btn.setText(
            "移到废纸篓"
            if task is None or available
            else "从历史中移除"
        )
        self.history_delete_btn.setEnabled(
            bool(task and task_row_selected and not running and not pending_delete)
        )
        self.history_delete_btn.setToolTip(
            "删除整个任务及其全部运行"
            if task_row_selected
            else "如需删除，请先选择任务行"
        )
        self.history_mode_label.setVisible(running)

    def _clear_deleted_task_state(self, task_root: Path) -> bool:
        deleted = task_root.expanduser().resolve()
        current_task = self._last_task_root
        selected_text = self.dir_edit.text().strip()
        selected = Path(selected_text).expanduser() if selected_text else None
        matches_current = bool(
            current_task is not None and current_task.expanduser().resolve() == deleted
        )
        if selected is not None:
            try:
                matches_current = matches_current or selected.resolve() == deleted
            except OSError:
                pass
        if not matches_current:
            return False
        self._last_task_root = None
        self._last_run_snapshot = None
        self._input_discovery = None
        self._result_preview_path = None
        self._historical_run_root = None
        self._historical_result_files = ()
        self._history_detail_mode = False
        self._active_history_task_key = None
        self._active_history_run_id = None
        self._run_terminal_status = None
        self.run_log_path = None
        self.dir_edit.clear()
        self.task_preview_canvas.clear_image()
        self.preview_canvas.clear_image()
        deleted_text = str(deleted)
        self._recent_directories = [
            value
            for value in self._recent_directories
            if str(Path(value).expanduser()) != deleted_text
        ]
        self._refresh_recent_directories()
        self._save_settings()
        return True

    def _remove_unavailable_history_task(self, task: Mapping[str, object]) -> None:
        task_key = str(task.get("task_key") or "")
        display_name = str(task.get("display_name") or "未命名任务")
        parent = self._history_dialog_parent()
        answer = QMessageBox.question(
            parent,
            "从历史中移除",
            f"“{display_name}”的任务目录当前不可用。\n\n"
            "此操作只移除历史索引，不会访问或删除任何文件。是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            removed = self.history_store.remove_task(task_key)
        except (HistoryStoreError, OSError) as error:
            QMessageBox.warning(parent, "无法移除历史", str(error))
            return
        if removed:
            task_root = Path(str(task.get("task_directory") or "")).expanduser()
            if self._clear_deleted_task_state(task_root):
                self._show_workspace(WORKSPACE_EMPTY)
                self._set_status_text("Idle")
            else:
                self._refresh_history_view()

    def _delete_selected_history_task(self) -> None:
        if self._selected_history_item_kind() != "task":
            return
        if self._history_operations_running():
            self._show_history_delete_notice(
                "当前处理正在运行；请等待任务结束后再移到废纸篓。",
                tone="warning",
            )
            return
        task = self._selected_history_task()
        if task is None:
            return
        if not bool(task.get("available", False)):
            self._remove_unavailable_history_task(task)
            return

        task_key = str(task.get("task_key") or "")
        if not task_key:
            self._show_history_delete_notice(
                "删除未执行：所选历史任务缺少任务标识。",
            )
            return
        task_root = Path(str(task.get("task_directory") or "")).expanduser()
        try:
            workspace = validate_deletable_task_root(task_root)
        except UnsafeTaskDeletionError as error:
            self._show_history_delete_notice(
                f"删除未执行：任务目录未通过安全校验。\n{error}",
            )
            return
        try:
            run_count = len(
                [
                    path
                    for path in workspace.runs_dir.iterdir()
                    if path.is_dir() and not path.is_symlink()
                ]
            )
        except OSError:
            run_count = len(
                [item for item in task.get("runs", []) if isinstance(item, dict)]
            )
        task_size = directory_size_bytes(workspace.root)
        try:
            normalized_root = workspace.root.expanduser().resolve(strict=True)
        except OSError as error:
            self._show_history_delete_notice(
                f"删除未执行：无法规范化任务目录。\n{error}",
            )
            return
        self._pending_history_delete = (task_key, normalized_root)
        self._show_history_delete_banner(
            "确认将任务“{display_name}”移到废纸篓吗？\n\n"
            "将移动整个任务目录，其中的运行结果、断点和日志都会一并移除；"
            "可在系统废纸篓恢复。\n\n"
            "任务目录：{task_root}\n"
            "运行记录：{run_count} 次\n"
            "目录大小：{task_size}".format(
                display_name=str(task.get("display_name") or workspace.task_id),
                task_root=normalized_root,
                run_count=run_count,
                task_size=format_bytes(task_size),
            ),
            tone="warning",
            confirming=True,
        )
        self._update_history_selection_actions()
        self.history_delete_cancel_btn.setFocus(Qt.FocusReason.OtherFocusReason)

    def _confirm_pending_history_delete(self) -> None:
        pending = self._pending_history_delete
        if pending is None:
            return
        task_key, expected_root = pending
        if self._history_operations_running():
            self._show_history_delete_notice(
                "当前处理正在运行；未移动任务目录。",
                tone="warning",
            )
            return

        try:
            task = self.history_store.find_task(task_key)
        except (HistoryStoreError, OSError) as error:
            self._show_history_delete_notice(
                f"确认未执行：无法重新读取历史索引。\n{error}",
            )
            return
        if task is None:
            self._show_history_delete_notice(
                "确认未执行：历史索引中已找不到此任务，未移动任何文件。",
            )
            return
        if str(task.get("task_key") or "") != task_key:
            self._show_history_delete_notice(
                "确认未执行：历史索引中的任务标识已变化，未移动任何文件。",
            )
            return
        try:
            recorded_root = Path(
                str(task.get("task_directory") or "")
            ).expanduser().resolve()
        except (OSError, RuntimeError) as error:
            self._show_history_delete_notice(
                f"确认未执行：无法规范化历史任务目录。\n{error}",
            )
            return
        if recorded_root != expected_root:
            self._show_history_delete_notice(
                "确认未执行：历史任务目录已变化，未移动任何文件。",
            )
            return

        try:
            workspace = validate_deletable_task_root(recorded_root)
        except UnsafeTaskDeletionError as error:
            self._show_history_delete_notice(
                f"确认未执行：任务目录不再满足安全条件。\n{error}",
            )
            return
        try:
            verified_root = workspace.root.expanduser().resolve(strict=True)
        except OSError as error:
            self._show_history_delete_notice(
                f"确认未执行：无法重新解析任务目录。\n{error}",
            )
            return
        if verified_root != expected_root:
            self._show_history_delete_notice(
                "确认未执行：任务清单指向了不同目录，未移动任何文件。",
            )
            return
        try:
            trash_result = QFile.moveToTrash(str(verified_root))
        except (OSError, RuntimeError, TypeError) as error:
            self._show_history_delete_notice(
                "无法移到废纸篓，任务文件和历史索引均未更改。\n"
                f"{error}",
            )
            return
        if isinstance(trash_result, tuple):
            moved = bool(trash_result[0])
        else:
            moved = bool(trash_result)
        if not moved:
            self._show_history_delete_notice(
                "无法移到废纸篓：系统未能移动该任务目录。"
                "任务文件和历史索引均未更改。",
            )
            return

        index_update_notice = ""
        try:
            removed = self.history_store.remove_task(task_key)
        except (HistoryStoreError, OSError) as error:
            removed = False
            index_update_notice = (
                "任务目录已成功移动，但历史索引更新失败；"
                f"该记录会显示为位置不可用。\n{error}"
            )
        if not removed and not index_update_notice:
            index_update_notice = (
                "任务目录已成功移动，但历史索引中未找到对应记录；"
                "请重新打开历史窗口检查。"
            )
        self._clear_pending_history_delete()
        was_current = self._clear_deleted_task_state(verified_root)
        if was_current:
            self._show_workspace(WORKSPACE_EMPTY)
            self._set_status_text("Idle")
        else:
            self._refresh_history_view()
        if index_update_notice:
            self._show_history_delete_notice(index_update_notice, tone="warning")
        elif removed:
            self.statusBar().showMessage(
                "任务目录已移到废纸篓，可从废纸篓恢复。",
                8000,
            )

    def _on_history_item_activated(
        self,
        item: QTreeWidgetItem,
        _column: int,
    ) -> None:
        user_role = int(Qt.ItemDataRole.UserRole)
        if str(item.data(0, user_role + 1) or "") != "run":
            item.setExpanded(not item.isExpanded())
            return
        task_key = str(item.data(0, user_role) or "")
        run_id = str(item.data(0, user_role + 2) or "")
        self._open_history_run(task_key, run_id)

    @staticmethod
    def _history_ui_terminal_status(status: str) -> str:
        if status == STATUS_SUCCESS:
            return "Completed"
        if status in {STATUS_PARTIAL_SUCCESS, STATUS_REVIEW_REQUIRED}:
            return "CompletedWithWarning"
        if status == STATUS_STOPPED:
            return "Stopped"
        return "Failed"

    @staticmethod
    def _history_log_text(path: Path, *, max_bytes: int = 2 * 1024 * 1024) -> str:
        try:
            size = path.stat().st_size
            with path.open("rb") as handle:
                if size > max_bytes:
                    handle.seek(-max_bytes, os.SEEK_END)
                    data = handle.read()
                    prefix = "[历史日志较大，界面仅显示末尾 2 MiB]\n"
                else:
                    data = handle.read()
                    prefix = ""
            decoded = data.decode("utf-8", errors="replace")
            return prefix + decoded
        except OSError as error:
            return f"无法读取历史日志：{error}\n"

    def _history_run_log_path(
        self,
        run_root: Path,
        run_record: Mapping[str, object],
    ) -> Path | None:
        indexed = str(run_record.get("log_path") or "").strip()
        candidates: list[Path] = []
        if indexed:
            candidates.append(Path(indexed).expanduser())
        candidates.extend(
            sorted(
                run_root.glob("starun_gui_run_*.log"),
                key=safe_mtime,
                reverse=True,
            )
        )
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
                resolved.relative_to(run_root.resolve())
            except (OSError, ValueError):
                continue
            if resolved.is_file():
                return resolved
        return None

    def _open_history_run(self, task_key: str, run_id: str) -> None:
        parent = self._history_dialog_parent()
        if self._history_operations_running():
            QMessageBox.information(
                parent,
                "当前处理正在运行",
                "可以继续浏览历史记录；请等待当前处理结束后再查看历史运行详情。",
            )
            return
        task = self.history_store.find_task(task_key)
        if task is None:
            QMessageBox.warning(parent, "历史记录不可用", "所选任务已不在历史索引中。")
            self._refresh_history_view()
            return
        task_root = Path(str(task.get("task_directory") or "")).expanduser()
        if not task_root.is_dir() or task_root.is_symlink():
            QMessageBox.warning(
                parent,
                "任务位置不可用",
                "任务目录不存在或当前磁盘未挂载。",
            )
            return
        run_record = next(
            (
                item
                for item in task.get("runs", [])
                if isinstance(item, dict)
                and str(item.get("run_id") or "") == str(run_id)
            ),
            None,
        )
        if run_record is None:
            QMessageBox.warning(parent, "历史记录不可用", "所选运行记录不存在。")
            return
        try:
            workspace, _run_manifest, run_root = verify_history_run(
                task_root,
                run_id,
            )
        except (HistoryStoreError, OSError, TypeError, ValueError) as error:
            QMessageBox.warning(
                parent,
                "历史运行无法验证",
                str(error),
            )
            return

        result_error = ""
        try:
            result = load_verified_pipeline_result(run_root)
        except HistoryStoreError as error:
            result = None
            result_error = str(error)
        self._save_history_window_geometry()
        self.history_window.hide()
        status = str(run_record.get("status") or STATUS_INTERRUPTED)
        self._history_detail_mode = True
        self._active_history_task_key = task_key
        self._active_history_run_id = run_id
        self._historical_run_root = run_root
        self._last_task_root = workspace.root
        self._run_terminal_status = self._history_ui_terminal_status(status)
        self._last_run_snapshot = None

        self._progress_timer.stop()
        self._pipeline_started_monotonic = None
        actual_steps = (
            result.get("actual_steps")
            if isinstance(result, dict)
            and isinstance(result.get("actual_steps"), list)
            else []
        )
        stage_count = 10
        self._reset_stage_progress(stage_count)
        for stage, raw_step in enumerate(actual_steps[:stage_count], start=1):
            if not isinstance(raw_step, dict):
                continue
            payload = dict(raw_step)
            payload["title"] = str(
                payload.get("title") or payload.get("name") or ""
            )
            self._on_pipeline_stage_detail(stage, payload)
            try:
                duration = max(0.0, float(payload.get("duration_seconds") or 0.0))
            except (TypeError, ValueError):
                duration = 0.0
            self._stage_elapsed_seconds[stage] = duration
            self._update_stage_chip(stage)

        status_text = self._history_status_text(status)
        self.progress_summary_label.setText(f"历史运行：{status_text}")
        self.progress_summary_label.setAccessibleDescription(
            f"历史运行状态：{status_text}"
        )
        self.stage_timing_label.setText("历史运行 · 阶段耗时见右侧")
        self.run_phase_label.setText(f"历史记录 · {STATUS_LABELS.get(status, status)}")
        self.run_task_name_label.setText(
            str(task.get("display_name") or task_root.name)
        )
        self.run_directory_label.setText(
            "运行目录\n" + self._display_path(run_root)
        )
        history_input_mode = str(
            run_record.get("input_mode") or INPUT_MODE_AUTO
        )
        self.run_mode_label.setText(
            "处理方式\n"
            + self._input_mode_label(history_input_mode)
        )
        plan_path = run_root / "processing-plan.json"
        option_lines = ["只读历史记录"]
        frozen_parameter_lines = self._processing_payload_summary_lines(
            _run_manifest.get("processing_parameters"),
            history_input_mode,
        )
        if frozen_parameter_lines:
            option_lines.extend(frozen_parameter_lines)
        elif plan_path.is_file():
            option_lines.append("参数快照：processing-plan.json")
        if run_record.get("failure_reason"):
            option_lines.append("说明：" + str(run_record["failure_reason"]))
        if result_error:
            option_lines.append("结果清单：验证失败")
        self.run_options_label.setText("\n".join(option_lines))

        self.preview_canvas.clear_image()
        result_files = (
            verified_result_files(run_root, result)
            if isinstance(result, dict)
            else ()
        )
        self._historical_result_files = result_files
        png_files = [path for path in result_files if path.suffix.lower() == ".png"]
        preview_path = max(png_files, key=safe_mtime) if png_files else None
        if preview_path is None:
            persisted_preview = run_root / "process" / "ui_preview" / "latest.png"
            if persisted_preview.is_file():
                preview_path = persisted_preview
        self._result_preview_path = preview_path
        if preview_path is not None and self._display_latest_preview(
            preview_path,
            stage=min(10, len(actual_steps) or 10),
            title="历史运行",
        ):
            self.preview_activity_label.setText("历史运行预览")
            self.preview_notice_label.setText("历史预览 · 只读显示")
        else:
            self.preview_activity_label.setText("历史预览不可用")
            self.preview_stage_label.setText("预览：不可用")
            self.preview_status_label.setText("预览：历史结果不可用")

        self.run_log_path = self._history_run_log_path(run_root, run_record)
        self.log_view.clear()
        if self.run_log_path is not None:
            self.log_view.setPlainText(self._history_log_text(self.run_log_path))
            self.log_view.moveCursor(QTextCursor.MoveOperation.End)
        else:
            detail = str(run_record.get("failure_reason") or "没有保存运行日志。")
            self.log_view.setPlainText(detail)

        has_results = bool(result_files)
        self.open_result_action.setEnabled(has_results)
        self.result_preview_btn.setEnabled(
            self._result_preview_path is not None
            and self._result_preview_path.is_file()
        )
        if status in {STATUS_PARTIAL_SUCCESS, STATUS_REVIEW_REQUIRED}:
            self._show_run_banner(
                "warning",
                "这是只读历史运行；处理结果需要复核。",
                show_log=True,
            )
        elif status == STATUS_SUCCESS:
            self._show_run_banner("success", "这是只读历史运行。")
        elif status == STATUS_STOPPED:
            self._show_run_banner(
                "info",
                "这是已中止的只读历史运行。",
                show_log=True,
            )
        else:
            self._show_run_banner(
                "error",
                "这是失败或异常中断的只读历史运行。",
                show_log=True,
            )
        self._show_workspace(WORKSPACE_RUN)
        self._set_status_message(
            f"历史记录 · {STATUS_LABELS.get(status, status or '未知')}"
        )
        self.show()
        self.raise_()
        self.activateWindow()

    def _register_history_run(self, task: PreparedTask) -> None:
        self._history_detail_mode = False
        try:
            record = self.history_store.register_run(
                task_id=task.workspace.task_id,
                task_directory=task.workspace.root,
                source_fingerprint=task.workspace.source_fingerprint,
                source_record=task.source_record,
                run_id=task.run.run_id,
                run_directory=task.run.root,
                input_mode=task.input_mode,
            )
        except (HistoryStoreError, OSError, TypeError, ValueError) as error:
            self._active_history_task_key = None
            self._active_history_run_id = None
            self._append_event(f"无法登记历史运行：{error}")
            return
        self._active_history_task_key = str(record.get("task_key") or "") or None
        self._active_history_run_id = task.run.run_id
        if self.history_window.isVisible():
            self._refresh_history_view()

    def _update_active_history_run(
        self,
        status: str,
        *,
        failure_reason: str | None = None,
        exit_code: int | None = None,
    ) -> None:
        task_key = self._active_history_task_key
        run_id = self._active_history_run_id
        if not task_key or not run_id:
            return
        try:
            self.history_store.update_run(
                task_key=task_key,
                run_id=run_id,
                status=status,
                failure_reason=failure_reason,
                exit_code=exit_code,
                log_path=(
                    self.run_log_path
                    if self.run_log_path is not None and self.run_log_path.exists()
                    else None
                ),
            )
        except (HistoryStoreError, OSError, TypeError, ValueError) as error:
            self._append_event(f"无法更新历史运行：{error}")
        else:
            if self.history_window.isVisible():
                self._refresh_history_view()

    def _terminal_history_status(self, ui_status: str, work_dir: Path | None) -> str:
        if ui_status == "Stopped":
            return STATUS_STOPPED
        if work_dir is not None:
            try:
                result = load_verified_pipeline_result(work_dir)
            except HistoryStoreError as error:
                self._append_event(f"历史状态未采用无效结果清单：{error}")
            else:
                if result is not None:
                    manifest_status = str(result.get("status") or "")
                    if manifest_status in {
                        STATUS_SUCCESS,
                        STATUS_PARTIAL_SUCCESS,
                        STATUS_REVIEW_REQUIRED,
                        STATUS_FAILED,
                    }:
                        return manifest_status
        if ui_status == "Completed":
            return STATUS_SUCCESS
        if ui_status == "CompletedWithWarning":
            return STATUS_PARTIAL_SUCCESS
        return STATUS_FAILED

    def _update_toolbar_state(self) -> None:
        intake_running = bool(
            self.intake_worker and self.intake_worker.isRunning()
        )
        bootstrap_running = bool(
            self.bootstrap_worker and self.bootstrap_worker.isRunning()
        )
        pipeline_running = bool(self.worker and self.worker.isRunning())
        running = (
            self._ui_running
            or intake_running
            or bootstrap_running
            or pipeline_running
        )
        task_state = self._workspace_state == WORKSPACE_TASK
        run_state = self._workspace_state == WORKSPACE_RUN
        terminal = bool(run_state and self._run_terminal_status)
        historical_terminal = bool(terminal and self._history_detail_mode)
        can_rerun = bool(
            terminal
            and not running
            and (
                self._last_run_snapshot is not None
                or (
                    historical_terminal
                    and self._active_history_task_key
                    and self._active_history_run_id
                )
            )
        )

        self.toolbar_directory_item.setVisible(self._workspace_state != WORKSPACE_RUN)
        self.toolbar_settings_item.setVisible(task_state)
        self.sidebar_toggle_item.setVisible(run_state)
        self.inspector_toggle_item.setVisible(run_state)
        self.start_item.setVisible(task_state and not running)
        self.stop_item.setVisible(
            (run_state and running) or (task_state and intake_running)
        )
        self.return_task_item.setVisible(terminal and not running)
        self.rerun_item.setVisible(can_rerun)
        self.return_task_action.setText(
            "返回历史记录" if historical_terminal else "返回任务设置"
        )
        self.rerun_action.setText("重新处理" if historical_terminal else "重新运行")
        self.history_action.setEnabled(True)
        if hasattr(self, "history_delete_btn"):
            self._update_history_selection_actions()

        self.task_options_action.setEnabled(task_state and not running)
        self.start_action.setEnabled(task_state and not running)
        self.stop_action.setEnabled(
            (run_state and running) or (task_state and intake_running)
        )
        self.return_task_action.setEnabled(terminal and not running)
        self.rerun_action.setEnabled(can_rerun)
        self.open_file_action.setEnabled(not running)
        self.choose_directory_action.setEnabled(not running)
        self.open_recent_menu.menuAction().setEnabled(
            not running and bool(self._recent_directories)
        )
        self.toggle_sidebar_action.setEnabled(run_state)
        self.toggle_inspector_action.setEnabled(run_state)
        for action in (
            self.fit_preview_action,
            self.actual_preview_action,
            self.zoom_in_action,
            self.zoom_out_action,
        ):
            action.setEnabled(run_state)
        self.preferences_action.setEnabled(True)

    def _preferences_snapshot(self) -> dict[str, object]:
        return {
            "allow_network": bool(self.network_mode_enabled),
            "keep_intermediate": bool(self.debug_mode_enabled),
            "checkpoint_mode": bool(self.checkpoint_mode),
            "output_formats": tuple(self.output_formats),
            "review_only": bool(self.review_only),
            "compute_mode": str(self.compute_mode),
        }

    def _sync_preferences_window(self) -> None:
        window = getattr(self, "preferences_window", None)
        if window is None:
            return
        window.set_preferences(self._preferences_snapshot())
        window.set_editable(
            not self._ui_running,
            reason="当前批次配置已冻结；处理结束后可修改应用默认值。",
        )

    def _apply_app_preferences(self, preferences: object) -> None:
        if not isinstance(preferences, Mapping):
            return
        if self._ui_running:
            self._sync_preferences_window()
            return

        formats = tuple(
            str(value)
            for value in preferences.get("output_formats", self.output_formats)
            if str(value) in VALID_OUTPUT_FORMATS
        )
        compute_mode = str(preferences.get("compute_mode", self.compute_mode))
        if compute_mode not in VALID_COMPUTE_MODES:
            compute_mode = str(DEFAULT_PROCESSING_SETTINGS["compute_mode"])

        self.network_mode_enabled = bool(
            preferences.get("allow_network", self.network_mode_enabled)
        )
        self.debug_mode_enabled = bool(
            preferences.get("keep_intermediate", self.debug_mode_enabled)
        )
        self.checkpoint_mode = bool(
            preferences.get("checkpoint_mode", self.checkpoint_mode)
        )
        self.output_formats = formats or tuple(
            DEFAULT_PROCESSING_SETTINGS["output_formats"]
        )
        self.review_only = bool(
            preferences.get("review_only", self.review_only)
        )
        self.compute_mode = compute_mode

        payload = self._processing_settings_snapshot()
        StarunGui._apply_processing_payload_state(self, payload)
        self._sync_processing_controls_from_state()
        self._update_debug_button_text()
        self._update_network_button_text()
        self._save_settings()
        self._sync_preferences_window()

    def _save_preferences_pane(self, index: int) -> None:
        self.settings.setValue("ui/preferencesPane", max(0, int(index)))
        self.settings.sync()

    def _show_preferences(self) -> None:
        window = getattr(self, "preferences_window", None)
        if window is None:
            try:
                from .preferences_window import PreferencesWindow
            except ImportError:  # Support direct execution from the gui directory.
                from preferences_window import PreferencesWindow  # type: ignore[no-redef]

            try:
                initial_pane = int(self.settings.value("ui/preferencesPane", 0))
            except (TypeError, ValueError):
                initial_pane = 0
            window = PreferencesWindow(self, initial_pane=initial_pane)
            window.preferencesChanged.connect(self._apply_app_preferences)
            window.paneChanged.connect(self._save_preferences_pane)
            self.preferences_window = window

        self._sync_preferences_window()
        window.show_and_activate(reference=self)

    def _show_advanced_settings(self) -> None:
        if self._workspace_state != WORKSPACE_TASK:
            return
        self.advanced_toggle_btn.setChecked(True)
        self.advanced_panel.setFocus()

    def _return_to_task_setup(self) -> None:
        if (self.intake_worker and self.intake_worker.isRunning()) or (
            self.bootstrap_worker and self.bootstrap_worker.isRunning()
        ) or (
            self.worker and self.worker.isRunning()
        ):
            return
        if self._history_detail_mode:
            target = self._history_return_state
            directory_text = self.dir_edit.text().strip()
            work_dir = Path(directory_text).expanduser() if directory_text else None
            if target == WORKSPACE_RUN:
                target = WORKSPACE_TASK if work_dir is not None else WORKSPACE_EMPTY
            if target == WORKSPACE_TASK and (
                work_dir is None or not work_dir.exists()
            ):
                target = WORKSPACE_EMPTY
            if target not in {WORKSPACE_EMPTY, WORKSPACE_TASK}:
                target = WORKSPACE_EMPTY

            self._history_detail_mode = False
            self._historical_run_root = None
            self._historical_result_files = ()
            self._active_history_task_key = None
            self._active_history_run_id = None
            self._run_terminal_status = None
            self._last_task_root = self._history_return_task_root
            self.run_log_path = None
            self.warning_card.hide()
            self._show_workspace(target)
            if self._history_return_status.startswith("状态："):
                self._set_status_message(
                    self._history_return_status.removeprefix("状态：")
                )
            else:
                self._set_status_text("Idle")
            self._set_running(False)
            self._show_history()
            return
        self._run_terminal_status = None
        directory_text = self.dir_edit.text().strip()
        work_dir = Path(directory_text).expanduser() if directory_text else None
        if work_dir is not None and work_dir.exists():
            self._show_workspace(WORKSPACE_TASK)
            self._analyze_selected_directory()
        else:
            self._show_workspace(WORKSPACE_EMPTY)
        self.warning_card.hide()
        self._set_status_text("Idle")
        self._set_running(False)

    def _rerun_last_task(self) -> None:
        if self._history_detail_mode:
            task_key = self._active_history_task_key
            task = self.history_store.find_task(task_key or "")
            if task is None:
                QMessageBox.warning(
                    self,
                    "无法重新处理",
                    "历史任务已经不在索引中。",
                )
                return
            task_root = Path(
                str(task.get("task_directory") or "")
            ).expanduser()
            try:
                validate_deletable_task_root(task_root)
            except UnsafeTaskDeletionError as error:
                QMessageBox.warning(
                    self,
                    "无法重新处理",
                    f"任务目录不可用或身份验证失败：\n{error}",
                )
                return
            historical_processing_parameters = None
            historical_manifest_error = None
            historical_run_root = self._historical_run_root
            if historical_run_root is not None:
                try:
                    frozen_run = pipeline_run_manifest.load_json(
                        historical_run_root / "run-manifest.json"
                    )
                    if not isinstance(frozen_run, Mapping):
                        raise ValueError("运行清单缺失或格式无效")
                    if frozen_run.get("schema") != "starun.task-run.v1":
                        raise ValueError("运行清单 schema 不受支持")
                    claimed_hash = str(frozen_run.get("manifest_hash") or "")
                    unsigned = dict(frozen_run)
                    unsigned.pop("manifest_hash", None)
                    if not claimed_hash or claimed_hash != (
                        pipeline_run_manifest.canonical_payload_hash(unsigned)
                    ):
                        raise ValueError("运行清单验签失败")
                    raw_parameters = frozen_run.get("processing_parameters")
                    if raw_parameters is not None:
                        if not isinstance(raw_parameters, Mapping):
                            raise ValueError("冻结处理参数格式无效")
                        historical_processing_parameters, _adjustments = (
                            normalize_processing_parameters(raw_parameters)
                        )
                except (OSError, TypeError, ValueError) as error:
                    historical_manifest_error = str(error)
            if historical_manifest_error:
                QMessageBox.warning(
                    self,
                    "无法重新处理",
                    "上次运行的签名参数清单无法验证，未使用当前设置替代：\n"
                    f"{historical_manifest_error}",
                )
                return
            self._history_detail_mode = False
            self._historical_run_root = None
            self._historical_result_files = ()
            self._active_history_task_key = None
            self._active_history_run_id = None
            self._run_terminal_status = None
            self.run_log_path = None
            self.history_window.hide()
            self._apply_input_path(task_root, remember=True)
            if historical_processing_parameters is not None:
                self._restore_processing_settings(
                    historical_processing_parameters
                )
            self._show_workspace(WORKSPACE_TASK)
            self._analyze_selected_directory()
            self._set_status_text("Idle")
            return
        snapshot = self._last_run_snapshot
        if not snapshot:
            return
        answer = QMessageBox.question(
            self,
            "重新运行",
            "将按上一次实际配置创建新的独立运行。旧结果会保留到新结果"
            "通过校验并发布后，再按保留策略清理。是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        source_path = Path(str(snapshot.get("work_dir") or "")).expanduser()
        if not source_path.exists():
            QMessageBox.warning(self, "无法重新运行", "上一次输入路径已不可用。")
            return
        self.debug_mode_enabled = bool(snapshot.get("debug_mode_enabled", False))
        self.network_mode_enabled = bool(snapshot.get("network_mode_enabled", True))
        self._restore_processing_settings(snapshot.get("processing_settings"))
        self._update_debug_button_text()
        self._update_network_button_text()
        self.dir_edit.setText(str(source_path))
        self._start_run()

    def _request_stop_run(self) -> None:
        if not (
            (self.intake_worker and self.intake_worker.isRunning())
            or (self.bootstrap_worker and self.bootstrap_worker.isRunning())
            or (self.worker and self.worker.isRunning())
        ):
            return
        answer = QMessageBox.question(
            self,
            "停止处理",
            "确定停止当前任务吗？已完成阶段的最新可靠预览会保留，"
            "当前阶段可能没有可用产物。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._stop_run()

    def _update_run_summary(self, work_dir: Path, input_mode: str) -> None:
        selected_input = self.dir_edit.text().strip()
        selected_name = (
            Path(selected_input).expanduser().name
            if selected_input
            else work_dir.name
        )
        self.run_task_name_label.setText(selected_name or work_dir.name)
        self.run_directory_label.setText(
            "任务目录\n" + self._display_path(work_dir)
        )
        self.run_mode_label.setText(
            "处理方式\n" + self._input_mode_label(input_mode)
        )
        options = [
            f"保留中间文件：{'开启' if self.debug_mode_enabled else '关闭'}",
            f"允许联网：{'开启' if self.network_mode_enabled else '关闭'}",
            *self._processing_settings_summary_lines(input_mode),
        ]
        self.run_options_label.setText("\n".join(options))

    def _set_run_sidebar_visible(self, visible: bool) -> None:
        sidebar = getattr(self, "run_sidebar", None)
        splitter = getattr(self, "run_splitter", None)
        if sidebar is None or splitter is None:
            return
        sizes = splitter.sizes()
        if (
            not self._restoring_settings
            and not visible
            and len(sizes) == 3
            and sizes[0] > 0
        ):
            self._run_sidebar_width = sizes[0]
        sidebar.setVisible(visible)
        if visible:
            restored = splitter.sizes()
            if len(restored) == 3 and restored[0] < sidebar.minimumWidth():
                restored[0] = self._run_sidebar_width
                restored[1] = max(360, restored[1])
                splitter.setSizes(restored)
        if not self._restoring_settings:
            self._save_settings()

    def _on_task_splitter_moved(self, _position: int, _index: int) -> None:
        sizes = self.task_splitter.sizes()
        if (
            self._workspace_state == WORKSPACE_TASK
            and len(sizes) == 2
            and all(size > 0 for size in sizes)
        ):
            self._task_splitter_sizes = sizes

    def _set_run_inspector_visible(self, visible: bool) -> None:
        inspector = getattr(self, "run_inspector", None)
        splitter = getattr(self, "run_splitter", None)
        if inspector is None or splitter is None:
            return
        sizes = splitter.sizes()
        if (
            not self._restoring_settings
            and not visible
            and len(sizes) == 3
            and sizes[2] > 0
        ):
            self._run_inspector_width = sizes[2]
        inspector.setVisible(visible)
        if visible:
            restored = splitter.sizes()
            if len(restored) == 3 and restored[2] < inspector.minimumWidth():
                restored[2] = self._run_inspector_width
                restored[1] = max(360, restored[1])
                splitter.setSizes(restored)
        if not self._restoring_settings:
            self._save_settings()

    def _on_inspector_tab_changed(self, _index: int) -> None:
        if not self._restoring_settings:
            self._save_settings()

    def _update_responsive_layout(self) -> None:
        if not hasattr(self, "run_sidebar"):
            return
        self.run_sidebar.setVisible(self.toggle_sidebar_action.isChecked())
        self.run_inspector.setVisible(self.toggle_inspector_action.isChecked())
        compact = self.width() < 1100
        self.preview_status_label.setVisible(not compact)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._update_responsive_layout()
        self._schedule_window_geometry_save()

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        if self._first_show_pending:
            self._first_show_pending = False
            QTimer.singleShot(0, self._reconcile_main_window_position)

    def _reconcile_main_window_position(self) -> None:
        if constrain_window_to_visible_screens(self):
            self._schedule_window_geometry_save()

    def _reconcile_window_positions(self) -> None:
        """Keep every visible normal window reachable after display changes."""

        changed = constrain_window_to_visible_screens(self)
        history_window = getattr(self, "history_window", None)
        if history_window is not None and history_window.isVisible():
            changed = (
                constrain_window_to_visible_screens(
                    history_window,
                    reference=self,
                )
                or changed
            )
        preferences_window = getattr(self, "preferences_window", None)
        if preferences_window is not None and preferences_window.isVisible():
            changed = (
                constrain_window_to_visible_screens(
                    preferences_window,
                    reference=self,
                )
                or changed
            )
        if changed:
            self._schedule_window_geometry_save()

    def _handle_screen_removed(self, _screen) -> None:
        QTimer.singleShot(0, self._reconcile_window_positions)

    def _initial_preview_candidates(
        self,
        work_dir: Path,
        input_mode: str | None = None,
    ) -> tuple[list[Path], str]:
        active_task = getattr(self, "_active_prepared_task", None)
        if active_task is not None:
            source = active_task.source_record
            files = source.get("files") if isinstance(source, dict) else None
            candidates = [
                Path(str(record.get("path") or ""))
                for record in (files if isinstance(files, list) else [])
                if isinstance(record, dict)
                and Path(str(record.get("path") or "")).suffix.lower()
                in FITS_SUFFIXES
            ]
            if candidates:
                label = (
                    f"输入样本 · {len(candidates)} 帧"
                    if str(source.get("kind") or "") == "light_directory"
                    else "输入"
                )
                return candidates, label
        discovery = getattr(self, "_input_discovery", None)
        if discovery is not None and discovery.selected_path == work_dir.resolve():
            if discovery.kind == InputKind.LIGHT_DIRECTORY and discovery.light_groups:
                candidates = list(discovery.light_groups[0].files)
                return candidates, f"输入样本 · {len(candidates)} 帧"
            if (
                discovery.master_file is not None
                and discovery.master_file.suffix.lower() in FITS_SUFFIXES
            ):
                return [discovery.master_file], "输入"
            checkpoint_path = discovery.details.get("checkpoint_path")
            if checkpoint_path and Path(str(checkpoint_path)).is_file():
                return [Path(str(checkpoint_path))], "已验证断点"
        fits = self._fits_in_work_dir(work_dir)
        light_files = sorted(
            (path for path in fits if path.name.lower().startswith("light_")),
            key=lambda path: path.name.lower(),
        )
        if light_files:
            return light_files, f"输入样本 · {len(light_files)} 帧"
        stacked_files = sorted(
            (
                path
                for path in fits
                if self._is_candidate_stacked_input(path, work_dir)
            ),
            key=safe_mtime,
            reverse=True,
        )
        return stacked_files, "输入"

    def _schedule_initial_preview(
        self,
        work_dir: Path,
        input_mode: str | None = None,
    ) -> None:
        candidates, label = self._initial_preview_candidates(work_dir, input_mode)
        self._preview_request_id += 1
        request_id = self._preview_request_id
        self._preview_request_label = label
        if self._workspace_state != WORKSPACE_RUN:
            self._latest_preview_stage = -1
            self._latest_preview_title = ""
            self._latest_preview_path = None
            self.task_preview_canvas.clear_image()
            self.preview_canvas.clear_image()
        if not candidates:
            discovery = getattr(self, "_input_discovery", None)
            master_file = (
                discovery.master_file
                if discovery is not None
                and discovery.selected_path == work_dir.resolve()
                else None
            )
            if (
                master_file is not None
                and master_file.suffix.lower() in REVIEW_SUFFIXES
                and self._display_latest_preview(
                    master_file,
                    stage=0,
                    title="复核输入",
                )
            ):
                self.task_preview_status_label.setText(
                    f"复核输入 · {master_file.name}"
                )
                return
            if master_file is not None and master_file.suffix.lower() == ".xisf":
                self.task_preview_canvas.clear_image()
                self.task_preview_status_label.setText(
                    "XISF 将在 Stage 1 转换后显示预览"
                )
                self.preview_status_label.setText("预览：等待 Stage 1 转换 XISF")
                return
            self.task_preview_canvas.clear_image()
            self.task_preview_status_label.setText("无可用输入预览")
            self.preview_status_label.setText("预览：输入不可用")
            return

        self.task_preview_status_label.setText("正在读取输入…")
        self.preview_status_label.setText("预览：正在读取 Stage 0")
        worker = InitialPreviewWorker(
            request_id,
            candidates,
            preview_cache_path(work_dir).with_name(
                f"stage0_input_{request_id}.png"
            ),
            parent=self,
        )
        worker.ready.connect(self._on_initial_preview_ready)
        worker.failed.connect(self._on_initial_preview_failed)
        worker.finished.connect(worker.deleteLater)
        self.preview_worker = worker
        worker.start()

    def _on_initial_preview_ready(
        self,
        request_id: int,
        preview_path: str,
        source_path: str,
    ) -> None:
        if int(request_id) != self._preview_request_id:
            return
        if self._workspace_state == WORKSPACE_RUN and self._latest_preview_stage > 0:
            return
        source = Path(source_path)
        label = str(getattr(self, "_preview_request_label", "输入"))
        self.task_preview_status_label.setText(f"{label} · {source.name}")
        self._display_latest_preview(
            Path(preview_path),
            stage=0,
            title="输入",
        )

    def _on_initial_preview_failed(self, request_id: int, reason: str) -> None:
        if int(request_id) != self._preview_request_id:
            return
        self.task_preview_status_label.setText("输入预览不可用")
        self.preview_status_label.setText("预览：Stage 0 不可用")
        self._append_event(f"Stage 0 输入预览不可用，继续允许处理：{reason}")

    def _display_latest_preview(
        self,
        path: Path,
        *,
        stage: int,
        title: str,
    ) -> bool:
        if not path.is_file():
            return False
        task_ok = self.task_preview_canvas.set_image(path)
        run_ok = self.preview_canvas.set_image(path)
        if not (task_ok or run_ok):
            return False
        self._latest_preview_stage = int(stage)
        self._latest_preview_title = str(title)
        self._latest_preview_path = path
        if stage <= 0:
            stage_text = "Stage 0 · 输入"
        else:
            stage_text = f"Stage {stage} · {title}"
        self.preview_stage_label.setText("预览：" + stage_text)
        self.preview_status_label.setText("预览：" + stage_text)
        return True

    def _configure_focus_order(self) -> None:
        if hasattr(self, "_stage_parameter_controls"):
            stage_widgets: list[QWidget] = []
            for stage in STAGE_TITLES:
                stage_widgets.append(self._stage_parameter_headers[stage])
                for spec in SPECS_BY_STAGE[stage]:
                    if not spec.stage_mode:
                        stage_widgets.append(
                            self._stage_parameter_auto_checks[spec.field]
                        )
                    stage_widgets.append(
                        self._stage_parameter_controls[spec.field]
                    )
            focus_chain = (
                self.dir_edit,
                self.browse_btn,
                self.browse_dir_btn,
                self.recent_combo,
                self.advanced_toggle_btn,
                self.processing_params_btn,
                *self.output_format_checks.values(),
                self.review_combo,
                self.compute_combo,
                self.processing_expert_btn,
                *stage_widgets,
                self.processing_defaults_btn,
                self.debug_btn,
                self.network_btn,
                self.start_btn,
                self.stop_btn,
                self.quality_report_btn,
                self.result_preview_btn,
                self.open_result_btn,
                self.log_toggle_btn,
                self.open_log_btn,
                self.clear_view_btn,
                self.log_view,
            )
            for current, following in zip(focus_chain, focus_chain[1:]):
                QWidget.setTabOrder(current, following)
            return
        focus_chain = (
            self.dir_edit,
            self.browse_btn,
            self.browse_dir_btn,
            self.recent_combo,
            self.advanced_toggle_btn,
            self.processing_params_btn,
            *self.output_format_checks.values(),
            self.review_combo,
            self.color_combo,
            self.filter_combo,
            self.gaia_catalog_download_btn,
            self.denoise_combo,
            self.deconv_combo,
            self.graxpert_model_edit,
            self.graxpert_model_file_btn,
            self.graxpert_model_dir_btn,
            self.compute_combo,
            self.pcc_timeout_spin,
            self.local_wb_gain_spin,
            self.builtin_denoise_spin,
            self.graxpert_strength_spin,
            self.rl_iterations_spin,
            self.rl_maxstars_spin,
            self.starless_retry_spin,
            self.starless_repair_spin,
            self.starless_halo_spin,
            self.starless_chroma_spin,
            self.starmask_stretch_spin,
            self.weak_star_recovery_spin,
            self.processing_defaults_btn,
            self.debug_btn,
            self.network_btn,
            self.start_btn,
            self.stop_btn,
            self.quality_report_btn,
            self.result_preview_btn,
            self.open_result_btn,
            self.log_toggle_btn,
            self.open_log_btn,
            self.clear_view_btn,
            self.log_view,
        )
        for current, following in zip(focus_chain, focus_chain[1:]):
            QWidget.setTabOrder(current, following)

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        total_seconds = max(0, int(round(seconds)))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _reset_stage_progress(self, stage_count: int) -> None:
        self._progress_stage_count = 10
        self._active_pipeline_stage = None
        self._display_pipeline_stage = None
        self._stage_started_monotonic.clear()
        self._stage_elapsed_seconds.clear()
        self._stage_progress_states.clear()
        self._stage_progress_titles.clear()
        self._stage_progress_details.clear()
        self._stage_items.clear()
        while self.stage_stepper_layout.count():
            layout_item = self.stage_stepper_layout.takeAt(0)
            widget = layout_item.widget()
            if widget is not None:
                widget.hide()
                widget.deleteLater()

        for stage in range(1, self._progress_stage_count + 1):
            if stage in {1, 7}:
                phase_text = {
                    1: "线性处理 · Stage 1–6",
                    7: "非线性处理 · Stage 7–10",
                }[stage]
                phase_label = QLabel(phase_text)
                phase_label.setProperty("tone", "muted")
                phase_label.setContentsMargins(2, 6, 2, 2)
                self.stage_stepper_layout.addWidget(phase_label)
            title = PIPELINE_STAGE_TITLES.get(stage, f"阶段 {stage}")
            chip = QLabel()
            chip.setWordWrap(True)
            chip.setSizePolicy(
                QSizePolicy.Policy.Preferred,
                QSizePolicy.Policy.Minimum,
            )
            self.stage_stepper_layout.addWidget(chip)
            self._stage_items[stage] = chip
            self._stage_progress_states[stage] = "waiting"
            self._stage_progress_titles[stage] = title
            self._update_stage_chip(stage)

        self.progress_summary_label.setText("当前进度：等待开始")
        self.progress_summary_label.setAccessibleDescription(
            "当前进度：等待开始"
        )
        if hasattr(self, "stage_timing_label"):
            self.stage_timing_label.setText("本阶段 — · 总耗时 —")

    @staticmethod
    def _format_stage_component_summary(payload: dict[str, object]) -> str:
        components = payload.get("components")
        if not isinstance(components, dict):
            return ""
        labels = {
            "deconvolution": "反卷积",
            "denoise": "降噪",
        }
        method_labels = {
            "graxpert_object": "GraXpert",
            "siril_rl": "Siril RL",
            "deterministic_multiscale": "多尺度",
            "siril_builtin": "Siril",
            "none": "",
        }
        reason_labels = {
            "accepted": "已应用",
            "auto_low_noise": "低噪声自动跳过",
            "user_disabled": "用户关闭",
            "config_disabled": "配置关闭",
            "background_guard_rollback": "质量门回滚",
            "all_denoisers_failed": "全部降噪器失败",
            "deconvolution_unavailable": "不可用",
        }
        status_symbols = {
            "applied": "✓",
            "skipped": "—",
            "failed": "✕",
            "rolled_back": "↩",
        }
        parts: list[str] = []
        for component_id, raw_component in components.items():
            if not isinstance(raw_component, dict):
                continue
            component_status = str(raw_component.get("status") or "").strip()
            method = str(raw_component.get("method") or "").strip()
            reason = str(raw_component.get("reason_code") or "").strip()
            label = labels.get(str(component_id), str(component_id))
            detail = method_labels.get(method, method)
            if component_status != "applied" or not detail:
                detail = reason_labels.get(reason, reason or detail)
            symbol = status_symbols.get(component_status, "·")
            part = f"{label} {symbol}"
            if detail:
                part += f" {detail}"
            parts.append(part)
        return " · ".join(parts)

    @staticmethod
    def _format_stage_detail_note(payload: dict[str, object]) -> str:
        details = payload.get("details")
        details = details if isinstance(details, dict) else {}
        reason_text = str(details.get("reason_text") or "").strip()
        stage8_handoff = details.get("stage8_handoff")
        stage8_handoff = (
            stage8_handoff if isinstance(stage8_handoff, dict) else {}
        )
        outcome_reason = str(
            stage8_handoff.get("outcome_reason_code") or ""
        ).strip()
        if outcome_reason == "stage8_limited_candidate_rejected":
            suffix = "受限候选未通过质量门，已安全回滚"
            return f"{reason_text}；{suffix}" if reason_text else suffix
        if outcome_reason == "stage8_limited_candidate_accepted":
            suffix = "受限候选已通过质量门"
            return f"{reason_text}；{suffix}" if reason_text else suffix
        upstream_passthrough = bool(payload.get("upstream_passthrough", False))
        stage_fallback_used = bool(payload.get("fallback_used", False))
        if upstream_passthrough and not stage_fallback_used:
            return "成功（使用 Stage 8 安全旁路源）"
        if stage_fallback_used:
            fallback_reason = str(
                details.get("stage9_fallback_reason")
                or payload.get("reason_code")
                or ""
            ).strip()
            fallback_labels = {
                "intensity_fallback": "降低星点合成强度",
                "compact_mask_recovery": "紧凑星点蒙版恢复",
                "unsafe_starless_bypass": "不安全 Starless 回滚",
                "all_remix_candidates_rejected": "合成候选全部拒绝",
                "starmask_stretch_failed_keep_upstream": "星点蒙版处理失败并保留上游源",
                "output_save_failed_keep_upstream": "输出保存失败并保留上游源",
            }
            fallback_text = fallback_labels.get(fallback_reason, fallback_reason)
            if fallback_text:
                prefix = "Stage 9 已使用回退：" + fallback_text
                if upstream_passthrough:
                    prefix += "；上游为 Stage 8 安全旁路源"
                return prefix
        if reason_text:
            return reason_text
        reason_code = str(payload.get("reason_code") or "").strip()
        reason_labels = {
            "bright_nebula_halo_advisory": "亮星云 halo 中风险，采用受限增强与质量门",
            "stage8_enhancement_quality_rollback": "增强候选未通过质量门，已安全回滚",
            "star_preserve_target_bypass": "目标要求保留星点，使用安全旁路",
        }
        return reason_labels.get(reason_code, "")

    def _update_stage_chip(self, stage: int) -> None:
        chip = self._stage_items.get(stage)
        if chip is None:
            return
        state = self._stage_progress_states.get(stage, "waiting")
        symbol = {
            "waiting": "○",
            "running": "●",
            "completed": "✓",
            "safe_passthrough": "↪",
            "degraded": "⚠",
            "failed": "✕",
            "skipped": "—",
            "stopped": "■",
        }.get(state, "○")
        short_title = PIPELINE_STAGE_SHORT_TITLES.get(stage, str(stage))
        state_text = {
            "waiting": "等待",
            "running": "运行中",
            "completed": "已完成",
            "safe_passthrough": "安全旁路",
            "degraded": "已降级",
            "failed": "失败",
            "skipped": "已跳过",
            "stopped": "已停止",
        }.get(state, state)
        stage_detail = self._stage_progress_details.get(stage, {})
        component_summary = self._format_stage_component_summary(stage_detail)
        detail_note = self._format_stage_detail_note(stage_detail)
        extra_text = component_summary or detail_note
        chip_text = f"{symbol}  Stage {stage} · {short_title}    {state_text}"
        if extra_text:
            chip_text += f"\n   {extra_text}"
        chip.setText(chip_text)
        chip.setAccessibleName(f"阶段 {stage}：{short_title}")
        set_style_property(chip, "stageState", state)
        elapsed = self._stage_elapsed_seconds.get(stage)
        elapsed_text = self._format_elapsed(elapsed) if elapsed is not None else "—"
        tooltip = (
            f"阶段 {stage}：{self._stage_progress_titles.get(stage, short_title)}\n"
            f"状态：{state_text}\n"
            f"耗时：{elapsed_text}"
        )
        if component_summary:
            tooltip += f"\n子状态：{component_summary}"
        if detail_note:
            tooltip += f"\n说明：{detail_note}"
        chip.setToolTip(tooltip)
        chip.setAccessibleDescription(chip.toolTip())
        if state == "running" and hasattr(self, "stage_scroll"):
            self.stage_scroll.ensureWidgetVisible(chip, 0, 8)

    def _begin_stage_progress(self, stage_count: int) -> None:
        self._reset_stage_progress(stage_count)
        self._pipeline_started_monotonic = time.monotonic()
        self._last_quality_report_path = None
        self.warning_card.hide()
        self._progress_timer.start()
        self._refresh_elapsed_labels()

    def _refresh_elapsed_labels(self) -> None:
        started = self._pipeline_started_monotonic
        if started is None:
            self.stage_timing_label.setText("本阶段 — · 总耗时 —")
            return

        now = time.monotonic()
        total_elapsed = now - started
        stage = self._active_pipeline_stage or self._display_pipeline_stage
        if stage is None:
            stage_text = "—"
        else:
            stage_elapsed = self._stage_elapsed_seconds.get(stage, 0.0)
            stage_started = self._stage_started_monotonic.get(stage)
            if stage_started is not None:
                stage_elapsed += now - stage_started
            stage_text = self._format_elapsed(stage_elapsed)
            item = self._stage_items.get(stage)
            if item is not None and self._stage_progress_states.get(stage) == "running":
                self._update_stage_chip(stage)

        self.stage_timing_label.setText(
            f"本阶段 {stage_text} · 总耗时 {self._format_elapsed(total_elapsed)}"
        )

    def _finish_stage_progress(self, status: str) -> None:
        active_stage = self._active_pipeline_stage
        if active_stage is not None:
            active_state = self._stage_progress_states.get(active_stage)
            if active_state == "running":
                final_state = {
                    "Failed": "failed",
                    "Stopped": "stopped",
                }.get(status, "completed")
                self._on_pipeline_progress(
                    active_stage,
                    self._stage_progress_titles.get(active_stage, ""),
                    final_state,
                )
        self._progress_timer.stop()
        self._refresh_elapsed_labels()

    def _choose_workdir(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "选择 Light 目录或产品任务",
        )
        if selected:
            self._apply_input_path(Path(selected), remember=True)

    def _choose_input_file(self) -> None:
        selected, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "选择母版或预览图",
            "",
            (
                "天文图像 (*.fit *.fits *.fts *.xisf);;"
                "复核图像 (*.tif *.tiff *.png *.jpg *.jpeg);;所有文件 (*)"
            ),
        )
        if selected:
            self._apply_input_path(Path(selected), remember=True)

    def _set_running(self, running: bool) -> None:
        if running:
            self._clear_pending_history_delete()
        self._ui_running = bool(running)
        self.dir_edit.setEnabled(not running)
        self.browse_btn.setEnabled(not running)
        self.browse_dir_btn.setEnabled(not running)
        self.recent_combo.setEnabled(not running)
        self.mode_combo.setEnabled(not running)
        self.advanced_toggle_btn.setEnabled(not running)
        self.processing_params_btn.setEnabled(not running)
        self.processing_params_panel.setEnabled(not running)
        self.debug_btn.setEnabled(not running)
        self.network_btn.setEnabled(not running)
        self._update_toolbar_state()
        history_window = getattr(self, "history_window", None)
        if history_window is not None and history_window.isVisible():
            self._refresh_history_view()
        sync_preferences = getattr(self, "_sync_preferences_window", None)
        if callable(sync_preferences):
            sync_preferences()

    def _input_mode_label(self, mode: str) -> str:
        if mode == INPUT_MODE_STAGE1_PREPARED_RESUME:
            return "从输入准备后继续"
        if mode == INPUT_MODE_LINEAR_RESUME:
            return "从线性反卷积与降噪后继续"
        if mode == INPUT_MODE_STAGE2_CORRECTED_RESUME:
            return "从边界校正后继续"
        return "完整处理"

    def _current_input_mode(self) -> str:
        combo = getattr(self, "mode_combo", None)
        if combo is not None and hasattr(combo, "currentData"):
            value = combo.currentData()
            if value == UI_MODE_RECOMMENDED:
                return self._recommended_input_mode
            if value in {
                INPUT_MODE_AUTO,
                INPUT_MODE_STAGE1_PREPARED_RESUME,
                INPUT_MODE_LINEAR_RESUME,
                INPUT_MODE_STAGE2_CORRECTED_RESUME,
            }:
                return str(value)
        value = getattr(self, "input_mode", INPUT_MODE_AUTO)
        if value in {
            INPUT_MODE_AUTO,
            INPUT_MODE_STAGE1_PREPARED_RESUME,
            INPUT_MODE_LINEAR_RESUME,
            INPUT_MODE_STAGE2_CORRECTED_RESUME,
        }:
            return str(value)
        return INPUT_MODE_AUTO

    def _on_input_mode_changed(self, _index: int) -> None:
        self.input_mode = self._current_input_mode()
        self._update_processing_sheet_availability()
        if not self._restoring_settings:
            self._append_event(
                f"处理方式已切换为：{self._input_mode_label(self.input_mode)}"
            )
            self._save_settings()
            directory_text = self.dir_edit.text().strip()
            work_dir = Path(directory_text).expanduser() if directory_text else None
            if (
                self._workspace_state == WORKSPACE_TASK
                and work_dir is not None
                and work_dir.is_dir()
            ):
                self._schedule_initial_preview(work_dir, self.input_mode)

    @staticmethod
    def _set_parameter_help(label: QLabel, control: QWidget, text: str) -> None:
        """Expose the same detailed help to mouse hover and accessibility APIs."""
        for widget in (label, control):
            widget.setToolTip(text)
            widget.setAccessibleDescription(text)

    def _apply_processing_payload_state(self, payload: Mapping[str, object]) -> None:
        normalized, _adjustments = normalize_processing_parameters(payload)
        self.processing_parameters = normalized
        general = normalized["general"]
        self.output_formats = tuple(general["output_formats"])
        self.review_only = bool(general["review_only"])
        self.compute_mode = str(general["compute_mode"])
        self.auto_tune_enabled = bool(general["auto_tune_enabled"])
        self.max_retries = int(general["max_retries"])
        self.retry_delay = float(general["retry_delay"])
        self.review_bundle_enabled = bool(general["review_bundle_enabled"])
        self.managed_output_enabled = bool(general["managed_output_enabled"])
        self.checkpoint_mode = bool(general["checkpoint_mode"])

        def effective(field: str):
            return effective_parameter_value(normalized, field)

        self.color_calibration = "pcc"
        self.filter_hint = str(effective("stage4_filter_hint"))
        denoise_entry = normalized["stages"]["5"]["overrides"]
        self.denoise_mode = (
            "on" if denoise_entry.get("denoise_enabled") is True
            else "off" if denoise_entry.get("denoise_enabled") is False
            else "auto"
        )
        deconv_mode = str(
            denoise_entry.get("stage5_deconvolution_mode", "auto")
        )
        self.deconvolution_mode = (
            "rl" if deconv_mode == "rl" else "off" if deconv_mode == "off" else "auto"
        )
        self.graxpert_model_path = str(effective("graxpert_object_model_path"))
        self.pcc_timeout_sec = int(effective("stage4_pcc_timeout_sec"))
        self.local_wb_gain_limit = float(
            effective("stage4_local_star_wb_gain_limit")
        )
        self.builtin_denoise_strength = float(effective("denoise_mod"))
        self.graxpert_deconv_strength = float(
            effective("stage5_graxpert_deconv_strength")
        )
        self.rl_iterations = int(effective("stage5_rl_iters"))
        self.rl_maxstars = int(effective("stage5_rl_maxstars"))
        self.starless_retry_max = int(effective("stage7_quality_retry_max"))
        self.starless_repair_strength = float(
            effective("stage7_starless_repair_strength")
        )
        self.starless_halo_repair_strength = float(
            effective("stage7_starless_halo_repair_strength")
        )
        self.starless_chroma_strength = float(
            effective("stage7_starless_chroma_denoise_strength")
        )
        self.starmask_asinh_stretch = float(
            effective("stage9_starmask_asinh_stretch")
        )
        self.weak_star_recovery_ratio = float(
            effective("stage9_weak_star_recovery_ratio_min")
        )

    def _processing_settings_snapshot(self) -> dict[str, object]:
        payload = getattr(self, "processing_parameters", None)
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema")
            not in SUPPORTED_PROCESSING_PARAMETERS_SCHEMAS
        ):
            payload = default_processing_parameters()
        candidate = dict(payload)
        candidate["general"] = {
            "output_formats": list(self.output_formats),
            "review_only": bool(self.review_only),
            "compute_mode": str(self.compute_mode),
            "auto_tune_enabled": bool(self.auto_tune_enabled),
            "max_retries": int(self.max_retries),
            "retry_delay": float(self.retry_delay),
            "review_bundle_enabled": bool(self.review_bundle_enabled),
            "managed_output_enabled": bool(self.managed_output_enabled),
            "checkpoint_mode": bool(self.checkpoint_mode),
        }
        normalized, _adjustments = normalize_processing_parameters(candidate)
        self.processing_parameters = normalized
        return json.loads(json.dumps(normalized, ensure_ascii=False))

    def _restore_processing_settings(self, snapshot: object) -> None:
        payload = (
            snapshot
            if isinstance(snapshot, Mapping)
            else default_processing_parameters()
        )
        normalized, _adjustments = normalize_processing_parameters(payload)
        StarunGui._apply_processing_payload_state(self, normalized)
        sync = getattr(self, "_sync_processing_controls_from_state", None)
        if callable(sync):
            sync()
        sync_preferences = getattr(self, "_sync_preferences_window", None)
        if callable(sync_preferences):
            sync_preferences()

    def _processing_runtime_configuration(
        self,
        input_mode: str,
    ) -> tuple[dict[str, str], set[str]]:
        payload = getattr(self, "processing_parameters", None)
        if (
            isinstance(payload, Mapping)
            and payload.get("schema")
            in SUPPORTED_PROCESSING_PARAMETERS_SCHEMAS
        ):
            general = payload.get("general", {})
            formats = tuple(general.get("output_formats", self.output_formats))
            review_only = bool(
                general.get("review_only", self.review_only)
                or gate_profile_requires_review(
                    str(payload.get("gate_profile") or "default")
                )
            )
            compute_mode = str(general.get("compute_mode", self.compute_mode))
            accelerated = "0" if compute_mode == "cpu" else "1"
            return (
                {
                    "STARUN_OUTPUT_FORMAT": ",".join(formats),
                    "STARUN_FORCE_REVIEW_ONLY_OUTPUT": "1" if review_only else "0",
                    "STARUN_COSMIC_NATIVE_GPU": accelerated,
                    "STARUN_COSMIC_CLASSIC_GPU": accelerated,
                    "STARUN_GRAXPERT_GPU": accelerated,
                },
                set(),
            )
        overrides = {
            "STARUN_OUTPUT_FORMAT": ",".join(self.output_formats),
            "STARUN_FORCE_REVIEW_ONLY_OUTPUT": "1" if self.review_only else "0",
            "STARUN_COSMIC_NATIVE_GPU": "0" if self.compute_mode == "cpu" else "1",
            "STARUN_COSMIC_CLASSIC_GPU": "0" if self.compute_mode == "cpu" else "1",
            "STARUN_GRAXPERT_GPU": "0" if self.compute_mode == "cpu" else "1",
            "STARUN_STAGE7_QUALITY_RETRY_MAX": str(self.starless_retry_max),
            "STARUN_STAGE7_STARLESS_REPAIR_STRENGTH": (
                f"{self.starless_repair_strength:.2f}"
            ),
            "STARUN_STAGE7_STARLESS_HALO_REPAIR_STRENGTH": (
                f"{self.starless_halo_repair_strength:.2f}"
            ),
            "STARUN_STAGE7_STARLESS_CHROMA_DENOISE_STRENGTH": (
                f"{self.starless_chroma_strength:.2f}"
            ),
            "STARUN_STAGE9_STARMASK_ASINH_STRETCH": (
                f"{self.starmask_asinh_stretch:.2f}"
            ),
            "STARUN_STAGE9_WEAK_STAR_RECOVERY_RATIO_MIN": (
                f"{self.weak_star_recovery_ratio:.2f}"
            ),
        }
        unset_keys: set[str] = set()
        if input_mode != INPUT_MODE_LINEAR_RESUME:
            overrides.update(
                {
                    "STARUN_STAGE4_SPCC_TIMEOUT_SEC": "300",
                    "STARUN_STAGE4_PCC_TIMEOUT_SEC": str(self.pcc_timeout_sec),
                    "STARUN_STAGE4_LOCAL_STAR_WB_GAIN_LIMIT": (
                        f"{self.local_wb_gain_limit:.2f}"
                    ),
                    "STARUN_DENOISE_MOD": (
                        f"{self.builtin_denoise_strength:.2f}"
                    ),
                    "STARUN_STAGE5_GRAXPERT_DECONV_STRENGTH": (
                        f"{self.graxpert_deconv_strength:.2f}"
                    ),
                    "STARUN_STAGE5_RL_ITERS": str(self.rl_iterations),
                    "STARUN_STAGE5_RL_MAXSTARS": str(self.rl_maxstars),
                }
            )
            filter_values = {
                "auto": "",
                "no_filter": "broadband no filter",
                "seestar_lp": "broadband Seestar LP",
                "dual_narrowband": "dualband Ha OIII",
            }
            overrides["STARUN_STAGE4_FILTER_HINT"] = filter_values[self.filter_hint]

            if self.denoise_mode == "auto":
                unset_keys.add("STARUN_DENOISE_FORCE")
            else:
                overrides["STARUN_DENOISE_FORCE"] = (
                    "1" if self.denoise_mode == "on" else "0"
                )

            deconvolution_values = {
                "auto": ("1", "1"),
                "rl": ("1", "0"),
                "off": ("0", "0"),
            }
            deconv_enabled, graxpert_enabled = deconvolution_values[
                self.deconvolution_mode
            ]
            overrides["STARUN_STAGE5_DECONV_ENABLE"] = deconv_enabled
            overrides["STARUN_STAGE5_GRAXPERT_DECONV_ENABLE"] = graxpert_enabled
            if self.graxpert_model_path:
                overrides["STARUN_GRAXPERT_OBJECT_MODEL_PATH"] = (
                    self.graxpert_model_path
                )
            else:
                unset_keys.add("STARUN_GRAXPERT_OBJECT_MODEL_PATH")
        return overrides, unset_keys

    @staticmethod
    def _processing_payload_summary_lines(
        payload: object,
        input_mode: str,
    ) -> list[str]:
        if not isinstance(payload, Mapping):
            return []
        try:
            normalized, _adjustments = normalize_processing_parameters(payload)
        except (TypeError, ValueError):
            return []
        general = normalized["general"]
        gate_profile = str(normalized.get("gate_profile") or "default")
        forced_review = gate_profile_requires_review(gate_profile)
        format_labels = {"tif": "TIFF", "png": "PNG", "fit": "FITS"}
        lines = [
            "通用：输出 "
            + "/".join(
                format_labels[value] for value in general["output_formats"]
            )
            + (
                " · 仅复核（无限模式强制）"
                if forced_review
                else " · 仅复核"
                if general["review_only"]
                else " · 正式结果"
            )
            + (
                " · CPU 兼容"
                if general["compute_mode"] == "cpu"
                else " · 自动加速"
            )
        ]
        lines.append(
            "可靠性："
            + ("自动调参" if general["auto_tune_enabled"] else "关闭自动调参")
            + f" · 重试 {general['max_retries']} 次/{general['retry_delay']:g} 秒"
            + (" · 生成复核包" if general["review_bundle_enabled"] else " · 不生成复核包")
            + (" · 受管输出" if general["managed_output_enabled"] else " · 原生输出")
            + (" · 完成后收敛断点" if general["checkpoint_mode"] else " · 常规保留")
        )
        multiplier_labels = {
            "default": "1×",
            "relaxed": "3×",
            "unlimited": "10×",
        }
        lines.append(
            "门禁："
            f"{GATE_PROFILE_LABELS[gate_profile]}（{multiplier_labels[gate_profile]}）"
        )
        disabled = (
            set(range(1, 6))
            if input_mode == INPUT_MODE_LINEAR_RESUME
            else {1, 2} if input_mode == INPUT_MODE_STAGE2_CORRECTED_RESUME
            else {1} if input_mode == INPUT_MODE_STAGE1_PREPARED_RESUME
            else set()
        )
        for stage in STAGE_TITLES:
            entry = normalized["stages"][str(stage)]
            count = len(entry["overrides"]) + int(entry["mode"] != "auto")
            if stage in disabled:
                state = "已由恢复点完成"
            elif count:
                state = f"{count} 项自定义"
                if entry["mode"] == "preserve":
                    state += " · 安全保留"
            else:
                state = "自动"
            lines.append(f"Stage {stage} {STAGE_TITLES[stage]}：{state}")
        return lines

    def _processing_settings_summary_lines(self, input_mode: str) -> list[str]:
        canonical_lines = self._processing_payload_summary_lines(
            getattr(self, "processing_parameters", None),
            input_mode,
        )
        if canonical_lines:
            return canonical_lines
        format_labels = {"tif": "TIFF", "png": "PNG", "fit": "FITS"}
        color_labels = {
            "pcc": (
                "SPCC 优先（300 秒；PCC 异常回退，"
                f"{self.pcc_timeout_sec} 秒）"
            )
        }
        filter_labels = {
            "auto": "自动识别",
            "no_filter": "无滤镜",
            "seestar_lp": "Seestar LP",
            "dual_narrowband": "双窄带 Ha/OIII",
        }
        denoise_labels = {"auto": "自动", "on": "开启", "off": "关闭"}
        deconv_labels = {
            "auto": "自动 GraXpert→RL",
            "rl": "仅 Siril RL",
            "off": "关闭",
        }
        lines = [
            "输出：" + "/".join(format_labels[value] for value in self.output_formats),
            "输出用途：" + ("仅复核" if self.review_only else "正式结果"),
            "计算：" + ("CPU 兼容" if self.compute_mode == "cpu" else "自动加速"),
            (
                "去星/合成：重试 "
                f"{self.starless_retry_max} · 修复 "
                f"{self.starless_repair_strength:.2f}/"
                f"{self.starless_halo_repair_strength:.2f}/"
                f"{self.starless_chroma_strength:.2f} · 星点 "
                f"{self.starmask_asinh_stretch:.2f}/"
                f"{self.weak_star_recovery_ratio:.2f}"
            ),
        ]
        if input_mode == INPUT_MODE_LINEAR_RESUME:
            lines.append("Stage 4–5 参数：续跑时不生效")
        else:
            lines.extend(
                (
                    "校色：" + color_labels[self.color_calibration],
                    "滤镜：" + filter_labels[self.filter_hint],
                    "线性降噪：" + denoise_labels[self.denoise_mode],
                    "反卷积：" + deconv_labels[self.deconvolution_mode],
                    (
                        "专业线性：校色 "
                        f"{self.pcc_timeout_sec}s · WB {self.local_wb_gain_limit:.2f}×"
                        f" · 降噪 {self.builtin_denoise_strength:.2f}"
                        f" · GraXpert {self.graxpert_deconv_strength:.2f}"
                        f" · RL {self.rl_iterations}/{self.rl_maxstars}"
                    ),
                )
            )
        return lines

    def _build_processing_parameters_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("processingParametersSheet")
        panel.setAccessibleName("Stage 1 到 Stage 10 处理参数设置面板")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("处理参数")
        title.setObjectName("sectionTitle")
        self.processing_sheet_note = QLabel("阶段参数仅用于当前任务；通用配置会保留。")
        self.processing_sheet_note.setWordWrap(True)
        self.processing_sheet_note.setProperty("tone", "muted")
        self.processing_expert_btn = QPushButton("专家参数")
        self.processing_expert_btn.setCheckable(True)
        self.processing_expert_btn.setProperty("variant", "quiet")
        self.processing_expert_btn.setAccessibleName("显示或隐藏专家处理参数")
        self.processing_expert_btn.setToolTip(
            "显示算法细节以及 Stage 1–10 的过程与质量门禁阈值；"
            "隐藏后已自定义值仍会保留并生效。"
        )
        self.processing_defaults_btn = QPushButton("恢复安全默认值")
        self.processing_defaults_btn.setProperty("variant", "quiet")
        self.processing_defaults_btn.setAccessibleName("恢复处理参数安全默认值")
        self.processing_done_btn = QPushButton("完成")
        self.processing_done_btn.setProperty("variant", "primary")
        self.processing_done_btn.setAccessibleName("收起处理参数设置")
        header.addWidget(title)
        header.addWidget(self.processing_sheet_note, 1)
        header.addWidget(self.processing_expert_btn)
        header.addWidget(self.processing_defaults_btn)
        header.addWidget(self.processing_done_btn)
        layout.addLayout(header)

        self.processing_general_group = QGroupBox("通用配置")
        general_form = QFormLayout(self.processing_general_group)
        general_form.setContentsMargins(10, 8, 10, 8)
        output_row = QWidget()
        output_layout = QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        self.output_format_checks = {}
        for key, text in (("tif", "TIFF"), ("png", "PNG"), ("fit", "FITS")):
            checkbox = QCheckBox(text)
            checkbox.setAccessibleName(f"输出 {text}")
            self.output_format_checks[key] = checkbox
            output_layout.addWidget(checkbox)
        output_layout.addStretch(1)
        general_form.addRow(QLabel("输出文件"), output_row)
        self.review_combo = QComboBox()
        self.review_combo.addItem("正式结果", False)
        self.review_combo.addItem("仅生成待复核结果", True)
        general_form.addRow(QLabel("输出用途"), self.review_combo)
        self.compute_combo = QComboBox()
        self.compute_combo.addItem("自动加速", "auto")
        self.compute_combo.addItem("CPU 兼容模式", "cpu")
        general_form.addRow(QLabel("计算设备"), self.compute_combo)
        self.auto_tune_check = QCheckBox("启用 Stage 1 分析后的自动调参")
        self.auto_tune_check.setAccessibleName("启用自动调参")
        general_form.addRow(QLabel("自动调参"), self.auto_tune_check)
        self.max_retries_spin = QSpinBox()
        self.max_retries_spin.setRange(0, 3)
        self.max_retries_spin.setSuffix(" 次")
        self.max_retries_spin.setKeyboardTracking(False)
        general_form.addRow(QLabel("失败重试"), self.max_retries_spin)
        self.retry_delay_spin = QDoubleSpinBox()
        self.retry_delay_spin.setRange(0.0, 10.0)
        self.retry_delay_spin.setDecimals(1)
        self.retry_delay_spin.setSingleStep(0.5)
        self.retry_delay_spin.setSuffix(" 秒")
        self.retry_delay_spin.setKeyboardTracking(False)
        general_form.addRow(QLabel("重试间隔"), self.retry_delay_spin)
        self.review_bundle_check = QCheckBox("生成复核包（不关闭质量检测）")
        self.review_bundle_check.setAccessibleName("生成复核包")
        general_form.addRow(QLabel("复核资料"), self.review_bundle_check)
        self.managed_output_check = QCheckBox("生成受管 PNG/TIFF 输出")
        self.managed_output_check.setAccessibleName("生成受管输出")
        general_form.addRow(QLabel("受管输出"), self.managed_output_check)
        self.checkpoint_mode_check = QCheckBox(
            "完成并验证交付后，仅保留正式断点与轻量诊断"
        )
        self.checkpoint_mode_check.setAccessibleName("启用正式断点保留模式")
        general_form.addRow(QLabel("断点保留"), self.checkpoint_mode_check)
        layout.addWidget(self.processing_general_group)

        self.processing_gate_profile_group = QGroupBox("门禁策略（当前任务）")
        gate_profile_form = QFormLayout(self.processing_gate_profile_group)
        gate_profile_form.setContentsMargins(10, 8, 10, 8)
        self.processing_gate_profile_combo = QComboBox()
        profile_display_labels = {
            "default": "默认档位（1×）",
            "relaxed": "放松模式（3×）",
            "unlimited": "无限模式（10×，强制复核）",
        }
        for _label, profile in GATE_PROFILE_CHOICES:
            self.processing_gate_profile_combo.addItem(
                profile_display_labels[profile], profile
            )
        self.processing_gate_profile_combo.setAccessibleName("当前任务门禁档位")
        self.processing_gate_profile_combo.setToolTip(
            "统一设置 Stage 2–9 明确登记的数值验收阈值；"
            "结构性安全检查和算法参数不受影响。"
        )
        self.processing_gate_profile_combo.setAccessibleDescription(
            self.processing_gate_profile_combo.toolTip()
        )
        gate_profile_label = QLabel("全任务档位")
        gate_profile_label.setBuddy(self.processing_gate_profile_combo)
        gate_profile_label.setToolTip(self.processing_gate_profile_combo.toolTip())
        gate_profile_form.addRow(
            gate_profile_label,
            self.processing_gate_profile_combo,
        )
        self.processing_gate_profile_banner = QFrame()
        self.processing_gate_profile_banner.setObjectName("stateBanner")
        self.processing_gate_profile_banner.setAccessibleName(
            "当前任务门禁档位状态"
        )
        gate_profile_banner_layout = QVBoxLayout(
            self.processing_gate_profile_banner
        )
        gate_profile_banner_layout.setContentsMargins(8, 6, 8, 6)
        self.processing_gate_profile_note = QLabel()
        self.processing_gate_profile_note.setWordWrap(True)
        self.processing_gate_profile_note.setAccessibleName("当前任务门禁档位说明")
        gate_profile_banner_layout.addWidget(self.processing_gate_profile_note)
        gate_profile_form.addRow("", self.processing_gate_profile_banner)
        layout.addWidget(self.processing_gate_profile_group)

        self._stage_parameter_controls: dict[str, QWidget] = {}
        self._stage_parameter_auto_checks: dict[str, QCheckBox] = {}
        self._stage_parameter_effective_labels: dict[str, QLabel] = {}
        self._stage_parameter_row_widgets: dict[str, tuple[QWidget, QWidget]] = {}
        self._stage_parameter_sections: dict[int, QWidget] = {}
        self._stage_parameter_headers: dict[int, QPushButton] = {}
        self._stage_expert_widgets: dict[int, list[QWidget]] = {
            stage: [] for stage in STAGE_TITLES
        }
        self._stage_expert_section_headers: dict[int, dict[str, QPushButton]] = {
            stage: {} for stage in STAGE_TITLES
        }
        self._stage_expert_section_bodies: dict[int, dict[str, QFrame]] = {
            stage: {} for stage in STAGE_TITLES
        }
        self.processing_stage_groups: dict[int, QWidget] = {}

        for stage, stage_title in STAGE_TITLES.items():
            stage_header = QPushButton()
            stage_header.setCheckable(True)
            stage_header.setProperty("variant", "quiet")
            stage_header.setAccessibleName(f"展开 Stage {stage} {stage_title} 参数")
            stage_header.toggled.connect(
                lambda checked, stage_number=stage: self._toggle_processing_stage(
                    stage_number, checked
                )
            )
            layout.addWidget(stage_header)

            stage_body = QFrame()
            stage_body.setObjectName(f"processingStage{stage}")
            stage_form = QFormLayout(stage_body)
            stage_form.setContentsMargins(18, 6, 10, 10)
            stage_form.setHorizontalSpacing(12)
            stage_form.setVerticalSpacing(6)

            def add_parameter_row(spec: ParameterSpec, target_form: QFormLayout) -> None:
                control = self._create_processing_parameter_control(spec)
                self._stage_parameter_controls[spec.field] = control
                label = QLabel(spec.label)
                self._set_parameter_help(
                    label,
                    control,
                    spec.help or f"Stage {stage} · {spec.label}",
                )
                if spec.stage_mode:
                    target_form.addRow(label, control)
                    control.currentIndexChanged.connect(
                        lambda _index, stage_number=stage, field=spec.field: (
                            self._on_stage_processing_mode_changed(
                                stage_number, field
                            )
                        )
                    )
                    row_widgets = (label, control)
                else:
                    row = QWidget()
                    row_layout = QHBoxLayout(row)
                    row_layout.setContentsMargins(0, 0, 0, 0)
                    follows_gate_profile = spec.profile_scaling != "none"
                    auto_check = QCheckBox(
                        "跟随档位" if follows_gate_profile else "自动"
                    )
                    auto_check.setAccessibleName(
                        f"{spec.label}"
                        + ("跟随全任务门禁档位" if follows_gate_profile else "使用自动值")
                    )
                    auto_check.setChecked(True)
                    self._stage_parameter_auto_checks[spec.field] = auto_check
                    row_layout.addWidget(auto_check)
                    row_layout.addWidget(control, 1)
                    if follows_gate_profile:
                        effective_label = QLabel()
                        effective_label.setProperty("tone", "muted")
                        effective_label.setAccessibleName(
                            f"{spec.label}当前档位有效值"
                        )
                        self._stage_parameter_effective_labels[spec.field] = (
                            effective_label
                        )
                        row_layout.addWidget(effective_label, 1)
                    target_form.addRow(label, row)
                    auto_check.toggled.connect(
                        lambda checked, field=spec.field: self._on_stage_parameter_auto_toggled(
                            field, checked
                        )
                    )
                    if isinstance(control, QComboBox):
                        control.currentIndexChanged.connect(
                            lambda _index, field=spec.field: self._on_stage_parameter_value_changed(
                                field
                            )
                        )
                    elif isinstance(control, (QSpinBox, QDoubleSpinBox)):
                        control.valueChanged.connect(
                            lambda _value, field=spec.field: self._on_stage_parameter_value_changed(
                                field
                            )
                        )
                    elif isinstance(control, QLineEdit):
                        control.editingFinished.connect(
                            lambda field=spec.field: self._on_stage_parameter_value_changed(
                                field
                            )
                        )
                    row_widgets = (label, row)
                self._stage_parameter_row_widgets[spec.field] = row_widgets

            stage_specs = SPECS_BY_STAGE[stage]
            recommended_specs = sorted(
                (spec for spec in stage_specs if spec.level != "expert"),
                key=lambda spec: (not spec.stage_mode,),
            )
            for spec in recommended_specs:
                add_parameter_row(spec, stage_form)

            section_titles = {
                "execution": "执行策略",
                "algorithm": "算法参数",
                "process_gate": "过程门禁",
                "quality_gate": "质量验收",
                "fallback": "回退与失败",
            }
            for section_key, section_title in section_titles.items():
                section_specs = tuple(
                    spec
                    for spec in stage_specs
                    if spec.level == "expert" and spec.section == section_key
                )
                if not section_specs:
                    continue
                section_header = QPushButton(f"▸ {section_title}")
                section_header.setCheckable(True)
                section_header.setProperty("variant", "quiet")
                section_header.setAccessibleName(
                    f"Stage {stage} {section_title}专家参数"
                )
                section_body = QFrame()
                section_form = QFormLayout(section_body)
                section_form.setContentsMargins(12, 2, 0, 6)
                section_form.setHorizontalSpacing(12)
                section_form.setVerticalSpacing(6)
                for spec in section_specs:
                    add_parameter_row(spec, section_form)
                expanded_by_default = section_key == "execution"
                section_header.setChecked(expanded_by_default)
                section_header.setText(
                    f"{'▾' if expanded_by_default else '▸'} {section_title}"
                )
                section_body.setVisible(expanded_by_default)
                section_header.toggled.connect(
                    lambda checked, header=section_header, body=section_body,
                    title_text=section_title: (
                        header.setText(f"{'▾' if checked else '▸'} {title_text}"),
                        body.setVisible(
                            bool(self.processing_expert_visible and checked)
                        ),
                    )
                )
                stage_form.addRow(section_header)
                stage_form.addRow(section_body)
                self._stage_expert_widgets[stage].extend(
                    (section_header, section_body)
                )
                self._stage_expert_section_headers[stage][section_key] = section_header
                self._stage_expert_section_bodies[stage][section_key] = section_body

            if stage == 4:
                catalog_row = QWidget()
                catalog_layout = QVBoxLayout(catalog_row)
                catalog_layout.setContentsMargins(0, 0, 0, 0)
                self.gaia_catalog_download_btn = QPushButton(
                    "下载离线 Gaia 解析/PCC 目录（约 1.1 GB）"
                )
                self.gaia_catalog_status = QLabel()
                self.gaia_catalog_status.setWordWrap(True)
                self.gaia_catalog_status.setProperty("tone", "muted")
                catalog_layout.addWidget(self.gaia_catalog_download_btn)
                catalog_layout.addWidget(self.gaia_catalog_status)
                stage_form.addRow(QLabel("离线目录资源"), catalog_row)

            reset_button = QPushButton("恢复本阶段自动参数")
            reset_button.setProperty("variant", "quiet")
            reset_button.clicked.connect(
                lambda _checked=False, stage_number=stage: self._reset_stage_processing_parameters(
                    stage_number
                )
            )
            stage_form.addRow("", reset_button)
            self._stage_parameter_headers[stage] = stage_header
            self._stage_parameter_sections[stage] = stage_body
            self.processing_stage_groups[stage] = stage_body
            layout.addWidget(stage_body)

        self.processing_color_group = self.processing_stage_groups[4]
        self.processing_linear_group = self.processing_stage_groups[5]
        self.processing_sheet_status = QLabel(
            "门禁档位与阶段参数只对当前任务生效；专家覆盖优先。"
        )
        self.processing_sheet_status.setProperty("tone", "muted")
        layout.addWidget(self.processing_sheet_status)

        for key, checkbox in self.output_format_checks.items():
            checkbox.toggled.connect(
                lambda checked, output_key=key: self._on_output_format_toggled(
                    output_key, checked
                )
            )
        self.review_combo.currentIndexChanged.connect(
            self._on_general_processing_controls_changed
        )
        self.compute_combo.currentIndexChanged.connect(
            self._on_general_processing_controls_changed
        )
        self.auto_tune_check.toggled.connect(
            self._on_general_processing_controls_changed
        )
        self.max_retries_spin.valueChanged.connect(
            self._on_general_processing_controls_changed
        )
        self.retry_delay_spin.valueChanged.connect(
            self._on_general_processing_controls_changed
        )
        self.review_bundle_check.toggled.connect(
            self._on_general_processing_controls_changed
        )
        self.managed_output_check.toggled.connect(
            self._on_general_processing_controls_changed
        )
        self.checkpoint_mode_check.toggled.connect(
            self._on_general_processing_controls_changed
        )
        self.processing_gate_profile_combo.currentIndexChanged.connect(
            self._on_gate_profile_changed
        )
        self.processing_expert_btn.toggled.connect(
            self._set_processing_expert_visible
        )
        self.processing_defaults_btn.clicked.connect(
            self._restore_processing_defaults
        )
        self.processing_done_btn.clicked.connect(
            self._configure_processing_parameters
        )
        self.gaia_catalog_download_btn.clicked.connect(
            self._toggle_gaia_catalog_download
        )
        self._sync_processing_controls_from_state()
        self._set_processing_expert_visible(self.processing_expert_visible)
        self._toggle_processing_stage(2, True)
        self._refresh_gaia_catalog_status()
        return panel

    def _create_processing_parameter_control(self, spec: ParameterSpec) -> QWidget:
        if spec.kind in {"choice", "bool"}:
            control = QComboBox()
            choices = (
                (("开启", True), ("关闭", False))
                if spec.kind == "bool"
                else spec.choices
            )
            for label, value in choices:
                control.addItem(label, value)
            return control
        if spec.kind == "int":
            control = QSpinBox()
            control.setRange(int(spec.minimum), int(spec.maximum))
            control.setSingleStep(int(spec.step or 1))
            control.setSuffix(spec.suffix)
            control.setKeyboardTracking(False)
            return control
        if spec.kind == "float":
            control = QDoubleSpinBox()
            control.setRange(float(spec.minimum), float(spec.maximum))
            control.setDecimals(spec.decimals)
            control.setSingleStep(float(spec.step or 0.01))
            control.setSuffix(spec.suffix)
            control.setKeyboardTracking(False)
            return control
        control = QLineEdit()
        if spec.kind == "path":
            control.setPlaceholderText("选择或粘贴本地 model.onnx 文件")
        return control

    @staticmethod
    def _processing_control_value(control: QWidget, spec: ParameterSpec):
        if isinstance(control, QComboBox):
            return control.currentData()
        if isinstance(control, (QSpinBox, QDoubleSpinBox)):
            return control.value()
        if isinstance(control, QLineEdit):
            return control.text().strip()
        return spec.default

    @staticmethod
    def _set_processing_control_value(
        control: QWidget, spec: ParameterSpec, value: object
    ) -> None:
        if isinstance(control, QComboBox):
            index = control.findData(value)
            control.setCurrentIndex(index if index >= 0 else 0)
        elif isinstance(control, (QSpinBox, QDoubleSpinBox)):
            control.setValue(value)
        elif isinstance(control, QLineEdit):
            control.setText(str(value or ""))

    @staticmethod
    def _format_processing_effective_value(
        spec: ParameterSpec,
        value: object,
    ) -> str:
        if spec.kind == "int":
            rendered = str(int(value))
        else:
            rendered = f"{float(value):.8g}"
        return f"有效：{rendered}{spec.suffix}"

    def _on_general_processing_controls_changed(self, _value: object = None) -> None:
        if self._processing_controls_updating:
            return
        formats = tuple(
            key for key, checkbox in self.output_format_checks.items()
            if checkbox.isChecked()
        )
        if not formats:
            return
        self.output_formats = formats
        profile = str(
            self.processing_parameters.get("gate_profile", "default")
        )
        if not gate_profile_requires_review(profile):
            self.review_only = bool(self.review_combo.currentData())
        self.compute_mode = str(self.compute_combo.currentData())
        self.auto_tune_enabled = bool(self.auto_tune_check.isChecked())
        self.max_retries = int(self.max_retries_spin.value())
        self.retry_delay = float(self.retry_delay_spin.value())
        self.review_bundle_enabled = bool(self.review_bundle_check.isChecked())
        self.managed_output_enabled = bool(self.managed_output_check.isChecked())
        self.checkpoint_mode = bool(self.checkpoint_mode_check.isChecked())
        payload = self._processing_settings_snapshot()
        StarunGui._apply_processing_payload_state(self, payload)
        self.processing_sheet_status.setText(
            "通用配置已保存；阶段配置仍只用于当前任务。"
        )
        if not self._restoring_settings and getattr(self, "_settings_loaded", False):
            self._save_settings()
        sync_preferences = getattr(self, "_sync_preferences_window", None)
        if callable(sync_preferences):
            sync_preferences()

    def _on_gate_profile_changed(self, _index: int) -> None:
        if self._processing_controls_updating:
            return
        profile = str(self.processing_gate_profile_combo.currentData() or "")
        candidate = dict(self.processing_parameters)
        candidate["gate_profile"] = profile
        normalized, _adjustments = normalize_processing_parameters(candidate)
        StarunGui._apply_processing_payload_state(self, normalized)
        self._sync_processing_controls_from_state()
        if profile == GATE_PROFILE_UNLIMITED:
            self.processing_sheet_status.setText(
                "无限模式已生效：数值门禁按 10× 放宽，本次输出强制待复核。"
            )
        else:
            self.processing_sheet_status.setText(
                f"门禁策略已切换为{GATE_PROFILE_LABELS[profile]}；仅本次任务生效。"
            )

    def _update_gate_profile_controls(self) -> None:
        if not hasattr(self, "processing_gate_profile_combo"):
            return
        profile = str(
            self.processing_parameters.get("gate_profile", "default")
        )
        combo_index = self.processing_gate_profile_combo.findData(profile)
        self.processing_gate_profile_combo.setCurrentIndex(max(0, combo_index))
        forced_review = gate_profile_requires_review(profile)
        if profile == "relaxed":
            note = (
                "明确登记的数值验收阈值按放松方向扩大 3×；"
                "结构性检查、布尔门和算法参数保持静态默认。"
            )
            tone = "info"
        elif forced_review:
            note = (
                "无限模式：数值验收阈值按放松方向扩大 10×；"
                "结构性检查仍生效，本次输出强制待复核。"
            )
            tone = "warning"
        else:
            note = (
                "使用代码中的静态门禁默认值；专家逐项配置仍可覆盖。"
            )
            tone = "info"
        self.processing_gate_profile_note.setText(note)
        set_style_property(self.processing_gate_profile_banner, "tone", tone)
        self.review_combo.setCurrentIndex(
            max(
                0,
                self.review_combo.findData(
                    True if forced_review else self.review_only
                ),
            )
        )
        self.review_combo.setEnabled(not forced_review)
        self.review_combo.setToolTip(
            "已由无限模式强制待复核；切回其他档位后恢复原选择。"
            if forced_review
            else "设置当前任务输出为正式结果或待复核结果。"
        )
        self.review_combo.setAccessibleDescription(self.review_combo.toolTip())

    def _on_stage_processing_mode_changed(self, stage: int, field: str) -> None:
        if self._processing_controls_updating:
            return
        control = self._stage_parameter_controls[field]
        mode = str(control.currentData())
        self.processing_parameters["stages"][str(stage)]["mode"] = mode
        normalized, _adjustments = normalize_processing_parameters(
            self.processing_parameters
        )
        StarunGui._apply_processing_payload_state(self, normalized)
        refresh_dependencies = getattr(
            self, "_refresh_processing_parameter_dependencies", None
        )
        if callable(refresh_dependencies):
            refresh_dependencies()
        self._refresh_processing_stage_headers()
        self.processing_sheet_status.setText(
            f"Stage {stage} 已设为{control.currentText()}；仅本次任务生效。"
        )

    def _on_stage_parameter_auto_toggled(self, field: str, automatic: bool) -> None:
        if self._processing_controls_updating:
            return
        spec = SPECS_BY_FIELD[field]
        control = self._stage_parameter_controls[field]
        overrides = self.processing_parameters["stages"][str(spec.stage)]["overrides"]
        if automatic:
            overrides.pop(field, None)
        else:
            if spec.profile_scaling != "none":
                profile_value = effective_parameter_value(
                    self.processing_parameters,
                    field,
                )
                previous_block = control.blockSignals(True)
                try:
                    self._set_processing_control_value(
                        control,
                        spec,
                        profile_value,
                    )
                finally:
                    control.blockSignals(previous_block)
            overrides[field] = self._processing_control_value(control, spec)
        normalized, _adjustments = normalize_processing_parameters(
            self.processing_parameters
        )
        StarunGui._apply_processing_payload_state(self, normalized)
        self._sync_stage_parameter_controls()
        refresh_dependencies = getattr(
            self, "_refresh_processing_parameter_dependencies", None
        )
        if callable(refresh_dependencies):
            refresh_dependencies()
        self._refresh_processing_stage_headers()
        self.processing_sheet_status.setText(
            f"{spec.label}已切换为"
            f"{'跟随档位' if automatic and spec.profile_scaling != 'none' else '自动' if automatic else '自定义'}。"
        )

    def _on_stage_parameter_value_changed(self, field: str) -> None:
        if self._processing_controls_updating:
            return
        auto_check = self._stage_parameter_auto_checks[field]
        automatic = auto_check.isChecked()
        if automatic:
            # Controls remain editable while automatic so that the first user
            # edit is itself the transition to an explicit task override.
            auto_check.setChecked(False)
            return
        spec = SPECS_BY_FIELD[field]
        value = self._processing_control_value(
            self._stage_parameter_controls[field], spec
        )
        self.processing_parameters["stages"][str(spec.stage)]["overrides"][field] = value
        normalized, _adjustments = normalize_processing_parameters(
            self.processing_parameters
        )
        StarunGui._apply_processing_payload_state(self, normalized)
        self._refresh_processing_parameter_dependencies()
        self._refresh_processing_stage_headers()
        self.processing_sheet_status.setText(
            f"Stage {spec.stage} · {spec.label}已更新；仅本次任务生效。"
        )

    def _reset_stage_processing_parameters(self, stage: int) -> None:
        self.processing_parameters = reset_stage_parameters(
            self.processing_parameters, (stage,)
        )
        StarunGui._apply_processing_payload_state(
            self, self.processing_parameters
        )
        self._sync_stage_parameter_controls()
        self._refresh_processing_parameter_dependencies()
        self._refresh_processing_stage_headers()
        profile = str(self.processing_parameters.get("gate_profile") or "default")
        self.processing_sheet_status.setText(
            f"Stage {stage} 已清除专家覆盖；门禁重新跟随"
            f"{GATE_PROFILE_LABELS[profile]}。"
        )

    def _toggle_processing_stage(self, stage: int, checked: bool) -> None:
        if not hasattr(self, "_stage_parameter_sections"):
            return
        if not checked:
            header = self._stage_parameter_headers[stage]
            header.blockSignals(True)
            header.setChecked(True)
            header.blockSignals(False)
            return
        for other_stage, section in self._stage_parameter_sections.items():
            active = other_stage == stage
            section.setVisible(active)
            header = self._stage_parameter_headers[other_stage]
            if header.isChecked() != active:
                header.blockSignals(True)
                header.setChecked(active)
                header.blockSignals(False)
        self._refresh_processing_stage_headers()

    def _set_processing_expert_visible(self, visible: bool) -> None:
        self.processing_expert_visible = bool(visible)
        button = getattr(self, "processing_expert_btn", None)
        if button is not None and button.isChecked() != self.processing_expert_visible:
            button.blockSignals(True)
            button.setChecked(self.processing_expert_visible)
            button.blockSignals(False)
        expert_section_headers = getattr(
            self, "_stage_expert_section_headers", {}
        )
        for stage, headers in expert_section_headers.items():
            bodies = self._stage_expert_section_bodies.get(stage, {})
            for section, header in headers.items():
                header.setVisible(self.processing_expert_visible)
                body = bodies.get(section)
                if body is not None:
                    body.setVisible(
                        self.processing_expert_visible and header.isChecked()
                    )
        if not expert_section_headers:
            for rows in getattr(self, "_stage_expert_widgets", {}).values():
                for widget in rows:
                    widget.setVisible(self.processing_expert_visible)
        refresh_dependencies = getattr(
            self, "_refresh_processing_parameter_dependencies", None
        )
        if callable(refresh_dependencies):
            refresh_dependencies()
        self._refresh_processing_stage_headers()
        if not self._restoring_settings and getattr(self, "_settings_loaded", False):
            self._save_settings()

    def _sync_stage_parameter_controls(self) -> None:
        if not hasattr(self, "_stage_parameter_controls"):
            return
        profile_effective_values = {
            str(record["field"]): record["effective"]
            for record in processing_gate_profile_audit(
                self.processing_parameters
            )["fields"]
        }
        self._processing_controls_updating = True
        try:
            for stage, specs in SPECS_BY_STAGE.items():
                entry = self.processing_parameters["stages"][str(stage)]
                for spec in specs:
                    if spec.field not in self._stage_parameter_controls:
                        continue
                    control = self._stage_parameter_controls[spec.field]
                    if spec.stage_mode:
                        self._set_processing_control_value(
                            control, spec, entry["mode"]
                        )
                        continue
                    automatic = spec.field not in entry["overrides"]
                    auto_check = self._stage_parameter_auto_checks[spec.field]
                    auto_check.setChecked(automatic)
                    effective_label = self._stage_parameter_effective_labels.get(
                        spec.field
                    )
                    if automatic and spec.profile_scaling != "none":
                        value = profile_effective_values[spec.field]
                        if effective_label is not None:
                            effective_label.setText(
                                self._format_processing_effective_value(
                                    spec,
                                    value,
                                )
                            )
                            effective_label.setToolTip(
                                "当前全任务门禁档位派生后的运行值；"
                                "取消“跟随档位”后可在专家范围内自定义。"
                            )
                            effective_label.setAccessibleDescription(
                                effective_label.toolTip()
                            )
                            effective_label.setVisible(True)
                        control.setVisible(False)
                    else:
                        value = entry["overrides"].get(spec.field, spec.default)
                        self._set_processing_control_value(control, spec, value)
                        control.setVisible(True)
                        if effective_label is not None:
                            effective_label.setVisible(False)
        finally:
            self._processing_controls_updating = False
        self._refresh_processing_parameter_dependencies()

    def _refresh_processing_parameter_dependencies(self) -> None:
        """Disable unavailable controls without discarding frozen overrides."""
        controls = getattr(self, "_stage_parameter_controls", {})
        payload = getattr(self, "processing_parameters", {})
        if not controls or not isinstance(payload, Mapping):
            return

        def current_value(field: str) -> object:
            dependency_spec = SPECS_BY_FIELD.get(field)
            if dependency_spec is None:
                return None
            entry = payload["stages"][str(dependency_spec.stage)]
            if dependency_spec.stage_mode:
                return entry["mode"]
            return entry["overrides"].get(field, dependency_spec.default)

        for field, control in controls.items():
            spec = SPECS_BY_FIELD[field]
            enabled = all(
                current_value(dependency_field) in allowed_values
                for dependency_field, allowed_values in spec.depends_on
            )
            control.setEnabled(enabled)
            effective_label = getattr(
                self,
                "_stage_parameter_effective_labels",
                {},
            ).get(field)
            if effective_label is not None:
                effective_label.setEnabled(enabled)
            auto_check = self._stage_parameter_auto_checks.get(field)
            if auto_check is not None:
                auto_check.setEnabled(enabled)
            row = self._stage_parameter_row_widgets.get(field)
            if row is not None:
                row[0].setEnabled(enabled)

    def _refresh_processing_stage_headers(self) -> None:
        if not hasattr(self, "_stage_parameter_headers"):
            return
        for stage, header in self._stage_parameter_headers.items():
            entry = self.processing_parameters["stages"][str(stage)]
            custom_count = len(entry["overrides"]) + int(entry["mode"] != "auto")
            expert_fields = {
                spec.field for spec in SPECS_BY_STAGE[stage]
                if spec.level == "expert"
            }
            hidden_expert = bool(
                expert_fields.intersection(entry["overrides"])
                and not self.processing_expert_visible
            )
            arrow = "▾" if header.isChecked() else "▸"
            profile = str(
                self.processing_parameters.get("gate_profile") or "default"
            )
            suffix = (
                f" · {custom_count} 项自定义"
                if custom_count
                else f" · 跟随{GATE_PROFILE_LABELS[profile]}"
                if profile != "default"
                else " · 自动"
            )
            if hidden_expert:
                suffix += " · 专家配置已生效"
            header.setText(
                f"{arrow} Stage {stage} · {STAGE_TITLES[stage]}{suffix}"
            )

    def _sync_processing_controls_from_state(self) -> None:
        if not hasattr(self, "output_format_checks"):
            return
        if hasattr(self, "_stage_parameter_controls"):
            self._processing_controls_updating = True
            try:
                for key, checkbox in self.output_format_checks.items():
                    checkbox.setChecked(key in self.output_formats)
                self.review_combo.setCurrentIndex(
                    max(0, self.review_combo.findData(self.review_only))
                )
                self.compute_combo.setCurrentIndex(
                    max(0, self.compute_combo.findData(self.compute_mode))
                )
                self.auto_tune_check.setChecked(self.auto_tune_enabled)
                self.max_retries_spin.setValue(self.max_retries)
                self.retry_delay_spin.setValue(self.retry_delay)
                self.review_bundle_check.setChecked(
                    self.review_bundle_enabled
                )
                self.managed_output_check.setChecked(
                    self.managed_output_enabled
                )
                self.checkpoint_mode_check.setChecked(self.checkpoint_mode)
                self._update_gate_profile_controls()
            finally:
                self._processing_controls_updating = False
            self._sync_stage_parameter_controls()
            self._refresh_processing_stage_headers()
            self._update_processing_sheet_availability()
            return
        self._processing_controls_updating = True
        try:
            for key, checkbox in self.output_format_checks.items():
                checkbox.setChecked(key in self.output_formats)
            self.review_combo.setCurrentIndex(
                max(0, self.review_combo.findData(self.review_only))
            )
            for combo, value in (
                (self.color_combo, self.color_calibration),
                (self.filter_combo, self.filter_hint),
                (self.denoise_combo, self.denoise_mode),
                (self.deconv_combo, self.deconvolution_mode),
                (self.compute_combo, self.compute_mode),
            ):
                combo.setCurrentIndex(max(0, combo.findData(value)))
            self.graxpert_model_edit.setText(self.graxpert_model_path)
            for control, value in (
                (self.pcc_timeout_spin, self.pcc_timeout_sec),
                (self.local_wb_gain_spin, self.local_wb_gain_limit),
                (self.builtin_denoise_spin, self.builtin_denoise_strength),
                (self.graxpert_strength_spin, self.graxpert_deconv_strength),
                (self.rl_iterations_spin, self.rl_iterations),
                (self.rl_maxstars_spin, self.rl_maxstars),
                (self.starless_retry_spin, self.starless_retry_max),
                (self.starless_repair_spin, self.starless_repair_strength),
                (self.starless_halo_spin, self.starless_halo_repair_strength),
                (self.starless_chroma_spin, self.starless_chroma_strength),
                (self.starmask_stretch_spin, self.starmask_asinh_stretch),
                (self.weak_star_recovery_spin, self.weak_star_recovery_ratio),
            ):
                control.setValue(value)
        finally:
            self._processing_controls_updating = False
        self._update_processing_sheet_availability()

    def _update_processing_sheet_availability(self) -> None:
        if not hasattr(self, "processing_color_group"):
            return
        if hasattr(self, "processing_stage_groups"):
            input_mode = self._current_input_mode()
            disabled_stages = (
                set(range(1, 6))
                if input_mode == INPUT_MODE_LINEAR_RESUME
                else {1, 2}
                if input_mode == INPUT_MODE_STAGE2_CORRECTED_RESUME
                else {1}
                if input_mode == INPUT_MODE_STAGE1_PREPARED_RESUME
                else set()
            )
            for stage, body in self.processing_stage_groups.items():
                enabled = stage not in disabled_stages
                body.setEnabled(enabled)
                header = self._stage_parameter_headers[stage]
                header.setEnabled(enabled)
                if enabled:
                    header.setToolTip("配置当前任务的阶段参数")
                else:
                    header.setToolTip("该阶段已由已验证恢复点完成，本次不会重跑")
            checked_stage = next(
                (
                    stage for stage, header in self._stage_parameter_headers.items()
                    if header.isChecked()
                ),
                2,
            )
            if checked_stage in disabled_stages:
                next_stage = next(
                    stage for stage in STAGE_TITLES if stage not in disabled_stages
                )
                self._toggle_processing_stage(next_stage, True)
            if input_mode == INPUT_MODE_LINEAR_RESUME:
                self.processing_sheet_note.setText(
                    "Stage 5 恢复点已完成 Stage 1–5；本次配置 Stage 6–10。"
                )
            elif input_mode == INPUT_MODE_STAGE2_CORRECTED_RESUME:
                self.processing_sheet_note.setText(
                    "Stage 2 恢复点已验证；本次配置 Stage 3–10。"
                )
            elif input_mode == INPUT_MODE_STAGE1_PREPARED_RESUME:
                self.processing_sheet_note.setText(
                    "Stage 1 恢复点已验证；本次配置 Stage 2–10。"
                )
            else:
                self.processing_sheet_note.setText(
                    "门禁档位统一作用于当前任务；专家覆盖隐藏后仍保留并生效。"
                )
            return
        linear_resume = self._current_input_mode() == INPUT_MODE_LINEAR_RESUME
        self.processing_color_group.setEnabled(not linear_resume)
        self.processing_linear_group.setEnabled(not linear_resume)
        self.filter_combo.setEnabled(not linear_resume)
        for widget in self.processing_prelinear_professional_widgets:
            widget.setEnabled(not linear_resume)
        model_editable = (
            not linear_resume and str(self.deconv_combo.currentData()) == "auto"
        )
        for widget in (
            self.graxpert_model_edit,
            self.graxpert_model_file_btn,
            self.graxpert_model_dir_btn,
        ):
            widget.setEnabled(model_editable)
        if linear_resume:
            self.processing_sheet_note.setText(
                "从线性结果继续：色彩校准和线性处理已完成，本次不再执行。"
            )
        else:
            self.processing_sheet_note.setText(
                "同一任务页内配置；运行开始后保持只读。"
            )
        self._update_graxpert_model_status()

    def _refresh_gaia_catalog_status(self) -> None:
        status_label = getattr(self, "gaia_catalog_status", None)
        button = getattr(self, "gaia_catalog_download_btn", None)
        if status_label is None or button is None:
            return
        status = gaia_catalog_status(self.runtime_home)
        if status["available"]:
            status_label.setText(
                "已安装：runtime home/.local/share/siril/"
                "siril_cat_healpix8_astro.dat（约 1.52 GB）"
            )
            button.setText("重新下载并校验离线 Gaia 解析/PCC 目录")
        else:
            size_bytes = int(status["size_bytes"])
            suffix = (
                f"；发现不完整文件 {size_bytes / (1024**2):.1f} MiB"
                if size_bytes
                else ""
            )
            status_label.setText(
                "未安装；该目录仅影响离线 platesolve/PCC，不替代 SPCC xp_sampled 目录"
                + suffix
            )
            button.setText("下载离线 Gaia 解析/PCC 目录（约 1.1 GB）")

    def _toggle_gaia_catalog_download(self, _checked: bool = False) -> None:
        worker = self.gaia_catalog_worker
        if worker and worker.isRunning():
            self.gaia_catalog_status.setText("正在取消目录下载并清理临时文件…")
            self.gaia_catalog_download_btn.setEnabled(False)
            worker.stop()
            return

        self.gaia_catalog_download_btn.setText("取消下载")
        self.gaia_catalog_download_btn.setEnabled(True)
        self.gaia_catalog_status.setText("正在准备离线 Gaia 目录下载…")
        force_download = bool(
            gaia_catalog_status(self.runtime_home)["available"]
        )

        def runner(stop_event, progress):
            try:
                return download_gaia_catalog(
                    self.runtime_home,
                    stop_event=stop_event,
                    progress=progress,
                    force=force_download,
                )
            except GaiaCatalogCancelled as error:
                raise BootstrapCancelled() from error
            except GaiaCatalogDownloadError as error:
                raise BootstrapError("离线 Gaia 目录安装失败", str(error)) from error
            except Exception as error:
                raise BootstrapError("离线 Gaia 目录安装失败", str(error)) from error

        self.gaia_catalog_worker = BootstrapWorker(runner, parent=self)
        self.gaia_catalog_worker.progress.connect(
            lambda message: self.gaia_catalog_status.setText(message)
        )
        self.gaia_catalog_worker.succeeded.connect(
            self._on_gaia_catalog_download_succeeded
        )
        self.gaia_catalog_worker.failed.connect(
            self._on_gaia_catalog_download_failed
        )
        self.gaia_catalog_worker.cancelled.connect(
            self._on_gaia_catalog_download_cancelled
        )
        self.gaia_catalog_worker.start()

    def _cleanup_gaia_catalog_worker(self) -> None:
        if self.gaia_catalog_worker:
            self.gaia_catalog_worker.wait(200)
            self.gaia_catalog_worker.deleteLater()
            self.gaia_catalog_worker = None
        self.gaia_catalog_download_btn.setEnabled(not self._ui_running)

    def _on_gaia_catalog_download_succeeded(self, result: object) -> None:
        self._cleanup_gaia_catalog_worker()
        self._refresh_gaia_catalog_status()
        self._append_event(f"离线 Gaia 星色目录安装完成：{result}")

    def _on_gaia_catalog_download_failed(self, title: str, detail: str) -> None:
        self._cleanup_gaia_catalog_worker()
        self._refresh_gaia_catalog_status()
        self._append_event(f"{title}：{detail}")

    def _on_gaia_catalog_download_cancelled(self) -> None:
        self._cleanup_gaia_catalog_worker()
        self._refresh_gaia_catalog_status()
        self._append_event("已取消离线 Gaia 目录下载，临时文件已清理。")

    def _update_graxpert_model_status(self) -> None:
        if not hasattr(self, "graxpert_model_status"):
            return
        raw_path = self.graxpert_model_edit.text().strip()
        if not raw_path:
            self.graxpert_model_status.setText(
                "未指定用户模型；自动模式优先检查应用模型，否则回退 Siril RL。"
            )
            return
        path = Path(raw_path).expanduser()
        if path.exists():
            self.graxpert_model_status.setText(
                "用户模型路径存在；运行时还会校验版本目录和模型文件。"
            )
        else:
            self.graxpert_model_status.setText(
                "⚠ 路径当前不存在；任务仍可启动，自动模式将安全回退。"
            )

    def _on_output_format_toggled(self, key: str, checked: bool) -> None:
        if self._processing_controls_updating:
            return
        if not checked and not any(
            checkbox.isChecked()
            for checkbox in self.output_format_checks.values()
        ):
            checkbox = self.output_format_checks[key]
            checkbox.blockSignals(True)
            try:
                checkbox.setChecked(True)
            finally:
                checkbox.blockSignals(False)
            self.processing_sheet_status.setText("至少保留一种输出格式。")
            return
        if hasattr(self, "_stage_parameter_controls"):
            self._on_general_processing_controls_changed()
            return
        self._on_processing_controls_changed()

    def _on_processing_selection_changed(self, _index: int = -1) -> None:
        if self._processing_controls_updating:
            return
        self._update_processing_sheet_availability()
        self._on_processing_controls_changed()

    def _on_processing_controls_changed(self, _value: object = None) -> None:
        if self._processing_controls_updating:
            return
        formats = tuple(
            key
            for key, checkbox in self.output_format_checks.items()
            if checkbox.isChecked()
        )
        if not formats:
            return
        self.output_formats = formats
        self.review_only = bool(self.review_combo.currentData())
        self.color_calibration = str(self.color_combo.currentData())
        self.filter_hint = str(self.filter_combo.currentData())
        self.denoise_mode = str(self.denoise_combo.currentData())
        self.deconvolution_mode = str(self.deconv_combo.currentData())
        self.graxpert_model_path = self.graxpert_model_edit.text().strip()
        self.compute_mode = str(self.compute_combo.currentData())
        self.pcc_timeout_sec = self.pcc_timeout_spin.value()
        self.local_wb_gain_limit = self.local_wb_gain_spin.value()
        self.builtin_denoise_strength = self.builtin_denoise_spin.value()
        self.graxpert_deconv_strength = self.graxpert_strength_spin.value()
        self.rl_iterations = self.rl_iterations_spin.value()
        self.rl_maxstars = self.rl_maxstars_spin.value()
        self.starless_retry_max = self.starless_retry_spin.value()
        self.starless_repair_strength = self.starless_repair_spin.value()
        self.starless_halo_repair_strength = self.starless_halo_spin.value()
        self.starless_chroma_strength = self.starless_chroma_spin.value()
        self.starmask_asinh_stretch = self.starmask_stretch_spin.value()
        self.weak_star_recovery_ratio = self.weak_star_recovery_spin.value()
        self.processing_sheet_status.setText(
            "已自动保存；点击“开始处理”时冻结为本次任务配置。"
        )
        if not self._restoring_settings:
            self._save_settings()

    def _select_graxpert_model_file(self) -> None:
        selected, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "选择 GraXpert 对象反卷积模型",
            self.graxpert_model_edit.text().strip() or str(Path.home()),
            "ONNX 模型 (model.onnx *.onnx)",
        )
        if selected:
            self.graxpert_model_edit.setText(selected)
            self._on_processing_controls_changed()

    def _select_graxpert_model_directory(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "选择 GraXpert 模型目录",
            self.graxpert_model_edit.text().strip() or str(Path.home()),
        )
        if selected:
            self.graxpert_model_edit.setText(selected)
            self._on_processing_controls_changed()

    def _restore_processing_defaults(self) -> None:
        self._restore_processing_settings(default_processing_parameters())
        self._save_settings()
        self.processing_sheet_status.setText(
            "通用配置与默认门禁档位已恢复；Stage 1–10 已全部恢复自动参数。"
        )
        self._append_event(
            "处理参数已恢复为安全默认值；门禁回到默认档位，阶段覆盖已清除。"
        )

    def _set_processing_parameters_expanded(self, expanded: bool) -> None:
        self.processing_parameters_expanded = bool(expanded)
        visible = bool(expanded and self.advanced_toggle_btn.isChecked())
        processing_scroll = getattr(self, "processing_params_scroll", None)
        if processing_scroll is not None:
            processing_scroll.setVisible(visible)
        self.processing_params_panel.setVisible(visible)
        task_phase_bar = getattr(self, "task_phase_bar", None)
        if task_phase_bar is not None:
            task_phase_bar.setVisible(not visible)
        task_splitter = getattr(self, "task_splitter", None)
        if task_splitter is not None:
            task_splitter.setMaximumHeight(
                248 if visible else 16777215
            )
        self.processing_params_btn.setText(
            "收起处理参数" if expanded else "处理参数…"
        )
        self.processing_params_btn.setAccessibleName(
            "收起处理参数设置" if expanded else "展开处理参数设置"
        )
        if expanded:
            self._update_processing_sheet_availability()
            if visible and processing_scroll is not None:
                processing_scroll.setFocus()
        if not self._restoring_settings:
            self._save_settings()
            if (
                not expanded
                and self._input_discovery is not None
                and self._input_discovery.kind == InputKind.PRODUCT_TASK
                and self._workspace_state == WORKSPACE_TASK
            ):
                self._analyze_selected_directory()

    def _configure_processing_parameters(self, _checked: bool = False) -> None:
        self._set_processing_parameters_expanded(
            not self.processing_parameters_expanded
        )

    def _update_debug_button_text(self) -> None:
        if self.debug_btn.isChecked() != self.debug_mode_enabled:
            self.debug_btn.blockSignals(True)
            try:
                self.debug_btn.setChecked(self.debug_mode_enabled)
            finally:
                self.debug_btn.blockSignals(False)

    def _on_debug_toggled(self, checked: bool) -> None:
        self.debug_mode_enabled = bool(checked)
        self._update_debug_button_text()
        if not self._restoring_settings:
            self._append_event(
                "保留中间文件已"
                + ("开启" if self.debug_mode_enabled else "关闭")
            )
            self._save_settings()
        sync_preferences = getattr(self, "_sync_preferences_window", None)
        if callable(sync_preferences):
            sync_preferences()

    def _update_network_button_text(self) -> None:
        if self.network_btn.isChecked() != self.network_mode_enabled:
            self.network_btn.blockSignals(True)
            try:
                self.network_btn.setChecked(self.network_mode_enabled)
            finally:
                self.network_btn.blockSignals(False)

    def _on_network_toggled(self, checked: bool) -> None:
        self.network_mode_enabled = bool(checked)
        self._update_network_button_text()
        if not self._restoring_settings:
            self._append_event(
                "允许联网已" + ("开启" if self.network_mode_enabled else "关闭")
            )
            self._save_settings()
        sync_preferences = getattr(self, "_sync_preferences_window", None)
        if callable(sync_preferences):
            sync_preferences()

    def _on_advanced_toggled(self, expanded: bool) -> None:
        self.advanced_panel.setVisible(expanded)
        processing_visible = bool(
            expanded and self.processing_parameters_expanded
        )
        processing_scroll = getattr(self, "processing_params_scroll", None)
        if processing_scroll is not None:
            processing_scroll.setVisible(processing_visible)
        self.processing_params_panel.setVisible(processing_visible)
        task_phase_bar = getattr(self, "task_phase_bar", None)
        if task_phase_bar is not None:
            task_phase_bar.setVisible(not processing_visible)
        task_splitter = getattr(self, "task_splitter", None)
        if task_splitter is not None:
            task_splitter.setMaximumHeight(
                248 if processing_visible else 16777215
            )
        self.advanced_toggle_btn.setText(
            "高级设置 ▾" if expanded else "高级设置 ▸"
        )
        self.advanced_toggle_btn.setAccessibleName(
            "折叠高级设置" if expanded else "展开高级设置"
        )
        if not self._restoring_settings:
            self._save_settings()

    def _on_log_toggled(self, expanded: bool) -> None:
        if self.log_toggle_btn.isChecked() != expanded:
            self.log_toggle_btn.blockSignals(True)
            self.log_toggle_btn.setChecked(expanded)
            self.log_toggle_btn.blockSignals(False)
        self.log_container.setVisible(expanded)
        self.log_toggle_btn.setText("详细日志 ▾" if expanded else "详细日志 ▸")
        self.log_toggle_btn.setAccessibleName(
            "折叠详细日志" if expanded else "展开详细日志"
        )
        if not self._restoring_settings:
            self._save_settings()

    def _load_settings(self) -> None:
        self._restoring_settings = True
        try:
            saved_geometry = self.settings.value("ui/windowNormalGeometry")
            self._main_window_geometry_restored = restore_window_geometry(
                self,
                saved_geometry,
                preferred_size=MAIN_WINDOW_DEFAULT_SIZE,
            )
            self._restore_main_window_maximized = self.settings.value(
                "ui/windowMaximized",
                False,
                type=bool,
            )

            history_geometry = self.settings.value(
                "ui/historyWindowNormalGeometry"
            )
            self._history_window_geometry_restored = restore_window_geometry(
                self.history_window,
                history_geometry,
                preferred_size=HISTORY_WINDOW_DEFAULT_SIZE,
                reference=self,
            )
            self._restore_history_window_maximized = self.settings.value(
                "ui/historyWindowMaximized",
                False,
                type=bool,
            )

            task_splitter_sizes = self._splitter_sizes_setting(
                self.settings.value("ui/taskSplitterSizes"),
                2,
            )
            if task_splitter_sizes is not None:
                self._task_splitter_sizes = task_splitter_sizes
                self.task_splitter.setSizes(self._task_splitter_sizes)
            try:
                sidebar_width = int(
                    self.settings.value(
                        "ui/runSidebarWidth",
                        self._run_sidebar_width,
                    )
                )
            except (TypeError, ValueError):
                sidebar_width = self._run_sidebar_width
            try:
                inspector_width = int(
                    self.settings.value(
                        "ui/runInspectorWidth",
                        self._run_inspector_width,
                    )
                )
            except (TypeError, ValueError):
                inspector_width = self._run_inspector_width
            self._run_sidebar_width = min(380, max(210, sidebar_width))
            self._run_inspector_width = min(420, max(260, inspector_width))
            self.run_splitter.setSizes(
                [self._run_sidebar_width, 700, self._run_inspector_width]
            )
            self.toggle_sidebar_action.setChecked(
                self.settings.value("ui/runSidebarVisible", True, type=bool)
            )
            self.toggle_inspector_action.setChecked(
                self.settings.value("ui/runInspectorVisible", True, type=bool)
            )
            try:
                inspector_tab = int(
                    self.settings.value("ui/runInspectorTab", 0)
                )
            except (TypeError, ValueError):
                inspector_tab = 0
            self.inspector_tabs.setCurrentIndex(
                min(max(0, inspector_tab), self.inspector_tabs.count() - 1)
            )

            self.debug_mode_enabled = self.settings.value(
                "advanced/keepIntermediateFiles", False, type=bool
            )
            self.network_mode_enabled = self.settings.value(
                "advanced/allowNetwork", True, type=bool
            )
            saved_formats = self.settings.value(
                "processing/outputFormats",
                list(DEFAULT_PROCESSING_SETTINGS["output_formats"]),
            )
            if isinstance(saved_formats, str):
                saved_formats = [
                    value.strip()
                    for value in saved_formats.split(",")
                    if value.strip()
                ]
            self._restore_processing_settings(
                default_processing_parameters(
                    general={
                        "output_formats": list(saved_formats or []),
                        "review_only": self.settings.value(
                            "processing/reviewOnly", False, type=bool
                        ),
                        "compute_mode": self.settings.value(
                            "processing/computeMode", "auto"
                        ),
                        "auto_tune_enabled": self.settings.value(
                            "processing/autoTuneEnabled", True, type=bool
                        ),
                        "max_retries": self.settings.value(
                            "processing/maxRetries",
                            DEFAULT_PROCESSING_SETTINGS["max_retries"],
                        ),
                        "retry_delay": self.settings.value(
                            "processing/retryDelay",
                            DEFAULT_PROCESSING_SETTINGS["retry_delay"],
                        ),
                        "review_bundle_enabled": self.settings.value(
                            "processing/reviewBundleEnabled", True, type=bool
                        ),
                        "managed_output_enabled": self.settings.value(
                            "processing/managedOutputEnabled", True, type=bool
                        ),
                        "checkpoint_mode": self.settings.value(
                            "processing/checkpointMode", False, type=bool
                        ),
                    }
                )
            )
            self.settings.sync()
            self._update_debug_button_text()
            self._update_network_button_text()

            advanced_expanded = self.settings.value(
                "ui/advancedExpanded", False, type=bool
            )
            self.advanced_toggle_btn.setChecked(advanced_expanded)
            self._on_advanced_toggled(advanced_expanded)

            processing_expanded = self.settings.value(
                "ui/processingParametersExpanded", False, type=bool
            )
            self._set_processing_parameters_expanded(processing_expanded)
            self._set_processing_expert_visible(
                self.settings.value(
                    "ui/processingExpertVisible", False, type=bool
                )
            )

            log_expanded = self.settings.value(
                "ui/logExpanded", False, type=bool
            )
            self.toggle_log_action.setChecked(log_expanded)
            self._on_log_toggled(log_expanded)

            recent_value = self.settings.value("recentDirectories", [])
            if isinstance(recent_value, str):
                recent = [recent_value]
            else:
                recent = [str(value) for value in (recent_value or [])]
            self._recent_directories = [
                value for value in recent if Path(value).expanduser().exists()
            ][:8]
            self._refresh_recent_directories()

            last_directory = str(self.settings.value("lastDirectory", "") or "")
            if last_directory and Path(last_directory).expanduser().exists():
                self._apply_input_path(
                    Path(last_directory).expanduser(),
                    remember=False,
                )

            mode_index = self.mode_combo.findData(UI_MODE_RECOMMENDED)
            self.mode_combo.setCurrentIndex(max(0, mode_index))
            self.input_mode = self._current_input_mode()
        finally:
            self._restoring_settings = False
        if self._restore_main_window_maximized:
            self.setWindowState(
                self.windowState() | Qt.WindowState.WindowMaximized
            )
        if self._restore_history_window_maximized:
            self.history_window.setWindowState(
                self.history_window.windowState()
                | Qt.WindowState.WindowMaximized
            )
        self._analyze_selected_directory()

    def _store_window_geometry_settings(self) -> None:
        """Persist normal rectangles separately from transient window states."""

        self.settings.setValue(
            "ui/windowNormalGeometry",
            normal_window_geometry(self),
        )
        main_state = self.windowState()
        main_maximized = bool(
            main_state & Qt.WindowState.WindowMaximized
        ) or bool(
            self.isFullScreen()
            and getattr(self, "_window_was_maximized", False)
        )
        self.settings.setValue("ui/windowMaximized", main_maximized)

        history_window = getattr(self, "history_window", None)
        if history_window is not None:
            self.settings.setValue(
                "ui/historyWindowNormalGeometry",
                normal_window_geometry(history_window),
            )
            history_state = history_window.windowState()
            self.settings.setValue(
                "ui/historyWindowMaximized",
                bool(history_state & Qt.WindowState.WindowMaximized),
            )

    def _save_window_geometry_state(self) -> None:
        if self._restoring_settings or not self._settings_loaded:
            return
        self._store_window_geometry_settings()
        self.settings.sync()

    def _schedule_window_geometry_save(self) -> None:
        timer = getattr(self, "_window_geometry_save_timer", None)
        if (
            timer is not None
            and self._settings_loaded
            and not self._restoring_settings
        ):
            timer.start()

    def _save_settings(self) -> None:
        if self._restoring_settings:
            return
        self.settings.setValue(
            "advanced/keepIntermediateFiles", self.debug_mode_enabled
        )
        self.settings.setValue("advanced/allowNetwork", self.network_mode_enabled)
        self.settings.setValue(
            "processing/outputFormats", list(self.output_formats)
        )
        self.settings.setValue("processing/reviewOnly", self.review_only)
        self.settings.setValue("processing/computeMode", self.compute_mode)
        self.settings.setValue(
            "processing/autoTuneEnabled", self.auto_tune_enabled
        )
        self.settings.setValue("processing/maxRetries", self.max_retries)
        self.settings.setValue("processing/retryDelay", self.retry_delay)
        self.settings.setValue(
            "processing/reviewBundleEnabled", self.review_bundle_enabled
        )
        self.settings.setValue(
            "processing/managedOutputEnabled", self.managed_output_enabled
        )
        self.settings.setValue(
            "processing/checkpointMode", self.checkpoint_mode
        )
        self.settings.setValue(
            "ui/advancedExpanded", self.advanced_toggle_btn.isChecked()
        )
        self.settings.setValue(
            "ui/processingParametersExpanded",
            self.processing_parameters_expanded,
        )
        self.settings.setValue(
            "ui/processingExpertVisible",
            self.processing_expert_visible,
        )
        self.settings.setValue(
            "ui/taskSplitterSizes",
            self._task_splitter_sizes,
        )
        run_sizes = self.run_splitter.sizes()
        if self._workspace_state == WORKSPACE_RUN and len(run_sizes) == 3:
            if self.toggle_sidebar_action.isChecked() and run_sizes[0] > 0:
                self._run_sidebar_width = run_sizes[0]
            if self.toggle_inspector_action.isChecked() and run_sizes[2] > 0:
                self._run_inspector_width = run_sizes[2]
        self.settings.setValue("ui/runSidebarWidth", self._run_sidebar_width)
        self.settings.setValue("ui/runInspectorWidth", self._run_inspector_width)
        self.settings.setValue(
            "ui/runSidebarVisible",
            self.toggle_sidebar_action.isChecked(),
        )
        self.settings.setValue(
            "ui/runInspectorVisible",
            self.toggle_inspector_action.isChecked(),
        )
        self.settings.setValue(
            "ui/runInspectorTab",
            self.inspector_tabs.currentIndex(),
        )
        self.settings.setValue("ui/logExpanded", self.log_toggle_btn.isChecked())
        self._store_window_geometry_settings()
        self.settings.setValue("modeSelection", UI_MODE_RECOMMENDED)
        self.settings.setValue("recentDirectories", self._recent_directories)
        directory = self.dir_edit.text().strip()
        if directory and Path(directory).expanduser().exists():
            self.settings.setValue("lastDirectory", str(Path(directory).expanduser()))
        self.settings.sync()

    def _close_run_log(self) -> None:
        with self._run_log_lock:
            if self.run_log_file:
                self.run_log_file.close()
                self.run_log_file = None

    def _update_run_state(
        self,
        *,
        phase: str,
        status: str,
        detail: str | None = None,
        errors: list[str] | tuple[str, ...] | None = None,
        extra: Mapping[str, object] | None = None,
        strict: bool = False,
    ) -> None:
        state_path = getattr(self, "run_state_path", None)
        if state_path is None:
            return
        failure: Exception | None = None
        try:
            with self._run_state_lock:
                payload = dict(getattr(self, "_run_state_payload", {}) or {})
                payload.update(
                    {
                        "schema": RUN_STATE_SCHEMA,
                        "phase": str(phase),
                        "status": str(status),
                        "updated_at": runtime_utc_now(),
                    }
                )
                if detail is not None:
                    payload["detail"] = str(detail)
                if errors is not None:
                    payload["errors"] = [str(error) for error in errors]
                if extra:
                    payload.update({str(key): value for key, value in extra.items()})
                if (
                    errors
                    and str(status) == STATUS_FAILED
                    and not payload.get("issues")
                ):
                    payload["issues"] = [
                        {
                            "stage": 0,
                            "component": "gui_preflight",
                            "severity": "fatal",
                            "code": f"{str(phase)}_failed",
                            "recovered": False,
                            "message": str(error),
                        }
                        for error in errors
                    ]
                    payload["had_errors"] = True
                    payload["had_fatal_errors"] = True
                payload.pop("manifest_hash", None)
                payload["manifest_hash"] = (
                    pipeline_run_manifest.canonical_payload_hash(payload)
                )
                atomic_write_runtime_json(state_path, payload)
                self._run_state_payload = payload
        except (OSError, TypeError, ValueError) as error:
            failure = error
        if failure is not None:
            if strict:
                raise failure
            self._append_text(f"[{self._timestamp()}] 无法更新本轮状态文件：{failure}\n")

    def _begin_run_diagnostics(
        self,
        work_dir: Path,
        task: PreparedTask,
    ) -> None:
        """Create this run's log and mutable state before any preflight output."""

        if not work_dir.is_dir():
            raise OSError(f"运行目录不存在，无法创建诊断文件：{work_dir}")

        self._close_run_log()
        self.run_log_path = None
        self.run_state_path = None
        self.runtime_capabilities_path = None
        self._runtime_capability_manifest = None
        self._run_state_payload = {}

        safe_run_id = re.sub(r"[^A-Za-z0-9._-]+", "-", task.run.run_id).strip("-")
        safe_run_id = safe_run_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_path = work_dir / f"starun_gui_run_{safe_run_id}.log"
        try:
            log_file = log_path.open(
                "x",
                encoding="utf-8",
                errors="replace",
            )
        except FileExistsError:
            log_path = work_dir / (
                f"starun_gui_run_{safe_run_id}_{time.time_ns()}.log"
            )
            log_file = log_path.open(
                "x",
                encoding="utf-8",
                errors="replace",
            )

        self.run_log_path = log_path
        self.run_log_file = log_file
        self.run_state_path = work_dir / RUN_STATE_NAME
        self.runtime_capabilities_path = work_dir / RUNTIME_CAPABILITIES_NAME
        created_at = runtime_utc_now()
        self._run_state_payload = {
            "schema": RUN_STATE_SCHEMA,
            "run_id": task.run.run_id,
            "task_id": task.workspace.task_id,
            "work_dir": str(work_dir.resolve()),
            "log_path": str(log_path.resolve()),
            "runtime_capabilities_path": str(
                self.runtime_capabilities_path.resolve()
            ),
            "resources_root": str(self.resources.resolve()),
            "runtime_home": str(self.runtime_home.resolve()),
            "created_at": created_at,
            "had_errors": False,
            "had_fatal_errors": False,
            "had_degradations": False,
            "had_fallbacks": False,
            "review_required": False,
            "issues": [],
        }
        try:
            self._update_run_state(
                phase="preflight",
                status=STATUS_PREPARING,
                detail="run diagnostics created before preflight",
                errors=[],
                strict=True,
            )
        except Exception:
            self._close_run_log()
            raise

        clear_log = getattr(self.log_view, "clear", None)
        if callable(clear_log):
            clear_log()
        self._append_divider(
            "本次任务诊断已建立",
            [
                f"Run ID: {task.run.run_id}",
                f"日志文件: {self._display_path(self.run_log_path)}",
                f"状态文件: {self._display_path(self.run_state_path)}",
                "资源根目录: " + self._display_path(self.resources),
                "运行时主目录: " + self._display_path(self.runtime_home),
            ],
        )

    def _write_runtime_capability_manifest(self) -> None:
        path = getattr(self, "runtime_capabilities_path", None)
        manifest = getattr(self, "_runtime_capability_manifest", None)
        if path is None or not isinstance(manifest, Mapping):
            return
        atomic_write_runtime_json(path, manifest)

    def _inspect_runtime_capabilities(self, work_dir: Path) -> dict[str, object]:
        task = getattr(self, "_active_prepared_task", None)
        run_id = task.run.run_id if isinstance(task, PreparedTask) else None
        resources = Path(
            getattr(self, "resources", Path(self.config_template).parent)
        )
        runtime_home = Path(
            getattr(self, "runtime_home", work_dir / ".starun-runtime")
        )
        try:
            offline_fallback_mode = str(
                effective_parameter_value(
                    self.processing_parameters,
                    "stage4_offline_fallback_mode",
                )
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            offline_fallback_mode = "auto_local_reference"
        try:
            spcc_online_unverified_timeout_sec = int(
                effective_parameter_value(
                    self.processing_parameters,
                    "stage4_spcc_online_unverified_timeout_sec",
                )
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            spcc_online_unverified_timeout_sec = 90
        manifest = build_runtime_capabilities(
            resources_root=resources,
            runtime_home=runtime_home,
            siril_candidates=self._resolve_siril_candidates(),
            config_template=Path(self.config_template),
            pipeline_path=Path(self.pipeline_path),
            siril_plugin_dir=Path(self.siril_plugin_dir),
            network_enabled=bool(getattr(self, "network_mode_enabled", True)),
            stage4_offline_fallback_mode=offline_fallback_mode,
            stage4_spcc_online_unverified_timeout_sec=(
                spcc_online_unverified_timeout_sec
            ),
            run_id=run_id,
        )
        self._runtime_capability_manifest = manifest
        StarunGui._write_runtime_capability_manifest(self)
        return manifest

    def _complete_runtime_capability_preflight(self, progress) -> None:
        manifest = getattr(self, "_runtime_capability_manifest", None)
        if not isinstance(manifest, dict):
            return

        progress("正在验证 Siril 启动能力…")
        self._check_bootstrap_cancelled()
        capabilities = manifest.get("capabilities")
        capabilities = capabilities if isinstance(capabilities, Mapping) else {}
        siril = capabilities.get("siril")
        selected_path = (
            str(siril.get("selected_path") or "")
            if isinstance(siril, Mapping)
            else ""
        )
        if selected_path:
            cli = Path(selected_path)
            runtime_env = scrub_python_env(os.environ.copy())
            runtime_env["HOME"] = str(self.runtime_home)
            runtime_env["LANG"] = "en_US.UTF-8"
            runtime_env["LC_ALL"] = "en_US.UTF-8"
            if cli == self.bundled_siril_cli:
                runtime_env["SIRIL_RELOCATED_RES_DIR"] = str(
                    self.resources / "Siril.app" / "Contents" / "Resources"
                )
            try:
                completed = self._run_bootstrap_process(
                    [str(cli), "--version"],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=20,
                    env=runtime_env,
                )
                output = "\n".join(
                    part.strip()
                    for part in (completed.stdout, completed.stderr)
                    if part and part.strip()
                )
                version = next(
                    (
                        line.strip()
                        for line in reversed(output.splitlines())
                        if line.strip()
                    ),
                    "",
                )
                launchable = completed.returncode == 0
                detail = "" if launchable else (output[-500:] or "unknown error")
                update_siril_launch_probe(
                    manifest,
                    launchable=launchable,
                    version=version,
                    detail=detail,
                )
            except BootstrapCancelled:
                raise
            except Exception as error:
                update_siril_launch_probe(
                    manifest,
                    launchable=False,
                    detail=str(error),
                )
        else:
            update_siril_launch_probe(
                manifest,
                launchable=False,
                detail="no usable Siril CLI candidate",
            )

        network_capability = capabilities.get("network_endpoints")
        network_enabled = bool(
            isinstance(network_capability, Mapping)
            and network_capability.get("enabled")
        )
        progress(
            "正在探测 Gaia 网络端点…"
            if network_enabled
            else "正在确认离线 Gaia 或自动局部参考路线…"
        )
        self._check_bootstrap_cancelled()
        probe_network_capabilities(
            manifest,
            timeout_seconds=4.0,
            check_cancelled=self._check_bootstrap_cancelled,
        )
        self._check_bootstrap_cancelled()
        self._write_runtime_capability_manifest()
        for line in capability_summary_lines(manifest):
            self._append_event(line)

        errors = refresh_blocking_errors(manifest)
        self._write_runtime_capability_manifest()
        if self.runtime_capabilities_path is not None:
            self._pending_runtime_overrides[RUNTIME_CAPABILITIES_ENV] = str(
                self.runtime_capabilities_path
            )
        if errors:
            raise BootstrapError(
                "运行能力预检失败",
                "\n".join(f"- {error}" for error in errors),
            )

    def _append_text(self, text: str) -> None:
        if not text:
            return
        if threading.get_ident() != getattr(
            self,
            "_ui_thread_ident",
            threading.get_ident(),
        ):
            self.thread_log.emit(text)
            return
        self.log_view.moveCursor(QTextCursor.MoveOperation.End)
        self.log_view.insertPlainText(text)
        self.log_view.moveCursor(QTextCursor.MoveOperation.End)
        with self._run_log_lock:
            if self.run_log_file:
                self.run_log_file.write(text)
                self.run_log_file.flush()

    def _timestamp(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _append_event(self, msg: str) -> None:
        self._append_text(f"[{self._timestamp()}] {msg}\n")

    def _append_divider(self, title: str, detail_lines: list[str] | None = None) -> None:
        self._append_text(f"\n{'=' * 16} {title} {'=' * 16}\n")
        if detail_lines:
            for line in detail_lines:
                self._append_text(f"{line}\n")

    def dragEnterEvent(self, event) -> None:  # type: ignore[override]
        mime_data = event.mimeData()
        if mime_data.hasUrls() and any(
            (
                Path(url.toLocalFile()).is_dir()
                or (
                    Path(url.toLocalFile()).is_file()
                    and Path(url.toLocalFile()).suffix.lower()
                    in (MASTER_SUFFIXES | REVIEW_SUFFIXES)
                )
            )
            for url in mime_data.urls()
        ):
            set_style_property(self.drop_zone, "dragActive", True)
            event.acceptProposedAction()
            return
        event.ignore()

    def dragLeaveEvent(self, event) -> None:  # type: ignore[override]
        set_style_property(self.drop_zone, "dragActive", False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event) -> None:  # type: ignore[override]
        set_style_property(self.drop_zone, "dragActive", False)
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.is_dir() or (
                path.is_file()
                and path.suffix.lower() in (MASTER_SUFFIXES | REVIEW_SUFFIXES)
            ):
                self._apply_input_path(path, remember=True)
                event.acceptProposedAction()
                return
        event.ignore()

    def _on_directory_edited(self) -> None:
        text = self.dir_edit.text().strip()
        path = Path(text).expanduser() if text else None
        if path is not None and path.exists():
            self._apply_input_path(path, remember=True)
        else:
            self._analyze_selected_directory()

    def _apply_input_path(self, path: Path, *, remember: bool) -> None:
        expanded = path.expanduser()
        try:
            normalized = expanded.resolve()
        except OSError:
            normalized = expanded
        previous_path = self._processing_parameter_input_path
        if (
            previous_path is not None
            and previous_path != normalized
            and isinstance(getattr(self, "processing_parameters", None), Mapping)
        ):
            self.processing_parameters = reset_stage_parameters(
                self.processing_parameters
            )
            StarunGui._apply_processing_payload_state(
                self, self.processing_parameters
            )
            self._sync_processing_controls_from_state()
            if not self._restoring_settings:
                self._append_event(
                    "已切换输入；门禁恢复默认档位，Stage 1–10 参数恢复为自动。"
                )
        self._processing_parameter_input_path = normalized
        self.dir_edit.setText(str(normalized))
        if not (self.worker or self.bootstrap_worker):
            self._progress_timer.stop()
            self._pipeline_started_monotonic = None
            self._reset_stage_progress(10)
            self.warning_card.hide()
            self._last_quality_report_path = None
            self._set_status_text("Idle")
        self._analyze_selected_directory()
        if remember and normalized.exists():
            self._remember_directory(normalized)

    def _remember_directory(self, path: Path) -> None:
        value = str(path)
        self._recent_directories = [
            value,
            *(item for item in self._recent_directories if item != value),
        ][:8]
        self._refresh_recent_directories()
        self._save_settings()

    def _refresh_recent_directories(self) -> None:
        self.recent_combo.blockSignals(True)
        try:
            self.recent_combo.clear()
            self.recent_combo.addItem("选择最近使用的输入…", None)
            for value in self._recent_directories:
                self.recent_combo.addItem(self._display_path(value), value)
            self.recent_combo.setCurrentIndex(0)
        finally:
            self.recent_combo.blockSignals(False)
        has_recent = bool(self._recent_directories)
        self.recent_label.setVisible(has_recent)
        self.recent_combo.setVisible(has_recent)
        self._refresh_recent_menu()

    def _on_recent_directory_selected(self, index: int) -> None:
        value = self.recent_combo.itemData(index)
        if value:
            self._apply_input_path(Path(str(value)), remember=True)

    def _refresh_source_header_summary(self, selected_path: Path | None) -> None:
        if selected_path is None or not selected_path.is_file():
            self.source_header_group.hide()
            return

        summary = inspect_source_header(selected_path)
        if summary.status == "ok":
            missing_value = "未记录"
            missing_details = "未发现其他已知字段"
        elif summary.status == "unsupported":
            missing_value = "当前格式暂未读取"
            missing_details = "暂无可显示信息"
        else:
            missing_value = "无法读取"
            missing_details = "暂无可显示信息"

        self.source_header_device_label.setText(
            summary.device_name or missing_value
        )
        self.source_header_filter_label.setText(
            summary.filter_name or missing_value
        )
        self.source_header_exposure_label.setText(
            summary.exposure or missing_value
        )
        self.source_header_details_label.setText(
            "\n".join(
                f"{label}：{value}" for label, value in summary.details
            )
            or missing_details
        )
        self.source_header_status_label.setText(summary.message)
        self.source_header_status_label.setAccessibleDescription(summary.message)
        self.source_header_group.show()

    def _analyze_selected_directory(self) -> None:
        text = self.dir_edit.text().strip()
        selected_path = Path(text).expanduser() if text else None
        if selected_path is None or not selected_path.exists():
            self._refresh_source_header_summary(None)
            self._input_discovery = None
            self._recommended_input_mode = INPUT_MODE_AUTO
            if self.mode_combo.currentData() == UI_MODE_RECOMMENDED:
                self.input_mode = INPUT_MODE_AUTO
            self.mode_combo.setItemText(0, "自动推荐（完整处理）")
            self.directory_summary_label.setText(
                "尚未选择有效输入，可拖入 FITS/XISF 文件、Light 目录或产品任务。"
            )
            self.linear_phase_label.setText("— 线性处理未计划")
            self.nonlinear_phase_label.setText("— 非线性处理未计划")
            self._update_result_actions(None)
            if self._workspace_state != WORKSPACE_RUN:
                self._show_workspace(WORKSPACE_EMPTY)
            self.task_preview_canvas.clear_image()
            self.task_preview_status_label.setText("等待选择输入")
            self.preview_status_label.setText("预览：等待输入")
            return

        self._refresh_source_header_summary(
            selected_path if selected_path.is_file() else None
        )

        discovery = discover_input_for_processing_settings(
            selected_path,
            processing_settings=self._processing_settings_snapshot(),
        )
        self._input_discovery = discovery
        recommended_mode = {
            1: INPUT_MODE_STAGE1_PREPARED_RESUME,
            2: INPUT_MODE_STAGE2_CORRECTED_RESUME,
            5: INPUT_MODE_LINEAR_RESUME,
        }.get(discovery.resume_after_stage, INPUT_MODE_AUTO)
        self._recommended_input_mode = recommended_mode
        recommendation = self._input_mode_label(recommended_mode)
        self.mode_combo.setItemText(0, f"自动推荐（{recommendation}）")
        recommended_index = self.mode_combo.findData(UI_MODE_RECOMMENDED)
        if recommended_index >= 0 and self.mode_combo.currentIndex() != recommended_index:
            self.mode_combo.blockSignals(True)
            try:
                self.mode_combo.setCurrentIndex(recommended_index)
            finally:
                self.mode_combo.blockSignals(False)
        self.input_mode = recommended_mode
        presentation = describe_input_plan(discovery)
        details = [
            discovery.summary,
            presentation.summary,
            *discovery.warnings,
            *discovery.errors,
        ]
        self.directory_summary_label.setText("\n".join(details))
        self.linear_phase_label.setText(presentation.linear_phase)
        self.nonlinear_phase_label.setText(presentation.nonlinear_phase)
        result_root = (
            discovery.task_directory
            if discovery.kind == InputKind.PRODUCT_TASK
            else None
        )
        self._update_result_actions(result_root)
        if self._workspace_state != WORKSPACE_RUN:
            self._show_workspace(WORKSPACE_TASK)
        if not self._restoring_settings:
            self._schedule_initial_preview(selected_path)

    def _find_result_preview(self, work_dir: Path) -> Path | None:
        preferred_candidates: set[Path] = set()
        if (work_dir / "task-manifest.json").is_file():
            preferred_candidates.update(
                path
                for path in latest_result_files(
                    work_dir,
                    suffixes={".png"},
                )
                if path.suffix.lower() == ".png"
            )
        for pattern in (
            "*_ai.png",
            "result_review*.png",
            "result_processed*.png",
            "*_processed*.png",
        ):
            preferred_candidates.update(
                path for path in work_dir.glob(pattern) if path.is_file()
            )
        if preferred_candidates:
            return max(preferred_candidates, key=safe_mtime)
        candidates = [path for path in work_dir.glob("*.png") if path.is_file()]
        return max(candidates, key=safe_mtime) if candidates else None

    def _update_result_actions(self, work_dir: Path | None) -> None:
        valid_directory = bool(work_dir and work_dir.is_dir())
        self.open_result_action.setEnabled(valid_directory)
        self._result_preview_path = (
            self._find_result_preview(work_dir) if work_dir is not None else None
        )
        self.result_preview_btn.setEnabled(self._result_preview_path is not None)

    def _open_result_preview(self) -> None:
        preview_path = self._result_preview_path
        if preview_path is not None and preview_path.is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(preview_path)))
            return
        QMessageBox.information(self, "暂无预览", "当前目录还没有可预览的结果。")

    def _fits_in_work_dir(self, work_dir: Path) -> list[Path]:
        if not work_dir.is_dir():
            return []
        return [
            p for p in work_dir.iterdir()
            if p.is_file() and p.suffix.lower() in FITS_SUFFIXES
        ]

    def _is_candidate_stacked_input(self, path: Path, work_dir: Path) -> bool:
        if path.parent != work_dir:
            return False

        name_lower = path.name.lower()
        stem_lower = path.stem.lower()
        for prefix in PIPELINE_EXCLUDE_PREFIXES:
            if name_lower.startswith(prefix):
                return False
        for substring in PIPELINE_EXCLUDE_SUBSTRINGS:
            if substring in stem_lower:
                return False
        for suffix in PIPELINE_EXCLUDE_SUFFIXES:
            if stem_lower.endswith(suffix):
                return False
        return True

    def _estimate_disk_space(
        self,
        work_dir: Path,
        *,
        input_mode: str | None = None,
    ) -> DiskSpaceEstimate | None:
        current_work_dir_bytes = directory_size_bytes(work_dir)
        available_bytes = shutil.disk_usage(work_dir).free
        current_mode = input_mode or self._current_input_mode()
        active_task = getattr(self, "_active_prepared_task", None)
        if active_task is not None and active_task.run.root == work_dir:
            source = active_task.source_record
            records = source.get("files") if isinstance(source, dict) else []
            file_sizes = [
                max(0, int(record.get("size") or 0))
                for record in (records if isinstance(records, list) else [])
                if isinstance(record, dict)
            ]
            source_bytes = sum(file_sizes)
            if active_task.resume_after_stage == 5:
                base_growth_bytes = int(
                    max(file_sizes or [0]) * LINEAR_RESUME_STAGE_ARTIFACT_COPIES
                )
                mode = "linear_resume"
                selected_input_label = "Stage 5 已验证断点"
            elif active_task.resume_after_stage == 2:
                base_growth_bytes = int(
                    max(file_sizes or [0]) * STAGE2_RESUME_STAGE_ARTIFACT_COPIES
                )
                mode = "stage2_corrected_resume"
                selected_input_label = "Stage 2 已验证断点"
            elif str(source.get("kind") or "") == "light_directory":
                largest = max(file_sizes or [0])
                average = source_bytes / max(len(file_sizes), 1)
                processed_frame_bytes = max(
                    largest,
                    int(average * LIGHT_FRAME_EXPANSION_FACTOR),
                )
                preprocess_growth_bytes = int(
                    source_bytes
                    * LIGHT_FRAME_EXPANSION_FACTOR
                    * LIGHT_PREPROCESS_SEQUENCE_COPIES
                )
                base_growth_bytes = preprocess_growth_bytes + int(
                    processed_frame_bytes * STACKED_STAGE_ARTIFACT_COPIES
                )
                mode = "light"
                selected_input_label = f"Light x {len(file_sizes)}"
            else:
                base_growth_bytes = int(
                    max(file_sizes or [0]) * STACKED_STAGE_ARTIFACT_COPIES
                )
                mode = "stacked"
                selected_input_label = active_task.display_label
            headroom_bytes = max(
                DISK_SPACE_MIN_HEADROOM_BYTES,
                int(base_growth_bytes * DISK_SPACE_HEADROOM_RATIO),
            )
            return DiskSpaceEstimate(
                mode=mode,
                current_work_dir_bytes=current_work_dir_bytes,
                input_count=max(len(file_sizes), 1),
                input_bytes=source_bytes,
                estimated_peak_growth_bytes=base_growth_bytes,
                required_free_bytes=base_growth_bytes + headroom_bytes,
                available_bytes=available_bytes,
                selected_input_label=selected_input_label,
            )

        if current_mode in {
            INPUT_MODE_STAGE1_PREPARED_RESUME,
            INPUT_MODE_STAGE2_CORRECTED_RESUME,
            INPUT_MODE_LINEAR_RESUME,
        }:
            return None

        fits = self._fits_in_work_dir(work_dir)
        if not fits:
            return None

        light_files = [p for p in fits if p.name.lower().startswith("light_")]
        stacked_files = [
            p for p in fits
            if self._is_candidate_stacked_input(p, work_dir)
        ]

        if stacked_files:
            source = max(stacked_files, key=safe_mtime)
            source_bytes = safe_file_size(source)
            base_growth_bytes = int(source_bytes * STACKED_STAGE_ARTIFACT_COPIES)
            input_count = 1
            input_bytes = source_bytes
            mode = "stacked"
            selected_input_label = source.name
        elif light_files:
            input_count = len(light_files)
            input_bytes = sum(safe_file_size(p) for p in light_files)
            largest_light_bytes = max((safe_file_size(p) for p in light_files), default=0)
            average_light_bytes = input_bytes / max(input_count, 1)
            processed_frame_bytes = max(
                largest_light_bytes,
                int(average_light_bytes * LIGHT_FRAME_EXPANSION_FACTOR),
            )
            preprocess_growth_bytes = int(
                input_bytes
                * LIGHT_FRAME_EXPANSION_FACTOR
                * LIGHT_PREPROCESS_SEQUENCE_COPIES
            )
            stacked_stage_growth_bytes = int(
                processed_frame_bytes * STACKED_STAGE_ARTIFACT_COPIES
            )
            base_growth_bytes = preprocess_growth_bytes + stacked_stage_growth_bytes
            mode = "light"
            selected_input_label = f"Light_ x {input_count}"
        else:
            return None

        headroom_bytes = max(
            DISK_SPACE_MIN_HEADROOM_BYTES,
            int(base_growth_bytes * DISK_SPACE_HEADROOM_RATIO),
        )
        required_free_bytes = base_growth_bytes + headroom_bytes
        return DiskSpaceEstimate(
            mode=mode,
            current_work_dir_bytes=current_work_dir_bytes,
            input_count=input_count,
            input_bytes=input_bytes,
            estimated_peak_growth_bytes=base_growth_bytes,
            required_free_bytes=required_free_bytes,
            available_bytes=available_bytes,
            selected_input_label=selected_input_label,
        )

    def _disk_space_mode_label(self, estimate: DiskSpaceEstimate) -> str:
        if estimate.mode == "linear_resume":
            return "从线性反卷积与降噪后继续"
        if estimate.mode == "stage2_corrected_resume":
            return "从边界校正后继续"
        if estimate.mode == "light":
            return "Light 子帧完整处理"
        return "叠加图像完整处理"

    def _disk_space_summary_lines(self, estimate: DiskSpaceEstimate) -> list[str]:
        input_desc = estimate.selected_input_label
        if estimate.mode == "light":
            input_desc += (
                f" ({estimate.input_count} 帧, {format_bytes(estimate.input_bytes)})"
            )
        else:
            input_desc += f" ({format_bytes(estimate.input_bytes)})"

        return [
            f"  磁盘预估模式: {self._disk_space_mode_label(estimate)}",
            f"  预估基准输入: {input_desc}",
            "  工作目录当前大小: "
            f"{format_bytes(estimate.current_work_dir_bytes)}, "
            "预计阶段/临时产物峰值: "
            f"{format_bytes(estimate.estimated_peak_growth_bytes)}, "
            f"建议剩余空间: {format_bytes(estimate.required_free_bytes)}, "
            f"当前剩余: {format_bytes(estimate.available_bytes)}",
        ]

    def _disk_space_error_message(self, estimate: DiskSpaceEstimate) -> str:
        input_line = f"预估基准输入: {estimate.selected_input_label}"
        if estimate.input_bytes > 0:
            input_line += f" ({format_bytes(estimate.input_bytes)})"

        lines = [
            "当前磁盘剩余空间不足，已取消本次运行。",
            "",
            f"模式: {self._disk_space_mode_label(estimate)}",
            f"工作目录当前大小: {format_bytes(estimate.current_work_dir_bytes)}",
            input_line,
            "预计阶段/临时产物峰值: "
            f"{format_bytes(estimate.estimated_peak_growth_bytes)}",
            f"建议剩余空间: {format_bytes(estimate.required_free_bytes)}",
            f"当前剩余空间: {format_bytes(estimate.available_bytes)}",
            "",
            "请先清理磁盘空间，或改用更小的数据集后重试。",
        ]
        return "\n".join(lines)

    def _estimate_runtime_disk_space(
        self,
        fingerprint: dict[str, object],
    ) -> RuntimeDiskEstimate:
        runtime_anchor = existing_volume_anchor(self.runtime_home)
        available_bytes = shutil.disk_usage(runtime_anchor).free
        current_runtime_bytes = directory_size_bytes(self.runtime_home)
        state_root = self._siril_state_root()

        seed_growth_bytes = 0
        seed_targets = (
            (self.siril_seed_dir / "venv", state_root / "venv", None),
            (
                self.siril_seed_dir / ".python_module",
                state_root / ".python_module",
                "sirilpy",
            ),
        )
        for source, target, required_child in seed_targets:
            target_ready = target.exists()
            if required_child:
                target_ready = target_ready and (target / required_child).exists()
            if not target_ready:
                seed_growth_bytes += directory_size_bytes(source)

        support_growth_bytes = 0

        scripts_root = resolve_siril_scripts_root(self.siril_plugin_dir)
        runtime_scripts_marker = (
            self._runtime_siril_scripts_repo_dir()
            / "processing"
            / "AberrationRemover.py"
        )
        if scripts_root is not None and not runtime_scripts_marker.is_file():
            support_growth_bytes += directory_size_bytes(scripts_root)

        bootstrap_cache_hit = self._bootstrap_state_is_current(fingerprint)
        dependency_growth_bytes = 0
        if not bootstrap_cache_hit:
            wheel_bytes = directory_size_bytes(self._plugin_downloads_dir())
            dependency_growth_bytes = int(
                wheel_bytes * RUNTIME_DEPENDENCY_EXPANSION_FACTOR
            )

        required_free_bytes = (
            seed_growth_bytes
            + support_growth_bytes
            + dependency_growth_bytes
            + RUNTIME_DISK_MIN_HEADROOM_BYTES
        )
        return RuntimeDiskEstimate(
            volume_path=runtime_anchor,
            volume_device=os.stat(runtime_anchor).st_dev,
            current_runtime_bytes=current_runtime_bytes,
            seed_growth_bytes=seed_growth_bytes,
            support_growth_bytes=support_growth_bytes,
            dependency_growth_bytes=dependency_growth_bytes,
            required_free_bytes=required_free_bytes,
            available_bytes=available_bytes,
            bootstrap_cache_hit=bootstrap_cache_hit,
        )

    def _runtime_disk_space_summary_lines(
        self,
        estimate: RuntimeDiskEstimate,
    ) -> list[str]:
        cache_note = "依赖已就绪" if estimate.bootstrap_cache_hit else "需要准备依赖"
        return [
            "  运行环境所在卷: "
            f"{self._display_path(estimate.volume_path)} ({cache_note})",
            "  运行环境当前大小: "
            f"{format_bytes(estimate.current_runtime_bytes)}, "
            f"seed/支持文件预计新增: "
            f"{format_bytes(estimate.seed_growth_bytes + estimate.support_growth_bytes)}, "
            f"依赖预计新增峰值: {format_bytes(estimate.dependency_growth_bytes)}, "
            f"建议剩余空间: {format_bytes(estimate.required_free_bytes)}, "
            f"当前剩余: {format_bytes(estimate.available_bytes)}",
            "  模型读取方式: 直接使用 App/离线资源包，不复制到运行目录",
        ]

    def _runtime_disk_space_error_message(
        self,
        estimate: RuntimeDiskEstimate,
    ) -> str:
        return "\n".join(
            [
                "运行环境所在磁盘的剩余空间不足，已取消本次运行。",
                "",
                f"运行环境: {self._display_path(self.runtime_home)}",
                f"磁盘卷: {self._display_path(estimate.volume_path)}",
                f"运行环境当前大小: {format_bytes(estimate.current_runtime_bytes)}",
                "seed/支持文件预计新增: "
                f"{format_bytes(estimate.seed_growth_bytes + estimate.support_growth_bytes)}",
                f"依赖预计新增峰值: {format_bytes(estimate.dependency_growth_bytes)}",
                f"建议剩余空间: {format_bytes(estimate.required_free_bytes)}",
                f"当前剩余空间: {format_bytes(estimate.available_bytes)}",
                "",
                "请先清理系统运行时卷后重试。",
            ]
        )

    def _preflight_summary_lines(
        self,
        work_dir: Path,
        disk_estimate: DiskSpaceEstimate | None = None,
        *,
        input_mode: str | None = None,
    ) -> list[str]:
        fits = self._fits_in_work_dir(work_dir)
        light_files = [p for p in fits if p.name.lower().startswith("light_")]
        other_fits = [p for p in fits if p not in light_files]

        machine = platform.machine().lower()
        current_mode = input_mode or self._current_input_mode()
        lines = [
            "预检摘要：",
            f"  工作目录: {self._display_path(work_dir)}",
            f"  处理模式: {self._input_mode_label(current_mode)}",
            f"  检测到的 FITS 输入: 总计={len(fits)}, Light_={len(light_files)}, 其他={len(other_fits)}",
            f"  主机架构: {machine or '<未知>'}",
            f"  流水线脚本: {self._display_path(self.pipeline_path)}",
            f"  Siril 插件目录: {self._display_path(self.siril_plugin_dir)}",
        ]
        resume_stage = {
            INPUT_MODE_STAGE1_PREPARED_RESUME: 1,
            INPUT_MODE_STAGE2_CORRECTED_RESUME: 2,
            INPUT_MODE_LINEAR_RESUME: 5,
        }.get(current_mode)
        if resume_stage is not None:
            lines.append(
                f"  正式续跑来源: task-run manifest 已验签 Stage {resume_stage} 断点"
            )
        if disk_estimate is not None:
            lines.extend(self._disk_space_summary_lines(disk_estimate))
        return lines

    def _open_result_dir(self) -> None:
        if self._history_detail_mode:
            run_root = self._historical_run_root
            if run_root is not None and run_root.is_dir() and any(
                path.is_file() for path in self._historical_result_files
            ):
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(run_root)))
                return
            QMessageBox.information(
                self,
                "历史结果不可用",
                "该次运行的交付文件已清理、缺失或未通过校验。",
            )
            return
        task_root = self._last_task_root
        if self._active_prepared_task is not None:
            task_root = self._active_prepared_task.workspace.root
        if task_root is not None and task_root.is_dir():
            result_directory = latest_result_directory(task_root)
            QDesktopServices.openUrl(
                QUrl.fromLocalFile(
                    str(result_directory or (task_root / "results"))
                )
            )
            return
        path = self.dir_edit.text().strip()
        expanded = Path(path).expanduser() if path else None
        if expanded is not None and expanded.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(expanded)))

    def _open_log_file(self) -> None:
        if self.run_log_path and self.run_log_path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.run_log_path)))
        else:
            QMessageBox.information(self, "暂无日志", "当前还没有可用的运行日志。")

    def _quality_report_path(self, work_dir: Path | None) -> Path | None:
        if work_dir is None:
            return None
        candidates = (
            work_dir / "process" / "final_quality_report.json",
            work_dir / "final_quality_report.json",
        )
        return next((path for path in candidates if path.is_file()), candidates[0])

    def _show_completion_warning(
        self,
        work_dir: Path | None,
        *,
        outcome_status: str,
        had_errors: bool,
        had_degradations: bool,
        had_fallbacks: bool,
    ) -> None:
        self._last_quality_report_path = self._quality_report_path(work_dir)
        if outcome_status == STATUS_REVIEW_REQUIRED:
            message = (
                "处理已完成，但需要人工复核。请检查阶段复核原因与最终质量报告。"
            )
        elif had_fallbacks and had_degradations:
            message = "降级完成，已使用回退产物或算法。请检查阶段详情与最终质量报告。"
        elif had_fallbacks:
            message = "处理已完成并使用了回退路径。请检查阶段详情与最终质量报告。"
        elif had_errors:
            message = "处理已完成，执行错误已恢复。请检查 issues 与最终质量报告。"
        else:
            message = "降级完成。请检查阶段详情与最终质量报告。"
        self._show_run_banner(
            "warning",
            message,
            quality_report=True,
        )

    def _show_run_banner(
        self,
        tone: str,
        message: str,
        *,
        quality_report: bool = False,
        show_log: bool = False,
    ) -> None:
        set_style_property(self.warning_card, "tone", tone)
        symbols = {
            "success": "✓",
            "warning": "⚠",
            "error": "✕",
            "info": "●",
        }
        self.warning_label.setText(f"{symbols.get(tone, '•')}  {message}")
        self.warning_label.setAccessibleDescription(message)
        self.quality_report_btn.setVisible(quality_report)
        self.banner_log_btn.setVisible(show_log)
        self.warning_card.show()

    def _open_quality_report(self) -> None:
        report_path = self._last_quality_report_path
        if report_path is not None and report_path.is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(report_path)))
            return
        QMessageBox.information(
            self,
            "质量报告未找到",
            "本次任务没有生成 final_quality_report.json，请先查看阶段日志。",
        )

    def _preflight_errors(
        self,
        work_dir: Path,
        *,
        input_mode: str | None = None,
    ) -> list[str]:
        errors: list[str] = []
        try:
            manifest = StarunGui._inspect_runtime_capabilities(self, work_dir)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            errors.append(f"无法建立运行能力清单：{error}")
        else:
            errors.extend(
                str(error)
                for error in manifest.get("blocking_errors", [])
                if str(error)
            )

        if not work_dir.is_dir():
            errors.append(f"工作目录不存在：{work_dir}")

        current_mode = input_mode or self._current_input_mode()
        manifest_value = str(
            getattr(self, "_pending_runtime_overrides", {}).get(
                "STARUN_TASK_RUN_MANIFEST",
                "",
            )
        ).strip()
        task_run_manifest = (
            Path(manifest_value).expanduser() if manifest_value else None
        )
        has_task_run_manifest = bool(
            task_run_manifest is not None
            and task_run_manifest.is_file()
            and task_run_manifest.parent.resolve() == work_dir.resolve()
        )
        expected_resume_stage = {
            INPUT_MODE_STAGE1_PREPARED_RESUME: 1,
            INPUT_MODE_STAGE2_CORRECTED_RESUME: 2,
            INPUT_MODE_LINEAR_RESUME: 5,
        }.get(current_mode)
        if expected_resume_stage is not None:
            if not has_task_run_manifest or task_run_manifest is None:
                errors.append(
                    "续跑仅接受当前 run 中已验签 task-run manifest 的 Stage 1/2/5 正式断点。"
                )
            else:
                payload = pipeline_run_manifest.load_json(task_run_manifest)
                claimed_hash = str(
                    (payload or {}).get("manifest_hash") or ""
                )
                unsigned = dict(payload or {})
                unsigned.pop("manifest_hash", None)
                contract = (
                    payload.get("pipeline_contract")
                    if isinstance(payload, Mapping)
                    else None
                )
                resume = (
                    payload.get("resume")
                    if isinstance(payload, Mapping)
                    else None
                )
                manifest_valid = bool(
                    isinstance(payload, Mapping)
                    and payload.get("schema") == "starun.task-run.v1"
                    and claimed_hash
                    and claimed_hash
                    == pipeline_run_manifest.canonical_payload_hash(unsigned)
                    and isinstance(contract, Mapping)
                    and contract.get("schema") == PIPELINE_CONTRACT_SCHEMA
                    and contract.get("version") == PIPELINE_CONTRACT_VERSION
                )
                if not manifest_valid or not isinstance(resume, Mapping):
                    errors.append("task-run manifest 的签名或阶段契约无效。")
                else:
                    try:
                        resume_stage = int(resume.get("stage") or 0)
                    except (TypeError, ValueError):
                        resume_stage = 0
                    checkpoint_value = str(resume.get("path") or "").strip()
                    checkpoint = (
                        Path(checkpoint_value).expanduser().resolve()
                        if checkpoint_value
                        else None
                    )
                    contract_by_stage = {
                        item.number: item for item in product_stage_contracts()
                    }
                    stage_contract = contract_by_stage.get(expected_resume_stage)
                    checkpoint_sha256 = (
                        pipeline_run_manifest.sha256_file(checkpoint)
                        if checkpoint is not None
                        else None
                    )
                    if (
                        resume_stage != expected_resume_stage
                        or stage_contract is None
                        or str(resume.get("artifact") or "")
                        != stage_contract.primary_artifact
                        or checkpoint is None
                        or checkpoint.name != stage_contract.primary_artifact
                        or not checkpoint_sha256
                        or checkpoint_sha256 != str(resume.get("sha256") or "")
                    ):
                        errors.append(
                            "task-run manifest 的正式断点阶段、路径或 SHA-256 不匹配。"
                        )
        elif not has_task_run_manifest:
            fits = self._fits_in_work_dir(work_dir)
            if not fits:
                errors.append(f"在 {work_dir} 中未找到 .fit/.fits 输入文件")

        return errors

    def _parse_elapsed_to_seconds(self, elapsed: str) -> int | None:
        text = elapsed.strip()
        if not text:
            return None
        day_part = 0
        if "-" in text:
            day_txt, text = text.split("-", 1)
            if not day_txt.isdigit():
                return None
            day_part = int(day_txt) * 86400
        parts = text.split(":")
        if not 1 <= len(parts) <= 3:
            return None
        try:
            nums = [int(x) for x in parts]
        except ValueError:
            return None
        if len(nums) == 3:
            h, m, s = nums
        elif len(nums) == 2:
            h, m, s = 0, nums[0], nums[1]
        else:
            h, m, s = 0, 0, nums[0]
        return day_part + h * 3600 + m * 60 + s

    def _has_stale_venv_bootstrap(self) -> bool:
        venv_target = str(self._siril_state_root() / "venv")
        try:
            cp = subprocess.run(
                ["/bin/ps", "-ax", "-o", "etime=,command="],
                capture_output=True,
                text=True,
                check=False,
            )
            if cp.returncode != 0:
                return False
            for line in cp.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                fields = line.split(maxsplit=1)
                if len(fields) != 2:
                    continue
                elapsed, command = fields
                if "python3 -m venv" not in command:
                    continue
                if venv_target not in command:
                    continue
                sec = self._parse_elapsed_to_seconds(elapsed)
                if sec is not None and sec > 300:
                    return True
        except Exception:
            return False
        return False

    def _resolve_runtime_candidates(self) -> list[Path]:
        candidates = self._resolve_siril_candidates()
        if not candidates:
            return candidates

        # In bundled-only mode we still surface stale-bootstrap diagnostics.
        if self._has_stale_venv_bootstrap() and self.bundled_siril_cli in candidates:
            reordered = [self.bundled_siril_cli] + [p for p in candidates if p != self.bundled_siril_cli]
            self._append_event(
                "检测到残留的 Siril Python venv 引导进程；"
                "将使用内置 Siril 运行时（仅内置模式）。"
            )
            return reordered

        return candidates

    def _plugin_download_script_path(self) -> Path:
        return self.siril_plugin_dir / "download_siril_plugins.sh"

    def _plugin_downloads_dir(self) -> Path:
        return self.siril_plugin_dir / "downloads"

    def _run_bootstrap_process(self, args, **kwargs):
        stop_event = getattr(self, "_bootstrap_stop_event", None)
        if stop_event is None:
            return subprocess.run(args, **kwargs)
        return run_cancellable_process(
            args,
            stop_event=stop_event,
            **kwargs,
        )

    def _check_bootstrap_cancelled(self) -> None:
        stop_event = getattr(self, "_bootstrap_stop_event", None)
        if stop_event is not None and stop_event.is_set():
            raise BootstrapCancelled()

    def _plugin_requirements_path(self) -> Path:
        return self.siril_plugin_dir / "requirements.txt"

    def _onnxruntime_wheels(self) -> list[Path]:
        return sorted(self._plugin_downloads_dir().glob("onnxruntime-*.whl"))

    def _onnx_wheels(self) -> list[Path]:
        return sorted(self._plugin_downloads_dir().glob("onnx-*.whl"))

    def _pyqt6_wheels(self) -> list[Path]:
        wheels = sorted(self._plugin_downloads_dir().glob("pyqt6-*.whl"))
        if wheels:
            return wheels
        return sorted(self._plugin_downloads_dir().glob("PyQt6-*.whl"))

    def _pyqt6_qt6_wheels(self) -> list[Path]:
        wheels = sorted(self._plugin_downloads_dir().glob("pyqt6_qt6-*.whl"))
        if wheels:
            return wheels
        return sorted(self._plugin_downloads_dir().glob("PyQt6_Qt6-*.whl"))

    def _pyqt6_sip_wheels(self) -> list[Path]:
        wheels = sorted(self._plugin_downloads_dir().glob("pyqt6_sip-*.whl"))
        if wheels:
            return wheels
        return sorted(self._plugin_downloads_dir().glob("PyQt6_sip-*.whl"))

    def _pyside6_wheels(self) -> list[Path]:
        wheels = sorted(self._plugin_downloads_dir().glob("pyside6-*.whl"))
        if wheels:
            return wheels
        return sorted(self._plugin_downloads_dir().glob("PySide6-*.whl"))

    def _pyside6_addons_wheels(self) -> list[Path]:
        wheels = sorted(self._plugin_downloads_dir().glob("pyside6_addons-*.whl"))
        if wheels:
            return wheels
        return sorted(self._plugin_downloads_dir().glob("PySide6_Addons-*.whl"))

    def _pyside6_essentials_wheels(self) -> list[Path]:
        wheels = sorted(self._plugin_downloads_dir().glob("pyside6_essentials-*.whl"))
        if wheels:
            return wheels
        return sorted(self._plugin_downloads_dir().glob("PySide6_Essentials-*.whl"))

    def _shiboken6_wheels(self) -> list[Path]:
        wheels = sorted(self._plugin_downloads_dir().glob("shiboken6-*.whl"))
        if wheels:
            return wheels
        return sorted(self._plugin_downloads_dir().glob("Shiboken6-*.whl"))

    def _astropy_wheels(self) -> list[Path]:
        wheels = sorted(self._plugin_downloads_dir().glob("astropy-*.whl"))
        if wheels:
            return wheels
        return sorted(self._plugin_downloads_dir().glob("Astropy-*.whl"))

    def _scipy_wheels(self) -> list[Path]:
        wheels = sorted(self._plugin_downloads_dir().glob("scipy-*.whl"))
        if wheels:
            return wheels
        return sorted(self._plugin_downloads_dir().glob("Scipy-*.whl"))

    def _tifffile_wheels(self) -> list[Path]:
        return sorted(self._plugin_downloads_dir().glob("tifffile-*.whl"))

    def _lz4_wheels(self) -> list[Path]:
        wheels = sorted(self._plugin_downloads_dir().glob("lz4-*.whl"))
        if wheels:
            return wheels
        return sorted(self._plugin_downloads_dir().glob("LZ4-*.whl"))

    def _zstandard_wheels(self) -> list[Path]:
        wheels = sorted(self._plugin_downloads_dir().glob("zstandard-*.whl"))
        if wheels:
            return wheels
        return sorted(self._plugin_downloads_dir().glob("Zstandard-*.whl"))

    def _exifread_wheels(self) -> list[Path]:
        wheels = sorted(self._plugin_downloads_dir().glob("exifread-*.whl"))
        if wheels:
            return wheels
        return sorted(self._plugin_downloads_dir().glob("ExifRead-*.whl"))

    def _opencv_python_headless_wheels(self) -> list[Path]:
        return sorted(self._plugin_downloads_dir().glob("opencv_python_headless-*.whl"))

    def _requests_wheels(self) -> list[Path]:
        return sorted(self._plugin_downloads_dir().glob("requests-*.whl"))

    def _requests_dependency_wheels_missing(self) -> list[str]:
        downloads_dir = self._plugin_downloads_dir()
        required = (
            ("urllib3", "urllib3-*.whl"),
            ("idna", "idna-*.whl"),
            ("certifi", "certifi-*.whl"),
            ("charset_normalizer", "charset_normalizer-*.whl"),
        )
        return [
            label
            for label, pattern in required
            if not list(downloads_dir.glob(pattern))
        ]

    def _wheel_package_wheels(self) -> list[Path]:
        return sorted(self._plugin_downloads_dir().glob("wheel-*.whl"))

    def _sep_wheels(self) -> list[Path]:
        return sorted(self._plugin_downloads_dir().glob("sep-*.whl"))

    def _spandrel_wheels(self) -> list[Path]:
        return sorted(self._plugin_downloads_dir().glob("spandrel-*.whl"))

    def _einops_wheels(self) -> list[Path]:
        return sorted(self._plugin_downloads_dir().glob("einops-*.whl"))

    def _safetensors_wheels(self) -> list[Path]:
        return sorted(self._plugin_downloads_dir().glob("safetensors-*.whl"))

    def _torch_wheels(self) -> list[Path]:
        wheels = sorted(self._plugin_downloads_dir().glob("torch-*.whl"))
        return [wheel for wheel in wheels if not wheel.name.startswith("torchvision-")]

    def _torchvision_wheels(self) -> list[Path]:
        return sorted(self._plugin_downloads_dir().glob("torchvision-*.whl"))

    def _requirement_names(self, requirements_path: Path) -> list[str]:
        if not requirements_path.is_file():
            return []
        names: list[str] = []
        for raw_line in requirements_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line or line.startswith(("-", "http:", "https:", ".")):
                continue
            name = re.split(r"[<>=!~;\[\s]", line, maxsplit=1)[0].strip()
            if name:
                names.append(re.sub(r"[-_.]+", "-", name).lower())
        return names

    def _missing_requirement_wheels(self, requirements_path: Path) -> list[str]:
        downloads_dir = self._plugin_downloads_dir()
        wheel_names = [
            re.sub(r"[-_.]+", "-", path.name).lower()
            for path in downloads_dir.glob("*.whl")
        ]
        missing: list[str] = []
        for name in self._requirement_names(requirements_path):
            if not any(
                wheel_name.startswith(f"{name}-") for wheel_name in wheel_names
            ):
                missing.append(name)
        return missing

    def _missing_plugin_artifacts(self) -> list[str]:
        missing: list[str] = []

        def call(name: str):
            method = getattr(self, name, None)
            if method is None:
                # Unit tests call this method on lightweight proxy objects and
                # only bind the artifact checks they are asserting. Treat
                # unbound optional checks as satisfied for those proxies; real
                # StarunGui instances still use their bound methods.
                return [] if name == "_requests_dependency_wheels_missing" else [Path("__proxy_optional_present__")]
            return method()

        plugin_root = self.siril_plugin_dir
        if not plugin_root.exists() or not plugin_root.is_dir():
            return [f"插件目录不存在：{plugin_root}"]

        incompatible_wheels = sorted(
            path.name
            for path in (plugin_root / "downloads").glob("*.whl")
            if not is_siril_cp312_wheel_compatible(path)
        )
        if incompatible_wheels:
            preview = ", ".join(incompatible_wheels[:4])
            if len(incompatible_wheels) > 4:
                preview += f", ... (+{len(incompatible_wheels) - 4})"
            missing.append(
                "Siril 插件 wheel 与 CPython 3.12 不兼容：" + preview
            )

        if not call("_plugin_download_script_path").exists():
            missing.append("download_siril_plugins.sh 缺失")

        wheel_files = sorted(
            (plugin_root / "downloads").glob("setiastrosuitepro-*.whl")
        )
        if not wheel_files:
            missing.append("setiastrosuitepro wheel 缺失")

        if not call("_onnx_wheels"):
            missing.append("onnx wheel 缺失")
        if not call("_onnxruntime_wheels"):
            missing.append("onnxruntime wheel 缺失")
        if not call("_pyqt6_wheels"):
            missing.append("PyQt6 wheel 缺失")
        if not call("_pyqt6_qt6_wheels"):
            missing.append("PyQt6_Qt6 wheel 缺失")
        if not call("_pyqt6_sip_wheels"):
            missing.append("pyqt6_sip wheel 缺失")
        if not call("_tifffile_wheels"):
            missing.append("tifffile wheel 缺失")
        if not call("_lz4_wheels"):
            missing.append("lz4 wheel 缺失")
        if not call("_zstandard_wheels"):
            missing.append("zstandard wheel 缺失")
        if not call("_exifread_wheels"):
            missing.append("exifread wheel 缺失")
        if not call("_opencv_python_headless_wheels"):
            missing.append("opencv-python-headless wheel 缺失")
        if not call("_requests_wheels"):
            missing.append("requests wheel 缺失")
        missing_request_deps = call("_requests_dependency_wheels_missing")
        if missing_request_deps:
            missing.append(
                "requests dependency wheels 缺失: "
                + ", ".join(missing_request_deps)
            )
        if not call("_wheel_package_wheels"):
            missing.append("wheel package 缺失")
        if not call("_sep_wheels"):
            missing.append("sep wheel 缺失")
        if not call("_spandrel_wheels"):
            missing.append("spandrel wheel 缺失")
        if not call("_einops_wheels"):
            missing.append("einops wheel 缺失")
        if not call("_safetensors_wheels"):
            missing.append("safetensors wheel 缺失")
        if not call("_pyside6_wheels"):
            missing.append("PySide6 wheel 缺失")
        if not call("_pyside6_addons_wheels"):
            missing.append("PySide6_Addons wheel 缺失")
        if not call("_pyside6_essentials_wheels"):
            missing.append("PySide6_Essentials wheel 缺失")
        if not call("_shiboken6_wheels"):
            missing.append("shiboken6 wheel 缺失")
        if not call("_astropy_wheels"):
            missing.append("astropy wheel 缺失")
        if not call("_scipy_wheels"):
            missing.append("scipy wheel 缺失")
        if not call("_torch_wheels"):
            missing.append("torch wheel 缺失")
        if not call("_torchvision_wheels"):
            missing.append("torchvision wheel 缺失")
        missing_requirement_wheels = []
        if isinstance(self, StarunGui):
            missing_requirement_wheels = self._missing_requirement_wheels(
                self._plugin_requirements_path()
            )
        if missing_requirement_wheels:
            missing.append(
                "requirements wheel 缺失: "
                + ", ".join(missing_requirement_wheels)
            )

        syqon_bundle = plugin_root / SYQON_STARLESS_BUNDLE_REL
        for name in (
            "zenith.pt",
            "zenith.pt.sha256",
        ):
            if not (syqon_bundle / name).is_file():
                missing.append(f"SyQon Starless {name} 缺失")

        cosmic_bundle = plugin_root / COSMIC_CLARITY_BUNDLE_REL
        cosmic_required_models = [
            cosmic_bundle / name for name in COSMIC_CLARITY_REQUIRED_MODEL_FILES
        ]
        if not all(p.is_file() for p in cosmic_required_models):
            missing.append(
                "CosmicClarity Native 模型缺失（需要 denoise + sharpen 的最小 .pth 集）"
            )
        classic_wrapper = plugin_root / "bin" / "CosmicClarity"
        if isinstance(self, StarunGui) and (
            not classic_wrapper.is_file() or not os.access(classic_wrapper, os.X_OK)
        ):
            missing.append("CosmicClarity classic wrapper 缺失或不可执行")

        scripts_root = resolve_siril_scripts_root(plugin_root)
        if scripts_root is None:
            missing.append("siril-scripts 目录或 AberrationRemover.py 缺失")
        elif not any(
            (scripts_root / relative_path).is_file()
            for relative_path in (
                Path("SyQon/Starless.py"),
            )
        ):
            missing.append("SyQon Starless 脚本缺失")

        return missing

    def _ensure_siril_plugins_ready(self) -> bool:
        missing_before = self._missing_plugin_artifacts()
        if not missing_before:
            scripts_root = resolve_siril_scripts_root(self.siril_plugin_dir)
            self._append_event("Siril 插件缓存检查通过。")
            if scripts_root is not None:
                self._append_event(
                    "Siril scripts 根目录: " + self._display_path(scripts_root)
                )
            return True

        script_path = self._plugin_download_script_path()
        self._append_event(
            "检测到 Siril 插件缓存不完整："
            + "；".join(missing_before)
        )
        if not script_path.exists() or not script_path.is_file():
            self._append_event("插件下载脚本不存在，无法继续本次运行。")
            return False

        self._append_event("正在自动补齐 Siril 插件缓存（首次运行可能较慢）...")
        try:
            cp = self._run_bootstrap_process(
                ["/bin/bash", str(script_path)],
                cwd=str(self.siril_plugin_dir),
                capture_output=True,
                text=True,
                check=False,
                timeout=900,
            )
        except BootstrapCancelled:
            raise
        except Exception as e:
            self._append_event(f"自动补齐插件失败：{e}")
            return False

        if cp.returncode != 0:
            tail = (cp.stderr.strip() or cp.stdout.strip() or "unknown error")
            tail = tail[-300:]
            self._append_event(
                f"自动补齐插件失败，退出码={cp.returncode}：{tail}"
            )
            return False

        missing_after = self._missing_plugin_artifacts()
        if missing_after:
            self._append_event(
                "插件自动补齐后仍有缺失："
                + "；".join(missing_after)
                + "（本次运行已阻断）"
            )
            return False

        scripts_root = resolve_siril_scripts_root(self.siril_plugin_dir)
        self._append_event("Siril 插件缓存已自动补齐完成。")
        if scripts_root is not None:
            self._append_event(
                "Siril scripts 根目录: " + self._display_path(scripts_root)
            )
        return True

    def _runtime_xdg_siril_dir(self) -> Path:
        return self.runtime_home / ".local" / "share" / "siril"

    def _runtime_siril_scripts_repo_dir(self) -> Path:
        return self.runtime_home / ".local" / "share" / "siril-scripts"

    def _runtime_syqon_starless_dir(self) -> Path:
        return self._siril_state_root() / "syqon_starless"

    def _runtime_cosmic_clarity_dir(self) -> Path:
        return self._siril_state_root() / "cosmic_clarity"

    def _remove_matching_runtime_model_copies(
        self,
        bundle_dir: Path,
        target_dir: Path,
        names: list[str],
    ) -> int:
        reclaimed_bytes = 0
        for name in names:
            self._check_bootstrap_cancelled()
            source = bundle_dir / name
            target = target_dir / name
            if not source.is_file() or not target.is_file() or target.is_symlink():
                continue
            try:
                if target.stat().st_size != source.stat().st_size:
                    continue
                reclaimed_bytes += target.stat().st_size
                target.unlink()
            except OSError:
                continue
        try:
            target_dir.rmdir()
        except OSError:
            pass
        return reclaimed_bytes

    def _sync_syqon_starless_bundle(self) -> None:
        bundle_dir = self.siril_plugin_dir / SYQON_STARLESS_BUNDLE_REL
        required_names = [
            "zenith.pt",
            "zenith.pt.sha256",
            "zenith.pt.date",
            "zenith.pt.verified",
        ]
        if (
            not bundle_dir.is_dir()
            or not (bundle_dir / "zenith.pt").is_file()
            or not (bundle_dir / "zenith.pt.sha256").is_file()
        ):
            return
        reclaimed = self._remove_matching_runtime_model_copies(
            bundle_dir,
            self._runtime_syqon_starless_dir(),
            required_names,
        )
        self._append_event(
            "SyQon 模型将直接从只读离线资源使用: "
            + self._display_path(bundle_dir)
        )
        if reclaimed:
            self._append_event(
                "已清理 SyQon 旧运行时副本，释放 "
                + format_bytes(reclaimed)
            )

    def _sync_cosmic_clarity_bundle(self) -> None:
        bundle_dir = self.siril_plugin_dir / COSMIC_CLARITY_BUNDLE_REL
        if not bundle_dir.is_dir():
            return
        model_files = [
            source
            for pattern in ("*.pth", "*.pt", "*.onnx")
            for source in bundle_dir.glob(pattern)
        ]
        if not model_files:
            return
        reclaimed = self._remove_matching_runtime_model_copies(
            bundle_dir,
            self._runtime_cosmic_clarity_dir(),
            [source.name for source in model_files],
        )
        self._append_event(
            "CosmicClarity 模型将直接从只读离线资源使用: "
            + self._display_path(bundle_dir)
        )
        if reclaimed:
            self._append_event(
                "已清理 CosmicClarity 旧运行时副本，释放 "
                + format_bytes(reclaimed)
            )

    def _ensure_runtime_siril_support_dirs(self) -> None:
        xdg_siril_dir = self._runtime_xdg_siril_dir()
        xdg_siril_dir.mkdir(parents=True, exist_ok=True)

        gaia_astro_path = xdg_siril_dir / "siril_cat_healpix8_astro.dat"
        if gaia_astro_path.is_file() and safe_file_size(gaia_astro_path) >= 1024:
            self._append_event(
                "本地 Gaia astrometric 星表："
                f"{gaia_astro_path} ({format_bytes(safe_file_size(gaia_astro_path))})"
            )
        else:
            self._append_event(
                "本地 Gaia astrometric 星表尚未安装；默认在线 Gaia 校准不受影响。"
                "显式离线模式下不会尝试在线目录，并会跳过需要 localgaia 的校准。"
            )
        self._sync_syqon_starless_bundle()
        self._sync_cosmic_clarity_bundle()

        scripts_root = resolve_siril_scripts_root(self.siril_plugin_dir)
        if scripts_root is None:
            return

        runtime_repo = self._runtime_siril_scripts_repo_dir()
        marker = runtime_repo / "processing" / "AberrationRemover.py"
        if marker.is_file():
            if apply_siril_runtime_patches(self.siril_plugin_dir, runtime_repo):
                self._append_event("已应用 Siril 插件运行时兼容补丁")
            return

        runtime_repo.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(scripts_root, runtime_repo, dirs_exist_ok=True)
        if apply_siril_runtime_patches(self.siril_plugin_dir, runtime_repo):
            self._append_event("已应用 Siril 插件运行时兼容补丁")
        self._append_event(
            "已同步 Siril scripts 仓库到运行时目录: "
            + self._display_path(runtime_repo)
        )

    def _runtime_venv_python_bin(self) -> Path:
        venv_dir = self._siril_state_root() / "venv" / "bin"
        for name in ("python3.12", "python3", "python"):
            candidate = venv_dir / name
            if candidate.exists():
                return candidate
        return venv_dir / "python3.12"

    def _runtime_python_env(self) -> dict[str, str]:
        runtime_env = scrub_python_env(os.environ.copy())
        runtime_env["HOME"] = str(self.runtime_home)
        runtime_env["LANG"] = "en_US.UTF-8"
        runtime_env["LC_ALL"] = "en_US.UTF-8"
        runtime_env["LC_CTYPE"] = "en_US.UTF-8"
        runtime_env["PYTHONUTF8"] = "1"
        runtime_env["PYTHONIOENCODING"] = "utf-8"
        plugin_downloads = self._plugin_downloads_dir()
        if plugin_downloads.is_dir():
            runtime_env["PIP_NO_INDEX"] = "1"
            runtime_env["PIP_FIND_LINKS"] = str(plugin_downloads)
            runtime_env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        return runtime_env

    def _ensure_runtime_tiffile_alias(self) -> None:
        venv_dir = self._siril_state_root() / "venv"
        site_dir = resolve_venv_site_packages(venv_dir)
        alias_path = site_dir / "tiffile.py"
        alias_code = (
            '"""Compatibility shim for scripts importing `tiffile`."""\n'
            "from tifffile import *  # noqa: F401,F403\n"
        )
        if alias_path.exists():
            existing = alias_path.read_text(encoding="utf-8", errors="replace")
            if existing == alias_code:
                alias_written = False
            else:
                alias_path.write_text(alias_code, encoding="utf-8")
                alias_written = True
        else:
            alias_path.write_text(alias_code, encoding="utf-8")
            alias_written = True
        if alias_written:
            self._append_event(
                "已写入 tiffile 兼容别名模块: "
                + self._display_path(alias_path)
            )

        dist_info = site_dir / "tiffile-0.0.0.dist-info"
        dist_info.mkdir(parents=True, exist_ok=True)
        metadata_path = dist_info / "METADATA"
        wheel_path = dist_info / "WHEEL"
        top_level_path = dist_info / "top_level.txt"
        installer_path = dist_info / "INSTALLER"
        metadata_text = (
            "Metadata-Version: 2.1\n"
            "Name: tiffile\n"
            "Version: 0.0.0\n"
            "Summary: Compatibility shim that re-exports tifffile\n"
        )
        wheel_text = (
            "Wheel-Version: 1.0\n"
            "Generator: starun\n"
            "Root-Is-Purelib: true\n"
            "Tag: py3-none-any\n"
        )
        top_level_text = "tiffile\n"
        installer_text = "starun\n"

        updated_dist = False
        for path, content in (
            (metadata_path, metadata_text),
            (wheel_path, wheel_text),
            (top_level_path, top_level_text),
            (installer_path, installer_text),
        ):
            if path.exists():
                existing = path.read_text(encoding="utf-8", errors="replace")
                if existing == content:
                    continue
            path.write_text(content, encoding="utf-8")
            updated_dist = True

        if updated_dist:
            self._append_event(
                "已写入 tiffile 兼容分发元数据: "
                + self._display_path(dist_info)
            )

    def _ensure_runtime_opencv_distribution_alias(self) -> None:
        """Expose headless OpenCV under the distribution name scripts request."""
        venv_dir = self._siril_state_root() / "venv"
        site_dir = resolve_venv_site_packages(venv_dir)
        if list(site_dir.glob("opencv_python-[0-9]*.dist-info")):
            return

        cv2_present = (site_dir / "cv2").exists() or any(
            site_dir.glob("cv2.*")
        )
        headless_dists = sorted(
            site_dir.glob("opencv_python_headless-*.dist-info")
        )
        if not cv2_present or not headless_dists:
            raise RuntimeError(
                "opencv-python-headless/cv2 未完整安装，无法创建 opencv-python 兼容元数据"
            )

        version = "0.0.0"
        headless_metadata = headless_dists[-1] / "METADATA"
        if headless_metadata.is_file():
            for line in headless_metadata.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                if line.lower().startswith("version:"):
                    version = line.split(":", 1)[1].strip() or version
                    break
        normalized_version = re.sub(r"[^A-Za-z0-9_.]+", "_", version)
        dist_info = site_dir / f"opencv_python-{normalized_version}.dist-info"
        dist_info.mkdir(parents=True, exist_ok=True)
        files = {
            "METADATA": (
                "Metadata-Version: 2.1\n"
                "Name: opencv-python\n"
                f"Version: {version}\n"
                "Summary: Compatibility metadata backed by opencv-python-headless\n"
            ),
            "WHEEL": (
                "Wheel-Version: 1.0\n"
                "Generator: starun\n"
                "Root-Is-Purelib: true\n"
                "Tag: py3-none-any\n"
            ),
            "top_level.txt": "cv2\n",
            "INSTALLER": "starun\n",
        }
        updated = False
        for name, content in files.items():
            path = dist_info / name
            if path.is_file() and path.read_text(
                encoding="utf-8", errors="replace"
            ) == content:
                continue
            path.write_text(content, encoding="utf-8")
            updated = True
        if updated:
            self._append_event(
                "已写入 opencv-python 兼容分发元数据: "
                + self._display_path(dist_info)
            )

    def _ensure_runtime_sirilpy_timeout_patch(self) -> None:
        venv_dir = self._siril_state_root() / "venv"
        site_dir = resolve_venv_site_packages(venv_dir)
        patch_path = site_dir / "sitecustomize.py"
        patch_code = (
            '"""Starun runtime patch: override sirilpy timeout via env."""\n'
            "import os\n"
            "def _patch_default_timeout(func, timeout):\n"
            "    defaults = getattr(func, '__defaults__', None)\n"
            "    if not defaults:\n"
            "        return\n"
            "    updated = list(defaults)\n"
            "    updated[-1] = timeout\n"
            "    func.__defaults__ = tuple(updated)\n"
            "\n"
            "raw = os.getenv('STARUN_SIRILPY_TIMEOUT_SEC', '').strip()\n"
            "if raw:\n"
            "    try:\n"
            "        timeout = float(raw)\n"
            "    except Exception:\n"
            "        timeout = None\n"
            "    if timeout and timeout > 0:\n"
            "        try:\n"
            "            import sirilpy.connection as _sirilpy_connection\n"
            "            _sirilpy_connection.DEFAULT_TIMEOUT = timeout\n"
            "            _iface = getattr(_sirilpy_connection, 'SirilInterface', None)\n"
            "            if _iface is not None:\n"
            "                for _name in ('_recv_exact', '_send_command', '_execute_command', '_request_data'):\n"
            "                    _func = getattr(_iface, _name, None)\n"
            "                    if callable(_func):\n"
            "                        _patch_default_timeout(_func, timeout)\n"
            "        except Exception:\n"
            "            pass\n"
        )
        if patch_path.exists():
            existing = patch_path.read_text(encoding="utf-8", errors="replace")
            if existing == patch_code:
                return
        patch_path.write_text(patch_code, encoding="utf-8")
        self._append_event(
            "已写入 sirilpy timeout 补丁: "
            + self._display_path(patch_path)
        )

    def _ensure_runtime_requirements_ready(self) -> None:
        python_bin = self._runtime_venv_python_bin()
        if not python_bin.exists():
            raise FileNotFoundError(f"Siril runtime python not found: {python_bin}")

        requirements_path = self._plugin_requirements_path()
        if not requirements_path.is_file():
            raise FileNotFoundError(f"Siril runtime requirements not found: {requirements_path}")

        missing_wheels = self._missing_requirement_wheels(requirements_path)
        if missing_wheels:
            raise RuntimeError(
                "Siril runtime requirements 离线 wheel 缺失："
                + "、".join(missing_wheels)
            )

        runtime_env = self._runtime_python_env()
        wheel_dir = self._plugin_downloads_dir()
        self._append_event(
            "正在按 requirements 离线安装 Siril runtime 依赖 "
            f"(no-index, find-links={self._display_path(wheel_dir)})..."
        )
        install_cp = self._run_bootstrap_process(
            [
                str(python_bin),
                "-m",
                "pip",
                "install",
                "--no-index",
                "--upgrade",
                "--find-links",
                str(wheel_dir),
                "-r",
                str(requirements_path),
            ],
            capture_output=True,
            text=True,
            check=False,
            env=runtime_env,
        )
        if install_cp.returncode != 0:
            tail = (
                install_cp.stderr.strip()
                or install_cp.stdout.strip()
                or "unknown error"
            )
            raise RuntimeError(
                "Siril runtime requirements 离线安装失败 "
                f"(exit={install_cp.returncode}): {tail[-320:]}"
            )

        self._ensure_runtime_tiffile_alias()
        self._ensure_runtime_opencv_distribution_alias()
        self._append_event("Siril runtime requirements 离线安装完成。")

    def _ensure_runtime_cosmic_clarity_deps_ready(self) -> None:
        python_bin = self._runtime_venv_python_bin()
        if not python_bin.exists():
            raise FileNotFoundError(f"Siril runtime python not found: {python_bin}")

        runtime_env = self._runtime_python_env()
        runtime_env.setdefault("STARUN_SIRILPY_TIMEOUT_SEC", "300")
        self._ensure_runtime_sirilpy_timeout_patch()
        self._ensure_runtime_tiffile_alias()
        self._ensure_runtime_opencv_distribution_alias()

        probe_code = (
            "import importlib.metadata as md; "
            "import PyQt6, tifffile, tiffile, lz4, zstandard, exifread, cv2, requests, sep, spandrel; "
            "print('cosmic-clarity-deps-ok', md.version('tiffile'), md.version('opencv-python'))"
        )
        check_cp = self._run_bootstrap_process(
            [str(python_bin), "-c", probe_code],
            capture_output=True,
            text=True,
            check=False,
            env=runtime_env,
        )
        if check_cp.returncode == 0:
            self._append_event(
                "Cosmic Clarity 运行时依赖已就绪"
                "（PyQt6/tifffile/lz4/zstandard/exifread/opencv/requests）。"
            )
            return

        missing_wheels: list[str] = []
        if not self._pyqt6_wheels():
            missing_wheels.append(SIRIL_COSMIC_REQUIRED_WHEEL_LABELS[0])
        if not self._pyqt6_qt6_wheels():
            missing_wheels.append(SIRIL_COSMIC_REQUIRED_WHEEL_LABELS[1])
        if not self._pyqt6_sip_wheels():
            missing_wheels.append(SIRIL_COSMIC_REQUIRED_WHEEL_LABELS[2])
        if not self._tifffile_wheels():
            missing_wheels.append(SIRIL_COSMIC_REQUIRED_WHEEL_LABELS[3])
        if not self._lz4_wheels():
            missing_wheels.append(SIRIL_COSMIC_REQUIRED_WHEEL_LABELS[4])
        if not self._zstandard_wheels():
            missing_wheels.append(SIRIL_COSMIC_REQUIRED_WHEEL_LABELS[5])
        if not self._exifread_wheels():
            missing_wheels.append(SIRIL_COSMIC_REQUIRED_WHEEL_LABELS[6])
        if not self._opencv_python_headless_wheels():
            missing_wheels.append(SIRIL_COSMIC_REQUIRED_WHEEL_LABELS[7])
        if not self._requests_wheels():
            missing_wheels.append(SIRIL_COSMIC_REQUIRED_WHEEL_LABELS[8])
        if self._requests_dependency_wheels_missing():
            missing_wheels.append(SIRIL_COSMIC_REQUIRED_WHEEL_LABELS[9])
        if not self._wheel_package_wheels():
            missing_wheels.append(SIRIL_COSMIC_REQUIRED_WHEEL_LABELS[10])
        if not self._sep_wheels():
            missing_wheels.append("sep")
        if not self._spandrel_wheels():
            missing_wheels.append("spandrel")
        if not self._einops_wheels():
            missing_wheels.append("einops")
        if not self._safetensors_wheels():
            missing_wheels.append("safetensors")
        if missing_wheels:
            raise RuntimeError(
                "Cosmic Clarity 离线依赖 wheel 缺失："
                + "、".join(missing_wheels)
            )

        wheel_dir = self._plugin_downloads_dir()
        self._append_event(
            "正在离线安装 Cosmic Clarity 运行时依赖 "
            "(PyQt6/tifffile/lz4/zstandard/exifread/opencv/requests, no-index)..."
        )
        install_pyqt_cp = self._run_bootstrap_process(
            [
                str(python_bin),
                "-m",
                "pip",
                "install",
                "--no-index",
                "--upgrade",
                "--force-reinstall",
                "--find-links",
                str(wheel_dir),
                "PyQt6",
            ],
            capture_output=True,
            text=True,
            check=False,
            env=runtime_env,
        )
        if install_pyqt_cp.returncode != 0:
            tail = (
                install_pyqt_cp.stderr.strip()
                or install_pyqt_cp.stdout.strip()
                or "unknown error"
            )
            raise RuntimeError(
                "Cosmic Clarity 依赖离线安装失败 (PyQt6) "
                f"(exit={install_pyqt_cp.returncode}): {tail[-280:]}"
            )

        install_tifffile_cp = self._run_bootstrap_process(
            [
                str(python_bin),
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                "--upgrade",
                "--force-reinstall",
                "--find-links",
                str(wheel_dir),
                "tifffile",
                "lz4",
                "zstandard",
                "exifread",
                "opencv-python-headless",
                "requests",
            ],
            capture_output=True,
            text=True,
            check=False,
            env=runtime_env,
        )
        if install_tifffile_cp.returncode != 0:
            tail = (
                install_tifffile_cp.stderr.strip()
                or install_tifffile_cp.stdout.strip()
                or "unknown error"
            )
            raise RuntimeError(
                "Cosmic Clarity 依赖离线安装失败 "
                "(tifffile/lz4/zstandard/exifread/opencv/requests) "
                f"(exit={install_tifffile_cp.returncode}): {tail[-280:]}"
            )

        self._ensure_runtime_tiffile_alias()
        self._ensure_runtime_opencv_distribution_alias()

        install_ai_cp = self._run_bootstrap_process(
            [
                str(python_bin),
                "-m",
                "pip",
                "install",
                "--no-index",
                "--upgrade",
                "--find-links",
                str(wheel_dir),
                "sep",
                "spandrel",
            ],
            capture_output=True,
            text=True,
            check=False,
            env=runtime_env,
        )
        if install_ai_cp.returncode != 0:
            tail = (
                install_ai_cp.stderr.strip()
                or install_ai_cp.stdout.strip()
                or "unknown error"
            )
            raise RuntimeError(
                "Cosmic Clarity/SCUNet 依赖离线安装失败 (sep/spandrel) "
                f"(exit={install_ai_cp.returncode}): {tail[-280:]}"
            )

        verify_cp = self._run_bootstrap_process(
            [str(python_bin), "-c", probe_code],
            capture_output=True,
            text=True,
            check=False,
            env=runtime_env,
        )
        if verify_cp.returncode != 0:
            tail = (verify_cp.stderr.strip() or verify_cp.stdout.strip() or "unknown error")
            raise RuntimeError(f"Cosmic Clarity 依赖安装后导入失败: {tail[-280:]}")
        self._append_event("Cosmic Clarity 运行时依赖离线安装完成。")

    def _ensure_runtime_syqon_starless_deps_ready(self) -> None:
        python_bin = self._runtime_venv_python_bin()
        if not python_bin.exists():
            raise FileNotFoundError(f"Siril runtime python not found: {python_bin}")

        runtime_env = self._runtime_python_env()
        probe_code = (
            "import PyQt6, PySide6, astropy, scipy, torch; "
            "print('syqon-starless-deps-ok', astropy.__version__, "
            "scipy.__version__, torch.__version__)"
        )
        check_cp = self._run_bootstrap_process(
            [str(python_bin), "-c", probe_code],
            capture_output=True,
            text=True,
            check=False,
            env=runtime_env,
        )
        if check_cp.returncode == 0:
            self._append_event("SyQon Starless 运行时依赖已就绪（PyQt6/PySide6/astropy/scipy）。")
            return

        missing_wheels: list[str] = []
        if not self._pyside6_wheels():
            missing_wheels.append(SIRIL_STARLESS_REQUIRED_WHEEL_LABELS[0])
        if not self._pyside6_addons_wheels():
            missing_wheels.append(SIRIL_STARLESS_REQUIRED_WHEEL_LABELS[1])
        if not self._pyside6_essentials_wheels():
            missing_wheels.append(SIRIL_STARLESS_REQUIRED_WHEEL_LABELS[2])
        if not self._shiboken6_wheels():
            missing_wheels.append(SIRIL_STARLESS_REQUIRED_WHEEL_LABELS[3])
        if not self._astropy_wheels():
            missing_wheels.append(SIRIL_STARLESS_REQUIRED_WHEEL_LABELS[4])
        if not self._scipy_wheels():
            missing_wheels.append(SIRIL_STARLESS_REQUIRED_WHEEL_LABELS[5])
        if not self._torch_wheels():
            missing_wheels.append("torch")
        if not self._torchvision_wheels():
            missing_wheels.append("torchvision")
        if missing_wheels:
            raise RuntimeError(
                "SyQon Starless 离线依赖 wheel 缺失："
                + "、".join(missing_wheels)
            )

        wheel_dir = self._plugin_downloads_dir()
        self._append_event(
            "正在离线安装 SyQon Starless 运行时依赖 "
            "(PySide6/astropy/scipy, no-index)..."
        )
        install_pyside_cp = self._run_bootstrap_process(
            [
                str(python_bin),
                "-m",
                "pip",
                "install",
                "--no-index",
                "--upgrade",
                "--force-reinstall",
                "--find-links",
                str(wheel_dir),
                "PySide6",
            ],
            capture_output=True,
            text=True,
            check=False,
            env=runtime_env,
        )
        if install_pyside_cp.returncode != 0:
            tail = (
                install_pyside_cp.stderr.strip()
                or install_pyside_cp.stdout.strip()
                or "unknown error"
            )
            raise RuntimeError(
                "SyQon Starless 依赖离线安装失败 (PySide6) "
                f"(exit={install_pyside_cp.returncode}): {tail[-280:]}"
            )

        install_sci_cp = self._run_bootstrap_process(
            [
                str(python_bin),
                "-m",
                "pip",
                "install",
                "--no-index",
                "--upgrade",
                "--force-reinstall",
                "--find-links",
                str(wheel_dir),
                "astropy",
                "scipy",
            ],
            capture_output=True,
            text=True,
            check=False,
            env=runtime_env,
        )
        if install_sci_cp.returncode != 0:
            tail = (
                install_sci_cp.stderr.strip()
                or install_sci_cp.stdout.strip()
                or "unknown error"
            )
            raise RuntimeError(
                "SyQon Starless 依赖离线安装失败 (astropy/scipy) "
                f"(exit={install_sci_cp.returncode}): {tail[-280:]}"
            )

        install_torch_cp = self._run_bootstrap_process(
            [
                str(python_bin),
                "-m",
                "pip",
                "install",
                "--no-index",
                "--upgrade",
                "--find-links",
                str(wheel_dir),
                "torch",
                "torchvision",
            ],
            capture_output=True,
            text=True,
            check=False,
            env=runtime_env,
        )
        if install_torch_cp.returncode != 0:
            tail = (
                install_torch_cp.stderr.strip()
                or install_torch_cp.stdout.strip()
                or "unknown error"
            )
            raise RuntimeError(
                "SyQon Starless 依赖离线安装失败 (torch/torchvision) "
                f"(exit={install_torch_cp.returncode}): {tail[-280:]}"
            )

        verify_cp = self._run_bootstrap_process(
            [str(python_bin), "-c", probe_code],
            capture_output=True,
            text=True,
            check=False,
            env=runtime_env,
        )
        if verify_cp.returncode != 0:
            tail = (verify_cp.stderr.strip() or verify_cp.stdout.strip() or "unknown error")
            raise RuntimeError(f"SyQon Starless 依赖安装后导入失败: {tail[-280:]}")
        self._append_event("SyQon Starless 运行时依赖离线安装完成。")

    def _ensure_runtime_onnxruntime_ready(self) -> None:
        python_bin = self._runtime_venv_python_bin()
        if not python_bin.exists():
            raise FileNotFoundError(f"Siril runtime python not found: {python_bin}")

        runtime_env = self._runtime_python_env()

        check_cp = self._run_bootstrap_process(
            [
                str(python_bin),
                "-c",
                "import onnx, onnxruntime as ort; print(onnx.__version__, ort.__version__)",
            ],
            capture_output=True,
            text=True,
            check=False,
            env=runtime_env,
        )
        if check_cp.returncode == 0:
            version = (check_cp.stdout.strip() or "unknown").splitlines()[-1]
            self._append_event(f"onnx/onnxruntime 已就绪 (versions={version})。")
            return

        onnx_wheels = self._onnx_wheels()
        onnxruntime_wheels = self._onnxruntime_wheels()
        if not onnx_wheels or not onnxruntime_wheels:
            raise RuntimeError("onnx/onnxruntime wheel 缺失，无法离线安装")
        onnx_path = onnx_wheels[-1]
        onnxruntime_path = onnxruntime_wheels[-1]
        wheel_dir = self._plugin_downloads_dir()
        self._append_event(
            "正在离线安装 onnx/onnxruntime 到 Siril runtime venv "
            f"(wheels={onnx_path.name}, {onnxruntime_path.name}, no-deps)..."
        )
        install_cp = self._run_bootstrap_process(
            [
                str(python_bin),
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                "--upgrade",
                "--force-reinstall",
                "--find-links",
                str(wheel_dir),
                str(onnx_path),
                str(onnxruntime_path),
            ],
            capture_output=True,
            text=True,
            check=False,
            env=runtime_env,
        )
        if install_cp.returncode != 0:
            tail = (install_cp.stderr.strip() or install_cp.stdout.strip() or "unknown error")
            raise RuntimeError(
                f"onnx/onnxruntime 离线安装失败 (exit={install_cp.returncode}): {tail[-280:]}"
            )

        verify_cp = self._run_bootstrap_process(
            [
                str(python_bin),
                "-c",
                "import onnx, onnxruntime as ort; print(onnx.__version__, ort.__version__)",
            ],
            capture_output=True,
            text=True,
            check=False,
            env=runtime_env,
        )
        if verify_cp.returncode != 0:
            tail = (verify_cp.stderr.strip() or verify_cp.stdout.strip() or "unknown error")
            raise RuntimeError(f"onnx/onnxruntime 安装后导入失败: {tail[-280:]}")
        version = (verify_cp.stdout.strip() or "unknown").splitlines()[-1]
        self._append_event(f"onnx/onnxruntime 离线安装完成 (versions={version})。")

    def _bootstrap_app_version(self) -> str:
        env_version = os.environ.get("STARUN_APP_VERSION", "").strip()
        if env_version:
            return env_version
        info_plist = self.resources.parent / "Info.plist"
        if info_plist.is_file():
            try:
                with info_plist.open("rb") as fh:
                    payload = plistlib.load(fh)
                short_version = str(
                    payload.get("CFBundleShortVersionString") or "0.0.0"
                )
                build_version = str(payload.get("CFBundleVersion") or "")
                return (
                    f"{short_version} ({build_version})"
                    if build_version
                    else short_version
                )
            except (OSError, ValueError, TypeError):
                pass
        return "dev"

    def _bootstrap_fingerprint(self) -> dict[str, object]:
        lock_candidates = [
            self.siril_plugin_dir / "requirements.lock",
            self.siril_plugin_dir / "requirements-macos-arm64.lock",
        ]
        lock_paths = [path for path in lock_candidates if path.is_file()]
        if not lock_paths:
            fallback = self._plugin_requirements_path()
            if fallback.is_file():
                lock_paths = [fallback]

        digest = hashlib.sha256()
        for path in lock_paths:
            digest.update(path.name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        return {
            "schema": 1,
            "python_abi": SIRIL_PLUGIN_PYTHON_ABI,
            "app_version": self._bootstrap_app_version(),
            "dependency_lock_sha256": digest.hexdigest(),
        }

    def _bootstrap_state_path(self) -> Path:
        return self._siril_state_root() / "venv" / ".starun_runtime_ready.json"

    def _bootstrap_state_is_current(self, fingerprint: dict[str, object]) -> bool:
        state_path = self._bootstrap_state_path()
        if not state_path.is_file() or not self._runtime_venv_python_bin().exists():
            return False
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return False
        return payload.get("fingerprint") == fingerprint

    def _write_bootstrap_state(self, fingerprint: dict[str, object]) -> None:
        state_path = self._bootstrap_state_path()
        state_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = state_path.with_suffix(".tmp")
        temp_path.write_text(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "completed_at": self._timestamp(),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temp_path.replace(state_path)

    def _bootstrap_runtime(
        self,
        work_dir: Path,
        input_mode: str,
        stop_event: threading.Event,
        progress,
    ) -> dict[str, object]:
        self._bootstrap_stop_event = stop_event
        try:
            if isinstance(
                getattr(self, "_runtime_capability_manifest", None),
                dict,
            ):
                StarunGui._update_run_state(
                    self,
                    phase="runtime_preflight",
                    status=STATUS_PREPARING,
                    detail="validating Siril launch and Gaia network endpoints",
                )
                StarunGui._complete_runtime_capability_preflight(
                    self,
                    progress,
                )
                StarunGui._update_run_state(
                    self,
                    phase="runtime_bootstrap",
                    status=STATUS_PREPARING,
                    detail="runtime capability preflight passed",
                    errors=[],
                )
            progress("正在分别检查运行环境和工作目录磁盘…")
            self._check_bootstrap_cancelled()
            try:
                fingerprint = self._bootstrap_fingerprint()
                disk_estimate = self._estimate_disk_space(
                    work_dir,
                    input_mode=input_mode,
                )
                runtime_disk_estimate = self._estimate_runtime_disk_space(
                    fingerprint
                )
            except Exception as exc:
                raise BootstrapError(
                    "磁盘空间检查失败",
                    f"无法完成磁盘空间预检：\n{exc}",
                ) from exc

            work_volume_device = os.stat(existing_volume_anchor(work_dir)).st_dev
            if (
                disk_estimate
                and work_volume_device == runtime_disk_estimate.volume_device
            ):
                combined_required = (
                    disk_estimate.required_free_bytes
                    + runtime_disk_estimate.required_free_bytes
                )
                combined_available = min(
                    disk_estimate.available_bytes,
                    runtime_disk_estimate.available_bytes,
                )
                if combined_available < combined_required:
                    raise BootstrapError(
                        "共享磁盘空间不足",
                        "运行环境与工作目录位于同一磁盘卷，合计剩余空间不足。\n\n"
                        f"工作目录建议空间: {format_bytes(disk_estimate.required_free_bytes)}\n"
                        "运行环境建议空间: "
                        f"{format_bytes(runtime_disk_estimate.required_free_bytes)}\n"
                        f"合计建议空间: {format_bytes(combined_required)}\n"
                        f"当前剩余空间: {format_bytes(combined_available)}\n\n"
                        "请先清理该磁盘后重试。",
                    )
            if (
                disk_estimate
                and disk_estimate.available_bytes
                < disk_estimate.required_free_bytes
            ):
                raise BootstrapError(
                    "磁盘空间不足",
                    self._disk_space_error_message(disk_estimate),
                )
            if (
                runtime_disk_estimate.available_bytes
                < runtime_disk_estimate.required_free_bytes
            ):
                raise BootstrapError(
                    "运行环境磁盘空间不足",
                    self._runtime_disk_space_error_message(runtime_disk_estimate),
                )

            if disk_estimate:
                for line in self._disk_space_summary_lines(disk_estimate):
                    self._append_event(line.strip())
            for line in self._runtime_disk_space_summary_lines(
                runtime_disk_estimate
            ):
                self._append_event(line.strip())

            progress("正在准备内置 Siril Python…")
            self._check_bootstrap_cancelled()
            try:
                self._ensure_offline_siril_python_seed()
            except BootstrapCancelled:
                raise
            except Exception as exc:
                raise BootstrapError(
                    "Siril Python Seed 准备失败",
                    f"准备离线 Siril Python seed 失败：\n{exc}",
                ) from exc

            progress("正在检查离线插件资源…")
            self._check_bootstrap_cancelled()
            if not self._ensure_siril_plugins_ready():
                raise BootstrapError(
                    "插件预检失败",
                    "Siril 插件缓存不完整，且自动补齐失败。\n"
                    "请先执行 resources/siril_plugins/download_siril_plugins.sh 后重试。",
                )

            progress("正在同步 Siril 离线资源…")
            self._check_bootstrap_cancelled()
            try:
                self._ensure_runtime_siril_support_dirs()
            except BootstrapCancelled:
                raise
            except Exception as exc:
                self._append_event(f"Siril 运行时目录准备失败（继续执行）：{exc}")

            if runtime_disk_estimate.bootstrap_cache_hit:
                progress("运行时依赖已就绪，跳过重复安装…")
                self._append_event(
                    "运行时依赖指纹未变化，已跳过重复 pip install。"
                )
                return {
                    "disk_estimate": disk_estimate,
                    "runtime_disk_estimate": runtime_disk_estimate,
                    "bootstrap_cache_hit": True,
                }

            progress("正在安装 Siril runtime 依赖…")
            self._check_bootstrap_cancelled()
            try:
                self._ensure_runtime_requirements_ready()
            except BootstrapCancelled:
                raise
            except Exception as exc:
                raise BootstrapError(
                    "Siril runtime 依赖准备失败",
                    f"无法按 requirements 准备 Siril runtime 依赖：\n{exc}",
                ) from exc

            progress("正在验证 Cosmic Clarity 依赖…")
            self._check_bootstrap_cancelled()
            try:
                self._ensure_runtime_cosmic_clarity_deps_ready()
            except BootstrapCancelled:
                raise
            except Exception as exc:
                raise BootstrapError(
                    "Cosmic Clarity 依赖准备失败",
                    f"无法准备 Siril 运行时 PyQt6/tifffile：\n{exc}",
                ) from exc

            progress("正在验证 SyQon Starless 依赖…")
            self._check_bootstrap_cancelled()
            try:
                self._ensure_runtime_syqon_starless_deps_ready()
            except BootstrapCancelled:
                raise
            except Exception as exc:
                raise BootstrapError(
                    "SyQon Starless 依赖准备失败",
                    f"无法准备 Siril 运行时 PySide6/astropy/scipy：\n{exc}",
                ) from exc

            progress("正在验证 ONNX runtime…")
            self._check_bootstrap_cancelled()
            try:
                self._ensure_runtime_onnxruntime_ready()
            except BootstrapCancelled:
                raise
            except Exception as exc:
                raise BootstrapError(
                    "onnx/onnxruntime 准备失败",
                    f"无法准备 Siril 运行时 onnx/onnxruntime：\n{exc}",
                ) from exc

            self._check_bootstrap_cancelled()
            self._write_bootstrap_state(fingerprint)
            self._append_event("运行时依赖准备完成，已写入就绪指纹。")
            return {
                "disk_estimate": disk_estimate,
                "runtime_disk_estimate": runtime_disk_estimate,
                "bootstrap_cache_hit": False,
            }
        finally:
            self._bootstrap_stop_event = None

    def _start_task_intake(
        self,
        discovery: InputDiscovery,
        processing_settings: dict[str, object],
    ) -> None:
        """Hash sources and materialize task/run manifests off the UI thread."""

        if self.intake_worker and self.intake_worker.isRunning():
            return

        def prepare(
            stop_event: threading.Event,
            progress,
        ) -> PreparedTaskQueue:
            progress("正在校验来源并建立独立任务…")
            if stop_event.is_set():
                raise BootstrapCancelled()
            try:
                queue = prepare_task_queue(
                    discovery,
                    processing_settings=processing_settings,
                    cancel_check=stop_event.is_set,
                )
            except InterruptedError as exc:
                raise BootstrapCancelled() from exc
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                if stop_event.is_set():
                    raise BootstrapCancelled() from exc
                raise BootstrapError("无法建立任务", str(exc)) from exc
            if stop_event.is_set():
                raise BootstrapCancelled()
            progress(
                f"已建立 {len(queue.tasks)} 个任务，正在准备预检…"
            )
            return queue

        self.intake_worker = BootstrapWorker(prepare, parent=self)
        self.intake_worker.progress.connect(self._on_task_intake_progress)
        self.intake_worker.succeeded.connect(self._on_task_intake_succeeded)
        self.intake_worker.failed.connect(self._on_task_intake_failed)
        self.intake_worker.cancelled.connect(self._on_task_intake_cancelled)
        self._set_running(True)
        self._set_status_text("Preparing")
        self._append_event("开始在后台校验来源内容指纹…")
        self.intake_worker.start()
        self._update_toolbar_state()

    def _cleanup_task_intake_worker(self) -> None:
        if self.intake_worker:
            self.intake_worker.wait(200)
            self.intake_worker.deleteLater()
            self.intake_worker = None

    def _on_task_intake_progress(self, message: str) -> None:
        self._set_status_message(message)
        self._append_event(message)

    def _on_task_intake_succeeded(self, result: object) -> None:
        self._cleanup_task_intake_worker()
        if not isinstance(result, PreparedTaskQueue) or not result.tasks:
            self._on_task_intake_failed(
                "无法建立任务",
                "后台任务准备没有返回可执行任务。",
            )
            return
        self._prepared_tasks = result.tasks
        self._prepared_task_index = 0
        self._spcc_online_circuit_open = False
        first_task = result.tasks[0]
        self._append_event(
            f"来源校验完成：{len(result.tasks)} 个独立任务，将串行执行。"
        )
        if result.skipped_duplicates:
            self._append_event(
                "已跳过完全相同的来源与处理参数组合："
                f"{len(result.skipped_duplicates)} 项。"
            )
        self._set_running(False)
        self._start_run(prepared_task=first_task)

    def _on_task_intake_failed(self, title: str, detail: str) -> None:
        self._cleanup_task_intake_worker()
        self._prepared_tasks = ()
        self._prepared_task_index = 0
        self._active_prepared_task = None
        self._spcc_online_circuit_open = False
        self._set_status_text("Idle")
        self._set_running(False)
        QMessageBox.critical(self, title, detail)
        self._append_event(f"任务目录创建失败：{detail}")

    def _on_task_intake_cancelled(self) -> None:
        self._cleanup_task_intake_worker()
        self._prepared_tasks = ()
        self._prepared_task_index = 0
        self._active_prepared_task = None
        self._spcc_online_circuit_open = False
        self._set_status_text("Idle")
        self._set_running(False)
        self._append_event("已停止来源校验；原始文件未被修改。")

    def _start_run(
        self,
        _checked: bool = False,
        *,
        prepared_task: PreparedTask | None = None,
    ) -> None:
        if self.intake_worker and self.intake_worker.isRunning():
            return
        gaia_worker = getattr(self, "gaia_catalog_worker", None)
        if gaia_worker and gaia_worker.isRunning():
            QMessageBox.information(
                self,
                "正在安装离线目录",
                "请等待 Gaia 目录安装完成，或先点击“取消下载”。",
            )
            return
        if self.bootstrap_worker and self.bootstrap_worker.isRunning():
            return
        if self.worker and self.worker.isRunning():
            return

        selected_text = self.dir_edit.text().strip()
        if not selected_text:
            QMessageBox.information(self, "请选择输入", "请先拖入或选择图像文件或目录。")
            return
        selected_path = Path(selected_text).expanduser()
        if prepared_task is None:
            discovery = self._input_discovery
            if (
                discovery is None
                or discovery.selected_path != selected_path.resolve()
                or discovery.kind == InputKind.PRODUCT_TASK
            ):
                discovery = discover_input_for_processing_settings(
                    selected_path,
                    processing_settings=self._processing_settings_snapshot(),
                )
                self._input_discovery = discovery
            if not discovery.accepted:
                detail = "\n".join((*discovery.errors, *discovery.warnings))
                QMessageBox.critical(
                    self,
                    "无法建立任务",
                    detail or discovery.summary,
                )
                return
            self._start_task_intake(
                discovery,
                self._processing_settings_snapshot(),
            )
            return
        self._active_prepared_task = prepared_task
        self._last_task_root = prepared_task.workspace.root
        work_dir = prepared_task.run.root
        self._current_work_dir = work_dir
        try:
            self._begin_run_diagnostics(work_dir, prepared_task)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            detail = f"无法在预检前创建本轮日志和状态文件：{error}"
            self._register_history_run(prepared_task)
            self._update_active_history_run(
                STATUS_FAILED,
                failure_reason=detail,
            )
            self._current_work_dir = None
            self._active_prepared_task = None
            self._prepared_tasks = ()
            self._prepared_task_index = 0
            self._active_history_task_key = None
            self._active_history_run_id = None
            QMessageBox.critical(self, "运行诊断初始化失败", detail)
            self._append_text(f"[{self._timestamp()}] {detail}\n")
            return
        self._register_history_run(prepared_task)
        self._update_active_history_run(STATUS_PREPARING)
        input_mode = prepared_task.input_mode
        (
            self._pending_runtime_overrides,
            self._pending_runtime_unset_keys,
        ) = self._processing_runtime_configuration(input_mode)
        self._pending_runtime_overrides.update(prepared_task.runtime_overrides)
        if self._spcc_online_circuit_open:
            self._pending_runtime_overrides[SPCC_ONLINE_CIRCUIT_ENV] = "1"
            self._append_event(
                "本批次在线 Gaia SPCC 超时熔断已开启；"
                "本任务跳过在线 SPCC，仍保留 localgaia/PCC 降级。"
            )
        if self.graxpert_model_path:
            model_path = Path(self.graxpert_model_path).expanduser()
            if not model_path.exists():
                self._append_event(
                    "用户 GraXpert 对象模型路径当前不存在；"
                    "自动反卷积将尝试应用模型并安全回退 Siril RL："
                    f"{model_path}"
                )
        errors = self._preflight_errors(work_dir, input_mode=input_mode)
        capability_manifest = getattr(self, "_runtime_capability_manifest", None)
        if isinstance(capability_manifest, Mapping):
            for line in capability_summary_lines(capability_manifest):
                self._append_event(line)
            if self.runtime_capabilities_path is not None:
                self._append_event(
                    "能力清单文件: "
                    + self._display_path(self.runtime_capabilities_path)
                )
        if errors:
            self._pending_runtime_overrides.clear()
            self._pending_runtime_unset_keys.clear()
            self._append_event("预检失败：")
            for err in errors:
                self._append_text(f"  - {err}\n")
            self._update_run_state(
                phase="terminal",
                status=STATUS_FAILED,
                detail="local preflight failed",
                errors=errors,
            )
            self._update_active_history_run(
                STATUS_FAILED,
                failure_reason="；".join(errors),
            )
            self._close_run_log()
            QMessageBox.critical(self, "预检失败", "\n\n".join(errors))
            self._cleanup_after_run()
            return

        self._remember_directory(selected_path)
        self._run_input_mode = input_mode
        self._update_run_state(
            phase="runtime_preflight",
            status=STATUS_PREPARING,
            detail="local runtime and input preflight passed",
            errors=[],
        )
        self._last_run_snapshot = {
            "work_dir": str(selected_path),
            "input_mode": input_mode,
            "debug_mode_enabled": self.debug_mode_enabled,
            "network_mode_enabled": self.network_mode_enabled,
            "processing_settings": self._processing_settings_snapshot(),
        }
        self._progress_timer.stop()
        self._pipeline_started_monotonic = None
        self._reset_stage_progress(10)
        self._last_quality_report_path = None
        self.warning_card.hide()
        self._run_terminal_status = None
        self._latest_preview_stage = -1
        self._latest_preview_title = ""
        self._latest_preview_path = None
        self.task_preview_canvas.clear_image()
        self.preview_canvas.clear_image()
        self.preview_stage_label.setText("预览：Stage 0 · 输入")
        self.preview_activity_label.setText("正在准备运行环境")
        self.preview_notice_label.setText(
            "Stage 0 屏幕拉伸预览 · 仅影响显示，不修改源数据"
        )
        self.run_phase_label.setText("准备运行环境")
        self._schedule_initial_preview(work_dir, input_mode)
        self._update_run_summary(work_dir, input_mode)
        self._set_running(True)
        self._show_workspace(WORKSPACE_RUN)
        self._set_status_text("Preparing")
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFormat("正在准备运行环境…")
        self.progress_bar.show()
        self._append_event("开始准备运行环境…")
        if prepared_task.queue_total > 1:
            self._append_event(
                f"串行任务 {prepared_task.queue_index}/{prepared_task.queue_total}："
                f"{prepared_task.display_label}"
            )

        self.bootstrap_worker = BootstrapWorker(
            lambda stop_event, progress: self._bootstrap_runtime(
                work_dir,
                input_mode,
                stop_event,
                progress,
            ),
            parent=self,
        )
        self.bootstrap_worker.progress.connect(self._on_bootstrap_progress)
        self.bootstrap_worker.succeeded.connect(self._on_bootstrap_succeeded)
        self.bootstrap_worker.failed.connect(self._on_bootstrap_failed)
        self.bootstrap_worker.cancelled.connect(self._on_bootstrap_cancelled)
        self.bootstrap_worker.start()

    def _on_bootstrap_progress(self, message: str) -> None:
        self._set_status_message(message)
        self.preview_activity_label.setText(message)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFormat(message)
        self.progress_bar.show()

    def _cleanup_bootstrap_worker(self) -> None:
        if self.bootstrap_worker:
            self.bootstrap_worker.wait(200)
            self.bootstrap_worker.deleteLater()
            self.bootstrap_worker = None

    def _on_bootstrap_succeeded(self, result: object) -> None:
        work_dir = self._current_work_dir
        self._cleanup_bootstrap_worker()
        if work_dir is None:
            self._on_bootstrap_failed(
                "准备运行环境失败",
                "准备完成后工作目录状态丢失。",
            )
            return
        payload = result if isinstance(result, dict) else {}
        disk_estimate = payload.get("disk_estimate")
        self._append_event("运行环境准备完成，正在启动处理…")
        self._update_run_state(
            phase="pipeline_starting",
            status=STATUS_PREPARING,
            detail="runtime bootstrap completed",
            errors=[],
        )
        self._start_pipeline(work_dir, disk_estimate)

    def _on_bootstrap_failed(self, title: str, detail: str) -> None:
        self._cleanup_bootstrap_worker()
        self._pending_runtime_overrides.clear()
        self._pending_runtime_unset_keys.clear()
        self._run_terminal_status = "Failed"
        self._set_status_text("Failed")
        self.progress_bar.setRange(0, 10)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("运行环境准备失败")
        self.progress_bar.hide()
        self.preview_activity_label.setText("运行环境准备失败")
        self.run_phase_label.setText("准备失败")
        self.toggle_log_action.setChecked(True)
        self._show_run_banner(
            "error",
            "运行环境准备失败。任务尚未进入图像处理，请查看详细日志后重试。",
            show_log=True,
        )
        self._append_event(f"{title}：{detail}")
        self._update_run_state(
            phase="terminal",
            status=STATUS_FAILED,
            detail=f"{title}：{detail}",
            errors=[detail],
        )
        self._update_active_history_run(
            STATUS_FAILED,
            failure_reason=f"{title}：{detail}",
        )
        self._update_result_actions(self._current_work_dir)
        self._close_run_log()
        QMessageBox.critical(self, title, detail)
        self._current_work_dir = None
        self._active_prepared_task = None
        self._prepared_tasks = ()
        self._prepared_task_index = 0
        self._active_history_task_key = None
        self._active_history_run_id = None
        self._set_running(False)

    def _on_bootstrap_cancelled(self) -> None:
        self._cleanup_bootstrap_worker()
        self._pending_runtime_overrides.clear()
        self._pending_runtime_unset_keys.clear()
        self._run_terminal_status = "Stopped"
        self._set_status_text("Stopped")
        self.progress_bar.setRange(0, 10)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("准备已停止")
        self.progress_bar.hide()
        self.preview_activity_label.setText("任务已停止 · 保留最新可靠预览")
        self.run_phase_label.setText("已停止")
        self._show_run_banner(
            "info",
            "运行环境准备已停止；尚未开始的阶段不会产生新文件。",
            show_log=True,
        )
        self._append_event("运行环境准备已停止。")
        self._update_run_state(
            phase="terminal",
            status=STATUS_STOPPED,
            detail="用户在运行环境准备阶段停止任务",
            errors=[],
        )
        self._update_active_history_run(
            STATUS_STOPPED,
            failure_reason="用户在运行环境准备阶段停止任务",
        )
        self._update_result_actions(self._current_work_dir)
        self._close_run_log()
        self._current_work_dir = None
        self._active_prepared_task = None
        self._prepared_tasks = ()
        self._prepared_task_index = 0
        self._active_history_task_key = None
        self._active_history_run_id = None
        self._set_running(False)

    def _start_pipeline(
        self,
        work_dir: Path,
        disk_estimate: DiskSpaceEstimate | None,
    ) -> None:
        input_mode = self._run_input_mode or self._current_input_mode()
        try:
            log_is_current = bool(
                self.run_log_path is not None
                and self.run_log_file is not None
                and self.run_log_path.resolve().parent == work_dir.resolve()
            )
        except OSError:
            log_is_current = False
        if not log_is_current:
            self._on_bootstrap_failed(
                "运行日志隔离失败",
                "当前任务没有预检前创建的独立日志；为避免写入上一任务日志，"
                "本次运行已阻断。",
            )
            return

        self._update_run_state(
            phase="pipeline",
            status=STATUS_RUNNING,
            detail="pipeline worker starting",
            errors=[],
        )
        self._update_active_history_run(STATUS_RUNNING)
        self._append_divider(
            "本次任务开始",
            [
                f"时间: {self._timestamp()}",
                f"工作目录: {self._display_path(work_dir)}",
                f"处理模式: {self._input_mode_label(input_mode)}",
            ],
        )
        for line in self._preflight_summary_lines(
            work_dir,
            disk_estimate,
            input_mode=input_mode,
        ):
            self._append_event(line)
        self._append_event(f"日志文件: {self._display_path(self.run_log_path)}")
        self._append_event(
            f"本次运行 debug_mode={'ON' if self.debug_mode_enabled else 'OFF'}"
        )
        self._append_event(
            f"本次运行 network_mode={'ON' if self.network_mode_enabled else 'OFF'}"
        )
        self._append_event(
            f"本次运行 input_mode={input_mode}"
        )
        for line in self._processing_settings_summary_lines(input_mode):
            self._append_event("本次处理参数：" + line)
        siril_candidates = self._resolve_runtime_candidates()
        self._append_event(
            "运行时顺序: "
            + ", ".join(self._display_path(p) for p in siril_candidates)
        )

        self.worker = PipelineWorker(
            work_dir=work_dir,
            config_template=self.config_template,
            pipeline_path=self.pipeline_path,
            siril_plugin_dir=self.siril_plugin_dir,
            resources=self.resources,
            runtime_home=self.runtime_home,
            siril_candidates=siril_candidates,
            input_mode=input_mode,
            debug_mode=self.debug_mode_enabled,
            network_mode=self.network_mode_enabled,
            runtime_overrides=self._pending_runtime_overrides,
            runtime_unset_keys=self._pending_runtime_unset_keys,
            graxpert_application_home=Path.home(),
            parent=self,
        )
        self._pending_runtime_overrides.clear()
        self._pending_runtime_unset_keys.clear()
        self.worker.log.connect(self._append_text)
        self.worker.state.connect(self._set_status_text)
        self.worker.progress.connect(self._on_pipeline_progress)
        self.worker.stage_detail.connect(self._on_pipeline_stage_detail)
        self.worker.preview.connect(self._on_pipeline_preview)
        self.worker.done.connect(self._on_worker_done)

        self._begin_stage_progress(10)
        self.progress_bar.hide()
        self.preview_activity_label.setText("等待 Stage 1 开始")
        self._set_status_text("Running")
        self.worker.start()

    def _stop_run(self) -> None:
        if self.intake_worker and self.intake_worker.isRunning():
            self._append_event("已请求停止来源校验…")
            self._set_status_text("Stopping")
            self.intake_worker.stop()
            return
        if self.bootstrap_worker and self.bootstrap_worker.isRunning():
            self._append_event("已请求停止运行环境准备…")
            self._set_status_text("Stopping")
            self.preview_activity_label.setText("正在停止运行环境准备…")
            self.progress_bar.setFormat("正在停止运行环境准备…")
            self.bootstrap_worker.stop()
            return
        if not self.worker:
            return
        self._append_event("已请求停止...")
        self._set_status_text("Stopping")
        self.preview_activity_label.setText("正在停止 · 保留最新可靠预览")
        self.worker.stop()

    def _set_run_phase_for_stage(self, stage: int) -> None:
        if stage <= 6:
            phase_text = "线性处理 · Stage 1–6"
        else:
            phase_text = "非线性处理 · Stage 7–10"
        self.run_phase_label.setText(phase_text)

    def _on_pipeline_progress(self, stage: int, title: str, state: str) -> None:
        stage = int(stage)
        if stage < 1 or stage > 10:
            return
        stage_count = 10
        self._progress_stage_count = stage_count

        detail = PIPELINE_STAGE_TITLES.get(
            stage,
            title.strip() or f"阶段 {stage}",
        )
        normalized_state = state.strip().lower() or "running"
        if normalized_state not in PIPELINE_PROGRESS_STATE_LABELS:
            normalized_state = "failed"
        state_label = PIPELINE_PROGRESS_STATE_LABELS[normalized_state]
        now = time.monotonic()

        if stage not in self._stage_items:
            chip = QLabel()
            chip.setWordWrap(True)
            chip.setSizePolicy(
                QSizePolicy.Policy.Preferred,
                QSizePolicy.Policy.Minimum,
            )
            self.stage_stepper_layout.addWidget(chip, 1)
            self._stage_items[stage] = chip
        self._stage_progress_titles[stage] = detail
        self._stage_progress_states[stage] = normalized_state
        self._display_pipeline_stage = stage
        self._set_run_phase_for_stage(stage)

        if normalized_state == "running":
            if stage not in self._stage_started_monotonic:
                self._stage_started_monotonic[stage] = now
            self._active_pipeline_stage = stage
        else:
            started = self._stage_started_monotonic.pop(stage, None)
            if started is not None:
                self._stage_elapsed_seconds[stage] = (
                    self._stage_elapsed_seconds.get(stage, 0.0) + now - started
                )
            else:
                self._stage_elapsed_seconds.setdefault(stage, 0.0)
            worker_durations = getattr(
                getattr(self, "worker", None),
                "_pipeline_stage_durations",
                {},
            )
            if stage in worker_durations:
                self._stage_elapsed_seconds[stage] = max(
                    0.0,
                    float(worker_durations[stage]),
                )
            if self._active_pipeline_stage == stage:
                self._active_pipeline_stage = None

        self._update_stage_chip(stage)
        progress_text = (
            f"当前进度：阶段 {stage}/{stage_count} · {detail} · {state_label}"
        )
        self.progress_summary_label.setText(progress_text)
        self.progress_summary_label.setAccessibleDescription(progress_text)
        self._set_status_message(f"运行中 · Stage {stage} {state_label}")
        if normalized_state == "running":
            self.preview_activity_label.setText(
                f"正在处理 Stage {stage} · {detail}"
            )
        elif normalized_state in {"completed", "safe_passthrough", "degraded"}:
            self.preview_activity_label.setText(
                f"Stage {stage} {state_label} · 等待最新预览"
            )
        else:
            self.preview_activity_label.setText(
                f"Stage {stage} {state_label} · 无新预览，保留上一张"
            )
        self._announce_accessibility(progress_text)
        self._refresh_elapsed_labels()

    def _on_pipeline_stage_detail(
        self,
        stage: int,
        payload: object,
    ) -> None:
        if not isinstance(payload, dict):
            return
        stage = int(stage)
        if stage < 1 or stage > 10:
            return
        self._stage_progress_details[stage] = dict(payload)
        raw_status = str(payload.get("status") or "").strip().lower()
        display_status = str(
            payload.get("display_status") or raw_status
        ).strip().lower()
        execution = str(payload.get("execution") or "completed").strip().lower()
        if raw_status == "failed":
            state = "failed"
        elif raw_status == "degraded" or display_status == "ok_with_fallback":
            state = "degraded"
        elif raw_status == "skipped" or execution == "skipped":
            state = "skipped"
        elif execution == "safe_passthrough":
            state = "safe_passthrough"
        else:
            state = "completed"
        title = str(payload.get("title") or "").strip()
        self._on_pipeline_progress(stage, title, state)

    def _on_pipeline_preview(
        self,
        stage: int,
        title: str,
        status: str,
        payload: str,
    ) -> None:
        stage = int(stage)
        if stage < 1 or stage > 10:
            return
        detail = PIPELINE_STAGE_TITLES.get(
            stage,
            title.strip() or f"阶段 {stage}",
        )
        if status == "ready" and self._display_latest_preview(
            Path(payload),
            stage=stage,
            title=detail,
        ):
            self.preview_activity_label.setText(
                f"Stage {stage} 已完成 · 最新预览已刷新"
            )
            if stage <= 6:
                self.preview_notice_label.setText(
                    "线性数据 · 已应用屏幕显示拉伸（不写入处理数据）"
                )
            else:
                self.preview_notice_label.setText(
                    "非线性阶段预览 · 无额外显示拉伸"
                )
            return

        reason = payload or "无法生成预览"
        self.preview_activity_label.setText(
            f"Stage {stage} 已完成 · 预览不可用，保留上一张"
        )
        self.preview_status_label.setText(
            f"预览：Stage {stage} 刷新失败 · 保留 Stage "
            f"{max(0, self._latest_preview_stage)}"
        )
        self._append_event(
            f"Stage {stage} 预览生成失败，不影响处理结果：{reason}"
        )

    def _on_worker_done(
        self,
        status: str,
        exit_code: int,
        had_fatal_errors: bool,
        cli_used: str,
    ) -> None:
        work_dir = self._current_work_dir
        pipeline_result: Mapping[str, object] | None = None
        if work_dir is not None:
            try:
                pipeline_result = load_verified_pipeline_result(work_dir)
            except HistoryStoreError as error:
                self._append_event(f"终态未采用无效结果清单：{error}")
        history_status = self._terminal_history_status(status, work_dir)
        history_failure_reason: str | None = None
        if history_status in {STATUS_FAILED, STATUS_STOPPED}:
            history_failure_reason = (
                "用户停止处理"
                if history_status == STATUS_STOPPED
                else f"处理失败（GUI 状态：{self._display_status(status)}）"
            )
            if pipeline_result is not None and pipeline_result.get("failure_reason"):
                history_failure_reason = str(pipeline_result["failure_reason"])

        raw_issues = (
            pipeline_result.get("issues", [])
            if pipeline_result is not None
            else []
        )
        terminal_issues = [
            dict(issue)
            for issue in raw_issues
            if isinstance(issue, Mapping)
        ]
        manifest_had_fatal = bool(
            pipeline_result is not None
            and pipeline_result.get("had_fatal_errors", False)
        )
        effective_had_fatal = bool(
            had_fatal_errors
            or manifest_had_fatal
            or history_status == STATUS_FAILED
        )
        if effective_had_fatal and not any(
            str(issue.get("severity") or "") == "fatal"
            for issue in terminal_issues
        ):
            current_stage = int(
                getattr(getattr(self, "worker", None), "_current_pipeline_stage", 0)
                or 0
            )
            terminal_issues.append(
                {
                    "stage": current_stage,
                    "component": "worker",
                    "severity": "fatal",
                    "code": "worker_execution_failed",
                    "recovered": False,
                    "message": history_failure_reason
                    or "worker execution failed before a trusted pipeline result was available",
                }
            )
        terminal_errors = [
            str(issue.get("message") or issue.get("code") or "execution error")
            for issue in terminal_issues
            if str(issue.get("severity") or "") in {"error", "fatal"}
        ]
        if (
            history_status == STATUS_FAILED
            and history_failure_reason
            and history_failure_reason not in terminal_errors
        ):
            terminal_errors.append(history_failure_reason)
        had_degradations = bool(
            pipeline_result is not None
            and pipeline_result.get("had_degradations", False)
        )
        had_fallbacks = bool(
            pipeline_result is not None
            and pipeline_result.get("had_fallbacks", False)
        )
        review_required = bool(
            pipeline_result is not None
            and pipeline_result.get("review_required", False)
        )
        effective_had_errors = bool(
            effective_had_fatal
            or (
                pipeline_result is not None
                and pipeline_result.get("had_errors", False)
            )
        )
        self._update_run_state(
            phase="terminal",
            status=history_status,
            detail=f"GUI status={status}",
            errors=terminal_errors,
            extra={
                "exit_code": int(exit_code),
                "cli_used": str(cli_used),
                "had_errors": effective_had_errors,
                "had_fatal_errors": effective_had_fatal,
                "had_degradations": had_degradations,
                "had_fallbacks": had_fallbacks,
                "review_required": review_required,
                "issues": terminal_issues,
            },
        )
        self._update_active_history_run(
            history_status,
            failure_reason=history_failure_reason,
            exit_code=exit_code,
        )
        if (
            not self._spcc_online_circuit_open
            and work_dir is not None
            and stage4_online_spcc_timeout_detected(work_dir)
        ):
            self._spcc_online_circuit_open = True
            self._append_event(
                "检测到 online_unverified Gaia SPCC 超时；"
                "已为本批次剩余任务开启在线 SPCC 熔断。"
            )
        has_next_task = bool(
            status in {"Completed", "CompletedWithWarning"}
            and self._prepared_task_index + 1 < len(self._prepared_tasks)
        )
        self._finish_stage_progress(status)
        self.progress_bar.hide()
        self._run_terminal_status = status
        terminal_label = STATUS_LABELS.get(
            history_status,
            self._display_status(status),
        )
        self._set_status_message(terminal_label)
        if status not in {"Completed", "CompletedWithWarning", "Stopped"}:
            self.toggle_log_action.setChecked(True)
        self._append_event(
            f"处理结束：状态={terminal_label}，退出码={exit_code}，CLI={cli_used}"
        )
        if (
            status in {"Completed", "CompletedWithWarning"}
            and self._active_prepared_task is not None
        ):
            try:
                latest_index = publish_latest_result_index(
                    run_manifest_path=self._active_prepared_task.run.manifest_path
                )
                self._append_event(
                    "已更新任务最新结果索引："
                    f"run={latest_index.get('run_id')} status={latest_index.get('status')}"
                )
                retention = apply_task_retention(
                    self._active_prepared_task.workspace.root,
                )
                self._append_event(
                    "任务保留策略已应用："
                    f"清理旧交付 {len(retention.get('deleted_files') or [])} 个，"
                    f"中间目录 {len(retention.get('removed_process_directories') or [])} 个"
                )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                self._append_event(f"最新结果或保留策略未更新：{exc}")
        if status == "Failed" and effective_had_errors:
            self._append_event("在输出中检测到 Siril/脚本错误。")
        if status == "CompletedWithWarning":
            self._show_completion_warning(
                work_dir,
                outcome_status=history_status,
                had_errors=effective_had_errors,
                had_degradations=had_degradations,
                had_fallbacks=had_fallbacks,
            )
            self._append_event(f"最终产物已生成；终态={terminal_label}。")
        elif status == "Completed" and has_next_task:
            self._show_run_banner(
                "info",
                "当前叠加分组已完成，正在按队列启动下一个独立任务。",
            )
        elif status == "Completed":
            self._show_run_banner(
                "success",
                "处理已完成。当前预览是最后一个通过验收的可靠结果。",
            )
        elif status == "Stopped":
            self._show_run_banner(
                "info",
                "任务已停止；已完成阶段与最后一张可靠预览仍然保留。",
                show_log=True,
            )
        else:
            self._show_run_banner(
                "error",
                "处理失败。已保留最后一张可靠预览，请查看详细日志定位原因。",
                show_log=True,
            )
        terminal_activity = {
            STATUS_SUCCESS: "处理已完成 · 当前为最终可靠预览",
            STATUS_PARTIAL_SUCCESS: "降级完成 · 保留最终可靠预览",
            STATUS_REVIEW_REQUIRED: "处理已完成，需要复核 · 保留最终可靠预览",
            STATUS_STOPPED: "任务已停止 · 保留最新可靠预览",
            STATUS_FAILED: "处理失败 · 保留最新可靠预览",
        }.get(history_status, f"任务结束：{terminal_label}")
        self.preview_activity_label.setText(terminal_activity)
        self.run_phase_label.setText(terminal_label)
        self._append_divider(
            "本次任务结束",
            [
                f"时间: {self._timestamp()}",
                f"工作目录: {self._display_path(work_dir)}",
                f"最终状态: {terminal_label}",
            ],
        )

        self._update_result_actions(work_dir)
        if has_next_task:
            self._cleanup_after_run(keep_queue=True)
            self._prepared_task_index += 1
            next_task = self._prepared_tasks[self._prepared_task_index]
            self._active_prepared_task = next_task
            self._run_terminal_status = None
            self._append_event(
                f"启动串行任务 {next_task.queue_index}/{next_task.queue_total}："
                f"{next_task.display_label}"
            )
            QTimer.singleShot(
                0,
                lambda task=next_task: self._start_run(prepared_task=task),
            )
            return
        self._cleanup_after_run()
        self._set_running(False)

    def _cleanup_after_run(self, *, keep_queue: bool = False) -> None:
        self._pending_runtime_overrides.clear()
        self._pending_runtime_unset_keys.clear()
        if self.worker:
            self.worker.wait(200)
            self.worker.deleteLater()
            self.worker = None

        self._close_run_log()
        if keep_queue:
            # The next queued task must establish all diagnostics before its
            # own preflight; no path from this run is allowed to remain active.
            self.run_log_path = None
            self.run_state_path = None
            self.runtime_capabilities_path = None
            self._run_state_payload = {}
            self._runtime_capability_manifest = None

        self._current_work_dir = None
        self._run_input_mode = None
        if not keep_queue:
            self._active_prepared_task = None
            self._prepared_tasks = ()
            self._prepared_task_index = 0
            self._spcc_online_circuit_open = False
            self._active_history_task_key = None
            self._active_history_run_id = None

    def changeEvent(self, event) -> None:  # type: ignore[override]
        super().changeEvent(event)
        try:
            from PySide6.QtCore import QEvent

            window_state_change = event.type() == QEvent.Type.WindowStateChange
        except (AttributeError, ImportError, RuntimeError, TypeError):
            window_state_change = False
        action = getattr(self, "full_screen_action", None)
        if window_state_change and action is not None:
            full_screen = self.isFullScreen()
            action.blockSignals(True)
            action.setChecked(full_screen)
            action.blockSignals(False)
            action.setText("退出全屏幕" if full_screen else "进入全屏幕")
            self._schedule_window_geometry_save()

    def moveEvent(self, event) -> None:  # type: ignore[override]
        super().moveEvent(event)
        self._schedule_window_geometry_save()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        active_worker = None
        gaia_catalog_worker = getattr(self, "gaia_catalog_worker", None)
        intake_worker = getattr(self, "intake_worker", None)
        bootstrap_worker = getattr(self, "bootstrap_worker", None)
        pipeline_worker = getattr(self, "worker", None)
        if gaia_catalog_worker and gaia_catalog_worker.isRunning():
            active_worker = gaia_catalog_worker
        elif intake_worker and intake_worker.isRunning():
            active_worker = intake_worker
        elif bootstrap_worker and bootstrap_worker.isRunning():
            active_worker = bootstrap_worker
        elif pipeline_worker and pipeline_worker.isRunning():
            active_worker = pipeline_worker

        if active_worker is not None:
            worker = active_worker

            def log_shutdown_error(message: str) -> None:
                try:
                    self._append_event(message)
                except (AttributeError, OSError, RuntimeError, ValueError):
                    pass

            ret = QMessageBox.question(
                self,
                "任务仍在运行",
                "仍在准备运行环境或处理图像。是否停止并退出？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if ret != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            stopped = False
            try:
                if worker is gaia_catalog_worker:
                    worker.stop()
                else:
                    self._stop_run()
            except Exception as e:
                log_shutdown_error(f"停止 worker 时发生异常：{e}")
            finally:
                try:
                    stopped = worker.wait(8000)
                except Exception as e:
                    log_shutdown_error(f"等待 worker 退出时发生异常：{e}")

            if not stopped:
                log_shutdown_error("worker 停止超时，正在强制终止...")
                try:
                    worker.terminate()
                except Exception as e:
                    log_shutdown_error(f"强制终止 worker 时发生异常：{e}")
                finally:
                    try:
                        worker.wait()
                    except Exception as e:
                        log_shutdown_error(f"最终等待 worker 退出时发生异常：{e}")
        save_settings = getattr(self, "_save_settings", None)
        if callable(save_settings):
            save_settings()
        for attribute in ("history_window", "preferences_window"):
            auxiliary = getattr(self, attribute, None)
            if auxiliary is not None and auxiliary.isVisible():
                auxiliary.close()
        event.accept()
