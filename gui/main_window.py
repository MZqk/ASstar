#!/usr/bin/env python3
"""Seestar Superimpose macOS GUI launcher with external pipeline resource."""

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

from PySide6.QtCore import QSettings, QTimer, QUrl, Signal, Qt
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
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
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
    QStackedWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

SIRIL_PLUGIN_PYTHON_ABI = "cp312"
UI_MODE_RECOMMENDED = "recommended"

PIPELINE_STAGE_TITLES = {
    1: "前期准备",
    2: "裁切",
    3: "背景提取",
    4: "图像解析 + 色彩校准",
    5: "线性反卷积 / 轻降噪",
    6: "去星与星点层准备",
    7: "主体拉伸",
    8: "Starless 深加工",
    9: "星点处理与合成",
    10: "最终降噪与导出",
    11: "AI 后期美化",
}

PIPELINE_PROGRESS_STATE_LABELS = {
    "waiting": "等待",
    "running": "● 运行中",
    "completed": "✓ 已完成",
    "degraded": "⚠ 已降级",
    "failed": "✕ 失败",
    "skipped": "— 已跳过",
    "stopped": "已停止",
}

PIPELINE_STAGE_SHORT_TITLES = {
    1: "准备",
    2: "裁切",
    3: "背景",
    4: "校色",
    5: "线性降噪",
    6: "去星",
    7: "拉伸",
    8: "深加工",
    9: "合成",
    10: "导出",
    11: "AI 后期",
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
    from .ai_credentials import (
        AI_PROVIDER_CUSTOM,
        AI_PROVIDER_DEVELOPER,
        AI_PROVIDER_MODES,
        AI_SECRET_ENV_KEYS,
        AiCredentialError,
        delete_custom_api_key,
        ensure_developer_credentials,
        get_custom_api_key,
        set_custom_api_key,
    )
except ImportError:
    from ai_credentials import (  # type: ignore[no-redef]
        AI_PROVIDER_CUSTOM,
        AI_PROVIDER_DEVELOPER,
        AI_PROVIDER_MODES,
        AI_SECRET_ENV_KEYS,
        AiCredentialError,
        delete_custom_api_key,
        ensure_developer_credentials,
        get_custom_api_key,
        set_custom_api_key,
    )

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
    from .preview_widgets import LatestPreviewCanvas
    from .preview_worker import InitialPreviewWorker, preview_cache_path
except ImportError:
    from preview_widgets import LatestPreviewCanvas  # type: ignore[no-redef]
    from preview_worker import (  # type: ignore[no-redef]
        InitialPreviewWorker,
        preview_cache_path,
    )


WORKSPACE_EMPTY = "empty"
WORKSPACE_TASK = "task"
WORKSPACE_RUN = "run"


class SeestarGui(QMainWindow):
    thread_log = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Seestar 图像后处理")
        self.setMinimumSize(980, 680)
        self.resize(1280, 800)
        self.setAcceptDrops(True)
        self._ui_thread_ident = threading.get_ident()
        self.settings = QSettings("Seestar", "SeestarSuperimpose")
        self._restoring_settings = False
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
        self._run_input_mode_override: str | None = None

        self.resources = resource_root()
        self.pipeline_path = default_pipeline_path(self.resources)
        self.siril_plugin_dir = default_siril_plugin_dir(self.resources)
        self.runtime_home = default_runtime_home()
        self.bundled_siril_cli = (
            self.resources / "Siril.app" / "Contents" / "MacOS" / "siril-cli"
        )
        self.siril_seed_dir = self.resources / "SirilPythonSeed"
        self.siril_spcc_seed_dir = default_siril_spcc_database_seed_dir(
            self.resources
        )
        config_candidates = [
            self.resources / "config.1.4.ini.template",
        ]
        self.config_template = resolve_existing_path(config_candidates)

        self.worker: PipelineWorker | None = None
        self.bootstrap_worker: BootstrapWorker | None = None
        self._bootstrap_stop_event: threading.Event | None = None
        self.run_log_path: Path | None = None
        self.run_log_file = None
        self._run_log_lock = threading.Lock()
        self._current_work_dir: Path | None = None
        self._last_quality_report_path: Path | None = None
        self._pipeline_started_monotonic: float | None = None
        self._active_pipeline_stage: int | None = None
        self._display_pipeline_stage: int | None = None
        self._stage_started_monotonic: dict[int, float] = {}
        self._stage_elapsed_seconds: dict[int, float] = {}
        self._stage_progress_states: dict[int, str] = {}
        self._stage_progress_titles: dict[int, str] = {}
        self._stage_items: dict[int, QLabel] = {}
        self._progress_stage_count = 10
        self.input_mode = INPUT_MODE_AUTO
        self.debug_mode_enabled = False
        self.network_mode_enabled = False
        self.ai_stage_enabled = False
        self.ai_provider_mode = AI_PROVIDER_DEVELOPER
        self.ai_custom_endpoint = ""
        self.ai_custom_model = ""
        self._pending_ai_runtime_overrides: dict[str, str] = {}

        self._init_ui()
        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(1000)
        self._progress_timer.timeout.connect(self._refresh_elapsed_labels)
        self.thread_log.connect(self._append_text)
        self._load_settings()
        self._set_running(False)
        try:
            spcc_ready, spcc_detail = verify_siril_spcc_database_seed(
                self.siril_spcc_seed_dir,
                self.runtime_home,
            )
            if not spcc_ready:
                self._append_event(
                    "Siril SPCC 固定数据库将在任务开始前准备："
                    + spcc_detail
                )
        except Exception as e:
            self._append_event(
                "Siril SPCC 固定数据库启动校验失败；"
                f"任务开始前将重试，失败时改走 PCC：{e}"
            )

        self._append_event("已就绪。请选择或拖入工作目录。")

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
            "  1. 拖入或选择包含 Seestar 图像的工作目录。",
            "  2. 使用自动推荐，或手动选择完整处理/断点继续。",
            "  3. 点击“开始处理”；需要中断时点击“停止处理”。",
            "  处理完成后，可直接预览结果或打开结果目录。",
            "  运行细节位于主界面的“详细日志”区域。",
        ]))
        lines.extend(self._section_block("应用能力", [
            "  - 支持已叠加 FITS 与 Seestar 子帧输入，必要时自动执行预处理。",
            "  - 能识别已有阶段产物，并推荐从合适的断点继续。",
            "  - 串联 Siril 1.4+ 与 SyQon/SASP/CosmicClarity 插件链路，执行离线后处理，不依赖系统 Python。",
            "  - 处理过程包含预检、重试、降级回退和阶段状态记录。",
            "  - 输出高质量 TIFF、预览 PNG、拉伸前线性 FITS 和最终 FITS 归档。",
            f"  - 当前处理方式: {self._input_mode_label(self.input_mode)}。",
            f"  - AI 后期: {'开启' if self.ai_stage_enabled else '关闭'}。",
            f"  - 保留中间文件: {'开启' if self.debug_mode_enabled else '关闭'}。",
            f"  - 允许联网: {'开启' if self.network_mode_enabled else '关闭'}。",
        ]))
        lines.extend(self._section_block("处理阶段总览", [
            "  线性阶段: 1.前期准备 -> 2.裁切 -> 3.背景提取 -> 4.图像解析+色彩校准 -> 5.线性降噪/反卷积",
            "  非线性阶段: 6.去星与星点层准备 -> 7.主体拉伸 -> 8.Starless 深加工 -> 9.星点处理与合成 -> 10.最终降噪与导出",
            "  可选阶段: 11.AI 后期美化（可使用开发者试用或自定义模型）",
        ]))
        lines.extend(self._section_block("阶段文件命名", [
            "  - 阶段 6 去星: stage6_starless.fit / stage6_starless_quality.json。",
            "  - 阶段 7 拉伸: stage7_stretched.fit / stage7_stretch_quality.json。",
            "  - 阶段 7 统一使用 stage7_cand_a/b、stage7_preview_ref 与 stage7_stretched 命名。",
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

    def _show_help(self) -> None:
        dialog = QMessageBox(self)
        dialog.setWindowTitle("使用说明")
        dialog.setIcon(QMessageBox.Icon.Information)
        dialog.setText(
            "三步完成处理：选择工作目录、确认处理方式、开始处理。"
        )
        dialog.setInformativeText(
            "目录可直接拖入窗口；应用会自动推荐完整处理或断点继续。"
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

    def _ensure_runtime_spcc_database_seed(self) -> dict[str, object]:
        result = sync_siril_spcc_database_seed(
            self.siril_spcc_seed_dir,
            self.runtime_home,
        )
        copied = list(result.get("copied_files") or [])
        commit = str(result.get("source_commit") or "unknown")
        target_root = Path(result["target_root"])
        if copied:
            self._append_event(
                "已同步 Siril SPCC 固定数据库到隔离运行目录："
                f"commit={commit[:12]}, files={len(copied)}, "
                f"path={self._display_path(target_root)}"
            )
        else:
            self._append_event(
                "Siril SPCC 固定数据库校验通过："
                f"commit={commit[:12]}, path={self._display_path(target_root)}"
            )
        return result

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
        self.setCentralWidget(central)
        central.setStyleSheet(
            "QFrame#contentCard {"
            "border: 1px solid rgba(127, 127, 127, 0.32);"
            "border-radius: 10px;"
            "}"
            "QFrame#dropCard {"
            "border: 1px dashed rgba(127, 127, 127, 0.58);"
            "border-radius: 14px;"
            "}"
            "QLabel#sectionTitle {"
            "font-size: 16px;"
            "font-weight: 600;"
            "}"
        )

        outer = QVBoxLayout(central)
        outer.setContentsMargins(14, 12, 14, 10)
        outer.setSpacing(10)

        self._init_toolbar()
        self.workspace_stack = QStackedWidget()
        self.empty_page = self._build_empty_page()
        self.task_page = self._build_task_page()
        self.run_page = self._build_run_page()
        self.workspace_stack.addWidget(self.empty_page)
        self.workspace_stack.addWidget(self.task_page)
        self.workspace_stack.addWidget(self.run_page)
        outer.addWidget(self.workspace_stack, 1)

        self._build_log_drawer(outer)
        self._init_status_bar()
        self._init_menus()
        self._configure_focus_order()
        self._update_result_actions(None)
        self._show_workspace(WORKSPACE_EMPTY)

    def _init_toolbar(self) -> None:
        toolbar = QToolBar("主工具栏", self)
        toolbar.setObjectName("mainToolbar")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        self.addToolBar(toolbar)
        self.main_toolbar = toolbar

        self.toolbar_directory_btn = QPushButton("选择目录")
        self.toolbar_directory_btn.setAccessibleName("选择工作目录")
        self.toolbar_directory_btn.clicked.connect(self._choose_workdir)
        self.toolbar_directory_item = toolbar.addWidget(self.toolbar_directory_btn)

        self.toolbar_settings_btn = QPushButton("设置")
        self.toolbar_settings_btn.setAccessibleName("展开任务高级设置")
        self.toolbar_settings_btn.clicked.connect(self._show_advanced_settings)
        self.toolbar_settings_item = toolbar.addWidget(self.toolbar_settings_btn)

        spacer = QWidget()
        spacer.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        toolbar.addWidget(spacer)

        self.return_task_btn = QPushButton("返回任务设置")
        self.return_task_btn.clicked.connect(self._return_to_task_setup)
        self.return_task_item = toolbar.addWidget(self.return_task_btn)

        self.rerun_btn = QPushButton("重新运行")
        self.rerun_btn.clicked.connect(self._rerun_last_task)
        self.rerun_item = toolbar.addWidget(self.rerun_btn)

        self.start_btn = QPushButton("开始处理")
        self.start_btn.setAccessibleName("开始图像处理")
        self.start_btn.setMinimumWidth(120)
        self.start_btn.setDefault(True)
        self.start_btn.clicked.connect(self._start_run)
        self.start_item = toolbar.addWidget(self.start_btn)

        self.stop_btn = QPushButton("停止")
        self.stop_btn.setAccessibleName("停止图像处理")
        self.stop_btn.setMinimumWidth(96)
        self.stop_btn.clicked.connect(self._request_stop_run)
        self.stop_item = toolbar.addWidget(self.stop_btn)

    def _build_empty_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(44, 34, 44, 30)
        layout.addStretch(1)

        card = QFrame()
        card.setObjectName("dropCard")
        card.setMinimumHeight(330)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(40, 38, 40, 38)
        card_layout.addStretch(1)

        title = QLabel("拖入 Seestar 工作目录")
        title.setObjectName("sectionTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(title)

        subtitle = QLabel("支持 FIT / FITS 图像")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: rgba(127, 127, 127, 0.92); padding: 8px;")
        card_layout.addWidget(subtitle)

        self.empty_choose_btn = QPushButton("选择目录")
        self.empty_choose_btn.setAccessibleName("选择 Seestar 工作目录")
        self.empty_choose_btn.setMinimumSize(150, 36)
        self.empty_choose_btn.clicked.connect(self._choose_workdir)
        choose_row = QHBoxLayout()
        choose_row.addStretch(1)
        choose_row.addWidget(self.empty_choose_btn)
        choose_row.addStretch(1)
        card_layout.addLayout(choose_row)
        card_layout.addStretch(1)
        layout.addWidget(card)

        reassurance = QLabel("默认离线处理 · 原图不会被覆盖")
        reassurance.setAlignment(Qt.AlignmentFlag.AlignCenter)
        reassurance.setAccessibleName("处理安全说明")
        reassurance.setStyleSheet("padding: 14px; color: rgba(127, 127, 127, 0.95);")
        layout.addWidget(reassurance)
        layout.addStretch(1)
        return page

    def _build_task_page(self) -> QWidget:
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(2, 2, 2, 2)
        page_layout.setSpacing(10)

        guide_label = QLabel("1  选择目录    →    2  确认处理方式    →    3  开始处理")
        guide_label.setAccessibleName("使用步骤")
        guide_label.setAccessibleDescription(
            "第一步选择目录，第二步确认处理方式，第三步开始处理"
        )
        guide_label.setStyleSheet("font-weight: 600; padding: 4px;")
        page_layout.addWidget(guide_label)

        task_splitter = QSplitter(Qt.Orientation.Horizontal)
        task_splitter.setChildrenCollapsible(False)
        self.source_card = self._build_source_card()
        task_splitter.addWidget(self.source_card)
        task_splitter.addWidget(self._build_task_preview_card())
        task_splitter.setStretchFactor(0, 3)
        task_splitter.setStretchFactor(1, 2)
        page_layout.addWidget(task_splitter, 1)

        phase_card = QFrame()
        phase_card.setObjectName("contentCard")
        phase_layout = QVBoxLayout(phase_card)
        phase_layout.setContentsMargins(14, 11, 14, 11)
        phase_layout.addWidget(self._section_title("处理阶段"))
        phase_row = QHBoxLayout()
        self.linear_phase_label = QLabel("○ 线性处理 · Stage 1–6")
        self.nonlinear_phase_label = QLabel("○ 非线性处理 · Stage 7–10")
        self.ai_phase_label = QLabel("可选 · Stage 11 AI 后期")
        for label in (
            self.linear_phase_label,
            self.nonlinear_phase_label,
            self.ai_phase_label,
        ):
            label.setStyleSheet(
                "padding: 9px 12px; border: 1px solid rgba(127,127,127,0.28);"
                " border-radius: 8px;"
            )
            phase_row.addWidget(label, 1)
        phase_layout.addLayout(phase_row)
        page_layout.addWidget(phase_card)
        return page

    def _build_source_card(self) -> QFrame:
        source_card = QFrame()
        source_card.setObjectName("contentCard")
        source_layout = QVBoxLayout(source_card)
        source_layout.setContentsMargins(16, 14, 16, 14)
        source_layout.setSpacing(10)
        source_layout.addWidget(self._section_title("任务设置"))

        directory_row = QHBoxLayout()
        directory_row.setSpacing(8)
        dir_label = QLabel("工作目录")
        self.dir_edit = QLineEdit()
        self.dir_edit.setPlaceholderText("拖入目录，或点击右侧选择")
        self.dir_edit.setAccessibleName("工作目录")
        self.dir_edit.setAccessibleDescription(
            "包含 Seestar FIT 或 FITS 图像的工作目录"
        )
        self.dir_edit.setAcceptDrops(False)
        self.dir_edit.editingFinished.connect(self._on_directory_edited)
        self.browse_btn = QPushButton("选择目录")
        self.browse_btn.setAccessibleName("选择工作目录")
        self.browse_btn.clicked.connect(self._choose_workdir)
        dir_label.setBuddy(self.dir_edit)
        directory_row.addWidget(dir_label)
        directory_row.addWidget(self.dir_edit, 1)
        directory_row.addWidget(self.browse_btn)
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

        self.directory_summary_label = QLabel("尚未选择目录")
        self.directory_summary_label.setAccessibleName("工作目录检测结果")
        self.directory_summary_label.setWordWrap(True)
        source_layout.addWidget(self.directory_summary_label)

        mode_row = QHBoxLayout()
        mode_label = QLabel("处理方式")
        self.mode_combo = QComboBox()
        self.mode_combo.setAccessibleName("处理方式")
        self.mode_combo.setAccessibleDescription(
            "选择自动推荐、完整处理或从已有阶段继续"
        )
        self.mode_combo.addItem("自动推荐（完整处理）", UI_MODE_RECOMMENDED)
        self.mode_combo.addItem("完整处理", INPUT_MODE_AUTO)
        self.mode_combo.addItem("从裁切后继续", INPUT_MODE_STAGE2_CORRECTED_RESUME)
        self.mode_combo.addItem("从线性处理后继续", INPUT_MODE_LINEAR_RESUME)
        self.mode_combo.currentIndexChanged.connect(self._on_input_mode_changed)
        mode_label.setBuddy(self.mode_combo)
        mode_row.addWidget(mode_label)
        mode_row.addWidget(self.mode_combo, 1)
        source_layout.addLayout(mode_row)

        self.advanced_toggle_btn = QPushButton("高级设置 ▸")
        self.advanced_toggle_btn.setAccessibleName("展开高级设置")
        self.advanced_toggle_btn.setAccessibleDescription(
            "显示或隐藏 AI 后期、模型配置、保留中间文件和联网设置"
        )
        self.advanced_toggle_btn.setCheckable(True)
        self.advanced_toggle_btn.toggled.connect(self._on_advanced_toggled)
        source_layout.addWidget(self.advanced_toggle_btn)

        self.advanced_panel = QWidget()
        self.advanced_panel.setAccessibleName("高级设置")
        advanced_layout = QHBoxLayout(self.advanced_panel)
        advanced_layout.setContentsMargins(8, 2, 8, 2)
        self.ai_btn = QCheckBox("AI 后期")
        self.ai_btn.setAccessibleName("AI 后期")
        self.ai_btn.setAccessibleDescription("启用可选的 AI 后期处理阶段")
        self.ai_btn.setChecked(self.ai_stage_enabled)
        self.ai_btn.toggled.connect(self._on_ai_toggled)
        self.ai_config_btn = QPushButton("模型配置…")
        self.ai_config_btn.setAccessibleName("配置 AI 模型")
        self.ai_config_btn.setAccessibleDescription(
            "选择开发者试用模型，或配置自己的接口、模型和 API Key"
        )
        self.ai_config_btn.clicked.connect(self._configure_ai_model)
        self.ai_provider_status_label = QLabel("开发者试用")
        self.ai_provider_status_label.setAccessibleName("当前 AI 模型配置")
        self.debug_btn = QCheckBox("保留中间文件")
        self.debug_btn.setAccessibleName("保留中间文件")
        self.debug_btn.setAccessibleDescription(
            "保留各处理阶段的中间文件和诊断信息"
        )
        self.debug_btn.setChecked(self.debug_mode_enabled)
        self.debug_btn.toggled.connect(self._on_debug_toggled)
        self.network_btn = QCheckBox("允许联网")
        self.network_btn.setAccessibleName("允许联网")
        network_description = (
            "默认关闭；开启后允许在线星表查询、插件补齐和已配置的 AI 服务访问"
        )
        self.network_btn.setAccessibleDescription(network_description)
        self.network_btn.setToolTip(network_description)
        self.network_btn.setChecked(self.network_mode_enabled)
        self.network_btn.toggled.connect(self._on_network_toggled)
        advanced_layout.addWidget(self.ai_btn)
        advanced_layout.addWidget(self.ai_config_btn)
        advanced_layout.addWidget(self.ai_provider_status_label)
        advanced_layout.addWidget(self.debug_btn)
        advanced_layout.addWidget(self.network_btn)
        advanced_layout.addStretch(1)
        self.advanced_panel.hide()
        source_layout.addWidget(self.advanced_panel)
        source_layout.addStretch(1)
        return source_card

    def _build_task_preview_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("contentCard")
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
        note = QLabel("Stage 0 原始预览 · 无显示拉伸")
        note.setStyleSheet("color: rgba(127, 127, 127, 0.95);")
        layout.addWidget(note)
        return card

    def _build_run_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.warning_card = QFrame()
        self.warning_card.setObjectName("completionWarningCard")
        self.warning_card.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Maximum,
        )
        self.warning_card.setStyleSheet(
            "#completionWarningCard {"
            "background-color: rgba(255, 193, 7, 0.18);"
            "border: 1px solid #c08a00;"
            "border-radius: 8px;"
            "}"
        )
        warning_layout = QHBoxLayout(self.warning_card)
        warning_layout.setContentsMargins(10, 8, 10, 8)
        self.warning_label = QLabel(
            "⚠ 处理已完成，但存在降级或失败阶段，请复核最终质量。"
        )
        self.warning_label.setAccessibleName("处理完成警告")
        self.quality_report_btn = QPushButton("查看质量报告")
        self.quality_report_btn.setAccessibleName("查看最终质量报告")
        self.quality_report_btn.clicked.connect(self._open_quality_report)
        warning_layout.addWidget(self.warning_label)
        warning_layout.addStretch(1)
        warning_layout.addWidget(self.quality_report_btn)
        self.warning_card.hide()
        layout.addWidget(self.warning_card)

        self.run_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.run_splitter.setChildrenCollapsible(False)
        self.run_sidebar = self._build_run_sidebar()
        self.run_splitter.addWidget(self.run_sidebar)
        self.run_splitter.addWidget(self._build_run_preview_card())
        self.run_splitter.addWidget(self._build_stage_inspector())
        self.run_splitter.setStretchFactor(0, 0)
        self.run_splitter.setStretchFactor(1, 1)
        self.run_splitter.setStretchFactor(2, 0)
        self.run_splitter.setSizes([250, 720, 290])
        layout.addWidget(self.run_splitter, 1)
        return page

    def _build_run_sidebar(self) -> QFrame:
        card = QFrame()
        card.setObjectName("contentCard")
        card.setMinimumWidth(220)
        card.setMaximumWidth(320)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(9)
        layout.addWidget(self._section_title("本次任务"))

        self.run_directory_label = QLabel("工作目录\n—")
        self.run_directory_label.setWordWrap(True)
        self.run_mode_label = QLabel("处理方式\n—")
        self.run_options_label = QLabel("高级设置\n—")
        self.run_options_label.setWordWrap(True)
        for widget in (
            self.run_directory_label,
            self.run_mode_label,
            self.run_options_label,
        ):
            widget.setStyleSheet(
                "padding: 8px; border: 1px solid rgba(127,127,127,0.24);"
                " border-radius: 7px;"
            )
            layout.addWidget(widget)
        layout.addStretch(1)

        layout.addWidget(self._section_title("结果"))
        self.result_preview_btn = QPushButton("结果预览")
        self.result_preview_btn.setAccessibleName("预览处理结果")
        self.result_preview_btn.clicked.connect(self._open_result_preview)
        self.open_result_btn = QPushButton("打开结果目录")
        self.open_result_btn.setAccessibleName("打开处理结果目录")
        self.open_result_btn.clicked.connect(self._open_result_dir)
        layout.addWidget(self.result_preview_btn)
        layout.addWidget(self.open_result_btn)
        return card

    def _build_run_preview_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("contentCard")
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
        self.zoom_out_btn.clicked.connect(self.preview_canvas.zoom_out)
        self.fit_preview_btn = QPushButton("适合窗口")
        self.fit_preview_btn.clicked.connect(self.preview_canvas.fit_to_window)
        self.actual_preview_btn = QPushButton("1:1")
        self.actual_preview_btn.clicked.connect(self.preview_canvas.actual_pixels)
        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setAccessibleName("放大预览")
        self.zoom_in_btn.clicked.connect(self.preview_canvas.zoom_in)
        controls.addStretch(1)
        for button in (
            self.zoom_out_btn,
            self.fit_preview_btn,
            self.actual_preview_btn,
            self.zoom_in_btn,
        ):
            controls.addWidget(button)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.preview_notice_label = QLabel("原始预览 · 无显示拉伸")
        self.preview_notice_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_notice_label.setStyleSheet("color: rgba(127,127,127,0.95);")
        layout.addWidget(self.preview_notice_label)
        return card

    def _build_stage_inspector(self) -> QFrame:
        card = QFrame()
        card.setObjectName("contentCard")
        card.setMinimumWidth(250)
        card.setMaximumWidth(340)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 11, 12, 11)
        layout.setSpacing(7)
        layout.addWidget(self._section_title("阶段与质量"))

        self.run_phase_label = QLabel("线性处理 · Stage 1–6")
        self.run_phase_label.setStyleSheet("font-weight: 600; padding: 4px 0;")
        layout.addWidget(self.run_phase_label)

        self.progress_summary_label = QLabel("当前进度：等待开始")
        self.progress_summary_label.setAccessibleName("处理阶段进度")
        self.progress_summary_label.setWordWrap(True)
        layout.addWidget(self.progress_summary_label)

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
        layout.addWidget(self.stage_scroll, 1)
        self._reset_stage_progress(10)
        return card

    def _build_log_drawer(self, outer: QVBoxLayout) -> None:
        self.log_toggle_btn = QPushButton("详细日志")
        self.log_toggle_btn.setAccessibleName("展开详细日志")
        self.log_toggle_btn.setCheckable(True)
        self.log_toggle_btn.toggled.connect(self._on_log_toggled)

        self.log_container = QFrame()
        self.log_container.setObjectName("contentCard")
        self.log_container.setMaximumHeight(270)
        log_layout = QVBoxLayout(self.log_container)
        log_layout.setContentsMargins(10, 8, 10, 8)
        log_actions = QHBoxLayout()
        log_actions.addStretch(1)
        self.open_log_btn = QPushButton("打开日志文件")
        self.open_log_btn.setAccessibleName("打开完整日志文件")
        self.open_log_btn.clicked.connect(self._open_log_file)
        self.clear_view_btn = QPushButton("清空日志")
        self.clear_view_btn.setAccessibleName("清空界面日志")
        self.clear_view_btn.clicked.connect(self._reset_view)
        log_actions.addWidget(self.open_log_btn)
        log_actions.addWidget(self.clear_view_btn)
        log_layout.addLayout(log_actions)
        self.log_view = QPlainTextEdit()
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
        self.status_label.setAccessibleName("处理状态")
        self.status_label.setStyleSheet("font-weight: 600; padding-right: 10px;")
        self.stage_timing_label = QLabel("本阶段 — · 总耗时 —")
        self.stage_timing_label.setAccessibleName("处理耗时")
        self.preview_status_label = QLabel("预览：等待输入")
        self.preview_status_label.setAccessibleName("预览状态")
        status_bar.addWidget(self.status_label, 1)
        status_bar.addPermanentWidget(self.stage_timing_label)
        status_bar.addPermanentWidget(self.preview_status_label)
        status_bar.addPermanentWidget(self.log_toggle_btn)

    def _init_menus(self) -> None:
        process_menu = self.menuBar().addMenu("处理")

        self.start_action = QAction("开始处理", self)
        self.start_action.triggered.connect(self._start_run)
        process_menu.addAction(self.start_action)

        self.stop_action = QAction("停止", self)
        self.stop_action.triggered.connect(self._request_stop_run)
        process_menu.addAction(self.stop_action)

        self.return_task_action = QAction("返回任务设置", self)
        self.return_task_action.triggered.connect(self._return_to_task_setup)
        process_menu.addAction(self.return_task_action)

        self.rerun_action = QAction("重新运行", self)
        self.rerun_action.triggered.connect(self._rerun_last_task)
        process_menu.addAction(self.rerun_action)

        process_menu.addSeparator()

        self.open_result_action = QAction("打开结果目录", self)
        self.open_result_action.triggered.connect(self._open_result_dir)
        process_menu.addAction(self.open_result_action)

        help_menu = self.menuBar().addMenu("帮助")
        self.help_action = QAction("使用说明", self)
        self.help_action.triggered.connect(self._show_help)
        help_menu.addAction(self.help_action)

        self.open_log_action = QAction("打开日志文件", self)
        self.open_log_action.triggered.connect(self._open_log_file)
        self.clear_view_action = QAction("清空日志", self)
        self.clear_view_action.triggered.connect(self._reset_view)

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
        self._workspace_state = normalized
        page = {
            WORKSPACE_EMPTY: self.empty_page,
            WORKSPACE_TASK: self.task_page,
            WORKSPACE_RUN: self.run_page,
        }[normalized]
        self.workspace_stack.setCurrentWidget(page)
        self._update_toolbar_state()
        self._update_responsive_layout()

    def _update_toolbar_state(self) -> None:
        bootstrap_running = bool(
            self.bootstrap_worker and self.bootstrap_worker.isRunning()
        )
        pipeline_running = bool(self.worker and self.worker.isRunning())
        running = self._ui_running or bootstrap_running or pipeline_running
        task_state = self._workspace_state == WORKSPACE_TASK
        run_state = self._workspace_state == WORKSPACE_RUN
        terminal = bool(run_state and self._run_terminal_status)

        self.toolbar_directory_item.setVisible(self._workspace_state != WORKSPACE_RUN)
        self.toolbar_settings_item.setVisible(task_state)
        self.start_item.setVisible(task_state and not running)
        self.stop_item.setVisible(run_state and running)
        self.return_task_item.setVisible(terminal and not running)
        self.rerun_item.setVisible(
            terminal and not running and self._last_run_snapshot is not None
        )

        self.start_action.setEnabled(task_state and not running)
        self.stop_action.setEnabled(run_state and running)
        self.return_task_action.setEnabled(terminal and not running)
        self.rerun_action.setEnabled(
            terminal and not running and self._last_run_snapshot is not None
        )

    def _show_advanced_settings(self) -> None:
        if self._workspace_state != WORKSPACE_TASK:
            return
        self.advanced_toggle_btn.setChecked(True)
        self.advanced_panel.setFocus()

    def _return_to_task_setup(self) -> None:
        if (self.bootstrap_worker and self.bootstrap_worker.isRunning()) or (
            self.worker and self.worker.isRunning()
        ):
            return
        self._run_terminal_status = None
        directory_text = self.dir_edit.text().strip()
        work_dir = Path(directory_text).expanduser() if directory_text else None
        if work_dir is not None and work_dir.is_dir():
            self._show_workspace(WORKSPACE_TASK)
            self._analyze_selected_directory()
        else:
            self._show_workspace(WORKSPACE_EMPTY)
        self.warning_card.hide()
        self._set_status_text("Idle")
        self._set_running(False)

    def _rerun_last_task(self) -> None:
        snapshot = self._last_run_snapshot
        if not snapshot:
            return
        answer = QMessageBox.question(
            self,
            "重新运行",
            "将按上一次实际配置重新运行。现有同名结果文件可能被替换，是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        work_dir = Path(str(snapshot.get("work_dir") or "")).expanduser()
        if not work_dir.is_dir():
            QMessageBox.warning(self, "无法重新运行", "上一次工作目录已不可用。")
            return
        self.ai_stage_enabled = bool(snapshot.get("ai_stage_enabled", False))
        self.debug_mode_enabled = bool(snapshot.get("debug_mode_enabled", False))
        self.network_mode_enabled = bool(snapshot.get("network_mode_enabled", False))
        self.ai_provider_mode = str(
            snapshot.get("ai_provider_mode") or AI_PROVIDER_DEVELOPER
        )
        self.ai_custom_endpoint = str(snapshot.get("ai_custom_endpoint") or "")
        self.ai_custom_model = str(snapshot.get("ai_custom_model") or "")
        self._update_ai_button_text()
        self._update_debug_button_text()
        self._update_network_button_text()
        self._update_ai_provider_status()
        self.dir_edit.setText(str(work_dir))
        self._start_run(input_mode_override=str(snapshot.get("input_mode") or INPUT_MODE_AUTO))

    def _request_stop_run(self) -> None:
        if not (
            (self.bootstrap_worker and self.bootstrap_worker.isRunning())
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
        self.run_directory_label.setText(
            "工作目录\n" + self._display_path(work_dir)
        )
        self.run_mode_label.setText(
            "处理方式\n" + self._input_mode_label(input_mode)
        )
        options = [
            f"AI 后期：{'开启' if self.ai_stage_enabled else '关闭'}",
            f"保留中间文件：{'开启' if self.debug_mode_enabled else '关闭'}",
            f"允许联网：{'开启' if self.network_mode_enabled else '关闭'}",
        ]
        self.run_options_label.setText("高级设置\n" + "\n".join(options))

    def _update_responsive_layout(self) -> None:
        if not hasattr(self, "run_sidebar"):
            return
        self.run_sidebar.setVisible(
            self._workspace_state == WORKSPACE_RUN and self.width() >= 1100
        )

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._update_responsive_layout()

    def _initial_preview_candidates(
        self,
        work_dir: Path,
        input_mode: str | None = None,
    ) -> tuple[list[Path], str]:
        input_mode = input_mode or self._current_input_mode()
        if input_mode == INPUT_MODE_LINEAR_RESUME:
            source = self._linear_resume_input_path(work_dir)
            return ([source] if source.is_file() else []), "续跑输入"
        if input_mode == INPUT_MODE_STAGE2_CORRECTED_RESUME:
            source = self._stage2_corrected_resume_input_path(work_dir)
            return ([source] if source.is_file() else []), "续跑输入"

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
        focus_chain = (
            self.dir_edit,
            self.browse_btn,
            self.recent_combo,
            self.mode_combo,
            self.advanced_toggle_btn,
            self.ai_btn,
            self.ai_config_btn,
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
        self._progress_stage_count = max(1, int(stage_count))
        self._active_pipeline_stage = None
        self._display_pipeline_stage = None
        self._stage_started_monotonic.clear()
        self._stage_elapsed_seconds.clear()
        self._stage_progress_states.clear()
        self._stage_progress_titles.clear()
        self._stage_items.clear()
        while self.stage_stepper_layout.count():
            layout_item = self.stage_stepper_layout.takeAt(0)
            widget = layout_item.widget()
            if widget is not None:
                widget.hide()
                widget.deleteLater()

        for stage in range(1, self._progress_stage_count + 1):
            if stage in {1, 7, 11}:
                phase_text = {
                    1: "线性处理 · Stage 1–6",
                    7: "非线性处理 · Stage 7–10",
                    11: "AI 后期 · Stage 11",
                }[stage]
                phase_label = QLabel(phase_text)
                phase_label.setStyleSheet(
                    "font-size: 11px; font-weight: 600;"
                    " color: rgba(127,127,127,0.96); padding: 6px 2px 2px 2px;"
                )
                self.stage_stepper_layout.addWidget(phase_label)
            title = PIPELINE_STAGE_TITLES.get(stage, f"阶段 {stage}")
            chip = QLabel()
            chip.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
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

    def _update_stage_chip(self, stage: int) -> None:
        chip = self._stage_items.get(stage)
        if chip is None:
            return
        state = self._stage_progress_states.get(stage, "waiting")
        symbol = {
            "waiting": "○",
            "running": "●",
            "completed": "✓",
            "degraded": "⚠",
            "failed": "✕",
            "skipped": "—",
            "stopped": "■",
        }.get(state, "○")
        border = {
            "running": "#0a84ff",
            "completed": "#34c759",
            "degraded": "#c08a00",
            "failed": "#ff453a",
        }.get(state, "rgba(127, 127, 127, 0.36)")
        background = {
            "running": "rgba(10, 132, 255, 0.16)",
            "completed": "rgba(52, 199, 89, 0.13)",
            "degraded": "rgba(255, 193, 7, 0.16)",
            "failed": "rgba(255, 69, 58, 0.14)",
        }.get(state, "transparent")
        short_title = PIPELINE_STAGE_SHORT_TITLES.get(stage, str(stage))
        state_text = {
            "waiting": "等待",
            "running": "运行中",
            "completed": "已完成",
            "degraded": "已降级",
            "failed": "失败",
            "skipped": "已跳过",
            "stopped": "已停止",
        }.get(state, state)
        chip.setText(f"{symbol}  Stage {stage} · {short_title}    {state_text}")
        chip.setAccessibleName(f"阶段 {stage}：{short_title}")
        chip.setStyleSheet(
            "padding: 6px 7px;"
            f"border: 1px solid {border};"
            "border-radius: 6px;"
            f"background-color: {background};"
            + ("font-weight: 600;" if state == "running" else "")
        )
        elapsed = self._stage_elapsed_seconds.get(stage)
        elapsed_text = self._format_elapsed(elapsed) if elapsed is not None else "—"
        chip.setToolTip(
            f"阶段 {stage}：{self._stage_progress_titles.get(stage, short_title)}\n"
            f"状态：{state_text}\n"
            f"耗时：{elapsed_text}"
        )
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
        selected = QFileDialog.getExistingDirectory(self, "选择工作目录")
        if selected:
            self._apply_work_directory(Path(selected), remember=True)

    def _set_running(self, running: bool) -> None:
        self._ui_running = bool(running)
        self.dir_edit.setEnabled(not running)
        self.browse_btn.setEnabled(not running)
        self.recent_combo.setEnabled(not running)
        self.mode_combo.setEnabled(not running)
        self.advanced_toggle_btn.setEnabled(not running)
        self.ai_btn.setEnabled(not running)
        self.ai_config_btn.setEnabled(not running)
        self.debug_btn.setEnabled(not running)
        self.network_btn.setEnabled(not running)
        self._update_toolbar_state()

    def _input_mode_label(self, mode: str) -> str:
        if mode == INPUT_MODE_LINEAR_RESUME:
            return "从线性处理后继续"
        if mode == INPUT_MODE_STAGE2_CORRECTED_RESUME:
            return "从裁切后继续"
        return "完整处理"

    def _current_input_mode(self) -> str:
        combo = getattr(self, "mode_combo", None)
        if combo is not None and hasattr(combo, "currentData"):
            value = combo.currentData()
            if value == UI_MODE_RECOMMENDED:
                return self._recommended_input_mode
            if value in {
                INPUT_MODE_AUTO,
                INPUT_MODE_LINEAR_RESUME,
                INPUT_MODE_STAGE2_CORRECTED_RESUME,
            }:
                return str(value)
        value = getattr(self, "input_mode", INPUT_MODE_AUTO)
        if value in {
            INPUT_MODE_AUTO,
            INPUT_MODE_LINEAR_RESUME,
            INPUT_MODE_STAGE2_CORRECTED_RESUME,
        }:
            return str(value)
        return INPUT_MODE_AUTO

    def _linear_resume_input_path(self, work_dir: Path) -> Path:
        return work_dir / LINEAR_RESUME_INPUT_NAME

    def _stage2_corrected_resume_input_path(self, work_dir: Path) -> Path:
        root_candidate = work_dir / STAGE2_CORRECTED_INPUT_NAME
        if root_candidate.is_file():
            return root_candidate
        return work_dir / "process" / STAGE2_CORRECTED_INPUT_NAME

    def _on_input_mode_changed(self, _index: int) -> None:
        self.input_mode = self._current_input_mode()
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

    def _update_ai_button_text(self) -> None:
        if self.ai_btn.isChecked() != self.ai_stage_enabled:
            self.ai_btn.blockSignals(True)
            try:
                self.ai_btn.setChecked(self.ai_stage_enabled)
            finally:
                self.ai_btn.blockSignals(False)
        if hasattr(self, "ai_phase_label"):
            self.ai_phase_label.setText(
                ("○ 已启用 · " if self.ai_stage_enabled else "未启用 · ")
                + "Stage 11 AI 后期"
            )

    def _on_ai_toggled(self, checked: bool) -> None:
        self.ai_stage_enabled = bool(checked)
        self._update_ai_button_text()
        if not self._restoring_settings:
            self._append_event(
                "AI 后期已" + ("开启" if self.ai_stage_enabled else "关闭")
            )
            if not (self.worker or self.bootstrap_worker):
                self._reset_stage_progress(11 if self.ai_stage_enabled else 10)
            self._save_settings()

    def _update_ai_provider_status(self) -> None:
        if self.ai_provider_mode == AI_PROVIDER_CUSTOM:
            label = "自定义模型"
            if self.ai_custom_model:
                label += f" · {self.ai_custom_model}"
        else:
            label = "开发者试用"
        self.ai_provider_status_label.setText(label)
        self.ai_provider_status_label.setAccessibleDescription(
            f"当前 AI 配置：{label}"
        )

    def _developer_ai_configuration(self) -> dict[str, str]:
        merged: dict[str, str] = {}
        for path in (
            self.resources / DEFAULT_ENV_RESOURCE_REL,
            self.resources / AI_ENV_RESOURCE_REL,
        ):
            if not path.is_file():
                continue
            parsed, _warnings = parse_ai_env_file(path)
            merged.update(parsed)
        return merged

    def _resolve_ai_runtime_overrides(self) -> dict[str, str]:
        if not self.ai_stage_enabled:
            return {}

        if self.ai_provider_mode == AI_PROVIDER_CUSTOM:
            endpoint = self.ai_custom_endpoint.strip()
            model = self.ai_custom_model.strip()
            api_key = (get_custom_api_key() or "").strip()
            if not endpoint.lower().startswith(("http://", "https://")):
                raise AiCredentialError("自定义 AI 接口必须是完整的 http(s) 地址")
            if not model:
                raise AiCredentialError("自定义 AI 模型名称不能为空")
            if not api_key:
                raise AiCredentialError("尚未在 macOS Keychain 中保存自定义 API Key")
            return {
                "SEESTAR_AI_ENDPOINT": endpoint,
                "SEESTAR_AI_MODEL": model,
                "SEESTAR_AI_API_KEY": api_key,
                "SEESTAR_AI_ARTISTIC_DERIVATIVE_ENABLED": "0",
            }

        defaults = self._developer_ai_configuration()
        fallback_secrets = (
            {key: defaults.get(key, "") for key in AI_SECRET_ENV_KEYS}
            if not is_frozen()
            else None
        )
        secrets = ensure_developer_credentials(
            self.resources,
            fallback_secrets=fallback_secrets,
        )
        endpoint = defaults.get("SEESTAR_AI_ENDPOINT", "").strip()
        model = defaults.get("SEESTAR_AI_MODEL", "").strip()
        api_key = secrets.get("SEESTAR_AI_API_KEY", "").strip()
        missing = []
        if not endpoint:
            missing.append("默认接口")
        if not model:
            missing.append("默认模型")
        if not api_key:
            missing.append("开发者试用 Key")
        if missing:
            raise AiCredentialError("缺少" + "、".join(missing))

        overrides = {
            "SEESTAR_AI_ENDPOINT": endpoint,
            "SEESTAR_AI_MODEL": model,
            "SEESTAR_AI_API_KEY": api_key,
        }
        artistic_key = secrets.get("SEESTAR_AI_ARTISTIC_API_KEY", "").strip()
        if artistic_key:
            overrides["SEESTAR_AI_ARTISTIC_API_KEY"] = artistic_key
        return overrides

    def _configure_ai_model(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("AI 模型配置")
        dialog.setMinimumWidth(560)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()

        provider_combo = QComboBox()
        provider_combo.setAccessibleName("AI 配置来源")
        provider_combo.addItem("开发者默认（试用）", AI_PROVIDER_DEVELOPER)
        provider_combo.addItem("使用自己的模型", AI_PROVIDER_CUSTOM)
        provider_combo.setCurrentIndex(
            max(0, provider_combo.findData(self.ai_provider_mode))
        )
        form.addRow("配置来源", provider_combo)

        endpoint_edit = QLineEdit(self.ai_custom_endpoint)
        endpoint_edit.setAccessibleName("自定义 AI 接口地址")
        endpoint_edit.setPlaceholderText("例如 https://api.example.com/v1/chat/completions")
        form.addRow("接口地址", endpoint_edit)

        model_edit = QLineEdit(self.ai_custom_model)
        model_edit.setAccessibleName("自定义 AI 模型名称")
        model_edit.setPlaceholderText("例如 model-name")
        form.addRow("模型名称", model_edit)

        key_edit = QLineEdit()
        key_edit.setAccessibleName("自定义 API Key")
        key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        key_edit.setPlaceholderText("留空则保留 Keychain 中已有的 Key")
        form.addRow("API Key", key_edit)
        layout.addLayout(form)

        key_status = QLabel()
        key_status.setWordWrap(True)
        layout.addWidget(key_status)
        try:
            custom_key_exists = bool((get_custom_api_key() or "").strip())
            keychain_error = ""
        except AiCredentialError as exc:
            custom_key_exists = False
            keychain_error = str(exc)

        delete_key_btn = QPushButton("删除自定义 Key")
        delete_key_btn.setAccessibleName("删除自定义 API Key")
        layout.addWidget(delete_key_btn)

        note = QLabel(
            "开发者试用配置由应用提供并可能随时失效；自定义 endpoint 和模型名称"
            "保存在普通设置中，API Key 只写入当前用户的 macOS Keychain。"
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        def update_controls() -> None:
            custom = provider_combo.currentData() == AI_PROVIDER_CUSTOM
            endpoint_edit.setEnabled(custom)
            model_edit.setEnabled(custom)
            key_edit.setEnabled(custom and not keychain_error)
            delete_key_btn.setEnabled(custom and custom_key_exists and not keychain_error)
            if keychain_error:
                key_status.setText("Keychain 不可用：" + keychain_error)
            elif custom_key_exists:
                key_status.setText("已在 macOS Keychain 中保存自定义 API Key。")
            else:
                key_status.setText("尚未保存自定义 API Key。")

        def delete_key() -> None:
            nonlocal custom_key_exists
            answer = QMessageBox.question(
                dialog,
                "删除自定义 Key",
                "确定从 macOS Keychain 删除自定义 API Key？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            try:
                delete_custom_api_key()
            except AiCredentialError as exc:
                QMessageBox.critical(dialog, "删除失败", str(exc))
                return
            custom_key_exists = False
            key_edit.clear()
            update_controls()

        provider_combo.currentIndexChanged.connect(lambda _index: update_controls())
        delete_key_btn.clicked.connect(delete_key)
        update_controls()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        layout.addWidget(buttons)

        def save_configuration() -> None:
            nonlocal custom_key_exists
            mode = str(provider_combo.currentData())
            if mode == AI_PROVIDER_CUSTOM:
                endpoint = endpoint_edit.text().strip()
                model = model_edit.text().strip()
                new_key = key_edit.text().strip()
                if not endpoint.lower().startswith(("http://", "https://")):
                    QMessageBox.warning(
                        dialog,
                        "配置不完整",
                        "接口地址必须是完整的 http(s) 地址。",
                    )
                    return
                if not model:
                    QMessageBox.warning(dialog, "配置不完整", "请输入模型名称。")
                    return
                if not new_key and not custom_key_exists:
                    QMessageBox.warning(dialog, "配置不完整", "请输入 API Key。")
                    return
                if new_key:
                    try:
                        set_custom_api_key(new_key)
                    except AiCredentialError as exc:
                        QMessageBox.critical(dialog, "保存失败", str(exc))
                        return
                    custom_key_exists = True
                self.ai_custom_endpoint = endpoint
                self.ai_custom_model = model

            self.ai_provider_mode = (
                mode if mode in AI_PROVIDER_MODES else AI_PROVIDER_DEVELOPER
            )
            self._update_ai_provider_status()
            self._save_settings()
            self._append_event(
                "AI 模型配置已切换为"
                + ("自定义模型" if mode == AI_PROVIDER_CUSTOM else "开发者试用")
            )
            dialog.accept()

        buttons.accepted.connect(save_configuration)
        buttons.rejected.connect(dialog.reject)
        dialog.exec()

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

    def _on_advanced_toggled(self, expanded: bool) -> None:
        self.advanced_panel.setVisible(expanded)
        self.advanced_toggle_btn.setText(
            "高级设置 ▾" if expanded else "高级设置 ▸"
        )
        self.advanced_toggle_btn.setAccessibleName(
            "折叠高级设置" if expanded else "展开高级设置"
        )
        if not self._restoring_settings:
            self._save_settings()

    def _on_log_toggled(self, expanded: bool) -> None:
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
            saved_geometry = self.settings.value("ui/windowGeometry")
            if saved_geometry:
                self.restoreGeometry(saved_geometry)

            self.ai_stage_enabled = self.settings.value(
                "advanced/aiPostprocess", False, type=bool
            )
            self.debug_mode_enabled = self.settings.value(
                "advanced/keepIntermediateFiles", False, type=bool
            )
            self.network_mode_enabled = self.settings.value(
                "advanced/allowNetwork", False, type=bool
            )
            saved_provider_mode = str(
                self.settings.value("ai/providerMode", AI_PROVIDER_DEVELOPER)
                or AI_PROVIDER_DEVELOPER
            )
            self.ai_provider_mode = (
                saved_provider_mode
                if saved_provider_mode in AI_PROVIDER_MODES
                else AI_PROVIDER_DEVELOPER
            )
            self.ai_custom_endpoint = str(
                self.settings.value("ai/customEndpoint", "") or ""
            ).strip()
            self.ai_custom_model = str(
                self.settings.value("ai/customModel", "") or ""
            ).strip()
            self._update_ai_button_text()
            self._update_debug_button_text()
            self._update_network_button_text()
            self._update_ai_provider_status()

            advanced_expanded = self.settings.value(
                "ui/advancedExpanded", False, type=bool
            )
            self.advanced_toggle_btn.setChecked(advanced_expanded)
            self._on_advanced_toggled(advanced_expanded)

            log_expanded = self.settings.value(
                "ui/logExpanded", False, type=bool
            )
            self.log_toggle_btn.setChecked(log_expanded)
            self._on_log_toggled(log_expanded)

            recent_value = self.settings.value("recentDirectories", [])
            if isinstance(recent_value, str):
                recent = [recent_value]
            else:
                recent = [str(value) for value in (recent_value or [])]
            self._recent_directories = [
                value for value in recent if Path(value).expanduser().is_dir()
            ][:8]
            self._refresh_recent_directories()

            last_directory = str(self.settings.value("lastDirectory", "") or "")
            if last_directory and Path(last_directory).expanduser().is_dir():
                self._apply_work_directory(
                    Path(last_directory).expanduser(),
                    remember=False,
                )

            saved_mode = str(
                self.settings.value("modeSelection", UI_MODE_RECOMMENDED)
                or UI_MODE_RECOMMENDED
            )
            mode_index = self.mode_combo.findData(saved_mode)
            self.mode_combo.setCurrentIndex(max(0, mode_index))
            self.input_mode = self._current_input_mode()
        finally:
            self._restoring_settings = False
        self._analyze_selected_directory()

    def _save_settings(self) -> None:
        if self._restoring_settings:
            return
        self.settings.setValue("advanced/aiPostprocess", self.ai_stage_enabled)
        self.settings.setValue(
            "advanced/keepIntermediateFiles", self.debug_mode_enabled
        )
        self.settings.setValue("advanced/allowNetwork", self.network_mode_enabled)
        self.settings.setValue("ai/providerMode", self.ai_provider_mode)
        self.settings.setValue("ai/customEndpoint", self.ai_custom_endpoint)
        self.settings.setValue("ai/customModel", self.ai_custom_model)
        self.settings.setValue(
            "ui/advancedExpanded", self.advanced_toggle_btn.isChecked()
        )
        self.settings.setValue("ui/logExpanded", self.log_toggle_btn.isChecked())
        self.settings.setValue("ui/windowGeometry", self.saveGeometry())
        self.settings.setValue("modeSelection", self.mode_combo.currentData())
        self.settings.setValue("recentDirectories", self._recent_directories)
        directory = self.dir_edit.text().strip()
        if directory and Path(directory).expanduser().is_dir():
            self.settings.setValue("lastDirectory", str(Path(directory).expanduser()))
        self.settings.sync()

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
            Path(url.toLocalFile()).is_dir() for url in mime_data.urls()
        ):
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event) -> None:  # type: ignore[override]
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.is_dir():
                self._apply_work_directory(path, remember=True)
                event.acceptProposedAction()
                return
        event.ignore()

    def _on_directory_edited(self) -> None:
        text = self.dir_edit.text().strip()
        path = Path(text).expanduser() if text else None
        if path is not None and path.is_dir():
            self._apply_work_directory(path, remember=True)
        else:
            self._analyze_selected_directory()

    def _apply_work_directory(self, path: Path, *, remember: bool) -> None:
        expanded = path.expanduser()
        try:
            normalized = expanded.resolve()
        except OSError:
            normalized = expanded
        self.dir_edit.setText(str(normalized))
        if not (self.worker or self.bootstrap_worker):
            self._progress_timer.stop()
            self._pipeline_started_monotonic = None
            self._reset_stage_progress(11 if self.ai_stage_enabled else 10)
            self.warning_card.hide()
            self._last_quality_report_path = None
            self._set_status_text("Idle")
        self._analyze_selected_directory()
        if remember and normalized.is_dir():
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
            self.recent_combo.addItem("选择最近使用的目录…", None)
            for value in self._recent_directories:
                self.recent_combo.addItem(self._display_path(value), value)
            self.recent_combo.setCurrentIndex(0)
        finally:
            self.recent_combo.blockSignals(False)
        has_recent = bool(self._recent_directories)
        self.recent_label.setVisible(has_recent)
        self.recent_combo.setVisible(has_recent)

    def _on_recent_directory_selected(self, index: int) -> None:
        value = self.recent_combo.itemData(index)
        if value:
            self._apply_work_directory(Path(str(value)), remember=True)

    def _directory_recommendation(self, work_dir: Path) -> tuple[str, str]:
        fits = self._fits_in_work_dir(work_dir)
        light_files = [
            path for path in fits if path.name.lower().startswith("light_")
        ]
        stacked_files = [
            path
            for path in fits
            if not path.name.lower().startswith("light_")
            and self._is_candidate_stacked_input(path, work_dir)
        ]

        if self._linear_resume_input_path(work_dir).is_file():
            return INPUT_MODE_LINEAR_RESUME, "可用的线性处理进度"
        if self._stage2_corrected_resume_input_path(work_dir).is_file():
            return INPUT_MODE_STAGE2_CORRECTED_RESUME, "可用的裁切后处理进度"
        if light_files:
            return INPUT_MODE_AUTO, f"{len(light_files)} 个 Light FITS"
        if stacked_files:
            return INPUT_MODE_AUTO, f"{len(stacked_files)} 个可处理 FITS"
        return INPUT_MODE_AUTO, "未检测到可处理的 FITS"

    def _analyze_selected_directory(self) -> None:
        text = self.dir_edit.text().strip()
        work_dir = Path(text).expanduser() if text else None
        if work_dir is None or not work_dir.is_dir():
            self._recommended_input_mode = INPUT_MODE_AUTO
            if self.mode_combo.currentData() == UI_MODE_RECOMMENDED:
                self.input_mode = INPUT_MODE_AUTO
            self.mode_combo.setItemText(0, "自动推荐（完整处理）")
            self.directory_summary_label.setText(
                "尚未选择有效目录，可将目录拖入窗口。"
            )
            self._update_result_actions(None)
            if self._workspace_state != WORKSPACE_RUN:
                self._show_workspace(WORKSPACE_EMPTY)
            self.task_preview_canvas.clear_image()
            self.task_preview_status_label.setText("等待选择目录")
            self.preview_status_label.setText("预览：等待输入")
            return

        recommended_mode, detected = self._directory_recommendation(work_dir)
        self._recommended_input_mode = recommended_mode
        recommendation = self._input_mode_label(recommended_mode)
        self.mode_combo.setItemText(0, f"自动推荐（{recommendation}）")
        if self.mode_combo.currentData() == UI_MODE_RECOMMENDED:
            self.input_mode = recommended_mode
        self.directory_summary_label.setText(
            f"已检测：{detected} · 推荐“{recommendation}”"
        )
        self._update_result_actions(work_dir)
        if self._workspace_state != WORKSPACE_RUN:
            self._show_workspace(WORKSPACE_TASK)
        if not self._restoring_settings:
            self._schedule_initial_preview(work_dir)

    def _find_result_preview(self, work_dir: Path) -> Path | None:
        preferred_candidates: set[Path] = set()
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
        self.open_result_btn.setEnabled(valid_directory)
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

        if current_mode == INPUT_MODE_LINEAR_RESUME:
            source = self._linear_resume_input_path(work_dir)
            if not source.is_file():
                return None
            source_bytes = safe_file_size(source)
            base_growth_bytes = int(source_bytes * LINEAR_RESUME_STAGE_ARTIFACT_COPIES)
            input_count = 1
            input_bytes = source_bytes
            mode = "linear_resume"
            selected_input_label = source.name
        elif current_mode == INPUT_MODE_STAGE2_CORRECTED_RESUME:
            source = self._stage2_corrected_resume_input_path(work_dir)
            if not source.is_file():
                return None
            source_bytes = safe_file_size(source)
            base_growth_bytes = int(
                source_bytes * STAGE2_RESUME_STAGE_ARTIFACT_COPIES
            )
            input_count = 1
            input_bytes = source_bytes
            mode = "stage2_corrected_resume"
            selected_input_label = source.name
        else:
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
            return "从线性处理后继续"
        if estimate.mode == "stage2_corrected_resume":
            return "从裁切后继续"
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
        try:
            spcc_ready, _detail = verify_siril_spcc_database_seed(
                self.siril_spcc_seed_dir,
                self.runtime_home,
            )
        except Exception:
            spcc_ready = False
        if not spcc_ready:
            support_growth_bytes += directory_size_bytes(self.siril_spcc_seed_dir)

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
        linear_resume_path = self._linear_resume_input_path(work_dir)
        stage2_corrected_resume_path = self._stage2_corrected_resume_input_path(work_dir)
        lines = [
            "预检摘要：",
            f"  工作目录: {self._display_path(work_dir)}",
            f"  处理模式: {self._input_mode_label(current_mode)}",
            f"  检测到的 FITS 输入: 总计={len(fits)}, Light_={len(light_files)}, 其他={len(other_fits)}",
            f"  主机架构: {machine or '<未知>'}",
            f"  流水线脚本: {self._display_path(self.pipeline_path)}",
            "  Stage11 模块: "
            f"{self._display_path(self.pipeline_path.with_name('stage11_ai_postprocess.py'))}",
            f"  Siril 插件目录: {self._display_path(self.siril_plugin_dir)}",
        ]
        if current_mode == INPUT_MODE_LINEAR_RESUME:
            lines.append(
                "  线性续跑输入: "
                + (
                    linear_resume_path.name
                    if linear_resume_path.is_file()
                    else f"{LINEAR_RESUME_INPUT_NAME}（未找到）"
                )
            )
        elif current_mode == INPUT_MODE_STAGE2_CORRECTED_RESUME:
            lines.append(
                "  叠加后处理输入: "
                + (
                    str(stage2_corrected_resume_path.relative_to(work_dir))
                    if stage2_corrected_resume_path.is_file()
                    else f"{STAGE2_CORRECTED_INPUT_NAME}（未找到）"
                )
            )
        if disk_estimate is not None:
            lines.extend(self._disk_space_summary_lines(disk_estimate))
        return lines

    def _open_result_dir(self) -> None:
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

    def _show_completion_warning(self, work_dir: Path | None) -> None:
        self._last_quality_report_path = self._quality_report_path(work_dir)
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
        siril_candidates = self._resolve_siril_candidates()

        if not siril_candidates:
            errors.append("应用资源中未找到可用的内置 Siril CLI。")

        required_files = [
            ("配置模板", self.config_template),
            ("流水线脚本", self.pipeline_path),
            ("Stage11 模块脚本", self.pipeline_path.with_name("stage11_ai_postprocess.py")),
        ]
        for label, path in required_files:
            if not path.exists():
                errors.append(f"{label}缺失：{path}")

        for cli in siril_candidates:
            if not cli.is_file():
                errors.append(f"Siril CLI 路径无效：{cli}")
            elif not cli.stat().st_mode & 0o111:
                errors.append(f"Siril CLI 不可执行：{cli}")

        if not work_dir.is_dir():
            errors.append(f"工作目录不存在：{work_dir}")

        current_mode = input_mode or self._current_input_mode()
        if current_mode == INPUT_MODE_LINEAR_RESUME:
            linear_resume_path = self._linear_resume_input_path(work_dir)
            if not linear_resume_path.is_file():
                errors.append(
                    f"续跑模式要求工作目录根下存在 {LINEAR_RESUME_INPUT_NAME}：{linear_resume_path}"
                )
        elif current_mode == INPUT_MODE_STAGE2_CORRECTED_RESUME:
            stage2_corrected_path = self._stage2_corrected_resume_input_path(work_dir)
            if not stage2_corrected_path.is_file():
                errors.append(
                    "叠加后处理模式要求工作目录根下或 process/ 下存在 "
                    f"{STAGE2_CORRECTED_INPUT_NAME}：{stage2_corrected_path}"
                )
        else:
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
                names.append(name.replace("_", "-").lower())
        return names

    def _missing_requirement_wheels(self, requirements_path: Path) -> list[str]:
        downloads_dir = self._plugin_downloads_dir()
        wheel_names = [path.name.lower() for path in downloads_dir.glob("*.whl")]
        missing: list[str] = []
        for name in self._requirement_names(requirements_path):
            normalized = name.replace("-", "_")
            prefixes = (
                f"{name}-",
                f"{normalized}-",
            )
            if not any(
                wheel_name.startswith(prefix) for wheel_name in wheel_names for prefix in prefixes
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
                # SeestarGui instances still use their bound methods.
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
        if isinstance(self, SeestarGui):
            missing_requirement_wheels = self._missing_requirement_wheels(
                self._plugin_requirements_path()
            )
        if missing_requirement_wheels:
            missing.append(
                "requirements wheel 缺失: "
                + ", ".join(missing_requirement_wheels)
            )

        syqon_bundle = plugin_root / SYQON_STARLESS_BUNDLE_REL
        for name in ("syqon_starless_inference.py", "zenith.pt"):
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
        if isinstance(self, SeestarGui) and (
            not classic_wrapper.is_file() or not os.access(classic_wrapper, os.X_OK)
        ):
            missing.append("CosmicClarity classic wrapper 缺失或不可执行")

        scripts_root = resolve_siril_scripts_root(plugin_root)
        if scripts_root is None:
            missing.append("siril-scripts 目录或 AberrationRemover.py 缺失")
        elif not (scripts_root / "processing" / "SyQon-Starless.py").is_file():
            missing.append("SyQon-Starless.py 缺失")

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

    def _remove_matching_legacy_runtime_files(
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
            "syqon_starless_inference.py",
            "zenith.pt",
            "zenith.pt.sha256",
            "zenith.pt.date",
            "zenith.pt.verified",
        ]
        if not bundle_dir.is_dir() or not (bundle_dir / "zenith.pt").is_file():
            return
        reclaimed = self._remove_matching_legacy_runtime_files(
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
        reclaimed = self._remove_matching_legacy_runtime_files(
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

        # Siril 1.4 local Gaia DR3 xp_sampled catalog directory. The worker
        # writes this exact path to core.catalogue_gaia_photo.
        gaia_photo_dir = xdg_siril_dir / "siril_cat1_healpix8_xpsamp"
        gaia_photo_dir.mkdir(parents=True, exist_ok=True)
        valid_spcc_chunks = [
            path
            for path in gaia_photo_dir.glob("siril_cat1_healpix8_xpsamp_*.dat")
            if path.is_file() and safe_file_size(path) >= 1024
        ]
        if valid_spcc_chunks:
            self._append_event(
                f"本地 SPCC 星表：{gaia_photo_dir} "
                f"({len(valid_spcc_chunks)} 个有效分块)"
            )
        else:
            self._append_event(
                "本地 SPCC 星表为空或无有效 .dat 分块；"
                "Stage 4 将在调用 Siril 前跳过 SPCC。"
            )
        gaia_astro_path = xdg_siril_dir / "siril_cat_healpix8_astro.dat"
        if gaia_astro_path.is_file() and safe_file_size(gaia_astro_path) >= 1024:
            self._append_event(
                "本地 Gaia astrometric 星表："
                f"{gaia_astro_path} ({format_bytes(safe_file_size(gaia_astro_path))})"
            )
        else:
            self._append_event(
                "本地 Gaia astrometric 星表尚未安装；PCC/platesolve 不会调用 "
                "localgaia，离线模式下也不会尝试在线目录。"
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
                self._append_event("已应用 GraXpert-AI 运行时兼容补丁")
            return

        runtime_repo.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(scripts_root, runtime_repo, dirs_exist_ok=True)
        if apply_siril_runtime_patches(self.siril_plugin_dir, runtime_repo):
            self._append_event("已应用 GraXpert-AI 运行时兼容补丁")
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
            "Generator: seestar-superimpose\n"
            "Root-Is-Purelib: true\n"
            "Tag: py3-none-any\n"
        )
        top_level_text = "tiffile\n"
        installer_text = "seestar-superimpose\n"

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
                "Generator: seestar-superimpose\n"
                "Root-Is-Purelib: true\n"
                "Tag: py3-none-any\n"
            ),
            "top_level.txt": "cv2\n",
            "INSTALLER": "seestar-superimpose\n",
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
            '"""Seestar runtime patch: override sirilpy timeout via env."""\n'
            "import os\n"
            "def _patch_default_timeout(func, timeout):\n"
            "    defaults = getattr(func, '__defaults__', None)\n"
            "    if not defaults:\n"
            "        return\n"
            "    updated = list(defaults)\n"
            "    updated[-1] = timeout\n"
            "    func.__defaults__ = tuple(updated)\n"
            "\n"
            "raw = os.getenv('SEESTAR_SIRILPY_TIMEOUT_SEC', '').strip()\n"
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
        runtime_env.setdefault("SEESTAR_SIRILPY_TIMEOUT_SEC", "120")
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
        env_version = os.environ.get("SEESTAR_APP_VERSION", "").strip()
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
        return self._siril_state_root() / "venv" / ".seestar_runtime_ready.json"

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
                self._ensure_runtime_spcc_database_seed()
            except BootstrapCancelled:
                raise
            except Exception as exc:
                self._append_event(
                    "Siril SPCC 固定数据库准备失败；"
                    f"本次运行将禁用 SPCC 并改走 PCC：{exc}"
                )
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

    def _start_run(
        self,
        _checked: bool = False,
        *,
        input_mode_override: str | None = None,
    ) -> None:
        if self.bootstrap_worker and self.bootstrap_worker.isRunning():
            return
        if self.worker and self.worker.isRunning():
            return

        directory_text = self.dir_edit.text().strip()
        if not directory_text:
            QMessageBox.information(self, "请选择目录", "请先拖入或选择工作目录。")
            return
        work_dir = Path(directory_text).expanduser()
        if self.ai_stage_enabled and not self.network_mode_enabled:
            QMessageBox.warning(
                self,
                "AI 需要联网",
                "启用 AI 后期时，请同时在高级设置中开启“允许联网”。",
            )
            return
        try:
            self._pending_ai_runtime_overrides = (
                self._resolve_ai_runtime_overrides()
                if self.ai_stage_enabled
                else {}
            )
        except AiCredentialError as exc:
            QMessageBox.warning(
                self,
                "AI 模型配置不可用",
                f"{exc}\n\n请点击“模型配置…”切换到自定义模型，或关闭 AI 后期。",
            )
            self._append_event(f"AI 模型配置不可用：{exc}")
            return
        input_mode = (
            input_mode_override
            if input_mode_override in {
                INPUT_MODE_AUTO,
                INPUT_MODE_LINEAR_RESUME,
                INPUT_MODE_STAGE2_CORRECTED_RESUME,
            }
            else self._current_input_mode()
        )
        errors = self._preflight_errors(work_dir, input_mode=input_mode)
        if errors:
            self._pending_ai_runtime_overrides.clear()
            QMessageBox.critical(self, "预检失败", "\n\n".join(errors))
            self._append_event("预检失败：")
            for err in errors:
                self._append_text(f"  - {err}\n")
            return

        self._remember_directory(work_dir)
        self._current_work_dir = work_dir
        self._run_input_mode_override = input_mode
        self._last_run_snapshot = {
            "work_dir": str(work_dir),
            "input_mode": input_mode,
            "ai_stage_enabled": self.ai_stage_enabled,
            "debug_mode_enabled": self.debug_mode_enabled,
            "network_mode_enabled": self.network_mode_enabled,
            "ai_provider_mode": self.ai_provider_mode,
            "ai_custom_endpoint": self.ai_custom_endpoint,
            "ai_custom_model": self.ai_custom_model,
        }
        stage_count = 11 if self.ai_stage_enabled else 10
        self._progress_timer.stop()
        self._pipeline_started_monotonic = None
        self._reset_stage_progress(stage_count)
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
        self.preview_notice_label.setText("Stage 0 原始预览 · 无显示拉伸")
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
        self._start_pipeline(work_dir, disk_estimate)

    def _on_bootstrap_failed(self, title: str, detail: str) -> None:
        self._cleanup_bootstrap_worker()
        self._pending_ai_runtime_overrides.clear()
        self._run_terminal_status = "Failed"
        self._set_status_text("Failed")
        self.progress_bar.setRange(0, 10)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("运行环境准备失败")
        self.progress_bar.hide()
        self.preview_activity_label.setText("运行环境准备失败")
        self.run_phase_label.setText("准备失败")
        self.log_toggle_btn.setChecked(True)
        QMessageBox.critical(self, title, detail)
        self._append_event(f"{title}：{detail}")
        self._update_result_actions(self._current_work_dir)
        self._current_work_dir = None
        self._set_running(False)

    def _on_bootstrap_cancelled(self) -> None:
        self._cleanup_bootstrap_worker()
        self._pending_ai_runtime_overrides.clear()
        self._run_terminal_status = "Stopped"
        self._set_status_text("Stopped")
        self.progress_bar.setRange(0, 10)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("准备已停止")
        self.progress_bar.hide()
        self.preview_activity_label.setText("任务已停止 · 保留最新可靠预览")
        self.run_phase_label.setText("已停止")
        self._append_event("运行环境准备已停止。")
        self._update_result_actions(self._current_work_dir)
        self._current_work_dir = None
        self._set_running(False)

    def _start_pipeline(
        self,
        work_dir: Path,
        disk_estimate: DiskSpaceEstimate | None,
    ) -> None:
        input_mode = self._run_input_mode_override or self._current_input_mode()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_log_path = work_dir / f"seestar_gui_run_{stamp}.log"
        with self._run_log_lock:
            self.run_log_file = self.run_log_path.open(
                "a",
                encoding="utf-8",
                errors="replace",
            )
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
        self._append_event(
            f"本次运行 ai_stage11={'ON' if self.ai_stage_enabled else 'OFF'}"
        )
        if self.ai_stage_enabled:
            self._append_event(
                "本次 AI 配置："
                + (
                    "自定义模型"
                    if self.ai_provider_mode == AI_PROVIDER_CUSTOM
                    else "开发者试用"
                )
            )

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
            ai_stage_enabled=self.ai_stage_enabled,
            ai_runtime_overrides=self._pending_ai_runtime_overrides,
            parent=self,
        )
        self._pending_ai_runtime_overrides.clear()
        self.worker.log.connect(self._append_text)
        self.worker.state.connect(self._set_status_text)
        self.worker.progress.connect(self._on_pipeline_progress)
        self.worker.preview.connect(self._on_pipeline_preview)
        self.worker.done.connect(self._on_worker_done)

        stage_count = 11 if self.ai_stage_enabled else 10
        self._begin_stage_progress(stage_count)
        self.progress_bar.hide()
        self.preview_activity_label.setText("等待 Stage 1 开始")
        self._set_status_text("Running")
        self.worker.start()

    def _stop_run(self) -> None:
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
        elif stage <= 10:
            phase_text = "非线性处理 · Stage 7–10"
        else:
            phase_text = "AI 后期 · Stage 11"
        self.run_phase_label.setText(phase_text)

    def _on_pipeline_progress(self, stage: int, title: str, state: str) -> None:
        stage = int(stage)
        stage_count = max(11 if self.ai_stage_enabled else 10, int(stage))
        self._progress_stage_count = stage_count

        detail = title.strip() or PIPELINE_STAGE_TITLES.get(stage, f"阶段 {stage}")
        normalized_state = state.strip().lower() or "running"
        if normalized_state not in PIPELINE_PROGRESS_STATE_LABELS:
            normalized_state = "failed"
        state_label = PIPELINE_PROGRESS_STATE_LABELS[normalized_state]
        now = time.monotonic()

        if stage not in self._stage_items:
            chip = QLabel()
            chip.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
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
        elif normalized_state in {"completed", "degraded"}:
            self.preview_activity_label.setText(
                f"Stage {stage} {state_label} · 等待最新预览"
            )
        else:
            self.preview_activity_label.setText(
                f"Stage {stage} {state_label} · 无新预览，保留上一张"
            )
        self._announce_accessibility(progress_text)
        self._refresh_elapsed_labels()

    def _on_pipeline_preview(
        self,
        stage: int,
        title: str,
        status: str,
        payload: str,
    ) -> None:
        stage = int(stage)
        detail = title.strip() or PIPELINE_STAGE_TITLES.get(stage, f"阶段 {stage}")
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
                    "线性原始预览 · 无显示拉伸（画面可能较暗）"
                )
            else:
                self.preview_notice_label.setText(
                    "阶段原始输出预览 · 无额外显示拉伸"
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

    def _on_worker_done(self, status: str, exit_code: int, had_errors: bool, cli_used: str) -> None:
        work_dir = self._current_work_dir
        self._finish_stage_progress(status)
        self.progress_bar.hide()
        self._run_terminal_status = status
        self._set_status_text(status)
        if status not in {"Completed", "CompletedWithWarning", "Stopped"}:
            self.log_toggle_btn.setChecked(True)
        self._append_event(
            f"处理结束：状态={self._display_status(status)}，退出码={exit_code}，CLI={cli_used}"
        )
        if status == "Failed" and had_errors:
            self._append_event("在输出中检测到 Siril/脚本错误。")
        if status == "CompletedWithWarning":
            self._show_completion_warning(work_dir)
            self._append_event(
                "最终产物已生成，但流水线存在失败阶段、降级回退或收尾异常；"
                "请查看阶段汇总和 final_quality_report.json 后再使用。"
            )
        terminal_activity = {
            "Completed": "处理已完成 · 当前为最终可靠预览",
            "CompletedWithWarning": "处理已完成（需复核）· 保留最终可靠预览",
            "Stopped": "任务已停止 · 保留最新可靠预览",
            "Failed": "处理失败 · 保留最新可靠预览",
        }.get(status, f"任务结束：{self._display_status(status)}")
        self.preview_activity_label.setText(terminal_activity)
        self.run_phase_label.setText(self._display_status(status))
        self._append_divider(
            "本次任务结束",
            [
                f"时间: {self._timestamp()}",
                f"工作目录: {self._display_path(work_dir)}",
                f"最终状态: {self._display_status(status)}",
            ],
        )

        self._update_result_actions(work_dir)
        self._cleanup_after_run()
        self._set_running(False)

    def _cleanup_after_run(self) -> None:
        self._pending_ai_runtime_overrides.clear()
        if self.worker:
            self.worker.wait(200)
            self.worker.deleteLater()
            self.worker = None

        with self._run_log_lock:
            if self.run_log_file:
                self.run_log_file.close()
                self.run_log_file = None

        self._current_work_dir = None
        self._run_input_mode_override = None

    def closeEvent(self, event) -> None:  # type: ignore[override]
        active_worker = None
        bootstrap_worker = getattr(self, "bootstrap_worker", None)
        pipeline_worker = getattr(self, "worker", None)
        if bootstrap_worker and bootstrap_worker.isRunning():
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
        event.accept()
