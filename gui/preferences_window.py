"""Dedicated application preferences window for the Starun desktop UI."""

from __future__ import annotations

from collections.abc import Mapping

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

try:
    from .ui_platform import (
        constrain_window_to_visible_screens,
        place_window_relative_to,
    )
except ImportError:  # Support direct execution from the gui directory.
    from ui_platform import (  # type: ignore[no-redef]
        constrain_window_to_visible_screens,
        place_window_relative_to,
    )


OUTPUT_FORMAT_OPTIONS = (
    ("tif", "TIFF"),
    ("png", "PNG"),
    ("fit", "FITS"),
)
VALID_COMPUTE_MODES = frozenset({"auto", "cpu"})


class PreferencesWindow(QDialog):
    """Non-modal singleton window for durable, application-wide defaults."""

    preferencesChanged = Signal(object)
    paneChanged = Signal(int)

    def __init__(self, parent=None, *, initial_pane: int = 0) -> None:
        super().__init__(parent, Qt.WindowType.Dialog)
        self._syncing = False
        self._positioned = False
        self._editable_controls: list[QWidget] = []

        self.setObjectName("preferencesWindow")
        self.setWindowTitle("Starun 设置")
        self.setModal(False)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        self.tabs = QTabWidget()
        self.tabs.setAccessibleName("Starun 设置分类")
        self.tabs.addTab(self._build_general_pane(), "通用")
        self.tabs.addTab(self._build_output_pane(), "输出")
        self.tabs.addTab(self._build_performance_pane(), "性能")
        self.tabs.setCurrentIndex(
            max(0, min(int(initial_pane), self.tabs.count() - 1))
        )
        self.tabs.currentChanged.connect(self.paneChanged.emit)
        layout.addWidget(self.tabs, 1)

        self.status_label = QLabel(
            "更改会自动保存，并用于当前尚未启动的任务和后续任务。"
        )
        self.status_label.setWordWrap(True)
        self.status_label.setProperty("tone", "muted")
        self.status_label.setAccessibleName("设置保存状态")
        layout.addWidget(self.status_label)

        self.allow_network_check.toggled.connect(self._emit_preferences)
        self.keep_intermediate_check.toggled.connect(self._emit_preferences)
        self.checkpoint_mode_check.toggled.connect(self._emit_preferences)
        self.review_combo.currentIndexChanged.connect(self._emit_preferences)
        self.compute_combo.currentIndexChanged.connect(self._emit_preferences)
        for key, checkbox in self.output_format_checks.items():
            checkbox.toggled.connect(
                lambda checked, output_key=key: self._on_output_toggled(
                    output_key,
                    checked,
                )
            )
        # The three compact panes have one intentional size. This keeps the
        # settings utility from presenting an unhelpful maximize/zoom affordance.
        self.setFixedSize(620, 420)

    def show_and_activate(self, *, reference: QWidget) -> None:
        """Present the singleton without restoring it automatically at launch."""

        if not self._positioned:
            place_window_relative_to(
                self,
                reference,
                preferred_size=(620, 420),
            )
            self._positioned = True
        else:
            constrain_window_to_visible_screens(self, reference=reference)
        if self.isMinimized():
            self.showNormal()
        self.show()
        self.raise_()
        self.activateWindow()

    def _build_general_pane(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 16, 14, 14)
        layout.setSpacing(8)

        self.allow_network_check = QCheckBox("默认允许联网")
        self.allow_network_check.setAccessibleDescription(
            "允许在线 Gaia 星表查询和已授权资源补齐；关闭后使用严格离线模式。"
        )
        network_note = QLabel(
            "用于在线 Gaia 校准和已授权资源补齐；关闭后按严格离线模式运行。"
        )
        network_note.setWordWrap(True)
        network_note.setProperty("tone", "muted")
        layout.addWidget(self.allow_network_check)
        layout.addWidget(network_note)

        layout.addSpacing(10)
        self.keep_intermediate_check = QCheckBox("默认保留中间文件")
        self.keep_intermediate_check.setAccessibleDescription(
            "保留阶段 FITS 和诊断报告，便于复核，但会增加磁盘占用。"
        )
        intermediate_note = QLabel(
            "保留阶段 FITS 与诊断报告，便于质量复核，但会显著增加磁盘占用。"
        )
        intermediate_note.setWordWrap(True)
        intermediate_note.setProperty("tone", "muted")
        layout.addWidget(self.keep_intermediate_check)
        layout.addWidget(intermediate_note)

        layout.addSpacing(10)
        self.checkpoint_mode_check = QCheckBox("完成后仅保留正式断点")
        self.checkpoint_mode_check.setAccessibleDescription(
            "成功验证最终交付后，删除非关键阶段 FITS，仅保留正式断点和轻量诊断。"
        )
        checkpoint_note = QLabel(
            "失败或交付校验不通过时不会收敛，现场文件会完整保留。"
        )
        checkpoint_note.setWordWrap(True)
        checkpoint_note.setProperty("tone", "muted")
        layout.addWidget(self.checkpoint_mode_check)
        layout.addWidget(checkpoint_note)
        layout.addStretch(1)

        self._editable_controls.extend(
            (
                self.allow_network_check,
                self.keep_intermediate_check,
                self.checkpoint_mode_check,
            )
        )
        return page

    def _build_output_pane(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 16, 14, 14)
        layout.setSpacing(12)

        form = QFormLayout()
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(12)

        output_row = QWidget()
        output_layout = QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.setSpacing(12)
        self.output_format_checks: dict[str, QCheckBox] = {}
        for key, title in OUTPUT_FORMAT_OPTIONS:
            checkbox = QCheckBox(title)
            checkbox.setAccessibleName(f"默认输出 {title}")
            self.output_format_checks[key] = checkbox
            self._editable_controls.append(checkbox)
            output_layout.addWidget(checkbox)
        output_layout.addStretch(1)
        form.addRow("默认输出格式", output_row)

        self.review_combo = QComboBox()
        self.review_combo.setAccessibleName("默认输出用途")
        self.review_combo.addItem("正式结果", False)
        self.review_combo.addItem("仅生成待复核结果", True)
        form.addRow("默认输出用途", self.review_combo)
        self._editable_controls.append(self.review_combo)

        layout.addLayout(form)
        output_note = QLabel(
            "这些是通用默认值；每次开始处理前仍可在当前任务中确认。"
        )
        output_note.setWordWrap(True)
        output_note.setProperty("tone", "muted")
        layout.addWidget(output_note)
        layout.addStretch(1)
        return page

    def _build_performance_pane(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 16, 14, 14)
        layout.setSpacing(12)

        form = QFormLayout()
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(12)
        self.compute_combo = QComboBox()
        self.compute_combo.setAccessibleName("默认计算设备")
        self.compute_combo.addItem("自动加速", "auto")
        self.compute_combo.addItem("CPU 兼容模式", "cpu")
        form.addRow("默认计算设备", self.compute_combo)
        self._editable_controls.append(self.compute_combo)
        layout.addLayout(form)

        compute_note = QLabel(
            "自动加速允许运行时选择可用设备；兼容模式会显式使用 CPU。"
        )
        compute_note.setWordWrap(True)
        compute_note.setProperty("tone", "muted")
        layout.addWidget(compute_note)
        layout.addStretch(1)
        return page

    def preferences(self) -> dict[str, object]:
        """Return a normalized snapshot of the visible durable preferences."""

        formats = tuple(
            key
            for key, checkbox in self.output_format_checks.items()
            if checkbox.isChecked()
        )
        compute_mode = str(self.compute_combo.currentData() or "auto")
        if compute_mode not in VALID_COMPUTE_MODES:
            compute_mode = "auto"
        return {
            "allow_network": self.allow_network_check.isChecked(),
            "keep_intermediate": self.keep_intermediate_check.isChecked(),
            "checkpoint_mode": self.checkpoint_mode_check.isChecked(),
            "output_formats": formats or ("tif",),
            "review_only": bool(self.review_combo.currentData()),
            "compute_mode": compute_mode,
        }

    def set_preferences(self, preferences: Mapping[str, object]) -> None:
        """Synchronize controls without treating the update as a user edit."""

        formats = {
            str(value)
            for value in preferences.get("output_formats", ("tif",))
            if str(value) in self.output_format_checks
        }
        if not formats:
            formats = {"tif"}
        compute_mode = str(preferences.get("compute_mode", "auto"))
        if compute_mode not in VALID_COMPUTE_MODES:
            compute_mode = "auto"

        self._syncing = True
        try:
            self.allow_network_check.setChecked(
                bool(preferences.get("allow_network", True))
            )
            self.keep_intermediate_check.setChecked(
                bool(preferences.get("keep_intermediate", False))
            )
            self.checkpoint_mode_check.setChecked(
                bool(preferences.get("checkpoint_mode", False))
            )
            for key, checkbox in self.output_format_checks.items():
                checkbox.setChecked(key in formats)
            review_index = self.review_combo.findData(
                bool(preferences.get("review_only", False))
            )
            self.review_combo.setCurrentIndex(max(0, review_index))
            compute_index = self.compute_combo.findData(compute_mode)
            self.compute_combo.setCurrentIndex(max(0, compute_index))
        finally:
            self._syncing = False

    def set_editable(self, editable: bool, *, reason: str = "") -> None:
        """Keep the window inspectable while preventing mid-run mutations."""

        for control in self._editable_controls:
            control.setEnabled(bool(editable))
        if editable:
            self.status_label.setText(
                "更改会自动保存，并用于当前尚未启动的任务和后续任务。"
            )
        else:
            self.status_label.setText(
                reason or "当前批次已冻结；处理结束后可修改这些默认值。"
            )

    def _on_output_toggled(self, key: str, checked: bool) -> None:
        if self._syncing:
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
            self.status_label.setText("至少保留一种默认输出格式。")
            return
        self._emit_preferences()

    def _emit_preferences(self, *_args) -> None:
        if not self._syncing:
            self.preferencesChanged.emit(self.preferences())


__all__ = ["PreferencesWindow"]
