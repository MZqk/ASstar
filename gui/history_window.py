"""Dedicated history browser window for the Starun desktop UI."""

from __future__ import annotations

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import QDialog, QVBoxLayout, QWidget


class HistoryWindow(QDialog):
    """Non-modal singleton host for the application's history browser."""

    aboutToClose = Signal()

    def __init__(self, content: QWidget, parent=None) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self.setObjectName("historyWindow")
        self.setWindowTitle("Starun 历史记录")
        self.setModal(False)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
        self.setMinimumSize(760, 500)
        self.resize(940, 620)
        self.setSizeGripEnabled(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.addWidget(content, 1)

    def show_and_activate(self, *, initial_focus: QWidget | None = None) -> None:
        """Present the existing window without resetting window-scoped state."""

        was_visible = self.isVisible()
        if self.isMinimized():
            self.showNormal()
        self.show()
        self.raise_()
        self.activateWindow()
        if not was_visible and initial_focus is not None:
            initial_focus.setFocus(Qt.FocusReason.OtherFocusReason)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.aboutToClose.emit()
        super().closeEvent(event)


__all__ = ["HistoryWindow"]
