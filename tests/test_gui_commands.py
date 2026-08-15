#!/usr/bin/env python3
"""Offscreen checks for shared desktop commands, menus, and toolbar entry points."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PySide6 = pytest.importorskip("PySide6")
if not getattr(PySide6, "__version__", None):
    pytest.skip("requires real PySide6 modules", allow_module_level=True)

from PySide6.QtCore import QSettings, Qt  # noqa: E402
from PySide6.QtGui import QAction, QKeySequence  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from gui.main_window import (  # noqa: E402
    StarunGui,
    WORKSPACE_EMPTY,
    WORKSPACE_RUN,
)
from gui.ui_platform import MACOS_PROFILE, standard_shortcuts  # noqa: E402


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance


def _window(root: Path) -> StarunGui:
    return StarunGui(
        resources_override=Path(__file__).resolve().parents[1] / "resources",
        runtime_home_override=root / "runtime",
        settings_override=QSettings(
            str(root / "settings.ini"),
            QSettings.Format.IniFormat,
        ),
        history_path_override=root / "history.json",
    )


def test_macos_custom_shortcuts_map_to_qt_command_modifier(app) -> None:
    shortcuts = standard_shortcuts(MACOS_PROFILE)
    portable = QKeySequence.SequenceFormat.PortableText

    assert shortcuts["open_folder"].toString(portable) == "Ctrl+Shift+O"
    assert shortcuts["history"].toString(portable) == "Ctrl+Y"
    assert shortcuts["minimize"].toString(portable) == "Ctrl+M"
    assert shortcuts["start"].toString(portable) == "Ctrl+Return"
    assert shortcuts["toggle_inspector"].toString(portable) == "Ctrl+Alt+I"
    assert shortcuts["toggle_sidebar"].toString(portable) == "Meta+Ctrl+S"


def test_toolbar_and_menus_share_the_same_commands(app) -> None:
    with tempfile.TemporaryDirectory() as td:
        window = _window(Path(td))
        window.show()
        app.processEvents()
        try:
            assert [action.text() for action in window.menuBar().actions()] == [
                "文件",
                "编辑",
                "显示",
                "处理",
                "窗口",
                "帮助",
            ]
            assert window.open_file_action in window.file_menu.actions()
            assert window.start_action in window.process_menu.actions()
            assert window.toggle_log_action in window.view_menu.actions()
            assert window.toggle_sidebar_action in window.view_menu.actions()
            assert window.toggle_inspector_action in window.view_menu.actions()
            assert window.minimize_action in window.window_menu.actions()
            assert window.main_window_action in window.window_menu.actions()
            assert window.history_action in window.window_menu.actions()
            assert window.history_action not in window.view_menu.actions()
            assert window.preferences_action.menuRole() == QAction.MenuRole.PreferencesRole
            assert window.quit_action.menuRole() == QAction.MenuRole.QuitRole
            assert window.about_action.menuRole() == QAction.MenuRole.AboutRole

            assert (
                window.toolbar_directory_btn.property("commandAction")
                == window.open_file_action.objectName()
            )
            assert (
                window.start_btn.property("commandAction")
                == window.start_action.objectName()
            )

            triggered: list[bool] = []
            window.history_action.triggered.connect(
                lambda *_args: triggered.append(True)
            )
            window.history_btn.click()
            app.processEvents()
            assert triggered == [True]
            assert window._workspace_state == WORKSPACE_EMPTY
            assert window.history_window.isVisible() is True
            assert window.history_action.isEnabled() is True
            assert window.history_btn.isEnabled() is True
            assert (
                window.history_action.shortcutContext()
                == Qt.ShortcutContext.ApplicationShortcut
            )

            window.history_window.activateWindow()
            app.processEvents()
            window.close_window_action.trigger()
            app.processEvents()
            assert window.history_window.isVisible() is False
            assert window.isVisible() is True
        finally:
            window.close()


def test_view_and_edit_commands_follow_current_ui_state(app) -> None:
    with tempfile.TemporaryDirectory() as td:
        window = _window(Path(td))
        window.show()
        app.processEvents()
        try:
            window.toggle_log_action.setChecked(True)
            app.processEvents()
            assert window.log_toggle_btn.isChecked() is True
            assert window.log_container.isVisible() is True

            window.log_toggle_btn.click()
            app.processEvents()
            assert window.toggle_log_action.isChecked() is False
            assert window.log_container.isVisible() is False

            window._show_workspace(WORKSPACE_RUN)
            assert window.fit_preview_action.isEnabled() is True
            assert window.zoom_in_btn.isEnabled() is True

            window.full_screen_action.setChecked(True)
            app.processEvents()
            assert window.isFullScreen() is True
            assert window.full_screen_action.text() == "退出全屏幕"
            window.full_screen_action.setChecked(False)
            app.processEvents()
            assert window.isFullScreen() is False

            window._show_history()
            editor = window.history_search_edit
            editor.setText("M42")
            editor.setFocus()
            editor.selectAll()
            app.processEvents()
            window._update_edit_actions()
            assert window.copy_action.isEnabled() is True
            assert window.cut_action.isEnabled() is True
            window.cut_action.trigger()
            assert editor.text() == ""

            window._leave_history()
            window.toggle_log_action.setChecked(True)
            window.log_view.setPlainText("read-only log")
            window.log_view.selectAll()
            window.log_view.setFocus()
            app.processEvents()
            window._update_edit_actions()
            assert window.copy_action.isEnabled() is True
            assert window.cut_action.isEnabled() is False
            assert window.paste_action.isEnabled() is False
        finally:
            window.close()


def test_recent_inputs_menu_tracks_and_clears_the_shared_recent_list(app) -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        window = _window(root)
        window.show()
        app.processEvents()
        try:
            recent_input = root / "recent-light"
            recent_input.mkdir()
            window._recent_directories = [str(recent_input)]
            window._refresh_recent_directories()

            actions = window.open_recent_menu.actions()
            assert actions[0].data() == str(recent_input)
            assert actions[-1].text() == "清除菜单"
            assert window.open_recent_menu.menuAction().isEnabled() is True

            actions[-1].trigger()
            assert window._recent_directories == []
            assert window.open_recent_menu.menuAction().isEnabled() is False
        finally:
            window.close()
