#!/usr/bin/env python3
"""Offscreen checks for the task detail inspector desktop layout."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PySide6 = pytest.importorskip("PySide6")
if not getattr(PySide6, "__version__", None):
    pytest.skip("requires real PySide6 modules", allow_module_level=True)

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from gui.main_window import (  # noqa: E402
    StarunGui,
    WORKSPACE_RUN,
    WORKSPACE_TASK,
)


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


def test_run_workspace_uses_sidebar_detail_and_tabbed_inspector(app) -> None:
    with tempfile.TemporaryDirectory() as td:
        window = _window(Path(td))
        window._show_workspace(WORKSPACE_RUN)
        window.show()
        app.processEvents()
        try:
            assert window.run_splitter.widget(0) is window.run_sidebar
            assert window.run_splitter.widget(1) is window.run_detail
            assert window.run_splitter.widget(2) is window.run_inspector
            assert [
                window.inspector_tabs.tabText(index)
                for index in range(window.inspector_tabs.count())
            ] == ["阶段", "任务"]
            assert window.run_inspector.isAncestorOf(window.run_options_label)
            assert not window.run_sidebar.isAncestorOf(window.run_options_label)

            window.toggle_sidebar_action.setChecked(False)
            app.processEvents()
            assert window.run_sidebar.isVisible() is False
            assert window.sidebar_toggle_btn.isChecked() is False

            window.toggle_sidebar_action.setChecked(True)
            window.toggle_inspector_action.setChecked(False)
            app.processEvents()
            assert window.run_sidebar.isVisible() is True
            assert window.run_inspector.isVisible() is False
            assert window.inspector_toggle_btn.isChecked() is False

            window.toggle_inspector_action.setChecked(True)
            window.resize(1000, 700)
            app.processEvents()
            assert window.run_sidebar.isVisible() is True
            assert window.run_inspector.isVisible() is True
        finally:
            window.close()


def test_splitter_widths_visibility_and_inspector_tab_restore(app) -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        window = _window(root)
        window._show_workspace(WORKSPACE_TASK)
        window.show()
        app.processEvents()
        window.advanced_toggle_btn.setChecked(True)
        window._set_processing_parameters_expanded(True)
        app.processEvents()
        assert window.task_section_splitter.widget(0) is window.task_top_region
        assert (
            window.task_section_splitter.widget(1)
            is window.processing_params_scroll
        )
        window.task_splitter.setSizes([640, 420])
        window.task_section_splitter.setSizes([260, 390])
        app.processEvents()
        saved_task_sizes = window.task_splitter.sizes()
        saved_task_section_sizes = window.task_section_splitter.sizes()
        window._show_workspace(WORKSPACE_RUN)
        window.run_splitter.setSizes([280, 620, 330])
        app.processEvents()
        saved_sidebar_width = window.run_splitter.sizes()[0]
        saved_inspector_width = window.run_splitter.sizes()[2]
        window.inspector_tabs.setCurrentIndex(1)
        window.toggle_sidebar_action.setChecked(False)
        window.close()

        restored = _window(root)
        restored._show_workspace(WORKSPACE_TASK)
        restored.show()
        app.processEvents()
        try:
            restored_task_sizes = restored.task_splitter.sizes()
            assert (
                abs(
                    restored_task_sizes[0] / restored_task_sizes[1]
                    - saved_task_sizes[0] / saved_task_sizes[1]
                )
                < 0.02
            )
            restored_task_section_sizes = (
                restored.task_section_splitter.sizes()
            )
            assert not restored.processing_params_scroll.isHidden()
            assert (
                abs(
                    restored_task_section_sizes[0]
                    / restored_task_section_sizes[1]
                    - saved_task_section_sizes[0]
                    / saved_task_section_sizes[1]
                )
                < 0.02
            )
            restored._show_workspace(WORKSPACE_RUN)
            app.processEvents()
            assert restored.toggle_sidebar_action.isChecked() is False
            assert restored.run_sidebar.isVisible() is False
            assert restored.toggle_inspector_action.isChecked() is True
            assert restored.inspector_tabs.currentIndex() == 1
            assert restored._run_sidebar_width == saved_sidebar_width
            assert restored._run_inspector_width == saved_inspector_width

            restored.toggle_sidebar_action.setChecked(True)
            app.processEvents()
            assert restored.run_sidebar.isVisible() is True
            assert restored.run_splitter.sizes()[0] == saved_sidebar_width
        finally:
            restored.close()


def test_splitter_setting_parser_rejects_invalid_geometry() -> None:
    assert StarunGui._splitter_sizes_setting([620, 420], 2) == [620, 420]
    assert StarunGui._splitter_sizes_setting("620, 420", 2) == [620, 420]
    assert StarunGui._splitter_sizes_setting([620, 0], 2) is None
    assert StarunGui._splitter_sizes_setting("invalid", 2) is None
