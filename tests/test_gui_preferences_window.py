#!/usr/bin/env python3
"""Offscreen checks for the dedicated application preferences window."""

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

from gui.main_window import StarunGui  # noqa: E402


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance


def _window(root: Path) -> StarunGui:
    settings = QSettings(
        str(root / "settings.ini"),
        QSettings.Format.IniFormat,
    )
    return StarunGui(
        resources_override=Path(__file__).resolve().parents[1] / "resources",
        runtime_home_override=root / "runtime",
        settings_override=settings,
        history_path_override=root / "history.json",
    )


def test_preferences_action_opens_singleton_and_syncs_defaults(app) -> None:
    with tempfile.TemporaryDirectory() as td:
        window = _window(Path(td))
        window.show()
        try:
            assert window.preferences_action.isEnabled() is True
            assert window.toolbar_settings_btn.text() == "任务选项"

            window.preferences_action.trigger()
            app.processEvents()
            preferences = window.preferences_window
            assert preferences.isVisible() is True
            assert preferences.windowTitle() == "Starun 设置"

            window.preferences_action.trigger()
            app.processEvents()
            assert window.preferences_window is preferences

            preferences.allow_network_check.setChecked(False)
            preferences.keep_intermediate_check.setChecked(True)
            preferences.output_format_checks["png"].setChecked(False)
            preferences.output_format_checks["fit"].setChecked(False)
            preferences.review_combo.setCurrentIndex(
                preferences.review_combo.findData(True)
            )
            preferences.compute_combo.setCurrentIndex(
                preferences.compute_combo.findData("cpu")
            )
            app.processEvents()

            assert window.network_mode_enabled is False
            assert window.debug_mode_enabled is True
            assert window.output_formats == ("tif",)
            assert window.review_only is True
            assert window.compute_mode == "cpu"
            assert window.processing_parameters["general"] == {
                "output_formats": ["tif"],
                "review_only": True,
                "compute_mode": "cpu",
                "auto_tune_enabled": True,
                "max_retries": 2,
                "retry_delay": 1.0,
                "review_bundle_enabled": True,
                "managed_output_enabled": True,
                "checkpoint_mode": False,
            }
            assert window.settings.value(
                "advanced/allowNetwork", type=bool
            ) is False
            assert window.settings.value(
                "advanced/keepIntermediateFiles", type=bool
            ) is True
            assert window.settings.value("processing/outputFormats") == ["tif"]
        finally:
            preferences = getattr(window, "preferences_window", None)
            if preferences is not None:
                preferences.close()
            window.close()


def test_running_batch_keeps_preferences_visible_but_read_only(app) -> None:
    with tempfile.TemporaryDirectory() as td:
        window = _window(Path(td))
        window.show()
        try:
            window._show_preferences()
            preferences = window.preferences_window
            original_network_mode = window.network_mode_enabled

            window._set_running(True)
            app.processEvents()

            assert window.preferences_action.isEnabled() is True
            assert preferences.tabs.isEnabled() is True
            assert preferences.allow_network_check.isEnabled() is False
            assert preferences.keep_intermediate_check.isEnabled() is False
            assert "已冻结" in preferences.status_label.text()

            preferences.allow_network_check.setChecked(not original_network_mode)
            app.processEvents()
            assert window.network_mode_enabled is original_network_mode
            assert preferences.allow_network_check.isChecked() is original_network_mode

            window._set_running(False)
            app.processEvents()
            assert preferences.allow_network_check.isEnabled() is True
        finally:
            preferences = getattr(window, "preferences_window", None)
            if preferences is not None:
                preferences.close()
            window.close()


def test_preferences_restore_last_pane_and_require_an_output_format(app) -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        window = _window(root)
        window.show()
        try:
            window._show_preferences()
            preferences = window.preferences_window
            preferences.tabs.setCurrentIndex(2)
            for checkbox in preferences.output_format_checks.values():
                checkbox.setChecked(False)
            app.processEvents()

            assert window.settings.value("ui/preferencesPane", type=int) == 2
            assert any(
                checkbox.isChecked()
                for checkbox in preferences.output_format_checks.values()
            )
        finally:
            preferences = getattr(window, "preferences_window", None)
            if preferences is not None:
                preferences.close()
            window.close()

        restored = _window(root)
        restored.show()
        try:
            restored._show_preferences()
            assert restored.preferences_window.tabs.currentIndex() == 2
        finally:
            preferences = getattr(restored, "preferences_window", None)
            if preferences is not None:
                preferences.close()
            restored.close()
