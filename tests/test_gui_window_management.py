#!/usr/bin/env python3
"""Offscreen regression checks for desktop window lifecycle policy."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PySide6 = pytest.importorskip("PySide6")
if not getattr(PySide6, "__version__", None):
    pytest.skip("requires real PySide6 modules", allow_module_level=True)

from PySide6.QtCore import QRect, QSettings, QSize, Qt  # noqa: E402
from PySide6.QtGui import QGuiApplication  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from gui.main_window import StarunGui  # noqa: E402
from gui.ui_platform import (  # noqa: E402
    MACOS_PROFILE,
    fitted_window_geometry,
)


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance


def _window(root: Path, *, settings: QSettings | None = None) -> StarunGui:
    return StarunGui(
        resources_override=Path(__file__).resolve().parents[1] / "resources",
        runtime_home_override=root / "runtime",
        settings_override=settings
        or QSettings(
            str(root / "settings.ini"),
            QSettings.Format.IniFormat,
        ),
        history_path_override=root / "history.json",
    )


def test_geometry_fits_removed_and_rotated_displays() -> None:
    available = QRect(0, 24, 1440, 876)
    restored = fitted_window_geometry(
        QRect(3200, -900, 1100, 700),
        available,
        minimum_size=QSize(500, 400),
    )
    working = available.adjusted(12, 12, -12, -12)

    assert restored.size() == QSize(1100, 700)
    assert working.contains(restored.topLeft())
    assert working.contains(restored.bottomRight())

    rotated = fitted_window_geometry(
        QRect(40, 80, 1200, 900),
        QRect(0, 0, 900, 1400),
        minimum_size=QSize(500, 400),
    )
    assert rotated.size() == QSize(876, 900)
    assert rotated.x() == 12


def test_main_toolbar_and_utility_windows_keep_standard_chrome(app) -> None:
    with tempfile.TemporaryDirectory() as td:
        window = _window(Path(td))
        window.show()
        app.processEvents()
        try:
            frameless = Qt.WindowType.FramelessWindowHint
            assert not bool(window.windowFlags() & frameless)
            assert not bool(window.history_window.windowFlags() & frameless)
            assert window.window_drag_region.property("windowDragRegion") is True
            if window.platform_profile == MACOS_PROFILE:
                assert window.window_drag_region.testAttribute(
                    Qt.WidgetAttribute.WA_TransparentForMouseEvents
                )

            window._show_preferences()
            preferences = window.preferences_window
            assert not bool(preferences.windowFlags() & frameless)
            assert preferences.testAttribute(
                Qt.WidgetAttribute.WA_QuitOnClose
            ) is False
            assert preferences.minimumSize() == QSize(620, 420)
            assert preferences.maximumSize() == QSize(620, 420)
        finally:
            preferences = getattr(window, "preferences_window", None)
            if preferences is not None:
                preferences.close()
            window.close()


def test_launch_restores_main_state_without_reopening_utilities(app) -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        settings = QSettings(
            str(root / "settings.ini"),
            QSettings.Format.IniFormat,
        )
        settings.setValue(
            "ui/windowNormalGeometry",
            QRect(9000, 9000, 1100, 740),
        )
        settings.setValue("ui/windowMaximized", True)
        settings.sync()

        window = _window(root, settings=settings)
        try:
            assert window.isMaximized() is True
            assert window.history_window.isVisible() is False
            assert not hasattr(window, "preferences_window")

            normal_geometry = window.normalGeometry()
            available = QGuiApplication.primaryScreen().availableGeometry()
            assert normal_geometry.intersects(available)
        finally:
            window.close()
