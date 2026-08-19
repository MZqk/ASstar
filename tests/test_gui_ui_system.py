from __future__ import annotations

import pytest

from gui.ui_platform import current_platform_profile

try:
    from PySide6.QtGui import QAccessible
    from PySide6.QtWidgets import QApplication, QComboBox, QTabWidget, QTreeWidget
    from gui.accessibility_safety import (
        _selection_safe_factory,
        should_install_macos_accessibility_selection_guard,
    )
    from gui.ui_theme import DARK_TOKENS, LIGHT_TOKENS, build_stylesheet
except ImportError:
    pytest.skip(
        "requires the real PySide6 modules, not the runtime-test Qt stubs",
        allow_module_level=True,
    )


def _relative_luminance(hex_color: str) -> float:
    channels = [
        int(hex_color[index : index + 2], 16) / 255.0
        for index in (1, 3, 5)
    ]
    linear = [
        value / 12.92
        if value <= 0.04045
        else ((value + 0.055) / 1.055) ** 2.4
        for value in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(first: str, second: str) -> float:
    lighter, darker = sorted(
        (_relative_luminance(first), _relative_luminance(second)),
        reverse=True,
    )
    return (lighter + 0.05) / (darker + 0.05)


def test_platform_profiles_change_only_desktop_policy() -> None:
    macos = current_platform_profile("darwin")
    windows = current_platform_profile("win32")

    assert macos.key == "macos"
    assert windows.key == "windows"
    assert macos.native_menu_bar is True
    assert windows.native_menu_bar is False
    assert macos.control_height == 30
    assert windows.control_height == 32
    assert macos.start_shortcut == "Ctrl+Return"
    assert windows.start_shortcut == "Ctrl+Return"


def test_primary_buttons_and_muted_text_meet_contrast_floor() -> None:
    for tokens in (LIGHT_TOKENS, DARK_TOKENS):
        assert _contrast(tokens.on_accent, tokens.accent) >= 4.5
        assert _contrast(tokens.text, tokens.surface) >= 4.5
        assert _contrast(tokens.text_muted, tokens.surface) >= 4.5


def test_stylesheet_covers_component_and_pipeline_states() -> None:
    stylesheet = build_stylesheet(
        DARK_TOKENS,
        current_platform_profile("darwin"),
    )

    for selector in (
        'QPushButton[variant="primary"]',
        'QPushButton[variant="destructive"]',
        'QPushButton[variant="quiet"]:checked',
        'QLabel#sidebarPrimary',
        'QLabel[stageState="running"]',
        'QLabel[stageState="completed"]',
        'QLabel[stageState="safe_passthrough"]',
        'QLabel[stageState="degraded"]',
        'QLabel[stageState="failed"]',
        'QFrame#stateBanner[tone="warning"]',
    ):
        assert selector in stylesheet

    lowered = stylesheet.lower()
    assert "gradient" not in lowered
    assert "backdrop-filter" not in lowered
    assert "text-shadow" not in lowered


def test_macos_accessibility_selection_guard_is_platform_scoped() -> None:
    assert should_install_macos_accessibility_selection_guard("darwin")
    assert not should_install_macos_accessibility_selection_guard("linux")
    assert not should_install_macos_accessibility_selection_guard("win32")


def test_selection_guard_omits_volatile_selected_items() -> None:
    app = QApplication.instance() or QApplication([])
    combo = QComboBox()
    combo.addItems(["first", "second"])
    history_tree = QTreeWidget()
    inspector_tabs = QTabWidget()
    inspector_tabs.addTab(QTreeWidget(), "Run")

    targets = (
        ("QComboBoxListView", combo.view(), QAccessible.Role.List),
        ("QTreeWidget", history_tree, QAccessible.Role.Tree),
        ("QTabBar", inspector_tabs.tabBar(), QAccessible.Role.PageTabList),
    )
    for class_name, widget, expected_role in targets:
        interface = _selection_safe_factory(class_name, widget)

        assert interface is not None
        assert interface.isValid()
        assert interface.role() == expected_role
        assert interface.selectionInterface() is None

    assert _selection_safe_factory("QPushButton", combo) is None
