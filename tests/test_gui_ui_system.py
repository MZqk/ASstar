from __future__ import annotations

import pytest

from gui.ui_platform import current_platform_profile

try:
    from PySide6.QtGui import QAccessible, QColor, QPalette
    from PySide6.QtWidgets import QApplication, QComboBox, QTabWidget, QTreeWidget
    from gui.accessibility_safety import (
        _selection_safe_factory,
        should_install_macos_accessibility_selection_guard,
    )
    from gui.ui_theme import (
        DARK_TOKENS,
        LIGHT_TOKENS,
        ThemeController,
        build_stylesheet,
    )
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
        assert _contrast(tokens.on_accent, tokens.accent_hover) >= 4.5
        assert _contrast(tokens.on_accent, tokens.accent_pressed) >= 4.5
        assert _contrast(tokens.text, tokens.surface) >= 4.5
        assert _contrast(tokens.text_muted, tokens.surface) >= 4.5
        assert _contrast(tokens.text_subtle, tokens.surface) >= 4.5
        assert _contrast(tokens.success, tokens.success_soft) >= 4.5
        assert _contrast(tokens.warning, tokens.warning_soft) >= 4.5
        assert _contrast(tokens.error, tokens.error_soft) >= 4.5
        assert _contrast(tokens.info, tokens.info_soft) >= 4.5


def test_light_and_dark_tokens_use_macos_semantic_surfaces() -> None:
    assert LIGHT_TOKENS.window == "#f5f5f7"
    assert LIGHT_TOKENS.surface == "#ffffff"
    assert LIGHT_TOKENS.border == "#d1d1d6"
    assert LIGHT_TOKENS.text == "#1d1d1f"

    assert DARK_TOKENS.window == "#1c1c1e"
    assert DARK_TOKENS.surface == "#2c2c2e"
    assert DARK_TOKENS.border == "#48484a"
    assert DARK_TOKENS.text == "#f5f5f7"


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
        'QFrame#inspectorPanel',
        'QTabWidget#runInspectorTabs',
        'QPlainTextEdit#runInspectorLogView',
        'QPushButton[stageState][selected="true"]',
        'QPushButton[stageState="review"]',
        'QLabel[stageState="running"]',
        'QLabel[stageState="completed"]',
        'QLabel[stageState="safe_passthrough"]',
        'QLabel[stageState="degraded"]',
        'QLabel[stageState="failed"]',
        'QLabel[outcomeState="selected"]',
        'QLabel[outcomeState="review"]',
        'QLabel[outcomeState="partial_success"]',
        'QLabel[outcomeState="verification_failed"]',
        'QFrame#stateBanner[tone="warning"]',
        'QFrame#taskBanner[tone="warning"]',
        'QFrame#taskBanner[tone="info"]',
        'QCheckBox:focus',
        'QTabBar:focus',
        'QTreeWidget:focus',
        'QPushButton[stageState]:focus',
    ):
        assert selector in stylesheet

    assert "selection-background-color: palette(highlight);" in stylesheet
    assert "selection-color: palette(highlighted-text);" in stylesheet
    assert "border: 2px solid palette(highlight);" in stylesheet
    stage_selection = stylesheet.split(
        'QPushButton[stageState][selected="true"] {',
        1,
    )[1].split("}", 1)[0]
    outcome_selection = stylesheet.split(
        'QLabel[outcomeState="selected"] {',
        1,
    )[1].split("}", 1)[0]
    for selected_rule in (stage_selection, outcome_selection):
        assert "background-color: palette(highlight);" in selected_rule
        assert "color: palette(highlighted-text);" in selected_rule
    assert stylesheet.count(DARK_TOKENS.accent) == 2
    assert stylesheet.count(DARK_TOKENS.accent_hover) == 2
    assert stylesheet.count(DARK_TOKENS.accent_pressed) == 2
    assert DARK_TOKENS.accent_text not in stylesheet
    assert DARK_TOKENS.accent_soft not in stylesheet
    assert DARK_TOKENS.focus not in stylesheet

    lowered = stylesheet.lower()
    assert "gradient" not in lowered
    assert "backdrop-filter" not in lowered
    assert "text-shadow" not in lowered
    assert "box-shadow" not in lowered
    assert "blur" not in lowered


def test_unified_toolbar_is_transparent_only_on_macos() -> None:
    macos_stylesheet = build_stylesheet(
        LIGHT_TOKENS,
        current_platform_profile("darwin"),
    )
    windows_stylesheet = build_stylesheet(
        LIGHT_TOKENS,
        current_platform_profile("win32"),
    )

    toolbar_selector = "QToolBar#mainToolbar {"
    macos_toolbar = macos_stylesheet.split(toolbar_selector, 1)[1].split("}", 1)[0]
    windows_toolbar = windows_stylesheet.split(toolbar_selector, 1)[1].split("}", 1)[0]
    assert "background-color: transparent;" in macos_toolbar
    assert f"background-color: {LIGHT_TOKENS.window};" in windows_toolbar


def test_theme_controller_preserves_native_highlight_and_accent() -> None:
    app = QApplication.instance() or QApplication([])
    previous_palette = QPalette(app.palette())
    previous_stylesheet = app.styleSheet()
    previous_font = app.font()
    native_palette = QPalette(previous_palette)
    native_palette.setColor(QPalette.ColorRole.Highlight, QColor("#0a84ff"))
    native_palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    if hasattr(QPalette.ColorRole, "Accent"):
        native_palette.setColor(QPalette.ColorRole.Accent, QColor("#ff9f0a"))
    app.setPalette(native_palette)

    controller = ThemeController(app, current_platform_profile("darwin"))
    try:
        applied = app.palette()
        assert applied.color(QPalette.ColorRole.Highlight) == QColor("#0a84ff")
        assert applied.color(QPalette.ColorRole.HighlightedText) == QColor("#ffffff")
        if hasattr(QPalette.ColorRole, "Accent"):
            assert applied.color(QPalette.ColorRole.Accent) == QColor("#ff9f0a")
    finally:
        try:
            app.styleHints().colorSchemeChanged.disconnect(controller._scheme_changed)
        except (RuntimeError, TypeError):
            pass
        controller.deleteLater()
        app.setStyleSheet(previous_stylesheet)
        app.setPalette(previous_palette)
        app.setFont(previous_font)
        app.processEvents()


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
