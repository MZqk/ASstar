"""Semantic design tokens and system-aware Qt styling for Starun."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QObject, Qt, Slot
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication, QWidget

try:
    from .ui_platform import (
        PlatformProfile,
        current_platform_profile,
        system_ui_font,
    )
except ImportError:
    from ui_platform import (  # type: ignore[no-redef]
        PlatformProfile,
        current_platform_profile,
        system_ui_font,
    )


@dataclass(frozen=True)
class ThemeTokens:
    """A compact semantic palette shared by all application states."""

    name: str
    window: str
    surface: str
    surface_subtle: str
    surface_raised: str
    preview: str
    border: str
    border_strong: str
    text: str
    text_muted: str
    text_subtle: str
    # ``accent`` is the Starun brand purple and is reserved for the primary
    # action. Native selection and keyboard focus continue to use the host
    # palette's Highlight role instead of this application color.
    accent: str
    accent_text: str
    accent_hover: str
    accent_pressed: str
    accent_soft: str
    on_accent: str
    focus: str
    success: str
    success_soft: str
    warning: str
    warning_soft: str
    error: str
    error_soft: str
    info: str
    info_soft: str
    selection: str
    disabled: str


LIGHT_TOKENS = ThemeTokens(
    name="light",
    window="#f5f5f7",
    surface="#ffffff",
    surface_subtle="#f2f2f4",
    surface_raised="#ffffff",
    preview="#111318",
    border="#d1d1d6",
    border_strong="#a7a7ac",
    text="#1d1d1f",
    text_muted="#5f5f65",
    text_subtle="#6e6e73",
    accent="#6269a8",
    accent_text="#6269a8",
    accent_hover="#565d99",
    accent_pressed="#4b5187",
    accent_soft="#ececf6",
    on_accent="#ffffff",
    focus="#007aff",
    success="#2d7746",
    success_soft="#e7f3eb",
    warning="#8a5a12",
    warning_soft="#f8edd9",
    error="#a43f44",
    error_soft="#fae8e9",
    info="#3e718c",
    info_soft="#e5f0f5",
    selection="#e8e8ed",
    disabled="#8e8e93",
)

DARK_TOKENS = ThemeTokens(
    name="dark",
    window="#1c1c1e",
    surface="#2c2c2e",
    surface_subtle="#242426",
    surface_raised="#323235",
    preview="#0e1014",
    border="#48484a",
    border_strong="#636366",
    text="#f5f5f7",
    text_muted="#c7c7cc",
    text_subtle="#aeaeb2",
    accent="#6870b0",
    accent_text="#a2a8e0",
    accent_hover="#646cab",
    accent_pressed="#5d65a2",
    accent_soft="#34344a",
    on_accent="#ffffff",
    focus="#0a84ff",
    success="#67b17f",
    success_soft="#203b2a",
    warning="#d0a052",
    warning_soft="#41341f",
    error="#e47b7e",
    error_soft="#44282b",
    info="#78a9c1",
    info_soft="#233844",
    selection="#3a3a3c",
    disabled="#7c7c80",
)


def system_prefers_dark(app: QApplication) -> bool:
    """Read the OS color scheme, with a palette fallback for older Qt builds."""

    try:
        scheme = app.styleHints().colorScheme()
        if scheme == Qt.ColorScheme.Dark:
            return True
        if scheme == Qt.ColorScheme.Light:
            return False
    except (AttributeError, RuntimeError):
        pass
    window_color = app.palette().color(QPalette.ColorRole.Window)
    return window_color.lightness() < 128


def tokens_for_application(app: QApplication) -> ThemeTokens:
    return DARK_TOKENS if system_prefers_dark(app) else LIGHT_TOKENS


def repolish(widget: QWidget) -> None:
    """Refresh a widget after changing a dynamic style property."""

    style = widget.style()
    if style is not None:
        style.unpolish(widget)
        style.polish(widget)
    widget.update()


def set_style_property(widget: QWidget, name: str, value: object) -> None:
    if widget.property(name) == value:
        return
    widget.setProperty(name, value)
    repolish(widget)


def build_stylesheet(tokens: ThemeTokens, profile: PlatformProfile) -> str:
    """Build one native-desktop stylesheet from semantic tokens."""

    control = profile.control_height
    compact = profile.compact_control_height
    radius = 6 if profile.key == "macos" else 5
    log_font = (
        '"Menlo", "Monaco", monospace'
        if profile.key == "macos"
        else '"Cascadia Mono", "Consolas", monospace'
    )
    # A unified macOS toolbar should reveal the native title-bar surface. Other
    # desktop profiles use the same opaque window color without faking blur.
    toolbar_background = "transparent" if profile.key == "macos" else tokens.window
    return f"""
QMainWindow, QDialog, QWidget#appRoot {{
    background-color: {tokens.window};
    color: {tokens.text};
}}
QWidget {{
    color: {tokens.text};
    selection-background-color: palette(highlight);
    selection-color: palette(highlighted-text);
}}
QLabel#pageTitle {{
    font-size: 20px;
    font-weight: 650;
}}
QLabel#pageDescription, QLabel[tone="muted"] {{
    color: {tokens.text_muted};
}}
QLabel#sectionTitle {{
    font-size: 15px;
    font-weight: 650;
}}
QLabel#sectionDescription, QLabel[tone="subtle"] {{
    color: {tokens.text_subtle};
}}
QToolBar#mainToolbar {{
    background-color: {toolbar_background};
    border: none;
    border-bottom: 1px solid {tokens.border};
    spacing: 6px;
    padding: 4px 8px;
}}
QToolBar#mainToolbar QLabel#toolbarTitle {{
    font-weight: 650;
    padding: 0 6px 0 2px;
}}
QMenuBar {{
    background-color: {tokens.surface_subtle};
    color: {tokens.text};
}}
QMenuBar::item:selected, QMenu::item:selected {{
    background-color: palette(highlight);
    color: palette(highlighted-text);
}}
QMenu {{
    background-color: {tokens.surface_raised};
    color: {tokens.text};
    border: 1px solid {tokens.border};
    padding: 4px;
}}
QStatusBar {{
    background-color: {tokens.surface_subtle};
    border-top: 1px solid {tokens.border};
    color: {tokens.text_muted};
}}
QStatusBar::item {{ border: none; }}
QFrame#contentPanel, QFrame#previewPanel, QFrame#inspectorPanel,
QFrame#processingParametersSheet, QFrame#logPanel {{
    background-color: {tokens.surface};
    border: 1px solid {tokens.border};
    border-radius: 8px;
}}
QFrame#sidebarPanel, QFrame#inspectorPanel {{
    background-color: {tokens.surface_subtle};
    border: none;
    border-radius: 0;
}}
QFrame#inspectorPanel {{
    border-left: 1px solid {tokens.border};
}}
QTabWidget#runInspectorTabs::pane {{
    background-color: {tokens.surface_subtle};
    border: none;
    border-top: 1px solid {tokens.border};
}}
QTabWidget#runInspectorTabs QTabBar::tab {{
    min-height: {compact}px;
    padding: 0 12px;
    color: {tokens.text_muted};
    background-color: transparent;
    border: 1px solid transparent;
    border-bottom: 2px solid transparent;
}}
QTabWidget#runInspectorTabs QTabBar::tab:hover {{
    color: {tokens.text};
    background-color: {tokens.selection};
}}
QTabWidget#runInspectorTabs QTabBar::tab:selected {{
    color: {tokens.text};
    background-color: {tokens.surface_subtle};
    border-bottom-color: palette(highlight);
    font-weight: 650;
}}
QTabBar:focus {{
    border: 2px solid palette(highlight);
    border-radius: {radius}px;
}}
QFrame#phaseBar {{
    background-color: {tokens.surface_subtle};
    border-top: 1px solid {tokens.border};
    border-bottom: 1px solid {tokens.border};
}}
QFrame#dropZone {{
    background-color: {tokens.surface};
    border: 1px dashed {tokens.border_strong};
    border-radius: 10px;
}}
QFrame#dropZone[dragActive="true"] {{
    background-color: {tokens.info_soft};
    border: 1px solid {tokens.info};
}}
QPushButton {{
    min-height: {control}px;
    padding: 0 12px;
    background-color: {tokens.surface_raised};
    color: {tokens.text};
    border: 1px solid {tokens.border};
    border-radius: {radius}px;
}}
QPushButton:hover {{
    background-color: {tokens.surface_subtle};
    border-color: {tokens.border_strong};
}}
QPushButton:pressed {{
    background-color: {tokens.selection};
}}
QPushButton:focus, QLineEdit:focus, QComboBox:focus,
QSpinBox:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus,
QTreeWidget:focus {{
    border: 2px solid palette(highlight);
}}
QPushButton:disabled {{
    color: {tokens.disabled};
    background-color: {tokens.surface_subtle};
    border-color: {tokens.border};
}}
QPushButton[variant="primary"] {{
    background-color: {tokens.accent};
    color: {tokens.on_accent};
    border-color: {tokens.accent};
    font-weight: 650;
}}
QPushButton[variant="primary"]:hover {{
    background-color: {tokens.accent_hover};
    border-color: {tokens.accent_hover};
}}
QPushButton[variant="primary"]:pressed {{
    background-color: {tokens.accent_pressed};
    border-color: {tokens.accent_pressed};
}}
QPushButton[variant="destructive"] {{
    background-color: {tokens.error_soft};
    color: {tokens.error};
    border-color: {tokens.error};
    font-weight: 600;
}}
QPushButton[variant="destructive"]:disabled {{
    background-color: {tokens.surface_subtle};
    color: {tokens.disabled};
    border-color: {tokens.border};
}}
QPushButton[variant="quiet"] {{
    background-color: transparent;
    border-color: transparent;
}}
QPushButton[variant="quiet"]:hover {{
    background-color: {tokens.selection};
    border-color: transparent;
}}
QPushButton[variant="quiet"]:checked {{
    background-color: {tokens.selection};
    border-color: transparent;
}}
QPushButton[variant="compact"] {{
    min-height: {compact}px;
    padding: 0 9px;
}}
QPushButton[variant="icon"] {{
    min-width: {control}px;
    max-width: {control}px;
    padding: 0;
}}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    min-height: {control}px;
    padding: 0 8px;
    background-color: {tokens.surface_subtle};
    color: {tokens.text};
    border: 1px solid {tokens.border};
    border-radius: {radius}px;
}}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled,
QDoubleSpinBox:disabled {{
    color: {tokens.disabled};
    background-color: {tokens.window};
}}
QLineEdit {{ placeholder-text-color: {tokens.text_muted}; }}
QComboBox::drop-down {{ border: none; width: 24px; }}
QCheckBox {{
    min-height: {compact}px;
    spacing: 7px;
    color: {tokens.text};
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: {radius}px;
    padding: 0 4px;
}}
QCheckBox:focus {{
    background-color: {tokens.selection};
    border-color: palette(highlight);
}}
QGroupBox {{
    background-color: transparent;
    border: 1px solid {tokens.border};
    border-radius: 7px;
    margin-top: 12px;
    padding-top: 8px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 5px;
    background-color: {tokens.surface};
}}
QPlainTextEdit#logView, QPlainTextEdit#runInspectorLogView {{
    background-color: {tokens.preview};
    color: {tokens.text_muted};
    border: 1px solid {tokens.border};
    border-radius: 5px;
    padding: 7px;
    font-family: {log_font};
    font-size: 12px;
}}
QTreeWidget#historyTree {{
    background-color: {tokens.surface};
    alternate-background-color: {tokens.surface_subtle};
    border: 1px solid {tokens.border};
    border-radius: 6px;
    outline: none;
}}
QTreeWidget#historyTree::item {{
    min-height: 32px;
    padding: 3px 6px;
}}
QTreeWidget#historyTree::item:selected {{
    background-color: palette(highlight);
    color: palette(highlighted-text);
}}
QLabel#historyStatusBadge {{
    border: 1px solid {tokens.border};
    border-radius: 5px;
    color: {tokens.text_muted};
}}
QLabel#historyStatusBadge[historyStatus="preparing"],
QLabel#historyStatusBadge[historyStatus="running"] {{
    background-color: {tokens.info_soft};
    color: {tokens.info};
    border-color: {tokens.info};
}}
QLabel#historyStatusBadge[historyStatus="success"] {{
    background-color: {tokens.success_soft};
    color: {tokens.success};
    border-color: {tokens.success};
}}
QLabel#historyStatusBadge[historyStatus="partial_success"],
QLabel#historyStatusBadge[historyStatus="review_required"] {{
    background-color: {tokens.warning_soft};
    color: {tokens.warning};
    border-color: {tokens.warning};
}}
QLabel#historyStatusBadge[historyStatus="failed"],
QLabel#historyStatusBadge[historyStatus="interrupted"] {{
    background-color: {tokens.error_soft};
    color: {tokens.error};
    border-color: {tokens.error};
}}
QLabel#historyStatusBadge[available="false"] {{
    background-color: {tokens.surface_subtle};
    color: {tokens.text_muted};
    border-color: {tokens.border};
}}
QGraphicsView#previewCanvas {{
    background-color: {tokens.preview};
    border: 1px solid {tokens.border};
    border-radius: 5px;
}}
QProgressBar {{
    min-height: 16px;
    max-height: 16px;
    text-align: center;
    background-color: {tokens.surface_subtle};
    color: {tokens.text};
    border: 1px solid {tokens.border};
    border-radius: 4px;
}}
QProgressBar::chunk {{
    background-color: {tokens.info};
    border-radius: 3px;
}}
QSplitter::handle {{
    background-color: {tokens.border};
    margin: 0 2px;
}}
QSplitter::handle:hover {{ background-color: {tokens.border_strong}; }}
QScrollArea {{ background-color: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background-color: transparent; }}
QLabel[role="phase"] {{
    padding: 7px 10px;
    color: {tokens.text_muted};
    border: 1px solid {tokens.border};
    border-radius: 5px;
}}
QLabel[role="summary"] {{
    padding: 8px 2px;
    color: {tokens.text_muted};
    border-bottom: 1px solid {tokens.border};
}}
QLabel#sidebarPrimary {{
    padding: 2px 2px 6px 2px;
    font-size: 14px;
    font-weight: 650;
}}
QLabel[role="phase"][active="true"] {{
    color: {tokens.info};
    background-color: {tokens.info_soft};
    border-color: {tokens.info};
    font-weight: 650;
}}
QLabel[stageState], QPushButton[stageState] {{
    padding: 7px 8px;
    border: 1px solid transparent;
    border-radius: {radius}px;
}}
QPushButton[stageState] {{
    min-height: {control}px;
    text-align: left;
    background-color: transparent;
    font-weight: 400;
}}
QPushButton[stageState]:hover {{
    background-color: {tokens.surface_raised};
    border-color: {tokens.border};
}}
QLabel[stageState="waiting"], QLabel[stageState="skipped"],
QPushButton[stageState="waiting"], QPushButton[stageState="skipped"] {{
    color: {tokens.text_muted};
}}
QLabel[stageState="running"], QPushButton[stageState="running"] {{
    color: {tokens.info};
    background-color: {tokens.info_soft};
    border-color: {tokens.info};
    font-weight: 650;
}}
QLabel[stageState="completed"], QPushButton[stageState="completed"] {{
    color: {tokens.success};
    background-color: transparent;
}}
QLabel[stageState="safe_passthrough"],
QPushButton[stageState="safe_passthrough"] {{
    color: {tokens.info};
    background-color: transparent;
}}
QLabel[stageState="degraded"], QLabel[stageState="review"],
QLabel[stageState="review_required"],
QPushButton[stageState="degraded"], QPushButton[stageState="review"],
QPushButton[stageState="review_required"] {{
    color: {tokens.warning};
    background-color: {tokens.warning_soft};
    border-color: {tokens.warning};
}}
QLabel[stageState="failed"], QLabel[stageState="stopped"],
QPushButton[stageState="failed"], QPushButton[stageState="stopped"] {{
    color: {tokens.error};
    background-color: {tokens.error_soft};
    border-color: {tokens.error};
}}
QLabel[stageState][selected="true"],
QPushButton[stageState][selected="true"] {{
    color: palette(highlighted-text);
    background-color: palette(highlight);
    border-color: palette(highlight);
    font-weight: 650;
}}
QLabel[stageState]:focus, QPushButton[stageState]:focus {{
    border: 2px solid palette(highlight);
}}
QLabel[outcomeState] {{
    padding: 7px 10px;
    color: {tokens.text_muted};
    background-color: {tokens.surface_subtle};
    border: 1px solid {tokens.border};
    border-radius: {radius}px;
}}
QLabel[outcomeState="selected"] {{
    color: palette(highlighted-text);
    background-color: palette(highlight);
    border-color: palette(highlight);
    font-weight: 650;
}}
QLabel[outcomeState="preparing"],
QLabel[outcomeState="running"] {{
    color: {tokens.info};
    background-color: {tokens.info_soft};
    border-color: {tokens.info};
}}
QLabel[outcomeState="success"] {{
    color: {tokens.success};
    background-color: {tokens.success_soft};
    border-color: {tokens.success};
}}
QLabel[outcomeState="review"],
QLabel[outcomeState="review_required"],
QLabel[outcomeState="partial_success"],
QLabel[outcomeState="interrupted"] {{
    color: {tokens.warning};
    background-color: {tokens.warning_soft};
    border-color: {tokens.warning};
}}
QLabel[outcomeState="failed"],
QLabel[outcomeState="verification_failed"] {{
    color: {tokens.error};
    background-color: {tokens.error_soft};
    border-color: {tokens.error};
}}
QLabel[outcomeState]:focus {{
    border: 2px solid palette(highlight);
}}
QFrame#stateBanner, QFrame#taskBanner {{
    border: 1px solid {tokens.border};
    border-radius: 7px;
    background-color: {tokens.surface_subtle};
}}
QFrame#stateBanner[tone="success"],
QFrame#taskBanner[tone="success"] {{
    background-color: {tokens.success_soft};
    border-color: {tokens.success};
}}
QFrame#stateBanner[tone="success"] QLabel,
QFrame#taskBanner[tone="success"] QLabel {{ color: {tokens.success}; }}
QFrame#stateBanner[tone="warning"],
QFrame#taskBanner[tone="warning"] {{
    background-color: {tokens.warning_soft};
    border-color: {tokens.warning};
}}
QFrame#stateBanner[tone="warning"] QLabel,
QFrame#taskBanner[tone="warning"] QLabel {{ color: {tokens.warning}; }}
QFrame#stateBanner[tone="error"],
QFrame#taskBanner[tone="error"] {{
    background-color: {tokens.error_soft};
    border-color: {tokens.error};
}}
QFrame#stateBanner[tone="error"] QLabel,
QFrame#taskBanner[tone="error"] QLabel {{ color: {tokens.error}; }}
QFrame#stateBanner[tone="info"],
QFrame#taskBanner[tone="info"] {{
    background-color: {tokens.info_soft};
    border-color: {tokens.info};
}}
QFrame#stateBanner[tone="info"] QLabel,
QFrame#taskBanner[tone="info"] QLabel {{ color: {tokens.info}; }}
QLabel#statusLabel {{ font-weight: 650; }}
QLabel#statusLabel[state="running"] {{ color: {tokens.info}; }}
QLabel#statusLabel[state="success"] {{ color: {tokens.success}; }}
QLabel#statusLabel[state="warning"] {{ color: {tokens.warning}; }}
QLabel#statusLabel[state="error"] {{ color: {tokens.error}; }}
QToolTip {{
    background-color: {tokens.surface_raised};
    color: {tokens.text};
    border: 1px solid {tokens.border_strong};
    padding: 5px;
}}
"""


class ThemeController(QObject):
    """Keep the stylesheet synchronized with the live OS appearance."""

    def __init__(
        self,
        app: QApplication,
        profile: PlatformProfile | None = None,
    ) -> None:
        super().__init__(app)
        self.app = app
        self.profile = profile or current_platform_profile()
        self.tokens = tokens_for_application(app)
        self.apply()
        try:
            app.styleHints().colorSchemeChanged.connect(self._scheme_changed)
        except (AttributeError, RuntimeError):
            pass

    def apply(self) -> None:
        self.tokens = tokens_for_application(self.app)
        self.app.setFont(system_ui_font())
        # Keep the native Highlight/Accent roles untouched. The stylesheet uses
        # them for focus and selection while Starun purple remains a primary-
        # action color only.
        self.app.setStyleSheet(build_stylesheet(self.tokens, self.profile))
        self.app.setProperty("starunTheme", self.tokens.name)

    @Slot(object)
    def _scheme_changed(self, _scheme: object) -> None:
        self.apply()


def install_application_theme(
    app: QApplication,
    profile: PlatformProfile | None = None,
) -> ThemeController:
    """Install and retain the system-aware theme controller on ``app``."""

    existing = getattr(app, "_starun_theme_controller", None)
    if isinstance(existing, ThemeController):
        return existing
    controller = ThemeController(app, profile)
    app._starun_theme_controller = controller  # type: ignore[attr-defined]
    return controller


__all__ = [
    "DARK_TOKENS",
    "LIGHT_TOKENS",
    "ThemeController",
    "ThemeTokens",
    "build_stylesheet",
    "install_application_theme",
    "repolish",
    "set_style_property",
    "system_prefers_dark",
    "tokens_for_application",
]
