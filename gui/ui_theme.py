"""Semantic design tokens and system-aware Qt styling for Starun."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QObject, Qt, Slot
from PySide6.QtGui import QColor, QPalette
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
    window="#f2f3f5",
    surface="#ffffff",
    surface_subtle="#f7f8fa",
    surface_raised="#ffffff",
    preview="#111318",
    border="#d7dbe2",
    border_strong="#aeb5c0",
    text="#20232a",
    text_muted="#59606c",
    text_subtle="#6c7481",
    accent="#6269a8",
    accent_text="#6269a8",
    accent_hover="#565d99",
    accent_pressed="#4b5187",
    accent_soft="#e9eaf6",
    on_accent="#ffffff",
    focus="#777fc1",
    success="#2f7c4a",
    success_soft="#e7f3eb",
    warning="#8a5a12",
    warning_soft="#f8edd9",
    error="#a43f44",
    error_soft="#fae8e9",
    info="#3e718c",
    info_soft="#e5f0f5",
    selection="#dfe2f3",
    disabled="#9aa0aa",
)

DARK_TOKENS = ThemeTokens(
    name="dark",
    window="#191b20",
    surface="#20232a",
    surface_subtle="#252830",
    surface_raised="#2a2e37",
    preview="#0e1014",
    border="#3a3f49",
    border_strong="#555d6a",
    text="#f1f3f7",
    text_muted="#b9bec8",
    text_subtle="#a4aab5",
    accent="#6870b0",
    accent_text="#a2a8e0",
    accent_hover="#747cbb",
    accent_pressed="#5d65a2",
    accent_soft="#30344d",
    on_accent="#ffffff",
    focus="#a2a8e0",
    success="#67b17f",
    success_soft="#203b2a",
    warning="#d0a052",
    warning_soft="#41341f",
    error="#df7376",
    error_soft="#44282b",
    info="#78a9c1",
    info_soft="#233844",
    selection="#3b405f",
    disabled="#777d87",
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
    return f"""
QMainWindow, QDialog, QWidget#appRoot {{
    background-color: {tokens.window};
    color: {tokens.text};
}}
QWidget {{
    color: {tokens.text};
    selection-background-color: {tokens.selection};
    selection-color: {tokens.text};
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
    background-color: {tokens.surface_subtle};
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
    background-color: {tokens.selection};
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
    background-color: {tokens.accent_soft};
    border: 1px solid {tokens.accent};
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
QSpinBox:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus {{
    border: 2px solid {tokens.focus};
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
QPlainTextEdit#logView {{
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
    background-color: {tokens.selection};
    color: {tokens.text};
}}
QLabel#historyStatusBadge {{
    border: 1px solid {tokens.border};
    border-radius: 5px;
    color: {tokens.text_muted};
}}
QLabel#historyStatusBadge[historyStatus="preparing"],
QLabel#historyStatusBadge[historyStatus="running"] {{
    background-color: {tokens.accent_soft};
    color: {tokens.accent_text};
    border-color: {tokens.accent};
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
    background-color: {tokens.accent};
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
    color: {tokens.accent_text};
    background-color: {tokens.accent_soft};
    border-color: {tokens.accent};
    font-weight: 650;
}}
QLabel[stageState] {{
    padding: 7px 8px;
    border: 1px solid transparent;
    border-radius: 5px;
}}
QLabel[stageState="waiting"], QLabel[stageState="skipped"] {{
    color: {tokens.text_muted};
}}
QLabel[stageState="running"] {{
    color: {tokens.text};
    background-color: {tokens.accent_soft};
    border-color: {tokens.accent};
    font-weight: 650;
}}
QLabel[stageState="completed"] {{
    color: {tokens.success};
    background-color: transparent;
}}
QLabel[stageState="safe_passthrough"] {{
    color: {tokens.info};
    background-color: transparent;
}}
QLabel[stageState="degraded"] {{
    color: {tokens.warning};
    background-color: {tokens.warning_soft};
    border-color: {tokens.warning};
}}
QLabel[stageState="failed"], QLabel[stageState="stopped"] {{
    color: {tokens.error};
    background-color: {tokens.error_soft};
    border-color: {tokens.error};
}}
QFrame#stateBanner {{
    border: 1px solid {tokens.border};
    border-radius: 7px;
    background-color: {tokens.surface_subtle};
}}
QFrame#stateBanner[tone="success"] {{
    background-color: {tokens.success_soft};
    border-color: {tokens.success};
}}
QFrame#stateBanner[tone="warning"] {{
    background-color: {tokens.warning_soft};
    border-color: {tokens.warning};
}}
QFrame#stateBanner[tone="error"] {{
    background-color: {tokens.error_soft};
    border-color: {tokens.error};
}}
QFrame#stateBanner[tone="info"] {{
    background-color: {tokens.info_soft};
    border-color: {tokens.info};
}}
QLabel#statusLabel {{ font-weight: 650; }}
QLabel#statusLabel[state="running"] {{ color: {tokens.accent_text}; }}
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
        palette = self.app.palette()
        palette.setColor(
            QPalette.ColorRole.Highlight,
            QColor(self.tokens.accent),
        )
        palette.setColor(
            QPalette.ColorRole.HighlightedText,
            QColor(self.tokens.on_accent),
        )
        if hasattr(QPalette.ColorRole, "Accent"):
            palette.setColor(
                QPalette.ColorRole.Accent,
                QColor(self.tokens.accent),
            )
        self.app.setPalette(palette)
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
