"""Small, reusable platform policy for the shared PySide6 interface.

The application keeps one widget tree on every desktop platform.  This module
contains only the differences that should follow the host OS: system font,
control metrics, native menu behavior, and shortcut conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
import sys


@dataclass(frozen=True)
class PlatformProfile:
    """Host-specific metrics used by the otherwise shared interface."""

    key: str
    display_name: str
    control_height: int
    compact_control_height: int
    toolbar_height: int
    window_margin: int
    panel_spacing: int
    native_menu_bar: bool
    start_shortcut: str


MACOS_PROFILE = PlatformProfile(
    key="macos",
    display_name="macOS",
    control_height=30,
    compact_control_height=26,
    toolbar_height=42,
    window_margin=16,
    panel_spacing=12,
    native_menu_bar=True,
    start_shortcut="Meta+Return",
)

WINDOWS_PROFILE = PlatformProfile(
    key="windows",
    display_name="Windows",
    control_height=32,
    compact_control_height=28,
    toolbar_height=46,
    window_margin=16,
    panel_spacing=12,
    native_menu_bar=False,
    start_shortcut="Ctrl+Return",
)

GENERIC_PROFILE = PlatformProfile(
    key="other",
    display_name="Desktop",
    control_height=32,
    compact_control_height=28,
    toolbar_height=44,
    window_margin=16,
    panel_spacing=12,
    native_menu_bar=False,
    start_shortcut="Ctrl+Return",
)


def current_platform_profile(platform_name: str | None = None) -> PlatformProfile:
    """Return the deterministic platform policy for ``platform_name``."""

    value = (platform_name or sys.platform).lower()
    if value.startswith("darwin"):
        return MACOS_PROFILE
    if value.startswith(("win32", "cygwin", "msys")):
        return WINDOWS_PROFILE
    return GENERIC_PROFILE


def system_ui_font() -> QFont:
    """Use the host's native application font without bundling a typeface."""

    from PySide6.QtGui import QFontDatabase

    return QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont)


def configure_main_window(window, profile: PlatformProfile) -> None:
    """Apply window behavior that legitimately differs by desktop platform."""

    from PySide6.QtCore import QSize

    menu_bar = window.menuBar()
    menu_bar.setNativeMenuBar(profile.native_menu_bar)
    if profile.key == "macos":
        window.setUnifiedTitleAndToolBarOnMac(True)
    toolbar = getattr(window, "main_toolbar", None)
    if toolbar is not None:
        toolbar.setMinimumHeight(profile.toolbar_height)
        toolbar.setIconSize(QSize(16, 16))


def standard_shortcuts(profile: PlatformProfile) -> dict[str, QKeySequence]:
    """Return platform-correct shortcuts while relying on Qt standard keys."""

    from PySide6.QtGui import QKeySequence

    return {
        "open": QKeySequence(QKeySequence.StandardKey.Open),
        "preferences": QKeySequence(QKeySequence.StandardKey.Preferences),
        "quit": QKeySequence(QKeySequence.StandardKey.Quit),
        "rerun": QKeySequence(QKeySequence.StandardKey.Refresh),
        "start": QKeySequence(profile.start_shortcut),
        "stop": QKeySequence(QKeySequence.StandardKey.Cancel),
    }


__all__ = [
    "PlatformProfile",
    "configure_main_window",
    "current_platform_profile",
    "standard_shortcuts",
    "system_ui_font",
]
