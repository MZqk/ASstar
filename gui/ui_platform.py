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
    # Qt maps the portable ``Ctrl`` modifier to the Command key on macOS.
    start_shortcut="Ctrl+Return",
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

    from PySide6.QtCore import QSize, Qt

    menu_bar = window.menuBar()
    menu_bar.setNativeMenuBar(profile.native_menu_bar)
    if profile.key == "macos":
        window.setUnifiedTitleAndToolBarOnMac(True)
    toolbar = getattr(window, "main_toolbar", None)
    if toolbar is not None:
        toolbar.setAllowedAreas(Qt.ToolBarArea.TopToolBarArea)
        toolbar.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)
        toolbar.setMinimumHeight(profile.toolbar_height)
        toolbar.setIconSize(QSize(16, 16))


def configure_toolbar_drag_region(widget, profile: PlatformProfile) -> None:
    """Turn a toolbar spacer into a reliable native title-bar drag channel.

    A child widget normally consumes mouse events even when it paints nothing.
    That makes the large empty part of a unified macOS toolbar feel "stuck".
    Passing events through the spacer leaves buttons interactive while allowing
    Qt's native unified toolbar/title bar to handle activation and dragging.
    """

    from PySide6.QtCore import Qt

    widget.setObjectName("windowDragRegion")
    widget.setProperty("windowDragRegion", True)
    widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    widget.setAttribute(
        Qt.WidgetAttribute.WA_TransparentForMouseEvents,
        profile.key == "macos",
    )


def fitted_window_geometry(
    geometry,
    available_geometry,
    *,
    minimum_size=None,
    force_center: bool = False,
    resize_to_fit: bool = True,
    margin: int = 12,
):
    """Return a window rectangle that remains reachable on one display.

    The calculation is intentionally independent from a concrete ``QScreen``
    so it can be regression-tested for removed, rotated, and narrow displays.
    When a window's minimum size is larger than the display, its top-left
    corner remains reachable instead of positioning the title bar off-screen.
    """

    from PySide6.QtCore import QRect, QSize

    available = QRect(available_geometry)
    if not available.isValid():
        return QRect(geometry)

    safe_margin = max(0, int(margin))
    if (
        available.width() > safe_margin * 2
        and available.height() > safe_margin * 2
    ):
        working = available.adjusted(
            safe_margin,
            safe_margin,
            -safe_margin,
            -safe_margin,
        )
    else:
        working = available

    source = QRect(geometry)
    minimum = QSize(minimum_size) if minimum_size is not None else QSize(1, 1)
    minimum_width = max(1, minimum.width())
    minimum_height = max(1, minimum.height())
    source_width = source.width() if source.width() > 0 else minimum_width
    source_height = source.height() if source.height() > 0 else minimum_height

    # A QWidget cannot be resized below its declared minimum. Preserve that
    # contract even on a synthetic/offscreen display smaller than the window.
    if resize_to_fit:
        width = max(minimum_width, min(source_width, working.width()))
        height = max(minimum_height, min(source_height, working.height()))
    else:
        width = max(minimum_width, source_width)
        height = max(minimum_height, source_height)

    intersects_display = source.intersects(available)
    if force_center or not intersects_display:
        x = working.center().x() - width // 2
        y = working.center().y() - height // 2
    else:
        x = source.x()
        y = source.y()

    if width <= working.width():
        x = min(max(x, working.left()), working.right() - width + 1)
    else:
        x = working.left()
    if height <= working.height():
        y = min(max(y, working.top()), working.bottom() - height + 1)
    else:
        y = working.top()
    return QRect(x, y, width, height)


def _fallback_available_geometry(reference=None):
    """Resolve a usable display without assuming one fixed monitor."""

    from PySide6.QtGui import QCursor, QGuiApplication

    screens = list(QGuiApplication.screens())
    if not screens:
        return None
    if reference is not None:
        try:
            reference_screen = reference.screen()
        except (AttributeError, RuntimeError):
            reference_screen = None
        if reference_screen in screens:
            return reference_screen.availableGeometry()
    cursor_screen = QGuiApplication.screenAt(QCursor.pos())
    if cursor_screen is not None:
        return cursor_screen.availableGeometry()
    primary = QGuiApplication.primaryScreen()
    return (primary or screens[0]).availableGeometry()


def constrain_window_to_visible_screens(
    window,
    *,
    reference=None,
    force_center: bool = False,
    resize_to_fit: bool = True,
) -> bool:
    """Clamp a normal window to a current display's usable region.

    Returns whether the geometry changed. Maximized and full-screen windows are
    left to the window server; their saved normal geometry is handled when the
    state is restored.
    """

    from PySide6.QtGui import QGuiApplication

    if window.isMaximized() or window.isFullScreen():
        return False
    screens = list(QGuiApplication.screens())
    if not screens:
        return False

    geometry = window.geometry()
    available_geometries = [screen.availableGeometry() for screen in screens]
    intersections = []
    for available in available_geometries:
        intersection = geometry.intersected(available)
        intersections.append(
            max(0, intersection.width()) * max(0, intersection.height())
        )
    best_index = max(range(len(intersections)), key=intersections.__getitem__)
    best_area = intersections[best_index]
    if best_area > 0:
        target_available = available_geometries[best_index]
    else:
        target_available = _fallback_available_geometry(reference)
        if target_available is None:
            return False

    target = fitted_window_geometry(
        geometry,
        target_available,
        minimum_size=window.minimumSize(),
        force_center=force_center or best_area <= 0,
        resize_to_fit=resize_to_fit,
    )
    if target == geometry:
        return False
    window.setGeometry(target)
    return True


def place_window_relative_to(
    window,
    reference,
    *,
    preferred_size: tuple[int, int],
) -> None:
    """Center a newly created utility window over its owning window."""

    from PySide6.QtCore import QRect

    width, height = (max(1, int(value)) for value in preferred_size)
    reference_geometry = reference.frameGeometry()
    geometry = QRect(
        reference_geometry.center().x() - width // 2,
        reference_geometry.center().y() - height // 2,
        width,
        height,
    )
    window.setGeometry(geometry)
    constrain_window_to_visible_screens(window, reference=reference)


def restore_window_geometry(
    window,
    saved_geometry,
    *,
    preferred_size: tuple[int, int],
    reference=None,
) -> bool:
    """Restore saved geometry and repair off-screen results."""

    from PySide6.QtCore import QRect

    restored = False
    if saved_geometry is not None:
        try:
            if isinstance(saved_geometry, QRect) and saved_geometry.isValid():
                window.setGeometry(saved_geometry)
                restored = True
            else:
                restored = bool(window.restoreGeometry(saved_geometry))
        except (RuntimeError, TypeError, ValueError):
            restored = False

    # Launch should always present a reachable window; maximized restoration is
    # persisted separately and intentionally reapplied by the owning scene.
    if restored and (
        window.isMinimized() or window.isMaximized() or window.isFullScreen()
    ):
        from PySide6.QtCore import Qt

        window.setWindowState(Qt.WindowState.WindowNoState)

    if not restored:
        if reference is not None:
            place_window_relative_to(
                window,
                reference,
                preferred_size=preferred_size,
            )
        else:
            window.resize(*preferred_size)
            constrain_window_to_visible_screens(
                window,
                force_center=True,
            )
    else:
        constrain_window_to_visible_screens(window, reference=reference)
    return restored


def normal_window_geometry(window):
    """Return the non-minimized/non-full-screen rectangle for persistence."""

    from PySide6.QtCore import QRect

    geometry = window.normalGeometry()
    if not geometry.isValid():
        geometry = window.geometry()
    return QRect(geometry)


def standard_shortcuts(profile: PlatformProfile) -> dict[str, QKeySequence]:
    """Return platform-correct shortcuts while relying on Qt standard keys."""

    from PySide6.QtGui import QKeySequence

    # In QKeySequence portable text, Ctrl maps to Command on macOS while Meta
    # maps to the physical Control key.  Keep Command-style shortcuts on Ctrl.
    command_modifier = "Ctrl"

    return {
        "open": QKeySequence(QKeySequence.StandardKey.Open),
        "open_folder": QKeySequence(f"{command_modifier}+Shift+O"),
        "close": QKeySequence(QKeySequence.StandardKey.Close),
        "preferences": QKeySequence(QKeySequence.StandardKey.Preferences),
        "quit": QKeySequence(QKeySequence.StandardKey.Quit),
        "undo": QKeySequence(QKeySequence.StandardKey.Undo),
        "redo": QKeySequence(QKeySequence.StandardKey.Redo),
        "cut": QKeySequence(QKeySequence.StandardKey.Cut),
        "copy": QKeySequence(QKeySequence.StandardKey.Copy),
        "paste": QKeySequence(QKeySequence.StandardKey.Paste),
        "select_all": QKeySequence(QKeySequence.StandardKey.SelectAll),
        "zoom_in": QKeySequence(QKeySequence.StandardKey.ZoomIn),
        "zoom_out": QKeySequence(QKeySequence.StandardKey.ZoomOut),
        "fit_preview": QKeySequence(f"{command_modifier}+0"),
        "actual_preview": QKeySequence(f"{command_modifier}+1"),
        "toggle_log": QKeySequence(f"{command_modifier}+Shift+L"),
        "toggle_sidebar": QKeySequence(
            "Meta+Ctrl+S" if profile.key == "macos" else "Ctrl+Alt+S"
        ),
        "toggle_inspector": QKeySequence(
            "Ctrl+Alt+I"
        ),
        "history": QKeySequence(
            "Ctrl+Y" if profile.key == "macos" else "Ctrl+Shift+H"
        ),
        "full_screen": QKeySequence(QKeySequence.StandardKey.FullScreen),
        "minimize": (
            QKeySequence("Ctrl+M")
            if profile.key == "macos"
            else QKeySequence()
        ),
        "help": QKeySequence(QKeySequence.StandardKey.HelpContents),
        "rerun": QKeySequence(QKeySequence.StandardKey.Refresh),
        "start": QKeySequence(profile.start_shortcut),
        "stop": QKeySequence(QKeySequence.StandardKey.Cancel),
    }


__all__ = [
    "PlatformProfile",
    "configure_main_window",
    "configure_toolbar_drag_region",
    "constrain_window_to_visible_screens",
    "current_platform_profile",
    "fitted_window_geometry",
    "normal_window_geometry",
    "place_window_relative_to",
    "restore_window_geometry",
    "standard_shortcuts",
    "system_ui_font",
]
