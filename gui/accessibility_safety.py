"""Safety guard for a Qt Cocoa accessibility selection crash on macOS.

Qt 6.11.1's Cocoa plugin can dereference a stale accessible item returned by
``selectedItems()`` for a few widget types.  The crash happens while macOS is
reading the accessibility hierarchy, outside application event handlers.  Use
ordinary widget accessibility for those controls, but deliberately omit their
volatile selection interface until the bundled Qt can be changed safely.
"""

from __future__ import annotations

import sys

from PySide6.QtGui import QAccessible
from PySide6.QtWidgets import QAccessibleWidget, QWidget


_SELECTION_SAFE_ROLES = {
    "QComboBoxListView": QAccessible.Role.List,
    "QTabBar": QAccessible.Role.PageTabList,
    "QTreeWidget": QAccessible.Role.Tree,
}
_installed = False


class _SelectionSafeAccessibleWidget(QAccessibleWidget):
    """Keep normal widget semantics while declining volatile selected items."""

    def selectionInterface(self):  # type: ignore[override]
        return None


def _selection_safe_factory(
    class_name: str,
    widget: QWidget,
) -> QAccessibleWidget | None:
    """Return a non-selection accessible interface for known unsafe widgets."""

    role = _SELECTION_SAFE_ROLES.get(str(class_name))
    if role is None:
        return None
    return _SelectionSafeAccessibleWidget(widget, role)


def should_install_macos_accessibility_selection_guard(
    platform_name: str | None = None,
) -> bool:
    """Keep the workaround strictly scoped to the affected platform."""

    return (platform_name or sys.platform) == "darwin"


def install_macos_accessibility_selection_guard() -> None:
    """Install the macOS-only factory before application widgets are created."""

    global _installed
    if _installed or not should_install_macos_accessibility_selection_guard():
        return
    QAccessible.installFactory(_selection_safe_factory)
    _installed = True
