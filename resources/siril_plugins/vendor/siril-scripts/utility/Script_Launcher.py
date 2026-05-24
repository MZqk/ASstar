# STARTUP_CAPABLE
# March 13 2026
# (c) Rich Stevenson - Deep Space Astro
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Deep Space Astro - Script Launcher
# A drag-and-drop script launcher for Siril.
#
# Scripts are discovered automatically from subfolders of your siril-scripts
# directory. Drag scripts to the right panel to queue them, reorder by
# dragging, star favorites, and launch with one click. Selections, favorites,
# and window geometry are all persisted between launches.
#
# Version History:
# v1.0.0 - Original release
# v1.0.1 - Exported workflows now use relative paths for cross-platform compatibility
# v1.0.2 - Fixed manual steps not restoring as favorites after reloading a workflow
#           Scripts now launch using Siril's current working directory
#           Added ability to mark items as complete in the right panel
# v1.0.3 - Steps with / or \ in the name now display correctly
#           New button now prompts for confirmation if there are unsaved changes
#           Step notes dialog also allows renaming the step
#           Save button remains visible when the left panel is hidden
#           Copying a step now creates an independent copy that can be renamed separately

import sirilpy as s
s.ensure_installed("PyQt6")
from sirilpy import LogColor

import base64
import json
import os
import uuid
from pathlib import Path
import subprocess
import sys

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QVBoxLayout, QHBoxLayout, QPushButton,
    QLineEdit, QScrollArea, QFrame,
    QSplitter, QMessageBox, QSizePolicy, QCheckBox, QFileDialog,
    QDialog, QDialogButtonBox, QTextEdit, QComboBox, QMenu
)
from PyQt6.QtCore import Qt, QMimeData, QPoint, QByteArray, QTimer, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QDrag, QPixmap, QPainter, QFont, QIcon, QPolygon, QColor

VERSION = "1.0.3"

ICON_B64 = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/4gHYSUNDX1BST0ZJTEUAAQEAAAHIAAAAAAQwAABtbnRyUkdCIFhZWiAH4AABAAEAAAAAAABhY3NwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAA9tYAAQAAAADTLQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlkZXNjAAAA8AAAACRyWFlaAAABFAAAABRnWFlaAAABKAAAABRiWFlaAAABPAAAABR3dHB0AAABUAAAABRyVFJDAAABZAAAAChnVFJDAAABZAAAAChiVFJDAAABZAAAAChjcHJ0AAABjAAAADxtbHVjAAAAAAAAAAEAAAAMZW5VUwAAAAgAAAAcAHMAUgBHAEJYWVogAAAAAAAAb6IAADj1AAADkFhZWiAAAAAAAABimQAAt4UAABjaWFlaIAAAAAAAACSgAAAPhAAAts9YWVogAAAAAAAA9tYAAQAAAADTLXBhcmEAAAAAAAQAAAACZmYAAPKnAAANWQAAE9AAAApbAAAAAAAAAABtbHVjAAAAAAAAAAEAAAAMZW5VUwAAACAAAAAcAEcAbwBvAGcAbABlACAASQBuAGMALgAgADIAMAAxADb/2wBDAAUDBAQEAwUEBAQFBQUGBwwIBwcHBw8LCwkMEQ8SEhEPERETFhwXExQaFRERGCEYGh0dHx8fExciJCIeJBweHx7/2wBDAQUFBQcGBw4ICA4eFBEUHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh7/wAARCAAfACADASIAAhEBAxEB/8QAGAAAAwEBAAAAAAAAAAAAAAAABQcIBgP/xAAsEAACAQMCBAUDBQAAAAAAAAABAgMEBREAEgYHITETIkFRYQgUQhVScYHR/8QAFwEBAQEBAAAAAAAAAAAAAAAABQQBBv/EACYRAAECBAUEAwAAAAAAAAAAAAECEQADBCEFMUFRYQYTInESsfD/2gAMAwEAAhEDEQA/AIy00uWXLs3S1Ul0uNda7ebjKVt61k4EkyocOY0xgnuMntrP8s4bzUPXUtqpZahZvAadEgEpMaSqxOME9CB2/j106+WtLw7xjwLFT1s8lNdaCJaK4QqhWSMJKzRAAjKg4ByuPMCD21JVVK6dIWmw32/H3HL9RYuqikkpLBwCR5EAixI0v7fKzuOE3KuYn7aKSWfJOIq+nUq/wChO0/PYaSfM7haThTiM0LwPTrJGJBE7bjHnIKhvyGRkH2I9c6qWqpOJrhm2friR07BJCIF8CcxjymNpMnG7GSRg+g0hPqLqWp77b+G5p46motlNiSQEs0QdmdYt7Es+1WU5PXrrKTE59UCicQrYsxHFoE6VxqsqqjtTpiVuCWAIIG7sNWDNrnpGz5BV80fKi611ZRNPFZpZ3o2jO2V1aMNJEjYypLbOo/cPYaE8m6LiyfmtJxNfKSSjjuzzpOk48Mys4LABD5toZVIOMZA663nLSxfbcgqSlopCKu5KJQ27bh5JQe/p5VA/rRy/Ja6C4Wa0zRxziqWNNgVkJfoPHRx1Rt3Tv/uiptQXWEh/kSD6ECTsQlmorJctDmapaeQkC5AyuXPJHEC/qI4qu/CXDtLJZIGWsqmIkqgu9aVR+WO29iwAJzjBwMkHUmzyyzzPNPI8ssjFnd2JZiepJJ7nVn3Cz0QuEUFZBJVxVUa08s1TUNK4UjO3afLj36HUz887JarDzBq6SzR+DSsquYguFjfqCFHt0z8Zx6aqwlSAjtJFxmd4a6HraRINJLR5kFRVuHy1ZnsPomP/2Q=="
)

# ── Config ────────────────────────────────────────────────────────────────────
# Entries in the siril-scripts root to skip when scanning for subfolders
EXCLUDED_ENTRIES = {".git", ".gitlab", "license.md", "readme.md"}

# STATE_FILE is resolved at runtime via siril.get_siril_configdir() in main()
STATE_FILE: str = ""  # set by main() before load_state() is called

# MIME type used for all drags — payload is "SOURCE|path"
MIME_TYPE = "application/x-siril-script"
STEP_PREFIX = "step://"  # prefix for custom manual steps in checked_paths
SRC_LEFT  = "left"
SRC_RIGHT = "right"


# ── Persistence ───────────────────────────────────────────────────────────────

def load_state() -> tuple[list, set, dict | None, dict, str, dict]:
    """Return (selections, favorites, geometry, workflows, active_workflow, notes) from state."""
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            selections     = data.get("selections", [])
            favorites      = set(data.get("favorites", []))
            geometry       = data.get("geometry", None)
            workflows      = data.get("workflows", {})
            active_workflow = data.get("active_workflow", "")
            notes          = data.get("notes", {})
            return selections, favorites, geometry, workflows, active_workflow, notes
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return [], set(), None, {}, "", {}


def save_state(paths: list, favs: set, geometry: dict | None = None,
               workflows: dict | None = None,
               active_workflow: str | None = None,
               notes: dict | None = None) -> None:
    """Write selections, favorites, geometry, workflows, active_workflow and notes to state."""
    existing: dict = {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as fh:
            existing = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    existing["selections"] = list(paths)
    existing["favorites"]  = sorted(favs)
    if geometry is not None:
        existing["geometry"] = geometry
    if workflows is not None:
        existing["workflows"] = workflows
    if active_workflow is not None:
        existing["active_workflow"] = active_workflow
    if notes is not None:
        existing["notes"] = notes
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=2)


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_siril_scripts_dir(user_data_dir: str | None = None) -> str | None:
    """Locate the siril-scripts directory.

    Tries, in order:
      1. SIRIL_SCRIPTS_DIR environment variable override
      2. user_data_dir from siril.get_siril_userdatadir() — cross-platform
      3. Falls back to a QFileDialog if nothing is found
    """
    # 1. Explicit env override always wins
    env_path = os.environ.get("SIRIL_SCRIPTS_DIR")
    if env_path and os.path.isdir(env_path):
        return env_path

    # 2. Derive siril-scripts as a sibling of the siril user data dir
    if user_data_dir:
        scripts_dir = str(Path(user_data_dir).parent / "siril-scripts")
        if os.path.isdir(scripts_dir):
            return scripts_dir

    return None


def scan_scripts(directory: str) -> list[tuple[str, str]]:
    """Scan subfolders of directory for .py scripts. Returns (label, path) pairs."""
    scripts = []
    launcher_path = os.path.realpath(__file__)
    excluded = {e.lower() for e in EXCLUDED_ENTRIES}
    for subdir in sorted(os.scandir(directory), key=lambda e: e.name.lower()):
        if not subdir.is_dir() or subdir.name.lower() in excluded:
            continue
        for entry in sorted(os.scandir(subdir.path), key=lambda e: e.name.lower()):
            if (entry.is_file() and entry.name.endswith(".py")
                    and os.path.realpath(entry.path) != launcher_path):
                scripts.append((f"[{subdir.name.capitalize()}]  {entry.name}", entry.path))
    return scripts


def scan_scripts_grouped(directory: str) -> dict[str, list[tuple[str, str]]]:
    """Return {subdir_label: [(name, path), ...]} for the grouped left panel."""
    groups: dict[str, list[tuple[str, str]]] = {}
    launcher_path = os.path.realpath(__file__)
    excluded = {e.lower() for e in EXCLUDED_ENTRIES}
    for subdir in sorted(os.scandir(directory), key=lambda e: e.name.lower()):
        if not subdir.is_dir() or subdir.name.lower() in excluded:
            continue
        entries = []
        for entry in sorted(os.scandir(subdir.path), key=lambda e: e.name.lower()):
            if (entry.is_file() and entry.name.endswith(".py")
                    and os.path.realpath(entry.path) != launcher_path):
                entries.append((entry.name, entry.path))
        if entries:
            groups[subdir.name.capitalize()] = entries
    return groups


def make_drag(widget: QWidget, source_id: str, path: str) -> QDrag:
    """Create a QDrag carrying SOURCE|path and a ghost pixmap of the widget."""
    drag = QDrag(widget)
    mime = QMimeData()
    payload = f"{source_id}|{path}"
    mime.setData(MIME_TYPE, QByteArray(payload.encode()))
    drag.setMimeData(mime)

    pixmap = QPixmap(widget.size())
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setOpacity(0.55)
    widget.render(painter)
    painter.end()
    drag.setPixmap(pixmap)
    return drag


def parse_mime(event) -> tuple[str, str] | None:
    """Return (source_id, path) from a drop event, or None if invalid."""
    if not event.mimeData().hasFormat(MIME_TYPE):
        return None
    raw = bytes(event.mimeData().data(MIME_TYPE)).decode()
    parts = raw.split("|", 1)
    if len(parts) != 2:
        return None
    return parts[0], parts[1]



def extract_docstring(path: str) -> str:
    """Return the module-level docstring from a .py file, or '' if none."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            source = fh.read()
        import ast as _ast
        tree = _ast.parse(source)
        doc = _ast.get_docstring(tree)
        return doc.strip() if doc else ""
    except Exception:
        return ""


def wrap_tooltip(text: str, width: int = 60) -> str:
    """Word-wrap plain text to `width` chars and return as an HTML tooltip
    so Qt renders it at a consistent width regardless of content length."""
    import textwrap
    if not text:
        return ""
    lines = []
    for paragraph in text.splitlines():
        if paragraph.strip():
            lines.extend(textwrap.wrap(paragraph, width))
        else:
            lines.append("")          # preserve blank lines between paragraphs
    escaped = "<br>".join(
        line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        for line in lines
    )
    return f"<html><body><p style='white-space:pre-wrap'>{escaped}</p></body></html>"

def drop_index_in_panel(scroll_area: "QScrollArea", rows_layout, drop_pos_local: QPoint) -> int:
    """Return the insertion index for a drop inside a scrollable panel.

    drop_pos_local is in the coordinate space of the QScrollArea viewport.
    We convert it into the container widget's coordinate space (accounting for
    scroll offset) and compare against each row's midpoint.
    """
    scroll_y = scroll_area.verticalScrollBar().value()
    container_y = drop_pos_local.y() + scroll_y

    # Collect only real widget rows (skip the trailing spacer/stretch)
    widget_rows = []
    for i in range(rows_layout.count()):
        item = rows_layout.itemAt(i)
        w = item.widget() if item else None
        if w is not None:
            widget_rows.append((i, w))

    for layout_i, w in widget_rows:
        mid_y = w.y() + w.height() // 2
        if container_y < mid_y:
            return layout_i  # insert before this row

    # Cursor is below all rows — append at end (index = number of widget rows)
    return len(widget_rows)



class PlayButton(QPushButton):
    """A QPushButton that paints a solid triangle play icon via QPainter."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(28, 28)
        self._is_running = False

    def set_running(self, running: bool):
        self._is_running = running
        self.setEnabled(not running)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._is_running:
            return  # show default disabled look when running
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Centre a proportional triangle in the button
        w, h = self.width(), self.height()
        margin = 9
        x0 = (w - (h - margin * 2) // 2) // 2 + 2  # nudge right slightly
        y0 = margin
        y1 = h - margin
        ymid = h // 2
        tri = QPolygon([
            QPoint(x0,              y0),
            QPoint(x0,              y1),
            QPoint(x0 + (y1 - y0) * 6 // 10, ymid),
        ])
        color = self.palette().buttonText().color()
        if not self.isEnabled():
            color = self.palette().mid().color()
        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPolygon(tri)
        painter.end()


# ── Left panel row ────────────────────────────────────────────────────────────

class LeftScriptRow(QFrame):
    """A single draggable row in the left Scripts panel."""

    def __init__(self, label: str, path: str, on_add, parent=None):
        super().__init__(parent)
        self.path = path
        self.on_add = on_add
        self._drag_start = QPoint()

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

        # Build tooltip: prefer docstring, fall back to generic hint
        doc = extract_docstring(path)
        raw = doc if doc else "Drag to Selected Scripts to add\nDouble-click to add instantly"
        self.setToolTip(wrap_tooltip(raw))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)

        lbl = QLabel(label)
        layout.addWidget(lbl, stretch=1)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.on_add(self.path)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.pos()

    def mouseMoveEvent(self, event):
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        if (event.pos() - self._drag_start).manhattanLength() < QApplication.startDragDistance():
            return
        drag = make_drag(self, SRC_LEFT, self.path)
        drag.exec(Qt.DropAction.CopyAction)


# ── Left panel ────────────────────────────────────────────────────────────────

class AnimatedRowContainer(QWidget):
    """A container whose height can be animated."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(2)

    def add_row(self, row: QWidget):
        self._layout.addWidget(row)


class SubdirHeader(QWidget):
    """Clickable collapsible section header with animation and script count."""

    def __init__(self, label: str, count: int = 0, parent=None):
        super().__init__(parent)
        self._collapsed = False
        self._container: AnimatedRowContainer | None = None
        self._anim: QPropertyAnimation | None = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        self._arrow = QLabel("▾")
        layout.addWidget(self._arrow)

        self._label = QLabel(f"<b>{label}</b>")
        layout.addWidget(self._label, stretch=1)

        self._count_lbl = QLabel(f"({count})")
        layout.addWidget(self._count_lbl)

        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_container(self, container: "AnimatedRowContainer"):
        self._container = container

    def set_count(self, count: int):
        self._count_lbl.setText(f"({count})")

    def _animate(self, collapse: bool):
        if self._container is None:
            return
        self._container.setVisible(True)
        full_h = self._container.sizeHint().height()
        start_h = 0 if collapse is False else full_h
        end_h   = full_h if collapse is False else 0

        anim = QPropertyAnimation(self._container, b"maximumHeight")
        anim.setDuration(150)
        anim.setStartValue(start_h)
        anim.setEndValue(end_h)
        anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        if collapse:
            anim.finished.connect(lambda: self._container.setVisible(False))
        else:
            self._container.setMaximumHeight(16777215)
        self._anim = anim
        anim.start()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._collapsed = not self._collapsed
            self._arrow.setText("▸" if self._collapsed else "▾")
            self._animate(collapse=self._collapsed)

    @property
    def rows(self) -> list[QWidget]:
        if self._container is None:
            return []
        layout = self._container._layout
        return [layout.itemAt(i).widget() for i in range(layout.count())
                if layout.itemAt(i).widget()]


class ScriptsPanel(QWidget):
    """Left panel: scripts grouped by subdir with collapsible headers.
    Accepts drops from the right panel to remove a script."""

    def __init__(self, scripts_dir: str, scripts: list[tuple[str, str]],
                 checked_paths: list, on_remove, on_add, parent=None):
        super().__init__(parent)
        self.scripts_dir = scripts_dir
        self.scripts = scripts
        self.checked_paths = checked_paths
        self.on_remove = on_remove
        self.on_add = on_add

        self.setAcceptDrops(True)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)

        title = QLabel("Available Scripts")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(title)

        hint = QLabel("Drag selected scripts here to remove")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setWordWrap(True)
        outer.addWidget(hint)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.StyledPanel)
        outer.addWidget(self._scroll)

        self._container = QWidget()
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(2)
        self._layout.addStretch()
        self._scroll.setWidget(self._container)

        # Track collapsed state by group label across rebuilds
        self._collapsed: dict[str, bool] = {}

        self.rebuild()

    def rebuild(self, scripts=None):
        if scripts is not None:
            self.scripts = scripts

        # Save current collapsed state before clearing
        for i in range(self._layout.count()):
            item = self._layout.itemAt(i)
            w = item.widget() if item else None
            if isinstance(w, SubdirHeader):
                key = w._label.text().replace("<b>","").replace("</b>","")
                self._collapsed[key] = w._collapsed

        while self._layout.count() > 1:
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Rebuild grouped — filter scripts to match current self.scripts list
        from collections import OrderedDict
        groups: OrderedDict[str, list[tuple[str, str]]] = OrderedDict()
        for label, path in self.scripts:
            # Extract group name from "[Group]  filename" label format
            if label.startswith("["):
                grp = label[1:label.index("]")]
            else:
                grp = "Other"
            groups.setdefault(grp, []).append((label.split("]  ", 1)[-1], path))

        pos = 0
        for grp, entries in groups.items():
            total_count = len(entries)
            header = SubdirHeader(grp, count=total_count)
            if self._collapsed.get(grp, False):
                header._collapsed = True
                header._arrow.setText("▸")
            self._layout.insertWidget(pos, header)
            pos += 1

            container = AnimatedRowContainer()
            for name, path in entries:
                row = LeftScriptRow(name, path, self.on_add)
                if path in self.checked_paths:
                    row.setVisible(False)
                container.add_row(row)

            header.set_container(container)
            if header._collapsed:
                container.setVisible(False)
            else:
                container.setMaximumHeight(16777215)
            self._layout.insertWidget(pos, container)
            pos += 1

    # Accept drops from the right panel (remove action)
    def dragEnterEvent(self, event):
        parsed = parse_mime(event)
        if parsed and parsed[0] == SRC_RIGHT:
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        parsed = parse_mime(event)
        if parsed and parsed[0] == SRC_RIGHT:
            event.acceptProposedAction()

    def dropEvent(self, event):
        parsed = parse_mime(event)
        if not parsed or parsed[0] != SRC_RIGHT:
            return
        _, path = parsed
        self.on_remove(path)
        event.acceptProposedAction()





class CheckButton(QPushButton):
    """A QPushButton that paints a checkmark icon; turns green when completed."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(26, 26)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self._checked = False

    def set_checked(self, checked: bool):
        self._checked = checked
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        from PyQt6.QtGui import QColor
        color = QColor("#4fc3f7") if self._checked else self.palette().buttonText().color()
        pen = p.pen()
        pen.setColor(color)
        pen.setWidth(2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        w, h = self.width(), self.height()
        # Draw checkmark: short left stroke + longer right stroke
        p.drawLine(w // 2 - 5, h // 2, w // 2 - 1, h // 2 + 4)
        p.drawLine(w // 2 - 1, h // 2 + 4, w // 2 + 5, h // 2 - 4)
        p.end()

class PencilButton(QPushButton):
    """A QPushButton that paints a pencil icon; turns blue when a note exists."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(26, 26)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self._has_note = False

    def set_has_note(self, has_note: bool):
        self._has_note = has_note
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._has_note:
            from PyQt6.QtGui import QColor
            color = QColor("#4fc3f7")
        else:
            color = self.palette().buttonText().color()

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color)

        w, h = self.width(), self.height()
        cx, cy = w // 2, h // 2

        from PyQt6.QtGui import QTransform
        # Pencil body — tall thin rectangle, rotated 45 degrees
        t = QTransform()
        t.translate(cx - 1, cy)
        t.rotate(45)
        p.setTransform(t)
        # Body rectangle
        p.drawRoundedRect(-2, -6, 4, 9, 1, 1)
        # Tip triangle
        tip = QPolygon([QPoint(-2, 3), QPoint(2, 3), QPoint(0, 6)])
        p.drawPolygon(tip)
        # Eraser (top cap)
        p.setTransform(QTransform())
        # reset and draw eraser as a small rounded rect at the top of the pencil
        t2 = QTransform()
        t2.translate(cx - 1, cy)
        t2.rotate(45)
        p.setTransform(t2)
        from PyQt6.QtGui import QColor as _QColor
        eraser_color = _QColor("#4fc3f7") if self._has_note else self.palette().mid().color()
        p.setBrush(eraser_color)
        p.drawRoundedRect(-2, -9, 4, 3, 1, 1)

        p.end()

class TrashButton(QPushButton):
    """A QPushButton that paints a simple trash-can icon via QPainter."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(26, 26)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.setToolTip("Remove from selected scripts")

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = self.palette().buttonText().color()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color)
        w, h = self.width(), self.height()
        cx = w // 2

        # Lid
        lid_x = cx - 6
        lid_w = 12
        lid_h = 2
        lid_y = h // 2 - 5
        p.drawRoundedRect(lid_x, lid_y, lid_w, lid_h, 1, 1)

        # Handle on lid
        handle_w = 4
        handle_h = 2
        p.drawRoundedRect(cx - handle_w // 2, lid_y - handle_h, handle_w, handle_h, 1, 1)

        # Body
        body_x = cx - 5
        body_w = 10
        body_y = lid_y + lid_h + 1
        body_h = 8
        p.drawRoundedRect(body_x, body_y, body_w, body_h, 1, 1)

        # Lines inside body (cutouts — draw in button background colour)
        bg = self.palette().button().color()
        p.setBrush(bg)
        line_h = body_h - 2
        line_w = 1
        line_y = body_y + 1
        for offset in (-2, 0, 2):
            p.drawRect(cx + offset - line_w // 2, line_y, line_w, line_h)

        p.end()


class NoteButton(QLabel):
    """A fixed-size label showing a coloured note emoji in place of a play button."""

    def __init__(self, parent=None):
        super().__init__("🟢", parent)
        self.setFixedSize(28, 28)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setToolTip("Manual step — no script to run")
        self.setCursor(Qt.CursorShape.ArrowCursor)
        font = self.font()
        font.setPixelSize(14)
        self.setFont(font)

    def set_running(self, running: bool):
        pass  # Manual steps have no run state

    def setEnabled(self, enabled: bool):
        pass  # Always visible, never disabled

    def setToolTip(self, tip: str):
        super().setToolTip("Manual step — no script to run")

# ── Right panel row ───────────────────────────────────────────────────────────

class RightScriptRow(QFrame):
    """A row in the Selected Scripts panel — draggable for reordering or
    dragging back to the left panel to remove."""

    def __init__(self, path: str, name: str, on_run, is_fav: bool = False,
                 on_fav_toggle=None, on_remove=None, on_copy=None,
                 missing: bool = False, note: str = "",
                 on_notes_changed=None, on_step_renamed=None,
                 completed: bool = False, on_completed_toggle=None, parent=None):
        super().__init__(parent)
        self.path = path
        self.name = name
        self._drag_start = QPoint()
        self._is_fav = is_fav
        self._on_fav_toggle = on_fav_toggle
        self._on_remove = on_remove
        self._on_copy = on_copy
        self._missing = missing
        self._note = note
        self._on_notes_changed = on_notes_changed
        self._on_step_renamed = on_step_renamed
        self._completed = completed
        self._on_completed_toggle = on_completed_toggle

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setToolTip("Drag to reorder, or drag back to Scripts to remove")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        handle = QLabel("⠿")
        handle.setCursor(Qt.CursorShape.OpenHandCursor)
        layout.addWidget(handle)

        _is_step = path.startswith(STEP_PREFIX)
        if _is_step:
            self._run_btn = NoteButton()
            layout.addWidget(self._run_btn)
        else:
            self._run_btn = PlayButton()
            self._run_btn.setToolTip(f"Run {name}")
            self._run_btn.setCursor(Qt.CursorShape.ArrowCursor)
            self._run_btn.clicked.connect(lambda: self._on_run_clicked(path, name, on_run))
            layout.addWidget(self._run_btn)

        if _is_step:
            lbl = QLabel(name)
            lbl.setToolTip("Double-click to rename this step")
            lbl.mouseDoubleClickEvent = lambda e: self._rename_step()
        else:
            lbl = QLabel(name)
        self._lbl = lbl
        layout.addWidget(lbl, stretch=1)

        self._check_btn = CheckButton()
        self._check_btn.setToolTip("Mark as complete")
        self._check_btn.clicked.connect(self._toggle_completed)
        layout.addWidget(self._check_btn)

        self._notes_btn = PencilButton()
        self._notes_btn.clicked.connect(self._open_notes)
        layout.addWidget(self._notes_btn)
        self._update_notes_btn()
        self._apply_completed_style()

        self._star_btn = QPushButton("★" if is_fav else "☆")
        self._star_btn.setFixedWidth(26)
        self._star_btn.setToolTip("Remove from favorites" if is_fav else "Add to favorites")
        self._star_btn.setCursor(Qt.CursorShape.ArrowCursor)
        self._star_btn.setStyleSheet("color: gold;" if is_fav else "")
        self._star_btn.clicked.connect(self._toggle_fav)
        layout.addWidget(self._star_btn)

        self._trash_btn = TrashButton()
        self._trash_btn.clicked.connect(self._confirm_remove)
        layout.addWidget(self._trash_btn)

        if missing:
            self.setToolTip("Script not installed.")
            self.setCursor(Qt.CursorShape.ForbiddenCursor)
            if not _is_step:
                self._run_btn.setEnabled(False)
                self._run_btn.setToolTip("Script not installed.")
            self._star_btn.setEnabled(False)
            lbl.setEnabled(False)
            handle.setEnabled(False)

    def _toggle_fav(self):
        self._is_fav = not self._is_fav
        self._star_btn.setText("★" if self._is_fav else "☆")
        self._star_btn.setToolTip("Remove from favorites" if self._is_fav else "Add to favorites")
        self._star_btn.setStyleSheet("color: gold;" if self._is_fav else "")
        if self._on_fav_toggle:
            self._on_fav_toggle(self.path, self._is_fav)

    def _confirm_remove(self):
        reply = QMessageBox.question(
            self, "Remove Script",
            f"Remove '{self.name}' from the selected scripts?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes and self._on_remove:
            self._on_remove(self.path)

    def _toggle_completed(self):
        self._completed = not self._completed
        self._apply_completed_style()
        if self._on_completed_toggle:
            self._on_completed_toggle(self.path, self._completed)

    def _apply_completed_style(self):
        if self._completed:
            self._lbl.setStyleSheet("text-decoration: line-through; color: gray;")
            self._check_btn.set_checked(True)
            self._check_btn.setToolTip("Mark as incomplete")
        else:
            self._lbl.setStyleSheet("")
            self._check_btn.set_checked(False)
            self._check_btn.setToolTip("Mark as complete")

    def _update_notes_btn(self):
        """Set notes button colour and tooltip based on whether a note exists."""
        self._notes_btn.set_has_note(bool(self._note))
        tip = self._note if self._note else "Add or edit notes"
        self._notes_btn.setToolTip(tip)

    def _open_notes(self):
        """Open a dialog to view/edit notes for this script."""
        _is_step = self.path.startswith(STEP_PREFIX)
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Notes — {self.name}")
        dlg.setMinimumWidth(360)
        dlg.setMinimumHeight(200)
        dlg.setWindowFlags(dlg.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(10, 10, 10, 8)
        layout.setSpacing(6)

        # Step name field — only for manual steps
        if _is_step:
            layout.addWidget(QLabel("Step name:"))
            name_edit = QLineEdit(self.name)
            layout.addWidget(name_edit)

        layout.addWidget(QLabel("Notes:"))
        editor = QTextEdit()
        editor.setPlaceholderText("Type your notes here…")
        editor.setText(self._note)
        layout.addWidget(editor)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save |
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        # Rename step if name changed
        if _is_step:
            new_name = name_edit.text().strip()
            if new_name and new_name != self.name and self._on_step_renamed:
                self._on_step_renamed(self.path, new_name)
        self._note = editor.toPlainText().strip()
        self._update_notes_btn()
        if self._on_notes_changed:
            self._on_notes_changed(self.path, self._note)

    def _rename_step(self):
        """Open a dialog to rename this manual step."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Rename Step")
        dlg.setMinimumWidth(300)
        dlg.setWindowFlags(dlg.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        layout = QVBoxLayout(dlg)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 8)
        layout.addWidget(QLabel("Step name:"))
        name_edit = QLineEdit(self.name)
        name_edit.selectAll()
        layout.addWidget(name_edit)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        new_name = name_edit.text().strip()
        if not new_name or new_name == self.name:
            return
        if self._on_step_renamed:
            self._on_step_renamed(self.path, new_name)

    def _on_run_clicked(self, path, name, on_run):
        if not path.startswith(STEP_PREFIX):
            self._run_btn.set_running(True)
        QApplication.processEvents()
        try:
            on_run(path, name)
        finally:
            self._flash_green()

    def _flash_green(self):
        """Briefly highlight the row green then restore."""
        self.setStyleSheet("background-color: #2d6a2d;")
        QTimer.singleShot(600, self._clear_flash)

    def _clear_flash(self):
        self.setStyleSheet("")
        if not self.path.startswith(STEP_PREFIX):
            self._run_btn.setEnabled(True)
            self._run_btn.set_running(False)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.pos()

    def mouseMoveEvent(self, event):
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        if (event.pos() - self._drag_start).manhattanLength() < QApplication.startDragDistance():
            return
        drag = make_drag(self, SRC_RIGHT, self.path)
        drag.exec(Qt.DropAction.MoveAction)

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        copy_action = menu.addAction("Copy")
        action = menu.exec(event.globalPos())
        if action == copy_action and self._on_copy:
            self._on_copy(self.path)



class WorkflowCombo(QComboBox):
    """QComboBox with a right-click context menu for Export, Import and Delete."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._on_export = None
        self._on_import = None
        self._on_delete = None

    def set_callbacks(self, on_export, on_import, on_delete=None):
        self._on_export = on_export
        self._on_import = on_import
        self._on_delete = on_delete

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        export_action = menu.addAction("Export Workflow…")
        import_action = menu.addAction("Import Workflow…")
        menu.addSeparator()
        delete_action = menu.addAction("Delete Workflow")
        action = menu.exec(event.globalPos())
        if action == export_action and self._on_export:
            self._on_export()
        elif action == import_action and self._on_import:
            self._on_import()
        elif action == delete_action and self._on_delete:
            self._on_delete()


# ── Right panel ───────────────────────────────────────────────────────────────

class SelectedPanel(QWidget):
    """Right panel: ordered queue of selected scripts.
    Accepts drops from the left (add) and supports internal reordering.
    Emits SRC_RIGHT drags that the left panel catches to remove."""

    def __init__(self, checked_paths: list, on_run, on_changed,
                 favorites: set = None, on_fav_toggle=None, on_remove=None,
                 on_copy=None, missing_paths: set = None,
                 notes: dict = None, on_notes_changed=None,
                 on_step_renamed=None, completed_paths: set = None,
                 on_completed_toggle=None, parent=None):
        super().__init__(parent)
        self.checked_paths = checked_paths
        self.on_run = on_run
        self.on_changed = on_changed  # called after any add/remove/reorder
        self.favorites = favorites if favorites is not None else set()
        self.on_fav_toggle = on_fav_toggle
        self.on_remove = on_remove
        self.on_copy = on_copy
        self.missing_paths: set = missing_paths if missing_paths is not None else set()
        self.notes: dict = notes if notes is not None else {}
        self.on_notes_changed = on_notes_changed
        self.on_step_renamed = on_step_renamed
        self.completed_paths: set = completed_paths if completed_paths is not None else set()
        self.on_completed_toggle = on_completed_toggle

        self.setAcceptDrops(True)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)

        title = QLabel("Selected Scripts")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(title)

        hint = QLabel("Drag available scripts here to add")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setWordWrap(True)
        outer.addWidget(hint)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(self._scroll)

        self._clear_completed_btn = QPushButton("Clear Completed")
        self._clear_completed_btn.setToolTip("Unmark all completed items")
        self._clear_completed_btn.setCursor(Qt.CursorShape.ArrowCursor)
        outer.addWidget(self._clear_completed_btn)

        # Workflow combo sits below the script list, above the footer buttons
        workflows_lbl = QLabel("Workflows")
        workflows_lbl.setAlignment(Qt.AlignmentFlag.AlignLeft)
        outer.addWidget(workflows_lbl)
        self._workflow_combo = WorkflowCombo()
        self._workflow_combo.setToolTip("Load a saved workflow — right-click for Export/Import")
        outer.addWidget(self._workflow_combo)

        self._container = QWidget()
        self._rows_layout = QVBoxLayout(self._container)
        self._rows_layout.setContentsMargins(4, 4, 4, 4)
        self._rows_layout.setSpacing(2)
        self._rows_layout.addStretch()
        self._scroll.setWidget(self._container)


        self._empty_min_width = 350  # px — generous drop-zone when empty
        self.rebuild()

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        if not self.checked_paths:
            from PyQt6.QtCore import QSize
            return QSize(self._empty_min_width, hint.height())
        return hint

    def sizeHint(self):
        hint = super().sizeHint()
        if not self.checked_paths:
            from PyQt6.QtCore import QSize
            return QSize(self._empty_min_width, hint.height())
        return hint

    def rebuild(self):
        while self._rows_layout.count() > 1:
            item = self._rows_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        # Favorites float to the top
        favs = self.favorites if hasattr(self, "favorites") else set()
        fav_paths = [p for p in self.checked_paths if p in favs]
        other_paths = [p for p in self.checked_paths if p not in favs]
        pos = 0
        for path in fav_paths:
            if path.startswith(STEP_PREFIX):
                _label = path[len(STEP_PREFIX):]
                # Strip unique copy suffix (#xxxxxx) for display
                _name = _label.rsplit("#", 1)[0] if "#" in _label and len(_label.rsplit("#", 1)[-1]) == 6 else _label
            else:
                _name = os.path.basename(path)
            row = RightScriptRow(path, _name, self.on_run,
                                 is_fav=True, on_fav_toggle=self.on_fav_toggle,
                                 on_remove=self.on_remove, on_copy=self.on_copy,
                                 missing=path in self.missing_paths,
                                 note=self.notes.get(path, ""),
                                 on_notes_changed=self.on_notes_changed,
                                 on_step_renamed=self.on_step_renamed,
                                 completed=path in self.completed_paths,
                                 on_completed_toggle=self.on_completed_toggle)
            self._rows_layout.insertWidget(pos, row)
            pos += 1
        # Divider only when both groups have entries
        if fav_paths and other_paths:
            divider = QFrame()
            divider.setFrameShape(QFrame.Shape.HLine)
            divider.setFrameShadow(QFrame.Shadow.Sunken)
            self._rows_layout.insertWidget(pos, divider)
            pos += 1
        for path in other_paths:
            if path.startswith(STEP_PREFIX):
                _label = path[len(STEP_PREFIX):]
                # Strip unique copy suffix (#xxxxxx) for display
                _name = _label.rsplit("#", 1)[0] if "#" in _label and len(_label.rsplit("#", 1)[-1]) == 6 else _label
            else:
                _name = os.path.basename(path)
            row = RightScriptRow(path, _name, self.on_run,
                                 is_fav=False, on_fav_toggle=self.on_fav_toggle,
                                 on_remove=self.on_remove, on_copy=self.on_copy,
                                 missing=path in self.missing_paths,
                                 note=self.notes.get(path, ""),
                                 on_notes_changed=self.on_notes_changed,
                                 on_step_renamed=self.on_step_renamed,
                                 completed=path in self.completed_paths,
                                 on_completed_toggle=self.on_completed_toggle)
            self._rows_layout.insertWidget(pos, row)
            pos += 1

    # ── drop handling ─────────────────────────────────────────────────────────

    def dragEnterEvent(self, event):
        if parse_mime(event):
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if parse_mime(event):
            event.acceptProposedAction()

    def dropEvent(self, event):
        parsed = parse_mime(event)
        if not parsed:
            return
        source_id, path = parsed

        # Map drop position into the scroll viewport, then into container coords
        pos_in_viewport = self._scroll.viewport().mapFromGlobal(
            self.mapToGlobal(event.position().toPoint())
        )
        idx = drop_index_in_panel(self._scroll, self._rows_layout, pos_in_viewport)

        if source_id == SRC_LEFT:
            # Add from left panel — ignore if already present
            if path in self.checked_paths:
                event.acceptProposedAction()
                return
            # idx is already the correct insertion point (clamped inside helper)
            self.checked_paths.insert(idx, path)

        elif source_id == SRC_RIGHT:
            if path not in self.checked_paths:
                return

            # The layout contains: [fav rows] [divider?] [other rows] [stretch]
            # We need to convert the raw layout drop index into a checked_paths index.
            favs = self.favorites if hasattr(self, "favorites") else set()
            fav_paths   = [p for p in self.checked_paths if p in favs]
            other_paths = [p for p in self.checked_paths if p not in favs]
            has_divider = bool(fav_paths and other_paths)

            # Build an ordered list of paths in the same order as layout rows
            ordered = fav_paths + other_paths

            # idx is a layout position; subtract 1 for divider if drop is past it
            divider_layout_pos = len(fav_paths) if has_divider else -1
            list_idx = idx
            if has_divider and idx > divider_layout_pos:
                list_idx = idx - 1  # skip past divider widget

            # Clamp to valid range
            list_idx = max(0, min(list_idx, len(ordered)))

            # Remove from ordered list and reinsert
            src_ordered_idx = ordered.index(path)
            ordered.pop(src_ordered_idx)
            dst_ordered_idx = list_idx if list_idx <= src_ordered_idx else list_idx - 1
            dst_ordered_idx = max(0, min(dst_ordered_idx, len(ordered)))
            ordered.insert(dst_ordered_idx, path)

            # Write back to checked_paths preserving the new order
            self.checked_paths[:] = ordered

        self.on_changed()
        self.rebuild()
        event.acceptProposedAction()


# ── Main window ───────────────────────────────────────────────────────────────

class ScriptLauncherWindow(QMainWindow):

    def __init__(self, scripts_dir: str, scripts: list[tuple[str, str]]):
        super().__init__()
        self.scripts_dir = scripts_dir
        self.scripts = scripts
        self._siril_wd = os.getcwd()  # Siril sets cwd to its home dir before launching

        valid_paths = {p for _, p in self.scripts}
        _sel, _favs, _geom, _workflows, _active, _notes = load_state()
        self.missing_paths: set = {p for p in _sel
                                   if p not in valid_paths
                                   and not p.startswith(STEP_PREFIX)}
        self.checked_paths: list = list(_sel)  # keep all, including missing
        self.favorites: set = {p for p in _favs if p in valid_paths or p.startswith(STEP_PREFIX)}
        self._saved_geometry: dict | None = _geom
        self.workflows: dict = _workflows
        self._active_workflow: str = _active
        self._workflow_dirty: bool = False
        self.notes: dict = _notes  # {path: note_text}
        self.completed_paths: set = set()  # reset each session — loaded per workflow

        self.setWindowTitle(f"Deep Space Astro - Script Launcher  v{VERSION}")
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        _icon_pm = QPixmap()
        _icon_pm.loadFromData(base64.b64decode(ICON_B64))
        _icon = QIcon(_icon_pm)
        self.setWindowIcon(_icon)
        QApplication.instance().setWindowIcon(_icon)

        self._build_ui()
        self._right_panel._clear_completed_btn.clicked.connect(self._clear_completed)
        self._right_panel._workflow_combo.activated.connect(self._load_workflow)
        self._right_panel._workflow_combo.set_callbacks(
            self._export_workflow, self._import_workflow, self._delete_workflow
        )
        self._rebuild_workflow_combo()
        if self._active_workflow:
            combo = self._right_panel._workflow_combo
            idx = combo.findText(self._active_workflow)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            # Restore completed state from the active workflow
            wf = self.workflows.get(self._active_workflow, {})
            self.completed_paths = set(wf.get("completed", []))
            self._right_panel.completed_paths = self.completed_paths
            self._right_panel.rebuild()

        # Check for missing scripts in the restored selection
        # Use raw _sel from state file (before valid_paths filtering)
        _valid = {p for _, p in self.scripts}
        _raw_sel = _sel  # captured from load_state() above
        _missing = [os.path.basename(p) for p in _raw_sel
                    if p not in _valid and not p.startswith(STEP_PREFIX)]
        if _missing:
            _ctx = (f"From workflow '{self._active_workflow}':\n"
                    if self._active_workflow else "")
            QTimer.singleShot(
                100,
                lambda m=_missing, c=_ctx: self._warn_missing_scripts(m, context=c)
            )

        if self._saved_geometry:
            try:
                from PyQt6.QtCore import QRect
                g = self._saved_geometry
                rect = QRect(g["x"], g["y"], g["w"], g["h"])
                self.setGeometry(rect)
                # Pre-seed so Show Scripts restores the correct expanded geometry
                self._expanded_geometry = rect
            except Exception:
                pass

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(4)

        # ── top bar (search + On Top on one line) ─────────────────────────
        top = QHBoxLayout()
        self._search_lbl = QLabel("Search:")
        top.addWidget(self._search_lbl)
        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Filter scripts…")
        self._search_input.setToolTip("Type to filter the Available Scripts list by name")
        self._search_input.textChanged.connect(self._apply_filter)
        top.addWidget(self._search_input)
        top.addStretch()
        self._ontop_chk = QCheckBox("On Top")
        self._ontop_chk.setChecked(True)
        self._ontop_chk.setToolTip("Keep window above other windows")
        self._ontop_chk.toggled.connect(self.toggle_ontop)
        top.addWidget(self._ontop_chk)
        root.addLayout(top)

        # ── splitter ──────────────────────────────────────────────────────────
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(self._splitter, stretch=1)

        # LEFT
        self._left_panel = ScriptsPanel(
            self.scripts_dir,
            self.scripts,
            self.checked_paths,
            on_remove=self._remove_script,
            on_add=self._add_script,
        )
        self._splitter.addWidget(self._left_panel)

        # RIGHT
        self._right_panel = SelectedPanel(
            self.checked_paths,
            on_run=self._run_one,
            on_changed=self._on_queue_changed,
            favorites=self.favorites,
            on_fav_toggle=self._on_fav_toggle,
            on_remove=self._remove_script,
            on_copy=self._copy_script,
            missing_paths=self.missing_paths,
            notes=self.notes,
            on_notes_changed=self._on_notes_changed,
            on_step_renamed=self._on_step_renamed,
            completed_paths=self.completed_paths,
            on_completed_toggle=self._on_completed_toggle,
        )
        self._splitter.addWidget(self._right_panel)

        # ── footer ────────────────────────────────────────────────────────────
        self._status_label = QLabel("")
        root.addWidget(self._status_label)

        btn_row = QHBoxLayout()
        self._toggle_btn = QPushButton("Hide Scripts")
        self._toggle_btn.setToolTip("Hide the Available Scripts panel")
        self._toggle_btn.clicked.connect(self._toggle_left)
        btn_row.addWidget(self._toggle_btn)
        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.setToolTip("Rescan the scripts directory for new or removed scripts")
        self._refresh_btn.clicked.connect(self._refresh)
        btn_row.addWidget(self._refresh_btn)
        btn_row.addStretch()
        self._add_step_btn = QPushButton("Add Step")
        self._add_step_btn.setToolTip("Add a manual reminder step to the right panel")
        self._add_step_btn.clicked.connect(self._add_manual_step)
        btn_row.addWidget(self._add_step_btn)
        self._save_workflow_btn = QPushButton("Save")
        self._save_workflow_btn.setToolTip("Save current scripts and favorites as a named workflow")
        self._save_workflow_btn.clicked.connect(self._save_workflow)
        btn_row.addWidget(self._save_workflow_btn)
        self._clear_btn = QPushButton("New")
        self._clear_btn.setToolTip("Clear the right panel to start a new workflow")
        self._clear_btn.clicked.connect(self._clear_all)
        btn_row.addWidget(self._clear_btn)
        help_btn = QPushButton("Help")
        help_btn.setToolTip("Show usage instructions")
        help_btn.clicked.connect(self._show_help)
        btn_row.addWidget(help_btn)
        close_btn = QPushButton("Close")
        close_btn.setToolTip("Close the Script Launcher")
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

        self._update_status()

    # ── queue callbacks ───────────────────────────────────────────────────────

    def _on_queue_changed(self):
        """Called after any add/reorder in the right panel."""
        self._workflow_dirty = True
        self._left_panel.rebuild()   # refresh dim state
        self._update_status()

    def _on_fav_toggle(self, path: str, is_fav: bool):
        if is_fav:
            self.favorites.add(path)
        else:
            self.favorites.discard(path)
        self._workflow_dirty = True
        self._right_panel.rebuild()  # re-sort so favs float to top

    def _add_manual_step(self):
        """Prompt for a label and add a custom manual reminder step to the right panel."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Add Manual Step")
        dlg.setMinimumWidth(300)
        dlg.setWindowFlags(dlg.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        layout = QVBoxLayout(dlg)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 8)
        layout.addWidget(QLabel("Step label (e.g. Crop, Plate Solve):"))
        name_edit = QLineEdit()
        name_edit.setPlaceholderText("Enter step name…")
        layout.addWidget(name_edit)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        label = name_edit.text().strip()
        if not label:
            return
        path = f"{STEP_PREFIX}{label}"
        self.checked_paths.append(path)
        self._workflow_dirty = True
        self._right_panel.rebuild()
        self._update_status()

    def _on_step_renamed(self, old_path: str, new_name: str):
        """Rename a manual step in checked_paths."""
        # Preserve any unique copy suffix from the old path
        old_label = old_path[len(STEP_PREFIX):]
        if "#" in old_label and len(old_label.rsplit("#", 1)[-1]) == 6:
            suffix = "#" + old_label.rsplit("#", 1)[-1]
        else:
            suffix = ""
        new_path = f"{STEP_PREFIX}{new_name}{suffix}"
        # Each step copy now has a unique path, so replace all occurrences safely
        for i, p in enumerate(self.checked_paths):
            if p == old_path:
                self.checked_paths[i] = new_path
        # Move notes if any
        if old_path in self.notes:
            self.notes[new_path] = self.notes.pop(old_path)
            self._right_panel.notes = self.notes
        # Move favorites if any
        if old_path in self.favorites:
            self.favorites.discard(old_path)
            self.favorites.add(new_path)
        self._workflow_dirty = True
        self._right_panel.rebuild()
        self._update_status()

    def _on_completed_toggle(self, path: str, is_completed: bool):
        if is_completed:
            self.completed_paths.add(path)
        else:
            self.completed_paths.discard(path)
        self._right_panel.completed_paths = self.completed_paths
        self._workflow_dirty = True

    def _clear_completed(self):
        self.completed_paths.clear()
        self._right_panel.completed_paths = self.completed_paths
        self._workflow_dirty = True
        self._right_panel.rebuild()

    def _on_notes_changed(self, path: str, note: str):
        """Save updated note for a script."""
        if note:
            self.notes[path] = note
        else:
            self.notes.pop(path, None)
        self._workflow_dirty = True

    def _add_script(self, path: str):
        """Called on double-click in the left panel."""
        if path not in self.checked_paths:
            self.checked_paths.append(path)
            self._workflow_dirty = True
            self._right_panel.rebuild()
            self._left_panel.rebuild()
            self._update_status()

    def _remove_script(self, path: str):
        """Called when a script is dragged from right back to left, or removed via trash."""
        if path in self.checked_paths:
            self.checked_paths.remove(path)
        self.favorites.discard(path)
        self._workflow_dirty = True
        self._right_panel.rebuild()
        self._left_panel.rebuild()
        self._update_status()

    def _copy_script(self, path: str):
        """Insert a duplicate of path immediately after its last occurrence."""
        last_idx = max(i for i, p in enumerate(self.checked_paths) if p == path)
        # For steps, create a unique path so each copy can be renamed independently
        if path.startswith(STEP_PREFIX):
            label = path[len(STEP_PREFIX):]
            copy_path = f"{STEP_PREFIX}{label}#{uuid.uuid4().hex[:6]}"
        else:
            copy_path = path
        self.checked_paths.insert(last_idx + 1, copy_path)
        self._workflow_dirty = True
        self._right_panel.rebuild()
        self._update_status()

    def _rebuild_workflow_combo(self):
        """Repopulate the workflow dropdown from self.workflows."""
        combo = self._right_panel._workflow_combo
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("— Load Workflow —")
        for name in sorted(self.workflows.keys()):
            combo.addItem(name)
        # Make the placeholder non-selectable when real workflows exist
        if self.workflows:
            model = combo.model()
            item = model.item(0)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
        combo.setCurrentIndex(0)
        combo.blockSignals(False)

    def _save_workflow(self):
        """Prompt for a name and save current scripts+favorites as a workflow."""
        if not self.checked_paths:
            QMessageBox.information(self, "Save Workflow",
                                    "No scripts are selected to save.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Save Workflow")
        dlg.setMinimumWidth(320)
        dlg.setWindowFlags(dlg.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        layout = QVBoxLayout(dlg)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 8)

        layout.addWidget(QLabel("Workflow name:"))
        name_edit = QLineEdit()
        name_edit.setPlaceholderText("Enter a name…")
        if self._active_workflow:
            name_edit.setText(self._active_workflow)
        layout.addWidget(name_edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        name = name_edit.text().strip()
        if not name:
            return

        if name in self.workflows:
            reply = QMessageBox.question(
                self, "Overwrite Workflow",
                f"Workflow '{name}' already exists. Overwrite it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        self.workflows[name] = {
            "selections": list(self.checked_paths),
            "favorites":  sorted(self.favorites),
            "completed":  sorted(self.completed_paths),
        }
        self._active_workflow = name
        self._workflow_dirty = False
        save_state(self.checked_paths, self.favorites, workflows=self.workflows,
                   active_workflow=name, notes=self.notes)
        self._rebuild_workflow_combo()
        combo = self._right_panel._workflow_combo
        idx = combo.findText(name)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        self._status_label.setText(f"Workflow '{name}' saved.")

    def _warn_missing_scripts(self, missing_names: list, context: str = "",
                               confirm: bool = False) -> bool:
        """Warn about missing scripts.

        If confirm=True, asks whether to continue loading and returns the
        user's choice (True = continue, False = cancel).
        If confirm=False, shows a plain info dialog and returns True.
        """
        if not missing_names:
            return True
        names_list = "\n".join(f"  • {n}" for n in missing_names)
        msg = (
            f"{context}The following script(s) were not found in your "
            f"siril-scripts folder:\n\n"
            f"{names_list}\n\n"
            "To use them, install them via Get Scripts in Siril."
        )
        if confirm:
            msg += "\n\nContinue loading the workflow without these scripts?"
            box = QMessageBox(self)
            box.setWindowTitle("Scripts Not Installed")
            box.setText(msg)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setStandardButtons(
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            box.setDefaultButton(QMessageBox.StandardButton.No)
            return box.exec() == QMessageBox.StandardButton.Yes
        QMessageBox.warning(self, "Scripts Not Installed", msg)
        return True

    def _load_workflow(self, index: int):
        """Load the workflow selected in the dropdown."""
        if index == 0:
            return
        name = self._right_panel._workflow_combo.itemText(index)
        workflow = self.workflows.get(name)
        if not workflow:
            return
        valid_paths = {p for _, p in self.scripts}
        raw_paths = workflow.get("selections", [])
        missing_paths = [p for p in raw_paths
                         if p not in valid_paths and not p.startswith(STEP_PREFIX)]
        # Inform user of any missing scripts (load continues regardless)
        if missing_paths:
            missing_names = [os.path.basename(p) for p in missing_paths]
            self._warn_missing_scripts(
                missing_names,
                context=f"From workflow '{name}':\n"
            )
        loaded = [p for p in raw_paths if p in valid_paths or p.startswith(STEP_PREFIX)]
        raw_favs = workflow.get("favorites", [])
        loaded_favs = {p for p in raw_favs if p in valid_paths or p.startswith(STEP_PREFIX)}
        self.missing_paths = {p for p in raw_paths
                              if p not in valid_paths and not p.startswith(STEP_PREFIX)}
        self._right_panel.missing_paths = self.missing_paths
        raw_completed = workflow.get("completed", [])
        self.completed_paths = set(raw_completed)
        self._right_panel.completed_paths = self.completed_paths
        self.checked_paths.clear()
        self.checked_paths.extend(raw_paths)  # keep all including missing
        self.favorites.clear()
        self.favorites.update(loaded_favs)
        self._active_workflow = name
        self._workflow_dirty = False
        save_state(self.checked_paths, self.favorites, active_workflow=name)
        self._right_panel.rebuild()
        self._left_panel.rebuild()
        self._update_status()
        # Keep the loaded workflow name selected in the dropdown
        self._right_panel._workflow_combo.setCurrentIndex(index)
        self._status_label.setText(f"Workflow '{name}' loaded.")
        # If the left panel is hidden, resize window to fit the new content
        if not self._left_panel.isVisible():
            QTimer.singleShot(0, self._shrink_to_right_panel)


    def _export_workflow(self):
        """Export the currently selected workflow to a standalone JSON file for sharing.

        Script paths are stored as forward-slash relative paths from the siril-scripts
        root so the file works on Windows, macOS and Linux.
        """
        if not self.workflows:
            QMessageBox.information(self, "Export Workflow",
                                    "No workflows have been saved yet.")
            return

        # Use whichever workflow is currently selected in the dropdown
        combo = self._right_panel._workflow_combo
        idx = combo.currentIndex()
        name = combo.itemText(idx) if idx > 0 else self._active_workflow
        if not name or name not in self.workflows:
            QMessageBox.information(self, "Export Workflow",
                                    "Please select a workflow from the dropdown first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, f"Export Workflow — {name}", f"{name}.json",
            "JSON Workflow Files (*.json);;All Files (*)"
        )
        if not path:
            return
        if not path.endswith(".json"):
            path += ".json"
        try:
            scripts_root = os.path.normpath(self.scripts_dir)

            def to_relative(p: str) -> str:
                """Convert absolute path to forward-slash relative from scripts root.
                Steps (step://) and already-relative paths are returned unchanged."""
                if p.startswith(STEP_PREFIX):
                    return p
                try:
                    rel = os.path.relpath(p, scripts_root)
                    return rel.replace("\\", "/")
                except ValueError:
                    # relpath fails across Windows drives — keep absolute as fallback
                    return p.replace("\\", "/")

            wf = self.workflows[name]
            rel_selections = [to_relative(p) for p in wf["selections"]]
            rel_favorites  = [to_relative(p) for p in wf["favorites"]]

            with open(path, "w", encoding="utf-8") as fh:
                json.dump({
                    "workflow_name": name,
                    "selections":    rel_selections,
                    "favorites":     rel_favorites,
                    "portable":      True,
                    # "completed" intentionally omitted — per-session state
                }, fh, indent=2)
            self._status_label.setText(
                f"Workflow '{name}' exported to {os.path.basename(path)}."
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", str(e))

    def _import_workflow(self):
        """Import a workflow from a JSON file shared by another user.

        Handles both portable (relative paths) and legacy (absolute paths) files.
        Relative paths are resolved against this machine's siril-scripts directory.
        """
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Workflow", "",
            "JSON Workflow Files (*.json);;All Files (*)"
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict) or "selections" not in data:
                raise ValueError("Invalid workflow file format.")

            # Use embedded name, fall back to filename
            name = data.get("workflow_name", "") or \
                   os.path.splitext(os.path.basename(path))[0]
            name = name.strip() or "Imported Workflow"

            portable = data.get("portable", False)
            scripts_root = os.path.normpath(self.scripts_dir)

            def to_absolute(p: str) -> str:
                """Resolve a path from the export file to an absolute local path."""
                if p.startswith(STEP_PREFIX):
                    return p
                # If portable, treat as relative to local siril-scripts root
                if portable or not os.path.isabs(p):
                    # Normalise separators then join
                    rel = p.replace("/", os.sep).replace("\\", os.sep)
                    return os.path.join(scripts_root, rel)
                return p  # legacy absolute path — use as-is

            valid_paths = {p for _, p in self.scripts}
            raw_paths   = [to_absolute(p) for p in data["selections"]]
            raw_favs    = [to_absolute(p) for p in data.get("favorites", [])]

            loaded      = [p for p in raw_paths if p in valid_paths or p.startswith(STEP_PREFIX)]
            missing_paths = [p for p in raw_paths
                             if p not in valid_paths and not p.startswith(STEP_PREFIX)]
            loaded_favs = [p for p in raw_favs if p in valid_paths or p.startswith(STEP_PREFIX)]

            # If name already exists, ask to overwrite
            if name in self.workflows:
                reply = QMessageBox.question(
                    self, "Import Workflow",
                    f"A workflow named '{name}' already exists. Overwrite it?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                if reply != QMessageBox.StandardButton.Yes:
                    return

            if missing_paths:
                missing_names = [os.path.basename(p) for p in missing_paths]
                if not self._warn_missing_scripts(
                    missing_names,
                    context=f"From imported workflow '{name}':\n",
                    confirm=True
                ):
                    return

            self.workflows[name] = {
                "selections": loaded,
                "favorites":  sorted(loaded_favs),
            }
            # Load it immediately into the right panel
            self.missing_paths = {p for p in raw_paths
                                  if p not in valid_paths and not p.startswith(STEP_PREFIX)}
            self._right_panel.missing_paths = self.missing_paths
            self.checked_paths.clear()
            self.checked_paths.extend(raw_paths)  # keep all including missing
            self.favorites.clear()
            self.favorites.update(set(loaded_favs))
            self._active_workflow = name
            self._workflow_dirty = False
            save_state(self.checked_paths, self.favorites,
                       workflows=self.workflows, active_workflow=name)
            self._rebuild_workflow_combo()
            combo = self._right_panel._workflow_combo
            idx = combo.findText(name)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            self._right_panel.rebuild()
            self._left_panel.rebuild()
            self._update_status()
            self._status_label.setText(f"Workflow '{name}' imported and loaded.")
            if not self._left_panel.isVisible():
                QTimer.singleShot(0, self._shrink_to_right_panel)
        except Exception as e:
            QMessageBox.critical(self, "Import Failed", str(e))

    def _delete_workflow(self):
        """Delete the currently selected workflow after confirmation."""
        combo = self._right_panel._workflow_combo
        idx = combo.currentIndex()
        name = combo.itemText(idx) if idx > 0 else self._active_workflow
        if not name or name not in self.workflows:
            QMessageBox.information(self, "Delete Workflow",
                                    "Please select a workflow to delete.")
            return
        reply = QMessageBox.question(
            self, "Delete Workflow",
            f"Delete workflow '{name}'? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self.workflows.pop(name, None)
        if self._active_workflow == name:
            self._active_workflow = ""
            self._workflow_dirty = False
            self.checked_paths.clear()
            self.missing_paths.clear()
            self._right_panel.missing_paths = self.missing_paths
            self.completed_paths.clear()
            self._right_panel.completed_paths = self.completed_paths
            self.favorites.clear()
        save_state(self.checked_paths, self.favorites,
                   workflows=self.workflows, active_workflow=self._active_workflow)
        self._rebuild_workflow_combo()
        self._right_panel.rebuild()
        self._left_panel.rebuild()
        self._update_status()
        self._status_label.setText(f"Workflow '{name}' deleted.")

    def _clear_all(self):
        if self._workflow_dirty:
            wf = self._active_workflow
            msg = (
                f"Workflow '{wf}' has unsaved changes.\nStart a new workflow without saving?"
                if wf else
                "You have unsaved changes.\nClear the panel without saving?"
            )
            reply = QMessageBox.question(
                self, "Unsaved Changes", msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        self.checked_paths.clear()
        self.missing_paths.clear()
        self._right_panel.missing_paths = self.missing_paths
        self.completed_paths.clear()
        self._right_panel.completed_paths = self.completed_paths
        self.favorites.clear()
        self._active_workflow = ""
        self._workflow_dirty = False
        save_state(self.checked_paths, self.favorites, active_workflow="")
        self._rebuild_workflow_combo()
        self._right_panel.rebuild()
        self._left_panel.rebuild()
        self._update_status()

    # ── filter / refresh ──────────────────────────────────────────────────────

    def _visible_scripts(self) -> list[tuple[str, str]]:
        query = self._search_input.text().lower()
        if not query:
            return self.scripts
        return [(lbl, p) for lbl, p in self.scripts if query in lbl.lower()]

    def _apply_filter(self):
        self._left_panel.rebuild(scripts=self._visible_scripts())

    def _refresh(self):
        self.scripts = scan_scripts(self.scripts_dir)
        valid_paths = {p for _, p in self.scripts}
        # Update missing_paths — don't remove anything from checked_paths
        self.missing_paths = {p for p in self.checked_paths
                              if p not in valid_paths and not p.startswith(STEP_PREFIX)}
        self._right_panel.missing_paths = self.missing_paths
        self._search_input.clear()
        self._left_panel.rebuild(scripts=self.scripts)
        self._right_panel.rebuild()
        self._update_status()

    # ── status ────────────────────────────────────────────────────────────────

    def _update_status(self):
        visible = len(self._visible_scripts())
        total = len(self.scripts)
        queued = len(self.checked_paths)
        if visible < total:
            self._status_label.setText(
                f"Showing {visible} of {total} scripts  •  {queued} selected"
            )
        else:
            self._status_label.setText(
                f"{total} scripts  •  {queued} selected"
            )

    # ── run ───────────────────────────────────────────────────────────────────

    def _get_siril_wd(self) -> str:
        """Get Siril's current working directory by connecting briefly.
        Falls back to the directory captured at startup if connection fails."""
        try:
            _siril = s.SirilInterface()
            _siril.connect()
            wd = _siril.get_siril_wd()
            _siril.disconnect()
            return wd
        except Exception:
            return self._siril_wd  # fall back to startup cwd

    def _run_one(self, script_path: str, name: str):
        self._status_label.setText(f"Launching {name}…")
        QApplication.processEvents()

        try:
            # Use Siril's own venv Python — same interpreter Siril uses when
            # launching scripts from its Scripts menu, giving the script a
            # fresh process with its own IPC connection to Siril.
            home = os.path.expanduser("~")
            if sys.platform == "win32":
                venv_python = os.path.join(
                    os.environ.get("LOCALAPPDATA", home),
                    "siril", "venv", "Scripts", "python.exe"
                )
            elif sys.platform == "darwin":
                venv_python = os.path.join(
                    home, "Library", "Application Support",
                    "org.siril.Siril", "venv", "bin", "python"
                )
            else:  # Linux
                venv_python = os.path.join(
                    home, ".local", "share", "siril", "venv", "bin", "python"
                )
            python_exe = venv_python if os.path.isfile(venv_python) else sys.executable
            proc = subprocess.Popen(
                [python_exe, script_path],
                cwd=self._get_siril_wd(),
            )
            self._status_label.setText(f"{name} running…")
            self._watch_process(proc, name)
        except Exception as e:
            QMessageBox.critical(self, "Launch Error",
                                 f"Could not launch '{name}':\n\n{str(e)}")
            self._status_label.setText(f"Failed to launch {name}.")

    def _watch_process(self, proc: subprocess.Popen, name: str):
        """Poll proc every second; update status when it exits."""
        timer = QTimer(self)
        timer.setInterval(1000)

        def _check():
            if proc.poll() is not None:
                timer.stop()
                timer.deleteLater()
                self._status_label.setText(f"{name} closed.")
                QTimer.singleShot(4000, self._update_status)

        timer.timeout.connect(_check)
        timer.start()

    def _show_help(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Deep Space Astro - Script Launcher — Help")
        dlg.setMinimumWidth(560)
        dlg.setWindowFlags(dlg.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)

        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(16, 16, 16, 12)
        layout.setSpacing(8)

        text = QTextEdit()
        text.setReadOnly(True)
        text.setFrameShape(QFrame.Shape.NoFrame)
        text.setHtml(
            "<style>h3{margin:8px 0 2px 0;} p{margin:0 0 0 12px;} "
            "body{white-space:nowrap;}</style>"
            "<h3>BROWSING SCRIPTS</h3>"
            "<p>&#8226; Scripts are grouped by subfolder in the left panel.</p>"
            "<p>&#8226; Click a group header to expand or collapse it.</p>"
            "<p>&#8226; Use the Search box to filter scripts by name.</p>"
            "<p>&#8226; Hover over a script to see its description.</p>"
            "<h3>ADDING SCRIPTS</h3>"
            "<p>&#8226; Drag a script from the left panel to the right panel to add it.</p>"
            "<p>&#8226; Or double-click a script to add it instantly.</p>"
            "<p>&#8226; Added scripts disappear from the left panel and reappear when removed.</p>"
            "<h3>REMOVING SCRIPTS</h3>"
            "<p>&#8226; Drag a script from the right panel back to the left panel to remove it.</p>"
            "<p>&#8226; Or click the &#128465; trash button on any row and confirm the prompt.</p>"
            "<h3>REORDERING &amp; DUPLICATING</h3>"
            "<p>&#8226; Drag scripts up or down within the right panel to change their order.</p>"
            "<p>&#8226; Right-click a script in the right panel and choose Copy to duplicate it.</p>"
            "<h3>MANUAL STEPS</h3>"
            "<p>&#8226; Click Add Step to add a reminder for a built-in Siril action (e.g. Crop).</p>"
            "<p>&#8226; Manual steps show a &#128994; green circle instead of a play button.</p>"
            "<p>&#8226; Double-click a step&#39;s label to rename it.</p>"
            "<p>&#8226; Or click the pencil button on a step to rename it and add notes together.</p>"
            "<p>&#8226; Steps can be reordered, starred, completed, and removed like any script.</p>"
            "<h3>NOTES</h3>"
            "<p>&#8226; Click the pencil button on any row to add or edit a note.</p>"
            "<p>&#8226; The pencil turns blue when a note exists.</p>"
            "<p>&#8226; Notes are saved with workflows.</p>"
            "<h3>FAVORITES</h3>"
            "<p>&#8226; Click the &#9734; star button on any row to mark it as a favorite.</p>"
            "<p>&#8226; Favorites are pinned to the top of the right panel above a divider line.</p>"
            "<p>&#8226; Favorites are saved with workflows.</p>"
            "<h3>LAUNCHING SCRIPTS</h3>"
            "<p>&#8226; Click the &#9654; play button on any row to launch that script.</p>"
            "<p>&#8226; The status bar shows when a script is running and when it has closed.</p>"
            "<p>&#8226; Multiple scripts can be running simultaneously.</p>"
            "<p>&#8226; Scripts not found on disk are shown disabled with a tooltip.</p>"
            "<h3>COMPLETING ITEMS</h3>"
            "<p>&#8226; Click the &#10003; check button on any row to mark it as done.</p>"
            "<p>&#8226; Completed items show a strikethrough label and a blue check button.</p>"
            "<p>&#8226; Click Clear Completed (above the Workflows menu) to reset all checkmarks.</p>"
            "<p>&#8226; Completed state is saved with workflows but not included in exports.</p>"
            "<h3>WORKFLOWS</h3>"
            "<p>&#8226; Click Save to name and save your scripts, steps, notes, favorites, and completed state.</p>"
            "<p>&#8226; Click New to clear the right panel and start fresh (prompts if unsaved changes exist).</p>"
            "<p>&#8226; Use the dropdown in the right panel to load a saved workflow.</p>"
            "<p>&#8226; Right-click the dropdown to Export, Import, or Delete a workflow.</p>"
            "<p>&#8226; Exporting saves a workflow to a .json file you can share with others.</p>"
            "<p>&#8226; Importing loads a workflow from a file and makes it the active workflow.</p>"
            "<p>&#8226; Any unsaved changes prompt a confirmation when closing.</p>"
            "<p>&#8226; Workflows are stored in script_launcher_state.json.</p>"
            "<h3>OTHER</h3>"
            "<p>&#8226; Click Hide Scripts / Show Scripts to toggle the left panel.</p>"
            "<p>&#8226; Click Refresh to rescan for new or removed scripts without clearing the right panel.</p>"
            "<p>&#8226; Check On Top to keep the launcher above other windows.</p>"
            "<p>&#8226; Window size and position are saved automatically between sessions.</p>"
        )
        layout.addWidget(text)

        btn_row = QHBoxLayout()
        release_btn = QPushButton("Release Notes")
        release_btn.setToolTip("Show version history and release notes")
        release_btn.clicked.connect(lambda: self._show_release_notes())
        btn_row.addWidget(release_btn)
        tutorial_btn = QPushButton("Tutorial")
        tutorial_btn.setToolTip("Watch the tutorial video on YouTube")
        tutorial_btn.clicked.connect(lambda: __import__("webbrowser").open(
            "https://youtu.be/CP26JwwXYAg"))
        btn_row.addWidget(tutorial_btn)
        btn_row.addStretch()
        ok_btn = QPushButton("OK")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(dlg.accept)
        btn_row.addWidget(ok_btn)
        layout.addLayout(btn_row)

        dlg.adjustSize()
        dlg.exec()
    def _show_release_notes(self):
        """Display the version history parsed from the script's own header."""
        # Read version history lines from the top of this file
        notes_lines = []
        try:
            with open(os.path.realpath(__file__), "r", encoding="utf-8") as fh:
                in_history = False
                for line in fh:
                    stripped = line.strip()
                    if stripped == "# Version History:":
                        in_history = True
                        continue
                    if in_history:
                        if stripped.startswith("#"):
                            entry = stripped.lstrip("#").strip()
                            if entry and not entry.startswith("="):
                                notes_lines.append(entry)
                        else:
                            break
        except Exception:
            notes_lines = ["Could not read release notes."]

        dlg = QDialog(self)
        dlg.setWindowTitle("Deep Space Astro - Script Launcher — Release Notes")
        dlg.setMinimumWidth(500)
        dlg.setWindowFlags(dlg.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(16, 16, 16, 12)
        layout.setSpacing(8)

        text = QTextEdit()
        text.setReadOnly(True)
        text.setFrameShape(QFrame.Shape.NoFrame)
        html = "<style>p{margin:2px 0;}</style>"
        for line in notes_lines:
            # Version lines start with "v" — make them bold
            if line.startswith("v"):
                parts = line.split(" - ", 2)
                ver = parts[0]
                # parts[1] is the version number already in ver, parts[2] is the note
                rest = f" — {parts[2]}" if len(parts) > 2 else (
                    f" — {parts[1]}" if len(parts) > 1 else ""
                )
                html += f"<p><b>{ver}</b>{rest}</p>"
            else:
                # Continuation line — align with the text after the version
                html += f"<p style='margin-left:48px'>{line}</p>"
        text.setHtml(html)
        layout.addWidget(text)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(dlg.accept)
        layout.addWidget(buttons)
        dlg.adjustSize()
        dlg.exec()

    def closeEvent(self, event):
        if self._workflow_dirty:
            wf = self._active_workflow
            msg = (
                f"Workflow '{wf}' has unsaved changes.\nClose without saving?"
                if wf else
                "You have unsaved changes in the right panel.\nClose without saving?"
            )
            reply = QMessageBox.question(
                self, "Unsaved Changes", msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            # Discard: revert notes, clear panel state, reset workflow
            try:
                with open(STATE_FILE, "r", encoding="utf-8") as fh:
                    saved = json.load(fh)
                self.notes = saved.get("notes", {})
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            self.checked_paths.clear()
            self.favorites.clear()
            self._active_workflow = ""
        g = self.geometry()
        geom = {"x": g.x(), "y": g.y(), "w": g.width(), "h": g.height()}
        save_state(self.checked_paths, self.favorites, geometry=geom,
                   active_workflow=self._active_workflow, notes=self.notes)
        super().closeEvent(event)

    def toggle_ontop(self, checked: bool):
        flags = self.windowFlags()
        if checked:
            self.setWindowFlags(flags | Qt.WindowType.WindowStaysOnTopHint)
        else:
            self.setWindowFlags(flags & ~Qt.WindowType.WindowStaysOnTopHint)
        self.show()

    # ── hide / show left panel ────────────────────────────────────────────────

    def _toggle_left(self):
        if self._left_panel.isVisible():
            self._expanded_geometry = self.geometry()
            self._left_panel.hide()
            self._search_lbl.hide()
            self._search_input.hide()
            self._refresh_btn.hide()
            self._add_step_btn.hide()
            self._clear_btn.hide()
            self._toggle_btn.setText("Show Scripts")
            self._toggle_btn.setToolTip("Show the Available Scripts panel")
            QTimer.singleShot(0, self._shrink_to_right_panel)
        else:
            self._left_panel.show()
            self._search_lbl.show()
            self._search_input.show()
            self._refresh_btn.show()
            self._add_step_btn.show()
            self._clear_btn.show()
            self._toggle_btn.setText("Hide Scripts")
            self._toggle_btn.setToolTip("Hide the Available Scripts panel")
            if hasattr(self, "_expanded_geometry"):
                # Keep current position (user may have moved the compact window)
                # but restore the saved expanded width and height
                current = self.geometry()
                g = self._expanded_geometry
                self.setGeometry(current.x(), current.y(), g.width(), g.height())

    def _shrink_to_right_panel(self):
        """Resize the window to fit the right panel with no scrollbars."""
        QApplication.processEvents()
        cw_margins = self.centralWidget().layout().contentsMargins()
        root_layout = self.centralWidget().layout()

        # ── width ─────────────────────────────────────────────────────────────
        container = self._right_panel._container
        content_w = container.sizeHint().width()
        sb_w = self._right_panel._scroll.verticalScrollBar().sizeHint().width()
        sc_margins = self._right_panel._scroll.contentsMargins()
        right_w = content_w + sb_w + sc_margins.left() + sc_margins.right() + 8
        total_w = right_w + cw_margins.left() + cw_margins.right()

        # ── height ────────────────────────────────────────────────────────────
        # Sum the height of every visible widget in the root layout that is NOT
        # the splitter (fixed chrome: top bar, search bar, status, buttons),
        # then add the actual content height of the right panel's scroll container.
        fixed_h = 0
        for i in range(root_layout.count()):
            item = root_layout.itemAt(i)
            w = item.widget() if item else None
            if w is not None and w is not self._splitter and w.isVisible():
                fixed_h += w.sizeHint().height() + root_layout.spacing()
            elif item and item.layout() is not None:
                # QHBoxLayout rows (top bar, button row)
                fixed_h += item.layout().sizeHint().height() + root_layout.spacing()

        # Scroll container content height (all rows stacked)
        content_h = container.sizeHint().height()
        # Add the non-scroll parts of SelectedPanel (title + hint labels + panel margins)
        panel_chrome_h = (self._right_panel.sizeHint().height()
                          - self._right_panel._scroll.sizeHint().height())

        total_h = (fixed_h + content_h + panel_chrome_h
                   + cw_margins.top() + cw_margins.bottom())

        # ── screen height cap ─────────────────────────────────────────────
        screen = QApplication.screenAt(self.geometry().center())
        if screen is None:
            screen = QApplication.primaryScreen()
        screen_h = screen.availableGeometry().height()
        max_h = int(screen_h * 0.95)
        if total_h > max_h:
            total_h = max_h
            # Ensure the scroll area in the right panel can scroll vertically
            self._right_panel._scroll.setVerticalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAsNeeded
            )
        else:
            self._right_panel._scroll.setVerticalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )

        self.resize(total_w, total_h)


# ── Entry-point ───────────────────────────────────────────────────────────────

def main():
    global STATE_FILE

    app = QApplication.instance() or QApplication(sys.argv)

    # Connect to Siril to get paths and print header, then disconnect.
    config_dir = None
    user_data_dir = None
    try:
        siril_instance = s.SirilInterface()
        siril_instance.connect()
        try:
            config_dir = siril_instance.get_siril_configdir()
        except Exception:
            pass
        try:
            user_data_dir = siril_instance.get_siril_userdatadir()
        except Exception:
            pass
        header_msg = (
            f"\n################################################################\n"
            f"# Deep Space Astro - Script Launcher v{VERSION}\n"
            "# Drag-and-drop script launcher for Siril\n"
            "# Author: Rich Stevenson - Deep Space Astro\n"
            "# YouTube:   https://www.youtube.com/@DeepSpaceAstro\n"
            "# Facebook:  https://www.facebook.com/@DeepSpaceAstro\n"
            "# Tiktok:    https://www.tiktok.com/@deepspaceastro\n"
            "# Instagram: https://www.instagram.com/deepspaceastro_official/\n"
            "# Discord:   https://discord.deepspaceastro.net\n"
            "#################################################################"
        )
        siril_instance.log(header_msg, color=LogColor.GREEN)
        siril_instance.disconnect()
    except Exception:
        pass  # Running outside Siril

    # Resolve state file path — prefer Siril's own config dir
    if config_dir:
        STATE_FILE = os.path.join(config_dir, "script_launcher_state.json")
    else:
        STATE_FILE = os.path.join(os.path.expanduser("~"), ".siril",
                                  "script_launcher_state.json")

    # Locate siril-scripts directory via Siril API
    scripts_dir = find_siril_scripts_dir(user_data_dir)

    if scripts_dir is None:
        from PyQt6.QtWidgets import QFileDialog
        scripts_dir = QFileDialog.getExistingDirectory(
            None, "Locate your siril-scripts directory"
        )
        if not scripts_dir:
            sys.exit("No scripts directory selected.")

    scripts = scan_scripts(scripts_dir)

    if not scripts:
        QMessageBox.warning(
            None, "No Scripts Found",
            f"No .py scripts were found in any subfolder of:\n{scripts_dir}"
        )
        sys.exit(0)

    window = ScriptLauncherWindow(scripts_dir, scripts)
    window.show()
    # If the user has scripts already selected, start with the left panel hidden.
    # If the right panel is empty, show the left panel so they can pick scripts.
    if window.checked_paths:
        QTimer.singleShot(0, window._toggle_left)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
