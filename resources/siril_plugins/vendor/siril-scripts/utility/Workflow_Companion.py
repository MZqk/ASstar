# STARTUP_CAPABLE
"""
March 13 2026
(c) Rich Stevenson - Deep Space Astro
SPDX-License-Identifier: GPL-3.0-or-later

Deep Space Astro - Workflow Companion
A drag-and-drop workflow companion for Siril.

Scripts are discovered automatically from subfolders of your siril-scripts
directory. Siril functions are discovered automatically from the Siril
function API. Drag scripts/functions to the right panel to queue them, reorder by
dragging, star favorites, and launch with one click. Selections, favorites, notes, workflows,
and window geometry are all persisted between launches.

Version history is maintained in the CHANGELOG dict below.

"""

import sirilpy as s
from sirilpy import LogColor

import base64
from collections import OrderedDict
import json
import os
import re
import tempfile
import uuid
from pathlib import Path
import ast as _ast
import html
import math
import sys
import textwrap
import webbrowser

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QVBoxLayout, QHBoxLayout, QPushButton,
    QLineEdit, QScrollArea, QFrame,
    QSplitter, QMessageBox, QCheckBox, QFileDialog,
    QDialog, QDialogButtonBox, QTextEdit, QComboBox, QMenu
)
from PyQt6.QtCore import Qt, QMimeData, QPoint, QByteArray, QSize, QTimer
from PyQt6.QtGui import QDrag, QPixmap, QPainter, QIcon, QPolygon, QColor, QTransform, QPen

VERSION = "1.1.1"

# Changelog entries shown in the What's New dialog.
# Each version maps to a list of one-liner strings matching the header comments.
# Only entries newer than last_seen_version are shown to the user.
CHANGELOG: dict[str, list[str]] = {
    "1.1.1": [
        "On Top state is now persistent between sessions",
        "Font size now inherits the OS system font size",
        "Replaced absolute paths, for group selections and notes, with relative paths on export",
        "Fixed Python scripts with spaces in the path failing to launch on macOS (e.g. Application Support)",
    ],
    "1.1.0": [
        "Scripts are now launched via Siril's native pyscript command, giving each script its own independent pipe connection and allowing multiple scripts to run simultaneously",
        "Fixed duplicate manual steps with the same name moving or deleting the wrong instance when reordered or removed",
        "Added Siril built-in dialogs (Functions) to the left panel (requires Siril 1.4.3 or later)",
        "Search now filters both scripts and functions including category names",
        "Already-selected items shown dimmed in left panel instead of hidden",
        "Left panel categories collapsed by default",
        "Added Collapse All and Expand All buttons to left panel",
        "Search box moved inside left panel and matches panel width",
        "Hide Scripts / Show Scripts renamed to Hide Panel / Show Panel",
        "Added optional description field to workflows",
        "Notes dialog now shows the script or step name as a header",
        "Step dot color can be changed from within the Notes dialog",
        "Added Groups - collapsible containers holding multiple alternative scripts or functions with radio selection and a single play button",
        "Items can be freely dragged between groups, the main list, and the left panel; duplicates are allowed",
        "Already-added items remain draggable in the left panel (shown dimmed)",
        "Added support for Siril .ssf scripts",
        "Dark astronomy-themed UI: deep navy/charcoal palette with teal/blue accents",
        "Script renamed from Script Launcher to Workflow Companion",
        "Script can start automatically when Siril loads (STARTUP_CAPABLE)",
        "Added Get Workflows button to open the Deep Space Astro workflows page",
        "Added What's New dialog shown once on first launch after an upgrade",
        "Replaced the right-click options for the Workflow menu with buttons for exporting, importing, and deleting workflows.",
    ],
    "1.0.3": [
        "Steps with / or \\ in the name now display correctly",
        "New button now prompts for confirmation if there are unsaved changes",
        "Step notes dialog also allows renaming the step",
        "Save button remains visible when the left panel is hidden",
        "Copying a step now creates an independent copy that can be renamed separately",
    ],
    "1.0.2": [
        "Fixed manual steps not restoring as favorites after reloading a workflow",
        "Scripts now launch using Siril's current working directory",
        "Added ability to mark items as complete in the right panel",
    ],
    "1.0.1": [
        "Exported workflows now use relative paths for cross-platform compatibility",
    ],
    "1.0.0": [
        "Original release",
    ],
}

# _DIALOG_GROUPS is built lazily on first rebuild() call and cached here
_SIRILPY_HAS_DIALOGS: bool = True   # optimistic; set False if import fails
_DIALOG_GROUPS: "OrderedDict | None" = None  # None = not yet built

ICON_B64 = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/4gHYSUNDX1BST0ZJTEUAAQEAAAHIAAAAAAQwAABtbnRyUkdCIFhZWiAH4AABAAEAAAAAAABhY3NwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAA9tYAAQAAAADTLQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlkZXNjAAAA8AAAACRyWFlaAAABFAAAABRnWFlaAAABKAAAABRiWFlaAAABPAAAABR3dHB0AAABUAAAABRyVFJDAAABZAAAAChnVFJDAAABZAAAAChiVFJDAAABZAAAAChjcHJ0AAABjAAAADxtbHVjAAAAAAAAAAEAAAAMZW5VUwAAAAgAAAAcAHMAUgBHAEJYWVogAAAAAAAAb6IAADj1AAADkFhZWiAAAAAAAABimQAAt4UAABjaWFlaIAAAAAAAACSgAAAPhAAAts9YWVogAAAAAAAA9tYAAQAAAADTLXBhcmEAAAAAAAQAAAACZmYAAPKnAAANWQAAE9AAAApbAAAAAAAAAABtbHVjAAAAAAAAAAEAAAAMZW5VUwAAACAAAAAcAEcAbwBvAGcAbABlACAASQBuAGMALgAgADIAMAAxADb/2wBDAAUDBAQEAwUEBAQFBQUGBwwIBwcHBw8LCwkMEQ8SEhEPERETFhwXExQaFRERGCEYGh0dHx8fExciJCIeJBweHx7/2wBDAQUFBQcGBw4ICA4eFBEUHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh7/wAARCAAfACADASIAAhEBAxEB/8QAGAAAAwEBAAAAAAAAAAAAAAAABQcIBgP/xAAsEAACAQMCBAUDBQAAAAAAAAABAgMEBREAEgYHITETIkFRYQgUQhVScYHR/8QAFwEBAQEBAAAAAAAAAAAAAAAABQQBBv/EACYRAAECBAUEAwAAAAAAAAAAAAECEQADBCEFMUFRYQYTInESsfD/2gAMAwEAAhEDEQA/AIy00uWXLs3S1Ul0uNda7ebjKVt61k4EkyocOY0xgnuMntrP8s4bzUPXUtqpZahZvAadEgEpMaSqxOME9CB2/j106+WtLw7xjwLFT1s8lNdaCJaK4QqhWSMJKzRAAjKg4ByuPMCD21JVVK6dIWmw32/H3HL9RYuqikkpLBwCR5EAixI0v7fKzuOE3KuYn7aKSWfJOIq+nUq/wChO0/PYaSfM7haThTiM0LwPTrJGJBE7bjHnIKhvyGRkH2I9c6qWqpOJrhm2friR07BJCIF8CcxjymNpMnG7GSRg+g0hPqLqWp77b+G5p46motlNiSQEs0QdmdYt7Es+1WU5PXrrKTE59UCicQrYsxHFoE6VxqsqqjtTpiVuCWAIIG7sNWDNrnpGz5BV80fKi611ZRNPFZpZ3o2jO2V1aMNJEjYypLbOo/cPYaE8m6LiyfmtJxNfKSSjjuzzpOk48Mys4LABD5toZVIOMZA663nLSxfbcgqSlopCKu5KJQ27bh5JQe/p5VA/rRy/Ja6C4Wa0zRxziqWNNgVkJfoPHRx1Rt3Tv/uiptQXWEh/kSD6ECTsQlmorJctDmapaeQkC5AyuXPJHEC/qI4qu/CXDtLJZIGWsqmIkqgu9aVR+WO29iwAJzjBwMkHUmzyyzzPNPI8ssjFnd2JZiepJJ7nVn3Cz0QuEUFZBJVxVUa08s1TUNK4UjO3afLj36HUz887JarDzBq6SzR+DSsquYguFjfqCFHt0z8Zx6aqwlSAjtJFxmd4a6HraRINJLR5kFRVuHy1ZnsPomP/2Q=="
)

# ── Config ────────────────────────────────────────────────────────────────────
# Entries in the siril-scripts root to skip when scanning for subfolders
EXCLUDED_ENTRIES = {".git", ".gitlab", "license.md", "readme.md"}

# STATE_FILE is resolved at runtime via siril.get_siril_configdir() in main()
STATE_FILE: str = ""  # set by main() before load_state() is called

# MIME type used for all drags - payload is "SOURCE|path"
MIME_TYPE = "application/x-siril-script"
GROUPITEM_MIME = "application/x-siril-groupitem"  # payload is "group_id|item_path"
STEP_PREFIX      = "step://"       # prefix for custom manual steps in checked_paths
FUNCTION_PREFIX  = "function://"   # prefix for Siril built-in dialog entries
GROUP_PREFIX     = "group://"      # prefix for group header entries
GROUPITEM_PREFIX = "groupitem://"  # prefix for items inside a group
SYSTEM_PREFIX    = "system://"     # prefix for Siril built-in system scripts (cross-platform)

# Tuple of all virtual prefixes - items starting with these are not real script files
NON_SCRIPT_PREFIXES = (STEP_PREFIX, FUNCTION_PREFIX, GROUP_PREFIX, GROUPITEM_PREFIX, SYSTEM_PREFIX)

# Step dot color palette - emoji: (display name, hex for highlight)
STEP_COLORS = [
    ("🟢", "Green",  "#2d6a2d"),
    ("🔴", "Red",    "#6a2d2d"),
    ("🟡", "Yellow", "#6a6a2d"),
    ("🔵", "Blue",   "#2d3d6a"),
    ("🟣", "Purple", "#4a2d6a"),
    ("🟠", "Orange", "#6a4a2d"),
    ("⚪", "White",  "#5a5a5a"),
]
DEFAULT_STEP_COLOR = "🟢"
SRC_LEFT  = "left"
SRC_RIGHT = "right"

# Qt's maximum widget dimension (QWidget::QWIDGETSIZE_MAX) - used to remove height caps
_MAX_WIDGET_SIZE = 16_777_215

# Tooltip strings for each Siril dialog requirement level (used in left panel rows)
_DIALOG_REQ_TOOLTIPS: dict[str, str] = {
    "NONE":        "",
    "ANY":         "Requires: an image or sequence to be loaded",
    "IMG":         "Requires: an image to be loaded",
    "RGB":         "Requires: an RGB image to be loaded",
    "SEQ":         "Requires: a sequence to be loaded",
    "MONO":        "Requires: a mono image to be loaded",
    "PLTSOLVD":    "Requires: a plate-solved image or sequence to be loaded",
    "RGBPLTSOLVD": "Requires: a plate-solved RGB image to be loaded",
}

# ── Persistence ───────────────────────────────────────────────────────────────


def path_to_relative(p: str, scripts_root: str) -> str:
    """Convert absolute script path to forward-slash relative from scripts_root.
    Virtual prefix paths are passed through unchanged."""
    if p.startswith((STEP_PREFIX, FUNCTION_PREFIX, GROUP_PREFIX, SYSTEM_PREFIX)):
        return p
    # For groupitem paths, convert the embedded item path
    if p.startswith(GROUPITEM_PREFIX):
        parsed = parse_groupitem(p)
        if parsed:
            gid, item_path = parsed
            rel_item = path_to_relative(item_path, scripts_root)
            return make_groupitem_path(gid, rel_item)
        return p
    try:
        rel = os.path.relpath(p, scripts_root)
        return rel.replace("\\", "/")
    except ValueError:
        return p.replace("\\", "/")

def path_to_absolute(p: str, scripts_root: str) -> str:
    """Resolve a relative path back to absolute using scripts_root.
    Virtual prefix paths are passed through unchanged; system:// paths
    are resolved via find_siril_system_scripts_dir()."""
    if p.startswith((STEP_PREFIX, FUNCTION_PREFIX, GROUP_PREFIX)):
        return p
    if p.startswith(SYSTEM_PREFIX):
        return p  # keep as virtual path - resolved to real path only at run time
    # For groupitem paths, convert the embedded item path
    if p.startswith(GROUPITEM_PREFIX):
        parsed = parse_groupitem(p)
        if parsed:
            gid, item_path = parsed
            abs_item = path_to_absolute(item_path, scripts_root)
            return make_groupitem_path(gid, abs_item)
        return p
    if os.path.isabs(p):
        return p
    rel = p.replace("/", os.sep).replace("\\", os.sep)
    return os.path.join(scripts_root, rel)

def _migrate_function_path(p: str) -> str:
    """Convert legacy function://INT paths to function://NAME format.
    Returns the path unchanged if already name-based or not a function path."""
    if not p.startswith(FUNCTION_PREFIX):
        return p
    suffix = p[len(FUNCTION_PREFIX):]
    if not suffix.isdigit():
        return p  # already name-based
    try:
        from sirilpy import DialogID
        dialog = DialogID(int(suffix))
        return f"{FUNCTION_PREFIX}{dialog.name}"
    except Exception:
        return p  # unknown ID, leave as-is

def load_state(scripts_dir: str | None = None) -> tuple[list, set, dict | None, dict, str, dict]:
    """Return (selections, favorites, geometry, workflows, active_workflow, notes) from state."""
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            root = os.path.normpath(scripts_dir) if scripts_dir else None
            def _abs(p):
                result = path_to_absolute(_migrate_function_path(p), root) if root else _migrate_function_path(p)
                return os.path.normpath(result) if result and not result.startswith(NON_SCRIPT_PREFIXES) else result
            selections      = [_abs(p) for p in data.get("selections", [])]
            favorites       = set(_abs(p) for p in data.get("favorites", []))
            geometry        = data.get("geometry", None)
            raw_workflows   = data.get("workflows", {})
            if root:
                workflows = {}
                for wf_name, wf in raw_workflows.items():
                    workflows[wf_name] = dict(wf)
                    workflows[wf_name]["selections"] = [
                        _abs(p) for p in wf.get("selections", [])]
                    workflows[wf_name]["favorites"] = [
                        _abs(p) for p in wf.get("favorites", [])]
            else:
                workflows = raw_workflows
            active_workflow  = data.get("active_workflow", "")
            notes            = data.get("notes", {})
            on_top           = data.get("on_top", True)
            return selections, favorites, geometry, workflows, active_workflow, notes, on_top
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return [], set(), None, {}, "", {}, True

def save_state(paths: list, favs: set, geometry: dict | None = None,
               workflows: dict | None = None,
               active_workflow: str | None = None,
               notes: dict | None = None,
               scripts_dir: str | None = None,
               last_step_color: str | None = None,
               on_top: bool | None = None) -> None:
    """Write selections, favorites, geometry, workflows, active_workflow and notes to state."""
    existing: dict = {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as fh:
            existing = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    # Convert absolute paths to relative for cross-platform portability
    if scripts_dir:
        root = os.path.normpath(scripts_dir)
        save_paths = [path_to_relative(p, root) for p in paths]
        save_favs  = sorted(path_to_relative(p, root) for p in favs)
    else:
        save_paths = list(paths)
        save_favs  = sorted(favs)
    existing["selections"] = save_paths
    existing["favorites"]  = save_favs
    if geometry is not None:
        existing["geometry"] = geometry
    if workflows is not None:
        if scripts_dir:
            root = os.path.normpath(scripts_dir)
            save_workflows = {}
            for wf_name, wf in workflows.items():
                save_workflows[wf_name] = dict(wf)
                save_workflows[wf_name]["selections"] = [
                    path_to_relative(p, root) for p in wf.get("selections", [])]
                save_workflows[wf_name]["favorites"] = [
                    path_to_relative(p, root) for p in wf.get("favorites", [])]
            existing["workflows"] = save_workflows
        else:
            existing["workflows"] = workflows
    if active_workflow is not None:
        existing["active_workflow"] = active_workflow
    if notes is not None:
        existing["notes"] = notes
    if last_step_color is not None:
        existing["last_step_color"] = last_step_color
    if on_top is not None:
        existing["on_top"] = on_top
    state_dir = os.path.dirname(STATE_FILE)
    os.makedirs(state_dir, exist_ok=True)
    # Write atomically: serialize to a temp file then rename so a crash mid-write
    # never leaves a corrupt state file.
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=state_dir, delete=False, suffix=".tmp"
        ) as tf:
            json.dump(existing, tf, indent=2)
            tmp_path = tf.name
        os.replace(tmp_path, STATE_FILE)
    except Exception:
        # Fall back to a direct write if the temp/replace fails (e.g. cross-device)
        with open(STATE_FILE, "w", encoding="utf-8") as fh:
            json.dump(existing, fh, indent=2)

# ── Helpers ───────────────────────────────────────────────────────────────────

_URL_RE = re.compile(r'(https?://\S+)', re.IGNORECASE)


def linkify(text: str) -> str:
    """Return HTML where any URLs in text are wrapped in clickable anchor tags.
    Non-URL portions are HTML-escaped so the label renders them as plain text."""
    parts = _URL_RE.split(text)
    result = []
    for i, part in enumerate(parts):
        if i % 2 == 1:  # odd indices are the captured URL groups
            escaped = html.escape(part)
            result.append(f'<a href="{escaped}" style="color:#58a6ff;">{escaped}</a>')
        else:
            result.append(html.escape(part))
    return "".join(result)

def find_siril_scripts_dir(user_data_dir: str | None = None) -> str | None:
    """Locate the siril-scripts directory.

    Tries, in order:
      1. SIRIL_SCRIPTS_DIR environment variable override
      2. user_data_dir from siril.get_siril_userdatadir() - cross-platform
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

def find_siril_system_scripts_dir() -> str | None:
    """Locate Siril's built-in system scripts directory (cross-platform)."""
    candidates = []
    if sys.platform == "win32":
        # Common Windows install locations
        for base in [r"C:\Program Files\SiriL", r"C:\Program Files (x86)\SiriL"]:
            candidates.append(os.path.join(base, "scripts"))
    elif sys.platform == "darwin":
        candidates = [
            "/Applications/Siril.app/Contents/Resources/scripts",
            os.path.expanduser("~/Applications/Siril.app/Contents/Resources/scripts"),
        ]
    else:  # Linux
        candidates = [
            "/usr/share/siril/scripts",
            "/usr/local/share/siril/scripts",
            "/opt/siril/scripts",
        ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return None


def make_group_id() -> str:
    """Generate a unique 6-char hex ID for a group."""
    return uuid.uuid4().hex[:6]


def parse_group_header(path: str) -> tuple[str, str] | None:
    """Parse a group:// path into (name, group_id) or None."""
    if not path.startswith(GROUP_PREFIX):
        return None
    suffix = path[len(GROUP_PREFIX):]
    if "#" in suffix:
        name, gid = suffix.rsplit("#", 1)
        return name, gid
    return suffix, ""


def parse_groupitem(path: str) -> tuple[str, str] | None:
    """Parse a groupitem:// path into (group_id, item_path) or None."""
    if not path.startswith(GROUPITEM_PREFIX):
        return None
    suffix = path[len(GROUPITEM_PREFIX):]
    if "|" in suffix:
        gid, item_path = suffix.split("|", 1)
        return gid, item_path
    return None


def make_groupitem_path(group_id: str, item_path: str) -> str:
    """Create a groupitem:// path from group_id and item_path."""
    return f"{GROUPITEM_PREFIX}{group_id}|{item_path}"


def _build_dialog_groups() -> "OrderedDict":
    """Build and return the _DIALOG_GROUPS OrderedDict from sirilpy.DialogID.

    Returns an empty OrderedDict and sets _SIRILPY_HAS_DIALOGS=False if the
    import fails (older sirilpy without DialogID support).
    """
    global _SIRILPY_HAS_DIALOGS
    try:
        from sirilpy import DialogID as _DID
        groups: OrderedDict = OrderedDict()
        for _dt in ["processing", "info", "science", "metadata", "application"]:
            for _d in _DID:
                if _d.dialog_type == _dt:
                    groups.setdefault(f"Functions - {_dt.title()}", []).append(_d)
        for _grp in list(groups.keys()):
            groups[_grp].sort(key=lambda d: d.label.lower())
        _SIRILPY_HAS_DIALOGS = True
        return groups
    except Exception:
        _SIRILPY_HAS_DIALOGS = False
        return OrderedDict()


def scan_scripts(directory: str) -> list[tuple[str, str]]:
    """Scan subfolders of directory for .py scripts. Returns (label, path) pairs."""
    scripts = []
    launcher_path = os.path.realpath(__file__)
    excluded = {e.lower() for e in EXCLUDED_ENTRIES}
    for subdir in sorted(os.scandir(directory), key=lambda e: e.name.lower()):
        if not subdir.is_dir() or subdir.name.lower() in excluded:
            continue
        for entry in sorted(os.scandir(subdir.path), key=lambda e: e.name.lower()):
            if not entry.is_file():
                continue
            grp_cap = subdir.name.capitalize()
            if (entry.name.endswith(".py")
                    and os.path.realpath(entry.path) != launcher_path):
                scripts.append((f"[{grp_cap}|py]  {entry.name}", os.path.normpath(entry.path)))
            elif entry.name.endswith(".ssf"):
                scripts.append((f"[{grp_cap}|ssf]  {entry.name}", os.path.normpath(entry.path)))
    return scripts

# Scripts in the system folder that belong to Processing group
_SYSTEM_PROCESSING_SCRIPTS = {"RGB_Composition.ssf"}


def scan_system_scripts(directory: str) -> list[tuple[str, str]]:
    """Scan Siril's system scripts folder and assign to appropriate groups.
    Returns system:// prefixed paths for cross-platform portability."""
    scripts = []
    try:
        for entry in sorted(os.scandir(directory), key=lambda e: e.name.lower()):
            if not entry.is_file() or not entry.name.endswith(".ssf"):
                continue
            sys_path = f"{SYSTEM_PREFIX}{entry.name}"
            if entry.name in _SYSTEM_PROCESSING_SCRIPTS:
                scripts.append((f"[Processing|ssf]  {entry.name}", sys_path))
            else:
                scripts.append((f"[Preprocessing|ssf]  {entry.name}", sys_path))
    except Exception:
        pass
    return scripts

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

_DOCSTRING_CACHE: dict[str, str] = {}


def extract_docstring(path: str) -> str:
    """Return the module-level docstring from a .py file, or '' if none."""
    if path.startswith(NON_SCRIPT_PREFIXES):
        return ""
    if path in _DOCSTRING_CACHE:
        return _DOCSTRING_CACHE[path]
    if path.endswith(".ssf"):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    line = line.strip()
                    if line.startswith("#") and len(line) > 1:
                        result = line.lstrip("#").strip()
                        _DOCSTRING_CACHE[path] = result
                        return result
        except Exception:
            pass
        _DOCSTRING_CACHE[path] = ""
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            source = fh.read()
        tree = _ast.parse(source)
        doc = _ast.get_docstring(tree)
        result = doc.strip() if doc else ""
        _DOCSTRING_CACHE[path] = result
        return result
    except Exception:
        _DOCSTRING_CACHE[path] = ""
        return ""

def build_item_tooltip(hint: str, description: str = "") -> str:
    """Build a left panel tooltip with description first (sets width),
    hint line at the bottom."""
    _hint_escaped = hint.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    hint_lines = "<br>".join(_hint_escaped.splitlines())
    if description:
        wrapped = textwrap.wrap(description, 60)
        desc_lines = "<br>".join(
            line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            for line in wrapped
        )
        return f"<html><body>{desc_lines}<br><br><i>{hint_lines}</i></body></html>"
    return f"<html><body><i>{hint_lines}</i></body></html>"

def drop_index_in_panel(scroll_area: "QScrollArea", rows_layout, drop_pos_local: QPoint,
                        skip_widget=None) -> int:
    """Return the insertion index for a drop inside a scrollable panel.

    drop_pos_local is in the coordinate space of the QScrollArea viewport.
    We convert it into the container widget's coordinate space (accounting for
    scroll offset) and compare against each row's midpoint.

    skip_widget: if provided, this widget is excluded from index calculation
    (used when ejecting an item from a group - the group row itself is skipped
    and insertion after it uses its bottom edge rather than midpoint).
    """
    scroll_y = scroll_area.verticalScrollBar().value()
    container_y = drop_pos_local.y() + scroll_y

    # Collect only real content rows (GroupRow / RightScriptRow).
    # Dividers, the workflow-name QLabel, and any other non-row widgets are
    # excluded so this index stays in sync with _reorder_main_list.
    widget_rows = []
    for i in range(rows_layout.count()):
        item = rows_layout.itemAt(i)
        w = item.widget() if item else None
        if isinstance(w, (GroupRow, RightScriptRow)):
            widget_rows.append(w)

    for content_i, w in enumerate(widget_rows):
        if w is skip_widget:
            if container_y < w.y() + w.height():
                return content_i
            continue
        mid_y = w.y() + w.height() // 2
        if container_y < mid_y:
            return content_i

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

class StarButton(QPushButton):
    """A QPushButton that paints a 5-pointed star; gold when favorited."""

    def __init__(self, is_fav: bool = False, parent=None):
        super().__init__(parent)
        self.setFixedSize(26, 26)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self._is_fav = is_fav

    def set_fav(self, is_fav: bool):
        self._is_fav = is_fav
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        outer_r = 7.0
        inner_r = 3.0
        points = []
        for i in range(10):
            angle = math.radians(-90 + i * 36)
            r = outer_r if i % 2 == 0 else inner_r
            points.append(QPoint(int(cx + r * math.cos(angle)),
                                 int(cy + r * math.sin(angle))))
        star = QPolygon(points)
        if self._is_fav:
            fill = QColor("#e3b341")   # warm gold
            p.setBrush(fill)
            p.setPen(Qt.PenStyle.NoPen)
        else:
            color = self.palette().buttonText().color()
            p.setBrush(Qt.BrushStyle.NoBrush)
            pen = p.pen()
            pen.setColor(color)
            pen.setWidth(1)
            p.setPen(pen)
        p.drawPolygon(star)
        p.end()

# ── Left panel row ────────────────────────────────────────────────────────────


class LeftScriptRow(QFrame):
    """A single draggable row in the left Scripts panel."""

    def __init__(self, label: str, path: str, on_add, tooltip: str = "", parent=None):
        super().__init__(parent)
        self.path = path
        self.on_add = on_add
        self._drag_start = QPoint()

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

        # Build tooltip: hint always on one line, description wrapped below
        self._tooltip_override = tooltip
        if tooltip:
            self.setToolTip(build_item_tooltip("Drag or double-click to add.", tooltip))
        else:
            doc = extract_docstring(path)
            self.setToolTip(build_item_tooltip("Drag or double-click to add.", doc))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)

        lbl = QLabel(label)
        layout.addWidget(lbl, stretch=1)

    def set_selected(self, selected: bool):
        """Dim the row when already in the right panel to indicate it's in use,
        but keep it draggable so duplicates are allowed."""
        if selected:
            self.setToolTip("Already in Selected Items - drag or double-click to add again")
            self.setStyleSheet("color: #484f58;")
        else:
            self.setStyleSheet("")
            self._restore_tooltip()

    def _restore_tooltip(self):
        if self._tooltip_override:
            self.setToolTip(build_item_tooltip("Drag or double-click to add.", self._tooltip_override))
        else:
            doc = extract_docstring(self.path)
            self.setToolTip(build_item_tooltip("Drag or double-click to add.", doc))

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

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        self._arrow = QLabel("▾")
        layout.addWidget(self._arrow)

        self._key = label  # plain label used as collapse state key
        self._label = QLabel(f"<b>{label}</b>")
        layout.addWidget(self._label, stretch=1)

        self._count_lbl = QLabel(f"({count})")
        layout.addWidget(self._count_lbl)

        self.setCursor(Qt.CursorShape.PointingHandCursor)
    def set_container(self, container: "AnimatedRowContainer"):
        self._container = container

    def set_count(self, count: int):
        self._count_lbl.setText(f"({count})")

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#1c2333"))
        p.drawRoundedRect(self.rect(), 3, 3)
        p.end()
        super().paintEvent(event)

    def _animate(self, collapse: bool):
        if self._container is None:
            return
        self._container.setVisible(not collapse)

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

        title = QLabel("Available Items")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(title)

        hint = QLabel("Drag selected items here to remove")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setWordWrap(True)
        outer.addWidget(hint)

        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Filter items…")
        self._search_input.setToolTip("Type to filter by name or category")
        self._search_input.setClearButtonEnabled(True)
        outer.addWidget(self._search_input)

        _btn_row = QHBoxLayout()
        _collapse_btn = QPushButton("Collapse All")
        _collapse_btn.setToolTip("Collapse all categories")
        _collapse_btn.clicked.connect(self._collapse_all)
        _btn_row.addWidget(_collapse_btn)
        _expand_btn = QPushButton("Expand All")
        _expand_btn.setToolTip("Expand all categories")
        _expand_btn.clicked.connect(self._expand_all)
        _btn_row.addWidget(_expand_btn)
        outer.addLayout(_btn_row)

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
        self._last_query: str = ""

        # path -> list of LeftScriptRow widgets (list because duplicates are allowed)
        # Populated by rebuild(); used by refresh_selected() to avoid findChildren()
        self._row_by_path: dict[str, list] = {}

        self.rebuild()

    def _save_collapsed_state(self):
        """Read collapsed state directly from current headers."""
        for i in range(self._layout.count()):
            item = self._layout.itemAt(i)
            w = item.widget() if item else None
            if isinstance(w, SubdirHeader):
                self._collapsed[w._key] = w._collapsed

    def _build_in_use(self) -> set:
        """Return the set of item paths currently in use in the right panel.
        Groupitem paths are unwrapped to their inner item path so they match
        the paths stored in LeftScriptRow.path."""
        in_use: set = set()
        for p in self.checked_paths:
            if p.startswith(GROUPITEM_PREFIX):
                parsed = parse_groupitem(p)
                if parsed:
                    in_use.add(parsed[1])
            else:
                in_use.add(p)
        return in_use

    def refresh_selected(self) -> None:
        """Update dimming state of existing rows without full rebuild.
        Much faster than rebuild() when only checked_paths has changed.
        Uses _row_by_path (populated by rebuild) instead of findChildren()
        to avoid an O(all-widgets) tree walk."""
        in_use = self._build_in_use()
        for path, rows in self._row_by_path.items():
            selected = path in in_use
            for row in rows:
                row.set_selected(selected)

    def rebuild(self, scripts=None, query: str = "", _skip_collapse_save: bool = False):
        if scripts is not None:
            self.scripts = scripts

        # Save current collapsed state before clearing (skip during search or explicit override)
        if not query and not self._last_query and not _skip_collapse_save:
            for i in range(self._layout.count()):
                item = self._layout.itemAt(i)
                w = item.widget() if item else None
                if isinstance(w, SubdirHeader):
                    self._collapsed[w._key] = w._collapsed
        self._last_query = query

        while self._layout.count() > 1:
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Rebuild grouped - {grp: {sub: [(name, path)]}}
        nested_groups: OrderedDict = OrderedDict()
        for label, path in self.scripts:
            if label.startswith("["):
                inner = label[1:label.index("]")]
                grp, sub = inner.split("|", 1) if "|" in inner else (inner, "")
            else:
                grp, sub = "Other", ""
            name = label.split("]  ", 1)[-1]
            nested_groups.setdefault(grp, OrderedDict()).setdefault(sub, []).append((name, path))
        # Sort entries within each sub-group alphabetically
        for grp in nested_groups:
            for sub in nested_groups[grp]:
                nested_groups[grp][sub].sort(key=lambda x: x[0].lower())

        # Build set of paths currently in use (direct or as groupitem)
        # Shared helper avoids duplicating this logic with refresh_selected()
        _in_use = self._build_in_use()

        # Reset the path->row index; rebuilt below as rows are created
        self._row_by_path = {}

        pos = 0

        # ── Siril built-in dialogs (requires sirilpy >= 1.0.21) ──────────
        if _SIRILPY_HAS_DIALOGS and _DIALOG_GROUPS:
            try:
                for grp_name, dialogs in _DIALOG_GROUPS.items():
                    _grp_matches = query and query in grp_name.lower()
                    visible_dialogs = dialogs if (not query or _grp_matches) else [
                        d for d in dialogs if query in d.label.lower()]
                    if not visible_dialogs:
                        continue
                    header = SubdirHeader(grp_name, count=len(visible_dialogs))
                    _should_collapse = False if query else self._collapsed.get(grp_name, True)
                    if _should_collapse:
                        header._collapsed = True
                        header._arrow.setText("▸")
                    self._layout.insertWidget(pos, header)
                    pos += 1
                    container = AnimatedRowContainer()
                    container._layout.setContentsMargins(16, 0, 0, 0)
                    for d in visible_dialogs:
                        fpath = f"{FUNCTION_PREFIX}{d.name}"
                        try:
                            _req = d.req.name if hasattr(d, "req") else ""
                            _tip = _DIALOG_REQ_TOOLTIPS.get(_req, "")
                        except Exception:
                            _tip = ""
                        row = LeftScriptRow(d.label, fpath, self.on_add,
                                           tooltip=_tip)
                        if fpath in _in_use:
                            row.set_selected(True)
                        self._row_by_path.setdefault(fpath, []).append(row)
                        container.add_row(row)
                    header.set_container(container)
                    if _should_collapse:
                        container.setVisible(False)
                    else:
                        container.setMaximumHeight(_MAX_WIDGET_SIZE)
                    self._layout.insertWidget(pos, container)
                    pos += 1
            except Exception:
                pass  # DialogID not available in this sirilpy version

        for grp, sub_groups in nested_groups.items():
            _grp_full = f"Scripts - {grp}"
            _grp_matches = query and query in _grp_full.lower()
            # Render flat (no sub-groups) when only one sub-group exists
            if len(sub_groups) == 1:
                all_entries = [(n, p) for sub_entries in sub_groups.values() for n, p in sub_entries]
                visible_entries = all_entries if (not query or _grp_matches) else [
                    (n, p) for n, p in all_entries if query in n.lower()]
                if not visible_entries:
                    continue
                header = SubdirHeader(_grp_full, count=len(visible_entries))
                should_collapse = False if query else self._collapsed.get(_grp_full, True)
                if should_collapse:
                    header._collapsed = True
                    header._arrow.setText("▸")
                self._layout.insertWidget(pos, header)
                pos += 1
                container = AnimatedRowContainer()
                container._layout.setContentsMargins(16, 0, 0, 0)
                for name, path in visible_entries:
                    row = LeftScriptRow(name, path, self.on_add)
                    if path in _in_use:
                        row.set_selected(True)
                    self._row_by_path.setdefault(path, []).append(row)
                    container.add_row(row)
                header.set_container(container)
                if header._collapsed:
                    container.setVisible(False)
                else:
                    container.setMaximumHeight(_MAX_WIDGET_SIZE)
                self._layout.insertWidget(pos, container)
                pos += 1
                continue
            all_visible = []
            for sub, entries in sorted(sub_groups.items(), key=lambda x: (
                    "Python Scripts" if x[0] == "py" else "Siril Scripts" if x[0] == "ssf" else x[0]).lower()):
                sub_label = "Python Scripts" if sub == "py" else "Siril Scripts" if sub == "ssf" else sub
                _sub_matches = query and query in sub_label.lower()
                vis = entries if (not query or _grp_matches or _sub_matches) else [
                    (n, p) for n, p in entries if query in n.lower()]
                if vis:
                    all_visible.append((sub_label, vis))
            if not all_visible:
                continue
            total = sum(len(v) for _, v in all_visible)
            # Parent header with animation
            parent_header = SubdirHeader(_grp_full, count=total)
            parent_collapse = False if query else self._collapsed.get(_grp_full, True)
            if parent_collapse:
                parent_header._collapsed = True
                parent_header._arrow.setText("▸")
            self._layout.insertWidget(pos, parent_header)
            pos += 1
            parent_container = AnimatedRowContainer()
            pc_layout = parent_container._layout
            pc_layout.setContentsMargins(0, 0, 0, 0)
            pc_layout.setSpacing(2)
            for sub_label, entries in all_visible:
                _sub_full = f"{_grp_full} - {sub_label}"
                sub_collapse = False if query else self._collapsed.get(_sub_full, True)
                sh = QFrame()
                sh.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
                sh.setStyleSheet("background-color: #243044; border-radius: 3px;")
                sh_layout = QHBoxLayout(sh)
                sh_layout.setContentsMargins(16, 1, 4, 1)
                sh_layout.setSpacing(4)
                sh_arrow = QLabel("▸" if sub_collapse else "▾")
                sh_layout.addWidget(sh_arrow)
                sh_layout.addWidget(QLabel(sub_label), stretch=1)
                sh_layout.addWidget(QLabel(f"({len(entries)})"))
                sh.setCursor(Qt.CursorShape.PointingHandCursor)
                sc = QWidget()
                sc_layout = QVBoxLayout(sc)
                sc_layout.setContentsMargins(24, 0, 0, 0)
                sc_layout.setSpacing(2)
                for name, path in entries:
                    row = LeftScriptRow(name, path, self.on_add)
                    if path in _in_use:
                        row.set_selected(True)
                    self._row_by_path.setdefault(path, []).append(row)
                    sc_layout.addWidget(row)
                sc.setVisible(not sub_collapse)
                def _make_toggle(arrow, container, key):
                    def _toggle(event):
                        new_state = not self._collapsed.get(key, True)
                        self._collapsed[key] = new_state
                        arrow.setText("▸" if new_state else "▾")
                        container.setVisible(not new_state)
                    return _toggle
                sh.mousePressEvent = _make_toggle(sh_arrow, sc, _sub_full)
                pc_layout.addWidget(sh)
                pc_layout.addWidget(sc)
            parent_header.set_container(parent_container)
            if parent_collapse:
                parent_container.setVisible(False)
            else:
                parent_container.setMaximumHeight(_MAX_WIDGET_SIZE)
            self._layout.insertWidget(pos, parent_container)
            pos += 1

    def _all_category_keys(self) -> list:
        """Return all category keys that exist for the current scripts."""
        keys = []
        nested_groups: dict = {}
        for label, _ in self.scripts:
            if label.startswith("["):
                inner = label[1:label.index("]")]
                grp, sub = inner.split("|", 1) if "|" in inner else (inner, "")
            else:
                grp, sub = "Other", ""
            nested_groups.setdefault(grp, set()).add(sub)
        for grp, subs in nested_groups.items():
            grp_full = f"Scripts - {grp}"
            keys.append(grp_full)
            if len(subs) > 1:
                for sub in subs:
                    sub_label = "Python Scripts" if sub == "py" else "Siril Scripts" if sub == "ssf" else sub
                    keys.append(f"{grp_full} - {sub_label}")
        # Function categories
        for dt in ["processing", "info", "science", "metadata", "application"]:
            keys.append(f"Functions - {dt.title()}")
        # Any other keys already tracked
        for key in self._collapsed:
            if key not in keys:
                keys.append(key)
        return keys

    def _collapse_all(self):
        """Collapse all category headers including nested sub-groups."""
        for key in self._all_category_keys():
            self._collapsed[key] = True
        for i in range(self._layout.count()):
            w = self._layout.itemAt(i).widget() if self._layout.itemAt(i) else None
            if isinstance(w, SubdirHeader):
                if not w._collapsed:
                    w._collapsed = True
                    w._arrow.setText("▸")
                    self._collapsed[w._key] = True
                    if w._container:
                        cl = w._container._layout if hasattr(w._container, '_layout') else w._container.layout()
                        if cl:
                            for j in range(cl.count()):
                                sub_w = cl.itemAt(j).widget() if cl.itemAt(j) else None
                                if sub_w and hasattr(sub_w, '_arrow'):
                                    sub_w._arrow.setText("▸")
                        w._container.setVisible(False)
            elif w is not None and not isinstance(w, SubdirHeader):
                if i > 0:
                    prev_w = self._layout.itemAt(i - 1).widget() if self._layout.itemAt(i - 1) else None
                    if isinstance(prev_w, SubdirHeader):
                        w.setVisible(False)

    def _expand_all(self):
        """Expand all category headers including nested sub-groups."""
        for key in self._all_category_keys():
            self._collapsed[key] = False
        for i in range(self._layout.count()):
            w = self._layout.itemAt(i).widget() if self._layout.itemAt(i) else None
            if isinstance(w, SubdirHeader):
                if w._collapsed:
                    w._collapsed = False
                    w._arrow.setText("▾")
                    self._collapsed[w._key] = False
                    if w._container:
                        cl = w._container._layout if hasattr(w._container, '_layout') else w._container.layout()
                        if cl:
                            for j in range(cl.count()):
                                sub_w = cl.itemAt(j).widget() if cl.itemAt(j) else None
                                if sub_w and hasattr(sub_w, '_arrow'):
                                    sub_w._arrow.setText("▾")
                                elif sub_w:
                                    sub_w.setVisible(True)
                        w._container.setMaximumHeight(_MAX_WIDGET_SIZE)
                        w._container.setVisible(True)
            elif w is not None and not isinstance(w, SubdirHeader):
                if i > 0:
                    prev_w = self._layout.itemAt(i - 1).widget() if self._layout.itemAt(i - 1) else None
                    if isinstance(prev_w, SubdirHeader):
                        w.setVisible(True)

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
        color = QColor("#09ee2f") if self._checked else self.palette().buttonText().color()
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
    """A QPushButton that paints a pencil icon; turns blue when a note exists.
    An optional gold border indicates that child items have notes."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(26, 26)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self._has_note = False
        self._has_item_notes = False

    def set_has_note(self, has_note: bool):
        self._has_note = has_note
        self.update()

    def set_has_item_notes(self, has_item_notes: bool):
        self._has_item_notes = has_item_notes
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Gold border when any child item has a note
        if self._has_item_notes:
            pen = QPen(QColor("#e3b341"), 1.0)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(1, 1, self.width() - 2, self.height() - 2, 4, 4)

        if self._has_note:
            color = QColor("#4fc3f7")
        else:
            color = self.palette().buttonText().color()

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color)

        w, h = self.width(), self.height()
        cx, cy = w // 2, h // 2

        t = QTransform()
        t.translate(cx, cy)
        t.rotate(45)
        p.setTransform(t)

        # Body rectangle
        p.drawRoundedRect(-2, -6, 4, 9, 1, 1)
        # Tip triangle
        tip = QPolygon([QPoint(-2, 3), QPoint(2, 3), QPoint(0, 6)])
        p.drawPolygon(tip)
        # Eraser cap - same color as body so size looks identical in both states
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

        # Lines inside body (cutouts - draw in button background colour)
        bg = self.palette().button().color()
        p.setBrush(bg)
        line_h = body_h - 2
        line_w = 1
        line_y = body_y + 1
        for offset in (-2, 0, 2):
            p.drawRect(cx + offset - line_w // 2, line_y, line_w, line_h)

        p.end()

class ArrowButton(QPushButton):
    """A QPushButton that paints a chevron - right (collapsed) or down (expanded)."""

    def __init__(self, collapsed: bool = True, parent=None):
        super().__init__(parent)
        self.setFixedSize(20, 20)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._collapsed = collapsed

    def set_collapsed(self, collapsed: bool):
        self._collapsed = collapsed
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = self.palette().buttonText().color()
        pen = QPen(color, 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        w, h = self.width(), self.height()
        cx, cy = w // 2, h // 2
        if self._collapsed:
            # Right-pointing chevron: > shape
            p.drawLine(cx - 2, cy - 4, cx + 3, cy)
            p.drawLine(cx - 2, cy + 4, cx + 3, cy)
        else:
            # Down-pointing chevron: ∨ shape
            p.drawLine(cx - 4, cy - 2, cx, cy + 3)
            p.drawLine(cx + 4, cy - 2, cx, cy + 3)
        p.end()


class RadioButton(QPushButton):
    """A QPushButton that paints a radio circle - filled (selected) or empty."""

    def __init__(self, selected: bool = False, parent=None):
        super().__init__(parent)
        self.setFixedSize(24, 24)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._selected = selected

    def set_selected(self, selected: bool):
        self._selected = selected
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = self.palette().buttonText().color()
        w, h = self.width(), self.height()
        cx, cy = w // 2, h // 2
        r_outer = 7
        p.setPen(QPen(color, 1.5))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPoint(cx, cy), r_outer, r_outer)
        if self._selected:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(color)
            p.drawEllipse(QPoint(cx, cy), 4, 4)
        p.end()


class NoteButton(QLabel):
    """A fixed-size label showing a coloured dot emoji in place of a play button."""

    def __init__(self, color: str = DEFAULT_STEP_COLOR, parent=None):
        super().__init__(color, parent)
        self._color = color
        self.setFixedSize(28, 28)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setToolTip("Manual step - no script to run")
        self.setCursor(Qt.CursorShape.ArrowCursor)
        font = self.font()
        font.setPointSize(font.pointSize())  # inherit system point size; scales correctly on HiDPI/Retina
        self.setFont(font)

    def set_color(self, color: str):
        self._color = color
        self.setText(color)

    def set_running(self, running: bool):
        pass  # Manual steps have no run state

    def setEnabled(self, enabled: bool):
        pass  # Always visible, never disabled

# ── Right panel row ───────────────────────────────────────────────────────────


class RightScriptRow(QFrame):
    """A row in the Selected Scripts panel - draggable for reordering or
    dragging back to the left panel to remove."""

    def __init__(self, path: str, name: str, on_run, is_fav: bool = False,
                 on_fav_toggle=None, on_remove=None, on_copy=None,
                 missing: bool = False, note: str = "",
                 on_notes_changed=None, on_step_renamed=None,
                 completed: bool = False, on_completed_toggle=None,
                 step_color: str = DEFAULT_STEP_COLOR, on_color_changed=None, parent=None):
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
        self._step_color = step_color
        self._on_color_changed = on_color_changed

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setToolTip("Drag to reorder, or drag back to Available Items to remove")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        handle = QLabel("⠿")
        handle.setCursor(Qt.CursorShape.OpenHandCursor)
        layout.addWidget(handle)

        _is_step = path.startswith(STEP_PREFIX)
        if _is_step:
            self._run_btn = NoteButton(step_color)
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

        self._star_btn = StarButton(is_fav=is_fav)
        self._star_btn.setToolTip("Remove from favorites" if is_fav else "Add to favorites")
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
            lbl.setStyleSheet("color: #c0392b;")
            handle.setEnabled(False)

    def _toggle_fav(self):
        self._is_fav = not self._is_fav
        self._star_btn.set_fav(self._is_fav)
        self._star_btn.setToolTip("Remove from favorites" if self._is_fav else "Add to favorites")
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
            self._lbl.setStyleSheet("text-decoration: line-through; color: #484f58;")
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
        dlg.setWindowTitle(f"Notes - {self.name}")
        dlg.setMinimumWidth(400)
        dlg.setMinimumHeight(200)
        dlg.setWindowFlags(dlg.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(10, 10, 10, 8)
        layout.setSpacing(6)

        # Header showing item name
        header_lbl = QLabel(self.name)
        header_font = header_lbl.font()
        header_font.setBold(True)
        header_font.setPointSize(header_font.pointSize() + 1)
        header_lbl.setFont(header_font)
        layout.addWidget(header_lbl)

        # Step name field - only for manual steps
        if _is_step:
            layout.addWidget(QLabel("Step name:"))
            name_edit = QLineEdit(self.name)
            layout.addWidget(name_edit)
            # Color picker row
            color_lbl = QLabel("Dot color:")
            layout.addWidget(color_lbl)
            color_row = QHBoxLayout()
            color_row.setSpacing(4)
            selected_color = [self._step_color]
            color_btns = []
            def _make_color_click(e, btns=color_btns, sel=selected_color):
                sel[0] = e
                for b, (em, _, hi) in zip(btns, STEP_COLORS):
                    if em == e:
                        b.setStyleSheet(f"QPushButton {{ background-color: {hi}; border: 2px solid #c9d1d9; border-radius: 16px; padding: 0; min-width: 32px; }}")
                    else:
                        b.setStyleSheet("QPushButton { border-radius: 16px; padding: 0; min-width: 32px; }")
            for emoji, cname, highlight in STEP_COLORS:
                btn = QPushButton(emoji)
                btn.setFixedSize(32, 32)
                btn.setToolTip(cname)
                btn.setCursor(Qt.CursorShape.PointingHandCursor)
                if emoji == self._step_color:
                    btn.setStyleSheet(f"QPushButton {{ background-color: {highlight}; border: 2px solid #c9d1d9; border-radius: 16px; padding: 0; min-width: 32px; }}")
                else:
                    btn.setStyleSheet("QPushButton { border-radius: 16px; padding: 0; min-width: 32px; }")
                btn.clicked.connect(lambda checked, em=emoji: _make_color_click(em))
                color_btns.append(btn)
                color_row.addWidget(btn)
            color_row.addStretch()
            layout.addLayout(color_row)

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
            # Apply color change
            new_color = selected_color[0]
            if new_color != self._step_color:
                self._step_color = new_color
                self._run_btn.set_color(new_color)
                if self._on_color_changed:
                    self._on_color_changed(self.path, new_color)
        self._note = editor.toPlainText().strip()
        self._update_notes_btn()
        if self._on_notes_changed:
            # Use _note_key if set (assigned by rebuild() for duplicate-aware keying),
            # otherwise fall back to self.path for backward compatibility.
            key = getattr(self, "_note_key", self.path)
            self._on_notes_changed(key, self._note)

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
        is_script = not path.startswith(STEP_PREFIX) and not path.startswith(FUNCTION_PREFIX)
        if is_script:
            self._run_btn.set_running(True)
        QApplication.processEvents()
        try:
            on_run(path, name)
        finally:
            self._flash_green()

    def _flash_green(self):
        """Briefly highlight the row green then restore."""
        self.setStyleSheet("background-color: #0d2a4a; border: 1px solid #1f6feb;")
        QTimer.singleShot(600, self._clear_flash)

    def _clear_flash(self):
        self.setStyleSheet("")
        if not self.path.startswith(STEP_PREFIX) and not self.path.startswith(FUNCTION_PREFIX):
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
    """QComboBox for selecting saved workflows."""

    def __init__(self, parent=None):
        super().__init__(parent)

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        # Draw downward triangle on the right side
        tx = w - 14
        ty = h // 2 - 2
        triangle = QPolygon([
            QPoint(tx,      ty),
            QPoint(tx + 8,  ty),
            QPoint(tx + 4,  ty + 5),
        ])
        color = QColor("#c9d1d9") if self.isEnabled() else QColor("#484f58")
        p.setBrush(color)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPolygon(triangle)
        p.end()


def drop_index_in_group(items_layout, drop_y_in_group: int) -> int:
    """Return insertion index within a group's items layout based on drop y position.
    drop_y_in_group is in the coordinate space of the items_widget."""
    for i in range(items_layout.count()):
        w = items_layout.itemAt(i).widget()
        if w and drop_y_in_group < w.y() + w.height() // 2:
            return i
    return items_layout.count()


# ── Group row ─────────────────────────────────────────────────────────────────


class GroupItemRow(QFrame):
    """A radio-selectable item inside a group row."""

    def __init__(self, item_path: str, name: str, is_selected: bool,
                 on_select, on_remove, group_id: str = "",
                 missing: bool = False, note: str = "",
                 on_notes_changed=None, on_group_notes_btn_refresh=None,
                 parent=None):
        super().__init__(parent)
        self.item_path = item_path
        self._group_id = group_id
        self._on_select = on_select
        self._on_remove = on_remove
        self._note = note
        self._on_notes_changed = on_notes_changed
        self._on_group_notes_btn_refresh = on_group_notes_btn_refresh
        self._drag_start = QPoint()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(24, 2, 4, 2)
        layout.setSpacing(6)

        handle = QLabel("⠿")
        handle.setCursor(Qt.CursorShape.OpenHandCursor)
        layout.addWidget(handle)

        self._radio = RadioButton(selected=is_selected)
        self._radio.setToolTip("Select this item")
        self._radio.clicked.connect(self._on_radio_clicked)
        layout.addWidget(self._radio)

        lbl = QLabel(name)
        layout.addWidget(lbl, stretch=1)

        self._notes_btn = PencilButton()
        self._notes_btn.clicked.connect(self._open_notes)
        layout.addWidget(self._notes_btn)
        self._update_notes_btn()

        trash = TrashButton()
        trash.clicked.connect(self._confirm_remove)
        layout.addWidget(trash)

        if missing:
            self.setToolTip("Script not installed.")
            self.setCursor(Qt.CursorShape.ForbiddenCursor)
            self._radio.setEnabled(False)
            self._radio.setToolTip("Script not installed.")
            lbl.setEnabled(False)
            lbl.setStyleSheet("color: #c0392b;")
            handle.setEnabled(False)

    def _update_notes_btn(self):
        self._notes_btn.set_has_note(bool(self._note))
        self._notes_btn.setToolTip(self._note if self._note else "Add or edit notes")

    def _open_notes(self):
        name = _resolve_path_name(self.item_path)
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Notes - {name}")
        dlg.setMinimumWidth(400)
        dlg.setMinimumHeight(200)
        dlg.setWindowFlags(dlg.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(10, 10, 10, 8)
        layout.setSpacing(6)
        header_lbl = QLabel(name)
        hf = header_lbl.font()
        hf.setBold(True)
        hf.setPointSize(hf.pointSize() + 1)
        header_lbl.setFont(hf)
        layout.addWidget(header_lbl)
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
        self._note = editor.toPlainText().strip()
        self._update_notes_btn()
        if self._on_notes_changed:
            # Key is the composite groupitem:// path so it's 100% independent
            note_key = make_groupitem_path(self._group_id, self.item_path)
            self._on_notes_changed(note_key, self._note)
        # Immediately refresh the parent GroupRow's pencil border
        if self._on_group_notes_btn_refresh:
            self._on_group_notes_btn_refresh()

    def _confirm_remove(self):
        name = _resolve_path_name(self.item_path)
        reply = QMessageBox.question(
            self, "Remove Item",
            f"Remove '{name}' from the group?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes and self._on_remove:
            self._on_remove(self.item_path)

    def set_selected(self, selected: bool):
        self._radio.set_selected(selected)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.pos()

    def mouseMoveEvent(self, event):
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        if (event.pos() - self._drag_start).manhattanLength() < QApplication.startDragDistance():
            return
        drag = QDrag(self)
        mime = QMimeData()
        payload = f"{self._group_id}|{self.item_path}"
        mime.setData(GROUPITEM_MIME, QByteArray(payload.encode()))
        # Also carry SRC_RIGHT MIME so SelectedPanel / other GroupRows can handle it
        mime.setData(MIME_TYPE, QByteArray(f"{SRC_RIGHT}|{self.item_path}".encode()))
        drag.setMimeData(mime)
        pixmap = QPixmap(self.size())
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setOpacity(0.55)
        self.render(painter)
        painter.end()
        drag.setPixmap(pixmap)
        drag.exec(Qt.DropAction.MoveAction)

    def _on_radio_clicked(self):
        if self._on_select:
            self._on_select(self.item_path)


class GroupRow(QFrame):
    """A collapsible group in the right panel containing radio-selectable items."""

    def __init__(self, group_path: str, name: str, item_paths: list,
                 selected_item: str, on_run, is_fav: bool,
                 on_fav_toggle, on_remove_group, on_remove_item,
                 on_select_item, on_collapsed_changed, on_add_to_group=None,
                 collapsed: bool = False, completed: bool = False,
                 note: str = "", on_notes_changed=None,
                 on_completed_toggle=None, on_group_renamed=None,
                 on_reorder_items=None, missing_paths: set = None,
                 item_notes: dict = None, parent=None):
        super().__init__(parent)
        self.group_path = group_path
        self.name = name
        self._on_run = on_run
        self._on_fav_toggle = on_fav_toggle
        self._on_remove_group = on_remove_group
        self._on_remove_item = on_remove_item
        self._on_select_item = on_select_item
        self._on_collapsed_changed = on_collapsed_changed
        self._on_add_to_group = on_add_to_group
        self._collapsed = collapsed
        self._is_fav = is_fav
        self._selected_item = selected_item
        self.item_paths = item_paths
        self._completed = completed
        self._note = note
        self._on_notes_changed = on_notes_changed
        self._on_completed_toggle = on_completed_toggle
        self._on_group_renamed = on_group_renamed
        self._on_reorder_items = on_reorder_items
        self._missing_paths: set = missing_paths if missing_paths is not None else set()
        self._item_notes: dict = item_notes if item_notes is not None else {}

        # If the initially selected item is missing, auto-select the first non-missing
        # item so the group is immediately usable (and the play button stays enabled).
        if self._selected_item and self._selected_item in self._missing_paths:
            fallback = next((ip for ip in item_paths if ip not in self._missing_paths), "")
            self._selected_item = fallback
            if fallback and on_select_item:
                parsed = parse_group_header(group_path)
                gid = parsed[1] if parsed else ""
                on_select_item(gid, fallback)

        self._drag_start = QPoint()

        parsed = parse_group_header(group_path)
        self._group_id = parsed[1] if parsed else ""

        self.setAcceptDrops(True)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Header row
        header = QWidget()
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(4, 2, 4, 2)
        h_layout.setSpacing(6)

        handle = QLabel("⠿")
        handle.setCursor(Qt.CursorShape.OpenHandCursor)
        h_layout.addWidget(handle)

        self._play_btn = PlayButton()
        self._play_btn.setToolTip(f"Run selected item in group '{name}'")
        self._play_btn.setCursor(Qt.CursorShape.ArrowCursor)
        self._play_btn.clicked.connect(self._run_selected)
        h_layout.addWidget(self._play_btn)

        self._arrow = ArrowButton(collapsed=collapsed)
        self._arrow.clicked.connect(self._toggle_collapsed)
        h_layout.addWidget(self._arrow)

        self._name_lbl = QLabel(
            f"<b>{name} [{_resolve_path_name(selected_item)}]</b>"
            if selected_item else f"<b>{name}</b>"
        )
        h_layout.addWidget(self._name_lbl, stretch=1)

        self._check_btn = CheckButton()
        self._check_btn.setToolTip("Mark group as complete")
        self._check_btn.clicked.connect(self._toggle_completed)
        h_layout.addWidget(self._check_btn)

        self._notes_btn = PencilButton()
        self._notes_btn.clicked.connect(self._open_notes)
        h_layout.addWidget(self._notes_btn)

        self._star_btn = StarButton(is_fav=is_fav)
        self._star_btn.setToolTip("Remove from favorites" if is_fav else "Add to favorites")
        self._star_btn.clicked.connect(self._toggle_fav)
        h_layout.addWidget(self._star_btn)

        trash = TrashButton()
        trash.setToolTip("Remove group")
        trash.clicked.connect(self._confirm_remove)
        h_layout.addWidget(trash)

        self._apply_completed_style()
        self._update_notes_btn()
        self._update_play_btn()
        outer.addWidget(header)

        # Items container
        self._items_widget = QWidget()
        self._items_layout = QVBoxLayout(self._items_widget)
        self._items_layout.setContentsMargins(0, 0, 0, 0)
        self._items_layout.setSpacing(1)
        self._items_widget.setVisible(not collapsed)
        outer.addWidget(self._items_widget)

        self._rebuild_items()

    def _rebuild_items(self):
        while self._items_layout.count():
            item = self._items_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for ip in self.item_paths:
            name = _resolve_path_name(ip)
            is_missing = ip in self._missing_paths
            note_key = make_groupitem_path(self._group_id, ip)
            row = GroupItemRow(ip, name,
                               is_selected=(ip == self._selected_item),
                               on_select=self._on_item_selected,
                               on_remove=self._on_remove_item_clicked,
                               group_id=self._group_id,
                               missing=is_missing,
                               note=self._item_notes.get(note_key, ""),
                               on_notes_changed=self._on_notes_changed,
                               on_group_notes_btn_refresh=self._update_notes_btn)
            self._items_layout.addWidget(row)
        # Give the empty items area a minimum height so it is a real drop target.
        # Without this it has zero height and childAt()/geometry() can't find it.
        self._items_widget.setMinimumHeight(0 if self.item_paths else 36)
        self._update_notes_btn()

    def _update_play_btn(self):
        """Enable or disable the play button based on whether the selected item is missing."""
        is_missing = bool(self._selected_item and self._selected_item in self._missing_paths)
        self._play_btn.setEnabled(not is_missing)
        if is_missing:
            self._play_btn.setToolTip("Script not installed.")
        else:
            self._play_btn.setToolTip(f"Run selected item in group '{self.name}'")

    def _on_item_selected(self, item_path: str):
        self._selected_item = item_path
        self._rebuild_items()
        # Update label to show selected item name
        item_name = _resolve_path_name(item_path)
        self._name_lbl.setText(f"<b>{self.name} [{item_name}]</b>")
        self._update_play_btn()
        if self._on_select_item:
            self._on_select_item(self._group_id, item_path)

    def _on_remove_item_clicked(self, item_path: str):
        if self._on_remove_item:
            self._on_remove_item(self._group_id, item_path)

    def _toggle_completed(self):
        self._completed = not self._completed
        self._apply_completed_style()
        # Disable/enable all items in the group
        for i in range(self._items_layout.count()):
            w = self._items_layout.itemAt(i).widget()
            if w:
                w.setEnabled(not self._completed)
        if self._on_completed_toggle:
            self._on_completed_toggle(self.group_path, self._completed)

    def _apply_completed_style(self):
        if self._completed:
            self._name_lbl.setStyleSheet("text-decoration: line-through; color: #484f58;")
            self._check_btn.set_checked(True)
            self._check_btn.setToolTip("Mark group as incomplete")
        else:
            self._name_lbl.setStyleSheet("")
            self._check_btn.set_checked(False)
            self._check_btn.setToolTip("Mark group as complete")

    def _update_notes_btn(self):
        self._notes_btn.set_has_note(bool(self._note))
        has_item_notes = any(
            self._item_notes.get(make_groupitem_path(self._group_id, ip), "")
            for ip in self.item_paths
        )
        self._notes_btn.set_has_item_notes(has_item_notes)
        self._notes_btn.setToolTip(self._note if self._note else "Add or edit notes")

    def _open_notes(self):
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Notes - {self.name}")
        dlg.setMinimumWidth(400)
        dlg.setMinimumHeight(200)
        dlg.setWindowFlags(dlg.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(10, 10, 10, 8)
        layout.setSpacing(6)
        header_lbl = QLabel(self.name)
        hf = header_lbl.font()
        hf.setBold(True)
        hf.setPointSize(hf.pointSize() + 1)
        header_lbl.setFont(hf)
        layout.addWidget(header_lbl)
        layout.addWidget(QLabel("Group name:"))
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
        # Rename group if name changed
        new_name = name_edit.text().strip()
        if new_name and new_name != self.name and self._on_group_renamed:
            self._on_group_renamed(self.group_path, new_name)
        self._note = editor.toPlainText().strip()
        self._update_notes_btn()
        if self._on_notes_changed:
            # Use _note_key if set (assigned by rebuild() for duplicate-aware keying),
            # otherwise fall back to self.group_path for backward compatibility.
            key = getattr(self, "_note_key", self.group_path)
            self._on_notes_changed(key, self._note)

    def _toggle_collapsed(self):
        self._collapsed = not self._collapsed
        self._arrow.set_collapsed(self._collapsed)
        self._items_widget.setVisible(not self._collapsed)
        if self._on_collapsed_changed:
            self._on_collapsed_changed(self._group_id, self._collapsed)

    def _toggle_fav(self):
        self._is_fav = not self._is_fav
        self._star_btn.set_fav(self._is_fav)
        self._star_btn.setToolTip("Remove from favorites" if self._is_fav else "Add to favorites")
        if self._on_fav_toggle:
            self._on_fav_toggle(self.group_path, self._is_fav)

    def _confirm_remove(self):
        reply = QMessageBox.question(
            self, "Remove Group",
            f"Remove group '{self.name}' and all its items?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes and self._on_remove_group:
            self._on_remove_group(self.group_path)

    def _run_selected(self):
        if not self._selected_item or not self._on_run:
            return
        name = _resolve_path_name(self._selected_item)
        self._on_run(self._selected_item, name)
        self._flash_green()

    def _flash_green(self):
        self.setStyleSheet("background-color: #0d2a4a; border: 1px solid #1f6feb;")
        QTimer.singleShot(600, self._clear_flash)

    def _clear_flash(self):
        self.setStyleSheet("")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.pos()

    def mouseMoveEvent(self, event):
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        if (event.pos() - self._drag_start).manhattanLength() < QApplication.startDragDistance():
            return
        drag = make_drag(self, SRC_RIGHT, self.group_path)
        drag.exec(Qt.DropAction.MoveAction)

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(GROUPITEM_MIME):
            event.acceptProposedAction()
            return
        parsed = parse_mime(event)
        if parsed and parsed[1].startswith(STEP_PREFIX):
            event.ignore()
            return
        if parsed and parsed[0] == SRC_LEFT and not parsed[1].startswith(STEP_PREFIX):
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(GROUPITEM_MIME):
            event.acceptProposedAction()
            return
        parsed = parse_mime(event)
        if parsed and parsed[1].startswith(STEP_PREFIX):
            event.ignore()
            return
        if parsed and parsed[0] == SRC_LEFT and not parsed[1].startswith(STEP_PREFIX):
            event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        pass  # cursor managed by SelectedPanel

    def dropEvent(self, event):
        # Handle intra-group reorder OR cross-group move
        if event.mimeData().hasFormat(GROUPITEM_MIME):
            raw = bytes(event.mimeData().data(GROUPITEM_MIME)).decode()
            parts = raw.split("|", 1)
            src_group_id = parts[0] if len(parts) == 2 else ""
            src_path = parts[1] if len(parts) == 2 else ""
            # Cross-group move: remove from source group, add to this group
            if len(parts) == 2 and src_group_id != self._group_id:
                if src_path not in self.item_paths:
                    # Update checked_paths via parent callbacks, then let rebuild() redraw everything
                    if self._on_add_to_group:
                        self._on_add_to_group(self._group_id, src_path)
                    if self._on_remove_item:
                        self._on_remove_item(src_group_id, src_path)
                event.acceptProposedAction()
                return
            if len(parts) == 2 and parts[0] == self._group_id:
                src_path = parts[1]
                if src_path in self.item_paths:
                    # Find drop position within items layout
                    drop_y = event.position().toPoint().y()
                    drop_idx = len(self.item_paths)
                    for i in range(self._items_layout.count()):
                        w = self._items_layout.itemAt(i).widget()
                        if w and drop_y < w.y() + w.height() // 2:
                            drop_idx = i
                            break
                    src_idx = self.item_paths.index(src_path)
                    self.item_paths.pop(src_idx)
                    # When dragging down, subtract 1 since item was removed above target
                    dst_idx = drop_idx - 1 if drop_idx > src_idx else drop_idx
                    dst_idx = max(0, min(dst_idx, len(self.item_paths)))
                    self.item_paths.insert(dst_idx, src_path)
                    # Update checked_paths order for this group
                    if self._on_reorder_items:
                        self._on_reorder_items(self._group_id, self.item_paths)
                    self._rebuild_items()
            event.acceptProposedAction()
            return
        parsed = parse_mime(event)
        if not parsed or parsed[0] != SRC_LEFT:
            return
        _, item_path = parsed
        if item_path.startswith(STEP_PREFIX):
            return  # steps cannot be added to groups
        # Map drop y into _items_widget coordinate space.
        # If the drop landed on the header (items widget not visible or y < 0), append.
        if self._items_widget.isVisible():
            drop_y = self._items_widget.mapFromGlobal(
                self.mapToGlobal(event.position().toPoint())
            ).y()
            insert_idx = drop_index_in_group(self._items_layout, drop_y) if drop_y >= 0 else len(self.item_paths)
        else:
            insert_idx = len(self.item_paths)
        self.item_paths.insert(insert_idx, item_path)
        # If this is the first item, auto-select it
        if len(self.item_paths) == 1:
            self._selected_item = item_path
            if self._on_select_item:
                self._on_select_item(self._group_id, item_path)
        # Directly notify parent to add groupitem to checked_paths
        if self._on_add_to_group:
            self._on_add_to_group(self._group_id, item_path)
        self._rebuild_items()
        event.acceptProposedAction()


def _resolve_path_name(path: str) -> str:
    """Return a display name for any path type: script, step, function, or group."""
    if path.startswith(GROUP_PREFIX):
        parsed = parse_group_header(path)
        return parsed[0] if parsed else path
    if path.startswith(GROUPITEM_PREFIX):
        parsed = parse_groupitem(path)
        return _resolve_path_name(parsed[1]) if parsed else path
    if path.startswith(STEP_PREFIX):
        label = path[len(STEP_PREFIX):]
        # Strip unique copy suffix (#xxxxxx) for display
        return label.rsplit("#", 1)[0] if "#" in label and len(label.rsplit("#", 1)[-1]) == 6 else label
    if path.startswith(FUNCTION_PREFIX):
        dialog_name = path[len(FUNCTION_PREFIX):]
        # Look up in pre-built cache first to avoid importing DialogID on every call
        if _DIALOG_GROUPS:
            for dialogs in _DIALOG_GROUPS.values():
                for d in dialogs:
                    if d.name == dialog_name:
                        return d.label
        return dialog_name
    if path.startswith(SYSTEM_PREFIX):
        return path[len(SYSTEM_PREFIX):]
    return os.path.basename(path)

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
                 on_completed_toggle=None, step_colors: dict = None,
                 on_color_changed=None, last_step_color: str = DEFAULT_STEP_COLOR,
                 on_group_renamed=None, on_group_collapsed=None, parent=None):
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
        self.on_group_renamed = on_group_renamed
        self.on_group_collapsed = on_group_collapsed
        self.completed_paths: set = completed_paths if completed_paths is not None else set()
        self.on_completed_toggle = on_completed_toggle
        self.step_colors: dict = step_colors if step_colors is not None else {}
        self.on_color_changed = on_color_changed
        self.last_step_color: str = last_step_color
        self.group_selections: dict = {}  # {group_id: selected_item_path}
        self.group_collapsed: dict = {}   # {group_id: bool}
        self._forbidden_cursor_active: bool = False
        self._workflow_name: str = ""
        self.setAcceptDrops(True)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)

        self._title_lbl = QLabel("Selected Items")
        self._title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(self._title_lbl)

        self._hint_lbl = QLabel("Drag available items here to add")
        self._hint_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hint_lbl.setWordWrap(True)
        outer.addWidget(self._hint_lbl)

        # ── Pinned favorites zone (does not scroll) ───────────────────────────
        self._fav_container = QWidget()
        self._fav_container.setAcceptDrops(True)
        self._fav_container.dragEnterEvent  = self._fav_dragEnterEvent
        self._fav_container.dragMoveEvent   = self._fav_dragMoveEvent
        self._fav_container.dragLeaveEvent  = self._fav_dragLeaveEvent
        self._fav_container.dropEvent       = self._fav_dropEvent
        self._fav_layout = QVBoxLayout(self._fav_container)
        self._fav_layout.setContentsMargins(4, 2, 4, 0)
        self._fav_layout.setSpacing(2)
        self._fav_container.setVisible(False)  # hidden until there are favorites
        outer.addWidget(self._fav_container)

        # Divider shown between pinned favs and the scrollable list
        self._fav_divider = QFrame()
        self._fav_divider.setFrameShape(QFrame.Shape.HLine)
        self._fav_divider.setFrameShadow(QFrame.Shadow.Sunken)
        self._fav_divider.setVisible(False)
        outer.addWidget(self._fav_divider)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(self._scroll)

        # Workflow combo sits below the script list, above the footer buttons
        workflows_lbl = QLabel("Workflows")
        workflows_lbl.setAlignment(Qt.AlignmentFlag.AlignLeft)
        outer.addWidget(workflows_lbl)

        # Workflow row: combo + Export / Import / Delete buttons
        wf_row = QHBoxLayout()
        wf_row.setSpacing(4)
        wf_row.setContentsMargins(0, 0, 0, 0)
        self._workflow_combo = WorkflowCombo()
        self._workflow_combo.setToolTip("Load a saved workflow")
        wf_row.addWidget(self._workflow_combo, stretch=1)

        self._wf_export_btn = QPushButton("Export")
        self._wf_export_btn.setToolTip("Export the current workflow to a JSON file")
        self._wf_export_btn.setFixedHeight(26)
        wf_row.addWidget(self._wf_export_btn)

        self._wf_import_btn = QPushButton("Import")
        self._wf_import_btn.setToolTip("Import a workflow from a JSON file")
        self._wf_import_btn.setFixedHeight(26)
        wf_row.addWidget(self._wf_import_btn)

        self._wf_delete_btn = TrashButton()
        self._wf_delete_btn.setToolTip("Delete the selected workflow")
        wf_row.addWidget(self._wf_delete_btn)

        outer.addLayout(wf_row)
        self._workflow_desc_lbl = QLabel("")
        self._workflow_desc_lbl.setWordWrap(True)
        self._workflow_desc_lbl.setStyleSheet("color: #6e7681; font-style: italic;")
        self._workflow_desc_lbl.setOpenExternalLinks(True)
        self._workflow_desc_lbl.setVisible(False)
        outer.addWidget(self._workflow_desc_lbl)

        self._container = QWidget()
        self._rows_layout = QVBoxLayout(self._container)
        self._rows_layout.setContentsMargins(4, 4, 4, 4)
        self._rows_layout.setSpacing(2)
        self._rows_layout.addStretch()
        self._scroll.setWidget(self._container)

        self._empty_min_width = 350  # px - generous drop-zone when empty
        self.rebuild()

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        if not self.checked_paths:
            return QSize(self._empty_min_width, hint.height())
        return hint

    def sizeHint(self):
        hint = super().sizeHint()
        if not self.checked_paths:
            return QSize(self._empty_min_width, hint.height())
        return hint

    def set_workflow_name(self, name: str) -> None:
        """Update the displayed workflow name label and refresh the layout."""
        self._workflow_name = name
        self.rebuild()

    def _make_fav_row(self, path: str, group_items: dict) -> "QWidget":
        """Build a row widget for a favorited item (used by both rebuild and fav-zone drop)."""
        _name = _resolve_path_name(path)
        if path.startswith(GROUP_PREFIX):
            parsed = parse_group_header(path)
            gid = parsed[1] if parsed else ""
            item_paths = group_items.get(gid, [])
            return GroupRow(path, _name, item_paths,
                            selected_item=self.group_selections.get(gid, item_paths[0] if item_paths else ""),
                            on_run=self.on_run, is_fav=True,
                            on_fav_toggle=self.on_fav_toggle,
                            on_remove_group=self.on_remove,
                            on_remove_item=self._on_remove_group_item,
                            on_select_item=self._on_group_item_selected,
                            on_collapsed_changed=self._on_group_collapsed,
                            on_add_to_group=self._on_add_to_group,
                            collapsed=self.group_collapsed.get(gid, False),
                            completed=path in self.completed_paths,
                            note=self.notes.get(path, ""),
                            on_notes_changed=self.on_notes_changed,
                            on_completed_toggle=self.on_completed_toggle,
                            on_group_renamed=self.on_group_renamed,
                            on_reorder_items=self._on_reorder_group_items,
                            missing_paths=self.missing_paths,
                            item_notes=self.notes)
        else:
            return RightScriptRow(path, _name, self.on_run,
                                  is_fav=True, on_fav_toggle=self.on_fav_toggle,
                                  on_remove=self.on_remove, on_copy=self.on_copy,
                                  missing=path in self.missing_paths,
                                  note=self.notes.get(path, ""),
                                  on_notes_changed=self.on_notes_changed,
                                  on_step_renamed=self.on_step_renamed,
                                  completed=path in self.completed_paths,
                                  on_completed_toggle=self.on_completed_toggle,
                                  step_color=self.step_colors.get(path, self.last_step_color),
                                  on_color_changed=self.on_color_changed)

    def rebuild(self, scroll_to_bottom: bool = False):
        # ── Clear pinned fav zone ─────────────────────────────────────────────
        while self._fav_layout.count():
            item = self._fav_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # ── Clear scrollable list ─────────────────────────────────────────────
        while self._rows_layout.count() > 1:
            item = self._rows_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Favourites are pinned above the scroll area (groups can be favourited too)
        favs = self.favorites
        # Single pass: split into fav/other top-level rows and collect groupitems per group
        fav_paths = []
        other_paths = []
        group_items: dict[str, list] = {}  # gid -> [item_path, ...]
        for p in self.checked_paths:
            if p.startswith(GROUPITEM_PREFIX):
                gi = parse_groupitem(p)
                if gi:
                    group_items.setdefault(gi[0], []).append(gi[1])
            elif p in favs:
                fav_paths.append(p)
            else:
                other_paths.append(p)

        # ── Populate pinned fav zone ──────────────────────────────────────────
        for path in fav_paths:
            row = self._make_fav_row(path, group_items)
            self._fav_layout.addWidget(row)

        self._fav_container.setVisible(bool(fav_paths))
        self._fav_divider.setVisible(bool(fav_paths))

        # ── Populate scrollable list ──────────────────────────────────────────
        pos = 0
        # Workflow name label — shown above main items list
        if self._workflow_name and self.checked_paths:
            wf_lbl = QLabel(self._workflow_name)
            wf_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            wf_lbl.setStyleSheet(
                "color: #8b949e; font-style: italic; font-size: 8pt;"
                " padding: 2px 0px 2px 0px;"
            )
            self._rows_layout.insertWidget(pos, wf_lbl)
            pos += 1
        # Track how many times each path has been seen so duplicate items get
        # independent note keys (path\x00N for the Nth occurrence, N>=1).
        _path_occurrence: dict[str, int] = {}
        for path in other_paths:
            _name = _resolve_path_name(path)
            # Compute a unique note key for this row occurrence.
            # Steps already have unique paths via #uuid suffix, so they don't need
            # special handling, but we apply the same logic uniformly for safety.
            _occ = _path_occurrence.get(path, 0)
            _path_occurrence[path] = _occ + 1
            note_key = path if _occ == 0 else f"{path}\x00{_occ}"
            if path.startswith(GROUP_PREFIX):
                parsed = parse_group_header(path)
                gid = parsed[1] if parsed else ""
                item_paths = group_items.get(gid, [])
                row = GroupRow(path, _name, item_paths,
                               selected_item=self.group_selections.get(gid, item_paths[0] if item_paths else ""),
                               on_run=self.on_run, is_fav=False,
                               on_fav_toggle=self.on_fav_toggle,
                               on_remove_group=self.on_remove,
                               on_remove_item=self._on_remove_group_item,
                               on_select_item=self._on_group_item_selected,
                               on_collapsed_changed=self._on_group_collapsed,
                               on_add_to_group=self._on_add_to_group,
                               collapsed=self.group_collapsed.get(gid, False),
                               completed=path in self.completed_paths,
                               note=self.notes.get(note_key, ""),
                               on_notes_changed=self.on_notes_changed,
                               on_completed_toggle=self.on_completed_toggle,
                               on_group_renamed=self.on_group_renamed,
                               on_reorder_items=self._on_reorder_group_items,
                               missing_paths=self.missing_paths,
                               item_notes=self.notes)
                # Store the unique note key on the row so _open_notes reports it correctly
                row._note_key = note_key
            else:
                row = RightScriptRow(path, _name, self.on_run,
                                     is_fav=False, on_fav_toggle=self.on_fav_toggle,
                                     on_remove=self.on_remove, on_copy=self.on_copy,
                                     missing=path in self.missing_paths,
                                     note=self.notes.get(note_key, ""),
                                     on_notes_changed=self.on_notes_changed,
                                     on_step_renamed=self.on_step_renamed,
                                     completed=path in self.completed_paths,
                                     on_completed_toggle=self.on_completed_toggle,
                                     step_color=self.step_colors.get(path, self.last_step_color),
                                     on_color_changed=self.on_color_changed)
                # Store the unique note key on the row so _open_notes reports it correctly
                row._note_key = note_key
            self._rows_layout.insertWidget(pos, row)
            pos += 1

        if scroll_to_bottom:
            sb = self._scroll.verticalScrollBar()
            def _on_range_changed(lo, hi):
                if hi > 0:
                    try:
                        sb.rangeChanged.disconnect(_on_range_changed)
                    except (RuntimeError, TypeError):
                        pass
                    sb.setValue(hi)
            sb.rangeChanged.connect(_on_range_changed)
            def _cleanup():
                try:
                    sb.rangeChanged.disconnect(_on_range_changed)
                except (RuntimeError, TypeError):
                    pass
            QTimer.singleShot(500, _cleanup)

    # ── Pinned fav zone drag-and-drop ─────────────────────────────────────────

    def _fav_drag_validate(self, event) -> bool:
        """Validate a drag event over the fav zone; manage the forbidden cursor.
        Returns True if the drag is acceptable (right-panel fav item), False otherwise."""
        parsed = parse_mime(event)
        is_valid = bool(parsed and parsed[0] == SRC_RIGHT and parsed[1] in self.favorites)
        if is_valid:
            if self._forbidden_cursor_active:
                self._forbidden_cursor_active = False
                QApplication.restoreOverrideCursor()
            event.acceptProposedAction()
        else:
            if not self._forbidden_cursor_active:
                self._forbidden_cursor_active = True
                QApplication.setOverrideCursor(Qt.CursorShape.ForbiddenCursor)
            event.setDropAction(Qt.DropAction.IgnoreAction)
            event.accept()
        return is_valid

    def _fav_dragEnterEvent(self, event):
        self._fav_drag_validate(event)

    def _fav_dragMoveEvent(self, event):
        self._fav_drag_validate(event)

    def _fav_dragLeaveEvent(self, event):
        if self._forbidden_cursor_active:
            self._forbidden_cursor_active = False
            QApplication.restoreOverrideCursor()

    def _fav_dropEvent(self, event):
        parsed = parse_mime(event)
        if not parsed:
            event.ignore()
            return
        source_id, path = parsed
        if source_id != SRC_RIGHT or path not in self.favorites:
            event.ignore()
            return

        # Find insertion index within _fav_layout based on drop Y position
        drop_y = event.position().toPoint().y()
        insert_idx = self._fav_layout.count()  # default: append
        for i in range(self._fav_layout.count()):
            item = self._fav_layout.itemAt(i)
            w = item.widget() if item else None
            if w is None:
                continue
            mid_y = w.y() + w.height() // 2
            if drop_y < mid_y:
                insert_idx = i
                break

        # Reorder within checked_paths: collect current fav order, apply new order
        favs = self.favorites
        fav_paths = [p for p in self.checked_paths
                     if p in favs and not p.startswith(GROUPITEM_PREFIX)]
        if path not in fav_paths:
            event.ignore()
            return

        fav_paths.remove(path)
        insert_idx = max(0, min(insert_idx, len(fav_paths)))
        fav_paths.insert(insert_idx, path)

        # Rebuild checked_paths preserving groupitems after their group headers
        new_paths = []
        for p in fav_paths:
            new_paths.append(p)
            if p.startswith(GROUP_PREFIX):
                parsed_gh = parse_group_header(p)
                if parsed_gh:
                    gid = parsed_gh[1]
                    new_paths.extend(
                        gi for gi in self.checked_paths
                        if gi.startswith(GROUPITEM_PREFIX)
                        and parse_groupitem(gi) is not None
                        and parse_groupitem(gi)[0] == gid
                    )
        for p in self.checked_paths:
            if p in favs and not p.startswith(GROUPITEM_PREFIX):
                continue  # already added above
            if p.startswith(GROUPITEM_PREFIX):
                gi = parse_groupitem(p)
                if gi:
                    # Only add orphaned groupitems (those whose group header is not a fav)
                    parent_group_path = next(
                        (cp for cp in self.checked_paths
                         if cp.startswith(GROUP_PREFIX)
                         and (_pgh := parse_group_header(cp)) is not None
                         and _pgh[1] == gi[0]),
                        None
                    )
                    if parent_group_path and parent_group_path not in favs:
                        new_paths.append(p)
                continue
            new_paths.append(p)
            if p.startswith(GROUP_PREFIX):
                parsed_gh = parse_group_header(p)
                if parsed_gh:
                    gid = parsed_gh[1]
                    new_paths.extend(
                        gi for gi in self.checked_paths
                        if gi.startswith(GROUPITEM_PREFIX)
                        and parse_groupitem(gi) is not None
                        and parse_groupitem(gi)[0] == gid
                    )

        self.checked_paths[:] = new_paths
        self.rebuild()
        if self.on_changed:
            self.on_changed()
        event.acceptProposedAction()

    def _on_add_to_group(self, group_id: str, item_path: str):
        """Add a groupitem path to checked_paths when dropped onto a group."""
        gi_path = make_groupitem_path(group_id, item_path)
        is_first = not any(
            p.startswith(GROUPITEM_PREFIX)
            and parse_groupitem(p) is not None
            and parse_groupitem(p)[0] == group_id
            for p in self.checked_paths
        )
        # Insert after the last existing groupitem for this group, or after the group header
        insert_idx = None
        for i, p in enumerate(self.checked_paths):
            if p.startswith(GROUP_PREFIX):
                parsed = parse_group_header(p)
                if parsed and parsed[1] == group_id:
                    insert_idx = i + 1
            elif p.startswith(GROUPITEM_PREFIX):
                parsed = parse_groupitem(p)
                if parsed and parsed[0] == group_id:
                    insert_idx = i + 1
        if insert_idx is not None:
            self.checked_paths.insert(insert_idx, gi_path)
        else:
            self.checked_paths.append(gi_path)
        # Auto-select first item and rebuild so label updates immediately
        if is_first:
            self.group_selections[group_id] = item_path
        self.rebuild(scroll_to_bottom=True)
        if self.on_changed:
            self.on_changed()

    def _on_group_item_selected(self, group_id: str, item_path: str):
        self.group_selections[group_id] = item_path

    def _on_remove_group_item(self, group_id: str, item_path: str):
        gi_path = make_groupitem_path(group_id, item_path)
        if gi_path in self.checked_paths:
            self.checked_paths.remove(gi_path)
        # If the removed item was selected, auto-select another item in the group
        if self.group_selections.get(group_id) == item_path:
            remaining = [
                parse_groupitem(p)[1]
                for p in self.checked_paths
                if p.startswith(GROUPITEM_PREFIX)
                and parse_groupitem(p)
                and parse_groupitem(p)[0] == group_id
            ]
            if remaining:
                self.group_selections[group_id] = remaining[0]
            else:
                self.group_selections.pop(group_id, None)
        # If also in main list, keep it there
        if self.on_changed:
            self.on_changed()
        self.rebuild()

    def _on_reorder_group_items(self, group_id: str, new_order: list):
        """Reorder groupitem entries in checked_paths to match new_order."""
        # Remove old groupitems for this group
        old_items = [p for p in self.checked_paths
                     if p.startswith(GROUPITEM_PREFIX)
                     and parse_groupitem(p) is not None
                     and parse_groupitem(p)[0] == group_id]
        for p in old_items:
            self.checked_paths.remove(p)
        # Find insertion point (after group header)
        insert_idx = next(
            (i + 1 for i, p in enumerate(self.checked_paths)
             if p.startswith(GROUP_PREFIX)
             and (_pgh := parse_group_header(p)) is not None
             and _pgh[1] == group_id),
            len(self.checked_paths)
        )
        for item_path in reversed(new_order):
            self.checked_paths.insert(insert_idx, make_groupitem_path(group_id, item_path))
        if self.on_changed:
            self.on_changed()

    def _on_group_collapsed(self, group_id: str, collapsed: bool):
        self.group_collapsed[group_id] = collapsed
        if self.on_group_collapsed:
            self.on_group_collapsed(group_id, collapsed)

    # ── drop handling ─────────────────────────────────────────────────────────

    def _is_over_group(self, pos_in_viewport) -> bool:
        """Return True if pos_in_viewport is over a GroupRow widget."""
        return self._group_row_at(pos_in_viewport) is not None

    def _group_row_at(self, pos_in_viewport) -> "GroupRow | None":
        """Return the GroupRow under pos_in_viewport, or None."""
        w = self._scroll.viewport().childAt(pos_in_viewport)
        while w is not None:
            if isinstance(w, GroupRow):
                return w
            w = w.parent() if hasattr(w, "parent") else None
        return None

    def _group_row_at_items_area(self, pos_in_viewport) -> "GroupRow | None":
        """Return the GroupRow under pos_in_viewport only if the cursor is over
        its expanded items area (not the header).

        This is used for SRC_RIGHT drops (main-list reorder) so that an item
        dragged past the top or bottom edge of a group is NOT accidentally
        swallowed into the group.  The header area acts as a neutral boundary:
        dropping on it places the item before or after the group in the main
        list rather than inside it.  The full group area (including header)
        continues to work as a drop target for SRC_LEFT drops (adding from the
        left panel) because those use _group_row_at instead.
        """
        w = self._scroll.viewport().childAt(pos_in_viewport)
        while w is not None:
            if isinstance(w, GroupRow):
                items_widget = w._items_widget
                if items_widget.isVisible():
                    local_pos = items_widget.mapFromGlobal(
                        self._scroll.viewport().mapToGlobal(pos_in_viewport))
                    if items_widget.rect().contains(local_pos):
                        return w
                return None  # cursor is over the header only → don't capture
            w = w.parent() if hasattr(w, "parent") else None
        return None

    def _viewport_pos(self, event) -> "QPoint":
        return self._scroll.viewport().mapFromGlobal(
            self.mapToGlobal(event.position().toPoint())
        )

    def _drop_groupitem_into_group(self, item_path: str, src_gid: str,
                                   tgt_group_row: "GroupRow") -> None:
        """Move a groupitem from src_gid into tgt_group_row."""
        tgt_gid = tgt_group_row._group_id
        self._on_remove_group_item(src_gid, item_path)
        gi_path = make_groupitem_path(tgt_gid, item_path)
        insert_idx = next(
            (i + 1 for i, p in enumerate(self.checked_paths)
             if p.startswith(GROUPITEM_PREFIX)
             and parse_groupitem(p) is not None
             and parse_groupitem(p)[0] == tgt_gid),
            len(self.checked_paths)
        )
        self.checked_paths.insert(insert_idx, gi_path)
        if not tgt_group_row.item_paths:
            self.group_selections[tgt_gid] = item_path
        tgt_group_row.item_paths.append(item_path)
        tgt_group_row._rebuild_items()
        if self.on_changed:
            self.on_changed()
        self.rebuild()

    def _eject_groupitem_to_main(self, item_path: str, src_gid: str,
                                  pos_in_viewport: "QPoint") -> None:
        """Eject a groupitem from its group and insert it as a standalone row."""
        # Compute the layout drop index first (widget positions are still valid).
        layout_idx = drop_index_in_panel(self._scroll, self._rows_layout,
                                         pos_in_viewport, skip_widget=None)
        # Remove from group BEFORE converting to a checked_paths index so that
        # _layout_idx_to_cp_idx sees the correct (reduced) item count for the
        # source group.  Computing cp_idx before removal caused it to be inflated
        # by the number of groupitems in the source group whenever the group
        # appeared above the drop point.
        gi_path = make_groupitem_path(src_gid, item_path)
        if gi_path in self.checked_paths:
            self.checked_paths.remove(gi_path)
        if self.group_selections.get(src_gid) == item_path:
            remaining = [parse_groupitem(p)[1] for p in self.checked_paths
                         if p.startswith(GROUPITEM_PREFIX)
                         and parse_groupitem(p) is not None
                         and parse_groupitem(p)[0] == src_gid]
            if remaining:
                self.group_selections[src_gid] = remaining[0]
            else:
                self.group_selections.pop(src_gid, None)
        # Now convert layout index → checked_paths index with the updated list.
        cp_idx = self._layout_idx_to_cp_idx(layout_idx)
        cp_idx = max(0, min(cp_idx, len(self.checked_paths)))
        if item_path not in self.checked_paths:
            self.checked_paths.insert(cp_idx, item_path)
        if self.on_changed:
            self.on_changed()
        self.rebuild()

    def _move_main_item_into_group(self, path: str, group_row: "GroupRow",
                                    pos_in_viewport: "QPoint") -> None:
        """Move a main-list item into a group at the drop position."""
        gid = group_row._group_id
        gi_path = make_groupitem_path(gid, path)
        if group_row._items_widget.isVisible():
            global_drop = self._scroll.viewport().mapToGlobal(pos_in_viewport)
            drop_y = group_row._items_widget.mapFromGlobal(global_drop).y()
            item_idx = (drop_index_in_group(group_row._items_layout, drop_y)
                        if drop_y >= 0 else len(group_row.item_paths))
        else:
            item_idx = len(group_row.item_paths)
        self.checked_paths.remove(path)
        self.favorites.discard(path)
        # Find insert position in checked_paths
        base_idx = None
        item_count = 0
        for i, p in enumerate(self.checked_paths):
            if p.startswith(GROUP_PREFIX):
                _gh = parse_group_header(p)
                if _gh and _gh[1] == gid:
                    base_idx = i + 1
            elif p.startswith(GROUPITEM_PREFIX):
                _gi = parse_groupitem(p)
                if _gi and _gi[0] == gid:
                    item_count += 1
        cp_insert = (base_idx + min(item_idx, item_count)
                     if base_idx is not None else len(self.checked_paths))
        self.checked_paths.insert(cp_insert, gi_path)
        if not group_row.item_paths:
            self.group_selections[gid] = path
        group_row.item_paths.insert(item_idx, path)
        group_row._rebuild_items()
        self.on_changed()
        self.rebuild()

    def _reorder_main_list(self, path: str, idx: int) -> None:
        """Reorder a main-list item to a new layout position.

        idx is a 0-based index into the scrollable list rows (GroupRow /
        RightScriptRow only, as returned by drop_index_in_panel).  Favourites
        live in the separate pinned zone above the scroll area, so they are
        never part of this list and must be kept out of the index maths.
        """
        favs = self.favorites

        # Only the non-fav, non-groupitem paths are displayed in _rows_layout.
        other_paths = [p for p in self.checked_paths
                       if p not in favs and not p.startswith(GROUPITEM_PREFIX)]

        # Find the source position within other_paths.
        # Guard against duplicate step paths by counting how many times the
        # path appears before the dragged widget in the layout.
        layout_paths = []
        for i in range(self._rows_layout.count()):
            w = self._rows_layout.itemAt(i).widget() if self._rows_layout.itemAt(i) else None
            if isinstance(w, GroupRow):
                layout_paths.append(w.group_path)
            elif isinstance(w, RightScriptRow):
                layout_paths.append(w.path)

        drag_layout_idx = next((i for i, p in enumerate(layout_paths) if p == path), None)
        occurrences = [i for i, p in enumerate(other_paths) if p == path]
        if drag_layout_idx is not None and occurrences:
            preceding = sum(1 for p in layout_paths[:drag_layout_idx] if p == path)
            src_idx = occurrences[min(preceding, len(occurrences) - 1)]
        else:
            src_idx = next((i for i, p in enumerate(other_paths) if p == path), None)
            if src_idx is None:
                return

        # idx from drop_index_in_panel directly maps to a position in other_paths
        # because _rows_layout contains exactly the same items in the same order.
        dst_idx = max(0, min(idx, len(other_paths)))

        other_paths.pop(src_idx)
        if src_idx < dst_idx:
            dst_idx -= 1
        dst_idx = max(0, min(dst_idx, len(other_paths)))
        other_paths.insert(dst_idx, path)

        # Rebuild checked_paths: favs first (preserving their order), then
        # other_paths with groupitems re-inserted after their group headers.
        fav_paths = [p for p in self.checked_paths
                     if p in favs and not p.startswith(GROUPITEM_PREFIX)]
        new_paths = []
        for p in fav_paths:
            new_paths.append(p)
            if p.startswith(GROUP_PREFIX):
                _gh = parse_group_header(p)
                if _gh:
                    gid = _gh[1]
                    new_paths.extend(
                        gi for gi in self.checked_paths
                        if gi.startswith(GROUPITEM_PREFIX)
                        and parse_groupitem(gi) is not None
                        and parse_groupitem(gi)[0] == gid
                    )
        for p in other_paths:
            new_paths.append(p)
            if p.startswith(GROUP_PREFIX):
                _gh = parse_group_header(p)
                if _gh:
                    gid = _gh[1]
                    new_paths.extend(
                        gi for gi in self.checked_paths
                        if gi.startswith(GROUPITEM_PREFIX)
                        and parse_groupitem(gi) is not None
                        and parse_groupitem(gi)[0] == gid
                    )
        self.checked_paths[:] = new_paths

    def _layout_idx_to_cp_idx(self, layout_idx: int) -> int:
        """Convert a layout row index to a checked_paths insertion index.
        Groups occupy 1 layout slot but 1 + N checked_paths slots.
        Favorites live in the pinned zone above the scroll area and are stored
        at the front of checked_paths, so the result must be offset past them."""
        # Count how many checked_paths entries belong to favorites (including
        # groupitems inside favorited groups) — these precede the scrollable rows.
        fav_cp_count = 0
        for p in self.checked_paths:
            if p.startswith(GROUPITEM_PREFIX):
                gi = parse_groupitem(p)
                gid = gi[0] if gi else None
                group_path = next(
                    (cp for cp in self.checked_paths
                     if cp.startswith(GROUP_PREFIX)
                     and (_pgh := parse_group_header(cp)) is not None
                     and _pgh[1] == gid),
                    None
                )
                if group_path in self.favorites:
                    fav_cp_count += 1
                else:
                    break  # groupitem belongs to a non-favorite group → done
            elif p in self.favorites:
                fav_cp_count += 1
            else:
                break  # first non-favorite non-groupitem → done
        cp_idx = fav_cp_count
        layout_row_count = 0
        for i in range(self._rows_layout.count()):
            if layout_row_count >= layout_idx:
                break
            w = self._rows_layout.itemAt(i).widget() if self._rows_layout.itemAt(i) else None
            if not isinstance(w, (GroupRow, RightScriptRow)):
                continue  # divider, workflow-name label, or other non-row widget
            layout_row_count += 1
            if isinstance(w, GroupRow):
                cp_idx += 1 + sum(
                    1 for p in self.checked_paths
                    if p.startswith(GROUPITEM_PREFIX)
                    and parse_groupitem(p) is not None
                    and parse_groupitem(p)[0] == w._group_id
                )
            else:
                cp_idx += 1
        return cp_idx

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(GROUPITEM_MIME) or parse_mime(event):
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if not (event.mimeData().hasFormat(GROUPITEM_MIME) or parse_mime(event)):
            return
        parsed = parse_mime(event)
        pos_in_viewport = self._scroll.viewport().mapFromGlobal(
            self.mapToGlobal(event.position().toPoint())
        )
        over_group = self._is_over_group(pos_in_viewport)
        # Forbidden if: a step is dragged over a group, OR a favorite is dragged
        # over the main scroll list (favorites can only be reordered in the pinned zone)
        is_forbidden = (
            (parsed and parsed[1].startswith(STEP_PREFIX) and over_group)
            or (parsed and parsed[0] == SRC_RIGHT and parsed[1] in self.favorites)
        )
        if is_forbidden:
            if not self._forbidden_cursor_active:
                self._forbidden_cursor_active = True
                QApplication.setOverrideCursor(Qt.CursorShape.ForbiddenCursor)
            event.setDropAction(Qt.DropAction.IgnoreAction)
            event.accept()
        else:
            if self._forbidden_cursor_active:
                self._forbidden_cursor_active = False
                QApplication.restoreOverrideCursor()
            event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        if self._forbidden_cursor_active:
            self._forbidden_cursor_active = False
            QApplication.restoreOverrideCursor()

    def dropEvent(self, event):
        if self._forbidden_cursor_active:
            self._forbidden_cursor_active = False
            QApplication.restoreOverrideCursor()

        pos = self._viewport_pos(event)
        if event.mimeData().hasFormat(GROUPITEM_MIME):
            raw = bytes(event.mimeData().data(GROUPITEM_MIME)).decode()
            parts = raw.split("|", 1)
            if len(parts) != 2:
                return
            src_gid, item_path = parts
            target = self._group_row_at(pos)
            if target is not None and target._group_id != src_gid:
                self._drop_groupitem_into_group(item_path, src_gid, target)
            else:
                self._eject_groupitem_to_main(item_path, src_gid, pos)
            event.acceptProposedAction()
            return

        parsed = parse_mime(event)
        if not parsed:
            return
        source_id, path = parsed
        # Find the widget being dragged (for skip_widget in drop_index_in_panel)
        src_widget = None
        if path.startswith(GROUP_PREFIX):
            for i in range(self._rows_layout.count()):
                w = self._rows_layout.itemAt(i).widget() if self._rows_layout.itemAt(i) else None
                if isinstance(w, GroupRow) and w.group_path == path:
                    src_widget = w
                    break
        idx = drop_index_in_panel(self._scroll, self._rows_layout, pos,
                                  skip_widget=src_widget)

        # ── Drop from left panel ───────────────────────────────────────────────
        if source_id == SRC_LEFT:
            target = self._group_row_at(pos)
            if target is not None:
                gid = target._group_id
                gi_path = make_groupitem_path(gid, path)
                insert_idx = next(
                    (i + 1 for i, p in enumerate(self.checked_paths)
                     if p.startswith(GROUPITEM_PREFIX)
                     and parse_groupitem(p) is not None
                     and parse_groupitem(p)[0] == gid),
                    len(self.checked_paths)
                )
                self.checked_paths.insert(insert_idx, gi_path)
                if not target.item_paths:
                    self.group_selections[gid] = path
                target.item_paths.append(path)
                target._rebuild_items()
                self.on_changed()
                sb = self._scroll.verticalScrollBar()
                def _on_range_changed(lo, hi):
                    if hi > 0:
                        try:
                            sb.rangeChanged.disconnect(_on_range_changed)
                        except (RuntimeError, TypeError):
                            pass
                        sb.setValue(hi)
                sb.rangeChanged.connect(_on_range_changed)
                def _cleanup():
                    try:
                        sb.rangeChanged.disconnect(_on_range_changed)
                    except (RuntimeError, TypeError):
                        pass
                QTimer.singleShot(500, _cleanup)
                event.acceptProposedAction()
                return
            self.checked_paths.insert(self._layout_idx_to_cp_idx(idx), path)

        # ── Reorder / move from right panel ───────────────────────────────────
        elif source_id == SRC_RIGHT:
            _in_use = path in self.checked_paths or any(
                parse_groupitem(p) is not None and parse_groupitem(p)[1] == path
                for p in self.checked_paths if p.startswith(GROUPITEM_PREFIX)
            )
            if not _in_use:
                return
            # Favorites live in the pinned zone above; ignore drops of fav items
            # onto the scrollable list (reorder them via the fav zone instead).
            if path in self.favorites:
                event.ignore()
                return
            # Use _group_row_at_items_area (not _group_row_at) so that a drop
            # on a group's header counts as "between items in the main list",
            # allowing an item to be placed before or after the group without
            # being absorbed into it.
            # Exception: if the group is collapsed its items area is hidden, so
            # fall back to _group_row_at so the header still acts as a drop target.
            target = self._group_row_at_items_area(pos)
            if target is None and not path.startswith((GROUP_PREFIX, STEP_PREFIX)):
                candidate = self._group_row_at(pos)
                if candidate is not None and candidate._collapsed:
                    target = candidate
            if target is not None and not path.startswith((GROUP_PREFIX, STEP_PREFIX)):
                self._move_main_item_into_group(path, target, pos)
                event.acceptProposedAction()
                return
            self._reorder_main_list(path, idx)

        self.on_changed()
        self.rebuild(scroll_to_bottom=(source_id == SRC_LEFT))
        event.acceptProposedAction()

# ── Main window ───────────────────────────────────────────────────────────────


class WorkflowCompanionWindow(QMainWindow):

    def __init__(self, scripts_dir: str, scripts: list[tuple[str, str]]):
        super().__init__()
        self.scripts_dir = scripts_dir
        self.scripts = scripts
        self._siril = s.SirilInterface()

        valid_paths = {p for _, p in self.scripts}
        _sel, _favs, _geom, _workflows, _active, _notes, _on_top = load_state(scripts_dir)
        self.missing_paths: set = {p for p in _sel
                                   if p not in valid_paths
                                   and not p.startswith(NON_SCRIPT_PREFIXES)}
        # Also capture missing scripts inside groups (groupitem:// paths)
        for _p in _sel:
            if _p.startswith(GROUPITEM_PREFIX):
                _gi = parse_groupitem(_p)
                if _gi:
                    _item_path = _gi[1]
                    if (_item_path not in valid_paths
                            and not _item_path.startswith(NON_SCRIPT_PREFIXES)):
                        self.missing_paths.add(_item_path)
        self.checked_paths: list = list(_sel)  # keep all, including missing
        self.favorites: set = {p for p in _favs if p in valid_paths or p.startswith((STEP_PREFIX, FUNCTION_PREFIX, GROUP_PREFIX))}
        self._saved_geometry: dict | None = _geom
        self.workflows: dict = _workflows
        self._active_workflow: str = _active
        self._workflow_dirty: bool = False
        self.notes: dict = _notes  # {path: note_text}
        self.step_colors: dict = {}  # {path: emoji} - loaded per workflow
        # Load last used step color from state
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as _fh:
                _sd = json.load(_fh)
            self._last_step_color: str = _sd.get("last_step_color", DEFAULT_STEP_COLOR)
        except Exception:
            self._last_step_color: str = DEFAULT_STEP_COLOR
        self._on_top: bool = _on_top
        self.completed_paths: set = set()

        self.setWindowTitle(f"Deep Space Astro - Workflow Companion  v{VERSION}")
        if _on_top:
            self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        else:
            self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowStaysOnTopHint)
        _icon_pm = QPixmap()
        _icon_pm.loadFromData(base64.b64decode(ICON_B64))
        _icon = QIcon(_icon_pm)
        self.setWindowIcon(_icon)
        QApplication.instance().setWindowIcon(_icon)

        self._build_ui()
        QTimer.singleShot(0, self._sync_footer)

        self._right_panel._workflow_combo.activated.connect(self._load_workflow)
        self._right_panel._wf_export_btn.clicked.connect(self._export_workflow)
        self._right_panel._wf_import_btn.clicked.connect(self._import_workflow)
        self._right_panel._wf_delete_btn.clicked.connect(self._delete_workflow)
        self._rebuild_workflow_combo()
        if self._active_workflow:
            combo = self._right_panel._workflow_combo
            idx = combo.findText(self._active_workflow)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            # Restore completed state and description from the active workflow
            wf = self.workflows.get(self._active_workflow, {})
            _desc = wf.get("description", "")
            self._right_panel._workflow_desc_lbl.setText(linkify(_desc))
            self._right_panel._workflow_desc_lbl.setVisible(bool(_desc))
            self.completed_paths = set(wf.get("completed", []))
            self._right_panel.completed_paths = self.completed_paths
            # Restore per-workflow notes, step colors, and group state
            self.notes = dict(wf.get("notes", {}))
            self._right_panel.notes = self.notes
            self.step_colors = dict(wf.get("step_colors", {}))
            self._right_panel.step_colors = self.step_colors
            self._right_panel.last_step_color = DEFAULT_STEP_COLOR
            self._right_panel.group_selections = dict(wf.get("group_selections", {}))
            self._right_panel.group_collapsed = dict(wf.get("group_collapsed", {}))
            # Force-expand any group that contains a missing item
            for _p in self.checked_paths:
                if _p.startswith(GROUPITEM_PREFIX):
                    _gi = parse_groupitem(_p)
                    if _gi and _gi[1] in self.missing_paths:
                        self._right_panel.group_collapsed[_gi[0]] = False
            self._right_panel._workflow_name = self._active_workflow
            self._right_panel.rebuild()
        else:
            self._right_panel._workflow_name = self._active_workflow

        # Warn about any missing scripts (missing_paths already fully populated above)
        _missing = sorted({os.path.basename(p) for p in self.missing_paths})
        if _missing:
            _ctx = (f"From workflow '{self._active_workflow}':\n"
                    if self._active_workflow else "")
            QTimer.singleShot(
                100,
                lambda m=_missing, c=_ctx: self._warn_missing_scripts(m, context=c)
            )

        # What's New dialog - show once per version upgrade
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as _fh:
                _sd = json.load(_fh)
            _last_seen = _sd.get("last_seen_version", "")
        except (FileNotFoundError, json.JSONDecodeError):
            _last_seen = ""

        def _ver_tuple(v: str):
            try:
                return tuple(int(x) for x in v.split("."))
            except Exception:
                return (0, 0, 0)

        _current_tuple = _ver_tuple(VERSION)
        _last_tuple    = _ver_tuple(_last_seen)

        if _last_seen and _current_tuple > _last_tuple:
            # Existing user upgrading - show only entries newer than last_seen
            _new_entries = {
                v: items
                for v, items in CHANGELOG.items()
                if _ver_tuple(v) > _last_tuple
            }
            if _new_entries:
                QTimer.singleShot(200, lambda e=_new_entries: self._show_whats_new(e))
        elif not _last_seen:
            # First run with this feature - show current version's entries
            _new_entries = {VERSION: CHANGELOG.get(VERSION, [])}
            if _new_entries.get(VERSION):
                QTimer.singleShot(200, lambda e=_new_entries: self._show_whats_new(e))
        else:
            # Already up to date - just ensure last_seen_version is recorded
            self._write_last_seen_version()

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

        # ── top bar (On Top only) ─────────────────────────────────────────
        top = QHBoxLayout()
        top.addStretch()
        self._ontop_chk = QCheckBox("On Top")
        self._ontop_chk.setChecked(self._on_top)
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
        self._left_panel._search_input.textChanged.connect(self._apply_filter)
        self._search_input = self._left_panel._search_input  # alias for compatibility

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
            step_colors=self.step_colors,
            on_color_changed=self._on_color_changed,
            last_step_color=self._last_step_color,
            on_group_renamed=self._on_group_renamed,
            on_group_collapsed=self._on_right_panel_group_collapsed,
        )
        self._splitter.addWidget(self._right_panel)

        # ── footer ────────────────────────────────────────────────────────────
        self._status_label = QLabel("")
        root.addWidget(self._status_label)

        # ── footer: two sub-rows mirroring the left/right split ─────────────
        footer = QHBoxLayout()
        footer.setSpacing(4)
        footer.setContentsMargins(0, 0, 0, 0)

        # Toggle button lives directly in the footer so it is never hidden
        self._toggle_btn = QPushButton("Hide Panel")
        self._toggle_btn.setToolTip("Hide the left panel")
        self._toggle_btn.clicked.connect(self._toggle_left)
        footer.addWidget(self._toggle_btn)

        # Collapsible left sub-widget: Refresh (hidden along with the left panel)
        left_btns = QHBoxLayout()
        left_btns.setSpacing(4)
        left_btns.setContentsMargins(0, 0, 0, 0)
        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.setToolTip(
            "Rescan for new or removed scripts and functions.")
        self._refresh_btn.clicked.connect(self._refresh)
        left_btns.addWidget(self._refresh_btn)
        left_btns.addStretch()

        # Right sub-row: New | Add Group | Add Step | Save ... Help | Close
        right_btns = QHBoxLayout()
        right_btns.setSpacing(4)
        self._clear_btn = QPushButton("New")
        self._clear_btn.setToolTip("Clear the right panel to start a new workflow")
        self._clear_btn.clicked.connect(self._clear_all)
        right_btns.addWidget(self._clear_btn)
        self._add_group_btn = QPushButton("Add Group")
        self._add_group_btn.setToolTip("Add a group to hold multiple alternative scripts or functions")
        self._add_group_btn.clicked.connect(self._add_group)
        right_btns.addWidget(self._add_group_btn)
        self._add_step_btn = QPushButton("Add Step")
        self._add_step_btn.setToolTip("Add a manual reminder step to the right panel")
        self._add_step_btn.clicked.connect(self._add_manual_step)
        right_btns.addWidget(self._add_step_btn)
        self._save_workflow_btn = QPushButton("Save")
        self._save_workflow_btn.setToolTip("Save current scripts and favorites as a named workflow")
        self._save_workflow_btn.clicked.connect(self._save_workflow)
        right_btns.addWidget(self._save_workflow_btn)
        self._clear_completed_btn = QPushButton("Clear Completed")
        self._clear_completed_btn.setToolTip("Unmark all completed items")
        self._clear_completed_btn.clicked.connect(self._clear_completed)
        right_btns.addWidget(self._clear_completed_btn)
        right_btns.addStretch()
        self._get_workflows_btn = QPushButton("Get Workflows")
        self._get_workflows_btn.setToolTip("Browse and download workflows from Deep Space Astro")
        self._get_workflows_btn.clicked.connect(self._open_get_workflows)
        right_btns.addWidget(self._get_workflows_btn)
        help_btn = QPushButton("Help")
        help_btn.setToolTip("Show usage instructions")
        help_btn.clicked.connect(self._show_help)
        right_btns.addWidget(help_btn)
        close_btn = QPushButton("Close")
        close_btn.setToolTip("Close the Workflow Companion")
        close_btn.clicked.connect(self.close)
        right_btns.addWidget(close_btn)

        # Wrap collapsible left sub-row and right sub-row in widgets
        self._left_btn_widget = QWidget()
        self._left_btn_widget.setLayout(left_btns)
        right_btn_widget = QWidget()
        right_btn_widget.setLayout(right_btns)

        footer.addWidget(self._left_btn_widget)
        footer.addWidget(right_btn_widget)
        root.addLayout(footer)

        # Keep the left/right footer widgets in sync with the splitter widths.
        # _left_btn_widget is hidden when the panel is hidden; toggle_btn stays visible.
        def _sync_footer_widths():
            if self._left_panel.isVisible():
                lw = self._left_panel.width()
                toggle_w = self._toggle_btn.width()
                target = lw - toggle_w - footer.spacing()
                if target > 0:
                    self._left_btn_widget.setFixedWidth(target)
                else:
                    self._left_btn_widget.setMaximumWidth(_MAX_WIDGET_SIZE)
            else:
                self._left_btn_widget.setMaximumWidth(_MAX_WIDGET_SIZE)
                self._left_btn_widget.setMinimumWidth(0)
        self._splitter.splitterMoved.connect(lambda pos, idx: _sync_footer_widths())
        self._sync_footer = _sync_footer_widths

        self._update_status()

    # ── queue callbacks ───────────────────────────────────────────────────────

    def _rebuild_panels(self) -> None:
        """Rebuild both panels - call after any change that affects both."""
        self._right_panel.rebuild()
        self._left_panel.rebuild()

    def _on_queue_changed(self):
        """Called after any add/reorder in the right panel."""
        self._workflow_dirty = True
        self._left_panel.refresh_selected()  # fast dim update, no full rebuild
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
        base_path = f"{STEP_PREFIX}{label}"
        # If a step with this name already exists, give this one a unique suffix
        # so each duplicate can be moved/deleted independently
        if base_path in self.checked_paths:
            path = f"{base_path}#{uuid.uuid4().hex[:6]}"
        else:
            path = base_path
        # New steps default to last used color
        self.step_colors[path] = self._last_step_color
        self._right_panel.step_colors = self.step_colors
        self.checked_paths.append(path)
        self._workflow_dirty = True
        self._right_panel.rebuild(scroll_to_bottom=True)
        self._update_status()

    def _add_group(self):
        """Prompt for a name and add a new group to the right panel."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Add Group")
        dlg.setMinimumWidth(300)
        dlg.setWindowFlags(dlg.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        layout = QVBoxLayout(dlg)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 8)
        layout.addWidget(QLabel("Group name (e.g. Noise Reduction):"))
        name_edit = QLineEdit()
        name_edit.setPlaceholderText("Enter group name…")
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
        gid = make_group_id()
        path = f"{GROUP_PREFIX}{label}#{gid}"
        self.checked_paths.append(path)
        self._workflow_dirty = True
        self._right_panel.rebuild(scroll_to_bottom=True)
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
        # Move step color if any
        if old_path in self.step_colors:
            self.step_colors[new_path] = self.step_colors.pop(old_path)
            self._right_panel.step_colors = self.step_colors
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
        reply = QMessageBox.question(
            self, "Clear Completed",
            "Unmark all completed items?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
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

    def _on_right_panel_group_collapsed(self, group_id: str, collapsed: bool):
        """Re-evaluate scroll policy and window height after a group expand/collapse
        when the left panel is hidden."""
        if not self._left_panel.isVisible():
            QTimer.singleShot(0, self._shrink_to_right_panel)

    def _on_group_renamed(self, old_path: str, new_name: str):
        """Rename a group by updating its path in checked_paths and related dicts."""
        parsed = parse_group_header(old_path)
        if not parsed:
            return
        _, gid = parsed
        new_path = f"{GROUP_PREFIX}{new_name}#{gid}"
        # Update checked_paths
        for i, p in enumerate(self.checked_paths):
            if p == old_path:
                self.checked_paths[i] = new_path
        # Update favorites, notes, completed
        if old_path in self.favorites:
            self.favorites.discard(old_path)
            self.favorites.add(new_path)
        if old_path in self.notes:
            self.notes[new_path] = self.notes.pop(old_path)
            self._right_panel.notes = self.notes
        if old_path in self.completed_paths:
            self.completed_paths.discard(old_path)
            self.completed_paths.add(new_path)
        self._workflow_dirty = True
        self._right_panel.rebuild()
        self._update_status()

    def _on_color_changed(self, path: str, color: str):
        """Save updated step dot color."""
        self.step_colors[path] = color
        self._last_step_color = color
        self._workflow_dirty = True
        # Persist the updated step_colors into the active workflow so each step
        # keeps its own color after close/reopen without requiring an explicit Save.
        # Without this, only last_step_color was written and every step fell back
        # to that single color on next load.
        if self._active_workflow and self._active_workflow in self.workflows:
            self.workflows[self._active_workflow]["step_colors"] = dict(self.step_colors)
        save_state(self.checked_paths, self.favorites, scripts_dir=self.scripts_dir,
                   workflows=self.workflows, last_step_color=color)

    def _add_script(self, path: str):
        """Called on double-click in the left panel."""
        self.checked_paths.append(path)
        self._workflow_dirty = True
        self._rebuild_panels()
        self._update_status()

    def _remove_script(self, path: str):
        """Called when a script is dragged from right back to left, or removed via trash."""
        # Use index-based removal so only the first occurrence is removed when
        # duplicate step names share the same path string
        try:
            self.checked_paths.pop(self.checked_paths.index(path))
        except ValueError:
            pass
        # If removing a group, also remove all its groupitems
        if path.startswith(GROUP_PREFIX):
            parsed = parse_group_header(path)
            if parsed:
                gid = parsed[1]
                self.checked_paths[:] = [p for p in self.checked_paths
                                         if not (p.startswith(GROUPITEM_PREFIX)
                                         and parse_groupitem(p) is not None
                                         and parse_groupitem(p)[0] == gid)]
                self._right_panel.group_selections.pop(gid, None)
                self._right_panel.group_collapsed.pop(gid, None)
        self.favorites.discard(path)
        self._workflow_dirty = True
        self._rebuild_panels()
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
        combo.addItem(" -  Load Workflow  - ")
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

        layout.addWidget(QLabel("Description (optional):"))
        desc_edit = QLineEdit()
        desc_edit.setPlaceholderText("Enter a description…")
        if self._active_workflow and self._active_workflow in self.workflows:
            desc_edit.setText(self.workflows[self._active_workflow].get("description", ""))
        layout.addWidget(desc_edit)

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
        description = desc_edit.text().strip()

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
            "selections":      list(self.checked_paths),
            "favorites":       sorted(self.favorites),
            "completed":       sorted(self.completed_paths),
            "description":     description,
            "notes":           dict(self.notes),
            "step_colors":     dict(self.step_colors),
            "group_selections": dict(self._right_panel.group_selections),
            "group_collapsed":  dict(self._right_panel.group_collapsed),
        }
        self._active_workflow = name
        self._workflow_dirty = False
        # Update description label immediately
        self._right_panel._workflow_desc_lbl.setText(linkify(description))
        self._right_panel._workflow_desc_lbl.setVisible(bool(description))
        self._right_panel._workflow_name = name
        self._right_panel.rebuild()
        save_state(self.checked_paths, self.favorites, scripts_dir=self.scripts_dir, workflows=self.workflows,
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
            "They may have been deprecated or renamed. Check Get Scripts."
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

        # Guard: if the current workflow has unsaved changes, ask before switching
        if self._workflow_dirty:
            wf = self._active_workflow
            msg = (
                f"Workflow '{wf}' has unsaved changes.\nSwitch to '{name}' without saving?"
                if wf else
                f"You have unsaved changes.\nSwitch to '{name}' without saving?"
            )
            reply = QMessageBox.question(
                self, "Unsaved Changes", msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                # Revert the combo back to the previously active workflow
                combo = self._right_panel._workflow_combo
                combo.blockSignals(True)
                if self._active_workflow:
                    prev_idx = combo.findText(self._active_workflow)
                    combo.setCurrentIndex(prev_idx if prev_idx >= 0 else 0)
                else:
                    combo.setCurrentIndex(0)
                combo.blockSignals(False)
                return
        workflow = self.workflows.get(name)
        if not workflow:
            return
        valid_paths = {p for _, p in self.scripts}
        raw_paths = workflow.get("selections", [])
        missing_paths = [p for p in raw_paths
                         if p not in valid_paths and not p.startswith(NON_SCRIPT_PREFIXES)]
        # Also check scripts inside groups (groupitem:// paths)
        for _p in raw_paths:
            if _p.startswith(GROUPITEM_PREFIX):
                _gi = parse_groupitem(_p)
                if _gi:
                    _item_path = _gi[1]
                    if (_item_path not in valid_paths
                            and not _item_path.startswith(NON_SCRIPT_PREFIXES)
                            and _item_path not in missing_paths):
                        missing_paths.append(_item_path)
        # Inform user of any missing scripts (load continues regardless)
        if missing_paths:
            missing_names = [os.path.basename(p) for p in missing_paths]
            self._warn_missing_scripts(
                missing_names,
                context=f"From workflow '{name}':\n"
            )
        raw_favs = workflow.get("favorites", [])
        loaded_favs = {p for p in raw_favs if p in valid_paths or p.startswith(STEP_PREFIX)
                       or p.startswith(FUNCTION_PREFIX) or p.startswith(GROUP_PREFIX)}
        self.missing_paths = {p for p in raw_paths
                              if p not in valid_paths and not p.startswith(NON_SCRIPT_PREFIXES)}
        # Also capture missing scripts inside groups
        for _p in raw_paths:
            if _p.startswith(GROUPITEM_PREFIX):
                _gi = parse_groupitem(_p)
                if _gi:
                    _item_path = _gi[1]
                    if (_item_path not in valid_paths
                            and not _item_path.startswith(NON_SCRIPT_PREFIXES)):
                        self.missing_paths.add(_item_path)
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
        # Load per-workflow notes, step colors, and group state
        self.notes = dict(workflow.get("notes", {}))
        self._right_panel.notes = self.notes
        self.step_colors = dict(workflow.get("step_colors", {}))
        self._right_panel.step_colors = self.step_colors
        self._right_panel.last_step_color = DEFAULT_STEP_COLOR
        self._right_panel.group_selections = dict(workflow.get("group_selections", {}))
        self._right_panel.group_collapsed = dict(workflow.get("group_collapsed", {}))
        # Force-expand any group that contains a missing item
        for _p in self.checked_paths:
            if _p.startswith(GROUPITEM_PREFIX):
                _gi = parse_groupitem(_p)
                if _gi and _gi[1] in self.missing_paths:
                    self._right_panel.group_collapsed[_gi[0]] = False
        save_state(self.checked_paths, self.favorites, scripts_dir=self.scripts_dir, active_workflow=name)
        desc = workflow.get("description", "")
        self._right_panel._workflow_desc_lbl.setText(linkify(desc))
        self._right_panel._workflow_desc_lbl.setVisible(bool(desc))
        self._right_panel._workflow_name = name
        self._rebuild_panels()
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
            self, f"Export Workflow - {name}", f"{name}.json",
            "JSON Workflow Files (*.json);;All Files (*)"
        )
        if not path:
            return
        if not path.endswith(".json"):
            path += ".json"
        try:
            scripts_root = os.path.normpath(self.scripts_dir)

            wf = self.workflows[name]
            rel_selections = [path_to_relative(p, scripts_root) for p in wf["selections"]]
            rel_favorites  = [path_to_relative(p, scripts_root) for p in wf["favorites"]]

            # Relativize group_selections values (selected item path per group)
            rel_group_selections = {
                gid: path_to_relative(item, scripts_root)
                for gid, item in wf.get("group_selections", {}).items()
            }

            # Relativize notes keys (which can be absolute script paths)
            rel_notes = {
                path_to_relative(k, scripts_root): v
                for k, v in wf.get("notes", {}).items()
            }

            with open(path, "w", encoding="utf-8") as fh:
                json.dump({
                    "workflow_name":    name,
                    "description":      wf.get("description", ""),
                    "selections":       rel_selections,
                    "favorites":        rel_favorites,
                    "step_colors":      wf.get("step_colors", {}),
                    "group_selections": rel_group_selections,
                    "group_collapsed":  wf.get("group_collapsed", {}),
                    "notes":            rel_notes,
                    "portable":         True,
                    # "completed" intentionally omitted - per-session state
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
                if (p.startswith(STEP_PREFIX) or p.startswith(FUNCTION_PREFIX)
                        or p.startswith(GROUP_PREFIX) or p.startswith(SYSTEM_PREFIX)):
                    return p
                if p.startswith(GROUPITEM_PREFIX):
                    parsed = parse_groupitem(p)
                    if parsed:
                        gid, item_path = parsed
                        return make_groupitem_path(gid, to_absolute(item_path))
                    return p
                # If portable, treat as relative to local siril-scripts root
                if portable or not os.path.isabs(p):
                    # Normalise separators then join
                    rel = p.replace("/", os.sep).replace("\\", os.sep)
                    return os.path.join(scripts_root, rel)
                return p  # legacy absolute path - use as-is

            valid_paths = {p for _, p in self.scripts}
            raw_paths   = [to_absolute(p) for p in data["selections"]]
            raw_favs    = [to_absolute(p) for p in data.get("favorites", [])]

            loaded      = [p for p in raw_paths if p in valid_paths or p.startswith(NON_SCRIPT_PREFIXES)]
            missing_paths = [p for p in raw_paths
                             if p not in valid_paths and not p.startswith(NON_SCRIPT_PREFIXES)]
            loaded_favs = [p for p in raw_favs if p in valid_paths or p.startswith(STEP_PREFIX)
                           or p.startswith(FUNCTION_PREFIX) or p.startswith(GROUP_PREFIX)]

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

            # Resolve group_selections values back to absolute paths
            abs_group_selections = {
                gid: to_absolute(item)
                for gid, item in data.get("group_selections", {}).items()
            }

            # Resolve notes keys back to absolute paths
            abs_notes = {
                to_absolute(k): v
                for k, v in data.get("notes", {}).items()
            }

            self.workflows[name] = {
                "selections":       loaded,
                "favorites":        sorted(loaded_favs),
                "description":      data.get("description", ""),
                "step_colors":      data.get("step_colors", {}),
                "group_selections": abs_group_selections,
                "group_collapsed":  data.get("group_collapsed", {}),
                "notes":            abs_notes,
            }
            # Load it immediately into the right panel
            self.missing_paths = {p for p in raw_paths
                                  if p not in valid_paths and not p.startswith(STEP_PREFIX)
                                   and not p.startswith(FUNCTION_PREFIX)}
            self._right_panel.missing_paths = self.missing_paths
            self.step_colors = dict(data.get("step_colors", {}))
            self._right_panel.step_colors = self.step_colors
            self._right_panel.last_step_color = DEFAULT_STEP_COLOR
            self.notes = dict(data.get("notes", {}))
            self._right_panel.notes = self.notes
            self.checked_paths.clear()
            self.checked_paths.extend(raw_paths)  # keep all including missing
            self.favorites.clear()
            self.favorites.update(loaded_favs)
            self._active_workflow = name
            self._workflow_dirty = False
            save_state(self.checked_paths, self.favorites, scripts_dir=self.scripts_dir,
                       workflows=self.workflows, active_workflow=name)
            self._rebuild_workflow_combo()
            combo = self._right_panel._workflow_combo
            idx = combo.findText(name)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            self._right_panel._workflow_name = name
            self._rebuild_panels()
            self._update_status()
            _idesc = data.get("description", "")
            self._right_panel._workflow_desc_lbl.setText(_idesc)
            self._right_panel._workflow_desc_lbl.setVisible(bool(_idesc))
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
        save_state(self.checked_paths, self.favorites, scripts_dir=self.scripts_dir,
                   workflows=self.workflows, active_workflow=self._active_workflow)
        self._rebuild_workflow_combo()
        self._right_panel._workflow_name = self._active_workflow
        self._rebuild_panels()
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
        self.notes.clear()
        self._right_panel.notes = self.notes
        self.step_colors.clear()
        self._right_panel.step_colors = self.step_colors
        self._right_panel.group_selections.clear()
        self._right_panel.group_collapsed.clear()
        self._active_workflow = ""
        self._workflow_dirty = False
        self._right_panel._workflow_desc_lbl.setText("")
        self._right_panel._workflow_desc_lbl.setVisible(False)
        save_state(self.checked_paths, self.favorites, scripts_dir=self.scripts_dir, active_workflow="")
        self._rebuild_workflow_combo()
        self._right_panel._workflow_name = ""
        self._rebuild_panels()
        self._update_status()

    # ── filter / refresh ──────────────────────────────────────────────────────

    def _visible_scripts(self) -> list[tuple[str, str]]:
        query = self._search_input.text().lower()
        if not query:
            return self.scripts
        result = []
        for lbl, p in self.scripts:
            # Match against script name
            if query in lbl.lower():
                result.append((lbl, p))
                continue
            # Match against group name (e.g. "Scripts - Core")
            if lbl.startswith("["):
                inner = lbl[1:lbl.index("]")]
                grp_name = inner.split("|")[0] if "|" in inner else inner
                sub_name = "Python Scripts" if inner.endswith("|py") else "Siril Scripts" if inner.endswith("|ssf") else ""
                full = f"Scripts - {grp_name}"
                if query in full.lower() or (sub_name and query in sub_name.lower()):
                    result.append((lbl, p))
        return result

    def _apply_filter(self):
        query = self._search_input.text().lower()
        visible = self._visible_scripts()
        self._left_panel.rebuild(scripts=visible, query=query)
        self._update_status()

    def _refresh(self):
        global _DIALOG_GROUPS, _SIRILPY_HAS_DIALOGS
        self.scripts = scan_scripts(self.scripts_dir)
        system_scripts_dir = find_siril_system_scripts_dir()
        if system_scripts_dir:
            self.scripts += scan_system_scripts(system_scripts_dir)

        # Re-scan functions (Siril built-in dialogs)
        _DIALOG_GROUPS = _build_dialog_groups()

        valid_paths = {p for _, p in self.scripts}
        # Build the set of valid function paths from the refreshed _DIALOG_GROUPS
        if _DIALOG_GROUPS:
            valid_paths |= {f"{FUNCTION_PREFIX}{d.name}"
                            for dialogs in _DIALOG_GROUPS.values() for d in dialogs}
        # Update missing_paths - don't remove anything from checked_paths
        self.missing_paths = {p for p in self.checked_paths
                              if p not in valid_paths and not p.startswith(STEP_PREFIX)
                                   and not p.startswith(FUNCTION_PREFIX)}
        self._right_panel.missing_paths = self.missing_paths
        self._search_input.clear()
        self._rebuild_panels()
        self._update_status()

    # ── status ────────────────────────────────────────────────────────────────

    def _update_status(self):
        # Count functions from _DIALOG_GROUPS (not stored in self.scripts)
        func_total = sum(len(v) for v in _DIALOG_GROUPS.values()) if _DIALOG_GROUPS else 0
        total = len(self.scripts) + func_total

        # Visible count: filtered scripts + filtered functions
        query = self._search_input.text().lower()
        visible = self._visible_scripts()          # compute once and reuse
        visible_script_count = len(visible)
        if query and _DIALOG_GROUPS:
            visible_funcs = sum(
                1 for dialogs in _DIALOG_GROUPS.values()
                for d in dialogs
                if query in d.label.lower() or query in d.dialog_type.lower()
            )
        else:
            visible_funcs = func_total
        visible_count = visible_script_count + visible_funcs

        # Queued count: exclude group headers and manual steps;
        # count groupitems (items inside groups) as individual selections
        queued = sum(
            1 for p in self.checked_paths
            if not p.startswith(GROUP_PREFIX) and not p.startswith(STEP_PREFIX)
        )

        if visible_count < total:
            self._status_label.setText(
                f"Showing {visible_count} of {total} items  •  {queued} selected"
            )
        else:
            self._status_label.setText(
                f"{total} items  •  {queued} selected"
            )

    # ── run ───────────────────────────────────────────────────────────────────

    def _run_one(self, script_path: str, name: str):
        # Resolve system:// paths to absolute
        if script_path.startswith(SYSTEM_PREFIX):
            filename = script_path[len(SYSTEM_PREFIX):]
            sys_dir = find_siril_system_scripts_dir()
            if sys_dir:
                script_path = os.path.normpath(os.path.join(sys_dir, filename))
            else:
                QMessageBox.warning(self, "System Scripts Not Found",
                    f"Could not locate Siril's system scripts directory to run '{name}'.")
                return
        # Handle Siril built-in dialog
        if script_path.startswith(FUNCTION_PREFIX):
            try:
                dialog_name = script_path[len(FUNCTION_PREFIX):]
                from sirilpy import DialogID
                from sirilpy.enums import DialogReq
                dialog = DialogID.from_name(dialog_name)
                if dialog is None:
                    QMessageBox.warning(self, "Unknown Dialog",
                        f"The dialog '{dialog_name}' is no longer available in this version of Siril.")
                    return
                # Check requirements before opening
                req = dialog.req
                _req_desc = {
                    DialogReq.ANY:         "an image or sequence to be loaded",
                    DialogReq.IMG:         "an image to be loaded",
                    DialogReq.RGB:         "an RGB image to be loaded",
                    DialogReq.MONO:        "a mono image to be loaded",
                    DialogReq.SEQ:         "a sequence to be loaded",
                    DialogReq.PLTSOLVD:    "a plate-solved image or sequence to be loaded",
                    DialogReq.RGBPLTSOLVD: "a plate-solved RGB image to be loaded",
                    DialogReq.NONE:        "",
                }
                self._siril.connect()
                try:
                    if req is not DialogReq.NONE:
                        try:
                            img_loaded = self._siril.is_image_loaded()
                            seq_loaded = self._siril.is_sequence_loaded()
                            any_loaded = img_loaded or seq_loaded
                            rgb = False
                            mono = False
                            pltsolvd = False
                            if any_loaded:
                                try:
                                    img_meta = self._siril.get_image(with_pixels=False)
                                    # naxes = (width, height, channels)
                                    nchans = img_meta.naxes[2] if hasattr(img_meta, "naxes") else img_meta.shape[2]
                                    rgb = nchans == 3
                                    mono = nchans == 1
                                    pltsolvd = bool(img_meta.keywords.pltsolvd)
                                except Exception:
                                    pass
                            _req_met = {
                                DialogReq.ANY:         any_loaded,
                                DialogReq.IMG:         img_loaded,
                                DialogReq.RGB:         img_loaded and rgb,
                                DialogReq.MONO:        img_loaded and mono,
                                DialogReq.SEQ:         seq_loaded,
                                DialogReq.PLTSOLVD:    any_loaded and pltsolvd,
                                DialogReq.RGBPLTSOLVD: img_loaded and rgb and pltsolvd,
                            }
                            if not _req_met.get(req, True):
                                QMessageBox.warning(self, dialog.label,
                                    f"'{dialog.label}' requires {_req_desc[req]}.")
                                return
                        except Exception:
                            pass  # If check fails, attempt to open anyway
                    success = self._siril.open_dialog(dialog)
                except Exception:
                    success = None
                finally:
                    self._siril.disconnect()
                if success is False:
                    hint = _req_desc.get(dialog.req, "")
                    msg = f"'{dialog.label}' could not be opened."
                    if hint:
                        msg += f"\n\nMake sure {hint}."
                    QMessageBox.warning(self, dialog.label, msg)
            except Exception as e:
                QMessageBox.critical(self, "Dialog Error", str(e))
            return
        # Handle .ssf legacy scripts - runs synchronously via siril cmd "@"
        if script_path.endswith(".ssf"):
            self._status_label.setText(f"Running {name}...")
            QApplication.processEvents()
            try:
                # Normalise to forward slashes - Siril's @ command requires them
                ssf_path = script_path.replace("\\", "/")
                self._siril.connect()
                try:
                    self._siril.cmd(f"@{ssf_path}")
                finally:
                    self._siril.disconnect()
                self._status_label.setText(f"{name} complete.")
                QTimer.singleShot(4000, self._update_status)
            except Exception as e:
                QMessageBox.critical(self, "Script Error",
                                     f"Could not run '{name}':\n\n{str(e)}")
                self._status_label.setText(f"Failed to run {name}.")
            return
        self._status_label.setText(f"Launching {name}…")
        QApplication.processEvents()

        try:
            # Ask Siril to launch the script natively via the pyscript -async
            # command (available since Siril 1.4.0). This gives the script its
            # own independent pipe connection, exactly as if launched from
            # Siril's Scripts menu, and returns immediately so the UI stays
            # responsive and multiple scripts can be launched simultaneously.
            self._siril.connect()
            try:
                self._siril.cmd(f'pyscript -async "{script_path}"')
            finally:
                self._siril.disconnect()
            self._status_label.setText(f"{name} launched.")
            QTimer.singleShot(4000, self._update_status)
        except Exception as e:
            QMessageBox.critical(self, "Launch Error",
                                 f"Could not launch '{name}':\n\n{str(e)}")
            self._status_label.setText(f"Failed to launch {name}.")

    def _open_get_workflows(self):
        webbrowser.open("https://buymeacoffee.com/deepspaceastro/extras/281066")

    def _show_help(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Deep Space Astro - Workflow Companion - Help")
        dlg.setMinimumWidth(700)
        dlg.setMinimumHeight(600)
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
            "<p><b>&#x2139;&#xFE0F;&nbsp; Siril 1.4.3 or later is required to use Functions. In older versions, Functions will not appear in the left panel and will generate an error if run from the right panel.</b></p>"
            "<h3>BROWSING SCRIPTS &amp; FUNCTIONS</h3>"
            "<p>&#8226; Scripts are grouped by subfolder; Siril built-in functions appear above them.</p>"
            "<p>&#8226; Click a group header to expand or collapse it.</p>"
            "<p>&#8226; Use Collapse All / Expand All to manage all categories at once.</p>"
            "<p>&#8226; Use the Filter Items box to search by name or category.</p>"
            "<p>&#8226; Items already in the right panel are shown dimmed in the left panel with a tooltip. Dimmed items can be added multiple times.</p>"
            "<p>&#8226; Hover over a script to see its description.</p>"
            "<h3>ADDING SCRIPTS &amp; FUNCTIONS</h3>"
            "<p>&#8226; Drag any item from the left panel to the right panel to add it, or double-click an item to add it instantly.</p>"
            "<p>&#8226; Siril built-in functions open the corresponding Siril dialog when launched. A warning message is displayed if pre-reqs aren't met for some functions.</p>"
            "<h3>REMOVING SCRIPTS &amp; FUNCTIONS</h3>"
            "<p>&#8226; Drag an item from the right panel back to the left panel to remove it.</p>"
            "<p>&#8226; Or click the &#128465; trash button on any row and confirm the prompt.</p>"
            "<h3>REORDERING &amp; DUPLICATING</h3>"
            "<p>&#8226; Drag items up or down within the right panel to change their order.</p>"
            "<p>&#8226; Items in groups can also be dragged to change their order.</p>"
            "<p>&#8226; Right-click an item in the right panel and choose Copy to duplicate it.</p>"
            "<h3>MANUAL STEPS</h3>"
            "<p>&#8226; Click Add Step to add a reminder step.</p>"
            "<p>&#8226; Manual steps show a colored circle instead of a play button. Color can be changed in the Notes dialog.</p>"
            "<p>&#8226; Double-click a step&#39;s label to rename it, or use the pencil button to rename it and add notes together.</p>"
            "<p>&#8226; Steps can be reordered, starred, completed, and removed.</p>"
            "<h3>GROUPS</h3>"
            "<p>&#8226; Click Add Group to create a group.</p>"
            "<p>&#8226; Drag items into the group and select which one runs when the Play button is clicked.</p>"
            "<p>&#8226; When the group is marked complete, all items within the group are disabled.</p>"
            "<h3>NOTES</h3>"
            "<p>&#8226; Click the pencil button on any row to add or edit a note.</p>"
            "<p>&#8226; The pencil turns blue when a note exists.</p>"
            "<p>&#8226; The Notes button border is gold when a note exists in a group item.</p>"
            "<p>&#8226; Notes are saved with workflows.</p>"
            "<h3>FAVORITES</h3>"
            "<p>&#8226; Click the &#9734; star button on any row to mark it as a favorite.</p>"
            "<p>&#8226; Favorites are pinned to the top of the right panel.</p>"
            "<p>&#8226; Favorites are saved with workflows and exports.</p>"
            "<h3>EXECUTING ITEMS &amp; FUNCTIONS</h3>"
            "<p>&#8226; Click the &#9654; play button on any row to launch that script or function.</p>"
            "<p>&#8226; The status bar shows when a script has been launched.</p>"
            "<p>&#8226; Scripts not found on disk are shown disabled with a tooltip.</p>"
            "<h3>COMPLETING ITEMS</h3>"
            "<p>&#8226; Click the &#10003; check button on any row to mark it as done.</p>"
            "<p>&#8226; Completed items show a strikethrough label and a green check button.</p>"
            "<p>&#8226; Click Clear Completed to reset all checkmarks.</p>"
            "<p>&#8226; Completed state is saved with workflows but not included in exports.</p>"
            "<h3>WORKFLOWS</h3>"
            "<p>&#8226; Click Save to name and save your scripts, functions, steps, notes, favorites, and completed state.</p>"
            "<p>&#8226; Click New to clear the right panel and start fresh.</p>"
            "<p>&#8226; Use the Workflows dropdown in the right panel to load a saved workflow.</p>"
            "<p>&#8226; Use the Workflow buttons to Export, Import, or Delete a workflow.</p>"
            "<p>&#8226; Exporting saves a workflow to a .json file you can share with others.</p>"
            "<p>&#8226; Importing loads a workflow from a file and makes it the active workflow.</p>"
            "<p>&#8226; Click Get Workflows to download additional workflows.</p>"
            "<h3>OTHER</h3>"
            "<p>&#8226; Click Hide Panel / Show Panel to toggle the left panel.</p>"
            "<p>&#8226; Click Refresh to rescan for new or removed scripts and functions without clearing the right panel.</p>"
            "<p>&#8226; Check On Top to keep the launcher above other windows.</p>"
            "<p>&#8226; Window size and position are saved automatically between sessions.</p>"
        )
        layout.addWidget(text)

        btn_row = QHBoxLayout()
        release_btn = QPushButton("Release Notes")
        release_btn.setToolTip("Show version history and release notes")
        release_btn.clicked.connect(self._show_release_notes)
        btn_row.addWidget(release_btn)
        tutorial_btn = QPushButton("Tutorial")
        tutorial_btn.setToolTip("Watch the tutorial video on YouTube")
        tutorial_btn.clicked.connect(lambda: webbrowser.open(
            "https://www.youtube.com/playlist?list=PLJeHInlBzFrYwZ4Dn1uM_azL9bTUsIgZZ"))
        btn_row.addWidget(tutorial_btn)
        siril_manual_btn = QPushButton("Siril Manual")
        siril_manual_btn.setToolTip("Open the Siril documentation")
        siril_manual_btn.clicked.connect(lambda: webbrowser.open(
            "https://siril.readthedocs.io/en/stable/"))
        btn_row.addWidget(siril_manual_btn)
        support_dsa_btn = QPushButton("Support DSA")
        support_dsa_btn.setToolTip("Support my efforts by becoming a member, or buying me a coffee!")
        support_dsa_btn.setStyleSheet(
            "QPushButton { background-color: #39ff14; color: #000000; font-weight: bold; border: none; border-radius: 4px; padding: 4px 12px; }"
            "QPushButton:hover { background-color: #2bcc0f; }"
            "QPushButton:pressed { background-color: #57ff3a; }"
        )
        support_dsa_btn.clicked.connect(lambda: webbrowser.open(
            "https://buymeacoffee.com/deepspaceastro"))
        btn_row.addWidget(support_dsa_btn)
        btn_row.addStretch()
        ok_btn = QPushButton("OK")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(dlg.accept)
        btn_row.addWidget(ok_btn)
        layout.addLayout(btn_row)

        dlg.adjustSize()
        dlg.exec()

    def _write_last_seen_version(self) -> None:
        """Persist the current VERSION as last_seen_version in the state file."""
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as _fh:
                _sd = json.load(_fh)
        except (FileNotFoundError, json.JSONDecodeError):
            _sd = {}
        _sd["last_seen_version"] = VERSION
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        with open(STATE_FILE, "w", encoding="utf-8") as _fh:
            json.dump(_sd, _fh, indent=2)

    def _show_whats_new(self, new_entries: dict[str, list[str]]) -> None:
        """Show a What's New dialog containing only changelog entries newer than last seen."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Deep Space Astro - Workflow Companion - What's New")
        dlg.setMinimumWidth(700)
        dlg.setWindowFlags(dlg.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(16, 16, 16, 12)
        layout.setSpacing(8)

        intro = QLabel(f"<b>What's new in Workflow Companion v{VERSION}</b>")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        text = QTextEdit()
        text.setReadOnly(True)
        text.setFrameShape(QFrame.Shape.NoFrame)
        html_content = "<style>p{margin:4px 0;} ul{margin:2px 0 8px 0; padding-left:20px;} li{margin:1px 0;}</style>"
        for ver, items in new_entries.items():
            html_content += f"<p><b>{ver}</b></p><ul>"
            for item in items:
                html_content += f"<li>{item}</li>"
            html_content += "</ul>"
        text.setHtml(html_content)
        layout.addWidget(text)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(dlg.accept)
        layout.addWidget(buttons)
        dlg.adjustSize()
        dlg.exec()
        self._write_last_seen_version()

    def _show_release_notes(self):
        """Display the full version history from the CHANGELOG dict."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Deep Space Astro - Workflow Companion - Release Notes")
        dlg.setMinimumWidth(700)
        dlg.setMinimumHeight(600)
        dlg.setWindowFlags(dlg.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(16, 16, 16, 12)
        layout.setSpacing(8)

        text = QTextEdit()
        text.setReadOnly(True)
        text.setFrameShape(QFrame.Shape.NoFrame)
        html_content = "<style>p{margin:2px 0;} ul{margin:2px 0 6px 0; padding-left:20px;} li{margin:1px 0;}</style>"
        for ver, items in CHANGELOG.items():
            html_content += f"<p><b>v{ver}</b></p><ul>"
            for item in items:
                html_content += f"<li>{item}</li>"
            html_content += "</ul>"
        text.setHtml(html_content)
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
            # Discard: revert notes from saved workflow, clear panel state
            try:
                with open(STATE_FILE, "r", encoding="utf-8") as fh:
                    saved = json.load(fh)
                saved_wf = saved.get("workflows", {}).get(self._active_workflow, {})
                self.notes = dict(saved_wf.get("notes", {}))
            except (FileNotFoundError, json.JSONDecodeError):
                self.notes = {}
            self.checked_paths.clear()
            self.favorites.clear()
            self._active_workflow = ""
        g = self.geometry()
        geom = {"x": g.x(), "y": g.y(), "w": g.width(), "h": g.height()}
        on_top = self._ontop_chk.isChecked()
        save_state(self.checked_paths, self.favorites, scripts_dir=self.scripts_dir, geometry=geom,
                   active_workflow=self._active_workflow, notes=self.notes, on_top=on_top)
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
            self._refresh_btn.hide()
            self._clear_btn.hide()
            self._add_group_btn.hide()
            self._add_step_btn.hide()
            self._get_workflows_btn.hide()
            self._toggle_btn.setText("Show Panel")
            self._toggle_btn.setToolTip("Show the left panel")
            self._left_btn_widget.hide()
            self._right_panel._title_lbl.hide()
            self._right_panel._hint_lbl.hide()
            self._right_panel._wf_export_btn.hide()
            self._right_panel._wf_import_btn.hide()
            self._right_panel._wf_delete_btn.hide()
            self._sync_footer()
            QTimer.singleShot(0, self._shrink_to_right_panel)
        else:
            self._left_panel.show()
            self._refresh_btn.show()
            self._clear_btn.show()
            self._add_group_btn.show()
            self._add_step_btn.show()
            self._get_workflows_btn.show()
            self._toggle_btn.setText("Hide Panel")
            self._toggle_btn.setToolTip("Hide the left panel")
            self._left_btn_widget.show()
            self._right_panel._title_lbl.show()
            self._right_panel._hint_lbl.show()
            self._right_panel._wf_export_btn.show()
            self._right_panel._wf_import_btn.show()
            self._right_panel._wf_delete_btn.show()
            QTimer.singleShot(0, self._sync_footer)
            if hasattr(self, "_expanded_geometry"):
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

        # Measure only the visible non-scroll widgets inside SelectedPanel
        # (workflow label, workflow row, description label) to avoid counting
        # any chrome that was previously inside the panel but has since moved.
        rp_layout = self._right_panel.layout()
        panel_chrome_h = 0
        if rp_layout is not None:
            for i in range(rp_layout.count()):
                item = rp_layout.itemAt(i)
                w = item.widget() if item else None
                ly = item.layout() if item else None
                if w is not None and w is not self._right_panel._scroll and w.isVisible():
                    panel_chrome_h += w.sizeHint().height() + rp_layout.spacing()
                elif ly is not None:
                    # QHBoxLayout sub-rows (e.g. workflow button row)
                    panel_chrome_h += ly.sizeHint().height() + rp_layout.spacing()
        rp_margins = self._right_panel.contentsMargins()
        panel_chrome_h += rp_margins.top() + rp_margins.bottom()

        total_h = (fixed_h + content_h + panel_chrome_h
                   + cw_margins.top() + cw_margins.bottom())

        # ── screen height cap ─────────────────────────────────────────────
        screen = QApplication.screenAt(self.geometry().center())
        if screen is None:
            screen = QApplication.primaryScreen()
        screen_h = screen.availableGeometry().height()
        max_h = int(screen_h * 0.90)
        if total_h > max_h:
            total_h = max_h

        self._update_scroll_policy()
        self.resize(total_w, total_h)

    def _update_scroll_policy(self):
        """Show the right panel scrollbar only when content exceeds the scroll area."""
        if not self._left_panel.isVisible():
            scroll = self._right_panel._scroll
            content_h = self._right_panel._container.sizeHint().height()
            viewport_h = scroll.viewport().height()
            if content_h > viewport_h:
                scroll.setVerticalScrollBarPolicy(
                    Qt.ScrollBarPolicy.ScrollBarAlwaysOn
                )
            else:
                scroll.setVerticalScrollBarPolicy(
                    Qt.ScrollBarPolicy.ScrollBarAlwaysOff
                )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_scroll_policy()

# ── Entry-point ───────────────────────────────────────────────────────────────


def main():
    global STATE_FILE, _DIALOG_GROUPS, _SIRILPY_HAS_DIALOGS

    app = QApplication.instance() or QApplication(sys.argv)

    # ── Deep Space Astro dark theme ───────────────────────────────────────────
    DSA_STYLESHEET = """
        /* ── Base ─────────────────────────────────────────────────────────── */
        QWidget {
            background-color: #0d1117;
            color: #c9d1d9;
        }
        QMainWindow, QDialog {
            background-color: #0d1117;
        }

        /* ── Scroll areas ─────────────────────────────────────────────────── */
        QScrollArea, QScrollArea > QWidget > QWidget {
            background-color: #0d1117;
            border: none;
        }
        QScrollBar:vertical {
            background: #161b22;
            width: 8px;
            margin: 0;
            border-radius: 4px;
        }
        QScrollBar::handle:vertical {
            background: #30363d;
            border-radius: 4px;
            min-height: 20px;
        }
        QScrollBar::handle:vertical:hover {
            background: #1f6feb;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0;
        }

        /* ── Frames / panels ──────────────────────────────────────────────── */
        QFrame[frameShape="1"] {
            border: 1px solid #21262d;
            border-radius: 4px;
        }
        QSplitter::handle {
            background: #21262d;
            width: 2px;
        }

        /* ── Labels ───────────────────────────────────────────────────────── */
        QLabel {
            background: transparent;
            color: #c9d1d9;
        }

        /* ── Line edit / search ───────────────────────────────────────────── */
        QLineEdit {
            background-color: #161b22;
            color: #c9d1d9;
            border: 1px solid #30363d;
            border-radius: 4px;
            padding: 3px 6px;
            selection-background-color: #1f6feb;
        }
        QLineEdit:focus {
            border: 1px solid #1f6feb;
        }

        /* ── Buttons ──────────────────────────────────────────────────────── */
        QPushButton {
            background-color: #21262d;
            color: #c9d1d9;
            border: 1px solid #30363d;
            border-radius: 4px;
            padding: 4px 10px;
        }
        QPushButton:hover {
            background-color: #1f6feb;
            border-color: #388bfd;
            color: #ffffff;
        }
        QPushButton:pressed {
            background-color: #1158c7;
            border-color: #1f6feb;
        }
        QPushButton:disabled {
            background-color: #161b22;
            color: #484f58;
            border-color: #21262d;
        }

        /* ── Checkbox ─────────────────────────────────────────────────────── */
        QCheckBox {
            color: #c9d1d9;
            spacing: 6px;
        }
        QCheckBox::indicator {
            width: 14px;
            height: 14px;
            border: 1px solid #30363d;
            border-radius: 3px;
            background: #161b22;
        }
        QCheckBox::indicator:checked {
            background: #1f6feb;
            border-color: #388bfd;
        }

        /* ── ComboBox ─────────────────────────────────────────────────────── */
        QComboBox {
            background-color: #21262d;
            color: #c9d1d9;
            border: 1px solid #30363d;
            border-radius: 4px;
            padding: 3px 8px;
        }
        QComboBox:hover {
            border-color: #1f6feb;
        }
        QComboBox QAbstractItemView {
            background-color: #161b22;
            color: #c9d1d9;
            border: 1px solid #30363d;
            selection-background-color: #1f6feb;
        }
        QComboBox::drop-down {
            border: none;
            width: 20px;
        }
        QComboBox::down-arrow {
            image: none;
        }

        /* ── Text edit (help / notes dialogs) ─────────────────────────────── */
        QTextEdit {
            background-color: #161b22;
            color: #c9d1d9;
            border: none;
        }

        /* ── Menu ─────────────────────────────────────────────────────────── */
        QMenu {
            background-color: #161b22;
            color: #c9d1d9;
            border: 1px solid #30363d;
        }
        QMenu::item:selected {
            background-color: #1f6feb;
            color: #ffffff;
        }

        /* ── Tooltip ──────────────────────────────────────────────────────── */
        QToolTip {
            background-color: #161b22;
            color: #c9d1d9;
            border: 1px solid #30363d;
            padding: 4px;
        }

        /* ── Dialog button box ────────────────────────────────────────────── */
        QDialogButtonBox QPushButton {
            min-width: 70px;
        }

        /* ── Divider line in right panel ──────────────────────────────────── */
        QFrame[frameShape="4"] {
            color: #30363d;
            background: #30363d;
            max-height: 1px;
        }
    """
    app.setStyleSheet(DSA_STYLESHEET)


    # Build dialog groups now, before the UI is shown, so it never flashes during use
    if _SIRILPY_HAS_DIALOGS and _DIALOG_GROUPS is None:
        _DIALOG_GROUPS = _build_dialog_groups()

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
            f"# Deep Space Astro - Workflow Companion v{VERSION}\n"
            "# Author: Rich Stevenson - Deep Space Astro\n"
            "# YouTube:   https://www.youtube.com/@DeepSpaceAstro\n"
            "# Facebook:  https://www.facebook.com/@DeepSpaceAstro\n"
            "# Tiktok:    https://www.tiktok.com/@deepspaceastro\n"
            "# Instagram: https://www.instagram.com/deepspaceastro_official/\n"
            "# Discord:   https://discord.deepspaceastro.net\n"
            "#################################################################"
        )
        siril_instance.log(header_msg, color=LogColor.GREEN)

        # Version check - Functions require Siril 1.4.3 or later
        try:
            siril_instance.cmd("requires", "1.4.3")
        except Exception:
            QMessageBox.information(
                None,
                "Siril Version Warning",
                "Siril 1.4.3 or later is required to use Functions. In older versions, "
                "Functions will not appear in the left panel and will generate an error "
                "if run from the right panel."
            )

        siril_instance.disconnect()
    except Exception:
        pass  # Running outside Siril

    # Resolve state file path - prefer Siril's own config dir
    if config_dir:
        STATE_FILE = os.path.join(config_dir, "workflow_companion.json")
        _old_state = os.path.join(config_dir, "script_launcher_state.json")
    else:
        STATE_FILE = os.path.join(os.path.expanduser("~"), ".siril",
                                  "workflow_companion.json")
        _old_state = os.path.join(os.path.expanduser("~"), ".siril",
                                  "script_launcher_state.json")

    # Migrate state file from old name if needed
    if os.path.exists(_old_state) and not os.path.exists(STATE_FILE):
        try:
            os.rename(_old_state, STATE_FILE)
        except Exception:
            pass  # Non-fatal - will start with a fresh state file

    # Locate siril-scripts directory via Siril API
    scripts_dir = find_siril_scripts_dir(user_data_dir)

    if scripts_dir is None:
        scripts_dir = QFileDialog.getExistingDirectory(
            None, "Locate your siril-scripts directory"
        )
        if not scripts_dir:
            sys.exit("No scripts directory selected.")

    scripts = scan_scripts(scripts_dir)

    # Also scan Siril's built-in system scripts folder
    system_scripts_dir = find_siril_system_scripts_dir()
    if system_scripts_dir:
        scripts += scan_system_scripts(system_scripts_dir)

    if not scripts:
        QMessageBox.warning(
            None, "No Scripts Found",
            f"No .py scripts were found in any subfolder of:\n{scripts_dir}"
        )
        sys.exit(0)

    window = WorkflowCompanionWindow(scripts_dir, scripts)
    window.show()
    # If the user has scripts already selected, start with the left panel hidden.
    # If the right panel is empty, show the left panel so they can pick scripts.
    if window.checked_paths:
        QTimer.singleShot(0, window._toggle_left)
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 