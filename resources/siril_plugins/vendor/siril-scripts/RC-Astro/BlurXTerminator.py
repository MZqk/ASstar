# Copyright (C) 2012-2026 team free-astro (see more in AUTHORS file)
# Reference site is https://siril.org
# SPDX-License-Identifier: GPL-3.0-or-later
"""
BlurXTerminator deconvolution for Siril.

A self-contained wrapper around RC-Astro's command-line tool (the ``bxt``
subcommand of the ``rc-astro`` executable). It sharpens / deconvolves the
currently loaded image or a loaded FITS sequence and loads the result back into
Siril.

The set of processing options, together with their ranges and defaults, is read
live from ``rc-astro bxt --json`` so the script stays in step with the tool. A
built-in copy of the parameters is used as a fallback.

The script offers a PyQt6 graphical interface when launched from the Scripts
menu, and a command-line interface for use with the ``pyscript`` command.

BlurXTerminator is a separate, commercial program by Russell Croman
(https://www.rc-astro.com) and must be installed and licensed separately. Its
location is configured inside this script.

Usage (GUI):
    Launch from the Siril Scripts menu.

Usage (CLI, via the pyscript command). Options mirror rc-astro's own flags,
read live from the schema (e.g. --ss, --sn, --ansr/--no-ansr, --correct-only):
    pyscript BlurXTerminator.py --sn 0.5 --ansr
    pyscript BlurXTerminator.py --sequence --ss 0.25
    pyscript BlurXTerminator.py --correct-only

Report bugs with this script to the siril-scripts project, not to RC-Astro.

Author: Cyril Richard
"""

# Version history
# 1.0.0 - Initial release: BlurXTerminator (rc-astro bxt) support, PyQt6 GUI +
#         CLI, single image and FITS sequence processing, native bit-depth FITS
#         exchange. The whole UI and command line are driven off the unified
#         --json schema v3 (verbatim `flag`, `groups`, `modes`,
#         `disabledIf`/`visibleIf`, gui-only params, enum/unit) with no
#         product-specific hard-coding; schemaVersion is checked. Processing
#         consumes the v3 NDJSON event stream on stdout (status, device,
#         progress, warning, error, update info).
# 1.0.1 - Add an inference-engine selector (GUI combo + --engine on the CLI).
#         The available back-ends are read from the installed build (auto, cpu
#         and the GPU back-end: cuda on Linux, dml on Windows); choosing 'cpu'
#         forces the software path on low-memory GPUs that run out of memory
#         under the auto-selected GPU back-end.
# 1.0.2 - Support rc-astro 0.9.7+ (schemaVersion 4): pick the compute target with
#         the global --device (GUI + CLI) and gate Run on the license. Add an AI
#         model selector above the parameters: switching model re-reads the schema
#         and rebuilds the controls, as a model can expose different parameters.
# 1.0.3 - Check for an RC-Astro update ('rc-astro update') in the background as
#         soon as the GUI opens, instead of only surfacing it in the log after a
#         run finishes. Shows a popup with the new version and an "Update Now"
#         button that runs 'rc-astro update --install' and re-probes the schema
#         afterward.
# 1.0.4 - Fix check_for_update() missing pending updates when only one version
#         is behind: rc-astro prints "An update is available" (singular) in
#         that case rather than "Updates are available" (plural), which the
#         detection only matched literally. Now matches either wording.
# 1.0.5 - Fix sequence processing not writing a .seq file (issue #116):
#         create_new_seq() was called with the prefix alone, but Siril names
#         each frame "<prefix><seqname>NNNNN.ext", so the sequence root is the
#         prefix prepended to the source sequence name. Build the full root.

import sirilpy as s

s.ensure_installed("PyQt6", "numpy", "astropy")

import os
import re
import sys
import json
import time
import shutil
import tempfile
import argparse
import threading
import subprocess
from pathlib import Path

import numpy as np
from astropy.io import fits

from sirilpy import LogColor, SequenceType

from PyQt6.QtCore import Qt, QSize, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap, QPainter
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QCheckBox, QSpinBox,
    QDoubleSpinBox, QRadioButton, QButtonGroup, QProgressBar,
    QMessageBox, QPlainTextEdit, QDialog, QSlider, QComboBox, QScrollArea,
    QFrame,
)
try:
    from PyQt6.QtSvg import QSvgRenderer
    _HAVE_QTSVG = True
except ImportError:  # QtSvg ships with PyQt6; degrade gracefully if absent
    _HAVE_QTSVG = False

VERSION = "1.0.5"
PRODUCT_LABEL = "BlurXTerminator"
CONFIG_SUBDIR = "BlurXTerminator"
CONFIG_FILENAME = "settings.json"
BXT_WEBSITE = "https://www.rc-astro.com"
# rc-astro is a multi-tool; BlurXTerminator is the 'bxt' subcommand.
SUBCOMMAND = "bxt"
BXT_EXE_NAMES = ("rc-astro", "rc-astro.exe", "RC-Astro", "RC-Astro.exe")
PREFIX_BXT = "bxt_"

# --json contract version this script targets (see rc-astro's README-DEVS.txt).
# We support both v3 (0.9.1/0.9.3: per-product `--engine`, top-level `device`
# event) and v4 (0.9.7+: global `--device`, `info`/`topic:"device"` event,
# `license` object on the parameter document, exit code 77 when unlicensed). The
# vocabulary used at run time is chosen from the document's own schemaVersion, so
# a host hitting either CLI keeps working; a newer binary only warns.
SCHEMA_VERSION = 4

# Built-in fallback schema (a copy of 'rc-astro bxt --json', schemaVersion 4),
# used only if the live schema cannot be read. The live schema always takes
# precedence, and the whole UI/command line is built from it generically, so this
# is purely a safety net. Nothing else in the script is product-specific.
DEFAULT_ML_VERSION = 4
DEFAULT_SCHEMA = {
    "schemaVersion": SCHEMA_VERSION,
    "key": SUBCOMMAND,
    "name": "RC-Astro BlurXTerminator",
    "mlVersion": DEFAULT_ML_VERSION,
    "parameters": [
        {"label": "Tile Overlap", "name": "overlap", "flag": "--overlap",
         "description": "Fractional overlap between adjacent tiles",
         "type": "float", "precision": 2, "default": 0.2, "min": 0.0, "max": 0.5},
        {"label": "Sharpen Stars", "name": "ss", "flag": "--ss",
         "description": "Amount of stellar sharpening",
         "type": "float", "precision": 2, "default": 0.0, "min": 0.0, "max": 0.7},
        {"label": "Adjust Star Halos", "name": "ash", "flag": "--ash",
         "description": "Reduce or enlarge star halos",
         "type": "float", "precision": 2, "default": 0.0, "min": -0.5, "max": 0.5},
        {"label": "Nonstellar Radius", "name": "nsr", "flag": "--nsr",
         "description": "Manual nonstellar PSF radius (must be 0.0 when "
                        "auto-nonstellar-radius is true)",
         "type": "float", "precision": 1, "default": 0.0, "min": 0.0, "max": 8.0,
         "disabledIf": "ansr", "disabledValue": 0.0, "unit": "pixels"},
        {"label": "Auto Nonstellar Radius", "name": "ansr", "flag": "--ansr",
         "description": "Estimate the nonstellar PSF radius automatically",
         "type": "bool", "default": True},
        {"label": "Sharpen Nonstellar", "name": "sn", "flag": "--sn",
         "description": "Amount of nonstellar sharpening",
         "type": "float", "precision": 2, "default": 0.0, "min": 0.0, "max": 1.0},
    ],
    "groups": [
        {"name": "optionsGroup", "label": "Options", "params": [],
         "modes": ["correctOnlyMode"]},
        {"name": "stellarGroup", "label": "Stellar Adjustments",
         "params": ["ss", "ash"]},
        {"name": "nonstellarGroup", "label": "Nonstellar Adjustments",
         "params": ["ansr", "nsr", "sn"]},
        {"name": "engineGroup", "label": "Engine", "params": ["overlap"]},
    ],
    "modes": [
        {"flag": "--correct-only", "name": "correctOnlyMode", "label": "Correct Only",
         "description": "Correct PSF aberrations without sharpening",
         "pins": {"ansr": True, "ash": 0.0, "nsr": 0.0, "sn": 0.0, "ss": 0.0}},
    ],
}


# ---------------------------------------------------------------------------
# Schema helpers (generic; nothing here is bxt-specific)
# ---------------------------------------------------------------------------

def schema_params(schema):
    """Mapping of parameter name -> parameter dict, preserving order."""
    return {p["name"]: p for p in schema.get("parameters", []) if "name" in p}


def schema_modes(schema):
    """Mapping of mode name -> mode dict."""
    return {m["name"]: m for m in schema.get("modes", []) if "name" in m}


def schema_uses_device(schema):
    """True when this schema's contract version selects the compute target with
    the global `--device` option (schemaVersion >= 4, rc-astro 0.9.7+); False for
    the older per-product `--engine` (v3). Defaults to the modern path."""
    try:
        return int(schema.get("schemaVersion", SCHEMA_VERSION)) >= 4
    except (TypeError, ValueError, AttributeError):
        return True


def truthy(value):
    """Boolean state of a parameter value for visibleIf/disabledIf evaluation:
    bools by their state, everything else true when non-zero."""
    if isinstance(value, bool):
        return value
    try:
        return float(value) != 0.0
    except (TypeError, ValueError):
        return bool(value)


def eval_condition(expr, state):
    """Evaluate a visibleIf/disabledIf expression: identifiers combined with
    ! && || and parentheses (see README-DEVS.txt section 6). `state` maps a
    parameter name to its current value. Unknown identifiers are false. On any
    parse error the expression is treated as false (fail safe)."""
    if not expr:
        return True
    tokens = re.findall(r"!|&&|\|\||\(|\)|[A-Za-z_][A-Za-z0-9_]*", expr)
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else None

    def advance():
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        return tok

    def parse_or():
        val = parse_and()
        while peek() == "||":
            advance()
            val = parse_and() or val
        return val

    def parse_and():
        val = parse_unary()
        while peek() == "&&":
            advance()
            rhs = parse_unary()
            val = val and rhs
        return val

    def parse_unary():
        if peek() == "!":
            advance()
            return not parse_unary()
        return parse_primary()

    def parse_primary():
        tok = peek()
        if tok == "(":
            advance()
            val = parse_or()
            if peek() == ")":
                advance()
            return val
        if tok is None or tok in ("&&", "||", ")"):
            raise ValueError("unexpected end of expression")
        advance()
        return truthy(state.get(tok))

    try:
        result = parse_or()
        if pos != len(tokens):
            raise ValueError("trailing tokens")
        return bool(result)
    except (ValueError, IndexError):
        return False


def normalize_schema(data):
    """Validate and lightly normalise a 'bxt --json' document. Parameters,
    groups and modes are passed through verbatim (per the forward-compatibility
    rules: drive off `name`, treat `flag` as opaque, ignore unknown keys), with
    only the fields the rest of the script relies on coerced to sane types."""
    params = []
    for p in data.get("parameters", []):
        if "name" not in p:
            continue
        q = dict(p)  # keep every key, including ones we do not interpret
        q.setdefault("label", p["name"])
        q.setdefault("type", "float")
        q.setdefault("description", "")
        if "precision" in q:
            try:
                q["precision"] = int(q["precision"])
            except (TypeError, ValueError):
                q["precision"] = 2
        if "default" not in q:
            q["default"] = False if q["type"] == "bool" else 0
        params.append(q)
    out = dict(data)
    out["parameters"] = params
    out.setdefault("mlVersion", DEFAULT_ML_VERSION)
    out.setdefault("groups", [])
    out.setdefault("modes", [])
    try:
        out["mlVersion"] = int(out["mlVersion"])
    except (TypeError, ValueError):
        out["mlVersion"] = DEFAULT_ML_VERSION
    return out


def about_html():
    """Rich-text content for the About dialog."""
    return (
        "<h3>BlurXTerminator for Siril</h3>"
        f"<p>Version {VERSION}</p>"
        "<p>BlurXTerminator is a third-party neural-network tool by Russell "
        "Croman (RC-Astro) that deconvolves and sharpens astronomical images, "
        "correcting and tightening stars and recovering non-stellar detail. "
        "This script drives the RC-Astro command-line tool from Siril and loads "
        "the result back.</p>"
        "<p>The available options and their ranges are read directly from the "
        "tool, so they stay current as BlurXTerminator is updated.</p>"
        "<p>BlurXTerminator is a separate, commercial product and must be "
        "installed and licensed separately. Please make sure it runs correctly "
        "on its own first. Set the location of the <code>rc-astro</code> "
        "executable in this dialog, or make it available on your system PATH.</p>"
        f'<p>Website: <a href="{BXT_WEBSITE}">{BXT_WEBSITE}</a></p>'
    )


# ---------------------------------------------------------------------------
# Product icon (bundled SVG next to the executable; README-DEVS.txt section 12)
# ---------------------------------------------------------------------------

def _default_install_dirs():
    """RC-Astro's default per-OS install directory for the 'rc-astro' CLI
    (README-DEVS.txt section 12). Used to locate the executable and its bundled
    icons when 'rc-astro' is not on PATH — notably macOS (/Applications) and
    Windows, whose install dirs the installer does not add to PATH."""
    if sys.platform.startswith("win"):
        return [r"C:\Program Files\RC-Astro\CLI"]
    if sys.platform == "darwin":
        return ["/Applications/RC-Astro/CLI"]
    return ["/opt/rc-astro"]


def _icon_search_dirs(exe):
    """Directories that may hold the bundled rsc/ icon folder, best first: next
    to the *real* binary (following symlinks — the Linux installer drops a
    symlink in /usr/local/bin pointing at /opt/rc-astro), next to the path as
    given, then the platform's default install dir (so a copy of the binary run
    from elsewhere, e.g. ~/bin, still finds the shipped icons)."""
    dirs = []
    if exe:
        dirs.append(os.path.dirname(os.path.realpath(exe)))
        dirs.append(os.path.dirname(os.path.abspath(exe)))
    dirs.extend(_default_install_dirs())
    return dirs


def find_icon_path(configured):
    """Locate the product's bundled SVG icon, <dir>/rsc/<SUBCOMMAND>.svg, across
    the candidate directories (README-DEVS.txt section 12). Returns the path, or
    None when no icon is found (non-fatal — the UI just omits it)."""
    seen = set()
    for d in _icon_search_dirs(resolve_exe(configured)):
        if not d or d in seen:
            continue
        seen.add(d)
        path = os.path.join(d, "rsc", f"{SUBCOMMAND}.svg")
        if os.path.isfile(path):
            return path
    return None


def render_icon_pixmap(path, px):
    """Render the SVG icon to a crisp, device-pixel-ratio-aware QPixmap of `px`
    logical pixels, or None if it cannot be drawn."""
    if not path:
        return None
    screen = QApplication.primaryScreen()
    ratio = screen.devicePixelRatio() if screen is not None else 1.0
    side = max(1, round(px * ratio))
    if _HAVE_QTSVG:
        try:
            renderer = QSvgRenderer(path)
            if renderer.isValid():
                pm = QPixmap(side, side)
                pm.fill(Qt.GlobalColor.transparent)
                painter = QPainter(pm)
                renderer.render(painter)
                painter.end()
                pm.setDevicePixelRatio(ratio)
                return pm
        except Exception:
            pass
    # Fallback: the qsvg icon engine bundled with PyQt6, via QIcon.
    try:
        pm = QIcon(path).pixmap(QSize(side, side))
        if not pm.isNull():
            pm.setDevicePixelRatio(ratio)
            return pm
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# FITS working-file I/O (native bit depth, via astropy for orientation symmetry)
# ---------------------------------------------------------------------------
#
# Siril gives pixel data as a planar numpy array: (H, W) mono or (C, H, W)
# colour, dtype uint16 or float32. BlurXTerminator handles full 32-bit FITS, so
# the array is written and read back unchanged at its native depth. Siril keeps
# the loaded image's own metadata (set_image_pixeldata only changes pixels), so
# the working file carries no header.
#
# Siril uses a bottom-up row order while rc-astro reads/writes FITS top-down, so
# the result comes back vertically mirrored. A single vertical flip on the way
# out cancels that (BlurXTerminator's processing is orientation independent).
# The height axis is the last-but-one for both mono (H, W) and colour (C, H, W).

def write_working_fits(path, arr):
    fits.PrimaryHDU(np.ascontiguousarray(arr)).writeto(path, overwrite=True)


def read_working_fits(path, like_dtype):
    with fits.open(path) as hdul:
        data = np.asarray(hdul[0].data)
    data = np.flip(data, axis=-2)
    if like_dtype == np.uint16:
        data = np.clip(np.rint(data), 0, 65535).astype(np.uint16)
    else:
        data = data.astype(np.float32)
    return np.ascontiguousarray(data)


def depth_for_dtype(dtype):
    return "16U" if dtype == np.uint16 else "32F"


# ---------------------------------------------------------------------------
# Option validation / argument building (driven by the schema)
# ---------------------------------------------------------------------------

def validate_values(values, schema):
    """Clamp/coerce a {name: value} dict to the schema's parameter types and
    bounds. Returns a complete dict keyed by every parameter name."""
    out = {}
    for name, p in schema_params(schema).items():
        ptype = p.get("type", "float")
        v = values.get(name, p.get("default"))
        if ptype == "bool":
            out[name] = bool(v)
        elif ptype == "enum":
            choices = [o.get("value") for o in p.get("options", [])]
            out[name] = v if (not choices or v in choices) else p.get("default")
        elif ptype == "int":
            try:
                v = int(round(float(v)))
            except (TypeError, ValueError):
                v = int(p.get("default", 0))
            if p.get("min") is not None:
                v = max(int(p["min"]), v)
            if p.get("max") is not None:
                v = min(int(p["max"]), v)
            out[name] = v
        else:  # float
            try:
                v = float(v)
            except (TypeError, ValueError):
                v = float(p.get("default", 0.0))
            if p.get("min") is not None:
                v = max(float(p["min"]), v)
            if p.get("max") is not None:
                v = min(float(p["max"]), v)
            out[name] = v
    return out


def _format_param_arg(p, value):
    """Format one parameter as CLI tokens, using its verbatim `flag`."""
    flag = p["flag"]
    ptype = p.get("type", "float")
    if ptype == "bool":
        # Bare switch: emit the flag to turn on, '<flag>=false' to turn off.
        return [flag] if value else [f"{flag}=false"]
    if ptype == "int":
        return [flag, str(int(round(float(value))))]
    if ptype == "enum":
        return [flag, str(value)]
    return [flag, f"{float(value):.{int(p.get('precision', 2))}f}"]


def build_command_args(values, schema, mode=None):
    """Build the rc-astro option list from validated values, honoring the schema
    (README-DEVS.txt section 8): emit an engaged mode's flag and skip its pinned
    parameters; skip gui-only parameters (no `flag`) and any parameter currently
    inactive — disabled by its `disabledIf` expression or hidden by its
    `visibleIf` expression. Emitting an inactive parameter can raise a conflict
    error (e.g. mutually exclusive denoise bands), so the command mirrors exactly
    the controls the schema's logic keeps live."""
    params = schema_params(schema)
    modes = schema_modes(schema)
    args = []
    pinned = set()

    effective = dict(values)
    if mode and mode in modes:
        pins = modes[mode].get("pins", {})
        effective.update(pins)
        pinned = set(pins)
        args.append(modes[mode]["flag"])

    state = {name: truthy(effective.get(name)) for name in params}
    for name, p in params.items():
        if name in pinned:
            continue
        if not p.get("flag") or p.get("guiOnly"):
            continue
        if p.get("disabledIf") and eval_condition(p["disabledIf"], state):
            continue
        if p.get("visibleIf") and not eval_condition(p["visibleIf"], state):
            continue
        args += _format_param_arg(p, effective.get(name, p.get("default")))
    return args


# ---------------------------------------------------------------------------
# rc-astro executable: resolution, schema probe, invocation
# ---------------------------------------------------------------------------

_bxt_lock = threading.Lock()


class BxtError(Exception):
    pass


def find_bxt_on_path():
    for name in BXT_EXE_NAMES:
        found = shutil.which(name)
        if found:
            return found
    # Not on PATH (the usual case on macOS and Windows): fall back to the
    # default per-OS install directory.
    for d in _default_install_dirs():
        for name in BXT_EXE_NAMES:
            cand = os.path.join(d, name)
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                return cand
    return None


def resolve_exe(configured):
    if configured:
        candidate = os.path.expanduser(configured.strip())
        if candidate:
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
            on_path = shutil.which(candidate)
            if on_path:
                return on_path
    return find_bxt_on_path()


def run_account_command(configured, args, timeout=120):
    """Run a one-shot 'rc-astro <SUBCOMMAND>' management command that exits without
    processing an image (currently --license and --activate). Returns (ok, text):
    ok is True on exit status 0, text is the combined stdout/stderr for display.
    Used by the License dialog so activation needs only an e-mail and key."""
    exe = resolve_exe(configured)
    if not exe:
        return (False, "rc-astro executable not found. Set its location first.")
    exe_dir = os.path.dirname(os.path.abspath(exe))
    try:
        proc = subprocess.run(
            [exe, SUBCOMMAND, "--no-banner", *args], cwd=exe_dir,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=timeout, text=True, errors="replace")
    except subprocess.TimeoutExpired:
        return (False, "rc-astro did not respond in time.")
    except OSError as exc:
        return (False, f"Could not run rc-astro: {exc}")
    return (proc.returncode == 0, (proc.stdout or "").strip())


def check_for_update(configured, timeout=15):
    """Run 'rc-astro update' to ask whether a newer RC-Astro build is
    available, without installing anything. Parsed as plain text rather than
    --json: the update subcommand's output has a confirmed, stable shape --

        Updates are available:
          0.9.3  Added automatic CPU fall-back if GPU compute fails.
          ...
          0.9.9  Improved error messages and update handling.
        Run 'rc-astro update --install' to update.

    or, when exactly one update is pending, the singular form instead:

        An update is available:
          0.9.10  Corrected BXT parameter ranges. ...
        Run 'rc-astro update --install' to update.

    or, when nothing is pending:

        You are running the latest version (0.9.9).

    ('--json' is a *global* option that must precede the subcommand, e.g.
    'rc-astro --json update' rather than 'rc-astro update --json' -- and its
    exact schema for this subcommand isn't confirmed, so plain text is the
    safer bet here.)

    Returns a dict {'new': ..., 'instruction': ...} when an update is
    available (the newest version listed, i.e. the last one), or None when
    none is available or the check could not be completed (missing
    executable, timeout, unreadable output) -- any of which should be
    treated as 'nothing to report' rather than an error."""
    exe = resolve_exe(configured)
    if not exe:
        return None
    exe_dir = os.path.dirname(os.path.abspath(exe))
    try:
        proc = subprocess.run(
            [exe, "update", "--no-banner"], cwd=exe_dir,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=timeout, text=True, errors="replace")
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = _ANSI_RE.sub("", proc.stdout or "")
    if not re.search(r"updates?\s+(?:is|are)\s+available", text, re.IGNORECASE):
        return None
    # Each listed update is "  <version>  <description>"; keep the last (i.e.
    # highest / newest) one.
    versions = re.findall(r"^\s*(\d+\.\d+\.\d+)\s+\S", text, re.MULTILINE)
    if not versions:
        return None
    instr_match = re.search(r"^Run .*update --install.*\.\s*$", text, re.MULTILINE)
    instr = (instr_match.group(0).strip() if instr_match
              else "Run 'rc-astro update --install' to update.")
    return {"new": versions[-1], "instruction": instr}


def install_update(configured, timeout=600):
    """Run 'rc-astro update --install' to install a pending update. Returns
    (ok, text) like run_account_command: ok is True on exit status 0, text is
    the combined stdout/stderr for display."""
    exe = resolve_exe(configured)
    if not exe:
        return (False, "rc-astro executable not found. Set its location first.")
    exe_dir = os.path.dirname(os.path.abspath(exe))
    try:
        proc = subprocess.run(
            [exe, "update", "--install"], cwd=exe_dir,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=timeout, text=True, errors="replace")
    except subprocess.TimeoutExpired:
        return (False, "rc-astro did not respond in time.")
    except OSError as exc:
        return (False, f"Could not run rc-astro: {exc}")
    return (proc.returncode == 0, (proc.stdout or "").strip())


# Default compute-selector list when the installed build cannot be queried. 'auto'
# and 'cpu' exist in every rc-astro build (v3 `--engine` and v4 `--device` alike),
# so the safe software path stays available on low-end / low-memory GPUs.
FALLBACK_DEVICES = ("auto", "cpu")


def detect_devices(configured):
    """Return the compute-selector tokens the installed rc-astro accepts, best
    first. On a schemaVersion-4 build (0.9.7+) these are `--device` ids read from
    the global `rc-astro --device --json` document (e.g. 'auto', 'gpu0', 'gpu1',
    'cpu'); on a v3 build they are the per-product `--engine` values parsed from
    the help (e.g. 'auto', 'cuda', 'cpu'). The choice is a host-level option, not
    a schema parameter, so it is discovered here. Falls back to FALLBACK_DEVICES —
    which always offers 'cpu' — when nothing can be read."""
    exe = resolve_exe(configured)
    if not exe:
        return list(FALLBACK_DEVICES)
    exe_dir = os.path.dirname(os.path.abspath(exe))
    # v4: the global --device document lists every device by its exact id.
    try:
        proc = subprocess.run(
            [exe, "--device", "--json"], cwd=exe_dir,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=15, text=True, errors="replace")
        data = json.loads(proc.stdout or "")
        ids = [d["id"] for d in data.get("devices", []) if d.get("id")]
        if ids:
            return ids
    except (OSError, subprocess.TimeoutExpired, ValueError, TypeError, KeyError):
        pass
    # v3: parse the per-product help, e.g. "--engine (text {auto,cuda,cpu}, ...)".
    try:
        proc = subprocess.run(
            [exe, SUBCOMMAND, "--no-banner", "--help"], cwd=exe_dir,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=15, text=True, errors="replace")
    except (OSError, subprocess.TimeoutExpired):
        return list(FALLBACK_DEVICES)
    found = re.search(r"--engine[^\n{]*\{([^}]*)\}", proc.stdout or "")
    if not found:
        return list(FALLBACK_DEVICES)
    devices = [e.strip() for e in found.group(1).split(",") if e.strip()]
    return devices or list(FALLBACK_DEVICES)


def detect_ml_versions(configured):
    """Return the AI model versions the installed rc-astro offers for this
    product, highest first. The first entry is the current default (the model
    used when --ml-version is 0); a host can offer the rest as alternatives.
    Read from the bare `rc-astro --json` product catalog, whose per-product
    `mlVersions` array enumerates every installed model (README-DEVS.txt section
    14). Returns a one-element list when a single model ships, and [] when the
    catalog cannot be read or the build predates this field — in either case the
    GUI offers no model choice and 'latest' (--ml-version 0) is used."""
    exe = resolve_exe(configured)
    if not exe:
        return []
    exe_dir = os.path.dirname(os.path.abspath(exe))
    try:
        proc = subprocess.run(
            [exe, "--json"], cwd=exe_dir,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=15, text=True, errors="replace")
        data = json.loads(proc.stdout or "")
    except (OSError, subprocess.TimeoutExpired, ValueError, TypeError):
        return []
    for prod in data.get("products", []):
        if prod.get("key") == SUBCOMMAND:
            out = []
            for v in prod.get("mlVersions", []):
                try:
                    out.append(int(v))
                except (TypeError, ValueError):
                    pass
            return out
    return []


def license_is_unlicensed(text):
    """True only when rc-astro's --license output positively reports the product
    is not licensed. Deliberately conservative: any other wording (licensed,
    trial, or an unexpected/transient message) is treated as 'do not block', so a
    valid user is never locked out of the Run button."""
    t = (text or "").lower()
    # rc-astro's unlicensed --license line reads e.g. "<Product>: A license is
    # required to use this software."; other code paths say "is not licensed" or
    # "No licensed products found.". Match all of these positively.
    return any(s in t for s in (
        "license is required", "requires a license", "license required",
        "not licensed", "no licensed"))


def fetch_schema(configured, mlversion=0):
    """Probe 'rc-astro bxt --ml-version N --json' (no input file = parameter
    query) and return (status, message, schema).

    ``mlversion`` selects which installed AI model the schema describes (0 =
    latest). This matters because a model can expose a different set of
    parameters, so the schema must be read for the model actually selected.

    status: 'ok'      - live schema read and parsed
            'unknown' - executable ran but the schema could not be read
                        (the built-in fallback schema is returned)
            'missing' - no executable found
    """
    exe = resolve_exe(configured)
    if not exe:
        if configured and configured.strip():
            return ("missing",
                    f"rc-astro executable not found or not executable: {configured.strip()}",
                    DEFAULT_SCHEMA)
        return ("missing",
                "rc-astro was not found. Install RC-Astro, or make the "
                "'rc-astro' executable available on your PATH.",
                DEFAULT_SCHEMA)

    exe_dir = os.path.dirname(os.path.abspath(exe))
    with _bxt_lock:
        try:
            proc = subprocess.run(
                [exe, SUBCOMMAND, "--ml-version", str(int(mlversion)), "--json"],
                cwd=exe_dir,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=30, text=True, errors="replace")
        except subprocess.TimeoutExpired:
            return ("unknown", "rc-astro did not respond to --json; using built-in defaults.",
                    DEFAULT_SCHEMA)
        except OSError as exc:
            return ("missing", f"Could not run rc-astro: {exc}", DEFAULT_SCHEMA)

    try:
        data = json.loads(proc.stdout)
        if data.get("key") == SUBCOMMAND or "parameters" in data:
            schema = normalize_schema(data)
            name = data.get("name", "BlurXTerminator")
            msg = f"{name} detected (model v{schema['mlVersion']})."
            return ("ok", msg, schema)
    except (ValueError, TypeError):
        pass
    return ("unknown",
            "Could not read the BlurXTerminator parameter schema; using built-in defaults.",
            DEFAULT_SCHEMA)


# ANSI colour codes can wrap the inference runtime's stderr lines; strip them
# before matching so a fallback error line is clean.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _parse_event(raw):
    """Parse one NDJSON STDOUT line into an event dict (README-DEVS.txt section
    10), or None if the line carries no JSON event. --json streams one
    self-contained JSON object per line: status, progress, warning, error and
    info (v3 also had a top-level device event, now an info/topic:"device")."""
    if b'"event"' not in raw:
        return None
    try:
        start = raw.index(b"{")
        return json.loads(raw[start:].decode("utf-8", "replace"))
    except (ValueError, UnicodeError):
        return None


def run_bxt(exe, input_fits, output_fits, param_args, mlversion, depth,
            progress_cb=None, log_cb=None, cancel_cb=None, device="auto",
            use_device=True):
    """Run 'rc-astro bxt' on input_fits producing output_fits. Raises BxtError on
    failure. Returns True, or 'cancelled'.

    The machine-readable output is the unified ``--json`` stream: NDJSON events on
    STDOUT (status, progress, warning, error, and info notices), with stderr empty
    on success. Events are dispatched to the progress/log callbacks; a fatal
    'error' event carries the failure detail.

    ``device`` chooses the compute target (``auto`` by default); pass ``cpu`` to
    force the software path on hardware where the GPU runs out of memory.
    ``use_device`` selects the contract: True emits the schemaVersion-4 global
    ``--device`` option (0.9.7+); False the older per-product ``--engine`` (v3)."""
    exe_dir = os.path.dirname(os.path.abspath(exe))
    compute_flag = "--device" if use_device else "--engine"
    argv = ([exe, SUBCOMMAND, "-o", output_fits, "--overwrite",
             "--ml-version", str(int(mlversion)), compute_flag, str(device),
             "--depth", depth, "--json"]
            + list(param_args) + [input_fits])

    errors = []  # 'error'-event messages: the failure detail, on stdout

    def log_device(evt):
        # The resolved compute device, reported as a top-level `device` event in
        # v3 and as `info`/`topic:"device"` in v4 (same payload). May arrive twice
        # (a GPU that fails its first inference falls back to the CPU); we just log
        # each, so the most recent line reflects the device actually used.
        if not log_cb:
            return
        dev = str(evt.get("device", "")).upper() or "?"
        prov = evt.get("provider") or ""
        detail = " / ".join(x for x in (evt.get("name") or "",
                            "" if prov.upper() == dev else prov) if x)
        log_cb(f"Using {dev}" + (f" ({detail})" if detail else ""))

    def handle_event(evt):
        kind = evt.get("event")
        if kind == "progress":
            if progress_cb:
                try:
                    progress_cb(min(1.0, max(0.0, float(evt["done"]) / 100.0)))
                except (KeyError, TypeError, ValueError):
                    pass
        elif kind == "status":
            if log_cb:
                msg = evt.get("message") or str(evt.get("phase", "")).capitalize()
                out = evt.get("output")
                if out and evt.get("phase") in ("saving", "complete"):
                    msg = f"{msg}: {os.path.basename(out)}"
                if msg:
                    log_cb(msg)
        elif kind == "device":  # v3 top-level device event
            log_device(evt)
        elif kind == "warning":
            if log_cb and evt.get("message"):
                log_cb(f"Warning: {evt['message']}")
        elif kind == "error":
            if evt.get("message"):
                errors.append(evt["message"])
                if log_cb:
                    log_cb(f"Error: {evt['message']}")
        elif kind == "info":
            # v4 routes device/trial-license/update notices through `info`, keyed
            # by `topic`; ignore unrecognized topics (e.g. "version").
            topic = evt.get("topic")
            if topic == "device":
                log_device(evt)
            elif topic == "license":
                if log_cb and evt.get("message"):
                    log_cb(evt["message"])
            elif topic == "update":
                if log_cb:
                    instr = (evt.get("instruction")
                             or "Run 'rc-astro update --install' to update.")
                    log_cb(f"A newer RC-Astro version ({evt.get('new', '?')}) is "
                           f"available. {instr}")

    with _bxt_lock:
        try:
            proc = subprocess.Popen(
                argv, cwd=exe_dir,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as exc:
            raise BxtError(f"Could not launch rc-astro: {exc}") from exc

        # In --json mode the whole event stream is NDJSON on STDOUT, one object
        # per line; parse and dispatch each as it arrives.
        def stdout_reader():
            for raw in iter(proc.stdout.readline, b""):
                evt = _parse_event(raw)
                if evt is not None:
                    handle_event(evt)

        # stderr stays empty on success but can still carry harmless library
        # chatter (e.g. an onnxruntime CUDA-provider load failure before rc-astro
        # falls back to the CPU). Drain it, keeping any genuine "Error:" line as a
        # fallback for the rare late failure reported on stderr instead of as an
        # error event.
        stderr_errors = []

        def stderr_reader():
            for raw in iter(proc.stderr.readline, b""):
                # rc-astro streams its NDJSON events on stdout, but a fatal
                # error (notably an unlicensed product) can be reported as an
                # event on stderr instead; dispatch it so the detail is kept.
                evt = _parse_event(raw)
                if evt is not None:
                    handle_event(evt)
                    continue
                text = _ANSI_RE.sub("", raw.decode("utf-8", "replace")).strip()
                if text.lower().startswith("error:"):
                    stderr_errors.append(text[len("error:"):].strip())

        threads = [threading.Thread(target=stderr_reader, daemon=True),
                   threading.Thread(target=stdout_reader, daemon=True)]
        for t in threads:
            t.start()

        cancelled = False
        while proc.poll() is None:
            if cancel_cb and cancel_cb():
                cancelled = True
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                break
            time.sleep(0.1)
        for t in threads:
            t.join(timeout=2)

    if cancelled:
        return "cancelled"
    if proc.returncode != 0:
        detail = (errors or stderr_errors or [""])[-1]
        # An unlicensed product is the usual cause of an otherwise cryptic
        # non-zero exit (e.g. an access-violation status on Windows). v4 gives it
        # a dedicated exit code (77, EX_NOPERM); older builds only say so in the
        # error message. Either way, surface a clear, actionable message instead
        # of the raw exit code.
        if proc.returncode == 77 or any(license_is_unlicensed(e) for e in errors) \
                or license_is_unlicensed(detail):
            raise BxtError(
                f"{PRODUCT_LABEL} is not licensed on this machine. Click "
                "“License…” to activate your license, then try again.")
        msg = f"rc-astro exited with status {proc.returncode}."
        if detail:
            msg += f" {detail}"
        raise BxtError(msg)
    if not os.path.isfile(output_fits):
        raise BxtError("BlurXTerminator did not produce an output file.")
    return True


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

class BxtOptions:
    """Plain options container shared by the GUI and the CLI."""

    def __init__(self):
        self.exe = ""
        self.mlversion = 0  # 0 = latest
        self.engine = "auto"  # compute-device token; key kept as 'engine' for back-compat
        self.values = {}    # {param_name: value}
        self.mode = None    # engaged preset mode name, or None

    def to_dict(self):
        return {"exe": self.exe, "mlversion": self.mlversion,
                "engine": self.engine,
                "values": dict(self.values), "mode": self.mode}

    @classmethod
    def from_dict(cls, d):
        o = cls()
        o.exe = d.get("exe", "")
        o.mlversion = int(d.get("mlversion", 0))
        o.engine = d.get("engine") or "auto"
        o.values = dict(d.get("values", {}))
        o.mode = d.get("mode") or None
        return o


def process_array(arr, opts, schema, progress_cb=None, log_cb=None,
                  cancel_cb=None):
    """Run BlurXTerminator on a single Siril pixel array. Returns the processed
    array (same dtype as input), or None if cancelled."""
    exe = resolve_exe(opts.exe)
    if not exe:
        raise BxtError(
            "rc-astro executable not found. Set its location, or make 'rc-astro' "
            "available on PATH.")

    values = validate_values(opts.values, schema)
    param_args = build_command_args(values, schema, mode=getattr(opts, "mode", None))
    depth = depth_for_dtype(arr.dtype)

    tmp = tempfile.mkdtemp(prefix="siril_bxt_")
    try:
        in_fits = os.path.join(tmp, "input.fits")
        out_fits = os.path.join(tmp, "output.fits")
        write_working_fits(in_fits, arr)

        result = run_bxt(exe, in_fits, out_fits, param_args, opts.mlversion,
                         depth, progress_cb=progress_cb, log_cb=log_cb,
                         cancel_cb=cancel_cb,
                         device=getattr(opts, "engine", "auto"),
                         use_device=schema_uses_device(schema))
        if result == "cancelled":
            return None
        return read_working_fits(out_fits, arr.dtype)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def process_single_image(siril, opts, schema, progress_cb=None, log_cb=None,
                         cancel_cb=None):
    """Run BlurXTerminator on the loaded image and load the result back."""
    arr = siril.get_image_pixeldata()
    result = process_array(arr, opts, schema, progress_cb=progress_cb,
                           log_cb=log_cb, cancel_cb=cancel_cb)
    if result is None:
        return False
    siril.undo_save_state("BlurXTerminator")
    with siril.image_lock():
        siril.set_image_pixeldata(result)
    # The result is applied in place to the loaded image; the preceding rc-astro
    # "Saving: output.fits" status names an internal temp file, not an output.
    if log_cb:
        log_cb("Loaded image updated.")
    return True


def process_sequence(siril, opts, schema, progress_cb=None, log_cb=None,
                     cancel_cb=None):
    """Run BlurXTerminator on every selected frame of the loaded FITS sequence,
    building a new 'bxt_' sequence."""
    seq = siril.get_seq()
    if seq.type != SequenceType.SEQ_REGULAR:
        raise BxtError(
            "Sequence processing requires a FITS sequence (SER / FITSEQ are not "
            "supported by this script). Convert the sequence to FITS first.")

    n = seq.number
    if log_cb:
        log_cb(f"Processing {n} frame(s) from sequence '{seq.seqname}'.")

    for i in range(n):
        if cancel_cb and cancel_cb():
            return False
        if log_cb:
            log_cb(f"Frame {i + 1}/{n}")

        def frame_progress(frac, base=i):
            if progress_cb:
                progress_cb((base + frac) / n)

        arr = siril.get_seq_frame_pixeldata(i)
        result = process_array(arr, opts, schema, progress_cb=frame_progress,
                               log_cb=None, cancel_cb=cancel_cb)
        if result is None:
            return False
        siril.set_seq_frame_pixeldata(i, result, prefix=PREFIX_BXT)
        if progress_cb:
            progress_cb((i + 1) / n)

    # Siril writes each processed frame as "<prefix><seqname>NNNNN<ext>", so the
    # new sequence root is the prefix prepended to the source sequence name, not
    # the prefix alone. Passing only PREFIX_BXT matches no files and no .seq is
    # created (issue #116); build the full root from the source sequence name.
    siril.create_new_seq(PREFIX_BXT + Path(seq.seqname).name)
    if log_cb:
        log_cb("Sequence processing complete.")
    return True


# ---------------------------------------------------------------------------
# Settings persistence
# ---------------------------------------------------------------------------

def settings_path(siril):
    cfg_dir = Path(siril.get_siril_configdir()) / CONFIG_SUBDIR
    cfg_dir.mkdir(parents=True, exist_ok=True)
    return cfg_dir / CONFIG_FILENAME


def load_options(siril):
    try:
        data = json.loads(settings_path(siril).read_text(encoding="utf-8"))
        return BxtOptions.from_dict(data)
    except FileNotFoundError:
        return BxtOptions()
    except Exception:
        return BxtOptions()


def save_options(siril, opts):
    try:
        settings_path(siril).write_text(
            json.dumps(opts.to_dict(), indent=2), encoding="utf-8")
    except Exception as exc:
        siril.log(f"BlurXTerminator: could not save settings: {exc}", LogColor.SALMON)


# ---------------------------------------------------------------------------
# PyQt6 worker thread
# ---------------------------------------------------------------------------

class BxtWorker(QThread):
    progress = pyqtSignal(float)
    message = pyqtSignal(str)
    finished_ok = pyqtSignal(bool, str)

    def __init__(self, siril, opts, schema, sequence_mode):
        super().__init__()
        self.siril = siril
        self.opts = opts
        self.schema = schema
        self.sequence_mode = sequence_mode
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def _cancelled(self):
        return self._cancel

    def run(self):
        try:
            if self.sequence_mode:
                ok = process_sequence(
                    self.siril, self.opts, self.schema,
                    progress_cb=self.progress.emit, log_cb=self.message.emit,
                    cancel_cb=self._cancelled)
            else:
                ok = process_single_image(
                    self.siril, self.opts, self.schema,
                    progress_cb=self.progress.emit, log_cb=self.message.emit,
                    cancel_cb=self._cancelled)
            if self._cancel:
                self.finished_ok.emit(False, "Cancelled.")
            elif ok:
                self.finished_ok.emit(True, "BlurXTerminator finished.")
            else:
                self.finished_ok.emit(False, "BlurXTerminator did not complete.")
        except Exception as exc:
            self.finished_ok.emit(False, str(exc))


# ---------------------------------------------------------------------------
# PyQt6 GUI
# ---------------------------------------------------------------------------

class UpdateCheckWorker(QThread):
    """Runs check_for_update() off the UI thread so a slow or unreachable
    update server never delays showing the window."""
    result = pyqtSignal(object)  # dict when available, else None

    def __init__(self, exe):
        super().__init__()
        self.exe = exe

    def run(self):
        try:
            info = check_for_update(self.exe)
        except Exception:
            info = None
        self.result.emit(info)


class UpdateInstallWorker(QThread):
    """Runs install_update() off the UI thread; installation can take a while."""
    finished_ok = pyqtSignal(bool, str)

    def __init__(self, exe):
        super().__init__()
        self.exe = exe

    def run(self):
        ok, text = install_update(self.exe)
        self.finished_ok.emit(ok, text)


class BxtGUI(QMainWindow):
    def __init__(self, siril):
        super().__init__()
        self.siril = siril
        self.opts = load_options(siril)
        self.schema = DEFAULT_SCHEMA
        self.worker = None
        self.param_widgets = {}   # name -> (param, widget)
        self.param_sliders = {}   # name -> companion slider (float/int params)
        self.param_labels = {}    # name -> QLabel (for show/hide)
        self.mode_widgets = {}    # mode name -> (mode dict, checkbox)
        self.group_boxes = []     # [(group dict, QGroupBox)] for group visibleIf
        self._engaged_mode = None
        self._pre_mode_values = {}  # values stashed while a mode is engaged
        self._suppress_logic = False
        self._fit_signature = None  # visible-widget set the window was last fitted to
        self.has_image = False
        self.has_sequence = False
        try:
            self.has_image = siril.is_image_loaded()
        except Exception:
            pass
        try:
            self.has_sequence = siril.is_sequence_loaded()
        except Exception:
            pass

        self.setWindowTitle(f"BlurXTerminator v{VERSION}")
        self._build_ui()
        # rc-astro is auto-located (PATH or the default per-OS install dir), so
        # probe straight away — there is no executable field for the user to fill.
        self._check_executable()  # fetches the live schema and builds params
        self._load_into_ui()
        self._schedule_fit()
        self._update_check_worker = None
        self._update_install_worker = None
        self._start_update_check()

    def _schedule_fit(self):
        """Re-fit on the next event-loop tick. Deferring is essential: right after
        rebuilding the controls Qt has not yet recomputed the layout, so a
        synchronous sizeHint() is stale and the window would keep its old width."""
        QTimer.singleShot(0, self._fit_to_screen)

    def _fit_to_screen(self):
        """Make the window always match its contents: the width is pinned to
        exactly fit the currently visible controls (so it is not user-resizable
        and a column is never truncated), and the height opens tall enough to
        show everything, capped to the screen — the scroll area then covers any
        overflow on a small display."""
        # Force any pending layout recomputation before measuring: right after
        # rebuilding the controls Qt has not laid them out yet, so sizeHint()
        # would otherwise be stale and the window would keep its old width.
        self.scroll_content.layout().activate()
        self.centralWidget().layout().activate()
        content = self.scroll_content.sizeHint()

        # WIDTH: derive it from the *content*, never from the QScrollArea — the
        # latter's own sizeHint is a small fixed default that ignores its widget,
        # which is what made the dialog shrink. Reserve room for a vertical
        # scrollbar so nothing is clipped if one appears. setFixedWidth both grows
        # and shrinks the window to match the content exactly.
        scrollbar_w = self.scroll.verticalScrollBar().sizeHint().width()
        self.setFixedWidth(content.width() + scrollbar_w + 2)

        # HEIGHT: force the scroll area tall enough to show the whole stack (no
        # scrollbar by default), lift the floor again so the user can still shrink
        # it, then cap to the available screen height.
        self.scroll.setMinimumHeight(content.height())
        self.adjustSize()
        self.scroll.setMinimumHeight(0)
        screen = self.screen() or QApplication.primaryScreen()
        if screen is not None:
            # Leave a little room for window decorations and desktop panels.
            max_h = max(400, screen.availableGeometry().height() - 80)
            self.setMaximumHeight(max_h)
            if self.height() > max_h:
                self.resize(self.width(), max_h)

    # -- UI construction --------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)

        # The settings stack can grow taller than a small screen, so put it in a
        # scroll area and keep the progress bar, log and action buttons pinned in
        # a footer that is always visible.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(scroll, 1)
        self.scroll = scroll

        content = QWidget()
        scroll.setWidget(content)
        self.scroll_content = content
        layout = QVBoxLayout(content)

        # Product header: the product name on the left and the bundled SVG icon
        # aligned to the right on the same line, shown discreetly at the top. The
        # icon is filled in once the executable (hence its sibling rsc/ folder) is
        # resolved; until then only the name shows.
        header = QHBoxLayout()
        header_title = QLabel(PRODUCT_LABEL)
        header_title.setStyleSheet("font-size: 15px; font-weight: 600;")
        self.header_icon = QLabel()
        self.header_icon.setVisible(False)
        header.addWidget(header_title)
        header.addStretch()
        header.addWidget(self.header_icon)
        layout.addLayout(header)

        # rc-astro is located automatically (PATH or the default per-OS install
        # directory), so there is no executable picker — just a status line that
        # reports detection and license state.
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # AI model version. rc-astro can keep older AI models installed next to
        # the latest (this product's previous "AI2" model is version 2). The
        # installed versions are read from the `rc-astro --json` product catalog
        # (highest first, the first being the current default); the selector is
        # shown only when more than one is installed. With a single model there is
        # nothing to choose and 'latest' (--ml-version 0) is always used.
        #
        # It sits above the parameters because a model can expose a different set
        # of parameters: changing it re-reads the schema for the chosen model and
        # rebuilds the controls below (see _on_model_changed), so the panel it
        # drives naturally follows it.
        self.mlversion_combo = None
        ml_versions = detect_ml_versions(self.opts.exe)
        if len(ml_versions) > 1:
            model_group = QGroupBox("AI model")
            model_layout = QHBoxLayout(model_group)
            self.mlversion_combo = QComboBox()
            # Map the newest version to 0 ("latest") so a saved choice keeps
            # following the current model as future versions ship; offer the
            # older models by their explicit version number.
            self.mlversion_combo.addItem(f"Latest (v{ml_versions[0]})", 0)
            for v in ml_versions[1:]:
                self.mlversion_combo.addItem(f"Version {v}", v)
            self.mlversion_combo.setToolTip(
                "AI model version used for processing:\n"
                "• Latest — always use the newest installed model (recommended)\n"
                "• Version N — use an older model, e.g. to reproduce earlier "
                "results\n"
                "Changing the model refreshes the parameters below to match it.")
            idx = self.mlversion_combo.findData(
                int(getattr(self.opts, "mlversion", 0) or 0))
            self.mlversion_combo.setCurrentIndex(idx if idx >= 0 else 0)
            # Connect only after the initial index is set, so building the combo
            # does not trigger a rebuild before the UI is ready.
            self.mlversion_combo.currentIndexChanged.connect(self._on_model_changed)
            model_layout.addWidget(self.mlversion_combo)
            model_layout.addStretch()
            layout.addWidget(model_group)

        # Parameters: grouped boxes built dynamically from the schema and laid
        # out in two columns (see _rebuild_params) to keep the dialog compact.
        self.params_holder = QWidget()
        self.params_holder_layout = QGridLayout(self.params_holder)
        self.params_holder_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.params_holder)

        # Target
        target_group = QGroupBox("Apply to")
        target_layout = QHBoxLayout(target_group)
        self.target_image = QRadioButton("Loaded image")
        self.target_seq = QRadioButton("Loaded sequence")
        self.target_image.setToolTip("Process the single image currently loaded in Siril.")
        self.target_seq.setToolTip("Process every selected frame of the loaded FITS sequence.")
        self.target_group = QButtonGroup(self)
        self.target_group.addButton(self.target_image)
        self.target_group.addButton(self.target_seq)
        target_layout.addWidget(self.target_image)
        target_layout.addWidget(self.target_seq)
        self.target_image.setEnabled(self.has_image)
        self.target_seq.setEnabled(self.has_sequence)
        if self.has_sequence and not self.has_image:
            self.target_seq.setChecked(True)
        else:
            self.target_image.setChecked(True)
        layout.addWidget(target_group)

        # Compute device. The default 'auto' picks a GPU when one is usable, but
        # low-memory GPUs (e.g. Intel Iris Xe) can run out of memory; 'cpu' forces
        # the slower-but-reliable software path. The available devices are read
        # from the installed build: device ids (auto/gpu0/gpu1/cpu) on v4 (0.9.7+),
        # or the legacy engine names (auto/cuda/dml/cpu) on v3.
        engine_group = QGroupBox("Compute device")
        engine_layout = QHBoxLayout(engine_group)
        self.engine_combo = QComboBox()
        self.engine_combo.addItems(detect_devices(self.opts.exe))
        self.engine_combo.setToolTip(
            "Device used for AI inference:\n"
            "• auto — use the best available (a GPU if usable, otherwise the CPU)\n"
            "• cpu — force the CPU; slower, but avoids GPU out-of-memory errors\n"
            "  on low-end hardware\n"
            "• gpu / gpu0 / gpu1 — force a specific GPU")
        idx = self.engine_combo.findText(getattr(self.opts, "engine", "auto"))
        if idx >= 0:
            self.engine_combo.setCurrentIndex(idx)
        engine_layout.addWidget(self.engine_combo)
        engine_layout.addStretch()
        layout.addWidget(engine_group)
        layout.addStretch()

        # Footer: pinned below the scroll area so it stays reachable even when
        # the settings are scrolled or the window is shorter than its contents.
        footer = QWidget()
        footer_layout = QVBoxLayout(footer)
        footer_layout.setContentsMargins(9, 0, 9, 9)
        outer.addWidget(footer)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        footer_layout.addWidget(self.progress_bar)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(500)
        self.log_view.setFixedHeight(80)
        footer_layout.addWidget(self.log_view)

        btn_row = QHBoxLayout()
        about_btn = QPushButton("About")
        about_btn.setToolTip("About BlurXTerminator and this script.")
        about_btn.clicked.connect(self._show_about)
        license_btn = QPushButton("License…")
        license_btn.setToolTip(
            "View this product's license status and activate it with your "
            "account e-mail and license key.")
        license_btn.clicked.connect(self._show_license)
        self.run_btn = QPushButton("Run BlurXTerminator")
        self.run_btn.clicked.connect(self._on_run)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        btn_row.addWidget(about_btn)
        btn_row.addWidget(license_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.close_btn)
        footer_layout.addLayout(btn_row)

    # -- dynamic parameter controls --------------------------------------
    def _clear_param_layout(self):
        while self.params_holder_layout.count():
            item = self.params_holder_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self.param_widgets = {}
        self.param_sliders = {}
        self.param_labels = {}
        self.mode_widgets = {}
        self.group_boxes = []
        self._engaged_mode = None
        self._pre_mode_values = {}
        # NB: _fit_signature is intentionally *not* reset here. Rebuilding the
        # controls for the same schema must not trigger a re-fit (that made the
        # height jump on every exe edit / Run); only a genuine change in the set
        # of visible parameters re-fits the window (see _recompute_logic).

    def _attach_slider(self, name, spin, lo, hi, is_float, prec):
        """Add a slider kept in two-way sync with a spin box (PixInsight-style).
        Returns the slider. Float values are mapped to integer slider ticks via
        the parameter's precision."""
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimumWidth(140)
        scale = 10 ** prec if is_float else 1

        def to_ticks(v):
            return int(round(v * scale))

        slider.setRange(to_ticks(lo), to_ticks(hi))
        slider.setValue(to_ticks(spin.value()))

        def on_slider(v):
            spin.blockSignals(True)
            spin.setValue(v / scale if is_float else int(v))
            spin.blockSignals(False)

        def on_spin(v):
            slider.blockSignals(True)
            slider.setValue(to_ticks(v))
            slider.blockSignals(False)

        slider.valueChanged.connect(on_slider)
        spin.valueChanged.connect(on_spin)
        self.param_sliders[name] = slider
        return slider

    def _add_param_row(self, grid, row, p):
        """Build one label + control (+ synced slider) row into a grid layout and
        register the widget under its schema `name`. Works for any parameter type;
        nothing here is product-specific."""
        name = p["name"]
        label = QLabel(p["label"] + ":")
        ptype = p.get("type", "float")
        slider = None
        if ptype == "bool":
            widget = QCheckBox()
            widget.setChecked(bool(p.get("default")))
            widget.toggled.connect(self._on_control_changed)
        elif ptype == "enum":
            widget = QComboBox()
            for opt in p.get("options", []):
                if isinstance(opt, dict):
                    widget.addItem(str(opt.get("label", opt.get("value"))),
                                   opt.get("value"))
                else:
                    widget.addItem(str(opt), opt)
            idx = widget.findData(p.get("default"))
            if idx >= 0:
                widget.setCurrentIndex(idx)
            widget.currentIndexChanged.connect(self._on_control_changed)
        elif ptype == "int":
            widget = QSpinBox()
            lo = int(p["min"]) if p.get("min") is not None else 0
            hi = int(p["max"]) if p.get("max") is not None else 1000000
            widget.setRange(lo, hi)
            widget.setValue(int(p.get("default", 0)))
            if p.get("unit"):
                widget.setSuffix(f" {p['unit']}")
            if p.get("min") is not None and p.get("max") is not None:
                slider = self._attach_slider(name, widget, lo, hi, False, 0)
            widget.valueChanged.connect(self._on_control_changed)
        else:  # float
            widget = QDoubleSpinBox()
            prec = int(p.get("precision", 2))
            widget.setDecimals(prec)
            lo = float(p["min"]) if p.get("min") is not None else -1e9
            hi = float(p["max"]) if p.get("max") is not None else 1e9
            widget.setRange(lo, hi)
            widget.setSingleStep(10 ** (-prec))
            widget.setValue(float(p.get("default", 0.0)))
            if p.get("unit"):
                widget.setSuffix(f" {p['unit']}")
            if p.get("min") is not None and p.get("max") is not None:
                slider = self._attach_slider(name, widget, lo, hi, True, prec)
            widget.valueChanged.connect(self._on_control_changed)
        widget.setToolTip(p.get("description", ""))
        label.setToolTip(p.get("description", ""))
        grid.addWidget(label, row, 0)
        grid.addWidget(widget, row, 1)
        if slider is not None:
            slider.setToolTip(p.get("description", ""))
            slider.valueChanged.connect(self._on_control_changed)
            grid.addWidget(slider, row, 2)
        grid.setColumnStretch(2, 1)
        self.param_widgets[name] = (p, widget)
        self.param_labels[name] = label

    def _add_mode_row(self, grid, row, m):
        """Add a preset-mode checkbox (README-DEVS.txt section 5)."""
        cb = QCheckBox(m.get("label", m["name"]))
        cb.setToolTip(m.get("description", ""))
        cb.toggled.connect(lambda checked, n=m["name"]: self._sync_modes(n, checked))
        grid.addWidget(cb, row, 0, 1, 3)
        self.mode_widgets[m["name"]] = (m, cb)

    def _rebuild_params(self, schema):
        self._clear_param_layout()
        params = schema_params(schema)
        modes = schema_modes(schema)

        # Collect the group boxes, then lay them out two per row so the dialog
        # stays compact instead of growing into one tall column.
        boxes = []

        # The AI model version is a host-level choice (like the compute device),
        # built once in _build_ui from the `rc-astro --json` catalog's mlVersions
        # list -- not a schema parameter -- so no model selector is created here.

        # Build groups in schema order; fall back to a single section if absent.
        groups = schema.get("groups") or [{
            "name": "paramsGroup", "label": "Parameters",
            "params": list(params.keys()), "modes": [],
        }]
        for group in groups:
            box = QGroupBox(group.get("label", group.get("name", "")))
            if group.get("description"):
                box.setToolTip(group["description"])
            grid = QGridLayout(box)
            r = 0
            for pname in group.get("params", []):
                p = params.get(pname)
                if p is None:
                    continue
                self._add_param_row(grid, r, p)
                r += 1
            for mname in group.get("modes", []):
                m = modes.get(mname)
                if m is None:
                    continue
                self._add_mode_row(grid, r, m)
                r += 1
            # Skip empty sections so an empty "Options" group adds no clutter.
            if r == 0:
                box.deleteLater()
                continue
            self.group_boxes.append((group, box))
            boxes.append(box)

        # Two equal columns, filled left-to-right, top-to-bottom.
        for i, box in enumerate(boxes):
            self.params_holder_layout.addWidget(box, i // 2, i % 2)
        self.params_holder_layout.setColumnStretch(0, 1)
        self.params_holder_layout.setColumnStretch(1, 1)

        self._recompute_logic()

    # -- control value helpers (type-aware, generic) ----------------------
    def _get_value(self, name):
        entry = self.param_widgets.get(name)
        if entry is None:
            return None
        p, w = entry
        ptype = p.get("type", "float")
        if ptype == "bool":
            return w.isChecked()
        if ptype == "enum":
            return w.currentData()
        return w.value()

    def _set_value(self, name, v):
        entry = self.param_widgets.get(name)
        if entry is None:
            return
        p, w = entry
        ptype = p.get("type", "float")
        w.blockSignals(True)
        try:
            if ptype == "bool":
                w.setChecked(bool(v))
            elif ptype == "enum":
                idx = w.findData(v)
                if idx >= 0:
                    w.setCurrentIndex(idx)
            elif ptype == "int":
                w.setValue(int(round(float(v))))
            else:
                w.setValue(float(v))
        except (TypeError, ValueError):
            pass
        finally:
            w.blockSignals(False)
        slider = self.param_sliders.get(name)
        if slider is not None and ptype in ("int", "float"):
            scale = 10 ** int(p.get("precision", 2)) if ptype == "float" else 1
            slider.blockSignals(True)
            slider.setValue(int(round(self._get_value(name) * scale)))
            slider.blockSignals(False)

    def _set_enabled(self, name, enabled):
        entry = self.param_widgets.get(name)
        if entry is not None:
            entry[1].setEnabled(enabled)
        slider = self.param_sliders.get(name)
        if slider is not None:
            slider.setEnabled(enabled)

    def _set_visible(self, name, visible):
        entry = self.param_widgets.get(name)
        if entry is not None:
            entry[1].setVisible(visible)
        slider = self.param_sliders.get(name)
        if slider is not None:
            slider.setVisible(visible)
        label = self.param_labels.get(name)
        if label is not None:
            label.setVisible(visible)

    # -- conditional logic / modes (generic, schema-driven) ---------------
    def _on_control_changed(self, *_):
        self._recompute_logic()

    def _sync_modes(self, toggled_name, checked):
        """Enforce mutual exclusivity of modes, stash/restore the values a mode
        pins, then re-derive all enable/visible state."""
        if self._suppress_logic:
            return
        if checked:
            for mname, (m, cb) in self.mode_widgets.items():
                if mname != toggled_name and cb.isChecked():
                    cb.blockSignals(True)
                    cb.setChecked(False)
                    cb.blockSignals(False)
        engaged = next((n for n, (m, cb) in self.mode_widgets.items()
                        if cb.isChecked()), None)
        if engaged != self._engaged_mode:
            if self._engaged_mode is not None:
                for n, v in self._pre_mode_values.items():
                    self._set_value(n, v)
                self._pre_mode_values = {}
            if engaged is not None:
                pins = self.mode_widgets[engaged][0].get("pins", {})
                self._pre_mode_values = {n: self._get_value(n)
                                         for n in pins if n in self.param_widgets}
            self._engaged_mode = engaged
        self._recompute_logic()

    def _recompute_logic(self):
        """Apply the engaged mode's pins, then evaluate every parameter's
        disabledIf / visibleIf (and group visibleIf) and update the controls."""
        if self._suppress_logic:
            return
        self._suppress_logic = True
        visible_names = []
        try:
            params = schema_params(self.schema)
            pinned = {}
            if self._engaged_mode and self._engaged_mode in self.mode_widgets:
                pinned = dict(self.mode_widgets[self._engaged_mode][0].get("pins", {}))
            for name, v in pinned.items():
                if name in self.param_widgets:
                    self._set_value(name, v)

            state = {name: truthy(self._get_value(name)) for name in self.param_widgets}

            for name, p in params.items():
                if name not in self.param_widgets:
                    continue
                disabled = name in pinned
                if not disabled and p.get("disabledIf"):
                    if eval_condition(p["disabledIf"], state):
                        disabled = True
                        if "disabledValue" in p:
                            self._set_value(name, p["disabledValue"])
                self._set_enabled(name, not disabled)
                visible = True
                if p.get("visibleIf"):
                    visible = eval_condition(p["visibleIf"], state)
                self._set_visible(name, visible)
                if visible:
                    visible_names.append(name)

            for group, box in self.group_boxes:
                if group.get("visibleIf"):
                    box.setVisible(eval_condition(group["visibleIf"], state))
        finally:
            self._suppress_logic = False

        # The visible parameter set decides how wide/tall the dialog must be, so
        # re-fit the window when it changes (e.g. toggling frequency separation
        # reveals extra rows) — otherwise the pinned width would truncate a
        # column. Gated on a signature so ordinary value edits (slider drags) do
        # not resize the window.
        signature = frozenset(visible_names)
        if signature != self._fit_signature:
            self._fit_signature = signature
            self._schedule_fit()

    # -- settings <-> widgets --------------------------------------------
    def _load_into_ui(self):
        for name in self.param_widgets:
            if name in self.opts.values:
                self._set_value(name, self.opts.values[name])
        # Engage the saved mode (if it still exists), then derive enable/visible
        # state. Setting the checkbox routes through _sync_modes.
        if self.opts.mode and self.opts.mode in self.mode_widgets:
            cb = self.mode_widgets[self.opts.mode][1]
            cb.blockSignals(True)
            cb.setChecked(True)
            cb.blockSignals(False)
            self._sync_modes(self.opts.mode, True)
        else:
            self._recompute_logic()

    def _collect_options(self):
        # Read the chosen AI model from the selector when present (multiple models
        # installed); otherwise 'latest' (--ml-version 0). currentData() is the
        # version int (0 for "Latest").
        self.opts.mlversion = (int(self.mlversion_combo.currentData() or 0)
                               if self.mlversion_combo is not None else 0)
        self.opts.engine = self.engine_combo.currentText()
        self.opts.mode = self._engaged_mode
        # Persist the user's real values, not a mode's pinned overrides: while a
        # mode is engaged, the pinned controls show forced values, so read the
        # stashed pre-mode values for those.
        values = {}
        for name in self.param_widgets:
            values[name] = self._get_value(name)
        for name, v in self._pre_mode_values.items():
            values[name] = v
        self.opts.values = values
        return self.opts

    # -- about ------------------------------------------------------------
    def _show_about(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("About BlurXTerminator")
        lay = QVBoxLayout(dlg)
        pm = render_icon_pixmap(find_icon_path(self.opts.exe), 48)
        if pm is not None:
            icon_label = QLabel()
            icon_label.setPixmap(pm)
            icon_row = QHBoxLayout()
            icon_row.addWidget(icon_label)
            icon_row.addStretch()
            lay.addLayout(icon_row)
        label = QLabel(about_html())
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setWordWrap(True)
        label.setOpenExternalLinks(True)
        lay.addWidget(label)
        row = QHBoxLayout()
        row.addStretch()
        close = QPushButton("Close")
        close.clicked.connect(dlg.accept)
        row.addWidget(close)
        lay.addLayout(row)
        dlg.exec()

    # -- license ----------------------------------------------------------
    def _show_license(self):
        """Small dialog to view license status and activate the product by passing
        the user's e-mail and key to 'rc-astro <product> --activate'."""
        dlg = QDialog(self)
        dlg.setWindowTitle(f"{PRODUCT_LABEL} — License")
        dlg.setMinimumWidth(460)
        lay = QVBoxLayout(dlg)

        status_lbl = QLabel()
        status_lbl.setWordWrap(True)
        lay.addWidget(status_lbl)

        def refresh_status():
            ok, text = run_account_command(
                self.opts.exe, ["--license"], timeout=30)
            status_lbl.setText(text or "License status unavailable.")
            licensed = ok and "licens" in text.lower()
            status_lbl.setStyleSheet(
                "color: %s;" % ("green" if licensed else "gray"))

        refresh_status()

        form = QGridLayout()
        form.addWidget(QLabel("Email:"), 0, 0)
        email_edit = QLineEdit()
        email_edit.setPlaceholderText("Account e-mail used at purchase")
        form.addWidget(email_edit, 0, 1)
        form.addWidget(QLabel("License key:"), 1, 0)
        key_edit = QLineEdit()
        key_edit.setPlaceholderText("Your RC-Astro license key")
        form.addWidget(key_edit, 1, 1)
        lay.addLayout(form)

        row = QHBoxLayout()
        row.addStretch()
        activate_btn = QPushButton("Activate")
        close_btn = QPushButton("Close")
        row.addWidget(activate_btn)
        row.addWidget(close_btn)
        lay.addLayout(row)
        close_btn.clicked.connect(dlg.accept)

        def do_activate():
            email = email_edit.text().strip()
            key = key_edit.text().strip()
            if not email or not key:
                QMessageBox.warning(
                    dlg, "License",
                    "Enter both your account e-mail and your license key.")
                return
            activate_btn.setEnabled(False)
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                ok, text = run_account_command(
                    self.opts.exe,
                    ["--activate", email, key], timeout=120)
            finally:
                QApplication.restoreOverrideCursor()
                activate_btn.setEnabled(True)
            refresh_status()
            if ok:
                QMessageBox.information(
                    dlg, "License", text or "Activation succeeded.")
                # Re-probe so the main window status and Run button reflect it.
                self._check_executable()
            else:
                QMessageBox.warning(dlg, "License", text or "Activation failed.")

        activate_btn.clicked.connect(do_activate)
        dlg.exec()

    # -- executable handling ---------------------------------------------
    def _apply_product_icon(self):
        """Show the product's bundled icon as the window icon and in the header,
        once the executable is resolved. A missing icon is non-fatal: the header
        simply stays text-only."""
        path = find_icon_path(self.opts.exe)
        if path:
            self.setWindowIcon(QIcon(path))
        pm = render_icon_pixmap(path, 28)
        if pm is not None:
            self.header_icon.setPixmap(pm)
            self.header_icon.setVisible(True)
        else:
            self.header_icon.setVisible(False)

    def _selected_mlversion(self):
        """The AI model version currently chosen in the selector (0 = latest), or
        0 when no selector is shown (a single model is installed)."""
        if self.mlversion_combo is not None:
            return int(self.mlversion_combo.currentData() or 0)
        return 0

    def _on_model_changed(self, *_):
        """React to a model change: a different AI model can expose a different
        set of parameters, so re-read the schema for the newly selected model and
        rebuild the whole parameter section (via _check_executable)."""
        self._check_executable()

    def _check_executable(self):
        exe = self.opts.exe
        # Preserve any values the user already set before rebuilding controls.
        if self.param_widgets:
            self._collect_options()
        # Read the schema for the model currently selected: its parameter set can
        # differ from other models, so the controls must be built from it.
        status, message, schema = fetch_schema(exe, self._selected_mlversion())
        self.schema = schema
        self._rebuild_params(schema)
        self._load_into_ui()
        self._apply_product_icon()
        self._exe_ok = status in ("ok", "unknown")
        # Gate Run on an active license too: rc-astro refuses to process an
        # unlicensed product. On v4 the parameter document already carries a
        # `license` object — gate deterministically on `license.valid` (no extra
        # probe). On v3 fall back to the `--license` text, blocking only on a
        # positive "not licensed" report so a licensed/trial product (or a
        # transient probe error) is never falsely blocked.
        colors = {"ok": "green", "missing": "gray", "unknown": "darkorange"}
        color = colors.get(status, "black")
        self._licensed = True
        if self._exe_ok:
            lic = schema.get("license") if isinstance(schema, dict) else None
            if isinstance(lic, dict) and "valid" in lic:
                if not lic.get("valid", True):
                    self._licensed = False
                    message = ((lic.get("message") or f"{PRODUCT_LABEL} is not "
                                "licensed.").strip()
                               + "  Click “License…” to activate.")
                    color = "crimson"
            else:
                _, lic_text = run_account_command(exe, ["--license"], timeout=15)
                if license_is_unlicensed(lic_text):
                    self._licensed = False
                    message = ((lic_text or f"{PRODUCT_LABEL} is not "
                                "licensed.").strip()
                               + "  Click “License…” to activate.")
                    color = "crimson"
        self.status_label.setText(message)
        self.status_label.setStyleSheet(f"color: {color};")
        if hasattr(self, "run_btn"):
            self.run_btn.setEnabled(self._exe_ok and self._licensed)
        return self._exe_ok and self._licensed

    # -- update check -------------------------------------------------------
    def _start_update_check(self):
        """Kick off a background 'rc-astro update --json' check as the GUI
        loads. Runs off the UI thread (UpdateCheckWorker) so a slow or
        unreachable update server never delays showing the window; the popup
        (if any) appears once the check completes."""
        self._update_check_worker = UpdateCheckWorker(self.opts.exe)
        self._update_check_worker.result.connect(self._on_update_check_result)
        self._update_check_worker.start()

    def _on_update_check_result(self, info):
        if not info:
            return
        new = info.get("new") or "?"
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle("Update available")
        box.setText(f"A newer RC-Astro version ({new}) is available.")
        update_btn = box.addButton("Update Now", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Later", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() is update_btn:
            self._run_update()

    def _run_update(self):
        """Run 'rc-astro update --install' in the background; installation
        can take a while, so keep the UI responsive and report the outcome
        when it completes."""
        self.status_label.setText("Updating RC-Astro\u2026")
        self.status_label.setStyleSheet("color: darkorange;")
        if hasattr(self, "run_btn"):
            self.run_btn.setEnabled(False)
        self._update_install_worker = UpdateInstallWorker(self.opts.exe)
        self._update_install_worker.finished_ok.connect(self._on_update_installed)
        self._update_install_worker.start()

    def _on_update_installed(self, ok, text):
        if hasattr(self, "run_btn"):
            self.run_btn.setEnabled(True)
        if ok:
            QMessageBox.information(
                self, "Update", "RC-Astro was updated successfully.")
            # Re-probe so the schema/params reflect the newly installed build.
            self._check_executable()
        else:
            QMessageBox.warning(
                self, "Update", text or "The update could not be installed.")
            self._check_executable()

    # -- run --------------------------------------------------------------
    def _on_run(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.run_btn.setEnabled(False)
            return

        opts = self._collect_options()
        if not self._check_executable():
            return
        # _check_executable rebuilds controls; re-collect to keep values.
        opts = self._collect_options()
        sequence_mode = self.target_seq.isChecked()
        if sequence_mode and not self.has_sequence:
            QMessageBox.warning(self, "No sequence", "No sequence is loaded.")
            return
        if not sequence_mode and not self.has_image:
            QMessageBox.warning(self, "No image", "No image is loaded.")
            return

        save_options(self.siril, opts)
        self.progress_bar.setRange(0, 0)  # indeterminate while running
        self.run_btn.setText("Cancel")

        self.worker = BxtWorker(self.siril, opts, self.schema, sequence_mode)
        self.worker.progress.connect(self._on_progress)
        self.worker.message.connect(self._on_message)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.start()

    def _on_progress(self, frac):
        if self.progress_bar.maximum() == 0:
            self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(int(max(0.0, min(1.0, frac)) * 100))
        try:
            self.siril.update_progress("BlurXTerminator", max(0.0, min(1.0, frac)))
        except Exception:
            pass

    def _on_message(self, text):
        self.log_view.appendPlainText(text)

    def _on_finished(self, ok, message):
        self.run_btn.setText("Run BlurXTerminator")
        self.run_btn.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self._on_message(message)
        try:
            self.siril.reset_progress()
        except Exception:
            pass
        if ok:
            self.progress_bar.setValue(100)
        else:
            self.progress_bar.setValue(0)
            QMessageBox.warning(self, "BlurXTerminator", message)

    def closeEvent(self, event):
        try:
            self._collect_options()
            save_options(self.siril, self.opts)
        except Exception:
            pass
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(5000)
        for w in (self._update_check_worker, self._update_install_worker):
            if w is not None and w.isRunning():
                w.wait(5000)
        try:
            self.siril.disconnect()
        except Exception:
            pass
        event.accept()


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def build_schema_parser(schema):
    """Build an argument parser whose options are generated from the live schema:
    one option per parameter (using its verbatim `flag`), the modes as a mutually
    exclusive group, plus the host extras (--exe, --ml-version, --sequence)."""
    p = argparse.ArgumentParser(
        prog="BlurXTerminator.py",
        description="Deconvolve/sharpen the loaded image or sequence using BlurXTerminator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--exe", help="Path to the rc-astro executable. Optional if "
                   "'rc-astro' is on PATH; otherwise defaults to the saved setting.")
    p.add_argument("--ml-version", type=int, dest="mlversion",
                   help="Model version (0 = latest).")
    p.add_argument("--device", "--engine", dest="engine",
                   help="Compute device: 'auto' (default), 'cpu', or a GPU "
                        "('gpu'/'gpu0'/'gpu1'). Use 'cpu' on low-memory GPUs. "
                        "(--engine is a deprecated alias for v3 builds.)")
    p.add_argument("--sequence", action="store_true",
                   help="Process the loaded FITS sequence instead of the loaded image.")

    for name, pp in schema_params(schema).items():
        flag = pp.get("flag")
        if not flag or pp.get("guiOnly"):
            continue
        ptype = pp.get("type", "float")
        helptext = pp.get("description", "")
        if ptype == "bool":
            # --flag / --no-flag, leaving it unset (None) unless given.
            p.add_argument(flag, dest=name, default=None,
                           action=argparse.BooleanOptionalAction, help=helptext)
        elif ptype == "enum":
            choices = [o.get("value") if isinstance(o, dict) else o
                       for o in pp.get("options", [])]
            p.add_argument(flag, dest=name, choices=choices, default=None, help=helptext)
        elif ptype == "int":
            p.add_argument(flag, dest=name, type=int, default=None, help=helptext)
        else:
            p.add_argument(flag, dest=name, type=float, default=None, help=helptext)

    modes = schema.get("modes") or []
    if modes:
        grp = p.add_mutually_exclusive_group()
        for m in modes:
            grp.add_argument(m["flag"], dest="_mode", action="store_const",
                             const=m["name"], default=None,
                             help=m.get("description", ""))
    return p


def run_cli(siril, argv):
    # First pass: locate the executable and the requested model so the live
    # schema (which can differ per model) can be fetched for it.
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--exe")
    base.add_argument("--ml-version", type=int, dest="mlversion")
    base_args, _ = base.parse_known_args(argv)

    opts = load_options(siril)
    if base_args.exe:
        opts.exe = base_args.exe
    mlversion = (base_args.mlversion if base_args.mlversion is not None
                 else opts.mlversion)

    status, message, schema = fetch_schema(opts.exe, mlversion)
    siril.log(f"BlurXTerminator: {message}",
              LogColor.GREEN if status == "ok" else LogColor.SALMON)
    if status == "missing":
        return 1

    # Second pass: full parser built from the live schema.
    args = build_schema_parser(schema).parse_args(argv)
    if args.mlversion is not None:
        opts.mlversion = args.mlversion
    if getattr(args, "engine", None):
        opts.engine = args.engine

    # Start from schema defaults, layer saved values, then any CLI overrides.
    values = {name: pp.get("default") for name, pp in schema_params(schema).items()}
    values.update({k: v for k, v in opts.values.items() if k in values})
    for name in schema_params(schema):
        v = getattr(args, name, None)
        if v is not None:
            values[name] = v
    opts.values = values
    if getattr(args, "_mode", None):
        opts.mode = args._mode

    def progress_cb(frac):
        try:
            siril.update_progress("BlurXTerminator", max(0.0, min(1.0, frac)))
        except Exception:
            pass

    def log_cb(text):
        siril.log(f"BlurXTerminator: {text}", LogColor.DEFAULT)

    try:
        if args.sequence:
            if not siril.is_sequence_loaded():
                siril.log("BlurXTerminator: no sequence is loaded.", LogColor.RED)
                return 1
            ok = process_sequence(siril, opts, schema, progress_cb=progress_cb,
                                  log_cb=log_cb)
        else:
            if not siril.is_image_loaded():
                siril.log("BlurXTerminator: no image is loaded.", LogColor.RED)
                return 1
            ok = process_single_image(siril, opts, schema, progress_cb=progress_cb,
                                      log_cb=log_cb)
    except BxtError as exc:
        siril.log(f"BlurXTerminator: {exc}", LogColor.RED)
        return 1
    finally:
        try:
            siril.reset_progress()
        except Exception:
            pass
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    cli_argv = [a for a in sys.argv[1:] if a not in ("--", )]

    siril = s.SirilInterface()
    try:
        siril.connect()
    except s.SirilConnectionError:
        print("BlurXTerminator: could not connect to Siril.", file=sys.stderr)
        sys.exit(1)

    try:
        siril.cmd("requires", "1.4.0")
    except Exception as exc:
        siril.log(f"BlurXTerminator: {exc}", LogColor.RED)
        siril.disconnect()
        sys.exit(1)

    if cli_argv:
        try:
            rc = run_cli(siril, cli_argv)
        finally:
            siril.disconnect()
        sys.exit(rc)

    app = QApplication.instance() or QApplication(sys.argv)
    gui = BxtGUI(siril)
    gui.show()
    app.exec()


if __name__ == "__main__":
    main()
