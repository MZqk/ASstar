"""
SPDX-License-Identifier: MIT
Author: Paweł Tomak (c) 2026 pawel <at> tomak <dot> eu

This script provides a GUI for fetching and classifying astrophotography frames
from a local directory or an AsiAir device over SMB.  It supports FITS and
DSLR RAW files, extracting metadata and allowing users to preview images and
copy them into organised subdirectories.
"""

# Version history
# 1.0.0 Initial release

import os
import sys
import re
import io
import shutil
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Protocol, Tuple

try:
    import sirilpy as s

    s.ensure_installed("PySide6")
    s.ensure_installed("numpy")
    s.ensure_installed("exifread")
    s.ensure_installed("smbprotocol")
except ImportError:
    pass

import numpy as np  # noqa: E402

from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QFileDialog,
    QMessageBox,
    QGroupBox,
    QRadioButton,
    QButtonGroup,
    QTreeWidget,
    QTreeWidgetItem,
    QHeaderView,
    QAbstractItemView,
    QDialog,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QTextBrowser,
)
from PySide6.QtCore import Qt, QEvent  # noqa: E402
from PySide6.QtGui import QFont, QImage, QPixmap  # noqa: E402

import exifread  # noqa: E402


VERSION = "1.0.0"

# Default SMB share name on AsiAir devices
ASIAIR_SHARE = "EMMC Images"
ASIAIR_AUTORUN = "Autorun"
ASIAIR_SMB_PORT = 445

# File extensions we consider as astro frames (stored lowercase;
# the check function normalises before lookup).
FITS_EXTENSIONS = {".fit", ".fits", ".fts"}

DSLR_RAW_EXTENSIONS = {
    # Canon
    ".cr2",
    ".cr3",
    ".crw",
    # Nikon
    ".nef",
    ".nrw",
    # Sony
    ".arw",
    ".srf",
    ".sr2",
    # Pentax
    ".pef",
    # Olympus / OM System
    ".orf",
    # Panasonic / Lumix
    ".rw2",
    # Fujifilm
    ".raf",
    # Samsung
    ".srw",
    # Sigma
    ".x3f",
    # Leica
    ".rwl",
    # Hasselblad
    ".3fr",
    ".fff",
    # Phase One
    ".iiq",
    # Adobe / generic
    ".dng",
    ".raw",
}

SUPPORTED_EXTENSIONS = FITS_EXTENSIONS | DSLR_RAW_EXTENSIONS

# Hostname / FQDN / IPv4 pattern
_HOST_RE = re.compile(r"^(?!\.)[A-Za-z0-9._-]+(?<!\.)$")


# ── Data model ─────────────────────────────────────────────────────────────────────


class FrameType(Enum):
    """Possible calibration-frame types."""

    LIGHTS = "Lights"
    DARKS = "Darks"
    FLATS = "Flats"
    BIAS = "Bias"


@dataclass
class FileEntry:
    """One file discovered on a source."""

    path: str  # relative path inside the source root
    frame_type: FrameType = FrameType.LIGHTS
    exposure: Optional[str] = field(default=None)
    iso: Optional[str] = field(default=None)
    date: Optional[str] = field(default=None)

    @property
    def filename(self) -> str:
        # Normalise backslashes (SMB/UNC paths) so basename works on Linux.
        return os.path.basename(self.path.replace("\\", "/"))


def is_supported_file(name: str) -> bool:
    """Return True if *name* ends with a recognised astro-image extension."""
    _, ext = os.path.splitext(name)
    return ext.lower() in SUPPORTED_EXTENSIONS


def is_valid_host(text: str) -> bool:
    """Return True when *text* looks like a valid IPv4 address, hostname, or FQDN."""
    return bool(_HOST_RE.match(text.strip()))


# ── AsiAir directory → frame-type mapping ────────────────────────────────────

# AsiAir stores images in a tree such as:
#   <root>/Light/<target>/...
#   <root>/Dark/...
#   <root>/Flat/...
#   <root>/Bias/...
# We map directory names (case-insensitive first component) to FrameType.

_ASIAIR_DIR_MAP = {
    "light": FrameType.LIGHTS,
    "lights": FrameType.LIGHTS,
    "dark": FrameType.DARKS,
    "darks": FrameType.DARKS,
    "flat": FrameType.FLATS,
    "flats": FrameType.FLATS,
    "bias": FrameType.BIAS,
    "biases": FrameType.BIAS,
}


def classify_asiair_path(relative_path: str) -> FrameType:
    """Determine the frame type from the *relative_path* inside an AsiAir
    share.  The first meaningful path component is used.

    >>> classify_asiair_path("Light/M31/img_001.fit")
    <FrameType.LIGHTS: 'Lights'>
    >>> classify_asiair_path("Dark/img_001.fit")
    <FrameType.DARKS: 'Darks'>
    """
    parts = relative_path.replace("\\", "/").strip("/").split("/")
    for part in parts:
        ft = _ASIAIR_DIR_MAP.get(part.lower())
        if ft is not None:
            return ft
    return FrameType.LIGHTS  # fallback


# Map FrameType → destination subdirectory name
FRAME_TYPE_DIR_MAP = {
    FrameType.LIGHTS: "lights",
    FrameType.DARKS: "darks",
    FrameType.FLATS: "flats",
    FrameType.BIAS: "biases",
}


# ── Metadata extraction ──────────────────────────────────────────────────────

# FITS keywords that carry exposure time (checked in order of preference)
_FITS_EXPOSURE_KEYS = ("EXPTIME", "EXPOSURE")
# FITS keywords that carry ISO / gain
_FITS_ISO_KEYS = ("ISOSPEED", "ISO", "GAIN")
# FITS keywords that carry the observation date
_FITS_DATE_KEYS = ("DATE-OBS", "DATE")


def _parse_fits_header(file_obj) -> dict[str, str]:
    """Read the primary FITS header from a binary file-like object.

    Returns a dict of keyword → value (both strings, values stripped).
    Only the primary HDU header is parsed (up to the first END card).
    """
    header: dict[str, str] = {}
    BLOCK = 2880
    while True:
        block = file_obj.read(BLOCK)
        if not block:
            break
        # Pad to full block if we got a short read
        block = block.ljust(BLOCK, b" ")
        for i in range(0, BLOCK, 80):
            card = block[i : i + 80].decode("ascii", errors="replace")
            if card.startswith("END"):
                return header
            if len(card) < 10 or card[8] != "=":
                continue
            keyword = card[:8].strip()
            raw_value = card[10:].split("/")[0].strip()
            # Strip surrounding quotes from string values
            if raw_value.startswith("'") and raw_value.endswith("'"):
                raw_value = raw_value[1:-1].strip()
            header[keyword] = raw_value
    return header


def _format_exposure(value: str) -> str:
    """Format an exposure-time value for display."""
    try:
        secs = float(value)
        if secs >= 1.0:
            return f"{secs:.1f}s"
        return f"1/{int(round(1 / secs))}s"
    except (ValueError, ZeroDivisionError):
        return value


def _format_date(value: str) -> str:
    """Normalise a date string for display.

    Handles FITS ISO-8601 (``2024-01-15T23:45:30.123``) and EXIF
    (``2024:01:15 23:45:30``) formats, returning
    ``YYYY-MM-DD HH:MM:SS``.  Fractional seconds are discarded.
    """
    # EXIF format uses colons in the date part
    normalised = value.replace("T", " ").strip()
    parts = normalised.split(" ", 1)
    if parts:
        parts[0] = parts[0].replace(":", "-", 2)
    # Strip fractional seconds (e.g. ".123")
    if len(parts) > 1:
        parts[1] = parts[1].split(".")[0]
    return " ".join(parts)


def read_fits_metadata(
    file_obj,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract (exposure, iso, date) from a FITS file-like object."""
    header = _parse_fits_header(file_obj)
    exposure = None
    for key in _FITS_EXPOSURE_KEYS:
        if key in header:
            exposure = _format_exposure(header[key])
            break
    iso = None
    for key in _FITS_ISO_KEYS:
        if key in header:
            iso = header[key]
            break
    date = None
    for key in _FITS_DATE_KEYS:
        if key in header:
            date = _format_date(header[key])
            break
    return exposure, iso, date


def read_raw_metadata(
    file_obj,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract (exposure, iso, date) from a DSLR RAW file-like object via EXIF."""
    tags = exifread.process_file(file_obj, details=False, stop_tag="DateTimeOriginal")
    exposure = None
    iso = None
    date = None
    if "EXIF ExposureTime" in tags:
        val = tags["EXIF ExposureTime"]
        try:
            # exifread returns Ratio objects; convert to float
            ratio = val.values[0]
            secs = float(ratio.num) / float(ratio.den)
            exposure = _format_exposure(str(secs))
        except (AttributeError, IndexError, ZeroDivisionError):
            exposure = str(val)
    if "EXIF ISOSpeedRatings" in tags:
        iso = str(tags["EXIF ISOSpeedRatings"])
    if "EXIF DateTimeOriginal" in tags:
        date = _format_date(str(tags["EXIF DateTimeOriginal"]))
    return exposure, iso, date


def read_file_metadata(
    file_obj, filename: str
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Dispatch to the correct metadata reader based on *filename* extension."""
    _, ext = os.path.splitext(filename)
    if ext.lower() in FITS_EXTENSIONS:
        return read_fits_metadata(file_obj)
    if ext.lower() in DSLR_RAW_EXTENSIONS:
        return read_raw_metadata(file_obj)
    return None, None, None


# ── Source providers ─────────────────────────────────────────────────────────


class FileSource(Protocol):
    """Abstract source of files (local dir or SMB)."""

    def list_files(self) -> List[FileEntry]: ...  # pragma: no cover


class LocalSource:
    """List FITS files from a local directory tree."""

    def __init__(self, directory: str):
        self.directory = directory

    def list_files(self) -> List[FileEntry]:
        entries: List[FileEntry] = []
        for root, _dirs, files in os.walk(self.directory):
            for fname in sorted(files):
                if is_supported_file(fname):
                    rel = os.path.relpath(os.path.join(root, fname), self.directory)
                    ftype = classify_asiair_path(rel)
                    entries.append(FileEntry(path=rel, frame_type=ftype))
        return entries


class SmbSource:
    """List FITS files from an AsiAir device over SMB.

    Uses *smbclient* from the ``smbprotocol`` package.
    """

    def __init__(
        self,
        server: str,
        share: str = ASIAIR_SHARE,
        username: str = "guest",
        password: str = "",
        port: int = ASIAIR_SMB_PORT,
    ):
        self.server = server
        self.share = share
        self.username = username
        self.password = password
        self.port = port

    # -- internal helpers (thin wrappers so we can mock easily) ---------------

    @staticmethod
    def _smb_register(server: str, username: str, password: str, port: int):
        """Register the SMB session.  Thin wrapper for testing."""
        import smbclient  # type: ignore[import-untyped]
        from smbclient import ClientConfig  # type: ignore[import-untyped]

        # AsiAir exposes an open (guest) share; disable signing and
        # secure-negotiate requirements so the guest session works.
        ClientConfig().require_secure_negotiate = False
        smbclient.register_session(
            server,
            username=username,
            password=password,
            port=port,
            require_signing=False,
        )

    @staticmethod
    def _smb_walk(unc_root: str):
        """Walk the SMB tree.  Thin wrapper for testing."""
        import smbclient  # type: ignore[import-untyped]

        yield from smbclient.walk(unc_root)

    # -- public API -----------------------------------------------------------

    def list_files(self) -> List[FileEntry]:
        self._smb_register(self.server, self.username, self.password, self.port)
        unc_root = f"\\\\{self.server}\\{self.share}\\{ASIAIR_AUTORUN}"
        entries: List[FileEntry] = []
        for dirpath, _dirnames, filenames in self._smb_walk(unc_root):
            for fname in sorted(filenames):
                if is_supported_file(fname):
                    full = f"{dirpath}\\{fname}"
                    rel = full[len(unc_root) :].lstrip("\\")
                    ftype = classify_asiair_path(rel)
                    entries.append(FileEntry(path=rel, frame_type=ftype))
        return entries


# ── Copy-to-directory dialog ─────────────────────────────────────────────────────────


class CopyDialog(QDialog):
    """Modal dialog for choosing a target directory and copying files."""

    BLUE_BTN_STYLE = (
        "QPushButton {"
        "  background-color: #2979FF; color: white;"
        "  padding: 6px 16px; border-radius: 4px; font-weight: bold; border: none;"
        "}"
        "QPushButton:hover { background-color: #2962FF; }"
        "QPushButton:pressed { background-color: #1E56CC; }"
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Copy Files to Directory")
        self.setFixedSize(500, 130)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # ── destination row
        dest_row = QHBoxLayout()
        dest_row.addWidget(QLabel("Destination:"))

        self._dest_entry = QLineEdit()
        self._dest_entry.setPlaceholderText("/path/to/target")
        dest_row.addWidget(self._dest_entry)

        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_dest)
        dest_row.addWidget(browse_btn)
        layout.addLayout(dest_row)

        layout.addSpacing(10)

        # ── button row
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        copy_btn = QPushButton("Copy Files")
        copy_btn.setStyleSheet(self.BLUE_BTN_STYLE)
        copy_btn.clicked.connect(self.accept)
        btn_row.addWidget(copy_btn)
        layout.addLayout(btn_row)

    def _browse_dest(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Select Target Directory", os.path.expanduser("~")
        )
        if directory:
            self._dest_entry.setText(directory)

    def destination(self) -> str:
        return self._dest_entry.text().strip()


# ── Image preview ────────────────────────────────────────────────────────────

# FITS BITPIX → numpy dtype (FITS is big-endian)
_BITPIX_DTYPE = {
    8: ">u1",
    16: ">i2",
    32: ">i4",
    -32: ">f4",
    -64: ">f8",
}


def _auto_stretch(data: np.ndarray) -> np.ndarray:
    """Percentile-based auto-stretch to 0-255 uint8."""
    lo = np.percentile(data, 1)
    hi = np.percentile(data, 99)
    if hi <= lo:
        hi = lo + 1
    stretched = (data.astype(np.float64) - lo) / (hi - lo)
    return np.clip(stretched * 255, 0, 255).astype(np.uint8)


def render_fits_preview(file_obj) -> Optional[QPixmap]:
    """Read FITS pixel data and return a QPixmap preview, or *None*."""
    header = _parse_fits_header(file_obj)
    bitpix = int(header.get("BITPIX", 0))
    naxis = int(header.get("NAXIS", 0))
    if bitpix == 0 or naxis < 2:
        return None
    naxis1 = int(header.get("NAXIS1", 0))  # width
    naxis2 = int(header.get("NAXIS2", 0))  # height
    naxis3 = int(header.get("NAXIS3", 1))  # colour planes (1 = mono)
    bzero = float(header.get("BZERO", 0.0))
    bscale = float(header.get("BSCALE", 1.0))

    dtype_str = _BITPIX_DTYPE.get(bitpix)
    if dtype_str is None or naxis1 == 0 or naxis2 == 0:
        return None

    # Seek to start of data (header blocks are multiples of 2880)
    header_bytes = file_obj.tell()
    # _parse_fits_header reads full 2880-byte blocks, so tell() is
    # already block-aligned; but round up to be safe.
    remainder = header_bytes % 2880
    if remainder:
        file_obj.read(2880 - remainder)

    npixels = naxis1 * naxis2 * naxis3
    dtype = np.dtype(dtype_str)
    raw = file_obj.read(npixels * dtype.itemsize)
    if len(raw) < npixels * dtype.itemsize:
        return None

    pixels = np.frombuffer(raw, dtype=dtype).astype(np.float64)
    pixels = pixels * bscale + bzero

    if naxis3 >= 3:
        # Colour FITS: shape is (3, H, W) – transpose to (H, W, 3)
        pixels = pixels.reshape(naxis3, naxis2, naxis1)
        rgb = np.stack(
            [_auto_stretch(pixels[c]) for c in range(3)], axis=-1
        )
        h, w = naxis2, naxis1
        img = QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888)
    else:
        pixels = pixels.reshape(naxis2, naxis1)
        grey = _auto_stretch(pixels)
        h, w = naxis2, naxis1
        img = QImage(
            grey.tobytes(), w, h, w, QImage.Format.Format_Grayscale8
        )

    return QPixmap.fromImage(img)


def render_raw_preview(file_obj) -> Optional[QPixmap]:
    """Extract the embedded JPEG thumbnail from a RAW file and return
    a QPixmap, or *None* if no thumbnail is present."""
    tags = exifread.process_file(file_obj, details=False)
    thumb = tags.get("JPEGThumbnail")
    if not thumb:
        return None
    pm = QPixmap()
    if isinstance(thumb, bytes):
        pm.loadFromData(thumb)
    else:
        pm.loadFromData(bytes(thumb))
    if pm.isNull():
        return None
    return pm


def render_preview(file_obj, filename: str) -> Optional[QPixmap]:
    """Return a QPixmap preview for the given file, or *None*."""
    _, ext = os.path.splitext(filename)
    if ext.lower() in FITS_EXTENSIONS:
        return render_fits_preview(file_obj)
    if ext.lower() in DSLR_RAW_EXTENSIONS:
        return render_raw_preview(file_obj)
    return None


class PreviewDialog(QDialog):
    """Modal dialog that shows an image preview."""

    MAX_W = 900
    MAX_H = 700

    def __init__(self, pixmap: QPixmap, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Preview – {title}")
        self._build_ui(pixmap)

    def _build_ui(self, pixmap: QPixmap):
        layout = QVBoxLayout(self)

        # Scale pixmap to fit within MAX dimensions, keeping aspect ratio
        scaled = pixmap.scaled(
            self.MAX_W,
            self.MAX_H,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        img_label = QLabel()
        img_label.setPixmap(scaled)
        img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        scroll = QScrollArea()
        scroll.setWidget(img_label)
        scroll.setWidgetResizable(False)
        scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(scroll)

        # ── info + close row
        info_label = QLabel(
            f"{pixmap.width()} × {pixmap.height()} px"
        )
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info_label)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        # Size the dialog to the scaled image + some padding
        self.resize(
            min(scaled.width() + 40, self.MAX_W + 40),
            min(scaled.height() + 100, self.MAX_H + 100),
        )


# ── About dialog ─────────────────────────────────────────────────────────────

_ABOUT_TEXT = """\
<h3>About</h3>
<p><b>AstroT3kFetch</b> is a utility for fetching, classifying, and
organising astrophotography frames captured with AsiAir or stored
locally.</p>

<h4>How to use</h4>
<ol>
  <li>Select a <b>source</b> &ndash; a local directory or an AsiAir
      device on the network.</li>
  <li>Click <b>Load</b> (or <b>Connect</b> for SMB) to scan for
      supported image files (FITS and DSLR RAW).</li>
  <li>Review the file tree. Frame types are auto-detected from
      directory names. Change them with the buttons below the tree
      or with keyboard shortcuts.</li>
  <li>Optionally click <b>Analyse</b> to read exposure, ISO/gain,
      and date metadata from the files.</li>
  <li>Double-click a file to preview its image.</li>
  <li>Click <b>Copy Files</b> to copy the frames into organised
      sub-directories (lights, darks, flats, biases).</li>
</ol>

<h4>Links</h4>
<ul>
  <li><b>Source code:</b>
      <a href="https://gitlab.com/grodzik/astro-t3k">
      gitlab.com/grodzik/astro-t3k</a></li>
  <li><b>Report bugs:</b>
      <a href="https://gitlab.com/grodzik/astro-t3k/-/issues">
      gitlab.com/grodzik/astro-t3k/-/issues</a></li>
</ul>
"""

_KEYS_TEXT = """\
<h3>Key Mappings</h3>
<table cellpadding="4">
  <tr><td><b>F</b></td><td>Set selected files as <i>Flats</i></td></tr>
  <tr><td><b>D</b></td><td>Set selected files as <i>Darks</i></td></tr>
  <tr><td><b>B</b></td><td>Set selected files as <i>Bias</i></td></tr>
  <tr><td><b>L</b></td><td>Set selected files as <i>Lights</i></td></tr>
  <tr><td><b>Delete</b></td>
      <td>Delete selected files / directories from source
          (with confirmation)</td></tr>
  <tr><td><b>Double&#8209;click</b></td>
      <td>Preview the image</td></tr>
</table>
"""

_LICENCE_TEXT = """\
<h3>Licence</h3>
<p>MIT Licence</p>
<p>Copyright &copy; 2026 Pawe\u0142 Tomak</p>
<p>Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation files
(the &ldquo;Software&rdquo;), to deal in the Software without
restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished
to do so, subject to the following conditions:</p>
<p>The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.</p>
<p>THE SOFTWARE IS PROVIDED &ldquo;AS IS&rdquo;, WITHOUT WARRANTY OF
ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.</p>
"""


class AboutDialog(QDialog):
    """Modal About / Help dialog with tabbed content."""

    _ACTIVE_BTN = (
        "QPushButton { font-weight: bold; border-bottom: 2px solid #2979FF; }"
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About AstroT3kFetch")
        self.setFixedSize(520, 430)
        self._build_ui()
        # Show the About tab by default
        self._show_about()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # ── header
        title = QLabel("AstroT3kFetch")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(14)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        author = QLabel("Pawe\u0142 Tomak\npawel@tomak.eu")
        author.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(author)

        layout.addSpacing(8)

        # ── tab buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self._btn_about = QPushButton("About / Help")
        self._btn_about.setFlat(True)
        self._btn_about.clicked.connect(self._show_about)
        btn_row.addWidget(self._btn_about)

        self._btn_keys = QPushButton("Key Mappings")
        self._btn_keys.setFlat(True)
        self._btn_keys.clicked.connect(self._show_keys)
        btn_row.addWidget(self._btn_keys)

        self._btn_licence = QPushButton("Licence")
        self._btn_licence.setFlat(True)
        self._btn_licence.clicked.connect(self._show_licence)
        btn_row.addWidget(self._btn_licence)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        # ── content area
        self._content = QTextBrowser()
        self._content.setOpenExternalLinks(True)
        self._content.setReadOnly(True)
        layout.addWidget(self._content)

        # ── close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignCenter)

    # ── tab switching helpers ────────────────────────────────────────────────

    def _highlight(self, active: QPushButton):
        """Apply the active style to *active* and reset the others."""
        for btn in (self._btn_about, self._btn_keys, self._btn_licence):
            btn.setStyleSheet(self._ACTIVE_BTN if btn is active else "")

    def _show_about(self):
        self._content.setHtml(_ABOUT_TEXT)
        self._highlight(self._btn_about)

    def _show_keys(self):
        self._content.setHtml(_KEYS_TEXT)
        self._highlight(self._btn_keys)

    def _show_licence(self):
        self._content.setHtml(_LICENCE_TEXT)
        self._highlight(self._btn_licence)


# ── Qt interface ─────────────────────────────────────────────────────────────


class Interface(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"AstroT3kFetch - v{VERSION}")
        self.setFixedSize(1200, 800)
        self._entries: List[FileEntry] = []
        self.create_widgets()

    # ── widget construction ──────────────────────────────────────────────────

    def create_widgets(self):
        """Build the entire UI."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # Title
        title = QLabel("AstroT3kFetch")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)
        main_layout.addSpacing(10)

        # ── Source selection ─────────────────────────────────────────────────
        source_group = QGroupBox("Source")
        source_layout = QVBoxLayout(source_group)

        # Radio buttons
        radio_row = QWidget()
        radio_layout = QHBoxLayout(radio_row)
        radio_layout.setContentsMargins(0, 0, 0, 0)

        self._radio_group = QButtonGroup(self)
        self._radio_local = QRadioButton("Local directory")
        self._radio_smb = QRadioButton("AsiAir (SMB)")
        self._radio_local.setChecked(True)
        self._radio_group.addButton(self._radio_local, 0)
        self._radio_group.addButton(self._radio_smb, 1)
        radio_layout.addWidget(self._radio_local)
        radio_layout.addWidget(self._radio_smb)
        radio_layout.addStretch()
        source_layout.addWidget(radio_row)

        # Path / IP entry
        entry_row = QWidget()
        entry_layout = QHBoxLayout(entry_row)
        entry_layout.setContentsMargins(0, 0, 0, 0)

        self._source_label = QLabel("Path:")
        entry_layout.addWidget(self._source_label)

        self._source_entry = QLineEdit()
        self._source_entry.setPlaceholderText("/path/to/frames")
        entry_layout.addWidget(self._source_entry)

        self._browse_btn = QPushButton("Browse")
        self._browse_btn.clicked.connect(self._browse_directory)
        entry_layout.addWidget(self._browse_btn)

        self._connect_btn = QPushButton("Load")
        self._connect_btn.clicked.connect(self._load_source)
        entry_layout.addWidget(self._connect_btn)

        source_layout.addWidget(entry_row)
        main_layout.addWidget(source_group)

        # Update labels when source type changes
        self._radio_group.idToggled.connect(self._on_source_type_changed)

        # ── File tree ────────────────────────────────────────────────────────
        table_group = QGroupBox("Files")
        table_layout = QVBoxLayout(table_group)

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["File", "Type", "Exposure", "ISO", "Date"])
        self._tree.header().setStretchLastSection(True)
        self._tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._tree.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        self._tree.installEventFilter(self)
        table_layout.addWidget(self._tree)

        # Map from entry index → QTreeWidgetItem (files only)
        self._file_items: list[QTreeWidgetItem] = []

        # Type-assignment buttons
        type_btn_row = QWidget()
        type_btn_layout = QHBoxLayout(type_btn_row)
        type_btn_layout.setContentsMargins(0, 0, 0, 0)
        type_btn_layout.addWidget(QLabel("Set selected as:"))

        for ft in FrameType:
            btn = QPushButton(ft.value)
            btn.clicked.connect(lambda checked=False, t=ft: self._set_selected_type(t))
            type_btn_layout.addWidget(btn)

        type_btn_layout.addStretch()
        table_layout.addWidget(type_btn_row)
        main_layout.addWidget(table_group)

        # ── Progress bar ─────────────────────────────────────────────────────
        progress_row = QWidget()
        progress_layout = QHBoxLayout(progress_row)
        progress_layout.setContentsMargins(0, 0, 0, 0)

        self._progress_label = QLabel("")
        progress_layout.addWidget(self._progress_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setMinimum(0)
        self._progress_bar.setValue(0)
        progress_layout.addWidget(self._progress_bar)

        progress_row.setVisible(False)
        self._progress_row = progress_row
        main_layout.addWidget(progress_row)

        # ── Bottom buttons ───────────────────────────────────────────────
        bottom = QWidget()
        bottom_layout = QHBoxLayout(bottom)

        about_btn = QPushButton("About")
        about_btn.clicked.connect(self._show_about_dialog)
        bottom_layout.addWidget(about_btn)

        bottom_layout.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        bottom_layout.addWidget(close_btn)

        bottom_layout.addStretch()

        self._analyse_btn = QPushButton("Analyse")
        self._analyse_btn.setEnabled(False)
        self._analyse_btn.clicked.connect(self._analyse_files)
        bottom_layout.addWidget(self._analyse_btn)

        self._copy_btn = QPushButton("Copy Files")
        self._copy_btn.setStyleSheet(CopyDialog.BLUE_BTN_STYLE)
        self._copy_btn.clicked.connect(self._show_copy_dialog)
        bottom_layout.addWidget(self._copy_btn)

        main_layout.addWidget(bottom)

    # ── slots ────────────────────────────────────────────────────────────────

    def _on_source_type_changed(self, button_id: int, checked: bool):
        """React to the user toggling between Local and SMB."""
        if not checked:
            return
        if button_id == 0:  # local
            self._source_label.setText("Path:")
            self._source_entry.setPlaceholderText("/path/to/frames")
            self._browse_btn.setVisible(True)
            self._connect_btn.setText("Load")
        else:  # SMB
            self._source_label.setText("Host:")
            self._source_entry.setPlaceholderText("192.168.1.120 or hostname")
            self._browse_btn.setVisible(False)
            self._connect_btn.setText("Connect")

    def _show_about_dialog(self):
        """Open the About / Help dialog."""
        dlg = AboutDialog(parent=self)
        dlg.exec()

    def _browse_directory(self):
        """Open a directory picker for local source."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory", os.path.expanduser("~")
        )
        if directory:
            self._source_entry.setText(directory)

    def _load_source(self):
        """Load files from the selected source and populate the table."""
        text = self._source_entry.text().strip()
        if not text:
            QMessageBox.warning(self, "Warning", "Please enter a path or host.")
            return

        try:
            if self._radio_local.isChecked():
                source: FileSource = LocalSource(text)
            else:
                if not is_valid_host(text):
                    QMessageBox.warning(
                        self, "Warning", "Please enter a valid IP address or hostname."
                    )
                    return
                source = SmbSource(text)
            self._entries = source.list_files()
            self._populate_tree()
            self._analyse_btn.setEnabled(bool(self._entries))
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))

    def _populate_tree(self):
        """Fill the tree widget from *self._entries*, grouping by directory."""
        self._tree.clear()
        self._file_items = []
        dir_items: dict[tuple[str, ...], QTreeWidgetItem] = {}

        for idx, entry in enumerate(self._entries):
            parts = entry.path.replace("\\", "/").split("/")
            filename = parts[-1]
            dir_parts = parts[:-1]

            # Ensure all ancestor directory items exist
            parent = self._tree.invisibleRootItem()
            for depth, dir_name in enumerate(dir_parts):
                key = tuple(dir_parts[: depth + 1])
                if key not in dir_items:
                    dir_item = QTreeWidgetItem(parent)
                    dir_item.setText(0, f"\U0001f4c1 {dir_name}")
                    dir_item.setData(
                        0, Qt.ItemDataRole.UserRole, None
                    )  # mark as directory
                    bold_font = QFont()
                    bold_font.setBold(True)
                    dir_item.setFont(0, bold_font)
                    dir_item.setExpanded(True)
                    dir_items[key] = dir_item
                parent = dir_items[key]

            # Create file leaf item
            file_item = QTreeWidgetItem(parent)
            file_item.setText(0, filename)
            file_item.setText(1, entry.frame_type.value)
            file_item.setText(2, entry.exposure or "")
            file_item.setText(3, entry.iso or "")
            file_item.setText(4, entry.date or "")
            file_item.setData(0, Qt.ItemDataRole.UserRole, idx)
            self._file_items.append(file_item)

    def _set_selected_type(self, frame_type: FrameType):
        """Assign *frame_type* to every currently-selected file item."""
        for item in self._tree.selectedItems():
            idx = item.data(0, Qt.ItemDataRole.UserRole)
            if idx is None:
                continue  # skip directory items
            self._entries[idx].frame_type = frame_type
            item.setText(1, frame_type.value)

    # ── keyboard shortcuts ────────────────────────────────────────────────

    # Key → FrameType mapping for quick type assignment
    _KEY_FRAME_MAP = {
        Qt.Key.Key_F: FrameType.FLATS,
        Qt.Key.Key_D: FrameType.DARKS,
        Qt.Key.Key_B: FrameType.BIAS,
        Qt.Key.Key_L: FrameType.LIGHTS,
    }

    def eventFilter(self, obj, event):
        """Intercept key presses on the file tree for shortcuts."""
        if obj is self._tree and event.type() == QEvent.Type.KeyPress:
            key = event.key()
            ft = self._KEY_FRAME_MAP.get(key)
            if ft is not None:
                self._set_selected_type(ft)
                return True
            if key == Qt.Key.Key_Delete:
                self._delete_selected()
                return True
        return super().eventFilter(obj, event)

    # ── delete logic ─────────────────────────────────────────────────────────

    def _get_item_dir_path(self, item: QTreeWidgetItem) -> str:
        """Reconstruct the relative directory path from a directory tree item."""
        parts: list[str] = []
        current = item
        while current is not None:
            text = current.text(0)
            # Directory items are prefixed with "\U0001f4c1 "
            if text.startswith("\U0001f4c1 "):
                parts.append(text[2:])  # strip emoji + space
            current = current.parent()
        parts.reverse()
        return "/".join(parts)

    def _collect_file_indices_under(self, item: QTreeWidgetItem) -> list[int]:
        """Recursively collect all file-entry indices under a directory item."""
        indices: list[int] = []
        for i in range(item.childCount()):
            child = item.child(i)
            idx = child.data(0, Qt.ItemDataRole.UserRole)
            if idx is not None:
                indices.append(idx)
            else:
                # It's a sub-directory; recurse
                indices.extend(self._collect_file_indices_under(child))
        return indices

    def _delete_selected(self):
        """Delete the selected files/directories from the source after
        confirmation."""
        selected = self._tree.selectedItems()
        if not selected:
            return

        source_root = self._source_entry.text().strip()
        is_smb = self._radio_smb.isChecked()

        # Separate file items and directory items
        file_indices: set[int] = set()
        dir_paths: list[str] = []

        for item in selected:
            idx = item.data(0, Qt.ItemDataRole.UserRole)
            if idx is not None:
                file_indices.add(idx)
            else:
                # Directory item – gather its path and all files underneath
                dir_rel = self._get_item_dir_path(item)
                dir_paths.append(dir_rel)
                for child_idx in self._collect_file_indices_under(item):
                    file_indices.add(child_idx)

        if not file_indices and not dir_paths:
            return

        # Build confirmation message
        parts: list[str] = []
        if dir_paths:
            n = len(dir_paths)
            parts.append(
                f"{n} {'directory' if n == 1 else 'directories'} "
                f"(and all contents)"
            )
        file_only_count = len(file_indices) - sum(
            len(self._collect_file_indices_under(item))
            for item in selected
            if item.data(0, Qt.ItemDataRole.UserRole) is None
        )
        if file_only_count > 0:
            parts.append(
                f"{file_only_count} {'file' if file_only_count == 1 else 'files'}"
            )
        msg = "Are you sure you want to permanently delete " + " and ".join(parts) + "?"

        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        errors: list[str] = []

        # Delete individual files (those NOT covered by a directory deletion)
        dir_prefixes = [d + "/" for d in dir_paths]
        for idx in sorted(file_indices):
            entry = self._entries[idx]
            path_normalised = entry.path.replace("\\", "/")
            # Skip files that are inside a directory being deleted entirely
            if any(path_normalised.startswith(p) for p in dir_prefixes):
                continue
            try:
                if is_smb:
                    self._delete_smb_file(source_root, entry.path)
                else:
                    os.remove(os.path.join(source_root, entry.path))
            except Exception as exc:
                errors.append(f"{entry.path}: {exc}")

        # Delete directories
        for dir_rel in dir_paths:
            try:
                if is_smb:
                    self._delete_smb_directory(source_root, dir_rel)
                else:
                    shutil.rmtree(
                        os.path.join(source_root, dir_rel), ignore_errors=False
                    )
            except Exception as exc:
                errors.append(f"{dir_rel}/: {exc}")

        # Remove deleted entries from the model
        self._entries = [
            e for i, e in enumerate(self._entries) if i not in file_indices
        ]
        self._populate_tree()
        self._analyse_btn.setEnabled(bool(self._entries))

        if errors:
            err_text = "\n".join(errors[:20])
            if len(errors) > 20:
                err_text += f"\n… and {len(errors) - 20} more"
            QMessageBox.warning(
                self,
                "Deletion completed with errors",
                f"Some items could not be deleted:\n\n{err_text}",
            )

    @staticmethod
    def _delete_smb_file(server: str, remote_rel: str):
        """Delete a single file from the SMB share."""
        import smbclient  # type: ignore[import-untyped]

        unc = f"\\\\{server}\\{ASIAIR_SHARE}\\{ASIAIR_AUTORUN}\\{remote_rel}"
        smbclient.remove(unc)

    @staticmethod
    def _delete_smb_directory(server: str, remote_rel: str):
        """Recursively delete a directory from the SMB share."""
        import smbclient  # type: ignore[import-untyped]

        unc = f"\\\\{server}\\{ASIAIR_SHARE}\\{ASIAIR_AUTORUN}\\{remote_rel}"
        # Walk bottom-up to delete files first, then directories
        for dirpath, dirnames, filenames in smbclient.walk(unc):
            for fname in filenames:
                smbclient.remove(f"{dirpath}\\{fname}")
        # Second pass to remove now-empty directories (bottom-up)
        dirs_to_remove: list[str] = []
        for dirpath, dirnames, _filenames in smbclient.walk(unc):
            dirs_to_remove.append(dirpath)
        for d in reversed(dirs_to_remove):
            smbclient.rmdir(d)
        # Remove the root directory itself
        try:
            smbclient.rmdir(unc)
        except Exception:
            pass

    def get_entries(self) -> List[FileEntry]:
        """Public accessor – returns the current file list with assigned types."""
        return list(self._entries)

    # ── preview logic ────────────────────────────────────────────────────────

    def _on_item_double_clicked(self, item, column):
        """Open a preview dialog when a file item is double-clicked."""
        idx = item.data(0, Qt.ItemDataRole.UserRole)
        if idx is None:
            return  # directory item

        entry = self._entries[idx]
        source_root = self._source_entry.text().strip()
        is_smb = self._radio_smb.isChecked()

        try:
            if is_smb:
                file_obj = self._open_smb_file(source_root, entry.path)
            else:
                file_obj = open(
                    os.path.join(source_root, entry.path), "rb"
                )
            with file_obj:
                pixmap = render_preview(file_obj, entry.filename)
        except Exception as exc:
            QMessageBox.warning(
                self, "Preview Error",
                f"Could not open file:\n{exc}",
            )
            return

        if pixmap is None or pixmap.isNull():
            QMessageBox.information(
                self, "Preview",
                "No preview available for this file.",
            )
            return

        dlg = PreviewDialog(pixmap, entry.filename, parent=self)
        dlg.exec()

    # ── analyse logic ────────────────────────────────────────────────────────

    def _analyse_files(self):
        """Read metadata (exposure, ISO) for selected files, or all if
        nothing is selected.  Updates entries and tree widget items."""
        if not self._entries:
            return

        # Determine which entries to analyse
        selected = self._tree.selectedItems()
        if selected:
            indices: list[int] = []
            for item in selected:
                idx = item.data(0, Qt.ItemDataRole.UserRole)
                if idx is not None:
                    indices.append(idx)
            if not indices:
                indices = list(range(len(self._entries)))
        else:
            indices = list(range(len(self._entries)))

        source_root = self._source_entry.text().strip()
        is_smb = self._radio_smb.isChecked()
        total = len(indices)

        # Show progress
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(0)
        self._progress_label.setText(f"Analysing file 0 of {total}")
        self._progress_row.setVisible(True)
        QApplication.processEvents()

        for count, idx in enumerate(indices, start=1):
            entry = self._entries[idx]
            try:
                if is_smb:
                    file_obj = self._open_smb_file(source_root, entry.path)
                else:
                    file_obj = open(os.path.join(source_root, entry.path), "rb")
                with file_obj:
                    exposure, iso_val, date_val = read_file_metadata(
                        file_obj, entry.filename
                    )
                entry.exposure = exposure
                entry.iso = iso_val
                entry.date = date_val
                # Update tree widget item
                tree_item = self._file_items[idx]
                tree_item.setText(2, exposure or "")
                tree_item.setText(3, iso_val or "")
                tree_item.setText(4, date_val or "")
            except Exception:
                pass  # silently skip files that cannot be read

            self._progress_bar.setValue(count)
            self._progress_label.setText(f"Analysing file {count} of {total}")
            QApplication.processEvents()

        self._progress_row.setVisible(False)

    @staticmethod
    def _open_smb_file(server: str, remote_rel: str):
        """Open an SMB file for reading and return a BytesIO object."""
        import smbclient  # type: ignore[import-untyped]

        unc = f"\\\\{server}\\{ASIAIR_SHARE}\\{ASIAIR_AUTORUN}\\{remote_rel}"
        buf = io.BytesIO()
        with smbclient.open_file(unc, mode="rb") as src:
            while chunk := src.read(1024 * 1024):
                buf.write(chunk)
        buf.seek(0)
        return buf

    # ── copy-to-directory logic ───────────────────────────────────────────────

    def _selected_entries(self) -> List[FileEntry]:
        """Return entries for the selected tree items, or *all* entries if
        nothing is selected."""
        selected = self._tree.selectedItems()
        if not selected:
            return list(self._entries)

        indices: set[int] = set()
        for item in selected:
            idx = item.data(0, Qt.ItemDataRole.UserRole)
            if idx is not None:
                indices.add(idx)
        if not indices:
            return list(self._entries)
        return [self._entries[i] for i in sorted(indices)]

    def _show_copy_dialog(self):
        """Open the copy-to-directory dialog and run the copy."""
        if not self._entries:
            QMessageBox.warning(
                self, "Warning", "No files loaded. Please load a source first."
            )
            return

        dlg = CopyDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        dest = dlg.destination()
        if not dest:
            QMessageBox.warning(self, "Warning", "Please enter a destination path.")
            return

        entries = self._selected_entries()
        self._copy_files(entries, dest)

    def _copy_files(self, entries: List[FileEntry], dest_root: str):
        """Create sub-directories and copy *entries* into them."""
        # Create destination sub-directories
        for subdir in FRAME_TYPE_DIR_MAP.values():
            os.makedirs(os.path.join(dest_root, subdir), exist_ok=True)

        source_root = self._source_entry.text().strip()
        is_smb = self._radio_smb.isChecked()
        errors: List[str] = []
        copied = 0
        total = len(entries)

        # Show progress bar
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(0)
        self._progress_label.setText(f"Copying file 0 of {total}")
        self._progress_row.setVisible(True)
        QApplication.processEvents()

        for idx, entry in enumerate(entries, start=1):
            subdir = FRAME_TYPE_DIR_MAP[entry.frame_type]
            dest_path = os.path.join(dest_root, subdir, entry.filename)
            try:
                if is_smb:
                    self._copy_smb_file(source_root, entry.path, dest_path)
                else:
                    src_path = os.path.join(source_root, entry.path)
                    shutil.copy2(src_path, dest_path)
                copied += 1
            except Exception as exc:
                errors.append(f"{entry.path}: {exc}")

            self._progress_bar.setValue(idx)
            self._progress_label.setText(f"Copying file {idx} of {total}")
            QApplication.processEvents()

        # Hide progress bar
        self._progress_row.setVisible(False)

        # Report result
        if errors:
            err_text = "\n".join(errors[:20])
            if len(errors) > 20:
                err_text += f"\n… and {len(errors) - 20} more"
            QMessageBox.warning(
                self,
                "Copy completed with errors",
                f"Copied {copied} file(s).\n\nErrors:\n{err_text}",
            )
        else:
            QMessageBox.information(
                self, "Success", f"Copied {copied} file(s) successfully."
            )

    @staticmethod
    def _copy_smb_file(server: str, remote_rel: str, local_dest: str):
        """Copy a single file from an SMB share to a local path."""
        import smbclient  # type: ignore[import-untyped]

        unc = f"\\\\{server}\\{ASIAIR_SHARE}\\{ASIAIR_AUTORUN}\\{remote_rel}"
        with smbclient.open_file(unc, mode="rb") as src:
            with open(local_dest, "wb") as dst:
                while chunk := src.read(1024 * 1024):
                    dst.write(chunk)


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    """Main entry point"""
    try:
        app = QApplication(sys.argv)
        window = Interface()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"Error initializing script: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
