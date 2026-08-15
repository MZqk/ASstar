# (c) Cyril Richard, adapted from Seti Astro Suite Pro code (2026)
# Palette Picker for Siril - Adapted from Seti Astro Suite Pro
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Version 1.0.2
#
# Palette algorithms originate from Seti Astro Suite Pro, adapted for Siril.
#
# Design note: this version intentionally departs from the original tool by
# dropping the ability to assemble linear images. To do so, the original had
# to apply an automatic statistical stretch before building the palette, which
# took control of the stretch away from the user. Here the channels are
# expected to be already stretched (non-linear), so the user keeps full
# control over how each channel is stretched before mixing.

from __future__ import annotations
import os
import sys

import sirilpy as s
from sirilpy import SirilConnectionError, LogColor

s.ensure_installed("PyQt6")
s.ensure_installed("pillow")
# numpy is guaranteed by the Siril module, no need for ensure_installed.

import numpy as np
from PIL import Image

# OpenCV is optional: used for higher-quality resizing, but we fall back to
# Pillow if it's missing (avoids depending on a system binary).
try:
    import cv2
except Exception:
    cv2 = None

# astropy: only needed to read/write FITS files "by hand"
try:
    s.ensure_installed("astropy")
    from astropy.io import fits as _afits
except Exception:
    _afits = None

from PyQt6.QtCore import (
    Qt, QSize, QEvent, QTimer, QPoint, QSettings,
)
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFileDialog, QMessageBox, QGridLayout,
    QSizePolicy, QDialog, QSlider, QToolButton,
)
from PyQt6.QtGui import (
    QPixmap, QImage, QIcon, QPainter, QPen, QColor, QFont, QFontMetrics, QCursor,
)

VERSION = "1.0.2"
# 1.0.2 CR: push via `new W H 3 RGB` so the result is always a proper RGB image
#           (fixes intermittent monochrome/identical-channels display); copy
#           source WCS/metadata from the channel FITS header instead
# 1.0.1 CR: push result in-memory as "Unsaved compositing result" (no temp file)
# 1.0.0 CR: initial commit


# ----------------------------------------------------------------------------
# Connect to Siril (established at module load)
# ----------------------------------------------------------------------------
siril = s.SirilInterface()
try:
    siril.connect()
    siril.log("Palette Picker: connected to Siril.", LogColor.GREEN)
except SirilConnectionError as e:
    print(f"Could not connect to Siril: {e}")
    sys.exit(1)


# ----------------------------------------------------------------------------
# Array <-> Siril conversion helpers
# ----------------------------------------------------------------------------
def _as_float01(arr) -> np.ndarray:
    """Normalize an array to float32 in [0, 1]."""
    a = np.asarray(arr)
    if a.dtype == np.uint8:
        return a.astype(np.float32) / 255.0
    if a.dtype == np.uint16:
        return a.astype(np.float32) / 65535.0
    a = a.astype(np.float32)
    mx = float(a.max()) if a.size else 1.0
    if mx > 1.5:  # integer data stored as float (e.g. 0..65535)
        a = a / mx
    return np.clip(a, 0.0, 1.0)


def _siril_to_hwc(data) -> np.ndarray:
    """
    Convert FFit.data (mono 2D or RGB channels-first (3,H,W)) to:
       - mono: (H, W)
       - RGB : (H, W, 3)   [channels-last, as expected everywhere else]
    Float32 in [0, 1].
    """
    a = np.asarray(data)
    if a.ndim == 2:
        return _as_float01(a)
    if a.ndim == 3:
        # Siril assumption: channels-first (3, H, W). See warning at top.
        if a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3):
            a = np.transpose(a, (1, 2, 0))
        elif a.shape[0] in (1, 3):
            # ambiguous case (small image): keep channels-first convention
            a = np.transpose(a, (1, 2, 0))
        if a.shape[2] == 1:
            a = a[..., 0]
        return _as_float01(a)
    raise ValueError(f"Unexpected image shape: {a.shape}")


def _hwc_to_siril(arr: np.ndarray) -> np.ndarray:
    """Convert (H,W,3) channels-last -> (3,H,W) channels-first float32 for Siril."""
    a = np.clip(np.asarray(arr, dtype=np.float32), 0.0, 1.0)
    if a.ndim == 2:
        return a
    return np.transpose(a, (2, 0, 1)).copy()


# ----------------------------------------------------------------------------
# Small tool button (replaces SASpro's themed_toolbtn)
# ----------------------------------------------------------------------------
def _toolbtn(text: str, tip: str) -> QToolButton:
    b = QToolButton()
    b.setText(text)
    b.setToolTip(tip)
    b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
    return b


# ----------------------------------------------------------------------------
# Main window
# ----------------------------------------------------------------------------
class PalettePicker(QWidget):
    PALETTES = [
        "SHO", "HOO", "HSO", "HOS",
        "OSS", "OHH", "OSH", "OHS",
        "HSS", "Realistic1", "Realistic2", "Foraxx",
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Palette Picker - Siril v{VERSION}")
        self._settings = QSettings("Siril", "PalettePicker")
        self._persist_prefix = "palette_picker"
        self._geom_restored = False

        # raw channels (float32 ~[0..1])
        self.ha = None
        self.oiii = None
        self.sii = None
        self.osc1 = None
        self.osc2 = None
        # Source file path for each loaded channel, used as a metadata template
        # (WCS, etc.) when pushing the result back into Siril.
        self._paths: dict[str, str] = {}
        self._dim_mismatch_accepted = False

        self.final = None
        self.current_palette = None
        self._thumb_base_pm: dict[str, QPixmap] = {}
        self._selected_name: str | None = None
        self._thumb_buttons: dict[str, QPushButton] = {}

        self._base_pm: QPixmap | None = None
        self._zoom = 1.0
        self._min_zoom = 0.05
        self._max_zoom = 6.0
        self._panning = False
        self._pan_last: QPoint | None = None

        self._build_ui()

    def _k(self, key: str) -> str:
        return f"{self._persist_prefix}/{key}"

    # ---------------- UI ----------------
    def _build_ui(self):
        root = QHBoxLayout(self)

        # -------- left column: controls
        left = QVBoxLayout()
        left_host = QWidget(self)
        left_host.setLayout(left)
        left_host.setFixedWidth(300)

        # Title
        title = QLabel("Palette Picker")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left.addWidget(title)

        # Inputs must already be stretched (non-linear) before mixing.
        note = QLabel("Inputs must be non-linear (already stretched).")
        note.setWordWrap(True)
        note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        note.setStyleSheet("color:#c80; margin-bottom:4px;")
        left.addWidget(note)

        left.addWidget(QLabel("<b>Load channels</b>"))

        self.btn_ha   = QPushButton("Load Ha...");   self.btn_ha.clicked.connect(lambda: self._load_channel("Ha"))
        self.btn_oiii = QPushButton("Load OIII..."); self.btn_oiii.clicked.connect(lambda: self._load_channel("OIII"))
        self.btn_sii  = QPushButton("Load SII...");  self.btn_sii.clicked.connect(lambda: self._load_channel("SII"))
        self.btn_osc1 = QPushButton("Load OSC1 (Ha/OIII)..."); self.btn_osc1.clicked.connect(lambda: self._load_channel("OSC1"))
        self.btn_osc2 = QPushButton("Load OSC2 (SII/OIII)..."); self.btn_osc2.clicked.connect(lambda: self._load_channel("OSC2"))

        self.lbl_ha   = QLabel("No Ha loaded.")
        self.lbl_oiii = QLabel("No OIII loaded.")
        self.lbl_sii  = QLabel("No SII loaded.")
        self.lbl_osc1 = QLabel("No OSC1 loaded.")
        self.lbl_osc2 = QLabel("No OSC2 loaded.")
        for lab in (self.lbl_ha, self.lbl_oiii, self.lbl_sii, self.lbl_osc1, self.lbl_osc2):
            lab.setWordWrap(True)
            lab.setStyleSheet("color:#888; margin-left:8px;")

        for btn, lab in (
            (self.btn_ha, self.lbl_ha),
            (self.btn_oiii, self.lbl_oiii),
            (self.btn_sii, self.lbl_sii),
            (self.btn_osc1, self.lbl_osc1),
            (self.btn_osc2, self.lbl_osc2),
        ):
            left.addWidget(btn)
            left.addWidget(lab)

        self.btn_clear = QPushButton("Clear loaded channels")
        self.btn_clear.clicked.connect(self._clear_channels)
        left.addWidget(self.btn_clear)

        self.btn_create = QPushButton("Create palettes")
        self.btn_create.clicked.connect(self._create_palettes)
        left.addWidget(self.btn_create)

        left.addStretch(1)

        # Footer: code origin + Siril port credit
        footer = QLabel(
            "Original code from Seti Astro\n"
            "Adapted for Siril by Cyril Richard\n"
            "www.setiastro.com"
        )
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer.setStyleSheet("color:#888; font-size:11px; margin-top:8px;")
        left.addWidget(footer)

        root.addWidget(left_host, 0)

        # -------- right column: preview + 4x3 grid
        right = QVBoxLayout()

        tools = QHBoxLayout()
        self.btn_zoom_in  = _toolbtn("Zoom +", "Zoom in")
        self.btn_zoom_out = _toolbtn("Zoom -", "Zoom out")
        self.btn_fit      = _toolbtn("Fit", "Fit to preview")
        self.btn_zoom_in.clicked.connect(lambda: self._zoom_at(1.25))
        self.btn_zoom_out.clicked.connect(lambda: self._zoom_at(0.8))
        self.btn_fit.clicked.connect(self._fit_to_preview)
        tools.addStretch(1)
        tools.addWidget(self.btn_zoom_out)
        tools.addWidget(self.btn_zoom_in)
        tools.addWidget(self.btn_fit)
        tools.addStretch(1)
        right.addLayout(tools)

        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.scroll.setWidget(self.preview)
        self.preview.setMouseTracking(True)
        self.preview.installEventFilter(self)
        self.scroll.viewport().installEventFilter(self)
        self.scroll.installEventFilter(self)
        self.scroll.horizontalScrollBar().installEventFilter(self)
        self.scroll.verticalScrollBar().installEventFilter(self)
        right.addWidget(self.scroll, 1)

        self.grid = QGridLayout()
        self.grid.setHorizontalSpacing(8)
        self.grid.setVerticalSpacing(8)
        self.grid.setContentsMargins(8, 8, 8, 8)

        self.thumb_size = QSize(220, 110)
        btn_w = self.thumb_size.width() + 2
        btn_h = self.thumb_size.height() + 2
        cols, rows = 4, 3

        for idx, name in enumerate(self.PALETTES):
            r, c = divmod(idx, cols)
            b = QPushButton("")
            b.setToolTip(f"{name} - double-click to send to Siril")
            b.setIconSize(self.thumb_size)
            b.setFixedSize(btn_w, btn_h)
            b.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            b.clicked.connect(lambda _=None, n=name: self._on_palette_clicked(n))
            b.setStyleSheet("QPushButton{background:#222;border:1px solid #333;} "
                            "QPushButton:hover{border-color:#555;}")
            b.installEventFilter(self)  # for double-click -> push to Siril
            self._thumb_buttons[name] = b
            self.grid.addWidget(b, r, c)

        grid_host = QWidget(self)
        grid_host.setLayout(self.grid)
        hspacing = self.grid.horizontalSpacing()
        vspacing = self.grid.verticalSpacing()
        m = self.grid.contentsMargins()
        grid_w = cols * btn_w + (cols - 1) * hspacing + m.left() + m.right()
        grid_h = rows * btn_h + (rows - 1) * vspacing + m.top() + m.bottom()
        grid_host.setFixedSize(grid_w, grid_h)
        grid_host.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        right.addWidget(grid_host, 0, alignment=Qt.AlignmentFlag.AlignHCenter)

        self.status = QLabel("")
        right.addWidget(self.status, 0)

        right_host = QWidget(self)
        right_host.setLayout(right)
        root.addWidget(right_host, 1)

        self.setLayout(root)
        self.setMinimumSize(left_host.width() + grid_w + 48, max(560, grid_h + 200))

    # ---------------- resizing ----------------
    def _resize_to(self, arr, size):
        """Resize a numpy array to (w, h). Keeps dtype/scale."""
        if arr is None:
            return None
        w, h = size
        if arr.ndim == 2:
            src_h, src_w = arr.shape
        else:
            src_h, src_w = arr.shape[:2]
        if (src_w, src_h) == (w, h):
            return arr
        if cv2 is None:
            return np.array(
                Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).resize((w, h))
            ).astype(np.float32) / 255.0
        interp = cv2.INTER_AREA if (w < src_w or h < src_h) else cv2.INTER_LINEAR
        return cv2.resize(arr, (w, h), interpolation=interp)

    # ---------------- UI persistence ----------------
    def _save_ui_state(self):
        try:
            st = self._settings
            st.setValue(self._k("window_geometry"), self.saveGeometry())
            st.sync()
        except Exception:
            pass

    def _restore_ui_state(self):
        try:
            st = self._settings
            g = st.value(self._k("window_geometry"), None)
            if g is not None:
                self.restoreGeometry(g)
        except Exception:
            pass

    # ---------------- view state (zoom/pan) ----------------
    def _capture_view_state(self):
        if self._base_pm is None:
            return None
        vp = self.scroll.viewport()
        anchor_vp = QPoint(vp.width() // 2, vp.height() // 2)
        anchor_lbl = self.preview.mapFrom(vp, anchor_vp)
        base_x = anchor_lbl.x() / max(self._zoom, 1e-6)
        base_y = anchor_lbl.y() / max(self._zoom, 1e-6)
        pm = self._base_pm.size()
        fx = 0.5 if pm.width() <= 0 else (base_x / pm.width())
        fy = 0.5 if pm.height() <= 0 else (base_y / pm.height())
        return {"zoom": float(self._zoom), "fx": float(fx), "fy": float(fy)}

    def _restore_view_state(self, state):
        if not state or self._base_pm is None:
            return
        self._zoom = max(self._min_zoom, min(self._max_zoom, float(state["zoom"])))
        self._update_preview_pixmap()
        pm = self._base_pm.size()
        fx = float(state.get("fx", 0.5))
        fy = float(state.get("fy", 0.5))
        lbl_x = int(fx * pm.width() * self._zoom)
        lbl_y = int(fy * pm.height() * self._zoom)
        vp = self.scroll.viewport()
        anchor_vp = QPoint(vp.width() // 2, vp.height() // 2)
        hbar = self.scroll.horizontalScrollBar()
        vbar = self.scroll.verticalScrollBar()
        hbar.setValue(max(hbar.minimum(), min(hbar.maximum(), lbl_x - anchor_vp.x())))
        vbar.setValue(max(vbar.minimum(), min(vbar.maximum(), lbl_y - anchor_vp.y())))

    # ---------------- status ----------------
    def _set_status_label(self, which: str, text: str | None):
        lab = getattr(self, f"lbl_{which.lower()}")
        if text:
            lab.setText(text)
            lab.setStyleSheet("color:#2a7; font-weight:600; margin-left:8px;")
        else:
            lab.setText(f"No {which} loaded.")
            lab.setStyleSheet("color:#888; margin-left:8px;")

    # ---------------- channel loading ----------------
    def _load_channel(self, which: str):
        # Siril only keeps one image loaded at a time, so for now channels are
        # always read from disk (a "From Siril" source would only let us capture
        # a single channel). Go straight to the file picker.
        out = self._load_from_file(which)
        if out is None:
            return

        img, label, path = out
        self._paths[which] = path

        # NB channels -> mono ; OSC -> RGB
        if which in ("Ha", "OIII", "SII"):
            if img.ndim == 3:
                img = img[:, :, 0]
        else:
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)

        setattr(self, which.lower(), _as_float01(img))
        # A new channel may introduce a fresh size mismatch, so ask again.
        self._dim_mismatch_accepted = False
        self._set_status_label(which, label)
        self.status.setText(
            f"{which} loaded ({'mono' if img.ndim == 2 else 'RGB'}) - shape {img.shape}"
        )

        if self.current_palette is None:
            self.current_palette = "SHO"

    def _load_from_file(self, which):
        filt = "Images (*.fit *.fits *.fts *.tif *.tiff *.png *.jpg *.jpeg)"
        path, _ = QFileDialog.getOpenFileName(self, f"Select {which} file", "", filt)
        if not path:
            return None
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext in (".fit", ".fits", ".fts"):
                if _afits is None:
                    raise RuntimeError("astropy unavailable to read FITS files.")
                with _afits.open(path) as hdul:
                    data = None
                    for hdu in hdul:
                        if getattr(hdu, "data", None) is not None:
                            data = hdu.data
                            break
                    if data is None:
                        raise RuntimeError("No image data in the FITS file.")
                    arr = _siril_to_hwc(data)
                    # Siril/FITS store images bottom-up; flip to top-down so the
                    # preview matches Siril's on-screen orientation. (TIFF/PNG are
                    # already top-down and must not be flipped.)
                    arr = np.flipud(arr).copy()
            else:
                im = Image.open(path)
                arr = _as_float01(np.asarray(im))
        except Exception as e:
            QMessageBox.critical(self, "Load error",
                                 f"Could not load {os.path.basename(path)}:\n{e}")
            return None
        return arr, f"From file: {os.path.basename(path)}", path

    # ---------------- window events ----------------
    def showEvent(self, e):
        super().showEvent(e)
        if self._geom_restored:
            return
        self._geom_restored = True

        def _after():
            self._restore_ui_state()
            if self._base_pm is None:
                self._center_scrollbars()
        QTimer.singleShot(0, _after)

    def closeEvent(self, e):
        try:
            self._save_ui_state()
        except Exception:
            pass
        super().closeEvent(e)

    # ---------------- thumbnails ----------------
    def _render_thumb(self, name: str):
        base = self._thumb_base_pm.get(name)
        if base is None:
            return
        pm = base.copy()
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = QFont("Helvetica", 10, QFont.Weight.DemiBold)
        p.setFont(font)
        fm = QFontMetrics(font)
        pad = 6
        strip_h = fm.height() + pad * 2
        strip = pm.rect().adjusted(0, pm.height() - strip_h, 0, 0)
        p.fillRect(strip, QColor(0, 0, 0, 160))
        color = QColor(102, 255, 102) if self._selected_name == name else QColor(255, 255, 255)
        p.setPen(QPen(color))
        p.drawText(strip, Qt.AlignmentFlag.AlignCenter, name)
        p.end()
        btn = self._thumb_buttons[name]
        btn.setIcon(QIcon(pm))
        btn.setIconSize(self.thumb_size)

    def _create_palettes(self):
        """Build the 12 thumbnails from the loaded (non-linear) channels."""
        ha, oo, si = self._prepared_channels(for_thumbs=True)
        if oo is None or (ha is None and si is None):
            QMessageBox.warning(self, "Missing channels", "Load at least OIII + (Ha or SII).")
            return

        built = 0
        for name in self.PALETTES:
            r, g, b = self._map_channels_or_special(name, ha, oo, si)
            if any(ch is None for ch in (r, g, b)):
                self._thumb_base_pm.pop(name, None)
                self._thumb_buttons[name].setIcon(QIcon())
                continue
            r = np.clip(np.nan_to_num(r), 0, 1)
            g = np.clip(np.nan_to_num(g), 0, 1)
            b = np.clip(np.nan_to_num(b), 0, 1)
            rgb = np.stack([r, g, b], axis=2).astype(np.float32)
            # Normalize like _generate_for_palette so the thumbnail matches the
            # full preview / pushed image in brightness.
            mx = float(rgb.max()) or 1.0
            rgb = (rgb / mx).astype(np.float32)
            pm = QPixmap.fromImage(self._to_qimage(rgb)).scaled(
                self.thumb_size, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._thumb_base_pm[name] = pm
            self._render_thumb(name)
            built += 1

        self.status.setText(f"Created {built} palette previews.")

    def _on_palette_clicked(self, name: str):
        self._selected_name = name
        for n in self.PALETTES:
            self._render_thumb(n)
        self.current_palette = name
        self._generate_for_palette(name)

    # ---------------- channel preparation ----------------
    def _prepared_channels(self, for_thumbs: bool = False):
        def pick(name):
            return getattr(self, name.lower())

        ha = pick("Ha")
        oo = pick("OIII")
        si = pick("SII")
        o1 = pick("OSC1")
        o2 = pick("OSC2")

        def _match(src, ref):
            # Resize src (2D) to ref's (H,W) so the fusion below never hits a
            # NumPy broadcast error when channels have different dimensions.
            if ref is None or src.shape == ref.shape[:2]:
                return src
            return self._resize_to(src, (ref.shape[1], ref.shape[0]))

        # synthesize from OSC (before crop)
        if o1 is not None:  # OSC1: R~Ha, mean(G,B)~OIII
            h1 = o1[..., 0]
            g1b1 = o1[..., 1:3].mean(axis=2)
            ha = h1 if ha is None else 0.5 * ha + 0.5 * _match(h1, ha)
            oo = g1b1 if oo is None else 0.5 * oo + 0.5 * _match(g1b1, oo)

        if o2 is not None:  # OSC2: R~SII, mean(G,B)~OIII
            s2 = o2[..., 0]
            g2b2 = o2[..., 1:3].mean(axis=2)
            si = s2 if si is None else 0.5 * si + 0.5 * _match(s2, si)
            oo = g2b2 if oo is None else 0.5 * oo + 0.5 * _match(g2b2, oo)

        # shapes must match for full size
        shapes = [x.shape[:2] for x in (ha, oo, si) if x is not None]
        if len(shapes) and len(set(shapes)) > 1 and not for_thumbs:
            ref = ha if ha is not None else (oo if oo is not None else si)
            ref_name = "Ha" if ha is not None else ("OIII" if oo is not None else "SII")
            ref_h, ref_w = ref.shape[:2]
            if not self._dim_mismatch_accepted:
                msg = (
                    "The loaded channels have different dimensions.\n\n"
                    f"- Ha:   {None if ha is None else ha.shape}\n"
                    f"- OIII: {None if oo is None else oo.shape}\n"
                    f"- SII:  {None if si is None else si.shape}\n\n"
                    "Resize the channels to match the reference frame?\n"
                    f"- Reference: {ref_name}\n"
                    f"- Target size: ({ref_w} x {ref_h})"
                )
                ret = QMessageBox.question(
                    self, "Size mismatch", msg,
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )
                if ret != QMessageBox.StandardButton.Yes:
                    return None, None, None
                self._dim_mismatch_accepted = True
            ha = self._resize_to(ha, (ref_w, ref_h)) if ha is not None else None
            oo = self._resize_to(oo, (ref_w, ref_h)) if oo is not None else None
            si = self._resize_to(si, (ref_w, ref_h)) if si is not None else None

        # thumbnails: crop/reduce AFTER synthesis
        if for_thumbs:
            ref = oo if oo is not None else (ha if ha is not None else si)
            if ref is not None:
                ref_h, ref_w = ref.shape[:2]
                ha = self._resize_to(ha, (ref_w, ref_h)) if ha is not None else None
                oo = self._resize_to(oo, (ref_w, ref_h)) if oo is not None else None
                si = self._resize_to(si, (ref_w, ref_h)) if si is not None else None
                half_w = max(1, int(ref_w * 0.5))
                half_h = max(1, int(ref_h * 0.5))
                ha = self._resize_to(ha, (half_w, half_h)) if ha is not None else None
                oo = self._resize_to(oo, (half_w, half_h)) if oo is not None else None
                si = self._resize_to(si, (half_w, half_h)) if si is not None else None

        return ha, oo, si

    def _generate_for_palette(self, pal: str):
        ha, oo, si = self._prepared_channels()
        if oo is None or (ha is None and si is None):
            return
        r, g, b = self._map_channels_or_special(pal, ha, oo, si)
        if any(ch is None for ch in (r, g, b)):
            QMessageBox.critical(self, "Palette error", f"Could not build palette {pal}.")
            return
        r = np.clip(np.nan_to_num(r), 0, 1)
        g = np.clip(np.nan_to_num(g), 0, 1)
        b = np.clip(np.nan_to_num(b), 0, 1)
        rgb = np.stack([r, g, b], axis=2).astype(np.float32)
        mx = float(rgb.max()) or 1.0
        self.final = (rgb / mx).astype(np.float32)
        first = (self._base_pm is None)
        self._set_preview_image(self._to_qimage(self.final), fit=first, preserve_view=True)
        self.status.setText(f"Preview generated: {pal}")

    # ---------------- main preview ----------------
    def _set_preview_image(self, qimg, *, fit=False, preserve_view=True):
        state = None
        if preserve_view and (not fit) and (self._base_pm is not None):
            state = self._capture_view_state()
        self._base_pm = QPixmap.fromImage(qimg)
        if fit or state is None:
            self._zoom = 1.0
            self._update_preview_pixmap()
            if fit:
                QTimer.singleShot(0, self._fit_to_preview)
            else:
                QTimer.singleShot(0, self._center_scrollbars)
            return
        self._restore_view_state(state)

    def _update_preview_pixmap(self):
        if self._base_pm is None:
            return
        base_sz = self._base_pm.size()
        w = max(1, int(base_sz.width() * self._zoom))
        h = max(1, int(base_sz.height() * self._zoom))
        scaled = self._base_pm.scaled(
            w, h, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview.setPixmap(scaled)
        self.preview.resize(scaled.size())

    def _set_zoom(self, new_zoom: float):
        self._zoom = max(self._min_zoom, min(self._max_zoom, new_zoom))
        self._update_preview_pixmap()

    def _zoom_at(self, factor=1.25, anchor_vp=None):
        if self._base_pm is None:
            return
        vp = self.scroll.viewport()
        if anchor_vp is None:
            anchor_vp = QPoint(vp.width() // 2, vp.height() // 2)
        lbl_before = self.preview.mapFrom(vp, anchor_vp)
        old_zoom = self._zoom
        new_zoom = max(self._min_zoom, min(self._max_zoom, old_zoom * factor))
        ratio = new_zoom / max(old_zoom, 1e-6)
        if abs(ratio - 1.0) < 1e-6:
            return
        self._zoom = new_zoom
        self._update_preview_pixmap()
        lbl_after_x = int(lbl_before.x() * ratio)
        lbl_after_y = int(lbl_before.y() * ratio)
        hbar = self.scroll.horizontalScrollBar()
        vbar = self.scroll.verticalScrollBar()
        hbar.setValue(max(hbar.minimum(), min(hbar.maximum(), lbl_after_x - anchor_vp.x())))
        vbar.setValue(max(vbar.minimum(), min(vbar.maximum(), lbl_after_y - anchor_vp.y())))

    def _fit_to_preview(self):
        if self._base_pm is None:
            return
        vp = self.scroll.viewport().size()
        pm = self._base_pm.size()
        if pm.width() == 0 or pm.height() == 0:
            return
        k = min(vp.width() / pm.width(), vp.height() / pm.height())
        self._set_zoom(max(self._min_zoom, min(self._max_zoom, k)))
        self._center_scrollbars()

    def _center_scrollbars(self):
        h = self.scroll.horizontalScrollBar()
        v = self.scroll.verticalScrollBar()
        h.setValue((h.maximum() + h.minimum()) // 2)
        v.setValue((v.maximum() + v.minimum()) // 2)

    # ---------------- palette mixing (UNCHANGED) ----------------
    def _map_channels_or_special(self, name, ha, oo, si):
        # Remember which NB channels were actually provided: the substitution
        # below fills the missing one, which would otherwise hide the genuine
        # "single NB channel" case (e.g. Foraxx bicolor needs si truly absent).
        si_missing = si is None
        # substitution
        if ha is None and si is not None: ha = si
        if si is None and ha is not None: si = ha

        basic = {
            "SHO": (si, ha, oo),
            "HOO": (ha, oo, oo),
            "HSO": (ha, si, oo),
            "HOS": (ha, oo, si),
            "OSS": (oo, si, si),
            "OHH": (oo, ha, ha),
            "OSH": (oo, si, ha),
            "OHS": (oo, ha, si),
            "HSS": (ha, si, si),
        }
        if name in basic:
            return basic[name]

        try:
            if name == "Realistic1":
                r = (ha + si) / 2 if (ha is not None and si is not None) else (ha if ha is not None else 0)
                g = 0.3 * (ha if ha is not None else 0) + 0.7 * (oo if oo is not None else 0)
                b = 0.9 * (oo if oo is not None else 0) + 0.1 * (ha if ha is not None else 0)
                return r, g, b
            if name == "Realistic2":
                r = 0.7 * (ha if ha is not None else 0) + 0.3 * (si if si is not None else 0)
                g = 0.3 * (si if si is not None else 0) + 0.7 * (oo if oo is not None else 0)
                b = (oo if oo is not None else 0)
                return r, g, b
            if name == "Foraxx":
                if ha is not None and oo is not None and si_missing:
                    r = ha; b = oo
                    t = ha * oo
                    g = (t ** (1 - t)) * ha + (1 - (t ** (1 - t))) * oo
                    return r, g, b
                if ha is not None and oo is not None and not si_missing:
                    t = np.clip(oo, 1e-6, 1.0) ** (1 - np.clip(oo, 1e-6, 1.0))
                    r = t * si + (1 - t) * ha
                    t2 = ha * oo
                    g = (t2 ** (1 - t2)) * ha + (1 - (t2 ** (1 - t2))) * oo
                    b = oo
                    return r, g, b
                return basic["SHO"]
        except Exception:
            return basic.get("SHO", (ha, oo, si))

        return basic.get("SHO", (ha, oo, si))

    # ---------------- push to Siril ----------------
    def _template_header(self) -> str | None:
        """Return the FITS header (as a string) of a source channel, if any.

        Used to carry the source metadata (WCS, etc.) over to the pushed
        result. Only FITS channels have a header; TIFF/PNG sources return None.
        """
        if _afits is None:
            return None
        for which in ("Ha", "OIII", "SII", "OSC1", "OSC2"):
            path = self._paths.get(which)
            if not path or not os.path.exists(path):
                continue
            if os.path.splitext(path)[1].lower() not in (".fit", ".fits", ".fts"):
                continue
            try:
                with _afits.open(path) as hdul:
                    for hdu in hdul:
                        if getattr(hdu, "data", None) is not None:
                            return hdu.header.tostring(sep="\n")
            except Exception:
                continue
        return None

    def _push_to_siril(self, name: str):
        """Build the chosen palette at full resolution and push it into Siril."""
        # Make sure the selected palette is the one we push.
        self._on_palette_clicked(name)
        if self.final is None:
            QMessageBox.warning(self, "No image", "Could not build the palette.")
            return

        # Siril's pixeldata buffer is channels-first and bottom-up. Our internal
        # array is top-down (H,W,3), so flip back before converting.
        data = _hwc_to_siril(np.flipud(self.final)).astype(np.float32)
        _, height, width = data.shape

        try:
            # Create a fresh 3-channel RGB image, then fill it with our pixels.
            #
            # We used to `load` a source channel as a template and swap its
            # pixeldata in place, but the source channels are mono: loading one
            # leaves Siril in a single-layer display state, and set_image_pixeldata()
            # does not fully rebuild that state, so the RGB result intermittently
            # rendered as monochrome (one channel shown for all three). The `new`
            # command runs Siril's full image-loaded path, giving a correct
            # 3-layer RGB display. Metadata (WCS, etc.) is copied over separately
            # from a source channel's FITS header.
            header = self._template_header()

            siril.cmd("new", str(width), str(height), "3", "RGB")
            if not siril.is_image_loaded():
                QMessageBox.warning(self, "No image",
                    "Could not create an RGB image in Siril.")
                return

            with siril.image_lock():
                siril.set_image_pixeldata(data)
                if header:
                    try:
                        siril.set_image_metadata_from_header_string(header)
                    except Exception as meta_err:
                        siril.log(f"Palette Picker: could not copy source "
                                  f"metadata ({meta_err}).", LogColor.SALMON)
            # Mark it as an unsaved in-memory result rather than a file.
            siril.set_image_filename("Unsaved compositing result")
            siril.undo_save_state(f"Palette Picker: {name}")
            siril.log(f"Palette {name} pushed into Siril.", LogColor.GREEN)
            self.status.setText(f"Palette {name} pushed to Siril (unsaved result).")
        except Exception as e:
            QMessageBox.critical(self, "Error",
                                 f"Could not push the result to Siril:\n{e}")

    # ---------------- utilities ----------------
    def _clear_channels(self):
        self.ha = self.oiii = self.sii = self.osc1 = self.osc2 = None
        self._paths.clear()
        self._dim_mismatch_accepted = False
        self.final = None
        self.preview.clear()
        for which in ("Ha", "OIII", "SII", "OSC1", "OSC2"):
            self._set_status_label(which, None)
        for name, b in self._thumb_buttons.items():
            b.setIcon(QIcon())
        self._thumb_base_pm.clear()
        self._selected_name = None
        self.status.setText("Cleared all loaded channels.")

    def _to_qimage(self, arr):
        a = np.clip(arr, 0, 1)
        if a.ndim == 2:
            u = (a * 255).astype(np.uint8)
            h, w = u.shape
            return QImage(u.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
        if a.ndim == 3 and a.shape[2] == 3:
            u = (a * 255).astype(np.uint8)
            h, w, _ = u.shape
            return QImage(u.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
        raise ValueError(f"Unexpected image shape: {a.shape}")

    # ---------------- global zoom/pan (wheel + drag) + thumb double-click ----------------
    def eventFilter(self, obj, ev):
        # Double-click on a palette thumbnail -> send it straight to Siril.
        if ev.type() == QEvent.Type.MouseButtonDblClick and obj in self._thumb_buttons.values():
            for n, btn in self._thumb_buttons.items():
                if btn is obj:
                    self._push_to_siril(n)
                    break
            return True

        if ev.type() == QEvent.Type.Wheel and (
            obj is self.preview
            or obj is self.scroll
            or obj is self.scroll.viewport()
            or obj is self.scroll.horizontalScrollBar()
            or obj is self.scroll.verticalScrollBar()
        ):
            ev.accept()
            vp = self.scroll.viewport()
            anchor_vp = vp.mapFromGlobal(ev.globalPosition().toPoint())
            r = vp.rect()
            if not r.contains(anchor_vp):
                anchor_vp.setX(max(r.left(), min(r.right(), anchor_vp.x())))
                anchor_vp.setY(max(r.top(), min(r.bottom(), anchor_vp.y())))
            dy = ev.pixelDelta().y()
            if dy != 0:
                abs_dy = abs(dy)
                ctrl_down = bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier)
                if abs_dy <= 3:
                    base_factor = 1.012 if ctrl_down else 1.010
                elif abs_dy <= 10:
                    base_factor = 1.025 if ctrl_down else 1.020
                else:
                    base_factor = 1.040 if ctrl_down else 1.030
                factor = base_factor if dy > 0 else 1.0 / base_factor
            else:
                dy = ev.angleDelta().y()
                if dy == 0:
                    return True
                ctrl_down = bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier)
                step = 1.25 if ctrl_down else 1.15
                factor = step if dy > 0 else 1.0 / step
            self._zoom_at(factor, anchor_vp)
            return True

        if obj is self.scroll.viewport():
            if ev.type() == QEvent.Type.MouseButtonPress and ev.button() == Qt.MouseButton.LeftButton:
                self._panning = True
                self._pan_last = ev.position().toPoint()
                self.scroll.viewport().setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
                return True
            if ev.type() == QEvent.Type.MouseMove and self._panning:
                cur = ev.position().toPoint()
                delta = cur - (self._pan_last or cur)
                self._pan_last = cur
                h = self.scroll.horizontalScrollBar()
                v = self.scroll.verticalScrollBar()
                h.setValue(h.value() - delta.x())
                v.setValue(v.value() - delta.y())
                return True
            if ev.type() == QEvent.Type.MouseButtonRelease and ev.button() == Qt.MouseButton.LeftButton:
                self._panning = False
                self._pan_last = None
                self.scroll.viewport().setCursor(QCursor(Qt.CursorShape.ArrowCursor))
                return True

        return super().eventFilter(obj, ev)


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def main():
    app = QApplication.instance() or QApplication(sys.argv)
    win = PalettePicker()
    win.resize(1200, 800)
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
