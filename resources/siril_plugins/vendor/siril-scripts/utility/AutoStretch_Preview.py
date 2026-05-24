# Feb. 27, 2026
# (c) Rich Stevenson - Deep Space Astro
# SPDX-License-Identifier: MIT
# Script for previewing a linear image with an MTF AutoStretch applied,
# with interactive sliders for Target Median and Sigma.
#
# Version History:
# ================
# v1.1.0 Original release
# v1.1.1 Added option to unlink/link channels

import sys
import numpy as np

import sirilpy as s
s.ensure_installed("PyQt6")

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QWidget, QLabel, QScrollArea, QStatusBar, QSlider, QPushButton, QFrame,
    QComboBox, QLineEdit, QCheckBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QImage, QPainter, QWheelEvent, QMouseEvent, QDoubleValidator

VERSION = "1.1.1"

# ── MTF defaults ───────────────────────────────────────────────────────────────
MAD_NORM            = 1.4826
DEFAULT_TARGET_BG   = 0.25    # Target median (background level)
DEFAULT_SIGMA       = -2.8    # Shadow clipping in sigma units

# ── MTF maths ─────────────────────────────────────────────────────────────────


def _mtf(x: float, m: float) -> float:
    """Midtone transfer function for a single scalar value."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if x == m:
        return 0.5
    return ((m - 1.0) * x) / ((2.0 * m - 1.0) * x - m)

# Inverse of MTF to solve for midtones.
def _mtf_inverse(x: float, y: float) -> float:
    """Return m such that _mtf(x, m) == y."""
    denom = (x + y - 2.0 * x * y)
    if denom <= 1e-12:
        return 0.5
    m = x * (1.0 - y) / denom
    return float(np.clip(m, 1e-6, 1.0 - 1e-6))


def _apply_mtf_array(data: np.ndarray, shadows: float, midtones: float, highlights: float) -> np.ndarray:
    """Apply MTF curve to a float32 channel array. Returns float32 in [0,1]."""
    span = highlights - shadows
    if span <= 0.0:
        span = 1.0
    data = (data - shadows) / span
    data = np.clip(data, 0.0, 1.0)
    m = midtones
    if abs(m - 0.5) < 1e-9:
        return data
    denom = (2.0 * m - 1.0) * data - m
    safe  = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
    return np.clip(((m - 1.0) * data) / safe, 0.0, 1.0)


def compute_channel_stats(image: np.ndarray):
    """Return per-channel (medians, mads) lists for a float32 CHW array."""
    c = image.shape[0]
    medians, mads = [], []
    for ch in range(c):
        flat = image[ch].ravel()
        med  = float(np.median(flat))
        mad  = float(np.median(np.abs(flat - med)))
        medians.append(med)
        mads.append(mad)
    return medians, mads


def compute_autostretch_params(image: np.ndarray,
                                target_bg: float = DEFAULT_TARGET_BG,
                                sigma: float     = DEFAULT_SIGMA,
                                linked: bool     = True
                                ) -> list[tuple[float, float, float]]:
    """
    Compute MTF autostretch parameters from image statistics.

    Args:
        image:     float32 array, shape (C, H, W)
        target_bg: target background / median level  [0.01 – 0.99]
        sigma:     shadow clipping in sigma units    [-5.0 – 0.0]
        linked:    if True, derive a single set of params from all channels
                   combined (classic behaviour); if False, compute independent
                   params per channel.

    Returns:
        List of (shadows, midtones, highlights) tuples — one per channel when
        unlinked, or one tuple repeated C times when linked.
    """
    c = image.shape[0]
    medians, mads = compute_channel_stats(image)

    def _params_for(med_mean, mad_mean):
        if mad_mean == 0.0:
            mad_mean = 0.001
        shadows  = max(0.0, med_mean + sigma * mad_mean)
        m2       = med_mean - shadows
        midtones = _mtf_inverse(m2, target_bg)
        return shadows, midtones, 1.0

    if linked:
        #  Filter out inverted channels (median > 0.5) before averaging.
        # The original code used an all-or-nothing check (fallback only when ALL
        # channels are inverted). A partially-inverted channel (e.g. only R > 0.5)
        # would inflate med_mean and produce an incorrect stretch for the whole image.
        # We now exclude those channels from the average; if none remain, fall back
        # to neutral params just as before.
        valid_medians = [m for m in medians if m <= 0.5]
        valid_mads    = [d for m, d in zip(medians, mads) if m <= 0.5]
        if not valid_medians:
            params = (0.0, 0.5, 1.0)
        else:
            n        = len(valid_medians)
            med_mean = sum(valid_medians) / n
            mad_mean = sum(valid_mads)    / n * MAD_NORM
            params   = _params_for(med_mean, mad_mean)
        return [params] * c

    # ── Unlinked: per-channel params ──────────────────────────────────────────
    result = []
    for ch in range(c):
        med = medians[ch]
        mad = mads[ch] * MAD_NORM
        if med > 0.5:
            # Inverted channel — neutral fallback
            result.append((0.0, 0.5, 1.0))
        else:
            result.append(_params_for(med, mad))
    return result


def apply_stretch(image: np.ndarray,
                  channel_params: list[tuple[float, float, float]]) -> np.ndarray:
    """
    Apply MTF stretch with per-channel (or shared) lo/mid/hi parameters.

    Args:
        image:          float32 array, shape (C, H, W)
        channel_params: list of (shadows, midtones, highlights) — length == C

    Returns:
        uint8 array of shape (H, W) for mono or (H, W, 3) for RGB
    """
    c = image.shape[0]
    out = np.empty_like(image)
    for ch in range(c):
        shadows, midtones, highlights = channel_params[ch]
        out[ch] = _apply_mtf_array(image[ch].copy(), shadows, midtones, highlights)

    out = (out * 255.0).clip(0, 255).astype(np.uint8)

    if c == 1:
        return out[0]                           # (H, W)
    return np.transpose(out, (1, 2, 0))         # (H, W, 3)


def _normalise_image(data: np.ndarray) -> np.ndarray:
    """Ensure image is float32 CHW."""
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    if data.ndim == 2:
        data = data[np.newaxis]                 # (H,W) -> (1,H,W)
    return data


def _ensure_unit_range(data: np.ndarray,
                       pct_low: float = 0.1,
                       pct_high: float = 99.9,
                       eps: float = 1e-12) -> tuple[np.ndarray, bool]:
    """Return (data01, changed). Preview-only guard to keep MTF math in [0,1]."""
    if data.size == 0:
        return data, False

    vals = data.ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return data, False

    vmin = float(vals.min())
    vmax = float(vals.max())

    if vmin >= -1e-6 and vmax <= 1.0 + 1e-6:
        return data, False

    lo = float(np.percentile(vals, pct_low))
    hi = float(np.percentile(vals, pct_high))
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) <= eps:
        lo, hi = vmin, vmax
        if (hi - lo) <= eps:
            return np.clip(data, 0.0, 1.0), True

    out = (data - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)
    return out, True


# ── Zoomable image widget ─────────────────────────────────────────────────────

class ZoomableLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: black;")
        self.setMinimumSize(400, 300)
        self._pixmap   = None
        self._scale    = 1.0
        self._pan_x    = 0.0
        self._pan_y    = 0.0
        self._drag_pos = None

    def load_uint8(self, data: np.ndarray, preserve_view: bool = False):
        """Display a uint8 (H,W) or (H,W,3) array."""
        if data.ndim == 2:
            h, w = data.shape
            qimg = QImage(data.tobytes(), w, h, w, QImage.Format.Format_Grayscale8)
        else:
            h, w, _ = data.shape
            qimg = QImage(data.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)

        qimg = qimg.mirrored(False, True)           # FITS row order
        self._pixmap = QPixmap.fromImage(qimg)

        if preserve_view:
            self._redraw()
        else:
            self._fit()

    def _fit(self):
        if self._pixmap is None:
            return
        ws = self.size()
        self._scale = min(ws.width() / self._pixmap.width(),
                          ws.height() / self._pixmap.height())
        self._pan_x = self._pan_y = 0.0
        self._redraw()

    def _one_to_one(self):
        if self._pixmap is None:
            return
        self._scale = 1.0
        self._pan_x = self._pan_y = 0.0
        self._redraw()

    def _redraw(self):
        if self._pixmap is None:
            return
        ws  = self.size()
        buf = QPixmap(ws)
        buf.fill(Qt.GlobalColor.black)
        p   = QPainter(buf)
        p.translate(
            (ws.width()  - self._pixmap.width()  * self._scale) / 2 + self._pan_x * self._scale,
            (ws.height() - self._pixmap.height() * self._scale) / 2 + self._pan_y * self._scale
        )
        p.scale(self._scale, self._scale)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.drawPixmap(0, 0, self._pixmap)
        p.end()
        self.setPixmap(buf)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._pixmap:
            self._redraw()

    def wheelEvent(self, event: QWheelEvent):
        if self._pixmap is None:
            return
        factor    = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        new_scale = self._scale * factor
        mp        = event.position()
        ws        = self.size()
        old       = self._scale
        img_x = (mp.x() - (ws.width()  - self._pixmap.width()  * old) / 2 - self._pan_x * old) / old
        img_y = (mp.y() - (ws.height() - self._pixmap.height() * old) / 2 - self._pan_y * old) / old
        self._scale = new_scale
        self._pan_x = (mp.x() - (ws.width()  - self._pixmap.width()  * new_scale) / 2 - img_x * new_scale) / new_scale
        self._pan_y = (mp.y() - (ws.height() - self._pixmap.height() * new_scale) / 2 - img_y * new_scale) / new_scale
        self._redraw()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.pos()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._drag_pos and event.buttons() & Qt.MouseButton.LeftButton:
            d = event.pos() - self._drag_pos
            self._drag_pos = event.pos()
            self._pan_x += d.x() / self._scale
            self._pan_y += d.y() / self._scale
            self._redraw()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = None

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        self._fit()


# ── Main window ───────────────────────────────────────────────────────────────

class PreviewWindow(QMainWindow):
    """
    Multi-image MTF autostretch preview window.

    Parameters
    ----------
    images : list of (str, np.ndarray)
        Between 1 and 3 ``(name, data)`` pairs.  ``data`` must be a float32
        array with shape ``(C, H, W)`` or ``(H, W)`` (mono).

    refresh_callback : callable or None
        Optional ``() -> list[(str, np.ndarray)]`` called when the user clicks
        *Refresh from Siril*.

    siril_conn : SirilInterface or None
        Siril connection used for the fallback single-image refresh.

    Channel linking
    ---------------
    When *Link Channels* is checked (default), a single set of lo/mid/hi
    parameters is derived from the average statistics of all channels — the
    classic behaviour.  When unchecked, each channel gets its own independent
    set of parameters, which can help reveal colour casts in linear data or
    provide a more balanced preview of narrowband RGB composites.

    The per-channel params are stored in ``self._channel_params`` —
    a list of (lo, mid, hi) tuples, one per channel.  The editable
    Lo / Mid / Hi fields always show the *first channel's* values; when
    channels are unlinked the fields become read-only hints labelled
    "Ch 0" to remind the user that each channel has its own values.

    Slider ranges
    -------------
    Target Median : integer 1–99   → float 0.01–0.99  (÷100)
    Sigma         : integer -500–0 → float -5.00–0.00 (÷100)
    """

    def __init__(self,
                 images: list[tuple[str, np.ndarray]],
                 refresh_callback=None,
                 siril_conn=None):
        super().__init__()

        if not images:
            raise ValueError("At least one image must be supplied.")
        if len(images) > 3:
            raise ValueError("At most three images are supported.")

        self.setWindowTitle(f"Deep Space Astro – MTF Autostretch Preview - v{VERSION}")
        self.resize(900, 800)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)

        header_msg = (
            f"\n################################################################\n"
            f"# MTF Autostretch Preview v{VERSION}\n"
            "# Interactive MTF AutoStretch Preview Tool\n"
            "# Author: Rich Stevenson - Deep Space Astro\n"
            "# YouTube:   https://www.youtube.com/@DeepSpaceAstro\n"
            "# Facebook:  https://www.facebook.com/@DeepSpaceAstro\n"
            "# Tiktok:    https://www.tiktok.com/@deepspaceastro\n"
            "# Instagram: https://www.instagram.com/deepspaceastro_official/\n"
            "# Discord:   https://discord.deepspaceastro.net\n"
            "#################################################################"
        )
        try: siril_conn.log(header_msg)
        except: print(header_msg)

        self._siril_conn       = siril_conn
        self._refresh_callback = refresh_callback

        # Normalise all images to float32 CHW
        self._images: list[tuple[str, np.ndarray]] = []
        _unit_guard_triggered = False
        for name, data in images:
            nd = _normalise_image(data)
            nd, changed = _ensure_unit_range(nd)
            _unit_guard_triggered = _unit_guard_triggered or changed
            self._images.append((name, nd))

        if _unit_guard_triggered:
            msg = (
                "Note: input data was not in [0,1]. "
                "Applied a preview-only normalization to unit range for consistent MTF math."
            )
            try:
                if self._siril_conn is not None:
                    self._siril_conn.log(msg)
            except Exception:
                pass

        self._active_idx = 0

        self._target_bg  = DEFAULT_TARGET_BG
        self._sigma      = DEFAULT_SIGMA
        self._linked     = True          # ← channels linked by default

        # Current MTF parameters — list of (lo, mid, hi) per channel
        c = self._images[0][1].shape[0]
        self._channel_params: list[tuple[float, float, float]] = [(0.0, 0.5, 1.0)] * c

        # Debounce: re-render 120 ms after the last slider move
        self._redraw_timer = QTimer()
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(120)
        self._redraw_timer.timeout.connect(self._recalculate_and_display)

        self._build_ui()

        # Compute initial autostretch from image[0] and display
        self.status.showMessage("Calculating MTF autostretch parameters …")
        QApplication.processEvents()
        self._recalculate_and_display(fit_view=False)
        QTimer.singleShot(0, self.img_label._fit)

    # ── helpers ────────────────────────────────────────────────────────────────

    @property
    def _ref_image(self) -> np.ndarray:
        return self._images[0][1]

    @property
    def _active_image(self) -> np.ndarray:
        return self._images[self._active_idx][1]

    @property
    def _active_name(self) -> str:
        return self._images[self._active_idx][0]

    def _image_info_str(self) -> str:
        data = self._active_image
        c, h, w = data.shape
        mode = "Mono" if c == 1 else "RGB"
        return f"{self._active_name}  |  {w}×{h}  |  {mode}"

    def _is_mono(self) -> bool:
        return self._ref_image.shape[0] == 1

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        root    = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 4)
        root.setSpacing(4)
        self.setCentralWidget(central)

        # Image area
        self.img_label = ZoomableLabel()
        scroll = QScrollArea()
        scroll.setWidget(self.img_label)
        scroll.setWidgetResizable(True)
        root.addWidget(scroll, stretch=1)

        # ── Image selector (only when >1 image supplied) ───────────────────────
        if len(self._images) > 1:
            sel_row = QHBoxLayout()
            sel_row.setContentsMargins(4, 2, 4, 0)
            sel_row.setSpacing(8)

            sel_label = QLabel("Display image:")
            sel_label.setStyleSheet("font-weight: bold;")
            sel_row.addWidget(sel_label)

            self._combo = QComboBox()
            for name, _ in self._images:
                self._combo.addItem(name)
            self._combo.setCurrentIndex(0)
            self._combo.currentIndexChanged.connect(self._on_image_selected)
            sel_row.addWidget(self._combo, stretch=1)

            note = QLabel("(autostretch always computed from first image)")
            note.setStyleSheet("color: grey; font-style: italic;")
            sel_row.addWidget(note)

            root.addLayout(sel_row)
        else:
            self._combo = None

        # ── Sliders panel ──────────────────────────────────────────────────────
        panel = QWidget()
        panel.setFixedHeight(84)
        grid  = QVBoxLayout(panel)
        grid.setContentsMargins(4, 4, 4, 4)
        grid.setSpacing(8)

        # — Target Median row —
        tm_row = QHBoxLayout()
        tm_row.setSpacing(8)

        self._tm_label = QLabel(f"Target Median: {self._target_bg:.3f}")
        self._tm_label.setFixedWidth(185)
        tm_row.addWidget(self._tm_label)

        self._tm_slider = QSlider(Qt.Orientation.Horizontal)
        self._tm_slider.setRange(1, 99)
        self._tm_slider.setValue(int(self._target_bg * 100))
        self._tm_slider.setTickInterval(10)
        self._tm_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._tm_slider.valueChanged.connect(self._on_tm_changed)
        tm_row.addWidget(self._tm_slider, stretch=1)

        tm_reset = QPushButton("Reset")
        tm_reset.setFixedWidth(55)
        tm_reset.setToolTip(f"Reset to default ({DEFAULT_TARGET_BG})")
        tm_reset.clicked.connect(lambda: self._tm_slider.setValue(int(DEFAULT_TARGET_BG * 100)))
        tm_row.addWidget(tm_reset)

        grid.addLayout(tm_row)

        # — Sigma row —
        sg_row = QHBoxLayout()
        sg_row.setSpacing(8)

        self._sg_label = QLabel(f"Sigma:          {self._sigma:.2f}")
        self._sg_label.setFixedWidth(185)
        sg_row.addWidget(self._sg_label)

        self._sg_slider = QSlider(Qt.Orientation.Horizontal)
        self._sg_slider.setRange(-500, 0)
        self._sg_slider.setValue(int(self._sigma * 100))
        self._sg_slider.setTickInterval(50)
        self._sg_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._sg_slider.valueChanged.connect(self._on_sg_changed)
        sg_row.addWidget(self._sg_slider, stretch=1)

        sg_reset = QPushButton("Reset")
        sg_reset.setFixedWidth(55)
        sg_reset.setToolTip(f"Reset to default ({DEFAULT_SIGMA})")
        sg_reset.clicked.connect(lambda: self._sg_slider.setValue(int(DEFAULT_SIGMA * 100)))
        sg_row.addWidget(sg_reset)

        grid.addLayout(sg_row)

        root.addWidget(panel)

        # ── MTF parameter fields (editable) ───────────────────────────────────
        params_frame = QFrame()
        params_frame.setFrameShape(QFrame.Shape.StyledPanel)
        params_frame.setFixedHeight(36)
        params_layout = QHBoxLayout(params_frame)
        params_layout.setContentsMargins(8, 2, 8, 2)
        params_layout.setSpacing(12)

        params_title = QLabel("Autostretch parameters:")
        params_title.setStyleSheet("font-weight: bold;")
        params_layout.addWidget(params_title)

        _validator = QDoubleValidator(0.0, 1.0, 8)
        _validator.setNotation(QDoubleValidator.Notation.StandardNotation)

        def _make_param_field(label_text: str) -> tuple[QLabel, QLineEdit]:
            lbl = QLabel(label_text)
            params_layout.addWidget(lbl)
            field = QLineEdit("—")
            field.setValidator(_validator)
            field.setFixedWidth(100)
            field.setAlignment(Qt.AlignmentFlag.AlignRight)
            field.setToolTip(
                "Edit manually and press Enter (or Tab away) to apply.\n"
                "Values must be in [0, 1].\n"
                "Only available when channels are linked."
            )
            params_layout.addWidget(field)
            return lbl, field

        self._lo_lbl,  self._lo_field  = _make_param_field("Lo:")
        self._mid_lbl, self._mid_field = _make_param_field("Mid:")
        self._hi_lbl,  self._hi_field  = _make_param_field("Hi:")

        self._lo_field.editingFinished.connect(
            lambda: self._on_param_field_edited("lo", self._lo_field)
        )
        self._mid_field.editingFinished.connect(
            lambda: self._on_param_field_edited("mid", self._mid_field)
        )
        self._hi_field.editingFinished.connect(
            lambda: self._on_param_field_edited("hi", self._hi_field)
        )

        params_layout.addStretch()
        root.addWidget(params_frame)

        # ── Bottom button row ──────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(4, 0, 4, 4)
        btn_row.setSpacing(8)

        recalc_btn = QPushButton("Recalculate Autostretch")
        recalc_btn.setToolTip(
            "Recompute lo/mid/hi from the first (reference) image "
            "using the current Target Median and Sigma"
        )
        recalc_btn.clicked.connect(self._on_recalculate)
        btn_row.addWidget(recalc_btn)

        refresh_btn = QPushButton("Refresh from Siril")
        refresh_btn.setToolTip(
            "Reload image data from Siril and re-apply the existing lo/mid/hi parameters "
            "(does not recalculate autostretch)"
        )
        refresh_btn.clicked.connect(self._on_refresh)
        btn_row.addWidget(refresh_btn)

        # ── Link Channels checkbox ─────────────────────────────────────────────
        self.chk_linked = QCheckBox("Link Channels")
        self.chk_linked.setChecked(self._linked)
        self.chk_linked.setToolTip(
            "Linked (checked): one shared set of lo/mid/hi parameters derived\n"
            "from the average statistics of all channels.\n\n"
            "Unlinked (unchecked): each channel gets its own independent\n"
            "lo/mid/hi values based on its individual statistics.\n"
            "Useful for revealing colour casts or balancing narrowband RGB."
        )
        self.chk_linked.toggled.connect(self._on_linked_toggled)
        #  Linked/unlinked has no meaning for a mono image (single channel).
        # Disable the checkbox to avoid confusing the user rather than silently
        # ignoring the toggle.
        if self._is_mono():
            self.chk_linked.setEnabled(False)
            self.chk_linked.setToolTip("Not applicable for mono images.")
        btn_row.addWidget(self.chk_linked)

        btn_row.addStretch()

        fit_btn = QPushButton("Scale to Fit")
        fit_btn.setFixedWidth(100)
        fit_btn.setToolTip("Scale image to fit the window (or double-click the image)")
        fit_btn.clicked.connect(lambda: self.img_label._fit())
        btn_row.addWidget(fit_btn)

        one_btn = QPushButton("1:1")
        one_btn.setFixedWidth(55)
        one_btn.setToolTip("Show image at 100% (1:1)")
        one_btn.clicked.connect(lambda: self.img_label._one_to_one())
        btn_row.addWidget(one_btn)

        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(70)
        close_btn.setToolTip("Close this window")
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)

        self.chk_ontop = QCheckBox("On Top")
        self.chk_ontop.setChecked(True)
        self.chk_ontop.setToolTip("Keep window above other windows")
        self.chk_ontop.toggled.connect(self.toggle_ontop)
        btn_row.addWidget(self.chk_ontop)

        root.addLayout(btn_row)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Loading …")

    # ── Slider / combo callbacks ───────────────────────────────────────────────

    def _on_tm_changed(self, value: int):
        self._target_bg = value / 100.0
        self._tm_label.setText(f"Target Median: {self._target_bg:.3f}")
        self._redraw_timer.start()

    def _on_sg_changed(self, value: int):
        self._sigma = value / 100.0
        self._sg_label.setText(f"Sigma:          {self._sigma:.2f}")
        self._redraw_timer.start()

    def _on_image_selected(self, index: int):
        self._active_idx = index
        self._apply_and_display(fit_view=False)

    def _on_linked_toggled(self, checked: bool):
        """Switch between linked and unlinked channel stretching."""
        self._linked = checked
        # Immediately recalculate so the display reflects the new mode
        self._recalculate_and_display(fit_view=False)

    # ── Param field callbacks ──────────────────────────────────────────────────

    def _on_param_field_edited(self, which: str, field: QLineEdit):
        """
        Called when the user finishes editing a lo/mid/hi field.
        Only active in linked mode — in unlinked mode the fields are read-only.
        """
        if not self._linked:
            #  Fields are already setReadOnly(True) in unlinked mode so
            # editingFinished should not normally fire. This guard is a safety net
            # in case the signal is triggered programmatically.
            return

        text = field.text().strip().replace(",", ".")
        try:
            value = float(text)
        except ValueError:
            self._update_param_fields()
            return

        value = max(0.0, min(1.0, value))

        # In linked mode all channels share the same params; update ch 0 tuple
        lo, mid, hi = self._channel_params[0]
        if which == "lo":
            lo = value
        elif which == "mid":
            mid = value
        elif which == "hi":
            hi = value

        c = self._ref_image.shape[0]
        self._channel_params = [(lo, mid, hi)] * c

        self._update_param_fields()
        self._apply_and_display(fit_view=False)

    def toggle_ontop(self, checked: bool):
        flags = self.windowFlags()
        if checked:
            self.setWindowFlags(flags | Qt.WindowType.WindowStaysOnTopHint)
        else:
            self.setWindowFlags(flags & ~Qt.WindowType.WindowStaysOnTopHint)
        self.show()

    def _on_recalculate(self):
        self._recalculate_and_display()

    def _on_refresh(self):
        try:
            if self._refresh_callback is not None:
                new_images = self._refresh_callback()
                if not new_images:
                    self.status.showMessage("Refresh returned no images.")
                    return
                self._images = [
                    (name, _normalise_image(data)) for name, data in new_images
                ]
                self._active_idx = min(self._active_idx, len(self._images) - 1)
                if self._combo is not None:
                    self._combo.blockSignals(True)
                    self._combo.clear()
                    for name, _ in self._images:
                        self._combo.addItem(name)
                    self._combo.setCurrentIndex(self._active_idx)
                    self._combo.blockSignals(False)

            elif self._siril_conn is not None:
                img = self._siril_conn.get_image()
                if img is None:
                    self.status.showMessage("Error: no image returned from Siril.")
                    return
                data = _normalise_image(img.data)
                try:
                    name = self._siril_conn.get_image_filename() or self._images[0][0]
                except Exception:
                    name = self._images[0][0]
                self._images[0] = (name, data)
                if self._combo is not None:
                    self._combo.blockSignals(True)
                    self._combo.setItemText(0, name)
                    self._combo.blockSignals(False)

            else:
                self.status.showMessage("No refresh source configured.")
                return

            self.status.showMessage("Refreshed — applying existing parameters …")
            QApplication.processEvents()
            self._apply_and_display(fit_view=False)

        except Exception as exc:
            self.status.showMessage(f"Refresh error: {exc}")
            raise

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _recalculate_and_display(self, fit_view: bool = False):
        """Recompute channel_params from the reference image, then render."""
        self._channel_params = compute_autostretch_params(
            self._ref_image, self._target_bg, self._sigma, linked=self._linked
        )
        self._apply_and_display(fit_view=fit_view)

    def _apply_and_display(self, fit_view: bool = False):
        """Render the active image using the stored per-channel params."""
        display = apply_stretch(self._active_image, self._channel_params)
        self.img_label.load_uint8(display, preserve_view=not fit_view)
        self._update_param_fields()
        #  In unlinked mode, params are per-channel stats of the reference
        # image. Applying them to a different active image may be misleading since
        # the two images can have very different per-channel distributions.
        # Flag this combination explicitly in the status bar.
        ref_note = (
            f"  |  stretch from: {self._images[0][0]}"
            if len(self._images) > 1 and self._active_idx != 0
            else ""
        )
        if not self._linked and len(self._images) > 1 and self._active_idx != 0:
            ref_note += "  ⚠ unlinked params from ref image — channel stats may differ"
        link_note = "linked" if self._linked else "unlinked"
        self.status.showMessage(
            f"{self._image_info_str()}{ref_note}  |  "
            f"target={self._target_bg:.3f}  sigma={self._sigma:.2f}  |  "
            f"channels: {link_note}  |  "
            "Scroll to zoom · Drag to pan · Double-click to fit"
        )

    def _update_param_fields(self):
        """
        Push current params into the editable fields.

        - Linked mode  : fields show the shared lo/mid/hi and are editable.
        - Unlinked mode: fields show ch0 values prefixed with "Ch0:" as a hint
                         and are made read-only (editing individual channels via
                         the UI is not supported; recalculation is per-channel).
        - Mono image   : linked/unlinked distinction is meaningless; always
                         show the single channel's values as editable.
        """
        lo, mid, hi = self._channel_params[0]
        is_mono     = self._is_mono()
        editable    = self._linked or is_mono

        for field, value in (
            (self._lo_field,  lo),
            (self._mid_field, mid),
            (self._hi_field,  hi),
        ):
            field.blockSignals(True)
            field.setText(f"{value:.6f}")
            field.setReadOnly(not editable)
            field.setStyleSheet(
                "" if editable else "color: grey; background: transparent;"
            )
            field.blockSignals(False)

        # Update labels to signal per-channel mode
        if not is_mono and not self._linked:
            self._lo_lbl.setText("Lo (ch0):")
            self._mid_lbl.setText("Mid (ch0):")
            self._hi_lbl.setText("Hi (ch0):")
        else:
            self._lo_lbl.setText("Lo:")
            self._mid_lbl.setText("Mid:")
            self._hi_lbl.setText("Hi:")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Custom Autostretch Preview")

    siril_conn = s.SirilInterface()
    try:
        siril_conn.connect()

        if not siril_conn.is_image_loaded():
            err = QMainWindow()
            err.setWindowTitle("Custom Autostretch Preview")
            lbl = QLabel("No image is currently open in Siril.\nPlease load an image first.")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            err.setCentralWidget(lbl)
            err.resize(400, 150)
            err.show()
            sys.exit(app.exec())

        img = siril_conn.get_image()
        if img is None:
            sys.exit(1)

        try:
            name = siril_conn.get_image_filename() or "Image"
        except Exception:
            name = "Image"

        images = [(name, img.data)]

        win = PreviewWindow(images, siril_conn=siril_conn)
        win.show()
        sys.exit(app.exec())

    finally:
        siril_conn.disconnect()


if __name__ == "__main__":
    main()