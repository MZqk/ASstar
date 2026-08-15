# SPDX-License-Identifier: GPL-3.0-or-later
#
# NarrowbandNormalization for Siril
# Yannick Dutertre (Cuiv), 2026
# cuivlazygeek@gmail.com
#
# Built using the math from Bill Blanshan & Mike Cranfield's 
# "NarrowbandNormalization" script as clean room implementation,
# with permission from Bill Blanshan.
#
# Coded with the help of Claude AI
#
# Original script: https://cosmicphotons.com/scripts/
#
# Version 1.0.5
# 1.0.0 Initial release
# 1.0.1 Housekeeping for merge request
# 1.0.2 RAM usage optimization
# 1.0.3 Fix preview too bright for some images
# 1.0.4 Match AstroColorMixer preview resolution and update debounce
# 1.0.5 Normalize preview pixels and display bounds to the same range

"""
NarrowbandNormalization for Siril
==================================

A Siril python script that ports the channel-normalization mathematics of
Bill Blanshan & Mike Cranfield's "NarrowbandNormalization" PixInsight
process/PixelMath script (HOSNormalization, cosmicphotons.com) to Siril,
with a PyQt6 GUI modeled on the PixInsight process dialog.

This supplements the Veralux Alchemy script, which is designed for OSC
images taken with dual band filters only. In contrast, this script allows
easy narrowband normalization on monochrome SHO combined images and on
OSC dual band images.

Unlike Veralux Alchemy, this script should be used on stretched data.
"""

import sys
import numpy as np

import sirilpy as s
from sirilpy import SirilConnectionError

s.ensure_installed("PyQt6")
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QComboBox, QSlider, QDoubleSpinBox, QPushButton,
    QMessageBox, QFrame, QScrollArea
)

VERSION = "1.0.5"


# ========================================================================
# CORE MATH ENGINE  (pure numpy -- no Qt / Siril dependency, unit-testable)
# ========================================================================

# Epsilon for divide-by-zero guards. Sized for float32 precision (~1.19e-7
# machine epsilon) rather than float64's ~2.2e-16 -- now that the pipeline
# runs in float32 throughout (see process_image), a 1e-12 threshold would be
# finer than float32 can even represent near typical operating magnitudes,
# making the guard effectively a no-op.
_EPS = 1e-6


def _mtf(m, x):
    """PixInsight-style midtones transfer function, m is a scalar."""
    x = np.asarray(x, dtype=np.float32)
    if abs(m - 0.5) < 1e-9:
        return x.copy()
    denom = (2.0 * m - 1.0) * x - m
    denom = np.where(np.abs(denom) < _EPS, np.copysign(_EPS, denom), denom)
    return ((m - 1.0) * x) / denom


def _rescale(x, lo, hi):
    if abs(hi - lo) < _EPS:
        return np.clip(x - lo, 0.0, 1.0)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _normalize_range(data):
    data = np.asarray(data, dtype=np.float32)
    if data.size and float(np.nanmax(data)) > 1.5:
        return data / 65535.0
    return data


def _channel_stats(ch, blackpoint):
    """Returns (M, E0) exactly as the PixelMath M / E0 expressions do.
    Reductions (mean) explicitly accumulate in float64 even though ch
    itself is float32, to avoid precision loss when summing millions of
    pixels -- this costs nothing extra in memory since the result is a
    scalar, not a full-resolution array."""
    mn = float(np.min(ch))
    med = float(np.median(ch))
    M = mn + blackpoint * (med - mn)
    mean = float(np.mean(ch, dtype=np.float64))
    adev = float(np.mean(np.abs(ch - mean), dtype=np.float64))       # PixelMath adev()
    E0 = adev / 1.2533 + mean - M
    return M, E0


def _boost_factor(a_target, a_ref, boost):
    """The E1 / E4 rational boost-strength expression."""
    denom = a_target - 2.0 * a_target * a_ref + a_ref
    if abs(denom) < 1e-9:
        denom = 1e-9
    return (a_target * (1.0 - a_ref) / denom) / boost


def _normalize_channel(ch, M_ch, strength):
    """E2/E5 (rescale) + E3/E6 (screen-blended mtf stretch)."""
    rescaled = _rescale(ch, M_ch, 1.0)
    stretched = _mtf(strength, rescaled)
    floor_part = np.minimum(ch, M_ch)
    out = 1.0 - (1.0 - stretched) * (1.0 - floor_part)
    return np.clip(out, 0.0, 1.0)


# --- sRGB <-> XYZ <-> LAB, used only for the "Lightness" feature ---

def _srgb_to_linear(c):
    c = np.clip(c, 0.0, None)
    return np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)


def _linear_to_srgb(c):
    c = np.clip(c, 0.0, None)
    return np.where(c > 0.0031308, 1.055 * (c ** (1.0 / 2.4)) - 0.055, 12.92 * c)


def _rgb_to_xyz(r, g, b):
    r1, g1, b1 = _srgb_to_linear(r), _srgb_to_linear(g), _srgb_to_linear(b)
    X = r1 * 0.4360747 + g1 * 0.3850649 + b1 * 0.1430804
    Y = r1 * 0.2225045 + g1 * 0.7168786 + b1 * 0.0606169
    Z = r1 * 0.0139322 + g1 * 0.0971045 + b1 * 0.7141733
    return X, Y, Z


def _f_lab(t):
    return np.where(t > 0.008856, np.cbrt(t), (7.787 * t) + 16.0 / 116.0)


def _f_lab_inv(t):
    return np.where(t > 0.206893, t ** 3, (t - 16.0 / 116.0) / 7.787)


def _xyz_to_lab(X, Y, Z):
    X1, Y1, Z1 = _f_lab(X), _f_lab(Y), _f_lab(Z)
    L = 116.0 * Y1 - 16.0
    a = 500.0 * (X1 - Y1)
    b = 200.0 * (Y1 - Z1)
    return L, a, b


def _lab_to_xyz(L, a, b):
    Y1 = (L + 16.0) / 116.0
    X1 = a / 500.0 + Y1
    Z1 = Y1 - b / 200.0
    return _f_lab_inv(X1), _f_lab_inv(Y1), _f_lab_inv(Z1)


def _xyz_to_rgb(X, Y, Z):
    R = X * 3.1338561 + Y * -1.6168667 + Z * -0.4906146
    G = X * -0.9787684 + Y * 1.9161415 + Z * 0.0334540
    B = X * 0.0719453 + Y * -0.2289914 + Z * 1.4052427
    return _linear_to_srgb(R), _linear_to_srgb(G), _linear_to_srgb(B)


def _cie_l_only(r, g, b):
    """CIEL(...) equivalent: just the L component, remapped like the
    Y2 formula does with (L+16)/116, ready to be used as Y2 directly."""
    X, Y, Z = _rgb_to_xyz(r, g, b)
    L, _, _ = _xyz_to_lab(X, Y, Z)
    return (L + 16.0) / 116.0


# --- synthetic green (HOO only) ---

def _synthetic_green(ha, oiii, mode, amount):
    amount = float(np.clip(amount, 0.0, 1.0))
    if mode == "Mode 1":       # linear alpha blend
        g = amount * ha + (1.0 - amount) * oiii
    elif mode == "Mode 2":     # geometric / "Cannistra-style" multiplicative blend
        g = (np.clip(ha, 0, 1) ** amount) * (np.clip(oiii, 0, 1) ** (1.0 - amount))
    else:                       # "Mode 3": screen blend
        g = 1.0 - (1.0 - amount * ha) * (1.0 - (1.0 - amount) * oiii)
    return np.clip(g, 0.0, 1.0)


# --- final tone-shaping stage (E11 / E12 / E13) ---

def _highlight_reduction(x, hl_reduction):
    hl_reduction = max(hl_reduction, 1e-3)
    m = 1.0 - 0.5 / hl_reduction
    term_a = _mtf(m, x) * x
    term_b = x * (1.0 - x)
    return term_a + term_b


def _brightness_stretch(x, brightness):
    brightness = max(brightness, 1e-3)
    return _mtf(1.0 / brightness * 0.5, x)


# ========================================================================
# PALETTE HANDLING
# ========================================================================

# Maps palette name -> {source_name: rgb_slot_index}.
# The letter order in the palette name IS the R,G,B slot assignment.
PALETTE_SLOTS = {
    "HOO": {"Ha": 0, "OIII": 2},                 # G is synthetic, no real SII
    "SHO": {"SII": 0, "Ha": 1, "OIII": 2},
    "HSO": {"Ha": 0, "SII": 1, "OIII": 2},
    "HOS": {"Ha": 0, "OIII": 1, "SII": 2},
}


def process_image(data, params):
    """
    data   : numpy array, shape (H, W, 3), float in [0, 1], channels already
             arranged in R,G,B slots matching the chosen palette's letter
             order (e.g. for 'SHO' slot 0 is SII, slot 1 is Ha, slot 2 is OIII).
    params : dict with keys:
             palette, lightness, blend_mode, blend_amount, scnr,
             oiii_boost, sii_boost, shadow_point, highlight_reduction,
             brightness
    returns: numpy array, shape (H, W, 3), float in [0, 1]
    """
    palette = params["palette"]
    slots = PALETTE_SLOTS[palette]
    data = np.asarray(data, dtype=np.float32)

    ha = data[:, :, slots["Ha"]]
    oiii = data[:, :, slots["OIII"]]
    sii = data[:, :, slots["SII"]] if "SII" in slots else None

    blackpoint = params["shadow_point"]

    # --- statistics & boost factors (reference is always Ha; the
    #     "M[1]" divisor in the original script is always the OIII M) ---
    M_ha, E0_ha = _channel_stats(ha, blackpoint)
    M_o, E0_o = _channel_stats(oiii, blackpoint)
    ref_denom = 1.0 - M_o
    if abs(ref_denom) < 1e-9:
        ref_denom = 1e-9
    A0_ha = E0_ha / ref_denom
    A0_o = E0_o / ref_denom

    E1 = _boost_factor(A0_o, A0_ha, params["oiii_boost"])
    oiii_norm = _normalize_channel(oiii, M_o, E1)

    if sii is not None:
        M_s, E0_s = _channel_stats(sii, blackpoint)
        A0_s = E0_s / ref_denom
        E4 = _boost_factor(A0_s, A0_ha, params["sii_boost"])
        sii_norm = _normalize_channel(sii, M_s, E4)
    else:
        sii_norm = None

    # --- assemble RGB into the palette's slot order ---
    out = np.empty_like(data)
    out[:, :, slots["Ha"]] = ha
    out[:, :, slots["OIII"]] = oiii_norm

    if sii is not None:
        out[:, :, slots["SII"]] = sii_norm
    else:
        green = _synthetic_green(ha, oiii_norm, params["blend_mode"], params["blend_amount"])
        # slot 1 is always the "G" slot for HOO in PALETTE_SLOTS
        out[:, :, 1] = green

    # --- SCNR (amount-based blend toward min(G, mean(R,B))) ---
    if sii is not None:
        scnr_amt = float(np.clip(params["scnr"], 0.0, 1.0))
        if scnr_amt > 0.0:
            r_slot, g_slot, b_slot = 0, 1, 2
            r_ch, g_ch, b_ch = out[:, :, r_slot], out[:, :, g_slot], out[:, :, b_slot]
            reduced = np.minimum(np.mean(np.stack([r_ch, b_ch]), axis=0), g_ch)
            out[:, :, g_slot] = (1.0 - scnr_amt) * g_ch + scnr_amt * reduced

    # --- Lightness (LAB) stage ---
    lightness = params["lightness"]
    if lightness != "Off":
        r, g, b = out[:, :, 0], out[:, :, 1], out[:, :, 2]
        X, Y, Z = _rgb_to_xyz(r, g, b)
        L, a, bb = _xyz_to_lab(X, Y, Z)
        del X, Y, Z, L  # L is never used below; free all four immediately

        if lightness == "Original":
            Y2 = _cie_l_only(data[:, :, 0], data[:, :, 1], data[:, :, 2])
        elif lightness == "Ha":
            Y2 = (ha + 0.16) / 1.16
        elif lightness == "SII" and sii is not None:
            Y2 = (sii + 0.16) / 1.16
        elif lightness == "SII" and sii is None:
            # No real SII in HOO: fall back to the raw (pre-normalization,
            # pre-blend) OIII channel, and let the caller know via a flag.
            Y2 = (oiii + 0.16) / 1.16
        else:  # "OIII"
            Y2 = (oiii + 0.16) / 1.16

        X2 = (a / 500.0) + Y2
        Z2 = Y2 - (bb / 200.0)
        del a, bb  # only needed to build X2/Z2, done with them now

        X3, Y3, Z3 = _f_lab_inv(X2), _f_lab_inv(Y2), _f_lab_inv(Z2)
        del X2, Y2, Z2

        r3, g3, b3 = _xyz_to_rgb(X3, Y3, Z3)
        del X3, Y3, Z3

        out[:, :, 0] = np.clip(r3, 0.0, 1.0)
        out[:, :, 1] = np.clip(g3, 0.0, 1.0)
        out[:, :, 2] = np.clip(b3, 0.0, 1.0)
        del r3, g3, b3

    # --- final highlight-reduction / brightness / clip ---
    out = _highlight_reduction(out, params["highlight_reduction"])
    out = _brightness_stretch(out, params["brightness"])
    out = _rescale(out, 0.0, 1.0)

    return out.astype(np.float32)


# ========================================================================
# GUI
# ========================================================================

def _to_hwc(data):
    """
    Normalize an incoming pixel array to (H, W, 3) regardless of whether
    Siril handed it over as (H, W, C) or as a planar (C, H, W) array.

    sirilpy's documentation describes FFit image *shape* as (H, W, C), but
    in practice the raw .data array can come back planar -- (C, H, W) --
    matching Siril's internal per-channel-plane storage. We detect this by
    looking for a size-3 (or size-1) axis and moving it to the end; height
    and width are essentially always much larger than 3, so this is
    unambiguous except for tiny test images.

    Returns (hwc_array, original_layout) where original_layout is "hwc" or
    "chw", so the result can be converted back before calling
    set_image_pixeldata().
    """
    data = np.asarray(data)
    if data.ndim == 2:
        return data[:, :, None], "hw"
    if data.ndim != 3:
        raise ValueError(f"Unexpected pixel data with {data.ndim} dimensions.")

    d0, d1, d2 = data.shape
    if d2 in (1, 3) and d2 != d0:
        return data, "hwc"
    if d0 in (1, 3) and d0 != d2:
        return np.moveaxis(data, 0, -1), "chw"
    # Ambiguous (e.g. a 3x3xN test array) -- assume already channel-last.
    return data, "hwc"


def _from_hwc(data, layout):
    if layout == "chw":
        return np.moveaxis(data, -1, 0)
    if layout == "hw":
        return data[:, :, 0]
    return data


def _downsample_area(data, max_dim):
    """Area-average an image proxy so reduced previews do not alias noise."""
    h, w = data.shape[:2]
    factor = max(1, int(np.ceil(max(h, w) / float(max_dim))))
    if factor == 1:
        return data
    pad_h = (-h) % factor
    pad_w = (-w) % factor
    pad_spec = ((0, pad_h), (0, pad_w)) + (((0, 0),) if data.ndim == 3 else ())
    padded = np.pad(np.asarray(data, dtype=np.float32), pad_spec, mode="edge")
    new_h = padded.shape[0] // factor
    new_w = padded.shape[1] // factor
    if data.ndim == 3:
        reduced = padded.reshape(
            new_h, factor, new_w, factor, data.shape[2]
        ).mean(axis=(1, 3), dtype=np.float32)
    else:
        reduced = padded.reshape(
            new_h, factor, new_w, factor
        ).mean(axis=(1, 3), dtype=np.float32)
    return reduced.astype(np.float32, copy=False)


def _array_to_qpixmap(data, display_range=None):
    """(H, W, 3) float array in [0,1] -> QPixmap, for display in a QLabel."""
    if display_range is not None:
        arr8 = _rescale(data, *display_range)
    else:
        arr8 = np.clip(data, 0.0, 1.0)
    arr8 = (arr8 * 255.0 + 0.5).astype(np.uint8)
    arr8 = np.ascontiguousarray(arr8)
    h, w, _ = arr8.shape
    qimg = QImage(arr8.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    # .copy() so the QImage owns its own buffer once arr8 goes out of scope
    return QPixmap.fromImage(qimg.copy())


def _make_slider_row(parent_layout, label_text, minv, maxv, default, decimals=3, step=None, on_change=None):
    """Builds a label + editable spinbox + slider row, kept in sync.
    Returns (spinbox, slider) so callers can read .value() / .value()/scale.

    If on_change is given, it's called on every genuine user interaction
    with either widget. Note this has to be wired to *both* raw signals:
    slider_to_spin/spin_to_slider update the other widget with its
    signals blocked (to avoid feedback loops), so listening only on the
    spinbox's valueChanged would silently miss slider drags entirely.
    """
    row = QHBoxLayout()
    lbl = QLabel(label_text)
    lbl.setFixedWidth(130)

    spin = QDoubleSpinBox()
    spin.setDecimals(decimals)
    spin.setRange(minv, maxv)
    spin.setSingleStep(step if step else (maxv - minv) / 100.0)
    spin.setValue(default)
    spin.setMinimumWidth(90)

    scale = 10 ** decimals
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setMinimum(int(minv * scale))
    slider.setMaximum(int(maxv * scale))
    slider.setValue(int(default * scale))

    def slider_to_spin(v):
        spin.blockSignals(True)
        spin.setValue(v / scale)
        spin.blockSignals(False)

    def spin_to_slider(v):
        slider.blockSignals(True)
        slider.setValue(int(round(v * scale)))
        slider.blockSignals(False)

    slider.valueChanged.connect(slider_to_spin)
    spin.valueChanged.connect(spin_to_slider)

    if on_change is not None:
        slider.valueChanged.connect(on_change)
        spin.valueChanged.connect(on_change)

    row.addWidget(lbl)
    row.addWidget(spin)
    row.addWidget(slider)
    parent_layout.addLayout(row)
    return spin, slider


class _ZoomPanScrollArea(QScrollArea):
    """A QScrollArea that also supports:
    - mouse-wheel zoom when the cursor is over the viewport
    - panning by holding the middle mouse button and dragging
    """

    def __init__(self, on_wheel_zoom, parent=None):
        super().__init__(parent)
        self._on_wheel_zoom = on_wheel_zoom
        self._panning = False
        self._pan_start = None
        self._scroll_start = (0, 0)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta != 0:
            self._on_wheel_zoom(delta)
            event.accept()
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = event.position().toPoint()
            self._scroll_start = (
                self.horizontalScrollBar().value(),
                self.verticalScrollBar().value(),
            )
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning and self._pan_start is not None:
            pos = event.position().toPoint()
            dx = pos.x() - self._pan_start.x()
            dy = pos.y() - self._pan_start.y()
            self.horizontalScrollBar().setValue(self._scroll_start[0] - dx)
            self.verticalScrollBar().setValue(self._scroll_start[1] - dy)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton and self._panning:
            self._panning = False
            self._pan_start = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class NarrowbandNormalizationWindow(QMainWindow):
    PREVIEW_MAX_DIM = 720          # longest side of the downsampled preview

    def __init__(self, siril):
        super().__init__()
        self.siril = siril
        self.proxy_data = None                # downsampled (H,W,3) float array for preview
        self.preview_display_range = None     # Siril min/max, used only for rendering
        self.original_pixmap_base = None      # unprocessed proxy, at proxy resolution
        self.processed_pixmap_base = None     # last processed proxy, at proxy resolution
        self.preview_zoom = 1.0
        self.show_original_active = False
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self._update_preview)
        self.setWindowTitle(f"NarrowbandNormalization - v{VERSION}")

        self._build_ui()
        self._on_palette_changed(self.palette_combo.currentText())
        self._refresh_preview_source()

    # ---- UI construction ----
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # ================= LEFT: controls =================
        controls_widget = QWidget()
        controls_widget.setFixedWidth(400)
        main = QVBoxLayout(controls_widget)

        # --- Palette ---
        g_palette = QGroupBox("Palette")
        l_palette = QVBoxLayout()
        row = QHBoxLayout()
        row.addWidget(QLabel("Palette:"))
        self.palette_combo = QComboBox()
        self.palette_combo.addItems(["HOO", "SHO", "HSO", "HOS"])
        self.palette_combo.currentTextChanged.connect(self._on_palette_changed)
        row.addWidget(self.palette_combo)
        l_palette.addLayout(row)
        g_palette.setLayout(l_palette)
        main.addWidget(g_palette)

        # --- Lightness ---
        g_light = QGroupBox("Lightness")
        l_light = QVBoxLayout()
        row = QHBoxLayout()
        row.addWidget(QLabel("Lightness:"))
        self.lightness_combo = QComboBox()
        self.lightness_combo.addItems(["Off", "Original", "Ha", "SII", "OIII"])
        self.lightness_combo.currentTextChanged.connect(self._schedule_preview_update)
        row.addWidget(self.lightness_combo)
        l_light.addLayout(row)
        g_light.setLayout(l_light)
        main.addWidget(g_light)

        # --- Synthetic green blend ---
        g_green = QGroupBox("Synthetic green blend")
        l_green = QVBoxLayout()
        row = QHBoxLayout()
        row.addWidget(QLabel("Blend mode:"))
        self.blend_mode_combo = QComboBox()
        self.blend_mode_combo.addItems(["Mode 1", "Mode 2", "Mode 3"])
        self.blend_mode_combo.currentTextChanged.connect(self._schedule_preview_update)
        row.addWidget(self.blend_mode_combo)
        l_green.addLayout(row)
        self.blend_amount_spin, self.blend_amount_slider = _make_slider_row(
            l_green, "Blend amount:", 0.0, 1.0, 0.600, on_change=self._schedule_preview_update)
        g_green.setLayout(l_green)
        main.addWidget(g_green)

        # --- Channel controls ---
        g_channels = QGroupBox("Channel controls")
        l_channels = QVBoxLayout()
        self.scnr_spin, self.scnr_slider = _make_slider_row(
            l_channels, "SCNR:", 0.0, 1.0, 0.000, on_change=self._schedule_preview_update)
        self.oiii_boost_spin, self.oiii_boost_slider = _make_slider_row(
            l_channels, "OIII boost:", 0.5, 2.0, 1.000, on_change=self._schedule_preview_update)
        self.sii_boost_spin, self.sii_boost_slider = _make_slider_row(
            l_channels, "SII boost:", 0.5, 2.0, 1.000, on_change=self._schedule_preview_update)
        g_channels.setLayout(l_channels)
        main.addWidget(g_channels)

        # --- Adjustments ---
        g_adj = QGroupBox("Adjustments")
        l_adj = QVBoxLayout()
        self.shadow_spin, self.shadow_slider = _make_slider_row(
            l_adj, "Shadow point:", 0.0, 1.0, 1.000, on_change=self._schedule_preview_update)
        self.hlreduction_spin, self.hlreduction_slider = _make_slider_row(
            l_adj, "Highlight reduction:", 0.1, 3.0, 1.000, on_change=self._schedule_preview_update)
        self.brightness_spin, self.brightness_slider = _make_slider_row(
            l_adj, "Brightness:", 0.1, 3.0, 1.000, on_change=self._schedule_preview_update)
        g_adj.setLayout(l_adj)
        main.addWidget(g_adj)

        # --- separator + buttons ---
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        main.addWidget(line)

        btn_row = QHBoxLayout()
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self._on_apply)
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self._on_reset)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        btn_row.addWidget(self.apply_btn)
        btn_row.addWidget(self.reset_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.close_btn)
        main.addLayout(btn_row)
        main.addStretch()

        root.addWidget(controls_widget)

        # ================= RIGHT: live preview =================
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)

        g_preview = QGroupBox("Preview")
        l_preview = QVBoxLayout()

        self.preview_label = QLabel("Loading preview...")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(360, 360)
        self.preview_label.setStyleSheet("background-color: #1b1b1b; color: #999;")

        self.preview_status_label = QLabel("")
        self.preview_status_label.setStyleSheet("color: #999;")
        l_preview.addWidget(self.preview_status_label)

        # Scroll area lets you pan around when zoomed in past the viewport;
        # the label is only ever added here (NOT also via l_preview.addWidget,
        # which would leave a stale, blank layout slot behind).
        self.preview_scroll = _ZoomPanScrollArea(self._on_wheel_zoom)
        self.preview_scroll.setWidget(self.preview_label)
        self.preview_scroll.setWidgetResizable(False)
        self.preview_scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_scroll.setStyleSheet("border: 1px solid #444;")
        l_preview.addWidget(self.preview_scroll, stretch=1)

        # Zoom controls
        zoom_row = QHBoxLayout()
        self.zoom_out_btn = QPushButton("-")
        self.zoom_out_btn.setFixedWidth(32)
        self.zoom_out_btn.clicked.connect(self._on_zoom_out)
        self.zoom_label = QLabel("100%")
        self.zoom_label.setFixedWidth(50)
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setFixedWidth(32)
        self.zoom_in_btn.clicked.connect(self._on_zoom_in)
        self.zoom_reset_btn = QPushButton("Reset zoom")
        self.zoom_reset_btn.clicked.connect(self._on_zoom_reset)
        zoom_row.addWidget(self.zoom_out_btn)
        zoom_row.addWidget(self.zoom_label)
        zoom_row.addWidget(self.zoom_in_btn)
        zoom_row.addStretch()
        zoom_row.addWidget(self.zoom_reset_btn)
        l_preview.addLayout(zoom_row)

        # Manual resync (e.g. after an Undo, or after loading a different
        # image in Siril) + click-and-hold "show original".
        action_row = QHBoxLayout()
        self.refresh_preview_btn = QPushButton("Refresh from Siril")
        self.refresh_preview_btn.clicked.connect(self._refresh_preview_source)
        self.show_original_btn = QPushButton("Show original (hold)")
        self.show_original_btn.pressed.connect(self._on_show_original_pressed)
        self.show_original_btn.released.connect(self._on_show_original_released)
        action_row.addWidget(self.refresh_preview_btn)
        action_row.addWidget(self.show_original_btn)
        l_preview.addLayout(action_row)

        g_preview.setLayout(l_preview)
        preview_layout.addWidget(g_preview)
        root.addWidget(preview_widget, stretch=1)

        self.resize(960, 760)

    # ---- live preview plumbing ----
    def _refresh_preview_source(self):
        """(Re-)fetch a downsampled proxy of whatever is currently loaded
        in Siril, then immediately reprocess/redraw the preview with it."""
        try:
            with self.siril.image_lock():
                img = self.siril.get_image()
                raw = np.asarray(img.data, dtype=np.float32)
                display_bounds = _normalize_range(
                    np.asarray([img.mini, img.maxi], dtype=np.float32)
                )
                if display_bounds[1] > display_bounds[0]:
                    self.preview_display_range = (
                        float(display_bounds[0]), float(display_bounds[1])
                    )
                else:
                    self.preview_display_range = None
            data, _layout = _to_hwc(raw)
            if data.ndim != 3 or data.shape[2] != 3:
                self.proxy_data = None
                self.original_pixmap_base = None
                self.preview_status_label.setText(
                    f"No 3-channel RGB image loaded (got shape {raw.shape})."
                )
                self.preview_label.setText("No preview available")
                return
            data = _normalize_range(data)
            # FITS data (which is what Siril reads) stores its first row as
            # the *bottom* of the image; Siril flips it for display. QImage
            # assumes the opposite (first row = top), so without this the
            # preview renders upside-down relative to the main Siril view.
            # This is purely a display concern -- it doesn't touch the
            # actual normalization math, which is orientation-agnostic.
            data = np.flipud(data)
            self.proxy_data = _downsample_area(data, self.PREVIEW_MAX_DIM)
            self.original_pixmap_base = _array_to_qpixmap(
                self.proxy_data, self.preview_display_range
            )
            self.preview_status_label.setText(
                f"Preview proxy: {self.proxy_data.shape[1]}x{self.proxy_data.shape[0]} "
                f"(source {raw.shape[1] if raw.ndim == 3 else '?'} px wide)"
            )
        except Exception as e:
            self.proxy_data = None
            self.original_pixmap_base = None
            self.preview_status_label.setText(f"Could not load preview: {e}")
            self.preview_label.setText("No preview available")
            return

        self._update_preview()

    def _schedule_preview_update(self, *_args):
        self.preview_timer.start(100)

    def _update_preview(self):
        if self.proxy_data is None:
            return
        try:
            params = self._collect_params()
            result = process_image(self.proxy_data, params)
            self.processed_pixmap_base = _array_to_qpixmap(
                result, self.preview_display_range
            )
        except Exception as e:
            self.processed_pixmap_base = None
            self.preview_label.setText(f"Preview error:\n{e}")
            return
        self._render_preview()

    def _render_preview(self):
        """Pick original-vs-processed and the current zoom level, and
        actually paint the QLabel."""
        base = (
            self.original_pixmap_base
            if self.show_original_active
            else self.processed_pixmap_base
        )
        if base is None or base.isNull():
            return
        target_w = max(1, int(base.width() * self.preview_zoom))
        target_h = max(1, int(base.height() * self.preview_zoom))
        scaled = base.scaled(
            target_w, target_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(scaled)
        self.preview_label.resize(scaled.size())

    # ---- zoom controls ----
    def _on_wheel_zoom(self, delta):
        # delta is QWheelEvent.angleDelta().y(); +/-120 per notch on most mice
        factor = 1.1 if delta > 0 else (1.0 / 1.1)
        self.preview_zoom = min(8.0, max(0.1, self.preview_zoom * factor))
        self._update_zoom_label()
        self._render_preview()

    def _on_zoom_in(self):
        self.preview_zoom = min(8.0, self.preview_zoom * 1.25)
        self._update_zoom_label()
        self._render_preview()

    def _on_zoom_out(self):
        self.preview_zoom = max(0.1, self.preview_zoom / 1.25)
        self._update_zoom_label()
        self._render_preview()

    def _on_zoom_reset(self):
        self.preview_zoom = 1.0
        self._update_zoom_label()
        self._render_preview()

    def _update_zoom_label(self):
        self.zoom_label.setText(f"{round(self.preview_zoom * 100)}%")

    # ---- click-and-hold "show original" ----
    def _on_show_original_pressed(self):
        self.show_original_active = True
        self._render_preview()

    def _on_show_original_released(self):
        self.show_original_active = False
        self._render_preview()

    # ---- enable/disable logic ----
    def _on_palette_changed(self, palette_text):
        is_hoo = (palette_text == "HOO")

        # Synthetic-green controls only make sense (and only exist) for HOO
        self.blend_mode_combo.setEnabled(is_hoo)
        self.blend_amount_spin.setEnabled(is_hoo)
        self.blend_amount_slider.setEnabled(is_hoo)

        # SCNR and SII boost only make sense when there is a real SII channel
        self.scnr_spin.setEnabled(not is_hoo)
        self.scnr_slider.setEnabled(not is_hoo)
        self.sii_boost_spin.setEnabled(not is_hoo)
        self.sii_boost_slider.setEnabled(not is_hoo)

        # Nicety: HOO has no real SII data, so drop that option from Lightness
        current = self.lightness_combo.currentText()
        self.lightness_combo.blockSignals(True)
        self.lightness_combo.clear()
        items = ["Off", "Original", "Ha", "OIII"] if is_hoo else ["Off", "Original", "Ha", "SII", "OIII"]
        self.lightness_combo.addItems(items)
        if current in items:
            self.lightness_combo.setCurrentText(current)
        self.lightness_combo.blockSignals(False)

        self._schedule_preview_update()

    # ---- parameter collection ----
    def _collect_params(self):
        return dict(
            palette=self.palette_combo.currentText(),
            lightness=self.lightness_combo.currentText(),
            blend_mode=self.blend_mode_combo.currentText(),
            blend_amount=self.blend_amount_spin.value(),
            scnr=self.scnr_spin.value(),
            oiii_boost=self.oiii_boost_spin.value(),
            sii_boost=self.sii_boost_spin.value(),
            shadow_point=self.shadow_spin.value(),
            highlight_reduction=self.hlreduction_spin.value(),
            brightness=self.brightness_spin.value(),
        )

    def _on_reset(self):
        self.lightness_combo.setCurrentText("Off")
        self.blend_mode_combo.setCurrentText("Mode 1")
        self.blend_amount_spin.setValue(0.600)
        self.scnr_spin.setValue(0.000)
        self.oiii_boost_spin.setValue(1.000)
        self.sii_boost_spin.setValue(1.000)
        self.shadow_spin.setValue(1.000)
        self.hlreduction_spin.setValue(1.000)
        self.brightness_spin.setValue(1.000)

    # ---- Apply: pull image from Siril, run engine, push back ----
    def _on_apply(self):
        params = self._collect_params()
        try:
            with self.siril.image_lock():
                img = self.siril.get_image()
                raw = np.asarray(img.data, dtype=np.float32)

                data, layout = _to_hwc(raw)
                self.siril.log(
                    f"NarrowbandNormalization: raw pixel array shape={raw.shape}, "
                    f"detected layout={layout}, working shape={data.shape}."
                )

                if data.ndim != 3 or data.shape[2] != 3:
                    raise ValueError(
                        "NarrowbandNormalization requires a 3-channel RGB "
                        "image with Ha/OIII(/SII) already combined into "
                        "R,G,B in the order matching the selected palette. "
                        f"Got array shape {raw.shape}."
                    )

                # Normalize to [0,1] working range if data looks like it is
                # in 16-bit integer range rather than float.
                if data.max() > 1.5:
                    data = data / 65535.0

                result = process_image(data, params)

                # Register this as an undoable step *before* pushing the
                # new pixels back, so Siril's Undo button (and the
                # Processing history) sees it like any other operation.
                self.siril.undo_save_state(
                    f"NarrowbandNormalization: palette={params['palette']}, "
                    f"lightness={params['lightness']}"
                )

                result = _from_hwc(result, layout)
                self.siril.set_image_pixeldata(result)

            self.siril.log(
                f"NarrowbandNormalization applied "
                f"(palette={params['palette']}, lightness={params['lightness']})."
            )
            self._refresh_preview_source()
        except Exception as e:
            QMessageBox.critical(self, "NarrowbandNormalization Error", str(e))
            try:
                self.siril.log(f"NarrowbandNormalization error: {e}")
            except Exception:
                pass


def main():
    siril = s.SirilInterface()
    try:
        siril.connect()
    except SirilConnectionError as e:
        print(f"Could not connect to Siril: {e}")
        sys.exit(1)

    app = QApplication(sys.argv)
    window = NarrowbandNormalizationWindow(siril)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
