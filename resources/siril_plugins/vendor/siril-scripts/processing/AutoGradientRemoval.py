# SPDX-License-Identifier: GPL-3.0-or-later
#
# AutoGradientRemoval for Siril
# Cyril Richard, 2026
#
# Version 1.0.0
# 1.0.0 Initial release
#
"""
AutoGradientRemoval for Siril
=============================

Automatic background/gradient correction. It places NO sample points: the
background is fitted on every pixel that survives an iterative robust rejection
of structures (stars, nebulae, galaxies).

Two model modes:
  * DEFAULT: a multiscale smooth surface fitted on the background pixels, with
    explicit Structure Protection (a mask of extended bright structures that
    are excluded from the fit and bridged by inpainting). Best for irregular /
    local gradients with visible background.
  * SIMPLIFIED MODEL (optional): a stiff low-degree polynomial instead. Best
    when a nebula fills the frame and the default model would hollow it out.
Common to both: iterative robust high/low rejection until convergence.

Parameters:
  Scale            relative model scale (1-10), higher = smoother
  Smoothness       extra regularisation of the model
  Structure protection: threshold + amount
  Simplified model + degree
  Mode             subtract (additive) or divide (vignetting/flat)

Command-Line Use
----------------
    pyscript AutoGradientRemoval.py [-scale F] [-smoothness F] [-noprotect]
        [-protect_threshold F] [-protect_amount F] [-simplified [-degree N]]
        [-downsample N] [-mode subtract|divide]
"""

import sys
import re
import argparse
import threading
import concurrent.futures
import sirilpy as s
s.ensure_installed("PyQt6")
import numpy as np

from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QLabel, QDoubleSpinBox, QComboBox, QSlider,
                             QCheckBox, QPushButton, QGroupBox, QMessageBox, QFrame)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

VERSION = "1.0.0"

if not s.check_module_version(">=0.7.41"):
    print("Error: requires sirilpy version 0.7.41 or higher")
    sys.exit(1)


# ==============
# Numerical core
# ==============

def _box1d(a, r, axis):
    """One separable box blur of radius r along `axis`, via running sums."""
    if r < 1:
        return a
    n = a.shape[axis]
    w = 2 * r + 1
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r, r)
    ap = np.pad(a, pad, mode="edge")
    cs = np.cumsum(ap, axis=axis)
    z = np.zeros_like(np.take(cs, [0], axis=axis))
    cs = np.concatenate([z, cs], axis=axis)
    hi = [slice(None)] * a.ndim; hi[axis] = slice(w, w + n)
    lo = [slice(None)] * a.ndim; lo[axis] = slice(0, n)
    return (cs[tuple(hi)] - cs[tuple(lo)]) / w


def _lowpass(img, r, passes=3):
    """Gaussian-like separable low-pass = `passes` box blurs."""
    a = img.astype(np.float64, copy=False)   # _box1d returns fresh arrays anyway
    for _ in range(passes):
        a = _box1d(a, r, axis=1)
        a = _box1d(a, r, axis=0)
    return a


def _normalized_lowpass(img, mask, r, passes=3, eps=1e-6):
    """Masked low-pass: smooth `img` using only pixels where mask is True."""
    m = mask.astype(np.float64)
    num = _lowpass(img * m, r, passes)
    den = _lowpass(m, r, passes)
    return num / np.maximum(den, eps)


def _mad_sigma(x):
    """Robust (median, sigma) from the Median Absolute Deviation."""
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return med, 1.4826 * mad + 1e-12


def _poly_basis(h, w, degree):
    """Tensor polynomial basis terms x^i*y^j (i+j<=degree) on [-1,1]^2."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    xn = xx / max(1, w - 1) * 2.0 - 1.0
    yn = yy / max(1, h - 1) * 2.0 - 1.0
    terms = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            terms.append((xn ** i) * (yn ** j))
    return terms


def _poly_fit(ch, mask, terms):
    """Least-squares fit of the polynomial basis over masked pixels."""
    A = np.stack([t[mask] for t in terms], axis=1)
    b = ch[mask]
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    model = np.zeros(ch.shape, dtype=np.float64)
    for c, t in zip(coef, terms):
        model += c * t
    return model


def _inpaint_lowpass(img, mask, radius, n_fill=10, passes=2):
    """Robust smooth fill that bridges arbitrarily large rejected holes.

    Harmonic-style inpainting: repeatedly low-pass then restore the kept
    pixels. Each pass diffuses background ~radius into the holes, so n_fill
    passes bridge holes up to ~n_fill*radius wide -- unlike a single
    normalized convolution which collapses to ~0 inside big holes.
    """
    if not mask.any():
        return np.zeros(img.shape, dtype=np.float64)
    if mask.all():                       # no holes: the fill loop is a no-op
        return _lowpass(img, radius, passes)
    known_mean = float(img[mask].mean())
    filled = np.where(mask, img, known_mean).astype(np.float64)
    for _ in range(n_fill):
        sm = _lowpass(filled, radius, passes)
        filled = np.where(mask, img, sm)
    return _lowpass(filled, radius, passes)


def _structure_mask(residual, model_radius, protect_threshold, protect_amount):
    """Spatially-coherent mask of EXTENDED bright structures to protect.

    Detect pixels brighter than the model by > protect_threshold, then grow
    the detection so the whole structure (with its dim wings) is covered.
    Returns True where a structure is to be excluded from the fit.
    """
    det = (residual > protect_threshold).astype(np.float64)
    if det.max() == 0:
        return np.zeros(residual.shape, dtype=bool)
    grow_r = max(1, int(round(model_radius * (0.5 + protect_amount))))
    grown = _lowpass(det, grow_r, passes=2)
    cutoff = (1.0 - protect_amount) * 0.5 + 1e-3
    return grown > cutoff


def estimate_background(ch, radius, smoothness=0.4, high_k=2.0, low_k=4.0,
                        protect=True, protect_threshold=0.05, protect_amount=0.5,
                        simplified=False, degree=2, n_iter=20, passes=3,
                        log=print):
    """Iterative background model for one 2D channel.

    Default: multiscale smooth surface fitted on the background pixels, with
    structure protection. simplified=True swaps it for a stiff polynomial of
    the given degree (the simplified model, best for nebulae that fill the
    frame). Common to both: robust high/low rejection + convergence.
    """
    radius = max(1, int(round(radius)))
    terms = _poly_basis(ch.shape[0], ch.shape[1], degree) if simplified else None

    def fit(keep):
        if simplified:
            return _poly_fit(ch, keep, terms)
        return _inpaint_lowpass(ch, keep, radius, passes=passes)

    keep = np.ones(ch.shape, dtype=bool)
    model = fit(keep)
    prev = keep.sum()
    min_keep = max(16, int(0.02 * keep.size))

    for it in range(n_iter):
        residual = ch - model
        ref = residual[keep] if keep.any() else residual.ravel()
        med, sigma = _mad_sigma(ref)
        new_keep = (residual <= med + high_k * sigma) & \
                   (residual >= med - low_k * sigma)
        if protect:
            struct = _structure_mask(residual - med, radius,
                                     protect_threshold, protect_amount)
            new_keep &= ~struct
        if new_keep.sum() < min_keep:            # never empty the fit set
            thr = np.percentile(residual, 100 * min_keep / residual.size)
            new_keep = residual <= thr
        model = fit(new_keep)

        kept = int(new_keep.sum())
        change = abs(kept - prev) / keep.size
        log(f"  iteration {it + 1}: {100 * kept / keep.size:.1f}% kept")
        keep = new_keep
        prev = kept
        if it > 0 and change < 1e-4:
            log("  converged")
            break
    else:
        log("  maximum iterations reached")

    if smoothness > 0:
        model = _lowpass(model, max(1, int(round(radius * smoothness))), passes)
    return model


def _downsample(img, f):
    """Area-average downsample by integer factor f (block mean)."""
    if f <= 1:
        return img.astype(np.float64, copy=True)
    h, w = img.shape
    hh, ww = (h // f) * f, (w // f) * f
    return img[:hh, :ww].reshape(hh // f, f, ww // f, f).mean(axis=(1, 3))


def _resize_bilinear(img, oh, ow):
    """Bilinear resize of a 2D array to (oh, ow)."""
    h, w = img.shape
    if (h, w) == (oh, ow):
        return img.astype(np.float64, copy=True)
    ys = np.linspace(0, h - 1, oh)
    xs = np.linspace(0, w - 1, ow)
    y0 = np.floor(ys).astype(int); y1 = np.minimum(y0 + 1, h - 1)
    x0 = np.floor(xs).astype(int); x1 = np.minimum(x0 + 1, w - 1)
    wy = (ys - y0)[:, None]; wx = (xs - x0)[None, :]
    Ia = img[np.ix_(y0, x0)]; Ib = img[np.ix_(y0, x1)]
    Ic = img[np.ix_(y1, x0)]; Id = img[np.ix_(y1, x1)]
    top = Ia * (1 - wx) + Ib * wx
    bot = Ic * (1 - wx) + Id * wx
    return top * (1 - wy) + bot * wy


def correct_channel(ch, scale, smoothness, downsample, mode,
                    protect=True, protect_threshold=0.05, protect_amount=0.5,
                    simplified=False, degree=2, log=print):
    """Gradient-correct one 2D channel. Returns (corrected, background)."""
    h, w = ch.shape
    small = _downsample(ch, downsample)
    # Scale is a RELATIVE scale: higher = smoother model.
    # Map it to a smoothing radius as a fraction of the image (resolution
    # independent): scale 5 -> ~5% of the smaller image dimension.
    radius = max(1, int(round(scale / 100.0 * min(small.shape))))
    model_small = estimate_background(
        small, radius, smoothness=smoothness, protect=protect,
        protect_threshold=protect_threshold, protect_amount=protect_amount,
        simplified=simplified, degree=degree, log=log)
    bg = _resize_bilinear(model_small, h, w)
    level = float(np.median(bg))
    if mode == "divide":
        corrected = ch / np.maximum(bg, 1e-6) * level
    else:
        corrected = ch - bg + level
    return corrected, bg


def correct_image(image, scale, smoothness, downsample, mode, protect=True,
                  protect_threshold=0.05, protect_amount=0.5,
                  simplified=False, degree=2, log=print):
    """Handle mono (H,W) or color (H,W,3). Returns (corrected, background)."""
    kw = dict(protect=protect, protect_threshold=protect_threshold,
              protect_amount=protect_amount, simplified=simplified,
              degree=degree)
    if image.ndim == 2:
        return correct_channel(image, scale, smoothness, downsample, mode,
                               log=log, **kw)
    out = np.empty_like(image, dtype=np.float64)
    bg = np.empty_like(image, dtype=np.float64)
    n = image.shape[2]

    # The per-channel work is heavy NumPy (box blur, lstsq, medians) that
    # releases the GIL, so running the channels in threads overlaps them
    # instead of doing them one after another. The functions are pure (no
    # shared state), so this is thread-safe; only `log` touches Siril and is
    # serialised behind a lock.
    log_lock = threading.Lock()

    def process(c):
        def clog(msg):
            with log_lock:
                log(f"[ch {c + 1}/{n}] {msg}")
        return c, correct_channel(image[..., c], scale, smoothness,
                                  downsample, mode, log=clog, **kw)

    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        for c, (corr, bgc) in ex.map(process, range(n)):
            out[..., c] = corr
            bg[..., c] = bgc
    return out, bg


# ==========================================================================
# Siril interface
# ==========================================================================

class ProcessingThread(QThread):
    finished = pyqtSignal()
    progress = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, processor):
        super().__init__()
        self.processor = processor

    def run(self):
        try:
            self.processor.run_correction(self.progress.emit)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class AutoGradientRemovalInterface:
    def __init__(self, siril, app=None, cli_args=None):
        self.siril = siril
        self.app = app
        self.cli_call = self.siril.is_cli() and len(sys.argv) > 1
        self.cli_args = cli_args

        # parameters (sensible defaults)
        self.scale = 5.0            # relative scale, higher = smoother
        self.smoothness = 1.0
        self.downsample = 4
        self.mode = "subtract"
        self.protect = True         # Structure Protection
        self.protect_threshold = 0.05
        self.protect_amount = 0.5
        self.simplified = False     # Simplified Model (stiff polynomial)
        self.degree = 2

        self.corrected_image = None
        self.background = None
        self.processing_thread = None
        self._channel_lines = {}    # per-channel status lines (parallel logs)

        if not self.siril.connected:
            try:
                self.siril.connect()
            except s.SirilConnectionError:
                self._fatal("Failed to connect to Siril")
                return

        try:
            self.siril.cmd("requires", "1.4.0-beta3")
        except s.CommandError:
            self._fatal("Siril version requirement not met (needs 1.4.0-beta3+)")
            return

        # check an image is available now (it is re-fetched on every Apply)
        self.originally_mono = True
        self.originally_16bit = False
        if self._fetch_current_image() is None:
            self._fatal("No image is loaded in Siril")
            return

        if self.cli_call:
            self.scale = cli_args.scale
            self.smoothness = cli_args.smoothness
            self.downsample = cli_args.downsample
            self.mode = cli_args.mode
            self.protect = not cli_args.noprotect
            self.protect_threshold = cli_args.protect_threshold
            self.protect_amount = cli_args.protect_amount
            self.simplified = cli_args.simplified
            self.degree = cli_args.degree
            self.siril.log(
                f"AutoGradientRemoval: scale={self.scale}, "
                f"smoothness={self.smoothness}, protect={self.protect} "
                f"(thr={self.protect_threshold}, amt={self.protect_amount}), "
                f"simplified={self.simplified} (deg={self.degree}), "
                f"mode={self.mode}")
            self.run_correction(self.siril.log)
        elif app:
            try:
                self.create_widgets()
                self.siril.log("AutoGradientRemoval: window opened")
            except Exception as e:
                self.siril.log(f"AutoGradientRemoval: failed to open window: {e}")
                raise
        else:
            print("Error: not called from the CLI and no app created")

    # ----- helpers --------------------------------------------------------
    def _fatal(self, msg):
        if self.cli_call or self.app is None:
            print(f"Error: {msg}", file=sys.stderr)
        else:
            QMessageBox.critical(None, "Error", msg)
        try:
            self.siril.disconnect()
        except Exception:
            pass

    def _fetch_current_image(self):
        """Fetch the image CURRENTLY loaded in Siril as an HWC/HW float array.

        Helper (not a Siril API call): it uses the sirilpy API is_image_loaded()
        and get_image_pixeldata(). Called on every Apply so changing the image
        in Siril (with the window open) is picked up instead of reusing a stale
        cached frame. Updates the mono/16-bit flags. Returns None if no image.
        """
        try:
            if not self.siril.is_image_loaded():
                self.image = None
                return None
        except s.SirilError:
            pass                              # fall back to the pixeldata fetch
        img = self.siril.get_image_pixeldata()
        if img is None:
            self.image = None
            return None
        if len(img.shape) == 3:              # Siril color is CHW
            img = img.transpose(1, 2, 0)
        self.originally_mono = len(img.shape) == 2
        self.originally_16bit = img.dtype == np.uint16
        if self.originally_16bit:
            img = img.astype(np.float32) / 65535.0
        self.image = img.astype(np.float32)
        self.siril.log(f"AutoGradientRemoval: image loaded "
                       f"({'mono' if self.originally_mono else 'color'}, "
                       f"shape {self.image.shape})")
        return self.image

    def _to_siril(self, hwc):
        """Convert an HWC/HW float image back to Siril's expected layout/dtype."""
        out = hwc
        if self.originally_16bit:
            out = (np.clip(out, 0, 1) * 65535).astype(np.uint16)
        else:
            out = out.astype(np.float32)     # Siril needs float32 (not float64)
        if self.originally_mono:
            return out if out.ndim == 2 else out[..., 0]
        return out.transpose(2, 0, 1)        # HWC -> CHW

    # ----- processing -----------------------------------------------------
    def run_correction(self, progress):
        progress("Loading current image...")
        if self._fetch_current_image() is None:   # re-fetch the live Siril image
            raise RuntimeError("No image is loaded in Siril")

        progress("Estimating background...")
        corrected, bg = correct_image(
            self.image.astype(np.float64), self.scale, self.smoothness,
            self.downsample, self.mode, protect=self.protect,
            protect_threshold=self.protect_threshold,
            protect_amount=self.protect_amount, simplified=self.simplified,
            degree=self.degree, log=progress)

        if not self.originally_16bit:
            corrected = np.clip(corrected, 0.0, 1.0)
        self.corrected_image = self._to_siril(corrected)
        self.background = self._to_siril(np.clip(bg, 0.0, 1.0))

        model = (f"simplified deg{self.degree}" if self.simplified
                 else f"multiscale scale{self.scale}")
        self.siril.undo_save_state(
            f"AutoGradientRemoval ({model}, smoothness={self.smoothness}, "
            f"protect={self.protect}, mode={self.mode})")
        with self.siril.image_lock():
            self.siril.set_image_pixeldata(self.corrected_image)

    # ----- GUI ------------------------------------------------------------
    def _slider_row(self, label, minv, maxv, step, value, decimals, tooltip):
        """A labelled row with a slider and an editable value box kept in sync.

        Returns (row layout, value box). The box holds the actual value; the
        slider is just a convenient way to drag it. Both share the tooltip.
        """
        row = QHBoxLayout()
        lab = QLabel(label)
        row.addWidget(lab)

        box = QDoubleSpinBox()
        box.setRange(minv, maxv)
        box.setSingleStep(step)
        box.setDecimals(decimals)
        box.setValue(value)
        # Not a fixed width: on Windows the up/down arrows are wider and would
        # clip the number. A minimum width lets the box grow to fit its value.
        box.setMinimumWidth(90)
        box.setToolTip(tooltip)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, int(round((maxv - minv) / step)))
        slider.setValue(int(round((value - minv) / step)))
        slider.setToolTip(tooltip)

        def from_slider(pos):
            box.setValue(minv + pos * step)

        def from_box(val):
            slider.blockSignals(True)
            slider.setValue(int(round((val - minv) / step)))
            slider.blockSignals(False)

        slider.valueChanged.connect(from_slider)
        box.valueChanged.connect(from_box)

        row.addWidget(slider)
        row.addWidget(box)
        box.companions = (lab, slider)        # so the whole row can be greyed out
        return row, box

    def _set_row_enabled(self, box, enabled):
        """Enable/disable a slider row built by _slider_row (box + label + slider)."""
        box.setEnabled(enabled)
        for w in box.companions:
            w.setEnabled(enabled)

    def _sync_enabled(self):
        """Grey out the settings that do not apply with the current checkboxes."""
        protect = self.protect_check.isChecked()
        self._set_row_enabled(self.pthr_spin, protect)
        self._set_row_enabled(self.pamt_spin, protect)
        self._set_row_enabled(self.degree_spin, self.simpl_check.isChecked())

    def create_widgets(self):
        self.window = QMainWindow()
        self.window.setWindowTitle(f"AutoGradientRemoval - v{VERSION}")
        self.window.setMinimumSize(540, 430)

        central = QWidget()
        self.window.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(10)

        title = QLabel("Auto Gradient Removal")
        f = QFont(); f.setBold(True); f.setPointSize(14)
        title.setFont(f); title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        params = QGroupBox("Parameters")
        pl = QVBoxLayout(params)

        # Scale
        row, self.scale_spin = self._slider_row(
            "Scale:", 1.0, 10.0, 0.5, self.scale, 1,
            "Relative scale of the multiscale model. Higher = smoother "
            "(large-scale only); lower = follows more complex / local gradients.")
        pl.addLayout(row)

        # Smoothness
        row, self.smooth_spin = self._slider_row(
            "Smoothness:", 0.0, 3.0, 0.1, self.smoothness, 1,
            "Extra smoothing of the final model. Higher gives a softer, more "
            "gradual background; 0 leaves the fitted model untouched.")
        pl.addLayout(row)

        # Structure Protection
        self.protect_check = QCheckBox("Structure protection")
        self.protect_check.setChecked(self.protect)
        self.protect_check.setToolTip("Mask extended bright structures (nebulae) "
                                      "so they are not absorbed into the model.")
        pl.addWidget(self.protect_check)

        row, self.pthr_spin = self._slider_row(
            "    Protection threshold:", 0.0, 1.0, 0.005, self.protect_threshold, 3,
            "Brightness above the model at which a pixel is treated as a "
            "structure. Lower = protects more.")
        pl.addLayout(row)

        row, self.pamt_spin = self._slider_row(
            "    Protection amount:", 0.0, 1.0, 0.05, self.protect_amount, 2,
            "How far the protection mask grows around detected structures.")
        pl.addLayout(row)

        # Simplified Model (stiff polynomial; best for frame-filling nebulae)
        self.simpl_check = QCheckBox("Simplified model")
        self.simpl_check.setChecked(self.simplified)
        self.simpl_check.setToolTip("Replace the multiscale model with a stiff "
                                    "polynomial. Use it when a nebula fills the "
                                    "frame and the default model hollows it out.")
        pl.addWidget(self.simpl_check)

        row, self.degree_spin = self._slider_row(
            "    Model degree:", 1, 6, 1, self.degree, 0,
            "Polynomial degree of the simplified model. Lower = stiffer "
            "(degree 1 = plane).")
        pl.addLayout(row)

        # Downsample
        row = QHBoxLayout()
        row.addWidget(QLabel("Downsample:"))
        self.down_combo = QComboBox()
        self.down_combo.addItems(["8", "4", "2", "1"])
        self.down_combo.setCurrentText(str(self.downsample))
        self.down_combo.setToolTip("Internal working scale factor. Higher = "
                                   "faster but coarser; the background scale "
                                   "itself is unaffected.")
        row.addWidget(self.down_combo)
        pl.addLayout(row)

        # Mode
        row = QHBoxLayout()
        row.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["subtract", "divide"])
        self.mode_combo.setToolTip("subtract: additive gradient. "
                                   "divide: multiplicative (vignetting/flat).")
        row.addWidget(self.mode_combo)
        pl.addLayout(row)

        layout.addWidget(params)

        buttons = QHBoxLayout()
        self.apply_button = QPushButton("Apply")
        self.apply_button.setToolTip("Estimate the background with the current "
                                     "settings and apply the correction to the "
                                     "image loaded in Siril. Can be run again "
                                     "after changing any setting.")
        self.apply_button.clicked.connect(self.start_processing)
        buttons.addWidget(self.apply_button)

        # toggle button: switch the Siril view between result and background model
        self.view_button = QPushButton("Show background model")
        self.view_button.setCheckable(True)
        self.view_button.setEnabled(False)
        self.view_button.setToolTip("After applying, switch the Siril view "
                                    "between the corrected image and the "
                                    "background model that was removed, to "
                                    "check what the correction did.")
        self.view_button.toggled.connect(self.toggle_output)
        buttons.addWidget(self.view_button)

        buttons.addStretch()
        close_button = QPushButton("Close")
        close_button.setToolTip("Close the window and disconnect from Siril. "
                                "The last applied correction stays on the image.")
        close_button.clicked.connect(self.close_dialog)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)

        self.status = QLabel("Ready")
        self.status.setFrameStyle(QFrame.Shape.StyledPanel)
        self.status.setContentsMargins(5, 5, 5, 5)
        self.status.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.status)

        # grey out settings that depend on a checkbox, and keep them in sync
        self.protect_check.toggled.connect(self._sync_enabled)
        self.simpl_check.toggled.connect(self._sync_enabled)
        self._sync_enabled()

        self.window.show()
        self.window.raise_()
        self.window.activateWindow()

    def start_processing(self):
        self.scale = self.scale_spin.value()
        self.smoothness = self.smooth_spin.value()
        self.protect = self.protect_check.isChecked()
        self.protect_threshold = self.pthr_spin.value()
        self.protect_amount = self.pamt_spin.value()
        self.simplified = self.simpl_check.isChecked()
        self.degree = int(self.degree_spin.value())
        self.downsample = int(self.down_combo.currentText())
        self.mode = self.mode_combo.currentText()

        self.apply_button.setEnabled(False)
        self.view_button.setEnabled(False)
        self.view_button.setChecked(False)

        self.processing_thread = ProcessingThread(self)
        self.processing_thread.progress.connect(self.update_status)
        self.processing_thread.finished.connect(self.processing_finished)
        self.processing_thread.error.connect(self.processing_error)
        self.processing_thread.start()

    def processing_finished(self):
        self.apply_button.setEnabled(True)
        self.view_button.setEnabled(True)
        self.status.setText("Done")

    def processing_error(self, msg):
        self.apply_button.setEnabled(True)
        self.status.setText("Error")
        QMessageBox.critical(self.window, "Processing Error", msg)

    def update_status(self, msg):
        # The channels run in parallel, so their progress messages (prefixed
        # "[ch k/n] ...") would otherwise fight over the single status label.
        # Keep the latest line per channel and show them stacked, so all
        # channels stay readable at once. Any other message resets to one line.
        m = re.match(r"\[ch (\d+)/\d+\]", msg)
        if m:
            self._channel_lines[int(m.group(1))] = msg
            self.status.setText("\n".join(self._channel_lines[k]
                                          for k in sorted(self._channel_lines)))
        else:
            self._channel_lines = {}
            self.status.setText(msg)
        QApplication.processEvents()

    def toggle_output(self):
        if self.corrected_image is None:
            return
        show_bg = self.view_button.isChecked()
        self.view_button.setText("Show corrected result" if show_bg
                                 else "Show background model")
        data = self.background if show_bg else self.corrected_image
        with self.siril.image_lock():
            self.siril.set_image_pixeldata(data)

    def close_dialog(self):
        try:
            self.siril.disconnect()
        except Exception:
            pass
        if hasattr(self, "window"):
            self.window.close()


def main():
    try:
        app = QApplication(sys.argv)
        siril = s.SirilInterface()
        try:
            siril.connect()
        except s.SirilConnectionError:
            if not siril.is_cli():
                QMessageBox.critical(None, "Error", "Failed to connect to Siril")
            else:
                print("Failed to connect to Siril")
            return

        if siril.is_cli() and len(sys.argv) > 1:
            parser = argparse.ArgumentParser(description="AutoGradientRemoval")
            parser.add_argument("-scale", type=float, default=5.0,
                                help="relative model scale, 1-10 "
                                     "(higher = smoother)")
            parser.add_argument("-smoothness", type=float, default=1.0)
            parser.add_argument("-noprotect", action="store_true",
                                help="disable structure protection")
            parser.add_argument("-protect_threshold", type=float, default=0.05)
            parser.add_argument("-protect_amount", type=float, default=0.5)
            parser.add_argument("-simplified", action="store_true",
                                help="use the stiff polynomial (Simplified Model)")
            parser.add_argument("-degree", type=int, default=2,
                                help="polynomial degree for the simplified model")
            parser.add_argument("-downsample", type=int, default=4,
                                choices=[1, 2, 4, 8])
            parser.add_argument("-mode", choices=["subtract", "divide"],
                                default="subtract")
            args = parser.parse_args()
            processor = AutoGradientRemovalInterface(siril, app, cli_args=args)
        else:
            # keep a reference: otherwise the instance (and its window) is
            # garbage-collected before the event loop shows it
            processor = AutoGradientRemovalInterface(siril, app)
            app.exec()
    except Exception as e:
        print(f"Error initializing application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
