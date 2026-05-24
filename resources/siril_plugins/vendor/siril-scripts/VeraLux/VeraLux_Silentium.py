##############################################
# VeraLux — Silentium
# Linear-Phase Noise Suppression Engine
# Author: Riccardo Paterniti (2025)
# Contact: info@veralux.space
##############################################
# (c) 2025 Riccardo Paterniti
# VeraLux — Silentium
# SPDX-License-Identifier: GPL-3.0-or-later
# Version 1.0.3

# Credits / Origin
# ----------------
# • Theoretical basis: Stationary Wavelet Transform (SWT) & Daubechies Wavelets
# • Logic: "Shadow Authority" adaptive masking & Relative Pedestal Detection
# • Context: Astrophotographic signal-to-noise optimization (Linear Stage)

"""
Overview
--------
A linear-phase noise suppression engine designed to separate stochastic noise
from physical signal without compromising stellar geometry or fine nebula details.

Silentium establishes a new paradigm in linear processing by decoupling the treatment
of low-signal areas (background) from high-signal structures. Through its "Shadow Authority"
logic and "Physics-Aware" architecture, it applies mathematically aggressive cleaning
to the Deep Space floor while creating an impervious protective shell around stars
and nebulosity, preserving the original photometric integrity of the data.

Core Architecture
-----------------
• Stationary Wavelet Transform (SWT):
  Utilizes Undecimated Multiscale decomposition via Daubechies (db2) wavelets.
  This Shift-Invariant approach ensures precise frequency separation with zero phase
  shift and no aliasing, providing superior texture quality compared to standard DWT.

• Photometric Signal Gating:
  The engine relies on a statistical "Signal Probability Map" derived from Linear Data
  analysis (Median + Sigma thresholds). This strictly decouples the image into
  distinct operating domains, preventing cross-contamination between background
  smoothing and structural sharpening.

• Shadow Authority & Exclusion Gate:
  Implements a strict "Winner-Take-All" logic for background cleaning. The Shadow
  Smoothness engine features a hard photometric cutoff (Exclusion Gate) that forces
  denoising aggression to zero as soon as any signal structure is detected.
  This prevents the "blurring" of highlights regardless of protection settings.

• PSF-Compensated Structural Guard:
  Detail Guard combines linear signal probability with morphological dilation and
  Seeing Compensation. It scales structural protection linearly with local FWHM,
  preventing the erosion of filaments that appear optically soft due to atmospheric
  blur, balancing the denoising pressure.

• Seeing-Adaptive Thresholding:
  Wavelet thresholds are spatially modulated by the local FWHM map. The algorithm
  automatically increases aggression in blurred areas (poor seeing) where fine detail
  is physically absent, and preserves micro-contrast in sharper zones.

• PSF-Aware Elliptical Masking:
  Unlike generic circular masks, Silentium leverages Siril's `findstar` data to
  generate rotated elliptical masks based on the real star geometry (Angle, Major/Minor axes).
  This handles coma and tracking errors perfectly, ensuring zero profile erosion.

• Loupe UX:
  Global View is static (Fit-only). Clicking zooms into 1:1 "Loupe Mode" for
  pixel-peeping. Dragging pans the Loupe.

• Shadow Report:
  Provides a statistical breakdown of noise reduction, SNR gain, and pedestal
  conservation upon completion.

Usage
-----
1. Pre-requisite: Image MUST be Linear (before stretching).
2. Setup:
   - Ensure 'Use findstar' is checked to leverage PSF protection.
   - Select 'Adaptive Noise Model' (Default ON).
3. Luminance Calibration:
   - 'Noise Intensity': Sets the global threshold for general grain reduction.
   - 'Detail Guard': Protects faint structures. Increasing this will restore
     micro-contrast in fine details without re-introducing background noise.
   - 'Shadow Smoothness': Target specific, aggressive cleaning for the background.
4. Preview Interaction:
   - Global View: Click to inspect area 1:1.
   - Loupe Mode (1:1): Drag to pan. Click (without dragging) to return to Global.
5. Process: Click PROCESS to apply to the full-resolution image.

Inputs & Outputs
----------------
Input: Linear FITS/TIFF (RGB/Mono). 16/32-bit Int or Float.
Output: Denoised Linear 32-bit Float FITS.

Compatibility
-------------
• Siril 1.3+
• Python 3.10+ (via sirilpy)
• Dependencies: sirilpy, PyQt6, numpy, scipy, PyWavelets (pywt)

License
-------
Released under GPL-3.0-or-later.
"""

import sys
import os
import time
import traceback
import numpy as np
import webbrowser

try:
    import sirilpy as s
    from sirilpy import LogColor
except ImportError:
    print("Error: sirilpy module not found. This script must be run within Siril.")
    sys.exit(1)

# Ensure deps are present
s.ensure_installed("PyQt6", "numpy", "PyWavelets", "scipy")

from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QLabel, QSlider, QPushButton, QGroupBox,
                             QMessageBox, QCheckBox, QProgressBar,
                             QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem)
from PyQt6.QtCore import Qt, QTimer, QEvent, QSettings, QPoint, QPointF
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QPainterPath, QKeyEvent, QPen

from scipy.signal import convolve2d
from scipy.ndimage import zoom, maximum_filter
import pywt

##############################################
# CONFIGURATION & DEFAULTS
##############################################

class SilentiumDefaults:
    """
    Centralized configuration for default values.
    Values are in Slider Units (0-100) or Boolean.
    This class is the SINGLE source of truth for defaults.
    """

    # Sliders
    INTENSITY = 25  # 0.50 (Slider 50)
    DETAIL_GUARD = 50  # 50%
    CHROMA = 30  # 30%
    SHADOW_SMOOTH = 10  # 10%

    # Checkboxes (Logic)
    ADAPTIVE_NOISE = True
    ENABLE_CHROMA = True
    USE_STARS = True
    AUTO_STARLESS = True

    VERSION = "1.0.3"

# ------------------------------------------------------------------------------
# VERSION HISTORY
# ------------------------------------------------------------------------------
# 1.0.3: Fixed critical crash on Undo/Redo with monochromatic images (2D/3D array mismatch).
# 1.0.2: Too much caffeine: fix coffee button stealing spacebar focus.
# 1.0.1: "Buy me a coffee" button added.
# ------------------------------------------------------------------------------

##############################################
# THEME (VeraLux)
##############################################

DARK_STYLESHEET = """
QWidget { background-color: #2b2b2b; color: #e0e0e0; font-size: 10pt; }
QToolTip { background-color: #333333; color: #ffffff; border: 1px solid #88aaff; }
QGroupBox { border: 1px solid #444444; margin-top: 5px; font-weight: bold; border-radius: 4px; padding-top: 12px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; color: #88aaff; }
QLabel { color: #cccccc; }
QCheckBox { spacing: 5px; color: #cccccc; }
QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #666666; background: #3c3c3c; border-radius: 3px; }
QCheckBox::indicator:checked { background-color: #285299; border: 1px solid #88aaff; }

/* Sliders Base */
QSlider { min-height: 24px; }
QSlider::groove:horizontal { background: #444444; height: 6px; border-radius: 3px; }
QSlider::handle:horizontal {
    background-color: #cccccc; border: 1px solid #666666;
    width: 14px; height: 14px; margin: -4px 0; border-radius: 7px;
}
QSlider::handle:horizontal:hover { background-color: #ffffff; border-color: #88aaff; }
QSlider::handle:horizontal:pressed { background-color: #ffffff; border-color: #ffffff; }

/* Generic Slider Fill (Blue) */
QSlider::sub-page:horizontal { background: #285299; border-radius: 3px; }
QSlider::add-page:horizontal { background: #444444; border-radius: 3px; }

/* Master Slider (Gold) */
QSlider#MainSlider::handle:horizontal { background-color: #ffb000; border: 1px solid #cc8800; }
QSlider#MainSlider::handle:horizontal:hover { background-color: #ffcc00; border-color: #ffffff; }
QSlider#MainSlider::groove:horizontal { background: #554400; }
QSlider#MainSlider::sub-page:horizontal { background: #cc8800; border-radius: 3px; }

/* Chroma Slider (Cyan) */
QSlider#ChromaSlider::handle:horizontal { background-color: #00cccc; border: 1px solid #008888; }
QSlider#ChromaSlider::handle:horizontal:hover { background-color: #00ffff; border-color: #ffffff; }
QSlider#ChromaSlider::groove:horizontal { background: #004444; }
QSlider#ChromaSlider::sub-page:horizontal { background: #008888; border-radius: 3px; }

/* Deep Space Slider (Violet) */
QSlider#DeepSlider::handle:horizontal { background-color: #9b59b6; border: 1px solid #8e44ad; }
QSlider#DeepSlider::handle:horizontal:hover { background-color: #af7ac5; border-color: #ffffff; }
QSlider#DeepSlider::groove:horizontal { background: #4a235a; }
QSlider#DeepSlider::sub-page:horizontal { background: #8e44ad; border-radius: 3px; }

/* Buttons */
QPushButton { background-color: #444444; color: #dddddd; border: 1px solid #666666; border-radius: 4px; padding: 6px; font-weight: bold;}
QPushButton:hover { background-color: #555555; border-color: #777777; }
QPushButton:disabled { background-color: #333333; color: #666666; border-color: #444444; }
QPushButton#ProcessButton { background-color: #285299; border: 1px solid #1e3f7a; }
QPushButton#ProcessButton:hover { background-color: #355ea1; }
QPushButton#CloseButton { background-color: #5a2a2a; border: 1px solid #804040; }
QPushButton#CloseButton:hover { background-color: #7a3a3a; }

/* TOGGLE BUTTON (1:1) */
QPushButton#Toggle11 {
    background-color: #3c3c3c;
    border: 1px solid #555555;
    color: #aaaaaa;
    min-width: 60px;
}
QPushButton#Toggle11:checked {
    background-color: #285299;
    border: 1px solid #88aaff;
    color: #ffffff;
}
QPushButton#Toggle11:hover {
    border-color: #999999;
}

/* GHOST HELP BUTTON */
QPushButton#HelpButton {
    background-color: transparent;
    color: #555555;
    border: none;
    font-weight: bold;
    min-width: 20px;
}
QPushButton#HelpButton:hover { color: #aaaaaa; }

QPushButton#CoffeeButton {
    background-color: transparent;
    border: none;
    font-size: 14pt;
    padding: 2px;
}
QPushButton#CoffeeButton:hover {
    background-color: rgba(255, 255, 255, 20);
    border-radius: 4px;
}

/* Zoom buttons */
QPushButton#ZoomBtn { min-width: 30px; font-weight: bold; background-color: #3c3c3c; }

/* Progress bar */
QProgressBar {
    border: 1px solid #555555;
    border-radius: 4px;
    background: #333333;
    text-align: center;
    color: #dddddd;
    font-size: 8pt;
}
QProgressBar::chunk {
    background-color: #285299;
    width: 10px;
}
"""

##############################################
# UTILITY WIDGETS
##############################################

class ResetSlider(QSlider):
    def __init__(self, orientation, default_val=0, parent=None):
        super().__init__(orientation, parent)
        self.default_val = default_val

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setValue(self.default_val)
            # Emit signal to trigger update logic without resetting view
            self.sliderReleased.emit()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

##############################################
# COLOR SPACE CONVERSIONS
##############################################

class ColorSpace:
    @staticmethod
    def rgb_to_xyz(rgb):
        M = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ], dtype=np.float32)
        return np.einsum('ij,jhw->ihw', M, rgb)

    @staticmethod
    def xyz_to_rgb(xyz):
        M_inv = np.array([
            [ 3.2404542, -1.5371385, -0.4985314],
            [-0.9692660,  1.8760108,  0.0415560],
            [ 0.0556434, -0.2040259,  1.0572252]
        ], dtype=np.float32)
        return np.einsum('ij,jhw->ihw', M_inv, xyz)

    @staticmethod
    def xyz_to_lab(xyz):
        Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
        X = xyz[0] / Xn
        Y = xyz[1] / Yn
        Z = xyz[2] / Zn

        epsilon = 0.008856
        kappa = 903.3

        def f(t):
            mask = t > epsilon
            result = np.zeros_like(t)
            result[mask] = np.cbrt(t[mask])
            result[~mask] = (kappa * t[~mask] + 16) / 116
            return result

        fX = f(X)
        fY = f(Y)
        fZ = f(Z)

        L = 116 * fY - 16
        a = 500 * (fX - fY)
        b = 200 * (fY - fZ)

        lab = np.stack([L, a, b], axis=0).astype(np.float32)
        return lab

    @staticmethod
    def lab_to_xyz(lab):
        L, a, b = lab[0], lab[1], lab[2]

        Xn, Yn, Zn = 0.95047, 1.00000, 1.08883

        fy = (L + 16) / 116
        fx = a / 500 + fy
        fz = fy - b / 200

        epsilon = 0.008856
        kappa = 903.3

        def f_inv(t):
            t3 = t ** 3
            mask = t3 > epsilon
            result = np.zeros_like(t)
            result[mask] = t3[mask]
            result[~mask] = (116 * t[~mask] - 16) / kappa
            return result

        X = Xn * f_inv(fx)
        Y = Yn * f_inv(fy)
        Z = Zn * f_inv(fz)

        xyz = np.stack([X, Y, Z], axis=0).astype(np.float32)
        return xyz

    @staticmethod
    def rgb_to_lab(rgb):
        xyz = ColorSpace.rgb_to_xyz(rgb)
        lab = ColorSpace.xyz_to_lab(xyz)
        return lab

    @staticmethod
    def lab_to_rgb(lab):
        xyz = ColorSpace.lab_to_xyz(lab)
        rgb = ColorSpace.xyz_to_rgb(xyz)
        return rgb

##############################################
# CORE MATH – Silentium Engine
##############################################

class SilentiumCore:
    @staticmethod
    def normalize_input(img):
        input_dtype = img.dtype
        img_float = img.astype(np.float32)

        if np.issubdtype(input_dtype, np.integer):
            if input_dtype == np.uint8:
                return img_float / 255.0
            elif input_dtype == np.uint16:
                return img_float / 65535.0
            else:
                return img_float / float(np.iinfo(input_dtype).max)
        elif np.issubdtype(input_dtype, np.floating):
            current_max = np.max(img_float)
            if current_max <= 1.0 + 1e-5:
                return img_float
            if current_max <= 65535.0:
                return img_float / 65535.0
            return img_float / max(current_max, 1e-6)

        return img_float

    @staticmethod
    def compute_luminance(img):
        if img.ndim == 2:
            return img.astype(np.float32), np.stack([img, img, img])

        L = (img[0] + img[1] + img[2]) / 3.0
        return L.astype(np.float32), img.astype(np.float32)

    @staticmethod
    def estimate_noise_map(channel, block_size=64):
        h, w = channel.shape
        sigma_map = np.zeros_like(channel, dtype=np.float32)

        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                y2 = min(y+block_size, h)
                x2 = min(x+block_size, w)
                patch = channel[y:y2, x:x2]
                q = np.quantile(patch, 0.5)
                bg = patch[patch <= q]
                if bg.size < 16:
                    bg = patch
                med = np.median(bg)
                mad = np.median(np.abs(bg - med))
                sigma = 1.4826 * mad if mad > 0 else np.std(bg)
                sigma_map[y:y2, x:x2] = sigma

        mask_zero = sigma_map <= 0
        if np.any(mask_zero):
            median_sigma = np.median(sigma_map[~mask_zero]) if np.any(~mask_zero) else 1e-6
            sigma_map[mask_zero] = median_sigma

        return sigma_map

    @staticmethod
    def compute_signal_probability(channel):
        """
        Statistical Signal Probability Map.
        Determines where real signal exists vs background noise based on Linear statistics.
        Returns: 0.0 (Background) -> 1.0 (Safe Signal)
        """
        # Robust Statistics
        med = float(np.median(channel))
        mad = float(np.median(np.abs(channel - med)))
        sigma = 1.4826 * mad if mad > 0 else 1e-6

        # Thresholds
        # Low: Start of signal transition (Median + 1.0 Sigma)
        # High: Safe signal threshold (Median + 3.5 Sigma)
        low_thr = med + (1.0 * sigma)
        high_thr = med + (3.5 * sigma)
        diff = high_thr - low_thr
        if diff < 1e-9:
            diff = 1e-9

        # Linear Ramp
        signal_map = (channel - low_thr) / diff
        signal_map = np.clip(signal_map, 0.0, 1.0)
        return signal_map.astype(np.float32)

    @staticmethod
    def _auto_stretch_proxy(img):
        """
        Internal MTF stretch calculation for edge detection.
        Replicates Siril's auto-stretch logic on a single channel.
        """
        # Constants from Siril
        MAD_NORM = 1.4826
        SHADOWS_CLIPPING = -2.8
        TARGET_BG = 0.25

        # 1. Calc Stats
        median = float(np.median(img))
        diff = np.abs(img - median)
        mad = float(np.median(diff)) * MAD_NORM
        if mad == 0.0:
            mad = 0.001

        shadows = max(0.0, median + SHADOWS_CLIPPING * mad)
        highlights = 1.0

        # 2. Find Midtones
        x = (median - shadows) / (highlights - shadows + 1e-9)
        x = np.clip(x, 0.0, 1.0)
        y = TARGET_BG

        if x == 0.5:
            midtones = 0.5
        elif x == y:
            midtones = 0.5
        else:
            num = x * (y - 1.0)
            den = 2.0 * x * y - x - y
            if den == 0.0:
                midtones = 0.5
            else:
                midtones = num / den
        midtones = np.clip(midtones, 0.01, 0.99)

        # 3. Apply MTF
        img_norm = (img - shadows) / (highlights - shadows + 1e-9)
        img_norm = np.clip(img_norm, 0.0, 1.0)
        m = midtones
        num_m = (m - 1.0) * img_norm
        den_m = (2.0 * m - 1.0) * img_norm - m
        with np.errstate(divide='ignore', invalid='ignore'):
            out = num_m / den_m
            out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)

        return out.astype(np.float32)

    @staticmethod
    def compute_edge_map(L):
        # Calculate Edge Map on STRETCHED luminance to see faint details
        L_str = SilentiumCore._auto_stretch_proxy(L)

        kx = np.array([[ -1,  0,  1],
                       [ -2,  0,  2],
                       [ -1,  0,  1]], dtype=np.float32)
        ky = np.array([[ -1, -2, -1],
                       [  0,  0,  0],
                       [  1,  2,  1]], dtype=np.float32)

        # Convolve on Stretched
        gx = convolve2d(L_str, kx, mode="same", boundary="symm")
        gy = convolve2d(L_str, ky, mode="same", boundary="symm")
        mag = np.sqrt(gx*gx + gy*gy)

        # Robust Normalization
        # Instead of max (stars), use 98th percentile to boost nebula edges
        mx = np.percentile(mag, 98)
        if mx <= 1e-9:
            mx = 1e-9
        mag /= mx
        mag = np.clip(mag, 0.0, 1.0)

        # Morphological Thickening (Safe due to Gating)
        mag = maximum_filter(mag, size=2)
        return mag.astype(np.float32)

    @staticmethod
    def _pad_for_swt(img, level):
        """Pads image to multiple of 2^level for SWT"""
        h, w = img.shape
        factor = 2**level
        pad_h = (factor - (h % factor)) % factor
        pad_w = (factor - (w % factor)) % factor

        if pad_h == 0 and pad_w == 0:
            return img, (0, 0, 0, 0)

        # Reflect padding to minimize edge artifacts
        padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')
        return padded, (0, pad_h, 0, pad_w)

    @staticmethod
    def compute_fwhm_modulation_map(lst_path, img_shape, star_mask=None):
        """Local seeing map from PSF stars (starless-safe fallback)"""
        H, W = img_shape
        fwhm_map = np.ones((H, W), dtype=np.float32) * 4.0  # Default conservativo

        if star_mask is None or np.max(star_mask) < 0.1:
            return fwhm_map  # Starless: seeing uniforme

        stars = parse_lst(lst_path)
        if len(stars) == 0:
            return fwhm_map

        for star in stars:
            cx, cy = star['X'], star['Y']
            fwhm = (star['FWHMx'] + star['FWHMy']) / 2.0
            r_inf = 1.5 * fwhm  # PSF influence radius

            yy, xx = np.ogrid[:H, :W]
            dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
            weight = np.exp(-0.5 * (dist / r_inf)**2)

            fwhm_map = np.minimum(fwhm_map, fwhm * weight + 0.1 * (1-weight))

        return fwhm_map

    @staticmethod
    def multiscale_denoise(channel, sigma_map, edge_map, intensity, detail_guard,
                           signal_map=None, deep_smooth=0.0, is_chroma=False,
                           progress_callback=None, progress_offset=0, progress_range=50):
        """
        SWT implementation (Stationary Wavelet Transform).
        Shadow Exclusion Gate (Hard Cutoff at 25% Signal)
        """
        max_levels = 4
        wavelet = "db2"

        # Pad image for SWT requirements
        padded_channel, (ph, _, pw, _) = SilentiumCore._pad_for_swt(channel, max_levels)

        # Decompose using Undecimated Wavelet Transform (SWT)
        coeffs = pywt.swt2(padded_channel, wavelet=wavelet, level=max_levels)
        base_sigma_mult = 4.5 * intensity

        # Frequency Domain Damping (Layer Weights)
        layer_weights = [0.60, 0.80, 1.0, 1.0]
        new_coeffs = []
        total_steps = len(coeffs)

        # Prepare maps (pad them to match padded_channel size)
        pad_sigma, _ = SilentiumCore._pad_for_swt(sigma_map, max_levels)
        pad_edge, _ = SilentiumCore._pad_for_swt(edge_map, max_levels)

        pad_signal = None
        if signal_map is not None:
            pad_signal, _ = SilentiumCore._pad_for_swt(signal_map, max_levels)
        else:
            # Fallback if no signal map (should not happen in main flow)
            pad_signal = np.ones_like(pad_sigma)

        # PSF-ADAPTIVE THRESHOLD: Seeing locale da findstar LST
        lst_path = os.path.join(os.getcwd(), "list.lst")
        star_mask = None
        if os.path.exists(lst_path):
            star_mask = build_star_mask_from_lst(lst_path, channel.shape)
        fwhm_map = SilentiumCore.compute_fwhm_modulation_map(
            lst_path, channel.shape, star_mask  # True PSF star mask
        )

        pad_fwhm, _ = SilentiumCore._pad_for_swt(fwhm_map, max_levels)
        pad_sigma *= pad_fwhm  # PSF-adaptive wavelet threshold

        for i, (cA, (cH, cV, cD)) in enumerate(coeffs):
            if progress_callback:
                progress_val = progress_offset + int(progress_range * (i+1) / total_steps)
                progress_callback(progress_val, silent=True)

            # Use fixed weights based on iteration index
            w_layer = layer_weights[i] if i < len(layer_weights) else 1.0

            # For scale calculation (inverse sqrt relation)
            level_idx = max_levels - i
            scale = 2**level_idx

            if is_chroma:
                lvl_degrade = 1.0 / (2**(4 - level_idx))
            else:
                lvl_degrade = 1.0

            sigma_lvl = pad_sigma

            # PHOTOMETRIC GATING (Domain Separation)
            # 1. Detail Guard Domain: Gated by Signal & PSF-Compensated
            # Multiply raw edge (geometric) by signal probability (photometric).
            # Added PSF Compensation. Poor seeing blurs structural edges, lowering their detection score.
            # We compensate by boosting protection where FWHM is high (Physics-Aware Normalization).
            robust_edge = pad_edge * pad_signal * (pad_fwhm ** 1.0)

            # High Gain (K=40) for structure
            K = 40.0
            guard_map = 1.0 + (K * detail_guard * robust_edge)

            # 2. Shadow Smoothness Domain: Shadow Exclusion Gate
            # Logic Change: Winner Take All.
            # Clip signal * 4.0. If signal is > 0.25 (25%), the shadow gate closes completely (0.0).
            # This ensures Shadow Smoothness CANNOT touch structures.
            shadow_gate_active = np.clip(pad_signal * 4.0, 0.0, 1.0)
            inv_signal = 1.0 - shadow_gate_active

            boost_map = 1.0
            if deep_smooth > 0.01:
                boost_map = 1.0 + (3.0 * deep_smooth * inv_signal)

            # Threshold Calculation
            thr = sigma_lvl * base_sigma_mult
            thr /= (scale**0.5)
            thr *= lvl_degrade

            # Apply Frequency Damping
            thr *= w_layer

            # Apply Protection & Boost
            # Guard divides (protects), Boost multiplies (attacks).
            # Now they act on disjoint domains thanks to Shadow Gate.
            thr = thr / guard_map
            thr = thr * boost_map

            cH = SilentiumCore._soft_threshold(cH, thr)
            cV = SilentiumCore._soft_threshold(cV, thr)
            cD = SilentiumCore._soft_threshold(cD, thr)

            new_coeffs.append((cA, (cH, cV, cD)))

        # Reconstruct
        channel_dn = pywt.iswt2(new_coeffs, wavelet=wavelet)

        # Crop back to original size
        orig_h, orig_w = channel.shape
        channel_dn = channel_dn[:orig_h, :orig_w]

        return channel_dn.astype(np.float32)

    @staticmethod
    def _resize_map(src, shape):
        h, w = shape
        H, W = src.shape
        if (h, w) == (H, W):
            return src
        zoom_y = h / H
        zoom_x = w / W
        return zoom(src, (zoom_y, zoom_x), order=1).astype(np.float32)

    @staticmethod
    def _soft_threshold(c, thr):
        return np.sign(c) * np.maximum(np.abs(c) - thr, 0)

    @staticmethod
    def _process_chunk(img_rgb, intensity, detail_guard, use_adaptive_noise, star_mask,
                       enable_chroma, chroma_strength,
                       deep_smooth,
                       progress_callback=None, base_progress=0,
                       job_id=None, gui=None, should_check_cancel=True):

        def check_abort():
            if should_check_cancel and gui and job_id is not None:
                if gui.current_job_id != job_id:
                    return True
            return False

        if check_abort():
            return None

        if img_rgb.ndim == 2:
            img_rgb = np.array([img_rgb, img_rgb, img_rgb])

        chroma_factor = np.clip(chroma_strength / 100.0, 0.0, 1.0)
        deep_factor = np.clip(deep_smooth / 100.0, 0.0, 1.0)

        # Compute Photometric Signal Map (Linear)
        L_full, _ = SilentiumCore.compute_luminance(img_rgb)
        signal_prob_map = SilentiumCore.compute_signal_probability(L_full)

        if enable_chroma:
            lab = ColorSpace.rgb_to_lab(img_rgb)
            if check_abort():
                return None

            L_norm = lab[0] / 100.0
            a_star = lab[1]
            b_star = lab[2]

            if use_adaptive_noise:
                sigma_map_L = SilentiumCore.estimate_noise_map(L_norm)
            else:
                med = np.median(L_norm)
                mad = np.median(np.abs(L_norm - med))
                sigma = 1.4826 * mad if mad > 0 else 1e-6
                sigma_map_L = np.full_like(L_norm, sigma, dtype=np.float32)

            edge_map = SilentiumCore.compute_edge_map(L_norm)
            if check_abort():
                return None

            # Denoise L
            L_dn = SilentiumCore.multiscale_denoise(
                L_norm, sigma_map_L, edge_map,
                intensity, detail_guard,
                signal_map=signal_prob_map, deep_smooth=deep_factor,
                is_chroma=False,
                progress_callback=progress_callback,
                progress_offset=base_progress + 5,
                progress_range=20
            )

            L_dn = np.clip(L_dn, 0.0, 1.0) * 100.0

            if chroma_factor > 0.01:
                if use_adaptive_noise:
                    sigma_map_a = SilentiumCore.estimate_noise_map(a_star)
                    sigma_map_b = SilentiumCore.estimate_noise_map(b_star)
                else:
                    sigma_map_a = np.full_like(a_star, np.std(a_star), dtype=np.float32)
                    sigma_map_b = np.full_like(b_star, np.std(b_star), dtype=np.float32)

                if check_abort():
                    return None

                soft_guard = detail_guard * 0.5

                a_dn = SilentiumCore.multiscale_denoise(
                    a_star, sigma_map_a, edge_map,
                    intensity=chroma_factor,
                    detail_guard=soft_guard,
                    signal_map=signal_prob_map, deep_smooth=deep_factor * 0.5,
                    is_chroma=True,
                    progress_callback=None
                )

                b_dn = SilentiumCore.multiscale_denoise(
                    b_star, sigma_map_b, edge_map,
                    intensity=chroma_factor,
                    detail_guard=soft_guard,
                    signal_map=signal_prob_map, deep_smooth=deep_factor * 0.5,
                    is_chroma=True,
                    progress_callback=None
                )
            else:
                a_dn, b_dn = a_star, b_star

            if check_abort():
                return None

            lab_dn = np.stack([L_dn, a_dn, b_dn], axis=0)
            out_rgb = ColorSpace.lab_to_rgb(lab_dn)

        else:
            # Luma Only Path
            L, _ = SilentiumCore.compute_luminance(img_rgb)

            if use_adaptive_noise:
                sigma_map = SilentiumCore.estimate_noise_map(L)
            else:
                sigma = np.std(L)
                sigma_map = np.full_like(L, sigma, dtype=np.float32)

            edge_map = SilentiumCore.compute_edge_map(L)
            if check_abort():
                return None

            L_dn = SilentiumCore.multiscale_denoise(
                L, sigma_map, edge_map,
                intensity, detail_guard,
                signal_map=signal_prob_map, deep_smooth=deep_factor,
                is_chroma=False,
                progress_callback=progress_callback,
                progress_offset=base_progress + 10,
                progress_range=40
            )

            L_dn = np.clip(L_dn, 0.0, 1.0)
            eps = 1e-8
            L_safe = np.maximum(L, eps)
            ratio = img_rgb / L_safe
            out_rgb = L_dn * ratio

        if star_mask is not None:
            # Maintain hard star protection if smoothing is aggressive
            if deep_factor > 0.1:
                k = 1.5
            else:
                k = 0.7

            alpha = np.clip(star_mask * k, 0.0, 1.0).astype(np.float32)
            out_rgb = out_rgb * (1.0 - alpha) + img_rgb * alpha

        return np.clip(out_rgb, 0.0, 1.0)

    @staticmethod
    def apply_noise_reduction(img, intensity_slider, detail_slider,
                              use_adaptive_noise=True, star_mask=None,
                              enable_chroma=False, chroma_strength=50,
                              deep_smooth=0,
                              progress_callback=None, job_id=None, gui=None,
                              tile_processing=False):

        intensity = np.clip(intensity_slider / 100.0, 0.0, 1.0)
        detail_guard = np.clip(detail_slider / 100.0, 0.0, 1.0)

        c, h, w = (img.shape[0], img.shape[1], img.shape[2]) if img.ndim == 3 else (1, img.shape[0], img.shape[1])
        use_tiles = tile_processing and (max(h, w) > 2500)

        if not use_tiles:
            if progress_callback:
                progress_callback(5, "Processing (Single Pass SWT)...", silent=True)

            return SilentiumCore._process_chunk(
                img, intensity, detail_guard, use_adaptive_noise, star_mask,
                enable_chroma, chroma_strength,
                deep_smooth,
                progress_callback, 0, job_id, gui
            )

        if progress_callback:
            progress_callback(0, "Initializing Tiled SWT Engine...", silent=False)

        TILE_SIZE = 2048
        PADDING = 128

        out_full = np.zeros_like(img)
        y_steps = range(0, h, TILE_SIZE)
        x_steps = range(0, w, TILE_SIZE)
        total_tiles = len(y_steps) * len(x_steps)
        tile_count = 0

        for y in y_steps:
            for x in x_steps:
                tile_count += 1

                y0 = max(0, y - PADDING)
                y1 = min(h, y + TILE_SIZE + PADDING)
                x0 = max(0, x - PADDING)
                x1 = min(w, x + TILE_SIZE + PADDING)

                if img.ndim == 3:
                    src_tile = img[:, y0:y1, x0:x1].copy()
                else:
                    src_tile = img[y0:y1, x0:x1].copy()

                mask_tile = None
                if star_mask is not None:
                    mask_tile = star_mask[y0:y1, x0:x1].copy()

                processed_tile = SilentiumCore._process_chunk(
                    src_tile, intensity, detail_guard, use_adaptive_noise, mask_tile,
                    enable_chroma, chroma_strength,
                    deep_smooth,
                    progress_callback=None,
                    job_id=None, gui=None, should_check_cancel=False
                )

                if processed_tile is None:
                    return None

                valid_y_start = (y - y0)
                valid_y_end = valid_y_start + min(TILE_SIZE, h - y)
                valid_x_start = (x - x0)
                valid_x_end = valid_x_start + min(TILE_SIZE, w - x)

                out_y_start = y
                out_y_end = min(y + TILE_SIZE, h)
                out_x_start = x
                out_x_end = min(x + TILE_SIZE, w)

                if out_full.ndim == 3:
                    out_full[:, out_y_start:out_y_end, out_x_start:out_x_end] = \
                        processed_tile[:, valid_y_start:valid_y_end, valid_x_start:valid_x_end]
                else:
                    out_full[out_y_start:out_y_end, out_x_start:out_x_end] = \
                        processed_tile[valid_y_start:valid_y_end, valid_x_start:valid_x_end]

                if progress_callback:
                    pct = int((tile_count / total_tiles) * 90)
                    msg = f"Processing Tile {tile_count}/{total_tiles}"
                    progress_callback(pct, msg, silent=False)

        return out_full

    @staticmethod
    def calculate_shadow_report(img_orig, img_denoised):
        """
        Computes robust metrics using MAD (Median Absolute Deviation) to ignore stars.
        Replicates Siril's 'bgnoise' logic using NumPy.
        """
        # Ensure we work on a single channel for stats (Green or Luminance)
        if img_orig.ndim == 3:
            # Weighted Luminance
            orig_flat = (img_orig[0] * 0.2126 + img_orig[1] * 0.7152 + img_orig[2] * 0.0722).ravel()
            den_flat = (img_denoised[0] * 0.2126 + img_denoised[1] * 0.7152 + img_denoised[2] * 0.0722).ravel()
        else:
            orig_flat = img_orig.ravel()
            den_flat = img_denoised.ravel()

        # 1. Robust Noise Estimation (MAD)
        # Median is robust against stars/nebulosity
        med_orig = np.median(orig_flat)
        mad_orig = np.median(np.abs(orig_flat - med_orig))
        # 1.4826 is the scaling factor for Normal Distribution
        sigma_orig = 1.4826 * mad_orig if mad_orig > 0 else 1e-6

        med_den = np.median(den_flat)
        mad_den = np.median(np.abs(den_flat - med_den))
        sigma_den = 1.4826 * mad_den if mad_den > 0 else 1e-6

        # 2. Metrics
        noise_reduction_pct = (1.0 - (sigma_den / sigma_orig)) * 100.0
        pedestal_shift = med_den - med_orig

        # SNR Gain (Linear) = Sigma_Old / Sigma_New
        snr_gain = sigma_orig / sigma_den

        # Equivalent Integration Time Factor = (SNR Gain)^2
        eit_factor = snr_gain ** 2

        report = (
            "\n------------------------------------------------------------\n"
            " VERALUX SILENTIUM - SHADOW REPORT\n"
            "------------------------------------------------------------\n"
            f" > Background Noise (Sigma): {sigma_orig:.5f} -> {sigma_den:.5f}\n"
            f" > Noise Reduction: -{noise_reduction_pct:.1f}%\n"
            f" > SNR Improvement: {snr_gain:.2f}x\n"
            f" > Effective Integration: +{eit_factor:.1f}x equivalent time\n"
            f" > Pedestal Shift (Blacks): {pedestal_shift:+.6f} (Flux conservation)\n"
            "------------------------------------------------------------\n"
        )

        return report

##############################################
# MTF PREVIEW (Siril Auto-Stretch Exact)
##############################################

def MTF(x, m, lo, hi):
    m = float(m)
    lo = float(lo)
    hi = float(hi)
    dist = hi - lo
    if dist < 1e-9:
        return np.where(x > lo, 1.0, 0.0).astype(np.float32)

    xp = (x - lo) / dist
    xp = np.clip(xp, 0.0, 1.0)

    num = (m - 1.0) * xp
    den = (2.0 * m - 1.0) * xp - m
    with np.errstate(divide='ignore', invalid='ignore'):
        y = num / den
        y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)

    return y.astype(np.float32)


def find_linked_midtones_balance_siril(img):
    MAD_NORM = 1.4826
    SHADOWS_CLIPPING = -2.8
    TARGET_BG = 0.25

    if img.dtype != np.float32:
        img = img.astype(np.float32)

    if img.ndim == 3:
        channels = [img[0], img[1], img[2]]
    else:
        channels = [img]
    nb_channels = len(channels)

    sum_c0 = 0.0
    sum_m = 0.0
    valid_stats = 0

    for ch in channels:
        median = float(np.median(ch))
        diff = np.abs(ch - median)
        mad = float(np.median(diff)) * MAD_NORM
        if mad == 0.0:
            mad = 0.001
        sum_c0 += median + SHADOWS_CLIPPING * mad
        sum_m += median
        valid_stats += 1

    if valid_stats == 0:
        return 0.0, 0.5, 1.0

    c0 = sum_c0 / valid_stats
    if c0 < 0.0:
        c0 = 0.0

    m_avg = sum_m / valid_stats
    m2 = m_avg - c0

    midtones = MTF(np.array([m2]), TARGET_BG, 0.0, 1.0)[0]
    midtones = float(midtones)
    highlights = 1.0

    return c0, midtones, highlights


def mtf_stretch_rgb_siril_exact(img):
    if img.dtype != np.float32:
        img_f = img.astype(np.float32)
    else:
        img_f = img

    shadows, midtones, highlights = find_linked_midtones_balance_siril(img_f)
    out = MTF(img_f, midtones, shadows, highlights)
    return out

##############################################
# STAR MASK from findstar (.lst) - PSF-AWARE
##############################################

def build_star_mask_from_lst(path, shape):
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r") as f:
            lines = f.readlines()
    except Exception:
        return None

    data_lines = [ln for ln in lines if not ln.startswith("#") and ln.strip()]
    if len(data_lines) < 5:
        return None

    H, W = shape
    mask = np.zeros((H, W), dtype=np.float32)

    for ln in data_lines:
        parts = ln.split()
        if len(parts) < 15:  # Need mag, angle, B, A, FWHM
            continue

        try:
            # Siril .lst format: star# layer B A beta X Y FWHMx FWHMy FWHMx["] FWHMy["] angle RMSE mag Sat Profile RA Dec
            x = float(parts[5])   # X [px]
            y = float(parts[6])   # Y [px]
            fwhmx = float(parts[7])  # FWHM x [px]
            fwhmy = float(parts[8])  # FWHM y [px]
            b = float(parts[2])   # B (minor axis)
            a = float(parts[3])   # A (major axis)
            mag = float(parts[13])  # Magnitude
            angle = float(parts[11])  # Rotation angle [deg]
        except Exception:
            continue

        # 1. PSF-AWARE RADIUS: 1.8x average FWHM (covers 99% stellar flux)
        fwhm_avg = (fwhmx + fwhmy) / 2.0
        r_mask = 1.8 * fwhm_avg

        # 2. ECCENTRICITY BOOST (B/A ratio) for trailing/elongated stars
        ba_ratio = min(b, a) / max(b, a)
        if ba_ratio < 0.3:  # Highly elliptical
            r_mask *= 1.3

        # 3. MAGNITUDE BOOST for bright/saturated stars (bloom/halo protection)
        if mag < -3.0:  # Bright stars with potential bloom
            r_mask *= 1.4

        # Ensure minimum size for faint stars
        r_mask = max(r_mask, 3.0)

        # Gaussian kernel with elliptical PSF (rotated by angle)
        cx, cy = int(round(x)), int(round(y))
        rad_x, rad_y = int(4 * fwhmx/2.355), int(4 * fwhmy/2.355)  # 4-sigma extent

        y0 = max(cy - rad_y, 0); y1 = min(cy + rad_y + 1, H)
        x0 = max(cx - rad_x, 0); x1 = min(cx + rad_x + 1, W)
        if y0 >= y1 or x0 >= x1:
            continue

        yy, xx = np.ogrid[y0:y1, x0:x1]

        # Rotate coordinates by angle for elliptical PSF alignment
        theta = np.radians(angle)
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)

        xx_rot = (xx - cx) * cos_theta - (yy - cy) * sin_theta
        yy_rot = (xx - cx) * sin_theta + (yy - cy) * cos_theta

        # Elliptical Gaussian: sigma_x = fwhmx/2.355, sigma_y = fwhmy/2.355
        sx = fwhmx / 2.355
        sy = fwhmy / 2.355
        g = np.exp(-0.5 * ((xx_rot / sx)**2 + (yy_rot / sy)**2))

        mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], g.astype(np.float32))

    if np.max(mask) > 0:
        mask /= np.max(mask)  # Normalize to [0,1]

    return mask


def parse_lst(lst_path):
    """Parse Siril .lst → list of star dicts (X,Y,FWHMx,FWHMy)"""
    if not os.path.exists(lst_path):
        return []

    try:
        with open(lst_path, "r") as f:
            lines = f.readlines()
    except:
        return []

    stars = []
    for ln in lines:
        if ln.startswith("#") or not ln.strip():
            continue

        parts = ln.split()
        if len(parts) < 9:
            continue

        try:
            star = {
                'X': float(parts[5]),     # X [px]
                'Y': float(parts[6]),     # Y [px]
                'FWHMx': float(parts[7]), # FWHM x [px]
                'FWHMy': float(parts[8])  # FWHM y [px]
            }
            stars.append(star)
        except:
            continue

    return stars

##############################################
# MAIN GUI
##############################################

class SilentiumGUI(QMainWindow):
    def __init__(self, siril, app):
        super().__init__()

        self.siril = siril
        self.app = app

        self.setWindowTitle(f"VeraLux Silentium v{SilentiumDefaults.VERSION}")
        self.setStyleSheet(DARK_STYLESHEET)
        self.resize(1350, 700)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)

        self.settings = QSettings("VeraLux", "Silentium")

        self.linear_full = None
        self.lst_file_abs = None

        # Proxies & View State
        self.linear_global_proxy = None
        self.star_mask_full = None
        self.star_mask_global_proxy = None

        # Dynamic ROI State & Drag handling
        self.scale_ratio = 1.0
        self.current_crop_center = (0, 0)
        self.is_crop_mode = False
        self.drag_start_pos = None

        # View Update Logic flags
        self.pending_view_op = "FIT"  # "FIT" or "1:1" or None
        self.denoised_proxy = None
        self.original_stretched = None
        self.show_original = False

        self.working_dir = os.getcwd()
        self.current_job_id = 0

        self.debounce = QTimer()
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(400)
        self.debounce.timeout.connect(self.run_preview_logic)

        header_msg = (
            "\n##############################################\n"
            "# VeraLux — Silentium\n"
            "# Linear-Phase Noise Suppression Engine\n"
            "# Author: Riccardo Paterniti (2025)\n"
            "# Contact: info@veralux.space\n"
            "##############################################"
        )

        try:
            self.siril.log(header_msg, color=LogColor.DEFAULT)
        except Exception:
            print(header_msg)

        self.init_ui()
        self.cache_input_and_starlist()

    def siril_log(self, text, color=LogColor.DEFAULT):
        try:
            self.siril.log(text, color=color)
        except Exception:
            print(text)

    def status_update(self, text):
        self.lbl_status.setText(text)

    def init_ui(self):
        main = QWidget()
        self.setCentralWidget(main)
        layout = QHBoxLayout(main)

        # LEFT PANEL
        left_container = QWidget()
        left_container.setFixedWidth(380)
        left = QVBoxLayout(left_container)
        left.setContentsMargins(0,0,0,0)

        # 1. Input
        g1 = QGroupBox("1. Input (Linear)")
        l1 = QVBoxLayout(g1)
        self.lbl_src = QLabel("Source: [Active Linear Image]")
        l1.addWidget(self.lbl_src)
        left.addWidget(g1)

        # 2. Silentium Core
        g2 = QGroupBox("2. Silentium Core")
        l2 = QVBoxLayout(g2)

        self.lbl_int = QLabel("Noise Intensity (Log S): 0.80")
        l2.addWidget(self.lbl_int)

        self.s_int = ResetSlider(Qt.Orientation.Horizontal, SilentiumDefaults.INTENSITY)
        self.s_int.setObjectName("MainSlider")
        self.s_int.setRange(0, 100)
        self.s_int.setValue(SilentiumDefaults.INTENSITY)
        self.s_int.setToolTip("Global noise reduction strength.\nHigher = stronger denoise.\nDouble-click to reset.")
        self.s_int.valueChanged.connect(self.update_labels)
        self.s_int.sliderReleased.connect(self.trigger_update_immediate)
        self.s_int.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        l2.addWidget(self.s_int)

        self.lbl_det = QLabel("Detail Guard: 0%")
        l2.addWidget(self.lbl_det)

        self.s_det = ResetSlider(Qt.Orientation.Horizontal, SilentiumDefaults.DETAIL_GUARD)
        self.s_det.setRange(0, 100)
        self.s_det.setValue(SilentiumDefaults.DETAIL_GUARD)
        self.s_det.setToolTip("Morphological Structure Protection with Photometric Gating.\n"
                              "Increase to protect faint filaments.\n"
                              "Unlike standard masks, it forces protection to ZERO on the background,\n"
                              "allowing maximum denoising of the sky floor.")
        self.s_det.valueChanged.connect(self.update_labels)
        self.s_det.sliderReleased.connect(self.trigger_update_immediate)
        self.s_det.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        l2.addWidget(self.s_det)

        self.chk_adapt = QCheckBox("Adaptive Noise Model")
        self.chk_adapt.setChecked(SilentiumDefaults.ADAPTIVE_NOISE)
        self.chk_adapt.setToolTip("Calculates local noise statistics (MAD).\n"
                                  "Required for accurate Signal Probability Mapping and Gating logic.\n"
                                  "Keep ON for best results.")
        self.chk_adapt.toggled.connect(self.trigger_update_immediate)
        self.chk_adapt.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        l2.addWidget(self.chk_adapt)

        left.addWidget(g2)

        # 2B. Chrominance
        g_chroma = QGroupBox("2B. Chrominance (Color Noise)")
        l_chroma = QVBoxLayout(g_chroma)

        self.chk_chroma = QCheckBox("Enable Chroma Denoise (LAB)")
        self.chk_chroma.setChecked(SilentiumDefaults.ENABLE_CHROMA)
        self.chk_chroma.setToolTip("Process color noise in LAB space (a*, b* channels).\n"
                                   "Disable for pure luminance-only denoising (faster).")
        self.chk_chroma.toggled.connect(self.on_chroma_toggle)
        self.chk_chroma.toggled.connect(self.trigger_update_immediate)
        self.chk_chroma.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        l_chroma.addWidget(self.chk_chroma)

        self.lbl_chroma = QLabel("Chroma Strength: 50%")
        l_chroma.addWidget(self.lbl_chroma)

        self.s_chroma = ResetSlider(Qt.Orientation.Horizontal, SilentiumDefaults.CHROMA)
        self.s_chroma.setObjectName("ChromaSlider")
        self.s_chroma.setRange(0, 100)
        self.s_chroma.setValue(SilentiumDefaults.CHROMA)
        self.s_chroma.setToolTip("Controls chroma denoise intensity relative to luminance.\n"
                                 "Lower = gentler (preserves color nuance).\n"
                                 "Higher = stronger (removes more color noise).")
        self.s_chroma.valueChanged.connect(self.update_labels)
        self.s_chroma.sliderReleased.connect(self.trigger_update_immediate)
        self.s_chroma.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        l_chroma.addWidget(self.s_chroma)

        left.addWidget(g_chroma)

        # 3. Deep Space
        g_deep = QGroupBox("3. Deep Space Smoothness (Shadows)")
        l_deep = QVBoxLayout(g_deep)

        self.lbl_deep = QLabel("Shadow Smoothness: 0%")
        l_deep.addWidget(self.lbl_deep)

        self.s_deep = ResetSlider(Qt.Orientation.Horizontal, SilentiumDefaults.SHADOW_SMOOTH)
        self.s_deep.setObjectName("DeepSlider")
        self.s_deep.setRange(0, 100)
        self.s_deep.setValue(SilentiumDefaults.SHADOW_SMOOTH)
        self.s_deep.setToolTip("Aggressive 'Shadow Authority' for the deep background.\n"
                               "Features a Hard Cutoff: it automatically shuts down on signal (>25%),\n"
                               "preventing blur on structures regardless of intensity.")
        self.s_deep.valueChanged.connect(self.update_labels)
        self.s_deep.sliderReleased.connect(self.trigger_update_immediate)
        self.s_deep.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        l_deep.addWidget(self.s_deep)

        left.addWidget(g_deep)

        # 4. Star Field
        g4 = QGroupBox("4. Star Field Handling")
        l4 = QVBoxLayout(g4)

        self.chk_use_stars = QCheckBox("Use findstar (.lst) for Star Protection")
        self.chk_use_stars.setChecked(SilentiumDefaults.USE_STARS)
        self.chk_use_stars.setToolTip("If enabled, stars are protected from both Wavelet Denoising\n"
                                      "and Shadow Smoothing.\n"
                                      "Ensures stars remain hard while background becomes creamy.")
        self.chk_use_stars.toggled.connect(self.trigger_update_immediate)
        self.chk_use_stars.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        l4.addWidget(self.chk_use_stars)

        self.chk_auto_starless = QCheckBox("Auto Starless Detection")
        self.chk_auto_starless.setChecked(SilentiumDefaults.AUTO_STARLESS)
        self.chk_auto_starless.setToolTip("If very few stars are detected, Silentium switches to Starless mode automatically.")
        self.chk_auto_starless.toggled.connect(self.trigger_update_immediate)
        self.chk_auto_starless.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        l4.addWidget(self.chk_auto_starless)

        left.addWidget(g4)

        # Status + progress
        status_box = QGroupBox("Status")
        ls = QVBoxLayout(status_box)

        self.lbl_status = QLabel("Silentium ready.")
        self.lbl_status.setStyleSheet("color:#bbbbbb; font-size:8pt;")
        ls.addWidget(self.lbl_status)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("%p%")
        ls.addWidget(self.progress)

        left.addWidget(status_box)

        # Footer buttons
        footer = QHBoxLayout()

        self.btn_help = QPushButton("?")
        self.btn_help.setObjectName("HelpButton")
        self.btn_help.setFixedWidth(20)
        self.btn_help.setToolTip("Print Operational Guide to Siril Console")
        self.btn_help.clicked.connect(self.print_help_to_console)
        self.btn_help.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        footer.addWidget(self.btn_help)

        b_res = QPushButton("Defaults")
        b_res.setToolTip("Reset all parameters to default values.")
        b_res.clicked.connect(self.set_defaults)
        b_res.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        footer.addWidget(b_res)

        b_cls = QPushButton("Close")
        b_cls.setObjectName("CloseButton")
        b_cls.setToolTip("Close the application.")
        b_cls.clicked.connect(self.close)
        b_cls.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        footer.addWidget(b_cls)

        b_proc = QPushButton("PROCESS")
        b_proc.setObjectName("ProcessButton")
        b_proc.setToolTip("Apply Silentium to the full-resolution linear image in Siril.")
        b_proc.clicked.connect(self.process_full_resolution)
        b_proc.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        footer.addWidget(b_proc)

        left.addLayout(footer)

        left.addStretch()
        layout.addWidget(left_container)

        # RIGHT PANEL (Preview)
        right = QVBoxLayout()

        tb = QHBoxLayout()

        # Toggle Button 1:1
        self.btn_preview_mode = QPushButton("🔍 1:1")
        self.btn_preview_mode.setObjectName("Toggle11")
        self.btn_preview_mode.setCheckable(True)
        self.btn_preview_mode.setToolTip("OFF: Fit Global View (Locked).\nON: 1:1 Loupe Mode (Interactive).")
        self.btn_preview_mode.toggled.connect(self.on_preview_mode_toggle)
        self.btn_preview_mode.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        tb.addWidget(self.btn_preview_mode)

        tb.addSpacing(10)

        # Zoom Buttons with NoFocus policy
        self.b_out = QPushButton("-"); self.b_out.setObjectName("ZoomBtn"); self.b_out.clicked.connect(self.zoom_out); self.b_out.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.b_fit = QPushButton("Fit"); self.b_fit.setObjectName("ZoomBtn"); self.b_fit.clicked.connect(self.fit_view); self.b_fit.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.b_11 = QPushButton("1:1"); self.b_11.setObjectName("ZoomBtn"); self.b_11.clicked.connect(self.zoom_1to1); self.b_11.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.b_in = QPushButton("+"); self.b_in.setObjectName("ZoomBtn"); self.b_in.clicked.connect(self.zoom_in); self.b_in.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        # Disable zoom controls initially
        self.toggle_zoom_controls(False)

        lbl_hint = QLabel("Click to Loupe | Hold SPACE to Compare")
        lbl_hint.setStyleSheet("color: #ffb000; font-size: 8pt; font-style: italic; margin-left: 10px; font-weight: bold;")

        self.btn_coffee = QPushButton("☕")
        self.btn_coffee.setObjectName("CoffeeButton")
        self.btn_coffee.setToolTip("Support the developer - Buy me a coffee!")
        self.btn_coffee.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_coffee.clicked.connect(lambda: webbrowser.open("https://buymeacoffee.com/riccardo.paterniti"))
        self.btn_coffee.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.chk_ontop = QCheckBox("On Top")
        self.chk_ontop.setToolTip("Keep window above Siril.")
        self.chk_ontop.setChecked(True)
        self.chk_ontop.toggled.connect(self.toggle_ontop)
        self.chk_ontop.setStyleSheet("color: #cccccc; font-weight: bold; margin-left: 10px;")
        self.chk_ontop.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        tb.addWidget(self.b_out); tb.addWidget(self.b_fit); tb.addWidget(self.b_11); tb.addWidget(self.b_in)
        tb.addWidget(lbl_hint); tb.addStretch(); tb.addWidget(self.chk_ontop); tb.addWidget(self.btn_coffee)

        right.addLayout(tb)

        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setStyleSheet("background-color: #151515; border: none;")
        self.view.setDragMode(QGraphicsView.DragMode.NoDrag)  # Default NoDrag
        self.view.viewport().setMouseTracking(True)
        self.view.setMouseTracking(True)
        self.view.viewport().installEventFilter(self)
        self.view.installEventFilter(self)
        right.addWidget(self.view)

        self.pix_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pix_item)

        # ROI Cursor (Red Square)
        self.roi_cursor = QGraphicsRectItem(0, 0, 100, 100)
        self.roi_cursor.setPen(QPen(QColor(255, 0, 0), 2))
        self.roi_cursor.setZValue(10)
        self.roi_cursor.setVisible(False)
        self.scene.addItem(self.roi_cursor)

        self.lbl_blink = QLabel("ORIGINAL", self.view)
        self.lbl_blink.setStyleSheet(
            "background-color: rgba(255, 160, 0, 200); "
            "color: #ffffff; "
            "font-size: 14pt; "
            "font-weight: bold; "
            "padding: 8px 16px; "
            "border-radius: 6px;"
        )
        self.lbl_blink.hide()

        layout.addLayout(right)

        self.update_overlays()
        self.on_chroma_toggle()

    ##############################################
    # SETTINGS & UI
    ##############################################

    def update_labels(self):
        val_int = self.s_int.value() / 50.0
        val_det = int(self.s_det.value())
        val_chroma = int(self.s_chroma.value())
        val_deep = int(self.s_deep.value())

        self.lbl_int.setText(f"Noise Intensity (Log S): {val_int:.2f}")
        self.lbl_det.setText(f"Detail Guard: {val_det}%")
        self.lbl_chroma.setText(f"Chroma Strength: {val_chroma}%")
        self.lbl_deep.setText(f"Shadow Smoothness: {val_deep}%")

    def on_chroma_toggle(self):
        enabled = self.chk_chroma.isChecked()
        self.s_chroma.setEnabled(enabled)
        self.lbl_chroma.setEnabled(enabled)

    def set_defaults(self):
        self.siril_log("Silentium: Resetting defaults", LogColor.BLUE)

        # Pull from Class Defaults exclusively
        self.s_int.setValue(SilentiumDefaults.INTENSITY)
        self.s_det.setValue(SilentiumDefaults.DETAIL_GUARD)
        self.s_chroma.setValue(SilentiumDefaults.CHROMA)
        self.s_deep.setValue(SilentiumDefaults.SHADOW_SMOOTH)

        self.chk_adapt.setChecked(SilentiumDefaults.ADAPTIVE_NOISE)
        self.chk_chroma.setChecked(SilentiumDefaults.ENABLE_CHROMA)
        self.chk_use_stars.setChecked(SilentiumDefaults.USE_STARS)
        self.chk_auto_starless.setChecked(SilentiumDefaults.AUTO_STARLESS)

        # Ensure we don't reset the preview mode
        # The update_labels call triggers a UI update but we want to force
        # a recalculation without resetting the view
        self.update_labels()
        self.on_chroma_toggle()
        self.trigger_update_immediate()

    def toggle_zoom_controls(self, enabled):
        self.b_out.setEnabled(enabled)
        self.b_fit.setEnabled(enabled)
        self.b_11.setEnabled(enabled)
        self.b_in.setEnabled(enabled)

    ##############################################
    # INPUT & CACHING
    ##############################################

    def cache_input_and_starlist(self):
        try:
            self.siril_log("Silentium: Caching input image", LogColor.BLUE)

            if not self.siril.connected:
                self.siril.connect()

            with self.siril.image_lock():
                img = self.siril.get_image_pixeldata()
                if img is None:
                    self.siril_log("Silentium: No image open in Siril.", LogColor.RED)
                    return

                self.linear_full = SilentiumCore.normalize_input(img)

            if self.linear_full.ndim == 3:
                _, h, w = self.linear_full.shape
            else:
                h, w = self.linear_full.shape

            # --- Global Proxy (Resized) & Scale Calculation ---
            max_dim = max(h, w)
            target_size = 2048

            if max_dim > target_size:
                step = 1
                while (max_dim / step) > (target_size * 2):
                    step *= 2

                if step > 1:
                    if self.linear_full.ndim == 3:
                        temp_proxy = self.linear_full[:, ::step, ::step]
                    else:
                        temp_proxy = self.linear_full[::step, ::step]
                else:
                    temp_proxy = self.linear_full

                if temp_proxy.ndim == 3:
                    _, h_t, w_t = temp_proxy.shape
                else:
                    h_t, w_t = temp_proxy.shape

                max_dim_t = max(h_t, w_t)
                scale = target_size / max_dim_t

                if self.linear_full.ndim == 3:
                    self.linear_global_proxy = zoom(temp_proxy, (1.0, scale, scale), order=1).astype(np.float32)
                else:
                    self.linear_global_proxy = zoom(temp_proxy, (scale, scale), order=1).astype(np.float32)

                if self.linear_global_proxy.ndim == 3:
                    pg_h, pg_w = self.linear_global_proxy.shape[1], self.linear_global_proxy.shape[2]
                else:
                    pg_h, pg_w = self.linear_global_proxy.shape

                self.scale_ratio = pg_w / w
            else:
                self.linear_global_proxy = self.linear_full.copy()
                self.scale_ratio = 1.0

            # Default Crop Center
            self.current_crop_center = (w // 2, h // 2)
            self.pending_view_op = "FIT"
            self.lbl_src.setText("Source: Active Linear Image (cached)")

            # STARLIST logic
            lst_path = os.path.join(self.working_dir, "list.lst")
            lst_txt = os.path.join(self.working_dir, "list.txt")
            if not os.path.exists(lst_path) and os.path.exists(lst_txt):
                lst_path = lst_txt

            if self.chk_use_stars.isChecked():
                if not os.path.exists(lst_path):
                    try:
                        self.siril_log("Silentium: Running 'findstar -out=list'", LogColor.BLUE)
                        self.siril.cmd("findstar", "-out=list")
                        self.siril.cmd("clearstar")
                    except Exception as e:
                        self.siril_log(f"Silentium: findstar/clearstar failed ({e})", LogColor.RED)

                    lst_path = os.path.join(self.working_dir, "list.lst")
                    if not os.path.exists(lst_path) and os.path.exists(lst_txt):
                        lst_path = lst_txt

                self.lst_file_abs = lst_path
                self.siril_log("Silentium: Building star mask from list", LogColor.BLUE)
                star_mask = build_star_mask_from_lst(lst_path, (h, w))
                self.star_mask_full = star_mask

                if star_mask is not None:
                    # Resize for Global
                    if self.linear_global_proxy is not None:
                        if self.linear_global_proxy.ndim == 3:
                            _, h2, w2 = self.linear_global_proxy.shape
                        else:
                            h2, w2 = self.linear_global_proxy.shape
                        self.star_mask_global_proxy = SilentiumCore._resize_map(star_mask, (h2, w2))
                    else:
                        self.star_mask_global_proxy = None
                else:
                    self.star_mask_full = None
                    self.star_mask_global_proxy = None
            else:
                self.star_mask_full = None
                self.star_mask_global_proxy = None

            self.update_labels()
            self.trigger_update_immediate()

        except Exception as e:
            self.siril_log(f"Silentium: Error caching input - {e}", LogColor.RED)
            traceback.print_exc()

    ##############################################
    # HELPER: Dynamic Crop Extraction
    ##############################################

    def extract_crop(self, center_x, center_y, size=1200):
        if self.linear_full is None:
            return None, None

        if self.linear_full.ndim == 3:
            _, h, w = self.linear_full.shape
        else:
            h, w = self.linear_full.shape

        y0 = max(0, center_y - size // 2)
        y1 = min(h, center_y + size // 2)
        x0 = max(0, center_x - size // 2)
        x1 = min(w, center_x + size // 2)

        if self.linear_full.ndim == 3:
            crop_img = self.linear_full[:, y0:y1, x0:x1].copy()
        else:
            crop_img = self.linear_full[y0:y1, x0:x1].copy()

        crop_mask = None
        if self.star_mask_full is not None:
            crop_mask = self.star_mask_full[y0:y1, x0:x1].copy()

        return crop_img, crop_mask

    ##############################################
    # PREVIEW LOGIC & INTERACTION
    ##############################################

    def on_preview_mode_toggle(self, checked):
        self.is_crop_mode = checked
        if not checked:
            # Switch to Global (Locked Fit)
            self.roi_cursor.setVisible(True)
            self.view.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.toggle_zoom_controls(False)
            self.pending_view_op = "FIT"
        else:
            # Switch to 1:1 Loupe (Navigable)
            self.roi_cursor.setVisible(False)
            self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.toggle_zoom_controls(True)
            self.pending_view_op = "1:1"

        self.trigger_update_immediate()

    def update_roi_cursor(self, view_pos):
        # view_pos is in scene coordinates (Global Proxy pixels)
        if self.is_crop_mode or self.scale_ratio == 0:
            self.roi_cursor.setVisible(False)
            return

        # Visual size of crop box on global view
        CROP_SIZE_FULL = 1200
        vis_size = CROP_SIZE_FULL * self.scale_ratio

        self.roi_cursor.setRect(0, 0, vis_size, vis_size)
        self.roi_cursor.setPos(view_pos.x() - vis_size/2, view_pos.y() - vis_size/2)
        self.roi_cursor.setVisible(True)

    def trigger_update_debounced(self):
        if self.linear_full is None:
            return
        self.debounce.start()
        self.status_update("Preview pending...")

    def trigger_update_immediate(self):
        if self.linear_full is None:
            return
        if self.debounce.isActive():
            self.debounce.stop()
        self.run_preview_logic()

    def run_preview_logic(self):
        try:
            self.current_job_id += 1
            current_job = self.current_job_id

            # Select Source
            if self.is_crop_mode:
                cx, cy = self.current_crop_center
                img, s_mask = self.extract_crop(cx, cy)
            else:
                img = self.linear_global_proxy
                s_mask = self.star_mask_global_proxy if self.chk_use_stars.isChecked() else None
                self.request_fit = True  # Always Fit in Global

            if img is None:
                return

            use_adapt = self.chk_adapt.isChecked()
            int_val = self.s_int.value()
            det_val = self.s_det.value()
            enable_chroma = self.chk_chroma.isChecked()
            chroma_val = self.s_chroma.value()
            deep_val = self.s_deep.value()

            if self.chk_auto_starless.isChecked() and s_mask is not None:
                if np.max(s_mask) < 0.1:
                    s_mask = None

            if img.ndim == 2:
                img = np.array([img, img, img])

            mode_str = "LAB" if enable_chroma else "Luma"
            view_str = "Loupe 1:1" if self.is_crop_mode else "Global"
            self.status_update(f"Computing ({view_str}, {mode_str})...")

            def progress_cb(val, msg="", silent=False):
                if self.current_job_id != current_job:
                    return
                self.progress.setValue(val)
                if not silent:
                    self.status_update(msg)
                self.app.processEvents()

            out = SilentiumCore.apply_noise_reduction(
                img, int_val, det_val,
                use_adaptive_noise=use_adapt,
                star_mask=s_mask,
                enable_chroma=enable_chroma,
                chroma_strength=chroma_val,
                deep_smooth=deep_val,
                progress_callback=progress_cb,
                job_id=current_job,
                gui=self,
                tile_processing=False
            )

            if self.current_job_id != current_job or out is None:
                self.status_update("Preview cancelled (obsolete)")
                return

            self.denoised_proxy = out

            # Stretch for original compare
            img_orig = img
            if img_orig.ndim == 2:
                img_orig = np.array([img_orig, img_orig, img_orig])

            img_orig_clip = np.clip(img_orig, 0.0, 1.0)
            self.original_stretched = mtf_stretch_rgb_siril_exact(img_orig_clip)

            self.update_view()
            self.progress.setValue(100)
            self.status_update(f"Preview ready ({view_str})")

        except Exception as e:
            self.siril_log(f"Silentium: Preview error - {e}", LogColor.RED)
            self.status_update("Preview error")
            traceback.print_exc()

    def update_view(self):
        if self.denoised_proxy is None:
            return

        if self.show_original and self.original_stretched is not None:
            preview = self.original_stretched
            self.lbl_blink.show()
        else:
            img_lin = np.clip(self.denoised_proxy, 0.0, 1.0)
            preview = mtf_stretch_rgb_siril_exact(img_lin)
            self.lbl_blink.hide()

        disp = np.clip(preview * 255, 0, 255).astype(np.uint8)

        if disp.ndim == 3:
            disp = np.ascontiguousarray(np.flipud(disp.transpose(1, 2, 0)))
            h, w, c = disp.shape
            qimg = QImage(disp.data.tobytes(), w, h, c*w, QImage.Format.Format_RGB888)
        else:
            disp = np.ascontiguousarray(np.flipud(disp))
            h, w = disp.shape
            qimg = QImage(disp.data.tobytes(), w, h, w, QImage.Format.Format_Grayscale8)

        self.pix_item.setPixmap(QPixmap.fromImage(qimg))
        self.scene.setSceneRect(0, 0, w, h)

        # Handle Pending View Operations (Fit or 1:1)
        if self.pending_view_op == "FIT":
            QTimer.singleShot(50, self.fit_view)
            self.pending_view_op = None
        elif self.pending_view_op == "1:1":
            self.view.resetTransform()
            self.pending_view_op = None

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            if self.denoised_proxy is not None and self.original_stretched is not None:
                self.show_original = True
                self.update_view()
            event.accept()
            return

        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            if self.denoised_proxy is not None:
                self.show_original = False
                self.update_view()
            event.accept()
            return

        super().keyReleaseEvent(event)

    def eventFilter(self, source, event):
        if source == self.view.viewport():
            if event.type() == QEvent.Type.MouseMove:
                if not self.is_crop_mode:
                    pos = self.view.mapToScene(event.pos())
                    self.update_roi_cursor(pos)
                return False

            elif event.type() == QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.LeftButton:
                    if not self.is_crop_mode:
                        # CLICK on Global View -> Switch to 1:1
                        self.drag_start_pos = None  # Reset release logic
                        pos = self.view.mapToScene(event.pos())
                        if self.scale_ratio > 0:
                            # 1. Flip correction for Y coordinate
                            # The display is flipped vertically (flipud in update_view)
                            if self.linear_global_proxy.ndim == 3:
                                ph = self.linear_global_proxy.shape[1]
                            else:
                                ph = self.linear_global_proxy.shape[0]
                            proxy_y = ph - pos.y()

                            # 2. Precise coordinate mapping with Rounding
                            full_x = int(round(pos.x() / self.scale_ratio))
                            full_y = int(round(proxy_y / self.scale_ratio))

                            self.current_crop_center = (full_x, full_y)
                            self.btn_preview_mode.setChecked(True)
                        return True
                    else:
                        self.drag_start_pos = event.pos()

            elif event.type() == QEvent.Type.MouseButtonRelease:
                if event.button() == Qt.MouseButton.LeftButton and self.is_crop_mode:
                    # Check for Drag vs Click in Loupe Mode
                    if self.drag_start_pos is not None:
                        dist = (event.pos() - self.drag_start_pos).manhattanLength()
                        if dist < 5:  # Threshold for Static Click
                            # Return to Global
                            self.btn_preview_mode.setChecked(False)
                            return True

            elif event.type() == QEvent.Type.Wheel:
                if not self.is_crop_mode:
                    return True  # Disable wheel in global

                if event.angleDelta().y() > 0:
                    self.zoom_in()
                else:
                    self.zoom_out()
                return True

            elif event.type() == QEvent.Type.MouseButtonDblClick:
                # Double click to toggle
                self.btn_preview_mode.toggle()
                return True

        if source == self.view and event.type() == QEvent.Type.Resize:
            if not self.is_crop_mode:
                self.fit_view()  # Enforce fit in global
            self.update_overlays()

        return super().eventFilter(source, event)

    def process_full_resolution(self):
        if self.linear_full is None:
            return

        try:
            self.setEnabled(False)
            self.progress.setValue(0)

            enable_chroma = self.chk_chroma.isChecked()
            mode_str = "LAB" if enable_chroma else "Luma"
            self.siril_log(f"Silentium: Processing full-resolution SWT ({mode_str})", LogColor.GREEN)

            def cb(val, msg="", silent=False):
                self.progress.setValue(val)
                if not silent:
                    self.status_update(msg)
                if val % 10 == 0 or val < 5 or val > 95:
                    self.siril_log(f" {msg} ({val}%)", LogColor.DEFAULT)
                self.app.processEvents()

            use_adapt = self.chk_adapt.isChecked()
            int_val = self.s_int.value()
            det_val = self.s_det.value()
            chroma_val = self.s_chroma.value()
            deep_val = self.s_deep.value()

            star_mask = self.star_mask_full if self.chk_use_stars.isChecked() else None
            if self.chk_auto_starless.isChecked() and star_mask is not None:
                if np.max(star_mask) < 0.1:
                    self.siril_log("Silentium: Starless mode (few stars)", LogColor.YELLOW)
                    star_mask = None

            img = self.linear_full
            if img.ndim == 2:
                img = np.array([img, img, img])

            out = SilentiumCore.apply_noise_reduction(
                img, int_val, det_val,
                use_adaptive_noise=use_adapt,
                star_mask=star_mask,
                enable_chroma=enable_chroma,
                chroma_strength=chroma_val,
                deep_smooth=deep_val,
                progress_callback=cb,
                job_id=None,
                gui=None,
                tile_processing=True
            )

            if out is None:
                raise RuntimeError("Processing returned None (Error or Memory failure)")

            # --- Shadow Report Calculation ---
            cb(95, "Computing Shadow Report...")
            try:
                report_msg = SilentiumCore.calculate_shadow_report(img, out)
                self.siril_log(report_msg, LogColor.BLUE)
            except Exception as e:
                print(f"Stats error: {e}")

            cb(98, "Writing result to Siril")
            
            if self.linear_full.ndim == 2 and out.ndim == 3:
                out = out[0]
                
            with self.siril.image_lock():
                self.siril.undo_save_state("VeraLux Silentium")
                self.siril.set_image_pixeldata(out.astype(np.float32))

            cb(100, "Completed")
            self.siril_log("VeraLux Silentium: Noise reduction applied.", LogColor.GREEN)

            self.settings.setValue("intensity", int_val)
            self.settings.setValue("detail_guard", det_val)
            self.settings.setValue("adaptive_noise", use_adapt)
            self.settings.setValue("enable_chroma", enable_chroma)
            self.settings.setValue("chroma_strength", chroma_val)
            self.settings.setValue("use_stars", self.chk_use_stars.isChecked())
            self.settings.setValue("auto_starless", self.chk_auto_starless.isChecked())
            self._cleanup_temp_files()

            self.close()

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.siril_log(f"Silentium: Processing error - {e}", LogColor.RED)
            traceback.print_exc()

        finally:
            self.setEnabled(True)

    def _cleanup_temp_files(self):
        # 1) clear star overlay (internal Siril state)
        try:
            self.siril.cmd("clearstar")
        except Exception as e:
            try:
                self.siril_log(f"Silentium: clearstar warning: {e}")
            except:
                pass

        # 2) remove temporary star list file from disk
        if not getattr(self, "lst_file_abs", None):
            return

        paths_to_try = set()
        paths_to_try.add(self.lst_file_abs)

        # Try both with/without .lst and case variants
        if self.lst_file_abs.lower().endswith(".lst"):
            paths_to_try.add(self.lst_file_abs[:-4])
        else:
            paths_to_try.add(self.lst_file_abs + ".lst")
            paths_to_try.add(self.lst_file_abs + ".LST")
            # Silentium usa anche .txt a volte come fallback
            paths_to_try.add(self.lst_file_abs + ".txt")

        for p in paths_to_try:
            try:
                if os.path.exists(p):
                    os.remove(p)
                    self.siril_log(f"Silentium: removed temp file: {p}")
            except Exception as e:
                try:
                    self.siril_log(f"Silentium: could not remove {p}: {e}")
                except:
                    pass

    def toggle_ontop(self, checked):
        pos = self.pos()
        if checked:
            self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        else:
            self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowStaysOnTopHint)
        self.show()
        self.move(pos)

    def resizeEvent(self, event):
        self.update_overlays()
        super().resizeEvent(event)

    def closeEvent(self, event):
        self._cleanup_temp_files()
        super().closeEvent(event)

    def update_overlays(self):
        w, h = self.view.width(), self.view.height()
        bw = self.lbl_blink.width()
        self.lbl_blink.move((w - bw) // 2, 10)

    def zoom_in(self): self.view.scale(1.2, 1.2)
    def zoom_out(self): self.view.scale(1/1.2, 1/1.2)
    def zoom_1to1(self): self.view.resetTransform()
    def fit_view(self): self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def print_help_to_console(self):
        guide_lines = [
            "==========================================================================",
            " VERALUX SILENTIUM v1.0 - OPERATIONAL GUIDE",
            " Linear-Phase Noise Suppression & Shadow Authority Engine",
            "==========================================================================",
            "",
            "OVERVIEW",
            "-----------------",
            "Silentium is a specialized linear processing engine designed to decouple",
            "the treatment of low-signal areas (shadows) from high-signal structures.",
            "It utilizes a Stationary Wavelet Transform (SWT) and Photometric Gating",
            "to separate stochastic noise while preserving stellar geometry.",
            "",
            "[1] INPUT REQUIREMENTS",
            " • Image State: Must be LINEAR (before stretching).",
            "   Silentium relies on physical statistics (Poisson distribution).",
            "   Do not apply to stretched images.",
            "",
            "[2] LUMINANCE CALIBRATION (The Tri-Layer Logic)",
            " • Noise Intensity: The Global Threshold.",
            "   Sets the baseline removal of salt & pepper noise across the image.",
            "   Adjust this first to remove coarse grain.",
            "",
            " • Detail Guard: Morphological Structure Protection.",
            "   It uses a 'Photometric Gate' to distinguish real structure from noise.",
            "   Unlike standard masks, it forces protection to ZERO on the background,",
            "   allowing maximum denoising on empty areas while creating an",
            "   impervious shell around faint filaments and galaxies.",
            "",
            " • Shadow Smoothness: The Exclusion Gate.",
            "   This applies aggressive cleaning (up to 4x) to the background floor.",
            "   It features a 'Hard Cutoff': it automatically shuts down as soon as",
            "   signal is detected (>25% probability), preventing the blurring of",
            "   highlights regardless of their size.",
            "",
            "[3] CHROMINANCE (Color Noise)",
            " • LAB Processing: Enabled via 'Chroma Denoise'.",
            "   Targets a* and b* channels to remove color blotches (green/magenta)",
            "   without affecting the structural detail in Luminance.",
            "",
            "[4] STAR PROTECTION (PSF)",
            " • Hard Masking: Silentium maps the Point Spread Function of stars via",
            "   Siril's 'findstar' command. This acts as a final safety net,",
            "   protecting stellar cores and profiles from any erosion.",
            "",
            "[5] PREVIEW & INTERACTION]",
            " • Global View: Static 'Fit' view. Use the mouse to aim (Red Box).",
            " • Loupe Mode: CLICK anywhere to enter 1:1 Pixel View (Interactive).",
            " • Compare: Hold SPACEBAR to flash the original image.",
            "",
            "[6] RECOMMENDED WORKFLOW",
            " 1. Set 'Shadow Smoothness' to 0 and 'Detail Guard' to 0.",
            " 2. Increase 'Noise Intensity' until coarse grain disappears (image will be soft).",
            " 3. Increase 'Detail Guard' until details/filaments snap back into focus.",
            " 4. Increase 'Shadow Smoothness' to creamy-fy the background floor.",
            "",
            "[7] THE SHADOW REPORT (Metrics)",
            "   Upon completion, Silentium prints a forensic analysis in this Console:",
            "   • SNR Improvement: The raw gain in Signal-to-Noise Ratio.",
            "   • Effective Integration: Multiplier of your exposure time (e.g. +3.0x).",
            "   • Pedestal Shift: Confirms that black point flux has been conserved.",
            "",
            "Support & Info: info@veralux.space",
            "=========================================================================="
        ]

        try:
            for line in guide_lines:
                msg = line if line.strip() else " "
                self.siril.log(msg)
        except Exception:
            print("\n".join(guide_lines))


def main():
    app = QApplication(sys.argv)
    siril = s.SirilInterface()

    try:
        siril.connect()
    except Exception:
        pass

    gui = SilentiumGUI(siril, app)
    gui.show()
    app.exec()


if __name__ == "__main__":
    main()