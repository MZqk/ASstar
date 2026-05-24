##############################################
# VeraLux — StarComposer
# High-Fidelity Star Reconstruction Engine
# Author: Riccardo Paterniti (2025)
# Contact: info@veralux.space
##############################################

# (c) 2025 Riccardo Paterniti
# VeraLux — StarComposer
# SPDX-License-Identifier: GPL-3.0-or-later
# Version 2.1.0
#
# Credits / Origin
# ----------------
#   • Architecture: Powered by VeraLux Core v2.1 (Hybrid Scalar/Vector Engine + Rational Tone Mapping)
#   • Stretch core: LogD-controlled rational tone-mapping curve (bounded, monotonic; 0→0, 1→1)
#   • Color Science: Hybrid Chrominance Recovery
#

"""
Overview
--------
A specialized photometric reconstruction engine designed for deep-sky 
astrophotography.

VeraLux StarComposer solves the "bloating" and "bleaching" issues inherent in 
standard star stretching by decoupling the stellar field from the main object.
It applies a LogD-controlled **rational tone-mapping** stretch to develop linear 
star masks with precision, preserving true stellar color and geometry (PSF) 
before compositing them onto non-linear starless images.

Key Features v2.1
-------------------
• **LogD Rational Stretch Engine**: Bounded rational tone-mapping core 
  delivering controlled star intensity expansion with preserved white cores 
  and stable halo geometry.
• **Toe-Based Profile Hardness (b)**: Surgical PSF shaping model. 
  Subtle response near default values, increasingly decisive toward extremes 
  for precise control of star compactness and halo spread.
• **Hybrid Scalar/Vector Physics**: Maintains luminance-solid white cores while 
  preserving chromatic ratios in stellar halos via Color Grip blending.
• **High-Fidelity Preview**: QHD architecture with energy-preserving 
  interpolation ensures strict visual consistency (WYSIWYG) between preview 
  and final render.
• **Lightweight Core**: Optimized execution without heavy scientific dependencies.
• **Smart Workflow**: Autosave to source directory, Green-light UI feedback, 
  and seamless Siril integration.

Design Goals
------------
• Treat stars as geometric entities (Gaussian/Moffat profiles).
• Preserve stellar chrominance ratios while controlling brightness and profile geometry.
• Repair optical defects (Chromatic Aberration) via chroma-only Gaussian smoothing.
• Automate the cleanup of star mask residual artifacts (Shadow Convergence).

Core Features
-------------
• **The VeraLux Engine**: LogD-controlled rational tone-mapping stretch for star-profile shaping.
• **Hybrid Physics**: Color Grip blends between Scalar (Crisp) and Vector (Color) modes.
• **Star Surgery**: Includes Optical Healing for halos and Dynamic LSR for galaxy core removal.

Usage
-----
1. **Load Data**: Import Starless (Non-Linear Base) and Starmask (Linear).
2. **Blend Mode**: Use 'Screen' if mask has bright residuals, 'Linear Add' otherwise.
3. **Intensity**: Increase "Star Intensity" (Gold Slider) to define brightness.
4. **Geometry**: Adjust "Profile Hardness" (b) to sculpt the PSF (Sharpen/Soften).
5. **Generate**: Click PROCESS. Result is loaded into Siril.

Inputs & Outputs
----------------
Input: Linear Starmask FITS + Stretched Starless FITS.
Output: Recombined RGB FITS (32-bit Float).

Compatibility
-------------
• Siril 1.3+
• Python 3.10+ (via sirilpy)
• Dependencies: sirilpy, PyQt6, numpy, opencv-python

License
-------
Released under GPL-3.0-or-later.
"""

import sys
import os
import math
import webbrowser

try:
    import sirilpy as s
except ImportError:
    print("Error: sirilpy module not found.")
    sys.exit(1)

s.ensure_installed("PyQt6", "numpy", "opencv-python", "astropy")

import numpy as np
import astropy.io.fits as fits
import cv2 

from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                            QWidget, QLabel, QDoubleSpinBox, QSlider,
                            QPushButton, QGroupBox, QMessageBox, QProgressBar,
                            QComboBox, QCheckBox, QFileDialog, QGraphicsView, 
                            QGraphicsScene, QGraphicsPixmapItem, QButtonGroup)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QEvent, QSettings, QPointF
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QBrush, QPainterPath

# ---------------------
#  THEME & STYLING
# ---------------------

DARK_STYLESHEET = """
QWidget { background-color: #2b2b2b; color: #e0e0e0; font-size: 10pt; }
QToolTip { background-color: #333333; color: #ffffff; border: 1px solid #88aaff; }
QGroupBox { border: 1px solid #444444; margin-top: 5px; font-weight: bold; border-radius: 4px; padding-top: 12px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; color: #88aaff; }
QLabel { color: #cccccc; }

QCheckBox { spacing: 5px; color: #cccccc; }
QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #666666; background: #3c3c3c; border-radius: 3px; }
QCheckBox::indicator:checked { background-color: #285299; border: 1px solid #88aaff; }

/* Robust Slider Styling */
QSlider { min-height: 24px; }
QSlider::groove:horizontal { background: #444444; height: 6px; border-radius: 3px; }
QSlider::sub-page:horizontal { background: transparent; }
QSlider::add-page:horizontal { background: transparent; }
QSlider::handle:horizontal { 
    background-color: #cccccc; border: 1px solid #666666; 
    width: 14px; height: 14px; margin: -4px 0; border-radius: 7px; 
}
QSlider::handle:horizontal:hover { background-color: #ffffff; border-color: #88aaff; }
QSlider::handle:horizontal:pressed { background-color: #ffffff; border-color: #ffffff; }

/* Master Control Handle (Solid Gold) */
QSlider#MainSlider::handle:horizontal { background-color: #ffb000; border: 1px solid #cc8800; }
QSlider#MainSlider::handle:horizontal:hover { background-color: #ffcc00; border-color: #ffffff; }
QSlider#MainSlider::groove:horizontal { background: #554400; }

/* Discrete / Fine Tune Slider Styling */
QSlider#FineTuneSlider { min-height: 20px; }
QSlider#FineTuneSlider::groove:horizontal { background: #333333; height: 4px; border-radius: 2px; }
QSlider#FineTuneSlider::handle:horizontal { 
    background-color: #888888; border: 1px solid #555555; 
    width: 10px; height: 14px; margin: -5px 0; border-radius: 3px; /* Rectangular handle */
}
QSlider#FineTuneSlider::handle:horizontal:hover { background-color: #aaaaaa; border-color: #88aaff; }

QPushButton { background-color: #444444; color: #dddddd; border: 1px solid #666666; border-radius: 4px; padding: 6px; font-weight: bold;}
QPushButton:hover { background-color: #555555; border-color: #777777; }

QPushButton#ProcessButton { background-color: #285299; border: 1px solid #1e3f7a; }
QPushButton#ProcessButton:hover { background-color: #355ea1; }

QPushButton#CloseButton { background-color: #5a2a2a; border: 1px solid #804040; }
QPushButton#CloseButton:hover { background-color: #7a3a3a; }

/* GHOST HELP BUTTON */
QPushButton#HelpButton { 
    background-color: transparent; 
    color: #555555; 
    border: none; 
    font-weight: bold; 
    min-width: 20px;
}
QPushButton#HelpButton:hover { 
    color: #aaaaaa; 
}

QPushButton#ZoomBtn { min-width: 30px; font-weight: bold; background-color: #3c3c3c; }

QPushButton#CoffeeButton { background-color: transparent; border: none; font-size: 15pt; padding: 2px; }
QPushButton#CoffeeButton:hover { background-color: rgba(255, 255, 255, 20); border-radius: 4px; }

/* HMS-Style ComboBox */
QComboBox { background-color: #3c3c3c; color: #ffffff; border: 1px solid #555555; padding: 3px; border-radius: 3px; }
QComboBox:hover { border-color: #777777; }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox::down-arrow { width: 0; height: 0; border-left: 4px solid transparent; border-right: 4px solid transparent; border-top: 6px solid #aaaaaa; margin-right: 6px; }
QComboBox QAbstractItemView { background-color: #3c3c3c; color: #ffffff; selection-background-color: #285299; border: 1px solid #555555; }
"""

VERSION = "2.1.0"

# ------------------------------------------------------------------------------
# VERSION HISTORY
# ------------------------------------------------------------------------------
# 2.1.0: Rational Stretch Engine Update.
#        • Replaced legacy IHS-based stretch core with a LogD-controlled
#          bounded rational tone-mapping curve (0→0, 1→1).
#        • Introduced Toe-Based Profile Hardness (b) for controlled PSF shaping
#          with stable core/halo balance.
# 2.0.3: "Buy me a coffee" button added.
# 2.0.2: Restore Astropy-based FITS loader to resolve corrupted data normalization
#.       causing black starmask display in files with negative data.
# 2.0.1: Enhanced numerical robustness by hardening input sanitization for  
#        non-standard 32-bit FITS data (prevents mathematical overflows).
# 2.0.0: Major Architecture & Engine Overhaul.
#        • Hybrid "Scalar Core" Engine: Solves the "soft/flat star" issue.
#          Stars now stretch R/G/B independently to achieve physical white cores
#          while preserving halo chromaticity via the new Color Grip logic.
#        • High-Performance Core: Removed 'astropy' dependency for instant load
#          times. Implemented multi-threaded processing (QThread) to prevent
#          UI freezing during parameter adjustments.
#        • Smart Proxy Preview: New generation logic with Morphological Dilation
#          (Max Pooling) ensures stars remain visible and bright even when
#          zoomed out. Added Live Histogram & Clipping Diagnostics overlay.
# 1.0.5: Metadata Preservation Fix. Switched output generation logic to use Siril's 
#        native I/O pipeline. Now preserves original FITS Headers, WCS coordinates, 
#        and Plate Solving data in the processed image.
# 1.0.4: "Nearest Neighbor" sampling (Slicing) preview fix (invisible starmask)
# 1.0.3: Sensor profiles update (v2.2)
# 1.0.2: Sensor DB update (same as VeraLux HyperMetric Stretch).
# 1.0.1: Import fix.
# ------------------------------------------------------------------------------

# =============================================================================
#  SENSOR PROFILES (Database v2.2 - Siril SPCC Derived)
# =============================================================================

SENSOR_PROFILES = {
    "Rec.709 (Recommended)": (0.2126, 0.7152, 0.0722),
    "Sony IMX571 (ASI2600/QHY268)": (0.2944, 0.5021, 0.2035),
    "Sony IMX533 (ASI533)": (0.2910, 0.5072, 0.2018),
    "Sony IMX455 (ASI6200/QHY600)": (0.2987, 0.5001, 0.2013),
    "Sony IMX410 (ASI2400)": (0.3015, 0.5050, 0.1935),
    "Sony IMX269 (Altair/ToupTek)": (0.3040, 0.5010, 0.1950),
    "Sony IMX294 (ASI294)": (0.3068, 0.5008, 0.1925),
    "Sony IMX676 (ASI676)": (0.2880, 0.5100, 0.2020),
    "Sony IMX183 (ASI183)": (0.2967, 0.4983, 0.2050),
    "Sony IMX178 (ASI178)": (0.2346, 0.5206, 0.2448),
    "Sony IMX224 (ASI224)": (0.3402, 0.4765, 0.1833),
    "Sony IMX585 (ASI585) - STARVIS 2": (0.3431, 0.4822, 0.1747),
    "Sony IMX662 (ASI662) - STARVIS 2": (0.3430, 0.4821, 0.1749),
    "Sony IMX678 (ASI678) - STARVIS 2": (0.3426, 0.4825, 0.1750),
    "Sony IMX715 (ASI715) - STARVIS 2": (0.3410, 0.4840, 0.1750),
    "Sony IMX462 (ASI462)": (0.3333, 0.4866, 0.1801),
    "Sony IMX482 (ASI482)": (0.3150, 0.4950, 0.1900),
    "Panasonic MN34230 (ASI1600/QHY163)": (0.2650, 0.5250, 0.2100),
    "Canon EOS (Modern - 60D/600D/500D)": (0.2600, 0.5200, 0.2200),
    "Canon EOS (Legacy - 300D/40D/20D)": (0.2450, 0.5350, 0.2200),
    "Nikon DSLR (Modern - D5100/D7200)": (0.2650, 0.5100, 0.2250),
    "Nikon DSLR (Legacy - D3/D300/D90)": (0.2500, 0.5300, 0.2200),
    "Fujifilm X-Trans 5 HR": (0.2800, 0.5100, 0.2100),
    "ZWO Seestar S50": (0.3333, 0.4866, 0.1801),
    "ZWO Seestar S30": (0.2928, 0.5053, 0.2019),
    "Narrowband HOO": (0.5000, 0.2500, 0.2500),
    "Narrowband SHO": (0.3333, 0.3400, 0.3267),
}

DEFAULT_PROFILE = "Rec.709 (Recommended)"

# =============================================================================
#  CORE MATH (VeraLux Core v2.1 - Hybrid Scalar/Vector + Rational Stretch)
# =============================================================================

class VeraLuxCore:
    @staticmethod
    def normalize_input(img_data):
        input_dtype = img_data.dtype
        img_float = img_data.astype(np.float32)
        img_float = np.nan_to_num(img_float, nan=0.0, posinf=1.0, neginf=0.0)
        if np.issubdtype(input_dtype, np.integer):
            if input_dtype == np.uint8: return img_float / 255.0
            elif input_dtype == np.uint16: return img_float / 65535.0
            else: return img_float / float(np.iinfo(input_dtype).max)
        elif np.issubdtype(input_dtype, np.floating):
            current_max = np.max(img_data)
            # Standard normalization for data > 1.0
            if current_max > 1.0 + 1e-5:
                if current_max <= 65535.0: return img_float / 65535.0
                return img_float / current_max
        return np.clip(img_float, 0.0, 1.0)

    @staticmethod
    def calculate_anchor_adaptive(data_norm, weights):
        stride = max(1, data_norm.size // 1000000) 
        if data_norm.ndim == 3:
            r, g, b = weights
            L = r * data_norm[0] + g * data_norm[1] + b * data_norm[2]
            sample = L.flatten()[::stride]
        else:
            sample = data_norm.flatten()[::stride]
            
        valid = sample[sample > 0]
        if valid.size == 0: return 0.0
        
        sparsity = valid.size / sample.size
        if sparsity < 0.05:
             return 0.0
             
        return max(0.0, np.percentile(valid, 5.0))

    @staticmethod
    def extract_luminance(data_norm, anchor, weights):
        r_w, g_w, b_w = weights
        img_anchored = np.maximum(data_norm - anchor, 0.0)
        if data_norm.ndim == 3:
            L = (r_w * img_anchored[0] + g_w * img_anchored[1] + b_w * img_anchored[2])
        else:
            L = img_anchored
        return L, img_anchored

    @staticmethod
    def rational_tonemap(data, D, b):
        """VeraLux StarComposer stretch core.

        This keeps the existing UX/controls (LogD is provided as D = 10**LogD; Profile Hardness 'b')
        and applies a bounded rational tone-mapping curve:

            y = (k * x) / ((k - 1) * x + 1)
            with k = a ** sf

        Notes:
        - `D` arrives as 10**(LogD) from the UI, so we recover LogD via log10(D).
        - `b` shapes the star profile via a toe control (subtle near the default, strong at extremes).
        """
        # Sanitize inputs
        x = np.clip(data, 0.0, 1.0).astype(np.float32)
        D = float(max(D, 1e-12))
        b = float(max(b, 0.1))

        # Recover UI LogD (since caller provides D_val = 10**LogD)
        logD = math.log10(D)

        # Map LogD (UI ~[1..21]) to stretch factor for the rational curve.
        # logD=1  -> 0.0
        # logD=21 -> 10.0
        # (sf is clamped to 12.0 max)
        sf = (logD - 1.0) / 2.0
        sf = max(0.0, min(sf, 12.0))

        # Rational curve strength
        a = 3.0
        k = a ** sf

        # Hardness shaping WITHOUT gamma ("toe" control) — corrected:
        # We want b to be subtle near the default (~50) and increasingly strong at extremes.
        # We apply a rational toe transform:
        #
        #     x_toe = x / (x + t*(1-x))
        #
        # IMPORTANT: t=1.0 is neutral (x_toe == x). Do NOT use t=0 as neutral.
        #
        # Mapping (surgical via cubic):
        #   b=50   -> t = 1.0 (neutral)
        #   b>50   -> t > 1.0 (harder: suppress low wings, tighter stars)
        #   b<50   -> t < 1.0 (softer: lift low wings, wider stars)

        # Normalize b around default 50
        u = (b - 50.0) / 50.0
        u = max(-1.5, min(1.5, u))

        # Cubic mapping: very gentle near 0, strong toward extremes
        s = u * u * u

        # Toe strength. Start conservative.
        strength = 0.60
        t = 1.0 + strength * s
        t = max(t, 1e-3)

        # Apply toe shaping (stable and monotone for t>0)
        eps_toe = 1e-9
        denom = x + t * (1.0 - x)  # t=1 => denom=1 => x_n=x
        denom = np.maximum(denom, eps_toe)
        x_n = x / denom
        x_n = np.clip(x_n, 0.0, 1.0)

        # Apply the rational tone-mapping curve
        den = ((k - 1.0) * x_n) + 1.0
        y = (k * x_n) / den

        return np.clip(y, 0.0, 1.0).astype(np.float32)

# =============================================================================
#  STAR PIPELINE (Hybrid Engine)
# =============================================================================

def apply_optical_healing(img_rgb, strength):
    if strength <= 0: return img_rgb
    img_cv = img_rgb.transpose(1, 2, 0)
    ycrcb = cv2.cvtColor(img_cv, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    ksize = int(strength * 2) + 1
    if ksize % 2 == 0: ksize += 1
    cr = cv2.GaussianBlur(cr, (ksize, ksize), 0)
    cb = cv2.GaussianBlur(cb, (ksize, ksize), 0)
    merged = cv2.merge([y, cr, cb])
    rgb_heal = cv2.cvtColor(merged, cv2.COLOR_YCrCb2RGB)
    return rgb_heal.transpose(2, 0, 1)

def apply_star_reduction(img_rgb, intensity):
    if intensity <= 0: return img_rgb
    k_size = 3 if intensity < 0.5 else 5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    img_hwc = img_rgb.transpose(1, 2, 0)
    eroded = cv2.erode(img_hwc, kernel, iterations=1)
    return (img_hwc * (1.0 - intensity) + eroded * intensity).transpose(2, 0, 1)

def apply_large_structure_rejection(img_rgb, intensity):
    """
    Core Rejection (LSR): Removes large blobs using Difference of Gaussians.
    Dynamic Kernel Size: scales with image to target actual structures.
    """
    if intensity <= 0: return img_rgb
    
    h, w = img_rgb.shape[1], img_rgb.shape[2]
    # Dynamic Kernel: 1/15th of image
    k_size_val = int(min(h, w) / 15.0) 
    if k_size_val % 2 == 0: k_size_val += 1 
    if k_size_val < 3: k_size_val = 3
    
    img_hwc = img_rgb.transpose(1, 2, 0)
    low_pass = cv2.GaussianBlur(img_hwc, (k_size_val, k_size_val), 0)
    high_pass = np.maximum(img_hwc - low_pass, 0.0)
    result = img_hwc * (1.0 - intensity) + high_pass * intensity
    
    return result.transpose(2, 0, 1)

def process_star_pipeline(starmask, D, b, grip, shadow, reduction, healing, lsr, weights, use_adaptive):
    # 1. Normalization
    img = VeraLuxCore.normalize_input(starmask)
    if img.ndim == 2: img = np.array([img, img, img]) 

    # 2. Input Signal Conditioning
    img = np.clip(img, 0.0, 1.0)
    
    # --- TRANSITION SMOOTHING (Micro-Blur) ---
    img_hwc = img.transpose(1, 2, 0)
    img_hwc = cv2.GaussianBlur(img_hwc, (0, 0), 0.5)
    img = img_hwc.transpose(2, 0, 1)

    # 3. Preparation (Anchoring)
    anchor = VeraLuxCore.calculate_anchor_adaptive(img, weights) if use_adaptive else 0.0
    img_anchored = np.maximum(img - anchor, 0.0)
    
    D_val = 10.0 ** D
    
    # --- HYBRID ENGINE v2.1 ---
    # A. Scalar Mapping (per-channel rational tone mapping)
    scalar = np.zeros_like(img)
    scalar[0] = VeraLuxCore.rational_tonemap(img_anchored[0], D_val, b)
    scalar[1] = VeraLuxCore.rational_tonemap(img_anchored[1], D_val, b)
    scalar[2] = VeraLuxCore.rational_tonemap(img_anchored[2], D_val, b)
    scalar = np.clip(scalar, 0.0, 1.0)
    
    # B. Vector Mapping (luminance-driven, ratio-preserving)
    if grip > 0.001:
        L_anchored, _ = VeraLuxCore.extract_luminance(img, anchor, weights)
        L_str = VeraLuxCore.rational_tonemap(L_anchored, D_val, b)
        L_str = np.clip(L_str, 0.0, 1.0)
        
        epsilon = 1e-9; L_safe = L_anchored + epsilon
        r_ratio = img_anchored[0] / L_safe
        g_ratio = img_anchored[1] / L_safe
        b_ratio = img_anchored[2] / L_safe
        
        vector = np.zeros_like(img)
        vector[0] = L_str * r_ratio
        vector[1] = L_str * g_ratio
        vector[2] = L_str * b_ratio
        vector = np.clip(vector, 0.0, 1.0)
    else:
        vector = scalar 
    
    # 4. Blending & Shadow Convergence (damping derived from scalar luminance)
    if grip > 0.001:
        grip_map = np.full_like(scalar[0], grip)
        if shadow > 0.01:
            r_w, g_w, b_w = weights
            L_ref = (r_w * scalar[0]) + (g_w * scalar[1]) + (b_w * scalar[2])
            damping = np.power(L_ref, shadow)
            grip_map = grip_map * damping
        final = (vector * grip_map) + (scalar * (1.0 - grip_map))
    else:
        final = scalar

    final = np.clip(final, 0.0, 1.0).astype(np.float32)
    
    # 5. Surgery
    if lsr > 0: final = apply_large_structure_rejection(final, lsr)
    if healing > 0: final = apply_optical_healing(final, healing)
    if reduction > 0: final = apply_star_reduction(final, reduction)
    
    return final

# =============================================================================
#  GUI WIDGETS
# =============================================================================

class ResetSlider(QSlider):
    def __init__(self, orientation, default_val=0, parent=None):
        super().__init__(orientation, parent)
        self.default_val = default_val
    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setValue(self.default_val); event.accept()
        else: super().mouseDoubleClickEvent(event)

class HistogramOverlay(QWidget):
    """
    VeraLux Standard Histogram Overlay.
    Draws RGB histogram with clipping bars and textual HUD stats.
    """
    stats_updated = pyqtSignal(str)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.hist_data = None
        self.use_log = False
        
        # State stats
        self.pct_black = 0.0; self.pct_white = 0.0
        self.black_level = 0; self.white_level = 0
        self.black_count = 0; self.white_count = 0
        
        # Dimensions
        self.setFixedHeight(120) 
        self.setFixedWidth(200)

        self.info_mode_active = False

    def set_info_mode(self, active):
        self.info_mode_active = active
        if active:
            self.setToolTip("") 
        else:
            if self.hist_data:
                self.update()
        
    def set_log_scale(self, val):
        self.use_log = val
        self.update()

    def set_data(self, img):
        if img is None: return
        bins = 256
        # Histogram Calculation
        r = np.histogram(img[0], bins=bins, range=(0, 1))[0]
        g = np.histogram(img[1], bins=bins, range=(0, 1))[0]
        b = np.histogram(img[2], bins=bins, range=(0, 1))[0]
        mx = max(np.max(r), np.max(g), np.max(b))
        if mx > 0: self.hist_data = (r/mx, g/mx, b/mx)
        else: self.hist_data = None
        
        # Stats Calculation
        h, w = img.shape[1], img.shape[2]
        tot = h * w
        epsilon = 1e-7
        
        # Clipping masks (Physical)
        white_px = np.any(img >= (1.0 - epsilon), axis=0)
        self.white_count = int(np.count_nonzero(white_px))
        self.pct_white = (self.white_count / tot) * 100.0 if tot > 0 else 0.0

        black_px = np.any(img <= epsilon, axis=0)
        self.black_count = int(np.count_nonzero(black_px))
        self.pct_black = (self.black_count / tot) * 100.0 if tot > 0 else 0.0

        # Levels
        def get_lvl(p): return 2 if p >= 0.1 else (1 if p >= 0.01 else 0)
        self.white_level = get_lvl(self.pct_white)
        self.black_level = get_lvl(self.pct_black)

        # Tooltip HTML
        report = self._generate_html_report(tot)
        
        if self.info_mode_active:
            self.setToolTip("")
        else:
            self.setToolTip(report)
            
        self.stats_updated.emit(report)
        self.update()

    def _generate_html_report(self, tot):
        colors = {0: "#cccccc", 1: "Orange", 2: "#ff4444"}
        c_blk = colors[self.black_level]
        c_wht = colors[self.white_level]
        
        html = "<div style='font-size:9pt; font-weight:bold; color:#eeeeee;'>Histogram Analysis</div>"
        html += f"<div style='margin-top:4px;'>Blacks: <span style='color:{c_blk};'>{self.pct_black:.2f}%</span> <span style='font-size:8pt; color:#999999;'>({self.black_count} px)</span></div>"
        html += f"<div>Whites: <span style='color:{c_wht};'>{self.pct_white:.2f}%</span> <span style='font-size:8pt; color:#999999;'>({self.white_count} px)</span></div>"
        html += "<div style='margin-top:4px; font-size:8pt; color:#888;'><i>Note: Black clipping is expected in starmasks.</i></div>"
        return html

    def paintEvent(self, event):
        if not self.hist_data: return
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        
        # Background
        p.setBrush(QColor(0,0,0,180)); p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(0,0,w,h,5,5)
        
        # Curves
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        cols = [QColor(255,50,50,180), QColor(50,255,50,180), QColor(50,100,255,180)]
        step = w/256
        log_norm = math.log10(1001)
        
        for i, ch in enumerate(self.hist_data):
            path = QPainterPath(); path.moveTo(0, h)
            for x, val in enumerate(ch):
                h_val = val
                if self.use_log and val > 0: h_val = math.log10(1 + val * 1000) / log_norm
                path.lineTo(x*step, h - h_val*(h-20)) 
            path.lineTo(w, h); path.closeSubpath()
            p.setBrush(cols[i]); p.drawPath(path)
            
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        
        # --- HUD & INDICATORS ---
        font = p.font(); font.setPointSize(8); font.setBold(True); p.setFont(font)
        
        # Color Map
        cols_map = {1: QColor(255, 165, 0, 200), 2: QColor(255, 50, 50, 200)}
        text_cols = {0: QColor(200,200,200), 1: QColor(255, 165, 0), 2: QColor(255, 50, 50)}

        # 1. Side Bars
        if self.black_level > 0:
            p.setPen(Qt.PenStyle.NoPen); p.setBrush(cols_map[self.black_level])
            p.drawRect(0, 0, 3, h)
            
        if self.white_level > 0:
            p.setPen(Qt.PenStyle.NoPen); p.setBrush(cols_map[self.white_level])
            p.drawRect(w - 3, 0, 3, h)

        # 2. HUD Text (Top Corners)
        # BLACKS
        p.setPen(text_cols[self.black_level])
        p.drawText(5, 12, f"BLK: {self.pct_black:.2f}%")
        
        # WHITES
        w_txt = f"WHT: {self.pct_white:.2f}%"
        fm = p.fontMetrics(); tw = fm.horizontalAdvance(w_txt)
        p.setPen(text_cols[self.white_level])
        p.drawText(w - tw - 5, 12, w_txt)

# =============================================================================
#  MAIN GUI
# =============================================================================

class PreviewWorker(QThread):
    result_ready = pyqtSignal(object)

    def __init__(self, sm, sl, params, use_screen):
        super().__init__()
        self.sm = sm
        self.sl = sl
        self.p = params
        self.use_screen = use_screen

    def run(self):
        # Unpack parameters
        stars = process_star_pipeline(
            self.sm, self.p['D'], self.p['b'],
            self.p['grip'], self.p['shadow'], self.p['red'],
            self.p['heal'], self.p['lsr'], self.p['w'], self.p['adapt']
        )
        
        if self.sl is not None:
            # Handle resizing if dimensions differ (safety check)
            if self.sl.shape != stars.shape:
                h, w = min(self.sl.shape[1], stars.shape[1]), min(self.sl.shape[2], stars.shape[2])
                sl_cut = self.sl[:, :h, :w]
                st_cut = stars[:, :h, :w]
            else:
                sl_cut, st_cut = self.sl, stars

            if self.use_screen:
                final = 1.0 - (1.0 - sl_cut) * (1.0 - st_cut)
            else:
                final = np.clip(sl_cut + st_cut, 0.0, 1.0)
        else:
            final = stars
            
        self.result_ready.emit(final)

class StarComposerGUI(QMainWindow):
    def __init__(self, siril, app):
        super().__init__()
        self.siril = siril
        self.app = app
        self.setWindowTitle(f"VeraLux StarComposer v{VERSION}")
        self.setStyleSheet(DARK_STYLESHEET)
        self.resize(1350, 650) 
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)

        # --- Persistent settings (QSettings) ---
        self.settings = QSettings("VeraLux", "StarComposer")
        
        self.sm_full = None; self.sl_full = None
        self.sm_proxy = None; self.sl_proxy = None
        self.comp_proxy = None
        self.pending_proxy = None
        self.request_fit = False 
        self.working_dir = os.getcwd()
        self.has_image = False
        self.is_fit_mode = True
        
        self.debounce = QTimer()
        self.debounce.setSingleShot(True); self.debounce.setInterval(150)
        self.debounce.timeout.connect(self.run_preview_logic)
        
        # Debounce for Slider Release (To allow Double-Click Reset)
        # This prevents the heavy full-res update from locking the UI 
        # immediately on the first click of a double-click.
        self.release_timer = QTimer()
        self.release_timer.setSingleShot(True); self.release_timer.setInterval(350)
        self.release_timer.timeout.connect(self.trigger_full_res_update)
        
        # --- HEADER LOG STARTUP ---
        header_msg = (
            "\n##############################################\n"
            "# VeraLux — StarComposer\n"
            "# High-Fidelity Star Reconstruction Engine\n"
            "# Author: Riccardo Paterniti (2025)\n"
            "# Contact: info@veralux.space\n"
            "##############################################"
        )
        try:
            self.siril.log(header_msg)
        except:
            print(header_msg)
        
        self.init_ui()
        
    def init_ui(self):
        main = QWidget(); self.setCentralWidget(main)
        layout = QHBoxLayout(main)
        
        # --- LEFT PANEL ---
        left_container = QWidget(); left_container.setFixedWidth(360)
        left = QVBoxLayout(left_container); left.setContentsMargins(0,0,0,0)
        
        # 1. Input
        g1 = QGroupBox("1. Input"); l1 = QVBoxLayout(g1)
        self.lbl_sm = QLabel("Starmask: [Empty]"); l1.addWidget(self.lbl_sm)
        self.b_sm = QPushButton("1: Load Starmask (Linear)")
        self.b_sm.setToolTip("Load a <b>Linear</b> starmask (required for correct color reconstruction).<br>This mask should come directly from StarNet/StarXTerminator before stretching.")
        self.b_sm.clicked.connect(self.load_starmask); l1.addWidget(self.b_sm)
        
        self.lbl_sl = QLabel("Starless: [Empty]"); l1.addWidget(self.lbl_sl)
        self.b_sl = QPushButton("2: Load Starless (Stretched)")
        self.b_sl.setToolTip("Load the <b>Non-Linear</b> starless image to use as background context.<br>This image should be already stretched.")
        self.b_sl.clicked.connect(self.load_starless); l1.addWidget(self.b_sl)
        
        # Blend Mode (Exclusive Checkboxes)
        l1.addWidget(QLabel("Composition Mode:"))
        row_blend = QHBoxLayout()
        
        # MODIFIED: Swapped Order, Screen is now Default/First
        self.chk_screen = QCheckBox("Screen (Safe)")
        self.chk_add = QCheckBox("Linear Add (Physical)")
        self.chk_screen.setChecked(True)
        
        self.chk_screen.setToolTip("<b>Screen:</b> Soft blend.<br>Prevents clipping and preserves galaxy cores under stars, but may lower local contrast.")
        self.chk_add.setToolTip("<b>Linear Add:</b> Physical light addition.<br>Preserves high contrast but risks clipping cores if the background is very bright.")
        
        # Logic Group for exclusivity
        self.grp_blend = QButtonGroup()
        self.grp_blend.addButton(self.chk_screen, 0)
        self.grp_blend.addButton(self.chk_add, 1)
        self.grp_blend.setExclusive(True)
        self.grp_blend.buttonToggled.connect(self.trigger_update)
        
        row_blend.addWidget(self.chk_screen)
        row_blend.addWidget(self.chk_add)
        l1.addLayout(row_blend)
        
        left.addWidget(g1)
        
        # 2. Sensor
        g2 = QGroupBox("2. Sensor Profile"); l2 = QVBoxLayout(g2)
        self.cmb_prof = QComboBox()
        self.cmb_prof.setToolTip("Select the sensor profile used to acquire the data.<br>This ensures accurate luminance extraction based on Quantum Efficiency.")
        for k in SENSOR_PROFILES:
            self.cmb_prof.addItem(k)

        # Restore last used profile
        try:
            saved_profile = self.settings.value("sensor_profile", DEFAULT_PROFILE, type=str)
        except Exception:
            saved_profile = DEFAULT_PROFILE
        if saved_profile in SENSOR_PROFILES:
            self.cmb_prof.setCurrentText(saved_profile)
        else:
            self.cmb_prof.setCurrentText(DEFAULT_PROFILE)

        # Persist selection on change
        self.cmb_prof.currentIndexChanged.connect(self.on_sensor_profile_changed)
        l2.addWidget(self.cmb_prof)
        left.addWidget(g2)
        
        # 3. Stretch
        g3 = QGroupBox("3. VeraLux Stretch"); l3 = QVBoxLayout(g3)
        self.lbl_D = QLabel("Star Intensity (Log D): 0.00")
        l3.addWidget(self.lbl_D)
        self.s_D = ResetSlider(Qt.Orientation.Horizontal, 0); self.s_D.setObjectName("MainSlider")
        self.s_D.setToolTip("<b>Star Intensity (Log D):</b><br>Controls the global brightness stretch.<br>Increase this to bring out faint stars. This is the primary control.")
        self.s_D.setRange(0, 1000); self.s_D.setValue(0)
        self.s_D.valueChanged.connect(self.update_labels)
        self.s_D.valueChanged.connect(self.trigger_update); l3.addWidget(self.s_D)
        
        self.lbl_b = QLabel("Profile Hardness (b): 50.0")
        l3.addWidget(self.lbl_b)
        
        # MODIFIED: Default to 50% (500)
        self.s_b = ResetSlider(Qt.Orientation.Horizontal, 500)
        self.s_b.setRange(10, 1000) 
        self.s_b.setValue(500)
        self.s_b.setToolTip(
        "<b>Profile Hardness (b):</b><br>"
        "Controls toe-based shaping of the star intensity profile.<br>"
        "• <b>50 (Neutral):</b> No profile bias.<br>"
        "• <b>Higher values:</b> Suppress low-intensity wings → tighter, crisper stars.<br>"
        "• <b>Lower values:</b> Lift low-intensity wings → wider, softer halos."
        )
        self.s_b.valueChanged.connect(self.update_labels)
        self.s_b.valueChanged.connect(self.trigger_update); l3.addWidget(self.s_b)
        
        # MODIFIED: Adaptive Anchor Row
        row_adapt = QHBoxLayout()
        self.chk_adapt = QCheckBox("Adaptive Anchor"); self.chk_adapt.setChecked(True)
        self.chk_adapt.setToolTip("<b>Adaptive Anchor:</b><br>Automatically detects the black point of the starmask to maximize contrast.<br>Keep enabled for best results.")
        self.chk_adapt.toggled.connect(self.trigger_update)
        row_adapt.addWidget(self.chk_adapt)
        
        row_adapt.addStretch()
                
        l3.addLayout(row_adapt)
        left.addWidget(g3)

        # 4. Physics
        g4 = QGroupBox("4. Physics (Hybrid)"); l4 = QVBoxLayout(g4)
        
        self.lbl_grip = QLabel("Color Grip (Blend): 50%")
        l4.addWidget(self.lbl_grip)
        
        # MODIFIED: Default to 50%
        self.s_grip = ResetSlider(Qt.Orientation.Horizontal, 50)
        self.s_grip.setRange(0, 100); self.s_grip.setValue(50)
        self.s_grip.setToolTip(
        "<b>Color Grip (Hybrid Blend):</b><br>"
        "Controls blending between per-channel (Scalar) and luminance-driven (Vector) mapping.<br>"
        "• <b>0% (Scalar):</b> Per-channel tone mapping. Stronger white cores, higher perceived sharpness.<br>"
        "• <b>100% (Vector):</b> Luminance-based mapping with RGB ratio preservation. "
        "More chromatic fidelity, slightly softer cores."
        )        
        self.s_grip.valueChanged.connect(self.update_labels)
        self.s_grip.valueChanged.connect(self.trigger_update); l4.addWidget(self.s_grip)
        
        self.lbl_shad = QLabel("Shadow Conv (Hide Artifacts): 0.0")
        l4.addWidget(self.lbl_shad)
        self.s_shad = ResetSlider(Qt.Orientation.Horizontal, 0); self.s_shad.setRange(0, 300); self.s_shad.setValue(0)
        self.s_shad.setToolTip("<b>Shadow Convergence:</b><br>Suppresses artifacts in the background.<br>Increases Scalar engine influence in dark areas to hide chromatic noise left by star removal tools.")
        self.s_shad.valueChanged.connect(self.update_labels)
        self.s_shad.valueChanged.connect(self.trigger_update); l4.addWidget(self.s_shad)
        left.addWidget(g4)
        
        # 5. Surgery
        self.chk_surgery = QCheckBox("Show Star Surgery (Advanced)")
        self.chk_surgery.setToolTip("Reveal advanced tools for morphological reduction and optical correction.")
        self.chk_surgery.toggled.connect(self.toggle_surgery)
        left.addWidget(self.chk_surgery)
        
        self.g5 = QGroupBox("5. Star Surgery"); l5 = QVBoxLayout(self.g5)
        
        # LSR
        self.lbl_lsr = QLabel("Core Rejection (LSR): 0%")
        l5.addWidget(self.lbl_lsr)
        self.s_lsr = ResetSlider(Qt.Orientation.Horizontal, 0); self.s_lsr.setRange(0, 100); self.s_lsr.setValue(0)
        self.s_lsr.setToolTip("<b>Large Structure Rejection (LSR):</b><br>Removes large non-stellar structures (like galaxy cores) from the starmask.")
        self.s_lsr.valueChanged.connect(self.update_labels)
        self.s_lsr.valueChanged.connect(self.trigger_update); l5.addWidget(self.s_lsr)

        self.lbl_red = QLabel("Reduction (Erosion): 0%")
        l5.addWidget(self.lbl_red)
        self.s_red = ResetSlider(Qt.Orientation.Horizontal, 0); self.s_red.setRange(0, 100); self.s_red.setValue(0)
        self.s_red.setToolTip("<b>Morphological Reduction:</b><br>Applies morphological erosion to physically shrink star diameters.<br><i>Use with caution.</i>")
        self.s_red.valueChanged.connect(self.update_labels)
        self.s_red.valueChanged.connect(self.trigger_update); l5.addWidget(self.s_red)
        
        self.lbl_heal = QLabel("Optical Healing (Halos): 0.0")
        l5.addWidget(self.lbl_heal)
        self.s_heal = ResetSlider(Qt.Orientation.Horizontal, 0); self.s_heal.setRange(0, 20); self.s_heal.setValue(0)
        self.s_heal.setToolTip("<b>Optical Healing:</b><br>Repairs chromatic aberration (colored halos) by aligning chrominance channels to luminance.<br><br><b>Note:</b> Works best close to star cores. Not intended to remove large-scale halos.")
        self.s_heal.valueChanged.connect(self.update_labels)
        self.s_heal.valueChanged.connect(self.trigger_update); l5.addWidget(self.s_heal)
        self.g5.hide() 
        left.addWidget(self.g5)

        # Full Resolution Triggers (On Release/Edit Finished)
        # Note: We use release_timer to buffer the heavy update, allowing double-clicks to pass through.
        self.s_D.sliderReleased.connect(self.release_timer.start)
        self.s_b.sliderReleased.connect(self.release_timer.start)
        self.s_grip.sliderReleased.connect(self.release_timer.start)
        self.s_shad.sliderReleased.connect(self.release_timer.start)
        self.s_lsr.sliderReleased.connect(self.release_timer.start)
        self.s_red.sliderReleased.connect(self.release_timer.start)
        self.s_heal.sliderReleased.connect(self.release_timer.start)
        
        self.chk_adapt.clicked.connect(self.trigger_full_res_update)
        self.grp_blend.buttonClicked.connect(self.trigger_full_res_update)
        self.cmb_prof.currentIndexChanged.connect(self.trigger_full_res_update)
        
        # Buttons
        footer = QHBoxLayout()
        
        self.btn_help = QPushButton("?"); self.btn_help.setObjectName("HelpButton"); self.btn_help.setFixedWidth(20)
        self.btn_help.setToolTip("Print Operational Guide to Siril Console")
        self.btn_help.clicked.connect(self.print_help_to_console)
        footer.addWidget(self.btn_help)
        
        b_res = QPushButton("Defaults"); b_res.clicked.connect(self.set_defaults)
        b_res.setToolTip("Reset all parameters to optimal starting values.")
        footer.addWidget(b_res)
        
        b_cls = QPushButton("Close"); b_cls.setObjectName("CloseButton")
        b_cls.setToolTip("Close the application.")
        b_cls.clicked.connect(self.close)
        footer.addWidget(b_cls)
        
        b_proc = QPushButton("PROCESS"); b_proc.setObjectName("ProcessButton")
        b_proc.setToolTip("Compute full-resolution image, save as 'VeraLux_StarComposer_result.fit' and load into Siril.")
        b_proc.clicked.connect(self.process_full_resolution)
        footer.addWidget(b_proc)
        
        left.addLayout(footer)
        left.addStretch()
        
        layout.addWidget(left_container)
        
        # --- RIGHT PANEL (Preview) ---
        right = QVBoxLayout()
        
        # Toolbar
        tb = QHBoxLayout()
        b_out = QPushButton("-"); b_out.setObjectName("ZoomBtn"); b_out.clicked.connect(self.zoom_out)
        b_fit = QPushButton("Fit"); b_fit.setObjectName("ZoomBtn"); b_fit.clicked.connect(self.fit_view)
        b_11 = QPushButton("1:1"); b_11.setObjectName("ZoomBtn"); b_11.clicked.connect(self.zoom_1to1)
        b_in = QPushButton("+"); b_in.setObjectName("ZoomBtn"); b_in.clicked.connect(self.zoom_in)
        
        lbl_hint = QLabel("Double-click to fit")
        lbl_hint.setStyleSheet("color: #ffb000; font-size: 8pt; font-style: italic; margin-left: 10px;")
        
        self.b_hist = QPushButton("Hist"); self.b_hist.setObjectName("ZoomBtn"); self.b_hist.setCheckable(True)
        self.b_hist.setToolTip("Toggle Histogram overlay.")
        self.b_hist.setChecked(True); self.b_hist.clicked.connect(self.toggle_hist)

        self.chk_ontop = QCheckBox("On Top")
        self.chk_ontop.setToolTip("Keep window above Siril.")
        self.chk_ontop.setChecked(True)
        self.chk_ontop.toggled.connect(self.toggle_ontop)
        self.chk_ontop.setStyleSheet("color: #cccccc; font-weight: bold; margin-left: 10px;")
        
        self.btn_coffee = QPushButton("☕")
        self.btn_coffee.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_coffee.setObjectName("CoffeeButton")
        self.btn_coffee.setToolTip("Support the developer - Buy me a coffee!")
        self.btn_coffee.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_coffee.clicked.connect(lambda: webbrowser.open("https://buymeacoffee.com/riccardo.paterniti"))

        tb.addWidget(b_out); tb.addWidget(b_fit); tb.addWidget(b_11); tb.addWidget(b_in)
        tb.addWidget(lbl_hint)

        tb.addStretch()
        
        self.pbar = QProgressBar()
        self.pbar.setRange(0, 0)
        self.pbar.setFixedWidth(500)
        self.pbar.setFixedHeight(6)
        self.pbar.setStyleSheet("QProgressBar { border: 1px solid #555; border-radius: 3px; background: #222; } QProgressBar::chunk { background-color: #ffb000; }")
        self.pbar.hide()
        tb.addWidget(self.pbar)

        tb.addStretch()
        
        tb.addWidget(self.b_hist)
        tb.addWidget(self.chk_ontop)
        tb.addWidget(self.btn_coffee)
        right.addLayout(tb)
        
        # View
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setStyleSheet("background-color: #151515; border: none;")
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.view.viewport().installEventFilter(self)
        self.view.installEventFilter(self)
        right.addWidget(self.view)
        
        self.pix_item = QGraphicsPixmapItem(); self.scene.addItem(self.pix_item)
        self.pix_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        
        # Floating Overlays
        
        # 1. Box Info Overlay
        self.lbl_info_overlay = QLabel(self.view)
        self.lbl_info_overlay.setFixedWidth(200)
        self.lbl_info_overlay.setStyleSheet("background-color: rgba(0, 0, 0, 180); border: none; border-radius: 5px; color: #eeeeee; padding: 6px;")
        self.lbl_info_overlay.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.lbl_info_overlay.setWordWrap(True)
        self.lbl_info_overlay.hide()

        # 2. Checkbox Log Scale
        self.chk_log = QCheckBox("Log Scale", self.view)
        self.chk_log.setToolTip("Toggle Logarithmic scale.")
        self.chk_log.setStyleSheet("""
            QCheckBox { color: #aaaaaa; font-weight: bold; background: transparent; }
            QCheckBox::indicator { border: 1px solid #666666; border-radius: 2px; background: #222222; width: 12px; height: 12px; }
            QCheckBox::indicator:checked { background: #88aaff; border: 1px solid #88aaff; }
        """)
        self.chk_log.toggled.connect(self.toggle_log)
        self.chk_log.hide()

        # 3. Info
        self.btn_info = QPushButton("Info", self.view) 
        self.btn_info.setCheckable(True)
        self.btn_info.setToolTip("Toggle Detailed Diagnostics Overlay")
        self.btn_info.setStyleSheet("""
            QPushButton { 
                background-color: #333333; 
                color: #dddddd; 
                border: 1px solid #666666; 
                border-radius: 3px; 
                font-size: 8pt; 
                padding: 2px 6px; 
                font-weight: bold;
            }
            QPushButton:hover { background-color: #444444; border-color: #888888; }
            QPushButton:checked { background-color: #285299; border-color: #88aaff; }
        """)
        self.btn_info.clicked.connect(self.toggle_info_overlay)
        self.btn_info.hide()

        # 4. Histogram
        self.hist = HistogramOverlay(self.view)
        self.hist.stats_updated.connect(self.lbl_info_overlay.setText) 
        self.hist.hide()
        
        layout.addLayout(right)

        # --- SETTINGS ---
        hist_on = self.settings.value("preview_hist_visible", True, type=bool)
        log_on = self.settings.value("preview_log_scale", False, type=bool)
        info_on = self.settings.value("preview_info_visible", False, type=bool)
        
        self.b_hist.blockSignals(True)
        self.chk_log.blockSignals(True)
        self.btn_info.blockSignals(True)
        
        self.b_hist.setChecked(hist_on)
        self.chk_log.setChecked(log_on)
        self.btn_info.setChecked(info_on)
        
        self.b_hist.blockSignals(False)
        self.chk_log.blockSignals(False)
        self.btn_info.blockSignals(False)
        
        self.hist.set_log_scale(log_on)
        self.hist.set_info_mode(info_on)
        self.toggle_hist() 
        
        self.update_overlays()
        self.update_labels()

    # --- HELP ---
    def print_help_to_console(self):
        guide_lines = [
            "==========================================================================",
            "   VERALUX STARCOMPOSER v2.1 - OPERATIONAL GUIDE",
            "   High-Fidelity Star Reconstruction & Compositing Engine",
            "==========================================================================",
            "",
            "OVERVIEW",
            "-----------------",
            "StarComposer is a photometric workstation that decouples star development",
            "from the main image. It applies rational tone mapping + a Hybrid (Scalar + Vector) engine to linear",
            "star masks to create pinpoint stars with physical white cores and color halos.",
            "",
            "[1] INPUT REQUIREMENTS",
            "    • Starmask: Must be LINEAR (unstretched). This is critical for color fidelity.",
            "    • Starless: Must be NON-LINEAR (stretched/processed). This serves as the base.",
            "",
            "[2] THE MAIN WORKFLOW",
            "    • Composition Mode: Choose 'Screen' (default) or 'Linear Add' for physical",
            "      accuracy.",
            "    • Sensor Profile: Select the correct profile to weight star luminosity correctly.",
            "    • Star Intensity (Gold Slider): This is your master gain. Start here.",
            "      Increase until you see the desired amount of faint stars.",
            "      Internally, Log D controls a bounded rational tone-mapping curve (0→0, 1→1).",
            "    • Profile Hardness (b): Toe-based profile shaping (subtle near default, strong at extremes).",
            "      - Higher values = Sharp, pinpoint stars.",
            "      - Lower values = Soft, large stars.",
            "",
            "[3] HYBRID PHYSICS & COLOR",
            "    • Color Grip: Blends between Crisp and Color modes.",
            "      - 100%: Pure Vector mode. Maximum color, softer cores.",
            "      - 50% (Default): Balanced mix.",
            "      - 0%: Pure Scalar mode. Hardest stars, white cores.",
            "    • Shadow Convergence: The 'Cleaner'.",
            "      Increases the influence of the scalar engine in shadows to hide colored noise artifacts.",
            "",
            "[4] STAR SURGERY (Advanced)",
            "    Enable this section only if necessary.",
            "    • Core Rejection (LSR): Dynamically removes large blobs (e.g. galaxy cores) from",
            "      the star mask, keeping only stars.",
            "    • Reduction: Applies morphological erosion *after* tone mapping.",
            "    • Optical Healing: Blurs chrominance to fix magenta/green halos.",
            "      Note: Targets near-star chromatic fringing. Not a large-scale halo remover.",
            "",
            "[5] INTERPRETING THE HISTOGRAM",
            "    • Right Side (Whites): Orange/Red bar indicates star cores reaching or clipping white.",
            "      - If Whites are high: reduce 'Star Intensity' to avoid clipping.",            
            "    • Left Side (Blacks): Computed for VeraLux consistency, but ignored here.",
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

    # --- INPUT HANDLING ---
    def load_fits(self):
        f, _ = QFileDialog.getOpenFileName(self, "Load FITS", "", "FITS (*.fit *.fits *.fit.gz *.fits.gz)")
        if not f: return None, None
        
        self.working_dir = os.path.dirname(f)
        
        try:
            with fits.open(f) as hdul:
                idx = 0
                if hdul[0].data is None and len(hdul) > 1:
                    idx = 1
                d = hdul[idx].data                
                d = d.astype(np.float32)
                
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Could not load FITS file.\n{e}")
            return None, None
            
        if d.ndim == 2: 
            d = np.array([d, d, d])
        elif d.ndim == 3:
            if d.shape[0] == 1:
                d = np.concatenate([d, d, d], axis=0)
        
        # Robust normalization (handles int and float FITS safely)
        d = VeraLuxCore.normalize_input(d)
            
        return d, f

    def make_proxy(self, img, target_size=3200):
        if img is None: return None
        
        h, w = img.shape[1], img.shape[2]
        
        if max(h, w) <= target_size: 
            return img

        scale = target_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        img_hwc = img.transpose(1, 2, 0)
            
        resized = cv2.resize(img_hwc, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        if resized.ndim == 2:
            resized = np.expand_dims(resized, axis=2)
            
        return resized.transpose(2, 0, 1)

    def load_starmask(self):
        d, f = self.load_fits()
        if d is not None:
            self.sm_full = d; self.sm_proxy = self.make_proxy(d)
            self.sm_path = f
            self.lbl_sm.setText(f"Mask: {os.path.basename(f)}")
            self.b_sm.setStyleSheet("background-color: #388e3c; color: white; border: 1px solid #66bb6a; font-weight: bold;")
            self.request_fit = True
            self.trigger_update()

    def load_starless(self):
        d, f = self.load_fits()
        if d is not None:
            self.sl_full = d; self.sl_proxy = self.make_proxy(d)
            self.sl_path = f
            self.lbl_sl.setText(f"Base: {os.path.basename(f)}") 
            self.b_sl.setStyleSheet("background-color: #388e3c; color: white; border: 1px solid #66bb6a; font-weight: bold;")
            self.request_fit = True
            self.trigger_update()

    def set_defaults(self):
        self.s_D.setValue(0); self.s_b.setValue(500); # Updated Default
        self.s_grip.setValue(50) # Updated Default
        self.s_shad.setValue(0); self.s_lsr.setValue(0); self.s_red.setValue(0); self.s_heal.setValue(0)
        self.chk_adapt.setChecked(True)
        self.chk_screen.setChecked(True) # Updated Default
        self.update_labels()
        self.trigger_update()

    def toggle_surgery(self, checked):
        self.g5.setVisible(checked)
        QApplication.processEvents()
        if not checked:
            QTimer.singleShot(10, lambda: self.resize(self.width(), 0))

    def on_sensor_profile_changed(self, _index=None):
        try: self.settings.setValue("sensor_profile", self.cmb_prof.currentText())
        except: pass
        self.trigger_update()

    def update_labels(self):
        val_D = self.s_D.value() / 100.0 * 2 + 1.0
        val_b = self.s_b.value() / 10.0 # Same Mapping logic, just extended range
        val_grip = int(self.s_grip.value())
        val_shad = self.s_shad.value() / 100.0
        val_lsr = int(self.s_lsr.value())
        val_red = int(self.s_red.value())
        val_heal = self.s_heal.value()
        
        self.lbl_D.setText(f"Star Intensity (Log D): <b style='color:#ffb000'>{val_D:.2f}</b>")
        self.lbl_b.setText(f"Profile Hardness (b): <b>{val_b:.1f}</b>")
        self.lbl_grip.setText(f"Color Grip (Blend): <b>{val_grip}%</b>")
        self.lbl_shad.setText(f"Shadow Conv (Hide Artifacts): <b>{val_shad:.2f}</b>")
        self.lbl_lsr.setText(f"Core Rejection (LSR): <b>{val_lsr}%</b>")
        self.lbl_red.setText(f"Reduction (Erosion): <b>{val_red}%</b>")
        self.lbl_heal.setText(f"Optical Healing (Halos): <b>{val_heal:.1f}</b>")

    def trigger_update(self):
        if self.sm_proxy is None: return
        self.debounce.start()

    def run_preview_logic(self):
        D = self.s_D.value() / 100.0 * 2 + 1.0
        b = self.s_b.value() / 10.0
                
        grip = self.s_grip.value() / 100.0
        shadow = self.s_shad.value() / 100.0
        lsr = self.s_lsr.value() / 100.0
        red = self.s_red.value() / 100.0
        heal = self.s_heal.value()
        adapt = self.chk_adapt.isChecked()
        w = SENSOR_PROFILES[self.cmb_prof.currentText()]
        
        stars = process_star_pipeline(self.sm_proxy, D, b, grip, shadow, red, heal, lsr, w, adapt)
        
        # Composition Logic
        use_screen = self.chk_screen.isChecked()
        if self.sl_proxy is not None:
            if self.sl_proxy.shape != stars.shape:
                min_h = min(self.sl_proxy.shape[1], stars.shape[1])
                min_w = min(self.sl_proxy.shape[2], stars.shape[2])
                sl = self.sl_proxy[:, :min_h, :min_w]
                st = stars[:, :min_h, :min_w]
                if use_screen: comp = 1.0 - (1.0 - sl) * (1.0 - st)
                else: comp = np.clip(sl + st, 0.0, 1.0)
            else:
                if use_screen: comp = 1.0 - (1.0 - self.sl_proxy) * (1.0 - stars)
                else: comp = np.clip(self.sl_proxy + stars, 0.0, 1.0)
        else:
            comp = stars
            
        # In manual zoom (1:1 / zoomed), avoid swapping to a smaller proxy pixmap.
        # This prevents the visible "shrink" jump while full-res is computing.
        if not self.is_fit_mode and self.sm_full is not None:
            self.pending_proxy = comp
            return

        self.comp_proxy = comp
        self.update_view()

    def get_current_params(self):
        return {
            'D': self.s_D.value() / 100.0 * 2 + 1.0,
            'b': self.s_b.value() / 10.0,
            'grip': self.s_grip.value() / 100.0,
            'shadow': self.s_shad.value() / 100.0,
            'lsr': self.s_lsr.value() / 100.0,
            'red': self.s_red.value() / 100.0,
            'heal': self.s_heal.value(),
            'adapt': self.chk_adapt.isChecked(),
            'w': SENSOR_PROFILES[self.cmb_prof.currentText()]
        }

    def trigger_full_res_update(self):
        if self.sm_full is None: return
        
        # Safety: Stop the timer if it was running (e.g. manual call)
        self.release_timer.stop()
        
        # Stop existing debounce or worker if running
        self.debounce.stop()
        
        self.pbar.show()
        self.setEnabled(False)
        
        params = self.get_current_params()
        use_screen = self.chk_screen.isChecked()
        
        self.worker = PreviewWorker(self.sm_full, self.sl_full, params, use_screen)
        self.worker.result_ready.connect(self.on_full_res_ready)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.start()

    def on_full_res_ready(self, img_result):
        self.pbar.hide()
        self.setEnabled(True)
            
        # Display policy:
        # - In Fit mode: keep a proxy for responsiveness.
        # - In manual zoom (1:1 / zoomed): keep full resolution to avoid perceived "shrink"
        #   when swapping pixmaps after the full-res computation.
        h, w = img_result.shape[1], img_result.shape[2]
        max_dim = max(h, w)

        if self.is_fit_mode:
            # Fit mode: proxy is fine (we're going to fitInView anyway)
            display_img = self.make_proxy(img_result, target_size=5000)
        else:
            # Manual zoom: avoid downscaling jumps. Still protect against extreme sizes.
            # (Adjust threshold if needed.)
            if max_dim > 8000:
                display_img = self.make_proxy(img_result, target_size=8000)
            else:
                display_img = img_result

        self.comp_proxy = display_img
        self.pending_proxy = None
        self.update_view(force_render=True)

    def update_view(self, force_render=False):
        if self.comp_proxy is None: return
        
        # 1. Activate UI first time
        if not self.has_image:
            self.has_image = True
            self.toggle_hist()

        # 2. CAPTURE STATE
        old_transform = None
        old_h_ratio = None
        old_v_ratio = None
        old_center_scene = None  # fallback only

        current_pix = self.pix_item.pixmap()
        if not current_pix.isNull() and not self.is_fit_mode:
            old_transform = self.view.transform()

            # Save scroll position as ratios (prevents 1px drift from centerOn/rounding)
            hsb = self.view.horizontalScrollBar()
            vsb = self.view.verticalScrollBar()
            old_h_ratio = (hsb.value() / hsb.maximum()) if hsb.maximum() > 0 else 0.0
            old_v_ratio = (vsb.value() / vsb.maximum()) if vsb.maximum() > 0 else 0.0

            # Optional fallback if you ever need it
            old_center_scene = self.view.mapToScene(self.view.viewport().rect().center())

        # 3. UPDATE DATA
        self.hist.set_data(self.comp_proxy)
        
        disp = np.clip(self.comp_proxy * 255, 0, 255).astype(np.uint8)
        disp = np.ascontiguousarray(np.flipud(disp.transpose(1, 2, 0)))
        h, w, c = disp.shape
        qimg = QImage(disp.data.tobytes(), w, h, c*w, QImage.Format.Format_RGB888)
        
        # 4. SWAP IMAGE
        self.pix_item.setPixmap(QPixmap.fromImage(qimg))
        self.scene.setSceneRect(0, 0, w, h)
        
        # 5. RESTORE STATE
        if self.request_fit or self.is_fit_mode:
            # If we are in Fit Mode, refit on the updated image
            self.fit_view()
            self.request_fit = False

        elif old_transform is not None:
            # Restore zoom first
            self.view.setTransform(old_transform)

            # Restore pan using scrollbars (ratio-based): stable across proxy size changes
            hsb = self.view.horizontalScrollBar()
            vsb = self.view.verticalScrollBar()

            if old_h_ratio is not None and hsb.maximum() > 0:
                hsb.setValue(int(round(old_h_ratio * hsb.maximum())))
            if old_v_ratio is not None and vsb.maximum() > 0:
                vsb.setValue(int(round(old_v_ratio * vsb.maximum())))
            elif old_center_scene is not None:
                # Fallback only
                self.view.centerOn(old_center_scene)

        self.update_overlays()

    def toggle_info_overlay(self):
        """Toggles the visibility of the detailed info overlay box."""
        is_checked = self.btn_info.isChecked()
        self.lbl_info_overlay.setVisible(is_checked)
        
        # Informs the histogram of the mode
        self.hist.set_info_mode(is_checked)
        
        # Force a data refresh to immediately apply the tooltip logic
        if self.comp_proxy is not None:
             self.hist.set_data(self.comp_proxy)

        self.update_overlays()

    def process_full_resolution(self):
        if self.sm_full is None: return
        self.setEnabled(False)
        
        D = self.s_D.value() / 100.0 * 2 + 1.0
        b = self.s_b.value() / 10.0
                
        grip = self.s_grip.value() / 100.0
        shadow = self.s_shad.value() / 100.0
        lsr = self.s_lsr.value() / 100.0
        red = self.s_red.value() / 100.0
        heal = self.s_heal.value()
        adapt = self.chk_adapt.isChecked()
        w = SENSOR_PROFILES[self.cmb_prof.currentText()]
        use_screen = self.chk_screen.isChecked()

        try:
            stars = process_star_pipeline(self.sm_full, D, b, grip, shadow, red, heal, lsr, w, adapt)            
            if self.sl_full is not None:
                if self.sl_full.shape != stars.shape:
                    min_h = min(self.sl_full.shape[1], stars.shape[1])
                    min_w = min(self.sl_full.shape[2], stars.shape[2])
                    sl = self.sl_full[:, :min_h, :min_w]
                    st = stars[:, :min_h, :min_w]
                    if use_screen: final = 1.0 - (1.0 - sl) * (1.0 - st)
                    else: final = np.clip(sl + st, 0.0, 1.0)
                else:
                    if use_screen: final = 1.0 - (1.0 - self.sl_full) * (1.0 - stars)
                    else: final = np.clip(self.sl_full + stars, 0.0, 1.0)
            else:
                final = stars                
            out_name = "VeraLux_StarComposer_result.fit"
            out_path = os.path.join(self.working_dir, out_name)
            
            out = final.astype(np.float32)
            
            # Decide which image to use as Metadata Template
            # Priority: Starless (Base) > Starmask
            if self.sl_full is not None and hasattr(self, 'sl_path'):
                template_path = self.sl_path
            elif hasattr(self, 'sm_path'):
                template_path = self.sm_path
            else:
                raise ValueError("Source path for metadata not found.")

            # 1. Load the template image to populate internal structures (WCS, Keywords)
            safe_template_path = template_path.replace(os.sep, '/')
            self.siril.cmd(f'load "{safe_template_path}"')

            # 2. Inject the NEW pixel data safely into the current image buffer
            with self.siril.image_lock():
                self.siril.set_image_pixeldata(out)
            
            # 3. CLEANUP: Now that the NEW image data is in place, remove its ICC profile.
            try: 
                self.siril.cmd("icc_remove")
            except Exception:
                pass

            self.siril.cmd("stat")

            try: 
                self.siril.cmd("visu 0 65535") 
            except: 
                pass
            
            # 4. Save via Siril (writes original headers + new pixels)
            safe_out_path = out_path.replace(os.sep, '/')
            self.siril.cmd(f'save "{safe_out_path}"')            
            self.siril.log(f"VeraLux: Saved to {out_name} (Header Preserved)")
            self.close()

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
        finally:
            self.setEnabled(True)

    # --- VIEWPORT EVENTS ---
    def toggle_ontop(self, checked):
        if checked:
            self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        else:
            self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowStaysOnTopHint)
        self.show()

    def toggle_hist(self):
        visible = self.b_hist.isChecked() and self.has_image
        
        self.hist.setVisible(visible)
        self.hist.is_visible = visible
        
        self.chk_log.setVisible(visible)
        self.btn_info.setVisible(visible)
        
        if not visible:
            self.lbl_info_overlay.setVisible(False)
        else:
            self.lbl_info_overlay.setVisible(self.btn_info.isChecked())
            
        self.update_overlays()

    def toggle_log(self, v):
        self.hist.set_log_scale(v)

    def closeEvent(self, event):
        # Save graphic preferences
        self.settings.setValue("preview_hist_visible", self.b_hist.isChecked())
        self.settings.setValue("preview_log_scale", self.chk_log.isChecked())
        self.settings.setValue("preview_info_visible", self.btn_info.isChecked())
        
        # Save last used sensor profile
        try: self.settings.setValue("sensor_profile", self.cmb_prof.currentText())
        except: pass
        
        event.accept()

    def resizeEvent(self, event):
        self.update_overlays()
        super().resizeEvent(event)

    def update_overlays(self):
        w, h = self.view.width(), self.view.height()
        
        hist_x = 10
        hist_y = h - 130  # Histogram height (120) + margin
        self.hist.move(hist_x, hist_y)
        
        spacing = 8
        
        self.chk_log.adjustSize()
        self.btn_info.adjustSize()
        
        row_height = max(self.chk_log.height(), self.btn_info.height())
        btns_y = hist_y - row_height - spacing 
        
        self.chk_log.move(hist_x, btns_y)
        
        info_btn_x = hist_x + self.chk_log.width() + 10 
        self.btn_info.move(info_btn_x, btns_y)
        
        if self.lbl_info_overlay.isVisible():
            self.lbl_info_overlay.adjustSize()
            self.lbl_info_overlay.setFixedWidth(200)
            
            box_h = self.lbl_info_overlay.height()
            box_y = btns_y - box_h - spacing
            
            self.lbl_info_overlay.move(hist_x, box_y)
            self.lbl_info_overlay.raise_()

    def fit_view(self):
        self.is_fit_mode = True
        if self.pix_item.pixmap(): 
            self.view.fitInView(self.pix_item, Qt.AspectRatioMode.KeepAspectRatio)

    def zoom_in(self): 
        self.is_fit_mode = False
        self.view.scale(1.2, 1.2)
        
    def zoom_out(self): 
        self.is_fit_mode = False
        self.view.scale(1/1.2, 1/1.2)
        
    def zoom_1to1(self): 
        self.is_fit_mode = False
        self.view.resetTransform()
        
    def eventFilter(self, source, event):
        if source == self.view.viewport() and event.type() == QEvent.Type.Wheel:
            self.is_fit_mode = False
            if event.angleDelta().y() > 0: self.zoom_in()
            else: self.zoom_out()
            return True
            
        if source == self.view and event.type() == QEvent.Type.Resize:
            self.update_overlays()
        
        if source == self.view.viewport() and event.type() == QEvent.Type.MouseButtonDblClick:
            self.fit_view()
            return True
            
        return super().eventFilter(source, event)

def main():
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
        
    siril = s.SirilInterface()
    try: 
        siril.connect()
    except Exception as e:
        print(f"Siril connection error: {e}")
        
    gui = StarComposerGUI(siril, app)
    gui.show()
    app.exec()

if __name__ == "__main__":
    main()