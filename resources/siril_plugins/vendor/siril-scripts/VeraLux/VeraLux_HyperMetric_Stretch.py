##############################################
# VeraLux — HyperMetric Stretch
# Photometric Hyperbolic Stretch Engine
# Author: Riccardo Paterniti (2025)
# Contact: info@veralux.space
##############################################

# (c) 2025 Riccardo Paterniti
# VeraLux — HyperMetric Stretch
# SPDX-License-Identifier: GPL-3.0-or-later
# Version 1.5.2
#
# Credits / Origin
# ----------------
#   • Inspired by: The "True Color" methodology of Dr. Roger N. Clark
#   • Math basis: Inverse Hyperbolic Stretch (IHS) & Vector Color Preservation
#   • Sensor Science: Hardware-specific Quantum Efficiency weighting
#

"""
Overview
--------
A precision linear-to-nonlinear stretching engine designed to maximize sensor 
fidelity while managing the transition to the visible domain.

HyperMetric Stretch (HMS) operates on a fundamental axiom: standard histogram 
transformations often destroy the photometric relationships between color channels 
(hue shifts) and clip high-dynamic range data. HMS solves this by decoupling 
Luminance geometry from Chromatic vectors.

Design Goals
------------
• Preserve original vector color ratios during extreme stretching (True Color)
• Optimize Luminance extraction based on specific hardware (Sensor Profiles)
• Provide a mathematically "Safe" expansion for high-dynamic targets
• Bridge the gap between numerical processing and visual feedback (Live Preview)
• Allow controlled hybrid tone-mapping via Color Grip & Shadow Convergence

Core Features
-------------
• Live Preview & Analytics:
  - Interactive floating window offering real-time feedback on parameter changes.
  - Real-time RGB Histogram overlay with smart clipping warnings (>0.1%).
  - Features Smart Proxy technology for fluid response even on massive files.
  - Includes professional navigation controls: Zoom, Pan, Fit-to-Screen, and **1:1 View**.
  - **Diagnostic Info Overlay:** Toggleable detailed readout of clipping statistics 
    and Linear Expansion bounds directly on the preview.

• Intelligent Calibration System:
  - Smart Iterative Solver: The Auto-Calculator performs a predictive "Floating 
    Sky Check". It simulates the post-stretch scaling to detect black clipping 
    and iteratively adjusts the target to maximize contrast without data loss.
  - Adaptive Anchor: Uses morphological analysis for precise black point detection 
    (Default ON), superior to standard percentile clipping on gradient-free data.

• Hybrid Color Engine:
  - **Scientific Mode:** Full manual control over **Linear Expansion**, Color Grip, 
    and Shadow Convergence. Includes "Smart Max" safety to reject hot pixels while 
    preserving star cores during normalization.
  - **Ready-to-Use Mode:** Orchestrated pipeline with "Smart Max" zero-clipping 
    logic and "Unified Color Strategy" slider for single-point balancing of 
    background noise vs highlight saturation.

• Unified Math Core:
  - Implements a "Single Source of Truth" architecture. The Auto-Solver, Live 
    Preview, and Main Processor share the exact same logic.

Usage
-----
1. Pre-requisite: Image MUST be Linear and Color Calibrated (SPCC).
2. Setup: Select Sensor Profile and Processing Mode.
3. Calibrate: 
   - Adaptive Anchor is ON by default (recommended for max dynamic range).
   - Click Calculate Optimal Log D. The solver will find the safe limit.
4. Refine (Live Preview): 
   - **Scientific:** Adjust Linear Expansion to fill dynamic range.
   - **Ready-to-Use:** Adjust Color Strategy to clean noise (Left) or save highlights (Right).
   - Use the Histogram/Info Overlay to verify clipping (Red bars/text).
5. Process: Click PROCESS.

Inputs & Outputs
----------------
Input: Linear FITS/TIFF (RGB/Mono). 16/32-bit Int or Float.
Output: Non-linear (Stretched) 32-bit Float FITS.

Compatibility
-------------
• Siril 1.3+
• Python 3.10+ (via sirilpy)
• Dependencies: sirilpy, PyQt6, numpy

License
-------
Released under GPL-3.0-or-later.
"""

import sys
import os
import traceback
import webbrowser

try:
    import sirilpy as s
    from sirilpy import LogColor
except ImportError:
    # Fallback for CLI testing or missing modules
    print("Error: sirilpy module not found. This script must be run within Siril.")
    sys.exit(1)

s.ensure_installed("PyQt6", "numpy")

import numpy as np
import math

from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                            QWidget, QLabel, QDoubleSpinBox, QSlider,
                            QPushButton, QGroupBox, QMessageBox, QProgressBar,
                            QComboBox, QRadioButton, QButtonGroup, QCheckBox, QFrame,
                            QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, 
                            QStackedWidget)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QEvent, QSettings, QPointF
from PyQt6.QtGui import (QImage, QPixmap, QPainter, QColor, QWheelEvent, QMouseEvent, 
                        QPen, QBrush, QPainterPath, QTextDocument)

# ---------------------
#  THEME & STYLING
# ---------------------

DARK_STYLESHEET = """
QWidget { background-color: #2b2b2b; color: #e0e0e0; font-size: 10pt; }
QToolTip { background-color: #333333; color: #ffffff; border: 1px solid #88aaff; }
QGroupBox { border: 1px solid #444444; margin-top: 5px; font-weight: bold; border-radius: 4px; padding-top: 12px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; color: #88aaff; }
QLabel { color: #cccccc; }

/* Windows Fix: Explicitly style indicators to ensure visibility on custom dark backgrounds */
QRadioButton, QCheckBox { color: #cccccc; spacing: 5px; }
QRadioButton::indicator, QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #666666; background: #3c3c3c; border-radius: 7px; }
QCheckBox::indicator { border-radius: 3px; }
QRadioButton::indicator:checked { background-color: #285299; border: 1px solid #88aaff; image: none; }
QCheckBox::indicator:checked { background-color: #285299; border: 1px solid #88aaff; image: none; }
QRadioButton::indicator:checked { background: qradialgradient(cx:0.5, cy:0.5, radius: 0.4, fx:0.5, fy:0.5, stop:0 #ffffff, stop:1 #285299); }
QCheckBox::indicator:checked { background: #285299; }

QDoubleSpinBox { background-color: #3c3c3c; color: #ffffff; border: 1px solid #555555; padding: 3px; border-radius: 3px; }
QDoubleSpinBox:hover { border-color: #777777; }
QComboBox { background-color: #3c3c3c; color: #ffffff; border: 1px solid #555555; padding: 3px; border-radius: 3px; }
QComboBox:hover { border-color: #777777; }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox::down-arrow { width: 0; height: 0; border-left: 4px solid transparent; border-right: 4px solid transparent; border-top: 6px solid #aaaaaa; margin-right: 6px; }
QComboBox QAbstractItemView { background-color: #3c3c3c; color: #ffffff; selection-background-color: #285299; border: 1px solid #555555; }

/* Robust styling to prevent flickering and transparency */
QSlider { min-height: 22px; }
QSlider::groove:horizontal { 
    background: #444444; 
    height: 6px; 
    border-radius: 3px; 
}
QSlider::handle:horizontal { 
    background-color: #aaaaaa; 
    width: 14px; 
    height: 14px; 
    margin: -4px 0; 
    border-radius: 7px; 
    border: 1px solid #555555; 
}
QSlider::handle:horizontal:hover { 
    background-color: #ffffff; 
    border: 1px solid #888888; 
}
QSlider::handle:horizontal:pressed { 
    background-color: #ffffff; 
    border: 1px solid #dddddd; 
}

QPushButton { background-color: #444444; color: #dddddd; border: 1px solid #666666; border-radius: 4px; padding: 6px; font-weight: bold;}
QPushButton:hover { background-color: #555555; border-color: #777777; }
QPushButton#ProcessButton { background-color: #285299; border: 1px solid #1e3f7a; }
QPushButton#ProcessButton:hover { background-color: #355ea1; }
QPushButton#AutoButton { background-color: #8c6a00; border: 1px solid #a37c00; }
QPushButton#AutoButton:hover { background-color: #bfa100; color: #000000;}
QPushButton#PreviewButton { background-color: #2a5a2a; border: 1px solid #408040; }
QPushButton#PreviewButton:hover { background-color: #3a7a3a; }
QPushButton#CloseButton { background-color: #5a2a2a; border: 1px solid #804040; }
QPushButton#CloseButton:hover { background-color: #7a3a3a; }

/* Preview Toolbar Buttons */
QPushButton#ZoomBtn { min-width: 30px; font-weight: bold; background-color: #3c3c3c; }

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

QPushButton#CoffeeButton{background-color:transparent;border:none;font-size:15pt;padding:2px}
QPushButton#CoffeeButton:hover{background-color:rgba(255,255,255,20);border-radius:4px}

QProgressBar { border: 1px solid #555555; border-radius: 3px; text-align: center; }
QProgressBar::chunk { background-color: #285299; width: 10px; }

QGraphicsView{border:none;background-color:#151515}
"""

def _nofocus(w):
    try:
        w.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    except Exception:
        pass

VERSION = "1.5.2"
# ------------------------------------------------------------------------------
# VERSION HISTORY
# ------------------------------------------------------------------------------
# 1.5.2: Stability & Consistency Patch.
#        • Fix: Removed duplicate signal connections (slide_d ↔ spin_d) that
#          caused double-fire callbacks and unnecessary preview redraws.
#        • Fix: Mono (1,H,W) images were misdetected as RGB, causing silent
#          corruption in the color vector pipeline. Added explicit channel check.
#        • Fix: Preview tracker now uses the active Sensor Profile weights
#          instead of hardcoded Rec.709, ensuring consistent luminance readout.
#        • Fix: Updated help guide version string from 1.4 to current.
#        • Cleanup: Removed dead code (le_critical), redundant hasattr checks,
#          and duplicate widget resets in set_defaults().
#        • Safety: Added re-entrancy guard on Solver and Processor threads.
#        • Math: Float comparison in hyperbolic_stretch norm_factor uses epsilon.
# 1.5.1: "Buy me a coffee" button added.
# 1.5.0: Scientific Precision & UX Overhaul.
#        • Scientific Linear Expansion: Implemented asymmetric normalization engine.
#          Uses "Smart Max" logic to maximize contrast while preserving star cores
#          (Zero-Clipping) and robustly handling hot pixels.
#        • Live Preview 2.0: Added "1:1" pixel view, persistent "Info" overlay
#          dashboard, and saved view preferences (Log/Info/Hist states).
#        • Enhanced Diagnostics: Decoupled physical clipping from expansion
#          clamping warnings. Added real-time stats to the main window.
#        • Core Stability: Added NaN/Inf sanitization to input normalization to
#          prevent black previews on corrupted data.
# 1.4.1: High-Dynamic Range Safety Update.
#        • Zero-Clipping Logic: Refactored Ready-to-Use scaling to anchor on the
#          absolute physical maximum instead of percentiles. Ensures zero data
#          loss in high-dynamic targets (bright cores/stars) before soft-clipping.
#        • Better ICC profile fix behavior (robust).
# 1.4.0: GUI Rationalization & Low-Signal Optimization.
#        • ICC-Safe Retrieval: Implemented robust fallback path. Attempts fast
#          shared memory retrieval first, falling back to full image data fetch
#          if ICC profiles block direct access.
#        • Anchor Logic Fix: Refined the Adaptive Anchor algorithm to correctly 
#          handle extremely low-signal data (near-zero histogram peaks).
#        • GUI Overhaul: Interface rationalization and alignment.
# 1.3.1: Sensor profiles update (v2.2)
# 1.3.0: Major Science & Engineering Upgrade.
#        • Smart Iterative Solver: Auto-calculator now employs a predictive 
#          feedback loop to optimize dynamic range allocation and preserve 
#          deep shadow structure ("Floating Sky" optimization) especially useful
#          in Ready-to-use mode.
#        • Visual Feedback: Added Live RGB Histogram with smart clipping analysis.
#        • Unified Color Strategy: Single intuitive slider for Ready-to-Use mode.
#        • Shadow Convergence: Photometric noise damping for the Scientific engine.
# 1.2.2: UX Upgrade. Added persistent settings (QSettings). VeraLux now remembers
#        Sensor Profile, Processing Mode, and Target Background between sessions.
# 1.2.1: Nomenclature refinement. Replaced generic GHS terms with accurate
#        "Inverse Hyperbolic Stretch" definitions. Minor UI text polish.
# 1.2.0: Major Upgrade. Added Live Preview Engine with Smart Proxy technology.
#        Introduced "Color Grip" Hybrid Stretch for star control.
# 1.1.0: Architecture Upgrade. Introduced VeraLuxCore (Single Source of Truth).
#        Fixed 32-bit/Mono input handling & visual refresh issues (visu reset).
#        Added robust input normalization & improved Solver precision.
# 1.0.3: Added help button (?) that prints Operational Guide to Siril Console.
#        Added contact e-mail. Texts consistency minor fixes.
# 1.0.2: Sensor Database Update (v2.0). Added real QE weights for 15+ sensors.
# 1.0.1: Fix Windows GUI artifacts (invisible checkboxes) and UI polish.
# ------------------------------------------------------------------------------

# =============================================================================
#  WORKING SPACE PROFILES (Database v2.2 - Siril SPCC Derived)
# =============================================================================

SENSOR_PROFILES = {
    # --- STANDARD ---
    "Rec.709 (Recommended)": {
        'weights': (0.2126, 0.7152, 0.0722),
        'description': "ITU-R BT.709 standard for sRGB/HDTV",
        'info': "Default choice. Best for general use, DSLR and unknown sensors.",
        'category': 'standard'
    },
    
    # --- SONY MODERN BSI (APS-C / Full Frame) ---
    "Sony IMX571 (ASI2600/QHY268)": {
        'weights': (0.2944, 0.5021, 0.2035),
        'description': "Sony IMX571 26MP APS-C BSI (STARVIS)",
        'info': "Gold standard APS-C. Balanced spectral response.",
        'category': 'sensor-specific'
    },
    
    "Sony IMX455 (ASI6200/QHY600)": {
        'weights': (0.2987, 0.5001, 0.2013),
        'description': "Sony IMX455 61MP Full Frame BSI",
        'info': "Full frame reference sensor. Excellent broadband balance.",
        'category': 'sensor-specific'
    },
    
    "Sony IMX410 (ASI2400)": {
        'weights': (0.3015, 0.5050, 0.1935),
        'description': "Sony IMX410 24MP Full Frame (Large Pixels)",
        'info': "High sensitivity, large pixel full frame sensor.",
        'category': 'sensor-specific'
    },

    "Sony IMX269 (Altair/ToupTek)": {
        'weights': (0.3040, 0.5010, 0.1950),
        'description': "Sony IMX269 20MP 4/3\" BSI",
        'info': "Standard 4/3 sensor found in many mid-range cameras.",
        'category': 'sensor-specific'
    },

    "Sony IMX294 (ASI294)": {
        'weights': (0.3068, 0.5008, 0.1925),
        'description': "Sony IMX294 11.7MP 4/3\" BSI",
        'info': "High sensitivity 4/3 format. High Red/Green response.",
        'category': 'sensor-specific'
    },

    # --- SONY MEDIUM FORMAT / SQUARE ---
    "Sony IMX533 (ASI533)": {
        'weights': (0.2910, 0.5072, 0.2018),
        'description': "Sony IMX533 9MP 1\" Square BSI",
        'info': "Popular square format. Very low noise profile.",
        'category': 'sensor-specific'
    },
    
    "Sony IMX676 (ASI676)": {
        'weights': (0.2880, 0.5100, 0.2020),
        'description': "Sony IMX676 12MP Square BSI (Starvis 2)",
        'info': "New generation square sensor. Wide dynamic range.",
        'category': 'sensor-specific'
    },

    # --- SONY PLANETARY / GUIDING (High Sensitivity) ---
    "Sony IMX585 (ASI585) - STARVIS 2": {
        'weights': (0.3431, 0.4822, 0.1747),
        'description': "Sony IMX585 8.3MP 1/1.2\" BSI (STARVIS 2)",
        'info': "High Red/NIR sensitivity. Excellent for uncooled DSO.",
        'category': 'sensor-specific'
    },
    
    "Sony IMX662 (ASI662) - STARVIS 2": {
        'weights': (0.3430, 0.4821, 0.1749),
        'description': "Sony IMX662 2.1MP 1/2.8\" BSI (STARVIS 2)",
        'info': "Planetary/Guiding. Enhanced NIR response.",
        'category': 'sensor-specific'
    },
    
    "Sony IMX678 (ASI678) - STARVIS 2": {
        'weights': (0.3426, 0.4825, 0.1750),
        'description': "Sony IMX678 8MP BSI (STARVIS 2)",
        'info': "4K resolution, small pixels, high sensitivity.",
        'category': 'sensor-specific'
    },

    "Sony IMX462 (ASI462)": {
        'weights': (0.3333, 0.4866, 0.1801),
        'description': "Sony IMX462 2MP 1/2.8\" (High NIR)",
        'info': "Extreme sensitivity in Red/NIR. High Red weight.",
        'category': 'sensor-specific'
    },

    "Sony IMX715 (ASI715)": {
        'weights': (0.3410, 0.4840, 0.1750),
        'description': "Sony IMX715 8MP (Starvis 2)",
        'info': "Ultra-small pixels planetary sensor.",
        'category': 'sensor-specific'
    },
    
    "Sony IMX482 (ASI482)": {
        'weights': (0.3150, 0.4950, 0.1900),
        'description': "Sony IMX482 2MP (Large Pixels)",
        'info': "Large pixels for high sensitivity.",
        'category': 'sensor-specific'
    },
    
    "Sony IMX183 (ASI183)": {
        'weights': (0.2967, 0.4983, 0.2050),
        'description': "Sony IMX183 20MP 1\" BSI",
        'info': "High resolution 1-inch sensor.",
        'category': 'sensor-specific'
    },
    
    "Sony IMX178 (ASI178)": {
        'weights': (0.2346, 0.5206, 0.2448),
        'description': "Sony IMX178 6.4MP 1/1.8\" BSI",
        'info': "Entry-level high res. Lower Red response than Starvis.",
        'category': 'sensor-specific'
    },
    
    "Sony IMX224 (ASI224)": {
        'weights': (0.3402, 0.4765, 0.1833),
        'description': "Sony IMX224 1.27MP 1/3\" BSI",
        'info': "Classic planetary sensor. Very high Red response.",
        'category': 'sensor-specific'
    },
    
    # --- CANON DSLR ---
    "Canon EOS (Modern - 60D/600D/500D)": {
        'weights': (0.2600, 0.5200, 0.2200),
        'description': "Canon CMOS (Digic 4/5 Era)",
        'info': "Profile for 60D, 600D, 550D, 500D. Standard IR-Cut.",
        'category': 'sensor-specific'
    },
    
    "Canon EOS (Legacy - 300D/40D/20D)": {
        'weights': (0.2450, 0.5350, 0.2200),
        'description': "Canon CMOS (Legacy Digic 2/3)",
        'info': "Profile for 300D, 40D, 20D, 350D.",
        'category': 'sensor-specific'
    },

    # --- NIKON DSLR ---
    "Nikon DSLR (Modern - D5100/D7200)": {
        'weights': (0.2650, 0.5100, 0.2250),
        'description': "Nikon DX/FX CMOS (Modern)",
        'info': "Profile for D5100, D7000 series, D500, D850.",
        'category': 'sensor-specific'
    },

    "Nikon DSLR (Legacy - D3/D300/D90)": {
        'weights': (0.2500, 0.5300, 0.2200),
        'description': "Nikon CMOS (Legacy)",
        'info': "Profile for D3, D300s, D90, D40, D50.",
        'category': 'sensor-specific'
    },

    # --- FUJI / OTHERS ---
    "Fujifilm X-Trans 5 HR": {
        'weights': (0.2800, 0.5100, 0.2100),
        'description': "Fujifilm X-Trans 5 (40MP)",
        'info': "Approximation for X-T5/X-H2 sensors.",
        'category': 'sensor-specific'
    },

    "Panasonic MN34230 (ASI1600)": {
        'weights': (0.2650, 0.5250, 0.2100),
        'description': "Panasonic MN34230 4/3\" CMOS",
        'info': "Classic Mono/OSC sensor (ASI1600).",
        'category': 'sensor-specific'
    },
    
    # --- SMART TELESCOPES ---
    "ZWO Seestar S50": {
        'weights': (0.3333, 0.4866, 0.1801),
        'description': "ZWO Seestar S50 (IMX462)",
        'info': "Optimized for Seestar S50 (High Red/NIR).",
        'category': 'sensor-specific'
    },
    
    "ZWO Seestar S30": {
        'weights': (0.2928, 0.5053, 0.2019),
        'description': "ZWO Seestar S30",
        'info': "Optimized for Seestar S30.",
        'category': 'sensor-specific'
    },
    
    # --- NARROWBAND ---
    "Narrowband HOO": {
        'weights': (0.5000, 0.2500, 0.2500),
        'description': "Bicolor palette: Hα=Red, OIII=Green+Blue",
        'info': "Balanced weighting for HOO synthetic palette processing.",
        'category': 'narrowband'
    },
    
    "Narrowband SHO": {
        'weights': (0.3333, 0.3400, 0.3267),
        'description': "Hubble palette: SII=Red, Hα=Green, OIII=Blue",
        'info': "Nearly uniform weighting for SHO tricolor narrowband.",
        'category': 'narrowband'
    }
}

DEFAULT_PROFILE = "Rec.709 (Recommended)"

# =============================================================================
#  CUSTOM WIDGETS
# =============================================================================

class ResetSlider(QSlider):
    """
    A specialized QSlider that resets to its default value on double-click.
    Used for the 'Color Strategy' unified control.
    """
    def __init__(self, orientation, default_value=0, parent=None):
        super().__init__(orientation, parent)
        self.default_val = default_value

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setValue(self.default_val)
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

class HistogramOverlay(QWidget):
    """
    A transparent overlay that draws an RGB histogram.
    Emits formatted HTML stats for external display.
    """
    stats_updated = pyqtSignal(str) # SEGNALE PER INVIARE I DATI AL BOX ESTERNO

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.hist_data = None
        self.is_visible = True
        self.use_log = False  
        self.tracker_val = None 
        
        # State variables
        self.clip_black = False; self.clip_white = False
        self.pct_black = 0.0; self.pct_white = 0.0
        self.black_level = 0; self.white_level = 0
        self.black_count = 0; self.white_count = 0
        
        # Context Data 
        self.source_saturated = False
        self.processing_mode = "ready_to_use"
        self.linear_expansion_diag = None
        
        self.setFixedHeight(120)
        self.setFixedWidth(200)

    def set_context(self, source_saturated, mode, linear_expansion_diag=None):
        self.source_saturated = source_saturated
        self.processing_mode = mode
        self.linear_expansion_diag = linear_expansion_diag

    def set_data(self, img_data):
        if img_data is None: return

        bins = 256
        if img_data.ndim == 3:
            r = np.histogram(img_data[0], bins=bins, range=(0, 1))[0]
            g = np.histogram(img_data[1], bins=bins, range=(0, 1))[0]
            b = np.histogram(img_data[2], bins=bins, range=(0, 1))[0]
            max_val = max(np.max(r), np.max(g), np.max(b))
            self.hist_data = (r/max_val, g/max_val, b/max_val) if max_val > 0 else None
            
            epsilon = 1e-7
            self.black_count = np.count_nonzero(np.any(img_data <= epsilon, axis=0))
            self.white_count = np.count_nonzero(np.any(img_data >= (1.0 - epsilon), axis=0))
            total_pixels = img_data.shape[1] * img_data.shape[2]
        else:
            l = np.histogram(img_data, bins=bins, range=(0, 1))[0]
            max_val = np.max(l)
            self.hist_data = (l / max_val,) if max_val > 0 else None
            
            epsilon = 1e-7
            self.black_count = np.count_nonzero(img_data <= epsilon)
            self.white_count = np.count_nonzero(img_data >= (1.0 - epsilon))
            total_pixels = img_data.shape[0] * img_data.shape[1]

        self.pct_black = (self.black_count / total_pixels) * 100.0
        self.pct_white = (self.white_count / total_pixels) * 100.0

        def get_lvl(p): return 2 if p >= 0.1 else (1 if p >= 0.01 else 0)
        self.black_level = get_lvl(self.pct_black)
        self.white_level = get_lvl(self.pct_white)

        self.clip_black = self.black_level > 0
        self.clip_white = self.white_level > 0 # Base physical check

        report = self._generate_html_report()
        self.setToolTip(report)
        self.stats_updated.emit(report) 
        self.update()

    def _generate_html_report(self):
        colors = {0: "#cccccc", 1: "Orange", 2: "#ff4444"}
        
        c_blk = colors[self.black_level]
        c_wht = colors[self.white_level] # Ora questo è il colore solo fisico
        
        html = "<div style='font-size:9pt; font-weight:bold; color:#eeeeee;'>Histogram Analysis</div>"
        html += f"<div style='margin-top:4px;'>Blacks: <span style='color:{c_blk};'>{self.pct_black:.2f}%</span> <span style='font-size:8pt; color:#999999;'>({self.black_count} px)</span></div>"
        html += f"<div>Whites: <span style='color:{c_wht};'>{self.pct_white:.2f}%</span> <span style='font-size:8pt; color:#999999;'>({self.white_count} px)</span></div>"

        if self.processing_mode != "ready_to_use" and self.linear_expansion_diag:
            pct_lo = float(self.linear_expansion_diag.get("pct_low", 0.0) or 0.0)
            pct_hi = float(self.linear_expansion_diag.get("pct_high", 0.0) or 0.0)
            low_v = self.linear_expansion_diag.get("low", None)
            high_v = self.linear_expansion_diag.get("high", None)

            if (pct_lo + pct_hi) > 0.0:
                html += "<div style='margin-top:6px; font-weight:bold; color:#88aaff;'>Linear Expansion</div>"
                if low_v is not None and high_v is not None:
                    html += f"<div style='font-size:8pt; color:#999999;'>B: {low_v:.4f} / {high_v:.4f}</div>"
                
                def get_col(p): return "#ff4444" if p>=0.1 else ("Orange" if p>=0.01 else "#cccccc")
                
                html += f"<div>Low: <span style='color:{get_col(pct_lo)};'>{pct_lo:.3f}%</span> • High: <span style='color:{get_col(pct_hi)};'>{pct_hi:.3f}%</span></div>"
        return html

    def set_tracker(self, val):
        self.tracker_val = val
        self.update()

    def toggle_visibility(self):
        self.is_visible = not self.is_visible
        self.setVisible(self.is_visible)

    def set_log_scale(self, enabled):
        self.use_log = enabled
        self.update()

    def paintEvent(self, event):
        if not self.hist_data or not self.is_visible: return
        painter = QPainter(self); painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        
        # Background
        painter.setBrush(QColor(0, 0, 0, 180)); painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(0, 0, w, h, 5, 5)
        
        # Curves
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        step_w = w / 256.0
        colors = [QColor(255, 50, 50, 180), QColor(50, 255, 50, 180), QColor(50, 100, 255, 180)]
        if len(self.hist_data) == 1: colors = [QColor(200, 200, 200, 180)]
        log_scale = 1000.0; log_norm = math.log10(1 + log_scale)
        for i, channel in enumerate(self.hist_data):
            path = QPainterPath(); path.moveTo(0, h)
            for x_idx, val in enumerate(channel):
                draw_val = val
                if self.use_log and val > 0: draw_val = math.log10(1 + val * log_scale) / log_norm
                path.lineTo(x_idx * step_w, h - (draw_val * (h - 20)))
            path.lineTo(w, h); path.closeSubpath()
            painter.setBrush(QBrush(colors[i])); painter.setPen(Qt.PenStyle.NoPen); painter.drawPath(path)

        # Tracker
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        if self.tracker_val is not None:
            tx = self.tracker_val * w
            painter.setPen(QPen(QColor(140, 106, 0), 1, Qt.PenStyle.DashLine))
            painter.drawLine(int(tx), 0, int(tx), h)

        # --- HUD & INDICATORS ---
        font = painter.font(); font.setPointSize(8); font.setBold(True); painter.setFont(font)
        cols = {1: QColor(255, 165, 0, 200), 2: QColor(255, 50, 50, 200)}
        
        # Calculation of combined levels for BARS (Bar shows the worst case)
        bar_black_lvl = self.black_level
        bar_white_lvl = self.white_level
        
        if self.processing_mode != "ready_to_use" and self.linear_expansion_diag:
            def gl(p): return 2 if p >= 0.1 else (1 if p >= 0.01 else 0)
            pct_lo = float(self.linear_expansion_diag.get("pct_low", 0.0) or 0.0)
            pct_hi = float(self.linear_expansion_diag.get("pct_high", 0.0) or 0.0)
            # Uniamo i livelli: se LE è critica, la barra si accende
            bar_black_lvl = max(bar_black_lvl, gl(pct_lo))
            bar_white_lvl = max(bar_white_lvl, gl(pct_hi))

        # Draw Side Bars
        if bar_black_lvl > 0:
            painter.setPen(Qt.PenStyle.NoPen); painter.setBrush(cols[bar_black_lvl])
            painter.drawRect(0, 0, 3, h)
        if bar_white_lvl > 0:
            painter.setPen(Qt.PenStyle.NoPen); painter.setBrush(cols[bar_white_lvl])
            painter.drawRect(w - 3, 0, 3, h)
            
        # Draw Text (Use the original PHYSICAL layers, decoupled from LE)
        def gc(l): return QColor(255, 50, 50) if l==2 else (QColor(255, 165, 0) if l==1 else QColor(200, 200, 200))
        
        painter.setPen(gc(self.black_level))
        painter.drawText(5, 12, f"BLK: {self.pct_black:.2f}%")
        
        w_txt = f"WHT: {self.pct_white:.2f}%"
        fm = painter.fontMetrics(); tw = fm.horizontalAdvance(w_txt)
        painter.setPen(gc(self.white_level))
        painter.drawText(w - tw - 5, 12, w_txt)

# =============================================================================
#  CORE ENGINE (Single Source of Truth)
# =============================================================================

class VeraLuxCore:
    # Diagnostic: last Linear Expansion clamp stats (sample-based)
    _last_linear_expansion_diag = {"pct_low": 0.0, "pct_high": 0.0, "low": 0.0, "high": 0.0}

    @staticmethod
    def normalize_input(img_data):
        img_data = np.nan_to_num(img_data, nan=0.0, posinf=None, neginf=0.0)
        
        input_dtype = img_data.dtype
        img_float = img_data.astype(np.float32)
        
        if np.issubdtype(input_dtype, np.integer):
            if input_dtype == np.uint8: return img_float / 255.0
            elif input_dtype == np.uint16: return img_float / 65535.0
            elif input_dtype == np.int16: return img_float / 32767.0
            else: return img_float / 4294967295.0
            
        elif np.issubdtype(input_dtype, np.floating):
            current_max = float(np.max(img_data))
            
            if current_max <= 1.1: # Tolerance for overshoot
                return img_float
                
            if current_max < 100000.0: 
                return img_float / 65535.0
                
            return img_float / 4294967295.0
            
        return img_float

    @staticmethod
    def calculate_anchor(data_norm):
        # Accept both (3,H,W) and (3,N) as RGB; (1,H,W) and (1,N) as mono
        if data_norm.ndim == 3 and data_norm.shape[0] == 3:
            floors = []
            stride = max(1, data_norm.size // 500000)
            for c in range(3):
                floors.append(np.percentile(data_norm[c].flatten()[::stride], 0.5))
            anchor = max(0.0, min(floors) - 0.00025)

        elif data_norm.ndim == 2 and data_norm.shape[0] == 3:
            floors = []
            stride = max(1, data_norm.size // 500000)
            for c in range(3):
                floors.append(np.percentile(data_norm[c].flatten()[::stride], 0.5))
            anchor = max(0.0, min(floors) - 0.00025)

        elif data_norm.ndim == 3 and data_norm.shape[0] == 1:
            stride = max(1, data_norm.size // 200000)
            floor = np.percentile(data_norm[0].flatten()[::stride], 0.5)
            anchor = max(0.0, floor - 0.00025)

        else:
            stride = max(1, data_norm.size // 200000)
            floor = np.percentile(data_norm.flatten()[::stride], 0.5)
            anchor = max(0.0, floor - 0.00025)

        return anchor

    @staticmethod
    def calculate_anchor_adaptive(data_norm, weights=None):
        """
        Adaptive (morphological) black point estimation.
        For RGB data, compute the histogram on a luminance proxy (photometrically coherent)
        instead of flattening all channels together.

        Accepts:
          - RGB: (3,H,W) or (3,N)
          - Mono: (H,W), (N,), or (1,H,W)

        weights: (r_w, g_w, b_w). If None and RGB, defaults to Rec.709 weights.
        """
        # Default weights if not provided
        if weights is None:
            weights = (0.2126, 0.7152, 0.0722)

        # Build luminance sample if RGB, else use mono directly
        if data_norm.ndim == 3 and data_norm.shape[0] == 3:
            r_w, g_w, b_w = weights
            L = r_w * data_norm[0] + g_w * data_norm[1] + b_w * data_norm[2]
            base = L
        elif data_norm.ndim == 2 and data_norm.shape[0] == 3:
            r_w, g_w, b_w = weights
            L = r_w * data_norm[0] + g_w * data_norm[1] + b_w * data_norm[2]
            base = L
        elif data_norm.ndim == 3 and data_norm.shape[0] == 1:
            base = data_norm[0]
        else:
            base = data_norm

        stride = max(1, base.size // 2000000)
        sample = base.flatten()[::stride]

        hist, bin_edges = np.histogram(sample, bins=65536, range=(0.0, 1.0))
        hist_smooth = np.convolve(hist, np.ones(50)/50, mode='same')

        # IMPORTANT: do not skip the first bins blindly.
        # For very low-signal linear data (e.g., median ~5e-4), the histogram peak can sit well below bin 100.
        # If we start searching at 100, we may miss the real peak and collapse to the percentile fallback.
        search_start = 100
        if np.max(hist_smooth[:search_start]) > 0:
            search_start = 0
        if search_start >= len(hist_smooth):
            search_start = 0

        peak_idx = int(np.argmax(hist_smooth[search_start:]) + search_start)
        peak_val = float(hist_smooth[peak_idx])
        target_val = peak_val * 0.06

        left_side = hist_smooth[:peak_idx]
        candidates = np.where(left_side < target_val)[0]

        if len(candidates) > 0:
            anchor_idx = candidates[-1]
            anchor = bin_edges[anchor_idx]
        else:
            anchor = np.percentile(sample, 0.5)

        return max(0.0, anchor)

    @staticmethod
    def extract_luminance(data_norm, anchor, weights):
        r_w, g_w, b_w = weights
        img_anchored = np.maximum(data_norm - anchor, 0.0)

        # RGB full image (3,H,W)
        if data_norm.ndim == 3 and data_norm.shape[0] == 3:
            L_anchored = (r_w * img_anchored[0] + g_w * img_anchored[1] + b_w * img_anchored[2])

        # RGB subsample (3,N)
        elif data_norm.ndim == 2 and data_norm.shape[0] == 3:
            L_anchored = (r_w * img_anchored[0] + g_w * img_anchored[1] + b_w * img_anchored[2])

        # Mono as (1,H,W)
        elif data_norm.ndim == 3 and data_norm.shape[0] == 1:
            L_anchored = img_anchored[0]
            img_anchored = img_anchored[0]

        else:
            # Mono as (H,W) or (N,)
            L_anchored = img_anchored

        return L_anchored, img_anchored

    @staticmethod
    def estimate_star_pressure(L_anchored):
        """
        Estimates global 'Star Pressure' from anchored luminance.
        This is a purely statistical, global metric (no masks, no spatial logic).

        Returns a normalized value in [0, 1], where:
          0   = negligible stellar pressure
          1   = extreme stellar dominance (high risk of highlight compression)
        """
        # Safety
        if L_anchored is None or L_anchored.size == 0:
            return 0.0

        # Work on a subsample for performance
        stride = max(1, L_anchored.size // 300000)
        sample = L_anchored.flatten()[::stride]

        # Ignore zeros (pure background floor)
        sample = sample[sample > 1e-7]
        if sample.size < 100:
            return 0.0

        # High-end statistics (stellar signature)
        p999 = np.percentile(sample, 99.9)
        p9999 = np.percentile(sample, 99.99)

        # Fraction of pixels in extreme tail (bright compact structures)
        bright_frac = np.count_nonzero(sample > p999) / sample.size

        # Normalize components
        # p9999 grows quickly when stars dominate
        p_term = np.clip(p9999 / (p999 + 1e-9), 1.0, 5.0)
        p_term = (p_term - 1.0) / 4.0  # normalize to [0,1]

        # bright_frac is usually tiny; rescale aggressively
        f_term = np.clip(bright_frac * 200.0, 0.0, 1.0)

        # Combine (weighted, conservative)
        star_pressure = 0.7 * p_term + 0.3 * f_term
        return float(np.clip(star_pressure, 0.0, 1.0))

    @staticmethod
    def hyperbolic_stretch(data, D, b, SP=0.0):
        D = max(D, 0.1); b = max(b, 0.1)
        term1 = np.arcsinh(D * (data - SP) + b)
        term2 = np.arcsinh(b)
        norm_factor = np.arcsinh(D * (1.0 - SP) + b) - term2
        if abs(norm_factor) < 1e-12: norm_factor = 1e-6
        return (term1 - term2) / norm_factor

    @staticmethod
    def solve_log_d(luma_sample, target_median, b_val):
        median_in = np.median(luma_sample)
        if median_in < 1e-9: return 2.0 
        low_log = 0.0; high_log = 7.0; best_log_D = 2.0
        for _ in range(40):
            mid_log = (low_log + high_log) / 2.0
            mid_D = 10.0 ** mid_log
            test_val = VeraLuxCore.hyperbolic_stretch(median_in, mid_D, b_val)
            if abs(test_val - target_median) < 0.0001: best_log_D = mid_log; break
            if test_val < target_median: low_log = mid_log
            else: high_log = mid_log
        return best_log_D

    @staticmethod
    def apply_mtf(data, m):
        term1 = (m - 1.0) * data
        term2 = (2.0 * m - 1.0) * data - m
        with np.errstate(divide='ignore', invalid='ignore'): res = term1 / term2
        return np.nan_to_num(res, nan=0.0, posinf=1.0, neginf=0.0)

    @staticmethod
    def apply_linear_expansion(data: np.ndarray, factor: float) -> np.ndarray:
        """
        Smart Linear Expansion.
        Uses Nearest Neighbor logic to distinguish between stars (preserve absolute max)
        and hot pixels (fallback to percentile robust clamping).
        """
        if factor <= 0.001:
            VeraLuxCore._last_linear_expansion_diag = {"pct_low": 0.0, "pct_high": 0.0, "low": 0.0, "high": 0.0}
            return data

        factor = float(np.clip(factor, 0.0, 1.0))

        # 1. Absolute Maximum Analysis
        abs_max = float(np.max(data))
        use_absolute_max = False

        if abs_max > 0.001:
            # Nearest Neighbor Check (Identical to RTU)
            idx_max = np.argmax(data)
            y_max, x_max = np.unravel_index(idx_max, data.shape)
            
            y0, y1 = max(0, y_max-1), min(data.shape[0], y_max+2)
            x0, x1 = max(0, x_max-1), min(data.shape[1], x_max+2)
            window = data[y0:y1, x0:x1]
            
            neighbors = window[window < abs_max]
            
            # If it's a real star (bright neighbours), let's use the Absolute Max -> ZERO CLIPPING
            if neighbors.size > 0:
                max_neighbor = np.max(neighbors)
                if max_neighbor >= (abs_max * 0.20):
                    use_absolute_max = True

        # 2. Bounds calculation
        stride = max(1, data.size // 500000)
        sample = data.flatten()[::stride]
        
        low = np.percentile(sample, 0.001)
        
        # Hybrid logic for the high limit
        if use_absolute_max:
            high = abs_max # Mathematical certainty: we include everything
        else:
            high = np.percentile(sample, 99.999) # Robust fallback for hot pixels

        if high <= low:
            VeraLuxCore._last_linear_expansion_diag = {"pct_low": 0.0, "pct_high": 0.0, "low": float(low), "high": float(high)}
            return data

        # 3. Diagnostics
        try:
            pct_low = (float(np.count_nonzero(sample <= low)) / float(sample.size)) * 100.0 if sample.size else 0.0
            # If we use absolute max, pct_high will be mathematically 0.0
            pct_high = (float(np.count_nonzero(sample >= high)) / float(sample.size)) * 100.0 if sample.size else 0.0
            
            VeraLuxCore._last_linear_expansion_diag = {
                "pct_low": float(pct_low),
                "pct_high": float(pct_high),
                "low": float(low),
                "high": float(high)
            }
        except Exception:
            pass

        # 4. Application
        normalized = (data - low) / (high - low)
        normalized = np.clip(normalized, 0.0, 1.0)

        return data * (1.0 - factor) + normalized * factor

# =============================================================================
#  HELPER FUNCTIONS (Ready-to-Use Logic)
# =============================================================================

# --- Ready-to-Use scaling constants (single source of truth) ---
RTU_PEDESTAL = 0.001
RTU_SOFT_CEIL_PERCENTILE = 99.0

def adaptive_output_scaling(img_data, working_space="Rec.709 (Recommended)", 
                            target_bg=0.20, progress_callback=None):
    if progress_callback: progress_callback("Adaptive Scaling: Analyzing Dynamic Range...")
    luma_r, luma_g, luma_b = SENSOR_PROFILES[working_space]['weights']
    is_rgb = (img_data.ndim == 3 and img_data.shape[0] == 3)
    
    if is_rgb:
        R, G, B = img_data[0], img_data[1], img_data[2]
        L_raw = luma_r * R + luma_g * G + luma_b * B
    else:
        L_raw = img_data

    median_L = float(np.median(L_raw))
    std_L = float(np.std(L_raw)); min_L = float(np.min(L_raw))
    global_floor = max(min_L, median_L - 2.7 * std_L)
    PEDESTAL = RTU_PEDESTAL
    
    abs_max = float(np.max(L_raw))
    valid_physical_max = True

    if abs_max > 0.001:
        idx_max = np.argmax(L_raw)
        y_max, x_max = np.unravel_index(idx_max, L_raw.shape)
        
        y0, y1 = max(0, y_max-1), min(L_raw.shape[0], y_max+2)
        x0, x1 = max(0, x_max-1), min(L_raw.shape[1], x_max+2)
        window = L_raw[y0:y1, x0:x1]
        
        neighbors = window[window < abs_max]
        
        if neighbors.size > 0:
            max_neighbor = np.max(neighbors)
            if max_neighbor < (abs_max * 0.20):
                valid_physical_max = False
                if progress_callback: progress_callback("Info: Hot Pixel detected and ignored.")
        elif window.size > 1:
            pass 

    if is_rgb:
        stride = max(1, R.size // 500000)
        soft_ceil = max(
            np.percentile(R.flatten()[::stride], RTU_SOFT_CEIL_PERCENTILE),
            np.percentile(G.flatten()[::stride], RTU_SOFT_CEIL_PERCENTILE),
            np.percentile(B.flatten()[::stride], RTU_SOFT_CEIL_PERCENTILE)
        )
    else:
        stride = max(1, L_raw.size // 200000)
        soft_ceil = np.percentile(L_raw.flatten()[::stride], RTU_SOFT_CEIL_PERCENTILE)
        
    if soft_ceil <= global_floor: soft_ceil = global_floor + 1e-6
    if abs_max <= soft_ceil: abs_max = soft_ceil + 1e-6
    
    scale_contrast = (0.98 - PEDESTAL) / (soft_ceil - global_floor + 1e-9)
    
    if valid_physical_max:
        scale_physical_limit = (1.0 - PEDESTAL) / (abs_max - global_floor + 1e-9)
        final_scale = min(scale_contrast, scale_physical_limit)
    else:
        final_scale = scale_contrast
    
    def expand_channel(c): return np.clip((c - global_floor) * final_scale + PEDESTAL, 0.0, 1.0)
    
    if is_rgb:
        img_data[0] = expand_channel(R); img_data[1] = expand_channel(G); img_data[2] = expand_channel(B)
        L = luma_r * img_data[0] + luma_g * img_data[1] + luma_b * img_data[2]
    else:
        img_data = expand_channel(L_raw); L = img_data
    
    current_bg = float(np.median(L))
    if current_bg > 0.0 and current_bg < 1.0 and abs(current_bg - target_bg) > 1e-3:
        if progress_callback: progress_callback(f"Applying MTF (Bg: {current_bg:.3f} -> {target_bg})")
        m = (current_bg * (target_bg - 1.0)) / (current_bg * (2.0 * target_bg - 1.0) - target_bg)
        if is_rgb:
            for i in range(3): img_data[i] = VeraLuxCore.apply_mtf(img_data[i], m)
        else:
            img_data = VeraLuxCore.apply_mtf(img_data, m)
    return img_data

def apply_ready_to_use_soft_clip(img_data, threshold=0.98, rolloff=2.0, progress_callback=None):
    if progress_callback: progress_callback(f"Final Polish: Soft-clip > {threshold:.2f}")
    def soft_clip_channel(c, thresh, roll):
        mask = c > thresh
        result = c.copy()
        if np.any(mask):
            t = np.clip((c[mask] - thresh) / (1.0 - thresh + 1e-9), 0.0, 1.0)
            result[mask] = thresh + (1.0 - thresh) * (1.0 - np.power(1.0 - t, roll))
        return np.clip(result, 0.0, 1.0)
    if img_data.ndim == 3:
        for i in range(img_data.shape[0]): img_data[i] = soft_clip_channel(img_data[i], threshold, rolloff)
    else:
        img_data = soft_clip_channel(img_data, threshold, rolloff)
    return img_data

def process_veralux_v6(img_data, log_D, protect_b, convergence_power, 
                       working_space="Rec.709 (Recommended)", 
                       processing_mode="ready_to_use",
                       target_bg=None,
                       color_grip=1.0, 
                       shadow_convergence=0.0,
                       linear_expansion=0.0,
                       use_adaptive_anchor=False,
                       progress_callback=None):
    
    if progress_callback: progress_callback("Normalization & Analysis...")
    img = VeraLuxCore.normalize_input(img_data)
    if img.ndim == 3 and img.shape[0] != 3 and img.shape[2] == 3: img = img.transpose(2, 0, 1)

    luma_weights = SENSOR_PROFILES[working_space]['weights']
    is_rgb = (img.ndim == 3 and img.shape[0] == 3)

    if use_adaptive_anchor:
        if progress_callback: progress_callback("Calculating Anchor (Adaptive)...")
        anchor = VeraLuxCore.calculate_anchor_adaptive(img, weights=luma_weights)
    else:
        if progress_callback: progress_callback("Calculating Anchor (Statistical)...")
        anchor = VeraLuxCore.calculate_anchor(img)
    
    if progress_callback: progress_callback(f"Extracting Luminance ({working_space})...")
    L_anchored, img_anchored = VeraLuxCore.extract_luminance(img, anchor, luma_weights)
    
    epsilon = 1e-9; L_safe = L_anchored + epsilon
    if is_rgb:
        r_ratio = img_anchored[0] / L_safe
        g_ratio = img_anchored[1] / L_safe
        b_ratio = img_anchored[2] / L_safe

    if progress_callback: progress_callback(f"Stretching (Log D={log_D:.2f})...")
    L_str = VeraLuxCore.hyperbolic_stretch(L_anchored, 10.0 ** log_D, protect_b)
    L_str = np.clip(L_str, 0.0, 1.0)

    # Reset Linear Expansion diagnostic when feature is off (prevents stale warnings)
    try:
        if processing_mode == "ready_to_use" or float(linear_expansion) <= 0.001:
            VeraLuxCore._last_linear_expansion_diag = {"pct_low": 0.0, "pct_high": 0.0, "low": 0.0, "high": 0.0}
    except Exception:
        pass

    # --- Linear Expansion (Scientific-only) ---
    # Apply AFTER luminance IHS stretch and BEFORE chromatic vector reconstruction.
    if processing_mode != "ready_to_use" and float(linear_expansion) > 0.001:
        if progress_callback:
            progress_callback(f"Linear Expansion: {float(linear_expansion):.2f}")
        L_str = VeraLuxCore.apply_linear_expansion(L_str, float(linear_expansion))
        L_str = np.clip(L_str, 0.0, 1.0)
    
    if progress_callback: progress_callback("Color Convergence & Hybrid Engine...")
    final = np.zeros_like(img)
    
    if is_rgb:
        k = np.power(L_str, convergence_power)
        r_final = r_ratio * (1.0 - k) + 1.0 * k
        g_final = g_ratio * (1.0 - k) + 1.0 * k
        b_final = b_ratio * (1.0 - k) + 1.0 * k
        
        final[0] = L_str * r_final; final[1] = L_str * g_final; final[2] = L_str * b_final
        
        needs_hybrid = (color_grip < 1.0) or (shadow_convergence > 0.01)
        
        if needs_hybrid:
            if progress_callback: progress_callback("Applying Hybrid Grip & Shadow Convergence...")
            D_val = 10.0 ** log_D
            scalar = np.zeros_like(final)
            scalar[0] = VeraLuxCore.hyperbolic_stretch(img_anchored[0], D_val, protect_b)
            scalar[1] = VeraLuxCore.hyperbolic_stretch(img_anchored[1], D_val, protect_b)
            scalar[2] = VeraLuxCore.hyperbolic_stretch(img_anchored[2], D_val, protect_b)
            scalar = np.clip(scalar, 0.0, 1.0)
            
            grip_map = np.full_like(L_str, color_grip)
            
            if shadow_convergence > 0.01:
                damping = np.power(L_str, shadow_convergence)
                grip_map = grip_map * damping
            
            final = (final * grip_map) + (scalar * (1.0 - grip_map))
    else:
        final = L_str

    final = final * (1.0 - 0.005) + 0.005
    final = np.clip(final, 0.0, 1.0).astype(np.float32)
    
    if processing_mode == "ready_to_use":
        # Ready-to-Use has its own output scaling; force Linear Expansion off.
        linear_expansion = 0.0
        effective_bg = 0.20 if target_bg is None else float(target_bg)
        final = adaptive_output_scaling(final, working_space, effective_bg, progress_callback)
        final = apply_ready_to_use_soft_clip(final, 0.98, 2.0, progress_callback)
    
    return final

# =============================================================================
#  LIVE PREVIEW SYSTEM
# =============================================================================

class VeraLuxPreviewWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.setWindowTitle("VeraLux Live Preview")
        self.resize(850, 600)
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint)
        self.setStyleSheet(DARK_STYLESHEET) 

        layout = QVBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(0)
        
        # --- TOOLBAR ---
        toolbar = QWidget(); toolbar.setStyleSheet("background-color: #333333; border-bottom: 1px solid #555555;")
        tb_layout = QHBoxLayout(toolbar); tb_layout.setContentsMargins(5, 5, 5, 5); tb_layout.setSpacing(10)
        
        btn_in = QPushButton("+"); btn_in.setObjectName("ZoomBtn"); btn_in.clicked.connect(self.zoom_in)
        btn_out = QPushButton("-"); btn_out.setObjectName("ZoomBtn"); btn_out.clicked.connect(self.zoom_out)
        btn_fit = QPushButton("Fit"); btn_fit.setObjectName("ZoomBtn"); btn_fit.clicked.connect(self.fit_to_view)

        btn_11 = QPushButton("1:1"); btn_11.setObjectName("ZoomBtn"); btn_11.clicked.connect(self.zoom_1_1)
        
        self.btn_hist = QPushButton("Hist"); self.btn_hist.setObjectName("ZoomBtn"); self.btn_hist.setCheckable(True)
        self.btn_hist.setChecked(True); self.btn_hist.clicked.connect(self.toggle_histogram)
        
        tb_layout.addWidget(btn_out); tb_layout.addWidget(btn_fit); tb_layout.addWidget(btn_11); tb_layout.addWidget(btn_in); tb_layout.addStretch()
        tb_layout.addWidget(self.btn_hist)
        layout.addWidget(toolbar)
        
        # --- SCENE & VIEW ---
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setDragMode(QGraphicsView.DragMode.NoDrag); self.view.setCursor(Qt.CursorShape.CrossCursor)
        self.view.setMouseTracking(True); self.view.viewport().installEventFilter(self)
        self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.view.setStyleSheet("background-color: #1e1e1e; border: none;")
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(self.view)
        
        self.pixmap_item = QGraphicsPixmapItem(); self.scene.addItem(self.pixmap_item)
        self.processed_pixmap = None; self.last_img_data = None; self._last_drag_pos = None
        self.luma_weights = (0.2126, 0.7152, 0.0722)  # Default, updated by set_image
        
        self.lbl_info = QLabel("Loading...", self.view)
        self.lbl_info.setStyleSheet("background-color: rgba(0,0,0,150); color: white; padding: 5px; border-radius: 3px;")
        self.lbl_info.move(10, 10)
        
        self.lbl_hint = QLabel("Double-click to fit", self.view)
        self.lbl_hint.setStyleSheet("color: rgba(255,255,255,100); font-size: 8pt; font-weight: bold;")
        self.lbl_hint.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        # --- FLOATING WIDGETS ---
        
        # 1. Info Overlay Box (NEW)
        self.lbl_info_overlay = QLabel(self.view)
        self.lbl_info_overlay.setFixedWidth(200)
        self.lbl_info_overlay.setStyleSheet("background-color: rgba(0, 0, 0, 180); border: none; border-radius: 5px; color: #eeeeee; padding: 6px;")
        self.lbl_info_overlay.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.lbl_info_overlay.setWordWrap(True)
        self.lbl_info_overlay.hide() # Hidden by default

        # 2. Checkbox Log Scale
        self.chk_log = QCheckBox("Log Scale", self.view)
        self.chk_log.setToolTip("Toggle Logarithmic Scale")
        self.chk_log.setStyleSheet("""
            QCheckBox { color: #aaaaaa; font-weight: bold; background: transparent; }
            QCheckBox::indicator { border: 1px solid #666666; border-radius: 2px; background: #222222; width: 12px; height: 12px; }
            QCheckBox::indicator:checked { background: #88aaff; border: 1px solid #88aaff; }
        """)
        
        # 3. Info Button
        self.btn_info = QPushButton("Info", self.view) 
        self.btn_info.setObjectName("ZoomBtn"); self.btn_info.setCheckable(True)
        self.btn_info.setToolTip("Force Detailed Diagnostics Overlay")
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
        
        # 4. Histogram
        self.histogram = HistogramOverlay(self.view)
        self.chk_log.toggled.connect(self.histogram.set_log_scale)
        self.histogram.stats_updated.connect(self.lbl_info_overlay.setText)

        # --- LOAD PREVIEW SETTINGS (PERSISTENCE) ---
        settings = QSettings("VeraLux", "HyperMetricStretch")
        
        # 1. Load the saved values (or defaults)
        hist_on = settings.value("preview_hist_visible", True, type=bool)
        log_on = settings.value("preview_log_scale", False, type=bool)
        info_on = settings.value("preview_info_visible", False, type=bool)
        
        # 2. Set the visual state of the buttons
        self.btn_hist.setChecked(hist_on)
        self.chk_log.setChecked(log_on)
        self.btn_info.setChecked(info_on)
        
        # 3. Apply logic (calls the methods to update the UI)
        self.histogram.set_log_scale(log_on)
        self.toggle_info_overlay() # Mostra/Nasconde il box info
        self.toggle_histogram()    # Mostra/Nasconde tutto il blocco istogramma

    def set_image(self, qimg, img_data_for_hist, source_saturated=False, mode="ready_to_use", luma_weights=None):
        self.last_img_data = img_data_for_hist # Store for tracker sampling
        if luma_weights is not None:
            self.luma_weights = luma_weights
        pixmap = QPixmap.fromImage(qimg)
        self.processed_pixmap = pixmap
        self.pixmap_item.setPixmap(pixmap)
        self.scene.setSceneRect(self.pixmap_item.boundingRect())
        
        diag = getattr(VeraLuxCore, "_last_linear_expansion_diag", None)

        self.histogram.set_context(source_saturated, mode, diag)
        self.histogram.set_data(img_data_for_hist)

        msg = "Preview Updated"
        try:
            if mode != "ready_to_use" and diag:
                pct_lo = float(diag.get("pct_low", 0.0) or 0.0)
                pct_hi = float(diag.get("pct_high", 0.0) or 0.0)
                if (pct_lo + pct_hi) >= 0.01:
                    msg = f"Linear Expansion clamp — low {pct_lo:.3f}% • high {pct_hi:.3f}%"
        except Exception:
            pass

        self.lbl_info.setText(msg)
        self.lbl_info.adjustSize()
        
        self.update_overlays_pos()

    def update_overlays_pos(self):
        w, h = self.view.width(), self.view.height()
        self.lbl_hint.move(w - 110, h - 25); self.lbl_hint.raise_()
        
        spacing = 8

        # A. Histogram (Background)
        hist_x = 10
        hist_y = h - 130 
        self.histogram.move(hist_x, hist_y); self.histogram.raise_()
        
        # B. Buttons (Above Histogram)
        self.chk_log.adjustSize(); self.btn_info.adjustSize()
        
        row_height = max(self.chk_log.height(), self.btn_info.height())
        
        btns_y = hist_y - row_height - spacing 
        
        self.chk_log.move(hist_x, btns_y); self.chk_log.raise_()
        
        info_btn_x = hist_x + self.chk_log.width() + 10 # 10px orizzontale tra Log e Info
        self.btn_info.move(info_btn_x, btns_y); self.btn_info.raise_()
        
        # C. Box Info Overlay (Above Buttons)
        if self.lbl_info_overlay.isVisible():
            self.lbl_info_overlay.adjustSize()
            self.lbl_info_overlay.setFixedWidth(200) 
            
            box_h = self.lbl_info_overlay.height()
            
            box_y = btns_y - box_h - spacing
            
            self.lbl_info_overlay.move(hist_x, box_y)
            self.lbl_info_overlay.raise_()

    def eventFilter(self, source, event):
        """
        Handles mouse events on the viewport.
        Version: Smart High-Pass 3x3 (Stabilized) + Y-Flip Fix.
        """
        # --- 1. MANUAL DRAG MANAGEMENT (PANNING) ---
        if event.type() == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.LeftButton:
                self._last_drag_pos = event.pos()
                self.view.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
                return True # Event consumed
                
        elif event.type() == QEvent.Type.MouseButtonRelease:
            if event.button() == Qt.MouseButton.LeftButton:
                self.view.viewport().setCursor(Qt.CursorShape.CrossCursor)
                self._last_drag_pos = None
                return True

        elif event.type() == QEvent.Type.MouseMove:
            # If dragging (Left button pressed)
            if event.buttons() & Qt.MouseButton.LeftButton:
                if self._last_drag_pos is not None:
                    delta = event.pos() - self._last_drag_pos
                    self._last_drag_pos = event.pos()
                    
                    # Manual scroll
                    hs = self.view.horizontalScrollBar()
                    vs = self.view.verticalScrollBar()
                    hs.setValue(hs.value() - delta.x())
                    vs.setValue(vs.value() - delta.y())
                return True # Stop, do not process tracker during drag

            # --- 2. TRACKER MANAGEMENT (SMART HIGH-PASS 3x3) ---
            self.view.viewport().setCursor(Qt.CursorShape.CrossCursor)
            
            if self.last_img_data is not None:
                pos = self.view.mapToScene(event.pos())
                if self.pixmap_item.contains(pos):
                    ix, iy = int(pos.x()), int(pos.y())
                    
                    # Matrix 3x3 (Radius 1)
                    radius = 1
                    h, w = self.last_img_data.shape[-2:]
                    
                    # Coordinate calculation with Y-Flip FIX and limits
                    # Note: Y-Flip applies to CENTER, then expand radius
                    iy_flipped = h - 1 - iy
                    
                    x_start, x_end = max(0, ix - radius), min(w, ix + radius + 1)
                    y_start, y_end = max(0, iy_flipped - radius), min(h, iy_flipped + radius + 1)
                    
                    if x_start < x_end and y_start < y_end:
                        if self.last_img_data.ndim == 3:
                            # RGB: Extract ROI
                            roi = self.last_img_data[:, y_start:y_end, x_start:x_end]
                            # Calculate Luminance for every pixel in matrix
                            luma_roi = self.luma_weights[0]*roi[0] + self.luma_weights[1]*roi[1] + self.luma_weights[2]*roi[2]
                        else:
                            # Mono
                            luma_roi = self.last_img_data[y_start:y_end, x_start:x_end]
                        
                        # --- SMART FILTER ---
                        # Sort pixels by brightness
                        flat = np.sort(luma_roi.flatten())
                        # Take top half (Top 50% + 1) to ignore blacks/lows
                        # On 9 pixels, takes 5.
                        n_pixels = max(1, len(flat) // 2 + 1)
                        val = np.mean(flat[-n_pixels:])
                        
                        self.histogram.set_tracker(val)
                    else:
                        self.histogram.set_tracker(None)
                else:
                    self.histogram.set_tracker(None)
                    
        return super().eventFilter(source, event)

    def resizeEvent(self, event):
        self.update_overlays_pos()
        super().resizeEvent(event)
        
    def toggle_histogram(self):
        visible = self.btn_hist.isChecked()
        
        self.histogram.setVisible(visible)
        self.histogram.is_visible = visible
        
        self.chk_log.setVisible(visible)
        self.btn_info.setVisible(visible)
        
        if visible and self.btn_info.isChecked():
            self.lbl_info_overlay.setVisible(True)
            self.update_overlays_pos()
        else:
            self.lbl_info_overlay.setVisible(False)

    def fit_to_view(self):
        if self.pixmap_item.pixmap(): self.view.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def zoom_1_1(self):
        self.view.resetTransform()
        if self.pixmap_item.pixmap():
            self.view.centerOn(self.pixmap_item)

    def zoom_in(self): self.view.scale(1.2, 1.2)

    def zoom_out(self): self.view.scale(1/1.2, 1/1.2)

    def wheelEvent(self, event: QWheelEvent):
        if event.angleDelta().y() > 0: self.zoom_in()
        else: self.zoom_out()

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton: self.fit_to_view()

    def toggle_info_overlay(self):
        is_checked = self.btn_info.isChecked()
        self.lbl_info_overlay.setVisible(is_checked)
        self.update_overlays_pos()

    def closeEvent(self, event):
        settings = QSettings("VeraLux", "HyperMetricStretch")
        settings.setValue("preview_hist_visible", self.btn_hist.isChecked())
        settings.setValue("preview_log_scale", self.chk_log.isChecked())
        settings.setValue("preview_info_visible", self.btn_info.isChecked())
        event.accept()

# =============================================================================
#  THREADING
# =============================================================================

class AutoSolverThread(QThread):
    result_ready = pyqtSignal(float)
    def __init__(self, data, target, b_val, luma_weights, adaptive, processing_mode):
        super().__init__()
        self.data = data; self.target = target; self.b_val = b_val
        self.luma_weights = luma_weights; self.adaptive = adaptive
        self.processing_mode = processing_mode
        
    def run(self):
        try:
            # 1. Prepare Data Subsample
            img_norm = VeraLuxCore.normalize_input(self.data) 
            if img_norm.ndim == 3 and img_norm.shape[0] != 3 and img_norm.shape[2] == 3:
                img_norm = img_norm.transpose(2, 0, 1)
            
            if img_norm.ndim == 3:
                h, w = img_norm.shape[1], img_norm.shape[2]
                num_pixels = h * w
                step = max(1, num_pixels // 100000)
                
                c0 = img_norm[0].flatten()[::step]
                c1 = img_norm[1].flatten()[::step]
                c2 = img_norm[2].flatten()[::step]
                sub_data = np.vstack((c0, c1, c2))
            else:
                h, w = img_norm.shape
                num_pixels = h * w
                step = max(1, num_pixels // 100000)
                
                sub_data = img_norm.flatten()[::step]

            # 2. Setup Anchor & Luminance
            if self.adaptive:
                anchor = VeraLuxCore.calculate_anchor_adaptive(sub_data, weights=self.luma_weights)
            else:
                anchor = VeraLuxCore.calculate_anchor(sub_data)

            # Extract Luminance
            L_anchored, _ = VeraLuxCore.extract_luminance(sub_data, anchor, self.luma_weights)
            # --- STAR PRESSURE ESTIMATION (GLOBAL, STATISTICAL) ---
            star_pressure = VeraLuxCore.estimate_star_pressure(L_anchored)
            
            # Use only VALID signal (non-zero) to model the Sky Background distribution
            valid = L_anchored[L_anchored > 1e-7]

            if len(valid) == 0:
                self.result_ready.emit(2.0)
                return

            # 3. Smart Iterative Solver Loop ("Floating Sky Check")
            target_temp = self.target
            best_log_d = 2.0
            
            # Max 15 iterations
            for _ in range(15):
                best_log_d = VeraLuxCore.solve_log_d(valid, target_temp, self.b_val)
                # --- STAR-AWARE ADAPTATION ---
                # If stellar pressure is high, reduce aggressiveness
                if star_pressure > 0.6:
                    # Damp target to avoid star-driven over-compression
                    target_temp *= (1.0 - 0.15 * star_pressure)

                # If Scientific Mode, no auto-protection needed
                if self.processing_mode != "ready_to_use":
                    break

                # B. SIMULATION: Stretch the sky background signal
                D = 10.0 ** best_log_d
                valid_str = VeraLuxCore.hyperbolic_stretch(valid, D, self.b_val)

                # C. CHECK: Does the noise tail land safely on zero?
                # Align with adaptive_output_scaling:
                # global_floor = max(min_L, median_L - 2.7 * std_L)
                med = float(np.median(valid_str))
                std = float(np.std(valid_str))
                min_v = float(np.min(valid_str))
                global_floor = max(min_v, med - (2.7 * std))

                # If global_floor is still above ~0, the later adaptive scaling will
                # need to force it down, effectively clipping real low-signal data.
                if global_floor <= 0.001:
                    break

                # D. Adjustment
                # Reduce target to lower the whole bell curve
                target_temp -= 0.015
                if target_temp < 0.05: break

            self.result_ready.emit(best_log_d)
            
        except Exception as e:
            print(f"Solver Error: {e}")
            self.result_ready.emit(2.0)

class ProcessingThread(QThread):
    finished = pyqtSignal(object); progress = pyqtSignal(str)
    def __init__(self, img, D, b, conv, working_space, processing_mode, target_bg, color_grip, shadow_convergence, linear_expansion, adaptive):
        super().__init__()
        self.img = img; self.D = D; self.b = b; self.conv = conv
        self.working_space = working_space; self.processing_mode = processing_mode; self.target_bg = target_bg
        self.color_grip = color_grip; self.shadow_convergence = shadow_convergence
        self.linear_expansion = linear_expansion
        self.adaptive = adaptive
    def run(self):
        try:
            res = process_veralux_v6(self.img, self.D, self.b, self.conv, self.working_space, 
                                   self.processing_mode, self.target_bg, self.color_grip, 
                                   self.shadow_convergence, self.linear_expansion, self.adaptive, self.progress.emit)
            self.finished.emit(res)
        except Exception as e: 
            traceback.print_exc(); self.progress.emit(f"Error: {str(e)}")

# =============================================================================
#  GUI
# =============================================================================

class VeraLuxInterface:
    def __init__(self, siril_app, qt_app):
        self.siril = siril_app
        self.app = qt_app
        
        # --- HEADER LOG ---
        header_msg = (
            "\n##############################################\n"
            "# VeraLux — HyperMetric Stretch\n"
            "# Photometric Hyperbolic Stretch Engine\n"
            "# Author: Riccardo Paterniti (2025)\n"
            "# Contact: info@veralux.space\n"
            "##############################################"
        )
        try:
            self.siril.log(header_msg)
        except:
            print(header_msg)

        self.linear_cache = None
        self.is_source_saturated = False # Metadata for Diagnostics
        self.preview_proxy = None # Low-res copy for preview
        self.preview_window = None
        
        self.window = QMainWindow()
        # Clean Exit handler
        self.window.closeEvent = self.handle_close_event
        
        # Init Settings (Persistence)
        self.settings = QSettings("VeraLux", "HyperMetricStretch")
        
        self.window.setWindowTitle(f"VeraLux v{VERSION}")
        self.app.setStyle("Fusion") 
        self.window.setStyleSheet(DARK_STYLESHEET)
        self.window.setMinimumWidth(620) 
        self.window.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)
        central = QWidget()
        self.window.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(8) 
        
        # Header
        h_head = QHBoxLayout()
        h_head.addSpacing(30)
        h_head.addStretch()
        head_title = QLabel(f"VeraLux HyperMetric Stretch v{VERSION}")
        head_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        head_title.setStyleSheet("font-size: 14pt; font-weight: bold; color: #88aaff;")
        h_head.addWidget(head_title)
        h_head.addStretch()

        self.btn_coffee = QPushButton("☕")
        _nofocus(self.btn_coffee)
        self.btn_coffee.setObjectName("CoffeeButton")
        self.btn_coffee.setToolTip("Support the developer - Buy me a coffee!")
        self.btn_coffee.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_coffee.clicked.connect(lambda: webbrowser.open("https://buymeacoffee.com/riccardo.paterniti"))
        h_head.addWidget(self.btn_coffee)
        layout.addLayout(h_head)
        
        subhead = QLabel("Requirement: Linear Data • Color Calibration (SPCC) Applied")
        subhead.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subhead.setStyleSheet("font-size: 9pt; color: #999999; font-style: italic; margin-bottom: 5px;")
        layout.addWidget(subhead)
        
        # --- GUI BLOCKS ---
        # 0. Mode
        grp_mode = QGroupBox("0. Processing Mode")
        l_mode = QVBoxLayout(grp_mode)
        self.radio_ready = QRadioButton("Ready-to-Use (Aesthetic)")
        self.radio_ready.setToolTip(
            "<b>Ready-to-Use Mode:</b><br>"
            "Produces an aesthetic, export-ready image.<br>"
            "• Applies <b>Smart Max</b> scaling (Zero-Clipping on stars).<br>"
            "• Applies Linked MTF to set background.<br>"
            "• Soft-clips highlights to reduce star blooming."
        )
        self.radio_scientific = QRadioButton("Scientific (Preserve)")
        self.radio_scientific.setToolTip(
            "<b>Scientific Mode:</b><br>"
            "Produces a 100% mathematically consistent output.<br>"
            "• Clips only at physical saturation (1.0).<br>"
            "• Ideal for manual tone mapping (Curves/Hyperbolic)."
        )
        self.radio_ready.setChecked(True) 
        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.radio_ready, 0)
        self.mode_group.addButton(self.radio_scientific, 1)
        l_mode.addWidget(self.radio_ready)
        l_mode.addWidget(self.radio_scientific)
        self.label_mode_info = QLabel("✓ Ready-to-Use selected")
        self.label_mode_info.setStyleSheet("color: #999999; font-size: 9pt;")
        l_mode.addWidget(self.label_mode_info)
        
        # 1. Sensor
        grp_space = QGroupBox("1. Sensor Calibration")
        l_space = QVBoxLayout(grp_space)
        l_combo = QHBoxLayout()
        l_combo.addWidget(QLabel("Sensor Profile:")) 
        self.combo_profile = QComboBox()
        self.combo_profile.setToolTip(
            "<b>Sensor Profile:</b><br>"
            "Defines the Luminance coefficients (Weights) used for the stretch.<br>"
            "Choose <b>Rec.709</b> for general use or a specific sensor profile if known."
        )
        for profile_name in SENSOR_PROFILES.keys(): self.combo_profile.addItem(profile_name)
        self.combo_profile.setCurrentText(DEFAULT_PROFILE)
        l_combo.addWidget(self.combo_profile)
        l_space.addLayout(l_combo)
        self.label_profile_info = QLabel("Rec.709 Standard")
        self.label_profile_info.setStyleSheet("color: #999999; font-size: 9pt;")
        l_space.addWidget(self.label_profile_info)
        
        top_row = QHBoxLayout()
        top_row.addWidget(grp_mode); top_row.addWidget(grp_space)
        layout.addLayout(top_row)
        
        # 2. Stretch & Calibration
        grp_combined = QGroupBox("2. Stretch Engine && Calibration")
        l_combined = QVBoxLayout(grp_combined)
        
        # Target + Auto Button + Preview Button
        l_calib = QHBoxLayout()
        lbl_target_bg = QLabel("Target Bg:")
        lbl_target_bg.setFixedWidth(60)
        lbl_target_bg.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        l_calib.addWidget(lbl_target_bg)
        self.spin_target = QDoubleSpinBox()
        self.spin_target.setToolTip(
            "<b>Target Background (Median):</b><br>"
            "The desired median value for the background sky.<br>"
            "• <b>0.20</b> is standard for good visibility (Statistical Stretch style).<br>"
            "• <b>0.12</b> for high-contrast dark skies."
        )
        self.spin_target.setRange(0.05, 0.50); self.spin_target.setValue(0.20); self.spin_target.setSingleStep(0.01)
        self.spin_target.setFixedWidth(70)
        l_calib.addWidget(self.spin_target)
        
        self.chk_adaptive = QCheckBox("Adaptive Anchor")
        self.chk_adaptive.setChecked(True)
        self.chk_adaptive.setToolTip(
            "<b>Adaptive Anchor:</b><br>"
            "Analyzes the histogram shape to find the true signal start<br>"
            "instead of using a fixed percentile.<br>"
            "Recommended for images with gradients or vignetting to maximize contrast."
        )
        l_calib.addWidget(self.chk_adaptive)
        
        self.btn_auto = QPushButton("⚡ Auto-Calc Log D")
        self.btn_auto.setToolTip(
            "<b>Auto-Solver:</b><br>"
            "Analyzes the image data to find the <b>Stretch Factor (Log D)</b><br>"
            "that places the current background median at the Target Level."
        )
        self.btn_auto.setObjectName("AutoButton")
        self.btn_auto.clicked.connect(self.run_solver)
        l_calib.addWidget(self.btn_auto)
        
        self.btn_preview = QPushButton("👁️ Live Preview")
        self.btn_preview.setObjectName("PreviewButton")
        self.btn_preview.setToolTip("Toggle Real-Time Interactive Preview Window")
        self.btn_preview.clicked.connect(self.toggle_preview)
        l_calib.addWidget(self.btn_preview)
        
        l_combined.addLayout(l_calib)
        l_combined.addSpacing(5)
        
        # Manual Sliders
        l_manual = QHBoxLayout()
        lbl_logd = QLabel("Log D:")
        lbl_logd.setFixedWidth(60)
        lbl_logd.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        l_manual.addWidget(lbl_logd)
        self.spin_d = QDoubleSpinBox()
        self.spin_d.setToolTip(
            "<b>Hyperbolic Intensity (Log D):</b><br>"
            "Controls the strength of the stretch."
        )
        self.spin_d.setRange(0.0, 7.0)
        self.spin_d.setValue(2.0)
        self.spin_d.setDecimals(2)
        self.spin_d.setSingleStep(0.1)
        self.spin_d.setFixedWidth(70)
        self.slide_d = ResetSlider(Qt.Orientation.Horizontal, default_value=200)
        self.slide_d.setRange(0, 700)
        self.slide_d.setValue(200)
        l_manual.addWidget(self.spin_d)
        l_manual.addWidget(self.slide_d)
        l_combined.addLayout(l_manual)

        # Sync Log D slider <-> spinbox
        self.slide_d.valueChanged.connect(lambda v: self.spin_d.setValue(v/100.0))
        self.spin_d.valueChanged.connect(lambda v: self.slide_d.setValue(int(v*100)))

        # --- Protect b Slider (Coherent with Log D) ---
        l_manual_b = QHBoxLayout()
        lbl_protect_b = QLabel("Protect b:")
        lbl_protect_b.setFixedWidth(60)
        lbl_protect_b.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        l_manual_b.addWidget(lbl_protect_b)
        
        self.spin_b = QDoubleSpinBox()
        self.spin_b.setToolTip(
            "<b>Highlight Protection (b):</b><br>"
            "Controls the knee of the Hyperbolic curve.<br>"
            "Higher values preserve stellar cores and highlights."
        )
        self.spin_b.setRange(0.1, 15.0)
        self.spin_b.setValue(6.0)
        self.spin_b.setSingleStep(0.1)
        self.spin_b.setDecimals(2)
        self.spin_b.setFixedWidth(70)
        
        self.slide_b = ResetSlider(Qt.Orientation.Horizontal, default_value=600)
        self.slide_b.setRange(1, 1500)
        self.slide_b.setValue(600)
        
        # Sync Protect b
        self.slide_b.valueChanged.connect(lambda v: self.spin_b.setValue(v/100.0))
        self.spin_b.valueChanged.connect(lambda v: self.slide_b.setValue(int(v*100)))
        
        l_manual_b.addWidget(self.spin_b)
        l_manual_b.addWidget(self.slide_b)
        l_combined.addLayout(l_manual_b)
        layout.addWidget(grp_combined)
        
        # 3. Physics
        grp_phys = QGroupBox("3. Physics && Color Engine")
        l_phys = QVBoxLayout(grp_phys)
        
        l_conv = QHBoxLayout()
        l_conv.addStretch() 
        l_conv.addWidget(QLabel("Star Core Recovery (White Point):"))
        
        self.spin_conv = QDoubleSpinBox()
        self.spin_conv.setToolTip(
            "<b>Color Convergence:</b><br>"
            "Controls how quickly saturated colors transition to white.<br>"
            "• Mimics the physical response of sensors/film.<br>"
            "• Higher values = Faster transition to white core (avoids color artifacts)."
        )
        self.spin_conv.setRange(1.0, 10.0); self.spin_conv.setValue(3.5); self.spin_conv.setSingleStep(0.1)
        self.spin_conv.setFixedWidth(70) 
        
        l_conv.addWidget(self.spin_conv)
        
        l_phys.addLayout(l_conv)
        
        # --- DYNAMIC COLOR CONTROLS ---        
        self.container_ready = QWidget()
        l_ready = QVBoxLayout(self.container_ready)
        l_ready.setContentsMargins(0,0,0,0)
        l_ready.setSpacing(2)
        
        l_uni = QHBoxLayout()
        l_uni.addWidget(QLabel("Color Strategy:"))
        self.slide_unified = ResetSlider(Qt.Orientation.Horizontal, default_value=0)
        self.slide_unified.setToolTip(
            "<b>Unified Color Strategy:</b><br>"
            "• <b>Center (0):</b> Balanced Vector stretch.<br>"
            "• <b>Left (<0):</b> Clean Noise (Increases Shadow Convergence).<br>"
            "• <b>Right (>0):</b> Soften Highlights (Decreases Color Grip).<br>"
            "<i>Double-click to reset.</i>"
        )
        self.slide_unified.setRange(-100, 100); self.slide_unified.setValue(0)
        l_uni.addWidget(self.slide_unified)
        l_ready.addLayout(l_uni)
        
        self.lbl_strategy_feedback = QLabel("Balanced (Pure Vector)")
        self.lbl_strategy_feedback.setStyleSheet("color: #999999; font-size: 8pt; font-style: italic; margin-left: 80px;")
        l_ready.addWidget(self.lbl_strategy_feedback)
        
        self.container_scientific = QWidget()
        l_sci = QVBoxLayout(self.container_scientific)
        l_sci.setContentsMargins(0,0,0,0)

        # Linear Expansion Slider (Scientific mode)
        l_exp = QHBoxLayout()
        lbl_exp = QLabel("Linear Expansion:")
        lbl_exp.setFixedWidth(120)
        lbl_exp.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        l_exp.addWidget(lbl_exp)

        self.spin_exp = QDoubleSpinBox()
        self.spin_exp.setToolTip(
            "<b>Linear Expansion (Scientific only):</b><br>"
            "Normalizes data to fill the dynamic range (0-1).<br>"
            "• <b>Low:</b> Anchors blacks (0.001%) to remove background haze.<br>"
            "• <b>High:</b> Expands to the absolute physical limit using <b>Smart Max</b> logic (preserves star cores, rejects hot pixels).<br>"
            "<i>Increases contrast and brightness simultaneously.</i>"
        )
        self.spin_exp.setRange(0.0, 1.0)
        self.spin_exp.setValue(0.0)
        self.spin_exp.setSingleStep(0.01)
        self.spin_exp.setDecimals(2)
        self.spin_exp.setFixedWidth(70)

        self.slide_exp = ResetSlider(Qt.Orientation.Horizontal, default_value=0)
        self.slide_exp.setRange(0, 100)
        self.slide_exp.setValue(0)

        # Sync Linear Expansion
        self.slide_exp.valueChanged.connect(lambda v: self.spin_exp.setValue(v/100.0))
        self.spin_exp.valueChanged.connect(lambda v: self.slide_exp.setValue(int(v*100)))

        l_exp.addWidget(self.spin_exp)
        l_exp.addWidget(self.slide_exp)
        l_sci.addLayout(l_exp)

        # Color Grip Slider (Scientific mode)
        l_grip = QHBoxLayout()
        lbl_grip = QLabel("Color Grip (Global):")
        lbl_grip.setFixedWidth(120)
        lbl_grip.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        l_grip.addWidget(lbl_grip)
        self.spin_grip = QDoubleSpinBox()
        self.spin_grip.setToolTip(
            "<b>Color Grip:</b> Controls the rigor of Color Vector preservation.<br>"
            "• <b>1.00 (Default):</b> Pure VeraLux. 100% Vector lock. Maximum vividness.<br>"
            "• <b>< 1.00:</b> Blends with standard Scalar stretch. Softens star cores and relaxes saturation in highlights."
        )
        self.spin_grip.setRange(0.0, 1.0)
        self.spin_grip.setValue(1.0)
        self.spin_grip.setSingleStep(0.05)
        self.spin_grip.setFixedWidth(70)
        self.slide_grip = ResetSlider(Qt.Orientation.Horizontal, default_value=100)
        self.slide_grip.setRange(0, 100)
        self.slide_grip.setValue(100)

        # Sync Grip
        self.slide_grip.valueChanged.connect(lambda v: self.spin_grip.setValue(v/100.0))
        self.spin_grip.valueChanged.connect(lambda v: self.slide_grip.setValue(int(v*100)))

        l_grip.addWidget(self.spin_grip)
        l_grip.addWidget(self.slide_grip)
        l_sci.addLayout(l_grip)

        # Shadow Convergence Slider (Scientific mode)
        l_shadow = QHBoxLayout()
        lbl_shadow = QLabel("Shadow Conv. (Noise):")
        lbl_shadow.setFixedWidth(120)
        lbl_shadow.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        l_shadow.addWidget(lbl_shadow)
        self.spin_shadow = QDoubleSpinBox()
        self.spin_shadow.setToolTip(
            "<b>Shadow Convergence:</b><br>"
            "Damps the vector preservation in deep shadows to prevent color noise bloom.<br>"
            "• <b>0.0:</b> Off (Pure Vector in shadows).<br>"
            "• <b>> 0.0:</b> Blends towards scalar stretch in dark areas.<br>"
            "Recommended for noisy images."
        )
        self.spin_shadow.setRange(0.0, 3.0)
        self.spin_shadow.setValue(0.0)
        self.spin_shadow.setSingleStep(0.1)
        self.spin_shadow.setFixedWidth(70)
        self.slide_shadow = ResetSlider(Qt.Orientation.Horizontal, default_value=0)
        self.slide_shadow.setRange(0, 300)
        self.slide_shadow.setValue(0)

        # Sync Shadow
        self.slide_shadow.valueChanged.connect(lambda v: self.spin_shadow.setValue(v/100.0))
        self.spin_shadow.valueChanged.connect(lambda v: self.slide_shadow.setValue(int(v*100)))

        l_shadow.addWidget(self.spin_shadow)
        l_shadow.addWidget(self.slide_shadow)
        l_sci.addLayout(l_shadow)

        l_phys.addWidget(self.container_ready)
        l_phys.addWidget(self.container_scientific)
        
        layout.addWidget(grp_phys)
        
        # Footer
        self.progress = QProgressBar(); self.progress.setTextVisible(True)
        layout.addWidget(self.progress)
        self.status = QLabel("Ready. Please cache input first.")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status)
        self.lbl_stats = QLabel("")
        self.lbl_stats.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_stats.setStyleSheet("font-size: 9pt; font-weight: bold; margin-bottom: 2px;")
        layout.addWidget(self.lbl_stats)
        
        # Buttons
        btns = QHBoxLayout()
        self.btn_help = QPushButton("?"); self.btn_help.setObjectName("HelpButton"); self.btn_help.setFixedWidth(20)
        self.btn_help.setToolTip("Print Operational Guide to Siril Console")
        self.chk_ontop = QCheckBox("Always on top"); self.chk_ontop.setChecked(True)
        self.chk_ontop.setToolTip("Keep this window above Siril")
        b_reset = QPushButton("Defaults")
        b_reset.setToolTip("Reset all sliders and dropdowns to default values.")
        b_reload = QPushButton("Reload Input")
        b_reload.setToolTip("Reload linear image from Siril memory. For Undo must use Siril back button.")
        b_proc = QPushButton("PROCESS"); b_proc.setObjectName("ProcessButton")
        b_proc.setToolTip("Apply the stretch to the image.")
        b_close = QPushButton("Close"); b_close.setObjectName("CloseButton")
        
        btns.addWidget(self.btn_help); btns.addWidget(self.chk_ontop)
        btns.addWidget(b_reset); btns.addWidget(b_reload); btns.addWidget(b_proc); btns.addWidget(b_close)
        layout.addLayout(btns)
        
        # CONNECT SIGNALS
        self.chk_ontop.toggled.connect(self.toggle_ontop)
        self.btn_help.clicked.connect(self.print_help_to_console)
        b_reset.clicked.connect(self.set_defaults)
        b_reload.clicked.connect(self.cache_input)
        b_proc.clicked.connect(self.run_process)
        b_close.clicked.connect(self.window.close)
        
        self.radio_ready.toggled.connect(self.update_mode_ui)
        self.radio_scientific.toggled.connect(self.update_mode_ui)
        
        self.combo_profile.currentTextChanged.connect(self.update_profile_info)
        self.slide_unified.valueChanged.connect(self.update_unified_feedback)
        
        # LIVE PREVIEW CONNECTIONS
        self.debounce_timer = QTimer()
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.setInterval(150) # 150ms delay
        self.debounce_timer.timeout.connect(self.update_preview_image)
        
        # Trigger preview update on changes
        for widget in [self.spin_d, self.spin_b, self.spin_conv, self.spin_target, self.spin_grip, self.spin_shadow, self.spin_exp]:
            widget.valueChanged.connect(self.trigger_preview_update)
        self.slide_exp.valueChanged.connect(self.trigger_preview_update)
        self.slide_unified.valueChanged.connect(self.trigger_preview_update)
        self.combo_profile.currentTextChanged.connect(self.trigger_preview_update)
        self.radio_ready.toggled.connect(self.trigger_preview_update)
        self.slide_d.valueChanged.connect(self.trigger_preview_update)
        self.slide_grip.valueChanged.connect(self.trigger_preview_update)
        self.slide_shadow.valueChanged.connect(self.trigger_preview_update)
        self.chk_adaptive.toggled.connect(self.trigger_preview_update)

        self.update_profile_info(DEFAULT_PROFILE)
        
        # --- LOAD SAVED SETTINGS (QSettings) ---
        # 1. Sensor Profile
        saved_profile = self.settings.value("sensor_profile", DEFAULT_PROFILE)
        if saved_profile in SENSOR_PROFILES:
            self.combo_profile.setCurrentText(saved_profile)
            
        # 2. Mode (Default: Ready-to-Use)
        is_ready = self.settings.value("mode_ready", True, type=bool)
        if is_ready: self.radio_ready.setChecked(True)
        else: self.radio_scientific.setChecked(True)
        
        # 3. Target Background
        saved_target = self.settings.value("target_bg", 0.20, type=float)
        self.spin_target.setValue(saved_target)

        # 4. Linear Expansion (Scientific)
        # Always start at 0.00 for Scientific safety (do not restore from settings)
        self.spin_exp.setValue(0.0)
        self.slide_exp.setValue(0)
        try:
            self.settings.setValue("linear_expansion", 0.0)
        except Exception:
            pass
        
        self.update_mode_ui()
        self.window.show()
        self.center_window()
        self.cache_input() # Initial cache

    # --- LIVE PREVIEW LOGIC ---    
    def get_effective_params(self):
        """Calculates grip, shadow convergence, and linear expansion based on mode."""
        if self.radio_ready.isChecked():
            val = self.slide_unified.value()
            if val < 0:
                # Left: Increase Shadow Convergence, Grip stays 1.0
                shadow = (abs(val) / 100.0) * 3.0
                grip = 1.0
            else:
                # Right: Decrease Grip, Shadow stays 0.0
                grip = 1.0 - ((val / 100.0) * 0.6) # Max reduction to 0.4
                shadow = 0.0
            linear_expansion = 0.0  # Forced off in Ready-to-Use
        else:
            grip = self.spin_grip.value()
            shadow = self.spin_shadow.value()
            linear_expansion = self.spin_exp.value()
        return grip, shadow, linear_expansion

    def update_unified_feedback(self):
        val = self.slide_unified.value()
        grip, shadow, _ = self.get_effective_params()
        
        if val < 0:
            txt = f"Action: Noise Cleaning (Shadow Conv: {shadow:.1f})"
        elif val > 0:
            txt = f"Action: Highlight Softening (Grip: {grip:.2f})"
        else:
            txt = "Balanced (Pure Vector)"
        self.lbl_strategy_feedback.setText(txt)

    def update_mode_ui(self):
        is_ready = self.radio_ready.isChecked()
        self.container_ready.setVisible(is_ready)
        self.container_scientific.setVisible(not is_ready)

        # Scientific-only control
        self.spin_exp.setVisible(not is_ready)
        self.slide_exp.setVisible(not is_ready)
        self.spin_exp.setEnabled(not is_ready)
        self.slide_exp.setEnabled(not is_ready)
        
        if is_ready:
            self.label_mode_info.setText("✓ Ready-to-Use: Unified Color Strategy enabled.")
        else:
            self.label_mode_info.setText("✓ Scientific: Full manual parameter control.")
        
        # Force re-layout
        QTimer.singleShot(10, self.window.adjustSize)

    def toggle_preview(self):
        if not self.preview_window:
            self.preview_window = VeraLuxPreviewWindow()
        
        if self.preview_window.isVisible():
            self.preview_window.hide()
        else:
            if self.preview_proxy is None:
                self.prepare_preview_proxy()
            self.preview_window.show()
            self.preview_window.raise_()
            self.preview_window.activateWindow()
            self.update_preview_image()
            self.preview_window.fit_to_view()

    def prepare_preview_proxy(self):
        """Creates a high-quality downsampled version of the image for fast preview."""
        if self.linear_cache is None: return
        
        # Smart Downsample (Max 1600px long edge)
        # We need to maintain (C, H, W) format
        img = VeraLuxCore.normalize_input(self.linear_cache)
        if img.ndim == 3 and img.shape[0] != 3 and img.shape[2] == 3:
            img = img.transpose(2, 0, 1) # Ensure (C, H, W)
            
        h = img.shape[1] if img.ndim == 3 else img.shape[0]
        w = img.shape[2] if img.ndim == 3 else img.shape[1]
        
        scale = 1600 / max(h, w)
        if scale >= 1.0:
            self.preview_proxy = img # Use original if small
        else:
            # Simple slicing for speed
            step = int(1 / scale)
            if img.ndim == 3:
                self.preview_proxy = img[:, ::step, ::step]
            else:
                self.preview_proxy = img[::step, ::step]

    def trigger_preview_update(self):
        """Starts timer to update preview (Debouncing)."""
        if self.preview_window and self.preview_window.isVisible():
            self.debounce_timer.start()

    def update_preview_image(self):
        """Runs the math on the proxy and updates the window."""
        if self.preview_proxy is None: return
        
        # Gather params
        D = self.spin_d.value()
        b = self.spin_b.value()
        conv = self.spin_conv.value()
        
        # Use Unified Logic for params
        grip, shadow, linear_expansion = self.get_effective_params()
        
        adaptive = self.chk_adaptive.isChecked()
        
        ws = self.combo_profile.currentText()
        target_bg = self.spin_target.value()
        mode_str = "ready_to_use" if self.radio_ready.isChecked() else "scientific"
        
        # Run Core Math on Proxy    
        res = process_veralux_v6(
            self.preview_proxy.copy(),
            D, b, conv,
            ws, mode_str, target_bg,
            grip, shadow,
            linear_expansion,
            adaptive,
            None
        )

        # Convert to Display
        qimg = self.numpy_to_qimage(res)
        
        # Pass Diagnostic Metadata to Preview Window
        luma_w = SENSOR_PROFILES[ws]['weights']
        self.preview_window.set_image(qimg, res, self.is_source_saturated, mode_str, luma_w)

        # Update Main GUI Stats
        self.update_main_stats(res, mode_str)

    def update_main_stats(self, img_data, mode):
        # 1. Calculate basic clipping from result
        epsilon = 1e-7
        total = img_data.shape[1] * img_data.shape[2] if img_data.ndim == 3 else img_data.size
        
        if img_data.ndim == 3:
            mask_w = np.any(img_data >= (1.0 - epsilon), axis=0)
            white_c = np.count_nonzero(mask_w)
        else:
            white_c = np.count_nonzero(img_data >= (1.0 - epsilon))
            
        pct_white = (white_c / total) * 100.0
        
        # 2. Get Linear Expansion stats
        diag = getattr(VeraLuxCore, "_last_linear_expansion_diag", {})
        le_high = float(diag.get("pct_high", 0.0))
        
        # 3. Build String
        parts = []
        
        # White Clipping status
        w_col = "#ff4444" if pct_white >= 0.1 else ("#ffaa00" if pct_white >= 0.01 else "#666666")
        parts.append(f"Whites: <font color='{w_col}'>{pct_white:.2f}%</font>")
        
        # Linear Expansion status (Only in Scientific)
        if mode != "ready_to_use" and self.spin_exp.value() > 0:
            le_col = "#ff4444" if le_high >= 0.1 else ("#ffaa00" if le_high >= 0.01 else "#88aaff")
            parts.append(f" • LE Clamp: <font color='{le_col}'>{le_high:.3f}%</font>")
            
        self.lbl_stats.setText("".join(parts))

    def numpy_to_qimage(self, img_data):
        """Converts float32 (C,H,W) to QImage for display."""
        # Convert to (H,W,C) for QImage
        if img_data.ndim == 3:
            disp = img_data.transpose(1, 2, 0)
        else:
            disp = img_data
            
        # Clip and Scale to 8-bit (Processed data is linear 0-1)
        disp = np.clip(disp * 255.0, 0, 255).astype(np.uint8)     
        disp = np.flipud(disp)

        # Force contiguous memory
        disp = np.ascontiguousarray(disp)
        
        h, w = disp.shape[0], disp.shape[1]
        bytes_per_line = disp.strides[0]
        data_bytes = disp.data.tobytes()
        
        if disp.ndim == 2: # Mono
            qimg = QImage(data_bytes, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
        else: # RGB
            if disp.shape[2] == 3:
                qimg = QImage(data_bytes, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            else:
                return QImage()
                
        return qimg.copy()

    # --- STANDARD METHODS ---
    def handle_close_event(self, event):
        """Ensures the preview window closes when the main window closes and saves settings."""
        # Save Preferences (QSettings)
        self.settings.setValue("sensor_profile", self.combo_profile.currentText())
        self.settings.setValue("mode_ready", self.radio_ready.isChecked())
        self.settings.setValue("target_bg", self.spin_target.value())
        self.settings.setValue("linear_expansion", float(self.spin_exp.value()))
        
        if self.preview_window:
            self.preview_window.close()
        event.accept()

    def print_help_to_console(self):
        guide_lines = [
            "==========================================================================",
            "   VERALUX HYPERMETRIC STRETCH v{} - OPERATIONAL GUIDE".format(VERSION),
            "   Physics-Based Photometric Hyperbolic Stretch Engine",
            "==========================================================================",
            "",
            "OVERVIEW",
            "-----------------",
            "VeraLux provides a mathematically precise linear-to-nonlinear stretch",
            "designed to preserve photometric color ratios (Vector Color), avoiding",
            "the chromatic drift (Hue Shift) typical of scalar histogram transformations.",
            "",
            "[1] CRITICAL PREREQUISITES",
            "    • Input MUST be Linear (not yet stretched).",
            "    • Background gradients must have been removed.",
            "    • RGB input must be Color Calibrated (SPCC) within Siril.",
            "",
            "[2] THE DUAL PHILOSOPHY (MODES)",
            "    VeraLux offers two distinct pipelines. Understanding them is crucial:",
            "",
            "    A. Ready-to-Use (Aesthetic & Adaptive)",
            "       • GOAL: Create a finished, high-contrast image filling the 0-1 range.",
            "       • BEHAVIOR: Performs 'Adaptive Output Scaling'. It automatically",
            "         expands the data to set the black point and white point.",
            "       • FEATURES: Unified Slider for ease of use; Soft-Clipping for stars.",
            "",
            "    B. Scientific Mode (Preserve & Raw)",
            "       • GOAL: Mathematical fidelity. Deterministic output.",
            "       • BEHAVIOR: No auto-scaling. The output is the pure result of the",
            "         math equation. If Log D is low, the image remains dark.",
            "       • FEATURES: Full manual control; Hard-Clipping only (No soft knee).",
            "         Ideal for mosaics or as a base for manual Tone Mapping.",
            "",
            "[3] RECOMMENDED WORKFLOWS & PARAMETER GUIDE",
            "",
            "    Workflow A: READY-TO-USE (The 'Finished Look')",
            "    1. SETUP: Select 'Ready-to-Use' and Sensor Profile.",
            "    2. SOLVE: Set 'Target Bg' (Default 0.20) and click 'Auto-Calc'.",
            "    3. REFINE (Live Preview):",
            "       • Too Dark/Bright? → Adjust 'Log D'.",
            "       • Stars too harsh? → Increase 'Protect b' (Higher = Sharper/Smaller).",
            "       • Balancing Noise vs Highlights? → Use 'Color Strategy':",
            "         - Move LEFT (<0): Cleans background noise (Increases Shadow Conv).",
            "         - Move RIGHT (>0): Saves star colors (Decreases Color Grip).",
            "    4. PROCESS: The result is contrast-optimized and ready for export.",
            "",
            "    Workflow B: SCIENTIFIC (The 'Data Foundation')",
            "    1. SETUP: Select 'Scientific' and Sensor Profile.",
            "    2. SOLVE: Click 'Auto-Calc'. This gives a mathematical safety baseline.",
            "    3. MANUAL TUNING (Advanced Control):",
            "       • Intensity: Increase 'Log D' to define the stretch curvature.",
            "       • Linear Expansion: Linearly realigns the occupied data range toward [0–1].",
            "         Expands or rebalances contrast depending on the data distribution.",
            "         Uses percentile-based bounds with diagnostic warnings to avoid hidden",
            "         clipping. Keep an eye on histograms statistics in the Preview window.",
            "       • Star Cores: Reduce 'Color Grip' (<1.0) to blend with scalar stretch",
            "         (helps if vector preservation creates hard edges on stars).",
            "       • Deep Noise: Increase 'Shadow Convergence' to damp chromatic noise.",
            "    4. PROCESS: The result is mathematically precise.",
            "    5. POST: Use this output as a high-fidelity base for manual",
            "       Tone Mapping (Curves, GHS, Local Histograms).",
            "",
            "[4] CALIBRATION & ANCHORING",
            "    • Adaptive Anchor: Analyzes histogram shape to find the true signal",
            "      start. Maximizes contrast but requires flat frames (no gradients).",
            "    • Auto-Calculate: Finds the optimal 'Log D' to place the background",
            "      at the Target Level. In 'Ready-to-Use' mode, it also simulates",
            "      the scaling pipeline to prevent black clipping.",
            "",
            "[5] PHYSICS TUNING",
            "    • Stretch (Log D): The intensity. Higher = Brighter.",
            "    • Protect b: Highlight protection. Higher (>6) = Sharper stars.",
            "    • Color Convergence: Controls the 'White Point' physics.",
            "",
            "[6] SATURATION & CHROMATIC INTEGRITY",
            "    While standard stretches often distort color ratios near saturation",
            "    (Hue Shift), VeraLux prioritizes strict Vector Color preservation.",
            "    However, at extreme highlights (stars), physical chromatic information",
            "    is naturally lost to pure Luminance.",
            "    VeraLux models this via 'Color Convergence', forcing saturated pixels",
            "    to White (1.0). Seeing 'Max: 65535' confirms the engine respects",
            "    the physical limit where color data gives way to pure energy.",
            "",
            "[7] LIVE HISTOGRAM & CLIPPING",
            "    • ORANGE BAR (> 0.01%): Marginal clipping (Bloated stars).",
            "    • RED BAR (> 0.1%): Significant clipping (Loss of structure).",
            "",
            "    INTERPRETING CLIPPING BY MODE:",
            "    • IN 'READY-TO-USE': Black Clipping is INTENTIONAL. The engine",
            "      sacrifices the Gaussian noise tail to ensure deep, true blacks.",
            "    • IN 'SCIENTIFIC': White Clipping is MATHEMATICAL TRUTH. Since",
            "      Soft-Clip is disabled, any Color Vector exceeding 1.0 is hard-cut",
            "      to preserve linearity in the non-saturated data.",
            "",
            "[8] TROUBLESHOOTING",
            "    • 'Scientific mode is too dark': This is expected (See Workflow B).",
            "    • 'My stars are too harsh': Increase 'Protect b' value.",
            "    • 'Linear Expansion darkens my image instead of brightening it':",
            "      This is expected. Linear Expansion is not a brightness control.",
            "      It linearly realigns the occupied data range to [0,1].",
            "      If your data already spans the upper range, the median may move down.",
            "",
            "Support & Info: info@veralux.space",
            "=========================================================================="
        ]
        
        try:
            for line in guide_lines:
                msg = line if line.strip() else " "
                self.siril.log(msg)
            self.status.setText("Full Guide printed to Console.")
        except Exception:
            print("\n".join(guide_lines))
            self.status.setText("Guide printed to standard output.")

    def center_window(self):
        screen = self.app.primaryScreen()
        if screen:
            self.window.move(self.window.frameGeometry().topLeft())
            frame_geo = self.window.frameGeometry()
            frame_geo.moveCenter(screen.availableGeometry().center())
            self.window.move(frame_geo.topLeft())

    def toggle_ontop(self, checked):
        pos = self.window.pos()
        if checked: self.window.setWindowFlags(self.window.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        else: self.window.setWindowFlags(self.window.windowFlags() & ~Qt.WindowType.WindowStaysOnTopHint)
        self.window.show(); self.window.move(pos)

    def update_profile_info(self, profile_name):
        if profile_name in SENSOR_PROFILES:
            profile = SENSOR_PROFILES[profile_name]
            r, g, b = profile['weights']
            self.label_profile_info.setText(f"{profile['description']} (R:{r:.2f} G:{g:.2f} B:{b:.2f})")

    def set_defaults(self):
        self.spin_d.setValue(2.0); self.spin_b.setValue(6.0); self.spin_target.setValue(0.20)
        self.spin_conv.setValue(3.5); self.spin_grip.setValue(1.0); self.spin_shadow.setValue(0.0)
        self.slide_unified.setValue(0)
        self.spin_exp.setValue(0.0)
        self.slide_exp.setValue(0)
        self.chk_adaptive.setChecked(True)
        self.combo_profile.setCurrentText(DEFAULT_PROFILE)
        self.radio_ready.setChecked(True)

    def _safe_get_image_pixeldata_locked(self):
        """
        Robust pixel retrieval. Tries Shared Memory (Fast) first.
        Falls back to FFit (Slow/Safe) if SHM fails (e.g., embedded ICC).
        """
        # 1. Try Shared Memory (Fastest)
        try:
            data = self.siril.get_image_pixeldata()
            if data is not None:
                return data
        except Exception:
            pass

        # 2. Fallback: Full Fit Retrieval (Slower but robust)
        try:
            self.siril.log("VeraLux: SHM retrieval failed (embedded ICC?), attempting robust fallback...", color=LogColor.ORANGE)
            fit_data = self.siril.get_image(with_pixels=True)
            if fit_data and hasattr(fit_data, "data"):
                # Convert list of lists to numpy array
                arr = np.array(fit_data.data, dtype=np.float32)
                
                # Reshape flattened arrays to (Channels, Height, Width)
                if arr.ndim == 2 and hasattr(fit_data, "width") and hasattr(fit_data, "height"):
                    h = fit_data.height
                    w = fit_data.width
                    c = arr.shape[0]
                    # Ensure dimensions match before reshaping
                    if arr.size == c * h * w:
                        arr = arr.reshape(c, h, w)
                        
                return arr
        except Exception as e:
            print(f"Fallback retrieval failed: {e}")
        
        return None

    def cache_input(self):
        try:
            if not self.siril.connected: self.siril.connect()
            self.status.setText("Caching Linear Data...")
            self.app.processEvents()
            
            with self.siril.image_lock(): 
                # Use robust retrieval method instead of direct call
                self.linear_cache = self._safe_get_image_pixeldata_locked()
                
            if self.linear_cache is None: 
                self.status.setText("Error: No pixel data (Check ICC/Image).")
            else:
                self.status.setText("Input Cached.")
                self.siril.log("VeraLux: Input Cached.", color=LogColor.GREEN)
                
                # --- SOURCE SATURATION CHECK (Tolerance 99.9%) ---
                # Normalize exactly like the processor to compare apples to apples
                # 65534.9 / 65535 = 0.999998 -> Saturated
                check_norm = VeraLuxCore.normalize_input(self.linear_cache)
                self.is_source_saturated = np.max(check_norm) > 0.999
                
                self.preview_proxy = None # Invalidate preview cache
                if self.preview_window and self.preview_window.isVisible():
                    self.prepare_preview_proxy()
                    self.update_preview_image()
        except Exception as e: self.status.setText("Connection Error."); print(e)

    def run_solver(self):
        if self.linear_cache is None: return
        if hasattr(self, 'solver') and self.solver and self.solver.isRunning(): return
        self.status.setText("Solving..."); self.btn_auto.setEnabled(False); self.progress.setRange(0, 0)
        tgt = self.spin_target.value(); b = self.spin_b.value(); ws = self.combo_profile.currentText()
        luma = SENSOR_PROFILES[ws]['weights']
        adaptive = self.chk_adaptive.isChecked()
        mode = "ready_to_use" if self.radio_ready.isChecked() else "scientific"
        
        self.solver = AutoSolverThread(self.linear_cache, tgt, b, luma, adaptive, mode)
        self.solver.result_ready.connect(self.apply_solver_result)
        self.solver.start()
        
    def apply_solver_result(self, log_d):
        self.spin_d.setValue(log_d); self.progress.setRange(0, 100); self.progress.setValue(100)
        self.btn_auto.setEnabled(True); ws = self.combo_profile.currentText()
        self.status.setText(f"Solved: Log D = {log_d:.2f}")
        self.siril.log(f"VeraLux Solver: Optimal Log D={log_d:.2f} [{ws}]", color=LogColor.GREEN)

    def run_process(self):
        if self.linear_cache is None: return
        if hasattr(self, 'worker') and self.worker and self.worker.isRunning(): return
        try: self.siril.undo_save_state(f"VeraLux v{VERSION} Stretch")
        except: pass
        D = self.spin_d.value(); b = self.spin_b.value(); conv = self.spin_conv.value()
        ws = self.combo_profile.currentText(); t_bg = self.spin_target.value()
        grip, shadow, linear_expansion = self.get_effective_params() # Use unified logic
        adaptive = self.chk_adaptive.isChecked()
        mode = "ready_to_use" if self.radio_ready.isChecked() else "scientific"
        
        self.status.setText("Processing..."); self.progress.setRange(0, 0)
        img_copy = self.linear_cache.copy()
        
        self.worker = ProcessingThread(img_copy, D, b, conv, ws, mode, t_bg, grip, shadow, linear_expansion, adaptive)
        self.worker.progress.connect(self.status.setText)
        self.worker.finished.connect(self.finish_process)
        self.worker.start()
        
    def finish_process(self, result_img):
        self.progress.setRange(0, 100); self.progress.setValue(100); self.status.setText("Complete.")
        mode = "Ready-to-Use" if self.radio_ready.isChecked() else "Scientific"
        ws = self.combo_profile.currentText()
        
        if result_img is not None:
            with self.siril.image_lock(): 
                self.siril.set_image_pixeldata(result_img)
            
            try:
                self.siril.cmd("icc_remove")
            except Exception:
                pass
            
            self.siril.cmd("stat")
            try: 
                self.siril.cmd("visu 0 65535") 
            except: 
                pass
            
            self.siril.log(f"VeraLux v{VERSION}: {mode} mode applied [{ws}]", color=LogColor.GREEN)

def main():
    try:
        app = QApplication.instance()
        if not app:
            app = QApplication(sys.argv)
        siril = s.SirilInterface()
        gui = VeraLuxInterface(siril, app)
        app.exec()
    except Exception as e:
        print(f"Error starting VeraLux: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()