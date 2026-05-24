##############################################
# VeraLux — Starting Point
# Interactive Workflow Guide & Manual
# Author: Riccardo Paterniti (2025)
# Contact: info@veralux.space
##############################################
# (c) 2025 Riccardo Paterniti
# VeraLux — Starting Point
# SPDX-License-Identifier: GPL-3.0-or-later
# Version 1.0.0

# Credits / Origin
# ----------------
# • Concept & Architecture: VeraLux Workflow Framework
# • UI Framework: PyQt6 (Dark Themed)
# • Integration Target: Siril Astrophotography Software

"""
Overview
--------
VeraLux Starting Point is an interactive workflow guide and visual manual
for the VeraLux ecosystem.

It is designed to provide a structured, problem-oriented entry point into
physically faithful astrophotography processing, guiding the user through
the VeraLux toolchain from linear data preparation to final color grading.

Rather than acting as a processing engine itself, Starting Point serves as
a cognitive and operational map: each module is contextualized within the
overall workflow, with clear intent, constraints, and best-use scenarios.

This tool is intended to:
• Help users choose the *right* VeraLux tool at the *right* stage
• Explain why each tool exists and what physical problem it solves
• Act as a living, navigable manual tightly coupled to the actual codebase

Starting Point is part of the VeraLux family —
Python utilities focused on physically faithful astrophotography workflows.
"""

import sys
import webbrowser
import re
import base64
import urllib.request
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QLabel, QPushButton, QGroupBox, QScrollArea,
                             QCheckBox, QTextBrowser, QLineEdit, QToolButton, QFrame,
                             QInputDialog, QMessageBox)
from PyQt6.QtCore import Qt, pyqtSignal, QEvent, QTimer

from PyQt6.QtGui import QDesktopServices

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

VERSION = "1.0.0"

# ---------------------
#  THEME & STYLING
# ---------------------

DARK_STYLESHEET = """
QWidget { 
    background-color: #2b2b2b; 
    color: #e0e0e0; 
    font-size: 10pt; /* System Font */
}

QToolTip { background-color: #333333; color: #ffffff; border: 1px solid #88aaff; }

/* --- NAVIGATION GROUP BOX --- */
QGroupBox {
    border: 1px solid #444444;
    margin-top: 5px;
    font-weight: bold;
    border-radius: 4px;
    padding-top: 12px;
    background-color: #333333;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 3px;
    color: #88aaff;
}

QLabel { color: #cccccc; }

/* --- CHECKBOX --- */
QCheckBox { spacing: 5px; color: #cccccc; }
QCheckBox::indicator {
    width: 14px; height: 14px;
    border: 1px solid #666666;
    background: #3c3c3c;
    border-radius: 3px;
}
QCheckBox::indicator:checked {
    background-color: #285299;
    border: 1px solid #88aaff;
}

/* --- MENU BUTTONS --- */
QPushButton.MenuButton { 
    background-color: transparent; 
    color: #cccccc; 
    border: none; 
    border-left: 3px solid transparent;
    border-radius: 0px; 
    padding: 8px 15px; 
    font-weight: 600; 
    font-size: 11pt; 
    text-align: left; 
}
QPushButton.MenuButton:hover { 
    background-color: #3a3a3a; 
    color: #ffffff; 
}

/* --- FOOTER BUTTONS --- */
QPushButton#CloseButton { 
    background-color: #5a2a2a; 
    color: #dddddd; 
    border: 1px solid #804040; 
    border-radius: 4px; 
    padding: 6px 20px;
    font-weight: bold;
}
QPushButton#CloseButton:hover { 
    background-color: #7a3a3a; 
    color: #ffffff;
    border-color: #905050;
}

QPushButton#CoffeeButton { background-color: transparent; border: none; font-size: 16pt; margin-right: 0px; }
QPushButton#CoffeeButton:hover { background-color: rgba(255,255,255,20); border-radius: 4px; }

QPushButton#HomeButton { 
    background-color: rgba(136,170,255,30); 
    border: 1px solid #88aaff; 
    border-radius: 4px;
    font-size: 13pt; 
    padding: 4px 10px; 
    margin-right: 6px;
    color: #88aaff;
    font-weight: bold;
}
QPushButton#HomeButton:hover { 
    background-color: rgba(136,170,255,60); 
    color: #ffffff;
}

/* --- SCROLL AREA --- */
QScrollArea { border: none; background-color: transparent; }
QScrollBar:vertical { 
    background: transparent; width: 8px; margin: 0px; 
}
QScrollBar::handle:vertical { 
    background: #444444; min-height: 30px; border-radius: 4px; 
}
QScrollBar::handle:vertical:hover { background: #666666; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }

/* --- DETAIL CARD --- */
QTextBrowser {
    background-color: #1e1e1e;
    border: 1px solid #333333;
    border-radius: 8px;
    padding: 25px;
    color: #e0e0e0;
}
"""

# --- EASTER EGG: CHAOS MODE STYLESHEET ---
CHAOS_STYLESHEET = """
QWidget { 
    background-color: #FFFF00; /* Warning Yellow */
    color: #FF00FF; /* Magenta */
    font-family: "Comic Sans MS";
    font-size: 12pt;
    font-weight: bold;
}

QGroupBox {
    border: 4px dashed #FF0000;
    margin-top: 15px; 
    padding-top: 35px;
    padding-left: 20px;
    padding-right: 20px;
    padding-bottom: 15px;
    background-color: #00FF00;
    color: #000000;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top center;
    padding: 5px 10px;
    color: #FFFFFF;
    background-color: #0000FF;
    border-radius: 5px;
}

QPushButton.MenuButton {
    background-color: #00FFFF;
    color: #000000;
    border: 3px solid #000000;
    border-radius: 15px;
    margin: 5px 20px;
    padding: 10px;
    text-align: center;
}

QPushButton.MenuButton:hover {
    background-color: #FF00FF;
    color: #FFFF00;
    border-style: dotted;
}

/* Footer buttons override in Chaos Mode */
QPushButton#CoffeeButton, QPushButton#HomeButton {
    background-color: #FF00FF;
    border: 2px solid black;
    border-radius: 50%;
    min-width: 40px;
    min-height: 40px;
}

QTextBrowser {
    background-color: #FFFFFF;
    border: 8px ridge #FF00FF;
    color: #000000;
}

QLabel { color: #FF0000; }
QLineEdit { background-color: #FFFFFF; color: #000000; border: 2px solid #000000; }
"""

# =============================================================================
#  CUSTOM WIDGET: VERALUX MODERN SEARCH BAR
# =============================================================================

class VeraLuxSearchBar(QFrame):
    textChanged = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(40)
        self.setObjectName("SearchFrame")
        self.apply_normal_style()
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 8, 0)
        layout.setSpacing(5)
        
        self.icon_lbl = QLabel("🔍")
        self.icon_lbl.setStyleSheet("border: none; background: transparent; color: #666666; font-size: 11pt;")
        layout.addWidget(self.icon_lbl)
        
        self.input = QLineEdit()
        self.input.setPlaceholderText("Problem Solver (e.g. 'noise')...")
        self.input.setStyleSheet("""
            QLineEdit {
                border: none;
                background: transparent;
                color: #ffffff;
                font-size: 10pt;
                padding-top: 2px;
            }
            QLineEdit::placeholder { color: #555555; font-style: italic; }
        """)
        self.input.textChanged.connect(self.on_text_changed)
        self.input.installEventFilter(self)
        layout.addWidget(self.input)
        
        self.btn_clear = QToolButton()
        self.btn_clear.setText("✕")
        self.btn_clear.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_clear.setFixedSize(24, 24)
        self.btn_clear.setStyleSheet("""
            QToolButton {
                border: none;
                background: #333333;
                border-radius: 12px;
                color: #aaaaaa;
                font-weight: bold;
                font-size: 8pt;
                padding-bottom: 2px;
            }
            QToolButton:hover { background: #555555; color: white; }
            QToolButton:pressed { background: #285299; }
        """)
        self.btn_clear.clicked.connect(self.clear_text)
        self.btn_clear.hide()
        layout.addWidget(self.btn_clear)

    def apply_normal_style(self):
        self.setStyleSheet("""
            QFrame#SearchFrame { 
                background-color: #181818; 
                border: 1px solid #444444; 
                border-radius: 20px; 
            }
            QFrame#SearchFrame:hover { 
                border-color: #666666; 
                background-color: #202020; 
            }
            font-family: "Comic Sans MS";
            font-size: 12pt;
            font-weight: bold;
        """)

    def on_text_changed(self, text):
        if text: self.btn_clear.show()
        else: self.btn_clear.hide()
        self.textChanged.emit(text)

    def clear_text(self):
        self.input.clear()
        self.input.setFocus()

    def eventFilter(self, obj, event):
        if obj == self.input:
            if event.type() == QEvent.Type.FocusIn:
                self.setStyleSheet("""
                    QFrame#SearchFrame { 
                        background-color: #181818; 
                        border: 1px solid #88aaff; 
                        border-radius: 20px; 
                    }
                    font-family: "Comic Sans MS";
                    font-size: 12pt;
                    font-weight: bold;
                """)
                self.icon_lbl.setStyleSheet("border: none; background: transparent; color: #88aaff; font-size: 11pt; font-family: 'Comic Sans MS';")
                
            elif event.type() == QEvent.Type.FocusOut:
                self.setStyleSheet("""
                    QFrame#SearchFrame { 
                        background-color: #181818; 
                        border: 1px solid #444444; 
                        border-radius: 20px; 
                    }
                    QFrame#SearchFrame:hover { 
                        border-color: #666666; 
                        background-color: #202020; 
                    }
                    font-family: "Comic Sans MS";
                    font-size: 12pt;
                    font-weight: bold;
                """)
                self.icon_lbl.setStyleSheet("border: none; background: transparent; color: #666666; font-size: 11pt; font-family: 'Comic Sans MS';")
        return super().eventFilter(obj, event)

# =============================================================================
#  KNOWLEDGE BASE
# =============================================================================

BADGE_STYLES = {
    "LINEAR": {"mode": "dot", "accent": "#5f7fa3"},
    "NON-LINEAR": {"mode": "hollow", "accent": "#a35fa3"},
    "STRETCH": {"mode": "dot", "accent": "#cda434"},
    "PHYSICS": {"mode": "dot", "accent": "#88aaff"},
    "COLOR": {"mode": "dot", "accent": "#7a8a6a"},
    "STARS": {"mode": "star", "accent": "#aaaaaa"},
}

# --- CHAOS MODE DATA MAPPING ---
CHAOS_TOOLS = {
    "nox": {
        "name": "TOXIC",
        "tagline": "The Gradient INJECTOR",
        "desc": """
        <h1>TOXIC</h1>
        <p><b>Overview:</b> Instead of removing gradients, this tool adds complex vignetting and random light pollution to your best images.</p>
        <p><b>Usage:</b> Click "Process" repeatedly until the image is completely green.</p>
        <p><b>Feature:</b> "Dark Hole Creator" creates artificial black holes in your nebulas.</p>
        """
    },
    "silentium": {
        "name": "LOUDNESS",
        "tagline": "Maximum Grain Engine",
        "desc": """
        <h1>LOUDNESS</h1>
        <p><b>Overview:</b> Why have a smooth background when you can have texture? Adds salt & pepper noise to every pixel.</p>
        <p><b>Feature:</b> "Detail Guard" actually erases nebulae.</p>
        <p><b>Shadow Authority:</b> Ensures shadows are as bright as stars.</p>
        """
    },
    "alchemy": {
        "name": "POTIONS",
        "tagline": "Random Color Generator",
        "desc": """
        <h1>POTIONS</h1>
        <p><b>Overview:</b> Randomly mixes R, G, and B channels based on the current weather.</p>
        <p><b>Result:</b> Guaranteed psychedelic art. Who needs correct colors?</p>
        <p><b>Mixing Matrix:</b> All sliders are disconnected and do nothing.</p>
        """
    },
    "hms": {
        "name": "HYPERMESS",
        "tagline": "The Data Destroyer",
        "desc": """
        <h1>HYPERMESS</h1>
        <p><b>Overview:</b> Stretches the histogram until it snaps. Zero dynamic range, maximum clipping.</p>
        <p><b>True Color?</b> No. We prefer Neon.</p>
        <p><b>Sensor Profile:</b> "Webcam 2003".</p>
        """
    },
    "starcomposer": {
        "name": "STARKILLER",
        "tagline": "Square Star Generator",
        "desc": """
        <h1>STARKILLER</h1>
        <p><b>Overview:</b> Replaces round stars with Minecraft blocks.</p>
        <p><b>Bloat:</b> Set to 100% for maximum visibility.</p>
        <p><b>Halos:</b> Adds purple fringes to every star, just because.</p>
        """
    },
    "curves": {
        "name": "WIGGLES",
        "tagline": "S-Curve Abuse",
        "desc": """
        <h1>WIGGLES</h1>
        <p><b>Overview:</b> Just add points randomly and drag them up and down.</p>
        <p><b>Spline:</b> Wobbly.</p>
        <p><b>Clipping:</b> Encouraged.</p>
        """
    },
    "revela": {
        "name": "BLURRA",
        "tagline": "Soft Focus Tool",
        "desc": """
        <h1>BLURRA</h1>
        <p><b>Overview:</b> Reduces detail to make your image look like it was taken through a soup bowl.</p>
        <p><b>Texture:</b> Removes it completely.</p>
        """
    },
    "vectra": {
        "name": "DISCO",
        "tagline": "Psychedelic Shift",
        "desc": """
        <h1>DISCO</h1>
        <p><b>Overview:</b> Rotates hue constantly. Don't look at it too long.</p>
        <p><b>Teal & Orange?</b> No. Pink & Lime.</p>
        """
    }
}

WORKFLOW_DATA = [
    {
        "phase": "1. LINEAR PHASE",
        "desc": "Raw Data Operations. Before stretching.",
        "color": "#00cccc", 
        "tools": [
            {
                "id": "nox",
                "name": "Nox",
                "tagline": "The Gradient Killer",
                "badges": ["LINEAR", "PHYSICS"],
                "keywords": "gradient vignetting light pollution background extraction linear remove",
                "desc_html": """
                <h2 style='color:#ffffff; margin-top:0px;'>Overview</h2>
                <p>
                <b>Nox</b> is a physically-faithful gradient reduction engine designed to model and subtract <b>additive</b>
                light pollution and vignetting while rigorously preserving faint signal.
                </p>
                <p>
                It operates on the <b>Zenith</b> architecture: instead of relying on manual sample points,
                it simulates a virtual elastic membrane draped over the image data. By using topological analysis
                on pure <b>linear</b> data, Nox separates low-frequency gradients (background) from structured signal
                (nebulae, galaxies), enabling a mathematically <i>subtraction-safe</i> model.
                </p>

                <h2 style='color:#ffffff; margin-top:22px;'>Design goals</h2>
                <ul>
                  <li>Operate exclusively on <b>Linear Data</b> to preserve photometric integrity</li>
                  <li>Eliminate manual sample placement via semantic analysis</li>
                  <li>Distinguish faint nebulosity (IFN) from background noise using macro-scale SNR</li>
                  <li>Enforce geometric constraints to prevent over-fitting (<i>"dark holes"</i>)</li>
                  <li>Provide a unified workflow for both stars (PSF) and dust (topology)</li>
                </ul>

                <h2 style='color:#ffffff; margin-top:22px;'>Core features</h2>

                <p><b>Zenith membrane engine</b></p>
                <ul>
                  <li><b>Geometric low-pass:</b> the solver operates on a rigid low-pass grid resolution, physically constraining
                      model flexibility. This prevents collapse into medium-scale structures (e.g. nebula cores), forcing the
                      membrane to glide over signal while modeling broad gradients.</li>
                  <li><b>Bicubic reconstruction:</b> artifact-resistant upscaling of the background model, reducing ringing/undershoot
                      around bright stars.</li>
                </ul>

                <p><b>Multiscale topological protection</b></p>
                <ul>
                  <li><b>Micro-scale:</b> real-time PSF analysis (FWHM) identifies and protects point sources (stars).</li>
                  <li><b>Macro-scale ("Deep Eye"):</b> dual-resolution downsampled variance analysis detects ultra-faint structures
                      (H-alpha, IFN) that are statistically invisible at pixel level.</li>
                  <li><b>Proximity principle:</b> intelligent morphological dilation respects extended halos around protected regions.</li>
                </ul>

                <p><b>Linear physics auto-tune</b></p>
                <ul>
                  <li><b>BVI logic:</b> automatically computes stiffness from the Background Variability Index (noise vs structure ratio).</li>
                  <li><b>Density-adaptive aggression:</b> dynamically caps subtraction strength based on signal coverage percentage.</li>
                </ul>

                <p><b>Interactive masking suite</b></p>
                <ul>
                  <li>Optional manual masking engine with brush, lasso, and eraser tools</li>
                  <li>Real-time overlay plus smart zoom/pan for inspection of linear data</li>
                </ul>

                <h2 style='color:#ffffff; margin-top:22px;'>Usage</h2>
                <ol>
                  <li><b>Pre-requisite:</b> image must be <b>Linear</b> (RGB or Mono). Crop stacking artifacts first.</li>
                  <li><b>Setup:</b> Auto-masking is <b>ON</b> by default (recommended).</li>
                  <li><b>Calibration:</b> click <b>AUTO-CALCULATE</b> to analyze PSF, density, and SNR and set optimal stiffness/aggression.</li>
                  <li><b>Refine (optional):</b> use paint/lasso for complex objects; lower aggression (&lt;50%) for safety, higher (&gt;70%) for dark skies.</li>
                  <li><b>Process:</b> click <b>PROCESS</b>.</li>
                </ol>

                <h2 style='color:#ffffff; margin-top:22px;'>Inputs &amp; outputs</h2>
                <p><b>Input:</b> Linear FITS/TIFF (RGB/Mono), 16/32-bit int or float.<br>
                <b>Output:</b> Background-neutralized 32-bit float FITS.</p>

                <div style="margin-top:22px; padding:0px; border:none; background:transparent;">
                  <table width="100%" cellspacing="0" cellpadding="0" style="border-collapse:collapse;">
                    <tr>
                      <td align="center">
                        <a href="https://youtu.be/eK9VU1d-VP0?si=8Fs3NZGuyGnG0JY-" style="text-decoration:none;">
                          <img src="https://img.youtube.com/vi/eK9VU1d-VP0/maxresdefault.jpg"
                               width="640" height="360"
                               style="display:block; border-radius:8px; border:1px solid #2a2a2a;">
                        </a>
                      </td>
                    </tr>
                  </table>
                </div>
                """
            },
            {
                "id": "silentium",
                "name": "Silentium",
                "tagline": "Linear Noise Suppression",
                "badges": ["LINEAR", "PHYSICS"],
                "keywords": "noise grain denoise reduction linear background smooth",
                "desc_html": """
                <h2 style='color:#ffffff; margin-top:0px;'>Overview</h2>
                <p>
                <b>Silentium</b> is a linear-phase noise suppression engine designed to separate stochastic noise
                from physical signal without compromising stellar geometry or fine nebula details.
                </p>
                <p>
                Silentium establishes a new paradigm in linear processing by decoupling the treatment
                of low-signal areas (background) from high-signal structures. Through its <b>Shadow Authority</b>
                logic and its <b>physics-aware</b> architecture, it applies mathematically aggressive cleaning
                to the deep-space floor while creating an impervious protective shell around stars
                and nebulosity, preserving the original photometric integrity of the data.
                </p>

                <h2 style='color:#ffffff; margin-top:22px;'>Core architecture</h2>
                <ul>
                  <li><b>Stationary Wavelet Transform (SWT):</b> Undecimated multiscale decomposition via Daubechies (db2) wavelets.
                      This shift-invariant approach ensures precise frequency separation with zero phase shift and no aliasing,
                      providing superior texture quality compared to standard DWT.</li>
                  <li><b>Photometric Signal Gating:</b> a statistical <i>Signal Probability Map</i> derived from linear-data analysis
                      (Median + Sigma thresholds) strictly decouples the image into distinct operating domains, preventing
                      cross-contamination between background smoothing and structural preservation.</li>
                  <li><b>Shadow Authority &amp; Exclusion Gate:</b> strict winner-take-all logic for background cleaning.
                      The Shadow Smoothness engine includes a hard photometric cutoff that forces denoising aggression to zero
                      as soon as any signal structure is detected, preventing highlight blurring regardless of protection settings.</li>
                  <li><b>PSF-compensated Structural Guard:</b> Detail Guard combines linear signal probability with morphological dilation
                      and seeing compensation. Protection scales with local FWHM, preventing erosion of filaments that appear optically
                      soft due to atmospheric blur.</li>
                  <li><b>Seeing-adaptive thresholding:</b> wavelet thresholds are spatially modulated by the local FWHM map. The algorithm
                      increases aggression in blurred areas (poor seeing) where fine detail is physically absent, and preserves
                      micro-contrast in sharper zones.</li>
                  <li><b>PSF-aware elliptical masking:</b> based on Siril <code>findstar</code> data, Silentium generates rotated elliptical masks
                      from real star geometry (angle, major/minor axes). This handles coma and tracking errors, ensuring zero profile erosion.</li>
                  <li><b>Loupe UX:</b> Global View is static (Fit-only). Clicking zooms into 1:1 <i>Loupe Mode</i> for pixel inspection.
                      Dragging pans the Loupe.</li>
                  <li><b>Shadow Report:</b> statistical breakdown of noise reduction, SNR gain, and pedestal conservation upon completion.</li>
                </ul>

                <h2 style='color:#ffffff; margin-top:22px;'>Usage</h2>
                <ol>
                  <li><b>Pre-requisite:</b> image must be <b>Linear</b> (before stretching).</li>
                  <li><b>Setup:</b>
                    <ul>
                      <li>Enable <b>Use findstar</b> to leverage PSF protection.</li>
                      <li>Select <b>Adaptive Noise Model</b> (default ON).</li>
                    </ul>
                  </li>
                  <li><b>Luminance calibration:</b>
                    <ul>
                      <li><b>Noise Intensity:</b> global threshold for general grain reduction.</li>
                      <li><b>Detail Guard:</b> protects faint structures. Increasing restores micro-contrast without reintroducing background noise.</li>
                      <li><b>Shadow Smoothness:</b> targeted, aggressive cleaning for the background.</li>
                    </ul>
                  </li>
                  <li><b>Preview interaction:</b>
                    <ul>
                      <li><b>Global View:</b> click to inspect an area at 1:1.</li>
                      <li><b>Loupe Mode (1:1):</b> drag to pan. Click (without dragging) to return to Global View.</li>
                    </ul>
                  </li>
                  <li><b>Process:</b> click <b>PROCESS</b> to apply to the full-resolution image.</li>
                </ol>

                <h2 style='color:#ffffff; margin-top:22px;'>Inputs &amp; outputs</h2>
                <p><b>Input:</b> Linear FITS/TIFF (RGB/Mono), 16/32-bit int or float.<br>
                <b>Output:</b> Denoised Linear 32-bit float FITS.</p>

                <div style="margin-top:22px; padding:0px; border:none; background:transparent;">
                  <table width="100%" cellspacing="0" cellpadding="0" style="border-collapse:collapse;">
                    <tr>
                      <td align="center">
                        <a href="https://youtu.be/N4CUQA2X-cE?si=DPrdb1Ikf0BOpfTU" style="text-decoration:none;">
                          <img src="https://img.youtube.com/vi/N4CUQA2X-cE/maxresdefault.jpg"
                               width="640" height="360"
                               style="display:block; border-radius:8px; border:1px solid #2a2a2a;">
                        </a>
                      </td>
                    </tr>
                  </table>
                </div>
                """
            },
            {
                "id": "alchemy",
                "name": "Alchemy",
                "tagline": "Narrowband Normalization",
                "badges": ["LINEAR", "COLOR"],
                "keywords": "color cast green red mixing hoo sho narrowband normalization linear",
                "desc_html": """
                <h2 style='color:#ffffff; margin-top:0px;'>Overview</h2>
                <p>
                <b>VeraLux Alchemy</b> is a linear-phase workstation designed to normalize, unmix,
                and mix narrowband signals (e.g., from OSC dual-band filters) prior to stretching.
                </p>
                <p>
                It addresses the common issue of red-dominated HOO/SHO composites by:
                </p>
                <ul>
                  <li>statistically aligning weak signals (OIII/SII) to the strong signal (H-alpha)</li>
                  <li>optionally separating Ha and OIII using a sensor-based crosstalk compensation model</li>
                  <li>providing a real-time linear mixing matrix to define the final color palette</li>
                </ul>
                <p>
                All operations are performed strictly in the <b>linear</b> domain.
                </p>

                <h2 style='color:#ffffff; margin-top:22px;'>Key features</h2>
                <ul>
                  <li><b>Linear-phase processing:</b> output remains linear and stretch-ready.</li>
                  <li><b>Robust normalization:</b> aligns background (Median) and signal strength (MAD-based) of weak channels
                      to the reference channel using robust statistics.</li>
                  <li><b>Quantum Unmixing (optional):</b> dual-band Ha/OIII separation using sensor-specific crosstalk compensation
                      coefficients derived from DBXtract. This is a physical signal model, not a cosmetic correction.</li>
                  <li><b>Real-time palette mixer:</b> define HOO, pseudo-SHO, or custom blends interactively.</li>
                  <li><b>WYSIWYG preview:</b> uses Siril's AutoStretch (MTF) engine to visualize the linear result exactly as it will
                      appear after stretching.</li>
                </ul>

                <h2 style='color:#ffffff; margin-top:22px;'>Usage</h2>
                <ol>
                  <li><b>Load data:</b> import a <b>Linear RGB</b> image (e.g., OSC dual-band result).</li>
                  <li><b>(Optional) Quantum Unmixing:</b> select the sensor profile and enable Quantum Unmixing to separate Ha and OIII
                      before normalization.</li>
                  <li><b>Normalize:</b> use Auto Signal Fit (MAD) and/or OIII Boost to align weak signals to H-alpha. Background Neutralization
                      aligns black points across channels.</li>
                  <li><b>Mix:</b> use the R, G, B sliders to define the final palette.</li>
                  <li><b>Process:</b> click <b>PROCESS</b> to generate a <b>LINEAR</b> (dark) output file. The result is intended to be stretched with
                      <b>VeraLux HMS</b>.</li>
                </ol>

                <h2 style='color:#ffffff; margin-top:22px;'>Inputs &amp; outputs</h2>
                <p><b>Input:</b> Linear RGB FITS (32-bit floating point).<br>
                <b>Output:</b> Balanced / Mixed Linear RGB FITS (32-bit floating point).</p>

                <div style="margin-top:22px; padding:0px; border:none; background:transparent;">
                  <table width="100%" cellspacing="0" cellpadding="0" style="border-collapse:collapse;">
                    <tr>
                      <td align="center">
                        <a href="https://youtu.be/2N6K2KnQy1o?si=X-EJox9UgyvwdqXS" style="text-decoration:none;">
                          <img src="https://img.youtube.com/vi/2N6K2KnQy1o/maxresdefault.jpg"
                               width="640" height="360"
                               style="display:block; border-radius:8px; border:1px solid #2a2a2a;">
                        </a>
                      </td>
                    </tr>
                  </table>
                </div>
                """
            }
        ]
    },
    {
        "phase": "2. STRETCHING & RECONSTRUCTION",
        "desc": "Transformation & Star Physics.",
        "color": "#ffb000",
        "tools": [
            {
                "id": "hms",
                "name": "HyperMetric Stretch",
                "tagline": "Photometric Stretching Engine",
                "badges": ["STRETCH", "PHYSICS"],
                "keywords": "stretch non-linear arcsinh histogram level brightness contrast vector color",
                "desc_html": """
                <h2 style='color:#ffffff; margin-top:0px;'>Overview</h2>
                <p>
                A precision linear-to-nonlinear stretching engine designed to maximize sensor
                fidelity while managing the transition to the visible domain.
                </p>
                <p>
                <b>HyperMetric Stretch (HMS)</b> operates on a fundamental axiom: standard histogram
                transformations often destroy the photometric relationships between color channels
                (hue shifts) and clip high-dynamic range data. HMS solves this by decoupling
                <b>luminance geometry</b> from <b>chromatic vectors</b>.
                </p>

                <h2 style='color:#ffffff; margin-top:22px;'>Design goals</h2>
                <ul>
                  <li>Preserve original vector color ratios during extreme stretching (<b>True Color</b>)</li>
                  <li>Optimize luminance extraction based on specific hardware (<b>Sensor Profiles</b>)</li>
                  <li>Provide a mathematically safe expansion for high-dynamic targets</li>
                  <li>Bridge the gap between numerical processing and visual feedback (<b>Live Preview</b>)</li>
                  <li>Allow controlled hybrid tone-mapping via <b>Color Grip</b> &amp; <b>Shadow Convergence</b></li>
                </ul>

                <h2 style='color:#ffffff; margin-top:22px;'>Core features</h2>
                <ul>
                  <li><b>Live Preview &amp; Analytics:</b>
                    <ul>
                      <li>Interactive floating window offering real-time feedback on parameter changes.</li>
                      <li>Real-time RGB histogram overlay with smart clipping warnings (&gt;0.1%).</li>
                      <li>Smart Proxy technology for fluid response even on massive files.</li>
                      <li>Professional navigation controls: Zoom, Pan, Fit-to-Screen, and <b>1:1 View</b>.</li>
                      <li><b>Diagnostic Info Overlay:</b> toggleable readout of clipping statistics and Linear Expansion bounds directly on the preview.</li>
                    </ul>
                  </li>
                  <li><b>Intelligent Calibration System:</b>
                    <ul>
                      <li><b>Smart Iterative Solver:</b> the Auto-Calculator performs a predictive <i>Floating Sky Check</i>, simulating post-stretch scaling
                          to detect black clipping and iteratively adjust the target for maximum contrast without data loss.</li>
                      <li><b>Adaptive Anchor:</b> morphological analysis for precise black point detection (default ON), superior to standard percentile clipping on gradient-free data.</li>
                    </ul>
                  </li>
                  <li><b>Hybrid Color Engine:</b>
                    <ul>
                      <li><b>Scientific Mode:</b> full manual control over <b>Linear Expansion</b>, Color Grip, and Shadow Convergence. Includes <i>Smart Max</i>
                          safety to reject hot pixels while preserving star cores during normalization.</li>
                      <li><b>Ready-to-Use Mode:</b> orchestrated pipeline with <i>Smart Max</i> zero-clipping logic and a <i>Unified Color Strategy</i> slider for one-knob balancing
                          of background noise vs highlight saturation.</li>
                    </ul>
                  </li>
                  <li><b>Unified Math Core:</b> single source of truth architecture: Auto-Solver, Live Preview, and Main Processor share the exact same logic.</li>
                </ul>

                <h2 style='color:#ffffff; margin-top:22px;'>Usage</h2>
                <ol>
                  <li><b>Pre-requisite:</b> image must be <b>Linear</b> and color-calibrated (SPCC).</li>
                  <li><b>Setup:</b> select Sensor Profile and Processing Mode.</li>
                  <li><b>Calibrate:</b>
                    <ul>
                      <li>Adaptive Anchor is ON by default (recommended for maximum dynamic range).</li>
                      <li>Click <b>Calculate Optimal Log D</b>. The solver will find the safe limit.</li>
                    </ul>
                  </li>
                  <li><b>Refine (Live Preview):</b>
                    <ul>
                      <li><b>Scientific:</b> adjust Linear Expansion to fill dynamic range.</li>
                      <li><b>Ready-to-Use:</b> adjust Color Strategy to clean noise (left) or save highlights (right).</li>
                      <li>Use the Histogram/Info Overlay to verify clipping (red bars/text).</li>
                    </ul>
                  </li>
                  <li><b>Process:</b> click <b>PROCESS</b>.</li>
                </ol>

                <h2 style='color:#ffffff; margin-top:22px;'>Inputs &amp; outputs</h2>
                <p><b>Input:</b> Linear FITS/TIFF (RGB/Mono), 16/32-bit int or float.<br>
                <b>Output:</b> Non-linear (stretched) 32-bit float FITS.</p>

                <div style="margin-top:22px; padding:0px; border:none; background:transparent;">
                  <table width="100%" cellspacing="0" cellpadding="0" style="border-collapse:collapse;">
                    <tr>
                      <td align="center">
                        <a href="https://youtu.be/LsD8SNXO_M8?si=Lxg5nLM1Pct4izjd" style="text-decoration:none;">
                          <img src="https://img.youtube.com/vi/LsD8SNXO_M8/maxresdefault.jpg"
                               width="640" height="360"
                               style="display:block; border-radius:8px; border:1px solid #2a2a2a;">
                        </a>
                      </td>
                    </tr>
                  </table>
                </div>

                <div style="margin-top:16px; padding:0px; border:none; background:transparent;">
                  <table width="100%" cellspacing="0" cellpadding="0" style="border-collapse:collapse;">
                    <tr>
                      <td align="center">
                        <a href="https://youtu.be/dwl5THmU5u4?si=UBKF5n5pYpIUvsYA" style="text-decoration:none;">
                          <img src="https://img.youtube.com/vi/dwl5THmU5u4/maxresdefault.jpg"
                               width="640" height="360"
                               style="display:block; border-radius:8px; border:1px solid #2a2a2a;">
                        </a>
                      </td>
                    </tr>
                  </table>
                </div>
                """
            }
        ]
    },
    {
        "phase": "3. TONE MAPPING & SCULPTING",
        "desc": "Detail, Volume and Contrast.",
        "color": "#ff55ff",
        "tools": [
            {
                "id": "curves",
                "name": "Curves",
                "tagline": "Spline-Based Contrast",
                "badges": ["NON-LINEAR"],
                "keywords": "contrast brightness saturation curve spline sculpt",
                "desc_html": """
                <h2 style='color:#ffffff; margin-top:0px;'>Overview</h2>
                <p>
                <b>VeraLux Curves</b> is a precision non-linear transformation engine designed to sculpt
                the tones and chromatic intensity of an image using advanced spline interpolation.
                Unlike standard curve tools that introduce overshoot artifacts (ringing) near
                control points, Curves uses <b>Akima splines</b> to ensure smooth, natural transitions
                mimicking the response of physical film.
                </p>
                <p>
                It features a <b>Split-Domain</b> architecture, allowing users to manipulate
                Luminance (CIE L*), Chrominance (CIE LCH), and Saturation (HSV) independently
                without cross-channel contamination.
                </p>

                <h2 style='color:#ffffff; margin-top:22px;'>Key features</h2>
                <ul>
                  <li><b>Akima Spline Engine:</b> natural oscillation-free interpolation.</li>
                  <li><b>Full Endpoint Freedom:</b>
                    <ul>
                      <li>Move the black point horizontally to clip shadows.</li>
                      <li>Move the white point horizontally to clip highlights.</li>
                      <li>Move vertically to lift blacks or dampen whites (matte effect).</li>
                    </ul>
                  </li>
                  <li><b>Luminance Range Control:</b> apply curves selectively to tonal zones (shadows/midtones/highlights)
                      with ultra-smooth feathering for invisible transitions. Each channel can target independent
                      luminance ranges without external masks.</li>
                  <li><b>Clipping Monitor:</b> real-time feedback showing the percentage of pixels clipped to pure black or pure white.</li>
                  <li><b>Dynamic Histogram:</b> real-time feedback showing how the curve reshapes the tonal distribution,
                      with visual dimming of excluded luminance ranges.</li>
                  <li><b>Photometric modes:</b>
                    <ul>
                      <li><b>L (Luminance):</b> adjusts contrast in CIE L*a*b* space without shifting saturation.</li>
                      <li><b>C (Chrominance):</b> boosts spectral intensity (CIE LCH) more naturally than digital saturation.</li>
                      <li><b>S (Saturation):</b> mathematical HSV purity control.</li>
                    </ul>
                  </li>
                  <li><b>Live pipette:</b> click anywhere on the image to locate the exact tonal value on the curve.</li>
                  <li><b>Detachable graph:</b> work on the curve in a separate, resizable window for maximum precision.</li>
                </ul>

                <h2 style='color:#ffffff; margin-top:22px;'>Usage</h2>
                <ol>
                  <li><b>Select channel:</b> choose the target domain (RGB/K, R, G, B, L, C, S).</li>
                  <li><b>Set luminance range (optional):</b>
                    <ul>
                      <li>Enable <b>Lum Range</b> to activate zone-based tonal control.</li>
                      <li>Toggle <b>Show Mask</b> to visualize the exact selection mask on the preview image.</li>
                      <li>Adjust Min/Max to define the luminance range (0–100%).</li>
                      <li>Set Feather to control transition smoothness (default: 25%).</li>
                      <li>Double-click any slider to reset to default values.</li>
                      <li>Excluded zones are grayed-out on the grid and dimmed on the histogram.</li>
                    </ul>
                  </li>
                  <li><b>Pipette:</b> click on the preview image to locate the tone you want to alter (a vertical marker appears on the graph).</li>
                  <li><b>Sculpt:</b>
                    <ul>
                      <li><b>Left click:</b> add points or drag existing ones.</li>
                      <li><b>Right click:</b> remove points.</li>
                      <li><b>Black point:</b> drag bottom-left horizontally to clip shadows.</li>
                      <li><b>White point:</b> drag top-right horizontally to clip highlights.</li>
                    </ul>
                  </li>
                  <li><b>Visual feedback:</b>
                    <ul>
                      <li>Active channels show a ● marker in the channel list.</li>
                      <li>Range-limited channels show an additional ⟨⟩ marker.</li>
                      <li>Excluded luminance zones appear grayed-out on the curve grid and dimmed on the histogram.</li>
                    </ul>
                  </li>
                  <li><b>Iterate &amp; apply:</b>
                    <ul>
                      <li>Edits are applied non-destructively on top of the last applied state.</li>
                      <li>Press <b>Apply</b> to freeze current active curves as a new iteration stage.</li>
                      <li>Applied stages are added to a linear stack for step-by-step construction.</li>
                      <li>After applying, channels reset to identity, ready for the next iteration.</li>
                    </ul>
                  </li>
                  <li><b>Undo:</b>
                    <ul>
                      <li>Press <b>Undo</b> to remove the most recent applied iteration.</li>
                      <li>The preview is rebuilt from the remaining stage stack.</li>
                      <li>Undo affects only applied stages, never intermediate (uncommitted) edits.</li>
                    </ul>
                  </li>
                  <li><b>Compare:</b> hold <b>Spacebar</b> to view the original image.</li>
                  <li><b>Process:</b> applies the full stage stack and current active curves to the full-resolution image.</li>
                </ol>

                <h2 style='color:#ffffff; margin-top:22px;'>Example workflows</h2>
                <ul>
                  <li>Boost Chrominance (C) only in deep shadows (0–30% luminance).</li>
                  <li>Lift midtone Luminance (L) without affecting highlights (30–70%).</li>
                  <li>Compress bright regions (70–100%) while preserving shadow detail.</li>
                  <li>Apply selective color grading to specific tonal zones per channel.</li>
                </ul>

                <h2 style='color:#ffffff; margin-top:22px;'>Technical details</h2>
                <ul>
                  <li><b>Interpolation:</b> Akima splines (oscillation-free, C1-continuous).</li>
                  <li><b>Precision:</b> full 32-bit float pipeline throughout.</li>
                  <li><b>Range masking algorithm:</b> sigmoid roll-off (k=20.0), no spatial blur, computed on input luminance (RGB K-average).</li>
                  <li><b>Feather range:</b> 0–100% (recommended: 20–40% for invisible transitions).</li>
                </ul>

                <div style="margin-top:22px; padding:0px; border:none; background:transparent;">
                  <table width="100%" cellspacing="0" cellpadding="0" style="border-collapse:collapse;">
                    <tr>
                      <td align="center">
                        <a href="https://youtu.be/2Of-6qmgCOM?si=CgRmXLBJUS2swaKU" style="text-decoration:none;">
                          <img src="https://img.youtube.com/vi/2Of-6qmgCOM/maxresdefault.jpg"
                               width="640" height="360"
                               style="display:block; border-radius:8px; border:1px solid #2a2a2a;">
                        </a>
                      </td>
                    </tr>
                  </table>
                </div>
                """
            },
            {
                "id": "revela",
                "name": "Revela",
                "tagline": "Local Contrast & Texture",
                "badges": ["NON-LINEAR", "PHYSICS"],
                "keywords": "detail texture sharp clarity structure volume micro-contrast",
                "desc_html": """
                <h2 style='color:#ffffff; margin-top:0px;'>Overview</h2>
                <p>
                <b>VeraLux Revela</b> is a post-stretch enhancement engine designed to reveal
                micro-contrast and volumetric structures in nebulae and galaxies without
                compromising the noise floor or star profiles.
                </p>
                <p>
                It employs a fully adaptive <b>signal-aware</b> approach:
                </p>
                <ol>
                  <li><b>Adaptive Noise Gate:</b> automatically calculates the noise floor (MAD/Sigma)
                      of the specific image. The <b>Shadow Authority</b> slider scales this physical limit.</li>
                  <li><b>Frequency separation:</b> decouples Micro-Contrast (Texture) from Macroscopic
                      Volume (Structure) using <b>\u00c0 trous wavelets</b>.</li>
                  <li><b>Geometric protection:</b> automatically identifies high-energy stellar profiles
                      to prevent sharpening artifacts (halos/ringing) on stars.</li>
                </ol>

                <h2 style='color:#ffffff; margin-top:22px;'>Key features</h2>
                <ul>
                  <li><b>High-Fidelity Preview:</b> dual-stage rendering (Proxy/FullRes) with energy-preserving interpolation.</li>
                  <li><b>Tri-domain control:</b> separate sliders for Texture and Structure.</li>
                  <li><b>Shadow Authority:</b> adaptive photometric gate based on image statistics.</li>
                  <li><b>Mask visualization:</b> real-time preview of the protection gate (white=active, black=protected).</li>
                  <li><b>Star protection:</b> energy-based masking to prevent raccoon-eyes artifacts.</li>
                  <li><b>Smart workflow:</b> non-blocking background processing for high-resolution output.</li>
                </ul>

                <h2 style='color:#ffffff; margin-top:22px;'>Usage</h2>
                <ol>
                  <li><b>Input:</b> designed for <b>Non-Linear (stretched)</b> images to enhance visual volume and detail.</li>
                  <li><b>Tuning:</b>
                    <ul>
                      <li><b>Texture:</b> pops fine details (shock fronts, dust lanes).</li>
                      <li><b>Structure:</b> enhances the 3D volume of the object.</li>
                      <li><b>Show Mask:</b> enable to see what is being protected.</li>
                      <li><b>Shadow Authority:</b> adjust while viewing the mask until the background is black.</li>
                    </ul>
                  </li>
                  <li><b>Process:</b> apply the enhancement.</li>
                </ol>

                <h2 style='color:#ffffff; margin-top:22px;'>Inputs &amp; outputs</h2>
                <p><b>Input:</b> Stretched RGB/Mono image (open in Siril).<br>
                <b>Output:</b> Enhanced image loaded back into Siril.</p>

                <div style="margin-top:22px; padding:0px; border:none; background:transparent;">
                  <table width="100%" cellspacing="0" cellpadding="0" style="border-collapse:collapse;">
                    <tr>
                      <td align="center">
                        <a href="https://youtu.be/cMV7fqNT1Lg?si=fGGxIcBTj1RwxCXy" style="text-decoration:none;">
                          <img src="https://img.youtube.com/vi/cMV7fqNT1Lg/maxresdefault.jpg"
                               width="640" height="360"
                               style="display:block; border-radius:8px; border:1px solid #2a2a2a;">
                        </a>
                      </td>
                    </tr>
                  </table>
                </div>
                """
            }
        ]
    },
    {
        "phase": "4. FINISHING",
        "desc": "Final Color Grading.",
        "color": "#55ff55",
        "tools": [
            {
                "id": "vectra",
                "name": "Vectra",
                "tagline": "Vector Color Grading",
                "badges": ["NON-LINEAR", "COLOR"],
                "keywords": "color grading hue saturation teal gold shift hubble palette",
                "desc_html": """
                <h2 style='color:#ffffff; margin-top:0px;'>Overview</h2>
                <p>
                <b>VeraLux Vectra</b> is a surgical color grading engine operating in the <b>LCH</b>
                (Lightness, Chroma, Hue) vector space. It is designed to manipulate specific
                color ranges without altering the luminance structure (histogram) or damaging
                background neutrality.
                </p>
                <p>
                It addresses the lack of selective color correction in Siril, allowing users to:
                </p>
                <ul>
                  <li>Shift hues (e.g., move Green towards Teal for SHO palettes).</li>
                  <li>Boost chroma selectively (e.g., pop the Blue OIII without oversaturating Red Ha).</li>
                  <li>Protect stars and background from unwanted color shifts.</li>
                </ul>

                <h2 style='color:#ffffff; margin-top:22px;'>Key features</h2>
                <ul>
                  <li><b>6-vector control:</b> independent Hue/Saturation control for Red, Green, Blue, Cyan, Magenta, Yellow.</li>
                  <li><b>Luminance lock:</b> operations are strictly chromatic. Lightness (L) is mathematically frozen to preserve
                      the structural work done by HMS and Revela.</li>
                  <li><b>Shadow Authority:</b> a neutrality lock that prevents color grading from tinting the background noise floor.</li>
                  <li><b>White star integrity:</b> energy-based protection to keep stellar cores neutral even during aggressive saturation boosts.</li>
                  <li><b>Vector scope:</b> real-time visual feedback of color shifts and saturation gains on a holographic spectral wheel.</li>
                </ul>

                <h2 style='color:#ffffff; margin-top:22px;'>Usage</h2>
                <ol>
                  <li><b>Input state:</b> image must be <b>Non-Linear (stretched)</b>. Do not use on linear data. Best used after HMS and Revela,
                      before final cropping.</li>
                  <li><b>Vectors:</b>
                    <ul>
                      <li>Select <b>Primary</b> (RGB) or <b>Secondary</b> (CMY) tabs.</li>
                      <li>Use <b>Hue Shift</b> to rotate the color (e.g., make Red more golden).</li>
                      <li>Use <b>Saturation</b> to boost the intensity of that specific color.</li>
                    </ul>
                  </li>
                  <li><b>Protection:</b>
                    <ul>
                      <li><b>Shadow Authority:</b> increase to lock the background gray.</li>
                      <li><b>Star Protect:</b> keeps stars white.</li>
                    </ul>
                  </li>
                  <li><b>Process:</b> apply the vector transformation.</li>
                </ol>

                <div style="margin-top:22px; padding:0px; border:none; background:transparent;">
                  <table width="100%" cellspacing="0" cellpadding="0" style="border-collapse:collapse;">
                    <tr>
                      <td align="center">
                        <a href="https://youtu.be/ljBuhvNF5Bk?si=m5-PlWJBK0mIsPuB" style="text-decoration:none;">
                          <img src="https://img.youtube.com/vi/ljBuhvNF5Bk/maxresdefault.jpg"
                               width="640" height="360"
                               style="display:block; border-radius:8px; border:1px solid #2a2a2a;">
                        </a>
                      </td>
                    </tr>
                  </table>
                </div>
                """
            }
        ]
    },
    {
        "phase": "5. RECOMPOSITION",
        "desc": "Star Management.",
        "color": "#e0c060",
        "tools": [
            {
                "id": "starcomposer",
                "name": "StarComposer",
                "tagline": "Star Reconstruction",
                "badges": ["STARS", "PHYSICS", "STRETCH"],
                "keywords": "stars bloat halos reconstruction mask combine composition",
                "desc_html": """
                <h2 style='color:#ffffff; margin-top:0px;'>Overview</h2>
                <p>
                A specialized photometric reconstruction engine designed for deep-sky
                astrophotography.
                </p>
                <p>
                <b>VeraLux StarComposer</b> solves the <i>bloating</i> and <i>bleaching</i> issues inherent in
                standard star stretching by decoupling the stellar field from the main object.
                It leverages the rigorous <b>Inverse Hyperbolic Stretch (IHS)</b> to develop linear
                star masks with precision, preserving true stellar color and geometry (PSF)
                before compositing them onto non-linear starless images.
                </p>

                <h2 style='color:#ffffff; margin-top:22px;'>Key features v2.0</h2>
                <ul>
                  <li><b>Hybrid Scalar Engine:</b> new math core that ensures stars maintain a solid, white energy core (solving the soft/flat star issue)
                      while preserving chromatic data in the halos.</li>
                  <li><b>High-Fidelity Preview:</b> upgraded QHD architecture with energy-preserving interpolation for strict WYSIWYG consistency between
                      live preview and final result.</li>
                  <li><b>Lightweight Core:</b> removed heavy dependencies (Astropy) for faster execution.</li>
                  <li><b>True Color Pipeline:</b> vector preservation available via Color Grip.</li>
                  <li><b>Smart Workflow:</b> autosave to source directory, green-light UI feedback, and Siril integration.</li>
                </ul>

                <h2 style='color:#ffffff; margin-top:22px;'>Design goals</h2>
                <ul>
                  <li>Treat stars as geometric entities (Gaussian/Moffat profiles).</li>
                  <li>Preserve stellar chrominance ratios while controlling brightness and profile geometry.</li>
                  <li>Repair optical defects (chromatic aberration) via chroma-only Gaussian smoothing.</li>
                  <li>Automate the cleanup of star-mask residual artifacts (Shadow Convergence).</li>
                </ul>

                <h2 style='color:#ffffff; margin-top:22px;'>Core features</h2>
                <ul>
                  <li><b>The VeraLux Engine:</b> anchored IHS function for conical star profiles.</li>
                  <li><b>Hybrid Physics:</b> Color Grip blends between Scalar (crisp) and Vector (color) modes.</li>
                  <li><b>Star Surgery:</b> includes Optical Healing for halos and Dynamic LSR for galaxy-core removal.</li>
                </ul>

                <h2 style='color:#ffffff; margin-top:22px;'>Usage</h2>
                <ol>
                  <li><b>Load data:</b> import Starless (non-linear base) and Starmask (linear).</li>
                  <li><b>Blend mode:</b> use <b>Screen</b> if the mask has bright residuals, <b>Linear Add</b> otherwise.</li>
                  <li><b>Intensity:</b> increase <b>Star Intensity</b> (gold slider) to define brightness.</li>
                  <li><b>Geometry:</b> adjust <b>Profile Hardness</b> (b) to sculpt the PSF (sharpen/soften).</li>
                  <li><b>Generate:</b> click <b>PROCESS</b>. Result is loaded into Siril.</li>
                </ol>

                <h2 style='color:#ffffff; margin-top:22px;'>Inputs &amp; outputs</h2>
                <p><b>Input:</b> Linear Starmask FITS + Stretched Starless FITS.<br>
                <b>Output:</b> Recombined RGB FITS (32-bit float).</p>

                <div style="margin-top:22px; padding:0px; border:none; background:transparent;">
                  <table width="100%" cellspacing="0" cellpadding="0" style="border-collapse:collapse;">
                    <tr>
                      <td align="center">
                        <a href="https://youtu.be/Pen9aOy2NmA?si=3RyPSPbScLcy3PfE" style="text-decoration:none;">
                          <img src="https://img.youtube.com/vi/Pen9aOy2NmA/maxresdefault.jpg"
                               width="640" height="360"
                               style="display:block; border-radius:8px; border:1px solid #2a2a2a;">
                        </a>
                      </td>
                    </tr>
                  </table>
                </div>
                """
            }
        ]
    }
]

# =============================================================================
#  HELPER
# =============================================================================

def hex_to_rgba_string(hex_color, alpha_percent):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        a = int(255 * (alpha_percent / 100.0))
        return f"rgba({r}, {g}, {b}, {a})"
    return hex_color

def generate_badges_html(badges_list):
    base_bg = "#262626"
    base_border = "#3a3a3a"
    base_text = "#bbbbbb"

    html_parts = []
    for b in badges_list:
        spec = BADGE_STYLES.get(b)
        if not spec:
            continue

        mode = spec.get("mode", "none")
        accent = spec.get("accent") or base_border

        style = (
            f"background-color:{base_bg};"
            f"color:{base_text};"
            f"border:1px solid {base_border};"
            f"border-radius:3px;"
            f"padding:2px 6px;"
            f"font-size:9px;"
            f"font-weight:700;"
            f"letter-spacing:0.5px;"
        )

        if mode == "dot":
            icon = f"<span style='color:{accent}; font-size:10px;'>●</span>"
            label = f"{icon}&nbsp;{b}"
        elif mode == "hollow":
            icon = f"<span style='color:{accent}; font-size:10px;'>○</span>"
            label = f"{icon}&nbsp;{b}"
        elif mode == "star":
            icon = f"<span style='color:{accent}; font-size:10px;'>✦</span>"
            label = f"{icon}&nbsp;{b}"
        else:
            label = b

        html_parts.append(f"<span style=\"{style}\">{label}</span>")

    return "&nbsp;&nbsp;".join(html_parts)

# =============================================================================
#  MAIN WINDOW
# =============================================================================

class StartingPointGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"VeraLux Starting Point v{VERSION}")
        self.setStyleSheet(DARK_STYLESHEET)
        self.resize(1150, 700)
        
        # State tracker for Chaos Mode
        self.is_chaos_mode = False
        self.menu_buttons = {}

        header_msg = (
            f"\n##############################################\n"
            f"# VeraLux — Starting Point v{VERSION}\n"
            "# Interactive Workflow Guide & Manual\n"
            "# Author: Riccardo Paterniti (2025)\n"
            "# Contact: info@veralux.space\n"
            "##############################################"
        )
        
        try:
            self.siril.log(header_msg)
        except:
            print(header_msg)

        self.tool_map = {}
        for phase in WORKFLOW_DATA:
            for tool in phase['tools']:
                tool_entry = tool.copy()
                tool_entry['phase_color'] = phase['color']
                self.tool_map[tool['id']] = tool_entry

        # Cache for remote thumbnails (URL -> data URI)
        self._thumb_cache = {}

        self.init_ui()

    def init_ui(self):
        main = QWidget()
        self.setCentralWidget(main)

        layout = QHBoxLayout(main)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(25)

        # --- LEFT COLUMN (Nav) ---
        left_container = QWidget()
        left_container.setFixedWidth(360)
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(15)

        # Header
        header_box = QVBoxLayout()
        header_box.setSpacing(5)

        # --- EASTER EGG TRIGGER ---
        self.lbl_title = QLabel("VeraLux Starting Point")
        self.lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_title.setStyleSheet("font-size: 18pt; font-weight: bold; color: #88aaff;")
        self.lbl_title.installEventFilter(self) # Listen for clicks

        self.lbl_sub = QLabel("Interactive Workflow Manual")
        self.lbl_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_sub.setStyleSheet("font-size: 10pt; color: #888888; font-style: italic;")

        header_box.addWidget(self.lbl_title)
        header_box.addWidget(self.lbl_sub)
        left_layout.addLayout(header_box)

        # Search Bar
        self.search_bar = VeraLuxSearchBar()
        self.search_bar.textChanged.connect(self.on_search)
        left_layout.addWidget(self.search_bar)

        # Scroll Area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 5, 0, 5)

        scroll_layout.setSpacing(25)

        for phase_data in WORKFLOW_DATA:
            p_color = phase_data['color']
            g_phase = QGroupBox(phase_data['phase'])
            g_phase.setStyleSheet(f"QGroupBox::title {{ color: {p_color}; border-color: {p_color}; }}")

            l_phase = QVBoxLayout(g_phase)
            l_phase.setSpacing(2)

            l_phase.setContentsMargins(5, 5, 5, 5)

            hover_bg = hex_to_rgba_string(p_color, 15)
            border_left_col = p_color

            btn_base_style = f"""
            QPushButton.MenuButton {{
                border-left: 3px solid transparent;
            }}
            QPushButton.MenuButton:hover {{
                background-color: {hover_bg};
                border-left: 3px solid {border_left_col};
                color: #ffffff;
            }}
            QPushButton.MenuButton:checked {{
                background-color: {hex_to_rgba_string(p_color, 25)};
                border-left: 3px solid {border_left_col};
                color: white;
            }}
            """

            for tool in phase_data['tools']:
                btn = QPushButton(f"{tool['name']}")
                btn.setProperty("class", "MenuButton")
                btn.setCursor(Qt.CursorShape.PointingHandCursor)
                btn.setStyleSheet(btn_base_style)

                btn.clicked.connect(lambda checked, tid=tool['id']: self.show_details(tid))
                
                # Store reference and original name for Chaos Mode swapping
                btn.tool_id = tool['id']
                btn.original_name = tool['name']
                self.menu_buttons[tool['id']] = btn
                
                l_phase.addWidget(btn)

            scroll_layout.addWidget(g_phase)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        left_layout.addWidget(scroll)

        # Footer
        footer_layout = QHBoxLayout()
        footer_layout.setSpacing(5)
        footer_layout.setContentsMargins(0, 0, 0, 0)

        self.chk_ontop = QCheckBox("On Top")
        self.chk_ontop.setChecked(False)
        self.chk_ontop.setToolTip("Keep the window always visible above other applications.")
        self.chk_ontop.toggled.connect(self.toggle_ontop)

        self.btn_coffee = QPushButton("☕")
        self.btn_coffee.setObjectName("CoffeeButton")
        self.btn_coffee.setToolTip("Support the developer - Buy me a coffee!")
        self.btn_coffee.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_coffee.clicked.connect(lambda: webbrowser.open("https://buymeacoffee.com/riccardo.paterniti"))

        self.btn_home = QPushButton("⌂")
        self.btn_home.setObjectName("HomeButton")
        self.btn_home.setToolTip("Back to the Starting Point home screen.")
        self.btn_home.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_home.clicked.connect(self.go_home)

        btn_close = QPushButton("Close")
        btn_close.setObjectName("CloseButton")
        btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_close.clicked.connect(self.close)

        footer_layout.addWidget(self.chk_ontop)
        footer_layout.addSpacing(15)
        footer_layout.addWidget(self.btn_coffee)
        footer_layout.addStretch()
        footer_layout.addWidget(self.btn_home)
        footer_layout.addWidget(btn_close)

        left_layout.addLayout(footer_layout)
        layout.addWidget(left_container)

        # --- RIGHT COLUMN ---
        self.detail_view = QTextBrowser()
        self.detail_view.setOpenExternalLinks(False)
        self.detail_view.setOpenLinks(False)
        self.detail_view.anchorClicked.connect(self.handle_anchor)

        self.show_welcome_screen()

        layout.addWidget(self.detail_view)

    # =========================================================================
    #  EASTER EGG LOGIC
    # =========================================================================
    
    def eventFilter(self, obj, event):
        # Intercept click on Title Label
        if obj == self.lbl_title and event.type() == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.LeftButton:
                self.trigger_easter_egg()
                return True
        return super().eventFilter(obj, event)

    def trigger_easter_egg(self):
        # 1. PANIC BUTTON: If already in chaos mode, Restore Normality
        if self.is_chaos_mode:
            self.restore_normality()
            return

        # 2. ASK THE QUESTION
        dialog = QInputDialog(self)
        dialog.setWindowTitle("🚨 SYSTEM OVERRIDE")
        dialog.setLabelText("The name of the one who inspires our every step:")
        dialog.setTextValue("")        
        dialog.setWindowFlags(
            Qt.WindowType.Dialog | 
            Qt.WindowType.WindowStaysOnTopHint | 
            Qt.WindowType.CustomizeWindowHint |
            Qt.WindowType.WindowTitleHint |
            Qt.WindowType.WindowCloseButtonHint
        )
        dialog.setModal(True)        
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        
        if dialog.exec() == QInputDialog.DialogCode.Accepted:
            text = dialog.textValue()
            if text.lower().strip() == "cyril":
                self.activate_chaos_mode()
            else:
                # Silent fail
                pass

    def activate_chaos_mode(self):
        self.is_chaos_mode = True

        url = "https://youtu.be/hnf8rFCTACU?si=EKTV6-sNtuYk7DmH"

        self.setStyleSheet(CHAOS_STYLESHEET)

        self.setWindowTitle("VeraLux — CHAOS POINT v999.0")
        self.lbl_title.setText("🦄 VeraSux - Chaos Point 🍄")
        self.lbl_sub.setText("Interactive Disaster Guide")
        self.btn_coffee.setText("🍺")
        self.btn_home.setText("🎪")
        self.search_bar.icon_lbl.setText("🤡")
        self.search_bar.input.setPlaceholderText("Destroy image (e.g. 'clipping')...")

        for tid, btn in self.menu_buttons.items():
            if tid in CHAOS_TOOLS:
                btn.setText(CHAOS_TOOLS[tid]['name'])
            btn.setStyleSheet("")

        self.show_welcome_screen()
        QApplication.processEvents()

        _was_visible = self.isVisible()
        self.hide()
        QApplication.processEvents()

        msg = QMessageBox()
        msg.setStyleSheet(CHAOS_STYLESHEET)
        msg.setWindowTitle("CHAOS MODE ACTIVATED")
        msg.setText("ACHIEVEMENT UNLOCKED: The Cyril Protocol.\n\nPhysics disabled. Logic removed.\nHave fun.")
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowModality(Qt.WindowModality.ApplicationModal)
        msg.setWindowFlags(msg.windowFlags() | Qt.WindowType.Dialog | Qt.WindowType.WindowStaysOnTopHint)

        msg.show()
        msg.raise_()
        msg.activateWindow()

        def _open_video():
            try:
                if sys.platform == "darwin":
                    import subprocess
                    subprocess.Popen(["open", "-g", url])
                else:
                    try:
                        webbrowser.open(url, new=2, autoraise=True)
                    except TypeError:
                        webbrowser.open(url, new=2)
            except Exception:
                try:
                    webbrowser.open(url, new=2)
                except Exception:
                    pass

            def _refront():
                try:
                    msg.raise_()
                    msg.activateWindow()
                except Exception:
                    pass

            QTimer.singleShot(250, _refront)
            QTimer.singleShot(800, _refront)

        QTimer.singleShot(200, _open_video)

        msg.exec()

        if _was_visible:
            try:
                self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
            except Exception:
                pass
            self.show()
            QApplication.processEvents()
            try:
                self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, False)
            except Exception:
                pass

    def restore_normality(self):
        self.is_chaos_mode = False
        
        # 1. Restore Stylesheet
        self.setStyleSheet(DARK_STYLESHEET)
        
        # 2. Restore Texts
        self.setWindowTitle(f"VeraLux Starting Point v{VERSION}")
        self.lbl_title.setText("VeraLux Starting Point")
        self.lbl_sub.setText("Interactive Workflow Manual")
        self.btn_coffee.setText("☕")
        self.btn_home.setText("⌂")
        self.search_bar.icon_lbl.setText("🔍")
        self.search_bar.input.setPlaceholderText("Problem Solver (e.g. 'noise')...")
        self.search_bar.apply_normal_style() # Fix search bar border

        # 3. Restore Menu Buttons
        for tid, btn in self.menu_buttons.items():
            btn.setText(btn.original_name)

            tool_data = self.tool_map.get(tid)
            if tool_data:
                p_color = tool_data['phase_color']
                hover_bg = hex_to_rgba_string(p_color, 15)
                border_left_col = p_color
                
                btn_style = f"""
                QPushButton.MenuButton {{
                    border-left: 3px solid transparent;
                }}
                QPushButton.MenuButton:hover {{
                    background-color: {hover_bg};
                    border-left: 3px solid {border_left_col};
                    color: #ffffff;
                }}
                QPushButton.MenuButton:checked {{
                    background-color: {hex_to_rgba_string(p_color, 25)};
                    border-left: 3px solid {border_left_col};
                    color: white;
                }}
                """
                btn.setStyleSheet(btn_style)

        # 4. Refresh content view
        self.show_welcome_screen()
        
        QApplication.processEvents()

        # 5. Show popup
        msg = QMessageBox(self)
        msg.setWindowTitle("Panic Button")
        msg.setText("Physics restored. Boring mode reactivated.")
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowModality(Qt.WindowModality.ApplicationModal)
        msg.setWindowFlags(msg.windowFlags() | Qt.WindowType.Dialog | Qt.WindowType.WindowStaysOnTopHint)
        msg.show()
        msg.raise_()
        msg.activateWindow()
        msg.exec()

    # =========================================================================
    #  STANDARD LOGIC & CONTENT HANDLING
    # =========================================================================

    def show_welcome_screen(self):
        if self.is_chaos_mode:
            html = """
            <div style='text-align:center; margin-top:50px; font-family:"Comic Sans MS";'>
                <h1 style='color:#FF00FF; font-size:40pt; background-color:yellow;'>WELCOME TO CHAOS</h1>
                <p style='color:#0000FF; font-size:18pt;'>
                    Forget signal-to-noise ratio.<br>
                    Embrace the <b style='color:red;'>NOISE</b>.
                </p>
                <p>Click any button on the left to destroy your data.</p>
                <p style='font-size:50pt;'>🤪</p>
                <p style='font-size:10pt;'>Click the title again to panic/exit.</p>
            </div>
            """
        else:
            html = """
<div style='text-align:center; margin-top:80px;'>
    <h1 style='color:#88aaff; font-size:30pt; margin-bottom:8px;'>VeraLux Starting Point</h1>
    <p style='color:#aaaaaa; font-size:13pt; margin-bottom:25px;'>
        An interactive workflow guide for the VeraLux ecosystem
    </p>

    <div style='margin:0 auto; width:70%; text-align:left; color:#cccccc; font-size:12pt; line-height:1.7;'>
        <p>
        <b>Starting Point</b> is not a processing tool.  
        It is a <b>context-aware map</b> designed to help you choose the right VeraLux tool
        at the right stage of astrophotography processing.
        </p>

        <p>
        Use it to understand <b>when</b> a tool should be used, <b>why</b> it exists,
        and <b>what physical problem</b> it is meant to solve — from linear data preparation,
        through stretching and reconstruction, to final color grading.
        </p>

        <p style="margin-top:20px;">
        <b>The VeraLux Suite</b> is a cohesive ecosystem of tools built around a single guiding principle:
        <i>respect the data before shaping the image</i>.
        </p>

        <p>
        Every component of VeraLux is designed with physical signal integrity in mind —
        prioritizing linear-domain correctness, photometric consistency, and transparent mathematics
        over cosmetic shortcuts or preset-driven aesthetics.
        </p>

        <p>
        Rather than enforcing a fixed “look”, VeraLux provides a set of rigorously defined instruments
        that allow the user to make informed, reversible, and physically meaningful decisions at each
        stage of the workflow.
        </p>

        <p>
        Select a tool from the left menu to explore its role in the workflow,  
        or use the search bar to describe a problem (e.g. <i>noise</i>, <i>stars</i>, <i>color cast</i>)
        and find the appropriate solution.
        </p>
    </div>

    <div style='margin-top:45px; border-top: 1px solid #333333; width: 60%; margin-left:auto; margin-right:auto;'></div>

    <p style='color:#666666; font-size:10pt; margin-top:18px;'>
        VeraLux Suite — Physically Faithful Astrophotography Workflow<br>
        v1.0
    </p>
</div>
            """
        self.detail_view.setHtml(html)

    def on_search(self, text):
        text = text.lower().strip()
        if not text:
            self.show_welcome_screen()
            return
            
        results_html = "<h2 style='color:#ffffff; margin-bottom:25px;'>Search Results</h2>"
        found = False
        
        for tid, data in self.tool_map.items():
            corpus = (data['name'] + " " + data['tagline'] + " " + data.get('keywords', '') + " " + data.get('desc_html', '')).lower()
            if text in corpus:
                found = True
                col = data['phase_color']
                bg_col = hex_to_rgba_string(col, 8) 
                
                # Check for Chaos Override on display Name
                if self.is_chaos_mode and tid in CHAOS_TOOLS:
                    display_name = CHAOS_TOOLS[tid]['name']
                else:
                    display_name = data['name']

                badges_html = generate_badges_html(data.get('badges', []))
                
                results_html += f"""
                <div style='
                    background-color: {bg_col};
                    border-left: 5px solid {col};
                    padding: 15px;
                    margin-bottom: 20px;
                    border-radius: 4px;
                '>
                    <a href='tool://{tid}' style='text-decoration:none;'>
                        <div style='font-size:18pt; font-weight:bold; color:{col}; margin-bottom:5px;'>{display_name}</div>
                        <div style='color:#cccccc; font-style:italic; font-size:12pt; margin-bottom:12px;'>{data['tagline']}</div>
                        <div>{badges_html}</div>
                    </a>
                </div>
                """
        
        if not found:
            results_html += "<p style='color:#ff5555; font-size:12pt; margin-top:30px;'>No tools found matching your query.</p>"
            
        self.detail_view.setHtml(results_html)

    def handle_anchor(self, url):
        url_str = url.toString()
        if url_str.startswith("tool://"):
            tool_id = url_str.replace("tool://", "")
            self.show_details(tool_id, from_search=True)
        elif url_str == "cmd://back":
            self.on_search(self.search_bar.input.text())
        else:
            QDesktopServices.openUrl(url)

    def _fetch_image_as_data_uri(self, url: str) -> str | None:
        """Fetch an image over HTTPS and return a data: URI. Returns None on failure."""
        if url in self._thumb_cache:
            return self._thumb_cache[url]

        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "VeraLux-StartingPoint/1.0",
                    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
                },
            )
            with urllib.request.urlopen(req, timeout=8) as r:
                data = r.read()
                ctype = r.headers.get("Content-Type", "").split(";")[0].strip().lower()

            # Conservative mime fallback
            if not ctype:
                if url.lower().endswith(".png"):
                    ctype = "image/png"
                else:
                    ctype = "image/jpeg"

            b64 = base64.b64encode(data).decode("ascii")
            data_uri = f"data:{ctype};base64,{b64}"
            self._thumb_cache[url] = data_uri
            return data_uri
        except Exception:
            return None

    def _inline_youtube_thumbnails(self, html: str) -> str:
        """Replace YouTube thumbnail <img src=...> URLs with inlined data URIs for Qt QTextBrowser."""
        # Only inline known YouTube thumbnail URLs to keep this safe and fast.
        pattern = re.compile(
            r'(<img\s+[^>]*src=")(?P<url>https?://img\.youtube\.com/vi/[^\"\']+/[^\"\']+)("[^>]*>)',
            re.IGNORECASE,
        )

        def repl(m):
            url = m.group("url")

            data_uri = self._fetch_image_as_data_uri(url)
            if data_uri:
                return m.group(1) + data_uri + m.group(3)

            # Fallback: if maxres is unavailable, try hqdefault.
            if url.lower().endswith("/maxresdefault.jpg"):
                fallback = url[:-len("maxresdefault.jpg")] + "hqdefault.jpg"
                data_uri2 = self._fetch_image_as_data_uri(fallback)
                if data_uri2:
                    return m.group(1) + data_uri2 + m.group(3)

            return m.group(0)

        return pattern.sub(repl, html)

    def show_details(self, tool_id, from_search=False):
        if tool_id not in self.tool_map: return
        
        # --- LOGIC TO SWAP CONTENT IF CHAOS MODE IS ACTIVE ---
        if self.is_chaos_mode and tool_id in CHAOS_TOOLS:
            # Load Troll Content
            chaos_data = CHAOS_TOOLS[tool_id]
            name = chaos_data['name']
            tagline = chaos_data['tagline']
            desc_html = chaos_data['desc']
            color_hex = "#FF00FF" # Forced Magenta
            
            badges_html = "<span style='background:yellow; color:black; font-weight:bold; padding:5px;'>[DESTRUCTIVE]</span> &nbsp; <span style='background:red; color:white; font-weight:bold; padding:5px;'>[UNSAFE]</span>"
        else:
            # Load Normal Content
            tool_data = self.tool_map[tool_id]
            name = tool_data['name'].upper()
            tagline = tool_data['tagline']
            desc_html = tool_data.get('desc_html', '')
            color_hex = tool_data['phase_color']
            badges_html = generate_badges_html(tool_data.get('badges', []))
            
            # Inline YouTube thumbnails if present (only needed in Normal Mode)
            desc_html = self._inline_youtube_thumbnails(desc_html)
        
        # Build Navigation Link
        back_link = ""
        if from_search and self.search_bar.input.text().strip():
            back_link = f"""
            <div style='margin-bottom:20px;'>
                <a href='cmd://back' style='color:#88aaff; text-decoration:none; font-weight:bold; font-size:11pt;'>
                 &larr; Back to Results
                </a>
            </div>
            """
        
        # Render
        text_color = "#000000" if self.is_chaos_mode else "#e0e0e0"
        
        html = f"""
        {back_link}
        <h1 style='color:{color_hex}; margin-bottom:0px; font-size:28pt; letter-spacing:1px;'>{name}</h1>
        <div style='font-style:italic; color:{color_hex}; opacity:0.9; margin-bottom:20px; font-size:14pt;'>{tagline}</div>
        
        <div style='margin-bottom:30px;'>{badges_html}</div>
        
        <hr style='border: 0; border-top: 1px solid #444444; margin-bottom: 25px;'>
        
        <div style='line-height:1.7; font-size:12pt; color:{text_color};'>
            {desc_html}
        </div>
        """
        self.detail_view.setHtml(html)


    def go_home(self):
        # Reset search and return to the welcome screen.
        try:
            self.search_bar.clear_text()
        except Exception:
            self.search_bar.input.clear()
        self.show_welcome_screen()

    def toggle_ontop(self, checked):
        # Preserve existing flags, only toggle the stays-on-top hint.
        flags = self.windowFlags()
        if checked:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        else:
            flags &= ~Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = StartingPointGUI()
    gui.show()
    sys.exit(app.exec())