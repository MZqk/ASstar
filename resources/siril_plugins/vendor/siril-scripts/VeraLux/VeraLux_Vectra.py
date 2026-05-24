##############################################
# VeraLux — Vectra
# Vector Color Grading & Chromatic Surgery Engine
# Author: Riccardo Paterniti (2025)
# Contact: info@veralux.space
##############################################

# (c) 2025 Riccardo Paterniti
# VeraLux — Vectra
# SPDX-License-Identifier: GPL-3.0-or-later
# Version 1.0.3
#
# Credits / Origin
# ----------------
#   • Architecture: VeraLux Shared GUI Framework (Revela based)
#   • Engine: LCH Vector Manipulation Logic
#   • Protection: Shadow Authority (MAD) & Star Energy (Wavelet)
#

"""
Overview
--------
VeraLux Vectra is a surgical color grading engine operating in the LCH 
(Lightness, Chroma, Hue) vector space. It is designed to manipulate specific 
color ranges without altering the luminance structure (Histogram) or damaging 
the background neutrality.

It addresses the lack of Selective Color Correction in Siril, allowing users to:
• Shift hues (e.g., move Green towards Teal for SHO palettes).
• Boost chroma selectively (e.g., pop the Blue OIII without oversaturating Red Ha).
• Protect stars and background from unwanted color shifts.

Key Features
------------
• **6-Vector Control**: Independent Hue/Saturation control for Red, Green, Blue,
  Cyan, Magenta, Yellow.
• **Luminance Lock**: Operations are strictly chromatic. Lightness (L) is mathematically
  frozen to preserve the structural work done by HMS and Revela.
• **Shadow Authority**: A "Neutrality Lock" that prevents color grading from 
  tinting the background noise floor.
• **White Star Integrity**: Energy-based protection to keep stellar cores neutral
  even during aggressive saturation boosts.
• **Vector Scope**: Real-time visual feedback of color shifts and saturation gains
  on a holographic spectral wheel.

Usage
-----
1. **Input State**: Image MUST be Non-Linear (Stretched). Do not use on Linear data.
   Best used after HMS and Revela, before final cropping.
2. **Vectors**: 
   - Select "Primary" (RGB) or "Secondary" (CMY) tabs.
   - Use **Hue Shift** to rotate the color (e.g., make Red more Golden).
   - Use **Saturation** to boost intensity of that specific color.
3. **Protection**:
   - **Shadow Authority**: Increase to lock the background gray.
   - **Star Protect**: Keeps stars white.
4. **Process**: Applies the vector transformation.

Compatibility
-------------
• Siril 1.3+
• Python 3.10+ (via sirilpy)
• Dependencies: sirilpy, PyQt6, numpy, scipy

License
-------
Released under GPL-3.0-or-later.
"""

import sys
import os
import math
import traceback
import webbrowser

try:
    import sirilpy as s
    from sirilpy import LogColor
except ImportError:
    print("Error: sirilpy module not found.")
    sys.exit(1)

# installa PRIMA, importa DOPO
s.ensure_installed("numpy", "scipy", "PyQt6")

import numpy as np
from scipy.ndimage import convolve as nd_convolve

from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QLabel, QSlider, QPushButton, QGroupBox,
                             QMessageBox, QCheckBox, QProgressBar, QGraphicsView,
                             QGraphicsScene, QGraphicsPixmapItem, QFrame, QTabWidget)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSettings, QEvent, QPointF
from PyQt6.QtGui import QImage, QPixmap, QColor, QKeyEvent, QPainter, QConicalGradient, QPen, QBrush, QRadialGradient

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

QTabWidget::pane { border: 1px solid #444444; }
QTabBar::tab { background: #3c3c3c; color: #aaaaaa; padding: 8px 12px; margin-right: 2px; border-top-left-radius: 4px; border-top-right-radius: 4px; }
QTabBar::tab:selected { background: #2b2b2b; color: #ffffff; border-bottom: 2px solid #88aaff; }
QTabBar::tab:hover { background: #444444; }

/* Sliders */
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

/* Vector Colors for Sliders handles */
QSlider#RedS::handle:horizontal { background-color: #ff5555; }
QSlider#GreenS::handle:horizontal { background-color: #55ff55; }
QSlider#BlueS::handle:horizontal { background-color: #5555ff; }
QSlider#CyanS::handle:horizontal { background-color: #00ffff; }
QSlider#MagS::handle:horizontal { background-color: #ff00ff; }
QSlider#YelS::handle:horizontal { background-color: #ffff00; }

QPushButton { background-color: #444444; color: #dddddd; border: 1px solid #666666; border-radius: 4px; padding: 6px; font-weight: bold;}
QPushButton:hover { background-color: #555555; border-color: #777777; }
QPushButton#ProcessButton { background-color: #285299; border: 1px solid #1e3f7a; }
QPushButton#ProcessButton:hover { background-color: #355ea1; }
QPushButton#CloseButton { background-color: #5a2a2a; border: 1px solid #804040; }
QPushButton#CloseButton:hover { background-color: #7a3a3a; }

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

QPushButton#CoffeeButton { background-color: transparent; border: none; font-size: 15pt; padding: 2px; margin-right: 2px; }
QPushButton#CoffeeButton:hover { background-color: rgba(255,255,255,20); border-radius: 4px; }
"""

VERSION = "1.0.3"
# ------------------------------------------------------------------------------
# VERSION HISTORY
# ------------------------------------------------------------------------------
# 1.0.3: "Buy me a coffee" button added.
# 1.0.2: Fix float input normalization to avoid no-op on small-range HDR floats; 
#        improve Shadow Authority masking (low-percentile baseline) and 
#        chroma gating (luminance-relative + weak-chroma assist) to better reach
#        faint chromatic structures; soften global mask to avoid killing low-L color;
#        set Shadow Authority default to 0.
# 1.0.1: Ensure SciPy is installed before import to prevent startup failure.
# ------------------------------------------------------------------------------

# =============================================================================
#  CORE MATH: LCH VECTOR ENGINE
# =============================================================================

class VectraCore:
    
    @staticmethod
    def normalize_input(img_data):
        input_dtype = img_data.dtype
        img_float = img_data.astype(np.float32)
        if np.issubdtype(input_dtype, np.integer):
            if input_dtype == np.uint8: return img_float / 255.0
            elif input_dtype == np.uint16: return img_float / 65535.0
            else: return img_float / float(np.iinfo(input_dtype).max)
        elif np.issubdtype(input_dtype, np.floating):
            current_max = float(np.max(img_data))
            # Already normalized
            if current_max <= 1.0 + 1e-5:
                return img_float
            # If this looks like true 16-bit-scaled float data, normalize as 16-bit.
            # Otherwise, assume it's a small-range HDR float and normalize to [0,1].
            if current_max > 256.0:
                return img_float / 65535.0
            return img_float / (current_max + 1e-12)
        return img_float

    @staticmethod
    def rgb_to_lab(rgb):
        """RGB to LAB."""
        # Matrix for sRGB D65
        M = np.array([[0.4124564, 0.3575761, 0.1804375],
                      [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
        rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
        xyz = rgb.reshape(-1, 3) @ M.T
        xyz = xyz.reshape(rgb.shape)
        # Reference White D65
        xyz[..., 0] /= 0.95047; xyz[..., 2] /= 1.08883
        
        delta = 6/29
        def f(t): return np.where(t > delta**3, np.cbrt(t), (t/(3*delta**2)) + (4/29))
        
        fx, fy, fz = f(xyz[..., 0]), f(xyz[..., 1]), f(xyz[..., 2])
        L = 116*fy - 16; a = 500*(fx - fy); b = 200*(fy - fz)
        return np.stack([L, a, b], axis=-1)

    @staticmethod
    def lab_to_rgb(lab):
        """LAB to RGB."""
        M_inv = np.array([[ 3.2404542, -1.5371385, -0.4985314],
                          [-0.9692660,  1.8760108,  0.0415560],
                          [ 0.0556434, -0.2040259,  1.0572252]], dtype=np.float32)
        delta = 6/29
        fy = (lab[..., 0] + 16.0) / 116.0
        fx = fy + lab[..., 1] / 500.0
        fz = fy - lab[..., 2] / 200.0
        def finv(t): return np.where(t > delta, t**3, 3*delta**2*(t - 4/29))
        
        X = 0.95047 * finv(fx); Y = finv(fy); Z = 1.08883 * finv(fz)
        xyz = np.stack([X, Y, Z], axis=-1)
        rgb = xyz.reshape(-1, 3) @ M_inv.T
        rgb = rgb.reshape(xyz.shape)
        return np.clip(rgb, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def atrous_decomposition(img2d, n_scales=3):
        """Simple ATWT for Star Protection analysis (Energy check on fine scales)."""
        kernel = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16.0
        current = img2d.astype(np.float32)
        planes = []
        for s in range(n_scales):
            step = 2 ** s
            k_len = len(kernel) + (len(kernel) - 1) * (step - 1)
            k_dilated = np.zeros(k_len, dtype=np.float32)
            k_dilated[0::step] = kernel
            tmp = nd_convolve(current, k_dilated.reshape(1, -1), mode='reflect')
            smooth = nd_convolve(tmp, k_dilated.reshape(-1, 1), mode='reflect')
            detail = current - smooth
            planes.append(detail)
            current = smooth
        return planes

    @staticmethod
    def compute_signal_mask(L, threshold_sigma):
        """Shadow Authority: Returns 0 for background, 1 for signal."""
        L_norm = L / 100.0
        median = np.median(L_norm)
        mad = np.median(np.abs(L_norm - median))
        sigma = 1.4826 * mad
        if sigma < 1e-6: sigma = 1e-6
        # Use a low-percentile baseline so "signal_mask" doesn't collapse when the median is not true background.
        bg = np.percentile(L_norm, 25.0)
        noise_floor = bg + (threshold_sigma * sigma)
        mask = (L_norm - noise_floor) / (2.0 * sigma + 1e-9)
        mask = np.clip(mask, 0.0, 1.0)
        mask = nd_convolve(mask, np.ones((3,3))/9.0, mode='reflect')
        return mask

    @staticmethod
    def compute_star_protection(L):
        """Energy-based Star Protection."""
        planes = VectraCore.atrous_decomposition(L, n_scales=2)
        energy = np.abs(planes[0]) + np.abs(planes[1])
        # Heuristic threshold for star energy
        star_map = np.clip((energy - 1.5) * 0.5, 0.0, 1.0)
        protection = 1.0 - star_map
        return np.clip(protection, 0.0, 1.0)

    @staticmethod
    def process_vectors(rgb_img_hwc, vectors, shadow_auth, protect_stars):
        """
        Main Engine. Expects H,W,C.
        """
        # 1. Convert to Lab
        lab = VectraCore.rgb_to_lab(rgb_img_hwc)
        L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
        
        # 2. Convert to LCH (Polar)
        C = np.sqrt(a**2 + b**2)
        H_rad = np.arctan2(b, a)
        H_deg = np.degrees(H_rad) % 360.0
        
        # 3. Calculate Global Protection Mask
        sigma_thresh = shadow_auth / 20.0
        signal_mask = VectraCore.compute_signal_mask(L, sigma_thresh)
        
        star_mask = 1.0
        if protect_stars:
            star_mask = VectraCore.compute_star_protection(L)
            
        # Chroma gate must not kill low-L but strongly chromatic regions (e.g., faint Ha).
        # Use a luminance-relative chroma proxy instead of an absolute C threshold.
        C_rel = C / (L + 1.0)
        chroma_stability = np.clip((C_rel - 0.015) / 0.07, 0.0, 1.0)
        # Weak-chroma assist: allow vector grading on very low-chroma signal (e.g., faint OIII in M31 arms)
        # without tinting neutral background noise. The assist is gated by signal_mask, so it vanishes on true background.
        assist = 0.25 * np.clip((signal_mask - 0.10) / 0.30, 0.0, 1.0)
        chroma_stability = np.maximum(chroma_stability, assist)
        
        # Combine protections without fully killing low-L but chromatic regions.
        # If chroma is strong (e.g., faint Ha), allow a small floor even when signal_mask is low.
        # This preserves background neutrality because chroma_stability ~0 in neutral dark noise.
        global_mask = star_mask * chroma_stability * (0.15 + 0.85 * signal_mask)
        
        # 4. Vector Manipulation
        targets = {
            'R': 0.0, 'Y': 60.0, 'G': 120.0, 
            'C': 180.0, 'B': 240.0, 'M': 300.0
        }
        
        total_H_shift = np.zeros_like(H_deg)
        total_S_gain = np.zeros_like(C)
        sigma_ang = 30.0
        
        for color_key, (hue_shift, sat_boost) in vectors.items():
            if hue_shift == 0 and sat_boost == 0: continue
            
            tgt = targets[color_key]
            
            diff = np.abs(H_deg - tgt)
            diff = np.minimum(diff, 360.0 - diff)
            
            weight = np.exp(-(diff**2) / (2 * sigma_ang**2))
            
            total_H_shift += (hue_shift * weight)
            total_S_gain += (sat_boost * weight)

        # 5. Apply Changes
        final_H_deg = H_deg + (total_H_shift * global_mask)
        final_C = C * (1.0 + (total_S_gain * global_mask))
        final_C = np.maximum(0.0, final_C)
        
        # 6. Back to LAB
        H_rad_new = np.radians(final_H_deg)
        a_new = final_C * np.cos(H_rad_new)
        b_new = final_C * np.sin(H_rad_new)
        
        # Reassemble with ORIGINAL Luminance (Luminance Lock)
        lab_new = np.stack([L, a_new, b_new], axis=-1)
        
        # 7. Back to RGB
        rgb_out = VectraCore.lab_to_rgb(lab_new)
        
        return rgb_out

# =============================================================================
#  WORKER THREAD
# =============================================================================

class VectraWorker(QThread):
    result_ready = pyqtSignal(object)
    
    def __init__(self, img_chw, vectors, auth, stars):
        super().__init__()
        self.img = img_chw
        self.v = vectors
        self.a = auth
        self.s = stars
        
    def run(self):
        try:
            # Transpose to HWC for math
            img_hwc = self.img.transpose(1, 2, 0)
            res_hwc = VectraCore.process_vectors(img_hwc, self.v, self.a, self.s)
            # Back to CHW
            res_chw = res_hwc.transpose(2, 0, 1)
            self.result_ready.emit(res_chw)
        except Exception as e:
            print(f"Worker Error: {e}")
            self.result_ready.emit(None)

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

class FitGraphicsView(QGraphicsView):
    def __init__(self, scene, on_double_click=None, parent=None):
        super().__init__(scene, parent)
        self._on_double_click = on_double_click
    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and callable(self._on_double_click):
            self._on_double_click(); event.accept()
            return
        super().mouseDoubleClickEvent(event)

class VectorScopeOverlay(QWidget):
    """
    HUD Overlay displaying real-time 6-axis vectors.
    Draws a hue circle and dots representing current shift/sat.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(220, 220)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.vectors = {}
        self.is_visible = False
        
    def set_vectors(self, vectors):
        self.vectors = vectors
        self.update()

    def set_visibility(self, visible):
        self.is_visible = visible
        self.setVisible(visible)

    def paintEvent(self, event):
        if not self.is_visible: return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w, h = self.width(), self.height()
        cx, cy = w/2, h/2
        radius = min(w, h) / 2 - 20
        
        # Background Box (Semi-transparent)
        painter.setBrush(QColor(0, 0, 0, 180))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(0, 0, w, h, 10, 10)
        
        # --- THE RAINBOW WHEEL BACKGROUND ---
        # QConicalGradient starts at 0 deg (3 o'clock). 
        # Standard HSV color map:
        # Red=0, Yellow=60, Green=120, Cyan=180, Blue=240, Magenta=300, Red=360
        # Since Qt Conical is Counter-Clockwise, this maps perfectly.
        
        gradient = QConicalGradient(cx, cy, 0)
        
        # Create Stops: Hue 0 to 1 (0 to 360 deg)
        # Use lower alpha for "photographic/holographic" look
        alpha = 100 # out of 255
        gradient.setColorAt(0.000, QColor(255, 0, 0, alpha))    # Red
        gradient.setColorAt(0.166, QColor(255, 255, 0, alpha))  # Yellow
        gradient.setColorAt(0.333, QColor(0, 255, 0, alpha))    # Green
        gradient.setColorAt(0.500, QColor(0, 255, 255, alpha))  # Cyan
        gradient.setColorAt(0.666, QColor(0, 0, 255, alpha))    # Blue
        gradient.setColorAt(0.833, QColor(255, 0, 255, alpha))  # Magenta
        gradient.setColorAt(1.000, QColor(255, 0, 0, alpha))    # Red loop
        
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(cx, cy), radius, radius)
        
        # --- GRID & GUIDES ---
        # Outer Ring
        painter.setPen(QPen(QColor(150, 150, 150, 100), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(cx, cy), radius, radius)
        
        # Inner Ring (Neutral zone)
        painter.setPen(QPen(QColor(100, 100, 100, 50), 1))
        painter.drawEllipse(QPointF(cx, cy), radius * 0.6, radius * 0.6)

        # Crosshair
        painter.setPen(QPen(QColor(80, 80, 80, 100), 1))
        painter.drawLine(int(cx - radius), int(cy), int(cx + radius), int(cy))
        painter.drawLine(int(cx), int(cy - radius), int(cx), int(cy + radius))

        # --- VECTORS (DOTS) ---
        # Base angles
        base_angles = {
            'R': 0.0, 'Y': 60.0, 'G': 120.0, 
            'C': 180.0, 'B': 240.0, 'M': 300.0
        }
        
        # Solid colors matching sliders
        colors = {
            'R': QColor(255, 80, 80), 'Y': QColor(255, 255, 0), 'G': QColor(80, 255, 80),
            'C': QColor(0, 255, 255), 'B': QColor(80, 80, 255), 'M': QColor(255, 0, 255)
        }
        
        if not self.vectors: return
        
        for key, (h_shift, s_boost) in self.vectors.items():
            base_deg = base_angles[key]
            
            # Apply shifts
            final_deg = base_deg + h_shift
            
            # Visual Radius Mapping:
            # 0.0 saturation slider -> 0.6 radius (Neutral circle)
            # +1.0 saturation -> 1.0 radius (Edge)
            # -1.0 saturation -> 0.2 radius (Center)
            r_factor = 0.6 + (s_boost * 0.4) 
            final_r = radius * r_factor
            
            # Math to Screen Coords (Standard trig, Qt Y is down, so -sin)
            rad = math.radians(final_deg)
            px = cx + final_r * math.cos(rad)
            py = cy - final_r * math.sin(rad) 
            
            # Draw Line from Center
            painter.setPen(QPen(QColor(200, 200, 200, 150), 1, Qt.PenStyle.DotLine))
            painter.drawLine(int(cx), int(cy), int(px), int(py))
            
            # Draw Dot
            painter.setPen(QPen(QColor(0,0,0), 1))
            painter.setBrush(colors[key])
            painter.drawEllipse(QPointF(px, py), 4, 4)

# =============================================================================
#  MAIN GUI
# =============================================================================

class VectraGUI(QMainWindow):
    def __init__(self, siril, app):
        super().__init__()
        self.siril = siril
        self.app = app
        self.setWindowTitle(f"VeraLux Vectra v{VERSION}")
        self.setStyleSheet(DARK_STYLESHEET)
        self.resize(1350, 650)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)
        
        self.settings = QSettings("VeraLux", "Vectra")
        
        self.img_full = None
        self.img_proxy = None
        self.img_processed = None
        self.show_original = False
        
        self.pending_view_op = "FIT"
        self._auto_fit = True 
        
        self.debounce = QTimer()
        self.debounce.setSingleShot(True); self.debounce.setInterval(200)
        self.debounce.timeout.connect(self.update_preview)
        
        # Header Log
        header_msg = (
            "\n##############################################\n"
            "# VeraLux — Vectra\n"
            "# Vector Color Grading Engine\n"
            "# Author: Riccardo Paterniti (2025)\n"
            "##############################################"
        )
        try: self.siril.log(header_msg)
        except: print(header_msg)
        
        self.init_ui()
        self.cache_input()
        
    def init_ui(self):
        main = QWidget(); self.setCentralWidget(main)
        layout = QHBoxLayout(main)
        
        # --- LEFT PANEL ---
        left_container = QWidget(); left_container.setFixedWidth(400)
        left = QVBoxLayout(left_container); left.setContentsMargins(0,0,0,0)
        
        # Header
        lbl_title = QLabel("VeraLux Vectra")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_title.setStyleSheet("font-size: 16pt; font-weight: bold; color: #88aaff; margin-top: 5px;")
        left.addWidget(lbl_title)
        
        lbl_sub = QLabel("Vector Color Grading (LCH Space)")
        lbl_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_sub.setStyleSheet("font-size: 10pt; color: #999999; font-style: italic; margin-bottom: 15px;")
        left.addWidget(lbl_sub)
        
        # TABS for Colors
        self.tabs = QTabWidget(); self.tabs.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        left.addWidget(self.tabs)
        
        # Tab 1: Primaries (RGB)
        tab_pri = QWidget(); l_pri = QVBoxLayout(tab_pri)
        self.sl_R = self.create_vector_control("Red Vector (0°)", "RedS", l_pri)
        self.sl_G = self.create_vector_control("Green Vector (120°)", "GreenS", l_pri)
        self.sl_B = self.create_vector_control("Blue Vector (240°)", "BlueS", l_pri)
        l_pri.addStretch()
        self.tabs.addTab(tab_pri, "Primary Vectors")
        
        # Tab 2: Secondaries (CMY)
        tab_sec = QWidget(); l_sec = QVBoxLayout(tab_sec)
        self.sl_Y = self.create_vector_control("Yellow Vector (60°)", "YelS", l_sec)
        self.sl_C = self.create_vector_control("Cyan Vector (180°)", "CyanS", l_sec)
        self.sl_M = self.create_vector_control("Magenta Vector (300°)", "MagS", l_sec)
        l_sec.addStretch()
        self.tabs.addTab(tab_sec, "Secondary Vectors")
        
        # Protection
        g_prot = QGroupBox("Protection (Neutrality Lock)")
        l_prot = QVBoxLayout(g_prot)
        
        l_prot.addWidget(QLabel("Shadow Authority (Background Lock):"))
        self.s_shad = ResetSlider(Qt.Orientation.Horizontal, 0); self.s_shad.setRange(0, 100); self.s_shad.setValue(0); self.s_shad.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.s_shad.setToolTip(
            "<b>Background Neutrality Lock</b><br>"
            "Uses robust statistics (MAD) to identify and lock the background noise floor.<br>"
            "Increase this if the dark areas start taking on a color tint."
        )
        self.s_shad.valueChanged.connect(self.trigger_update)
        l_prot.addWidget(self.s_shad)
        
        self.chk_star = QCheckBox("White Star Integrity (Energy Protection)"); self.chk_star.setChecked(True); self.chk_star.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.chk_star.setToolTip(
            "<b>Star Protection</b><br>"
            "Analyzes local energy to identify stars.<br>"
            "Prevents white stellar cores from being tinted by aggressive Hue/Saturation shifts."
        )
        self.chk_star.toggled.connect(self.trigger_update)
        l_prot.addWidget(self.chk_star)
        left.addWidget(g_prot)
        
        left.addStretch()
        
        # Footer
        footer = QHBoxLayout()
        self.btn_help = QPushButton("?"); self.btn_help.setObjectName("HelpButton"); self.btn_help.setFixedWidth(20); self.btn_help.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_help.clicked.connect(self.print_help)
        self.btn_help.setToolTip("Print Operational Guide to Siril Console")
        footer.addWidget(self.btn_help)
        
        b_def = QPushButton("Defaults"); b_def.clicked.connect(self.set_defaults); b_def.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        b_def.setToolTip("Reset all vectors and protection settings to default.")
        footer.addWidget(b_def)
        
        b_cls = QPushButton("Close"); b_cls.setObjectName("CloseButton"); b_cls.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        b_cls.clicked.connect(self.close)
        b_cls.setToolTip("Close the tool without applying changes.")
        footer.addWidget(b_cls)
        
        b_proc = QPushButton("PROCESS"); b_proc.setObjectName("ProcessButton"); b_proc.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        b_proc.clicked.connect(self.apply_process)
        b_proc.setToolTip("Apply the color grading to the full-resolution image and save to Siril.")
        footer.addWidget(b_proc)
        left.addLayout(footer)
        
        layout.addWidget(left_container)
        
        # --- RIGHT PANEL ---
        right_widget = QWidget(); right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0,0,0,0); right_layout.setSpacing(0)
        
        # Toolbar
        tb_w = QWidget(); tb = QHBoxLayout(tb_w); tb.setContentsMargins(5,5,5,5)
        b_out = QPushButton("-"); b_out.setObjectName("ZoomBtn"); b_out.clicked.connect(self.zoom_out); b_out.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        b_out.setToolTip("Zoom Out")
        b_fit = QPushButton("Fit"); b_fit.setObjectName("ZoomBtn"); b_fit.clicked.connect(self.fit_view); b_fit.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        b_fit.setToolTip("Fit image to window")
        b_11 = QPushButton("1:1"); b_11.setObjectName("ZoomBtn"); b_11.clicked.connect(self.zoom_1to1); b_11.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        b_11.setToolTip("View actual pixels (100%)")
        b_in = QPushButton("+"); b_in.setObjectName("ZoomBtn"); b_in.clicked.connect(self.zoom_in); b_in.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        b_in.setToolTip("Zoom In")
        
        lbl_h = QLabel("Hold SPACE to Compare / Double-click to Fit")
        lbl_h.setStyleSheet("color: #ffb000; font-size: 8pt; font-style: italic; margin-left: 10px; font-weight: bold;")
        
        # Vector Scope Checkbox
        self.chk_scope = QCheckBox("Show Vector Scope"); self.chk_scope.setChecked(False); self.chk_scope.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.chk_scope.setToolTip("Show/Hide the 6-Axis LCH Vector HUD")
        self.chk_scope.toggled.connect(self.toggle_scope)
        
        self.btn_coffee = QPushButton("☕")
        self.btn_coffee.setObjectName("CoffeeButton")
        self.btn_coffee.setToolTip("Support the developer - Buy me a coffee!")
        self.btn_coffee.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_coffee.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_coffee.clicked.connect(lambda: webbrowser.open("https://buymeacoffee.com/riccardo.paterniti"))

        chk_ot = QCheckBox("On Top"); chk_ot.setChecked(True); chk_ot.toggled.connect(self.toggle_ontop); chk_ot.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        chk_ot.setToolTip("Keep window above Siril")
        
        tb.addWidget(b_out); tb.addWidget(b_fit); tb.addWidget(b_11); tb.addWidget(b_in); tb.addWidget(lbl_h)
        tb.addStretch()
        tb.addWidget(self.chk_scope)
        tb.addSpacing(15)
        tb.addWidget(chk_ot)
        tb.addWidget(self.btn_coffee)
        right_layout.addWidget(tb_w)
        
        # View
        self.scene = QGraphicsScene()
        self.view = FitGraphicsView(self.scene, on_double_click=self.fit_view)
        self.view.setStyleSheet("background-color: #151515; border: none;")
        self.view.setFrameShape(QFrame.Shape.NoFrame)
        self.view.setContentsMargins(0,0,0,0)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.view.setVisible(False)
        
        right_layout.addWidget(self.view)
        layout.addWidget(right_widget)
        
        self.pix_item = QGraphicsPixmapItem(); self.scene.addItem(self.pix_item)
        self.lbl_blink = QLabel("ORIGINAL", self.view)
        self.lbl_blink.setStyleSheet("background-color: rgba(255, 160, 0, 200); color: white; font-weight: bold; padding: 8px; border-radius: 4px;")
        self.lbl_blink.hide()
        
        # VECTOR SCOPE OVERLAY
        self.vector_scope = VectorScopeOverlay(self.view)
        self.vector_scope.setVisible(False)

    def create_vector_control(self, title, obj_name, layout):
        g = QGroupBox(title)
        v = QVBoxLayout(g)
        
        h_hue = QHBoxLayout()
        h_hue.addWidget(QLabel("Hue Shift:"))
        s_hue = ResetSlider(Qt.Orientation.Horizontal, 0); s_hue.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        s_hue.setRange(-60, 60); s_hue.setValue(0); s_hue.setObjectName(obj_name)
        s_hue.valueChanged.connect(self.trigger_update)
        s_hue.valueChanged.connect(self.update_scope_ui)
        s_hue.setToolTip(
            "<b>Hue Rotation</b><br>"
            "Shifts the tonal angle of this color vector.<br>"
            "Useful for correcting Green to Teal (SHO) or Red to Gold."
        )
        h_hue.addWidget(s_hue)
        v.addLayout(h_hue)
        
        h_sat = QHBoxLayout()
        h_sat.addWidget(QLabel("Saturation:"))
        s_sat = ResetSlider(Qt.Orientation.Horizontal, 0); s_sat.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        s_sat.setRange(-100, 100); s_sat.setValue(0); s_sat.setObjectName(obj_name)
        s_sat.valueChanged.connect(self.trigger_update)
        s_sat.valueChanged.connect(self.update_scope_ui)
        s_sat.setToolTip(
            "<b>Chroma Boost</b><br>"
            "Amplifies or reduces the vividness of this color.<br>"
            "Use to pop specific OIII/Ha regions."
        )
        h_sat.addWidget(s_sat)
        v.addLayout(h_sat)
        
        layout.addWidget(g)
        return (s_hue, s_sat)

    # --- LOGIC ---
    def get_vectors(self):
        return {
            'R': (self.sl_R[0].value(), self.sl_R[1].value()/100.0),
            'G': (self.sl_G[0].value(), self.sl_G[1].value()/100.0),
            'B': (self.sl_B[0].value(), self.sl_B[1].value()/100.0),
            'C': (self.sl_C[0].value(), self.sl_C[1].value()/100.0),
            'M': (self.sl_M[0].value(), self.sl_M[1].value()/100.0),
            'Y': (self.sl_Y[0].value(), self.sl_Y[1].value()/100.0),
        }


    def _to_chw_rgb(self, arr):
        a = np.asarray(arr)
        if a.ndim == 2:
            return np.stack([a, a, a], axis=0)
        if a.ndim != 3:
            return None
        # HWC -> CHW if needed
        if a.shape[2] in (1, 3, 4) and a.shape[0] not in (1, 3, 4):
            a = a.transpose(2, 0, 1)
        # Drop alpha / expand mono
        if a.shape[0] == 4:
            a = a[:3, :, :]
        elif a.shape[0] == 1:
            a = np.repeat(a, 3, axis=0)
        return a

    def _safe_get_image_pixeldata_locked(self):
        img = None
        err = None
        try:
            img = self.siril.get_image_pixeldata()
        except Exception as e:
            err = e
            img = None

        if img is not None:
            return self._to_chw_rgb(img)

        icc = None
        try:
            icc = self.siril.get_image_iccprofile()
        except Exception:
            icc = None

        # FFit fallback when SHM pixeldata is unavailable
        if icc:
            try:
                self.siril.log("Vectra: ICC detected — using FFit fallback.", LogColor.YELLOW)
            except Exception:
                pass
        else:
            try:
                self.siril.log("Vectra: pixeldata unavailable — trying FFit fallback.", LogColor.YELLOW)
            except Exception:
                pass

        try:
            ffit = self.siril.get_image(with_pixels=True)
            data = getattr(ffit, "data", None)
            if data is not None:
                return self._to_chw_rgb(data)
        except Exception as e2:
            try:
                self.siril.log(f"Vectra: FFit fallback failed: {type(e2).__name__}: {e2}", LogColor.RED)
            except Exception:
                pass

        try:
            if err is not None:
                self.siril.log(f"Vectra: get_image_pixeldata() failed: {type(err).__name__}: {err}", LogColor.RED)
            else:
                self.siril.log("Vectra: get_image_pixeldata() returned None.", LogColor.RED)
        except Exception:
            pass
        return None

    def cache_input(self):
        try:
            if not self.siril.connected: self.siril.connect()
            with self.siril.image_lock():
                img = self._safe_get_image_pixeldata_locked()
            if img is None:
                QMessageBox.warning(self, "Vectra — Image read error",
                                    "Unable to read pixel data from the current image.\n"
                                    "If the file has an embedded ICC profile, export without ICC (or use FITS/XISF).")
                return
            img = VectraCore.normalize_input(img)
            # Ensure C,H,W
            if img.ndim == 2: img = np.stack([img, img, img], axis=0)
            self.img_full = img 
            
            # Proxy
            h, w = self.img_full.shape[1], self.img_full.shape[2]
            scale = 1600 / max(h, w)
            if scale < 1.0:
                step = int(1/scale)
                self.img_proxy = self.img_full[:, ::step, ::step].copy()
            else:
                self.img_proxy = self.img_full.copy()
            
            self.img_processed = None
            self.pending_view_op = "FIT"
            self.update_view_image()
            self.trigger_update()
        except: pass

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self.final_init_view)

    def final_init_view(self):
        if self.img_proxy is not None:
            self.view.setVisible(True)
            self.fit_view()
            self.update_overlays()

    def trigger_update(self): self.debounce.start()
    
    def update_scope_ui(self):
        if self.chk_scope.isChecked():
            v = self.get_vectors()
            self.vector_scope.set_vectors(v)

    def update_preview(self):
        if self.img_proxy is None: return
        vectors = self.get_vectors()
        auth = self.s_shad.value()
        stars = self.chk_star.isChecked()
        
        self.worker = VectraWorker(self.img_proxy, vectors, auth, stars)
        self.worker.result_ready.connect(self.set_result)
        self.worker.start()

    def set_result(self, res):
        self.img_processed = res
        self.update_view_image()

    def update_view_image(self):
        data = self.img_proxy if self.show_original else (self.img_processed if self.img_processed is not None else self.img_proxy)
        if data is None: return
        self.lbl_blink.setVisible(self.show_original)
        
        # Display prep: Input is CHW -> Convert to HWC for QImage -> FlipUD
        if data.ndim == 3:
            disp = np.clip(data * 255, 0, 255).astype(np.uint8)
            disp = np.ascontiguousarray(np.flipud(disp.transpose(1, 2, 0)))
        else:
            disp = np.zeros((100,100,3), dtype=np.uint8)

        h, w, c = disp.shape
        qimg = QImage(disp.data, w, h, c*w, QImage.Format.Format_RGB888)
        self.pix_item.setPixmap(QPixmap.fromImage(qimg))
        self.scene.setSceneRect(0, 0, w, h)
        
        bw = self.lbl_blink.width()
        self.lbl_blink.move((self.view.width()-bw)//2, 10)
        
        if self.pending_view_op == "FIT":
            QTimer.singleShot(0, self.fit_view); self.pending_view_op = None

    def apply_process(self):
        if self.img_full is None: return
        self.setEnabled(False)
        try:
            vectors = self.get_vectors()
            auth = self.s_shad.value()
            stars = self.chk_star.isChecked()
            
            # Full Process (CHW Input)
            full_hwc = self.img_full.transpose(1, 2, 0)
            res_hwc = VectraCore.process_vectors(full_hwc, vectors, auth, stars)
            out = res_hwc.transpose(2, 0, 1)
            
            with self.siril.image_lock():
                self.siril.undo_save_state("VeraLux Vectra")
                self.siril.set_image_pixeldata(out.astype(np.float32))
            self.siril.log("VeraLux Vectra applied.", LogColor.GREEN)
            self.close()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
        finally: self.setEnabled(True)

    def set_defaults(self):
        for s in [self.sl_R, self.sl_G, self.sl_B, self.sl_C, self.sl_M, self.sl_Y]:
            s[0].setValue(0); s[1].setValue(0)
        self.s_shad.setValue(0); self.chk_star.setChecked(True)
        self.update_scope_ui()

    # --- INPUT ---
    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Space and not e.isAutoRepeat():
            self.show_original = True; self.update_view_image()
    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key.Key_Space and not e.isAutoRepeat():
            self.show_original = False; self.update_view_image()
            
    # Added Auto Fit Logic to ResizeEvent
    def resizeEvent(self, e):
        self.update_view_image()
        self.update_overlays()
        if getattr(self, '_auto_fit', False):
            self.fit_view()
        super().resizeEvent(e)

    # --- VIEWPORT ---
    def toggle_ontop(self, c):
        flags = self.windowFlags()
        if c: flags |= Qt.WindowType.WindowStaysOnTopHint
        else: flags &= ~Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags); self.show()
        
    def zoom_in(self): 
        self._auto_fit = False
        self.view.scale(1.2, 1.2)
        
    def zoom_out(self): 
        self._auto_fit = False
        self.view.scale(1/1.2, 1/1.2)
        
    def zoom_1to1(self): 
        self._auto_fit = False
        self.view.resetTransform()
        
    def fit_view(self): 
        self._auto_fit = True
        self.view.fitInView(self.pix_item, Qt.AspectRatioMode.KeepAspectRatio)
        
    def toggle_scope(self, checked):
        self.vector_scope.set_visibility(checked)
        if checked:
            self.update_scope_ui()

    def update_overlays(self):
        if self.vector_scope:
            w, h = self.view.width(), self.view.height()
            # Padding 10px from bottom/left
            # Size is 220x220
            self.vector_scope.move(10, h - 230)

    def print_help(self):
        msg = [
            "===========================================================",
            "   VERALUX VECTRA v1.0 — OPERATIONAL GUIDE",
            "   Vector Color Grading Engine (LCH)",
            "===========================================================",
            "",
            "OVERVIEW",
            "-----------------",
            "Vectra allows surgical manipulation of colors using 6 vectors",
            "(R, G, B, C, M, Y) in LCH space, while preserving luminance integrity.",
            "",
            "[1] IMPORTANT REQUIREMENTS",
            "    • Image State: MUST be Non-Linear (Stretched).",
            "      Do not apply to linear images.",
            "",
            "[2] VECTORS (Primary & Secondary Tabs)",
            "    • Hue Shift: Rotates the color angle.",
            "      - Example: Move 'Green' towards 'Cyan' (+Hue) for SHO style.",
            "    • Saturation: Boosts or mutes intensity.",
            "      - Example: Boost 'Blue' (+Sat) to pop OIII / teal regions.",
            "",
            "[3] VECTOR SCOPE (HUD)",
            "    • Enable 'Show Vector Scope' to see real-time feedback.",
            "    • Dots move radially (Saturation) and rotate (Hue).",
            "",
            "[4] PROTECTION",
            "    • Shadow Authority (Background Lock): Default is 0.",
            "      Increase ONLY if the dark background starts taking on a tint.",
            "      Tip: keep it low to reach very faint chromatic structures.",
            "    • White Star Integrity: Keeps bright stars neutral white.",
            "      Disable only if you want stars to follow the color grading.",
            "",
            "Support: info@veralux.space",
            "==========================================================="
        ]
        try:
            for l in msg: self.siril.log(l if l.strip() else " ")
        except: print("\n".join(msg))

def main():
    app = QApplication(sys.argv)
    siril = s.SirilInterface()
    try: siril.connect()
    except: pass
    gui = VectraGUI(siril, app)
    gui.show()
    app.exec()

if __name__ == "__main__":
    main()