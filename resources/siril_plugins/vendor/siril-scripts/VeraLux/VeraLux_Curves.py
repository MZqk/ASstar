##############################################
# VeraLux — Curves
# Spline-Based Photometric Sculpting Engine
# Author: Riccardo Paterniti (2025)
# Contact: info@veralux.space
##############################################
# (c) 2025 Riccardo Paterniti
# VeraLux — Curves
# SPDX-License-Identifier: GPL-3.0-or-later
# Version 1.0.1

# Credits / Origin
# ----------------
# • Engine: Akima Spline Interpolation (Scipy) & Vector Color Math
# • Architecture: VeraLux Shared GUI Framework
# • Zonal Control: Luminance-based range masking with soft roll-off

"""
Overview
--------
VeraLux Curves is a precision non-linear transformation engine designed to sculpt
the tones and chromatic intensity of an image using advanced spline interpolation.
Unlike standard curve tools that introduce "overshoot" artifacts (ringing) near
control points, Curves uses Akima Splines to ensure smooth, natural transitions
mimicking the response of physical film.

It features a "Split-Domain" architecture, allowing users to manipulate
Luminance (CIE L*), Chrominance (CIE LCH), and Saturation (HSV) independently
without cross-channel contamination.

Key Features
------------
• **Akima Spline Engine**: Natural oscillation-free interpolation.
• **Full Endpoint Freedom**:
  - Move the Black Point horizontally to clip shadows.
  - Move the White Point horizontally to clip highlights.
  - Move vertically to lift blacks or dampen whites (Matte effect).
• **Luminance Range Control**: Apply curves selectively to specific tonal zones
  (shadows/midtones/highlights) with ultra-smooth feathering for invisible transitions.
  Each channel can target independent luminance ranges without external masks.
• **Clipping Monitor**: Real-time feedback showing the percentage of pixels clipped
  to pure black or pure white.
• **Dynamic Histogram**: Real-time feedback showing how the curve reshapes the tonal
  distribution, with visual dimming of excluded luminance ranges.
• **Photometric Modes**:
  - **L (Luminance)**: Adjusts contrast in CIE L*a*b* space without shifting saturation.
  - **C (Chrominance)**: Boosts spectral intensity (CIE LCH) more naturally than digital saturation.
  - **S (Saturation)**: Mathematical HSV purity control.
• **Live Pipette**: Click anywhere on the image to locate the exact tonal value on the curve.
• **Detachable Graph**: Work on the curve in a separate, resizable window for maximum precision.

Usage
-----
1. **Select Channel**: Choose the target domain (RGB/K, R, G, B, L, C, S).
2. **Set Luminance Range** (Optional): 
   - Enable "Lum Range" checkbox to activate zone-based tonal control.
   - Toggle "Show Mask" to visualize the exact selection mask on the preview image.
   - Adjust Min/Max sliders to define the luminance range (0–100%).
   - Set Feather slider to control transition smoothness (default: 25%).
   - Double-click any slider to reset to default values.
   - Visual feedback: excluded zones appear grayed-out on grid and dimmed on histogram.
3. **Pipette**: Click on the preview image to find the tone you want to alter.
   A vertical line will appear on the graph.
4. **Sculpt**:
   - **Left Click**: Add points or drag existing ones.
   - **Right Click**: Remove points.
   - **Black Point**: Drag the bottom-left point horizontally to clip shadows.
   - **White Point**: Drag the top-right point horizontally to clip highlights.
5. **Visual Feedback**: 
   - Active channels display ● symbol in the channel list.
   - Range-limited channels show additional ⟨⟩ marker.
   - Excluded luminance zones appear grayed-out on the curve grid.
   - Histogram dims excluded regions for clear visual reference.
6. **Iterate & Apply**:
   - All edits are applied non-destructively on top of the last applied state.
   - Press **Apply** to freeze the current active curves as a new iteration stage.
   - Each applied stage is added to a linear stack, allowing step-by-step construction of complex processing.
   - After applying, all channels are reset to identity, ready for the next iteration.
7. **Undo**:
   - Press **Undo** to remove the most recent applied iteration.
   - The preview is automatically rebuilt from the remaining stage stack.
   - Undo affects only applied stages, never intermediate (uncommitted) edits.
8. **Compare**: Hold SPACEBAR to view the original image.
9. **Process**: Applies the full stage stack and current active curves to the full-resolution image.

Example Workflows
-----------------
• Boost Chrominance (C) only in deep shadows (0-30% luminance)
• Lift midtone Luminance (L) without affecting highlights (30-70%)
• Compress bright regions (70-100%) while preserving shadow detail
• Apply selective color grading to specific tonal zones per channel

Compatibility
-------------
• Siril 1.4+
• Python 3.10+
• Dependencies: sirilpy, PyQt6, numpy, scipy, opencv-python

Technical Details
-----------------
• **Interpolation**: Akima Splines (oscillation-free, C1-continuous)
• **Precision**: Full 32-bit float pipeline throughout
• **Range Masking Algorithm**: 
  - Sigmoid function with k=20.0 for precise roll-off
  - No spatial blur for maximum pixel-to-pixel fidelity
  - Computed on input image luminance (RGB K-average)
• **Feather Range**: 0-100% (recommended: 20-40% for invisible transitions)

License
-------
Released under GPL-3.0-or-later.

This script is part of the VeraLux family of tools —
Python utilities focused on physically faithful astrophotography workflows.
"""

import sys
import traceback
import copy
import webbrowser

import sirilpy as s
from sirilpy import LogColor

s.ensure_installed("numpy", "opencv-python", "scipy", "PyQt6")

import numpy as np
import cv2
from scipy.interpolate import Akima1DInterpolator

from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QLabel, QPushButton, QGroupBox,
                             QMessageBox, QCheckBox, QGraphicsView, QGraphicsScene,
                             QGraphicsPixmapItem, QRadioButton, QButtonGroup, QGridLayout,
                             QSlider)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QPointF, QRectF, QEvent
from PyQt6.QtGui import (QImage, QPixmap, QColor, QPainter, QPen, QBrush,
                        QPainterPath, QLinearGradient, QRadialGradient)

VERSION = "1.0.1"
# ------------------------------------------------------------------------------
# VERSION HISTORY
# ------------------------------------------------------------------------------
# 1.0.1: "Buy me a coffee" button added.
# ------------------------------------------------------------------------------

# =============================================================================
# STYLING & CONFIG
# =============================================================================

def _nofocus(w):
    try:
        w.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    except Exception:
        pass

DARK_STYLESHEET = """
QWidget{background-color:#2b2b2b;color:#e0e0e0;font-size:10pt}

QToolTip{background-color:#333333;color:#ffffff;border:1px solid #88aaff}

QGroupBox{border:1px solid #444444;margin-top:5px;font-weight:bold;border-radius:4px;padding-top:12px}
QGroupBox::title{subcontrol-origin:margin;left:8px;padding:0 3px;color:#88aaff}

QLabel{color:#cccccc}

QRadioButton,QCheckBox{color:#cccccc;spacing:5px}
QRadioButton::indicator,QCheckBox::indicator{width:14px;height:14px;border:1px solid #666666;background:#3c3c3c;border-radius:7px}
QCheckBox::indicator{border-radius:3px}
QRadioButton::indicator:checked{background:qradialgradient(cx:0.5,cy:0.5,radius:0.4,fx:0.5,fy:0.5,stop:0 #ffffff,stop:1 #285299);border:1px solid #88aaff;image:none}
QCheckBox::indicator:checked{background:#285299;border:1px solid #88aaff;image:none}
QRadioButton:hover{background-color:#333333;border-radius:4px}

QSlider{min-height:22px}
QSlider::groove:horizontal{background:#444444;height:6px;border-radius:3px}
QSlider::handle:horizontal{background-color:#aaaaaa;width:14px;height:14px;margin:-4px 0;border-radius:7px;border:1px solid #555555}
QSlider::handle:horizontal:hover{background-color:#ffffff;border:1px solid #888888}
QSlider::handle:horizontal:pressed{background-color:#ffffff;border:1px solid #dddddd}

QPushButton{background-color:#444444;color:#dddddd;border:1px solid #666666;border-radius:4px;padding:6px;font-weight:bold}
QPushButton:hover{background-color:#555555;border-color:#777777}
QPushButton#ProcessButton{background-color:#285299;border:1px solid #1e3f7a}
QPushButton#ProcessButton:hover{background-color:#355ea1}
QPushButton#CloseButton{background-color:#5a2a2a;border:1px solid #804040}
QPushButton#CloseButton:hover{background-color:#7a3a3a}
QPushButton#ApplyButton{background-color:#3b4540;border:1px solid #4c6355}
QPushButton#ApplyButton:hover{background-color:#445048;border-color:#5c7a67}
QPushButton#UndoButton{background-color:#45403b;border:1px solid #63554c}
QPushButton#UndoButton:hover{background-color:#504844;border-color:#7a675c}
QPushButton#ResetChannelButton{background-color:#3c3c3c;border:1px solid #666666}
QPushButton#ResetAllButton{background-color:#3c3c3c;border:1px solid #666666}
QPushButton#ResetChannelButton:hover,QPushButton#ResetAllButton:hover{background-color:#4a4a4a;border-color:#777777}
QPushButton#ZoomBtn{min-width:30px;font-weight:bold;background-color:#3c3c3c}
QPushButton#HelpButton{background-color:transparent;color:#555555;border:none;font-weight:bold;min-width:20px}
QPushButton#HelpButton:hover{color:#aaaaaa}
QPushButton#PopOutButton{background-color:#3c3c3c;color:#aaaaaa;border:1px solid #555555;border-radius:3px;font-size:10px;padding:2px}
QPushButton#PopOutButton:hover{background-color:#555555;color:#ffffff;border-color:#888888}

QPushButton#CoffeeButton{background-color:transparent;border:none;font-size:15pt;padding:2px;margin-right:2px}
QPushButton#CoffeeButton:hover{background-color:rgba(255,255,255,20);border-radius:4px}

QGraphicsView{border:none;background-color:#151515}
"""

# =============================================================================
# CORE MATH
# =============================================================================

class CurvesCore:
    
    @staticmethod
    def normalize_input(img_data):
        """Normalize input to 0.0-1.0 float32."""
        img = img_data.astype(np.float32, copy=False)
        
        if img_data.dtype == np.uint8:
            img = img / 255.0
        elif img_data.dtype == np.uint16:
            img = img / 65535.0
        elif img_data.dtype == np.int16:
            img = (img + 32768.0) / 65535.0
        
        mn = float(np.min(img))
        if mn < 0.0:
            img = np.maximum(img, 0.0)
        
        mx = float(np.max(img))
        if mx > 1.0 + 1e-4:
            p = float(np.percentile(img, 99.99))
            if p > 0.0:
                img = img / p
        
        return np.clip(img, 0.0, 1.0).astype(np.float32, copy=False)

    @staticmethod
    def ensure_hwc(img):
        """Ensures the image is in Height-Width-Channels format (H, W, C)."""
        if img.ndim == 2:
            return np.dstack([img, img, img])
        if img.ndim == 3:
            if img.shape[0] == 1 and img.shape[2] != 3:
                m = img[0, :, :]
                return np.dstack([m, m, m])
            if img.shape[2] == 1:
                m = img[:, :, 0]
                return np.dstack([m, m, m])
            if img.shape[2] == 3:
                return img
            if img.shape[0] == 3:
                return img.transpose(1, 2, 0)
        return img

    @staticmethod
    def generate_lut(points, size=65536):
        """Generates a high-precision LUT using Akima Spline Interpolation."""
        points = sorted(points, key=lambda p: p[0])
        
        cleaned_points = []
        last_x = -1.0
        for px, py in points:
            if px <= last_x:
                px = last_x + 1e-6
            px = 0.0 if px < 0.0 else (1.0 if px > 1.0 else px)
            cleaned_points.append((px, py))
            last_x = px
        
        x, y = zip(*cleaned_points)
        
        dom = np.linspace(0, 1, size, dtype=np.float32)
        lut = np.zeros_like(dom)
        
        min_x, max_x = x[0], x[-1]
        mask_inner = (dom >= min_x) & (dom <= max_x)
        
        if len(cleaned_points) > 2:
            try:
                interpolator = Akima1DInterpolator(x, y)
                lut[mask_inner] = interpolator(dom[mask_inner])
            except Exception:
                lut[mask_inner] = np.interp(dom[mask_inner], x, y)
        else:
            lut[mask_inner] = np.interp(dom[mask_inner], x, y)
        
        if min_x > 0.0:
            lut[dom < min_x] = y[0]
        if max_x < 1.0:
            lut[dom > max_x] = y[-1]
        
        return np.clip(lut, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def compute_luminance_mask(img, lum_min, lum_max, feather_sigma=0.25):
        """Compute a precise luminance-based mask with high pixel fidelity."""
        from scipy.special import expit
        
        lum = np.mean(img, axis=2)
        feather_width = feather_sigma
        
        if feather_width < 1e-6:
            mask = ((lum >= lum_min) & (lum <= lum_max)).astype(np.float32)
            return mask
        
        mask = np.ones_like(lum, dtype=np.float32)
        k_smooth = 2.5 # Sigmoid roll-off
        
        if lum_min > 0.0:
            dist_lower = (lum - lum_min) / feather_width
            lower_mask = expit(k_smooth * dist_lower)
            mask = np.minimum(mask, lower_mask)

        if lum_max < 1.0:
            dist_upper = (lum_max - lum) / feather_width
            upper_mask = expit(k_smooth * dist_upper)
            mask = np.minimum(mask, upper_mask)
                
        return np.clip(mask, 0.0, 1.0)

    @staticmethod
    def apply_pipeline(img, channel_states, input_luminance=None):
        """Applies the full curve pipeline to an RGB image."""
        res = img.astype(np.float32, copy=True)
        res = np.clip(res, 0.0, 1.0)
        
        if input_luminance is None:
            input_luminance = np.mean(img, axis=2)
        
        def apply_lut_fast(arr, lut):
            x_lut = np.linspace(0.0, 1.0, lut.size, dtype=np.float32)
            return np.interp(arr.ravel(), x_lut, lut).reshape(arr.shape)
        
        def apply_with_mask(data, lut, mask):
            if mask is None:
                return apply_lut_fast(data, lut).astype(np.float32, copy=False)
            else:
                original = data.copy()
                transformed = apply_lut_fast(data, lut)
                if data.ndim == 3:
                    mask_3d = mask[:, :, np.newaxis]
                    return (original * (1.0 - mask_3d) + transformed * mask_3d).astype(np.float32)
                else:
                    return (original * (1.0 - mask) + transformed * mask).astype(np.float32)
        
        if channel_states.get('RGB/K', {}).get('active', False):
            lut = channel_states['RGB/K'].get('lut', None)
            if lut is not None:
                mask = None
                if channel_states['RGB/K'].get('lum_range_enabled', False):
                    lum_min = channel_states['RGB/K'].get('lum_min', 0.0)
                    lum_max = channel_states['RGB/K'].get('lum_max', 1.0)
                    feather = channel_states['RGB/K'].get('feather_sigma', 0.25)
                    mask = CurvesCore.compute_luminance_mask(res, lum_min, lum_max, feather)
                
                res = apply_with_mask(res, lut, mask)
                res = np.clip(res, 0.0, 1.0)
        
        for i, c in enumerate(['R', 'G', 'B']):
            if channel_states.get(c, {}).get('active', False):
                lut = channel_states[c].get('lut', None)
                if lut is not None:
                    mask = None
                    if channel_states[c].get('lum_range_enabled', False):
                        lum_min = channel_states[c].get('lum_min', 0.0)
                        lum_max = channel_states[c].get('lum_max', 1.0)
                        feather = channel_states[c].get('feather_sigma', 0.25)
                        mask = CurvesCore.compute_luminance_mask(res, lum_min, lum_max, feather)
                    
                    res[:, :, i] = apply_with_mask(res[:, :, i], lut, mask)
        
        res = np.clip(res, 0.0, 1.0)
        
        if channel_states.get('L', {}).get('active', False):
            lut = channel_states['L'].get('lut', None)
            if lut is not None:
                mask = None
                if channel_states['L'].get('lum_range_enabled', False):
                    lum_min = channel_states['L'].get('lum_min', 0.0)
                    lum_max = channel_states['L'].get('lum_max', 1.0)
                    feather = channel_states['L'].get('feather_sigma', 0.25)
                    mask = CurvesCore.compute_luminance_mask(res, lum_min, lum_max, feather)
                
                lab = cv2.cvtColor(res, cv2.COLOR_RGB2Lab)
                L = lab[:, :, 0] / 100.0
                L_new = apply_with_mask(L, lut, mask)
                lab[:, :, 0] = np.clip(L_new, 0.0, 1.0) * 100.0
                res = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
                res = np.clip(res, 0.0, 1.0)
        
        if channel_states.get('S', {}).get('active', False):
            lut = channel_states['S'].get('lut', None)
            if lut is not None:
                mask = None
                if channel_states['S'].get('lum_range_enabled', False):
                    lum_min = channel_states['S'].get('lum_min', 0.0)
                    lum_max = channel_states['S'].get('lum_max', 1.0)
                    feather = channel_states['S'].get('feather_sigma', 0.25)
                    mask = CurvesCore.compute_luminance_mask(res, lum_min, lum_max, feather)
                
                hsv = cv2.cvtColor(res, cv2.COLOR_RGB2HSV)
                S = hsv[:, :, 1]
                S_new = apply_with_mask(S, lut, mask)
                hsv[:, :, 1] = np.clip(S_new, 0.0, 1.0)
                res = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                res = np.clip(res, 0.0, 1.0)
        
        if channel_states.get('C', {}).get('active', False):
            lut = channel_states['C'].get('lut', None)
            if lut is not None:
                mask = None
                if channel_states['C'].get('lum_range_enabled', False):
                    lum_min = channel_states['C'].get('lum_min', 0.0)
                    lum_max = channel_states['C'].get('lum_max', 1.0)
                    feather = channel_states['C'].get('feather_sigma', 0.25)
                    mask = CurvesCore.compute_luminance_mask(res, lum_min, lum_max, feather)
                
                lab = cv2.cvtColor(res, cv2.COLOR_RGB2Lab)
                a = lab[:, :, 1]
                b = lab[:, :, 2]
                chroma = np.sqrt(a**2 + b**2)
                
                scale = 128.0
                c_norm = np.clip(chroma / scale, 0.0, 1.0)
                c_new_norm = apply_with_mask(c_norm, lut, mask)
                c_new = c_new_norm * scale
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    mul = c_new / chroma
                    mul[chroma == 0] = 1.0
                
                lab[:, :, 1] = a * mul
                lab[:, :, 2] = b * mul
                res = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
                res = np.clip(res, 0.0, 1.0)
        
        return res

# =============================================================================
# UI: CURVE EDITOR WIDGET
# =============================================================================

class CurveEditor(QWidget):
    curveChanged = pyqtSignal(float, float)
    popoutRequested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 300)
        self.setMouseTracking(True)
        
        self.points = [[0.0, 0.0], [1.0, 1.0]]
        self.lut_preview = None
        self.histogram_pdf = None
        self.histogram_display = None
        self.histogram_zone_pdf = None
        self.histogram_zone_display = None
        self.pipette_val = None
        
        self.line_color = QColor(255, 255, 255)
        self.fill_color = QColor(255, 255, 255, 40)
        self.mode_type = "Standard"
        
        self.drag_idx = -1
        self.hover_idx = -1
        
        self.range_enabled = False
        self.range_min = 0.0
        self.range_max = 1.0
        
        self._recalc_lut()
        
        self.setToolTip(
            "Spline Editor\n"
            "• Left Click: Add new point or Drag.\n"
            "• Right Click: Remove point.\n"
            "• Drag Endpoint: Move horizontally to clip black/white points,\n"
            "  or vertically to lift blacks or compress highlights."
        )

        # Pop-out button overlay
        self.btn_popout = QPushButton("⇱", self)
        self.btn_popout.setObjectName("PopOutButton")
        self.btn_popout.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_popout.setToolTip("Open in a separate window (Pop-out)")
        self.btn_popout.setFixedSize(24, 24)
        self.btn_popout.clicked.connect(self.popoutRequested.emit)
        self.btn_popout.show()
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        m = 5
        self.btn_popout.move(m, m)
    
    def set_config(self, color, mode="Standard"):
        self.line_color = color
        self.fill_color = QColor(color)
        self.fill_color.setAlpha(40)
        self.mode_type = mode
        self.update()
    
    def load_points(self, points):
        self.points = [list(p) for p in points]
        self._recalc_lut()
    
    def set_histograms(self, main_pdf, zone_pdf=None):
        self.histogram_pdf = main_pdf
        self.histogram_zone_pdf = zone_pdf
        self._recalc_lut()

    def set_histogram(self, pdf_array):
        # Backward compatible wrapper
        self.set_histograms(pdf_array, None)
    
    def set_pipette(self, val):
        self.pipette_val = val
        self.update()
    
    def set_range(self, enabled, min_val, max_val):
        self.range_enabled = enabled
        self.range_min = min_val
        self.range_max = max_val
        self.update()
    
    def _recalc_lut(self):
        self.lut_preview = CurvesCore.generate_lut(self.points, size=256)
        
        b_clip = 0.0
        w_clip = 0.0
        self.histogram_zone_display = None
        
        if self.histogram_pdf is not None:
            n_bins = len(self.histogram_pdf)
            lut_indices = (self.lut_preview * (n_bins - 1)).astype(int)
            lut_indices = np.clip(lut_indices, 0, n_bins - 1)
            
            self.histogram_display = np.zeros(n_bins, dtype=np.float32)
            np.add.at(self.histogram_display, lut_indices, self.histogram_pdf)
            
            if np.sum(self.histogram_display) > 0:
                k_size = 15
                sigma = 2.5
                x = np.linspace(-(k_size // 2), k_size // 2, k_size)
                kernel = np.exp(-0.5 * (x / sigma) ** 2)
                kernel /= np.sum(kernel)
                self.histogram_display = np.convolve(self.histogram_display, kernel, mode='same')
                
                mx = np.max(self.histogram_display)
                if mx > 0:
                    self.histogram_display /= mx
            
            epsilon = 1e-3
            black_mask = (self.lut_preview <= epsilon)
            white_mask = (self.lut_preview >= 1.0 - epsilon)
            
            if np.sum(self.histogram_pdf) > 0:
                b_clip = np.sum(self.histogram_pdf[black_mask]) * 100.0
                w_clip = np.sum(self.histogram_pdf[white_mask]) * 100.0
        else:
            self.histogram_display = None
        
        # Compute zone overlay histogram display (no LUT remapping, but same smoothing/normalization)
        self.histogram_zone_display = None
        if self.histogram_zone_pdf is not None:
            z = self.histogram_zone_pdf.astype(np.float32, copy=False)
            # Smooth and normalize using the same visual logic as the main histogram
            if np.sum(z) > 0:
                k_size = 15
                sigma = 2.5
                x = np.linspace(-(k_size // 2), k_size // 2, k_size)
                kernel = np.exp(-0.5 * (x / sigma) ** 2)
                kernel /= np.sum(kernel)
                z = np.convolve(z, kernel, mode='same')
                mxz = np.max(z)
                if mxz > 0:
                    z = z / mxz
            self.histogram_zone_display = z
        
        if not np.isfinite(b_clip):
            b_clip = 0.0
        if not np.isfinite(w_clip):
            w_clip = 0.0
        b_clip = float(np.clip(b_clip, 0.0, 100.0))
        w_clip = float(np.clip(w_clip, 0.0, 100.0))
        
        self.curveChanged.emit(b_clip, w_clip)
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w, h = self.width(), self.height()
        
        if self.mode_type in ["Saturation", "Chrominance"]:
            grad = QLinearGradient(0, h, 0, 0)
            grad.setColorAt(0.0, QColor(25, 25, 25))
            grad.setColorAt(1.0, QColor(40, 30, 70) if "Sat" in self.mode_type else QColor(70, 40, 30))
            painter.fillRect(0, 0, w, h, QBrush(grad))
            
            painter.setPen(QColor(100, 100, 100))
            lbl = "Sat" if "Sat" in self.mode_type else "Chroma"
            painter.drawText(5, 12, f"High {lbl}")
            painter.drawText(5, h-5, f"Low {lbl}")
        else:
            painter.fillRect(0, 0, w, h, QColor(25, 25, 25))
        
        painter.setPen(QPen(QColor(60, 60, 60), 1, Qt.PenStyle.DotLine))
        for i in range(1, 4):
            v_x = int(i * 0.25 * w)
            painter.drawLine(v_x, 0, v_x, h)
            v_y = int(i * 0.25 * h)
            painter.drawLine(0, v_y, w, v_y)
        
        painter.setPen(QPen(QColor(80, 80, 80), 1))
        painter.drawLine(0, h, w, 0)
        
        if self.range_enabled:
            overlay_color = QColor(25, 25, 25, 180)
            
            if self.range_min > 0.0:
                min_x = int(self.range_min * w)
                painter.fillRect(0, 0, min_x, h, overlay_color)
            
            if self.range_max < 1.0:
                max_x = int(self.range_max * w)
                painter.fillRect(max_x, 0, w - max_x, h, overlay_color)
            
            painter.setPen(QPen(QColor(255, 170, 0, 150), 2, Qt.PenStyle.DashLine))
            if self.range_min > 0.0:
                min_x = int(self.range_min * w)
                painter.drawLine(min_x, 0, min_x, h)
            if self.range_max < 1.0:
                max_x = int(self.range_max * w)
                painter.drawLine(max_x, 0, max_x, h)
        
        # Histogram (main) + optional luminance zone overlay
        if self.histogram_display is not None:
            step = w / len(self.histogram_display)

            # 0) Optional zone overlay histogram (ghost), used to visualize the luminance range mask
            if self.histogram_zone_display is not None:
                # Use a very faint alpha of the existing fill color
                zone_color_full = QColor(self.fill_color)
                zone_color_full.setAlpha(24)
                zone_color_dim = QColor(self.fill_color)
                zone_color_dim.setAlpha(12)
                zone_stroke = QColor(self.fill_color)
                zone_stroke.setAlpha(48)

                z = self.histogram_zone_display

                if self.range_enabled:
                    # Draw zone histogram in three sections with dimming
                    if self.range_min > 0.0:
                        min_idx = int(self.range_min * len(z))
                        path_low = QPainterPath()
                        path_low.moveTo(0, h)
                        for i in range(min(min_idx + 1, len(z))):
                            val = z[i]
                            path_low.lineTo(i * step, h - (val * h * 0.9))
                        path_low.lineTo(min_idx * step, h)
                        path_low.closeSubpath()
                        painter.setBrush(zone_color_dim)
                        painter.setPen(Qt.PenStyle.NoPen)
                        painter.drawPath(path_low)

                    min_idx = int(self.range_min * len(z))
                    max_idx = int(self.range_max * len(z))
                    path_active = QPainterPath()
                    path_active.moveTo(min_idx * step, h)
                    for i in range(min_idx, min(max_idx + 1, len(z))):
                        val = z[i]
                        path_active.lineTo(i * step, h - (val * h * 0.9))
                    path_active.lineTo(max_idx * step, h)
                    path_active.closeSubpath()
                    painter.setBrush(zone_color_full)
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.drawPath(path_active)
                    # Elegant, subtle stroke for ghost histogram (active range only)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.setPen(QPen(zone_stroke, 1))
                    painter.drawPath(path_active)
                    painter.setPen(Qt.PenStyle.NoPen)

                    if self.range_max < 1.0:
                        max_idx = int(self.range_max * len(z))
                        path_high = QPainterPath()
                        path_high.moveTo(max_idx * step, h)
                        for i in range(max_idx, len(z)):
                            val = z[i]
                            path_high.lineTo(i * step, h - (val * h * 0.9))
                        path_high.lineTo(w, h)
                        path_high.closeSubpath()
                        painter.setBrush(zone_color_dim)
                        painter.setPen(Qt.PenStyle.NoPen)
                        painter.drawPath(path_high)
                else:
                    # No range limiting: draw full zone histogram as faint
                    path = QPainterPath()
                    path.moveTo(0, h)
                    for i, val in enumerate(z):
                        path.lineTo(i * step, h - (val * h * 0.9))
                    path.lineTo(w, h)
                    path.closeSubpath()
                    painter.setBrush(zone_color_full)
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.drawPath(path)
                    # Elegant, subtle stroke for ghost histogram (full)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.setPen(QPen(zone_stroke, 1))
                    painter.drawPath(path)
                    painter.setPen(Qt.PenStyle.NoPen)

            # 1) Main histogram (always drawn on top)
            if self.histogram_zone_display is not None:
                # When zone overlay is present, keep main histogram full-bright for expected behavior
                path = QPainterPath()
                path.moveTo(0, h)
                for i, val in enumerate(self.histogram_display):
                    path.lineTo(i * step, h - (val * h * 0.9))
                path.lineTo(w, h)
                path.closeSubpath()
                painter.setBrush(self.fill_color)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawPath(path)
            else:
                # Original behavior (range-based dimming applied to the displayed histogram)
                if self.range_enabled:
                    if self.range_min > 0.0:
                        min_idx = int(self.range_min * len(self.histogram_display))
                        path_low = QPainterPath()
                        path_low.moveTo(0, h)
                        for i in range(min(min_idx + 1, len(self.histogram_display))):
                            val = self.histogram_display[i]
                            path_low.lineTo(i * step, h - (val * h * 0.9))
                        path_low.lineTo(min_idx * step, h)
                        path_low.closeSubpath()
                        dimmed_color = QColor(self.fill_color)
                        dimmed_color.setAlpha(15)
                        painter.setBrush(dimmed_color)
                        painter.setPen(Qt.PenStyle.NoPen)
                        painter.drawPath(path_low)

                    min_idx = int(self.range_min * len(self.histogram_display))
                    max_idx = int(self.range_max * len(self.histogram_display))
                    path_active = QPainterPath()
                    path_active.moveTo(min_idx * step, h)
                    for i in range(min_idx, min(max_idx + 1, len(self.histogram_display))):
                        val = self.histogram_display[i]
                        path_active.lineTo(i * step, h - (val * h * 0.9))
                    path_active.lineTo(max_idx * step, h)
                    path_active.closeSubpath()
                    painter.setBrush(self.fill_color)
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.drawPath(path_active)

                    if self.range_max < 1.0:
                        max_idx = int(self.range_max * len(self.histogram_display))
                        path_high = QPainterPath()
                        path_high.moveTo(max_idx * step, h)
                        for i in range(max_idx, len(self.histogram_display)):
                            val = self.histogram_display[i]
                            path_high.lineTo(i * step, h - (val * h * 0.9))
                        path_high.lineTo(w, h)
                        path_high.closeSubpath()
                        dimmed_color = QColor(self.fill_color)
                        dimmed_color.setAlpha(15)
                        painter.setBrush(dimmed_color)
                        painter.setPen(Qt.PenStyle.NoPen)
                        painter.drawPath(path_high)
                else:
                    path = QPainterPath()
                    path.moveTo(0, h)
                    for i, val in enumerate(self.histogram_display):
                        path.lineTo(i * step, h - (val * h * 0.9))
                    path.lineTo(w, h)
                    path.closeSubpath()
                    painter.setBrush(self.fill_color)
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.drawPath(path)
        
        if self.lut_preview is not None:
            path = QPainterPath()
            path.moveTo(0, h - self.lut_preview[0] * h)
            for i, val in enumerate(self.lut_preview):
                x_pos = (i / 255.0) * w
                y_pos = h - (val * h)
                path.lineTo(x_pos, y_pos)
            
            painter.setPen(QPen(self.line_color, 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)
        
        for i, (nx, ny) in enumerate(self.points):
            sx, sy = nx * w, h - (ny * h)
            active = (i == self.hover_idx or i == self.drag_idx)
            r = 6 if active else 4
            
            if active:
                grad = QRadialGradient(sx, sy, r + 4)
                grad.setColorAt(0, self.line_color)
                grad.setColorAt(1, Qt.GlobalColor.transparent)
                painter.setBrush(grad)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(QPointF(sx, sy), r + 4, r + 4)
            
            painter.setBrush(QColor(20, 20, 20))
            painter.setPen(QPen(self.line_color, 2))
            painter.drawEllipse(QPointF(sx, sy), r, r)
        
        if self.pipette_val is not None:
            px = self.pipette_val * w
            painter.setPen(QPen(QColor(255, 255, 255, 180), 1, Qt.PenStyle.DashLine))
            painter.drawLine(int(px), 0, int(px), h)
            
            idx = int(self.pipette_val * 255)
            idx = max(0, min(255, idx))
            if idx < len(self.lut_preview):
                y_val = self.lut_preview[idx]
                painter.setBrush(self.line_color)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(QPointF(px, h - (y_val * h)), 4, 4)
    
    def mousePressEvent(self, event):
        pos = event.pos()
        w, h = self.width(), self.height()
        nx, ny = pos.x() / w, 1.0 - (pos.y() / h)
        
        hit_dist = 0.05
        clicked = -1
        for i, (px, py) in enumerate(self.points):
            if abs(px - nx) < hit_dist and abs(py - ny) < hit_dist:
                clicked = i
                break
        
        if event.button() == Qt.MouseButton.LeftButton:
            if clicked != -1:
                self.drag_idx = clicked
            else:
                self.points.append([nx, ny])
                self.points.sort(key=lambda p: p[0])
                for i, p in enumerate(self.points):
                    if p[0] == nx:
                        self.drag_idx = i
                        break
            self._recalc_lut()
        elif event.button() == Qt.MouseButton.RightButton:
            if clicked != -1 and 0 < clicked < len(self.points) - 1:
                self.points.pop(clicked)
                self.hover_idx = -1
                self._recalc_lut()
    
    def mouseMoveEvent(self, event):
        pos = event.pos()
        w, h = self.width(), self.height()
        nx, ny = np.clip(pos.x() / w, 0, 1), np.clip(1.0 - (pos.y() / h), 0, 1)
        
        if self.drag_idx != -1:
            idx = self.drag_idx
            if idx == 0:
                min_x = 0.0
                max_x = self.points[1][0] - 0.005
            elif idx == len(self.points) - 1:
                min_x = self.points[idx - 1][0] + 0.005
                max_x = 1.0
            else:
                min_x = self.points[idx - 1][0] + 0.005
                max_x = self.points[idx + 1][0] - 0.005
            
            new_x = max(min_x, min(max_x, nx))
            self.points[idx] = [new_x, ny]
            self._recalc_lut()
        else:
            hit_dist = 0.05
            self.hover_idx = -1
            for i, (px, py) in enumerate(self.points):
                if abs(px - nx) < hit_dist and abs(py - ny) < hit_dist:
                    self.hover_idx = i
                    break
            self.update()
    
    def mouseReleaseEvent(self, e):
        self.drag_idx = -1
        self.update()

# =============================================================================
# WORKER & MAIN WINDOW
# =============================================================================

class CurvesWorker(QThread):
    result_ready = pyqtSignal(object)
    
    def __init__(self, img, channels, input_luminance=None, show_mask=False, current_chan='RGB/K'):
        super().__init__()
        self.img = img
        self.channels = channels
        self.input_luminance = input_luminance
        self.show_mask = show_mask
        self.current_chan = current_chan
    
    def run(self):
        try:
            # If Show Mask is enabled and current channel has range limiting, just compute mask
            if self.show_mask and self.channels.get(self.current_chan, {}).get('lum_range_enabled', False):
                c = self.channels[self.current_chan]
                m = CurvesCore.compute_luminance_mask(
                    self.img, 
                    c.get('lum_min', 0.0), 
                    c.get('lum_max', 1.0), 
                    c.get('feather_sigma', 0.25)
                )
                res = np.dstack([m, m, m]) # Grayscale visualization
                self.result_ready.emit(res)
                return

            for k in self.channels:
                if self.channels[k]['active']:
                    self.channels[k]['lut'] = CurvesCore.generate_lut(self.channels[k]['points'], 65536)
            
            res = CurvesCore.apply_pipeline(self.img, self.channels, self.input_luminance)
            self.result_ready.emit(res)
        except Exception:
            traceback.print_exc()
            self.result_ready.emit(None)

class DetachedCurveWindow(QMainWindow):
    """Container window for the detached curve editor."""
    closed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint)
        
        self.setWindowTitle("VeraLux Curves - Editor")
        self.setStyleSheet(DARK_STYLESHEET)
        self.resize(800, 800)
        
    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)

class VeraLuxCurves(QMainWindow):
    def __init__(self, siril, app):
        super().__init__()
        self.siril = siril
        self.app = app
        
        self.setWindowTitle(f"VeraLux Curves v{VERSION}")
        self.setStyleSheet(DARK_STYLESHEET)
        self.resize(1500, 800)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)
        
        header_msg = (
            f"\n##############################################\n"
            f"# VeraLux — Curves v{VERSION}\n"
            "# Spline-Based Photometric Sculpting Engine\n"
            "# Author: Riccardo Paterniti (2025)\n"
            "# Contact: info@veralux.space\n"
            "##############################################"
        )
        
        try: self.siril.log(header_msg)
        except: print(header_msg)
        
        self.img_full = None
        self.img_proxy = None
        self.processed_proxy = None
        self.input_luminance_proxy = None
        self.input_luminance_full = None
        self.is_mono_source = False
        self._switching_channel = False

        self.staged_proxy = None
        self.staged_luminance_proxy = None
        self.stage_stack = []
        
        self.channels = {
            'RGB/K': {
                'points': [[0,0],[1,1]], 
                'color': '#ffffff', 
                'hist': None, 
                'active': False,
                'lum_range_enabled': False,
                'lum_show_mask': False,
                'lum_min': 0.0,
                'lum_max': 1.0,
                'feather_sigma': 0.25
            },
            'R': {
                'points': [[0,0],[1,1]], 
                'color': '#ff4444', 
                'hist': None, 
                'active': False,
                'lum_range_enabled': False,
                'lum_show_mask': False,
                'lum_min': 0.0,
                'lum_max': 1.0,
                'feather_sigma': 0.25
            },
            'G': {
                'points': [[0,0],[1,1]], 
                'color': '#44ff44', 
                'hist': None, 
                'active': False,
                'lum_range_enabled': False,
                'lum_show_mask': False,
                'lum_min': 0.0,
                'lum_max': 1.0,
                'feather_sigma': 0.25
            },
            'B': {
                'points': [[0,0],[1,1]], 
                'color': '#4444ff', 
                'hist': None, 
                'active': False,
                'lum_range_enabled': False,
                'lum_show_mask': False,
                'lum_min': 0.0,
                'lum_max': 1.0,
                'feather_sigma': 0.25
            },
            'L': {
                'points': [[0,0],[1,1]], 
                'color': '#aaaaaa', 
                'hist': None, 
                'active': False,
                'lum_range_enabled': False,
                'lum_show_mask': False,
                'lum_min': 0.0,
                'lum_max': 1.0,
                'feather_sigma': 0.25
            },
            'C': {
                'points': [[0,0],[1,1]], 
                'color': '#ffaa00', 
                'hist': None, 
                'active': False,
                'lum_range_enabled': False,
                'lum_show_mask': False,
                'lum_min': 0.0,
                'lum_max': 1.0,
                'feather_sigma': 0.25
            },
            'S': {
                'points': [[0,0],[1,1]], 
                'color': '#aa00aa', 
                'hist': None, 
                'active': False,
                'lum_range_enabled': False,
                'lum_show_mask': False,
                'lum_min': 0.0,
                'lum_max': 1.0,
                'feather_sigma': 0.25
            },
        }
        
        self.current_chan = 'RGB/K'
        self.show_original = False
        
        self.tips = {
            'RGB/K': "Master Channel\nAffects Luminance and Color simultaneously. Use for main contrast.",
            'R': "Red Channel\nSpecific color balancing for Red.",
            'G': "Green Channel\nSpecific color balancing for Green.",
            'B': "Blue Channel\nSpecific color balancing for Blue.",
            'L': "Luminance (L*)\nOperates in CIE L*a*b* space. Adjusts contrast without shifting saturation.",
            'C': "Chrominance (LCH)\nSpectral intensity. Use this to boost color presence naturally.",
            'S': "Saturation (HSV)\nMathematical saturation. Strong, vivid adjustments."
        }

        self.detached_window = None
        
        # FIX: Define debounce before init_ui
        self.debounce = QTimer()
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(50)
        self.debounce.timeout.connect(self.run_preview)
        
        self.init_ui()
        
        try:
            self.siril.log(f"VeraLux Curves v{VERSION} Loaded.")
        except:
            pass
        
        self.cache_input()
    
    def init_ui(self):
        main = QWidget()
        self.setCentralWidget(main)
        layout = QHBoxLayout(main)
        
        left = QWidget()
        left.setFixedWidth(400)
        self.l_layout = QVBoxLayout(left)
        self.l_layout.setContentsMargins(0,0,0,0)
        
        lbl_t = QLabel("VeraLux Curves")
        lbl_t.setStyleSheet("font-size: 16pt; font-weight: bold; color: #88aaff; margin-top: 5px;")
        lbl_t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.l_layout.addWidget(lbl_t)
        
        lbl_s = QLabel("Spline-Based Photometric Sculpting")
        lbl_s.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_s.setStyleSheet("font-size: 10pt; color: #999999; font-style: italic; margin-bottom: 10px;")
        self.l_layout.addWidget(lbl_s)
        
        g_chan = QGroupBox("Target Domain")
        lc = QVBoxLayout(g_chan)
        
        self.chan_grp = QButtonGroup()
        self.radio_btns = {}
        
        grid = QGridLayout()
        grid.setVerticalSpacing(12)
        grid.setHorizontalSpacing(10)
        
        items = [('RGB/K', 0, 0), ('R', 0, 1), ('G', 0, 2), ('B', 0, 3),
                 ('L', 1, 0), ('C', 1, 1), ('S', 1, 2)]
        
        for (name, r, c) in items:
            rb = QRadioButton(name)
            _nofocus(rb)
            col = self.channels[name]['color']
            base_style = f"QRadioButton::indicator:checked {{ background-color: {col}; }}"
            rb.setStyleSheet(base_style)
            rb.setToolTip(self.tips.get(name, ""))
            self.chan_grp.addButton(rb)
            self.radio_btns[name] = rb
            rb.clicked.connect(lambda _, n=name: self.change_channel(n))
            grid.addWidget(rb, r, c)
            if name == 'RGB/K': rb.setChecked(True)
        
        lc.addLayout(grid)
        self.l_layout.addWidget(g_chan)
        
        self.editor = CurveEditor()
        self.editor.curveChanged.connect(self.on_editor_changed)
        self.editor.popoutRequested.connect(self.toggle_popout)
        self.l_layout.addWidget(self.editor, 1)

        self.placeholder = QLabel("Editor Detached")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setStyleSheet("background-color: #202020; color: #444; border: 1px dashed #333;")
        self.placeholder.hide()
        
        g_range = QGroupBox("Luminance Range Limiting")
        lr_layout = QVBoxLayout(g_range)
        
        range_header_layout = QHBoxLayout()
        self.chk_range = QCheckBox("Enable Range Limiting")
        _nofocus(self.chk_range)
        self.chk_range.setToolTip("Apply curve only within specified luminance range")
        self.chk_range.toggled.connect(self.on_range_toggle)
        
        self.chk_show_mask = QCheckBox("Show Mask")
        _nofocus(self.chk_show_mask)
        self.chk_show_mask.setToolTip("Visualize the luminance mask for the current range settings")
        self.chk_show_mask.setEnabled(False)
        self.chk_show_mask.toggled.connect(self.on_show_mask_toggle)
        
        range_header_layout.addWidget(self.chk_range)
        range_header_layout.addWidget(self.chk_show_mask)
        lr_layout.addLayout(range_header_layout)
        
        lbl_min = QLabel("Min Luminance: 0%")
        self.lbl_min = lbl_min
        lr_layout.addWidget(lbl_min)
        
        self.slider_min = QSlider(Qt.Orientation.Horizontal)
        _nofocus(self.slider_min)
        self.slider_min.setRange(0, 100)
        self.slider_min.setValue(0)
        self.slider_min.setEnabled(False)
        self.slider_min.setToolTip("Minimum luminance range\nDouble-click to reset to 0%")
        self.slider_min.valueChanged.connect(self.on_range_changed)
        self.slider_min.mouseDoubleClickEvent = lambda e: self.reset_slider(self.slider_min, 0, 'min')
        lr_layout.addWidget(self.slider_min)
        
        lbl_max = QLabel("Max Luminance: 100%")
        self.lbl_max = lbl_max
        lr_layout.addWidget(lbl_max)
        
        self.slider_max = QSlider(Qt.Orientation.Horizontal)
        _nofocus(self.slider_max)
        self.slider_max.setRange(0, 100)
        self.slider_max.setValue(100)
        self.slider_max.setEnabled(False)
        self.slider_max.setToolTip("Maximum luminance range\nDouble-click to reset to 100%")
        self.slider_max.valueChanged.connect(self.on_range_changed)
        self.slider_max.mouseDoubleClickEvent = lambda e: self.reset_slider(self.slider_max, 100, 'max')
        lr_layout.addWidget(self.slider_max)
        
        feather_layout = QHBoxLayout()
        lbl_feather = QLabel("Feather: 25%")
        self.lbl_feather = lbl_feather
        feather_layout.addWidget(lbl_feather)
        
        self.slider_feather = QSlider(Qt.Orientation.Horizontal)
        _nofocus(self.slider_feather)
        self.slider_feather.setRange(0, 100)
        self.slider_feather.setValue(25)
        self.slider_feather.setEnabled(False)
        self.slider_feather.setToolTip("Soft roll-off width at range boundaries\nDouble-click to reset to default (25%)")
        self.slider_feather.valueChanged.connect(self.on_feather_changed)
        self.slider_feather.mouseDoubleClickEvent = lambda e: self.reset_slider(self.slider_feather, 25, 'feather')
        feather_layout.addWidget(self.slider_feather)
        
        lr_layout.addLayout(feather_layout)
        self.l_layout.addWidget(g_range)
        
        h_res = QHBoxLayout()
        # Create lbl_applied before adding to h_res
        self.lbl_applied = QLabel("")
        self.lbl_applied.setStyleSheet("font-size: 8pt; color: #666; font-weight: bold;")

        # RESET BUTTONS
        reset_btn_width = 90
        b_r1 = QPushButton("Reset Channel")
        _nofocus(b_r1)
        b_r1.setObjectName("ResetChannelButton")
        b_r1.setMinimumWidth(reset_btn_width)
        b_r1.setToolTip("Reset the currently selected channel to default linear state")
        b_r1.clicked.connect(self.reset_current_channel)

        b_r2 = QPushButton("Reset All")
        _nofocus(b_r2)
        b_r2.setObjectName("ResetAllButton")
        b_r2.setMinimumWidth(reset_btn_width)
        b_r2.setToolTip("Reset all channels to default linear state")
        b_r2.clicked.connect(self.reset_all)

        # ACTION BUTTONS (Undo/Apply)
        action_btn_width = 50
        self.btn_undo = QPushButton("Undo")
        _nofocus(self.btn_undo)
        self.btn_undo.setObjectName("UndoButton")
        self.btn_undo.setMinimumWidth(action_btn_width)
        self.btn_undo.setToolTip("Undo the last applied stage")
        self.btn_undo.clicked.connect(self.undo_stage)

        self.btn_apply = QPushButton("Apply")
        _nofocus(self.btn_apply)
        self.btn_apply.setObjectName("ApplyButton")
        self.btn_apply.setMinimumWidth(action_btn_width)
        self.btn_apply.setToolTip("Apply current curves to a new stage and reset controls for further editing")
        self.btn_apply.clicked.connect(self.apply_stage)

        h_res.addWidget(b_r1)
        h_res.addWidget(b_r2)
        h_res.addStretch()
        h_res.addWidget(self.lbl_applied)
        h_res.addWidget(self.btn_undo)
        h_res.addWidget(self.btn_apply)
        self.l_layout.addLayout(h_res)

        ft = QHBoxLayout()

        self.btn_help = QPushButton("?")
        _nofocus(self.btn_help)
        self.btn_help.setObjectName("HelpButton")
        self.btn_help.setToolTip("Show Help (Operational Guide)")
        self.btn_help.clicked.connect(self.print_help)

        self.lbl_clip = QLabel("")
        self.lbl_clip.setStyleSheet("font-size: 8pt; font-weight: bold;")

        b_cl = QPushButton("Close")
        _nofocus(b_cl)
        b_cl.setObjectName("CloseButton")
        b_cl.setToolTip("Close dialog without applying changes")
        b_cl.clicked.connect(self.close)
        b_cl.setMinimumWidth(90)

        b_ap = QPushButton("PROCESS")
        _nofocus(b_ap)
        b_ap.setObjectName("ProcessButton")
        b_ap.setToolTip("Render full resolution image with all applied stages and send to Siril")
        b_ap.clicked.connect(self.apply_process)
        b_ap.setMinimumWidth(90)

        ft.addWidget(self.btn_help)
        ft.addWidget(self.lbl_clip)
        ft.addStretch()
        ft.addWidget(b_cl)
        ft.addWidget(b_ap)
        self.l_layout.addLayout(ft)

        self._update_stage_ui()

        layout.addWidget(left)
        
        right = QWidget()
        r_layout = QVBoxLayout(right)
        r_layout.setContentsMargins(0,0,0,0)
        
        tb = QHBoxLayout()
        tb.setContentsMargins(5,5,5,5)
        
        b_zm = QPushButton("-")
        _nofocus(b_zm)
        b_zm.setObjectName("ZoomBtn")
        b_zm.setToolTip("Zoom Out")
        b_zm.clicked.connect(self.zoom_out)
        
        b_fit = QPushButton("Fit")
        _nofocus(b_fit)
        b_fit.setObjectName("ZoomBtn")
        b_fit.setToolTip("Fit image to view")
        b_fit.clicked.connect(self.fit_view)
        
        b_11 = QPushButton("1:1")
        _nofocus(b_11)
        b_11.setObjectName("ZoomBtn")
        b_11.setToolTip("View at 100% proxy resolution (1:1 preview)")
        b_11.clicked.connect(self.zoom_11)
        
        b_zp = QPushButton("+")
        _nofocus(b_zp)
        b_zp.setObjectName("ZoomBtn")
        b_zp.setToolTip("Zoom In")
        b_zp.clicked.connect(self.zoom_in)
        
        lbl_h = QLabel("Hold Space to Compare - Click for Pipette")
        lbl_h.setStyleSheet("color: #ffb000; font-style: italic; margin-left: 10px;")
        
        self.lbl_readout = QLabel("")
        self.lbl_readout.setStyleSheet("font-size: 9pt; font-weight: bold; margin-left: 20px;")
        self.lbl_readout.setMinimumWidth(450)
        
        self.btn_coffee = QPushButton("☕")
        _nofocus(self.btn_coffee)
        self.btn_coffee.setObjectName("CoffeeButton")
        self.btn_coffee.setToolTip("Support the developer - Buy me a coffee!")
        self.btn_coffee.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_coffee.clicked.connect(lambda: webbrowser.open("https://buymeacoffee.com/riccardo.paterniti"))

        self.chk_ontop = QCheckBox("On Top")
        _nofocus(self.chk_ontop)
        self.chk_ontop.setChecked(True)
        self.chk_ontop.setToolTip("Keep window above other windows")
        self.chk_ontop.toggled.connect(self.toggle_ontop)
        
        tb.addWidget(b_zm)
        tb.addWidget(b_fit)
        tb.addWidget(b_11)
        tb.addWidget(b_zp)
        tb.addWidget(lbl_h)
        tb.addWidget(self.lbl_readout)
        tb.addStretch()
        tb.addWidget(self.chk_ontop)
        tb.addWidget(self.btn_coffee)
        
        r_layout.addWidget(QWidget())
        r_layout.addLayout(tb)
        
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.view.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.view.setCursor(Qt.CursorShape.CrossCursor)
        self.view.setMouseTracking(True)
        self.view.viewport().installEventFilter(self)
        self.view.viewport().setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.lbl_blink = QLabel("ORIGINAL", self.view)
        self.lbl_blink.setStyleSheet("background-color: rgba(255, 160, 0, 200); color: #ffffff; font-size: 14pt; font-weight: bold; padding: 8px 16px; border-radius: 6px;")
        self.lbl_blink.hide()
        
        self.pix_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pix_item)
        
        r_layout.addWidget(self.view)
        
        layout.addWidget(right, 1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'lbl_blink') and self.lbl_blink:
            vx = (self.view.width() - self.lbl_blink.width()) // 2
            self.lbl_blink.move(vx, 10)
    
    def showEvent(self, event):
        super().showEvent(event)
        try:
            self.view.viewport().setFocus()
        except Exception:
            pass
    
    def cache_input(self):
        try:
            if not self.siril.connected:
                self.siril.connect()

            with self.siril.image_lock():
                img = self.siril.get_image_pixeldata()
                if img is None:
                    return

                self.is_mono_source = (img.ndim == 2) or (img.ndim == 3 and img.shape[0] == 1 and img.shape[2] != 3)

                img = CurvesCore.normalize_input(img)
                self.img_full = CurvesCore.ensure_hwc(img)

                self.input_luminance_full = np.mean(self.img_full, axis=2)

                h, w = self.img_full.shape[:2]
                scale = 1200 / max(h,w)
                self.img_proxy = cv2.resize(self.img_full, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
                self.proxy_scale = scale

                self.input_luminance_proxy = np.mean(self.img_proxy, axis=2)

                self.processed_proxy = self.img_proxy.copy()

                self.staged_proxy = self.img_proxy.copy()
                self.staged_luminance_proxy = np.mean(self.staged_proxy, axis=2)
                self.stage_stack = []
                self._update_stage_ui()

                self.calc_histograms(self.staged_proxy)
                self.change_channel('RGB/K')
                self.update_view()

                QTimer.singleShot(100, self.fit_view)

        except Exception as e:
            print(f"Init Error: {e}")
            traceback.print_exc()
    
    def calc_histograms(self, img_src=None):
        if img_src is None:
            img_src = self.img_proxy
        img = img_src.astype(np.float32)
        bins = 256

        def get_pdf(arr):
            h, _ = np.histogram(arr, bins=bins, range=(0,1))
            total = np.sum(h)
            return h / total if total > 0 else h

        for c in ['R', 'G', 'B']:
            self.channels[c]['hist'] = get_pdf(img[:,:,['R','G','B'].index(c)])

        self.channels['RGB/K']['hist'] = get_pdf(np.mean(img, axis=2))

        lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
        self.channels['L']['hist'] = get_pdf(lab[:,:,0]/100.0)

        C = np.sqrt(lab[:,:,1]**2 + lab[:,:,2]**2) / 128.0
        self.channels['C']['hist'] = get_pdf(np.clip(C,0,1))

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        self.channels['S']['hist'] = get_pdf(hsv[:,:,1])
    
    def change_channel(self, name):
        self._switching_channel = True
        try:
            self.current_chan = name
            c = self.channels[name]
            
            mode = "Saturation" if name == "S" else ("Chrominance" if name == "C" else "Standard")
            self.editor.set_config(QColor(c['color']), mode)
            self.editor.load_points([list(p) for p in c['points']])
            if name in ['S', 'C']:
                zone_pdf = self.channels['RGB/K'].get('hist', None)
                self.editor.set_histograms(c['hist'], zone_pdf)
            else:
                self.editor.set_histograms(c['hist'], None)
            
            # Gestione Range Limiting
            range_enabled = c.get('lum_range_enabled', False)
            self.chk_range.setChecked(range_enabled)
            
            # Gestione Show Mask (Eredita lo stato specifico del canale)
            self.chk_show_mask.setChecked(c.get('lum_show_mask', False))
            self.chk_show_mask.setEnabled(range_enabled)
            
            self.slider_min.setValue(int(c.get('lum_min', 0.0) * 100))
            self.slider_max.setValue(int(c.get('lum_max', 1.0) * 100))
            feather_val = int(c.get('feather_sigma', 0.25) * 100)
            self.slider_feather.setValue(feather_val)
            self.lbl_feather.setText(f"Feather: {feather_val}%")
            
            self.editor.set_range(
                range_enabled,
                c.get('lum_min', 0.0),
                c.get('lum_max', 1.0)
            )

        finally:
            self._switching_channel = False
    
    def on_range_toggle(self, checked):
        if self._switching_channel:
            return
        
        self.channels[self.current_chan]['lum_range_enabled'] = checked
        self.slider_min.setEnabled(checked)
        self.slider_max.setEnabled(checked)
        self.slider_feather.setEnabled(checked)
        self.chk_show_mask.setEnabled(checked)
        if not checked:
            self.chk_show_mask.setChecked(False)
        
        self.editor.set_range(
            checked,
            self.channels[self.current_chan]['lum_min'],
            self.channels[self.current_chan]['lum_max']
        )
        
        self.update_ui_indicators()
        self.debounce.start()
    
    def on_show_mask_toggle(self, checked):
        if self._switching_channel:
            return
        self.channels[self.current_chan]['lum_show_mask'] = checked
        self.debounce.start()

    def on_range_changed(self):
        if self._switching_channel:
            return
        
        min_val = self.slider_min.value() / 100.0
        max_val = self.slider_max.value() / 100.0
        
        if min_val >= max_val:
            if self.sender() == self.slider_min:
                max_val = min(min_val + 0.05, 1.0)
                self.slider_max.setValue(int(max_val * 100))
            else:
                min_val = max(max_val - 0.05, 0.0)
                self.slider_min.setValue(int(min_val * 100))
        
        self.channels[self.current_chan]['lum_min'] = min_val
        self.channels[self.current_chan]['lum_max'] = max_val
        
        self.lbl_min.setText(f"Min Luminance: {int(min_val*100)}%")
        self.lbl_max.setText(f"Max Luminance: {int(max_val*100)}%")
        
        self.editor.set_range(
            self.channels[self.current_chan]['lum_range_enabled'],
            min_val,
            max_val
        )
        
        self.debounce.start()
    
    def on_feather_changed(self, value):
        if self._switching_channel:
            return
        
        self.channels[self.current_chan]['feather_sigma'] = value / 100.0
        self.lbl_feather.setText(f"Feather: {value}%")
        self.debounce.start()
    
    def reset_slider(self, slider, default_value, slider_type):
        """Reset slider to default value on double-click."""
        slider.setValue(default_value)
        
        if slider_type == 'feather':
            self.channels[self.current_chan]['feather_sigma'] = default_value / 100.0
            self.lbl_feather.setText(f"Feather: {default_value}%")
        elif slider_type == 'min':
            self.channels[self.current_chan]['lum_min'] = default_value / 100.0
            self.lbl_min.setText(f"Min Luminance: {default_value}%")
        elif slider_type == 'max':
            self.channels[self.current_chan]['lum_max'] = default_value / 100.0
            self.lbl_max.setText(f"Max Luminance: {default_value}%")
        
        self.editor.set_range(
            self.channels[self.current_chan]['lum_range_enabled'],
            self.channels[self.current_chan]['lum_min'],
            self.channels[self.current_chan]['lum_max']
        )
        
        if not self._switching_channel:
            self.debounce.start()
    
    def _update_clip_label(self, b_pct, w_pct, is_active=True):
        try: b = float(b_pct)
        except: b = 0.0
        try: w = float(w_pct)
        except: w = 0.0
        
        if not np.isfinite(b): b = 0.0
        if not np.isfinite(w): w = 0.0
        
        b = float(np.clip(b, 0.0, 100.0))
        w = float(np.clip(w, 0.0, 100.0))
        
        eps_show = 0.005
        
        s_b = f"{b:.2f}%"
        s_w = f"{w:.2f}%"
        
        if not is_active:
            col_b = "#666"
            col_w = "#666"
        else:
            col_b = "#ff4444" if b > 0.5 else ("#ffaa00" if b > eps_show else "#666")
            col_w = "#ff4444" if w > 0.5 else ("#ffaa00" if w > eps_show else "#666")
        
        txt = (f"<span style='color:{col_b}'>Blacks: {s_b}</span> | "
               f"<span style='color:{col_w}'>Whites: {s_w}</span>")
        self.lbl_clip.setText(txt)
    
    def update_ui_indicators(self):
        for name, rb in self.radio_btns.items():
            is_active = self.channels[name]['active']
            is_range = self.channels[name].get('lum_range_enabled', False)
            col_indicator = self.channels[name]['color']
            
            base_sheet = f"QRadioButton::indicator:checked {{ background-color: {col_indicator}; }}"
            
            if is_active:
                new_style = base_sheet + """
                    QRadioButton {
                        color: #ffaa00;
                        font-weight: bold;
                    }
                """
                text = f"{name} ●" + (" ⟨⟩" if is_range else "")
            else:
                new_style = base_sheet + "QRadioButton { color: #cccccc; font-weight: normal; }"
                text = name
            
            rb.setStyleSheet(new_style)
            if rb.text() != text:
                rb.setText(text)
    
    def on_editor_changed(self, b_pct, w_pct):
        pts = self.editor.points
        is_linear = (len(pts) == 2 and
                     pts[0][0] == 0.0 and pts[0][1] == 0.0 and
                     pts[1][0] == 1.0 and pts[1][1] == 1.0)
        
        real_active = not is_linear
        switching = getattr(self, '_switching_channel', False)
        label_active = real_active and not switching
        
        self._update_clip_label(b_pct, w_pct, is_active=label_active)
        
        if switching:
            return
        
        self.channels[self.current_chan]['points'] = [list(p) for p in pts]
        self.channels[self.current_chan]['active'] = real_active
        
        self.update_ui_indicators()
        self.debounce.start()
    
    def run_preview(self):
        if self.img_proxy is None:
            return

        base = self.staged_proxy if self.staged_proxy is not None else self.img_proxy
        base_lum = self.staged_luminance_proxy if self.staged_luminance_proxy is not None else self.input_luminance_proxy

        chan_copy = copy.deepcopy(self.channels)
        
        # Pass Show Mask flag and current channel to worker
        self.worker = CurvesWorker(
            base, 
            chan_copy, 
            base_lum, 
            show_mask=self.chk_show_mask.isChecked(),
            current_chan=self.current_chan
        )
        self.worker.result_ready.connect(self.on_preview_ready)
        self.worker.start()

    def _channel_is_linear(self, ch):
        pts = ch.get('points', None)
        if pts is None:
            return True
        return (len(pts) == 2 and
                pts[0][0] == 0.0 and pts[0][1] == 0.0 and
                pts[1][0] == 1.0 and pts[1][1] == 1.0)

    def _snapshot_channels(self, source_channels):
        snap = {}
        for k, v in source_channels.items():
            snap[k] = {
                'points': [list(p) for p in v.get('points', [[0.0, 0.0], [1.0, 1.0]])],
                'active': bool(v.get('active', False)),
                'lum_range_enabled': bool(v.get('lum_range_enabled', False)),
                'lum_min': float(v.get('lum_min', 0.0)),
                'lum_max': float(v.get('lum_max', 1.0)),
                'feather_sigma': float(v.get('feather_sigma', 0.25))
            }
        return snap

    def _has_any_active_curve(self, channels_state):
        for k in channels_state:
            if channels_state[k].get('active', False):
                return True
        return False

    def _apply_stage(self, img, channels_state, input_luminance=None):
        stage = copy.deepcopy(channels_state)
        for k in stage:
            stage[k].pop('lut', None)
        for k in stage:
            if stage[k].get('active', False):
                stage[k]['lut'] = CurvesCore.generate_lut(stage[k]['points'], 65536)
        return CurvesCore.apply_pipeline(img, stage, input_luminance)

    def _reset_all_channels_to_identity(self):
        for k in self.channels:
            self.channels[k]['points'] = [[0.0, 0.0], [1.0, 1.0]]
            self.channels[k]['active'] = False
            self.channels[k]['lum_range_enabled'] = False
            self.channels[k]['lum_min'] = 0.0
            self.channels[k]['lum_max'] = 1.0
            self.channels[k]['feather_sigma'] = 0.25
            self.channels[k].pop('lut', None)

    def _rebuild_staged_proxy(self):
        if self.img_proxy is None:
            return

        base = self.img_proxy.copy()
        base_lum = np.mean(base, axis=2)

        for snap in self.stage_stack:
            if not self._has_any_active_curve(snap):
                continue
            base = self._apply_stage(base, snap, base_lum)
            base = np.clip(base, 0.0, 1.0).astype(np.float32, copy=False)
            base_lum = np.mean(base, axis=2)

        self.staged_proxy = base
        self.staged_luminance_proxy = base_lum

    def _update_stage_ui(self):
        n = len(self.stage_stack)
        try:
            self.lbl_applied.setText(f"Applied: {n}")
            if n > 0:
                self.lbl_applied.setStyleSheet("font-size: 8pt; color: #ffaa00; font-weight: bold;")
            else:
                self.lbl_applied.setStyleSheet("font-size: 8pt; color: #666; font-weight: bold;")
        except Exception:
            pass
        try:
            self.btn_undo.setEnabled(n > 0)
        except Exception:
            pass

    def apply_stage(self):
        if self.img_proxy is None:
            return

        snap = self._snapshot_channels(self.channels)
        if not self._has_any_active_curve(snap):
            return

        base = self.staged_proxy if self.staged_proxy is not None else self.img_proxy
        base_lum = self.staged_luminance_proxy if self.staged_luminance_proxy is not None else np.mean(base, axis=2)

        out = self._apply_stage(base, snap, base_lum)
        out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

        self.stage_stack.append(snap)
        self.staged_proxy = out
        self.staged_luminance_proxy = np.mean(out, axis=2)

        self._reset_all_channels_to_identity()
        self._update_stage_ui()

        self.calc_histograms(self.staged_proxy)
        self.change_channel(self.current_chan)
        self.update_ui_indicators()
        self.debounce.start()

    def undo_stage(self):
        if self.img_proxy is None:
            return
        if len(self.stage_stack) == 0:
            return

        self.stage_stack.pop()
        self._rebuild_staged_proxy()
        self._reset_all_channels_to_identity()
        self._update_stage_ui()

        if self.staged_proxy is not None:
            self.calc_histograms(self.staged_proxy)
        self.change_channel(self.current_chan)
        self.update_ui_indicators()
        self.debounce.start()
    
    def on_preview_ready(self, res):
        if res is not None:
            self.processed_proxy = res
            self.update_view()
    
    def update_view(self):
        data = self.img_proxy if self.show_original else self.processed_proxy
        if data is None:
            return

        if self.show_original:
            self.lbl_blink.show()
            vx = (self.view.width() - self.lbl_blink.width()) // 2
            self.lbl_blink.move(vx, 10)
        else:
            self.lbl_blink.hide()
        
        disp = np.clip(data * 255, 0, 255).astype(np.uint8)
        disp = np.flipud(disp)
        disp = np.ascontiguousarray(disp)
        
        qimg = QImage(disp.data, disp.shape[1], disp.shape[0], disp.shape[1]*3, QImage.Format.Format_RGB888)
        self.pix_item.setPixmap(QPixmap.fromImage(qimg))
        self.scene.setSceneRect(0, 0, disp.shape[1], disp.shape[0])
    
    def eventFilter(self, source, event):
        if source == self.view.viewport():
            if event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Space:
                if not event.isAutoRepeat():
                    self.show_original = True
                    self.update_view()
                return True
            
            if event.type() == QEvent.Type.KeyRelease and event.key() == Qt.Key.Key_Space:
                if not event.isAutoRepeat():
                    self.show_original = False
                    self.update_view()
                return True
            
            if event.type() == QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.LeftButton:
                    if self.img_proxy is not None:
                        pos = self.view.mapToScene(event.pos())
                        ix, iy = int(pos.x()), int(pos.y())
                        h = self.img_proxy.shape[0]
                        data_y = h - 1 - iy
                        if 0 <= ix < self.img_proxy.shape[1] and 0 <= data_y < h:
                            self.handle_pipette(ix, data_y)
                    self._last_pan_pos = event.pos()
                return False
            
            elif event.type() == QEvent.Type.MouseMove:
                if self.img_proxy is not None:
                    pos = self.view.mapToScene(event.pos())
                    ix, iy = int(pos.x()), int(pos.y())
                    h = self.img_proxy.shape[0]
                    data_y = h - 1 - iy
                    
                    if 0 <= ix < self.img_proxy.shape[1] and 0 <= data_y < h:
                        s = getattr(self, 'proxy_scale', 1.0)
                        fx = int(ix / s)
                        fy = int(iy / s)
                        
                        src = self.img_proxy if self.show_original else self.processed_proxy
                        p = src[data_y, ix]
                        r_pct = p[0] * 100
                        g_pct = p[1] * 100
                        b_pct = p[2] * 100
                        
                        txt = (f"proxy x:{ix} y:{iy} → full x:{fx} y:{fy} | "
                               f"<span style='color:#ff6666'>R:{r_pct:.1f}%</span> "
                               f"<span style='color:#66ff66'>G:{g_pct:.1f}%</span> "
                               f"<span style='color:#6666ff'>B:{b_pct:.1f}%</span>")
                        self.lbl_readout.setText(txt)
                    else:
                        self.lbl_readout.setText("")
                
                if event.buttons() & Qt.MouseButton.LeftButton:
                    if hasattr(self, '_last_pan_pos'):
                        delta = event.pos() - self._last_pan_pos
                        self._last_pan_pos = event.pos()
                        hs = self.view.horizontalScrollBar()
                        vs = self.view.verticalScrollBar()
                        hs.setValue(hs.value() - delta.x())
                        vs.setValue(vs.value() - delta.y())
                return True
            
            elif event.type() == QEvent.Type.MouseButtonRelease:
                if hasattr(self, '_last_pan_pos'):
                    del self._last_pan_pos
        
        return super().eventFilter(source, event)
    
    def handle_pipette(self, x, y):
        p = self.img_proxy[y, x]
        c = self.current_chan
        
        val = 0.0
        if c == 'RGB/K':
            val = np.mean(p)
        elif c in ['R','G','B']:
            val = p[['R','G','B'].index(c)]
        elif c == 'L':
            val = 0.2126*p[0] + 0.7152*p[1] + 0.0722*p[2]
        elif c == 'S':
            mx = max(p)
            val = (mx-min(p))/mx if mx > 0 else 0
        elif c == 'C':
            val = (max(p)-min(p))
        
        self.editor.set_pipette(val)
    
    def zoom_in(self):
        self.view.scale(1.2, 1.2)
    
    def zoom_out(self):
        self.view.scale(1/1.2, 1/1.2)
    
    def zoom_11(self):
        self.view.resetTransform()
    
    def fit_view(self):
        if self.pix_item.pixmap():
            self.view.fitInView(self.pix_item, Qt.AspectRatioMode.KeepAspectRatio)
    
    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Space and not e.isAutoRepeat():
            self.show_original = True
            self.update_view()
    
    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key.Key_Space and not e.isAutoRepeat():
            self.show_original = False
            self.update_view()
    
    def reset_current_channel(self):
        self.channels[self.current_chan]['points'] = [[0.0, 0.0], [1.0, 1.0]]
        self.channels[self.current_chan]['active'] = False
        self.channels[self.current_chan]['lum_range_enabled'] = False
        self.channels[self.current_chan]['lum_min'] = 0.0
        self.channels[self.current_chan]['lum_max'] = 1.0
        self.channels[self.current_chan]['feather_sigma'] = 0.25
        self.channels[self.current_chan].pop('lut', None)
        
        self.update_ui_indicators()
        self.change_channel(self.current_chan)
        self.debounce.start()
    
    def reset_all(self):
        current_save = self.current_chan

        for k in self.channels:
            self.channels[k]['points'] = [[0.0, 0.0], [1.0, 1.0]]
            self.channels[k]['active'] = False
            self.channels[k]['lum_range_enabled'] = False
            self.channels[k]['lum_min'] = 0.0
            self.channels[k]['lum_max'] = 1.0
            self.channels[k]['feather_sigma'] = 0.25
            self.channels[k].pop('lut', None)
        
        self.update_ui_indicators()
        self.debounce.stop()
        
        self.change_channel(current_save) 
        
        self.debounce.start()
    
    def toggle_ontop(self, chk):
        flags = self.windowFlags()
        if chk:
            self.setWindowFlags(flags | Qt.WindowType.WindowStaysOnTopHint)
        else:
            self.setWindowFlags(flags & ~Qt.WindowType.WindowStaysOnTopHint)
        self.show()

    def toggle_popout(self):
        """Move the editor between the main window and a detached window."""
        if self.detached_window is None:
            self.detached_window = DetachedCurveWindow(self)
            self.detached_window.closed.connect(self.toggle_popout)
            
            self.l_layout.removeWidget(self.editor)
            
            self.placeholder.show()
            self.l_layout.insertWidget(3, self.placeholder, 1)
            
            self.editor.setParent(None)
            self.detached_window.setCentralWidget(self.editor)
            self.detached_window.show()
            self.editor.show()
            
        else:
            self.detached_window.takeCentralWidget()
            self.detached_window.close()
            self.detached_window = None
            
            self.l_layout.removeWidget(self.placeholder)
            self.placeholder.hide()
            
            self.l_layout.insertWidget(3, self.editor, 1)
            self.editor.show()
    
    def print_help(self):
        msg = [
            f"==========================================================================",
            f" VERALUX CURVES v{VERSION} — OPERATIONAL GUIDE",
            "   Spline-Based Photometric Sculpting Engine",
            "==========================================================================",
            "",
            "OVERVIEW",
            "-----------------",
            "VeraLux Curves is a spline-based photometric sculpting engine designed",
            "for surgical tonal mapping. It utilizes oscillation-free Akima interpolation",
            "to manipulate contrast across independent color domains (RGB, CIE Lab, HSV),",
            "allowing precise separation of Luminance structure from Chromatic intensity.",
            "",
            "Processing is built through explicit, non-destructive iterations.",
            "Each applied stage is part of a linear processing stack,",
            "allowing complex transformations to be constructed step-by-step",
            "with full control over rollback and refinement.",
            "",
            "Each channel supports selective application to specific luminance ranges,",
            "enabling zone-based tonal control without requiring external masks.",
            "",
            "Example workflows:",
            "  • Boost Chrominance (C) only in shadows (0-30%)",
            "  • Lift midtones Luminance (L) without affecting highlights (30-70%)",
            "  • Compress highlights (70-100%) while preserving shadow detail",
            "  • Build multi-stage color grading via successive apply iterations",
            "",
            "[1] TARGET DOMAINS (CHANNELS)",
            "   • RGB/K (Master): Linked manipulation. Affects Luminance",
            "     and Chrominance simultaneously (Standard RGB curve).",
            "   • R / G / B: Per-channel balancing for color cast removal.",
            "   • L (Luminance): Operates in CIE L*a*b* space. Adjusts",
            "     perceptual contrast without shifting Saturation or Hue.",
            "   • C (Chrominance): CIE LCH Chroma mode. Increases spectral",
            "     vivacity without altering Lightness (Structure).",
            "   • S (Saturation): HSV mode. Aggressive mathematical saturation.",
            "",
            "[2] LUMINANCE RANGE LIMITING",
            "   • Enable: Check 'Enable Range Limiting' for current channel",
            "   • Show Mask: Visualize the selection mask for the defined range.",
            "   • Min/Max Sliders: Define the luminance range (0-100%)",
            "   • Feather: Control transition softness at range boundaries",
            "   • Visual Feedback: Excluded zones appear grayed out on the curve",
            "   • Range is based on input image luminance (RGB K-average)",
            "   • Double-click any slider to reset to default values",
            "",
            "[3] INTERACTION & EDITING",
            "   • Add Point: Left Click on the grid.",
            "   • Move Point: Left Click & Drag.",
            "   • Remove Point: Right Click on a node.",
            "   • Black/White Point: Drag the start/end nodes horizontally",
            "     along the X-axis to set clipping limits.",
            "   • Pop-out (⇱): Open the graph in a separate, resizable window.",
            "",
            "[4] ITERATION, APPLY & UNDO",
            "   • All edits are applied non-destructive on top of the last stage.",
            "   • Apply: Freeze the current active curves as a new processing stage.",
            "   • Each apply is appended to a linear stage stack.",
            "   • After applying, all channels are reset to identity for the next iteration.",
            "   • Undo: Removes the most recent applied stage and rebuilds the preview.",
            "   • Undo affects applied stages only, never intermediate edits.",
            "",
            "[5] VIEWPORT & NAVIGATION",
            "   • Pan Image: Click & Drag on the preview area.",
            "   • Zoom: Mouse Wheel or GUI Buttons (+ / - / Fit / 1:1).",
            "   • Compare: Hold SPACEBAR to view the original state.",
            "   • Pipette: Click on the image to visualize the specific",
            "     pixel value on the curve graph.",
            "",
            "[6] FEEDBACK SYSTEMS",
            "   • Channel State: Modified channels appear in GOLD + Bold.",
            "   • Range Active: Channels with range limiting show ⟨⟩ symbol.",
            "   • Clipping Monitor (Blacks/Whites):",
            "     - GREY values: Pre-existing clipping in source data.",
            "     - RED/ORANGE values: Active clipping caused by the curve.",
            "   • On Saturation and Chrominance channels, the faint background histogram",
            "     represents the luminance distribution of the current applied image stage",
            "     It updates only after Apply or Undo, ensuring stable and meaningful",
            "     zone feedback.",
            "   • Mask Visualization: When 'Show Mask' is active, the preview displays the",
            "     selection as a grayscale map: WHITE represents 100% curve application,",
            "     BLACK represents total exclusion. Transitions are pixel-perfect.",
            "",
            "[7] ENGINE SPECS",
            "   • Interpolation: Akima Splines (Oscillation-free).",
            "   • Precision: Full 32-bit Float pipeline.",
            "   • Range Masking: Sigmoid with high pixel-to-pixel fidelity (no blur).",
            "",
            "Support & Info: info@veralux.space",
            "=========================================================================="
            ""
        ]
        
        try:
            [self.siril.log(l if l.strip() else " ") for l in msg]
        except:
            print("\n".join(msg))
    
    def apply_process(self):
        if self.img_full is None:
            return

        self.setEnabled(False)

        try:
            base = self.img_full
            base_lum = self.input_luminance_full

            for snap in self.stage_stack:
                if not self._has_any_active_curve(snap):
                    continue
                base = self._apply_stage(base, snap, base_lum)
                base = np.clip(base, 0.0, 1.0).astype(np.float32, copy=False)
                base_lum = np.mean(base, axis=2)

            for k in self.channels:
                self.channels[k].pop('lut', None)

            for k in self.channels:
                if self.channels[k]['active']:
                    self.channels[k]['lut'] = CurvesCore.generate_lut(self.channels[k]['points'], 65536)

            out = CurvesCore.apply_pipeline(base, self.channels, base_lum)

            if self.is_mono_source:
                out = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
                out = out.astype(np.float32)
            else:
                out = out.transpose(2, 0, 1)

            with self.siril.image_lock():
                self.siril.undo_save_state("VeraLux Curves")
                self.siril.set_image_pixeldata(out.astype(np.float32))

            self.siril.log("VeraLux Curves Applied.", LogColor.GREEN)
            self.close()

        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Error", str(e))
            self.setEnabled(True)


def main():
    app = QApplication(sys.argv)
    siril = s.SirilInterface()
    
    try:
        siril.connect()
    except:
        pass
    
    gui = VeraLuxCurves(siril, app)
    gui.show()
    app.exec()


if __name__ == "__main__":
    main()