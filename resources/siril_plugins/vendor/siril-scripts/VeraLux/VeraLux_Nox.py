##############################################
# VeraLux — Nox
# Physically-Faithful Photometric Gradient Reduction
# Author: Riccardo Paterniti (2025)
# Contact: [info@veralux.space](mailto:info@veralux.space)
##############################################

# (c) 2025 Riccardo Paterniti
# VeraLux — Nox
# SPDX-License-Identifier: GPL-3.0-or-later
# Version 1.0.1
#
# Credits / Origin
# ----------------
#   • Inspired by: The physical behavior of elastic membranes (Plate Theory)
#   • Math basis: Discrete Poisson Equation & Multiscale Variance Analysis
#   • Signal Science: Topological SNR detection & Geometric Low-Pass Filtering
#

"""
Overview
--------
A physically-faithful gradient reduction engine designed to model and subtract 
additive light pollution and vignetting while rigorously preserving faint signal.

Nox operates on the "Zenith" architecture: instead of relying on manual sample 
points, it simulates a virtual elastic membrane draped over the image data. 
By utilizing topological analysis on pure linear data, Nox distinguishes between 
low-frequency gradients (background) and high-frequency or structured signal 
(nebulae, galaxies), ensuring a mathematically subtraction-safe model.

Design Goals
------------
• Operate exclusively on Linear Data to preserve photometric integrity
• Eliminate the need for manual sample placement via Semantic Analysis
• Distinguish faint nebulosity (IFN) from background noise using Macro-Scale SNR
• Enforce Geometric Constraints to prevent model over-fitting ("dark holes")
• Provide a unified workflow that handles both stars (PSF) and dust (Topology)

Core Features
-------------
• Zenith Membrane Engine:
  - **Geometric Low-Pass:** The solver operates on a Rigid Low-Pass Grid resolution,
    physically constraining the model's flexibility. This prevents the membrane 
    from collapsing into medium-scale structures like nebula cores, forcing it 
    to glide over signal while accurately modeling broad gradients.
  - **Bicubic Reconstruction:** Ensures artifact-free upscaling of the background 
    model, eliminating ringing or undershoot around bright stars.

• Multiscale Topological Protection:
  - **Micro-Scale:** Integrates real-time PSF analysis (FWHM) to identify and 
    protect point sources (stars) from being absorbed into the gradient model.
  - **Macro-Scale (The "Deep Eye"):** Utilizes Dual-Resolution downsampled variance 
    analysis to detect ultra-faint structures (H-alpha, IFN) that are statistically 
    invisible at the pixel level.
  - **Proximity Principle:** Applies intelligent morphological dilation to 
    protected areas, recognizing that celestial objects have extended "halos" 
    that must be respected.

• Linear Physics Auto-Tune:
  - **BVI Logic:** Automatically calculates Stiffness based on the Background 
    Variability Index (Noise vs. Structure ratio).
  - **Density-Adaptive Aggression:** Dynamically caps the subtraction strength 
    based on the signal coverage percentage, ensuring safety for dense fields.

• Interactive Masking Suite:
  - While fully automated, Nox retains a complete manual masking engine.
  - Includes Brush, Lasso, and Eraser tools with real-time overlay.
  - Smart Zoom/Pan controls for precise inspection of the linear data.

Usage
-----
1. Pre-requisite: Image MUST be Linear (RGB or Mono). Crop stacking artifacts first.
2. Setup: 
   - **Auto-Masking** is ON by default (Recommended).
3. Calibration: 
   - Click **AUTO-CALCULATE**. The engine will analyze the image physics 
     (PSF, Density, SNR) and set the optimal Stiffness and Aggression.
4. Refine (Optional): 
   - Use the Paint/Lasso tools if specific complex objects need manual protection.
   - Adjust Aggression: Lower (<50%) for safety, Higher (>70%) for dark skies.
5. Process: Click PROCESS.

Inputs & Outputs
----------------
Input: Linear FITS/TIFF (RGB/Mono). 16/32-bit Int or Float.
Output: Background-Neutralized 32-bit Float FITS.

Compatibility
-------------
• Siril 1.3+
• Python 3.10+ (via sirilpy)
• Dependencies: sirilpy, PyQt6, numpy, scipy, opencv-python

License
-------
Released under GPL-3.0-or-later.
"""

import sys
import os
import glob
import traceback
import time
import math
import webbrowser

try:
    import sirilpy as s
    from sirilpy import LogColor

    s.ensure_installed(
        "PyQt6",
        "numpy",
        "opencv-python",
        "astropy",
        "scipy"
    )

except ImportError:
    print(
        "Error: sirilpy module not found.\n"
        "This script must be executed from within the Siril Python environment."
    )
    sys.exit(1)

except Exception as e:
    print("Fatal error during environment initialization:")
    traceback.print_exc()
    sys.exit(1)

import numpy as np
import cv2
from astropy.io import fits

from scipy import sparse
from scipy.sparse.linalg import cg, spsolve 
from scipy.ndimage import uniform_filter
from scipy.special import expit

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton, QGroupBox,
    QMessageBox, QProgressBar, QCheckBox, QButtonGroup,
    QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsPathItem, QGraphicsEllipseItem
)

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import (
    QImage, QPixmap, QPainter,
    QColor, QPen, QCursor, QPainterPath
)

# ---------------------
# THEME & STYLING
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
QSlider { min-height: 24px; }
QSlider::groove:horizontal { background: #444444; height: 6px; border-radius: 3px; }
QSlider::handle:horizontal { background-color: #cccccc; border: 1px solid #666666; width: 14px; height: 14px; margin: -4px 0; border-radius: 7px; }
QSlider::handle:horizontal:hover { background-color: #ffffff; border-color: #88aaff; }
QSlider#MainSlider::handle:horizontal { background-color: #ffb000; border: 1px solid #cc8800; }
QSlider#MainSlider::handle:horizontal:hover { background-color: #ffcc00; border-color: #ffffff; }
QSlider#MainSlider::sub-page:horizontal { background: #cc8800; border-radius: 3px; }
QPushButton { background-color: #444444; color: #dddddd; border: 1px solid #666666; border-radius: 4px; padding: 6px; font-weight: bold;}
QPushButton:hover { background-color: #555555; border-color: #777777; }
QPushButton:checked { background-color: #ffb000; border: 1px solid #ffcc00; color: #1a1a1a; }
QPushButton:disabled { background-color: #333333; color: #666666; border-color: #444444; }
QPushButton#ProcessButton { background-color: #285299; border: 1px solid #1e3f7a; }
QPushButton#ProcessButton:hover { background-color: #355ea1; }
QPushButton#AutoTuneButton { background-color: #285299; border: 1px solid #1e3f7a; }
QPushButton#AutoTuneButton:hover { background-color: #355ea1; }
QPushButton#CloseButton { background-color: #5a2a2a; border: 1px solid #804040; }
QPushButton#CloseButton:hover { background-color: #7a3a3a; }
QPushButton#ToolBtn { height: 28px; font-size: 9pt; font-weight: bold; color: #cccccc; background-color: #3c3c3c; border: 1px solid #555555; }
QPushButton#ToolBtn:hover { border-color: #888888; color: #ffffff; }
QPushButton#ToolBtn:checked { background-color: #ffb000; border: 1px solid #ffcc00; color: #1a1a1a; }
QPushButton#ZoomBtn { min-width: 30px; font-weight: bold; background-color: #3c3c3c; }
QProgressBar { border: 1px solid #555555; border-radius: 3px; text-align: center; }
QProgressBar::chunk { background-color: #285299; width: 10px; }
QPushButton#HelpButton { background-color: transparent; color: #555555; border: none; font-weight: bold; font-size: 11pt; min-width: 25px; }
QPushButton#HelpButton:hover { color: #aaaaaa; }

QPushButton#CoffeeButton { background-color: transparent; border: none; font-size: 15pt; padding: 2px; margin-right: 2px; }
QPushButton#CoffeeButton:hover { background-color: rgba(255,255,255,20); border-radius: 4px; }
"""

VERSION = "1.0.1"
# ------------------------------------------------------------------------------
# VERSION HISTORY
# ------------------------------------------------------------------------------
# 1.0.1: "Buy me a coffee" button added.
# ------------------------------------------------------------------------------

# =============================================================================
# HELPER: PSF PARSING
# =============================================================================

def parse_lst(lst_path):
    if not os.path.exists(lst_path): return []
    try:
        with open(lst_path, "r") as f: lines = f.readlines()
    except: return []
    
    stars = []
    for ln in lines:
        if ln.startswith("#") or not ln.strip(): continue
        parts = ln.split()
        if len(parts) < 15: continue
        try:
            stars.append({
                'X': float(parts[5]), 'Y': float(parts[6]),
                'FWHMx': float(parts[7]), 'FWHMy': float(parts[8]),
                'B': float(parts[2]), 'A': float(parts[3]),
                'Angle': float(parts[11]), 'Mag': float(parts[13])
            })
        except: continue
    return stars

def build_star_mask_from_lst(path, shape):
    stars = parse_lst(path)
    if not stars: return None
    
    H, W = shape
    mask = np.zeros((H, W), dtype=np.float32)
    
    for s in stars:
        fwhm = (s['FWHMx'] + s['FWHMy']) / 2.0
        r_mask = 1.8 * fwhm
        
        ba_ratio = min(s['B'], s['A']) / max(s['B'], s['A'])
        if ba_ratio < 0.3: r_mask *= 1.3
        if s['Mag'] < -3.0: r_mask *= 1.4
        r_mask = max(r_mask, 3.0)
        
        cx, cy = int(round(s['X'])), int(round(s['Y']))
        rad = int(r_mask + 2)
        
        y0, y1 = max(0, cy-rad), min(H, cy+rad+1)
        x0, x1 = max(0, cx-rad), min(W, cx+rad+1)
        
        if y0 >= y1 or x0 >= x1: continue
        
        yy, xx = np.ogrid[y0:y1, x0:x1]
        
        theta = np.radians(s['Angle'])
        c, sn = np.cos(theta), np.sin(theta)
        xx_rot = (xx - cx) * c - (yy - cy) * sn
        yy_rot = (xx - cx) * sn + (yy - cy) * c
        
        sx = s['FWHMx'] / 2.355
        sy = s['FWHMy'] / 2.355
        g = np.exp(-0.5 * ((xx_rot/sx)**2 + (yy_rot/sy)**2))
        
        mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], g.astype(np.float32))
        
    if np.max(mask) > 0: mask /= np.max(mask)
    return mask

# =============================================================================
# CUSTOM WIDGETS
# =============================================================================

class ResetSlider(QSlider):
    def __init__(self, orientation, default_val=0, parent=None):
        super().__init__(orientation, parent)
        self.default_val = default_val

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setValue(self.default_val)
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

class PaintView(QGraphicsView):
    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.brush_size = 50
        self.tool_mode = 'brush'
        self.is_drawing = False
        self.is_space_held = False
        self.mask_pixmap_item = None
        self.mask_image = None
        self.temp_path_item = None
        self.current_lasso_path = None
        self.preview_item = None
        self.paint_color = QColor(255, 176, 0, 100)

    def set_content(self, qimg_bg):
        # Clear existing items but preserve mask if possible or recreate
        # However, for simplicity in this flow we reset mask if size changes significantly or rely on external clear
        self.scene().clear()
        self.scene().addPixmap(QPixmap.fromImage(qimg_bg))
        w, h = qimg_bg.width(), qimg_bg.height()
        
        # If mask image exists and matches size, keep it (to allow swapping bg)
        if self.mask_image is None or self.mask_image.width() != w or self.mask_image.height() != h:
            self.mask_image = QImage(w, h, QImage.Format.Format_ARGB32)
            self.mask_image.fill(QColor(0,0,0,0))
            
        self.mask_pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(self.mask_image))
        self.mask_pixmap_item.setZValue(10) # Ensure mask is above image
        self.scene().addItem(self.mask_pixmap_item)
        
        self.scene().setSceneRect(0, 0, w, h)
        
        self.preview_item = QGraphicsEllipseItem()
        self.preview_item.setPen(QPen(QColor(136, 170, 255), 2, Qt.PenStyle.DashLine))
        self.preview_item.setZValue(100)
        self.preview_item.hide()
        self.scene().addItem(self.preview_item)

    def set_mask_visibility(self, visible):
        """Controls visibility of the red/yellow user mask overlay"""
        if self.mask_pixmap_item:
            self.mask_pixmap_item.setVisible(visible)
        # Also hide any temporary drawing paths if they exist
        if self.temp_path_item:
            self.temp_path_item.setVisible(visible)

    def fit_view(self):
        if self.scene().items():
            self.fitInView(self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def update_mask_display(self):
        if self.mask_pixmap_item and self.mask_image:
            self.mask_pixmap_item.setPixmap(QPixmap.fromImage(self.mask_image))

    def show_brush_preview(self):
        if self.preview_item:
            self.update_brush_preview_geometry()
            self.preview_item.show()

    def hide_brush_preview(self):
        if self.preview_item and not self.underMouse():
            self.preview_item.hide()

    def update_brush_preview_geometry(self, scene_pos=None):
        if not self.preview_item: return
        if scene_pos is None:
             scene_pos = self.mapToScene(self.viewport().rect().center())
        
        r = self.brush_size / 2.0
        self.preview_item.setRect(scene_pos.x() - r, scene_pos.y() - r, self.brush_size, self.brush_size)

    def enterEvent(self, event):
        self.setFocus()
        if self.tool_mode in ['brush', 'eraser'] and not self.is_space_held:
            if self.preview_item: self.preview_item.show()
        super().enterEvent(event)

    def leaveEvent(self, event):
        if self.preview_item: self.preview_item.hide()
        super().leaveEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self.is_space_held = True
            if self.preview_item: self.preview_item.hide()
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self.is_space_held = False
            self.update_custom_cursor()
        super().keyReleaseEvent(event)

    def update_custom_cursor(self):
        if self.is_space_held: return
        
        if self.tool_mode in ['brush', 'eraser']:
            self.setCursor(Qt.CursorShape.CrossCursor)
            
            if self.preview_item:
                if self.tool_mode == 'eraser':
                    self.preview_item.setPen(QPen(QColor(255, 255, 255), 2, Qt.PenStyle.DashLine))
                else:
                    self.preview_item.setPen(QPen(QColor(136, 170, 255), 2, Qt.PenStyle.DashLine))
                self.preview_item.show()
                pos = self.mapFromGlobal(QCursor.pos())
                if self.rect().contains(pos):
                    self.update_brush_preview_geometry(self.mapToScene(pos))
                    
        elif self.tool_mode == 'lasso':
            if self.preview_item: self.preview_item.hide()
            self.set_lasso_cursor()
        else:
            if self.preview_item: self.preview_item.hide()
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def set_lasso_cursor(self):
        pix = QPixmap(32, 32)
        pix.fill(QColor(0,0,0,0))
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(136, 170, 255), 2))
        path = QPainterPath()
        path.moveTo(20, 20)
        path.cubicTo(28, 10, 10, 0, 10, 10)
        path.cubicTo(10, 20, 25, 25, 20, 20)
        painter.drawPath(path)
        painter.drawLine(20, 20, 28, 28)
        painter.end()
        self.setCursor(QCursor(pix, 0, 0))

    def paint_brush_at(self, pos):
        if not self.mask_image: return
        painter = QPainter(self.mask_image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        mode = QPainter.CompositionMode.CompositionMode_Clear if self.tool_mode == 'eraser' else QPainter.CompositionMode.CompositionMode_SourceOver
        color = QColor(0,0,0,0) if self.tool_mode == 'eraser' else self.paint_color
        painter.setCompositionMode(mode)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawEllipse(pos, self.brush_size/2, self.brush_size/2)
        painter.end()
        self.update_mask_display()

    def finish_lasso(self):
        if not self.mask_image or not self.current_lasso_path: return
        if self.temp_path_item:
            self.scene().removeItem(self.temp_path_item)
            self.temp_path_item = None
        self.current_lasso_path.closeSubpath()
        painter = QPainter(self.mask_image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self.paint_color)
        painter.drawPath(self.current_lasso_path)
        painter.end()
        self.update_mask_display()
        self.current_lasso_path = None

    def mousePressEvent(self, event):
        if self.is_space_held:
            super().mousePressEvent(event)
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_drawing = True
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            pos = self.mapToScene(event.pos())
            if self.tool_mode in ['brush', 'eraser']:
                self.paint_brush_at(pos)
            elif self.tool_mode == 'lasso':
                self.current_lasso_path = QPainterPath(pos)
                self.temp_path_item = QGraphicsPathItem(self.current_lasso_path)
                pen = QPen(self.paint_color, 2)
                pen.setStyle(Qt.PenStyle.DashLine)
                self.temp_path_item.setPen(pen)
                self.temp_path_item.setBrush(QColor(0,0,0,0))
                self.scene().addItem(self.temp_path_item)
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        pos = self.mapToScene(event.pos())
        if self.tool_mode in ['brush', 'eraser'] and not self.is_space_held:
            self.update_brush_preview_geometry(pos)
            if self.preview_item and not self.preview_item.isVisible():
                 self.preview_item.show()
        if self.is_space_held:
            super().mouseMoveEvent(event)
            return
        if self.is_drawing:
            if self.tool_mode in ['brush', 'eraser']:
                self.paint_brush_at(pos)
            elif self.tool_mode == 'lasso' and self.current_lasso_path:
                self.current_lasso_path.lineTo(pos)
                if self.temp_path_item:
                    self.temp_path_item.setPath(self.current_lasso_path)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.is_drawing:
            self.is_drawing = False
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            if self.tool_mode == 'lasso':
                self.finish_lasso()
        super().mouseReleaseEvent(event)

    def clear_mask(self):
        if self.mask_image:
            self.mask_image.fill(QColor(0,0,0,0))
            self.update_mask_display()

    def get_mask_array(self):
        if not self.mask_image: return None
        ptr = self.mask_image.bits()
        ptr.setsize(self.mask_image.sizeInBytes())
        arr = np.array(ptr).reshape(self.mask_image.height(), self.mask_image.width(), 4)
        return np.flipud(arr[:, :, 3] == 0)

# =============================================================================
# CORE MATH: MEMBRANE ENGINE
# =============================================================================

class NoxCore:

    @staticmethod
    def normalize_input_img(img_data):
        if np.issubdtype(img_data.dtype, np.integer):
            info = np.iinfo(img_data.dtype)
            return img_data.astype(np.float32) / float(info.max)

        img = img_data.astype(np.float32)

        if not np.isfinite(img).any():
            return img

        vmax = np.nanmax(img)

        if vmax <= 2.0:
            return img

        if vmax <= 10.0:
            # Fast outlier guard on spatial subsample
            if img.ndim == 3:
                sub = img[::8, ::8, :]
            else:
                sub = img[::8, ::8]
            hi = np.nanpercentile(sub, 99.99)
            if hi <= 2.0:
                return img

        if vmax <= 255.0:
            return img / 255.0

        return img / 65535.0

    @staticmethod
    def MTF(x, m, lo, hi):
        m, lo, hi = float(m), float(lo), float(hi)
        dist = hi - lo
        if dist < 1e-9: return np.zeros_like(x)
        xp = np.clip((x - lo) / dist, 0.0, 1.0)
        num = (m - 1.0) * xp
        den = (2.0 * m - 1.0) * xp - m
        return num / (den + 1e-9)

    @staticmethod
    def apply_autostretch(img):
        MAD_NORM = 1.4826
        SHADOWS_CLIPPING = -2.8
        TARGET_BG = 0.25

        if img.ndim == 2:
            stride = max(1, img.size // 100000)
            sample = img.ravel()[::stride]
            median = float(np.median(sample))
            mad = float(np.median(np.abs(sample - median))) * MAD_NORM
            if mad == 0: mad = 1e-5
            c0 = median + SHADOWS_CLIPPING * mad
            c0 = max(0.0, c0)
            m_avg = median
            m2 = m_avg - c0
            midtones = NoxCore.MTF(np.array([m2]), TARGET_BG, 0.0, 1.0)[0]
            return NoxCore.MTF(img, midtones, c0, 1.0)

        channels = [img[..., 0], img[..., 1], img[..., 2]]
        sum_c0, sum_m = 0.0, 0.0
        
        for ch in channels:
            stride = max(1, ch.size // 100000)
            sample = ch.ravel()[::stride]
            median = float(np.median(sample))
            mad = float(np.median(np.abs(sample - median))) * MAD_NORM
            if mad == 0: mad = 1e-5
            sum_c0 += median + SHADOWS_CLIPPING * mad
            sum_m += median
        
        num_channels = len(channels)
        c0 = max(0.0, sum_c0 / num_channels)
        m_avg = sum_m / num_channels
        m2 = m_avg - c0
        midtones = NoxCore.MTF(np.array([m2]), TARGET_BG, 0.0, 1.0)[0]

        return NoxCore.MTF(img, midtones, c0, 1.0)

    @staticmethod
    def construct_difference_matrix_1d(n, order=2):
        if order == 1:
            e = np.ones(n)
            return sparse.spdiags([-e, e], [0, 1], n-1, n)
        elif order == 2:
            e = np.ones(n)
            return sparse.spdiags([e, -2*e, e], [0, 1, 2], n-2, n)
        return None

    @staticmethod
    def construct_regularizer_kron(h, w):
        I_h = sparse.eye(h)
        I_w = sparse.eye(w)
        D_h = NoxCore.construct_difference_matrix_1d(h, order=2)
        D_w = NoxCore.construct_difference_matrix_1d(w, order=2)
        DTD_h = D_h.T @ D_h
        DTD_w = D_w.T @ D_w
        P_y = sparse.kron(DTD_h, I_w)
        P_x = sparse.kron(I_h, DTD_w)
        return P_x + P_y

    @staticmethod
    def compute_pyramid_variance(img_channel, grid_h, grid_w, fwhm_avg):
        h, w = img_channel.shape
        scale_proxy = min(1.0, 1024.0 / max(h, w))
        h_p = int(h * scale_proxy)
        w_p = int(w * scale_proxy)
        img_proxy = cv2.resize(img_channel, (w_p, h_p), interpolation=cv2.INTER_AREA)
        
        fwhm_proxy = fwhm_avg * scale_proxy
        if fwhm_proxy < 1.0: fwhm_proxy = 1.0
        
        # 1. MICRO-SCALE LEVELS (Extended for diffuse)
        s1 = max(3, int(fwhm_proxy * 0.5)) | 1   # Sub-stellar
        s2 = max(3, int(fwhm_proxy * 2.0)) | 1   # Stellar core
        s3 = max(9, int(fwhm_proxy * 8.0)) | 1   # Stellar halo
        s4 = max(15, int(fwhm_proxy * 20.0)) | 1  # Small diffuse nebulae

        scales = [s1, s2, s3, s4]

        maps = []
        
        for win_size in scales:
            mean = uniform_filter(img_proxy, win_size, mode='reflect')
            mean_sq = uniform_filter(img_proxy**2, win_size, mode='reflect')
            variance = mean_sq - mean**2
            variance[variance < 0] = 0 
            std_dev = np.sqrt(variance)
            
            v_med = np.median(std_dev)
            v_mad = np.median(np.abs(std_dev - v_med))
            threshold = v_med + (3.0 * 1.4826 * v_mad) + (1.5 * 1.4826 * v_mad)
            if threshold < 1e-7: threshold = 1e-7
            
            arg = 10.0 * (std_dev - threshold) / threshold
            arg = np.nan_to_num(arg, nan=50.0, posinf=50.0, neginf=-50.0)
            weights = expit(-arg)
            maps.append(weights)
            
        combined_micro = (0.50 * maps[3]) + (0.30 * maps[2]) + (0.15 * maps[1]) + (0.05 * maps[0])
        
        # 2. MACRO-SCALE LAYER (Dual Resolution for diffuse)
        # Reuse proxy resolution already calculated above
        h_p_orig = h_p
        w_p_orig = w_p

        # === MACRO SCALE 1: 128x128 (Medium structures) ===
        h_macro1 = 128
        w_macro1 = 128
        img_macro1 = cv2.resize(img_channel, (w_macro1, h_macro1), interpolation=cv2.INTER_AREA)

        m1_mean = uniform_filter(img_macro1, 3, mode='reflect')
        m1_sq = uniform_filter(img_macro1**2, 3, mode='reflect')
        m1_var = m1_sq - m1_mean**2
        m1_var[m1_var < 0] = 0
        m1_std = np.sqrt(m1_var)

        m1_med = np.median(m1_std)
        m1_mad = np.median(np.abs(m1_std - m1_med))

        # SOGLIA SICURA: 1.0 sigma (Manteniamo questa perché max_grid=50 fa il resto)
        m1_thresh = m1_med + (1.0 * 1.4826 * m1_mad)
        if m1_thresh < 1e-9: m1_thresh = 1e-9

        m1_arg = 9.0 * (m1_std - m1_thresh) / m1_thresh
        m1_arg = np.nan_to_num(m1_arg, nan=50.0, posinf=50.0, neginf=-50.0)
        weights_macro1 = expit(-m1_arg)

        # Erosion for 128x128 (Standard)
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        weights_macro1 = cv2.erode(weights_macro1, kernel1, iterations=1)

        # === MACRO SCALE 2: 64x64 (Large structures) ===
        h_macro2 = 64
        w_macro2 = 64
        img_macro2 = cv2.resize(img_channel, (w_macro2, h_macro2), interpolation=cv2.INTER_AREA)

        m2_mean = uniform_filter(img_macro2, 3, mode='reflect')
        m2_sq = uniform_filter(img_macro2**2, 3, mode='reflect')
        m2_var = m2_sq - m2_mean**2
        m2_var[m2_var < 0] = 0
        m2_std = np.sqrt(m2_var)

        m2_med = np.median(m2_std)
        m2_mad = np.median(np.abs(m2_std - m2_med))

        # Original threshold for 64x64 (maintained)
        m2_thresh = m2_med + (0.70 * 1.4826 * m2_mad)
        if m2_thresh < 1e-9: m2_thresh = 1e-9

        m2_arg = 9.0 * (m2_std - m2_thresh) / m2_thresh
        m2_arg = np.nan_to_num(m2_arg, nan=50.0, posinf=50.0, neginf=-50.0)
        weights_macro2 = expit(-m2_arg)

        # Erosion for 64x64
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        weights_macro2 = cv2.erode(weights_macro2, kernel2, iterations=1)

        # === COMBINE MACRO SCALES (OR logic) ===
        # Upscale both to proxy resolution
        w_macro1_up = cv2.resize(weights_macro1, (w_p_orig, h_p_orig), interpolation=cv2.INTER_LINEAR)
        w_macro2_up = cv2.resize(weights_macro2, (w_p_orig, h_p_orig), interpolation=cv2.INTER_LINEAR)

        # Protection if EITHER scale detects signal
        w_map_macro_proxy = np.maximum(w_macro1_up, w_macro2_up)

        final_combined = combined_micro * w_map_macro_proxy
        
        w_map_grid = cv2.resize(final_combined, (grid_w, grid_h), interpolation=cv2.INTER_AREA)
        return w_map_grid

    @staticmethod
    def calculate_heuristics(img_data, star_mask=None, fwhm_val=3.0):
        """
        Scientific Auto-Tune
        """
        h, w = img_data.shape[:2]
        if img_data.ndim == 3:
            img = np.max(img_data, axis=2)
        else:
            img = img_data
            
        scale = min(1.0, 2048.0 / max(h, w))
        proxy = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        
        mask_proxy = None
        if star_mask is not None:
            mask_proxy = cv2.resize(star_mask, (proxy.shape[1], proxy.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            med = np.median(proxy)
            mad = np.median(np.abs(proxy - med))
            thresh = med + 3.0 * (1.4826 * mad)
            mask_proxy = (proxy > thresh).astype(np.float32)
            
        # 1. Feature Density
        feature_density = np.mean(mask_proxy > 0.1)
        
        aggr = 30.0 + (feature_density / 0.25) * 40.0
        aggr = np.clip(aggr, 25.0, 72.0) 
        
        # 2. Stiffness
        bg_pixels = proxy[mask_proxy <= 0.1]
        
        if len(bg_pixels) > 100:
            bg_med = np.median(bg_pixels)
            bg_mad = np.median(np.abs(bg_pixels - bg_med))
            bg_std = np.std(bg_pixels)
            
            noise_sigma = 1.4826 * bg_mad
            if noise_sigma < 1e-9: noise_sigma = 1e-9
            
            bvi = bg_std / noise_sigma
            stiff = 1.0 + (bvi - 1.0) * 1.5
        else:
            stiff = 2.0
            
        # FWHM Modulation
        if fwhm_val > 4.0:
            stiff += (fwhm_val - 4.0) * 0.15

        stiff = np.clip(stiff, 1.0, 4.0)
        
        return float(stiff), float(aggr)

    @staticmethod
    def membrane_solve_channel(img_2d, mask_2d, precomputed_variance, stiffness_val, aggr_percent, max_grid=96):
        """
        Membrane Engine
        Default grid resolution set to 96px.
        """
        h_orig, w_orig = img_2d.shape
        
        # Scale to lower resolution grid to act as geometric low-pass
        scale = min(1.0, max_grid / max(h_orig, w_orig))
        h_grid = int(h_orig * scale)
        w_grid = int(w_orig * scale)
        if h_grid < 5 or w_grid < 5: 
            return cv2.blur(img_2d, (h_orig//2, w_orig//2))
        
        y_raw = cv2.resize(img_2d, (w_grid, h_grid), interpolation=cv2.INTER_AREA)
        
        # Handle User Mask with Geometric Safety Buffer
        if mask_2d is not None:
            # Downscale using INTER_AREA to detect even fractional coverage
            m_small = cv2.resize(mask_2d.astype(np.float32), (w_grid, h_grid), interpolation=cv2.INTER_AREA)
            
            # Strict Veto: Treat any pixel that isn't 100% Sky as Masked
            m_bin = m_small > 0.999
            
            # Apply Safety Buffer: Erode 'Sky' (True) to dilate 'Protection' (False).
            # This pushes the model anchor points away from signal halos.
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            m_raw = cv2.erode(m_bin.astype(np.uint8), kernel, iterations=1).astype(bool)
        else:
            m_raw = np.ones((h_grid, w_grid), dtype=bool)

        pad_y = int(h_grid * 0.10)
        pad_x = int(w_grid * 0.10)
        
        y_pad = cv2.copyMakeBorder(y_raw, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE)
        m_pad_u8 = cv2.copyMakeBorder(m_raw.astype(np.uint8), pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE)
        v_pad = cv2.copyMakeBorder(precomputed_variance, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE)
        
        m_pad = m_pad_u8.astype(bool)
        h_p, w_p = y_pad.shape
        n_p = h_p * w_p
        y_flat = y_pad.flatten()
        v_flat = v_pad.flatten().astype(np.float32)
        
        lam = 10 ** ((float(stiffness_val) - 1.0) * 1.66 + 0.7)
        DTD = NoxCore.construct_regularizer_kron(h_p, w_p)
        
        # MAIN SOLVE
        w_flat = v_flat * m_pad.flatten().astype(np.float32)
        z_flat = np.zeros_like(y_flat) 
        n_iter = 4
        p_base = 10 ** (-3.0 - (aggr_percent / 40.0))
        max_outer_loops = 10
        epsilon = 1e-6  # Numerical stabilizer
        
        for i in range(max_outer_loops):
            # 1. Stabilized Left-Hand Side (A)
            # Add epsilon to diagonal to ensure A is always SPD/invertible
            W_lhs = sparse.spdiags(w_flat + epsilon, 0, n_p, n_p)
            A = W_lhs + lam * DTD
            
            # 2. Pure Right-Hand Side (b)
            # Use original weights to preserve target data fidelity
            b = w_flat * y_flat
            
            z_prev = z_flat.copy()
            
            use_fallback = False
            try:
                if not use_fallback:
                    z_flat, info = cg(A, b, x0=z_flat, rtol=1e-7, maxiter=500)
                    if info > 0: use_fallback = True
            except:
                use_fallback = True
                
            if use_fallback:
                z_flat = spsolve(A.tocsc(), b)
            
            diff = np.mean(np.abs(z_flat - z_prev))
            
            res = y_flat - z_flat
            mask_user = m_pad.flatten()
            new_w_dyn = np.ones_like(w_flat)
            is_above = res > 0
            
            eff_i = min(i, 4)
            p_curr = p_base / (10.0 ** eff_i)
            new_w_dyn[is_above] = p_curr
            new_w_dyn[~is_above] = 1.0 
            
            w_final = new_w_dyn * v_flat * mask_user.astype(np.float32)
            w_flat = w_final
            
            if i > 2 and diff < 1e-4:
                break

        z_pad_2d = z_flat.reshape((h_p, w_p))
        z_crop = z_pad_2d[pad_y:pad_y+h_grid, pad_x:pad_x+w_grid]
        
        # Use Cubic interpolation to avoid Lanczos ringing artifacts
        z_final = cv2.resize(z_crop, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
        
        return z_final

# =============================================================================
# WORKER THREAD
# =============================================================================

class NoxWorker(QThread):
    progress = pyqtSignal(str, int)
    finished = pyqtSignal(object, object, str)

    def __init__(self, siril_iface, img_data, user_mask, params):
        super().__init__()
        self.siril = siril_iface
        self.img = img_data
        self.user_mask = user_mask
        self.p = params
        try:
            self.wd = self.siril.get_siril_wd()
        except:
            self.wd = os.getcwd()

    def run(self):
        try:
            t0 = time.time()
            h, w = self.img.shape[:2]
            num_channels = self.img.shape[2] if self.img.ndim == 3 else 1
            
            # 1. SETUP PHYSICS (PSF & Variance)
            # Worker receives clean physics data from GUI
            self.progress.emit("Phase A: Variance Physics...", 5)
            
            fwhm_avg = self.p.get('fwhm_val', 4.0)
            star_mask = None
            
            if self.p.get('automask', True):
                star_mask = self.p.get('star_mask_data')
            
            # 2. VARIANCE PYRAMID (FWHM-Scaled + Macro SNR)
            self.progress.emit("Phase B: Pyramid (Topological)...", 15)
            
            # Grid size to 96
            max_grid = 50
            
            scale = min(1.0, max_grid / max(h, w))
            h_grid = int(h * scale)
            w_grid = int(w * scale)
            if h_grid < 5: h_grid = 5
            if w_grid < 5: w_grid = 5

            if num_channels == 3:
                img_master = np.max(self.img, axis=2)
            else:
                img_master = self.img

            master_v_map = NoxCore.compute_pyramid_variance(img_master, h_grid, w_grid, fwhm_avg)
            
            if star_mask is not None:
                # 1. Use INTER_AREA to detect even partial stars in grid cells
                sm_small = cv2.resize(star_mask, (w_grid, h_grid), interpolation=cv2.INTER_AREA)
                
                # 2. Low Threshold: If any star light is present (>1%), mark as Star
                sm_bin = sm_small > 0.01
                
                # 3. Safety Buffer: Dilate the star mask (expand protection)
                # exactly like we did for the manual mask erosion.
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                sm_dilated = cv2.dilate(sm_bin.astype(np.uint8), kernel, iterations=1)
                
                # Create Veto Map (1=Sky, 0=Star)
                veto_map = 1.0 - sm_dilated.astype(np.float32)
                master_v_map *= veto_map

            if not self.p.get('automask', True):
                master_v_map = np.ones_like(master_v_map)

            # 3. MANUAL MASK
            if self.user_mask is None:
                mask_map = np.ones((h, w), dtype=bool)
            else:
                mask_map = self.user_mask.astype(bool)

            # 4. MEMBRANE SOLVE
            stiff = self.p['stiffness']
            aggr = self.p['aggression']
            
            model_out = np.zeros_like(self.img)
            
            if num_channels == 1:
                self.progress.emit("Solving Zenith Membrane (Mono)...", 30)
                model_ch = NoxCore.membrane_solve_channel(
                    self.img, mask_map, master_v_map, stiff, aggr, max_grid
                )
                model_out = model_ch
            else:
                for c_idx in range(3):
                    p_val = 30 + (c_idx * 20)
                    self.progress.emit(f"Solving Zenith Membrane (Channel {c_idx+1}/3)...", p_val)
                    ch_data = self.img[:, :, c_idx]
                    model_ch = NoxCore.membrane_solve_channel(
                        ch_data, mask_map, master_v_map, stiff, aggr, max_grid
                    )
                    model_out[:, :, c_idx] = model_ch

            self.progress.emit("Finalizing (Smart Pedestal)...", 95)
            
            corrected = self.img - model_out
            
            TARGET_FLOOR = 0.001 
            ped_str = ""
            
            if num_channels == 3:
                # LINKED PEDESTAL LOGIC (Physically Faithful)
                # 1. Calculate Floor for each channel independently
                floors = []
                for c_idx in range(3):
                    ch_data = corrected[:, :, c_idx]
                    stride = 5
                    sample = ch_data[::stride, ::stride]
                    floors.append(float(np.percentile(sample, 0.1)))
                
                # 2. Determine the SINGLE offset needed to save the lowest channel.
                # Applying the same offset to all channels preserves RGB ratios.
                min_floor = min(floors)
                common_offset = 0.0
                
                if min_floor < TARGET_FLOOR:
                    common_offset = TARGET_FLOOR - min_floor
                
                # 3. Apply the common offset to all channels
                for c_idx in range(3):
                    corrected[:, :, c_idx] += common_offset
                    
                ped_str = f"Unified Offset: {common_offset:.5f} (RGB Ratios Preserved)"
            else:
                # Mono Logic
                stride = 5
                sample = corrected[::stride, ::stride]
                floor_val = float(np.percentile(sample, 0.1))
                offset = 0.0
                if floor_val < TARGET_FLOOR:
                    offset = TARGET_FLOOR - floor_val
                    corrected += offset
                ped_str = f"Mono Offset: {offset:.5f}"
            
            corrected = np.clip(corrected, 0.0, 1.0)
            
            dt = time.time() - t0
            
            report = (
                f"\n------------------------------------------------------------\n"
                f" VERALUX NOX v{VERSION} - SCIENTIFIC ENGINE REPORT\n"
                f"------------------------------------------------------------\n"
                f" > Engine: Zenith (PSF-Aware Membrane)\n"
                f" > Logic: Linear Physics (Golden Mean Grid)\n"
                f" > Auto-Mask: {'ON' if self.p.get('automask',True) else 'OFF'}\n"
                f" > Stiffness: {stiff:.2f}\n"
                f" > Signal Rejection: {aggr:.1f}%\n"
                f" > Smart Pedestal: Active | {ped_str}\n"
                f" > Process Time: {dt:.2f}s\n"
                f"------------------------------------------------------------\n"
            )

            self.progress.emit("Done.", 100)
            self.finished.emit(corrected, model_out, report)

        except Exception as e:
            traceback.print_exc()
            self.finished.emit(None, None, f"Error: {str(e)}")

# =============================================================================
# MAIN GUI
# =============================================================================

class NoxGui(QMainWindow):
    def __init__(self, siril, app):
        super().__init__()
        self.siril = siril
        self.app = app
        self.setWindowTitle(f"VeraLux Nox v{VERSION}")
        self.setStyleSheet(DARK_STYLESHEET)
        self.resize(1390, 650)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)
        self.img_full = None
        self.lst_file_abs = None 
        self.cached_model = None
        
        # PSF Cache State
        self.cached_star_mask = None
        self.cached_fwhm = 4.0
        self.psf_done = False
        
        print("##############################################")
        print("# VeraLux — Nox")
        print("# Physically-Faithful Photometric Gradient Reduction")
        print("# Author: Riccardo Paterniti (2025)")
        print("# Contact: info@veralux.space")
        print("##############################################")
        print(f"VeraLux Nox v{VERSION} Initialized.")
        sys.stdout.flush()

        self.init_ui()
        self.cache_input()

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(100, self.view.fit_view)

    def init_ui(self):
        main = QWidget()
        self.setCentralWidget(main)
        layout = QHBoxLayout(main)
        left_container = QWidget()
        left_container.setFixedWidth(380)
        left = QVBoxLayout(left_container)
        left.setContentsMargins(0,0,0,0)
        
        # Title Blocks
        lbl_t = QLabel("VeraLux Nox")
        lbl_t.setStyleSheet("font-size: 16pt; font-weight: bold; color: #88aaff; margin-top:5px;")
        lbl_t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left.addWidget(lbl_t)
        lbl_s = QLabel("Physically-Faithful Photometric Gradient Reduction")
        lbl_s.setStyleSheet("color: #999999; font-style: italic; margin-bottom:10px;")
        lbl_s.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left.addWidget(lbl_s)
        
        # Group 1: Physics & Auto
        g_conf = QGroupBox("1. Physics & Automation")
        l_conf = QVBoxLayout(g_conf)
        
        h_auto = QHBoxLayout()
        self.chk_automask = QCheckBox("Use PSF Auto-Masking")
        self.chk_automask.setChecked(True)
        self.chk_automask.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.chk_automask.setToolTip(
            "Automatically detects stars and faint nebulosity.\n"
            "Recommended: Leave ON (Default) for best results."
        )
        
        self.btn_autotune = QPushButton("AUTO-CALCULATE")
        self.btn_autotune.setObjectName("AutoTuneButton")
        self.btn_autotune.clicked.connect(self.run_autotune)
        self.btn_autotune.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_autotune.setToolTip("Analyzes image physics to set optimal Stiffness and Signal Rejection Power.")
        
        h_auto.addWidget(self.chk_automask)
        l_conf.addLayout(h_auto)
        l_conf.addWidget(self.btn_autotune)
        l_conf.addSpacing(10)

        l_conf.addWidget(QLabel("Signal Rejection Power:"))
        self.s_aggr = ResetSlider(Qt.Orientation.Horizontal, 50)
        self.s_aggr.setObjectName("MainSlider")
        self.s_aggr.setRange(0, 100)
        self.s_aggr.setValue(50)
        self.s_aggr.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.s_aggr.valueChanged.connect(self.update_labels)
        self.s_aggr.setToolTip(
            "Determines how strongly the model ignores non-background structures.\n"
            "Higher values (>70) = Protective (Prioritizes signal preservation).\n"
            "Lower values (<30) = Adaptive (Follows complex gradients closely)."
        )
        l_conf.addWidget(self.s_aggr)
        
        self.lbl_aggr = QLabel("Balanced")
        self.lbl_aggr.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.lbl_aggr.setStyleSheet("color: #ffb000;")
        l_conf.addWidget(self.lbl_aggr)
        
        l_conf.addWidget(QLabel("Membrane Stiffness:"))
        self.s_poly = ResetSlider(Qt.Orientation.Horizontal, 20)
        self.s_poly.setRange(10, 40)
        self.s_poly.setValue(20)
        self.s_poly.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.s_poly.valueChanged.connect(self.update_labels)
        self.s_poly.setToolTip("Tension of the background model surface.")
        l_conf.addWidget(self.s_poly)
        
        self.lbl_poly = QLabel("2.0")
        self.lbl_poly.setAlignment(Qt.AlignmentFlag.AlignRight)
        l_conf.addWidget(self.lbl_poly)
        left.addWidget(g_conf)

        # Group 2: Masking
        g2 = QGroupBox("2. Manual Masking (Optional)")
        l2 = QVBoxLayout(g2)
        h_tools = QHBoxLayout()
        
        self.btn_brush = QPushButton("PAINT")
        self.btn_brush.setObjectName("ToolBtn")
        self.btn_brush.setCheckable(True)
        self.btn_brush.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_brush.setToolTip("Manually paint areas to PROTECT from gradient removal.")
        
        self.btn_lasso = QPushButton("LASSO")
        self.btn_lasso.setObjectName("ToolBtn")
        self.btn_lasso.setCheckable(True)
        self.btn_lasso.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_lasso.setToolTip("Draw a polygon selection to PROTECT large areas.")
        
        self.btn_erase = QPushButton("ERASE")
        self.btn_erase.setObjectName("ToolBtn")
        self.btn_erase.setCheckable(True)
        self.btn_erase.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_erase.setToolTip("Remove manual protection.")
        
        self.tool_grp = QButtonGroup()
        self.tool_grp.addButton(self.btn_brush)
        self.tool_grp.addButton(self.btn_lasso)
        self.tool_grp.addButton(self.btn_erase)
        self.btn_brush.setChecked(True)
        self.btn_brush.toggled.connect(self.on_tool_change)
        self.btn_lasso.toggled.connect(self.on_tool_change)
        self.btn_erase.toggled.connect(self.on_tool_change)
        
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self.clear_mask)
        self.btn_clear.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_clear.setToolTip("Clear all manual masks.")
        
        h_tools.addWidget(self.btn_brush)
        h_tools.addWidget(self.btn_lasso)
        h_tools.addWidget(self.btn_erase)
        l2.addLayout(h_tools)
        l2.addWidget(self.btn_clear)
        
        l2.addWidget(QLabel("Brush Size:"))
        self.s_brush = ResetSlider(Qt.Orientation.Horizontal, default_val=50)
        self.s_brush.setRange(10, 200)
        self.s_brush.setValue(50)
        self.s_brush.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.s_brush.valueChanged.connect(self.update_brush_size)
        self.s_brush.setToolTip("Adjust the diameter of the paint brush.")
        l2.addWidget(self.s_brush)
        left.addWidget(g2)
        
        # Group 3: Output
        g4 = QGroupBox("3. Output")
        l4 = QVBoxLayout(g4)
        h_out = QHBoxLayout()
        
        self.chk_model = QCheckBox("Save Gradient Model")
        self.chk_model.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.chk_model.setToolTip("Saves the extracted background model as FITS.")
        
        self.btn_preview_model = QPushButton("👁")
        self.btn_preview_model.setFixedWidth(40)
        self.btn_preview_model.setCheckable(True)
        self.btn_preview_model.setEnabled(False) # Disabled until model exists
        self.btn_preview_model.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_preview_model.setToolTip("Toggle to preview the extracted Gradient Model.")
        self.btn_preview_model.toggled.connect(self.toggle_model_preview)
        
        h_out.addWidget(self.chk_model)
        h_out.addWidget(self.btn_preview_model)
        l4.addLayout(h_out)
        left.addWidget(g4)
        
        left.addStretch()
        self.progress = QProgressBar()
        left.addWidget(self.progress)
        
        lf = QHBoxLayout()
        
        # --- 1. HELP BUTTON (Ghost Style with Hover effect) ---
        self.btn_help = QPushButton("?")
        self.btn_help.setObjectName("HelpButton")
        # Definiamo sia lo stato normale che lo stato :hover
        self.btn_help.setStyleSheet("""
            QPushButton { background-color: transparent; border: none; color: #555555; font-weight: bold; font-size: 12pt; }
            QPushButton:hover { color: #aaaaaa; } 
        """)
        self.btn_help.setFixedSize(30, 30)
        self.btn_help.clicked.connect(self.print_help_to_console)
        self.btn_help.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_help.setToolTip("Print Manual to Console")
        lf.addWidget(self.btn_help)
        
        # Spazietto fisso
        lf.addSpacing(10)
        
        # --- 2. CLOSE BUTTON (Expandable 1x) ---
        b_cls = QPushButton("Close")
        b_cls.setObjectName("CloseButton")
        b_cls.clicked.connect(self.close)
        b_cls.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        lf.addWidget(b_cls, 1)

        # --- 3. PROCESS BUTTON (Expandable 2x) ---
        self.btn_proc = QPushButton("PROCESS")
        self.btn_proc.setObjectName("ProcessButton")
        self.btn_proc.clicked.connect(self.run_process)
        self.btn_proc.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_proc.setToolTip("Run the Zenith Engine.")
        lf.addWidget(self.btn_proc, 2)
        
        left.addLayout(lf)

        layout.addWidget(left_container)
        
        # Right Side (Preview)
        right_box = QVBoxLayout()
        tb_zoom = QHBoxLayout()
        
        # Zoom Buttons
        for lbl, slot, tip in [("-", self.zoom_out, "Zoom Out"), 
                               ("Fit", self.fit_view, "Fit Image"), 
                               ("1:1", self.zoom_1to1, "100% View"), 
                               ("+", self.zoom_in, "Zoom In")]:
            btn = QPushButton(lbl)
            btn.setObjectName("ZoomBtn")
            btn.clicked.connect(slot)
            btn.setToolTip(tip)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            tb_zoom.addWidget(btn)
            
        lbl_info = QLabel("(In Zoom mode, hold SPACE to Pan)")
        lbl_info.setStyleSheet("color: #ffb000; font-style: italic; margin-left: 10px;")
        tb_zoom.addWidget(lbl_info)
        tb_zoom.addStretch()

        self.btn_stretch_link = QPushButton("🔗")
        self.btn_stretch_link.setFixedWidth(30)
        self.btn_stretch_link.setCheckable(True)
        self.btn_stretch_link.setChecked(False)
        self.btn_stretch_link.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_stretch_link.setToolTip(
            "Toggle Stretch Mode:\n"
            "UNCHECKED (Default) = Unlinked (Maximizes Contrast/Analysis)\n"
            "CHECKED = Linked (Preserves Color Fidelity)"
        )
        self.btn_stretch_link.clicked.connect(lambda: self.create_proxy())
        tb_zoom.addWidget(self.btn_stretch_link)
        
        self.btn_coffee = QPushButton("☕")
        self.btn_coffee.setObjectName("CoffeeButton")
        self.btn_coffee.setToolTip("Support the developer - Buy me a coffee!")
        self.btn_coffee.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_coffee.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_coffee.clicked.connect(lambda: webbrowser.open("https://buymeacoffee.com/riccardo.paterniti"))

        self.chk_ontop = QCheckBox("On Top")
        self.chk_ontop.setChecked(True)
        self.chk_ontop.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.chk_ontop.toggled.connect(self.toggle_on_top)
        self.chk_ontop.setStyleSheet("margin-left: 10px;")
        tb_zoom.addWidget(self.chk_ontop)
        tb_zoom.addWidget(self.btn_coffee)
                
        right_box.addLayout(tb_zoom)
        
        self.scene = QGraphicsScene()
        self.view = PaintView(self.scene)
        self.view.setStyleSheet("background: #151515; border: none;")
        self.view.setFocusPolicy(Qt.FocusPolicy.StrongFocus) 
        right_box.addWidget(self.view, stretch=1)
        
        layout.addLayout(right_box, stretch=1)
        
        self.s_brush.sliderPressed.connect(self.view.show_brush_preview)
        self.s_brush.sliderReleased.connect(self.view.hide_brush_preview)
        
        self.update_labels()
        self.update_brush_size()

    def cache_input(self):
        if not self.siril.connected:
            try: self.siril.connect()
            except: pass
        try:
            with self.siril.image_lock():
                img = self.siril.get_image_pixeldata()
                if img is not None:
                    if img.ndim == 3 and img.shape[0] in [1, 3]:
                        img = img.transpose(1, 2, 0)
                    self.img_full = NoxCore.normalize_input_img(img)
                    
                    # Reset PSF Cache on new image load
                    self.cached_star_mask = None
                    self.cached_fwhm = 4.0
                    self.psf_done = False
                    self.cached_model = None
                    self.btn_preview_model.setEnabled(False)
                    self.btn_preview_model.setChecked(False)
                    
                    self.create_proxy()
                    self.btn_proc.setEnabled(True)
                else: self.btn_proc.setEnabled(False)
        except Exception as e: print(f"Cache Error: {e}")

    def create_proxy(self, data_override=None):
        """
        Generates the display image.
        If data_override is provided (e.g., the gradient model), it uses that instead of self.img_full.
        """
        src = data_override if data_override is not None else self.img_full
        if src is None: return

        h, w = src.shape[:2]
        scale = 1200 / max(h, w)
        
        if scale < 1.0:
            proxy = cv2.resize(src, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        else:
            proxy = src.copy()
            
        self.progress.setValue(10)
        QApplication.processEvents()
        
        if proxy.ndim == 3:
            if self.btn_stretch_link.isChecked():
                stretch = NoxCore.apply_autostretch(proxy)
            else:
                stretch = np.stack([NoxCore.apply_autostretch(proxy[..., c]) for c in range(3)], axis=-1)
        else:
            # Mono
            stretch = NoxCore.apply_autostretch(proxy)
            
        disp = np.flipud(np.clip(stretch * 255, 0, 255).astype(np.uint8))
        
        if disp.ndim == 3:
            h, w, c = disp.shape
            qimg = QImage(disp.data.tobytes(), w, h, c*w, QImage.Format.Format_RGB888)
        else:
            h, w = disp.shape
            qimg = QImage(disp.data.tobytes(), w, h, w, QImage.Format.Format_Grayscale8)
            
        self.view.set_content(qimg)
        self.progress.setValue(0)

    def toggle_model_preview(self, checked):
        if checked:
            # Store current stretch state
            self._last_stretch_mode = self.btn_stretch_link.isChecked()
            
            # Enforce Linked stretch for model visualization to ensure accurate color representation
            self.btn_stretch_link.blockSignals(True)
            self.btn_stretch_link.setChecked(True)
            self.btn_stretch_link.blockSignals(False)

            if self.cached_model is not None:
                self.create_proxy(self.cached_model)
                self.view.set_mask_visibility(False)
            else:
                self.btn_preview_model.setChecked(False)
        else:
            # Restore user's previous stretch mode
            if hasattr(self, '_last_stretch_mode'):
                self.btn_stretch_link.blockSignals(True)
                self.btn_stretch_link.setChecked(self._last_stretch_mode)
                self.btn_stretch_link.blockSignals(False)

            self.create_proxy(self.img_full)
            self.view.set_mask_visibility(True)

    def toggle_on_top(self):
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, self.chk_ontop.isChecked())
        self.show()
        self.raise_()
        self.activateWindow()

    def update_labels(self):
        val = self.s_aggr.value()
        txt = "Balanced"
        if val < 30: txt = "Risky (Absorbs Signal)"
        elif val > 70: txt = "Safe (Protects Signal)"
        self.lbl_aggr.setText(f"{val}% - {txt}")
        
        # Convert slider int (10-40) to float (1.0-4.0)
        stiff_raw = self.s_poly.value()
        stiff = stiff_raw / 10.0
        
        s_txt = ""
        if stiff <= 1.5: s_txt = "Elastic (Vignette)"
        elif stiff >= 3.5: s_txt = "Rigid (Nebula)"
        self.lbl_poly.setText(f"{stiff:.1f} {s_txt}")

    def on_tool_change(self):
        if self.btn_brush.isChecked(): self.view.tool_mode = 'brush'
        elif self.btn_lasso.isChecked(): self.view.tool_mode = 'lasso'
        elif self.btn_erase.isChecked(): self.view.tool_mode = 'eraser'
        self.view.update_custom_cursor()

    def update_brush_size(self):
        self.view.brush_size = self.s_brush.value()
        self.view.update_custom_cursor()
        if self.view.preview_item and self.view.preview_item.isVisible():
            self.view.update_brush_preview_geometry()

    def clear_mask(self):
        self.view.clear_mask()

    def zoom_in(self): self.view.scale(1.2, 1.2)
    def zoom_out(self): self.view.scale(1/1.2, 1/1.2)
    def zoom_1to1(self): self.view.resetTransform()
    def fit_view(self): self.view.fit_view()

    # Centralized PSF Analysis
    def ensure_psf_analysis(self):
        """
        Runs findstar once on the current image state (usually original)
        and caches the result. This avoids running findstar on an already
        processed image during subsequent 'Process' clicks.
        """
        if self.psf_done:
            return # Use cached data
            
        if self.img_full is None: return

        try:
            self.siril.log("VeraLux: Measuring PSF for Physics Engine...", LogColor.BLUE)
            
            # Run findstar on the CURRENT image in Siril (which is original at first load)
            self.siril.cmd("findstar -out=nox_stars")
            
            try:
                wd = self.siril.get_siril_wd()
            except:
                wd = os.getcwd()
                
            lst_path = os.path.join(wd, "nox_stars.lst")
            if not os.path.exists(lst_path):
                lst_path = os.path.join(wd, "nox_stars.txt")
            
            if os.path.exists(lst_path):
                h, w = self.img_full.shape[:2]
                self.cached_star_mask = build_star_mask_from_lst(lst_path, (h, w))
                
                stars = parse_lst(lst_path)
                if stars:
                    fwhms = [(s['FWHMx'] + s['FWHMy'])/2.0 for s in stars]
                    med_f = float(np.median(fwhms))
                    if med_f < 1.0: med_f = 1.0
                    self.cached_fwhm = med_f
                    self.siril.log(f"VeraLux: Measured FWHM = {self.cached_fwhm:.2f} px")
            
            # Clean up the visualization
            self.siril.cmd("clearstar")
            self.psf_done = True
            
        except Exception as e:
             self.siril.log(f"VeraLux: PSF Analysis Warning: {e}")
             self.psf_done = True
             
    def run_autotune(self):
        if self.img_full is None: return
        self.progress.setValue(10)
        QApplication.processEvents()
        
        # 1. Ensure we have physics data
        self.ensure_psf_analysis()
        
        self.progress.setValue(30)
        
        try:
            stiff, aggr = NoxCore.calculate_heuristics(
                self.img_full, 
                star_mask=self.cached_star_mask, 
                fwhm_val=self.cached_fwhm
            )
            
            # Map float stiffness to integer slider (x10)
            s_val = int(round(stiff * 10))
            s_val = max(10, min(40, s_val))
            
            self.s_poly.setValue(s_val)
            self.s_aggr.setValue(int(round(aggr)))
            
            try: self.siril.log(f"Auto-Tune (Linear Physics): Stiffness={stiff:.2f}, Rejection Power={aggr:.1f}%")
            except: pass
        except: pass
        self.progress.setValue(0)

    def run_process(self):
        if self.img_full is None: return

        # Reset Preview Mode if active
        if self.btn_preview_model.isChecked():
            self.btn_preview_model.setChecked(False)
        self.btn_preview_model.setEnabled(False)

        mask_proxy = self.view.get_mask_array()
        h_full, w_full = self.img_full.shape[:2]
        
        if mask_proxy is not None:
            # Resize alla risoluzione piena
            temp_mask = cv2.resize(mask_proxy.astype(np.uint8), (w_full, h_full), interpolation=cv2.INTER_NEAREST)
            
            # Micro-Espansione della Protezione (Safety Seal)
            # Vogliamo espandere leggermente l'area dipinta (valore 0) per coprire l'anti-aliasing del pennello.
            # Poiché Paint=0 e Cielo=1, usiamo ERODE sul Cielo per far crescere il Paint.
            # Un kernel 3x3 espande la protezione di 1 solo pixel: chirurgico.
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            temp_mask = cv2.erode(temp_mask, kernel, iterations=1)
            
            user_mask = temp_mask.astype(bool)
        else:
            user_mask = None
        
        # 1. Ensure physics data is ready
        self.ensure_psf_analysis()
        
        p = { 
            'aggression': self.s_aggr.value(), 
            # Convert integer slider back to float for the worker
            'stiffness': self.s_poly.value() / 10.0,
            'automask': self.chk_automask.isChecked(),
            # Pass cached physics data directly to worker
            'star_mask_data': self.cached_star_mask,
            'fwhm_val': self.cached_fwhm
        }
        
        self.worker = NoxWorker(self.siril, self.img_full, user_mask, p)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.setEnabled(False)
        self.worker.start()

    def on_progress(self, msg, val):
        self.progress.setValue(val)
        self.progress.setFormat(msg)
        try:
            if isinstance(msg, str) and msg.startswith("VeraLux"):
                self.siril.log(msg)
        except: pass

    def on_finished(self, corrected, model, report):
        self.setEnabled(True)
        if corrected is None:
            QMessageBox.critical(self, "Error", report)
            return

        # Store the model for preview
        self.cached_model = model
        self.btn_preview_model.setEnabled(True)
            
        out_siril = corrected.transpose(2, 0, 1) if corrected.ndim == 3 else corrected
        with self.siril.image_lock():
            self.siril.undo_save_state("VeraLux Nox")
            self.siril.set_image_pixeldata(out_siril.astype(np.float32))
            
            for line in report.split('\n'):
                if line.strip(): self.siril.log(line)
            
            if self.chk_model.isChecked():
                model_siril = model.transpose(2, 0, 1) if model.ndim == 3 else model
                path = os.path.join(self.siril.get_siril_wd(), "VeraLux_Nox_Background_Model.fit")
                try:
                    hdu = fits.PrimaryHDU(np.ascontiguousarray(model_siril, dtype=np.float32))
                    hdu.writeto(path, overwrite=True)
                    self.siril.log(f"Model saved to {path}", LogColor.BLUE)
                except Exception as e:
                    self.siril.log(f"Error saving model: {e}", LogColor.RED)
            
        self.progress.setValue(100)
        self.progress.setFormat("Done. Check Siril.")

    def cleanup_temp_files(self):
        try:
            try:
                wd = self.siril.get_siril_wd()
            except:
                wd = os.getcwd()
            
            targets = ["nox_stars.lst", "nox_stars.txt", "nox_stars"]
            
            for fname in targets:
                fpath = os.path.join(wd, fname)
                if os.path.exists(fpath):
                    try:
                        os.remove(fpath)
                    except:
                        pass
        except:
            pass  
        
    def closeEvent(self, event):
        self.cleanup_temp_files()
        QApplication.instance().quit()
        super().closeEvent(event)

    def print_help_to_console(self):
        guide_lines = [
            "==========================================================================",
            f"   VERALUX NOX v{VERSION} - OPERATIONAL GUIDE",
            "   Physically-Faithful Photometric Gradient Reduction",
            "==========================================================================",
            "",
            "OVERVIEW",
            "-----------------",
            "VeraLux Nox implements a deterministic 'Zenith Membrane' engine.",
            "Instead of relying on statistical curve fitting via manual samples,",
            "it solves the Discrete Poisson Equation to model the background",
            "as a physical elastic surface. This approach topologically separates",
            "light pollution from faint signal, ensuring photometric integrity.",
            "",
            "1. PREPARATION",
            "   • This tool works best on LINEAR data (right after stacking/cropping).",
            "   • Crop stacking artifacts (black borders) before starting.",
            "",
            "2. TUNING & WORKFLOW",
            "   • STEP A: Click [AUTO-CALCULATE].",
            "     The engine analyzes density, PSF (Stars), and background variability",
            "     to set optimal Stiffness and Aggression.",
            "   • STEP B (Optional): Use Paint/Lasso to manually protect specific areas",
            "     like galaxies or nebulae if Auto-Mask misses them. You will notice that,",
            "     in most cases, there is no need to mask manually (even if painting masks",
            "     is oddly satisfying).",
            "   • STEP C: Click [PROCESS].",
            "",
            "3. UNDO MANAGEMENT",
            "   • The script caches the pristine original image upon launch.",
            "   • You can iterate freely: adjust parameters and click [PROCESS] again",
            "     without needing to perform an Undo in Siril first.",
            "   • The engine always applies the new model to the cached original data,",
            "     ensuring mathematical consistency.",
            "",
            "4. CORE PHYSICS EXPLAINED",
            "   • Auto-Masking: Uses a dual-layer strategy.",
            "     1. Micro-Scale: Protects stars using FWHM measurements.",
            "     2. Macro-Scale: Detects faint H-alpha/IFN veils using binning statistics.",
            "",
            "   • Membrane Stiffness (Tension):",
            "     Low (1.0-1.5): 'Elastic'. Good for vignetting.",
            "     High (3.0-4.0): 'Rigid'. Best for large objects.",
            "",
            "   • Signal Rejection Power:",
            "     Lower (<30%): Adaptive. Allows the model to follow complex local gradients.",
            "     Higher (>70%): Protective. Maximizes preservation of faint structures.",
            "",
            "Support & Info: info@veralux.space",
            "=========================================================================="
        ]
        try:
            for line in guide_lines:
                msg = line if line.strip() else " "
                self.siril.log(msg)
        except:
            print("\n".join(guide_lines))

def main():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    siril = s.SirilInterface()
    try: siril.connect()
    except: pass

    gui = NoxGui(siril, app)
    gui.show()
    app.exec()

if __name__ == "__main__":
    main()