##############################################
# VeraLux — Revela
# Photometric Local Contrast & Texture Engine
# Author: Riccardo Paterniti (2025)
# Contact: info@veralux.space
##############################################

# (c) 2025 Riccardo Paterniti
# VeraLux — Revela
# SPDX-License-Identifier: GPL-3.0-or-later
# Version 1.0.2
#
# Credits / Origin
# ----------------
#   • Architecture: VeraLux Shared GUI Framework (StarComposer Style)
#   • Engine: Modified À Trous Wavelet Transform (ATWT)
#   • Logic: Tri-Domain Scale Processing (Texture/Structure/Dynamic)
#   • Protection: VeraLux Shadow Authority & Star Isolation
#

"""
Overview
--------
VeraLux Revela is a post-stretch enhancement engine designed to reveal
micro-contrast and volumetric structures in nebulae and galaxies without
compromising the noise floor or star profiles.

It employs a fully adaptive "Signal-Aware" approach:
1. **Adaptive Noise Gate**: Automatically calculates the noise floor (MAD/Sigma) 
   of the specific image. The 'Shadow Authority' slider scales this physical limit.
2. **Frequency Separation**: Decouples Micro-Contrast (Texture) from Macroscopic 
   Volume (Structure) using À Trous Wavelets.
3. **Geometric Protection**: Automatically identifies high-energy stellar profiles 
   to prevent sharpening artifacts (haloes/ringing) on stars.

Key Features
------------
• **High-Fidelity Preview**: Dual-stage rendering (Proxy/FullRes) with 
  Energy-Preserving Interpolation.
• **Tri-Domain Control**: Separate sliders for Texture and Structure.
• **Shadow Authority**: Adaptive photometric gate based on image statistics.
• **Mask Visualization**: Real-time preview of the protection gate (White=Active, Black=Protected).
• **Star Protection**: Energy-based masking to prevent "raccoon eyes".
• **Smart Workflow**: Non-blocking background processing for high-resolution output.

Usage
-----
1. **Input**: Designed for NON-LINEAR (Stretched) images to enhance visual
   volume and detail.
2. **Tuning**:
   - **Texture**: Pops fine details (shock fronts, dust lanes).
   - **Structure**: Enhances the 3D volume of the object.
   - **Show Mask**: Check this to see what is being protected.
   - **Shadow Authority**: Adjust this while viewing the mask until the background is black.
3. **Process**: Applies the enhancement.

Inputs & Outputs
----------------
Input: Stretched RGB/Mono Image (open in Siril).
Output: Enhanced Image loaded back into Siril.

Compatibility
-------------
• Siril 1.3+
• Python 3.10+ (via sirilpy)
• Dependencies: sirilpy, PyQt6, numpy, opencv-python

License
-------
Released under GPL-3.0-or-later.

This script is part of the VeraLux family of tools —
Python utilities focused on physically faithful astrophotography workflows.
"""

import sys
import os
import traceback
import webbrowser

try:
    import sirilpy as s
    from sirilpy import LogColor
except ImportError:
    print("Error: sirilpy module not found.")
    sys.exit(1)

try:
    s.ensure_installed("PyQt6", "numpy", "opencv-python")
except AttributeError:
    print("Warning: sirilpy.ensure_installed not found. Assuming dependencies are met.")

import numpy as np
import cv2

from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                            QWidget, QLabel, QSlider, QPushButton, QGroupBox,
                            QMessageBox, QCheckBox, QProgressBar, QGraphicsView, 
                            QGraphicsScene, QGraphicsPixmapItem)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSettings, QEvent, QPointF
from PyQt6.QtGui import QImage, QPixmap, QColor, QKeyEvent, QPainter

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

/* Texture Slider (Cyan) */
QSlider#TexSlider::handle:horizontal { background-color: #00cccc; border: 1px solid #008888; }
QSlider#TexSlider::handle:horizontal:hover { background-color: #00ffff; border-color: #ffffff; }

/* Structure Slider (Gold) */
QSlider#StrSlider::handle:horizontal { background-color: #ffb000; border: 1px solid #cc8800; }
QSlider#StrSlider::handle:horizontal:hover { background-color: #ffcc00; border-color: #ffffff; }

QPushButton { background-color: #444444; color: #dddddd; border: 1px solid #666666; border-radius: 4px; padding: 6px; font-weight: bold;}
QPushButton:hover { background-color: #555555; border-color: #777777; }
QPushButton#ProcessButton { background-color: #285299; border: 1px solid #1e3f7a; }
QPushButton#ProcessButton:hover { background-color: #355ea1; }
QPushButton#CloseButton { background-color: #5a2a2a; border: 1px solid #804040; }
QPushButton#CloseButton:hover { background-color: #7a3a3a; }

/* Zoom buttons */
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

/* Progress bar style */
QProgressBar {
    background-color: #3c3c3c;
    border: none;
    border-radius: 2px;
}
QProgressBar::chunk {
    background-color: #ffb000;
    border-radius: 2px;
}
"""

VERSION = "1.0.2"
# ------------------------------------------------------------------------------
# VERSION HISTORY
# ------------------------------------------------------------------------------
# 1.0.2: Fix threading crash (Sig 6) & extended Mask dynamic range.
# 1.0.1: "Buy me a coffee" button added.
# ------------------------------------------------------------------------------

# =============================================================================
#  CORE MATH: WAVELETS & MASKS
# =============================================================================

class StructureCore:
    
    @staticmethod
    def normalize_input(img_data):
        input_dtype = img_data.dtype
        img_float = img_data.astype(np.float32)
        if np.issubdtype(input_dtype, np.integer):
            if input_dtype == np.uint8: return img_float / 255.0
            elif input_dtype == np.uint16: return img_float / 65535.0
            else: return img_float / float(np.iinfo(input_dtype).max)
        elif np.issubdtype(input_dtype, np.floating):
            current_max = np.max(img_data)
            if current_max <= 1.0 + 1e-5: return img_float
            if current_max <= 65535.0: return img_float / 65535.0
            return img_float
        return img_float

    @staticmethod
    def rgb_to_lab(rgb_float32):
        """RGB to Lab conversion using OpenCV for speed and accuracy."""
        return cv2.cvtColor(rgb_float32, cv2.COLOR_RGB2Lab)

    @staticmethod
    def lab_to_rgb(lab_float32):
        """Lab to RGB conversion using OpenCV."""
        return cv2.cvtColor(lab_float32, cv2.COLOR_Lab2RGB)

    @staticmethod
    def atrous_decomposition(img2d, n_scales=6):
        current = img2d.astype(np.float32)
        planes = []
        # Kernel B3 Spline 1D
        kernel_1d = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16.0
        
        for s in range(n_scales):
            step = 2 ** s
            # Create dilated kernel (à trous)
            k_size = 5 + (4 * (step - 1))
            k_dilated = np.zeros((k_size,), dtype=np.float32)
            k_dilated[0::step] = kernel_1d
            
            # Fast separable convolution with OpenCV
            smooth = cv2.sepFilter2D(current, -1, k_dilated, k_dilated, borderType=cv2.BORDER_REFLECT)
            
            detail = current - smooth
            planes.append(detail)
            current = smooth
            
        return planes, current

    @staticmethod
    def compute_signal_mask(L, threshold_sigma):
        """VeraLux Shadow Authority Logic."""
        L_norm = L / 100.0
        median = np.median(L_norm)
        mad = np.median(np.abs(L_norm - median))
        sigma = 1.4826 * mad
        if sigma < 1e-6: sigma = 1e-6
        
        noise_floor = median + (threshold_sigma * sigma)
        mask = (L_norm - noise_floor) / (2.0 * sigma + 1e-9)
        mask = np.clip(mask, 0.0, 1.0)
        mask = cv2.blur(mask, (3, 3), borderType=cv2.BORDER_REFLECT)
        return mask

    @staticmethod
    def compute_star_protection(planes, strength=1.0):
        """Heuristic Star Protection."""
        if strength <= 0: return np.ones_like(planes[0])

        e_fine = np.abs(planes[0]) + np.abs(planes[1])
        e_mid  = np.abs(planes[2]) + np.abs(planes[3])
        energy = e_fine + 0.5 * e_mid

        med = np.median(energy)
        mad = np.median(np.abs(energy - med))
        sigma = 1.4826 * mad
        if sigma < 1e-6: sigma = 1e-6

        thr = med + (4.0 * sigma)
        width = 2.0 * sigma
        star_map = np.clip((energy - thr) / (width + 1e-9), 0.0, 1.0)

        star_map = cv2.blur(star_map, (5, 5), borderType=cv2.BORDER_REFLECT)
        star_map = np.clip(star_map * 1.5, 0.0, 1.0)
        star_map = cv2.blur(star_map, (5, 5), borderType=cv2.BORDER_REFLECT)

        protection = 1.0 - (star_map * strength)
        return np.clip(protection, 0.0, 1.0)

    @staticmethod
    def process_structure(img_input, texture_amt, structure_amt, shadow_auth, protect_stars, return_mask=False):
        """
        Main Engine. Supports both RGB (3 channels) and Mono (2D arrays).
        """
        is_mono = (img_input.ndim == 2)

        if is_mono:
            # For mono, L is just the scaled image data.
            L = img_input * 100.0
        else:
            # For color, convert to Lab and extract L channel.
            lab = StructureCore.rgb_to_lab(img_input.astype(np.float32))
            L = lab[..., 0]
        
        planes, residual = StructureCore.atrous_decomposition(L, n_scales=6)
        
        sigma_thresh = (shadow_auth * 0.12) - 3.0 
        signal_mask = StructureCore.compute_signal_mask(L, sigma_thresh)
        
        star_mask = 1.0
        star_mask_structure = 1.0
        if protect_stars:
            star_mask = StructureCore.compute_star_protection(planes, strength=1.0)
            star_mask_structure = np.clip(star_mask ** 2.0, 0.0, 1.0)

        active_mask_tex = signal_mask * star_mask
        active_mask_str = signal_mask * star_mask_structure
        
        # FEATURE: Return Mask Mode
        if return_mask:
            # We return the "Structure" mask as it is the most critical one (includes star core protection)
            # Normalizing 0.0-1.0 (Black=Protected, White=Active)
            return active_mask_str

        t_gain = 1.0 + (texture_amt * 1.5) 
        planes[0] *= (1.0 + (t_gain - 1.0) * active_mask_tex)
        planes[1] *= (1.0 + (t_gain - 1.0) * active_mask_tex)
        
        s_gain = 1.0 + (structure_amt * 1.0) 
        planes[2] *= (1.0 + (s_gain - 1.0) * active_mask_str)
        planes[3] *= (1.0 + (s_gain - 1.0) * active_mask_str)
        planes[4] *= (1.0 + (s_gain - 1.0) * active_mask_str)
        
        L_new = residual + sum(planes)
        L_new = np.clip(L_new, 0.0, 100.0)
        
        if is_mono:
            return L_new / 100.0
        else:
            lab[..., 0] = L_new
            rgb_out = StructureCore.lab_to_rgb(lab)
            return np.clip(rgb_out, 0.0, 1.0)

# =============================================================================
#  WORKER THREADS
# =============================================================================

class ProcessingWorker(QThread):
    result_ready = pyqtSignal(object, int)
    
    def __init__(self, img, params, job_id):
        super().__init__()
        self.img = img
        self.p = params
        self.job_id = job_id
        
    def run(self):
        try:
            res = StructureCore.process_structure(self.img, **self.p)
            self.result_ready.emit(res, self.job_id)
        except Exception as e:
            print(f"Error in worker: {e}")
            self.result_ready.emit(None, self.job_id)

# =============================================================================
#  GUI HELPERS
# =============================================================================

class ResetSlider(QSlider):
    def __init__(self, orientation, default_val=0, parent=None):
        super().__init__(orientation, parent)
        self.default_val = default_val
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        
    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setValue(self.default_val); event.accept()
        else: super().mouseDoubleClickEvent(event)

# =============================================================================
#  MAIN GUI
# =============================================================================

class StructureGUI(QMainWindow):
    def __init__(self, siril, app):
        super().__init__()
        self.siril = siril
        self.app = app
        self.setWindowTitle(f"VeraLux Revela v{VERSION}")
        self.setStyleSheet(DARK_STYLESHEET)
        self.resize(1200, 600)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)
        
        self.settings = QSettings("VeraLux", "Revela")
        
        self.img_full = None
        self.img_proxy = None
        self.comp_proxy = None
        self._current_display_buffer = None
        
        self.show_original = False
        self.is_fit_mode = True
        self.has_image = False

        self.worker = None
        self.current_job_id = 0

        self.debounce = QTimer()
        self.debounce.setSingleShot(True); self.debounce.setInterval(150)
        self.debounce.timeout.connect(self.run_preview_logic)
        
        self.release_timer = QTimer()
        self.release_timer.setSingleShot(True); self.release_timer.setInterval(350)
        self.release_timer.timeout.connect(self.trigger_full_res_update)
        
        header_msg = (
            f"\n##############################################\n"
            f"# VeraLux — Revela v{VERSION}\n"
            "# Photometric Local Contrast & Texture Engine\n"
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
        
        left_container = QWidget(); left_container.setFixedWidth(380)
        left = QVBoxLayout(left_container); left.setContentsMargins(0,0,0,0)
        
        lbl_title = QLabel("VeraLux Revela"); lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_title.setStyleSheet("font-size: 16pt; font-weight: bold; color: #88aaff; margin-top: 5px;")
        left.addWidget(lbl_title)

        lbl_subtitle = QLabel("Photometric Local Contrast & Texture Engine"); lbl_subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_subtitle.setStyleSheet("font-size: 10pt; color: #999999; font-style: italic; margin-bottom: 15px;")
        left.addWidget(lbl_subtitle)
        
        g1 = QGroupBox("1. Enhancement")
        v1 = QVBoxLayout(g1)
        v1.addWidget(QLabel("Texture (Micro-Contrast):"))
        self.s_tex = ResetSlider(Qt.Orientation.Horizontal, 0); self.s_tex.setRange(0, 100)
        self.s_tex.setObjectName("TexSlider"); self.s_tex.setToolTip("Enhances fine high-frequency details (dust lanes, shock fronts).")
        self.s_tex.valueChanged.connect(self.trigger_update); self.s_tex.sliderReleased.connect(self.release_timer.start)
        v1.addWidget(self.s_tex)
        
        v1.addWidget(QLabel("Structure (Volume):"))
        self.s_str = ResetSlider(Qt.Orientation.Horizontal, 0); self.s_str.setRange(0, 100)
        self.s_str.setObjectName("StrSlider"); self.s_str.setToolTip("Enhances medium-frequency volume and body of the nebula.")
        self.s_str.valueChanged.connect(self.trigger_update); self.s_str.sliderReleased.connect(self.release_timer.start)
        v1.addWidget(self.s_str)
        left.addWidget(g1)
        
        g2 = QGroupBox("2. Protection (The Gate)")
        v2 = QVBoxLayout(g2)
        v2.addWidget(QLabel("Shadow Authority (Noise Gate):"))
        
        # Layout for Checkbox + Slider
        h_mask = QHBoxLayout()
        h_mask.setContentsMargins(0,0,0,0)
        
        self.s_shad = ResetSlider(Qt.Orientation.Horizontal, 33); self.s_shad.setRange(0, 100); self.s_shad.setValue(33)
        self.s_shad.setToolTip("<b>Adaptive Gate:</b> Move this to adjust the black areas in the Mask View.")
        self.s_shad.valueChanged.connect(self.trigger_update); self.s_shad.sliderReleased.connect(self.release_timer.start)
        
        self.chk_show_mask = QCheckBox("Show")
        self.chk_show_mask.setToolTip("Enable to visualize the protection mask.\nWhite = Active, Black = Protected.")
        self.chk_show_mask.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.chk_show_mask.toggled.connect(self.trigger_update) # Instant update
        self.chk_show_mask.toggled.connect(self.trigger_full_res_update) # Full quality update
        
        h_mask.addWidget(self.s_shad)
        h_mask.addWidget(self.chk_show_mask)
        v2.addLayout(h_mask)
        
        self.chk_star = QCheckBox("Isolate Stars (Prevent Raccoon Eyes)"); self.chk_star.setChecked(True)
        self.chk_star.setToolTip("Detects high-energy stellar profiles and excludes them from sharpening.")
        self.chk_star.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.chk_star.toggled.connect(self.trigger_full_res_update)
        v2.addWidget(self.chk_star)
        left.addWidget(g2)
        
        left.addStretch()
        
        footer = QHBoxLayout()
        self.btn_help = QPushButton("?"); self.btn_help.setObjectName("HelpButton"); self.btn_help.setFixedWidth(20); self.btn_help.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_help.clicked.connect(self.print_help); footer.addWidget(self.btn_help)
        
        b_def = QPushButton("Defaults"); b_def.setFocusPolicy(Qt.FocusPolicy.NoFocus); b_def.clicked.connect(self.set_defaults); footer.addWidget(b_def)
        b_cls = QPushButton("Close"); b_cls.setObjectName("CloseButton"); b_cls.setFocusPolicy(Qt.FocusPolicy.NoFocus); b_cls.clicked.connect(self.close); footer.addWidget(b_cls)
        b_proc = QPushButton("PROCESS"); b_proc.setObjectName("ProcessButton"); b_proc.setFocusPolicy(Qt.FocusPolicy.NoFocus); b_proc.clicked.connect(self.apply_process); footer.addWidget(b_proc)
        left.addLayout(footer); layout.addWidget(left_container)
        
        right = QVBoxLayout()
        tb = QHBoxLayout()
        b_out = QPushButton("-"); b_out.setObjectName("ZoomBtn"); b_out.setFocusPolicy(Qt.FocusPolicy.NoFocus); b_out.clicked.connect(self.zoom_out)
        b_fit = QPushButton("Fit"); b_fit.setObjectName("ZoomBtn"); b_fit.setFocusPolicy(Qt.FocusPolicy.NoFocus); b_fit.clicked.connect(self.fit_view)
        b_11 = QPushButton("1:1"); b_11.setObjectName("ZoomBtn"); b_11.setFocusPolicy(Qt.FocusPolicy.NoFocus); b_11.clicked.connect(self.zoom_1to1)
        b_in = QPushButton("+"); b_in.setObjectName("ZoomBtn"); b_in.setFocusPolicy(Qt.FocusPolicy.NoFocus); b_in.clicked.connect(self.zoom_in)
        
        lbl_hint = QLabel("Hold SPACE to Compare / Double-click to Fit"); lbl_hint.setStyleSheet("color: #ffb000; font-size: 8pt; font-style: italic; margin-left: 10px; font-weight: bold;")
        self.chk_ontop = QCheckBox("On Top"); self.chk_ontop.setChecked(True); self.chk_ontop.toggled.connect(self.toggle_ontop)
        self.chk_ontop.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        
        tb.addWidget(b_out); tb.addWidget(b_fit); tb.addWidget(b_11); tb.addWidget(b_in); tb.addWidget(lbl_hint); tb.addStretch()
        self.pbar = QProgressBar(); self.pbar.setRange(0, 0); self.pbar.setFixedWidth(210); self.pbar.setFixedHeight(4); self.pbar.setTextVisible(False); self.pbar.hide()
        
        self.btn_coffee = QPushButton("☕")
        _nofocus(self.btn_coffee)
        self.btn_coffee.setObjectName("CoffeeButton")
        self.btn_coffee.setToolTip("Support the developer - Buy me a coffee!")
        self.btn_coffee.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_coffee.clicked.connect(lambda: webbrowser.open("https://buymeacoffee.com/riccardo.paterniti"))

        tb.addWidget(self.pbar); tb.addWidget(self.chk_ontop); tb.addWidget(self.btn_coffee); right.addLayout(tb)
        
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene); self.view.setStyleSheet("background-color: #151515; border: none;")
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag); self.view.viewport().installEventFilter(self); self.view.installEventFilter(self)
        right.addWidget(self.view)
        
        self.pix_item = QGraphicsPixmapItem(); self.scene.addItem(self.pix_item); self.pix_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.lbl_blink = QLabel("ORIGINAL", self.view); self.lbl_blink.setStyleSheet("background-color: rgba(255, 160, 0, 200); color: #ffffff; font-size: 14pt; font-weight: bold; padding: 8px 16px; border-radius: 6px;"); self.lbl_blink.hide()
        layout.addLayout(right)

    def make_proxy(self, img):
        if img is None: return None
        target_size = 2560; h, w = img.shape[:2]
        if max(h, w) <= target_size: return img
        scale = target_size / max(h, w); new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def cache_input(self):
        try:
            if not self.siril.connected: self.siril.connect()
            with self.siril.image_lock(): img = self.siril.get_image_pixeldata()
            if img is None: return
            
            img = StructureCore.normalize_input(img)
            self.img_full = img.transpose(1, 2, 0) if img.ndim == 3 else img
            self.img_proxy = self.make_proxy(self.img_full)
            self.comp_proxy = self.img_proxy.copy()
            
            full_h, full_w = self.img_full.shape[:2]
            self.scene.setSceneRect(0, 0, full_w, full_h)
            self.has_image = True
            
            self.update_view(force_render=True)
            self.fit_view()
            self.trigger_full_res_update()
            
        except Exception as e: print(f"Cache Error: {e}")

    def get_current_params(self):
        return {
            'texture_amt': self.s_tex.value()/100.0, 
            'structure_amt': self.s_str.value()/100.0,
            'shadow_auth': self.s_shad.value(), 
            'protect_stars': self.chk_star.isChecked(),
            'return_mask': self.chk_show_mask.isChecked()
        }

    def trigger_update(self):
        if self.img_proxy is None: return
        self.debounce.start()

    def run_preview_logic(self):
        params = self.get_current_params()
        self.comp_proxy = StructureCore.process_structure(self.img_proxy, **params)
        self.update_view()

    def trigger_full_res_update(self):
        if self.img_full is None: return
        self.release_timer.stop(); self.debounce.stop()
        self.pbar.show()
        
        self.current_job_id += 1 
        params = self.get_current_params()
        
        if self.worker is not None:
            try:
                if self.worker.isRunning():
                    try: self.worker.result_ready.disconnect()
                    except: pass                    
                    self.worker.setParent(self)
            except RuntimeError:
                pass 

        self.worker = ProcessingWorker(self.img_full, params, self.current_job_id)
        self.worker.result_ready.connect(self.on_full_res_ready)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.start()

    def on_full_res_ready(self, img_result, job_id):
        if job_id != self.current_job_id: return 

        self.pbar.hide()
        if img_result is not None:
            self.comp_proxy = img_result
            self.update_view(force_render=True)

    def update_view(self, force_render=False):
        if self.show_original:
            display_data = self.img_full if self.comp_proxy.shape == self.img_full.shape else self.img_proxy
            self.lbl_blink.show()
        else:
            display_data = self.comp_proxy
            self.lbl_blink.hide()
            
        if display_data is None: return
        
        disp = np.clip(display_data * 255, 0, 255).astype(np.uint8)
        disp = np.flipud(disp)
        
        # FIX: Handle single channel output (Mask View)
        if disp.ndim == 2:
            disp = np.stack([disp]*3, axis=-1)
        
        h, w, c = disp.shape
        if not disp.flags['C_CONTIGUOUS']: disp = np.ascontiguousarray(disp)
        
        self._current_display_buffer = disp
        qimg = QImage(self._current_display_buffer.data, w, h, c * w, QImage.Format.Format_RGB888)
        new_pixmap = QPixmap.fromImage(qimg)

        full_h, full_w = self.img_full.shape[:2]
        scale_x = full_w / w
        
        self.pix_item.setPixmap(new_pixmap)
        self.pix_item.setTransformOriginPoint(0, 0)
        self.pix_item.setScale(scale_x)
        self.lbl_blink.move((self.view.width() - self.lbl_blink.width()) // 2, 10)

    def apply_process(self):
        if self.img_full is None: return
        self.setEnabled(False)
        self.pbar.show()
        
        params = self.get_current_params()
        # Ensure mask is FALSE for final apply even if checkbox is checked
        params['return_mask'] = False
        
        self.apply_worker = ProcessingWorker(self.img_full, params, 0)
        self.apply_worker.result_ready.connect(self._on_apply_finished)
        self.apply_worker.finished.connect(self.apply_worker.deleteLater)
        self.apply_worker.start()

    def _on_apply_finished(self, res, job_id):
        try:
            if res is not None:
                out = res.transpose(2, 0, 1) if res.ndim == 3 else res[np.newaxis, ...]
                with self.siril.image_lock():
                    self.siril.undo_save_state("VeraLux Revela")
                    self.siril.set_image_pixeldata(out.astype(np.float32))
                self.siril.log("VeraLux Revela applied successfully.", LogColor.GREEN)
                self.close()
            else:
                raise Exception("Processing failed in the final apply step.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            traceback.print_exc()
        finally:
            self.pbar.hide()
            self.setEnabled(True)

    def set_defaults(self):
        self.s_tex.setValue(0); self.s_str.setValue(0); self.s_shad.setValue(33)
        self.chk_star.setChecked(True); self.chk_show_mask.setChecked(False)
        self.trigger_full_res_update()

    def showEvent(self, event):
        QTimer.singleShot(0, self.fit_view); super().showEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self.show_original = True; self.update_view(); event.accept()
        else: super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self.show_original = False; self.update_view(); event.accept()
        else: super().keyReleaseEvent(event)

    def resizeEvent(self, event):
        self.lbl_blink.move((self.view.width() - self.lbl_blink.width()) // 2, 10); super().resizeEvent(event)

    def toggle_ontop(self, checked):
        if checked: self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        else: self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowStaysOnTopHint)
        self.show()

    def zoom_in(self): self.is_fit_mode = False; self.view.scale(1.2, 1.2)
    def zoom_out(self): self.is_fit_mode = False; self.view.scale(1/1.2, 1/1.2)
    def zoom_1to1(self): self.is_fit_mode = False; self.view.resetTransform()
    def fit_view(self):
        self.is_fit_mode = True
        if self.pix_item.pixmap(): self.view.fitInView(self.pix_item, Qt.AspectRatioMode.KeepAspectRatio)
            
    def eventFilter(self, source, event):
        if source == self.view.viewport():
            if event.type() == QEvent.Type.Wheel:
                self.is_fit_mode = False
                if event.angleDelta().y() > 0: self.zoom_in()
                else: self.zoom_out()
                return True
            if event.type() == QEvent.Type.MouseButtonDblClick:
                self.fit_view(); return True
        return super().eventFilter(source, event)

    def print_help(self):
        msg = [
            f"==========================================================================",
            f"   VERALUX REVELA v{VERSION} — OPERATIONAL GUIDE",
            "   Photometric Local Contrast & Texture Engine",
            "==========================================================================","",
            "OVERVIEW","-----------------",
            "VeraLux Revela is a post-stretch enhancement engine designed to reveal",
            "micro-contrast and volumetric structures in nebulae and galaxies.",
            "It uses a Signal-Aware approach to modulate enhancement across noise, stars,",
            "and real nebular structure.","","[1] SLIDERS",
            "    • Texture (Micro-Contrast):",
            "      Enhances fine details (shock fronts, dust lanes) using high-frequency scales.",
            "    • Structure (Volume):",
            "      Enhances the 3D volume and body of the object using medium-frequency scales.","","[2] PROTECTION (The Gate)",
            "    • Shadow Authority (Noise Gate):",
            "      A photometric gate that prevents contrast enhancement from amplifying",
            "      background noise. If the background looks gritty, increase this.",
            "    • Show Mask:",
            "      Check to visualize the protection gate. White=Active, Black=Protected.",
            "      Adjust 'Shadow Authority' until the background becomes black.",
            "    • Isolate Stars:",
            "      Automatically detects high-frequency stellar peaks to prevent",
            "      'raccoon eyes' (dark halos) around stars.","","[3] PREVIEW CONTROLS",
            "    • Navigation: Zoom (+/-), 1:1, Fit.",
            "    • Compare: Hold SPACEBAR to blink the original image.",
            "    • Double-click the image to Fit to View.","",
            "Support & Info: info@veralux.space",
            "=========================================================================="
        ]
        try: [self.siril.log(l if l.strip() else " ") for l in msg]
        except: print("\n".join(msg))

def _nofocus(w):
    try:
        w.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    except Exception:
        pass

def main():
    app = QApplication.instance(); app = app or QApplication(sys.argv)
    siril = s.SirilInterface()
    try: siril.connect()
    except: pass
    gui = StructureGUI(siril, app); gui.show()
    app.exec()

if __name__ == "__main__":
    main()