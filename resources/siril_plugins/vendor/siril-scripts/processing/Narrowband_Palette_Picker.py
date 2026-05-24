"""
Narrowband palette picker V2 for Siril
Helps preview different palettes for narrowband image composition.

New in V2:
Basically just making it presentable.

V2.0.1:
- Added contact details as per Git requirements
- Added version ID

V2.1
- Made changes to script to comply with MR requirements
- Added handling of compressed fits files
- UI changes to address the height issue
- Added scaling system (Which is memorized) so users with less resolution can still use the tool)
- Addressed that the saved result only comes out at Ha
- Added Synthetic Luminance feature
- Added purple star removal feature (Invert + SCNR + Invert)
- Fixed bug where unlinked stretch wasn't occuring correctly.
- Added a usage note to specify that saves are set to whatever the siril home path is
- Updated to use new push methodology instead of RGB comp with temp files

V2.1.1
- Fixed bug where would error out without image already loaded.
- Fixed bug with Synthetic luminance not un-stretching.

V2.2 
- Various Bug fixes
- Linked stretch preview button not working, its always unlinked.
- Get rid of "(Creats_result.fit)" text in the main window, it's unnecessary.
- Not remembering last used unstretch agression value used.
- Adjust blend strength seems to do nothing anymore - Re-design of implementation
- Remove purple stars warning
    
V3.0
- Implement Custom palette tab.

V3.1
- Live preview window (non-modal, sliders update in real time)
- Channel balance sliders on Tab 1 with remembered values
- Auto balance using geometric mean of signal medians
- Clear balance resets sliders and temp files
- Fixed preview/balance buttons not enabling on startup from saved settings
- Fixed Tab 2/3 palette previews not respecting balance
- Fixed OIII combine mode not reflected in live preview

V3.2
- Revised auto balance algorithm: estimates and subtracts per-channel sky background
  before measuring signal medians, so balanced channels produce identical results
  under both linked and unlinked statistical stretch.
- Background estimated using the mode of a low-percentile sample (robust to large nebulae)
  with a sigma-clipped fallback.

V3.3
- Added multi-channel crop tool with zoom/pan controls
- Added zoom controls to live preview window
- Synthetic luminance now visible in live preview with proper LRGB blending
- Crop tool implementation based on SetiAstroSuite by Seti Astro
  (https://github.com/setiastro/setiastrosuite)

V3.4
- Fixed crop tool data corruption (was causing uint16 overflow)
- Changed auto balance to match brightest channel instead of geometric mean
  (never reduces signal, only boosts dimmer channels)
  
V3.5
- Bugfix with cropping tool destorying signal due to 16-32 conversions.


"""


import sys
import os

# Use sirilpy for dependency management (required for Siril scripts)
import sirilpy as s

# Ensure all dependencies are installed
s.ensure_installed("numpy", version_constraints="<2.4.0")  # Constraint for numba compatibility
s.ensure_installed("PyQt6", "astropy", "numba", "scipy", "Pillow")

# Check for optional astroalign
try:
    import astroalign
    ASTROALIGN_AVAILABLE = True
except ImportError:
    ASTROALIGN_AVAILABLE = False
    print("Note: astroalign not available - alignment checks will be skipped")

import time
import traceback
import json
from pathlib import Path
import numpy as np
import sirilpy as s
from PIL import Image
from scipy import ndimage
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, 
                             QLabel, QComboBox, QMessageBox, QCheckBox, 
                             QFileDialog, QHBoxLayout, QLineEdit, QSlider,
                             QGridLayout, QGroupBox, QScrollArea, QSizePolicy,
                             QDialog, QTabWidget, QGraphicsView, QGraphicsScene,
                             QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsPixmapItem,
                             QGraphicsItem)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QRectF, QPointF
from PyQt6.QtGui import QImage, QPixmap, QPainter, QFont, QPen, QColor, QIcon, QBrush, QCursor
from astropy.io import fits
from numba import njit, prange
import cv2
import math

# Configuration file location
CONFIG_FILE = Path.home() / ".siril_palette_settings.json"

# ============================================================================
# PPP STRETCHING SYSTEM - Numba-optimized functions
# ============================================================================

@njit(parallel=True, fastmath=True)
def numba_mono_final_formula(rescaled, median_rescaled, target_median):
    """
    Applies the final PPP formula for mono stretch.
    rescaled: already (image - black_point) / (1 - black_point)
    """
    H, W = rescaled.shape
    out = np.empty_like(rescaled)

    for y in prange(H):
        for x in range(W):
            r = rescaled[y, x]
            numer = (median_rescaled - 1.0) * target_median * r
            denom = median_rescaled * (target_median + r - 1.0) - target_median * r
            if np.abs(denom) < 1e-12:
                denom = 1e-12
            out[y, x] = numer / denom

    return out

@njit(parallel=True, fastmath=True)
def numba_color_final_formula_linked(rescaled, median_rescaled, target_median):
    """
    Linked color transform: one median for all channels.
    """
    H, W, C = rescaled.shape
    out = np.empty_like(rescaled)

    for y in prange(H):
        for x in range(W):
            for c in range(C):
                r = rescaled[y, x, c]
                numer = (median_rescaled - 1.0) * target_median * r
                denom = median_rescaled * (target_median + r - 1.0) - target_median * r
                if np.abs(denom) < 1e-12:
                    denom = 1e-12
                out[y, x, c] = numer / denom

    return out

@njit(parallel=True, fastmath=True)
def numba_color_final_formula_unlinked(rescaled, medians_rescaled, target_median):
    """
    Unlinked color transform: separate median per channel.
    """
    H, W, C = rescaled.shape
    out = np.empty_like(rescaled)

    for y in prange(H):
        for x in range(W):
            for c in range(C):
                r = rescaled[y, x, c]
                med = medians_rescaled[c]
                numer = (med - 1.0) * target_median * r
                denom = med * (target_median + r - 1.0) - target_median * r
                if np.abs(denom) < 1e-12:
                    denom = 1e-12
                out[y, x, c] = numer / denom

    return out

@njit
def piecewise_linear(val, xvals, yvals):
    """Piecewise linear interpolation for curves adjustment"""
    if val <= xvals[0]:
        return yvals[0]
    for i in range(len(xvals)-1):
        if val < xvals[i+1]:
            dx = xvals[i+1] - xvals[i]
            dy = yvals[i+1] - yvals[i]
            ratio = (val - xvals[i]) / dx
            return yvals[i] + ratio * dy
    return yvals[-1]

@njit(parallel=True, fastmath=True)
def apply_curves_numba(image, xvals, yvals):
    """Apply curves adjustment to image"""
    if image.ndim == 2:
        H, W = image.shape
        out = np.empty((H, W), dtype=np.float32)
        for y in prange(H):
            for x in range(W):
                val = image[y, x]
                out[y, x] = piecewise_linear(val, xvals, yvals)
        return out
    elif image.ndim == 3:
        H, W, C = image.shape
        out = np.empty((H, W, C), dtype=np.float32)
        for y in prange(H):
            for x in range(W):
                for c in range(C):
                    val = image[y, x, c]
                    out[y, x, c] = piecewise_linear(val, xvals, yvals)
        return out
    else:
        return image

@njit(parallel=True, fastmath=True)
def bin2x2_numba(image):
    """Fast 2x2 binning for preview generation"""
    h, w = image.shape[:2]
    h2 = h // 2
    w2 = w // 2

    if image.ndim == 2:
        out = np.empty((h2, w2), dtype=np.float32)
        for i in prange(h2):
            for j in prange(w2):
                s = image[2*i, 2*j] + image[2*i+1, 2*j] + \
                    image[2*i, 2*j+1] + image[2*i+1, 2*j+1]
                out[i, j] = s * 0.25
    else:
        c = image.shape[2]
        out = np.empty((h2, w2, c), dtype=np.float32)
        for i in prange(h2):
            for j in prange(w2):
                for k in range(c):
                    s = image[2*i, 2*j, k] + image[2*i+1, 2*j, k] + \
                        image[2*i, 2*j+1, k] + image[2*i+1, 2*j+1, k]
                    out[i, j, k] = s * 0.25
    return out

# ============================================================================
# PPP STRETCHING FUNCTIONS - ported from Statistical_Stretch reference script
# ============================================================================

def _sample_flat(x, max_n=400_000):
    flat = np.asarray(x, np.float32).reshape(-1)
    n = flat.size
    if n <= max_n:
        return flat
    stride = max(1, n // max_n)
    return flat[::stride]

def _robust_sigma_lower_half_fast(x, max_n=400_000):
    """MAD-based robust sigma estimate using lower half only"""
    s = _sample_flat(x, max_n=max_n)
    med = float(np.median(s))
    lo = s[s <= med]
    if lo.size < 16:
        mad = float(np.median(np.abs(s - med)))
    else:
        med_lo = float(np.median(lo))
        mad = float(np.median(np.abs(lo - med_lo)))
    return 1.4826 * mad

def _compute_blackpoint_sigma(img, sigma=5.0):
    """Compute black point using robust MAD-based sigma"""
    img = np.asarray(img, dtype=np.float32)
    med = float(np.median(img))
    noise = _robust_sigma_lower_half_fast(img)
    bp = med - float(sigma) * noise
    mn = float(img.min())
    bp = max(mn, bp)
    bp = min(bp, 0.99)
    return float(bp), med

def unstretch_image(image, target_median=0.02):
    """
    PPP's inverse stretch - returns image to approximately linear state.
    Works on RGB images where each channel was independently stretched.
    """
    was_single_channel = False
    
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        was_single_channel = True
        image = np.stack([image.squeeze()] * 3, axis=-1)
    
    image = image.astype(np.float32).copy()
    unstretched_image = image.copy()
    
    for c in range(3):
        channel = unstretched_image[..., c]
        channel_median = np.median(channel)
        if channel_median > 0:
            numerator = (channel_median - 1) * target_median * channel
            denominator = channel_median * (target_median + channel - 1) - target_median * channel
            denominator = np.where(np.abs(denominator) < 1e-6, 1e-6, denominator)
            unstretched_image[..., c] = numerator / denominator
    
    unstretched_image = np.clip(unstretched_image, 0.0, 1.0)
    
    if was_single_channel:
        unstretched_image = np.mean(unstretched_image, axis=2, keepdims=True)
    
    return unstretched_image

def stretch_mono_image(image, target_median, normalize=False, apply_curves=False,
                       curves_boost=0.0, blackpoint_sigma=5.0):
    """Mono stretch - exact port from reference StatisticalStretch script"""
    target_median = max(0.01, min(0.99, target_median))
    image = np.asarray(image, dtype=np.float32)

    black_point, _ = _compute_blackpoint_sigma(image, blackpoint_sigma)

    denom = max(1.0 - black_point, 1e-12)
    rescaled = (image - black_point) / denom
    median_rescaled = float(np.median(rescaled))

    stretched = numba_mono_final_formula(rescaled, median_rescaled, target_median)

    if apply_curves:
        stretched = apply_curves_adjustment(stretched, target_median, curves_boost)

    if normalize:
        mx = stretched.max()
        if mx > 0:
            stretched = stretched / mx

    return np.clip(stretched, 0.0, 1.0).astype(np.float32)

def stretch_color_image_linked(image, target_median, normalize=False, apply_curves=False,
                                curves_boost=0.0, blackpoint_sigma=5.0):
    """Linked color stretch - exact port from reference StatisticalStretch script"""
    target_median = max(0.01, min(0.99, target_median))
    image = np.asarray(image, dtype=np.float32)

    black_point, _ = _compute_blackpoint_sigma(image, blackpoint_sigma)

    rescaled = (image - black_point) / (1.0 - black_point)
    median_rescaled = float(np.median(rescaled))

    stretched = numba_color_final_formula_linked(rescaled, median_rescaled, target_median)

    if apply_curves:
        stretched = apply_curves_adjustment(stretched, target_median, curves_boost)

    if normalize:
        mx = stretched.max()
        if mx > 0:
            stretched = stretched / mx

    return np.clip(stretched, 0.0, 1.0).astype(np.float32)

def stretch_color_image_unlinked(image, target_median, normalize=False, apply_curves=False,
                                  curves_boost=0.0, blackpoint_sigma=5.0):
    """Unlinked color stretch - exact port from reference StatisticalStretch script"""
    target_median = max(0.01, min(0.99, target_median))
    image = np.asarray(image, dtype=np.float32)

    stretched_image = np.zeros_like(image, dtype=np.float32)

    for c in range(3):
        channel = image[..., c]
        black_point, _ = _compute_blackpoint_sigma(channel, blackpoint_sigma)
        rescaled = (channel - black_point) / (1.0 - black_point)
        median_rescaled = float(np.median(rescaled))

        num = (median_rescaled - 1.0) * target_median * rescaled
        den = median_rescaled * (target_median + rescaled - 1.0) - target_median * rescaled
        den = np.where(np.abs(den) < 1e-12, 1e-12, den)
        stretched_image[..., c] = num / den

    if apply_curves:
        stretched_image = apply_curves_adjustment(stretched_image, target_median, curves_boost)

    if normalize:
        mx = stretched_image.max()
        if mx > 0:
            stretched_image = stretched_image / mx

    return np.clip(stretched_image, 0.0, 1.0).astype(np.float32)

def apply_curves_adjustment(image, target_median, curves_boost):
    """Apply PPP curves adjustment"""
    if curves_boost <= 0.0:
        return np.clip(image, 0.0, 1.0).astype(np.float32)
    
    tm = float(target_median)
    cb = float(curves_boost)
    p3x = 0.25 * (1.0 - tm) + tm
    p4x = 0.75 * (1.0 - tm) + tm
    p3y = p3x ** (1.0 - cb)
    p4y = (p4x ** (1.0 - cb)) ** (1.0 - cb)
    
    xvals = np.array([0.0, 0.5 * tm, tm, p3x, p4x, 1.0], dtype=np.float32)
    yvals = np.array([0.0, 0.5 * tm, tm, p3y, p4y, 1.0], dtype=np.float32)
    
    image_32 = np.clip(image.astype(np.float32, copy=False), 0.0, 1.0)
    adjusted = apply_curves_numba(image_32, xvals, yvals)
    return np.clip(adjusted, 0.0, 1.0).astype(np.float32)

# ============================================================================
# PREVIEW GENERATION THREAD
# ============================================================================

# ============================================================================
# BLEND STRENGTH DIALOG
# ============================================================================

class BlendStrengthDialog(QDialog):
    """Dialog for adjusting channel blend strength"""
    def __init__(self, ha_value=100, oiii_value=100, sii_value=100, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Channel Blend Strength Adjustment')
        self.setModal(True)
        self.resize(450, 280)
        
        layout = QVBoxLayout()
        
        # Instructions
        instructions = QLabel(
            "<b>Adjust Channel Blend Strength</b><br>"
            "<i>Applies to unstretched/linear output</i>"
        )
        instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        instructions.setStyleSheet("padding: 10px; font-size: 11pt;")
        layout.addWidget(instructions)
        
        # Ha balance
        ha_layout = QHBoxLayout()
        ha_layout.addWidget(QLabel("Ha Strength:"))
        self.ha_slider = QSlider(Qt.Orientation.Horizontal)
        self.ha_slider.setMinimum(50)
        self.ha_slider.setMaximum(150)
        self.ha_slider.setValue(ha_value)
        self.ha_label = QLabel(f"{ha_value/100:.2f}x")
        self.ha_label.setMinimumWidth(50)
        self.ha_slider.valueChanged.connect(lambda v: self.ha_label.setText(f"{v/100:.2f}x"))
        ha_layout.addWidget(self.ha_slider)
        ha_layout.addWidget(self.ha_label)
        layout.addLayout(ha_layout)
        
        # OIII balance
        oiii_layout = QHBoxLayout()
        oiii_layout.addWidget(QLabel("OIII Strength:"))
        self.oiii_slider = QSlider(Qt.Orientation.Horizontal)
        self.oiii_slider.setMinimum(50)
        self.oiii_slider.setMaximum(150)
        self.oiii_slider.setValue(oiii_value)
        self.oiii_label = QLabel(f"{oiii_value/100:.2f}x")
        self.oiii_label.setMinimumWidth(50)
        self.oiii_slider.valueChanged.connect(lambda v: self.oiii_label.setText(f"{v/100:.2f}x"))
        oiii_layout.addWidget(self.oiii_slider)
        oiii_layout.addWidget(self.oiii_label)
        layout.addLayout(oiii_layout)
        
        # SII balance
        sii_layout = QHBoxLayout()
        sii_layout.addWidget(QLabel("SII Strength:"))
        self.sii_slider = QSlider(Qt.Orientation.Horizontal)
        self.sii_slider.setMinimum(50)
        self.sii_slider.setMaximum(150)
        self.sii_slider.setValue(sii_value)
        self.sii_label = QLabel(f"{sii_value/100:.2f}x")
        self.sii_label.setMinimumWidth(50)
        self.sii_slider.valueChanged.connect(lambda v: self.sii_label.setText(f"{v/100:.2f}x"))
        sii_layout.addWidget(self.sii_slider)
        sii_layout.addWidget(self.sii_label)
        layout.addLayout(sii_layout)
        
        layout.addSpacing(15)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        # Restore defaults button
        restore_btn = QPushButton("Restore Defaults")
        restore_btn.setStyleSheet("padding: 8px 20px; background-color: #95a5a6; color: white;")
        restore_btn.clicked.connect(self.restore_defaults)
        button_layout.addWidget(restore_btn)
        
        button_layout.addStretch()
        
        ok_btn = QPushButton("OK")
        ok_btn.setStyleSheet("padding: 8px 20px;")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet("padding: 8px 20px;")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def restore_defaults(self):
        """Reset all sliders to 100"""
        self.ha_slider.setValue(100)
        self.oiii_slider.setValue(100)
        self.sii_slider.setValue(100)


# ============================================================================
# PREVIEW GENERATION THREAD
# ============================================================================

class PreviewGeneratorThread(QThread):
    """Background thread for generating palette previews"""
    preview_ready = pyqtSignal(str, QPixmap)
    all_complete = pyqtSignal()
    
    def __init__(self, ha_data, oiii_data, sii_data, palette_processor, 
                 preview_size=300, 
                 stretch_mode=0, stretch_strength=0.25, apply_curves=False,
                 preview_linked=True, preview_curves=False):
        super().__init__()
        self.ha_data = ha_data.copy()
        self.oiii_data = oiii_data.copy()
        self.sii_data = sii_data.copy()
        self.palette_processor = palette_processor
        self.preview_size = preview_size
        
        
        # Stretch settings for CHANNEL stretching (before palette)
        self.stretch_mode = stretch_mode
        self.stretch_strength = stretch_strength
        self.apply_curves = apply_curves
        
        # Preview display settings (only used if no channel stretch)
        self.preview_linked = preview_linked
        self.preview_curves = preview_curves
        
    def run(self):
        """Generate previews for all palettes"""
        palette_names = [
            "Foraxx", "Realistic1", "Realistic2",
            "HOO (Narrowband Ha-OIII)", "HOS", "HSO", "HSS",
            "OHH", "OHS", "OSH", "OSS",
            "SHH", "SHO (Hubble)", "SOH", "SOO", "SSH"
        ]
        
        #print(f"Preview thread started - generating {len(palette_names)} previews...")
        #print(f"  Channel stretch mode: {['None', 'Linked', 'Unlinked'][self.stretch_mode]}")
        #print(f"  Stretch strength: {self.stretch_strength}")
        #print(f"  Curves adjustment: {'Enabled' if self.apply_curves else 'Disabled'}")
        
        # Apply stretch to channels BEFORE palette processing (same as final output!)
        ha_stretched = self.ha_data
        oiii_stretched = self.oiii_data
        sii_stretched = self.sii_data
        
        if self.stretch_mode == 1:  # Linked
            #print("  Applying PPP Linked stretch to channels before palette...")
            rgb_combined = np.stack([self.ha_data, self.oiii_data, self.sii_data], axis=2)
            rgb_result = stretch_color_image_linked(rgb_combined, self.stretch_strength, 
                                                    apply_curves=self.apply_curves, 
                                                    curves_boost=0.15)
            ha_stretched = rgb_result[:, :, 0]
            oiii_stretched = rgb_result[:, :, 1]
            sii_stretched = rgb_result[:, :, 2]
        elif self.stretch_mode == 2:  # Unlinked
            #print("  Applying PPP Unlinked stretch to channels before palette...")
            ha_stretched = stretch_mono_image(self.ha_data, self.stretch_strength, 
                                             apply_curves=self.apply_curves, curves_boost=0.15)
            oiii_stretched = stretch_mono_image(self.oiii_data, self.stretch_strength,
                                               apply_curves=self.apply_curves, curves_boost=0.15)
            sii_stretched = stretch_mono_image(self.sii_data, self.stretch_strength,
                                              apply_curves=self.apply_curves, curves_boost=0.15)
        
        for palette_name in palette_names:
            try:
                #print(f"  Generating preview for: {palette_name}")
                
                # Use stretched data for all palettes when stretch is enabled
                if self.stretch_mode > 0:
                    use_ha = ha_stretched
                    use_oiii = oiii_stretched
                    use_sii = sii_stretched
                else:
                    use_ha = self.ha_data
                    use_oiii = self.oiii_data
                    use_sii = self.sii_data
                
                # Generate palette
                r, g, b = self.palette_processor.process_palette(
                    use_ha.copy(), 
                    use_oiii.copy(), 
                    use_sii.copy(), 
                    palette_name
                )
                
                if r is None or g is None or b is None:
                    #print(f"    Skipping {palette_name} - process_palette returned None")
                    continue
                
                # Downsample using fast 2x2 binning
                rgb = np.stack([r, g, b], axis=2).astype(np.float32)
                
                # Bin down to preview size
                bin_count = 0
                while rgb.shape[0] > self.preview_size * 2 or rgb.shape[1] > self.preview_size * 2:
                    rgb = bin2x2_numba(rgb)
                    bin_count += 1
                
                # Final resize using scipy zoom - works on float32 without data loss
                if rgb.shape[0] > self.preview_size or rgb.shape[1] > self.preview_size:
                    scale = min(self.preview_size / rgb.shape[0], self.preview_size / rgb.shape[1])
                    
                    # Use scipy zoom which works on float32 arrays directly
                    rgb = ndimage.zoom(rgb, (scale, scale, 1), order=1)  # order=1 = bilinear
                
                # ALWAYS apply preview stretch for consistent visibility
                # (Preview settings are independent of channel stretch)
                if self.preview_linked:
                    rgb = stretch_color_image_linked(
                        rgb, target_median=0.25, 
                        apply_curves=self.preview_curves, 
                        curves_boost=0.15
                    )
                else:
                    rgb = stretch_color_image_unlinked(
                        rgb, target_median=0.25, 
                        apply_curves=self.preview_curves, 
                        curves_boost=0.15
                    )
                
                # Convert to 8-bit
                rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                
                # Convert to QPixmap
                height, width = rgb_uint8.shape[:2]
                bytes_per_line = 3 * width
                
                
                rgb_copy = np.ascontiguousarray(rgb_uint8)
                q_image = QImage(rgb_copy.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image.copy())
                
                
                # Add text label
                painter = QPainter(pixmap)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                painter.fillRect(0, 0, width, 25, QColor(0, 0, 0, 180))
                painter.setPen(QPen(QColor(255, 255, 255)))
                painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
                painter.drawText(5, 18, palette_name)
                painter.end()
                
                # Emit preview
                #print(f"    Emitting preview for {palette_name}")
                self.preview_ready.emit(palette_name, pixmap)
                
            except Exception as e:
                print(f"ERROR generating preview for {palette_name}: {e}")
                traceback.print_exc()
                continue
        
        #print("All previews generated - emitting completion signal")
        self.all_complete.emit()

# ============================================================================
# PREVIEW WINDOW
# ============================================================================

class PreviewWindow(QDialog):
    """Modal dialog displaying palette previews in a grid"""
    palette_selected = pyqtSignal(str)
    
    def __init__(self, parent=None, scale=100, preview_size=300):
        super().__init__(parent)
        self.setWindowTitle('Palette Previews - Click to Select')
        self.setModal(True)
        self.scale = scale / 100.0
        self.preview_size = preview_size
        
        # Calculate exact window size needed
        # Button size = preview image + 10px border
        button_size = preview_size + 10
        
        # Spacing between buttons (scaled, minimum 3px, max 5px)
        spacing = max(3, min(5, int(5 * self.scale)))
        
        # Grid dimensions: 4 columns x 4 rows
        # Width = (4 buttons) + (3 gaps) + (margins)
        content_width = (button_size * 4) + (spacing * 3)
        
        # Height = (4 rows) + (3 gaps) + header(~40px) + footer(~30px)  
        content_height = (button_size * 4) + (spacing * 3) + 70
        
        # Add reasonable margins
        window_width = content_width + 50
        window_height = content_height + 50
        
        self.resize(window_width, window_height)
        self.setMinimumSize(window_width, window_height)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 10, 20, 10)
        layout.setSpacing(5)
        
        # Instructions
        #instructions = QLabel("Click any palette to select it")
        #instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        #instructions.setStyleSheet("font-size: 11px; padding: 3px;")
        #layout.addWidget(instructions)
        
        # "Please wait" notice
        self.wait_label = QLabel(
            "<span style='color: #e74c3c; font-size: 14pt; font-weight: bold;'>"
            "Generating previews...</span>"
        )
        self.wait_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.wait_label)
        
        # Grid container (let it center naturally)
        grid_widget = QWidget()
        self.grid_layout = QGridLayout(grid_widget)
        self.grid_layout.setSpacing(spacing)  # This sets both horizontal and vertical
        self.grid_layout.setHorizontalSpacing(spacing)
        self.grid_layout.setVerticalSpacing(spacing)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(grid_widget)
        
        # Status label
        #self.status_label = QLabel("Generating previews...")
        #self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        #self.status_label.setStyleSheet("font-size: 11px; padding: 3px;")  # Reduced padding
        #layout.addWidget(self.status_label)
        
        # Close button
        #close_btn = QPushButton("Close Without Selecting")
        #close_btn.setStyleSheet("background-color: #95a5a6; color: white; font-weight: bold; height: 40px;")
        #close_btn.clicked.connect(self.reject)
        #layout.addWidget(close_btn)
        
        self.setLayout(layout)
        self.palette_buttons = {}
    
    def add_preview(self, palette_name, pixmap):
        """Add a preview thumbnail to the grid"""
        
        # Force pixmap to device pixel ratio of 1.0 to avoid DPI scaling
        if pixmap.devicePixelRatio() != 1.0:
            print(f"  WARNING: Pixmap has devicePixelRatio={pixmap.devicePixelRatio()}, setting to 1.0")
            pixmap.setDevicePixelRatio(1.0)
        
        # Create a fixed-size container
        container = QWidget()
        container.setFixedSize(pixmap.width() + 6, pixmap.height() + 6)
        container.setCursor(Qt.CursorShape.PointingHandCursor)
        container.setToolTip(f"Click to select {palette_name}")
        
        # QLabel with ABSOLUTE positioning - no layout!
        label = QLabel(container)
        label.setPixmap(pixmap)
        label.setScaledContents(False)
        label.setGeometry(3, 3, pixmap.width(), pixmap.height())  # Absolute position
        
        
        # Style the container
        container.setStyleSheet(
            "QWidget { "
            "  background-color: transparent; "
            "  border: 0px solid transparent; "
            "}"
            "QWidget:hover { "
            "  border: 3px solid #3498db; "
            "}"
        )
        
        # Make it clickable
        container.mousePressEvent = lambda event: self.on_palette_clicked(palette_name)
        
        # Add to grid (4 columns)
        index = len(self.palette_buttons)
        row = index // 4
        col = index % 4
        self.grid_layout.addWidget(container, row, col, Qt.AlignmentFlag.AlignCenter)
        
        self.palette_buttons[palette_name] = container
        #self.status_label.setText(f"Generated {len(self.palette_buttons)} of 16 previews...")
    
    def on_palette_clicked(self, palette_name):
        """Handle palette selection"""
        self.palette_selected.emit(palette_name)
        self.accept()
    
    def generation_complete(self):
        """Called when all previews are generated"""
        self.wait_label.hide()  # Hide the "Please wait" message
        #self.status_label.setText("✓ All previews ready! Click any palette to select.")
        #self.status_label.setStyleSheet("font-size: 12px; padding: 5px; color: #27ae60; font-weight: bold;")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

# Crop tool components for NPP

import numpy as np
import cv2
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                              QPushButton, QComboBox, QGraphicsView, QGraphicsScene,
                              QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsPixmapItem,
                              QMessageBox, QTabWidget, QWidget, QGraphicsItem)
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal
from PyQt6.QtGui import QPen, QBrush, QPixmap, QImage, QCursor
import math
from astropy.io import fits
import os

HANDLE_SIZE = 12

class ResizableRectItem(QGraphicsRectItem):
    """Resizable and rotatable rectangle with corner handles"""
    
    def __init__(self, rect: QRectF, parent=None):
        super().__init__(rect, parent)
        pen = QPen(Qt.GlobalColor.green, 2)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        
        self._fixed_aspect_ratio = None
        self._handles = {}
        self._active_handle = None
        self._rotating = False
        self._rotation_start = 0.0
        self._pivot_scene = QPointF()
        
        self._initHandles()
        self.setTransformOriginPoint(self.rect().center())
    
    def setFixedAspectRatio(self, ratio):
        """Lock to aspect ratio (width/height) or None for free"""
        self._fixed_aspect_ratio = ratio
    
    def _initHandles(self):
        """Create corner handles"""
        pen = QPen(Qt.GlobalColor.green, 2)
        pen.setCosmetic(True)
        brush = QBrush(Qt.GlobalColor.white)
        
        for pos in ["tl", "tr", "br", "bl"]:
            h = QGraphicsEllipseItem(0, 0, HANDLE_SIZE, HANDLE_SIZE, self)
            h.setPen(pen)
            h.setBrush(brush)
            h.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
            self._handles[pos] = h
        
        self._updateHandlePositions()
    
    def _updateHandlePositions(self):
        """Update handle positions to match rectangle corners"""
        r = self.rect()
        s = HANDLE_SIZE
        corners = {
            "tl": QPointF(r.left() - s/2, r.top() - s/2),
            "tr": QPointF(r.right() - s/2, r.top() - s/2),
            "br": QPointF(r.right() - s/2, r.bottom() - s/2),
            "bl": QPointF(r.left() - s/2, r.bottom() - s/2),
        }
        for pos, item in self._handles.items():
            item.setPos(corners[pos])
    
    def hoverMoveEvent(self, ev):
        """Change cursor based on handle hover"""
        for pos, h in self._handles.items():
            if h.contains(h.mapFromScene(ev.scenePos())):
                self._setCursorForHandle(pos)
                return
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        super().hoverMoveEvent(ev)
    
    def _setCursorForHandle(self, pos):
        """Set cursor shape for handle"""
        cursors = {
            "tl": Qt.CursorShape.SizeFDiagCursor,
            "tr": Qt.CursorShape.SizeBDiagCursor,
            "br": Qt.CursorShape.SizeFDiagCursor,
            "bl": Qt.CursorShape.SizeBDiagCursor,
        }
        self.setCursor(cursors.get(pos, Qt.CursorShape.SizeAllCursor))
    
    def mousePressEvent(self, ev):
        """Handle mouse press for rotation or resize"""
        if ev.modifiers() == Qt.KeyboardModifier.ShiftModifier:
            # Start rotation
            self._rotating = True
            pivot = self.mapToScene(self.rect().center())
            self._pivot_scene = pivot
            v0 = ev.scenePos() - pivot
            self._angle_ref = math.degrees(math.atan2(v0.y(), v0.x()))
            self._rotation_start = self.rotation()
            ev.accept()
            return
        
        # Check for handle resize
        for pos, h in self._handles.items():
            if h.contains(h.mapFromScene(ev.scenePos())):
                self._active_handle = pos
                ev.accept()
                return
        
        super().mousePressEvent(ev)
    
    def mouseMoveEvent(self, ev):
        """Handle mouse move for rotation or resize"""
        if self._rotating:
            # Rotate around center
            pivot = self._pivot_scene
            v = ev.scenePos() - pivot
            angle_now = math.degrees(math.atan2(v.y(), v.x()))
            delta = angle_now - self._angle_ref
            self.setRotation(self._rotation_start + delta)
            ev.accept()
            return
        
        if self._active_handle:
            # Resize from handle
            scene_pos = ev.scenePos()
            local_pos = self.mapFromScene(scene_pos)
            self._resizeFromHandle(self._active_handle, local_pos)
            ev.accept()
            return
        
        super().mouseMoveEvent(ev)
    
    def mouseReleaseEvent(self, ev):
        """End rotation or resize"""
        self._rotating = False
        self._active_handle = None
        super().mouseReleaseEvent(ev)
    
    def _resizeFromHandle(self, handle_pos, local_pos):
        """Resize rectangle from handle position"""
        r = self.rect()
        
        if handle_pos == "br":
            new_w = max(20, local_pos.x() - r.left())
            new_h = max(20, local_pos.y() - r.top())
            
            if self._fixed_aspect_ratio:
                new_h = new_w / self._fixed_aspect_ratio
            
            r.setWidth(new_w)
            r.setHeight(new_h)
        
        elif handle_pos == "tl":
            new_left = min(local_pos.x(), r.right() - 20)
            new_top = min(local_pos.y(), r.bottom() - 20)
            
            new_w = r.right() - new_left
            new_h = r.bottom() - new_top
            
            if self._fixed_aspect_ratio:
                new_h = new_w / self._fixed_aspect_ratio
                new_top = r.bottom() - new_h
            
            r.setLeft(new_left)
            r.setTop(new_top)
        
        elif handle_pos == "tr":
            new_w = max(20, local_pos.x() - r.left())
            new_top = min(local_pos.y(), r.bottom() - 20)
            new_h = r.bottom() - new_top
            
            if self._fixed_aspect_ratio:
                new_h = new_w / self._fixed_aspect_ratio
                new_top = r.bottom() - new_h
            
            r.setWidth(new_w)
            r.setTop(new_top)
        
        elif handle_pos == "bl":
            new_left = min(local_pos.x(), r.right() - 20)
            new_h = max(20, local_pos.y() - r.top())
            new_w = r.right() - new_left
            
            if self._fixed_aspect_ratio:
                new_h = new_w / self._fixed_aspect_ratio
            
            r.setLeft(new_left)
            r.setHeight(new_h)
        
        self.setRect(r)
        self.setTransformOriginPoint(r.center())
        self._updateHandlePositions()


class CropToolDialog(QDialog):
    """Multi-image crop tool with alignment and batch cropping"""
    
    def __init__(self, file_dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Crop Images")
        self.setGeometry(100, 100, 1200, 800)
        
        # file_dict: {'ha': path, 'oiii': path, 'oiii2': path (optional), 'sii': path}
        self.file_dict = file_dict
        self.image_data = {}  # Store loaded images
        self.headers = {}     # Store FITS headers
        self.scenes = {}      # Store scenes per channel
        self.views = {}       # Store views per channel
        self.pixmap_items = {}
        self.rect_items = {}  # One rect per channel
        self.stretched_images = {}
        
        # Load all images
        self._load_all_images()
        
        # Build UI
        self._build_ui()
    
    def _load_all_images(self):
        """Load all FITS files"""
        
        for channel, filepath in self.file_dict.items():
            if filepath is None:
                continue
            
            try:
                with fits.open(filepath) as hdul:
                    if hdul[0].data is None and len(hdul) > 1:
                        data = hdul[1].data
                        header = hdul[1].header
                    else:
                        data = hdul[0].data
                        header = hdul[0].header
                    
                    # Handle 3D data
                    if data.ndim == 3:
                        if data.shape[0] == 1:
                            data = data[0]
                        elif data.shape[0] == 3:
                            # Convert to mono (average channels)
                            data = np.mean(data, axis=0)
                    
                    # Keep as original data type (uint16) - don't normalize!
                    # This preserves full precision during crop operations
                    if data.dtype == np.uint16:
                        print(f"[Crop Load] {channel}: Already uint16, max={data.max()}, median={np.median(data[data>0]) if np.any(data>0) else 0:.1f}")
                    elif data.dtype.kind == 'f':  # Any float type (f4, f8, etc.)
                        print(f"[Crop Load] {channel}: Float dtype={data.dtype}, max={data.max():.1f}, median={np.median(data[data>0]) if np.any(data>0) else 0:.6f}")
                        # Float data - check if normalized (0-1) or raw (0-65535)
                        if data.max() <= 1.5:
                            # Normalized float - scale to uint16
                            data = (data * 65535.0).astype(np.uint16)
                            print(f"[Crop Load] {channel}: Scaled normalized float to uint16")
                        else:
                            # Unnormalized float in uint16 range - just cast
                            # But clip to avoid overflow
                            data = np.clip(data, 0, 65535).astype(np.uint16)
                            print(f"[Crop Load] {channel}: Clipped and cast float to uint16")
                    else:
                        print(f"[Crop Load] {channel}: dtype={data.dtype}, max={data.max():.1f}, converting to uint16")
                        # Other integer types - clip and convert
                        data = np.clip(data, 0, 65535).astype(np.uint16)
                    
                    self.image_data[channel] = data
                    self.headers[channel] = header
                    
                    # Final diagnostic - confirm data is in correct range
                    final_max = data.max()
                    final_median = np.median(data[data > 0]) if np.any(data > 0) else 0
                    print(f"[Crop Load] {channel}: FINAL dtype={data.dtype}, max={final_max}, median={final_median:.1f}")
                    
            except Exception as e:
                print(f"[Crop] Error loading {channel}: {e}")
                QMessageBox.warning(self, "Load Error", f"Failed to load {channel}:\n{e}")
    
    def _build_ui(self):
        """Build the UI"""
        layout = QVBoxLayout(self)
        
        # Instructions in styled groupbox
        instr_group = QGroupBox("Crop Tool Instructions")
        instr_group.setStyleSheet(
            "QGroupBox { font-weight: bold; color: #e0e0e0; border: 1px solid #5a5a5a; "
            "border-radius: 4px; margin-top: 6px; padding-top: 10px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }"
        )
        instr_layout = QVBoxLayout(instr_group)
        instr_layout.setContentsMargins(12, 8, 12, 8)
        
        instr = QLabel(
            "• Click-and-drag to draw crop rectangle\n"
            "• Drag corners to resize | Shift + drag to rotate\n"
            "• Ctrl + Mouse Wheel to zoom | Middle-click drag to pan\n"
            "• Same crop will be applied to all channels"
        )
        instr.setAlignment(Qt.AlignmentFlag.AlignLeft)
        instr.setStyleSheet("color: #b0b0b0; font-size: 9pt; font-style: italic;")
        instr_layout.addWidget(instr)
        
        layout.addWidget(instr_group)
        
        # Aspect ratio selection
        ratio_layout = QHBoxLayout()
        ratio_layout.addStretch()
        ratio_layout.addWidget(QLabel("Aspect Ratio:"))
        
        self.aspect_combo = QComboBox()
        self.aspect_combo.addItems(["Free", "Original", "1:1", "16:9", "4:3", "3:2"])
        self.aspect_combo.currentTextChanged.connect(self._on_aspect_changed)
        ratio_layout.addWidget(self.aspect_combo)
        ratio_layout.addStretch()
        layout.addLayout(ratio_layout)
        
        # Tab widget for multiple channels
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Create tab for each channel
        for channel in ['ha', 'oiii', 'oiii2', 'sii']:
            if channel not in self.image_data:
                continue
            
            tab = QWidget()
            tab_layout = QVBoxLayout(tab)
            
            # Graphics view
            scene = QGraphicsScene()
            view = QGraphicsView(scene)
            view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)  # Enable panning
            view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
            view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
            
            self.scenes[channel] = scene
            self.views[channel] = view
            
            # Load stretched image
            stretched = self._autostretch_image(self.image_data[channel])
            self.stretched_images[channel] = stretched
            
            # Convert to QPixmap
            height, width = stretched.shape
            bytes_per_line = width
            q_img = QImage(stretched.data, width, height, bytes_per_line, 
                          QImage.Format.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_img)
            
            pixmap_item = QGraphicsPixmapItem(pixmap)
            scene.addItem(pixmap_item)
            self.pixmap_items[channel] = pixmap_item
            
            # Install event filter for drawing
            view.viewport().installEventFilter(self)
            view.setProperty('channel', channel)  # Store channel name
            
            tab_layout.addWidget(view)
            self.tab_widget.addTab(tab, channel.upper())
        
        # Zoom controls
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QLabel("Zoom:"))
        
        zoom_in_btn = QPushButton("➕ Zoom In")
        zoom_in_btn.clicked.connect(lambda: self._zoom(1.2))
        zoom_layout.addWidget(zoom_in_btn)
        
        zoom_out_btn = QPushButton("➖ Zoom Out")
        zoom_out_btn.clicked.connect(lambda: self._zoom(0.8))
        zoom_layout.addWidget(zoom_out_btn)
        
        fit_btn = QPushButton("⬌ Fit to Window")
        fit_btn.clicked.connect(self._fit_to_window)
        zoom_layout.addWidget(fit_btn)
        
        reset_btn = QPushButton("🔄 100%")
        reset_btn.clicked.connect(self._reset_zoom)
        zoom_layout.addWidget(reset_btn)
        
        zoom_layout.addStretch()
        layout.addLayout(zoom_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        apply_btn = QPushButton("✓ Apply Crop to All Channels")
        apply_btn.setStyleSheet(
            "QPushButton {"
            "  background-color: #27ae60; color: white; font-weight: bold; "
            "  font-size: 11pt; padding: 12px; border-radius: 5px;"
            "}"
            "QPushButton:hover { background-color: #229954; }"
        )
        apply_btn.clicked.connect(self._apply_crop)
        button_layout.addWidget(apply_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet("font-size: 11pt; padding: 12px;")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
        # Remember original aspect ratio
        first_channel = list(self.image_data.keys())[0]
        h, w = self.image_data[first_channel].shape
        self._orig_ar = w / h
        
        self.drawing = False
        self.origin = QPointF()
        self.current_channel_rect = None
        
        # Fit all views to window on startup
        QTimer.singleShot(100, self._fit_to_window)
    
    def _autostretch_image(self, data):
        """Apply aggressive autostretch for visibility"""
        # Normalize if needed (data might be uint16)
        if data.max() > 1.5:
            data_norm = data.astype(np.float32) / 65535.0
        else:
            data_norm = data.astype(np.float32)
        
        # Use tighter percentiles
        valid = data_norm[data_norm > 0]
        if len(valid) == 0:
            valid = data_norm
        
        p_low, p_high = np.percentile(valid, [0.01, 99.9])
        
        if p_high > p_low:
            stretched = (data_norm - p_low) / (p_high - p_low)
            stretched = np.clip(stretched, 0, 1)
            # Apply gamma for more visibility
            stretched = np.power(stretched, 0.7)
        else:
            stretched = data_norm
        
        return (stretched * 255).astype(np.uint8)
    
    def _zoom(self, factor):
        """Zoom current view by factor"""
        current_tab = self.tab_widget.currentIndex()
        channel = list(self.views.keys())[current_tab]
        view = self.views[channel]
        view.scale(factor, factor)
    
    def _fit_to_window(self):
        """Fit image to window for all views"""
        for channel, view in self.views.items():
            view.fitInView(view.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
    
    def _reset_zoom(self):
        """Reset zoom to 100% for current view"""
        current_tab = self.tab_widget.currentIndex()
        channel = list(self.views.keys())[current_tab]
        view = self.views[channel]
        view.resetTransform()
    
    def _on_aspect_changed(self, text):
        """Update aspect ratio for all rectangles"""
        if text == "Free":
            ar = None
        elif text == "Original":
            ar = self._orig_ar
        else:
            parts = text.split(":")
            if len(parts) == 2:
                ar = float(parts[0]) / float(parts[1])
            else:
                ar = None
        
        # Update all rectangles
        for rect_item in self.rect_items.values():
            rect_item.setFixedAspectRatio(ar)
            
            # Reshape if ratio set
            if ar is not None:
                old = rect_item.rect()
                w = old.width()
                h = w / ar
                cx, cy = old.center().x(), old.center().y()
                new_rect = QRectF(cx - w/2, cy - h/2, w, h)
                rect_item.setRect(new_rect)
                rect_item.setTransformOriginPoint(new_rect.center())
                rect_item._updateHandlePositions()
    
    def eventFilter(self, obj, event):
        """Handle drawing rectangle on viewport and mouse wheel zoom"""
        # Get the view that generated this event
        view = None
        for v in self.views.values():
            if obj == v.viewport():
                view = v
                break
        
        if view is None:
            return super().eventFilter(obj, event)
        
        channel = view.property('channel')
        scene = self.scenes[channel]
        
        # Handle mouse wheel zoom
        if event.type() == event.Type.Wheel:
            if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
                # Zoom with Ctrl+Wheel
                delta = event.angleDelta().y()
                if delta > 0:
                    factor = 1.15
                else:
                    factor = 1 / 1.15
                view.scale(factor, factor)
                return True
        
        if event.type() == event.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.LeftButton:
                # Start drawing if clicking on empty area (not on existing rect)
                scene_pos = view.mapToScene(event.pos())
                item = scene.itemAt(scene_pos, view.transform())
                
                if not isinstance(item, (ResizableRectItem, QGraphicsEllipseItem)):
                    self.drawing = True
                    self.origin = scene_pos
                    self.current_channel_rect = channel
                    # Disable panning while drawing
                    view.setDragMode(QGraphicsView.DragMode.NoDrag)
                    return True
        
        elif event.type() == event.Type.MouseMove:
            if self.drawing:
                scene_pos = view.mapToScene(event.pos())
                
                # Remove old temp rect if exists
                if channel in self.rect_items:
                    scene.removeItem(self.rect_items[channel])
                
                # Create new rect
                rect = QRectF(self.origin, scene_pos).normalized()
                rect_item = ResizableRectItem(rect)
                
                # Apply current aspect ratio
                text = self.aspect_combo.currentText()
                if text == "Free":
                    ar = None
                elif text == "Original":
                    ar = self._orig_ar
                else:
                    parts = text.split(":")
                    ar = float(parts[0]) / float(parts[1]) if len(parts) == 2 else None
                
                rect_item.setFixedAspectRatio(ar)
                scene.addItem(rect_item)
                self.rect_items[channel] = rect_item
                
                return True
        
        elif event.type() == event.Type.MouseButtonRelease:
            if event.button() == Qt.MouseButton.LeftButton and self.drawing:
                self.drawing = False
                
                # Re-enable panning
                view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
                
                # Synchronize rectangle to all other channels
                if channel in self.rect_items:
                    self._sync_rectangles_from(channel)
                
                return True
        
        return super().eventFilter(obj, event)
    
    def _sync_rectangles_from(self, source_channel):
        """Copy rectangle from source channel to all other channels"""
        if source_channel not in self.rect_items:
            return
        
        source_rect = self.rect_items[source_channel].rect()
        source_rotation = self.rect_items[source_channel].rotation()
        
        for channel in self.scenes.keys():
            if channel == source_channel:
                continue
            
            # Remove old rect
            if channel in self.rect_items:
                self.scenes[channel].removeItem(self.rect_items[channel])
            
            # Create new rect with same dimensions
            rect_item = ResizableRectItem(source_rect)
            rect_item.setRotation(source_rotation)
            
            # Apply aspect ratio
            text = self.aspect_combo.currentText()
            if text == "Free":
                ar = None
            elif text == "Original":
                ar = self._orig_ar
            else:
                parts = text.split(":")
                ar = float(parts[0]) / float(parts[1]) if len(parts) == 2 else None
            rect_item.setFixedAspectRatio(ar)
            
            self.scenes[channel].addItem(rect_item)
            self.rect_items[channel] = rect_item
    
    def _apply_crop(self):
        """Apply crop to all channels and save"""
        if not self.rect_items:
            QMessageBox.warning(self, "No Selection", "Draw a crop rectangle first")
            return
        
        # Use first channel's rectangle as reference
        first_channel = list(self.rect_items.keys())[0]
        rect_item = self.rect_items[first_channel]
        
        # Get rectangle corners in scene coords
        rect_local = rect_item.rect()
        corners_local = [
            rect_local.topLeft(), rect_local.topRight(),
            rect_local.bottomRight(), rect_local.bottomLeft()
        ]
        corners_scene = [rect_item.mapToScene(pt) for pt in corners_local]
        
        # Map to image pixel coordinates
        pm = self.pixmap_items[first_channel].pixmap()
        pm_w, pm_h = pm.width(), pm.height()
        h_img, w_img = self.image_data[first_channel].shape
        sx, sy = w_img / pm_w, h_img / pm_h
        
        src_pts = np.array([
            [p.x() * sx, p.y() * sy] for p in corners_scene
        ], dtype=np.float32)
        
        # Compute output size
        width = np.linalg.norm(src_pts[1] - src_pts[0])
        height = np.linalg.norm(src_pts[3] - src_pts[0])
        w_out, h_out = int(round(width)), int(round(height))
        
        dst_pts = np.array([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ], dtype=np.float32)
        
        # Get perspective transform
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # Apply to all channels
        cropped_files = {}
        
        for channel, data in self.image_data.items():
            
            # Convert to float32 for warp to avoid uint16 rounding errors
            # This preserves precision during interpolation
            if data.dtype == np.uint16:
                data_float = data.astype(np.float32)
            else:
                data_float = data
            
            # Apply perspective warp on float32 data
            cropped_float = cv2.warpPerspective(
                data_float, M, (w_out, h_out),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            
            # Convert back to uint16 for saving
            cropped = cropped_float.astype(np.uint16)
            
            # Diagnostic: check data integrity
            orig_median = np.median(data[data > 0]) if np.any(data > 0) else 0
            crop_median = np.median(cropped[cropped > 0]) if np.any(cropped > 0) else 0
            print(f"[Crop] {channel}: Original median={orig_median:.1f}, Cropped median={crop_median:.1f}, dtype={cropped.dtype}")
            
            # Save with _Cropped suffix
            original_path = self.file_dict[channel]
            base, ext = os.path.splitext(original_path)
            # Handle .fit.fz
            if base.endswith('.fit'):
                base = base[:-4]
                ext = '.fit'
            
            cropped_path = f"{base}_Cropped{ext}"
            
            # Save FITS as 16-bit integer (already in correct format from warpPerspective)
            fits.writeto(cropped_path, cropped,
                        self.headers[channel], overwrite=True)
            
            cropped_files[channel] = cropped_path
        
        # Store results and close
        self.cropped_files = cropped_files
        QMessageBox.information(self, "Crop Complete", 
                               f"✓ Cropped {len(cropped_files)} files\n\n"
                               f"Output size: {w_out}x{h_out} pixels")
        self.accept()
class SASPaletteReplicator(QWidget):
    def __init__(self):
        super().__init__()
        self.siril = s.SirilInterface()
        
        self.ha_file = None
        self.oiii_file = None
        self.oiii2_file = None
        self.sii_file = None
        
        self.fits_header = None
        self.temp_files = []
        
        self.preview_window = None
        self.preview_thread = None
class SASPaletteReplicator(QWidget):
    def __init__(self):
        super().__init__()
        self.siril = s.SirilInterface()
        
        self.ha_file = None
        self.oiii_file = None
        self.oiii2_file = None
        self.sii_file = None
        
        self.fits_header = None
        self.temp_files = []
        
        self.preview_window = None
        self.preview_thread = None
        
        # Live preview state
        self._live_preview_label  = None
        self._live_preview_dialog = None
        self._live_ha_raw    = None
        self._live_oiii_base = None
        self._live_oiii2_base = None
        self._live_sii_raw   = None

        # Debounce timer — delays preview render until 150ms after last slider move
        self._preview_refresh_timer = QTimer(self)
        self._preview_refresh_timer.setSingleShot(True)
        self._preview_refresh_timer.setInterval(150)
        self._preview_refresh_timer.timeout.connect(self._refresh_live_preview)
        
        # Channel balance values (for blend strength dialog)
        self.ha_balance_value = 100
        self.oiii_balance_value = 100
        self.sii_balance_value = 100
        
        # Preview scale (50% to 150%)
        self.preview_scale = 100  # Default 100%
        
        self.initUI()
        self.load_settings()  # Load saved settings after UI is initialized
    
    def load_settings(self):
        """Load settings from config file"""
        try:
            if CONFIG_FILE.exists():
                with open(CONFIG_FILE, 'r') as f:
                    settings = json.load(f)
                
                #print(f"Loading saved settings from {CONFIG_FILE}")
                
                # Restore file paths
                if settings.get('last_ha_path') and os.path.exists(settings['last_ha_path']):
                    self.ha_file = settings['last_ha_path']
                    self.ha_label.setText(os.path.basename(self.ha_file))
                
                if settings.get('last_oiii_path') and os.path.exists(settings['last_oiii_path']):
                    self.oiii_file = settings['last_oiii_path']
                    self.oiii_label.setText(os.path.basename(self.oiii_file))
                
                if settings.get('last_sii_path') and os.path.exists(settings['last_sii_path']):
                    self.sii_file = settings['last_sii_path']
                    self.sii_label.setText(os.path.basename(self.sii_file))
                
                if settings.get('last_oiii2_path') and os.path.exists(settings['last_oiii2_path']):
                    self.oiii2_file = settings['last_oiii2_path']
                    self.oiii2_label.setText(os.path.basename(self.oiii2_file))
                
                # Re-evaluate button state now that files are restored
                self.update_balance_button_state()
                
                # Restore UI settings
                self.stretch_mode.setCurrentIndex(settings.get('stretch_mode', 2))
                self.stretch_slider.setValue(int(settings.get('stretch_strength', 0.25) * 100))
                self.palette_pick.setCurrentIndex(settings.get('palette_index', 0))
                
                self.ha_balance_value = int(settings.get('ha_balance', 1.0) * 100)
                self.oiii_balance_value = int(settings.get('oiii_balance', 1.0) * 100)
                self.sii_balance_value = int(settings.get('sii_balance', 1.0) * 100)
                
                self.oiii_combine_mode.setCurrentIndex(settings.get('oiii_combine_mode', 0))
                
                self.curves_checkbox.setChecked(settings.get('curves_enabled', False))
                self.preview_curves_checkbox.setChecked(settings.get('preview_curves_enabled', False))
                self.preview_linked_checkbox.setChecked(settings.get('preview_linked', False))
                self.unstretch_checkbox.setChecked(settings.get('unstretch_enabled', False))
                self.unstretch_strength.setCurrentIndex(settings.get('unstretch_strength_index', 1))
                self.save_result_checkbox.setChecked(settings.get('save_result', False))
                self.blend_strength_checkbox.setChecked(settings.get('blend_strength_enabled', False))
                self.scnr_checkbox.setChecked(settings.get('scnr_enabled', False))
                self.synth_lum_checkbox.setChecked(settings.get('synth_lum_enabled', False))
                self.synth_lum_strength.setValue(settings.get('synth_lum_strength', 100))
                
                self.preview_scale = settings.get('preview_scale', 100)
                self.preview_scale_slider.setValue(self.preview_scale)
                
                # Don't restore slider positions - always start at 100 (1.0x)
                self.ha_gain_slider.setValue(100)
                self.oiii_gain_slider.setValue(100)
                self.sii_gain_slider.setValue(100)
                
                #print("Settings loaded successfully")
                self.status_label.setText("Previous settings restored")
        
        except Exception as e:
            print(f"Could not load settings: {e}")
    
    def save_settings(self):
        """Save current settings to config file"""
        try:
            settings = {
                'last_ha_path': self.ha_file or '',
                'last_oiii_path': self.oiii_file or '',
                'last_sii_path': self.sii_file or '',
                'last_oiii2_path': self.oiii2_file or '',
                'stretch_mode': self.stretch_mode.currentIndex(),
                'stretch_strength': self.stretch_slider.value() / 100.0,
                'palette_index': self.palette_pick.currentIndex(),
                'ha_balance': self.ha_balance_value / 100.0,
                'oiii_balance': self.oiii_balance_value / 100.0,
                'sii_balance': self.sii_balance_value / 100.0,
                'oiii_combine_mode': self.oiii_combine_mode.currentIndex(),
                'curves_enabled': self.curves_checkbox.isChecked(),
                'preview_curves_enabled': self.preview_curves_checkbox.isChecked(),
                'preview_linked': self.preview_linked_checkbox.isChecked(),
                'unstretch_enabled': self.unstretch_checkbox.isChecked(),
                'unstretch_strength_index': self.unstretch_strength.currentIndex(),
                'save_result': self.save_result_checkbox.isChecked(),
                'blend_strength_enabled': self.blend_strength_checkbox.isChecked(),
                'scnr_enabled': self.scnr_checkbox.isChecked(),
                'synth_lum_enabled': self.synth_lum_checkbox.isChecked(),
                'synth_lum_strength': self.synth_lum_strength.value(),
                'preview_scale': self.preview_scale,
                # Don't save slider positions - always start fresh at 100
            }
            
            with open(CONFIG_FILE, 'w') as f:
                json.dump(settings, indent=2, fp=f)
            
            #print(f"Settings saved to {CONFIG_FILE}")
        
        except Exception as e:
            print(f"Could not save settings: {e}")
    
    def initUI(self):
        main_layout = QVBoxLayout()
        
        self.setWindowTitle('Narrowband Palette Picker - V3.5')
        self.setMinimumWidth(650)  # Reduced from 750 since we removed sliders
        # Position window at top of screen
        screen = QApplication.primaryScreen().availableGeometry()
        self.move(screen.x() + (screen.width() // 2) , screen.y())
        # ==================== HEADER SECTION ====================
        # Usage notes and restore button in horizontal layout
        header_layout = QHBoxLayout()
        header_layout.setSpacing(10)
        
        # Usage notes (subtle, informative)
        usage_frame = QWidget()
        usage_frame.setStyleSheet(
            "background-color: #fff9e6; "
            "border-left: 4px solid #f39c12; "
            "border-radius: 3px; "
            "padding: 8px;"
        )
        usage_layout = QVBoxLayout(usage_frame)
        usage_layout.setContentsMargins(8, 5, 8, 5)
        
        usage_label = QLabel(
            "<span style='color: #7f8c8d; font-size: 9pt; font-weight: bold;'>USAGE NOTES:</span><br>"
            "<span style='color: #34495e; font-size: 9pt;'>"
            "• Use <b>starless, cropped, and aligned linear data</b> where possible<br>"
            "• If using the save palette result to file function, the set home path will be used.<br>"
            "• If missing Sii data, you may re-use another channel<br>"
            "• <b>Foraxx palette requires a stretch mode</b> to display correctly"
            "</span>"
        )
        usage_label.setWordWrap(True)
        usage_layout.addWidget(usage_label)
        header_layout.addWidget(usage_frame)
        
        # Restore defaults button (on the right)
        restore_btn = QPushButton("⟲ Restore\nDefaults")
        restore_btn.setStyleSheet(
            "QPushButton {"
            "  background-color: #95a5a6; "
            "  color: white; "
            "  font-weight: bold; "
            "  padding: 8px 12px; "
            "  border-radius: 4px; "
            "  font-size: 9pt; "
            "}"
            "QPushButton:hover {"
            "  background-color: #7f8c8d;"
            "}"
        )
        restore_btn.setMaximumWidth(100)
        restore_btn.clicked.connect(self.restore_defaults)
        header_layout.addWidget(restore_btn, alignment=Qt.AlignmentFlag.AlignTop)
        
        main_layout.addLayout(header_layout)
        
        main_layout.addSpacing(10)
        
        # ==================== TABBED INTERFACE ====================
        tabs = QTabWidget()
        tabs.setStyleSheet(
            "QTabWidget::pane { "
            "  border: 2px solid #5a5a5a; "
            "  border-radius: 5px; "
            "  background-color: #3c3c3c; "  # Dark grey to match Siril
            "  padding: 5px; "
            "}"
            "QTabBar::tab { "
            "  background-color: #2d2d2d; "
            "  color: #e0e0e0; "  # Light text on dark background
            "  padding: 10px 20px; "
            "  margin-right: 2px; "
            "  border: 1px solid #5a5a5a; "
            "  border-bottom: none; "
            "  border-top-left-radius: 4px; "
            "  border-top-right-radius: 4px; "
            "  font-weight: bold; "
            "  font-size: 11pt; "
            "}"
            "QTabBar::tab:selected { "
            "  background-color: #3c3c3c; "  # Match content background
            "  color: #ffffff; "
            "  border-bottom: 2px solid #3c3c3c; "  # Hide border with content
            "}"
            "QTabBar::tab:hover { "
            "  background-color: #4a4a4a; "
            "}"
        )
        
        # ==================== TAB 1: FILE SELECTION ====================
        file_tab = QWidget()
        file_tab.setStyleSheet("background-color: #3c3c3c; color: #e0e0e0;")  # Dark grey with light text
        file_layout = QVBoxLayout(file_tab)
        file_layout.setContentsMargins(10, 10, 10, 10)
        file_layout.setSpacing(8)
        
        # Ha file
        ha_layout = QHBoxLayout()
        ha_layout.addWidget(QLabel("Ha (Hydrogen):"))
        self.ha_label = QLineEdit("No file selected")
        self.ha_label.setReadOnly(True)
        ha_layout.addWidget(self.ha_label)
        ha_btn = QPushButton("Browse...")
        ha_btn.clicked.connect(lambda: self.select_file('ha'))
        ha_layout.addWidget(ha_btn)
        ha_clear_btn = QPushButton("Clear")
        ha_clear_btn.clicked.connect(lambda: self.clear_file('ha'))
        ha_layout.addWidget(ha_clear_btn)
        file_layout.addLayout(ha_layout)
        
        # OIII file
        oiii_layout = QHBoxLayout()
        oiii_layout.addWidget(QLabel("OIII (Oxygen):"))
        self.oiii_label = QLineEdit("No file selected")
        self.oiii_label.setReadOnly(True)
        oiii_layout.addWidget(self.oiii_label)
        oiii_btn = QPushButton("Browse...")
        oiii_btn.clicked.connect(lambda: self.select_file('oiii'))
        oiii_layout.addWidget(oiii_btn)
        oiii_clear_btn = QPushButton("Clear")
        oiii_clear_btn.clicked.connect(lambda: self.clear_file('oiii'))
        oiii_layout.addWidget(oiii_clear_btn)
        file_layout.addLayout(oiii_layout)
        
        # OIII #2 file
        oiii2_layout = QHBoxLayout()
        oiii2_layout.addWidget(QLabel("OIII #2 (Optional):"))
        self.oiii2_label = QLineEdit("No file selected (will combine if provided)")
        self.oiii2_label.setReadOnly(True)
        oiii2_layout.addWidget(self.oiii2_label)
        oiii2_btn = QPushButton("Browse...")
        oiii2_btn.clicked.connect(lambda: self.select_file('oiii2'))
        oiii2_layout.addWidget(oiii2_btn)
        oiii2_clear_btn = QPushButton("Clear")
        oiii2_clear_btn.clicked.connect(lambda: self.clear_file('oiii2'))
        oiii2_layout.addWidget(oiii2_clear_btn)
        file_layout.addLayout(oiii2_layout)
        
        # OIII combine mode
        oiii_combine_layout = QHBoxLayout()
        oiii_combine_layout.addWidget(QLabel("OIII Combine Mode:"))
        self.oiii_combine_mode = QComboBox()
        self.oiii_combine_mode.addItems(["Average", "Maximum", "Weighted (70/30)"])
        oiii_combine_layout.addWidget(self.oiii_combine_mode)
        oiii_combine_layout.addStretch()  # Push everything left
        file_layout.addLayout(oiii_combine_layout)
        
        # SII file
        sii_layout = QHBoxLayout()
        sii_layout.addWidget(QLabel("SII (Sulfur):"))
        self.sii_label = QLineEdit("No file selected")
        self.sii_label.setReadOnly(True)
        sii_layout.addWidget(self.sii_label)
        sii_btn = QPushButton("Browse...")
        sii_btn.clicked.connect(lambda: self.select_file('sii'))
        sii_layout.addWidget(sii_btn)
        sii_clear_btn = QPushButton("Clear")
        sii_clear_btn.clicked.connect(lambda: self.clear_file('sii'))
        sii_layout.addWidget(sii_clear_btn)
        file_layout.addLayout(sii_layout)
        
        file_layout.addSpacing(15)
        
        # Crop button
        crop_layout = QHBoxLayout()
        self.crop_button = QPushButton("✂ Crop All Images")
        self.crop_button.setEnabled(False)
        self.crop_button.setStyleSheet(
            "QPushButton {"
            "  background-color: #e67e22; "  # Orange
            "  color: white; "
            "  font-weight: bold; "
            "  font-size: 11pt; "
            "  padding: 12px; "
            "  border-radius: 5px;"
            "}"
            "QPushButton:hover {"
            "  background-color: #d35400;"
            "}"
            "QPushButton:pressed {"
            "  background-color: #ba4a00;"
            "}"
            "QPushButton:disabled {"
            "  background-color: #7f8c8d;"
            "  color: #bdc3c7;"
            "}"
        )
        self.crop_button.setToolTip(
            "Crop all loaded images with the same selection.\n"
            "Saves cropped files with _Cropped suffix and updates file selection."
        )
        self.crop_button.clicked.connect(self.show_crop_tool)
        crop_layout.addWidget(self.crop_button)
        crop_layout.addStretch()
        file_layout.addLayout(crop_layout)
        
        file_layout.addSpacing(15)
        
        # ---- Channel Balance Sliders ----
        balance_group = QGroupBox("Channel Balance (linear gain applied before processing)")
        balance_group.setStyleSheet(
            "QGroupBox { font-weight: bold; color: #e0e0e0; border: 1px solid #5a5a5a; "
            "border-radius: 4px; margin-top: 6px; padding-top: 6px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }"
        )
        balance_sliders_layout = QVBoxLayout(balance_group)
        balance_sliders_layout.setContentsMargins(8, 8, 8, 6)
        balance_sliders_layout.setSpacing(4)

        def _make_balance_slider(label_text, init_val):
            row = QHBoxLayout()
            lbl = QLabel(label_text)
            lbl.setFixedWidth(38)
            lbl.setStyleSheet("color: #e0e0e0; font-size: 9pt;")
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(10)    # 0.10x
            slider.setMaximum(1500)  # 15.00x (increased for very weak channels)
            slider.setValue(init_val)
            val_lbl = QLabel(f"{init_val / 100:.2f}x")
            val_lbl.setFixedWidth(48)
            val_lbl.setStyleSheet("color: #e0e0e0; font-size: 9pt;")
            slider.valueChanged.connect(lambda v, vl=val_lbl: vl.setText(f"{v/100:.2f}x"))
            row.addWidget(lbl)
            row.addWidget(slider)
            row.addWidget(val_lbl)
            return row, slider

        ha_row,   self.ha_gain_slider   = _make_balance_slider("Ha:",   100)
        oiii_row, self.oiii_gain_slider = _make_balance_slider("OIII:", 100)
        sii_row,  self.sii_gain_slider  = _make_balance_slider("SII:",  100)
        # Store value labels so we can update them when blockSignals suppresses valueChanged
        self._ha_gain_label   = ha_row.itemAt(2).widget()
        self._oiii_gain_label = oiii_row.itemAt(2).widget()
        self._sii_gain_label  = sii_row.itemAt(2).widget()
        balance_sliders_layout.addLayout(ha_row)
        balance_sliders_layout.addLayout(oiii_row)
        balance_sliders_layout.addLayout(sii_row)

        # Enable Clear Balance whenever any slider moves away from 1.0x
        def _on_gain_slider_changed(_):
            any_non_default = (
                self.ha_gain_slider.value()   != 100 or
                self.oiii_gain_slider.value() != 100 or
                self.sii_gain_slider.value()  != 100
            )
            self.clear_balance_button.setEnabled(any_non_default)

        # Connected after clear_balance_button is created — use a post-init hook via a timer
        self._gain_slider_changed_hook = _on_gain_slider_changed

        file_layout.addWidget(balance_group)

        file_layout.addSpacing(6)
        
        # Channel balancing section
        # Channel tools row: Preview | Balance | Clear Balance | Status
        tools_layout = QHBoxLayout()
        
        # Live Preview button (left)
        self.file_preview_btn = QPushButton('🔍 Live Preview')
        self.file_preview_btn.setEnabled(False)
        self.file_preview_btn.setStyleSheet(
            "QPushButton {"
            "  background-color: #9b59b6; "  # Purple
            "  color: white; "
            "  font-weight: bold; "
            "  font-size: 11pt; "
            "  padding: 12px; "
            "  border-radius: 5px;"
            "}"
            "QPushButton:hover {"
            "  background-color: #8e44ad;"
            "}"
            "QPushButton:pressed {"
            "  background-color: #7d3c98;"
            "}"
            "QPushButton:disabled {"
            "  background-color: #7f8c8d;"
            "  color: #bdc3c7;"
            "}"
        )
        self.file_preview_btn.setToolTip("Preview combined channels with basic stretch")
        self.file_preview_btn.clicked.connect(self.show_file_preview)
        tools_layout.addWidget(self.file_preview_btn)
        
        tools_layout.addSpacing(10)
        
        # Auto Balance button
        self.balance_button = QPushButton("⚖ Auto Balance Channels")
        self.balance_button.setEnabled(False)
        self.balance_button.setStyleSheet(
            "QPushButton {"
            "  background-color: #3498db; "  # Blue
            "  color: white; "
            "  font-weight: bold; "
            "  font-size: 11pt; "
            "  padding: 12px; "
            "  border-radius: 5px;"
            "}"
            "QPushButton:hover {"
            "  background-color: #2980b9;"
            "}"
            "QPushButton:pressed {"
            "  background-color: #21618c;"
            "}"
            "QPushButton:disabled {"
            "  background-color: #7f8c8d;"
            "  color: #bdc3c7;"
            "}"
        )
        self.balance_button.setToolTip(
            "Automatically balance channel brightness.\n"
            "1. Subtracts sky background from each channel\n"
            "2. Boosts dimmer channels to match the brightest\n"
            "Makes colors consistent between linked and unlinked stretch."
        )
        self.balance_button.clicked.connect(self.auto_balance_channels)
        tools_layout.addWidget(self.balance_button)
        
        # Clear Balance button
        self.clear_balance_button = QPushButton("✖ Clear Balance")
        self.clear_balance_button.setEnabled(False)
        self.clear_balance_button.setStyleSheet(
            "QPushButton {"
            "  background-color: #e74c3c; "  # Red
            "  color: white; "
            "  font-weight: bold; "
            "  font-size: 11pt; "
            "  padding: 12px; "
            "  border-radius: 5px;"
            "}"
            "QPushButton:hover {"
            "  background-color: #c0392b;"
            "}"
            "QPushButton:pressed {"
            "  background-color: #a93226;"
            "}"
            "QPushButton:disabled {"
            "  background-color: #7f8c8d;"
            "  color: #bdc3c7;"
            "}"
        )
        self.clear_balance_button.setToolTip("Revert to original unbalanced channels and reset sliders to 1.0x")
        self.clear_balance_button.clicked.connect(self.clear_balance)

        # Now wire gain sliders to enable Clear Balance whenever any is off 1.0x
        self.ha_gain_slider.valueChanged.connect(self._gain_slider_changed_hook)
        self.oiii_gain_slider.valueChanged.connect(self._gain_slider_changed_hook)
        self.sii_gain_slider.valueChanged.connect(self._gain_slider_changed_hook)
        tools_layout.addWidget(self.clear_balance_button)
        
        # Status label
        self.balance_status = QLabel("")
        self.balance_status.setStyleSheet("color: #4CAF50; font-style: italic; font-size: 10pt;")
        tools_layout.addWidget(self.balance_status)
        
        tools_layout.addStretch()
        file_layout.addLayout(tools_layout)
        
        file_layout.addSpacing(15)
        
        # Synthetic luminance option
        synth_lum_layout = QHBoxLayout()
        synth_lum_layout.addSpacing(20)
        self.synth_lum_checkbox = QCheckBox("Create Synthetic Luminance")
        self.synth_lum_checkbox.setChecked(False)
        self.synth_lum_checkbox.setToolTip(
            "Combines all channels (Ha + OIII + SII) to create a high-SNR luminance layer.\n"
            "This preserves fine detail and improves star sharpness.\n"
            "The palette provides color, while the synthetic luminance provides structure."
        )
        synth_lum_layout.addWidget(self.synth_lum_checkbox)
        
        # Luminance blend strength slider
        synth_lum_layout.addWidget(QLabel("Blend:"))
        self.synth_lum_strength = QSlider(Qt.Orientation.Horizontal)
        self.synth_lum_strength.setMinimum(0)
        self.synth_lum_strength.setMaximum(100)
        self.synth_lum_strength.setValue(100)
        self.synth_lum_strength.setMaximumWidth(150)
        self.synth_lum_strength.setToolTip("0% = pure palette color, 100% = full luminance detail")
        synth_lum_layout.addWidget(self.synth_lum_strength)
        self.synth_lum_label = QLabel("100%")
        self.synth_lum_label.setMinimumWidth(40)
        self.synth_lum_strength.valueChanged.connect(lambda v: self.synth_lum_label.setText(f"{v}%"))
        synth_lum_layout.addWidget(self.synth_lum_label)
        synth_lum_layout.addStretch()
        file_layout.addLayout(synth_lum_layout)
        
        file_layout.addStretch()  # Push content to top
        
        # ==================== TAB 2: PROCESSING OPTIONS ====================
        options_tab = QWidget()
        options_tab.setStyleSheet("background-color: #3c3c3c; color: #e0e0e0;")  # Dark grey with light text
        options_layout = QVBoxLayout(options_tab)
        options_layout.setContentsMargins(10, 10, 10, 10)
        options_layout.setSpacing(8)

        # --- Palette Selector Subsection ---
        palette_subsection = QLabel("Palette Selector")
        palette_subsection.setStyleSheet(
            "font-size: 10pt; "
            "font-weight: bold; "
            "color: #e0e0e0; "
            "padding: 5px 0px 2px 5px;"
        )
        options_layout.addWidget(palette_subsection)

        self.palette_pick = QComboBox()
        self.palette_pick.addItems([
            "Foraxx", "Realistic1", "Realistic2",
            "HOO (Narrowband Ha-OIII)", "HOS", "HSO", "HSS",
            "OHH", "OHS", "OSH", "OSS",
            "SHH", "SHO (Hubble)", "SOH", "SOO", "SSH"
        ])
        options_layout.addWidget(self.palette_pick)
        
        options_layout.addSpacing(5)  # Reduced from 10

        # --- Stretch Options Subsection ---
        stretch_subsection = QLabel("Stretch Options")
        stretch_subsection.setStyleSheet(
            "font-size: 10pt; "
            "font-weight: bold; "
            "color: #e0e0e0; "
            "padding: 5px 0px 2px 5px;"
        )
        options_layout.addWidget(stretch_subsection)

        self.stretch_mode = QComboBox()
        self.stretch_mode.addItems([
            "No stretch (linear output)",
            "Statistical Stretch - Linked (same parameters for all channels)",
            "Statistical Stretch - Unlinked (independent per channel)"
        ])
        self.stretch_mode.setCurrentIndex(2)
        self.stretch_mode.currentIndexChanged.connect(self.on_stretch_mode_changed)
        options_layout.addWidget(self.stretch_mode)

        # Stretch strength slider
        stretch_layout = QHBoxLayout()
        stretch_layout.addWidget(QLabel("Stretch Strength:"))
        self.stretch_slider = QSlider(Qt.Orientation.Horizontal)
        self.stretch_slider.setMinimum(10)
        self.stretch_slider.setMaximum(50)
        self.stretch_slider.setValue(15)
        self.stretch_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.stretch_slider.setTickInterval(5)
        self.stretch_slider.valueChanged.connect(self.update_stretch_label)
        stretch_layout.addWidget(self.stretch_slider)
        self.stretch_value_label = QLabel("0.15")
        stretch_layout.addWidget(self.stretch_value_label)
        options_layout.addLayout(stretch_layout)

        # Stretch-related checkboxes (indented slightly)
        self.curves_checkbox = QCheckBox("Apply auto curves adjustment to output")
        self.curves_checkbox.setChecked(False)
        self.curves_checkbox.setStyleSheet("padding-left: 20px;")
        self.curves_checkbox.setToolTip(
            "Applies PPP's curves adjustment after stretching.\n"
            "This enhances contrast and adds visual 'pop' to the image.\n"
            "Only works when stretch mode is enabled."
        )
        options_layout.addWidget(self.curves_checkbox)
        
        self.save_result_checkbox = QCheckBox("Save palette result to file")
        self.save_result_checkbox.setChecked(False)
        self.save_result_checkbox.setStyleSheet("padding-left: 20px;")
        self.save_result_checkbox.setToolTip(
            "When checked, saves the combined palette as 'PaletteName.fit'.\n"
            "When unchecked, only loads the result into Siril without saving."
        )
        options_layout.addWidget(self.save_result_checkbox)
        
        options_layout.addSpacing(5)
        
        # --- Inverse Stretch Subsection ---
        unstretch_subsection = QLabel("Inverse Stretch Options")
        unstretch_subsection.setStyleSheet(
            "font-size: 10pt; "
            "font-weight: bold; "
            "color: #e0e0e0; "
            "padding: 5px 0px 2px 5px;"
        )
        options_layout.addWidget(unstretch_subsection)
        
        self.unstretch_checkbox = QCheckBox("Apply inverse stretch to return to linear data")
        self.unstretch_checkbox.setChecked(False)
        self.unstretch_checkbox.setStyleSheet("padding-left: 20px;")
        self.unstretch_checkbox.setToolTip(
            "After palette processing, applies PPP's inverse stretch to return to linear brightness.\n"
            "This gives you linear data that you can re-stretch with other tools.\n"
            "Note: Color ratios will differ from original due to palette processing."
        )
        self.unstretch_checkbox.stateChanged.connect(self.on_unstretch_changed)
        options_layout.addWidget(self.unstretch_checkbox)
        
        unstretch_strength_layout = QHBoxLayout()
        unstretch_strength_layout.addSpacing(20)
        unstretch_strength_layout.addWidget(QLabel("Unstretch Strength:"))
        self.unstretch_strength = QComboBox()
        self.unstretch_strength.addItems(["Light", "Moderate", "Aggressive"])
        self.unstretch_strength.setCurrentIndex(1)
        self.unstretch_strength.setEnabled(False)
        self.unstretch_strength.setToolTip(
            "Light: Less aggressive, brighter result (good for faint targets)\n"
            "Moderate: Balanced approach (recommended)\n"
            "Aggressive: Most linear, darker result (closest to original brightness)"
        )
        unstretch_strength_layout.addWidget(self.unstretch_strength)
        unstretch_strength_layout.addStretch()
        options_layout.addLayout(unstretch_strength_layout)
        
        options_layout.addSpacing(5)
        
        # --- Preview Options Subsection ---
        preview_subsection = QLabel("Preview Options")
        preview_subsection.setStyleSheet(
            "font-size: 10pt; "
            "font-weight: bold; "
            "color: #e0e0e0; "
            "padding: 5px 0px 2px 5px;"
        )
        options_layout.addWidget(preview_subsection)
        
        self.preview_linked_checkbox = QCheckBox("Use linked stretch for previews")
        self.preview_linked_checkbox.setChecked(False)
        self.preview_linked_checkbox.setStyleSheet("padding-left: 20px;")
        self.preview_linked_checkbox.setToolTip(
            "Linked: All channels stretched together (more natural colors)\n"
            "Unlinked: Each channel stretched independently (more color separation)"
        )
        options_layout.addWidget(self.preview_linked_checkbox)
        
        self.preview_curves_checkbox = QCheckBox("Apply auto curves to previews")
        self.preview_curves_checkbox.setChecked(False)
        self.preview_curves_checkbox.setStyleSheet("padding-left: 20px;")
        self.preview_curves_checkbox.setToolTip("Applies curves to preview thumbnails for better comparison")
        options_layout.addWidget(self.preview_curves_checkbox)
        
        # Preview scale slider
        scale_layout = QHBoxLayout()
        scale_layout.addSpacing(20)
        scale_layout.addWidget(QLabel("Preview Scale:"))
        self.preview_scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.preview_scale_slider.setMinimum(50)
        self.preview_scale_slider.setMaximum(150)
        self.preview_scale_slider.setValue(100)
        self.preview_scale_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.preview_scale_slider.setTickInterval(25)
        self.preview_scale_slider.valueChanged.connect(self.update_preview_scale_label)
        scale_layout.addWidget(self.preview_scale_slider)
        self.preview_scale_label = QLabel("100%")
        self.preview_scale_label.setMinimumWidth(45)
        scale_layout.addWidget(self.preview_scale_label)
        options_layout.addLayout(scale_layout)
        
        options_layout.addSpacing(5)
        
        # --- Channel Balance Subsection ---
        balance_subsection = QLabel("Channel Balance")
        balance_subsection.setStyleSheet(
            "font-size: 10pt; "
            "font-weight: bold; "
            "color: #e0e0e0; "
            "padding: 5px 0px 2px 5px;"
        )
        options_layout.addWidget(balance_subsection)
        
        balance_note = QLabel("<i>Multipliers applied to palette channel assignments (works with all stretch modes)</i>")
        balance_note.setStyleSheet("color: #95a5a6; font-size: 8pt; padding-left: 20px;")
        options_layout.addWidget(balance_note)
        
        # Checkbox and button for blend strength
        blend_layout = QHBoxLayout()
        blend_layout.addSpacing(20)
        self.blend_strength_checkbox = QCheckBox("Adjust Blend Strength")
        self.blend_strength_checkbox.setChecked(False)
        self.blend_strength_checkbox.stateChanged.connect(self.on_blend_strength_changed)
        blend_layout.addWidget(self.blend_strength_checkbox)
        
        self.blend_strength_btn = QPushButton("Adjust...")
        self.blend_strength_btn.setEnabled(False)
        self.blend_strength_btn.clicked.connect(self.open_blend_strength_dialog)
        self.blend_strength_btn.setMaximumWidth(100)
        blend_layout.addWidget(self.blend_strength_btn)
        blend_layout.addStretch()
        options_layout.addLayout(blend_layout)
        
        # SCNR Remove Purple Stars checkbox
        scnr_layout = QHBoxLayout()
        scnr_layout.addSpacing(20)
        self.scnr_checkbox = QCheckBox("Remove Purple Stars (SCNR)")
        self.scnr_checkbox.setChecked(False)
        self.scnr_checkbox.setToolTip(
            "Removes magenta/purple cast from stars in the final output.\n"
            "Applied after palette processing, before sending to Siril.\n"
            "Uses SCNR: Inverts → removes green → inverts back."
        )
        scnr_layout.addWidget(self.scnr_checkbox)
        scnr_layout.addStretch()
        options_layout.addLayout(scnr_layout)
        
        # Warning note about SCNR
        scnr_note = QLabel("<i>⚠️ May reduce color in purple-heavy palettes (SOH, SHO variants)</i>")
        scnr_note.setStyleSheet("color: #e67e22; font-size: 8pt; padding-left: 40px;")
        options_layout.addWidget(scnr_note)
        
        options_layout.addSpacing(8)
        
        options_layout.addSpacing(8)
        
        # Action buttons (in options tab)
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)
        
        self.preview_btn = QPushButton('🔍 Show Palette Previews')
        self.preview_btn.setStyleSheet(
            "QPushButton {"
            "  background-color: #3498db; "
            "  color: white; "
            "  font-weight: bold; "
            "  font-size: 11pt; "
            "  padding: 12px; "
            "  border-radius: 5px;"
            "}"
            "QPushButton:hover {"
            "  background-color: #2980b9;"
            "}"
            "QPushButton:pressed {"
            "  background-color: #21618c;"
            "}"
        )
        self.preview_btn.clicked.connect(self.show_previews)
        button_layout.addWidget(self.preview_btn)
        
        self.run_btn = QPushButton('▶ Generate Selected Palette')
        self.run_btn.setStyleSheet(
            "QPushButton {"
            "  background-color: #27ae60; "
            "  color: white; "
            "  font-weight: bold; "
            "  font-size: 11pt; "
            "  padding: 12px; "
            "  border-radius: 5px;"
            "}"
            "QPushButton:hover {"
            "  background-color: #229954;"
            "}"
            "QPushButton:pressed {"
            "  background-color: #1e8449;"
            "}"
        )
        self.run_btn.clicked.connect(self.run_process)
        button_layout.addWidget(self.run_btn)
        
        options_layout.addLayout(button_layout)
        options_layout.addStretch()  # Push content to top
        
        # ==================== TAB 3: CUSTOM PALETTE ====================
        custom_tab = QWidget()
        custom_tab.setStyleSheet("background-color: #3c3c3c; color: #e0e0e0;")
        custom_layout = QVBoxLayout(custom_tab)
        custom_layout.setContentsMargins(10, 10, 10, 10)
        custom_layout.setSpacing(8)
        
        # --- Channel Assignment Subsection ---
        assignment_subsection = QLabel("Channel Assignment")
        assignment_subsection.setStyleSheet(
            "font-size: 10pt; font-weight: bold; color: #e0e0e0; padding: 5px 0px 2px 5px;"
        )
        custom_layout.addWidget(assignment_subsection)
        
        assignment_note = QLabel("<i>Assign narrowband channels to RGB output</i>")
        assignment_note.setStyleSheet("color: #95a5a6; font-size: 8pt; padding-left: 20px;")
        custom_layout.addWidget(assignment_note)
        
        # Create assignment UI - each channel gets ONE color
        assignment_grid = QGridLayout()
        assignment_grid.setHorizontalSpacing(15)
        assignment_grid.setVerticalSpacing(10)
        
        # Store color values as RGB tuples (0-255) and recent custom colors
        self.custom_recent_colors = []  # Track last 8 custom colors used
        
        # Default assignments (match first row of presets)
        self.custom_ha_color = (255, 0, 0)      # Red
        self.custom_oiii_color = (0, 255, 0)    # Green (first preset)
        self.custom_sii_color = (0, 0, 255)     # Blue
        
        # Preset colors (exact hex values as specified)
        self.preset_colors = [
            (255, 0, 0),     # #FF0000 Red
            (0, 255, 0),     # #00FF00 Green
            (0, 0, 255),     # #0000FF Blue
            (255, 255, 0),   # #FFFF00 Yellow
            (255, 0, 255),   # #FF00FF Magenta
            (0, 255, 255),   # #00FFFF Cyan
        ]
        
        self.preset_colors_dark = [
            (127, 0, 0),     # #7F0000 Darker Red
            (0, 127, 0),     # #007F00 Darker Green
            (0, 0, 127),     # #00007F Darker Blue
            (127, 127, 0),   # #7F7F00 Darker Yellow
            (127, 0, 127),   # #7F007F Darker Magenta
            (0, 127, 127),   # #007F7F Darker Cyan
        ]
        
        # Natural element colors (exact hex values as specified)
        self.element_colors = {
            'H-alpha': (187, 0, 0),      # #BB0000
            'H-beta': (0, 154, 255),     # #009AFF
            'O III': (0, 212, 195),      # #00D4C3
            'S II': (115, 0, 0),         # #730000
            'N II': (177, 0, 0),         # #B10000
            'Ca II': (0, 0, 57),         # #000039
        }
        
        # Ha row
        row = 0
        assignment_grid.addWidget(QLabel("Ha:"), row, 0)
        self.custom_ha_btn, ha_widgets = self.create_color_selector_row('ha', self.custom_ha_color)
        for i, widget in enumerate(ha_widgets):
            assignment_grid.addWidget(widget, row, i + 1)
        
        # OIII row (combined with OIII #2 if present)
        row += 1
        assignment_grid.addWidget(QLabel("OIII:"), row, 0)
        self.custom_oiii_btn, oiii_widgets = self.create_color_selector_row('oiii', self.custom_oiii_color)
        for i, widget in enumerate(oiii_widgets):
            assignment_grid.addWidget(widget, row, i + 1)
        
        # SII row
        row += 1
        assignment_grid.addWidget(QLabel("SII:"), row, 0)
        self.custom_sii_btn, sii_widgets = self.create_color_selector_row('sii', self.custom_sii_color)
        for i, widget in enumerate(sii_widgets):
            assignment_grid.addWidget(widget, row, i + 1)
        
        assignment_container = QWidget()
        assignment_container.setLayout(assignment_grid)
        assignment_container.setStyleSheet("padding-left: 20px;")
        custom_layout.addWidget(assignment_container)
        
        custom_layout.addSpacing(10)
        
        # Copy stretch controls from options tab
        custom_stretch_subsection = QLabel("Statistical Stretch")
        custom_stretch_subsection.setStyleSheet(
            "font-size: 10pt; font-weight: bold; color: #e0e0e0; padding: 5px 0px 2px 5px;"
        )
        custom_layout.addWidget(custom_stretch_subsection)
        
        # Stretch mode dropdown (same as options tab)
        stretch_mode_layout = QHBoxLayout()
        stretch_mode_layout.addSpacing(20)
        stretch_mode_layout.addWidget(QLabel("Stretch Mode:"))
        self.custom_stretch_mode = QComboBox()
        self.custom_stretch_mode.addItems(["No Stretch (Linear)", "Linked Stretch", "Unlinked Stretch"])
        self.custom_stretch_mode.setCurrentIndex(2)
        self.custom_stretch_mode.currentIndexChanged.connect(lambda: self.update_custom_preview())
        stretch_mode_layout.addWidget(self.custom_stretch_mode)
        stretch_mode_layout.addStretch()
        custom_layout.addLayout(stretch_mode_layout)
        
        # Stretch strength slider
        strength_layout = QHBoxLayout()
        strength_layout.addSpacing(20)
        strength_layout.addWidget(QLabel("Target Median:"))
        self.custom_stretch_slider = QSlider(Qt.Orientation.Horizontal)
        self.custom_stretch_slider.setMinimum(1)
        self.custom_stretch_slider.setMaximum(50)
        self.custom_stretch_slider.setValue(15)
        self.custom_stretch_slider.setMaximumWidth(200)
        strength_layout.addWidget(self.custom_stretch_slider)
        self.custom_stretch_label = QLabel("0.15")
        self.custom_stretch_label.setMinimumWidth(40)
        self.custom_stretch_slider.valueChanged.connect(lambda v: self.custom_stretch_label.setText(f"{v/100:.2f}"))
        self.custom_stretch_slider.valueChanged.connect(lambda: self.update_custom_preview())
        strength_layout.addWidget(self.custom_stretch_label)
        strength_layout.addStretch()
        custom_layout.addLayout(strength_layout)
        
        # Curves checkbox
        custom_curves_layout = QHBoxLayout()
        custom_curves_layout.addSpacing(20)
        self.custom_curves_checkbox = QCheckBox("Apply curves adjustment")
        self.custom_curves_checkbox.stateChanged.connect(lambda: self.update_custom_preview())
        custom_curves_layout.addWidget(self.custom_curves_checkbox)
        custom_curves_layout.addStretch()
        custom_layout.addLayout(custom_curves_layout)
        
        custom_layout.addSpacing(10)
        
        # Action buttons
        custom_button_layout = QHBoxLayout()
        custom_button_layout.setSpacing(15)
        
        self.custom_preview_btn = QPushButton('🔍 Show Preview')
        self.custom_preview_btn.setMinimumHeight(40)
        self.custom_preview_btn.setStyleSheet("background-color: #3498db; color: white; font-weight: bold;")
        self.custom_preview_btn.clicked.connect(self.show_custom_preview)
        custom_button_layout.addWidget(self.custom_preview_btn)
        
        self.custom_generate_btn = QPushButton('✨ Generate Custom Palette')
        self.custom_generate_btn.setMinimumHeight(40)
        self.custom_generate_btn.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold;")
        self.custom_generate_btn.clicked.connect(self.run_custom_process)
        custom_button_layout.addWidget(self.custom_generate_btn)
        
        custom_layout.addLayout(custom_button_layout)
        custom_layout.addStretch()
        
        # Add all tabs to tab widget
        tabs.addTab(file_tab, "1. Load Files")
        tabs.addTab(options_tab, "2. Processing Options")
        tabs.addTab(custom_tab, "3. Custom Palette")
        
        # Add tabs to main layout (full width now)
        main_layout.addWidget(tabs)
        
        # Status label (hidden but kept for code compatibility)
        self.status_label = QLabel("")
        self.status_label.hide()
        
        main_layout.addSpacing(10)
        
        # ==================== FOOTER ====================
        credit_label = QLabel(
            "<i>Perfect Palette Picker and Statistical Stretch code created by Seti Astro "
            "<a href='https://www.setiastro.com' style='color: #3498db;'>www.setiastro.com</a></i><br>"
            "<i>Script created by Jaydn Hubalek</i><br>"
            "<i>Contactable via Facebook 'Siril users group' or Discord 'drwaffles90#0145'</i>"
        )
        credit_label.setOpenExternalLinks(True)
        credit_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        credit_label.setStyleSheet(
            "color: #95a5a6; "
            "font-size: 8pt; "
            "padding: 8px; "
            "background-color: #ecf0f1; "
            "border-radius: 3px;"
        )
        main_layout.addWidget(credit_label)
        
        self.setLayout(main_layout)
        
        # Trigger initial stretch mode check to disable unstretch if needed
        self.on_stretch_mode_changed(self.stretch_mode.currentIndex())
    
    def update_stretch_label(self, value):
        """Update stretch value label"""
        self.stretch_value_label.setText(f"{value/100:.2f}")
    
    def update_preview_scale_label(self, value):
        """Update preview scale label and store value"""
        self.preview_scale_label.setText(f"{value}%")
        self.preview_scale = value
        # Don't save on every slider movement - only on release or when used
    
    def on_stretch_mode_changed(self, index):
        """Handle stretch mode changes - enable/disable related options"""
        if index == 0:  # No stretch selected
            self.unstretch_checkbox.setChecked(False)
            self.unstretch_checkbox.setEnabled(False)
            self.unstretch_strength.setEnabled(False)
        else:  # Linked or Unlinked stretch
            self.unstretch_checkbox.setEnabled(True)
    
    def on_unstretch_changed(self, state):
        """Enable/disable unstretch strength dropdown based on checkbox"""
        self.unstretch_strength.setEnabled(state == Qt.CheckState.Checked.value)
    
    def on_blend_strength_changed(self, state):
        """Enable/disable blend strength button based on checkbox"""
        self.blend_strength_btn.setEnabled(state == Qt.CheckState.Checked.value)
    
    def open_blend_strength_dialog(self):
        """Open dialog to adjust blend strength"""
        dialog = BlendStrengthDialog(
            self.ha_balance_value,
            self.oiii_balance_value,
            self.sii_balance_value,
            self
        )
        if dialog.exec():
            self.ha_balance_value = dialog.ha_slider.value()
            self.oiii_balance_value = dialog.oiii_slider.value()
            self.sii_balance_value = dialog.sii_slider.value()
            # Values updated - will be saved with other settings
    
    def restore_defaults(self):
        """Restore all settings to defaults and clear file paths"""
        # Clear file paths
        self.ha_file = None
        self.oiii_file = None
        self.oiii2_file = None
        self.sii_file = None
        
        self.ha_label.setText("No file selected")
        self.oiii_label.setText("No file selected")
        self.oiii2_label.setText("No file selected (will combine if provided)")
        self.sii_label.setText("No file selected")
        
        # Reset UI to defaults
        self.stretch_mode.setCurrentIndex(2)  # Default to Unlinked
        self.stretch_slider.setValue(15)
        self.ha_balance_value = 100
        self.oiii_balance_value = 100
        self.sii_balance_value = 100
        self.oiii_combine_mode.setCurrentIndex(0)
        self.palette_pick.setCurrentIndex(0)
        self.curves_checkbox.setChecked(False)
        self.preview_linked_checkbox.setChecked(False)
        self.preview_curves_checkbox.setChecked(False)
        self.unstretch_checkbox.setChecked(False)
        self.unstretch_strength.setCurrentIndex(1)  # Moderate
        self.save_result_checkbox.setChecked(False)
        self.blend_strength_checkbox.setChecked(False)
        self.scnr_checkbox.setChecked(False)
        self.synth_lum_checkbox.setChecked(False)
        self.synth_lum_strength.setValue(100)
        self.preview_scale = 100
        self.preview_scale_slider.setValue(100)
        self.ha_gain_slider.setValue(100)
        self.oiii_gain_slider.setValue(100)
        self.sii_gain_slider.setValue(100)
        
        # Delete saved settings file
        try:
            if CONFIG_FILE.exists():
                CONFIG_FILE.unlink()
                print(f"Deleted settings file: {CONFIG_FILE}")
        except Exception as e:
            print(f"Could not delete settings file: {e}")
        
        self.status_label.setText("Settings restored to defaults and file paths cleared")
    
    def select_file(self, channel_type):
        """Select FITS file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"Select {channel_type.upper()} FITS file", "",
            "FITS Files (*.fit *.fits *.fts *.fit.fz);;All Files (*.*)"
        )
        
        if file_path:
            if channel_type == 'ha':
                # If Ha file changes, clear OIII and SII (different target)
                if self.ha_file and self.ha_file != file_path:
                    self.clear_file('oiii')
                    self.clear_file('sii')
                    #print("Ha file changed - cleared OIII and SII selections")
                
                self.ha_file = file_path
                self.ha_label.setText(os.path.basename(file_path))
            elif channel_type == 'oiii':
                self.oiii_file = file_path
                self.oiii_label.setText(os.path.basename(file_path))
            elif channel_type == 'oiii2':
                self.oiii2_file = file_path
                self.oiii2_label.setText(os.path.basename(file_path))
            elif channel_type == 'sii':
                self.sii_file = file_path
                self.sii_label.setText(os.path.basename(file_path))
            
            # Update balance button state
            self.update_balance_button_state()
    
    def update_balance_button_state(self):
        """Enable balance/preview buttons when Ha, at least one OIII channel, and SII are loaded"""
        oiii_available = (self.oiii_file is not None or self.oiii2_file is not None)
        all_loaded = (self.ha_file is not None and oiii_available and self.sii_file is not None)
        self.balance_button.setEnabled(all_loaded)
        self.file_preview_btn.setEnabled(all_loaded)
        self.crop_button.setEnabled(all_loaded)
    
    def clear_file(self, channel_type):
        """Clear selected file"""
        if channel_type == 'ha':
            self.ha_file = None
            self.ha_label.setText("No file selected")
        elif channel_type == 'oiii':
            self.oiii_file = None
            self.oiii_label.setText("No file selected")
        elif channel_type == 'oiii2':
            self.oiii2_file = None
            self.oiii2_label.setText("No file selected (will combine if provided)")
        elif channel_type == 'sii':
            self.sii_file = None
            self.sii_label.setText("No file selected")
        
        # Update balance button state
        self.update_balance_button_state()
    
    def load_fits_data(self, file_path, channel_name):
        """Load FITS file and store header - handles compressed FITS"""
        try:
            #print(f"Loading {channel_name} from: {os.path.basename(file_path)}")
            
            with fits.open(file_path) as hdul:
                # Handle compressed FITS - data might be in extension 1
                if hdul[0].data is None and len(hdul) > 1:
                    data = hdul[1].data.astype(np.float64)
                    header = hdul[1].header
                else:
                    data = hdul[0].data
                    header = hdul[0].header
                    
                    # Check if data is None
                    if data is None:
                        #print(f"  ERROR: No image data found in {channel_name}")
                        return None
                    
                    data = data.astype(np.float64)
                
                if channel_name == "Ha" and self.fits_header is None:
                    self.fits_header = header.copy()
                    #print(f"  Stored FITS header from Ha file")
                
                if data.ndim == 3:
                    data = data[0, :, :]
                
                data_median = np.median(data[data > 0])
                #print(f"  {channel_name}: shape={data.shape}, median={data_median:.2f}")
                
                return data
        
        except Exception as e:
            print(f"Error loading {channel_name}: {e}")
            traceback.print_exc()
            return None
    
    def save_channel_data(self, data, channel_name):
        """Save channel as hidden FITS file"""
        try:
            output_path = f".{channel_name}.fit"
            data_16bit = np.clip(data * 65535.0, 0, 65535).astype(np.uint16)
            
            hdu = fits.PrimaryHDU(data_16bit)
            if self.fits_header is not None:
                hdu.header = self.fits_header.copy()
                hdu.header['NAXIS1'] = data_16bit.shape[1]
                hdu.header['NAXIS2'] = data_16bit.shape[0]
                hdu.header['BITPIX'] = 16
            
            hdu.writeto(output_path, overwrite=True)
            self.temp_files.append(output_path)
            return channel_name
        except Exception as e:
            print(f"Error saving channel {channel_name}: {e}")
            traceback.print_exc()
            return None
    
    def cleanup_temp_files(self):
        """Remove temporary files"""
        #print("\nCleaning up temporary files...")
        for file_path in self.temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"  Removed: {file_path}")
            except Exception as e:
                print(f"  Could not remove {file_path}: {e}")
        self.temp_files = []
    
    def check_alignment(self, ha_data, oiii_data, sii_data):
        """
        Check if images are aligned by comparing star positions.
        Returns (aligned, offset) where aligned is bool and offset is max pixel shift detected.
        """
        if not ASTROALIGN_AVAILABLE:
            return True, 0  # Skip check if astroalign not available
        
        try:
            #print("Checking image alignment...")
            
            # Use a small sample for speed (center 512x512 region)
            def get_sample(data):
                h, w = data.shape
                cy, cx = h // 2, w // 2
                size = min(512, h, w) // 2
                return data[cy-size:cy+size, cx-size:cx+size]
            
            ha_sample = get_sample(ha_data)
            oiii_sample = get_sample(oiii_data)
            sii_sample = get_sample(sii_data)
            
            # Find transformation between Ha and OIII
            try:
                transf_oiii, (source_pos, target_pos) = astroalign.find_transform(
                    oiii_sample, ha_sample
                )
                offset_oiii = np.max(np.abs(transf_oiii.translation))
            except Exception as e:
                #print(f"  Could not determine Ha-OIII alignment: {e}")
                offset_oiii = 0
            
            # Find transformation between Ha and SII
            try:
                transf_sii, (source_pos, target_pos) = astroalign.find_transform(
                    sii_sample, ha_sample
                )
                offset_sii = np.max(np.abs(transf_sii.translation))
            except Exception as e:
                #print(f"  Could not determine Ha-SII alignment: {e}")
                offset_sii = 0
            
            max_offset = max(offset_oiii, offset_sii)
            
            # Consider aligned if offset < 2 pixels
            aligned = max_offset < 2.0
            
            #print(f"  Ha-OIII offset: {offset_oiii:.2f} pixels")
            #print(f"  Ha-SII offset: {offset_sii:.2f} pixels")
            #print(f"  Alignment status: {'ALIGNED' if aligned else 'NOT ALIGNED'}")
            
            return aligned, max_offset
        
        except Exception as e:
            #print(f"Alignment check failed: {e}")
            return True, 0  # Assume aligned if check fails
    
    def process_palette(self, ha, oiii, sii, palette_name):
        """Process any palette type with channel balance scalars applied"""
        # Get channel balance scalars if enabled
        if self.blend_strength_checkbox.isChecked():
            ha_scalar = self.ha_balance_value / 100.0
            oiii_scalar = self.oiii_balance_value / 100.0
            sii_scalar = self.sii_balance_value / 100.0
            
        else:
            ha_scalar = oiii_scalar = sii_scalar = 1.0
        
        # Apply scalars to channels
        ha = ha * ha_scalar
        oiii = oiii * oiii_scalar
        sii = sii * sii_scalar
        
        if palette_name == "Foraxx":
            return self.process_foraxx_palette(ha, oiii, sii)
        elif palette_name == "Realistic1":
            return self.process_realistic1_palette(ha, oiii, sii)
        elif palette_name == "Realistic2":
            return self.process_realistic2_palette(ha, oiii, sii)
        
        mapping = {
            "SHO (Hubble)": [sii, ha, oiii],
            "HOO (Narrowband Ha-OIII)": [ha, oiii, oiii],
            "HSO": [ha, sii, oiii],
            "HOS": [ha, oiii, sii],
            "HSS": [ha, sii, sii],
            "OSS": [oiii, sii, sii],
            "OHH": [oiii, ha, ha],
            "OSH": [oiii, sii, ha],
            "OHS": [oiii, ha, sii],
            "SHH": [sii, ha, ha],
            "SOH": [sii, oiii, ha],
            "SOO": [sii, oiii, oiii],
            "SSH": [sii, sii, ha],
        }
        
        if palette_name in mapping:
            r, g, b = mapping[palette_name]
            r = np.clip(np.nan_to_num(r, nan=0.0), 0, 1)
            g = np.clip(np.nan_to_num(g, nan=0.0), 0, 1)
            b = np.clip(np.nan_to_num(b, nan=0.0), 0, 1)
            return r, g, b
        
        return sii, ha, oiii
    
    def process_foraxx_palette(self, ha, oiii, sii):
        """Process Foraxx palette"""
        try:
            if ha is not None and oiii is not None and sii is not None:
                eps = 1e-10
                oiii_safe = np.clip(oiii, eps, 1.0)
                ha_safe = np.clip(ha, eps, 1.0)
                
                temp = np.power(oiii_safe, (1 - oiii_safe))
                expr_r = (temp * sii) + ((1 - temp) * ha)
                
                temp_ha_oiii = ha_safe * oiii_safe
                temp_ha_oiii = np.clip(temp_ha_oiii, eps, 1.0)
                expr_g = (np.power(temp_ha_oiii, (1 - temp_ha_oiii)) * ha) + ((1 - np.power(temp_ha_oiii, (1 - temp_ha_oiii))) * oiii)
                
                expr_b = oiii
            elif ha is not None and oiii is not None:
                eps = 1e-10
                ha_safe = np.clip(ha, eps, 1.0)
                oiii_safe = np.clip(oiii, eps, 1.0)
                
                expr_r = ha
                temp = ha_safe * oiii_safe
                temp = np.clip(temp, eps, 1.0)
                expr_g = (np.power(temp, (1 - temp)) * ha) + ((1 - np.power(temp, (1 - temp))) * oiii)
                expr_b = oiii
            else:
                return sii, ha, oiii
            
            expr_r = np.clip(np.nan_to_num(expr_r, nan=0.0), 0, 1)
            expr_g = np.clip(np.nan_to_num(expr_g, nan=0.0), 0, 1)
            expr_b = np.clip(np.nan_to_num(expr_b, nan=0.0), 0, 1)
            
            return expr_r, expr_g, expr_b
        except Exception as e:
            print(f"Error in Foraxx processing: {e}")
            traceback.print_exc()
            return None, None, None
    
    def process_realistic1_palette(self, ha, oiii, sii):
        """Process Realistic1"""
        expr_r = (ha + sii) / 2
        expr_g = (0.3 * ha) + (0.7 * oiii)
        expr_b = (0.9 * oiii) + (0.1 * ha)
        
        return np.clip(expr_r, 0, 1), np.clip(expr_g, 0, 1), np.clip(expr_b, 0, 1)
    
    def process_realistic2_palette(self, ha, oiii, sii):
        """Process Realistic2"""
        expr_r = (0.7 * ha + 0.3 * sii)
        expr_g = (0.3 * sii + 0.7 * oiii)
        expr_b = oiii
        
        return np.clip(expr_r, 0, 1), np.clip(expr_g, 0, 1), np.clip(expr_b, 0, 1)
    
    def show_previews(self):
        """Generate and show previews"""
        try:
            if not all([self.ha_file, self.oiii_file, self.sii_file]):
                QMessageBox.warning(self, "Missing Files", 
                    "Please load all three narrowband files (Ha, OIII, SII) before generating previews.")
                return
            
            # Save settings before generating
            self.save_settings()
            
            self.status_label.setText("Loading data for previews...")
            QApplication.processEvents()
            
            # Load data (use balanced files if available)
            ha_data = self.load_fits_data(self.get_file_to_load('ha'), "Ha")
            ha_linear = ha_data.copy() if ha_data is not None else None
            oiii_data = self.load_fits_data(self.get_file_to_load('oiii'), "OIII")
            oiii_linear = oiii_data.copy() if oiii_data is not None else None
            sii_data = self.load_fits_data(self.get_file_to_load('sii'), "SII")
            sii_linear = sii_data.copy() if sii_data is not None else None

            if ha_data is None or oiii_data is None or sii_data is None:
                raise ValueError("Failed to load files")

            # Combine OIII if second file provided (skip if already baked into balanced file)
            if self.oiii2_file and not self._oiii2_already_baked():
                oiii2_data = self.load_fits_data(self.oiii2_file, "OIII #2")
                if oiii2_data is not None:
                    combine_mode = self.oiii_combine_mode.currentText()
                    if combine_mode == "Average":
                        oiii_data = (oiii_data + oiii2_data) / 2.0
                        oiii_linear = oiii_data.copy()
                    elif combine_mode == "Maximum":
                        oiii_data = np.maximum(oiii_data, oiii2_data)
                        oiii_linear = oiii_data.copy()
                    else:
                        oiii_data = (0.7 * oiii_data + 0.3 * oiii2_data)
                        oiii_linear = oiii_data.copy()
            
            # Normalize to [0,1]
            #print("Normalizing data for preview generation...")
            if ha_data.max() > 1.0:
                ha_data = ha_data / 65535.0
            if oiii_data.max() > 1.0:
                oiii_data = oiii_data / 65535.0
            if sii_data.max() > 1.0:
                sii_data = sii_data / 65535.0
            
            # Calculate scaled preview size
            scaled_preview_size = int(300 * (self.preview_scale / 100.0))
            #print(f"="*60)
            #print(f"="*60)
            
            # Create preview window
            self.preview_window = PreviewWindow(self, scale=self.preview_scale, preview_size=scaled_preview_size)
            self.preview_window.palette_selected.connect(self.on_preview_selected)
            self.preview_window.show()
            QApplication.processEvents()
            
            # Start preview thread
            #print("Starting preview generation thread...")
            
            # Get current stretch settings (CRITICAL: use same as final output)
            stretch_mode = self.stretch_mode.currentIndex()
            stretch_strength = self.stretch_slider.value() / 100.0
            apply_curves = self.curves_checkbox.isChecked()
            
            # Get preview display settings (only used if no channel stretch)
            preview_linked = self.preview_linked_checkbox.isChecked()
            preview_curves = self.preview_curves_checkbox.isChecked()
            
            #print (f"Preview settings: linked={preview_linked}, curves={preview_curves}")
            
            self.preview_thread = PreviewGeneratorThread(
                ha_data, oiii_data, sii_data, 
                self, preview_size=scaled_preview_size,
                stretch_mode=stretch_mode,
                stretch_strength=stretch_strength,
                apply_curves=apply_curves,
                preview_linked=preview_linked,
                preview_curves=preview_curves
            )
            self.preview_thread.preview_ready.connect(self.preview_window.add_preview)
            self.preview_thread.all_complete.connect(self.preview_window.generation_complete)
            self.preview_thread.start()
            
            self.status_label.setText("Preview window opened - generating thumbnails...")
            
            result = self.preview_window.exec()
            
            if result == QDialog.DialogCode.Accepted:
                self.status_label.setText(f"Palette selected from preview")
            else:
                self.status_label.setText("Preview closed without selection")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate previews: {str(e)}")
            self.status_label.setText(f"Error: {str(e)}")
    
    def on_preview_selected(self, palette_name):
        """Handle palette selection from preview"""
        for i in range(self.palette_pick.count()):
            if self.palette_pick.itemText(i) == palette_name:
                self.palette_pick.setCurrentIndex(i)
                break
        
        self.status_label.setText(f"Selected: {palette_name} - Ready to generate")
    
    def create_color_selector_row(self, channel_id, default_color):
        """Create a Siril-style color selector row with presets, custom button, and dropdown"""
        widgets = []
        
        # Current color display (larger square)
        color_btn = QPushButton()
        color_btn.setFixedSize(50, 50)
        r, g, b = default_color
        color_btn.setStyleSheet(f"background-color: rgb({r},{g},{b}); border: 3px solid #555; border-radius: 3px;")
        color_btn.setProperty('channel_id', channel_id)
        widgets.append(color_btn)
        
        # Element dropdown (create early so we can pass it to buttons)
        element_combo = QComboBox()
        element_combo.addItems(['---', 'H-alpha', 'H-beta', 'O III', 'S II', 'N II', 'Ca II'])
        element_combo.setFixedWidth(100)
        
        # Preset colors row 1 (bright)
        preset_container = QWidget()
        preset_layout = QHBoxLayout(preset_container)
        preset_layout.setSpacing(3)
        preset_layout.setContentsMargins(0, 0, 0, 0)
        
        for color in self.preset_colors:
            btn = self.create_preset_button(color, channel_id, color_btn, element_combo)
            preset_layout.addWidget(btn)
        
        widgets.append(preset_container)
        
        # Preset colors row 2 (dark)
        preset_dark_container = QWidget()
        preset_dark_layout = QHBoxLayout(preset_dark_container)
        preset_dark_layout.setSpacing(3)
        preset_dark_layout.setContentsMargins(0, 0, 0, 0)
        
        for color in self.preset_colors_dark:
            btn = self.create_preset_button(color, channel_id, color_btn, element_combo)
            preset_dark_layout.addWidget(btn)
        
        widgets.append(preset_dark_container)
        
        # Custom color button (+)
        custom_btn = QPushButton("+")
        custom_btn.setFixedSize(30, 30)
        custom_btn.setStyleSheet("background-color: #555; font-weight: bold; font-size: 14pt;")
        custom_btn.clicked.connect(lambda: self.pick_custom_color(channel_id, color_btn, element_combo))
        widgets.append(custom_btn)
        
        # Element dropdown with header
        element_label = QLabel("Spectrum")
        element_label.setStyleSheet("font-size: 8pt; color: #95a5a6;")
        element_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        widgets.append(element_label)
        
        element_combo.currentTextChanged.connect(lambda text: self.apply_element_color(text, channel_id, color_btn, element_combo))
        widgets.append(element_combo)
        
        return color_btn, widgets
    
    def create_preset_button(self, color, channel_id, display_btn, element_combo):
        """Create a small preset color button"""
        r, g, b = color
        btn = QPushButton()
        btn.setFixedSize(30, 30)
        btn.setStyleSheet(f"background-color: rgb({r},{g},{b}); border: 1px solid #333;")
        btn.clicked.connect(lambda: self.apply_color(color, channel_id, display_btn, element_combo))
        return btn
    
    def apply_color(self, color, channel_id, display_btn, element_combo, reset_dropdown=True):
        """Apply a color to a channel"""
        r, g, b = color
        setattr(self, f'custom_{channel_id}_color', color)
        display_btn.setStyleSheet(f"background-color: rgb({r},{g},{b}); border: 3px solid #555; border-radius: 3px;")
        
        # Reset dropdown to "---" so user can select same element again (but not if called FROM dropdown)
        if reset_dropdown:
            element_combo.setCurrentIndex(0)
        
        # Update live preview if open
        self.update_custom_preview()
    
    def apply_element_color(self, element_name, channel_id, display_btn, element_combo):
        """Apply natural element color from dropdown"""
        # Ignore default "---" selection
        if element_name == '---':
            return
        if element_name in self.element_colors:
            color = self.element_colors[element_name]
            # Don't reset dropdown when applying FROM the dropdown - let it show the selection
            self.apply_color(color, channel_id, display_btn, element_combo, reset_dropdown=False)
    
    def pick_custom_color(self, channel_id, display_btn, element_combo):
        """Open color picker for custom color"""
        from PyQt6.QtWidgets import QColorDialog
        from PyQt6.QtGui import QColor
        
        # Get current color
        current_color = getattr(self, f'custom_{channel_id}_color')
        qcolor = QColor(*current_color)
        
        # Open color picker
        color = QColorDialog.getColor(qcolor, self, "Select Color")
        
        if color.isValid():
            new_color = (color.red(), color.green(), color.blue())
            self.apply_color(new_color, channel_id, display_btn, element_combo)
            
            # Add to recent colors (keep last 8)
            if new_color not in self.custom_recent_colors:
                self.custom_recent_colors.insert(0, new_color)
                if len(self.custom_recent_colors) > 8:
                    self.custom_recent_colors.pop()
    
    def show_custom_preview(self):
        """Show live preview of custom palette assignment"""
        if not self.ha_file or not self.oiii_file or not self.sii_file:
            QMessageBox.warning(self, "Files Required", "Please load Ha, OIII, and SII files first.")
            return
        
        try:
            # Load data (use balanced files if available)
            ha_data = self.load_fits_data(self.get_file_to_load('ha'), "Ha")
            oiii_data = self.load_fits_data(self.get_file_to_load('oiii'), "OIII")
            sii_data = self.load_fits_data(self.get_file_to_load('sii'), "SII")
            
            # Handle second OIII if present
            if self.oiii2_file:
                oiii2_raw = self.load_fits_data(self.oiii2_file, "OIII #2")
                combine_mode = self.oiii_combine_mode.currentText()
                if combine_mode == "Average":
                    oiii_data = (oiii_data + oiii2_raw) / 2.0
                elif combine_mode == "Maximum":
                    oiii_data = np.maximum(oiii_data, oiii2_raw)
                else:  # Weighted
                    oiii_data = (0.7 * oiii_data + 0.3 * oiii2_raw)
            
            # Create preview window
            from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel
            from PyQt6.QtGui import QPixmap, QImage
            
            self.custom_preview_dialog = QDialog(self)
            self.custom_preview_dialog.setWindowTitle("Custom Palette Preview (Live)")
            self.custom_preview_dialog.resize(800, 800)
            
            layout = QVBoxLayout()
            self.custom_preview_label = QLabel()
            self.custom_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(self.custom_preview_label)
            
            self.custom_preview_dialog.setLayout(layout)
            
            # Store data for live updates
            self.custom_preview_ha = ha_data
            self.custom_preview_oiii = oiii_data
            self.custom_preview_sii = sii_data
            
            # Show dialog first so isVisible() works
            self.custom_preview_dialog.show()
            
            # Generate initial preview
            self.update_custom_preview()
            
        except Exception as e:
            print(f"Error in custom preview: {e}")
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to generate preview:\n{str(e)}")
    
    def update_custom_preview(self):
        """Update the live preview with current color assignments"""
        if not hasattr(self, 'custom_preview_dialog'):
            return
        if not self.custom_preview_dialog.isVisible():
            return
        
        try:
            # print("DEBUG: Updating custom preview...")
            
            # Get stretch settings
            stretch_mode = self.custom_stretch_mode.currentIndex()
            target_median = self.custom_stretch_slider.value() / 100.0
            apply_curves = self.custom_curves_checkbox.isChecked()
            
            # print(f"DEBUG: Stretch mode={stretch_mode}, target={target_median}, curves={apply_curves}")
            
            # Copy data (OIII is already combined if OIII#2 was present)
            ha = self.custom_preview_ha.copy()
            oiii = self.custom_preview_oiii.copy()
            sii = self.custom_preview_sii.copy()
            
            # print(f"DEBUG: Data loaded - ha shape={ha.shape}, max={ha.max()}")
            
            # Normalize
            if ha.max() > 1.0:
                ha = ha / 65535.0
            if oiii.max() > 1.0:
                oiii = oiii / 65535.0
            if sii.max() > 1.0:
                sii = sii / 65535.0
            
            # Apply stretch if needed
            if stretch_mode == 1:  # Linked
                rgb_combined = np.stack([ha, oiii, sii], axis=2)
                rgb_stretched = stretch_color_image_linked(rgb_combined, target_median, 
                                                          apply_curves=apply_curves, curves_boost=0.15)
                ha = rgb_stretched[:, :, 0]
                oiii = rgb_stretched[:, :, 1]
                sii = rgb_stretched[:, :, 2]
            elif stretch_mode == 2:  # Unlinked
                ha = stretch_mono_image(ha, target_median, apply_curves=apply_curves, curves_boost=0.15)
                oiii = stretch_mono_image(oiii, target_median, apply_curves=apply_curves, curves_boost=0.15)
                sii = stretch_mono_image(sii, target_median, apply_curves=apply_curves, curves_boost=0.15)
            
            # Apply custom color assignments
            r_channel = np.zeros_like(ha)
            g_channel = np.zeros_like(ha)
            b_channel = np.zeros_like(ha)
            
            # print(f"DEBUG: ha range: [{ha.min():.4f}, {ha.max():.4f}], median={np.median(ha):.4f}")
            # print(f"DEBUG: oiii range: [{oiii.min():.4f}, {oiii.max():.4f}], median={np.median(oiii):.4f}")
            # print(f"DEBUG: sii range: [{sii.min():.4f}, {sii.max():.4f}], median={np.median(sii):.4f}")
            # print(f"DEBUG: Colors - Ha:{self.custom_ha_color}, OIII:{self.custom_oiii_color}, SII:{self.custom_sii_color}")
            
            # Ha contribution
            ha_r, ha_g, ha_b = self.custom_ha_color
            r_channel += ha * (ha_r / 255.0)
            g_channel += ha * (ha_g / 255.0)
            b_channel += ha * (ha_b / 255.0)
            
            # OIII contribution
            oiii_r, oiii_g, oiii_b = self.custom_oiii_color
            r_channel += oiii * (oiii_r / 255.0)
            g_channel += oiii * (oiii_g / 255.0)
            b_channel += oiii * (oiii_b / 255.0)
            
            # SII contribution
            sii_r, sii_g, sii_b = self.custom_sii_color
            r_channel += sii * (sii_r / 255.0)
            g_channel += sii * (sii_g / 255.0)
            b_channel += sii * (sii_b / 255.0)
            
            # print(f"DEBUG: Final RGB ranges - R:[{r_channel.min():.4f}, {r_channel.max():.4f}], G:[{g_channel.min():.4f}, {g_channel.max():.4f}], B:[{b_channel.min():.4f}, {b_channel.max():.4f}]")
            
            # Clip to valid range
            rgb = np.stack([r_channel, g_channel, b_channel], axis=2)
            rgb = np.clip(rgb, 0, 1)
            
            # Downsample for preview
            from scipy import ndimage
            scale = min(700 / rgb.shape[0], 700 / rgb.shape[1])
            if scale < 1.0:
                rgb = ndimage.zoom(rgb, (scale, scale, 1), order=1)
            
            # Convert to QPixmap
            rgb_uint8 = (rgb * 255).astype(np.uint8)
            height, width = rgb_uint8.shape[:2]
            qimage = QImage(rgb_uint8.data, width, height, width * 3, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            
            self.custom_preview_label.setPixmap(pixmap)
            
        except Exception as e:
            print(f"Error updating preview: {e}")
            traceback.print_exc()
    
    def get_file_to_load(self, channel):
        """
        Return the file path to use for a channel.
        Priority: balanced temp file (BP-shifted + gain) > slider gain on original > original.
        """
        if channel == 'ha':
            if hasattr(self, '_balanced_ha_path') and self._balanced_ha_path and os.path.exists(self._balanced_ha_path):
                gain = self.ha_gain_slider.value() / 100.0
                if gain != 1.0:
                    return self._apply_gain_to_file(self._balanced_ha_path, gain, 'ha')
                return self._balanced_ha_path
            gain = self.ha_gain_slider.value() / 100.0
            return self._apply_gain_to_file(self.ha_file, gain, 'ha') if gain != 1.0 and self.ha_file else self.ha_file
        elif channel == 'oiii':
            if hasattr(self, '_balanced_oiii_path') and self._balanced_oiii_path and os.path.exists(self._balanced_oiii_path):
                gain = self.oiii_gain_slider.value() / 100.0
                if gain != 1.0:
                    return self._apply_gain_to_file(self._balanced_oiii_path, gain, 'oiii')
                return self._balanced_oiii_path
            gain = self.oiii_gain_slider.value() / 100.0
            return self._apply_gain_to_file_oiii(gain) if gain != 1.0 and self.oiii_file else self.oiii_file
        elif channel == 'sii':
            if hasattr(self, '_balanced_sii_path') and self._balanced_sii_path and os.path.exists(self._balanced_sii_path):
                gain = self.sii_gain_slider.value() / 100.0
                if gain != 1.0:
                    return self._apply_gain_to_file(self._balanced_sii_path, gain, 'sii')
                return self._balanced_sii_path
            gain = self.sii_gain_slider.value() / 100.0
            return self._apply_gain_to_file(self.sii_file, gain, 'sii') if gain != 1.0 and self.sii_file else self.sii_file
        elif channel == 'oiii2':
            return self.oiii2_file
        return None

    def _apply_gain_to_file(self, source_path, gain, channel_name):
        """Apply a scalar gain to a FITS file and save a temp copy; return its path"""
        import tempfile
        try:
            data = self.load_fits_data(source_path, channel_name)
            if data is None:
                return source_path
            normalised = data / 65535.0 if data.max() > 1.0 else data.astype(np.float32)
            scaled = np.clip(normalised * gain, 0, 1).astype(np.float32)
            temp_path = os.path.join(tempfile.gettempdir(), f"npp_{channel_name}_gain.fit")
            fits.writeto(temp_path, scaled, self.fits_header, overwrite=True)
            return temp_path
        except Exception as e:
            print(f"[Balance] Could not apply gain to {channel_name}: {e}")
            return source_path
    
    def _apply_gain_to_file_oiii(self, gain):
        """Combine OIII+OIII2 (respecting combine mode) then apply gain; return temp file path"""
        import tempfile
        try:
            data = self.load_fits_data(self.oiii_file, "oiii")
            if data is None:
                return self.oiii_file
            normalised = data / 65535.0 if data.max() > 1.0 else data.astype(np.float32)
            # Combine OIII2 if present
            if self.oiii2_file:
                data2 = self.load_fits_data(self.oiii2_file, "oiii2")
                if data2 is not None:
                    n2 = data2 / 65535.0 if data2.max() > 1.0 else data2.astype(np.float32)
                    mode = self.oiii_combine_mode.currentText()
                    if mode == "Average":
                        normalised = (normalised + n2) / 2.0
                    elif mode == "Maximum":
                        normalised = np.maximum(normalised, n2)
                    else:
                        normalised = (0.7 * normalised + 0.3 * n2)
            scaled = np.clip(normalised * gain, 0, 1).astype(np.float32)
            temp_path = os.path.join(tempfile.gettempdir(), "npp_oiii_gain.fit")
            fits.writeto(temp_path, scaled, self.fits_header, overwrite=True)
            return temp_path
        except Exception as e:
            print(f"[Balance] Could not apply gain to oiii: {e}")
            return self.oiii_file

    def auto_balance_channels(self):
        """
        Balance channels in two steps:
          1. Black point shift: subtract each channel's own sky background (computed
             by _compute_blackpoint_sigma, the same function the stretch uses).
             This aligns all three channels to a common zero background — the root
             cause of linked vs unlinked looking different.
          2. Signal median equalisation: after the BP shift, scale each channel to
             match the BRIGHTEST channel's median. This only boosts dimmer channels,
             never reduces signal.

        The result is written directly to temp files (not via sliders) because the
        BP shift is an additive operation the sliders can't represent. Sliders are
        set to 1.0x and the balanced files are stored for get_file_to_load() to use.
        """
        import tempfile
        print("[Balance] Computing black point shift + median equalisation...")

        ha_data   = self.load_fits_data(self.ha_file,   "Ha")
        oiii_data = self.load_fits_data(self.oiii_file, "OIII")
        sii_data  = self.load_fits_data(self.sii_file,  "SII")

        if ha_data is None or oiii_data is None or sii_data is None:
            QMessageBox.warning(self, "Error", "Failed to load channel data")
            return

        def _norm(d):
            return d / 65535.0 if d.max() > 1.0 else d.astype(np.float32)

        # Handle OIII #2 — keep separate for live preview combine mode, combine for BP measurement
        oiii2_n = None
        if self.oiii2_file:
            oiii2_data = self.load_fits_data(self.oiii2_file, "OIII #2")
            if oiii2_data is not None:
                oiii2_n = _norm(oiii2_data)
                combine_mode = self.oiii_combine_mode.currentText()
                if combine_mode == "Average":
                    oiii_data = (oiii_data + oiii2_data) / 2.0
                elif combine_mode == "Maximum":
                    oiii_data = np.maximum(oiii_data, oiii2_data)
                else:
                    oiii_data = (0.7 * oiii_data + 0.3 * oiii2_data)

        ha_n   = _norm(ha_data)
        oiii_n = _norm(oiii_data)   # combined version for BP measurement + temp file
        sii_n  = _norm(sii_data)

        # Step 1: compute and subtract each channel's own black point
        # Use downsampled data for speed — statistics are identical
        ha_bp,   _ = _compute_blackpoint_sigma(ha_n.ravel()[::4],   sigma=5.0)
        oiii_bp, _ = _compute_blackpoint_sigma(oiii_n.ravel()[::4], sigma=5.0)
        sii_bp,  _ = _compute_blackpoint_sigma(sii_n.ravel()[::4],  sigma=5.0)

        print(f"[Balance] Black points — Ha: {ha_bp:.6f}, OIII: {oiii_bp:.6f}, SII: {sii_bp:.6f}")

        # Shift: subtract BP then rescale to [0,1] range exactly as the stretch does
        def _bp_shift(data, bp):
            denom = max(1.0 - bp, 1e-12)
            return np.clip((data - bp) / denom, 0.0, 1.0).astype(np.float32)

        ha_shifted   = _bp_shift(ha_n,   ha_bp)
        oiii_shifted = _bp_shift(oiii_n, oiii_bp)
        sii_shifted  = _bp_shift(sii_n,  sii_bp)

        # Step 2: equalise signal medians on the BP-shifted data
        def _signal_median(data):
            flat = data.ravel()[::4]
            sig = flat[flat > 0]
            return float(np.median(sig)) if sig.size > 100 else float(np.median(flat))

        ha_med   = _signal_median(ha_shifted)
        oiii_med = _signal_median(oiii_shifted)
        sii_med  = _signal_median(sii_shifted)

        print(f"[Balance] Signal medians (post-BP) — Ha: {ha_med:.6f}, OIII: {oiii_med:.6f}, SII: {sii_med:.6f}")

        # Match to brightest channel - only boost, never reduce
        target = max(ha_med, oiii_med, sii_med)

        ha_gain   = target / ha_med   if ha_med   > 1e-9 else 1.0
        oiii_gain = target / oiii_med if oiii_med > 1e-9 else 1.0
        sii_gain  = target / sii_med  if sii_med  > 1e-9 else 1.0

        print(f"[Balance] Gains — Ha: {ha_gain:.3f}x, OIII: {oiii_gain:.3f}x, SII: {sii_gain:.3f}x")

        # Write BP-shifted (gain=1.0x) data to temp files — gain lives in sliders
        ha_bp_only   = np.clip(ha_shifted,   0, 1).astype(np.float32)
        oiii_bp_only = np.clip(oiii_shifted, 0, 1).astype(np.float32)
        sii_bp_only  = np.clip(sii_shifted,  0, 1).astype(np.float32)

        temp_dir = tempfile.gettempdir()
        self._balanced_ha_path   = os.path.join(temp_dir, "npp_ha_balanced.fit")
        self._balanced_oiii_path = os.path.join(temp_dir, "npp_oiii_balanced.fit")
        self._balanced_sii_path  = os.path.join(temp_dir, "npp_sii_balanced.fit")

        fits.writeto(self._balanced_ha_path,   ha_bp_only,   self.fits_header, overwrite=True)
        fits.writeto(self._balanced_oiii_path, oiii_bp_only, self.fits_header, overwrite=True)
        fits.writeto(self._balanced_sii_path,  sii_bp_only,  self.fits_header, overwrite=True)

        # Flag that OIII2 is already baked into the balanced OIII file
        self._balance_oiii2_baked = True

        # Update live preview raw buffers — keep OIII and OIII2 separate so
        # combine mode dropdown continues to work live in the preview
        if hasattr(self, '_live_preview_label') and self._live_preview_label is not None:
            self._live_ha_raw  = ha_bp_only.copy()
            self._live_sii_raw = sii_bp_only.copy()
            if oiii2_n is not None:
                # BP-shift each independently so combine mode can vary live
                oiii_only_n = _norm(self.load_fits_data(self.oiii_file, "OIII"))
                oiii_only_bp, _ = _compute_blackpoint_sigma(oiii_only_n.ravel()[::4], sigma=5.0)
                self._live_oiii_base  = _bp_shift(oiii_only_n, oiii_only_bp)
                oiii2_bp, _ = _compute_blackpoint_sigma(oiii2_n.ravel()[::4], sigma=5.0)
                self._live_oiii2_base = _bp_shift(oiii2_n, oiii2_bp)
            else:
                self._live_oiii_base  = oiii_bp_only.copy()
                self._live_oiii2_base = None

        # Push gains to sliders so user can see and fine-tune them
        def _to_slider(g):
            return max(10, min(500, int(round(g * 100))))

        for slider in (self.ha_gain_slider, self.oiii_gain_slider, self.sii_gain_slider):
            slider.blockSignals(True)
        self.ha_gain_slider.setValue(_to_slider(ha_gain))
        self.oiii_gain_slider.setValue(_to_slider(oiii_gain))
        self.sii_gain_slider.setValue(_to_slider(sii_gain))
        for slider in (self.ha_gain_slider, self.oiii_gain_slider, self.sii_gain_slider):
            slider.blockSignals(False)
        self._sync_gain_labels()

        status_text = (f"✓ BP-shifted + balanced (Ha BP:{ha_bp:.4f}, OIII BP:{oiii_bp:.4f}, SII BP:{sii_bp:.4f})")
        if self.oiii2_file:
            status_text += f" [OIII: {self.oiii_combine_mode.currentText()}]"
        self.balance_status.setText(status_text)
        self.clear_balance_button.setEnabled(True)

        self._refresh_live_preview()

        print("[Balance] ✓ Done — BP shift + gain written to temp files")
        msg = (f"Channels balanced:\n\n"
               f"  Black points subtracted:\n"
               f"    Ha:   {ha_bp:.5f}\n"
               f"    OIII: {oiii_bp:.5f}\n"
               f"    SII:  {sii_bp:.5f}\n\n"
               f"  Signal gains applied:\n"
               f"    Ha:   {ha_gain:.2f}x\n"
               f"    OIII: {oiii_gain:.2f}x\n"
               f"    SII:  {sii_gain:.2f}x\n\n"
               f"Fine-tune with the sliders if needed.")
        if self.oiii2_file:
            msg += f"\n\n✓ OIII+OIII2 combined using {self.oiii_combine_mode.currentText()}"
        QMessageBox.information(self, "Balance Complete", msg)
    
    def clear_balance(self):
        """Reset gain sliders to 1.0x and delete all balance temp files"""
        import tempfile

        print("[Balance] Clearing balance...")

        temp_dir = tempfile.gettempdir()
        for name in ("npp_ha_balanced.fit", "npp_oiii_balanced.fit", "npp_sii_balanced.fit",
                     "npp_ha_gain.fit", "npp_oiii_gain.fit", "npp_sii_gain.fit"):
            p = os.path.join(temp_dir, name)
            if os.path.exists(p):
                os.remove(p)
                print(f"[Balance] Deleted: {p}")

        self._balanced_ha_path   = None
        self._balanced_oiii_path = None
        self._balanced_sii_path  = None
        self._balance_oiii2_baked = False

        for slider in (self.ha_gain_slider, self.oiii_gain_slider, self.sii_gain_slider):
            slider.blockSignals(True)
            slider.setValue(100)
            slider.blockSignals(False)
        self._sync_gain_labels()

        self.balance_gains = {'ha': 1.0, 'oiii': 1.0, 'sii': 1.0}
        self.balance_status.setText("")
        self.clear_balance_button.setEnabled(False)

        print("[Balance] ✓ Balance cleared")

        # Reload original raw data into preview buffers if preview is open
        if hasattr(self, '_live_preview_label') and self._live_preview_label is not None:
            def _load_raw(f):
                d = self.load_fits_data(f, "")
                if d is None: return None
                return (d / 65535.0 if d.max() > 1.0 else d.astype(np.float32)).copy()
            self._live_ha_raw    = _load_raw(self.ha_file)
            self._live_oiii_base = _load_raw(self.oiii_file)
            self._live_oiii2_base = _load_raw(self.oiii2_file) if self.oiii2_file else None
            self._live_sii_raw   = _load_raw(self.sii_file)

        self._refresh_live_preview()
        QMessageBox.information(self, "Balance Cleared", "Reverted to original unbalanced channels")
    

    
    
    def show_crop_tool(self):
        """Show crop tool for batch cropping all loaded images"""
        
        # Build file dict for crop tool
        file_dict = {}
        if self.ha_file:
            file_dict['ha'] = self.ha_file
        if self.oiii_file:
            file_dict['oiii'] = self.oiii_file
        if self.oiii2_file:
            file_dict['oiii2'] = self.oiii2_file
        if self.sii_file:
            file_dict['sii'] = self.sii_file
        
        if not file_dict:
            QMessageBox.warning(self, "No Files", "No files loaded to crop")
            return
        
        # Show crop dialog
        dialog = CropToolDialog(file_dict, self)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Update file paths to use cropped versions
            cropped_files = dialog.cropped_files
            
            if 'ha' in cropped_files:
                self.ha_file = cropped_files['ha']
                self.ha_label.setText(os.path.basename(cropped_files['ha']))
            
            if 'oiii' in cropped_files:
                self.oiii_file = cropped_files['oiii']
                self.oiii_label.setText(os.path.basename(cropped_files['oiii']))
            
            if 'oiii2' in cropped_files:
                self.oiii2_file = cropped_files['oiii2']
                self.oiii2_label.setText(os.path.basename(cropped_files['oiii2']))
            
            if 'sii' in cropped_files:
                self.sii_file = cropped_files['sii']
                self.sii_label.setText(os.path.basename(cropped_files['sii']))
    
    def _oiii2_already_baked(self):
        """Returns True if the OIII balanced temp file already has OIII2 combined in."""
        return getattr(self, '_balance_oiii2_baked', False)

    def _sync_gain_labels(self):
        """Update gain slider value labels to match current slider positions.
        Call this after setValue with blockSignals=True."""
        if hasattr(self, '_ha_gain_label') and self._ha_gain_label:
            self._ha_gain_label.setText(f"{self.ha_gain_slider.value()/100:.2f}x")
        if hasattr(self, '_oiii_gain_label') and self._oiii_gain_label:
            self._oiii_gain_label.setText(f"{self.oiii_gain_slider.value()/100:.2f}x")
        if hasattr(self, '_sii_gain_label') and self._sii_gain_label:
            self._sii_gain_label.setText(f"{self.sii_gain_slider.value()/100:.2f}x")

    def _refresh_live_preview(self):
        """Re-render the live preview using current gain slider values and combine mode."""
        if not hasattr(self, '_live_preview_label') or self._live_preview_label is None:
            print(f"[Preview] BAIL: _live_preview_label is None")
            return
        if not hasattr(self, '_live_preview_dialog') or self._live_preview_dialog is None:
            print(f"[Preview] BAIL: _live_preview_dialog is None")
            return
        try:
            if not self._live_preview_dialog.isVisible():
                print(f"[Preview] BAIL: dialog not visible")
                return
        except RuntimeError:
            print(f"[Preview] BAIL: RuntimeError on isVisible")
            return

        from PIL import Image

        try:
            ha  = self._live_ha_raw.copy()
            sii = self._live_sii_raw.copy()

            # Re-apply OIII combine mode every render so combo box changes are reflected live
            oiii = self._live_oiii_base.copy()
            if self._live_oiii2_base is not None:
                combine_mode = self.oiii_combine_mode.currentText()
                if combine_mode == "Average":
                    oiii = (oiii + self._live_oiii2_base) / 2.0
                elif combine_mode == "Maximum":
                    oiii = np.maximum(oiii, self._live_oiii2_base)
                else:  # Weighted 70/30
                    oiii = (0.7 * oiii + 0.3 * self._live_oiii2_base)

            # Apply current slider gains
            ha   = np.clip(ha   * (self.ha_gain_slider.value()   / 100.0), 0, 1)
            oiii = np.clip(oiii * (self.oiii_gain_slider.value() / 100.0), 0, 1)
            sii  = np.clip(sii  * (self.sii_gain_slider.value()  / 100.0), 0, 1)

            # Linked PPP autostretch — shows the effect of slider changes clearly
            # Check shapes match before stacking
            if not (ha.shape == oiii.shape == sii.shape):
                print(f"[Preview] ERROR: Shape mismatch - Ha:{ha.shape}, OIII:{oiii.shape}, SII:{sii.shape}")
                return  # Bail out safely
            
            rgb_combined = np.stack([sii, ha, oiii], axis=2).astype(np.float32)
            rgb_stretched = stretch_color_image_linked(rgb_combined, target_median=0.25, curves_boost=0.0)
            rgb = np.clip(rgb_stretched, 0, 1)
            
            # Apply synthetic luminance if checkbox is checked
            if self.synth_lum_checkbox.isChecked():
                blend_strength = self.synth_lum_strength.value() / 100.0
                
                # Create weighted synthetic luminance from linear (pre-gain) data
                ha_linear = self._live_ha_raw.copy()
                oiii_linear = self._live_oiii_base.copy()
                
                # Apply OIII2 combine if present
                if self._live_oiii2_base is not None:
                    combine_mode = self.oiii_combine_mode.currentText()
                    if combine_mode == "Average":
                        oiii_linear = (oiii_linear + self._live_oiii2_base) / 2.0
                    elif combine_mode == "Maximum":
                        oiii_linear = np.maximum(oiii_linear, self._live_oiii2_base)
                    else:  # Weighted 70/30
                        oiii_linear = (0.7 * oiii_linear + 0.3 * self._live_oiii2_base)
                
                sii_linear = self._live_sii_raw.copy()
                
                # Weight by signal strength
                ha_weight = np.median(ha_linear[ha_linear > 0]) if np.any(ha_linear > 0) else 1.0
                oiii_weight = np.median(oiii_linear[oiii_linear > 0]) if np.any(oiii_linear > 0) else 1.0
                sii_weight = np.median(sii_linear[sii_linear > 0]) if np.any(sii_linear > 0) else 1.0
                
                total_weight = ha_weight + oiii_weight + sii_weight
                ha_weight /= total_weight
                oiii_weight /= total_weight
                sii_weight /= total_weight
                
                # Create weighted synthetic luminance
                synth_lum = (ha_linear * ha_weight + oiii_linear * oiii_weight + sii_linear * sii_weight)
                
                # Apply same gains as preview channels
                synth_lum = np.clip(synth_lum * (
                    ha_weight * (self.ha_gain_slider.value() / 100.0) +
                    oiii_weight * (self.oiii_gain_slider.value() / 100.0) +
                    sii_weight * (self.sii_gain_slider.value() / 100.0)
                ), 0, 1)
                
                # Stretch synthetic luminance
                synth_lum_stretched = stretch_mono_image(synth_lum, target_median=0.25, curves_boost=0.0)
                
                # LRGB blending - preserve color while replacing luminance
                # Compute current luminance of color image (rec709 weights)
                palette_lum = 0.2126 * rgb[:,:,0] + 0.7152 * rgb[:,:,1] + 0.0722 * rgb[:,:,2]
                palette_lum = np.clip(palette_lum, 1e-6, 1.0)  # Avoid divide by zero
                
                # Compute scaling factor to replace palette luminance with synthetic
                lum_scale = synth_lum_stretched / palette_lum
                
                # Apply scale to each RGB channel (LRGB technique)
                r_lrgb = rgb[:,:,0] * lum_scale
                g_lrgb = rgb[:,:,1] * lum_scale
                b_lrgb = rgb[:,:,2] * lum_scale
                
                # Blend based on strength slider
                rgb[:,:,0] = rgb[:,:,0] * (1.0 - blend_strength) + r_lrgb * blend_strength
                rgb[:,:,1] = rgb[:,:,1] * (1.0 - blend_strength) + g_lrgb * blend_strength
                rgb[:,:,2] = rgb[:,:,2] * (1.0 - blend_strength) + b_lrgb * blend_strength
                
                rgb = np.clip(rgb, 0, 1)

            # Resize for display
            h, w = rgb.shape[:2]
            max_size = 1000
            if w > max_size or h > max_size:
                scale = min(max_size / w, max_size / h)
                img_pil = Image.fromarray((rgb * 255).astype(np.uint8))
                img_pil = img_pil.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
                rgb = np.array(img_pil).astype(np.float32) / 255.0

            rgb_8bit = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            h2, w2 = rgb_8bit.shape[:2]
            rgb_cont = np.ascontiguousarray(rgb_8bit)
            q_img = QImage(rgb_cont.data, w2, h2, w2 * 3, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img.copy())
            
            # Store original size for fit-to-window
            self._live_preview_original_size = (w2, h2)
            
            # Apply current zoom scale
            if hasattr(self, '_preview_scale') and self._preview_scale != 100:
                scaled_w = int(w2 * self._preview_scale / 100)
                scaled_h = int(h2 * self._preview_scale / 100)
                pixmap = pixmap.scaled(scaled_w, scaled_h, Qt.AspectRatioMode.KeepAspectRatio, 
                                      Qt.TransformationMode.SmoothTransformation)
                self._live_preview_label.setPixmap(pixmap)
                self._live_preview_label.setFixedSize(scaled_w, scaled_h)
            else:
                self._live_preview_label.setPixmap(pixmap)
                self._live_preview_label.setFixedSize(w2, h2)
            
            self._live_preview_label.update()
            self._live_preview_label.repaint()

        except Exception as e:
            print(f"[Preview] Render error: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_preview_scale(self, preview_label):
        """Apply current zoom scale to preview without re-rendering"""
        if not hasattr(self, '_live_preview_original_size'):
            return
        
        w_orig, h_orig = self._live_preview_original_size
        
        # Get the current pixmap at original size
        if hasattr(self, '_live_preview_label'):
            # Trigger a re-render which will apply the scale
            self._refresh_live_preview()

    def show_file_preview(self):
        """Open a non-modal live preview. Gain sliders on the main window update the image in real time."""
        from PIL import Image

        # If preview is already open, just bring it to front
        if hasattr(self, '_live_preview_dialog') and self._live_preview_dialog is not None:
            try:
                if self._live_preview_dialog.isVisible():
                    self._live_preview_dialog.raise_()
                    self._live_preview_dialog.activateWindow()
                    return
            except RuntimeError:
                pass  # dialog was deleted

        # Load from balanced temp files if available, otherwise raw originals
        def _load_raw(file_path, name):
            if file_path is None:
                return None
            data = self.load_fits_data(file_path, name)
            if data is None:
                return None
            d = data / 65535.0 if data.max() > 1.0 else data.astype(np.float32)
            return d.copy()

        ha_src   = self._balanced_ha_path   if getattr(self, '_balanced_ha_path',   None) and os.path.exists(self._balanced_ha_path)   else self.ha_file
        oiii_src = self._balanced_oiii_path if getattr(self, '_balanced_oiii_path', None) and os.path.exists(self._balanced_oiii_path) else self.oiii_file
        sii_src  = self._balanced_sii_path  if getattr(self, '_balanced_sii_path',  None) and os.path.exists(self._balanced_sii_path)  else self.sii_file

        ha_raw   = _load_raw(ha_src,   "Ha")
        oiii_raw = _load_raw(oiii_src, "OIII")
        sii_raw  = _load_raw(sii_src,  "SII")

        if ha_raw is None or oiii_raw is None or sii_raw is None:
            QMessageBox.warning(self, "Error", "Failed to load channel data")
            return

        # OIII2: only load and store separately if not already baked into balanced file
        oiii2_raw = None
        if self.oiii2_file and not self._oiii2_already_baked():
            oiii2_raw = _load_raw(self.oiii2_file, "OIII #2")

        # Store raw (pre-gain, pre-combine) arrays on self
        self._live_ha_raw    = ha_raw
        self._live_oiii_base = oiii_raw   # base OIII without combining
        self._live_oiii2_base = oiii2_raw  # None if no second file
        self._live_sii_raw   = sii_raw

        # Build non-modal dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Live Preview - SHO Composite (adjust sliders to update)")
        dialog.setModal(False)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

        def _on_close():
            self._live_preview_label  = None
            self._live_preview_dialog = None

        dialog.finished.connect(_on_close)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        # Header in styled groupbox
        info_group = QGroupBox("Live Preview Controls")
        info_group.setStyleSheet(
            "QGroupBox { font-weight: bold; color: #e0e0e0; border: 1px solid #5a5a5a; "
            "border-radius: 4px; margin-top: 6px; padding-top: 10px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }"
        )
        info_layout = QVBoxLayout(info_group)
        info_layout.setContentsMargins(12, 8, 12, 8)
        
        info = QLabel(
            "SHO composite (R=SII, G=Ha, B=OIII) with linked autostretch for display\n"
            "Adjust Channel Balance sliders on main window to see real-time changes"
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #b0b0b0; font-size: 9pt; font-style: italic;")
        info_layout.addWidget(info)
        layout.addWidget(info_group)

        scroll = QScrollArea()
        scroll.setWidgetResizable(False)
        scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)

        preview_label = QLabel()
        preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scroll.setWidget(preview_label)
        layout.addWidget(scroll)

        # Zoom controls
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QLabel("Zoom:"))
        
        self._preview_scale = 100  # Track current scale percentage
        
        zoom_in_btn = QPushButton("➕ Zoom In")
        def zoom_in():
            self._preview_scale = min(400, int(self._preview_scale * 1.2))
            self._update_preview_scale(preview_label)
        zoom_in_btn.clicked.connect(zoom_in)
        zoom_layout.addWidget(zoom_in_btn)
        
        zoom_out_btn = QPushButton("➖ Zoom Out")
        def zoom_out():
            self._preview_scale = max(25, int(self._preview_scale / 1.2))
            self._update_preview_scale(preview_label)
        zoom_out_btn.clicked.connect(zoom_out)
        zoom_layout.addWidget(zoom_out_btn)
        
        fit_btn = QPushButton("⬌ Fit to Window")
        def fit_to_window():
            if hasattr(self, '_live_preview_original_size'):
                w_orig, h_orig = self._live_preview_original_size
                w_avail = scroll.viewport().width() - 20
                h_avail = scroll.viewport().height() - 20
                scale = min(100, min(w_avail / w_orig, h_avail / h_orig) * 100)
                self._preview_scale = int(scale)
                self._update_preview_scale(preview_label)
        fit_btn.clicked.connect(fit_to_window)
        zoom_layout.addWidget(fit_btn)
        
        reset_btn = QPushButton("🔄 100%")
        def reset_zoom():
            self._preview_scale = 100
            self._update_preview_scale(preview_label)
        reset_btn.clicked.connect(reset_zoom)
        zoom_layout.addWidget(reset_btn)
        
        zoom_layout.addStretch()
        layout.addLayout(zoom_layout)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)

        # Store references so _refresh_live_preview can target this label
        self._live_preview_label  = preview_label
        self._live_preview_dialog = dialog
        self._live_preview_scroll = scroll

        dialog.resize(1050, 900)
        dialog.show()

        # Wire gain sliders and combine mode through the debounce timer
        def _schedule_refresh():
            self._preview_refresh_timer.start()  # restarts the 150ms countdown

        self.ha_gain_slider.valueChanged.connect(_schedule_refresh)
        self.oiii_gain_slider.valueChanged.connect(_schedule_refresh)
        self.sii_gain_slider.valueChanged.connect(_schedule_refresh)
        self.oiii_combine_mode.currentIndexChanged.connect(_schedule_refresh)
        self.synth_lum_checkbox.stateChanged.connect(_schedule_refresh)
        self.synth_lum_strength.valueChanged.connect(_schedule_refresh)

        def _disconnect():
            try: self.ha_gain_slider.valueChanged.disconnect(_schedule_refresh)
            except: pass
            try: self.oiii_gain_slider.valueChanged.disconnect(_schedule_refresh)
            except: pass
            try: self.sii_gain_slider.valueChanged.disconnect(_schedule_refresh)
            except: pass
            try: self.oiii_combine_mode.currentIndexChanged.disconnect(_schedule_refresh)
            except: pass
            try: self.synth_lum_checkbox.stateChanged.disconnect(_schedule_refresh)
            except: pass
            try: self.synth_lum_strength.valueChanged.disconnect(_schedule_refresh)
            except: pass

        dialog.finished.connect(_disconnect)

        # Render initial image
        self._refresh_live_preview()
    
    
    def run_custom_process(self):
        """Generate custom palette based on channel assignments"""
        if not self.ha_file or not self.oiii_file or not self.sii_file:
            QMessageBox.warning(self, "Files Required", "Please load Ha, OIII, and SII files first.")
            return
        
        try:
            # Reconnect to Siril (same as regular palette)
            self.siril.connect()
            
            # Load data (use balanced files if available)
            #print("\n[1/4] Loading narrowband files...")
            ha_data = self.load_fits_data(self.get_file_to_load('ha'), "Ha")
            oiii_data = self.load_fits_data(self.get_file_to_load('oiii'), "OIII")
            sii_data = self.load_fits_data(self.get_file_to_load('sii'), "SII")
            
            # Handle second OIII if present
            if self.oiii2_file:
                oiii2_raw = self.load_fits_data(self.oiii2_file, "OIII #2")
                combine_mode = self.oiii_combine_mode.currentText()
                if combine_mode == "Average":
                    oiii_data = (oiii_data + oiii2_raw) / 2.0
                elif combine_mode == "Maximum":
                    oiii_data = np.maximum(oiii_data, oiii2_raw)
                else:  # Weighted
                    oiii_data = (0.7 * oiii_data + 0.3 * oiii2_raw)
            
            # Get stretch settings
            stretch_mode = self.custom_stretch_mode.currentIndex()
            target_median = self.custom_stretch_slider.value() / 100.0
            apply_curves = self.custom_curves_checkbox.isChecked()
            
            # Normalize
            if ha_data.max() > 1.0:
                ha_data = ha_data / 65535.0
            if oiii_data.max() > 1.0:
                oiii_data = oiii_data / 65535.0
            if sii_data.max() > 1.0:
                sii_data = sii_data / 65535.0
            
            # Apply stretch if needed
            if stretch_mode == 1:  # Linked
                #print(f"\n[2/4] Applying Linked Stretch (target={target_median:.2f})...")
                rgb_combined = np.stack([ha_data, oiii_data, sii_data], axis=2)
                rgb_stretched = stretch_color_image_linked(rgb_combined, target_median, 
                                                          apply_curves=apply_curves, curves_boost=0.15)
                ha_data = rgb_stretched[:, :, 0]
                oiii_data = rgb_stretched[:, :, 1]
                sii_data = rgb_stretched[:, :, 2]
            elif stretch_mode == 2:  # Unlinked
                #print(f"\n[2/4] Applying Unlinked Stretch (target={target_median:.2f})...")
                ha_data = stretch_mono_image(ha_data, target_median, apply_curves=apply_curves, curves_boost=0.15)
                oiii_data = stretch_mono_image(oiii_data, target_median, apply_curves=apply_curves, curves_boost=0.15)
                sii_data = stretch_mono_image(sii_data, target_median, apply_curves=apply_curves, curves_boost=0.15)
            #else:
                #print("\n[2/4] No stretch applied")
            
            # Apply custom color assignments
            #print("\n[3/4] Applying custom color assignments...")
            #print(f"  Ha: RGB({self.custom_ha_color[0]}, {self.custom_ha_color[1]}, {self.custom_ha_color[2]})")
            #print(f"  OIII: RGB({self.custom_oiii_color[0]}, {self.custom_oiii_color[1]}, {self.custom_oiii_color[2]})")
            #print(f"  SII: RGB({self.custom_sii_color[0]}, {self.custom_sii_color[1]}, {self.custom_sii_color[2]})")
            
            r_channel = np.zeros_like(ha_data)
            g_channel = np.zeros_like(ha_data)
            b_channel = np.zeros_like(ha_data)
            
            # Ha contribution
            ha_r, ha_g, ha_b = self.custom_ha_color
            r_channel += ha_data * (ha_r / 255.0)
            g_channel += ha_data * (ha_g / 255.0)
            b_channel += ha_data * (ha_b / 255.0)
            
            # OIII contribution (combined with OIII#2 if present)
            oiii_r, oiii_g, oiii_b = self.custom_oiii_color
            r_channel += oiii_data * (oiii_r / 255.0)
            g_channel += oiii_data * (oiii_g / 255.0)
            b_channel += oiii_data * (oiii_b / 255.0)
            
            # SII contribution
            sii_r, sii_g, sii_b = self.custom_sii_color
            r_channel += sii_data * (sii_r / 255.0)
            g_channel += sii_data * (sii_g / 255.0)
            b_channel += sii_data * (sii_b / 255.0)
            
            # Clip to valid range
            r_channel = np.clip(r_channel, 0, 1)
            g_channel = np.clip(g_channel, 0, 1)
            b_channel = np.clip(b_channel, 0, 1)
            
            #print("\n[4/4] Creating RGB image...")
            
            # Stack RGB channels
            rgb_data = np.stack([r_channel, g_channel, b_channel], axis=0)
            rgb_32bit = np.clip(rgb_data, 0.0, 1.0).astype(np.float32)
            
            # Update FITS header
            if self.fits_header is not None:
                header_dict = self.fits_header.copy()
                header_dict['FILTER'] = "Custom"
                header_dict['NAXIS'] = 3
                header_dict['NAXIS1'] = rgb_32bit.shape[2]
                header_dict['NAXIS2'] = rgb_32bit.shape[1]
                header_dict['NAXIS3'] = 3
                if 'BG-PTS' in header_dict:
                    del header_dict['BG-PTS']
                hdu = fits.PrimaryHDU(header=header_dict)
                hdu.verify('silentfix')
                header_str = hdu.header.tostring(sep='\n')
            else:
                header_dict = fits.Header()
                header_dict['FILTER'] = ('Custom', 'Custom palette')
                header_dict['NAXIS'] = 3
                header_dict['NAXIS1'] = rgb_32bit.shape[2]
                header_dict['NAXIS2'] = rgb_32bit.shape[1]
                header_dict['NAXIS3'] = 3
                hdu = fits.PrimaryHDU(header=header_dict)
                header_str = hdu.header.tostring(sep='\n')
            
            # Create filename
            output_filename = "Custom_Palette.fit"
            output_path = os.path.join(os.getcwd(), output_filename)
            
            # Save image file (required before we can load it into Siril)
            self.siril.save_image_file(rgb_32bit, header_str, output_path)
            print(f"  Created: {output_filename}")
            
            # Load it into Siril
            self.siril.cmd("load", output_filename.replace('.fit', ''))
            
            # Create undo point
            self.siril.undo_save_state("Custom Palette")
            
            #print(f"  Image loaded in Siril")
            
            #print("\n" + "="*60)
            print("SUCCESS! Custom palette created!")
            #print("="*60)
            
            
        except Exception as e:
            print(f"Error in custom generation: {e}")
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to generate custom palette:\n{str(e)}")
    
    def run_process(self):
        try:
          
            # Save settings before processing
            self.save_settings()
            
            #print("="*60)
            #print("Siril Foraxx Palette Generator v5 - Statistical Stretch")
            #print("="*60)
            
            if not all([self.ha_file, self.oiii_file, self.sii_file]):
                missing = []
                if not self.ha_file: missing.append("Ha")
                if not self.oiii_file: missing.append("OIII")
                if not self.sii_file: missing.append("SII")
                raise ValueError(f"Please select all files. Missing: {', '.join(missing)}")
            
            self.siril.connect()
            
            # Load data (use balanced files if available)
            #print("\n[1/4] Loading narrowband files...")
            ha_data = self.load_fits_data(self.get_file_to_load('ha'), "Ha")
            ha_linear = self.load_fits_data(self.get_file_to_load('ha'), "Ha")
            oiii_data = self.load_fits_data(self.get_file_to_load('oiii'), "OIII")
            oiii_data = self.load_fits_data(self.get_file_to_load('oiii'), "OIII")
            oiii_linear = self.load_fits_data(self.get_file_to_load('oiii'), "OIII")  # Separate copy
            sii_data = self.load_fits_data(self.get_file_to_load('sii'), "SII")
            sii_linear = self.load_fits_data(self.get_file_to_load('sii'), "SII")
            
            if ha_data is None or oiii_data is None or sii_data is None:
                raise ValueError("Failed to load files")
            
            # Combine OIII if provided (skip if already baked into balanced temp file)
            if self.oiii2_file and not self._oiii2_already_baked():
                oiii2_data = self.load_fits_data(self.oiii2_file, "OIII #2")
                if oiii2_data is not None:
                    combine_mode = self.oiii_combine_mode.currentText()
                    if combine_mode == "Average":
                        oiii_data = (oiii_data + oiii2_data) / 2.0
                        oiii_linear = (oiii_data + oiii2_data) / 2.0
                    elif combine_mode == "Maximum":
                        oiii_data = np.maximum(oiii_data, oiii2_data)
                        oiii_linear =  np.maximum(oiii_data, oiii2_data)
                    else:
                        oiii_data = (0.7 * oiii_data + 0.3 * oiii2_data)
                        oiii_linear =  (0.7 * oiii_data + 0.3 * oiii2_data)
            
            if not (ha_data.shape == oiii_data.shape == sii_data.shape):
                error_msg = (
                    f"Image dimensions don't match!\n\n"
                    f"Ha:   {ha_data.shape[1]}x{ha_data.shape[0]} pixels\n"
                    f"OIII: {oiii_data.shape[1]}x{oiii_data.shape[0]} pixels\n"
                    f"SII:  {sii_data.shape[1]}x{sii_data.shape[0]} pixels\n\n"
                    f"All images must have identical dimensions.\n"
                    f"Please crop/resize your images to match."
                )
                QMessageBox.critical(self, "Dimension Mismatch", error_msg)
                raise ValueError(error_msg)
            
            # Normalize linear copies for synthetic luminance (must match stretched data range)
            if ha_linear.max() > 1.0:
                ha_linear = ha_linear / 65535.0
            if oiii_linear.max() > 1.0:
                oiii_linear = oiii_linear / 65535.0
            if sii_linear.max() > 1.0:
                sii_linear = sii_linear / 65535.0
            
            # Check alignment (if astroalign is available)
            if ASTROALIGN_AVAILABLE:
                aligned, offset = self.check_alignment(ha_data, oiii_data, sii_data)
                
                if not aligned:
                    # Show alignment warning
                    QMessageBox.warning(
                        self,
                        "Alignment Warning",
                        f"⚠️ Images appear to be misaligned!\n\n"
                        f"Maximum detected offset: {offset:.2f} pixels\n\n"
                        f"Recommendation: Use cropped and aligned images for best results.\n"
                        f"Misaligned images will result in color fringing and artifacts.\n\n"
                        f"Proceeding anyway..."
                    )
                    print(f"WARNING: Images are misaligned (offset: {offset:.2f}px) - proceeding anyway")
            
            # Determine stretch mode
            stretch_mode = self.stretch_mode.currentIndex()
            target_median = self.stretch_slider.value() / 100.0
            apply_curves = self.curves_checkbox.isChecked()
            
            # Store original data for unstretch if needed
            if self.unstretch_checkbox.isChecked() and stretch_mode > 0:
                #print("  Storing parameters for inverse stretch...")
                # We'll need to store the stretch parameters
                self.stored_target_median = target_median
            
            if stretch_mode == 1:  # Linked - combine into RGB first, stretch together
                #print(f"\n[2/4] Applying PPP Linked Stretch (target median={target_median:.2f})...")
                
                # Normalize to [0,1] first
                if ha_data.max() > 1.0:
                    ha_data = ha_data / 65535.0
                if oiii_data.max() > 1.0:
                    oiii_data = oiii_data / 65535.0
                if sii_data.max() > 1.0:
                    sii_data = sii_data / 65535.0
                
                # Combine into RGB array for linked stretch
                # (We'll split back out after stretching)
                rgb_combined = np.stack([ha_data, oiii_data, sii_data], axis=2)
                rgb_stretched = stretch_color_image_linked(rgb_combined, target_median, 
                                                          apply_curves=apply_curves, 
                                                          curves_boost=0.15)
                
                # Split back into individual channels
                ha_data = rgb_stretched[:, :, 0]
                oiii_data = rgb_stretched[:, :, 1]
                sii_data = rgb_stretched[:, :, 2]
                
                #print(f"  PPP Linked stretch complete (all channels stretched together)")
                
            elif stretch_mode == 2:  # Unlinked - stretch each channel independently
                #print(f"\n[2/4] Applying PPP Unlinked Stretch (target median={target_median:.2f})...")
                # Normalize to [0,1] first (same as linked path)
                if ha_data.max() > 1.0:
                    ha_data = ha_data / 65535.0
                if oiii_data.max() > 1.0:
                    oiii_data = oiii_data / 65535.0
                if sii_data.max() > 1.0:
                    sii_data = sii_data / 65535.0
                ha_data = stretch_mono_image(ha_data, target_median, apply_curves=apply_curves, curves_boost=0.15)
                oiii_data = stretch_mono_image(oiii_data, target_median, apply_curves=apply_curves, curves_boost=0.15)
                sii_data = stretch_mono_image(sii_data, target_median, apply_curves=apply_curves, curves_boost=0.15)
                #print(f"  PPP Unlinked stretch complete (each channel independent)")
            else:
                #print(f"\n[2/4] Normalizing data to [0, 1]...")
                if ha_data.max() > 1.0:
                    ha_data = ha_data / 65535.0
                if oiii_data.max() > 1.0:
                    oiii_data = oiii_data / 65535.0
                if sii_data.max() > 1.0:
                    sii_data = sii_data / 65535.0
            
            # Process palette
            mode = self.palette_pick.currentText()
            #print(f"\n[3/4] Processing '{mode}' palette...")
            
            # Use stretched data for all palettes when stretch is enabled
            if stretch_mode > 0:
                #print(f"  Using stretched channels for {mode} palette")
                use_ha = ha_data
                use_oiii = oiii_data
                use_sii = sii_data
            else:
                use_ha = ha_linear
                use_oiii = oiii_linear
                use_sii = sii_linear
            
            r_data, g_data, b_data = self.process_palette(use_ha, use_oiii, use_sii, mode)
            
            if r_data is None or g_data is None or b_data is None:
                raise ValueError("Failed to process palette")
            
            # Apply inverse stretch if requested
            if self.unstretch_checkbox.isChecked() and stretch_mode > 0:
                # Get unstretch strength from dropdown
                strength_map = {
                    0: 0.05,  # Light
                    1: 0.01,  # Moderate
                    2: 0.001   # Aggressive
                }
                unstretch_target = strength_map[self.unstretch_strength.currentIndex()]
                strength_name = ["Light", "Moderate", "Aggressive"][self.unstretch_strength.currentIndex()]
                
                #print(f"\n  Applying inverse stretch ({strength_name}, target={unstretch_target})...")
                # Combine RGB for unstretch
                rgb_palette = np.stack([r_data, g_data, b_data], axis=2)
                rgb_unstretched = unstretch_image(rgb_palette, target_median=unstretch_target)
                
                # Split back
                r_data = rgb_unstretched[:, :, 0]
                g_data = rgb_unstretched[:, :, 1]
                b_data = rgb_unstretched[:, :, 2]
                
                print(f"  Inverse stretch complete - data returned to approximately linear")
                print(f"  Note: Color ratios will differ from original due to palette processing")
            
            # Apply SCNR to remove purple/magenta stars if requested
            if self.scnr_checkbox.isChecked():
                #print(f"\n  Applying SCNR to remove purple stars...")
                rgb_palette = np.clip(np.stack([r_data, g_data, b_data], axis=2), 0.0, 1.0).astype(np.float32)
                
                # SCNR: invert → remove green (= remove magenta in original) → invert back
                inverted = 1.0 - rgb_palette
                # rmgreen mode 0 = Average Neutral, strength 1.0
                # Green channel reduced to min(G, avg(R,B))
                r_inv = inverted[:, :, 0]
                g_inv = inverted[:, :, 1]
                b_inv = inverted[:, :, 2]
                avg_rb = (r_inv + b_inv) * 0.5
                g_scnr = np.minimum(g_inv, avg_rb)
                inverted[:, :, 1] = g_scnr
                rgb_palette = 1.0 - inverted
                
                r_data = rgb_palette[:, :, 0]
                g_data = rgb_palette[:, :, 1]
                b_data = rgb_palette[:, :, 2]
                #print(f"  SCNR complete")
            
            # Apply synthetic luminance if requested
            if self.synth_lum_checkbox.isChecked():
                #print(f"\n  Creating synthetic luminance from all channels...")
                blend_strength = self.synth_lum_strength.value() / 100.0
                
                # Create luminance from original linear channels (before palette processing)
                # Weight by signal strength (median brightness)
                ha_weight = np.median(ha_linear[ha_linear > 0]) if np.any(ha_linear > 0) else 1.0
                oiii_weight = np.median(oiii_linear[oiii_linear > 0]) if np.any(oiii_linear > 0) else 1.0
                sii_weight = np.median(sii_linear[sii_linear > 0]) if np.any(sii_linear > 0) else 1.0
                
                total_weight = ha_weight + oiii_weight + sii_weight
                ha_weight /= total_weight
                oiii_weight /= total_weight
                sii_weight /= total_weight
                
                #print(f"    Channel weights: Ha={ha_weight:.2f}, OIII={oiii_weight:.2f}, SII={sii_weight:.2f}")
                
                # Create weighted synthetic luminance
                synth_lum = (ha_linear * ha_weight + oiii_linear * oiii_weight + sii_linear * sii_weight)
                synth_lum = np.clip(synth_lum, 0.0, 1.0).astype(np.float32)
                
                # Apply the same stretch that was applied to palette channels
                if stretch_mode == 1:  # Linked
                    synth_lum = stretch_color_image_linked(
                        np.stack([synth_lum, synth_lum, synth_lum], axis=2),
                        target_median, apply_curves=apply_curves, curves_boost=0.15
                    )[:, :, 0]
                elif stretch_mode == 2:  # Unlinked
                    synth_lum = stretch_mono_image(
                        synth_lum, target_median, apply_curves=apply_curves, curves_boost=0.15
                    )
                
                # If inverse stretch was applied to palette, apply it to luminance too
                if self.unstretch_checkbox.isChecked():
                    #print(f"    Applying inverse stretch to synthetic luminance...")
                    synth_lum_rgb = np.stack([synth_lum, synth_lum, synth_lum], axis=2)
                    synth_lum_unstretched = unstretch_image(synth_lum_rgb, target_median=unstretch_target)
                    synth_lum = synth_lum_unstretched[:, :, 0]
                
                # Blend luminance with palette using LRGB technique
                # Convert palette RGB to preserve color while replacing luminance
                rgb_palette = np.stack([r_data, g_data, b_data], axis=2)
                rgb_palette = np.clip(rgb_palette, 0.0, 1.0).astype(np.float32)
                
                # Compute current luminance of palette (rec709)
                palette_lum = 0.2126 * r_data + 0.7152 * g_data + 0.0722 * b_data
                palette_lum = np.clip(palette_lum, 1e-6, 1.0)  # Avoid divide by zero
                
                # Compute scaling factor to replace palette luminance with synthetic
                lum_scale = synth_lum / palette_lum
                
                # Apply scale to RGB channels (LRGB blending)
                r_lrgb = r_data * lum_scale
                g_lrgb = g_data * lum_scale
                b_lrgb = b_data * lum_scale
                
                # Blend based on strength slider
                r_data = r_data * (1.0 - blend_strength) + r_lrgb * blend_strength
                g_data = g_data * (1.0 - blend_strength) + g_lrgb * blend_strength
                b_data = b_data * (1.0 - blend_strength) + b_lrgb * blend_strength
                
                r_data = np.clip(r_data, 0.0, 1.0)
                g_data = np.clip(g_data, 0.0, 1.0)
                b_data = np.clip(b_data, 0.0, 1.0)
                
                #print(f"  Synthetic luminance applied (blend: {blend_strength*100:.0f}%)")
            
            # Create RGB image and push directly to Siril (MR #3 and #4)
            #print(f"\n[4/4] Creating RGB image...")
            
            # Stack RGB channels (CWH format for Siril: channels first)
            rgb_data = np.stack([r_data, g_data, b_data], axis=0)
            
            # Keep as 32-bit float (Siril's native processing format)
            rgb_32bit = np.clip(rgb_data, 0.0, 1.0).astype(np.float32)
            
            # Update FITS header with palette name (#4 from MR)
            if self.fits_header is not None:
                # Parse existing header
                header_dict = self.fits_header.copy()
                
                # Update filter name to palette type
                filter_name = mode.replace(' ', '_').replace('(', '').replace(')', '')
                header_dict['FILTER'] = filter_name
                
                # Update dimensions for RGB
                header_dict['NAXIS'] = 3
                header_dict['NAXIS1'] = rgb_32bit.shape[2]
                header_dict['NAXIS2'] = rgb_32bit.shape[1]
                header_dict['NAXIS3'] = 3
                
                # Remove broken GraXpert header if present
                if 'BG-PTS' in header_dict:
                    del header_dict['BG-PTS']
                
                # Create HDU and verify
                hdu = fits.PrimaryHDU(header=header_dict)
                hdu.verify('silentfix')
                header_str = hdu.header.tostring(sep='\n')
                
                #print(f"  Updated FITS header: FILTER={filter_name}")
            else:
                # No header available - create minimal one
                header_dict = fits.Header()
                filter_name = mode.replace(' ', '_').replace('(', '').replace(')', '')
                header_dict['FILTER'] = (filter_name, 'Narrowband palette type')
                header_dict['NAXIS'] = 3
                header_dict['NAXIS1'] = rgb_32bit.shape[2]
                header_dict['NAXIS2'] = rgb_32bit.shape[1]
                header_dict['NAXIS3'] = 3
                hdu = fits.PrimaryHDU(header=header_dict)
                header_str = hdu.header.tostring(sep='\n')
            
            # Create output filename
            # Clean palette name for filename
            clean_name = mode.replace(' ', '_').replace('(', '').replace(')', '')
            
            if self.save_result_checkbox.isChecked():
                output_filename = f"{clean_name}.fit"
            else:
                # No .temp prefix - just the palette name
                output_filename = f"{clean_name}.fit"
            
            output_path = os.path.join(os.getcwd(), output_filename)
            
            # Save image file (required before we can load it into Siril)
            self.siril.save_image_file(rgb_32bit, header_str, output_path)
            print(f"  Created: {output_filename}")
            
            # Load it into Siril
            self.siril.cmd("load", output_filename.replace('.fit', ''))
            
            # Create undo point
            self.siril.undo_save_state(f"Narrowband Palette: {mode}")
            
            # If not saving, mark file for cleanup (will be deleted when script exits)
            if not self.save_result_checkbox.isChecked():
                self.temp_files.append(output_path)
                #print(f"  Loaded as temporary (will be cleaned up)")
            else:
                print(f"  Saved to: {output_filename}")
            
            #print(f"  Image loaded in Siril")
            
            # Cleanup any temp files
            self.cleanup_temp_files()
            
            #print("\n" + "="*60)
            #print(f"SUCCESS! {mode} palette created!")
            if self.save_result_checkbox.isChecked():
                print(f"Output saved: {output_filename}")
            else:
                print(f"Output loaded in Siril as unsaved image")
            #print("="*60)
            
            self.siril.disconnect()
            
            if self.save_result_checkbox.isChecked():
                self.status_label.setText(f"✓ {mode} created: {output_filename}")
            else:
                self.status_label.setText(f"✓ {mode} loaded in Siril")
            
        except Exception as e:
            error_msg = str(e)
            print(f"\nERROR: {error_msg}")
            traceback.print_exc()
            self.status_label.setText(f"✗ Error: {error_msg}")
            QMessageBox.critical(self, "Error", error_msg)
            if hasattr(self, 'siril'):
                try:
                    self.siril.disconnect()
                except:
                    pass

if __name__ == '__main__':
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    ex = SASPaletteReplicator()
    ex.show()
    sys.exit(app.exec())