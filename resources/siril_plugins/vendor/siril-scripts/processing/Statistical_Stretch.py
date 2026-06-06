# (c) Cyril Richard 2026
# Code From Seti Astro Statistical Stretch - PyQt Version
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Version 3.2.1 - Exact port from SAS Pro

import sirilpy as s
s.ensure_installed('PyQt6')

import sys
import argparse
import numpy as np
import math

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QLabel, QSlider, QCheckBox, QPushButton,
                            QGroupBox, QMessageBox, QComboBox, QSplitter,
                            QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
                            QProgressBar)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QThread
from PyQt6.QtGui import QFont, QImage, QPixmap

VERSION = "3.1.2"
REQUIRES_SIRILPY = "0.6.10"

# Changelog:
# 3.0.0 CR: Full feature parity with SAS Pro (HDR + luminance-only)
# 3.1.0 CR: Add non-destructive preview with zoom/pan, clip stats, and toggle
# 3.1.1 CR: Fixing bug in "Luminance Only" mode
# 3.1.2 RS: If no CLI arguments, run in GUI mode by default

if not s.check_module_version(f'>={REQUIRES_SIRILPY}'):
    print(f"Please install sirilpy version {REQUIRES_SIRILPY} or higher")
    sys.exit(1)

# ============================================================================
# LUMINANCE PROFILES
# ============================================================================

_LUMA_REC709  = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
_LUMA_REC601  = np.array([0.2990, 0.5870, 0.1140], dtype=np.float32)
_LUMA_REC2020 = np.array([0.2627, 0.6780, 0.0593], dtype=np.float32)

LUMA_PROFILES = {
    "rec709": {"weights": _LUMA_REC709, "category": "Standard"},
    "rec601": {"weights": _LUMA_REC601, "category": "Standard"},
    "rec2020": {"weights": _LUMA_REC2020, "category": "Standard"},
    "cie1931": {"weights": [0.176204, 0.812985, 0.0108109], "category": "CIE"},
    "average": {"weights": [0.3333, 0.3333, 0.3334], "category": "Simple"},
}

def resolve_luma_profile_weights(mode: str):
    key = str(mode).strip().lower()
    alias = {
        "rec.709": "rec709", "rec-709": "rec709", "rgb": "rec709", "k": "rec709",
        "rec.601": "rec601", "rec-601": "rec601",
        "rec.2020": "rec2020", "rec-2020": "rec2020",
    }
    key = alias.get(key, key)
    prof = LUMA_PROFILES.get(key)
    if not prof:
        return ("rec709", _LUMA_REC709, None)
    w = prof.get("weights", None)
    if w is not None:
        w = np.asarray(w, dtype=np.float32)
    return (key, w, None)

def compute_luminance(img: np.ndarray, method: str = "rec709", 
                     weights: np.ndarray = None, noise_sigma=None) -> np.ndarray:
    f = np.clip(img, 0.0, 1.0).astype(np.float32, copy=False)
    if f.ndim == 2:
        return f
    if f.ndim != 3:
        raise ValueError("compute_luminance: expected 2-D or 3-D array.")

    H, W, C = f.shape
    if C == 1:
        return f[..., 0]

    if weights is not None:
        w = np.asarray(weights, dtype=np.float32)
        if w.size == 3:
            lum = np.tensordot(f[..., :3], w, axes=([2], [0]))
        else:
            lum = np.tensordot(f[..., :3], _LUMA_REC709, axes=([2], [0]))
    elif method == "rec601":
        lum = np.tensordot(f[..., :3], _LUMA_REC601, axes=([2],[0]))
    elif method == "rec2020":
        lum = np.tensordot(f[..., :3], _LUMA_REC2020, axes=([2],[0]))
    else:
        lum = np.tensordot(f[..., :3], _LUMA_REC709, axes=([2],[0]))

    return np.clip(lum.astype(np.float32, copy=False), 0.0, 1.0)

def recombine_luminance_linear_scale(target_rgb: np.ndarray, new_L: np.ndarray,
                                     weights: np.ndarray = _LUMA_REC709,
                                     eps: float = 1e-6, blend: float = 1.0,
                                     highlight_soft_knee: float = 0.0) -> np.ndarray:
    rgb = np.clip(target_rgb, 0.0, 1.0).astype(np.float32, copy=False)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Recombine Luminance requires an RGB target image.")

    H, W, _ = rgb.shape
    L = new_L.astype(np.float32)

    w = np.asarray(weights, dtype=np.float32)
    if w.shape != (3,):
        raise ValueError("weights must be length-3 for RGB recombine.")

    Y = rgb[..., 0]*w[0] + rgb[..., 1]*w[1] + rgb[..., 2]*w[2]
    s = L / (Y + eps)

    if highlight_soft_knee > 0.0:
        k = np.clip(highlight_soft_knee, 0.0, 1.0)
        s = s / (1.0 + k*(s - 1.0))

    out = rgb * s[..., None]
    out = np.clip(out, 0.0, 1.0)

    if 0.0 <= blend < 1.0:
        out = rgb*(1.0 - blend) + out*blend

    return out.astype(np.float32, copy=False)

# ============================================================================
# HDR HIGHLIGHT COMPRESSION
# ============================================================================

def hdr_compress_highlights(x: np.ndarray, amount: float, knee: float = 0.75) -> np.ndarray:
    a = float(np.clip(amount, 0.0, 1.0))
    if a <= 0.0:
        return x.astype(np.float32, copy=False)
    k = float(np.clip(knee, 0.0, 0.99))
    y = x.astype(np.float32, copy=False)
    hi = y > k
    if not np.any(hi):
        return np.clip(y, 0.0, 1.0).astype(np.float32, copy=False)
    t = (y[hi] - k) / (1.0 - k)
    t = np.clip(t, 0.0, 1.0)
    m1 = 1.0 + 4.0 * a
    m1 = float(np.clip(m1, 1.0, 5.0))
    t2 = t * t
    t3 = t2 * t
    h10 = (t3 - 2.0 * t2 + t)
    h01 = (-2.0 * t3 + 3.0 * t2)
    h11 = (t3 - t2)
    f = h10 * 1.0 + h01 * 1.0 + h11 * m1
    y2 = y.copy()
    y2[hi] = k + (1.0 - k) * np.clip(f, 0.0, 1.0)
    return np.clip(y2, 0.0, 1.0).astype(np.float32, copy=False)

def hdr_compress_color_luminance(rgb: np.ndarray, amount: float, knee: float,
                                 luma_mode: str = "rec709") -> np.ndarray:
    a = float(np.clip(amount, 0.0, 1.0))
    if a <= 0.0:
        return rgb.astype(np.float32, copy=False)

    resolved_method, w, _ = resolve_luma_profile_weights(luma_mode)
    if w is not None and np.asarray(w).size == 3:
        rw = np.asarray(w, dtype=np.float32)
    else:
        if resolved_method == "rec601":
            rw = _LUMA_REC601
        elif resolved_method == "rec2020":
            rw = _LUMA_REC2020
        else:
            rw = _LUMA_REC709

    Y = compute_luminance(rgb, method=resolved_method, weights=rw)
    Yc = hdr_compress_highlights(Y, a, knee=float(knee))

    return recombine_luminance_linear_scale(rgb, Yc, weights=rw, blend=1.0, 
                                           highlight_soft_knee=0.25)

# ============================================================================
# ROBUST STATISTICS
# ============================================================================

def _sample_flat(x: np.ndarray, max_n: int = 400_000) -> np.ndarray:
    flat = np.asarray(x, np.float32).reshape(-1)
    n = flat.size
    if n <= max_n:
        return flat
    stride = max(1, n // max_n)
    return flat[::stride]

def _robust_sigma_lower_half_fast(x: np.ndarray, max_n: int = 400_000) -> float:
    s = _sample_flat(x, max_n=max_n)
    med = float(np.median(s))
    lo = s[s <= med]
    if lo.size < 16:
        mad = float(np.median(np.abs(s - med)))
    else:
        med_lo = float(np.median(lo))
        mad = float(np.median(np.abs(lo - med_lo)))
    return 1.4826 * mad

def _compute_blackpoint_sigma(img: np.ndarray, sigma: float) -> tuple:
    img = np.asarray(img, dtype=np.float32)
    med = float(np.median(img))
    sig = float(sigma)
    noise = _robust_sigma_lower_half_fast(img)
    bp = med - sig * noise
    mn = float(img.min())
    bp = max(mn, bp)
    bp = min(bp, 0.99)
    return float(bp), med

def _compute_blackpoint_sigma_per_channel(img: np.ndarray, sigma: float) -> np.ndarray:
    sig = float(sigma)
    bp = np.zeros(3, dtype=np.float32)
    for c in range(3):
        ch = img[..., c].astype(np.float32, copy=False)
        med = float(np.median(ch))
        noise = _robust_sigma_lower_half_fast(ch)
        b = med - sig * noise
        b = max(float(ch.min()), b)
        b = min(b, 0.99)
        bp[c] = b
    return bp

# ============================================================================
# CLIP STATS
# ============================================================================

def compute_clip_stats(img: np.ndarray, blackpoint_sigma: float, no_black_clip: bool,
                      luma_only: bool, linked: bool, luma_profile: str = "rec709") -> str:
    try:
        sig = float(blackpoint_sigma)
        
        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
            mono = img.squeeze().astype(np.float32, copy=False)
            if no_black_clip:
                bp = float(mono.min())
            else:
                bp, _ = _compute_blackpoint_sigma(mono, sig)
            clipped = (mono <= bp)
            clipped_count = int(np.count_nonzero(clipped))
            total = int(clipped.size)
            pct = 100.0 * clipped_count / max(1, total)
            mode = "Mono"
            bp_text = f"bp={bp:.6f}"
        else:
            rgb = img.astype(np.float32, copy=False)
            
            if luma_only or linked:
                L = compute_luminance(rgb, luma_profile)
                if no_black_clip:
                    bp = float(L.min())
                else:
                    bp, _ = _compute_blackpoint_sigma(L, sig)
                clipped = (L <= bp)
                mode = "Luma-only (L ≤ bp)" if luma_only else "Linked (L ≤ bp)"
                bp_text = f"L bp={bp:.6f}"
            else:
                if no_black_clip:
                    bp3 = np.array([float(rgb[...,0].min()),
                                   float(rgb[...,1].min()),
                                   float(rgb[...,2].min())], dtype=np.float32)
                else:
                    bp3 = _compute_blackpoint_sigma_per_channel(rgb, sig)
                clipped = np.any(rgb <= bp3.reshape((1, 1, 3)), axis=2)
                mode = "Unlinked (any channel ≤ bp)"
                bp_text = f"R={bp3[0]:.6f}, G={bp3[1]:.6f}, B={bp3[2]:.6f}"
            
            clipped_count = int(np.count_nonzero(clipped))
            total = int(clipped.size)
            pct = 100.0 * clipped_count / max(1, total)
        
        if no_black_clip:
            return f"No clipping ({mode}). Threshold={bp_text}: {clipped_count:,}/{total:,} pixels ({pct:.4f}%)"
        else:
            return f"Black clip ({mode}) @ {bp_text}: {clipped_count:,}/{total:,} pixels ({pct:.4f}%)"
    
    except Exception as e:
        return f"Error computing stats: {e}"

# ============================================================================
# CORE STRETCH PROCESSOR (exact SAS Pro port)
# ============================================================================

class StatisticalStretchProcessor:
    def __init__(self):
        pass

    def floor_value(self, value, decimals=2):
        factor = 10 ** decimals
        return math.floor(value * factor) / factor

    def apply_curves_adjustment(self, image, target_median, curves_boost):
        if curves_boost <= 0.0:
            return np.clip(image, 0.0, 1.0).astype(np.float32)
        img = np.clip(image.astype(np.float32), 0.0, 1.0)
        tm = float(target_median)
        cb = float(curves_boost)
        p3x = 0.25 * (1.0 - tm) + tm
        p4x = 0.75 * (1.0 - tm) + tm
        p3y = p3x ** (1.0 - cb)
        p4y = (p4x ** (1.0 - cb)) ** (1.0 - cb)
        xvals = np.array([0.0, 0.5 * tm, tm, p3x, p4x, 1.0], dtype=np.float32)
        yvals = np.array([0.0, 0.5 * tm, tm, p3y, p4y, 1.0], dtype=np.float32)
        out = np.interp(img, xvals, yvals).astype(np.float32, copy=False)
        return np.clip(out, 0.0, 1.0)

    def stretch_mono_image(self, img, target_median, normalize=False, apply_curves=False, 
                          curves_boost=0.0, blackpoint_sigma=5.0, no_black_clip=False,
                          hdr_compress=False, hdr_amount=0.0, hdr_knee=0.75):
        target_median = max(0.01, min(0.99, target_median))
        
        if no_black_clip:
            black_point = np.min(img)
            median_image = np.median(img)
        else:
            black_point, median_image = _compute_blackpoint_sigma(img, blackpoint_sigma)
        
        denom = max(1.0 - black_point, 1e-12)
        rescaled_image = (img - black_point) / denom
        median_rescaled = np.median(rescaled_image)
        
        num = (median_rescaled - 1) * target_median * rescaled_image
        den = median_rescaled * (target_median + rescaled_image - 1) - target_median * rescaled_image
        den = np.where(np.abs(den) < 1e-12, 1e-12, den)
        stretched_image = num / den
        
        if apply_curves:
            stretched_image = self.apply_curves_adjustment(stretched_image, target_median, curves_boost)
        
        if hdr_compress and hdr_amount > 0.0:
            stretched_image = hdr_compress_highlights(stretched_image, hdr_amount, knee=hdr_knee)
        
        if normalize:
            stretched_image = stretched_image / np.max(stretched_image)
        
        return np.clip(stretched_image, 0.0, 1.0).astype(np.float32)

    def stretch_color_image(self, img, target_median, linked=True, normalize=False, 
                           apply_curves=False, curves_boost=0.0, blackpoint_sigma=5.0, 
                           no_black_clip=False, hdr_compress=False, hdr_amount=0.0, 
                           hdr_knee=0.75, luma_only=False, luma_mode="rec709",
                           luma_blend=1.0):
        target_median = max(0.01, min(0.99, target_median))
        sig = float(blackpoint_sigma)
        
        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
            mono = img.squeeze()
            mono_out = self.stretch_mono_image(
                mono, target_median, normalize, apply_curves, curves_boost,
                blackpoint_sigma, no_black_clip, hdr_compress, hdr_amount, hdr_knee
            )
            return np.stack([mono_out]*3, axis=-1)
        
        if luma_only:
            b = float(np.clip(luma_blend, 0.0, 1.0))
            
            # A) Normal linked RGB stretch
            if no_black_clip:
                bp = float(img.min())
                med_img = float(np.median(img))
            else:
                bp, med_img = _compute_blackpoint_sigma(img, sig)
            
            denom = max(1.0 - bp, 1e-12)
            med_rescaled = (med_img - bp) / denom
            
            rescaled_image = (img - bp) / denom
            median_image = np.median(rescaled_image)
            
            num = (median_image - 1) * target_median * rescaled_image
            den = median_image * (target_median + rescaled_image - 1) - target_median * rescaled_image
            den = np.where(np.abs(den) < 1e-12, 1e-12, den)
            linked_out = num / den
            
            if apply_curves:
                linked_out = self.apply_curves_adjustment(linked_out, target_median, curves_boost)
            
            if hdr_compress and hdr_amount > 0.0:
                linked_out = hdr_compress_color_luminance(linked_out, hdr_amount, hdr_knee, "rec709")
            
            if normalize:
                mx = float(linked_out.max())
                if mx > 0:
                    linked_out = linked_out / mx
            
            linked_out = np.clip(linked_out, 0.0, 1.0).astype(np.float32, copy=False)
            
            if b <= 0.0:
                return linked_out
            
            # B) Luma-only recombine stretch
            resolved_method, w, _profile_name = resolve_luma_profile_weights(luma_mode)
            
            L = compute_luminance(img, method=resolved_method, weights=w)
            
            Ls = self.stretch_mono_image(
                L, target_median, normalize=False, apply_curves=apply_curves,
                curves_boost=curves_boost, blackpoint_sigma=sig, no_black_clip=no_black_clip,
                hdr_compress=False, hdr_amount=0.0, hdr_knee=hdr_knee
            )
            
            if hdr_compress and hdr_amount > 0.0:
                Ls = hdr_compress_highlights(Ls, hdr_amount, knee=hdr_knee)
            
            if w is not None and np.asarray(w).size == 3:
                rw = np.asarray(w, dtype=np.float32)
                s = float(rw.sum())
                if s > 0:
                    rw = rw / s
            else:
                if resolved_method == "rec601":
                    rw = _LUMA_REC601
                elif resolved_method == "rec2020":
                    rw = _LUMA_REC2020
                else:
                    rw = _LUMA_REC709
            
            luma_out = recombine_luminance_linear_scale(
                img, Ls, weights=rw, blend=1.0, highlight_soft_knee=0.0
            )
            
            if normalize:
                mx = float(luma_out.max())
                if mx > 0:
                    luma_out = luma_out / mx
            
            luma_out = np.clip(luma_out, 0.0, 1.0).astype(np.float32, copy=False)
            
            out = (1.0 - b) * linked_out + b * luma_out
            return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)
        
        # Normal RGB mode
        if linked:
            if no_black_clip:
                black_point = np.min(img)
                combined_median = np.median(img)
            else:
                black_point, combined_median = _compute_blackpoint_sigma(img, blackpoint_sigma)
            
            rescaled_image = (img - black_point) / (1 - black_point)
            median_image = np.median(rescaled_image)
            
            num = (median_image - 1) * target_median * rescaled_image
            den = median_image * (target_median + rescaled_image - 1) - target_median * rescaled_image
            den = np.where(np.abs(den) < 1e-12, 1e-12, den)
            stretched_image = num / den
        else:
            stretched_image = np.zeros_like(img)
            for c in range(3):
                channel_data = img[..., c]
                if no_black_clip:
                    black_point = np.min(channel_data)
                    median_channel_data = np.median(channel_data)
                else:
                    black_point, median_channel_data = _compute_blackpoint_sigma(channel_data, blackpoint_sigma)
                
                rescaled_channel = (channel_data - black_point) / (1 - black_point)
                median_channel = np.median(rescaled_channel)
                
                num = (median_channel - 1) * target_median * rescaled_channel
                den = median_channel * (target_median + rescaled_channel - 1) - target_median * rescaled_channel
                den = np.where(np.abs(den) < 1e-12, 1e-12, den)
                stretched_channel = num / den
                
                stretched_image[..., c] = stretched_channel
        
        if apply_curves:
            stretched_image = self.apply_curves_adjustment(stretched_image, target_median, curves_boost)
        
        if hdr_compress and hdr_amount > 0.0:
            stretched_image = hdr_compress_color_luminance(stretched_image, hdr_amount, hdr_knee, "rec709")
        
        if normalize:
            stretched_image = stretched_image / np.max(stretched_image)
        
        return np.clip(stretched_image, 0.0, 1.0).astype(np.float32)

# ============================================================================
# WORKER THREAD
# ============================================================================

class StretchWorker(QObject):
    progress_update = pyqtSignal(str, int)
    finished = pyqtSignal(object, str)
    
    def __init__(self, img, params):
        super().__init__()
        self.img = img
        self.params = params
        self.processor = StatisticalStretchProcessor()
    
    def run(self):
        try:
            self.progress_update.emit("Stretching image...", 50)
            
            if self.img.ndim == 2 or (self.img.ndim == 3 and self.img.shape[2] == 1):
                mono = self.img.squeeze()
                out = self.processor.stretch_mono_image(
                    mono, **{k:v for k,v in self.params.items() 
                            if k not in ['linked', 'luma_only', 'luma_mode', 'luma_blend']}
                )
                if out.ndim == 2:
                    out = np.stack([out]*3, axis=-1)
            else:
                out = self.processor.stretch_color_image(self.img, **self.params)
            
            self.progress_update.emit("Computing clip stats...", 90)
            stats = compute_clip_stats(
                self.img, self.params['blackpoint_sigma'], self.params['no_black_clip'],
                self.params.get('luma_only', False), self.params.get('linked', False),
                self.params.get('luma_mode', 'rec709')
            )
            
            self.finished.emit(out, stats)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished.emit(None, f"Error: {e}")

# ============================================================================
# ZOOMABLE GRAPHICS VIEW
# ============================================================================

class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, scene, parent=None):
        super().__init__(parent)
        self.setScene(scene)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self._zoom = 1.0
        self._zoom_min = 0.05
        self._zoom_max = 10.0
        self._zoom_step = 1.25
    
    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()
        event.accept()
    
    def zoom_in(self):
        new_zoom = min(self._zoom * self._zoom_step, self._zoom_max)
        self._apply_zoom(new_zoom)
    
    def zoom_out(self):
        new_zoom = max(self._zoom / self._zoom_step, self._zoom_min)
        self._apply_zoom(new_zoom)
    
    def fit_item(self, item):
        if item and not item.pixmap().isNull():
            self.fitInView(item, Qt.AspectRatioMode.KeepAspectRatio)
            self._zoom = 1.0
    
    def _apply_zoom(self, new_zoom):
        factor = new_zoom / self._zoom
        self.scale(factor, factor)
        self._zoom = new_zoom

# ============================================================================
# MAIN DIALOG
# ============================================================================

class StatisticalStretchDialog(QMainWindow):
    def __init__(self, siril):
        super().__init__()
        self.setWindowTitle(f"Statistical Stretch v{VERSION}")
        self.resize(1200, 700)
        
        self.siril = siril
        self.processor = StatisticalStretchProcessor()
        
        if not self.siril.is_image_loaded():
            QMessageBox.critical(self, "Error", "No image loaded in Siril")
            raise SystemExit(1)
        
        try:
            self.siril.cmd("requires", "1.3.6")
        except Exception:
            pass
        
        # Get image
        fit = self.siril.get_image()
        fit.ensure_data_type(np.float32)
        img = fit.data
        
        # Convert to HWC
        if img.ndim == 2:
            img_hwc = np.stack([img]*3, axis=-1)
            self._was_mono = True
        elif img.ndim == 3:
            if img.shape[0] in [1, 3]:
                img_hwc = np.moveaxis(img, 0, -1)
                self._was_mono = (img.shape[0] == 1)
            else:
                img_hwc = img
                self._was_mono = False
        else:
            img_hwc = img
            self._was_mono = False
        
        if img_hwc.shape[2] == 1:
            img_hwc = np.repeat(img_hwc, 3, axis=2)
            self._was_mono = True
        
        img_hwc = np.clip(img_hwc, 0.0, 1.0).astype(np.float32, copy=False)
        
        self.original_hwc = img_hwc
        self.preview_hwc = img_hwc.copy()
        
        self.init_ui()
    
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        splitter.setSizes([400, 800])
        
        self._set_pix(self.preview_hwc)
        self.hdr_knee_locked = False
    
    def create_left_panel(self):
        left_widget = QWidget()
        left_widget.setFixedWidth(400)
        layout = QVBoxLayout(left_widget)
        layout.setSpacing(8)
        
        title = QLabel("Statistical Stretch")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Parameters
        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout(params_group)
        params_layout.setSpacing(6)
        
        # Target Median
        median_layout = QHBoxLayout()
        median_layout.addWidget(QLabel("Target median:"))
        self.target_median_slider = QSlider(Qt.Orientation.Horizontal)
        self.target_median_slider.setRange(1, 99)
        self.target_median_slider.setValue(25)
        self.target_median_slider.valueChanged.connect(self.update_target_median_display)
        self.target_median_slider.valueChanged.connect(self.auto_suggest_hdr_knee)
        median_layout.addWidget(self.target_median_slider)
        self.target_median_display = QLabel("0.25")
        self.target_median_display.setFixedWidth(45)
        median_layout.addWidget(self.target_median_display)
        params_layout.addLayout(median_layout)
        
        # Black Point Sigma
        sigma_layout = QHBoxLayout()
        sigma_layout.addWidget(QLabel("Black point σ:"))
        self.blackpoint_sigma_slider = QSlider(Qt.Orientation.Horizontal)
        self.blackpoint_sigma_slider.setRange(50, 600)
        self.blackpoint_sigma_slider.setValue(500)
        self.blackpoint_sigma_slider.valueChanged.connect(self.update_blackpoint_sigma_display)
        sigma_layout.addWidget(self.blackpoint_sigma_slider)
        self.blackpoint_sigma_display = QLabel("5.00")
        self.blackpoint_sigma_display.setFixedWidth(45)
        sigma_layout.addWidget(self.blackpoint_sigma_display)
        params_layout.addLayout(sigma_layout)
        
        layout.addWidget(params_group)
        
        # Options
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout(options_group)
        options_layout.setSpacing(4)
        
        self.no_black_clip_checkbox = QCheckBox("No black clipping")
        self.no_black_clip_checkbox.setChecked(False)
        self.no_black_clip_checkbox.toggled.connect(self.toggle_blackpoint_slider)
        options_layout.addWidget(self.no_black_clip_checkbox)
        
        self.linked_stretch_checkbox = QCheckBox("Linked Stretch")
        self.linked_stretch_checkbox.setChecked(False)
        options_layout.addWidget(self.linked_stretch_checkbox)
        
        self.normalize_checkbox = QCheckBox("Normalize")
        self.normalize_checkbox.setChecked(False)
        options_layout.addWidget(self.normalize_checkbox)
        
        layout.addWidget(options_group)
        
        # HDR
        hdr_group = QGroupBox("HDR Highlight Compress")
        hdr_layout = QVBoxLayout(hdr_group)
        hdr_layout.setSpacing(4)
        
        self.hdr_checkbox = QCheckBox("Enable HDR compress")
        self.hdr_checkbox.setChecked(False)
        self.hdr_checkbox.toggled.connect(self.toggle_hdr_controls)
        hdr_layout.addWidget(self.hdr_checkbox)
        
        hdr_amount_layout = QHBoxLayout()
        hdr_amount_layout.addWidget(QLabel("Amount:"))
        self.hdr_amount_slider = QSlider(Qt.Orientation.Horizontal)
        self.hdr_amount_slider.setRange(0, 100)
        self.hdr_amount_slider.setValue(15)
        self.hdr_amount_slider.valueChanged.connect(self.update_hdr_amount_display)
        self.hdr_amount_slider.setEnabled(False)
        hdr_amount_layout.addWidget(self.hdr_amount_slider)
        self.hdr_amount_display = QLabel("0.15")
        self.hdr_amount_display.setFixedWidth(45)
        hdr_amount_layout.addWidget(self.hdr_amount_display)
        hdr_layout.addLayout(hdr_amount_layout)
        
        hdr_knee_layout = QHBoxLayout()
        hdr_knee_layout.addWidget(QLabel("Knee:"))
        self.hdr_knee_slider = QSlider(Qt.Orientation.Horizontal)
        self.hdr_knee_slider.setRange(10, 95)
        self.hdr_knee_slider.setValue(75)
        self.hdr_knee_slider.valueChanged.connect(self.update_hdr_knee_display)
        self.hdr_knee_slider.sliderPressed.connect(self.lock_hdr_knee)
        self.hdr_knee_slider.setEnabled(False)
        hdr_knee_layout.addWidget(self.hdr_knee_slider)
        self.hdr_knee_display = QLabel("0.75")
        self.hdr_knee_display.setFixedWidth(45)
        hdr_knee_layout.addWidget(self.hdr_knee_display)
        hdr_layout.addLayout(hdr_knee_layout)
        
        layout.addWidget(hdr_group)
        
        # Luminance
        luma_group = QGroupBox("Luminance-Only Mode")
        luma_layout = QVBoxLayout(luma_group)
        luma_layout.setSpacing(4)
        
        self.luma_checkbox = QCheckBox("Stretch luminance only")
        self.luma_checkbox.setChecked(False)
        self.luma_checkbox.toggled.connect(self.toggle_luma_controls)
        luma_layout.addWidget(self.luma_checkbox)
        
        luma_profile_layout = QHBoxLayout()
        luma_profile_layout.addWidget(QLabel("Profile:"))
        self.luma_profile_combo = QComboBox()
        self.luma_profile_combo.addItems(sorted(LUMA_PROFILES.keys()))
        self.luma_profile_combo.setCurrentText("rec709")
        self.luma_profile_combo.setEnabled(False)
        luma_profile_layout.addWidget(self.luma_profile_combo, 1)
        luma_layout.addLayout(luma_profile_layout)
        
        luma_blend_layout = QHBoxLayout()
        luma_blend_layout.addWidget(QLabel("Luma blend:"))
        self.luma_blend_slider = QSlider(Qt.Orientation.Horizontal)
        self.luma_blend_slider.setRange(0, 100)
        self.luma_blend_slider.setValue(60)
        self.luma_blend_slider.valueChanged.connect(self.update_luma_blend_display)
        self.luma_blend_slider.setEnabled(False)
        luma_blend_layout.addWidget(self.luma_blend_slider)
        self.luma_blend_display = QLabel("0.60")
        self.luma_blend_display.setFixedWidth(45)
        luma_blend_layout.addWidget(self.luma_blend_display)
        luma_layout.addLayout(luma_blend_layout)
        
        layout.addWidget(luma_group)
        
        # Curves
        curves_group = QGroupBox("Curves Adjustment")
        curves_layout = QVBoxLayout(curves_group)
        curves_layout.setSpacing(4)
        
        self.apply_curve_checkbox = QCheckBox("Apply curves boost")
        self.apply_curve_checkbox.setChecked(False)
        self.apply_curve_checkbox.toggled.connect(self.toggle_curves_boost)
        curves_layout.addWidget(self.apply_curve_checkbox)
        
        curves_boost_layout = QHBoxLayout()
        curves_boost_layout.addWidget(QLabel("Strength:"))
        self.curves_boost_slider = QSlider(Qt.Orientation.Horizontal)
        self.curves_boost_slider.setRange(0, 100)
        self.curves_boost_slider.setValue(0)
        self.curves_boost_slider.valueChanged.connect(self.update_curves_boost_display)
        self.curves_boost_slider.setEnabled(False)
        curves_boost_layout.addWidget(self.curves_boost_slider)
        self.curves_boost_display = QLabel("0.00")
        self.curves_boost_display.setFixedWidth(45)
        curves_boost_layout.addWidget(self.curves_boost_display)
        curves_layout.addLayout(curves_boost_layout)
        
        layout.addWidget(curves_group)
        
        # Preview buttons
        preview_layout = QHBoxLayout()
        self.btn_preview = QPushButton("Preview")
        self.btn_preview.clicked.connect(self._start_preview)
        preview_layout.addWidget(self.btn_preview)
        
        self.btn_toggle = QPushButton("Show Original")
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.clicked.connect(self._toggle)
        preview_layout.addWidget(self.btn_toggle)
        layout.addLayout(preview_layout)
        
        # Progress
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        self.lbl_step = QLabel("Idle")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.lbl_step)
        progress_layout.addWidget(self.progress_bar)
        layout.addWidget(progress_group)
        
        # Clip stats
        stats_group = QGroupBox("Clip Statistics")
        stats_layout = QVBoxLayout(stats_group)
        self.lbl_clipstats = QLabel("Click Preview to compute")
        self.lbl_clipstats.setWordWrap(True)
        self.lbl_clipstats.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl_clipstats.setFrameShape(QLabel.Shape.StyledPanel)
        self.lbl_clipstats.setFrameShadow(QLabel.Shadow.Sunken)
        self.lbl_clipstats.setContentsMargins(4, 4, 4, 4)
        self.lbl_clipstats.setMinimumHeight(50)
        stats_layout.addWidget(self.lbl_clipstats)
        layout.addWidget(stats_group)
        
        # Bottom buttons
        bottom_layout = QHBoxLayout()
        self.btn_apply = QPushButton("Apply to Siril")
        self.btn_apply.setEnabled(False)
        self.btn_apply.clicked.connect(self._apply_to_siril)
        bottom_layout.addWidget(self.btn_apply)
        
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self._reset)
        bottom_layout.addWidget(self.btn_reset)
        
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.close)
        bottom_layout.addWidget(self.btn_close)
        layout.addLayout(bottom_layout)
        
        layout.addStretch()
        
        footer = QLabel(f"v{VERSION} - Seti Astro")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(footer)
        
        return left_widget
    
    def create_right_panel(self):
        right_widget = QWidget()
        layout = QVBoxLayout(right_widget)
        
        zoom_layout = QHBoxLayout()
        btn_zoom_in = QPushButton("Zoom In")
        btn_zoom_in.clicked.connect(lambda: self.view.zoom_in())
        zoom_layout.addWidget(btn_zoom_in)
        
        btn_zoom_out = QPushButton("Zoom Out")
        btn_zoom_out.clicked.connect(lambda: self.view.zoom_out())
        zoom_layout.addWidget(btn_zoom_out)
        
        btn_fit = QPushButton("Fit to View")
        btn_fit.clicked.connect(lambda: self.view.fit_item(self.pix))
        zoom_layout.addWidget(btn_fit)
        
        zoom_layout.addStretch()
        layout.addLayout(zoom_layout)
        
        self.scene = QGraphicsScene(self)
        self.view = ZoomableGraphicsView(self.scene, self)
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pix = QGraphicsPixmapItem()
        self.scene.addItem(self.pix)
        layout.addWidget(self.view)
        
        return right_widget
    
    def update_target_median_display(self):
        value = self.target_median_slider.value() / 100.0
        self.target_median_display.setText(f"{self.processor.floor_value(value):.2f}")
    
    def update_blackpoint_sigma_display(self):
        value = self.blackpoint_sigma_slider.value() / 100.0
        self.blackpoint_sigma_display.setText(f"{self.processor.floor_value(value):.2f}")
    
    def update_curves_boost_display(self):
        value = self.curves_boost_slider.value() / 100.0
        self.curves_boost_display.setText(f"{self.processor.floor_value(value):.2f}")
    
    def update_hdr_amount_display(self):
        value = self.hdr_amount_slider.value() / 100.0
        self.hdr_amount_display.setText(f"{self.processor.floor_value(value):.2f}")
    
    def update_hdr_knee_display(self):
        value = self.hdr_knee_slider.value() / 100.0
        self.hdr_knee_display.setText(f"{self.processor.floor_value(value):.2f}")
    
    def update_luma_blend_display(self):
        value = self.luma_blend_slider.value() / 100.0
        self.luma_blend_display.setText(f"{self.processor.floor_value(value):.2f}")
    
    def toggle_blackpoint_slider(self):
        enabled = not self.no_black_clip_checkbox.isChecked()
        self.blackpoint_sigma_slider.setEnabled(enabled)
    
    def toggle_curves_boost(self):
        enabled = self.apply_curve_checkbox.isChecked()
        self.curves_boost_slider.setEnabled(enabled)
    
    def toggle_hdr_controls(self):
        enabled = self.hdr_checkbox.isChecked()
        self.hdr_amount_slider.setEnabled(enabled)
        self.hdr_knee_slider.setEnabled(enabled)
    
    def toggle_luma_controls(self):
        enabled = self.luma_checkbox.isChecked()
        self.luma_profile_combo.setEnabled(enabled)
        self.luma_blend_slider.setEnabled(enabled)
        self.linked_stretch_checkbox.setEnabled(not enabled)
    
    def lock_hdr_knee(self):
        self.hdr_knee_locked = True
    
    def auto_suggest_hdr_knee(self):
        if getattr(self, 'hdr_knee_locked', False):
            return
        target = self.target_median_slider.value() / 100.0
        knee = np.clip(target + 0.10, 0.10, 0.95)
        self.hdr_knee_slider.blockSignals(True)
        self.hdr_knee_slider.setValue(int(knee * 100))
        self.hdr_knee_slider.blockSignals(False)
        self.update_hdr_knee_display()
    
    def _set_pix(self, rgb):
        display_rgb = np.flipud(rgb)
        arr = (np.clip(display_rgb, 0, 1) * 255).astype(np.uint8)
        arr = np.ascontiguousarray(arr)
        h, w, _ = arr.shape
        q = QImage(arr.data, w, h, 3*w, QImage.Format.Format_RGB888)
        self.pix.setPixmap(QPixmap.fromImage(q))
        self.view.setSceneRect(self.pix.boundingRect())
    
    def _toggle(self):
        if self.btn_toggle.isChecked():
            self.btn_toggle.setText("Show Preview")
            self._set_pix(self.original_hwc)
        else:
            self.btn_toggle.setText("Show Original")
            self._set_pix(self.preview_hwc)
    
    def _reset(self):
        self.target_median_slider.setValue(25)
        self.blackpoint_sigma_slider.setValue(500)
        self.no_black_clip_checkbox.setChecked(False)
        self.linked_stretch_checkbox.setChecked(False)
        self.normalize_checkbox.setChecked(False)
        self.hdr_checkbox.setChecked(False)
        self.hdr_amount_slider.setValue(15)
        self.hdr_knee_slider.setValue(75)
        self.luma_checkbox.setChecked(False)
        self.luma_profile_combo.setCurrentText("rec709")
        self.luma_blend_slider.setValue(60)
        self.apply_curve_checkbox.setChecked(False)
        self.curves_boost_slider.setValue(0)
        self.preview_hwc = self.original_hwc.copy()
        self._set_pix(self.preview_hwc)
        self.lbl_step.setText("Idle")
        self.lbl_clipstats.setText("Click Preview to compute")
        self.progress_bar.setValue(0)
        self.btn_apply.setEnabled(False)
        self.btn_toggle.setChecked(False)
        self.btn_toggle.setText("Show Original")
        self.hdr_knee_locked = False
    
    def _start_preview(self):
        self.btn_preview.setEnabled(False)
        self.btn_apply.setEnabled(False)
        
        params = {
            'target_median': self.target_median_slider.value() / 100.0,
            'blackpoint_sigma': self.blackpoint_sigma_slider.value() / 100.0,
            'no_black_clip': self.no_black_clip_checkbox.isChecked(),
            'linked': self.linked_stretch_checkbox.isChecked(),
            'normalize': self.normalize_checkbox.isChecked(),
            'apply_curves': self.apply_curve_checkbox.isChecked(),
            'curves_boost': self.curves_boost_slider.value() / 100.0 if self.apply_curve_checkbox.isChecked() else 0.0,
            'hdr_compress': self.hdr_checkbox.isChecked(),
            'hdr_amount': self.hdr_amount_slider.value() / 100.0 if self.hdr_checkbox.isChecked() else 0.0,
            'hdr_knee': self.hdr_knee_slider.value() / 100.0,
            'luma_only': self.luma_checkbox.isChecked(),
            'luma_mode': self.luma_profile_combo.currentText(),
            'luma_blend': self.luma_blend_slider.value() / 100.0,
        }
        
        self.thread = QThread(self)
        self.worker = StretchWorker(self.original_hwc, params)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress_update.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()
    
    def _on_progress(self, step, pct):
        self.lbl_step.setText(step)
        self.progress_bar.setValue(pct)
    
    def _on_finished(self, out, stats):
        self.btn_preview.setEnabled(True)
        if out is None:
            QMessageBox.critical(self, "Error", f"Stretch failed:\n{stats}")
            return
        
        self.preview_hwc = out
        self._set_pix(self.preview_hwc)
        self.btn_apply.setEnabled(True)
        self.btn_toggle.setChecked(False)
        self.btn_toggle.setText("Show Original")
        self.lbl_step.setText("Preview ready")
        self.lbl_clipstats.setText(stats)
        self.progress_bar.setValue(100)
    
    def _apply_to_siril(self):
        out = self.preview_hwc
        
        if self._was_mono:
            mono = np.mean(out, axis=2, dtype=np.float32)
            out = mono
        
        out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)
        
        if out.ndim == 3:
            out = np.moveaxis(out, -1, 0)
        
        try:
            with self.siril.image_lock():
                parts = [f"m={self.target_median_slider.value()/100:.2f}"]
                if self.luma_checkbox.isChecked():
                    parts.append(f"luma={self.luma_profile_combo.currentText()}")
                    parts.append(f"blend={self.luma_blend_slider.value()/100:.2f}")
                else:
                    parts.append(f"l={self.linked_stretch_checkbox.isChecked()}")
                if self.hdr_checkbox.isChecked():
                    parts.append(f"hdr={self.hdr_amount_slider.value()/100:.2f}")
                undo_name = f"StatStretch: {' '.join(parts)}"
                self.siril.undo_save_state(undo_name)
                self.siril.set_image_pixeldata(out)
            
            self.siril.log("Statistical Stretch applied", s.LogColor.GREEN)
            QMessageBox.information(self, "Success", "Image applied to Siril")
            self.close()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply:\n{e}")

# ============================================================================
# MAIN / CLI
# ============================================================================

def run_cli_mode(siril, args):
    if not siril.is_image_loaded():
        print("No image is loaded")
        return
    
    try:
        siril.cmd("requires", "1.3.6")
    except s.CommandError:
        return
    
    args.median = max(0.01, min(0.99, args.median))
    args.sigma = max(0.5, min(6.0, args.sigma))
    args.hdramount = max(0.0, min(1.0, args.hdramount))
    args.hdrknee = max(0.1, min(0.95, args.hdrknee))
    args.lumablend = max(0.0, min(1.0, args.lumablend))
    
    processor = StatisticalStretchProcessor()
    
    try:
        with siril.image_lock():
            fit = siril.get_image()
            fit.ensure_data_type(np.float32)
            img = fit.data
            
            if img.ndim == 2:
                img_hwc = img
                was_mono = True
            elif img.ndim == 3:
                if img.shape[0] in [1, 3]:
                    img_hwc = np.moveaxis(img, 0, -1)
                    was_mono = (img.shape[0] == 1)
                else:
                    img_hwc = img
                    was_mono = False
            
            if was_mono:
                mono = img_hwc[..., 0] if img_hwc.ndim == 3 else img_hwc
                out = processor.stretch_mono_image(
                    mono, args.median, args.normalize, args.boost > 0, args.boost,
                    args.sigma, args.noblackclip, args.hdr, args.hdramount, args.hdrknee
                )
            else:
                out = processor.stretch_color_image(
                    img_hwc, args.median, args.linked, args.normalize, args.boost > 0, args.boost,
                    args.sigma, args.noblackclip, args.hdr, args.hdramount, args.hdrknee,
                    args.luma, args.lumaprofile, args.lumablend
                )
            
            if was_mono and out.ndim == 3:
                out = out[..., 0]
            if out.ndim == 3:
                out = np.moveaxis(out, -1, 0)
            
            parts = [f"m={args.median:.2f}"]
            if args.luma:
                parts.append(f"luma={args.lumaprofile}")
                parts.append(f"blend={args.lumablend:.2f}")
            siril.undo_save_state(f"StatStretch: {' '.join(parts)}")
            fit.data[:] = np.clip(out, 0, 1)
            siril.set_image_pixeldata(fit.data)
        
        print("Statistical stretch applied successfully.")
    
    except Exception as e:
        print(f"Error: {e}")

def main():
    try:
        siril = s.SirilInterface()
        try:
            siril.connect()
        except s.SirilConnectionError:
            if not siril.is_cli():
                app = QApplication(sys.argv)
                app.setStyle('Fusion')
                QMessageBox.critical(None, "Error", "Failed to connect to Siril")
                sys.exit(1)
            else:
                print("Failed to connect to Siril")
            return
        
        if siril.is_cli() and len(sys.argv) > 1:
            parser = argparse.ArgumentParser(description="Statistical Stretch")
            parser.add_argument("-median", type=float, default=0.25)
            parser.add_argument("-sigma", type=float, default=5.0)
            parser.add_argument("-noblackclip", action="store_true")
            parser.add_argument("-boost", type=float, default=0.0)
            parser.add_argument("-linked", action="store_true")
            parser.add_argument("-normalize", action="store_true")
            parser.add_argument("-hdr", action="store_true")
            parser.add_argument("-hdramount", type=float, default=0.15)
            parser.add_argument("-hdrknee", type=float, default=0.75)
            parser.add_argument("-luma", action="store_true")
            parser.add_argument("-lumaprofile", type=str, default="rec709")
            parser.add_argument("-lumablend", type=float, default=0.6)
            
            args = parser.parse_args()
            run_cli_mode(siril, args)
        else:
            app = QApplication(sys.argv)
            app.setStyle('Fusion')
            app.setApplicationName("Statistical Stretch")
            window = StatisticalStretchDialog(siril)
            window.show()
            sys.exit(app.exec())
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
